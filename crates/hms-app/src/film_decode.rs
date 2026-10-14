//! Decoding a Reach saved film's per-tick records (`flmd` type 0).
//!
//! WHAT THIS DOES AND DOES NOT DO, precisely, because the difference decides what a viewer can
//! show:
//!
//! * It decodes the record ENVELOPE — the MSB-first bitstream the engine's
//!   `MapContext_EncodeToBitstream` (`haloreach.dll+0x3C80B0`) produces: the five optional
//!   sub-blocks, the four subsystem channels, the `VariantContext` block with its 32-bit tick,
//!   and both command lists with every command's header and payload span.
//! * It pulls each **entity message** (command type 2) apart into `entity_type` (5 bits),
//!   `net_id` (16 + 11 bits) and a 64-bit field dirty mask — i.e. WHICH entity changed.
//! * It does **NOT** decode an entity's payload, so it does not yet produce positions. The
//!   payload is written by a per-entity-type handler (`vtbl+0x60` for creation, `vtbl+0x70` for
//!   the masked dirty fields; 19 types, dispatched off `g_184E0E918`), and those field tables are
//!   still being reverse-engineered. `EntityMessage::payload` is the raw span so that work can
//!   drop in without changing anything here.
//! * Two sub-records are not implemented yet and stop a record cleanly rather than guessing:
//!   the 16 × 0x74 per-player update block (`sub_18043FC00`) and the 0xA8 per-player record
//!   (`sub_1803C7698`). Together they stop most records in most films — see `ScanStats`.
//!
//! THE CORRECTNESS GATE is strong and cheap, and this module reports it rather than asserting a
//! layout is right: a record that decoded correctly (a) ends within 7 bits of its byte length,
//! because the writer rounds `ceil(bits/8)`, (b) has a leading `VariantContext` tick exactly one
//! greater than the previous record's, and (c) has command lists whose
//! `sum(payload_len + 0x20)` equals the stored total. All three are checked.

/// MSB-first bit reader — the same order HMS's `.mvar` reader uses, confirmed for the film codec
/// from `Bitstream_ReadBits` (`+0xDD10C`), which refills eight bytes at a time with `bswap`.
pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    len_bits: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, pos: 0, len_bits: buf.len() * 8 }
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn remaining(&self) -> usize {
        self.len_bits.saturating_sub(self.pos)
    }
    /// Read `n` bits (n <= 64), MSB-first.
    pub fn bits(&mut self, n: u32) -> Result<u64, DecodeStop> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 || self.pos + n as usize > self.len_bits {
            return Err(DecodeStop::Overrun);
        }
        let mut v = 0u64;
        let mut need = n;
        while need > 0 {
            let byte = self.buf[self.pos >> 3];
            let avail = 8 - (self.pos & 7) as u32;
            let take = need.min(avail);
            let shift = avail - take;
            let mask = if take == 8 { 0xFFu8 } else { ((1u16 << take) - 1) as u8 };
            v = (v << take) | ((byte >> shift) & mask) as u64;
            self.pos += take as usize;
            need -= take;
        }
        Ok(v)
    }
    pub fn flag(&mut self) -> Result<bool, DecodeStop> {
        Ok(self.bits(1)? != 0)
    }
    /// Skip `n` bits.
    pub fn skip(&mut self, n: usize) -> Result<(), DecodeStop> {
        if self.pos + n > self.len_bits {
            return Err(DecodeStop::Overrun);
        }
        self.pos += n;
        Ok(())
    }
    /// Read `n` whole bytes out of the bitstream (NOT byte-aligned in the buffer).
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>, DecodeStop> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.bits(8)? as u8);
        }
        Ok(out)
    }
}

/// Why a record's decode stopped. Everything except `Ok` means "this reader does not understand
/// this record yet", NOT "the file is bad" — the distinction is reported, never hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeStop {
    /// Ran off the end of the record.
    Overrun,
    /// The unit controller-input block is present. Empty in every film measured so far, so this
    /// has never been seen in real data.
    UnitControlBlock,
    /// The 16 × 0x74 per-player update block (`sub_18043FC00`) — not implemented.
    PlayerUpdateBlock,
    /// The 0xA8 per-player record (`sub_1803C7698`) — not implemented.
    PlayerRecord,
    /// The optional block at MapContext +0xD71C (`sub_1803C800C`) — not implemented.
    OptionalBlockD71C,
    /// A command list header was out of range, so the record is not laid out as expected.
    BadCommandList,
    /// `sum(payload_len + 0x20)` did not equal the stored total.
    CommandListChecksum,
    /// A `compressed_vector3` took one of the two paths that no real film has ever exercised, so
    /// decoding it would be guesswork.
    UnexercisedVectorPath,
    /// This entity type's creation reader has not been reverse-engineered.
    UnknownEntityCreation,
}

impl std::fmt::Display for DecodeStop {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            DecodeStop::Overrun => "ran off the end of the record",
            DecodeStop::UnitControlBlock => "unit controller-input block (not implemented)",
            DecodeStop::PlayerUpdateBlock => "per-player update block (not implemented)",
            DecodeStop::PlayerRecord => "per-player record (not implemented)",
            DecodeStop::OptionalBlockD71C => "optional block at +0xD71C (not implemented)",
            DecodeStop::BadCommandList => "command list header out of range",
            DecodeStop::CommandListChecksum => "command list size checksum failed",
            DecodeStop::UnexercisedVectorPath => "compressed_vector3 took an unexercised path",
            DecodeStop::UnknownEntityCreation => "entity type has no known creation reader",
        };
        f.write_str(s)
    }
}

/// One entry of a `VariantContext` command list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    /// 4 bits. 2 = an entity create/update message; 0xC = a unit action.
    pub kind: u8,
    pub byte9: u8,
    pub byte_a: u8,
    /// 16-bit sequence number.
    pub seq: u16,
    /// Payload bytes, read out of the bitstream.
    pub payload: Vec<u8>,
}

/// An entity create/update message (command kind 2), as far as it is understood.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityMessage {
    /// 5 bits, 0..18. Selects the per-type handler that wrote the payload.
    pub entity_type: u8,
    /// `(bits(16) << 16) | bits(11)` — the network entity id. Sequential per creation:
    /// `(salt << 16) | index`.
    pub net_id: u32,
    /// 64-bit per-field dirty mask. Zero on a pure creation.
    pub dirty_mask: u64,
    /// The type-specific payload.
    pub payload: Vec<u8>,
    /// Command kind 2 is a CREATION (its payload leads with the creation block); kind 3 is an
    /// UPDATE of an entity that already exists, and carries only the dirty block. Kind 3 is the
    /// per-tick traffic -- 9 to 16 messages per tick against kind 2's 0.07 -- so it is where all
    /// the MOTION is.
    pub is_update: bool,
}

impl EntityMessage {
    /// The low 16 bits of the net id: the entity's slot in the engine's 1024-entry simulation
    /// entity table. Stable for an entity's whole life, so it is the key a viewer would track.
    pub fn entity_index(&self) -> u16 {
        (self.net_id & 0xFFFF) as u16
    }
}

/// One decoded record.
#[derive(Clone, Debug, Default)]
pub struct TickRecord {
    /// The `VariantContext` 32-bit tick. Increments by exactly 1 per record.
    pub tick: u32,
    /// Commands from list A then list B, in order; the first `commands_a` are list A's.
    pub commands: Vec<Command>,
    /// How many of `commands` came from list A (a different command namespace from list B).
    pub commands_a: usize,
    /// Entity messages pulled out of the kind-2 (create) and kind-3 (update) commands.
    pub entities: Vec<EntityMessage>,
    /// Entities DESTROYED this tick, as `(entity type, net id)`.
    ///
    /// List B **kind 4** is a 5-byte message: the same 5-bit entity type and 27-bit object
    /// reference a create or update carries, and nothing else. It is the destroy.
    ///
    /// File-proven: across 8 films and **356,957 kind-4 messages, 100.0% name an entity that
    /// was created earlier in the same film and not yet destroyed**, with zero double-destroys
    /// and zero unknown ids. Its per-type histogram also tracks the creations almost exactly
    /// (`13:7255 12:1542 17:1026 11:523` against `13:7790 12:1610 17:1126 11:559` on one film —
    /// the shortfall is the entities still alive when the recording stopped).
    ///
    /// This is what gives an entity a LIFETIME. Without it a parked vehicle, which sends one
    /// creation and then never reports again, has `last_tick == first_tick` and is never alive at
    /// any later moment.
    pub destroyed: Vec<(u8, u32)>,
    /// How many `s_player_action` records this tick carried.
    pub player_actions: usize,
    /// #player-unit  The `s_player_action` blocks that named at least one object, slot-tagged.
    /// This is the engine's own player-to-unit assignment on the wire.
    pub actions: Vec<PlayerAction>,
    /// How many per-player records this tick carried.
    pub players: usize,
    /// The aim of each player whose record was present this tick, slot-tagged.
    pub aims: Vec<PlayerAim>,
    /// Bits left after the command lists. Must be 0..=7 on a correct decode.
    pub leftover_bits: usize,
    /// The up-to-four optional SUBSYSTEM CHANNELS this tick carried, as
    /// `(subsystem id, bit length, bit offset of the payload inside the record body)`.
    ///
    /// Recorded rather than decoded. Exactly one channel is present on 1.000 ticks of every real
    /// film, and the engine has a `game_variant_state_codec` (`+0x3C8870`) on this path — which
    /// is the one place a Megalo score could still be hiding. Carrying the id, the length and the
    /// offset costs no allocation (a fixed array, not a `Vec`) and lets a probe read the payload
    /// back out of the record without re-deriving the whole walk.
    pub channels: [(u32, u32, u32); 4],
    /// How many of `channels` are in use.
    pub channels_used: u8,
}

const CMD_HEADER_BITS: usize = 0x20; // the 4+8+8+16+13 = 49-bit header, counted as 0x20 in the
                                     // list's own size total (the engine's own accounting).

fn command_list(r: &mut BitReader, flags: u32) -> Result<Vec<Command>, DecodeStop> {
    let count = r.bits(12)? as usize;
    let total = r.bits(17)? as usize;
    for _ in 0..flags {
        r.bits(1)?;
    }
    if count > 0x1000 || total > 0x20000 {
        return Err(DecodeStop::BadCommandList);
    }
    let mut out = Vec::with_capacity(count.min(64));
    let mut acc = 0usize;
    for _ in 0..count {
        let kind = r.bits(4)? as u8;
        let byte9 = r.bits(8)? as u8;
        let byte_a = r.bits(8)? as u8;
        let seq = r.bits(16)? as u16;
        let len = r.bits(13)? as usize;
        if kind == 0 || len == 0 {
            break;
        }
        let payload = r.read_bytes(len)?;
        acc += len + CMD_HEADER_BITS;
        out.push(Command { kind, byte9, byte_a, seq, payload });
    }
    if acc != total {
        return Err(DecodeStop::CommandListChecksum);
    }
    Ok(out)
}

fn entity_message(payload: &[u8], is_update: bool) -> Option<EntityMessage> {
    if payload.len() < 12 {
        return None;
    }
    let mut r = BitReader::new(payload);
    let entity_type = r.bits(5).ok()? as u8;
    let hi = r.bits(16).ok()?;
    let lo = r.bits(11).ok()?;
    let dirty_mask = r.bits(64).ok()?;
    let used = (r.pos() + 7) / 8;
    Some(EntityMessage {
        entity_type,
        net_id: ((hi << 16) | lo) as u32,
        dirty_mask,
        payload: payload.get(used..).unwrap_or(&[]).to_vec(),
        is_update,
    })
}

/// The three pieces of state a record's layout depends on that are NOT in the film. Each is
/// solved per film by trial decode (`solve_params`), scored against the byte-boundary and
/// tick-continuity oracles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeParams {
    /// `Network_IsClientHostAuthority()` (`+0x71540`), read from session state and never
    /// transmitted. It widens three fields.
    pub host_authority: bool,
    /// Total bits of a `compressed_vector3` (`sub_1803C6884`), summed over the three axes. The
    /// engine computes the per-axis widths at init from the MAP's world bounds
    /// (`sub_1803C9A68`), so this is map state, not film state.
    pub cv3_total_bits: u32,
    /// How `cv3_total_bits` splits across x, y, z. The envelope decode only needs the total; the
    /// split matters as soon as a POSITION is read, and is solved separately (`solve_axis_split`).
    pub cv3_axis_bits: [u32; 3],
    /// Width of the shared creation block's `TagReference_BitstreamDecode` (`+0x3CD34C`). For
    /// `group == -1` the engine uses a HARD-CODED 14 (`or ebx,-1; lea eax,[rbx+0xf]`), so this is
    /// NOT map state; only the typed groups ('weap', 'eqip', 'jpt!') use
    /// `bits_needed(group_count + 1)`.
    pub tagref_bits: u32,
    /// Base field 2: the 8-bit angle follows even when the axis is the implicit `+Z`, so the
    /// field costs 9 or 28 bits (not 1 or 28).
    ///
    /// This contradicts one reading of `sub_1803D9330`'s disassembly, and THE DATA SETTLED IT:
    /// with the angle always present, three independent oracles improve together on real films —
    /// the forward vector of static Forge objects goes from 52-90% to 95-99% agreement with the
    /// authored one, its implicit-axis subset from ~5% to ~97%, and the payload-length oracle
    /// from 70% to 92.9%. A wrong field width cannot improve all three at once.
    pub angle_always: bool,
    /// Bits of prefix before the dirty block in a kind-3 UPDATE message (kind 2 reaches the
    /// dirty block through the creation block instead).
    ///
    /// **1**, established by a continuity oracle that is far sharper than payload length: a
    /// moving entity's position must change smoothly at 60 Hz, so a wrong prefix scatters it
    /// across the map. At 1 bit, 1915 of 1915 and 18285 of 18370 consecutive position steps on
    /// two real films are under 0.5 world units per tick; every other prefix falls to 1-41%.
    pub update_prefix_bits: u32,
}

impl Default for DecodeParams {
    fn default() -> Self {
        DecodeParams { host_authority: false, cv3_total_bits: 12, tagref_bits: 14, cv3_axis_bits: [4, 4, 4], angle_always: true, update_prefix_bits: 0 }
    }
}

/// `compressed_vector3` (`sub_1803C6884`). Only the first path has ever been exercised in real
/// film data, so the other two stop the decode instead of guessing.
fn compressed_vector3(r: &mut BitReader, p: &DecodeParams) -> Result<(), DecodeStop> {
    if r.flag()? {
        r.skip(p.cv3_total_bits as usize)
    } else {
        Err(DecodeStop::UnexercisedVectorPath)
    }
}

/// `sub_1803D9330` — an axis+angle rotation: the AXIS is optional (a set flag means the implicit
/// +Z axis), the ANGLE is always present. Costs 1+angle or 1+axis+angle bits, which is how the
/// engine's own "9 or 28" cost for base field 2 decomposes.
fn axis_and_angle(r: &mut BitReader, axis_bits: u32, angle_bits: u32) -> Result<(), DecodeStop> {
    if !r.flag()? {
        r.bits(axis_bits)?;
    }
    r.bits(angle_bits)?;
    Ok(())
}

/// `sub_1800DC848` — a direction + logarithmic magnitude where BOTH are under the flag (a set
/// flag means the whole vector is absent). Costs 1 or 1+dir+mag, i.e. the engine's "1 or 30".
/// NOTE this differs from `axis_and_angle` above: getting the two confused desynchronises the
/// rest of the record, which is exactly what the byte-boundary and tick oracles catch.
fn dir_and_log_magnitude(r: &mut BitReader, dir_bits: u32, mag_bits: u32) -> Result<(), DecodeStop> {
    if !r.flag()? {
        r.bits(dir_bits)?;
        r.bits(mag_bits)?;
    }
    Ok(())
}

/// A 27-bit object reference: `(bits(16) << 16) | bits(11)` (`sub_1803D2CF0`).
fn object_ref(r: &mut BitReader) -> Result<u32, DecodeStop> {
    let hi = r.bits(16)?;
    let lo = r.bits(11)?;
    Ok(((hi << 16) | lo) as u32)
}

/// #player-unit  THE THREE OBJECT REFERENCES a player's own per-tick block carries.
///
/// `s_player_action` (`sub_18043FC00`, 0x74 bytes) is read once per PLAYER SLOT behind a 16-bit
/// mask, and the mask bit position IS the slot. Three of its fields are object references in the
/// engine's 27-bit `(bits(16) << 16) | bits(11)` form (`sub_1803D2D84`) — the same encoding the
/// event channel uses, which HMS already resolves to a network entity id. The engine initialises
/// all three to `-1` and only reads each behind a flag, which is the "no object" case.
///
/// This reader walked all three and **threw them away**. They are the player-to-unit assignment
/// the engine itself replicates: see [`FilmIndex::unit_of_slot_at`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerAction {
    /// 0..15, the player slot — the mask bit position, the same index as the header's
    /// `s_player_configuration` array.
    pub slot: u8,
    /// `+0x14`: the reference read on EVERY block, outside the extended-data gate.
    pub ref14: Option<u32>,
    /// `+0x1C`: inside the extended block, the one the engine's own validator pairs with the
    /// 2-bit field at `+0x20`.
    pub ref1c: Option<u32>,
    /// `+0x2C`: inside the extended block.
    pub ref2c: Option<u32>,
}

/// `s_player_action`, 0x74 bytes (`sub_18043FC00`). Bit widths from the decoder's own disassembly.
fn player_action(r: &mut BitReader, p: &DecodeParams) -> Result<PlayerAction, DecodeStop> {
    let mut out = PlayerAction::default();
    let a = p.host_authority as u32;
    let flags = r.bits(2)?; // +0x00
    r.bits(2 * a + 2)?; // +0x01
    r.bits(13)?; // +0x04 yaw
    r.bits(12)?; // +0x08 pitch
    if r.flag()? {
        out.ref14 = Some(object_ref(r)?); // +0x14
    }
    // MSB-first: `flags & 2` is the FIRST of the two bits read. Clear => the short form ends here.
    if flags & 2 == 0 {
        return Ok(out);
    }
    if r.flag()? {
        out.ref1c = Some(object_ref(r)?); // +0x1C
    }
    r.bits(2)?; // +0x20
    r.bits(5)?; // +0x24
    if r.flag()? {
        out.ref2c = Some(object_ref(r)?); // +0x2C
    }
    r.bits(3)?; // +0x30
    if !r.flag()? {
        r.bits(2)?; // +0x32
    }
    if !r.flag()? {
        r.bits(2)?; // +0x33
    }
    compressed_vector3(r, p)?; // +0x34
    if p.host_authority {
        axis_and_angle(r, 20, 14)?; // +0x4C axis + angle (sub_1800708A4)
    } else {
        axis_and_angle(r, 19, 8)?; // +0x4C axis + angle (sub_1803D9330)
    }
    dir_and_log_magnitude(r, 19, 10)?; // +0x58
    dir_and_log_magnitude(r, 19, 8)?; // +0x64
    r.bits(1)?; // +0x70
    Ok(out)
}

/// `sub_18010E794` — the optional trailing block of the per-player record.
fn player_record_tail(r: &mut BitReader) -> Result<(), DecodeStop> {
    if !r.flag()? {
        return Ok(());
    }
    if r.flag()? {
        match r.bits(2)? {
            1 => {
                r.bits(32)?;
                if r.flag()? {
                    r.bits(5)?;
                }
            }
            2 => {
                r.bits(32)?;
            }
            _ => {}
        }
    }
    r.bits(4)?;
    r.bits(4)?;
    let b = r.bits(3)?;
    if b & 2 != 0 {
        dir_and_log_magnitude(r, 20, 14)?; // sub_1800DC848
    }
    Ok(())
}

/// The per-player record, 0xA8 bytes (`sub_1803C7698`).
/// The part of a per-tick `s_player_configuration` update this reader can USE: the player's
/// aim. The two angle fields sit at a fixed place near the head of the record, so they decode
/// without having to understand the rest of it.
///
/// `yaw` is 13 bits over a full turn and `pitch` 12 bits over (presumably) a half turn. These
/// are the engine's own *aim* angles, not the biped's body facing, which is why first person
/// driven from them looks like the player was looking rather than like the body was pointing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerAim {
    /// 0..15, the player slot -- the same index as the film header's player array.
    pub slot: u8,
    pub yaw_q: u16,
    pub pitch_q: u16,
    /// +0x08, 48 bits behind a flag (`sub_1800DD274(bs, 0x30)`).
    ///
    /// **NOT an object handle.** Measured over 364,882 present samples on the 16-player test clip
    /// and 44k-49k on each of six more: every way of cutting the 48 bits that could hold a handle
    /// or an index (hi32, lo32, hi16, mid16, lo16, swapped) matches a live entity id **0.0%** of
    /// the time. Whatever it is, it is not the player's unit.
    pub f08: Option<u64>,
    /// +0x40, 32 bits behind a flag.
    ///
    /// **NOT the player's unit handle**, which is what this field was long guessed to be. It is
    /// barely ever sent (8 samples on the whole 7-minute test clip, 1 distinct value; absent
    /// entirely on four of eight films) and matches a biped id **0.0%** of the time. The real
    /// player-to-unit assignment is in `s_player_action` — see [`PlayerAction`].
    pub f40: Option<u32>,
}

impl PlayerAim {
    /// Radians, counter-clockwise, in the same convention as a decoded entity forward vector.
    pub fn yaw(&self) -> f32 {
        self.yaw_q as f32 * (std::f32::consts::TAU / 8192.0)
    }
    /// Radians, positive looking up. 12 bits over a full turn with **2048 = level**.
    ///
    /// The bias is measured, not assumed: over 388,624 real aim samples the median is exactly
    /// 2048 and 93.7% of them fall in the 2048..2303 bucket, with the whole range spanning
    /// 1075..2440 -- about 85 degrees down to 35 degrees up, which is what a player's look limits
    /// allow. (Reading the field as a plain unsigned angle instead put LEVEL at half a turn and
    /// pointed a first-person camera at the sky.)
    pub fn pitch(&self) -> f32 {
        (self.pitch_q as f32 - PITCH_LEVEL_Q as f32) * (std::f32::consts::TAU / 4096.0)
    }
    /// The aim direction as a unit vector in HMS's world axes (+Z up).
    pub fn direction(&self) -> [f32; 3] {
        let (y, p) = (self.yaw(), self.pitch());
        [y.cos() * p.cos(), y.sin() * p.cos(), p.sin()]
    }
}

fn player_record(r: &mut BitReader, p: &DecodeParams) -> Result<PlayerAim, DecodeStop> {
    let a = p.host_authority as u32;
    let k = r.bits(1)?; // +0x00
    if k == 1 {
        r.bits(2 * a + 2)?; // +0x04
    }
    let f08 = if r.flag()? { Some(r.bits(48)?) } else { None }; // +0x08
    let f40 = if r.flag()? { Some(r.bits(32)? as u32) } else { None }; // +0x40
    let yaw_q = r.bits(13)? as u16; // +0x10 yaw
    let pitch_q = r.bits(12)? as u16; // +0x14 pitch
    if r.flag()? {
        r.bits(12)?; // +0x18
    }
    r.bits(5)?; // +0x1C
    r.bits(5)?; // +0x20
    if r.flag()? {
        r.bits(5)?; // +0x24
    }
    let f28 = r.flag()?; // +0x28
    r.flag()?; // +0x29
    if f28 {
        r.bits(5)?; // +0x2C
        r.bits(5)?; // +0x30
    }
    r.bits(2)?; // +0x34
    if r.flag()? {
        r.bits(5)?; // +0x38
        if r.flag()? {
            r.bits(5)?; // +0x3C
        }
    }
    r.bits(3)?; // +0x44
    if !r.flag()? {
        r.bits(2)?; // +0x46
    }
    if !r.flag()? {
        r.bits(2)?; // +0x47
    }
    if !r.flag()? {
        r.bits(1)?; // +0x48
    }
    r.bits(2)?; // +0x4A
    if r.flag()? {
        r.bits(3)?; // +0x4C
        r.bits(p.tagref_bits)?; // +0x50 TagReference('unit') -- map-dependent width
    }
    if r.flag()? {
        // three FIXED iterations, with no early break on kk == 0
        for _ in 0..3 {
            let kk = r.bits(4)?;
            if kk == 0 {
                continue;
            }
            r.bits(32)?;
            if kk == 2 || kk == 0xA {
                r.bits(4)?;
            } else if kk == 5 || kk == 6 || kk == 9 {
                if !r.flag()? {
                    r.bits(6)?;
                }
            }
        }
    }
    player_record_tail(r)?;
    if r.flag()? {
        r.bits(4)?;
    }
    r.bits(2)?; // +0xA4
    r.bits(32)?; // +0xA0
    Ok(PlayerAim { slot: 0, yaw_q, pitch_q, f08, f40 })
}

/// Decode one type-0 record body.
pub fn decode_record(body: &[u8], p: &DecodeParams) -> Result<TickRecord, DecodeStop> {
    let mut r = BitReader::new(body);
    // MapContext +0xAAD0: the unit controller-input block. Never set in any real film.
    if r.flag()? {
        return Err(DecodeStop::UnitControlBlock);
    }
    // +0xCEE0: 17 x 48 raw bits.
    if r.flag()? {
        let mask = r.bits(17)?;
        for k in 0..17 {
            if (mask >> k) & 1 != 0 {
                r.bits(48)?;
            }
        }
    }
    // +0xCF4C: up to 16 player actions behind a 16-bit mask.
    //
    // #player-unit  The mask bit position IS the player slot (the same index as the header's
    // `s_player_configuration` array, exactly as for the aim block below), so each block's three
    // object references are slot-tagged the moment they are read.
    let mut player_actions = 0usize;
    let mut actions: Vec<PlayerAction> = Vec::new();
    if r.flag()? {
        let mask = r.bits(16)?;
        for k in 0..16 {
            if (mask >> k) & 1 != 0 {
                let mut act = player_action(&mut r, p)?;
                act.slot = k as u8;
                if act.ref14.is_some() || act.ref1c.is_some() || act.ref2c.is_some() {
                    actions.push(act);
                }
                player_actions += 1;
            }
        }
    }
    // +0xD690: 17-bit mask, 4 bits each.
    let mask = r.bits(17)?;
    for k in 0..17 {
        if (mask >> k) & 1 != 0 {
            r.bits(4)?;
        }
    }
    // +0xD71C: an optional block. Never set in any real film.
    if r.flag()? {
        return Err(DecodeStop::OptionalBlockD71C);
    }
    // four optional subsystem channels: present bit, 32-bit id, 32-bit bit length, nested bits.
    let mut channels = [(0u32, 0u32, 0u32); 4];
    let mut channels_used = 0u8;
    for k in 0..4 {
        if r.flag()? {
            let id = r.bits(32)? as u32;
            let bits = r.bits(32)? as usize;
            if bits > body.len() * 8 {
                return Err(DecodeStop::Overrun);
            }
            channels[k] = (id, bits as u32, r.pos() as u32);
            channels_used += 1;
            r.skip(bits)?;
        }
    }
    // the VariantContext block
    let tick = r.bits(32)? as u32;
    let _f04 = r.bits(5)?;
    let mut players = 0usize;
    let mut aims: Vec<PlayerAim> = Vec::new();
    if r.flag()? {
        let mask = r.bits(16)?;
        for k in 0..16 {
            if (mask >> k) & 1 != 0 {
                // The mask bit position IS the player slot, the same index the film header's
                // 16 `s_player_configuration` slots use -- which is what makes an aim track
                // attributable to a named player.
                let mut a = player_record(&mut r, p)?;
                a.slot = k as u8;
                aims.push(a);
                players += 1;
            }
        }
    }
    let _a90 = r.bits(2)?;
    let _a94 = r.bits(2)?;
    // TWO command lists with DIFFERENT namespaces. Only list B's kind-2 entries are replicated
    // objects; parsing list A's kind 2 the same way yields garbage (types 0..2 with multi-KB
    // payloads), so the entity extraction filters on the list.
    let list_a = command_list(&mut r, 0)?;
    let list_b = command_list(&mut r, 2)?;
    let entities: Vec<EntityMessage> = list_b
        .iter()
        .filter(|c| c.kind == 2 || c.kind == 3)
        .filter_map(|c| entity_message(&c.payload, c.kind == 3))
        .collect();
    // Kind 4: the destroy. Header only -- 5-bit type, 27-bit object reference, 5 bytes total.
    let destroyed: Vec<(u8, u32)> = list_b
        .iter()
        .filter(|c| c.kind == 4 && c.payload.len() >= 4)
        .filter_map(|c| {
            let mut r = BitReader::new(&c.payload);
            let ty = r.bits(5).ok()? as u8;
            let hi = r.bits(16).ok()?;
            let lo = r.bits(11).ok()?;
            Some((ty, ((hi << 16) | lo) as u32))
        })
        .collect();
    let mut commands = list_a;
    let commands_a = commands.len();
    commands.extend(list_b);
    Ok(TickRecord {
        tick,
        commands,
        commands_a,
        entities,
        destroyed,
        player_actions,
        actions,
        aims,
        players,
        leftover_bits: r.remaining(),
        channels,
        channels_used,
    })
}

/// # NOT YET VALIDATED
///
/// Everything from here to `solve_position_frame` decodes an entity message's PAYLOAD, and it
/// does **not** pass its own correctness oracle yet. A message whose dirty mask uses only known
/// fields must consume its payload to within 7 bits; measured over 46 real mask-{0,2} messages,
/// the best parameter sweep reaches **6 of 46**. The record envelope around it is at 100%, so
/// the error is inside the field walk — most likely the per-type field counts (the base class
/// has 20 fields but type 13 has 21, types 12/15 have 25, type 17 has 26, type 18 has 29) or the
/// exact shape of base field 2.
///
/// It is kept because it is nearly right and the remaining gap is small and well-characterised,
/// and because `entity_payload_oracle` measures it so a fix shows up immediately. **Nothing in
/// HMS consumes the positions it produces**, and nothing should until that number is ~1.0.
///
/// Everything a decoded entity message yields beyond its header. Fields are `None` when the
/// message did not carry them (the dirty mask selects what is present).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EntityState {
    /// Base field 0, as the three RAW quantised axis values. Turning these into world units
    /// needs the per-axis affine map (`PositionFrame`), which is map state.
    pub position_raw: Option<[u32; 3]>,
    /// Base field 2: the 19-bit quantised axis (`None` = the implicit +Z axis) and the 8-bit
    /// angle, both raw.
    pub orientation_raw: Option<(Option<u32>, u32)>,
    /// How many of the 20 base fields were walked before the decode stopped (20 = the whole walk).
    pub fields_walked: u32,
    /// Set when the walk stopped early because a present field has no known codec.
    pub stopped_at_unknown_field: Option<u32>,
    /// A BIPED's body facing: a 13-bit yaw over a full turn, found two bits past the end of the
    /// base-class block. See `BIPED_FACING_OFFSET_BITS`.
    pub facing_yaw_q: Option<u16>,
    /// Which gated field latched the base walk (ending the base block), if one did.
    pub latched_at: Option<u32>,
    /// Bit offset into the payload at which the base-class walk stopped. The derived class's own
    /// dirty block starts here, so this is the anchor any search for a biped field needs.
    pub bits_after_base: usize,
    /// The dirty mask REBUILT from the presence bits the walk actually read. On a correct decode
    /// this equals the message's own 64-bit mask -- an independent check that costs nothing,
    /// because the stream carries the information twice.
    pub rebuilt_mask: u64,
    /// #film-attach  Base field 12: the object this one is attached to, when the message carries
    /// that field. `Some` with a parent means "riding or stuck to"; the field being PRESENT with
    /// no parent is how an attachment is cleared, which this reports as `None` -- the caller sees
    /// the difference through the dirty mask.
    pub attach: Option<BaseAttachment>,
}

/// The shared creation reader `sub_1803D5A8C`, used by types 11, 12, 13, 14, 15, 17 and 18.
fn creation_shared(r: &mut BitReader, p: &DecodeParams) -> Result<(), DecodeStop> {
    r.bits(p.tagref_bits)?; // +0x0C  TagReference(group = -1): the object's DEFINITION tag
    r.bits(13)?; // +0x00
    if r.flag()? {
        r.bits(10)?; // +0x02
    }
    r.bits(4)?; // +0x04
    if !r.flag()? {
        r.bits(4)?; // +0x05 -- note the INVERTED sense: a set flag means 0xFF and reads nothing
    }
    r.bits(7)?; // +0x08
    r.bits(32)?; // +0x10
    Ok(())
}

/// The per-type creation payload. Only the types whose readers are known are handled; the rest
/// stop the walk.
fn creation(r: &mut BitReader, entity_type: u8, p: &DecodeParams) -> Result<(), DecodeStop> {
    match entity_type {
        // engine singletons with a null creation writer
        8 | 10 => Ok(()),
        // tiny bespoke readers, file-validated exactly
        3 | 5 => {
            r.bits(5)?;
            Ok(())
        }
        4 => {
            r.bits(4)?;
            Ok(())
        }
        12 | 13 | 14 | 15 | 18 => creation_shared(r, p),
        17 => {
            creation_shared(r, p)?;
            r.bits(12)?;
            r.bits(7)?; // qreal(0,1,7,unsigned,exact)
            Ok(())
        }
        _ => Err(DecodeStop::UnknownEntityCreation),
    }
}

/// The base-class dirty-field walk (`sub_1803D7E58`), fields 0..19. Each field is preceded by one
/// presence bit whether or not it is set, so the walk is self-describing: it reads the fields it
/// knows and stops cleanly at the first PRESENT field whose codec is unresolved.
///
/// Position is field 0 and orientation is field 2, so both are reached before the first gap
/// (field 11, whose leaf helper is unresolved).
fn base_dirty_walk(r: &mut BitReader, p: &DecodeParams, st: &mut EntityState) -> Result<(), DecodeStop> {
    // The iterator carries a GATE LIST (`sub_18043442C`); the base walk's is `{2, 8}` (read off
    // `0x180A49258`). Before the presence bit of a gated field the stream carries one extra gate
    // bit, and a 0 there LATCHES the iterator: every later field is skipped with no bits read at
    // all. That is how the writer truncates the tail of the block.
    let gate_list = [2u32, 8];
    let mut next_gate = 0usize;
    for field in 0u32..20 {
        if next_gate < gate_list.len() && field == gate_list[next_gate] {
            next_gate += 1;
            if !r.flag()? {
                st.latched_at = Some(field);
                return Ok(()); // latched: the rest of the walk is absent
            }
        }
        let present = r.flag()?;
        st.fields_walked = field + 1;
        if !present {
            continue;
        }
        st.rebuilt_mask |= 1u64 << field;
        match field {
            0 => {
                // compressed_vector3(bits = 0x10): one flag, then the three per-axis values.
                if !r.flag()? {
                    return Err(DecodeStop::UnexercisedVectorPath);
                }
                let w = p.cv3_axis_bits;
                st.position_raw = Some([r.bits(w[0])? as u32, r.bits(w[1])? as u32, r.bits(w[2])? as u32]);
            }
            1 => dir_and_log_magnitude(r, 19, 10)?,
            2 => {
                // sub_1803D9330. The flag SET is the CHEAP case: axis = (0,0,1) and the function
                // jumps straight to its epilogue, so NO angle follows. Clear reads a 19-bit axis
                // AND an 8-bit angle. Cost is 1 or 28 bits, not 9 or 28.
                let (axis_bits, angle_bits) = if p.host_authority { (20, 14) } else { (19, 8) };
                let axis = if r.flag()? { None } else { Some(r.bits(axis_bits)? as u32) };
                let angle = if p.angle_always { r.bits(angle_bits)? as u32 } else if axis.is_some() { r.bits(angle_bits)? as u32 } else { 0 };
                st.orientation_raw = Some((axis, angle));
            }
            3 => dir_and_log_magnitude(r, 19, 8)?,
            4 => {
                r.bits(8)?; // qreal(-1,1,8,signed,exact)
                r.bits(1)?;
            }
            5 => {} // presence bit only
            6 | 7 => {
                r.bits(8)?; // qreal(0,4,8,unsigned,exact)
                r.bits(4)?; // 4 flags
            }
            8 => {
                r.bits(5)?;
                r.bits(48)?; // 16 x 3 bits
            }
            9 => {
                r.bits(5)?;
                for _ in 0..16 {
                    if r.flag()? {
                        r.bits(7)?; // qreal(0,1,7,unsigned,exact)
                    }
                }
            }
            10 => {
                let n = r.bits(5)? as u32;
                if n != 0 {
                    r.bits(n)?;
                    r.bits(n)?;
                }
            }
            // 15 + 4 + `sub_1804EC888`, the same damage-state tail the biped's weapon-slot
            // field 0x19 uses. Costed only so field 12 can be reached.
            11 => {
                r.bits(15)?;
                r.bits(4)?;
                damage_state(r)?;
            }
            // #film-attach  THE OBJECT-TO-OBJECT ATTACHMENT RECORD.
            12 => st.attach = base_attachment(r, p)?,
            _ => {
                // Fields 13..19 have unresolved leaf codecs. Stop here rather than guess: the
                // caller still has position, orientation and the attachment.
                st.stopped_at_unknown_field = Some(field);
                return Ok(());
            }
        }
    }
    Ok(())
}

/// `sub_1804EC888`: a 2-bit mode then 0, 1, 3 or 4 eleven-bit quantised reals.
fn damage_state(r: &mut BitReader) -> Result<(), DecodeStop> {
    let n = match r.bits(2)? {
        0 => 0,
        1 => 1,
        2 => 3,
        _ => 4,
    };
    for _ in 0..n {
        r.bits(11)?;
    }
    Ok(())
}

/// Per-axis `(min, max)` for `sub_1803C6CE4`'s quantisation modes, read off `0x180A48E30`
/// (24 bytes per mode: three `(min, max)` float pairs). The attachment record selects mode 0 or
/// mode 2.
pub const ATTACH_OFFSET_RANGE: [[f32; 2]; 3] = [[-3.0, 3.0], [-0.7, 0.7], [-100.0, 100.0]];
/// Bits per axis in an attachment's local offset: `sub_1803C6CE4(.., .., 0x10)`.
pub const ATTACH_OFFSET_BITS: u32 = 16;

/// `sub_1803C6CE4`'s dequantisation: `raw * step + min + step/2`, `step = (max - min) / 2^bits`.
pub fn dequant_range(raw: u32, lo: f32, hi: f32, bits: u32) -> f32 {
    let step = (hi - lo) / (1u64 << bits) as f32;
    raw as f32 * step + lo + step * 0.5
}

/// #film-attach  What base field 12 states: this object is RIDING or STUCK TO another one.
///
/// `sub_1803D7E58` gate-group 0xC, non-host-authority:
///
/// ```text
/// if ReadFlag():                      // clear = this object has no parent
///     sub_1803D2D84      27 bits      // the PARENT, bits(16) << 16 | bits(11) -- the same
///                                     //   namespace as a message's own net id
///     ReadBits(8)                     // the attach point on the parent
///     ReadFlag(); ReadFlag()          // the second selects the offset's quantisation mode
///     sub_1803C6CE4(.., mode, 0x10)   // 3 x 16 bits, NO leading flag: the LOCAL OFFSET
///     sub_1803D9330(..)               // the LOCAL ORIENTATION -- base field 2's own decoder
/// ReadFlag() ? -1 : ReadBits(6)       // a SEAT INDEX, note the INVERTED sense
/// ReadFlag()
/// ```
///
/// Proven by the vehicle agent against the sharpest oracle this format offers: a 27-bit
/// reference read at the wrong offset is noise in a 134-million-value space, and on a 12:16 clip
/// **53 of 53** records that state a parent name a net id that film created AND that is alive at
/// that tick. The decoded pairs are the user's own report by name -- a fusion coil on a Warthog
/// turret, a plasma grenade stuck to a Spartan, and a Spartan in a Warthog's seat 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BaseAttachment {
    /// The parent object's net id, in the same namespace as `EntityTrack::net_id`.
    pub parent: u32,
    /// The attach point on the parent (the 8-bit field).
    pub node: u8,
    /// The offset from the parent, in the PARENT's frame, world units.
    pub offset: [f32; 3],
    /// The child's orientation in the parent's frame as `(axis_raw, angle_raw)`, exactly as base
    /// field 2 carries it. `None` axis = the implicit +Z, i.e. aligned with the parent.
    pub orientation_raw: (Option<u32>, u32),
    /// The 6-bit seat index, `None` when the record states -1 (which is every non-seated child).
    pub seat: Option<u8>,
    /// Which of [`ATTACH_OFFSET_RANGE`] the offset was quantised against.
    pub mode: u8,
}

/// Walk base field 12. `Ok(None)` is a record that states NO parent, which is how an attachment
/// is cleared -- the bits after it are still consumed either way.
fn base_attachment(r: &mut BitReader, p: &DecodeParams) -> Result<Option<BaseAttachment>, DecodeStop> {
    let mut out: Option<BaseAttachment> = None;
    if r.flag()? {
        let hi = r.bits(16)?;
        let lo = r.bits(11)?;
        let node = r.bits(8)? as u8;
        r.flag()?;
        // The engine passes `second_flag ? 0 : 2` as the range-table index.
        let mode: u8 = if r.flag()? { 0 } else { 2 };
        let [mlo, mhi] = ATTACH_OFFSET_RANGE[mode as usize];
        let mut offset = [0.0f32; 3];
        for o in offset.iter_mut() {
            *o = dequant_range(r.bits(ATTACH_OFFSET_BITS)? as u32, mlo, mhi, ATTACH_OFFSET_BITS);
        }
        // base field 2's decoder, `sub_1803D9330`: a SET flag means axis (0,0,1) and the cheap
        // path; clear reads a 19-bit axis. The angle follows on both paths in a film.
        let (axis_bits, angle_bits) = if p.host_authority { (20, 14) } else { (19, 8) };
        let axis = if r.flag()? { None } else { Some(r.bits(axis_bits)? as u32) };
        let angle = if p.angle_always {
            r.bits(angle_bits)? as u32
        } else if axis.is_some() {
            r.bits(angle_bits)? as u32
        } else {
            0
        };
        out = Some(BaseAttachment {
            parent: ((hi << 16) | lo) as u32,
            node,
            offset,
            orientation_raw: (axis, angle),
            seat: None,
            mode,
        });
    }
    // The seat and the trailing flag are outside the `if`: they are stated whether or not there
    // is a parent, which is what makes "every non-seated child reads -1" a credible measurement
    // rather than an artefact of never reading the field.
    let seat = if r.flag()? { None } else { Some(r.bits(6)? as u8) };
    r.flag()?;
    if let Some(a) = out.as_mut() {
        a.seat = seat;
    }
    Ok(out)
}

/// Decode an entity message's payload as far as it is understood. Returns `None` when the type's
/// creation reader is unknown; returns a state with `stopped_at_unknown_field` set when the walk
/// stopped part-way (which still yields position and orientation).
pub fn decode_entity_payload(e: &EntityMessage, p: &DecodeParams) -> Option<EntityState> {
    let mut r = BitReader::new(&e.payload);
    let mut st = EntityState::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        creation(&mut r, e.entity_type, p).ok()?;
    }
    if e.dirty_mask != 0 {
        // Only the types that use the shared base class have a known dirty walk.
        if !matches!(e.entity_type, 11 | 12 | 13 | 15 | 16 | 17 | 18) {
            return Some(st);
        }
        base_dirty_walk(&mut r, p, &mut st).ok()?;
        st.bits_after_base = r.pos();
        // A biped does not use base field 2 for its facing: the base walk's gate bit before that
        // field is almost always clear for a unit, which latches the walk and ends the base block
        // right after the position. The facing is the first field of the DERIVED (unit) block
        // that follows -- a 13-bit yaw over a full turn, the same width the per-player aim uses,
        // two bits past the base block (the derived block's own gate and presence bits).
        //
        // Not read off the engine: found by searching (offset, width) against the direction of
        // travel, which a running biped mostly faces. Widths 13, 14 and 16 all scored the same
        // 0.418 rad median error while 12 was clearly worse, which is the signature of a 13-bit
        // field -- and it is confirmed against the much sharper aim-yaw oracle in
        // `biped_facing_tracks_the_player_aim`.
        // Only when the walk latched at field 2, which is the case the offset was measured on;
        // a base block that ran further puts the derived block somewhere else.
        if e.entity_type == 11 && st.position_raw.is_some() && st.latched_at == Some(2) {
            let mut t = BitReader::new(&e.payload);
            if t.skip(st.bits_after_base + BIPED_FACING_OFFSET_BITS).is_ok() {
                if let Ok(v) = t.bits(BIPED_FACING_BITS) {
                    st.facing_yaw_q = Some(v as u16);
                }
            }
        }
        return Some(st);
    }
    st.bits_after_base = r.pos();
    Some(st)
}

// ---------------------------------------------------------------------------------------------
// the SCOREBOARD -- entity type 5, `game-engine-statborg`  (#film-stats)
// ---------------------------------------------------------------------------------------------

/// How many statborg slots there are: **24**, and the split is proven in the engine
/// (`sub_180079A7C` indexes teams at `base + 0x80 + team*8`, i.e. past the 16 players).
pub const STATBORG_SLOTS: u8 = 24;
/// Statborg indices below this are PLAYER slots -- the same 0..15 numbering as the film header's
/// `s_player_configuration` array, which is why a statborg score needs no owner-group bridge.
pub const STATBORG_PLAYERS: u8 = 16;
/// The engine clamps every statborg statistic to this range (`sub_18018F8EC` for players,
/// `sub_18018F974` for teams), so a decoded value outside it is a decode error, not a big score.
pub const STATBORG_CLAMP: i32 = 30_000;

/// One `game-engine-statborg` message, decoded. **This is the scoreboard.**
///
/// Entity type 5 is `game-engine-statborg` (the name is in the binary: vtable `0x180A4E760`
/// slot `+0x08`), there are 24 of them, and its state is `s_simulation_game_statborg_state_data`
/// -- 8 bytes, four fields. Unlike the object types it has **no gate list**: its dirty reader
/// (`sub_180459B88`) is a hand-rolled four-iteration walk with one presence bit per field, so
/// the layout is fixed and tiny.
///
/// | field | bits | meaning |
/// |-------|------|---------|
/// | 0 | flag + 16, sign-extended | `round_score` -- the score Megalo sets, statistic index 0 |
/// | 1 | flag + 16, sign-extended | score carried in from COMPLETED rounds |
/// | 2 | flag + 16, sign-extended | `rounds_won` |
/// | 3 | flag + 1 | `final` -- when set, the displayed total is field 1 alone |
///
/// The displayed total is `total_score + round_score` while a round is live and `total_score`
/// alone once `final` is set, which is the engine's own rule.
///
/// ## What the fields do, measured (`film_stats::probes::probe_statborg_behaviour`)
///
/// Over 25 films and 943 oracle-passing messages (295 updates, 648 creations):
///
/// * **presence** — field 0 is sent on 312 messages (33.1%), field 1 on **16** (1.7%), field 2 on
///   20 (2.1%), field 3 on 16 (1.7%, always TRUE when sent). Counting the PRESENCE bits rather
///   than the folded values is what distinguishes "always zero" from "never sent", and it is why
///   field 1 reading 0 on most clips is not a misread offset: it carries 1..10 on the two films
///   that actually contain a completed round.
/// * **behaviour** — the per-slot `round_score` series has 230 consecutive pairs: 212 up, 0
///   unchanged, 18 down. **Seventeen of the eighteen drops are a reset to zero** (a round
///   boundary); exactly one is a genuine decrement of a single point (14 → 13), and it lands on a
///   round boundary too. So this is a round-scoped cumulative counter.
/// * **the sign bit is never exercised.** `round_score` spans 0..31 and no negative appears in
///   any of 91 scoreboard rows. The 16-bit statistics are read sign-extended because the engine's
///   store is `i16` with a ±30000 clamp, i.e. from the CODE — the file has never demanded it.
///
/// A warning against a tempting non-oracle: on some clips the final score equals the number of
/// updates, which looks like confirmation ("one point per award") and is not. A score is a
/// cumulative total replicated as an ABSOLUTE value, so that equality only holds for a clip that
/// starts at zero; it is true of just 31 of 91 rows. Confidence in this decode rests on the three
/// structural oracles in [`StatborgMsg::trustworthy`], nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatborgMsg {
    /// Creation only: the statborg index. `< STATBORG_PLAYERS` is a player SLOT, the rest are
    /// teams (`index - 16`).
    pub index: Option<u8>,
    pub round_score: Option<i16>,
    pub total_score: Option<i16>,
    pub rounds_won: Option<i16>,
    pub is_final: Option<bool>,
    /// The dirty mask rebuilt from the presence bits. Equal to the message's own 64-bit mask on a
    /// correct decode -- the stream states the mask TWICE, which is a free oracle.
    pub rebuilt_mask: u64,
    /// Bits left after the walk. 0..=7 on a correct decode (the writer pads to a byte).
    pub leftover_bits: usize,
}

impl StatborgMsg {
    /// Does this decode pass BOTH free oracles -- the rebuilt mask equal to the message's own,
    /// and the payload consumed to within the writer's byte padding?
    pub fn trustworthy(&self, dirty_mask: u64) -> bool {
        self.rebuilt_mask == dirty_mask
            && self.leftover_bits <= 7
            && self.round_score.map_or(true, |v| (v as i32).abs() <= STATBORG_CLAMP)
            && self.total_score.map_or(true, |v| (v as i32).abs() <= STATBORG_CLAMP)
            && self.rounds_won.map_or(true, |v| (v as i32).abs() <= STATBORG_CLAMP)
    }
}

/// Decode a type-5 (`game-engine-statborg`) message. `None` when the payload ran out.
pub fn decode_statborg(e: &EntityMessage, p: &DecodeParams) -> Option<StatborgMsg> {
    if e.entity_type != 5 {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut st = StatborgMsg::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        // `statborg-create`: a 5-bit `statborg-index`, valid `< k_statborg_indices_count` = 24.
        let i = r.bits(5).ok()? as u8;
        if i >= STATBORG_SLOTS {
            return None;
        }
        st.index = Some(i);
    }
    // A creation with a zero mask carries no dirty block at all.
    if e.dirty_mask != 0 || e.is_update {
        let i16_from = |v: u64| v as u16 as i16;
        for field in 0u32..4 {
            if !r.flag().ok()? {
                continue;
            }
            st.rebuilt_mask |= 1 << field;
            match field {
                0 => st.round_score = Some(i16_from(r.bits(16).ok()?)),
                1 => st.total_score = Some(i16_from(r.bits(16).ok()?)),
                2 => st.rounds_won = Some(i16_from(r.bits(16).ok()?)),
                _ => st.is_final = Some(r.flag().ok()?),
            }
        }
    }
    st.leftover_bits = r.remaining();
    Some(st)
}

/// Bit offset, inside a biped's CREATION payload, of the 4-bit index of the player who owns it.
///
/// Found structurally, with no oracle: a real owner field has to partition a film's bipeds into
/// groups whose LIVES NEVER OVERLAP, because one player cannot drive two bipeds at once. Sweeping
/// every offset and width over 109 bipeds on a 16-player film, this one field gives **exactly 16
/// groups** with 92.3% of same-group life pairs disjoint; nothing else comes close at that group
/// count. (The residual overlap is corpses and ragdolls, whose entity lingers past the respawn.)
pub const CREATION_OWNER_BIT: usize = 272;
/// Width the fixed-offset reader used: 4 bits. **The owner field is five bits wide** -- see
/// [`BipedCreation::owner`]. Four bits at this offset read its low four, which is how slot 15
/// came to look like "nobody".
pub const CREATION_OWNER_BITS: u32 = 4;

/// The player slot that owns a biped, read at the FIXED offset above.
///
/// **Superseded by [`creation_biped`], and kept only so the probes that found the field can still
/// be run.** The offset is fixed and the field is not: `sub_1803B2590` reads it *after* a shared
/// header with two optional sub-fields and after an optional APPEARANCE block whose own cost
/// swings between 108 and 192 bits, so one constant can only ever be right for one shape of
/// message. See [`creation_biped`] for what that cost HMS.
pub fn creation_owner_slot(payload: &[u8]) -> Option<u8> {
    let mut r = BitReader::new(payload);
    r.skip(CREATION_OWNER_BIT).ok()?;
    r.bits(CREATION_OWNER_BITS).ok().map(|v| v as u8)
}

/// `sub_1803A296C`: a biped creation's APPEARANCE block -- the armour pieces and the colours.
///
/// Present only when the flag before it is set, which is what distinguishes a PLAYER's biped from
/// one the gametype or the AI spawned. Its own cost is variable: twelve of its sub-fields are
/// `flag ? -1 : ReadBits(7)`, so the block is 108 bits when every one of them is absent and 192
/// when every one is present -- an 84-bit swing, and the reason a fixed offset for anything that
/// follows it cannot work.
fn creation_appearance(r: &mut BitReader) -> Result<BipedAppearance, DecodeStop> {
    let mut a = BipedAppearance::default();
    a.flags = (r.flag()? as u8) | ((r.flag()? as u8) << 1); // +0x00
    for c in a.colors.iter_mut() {
        *c = r.bits(6)? as i8 - 1; // +0x01, +0x02, +0x03
    }
    a.model_choice = r.bits(1)? as u8; // +0x04
    // +0x10..0x17 then +0x18..0x1B: twelve `flag ? -1 : ReadBits(7)` sub-fields. Note the
    // INVERTED sense -- a CLEAR flag is the one that carries a value.
    for sl in a.slots.iter_mut() {
        *sl = if r.flag()? { -1 } else { r.bits(7)? as i8 };
    }
    for id in a.ids.iter_mut() {
        *id = r.bits(8)? as u8; // +0x1C, +0x1E, +0x20, +0x22 -- through `sub_1803A31E8`
    }
    r.bits(7)?; // +0x08
    r.bits(7)?; // +0x09
    r.bits(3)?; // +0x0A
    r.bits(6)?; // +0x0B
    r.bits(6)?; // +0x0C
    r.bits(6)?; // +0x0D
    r.bits(8)?; // +0x26
    Ok(a)
}

/// `s_player_appearance` as a biped's own creation message carries it -- the SAME struct the film
/// header carries per player slot ([`crate::film::PlayerAppearance`]), at the same internal
/// offsets, and here stated per BODY.
///
/// Corroborated twice in `haloreach.dll`: the reader `sub_1803A296C` writes +0x00 flags,
/// +0x01..+0x03 three colour indices, +0x04 a 1-bit value and +0x10.. the slot options; and the
/// buffer builder `PlayerCustomization_BuildArmorAppearanceBuffer` fills the same five fields in
/// the same order, **clamping +0x04 to 0..1** -- which is `player_model_choice`
/// (`k_player_model_choice_count` = 2). The header's record puts flags at +0x48, the colours at
/// +0x49..+0x4B, `player_model_choice` at +0x4C and the slot options at +0x58, i.e. +0x00, +0x01,
/// +0x04 and +0x10 inside the struct. **So the header is a free answer key for this decode.**
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BipedAppearance {
    /// +0x00, two bits: bit 0 `female-voice`, bit 1 `female-model`. The header calls it `f48`.
    pub flags: u8,
    /// +0x01..+0x03: primary, secondary and tertiary armour colour indices, `-1` = none. The
    /// header calls them `color_primary`, `color_secondary` and `f4b`.
    pub colors: [i8; 3],
    /// +0x04 `player_model_choice`: **0 = Spartan, 1 = Elite**.
    pub model_choice: u8,
    /// +0x10..+0x1B: twelve armour-slot option indices, `-1` where the slot is the default. The
    /// header's `PlayerAppearance::slots` is the first ten of these, in the same order.
    pub slots: [i8; 12],
    /// +0x1C/+0x1E/+0x20/+0x22, as read (the engine maps each through `sub_1803A31E8`'s
    /// fourteen-range table before storing it).
    pub ids: [u8; 4],
}

impl BipedAppearance {
    /// #appearance  Is this the DEFAULT appearance -- i.e. nothing customised?
    ///
    /// **The default is not an identity, it is the absence of one**, and it must never name
    /// anybody. A guest, a player who never touched the armoury, and a body whose appearance
    /// simply carries nothing all decode to the same all-zero value, so "it matches exactly one
    /// listed player" is meaningless here: the uniqueness rule only asks how many LISTED players
    /// share it, and on a film where just one listed player is uncustomised that count is one.
    ///
    /// This is what put `Danielsan413(1)` -- a guest with no customisation -- on a body the owner
    /// stamp attributed to `Ender Hunter868`, and it is why the armour/stamp cross-check fired.
    pub fn is_default(&self) -> bool {
        self.flags == 0
            && self.model_choice == 0
            && self.colors.iter().all(|c| *c <= 0)
            && self.slots.iter().all(|s| *s <= 0)
    }
}

/// What a biped's CREATION payload states, walked rather than guessed at a fixed offset.
///
/// `sub_1803B2590`, the engine's own type-11 creation reader, in order: the shared creation header
/// (`sub_1803D5A8C`), one flag, and if that flag is set two bits plus the whole appearance block
/// (`sub_1803A296C`), then **`ReadBits(4) - 1` into `+0x3E` -- the owner -- and `ReadBits(5) - 1`
/// into `+0x3F`**, then a `TagReference('eqip')`, and if that reference is not null a
/// `compressed_vector3` and one more optional 4-bit field.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BipedCreation {
    /// `+0x3F`: `ReadBits(5) - 1`, `-1` = none. **It is the TEAM index, which is the player slot
    /// only because Reach's FFA puts every player on their own team.**
    ///
    /// SOLVED 2026-10-13 (#owner-team), and it explains every measurement that made this field
    /// look inconsistent. On the 16-player FFA clip `*21897D8A*` it takes 16 values and names
    /// every player, because FFA assigns each player a distinct team. On the team clip
    /// `*53C14E9D*` it takes exactly TWO, and the appearance block right before it resolves the
    /// bodies into **18 distinct players -- 8 carrying value 0 and 10 carrying value 1**, whose
    /// body counts sum to **207 and 183, the measured split, exactly**. Two teams of 8 and 10.
    /// The corpus then reads straight off: 10-17 groups on the FFA clips, **3** groups on
    /// `asq_mglo-16_*05C1DB83*` (a three-team game, which is why its share was 67% and why there
    /// was never a "clean gap" in the distribution), and 2 on every `asq_mglo-0` clip.
    ///
    /// So `slot = wire - 1` is **only valid on FFA**. HMS's per-film recorder-anchor gate is what
    /// stopped it mis-naming team games, and it was doing exactly the right thing for a reason
    /// nobody had understood. Naming a body on a team clip needs the APPEARANCE route
    /// ([`FilmIndex::bind_by_appearance`]), not this field.
    ///
    /// FIVE bits, not four, and that one fact dissolves a question this project got wrong four
    /// times. The old reader took four bits at a fixed offset that lands on this field's low
    /// four bits, so slot 15 -- wire 16 -- came back as wire 0, indistinguishable from "nobody".
    /// Every "wire 0 is a mixing bucket" / "wrap `(owner-1) & 15`" / "the field can only name
    /// fifteen players" conclusion came from that aliasing. Five bits hold `15 + 1` with room to
    /// spare, so **wire 0 means nobody and nothing else**, and `slot = wire - 1` needs no special
    /// case at all.
    pub owner: i8,
    /// `+0x3E`: the FOUR-bit field just before the owner, `ReadBits(4) - 1`. Measured `-1` on
    /// every one of the 112 bipeds of the 16-player test clip, so it is not a per-player index;
    /// what it is has not been established.
    pub f3e: i8,
    /// Whether the appearance block was present. A biped with one is a player's.
    pub appearance: bool,
    /// The appearance block's CONTENTS when it was present: species, armour colours and the
    /// twelve slot options. The film header carries the same struct per player slot, which makes
    /// it an answer key -- see [`BipedAppearance`].
    pub look: Option<BipedAppearance>,
    /// The `'eqip'` reference as read, before resolution: 0 is the null reference.
    pub eqip_raw: u32,
    /// Bits left in the payload when the walk finished. The writer rounds up to a byte, so a
    /// correct walk ends within 7 -- which is a free and very sharp oracle on the walk's SHAPE,
    /// and the one that solves `eqip_bits` per film. It says nothing about the owner, which is
    /// read earlier; see `bits_before_owner`.
    pub leftover_bits: usize,
    /// Whether the walk closed the payload (`leftover_bits < 8`). False also when the tail could
    /// not be walked at all, which is why it is separate from `leftover_bits`.
    pub tail_closed: bool,
    /// The bit offset the owner field was found at, which is what the old fixed
    /// [`CREATION_OWNER_BIT`] tried to be. It MOVES -- `creation_walk_probe` prints the
    /// distribution per film.
    pub bits_before_owner: usize,
}

/// Walk a biped's creation payload. `eqip_bits` is the width of a `TagReference('eqip')` in this
/// film's stream, which is `bits_needed(eqip_count + 1)` over the MAP -- see
/// [`solve_eqip_bits`], which measures it from the payloads themselves.
pub fn creation_biped(payload: &[u8], p: &DecodeParams, eqip_bits: u32) -> Option<BipedCreation> {
    let mut r = BitReader::new(payload);
    creation_shared(&mut r, p).ok()?;
    let mut c = BipedCreation::default();
    if r.flag().ok()? {
        c.appearance = true;
        r.bits(2).ok()?;
        c.look = Some(creation_appearance(&mut r).ok()?);
    }
    // THE OWNER IS DETERMINED HERE, and by nothing after here. Everything before it is
    // self-describing -- the shared header's two optional sub-fields and the appearance block's
    // twelve are each introduced by their own flag -- so reaching this point IS the walk's
    // correctness condition for the owner. `tail_closed` below checks the rest of the payload,
    // which is an oracle on the walk's SHAPE and worth reporting, but it must not be allowed to
    // veto a field that was read before it. Gating the owner on it cost one clip every name it
    // had: the `'eqip'` tail is the one part of this message whose widths come from outside the
    // stream, so a film HMS gets wrong there would lose sixteen gamertags it had read correctly.
    // `state[0x3E] = ReadBits(4) - 1` then `state[0x3F] = ReadBits(5) - 1`. **The OWNER is the
    // five-bit one.** See `BipedCreation::owner`.
    c.f3e = r.bits(4).ok()? as i8 - 1;
    c.bits_before_owner = r.pos();
    c.owner = r.bits(5).ok()? as i8 - 1;
    let Ok(eqip) = r.bits(eqip_bits) else { return Some(c) };
    c.eqip_raw = eqip as u32;
    if c.eqip_raw != 0 {
        // the same `compressed_vector3` shape base field 0 uses: one flag, then three axes
        if !r.flag().unwrap_or(false) {
            return Some(c);
        }
        for width in p.cv3_axis_bits {
            if r.bits(width).is_err() {
                return Some(c);
            }
        }
        match r.flag() {
            Ok(false) => {
                if r.bits(4).is_err() {
                    return Some(c);
                }
            }
            Ok(true) => {}
            Err(_) => return Some(c),
        }
    }
    c.tail_closed = r.remaining() < 8;
    c.leftover_bits = r.remaining();
    Some(c)
}

/// The `'eqip'` reference width for a film, solved from its own biped creations.
///
/// The width is `bits_needed(eqip_count + 1)` over the MAP, which the decode worker does not
/// have. It does not need it: the walk in [`creation_biped`] runs to the END of the payload, and
/// the writer pads to a byte, so the right width is the one that leaves 0..=7 bits on (nearly)
/// every creation. Returns the width and the fraction of creations it closed.
pub fn solve_eqip_bits(creations: &[Vec<u8>], p: &DecodeParams) -> (u32, f64) {
    let mut best = (EQIP_BITS_FALLBACK, -1.0f64);
    for bits in 1..=12u32 {
        let mut ok = 0usize;
        for pl in creations {
            if creation_biped(pl, p, bits).map_or(false, |c| c.tail_closed) {
                ok += 1;
            }
        }
        let score = ok as f64 / creations.len().max(1) as f64;
        if score > best.1 {
            best = (bits, score);
        }
    }
    best
}

/// The width used when a film carries too few biped creations to solve one. Reach's `'eqip'`
/// group is small, and this is the width the solve returns on every film measured here.
pub const EQIP_BITS_FALLBACK: u32 = 5;

/// #identity  How many bodies of ONE player can exist at the same moment.
///
/// **Measured, not chosen**: over 39 real films the peak for a group that behaves like a player
/// is three -- a live body plus two that are still lingering as a corpse and a ragdoll -- and
/// every value above it in the corpus belongs to a group that is plainly several people (16
/// bodies with 13 alive at once, 68 with 7, 11 with 4). One player drives one biped.
///
/// **Every identity route must apply this**, because none of the other guards implies it: a
/// route can match a unique roster slot, on an occupied slot, inside that slot's record range,
/// and still be hanging a hundred bodies on one gamertag.
pub const MAX_PLAYER_BODIES_AT_ONCE: i32 = 3;

/// Peak number of `(first, last)` spans alive simultaneously.
pub fn peak_bodies_alive(spans: &[(u32, u32)]) -> i32 {
    let mut edges: Vec<(u32, i32)> = Vec::with_capacity(spans.len() * 2);
    for &(a, b) in spans {
        edges.push((a, 1));
        edges.push((b.saturating_add(1), -1));
    }
    edges.sort_unstable();
    let (mut cur, mut peak) = (0i32, 0i32);
    for (_, d) in edges {
        cur += d;
        peak = peak.max(cur);
    }
    peak
}

/// The quantised aim pitch that means LEVEL. Measured: the exact median over 388,624 real samples.
pub const PITCH_LEVEL_Q: u16 = 2048;

/// Bits between the end of a biped's base-class block and its 13-bit body yaw.
pub const BIPED_FACING_OFFSET_BITS: usize = 2;
/// Width of a biped's body-yaw field, over a full turn.
pub const BIPED_FACING_BITS: u32 = 13;

/// A 13-bit quantised yaw, as radians.
pub fn yaw_from_q13(q: u16) -> f32 {
    q as f32 * (std::f32::consts::TAU / 8192.0)
}

/// Mask bits whose field codec is fully known (base fields 0..10). A message whose dirty mask
/// has no bits outside this set can be decoded to the END of its payload, which makes its byte
/// length an exact oracle.
pub const KNOWN_FIELD_MASK: u64 = 0x7FF;

/// Decode an entity payload that uses only known fields, and return how many bits were left over.
/// A correct decode leaves 0..=7 (the writer pads to a byte). `None` when the message uses a
/// field this reader does not know, or the decode failed.
pub fn entity_payload_leftover(e: &EntityMessage, p: &DecodeParams) -> Option<usize> {
    if e.dirty_mask & !KNOWN_FIELD_MASK != 0 {
        return None;
    }
    if !matches!(e.entity_type, 11 | 12 | 13 | 15 | 16 | 17 | 18) {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut st = EntityState::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        creation(&mut r, e.entity_type, p).ok()?;
    }
    if e.dirty_mask != 0 {
        base_dirty_walk(&mut r, p, &mut st).ok()?;
        if st.stopped_at_unknown_field.is_some() {
            return None; // the walk did not reach the end, so the length proves nothing
        }
        // The stream carries the mask twice: once as the 64-bit header field and once as the
        // presence bits. A walk that is right reproduces it exactly.
        if st.rebuilt_mask != e.dirty_mask {
            return None;
        }
    }
    Some(r.remaining())
}

/// Solve the TOTAL `compressed_vector3` width for a film from the entity payloads themselves:
/// the only unknown left in a fully-known-mask message is how many bits base field 0 consumes,
/// so the byte length of those messages pins it. Returns the width and the fraction of samples
/// it satisfied.
pub fn solve_cv3_total(bytes: &[u8], data: &crate::film::FilmData, base: DecodeParams) -> Option<(u32, f64)> {
    // Collect the fully-known-mask messages once.
    let mut msgs: Vec<EntityMessage> = Vec::new();
    for rec in data.records.iter().take(400) {
        let Ok(t) = decode_record(rec.body(bytes), &base) else { continue };
        for e in t.entities {
            if e.dirty_mask & !KNOWN_FIELD_MASK == 0
                && e.dirty_mask & 1 != 0
                && matches!(e.entity_type, 11 | 12 | 13 | 15 | 16 | 17 | 18)
            {
                msgs.push(e);
            }
        }
        if msgs.len() > 2000 {
            break;
        }
    }
    if msgs.len() < 8 {
        return None;
    }
    let mut best = (0u32, 0.0f64);
    for total in 3..=78u32 {
        let p = DecodeParams {
            cv3_axis_bits: [total / 3, total / 3, total - 2 * (total / 3)],
            cv3_total_bits: total,
            ..base
        };
        let ok = msgs.iter().filter(|e| entity_payload_leftover(e, &p).map_or(false, |l| l <= 7)).count();
        let score = ok as f64 / msgs.len() as f64;
        if score > best.1 {
            best = (total, score);
        }
    }
    (best.1 > 0.0).then_some(best)
}

/// EXPERIMENTAL (see the NOT YET VALIDATED note above).
/// The affine map from a `compressed_vector3`'s raw per-axis value to world units:
/// `world[k] = scale[k] * (raw[k] as f32 + 0.5) + offset[k]`.
///
/// The engine derives it from the loaded MAP's world box, which is not in the film. HMS instead
/// FITS it from the film's own data: the type-13 entities are the Forge objects, and the film
/// already carries their true positions in its embedded map variant, so the two sets pin the
/// mapping. `PositionFrame::fit` does that and reports how well it fits.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PositionFrame {
    pub scale: [f32; 3],
    pub offset: [f32; 3],
    /// Fraction of fitted samples that landed within a quarter of a world unit of a real object.
    pub quality: f32,
}

impl PositionFrame {
    pub fn world(&self, raw: [u32; 3]) -> [f32; 3] {
        [
            self.scale[0] * (raw[0] as f32 + 0.5) + self.offset[0],
            self.scale[1] * (raw[1] as f32 + 0.5) + self.offset[1],
            self.scale[2] * (raw[2] as f32 + 0.5) + self.offset[2],
        ]
    }
}

impl PositionFrame {
    /// Fit the affine map from raw quantised values to world units WITHOUT needing a
    /// correspondence between the two sets: the raw values and the true positions are matched by
    /// QUANTILE, which is robust to the two sets having different sizes (a film creates fewer
    /// entities than the variant has objects, because some objects never replicate).
    ///
    /// `quality` is then measured honestly: the fraction of decoded positions whose nearest true
    /// object position is within a quarter of a world unit.
    pub fn fit(raws: &[[u32; 3]], truth: &[[f32; 3]]) -> Option<PositionFrame> {
        if raws.len() < 16 || truth.len() < 16 {
            return None;
        }
        let mut scale = [0.0f32; 3];
        let mut offset = [0.0f32; 3];
        for k in 0..3 {
            let mut r: Vec<f32> = raws.iter().map(|v| v[k] as f32 + 0.5).collect();
            let mut t: Vec<f32> = truth.iter().map(|v| v[k]).collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |v: &[f32], f: f32| v[((v.len() - 1) as f32 * f).round() as usize];
            // Two widely separated quantiles define the line; 10/90 avoids the tails where the
            // two populations genuinely differ.
            let (r0, r1) = (q(&r, 0.10), q(&r, 0.90));
            let (t0, t1) = (q(&t, 0.10), q(&t, 0.90));
            if (r1 - r0).abs() < 1.0 {
                return None;
            }
            scale[k] = (t1 - t0) / (r1 - r0);
            offset[k] = t0 - scale[k] * r0;
        }
        let mut frame = PositionFrame { scale, offset, quality: 0.0 };
        let mut hit = 0usize;
        for raw in raws {
            let w = frame.world(*raw);
            let near = truth.iter().any(|t| {
                (0..3).all(|k| (t[k] - w[k]).abs() <= 0.25)
            });
            if near {
                hit += 1;
            }
        }
        frame.quality = hit as f32 / raws.len() as f32;
        Some(frame)
    }
}

/// Solve the per-axis split of `cv3_total_bits` for one film, and with it the world-unit mapping,
/// by decoding the type-13 entity positions and fitting them against the film's own embedded map
/// variant. Returns the best `(params, frame)` found and its quality, or `None` when the film has
/// no Forge variant to fit against.
pub fn solve_position_frame(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
    base: DecodeParams,
) -> Option<(DecodeParams, PositionFrame, f64)> {
    if variant.is_empty() {
        return None;
    }
    let truth: Vec<[f32; 3]> = variant.present_objects().map(|o| o.position).collect();
    // Collect the messages whose mask uses only known fields, once.
    let mut msgs: Vec<EntityMessage> = Vec::new();
    for rec in data.records.iter().take(600) {
        let Ok(t) = decode_record(rec.body(bytes), &base) else { continue };
        for e in t.entities {
            if e.dirty_mask & 1 != 0 && e.dirty_mask & !KNOWN_FIELD_MASK == 0 {
                msgs.push(e);
            }
        }
        if msgs.len() > 6000 {
            break;
        }
    }
    if msgs.len() < 16 {
        return None;
    }
    // Stage 1: the per-axis widths, scored by the payload oracle (exact consumption AND the
    // rebuilt mask matching the header mask).
    let mut best: Option<(DecodeParams, f64)> = None;
    for w in 1..=26u32 {
        for extra in 0..3u32 {
            // allow a one-bit difference on one axis, which is what unequal spans produce
            let axes = match extra {
                0 => [w, w, w],
                1 => [w + 1, w, w],
                _ => [w, w, w + 1],
            };
            let p = DecodeParams { cv3_axis_bits: axes, cv3_total_bits: axes.iter().sum(), ..base };
            let ok = msgs.iter().filter(|e| entity_payload_leftover(e, &p).map_or(false, |l| l <= 7)).count();
            let score = ok as f64 / msgs.len() as f64;
            if best.as_ref().map_or(true, |(_, b)| score > *b) {
                best = Some((p, score));
            }
        }
    }
    let (p, oracle) = best?;
    if oracle < 0.2 {
        return None;
    }
    // Stage 2: fit the affine map using ONLY the messages that pass the oracle, so a
    // mis-decoded payload cannot poison the fit.
    let mut raws: Vec<[u32; 3]> = Vec::new();
    for e in &msgs {
        if entity_payload_leftover(e, &p).map_or(true, |l| l > 7) {
            continue;
        }
        if let Some(st) = decode_entity_payload(e, &p) {
            if let Some(r) = st.position_raw {
                raws.push(r);
            }
        }
    }
    let frame = PositionFrame::fit(&raws, &truth)?;
    Some((p, frame, oracle))
}

/// Turn a decoded `orientation_raw` into world (forward, up), reusing `mvar`'s Rodrigues
/// convention with the film's own widths: a 19-bit axis (or the implicit `+Z`) and an 8-bit
/// angle over `[-pi, pi]`.
pub fn orientation_world(raw: (Option<u32>, u32)) -> ([f32; 3], [f32; 3]) {
    match raw.0 {
        None => crate::mvar::decode_orientation_bits(true, 0, 19, raw.1, 8),
        Some(axis) => crate::mvar::decode_orientation_bits(false, axis, 19, raw.1, 8),
    }
}

/// `ceil(log2(v))` the way the engine computes it (`sub_180015B78`: the highest set bit, plus one
/// when `v` is not a power of two).
pub fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        return 0;
    }
    let b = 31 - v.leading_zeros();
    b + if (1u32 << b) == v { 0 } else { 1 }
}

/// The per-axis `compressed_vector3` widths the engine derives from a world box, for
/// `bits = 0x10` (where its tolerance is `1/120`):
/// `w[k] = min(ceil_log2(min(ceil(span[k] * 60), 0x800000)), 26)`.
///
/// The widths are NOT equal across axes — a wider axis gets more bits — which is why a search
/// over near-equal splits never finds them.
pub fn axis_bits_from_bounds(bounds: [f32; 6]) -> [u32; 3] {
    let mut out = [0u32; 3];
    for k in 0..3 {
        let span = bounds[2 * k + 1] - bounds[2 * k];
        let n = (span * 60.0).ceil().max(1.0).min(0x800000 as f32) as u32;
        out[k] = ceil_log2(n).min(26);
    }
    out
}

/// The world mapping that goes with those widths: `world[k] = lo[k] + (raw[k] + 0.5) * span / 2^w`.
pub fn frame_from_bounds(bounds: [f32; 6]) -> (PositionFrame, [u32; 3]) {
    let w = axis_bits_from_bounds(bounds);
    let mut scale = [0.0f32; 3];
    let mut offset = [0.0f32; 3];
    for k in 0..3 {
        let span = bounds[2 * k + 1] - bounds[2 * k];
        scale[k] = span / (1u64 << w[k]) as f32;
        offset[k] = bounds[2 * k];
    }
    (PositionFrame { scale, offset, quality: 0.0 }, w)
}

/// Solve `DecodeParams` for one film by trial decode/// Solve `DecodeParams` for one film by trial decode: every candidate is scored by how many of a
/// sample of records decode byte-aligned with a continuous tick, and the best is returned with
/// its score. A film whose best score is not ~1.0 is reported as such rather than forced.
pub fn solve_params(bytes: &[u8], data: &crate::film::FilmData) -> (DecodeParams, f64) {
    // A sample spread across the film: the first records (creation wave) plus an even stride.
    let n = data.records.len();
    if n == 0 {
        return (DecodeParams::default(), 0.0);
    }
    let mut idx: Vec<usize> = (0..n.min(64)).collect();
    let stride = (n / 256).max(1);
    idx.extend((0..n).step_by(stride));
    idx.sort_unstable();
    idx.dedup();
    let mut best = (DecodeParams::default(), -1.0f64);
    for host_authority in [false, true] {
        for cv3_total_bits in 0..=78u32 {
            let p = DecodeParams {
                host_authority,
                cv3_total_bits,
                tagref_bits: 14,
                cv3_axis_bits: [cv3_total_bits / 3, cv3_total_bits / 3, cv3_total_bits - 2 * (cv3_total_bits / 3)],
                angle_always: true,
                update_prefix_bits: 0,
            };
            let mut ok = 0usize;
            for &i in &idx {
                let body = data.records[i].body(bytes);
                if let Ok(t) = decode_record(body, &p) {
                    if t.leftover_bits <= 7 {
                        ok += 1;
                    }
                }
            }
            let score = ok as f64 / idx.len() as f64;
            if score > best.1 {
                best = (p, score);
            }
            if score >= 1.0 {
                return best;
            }
        }
    }
    best
}

/// What a whole-film decode pass measured. Reported, never asserted — the point is to say how
/// much of the stream this reader really understands.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    pub records: usize,
    /// Records that decoded with 0..=7 leftover bits.
    pub decoded: usize,
    /// Of those, how many carried the tick their POSITION implies: `tick == first_tick + index`.
    /// There is exactly one record per tick, so this holds for every correctly decoded record
    /// regardless of how many records in between this reader could not decode. A decoded record
    /// that fails it means the envelope was mis-read and the 32 bits landed somewhere else.
    pub tick_continuous: usize,
    /// The tick of record 0 (films start at an arbitrary value).
    pub first_tick: u32,
    /// Records whose decode ended with leftover bits > 7 (a layout error this reader would
    /// otherwise have silently accepted).
    pub misaligned: usize,
    /// Why the rest stopped.
    pub stops: std::collections::HashMap<DecodeStop, usize>,
    pub commands: usize,
    pub entity_messages: usize,
    /// How many distinct entity indices were seen.
    pub entities_seen: usize,
    /// Count per 5-bit entity type.
    pub entity_types: [usize; 32],
    /// Payload bytes walked.
    pub payload_bytes: usize,
    pub player_actions: usize,
    pub players: usize,
    /// The parameters `solve_params` chose, and the score it chose them on.
    pub params: DecodeParams,
    pub params_score: f64,
    pub elapsed: std::time::Duration,
}

impl ScanStats {
    pub fn decode_fraction(&self) -> f64 {
        if self.records == 0 { 0.0 } else { self.decoded as f64 / self.records as f64 }
    }
    /// Records per second.
    pub fn records_per_second(&self) -> f64 {
        let s = self.elapsed.as_secs_f64();
        if s <= 0.0 { 0.0 } else { self.records as f64 / s }
    }
    /// How many times faster than real time (the film runs at 60 ticks per second).
    pub fn realtime_multiple(&self) -> f64 {
        self.records_per_second() / 60.0
    }
    pub fn report(&self) -> String {
        let mut stops: Vec<(&DecodeStop, &usize)> = self.stops.iter().collect();
        stops.sort_by(|a, b| b.1.cmp(a.1));
        let mut types: Vec<(usize, usize)> =
            self.entity_types.iter().copied().enumerate().filter(|(_, n)| *n > 0).collect();
        types.sort_by(|a, b| b.1.cmp(&a.1));
        let mut s = format!(
            "{} records, {} decoded ({:.1}%), {} tick-continuous, {} misaligned; \
             {} commands, {} entity messages, {} entities; {:.0} records/s = {:.0}x real time ({:.3} s for {:.1} MB)",
            self.records,
            self.decoded,
            self.decode_fraction() * 100.0,
            self.tick_continuous,
            self.misaligned,
            self.commands,
            self.entity_messages,
            self.entities_seen,
            self.records_per_second(),
            self.realtime_multiple(),
            self.elapsed.as_secs_f64(),
            self.payload_bytes as f64 / 1.0e6,
        );
        if !stops.is_empty() {
            s.push_str("\n  stopped: ");
            s.push_str(&stops.iter().map(|(k, n)| format!("{n}x {k}")).collect::<Vec<_>>().join(", "));
        }
        s.push_str(&format!(
            "\n  params: host_authority={} cv3={} bits tagref={} bits (score {:.3}); {} player actions, {} player records",
            self.params.host_authority, self.params.cv3_total_bits, self.params.tagref_bits,
            self.params_score, self.player_actions, self.players
        ));
        if !types.is_empty() {
            s.push_str("\n  entity types: ");
            s.push_str(&types.iter().take(10).map(|(t, n)| format!("{t}:{n}")).collect::<Vec<_>>().join(" "));
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// the playback index
// ---------------------------------------------------------------------------------------------

/// One replicated entity's track through the film: where it was, each time the stream said so.
#[derive(Clone, Debug, Default)]
pub struct EntityTrack {
    pub net_id: u32,
    /// The 5-bit engine entity type (11 = unit/biped, 13 = static/Forge object, ...).
    pub entity_type: u8,
    pub first_tick: u32,
    pub last_tick: u32,
    /// `(tick, position)` for every tick that carried base field 0, in tick order.
    pub positions: Vec<(u32, [f32; 3])>,
    /// `(tick, forward, up)` for every tick that carried base field 2.
    pub orientations: Vec<(u32, [f32; 3], [f32; 3])>,
    /// `(tick, yaw radians)` for every tick a BIPED reported its body facing. Bipeds use this
    /// instead of `orientations`, which their base block almost never carries.
    pub facings: Vec<(u32, f32)>,
    /// The raw 4-bit OWNER value out of this entity's creation payload, as the wire carries it
    /// (`slot + 1`, 0 = the engine's -1 "nobody"). Walked to its real place by
    /// [`creation_biped`], NOT read at a fixed offset. Every life of one player carries the same
    /// value, so this groups a player's respawns together exactly -- but it is NOT the per-tick
    /// player slot, so turning it into a gamertag still needs the group matched to a slot.
    pub owner_raw: Option<u8>,
    /// Whether this biped's creation carried an APPEARANCE block (armour pieces and colours).
    /// Only a PLAYER's biped does; the gametype's and the AI's do not, which is an independent
    /// statement of "this body belongs to somebody" that costs nothing to read.
    pub appearance: bool,
    /// #appearance  That block's CONTENTS: this body's own species, armour colours and twelve
    /// slot options. The film header carries the same struct per player slot, which makes this a
    /// route to the player that does not go through the owner stamp at all.
    pub look: Option<BipedAppearance>,
    /// #film-attach  Every time this object stated base field 12: `(tick, Some(parent record))`
    /// when it became attached, `(tick, None)` when the attachment was cleared. In tick order,
    /// so `attached_at` is a `partition_point` like `weapon_at`.
    pub attachments: Vec<(u32, Option<BaseAttachment>)>,
    /// #film-action  The 14-bit network tag reference at bit 0 of this entity's CREATION
    /// payload: **which object definition it was spawned from**, as `index + 1` into the
    /// engine's 9-group network tag-reference table (0 = the null reference).
    ///
    /// Resolve it with `crate::film_action::RefTable` built from the loaded map, which needs the
    /// cache's per-tag datum salts (`film_action::reach_tag_salts`) because the table is sorted
    /// by datum handle, not by tag index. Kept RAW here so the index stays map-free.
    pub def_wire: Option<u32>,
    /// Whether this entity ever moved, computed ONCE when the index is built.
    ///
    /// `moves()` used to sum the whole path on every call, and the per-frame draw filter calls it
    /// for every track — on a clip with 1,689 bodies of ~1,100 positions each that is two million
    /// square roots per filter pass, several times a frame. It is a property of the finished
    /// track, so it is computed once.
    pub moves_cached: Option<bool>,
    /// `(tick, 'weap' tag index)` each time this unit reported the weapon in its hands.
    ///
    /// Derived field 0x2A of the biped's dirty block — a `TagReference('weap')`. It arrives on
    /// about 0.13% of unit messages (6,945 over 5.4 M), i.e. when a player picks something up or
    /// swaps, which is exactly when it changes; between reports the last one holds.
    pub weapons: Vec<(u32, u32)>,
    /// #gamevariant  The weapon this body's player SPAWNED holding, in the same space as
    /// `weapons` (an index into the map's `weap` group list).
    ///
    /// The recording never says this: a unit reports its weapon only when it CHANGES. It comes
    /// from the film's own embedded GAME VARIANT instead — the base player traits' *initial
    /// primary weapon*, overridden by the loadout palette's *primary weapon* — and is filled in
    /// by `film_variant::apply_spawn_weapons` once a map is open to resolve the index.
    /// `weapon_at` falls back to it before the first `'weap'` report, so a real pickup still
    /// wins from the tick it arrives.
    pub spawn_weapon: Option<u32>,
    /// The FORGE SPAWN POINT this entity first appeared on: `(quota index, variant item,
    /// distance in wu)` of the nearest present object in the film's own embedded map variant.
    ///
    /// This is how a stream entity is IDENTIFIED. The creation payload's definition tag
    /// reference is not readable at any fixed offset (swept and refuted -- see
    /// `film_anim::probes::probe_sweep_creation_tagref`), but a Forge-placed object spawns at its
    /// Forge spawn point, and the quota index resolves through the scenario sandbox palette to a
    /// real tag with code HMS already has. Measured over 7 films: 97% of type-15 entities land
    /// within 1.5 wu of a palette object and **every one of the 17 distinct objects they land on
    /// is a `vehi` tag**, which is what proves type 15 is the vehicle.
    ///
    /// Not computed for type 13 (the Forge objects themselves, which trivially sit on their own
    /// spawn point and can number 370,000 in one film).
    pub spawned_on: Option<(u16, u8, f32)>,
    /// The tick a kind-4 DESTROY message named this entity, if one did. `None` means it was still
    /// alive when the recording stopped.
    pub destroyed_tick: Option<u32>,
}

impl EntityTrack {
    /// How far this entity travelled in total — the cheap test of "did it actually move".
    pub fn path_length(&self) -> f32 {
        self.positions
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0].1, w[1].1);
                ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
            })
            .sum()
    }
    pub fn moves(&self) -> bool {
        match self.moves_cached {
            Some(m) => m,
            // Only a track built outside `build_index_for` (a test fixture) pays for this.
            None => self.path_length() > 0.5,
        }
    }
    /// Does this entity EXIST at `tick`?
    ///
    /// Not the same question as "did it report at `tick`". An entity that does not move sends one
    /// creation message and then nothing at all, so `last_tick` is its creation tick — using that
    /// as a lifetime makes every parked vehicle and every dropped weapon invisible one tick after
    /// it appears. The real end of life is the kind-4 destroy, or the end of the film.
    pub fn alive_at(&self, tick: u32) -> bool {
        tick >= self.first_tick && self.destroyed_tick.map_or(true, |d| tick <= d)
    }
    /// The last known position at or before `tick`, with the next key for interpolation.
    pub fn sample(&self, tick: u32) -> Option<[f32; 3]> {
        let i = self.positions.partition_point(|(t, _)| *t <= tick);
        if i == 0 {
            return None;
        }
        let (t0, p0) = self.positions[i - 1];
        match self.positions.get(i) {
            Some(&(t1, p1)) if t1 > t0 => {
                let f = ((tick.saturating_sub(t0)) as f32 / (t1 - t0) as f32).clamp(0.0, 1.0);
                Some([
                    p0[0] + (p1[0] - p0[0]) * f,
                    p0[1] + (p1[1] - p0[1]) * f,
                    p0[2] + (p1[2] - p0[2]) * f,
                ])
            }
            _ => Some(p0),
        }
    }
    /// Ground speed in world units per second around `tick`, from the two bracketing keys.
    /// Returns `None` where the track has no usable pair (a respawn teleport is reported as the
    /// huge value it is; callers clamp).
    pub fn speed_at(&self, tick: u32) -> Option<f32> {
        let i = self.positions.partition_point(|(t, _)| *t <= tick);
        if i == 0 || i >= self.positions.len() {
            return None;
        }
        let (t0, p0) = self.positions[i - 1];
        let (t1, p1) = self.positions[i];
        if t1 <= t0 {
            return None;
        }
        let dt = (t1 - t0) as f32 / 60.0;
        let d = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        Some(d / dt)
    }

    /// Distance travelled along the track up to `tick`. Used to drive a locomotion cycle at the
    /// ground speed, which is what stops the feet skating — and it is a pure function of the
    /// tick, so scrubbing to the same moment always gives the same pose.
    pub fn distance_to(&self, tick: u32) -> f32 {
        let mut d = 0.0f32;
        for w in self.positions.windows(2) {
            let (t0, a) = w[0];
            let (t1, b) = w[1];
            if t0 >= tick {
                break;
            }
            if t1 <= t0 {
                continue;
            }
            let step = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            let dt = (t1 - t0) as f32 / 60.0;
            // A respawn teleport is a huge SPEED, not merely a long step -- filtering on raw
            // distance would also throw away a legitimate long gap between sparse keys.
            if step / dt > 12.0 {
                continue;
            }
            // Interpolate the segment the tick falls inside, so the phase is continuous.
            let f = if t1 <= tick { 1.0 } else { (tick - t0) as f32 / (t1 - t0) as f32 };
            d += step * f;
        }
        d
    }

    /// #film-attach  What this object was attached to at `tick`, from the last statement at or
    /// before it. `None` when it has stated none, or when the last one cleared it.
    pub fn attached_at(&self, tick: u32) -> Option<BaseAttachment> {
        let i = self.attachments.partition_point(|(t, _)| *t <= tick);
        if i == 0 {
            return None;
        }
        self.attachments[i - 1].1
    }

    /// The `'weap'` tag index this unit was holding at `tick`, from the last report at or before
    /// it. `None` before the first report — the recording simply had not said yet.
    pub fn weapon_at(&self, tick: u32) -> Option<u32> {
        let i = self.weapons.partition_point(|(t, _)| *t <= tick);
        // #gamevariant  Before the first report the body still spawned holding SOMETHING, and
        // the game variant is the only place that says what.
        if i == 0 { return self.spawn_weapon }
        Some(self.weapons[i - 1].1)
    }

    /// A biped's body yaw (radians) at `tick`, from the nearest facing sample at or before it.
    pub fn facing_at(&self, tick: u32) -> Option<f32> {
        if self.facings.is_empty() {
            return None;
        }
        let i = match self.facings.binary_search_by_key(&tick, |(t, _)| *t) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        Some(self.facings[i].1)
    }

    /// The last reported orientation at or before `tick`, HELD (no interpolation).
    pub fn sample_orientation(&self, tick: u32) -> Option<([f32; 3], [f32; 3])> {
        let i = self.orientations.partition_point(|(t, _, _)| *t <= tick);
        (i > 0).then(|| (self.orientations[i - 1].1, self.orientations[i - 1].2))
    }

    /// #vehicle-rot  The recorded orientation at `tick` as a **full basis**, SLERPED between the
    /// two bracketing samples, in exactly the shape [`EntityTrack::sample`] gives the position.
    ///
    /// Base field 2 is a `(19-bit up axis, 8-bit angle about it)` pair, and the engine's decode
    /// (`sub_1803D9330` -> `mvar::decode_orientation_bits`) turns it into an orthonormal
    /// `(forward, up)` — so pitch and roll are in the recording, not just a compass heading.
    /// Holding the last sample is honest but makes a tumbling vehicle step between reports; a
    /// vehicle reports field 2 on a large fraction of its updates but not every tick, so the two
    /// keys are converted to quaternions, slerped and converted back.
    ///
    /// A **pure function of the tick**: scrubbing backwards reproduces the same basis exactly,
    /// which is the rule every Theater placement obeys.
    pub fn orientation_at(&self, tick: u32) -> Option<([f32; 3], [f32; 3])> {
        let i = self.orientations.partition_point(|(t, _, _)| *t <= tick);
        if i == 0 {
            return None;
        }
        let (t0, f0, u0) = self.orientations[i - 1];
        match self.orientations.get(i) {
            Some(&(t1, f1, u1)) if t1 > t0 => {
                let w = ((tick.saturating_sub(t0)) as f32 / (t1 - t0) as f32).clamp(0.0, 1.0);
                let q = |f: [f32; 3], u: [f32; 3]| -> glam::Quat {
                    glam::Quat::from_mat3(&basis_mat3(f, u))
                };
                let m = glam::Mat3::from_quat(q(f0, u0).slerp(q(f1, u1), w));
                Some((m.x_axis.to_array(), m.z_axis.to_array()))
            }
            _ => Some((f0, u0)),
        }
    }
}

/// #vehicle-rot  The engine's object basis as a `Mat3`: `col0 = forward` (local +X),
/// `col1 = cross(up, forward)` (local +Y), `col2 = up` (local +Z).
///
/// This is `Matrix_BuildOrientation` (`haloreachnew.dll` 0x1800a7bec), the same rule
/// `scene::object_matrix` places every Forge object with — normalised here, because the only
/// consumer is a rotation that has to round-trip through a quaternion.
pub fn basis_mat3(fwd: [f32; 3], up: [f32; 3]) -> glam::Mat3 {
    let mut f = glam::Vec3::from(fwd);
    let mut u = glam::Vec3::from(up);
    if f.length_squared() < 1e-9 {
        f = glam::Vec3::X;
    }
    if u.length_squared() < 1e-9 {
        u = glam::Vec3::Z;
    }
    f = f.normalize();
    u = u.normalize();
    let mut l = u.cross(f);
    if l.length_squared() < 1e-12 {
        l = glam::Vec3::Y;
    }
    // Re-orthogonalise forward against up so the result is a true rotation even when the two
    // decoded vectors are a fraction of a degree off perpendicular (they are quantised).
    let l = l.normalize();
    glam::Mat3::from_cols(l.cross(u).normalize(), l, u)
}

/// A whole film decoded into entity tracks: what HMS needs to scrub a timeline.
#[derive(Clone, Debug, Default)]
pub struct FilmIndex {
    pub params: DecodeParams,
    pub frame: PositionFrame,
    /// Ticks in the film (== records).
    pub ticks: u32,
    /// The first record's tick value (films do not start at 0).
    pub first_tick: u32,
    pub tracks: Vec<EntityTrack>,
    /// Records whose envelope decoded.
    pub records_decoded: usize,
    pub records: usize,
    /// Entity messages whose payload decoded far enough to yield a position.
    pub positions_decoded: usize,
    pub entity_messages: usize,
    /// Per player slot (0..15), the aim samples that slot's per-tick records carried.
    pub player_aims: Vec<Vec<(u32, PlayerAim)>>,
    /// #player-unit  How many `s_player_action` blocks the whole film carried, named or not.
    /// Separating this from `player_objects` is what distinguishes "the channel is absent" from
    /// "the channel is there and states no object".
    pub player_action_blocks: usize,
    /// #player-unit  Per player slot (0..15), every `s_player_action` block that named an object,
    /// in tick order. **This is the engine's own player-to-unit assignment, replicated.**
    ///
    /// The engine keeps the mapping both ways at runtime: `player_datum + 0x28` is the unit's
    /// object index (`Player_SetUnit`, `haloreach.dll+0x45814`) and the object carries the
    /// reverse link at `+0x1CC`. The wire form is this block — see [`PlayerAction`] and
    /// [`FilmIndex::unit_of_slot_at`].
    pub player_objects: Vec<Vec<(u32, PlayerAction)>>,
    /// Every present object in the film's own embedded map variant, as
    /// `(quota index, variant item, position)` — the table a stream entity is IDENTIFIED against
    /// (see `EntityTrack::spawned_on`). Carried here so a consumer that has the loaded map's
    /// sandbox palette can do its own typed nearest-neighbour search (e.g. "the nearest VEHICLE
    /// spawn point") without needing the film back. ~640 rows.
    pub variant_objects: Vec<(u16, u8, [f32; 3])>,
    /// Width of a `TagReference('weap')` in this film's stream, bits.
    ///
    /// `bits_needed(weap_tag_count + 1)` over the MAP, which the decode worker does not have — so
    /// it is solved the same way the other three pieces of outside state are, by trial decode
    /// against an oracle. The oracle here is free: a unit message states its dirty mask twice,
    /// so a wrong width desynchronises the rest of the walk and the rebuilt mask stops matching.
    pub weapref_bits: u32,
    /// Width of a `TagReference('eqip')` in a biped's CREATION payload, solved per film by
    /// [`solve_eqip_bits`]. The owner stamp sits just before it, so the walk that reaches the
    /// stamp has to get this right to be checkable at all.
    pub eqip_bits: u32,
    /// The fraction of this film's biped creations that the walk closed to within 7 bits -- the
    /// oracle on the owner stamp. Below ~1.0 means some creations were not understood and their
    /// bodies carry no owner rather than a guessed one.
    pub creations_closed: f64,
    /// #appearance  How many bodies [`FilmIndex::bind_by_appearance`] named from their armour,
    /// i.e. ones the owner stamp could not name.
    pub appearance_named: usize,
    /// #appearance  `(net_id, gamertag, source film)` for every body whose armour matched a
    /// player named in ANOTHER recording's roster and nowhere ambiguous. Read it through
    /// [`FilmIndex::inferred_name`]. **Inferred across recordings, not stated by this file** --
    /// see [`FilmIndex::bind_by_appearance_across_films`].
    pub cross_film_names: Vec<(u32, String, String)>,
    /// How many bodies got a name that way. Separate from `appearance_named` on purpose: these
    /// are a weaker claim and are counted, labelled and drawn separately.
    pub appearance_cross_named: usize,
    /// #appearance  Every gamertag the cross-film route MATCHED and then REFUSED, because the
    /// bodies claiming it overlap beyond [`MAX_PLAYER_BODIES_AT_ONCE`] -- one player cannot drive
    /// four bipeds, so the armour is not identifying one person and no subset of them is safe.
    ///
    /// **This is bookkeeping a gate depends on, not a diagnostic.** A refusal and a route that
    /// was never called both show up as a missing name, and
    /// `production_path_gate::the_entry_point_the_app_calls_yields_every_inferable_name` exists
    /// precisely to tell those two apart -- so the refusal has to be recorded where that gate can
    /// see it. Without this the gate's only options were to recompute the guard itself (grading
    /// production against a copy of its own rule) or to weaken to `>=` (which is exactly the hole
    /// the gate was written to close: a silently dead route passes `>=` with zero names).
    pub cross_film_refused: Vec<String>,
    /// How many times a body's HELD weapon changed, over the whole film. A unit states its four
    /// weapon references and its selection separately and only on change, so this counts the
    /// resolved changes, not the raw `'weap'` references.
    pub weapons_decoded: usize,
    /// THE SCOREBOARD. Per statborg slot (0..23: 0..15 players, 16..23 teams), every score
    /// update the recording carried, as `(tick, round_score, total_score, rounds_won, final)`.
    ///
    /// Entity type 5 is `game-engine-statborg` and its index IS the player configuration slot,
    /// so this needs no owner bridge. Only messages that passed both free oracles (the rebuilt
    /// presence mask equal to the message's own 64-bit mask, and the payload consumed to within
    /// the writer's byte padding) are kept.
    pub statborg: Vec<Vec<(u32, StatborgSample)>>,
    /// Statborg messages seen, and how many passed both oracles. A film where these differ is
    /// one whose scores should not be trusted, and the UI says so.
    pub statborg_messages: usize,
    pub statborg_decoded: usize,
    /// The configuration slot of the player who RECORDED this film, from the header. Supplied by
    /// the caller because it comes from `chdr`, not the stream.
    pub recorder_slot: Option<u8>,
    /// Set when the recorder anchor corroborated `slot = owner - 1` on THIS film, which is what
    /// licenses naming every body rather than only the recorder's.
    pub owner_rule_ok: bool,
    /// WHICH anchor did the corroborating, and therefore whether the promoted names are proven
    /// or derived. See [`OwnerAnchor`].
    pub owner_anchor: OwnerAnchor,
    /// Bit per player slot: the header slots that are OCCUPIED. Supplied by the caller because it
    /// comes from `s_game_options`, not the stream.
    ///
    /// Zero means "the caller did not say", not "nobody" — the derived-anchor path then falls
    /// back to the weaker proxy of a slot carrying per-tick records.
    pub occupied_slots: u16,
    /// #catchall  Bit per player slot: the OCCUPIED slots whose header record says ELITE
    /// (`player_model_choice` = 1). Supplied by the caller for the same reason as
    /// `occupied_slots` -- it comes from `s_game_options`, not the stream.
    ///
    /// Zero means "the caller did not say", which leaves [`FilmIndex::bind_by_elimination`]
    /// with nothing to eliminate on and so names nobody extra.
    pub elite_slots: u16,
    /// Which network entity each player slot turned out to be driving. Derived, with a margin --
    /// see `bind_players`.
    pub bindings: Vec<PlayerBinding>,
    /// #film-action  THE EVENT CHANNEL -- list B command **kind 1**, every occurrence, folded
    /// onto the film's tick line and in tick order.
    ///
    /// This is the channel the engine replicates discrete ACTIONS on: 72 named event types
    /// (`biped_assassinate`, `biped_enter_vehicle`, `biped_melee_initiate`, `weapon_reload`,
    /// `incident` -- the kill feed -- ...), each carrying up to two object references. It is
    /// NOT the unit's derived field 0x36, which a film essentially never carries (2 messages in
    /// 7,367,920). See `crate::film_action` and `re-notes/reach_saved_film_layout.md`.
    pub events: Vec<crate::film_action::ActionEvent>,
    /// Kind-1 commands seen, and how many the event decoder accepted. A film where these differ
    /// is one whose events should not be trusted.
    pub event_commands: usize,
    pub elapsed: std::time::Duration,
}

/// Which anchor corroborated `slot = owner - 1` on a film, and therefore how much weight the
/// resulting names carry.
///
/// The rule is never applied blind (see [`FilmIndex::apply_owner_rule`]); it is corroborated
/// per film against a player whose slot AND owner are both known independently. There are two
/// ways to be that player, and they are not equally strong.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerAnchor {
    /// Nothing corroborated the rule, so nothing was promoted. Only aim-correlated groups keep a
    /// name.
    #[default]
    None,
    /// The `chdr` AUTHOR RECORD named the recording player, and their slot agreed with the rule.
    /// This is the proven path and it is what every finished film uses.
    AuthorRecord,
    /// **Derived.** The film carried no author record — an in-progress `.temp` has none, because
    /// the author fields are only written when the recording is finalised — so the recorder was
    /// identified instead by the one player slot whose AIM IS LIVE.
    ///
    /// That identification is sound: a player's own input is captured locally and the aim field
    /// is zero-filled for everyone else, so exactly one slot per film has a yaw that actually
    /// moves. It is still an inference rather than a stated fact, so names promoted under it are
    /// **unverified**, and the rule is additionally required to land on an OCCUPIED header slot
    /// with no two owners colliding.
    LiveAim,
}

impl OwnerAnchor {
    pub fn label(self) -> &'static str {
        match self {
            OwnerAnchor::None => "none (the rule was not corroborated on this film)",
            OwnerAnchor::AuthorRecord => "the film's author record (proven)",
            OwnerAnchor::LiveAim => "the live-aim player slot (derived, unverified)",
        }
    }
    /// True when the names this anchor promoted are inferences rather than stated facts.
    pub fn is_derived(self) -> bool {
        matches!(self, OwnerAnchor::LiveAim)
    }
}

/// One statborg update, folded into a running state. `None` means that field was not in this
/// particular message and the previous value still holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatborgSample {
    pub round_score: Option<i16>,
    pub total_score: Option<i16>,
    pub rounds_won: Option<i16>,
    pub is_final: Option<bool>,
}

/// #catchall  WHICH EVIDENCE named a body, and therefore how much weight the name carries.
///
/// A wrong gamertag on a body is worse than no gamertag, so every binding records how it was
/// arrived at and the UI says so wherever it was not the direct stamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindOrigin {
    /// The AIM CORRELATION alone: the biped's heading scored against each slot's recorded aim
    /// yaw. Only one slot per film has a live aim, so in practice this names the recorder.
    #[default]
    AimCorrelation,
    /// The creation message's 4-bit OWNER STAMP, through the `slot = owner - 1` rule after that
    /// rule was corroborated on this film. This is how almost every body is named.
    OwnerStamp,
    /// #catchall  **ELIMINATION on species.** The stamp could not name this body (it carries the
    /// catch-all 0), but the header leaves exactly ONE occupied slot unclaimed, that slot's
    /// `player_model_choice` picks one biped definition class, and this body is the only body of
    /// that class left over -- with no two of them alive at once. See
    /// [`FilmIndex::bind_by_elimination`].
    SpeciesElimination,
    /// #appearance  **THE ARMOUR.** The body's creation message carries its player's whole
    /// `s_player_appearance` -- species, three armour colours and the twelve slot options -- and
    /// the film header carries the same struct per slot, so a body whose appearance reproduces
    /// EXACTLY ONE present header slot's is that player's. Needs no owner field at all. See
    /// [`FilmIndex::bind_by_appearance`].
    Appearance,
}

impl BindOrigin {
    pub fn label(self) -> &'static str {
        match self {
            BindOrigin::AimCorrelation => "the aim correlation",
            BindOrigin::OwnerStamp => "the creation message's owner stamp",
            BindOrigin::Appearance => {
                "the armour: this body's appearance matches exactly one header slot's"
            }
            BindOrigin::SpeciesElimination => {
                "elimination: the only header slot left unclaimed, and its species"
            }
        }
    }
}

/// One LIFE of one player: a network entity, the player slot that was driving it, and the tick
/// range over which it existed.
///
/// Nothing in the stream states this link outright, so it is *derived*. A player's per-tick
/// record carries that player's aim yaw, tagged with the player slot; the biped they are driving
/// turns to follow that aim. So each biped entity is scored against each slot over the biped's
/// own lifetime, and the best-scoring slot wins it.
///
/// The binding is per entity, NOT per player, because a player gets a NEW network entity every
/// time they respawn -- 83 biped entities for 4 players in one 8-minute film measured here. A
/// camera that follows "a player" therefore has to look up the entity that was theirs *at that
/// tick*, which is what `entity_for_slot` does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerBinding {
    pub slot: u8,
    pub net_id: u32,
    pub entity_type: u8,
    pub first_tick: u32,
    pub last_tick: u32,
    /// Mean absolute circular difference between the biped's heading and the slot's aim, radians.
    pub cost: f32,
    /// Runner-up slot's cost minus `cost`, radians. Big = unambiguous.
    pub margin: f32,
    /// How many ticks were compared.
    pub samples: usize,
    /// True when the heading came from the decoded facing; false when it came from the direction
    /// of travel (the fallback, which strafing and backpedalling make noisier).
    pub from_facing: bool,
    /// True when this binding comes from the `slot = owner - 1` rule rather than from the aim
    /// correlation. Only ever set on a film where the recorder anchor CORROBORATED that rule.
    pub from_owner_rule: bool,
    /// #catchall  WHICH evidence named this body. `from_owner_rule` is the same statement for
    /// the stamp case and is kept because the existing gates are written against it.
    pub origin: BindOrigin,
}

impl PlayerBinding {
    /// A match this reader is willing to put a gamertag on. The facing bound is tight because
    /// the engine turns the body onto the aim; the travel-direction fallback is allowed a much
    /// wider error because strafing legitimately points the body away from the aim.
    pub fn is_confident(&self) -> bool {
        // A rule-derived binding is only ever produced on a film whose recorder anchor agreed
        // with the rule, which is a per-film proof rather than a guess, so it needs no angle.
        if self.from_owner_rule {
            return true;
        }
        // #appearance  An APPEARANCE bind carries no angle either. Its confidence is the
        // structural one `bind_by_appearance` enforces: the body reproduces exactly one present
        // header slot's whole appearance, bit for bit, and no other slot's.
        if self.origin == BindOrigin::Appearance {
            return true;
        }
        // #catchall  Nor does an ELIMINATION bind, for the same reason: it carries no angle at
        // all, and its confidence is the structural one `bind_by_elimination` already enforced
        // (one free slot of that species, one class, no two bodies alive at once). Scoring it by
        // `margin` -- which is zero, because there is no runner-up angle to subtract -- would
        // throw away every name it produces.
        if matches!(self.origin, BindOrigin::SpeciesElimination) {
            return true;
        }
        let limit = if self.from_facing { 0.45 } else { 1.0 };
        self.cost < limit && self.margin > 0.15 && self.samples >= 45
    }
}

impl FilmIndex {
    pub fn seconds(&self) -> f32 {
        self.ticks as f32 / 60.0
    }
    /// Tracks that actually move — the ones worth drawing as motion.
    pub fn moving(&self) -> impl Iterator<Item = &EntityTrack> {
        self.tracks.iter().filter(|t| t.moves())
    }
    /// Every entity alive at `tick`, with its interpolated position and held orientation.
    pub fn sample(&self, tick: u32) -> Vec<(&EntityTrack, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        self.tracks
            .iter()
            .filter(|t| tick >= t.first_tick && tick <= t.last_tick + 60)
            .filter_map(|t| t.sample(tick).map(|p| (t, p, t.sample_orientation(tick))))
            .collect()
    }
    /// The first tick at which any player biped exists. A Halo round opens with a countdown and
    /// a spawn sequence, so for the first seconds of a clip there are NO player entities at all —
    /// the map and its Forge objects are there and the players are not. Seeking to tick 0 and
    /// concluding that nothing is drawn is therefore wrong, and this is what says so.
    /// The aim this slot recorded at (or just before) `tick`, if the slot has any samples.
    pub fn aim_at(&self, slot: u8, tick: u32) -> Option<PlayerAim> {
        let v = self.player_aims.get(slot as usize)?;
        if v.is_empty() {
            return None;
        }
        let i = match v.binary_search_by_key(&tick, |(t, _)| *t) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        Some(v[i].1)
    }

    /// The binding for a player slot, if one was derived.
    pub fn binding_for_slot(&self, slot: u8) -> Option<&PlayerBinding> {
        self.bindings.iter().find(|b| b.slot == slot)
    }

    /// The player slot driving this entity, if one was derived confidently.
    /// #stamp15  Is this OWNER GROUP named, by any route this reader has?
    ///
    /// `slot_of_owner` answers only for the stamp itself, and returns `None` for wire 0 by
    /// design. But a stamp-0 group can still be named -- by the four-bit truncation recovery
    /// ([`FilmIndex::bind_stamp_zero_by_truncation`]), by species elimination, or by the aim
    /// correlation -- and all three land in `bindings`, which is what actually feeds the
    /// nameplates and the camera list through [`FilmIndex::slot_of_entity`]. So any count of
    /// "how many players can be named" has to ask the bindings too, or it reports a player as
    /// nameless while their gamertag is being drawn over their head.
    pub fn owner_group_is_named(&self, owner: u8) -> bool {
        if self.slot_of_owner(owner).is_some() {
            return true;
        }
        self.tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner))
            .any(|t| self.slot_of_entity(t.net_id).is_some())
    }

    pub fn slot_of_entity(&self, net_id: u32) -> Option<u8> {
        self.bindings.iter().find(|b| b.net_id == net_id && b.is_confident()).map(|b| b.slot)
    }

    /// Which entity was this slot's biped at `tick`.
    pub fn entity_for_slot(&self, slot: u8, tick: u32) -> Option<&PlayerBinding> {
        self.bindings
            .iter()
            .filter(|b| b.slot == slot && b.is_confident() && tick >= b.first_tick && tick <= b.last_tick)
            .min_by_key(|b| b.last_tick - b.first_tick)
    }

    /// Every life this reader attributed to a slot, in tick order.
    pub fn lives_of_slot(&self, slot: u8) -> Vec<&PlayerBinding> {
        let mut v: Vec<&PlayerBinding> =
            self.bindings.iter().filter(|b| b.slot == slot && b.is_confident()).collect();
        v.sort_by_key(|b| b.first_tick);
        v
    }

    /// The heading samples for a biped track: the decoded facing where there is one, else the
    /// direction of travel. Returns `(samples, from_facing)`.
    #[cfg(test)]
    pub fn heading_samples_pub(&self, tr: &EntityTrack) -> (Vec<(u32, f32)>, bool) {
        self.heading_samples(tr)
    }

    fn heading_samples(&self, tr: &EntityTrack) -> (Vec<(u32, f32)>, bool) {
        // A biped's own body-yaw field, where it has one.
        if tr.facings.len() >= 30 {
            return (tr.facings.clone(), true);
        }
        if tr.orientations.len() >= 30 {
            let v = tr.orientations.iter().map(|(t, f, _)| (*t, f[1].atan2(f[0]))).collect();
            return (v, true);
        }
        // Fallback: where the entity was actually going. Only segments with real speed carry a
        // meaningful heading, so near-stationary ticks are dropped rather than fed in as noise.
        let mut v = Vec::new();
        for w in tr.positions.windows(2) {
            let ((t0, a), (t1, b)) = (w[0], w[1]);
            let dt = t1.saturating_sub(t0) as f32 / 60.0;
            if dt <= 0.0 || dt > 0.5 {
                continue;
            }
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            if (dx * dx + dy * dy).sqrt() / dt < 1.0 {
                continue;
            }
            v.push((t1, dy.atan2(dx)));
        }
        (v, false)
    }

    /// The tick range over which a slot had per-tick records at all.
    pub fn slot_tick_range(&self, slot: u8) -> Option<(u32, u32)> {
        let v = self.player_aims.get(slot as usize)?;
        Some((v.first()?.0, v.last()?.0))
    }

    /// Match every biped entity to the player slot that was driving it.
    ///
    /// Two independent facts do the work:
    ///
    /// 1. **The creation payload stamps the owner** (`CREATION_OWNER_BIT`), so every life of one
    ///    player is grouped together exactly -- 15 respawns on one measured film. That numbering
    ///    is not the per-tick player slot, so it names nobody on its own.
    /// 2. **A player's aim yaw is recorded per tick, tagged with the slot**, and their biped's
    ///    body turns to follow it.
    ///
    /// So: group the bipeds by owner, pool *every* heading sample in a group, and score the whole
    /// group against each slot. Pooling is what makes this work -- a single 4-second life gives a
    /// noisy score and used to name about one player in five, while a group pools every life that
    /// player ever had. Groups are then assigned to slots one-to-one, best cost first, and every
    /// life in a group inherits the group's slot.
    ///
    /// Bipeds with no owner field fall back to being scored individually, as before.
    pub fn bind_players(&mut self) {
        self.bindings.clear();
        let slots: Vec<usize> =
            (0..self.player_aims.len().min(16)).filter(|&i| self.player_aims[i].len() >= 30).collect();
        if slots.is_empty() {
            return;
        }
        // Group the bipeds by owner; `None` owners each form their own singleton group.
        let mut groups: std::collections::BTreeMap<i16, Vec<usize>> = Default::default();
        for ti in 0..self.tracks.len() {
            if self.tracks[ti].entity_type != 11 {
                continue;
            }
            let key = self.tracks[ti].owner_raw.map(|o| o as i16).unwrap_or(-1 - ti as i16);
            groups.entry(key).or_default().push(ti);
        }
        // Score each group against each slot on its POOLED heading samples.
        // (cost, group key, slot, samples, from_facing)
        let mut cand: Vec<(f32, i16, u8, usize, bool)> = Vec::new();
        let mut runner_up: std::collections::HashMap<i16, f32> = Default::default();
        for (key, tis) in &groups {
            // #catchall  A group that is several people is not worth scoring and must not be
            // named: pooling thirteen players' headings and handing the winner one gamertag is
            // how `4ED6DDC1_67FF2AFF` came to have 16 bodies on slot 0 with 13 alive at once.
            // See `owner_group_is_one_player` for the measured threshold.
            if *key >= 0 && !self.owner_group_is_one_player(*key as u8) {
                continue;
            }
            let mut pooled: Vec<(u32, f32)> = Vec::new();
            let mut any_facing = false;
            for &ti in tis {
                let (h, from_facing) = self.heading_samples(&self.tracks[ti]);
                any_facing |= from_facing;
                pooled.extend(h);
            }
            if pooled.len() < 45 {
                continue;
            }
            let mut scores: Vec<(f32, u8, usize)> = Vec::new();
            for &slot in &slots {
                let mut d: Vec<f32> = Vec::with_capacity(pooled.len());
                for (t, h) in &pooled {
                    let Some(aim) = self.aim_at(slot as u8, *t) else { continue };
                    d.push(ang_diff(*h, aim.yaw()).abs());
                }
                if d.len() >= 45 {
                    let n = d.len();
                    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    scores.push((d[n / 2], slot as u8, n));
                }
            }
            scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(r) = scores.get(1) {
                runner_up.insert(*key, r.0);
            }
            for (cost, slot, n) in scores {
                cand.push((cost, *key, slot, n, any_facing));
            }
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut taken_group: std::collections::HashSet<i16> = Default::default();
        let mut taken_slot = [false; 16];
        let mut out: Vec<PlayerBinding> = Vec::new();
        for &(cost, key, slot, n, from_facing) in &cand {
            if taken_group.contains(&key) || taken_slot[(slot & 15) as usize] {
                continue;
            }
            taken_group.insert(key);
            taken_slot[(slot & 15) as usize] = true;
            let margin = runner_up.get(&key).copied().unwrap_or(std::f32::consts::PI) - cost;
            // Every life in the group inherits the group's slot: identity persists for a life, and
            // one confident group names all of that player's lives at once.
            for &ti in &groups[&key] {
                let tr = &self.tracks[ti];
                out.push(PlayerBinding {
                    slot,
                    net_id: tr.net_id,
                    entity_type: tr.entity_type,
                    first_tick: tr.first_tick,
                    last_tick: tr.last_tick,
                    cost,
                    margin,
                    samples: n,
                    from_facing,
                    from_owner_rule: false,
                    origin: BindOrigin::AimCorrelation,
                });
            }
        }
        out.sort_by_key(|b| (b.slot, b.first_tick));
        self.bindings = out;
        self.apply_owner_rule(&groups);
        // #catchall  Last: the bodies neither channel named. The stamp has had its say and the
        // claimed slots are settled, which is exactly what elimination needs.
        self.bind_by_elimination();
        self.drop_bindings_to_unoccupied_slots();
    }

    /// A binding to a slot the header does not occupy can never show a gamertag, because the
    /// gamertag IS the header record -- so it names nobody and is dropped.
    ///
    /// **Why this is here rather than in each route.** `apply_owner_rule`'s non-strict path
    /// accepts any slot with per-tick player records (`slot > 15 || player_aims[slot].is_empty()`
    /// is its whole test), so on a film whose header lists one player while records exist for
    /// several it binds slots that have no name attached. That was invisible while it was the
    /// only route -- `one_slot_is_one_player_over_real_films` asserts "no more names than
    /// players" and one such binding on a one-player header still satisfied `1 <= 1`. Adding the
    /// armour route, which DOES check occupancy, pushed the count to 2 and exposed it.
    ///
    /// Doing it centrally and last is deliberate: it is the invariant's own statement, it applies
    /// to every route at once, and **it cannot rename anybody** -- it only removes bindings whose
    /// slot has no gamertag to display. Tightening the non-strict path in place was the
    /// alternative, and its own comment warns that changing that path's acceptance "would
    /// silently rename people in every existing clip".
    fn drop_bindings_to_unoccupied_slots(&mut self) {
        if self.occupied_slots == 0 {
            return; // no header roster to judge against; `slot_occupied` falls back to records
        }
        let occupied = self.occupied_slots;
        self.bindings.retain(|b| b.slot <= 15 && occupied & (1 << b.slot) != 0);
    }

    /// #catchall  Name a body the OWNER STAMP could not, by ELIMINATION on species.
    ///
    /// ## The problem
    ///
    /// The creation stamp is `slot + 1` in a four-bit field, so a slot-15 player writes 0 -- but
    /// **0 is also what a body carries when the recording attributed it to nobody**, and on some
    /// films group 0 is a CATCH-ALL holding several players' bodies at once. Measured over this
    /// machine's 25 films, 13 carry a stamp-0 body. The engine's own encoder settles what that
    /// means: it writes `index + 1` in four bits, so wire 0 is its `-1` sentinel AND the field
    /// tops out at index 14 -- it cannot name a sixteenth player at all.
    /// [`FilmIndex::owner_to_slot`] therefore refuses every stamp-0 body -- and one
    /// player cannot drive two bipeds at the same moment, so a real owner group's lives never
    /// overlap -- and that leaves those bodies with no name, which is what the user sees as
    /// "spartans walking around with no name attached".
    ///
    /// ## The second, independent signal
    ///
    /// The header states each player's SPECIES: `s_player_configuration` +0x4C
    /// `player_model_choice`, 0 Spartan / 1 Elite ([`elite_slot_mask`]). The stream states, for
    /// every body, the OBJECT DEFINITION it was spawned from -- a network tag reference at bit 0
    /// of its creation message (`EntityTrack::def_wire`). Those two facts come from different
    /// halves of the file and neither is the owner stamp.
    ///
    /// So: the bodies the stamp DID name teach the definition-class-to-species map on this very
    /// film (class 778 only ever belongs to Elite slots, class 772 only to Spartan ones), and
    /// then the leftovers are matched to the leftover slots by that map. The bind is taken only
    /// where it is FORCED:
    ///
    /// 1. the stamp rule was corroborated on this film, so "claimed" means something;
    /// 2. exactly ONE occupied header slot of that species is still unclaimed;
    /// 3. the class is a clean species discriminator here -- every body of it that the stamp
    ///    named belongs to a slot of one species, never both;
    /// 4. the leftover bodies of that class never coexist, so they really are one player; and
    /// 5. every one of them falls inside that slot's own per-tick record range.
    ///
    /// Anything less is ambiguous and names nobody.
    ///
    /// ## Measured on the test clip
    ///
    /// `asq_mglo-1_forge_hal_21897D8A`: 16 occupied slots, 112 bodies, owner groups 1..15 naming
    /// slots 0..14 (106 bodies), and group 0 a catch-all of 6. The header's Elites are slots 3, 6,
    /// 9 and 15; the stamp names 7 + 10 + 23 = **40** Elite-class bodies for slots 3/6/9, the
    /// clip holds **41** in total, and the one left over is the single Elite body in group 0
    /// (`0xeb940221`, 0:10.9..5:22.8). Slot 15 (`Jega Rdomnai544`, Elite) is the only unclaimed
    /// occupied slot. One slot, one body, one species: the match is forced, not chosen.
    ///
    /// The other five bodies in that group are Spartan-class and **stay unnamed**, which is the
    /// honest answer rather than a failure: all twelve Spartan slots are already claimed, so the
    /// file holds no name that could belong to them. They are late arrivals the header -- written
    /// once, at the start -- never recorded.
    fn bind_by_elimination(&mut self) {
        // (1) Only on a film whose stamp rule was corroborated. Without it the bindings are the
        // aim correlation's one answer, nearly every slot is "unclaimed", and eliminating against
        // that would be eliminating against nothing.
        if !self.owner_rule_ok || self.elite_slots == 0 && self.occupied_slots == 0 {
            return;
        }
        let bound: std::collections::HashSet<u32> =
            self.bindings.iter().filter(|b| b.is_confident()).map(|b| b.net_id).collect();
        let claimed: std::collections::HashSet<u8> =
            self.bindings.iter().filter(|b| b.is_confident()).map(|b| b.slot).collect();
        // (3) The definition-class -> species map, learned from the bodies the stamp named on
        // THIS film: (spartan votes, elite votes) per class.
        let mut votes: std::collections::HashMap<u32, (usize, usize)> = Default::default();
        for b in self.bindings.iter().filter(|b| b.is_confident()) {
            if !self.slot_occupied(b.slot) {
                continue;
            }
            let Some(w) = self.tracks.iter().find(|t| t.net_id == b.net_id).and_then(|t| t.def_wire)
            else {
                continue;
            };
            let e = votes.entry(w).or_default();
            if self.elite_slots & (1 << b.slot) != 0 {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
        // A class speaks for a species only if it has never been seen for the other one. A
        // gametype that FORCES a species makes every class carry both kinds of vote, and this is
        // what makes the rule stand down there instead of guessing.
        let class_elite = |w: u32| -> Option<bool> {
            match votes.get(&w) {
                Some((0, e)) if *e > 0 => Some(true),
                Some((s, 0)) if *s > 0 => Some(false),
                _ => None,
            }
        };
        let free: Vec<u8> =
            (0..16u8).filter(|s| self.slot_occupied(*s) && !claimed.contains(s)).collect();
        if free.is_empty() {
            return;
        }
        let leftover: Vec<(u32, u32, u32, u8, Option<u32>)> = self
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11 && !bound.contains(&t.net_id))
            .map(|t| (t.net_id, t.first_tick, t.last_tick, t.entity_type, t.def_wire))
            .collect();
        if leftover.is_empty() {
            return;
        }
        // 2 s at 60 Hz, the same grace the overlap rule uses: a corpse lingering past its own
        // respawn is the ordinary case and is not two bodies at once.
        const GRACE: u32 = 120;
        let mut out: Vec<PlayerBinding> = Vec::new();
        for want_elite in [false, true] {
            // (2) exactly one unclaimed occupied slot of this species
            let mut slots = free
                .iter()
                .copied()
                .filter(|s| (self.elite_slots & (1 << *s) != 0) == want_elite);
            let Some(slot) = slots.next() else { continue };
            if slots.next().is_some() {
                continue; // two players it could be: ambiguous, so nobody
            }
            let mine: Vec<&(u32, u32, u32, u8, Option<u32>)> = leftover
                .iter()
                .filter(|(_, _, _, _, w)| w.and_then(class_elite) == Some(want_elite))
                .collect();
            if mine.is_empty() {
                continue;
            }
            // (4) one player, one body at a time. FULLY pairwise, not just adjacent-in-start-
            // order: a long body can straddle two later short ones, and comparing only
            // neighbours misses that.
            let spans: Vec<(u32, u32)> = mine.iter().map(|(_, a, b, _, _)| (*a, *b)).collect();
            let coexist = |x: (u32, u32), y: (u32, u32)| {
                x.1 > y.0 + GRACE && y.1 > x.0 + GRACE
            };
            if (0..spans.len())
                .any(|i| (i + 1..spans.len()).any(|j| coexist(spans[i], spans[j])))
            {
                continue; // several players share this class here: nobody
            }
            // (5) containment in that slot's own per-tick record range
            let Some((lo, hi)) = self.slot_tick_range(slot) else { continue };
            if spans.iter().any(|(a, b)| *a + GRACE < lo || *b > hi + GRACE) {
                continue;
            }
            for (net_id, first_tick, last_tick, entity_type, _) in mine {
                out.push(PlayerBinding {
                    slot,
                    net_id: *net_id,
                    entity_type: *entity_type,
                    first_tick: *first_tick,
                    last_tick: *last_tick,
                    // The elimination carries no angle of its own; the margin is the structural
                    // one and is recorded in `origin`, not as a number that would read like a
                    // correlation score.
                    cost: 0.0,
                    margin: 0.0,
                    samples: 0,
                    from_facing: false,
                    from_owner_rule: false,
                    origin: BindOrigin::SpeciesElimination,
                });
            }
        }
        if out.is_empty() {
            return;
        }
        self.bindings.extend(out);
        self.bindings.sort_by_key(|b| (b.slot, b.first_tick));
    }

/// #owner-wrap  The player CONFIGURATION SLOT a 4-bit owner stamp names, if any.
///
/// **This is the engine's own arithmetic, read off its encoder and decoder — it is not a fit.**
/// The field is a SIGNED BYTE in the unit's creation record at `+0x3E`, and `-1` is its "nobody"
/// sentinel:
///
/// * the writer, `haloreach.dll+0x3B242C`, does `write_bits(*(char *)(obj + 0x3E) + 1, 4)`;
/// * the reader, `haloreach.dll+0x3B2590` (the unit creation reader — it is the one that goes on
///   to decode the `'eqip'` tag reference, which is what identifies it as entity type 11), does
///   `*(a3 + 0x3E) = Bitstream_ReadBits(bs, 4) - 1`.
///
/// Two consequences, and the first corrects what this reader used to do:
///
/// 1. **Wire 0 means −1 means NOBODY. There is no wrap.** This function previously computed
///    `(owner - 1) & 0x0F`, which sends wire 0 to slot 15 — an inference drawn from a 16-player
///    clip whose sixteenth player happened to be the one unclaimed slot. The engine never writes
///    16: it writes `index + 1` and the largest value four bits hold is 15.
/// 2. **The field tops out at index 14, so it can name at most FIFTEEN players.** On a
///    sixteen-player recording at least one player's bodies necessarily carry "nobody" here, and
///    the file does not state who they are *through this field*. That is a limit of the format,
///    not of this reader, and it is why [`FilmIndex::bind_by_elimination`] exists.
pub(crate) fn owner_to_slot(&self, owner: u8) -> Option<u8> {
        // Wire 0 is the engine's `-1`: this body is attributed to nobody. Full stop — no lobby
        // size makes it mean slot 15.
        if owner == 0 {
            return None;
        }
        let slot = owner - 1;
        // #catchall  The structural test, for EVERY stamp. See
        // `owner_group_is_one_player`: the guard above was written for stamp 0 because that is
        // where the wrap made it urgent, but a stamp is a stamp and the corpus holds groups with
        // other numbers that are plainly several people too.
        if !self.owner_group_is_one_player(owner) {
            return None;
        }
        Some(slot)
    }

    /// #appearance  A gamertag inferred for `net_id` from ANOTHER recording's roster, with the
    /// film it came from. `None` for every body named a stated way, or not named at all.
    ///
    /// **This is deliberately NOT a `PlayerBinding`.** A binding names a SLOT in this film's
    /// header, and the whole point of a cross-film name is that this film's header does not
    /// contain the player. Keeping it in its own channel means nothing can mistake inferred
    /// evidence for stated evidence, and the UI has to opt in to drawing it.
    pub fn inferred_name(&self, net_id: u32) -> Option<(&str, &str)> {
        self.cross_film_names
            .iter()
            .find(|(id, _, _)| *id == net_id)
            .map(|(_, tag, src)| (tag.as_str(), src.as_str()))
    }

    /// #appearance  NAME A BODY BY ITS ARMOUR, with no owner field involved.
    ///
    /// A biped's creation message carries its player's whole `s_player_appearance` -- two flag
    /// bits, three armour colour indices, `player_model_choice` and twelve armour-slot options --
    /// and the film header carries the same struct, at the same internal offsets, per player
    /// slot. So the header is an answer key: **a body whose appearance reproduces exactly one
    /// present slot's, bit for bit, is that player's body.**
    ///
    /// Measured. On `*21897D8A*` 104 of 108 player bodies reproduce a header slot exactly across
    /// 14 slots -- which is also what proves the creation walk lands on real data. On
    /// `*53C14E9D*`, where the owner field collapses to two values and names nothing, 77 of 390
    /// bodies still reproduce a slot exactly (slots 0, 5 and 12), so this route names bodies the
    /// stamp cannot.
    ///
    /// # Why it is safe
    ///
    /// Two players can choose identical armour, and the all-zero default appearance is shared by
    /// anyone who has not customised. So a body is named **only when exactly one present slot
    /// matches**; an appearance that fits two slots, or none, leaves the body unnamed. And this
    /// runs LAST, so it never overrides a binding the stamp or the correlation already made --
    /// it fills gaps, exactly as `bind_by_elimination` does.
    ///
    /// Returns how many bodies it named.
    pub fn bind_by_appearance(&mut self, players: &[crate::film::FilmPlayer]) -> usize {
        // The answer key: every PRESENT slot's appearance, in this decode's own terms.
        let key = Self::appearance_key(players);
        if key.is_empty() {
            return 0;
        }
        let already: std::collections::HashSet<u32> =
            self.bindings.iter().filter(|b| b.is_confident()).map(|b| b.net_id).collect();
        let mut out: Vec<PlayerBinding> = Vec::new();
        for tr in self.tracks.iter().filter(|t| t.entity_type == 11) {
            if already.contains(&tr.net_id) {
                continue; // condition 3: a stated name always wins
            }
            let Some(slot) = self.armour_slot(tr, &key) else { continue };
            out.push(PlayerBinding {
                slot,
                net_id: tr.net_id,
                entity_type: tr.entity_type,
                first_tick: tr.first_tick,
                last_tick: tr.last_tick,
                cost: 0.0,
                margin: 0.0,
                samples: 0,
                from_facing: false,
                from_owner_rule: false,
                origin: BindOrigin::Appearance,
            });
        }
        // CONCURRENCY -- the fourth guard, and the one this route was missing.
        //
        // Uniqueness asks "does exactly one roster slot match this body's armour". It never asks
        // "how many BODIES am I about to hang on that one slot". On `*4832BB46*` -- 1,689 bodies
        // where many players wear the same gear -- that put **110 bindings on slot 11, four of
        // them alive at once**: one gamertag on a hundred and ten bodies, which is the same
        // mistake as the `(owner - 1) & 15` wrap and the sixteen-bodies-on-one-slot rule before
        // it.
        //
        // The slot is refused ENTIRELY rather than trimmed to fit, because bodies overlapping in
        // time are evidence the appearance is not identifying one person -- so no subset of them
        // is trustworthy either.
        let mut spans: std::collections::HashMap<u8, Vec<(u32, u32)>> = Default::default();
        for b in &out {
            spans.entry(b.slot).or_default().push((b.first_tick, b.last_tick));
        }
        let refused: std::collections::HashSet<u8> = spans
            .iter()
            .filter(|(_, s)| peak_bodies_alive(s) > MAX_PLAYER_BODIES_AT_ONCE)
            .map(|(slot, _)| *slot)
            .collect();
        out.retain(|b| !refused.contains(&b.slot));
        let named = out.len();
        self.bindings.extend(out);
        self.bindings.sort_by_key(|b| (b.slot, b.first_tick));
        named
    }

    /// #appearance  The header appearance key for [`FilmIndex::armour_slot`]: every PRESENT
    /// slot's appearance, in this decode's terms.
    ///
    /// `p.index < 16` matters: `occupied_slot_mask` drops out-of-range indices and a stale header
    /// can carry them, so without it the route could name a "slot 17" the index does not consider
    /// occupied -- which is how it came to name more slots than the header occupies.
    pub(crate) fn appearance_key(
        players: &[crate::film::FilmPlayer],
    ) -> Vec<(u8, u8, [i8; 3], u8, [i8; 10])> {
        let mut key = Vec::new();
        for p in players.iter().filter(|p| p.present() && p.index < 16) {
            let Some(a) = p.appearance() else { continue };
            let mut slots = [0i8; 10];
            for (i, v) in a.slots.iter().enumerate() {
                slots[i] = *v as i8;
            }
            key.push((
                p.index as u8,
                a.f48,
                [a.color_primary as i8, a.color_secondary as i8, a.f4b as i8],
                a.model_choice,
                slots,
            ));
        }
        key
    }

    /// #appearance  THE slot this body's armour names -- the single decision the armour route
    /// makes, under every constraint it applies.
    ///
    /// > **A NEW IDENTITY ROUTE MUST INHERIT EVERY CONSTRAINT THE EXISTING ONE CARRIES.**
    /// > Uniqueness, occupancy, containment, concurrency -- each was learned from a specific
    /// > failure and **none of them is implied by the others**. This route was written with
    /// > uniqueness alone and then acquired the other three one production failure at a time:
    /// > occupancy (named a slot the header does not occupy, so no gamertag existed),
    /// > containment (a roster row attached to a body whose life began 49,000 ticks earlier),
    /// > and concurrency (110 bindings on one slot with four bodies alive at once -- the same
    /// > mistake as the `(owner - 1) & 15` wrap and the sixteen-bodies-on-one-slot rule). A
    /// > route can satisfy three of the four and still hang a hundred bodies on one gamertag.
    ///
    /// **One implementation, two callers.** `bind_by_appearance` uses it to bind, and
    /// `appearance_gates` uses it to cross-check against the owner stamp. They had drifted: the
    /// gate applied only the uniqueness rule and so reported a contradiction over a name the
    /// route would never have produced. Keeping it in one place is also what stops this becoming
    /// two writers for one fact.
    ///
    /// Three conditions, each earned:
    ///
    /// 0. **The appearance is not the DEFAULT.** An uncustomised appearance is the absence of an
    ///    identity, not an identity, so it names nobody even when exactly one listed player
    ///    shares it. See [`BipedAppearance::is_default`].
    /// 1. **Exactly one** present slot's appearance matches, bit for bit. Two players can pick
    ///    identical armour, so an ambiguous match names nobody.
    /// 2. **The slot is occupied** by the index's own reckoning (`slot_occupied`), not a second
    ///    opinion -- using `present()` alone let this route name a slot the rest of the index
    ///    treats as empty.
    /// 3. **Containment**: the body's life must overlap that slot's per-tick player records. A
    ///    player cannot be driving a body at a tick where they have no record. The stamp route
    ///    always had this; without it, a roster row whose records begin at tick 49,162 was
    ///    attached to a body whose life begins at tick 2.
    pub(crate) fn armour_slot(
        &self,
        tr: &EntityTrack,
        key: &[(u8, u8, [i8; 3], u8, [i8; 10])],
    ) -> Option<u8> {
        let l = tr.look?;
        // Condition 0, and it outranks uniqueness: an appearance that customises NOTHING
        // identifies nobody, however many listed players happen to share it.
        if l.is_default() {
            return None;
        }
        let mut ten = [0i8; 10];
        ten.copy_from_slice(&l.slots[..10]);
        let mut hit: Option<u8> = None;
        let mut n = 0usize;
        for (slot, flags, colors, model, slots) in key {
            if *flags == l.flags && *colors == l.colors && *model == l.model_choice && *slots == ten
            {
                n += 1;
                hit = Some(*slot);
            }
        }
        if n != 1 {
            return None;
        }
        let slot = hit?;
        if !self.slot_occupied(slot) {
            return None;
        }
        if let Some(v) = self.player_aims.get(slot as usize) {
            if let (Some(first), Some(last)) = (v.first(), v.last()) {
                if tr.last_tick < first.0 || tr.first_tick > last.0 {
                    return None;
                }
            }
        }
        Some(slot)
    }

    /// #appearance  NAME A BODY FROM ANOTHER RECORDING'S ROSTER, when this film's cannot.
    ///
    /// A film's player list is who was present when the list was WRITTEN, not everyone who
    /// played, so a clip can be full of bodies its own header never names. On one measured clip
    /// the header names 3 of roughly 18 participants. But these recordings are one group of
    /// friends on one map, so the players the header forgot are usually named in a *different*
    /// film -- and a body's armour identifies them.
    ///
    /// **Armour is very nearly a unique per-player fingerprint.** Measured over this machine's
    /// whole corpus: **171 roster entries collapse to 88 distinct appearances, and exactly ONE of
    /// those 88 maps to more than one gamertag.** So an exact appearance match is strong evidence
    /// of identity -- which is why this is allowed at all.
    ///
    /// # The five conditions, and why each is there
    ///
    /// 1. **Exact bit-match** on flags, three colours, `player_model_choice` and ten slot
    ///    options. Nothing fuzzy.
    /// 2. **Unique gamertag corpus-wide.** An appearance mapping to two or more distinct
    ///    gamertags names nobody, even if it maps to a single roster entry in several files.
    /// 3. **This film's own roster wins.** [`FilmIndex::bind_by_appearance`] runs first and this
    ///    never overrides it, nor the owner stamp, nor the correlation.
    /// 4. **It is labelled.** The result goes in `cross_film_names`, NOT in `bindings`, because a
    ///    binding names a header SLOT and the whole point here is that this header has no slot
    ///    for the player. Nothing can mistake inferred evidence for stated evidence.
    /// 5. **The source is kept**, so a roster line can say which recording the name came from.
    ///
    /// Measured effect on `*53C14E9D*`: **77 stated -> 305 with inference**, with 22 refused as
    /// ambiguous and 63 matching no roster anywhere (genuinely unrecoverable).
    ///
    /// # The key is shared, so this is cheap enough to be automatic
    ///
    /// `build_index_players` runs this for any clip that still has unnamed player bodies, using
    /// [`AppearanceKey::shared`] -- one corpus scan per process, not one per call. Making it
    /// caller-driven was tried first and was worse: nothing called it, so the route silently did
    /// nothing.
    ///
    /// # It depends on what else is on the machine, and that is not a bug
    ///
    /// The key is built from the films present. Someone with fewer recordings gets fewer names
    /// and someone with only this one gets the 77. The roster note must say so, or the same clip
    /// naming differently on two machines looks like a defect.
    pub fn bind_by_appearance_across_films(&mut self, key: &AppearanceKey) -> usize {
        let named: std::collections::HashSet<u32> =
            self.bindings.iter().filter(|b| b.is_confident()).map(|b| b.net_id).collect();
        let mut out: Vec<(u32, String, String)> = Vec::new();
        for tr in self.tracks.iter().filter(|t| t.entity_type == 11) {
            if named.contains(&tr.net_id) {
                continue; // condition 3: a stated name always wins
            }
            let Some(l) = tr.look else { continue };
            let Some((tag, src)) = key.lookup(&l) else { continue };
            out.push((tr.net_id, tag.to_string(), src.to_string()));
        }
        // CONCURRENCY, here too -- and nothing else would have caught it.
        //
        // `one_slot_is_one_player_over_real_films` counts `bindings`, and these names live in
        // `cross_film_names`, so an unguarded cross-film route could hang one gamertag on a
        // hundred bodies with no gate firing at all. Same constraint, same reason, same
        // threshold: one player drives one biped, at most three of their bodies coexisting.
        // Grouped by GAMERTAG rather than slot, because that is the identity being claimed.
        let mut spans: std::collections::HashMap<&str, Vec<(u32, u32)>> = Default::default();
        for (id, tag, _) in &out {
            if let Some(tr) = self.tracks.iter().find(|t| t.net_id == *id) {
                spans.entry(tag.as_str()).or_default().push((tr.first_tick, tr.last_tick));
            }
        }
        let refused: std::collections::HashSet<String> = spans
            .iter()
            .filter(|(_, s)| peak_bodies_alive(s) > MAX_PLAYER_BODIES_AT_ONCE)
            .map(|(tag, _)| (*tag).to_string())
            .collect();
        out.retain(|(_, tag, _)| !refused.contains(tag));
        let n = out.len();
        out.sort_by_key(|(id, _, _)| *id);
        self.cross_film_names = out;
        self.appearance_cross_named = n;
        // Record WHICH gamertags were refused, not just that some were. A refusal and a route
        // nobody called are indistinguishable from the outside -- both are simply a body with no
        // name -- and the production-path gate has to be able to tell them apart.
        self.cross_film_refused = {
            let mut v: Vec<String> = refused.into_iter().collect();
            v.sort();
            v
        };
        n
    }

    /// #catchall  Could this owner group be ONE PLAYER, by the only test the format allows?
    ///
    /// A player drives one biped. Their *tracks* nonetheless pile up, because a corpse and its
    /// ragdoll are entities that linger past the respawn — so the test cannot be "no two bodies
    /// overlap". It is "how many of this group's bodies exist at the same moment, at its worst".
    ///
    /// **The threshold is measured, not chosen** (`one_slot_is_one_player_over_real_films` prints
    /// the table it comes from, one line per film). Over 39 real films the peak for a group that
    /// behaves like a player is **3** — a live body plus two lingering ones — and the overlaps
    /// that produce it are short: on the 16-player test clip slot 3 peaks at 3 with its longest
    /// pair overlapping 11.7 s out of a 7-minute clip. Every value ABOVE 3 in the corpus belongs
    /// to a group that is obviously several people:
    ///
    /// | film | stamp | bodies | alive at once | named by |
    /// |------|-------|--------|---------------|----------|
    /// | `4ED6DDC1_67FF2AFF` | 1 | 16 | **13** | the aim correlation |
    /// | `2BAB2586_6801934F` | 4 | 68 | **7** | the owner stamp |
    /// | `5A017B8B` (condemned) | 1 | 11 | **4** | the owner stamp |
    ///
    /// Thirteen bodies alive together under one gamertag is not a player, and that film's worst
    /// pair overlaps for **353 s**. Refusing those groups costs three films a name they should
    /// never have had; a wrong gamertag on a body is worse than none.
    pub(crate) fn owner_group_is_one_player(&self, owner: u8) -> bool {
        let spans: Vec<(u32, u32)> = self
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner))
            .map(|t| (t.first_tick, t.last_tick))
            .collect();
        peak_bodies_alive(&spans) <= MAX_PLAYER_BODIES_AT_ONCE
    }

    /// Promote the correlation's one reliable answer into a rule for the whole film.
    ///
    /// Evidence for `slot = owner - 1` (a 1-based owner index, 0 meaning nobody): it fits 5 of 7
    /// independent recorder anchors across the corpus, and on a containment oracle over 223 owner
    /// groups in 21 films -- a group's lives must fall inside its slot's per-tick record range --
    /// it scores 30.0% against 21.2% for `owner` and 19.1% for `owner + 1`. That is a strong lead
    /// but not a proof, so it is never applied blind.
    ///
    /// Instead it is CORROBORATED PER FILM. The recorder is a free anchor on every film: their
    /// configuration slot comes from the header, and their owner index comes from the bipeds that
    /// aim-correlate -- which are theirs precisely because their aim is local input, recorded
    /// exactly, rather than replicated. If the anchor agrees with the rule on this film, the rule
    /// is used for every group here; if it disagrees, nothing is promoted and only the correlated
    /// player keeps a name. A film that cannot be checked is left alone.
    fn apply_owner_rule(&mut self, groups: &std::collections::BTreeMap<i16, Vec<usize>>) {
        self.owner_rule_ok = false;
        self.owner_anchor = OwnerAnchor::None;
        // THE ANCHOR. The author record is preferred wherever it exists, because it is a stated
        // fact; the live-aim slot is the fallback and is marked as the inference it is.
        //
        // The fallback exists because an IN-PROGRESS `.temp` recording has no author record at
        // all -- those fields are written when a film is finalised -- so every `.temp` failed the
        // gate and lost every name, every armour selection and every colour at once, all three
        // hanging off the slot the gate refused to supply. Measured on a live recording: 16 owner
        // groups, 16 slots carrying per-tick records, 3 named header slots, and 0 bodies bound.
        let (rec_slot, anchor) = match self.recorder_slot {
            Some(s) => (s, OwnerAnchor::AuthorRecord),
            None => match self.live_aim_slot() {
                Some(s) => (s, OwnerAnchor::LiveAim),
                None => return,
            },
        };
        // The recorder's own group: the best-correlated one.
        let Some(best) = self
            .bindings
            .iter()
            .filter(|b| b.from_facing && b.cost < 0.35)
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
        else {
            return;
        };
        let Some(rec_owner) = self
            .tracks
            .iter()
            .find(|t| t.net_id == best.net_id)
            .and_then(|t| t.owner_raw)
        else {
            return;
        };
        if self.owner_to_slot(rec_owner) != Some(rec_slot) {
            return; // the anchor contradicts the rule on this film: promote nothing
        }
        // A DERIVED anchor buys a stricter test, not a looser one. The author-record path keeps
        // exactly the acceptance it has always had -- changing it would silently rename people in
        // every existing clip -- while the live-aim path additionally requires every promoted
        // slot to be one the header really occupies, and requires the mapping to be injective.
        let strict = anchor.is_derived();
        if strict {
            let mut seen = [false; 16];
            for key in groups.keys() {
                if *key < 0 {
                    continue; // a biped with no owner stamp at all
                }
                let Some(slot) = self.owner_to_slot(*key as u8) else { continue };
                if std::mem::replace(&mut seen[slot as usize], true) {
                    return; // two owners want the same slot: the rule does not hold here
                }
            }
        }
        self.owner_rule_ok = true;
        self.owner_anchor = anchor;
        let mut out: Vec<PlayerBinding> = Vec::new();
        for (key, tis) in groups {
            if *key < 0 {
                continue; // a biped with no owner stamp: the rule says nothing about it
            }
            let owner = *key as u8;
            let Some(slot) = self.owner_to_slot(owner) else { continue };
            if strict {
                // The name, the armour and the colours all come from the header record, so a slot
                // the header does not occupy has nothing to give. The recorder's OWN body can
                // land here -- they joined after the header was written -- and that is correct:
                // the anchor only needs the recorder to be identifiable, not named.
                if !self.slot_occupied(slot) {
                    continue;
                }
            } else if slot > 15 || self.player_aims.get(slot as usize).map_or(true, |v| v.is_empty()) {
                continue; // no such player in this recording
            }
            for &ti in tis {
                let tr = &self.tracks[ti];
                out.push(PlayerBinding {
                    slot,
                    net_id: tr.net_id,
                    entity_type: tr.entity_type,
                    first_tick: tr.first_tick,
                    last_tick: tr.last_tick,
                    cost: best.cost,
                    margin: best.margin,
                    samples: best.samples,
                    from_facing: best.from_facing,
                    from_owner_rule: true,
                    origin: BindOrigin::OwnerStamp,
                });
            }
        }
        if !out.is_empty() {
            out.sort_by_key(|b| (b.slot, b.first_tick));
            self.bindings = out;
        }
    }

    /// How many distinct owner groups the bipeds fall into -- i.e. how many players actually had
    /// a body in this recording, regardless of whether any of them could be named.
    pub fn owner_group_count(&self) -> usize {
        let mut v: std::collections::HashSet<u8> = Default::default();
        for t in &self.tracks {
            if t.entity_type == 11 {
                if let Some(o) = t.owner_raw {
                    v.insert(o);
                }
            }
        }
        v.len()
    }

    /// Every distinct owner group, sorted. Each one is a single player's bodies across every
    /// respawn, whether or not that player can be named.
    ///
    /// This is the cast list a viewer can actually watch, and it is NOT the film header's roster:
    /// the header records who was present when it was WRITTEN, so on one measured clip it names 3
    /// players while 13 slots carry per-tick records and 14 owner groups had a body on the map.
    /// Following a body needs no gamertag — an entity id is enough to aim a camera — so the UI
    /// builds its rows from here.
    pub fn owner_groups(&self) -> Vec<u8> {
        let mut v: Vec<u8> = self
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11)
            .filter_map(|t| t.owner_raw)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// How many bodies (lives) this owner group had.
    pub fn owner_lives(&self, owner: u8) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner))
            .count()
    }

    /// The tick range over which this owner group had any body at all.
    pub fn owner_tick_range(&self, owner: u8) -> Option<(u32, u32)> {
        let mut r: Option<(u32, u32)> = None;
        for t in self.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner)) {
            r = Some(match r {
                None => (t.first_tick, t.last_tick),
                Some((a, b)) => (a.min(t.first_tick), b.max(t.last_tick)),
            });
        }
        r
    }

    /// The body this owner group was driving at `tick`, if any.
    ///
    /// A player gets a NEW network entity every time they respawn, so "follow this player" has to
    /// look up the body that was theirs *at that tick*. Where two of a group's bodies overlap
    /// (a corpse lingering past the respawn), the one that most recently appeared wins — that is
    /// the one being driven.
    pub fn entity_for_owner(&self, owner: u8, tick: u32) -> Option<&EntityTrack> {
        self.tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner))
            .filter(|t| tick >= t.first_tick && tick <= t.last_tick)
            .max_by_key(|t| t.first_tick)
    }

    /// The configuration slot this owner group is confidently bound to, if any. `None` means the
    /// recording holds no name for this player — not that HMS failed.
    ///
    /// #catchall  A GROUP-level answer takes only GROUP-level evidence, and the group has to
    /// agree with itself.
    ///
    /// Both halves matter. [`FilmIndex::bind_by_elimination`] names a SINGLE BODY, and owner
    /// group 0 is a CATCH-ALL on some films — so if that per-body name counted here, the one
    /// Elite body elimination identifies would have put its player's gamertag on the whole
    /// six-body row, which is the six-bodies-one-name mistake in a different place. Elimination
    /// binds are therefore skipped: a group is named by the stamp rule or the aim correlation,
    /// which bind every body of a group together, or not at all. And where two of a group's
    /// bodies disagree about the slot, the group names nobody.
    pub fn slot_of_owner(&self, owner: u8) -> Option<u8> {
        let mut want: Option<u8> = None;
        for t in self.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner)) {
            let Some(b) = self.bindings.iter().find(|b| {
                b.net_id == t.net_id
                    && b.is_confident()
                    && b.origin != BindOrigin::SpeciesElimination
            }) else {
                continue;
            };
            match want {
                None => want = Some(b.slot),
                Some(s) if s == b.slot => {}
                Some(_) => return None, // two players' bodies share this stamp: it names nobody
            }
        }
        want
    }

    /// #catchall  Every body this reader attributed to a player slot, in tick order.
    ///
    /// Unlike [`FilmIndex::owner_of_slot`] this does not go through the owner GROUP, so it still
    /// answers for a player whose body was identified on its own -- which is the whole point of
    /// the elimination bind, because such a body sits in a group that names nobody.
    pub fn bodies_of_slot(&self, slot: u8) -> Vec<&EntityTrack> {
        let mut v: Vec<&EntityTrack> = self
            .bindings
            .iter()
            .filter(|b| b.slot == slot && b.is_confident())
            .filter_map(|b| self.tracks.iter().find(|t| t.net_id == b.net_id))
            .collect();
        v.sort_by_key(|t| t.first_tick);
        v.dedup_by_key(|t| t.net_id);
        v
    }

    /// #catchall  WHICH EVIDENCE named this body, if anything did. The UI uses it to say how a
    /// name was arrived at wherever it was not read straight off the body's own owner stamp.
    pub fn origin_of_entity(&self, net_id: u32) -> Option<BindOrigin> {
        self.bindings.iter().find(|b| b.net_id == net_id && b.is_confident()).map(|b| b.origin)
    }

    /// The owner group a configuration slot belongs to, if it has one.
    pub fn owner_of_slot(&self, slot: u8) -> Option<u8> {
        self.owner_groups().into_iter().find(|&o| self.slot_of_owner(o) == Some(slot))
    }

    /// The scoreboard state of one statborg slot at `tick`, folded forward from its updates.
    ///
    /// Returns `(round_score, carried_score, rounds_won, final, updates seen)`. `None` when this
    /// slot never reported -- which is the common case, and does NOT mean zero: it means the
    /// recording was never sent a score update for it.
    pub fn statborg_at(&self, slot: u8, tick: u32) -> Option<(i16, i16, i16, bool, usize)> {
        let v = self.statborg.get(slot as usize)?;
        if v.is_empty() {
            return None;
        }
        let (mut r, mut t, mut w, mut f, mut n) = (0i16, 0i16, 0i16, false, 0usize);
        for (at, s) in v.iter() {
            if *at > tick {
                break;
            }
            n += 1;
            if let Some(x) = s.round_score {
                r = x;
            }
            if let Some(x) = s.total_score {
                t = x;
            }
            if let Some(x) = s.rounds_won {
                w = x;
            }
            if let Some(x) = s.is_final {
                f = x;
            }
        }
        (n > 0).then_some((r, t, w, f, n))
    }

    /// The score the game would DISPLAY for a statborg slot at `tick`.
    ///
    /// The engine's own rule: the carried-in total plus the live round's score while the round is
    /// running, and the carried-in total alone once `final` is set.
    pub fn score_at(&self, slot: u8, tick: u32) -> Option<i32> {
        let (r, t, _w, f, _n) = self.statborg_at(slot, tick)?;
        Some(if f { t as i32 } else { t as i32 + r as i32 })
    }

    /// Statborg slots that ever reported, as `(slot, is a team)`.
    pub fn statborg_slots(&self) -> Vec<(u8, bool)> {
        self.statborg
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(i, _)| (i as u8, i as u8 >= STATBORG_PLAYERS))
            .collect()
    }

    /// The player slot whose AIM IS LIVE: the one the recording player was driving.
    ///
    /// **Only one slot per film has a live aim.** A player's own input is captured locally and
    /// written exactly; every other slot's aim is replicated, and in practice the field is
    /// zero-filled for them — on 14- and 16-player clips every non-recorder slot carries the
    /// *same* `yaw_q` on all ~25,000 of its samples with the pitch exactly level. So a slot whose
    /// quantised yaw actually moves is the recorder, however many samples the others have.
    ///
    /// Returns `None` unless EXACTLY ONE slot qualifies: two live aims would mean the predicate
    /// is not measuring what it is believed to measure, and an ambiguous anchor is no anchor.
    pub fn live_aim_slot(&self) -> Option<u8> {
        let mut found: Option<u8> = None;
        for (slot, v) in self.player_aims.iter().enumerate().take(16) {
            if v.len() < 30 {
                continue;
            }
            let first = v[0].1.yaw_q;
            // A handful of stray changes is noise, not a player looking around; require a real
            // share of the samples to differ.
            let changed = v.iter().filter(|(_, a)| a.yaw_q != first).count();
            if changed * 20 > v.len() {
                if found.is_some() {
                    return None; // ambiguous
                }
                found = Some(slot as u8);
            }
        }
        found
    }

    /// Is this a header slot a player actually occupies?
    ///
    /// Falls back to "the slot carries per-tick records" when the caller supplied no occupancy
    /// mask, which is the weaker proxy this code used before the mask existed.
    fn slot_occupied(&self, slot: u8) -> bool {
        if slot > 15 {
            return false;
        }
        if self.occupied_slots != 0 {
            return self.occupied_slots & (1 << slot) != 0;
        }
        self.player_aims.get(slot as usize).map_or(false, |v| !v.is_empty())
    }

    pub fn first_player_tick(&self) -> Option<u32> {
        self.tracks
            .iter()
            .filter(|t| t.entity_type == 11)
            .map(|t| t.first_tick)
            .min()
    }

    /// How many player bipeds exist at `tick`. Players die and respawn, so this rises and falls
    /// through a clip; a drop is a death, not a decode failure.
    pub fn players_at(&self, tick: u32) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.entity_type == 11 && tick >= t.first_tick && tick <= t.last_tick)
            .count()
    }

    /// Ticks where the player population CHANGES, for markers on a scrub bar. Capped so a long
    /// clip cannot produce an unbounded list.
    pub fn player_population_changes(&self, max: usize) -> Vec<(u32, usize)> {
        let mut edges: Vec<u32> = Vec::new();
        for t in self.tracks.iter().filter(|t| t.entity_type == 11) {
            edges.push(t.first_tick);
            edges.push(t.last_tick.saturating_add(1));
        }
        edges.sort_unstable();
        edges.dedup();
        let mut out: Vec<(u32, usize)> = Vec::new();
        let mut last = usize::MAX;
        for e in edges {
            let n = self.players_at(e);
            if n != last {
                out.push((e, n));
                last = n;
            }
            if out.len() >= max {
                break;
            }
        }
        out
    }

    pub fn report(&self) -> String {
        let mut by_type = [0usize; 32];
        let mut moving = 0usize;
        for t in &self.tracks {
            by_type[(t.entity_type & 31) as usize] += 1;
            if t.moves() {
                moving += 1;
            }
        }
        let mut types: Vec<(usize, usize)> =
            by_type.iter().copied().enumerate().filter(|(_, n)| *n > 0).collect();
        types.sort_by(|a, b| b.1.cmp(&a.1));
        format!(
            "{} ticks ({:.1} s), {} records decoded; {} entities ({} moving), {} of {} entity messages gave a position; built in {:.3} s{}\n  types: {}",
            self.ticks,
            self.seconds(),
            self.records_decoded,
            self.tracks.len(),
            moving,
            self.positions_decoded,
            self.entity_messages,
            self.elapsed.as_secs_f64(),
            match self.first_player_tick() {
                Some(t) => format!("; first player appears at {}:{:04.1}", (t / 60) / 60, (t as f32 / 60.0) % 60.0),
                None => "; NO player entities in this film".into(),
            },
            types.iter().map(|(t, n)| format!("{t}:{n}")).collect::<Vec<_>>().join(" ")
        )
    }
}

/// Circular difference between two angles, folded into -pi..pi.
fn ang_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// Decode a whole film into entity tracks. Sub-second even on the longest film measured, so this
/// is meant to run once on open (on a worker thread) with no progressive streaming.
/// Width of a `TagReference('weap')` in a unit's derived block, and the meaning of 0.
///
/// **Measured, with a clean margin.** The free mask oracle cannot see this field at all — widths
/// 4 through 9 all score identically (325/325/325 on one film), because nothing after it is
/// exercised in the messages that carry it. So the VALUE had to decide, against an oracle that
/// is obvious once stated: a unit's held weapon should be a **hand** weapon.
///
/// Resolving the value as an index into the map's `'weap'` group list, over the non-zero values:
///
/// | width | non-zero values that are hand weapons |
/// |-------|----------------------------------------|
/// | 4 bits | 0% — `scorpion_anti_infantry`, `machinegun_turret`, `falcon_machinegun` |
/// | **5 bits** | **100%** — `shotgun`, `plasma_repeater`, `dmr`, `energy_sword`, `focus_rifle`, `needle_rifle`, `plasma_pistol` |
/// | 6 bits | 0% — `sabre_missile`, `monitor_beam`, `flag`, `seraph_quad_cannon` |
/// | 7-9 bits | 0% — mostly out of range |
///
/// and the same at 5 bits on a second film (`needle_rifle`, `energy_sword`). 100% against 0% at
/// both neighbouring widths is as sharp as this corpus gives.
///
/// **0 means NO REFERENCE**, not "the first weapon": it is ~87-100% of all values, which no real
/// weapon distribution looks like, and taking it literally would put a ghost's dual cannon in
/// every Spartan's hands.
///
/// # Coverage is low, and that bounds what this can be used for
///
/// Measured over 12 films, the share of bodies that ever report a weapon at all:
/// **0%, 4%, 0%, 1%, 1%, 4%, 0%, 3%, 0%, 0%, 5%, 8%** — zero on half of them, including the
/// 16-player "Assassination." clip. The field reports on CHANGE (a pickup or a swap), not on
/// spawn, so most bodies live and die without it ever being sent. A player's STARTING weapon is
/// not here; where it is has not been established.
///
/// So HMS draws the weapon for the handful of bodies the recording actually names one for, and
/// the rest are drawn empty-handed. That is the recording's limit, not a decode failure, and the
/// caption says which bodies are which.
pub const WEAPREF_BITS: u32 = 5;

pub fn build_index(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
) -> FilmIndex {
    build_index_for(bytes, data, variant, None)
}

/// As `build_index`, with the recording player's configuration slot from the film header. That
/// slot is the anchor that lets `slot = owner - 1` be corroborated on this particular film, so
/// passing it is what turns one named player into all of them.
/// The film header's OCCUPIED SLOT mask, one bit per configuration slot — what
/// [`build_index_with`] wants. Lives here rather than on `Film` so the index's own callers can
/// build it without reaching into another module's API surface.
pub fn occupied_slot_mask(players: &[crate::film::FilmPlayer]) -> u16 {
    let mut m = 0u16;
    for p in players.iter().filter(|p| p.present()) {
        if p.index < 16 {
            m |= 1 << p.index;
        }
    }
    m
}

/// #catchall  The film header's ELITE mask: a bit per configuration slot whose own
/// `player_model_choice` (`s_player_configuration` +0x4C, 0 Spartan / 1 Elite) says Elite.
///
/// This is the header's INDEPENDENT statement about a player, and it is what lets a body the
/// owner stamp could not name be matched to a slot by elimination -- see
/// [`FilmIndex::bind_by_elimination`]. A slot that is not occupied, or whose appearance block
/// fails its own checks, contributes no bit, which reads as Spartan; the elimination rule only
/// ever uses the mask for slots it has already established are occupied.
pub fn elite_slot_mask(players: &[crate::film::FilmPlayer]) -> u16 {
    let mut m = 0u16;
    for p in players.iter().filter(|p| p.present()) {
        if p.index < 16 && p.appearance().map_or(false, |a| a.model_choice == 1) {
            m |= 1 << p.index;
        }
    }
    m
}

/// As [`build_index_with`], taking the header's player array and deriving BOTH masks from it.
///
/// Prefer this wherever the header is in hand: the occupancy mask alone leaves
/// [`FilmIndex::bind_by_elimination`] without the species statement it needs, so a body the
/// owner stamp could not name stays unnamed even where the header settles it.
pub fn build_index_players(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
    recorder_slot: Option<u8>,
    players: &[crate::film::FilmPlayer],
) -> FilmIndex {
    let mut idx = build_index_full(
        bytes,
        data,
        variant,
        recorder_slot,
        occupied_slot_mask(players),
        elite_slot_mask(players),
    );
    // #appearance  The armour route, applied last so it only fills gaps the stamp left.
    idx.appearance_named = idx.bind_by_appearance(players);
    // #appearance  And the CROSS-FILM fallback, for the bodies this film's own roster cannot
    // name. Two things keep it cheap: it only runs when a clip still HAS unnamed player bodies
    // (so a clip the stamp handles pays nothing), and the corpus key is built once per process
    // (`AppearanceKey::shared`) rather than per call.
    let unnamed = idx
        .tracks
        .iter()
        .filter(|t| t.entity_type == 11 && t.look.is_some() && idx.slot_of_entity(t.net_id).is_none())
        .count();
    if unnamed > 0 {
        let key = AppearanceKey::shared();
        if !key.is_empty() {
            idx.bind_by_appearance_across_films(key);
        }
    }
    idx
}

pub fn build_index_for(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
    recorder_slot: Option<u8>,
) -> FilmIndex {
    // 0 = "the caller did not say which header slots are occupied"; the derived-anchor path then
    // falls back to its weaker proxy. Callers that have the header should use `build_index_with`.
    build_index_with(bytes, data, variant, recorder_slot, 0)
}

/// As [`build_index_for`], with the film header's OCCUPIED SLOT mask (bit per slot).
///
/// The mask only matters on a film with no author record — an in-progress `.temp` — where the
/// recorder is identified from the live-aim slot instead. That derived anchor promotes a name
/// only onto a slot the header really occupies, and the index cannot know which those are from
/// the stream alone, so the caller supplies it.
pub fn build_index_with(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
    recorder_slot: Option<u8>,
    occupied_slots: u16,
) -> FilmIndex {
    // 0 = "the caller did not say which of those slots are Elites", which leaves the elimination
    // rule with no species statement and so names nobody extra. Callers with the header should
    // use `build_index_players`.
    build_index_full(bytes, data, variant, recorder_slot, occupied_slots, 0)
}

/// As [`build_index_with`], with the header's ELITE mask as well ([`elite_slot_mask`]).
pub fn build_index_full(
    bytes: &[u8],
    data: &crate::film::FilmData,
    variant: &crate::film::EmbeddedMapVariant,
    recorder_slot: Option<u8>,
    occupied_slots: u16,
    elite_slots: u16,
) -> FilmIndex {
    let t0 = std::time::Instant::now();
    let (base, _score) = solve_params(bytes, data);
    // The per-axis position widths and the world mapping come from the engine's own formula
    // applied to the map variant's world bounds -- NOT from a fit. See `frame_from_bounds`.
    let (frame, w) = frame_from_bounds(variant.world_bounds);
    let params = DecodeParams {
        cv3_axis_bits: w,
        cv3_total_bits: w.iter().sum(),
        angle_always: true,
        update_prefix_bits: 1,
        ..base
    };
    // The `'weap'` reference width. NOT solved per film: the mask oracle is blind to this field
    // (every width scores identically), so it is a measured constant -- see `WEAPREF_BITS`.
    let weapref_bits = WEAPREF_BITS;
    let mut idx = FilmIndex {
        params,
        frame,
        records: data.records.len(),
        recorder_slot,
        occupied_slots,
        elite_slots,
        weapref_bits,
        ..Default::default()
    };
    if variant.is_empty() {
        // No world box to quantise against, so positions cannot be placed.
        idx.elapsed = t0.elapsed();
        return idx;
    }
    let mut by_id: std::collections::HashMap<u32, usize> = Default::default();
    // A statborg's SLOT is stated once, in its creation message; its updates only carry the net
    // id, so the mapping has to be remembered.
    let mut statborg_slot: std::collections::HashMap<u32, u8> = Default::default();
    // Each biped's CREATION payload, by track index, kept so the owner stamp can be walked once
    // the film's `'eqip'` width is known. See `creation_biped` / `solve_eqip_bits`.
    let mut biped_creations: std::collections::HashMap<usize, Vec<u8>> = Default::default();
    // The four weapon-slot references and the selected slot index, carried across a body's
    // messages: the stream states each separately and only when it CHANGES.
    let mut weapon_state: std::collections::HashMap<u32, ([Option<u32>; 4], i8)> = Default::default();
    let mut base_tick: Option<u32> = None;
    for (i, rec) in data.records.iter().enumerate() {
        let Ok(t) = decode_record(rec.body(bytes), &params) else { continue };
        idx.records_decoded += 1;
        let b = *base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
        let tick = t.tick.wrapping_sub(b);
        for a in &t.aims {
            if idx.player_aims.len() < 16 {
                idx.player_aims.resize(16, Vec::new());
            }
            idx.player_aims[(a.slot & 15) as usize].push((tick, *a));
        }
        // #player-unit  The engine's own assignment, folded per slot in tick order.
        idx.player_action_blocks += t.player_actions;
        for act in &t.actions {
            if idx.player_objects.len() < 16 {
                idx.player_objects.resize(16, Vec::new());
            }
            idx.player_objects[(act.slot & 15) as usize].push((tick, *act));
        }
        // #film-action  The EVENT channel. List A is a different namespace, so only list B's
        // entries are folded; `commands_a` is the split.
        for c in t.commands.iter().skip(t.commands_a).filter(|c| c.kind == 1) {
            idx.event_commands += 1;
            if let Some(ev) = crate::film_action::fold(tick, &c.payload) {
                idx.events.push(ev);
            }
        }
        for &(_ty, net) in &t.destroyed {
            if let Some(&i) = by_id.get(&net) {
                idx.tracks[i].destroyed_tick.get_or_insert(tick);
            }
        }
        for e in &t.entities {
            idx.entity_messages += 1;
            // The OWNER stamp is read from a biped's creation payload before anything else,
            // because this reader has no walker for a type-11 creation body -- the payload decode
            // below returns None for it -- and the owner is the one field worth keeping from it.
            // It is also usually the entity's FIRST message, so the track has to be opened here.
            // #film-action  The object DEFINITION, from the creation payload's first 14 bits.
            // Every type whose creation goes through the shared reader states it, and the engine
            // REJECTS a creation whose reference is null -- so a valid creation always names one.
            if !e.is_update {
                if let Some(w) = crate::film_action::creation_def_wire(e) {
                    let slot = *by_id.entry(e.net_id).or_insert_with(|| {
                        idx.tracks.push(EntityTrack {
                            net_id: e.net_id,
                            entity_type: e.entity_type,
                            first_tick: tick,
                            last_tick: tick,
                            ..Default::default()
                        });
                        idx.tracks.len() - 1
                    });
                    idx.tracks[slot].def_wire.get_or_insert(w);
                }
            }
            // The OWNER STAMP is not at a fixed offset (see `creation_biped`), and the walk that
            // finds it needs the film's `'eqip'` reference width, which is solved from the whole
            // set of creations. So the payloads are kept here and the owners resolved once the
            // pass is over.
            if e.entity_type == 11 && !e.is_update {
                let slot = *by_id.entry(e.net_id).or_insert_with(|| {
                    idx.tracks.push(EntityTrack {
                        net_id: e.net_id,
                        entity_type: e.entity_type,
                        first_tick: tick,
                        last_tick: tick,
                        ..Default::default()
                    });
                    idx.tracks.len() - 1
                });
                biped_creations.entry(slot).or_insert_with(|| e.payload.clone());
            }
            let Some(st) = decode_entity_payload(e, &params) else { continue };
            let slot = *by_id.entry(e.net_id).or_insert_with(|| {
                idx.tracks.push(EntityTrack {
                    net_id: e.net_id,
                    entity_type: e.entity_type,
                    first_tick: tick,
                    last_tick: tick,
                    ..Default::default()
                });
                idx.tracks.len() - 1
            });
            let tr = &mut idx.tracks[slot];
            tr.last_tick = tick;

            if let Some(raw) = st.position_raw {
                let p = frame.world(raw);
                if p.iter().all(|v| v.is_finite()) {
                    tr.positions.push((tick, p));
                    idx.positions_decoded += 1;
                }
            }
            if let Some(or) = st.orientation_raw {
                let (f, u) = orientation_world(or);
                tr.orientations.push((tick, f, u));
            }
            // #film-attach  Base field 12, when the message stated it. Recorded on CHANGE, so a
            // repeat of the same parent is not stored twice; a cleared attachment IS stored,
            // because "no longer riding" is a change the camera has to see.
            // The condition is that the walk REALLY READ field 12 -- `rebuilt_mask` is what the
            // walk consumed, not what the header claimed -- so a message that stops at field 13
            // still keeps the attachment it had already decoded, and one that stopped before
            // field 12 contributes nothing.
            if st.rebuilt_mask & (1 << 12) != 0 {
                let last = tr.attachments.last().map(|(_, a)| *a);
                if last != Some(st.attach) {
                    tr.attachments.push((tick, st.attach));
                }
            }
            if let Some(q) = st.facing_yaw_q {
                tr.facings.push((tick, yaw_from_q13(q)));
            }
            // The weapon in this unit's hands -- derived field 0x2A. Taken only from a walk that
            // passed BOTH oracles (payload consumed to within 7 bits AND the rebuilt mask equal
            // to the message's own), so a wrong `weapref_bits` yields nothing rather than a wrong
            // weapon.
            // #film-stats  The SCOREBOARD. Type 5 is `game-engine-statborg`; its creation names
            // the slot and its dirty block carries the round score, the carried-in total and the
            // rounds won. Kept only when both free oracles pass, so a wrong walk yields nothing
            // rather than a wrong score.
            if e.entity_type == 5 {
                idx.statborg_messages += 1;
                if let Some(sb) = decode_statborg(e, &params) {
                    if sb.trustworthy(e.dirty_mask) {
                        idx.statborg_decoded += 1;
                        if let Some(i) = sb.index {
                            statborg_slot.insert(e.net_id, i);
                        }
                        if let Some(&slot) = statborg_slot.get(&e.net_id) {
                            if idx.statborg.len() < STATBORG_SLOTS as usize {
                                idx.statborg.resize(STATBORG_SLOTS as usize, Vec::new());
                            }
                            let s = StatborgSample {
                                round_score: sb.round_score,
                                total_score: sb.total_score,
                                rounds_won: sb.rounds_won,
                                is_final: sb.is_final,
                            };
                            if s != StatborgSample::default() {
                                idx.statborg[slot as usize].push((tick, s));
                            }
                        }
                    }
                }
            }
            // #weapon-held  WHICH of a unit's four weapons is in its HANDS.
            //
            // A unit carries up to four weapons (derived fields 0x2A..0x2D, one `'weap'`
            // reference each) and a SELECTION (`sub_1803C94B8`, written by field 0x29 and again
            // by every one of 0x2A..0x2D, so the last one read wins). The engine's own
            // `unit_get_primary_weapon` calls `GetPlayerWeaponBySlot(unit, *(char *)(unit +
            // 0x34A))`, and `unit + 0x34A` is that selection struct's first 2-bit sub-field --
            // so the weapon being held is the SELECTED slot's reference, not slot 0's.
            //
            // Both halves arrive only on CHANGE, and in separate messages, so the four refs and
            // the selected index are carried across the body's life and the held weapon is
            // recomputed whenever either moves.
            if e.entity_type == 11 && e.is_update {
                if let Some(u) = crate::film_anim::unit::decode(e, &params, weapref_bits) {
                    if u.trustworthy(e.dirty_mask) {
                        // A body that has never stated a selection is holding its first weapon:
                        // that is what the engine readies on spawn, and it is what HMS drew
                        // before the selection was decoded at all.
                        let st = weapon_state.entry(e.net_id).or_insert(([None; 4], 0));
                        for (k, s) in u.weapon_slots.iter().enumerate() {
                            if let Some(v) = s {
                                st.0[k] = Some(*v);
                            }
                        }
                        if let Some(i) = u.weapon_sel_ab[0] {
                            st.1 = i;
                        }
                        let held = usize::try_from(st.1).ok().and_then(|i| st.0.get(i).copied()).flatten();
                        if let Some(w) = held {
                            if tr.weapons.last().map_or(true, |(_, prev)| *prev != w) {
                                tr.weapons.push((tick, w));
                                idx.weapons_decoded += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    idx.first_tick = base_tick.unwrap_or(0);
    idx.ticks = idx.records as u32;
    // THE OWNER STAMP, walked. The `'eqip'` width is the only unknown left in a biped creation,
    // and the walk runs to the end of the payload, so the payloads themselves pin it.
    {
        let pls: Vec<Vec<u8>> = biped_creations.values().cloned().collect();
        let (eqip, closed) = solve_eqip_bits(&pls, &params);
        idx.eqip_bits = eqip;
        idx.creations_closed = closed;
        for (ti, pl) in &biped_creations {
            // Reaching the owner field is the condition; closing the payload is not. See
            // `creation_biped`.
            let Some(c) = creation_biped(pl, &params, eqip) else { continue };
            idx.tracks[*ti].owner_raw = Some((c.owner + 1) as u8);
            idx.tracks[*ti].appearance = c.appearance;
            idx.tracks[*ti].look = c.look;
        }
    }
    // Identify every non-Forge entity by the Forge spawn point it appeared on. Skipped for type
    // 13 because those ARE the Forge objects (and a pathological film carries 370k of them).
    // Fold the per-track constants ONCE, now that every track is complete.
    for tr in idx.tracks.iter_mut() {
        tr.moves_cached = Some(tr.path_length() > 0.5);
    }
    idx.variant_objects = variant.present_objects().map(|o| (o.quota_index, o.item, o.position)).collect();
    {
        let objs: Vec<(&crate::film::VariantObject, [f32; 3])> =
            variant.present_objects().map(|o| (o, o.position)).collect();
        for tr in idx.tracks.iter_mut() {
            if tr.entity_type == 13 {
                continue;
            }
            let Some(&(_, first)) = tr.positions.first() else { continue };
            let mut best: Option<(f32, u16, u8)> = None;
            for (o, p) in &objs {
                let d = (p[0] - first[0]).powi(2) + (p[1] - first[1]).powi(2) + (p[2] - first[2]).powi(2);
                if best.map_or(true, |b| d < b.0) {
                    best = Some((d, o.quota_index, o.item));
                }
            }
            tr.spawned_on = best.map(|(d2, q, i)| (q, i, d2.sqrt()));
        }
    }
    idx.bind_players();
    idx.elapsed = t0.elapsed();
    idx
}

/// Decode every record of a film and report what was understood. `bytes` is the whole file (as
/// `film::open` read it) and `data` its framed `flmd` chunk.
pub fn scan(bytes: &[u8], data: &crate::film::FilmData) -> ScanStats {
    let (params, score) = solve_params(bytes, data);
    let t0 = std::time::Instant::now();
    let mut st = ScanStats::default();
    st.params = params;
    st.params_score = score;
    let mut seen: std::collections::HashSet<u16> = Default::default();
    // The expected tick is derived from the record INDEX, not from the previous decoded record:
    // one record == one tick, so a gap of undecoded records must not break the check.
    let mut base: Option<u32> = None;
    for (i, rec) in data.records.iter().enumerate() {
        st.records += 1;
        let body = rec.body(bytes);
        st.payload_bytes += body.len();
        match decode_record(body, &params) {
            Ok(t) => {
                if t.leftover_bits > 7 {
                    st.misaligned += 1;
                    continue;
                }
                st.decoded += 1;
                let b = *base.get_or_insert(t.tick.wrapping_sub(i as u32));
                if t.tick == b.wrapping_add(i as u32) {
                    st.tick_continuous += 1;
                }
                st.commands += t.commands.len();
                st.entity_messages += t.entities.len();
                st.player_actions += t.player_actions;
                st.players += t.players;
                for e in &t.entities {
                    st.entity_types[(e.entity_type & 31) as usize] += 1;
                    seen.insert(e.entity_index());
                }
            }
            Err(e) => {
                *st.stops.entry(e).or_insert(0) += 1;
            }
        }
    }
    st.entities_seen = seen.len();
    st.first_tick = base.unwrap_or(0);
    st.elapsed = t0.elapsed();
    st
}

/// Just walk the records without decoding them — the framing-only cost, which is the floor on
/// any indexing pass.
pub fn frame_only(bytes: &[u8], data: &crate::film::FilmData) -> (usize, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    let mut acc = 0u64;
    for rec in &data.records {
        let b = rec.body(bytes);
        // touch the bytes so the loop cannot be optimised away
        acc = acc.wrapping_add(b.first().copied().unwrap_or(0) as u64);
        n += 1;
    }
    std::hint::black_box(acc);
    (n, t0.elapsed())
}

/// #film-temp  The owner->slot bridge must prefer the PROVEN anchor and must not be loosened by
/// the derived one. Both halves are asserted because getting either wrong renames people:
/// dropping the author record would make every finished film unverified, and letting the derived
/// path promote onto an empty header slot would attach a name that does not exist.
#[cfg(test)]
mod anchor_tests {
    use super::*;

    /// Build an index with one owner group per slot and a live aim on `live`.
    fn fixture(recorder: Option<u8>, live: u8, owners: &[u8], occupied: u16) -> FilmIndex {
        let mut idx = FilmIndex { ticks: 1000, recorder_slot: recorder, occupied_slots: occupied, ..Default::default() };
        idx.player_aims = vec![Vec::new(); 16];
        for slot in 0..16u8 {
            for t in 0..100u32 {
                // Only `live` actually moves its yaw; every other slot is zero-filled, which is
                // what a real film does for a player whose input was not captured locally.
                let yaw_q = if slot == live { (t * 37) as u16 & 0x1FFF } else { 1234 };
                idx.player_aims[slot as usize].push((t, PlayerAim { slot, yaw_q, ..Default::default() }));
            }
        }
        for (i, &o) in owners.iter().enumerate() {
            idx.tracks.push(EntityTrack {
                net_id: 0x1000 + i as u32,
                entity_type: 11,
                first_tick: 0,
                last_tick: 999,
                owner_raw: Some(o),
                moves_cached: Some(false),
                ..Default::default()
            });
        }
        idx
    }

    #[test]
    fn the_live_aim_slot_is_the_one_whose_yaw_moves() {
        let idx = fixture(None, 7, &[1, 2, 3], 0b0000_0111);
        assert_eq!(idx.live_aim_slot(), Some(7));
    }

    /// Two moving aims is not an anchor, it is a sign the predicate is measuring the wrong thing.
    #[test]
    fn two_live_aims_is_no_anchor() {
        let mut idx = fixture(None, 7, &[1], 0b1);
        for t in 0..100u32 {
            idx.player_aims[3][t as usize].1.yaw_q = (t * 11) as u16;
        }
        assert_eq!(idx.live_aim_slot(), None);
    }

    /// The derived anchor must not promote a name onto a header slot nobody occupies — the name,
    /// the armour and the colours all come from that record, so there would be nothing there.
    #[test]
    fn the_derived_anchor_only_promotes_onto_occupied_slots() {
        // Owners 1,2,3 map to slots 0,1,2; the header occupies only 0 and 1.
        let mut idx = fixture(None, 0, &[1, 2, 3], 0b0000_0011);
        idx.occupied_slots = 0b0000_0011;
        // Force the correlation to pick owner 1 (slot 0) as the recorder's group.
        idx.bindings = vec![PlayerBinding {
            slot: 0,
            net_id: 0x1000,
            entity_type: 11,
            cost: 0.05,
            margin: 1.0,
            samples: 100,
            from_facing: true,
            ..Default::default()
        }];
        let mut groups: std::collections::BTreeMap<i16, Vec<usize>> = Default::default();
        for (i, t) in idx.tracks.iter().enumerate() {
            groups.entry(t.owner_raw.unwrap() as i16).or_default().push(i);
        }
        idx.apply_owner_rule(&groups);
        assert_eq!(idx.owner_anchor, OwnerAnchor::LiveAim, "no author record: the live aim is the anchor");
        let slots: std::collections::BTreeSet<u8> = idx.bindings.iter().map(|b| b.slot).collect();
        assert_eq!(slots, [0u8, 1].into_iter().collect(), "slot 2 is unoccupied and must not be named");
    }

    /// A film WITH an author record must use it, not the derived anchor.
    #[test]
    fn the_author_record_wins_over_the_live_aim() {
        let mut idx = fixture(Some(0), 0, &[1, 2], 0b0000_0011);
        idx.bindings = vec![PlayerBinding {
            slot: 0,
            net_id: 0x1000,
            entity_type: 11,
            cost: 0.05,
            margin: 1.0,
            samples: 100,
            from_facing: true,
            ..Default::default()
        }];
        let mut groups: std::collections::BTreeMap<i16, Vec<usize>> = Default::default();
        for (i, t) in idx.tracks.iter().enumerate() {
            groups.entry(t.owner_raw.unwrap() as i16).or_default().push(i);
        }
        idx.apply_owner_rule(&groups);
        assert_eq!(idx.owner_anchor, OwnerAnchor::AuthorRecord);
        assert!(!idx.owner_anchor.is_derived(), "a film with an author record is never unverified");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film;

    #[test]
    fn the_bit_reader_is_msb_first() {
        let mut r = BitReader::new(&[0b1010_0110, 0b1111_0000]);
        assert_eq!(r.bits(1).unwrap(), 1);
        assert_eq!(r.bits(2).unwrap(), 0b01);
        assert_eq!(r.bits(5).unwrap(), 0b00110);
        assert_eq!(r.bits(4).unwrap(), 0b1111);
        assert_eq!(r.bits(4).unwrap(), 0b0000);
        assert!(matches!(r.bits(1), Err(DecodeStop::Overrun)));
        // a 64-bit read, and a span crossing many bytes
        let mut r = BitReader::new(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89]);
        assert_eq!(r.bits(64).unwrap(), 0xDEAD_BEEF_0123_4567);
        assert_eq!(r.bits(8).unwrap(), 0x89);
        // reading n then 8-n must equal reading the byte
        for split in 1..8u32 {
            let mut a = BitReader::new(&[0b1101_0011]);
            let hi = a.bits(split).unwrap();
            let lo = a.bits(8 - split).unwrap();
            assert_eq!((hi << (8 - split)) | lo, 0b1101_0011);
        }
    }

    #[test]
    fn read_bytes_is_the_same_as_eight_bit_reads() {
        let buf: Vec<u8> = (0u8..32).collect();
        let mut a = BitReader::new(&buf);
        a.bits(3).unwrap();
        let bytes = a.read_bytes(10).unwrap();
        let mut b = BitReader::new(&buf);
        b.bits(3).unwrap();
        let one: Vec<u8> = (0..10).map(|_| b.bits(8).unwrap() as u8).collect();
        assert_eq!(bytes, one);
        assert_eq!(a.pos(), b.pos());
    }

    #[test]
    fn a_truncated_record_stops_rather_than_panicking() {
        for p in [DecodeParams::default(), DecodeParams { host_authority: true, cv3_total_bits: 66, tagref_bits: 14, cv3_axis_bits: [22, 22, 22], angle_always: true, update_prefix_bits: 0 }] {
            for n in 0..64 {
                let _ = decode_record(&vec![0u8; n], &p);
            }
            for pattern in [0x00u8, 0xFF, 0xAA, 0x55, 0x0F, 0xF0] {
                for n in 0..200 {
                    let _ = decode_record(&vec![pattern; n], &p);
                }
            }
        }
    }

    #[test]
    fn an_entity_message_unpacks_its_header() {
        // 5-bit type, 16 + 11 bits of id, 64-bit mask, then payload.
        let mut bits: Vec<bool> = Vec::new();
        let mut push = |v: u64, n: u32| {
            for i in (0..n).rev() {
                bits.push((v >> i) & 1 != 0);
            }
        };
        push(11, 5);
        push(0xE975, 16);
        push(0x001, 11);
        push(0x0009_C600_0270_0A07, 64);
        for b in [0xAAu8, 0xBB, 0xCC] {
            push(b as u64, 8);
        }
        let mut bytes = vec![0u8; (bits.len() + 7) / 8];
        for (i, b) in bits.iter().enumerate() {
            if *b {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        let e = entity_message(&bytes, false).expect("parses");
        assert_eq!(e.entity_type, 11);
        assert_eq!(e.net_id, 0xE975_0001);
        assert_eq!(e.entity_index(), 1);
        assert_eq!(e.dirty_mask, 0x0009_C600_0270_0A07);
        assert_eq!(&e.payload[..3], &[0xAA, 0xBB, 0xCC]);
    }

    fn sample_films() -> Vec<std::path::PathBuf> {
        // ONE implementation, in `film.rs`. There were six copies of this walk and they
        // disagreed about which films exist -- and all of them missed the entire Linux install,
        // because a Proton prefix is seven levels down and uses a lowercase `users`.
        crate::film::spec_corpus()
    }

    /// THE MEASUREMENT the design depends on: how fast can a film be decoded, and how much of it
    /// does this reader understand? Prints per film and in total; asserts only the things that
    /// are genuinely required (no panic, framing exact, and every record this reader claims to
    /// have decoded really did land on a byte boundary with a continuous tick).
    #[test]
    fn decode_throughput_over_every_real_film() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films on this machine -- throughput not measured");
            return;
        }
        let (mut tot_rec, mut tot_dec, mut tot_bytes, mut tot_time) = (0usize, 0usize, 0usize, std::time::Duration::ZERO);
        let mut frame_time = std::time::Duration::ZERO;
        let mut frame_rec = 0usize;
        let mut rows: Vec<String> = Vec::new();
        for p in &films {
            let read0 = std::time::Instant::now();
            let Ok(f) = film::open(p) else { continue };
            let read_time = read0.elapsed();
            if !f.data.framing_exact {
                panic!("{}: framing is not exact", p.display());
            }
            let (n, ft) = frame_only(&f.bytes, &f.data);
            frame_rec += n;
            frame_time += ft;
            let st = scan(&f.bytes, &f.data);
            assert_eq!(st.records, f.data.records.len());
            assert_eq!(st.records as i32, f.header.tick_count, "{}: record count", p.display());
            // every record counted as decoded must also be tick-continuous: if the envelope were
            // wrong, the tick would not advance by exactly one.
            assert_eq!(
                st.decoded, st.tick_continuous,
                "{}: {} records decoded but only {} had a continuous tick -- the envelope is wrong",
                p.display(), st.decoded, st.tick_continuous
            );
            rows.push(format!(
                "{:<46} {:>7} rec {:>6.1} MB  read {:>6.3}s  frame {:>7.4}s  decode {:>6.3}s  {:>8.0} rec/s = {:>6.0}x RT  decoded {:>5.1}%",
                p.file_name().unwrap_or_default().to_string_lossy(),
                st.records,
                st.payload_bytes as f64 / 1.0e6,
                read_time.as_secs_f64(),
                ft.as_secs_f64(),
                st.elapsed.as_secs_f64(),
                st.records_per_second(),
                st.realtime_multiple(),
                st.decode_fraction() * 100.0
            ));
            tot_rec += st.records;
            tot_dec += st.decoded;
            tot_bytes += st.payload_bytes;
            tot_time += st.elapsed;
        }
        for r in &rows {
            eprintln!("film_decode: {r}");
        }
        let secs = tot_time.as_secs_f64();
        eprintln!(
            "film_decode: TOTAL {} films, {tot_rec} records ({:.1} MB) in {secs:.3} s = {:.0} records/s = {:.0}x real time; \
             envelope decoded {tot_dec} ({:.1}%)",
            rows.len(),
            tot_bytes as f64 / 1.0e6,
            tot_rec as f64 / secs.max(1e-9),
            (tot_rec as f64 / secs.max(1e-9)) / 60.0,
            100.0 * tot_dec as f64 / tot_rec.max(1) as f64
        );
        eprintln!(
            "film_decode: framing alone: {frame_rec} records in {:.4} s = {:.0} records/s",
            frame_time.as_secs_f64(),
            frame_rec as f64 / frame_time.as_secs_f64().max(1e-9)
        );
        assert!(tot_rec > 1000, "too few records to mean anything");
    }

    /// THE ORACLE for the entity payload walk, reported rather than asserted: a message whose
    /// dirty mask uses only known fields (bits 0..10) must consume its payload to within 7 bits.
    /// Today this sits far below 1.0 — the walk is wrong somewhere — and this test exists to say
    /// so out loud and to make a fix visible the moment it lands. It asserts only that the
    /// machinery runs and that the ENVELOPE around it is still perfect.
    #[test]
    fn entity_payload_oracle() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- entity payload oracle not measured");
            return;
        }
        let (mut films_seen, mut tot_msgs, mut tot_ok) = (0usize, 0usize, 0usize);
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let (base, score) = solve_params(&f.bytes, &f.data);
            assert_eq!(score, 1.0, "{}: the record envelope no longer decodes at 100%", p.display());
            // every fully-known-mask message in the first part of the film
            let mut msgs: Vec<EntityMessage> = Vec::new();
            for rec in f.data.records.iter().take(400) {
                let Ok(t) = decode_record(rec.body(&f.bytes), &base) else { continue };
                for e in t.entities {
                    if e.dirty_mask != 0 && e.dirty_mask & !KNOWN_FIELD_MASK == 0 {
                        msgs.push(e);
                    }
                }
            }
            if msgs.is_empty() {
                continue;
            }
            films_seen += 1;
            // best over the two unknown widths
            let mut best = 0usize;
            for tag in 8..=16u32 {
                for total in 3..=60u32 {
                    let pp = DecodeParams {
                        tagref_bits: tag,
                        cv3_axis_bits: [total / 3, total / 3, total - 2 * (total / 3)],
                        cv3_total_bits: total,
                        ..base
                    };
                    let ok = msgs.iter().filter(|e| entity_payload_leftover(e, &pp).map_or(false, |l| l <= 7)).count();
                    best = best.max(ok);
                }
            }
            tot_msgs += msgs.len();
            tot_ok += best;
            if films_seen <= 6 {
                eprintln!(
                    "film_decode: {:<46} payload oracle {}/{} ({:.0}%)",
                    p.file_name().unwrap_or_default().to_string_lossy(),
                    best, msgs.len(), 100.0 * best as f64 / msgs.len() as f64
                );
            }
        }
        if tot_msgs == 0 {
            eprintln!("film_decode: no fully-known-mask entity messages found");
            return;
        }
        let pct = 100.0 * tot_ok as f64 / tot_msgs as f64;
        eprintln!(
            "film_decode: ENTITY PAYLOAD WALK: {tot_ok}/{tot_msgs} = {pct:.1}% consumed exactly, over {films_seen} film(s). \
             This is the number that must reach ~100% before positions can be trusted; the record \
             ENVELOPE around it is already at 100%."
        );
    }

    /// Does the position that comes out of the walk land on a real object? Measured on the
    /// subset of messages that PASS the payload oracle, against the film's own embedded map
    /// variant (type-13 entities are the Forge objects, whose true positions HMS already has).
    /// REPORTED, not asserted: the walk is not yet universal, so this is evidence, not a gate.
    #[test]
    fn position_fit_against_the_embedded_variant() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- position fit not measured");
            return;
        }
        let mut shown = 0usize;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let (base, score) = solve_params(&f.bytes, &f.data);
            if score < 1.0 {
                continue;
            }
            let Some((pp, frame, oracle)) = solve_position_frame(&f.bytes, &f.data, &f.header.map_variant, base)
            else { continue };
            shown += 1;
            if shown > 8 {
                break;
            }
            eprintln!(
                "film_decode: {:<46} axes {:?} oracle {:.0}%  POSITION QUALITY {:.1}%  (scale {:?})",
                p.file_name().unwrap_or_default().to_string_lossy(),
                pp.cv3_axis_bits, oracle * 100.0, frame.quality * 100.0,
                frame.scale.map(|v| (v * 1000.0).round() / 1000.0)
            );
            for k in 0..3 {
                assert!(frame.scale[k].is_finite() && frame.offset[k].is_finite());
            }
        }
        if shown == 0 {
            eprintln!("film_decode: no film produced a position fit");
        }
    }

    /// THE ENGINE-FORMULA HYPOTHESIS, tested directly: derive the per-axis widths and the world
    /// mapping from the film's OWN embedded map variant bounds with the engine's formula, decode
    /// the type-13 entities, and measure how many land on a real object position. Static Forge
    /// objects cannot move, so their decoded position must equal the authored one.
    #[test]
    fn engine_formula_positions() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- engine-formula positions not measured");
            return;
        }
        let mut shown = 0usize;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            let mv = &f.header.map_variant;
            if mv.is_empty() {
                continue;
            }
            let (base, score) = solve_params(&f.bytes, &f.data);
            if score < 1.0 {
                continue;
            }
            let (frame, w) = frame_from_bounds(mv.world_bounds);
            let pp = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, ..base };
            let truth: Vec<[f32; 3]> = mv.present_objects().map(|o| o.position).collect();
            let (mut n, mut hit, mut exact_payload) = (0usize, 0usize, 0usize);
            let (mut ori_n, mut ori_up, mut ori_fwd) = (0usize, 0usize, 0usize);
            let (mut imp_n, mut imp_ok, mut exp_n, mut exp_ok) = (0usize, 0usize, 0usize, 0usize);
            let mut imp_ang: Vec<f64> = Vec::new();
            let mut imp_upz = 0usize;
            for rec in f.data.records.iter().take(400) {
                let Ok(t) = decode_record(rec.body(&f.bytes), &pp) else { continue };
                for e in &t.entities {
                    if e.entity_type != 13 || e.dirty_mask & 1 == 0 {
                        continue;
                    }
                    if entity_payload_leftover(e, &pp).map_or(false, |l| l <= 7) {
                        exact_payload += 1;
                    }
                    let Some(st) = decode_entity_payload(e, &pp) else { continue };
                    let Some(raw) = st.position_raw else { continue };
                    n += 1;
                    let wpos = frame.world(raw);
                    // Position gives the CORRESPONDENCE: the matched object is the ground truth
                    // for this entity's orientation too.
                    if let Some(o) = mv
                        .present_objects()
                        .find(|o| (0..3).all(|k| (o.position[k] - wpos[k]).abs() <= 0.25))
                    {
                        hit += 1;
                        if let Some(or) = st.orientation_raw {
                            let (fwd, up) = orientation_world(or);
                            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                            ori_n += 1;
                            let implicit = or.0.is_none();
                            if implicit {
                                imp_n += 1;
                                // what IS the true forward of an implicit-axis object?
                                let ang = (o.forward[1] as f64).atan2(o.forward[0] as f64);
                                imp_ang.push(ang);
                                if o.up[2] > 0.999 { imp_upz += 1 }
                            } else { exp_n += 1 }
                            if dot(up, o.up) > 0.99 {
                                ori_up += 1;
                            }
                            if dot(fwd, o.forward) > 0.99 {
                                ori_fwd += 1;
                                if implicit { imp_ok += 1 } else { exp_ok += 1 }
                            }
                        }
                    }
                }
            }
            if n == 0 {
                continue;
            }
            shown += 1;
            if shown > 8 {
                break;
            }
            eprintln!(
                "film_decode: {:<46} W {:?} -> {hit}/{n} pos ({:.1}%)  up {}/{}  fwd {}/{} [implicit-axis {imp_ok}/{imp_n}, explicit {exp_ok}/{exp_n}]  ({exact_payload} exact)",
                p.file_name().unwrap_or_default().to_string_lossy(),
                w, 100.0 * hit as f64 / n as f64, ori_up, ori_n, ori_fwd, ori_n
            );
            if !imp_ang.is_empty() {
                let mut a = imp_ang.clone();
                a.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let deg = |v: f64| (v.to_degrees() * 10.0).round() / 10.0;
                eprintln!(
                    "film_decode:   implicit-axis objects: {} of them, up=+Z on {imp_upz}; true forward yaw deg min {} q25 {} med {} q75 {} max {}",
                    a.len(), deg(a[0]), deg(a[a.len() / 4]), deg(a[a.len() / 2]), deg(a[3 * a.len() / 4]), deg(a[a.len() - 1])
                );
            }
        }
    }

    /// THE PLAYBACK TEST: build the whole index and check that it describes MOTION — entities
    /// that move, over the film's duration, inside the map. Reports per film and asserts the
    /// invariants a usable index must have.
    #[test]
    fn index_describes_motion() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- the playback index was not built");
            return;
        }
        let mut shown = 0usize;
        let mut total_build = std::time::Duration::ZERO;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            total_build += idx.elapsed;
            if idx.tracks.is_empty() {
                continue;
            }
            // Invariants of a usable index.
            assert_eq!(idx.ticks as i32, f.header.tick_count, "{}: tick count", p.display());
            for t in &idx.tracks {
                assert!(t.first_tick <= t.last_tick, "{}: track time runs backwards", p.display());
                assert!(t.positions.windows(2).all(|w| w[0].0 <= w[1].0), "{}: positions out of order", p.display());
                for (_, q) in &t.positions {
                    assert!(q.iter().all(|v| v.is_finite()), "{}: non-finite position", p.display());
                }
            }
            // Sampling must work at both ends and in the middle.
            for tk in [0u32, idx.ticks / 2, idx.ticks.saturating_sub(1)] {
                let _ = idx.sample(tk);
            }
            let movers: Vec<&EntityTrack> = idx.moving().collect();
            let longest = movers.iter().map(|t| t.path_length()).fold(0.0f32, f32::max);
            let mut mover_types: std::collections::HashMap<u8, usize> = Default::default();
            for t in &movers {
                *mover_types.entry(t.entity_type).or_insert(0) += 1;
            }
            let mut mt: Vec<(u8, usize)> = mover_types.into_iter().collect();
            mt.sort_by(|a, b| b.1.cmp(&a.1));
            shown += 1;
            if shown <= 8 {
                eprintln!(
                    "film_decode: {:<46} {}",
                    p.file_name().unwrap_or_default().to_string_lossy(),
                    idx.report().replace('\n', " |")
                );
                eprintln!(
                    "film_decode:   movers {} (types {:?}), longest path {:.0} wu, build {:.3} s",
                    movers.len(),
                    mt.iter().take(5).collect::<Vec<_>>(),
                    longest,
                    idx.elapsed.as_secs_f64()
                );
            }
        }
        eprintln!("film_decode: all indexes built in {:.3} s total", total_build.as_secs_f64());
    }

    /// What do the kind-3 UPDATE messages look like? Run with `--ignored`.
    #[test]
    #[ignore]
    fn diagnose_update_messages() {
        let mut done = 0usize;
        for p in sample_films() {
            if done >= 2 { break }
            let Ok(f) = film::open(&p) else { continue };
            if f.header.map_variant.is_empty() { continue }
            done += 1;
            let (base0, _) = solve_params(&f.bytes, &f.data);
            let (_, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let pr = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, ..base0 };
            let mut tm: std::collections::HashMap<(u8, u64), usize> = Default::default();
            let mut lens: std::collections::HashMap<(u8, usize), usize> = Default::default();
            let (mut n, mut bit0, mut decoded, mut gotpos) = (0usize, 0usize, 0usize, 0usize);
            for rec in f.data.records.iter().take(600) {
                let Ok(t) = decode_record(rec.body(&f.bytes), &pr) else { continue };
                for e in &t.entities {
                    if !e.is_update { continue }
                    n += 1;
                    *tm.entry((e.entity_type, e.dirty_mask)).or_insert(0) += 1;
                    *lens.entry((e.entity_type, e.payload.len() + 12)).or_insert(0) += 1;
                    if e.dirty_mask & 1 != 0 { bit0 += 1 }
                    if let Some(st) = decode_entity_payload(e, &pr) {
                        decoded += 1;
                        if st.position_raw.is_some() { gotpos += 1 }
                    }
                }
            }
            let mut v: Vec<_> = tm.into_iter().collect(); v.sort_by(|a, b| b.1.cmp(&a.1));
            let mut lv: Vec<_> = lens.into_iter().collect(); lv.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("=== {}", p.file_name().unwrap_or_default().to_string_lossy());
            eprintln!("  updates {n}, mask bit0 set {bit0}, payload-decoded {decoded}, position {gotpos}");
            eprintln!("  top (type,mask): {:?}", v.iter().take(8).map(|((t, m), c)| format!("t{t} {m:#x} x{c}")).collect::<Vec<_>>());
            eprintln!("  top (type,bytes): {:?}", lv.iter().take(8).map(|((t, l), c)| format!("t{t} {l}B x{c}")).collect::<Vec<_>>());
            // sweep the update prefix: score by exact payload consumption on fully-known masks
            let mut ups: Vec<EntityMessage> = Vec::new();
            for rec in f.data.records.iter().take(600) {
                let Ok(t) = decode_record(rec.body(&f.bytes), &pr) else { continue };
                for e in t.entities {
                    if e.is_update && e.dirty_mask != 0 && e.dirty_mask & !KNOWN_FIELD_MASK == 0 { ups.push(e) }
                }
            }
            let mut curve: Vec<(u32, usize)> = Vec::new();
            for pre in 0..=40u32 {
                let pp = DecodeParams { update_prefix_bits: pre, ..pr };
                let ok = ups.iter().filter(|e| entity_payload_leftover(e, &pp).map_or(false, |l| l <= 7)).count();
                if ok > 0 { curve.push((pre, ok)) }
            }
            curve.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("  update-prefix sweep over {} msgs: {:?}", ups.len(), curve.iter().take(6).collect::<Vec<_>>());
            // THE CONTINUITY ORACLE, which is far sharper than payload length: a moving entity's
            // position must change smoothly at 60 Hz. A wrong prefix scatters it across the map.
            for pre in 0..=16u32 {
                let pp = DecodeParams { update_prefix_bits: pre, ..pr };
                let mut last: std::collections::HashMap<u32, (u32, [f32; 3])> = Default::default();
                let (mut steps, mut small) = (0usize, 0usize);
                let (frame2, _) = frame_from_bounds(f.header.map_variant.world_bounds);
                for (i, rec) in f.data.records.iter().take(2000).enumerate() {
                    let Ok(t) = decode_record(rec.body(&f.bytes), &pp) else { continue };
                    for e in &t.entities {
                        if !e.is_update || e.dirty_mask & 1 == 0 { continue }
                        let Some(st) = decode_entity_payload(e, &pp) else { continue };
                        let Some(raw) = st.position_raw else { continue };
                        let w = frame2.world(raw);
                        if let Some((pt, pw)) = last.insert(e.net_id, (i as u32, w)) {
                            let dt = (i as u32).saturating_sub(pt).max(1) as f32;
                            let d = ((w[0] - pw[0]).powi(2) + (w[1] - pw[1]).powi(2) + (w[2] - pw[2]).powi(2)).sqrt();
                            steps += 1;
                            // 0.5 wu per tick = 30 wu/s, far above a sprinting Spartan
                            if d / dt < 0.5 { small += 1 }
                        }
                    }
                }
                if steps > 20 {
                    eprintln!("    prefix {:>2}: {}/{} steps continuous ({:.0}%)", pre, small, steps, 100.0 * small as f64 / steps as f64);
                }
            }
            // per TYPE: does each entity type have its own consistent prefix?
            let mut types: Vec<u8> = ups.iter().map(|e| e.entity_type).collect();
            types.sort_unstable(); types.dedup();
            for ty in types {
                let sub: Vec<&EntityMessage> = ups.iter().filter(|e| e.entity_type == ty).collect();
                if sub.len() < 10 { continue }
                let mut bestp = (0u32, 0usize);
                for pre in 0..=40u32 {
                    let pp = DecodeParams { update_prefix_bits: pre, ..pr };
                    let ok = sub.iter().filter(|e| entity_payload_leftover(e, &pp).map_or(false, |l| l <= 7)).count();
                    if ok > bestp.1 { bestp = (pre, ok) }
                }
                eprintln!("    type {:>2}: {} msgs, best prefix {} -> {}/{} ({:.0}%)",
                    ty, sub.len(), bestp.0, bestp.1, sub.len(), 100.0 * bestp.1 as f64 / sub.len() as f64);
            }
        }
    }

    /// How fast do players actually move, in world units per second? Sets the locomotion
    /// thresholds from data instead of guesswork. Run with `--ignored`.
    #[test]
    #[ignore]
    fn diagnose_player_speeds() {
        let mut done = 0usize;
        let mut all: Vec<f32> = Vec::new();
        for p in sample_films() {
            if done >= 3 { break }
            let Ok(f) = film::open(&p) else { continue };
            if f.header.map_variant.is_empty() { continue }
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.tracks.is_empty() { continue }
            done += 1;
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                for w in t.positions.windows(2) {
                    let dt = (w[1].0 - w[0].0) as f32 / 60.0;
                    if dt <= 0.0 || dt > 0.5 { continue }
                    let d = ((w[1].1[0] - w[0].1[0]).powi(2) + (w[1].1[1] - w[0].1[1]).powi(2) + (w[1].1[2] - w[0].1[2]).powi(2)).sqrt();
                    all.push(d / dt);
                }
            }
        }
        if all.is_empty() { eprintln!("film_decode: no player speeds"); return }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f32| all[((all.len() - 1) as f32 * f) as usize];
        eprintln!(
            "film_decode: player speed wu/s over {} samples: p10 {:.2} p25 {:.2} med {:.2} p75 {:.2} p90 {:.2} p99 {:.2} max {:.2}",
            all.len(), q(0.10), q(0.25), q(0.50), q(0.75), q(0.90), q(0.99), all[all.len() - 1]
        );
        let frac = |lo: f32, hi: f32| 100.0 * all.iter().filter(|v| **v >= lo && **v < hi).count() as f32 / all.len() as f32;
        eprintln!(
            "film_decode:   <0.3 {:.0}%  0.3-1.0 {:.0}%  1.0-2.0 {:.0}%  2.0-3.5 {:.0}%  >3.5 {:.0}%",
            frac(0.0, 0.3), frac(0.3, 1.0), frac(1.0, 2.0), frac(2.0, 3.5), frac(3.5, 1e9)
        );
    }

    /// WHERE IS THE PER-TICK MOTION? Census of command kinds per list over a whole film, plus
    /// how the kind-2 entity messages are distributed in time. Run with `--ignored`.
    #[test]
    #[ignore]
    fn diagnose_where_motion_lives() {
        let mut done = 0usize;
        for p in sample_films() {
            if done >= 2 { break }
            let Ok(f) = film::open(&p) else { continue };
            if f.header.map_variant.is_empty() { continue }
            done += 1;
            let (base0, _) = solve_params(&f.bytes, &f.data);
            let (_, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let pr = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, ..base0 };
            let mut kinds_a: std::collections::HashMap<u8, usize> = Default::default();
            let mut kinds_b: std::collections::HashMap<u8, usize> = Default::default();
            let (mut ticks, mut players, mut actions, mut ents) = (0usize, 0usize, 0usize, 0usize);
            // kind-2 messages in the first 120 ticks vs the rest
            let (mut early, mut late) = (0usize, 0usize);
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &pr) else { continue };
                ticks += 1;
                players += t.players;
                actions += t.player_actions;
                ents += t.entities.len();
                if i < 120 { early += t.entities.len() } else { late += t.entities.len() }
                for (j, c) in t.commands.iter().enumerate() {
                    if j < t.commands_a { *kinds_a.entry(c.kind).or_insert(0) += 1 }
                    else { *kinds_b.entry(c.kind).or_insert(0) += 1 }
                }
            }
            let mut ka: Vec<_> = kinds_a.into_iter().collect(); ka.sort_by(|a, b| b.1.cmp(&a.1));
            let mut kb: Vec<_> = kinds_b.into_iter().collect(); kb.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("=== {} ({ticks} ticks)", p.file_name().unwrap_or_default().to_string_lossy());
            eprintln!("  list A kinds: {:?}", ka.iter().take(8).map(|(k, n)| format!("k{k} x{n} ({:.2}/tick)", *n as f64 / ticks as f64)).collect::<Vec<_>>());
            eprintln!("  list B kinds: {:?}", kb.iter().take(8).map(|(k, n)| format!("k{k} x{n} ({:.2}/tick)", *n as f64 / ticks as f64)).collect::<Vec<_>>());
            eprintln!("  per tick: players {:.2}, actions {:.2}, kind-2 entities {:.3} (first 120 ticks {early}, rest {late})",
                players as f64 / ticks as f64, actions as f64 / ticks as f64, ents as f64 / ticks as f64);
        }
    }

    /// Diagnostic for the entity payload walk: the entity-type census, the mask histogram, how
    /// far the walk gets, and the algebraic solve for the position width. Run with `--ignored`.
    #[test]
    #[ignore]
    fn diagnose_entity_payloads() {
        let mut done = 0usize;
        for p in sample_films() {
            if done >= 3 { break }
            let Ok(f) = film::open(&p) else { continue };
            if f.header.map_variant.is_empty() { continue }
            done += 1;
            let (base, score) = solve_params(&f.bytes, &f.data);
            eprintln!("=== {} score {score}", p.file_name().unwrap_or_default().to_string_lossy());
            let mut masks: std::collections::HashMap<(u8, u64), usize> = Default::default();
            let mut stopped: std::collections::HashMap<u32, usize> = Default::default();
            let (mut n13, mut pos13) = (0usize, 0usize);
            let mut msgs: Vec<EntityMessage> = Vec::new();
            for rec in f.data.records.iter().take(400) {
                let Ok(t) = decode_record(rec.body(&f.bytes), &base) else { continue };
                for e in &t.entities {
                    *masks.entry((e.entity_type, e.dirty_mask)).or_insert(0) += 1;
                    if e.dirty_mask == 0x5 && matches!(e.entity_type, 11 | 12 | 13 | 15 | 16 | 17 | 18) {
                        msgs.push(e.clone());
                    }
                    if e.entity_type != 13 { continue }
                    n13 += 1;
                    if let Some(st) = decode_entity_payload(e, &base) {
                        if st.position_raw.is_some() { pos13 += 1 }
                        if let Some(fl) = st.stopped_at_unknown_field { *stopped.entry(fl).or_insert(0) += 1 }
                    }
                }
            }
            let mut mv: Vec<_> = masks.into_iter().collect();
            mv.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("  top (type,mask): {:?}", mv.iter().take(8).map(|((t, m), n)| format!("t{t} {m:#x} x{n}")).collect::<Vec<_>>());
            eprintln!("  type13: {n13} seen, {pos13} reached a position; walk stopped at {stopped:?}");
            // ALGEBRA for mask {0,2}: SUM_W = payload_bits - after_creation - 21 - cost2 - pad
            let mut cand: std::collections::HashMap<u32, usize> = Default::default();
            for e in &msgs {
                let mut r = BitReader::new(&e.payload);
                if creation(&mut r, e.entity_type, &base).is_err() { continue }
                let avail = e.payload.len() * 8 - r.pos();
                for cost2 in [9usize, 28] {
                    for pad in 0..8usize {
                        if avail >= 21 + cost2 + pad {
                            let w = avail - 21 - cost2 - pad;
                            if (3..=78).contains(&w) { *cand.entry(w as u32).or_insert(0) += 1; }
                        }
                    }
                }
            }
            let mut cv: Vec<_> = cand.into_iter().collect();
            cv.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("  mask-0x5 messages: {}; algebraic SUM_W candidates: {:?}", msgs.len(), cv.iter().take(8).collect::<Vec<_>>());
            let mut bestsweep = (0u32, 0u32, 0usize);
            for tag in 0..=20u32 {
                for total in 3..=78u32 {
                    let pp = DecodeParams {
                        tagref_bits: tag,
                        cv3_axis_bits: [total / 3, total / 3, total - 2 * (total / 3)],
                        cv3_total_bits: total,
                        ..base
                    };
                    let ok = msgs.iter().filter(|e| entity_payload_leftover(e, &pp).map_or(false, |l| l <= 7)).count();
                    if ok > bestsweep.2 { bestsweep = (tag, total, ok); }
                }
            }
            eprintln!("  walk sweep best: tagref={} SUM_W={} -> {}/{} exact", bestsweep.0, bestsweep.1, bestsweep.2, msgs.len());
        }
    }

    /// Is the per-player record's flagged 32-bit field (+0x40) the handle of that player's own
    /// unit? If it is, the player/entity link is a STATED fact and the aim correlation is only a
    /// fallback. This test reports the hit rate rather than asserting one, so it is a probe.
    #[test]
    #[ignore = "probe: prints what the unexplained player fields correlate with"]
    fn probe_player_record_unit_handle() {
        let films = sample_films();
        for p in films.iter().take(6) {
            let Ok(f) = film::open(p) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let ids: std::collections::HashSet<u32> = idx.tracks.iter().map(|t| t.net_id).collect();
            let bipeds: std::collections::HashSet<u32> =
                idx.tracks.iter().filter(|t| t.entity_type == 11).map(|t| t.net_id).collect();
            let low: std::collections::HashSet<u16> = ids.iter().map(|v| *v as u16).collect();
            let biped_low: std::collections::HashSet<u16> = bipeds.iter().map(|v| *v as u16).collect();
            // Every way of cutting 48 bits that could hold a 32-bit handle or a 16-bit index.
            let cuts: [(&str, fn(u64) -> u32); 6] = [
                ("hi32", |v| (v >> 16) as u32),
                ("lo32", |v| (v & 0xFFFF_FFFF) as u32),
                ("hi16", |v| (v >> 32) as u32),
                ("mid16", |v| ((v >> 16) & 0xFFFF) as u32),
                ("lo16", |v| (v & 0xFFFF) as u32),
                ("swap", |v| (((v >> 32) as u32) | (((v >> 16) & 0xFFFF) as u32) << 16)),
            ];
            let mut hits = [0usize; 6];
            let mut hits_b = [0usize; 6];
            let mut n = 0usize;
            let mut distinct: std::collections::HashMap<u8, std::collections::HashSet<u64>> =
                Default::default();
            let mut sample: Vec<(u8, u32, u64)> = Vec::new();
            for slot in 0..idx.player_aims.len() {
                for (t, a) in &idx.player_aims[slot] {
                    let Some(v) = a.f08 else { continue };
                    n += 1;
                    distinct.entry(slot as u8).or_default().insert(v);
                    if sample.len() < 8 && slot == 0 {
                        sample.push((slot as u8, *t, v));
                    }
                    for (k, (_, cut)) in cuts.iter().enumerate() {
                        let c = cut(v);
                        if ids.contains(&c) || low.contains(&(c as u16)) {
                            hits[k] += 1;
                        }
                        if bipeds.contains(&c) || biped_low.contains(&(c as u16)) {
                            hits_b[k] += 1;
                        }
                    }
                }
            }
            if n == 0 {
                continue;
            }
            let mut per_slot: Vec<(u8, usize)> = distinct.iter().map(|(k, v)| (*k, v.len())).collect();
            per_slot.sort();
            eprintln!("probe: {}", p.file_name().unwrap_or_default().to_string_lossy());
            eprintln!("  +0x08 present {n}; distinct per slot {per_slot:?}");
            for (k, (name, _)) in cuts.iter().enumerate() {
                eprintln!(
                    "    {name:<6} any-entity {:>7} ({:>5.1}%)  biped {:>7} ({:>5.1}%)",
                    hits[k],
                    hits[k] as f64 / n as f64 * 100.0,
                    hits_b[k],
                    hits_b[k] as f64 / n as f64 * 100.0
                );
            }
            for (s, t, v) in &sample {
                eprintln!("    slot {s} tick {t:>6} = 0x{v:012X}");
            }
        }
    }

    /// The biped body-yaw field, checked against the sharpest oracle available: the aim yaw out
    /// of the same tick's per-player record. A player's body turns onto their aim, so on the
    /// right field this error is small; on a wrong field it is a random angle (median ~1.57 rad).
    ///
    /// Scored on the BEST-matching player slot per biped, because which player drives which
    /// biped is not stated -- but the point stands either way: a wrong field cannot match ANY
    /// slot.
    #[test]
    fn biped_facing_tracks_the_player_aim() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- biped facing oracle skipped");
            return;
        }
        let mut worst: Option<(f32, String)> = None;
        let mut tested = 0;
        for path in films.iter().take(8) {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let good: Vec<&EntityTrack> =
                idx.tracks.iter().filter(|t| t.entity_type == 11 && t.facings.len() >= 300).collect();
            if good.is_empty() {
                continue;
            }
            let slots: Vec<u8> = (0..idx.player_aims.len().min(16))
                .filter(|&i| idx.player_aims[i].len() >= 300)
                .map(|i| i as u8)
                .collect();
            if slots.is_empty() {
                continue;
            }
            let mut per_film: Vec<f32> = Vec::new();
            for tr in &good {
                let mut best = std::f32::consts::PI;
                for &sl in &slots {
                    let mut d: Vec<f32> = Vec::new();
                    for (t, yaw) in &tr.facings {
                        let Some(a) = idx.aim_at(sl, *t) else { continue };
                        d.push(ang_diff(*yaw, a.yaw()).abs());
                    }
                    if d.len() >= 200 {
                        let n = d.len();
                        d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                        best = best.min(d[n / 2]);
                    }
                }
                per_film.push(best);
            }
            if per_film.is_empty() {
                continue;
            }
            per_film.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = per_film[per_film.len() / 2];
            eprintln!(
                "film_decode: {:<46} {} bipeds with a facing; best-slot median error: min {:.3} med {:.3} rad ({:.1} deg)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                good.len(),
                per_film[0],
                med,
                med.to_degrees()
            );
            tested += 1;
            if worst.as_ref().map_or(true, |w| per_film[0] > w.0) {
                worst = Some((per_film[0], path.file_name().unwrap_or_default().to_string_lossy().into_owned()));
            }
        }
        if tested == 0 {
            eprintln!("film_decode: no film had both biped facings and player aims");
            return;
        }
        let (best_err, which) = worst.unwrap();
        // Even the film whose best biped matches worst must beat a random field by a wide margin.
        assert!(
            best_err < 0.6,
            "the biped facing field should track SOME player's aim closely; worst film {which} best {best_err:.3} rad (random would be ~1.571)"
        );
    }

    /// The per-player APPEARANCE bytes: which parts of the 160-byte `s_player_configuration`
    /// record vary between players, and how many distinct values each byte takes.
    ///
    /// Reach armour customisation is one permutation per region of the Spartan render model, so
    /// the block should be a short run of small indices. A byte whose value set is large, or
    /// which never varies, is not one.
    #[test]
    #[ignore = "probe: find the armour/appearance bytes in the player record"]
    fn probe_player_appearance_bytes() {
        let mut recs: Vec<(String, String, Vec<u8>)> = Vec::new();
        let mut seen: std::collections::HashSet<(u64, Vec<u8>)> = Default::default();
        for path in sample_films().iter() {
            let Ok(f) = film::open_header(path) else { continue };
            for pl in f.header.players.iter().filter(|p| p.present()) {
                if pl.raw.len() < 160 {
                    continue;
                }
                let key = (pl.xuid, pl.raw.clone());
                if !seen.insert(key) {
                    continue;
                }
                recs.push((
                    path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                    pl.gamertag.clone(),
                    pl.raw.clone(),
                ));
            }
        }
        eprintln!("probe: {} distinct player records", recs.len());
        if recs.len() < 2 {
            return;
        }
        // Which bytes vary at all, and how widely.
        let mut varying: Vec<(usize, Vec<u8>)> = Vec::new();
        for i in 0..160usize {
            let mut vals: Vec<u8> = recs.iter().map(|r| r.2[i]).collect();
            vals.sort_unstable();
            vals.dedup();
            if vals.len() > 1 {
                varying.push((i, vals));
            }
        }
        eprintln!("probe: {} of 160 bytes vary between players", varying.len());
        // The ones a permutation index could live in: a handful of small values.
        for (i, vals) in &varying {
            let small = vals.iter().all(|v| *v < 64);
            eprintln!(
                "  +0x{i:02X}: {:>3} distinct{} {:?}",
                vals.len(),
                if small { "  <- small values" } else { "" },
                &vals[..vals.len().min(20)]
            );
        }
        // Exact min/max per byte in the candidate appearance window, which is what bounds a
        // permutation index: the max observed index must be less than the region's permutation
        // count, and a region is only a plausible home for a byte if that holds.
        eprintln!("probe: candidate appearance window 0x44..0x68 (count, min, max)");
        for i in 0x44..0x68usize {
            let mut vals: Vec<u8> = recs.iter().map(|r| r.2[i]).collect();
            vals.sort_unstable();
            let (lo, hi) = (vals[0], vals[vals.len() - 1]);
            vals.dedup();
            eprintln!("  +0x{i:02X}: {:>3} distinct, {lo}..{hi}", vals.len());
        }
        // And the raw gap regions, so the shape is visible.
        for (film, gt, raw) in recs.iter().take(8) {
            eprintln!("  {gt:<18} {film:.28}");
            eprintln!("    0x40..0x64 {}", raw[0x40..0x64].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
            eprintln!("    0x6C..0xA0 {}", raw[0x6C..0xA0].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
        }
    }

    /// #emblem  WITHIN ONE FILM, which bytes of the 160-byte `s_player_configuration` record
    /// differ between the players in it?
    ///
    /// This is the discriminating question for the emblem. A Reach emblem is six quantities
    /// (foreground index, background index, flags, primary / secondary / backdrop colour) and no
    /// two players in a 16-player lobby share one, so if the record carried an emblem then six
    /// of its columns would vary across the players of a SINGLE recording. Comparing players
    /// across films cannot tell an emblem from a per-session constant; comparing them inside one
    /// film can.
    #[test]
    #[ignore = "probe: per-film column variance of the player record"]
    fn probe_emblem_columns_within_one_film() {
        for path in sample_films().iter() {
            let Ok(f) = film::open_header(path) else { continue };
            let recs: Vec<&crate::film::FilmPlayer> = f.header.players.iter().filter(|p| p.present() && p.raw.len() >= 160).collect();
            if recs.len() < 8 {
                continue;
            }
            eprintln!(
                "\n=== {} : {} present players",
                path.file_name().unwrap_or_default().to_string_lossy(),
                recs.len()
            );
            let mut varying: Vec<(usize, Vec<u8>)> = Vec::new();
            for i in 0..160usize {
                let mut v: Vec<u8> = recs.iter().map(|r| r.raw[i]).collect();
                v.sort_unstable();
                v.dedup();
                if v.len() > 1 {
                    varying.push((i, v));
                }
            }
            // Columns OUTSIDE the known fields: valid/team, the two xuids, the two gamertags,
            // the service tag, the ten armour slots and the player index.
            let known = |i: usize| -> &'static str {
                match i {
                    0x00..=0x0F => "valid/team/machine",
                    0x10..=0x1F => "machine+xuid",
                    0x20..=0x3F => "gamertag",
                    0x40..=0x47 => "xuid#2",
                    0x58..=0x61 => "armour slot",
                    0x64..=0x6B => "service tag",
                    0x78..=0x97 => "gamertag#2",
                    0x98 => "player index",
                    _ => "",
                }
            };
            for (i, v) in &varying {
                if known(*i).is_empty() {
                    eprintln!("  +0x{i:02X}: {:>2} distinct {:?}   <- UNACCOUNTED", v.len(), v);
                }
            }
            let unaccounted = varying.iter().filter(|(i, _)| known(*i).is_empty()).count();
            eprintln!("  {} of 160 columns vary; {unaccounted} of those are outside the known fields", varying.len());
            // And the full matrix, so the shape is visible rather than described. `s_emblem_info`
            // is at +0x50 and runs eight bytes; its two trailing bytes are the engine's own
            // must-be-zero pad.
            for r in &recs {
                eprintln!(
                    "  {:<16} team {:>3} emblem[0x50..0x58] {}  {}",
                    r.gamertag,
                    r.team,
                    r.raw[0x50..0x58].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
                    match r.emblem() {
                        Some(e) => e.summary(),
                        None => "(fails the engine's own validity rules)".to_string(),
                    }
                );
            }
            // THE question: do the emblems differ between the players of this one recording?
            let mut set: std::collections::HashSet<[u8; 8]> = Default::default();
            for r in &recs {
                let mut k = [0u8; 8];
                k.copy_from_slice(&r.raw[0x50..0x58]);
                set.insert(k);
            }
            eprintln!("  DISTINCT emblem records in this one film: {} (of {} players)", set.len(), recs.len());
        }
    }

    /// What does the 12-bit aim PITCH actually span? A player spends most of their time looking
    /// near level, so the histogram's mode is "level" and the spread bounds the range.
    #[test]
    #[ignore = "probe: the aim pitch quantisation"]
    fn probe_aim_pitch_range() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let mut v: Vec<u16> = Vec::new();
            for slot in 0..idx.player_aims.len() {
                for (_, a) in &idx.player_aims[slot] {
                    v.push(a.pitch_q);
                }
            }
            if v.len() < 5000 {
                continue;
            }
            v.sort_unstable();
            let pct = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
            // 16 buckets over the 12-bit range, to see where the mass is.
            let mut hist = [0usize; 16];
            for q in &v {
                hist[(*q as usize) >> 8] += 1;
            }
            eprintln!(
                "probe: {} -- {} aim samples; min {} p1 {} p25 {} MEDIAN {} p75 {} p99 {} max {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                v.len(),
                v[0],
                pct(0.01),
                pct(0.25),
                pct(0.5),
                pct(0.75),
                pct(0.99),
                v[v.len() - 1]
            );
            eprintln!(
                "  buckets of 256: {}",
                hist.iter()
                    .enumerate()
                    .map(|(i, n)| format!("{}:{:.1}%", i * 256, *n as f64 / v.len() as f64 * 100.0))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            return;
        }
    }

    /// Test `slot = owner - 1` (a 1-based owner index, 0 meaning "nobody") against an oracle that
    /// needs no correlation at all and uses EVERY owner group in every film rather than one
    /// anchor: a player's bodies can only exist while that player is in the game, so each group's
    /// lives must lie inside the tick range over which its slot carries per-tick records.
    ///
    /// Offsets -1, 0 and +1 are all scored, so the right one has to beat its neighbours rather
    /// than merely look plausible.
    #[test]
    #[ignore = "probe: does slot = owner - 1 survive a containment oracle"]
    fn probe_owner_minus_one() {
        let mut totals = [(0usize, 0usize); 3]; // (inside, total) for offsets -1, 0, +1
        let mut films = 0usize;
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.owner_group_count() < 2 {
                continue;
            }
            films += 1;
            let mut owners: Vec<u8> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11)
                .filter_map(|t| t.owner_raw)
                .collect();
            owners.sort_unstable();
            owners.dedup();
            for o in owners {
                // The group's overall life span.
                let (mut lo, mut hi) = (u32::MAX, 0u32);
                for t in idx.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw == Some(o)) {
                    lo = lo.min(t.first_tick);
                    hi = hi.max(t.last_tick);
                }
                if lo > hi {
                    continue;
                }
                for (k, delta) in [-1i32, 0, 1].iter().enumerate() {
                    let s = o as i32 + delta;
                    if s < 0 || s > 15 {
                        continue;
                    }
                    totals[k].1 += 1;
                    if let Some((a, b)) = idx.slot_tick_range(s as u8) {
                        // A tick of slack either side: the body can be created on the same tick
                        // the player's first record lands.
                        if lo + 2 >= a && hi <= b + 120 {
                            totals[k].0 += 1;
                        }
                    }
                }
            }
        }
        eprintln!("probe: {films} films");
        for (k, name) in ["slot = owner - 1", "slot = owner", "slot = owner + 1"].iter().enumerate() {
            let (ok, n) = totals[k];
            eprintln!(
                "  {name:<18}: {ok}/{n} owner groups whose lives fit inside that slot's record range ({:.1}%)",
                if n == 0 { 0.0 } else { ok as f64 / n as f64 * 100.0 }
            );
        }
    }

    /// THE BRIDGE: what maps a biped's 4-bit owner index to a player's configuration slot?
    ///
    /// The recorder is a free, trustworthy anchor on every film. Their configuration slot comes
    /// from the header (the `chdr` "saved by" XUID matched against the 16 player records), and
    /// their owner index comes from the bipeds that aim-correlate -- which are theirs precisely
    /// because their aim is local input, recorded exactly, rather than replicated. One anchor per
    /// film across the whole corpus discriminates between hypotheses instead of fitting one.
    ///
    /// Hypotheses, in the order they were ranked:
    ///   1. owner == rank of the slot among OCCUPIED slots, ascending
    ///   2. owner == index into the `s_game_options` machine table (via player record +0x10)
    ///   3. owner == join order (the order slots first acquire per-tick records)
    #[test]
    #[ignore = "probe: harvest recorder anchors and test the owner->slot bridge"]
    fn probe_owner_to_slot_bridge() {
        struct Anchor {
            film: String,
            tag: String,
            slot: u8,
            owner: u8,
            occupied: Vec<u8>,
            rank_header: Option<usize>,
            rank_records: Option<usize>,
            join_order: Option<usize>,
            cost: f32,
        }
        let mut anchors: Vec<Anchor> = Vec::new();
        let mut no_anchor = 0usize;
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.owner_group_count() == 0 {
                continue;
            }
            // The recorder's configuration slot, by XUID (robust) then by gamertag.
            let xuid = f.content.created_xuid;
            let tag = f.content.created_by.clone();
            let slot = f
                .header
                .players
                .iter()
                .position(|p| p.present() && xuid != 0 && p.xuid == xuid)
                .or_else(|| {
                    f.header.players.iter().position(|p| p.present() && !tag.is_empty() && p.gamertag == tag)
                });
            // The recorder's owner index: the owner of the single most confident group.
            let best = idx
                .bindings
                .iter()
                .filter(|b| b.from_facing && b.cost < 0.35)
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
            let owner = best.and_then(|b| {
                idx.tracks.iter().find(|t| t.net_id == b.net_id).and_then(|t| t.owner_raw)
            });
            let (Some(slot), Some(owner), Some(best)) = (slot, owner, best) else {
                no_anchor += 1;
                continue;
            };
            // Occupied slots, two readings: the header's roster, and every slot that ever carried
            // a per-tick record (which includes players who joined later).
            let occ_header: Vec<u8> =
                (0..16u8).filter(|&i| f.header.players.get(i as usize).map_or(false, |p| p.present())).collect();
            let occ_records: Vec<u8> = (0..16u8)
                .filter(|&i| idx.player_aims.get(i as usize).map_or(false, |v| v.len() >= 30))
                .collect();
            // Join order: slots sorted by the tick of their first per-tick record.
            let mut join: Vec<(u32, u8)> = (0..16u8)
                .filter_map(|i| {
                    idx.player_aims.get(i as usize).and_then(|v| v.first()).map(|(t, _)| (*t, i))
                })
                .collect();
            join.sort();
            anchors.push(Anchor {
                film: path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                tag,
                slot: slot as u8,
                owner,
                occupied: occ_header.clone(),
                rank_header: occ_header.iter().position(|&s| s == slot as u8),
                rank_records: occ_records.iter().position(|&s| s == slot as u8),
                join_order: join.iter().position(|&(_, s)| s == slot as u8),
                cost: best.cost,
            });
        }
        eprintln!("probe: {} anchors harvested ({no_anchor} films gave none)", anchors.len());
        eprintln!(
            "  {:<44} {:<16} {:>4} {:>6} | {:>6} {:>7} {:>5} | cost",
            "film", "recorder", "slot", "owner", "rankH", "rankR", "join"
        );
        let (mut h_ok, mut r_ok, mut j_ok, mut id_ok) = (0usize, 0usize, 0usize, 0usize);
        for a in &anchors {
            let f = |o: Option<usize>| o.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
            eprintln!(
                "  {:<44} {:<16} {:>4} {:>6} | {:>6} {:>7} {:>5} | {:.3}",
                &a.film[..a.film.len().min(44)],
                &a.tag[..a.tag.len().min(16)],
                a.slot,
                a.owner,
                f(a.rank_header),
                f(a.rank_records),
                f(a.join_order),
                a.cost
            );
            if a.rank_header == Some(a.owner as usize) {
                h_ok += 1;
            }
            if a.rank_records == Some(a.owner as usize) {
                r_ok += 1;
            }
            if a.join_order == Some(a.owner as usize) {
                j_ok += 1;
            }
            if a.slot == a.owner {
                id_ok += 1;
            }
            let _ = &a.occupied;
        }
        let n = anchors.len();
        eprintln!("probe: owner == slot                      : {id_ok}/{n}");
        eprintln!("probe: owner == rank among HEADER players  : {h_ok}/{n}");
        eprintln!("probe: owner == rank among slots w/ records: {r_ok}/{n}");
        eprintln!("probe: owner == join order                 : {j_ok}/{n}");
    }

    /// The full cost matrix: every owner group against every player slot. This is what decides
    /// whether a group can be named, so when a player goes unnamed this says whether the right
    /// answer was there but below threshold, or simply absent.
    #[test]
    #[ignore = "probe: the group-vs-slot cost matrix"]
    fn probe_identity_cost_matrix() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.owner_group_count() < 6 {
                continue;
            }
            eprintln!(
                "probe: {}  {} owner groups, {} slots with aims",
                path.file_name().unwrap_or_default().to_string_lossy(),
                idx.owner_group_count(),
                (0..idx.player_aims.len()).filter(|&i| idx.player_aims[i].len() >= 30).count()
            );
            let mut owners: Vec<u8> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11)
                .filter_map(|t| t.owner_raw)
                .collect();
            owners.sort_unstable();
            owners.dedup();
            let slots: Vec<usize> =
                (0..idx.player_aims.len().min(16)).filter(|&i| idx.player_aims[i].len() >= 30).collect();
            eprintln!("  owner | lives facings travel |  best slots (cost)");
            for o in owners {
                let tis: Vec<&EntityTrack> = idx
                    .tracks
                    .iter()
                    .filter(|t| t.entity_type == 11 && t.owner_raw == Some(o))
                    .collect();
                let mut pooled: Vec<(u32, f32)> = Vec::new();
                let mut nf = 0;
                let mut nt = 0;
                for tr in &tis {
                    let (h, ff) = idx.heading_samples_pub(tr);
                    if h.len() >= 45 {
                        if ff { nf += 1 } else { nt += 1 }
                    }
                    pooled.extend(h);
                }
                let mut scores: Vec<(f32, usize, usize)> = Vec::new();
                for &sl in &slots {
                    let mut d: Vec<f32> = Vec::new();
                    for (t, h) in &pooled {
                        if let Some(a) = idx.aim_at(sl as u8, *t) {
                            d.push(ang_diff(*h, a.yaw()).abs());
                        }
                    }
                    if d.len() >= 45 {
                        let n = d.len();
                        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        scores.push((d[n / 2], sl, n));
                    }
                }
                scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!(
                    "  {o:>5} | {:>5} {:>7} {:>6} | {}",
                    tis.len(),
                    nf,
                    nt,
                    scores
                        .iter()
                        .take(4)
                        .map(|(c, s, n)| format!("slot{s}:{c:.2}(n={n})"))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
            }
            return;
        }
    }

    /// Separate two possibilities: the field at offset 272 IS the owner but numbered differently
    /// from the per-tick mask bit, versus it is not the owner at all. The test is whether bodies
    /// the aim correlation assigns to the SAME slot share ONE value of the field, and whether the
    /// mapping value->slot is one-to-one.
    #[test]
    #[ignore = "probe: is the owner field consistent with the aim slots up to renumbering"]
    fn probe_owner_field_consistency() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let known: Vec<(u32, u8)> = idx
                .bindings
                .iter()
                .filter(|b| b.from_facing && b.cost < 0.35 && b.margin > 0.4)
                .map(|b| (b.net_id, b.slot))
                .collect();
            if known.len() < 3 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let want: std::collections::HashMap<u32, u8> = known.iter().copied().collect();
            let mut rows: Vec<(u32, u8, Vec<u8>)> = Vec::new();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update {
                        if let Some(slot) = want.get(&e.net_id) {
                            if !rows.iter().any(|(i, _, _)| *i == e.net_id) {
                                rows.push((e.net_id, *slot, e.payload.clone()));
                            }
                        }
                    }
                }
            }
            if rows.len() < 3 {
                continue;
            }
            eprintln!(
                "probe: {}  {} aim-confident bodies",
                path.file_name().unwrap_or_default().to_string_lossy(),
                rows.len()
            );
            // For a few candidate offsets, show the field value beside the aim slot.
            for (off, width) in [(272usize, 4u32), (271, 5), (270, 6), (82, 6), (79, 6)] {
                let vals: Vec<String> = rows
                    .iter()
                    .map(|(_, slot, pl)| {
                        let mut r = BitReader::new(pl);
                        let v = if r.skip(off).is_ok() { r.bits(width).ok() } else { None };
                        match v {
                            Some(v) => format!("slot{slot}->{v}"),
                            None => format!("slot{slot}->?"),
                        }
                    })
                    .collect();
                // Is value -> slot a function (same value never maps to two slots)?
                let mut map: std::collections::HashMap<u64, std::collections::HashSet<u8>> = Default::default();
                for (_, slot, pl) in &rows {
                    let mut r = BitReader::new(pl);
                    if r.skip(off).is_ok() {
                        if let Ok(v) = r.bits(width) {
                            map.entry(v).or_default().insert(*slot);
                        }
                    }
                }
                let consistent = map.values().all(|s| s.len() == 1);
                eprintln!(
                    "  offset {off:<4} width {width}: value->slot a function? {consistent}  [{}]",
                    vals.join(" ")
                );
            }
            return;
        }
    }

    /// Cross-validate the candidate owner field (4 bits at creation-payload offset 272) against
    /// the independent aim-correlation oracle, on every film that has one.
    #[test]
    #[ignore = "probe: does the owner field agree with the aim correlation"]
    fn probe_owner_field_agrees_with_aim() {
        let (mut agree, mut total) = (0usize, 0usize);
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let known: Vec<(u32, u8)> = idx
                .bindings
                .iter()
                .filter(|b| b.from_facing && b.cost < 0.3 && b.margin > 0.5)
                .map(|b| (b.net_id, b.slot))
                .collect();
            if known.is_empty() {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let want: std::collections::HashMap<u32, u8> = known.iter().copied().collect();
            let mut owner: std::collections::HashMap<u32, u8> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update && want.contains_key(&e.net_id) {
                        if let Some(o) = creation_owner_slot(&e.payload) {
                            owner.entry(e.net_id).or_insert(o);
                        }
                    }
                }
            }
            if owner.is_empty() {
                continue;
            }
            let mut ok = 0usize;
            for (id, slot) in &want {
                if let Some(o) = owner.get(id) {
                    total += 1;
                    if o == slot {
                        ok += 1;
                        agree += 1;
                    }
                }
            }
            eprintln!(
                "probe: {:<46} {ok}/{} aim-confident bodies agree with the owner field",
                path.file_name().unwrap_or_default().to_string_lossy(),
                owner.len()
            );
        }
        eprintln!("probe: TOTAL {agree}/{total} agree");
    }

    /// Find the OWNER field in a biped's creation payload without needing to know any owner.
    ///
    /// The test is structural, so it needs no oracle at all: a real owner field must partition the
    /// film's bipeds into groups whose LIVES NEVER OVERLAP, because one player cannot be driving
    /// two bipeds at the same moment. A field that is anything else -- a tag index, a flag word,
    /// part of a float -- scatters overlapping lives across its groups immediately. The number of
    /// distinct values should also land near the number of players.
    #[test]
    #[ignore = "probe: structural search for the biped owner field"]
    fn probe_biped_owner_by_nonoverlap() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let lives: Vec<(u32, u32, u32)> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11 && t.positions.len() >= 20)
                .map(|t| (t.net_id, t.first_tick, t.last_tick))
                .collect();
            if lives.len() < 12 {
                continue;
            }
            let named = f.header.players.iter().filter(|p| p.present()).count();
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let want: std::collections::HashSet<u32> = lives.iter().map(|(i, _, _)| *i).collect();
            let mut creation: std::collections::HashMap<u32, Vec<u8>> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update && want.contains(&e.net_id) {
                        creation.entry(e.net_id).or_insert_with(|| e.payload.clone());
                    }
                }
            }
            let have: Vec<(u32, u32, u32)> =
                lives.into_iter().filter(|(i, _, _)| creation.contains_key(i)).collect();
            if have.len() < 12 {
                continue;
            }
            let minbits = have.iter().map(|(i, _, _)| creation[i].len()).min().unwrap_or(0) * 8;
            eprintln!(
                "probe: {}  {named} named players, {} bipeds with a creation payload, {minbits} payload bits",
                path.file_name().unwrap_or_default().to_string_lossy(),
                have.len()
            );
            // Score every candidate field.
            let mut best: Vec<(f64, usize, u32, usize, usize)> = Vec::new();
            for off in 0..minbits.saturating_sub(4) {
                for width in 4..=6u32 {
                    if off + width as usize > minbits {
                        continue;
                    }
                    let mut groups: std::collections::HashMap<u64, Vec<(u32, u32)>> = Default::default();
                    let mut ok = true;
                    for (id, a, b) in &have {
                        let mut r = BitReader::new(&creation[id]);
                        if r.skip(off).is_err() {
                            ok = false;
                            break;
                        }
                        match r.bits(width) {
                            Ok(v) => groups.entry(v).or_default().push((*a, *b)),
                            Err(_) => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok || groups.len() < 2 {
                        continue;
                    }
                    // Count overlapping pairs within each group.
                    let mut pairs = 0usize;
                    let mut bad = 0usize;
                    for v in groups.values() {
                        for i in 0..v.len() {
                            for j in i + 1..v.len() {
                                pairs += 1;
                                if v[i].0 <= v[j].1 && v[j].0 <= v[i].1 {
                                    bad += 1;
                                }
                            }
                        }
                    }
                    if pairs == 0 {
                        continue; // every group a singleton: no information
                    }
                    let clean = 1.0 - bad as f64 / pairs as f64;
                    best.push((clean, off, width, groups.len(), pairs));
                }
            }
            best.sort_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(b.4.cmp(&a.4))
            });
            eprintln!("  top candidates (clean = fraction of same-group life pairs that do NOT overlap):");
            for (clean, off, width, groups, pairs) in best.iter().take(10) {
                eprintln!("    offset {off:<4} width {width}  clean {:.3}  groups {groups:<3} pairs {pairs}", clean);
            }
            // For reference: how clean a RANDOM partition into the same number of groups is.
            eprintln!("  (a field carrying no owner information scores far below 1.0 here)");
            return;
        }
        eprintln!("probe: no film with 12+ bipeds carrying creation payloads");
    }

    /// Is the owning PLAYER stamped in a biped's CREATION message?
    ///
    /// The aim correlation gives a handful of bodies whose owner is known beyond doubt (cost under
    /// 0.2 rad, margin over 0.9). Those are the oracle: sweep every bit offset and small width in
    /// the creation payload for a field that equals the known player slot on ALL of them. A 4-bit
    /// field agreeing with 7 independent known slots is 1 in 16^7 by chance.
    #[test]
    #[ignore = "probe: sweep a biped creation payload for the owning player slot"]
    fn probe_biped_creation_owner() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let known: Vec<(u32, u8)> = idx
                .bindings
                .iter()
                .filter(|b| b.from_facing && b.cost < 0.25 && b.margin > 0.6)
                .map(|b| (b.net_id, b.slot))
                .collect();
            if known.len() < 3 {
                continue;
            }
            // Re-walk the stream and keep each known entity's CREATION payload.
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let want: std::collections::HashMap<u32, u8> = known.iter().copied().collect();
            let mut creation: std::collections::HashMap<u32, Vec<u8>> = Default::default();
            let mut first_update: std::collections::HashMap<u32, Vec<u8>> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type != 11 || !want.contains_key(&e.net_id) {
                        continue;
                    }
                    if e.is_update {
                        first_update.entry(e.net_id).or_insert_with(|| e.payload.clone());
                    } else {
                        creation.entry(e.net_id).or_insert_with(|| e.payload.clone());
                    }
                }
            }
            eprintln!(
                "probe: {}  {} known owners, {} creations captured",
                path.file_name().unwrap_or_default().to_string_lossy(),
                known.len(),
                creation.len()
            );
            for (label, bag) in [("creation", &creation), ("first update", &first_update)] {
                if bag.len() < 3 {
                    eprintln!("  {label}: only {} captured, skipped", bag.len());
                    continue;
                }
                let minlen = bag.values().map(|v| v.len()).min().unwrap_or(0) * 8;
                let mut hits: Vec<(usize, u32)> = Vec::new();
                for off in 0..minlen.saturating_sub(8) {
                    for width in 4..=8u32 {
                        if off + width as usize > minlen {
                            continue;
                        }
                        let mut all = true;
                        for (id, slot) in &want {
                            let Some(pl) = bag.get(id) else { all = false; break };
                            let mut r = BitReader::new(pl);
                            if r.skip(off).is_err() {
                                all = false;
                                break;
                            }
                            match r.bits(width) {
                                Ok(v) if v as u8 == *slot => {}
                                _ => {
                                    all = false;
                                    break;
                                }
                            }
                        }
                        if all {
                            hits.push((off, width));
                        }
                    }
                }
                eprintln!(
                    "  {label}: payload {} bits; {} (offset,width) candidates match ALL {} known slots: {:?}",
                    minlen,
                    hits.len(),
                    bag.len(),
                    &hits[..hits.len().min(12)]
                );
            }
            // Also show the known slots so the result can be judged.
            let mut k = known.clone();
            k.sort_by_key(|(_, s)| *s);
            eprintln!(
                "  known: {}",
                k.iter().map(|(id, s)| format!("0x{id:08X}=slot{s}")).collect::<Vec<_>>().join(" ")
            );
            return;
        }
        eprintln!("probe: no film had 3+ high-confidence owners");
    }

    /// Why do so few type-11 messages yield a POSITION? On some clips bipeds carry a median of
    /// ZERO position samples, which starves every downstream thing (drawing, identity, cameras).
    /// This classifies every biped message by what became of it.
    #[test]
    #[ignore = "probe: what happens to each biped message"]
    fn probe_biped_message_fate() {
        for path in sample_films().iter().take(14) {
            let Ok(f) = film::open(path) else { continue };
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_frame, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let mut n = 0usize;
            let (mut creations, mut updates) = (0usize, 0usize);
            let mut mask_zero = 0usize;
            let mut payload_none = 0usize;
            let mut no_field0 = 0usize;
            let mut got_pos = 0usize;
            let mut latched = [0usize; 24];
            let mut masks: std::collections::HashMap<u64, usize> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type != 11 {
                        continue;
                    }
                    n += 1;
                    if e.is_update { updates += 1 } else { creations += 1 }
                    *masks.entry(e.dirty_mask & 0xFFFF).or_default() += 1;
                    if e.dirty_mask == 0 {
                        mask_zero += 1;
                        continue;
                    }
                    match decode_entity_payload(e, &params) {
                        None => payload_none += 1,
                        Some(st) => {
                            if let Some(l) = st.latched_at {
                                latched[(l as usize).min(23)] += 1;
                            }
                            if st.position_raw.is_some() {
                                got_pos += 1
                            } else {
                                no_field0 += 1
                            }
                        }
                    }
                }
            }
            if n < 200 {
                continue;
            }
            let mut mv: Vec<(u64, usize)> = masks.into_iter().collect();
            mv.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            eprintln!(
                "probe: {}\n  {n} biped messages ({creations} creations, {updates} updates); mask==0 {mask_zero}, payload undecodable {payload_none}, decoded-but-no-field-0 {no_field0}, POSITION {got_pos} ({:.1}%)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                got_pos as f64 / n as f64 * 100.0
            );
            eprintln!(
                "  top dirty masks (low 16 bits): {}",
                mv.iter().take(6).map(|(m, c)| format!("{m:#06x}:{c}")).collect::<Vec<_>>().join(" ")
            );
            eprintln!(
                "  latched at field: {}",
                latched.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(" ")
            );
        }
    }

    /// WHY does the player/entity binding fail? The successes are 5.8 degrees; this reports the
    /// distribution of the FAILURES so the cause is visible rather than guessed: no aim data for
    /// the slot, no heading data for the biped, a cost above threshold, or a margin too small.
    #[test]
    #[ignore = "probe: why player bindings fail"]
    fn probe_identity_failures() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            let bipeds: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == 11).collect();
            if bipeds.len() < 8 {
                continue;
            }
            let named = f.header.players.iter().filter(|p| p.present()).count();
            let slots: Vec<usize> =
                (0..idx.player_aims.len().min(16)).filter(|&i| idx.player_aims[i].len() >= 30).collect();
            eprintln!(
                "probe: {}  {named} named players, {} slots with aims, {} bipeds",
                path.file_name().unwrap_or_default().to_string_lossy(),
                slots.len(),
                bipeds.len()
            );
            // Per biped: how it was scored and what happened.
            let mut no_heading = 0;
            let mut facing = 0;
            let mut travel = 0;
            let mut costs_f: Vec<f32> = Vec::new();
            let mut costs_t: Vec<f32> = Vec::new();
            let mut margins: Vec<f32> = Vec::new();
            for tr in &bipeds {
                let (h, from_facing) = idx.heading_samples_pub(tr);
                if h.len() < 45 {
                    no_heading += 1;
                    continue;
                }
                if from_facing { facing += 1 } else { travel += 1 }
                if let Some(b) = idx.bindings.iter().find(|b| b.net_id == tr.net_id) {
                    if b.from_facing { costs_f.push(b.cost) } else { costs_t.push(b.cost) }
                    margins.push(b.margin);
                }
            }
            let stat = |v: &mut Vec<f32>| -> String {
                if v.is_empty() {
                    return "none".into();
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                format!("n={} min {:.2} med {:.2} max {:.2}", v.len(), v[0], v[v.len() / 2], v[v.len() - 1])
            };
            eprintln!(
                "  scored by facing {facing}, by travel {travel}, unscorable (too few heading samples) {no_heading}"
            );
            eprintln!("  facing costs: {}", stat(&mut costs_f));
            eprintln!("  travel costs: {}", stat(&mut costs_t));
            eprintln!("  margins:      {}", stat(&mut margins));
            // How much FACING data each biped actually has -- that is the real gate.
            let mut fac: Vec<usize> = bipeds.iter().map(|t| t.facings.len()).collect();
            fac.sort_unstable();
            eprintln!(
                "  facing samples per biped: min {} med {} max {}; {} of {} have >= 30",
                fac[0],
                fac[fac.len() / 2],
                fac[fac.len() - 1],
                fac.iter().filter(|n| **n >= 30).count(),
                fac.len()
            );
            let mut pos: Vec<usize> = bipeds.iter().map(|t| t.positions.len()).collect();
            pos.sort_unstable();
            eprintln!(
                "  position samples per biped: min {} med {} max {}",
                pos[0],
                pos[pos.len() / 2],
                pos[pos.len() - 1]
            );
            return;
        }
    }

    /// Where is a BIPED's facing?
    ///
    /// Every entity type reports its orientation on ~100% of its position updates except type 11,
    /// which reports it on 0.1% -- because the base walk's gate bit before field 2 is almost
    /// always clear for a biped, which latches the walk and ends the base block right after the
    /// position. So a biped's facing must live in the DERIVED (unit) block that follows.
    ///
    /// Rather than guess at the unit block's layout, search for it against an oracle: where a
    /// biped is moving quickly it is mostly facing the way it is going, so a field whose decoded
    /// angle tracks the direction of travel is the facing. This probe sweeps (bit offset after
    /// the base block, field width) and prints the best candidates.
    #[test]
    #[ignore = "probe: search the biped payload tail for the body facing"]
    fn probe_biped_facing_field() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (frame, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            // Collect every type-11 update that yielded a position, with its payload tail.
            struct Samp {
                net_id: u32,
                tick: u32,
                pos: [f32; 3],
                payload: Vec<u8>,
                at: usize,
            }
            let mut samples: Vec<Samp> = Vec::new();
            let mut tick0: Option<u32> = None;
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                let b = *tick0.get_or_insert(t.tick.wrapping_sub(i as u32));
                let tick = t.tick.wrapping_sub(b);
                for e in &t.entities {
                    if e.entity_type != 11 {
                        continue;
                    }
                    let Some(st) = decode_entity_payload(e, &params) else { continue };
                    let Some(raw) = st.position_raw else { continue };
                    let pos = frame.world(raw);
                    if !pos.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    samples.push(Samp {
                        net_id: e.net_id,
                        tick,
                        pos,
                        payload: e.payload.clone(),
                        at: st.bits_after_base,
                    });
                }
            }
            if samples.len() < 2000 {
                continue;
            }
            // Travel heading per sample, from the previous sample of the SAME entity.
            let mut prev: std::collections::HashMap<u32, (u32, [f32; 3])> = Default::default();
            let mut oracle: Vec<(usize, f32)> = Vec::new(); // (sample index, heading)
            for (i, s) in samples.iter().enumerate() {
                if let Some(&(pt, pp)) = prev.get(&s.net_id) {
                    let dt = s.tick.saturating_sub(pt) as f32 / 60.0;
                    let (dx, dy) = (s.pos[0] - pp[0], s.pos[1] - pp[1]);
                    let d = (dx * dx + dy * dy).sqrt();
                    if dt > 0.0 && dt <= 0.2 && d / dt > 2.0 {
                        oracle.push((i, dy.atan2(dx)));
                    }
                }
                prev.insert(s.net_id, (s.tick, s.pos));
            }
            if oracle.len() < 500 {
                continue;
            }
            // Sweep: offset past the base block, and field width.
            let mut best: Vec<(f32, usize, u32, usize)> = Vec::new();
            for off in 0..48usize {
                for width in [7u32, 8, 9, 10, 11, 12, 13, 14, 16] {
                    let mut d: Vec<f32> = Vec::new();
                    for &(i, head) in &oracle {
                        let s = &samples[i];
                        let start = s.at + off;
                        if start + width as usize > s.payload.len() * 8 {
                            continue;
                        }
                        let mut r = BitReader::new(&s.payload);
                        if r.skip(start).is_err() {
                            continue;
                        }
                        let Ok(v) = r.bits(width) else { continue };
                        let ang = v as f32 / (1u64 << width) as f32 * std::f32::consts::TAU;
                        d.push(ang_diff(ang, head).abs());
                    }
                    if d.len() < 400 {
                        continue;
                    }
                    let n = d.len();
                    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    best.push((d[n / 2], off, width, n));
                }
            }
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "probe: {} -- {} biped position samples, {} with a usable travel heading",
                path.file_name().unwrap_or_default().to_string_lossy(),
                samples.len(),
                oracle.len()
            );
            eprintln!("  (a random field would score about {:.3} rad)", std::f32::consts::PI / 2.0);
            for (cost, off, width, n) in best.iter().take(8) {
                eprintln!(
                    "  offset +{off:<3} width {width:<3} median |err| {cost:.3} rad ({:.1} deg) over {n}",
                    cost.to_degrees()
                );
            }
            return;
        }
        eprintln!("probe: no film with enough moving bipeds");
    }

    /// How often each entity type carries a POSITION vs an ORIENTATION. The player camera and the
    /// identity correlation both want the facing, so a type that moves but never reports its
    /// orientation is the gap to close.
    #[test]
    #[ignore = "probe: position vs orientation coverage per entity type"]
    fn probe_orientation_coverage() {
        let mut shown = 0;
        for p in sample_films().iter() {
            if shown >= 4 {
                break;
            }
            let Ok(f) = film::open(p) else { continue };
            let idx = build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.tracks.is_empty() {
                continue;
            }
            shown += 1;
            let mut agg: std::collections::BTreeMap<u8, (usize, usize, usize)> = Default::default();
            for t in &idx.tracks {
                let e = agg.entry(t.entity_type).or_default();
                e.0 += 1;
                e.1 += t.positions.len();
                e.2 += t.orientations.len();
            }
            eprintln!("probe: {}", p.file_name().unwrap_or_default().to_string_lossy());
            for (ty, (n, pos, ori)) in agg {
                eprintln!(
                    "  type {ty:>2}: {n:>4} tracks, {pos:>7} positions, {ori:>7} orientations ({:.1}% as many)",
                    if pos == 0 { 0.0 } else { ori as f64 / pos as f64 * 100.0 }
                );
            }
        }
    }

    /// The owner rule must only ever fire when the recorder anchor corroborates it on that film,
    /// and when it does, it must name every owner group rather than one.
    #[test]
    fn the_owner_rule_is_gated_on_its_anchor() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- owner rule gate skipped");
            return;
        }
        let mut fired = 0usize;
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let slot = f.recorder_slot();
            let idx = build_index_for(&f.bytes, &f.data, &f.header.map_variant, slot);
            if !idx.owner_rule_ok {
                // Nothing may be rule-derived on a film where the rule was not corroborated.
                assert!(
                    !idx.bindings.iter().any(|b| b.from_owner_rule),
                    "{:?} promoted the owner rule without corroborating it",
                    path.file_name()
                );
                continue;
            }
            fired += 1;
            let named: std::collections::HashSet<u8> =
                idx.bindings.iter().filter(|b| b.is_confident()).map(|b| b.slot).collect();
            // WHICH anchor corroborated the rule decides what else must hold. The author record
            // is a stated fact and the recorder is necessarily among the named; the live-aim
            // anchor is an inference used only when there IS no author record, and it does NOT
            // require the recorder to be named -- a player who joined after the header was
            // written occupies no header slot, so there is no name to give them. That asymmetry
            // is correct and is asserted rather than glossed.
            let rec = match idx.owner_anchor {
                OwnerAnchor::AuthorRecord => {
                    let rec = slot.expect("the author-record anchor IS the recorder slot");
                    assert!(named.contains(&rec), "{:?} named everyone but the recorder", path.file_name());
                    rec
                }
                OwnerAnchor::LiveAim => {
                    assert!(
                        slot.is_none(),
                        "{:?} has an author record and must not fall back to the live aim",
                        path.file_name()
                    );
                    let rec = idx.live_aim_slot().expect("the live-aim anchor IS a live-aim slot");
                    // Injective, and only onto slots the recording actually has.
                    let mut seen = [false; 16];
                    for b in idx.bindings.iter().filter(|b| b.from_owner_rule) {
                        assert!(
                            idx.slot_occupied(b.slot),
                            "{:?} promoted onto unoccupied slot {}",
                            path.file_name(),
                            b.slot
                        );
                        if std::mem::replace(&mut seen[b.slot as usize], true) {
                            // Several lives of ONE owner share a slot, which is expected; two
                            // different owners sharing one is not, and the gate rejects it.
                        }
                    }
                    rec
                }
                OwnerAnchor::None => unreachable!("owner_rule_ok implies an anchor"),
            };
            for b in idx.bindings.iter().filter(|b| b.from_owner_rule) {
                let owner = idx
                    .tracks
                    .iter()
                    .find(|t| t.net_id == b.net_id)
                    .and_then(|t| t.owner_raw)
                    .expect("a rule-derived binding has an owner stamp");
                // #owner-wrap  The engine writes `index + 1` in a FOUR-BIT field and reads back
                // `read_bits(4) - 1`, so the subtraction does NOT wrap: a rule-derived binding is
                // always `owner - 1`, and owner 0 (the `-1` sentinel) never produces one at all.
                // Written out rather than deferred to `owner_to_slot`, so this still pins the
                // rule instead of restating the code.
                assert!(owner >= 1, "owner 0 is the engine's -1 sentinel and must bind nobody");
                assert_eq!(
                    b.slot,
                    owner - 1,
                    "rule-derived binding broke slot == owner - 1: {b:?} (owner {owner})"
                );
            }
            // Every owner group that maps to a slot present in the recording must be named --
            // that is the point of the rule, as against correlation naming only the recorder.
            let mappable = {
                let mut owners: Vec<u8> = idx
                    .tracks
                    .iter()
                    .filter(|t| t.entity_type == 11)
                    .filter_map(|t| t.owner_raw)
                    .collect();
                owners.sort_unstable();
                owners.dedup();
                owners
                    .into_iter()
                    .filter(|o| {
                        // #owner-wrap  A group is mappable when the rule gives it a slot AND the
                        // recording actually has that player. The slot question is deferred to
                        // the one implementation (wire 0 is the engine's -1 and names nobody;
                        // every other stamp is `wire - 1` and must behave like a single player).
                        // `the_owner_stamp_follows_the_engines_own_arithmetic` gates that rule,
                        // so this assertion stays about NAMING, not about the stamp arithmetic.
                        let Some(sl) = idx.owner_to_slot(*o) else { return false };
                        idx.player_aims.get(sl as usize).map_or(false, |v| !v.is_empty())
                    })
                    .count()
            };
            assert_eq!(
                named.len(),
                mappable,
                "{:?} corroborated the rule ({:?}) but named {} of {mappable} mappable owner groups",
                path.file_name(),
                idx.owner_anchor,
                named.len()
            );
            eprintln!(
                "film_decode: {:<46} rule corroborated by slot {rec} via {:?}; {} players named, {} bodies grouped",
                path.file_name().unwrap_or_default().to_string_lossy(),
                idx.owner_anchor,
                named.len(),
                idx.owner_group_count()
            );
        }
        eprintln!("film_decode: the owner rule fired on {fired} film(s)");
    }

    /// The player/entity link, measured on every real film this machine has.
    ///
    /// The link is DERIVED (nothing in the stream states it), so this test is the oracle on the
    /// derivation: on a film where the bipeds carry a decoded facing, the matched slot's cost has
    /// to be a small angle and the runner-up has to lose clearly. It also records the shape of
    /// the problem that is NOT solved -- bipeds with too few heading samples to score, and player
    /// slots whose records appear mid-film because that player joined late.
    #[test]
    fn player_identity_over_real_films() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- identity oracle skipped");
            return;
        }
        let mut facing_costs: Vec<f32> = Vec::new();
        for p in films.iter().take(12) {
            let Ok(f) = film::open(p) else { continue };
            let idx = build_index_for(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot());
            let bipeds = idx.tracks.iter().filter(|t| t.entity_type == 11).count();
            if bipeds == 0 {
                continue;
            }
            let named = f.header.players.iter().filter(|p| p.present()).count();
            let with_aims = (0..idx.player_aims.len()).filter(|&i| idx.player_aims[i].len() >= 30).count();
            let conf = idx.bindings.iter().filter(|b| b.is_confident()).count();
            let slots_named: std::collections::HashSet<u8> =
                idx.bindings.iter().filter(|b| b.is_confident()).map(|b| b.slot).collect();
            eprintln!(
                "film_decode: {:<46} {named} header, {with_aims} aims, {bipeds} bipeds/{} groups, {} bound, {conf} conf, {} NAMED{}",
                p.file_name().unwrap_or_default().to_string_lossy(),
                idx.owner_group_count(),
                idx.bindings.len(),
                slots_named.len(),
                if idx.owner_rule_ok { "  [owner rule corroborated]" } else { "" }
            );
            for b in idx.bindings.iter().filter(|b| b.from_facing && b.is_confident() && !b.from_owner_rule) {
                facing_costs.push(b.cost);
                assert!(
                    b.cost < 0.45,
                    "a confident facing match must be a small angle: {:?} in {:?}",
                    b,
                    p.file_name()
                );
                assert!(b.margin > 0.15, "a confident match must beat the runner-up: {b:?}");
                // A binding must be inside its own entity's lifetime.
                let tr = idx.tracks.iter().find(|t| t.net_id == b.net_id).expect("bound entity exists");
                assert_eq!((b.first_tick, b.last_tick), (tr.first_tick, tr.last_tick));
                // And the entity must be findable through the slot at a tick inside it.
                let mid = b.first_tick + (b.last_tick - b.first_tick) / 2;
                assert!(idx.entity_for_slot(b.slot, mid).is_some(), "{b:?} not reachable at {mid}");
            }
            // A slot owns at most one owner GROUP, and a group's every life carries the same
            // cost and slot. (Lives inside a group CAN overlap in time: a corpse or ragdoll keeps
            // its entity alive past the respawn, so the owner legitimately has two bodies for a
            // moment. The group, not the time range, is what makes the identity exclusive.)
            let mut per_slot: std::collections::HashMap<u8, std::collections::HashSet<u32>> =
                Default::default();
            for b in &idx.bindings {
                per_slot.entry(b.slot).or_default().insert(b.cost.to_bits());
            }
            for (s, costs) in &per_slot {
                assert_eq!(costs.len(), 1, "slot {s} was given more than one owner group: {costs:?}");
            }
        }
        if facing_costs.is_empty() {
            eprintln!("film_decode: no facing-based player matches in these films (travel-only)");
        } else {
            let mean = facing_costs.iter().sum::<f32>() / facing_costs.len() as f32;
            eprintln!(
                "film_decode: {} facing-based player matches, mean cost {:.3} rad ({:.1} degrees)",
                facing_costs.len(),
                mean,
                mean.to_degrees()
            );
            assert!(mean < 0.35, "facing matches should average a few degrees, got {mean}");
        }
    }

    /// What the stream says about entity identity, per film: which 5-bit entity types appear, how
    /// many distinct entities, and whether the creation wave matches the film's own Forge object
    /// count (a cross-check on the type that dominates it).
    #[test]
    fn entity_census_over_real_films() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- entity census skipped");
            return;
        }
        let mut any = false;
        for p in films.iter().take(8) {
            let Ok(f) = film::open(p) else { continue };
            let st = scan(&f.bytes, &f.data);
            if st.decoded == 0 {
                eprintln!(
                    "film_decode: {:<46} 0 of {} records decoded -- {}",
                    p.file_name().unwrap_or_default().to_string_lossy(),
                    st.records,
                    st.report().lines().last().unwrap_or("")
                );
                continue;
            }
            any = true;
            eprintln!(
                "film_decode: {:<46} forge objects {:>4} | {}",
                p.file_name().unwrap_or_default().to_string_lossy(),
                f.forge_object_count(),
                st.report().replace('\n', " |")
            );
        }
        if !any {
            eprintln!("film_decode: no film decoded far enough for a census (the two unimplemented player records stop most)");
        }
    }

    /// #owner-wrap  THE OWNER STAMP IS THE ENGINE'S ARITHMETIC, over the whole film corpus.
    ///
    /// This gate replaces `owner_zero_only_happens_when_slot_15_is_occupied`, which asserted that
    /// a stamp of 0 is the *wrapped sixteenth player*. **The engine refutes that**, and the gate
    /// now asserts what the engine actually does (see [`FilmIndex::owner_to_slot`], which cites
    /// both sites): the writer at `haloreach.dll+0x3B242C` emits `*(char *)(obj + 0x3E) + 1` in
    /// four bits and the reader at `+0x3B2590` takes `read_bits(4) - 1`, so wire 0 is the field's
    /// `-1` sentinel and the largest index four bits can carry is 14.
    ///
    /// What is checked:
    ///
    /// * wire 0 names NOBODY, on every film, whatever the lobby size;
    /// * every other stamp maps to `wire - 1`, which is at most slot 14 — a stamp can never name
    ///   slot 15;
    /// * the mapping is INJECTIVE: two stamps never name one slot.
    ///
    /// It also prints how many films carry a wire-0 body, which is the measurement that says what
    /// the four-bit ceiling costs.
    #[test]
    #[ignore = "needs the real film corpus; run alone with --ignored --nocapture --test-threads=1"]
    fn the_owner_stamp_follows_the_engines_own_arithmetic() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("no films on this machine - skipped");
            return;
        }
        let (mut checked, mut with_zero, mut zero_bodies, mut sixteen) =
            (0usize, 0usize, 0usize, 0usize);
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            let idx = build_index_players(
                &f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players,
            );
            let owners = idx.owner_groups();
            if owners.is_empty() {
                continue;
            }
            checked += 1;
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if f.header.players.iter().filter(|p| p.present()).count() >= 16 {
                sixteen += 1;
            }
            // Injective, follows `wire - 1`, and never onto slot 15.
            let mut seen = [false; 16];
            for &o in &owners {
                let Some(sl) = idx.owner_to_slot(o) else { continue };
                assert!(
                    sl <= 14,
                    "{name}: stamp {o} named slot {sl}; the four-bit field cannot carry an index \
                     above 14, because the engine writes index + 1"
                );
                assert_eq!(
                    sl,
                    o - 1,
                    "{name}: stamp {o} must map to slot {} by the engine's own `read_bits(4) - 1`",
                    o - 1
                );
                assert!(
                    !std::mem::replace(&mut seen[sl as usize], true),
                    "{name}: two owner stamps both map to slot {sl} (owners {owners:?})"
                );
            }
            // Wire 0 names nobody, always.
            if owners.contains(&0) {
                with_zero += 1;
                zero_bodies += idx.owner_lives(0);
                assert_eq!(
                    idx.owner_to_slot(0),
                    None,
                    "{name}: a stamp of 0 is the engine's -1 sentinel and must name nobody"
                );
            }
        }
        eprintln!(
            "film_decode: the owner stamp follows `read_bits(4) - 1` on {checked} film(s); \
             {with_zero} carry a stamp-0 body ({zero_bodies} bod(ies) the field attributes to \
             nobody); {sixteen} film(s) occupy all sixteen header slots, which a four-bit field \
             cannot cover -- so at least one player there is unnameable from the stamp alone"
        );
    }

    /// #player-unit  WHICH of `s_player_action`'s three object references is the player's UNIT?
    ///
    /// The engine assigns a player a unit in `Player_SetUnit` (`haloreach.dll+0x45814`): it writes
    /// the unit's object index to `player_datum + 0x28` and the owning player index back to the
    /// object at `+0x1CC`. The wire form of that state is the per-player `s_player_action` block
    /// (`sub_18043FC00`), read once per slot behind a 16-bit mask, and three of its fields are
    /// 27-bit object references (`sub_1803D2D84`: `bits(16) << 16 | bits(11)`, the same encoding
    /// the event channel uses). HMS decoded all three and discarded them.
    ///
    /// The ORACLE is free and independent: the owner stamp already names most bodies, so for a
    /// reference that appears in slot S's own block, "is this a biped the stamp says belongs to
    /// slot S?" can be answered without the reference. A field that is the unit agrees; a field
    /// that is something else (the vehicle, the interaction target, the aim target) does not.
    #[test]
    #[ignore = "probe: which s_player_action reference is the player's unit"]
    fn probe_which_player_ref_is_the_unit() {
        let mut tot = [(0usize, 0usize, 0usize); 3]; // (agree, disagree, non-biped) per field
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players,
            );
            // net id -> the slot the OWNER STAMP bound it to (the independent oracle)
            let mut stamped: std::collections::HashMap<u32, u8> = Default::default();
            for b in idx.bindings.iter().filter(|b| b.is_confident() && b.from_owner_rule) {
                stamped.insert(b.net_id, b.slot);
            }
            let bipeds: std::collections::HashSet<u32> =
                idx.tracks.iter().filter(|t| t.entity_type == 11).map(|t| t.net_id).collect();
            if stamped.is_empty() {
                continue;
            }
            // Is the channel absent, or present and empty? Count the blocks themselves.
            eprintln!(
                "{:<46} {} s_player_action block(s) over {} tick(s); {} of them named an object",
                path.file_name().unwrap_or_default().to_string_lossy(),
                idx.player_action_blocks,
                idx.records_decoded,
                idx.player_objects.iter().map(|v| v.len()).sum::<usize>(),
            );
            let mut per = [(0usize, 0usize, 0usize); 3];
            for slot in 0..idx.player_objects.len().min(16) {
                for (_, act) in &idx.player_objects[slot] {
                    for (k, r) in [act.ref14, act.ref1c, act.ref2c].into_iter().enumerate() {
                        let Some(r) = r else { continue };
                        if !bipeds.contains(&r) {
                            per[k].2 += 1;
                            continue;
                        }
                        match stamped.get(&r) {
                            Some(s) if *s == slot as u8 => per[k].0 += 1,
                            Some(_) => per[k].1 += 1,
                            None => per[k].2 += 1,
                        }
                    }
                }
            }
            for k in 0..3 {
                tot[k].0 += per[k].0;
                tot[k].1 += per[k].1;
                tot[k].2 += per[k].2;
            }
            let n: usize = (0..3).map(|k| per[k].0 + per[k].1 + per[k].2).sum();
            if n == 0 {
                continue;
            }
            eprintln!(
                "{:<46} +0x14 {:>6}/{:<6} +0x1C {:>6}/{:<6} +0x2C {:>6}/{:<6}  (agree/disagree with the stamp)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                per[0].0, per[0].1, per[1].0, per[1].1, per[2].0, per[2].1
            );
        }
        for (k, name) in ["+0x14", "+0x1C", "+0x2C"].iter().enumerate() {
            let (a, d, o) = tot[k];
            let graded = a + d;
            eprintln!(
                "TOTAL {name}: {a} agree, {d} disagree with the owner stamp ({:.1}% of {graded} \
                 gradeable), {o} named a non-biped or an unstamped body",
                if graded > 0 { a as f64 / graded as f64 * 100.0 } else { 0.0 }
            );
        }
    }

    /// #catchall  WHAT ELSE IN THE STREAM NAMES A BODY?
    ///
    /// Owner stamp 0 is a catch-all on 8 of 25 films, so the bodies in it cannot be named from
    /// the stamp. This prints, per film, what the other candidate channels actually offer:
    ///
    /// * the per-player record's two unexplained handle-shaped fields (`f40`, 32 bits; `f08`,
    ///   48 bits) measured against the set of biped net ids -- if either IS the player's own
    ///   unit handle the whole problem is a stated fact;
    /// * the per-slot statborg score updates, as ticks, next to the ticks at which each body
    ///   was the RESPONSIBLE unit of a `damage_aftermath`.
    #[test]
    #[ignore = "probe: which channels can name a catch-all body"]
    fn probe_what_else_names_a_body() {
        let films = sample_films();
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let bipeds: std::collections::HashSet<u32> =
                idx.tracks.iter().filter(|t| t.entity_type == 11).map(|t| t.net_id).collect();
            let all: std::collections::HashSet<u32> = idx.tracks.iter().map(|t| t.net_id).collect();
            if bipeds.is_empty() {
                continue;
            }
            eprintln!(
                "\n=== {} : {} bipeds, {} owner groups, statborg {}/{} ===",
                path.file_name().unwrap_or_default().to_string_lossy(),
                bipeds.len(),
                idx.owner_group_count(),
                idx.statborg_decoded,
                idx.statborg_messages
            );
            // ---- f40 / f08 as a unit handle -----------------------------------------------
            let cuts: [(&str, fn(u64) -> u32); 4] = [
                ("f08.hi32", |v| (v >> 16) as u32),
                ("f08.lo32", |v| (v & 0xFFFF_FFFF) as u32),
                ("f08.hi16", |v| (v >> 32) as u32),
                ("f08.lo16", |v| (v & 0xFFFF) as u32),
            ];
            let (mut n40, mut h40b, mut h40a) = (0usize, 0usize, 0usize);
            let mut n08 = 0usize;
            let mut h08 = [0usize; 4];
            // f40 value -> which slots carried it (is it per-slot distinct?)
            let mut f40_slots: std::collections::HashMap<u32, std::collections::BTreeSet<u8>> =
                Default::default();
            for slot in 0..idx.player_aims.len().min(16) {
                for (_, a) in &idx.player_aims[slot] {
                    if let Some(v) = a.f40 {
                        n40 += 1;
                        if bipeds.contains(&v) {
                            h40b += 1;
                        }
                        if all.contains(&v) {
                            h40a += 1;
                        }
                        f40_slots.entry(v).or_default().insert(slot as u8);
                    }
                    if let Some(v) = a.f08 {
                        n08 += 1;
                        for (k, (_, cut)) in cuts.iter().enumerate() {
                            if bipeds.contains(&cut(v)) {
                                h08[k] += 1;
                            }
                        }
                    }
                }
            }
            if n40 > 0 {
                let shared = f40_slots.values().filter(|s| s.len() > 1).count();
                eprintln!(
                    "  f40 present {n40}: biped {h40b} ({:.1}%) any-entity {h40a} ({:.1}%); {} distinct, {shared} shared by >1 slot",
                    h40b as f64 / n40 as f64 * 100.0,
                    h40a as f64 / n40 as f64 * 100.0,
                    f40_slots.len()
                );
            } else {
                eprintln!("  f40 never present");
            }
            if n08 > 0 {
                let s: Vec<String> = cuts
                    .iter()
                    .enumerate()
                    .map(|(k, (nm, _))| format!("{nm} {:.1}%", h08[k] as f64 / n08 as f64 * 100.0))
                    .collect();
                eprintln!("  f08 present {n08}: {}", s.join("  "));
            }
            // ---- statborg vs damage_aftermath ---------------------------------------------
            // Per body: the ticks at which it was the responsible unit of a damage_aftermath
            // whose VICTIM died (the victim's track ends near that tick).
            let mut last_tick: std::collections::HashMap<u32, u32> = Default::default();
            for t in &idx.tracks {
                last_tick.insert(t.net_id, t.last_tick);
            }
            let mut kill_ticks: std::collections::HashMap<u32, Vec<u32>> = Default::default();
            for ev in &idx.events {
                if ev.kind != crate::film_action::EV_DAMAGE_AFTERMATH {
                    continue;
                }
                let (Some(v), Some(k)) = (ev.subject, ev.other) else { continue };
                if !bipeds.contains(&k) || !bipeds.contains(&v) || v == k {
                    continue;
                }
                // fatal only: the victim's body ends within 2 s of this damage
                let Some(&lt) = last_tick.get(&v) else { continue };
                if lt < ev.tick || lt - ev.tick > 120 {
                    continue;
                }
                kill_ticks.entry(k).or_default().push(ev.tick);
            }
            // Per slot: the ticks at which its round score INCREASED.
            let mut score_up: Vec<Vec<u32>> = vec![Vec::new(); 16];
            for slot in 0..16usize {
                let Some(v) = idx.statborg.get(slot) else { continue };
                let mut prev: i32 = 0;
                for (t, s) in v {
                    let Some(r) = s.round_score else { continue };
                    if (r as i32) > prev {
                        score_up[slot].push(*t);
                    }
                    prev = r as i32;
                }
            }
            let tot_up: usize = score_up.iter().map(|v| v.len()).sum();
            eprintln!(
                "  damage_aftermath fatal attributions: {} bodies, {} kills; score-up events {tot_up} over {} slots",
                kill_ticks.len(),
                kill_ticks.values().map(|v| v.len()).sum::<usize>(),
                score_up.iter().filter(|v| !v.is_empty()).count()
            );
            // For each body with kills, which slots' score-ups line up?
            let mut named_uniquely = 0usize;
            let mut bodies_with_kills = 0usize;
            let mut agree_with_owner = 0usize;
            let mut disagree_with_owner = 0usize;
            let mut kk: Vec<(&u32, &Vec<u32>)> = kill_ticks.iter().collect();
            kk.sort();
            for (body, ticks) in kk {
                bodies_with_kills += 1;
                // vote: slot gets a vote when it has a score-up within +-W ticks of a kill
                const W: u32 = 45;
                let mut votes = [0usize; 16];
                for &kt in ticks {
                    for slot in 0..16usize {
                        if score_up[slot].iter().any(|&st| st.abs_diff(kt) <= W) {
                            votes[slot] += 1;
                        }
                    }
                }
                let best = (0..16).max_by_key(|&s| votes[s]).unwrap();
                let second = (0..16).filter(|&s| s != best).max_by_key(|&s| votes[s]).unwrap();
                if votes[best] == 0 {
                    continue;
                }
                let owner = idx.tracks.iter().find(|t| t.net_id == *body).and_then(|t| t.owner_raw);
                let stamp_slot = owner.and_then(|o| idx.owner_to_slot(o));
                let unique = votes[best] >= votes[second] * 2 && votes[best] >= 2;
                if unique {
                    named_uniquely += 1;
                    if let Some(ss) = stamp_slot {
                        if ss == best as u8 {
                            agree_with_owner += 1;
                        } else {
                            disagree_with_owner += 1;
                        }
                    }
                }
                if bodies_with_kills <= 40 {
                    eprintln!(
                        "    body {body:#010x} owner {owner:?} stamp_slot {stamp_slot:?} kills {} -> best slot {best} ({} votes) second {second} ({}) {}",
                        ticks.len(),
                        votes[best],
                        votes[second],
                        if unique { "UNIQUE" } else { "" }
                    );
                }
            }
            eprintln!(
                "  -> {named_uniquely}/{bodies_with_kills} bodies uniquely voted; vs owner stamp: {agree_with_owner} agree, {disagree_with_owner} disagree"
            );
        }
    }

    /// #catchall  IS THE OWNER FIELD AT A VARIABLE OFFSET?
    ///
    /// `CREATION_OWNER_BIT` is an ABSOLUTE bit offset, but the shared creation reader
    /// (`creation_shared`) is VARIABLE-LENGTH: two of its fields are flagged (`+0x02`, 10 bits
    /// behind a set flag; `+0x05`, 4 bits behind a CLEAR flag). So a creation message whose flags
    /// differ puts every later field at a different place, and a fixed offset reads a misaligned
    /// window -- which is one way a body could carry owner 0 without belonging to slot 15.
    ///
    /// This prints, per film: the shared header's length for each biped, cross-tabulated against
    /// the owner value the fixed offset reads; and then a sweep of offsets RELATIVE to the end of
    /// the shared header, scored by the same structural test that found the field in the first
    /// place (same-group lives must not overlap).
    #[test]
    #[ignore = "probe: is the creation owner field at a variable offset"]
    fn probe_owner_field_alignment() {
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let lives: Vec<(u32, u32, u32)> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11)
                .map(|t| (t.net_id, t.first_tick, t.last_tick))
                .collect();
            if lives.len() < 12 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let want: std::collections::HashSet<u32> = lives.iter().map(|(i, _, _)| *i).collect();
            let mut creation: std::collections::HashMap<u32, Vec<u8>> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update && want.contains(&e.net_id) {
                        creation.entry(e.net_id).or_insert_with(|| e.payload.clone());
                    }
                }
            }
            let have: Vec<(u32, u32, u32)> =
                lives.into_iter().filter(|(i, _, _)| creation.contains_key(i)).collect();
            if have.len() < 12 {
                continue;
            }
            // Walk the shared header and report where it ENDS, plus the two flags.
            let shared_end = |p: &[u8]| -> Option<(usize, bool, bool)> {
                let mut r = BitReader::new(p);
                r.bits(params.tagref_bits).ok()?;
                r.bits(13).ok()?;
                let fa = r.flag().ok()?;
                if fa {
                    r.bits(10).ok()?;
                }
                r.bits(4).ok()?;
                let fb = r.flag().ok()?;
                if !fb {
                    r.bits(4).ok()?;
                }
                r.bits(7).ok()?;
                r.bits(32).ok()?;
                Some((r.pos(), fa, fb))
            };
            eprintln!(
                "\n=== {} : {} bipeds, tagref {} bits, occupied {:#06x} ===",
                path.file_name().unwrap_or_default().to_string_lossy(),
                have.len(),
                params.tagref_bits,
                occupied_slot_mask(&f.header.players)
            );
            let mut tab: std::collections::BTreeMap<(usize, bool, bool), Vec<u8>> = Default::default();
            for (id, _, _) in &have {
                let p = &creation[id];
                let Some((e, fa, fb)) = shared_end(p) else { continue };
                let o = creation_owner_slot(p).unwrap_or(255);
                tab.entry((e, fa, fb)).or_default().push(o);
            }
            for ((e, fa, fb), mut owners) in tab {
                owners.sort_unstable();
                let n = owners.len();
                owners.dedup();
                eprintln!("  shared header ends at bit {e:<4} (flagA {fa} flagB {fb}): {n:>3} bipeds, owner values {owners:?}");
            }
            // Sweep offsets RELATIVE to the shared header's end.
            let minlen = have.iter().map(|(i, _, _)| creation[i].len()).min().unwrap_or(0) * 8;
            let mut best: Vec<(f64, usize, u32, usize, usize)> = Vec::new();
            for k in 0..minlen.saturating_sub(4) {
                for width in 4..=5u32 {
                    let mut groups: std::collections::HashMap<u64, Vec<(u32, u32)>> = Default::default();
                    let mut ok = true;
                    for (id, a, b) in &have {
                        let p = &creation[id];
                        let Some((e, _, _)) = shared_end(p) else { ok = false; break };
                        let mut r = BitReader::new(p);
                        if r.skip(e + k).is_err() { ok = false; break }
                        match r.bits(width) {
                            Ok(v) => groups.entry(v).or_default().push((*a, *b)),
                            Err(_) => { ok = false; break }
                        }
                    }
                    if !ok || groups.len() < 2 {
                        continue;
                    }
                    let (mut pairs, mut bad) = (0usize, 0usize);
                    for v in groups.values() {
                        for i in 0..v.len() {
                            for j in i + 1..v.len() {
                                pairs += 1;
                                // the 2 s grace the real rule uses, so a lingering corpse is not
                                // counted as two bodies at once
                                if v[i].1 > v[j].0 + 120 && v[j].1 > v[i].0 + 120 {
                                    bad += 1;
                                }
                            }
                        }
                    }
                    if pairs == 0 {
                        continue;
                    }
                    best.push((1.0 - bad as f64 / pairs as f64, k, width, groups.len(), pairs));
                }
            }
            best.sort_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(b.3.cmp(&a.3)).then(b.4.cmp(&a.4))
            });
            eprintln!("  top relative-offset candidates (clean, groups, pairs):");
            for (clean, k, width, groups, pairs) in best.iter().take(12) {
                eprintln!("    +{k:<4} width {width}  clean {clean:.3}  groups {groups:<3} pairs {pairs}");
            }
        }
    }

    /// #catchall  THE ANATOMY OF AN UNNAMED BODY, over the whole corpus.
    ///
    /// For every film: each owner group's bodies, their lives, their biped DEFINITION class
    /// (`def_wire` -- the network tag reference the creation message carries, which separates
    /// Spartan from Elite from Monitor with no map needed), and whether the group's lives
    /// overlap. Next to it the header's own species census (`player_model_choice`), so the two
    /// independent statements can be counted against each other.
    ///
    /// The question it answers: when a body cannot be named from the owner stamp, is there a
    /// FREE header slot it could belong to at all?
    #[test]
    #[ignore = "probe: anatomy of the unnamed bodies"]
    fn probe_catchall_anatomy() {
        let mut tot_bodies = 0usize;
        let mut tot_unnamed = 0usize;
        let mut tot_elim = 0usize;
        for path in sample_films().iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let bipeds: Vec<&EntityTrack> =
                idx.tracks.iter().filter(|t| t.entity_type == 11).collect();
            if bipeds.is_empty() {
                continue;
            }
            // header census
            let mut elite_slots: Vec<u8> = Vec::new();
            let mut spartan_slots: Vec<u8> = Vec::new();
            for p in f.header.players.iter().filter(|p| p.present()) {
                match p.appearance().map(|a| a.model_choice) {
                    Some(1) => elite_slots.push(p.index as u8),
                    Some(_) => spartan_slots.push(p.index as u8),
                    None => {}
                }
            }
            // def_wire classes among bipeds
            let mut classes: std::collections::BTreeMap<u32, usize> = Default::default();
            for t in &bipeds {
                if let Some(w) = t.def_wire {
                    *classes.entry(w).or_default() += 1;
                }
            }
            eprintln!(
                "\n=== {} : {} bipeds, {} groups, header {} elite {} spartan, def classes {:?} ===",
                path.file_name().unwrap_or_default().to_string_lossy(),
                bipeds.len(),
                idx.owner_group_count(),
                elite_slots.len(),
                spartan_slots.len(),
                classes
            );
            let mut unnamed_here = 0usize;
            let mut claimed: Vec<u8> = Vec::new();
            for o in idx.owner_groups() {
                let mut v: Vec<&&EntityTrack> =
                    bipeds.iter().filter(|t| t.owner_raw == Some(o)).collect();
                v.sort_by_key(|t| t.first_tick);
                let mut overlaps = 0usize;
                for i in 0..v.len() {
                    for j in i + 1..v.len() {
                        if v[i].last_tick > v[j].first_tick + 120
                            && v[j].last_tick > v[i].first_tick + 120
                        {
                            overlaps += 1;
                        }
                    }
                }
                let slot = idx.owner_to_slot(o);
                if let Some(s) = slot {
                    claimed.push(s);
                }
                let cls: Vec<u32> = {
                    let mut c: Vec<u32> = v.iter().filter_map(|t| t.def_wire).collect();
                    c.sort_unstable();
                    c.dedup();
                    c
                };
                let said = slot
                    .and_then(|s| f.header.players.iter().find(|p| p.present() && p.index == s as usize))
                    .and_then(|p| p.appearance())
                    .map(|a| if a.model_choice == 1 { "elite" } else { "spartan" })
                    .unwrap_or("-");
                eprintln!(
                    "  owner {o:<3} -> slot {:<5} lives {:<3} overlapping pairs {overlaps:<3} classes {cls:?} header says {said}",
                    slot.map(|s| s.to_string()).unwrap_or_else(|| "none".into()),
                    v.len()
                );
                if slot.is_none() {
                    unnamed_here += v.len();
                    for t in &v {
                        eprintln!(
                            "      body {:#010x} ticks {:>6}..{:<6} ({:>6.1}s..{:<6.1}s) class {:?}",
                            t.net_id, t.first_tick, t.last_tick,
                            t.first_tick as f32 / 60.0, t.last_tick as f32 / 60.0, t.def_wire
                        );
                    }
                }
            }
            claimed.sort_unstable();
            let free: Vec<u8> = (0..16u8)
                .filter(|s| idx.slot_occupied(*s) && !claimed.contains(s))
                .collect();
            eprintln!("  claimed slots {claimed:?}; FREE occupied slots {free:?}; {unnamed_here} unnamed bodies");
            // Elimination: a free slot whose species class matches exactly one unnamed body's
            // class, with that body's lives not overlapping any other unnamed body of the class.
            tot_bodies += bipeds.len();
            tot_unnamed += unnamed_here;
            if free.len() == 1 && unnamed_here > 0 {
                tot_elim += 1;
            }
        }
        eprintln!("\nTOTAL {tot_bodies} bodies, {tot_unnamed} unnamed; {tot_elim} film(s) with exactly one free slot");
    }

    // ---------------------------------------------------------------------------------------
    // #catchall  the gates on the ELIMINATION bind
    // ---------------------------------------------------------------------------------------

    /// A fixture with a corroborated stamp rule and a catch-all group, so the elimination rule's
    /// preconditions can be switched on and off one at a time.
    ///
    /// Shape: 4 occupied slots (0..3), slots 0 and 3 are Elites. Owner stamps 1, 2, 3 name slots
    /// 0, 1, 2 -- so slot 3 is the only one left. Group 0 holds two bodies: one Elite-class
    /// (which only slot 3 can be) and one Spartan-class (which no free slot can be).
    #[cfg(test)]
    fn elimination_fixture() -> FilmIndex {
        const SPARTAN: u32 = 772;
        const ELITE: u32 = 778;
        let mut idx = FilmIndex::default();
        idx.ticks = 1200;
        idx.occupied_slots = 0b1111;
        idx.elite_slots = 0b1001; // slots 0 and 3 are Elites
        idx.player_aims = vec![Vec::new(); 16];
        for slot in 0..4u8 {
            for t in 0..1200u32 {
                idx.player_aims[slot as usize].push((t, PlayerAim { slot, ..Default::default() }));
            }
        }
        let mut mk = |net: u32, owner: u8, a: u32, b: u32, class: u32| {
            idx.tracks.push(EntityTrack {
                net_id: net,
                entity_type: 11,
                first_tick: a,
                last_tick: b,
                owner_raw: Some(owner),
                def_wire: Some(class),
                ..Default::default()
            });
        };
        // stamp 1 -> slot 0 (Elite), stamp 2 -> slot 1 (Spartan), stamp 3 -> slot 2 (Spartan)
        mk(0x101, 1, 0, 1200, ELITE);
        mk(0x102, 2, 0, 1200, SPARTAN);
        mk(0x103, 3, 0, 1200, SPARTAN);
        // the catch-all
        mk(0x200, 0, 100, 900, ELITE);
        mk(0x201, 0, 300, 1100, SPARTAN);
        idx.owner_rule_ok = true;
        idx.owner_anchor = OwnerAnchor::AuthorRecord;
        idx.bindings = vec![
            (0u8, 0x101u32, 1200u32),
            (1, 0x102, 1200),
            (2, 0x103, 1200),
        ]
        .into_iter()
        .map(|(slot, net_id, last_tick)| PlayerBinding {
            slot,
            net_id,
            entity_type: 11,
            first_tick: 0,
            last_tick,
            from_owner_rule: true,
            origin: BindOrigin::OwnerStamp,
            ..Default::default()
        })
        .collect();
        idx
    }

    /// #catchall  THE RULE, and every precondition it refuses on.
    ///
    /// This is the gate that has to bite: the mistake it exists to prevent is putting one
    /// gamertag on several people's bodies, which is what reading the four-bit wrap too eagerly
    /// did. So it checks the bind in the one case where it is FORCED and then breaks each
    /// precondition in turn and requires the name to disappear.
    #[test]
    fn elimination_names_the_forced_body_and_nothing_else() {
        const SPARTAN: u32 = 772;
        const ELITE: u32 = 778;

        // (a) FORCED: one free slot, one leftover body of that slot's species.
        let mut idx = elimination_fixture();
        idx.bind_by_elimination();
        assert_eq!(
            idx.slot_of_entity(0x200),
            Some(3),
            "the only Elite-class body left over, with slot 3 the only free Elite slot, is forced"
        );
        assert_eq!(
            idx.bindings.iter().find(|b| b.net_id == 0x200).map(|b| b.origin),
            Some(BindOrigin::SpeciesElimination),
            "a name worked out this way must SAY it was worked out"
        );
        // The Spartan body in the same catch-all has no free Spartan slot to belong to.
        assert_eq!(
            idx.slot_of_entity(0x201),
            None,
            "every Spartan slot is already claimed, so this body must stay unnamed"
        );

        // (b) TWO free slots of that species: ambiguous, so nobody.
        let mut idx = elimination_fixture();
        idx.bindings.retain(|b| b.slot != 0); // slot 0 (an Elite) is now free too
        idx.bind_by_elimination();
        assert_eq!(
            idx.slot_of_entity(0x200),
            None,
            "two unclaimed Elite slots: the body could be either, so it must be named neither"
        );

        // (c) TWO leftover bodies of that species that are ALIVE AT ONCE: several players, so
        // nobody. This is the catch-all failure itself, reproduced at the body level.
        let mut idx = elimination_fixture();
        idx.tracks.push(EntityTrack {
            net_id: 0x202,
            entity_type: 11,
            first_tick: 200,
            last_tick: 1000,
            owner_raw: Some(0),
            def_wire: Some(ELITE),
            ..Default::default()
        });
        idx.bind_by_elimination();
        assert_eq!(idx.slot_of_entity(0x200), None, "two Elite bodies alive together are two players");
        assert_eq!(idx.slot_of_entity(0x202), None, "...and neither of them may take slot 3");

        // (c2) ...but two leftover bodies of that species that follow one another ARE one
        // player's successive lives, and both get the name.
        let mut idx = elimination_fixture();
        idx.tracks.push(EntityTrack {
            net_id: 0x202,
            entity_type: 11,
            first_tick: 950,
            last_tick: 1150,
            owner_raw: Some(0),
            def_wire: Some(ELITE),
            ..Default::default()
        });
        idx.bind_by_elimination();
        assert_eq!(idx.slot_of_entity(0x200), Some(3), "a respawn chain is one player");
        assert_eq!(idx.slot_of_entity(0x202), Some(3), "...so both of its bodies are named");

        // (d) The class is NOT a species discriminator on this film -- a gametype that forces a
        // species puts Spartan-slot and Elite-slot players in the same class -- so the species
        // signal says nothing and nobody is named.
        let mut idx = elimination_fixture();
        for t in idx.tracks.iter_mut() {
            t.def_wire = Some(SPARTAN);
        }
        idx.bind_by_elimination();
        assert!(
            idx.bindings.iter().all(|b| b.origin != BindOrigin::SpeciesElimination),
            "one class covering both species carries no species information"
        );

        // (e) The stamp rule was never corroborated on this film, so "claimed" means nothing.
        let mut idx = elimination_fixture();
        idx.owner_rule_ok = false;
        idx.bind_by_elimination();
        assert_eq!(idx.slot_of_entity(0x200), None, "no corroborated stamp rule, no elimination");

        // (f) The caller supplied no species mask. Then every slot reads as a Spartan, both
        // definition classes collect Spartan votes, and the two catch-all bodies -- which ARE
        // alive together -- land in the same bucket and cancel each other out. The species
        // statement is what separates them, so without it nobody is named.
        let mut idx = elimination_fixture();
        idx.elite_slots = 0;
        idx.bind_by_elimination();
        assert!(
            idx.bindings.iter().all(|b| b.origin != BindOrigin::SpeciesElimination),
            "with no header species mask there is nothing to eliminate on"
        );

        // (g) The body falls OUTSIDE the free slot's own per-tick record range, so it cannot be
        // that player's.
        let mut idx = elimination_fixture();
        idx.player_aims[3].retain(|(t, _)| *t < 400);
        idx.bind_by_elimination();
        assert_eq!(
            idx.slot_of_entity(0x200),
            None,
            "slot 3 stopped reporting at tick 400 and the body lives to 900: not theirs"
        );
    }

    /// #catchall  ONE SLOT IS ONE PLAYER -- over every real film.
    ///
    /// The invariant the binding rests on, stated against what the recording actually does rather
    /// than against an idea of it. It is NOT "no two of a slot's bodies ever overlap", and it is
    /// not even "never three at once": a corpse and its ragdoll are entities too, so one player's
    /// tracks genuinely pile up. **Measured on this corpus** (the figures this test prints), the
    /// stamp rule's own groups reach three bodies at once -- slot 3 of the 16-player test clip
    /// has `0xECA80213` (2:11.7..6:09.4), `0xEE620219` (5:57.7..6:11.0) and a third life in the
    /// same window. So a flat "peak <= 2" would be asserting something false about the format,
    /// and the first version of this gate did exactly that.
    ///
    /// What IS asserted is the thing that distinguishes a real player from a catch-all, and the
    /// thing a per-body name could get wrong:
    ///
    /// * **an ELIMINATION bind never stacks.** The body it names must not coexist (beyond the 2 s
    ///   hand-over grace) with any other body bound to that same slot, and two elimination binds
    ///   for one slot must not coexist either. This is the assertion that would have caught the
    ///   mistake the brief warns about: on the test clip the catch-all's three Spartans are alive
    ///   together for about 90 s, so naming them all for one slot fails here immediately.
    /// * **no more gamertags than the header holds.**
    ///
    /// The peak simultaneity and the worst overlap are PRINTED for every film, because those are
    /// the numbers that set the rule's grace and a change in them should be visible.
    #[test]
    fn one_slot_is_one_player_over_real_films() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- one-slot-one-player gate skipped");
            return;
        }
        // 2 s at 60 Hz: the same grace `bind_by_elimination` uses.
        const GRACE: u32 = 120;
        let (mut checked, mut worst_overlap, mut worst_at_once, mut elim_checked) =
            (0usize, 0u32, 0usize, 0usize);
        let mut worst_where = String::new();
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let idx =
                build_index_players(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players);
            checked += 1;
            let mut film_peak: (i32, u8, usize, Vec<BindOrigin>) = (0, 0, 0, Vec::new());
            for slot in 0..16u8 {
                let mut lives: Vec<(u32, u32, u32, BindOrigin)> = idx
                    .bindings
                    .iter()
                    .filter(|b| b.slot == slot && b.is_confident())
                    .map(|b| (b.first_tick, b.last_tick, b.net_id, b.origin))
                    .collect();
                lives.sort_unstable_by_key(|l| (l.0, l.1, l.2));
                lives.dedup_by_key(|l| l.2);
                let coexist = |a: &(u32, u32, u32, BindOrigin), b: &(u32, u32, u32, BindOrigin)| {
                    a.1 > b.0 + GRACE && b.1 > a.0 + GRACE
                };
                for i in 0..lives.len() {
                    for j in i + 1..lives.len() {
                        let (a, b) = (&lives[i], &lives[j]);
                        let (lo, hi) = (a.0.max(b.0), a.1.min(b.1));
                        if hi > lo && hi - lo > worst_overlap {
                            worst_overlap = hi - lo;
                            worst_where =
                                format!("{name} slot {slot} {:#010x} x {:#010x}", a.2, b.2);
                        }
                        // THE ASSERTION: a per-body name must not land on a slot that already
                        // has a body at that moment, and two per-body names must not collide.
                        if a.3 == BindOrigin::SpeciesElimination
                            || b.3 == BindOrigin::SpeciesElimination
                        {
                            elim_checked += 1;
                            assert!(
                                !coexist(a, b),
                                "{name}: slot {slot} is bound to {:#010x} ({}..{}, {:?}) and \
                                 {:#010x} ({}..{}, {:?}) AT THE SAME TIME, and at least one of \
                                 those names came from elimination -- one player drives one biped",
                                a.2, a.0, a.1, a.3, b.2, b.0, b.1, b.3
                            );
                        }
                    }
                }
                // How many of this slot's bodies are alive at the same moment, at its worst?
                let mut edges: Vec<(u32, i32)> = Vec::new();
                for (a, b, _, _) in &lives {
                    edges.push((*a, 1));
                    edges.push((*b + 1, -1));
                }
                edges.sort_unstable();
                let (mut cur, mut peak) = (0i32, 0i32);
                for (_, d) in edges {
                    cur += d;
                    peak = peak.max(cur);
                }
                worst_at_once = worst_at_once.max(peak as usize);
                if peak > film_peak.0 {
                    film_peak = (peak, slot, lives.len(), lives.iter().map(|l| l.3).collect());
                }
                // THE ASSERTION that bites on real data: no gamertag may be on more bodies at
                // once than one player can leave behind. The threshold and the table it comes
                // from are in `FilmIndex::owner_group_is_one_player`; before that guard existed
                // this corpus had slots holding 4, 7 and 13 bodies at the same moment.
                assert!(
                    peak <= 3,
                    "{name}: slot {slot} is bound to {peak} bodies at the same moment -- a player \
                     can leave a corpse and a ragdoll behind, not {} bodies: {lives:?}",
                    peak - 1
                );
            }
            // ...and no more names than there are players to give them.
            let named: std::collections::HashSet<u8> =
                idx.bindings.iter().filter(|b| b.is_confident()).map(|b| b.slot).collect();
            let occupied = occupied_slot_mask(&f.header.players).count_ones() as usize;
            if occupied > 0 {
                assert!(
                    named.len() <= occupied,
                    "{name}: {} slots named but only {occupied} occupied",
                    named.len()
                );
            }
            // PRINTED, per film: the worst "one gamertag, N bodies at once" this reader produces
            // and which rule produced it. This is the number that says whether a group is one
            // player or a catch-all, so it belongs in the record rather than in a comment.
            let mut origins: Vec<BindOrigin> = film_peak.3.clone();
            origins.sort_by_key(|o| format!("{o:?}"));
            origins.dedup();
            eprintln!(
                "film_decode: {name:<46} worst slot {} holds {} bod(ies), {} alive at once, by {origins:?}",
                film_peak.1, film_peak.2, film_peak.0
            );
        }
        eprintln!(
            "film_decode: one-slot-one-player held on {checked} film(s); {elim_checked} \
             elimination-vs-other body pair(s) checked for coexistence; worst same-slot overlap \
             {worst_overlap} ticks ({:.1} s) at {worst_where}; most bodies of one slot alive at \
             once {worst_at_once}",
            worst_overlap as f32 / 60.0
        );
    }

    /// #catchall  EVERY NAMED BODY IS THE RIGHT SPECIES -- over every real film.
    ///
    /// Two independent statements about one player: the header's own `player_model_choice`
    /// (+0x4C) and the biped DEFINITION the stream names in the body's creation message. They
    /// must agree -- a body dressed from one species' record while drawn as the other is exactly
    /// how armour goes missing -- and the agreement is the arithmetic that forces the elimination
    /// bind in the first place.
    ///
    /// The class-to-species map is not assumed: it is learned per film from the bodies the OWNER
    /// STAMP named. A film where some class collects BOTH kinds of vote is skipped and said to be
    /// skipped: a gametype can force a species, which makes the classes carry no species
    /// information at all. On a film where the map IS clean, every confident binding must agree
    /// with it -- the stamped ones as well as the derived ones, so a future change to either rule
    /// that crosses the species line fails here.
    #[test]
    fn every_named_body_agrees_with_its_slots_species() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- species agreement gate skipped");
            return;
        }
        let (mut clean, mut mixed, mut bodies) = (0usize, 0usize, 0usize);
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let idx =
                build_index_players(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players);
            let elite = elite_slot_mask(&f.header.players);
            // learn the map from the STAMPED bodies only
            let mut votes: std::collections::BTreeMap<u32, (usize, usize)> = Default::default();
            for b in idx.bindings.iter().filter(|b| b.is_confident() && b.from_owner_rule) {
                let Some(w) =
                    idx.tracks.iter().find(|t| t.net_id == b.net_id).and_then(|t| t.def_wire)
                else {
                    continue;
                };
                let e = votes.entry(w).or_default();
                if elite & (1 << b.slot) != 0 {
                    e.1 += 1;
                } else {
                    e.0 += 1;
                }
            }
            if votes.is_empty() {
                continue;
            }
            if votes.values().any(|(sp, el)| *sp > 0 && *el > 0) {
                mixed += 1;
                eprintln!(
                    "film_decode: {name:<46} definition classes carry BOTH species ({votes:?}) -- \
                     no species information here, skipped"
                );
                continue;
            }
            clean += 1;
            // The whole-clip census that forces the bind: per class, how many bodies exist and
            // how many the stamp already accounted for.
            let mut total: std::collections::BTreeMap<u32, usize> = Default::default();
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                if let Some(w) = t.def_wire {
                    *total.entry(w).or_default() += 1;
                }
            }
            let census: Vec<String> = votes
                .iter()
                .map(|(w, (sp, el))| {
                    format!(
                        "class {w}: {} in clip, {} stamped ({})",
                        total.get(w).copied().unwrap_or(0),
                        sp + el,
                        if *el > 0 { "Elite" } else { "Spartan" }
                    )
                })
                .collect();
            eprintln!("film_decode: {name:<46} {}", census.join("; "));
            for b in idx.bindings.iter().filter(|b| b.is_confident()) {
                let Some(w) =
                    idx.tracks.iter().find(|t| t.net_id == b.net_id).and_then(|t| t.def_wire)
                else {
                    continue;
                };
                let Some((sp, el)) = votes.get(&w) else { continue };
                let want = *el > 0 && *sp == 0;
                bodies += 1;
                assert_eq!(
                    want,
                    elite & (1 << b.slot) != 0,
                    "{name}: body {:#010x} is {} by its definition class but was named for slot {} \
                     whose header record says {} ({:?})",
                    b.net_id,
                    if want { "Elite" } else { "Spartan" },
                    b.slot,
                    if elite & (1 << b.slot) != 0 { "Elite" } else { "Spartan" },
                    b.origin
                );
            }
        }
        eprintln!(
            "film_decode: {bodies} named bod(ies) agree with their slot's header species across \
             {clean} film(s) whose definition classes separate the species; {mixed} film(s) \
             carried no species information and were skipped"
        );
    }

    /// #player-unit  THERE IS NO CEILING -- the stamp can name all SIXTEEN slots -- and
    /// elimination still only fills gaps.
    ///
    /// **This test used to assert the opposite**, and it was right to fail when the field was
    /// read correctly. Its premise was `write_bits(field + 1, 4)`: four bits hold `index + 1`
    /// only for `index` 0..14, so "the stamp can never name slot 15". The premise was wrong
    /// because the field is **five** bits (`state[0x3F] = ReadBits(5) - 1`), and the four-bit
    /// reader HMS used was taking its low four bits -- which sent wire 16, slot 15, to wire 0,
    /// "nobody". Measured on `*21897D8A*` after the fix: wire 16 appears on **2** bodies and
    /// maps to slot 15, and the film goes from 107 named bodies to 108, with every one of its
    /// 108 player bodies named.
    ///
    /// So what is asserted now is the part that survived: **an elimination bind only ever lands
    /// on a slot no stamp named.** It fills gaps, it never competes. That fails if elimination
    /// ever contradicts the stamp. Slot 15 being nameable is asserted the other way round --
    /// nothing may re-impose a fifteen-slot ceiling.
    #[test]
    fn the_stamp_can_name_all_sixteen_slots_and_elimination_fills_the_gap() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- four-bit ceiling gate skipped");
            return;
        }
        let (mut checked, mut full, mut filled) = (0usize, 0usize, 0usize);
        let mut named_fifteen = 0usize;
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let idx = build_index_players(
                &f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players,
            );
            checked += 1;
            let stamped: std::collections::HashSet<u8> = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.from_owner_rule)
                .map(|b| b.slot)
                .collect();
            assert!(
                stamped.len() <= 16,
                "{name}: the owner stamp named {} slots; there are only 16",
                stamped.len()
            );
            if stamped.contains(&15) {
                named_fifteen += 1;
            }
            if f.header.players.iter().filter(|p| p.present()).count() >= 16 {
                full += 1;
            }
            for b in idx.bindings.iter().filter(|b| {
                b.is_confident() && b.origin == BindOrigin::SpeciesElimination
            }) {
                assert!(
                    !stamped.contains(&b.slot),
                    "{name}: elimination named slot {} for body {:#010x}, but the owner stamp \
                     already named that slot -- elimination fills the ceiling's gap, it does not \
                     compete",
                    b.slot,
                    b.net_id
                );
                filled += 1;
            }
        }
        eprintln!(
            "film_decode: the stamp named at most 16 slots on {checked} film(s); it named SLOT 15 \
             on {named_fifteen} of them (which a four-bit read could never do); {full} occupy all \
             16 header slots, and elimination filled {filled} further bod(ies)"
        );
        // The five-bit read's whole point: slot 15 is reachable. If no film reaches it, either
        // the corpus genuinely has no sixteenth player with a body or a four-bit read is back.
        if full > 0 {
            assert!(
                named_fifteen > 0,
                "{full} film(s) occupy all 16 header slots and NOT ONE had slot 15 named by the \
                 stamp -- that is the signature of the field being read four bits wide again, \
                 which aliases wire 16 to wire 0 ('nobody')"
            );
        }
    }

    /// #catchall  WHAT THE CHANGE IS WORTH, over the whole corpus. Prints; asserts only that it
    /// never names a body twice.
    #[test]
    #[ignore = "measurement: how many previously-unnamed bodies elimination names"]
    fn measure_elimination_coverage() {
        let films = sample_films();
        let (mut tot_bodies, mut tot_named, mut tot_elim, mut tot_left) = (0usize, 0usize, 0usize, 0usize);
        for path in films.iter() {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx =
                build_index_players(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players);
            let bipeds = idx.tracks.iter().filter(|t| t.entity_type == 11).count();
            let named: std::collections::HashSet<u32> =
                idx.bindings.iter().filter(|b| b.is_confident()).map(|b| b.net_id).collect();
            let elim: Vec<&PlayerBinding> = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.origin == BindOrigin::SpeciesElimination)
                .collect();
            tot_bodies += bipeds;
            tot_named += named.len();
            tot_elim += elim.len();
            tot_left += bipeds - named.len();
            if !elim.is_empty() || bipeds != named.len() {
                eprintln!(
                    "{:<46} {bipeds:>4} bodies, {:>4} named ({} by elimination), {:>3} still unnamed",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    named.len(),
                    elim.len(),
                    bipeds - named.len()
                );
                for b in &elim {
                    let tag = f
                        .header
                        .players
                        .iter()
                        .find(|p| p.present() && p.index == b.slot as usize)
                        .map(|p| p.gamertag.clone())
                        .unwrap_or_default();
                    eprintln!(
                        "      ELIMINATION body {:#010x} ticks {}..{} -> slot {} {tag}",
                        b.net_id, b.first_tick, b.last_tick, b.slot
                    );
                }
                // every unnamed body, by what it IS -- plus the TEMPORAL elimination: which
                // occupied slots have NO body of their own anywhere across this body's life?
                // If exactly one, and its species matches, the match would be forced in time
                // the same way the whole-film one is forced in count.
                let elite = elite_slot_mask(&f.header.players);
                let mut votes: std::collections::HashMap<u32, (usize, usize)> = Default::default();
                for b in idx.bindings.iter().filter(|b| b.is_confident() && b.from_owner_rule) {
                    let Some(w) =
                        idx.tracks.iter().find(|t| t.net_id == b.net_id).and_then(|t| t.def_wire)
                    else {
                        continue;
                    };
                    let e = votes.entry(w).or_default();
                    if elite & (1 << b.slot) != 0 { e.1 += 1 } else { e.0 += 1 }
                }
                let class_elite = |w: u32| match votes.get(&w) {
                    Some((0, e)) if *e > 0 => Some(true),
                    Some((s, 0)) if *s > 0 => Some(false),
                    _ => None,
                };
                for t in idx.tracks.iter().filter(|t| t.entity_type == 11 && !named.contains(&t.net_id)) {
                    let want = t.def_wire.and_then(class_elite);
                    let gaps: Vec<u8> = (0..16u8)
                        .filter(|s| idx.slot_occupied(*s))
                        .filter(|s| want.map_or(true, |w| (elite & (1 << *s) != 0) == w))
                        .filter(|s| {
                            !idx.bindings.iter().any(|b| {
                                b.slot == *s
                                    && b.is_confident()
                                    && b.last_tick >= t.first_tick
                                    && b.first_tick <= t.last_tick
                            })
                        })
                        .collect();
                    eprintln!(
                        "      unnamed body {:#010x} owner {:?} class {:?} ({}) ticks {}..{} ({:.1}s..{:.1}s)  slots with a full gap here: {gaps:?}",
                        t.net_id, t.owner_raw, t.def_wire,
                        match want { Some(true) => "elite", Some(false) => "spartan", None => "class unknown" },
                        t.first_tick, t.last_tick,
                        t.first_tick as f32 / 60.0, t.last_tick as f32 / 60.0
                    );
                }
            }
        }
        eprintln!(
            "\nTOTAL {tot_bodies} bodies: {tot_named} named ({tot_elim} of them by ELIMINATION), {tot_left} unnamed"
        );
    }

}

/// #stamp15  WHAT WIRE 0 ACTUALLY CONTAINS — the measurement that killed a plausible fix.
///
/// The owner stamp is `slot + 1` in four bits, so `15 + 1 = 16` overflows to 0, the same wire
/// value the engine's `-1` ("nobody") produces. That is true, and it suggested an appealing
/// rule: on a sixteen-player film the stamp-0 group of bipeds must BE slot 15, so bind the whole
/// group to it. That rule was written, gated six ways, and then **measured against the real
/// sixteen-player clip, where it refused to fire for two independent reasons** — both of which
/// say the rule was wrong rather than that the film was odd:
///
/// 1. **Slot 15 was already claimed.** Every one of the sixteen slots was bound, because the
///    species-elimination path ([`FilmIndex::bind_by_elimination`]) had already named it. There
///    was no leftover slot to recover; the machinery for this already existed.
/// 2. **The stamp-0 group had FOUR bodies alive at once.** One player drives one biped, and the
///    measured allowance for a live body plus lingering corpses is three. Four means the group is
///    not one player's, so binding it as one would have put a gamertag on bodies belonging to
///    somebody else — the exact failure an earlier round already made once, when sixteen bodies
///    were bound to one slot with thirteen alive simultaneously.
///
/// So wire 0 is a **mixing bucket**: genuinely unowned bipeds AND any player the field cannot
/// encode both land in it, and only a PER-BODY test can separate them. That is what elimination
/// does, and it is why it is deliberately conservative. The whole-group rule was deleted rather
/// than left in place to never fire, because dead code with passing synthetic gates is worse than
/// no code: the gates prove the rule's shape, not that it does anything.
///
/// What remains is this gate, which asserts the two properties above on real data, so the next
/// person who has the same good idea finds the measurement instead of repeating the work. It also
/// guards [`FilmIndex::owner_group_is_named`], which is the part that WAS wrong and is now right:
/// a reader must not call a group nameless just because `slot_of_owner` declines wire 0, when
/// bodies in that group are being named through the bindings and drawn with a gamertag.
#[cfg(test)]
mod stamp_zero_tests {
    use super::*;

    /// `HMS_FILM=<a 16-player film> cargo test -p hms-app wire_zero -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn wire_zero_is_a_mixing_bucket_not_one_player() {
        let Ok(path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM to a sixteen-player Reach film");
            return;
        };
        let f = crate::film::open(std::path::Path::new(&path)).expect("film opens");
        let present = f.present_players().count();
        let idx = build_index_players(
            &f.bytes,
            &f.data,
            &f.header.map_variant,
            f.recorder_slot(),
            &f.header.players,
        );

        let claimed: std::collections::HashSet<u8> =
            idx.bindings.iter().filter(|b| b.is_confident()).map(|b| b.slot).collect();
        let zero: Vec<&EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw == Some(0)).collect();
        let mut edges: Vec<(u32, i32)> = Vec::new();
        for t in &zero {
            edges.push((t.first_tick, 1));
            edges.push((t.last_tick + 1, -1));
        }
        edges.sort_unstable();
        let (mut cur, mut peak) = (0i32, 0i32);
        for (_, d) in edges {
            cur += d;
            peak = peak.max(cur);
        }
        let named_zero =
            zero.iter().filter(|t| idx.slot_of_entity(t.net_id).is_some()).count();
        println!(
            "  {present} present player(s); {} of 16 slot(s) claimed; stamp-0 group: {} biped(s), \
             peak {peak} alive at once, {named_zero} of them named through the bindings",
            claimed.len(),
            zero.len()
        );

        if present < 16 || zero.is_empty() {
            println!("  (not a full lobby with a stamp-0 group; nothing to assert)");
            return;
        }
        assert!(
            peak > 1,
            "the whole point is that wire 0 holds MORE than one body at a time on a full lobby; \
             if it ever holds at most one, the whole-group rule deleted here becomes viable again \
             and this note should be revisited"
        );
        assert!(
            claimed.contains(&15) || named_zero > 0,
            "slot 15 must be reachable by SOME route on a full lobby -- elimination claiming the \
             slot, or a stamp-0 body being named through the bindings. Neither happening means \
             the sixteenth player is unnamed again and `owner_group_is_named` will under-report."
        );
        // The reporting helper must agree with what is actually drawn: if any stamp-0 body is
        // named, the group is named, whatever `slot_of_owner` says about wire 0.
        if named_zero > 0 {
            assert!(
                idx.owner_group_is_named(0),
                "a stamp-0 body is being named through the bindings, so the group must not be \
                 reported as nameless -- that mismatch is what made the roster say a player could \
                 not be named while their gamertag was drawn over their head"
            );
            assert!(
                idx.slot_of_owner(0).is_none(),
                "and wire 0 must still refuse to answer on its own, which is why the helper exists"
            );
        }
    }
}

/// #naming  HOW MANY BIPEDS END UP WITH A NAME, and what the unnamed ones look like.
///
/// The user's report is that bipeds walk around unnamed while the header holds every gamertag.
/// This measures it end to end rather than reasoning about the stamp.
///
/// The key column is **peak(LIVES)** against **peak(track extent)**.
/// [`FilmIndex::owner_group_is_one_player`] sweeps `first_tick ..= last_tick`, a track's FULL
/// EXTENT — but a corpse and its ragdoll linger past the respawn, so a player who dies and
/// respawns quickly shows several overlapping TRACKS while only ever one body is ALIVE. Calling
/// such a group "several people" off the extent measure would be wrong, and that is exactly the
/// judgement that decides whether a stamp can be named.
#[cfg(test)]
mod naming_coverage {
    use super::*;

    #[test]
    #[ignore]
    fn how_many_bipeds_get_a_name() {
        let Ok(path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM");
            return;
        };
        let f = crate::film::open(std::path::Path::new(&path)).expect("film opens");
        let present = f.present_players().count();
        let idx = build_index_players(
            &f.bytes,
            &f.data,
            &f.header.map_variant,
            f.recorder_slot(),
            &f.header.players,
        );
        let pb = crate::film_play::Playback::new(idx);
        let idx = &pb.index;
        let bipeds: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == 11).collect();
        let named = bipeds.iter().filter(|t| idx.slot_of_entity(t.net_id).is_some()).count();
        println!(
            "\n  {present} header player(s) present | {} biped track(s) | {named} NAMED | {} unnamed",
            bipeds.len(),
            bipeds.len() - named
        );
        // The user's bar is about PLAYERS' bodies, and the stream says which those are: only a
        // player's biped creation carries an appearance block (armour and colours). A gametype
        // or AI spawn has none, and nobody expects a name over it.
        let players: Vec<&&EntityTrack> = bipeds.iter().filter(|t| t.appearance).collect();
        let pnamed = players.iter().filter(|t| idx.slot_of_entity(t.net_id).is_some()).count();
        println!(
            "  of those, {} carry an APPEARANCE block (a player's body): {pnamed} NAMED, {} unnamed; \
             {} have none (gametype/AI spawns)",
            players.len(),
            players.len() - pnamed,
            bipeds.len() - players.len()
        );
        println!(
            "  eqip width {} closed {:.1}% of the biped creations",
            idx.eqip_bits,
            idx.creations_closed * 100.0
        );
        // #appearance  The two armour routes, counted apart: the file's own roster, and the
        // weaker cross-recording inference.
        println!(
            "  named by ARMOUR from this film's roster: {} | INFERRED from another recording's \
             roster: {} | neither: {}",
            idx.appearance_named,
            idx.appearance_cross_named,
            bipeds
                .iter()
                .filter(|t| {
                    idx.slot_of_entity(t.net_id).is_none() && idx.inferred_name(t.net_id).is_none()
                })
                .count()
        );
        for (net, tag, src) in idx.cross_film_names.iter().take(4) {
            println!("    body {net:#010x} -> {tag:?} (from {src})");
        }

        let mut stamps: std::collections::BTreeMap<i16, Vec<&EntityTrack>> = Default::default();
        for t in &bipeds {
            stamps.entry(t.owner_raw.map(|o| o as i16).unwrap_or(-1)).or_default().push(t);
        }
        println!("  stamp  bodies  named  peak(extent)  peak(LIVES)  moved  slot");
        for (stamp, ts) in &stamps {
            let n = ts.iter().filter(|t| idx.slot_of_entity(t.net_id).is_some()).count();
            let peak = |use_death: bool| -> i32 {
                let mut e: Vec<(u32, i32)> = Vec::new();
                for t in ts {
                    let end = if use_death {
                        pb.death_at(t).unwrap_or(t.last_tick)
                    } else {
                        t.last_tick
                    };
                    e.push((t.first_tick, 1));
                    e.push((end.max(t.first_tick) + 1, -1));
                }
                e.sort_unstable();
                let (mut c, mut p) = (0, 0);
                for (_, d) in e {
                    c += d;
                    p = p.max(c);
                }
                p
            };
            let moved = ts.iter().filter(|t| t.moves_cached == Some(true)).count();
            let slot = if *stamp >= 0 {
                idx.slot_of_owner(*stamp as u8).map(|s| s.to_string()).unwrap_or_else(|| "-".into())
            } else {
                "-".into()
            };
            println!(
                "  {stamp:>5}  {:>6}  {n:>5}  {:>12}  {:>11}  {moved:>5}  {slot:>4}",
                ts.len(),
                peak(false),
                peak(true)
            );
        }
    }
}

/// #appearance  An appearance -> gamertag key built from EVERY film on the machine.
///
/// Keeps **only** appearances that map to exactly one gamertag across the whole corpus, so a
/// lookup that answers is unambiguous by construction. Measured here: 171 roster entries over
/// 21 films collapse to 88 distinct appearances, of which **1** is ambiguous and dropped.
pub struct AppearanceKey {
    map: std::collections::HashMap<(u8, [i8; 3], u8, [i8; 10]), (String, String)>,
    /// Roster entries scanned.
    pub entries: usize,
    /// Distinct appearances seen.
    pub distinct: usize,
    /// Appearances dropped because two or more gamertags share them.
    pub ambiguous: usize,
}

impl AppearanceKey {
    /// Build it by reading the HEADER of every film in the corpus. Header-only reads, so this is
    /// cheap relative to decoding a film.
    pub fn from_corpus() -> AppearanceKey {
        Self::from_films(&crate::film::spec_corpus())
    }

    /// As `from_corpus`, over a given list -- which is what the gates use so they can be exact.
    pub fn from_films(films: &[std::path::PathBuf]) -> AppearanceKey {
        // appearance -> (gamertags seen, first source film)
        let mut seen: std::collections::HashMap<
            (u8, [i8; 3], u8, [i8; 10]),
            (std::collections::BTreeSet<String>, String),
        > = Default::default();
        let mut entries = 0usize;
        for path in films {
            let Ok(f) = crate::film::open_header(path) else { continue };
            let src = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            for p in f.header.players.iter().filter(|p| p.present()) {
                let Some(a) = p.appearance() else { continue };
                if p.gamertag.is_empty() {
                    continue;
                }
                // The DEFAULT appearance identifies nobody, so it must not enter the key either
                // -- otherwise cross-film inference would name every uncustomised body after
                // whichever uncustomised player happened to be unique corpus-wide.
                if a.f48 == 0
                    && a.model_choice == 0
                    && a.color_primary == 0
                    && a.color_secondary == 0
                    && a.f4b == 0
                    && a.slots.iter().all(|v| *v == 0)
                {
                    continue;
                }
                let mut slots = [0i8; 10];
                for (i, v) in a.slots.iter().enumerate() {
                    slots[i] = *v as i8;
                }
                let k = (
                    a.f48,
                    [a.color_primary as i8, a.color_secondary as i8, a.f4b as i8],
                    a.model_choice,
                    slots,
                );
                entries += 1;
                let e = seen.entry(k).or_insert_with(|| (Default::default(), src.clone()));
                e.0.insert(p.gamertag.clone());
            }
        }
        let distinct = seen.len();
        let ambiguous = seen.values().filter(|(g, _)| g.len() > 1).count();
        // Condition 2: only the unambiguous ones survive into the key.
        let map = seen
            .into_iter()
            .filter(|(_, (g, _))| g.len() == 1)
            .map(|(k, (g, src))| (k, (g.into_iter().next().unwrap_or_default(), src)))
            .collect();
        AppearanceKey { map, entries, distinct, ambiguous }
    }

    /// The one gamertag this appearance belongs to, and the film it was read from.
    pub fn lookup(&self, l: &BipedAppearance) -> Option<(&str, &str)> {
        let mut ten = [0i8; 10];
        ten.copy_from_slice(&l.slots[..10]);
        self.map
            .get(&(l.flags, l.colors, l.model_choice, ten))
            .map(|(tag, src)| (tag.as_str(), src.as_str()))
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The corpus key, built **once per process**.
    ///
    /// This is why the cross-film pass can be automatic without putting a filesystem walk in a
    /// hot path: `build_index_players` is called in a loop by several gates, and with this they
    /// share one scan. Making the pass caller-driven instead was tried and was worse -- nothing
    /// called it, so the route silently did nothing (`INFERRED: 0` on the one clip it exists
    /// for).
    pub fn shared() -> &'static AppearanceKey {
        static KEY: std::sync::OnceLock<AppearanceKey> = std::sync::OnceLock::new();
        KEY.get_or_init(AppearanceKey::from_corpus)
    }
}

/// #naming  WHO DOES THE STREAM ITSELF SAY IS CONTROLLING EACH BODY?
///
/// The creation stamp (`CREATION_OWNER_BIT`) is one statement of ownership and it is the only one
/// HMS has ever used. The unit's DERIVED block carries a second, independent one: field `0x2F`,
/// whose first sub-field is a 4-bit player index the engine hands straight to its player
/// collection. This probe measures both on the same films and cross-tabulates them, and applies
/// the SAME structural oracle that found the stamp in the first place -- a field that really is a
/// per-player index puts few bodies alive at once under one value.
#[cfg(test)]
mod controlling_player_probe {
    use super::*;
    use crate::film;

    fn probe_films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        crate::film::spec_corpus()
    }

    /// Peak number of spans alive at once, over a set of `(first, last)` tick ranges.
    fn peak_concurrent(spans: &[(u32, u32)]) -> i32 {
        let mut e: Vec<(u32, i32)> = Vec::with_capacity(spans.len() * 2);
        for &(a, b) in spans {
            e.push((a, 1));
            e.push((b.max(a) + 1, -1));
        }
        e.sort_unstable();
        let (mut c, mut p) = (0i32, 0i32);
        for (_, d) in e {
            c += d;
            p = p.max(c);
        }
        p
    }

    #[test]
    #[ignore = "probe: field 0x2F as the player who controls a body"]
    fn probe_controlling_player() {
        for path in probe_films() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            // per biped: tick span, creation stamp, and the histogram of 0x2F sub-field 0
            let mut span: std::collections::HashMap<u32, (u32, u32)> = Default::default();
            let mut stamp: std::collections::HashMap<u32, u8> = Default::default();
            let mut ctrl: std::collections::HashMap<u32, std::collections::BTreeMap<u8, usize>> =
                Default::default();
            // the weapon side, in the same pass: slot refs and the selection sub-fields
            let mut wslots: std::collections::HashMap<u32, std::collections::BTreeMap<(usize, u32), usize>> =
                Default::default();
            let mut wsel: std::collections::HashMap<u32, std::collections::BTreeMap<(i8, i8, i8), usize>> =
                Default::default();
            let (mut msgs, mut trust, mut with2f) = (0usize, 0usize, 0usize);
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                let tick = i as u32;
                for e in &t.entities {
                    if e.entity_type != 11 {
                        continue;
                    }
                    let s = span.entry(e.net_id).or_insert((tick, tick));
                    s.1 = tick;
                    if !e.is_update {
                        if let Some(o) = creation_owner_slot(&e.payload) {
                            stamp.insert(e.net_id, o);
                        }
                        continue;
                    }
                    msgs += 1;
                    let Some(u) = crate::film_anim::unit::decode(e, &params, WEAPREF_BITS) else {
                        continue;
                    };
                    if !u.trustworthy(e.dirty_mask) {
                        continue;
                    }
                    trust += 1;
                    if let Some(c) = u.controlling_player {
                        with2f += 1;
                        *ctrl.entry(e.net_id).or_default().entry(c).or_default() += 1;
                    }
                    for (k, s) in u.weapon_slots.iter().enumerate() {
                        if let Some(v) = s {
                            *wslots.entry(e.net_id).or_default().entry((k, *v)).or_default() += 1;
                        }
                    }
                    if let Some(sel) = u.weapon_sel {
                        let key = (
                            sel,
                            u.weapon_sel_ab[0].unwrap_or(-1),
                            u.weapon_sel_ab[1].unwrap_or(-1),
                        );
                        *wsel.entry(e.net_id).or_default().entry(key).or_default() += 1;
                    }
                }
            }
            let bipeds: Vec<u32> = span.keys().copied().collect();
            println!(
                "\n=== {} ===\n  {} present | {} bipeds | {msgs} unit updates, {trust} trustworthy, \
                 {with2f} carrying field 0x2F | {} bodies named by 0x2F",
                path.file_name().unwrap_or_default().to_string_lossy(),
                f.present_players().count(),
                bipeds.len(),
                ctrl.len()
            );
            for p in f.present_players() {
                println!("    header slot {:>2} team {} {:?}", p.index, p.team, p.gamertag);
            }
            // consistency: does one body ever report two different controlling players?
            let mixed = ctrl.values().filter(|h| h.len() > 1).count();
            println!("  bodies whose 0x2F value is NOT constant: {mixed} of {}", ctrl.len());
            // group by value and apply the structural oracle
            let mut groups: std::collections::BTreeMap<u8, Vec<(u32, u32)>> = Default::default();
            for (id, h) in &ctrl {
                let (v, _) = h.iter().max_by_key(|(_, n)| **n).unwrap();
                groups.entry(*v).or_default().push(span[id]);
            }
            println!("  0x2F value  bodies  peak(alive)");
            for (v, spans) in &groups {
                println!("  {v:>10}  {:>6}  {:>11}", spans.len(), peak_concurrent(spans));
            }
            // and the stamp, for comparison
            let mut sgroups: std::collections::BTreeMap<i16, Vec<(u32, u32)>> = Default::default();
            for id in &bipeds {
                let k = stamp.get(id).map(|v| *v as i16).unwrap_or(-1);
                sgroups.entry(k).or_default().push(span[id]);
            }
            println!("  stamp       bodies  peak(alive)");
            for (v, spans) in &sgroups {
                println!("  {v:>10}  {:>6}  {:>11}", spans.len(), peak_concurrent(spans));
            }
            // cross-tab: stamp vs 0x2F, per body
            let mut cross: std::collections::BTreeMap<(i16, i16), usize> = Default::default();
            for id in &bipeds {
                let s = stamp.get(id).map(|v| *v as i16).unwrap_or(-1);
                let c = ctrl
                    .get(id)
                    .and_then(|h| h.iter().max_by_key(|(_, n)| **n).map(|(v, _)| *v as i16))
                    .unwrap_or(-1);
                *cross.entry((s, c)).or_default() += 1;
            }
            println!("  (stamp, 0x2F) -> bodies");
            for ((s, c), n) in &cross {
                println!("    ({s:>3}, {c:>3}) -> {n}");
            }
            // THE WEAPON SIDE
            let two_slots = wslots.values().filter(|m| {
                m.keys().map(|(k, _)| *k).collect::<std::collections::BTreeSet<_>>().len() > 1
            }).count();
            let two_refs_slot0 = wslots
                .values()
                .filter(|m| m.keys().filter(|(k, _)| *k == 0).count() > 1)
                .count();
            println!(
                "  weapons: {} bodies report a slot ref; {two_slots} report MORE THAN ONE SLOT; \
                 {two_refs_slot0} report two different refs in slot 0",
                wslots.len()
            );
            let mut shown = 0;
            for (id, m) in &wslots {
                let slots: std::collections::BTreeSet<usize> = m.keys().map(|(k, _)| *k).collect();
                if slots.len() < 2 {
                    continue;
                }
                println!("    body {id:#010x} slot refs {:?}  sel {:?}", m, wsel.get(id));
                shown += 1;
                if shown >= 8 {
                    break;
                }
            }
            let selvals: std::collections::BTreeMap<(i8, i8, i8), usize> =
                wsel.values().flatten().fold(Default::default(), |mut a, (k, n)| {
                    *a.entry(*k).or_default() += n;
                    a
                });
            println!("  0x29 (sel, a, b) histogram: {selvals:?}");
        }
    }
}

/// #naming  DOES THE WALKED CREATION PAYLOAD CLOSE, AND WHAT OWNER DOES IT STATE?
///
/// The old fixed offset ([`CREATION_OWNER_BIT`]) has no oracle of its own: it reads four bits
/// from wherever and cannot fail. [`creation_biped`] walks to the END of the payload, so it is
/// checkable -- a correct walk leaves 0..=7 bits. This prints, per film: the solved `'eqip'`
/// width and how much of the corpus it closes, the owner distribution the WALK gives against the
/// one the fixed offset gives, and the structural oracle (bodies and peak concurrency per owner)
/// for both.
#[cfg(test)]
mod creation_walk_probe {
    use super::*;
    use crate::film;

    fn probe_films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        crate::film::spec_corpus()
    }

    fn peak(spans: &[(u32, u32)]) -> i32 {
        let mut e: Vec<(u32, i32)> = Vec::with_capacity(spans.len() * 2);
        for &(a, b) in spans {
            e.push((a, 1));
            e.push((b.max(a) + 1, -1));
        }
        e.sort_unstable();
        let (mut c, mut p) = (0i32, 0i32);
        for (_, d) in e {
            c += d;
            p = p.max(c);
        }
        p
    }

    #[test]
    #[ignore = "probe: walk the biped creation payload instead of a fixed offset"]
    fn probe_creation_walk() {
        for path in probe_films() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            let mut creations: Vec<(u32, Vec<u8>)> = Vec::new();
            let mut span: std::collections::HashMap<u32, (u32, u32)> = Default::default();
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type != 11 {
                        continue;
                    }
                    let s = span.entry(e.net_id).or_insert((i as u32, i as u32));
                    s.1 = i as u32;
                    if !e.is_update && !creations.iter().any(|(id, _)| *id == e.net_id) {
                        creations.push((e.net_id, e.payload.clone()));
                    }
                }
            }
            let pls: Vec<Vec<u8>> = creations.iter().map(|(_, p)| p.clone()).collect();
            let (eqip, score) = solve_eqip_bits(&pls, &params);
            println!(
                "\n=== {} ===\n  {} present | {} biped creation(s) | eqip width {eqip} closes {:.1}%",
                path.file_name().unwrap_or_default().to_string_lossy(),
                f.present_players().count(),
                creations.len(),
                score * 100.0
            );
            // every width, so the margin is visible rather than asserted
            let mut row = String::new();
            for bits in 1..=12u32 {
                let ok = pls
                    .iter()
                    .filter(|p| creation_biped(p, &params, bits).map_or(false, |c| c.leftover_bits < 8))
                    .count();
                row.push_str(&format!(" {bits}:{:.0}%", 100.0 * ok as f64 / pls.len().max(1) as f64));
            }
            println!("  closes by width:{row}");
            let mut app = 0usize;
            let mut closed = 0usize;
            let mut offs: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut walk: std::collections::BTreeMap<i16, Vec<(u32, u32)>> = Default::default();
            let mut fixed: std::collections::BTreeMap<i16, Vec<(u32, u32)>> = Default::default();
            let mut cross: std::collections::BTreeMap<(i16, i16), usize> = Default::default();
            let mut pools: std::collections::BTreeMap<i16, usize> = Default::default();
            for (id, pl) in &creations {
                let sp = span[id];
                let fx = creation_owner_slot(pl).map(|v| v as i16).unwrap_or(-2);
                fixed.entry(fx).or_default().push(sp);
                let Some(c) = creation_biped(pl, &params, eqip) else {
                    cross.entry((fx, -2)).and_modify(|n| *n += 1).or_insert(1);
                    continue;
                };
                *offs.entry(c.bits_before_owner).or_default() += 1;
                if c.tail_closed {
                    closed += 1;
                }
                if c.appearance {
                    app += 1;
                }
                *pools.entry(c.f3e as i16).or_default() += 1;
                walk.entry(c.owner as i16).or_default().push(sp);
                *cross.entry((fx, c.owner as i16)).or_default() += 1;
            }
            println!(
                "  {app} of {} creation(s) carry an APPEARANCE block; {closed} closed the payload",
                creations.len()
            );
            println!(
                "  owner-field bit OFFSETS (offset -> bodies): {offs:?}  [the old fixed constant \
                 was {CREATION_OWNER_BIT}]"
            );
            println!("  WALKED owner  bodies  peak(alive)");
            for (v, s) in &walk {
                println!("  {v:>12}  {:>6}  {:>11}", s.len(), peak(s));
            }
            println!("  FIXED-offset  bodies  peak(alive)");
            for (v, s) in &fixed {
                println!("  {v:>12}  {:>6}  {:>11}", s.len(), peak(s));
            }
            println!("  +0x3E (the 4-bit field before the owner) values: {pools:?}");
            println!("  (fixed, walked) -> bodies: {cross:?}");
        }
    }
}

/// #naming  THE OWNER STAMP IS WALKED, NOT GUESSED -- and these gates bite if it stops being.
///
/// The fixed offset [`CREATION_OWNER_BIT`] had no oracle of its own: four bits read from a
/// constant place always succeed, however wrong the place is. [`creation_biped`] walks the
/// engine's own reader to the END of the payload, which makes two independent things checkable --
/// that the payload closes (the writer pads to a byte), and that the field it lands on behaves
/// like a per-player index (one player drives one body, so few of a group's bodies are alive at
/// once).
#[cfg(test)]
mod owner_walk_gates {
    use super::*;
    use crate::film;

    /// Pack a bit list (value, width) into a payload, MSB first -- the stream's own order.
    fn pack(bits: &[(u64, u32)]) -> Vec<u8> {
        let mut v: Vec<bool> = Vec::new();
        for &(val, n) in bits {
            for i in (0..n).rev() {
                v.push((val >> i) & 1 != 0);
            }
        }
        let mut bytes = vec![0u8; (v.len() + 7) / 8];
        for (i, b) in v.iter().enumerate() {
            if *b {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        bytes
    }

    fn params() -> DecodeParams {
        DecodeParams { cv3_axis_bits: [15, 16, 14], cv3_total_bits: 45, tagref_bits: 14, update_prefix_bits: 1, ..Default::default() }
    }

    /// The shared creation header in its CHEAPEST shape: no 10-bit sub-field, the 4-bit one
    /// present. 14 + 13 + 1 + 4 + 1 + 4 + 7 + 32 = 76 bits.
    fn shared_cheap() -> Vec<(u64, u32)> {
        vec![(0x2A, 14), (7, 13), (0, 1), (3, 4), (0, 1), (9, 4), (5, 7), (0xDEAD_BEEF, 32)]
    }

    /// The same header with BOTH optional sub-fields the other way: 14+13+1+10+4+1+7+32 = 82.
    fn shared_fat() -> Vec<(u64, u32)> {
        vec![(0x2A, 14), (7, 13), (1, 1), (0x3FF, 10), (3, 4), (1, 1), (5, 7), (0xDEAD_BEEF, 32)]
    }

    /// The appearance block in its cheapest shape (all twelve optional sub-fields ABSENT, i.e.
    /// their flags SET) and its fattest (all present). 108 and 192 bits.
    fn appearance(fat: bool) -> Vec<(u64, u32)> {
        let mut v: Vec<(u64, u32)> = vec![(1, 1), (0, 1), (5, 6), (6, 6), (7, 6), (1, 1)];
        for _ in 0..12 {
            if fat {
                v.push((0, 1));
                v.push((0x55, 7));
            } else {
                v.push((1, 1));
            }
        }
        for _ in 0..4 {
            v.push((0xA3, 8));
        }
        v.extend([(0x42, 7), (0x11, 7), (5, 3), (0x2A, 6), (0x1B, 6), (0x0C, 6), (0x77, 8)]);
        v
    }

    /// THE STRUCTURAL GATE on the walk, and it bites on both halves of what was wrong.
    ///
    /// 1. The owner is read at a place that MOVES; four message shapes put it at four different
    ///    bit offsets and the walk has to move with it. None of the four is the 272 the old
    ///    constant assumed.
    /// 2. The field is FIVE bits. Wire 16 is slot 15, and a four-bit reader hands it back as
    ///    wire 0 -- "nobody". The gate asserts slot 15 survives, which is the exact body of
    ///    evidence that produced four wrong answers about "wire 0 is a mixing bucket".
    #[test]
    fn the_owner_stamp_is_read_where_the_message_puts_it() {
        let p = params();
        let mut offsets: Vec<usize> = Vec::new();
        for (hdr, fat_hdr) in [(shared_cheap(), false), (shared_fat(), true)] {
            for app in [false, true] {
                // wire values: 0 = nobody, and 1..=16 are slots 0..=15. 16 is the one a
                // four-bit read destroys.
                for wire in [0u64, 1, 9, 16] {
                    let mut bits = hdr.clone();
                    let mut off: usize = hdr.iter().map(|(_, n)| *n as usize).sum();
                    if app {
                        bits.push((1, 1)); // the appearance flag
                        bits.push((2, 2));
                        let a = appearance(fat_hdr);
                        off += 3 + a.iter().map(|(_, n)| *n as usize).sum::<usize>();
                        bits.extend(a);
                    } else {
                        bits.push((0, 1));
                        off += 1;
                    }
                    bits.push((5, 4)); // +0x3E, the four-bit field that is NOT the owner
                    off += 4;
                    bits.push((wire, 5)); // +0x3F, the owner, as `value + 1`
                    bits.push((0, EQIP_BITS_FALLBACK)); // a NULL 'eqip', so nothing follows
                    let payload = pack(&bits);
                    let c = creation_biped(&payload, &p, EQIP_BITS_FALLBACK)
                        .expect("the walk reaches the end of a well-formed creation");
                    assert_eq!(
                        c.owner,
                        wire as i8 - 1,
                        "wire {wire} at bit {off} (appearance {app}, fat header {fat_hdr})"
                    );
                    assert_eq!(c.bits_before_owner, off, "the walk must report where it looked");
                    assert_eq!(c.f3e, 4, "the four-bit field before the owner is its own field");
                    assert_eq!(c.appearance, app);
                    assert!(
                        c.tail_closed,
                        "the walk must close the payload: {} bits left",
                        c.leftover_bits
                    );
                    // THE BITE. The old constant 272 is this field's start **plus one** on a
                    // real message of the appearance shape (measured: the field begins at 271
                    // on `*21897D8A*`), so the old reader took the LOW four bits of a five-bit
                    // value. `0b10000` has low four bits `0000`, which is why slot 15 came back
                    // as "nobody" -- and why four earlier answers were about a mixing bucket.
                    let low_four = {
                        let mut rr = BitReader::new(&payload);
                        rr.skip(off + 1).unwrap();
                        rr.bits(4).unwrap()
                    };
                    assert_eq!(low_four, wire & 0xF, "the old reader's window, reconstructed");
                    if wire == 16 {
                        assert_eq!(low_four, 0, "wire 16's low four bits are 0 -- 'nobody'");
                        assert_eq!(c.owner, 15, "and five bits recover slot 15");
                    }
                    offsets.push(off);
                }
            }
        }
        offsets.sort_unstable();
        offsets.dedup();
        assert_eq!(
            offsets.len(),
            4,
            "the four message shapes must put the stamp at FOUR different offsets, or this gate \
             is not testing the thing that broke: {offsets:?}"
        );
        assert!(
            !offsets.contains(&CREATION_OWNER_BIT),
            "none of these four shapes happens to be the old fixed offset, so a reader using it \
             gets all four wrong: {offsets:?} vs {CREATION_OWNER_BIT}"
        );
    }

    /// THE GATE THAT BITES on real films: for every film with enough bipeds, the walk must close
    /// the creations, and the owner groups it produces must behave like players.
    ///
    /// The fixed offset fails this. Measured before the walk landed, on
    /// `asq_mglo-0_forge_hal_53C14E9D` (Castle Wars, 16 header players): the fixed offset put all
    /// 390 bodies into TWO groups with **13 and 21 bodies alive at once**, and the one-player
    /// guard then correctly refused to name any of them -- 0 of 390 named. A group with 21 live
    /// bodies is not a player, so `well_behaved` is the number this gate turns on.
    #[test]
    fn the_walked_owner_groups_behave_like_players_on_every_real_film() {
        let films = film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_decode: no real films on this machine -- owner walk not measured");
            return;
        }
        /// The measured ceiling for one player: a live body plus two lingering corpses.
        const MAX_AT_ONCE: i32 = 3;
        /// How many biped creations a film needs before its groups mean anything.
        const MIN_BODIES: usize = 12;
        let mut films_judged = 0usize;
        let (mut high_films, mut low_films) = (0usize, 0usize);
        for path in &films {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let bipeds: Vec<&EntityTrack> =
                idx.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw.is_some()).collect();
            if bipeds.len() < MIN_BODIES {
                continue;
            }
            films_judged += 1;
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let mut groups: std::collections::BTreeMap<u8, Vec<(u32, u32)>> = Default::default();
            for t in &bipeds {
                groups.entry(t.owner_raw.unwrap_or(0)).or_default().push((t.first_tick, t.last_tick));
            }
            let mut well_behaved = 0usize;
            let mut worst = 0i32;
            for spans in groups.values() {
                let mut e: Vec<(u32, i32)> = Vec::new();
                for &(a, b) in spans {
                    e.push((a, 1));
                    e.push((b.max(a) + 1, -1));
                }
                e.sort_unstable();
                let (mut c, mut pk) = (0i32, 0i32);
                for (_, d) in e {
                    c += d;
                    pk = pk.max(c);
                }
                worst = worst.max(pk);
                if pk <= MAX_AT_ONCE {
                    well_behaved += 1;
                }
            }
            let share = well_behaved as f64 / groups.len() as f64;
            // How many of this film's bodies HMS actually puts a gamertag on through the stamp.
            let named = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.from_owner_rule)
                .count();
            eprintln!(
                "  {name}: {} bodies, {} owner group(s), {well_behaved} behave like one player \
                 ({:.0}%), worst {worst} alive at once, {named} named through the stamp",
                bipeds.len(),
                groups.len(),
                share * 100.0,
            );
            // THE SAFETY PROPERTY, per BODY, and the only one the data supports. Every body
            // HMS names through the stamp must belong to an owner group that could be one
            // player. That is the rule which stops a gamertag landing on a crowd, and it is the
            // mistake this project has already made twice -- once as a `(owner - 1) & 15` wrap
            // (one gamertag on six bodies belonging to several players), once as a whole-group
            // truncation rule (sixteen bodies on one slot with thirteen alive at once).
            //
            // An earlier version of this gate asserted something stronger and WRONG: that a
            // film's share is always either >= 90% or <= 10%, never in between. The corpus
            // refuted it immediately -- `asq_mglo-16_*05C1DB83*` has 13 bodies in 3 groups, 2 of
            // which behave like one player, i.e. 67%. With three groups a single bad one is 33%,
            // so a "clean gap" was an artefact of looking only at films with many groups. The
            // per-body rule needs no such claim.
            for b in idx.bindings.iter().filter(|b| b.is_confident() && b.from_owner_rule) {
                let Some(owner) = idx
                    .tracks
                    .iter()
                    .find(|t| t.net_id == b.net_id)
                    .and_then(|t| t.owner_raw)
                else {
                    continue;
                };
                assert!(
                    idx.owner_group_is_one_player(owner),
                    "{name}: body {:#010x} is named for slot {} out of owner group {owner}, which \
                     cannot be one player. A wrong gamertag is worse than none.",
                    b.net_id,
                    b.slot
                );
            }
            if share >= 0.90 {
                high_films += 1;
            } else {
                low_films += 1;
            }
        }
        eprintln!(
            "film_decode: {films_judged} film(s) judged -- {high_films} where 90%+ of owner \
             groups behave like one player, {low_films} below that; no body was named out of a \
             group that could not be one player on ANY of them"
        );
        assert!(films_judged > 0, "no film on this machine carried {MIN_BODIES} stamped bipeds");
        // THE REGRESSION FLOOR. Measured when the five-bit walk landed: of 14 films judged,
        // **10 are at 100%**, one at 93% (`asq_mglo-8_*E498B50C*`, 15 groups, worst 5 alive at
        // once), one at 67% (`asq_mglo-16_*05C1DB83*`, 3 groups of 13 bodies, worst 4) and three
        // at 0% -- all three `asq_mglo-0`, with 2 groups holding every body and 8, 17 and 21
        // alive at once. If the walk breaks on films that currently work they drop out of the
        // high class and this fails before anyone has to notice the names are gone.
        assert!(
            high_films >= 9,
            "only {high_films} film(s) have 90%+ of their owner groups behaving like one player; \
             11 did when the five-bit walk landed, so something has stopped reading the field"
        );
    }
}

// -------------------------------------------------------------------------------------------
// #player-unit  ENTITY TYPE 3 IS `game-engine-player`, AND IT NAMES THE UNIT
// -------------------------------------------------------------------------------------------

/// The entity type whose records ARE the players: `game-engine-player`, one per slot.
///
/// Sixteen of them on a 16-player film, on every film measured, sitting beside type 5's 24
/// `game-engine-statborg` slots. Its creation reader `sub_1804562E4` is
/// `ReadBits(5)`, **validated `<= 0xF`** -- the player slot, 0..15 -- which is exactly how the
/// statborg's own index is stated.
pub const PLAYER_ENTITY_TYPE: u8 = 3;
/// Field count of the player record's dirty block (`sub_18043442C(.., 0, 0x2D, .., 6)`).
pub const PLAYER_ENTITY_FIELDS: u32 = 0x2D;
/// Its gate list, read off `0x180A4E170`. The six groups are 0..1, 2..7, 8..0x10, 0x11..0x17,
/// 0x18..0x1C and 0x1D..0x2C -- and the *variable* script-action fields are 2..7, i.e. behind
/// their own gate, so the unit reference at 0x10 is reachable through fixed-cost fields alone
/// whenever that gate is clear.
pub const PLAYER_ENTITY_GATES: [u32; 6] = [0, 2, 8, 0x11, 0x18, 0x1D];
/// Field 0x10: `sub_1803CCC04(state + 0x138, bs, 8, host_authority)` -- an optional object
/// reference, i.e. the unit this player is driving.
///
/// **MEASURED ABSENT, and that is the finding.** `Player_SetUnit` (`+0x45814`) maintains both
/// halves of the link -- the unit at `player_datum + slot*0x490 + 0x28` and the player back at
/// `object + 0x1D0` -- and neither half is on the wire. The object half was already known
/// (`SessionBitStream_WriteUnitState` never writes `+0x1CC`); this field is the player half, and
/// over two films its flag is **clear on every single message**:
///
/// | film | type-3 messages | passed BOTH oracles | field 0x10 present |
/// |---|---|---|---|
/// | `*21897D8A*` | 2,038 | 1,802 | **0** |
/// | `*53C14E9D*` | 3,619 | 3,208 | **0** |
///
/// The walk is not the limitation: those messages rebuilt their own header mask AND closed their
/// payload, with walks running on to fields 0x12..0x1A. See `#player-unit` in
/// `re-notes/reach_saved_film_layout.md`. Kept because the slot-per-record mapping is real and
/// because this negative should stay cheap to re-check.
pub const PLAYER_UNIT_FIELD: u32 = 0x10;

/// `sub_1803CCC04` on a film (host authority 0): one flag, and a **SET** flag means a 27-bit
/// object reference follows; clear means the null reference.
///
/// **Note the sense, and the TRAP.** `film_anim::unit::object_ref` reads it the other way round
/// -- it returns the null on a SET flag and reads 27 bits on a clear one -- which is a real bug
/// affecting the biped's equipment reference (field 0x32) and its assassination target (field
/// 0x35). **It is not safe to flip on its own.** Field 0x35's internal layout was pinned by a
/// measured bit BUDGET (105/109/110/112 bits remaining across four films, differing from the
/// per-axis position widths by exactly 60 = 28 + 32) taken against the INVERTED reading, so the
/// flip changes that field's cost by 27 bits. The flip and the re-derivation of the 0x35 budget
/// have to land together, with `film_anim`'s mask oracle re-measured in the same change.
fn optional_object_ref(r: &mut BitReader) -> Result<Option<u32>, DecodeStop> {
    if !r.flag()? {
        return Ok(None);
    }
    let hi = r.bits(16)?;
    let lo = r.bits(11)?;
    Ok(Some(((hi << 16) | lo) as u32))
}

/// What a `game-engine-player` message yielded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerEntityState {
    /// The player slot, from a CREATION message's 5-bit index. Updates do not restate it, so the
    /// caller remembers it per net id the way it does for a statborg.
    pub slot: Option<u8>,
    /// Field 0x10: the unit this player is driving, as a 27-bit object reference in the same
    /// namespace as `EntityTrack::net_id`. `None` either because the field was absent or because
    /// it stated the null reference -- `unit_field_present` separates those.
    pub unit: Option<u32>,
    /// Whether field 0x10 was present at all (so a null can be told from an absence).
    pub unit_field_present: bool,
    /// The mask rebuilt from the presence bits the walk read. The stream states the mask twice,
    /// so this must equal the message's own header mask on a correct walk.
    pub rebuilt_mask: u64,
    /// Bits left at the end; the writer pads to a byte, so a complete walk ends within 7.
    pub leftover_bits: usize,
    /// The first field the walk could not cost, if it hit one.
    pub stopped_at: Option<u32>,
}

impl PlayerEntityState {
    /// Did this walk pass both free oracles? Only then may its unit reference be believed.
    pub fn trustworthy(&self, header_mask: u64) -> bool {
        self.stopped_at.is_none() && self.leftover_bits < 8 && self.rebuilt_mask == header_mask
    }
}

/// Walk a `game-engine-player` message (`sub_1804578D4`).
///
/// Fields 2..7 are the Megalo script-action channel and are not costed; the walk stops cleanly
/// at the first of them that is present, which costs nothing when its gate is clear.
pub fn decode_player_entity(e: &EntityMessage, p: &DecodeParams) -> Option<PlayerEntityState> {
    if e.entity_type != PLAYER_ENTITY_TYPE {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut st = PlayerEntityState::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        // `sub_1804562E4`: the slot, 5 bits, valid 0..15.
        let i = r.bits(5).ok()? as u8;
        if i > 0xF {
            return None;
        }
        st.slot = Some(i);
    }
    if e.dirty_mask == 0 && !e.is_update {
        st.leftover_bits = r.remaining();
        return Some(st);
    }
    let mut g = 0usize;
    let mut field = 0u32;
    while field < PLAYER_ENTITY_FIELDS {
        if g < PLAYER_ENTITY_GATES.len() && field == PLAYER_ENTITY_GATES[g] {
            g += 1;
            let Ok(open) = r.flag() else { break };
            if !open {
                // A clear gate skips this gate's whole GROUP, up to the next gate.
                field = PLAYER_ENTITY_GATES.get(g).copied().unwrap_or(PLAYER_ENTITY_FIELDS);
                continue;
            }
        }
        // Running out of bits AT a presence bit is a clean end: the writer stops emitting once
        // every remaining field is absent, and pads to a byte.
        let Ok(present) = r.flag() else { break };
        let this = field;
        field += 1;
        if !present {
            continue;
        }
        st.rebuilt_mask |= 1u64 << this;
        let ok = match this {
            0x00 => r.bits(3).map(|_| ()),
            0x01 => (|| {
                r.flag()?;
                r.flag()?;
                r.flag()?;
                r.bits(10)?;
                Ok(())
            })(),
            0x08 | 0x0B | 0x0C => r.bits(16).map(|_| ()),
            0x09 => r.bits(10).map(|_| ()),
            0x0A => r.bits(4).map(|_| ()),
            0x0D | 0x0E => r.flag().map(|_| ()),
            0x0F => (|| {
                r.bits(3)?;
                r.bits(3)?;
                Ok(())
            })(),
            PLAYER_UNIT_FIELD => match optional_object_ref(&mut r) {
                Ok(u) => {
                    st.unit_field_present = true;
                    st.unit = u;
                    Ok(())
                }
                Err(err) => Err(err),
            },
            _ => {
                // 2..7 are the Megalo script-action channel; 0x11.. are past what is needed.
                st.stopped_at = Some(this);
                st.leftover_bits = r.remaining();
                return Some(st);
            }
        };
        if ok.is_err() {
            st.stopped_at = Some(this);
            break;
        }
    }
    st.leftover_bits = r.remaining();
    Some(st)
}

/// #player-unit  DOES THE PLAYER RECORD NAME A LIVE BIPED?
///
/// The decisive oracle, and free: a 27-bit reference read at the wrong offset is noise in a
/// 134-million-value space, so if the value names a biped THIS FILM CREATED, and one that is
/// alive at that tick, the field is the unit and the read is right. The same test proved the
/// kind-4 destroy and base field 12's parent at 100%.
#[cfg(test)]
mod player_entity_probe {
    use super::*;
    use crate::film;

    fn probe_films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        crate::film::spec_corpus()
    }

    #[test]
    #[ignore = "probe: does the game-engine-player record name the unit"]
    fn probe_player_entity_names_the_unit() {
        for path in probe_films() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            // bipeds: net id -> (first, last) tick, so the oracle can ask "alive at that tick?"
            let mut biped: std::collections::HashMap<u32, (u32, u32)> = Default::default();
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 {
                        let s = biped.entry(e.net_id).or_insert((i as u32, i as u32));
                        s.1 = i as u32;
                    }
                }
            }
            let mut slot_of: std::collections::HashMap<u32, u8> = Default::default();
            let (mut msgs, mut trust, mut present, mut nonnull) = (0usize, 0usize, 0usize, 0usize);
            let (mut known, mut alive) = (0usize, 0usize);
            let mut stops: std::collections::BTreeMap<u32, usize> = Default::default();
            let mut per_slot: std::collections::BTreeMap<u8, std::collections::BTreeSet<u32>> =
                Default::default();
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type != PLAYER_ENTITY_TYPE {
                        continue;
                    }
                    msgs += 1;
                    let Some(st) = decode_player_entity(e, &params) else { continue };
                    if let Some(s) = st.slot {
                        slot_of.insert(e.net_id, s);
                    }
                    if let Some(fd) = st.stopped_at {
                        *stops.entry(fd).or_default() += 1;
                    }
                    if !st.trustworthy(e.dirty_mask) {
                        continue;
                    }
                    trust += 1;
                    if st.unit_field_present {
                        present += 1;
                    }
                    let Some(u) = st.unit else { continue };
                    nonnull += 1;
                    if let Some(&(a, b)) = biped.get(&u) {
                        known += 1;
                        if (a..=b).contains(&(i as u32)) {
                            alive += 1;
                        }
                        if let Some(&s) = slot_of.get(&e.net_id) {
                            per_slot.entry(s).or_default().insert(u);
                        }
                    }
                }
            }
            println!(
                "\n=== {} ===\n  {} present | {} biped(s) | {msgs} type-3 message(s), {trust} \
                 passed both oracles | field 0x10 present on {present}, non-null on {nonnull}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                f.present_players().count(),
                biped.len(),
            );
            println!(
                "  THE ORACLE: {known} of {nonnull} name a net id THIS FILM CREATED ({:.2}%), \
                 {alive} of those are ALIVE at that tick ({:.2}%)",
                100.0 * known as f64 / nonnull.max(1) as f64,
                100.0 * alive as f64 / nonnull.max(1) as f64
            );
            println!("  walk stopped at field -> count: {stops:?}");
            // WHICH FIELDS ARE ACTUALLY PRESENT. The header mask states this outright, so no
            // field width can be blamed for a zero -- the same reasoning that made the field-0x36
            // negative airtight. This is the census that says where to look next.
            let mut field_hist: std::collections::BTreeMap<u32, usize> = Default::default();
            let mut masks: std::collections::BTreeMap<u64, usize> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in t.entities.iter().filter(|e| e.entity_type == PLAYER_ENTITY_TYPE) {
                    *masks.entry(e.dirty_mask).or_default() += 1;
                    for b in 0..PLAYER_ENTITY_FIELDS {
                        if e.dirty_mask & (1u64 << b) != 0 {
                            *field_hist.entry(b).or_default() += 1;
                        }
                    }
                }
            }
            println!("  FIELD PRESENCE over every type-3 message (from the header mask):");
            for (fd, n) in &field_hist {
                println!("    field {fd:#04x} present on {n}");
            }
            let mut top: Vec<(u64, usize)> = masks.into_iter().collect();
            top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            println!("  commonest masks: {:?}", &top[..top.len().min(8)]);
            println!("  slots seen: {} | distinct units per slot:", slot_of.len());
            for (s, units) in &per_slot {
                println!("    slot {s:>2}: {} distinct unit(s)", units.len());
            }
        }
    }
}

/// #owner-walk  DOES THE DECODED PER-BODY APPEARANCE MATCH THE HEADER ROSTER?
///
/// **The oracle that separates "this clip forces one appearance" from "the walk is landing in a
/// consistent wrong place".** The creation's appearance block is `s_player_appearance` at the
/// same internal offsets the film HEADER uses per player slot, so the header is a free answer
/// key: a correct per-body decode must reproduce some header slot's flags, three colours,
/// `player_model_choice` and ten slot options exactly.
///
/// Why it matters: the owner field sits immediately after this block, and the block's cost is
/// decided by which of its twelve optional sub-fields are present. On `*21897D8A*` the resulting
/// offsets VARY (271 for 108 bodies, 77 for 4). On `*53C14E9D*` every one of 390 bodies lands at
/// 271 -- and sixteen players with different armour should not produce one identical sub-field
/// pattern. Either the clip really forces one appearance, or every message is taking the same
/// wrong branch and the "owner" read after it is some neighbouring quantity.
#[cfg(test)]
mod appearance_oracle {
    use super::*;
    use crate::film;

    fn probe_films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        crate::film::spec_corpus()
    }

    /// The header's `s_player_appearance` for a slot, in this decode's own terms.
    fn header_key(p: &crate::film::FilmPlayer) -> Option<(u8, [i8; 3], u8, [i8; 10])> {
        let a = p.appearance()?;
        let mut slots = [0i8; 10];
        for (i, s) in a.slots.iter().enumerate() {
            slots[i] = *s as i8;
        }
        Some((a.f48, [a.color_primary as i8, a.color_secondary as i8, a.f4b as i8], a.model_choice, slots))
    }

    #[test]
    #[ignore = "oracle: the per-body appearance against the header roster"]
    fn probe_appearance_matches_the_header() {
        for path in probe_films() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams {
                cv3_axis_bits: w,
                cv3_total_bits: w.iter().sum(),
                angle_always: true,
                update_prefix_bits: 1,
                ..base
            };
            // THE ANSWER KEY: every present header slot's appearance.
            let mut key: Vec<(u8, (u8, [i8; 3], u8, [i8; 10]))> = Vec::new();
            for p in f.header.players.iter().filter(|p| p.present()) {
                if let Some(k) = header_key(p) {
                    key.push((p.index as u8, k));
                }
            }
            // every biped creation, once
            let mut creations: Vec<(u32, Vec<u8>)> = Vec::new();
            let mut seen: std::collections::HashSet<u32> = Default::default();
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update && seen.insert(e.net_id) {
                        creations.push((e.net_id, e.payload.clone()));
                    }
                }
            }
            let pls: Vec<Vec<u8>> = creations.iter().map(|(_, p)| p.clone()).collect();
            let (eqip, _) = solve_eqip_bits(&pls, &params);
            println!(
                "\n=== {} ===\n  {} header slot(s) with an appearance | {} biped creation(s)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                key.len(),
                creations.len()
            );
            let mut looks: std::collections::BTreeMap<(u8, [i8; 3], u8, [i8; 12]), Vec<i8>> =
                Default::default();
            let mut matched = 0usize;
            let mut unmatched = 0usize;
            let mut which: std::collections::BTreeMap<u8, usize> = Default::default();
            for (_, pl) in &creations {
                let Some(c) = creation_biped(pl, &params, eqip) else { continue };
                let Some(l) = c.look else { continue };
                looks.entry((l.flags, l.colors, l.model_choice, l.slots)).or_default().push(c.owner);
                // does it reproduce SOME header slot exactly (first ten slot options)?
                let mut ten = [0i8; 10];
                ten.copy_from_slice(&l.slots[..10]);
                let hit = key
                    .iter()
                    .find(|(_, k)| k.0 == l.flags && k.1 == l.colors && k.2 == l.model_choice && k.3 == ten);
                match hit {
                    Some((slot, _)) => {
                        matched += 1;
                        *which.entry(*slot).or_default() += 1;
                    }
                    None => unmatched += 1,
                }
            }
            println!(
                "  DISTINCT decoded appearances: {}  |  reproduce a header slot EXACTLY: \
                 {matched}, no header slot matches: {unmatched}",
                looks.len()
            );
            println!("  header slots hit (slot -> bodies): {which:?}");
            println!("  the decoded appearances, with the owner values that carry them:");
            for ((flags, colors, model, slots), mut owners) in looks {
                owners.sort_unstable();
                let n = owners.len();
                owners.dedup();
                println!(
                    "    flags {flags} colors {colors:?} model_choice {model} slots {slots:?} \
                     -> {n} bod(ies), owner value(s) {owners:?}"
                );
            }
            // THE WIDER ANSWER KEY. These films are one group of players playing repeatedly on
            // one map, so a body whose appearance matches no entry in THIS film's roster may
            // still match an entry in another film's. If it does, the decode is right and this
            // film's header is simply stale -- which is a very different finding from "the walk
            // is wrong", and the two are told apart by data rather than argument.
            {
                let mut wider: Vec<(String, u8, (u8, [i8; 3], u8, [i8; 10]))> = Vec::new();
                // gamertag per roster entry, so a cross-film match can be checked for ambiguity
                let mut tags: Vec<(String, (u8, [i8; 3], u8, [i8; 10]))> = Vec::new();
                for other in crate::film::spec_corpus() {
                    let Ok(of) = film::open_header(&other) else { continue };
                    let tag = other.file_name().unwrap_or_default().to_string_lossy().to_string();
                    for p in of.header.players.iter().filter(|p| p.present()) {
                        if let Some(k) = header_key(p) {
                            wider.push((tag.clone(), p.index as u8, k));
                            tags.push((p.gamertag.clone(), k));
                        }
                    }
                }
                // THE COLLISION RATE. How many DISTINCT GAMERTAGS does one appearance map to
                // across the whole corpus? Matching a body to a name from another FILE is an
                // inference across recordings, and it is only safe where that number is 1.
                {
                    let mut per: std::collections::BTreeMap<(u8, [i8; 3], u8, [i8; 10]), std::collections::BTreeSet<String>> =
                        Default::default();
                    for (g, k) in &tags {
                        per.entry(*k).or_default().insert(g.clone());
                    }
                    let ambiguous = per.values().filter(|v| v.len() > 1).count();
                    println!(
                        "  COLLISIONS: {} distinct appearance(s) across {} roster entries; {} map \
                         to MORE THAN ONE gamertag",
                        per.len(),
                        tags.len(),
                        ambiguous
                    );
                    for (k, names) in per.iter().filter(|(_, v)| v.len() > 1) {
                        println!("    colours {:?} model {} -> {:?}", k.1, k.2, names);
                    }
                    // and how many of THIS film's bodies land on an ambiguous appearance
                    let (mut uniq, mut amb) = (0usize, 0usize);
                    for (_, pl) in &creations {
                        let Some(c) = creation_biped(pl, &params, eqip) else { continue };
                        let Some(l) = c.look else { continue };
                        let mut ten = [0i8; 10];
                        ten.copy_from_slice(&l.slots[..10]);
                        let k = (l.flags, l.colors, l.model_choice, ten);
                        match per.get(&k).map(|v| v.len()) {
                            Some(1) => uniq += 1,
                            Some(_) => amb += 1,
                            None => {}
                        }
                        }
                    println!(
                        "  of this film's bodies: {uniq} match exactly ONE gamertag corpus-wide \
                         (safe to name), {amb} match several (must stay unnamed)"
                    );
                }
                let mut hit_here = 0usize;
                let mut hit_elsewhere: std::collections::BTreeMap<String, usize> = Default::default();
                let mut hit_nowhere = 0usize;
                for (_, pl) in &creations {
                    let Some(c) = creation_biped(pl, &params, eqip) else { continue };
                    let Some(l) = c.look else { continue };
                    let mut ten = [0i8; 10];
                    ten.copy_from_slice(&l.slots[..10]);
                    let same = |k: &(u8, [i8; 3], u8, [i8; 10])| {
                        k.0 == l.flags && k.1 == l.colors && k.2 == l.model_choice && k.3 == ten
                    };
                    if key.iter().any(|(_, k)| same(k)) {
                        hit_here += 1;
                        continue;
                    }
                    let mut found = false;
                    for (tag, _, k) in &wider {
                        if same(k) {
                            *hit_elsewhere.entry(tag.clone()).or_default() += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        hit_nowhere += 1;
                    }
                }
                println!(
                    "  WIDER KEY ({} roster entries over the whole corpus): {hit_here} match THIS \
                     film's roster, {} match ANOTHER film's, {hit_nowhere} match no roster anywhere",
                    wider.len(),
                    hit_elsewhere.values().sum::<usize>()
                );
                for (tag, n) in &hit_elsewhere {
                    println!("    {n:>4} bodies match a roster entry from {tag}");
                }
            }
            println!("  the HEADER's own answer key:");
            for (slot, (flags, colors, model, slots)) in &key {
                println!(
                    "    slot {slot:>2}: flags {flags} colors {colors:?} model_choice {model} \
                     slots {slots:?}"
                );
            }
        }
    }
}

/// #appearance  THE ARMOUR ROUTE AND THE OWNER STAMP MUST AGREE.
///
/// Two independent fields of the same creation message name the same player: the owner stamp at
/// `+0x3F`, and the appearance block that sits immediately before it. Where both can name a body
/// they must name the SAME slot. That is a genuine cross-check -- not a tautology -- because the
/// two are different bits read by different code against different evidence (the stamp through
/// the per-film recorder anchor, the armour through the header roster).
///
/// It is also the check that would have caught the `asq_mglo-0` problem from the other side: on
/// those clips the armour partitions the bodies into ~18 groups while the stamp collapses to 2,
/// so any slot the stamp did claim there would be likely to disagree with the armour.
#[cfg(test)]
mod appearance_gates {
    use super::*;
    use crate::film;

    /// #appearance  THE DEFAULT APPEARANCE MUST NEVER NAME ANYBODY.
    ///
    /// It is the absence of an identity, and the uniqueness rule cannot see that: it only counts
    /// how many LISTED players share an appearance, and on a film where exactly one listed
    /// player is uncustomised that count is one. That is how a guest with no customisation
    /// (`Danielsan413(1)`) came to be named on a body the owner stamp attributed to someone
    /// else. The cost of the guard is real and worth stating: 22 of the 77 bodies the armour
    /// route named on `*53C14E9D*` were matched to slot 12 purely by the default, and they are
    /// correctly nameless now.
    #[test]
    fn the_default_appearance_names_nobody() {
        let films = film::spec_corpus();
        if films.is_empty() {
            return;
        }
        let mut checked = 0usize;
        let mut defaults = 0usize;
        for path in &films {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            checked += 1;
            let by_armour: std::collections::HashSet<u32> = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.origin == BindOrigin::Appearance)
                .map(|b| b.net_id)
                .collect();
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let Some(l) = tr.look else { continue };
                if !l.is_default() {
                    continue;
                }
                defaults += 1;
                assert!(
                    !by_armour.contains(&tr.net_id),
                    "{name}: body {:#010x} has the DEFAULT appearance -- nothing customised -- and \
                     the armour route named it anyway. An uncustomised appearance is the absence \
                     of an identity, not an identity.",
                    tr.net_id
                );
                assert!(
                    idx.inferred_name(tr.net_id).is_none(),
                    "{name}: body {:#010x} has the DEFAULT appearance and was given a cross-film \
                     name, which would attribute every uncustomised body to one player",
                    tr.net_id
                );
            }
        }
        eprintln!(
            "film_decode: over {checked} film(s), {defaults} bod(ies) carry the default \
             appearance and none of them is named by either armour route"
        );
        assert!(
            defaults > 0,
            "no body on any film carries the default appearance, so this gate proved nothing"
        );
    }

    #[test]
    fn the_armour_route_never_contradicts_the_owner_stamp() {
        let films = film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- the armour/stamp cross-check is not measured");
            return;
        }
        let (mut checked, mut both, mut agreed, mut armour_only) = (0usize, 0usize, 0usize, 0usize);
        for path in &films {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            // Every slot the STAMP route confidently gave a body.
            let by_stamp: std::collections::HashMap<u32, u8> = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.origin == BindOrigin::OwnerStamp)
                .map(|b| (b.net_id, b.slot))
                .collect();
            let mut film_both = 0usize;
            let mut film_agreed = 0usize;
            // THE ARMOUR ROUTE'S OWN DECISION, through the same function the route uses.
            //
            // It has to be that function and not a local re-implementation. This gate first
            // applied only the uniqueness rule, so it reported a contradiction on
            // `asq_mglo-16_*28AC2CC7*` over a slot the route itself refuses on occupancy or
            // containment grounds -- a disagreement with a name the program never produces.
            //
            // Note the route skips bodies the stamp already named, so the two sets of BINDINGS
            // are disjoint by construction and comparing those would make this vacuous. Asking
            // `armour_slot` directly is what gives the gate cases at all.
            let key = FilmIndex::appearance_key(&f.header.players);
            // NOT SCOPED. An earlier version skipped films whose roster covers few of their
            // bodies, on the grounds that a stale list makes the comparison meaningless. That
            // reasoning is true but it was treating a symptom: the single contradiction in the
            // corpus was a body with the DEFAULT appearance being matched to an uncustomised
            // guest, and `armour_slot` now refuses the default outright, so the contradiction is
            // gone at source. Scoping on top of that would hide the check rather than pass it,
            // and every film is compared.
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let Some(armour_slot) = idx.armour_slot(tr, &key) else { continue };
                match by_stamp.get(&tr.net_id) {
                    Some(&stamp_slot) => {
                        film_both += 1;
                        if stamp_slot == armour_slot {
                            film_agreed += 1;
                        } else {
                            panic!(
                                "{name}: body {:#010x} -- the owner stamp says slot {stamp_slot} \
                                 and its ARMOUR reproduces slot {armour_slot}'s appearance \
                                 exactly. Two independent fields of one creation message \
                                 disagree, so at least one of them is being read wrong.",
                                tr.net_id
                            );
                        }
                    }
                    None => armour_only += 1,
                }
            }
            checked += 1;
            both += film_both;
            agreed += film_agreed;
            if film_both > 0 {
                eprintln!(
                    "  {name}: {film_agreed} of {film_both} bod(ies) nameable BOTH ways agree; \
                     {} named by the armour alone",
                    idx.appearance_named
                );
            }
        }
        eprintln!(
            "film_decode: over {checked} film(s), {agreed} of {both} bodies that BOTH the owner \
             stamp and the armour can name agree on the slot; the armour alone named \
             {armour_only} further bod(ies)"
        );
        assert_eq!(agreed, both, "the two routes must agree wherever both answer");
        // The cross-check is only worth anything if it actually has cases.
        assert!(
            both > 0,
            "no body on any film could be named BOTH by the owner stamp and by its armour, so \
             this cross-check proved nothing -- either route has stopped working"
        );
    }
}

/// #appearance  THE CROSS-FILM NAME ROUTE, and the three properties that make it honest.
///
/// Naming a body from ANOTHER recording's roster is inference, not a stated fact, so it is only
/// defensible under conditions that can be checked. These are them.
#[cfg(test)]
mod cross_film_gates {
    use super::*;
    use crate::film;

    #[test]
    fn armour_is_a_near_unique_fingerprint_and_the_key_drops_the_rest() {
        let films = film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- the appearance key is not measured");
            return;
        }
        let key = AppearanceKey::from_films(&films);
        eprintln!(
            "film_decode: {} roster entr(ies) over {} film(s) -> {} distinct appearance(s), {} \
             ambiguous (dropped), {} usable",
            key.entries,
            films.len(),
            key.distinct,
            key.ambiguous,
            key.distinct - key.ambiguous
        );
        assert!(key.entries > 0, "no roster entry anywhere carried an appearance");
        // THE FACT THE ROUTE RESTS ON. Measured: 171 entries -> 88 distinct, 1 ambiguous. If
        // armour stopped being near-unique this route would stop being defensible, and this is
        // the number that would say so.
        let share = key.ambiguous as f64 / key.distinct.max(1) as f64;
        assert!(
            share < 0.10,
            "{} of {} distinct appearances map to more than one gamertag ({:.0}%). Armour is no \
             longer a near-unique fingerprint on this corpus, so naming a body from another \
             recording's roster is no longer defensible.",
            key.ambiguous,
            key.distinct,
            share * 100.0
        );
    }

    #[test]
    fn a_cross_film_name_never_overrides_a_stated_one_and_is_never_ambiguous() {
        let films = film::spec_corpus();
        if films.is_empty() {
            return;
        }
        let key = AppearanceKey::from_films(&films);
        let (mut checked, mut cross, mut stated) = (0usize, 0usize, 0usize);
        for path in &films {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            // NOTE: the gate does NOT call the pass itself. `build_index_players` is the
            // production entry point and must do it, or this gate would pass while the feature
            // is dead -- which is exactly what happened once.
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            checked += 1;
            stated += idx.bindings.iter().filter(|b| b.is_confident()).count();
            cross += idx.cross_film_names.len();
            for (net_id, tag, _src) in &idx.cross_film_names {
                // CONDITION 3: never over a stated name.
                assert!(
                    idx.slot_of_entity(*net_id).is_none(),
                    "{name}: body {net_id:#010x} has a STATED name and a cross-film name \
                     ({tag}) at the same time -- inference must never override what the file says"
                );
                // CONDITION 2: the key only holds unambiguous appearances, so a name that came
                // out of it must still be unambiguous when asked again.
                let tr = idx.tracks.iter().find(|t| t.net_id == *net_id).expect("track exists");
                let l = tr.look.expect("a cross-film name requires a decoded appearance");
                let (again, _) = key.lookup(&l).expect("the key must still answer");
                assert_eq!(
                    again, tag,
                    "{name}: body {net_id:#010x} resolved to {tag} but the key now says {again}"
                );
            }
            if !idx.cross_film_names.is_empty() {
                eprintln!(
                    "  {name}: {} stated, {} inferred from another recording's roster",
                    idx.bindings.iter().filter(|b| b.is_confident()).count(),
                    idx.cross_film_names.len()
                );
            }
        }
        eprintln!(
            "film_decode: over {checked} film(s), {stated} stated name(s) and {cross} inferred \
             from other recordings; no inferred name ever landed on a body that had a stated one"
        );
    }
}

/// #appearance  THE PRODUCTION PATH MUST ACTUALLY PRODUCE THE INFERRED NAMES.
///
/// **This gate exists because its absence cost the feature.** `bind_by_appearance_across_films`
/// was briefly made caller-driven for a good reason -- a corpus scan per index build is real
/// repeated work -- but nothing then called it, so the route silently did nothing while the two
/// gates written for it went on passing: they built their own key and invoked the pass directly,
/// testing the code where it sat rather than where a user reaches it. The measurable symptom was
/// `INFERRED: 0` on the one clip the route exists for.
///
/// So this asserts the only thing that matters: calling the entry point the app calls, and
/// NOTHING else, yields every name the corpus can infer.
#[cfg(test)]
mod production_path_gate {
    use super::*;
    use crate::film;

    #[test]
    fn the_entry_point_the_app_calls_yields_every_inferable_name() {
        let films = film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_decode: no real films -- the production path is not measured");
            return;
        }
        let key = AppearanceKey::shared();
        if key.is_empty() {
            eprintln!("film_decode: no roster entry anywhere carried an appearance");
            return;
        }
        let mut total_inferable = 0usize;
        let mut total_produced = 0usize;
        let mut films_with_any = 0usize;
        for path in &films {
            let Ok(f) = film::open(path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            // THE PRODUCTION ENTRY POINT, and nothing else. No helper calls, no manual pass.
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            // What the corpus COULD infer for this film, computed independently of the index's
            // own bookkeeping: a player body, no stated name, and the key answers.
            let candidates: Vec<(u32, &str)> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11 && idx.slot_of_entity(t.net_id).is_none())
                .filter_map(|t| {
                    let l = t.look.as_ref()?;
                    let (tag, _) = key.lookup(l)?;
                    Some((t.net_id, tag))
                })
                .collect();
            let inferable = candidates.len();
            let produced = idx.cross_film_names.len();
            // #identity  Every candidate must be ACCOUNTED FOR -- either named, or belonging to a
            // gamertag the route explicitly refused for overlapping bodies. Those are the only
            // two legitimate outcomes, and the distinction is the whole point of this gate:
            //
            // * A gate asserting `produced == inferable` asserts a premise the concurrency guard
            //   made false. It failed the moment that guard landed, on 128 bodies of one clip
            //   that the route is RIGHT to refuse.
            // * A gate relaxed to `produced <= inferable` would be worse than no gate, because a
            //   route nobody calls produces zero and passes.
            //
            // So: accounted-for, with the refusal read from the index's own record of what it
            // refused -- not recomputed here, which would grade production against a copy of its
            // own rule.
            let named: std::collections::HashSet<u32> =
                idx.cross_film_names.iter().map(|(id, _, _)| *id).collect();
            let unaccounted: Vec<(u32, &str)> = candidates
                .iter()
                .filter(|(id, tag)| {
                    !named.contains(id) && !idx.cross_film_refused.iter().any(|r| r == tag)
                })
                .copied()
                .collect();
            if inferable > 0 {
                films_with_any += 1;
                eprintln!(
                    "  {name}: {produced} inferred name(s) produced, {inferable} inferable, \
                     {} refused for overlapping bodies ({} tag(s))",
                    inferable - produced,
                    idx.cross_film_refused.len()
                );
            }
            assert!(
                unaccounted.is_empty(),
                "{name}: the corpus can infer a name for {} body/bodies that \
                 `build_index_players` neither named nor refused -- e.g. {:#010x} -> {:?}. The \
                 route is not reached from the entry point the app uses, so the capability \
                 exists and nothing invokes it. ({produced} produced, {inferable} inferable, \
                 refused tags: {:?})",
                unaccounted.len(),
                unaccounted[0].0,
                unaccounted[0].1,
                idx.cross_film_refused
            );
            total_inferable += inferable;
            total_produced += produced;
        }
        eprintln!(
            "film_decode: the production path produced {total_produced} of {total_inferable} \
             inferable name(s) across {films_with_any} film(s) that have any"
        );
        // And the gate must have a case, or it proves nothing -- the same trap as above.
        assert!(
            total_inferable > 0,
            "no film on this machine has a body nameable only by cross-film armour, so this gate \
             proved nothing about the production path"
        );
        // "Accounted for" is satisfied by refusing everything, and a route that names nobody is
        // the exact failure this gate was written for. So it must also have PRODUCED names.
        assert!(
            total_produced > 0,
            "the production path accounted for all {total_inferable} inferable name(s) but \
             produced NONE of them -- every one was refused. That is indistinguishable from the \
             route being unreachable, which is what this gate exists to catch."
        );
    }
}

/// #appearance  THE ONE CONTRADICTION THE CROSS-CHECK FOUND, examined.
///
/// `the_armour_route_never_contradicts_the_owner_stamp` fires on
/// `asq_mglo-16_*28AC2CC7*`: a body whose stamp says one slot and whose armour reproduces
/// another's exactly. Two readings are possible and they need different fixes:
///
/// * the film is FREE-FOR-ALL, where the stamp really is the player, and one route is wrong; or
/// * the film is a TEAM game, where the stamp is the team index and comparing it to a player
///   slot is meaningless -- in which case the GATE's premise is wrong and it must be scoped.
///
/// This prints what separates them: the team census from the header, the owner-group structure,
/// and for every contradicting body both slots' gamertags and appearances.
#[cfg(test)]
mod contradiction_probe {
    use super::*;
    use crate::film;

    #[test]
    #[ignore = "probe: examine the armour/stamp contradiction"]
    fn probe_the_contradiction() {
        for path in film::spec_corpus() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let idx = build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let key: Vec<(u8, String, u8, [i8; 3], u8, [i8; 10])> = f
                .header
                .players
                .iter()
                .filter(|p| p.present() && p.index < 16)
                .filter_map(|p| {
                    let a = p.appearance()?;
                    let mut sl = [0i8; 10];
                    for (i, v) in a.slots.iter().enumerate() {
                        sl[i] = *v as i8;
                    }
                    Some((
                        p.index as u8,
                        p.gamertag.clone(),
                        a.f48,
                        [a.color_primary as i8, a.color_secondary as i8, a.f4b as i8],
                        a.model_choice,
                        sl,
                    ))
                })
                .collect();
            let by_stamp: std::collections::HashMap<u32, u8> = idx
                .bindings
                .iter()
                .filter(|b| b.is_confident() && b.origin == BindOrigin::OwnerStamp)
                .map(|b| (b.net_id, b.slot))
                .collect();
            let mut bad: Vec<(u32, u8, u8)> = Vec::new();
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let Some(l) = tr.look else { continue };
                let mut ten = [0i8; 10];
                ten.copy_from_slice(&l.slots[..10]);
                let hits: Vec<u8> = key
                    .iter()
                    .filter(|(_, _, fl, co, mo, sl)| {
                        *fl == l.flags && *co == l.colors && *mo == l.model_choice && *sl == ten
                    })
                    .map(|(s, ..)| *s)
                    .collect();
                if hits.len() != 1 {
                    continue;
                }
                if let Some(&st) = by_stamp.get(&tr.net_id) {
                    if st != hits[0] {
                        bad.push((tr.net_id, st, hits[0]));
                    }
                }
            }
            if bad.is_empty() {
                continue;
            }
            // THE TEAM CENSUS -- what decides whether the gate's premise holds here.
            let mut teams: std::collections::BTreeMap<u8, Vec<String>> = Default::default();
            for p in f.header.players.iter().filter(|p| p.present()) {
                teams.entry(p.team).or_default().push(p.gamertag.clone());
            }
            let groups = idx.owner_group_count();
            println!(
                "\n=== {name} ===\n  {} present | {} owner group(s) | header TEAMS: {} distinct",
                f.present_players().count(),
                groups,
                teams.len()
            );
            for (t, who) in &teams {
                println!("    team {t}: {} player(s) {:?}", who.len(), who);
            }
            println!("  {} contradicting bod(ies):", bad.len());
            for (net, st, ar) in bad.iter().take(6) {
                let g = |s: u8| {
                    key.iter().find(|(i, ..)| *i == s).map(|(_, g, ..)| g.clone()).unwrap_or_default()
                };
                println!("    body {net:#010x}: stamp -> slot {st} {:?} | armour -> slot {ar} {:?}", g(*st), g(*ar));
                for s in [*st, *ar] {
                    if let Some((_, gt, fl, co, mo, sl)) = key.iter().find(|(i, ..)| *i == s) {
                        println!("      slot {s:>2} {gt:?}: flags {fl} colors {co:?} model {mo} slots {sl:?}");
                    }
                }
            }
        }
    }
}
