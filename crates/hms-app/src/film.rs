//! Halo: Reach (MCC) THEATER SAVED FILM reader.
//!
//! Full format spec + the playback feasibility verdict: `re-notes/reach_saved_film_layout.md`.
//!
//! A Reach saved film is a BLF container, `_blf` / `chdr` / `athr` / `flmh` / `ssig` / `flmd` /
//! `_eof`, where:
//!
//! * `flmh` is a FIXED 128276-byte film header. It is a verbatim memory dump of the engine's
//!   `s_game_options`, so it embeds the WHOLE active game variant (0xFC04 bytes) and the WHOLE
//!   active map variant (`c_map_variant`, 0xD9BC bytes) — the Forge object table with real `f32`
//!   positions and orientation basis vectors, NOT the bit-packed form a `.mvar` file uses.
//! * `flmd` is the recorded stream: a flat sequence of type/length-prefixed, bit-packed
//!   per-tick network SIMULATION UPDATES, one record per tick at 60 Hz. This module frames the
//!   records and exposes their bytes; it does not decode a record's bit-packed body (that is the
//!   engine's `MapContext_DecodeFromBitstream`, `haloreach.dll+0x3C846C`, and reproducing it is
//!   the whole cost of real playback — see the spec).
//!
//! WHY THIS IS NOT `mvar.rs`'s READER: `mvar.rs` decodes the content header as an MSB-first
//! BITSTREAM, because that is how the header is stored *inside an `mvar` chunk payload*. The
//! `chdr` CHUNK that both `.mvar` and film files carry is a different encoding of the same
//! fields — a plain, byte-aligned, LITTLE-endian C struct (`s_content_item_metadata`, 0x2B0
//! bytes, preceded in the chunk by one u32). The two encodings were decoded from the same
//! shipped `.mvar` and agree field for field; the test `chdr_struct_matches_the_bit_packed_header`
//! keeps them honest. Everything shared beyond that is a short chunk walk, which `mvar.rs`'s own
//! walker cannot serve (it is private, hard-wired to the `mvar` magic, and assumes a big-endian
//! size — `flmd` stores its size little-endian).

use std::path::Path;

// ---------------------------------------------------------------------------------------------
// BLF container
// ---------------------------------------------------------------------------------------------

/// One BLF chunk. `size` is the total chunk length in the file, including its header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlfChunk {
    pub magic: [u8; 4],
    pub offset: usize,
    pub size: usize,
    pub version: u16,
    pub flags: u16,
    pub payload_at: usize,
    pub payload_len: usize,
}

impl BlfChunk {
    pub fn name(&self) -> String {
        String::from_utf8_lossy(&self.magic).into_owned()
    }
    pub fn payload<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let end = (self.payload_at + self.payload_len).min(data.len());
        &data[self.payload_at.min(data.len())..end]
    }
}

const FLMD: &[u8; 4] = b"flmd";

/// Walk the BLF chunk list, stopping at the first chunk that does not fit.
///
/// `flmd` is special and the engine makes it so (`haloreach.dll+0xE160` lays the whole header
/// image out, `+0x88FBC` patches the size at finalize): its 16-byte header is
/// `magic[4]`, `u32 size` written LITTLE-endian as `payload_len + 12`, `u16 BE version`,
/// `u16 BE flags`, `u32 reserved = 0`; the stream starts at +0x10. So the true chunk length is
/// `size + 4`, which lands exactly on `_eof` in every sample film.
pub fn blf_chunks(data: &[u8]) -> Vec<BlfChunk> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let magic = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
        let size_be = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let size_le = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let version = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);
        let flags = u16::from_be_bytes([data[pos + 10], data[pos + 11]]);
        let (size, payload_at) = if &magic == FLMD {
            // `flmd`'s size is written at FINALISATION. In a recording that is still in progress
            // (`.temp`, which MCC holds open while a match is being played) it is still the
            // template's 12 — header only — so trusting it reads a zero-length stream and the
            // clip looks empty. The stream is the rest of the file in that case: the framer walks
            // it forward and stops at the last complete record, which is what it does anyway.
            let declared = size_le.wrapping_add(4);
            if declared <= 16 || pos + declared > data.len() {
                (data.len() - pos, pos + 16)
            } else {
                (declared, pos + 16)
            }
        } else {
            (size_be, pos + 12)
        };
        if size < 12 || size > data.len() - pos || payload_at > pos + size {
            break;
        }
        out.push(BlfChunk {
            magic,
            offset: pos,
            size,
            version,
            flags,
            payload_at,
            payload_len: pos + size - payload_at,
        });
        pos += size;
    }
    out
}

fn find_chunk<'a>(chunks: &'a [BlfChunk], magic: &[u8; 4]) -> Option<&'a BlfChunk> {
    chunks.iter().find(|c| &c.magic == magic)
}

// ---------------------------------------------------------------------------------------------
// little readers
// ---------------------------------------------------------------------------------------------

fn u16le(b: &[u8], o: usize) -> u16 {
    if o + 2 > b.len() { return 0 }
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() { return 0 }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn i32le(b: &[u8], o: usize) -> i32 {
    u32le(b, o) as i32
}
fn u64le(b: &[u8], o: usize) -> u64 {
    if o + 8 > b.len() { return 0 }
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}
fn f32le(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32le(b, o))
}
fn v3(b: &[u8], o: usize) -> [f32; 3] {
    [f32le(b, o), f32le(b, o + 4), f32le(b, o + 8)]
}
fn byte(b: &[u8], o: usize) -> u8 {
    b.get(o).copied().unwrap_or(0)
}
/// NUL-terminated string in a fixed field, decoded as UTF-8 (lossily) — what the engine writes
/// for gamertags, and what `mvar.rs` does with the same bytes out of the bitstream.
fn cstr(b: &[u8], o: usize, max: usize) -> String {
    if o >= b.len() { return String::new() }
    let s = &b[o..(o + max).min(b.len())];
    let n = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..n]).into_owned()
}
/// NUL-terminated UTF-16LE in a fixed field of `max_wc` code units.
fn wstr(b: &[u8], o: usize, max_wc: usize) -> String {
    let mut units = Vec::with_capacity(max_wc);
    for i in 0..max_wc {
        if o + i * 2 + 2 > b.len() { break }
        let c = u16le(b, o + i * 2);
        if c == 0 { break }
        units.push(c);
    }
    String::from_utf16_lossy(&units)
}

// ---------------------------------------------------------------------------------------------
// s_content_item_metadata
// ---------------------------------------------------------------------------------------------

/// `s_content_item_metadata`, 0x2B0 bytes, byte-aligned little-endian. Appears in a film THREE
/// times: as the `chdr` chunk payload (preceded by one u32), inside the embedded map variant, and
/// inside the embedded game variant. Field for field the same set `mvar::parse_header` reads out
/// of the bit-packed header.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContentHeader {
    /// +0x00. 3 = saved film, 4 = film clip, 5 = map variant, 6 = game variant.
    pub content_type: u32,
    /// +0x04. Length of the real content: the offset just past the `_eof` chunk. The file on
    /// disk is then zero-padded up to a 16-byte multiple.
    pub file_size: u32,
    pub uid: u64,
    pub parent_uid: u64,
    pub root_uid: u64,
    /// +0x20. The match's game id; also the first hex group of the film's file name.
    pub game_id: u64,
    /// +0x28. 2 = matchmaking, 3 = custom/local, 4 = a variant's own default.
    pub activity: u8,
    /// +0x29. 0 none, 1 campaign, 2 firefight, 3 multiplayer.
    pub game_mode: u8,
    /// +0x2A. Game engine type; the finalize copies it out of the game variant.
    pub engine: u8,
    /// +0x2C. Reach map id — the same key `mvar::Variant::map_id` uses, so `mapcat` resolves the
    /// base map straight from a film.
    pub map_id: u32,
    /// +0x30, a dword holding a byte value (0xFF = none).
    pub category: u32,
    pub unknown34: u32,
    pub created_unix: u64,
    pub created_xuid: u64,
    pub created_by: String,
    pub created_online: u32,
    pub modified_unix: u64,
    pub modified_xuid: u64,
    pub modified_by: String,
    pub modified_online: u32,
    /// +0x80, 128 UTF-16 code units.
    pub title: String,
    /// +0x180, 128 UTF-16 code units.
    pub description: String,
    /// +0x280. For a film (content type 3/4) this is the LENGTH IN SECONDS, which the finalize
    /// computes as `tick_count / 60`.
    pub seconds: u32,
    /// +0x284..+0x2AF, 44 bytes. Zero on every sample film; this is where the bit-packed form's
    /// conditional hopper-id / campaign / firefight tail lives.
    pub tail: Vec<u8>,
}

/// `sizeof(s_content_item_metadata)`.
pub const METADATA_SIZE: usize = 0x2B0;
/// The `chdr` chunk payload: one u32 then the metadata.
pub const CHDR_PAYLOAD_SIZE: usize = 4 + METADATA_SIZE;

impl ContentHeader {
    /// Parse the 0x2B0-byte `s_content_item_metadata`. Short input yields zeroed fields.
    pub fn parse(c: &[u8]) -> ContentHeader {
        ContentHeader {
            content_type: u32le(c, 0x00),
            file_size: u32le(c, 0x04),
            uid: u64le(c, 0x08),
            parent_uid: u64le(c, 0x10),
            root_uid: u64le(c, 0x18),
            game_id: u64le(c, 0x20),
            activity: byte(c, 0x28),
            game_mode: byte(c, 0x29),
            engine: byte(c, 0x2A),
            map_id: u32le(c, 0x2C),
            category: u32le(c, 0x30),
            unknown34: u32le(c, 0x34),
            created_unix: u64le(c, 0x38),
            created_xuid: u64le(c, 0x40),
            created_by: cstr(c, 0x48, 16),
            created_online: u32le(c, 0x58),
            modified_unix: u64le(c, 0x5C),
            modified_xuid: u64le(c, 0x64),
            modified_by: cstr(c, 0x6C, 16),
            modified_online: u32le(c, 0x7C),
            title: wstr(c, 0x80, 128),
            description: wstr(c, 0x180, 128),
            seconds: u32le(c, 0x280),
            tail: c.get(0x284..0x2B0).map(|s| s.to_vec()).unwrap_or_default(),
        }
    }

    pub fn content_type_name(&self) -> &'static str {
        match self.content_type {
            3 => "saved film",
            4 => "film clip",
            5 => "map variant",
            6 => "game variant",
            _ => "?",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// athr
// ---------------------------------------------------------------------------------------------

/// `s_blf_chunk_author` — the build that wrote the file. The load path never compares any of it,
/// so a film is NOT gated on an exact build match (the gate is the `flmh` chunk version).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AuthorChunk {
    pub build_name: String,
    pub build_identifier: u64,
    pub build_string: String,
}

impl AuthorChunk {
    pub fn parse(p: &[u8]) -> AuthorChunk {
        AuthorChunk {
            build_name: cstr(p, 0x00, 16),
            build_identifier: u64le(p, 0x10),
            build_string: cstr(p, 0x18, 28),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// offsets
// ---------------------------------------------------------------------------------------------

/// Offsets inside the `flmh` chunk PAYLOAD (file offset 0x34C; always 128276 bytes on Reach).
pub mod flmh {
    /// Total payload length, identical in every Reach film.
    pub const PAYLOAD_LEN: usize = 128276;
    /// The chunk `version` the engine demands EXACTLY (`k_saved_film_header_chunk_version`).
    pub const CHUNK_VERSION: u16 = 0x360E;

    /// char[32] engine build string ("untracked version" on retail MCC). Never compared on load.
    pub const BUILD_NAME: usize = 0x004;
    /// u32, the constant 5 — a film-format revision the writer stamps and the loader ignores.
    pub const FORMAT_REVISION: usize = 0x024;
    /// u32 length (60) of the opaque blob at `COOKIE`.
    pub const COOKIE_LEN: usize = 0x044;
    /// 60 opaque bytes, unique per film. Purpose UNKNOWN.
    pub const COOKIE: usize = 0x048;
    /// bool — this film is a CLIP/snippet (content type 4) rather than a whole film (3).
    pub const IS_CLIP: usize = 0x086;
    /// char[128] session id: "<Day>_<MMDDYYYY>_<HHMMSS>@<arch>@<windows user>@<machine>", or
    /// the literal "UNINIT_SESSIONID".
    /// PRIVACY: this embeds the recording machine's Windows account name and computer name.
    pub const SESSION_ID: usize = 0x08C;
    /// u32 unix time the recording started.
    pub const RECORD_UNIX: usize = 0x10C;
    /// i32 TOTAL TICK COUNT = the number of records in `flmd`. Seconds = this / 60.
    pub const TICK_COUNT: usize = 0x110;
    /// i32 first tick of a clip (-1 on a whole film).
    pub const CLIP_START_TICK: usize = 0x114;
    /// 32 x `{ i32 tick; char text[0x20]; }` diagnostic/marker log. NOT a seek index.
    pub const MARKERS: usize = 0x118;
    pub const MARKER_STRIDE: usize = 0x24;
    pub const MARKER_SLOTS: usize = 32;
    /// i32 number of marker entries in use.
    pub const MARKER_COUNT: usize = 0x598;
    /// The `s_game_options` blob — a verbatim memory dump.
    pub const GAME_OPTIONS: usize = 0x59C;
    pub const GAME_OPTIONS_LEN: usize = 0x1EAA0;
    /// Reserved tail, byte-exact zero in every sample; no code path writes it.
    pub const RESERVED: usize = 0x1F03C;
    pub const RESERVED_LEN: usize = 0x4D8;
}

/// Offsets inside `s_game_options` (which begins at `flmh::GAME_OPTIONS`).
pub mod game_options {
    /// u16 ticks per second. 60 in every sample; `seconds == ticks / 60` holds exactly on all of
    /// them, and the engine's own divide is by the constant 60.
    pub const TICK_RATE: usize = 0x006;
    /// u32 film/game id — the first hex group of the file name.
    pub const FILM_ID: usize = 0x00C;
    /// i32 options version; the loader demands `<= 4`.
    pub const OPTIONS_VERSION: usize = 0x018;
    /// u32 map id.
    pub const MAP_ID: usize = 0x040;
    /// char[260] `haloreach\maps\<scenario>`; must be non-empty or the film is rejected.
    pub const SCENARIO_PATH: usize = 0x044;
    /// u32 bitmask (17 bits) of valid machine slots.
    pub const MACHINE_MASK: usize = 0x254;
    /// 17 x 6-byte machine identifiers.
    pub const MACHINES: usize = 0x258;
    pub const MACHINE_STRIDE: usize = 6;
    pub const MACHINE_SLOTS: usize = 17;
    /// 16 x `s_player_configuration`.
    pub const PLAYERS: usize = 0x2C8;
    pub const PLAYER_STRIDE: usize = 0xA0;
    pub const PLAYER_SLOTS: usize = 16;
    /// `c_game_variant`, a verbatim memory dump. Its own +0x00 is a u32 GAME ENGINE TYPE (the
    /// finalize copies it into `s_content_item_metadata::engine`), and its
    /// `s_content_item_metadata` starts at +0x0C — verified: the metadata's title field then
    /// lands on the game-variant name string seen at `flmh` +0x1B00.
    pub const GAME_VARIANT: usize = 0x14D8;
    pub const GAME_VARIANT_ENGINE: usize = 0x14D8;
    pub const GAME_VARIANT_METADATA: usize = 0x14D8 + 0x0C;
    pub const GAME_VARIANT_LEN: usize = 0xFC04;
    /// `c_map_variant`, a verbatim memory dump.
    pub const MAP_VARIANT: usize = 0x110E0;
    pub const MAP_VARIANT_LEN: usize = 0xD9BC;
}

/// Offsets inside the embedded `c_map_variant`.
pub mod map_variant {
    /// `s_content_item_metadata` (content type 5).
    pub const METADATA: usize = 0x0000;
    /// u16 map-variant version; the loader demands exactly 0x20 (32).
    pub const VERSION: usize = 0x02B0;
    /// u16 `number_of_placeable_object_quotas` — verified equal to the 9-bit field of the
    /// matching real `.mvar` (145 on Forge World, 71 on Condemned).
    pub const NUM_QUOTAS: usize = 0x02B2;
    pub const MAP_ID: usize = 0x02B4;
    /// The 16 bytes between `MAP_ID` and `WORLD_BOUNDS` are NOT MCC's map GUID, though the width
    /// matches: measured, they are a small integer (209 on Forge World) followed by zeros, the
    /// same on every film of that map. A film recorded on a Workshop or modded map therefore
    /// still has no resolvable GUID here, and `map_id` alone cannot name such a map.
    pub const UNIDENTIFIED_2B8: usize = 0x02B8;
    /// f32[6] world bounds, x0 x1 y0 y1 z0 z1. On a film recorded WITHOUT a Forge variant these
    /// are the inverted `[+FLT_MAX, -FLT_MAX]` sentinel.
    pub const WORLD_BOUNDS: usize = 0x02C8;
    pub const BUDGET_MAX: usize = 0x02E0;
    pub const BUDGET_SPENT: usize = 0x02E4;
    /// u32 `map_variant_checksum` — verified equal to the matching `.mvar`'s (0xFFFFFFFF).
    pub const CHECKSUM: usize = 0x02EC;
    /// u32 `m_scenario_palette_crc` — verified equal to the matching `.mvar`'s.
    pub const PALETTE_CRC: usize = 0x02F0;
    /// char[0x1000] NUL-separated Forge label string pool.
    pub const LABEL_POOL: usize = 0x02F4;
    pub const LABEL_POOL_LEN: usize = 0x1000;
    /// u32 bytes of `LABEL_POOL` in use.
    pub const LABEL_POOL_USED: usize = 0x12F4;
    /// u16[256] offsets into `LABEL_POOL`.
    pub const LABEL_OFFSETS: usize = 0x12F8;
    /// u32 number of labels.
    pub const LABEL_COUNT: usize = 0x14F8;
    /// 651 x `s_variant_object`.
    pub const OBJECTS: usize = 0x14FC;
    pub const OBJECT_STRIDE: usize = 0x4C;
    pub const OBJECT_SLOTS: usize = 651;
    /// 256 x 3 bytes of quota records.
    pub const QUOTAS: usize = 0xD640;
    pub const QUOTA_STRIDE: usize = 3;
    pub const QUOTA_SLOTS: usize = 256;
}

// ---------------------------------------------------------------------------------------------
// flmh contents
// ---------------------------------------------------------------------------------------------

/// One entry of the 32-slot `flmh` marker log. This is a diagnostic/marker list, not a seek
/// index — Theater scrubbing uses a separate history file that a `.film` does not contain.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilmMarker {
    pub tick: i32,
    pub text: String,
}

/// One of the 16 recorded `s_player_configuration` slots (160 bytes).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilmPlayer {
    pub index: usize,
    /// +0x00; non-zero in the low byte = the slot is occupied. The engine derives the player
    /// count by counting these, there is no stored count.
    pub valid: u32,
    /// +0x08, 0..3. INFERRED to be the team index (the range is code-enforced).
    pub team: u8,
    /// +0x10, the 6-byte machine identifier; cross-checked by the loader against the
    /// `s_game_options` machine table.
    pub machine: [u8; 6],
    /// +0x16 (unaligned in the file).
    pub xuid: u64,
    /// +0x20, UTF-16[16].
    pub gamertag: String,
    /// +0x64, UTF-16[4].
    pub service_tag: String,
    /// The whole 160-byte record, so the armour/appearance bytes stay available.
    pub raw: Vec<u8>,
}

impl FilmPlayer {
    pub fn present(&self) -> bool {
        (self.valid & 0xFF) != 0
    }
    /// The recorded armour customisation, if the record is long enough to hold it.
    pub fn appearance(&self) -> Option<PlayerAppearance> {
        PlayerAppearance::parse(&self.raw)
    }
    /// #emblem  The recorded EMBLEM, if the record is long enough and the engine's own validity
    /// rules hold. See [`EmblemInfo`].
    pub fn emblem(&self) -> Option<EmblemInfo> {
        EmblemInfo::parse(&self.raw)
    }
}

/// Offset of `s_emblem_info` inside a 160-byte player record.
///
/// **Engine-proven, not inferred.** The record is a 0x20-byte identity header followed by the
/// 0x80-byte `s_player_configuration` (`sub_140B05D80` compares one with `memcmp(..., 0x80)`),
/// so the configuration starts at record +0x20. Inside it, `client.appearance` is at +0x28
/// (`player_appearance_valid(&player_data->client.appearance)` is asserted on `a3 + 0x28`), and
/// inside the 0x28-byte `s_player_appearance` the `s_emblem_info` is at +0x08. 0x20 + 0x28 +
/// 0x08 = **0x50**. Two independent checks on the same arithmetic land on fields this reader had
/// already solved from the file: the gamertag at record +0x20 is `client.desired_name` (+0x00),
/// and the service tag at record +0x64 is `appearance.service_tag` (0x20 + 0x28 + 0x1C).
pub const EMBLEM_AT: usize = 0x50;

/// The per-player EMBLEM recorded in `s_player_configuration`.
///
/// `s_emblem_info` is 8 bytes and `c_player_profile_interface::set_emblem_info` range-checks
/// every one of them:
///
/// ```text
/// +0x00 byte  foreground emblem index   asserted < 0x7F   (80 designs exist)
/// +0x01 byte  background emblem index   asserted < 0x7F   (53 designs exist)
/// +0x02 byte  flags, 3 bits             bit0 alternate-foreground toggle,
///                                       bit1 flip foreground, bit2 flip background
/// +0x03 byte  primary colour index      asserted <= 0x1F
/// +0x04 byte  secondary colour index    asserted <= 0x1F
/// +0x05 byte  background colour index   asserted <= 0x1F
/// +0x06 word  pad, asserted == 0
/// ```
///
/// The colour indices are into `matg` +0x4A4 `emblem colors` (30 entries), which is a different
/// list from the armour palette at +0x498 `profile colors`. They are stored RAW -- unlike the
/// armour colours in the same struct, which are stored as wire-minus-one so that 0xFF means NONE.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmblemInfo {
    pub foreground: u8,
    pub background: u8,
    pub flags: u8,
    pub color_primary: u8,
    pub color_secondary: u8,
    pub color_background: u8,
}

/// How many foreground and background emblem designs Reach ships
/// (`wigl` +0x0AC / +0x0AE, and the `mlib` runtime block lengths agree).
pub const EMBLEM_FOREGROUND_COUNT: u8 = 80;
pub const EMBLEM_BACKGROUND_COUNT: u8 = 53;
/// How many entries `matg` +0x4A4 `emblem colors` has.
pub const EMBLEM_COLOR_COUNT: u8 = 30;

impl EmblemInfo {
    pub fn parse(raw: &[u8]) -> Option<EmblemInfo> {
        let e = raw.get(EMBLEM_AT..EMBLEM_AT + 8)?;
        let out = EmblemInfo {
            foreground: e[0],
            background: e[1],
            flags: e[2],
            color_primary: e[3],
            color_secondary: e[4],
            color_background: e[5],
        };
        // The engine's own `pad == 0` assert. A record that fails it is not an emblem and this
        // reader says so instead of drawing something.
        if e[6] != 0 || e[7] != 0 {
            return None;
        }
        out.valid().then_some(out)
    }
    /// The engine's own range checks, every one of them.
    pub fn valid(&self) -> bool {
        self.foreground < 0x7F
            && self.background < 0x7F
            && self.flags & !7 == 0
            && self.color_primary <= 0x1F
            && self.color_secondary <= 0x1F
            && self.color_background <= 0x1F
    }
    /// True when the indices also fall inside the art Reach actually ships, which is stricter
    /// than the engine's assert (the assert allows 0..0x7E for a 80-entry list).
    pub fn in_range(&self) -> bool {
        self.foreground < EMBLEM_FOREGROUND_COUNT
            && self.background < EMBLEM_BACKGROUND_COUNT
            && self.color_primary < EMBLEM_COLOR_COUNT
            && self.color_secondary < EMBLEM_COLOR_COUNT
            && self.color_background < EMBLEM_COLOR_COUNT
    }
    /// bit0 — is the emblem's alternate (second) foreground layer switched on.
    pub fn alternate_foreground(&self) -> bool {
        self.flags & 1 != 0
    }
    /// bit1 / bit2 — "flip" is a CHANNEL INVERSION in the shipped shader, not a mirror.
    pub fn flip_foreground(&self) -> bool {
        self.flags & 2 != 0
    }
    pub fn flip_background(&self) -> bool {
        self.flags & 4 != 0
    }
    pub fn summary(&self) -> String {
        format!(
            "emblem fg {} bg {} flags {:#03b} colours {}/{}/{}",
            self.foreground, self.background, self.flags, self.color_primary, self.color_secondary, self.color_background
        )
    }
}

/// How many armour slots the recorded appearance selects. One byte each, consecutive.
pub const APPEARANCE_SLOTS: usize = 10;

/// Offset of the first armour slot byte inside a 160-byte player record.
pub const APPEARANCE_AT: usize = 0x58;

/// Which SPECIES each of the ten slots dresses.
///
/// Not inferred: the ten blocks are named in the engine's own reflection data
/// (`player_model_customization_globals_struct_definition`, `reach_tag_test.exe`) and they are
/// `helmet`, `left shoulder`, `right shoulder`, `chest`, `wrist`, `utility`, `paint scheme`,
/// **`elite`**, `spartan armor effect`, **`elite armor effect`** -- in that order, at tag offsets
/// +0x20 .. +0x8C. So Reach does NOT store a second ten-slot block for the Elite: a player's
/// whole Elite loadout is ONE armour byte (slot 7, 8 options = Minor / Spec Ops / Space / Ultra /
/// General / Officer / Story / Zealot) plus their Elite armour effect (slot 9), and the other
/// eight bytes are the Spartan's.
///
/// This matters because slots 8 and 9 both override the SAME render-model region (`armor_fx`,
/// whose six permutations exist on both models), so applying all ten to one body would let a
/// player's Spartan armour effect leak onto their Elite and the other way round. Every other
/// slot is already disjoint by region -- the Spartan render model carries no `armor_pieces` and
/// the Elite carries no `flair_*` -- but the species split is what makes that a rule instead of
/// a coincidence.
pub const SPARTAN_APPEARANCE_SLOTS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 8];

/// The slots that dress an ELITE body: the `elite selections` block and the
/// `elite armor effect selections` block. See [`SPARTAN_APPEARANCE_SLOTS`].
pub const ELITE_APPEARANCE_SLOTS: [usize; 2] = [7, 9];

/// The names of the ten armour slots, in record order.
///
/// CONFIRMED, not inferred: each slot is named after the render-model regions that its options
/// actually override in `globals\player_model_customization` -- slot 1's options all override
/// `flair_left_shoulder`, slot 4's all override `flair_forearm`, and so on. Slots 8 and 9 are the
/// tag's SEPARATE `spartan armor effect selections` and `elite armor effect selections` fields,
/// whose option lists are identical (seven) and which the cache therefore stores once -- both
/// read back at the same block pointer, and both touch `armor_fx`.
pub const APPEARANCE_SLOT_NAMES: [&str; APPEARANCE_SLOTS] = [
    "helmet",
    "left shoulder",
    "right shoulder",
    "chest",
    "wrist",
    "utility",
    "knees",
    "elite armour",
    "armour effect",
    "elite armour effect",
];

/// The render-model regions each slot's options override, in slot order. Read out of
/// `globals\player_model_customization` in `forge_halo.map`. Slot 7's `armor_pieces` exists only
/// on the ELITE render model and slot 9 is the Elite's half of `armor_fx` -- see
/// [`SPARTAN_APPEARANCE_SLOTS`].
pub const APPEARANCE_SLOT_REGIONS: [&str; APPEARANCE_SLOTS] = [
    "head, flair_head_a, flair_head_b, flair_head_c",
    "flair_left_shoulder",
    "flair_right_shoulder",
    "flair_chest_a, flair_chest_b, arms",
    "flair_forearm",
    "flair_thigh",
    "flair_knees",
    "armor_pieces",
    "armor_fx",
    "armor_fx",
];

/// How many options each slot has, read out of `globals\player_model_customization` in
/// `forge_halo.map`. The ten slot bytes line up with these ten blocks, in order: across 113
/// distinct recorded players the maximum value seen in each byte was strictly less than the
/// matching count, and exactly `count - 1` for six of the ten. That ordered cardinality match is
/// what pins the layout -- it is not a guess about which byte is which.
pub const APPEARANCE_COUNTS: [u8; APPEARANCE_SLOTS] = [90, 18, 18, 26, 6, 7, 4, 8, 7, 7];

/// The per-player armour customisation recorded in `s_player_configuration`.
///
/// Each slot byte is an index into one block of `globals\player_model_customization`; each option
/// in such a block carries a list of (render-model region, permutation) overrides, which is the
/// same region/permutation machinery an `hlmt` model variant uses. So dressing a recorded player
/// is: resolve the model's default variant, then apply that player's per-slot overrides on top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerAppearance {
    /// Ten option indices, one per slot, in `APPEARANCE_SLOT_NAMES` order.
    pub slots: [u8; APPEARANCE_SLOTS],
    /// +0x49 and +0x4A. Observed 0..30 over 113 real players, which is wider than any region's
    /// permutation count, so these are NOT region indices; they are believed to be the primary
    /// and secondary armour COLOUR indices into the game's colour table. INFERRED.
    pub color_primary: u8,
    pub color_secondary: u8,
    /// +0x48 and +0x4B. **No longer unidentified**: `player_appearance_encode` names them.
    /// `f48` is `s_player_appearance` +0x00, two flag bits -- bit0 `female-voice`, bit1
    /// `female-model` -- which is exactly the observed 0..3. `f4b` is +0x03, the TERTIARY armour
    /// colour index, which is why it shares the 0..30 palette with the other two and is 0 on
    /// most players. The field names are kept as they were so no other reader has to move.
    pub f48: u8,
    pub f4b: u8,
    /// +0x4C `player_model_choice`: 0 = Spartan, 1 = Elite (`k_player_model_choice_count` = 2).
    /// Measured 0 or 1 on all 113 recorded players.
    pub model_choice: u8,
}

impl PlayerAppearance {
    pub fn parse(raw: &[u8]) -> Option<PlayerAppearance> {
        if raw.len() < APPEARANCE_AT + APPEARANCE_SLOTS {
            return None;
        }
        let mut slots = [0u8; APPEARANCE_SLOTS];
        slots.copy_from_slice(&raw[APPEARANCE_AT..APPEARANCE_AT + APPEARANCE_SLOTS]);
        Some(PlayerAppearance {
            slots,
            color_primary: raw[0x49],
            color_secondary: raw[0x4A],
            f48: raw[0x48],
            f4b: raw[0x4B],
            model_choice: raw[0x4C],
        })
    }
    /// True when every slot index is inside its block, i.e. this record could be applied.
    pub fn in_range(&self) -> bool {
        self.slots.iter().zip(APPEARANCE_COUNTS.iter()).all(|(v, n)| v < n)
    }
    /// All-zero slots: the player is wearing the default loadout.
    pub fn is_default(&self) -> bool {
        self.slots.iter().all(|v| *v == 0)
    }
    /// A one-line human summary, slots that differ from the default only.
    pub fn summary(&self) -> String {
        if self.is_default() {
            return "default armour".into();
        }
        let mut parts: Vec<String> = Vec::new();
        for (i, v) in self.slots.iter().enumerate() {
            if *v != 0 {
                parts.push(format!("{} {}", APPEARANCE_SLOT_NAMES[i], v));
            }
        }
        format!("{} (colours {}/{})", parts.join(", "), self.color_primary, self.color_secondary)
    }
}

/// World units per unit of the `.mvar`'s 11-bit boundary field: `f = (q - 0.5) * 100/1023`.
/// **File-proven**: this reproduces the 11-bit value of the paired `.mvar` on 598 of 598
/// boundary dimensions across 18 film/`.mvar` pairs, and reconstructs the float to within 2e-6
/// on 594 of them (the rest is `f32` rounding).
pub const BOUNDARY_STEP: f32 = 100.0 / 1023.0;

/// `.mvar` 11-bit boundary value -> world units.
pub fn dequantize_boundary(q: u16) -> f32 {
    if q == 0 { 0.0 } else { (q as f32 - 0.5) * BOUNDARY_STEP }
}

/// World units -> the `.mvar`'s 11-bit boundary value.
pub fn quantize_boundary(f: f32) -> u16 {
    if f == 0.0 { return 0 }
    ((f / BOUNDARY_STEP + 0.5).round()).clamp(0.0, 2047.0) as u16
}

/// Which of the record's four boundary floats the `.mvar`'s `boundary_shape`-many values map to.
/// Sphere stores one radius; a cylinder stores radius/top/bottom and leaves the record's `length`
/// slot unused (always 0.0 on all 94 real cylinders); a box stores all four.
/// **File-proven** by the same 598-dimension comparison.
pub fn boundary_float_slots(shape: u8) -> &'static [usize] {
    match shape {
        1 => &[0],
        2 => &[0, 2, 3],
        3 => &[0, 1, 2, 3],
        _ => &[],
    }
}

/// One `s_variant_object` (76 bytes) of the embedded Forge table.
///
/// The field map below is **file-proven**: every named field was solved against 3910 objects from
/// 18 film/`.mvar` pairs, by comparing the raw record bytes with what `mvar::parse_variant`
/// decoded for the same object out of the bit-packed `.mvar`. Each one matched on 100% of rows
/// (see the test `variant_object_fields_match_the_mvar`). Only +0x01, +0x2F and +0x4B are
/// still unknown, and they are zero on all 3910.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantObject {
    pub slot: usize,
    /// +0x00. `mvar::PlacedObject::flags`; bit 0 = occupied. 1 on every present slot seen.
    pub flags: u16,
    /// +0x02. `mvar::PlacedObject::folder` — the variant QUOTA index, a flat index into the
    /// scenario sandbox palette. 0xFFFF on an absent slot. [file: 3910/3910]
    pub quota_index: u16,
    /// +0x04. `mvar::PlacedObject::spawn_rel` — parent object slot, -1 = none.
    pub parent: i32,
    /// +0x08, world units, same frame as `mvar::PlacedObject::pos`. [file: 3910/3910]
    pub position: [f32; 3],
    /// +0x14.
    pub forward: [f32; 3],
    /// +0x20.
    pub up: [f32; 3],
    /// +0x2C. `mvar::PlacedObject::location_name` (0xFFFF = none; only meaningful for
    /// `cached_type` 19). 0xFFFF on all 3910 rows, so the mapping is [inferred], not proven.
    pub location_name: u16,
    /// +0x2E. `mvar::PlacedObject::item` — the variant/permutation index within the quota's
    /// object type. [file: 3910/3910]
    pub item: u8,
    /// +0x30. Boundary dimensions as real floats (the `.mvar` stores these 11-bit quantised, so
    /// they do not compare exactly; shape and presence do).
    pub boundary: [f32; 4],
    /// +0x40. `mvar::PlacedObject::boundary_shape`: 0 none, 1 sphere, 2 cylinder, 3 box.
    /// [file: 3910/3910]
    pub boundary_shape: u8,
    /// +0x41. `mvar::PlacedObject::spawn_seq` (-100..100). [file: 3910/3910]
    pub spawn_seq: i8,
    /// +0x42. `mvar::PlacedObject::respawn`, seconds. [file: 3910/3910]
    pub respawn: u8,
    /// +0x43. `mvar::PlacedObject::cached_type` (1 weapon, 12..14 teleporter, 19 location).
    /// [file: 3910/3910]
    pub cached_type: u8,
    /// +0x44. `mvar::PlacedObject::label_idx` (0xFFFF = none). [file: 3910/3910]
    pub label_idx: u16,
    /// +0x46. `mvar::PlacedObject::placement` bitfield. [file: 3910/3910]
    pub placement: u8,
    /// +0x47. `mvar::PlacedObject::team` (0xFF = none). [file: 3910/3910]
    pub team: u8,
    /// +0x48. A UNION, keyed by `cached_type`: `weapon_clips` when 1, `tele_channel` when
    /// 12..14, otherwise 0. [file: 72/72 weapons, 73/73 teleporters, 0 on all 3765 others]
    pub clips_or_tele_channel: u8,
    /// +0x49. `mvar::PlacedObject::tele_passability` (only for `cached_type` 12..14).
    /// [file: 73/73]
    pub tele_passability: u8,
    /// +0x4A. `mvar::PlacedObject::color` (-1 = none). [file: 3910/3910]
    pub color: i8,
    /// The whole 76-byte record, so the three unknown bytes stay available.
    pub raw: Vec<u8>,
}

impl VariantObject {
    /// A placed object, as opposed to an unused slot (which carries the engine's absent pattern:
    /// zero transform, `quota_index` 0xFFFF, `parent` -1).
    pub fn present(&self) -> bool {
        self.position != [0.0; 3] || self.forward != [0.0; 3] || self.up != [0.0; 3]
    }

    /// Convert to the shape `mvar.rs` encodes, so an extracted Forge scene can be written back
    /// out as a real `.mvar` and drawn by HMS's existing object scene. The rotation is
    /// re-quantised by `mvar::encode_orientation` (the `.mvar` form stores a 20-bit up vector
    /// plus a 14-bit forward angle, not two float vectors).
    pub fn to_placed_object(&self) -> crate::mvar::PlacedObject {
        let (up_is_global, up_quant, forward_angle_q) = crate::mvar::encode_orientation(self.forward, self.up);
        let (fwd, up) = crate::mvar::decode_orientation(up_is_global, up_quant, forward_angle_q);
        let mut boundary = [0u16; 4];
        for (i, &slot) in boundary_float_slots(self.boundary_shape).iter().enumerate() {
            boundary[i] = quantize_boundary(self.boundary[slot]);
        }
        let (weapon_clips, tele_channel) = match self.cached_type {
            1 => (self.clips_or_tele_channel, 0),
            12..=14 => (0, self.clips_or_tele_channel),
            _ => (0, 0),
        };
        crate::mvar::PlacedObject {
            folder: self.quota_index,
            item: self.item,
            pos: self.position,
            team: self.team,
            color: self.color as i32,
            fwd,
            up,
            up_is_global,
            up_quant,
            forward_angle_q,
            spawn_seq: self.spawn_seq as i32,
            respawn: self.respawn,
            cached_type: self.cached_type,
            label_idx: self.label_idx,
            placement: self.placement,
            boundary_shape: self.boundary_shape,
            boundary,
            flags: (self.flags & 0x3) as u8,
            in_bounds: true,
            bsp_index: -1,
            spawn_rel: self.parent,
            slot: self.slot as u16,
            weapon_clips,
            tele_channel,
            tele_passability: self.tele_passability,
            location_name: self.location_name,
        }
    }
}

/// The map variant the film was recorded on — embedded WHOLE, as the in-memory `c_map_variant`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmbeddedMapVariant {
    pub metadata: ContentHeader,
    /// 32 on every MCC-era variant; the loader rejects anything else.
    pub version: u16,
    pub num_quotas: u16,
    pub map_id: u32,
    pub world_bounds: [f32; 6],
    pub budget_max: u32,
    pub budget_spent: u32,
    pub checksum: u32,
    pub palette_crc: u32,
    /// The Forge label strings, in index order.
    pub labels: Vec<String>,
    /// All 651 slots in order; filter with `VariantObject::present()`.
    pub objects: Vec<VariantObject>,
    /// 256 x 3 raw quota bytes. `mvar.rs` reads the same table as
    /// `(minimum_count, maximum_count, placed_on_map)`.
    pub quotas: Vec<[u8; 3]>,
}

impl EmbeddedMapVariant {
    pub fn present_objects(&self) -> impl Iterator<Item = &VariantObject> {
        self.objects.iter().filter(|o| o.present())
    }
    /// A film recorded on a STOCK map (no Forge variant loaded) still carries the whole
    /// `c_map_variant`, but the engine's empty one: inverted `[+FLT_MAX, -FLT_MAX]` bounds, zero
    /// budget, and all 651 slots absent. Confirmed on a real Countdown film.
    pub fn is_empty(&self) -> bool {
        !(self.world_bounds[1] > self.world_bounds[0]
            && self.world_bounds[3] > self.world_bounds[2]
            && self.world_bounds[5] > self.world_bounds[4])
    }

    fn parse(mv: &[u8]) -> EmbeddedMapVariant {
        use map_variant::*;
        let mut objects = Vec::with_capacity(OBJECT_SLOTS);
        for s in 0..OBJECT_SLOTS {
            let at = OBJECTS + s * OBJECT_STRIDE;
            let Some(r) = mv.get(at..at + OBJECT_STRIDE) else { break };
            objects.push(VariantObject {
                slot: s,
                flags: u16le(r, 0x00),
                quota_index: u16le(r, 0x02),
                parent: i32le(r, 0x04),
                position: v3(r, 0x08),
                forward: v3(r, 0x14),
                up: v3(r, 0x20),
                location_name: u16le(r, 0x2C),
                item: r[0x2E],
                boundary: [f32le(r, 0x30), f32le(r, 0x34), f32le(r, 0x38), f32le(r, 0x3C)],
                boundary_shape: r[0x40],
                spawn_seq: r[0x41] as i8,
                respawn: r[0x42],
                cached_type: r[0x43],
                label_idx: u16le(r, 0x44),
                placement: r[0x46],
                team: r[0x47],
                clips_or_tele_channel: r[0x48],
                tele_passability: r[0x49],
                color: r[0x4A] as i8,
                raw: r.to_vec(),
            });
        }
        let mut quotas = Vec::with_capacity(QUOTA_SLOTS);
        for q in 0..QUOTA_SLOTS {
            let at = QUOTAS + q * QUOTA_STRIDE;
            match mv.get(at..at + QUOTA_STRIDE) {
                Some(r) => quotas.push([r[0], r[1], r[2]]),
                None => break,
            }
        }
        // Labels: a NUL-separated pool plus a u16 offset table.
        let label_count = u32le(mv, LABEL_COUNT).min(256) as usize;
        let pool_used = (u32le(mv, LABEL_POOL_USED) as usize).min(LABEL_POOL_LEN);
        let mut labels = Vec::with_capacity(label_count);
        for i in 0..label_count {
            let off = u16le(mv, LABEL_OFFSETS + i * 2) as usize;
            if off >= pool_used {
                labels.push(String::new());
            } else {
                labels.push(cstr(mv, LABEL_POOL + off, pool_used - off));
            }
        }
        EmbeddedMapVariant {
            metadata: mv.get(METADATA..METADATA + METADATA_SIZE).map(ContentHeader::parse).unwrap_or_default(),
            version: u16le(mv, VERSION),
            num_quotas: u16le(mv, NUM_QUOTAS),
            map_id: u32le(mv, MAP_ID),
            world_bounds: [
                f32le(mv, WORLD_BOUNDS), f32le(mv, WORLD_BOUNDS + 4), f32le(mv, WORLD_BOUNDS + 8),
                f32le(mv, WORLD_BOUNDS + 12), f32le(mv, WORLD_BOUNDS + 16), f32le(mv, WORLD_BOUNDS + 20),
            ],
            budget_max: u32le(mv, BUDGET_MAX),
            budget_spent: u32le(mv, BUDGET_SPENT),
            checksum: u32le(mv, CHECKSUM),
            palette_crc: u32le(mv, PALETTE_CRC),
            labels,
            objects,
            quotas,
        }
    }
}

/// The parsed `flmh` chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilmHeader {
    /// Chunk `version` field; the loader demands `flmh::CHUNK_VERSION` exactly.
    pub chunk_version: u16,
    pub build_name: String,
    pub format_revision: u32,
    /// See the privacy note on `flmh::SESSION_ID`.
    pub session_id: String,
    pub record_unix: u32,
    /// Number of recorded ticks == number of `flmd` records.
    pub tick_count: i32,
    pub clip_start_tick: i32,
    pub is_clip: bool,
    pub markers: Vec<FilmMarker>,
    /// Ticks per second (60).
    pub tick_rate: u16,
    pub film_id: u32,
    pub options_version: i32,
    pub map_id: u32,
    /// `haloreach\maps\<scenario>`.
    pub scenario_path: String,
    pub players: Vec<FilmPlayer>,
    /// `c_game_variant` +0x00: the game ENGINE TYPE (0 none, 1 CTF, 2 slayer, ...). The
    /// finalize copies this into `ContentHeader::engine`.
    pub game_engine_type: u32,
    /// The embedded `c_game_variant`'s own metadata (content type 6) — its title, description
    /// and author. The rest of the 0xFC04-byte variant (the Megalo script state and its
    /// localized string tables) is left unparsed.
    pub game_variant: ContentHeader,
    /// #gamevariant  The embedded variant's DECLARED SETTINGS — loadout palettes, player
    /// traits, respawn and round options, player species. `None` when the film carries no
    /// variant at all (its content type is not 6), which is what a pure Forge recording looks
    /// like. See `film_variant`.
    pub gametype: Option<crate::film_variant::GameVariant>,
    pub map_variant: EmbeddedMapVariant,
}

impl FilmHeader {
    pub fn parse(p: &[u8], chunk_version: u16) -> FilmHeader {
        use flmh::*;
        let go = GAME_OPTIONS;
        let mut markers = Vec::new();
        let count = (i32le(p, MARKER_COUNT).max(0) as usize).min(MARKER_SLOTS);
        for i in 0..count {
            let at = MARKERS + i * MARKER_STRIDE;
            markers.push(FilmMarker { tick: i32le(p, at), text: cstr(p, at + 4, 0x20) });
        }
        let mut players = Vec::with_capacity(game_options::PLAYER_SLOTS);
        for i in 0..game_options::PLAYER_SLOTS {
            let at = go + game_options::PLAYERS + i * game_options::PLAYER_STRIDE;
            let raw = p.get(at..at + game_options::PLAYER_STRIDE).map(|s| s.to_vec()).unwrap_or_default();
            let mut machine = [0u8; 6];
            if raw.len() >= 0x16 {
                machine.copy_from_slice(&raw[0x10..0x16]);
            }
            players.push(FilmPlayer {
                index: i,
                valid: u32le(&raw, 0x00),
                team: byte(&raw, 0x08),
                machine,
                xuid: u64le(&raw, 0x16),
                gamertag: wstr(&raw, 0x20, 16),
                service_tag: wstr(&raw, 0x64, 4),
                raw,
            });
        }
        let gv = p
            .get(go + game_options::GAME_VARIANT_METADATA..go + game_options::GAME_VARIANT_METADATA + METADATA_SIZE)
            .map(ContentHeader::parse)
            .unwrap_or_default();
        let mv = p
            .get(go + game_options::MAP_VARIANT..go + game_options::MAP_VARIANT + game_options::MAP_VARIANT_LEN)
            .map(EmbeddedMapVariant::parse)
            .unwrap_or_default();
        let gametype = p
            .get(go + game_options::GAME_VARIANT..go + game_options::GAME_VARIANT + game_options::GAME_VARIANT_LEN)
            .and_then(crate::film_variant::GameVariant::parse);
        FilmHeader {
            chunk_version,
            build_name: cstr(p, BUILD_NAME, 32),
            format_revision: u32le(p, FORMAT_REVISION),
            session_id: cstr(p, SESSION_ID, 128),
            record_unix: u32le(p, RECORD_UNIX),
            tick_count: i32le(p, TICK_COUNT),
            clip_start_tick: i32le(p, CLIP_START_TICK),
            is_clip: byte(p, IS_CLIP) != 0,
            markers,
            tick_rate: u16le(p, go + game_options::TICK_RATE),
            film_id: u32le(p, go + game_options::FILM_ID),
            options_version: i32le(p, go + game_options::OPTIONS_VERSION),
            map_id: u32le(p, go + game_options::MAP_ID),
            scenario_path: cstr(p, go + game_options::SCENARIO_PATH, 260),
            players,
            game_engine_type: u32le(p, go + game_options::GAME_VARIANT_ENGINE),
            game_variant: gv,
            gametype,
            map_variant: mv,
        }
    }

    /// `forge_halo` out of `haloreach\maps\forge_halo`.
    pub fn scenario_name(&self) -> &str {
        self.scenario_path.rsplit(|c| c == '\\' || c == '/').next().unwrap_or(&self.scenario_path)
    }
}

// ---------------------------------------------------------------------------------------------
// flmd -- the recorded stream
// ---------------------------------------------------------------------------------------------

/// What a `flmd` record is. The 2-bit type lives in the top bits of the record's length dword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordKind {
    /// One simulation tick: a bit-packed network game-state update (the ONLY kind present in
    /// every retail film examined). The body is the engine's `MapContext_EncodeToBitstream`
    /// output; this reader does not decode it.
    Tick,
    /// A full game-state blob, prefixed `{u32 size, u32 crc, u32}`.
    GameState,
    /// A 28-byte marker / segment descriptor.
    Segment,
    /// A chunked, zlib-deflated, CRC-checked blob.
    Compressed,
}

impl RecordKind {
    fn from_bits(t: u32) -> RecordKind {
        match t {
            0 => RecordKind::Tick,
            1 => RecordKind::GameState,
            2 => RecordKind::Segment,
            _ => RecordKind::Compressed,
        }
    }
}

/// One framed `flmd` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilmRecord {
    /// Tick index == the record's ordinal position; there is no timestamp in the stream.
    pub index: usize,
    pub kind: RecordKind,
    /// File offset of the record's 4-byte length dword.
    pub header_at: usize,
    /// File offset of the record body.
    pub body_at: usize,
    /// Body length in bytes (the 30-bit field).
    pub len: usize,
}

impl FilmRecord {
    pub fn body<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let end = (self.body_at + self.len).min(data.len());
        &data[self.body_at.min(data.len())..end]
    }
}

/// The `flmd` chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilmData {
    pub chunk_offset: usize,
    /// The little-endian `size` field verbatim; equals `stream_len + 12`.
    pub size_field: u32,
    /// The zero dword at `chunk_offset + 0x0C`.
    pub reserved: u32,
    pub stream_at: usize,
    pub stream_len: usize,
    /// Framed records, in order.
    pub records: Vec<FilmRecord>,
    /// True when the records tiled the stream exactly (no leftover bytes, no truncation).
    pub framing_exact: bool,
}

/// The engine's per-record sanity bound (`len <= 0x10000`) on a tick record.
pub const MAX_TICK_RECORD_LEN: usize = 0x10000;

impl FilmData {
    fn parse(c: &BlfChunk, data: &[u8]) -> FilmData {
        let stream_at = c.payload_at;
        let stream_len = c.payload_len;
        let end = stream_at + stream_len;
        let mut records = Vec::new();
        let mut at = stream_at;
        let mut exact = true;
        while at + 4 <= end {
            let hdr = u32le(data, at);
            let len = (hdr & 0x3FFF_FFFF) as usize;
            let kind = RecordKind::from_bits(hdr >> 30);
            if len == 0 || len > end - (at + 4) {
                exact = false;
                break;
            }
            records.push(FilmRecord { index: records.len(), kind, header_at: at, body_at: at + 4, len });
            at += 4 + len;
        }
        if at != end {
            exact = false;
        }
        FilmData {
            chunk_offset: c.offset,
            size_field: u32le(data, c.offset + 4),
            reserved: u32le(data, c.offset + 0x0C),
            stream_at,
            stream_len,
            records,
            framing_exact: exact,
        }
    }
    pub fn stream<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let end = (self.stream_at + self.stream_len).min(data.len());
        &data[self.stream_at.min(data.len())..end]
    }
    /// Records whose kind is `Tick` — i.e. simulation ticks.
    pub fn ticks(&self) -> impl Iterator<Item = &FilmRecord> {
        self.records.iter().filter(|r| r.kind == RecordKind::Tick)
    }
}

// ---------------------------------------------------------------------------------------------
// the film
// ---------------------------------------------------------------------------------------------

/// A parsed Reach saved film. `bytes` is retained so record bodies stay reachable.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Film {
    pub bytes: Vec<u8>,
    pub chunks: Vec<BlfChunk>,
    /// `_blf` payload +0x02: "reach saved film".
    pub container_name: String,
    /// The `chdr` payload's leading u32. 0xFFFF on every sample (film and map variant alike);
    /// believed to be the matchmaking hopper id (-1 = none), NOT proven.
    pub chdr_leading: u32,
    pub content: ContentHeader,
    pub author: AuthorChunk,
    pub header: FilmHeader,
    pub data: FilmData,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilmError {
    Io(String),
    NotBlf,
    WrongGame(String),
    MissingChunk(&'static str),
    NotAFilm(u32),
}

impl std::fmt::Display for FilmError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FilmError::Io(e) => write!(f, "{e}"),
            FilmError::NotBlf => write!(f, "not a BLF file (no `_blf` chunk)"),
            FilmError::WrongGame(n) => write!(f, "not a Reach film (container is {n:?})"),
            FilmError::MissingChunk(c) => write!(f, "missing `{c}` chunk"),
            FilmError::NotAFilm(t) => write!(f, "content type {t} is not a saved film"),
        }
    }
}

pub fn parse(bytes: &[u8]) -> Result<Film, FilmError> {
    let chunks = blf_chunks(bytes);
    let blf = find_chunk(&chunks, b"_blf").ok_or(FilmError::NotBlf)?;
    let container_name = cstr(blf.payload(bytes), 0x02, 32);
    if container_name != "reach saved film" {
        return Err(FilmError::WrongGame(container_name));
    }
    let chdr = find_chunk(&chunks, b"chdr").ok_or(FilmError::MissingChunk("chdr"))?;
    let cp = chdr.payload(bytes);
    let chdr_leading = u32le(cp, 0);
    let content = ContentHeader::parse(cp.get(4..).unwrap_or(&[]));
    if content.content_type != 3 && content.content_type != 4 {
        return Err(FilmError::NotAFilm(content.content_type));
    }
    let author = find_chunk(&chunks, b"athr").map(|c| AuthorChunk::parse(c.payload(bytes))).unwrap_or_default();
    let flmh_c = find_chunk(&chunks, b"flmh").ok_or(FilmError::MissingChunk("flmh"))?;
    let header = FilmHeader::parse(flmh_c.payload(bytes), flmh_c.version);
    let data = find_chunk(&chunks, FLMD).map(|c| FilmData::parse(c, bytes)).unwrap_or_default();
    Ok(Film { bytes: bytes.to_vec(), chunks, container_name, chdr_leading, content, author, header, data })
}

/// Every folder on this machine that can hold a Reach saved film, and the extensions they use.
///
/// **One implementation, because there were six** — the app's browser, the metadata card, the
/// export path and three test harnesses each had their own copy, and they disagreed about which
/// films exist.
///
/// Two bugs made the whole Linux install invisible, and they were independent:
///
/// 1. The scan went **two levels** below `/mnt`, `/run/media` and `/media`, but a Proton prefix
///    is **seven** levels down
///    (`<lib>/steamapps/compatdata/<appid>/pfx/drive_c/users/<user>/AppData/LocalLow/...`).
/// 2. It joined **`Users`** with a capital U. A Windows partition has that; a Proton prefix has
///    lowercase **`users`**, and on ext4 that is a hard miss.
///
/// So `film folders` listed only the NTFS Windows partition, and none of the films the Linux
/// install writes — including the one the game is recording right now.
pub const FILM_EXTENSIONS: [&str; 3] = ["mov", "film", "temp"];

/// Case-insensitive child lookup: the one thing that makes a Windows-shaped path work on a
/// case-sensitive filesystem.
pub(crate) fn child_ci(dir: &Path, name: &str) -> Option<std::path::PathBuf> {
    let exact = dir.join(name);
    if exact.is_dir() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case(name))
        .map(|e| e.path())
        .filter(|p| p.is_dir())
}

/// Follow a chain of case-insensitive path segments.
pub(crate) fn descend_ci(dir: &Path, parts: &[&str]) -> Option<std::path::PathBuf> {
    let mut p = dir.to_path_buf();
    for part in parts {
        p = child_ci(&p, part)?;
    }
    Some(p)
}

/// The two film folders under one `.../MCC*/Temporary`.
fn temp_film_dirs(temp: &Path, out: &mut Vec<std::path::PathBuf>) {
    if let Some(d) = descend_ci(temp, &["UserContent", "HaloReach", "Movie"]) {
        out.push(d);
    }
    // The autosave queue, which is where an IN-PROGRESS recording lives (`.temp`) alongside the
    // finished copies (`.film`).
    if let Some(d) = descend_ci(temp, &["HaloReach", "autosave"]) {
        out.push(d);
    }
}

/// Given a directory that may be a Windows user tree (`<root>/users/<user>/AppData/LocalLow`),
/// add every `MCC*/Temporary` film folder under it.
fn scan_user_tree(users: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(users) else { return };
    for u in rd.flatten() {
        let Some(ll) = descend_ci(&u.path(), &["AppData", "LocalLow"]) else { continue };
        let Ok(rd2) = std::fs::read_dir(&ll) else { continue };
        for mcc in rd2.flatten() {
            if !mcc.file_name().to_string_lossy().to_ascii_uppercase().starts_with("MCC") {
                continue;
            }
            if let Some(t) = child_ci(&mcc.path(), "Temporary") {
                temp_film_dirs(&t, out);
            }
        }
    }
}

/// The candidate roots every MCC-directory search starts from: mounted volumes (two levels,
/// which is where a Windows partition sits), the home directory, and a literal Windows drive for
/// the Windows build.
///
/// Film folders and the MCC *settings* tree ([`crate::mccimport`]) live in the same places, so
/// there is ONE enumeration of those places and both callers use it. The alternative — a second
/// copy in the settings importer — is how the two drift apart the next time a drive moves.
pub(crate) fn search_roots() -> Vec<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from("C:\\")];
    for base in ["/mnt", "/run/media", "/media"] {
        let Ok(rd) = std::fs::read_dir(base) else { continue };
        for e in rd.flatten() {
            roots.push(e.path());
            let Ok(rd2) = std::fs::read_dir(e.path()) else { continue };
            for e2 in rd2.flatten() {
                roots.push(e2.path());
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(home));
    }
    roots
}

/// Every folder that can hold a Reach saved film, deduplicated, existing ones only.
pub fn film_search_dirs() -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = Vec::new();
    // An explicit override, for tests and for an install this cannot guess.
    if let Ok(v) = std::env::var("HMS_FILM_DIRS") {
        for p in v.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            out.push(std::path::PathBuf::from(p));
        }
    }
    let roots = search_roots();

    for root in &roots {
        // (a) a Windows partition mounted on Linux, or the Windows build's own drive.
        if let Some(users) = child_ci(root, "Users") {
            scan_user_tree(&users, &mut out);
        }
        // (b) a STEAM LIBRARY: every Proton prefix under it. Enumerated rather than hard-coded,
        // because this machine alone has three MCC prefixes (`976730`, `976730-assembly`,
        // `976730-modified-by-assembly`).
        for lib in steam_libraries(root) {
            let Some(compat) = descend_ci(&lib, &["steamapps", "compatdata"]) else { continue };
            let Ok(rd) = std::fs::read_dir(&compat) else { continue };
            for app in rd.flatten() {
                let Some(users) = descend_ci(&app.path(), &["pfx", "drive_c", "users"]) else { continue };
                scan_user_tree(&users, &mut out);
            }
        }
    }
    out.sort();
    out.dedup();
    out.retain(|p| p.is_dir());
    out
}

/// Steam libraries at or just below `root`. A library is any directory containing `steamapps`.
pub(crate) fn steam_libraries(root: &Path) -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if child_ci(root, "steamapps").is_some() {
        v.push(root.to_path_buf());
    }
    // One level down covers `<mount>/SteamLibrary` and `~/.local/share/Steam`.
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten().take(4096) {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            if child_ci(&p, "steamapps").is_some() {
                v.push(p.clone());
            }
            // and the standard `~/.local/share/Steam` / `~/.steam/steam` shapes
            for deeper in ["share/Steam", "steam"] {
                let parts: Vec<&str> = deeper.split('/').collect();
                if let Some(d) = descend_ci(&p, &parts) {
                    if child_ci(&d, "steamapps").is_some() {
                        v.push(d);
                    }
                }
            }
        }
    }
    v.sort();
    v.dedup();
    v
}

/// Is this a file HMS will try to open as a Reach saved film?
pub fn is_film_file(p: &Path) -> bool {
    p.extension().map_or(false, |x| FILM_EXTENSIONS.iter().any(|e| x.eq_ignore_ascii_case(e)))
}

/// Is this file an IN-PROGRESS recording rather than a finished film?
///
/// MCC writes `.temp` while a match is being recorded and renames it on finalisation. The
/// container and the 128,288-byte `flmh` are already complete and identical to a finished film;
/// only `flmd`'s size field — written at finalisation — is absent, and this reader never trusts
/// it anyway because it frames the stream forward and stops at the last complete record.
pub fn is_in_progress(p: &Path) -> bool {
    p.extension().map_or(false, |x| x.eq_ignore_ascii_case("temp"))
}

/// Finished films only — the ones whose container invariants are complete.
///
/// The spec tests assert things that are only true once a film has been finalised (`flmd`'s size
/// field, the `_eof` offset, the 16-byte pad), so they take this rather than [`find_films`].
pub fn find_finished_films() -> Vec<std::path::PathBuf> {
    find_films().into_iter().filter(|p| !is_in_progress(p)).collect()
}

/// The corpus the spec gates were MEASURED against, reproduced EXACTLY as it was before film
/// discovery was widened.
///
/// This is deliberately the old narrow walk and not a filter over the new one. Widening discovery
/// to the Linux install is the point of [`find_films`], but it also changes which films — and,
/// through them, which `.mvar` candidate folders — these gates see, and a gate whose corpus
/// changes underneath it no longer asserts what it was written to assert. Keeping the two apart
/// means the feature cannot quietly weaken the spec.
///
/// Two things the wider corpus surfaced immediately, both recorded in
/// `re-notes/reach_saved_film_layout.md` rather than papered over:
/// a film and its own profile's `.mvar` disagreeing on the Forge LABEL list, and the
/// film-to-`.mvar` match being ambiguous (several profiles hold the same variant at different
/// points in its editing history, agreeing on title, map id, object count, spent budget, palette
/// CRC and variant checksum). Both are worth fixing on their own terms.
pub fn spec_corpus() -> Vec<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from("C:\\")];
    for base in ["/mnt", "/run/media", "/media"] {
        let Ok(rd) = std::fs::read_dir(base) else { continue };
        for e in rd.flatten() {
            roots.push(e.path());
            let Ok(rd2) = std::fs::read_dir(e.path()) else { continue };
            for e2 in rd2.flatten() {
                roots.push(e2.path());
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(home));
    }
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    for root in roots {
        let Ok(rd) = std::fs::read_dir(root.join("Users")) else { continue };
        for u in rd.flatten() {
            let Ok(rd2) = std::fs::read_dir(u.path().join("AppData").join("LocalLow")) else { continue };
            for mcc in rd2.flatten() {
                if !mcc.file_name().to_string_lossy().starts_with("MCC") {
                    continue;
                }
                let t = mcc.path().join("Temporary");
                dirs.push(t.join("UserContent").join("HaloReach").join("Movie"));
                dirs.push(t.join("HaloReach").join("autosave"));
            }
        }
    }
    let mut out: Vec<std::path::PathBuf> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let ok = p
                .extension()
                .map_or(false, |x| x.eq_ignore_ascii_case("mov") || x.eq_ignore_ascii_case("film"));
            if ok && std::fs::metadata(&p).map_or(false, |m| m.len() > 0x20000) {
                out.push(p);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every film file in every discovered folder, biggest first.
///
/// An in-progress `.temp` is included: it is the same container, its `flmh` is complete at the
/// same 128,288 bytes, and only the `flmd` size field (written at finalisation) differs — which
/// this reader never trusts anyway, because it frames the stream forward and stops at the last
/// complete record.
pub fn find_films() -> Vec<std::path::PathBuf> {
    let mut v: Vec<(u64, std::path::PathBuf)> = Vec::new();
    for d in film_search_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !is_film_file(&p) {
                continue;
            }
            let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            // Below the header prefix there is nothing to read; MCC also leaves zero-byte stubs.
            if len as usize >= HEADER_PREFIX_LEN {
                v.push((len, p));
            }
        }
    }
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.dedup_by(|a, b| a.1 == b.1);
    v.into_iter().map(|(_, p)| p).collect()
}

pub fn open(path: &Path) -> Result<Film, FilmError> {
    let bytes = std::fs::read(path).map_err(|e| FilmError::Io(format!("{}: {e}", path.display())))?;
    parse(&bytes)
}

/// The bytes the engine itself reads to validate and set up a film: every chunk up to and
/// including `flmd`'s 16-byte header (`ReadFilmChunks` reads exactly `min(file_size, 0x1F8A8)`).
pub const HEADER_PREFIX_LEN: usize = 0x1F8A8;

/// Open a film reading ONLY its header prefix. A saved film can be 150 MB; a file browser row or
/// a metadata card must not read all of it. Everything in `content` / `author` / `header` is
/// complete (the `flmh` chunk ends well inside the prefix); `data` comes back empty, so
/// `FilmData::records` is NOT available from this path — use `open` for that.
pub fn open_header(path: &Path) -> Result<Film, FilmError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| FilmError::Io(format!("{}: {e}", path.display())))?;
    let mut buf = vec![0u8; HEADER_PREFIX_LEN];
    let mut got = 0usize;
    while got < buf.len() {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) => return Err(FilmError::Io(format!("{}: {e}", path.display()))),
        }
    }
    buf.truncate(got);
    parse(&buf)
}

impl Film {
    /// Film length in seconds, as the content header stamps it (= `tick_count / 60`).
    pub fn seconds(&self) -> u32 {
        self.content.seconds
    }
    /// Recorded ticks. One tick = one `flmd` record = 1/60 s.
    pub fn tick_count(&self) -> i32 {
        self.header.tick_count
    }
    /// The configuration slot of the player who RECORDED this film, matched by XUID (robust) and
    /// falling back to the gamertag. Needed to bridge a biped's owner index to a named player.
    pub fn recorder_slot(&self) -> Option<u8> {
        let xuid = self.content.created_xuid;
        let tag = &self.content.created_by;
        self.header
            .players
            .iter()
            .position(|p| p.present() && xuid != 0 && p.xuid == xuid)
            .or_else(|| {
                self.header.players.iter().position(|p| p.present() && !tag.is_empty() && &p.gamertag == tag)
            })
            .map(|i| i as u8)
    }

    pub fn present_players(&self) -> impl Iterator<Item = &FilmPlayer> {
        self.header.players.iter().filter(|p| p.present())
    }
    pub fn player_names(&self) -> Vec<String> {
        self.present_players().map(|p| p.gamertag.clone()).collect()
    }
    pub fn forge_object_count(&self) -> usize {
        self.header.map_variant.present_objects().count()
    }
    /// A one-screen diagnostic dump.
    pub fn summary(&self) -> String {
        let mv = &self.header.map_variant;
        let h = &self.header;
        let mut s = String::new();
        s.push_str(&format!("container        {}\n", self.container_name));
        s.push_str(&format!("content          {} (type {})\n", self.content.content_type_name(), self.content.content_type));
        s.push_str(&format!("title            {:?}\n", self.content.title));
        s.push_str(&format!("description      {:?}\n", self.content.description));
        s.push_str(&format!("length           {} s = {} ticks @ {} Hz\n", self.content.seconds, h.tick_count, h.tick_rate));
        s.push_str(&format!("author           {:?} (xuid {:#x})\n", self.content.created_by, self.content.created_xuid));
        s.push_str(&format!("game id          {:#018x}  film id {:#x}\n", self.content.game_id, h.film_id));
        s.push_str(&format!("activity/mode/engine  {} / {} / {}\n", self.content.activity, self.content.game_mode, self.content.engine));
        s.push_str(&format!("map id           {}\n", self.content.map_id));
        s.push_str(&format!("scenario         {}\n", h.scenario_path));
        s.push_str(&format!("build            {:?} / {:?}  (flmh ver {})\n", self.author.build_string, h.build_name, h.chunk_version));
        s.push_str(&format!("game variant     {:?} by {:?} (engine type {})\n", h.game_variant.title, h.game_variant.created_by, h.game_engine_type));
        if mv.is_empty() {
            s.push_str("map variant      <none: stock map>\n");
        } else {
            s.push_str(&format!("map variant      {:?} by {:?} (v{})\n", mv.metadata.title, mv.metadata.created_by, mv.version));
            s.push_str(&format!("  bounds         {:?}\n", mv.world_bounds));
            s.push_str(&format!("  budget         {} / {}   quotas {}  palette crc {:#x}\n", mv.budget_spent, mv.budget_max, mv.num_quotas, mv.palette_crc));
            s.push_str(&format!("  forge objects  {} of {} slots\n", self.forge_object_count(), mv.objects.len()));
            if !mv.labels.is_empty() {
                s.push_str(&format!("  labels         {:?}\n", mv.labels));
            }
        }
        for p in self.present_players() {
            s.push_str(&format!("player {:>2}        {:?} [{}] xuid {:#x} team {}\n", p.index, p.gamertag, p.service_tag, p.xuid, p.team));
        }
        for m in &h.markers {
            s.push_str(&format!("marker           tick {:>7} {:?}\n", m.tick, m.text));
        }
        let (mut t, mut g, mut sg, mut cz) = (0, 0, 0, 0);
        for r in &self.data.records {
            match r.kind {
                RecordKind::Tick => t += 1,
                RecordKind::GameState => g += 1,
                RecordKind::Segment => sg += 1,
                RecordKind::Compressed => cz += 1,
            }
        }
        s.push_str(&format!(
            "flmd             {} bytes at {:#x}: {} records ({} tick, {} state, {} segment, {} zlib), framing {}\n",
            self.data.stream_len, self.data.stream_at, self.data.records.len(), t, g, sg, cz,
            if self.data.framing_exact { "exact" } else { "BROKEN" }
        ));
        if let (Some(mn), Some(mx)) = (
            self.data.records.iter().map(|r| r.len).min(),
            self.data.records.iter().map(|r| r.len).max(),
        ) {
            let sum: usize = self.data.records.iter().map(|r| r.len).sum();
            s.push_str(&format!("  record bytes   min {mn} max {mx} mean {}\n", sum / self.data.records.len().max(1)));
        }
        for c in &self.chunks {
            s.push_str(&format!("  chunk {:<5} @{:#08x} size {:<9} ver {:<6} flags {}\n", c.name(), c.offset, c.size, c.version, c.flags));
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------- #emblem  the recorded emblem --------

    /// The emblem offsets, against a record laid out exactly as the engine lays it out.
    ///
    /// Cache-free and film-free: `s_emblem_info` is at record +0x50 because the record is a
    /// 0x20-byte identity header plus the 0x80-byte `s_player_configuration`, whose
    /// `client.appearance` is at +0x28, whose `s_emblem_info` is at +0x08. This writes a known
    /// emblem at that offset and requires it to come back out, and requires the engine's own
    /// validity rules -- the must-be-zero pad above all -- to be enforced rather than ignored.
    #[test]
    fn the_emblem_is_read_from_the_offset_the_engine_puts_it_at() {
        let mut raw = vec![0u8; 0xA0];
        // fg 16 (smiley), bg 44 (cog), no flags, colours 21 / 12 / 19 -- the exact emblem the
        // 16-player test clip records for every one of its slots.
        raw[0x50..0x58].copy_from_slice(&[16, 44, 0, 21, 12, 19, 0, 0]);
        let e = EmblemInfo::parse(&raw).expect("a well-formed emblem");
        assert_eq!(
            (e.foreground, e.background, e.flags, e.color_primary, e.color_secondary, e.color_background),
            (16, 44, 0, 21, 12, 19)
        );
        assert!(e.valid() && e.in_range());
        assert!(!e.alternate_foreground() && !e.flip_foreground() && !e.flip_background());
        // The flags, bit by bit.
        raw[0x52] = 0b101;
        let e = EmblemInfo::parse(&raw).unwrap();
        assert!(e.alternate_foreground() && !e.flip_foreground() && e.flip_background());
        // A flag bit outside the engine's three is not an emblem.
        raw[0x52] = 0b1000;
        assert!(EmblemInfo::parse(&raw).is_none(), "flags & !7 must be rejected");
        raw[0x52] = 0;
        // The PAD. The engine asserts it is zero, so a record whose two bytes there are not is
        // not an emblem -- and this is the check that pins the offset: if +0x50 were the wrong
        // place, real records would fail it, and 113 of them do not.
        raw[0x56] = 1;
        assert!(EmblemInfo::parse(&raw).is_none(), "a non-zero pad must be rejected");
        raw[0x56] = 0;
        // Colour index out of the engine's 0..0x1F.
        raw[0x53] = 0x20;
        assert!(EmblemInfo::parse(&raw).is_none());
        raw[0x53] = 21;
        // In the engine's range but outside the art Reach ships.
        raw[0x50] = 0x7E;
        let e = EmblemInfo::parse(&raw).expect("the engine's own assert still passes");
        assert!(!e.in_range(), "0x7E is a legal byte but there are only 80 foreground designs");
        // A record too short to hold one says so instead of reading past the end.
        assert!(EmblemInfo::parse(&raw[..0x50]).is_none());
    }

    /// Every player of every real recording on this machine carries a WELL-FORMED emblem at
    /// +0x50, and the clips say how many DISTINCT ones they hold.
    ///
    /// The second half is the finding the UI reports: MCC writes the same emblem into every
    /// slot. Measured here as one distinct emblem across the sixteen players of the 7:11 clip.
    /// The assertion is only on well-formedness, because "how many distinct" is a property of
    /// the recordings on this machine and not something to pin.
    #[test]
    fn every_recorded_player_carries_a_wellformed_emblem() {
        let films = spec_corpus();
        if films.is_empty() {
            eprintln!("film: no real recordings on this machine -- emblem corpus not measured");
            return;
        }
        let (mut players, mut with) = (0usize, 0usize);
        for path in films.iter() {
            let Ok(f) = open_header(path) else { continue };
            let present: Vec<&FilmPlayer> = f.header.players.iter().filter(|p| p.present()).collect();
            if present.is_empty() {
                continue;
            }
            let mut set: std::collections::HashSet<[u8; 6]> = Default::default();
            for p in &present {
                players += 1;
                let Some(e) = p.emblem() else { continue };
                with += 1;
                assert!(e.in_range(), "{:?} slot {}: emblem outside the shipped art: {}", path.file_name(), p.index, e.summary());
                set.insert([e.foreground, e.background, e.flags, e.color_primary, e.color_secondary, e.color_background]);
            }
            eprintln!(
                "film: {:<46} {} player(s), {} distinct emblem(s)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                present.len(),
                set.len()
            );
        }
        eprintln!("film: {with} of {players} recorded players carry a well-formed emblem at +0x{EMBLEM_AT:02X}");
        // The claim this gate exists to defend: the field IS there, on essentially every record.
        assert!(players > 0);
        assert!(
            with * 10 >= players * 9,
            "only {with} of {players} records parsed as an emblem -- the offset or the rules are wrong"
        );
    }

    // -------- synthetic films, built straight from the spec --------

    fn push_chunk(out: &mut Vec<u8>, magic: &[u8; 4], version: u16, flags: u16, payload: &[u8]) {
        out.extend_from_slice(magic);
        out.extend_from_slice(&((payload.len() + 12) as u32).to_be_bytes());
        out.extend_from_slice(&version.to_be_bytes());
        out.extend_from_slice(&flags.to_be_bytes());
        out.extend_from_slice(payload);
    }
    fn put(buf: &mut [u8], at: usize, bytes: &[u8]) {
        buf[at..at + bytes.len()].copy_from_slice(bytes);
    }
    fn put_w(buf: &mut [u8], at: usize, s: &str) {
        let mut o = at;
        for u in s.encode_utf16() {
            buf[o..o + 2].copy_from_slice(&u.to_le_bytes());
            o += 2;
        }
    }

    /// `s_content_item_metadata` (0x2B0 bytes).
    fn synth_metadata(content_type: u32, map_id: u32, seconds: u32, title: &str, who: &str) -> Vec<u8> {
        let mut c = vec![0u8; METADATA_SIZE];
        put(&mut c, 0x00, &content_type.to_le_bytes());
        put(&mut c, 0x08, &0x1122_3344_5566_7788u64.to_le_bytes());
        put(&mut c, 0x10, &0x1122_3344_5566_7788u64.to_le_bytes());
        put(&mut c, 0x18, &0x1122_3344_5566_7788u64.to_le_bytes());
        put(&mut c, 0x20, &0xDEAD_BEEF_0BAD_F00Du64.to_le_bytes());
        c[0x28] = 3;
        c[0x29] = 3;
        c[0x2A] = 2;
        put(&mut c, 0x2C, &map_id.to_le_bytes());
        put(&mut c, 0x30, &16u32.to_le_bytes());
        put(&mut c, 0x38, &0x6774_A7DFu64.to_le_bytes());
        put(&mut c, 0x40, &0x0009_01F7_6D76_2AB6u64.to_le_bytes());
        put(&mut c, 0x48, who.as_bytes());
        put(&mut c, 0x58, &1u32.to_le_bytes());
        put(&mut c, 0x5C, &0x6774_A7DFu64.to_le_bytes());
        put(&mut c, 0x64, &0x0009_01F7_6D76_2AB6u64.to_le_bytes());
        put(&mut c, 0x6C, who.as_bytes());
        put(&mut c, 0x7C, &1u32.to_le_bytes());
        put_w(&mut c, 0x80, title);
        put_w(&mut c, 0x180, "a synthetic header");
        put(&mut c, 0x280, &seconds.to_le_bytes());
        c
    }

    fn synth_object(i: usize) -> Vec<u8> {
        let mut r = vec![0u8; map_variant::OBJECT_STRIDE];
        put(&mut r, 0x00, &1u16.to_le_bytes()); // occupied
        put(&mut r, 0x02, &((71 + i) as u16).to_le_bytes()); // quota index
        put(&mut r, 0x04, &(-1i32).to_le_bytes());
        for (k, v) in [10.0 + i as f32, 20.0 + i as f32, 30.0 + i as f32].iter().enumerate() {
            put(&mut r, 0x08 + k * 4, &v.to_le_bytes());
        }
        put(&mut r, 0x14, &1.0f32.to_le_bytes()); // forward = +X
        put(&mut r, 0x28, &1.0f32.to_le_bytes()); // up = +Z
        put(&mut r, 0x2C, &0xFFFFu16.to_le_bytes());
        r[0x40] = 3; // box boundary
        for (k, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            put(&mut r, 0x30 + k * 4, &v.to_le_bytes());
        }
        r[0x42] = 30; // respawn
        r[0x43] = 1; // weapon
        r
    }

    fn synth_map_variant(n_objects: usize) -> Vec<u8> {
        use map_variant::*;
        let mut mv = vec![0u8; game_options::MAP_VARIANT_LEN];
        put(&mut mv, METADATA, &synth_metadata(5, 3006, 0, "Synthetic Canvas", "Sopitive6153"));
        put(&mut mv, VERSION, &32u16.to_le_bytes());
        put(&mut mv, NUM_QUOTAS, &145u16.to_le_bytes());
        put(&mut mv, MAP_ID, &3006u32.to_le_bytes());
        for (k, v) in [-203.6f32, 277.664, -360.826, 324.925, -25.0101, 117.791].iter().enumerate() {
            put(&mut mv, WORLD_BOUNDS + k * 4, &v.to_le_bytes());
        }
        put(&mut mv, BUDGET_MAX, &30000u32.to_le_bytes());
        put(&mut mv, BUDGET_SPENT, &9160u32.to_le_bytes());
        put(&mut mv, CHECKSUM, &0xFFFF_FFFFu32.to_le_bytes());
        put(&mut mv, PALETTE_CRC, &0x8B33_E8ACu32.to_le_bytes());
        // two labels in the pool, with their offsets
        let pool = b"team_zone\0spartan\0";
        put(&mut mv, LABEL_POOL, pool);
        put(&mut mv, LABEL_POOL_USED, &(pool.len() as u32).to_le_bytes());
        put(&mut mv, LABEL_OFFSETS, &0u16.to_le_bytes());
        put(&mut mv, LABEL_OFFSETS + 2, &10u16.to_le_bytes());
        put(&mut mv, LABEL_COUNT, &2u32.to_le_bytes());
        for i in 0..n_objects {
            put(&mut mv, OBJECTS + i * OBJECT_STRIDE, &synth_object(i));
        }
        // leave the rest of the 651 slots absent, but stamp the engine's sentinel on them
        for i in n_objects..OBJECT_SLOTS {
            let at = OBJECTS + i * OBJECT_STRIDE;
            put(&mut mv, at + 0x02, &0xFFFFu16.to_le_bytes());
            put(&mut mv, at + 0x04, &(-1i32).to_le_bytes());
        }
        for q in 0..145 {
            put(&mut mv, QUOTAS + q * QUOTA_STRIDE, &[0, 4, (q % 7) as u8]);
        }
        mv
    }

    fn synth_flmh(n_objects: usize, ticks: i32) -> Vec<u8> {
        use flmh::*;
        let mut p = vec![0u8; PAYLOAD_LEN];
        put(&mut p, BUILD_NAME, b"untracked version");
        put(&mut p, FORMAT_REVISION, &5u32.to_le_bytes());
        put(&mut p, COOKIE_LEN, &60u32.to_le_bytes());
        put(&mut p, COOKIE, &[0xA5u8; 60]);
        put(&mut p, SESSION_ID, b"Tue_12312024_192639@x64@user@machine");
        put(&mut p, RECORD_UNIX, &0x6774_A7DFu32.to_le_bytes());
        put(&mut p, TICK_COUNT, &ticks.to_le_bytes());
        put(&mut p, CLIP_START_TICK, &(-1i32).to_le_bytes());
        put(&mut p, MARKERS, &601i32.to_le_bytes());
        put(&mut p, MARKERS + 4, b"rep qual min 0.99 for mchn 1");
        put(&mut p, MARKER_COUNT, &1i32.to_le_bytes());
        let go = GAME_OPTIONS;
        put(&mut p, go + game_options::TICK_RATE, &60u16.to_le_bytes());
        put(&mut p, go + game_options::FILM_ID, &0x19209AF3u32.to_le_bytes());
        put(&mut p, go + game_options::OPTIONS_VERSION, &4i32.to_le_bytes());
        put(&mut p, go + game_options::MAP_ID, &3006u32.to_le_bytes());
        put(&mut p, go + game_options::SCENARIO_PATH, br"haloreach\maps\forge_halo");
        put(&mut p, go + game_options::MACHINE_MASK, &0x7u32.to_le_bytes());
        for i in 0..4 {
            let at = go + game_options::PLAYERS + i * game_options::PLAYER_STRIDE;
            put(&mut p, at + 0x00, &1u32.to_le_bytes());
            p[at + 0x08] = (i % 2) as u8;
            put(&mut p, at + 0x10, &[1, 2, 3, 4, 5, i as u8]);
            put(&mut p, at + 0x16, &(0x0009_0000_0000_0001u64 + i as u64).to_le_bytes());
            put_w(&mut p, at + 0x20, &format!("Player{i}"));
            put_w(&mut p, at + 0x64, "117");
        }
        put(&mut p, go + game_options::GAME_VARIANT_ENGINE, &2u32.to_le_bytes());
        put(&mut p, go + game_options::GAME_VARIANT_METADATA, &synth_metadata(6, 3006, 0, "Synthetic Slayer", "Sopitive6153"));
        put(&mut p, go + game_options::MAP_VARIANT, &synth_map_variant(n_objects));
        p
    }

    /// A whole synthetic film, laid out exactly as the spec describes: big-endian chunk sizes
    /// everywhere except `flmd`'s little-endian `payload + 12`, and `_eof`'s little-endian
    /// self-offset.
    fn synth_film(n_objects: usize, record_lens: &[usize]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut blf = vec![0u8; 36];
        blf[0] = 0xFF;
        blf[1] = 0xFE;
        put(&mut blf, 0x02, b"reach saved film");
        push_chunk(&mut out, b"_blf", 1, 2, &blf);
        let mut chdr = vec![0u8; CHDR_PAYLOAD_SIZE];
        put(&mut chdr, 0, &0xFFFFu32.to_le_bytes());
        put(&mut chdr, 4, &synth_metadata(3, 3006, (record_lens.len() / 60) as u32, "Synthetic Film", "Sopitive6153"));
        push_chunk(&mut out, b"chdr", 10, 2, &chdr);
        let mut athr = vec![0u8; 68];
        put(&mut athr, 0x18, b"untracked version");
        push_chunk(&mut out, b"athr", 3, 1, &athr);
        push_chunk(&mut out, b"flmh", flmh::CHUNK_VERSION, 2, &synth_flmh(n_objects, record_lens.len() as i32));
        push_chunk(&mut out, b"ssig", 1, 1, &vec![0u8; 44]);
        let mut stream = Vec::new();
        for (i, &l) in record_lens.iter().enumerate() {
            stream.extend_from_slice(&(l as u32).to_le_bytes()); // type 0 = tick
            stream.extend((0..l).map(|k| (i + k) as u8));
        }
        out.extend_from_slice(b"flmd");
        out.extend_from_slice(&((12 + stream.len()) as u32).to_le_bytes());
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&stream);
        let eof_at = out.len() as u32;
        out.extend_from_slice(b"_eof");
        out.extend_from_slice(&17u32.to_be_bytes());
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&eof_at.to_le_bytes());
        out.push(0);
        let content_len = out.len() as u32;
        let meta_at = 48 + 12 + 4;
        out[meta_at + 0x04..meta_at + 0x08].copy_from_slice(&content_len.to_le_bytes());
        while out.len() % 16 != 0 {
            out.push(0);
        }
        out
    }

    fn lens(n: usize) -> Vec<usize> {
        (0..n).map(|i| 26 + (i * 37) % 600).collect()
    }

    #[test]
    fn synthetic_film_round_trips_the_container() {
        let rl = lens(125);
        let bytes = synth_film(3, &rl);
        let f = parse(&bytes).expect("synthetic film parses");
        assert_eq!(
            f.chunks.iter().map(|c| c.name()).collect::<Vec<_>>(),
            ["_blf", "chdr", "athr", "flmh", "ssig", "flmd", "_eof"]
        );
        let mut at = 0usize;
        for c in &f.chunks {
            assert_eq!(c.offset, at, "chunk {} is not contiguous", c.name());
            at += c.size;
        }
        assert_eq!(at, f.content.file_size as usize, "chdr file_size must be the end of `_eof`");
        assert!(bytes.len() % 16 == 0 && bytes.len() - at < 16);
        let flmd = f.chunks.iter().find(|c| &c.magic == b"flmd").unwrap();
        assert_eq!(flmd.size, f.data.size_field as usize + 4, "flmd size under-counts by 4");
        assert_eq!(f.data.stream_at, flmd.offset + 0x10);
        assert_eq!(f.data.size_field as usize, f.data.stream_len + 12);
        assert_eq!(f.chdr_leading, 0xFFFF);
    }

    #[test]
    fn synthetic_film_frames_every_record() {
        let rl = lens(300);
        let bytes = synth_film(1, &rl);
        let f = parse(&bytes).unwrap();
        assert!(f.data.framing_exact, "records must tile the flmd stream exactly");
        assert_eq!(f.data.records.len(), rl.len());
        assert_eq!(f.data.ticks().count(), rl.len());
        assert_eq!(f.data.records.len() as i32, f.header.tick_count, "records == tick count");
        assert_eq!(f.seconds(), (rl.len() / 60) as u32);
        for (i, r) in f.data.records.iter().enumerate() {
            assert_eq!(r.kind, RecordKind::Tick);
            assert_eq!(r.len, rl[i]);
            assert_eq!(r.index, i);
            let b = r.body(&bytes);
            assert_eq!(b.len(), rl[i]);
            assert_eq!(b[0], i as u8, "record {i} body");
        }
    }

    #[test]
    fn synthetic_film_reads_its_header_fields() {
        let bytes = synth_film(5, &lens(60));
        let f = parse(&bytes).unwrap();
        assert_eq!(f.container_name, "reach saved film");
        assert_eq!(f.content.content_type, 3);
        assert_eq!(f.content.content_type_name(), "saved film");
        assert_eq!(f.content.title, "Synthetic Film");
        assert_eq!(f.content.map_id, 3006);
        assert_eq!(f.content.created_by, "Sopitive6153");
        assert_eq!(f.content.game_id, 0xDEAD_BEEF_0BAD_F00D);
        assert_eq!(f.header.chunk_version, flmh::CHUNK_VERSION);
        assert_eq!(f.header.format_revision, 5);
        assert_eq!(f.header.tick_rate, 60);
        assert_eq!(f.header.options_version, 4);
        assert_eq!(f.header.film_id, 0x19209AF3);
        assert!(!f.header.is_clip);
        assert_eq!(f.header.clip_start_tick, -1);
        assert_eq!(f.header.scenario_path, r"haloreach\maps\forge_halo");
        assert_eq!(f.header.scenario_name(), "forge_halo");
        assert_eq!(f.header.map_id, 3006);
        assert_eq!(f.header.game_variant.content_type, 6);
        assert_eq!(f.header.game_variant.title, "Synthetic Slayer");
        assert_eq!(f.header.game_engine_type, 2);
        assert_eq!(f.header.markers.len(), 1);
        assert_eq!(f.header.markers[0].tick, 601);
        assert_eq!(f.player_names(), ["Player0", "Player1", "Player2", "Player3"]);
        assert_eq!(f.header.players[0].xuid, 0x0009_0000_0000_0001);
        assert_eq!(f.header.players[0].service_tag, "117");
        assert_eq!(f.header.players[1].team, 1);
        assert_eq!(f.header.players[0].machine, [1, 2, 3, 4, 5, 0]);
        assert!(!f.header.players[4].present());
        assert_eq!(f.present_players().count(), 4);
    }

    #[test]
    fn synthetic_film_reads_the_embedded_map_variant() {
        let bytes = synth_film(7, &lens(10));
        let f = parse(&bytes).unwrap();
        let mv = &f.header.map_variant;
        assert_eq!(mv.metadata.content_type, 5, "the embedded metadata must be a MAP VARIANT");
        assert_eq!(mv.metadata.title, "Synthetic Canvas");
        assert_eq!(mv.version, 32);
        assert_eq!(mv.num_quotas, 145);
        assert_eq!(mv.map_id, 3006);
        assert_eq!(mv.budget_max, 30000);
        assert_eq!(mv.budget_spent, 9160);
        assert_eq!(mv.checksum, 0xFFFF_FFFF);
        assert_eq!(mv.palette_crc, 0x8B33_E8AC);
        assert_eq!(mv.labels, ["team_zone", "spartan"]);
        assert_eq!(mv.objects.len(), map_variant::OBJECT_SLOTS);
        assert_eq!(mv.quotas.len(), map_variant::QUOTA_SLOTS);
        assert!(!mv.is_empty());
        assert_eq!(f.forge_object_count(), 7);
        let o = &mv.objects[3];
        assert_eq!(o.flags, 1);
        assert_eq!(o.quota_index, 74);
        assert_eq!(o.parent, -1);
        assert_eq!(o.position, [13.0, 23.0, 33.0]);
        assert_eq!(o.forward, [1.0, 0.0, 0.0]);
        assert_eq!(o.up, [0.0, 0.0, 1.0]);
        assert_eq!(o.boundary_shape, 3);
        assert_eq!(o.boundary, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(o.respawn, 30);
        assert_eq!(o.cached_type, 1);
        assert!(!mv.objects[7].present());
        assert_eq!(mv.objects[7].quota_index, 0xFFFF);
    }

    /// `summary()` is the diagnostic surface the GUI/CLI will print; make sure it renders for
    /// both a Forge film and a stock-map one without panicking, and names the key facts.
    #[test]
    fn summary_renders() {
        let bytes = synth_film(5, &lens(120));
        let f = parse(&bytes).unwrap();
        let s = f.summary();
        for want in [
            "reach saved film", "Synthetic Film", "forge_halo", "Synthetic Canvas",
            "Synthetic Slayer", "Player0", "120 records", "framing exact", "2 s = 120 ticks @ 60 Hz",
        ] {
            assert!(s.contains(want), "summary is missing {want:?}:\n{s}");
        }
        // a film with no map variant must say so rather than print a garbage bbox
        let mut bytes = synth_film(0, &lens(10));
        let mv_bounds = 48 + 12 /*chdr*/ + 0x2C0 - 12 + 0x50 + 12
            + flmh::GAME_OPTIONS + game_options::MAP_VARIANT + map_variant::WORLD_BOUNDS;
        for k in 0..3 {
            bytes[mv_bounds + k * 8..mv_bounds + k * 8 + 4].copy_from_slice(&f32::MAX.to_le_bytes());
            bytes[mv_bounds + k * 8 + 4..mv_bounds + k * 8 + 8].copy_from_slice(&(-f32::MAX).to_le_bytes());
        }
        let f = parse(&bytes).unwrap();
        assert!(f.header.map_variant.is_empty(), "the FLT_MAX sentinel must read as empty");
        assert!(f.summary().contains("<none: stock map>"));
    }

    #[test]
    fn non_films_are_rejected_not_mis_parsed() {
        let mut bytes = synth_film(1, &lens(5));
        let meta_at = 48 + 12 + 4;
        bytes[meta_at..meta_at + 4].copy_from_slice(&5u32.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(FilmError::NotAFilm(5))));
        assert!(matches!(parse(b"not a blf file at all, really"), Err(FilmError::NotBlf)));
        // A BLF whose container name is another game's must not be read as Reach.
        let mut bytes = synth_film(1, &lens(5));
        let name_at = 12 + 0x02;
        bytes[name_at..name_at + 16].copy_from_slice(b"halo4 saved film");
        assert!(matches!(parse(&bytes), Err(FilmError::WrongGame(_))));
    }

    #[test]
    fn truncated_and_corrupt_films_do_not_panic() {
        let full = synth_film(4, &lens(40));
        for n in 0..full.len().min(3000) {
            let _ = parse(&full[..n]);
        }
        for n in (0..full.len()).step_by(499) {
            let _ = parse(&full[..n]);
        }
        let flmd = blf_chunks(&full).into_iter().find(|c| &c.magic == b"flmd").unwrap();
        // A zero-length record must stop the framing, not loop forever.
        let mut bad = full.clone();
        bad[flmd.payload_at..flmd.payload_at + 4].copy_from_slice(&0u32.to_le_bytes());
        let f = parse(&bad).unwrap();
        assert!(!f.data.framing_exact);
        assert!(f.data.records.is_empty());
        // An over-long record must be rejected as well (and must not overflow the offset math).
        let mut bad = full.clone();
        bad[flmd.payload_at..flmd.payload_at + 4].copy_from_slice(&0x3FFF_FFFFu32.to_le_bytes());
        let f = parse(&bad).unwrap();
        assert!(!f.data.framing_exact);
    }

    // -------- real films, when this machine has any --------

    /// Reach films live in MCC's LocalLow tree. Found generically — no user path is hard-coded:
    /// any `<root>/Users/<user>/AppData/LocalLow/MCC*/Temporary/...`. Saved films carry the
    /// `.mov` extension, autosave-queue copies `.film`.
    fn sample_films() -> Vec<std::path::PathBuf> {
        // ONE implementation, in `film.rs`. There were six copies of this walk and they
        // disagreed about which films exist -- and all of them missed the entire Linux install,
        // because a Proton prefix is seven levels down and uses a lowercase `users`.
        crate::film::spec_corpus()
    }

    /// Every real Reach film on this machine must satisfy every invariant the spec claims. SKIPS
    /// (loudly) when no film is present, so the suite is green on a bare checkout; the synthetic
    /// tests above carry the format regardless.
    #[test]
    fn real_films_match_the_spec() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film: no real Reach saved films on this machine -- synthetic coverage only");
            return;
        }
        let (mut reach, mut forge, mut stock) = (0usize, 0usize, 0usize);
        for p in &films {
            let Ok(bytes) = std::fs::read(p) else { continue };
            let f = match parse(&bytes) {
                Ok(f) => f,
                // `.mov`/`.film` also exist for other MCC titles; not a failure here.
                Err(FilmError::WrongGame(_)) | Err(FilmError::NotBlf) => continue,
                Err(e) => panic!("{}: {e}", p.display()),
            };
            reach += 1;
            let d = p.display();
            // --- container
            assert_eq!(
                f.chunks.iter().map(|c| c.name()).collect::<Vec<_>>(),
                ["_blf", "chdr", "athr", "flmh", "ssig", "flmd", "_eof"],
                "{d}: unexpected chunk list"
            );
            let mut at = 0usize;
            for c in &f.chunks {
                assert_eq!(c.offset, at, "{d}: chunk {} not contiguous", c.name());
                at += c.size;
            }
            assert_eq!(at, f.content.file_size as usize, "{d}: chdr file_size != end of _eof");
            assert!(bytes.len() - at < 16 && bytes.len() % 16 == 0, "{d}: tail padding is not the 16-byte pad");
            assert_eq!(f.chunks[0].size, 0x30, "{d}: _blf size");
            assert_eq!(f.chunks[1].size, 0x2C0, "{d}: chdr size");
            assert_eq!(f.chunks[2].size, 0x50, "{d}: athr size");
            assert_eq!(f.chunks[4].size, 0x38, "{d}: ssig size");
            let flmh_c = &f.chunks[3];
            assert_eq!(flmh_c.payload_len, flmh::PAYLOAD_LEN, "{d}: flmh size drifted");
            assert_eq!(flmh_c.version, flmh::CHUNK_VERSION, "{d}: flmh version drifted");
            assert_eq!(f.chdr_leading, 0xFFFF, "{d}: chdr leading dword");
            // --- header cross-checks
            assert!(f.content.content_type == 3 || f.content.content_type == 4, "{d}: content type");
            assert_eq!(f.header.map_id, f.content.map_id, "{d}: flmh/chdr map id disagree");
            assert_eq!(f.header.tick_rate, 60, "{d}: tick rate");
            assert_eq!(f.header.options_version, 4, "{d}: game-options version");
            assert_eq!(f.header.format_revision, 5, "{d}: flmh format revision");
            assert!(f.header.scenario_path.starts_with("haloreach\\maps\\"), "{d}: scenario {:?}", f.header.scenario_path);
            assert_eq!(
                f.content.seconds, (f.header.tick_count.max(0) / 60) as u32,
                "{d}: seconds != ticks/60 ({} ticks, {} s)", f.header.tick_count, f.content.seconds
            );
            // --- flmd framing: the spec's central claim about the stream
            assert_eq!(f.data.reserved, 0, "{d}: flmd +0x0C not zero");
            assert_eq!(f.data.size_field as usize, f.data.stream_len + 12, "{d}: flmd size field");
            assert!(f.data.framing_exact, "{d}: flmd records do not tile the stream");
            assert_eq!(
                f.data.records.len() as i32, f.header.tick_count,
                "{d}: {} records but flmh says {} ticks", f.data.records.len(), f.header.tick_count
            );
            assert_eq!(f.data.ticks().count(), f.data.records.len(), "{d}: a retail film held a non-tick record");
            for r in &f.data.records {
                assert!(r.len <= MAX_TICK_RECORD_LEN, "{d}: record {} is {} bytes", r.index, r.len);
            }
            // --- the embedded map variant
            let mv = &f.header.map_variant;
            assert_eq!(mv.objects.len(), map_variant::OBJECT_SLOTS, "{d}: slot count");
            assert_eq!(mv.map_id, f.content.map_id, "{d}: embedded variant targets another map");
            assert_eq!(mv.metadata.content_type, 5, "{d}: embedded metadata is not a map variant");
            assert_eq!(mv.version, 32, "{d}: map variant version");
            assert_eq!(f.header.game_variant.content_type, 6, "{d}: embedded game variant metadata");
            assert_eq!(f.header.game_engine_type, f.content.engine as u32, "{d}: engine type != chdr engine");
            let present: Vec<usize> = mv.present_objects().map(|o| o.slot).collect();
            let n = present.len();
            if mv.is_empty() {
                assert_eq!(n, 0, "{d}: empty map variant but {n} placed objects");
                assert_eq!(mv.budget_spent, 0, "{d}: empty map variant with a spent budget");
                stock += 1;
            } else {
                assert!(n > 0, "{d}: real bounds but no Forge objects");
                assert_eq!(present, (0..n).collect::<Vec<_>>(), "{d}: object slots not contiguous from 0");
                let bb = mv.world_bounds;
                for o in mv.present_objects() {
                    let l = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    assert!((l(o.forward) - 1.0).abs() < 0.01, "{d}: slot {} forward not unit", o.slot);
                    assert!((l(o.up) - 1.0).abs() < 0.01, "{d}: slot {} up not unit", o.slot);
                    let dot = o.forward[0] * o.up[0] + o.forward[1] * o.up[1] + o.forward[2] * o.up[2];
                    assert!(dot.abs() < 0.02, "{d}: slot {} basis not orthogonal ({dot})", o.slot);
                    assert!(o.quota_index < mv.num_quotas, "{d}: slot {} quota {} >= {}", o.slot, o.quota_index, mv.num_quotas);
                    // 0 none, 1 sphere, 2 cylinder, 3 box. Non-zero dimensions always imply a
                    // shape; the converse does NOT hold (10 of 9791 real slots are a shape with
                    // all-zero dimensions -- degenerate authored data).
                    assert!(o.boundary_shape <= 3, "{d}: slot {} boundary shape {}", o.slot, o.boundary_shape);
                    assert!(o.boundary == [0.0; 4] || o.boundary_shape != 0, "{d}: slot {} has dims but no shape", o.slot);
                }
                let inside = mv
                    .present_objects()
                    .filter(|o| {
                        let q = o.position;
                        q[0] >= bb[0] - 5.0 && q[0] <= bb[1] + 5.0
                            && q[1] >= bb[2] - 5.0 && q[1] <= bb[3] + 5.0
                            && q[2] >= bb[4] - 5.0 && q[2] <= bb[5] + 5.0
                    })
                    .count();
                assert!(inside * 10 >= n * 9, "{d}: only {inside}/{n} objects inside the variant bounds");
                forge += 1;
            }
            eprintln!(
                "film OK {:<44} {:>4}s {:>6} ticks  map {:<5} {:<20} {:>4} obj  flmd {:>9} B / {:>5} rec  {:?}",
                p.file_name().unwrap_or_default().to_string_lossy(),
                f.seconds(), f.header.tick_count, f.content.map_id, f.header.scenario_name(), n,
                f.data.stream_len, f.data.records.len(), mv.metadata.title
            );
        }
        assert!(reach > 0, "found film-shaped files but none were Reach films");
        eprintln!("film: {reach} Reach film(s) checked -- {forge} with a Forge variant, {stock} on a stock map");
    }

    /// `open_header` must agree with `open` on everything but the stream, on every real film —
    /// that is what lets a 150 MB film show up in a file list without being read.
    #[test]
    fn open_header_matches_a_full_open() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film: no real films -- open_header not cross-checked");
            return;
        }
        let mut n = 0usize;
        for p in &films {
            let Ok(full) = open(p) else { continue };
            let head = open_header(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert_eq!(head.content, full.content, "{}: content header", p.display());
            assert_eq!(head.author, full.author, "{}: author", p.display());
            assert_eq!(head.header, full.header, "{}: film header", p.display());
            assert_eq!(head.container_name, full.container_name);
            // the stream is deliberately absent
            assert!(head.data.records.is_empty(), "{}: open_header read records", p.display());
            assert!(head.bytes.len() <= HEADER_PREFIX_LEN);
            n += 1;
        }
        eprintln!("film: open_header == open (minus the stream) on {n} film(s)");
        assert!(n > 0);
    }

    // -------- agreement with the rest of HMS --------

    fn sample_mvars() -> Vec<std::path::PathBuf> {
        let mut found: Vec<std::path::PathBuf> = Vec::new();
        for d in crate::mapcat::variant_dirs() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
                    found.push(p);
                }
            }
        }
        found.sort();
        found.truncate(16);
        found
    }

    /// The `chdr` STRUCT and the bit-packed header at the head of an `mvar` chunk are two
    /// encodings of the same fields. Proving it on real `.mvar` files is what justifies reading
    /// the struct form here instead of reusing `mvar::parse_header`.
    #[test]
    fn chdr_struct_matches_the_bit_packed_header() {
        let mut checked = 0usize;
        let mut blank_payload = 0usize;
        for p in sample_mvars() {
            let Ok(raw) = std::fs::read(&p) else { continue };
            let data = crate::mvar::expand_blf_cmp(&raw);
            let chunks = blf_chunks(&data);
            let Some(chdr) = find_chunk(&chunks, b"chdr") else { continue };
            let cp = chdr.payload(&data);
            let h = ContentHeader::parse(cp.get(4..).unwrap_or(&[]));
            let Some(v) = crate::mvar::parse_variant(&p) else { continue };
            let d = p.display();
            assert_eq!(h.content_type, 5, "{d}: content type");
            assert_eq!(h.map_id, v.globals.header_map_id, "{d}: map id");
            assert_eq!(h.uid, v.globals.uid, "{d}: uid");
            assert_eq!(h.parent_uid, v.globals.parent_uid, "{d}: parent uid");
            assert_eq!(h.root_uid, v.globals.root_uid, "{d}: root uid");
            assert_eq!(h.game_id, v.globals.game_id, "{d}: game id");
            assert_eq!(h.activity as i8, v.globals.activity, "{d}: activity");
            assert_eq!(h.game_mode, v.globals.game_mode, "{d}: game mode");
            assert_eq!(h.engine, v.globals.engine, "{d}: engine");
            assert_eq!(h.category as u8 as i8, v.globals.category, "{d}: category");
            // The two encodings share a layout but NOT always a value: a few shipped hopper
            // variants carry a `chdr` title/description the bit-packed payload leaves blank
            // (HReach_Chapter_*_Title.mvar). So require agreement unless the payload value is
            // empty, and count how often that happens.
            for (what, a, b) in [
                ("title", &h.title, &v.title),
                ("description", &h.description, &v.description),
                ("author", &h.created_by, &v.author),
                ("editor", &h.modified_by, &v.editor),
            ] {
                if a != b {
                    assert!(b.is_empty(), "{d}: {what} differs ({a:?} vs {b:?}) and the payload is not blank");
                    blank_payload += 1;
                }
            }
            assert_eq!((h.created_unix, h.created_xuid), v.globals.created, "{d}: created stamp");
            assert_eq!((h.modified_unix, h.modified_xuid), v.globals.modified, "{d}: modified stamp");
            // Compare with the payload's own field, not the file length: some shipped hopper
            // variants store the `mvar` chunk inside a `_cmp` chunk, so the file on disk is
            // shorter than the size both headers record.
            assert_eq!(h.file_size, v.globals.file_size, "{d}: file size");
            checked += 1;
        }
        if checked == 0 {
            eprintln!("film: no .mvar on this machine -- chdr/bitstream agreement not re-checked");
        } else {
            eprintln!("film: chdr struct == bit-packed header on {checked} .mvar file(s) ({blank_payload} blank-payload string(s))");
        }
    }

    /// Every film paired with the real `.mvar` of the same map variant revision. The join key is
    /// (title, map id) narrowed by (object count, spent budget) — i.e. the same revision, not
    /// merely the same name. Returns the parsed film and the `mvar.rs`-parsed variant.
    fn film_mvar_pairs() -> Vec<(std::path::PathBuf, Film, std::path::PathBuf, crate::mvar::Variant)> {
        let films = sample_films();
        // (uid, path) per candidate. The UID is the variant's real identity; the rest of the key
        // is not unique.
        let mut by_key: std::collections::HashMap<(String, u32), Vec<(u64, std::path::PathBuf)>> = Default::default();
        let mut mvar_dirs: Vec<std::path::PathBuf> = crate::mapcat::variant_dirs();
        for f in &films {
            if let Some(par) = f.parent().and_then(|p| p.parent()) {
                mvar_dirs.push(par.join("Map"));
                mvar_dirs.push(par.to_path_buf());
            }
        }
        for d in mvar_dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if !p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) { continue }
                let Ok(raw) = std::fs::read(&p) else { continue };
                let data = crate::mvar::expand_blf_cmp(&raw);
                let chunks = blf_chunks(&data);
                let Some(c) = find_chunk(&chunks, b"chdr") else { continue };
                let h = ContentHeader::parse(c.payload(&data).get(4..).unwrap_or(&[]));
                by_key.entry((h.title.clone(), h.map_id)).or_default().push((h.uid, p));
            }
        }
        let mut out = Vec::new();
        for fp in films {
            let Ok(bytes) = std::fs::read(&fp) else { continue };
            let Ok(f) = parse(&bytes) else { continue };
            if f.header.map_variant.is_empty() { continue }
            let mv = &f.header.map_variant;
            let Some(cands) = by_key.get(&(mv.metadata.title.clone(), mv.map_id)) else { continue };
            let n = mv.present_objects().count();
            let spent = mv.budget_spent;
            // Title + map id + object count + spent budget is NOT a unique key: with every
            // Proton prefix's own variant folder in scope there are enough candidates for two
            // different variants of the same map to agree on all four. The palette CRC and the
            // variant checksum are what actually identify the variant, so they are part of the
            // MATCH rather than something asserted afterwards -- a film with no matching `.mvar`
            // on this machine is skipped, not paired with a lookalike and then failed.
            // **This pairing used to be nondeterministic.** `std::fs::read_dir` returns entries
            // in filesystem order, so the candidate list order varied between runs of the SAME
            // binary — and the same map variant exists several times across the profiles on this
            // machine at different points in its editing history, agreeing on title, map id,
            // object count and spent budget while differing in its Forge labels. The gate
            // therefore passed or failed depending on which lookalike happened to be enumerated
            // first. Sorting makes the choice stable, and the CONTENT UID picks the right one.
            let mut cands: Vec<&(u64, std::path::PathBuf)> = cands.iter().collect();
            cands.sort_by(|a, b| {
                (a.0 != mv.metadata.uid).cmp(&(b.0 != mv.metadata.uid)).then(a.1.cmp(&b.1))
            });
            let Some((mp, v)) = cands.iter().find_map(|(uid, p)| {
                // A different uid is a different variant, not a different save of this one.
                if mv.metadata.uid != 0 && *uid != mv.metadata.uid {
                    return None;
                }
                let v = crate::mvar::parse_variant(p)?;
                // The Forge LABEL POOL is part of the identity, not something to assert
                // afterwards. The same variant is saved repeatedly as it is edited, and two saves
                // agree on title, map id, object count, spent budget, palette CRC and variant
                // checksum while differing in their labels — one measured pair has the film's
                // variant carrying `spartan` and `NoBots` that the `.mvar` on disk does not, and
                // an object whose label index is 7 in one and "none" in the other. Those are
                // genuinely different saves; pairing them and then asserting they agree compares
                // a film against someone else's edit.
                (v.objects.len() == n && v.globals.budget_spent == spent && v.labels == mv.labels)
                    .then_some((p.clone(), v))
            }) else { continue };
            out.push((fp, f, mp, v));
        }
        out
    }

    /// EVERY named field of the 76-byte `s_variant_object` must reproduce what `mvar.rs` decodes
    /// from the bit-packed `.mvar` for the same object. This is what the field map in the struct
    /// doc is based on; it must keep holding.
    #[test]
    fn variant_object_fields_match_the_mvar() {
        let pairs = film_mvar_pairs();
        if pairs.is_empty() {
            eprintln!("film: no film/.mvar pair -- object field map not re-checked");
            return;
        }
        let mut rows = 0usize;
        let (mut weapons, mut teleporters, mut shaped) = (0usize, 0usize, 0usize);
        assert!(
            pairs.len() >= 5,
            "only {} film/.mvar pair(s) formed -- this gate has gone vacuous, which is worse than failing",
            pairs.len()
        );
        for (fp, f, mp, v) in &pairs {
            let mv = &f.header.map_variant;
            let film: Vec<&VariantObject> = mv.present_objects().collect();
            let d = format!(
                "{} <-> {}",
                fp.file_name().unwrap_or_default().to_string_lossy(),
                mp.file_name().unwrap_or_default().to_string_lossy()
            );
            for (i, o) in v.objects.iter().enumerate() {
                let Some(e) = film.get(i) else { break };
                rows += 1;
                let w = format!("{d} object {i}");
                assert_eq!(e.quota_index, o.folder, "{w}: quota index / folder");
                assert_eq!(e.item, o.item, "{w}: item (variant permutation)");
                assert_eq!(e.team, o.team, "{w}: team");
                assert_eq!(e.color as i32, o.color, "{w}: color");
                assert_eq!(e.label_idx, o.label_idx, "{w}: label index");
                assert_eq!(e.placement, o.placement, "{w}: placement flags");
                assert_eq!(e.spawn_seq as i32, o.spawn_seq, "{w}: spawn sequence");
                assert_eq!(e.respawn, o.respawn, "{w}: respawn time");
                assert_eq!(e.cached_type, o.cached_type, "{w}: cached object type");
                assert_eq!(e.boundary_shape, o.boundary_shape, "{w}: boundary shape");
                assert_eq!(e.parent, o.spawn_rel, "{w}: parent / spawn_rel");
                assert_eq!(e.flags as u8 & 0x3, o.flags, "{w}: flags");
                // the +0x48 union
                match e.cached_type {
                    1 => {
                        assert_eq!(e.clips_or_tele_channel, o.weapon_clips, "{w}: weapon clips");
                        weapons += 1;
                    }
                    12..=14 => {
                        assert_eq!(e.clips_or_tele_channel, o.tele_channel, "{w}: teleporter channel");
                        assert_eq!(e.tele_passability, o.tele_passability, "{w}: teleporter passability");
                        teleporters += 1;
                    }
                    _ => {
                        assert_eq!(o.weapon_clips, 0, "{w}: non-weapon with clips");
                        assert_eq!(o.tele_channel, 0, "{w}: non-teleporter with a channel");
                        assert_eq!(e.clips_or_tele_channel, 0, "{w}: union byte set on a plain object");
                    }
                }
                // the boundary quantisation and its float-slot map
                let slots = boundary_float_slots(e.boundary_shape);
                for (k, &slot) in slots.iter().enumerate() {
                    assert_eq!(
                        quantize_boundary(e.boundary[slot]), o.boundary[k],
                        "{w}: boundary dim {k} (float slot {slot}, f={})", e.boundary[slot]
                    );
                    let back = dequantize_boundary(o.boundary[k]);
                    assert!((back - e.boundary[slot]).abs() < 1e-4, "{w}: boundary dim {k} round trip");
                }
                for slot in 0..4 {
                    if !slots.contains(&slot) {
                        assert_eq!(e.boundary[slot], 0.0, "{w}: unused boundary float slot {slot}");
                    }
                }
                if e.boundary_shape != 0 { shaped += 1 }
                // the three bytes that are still unknown are zero on all real data
                assert_eq!(e.raw[0x01], 0, "{w}: +0x01 is no longer zero");
                assert_eq!(e.raw[0x2F], 0, "{w}: +0x2F is no longer zero");
                assert_eq!(e.raw[0x4B], 0, "{w}: +0x4B is no longer zero");
            }
        }
        eprintln!(
            "film: object field map holds on {rows} object(s) from {} pair(s) -- {weapons} weapon(s), {teleporters} teleporter(s), {shaped} shaped",
            pairs.len()
        );
        assert!(rows > 100, "too few objects compared to mean anything ({rows})");
    }

    /// THE END-TO-END GATE for "a film carries the same Forge scene HMS already renders": pair a
    /// film with a real `.mvar` of the same map variant (same title, map id, object count and
    /// spent budget) and require the object tables to agree — quota indices and positions, to
    /// within the `.mvar`'s own position quantisation. SKIPS when no pair exists here.
    #[test]
    fn embedded_map_variant_agrees_with_the_matching_mvar() {
        let pairs = film_mvar_pairs();
        if pairs.is_empty() {
            eprintln!("film: no film/.mvar pair on this machine -- embedded-variant agreement not checked");
            return;
        }
        for (fp, f, mp, v) in &pairs {
            let mv = &f.header.map_variant;
            let d = format!(
                "{} <-> {}",
                fp.file_name().unwrap_or_default().to_string_lossy(),
                mp.file_name().unwrap_or_default().to_string_lossy()
            );
            assert_eq!(v.globals.map_id, mv.map_id, "{d}: map id");
            assert_eq!(v.globals.palette_crc, mv.palette_crc, "{d}: palette crc");
            assert_eq!(v.globals.checksum, mv.checksum, "{d}: variant checksum");
            assert_eq!(v.globals.num_quotas, mv.num_quotas, "{d}: quota count");
            assert_eq!(v.globals.budget_max, mv.budget_max, "{d}: budget max");
            // (The label pool is an IDENTITY criterion above, so it is equal by construction.)
            for k in 0..6 {
                assert!((v.globals.bounds[k] - mv.world_bounds[k]).abs() < 0.01, "{d}: bound {k}");
            }
            // Positions: the .mvar's are quantised over the world box, so allow one step of the
            // coarsest axis plus slack.
            let ext = [
                mv.world_bounds[1] - mv.world_bounds[0],
                mv.world_bounds[3] - mv.world_bounds[2],
                mv.world_bounds[5] - mv.world_bounds[4],
            ];
            let tol = ext.iter().cloned().fold(0.0f32, f32::max) / (1 << 14) as f32 + 0.05;
            let film: Vec<&VariantObject> = mv.present_objects().collect();
            let n = film.len();
            let mut matched = 0usize;
            for (i, o) in v.objects.iter().enumerate() {
                let Some(e) = film.get(i) else { break };
                if (0..3).all(|k| (o.pos[k] - e.position[k]).abs() <= tol) { matched += 1 }
            }
            assert_eq!(matched, n, "{d}: only {matched}/{n} object positions agree (tol {tol})");
            eprintln!("film: embedded variant == {} -- {n}/{n} objects", mp.file_name().unwrap_or_default().to_string_lossy());
        }
    }

    /// The recorded armour indices must all be inside their option lists. The lists come from
    /// `globals\\player_model_customization` (counts in `APPEARANCE_COUNTS`); an index past the end
    /// would mean the ten slot bytes are not where this reader thinks they are.
    #[test]
    fn recorded_armour_indices_are_inside_their_option_lists() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film: no real films -- armour range oracle skipped");
            return;
        }
        let mut seen = 0usize;
        let mut non_default = 0usize;
        let mut maxima = [0u8; APPEARANCE_SLOTS];
        for p in films.iter() {
            let Ok(f) = open_header(p) else { continue };
            for pl in f.header.players.iter().filter(|p| p.present()) {
                let Some(a) = pl.appearance() else { continue };
                seen += 1;
                if !a.is_default() {
                    non_default += 1;
                }
                for (i, v) in a.slots.iter().enumerate() {
                    maxima[i] = maxima[i].max(*v);
                }
                assert!(
                    a.in_range(),
                    "{} has an armour index outside its option list: {:?} against {:?}",
                    pl.gamertag,
                    a.slots,
                    APPEARANCE_COUNTS
                );
            }
        }
        if seen == 0 {
            return;
        }
        eprintln!("film: {seen} recorded players ({non_default} with non-default armour); per-slot maxima {maxima:?} against counts {APPEARANCE_COUNTS:?}");
        assert!(non_default > 0, "no recorded player had any non-default armour, which would mean the slot bytes are the wrong ones");
    }

}

// ---------------------------------------------------------------------------------------------
// #film-live  following a recording that is still being written
// ---------------------------------------------------------------------------------------------

/// Where the "follow the live recording" choice is remembered.
pub const FOLLOW_LIVE_SETTING_FILE: &str = "follow_live.txt";

/// Follow an in-progress recording as it grows? Default ON: opening a `.temp` is a deliberate
/// act, and the only reason to open one is to watch what is happening.
pub fn load_follow_live_setting() -> bool {
    crate::project::settings_dir().map(|d| load_follow_live_from_dir(&d)).unwrap_or(true)
}

pub fn load_follow_live_from_dir(dir: &std::path::Path) -> bool {
    match std::fs::read_to_string(dir.join(FOLLOW_LIVE_SETTING_FILE)) {
        Ok(s) => !matches!(s.trim(), "0" | "off" | "false"),
        Err(_) => true,
    }
}

pub fn save_follow_live_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() {
        save_follow_live_to_dir(&dir, on);
    }
}

pub fn save_follow_live_to_dir(dir: &std::path::Path, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(FOLLOW_LIVE_SETTING_FILE), if on { "1" } else { "0" });
}

/// #film-live  Following a recording that is still being written.
///
/// The whole feature rests on one property of this reader: a PARTIALLY written film decodes
/// cleanly, to fewer ticks, because the stream is framed forward and stopped at the last
/// complete record rather than trusted to a size field written at finalisation. If that ever
/// stops being true, "live" silently becomes "broken", so it is gated here.
#[cfg(test)]
mod live_follow_tests {
    /// Truncating a real film to a fraction of its length must still open, and must report
    /// proportionally fewer ticks -- never an error, and never the full length.
    ///
    /// `#[ignore]`d: it reads a film off the user's drive.
    /// `HMS_FILM=<film> cargo test -p hms-app a_truncated_film -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn a_truncated_film_decodes_to_fewer_ticks() {
        let Ok(path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM");
            return;
        };
        let whole = std::fs::read(&path).expect("film reads");
        let full = super::open(std::path::Path::new(&path)).expect("the whole film opens");
        let full_ticks = crate::film_decode::build_index(
            &full.bytes,
            &full.data,
            &full.header.map_variant,
        )
        .ticks;
        assert!(full_ticks > 100, "need a clip of some length to truncate");

        let dir = std::env::temp_dir().join("hms-live-follow-test");
        let _ = std::fs::create_dir_all(&dir);
        // `.temp` so `is_in_progress` agrees with what this is standing in for.
        let part_path = dir.join("partial.temp");
        let mut last = 0u32;
        // A quarter, a half, three quarters -- INCREASING, which is what makes
        // "more bytes must never mean fewer ticks" a meaningful assertion. (It first read
        // `[4, 2, 4]`, i.e. full/half/full, and the assertion duly caught it.)
        for num in [1u64, 2, 3] {
            let cut = (whole.len() as u64 * num / 4).min(whole.len() as u64) as usize;
            std::fs::write(&part_path, &whole[..cut]).expect("partial writes");
            let part = match super::open(&part_path) {
                Ok(p) => p,
                Err(e) => panic!("a film truncated to {cut} bytes must still open, got {e}"),
            };
            let ticks = crate::film_decode::build_index(
                &part.bytes,
                &part.data,
                &part.header.map_variant,
            )
            .ticks;
            println!("  {cut:>10} of {:>10} bytes -> {ticks:>6} ticks", whole.len());
            assert!(
                ticks <= full_ticks,
                "a truncated film cannot hold more ticks than the whole one"
            );
            assert!(ticks >= last, "more bytes must never mean fewer ticks");
            last = ticks;
            assert!(super::is_in_progress(&part_path), "a .temp must read as in progress");
        }
        assert!(last > 0, "three quarters of a film must decode SOME ticks");
        assert!(last < full_ticks, "and must still be short of the whole thing");
    }

    /// The transplant itself, on two REAL indices of the same film at different lengths: the
    /// playhead and every viewer setting must survive, and the clip must get longer.
    ///
    /// This is the gate that matters for "instant replay": `replace_index` is the only thing
    /// standing between a growing file and the viewer losing their camera every two seconds.
    ///
    /// `#[ignore]`d: it reads a film off the user's drive.
    #[test]
    #[ignore]
    fn growing_the_film_keeps_the_viewers_place() {
        let Ok(path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM");
            return;
        };
        let whole = std::fs::read(&path).expect("film reads");
        let dir = std::env::temp_dir().join("hms-live-follow-test");
        let _ = std::fs::create_dir_all(&dir);
        let part = dir.join("growing.temp");

        let index_of = |cut: usize| {
            std::fs::write(&part, &whole[..cut]).expect("write");
            let f = super::open(&part).expect("opens");
            crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant)
        };

        let half = index_of(whole.len() / 2);
        let mut pb = crate::film_play::Playback::new(half);
        let ticks_half = pb.total_ticks();
        assert!(ticks_half > 60, "half a film should decode more than a second");

        // Park the viewer somewhere specific, with non-default settings.
        pb.tick = (ticks_half / 3) as f32;
        pb.playing = true;
        pb.speed = 0.5;
        pb.show_all_types = true;
        pb.moving_only = false;
        pb.camera = crate::film_play::FilmCamera::Follow(7);
        let want_tick = pb.tick;

        let grown = index_of(whole.len() * 3 / 4);
        let ticks_grown = grown.ticks;
        pb.replace_index(grown);

        assert!(
            pb.total_ticks() >= ticks_half,
            "three quarters of a film cannot be shorter than half of it ({} vs {})",
            pb.total_ticks(),
            ticks_half
        );
        assert_eq!(pb.total_ticks(), ticks_grown, "the new index is the one in use");
        assert_eq!(pb.tick, want_tick, "the viewer stays exactly where they were parked");
        assert!(pb.playing, "and keeps playing");
        assert_eq!(pb.speed, 0.5, "at the speed they chose");
        assert!(pb.show_all_types && !pb.moving_only, "with their filters intact");
        assert_eq!(
            pb.camera,
            crate::film_play::FilmCamera::Follow(7),
            "and on the camera they chose -- this is the whole point of transplanting the index \
             rather than rebuilding Playback"
        );
        println!("  half {ticks_half} ticks -> three quarters {ticks_grown} ticks, playhead {want_tick} kept");

        // And a playhead PAST the end of a shorter index is clamped, not left dangling.
        let shrunk = index_of(whole.len() / 2);
        pb.tick = (ticks_grown - 1) as f32;
        pb.replace_index(shrunk);
        assert!(
            pb.tick <= pb.total_ticks().saturating_sub(1) as f32,
            "the playhead must be clamped into the new range"
        );
    }
}
