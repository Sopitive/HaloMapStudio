//! #gamevariant-bin  The Halo: Reach game variant (`.bin`) file: BLF container + the bit-packed
//! `mpvr` chunk, in BOTH directions.
//!
//! A Reach gametype on disk is `_blf` + `chdr` + `mpvr` + `_eof` (+ an optional `_fsm`). The
//! `mpvr` chunk (version 54) holds a bitstream that the engine reads field by field into the
//! 64,516-byte in-memory `c_game_variant`, the very struct a saved film embeds whole. This module
//! is that codec: ONE description of the bitstream that runs both ways ([`Codec`]), so decode and
//! encode cannot drift apart. The in-memory side is a plain byte image ([`IMAGE_LEN`]) with the
//! engine's own offsets, so a film's embedded variant goes straight in.
//!
//! # Where the facts come from
//! Everything below is read out of `haloreach.dll` (MCC, sha16 `738dd2d24ea3aea1`) with ReverseMe;
//! no third-party gametype code was read. The functions are named in `re-notes`/the memory note
//! `reach-game-variant-bin-format.md`; the key ones are
//!
//! | what | `haloreach.dll` RVA |
//! |---|---|
//! | bit reader `ReadBits` (MSB-first) | `0xDD10C` |
//! | chunk decode (reads 4-bit engine index, then slot 5) | `0x56824` |
//! | base variant decode / encode (vtable `0x9906D0` slot 5 / 4) | `0xE1148` / `0xE0E44` |
//! | megalo variant decode / encode (vtable `0x990618` slot 5 / 4) | `0xDEB40` / `0xDE090` |
//! | megalo script context decode | `0x55044` |
//! | condition record / action record | `0xF6BA8` / `0xF52F4` |
//! | metadata (`s_content_item_metadata`) | `0x103638` |
//!
//! Layout offsets in this file are written as the decompiler shows them: relative to the variant
//! OBJECT (`a1`), which starts 4 bytes into the `c_game_variant` image (the image's first dword is
//! the engine type). [`A1`] adds that 4.

// The decode / classify side is exercised by the tests and by `export_game_variant`'s read-back;
// the app itself only needs `build_bin`.
#![allow(dead_code)]

/// Size of the in-memory `c_game_variant` (what a film embeds).
pub const IMAGE_LEN: usize = 0xFC04;
/// The variant object starts this far into the image (the engine type dword comes first).
pub const A1: usize = 4;
/// The `mpvr` chunk's payload capacity: 20480 bytes of bitstream.
pub const MPVR_CAPACITY: usize = 0x5000;

// ===================================================================================================
// Bit I/O (MSB-first, like the engine's c_bitstream)
// ===================================================================================================

/// Reads bits MSB-first. Reading past the end yields zeros, like the engine; `overrun` records it.
pub struct BitRd<'a> {
    data: &'a [u8],
    pub pos: usize,
    pub overrun: bool,
}

impl<'a> BitRd<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitRd { data, pos: 0, overrun: false }
    }
    pub fn bits(&mut self, n: u32) -> u64 {
        let mut v = 0u64;
        for _ in 0..n {
            let byte = self.pos >> 3;
            let b = if byte < self.data.len() {
                (self.data[byte] >> (7 - (self.pos & 7))) & 1
            } else {
                self.overrun = true;
                0
            };
            v = (v << 1) | b as u64;
            self.pos += 1;
        }
        v
    }
}

/// Writes bits MSB-first.
#[derive(Default)]
pub struct BitWr {
    pub buf: Vec<u8>,
    pub pos: usize,
}

impl BitWr {
    pub fn bits(&mut self, n: u32, v: u64) {
        for i in (0..n).rev() {
            let b = ((v >> i) & 1) as u8;
            if self.pos & 7 == 0 {
                self.buf.push(0);
            }
            let last = self.buf.len() - 1;
            self.buf[last] |= b << (7 - (self.pos & 7));
            self.pos += 1;
        }
    }
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

// ===================================================================================================
// The two-way codec
// ===================================================================================================

/// One codec for both directions. In decode mode every `fld`-style call READS bits and STORES the
/// value into the image; in encode mode it LOADS the value from the image and WRITES bits. All the
/// branching in the layout code looks at the value `fld` returns, which is the stored value in
/// both modes, so the two directions walk the identical path.
pub struct Codec<'a> {
    pub dec: bool,
    pub rd: BitRd<'a>,
    pub wr: BitWr,
    pub img: Vec<u8>,
    /// First problem seen (out-of-range count, ...). Decoding continues so a trace is complete.
    pub err: Option<String>,
    /// When set, every bit-field is logged as (width, value): the oracle comparison against the
    /// engine's own reads.
    pub trace: Option<Vec<(u32, u64, u32)>>,
    /// Which image bytes the decoder wrote (decode only): lets a comparison against the engine's
    /// image ignore the engine's own default fills.
    pub written: Vec<bool>,
    /// Named bit positions of the few fields whose bits are NOT a function of the image
    /// (see [`classify_roundtrip`]).
    pub marks: Vec<(&'static str, usize)>,
    /// A value outside its field's legal domain was met (the engine's own decoder then overruns
    /// into a neighbouring field). Decoding goes on; the flag lets tests tell such input apart.
    pub out_of_domain: bool,
}

impl<'a> Codec<'a> {
    pub fn decoder(data: &'a [u8], image_len: usize) -> Self {
        Codec { dec: true, rd: BitRd::new(data), wr: BitWr::default(), img: vec![0; image_len], err: None, trace: None, written: vec![false; image_len], marks: Vec::new(), out_of_domain: false }
    }
    pub fn encoder(img: Vec<u8>) -> Codec<'static> {
        Codec { dec: false, rd: BitRd::new(&[]), wr: BitWr::default(), img, err: None, trace: None, written: Vec::new(), marks: Vec::new(), out_of_domain: false }
    }

    pub fn fail(&mut self, msg: impl Into<String>) {
        if self.err.is_none() {
            self.err = Some(msg.into());
        }
    }

    fn load(&mut self, off: usize, sz: usize) -> u64 {
        if off + sz > self.img.len() {
            self.fail(format!("image read out of range at {off:#x}+{sz}"));
            return 0;
        }
        let mut v = 0u64;
        for i in (0..sz).rev() {
            v = (v << 8) | self.img[off + i] as u64;
        }
        v
    }
    fn store(&mut self, off: usize, sz: usize, v: u64) {
        if off + sz > self.img.len() {
            self.fail(format!("image write out of range at {off:#x}+{sz}"));
            return;
        }
        for i in 0..sz {
            self.img[off + i] = (v >> (8 * i)) as u8;
            if self.dec { self.written[off + i] = true; }
        }
    }

    /// Raw bit move with no image side: decode reads, encode writes `v`. Returns the value.
    #[track_caller]
    pub fn io(&mut self, bits: u32, v: u64) -> u64 {
        if self.dec {
            let r = self.rd.bits(bits);
            let line = std::panic::Location::caller().line();
            if let Some(t) = self.trace.as_mut() {
                t.push((bits, r, line));
            }
            r
        } else {
            let v = v & mask(bits);
            self.wr.bits(bits, v);
            let line = std::panic::Location::caller().line();
            if let Some(t) = self.trace.as_mut() {
                t.push((bits, v, line));
            }
            v
        }
    }

    /// An unsigned field: `bits` on the wire, `sz` bytes in the image, stored value = wire + `bias`.
    /// Returns the stored value, zero-extended to the field size.
    #[track_caller]
    pub fn fld(&mut self, off: usize, sz: usize, bits: u32, bias: i64) -> u64 {
        if self.dec {
            let v = self.io(bits, 0);
            let s = (v as i64).wrapping_add(bias) as u64 & mask(sz as u32 * 8);
            self.store(off, sz, s);
            s
        } else {
            let s = self.load(off, sz);
            // (a wire value wider than its field is stored truncated; the engine's writer writes
            // the stored field zero-extended -- proven against its own encoder)
            let ext = s;
            self.io(bits, (ext as i64).wrapping_sub(bias) as u64);
            s
        }
    }

    /// Like [`fld`] but the decoded value is sign-extended from `bits` to the field size.
    #[track_caller]
    pub fn fld_s(&mut self, off: usize, sz: usize, bits: u32) -> u64 {
        if self.dec {
            let v = self.io(bits, 0);
            let sh = 64 - bits;
            let sx = (((v << sh) as i64) >> sh) as u64 & mask(sz as u32 * 8);
            self.store(off, sz, sx);
            sx
        } else {
            let s = self.load(off, sz);
            self.io(bits, s);
            s
        }
    }

    /// A one-bit flag stored as bit `bit` of a `sz`-byte word (other bits kept).
    #[track_caller]
    pub fn flag(&mut self, off: usize, sz: usize, bit: u32) -> bool {
        if self.dec {
            let v = self.io(1, 0) != 0;
            let w = self.load(off, sz);
            let w = if v { w | (1 << bit) } else { w & !(1 << bit) };
            self.store(off, sz, w);
            v
        } else {
            let v = (self.load(off, sz) >> bit) & 1 != 0;
            self.io(1, v as u64);
            v
        }
    }

    /// A field behind a "none" flag: flag 1 = absent (stored as all ones), flag 0 = `bits` follow.
    #[track_caller]
    pub fn opt(&mut self, off: usize, sz: usize, bits: u32) -> u64 {
        let none = mask(sz as u32 * 8);
        if self.dec {
            let f = self.io(1, 0);
            let v = if f == 0 { self.io(bits, 0) } else { none };
            self.store(off, sz, v);
            v
        } else {
            let s = self.load(off, sz);
            if s == none {
                self.io(1, 1);
            } else {
                self.io(1, 0);
                self.io(bits, s);
            }
            s
        }
    }

    /// Raw bytes (`n` of them) on the wire, 8 bits each.
    #[track_caller]
    pub fn bytes(&mut self, off: usize, n: usize) {
        for i in 0..n {
            self.fld(off + i, 1, 8, 0);
        }
    }

    /// A bit that is read and thrown away; written as `val` (the engine writes a constant).
    #[track_caller]
    pub fn discard(&mut self, val: u64) {
        self.io(1, val);
    }

    /// A quantised float (the engine's `0xDC770`): `bits` wide, range `[lo, hi]`, with the two
    /// end values exact and the rest `(hi-lo)/(2^bits-2)` apart, cell centres.
    #[track_caller]
    pub fn qfloat(&mut self, off: usize, lo: f32, hi: f32, bits: u32) {
        if self.dec {
            let v = self.io(bits, 0);
            let f = dequant(v, lo, hi, bits);
            self.store(off, 4, f.to_bits() as u64);
        } else {
            let f = f32::from_bits(self.load(off, 4) as u32);
            let v = quant(f, lo, hi, bits);
            self.io(bits, v);
        }
    }
}

/// `0xDC770` with `a6 = 0`, `a7 = 1` (the way every call in the variant uses it), single precision.
pub fn dequant(v: u64, lo: f32, hi: f32, bits: u32) -> f32 {
    let n = 1u64 << bits;
    if v == 0 {
        lo
    } else if v != n - 1 {
        let step = (hi - lo) / (n - 2) as f32;
        let a = (v - 1) as f32 * step;
        let b = step * 0.5 + lo;
        a + b
    } else {
        hi
    }
}

/// The engine's quantiser (`0xDC234`, single precision): `lo` -> 0, `hi` -> 2^bits-1, anything
/// else `trunc((f-lo)/step)+1` clamped to `[1, 2^bits-2]`.
pub fn quant(f: f32, lo: f32, hi: f32, bits: u32) -> u64 {
    let n = 1i64 << bits;
    if f == lo {
        return 0;
    }
    if f == hi {
        return (n - 1) as u64;
    }
    let step = (hi - lo) / (n - 2) as f32;
    let v4 = ((f - lo) / step) as i64 + 1;
    v4.clamp(1, n - 2) as u64
}

// ===================================================================================================
// Megalo parameter encodings. Each takes the ABSOLUTE image offset of its record.
// ===================================================================================================

/// `0x146F88`: the general 3-byte value reference (6-bit type; [0]=index/value, [1]=aux, [2]=type).
pub fn var6(c: &mut Codec, o: usize) {
    if c.dec { c.store(o, 2, 0); c.store(o + 2, 1, 0); }
    let t = c.fld(o + 2, 1, 6, 0);
    match t {
        0 => { c.fld_s(o, 2, 16); }
        1 | 2 => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 3, 0); }
        3 => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 3, 0); }
        4 | 5 | 0x2C => { c.fld(o + 1, 1, 4, 0); }
        6 | 8 | 9 | 0xA => { c.fld(o, 1, 5, 0); }
        7 => { c.fld(o, 1, 5, -1); }
        0xB => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 2, 0); }
        0xC => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 2, 0); }
        _ => {}
    }
}

/// `0x1F4AE0`: 2-bit-typed reference.
pub fn var2(c: &mut Codec, o: usize) {
    let t = c.fld(o + 2, 1, 2, 0);
    match t {
        0 => { c.fld(o, 1, 5, 0); }
        1 | 2 => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 2, 0); }
        _ => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 2, 0); }
    }
}

/// `0x1F50A8`: 3-bit-typed reference, flavour A.
pub fn var3a(c: &mut Codec, o: usize) {
    let t = c.fld(o + 2, 1, 3, 0);
    match t {
        0 => { c.fld(o, 1, 5, -1); }
        1 => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 2, 0); }
        2 => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 1, 0); }
        3 => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 2, 0); }
        4 | 5 => { c.fld(o, 1, 5, 0); }
        _ => {}
    }
}

/// `0x1F584C`: 3-bit-typed reference, flavour B (the commonest "player/object/team" operand).
pub fn var3b(c: &mut Codec, o: usize) {
    let t = c.fld(o + 2, 1, 3, 0);
    match t {
        0 | 4 => { c.fld(o, 1, 5, 0); }
        1 | 2 | 5 | 6 => { c.fld(o, 1, 5, 0); c.fld(o + 1, 1, 2, 0); }
        3 => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 3, 0); }
        _ => { c.fld(o, 1, 5, -1); c.fld(o + 1, 1, 2, 0); }
    }
}

/// `0x1F6D90`: reference whose TYPE is byte 1 (3 bits), the payload byte 0, the scope byte 2.
pub fn var3c(c: &mut Codec, o: usize) {
    let t = c.fld(o + 1, 1, 3, 0);
    match t {
        0 => { c.fld(o + 2, 1, 3, 0); }
        1 | 3 => { c.fld(o, 1, 5, 0); c.fld(o + 2, 1, 2, 0); }
        2 => { c.fld(o, 1, 5, -1); c.fld(o + 2, 1, 2, 0); }
        _ => {}
    }
}

/// `0x1F5160`: a dword, "none" (-1) or an 11-bit value.
pub fn i11(c: &mut Codec, o: usize) { c.opt(o, 4, 11); }
/// `0xF6154`: a dword, "none" (-1) or a 2-bit value.
pub fn i2(c: &mut Codec, o: usize) { c.opt(o, 4, 2); }

/// `0x1F5A4C`.
pub fn t5a4c(c: &mut Codec, o: usize) {
    let t = c.fld(o, 1, 3, 0);
    if t == 4 {
        var2(c, o + 1);
        var6(c, o + 4);
    }
}

/// `0x1F5D2C`: an operand tagged by byte 3.
pub fn u5d2c(c: &mut Codec, o: usize) {
    let t = c.fld(o + 3, 1, 3, 0);
    match t {
        0 => var6(c, o),
        1 => var2(c, o),
        2 => var3b(c, o),
        3 => var3a(c, o),
        4 => var3c(c, o),
        _ => {}
    }
}

/// `0xF5F14`.
pub fn p5f14(c: &mut Codec, o: usize) {
    let t = c.fld(o + 3, 1, 2, 0);
    match t {
        0 => var3a(c, o),
        1 => var2(c, o),
        _ => {}
    }
}

/// `0xF61F8`.
pub fn p61f8(c: &mut Codec, o: usize) {
    var3b(c, o);
    c.fld(o + 3, 1, 4, 0);
    var6(c, o + 4);
}

/// `0x1F7268`.
pub fn p7268(c: &mut Codec, o: usize) {
    if c.dec { c.store(o, 4, 0); c.store(o + 4, 2, 0); c.store(o + 6, 1, 0); }
    let t = c.fld(o, 1, 2, 0);
    match t {
        1 => { var6(c, o + 1); var6(c, o + 4); }
        2 => var3c(c, o + 1),
        _ => {}
    }
}

/// `0x1F6128`: the 10-byte "shape/filter" record: a 7-bit id, then up to two typed operands.
pub fn sh6128(c: &mut Codec, o: usize) {
    if c.dec {
        c.store(o + 8, 2, 0xFFFF);
        c.store(o + 3, 1, 0xFF);
        c.store(o + 7, 1, 0xFF);
    }
    c.fld(o + 8, 2, 7, -1);
    // the entry count (2 bits) is not stored: the engine derives it from the filled slots
    let n = if c.dec {
        c.io(2, 0) as usize
    } else {
        // the engine counts the leading slots whose type byte is not "none" (0xFF), at most two
        let mut n = 0usize;
        while n < 2 && c.load(o + 4 * n + 3, 1) != 0xFF {
            n += 1;
        }
        c.io(2, n as u64) as usize
    };
    if n > 2 {
        c.out_of_domain = true;
    }
    for i in 0..n.min(3) {
        let e = o + 4 * i;
        c.fld(e + 3, 1, 3, -1);
        let t = c.load(e + 3, 1);
        match t {
            0 => var2(c, e),
            1 => var3a(c, e),
            2 => var3b(c, e),
            3 => var6(c, e),
            4 => var3c(c, e),
            _ => {}
        }
    }
}

/// `0x1F700C`: four words (9, 10, 10, 11 bits).
pub fn p700c(c: &mut Codec, o: usize) {
    c.fld(o, 2, 9, 0);
    c.fld(o + 2, 2, 10, 0);
    c.fld(o + 4, 2, 10, 0);
    c.fld(o + 6, 2, 11, 0);
}

// ===================================================================================================
// Megalo condition (16 bytes) and action (20 bytes) records
// ===================================================================================================

/// `0xF6BA8`: one condition record at `c0`.
pub fn condition(c: &mut Codec, c0: usize) {
    let t = c.fld(c0 + 0xC, 1, 5, 0);
    if t == 0 {
        return;
    }
    c.fld(c0 + 0xD, 1, 1, 0);
    c.fld(c0 + 0xE, 1, 9, 0);
    c.fld(c0 + 0xF, 1, 10, 0);
    match t {
        1 => { u5d2c(c, c0); u5d2c(c, c0 + 4); c.fld(c0 + 8, 1, 3, 0); }
        2 => { var3b(c, c0); var3b(c, c0 + 3); }
        3 => { var2(c, c0); c.fld(c0 + 3, 1, 5, 0); }
        4 => { var3a(c, c0); var3a(c, c0 + 3); c.fld(c0 + 6, 1, 2, 0); }
        5 => var3c(c, c0),
        6 => { var3b(c, c0); i11(c, c0 + 4); }
        7 => var3a(c, c0),
        8 | 0xD => var3b(c, c0),
        9 | 0xC | 0xE | 0xF | 0x10 => var2(c, c0),
        0xA => { var2(c, c0); var2(c, c0 + 3); }
        0xB => { var3b(c, c0); c.opt(c0 + 4, 4, 4); }
        _ => {}
    }
}

/// `0xF52F4`: one action record at `a`.
pub fn action(c: &mut Codec, a: usize) {
    let op = c.fld(a + 0x10, 1, 7, 0);
    match op {
        0 => {}
        0x01 => { p5f14(c, a + 4); c.fld(a + 3, 1, 4, 0); var6(c, a); }
        0x02 => {
            i11(c, a); var3b(c, a + 4); var3b(c, a + 7);
            c.opt(a + 0xA, 1, 4); c.fld(a + 0xB, 1, 3, 0); c.bytes(a + 0xC, 3);
            c.fld(a + 0xF, 1, 8, -1);
        }
        0x03 => var3b(c, a),
        0x04 | 0x0C | 0x0D | 0x16 => { var3b(c, a); t5a4c(c, a + 3); }
        0x05 => {
            var3b(c, a);
            let t = c.fld(a + 3, 1, 5, -1);
            if t == 0xB { var6(c, a + 4); }
        }
        0x06 => { var3b(c, a); c.fld(a + 3, 1, 2, 0); }
        0x07 | 0x62 => { var3b(c, a); c.opt(a + 3, 1, 2); }
        0x08 => { var3b(c, a); var6(c, a + 3); var6(c, a + 6); }
        0x09 => { u5d2c(c, a); u5d2c(c, a + 4); c.fld(a + 8, 1, 4, 0); }
        0x0A => {
            var3b(c, a);
            let t = c.fld(a + 3, 1, 2, 0);
            match t {
                1 => var6(c, a + 4),
                2 => { var6(c, a + 4); var6(c, a + 0xA); var6(c, a + 0xD); }
                3 => { var6(c, a + 4); var6(c, a + 7); var6(c, a + 0xA); var6(c, a + 0xD); }
                _ => {}
            }
        }
        0x0B => { var2(c, a); c.fld(a + 4, 4, 4, 0); }
        0x0E => { var3b(c, a); c.fld(a + 3, 1, 8, 0); }
        0x0F => { var3b(c, a); t5a4c(c, a + 3); c.opt(a + 0xA, 1, 2); }
        0x10 => { p5f14(c, a + 0xB); c.fld(a + 0xA, 1, 7, -1); sh6128(c, a); }
        0x11 => { var3c(c, a); c.fld(a + 3, 1, 5, 0); }
        0x12 => sh6128(c, a),
        0x13 => { var3b(c, a); var2(c, a + 3); }
        0x14 => { c.fld(a, 4, 9, 0); }
        0x15 => {}
        0x17 => { var3b(c, a); c.flag(a + 3, 1, 0); }
        0x18 => { var3b(c, a); var6(c, a + 3); }
        0x19 => { var6(c, a); var6(c, a + 3); }
        0x1B | 0x1C => { var3b(c, a); var6(c, a + 3); }
        0x1D => { var2(c, a); var2(c, a + 3); }
        0x1E | 0x1F => { var2(c, a); var6(c, a + 3); }
        0x20 => { c.flag(a, 1, 0); }
        0x21 => { var3b(c, a); var3b(c, a + 3); c.bytes(a + 6, 3); c.flag(a + 9, 1, 0); }
        0x22 => var3b(c, a),
        0x23 | 0x25 => { var2(c, a); var6(c, a + 3); }
        0x24 => { var3a(c, a); var6(c, a + 3); }
        0x26 => { var2(c, a); c.fld(a + 3, 1, 4, 0); var6(c, a + 4); }
        0x27 => { var2(c, a); var6(c, a + 3); c.fld(a + 6, 1, 5, 0); }
        0x28 | 0x29 | 0x2A => { var2(c, a); var3b(c, a + 3); }
        0x2B => var3c(c, a),
        0x2C => { var3b(c, a); c.fld(a + 3, 1, 2, 0); }
        0x2D => var3b(c, a),
        0x2E | 0x2F => { i2(c, a); sh6128(c, a + 4); }
        0x30 => { i2(c, a); p7268(c, a + 4); }
        0x31 => { i2(c, a); c.opt(a + 4, 4, 6); }
        0x32 => { i2(c, a); var2(c, a + 4); c.flag(a + 7, 1, 0); }
        0x33 => { c.fld(a + 4, 1, 7, -1); c.flag(a + 5, 1, 0); p5f14(c, a); }
        0x34 | 0x36 | 0x37 => { var3b(c, a); var6(c, a + 3); }
        0x35 => { var3b(c, a); sh6128(c, a + 4); }
        0x38 | 0x39 => { var2(c, a); sh6128(c, a + 4); }
        0x3A => { var2(c, a); c.opt(a + 4, 4, 7); }
        0x3B => { var3a(c, a); c.flag(a + 3, 1, 0); }
        0x3C => { var3a(c, a); var3b(c, a + 3); }
        0x3D => { var2(c, a); var3b(c, a + 3); }
        0x3E | 0x3F => { var2(c, a); var6(c, a + 3); }
        0x42 => { var3b(c, a); var3b(c, a + 3); var6(c, a + 6); }
        0x40 | 0x41 | 0x43 | 0x44 => p61f8(c, a),
        0x45 => { var2(c, a); c.fld(a + 3, 1, 4, 0); }
        0x46 | 0x47 | 0x48 | 0x49 => { var3b(c, a); var6(c, a + 3); }
        0x4A => { var2(c, a); c.fld(a + 3, 1, 1, 0); c.fld(a + 4, 1, 4, 0); var6(c, a + 5); }
        0x4B => { c.fld(a, 4, 10, -1); p5f14(c, a + 4); p5f14(c, a + 8); }
        0x4C => { c.fld(a, 4, 10, -1); p5f14(c, a + 4); p5f14(c, a + 8); var6(c, a + 0xC); }
        0x4D => { p5f14(c, a); c.fld(a + 4, 1, 3, -1); }
        0x4E => { var3b(c, a); c.fld(a + 4, 4, 8, -1); var6(c, a + 8); }
        0x4F => { var3b(c, a); var6(c, a + 3); var6(c, a + 6); var6(c, a + 9); var6(c, a + 0xC); }
        0x50 => { var3b(c, a); var6(c, a + 3); }
        0x51 => { var6(c, a); sh6128(c, a + 4); }
        0x52 | 0x55 => { var3b(c, a); var6(c, a + 3); }
        0x53 => { var2(c, a); c.flag(a + 3, 1, 0); var3b(c, a + 4); }
        0x54 | 0x56 => { var2(c, a); var3b(c, a + 3); }
        0x57 => { var3b(c, a + 4); var3b(c, a + 7); i11(c, a); var6(c, a + 0xA); var3b(c, a + 0xD); }
        0x58 => var6(c, a),
        0x59 => { var2(c, a); var3b(c, a + 3); }
        0x5A | 0x67 => { var2(c, a); c.flag(a + 3, 1, 0); }
        0x5B => { var3b(c, a); var3b(c, a + 3); c.flag(a + 6, 1, 0); }
        0x5C => { var3b(c, a); var3b(c, a + 3); c.bytes(a + 6, 3); }
        0x5D => { var3b(c, a); i11(c, a + 4); c.fld(a + 8, 1, 2, 0); }
        0x5E => { var3b(c, a); c.flag(a + 3, 1, 0); c.flag(a + 4, 1, 0); }
        0x5F => { var6(c, a); var6(c, a + 3); }
        0x60 => { var3b(c, a); var3b(c, a + 3); c.opt(a + 6, 1, 4); }
        0x61 => { var2(c, a); var6(c, a + 3); }
        0x63 => p700c(c, a),
        0x64 => { c.fld(a, 4, 8, -1); }
        0x65 => { var2(c, a); c.fld(a + 3, 1, 5, 0); var6(c, a + 4); }
        0x66 | 0x6A => { if op == 0x66 { var3a(c, a); } else { var3b(c, a); } c.flag(a + 3, 1, 0); }
        0x68 => { i11(c, a); var2(c, a + 4); }
        0x69 => { i11(c, a); var3a(c, a + 4); }
        _ => {}
    }
}

// ===================================================================================================
// Strings and blobs
// ===================================================================================================

/// zlib level-9 deflate as the engine writes it (`0x1800A6F4C`): `[BE u32 raw length][zlib stream]`,
/// only if the whole thing fits in `raw.len()` bytes. See `zdeflate.rs` for the exact port.
pub fn engine_compress(raw: &[u8]) -> Option<Vec<u8>> {
    if raw.len() < 4 {
        return None;
    }
    let z = crate::zdeflate::compress9(raw, raw.len() - 4)?;
    let mut out = Vec::with_capacity(z.len() + 4);
    out.extend_from_slice(&(raw.len() as u32).to_be_bytes());
    out.extend_from_slice(&z);
    Some(out)
}

/// The inverse: `[BE u32 raw length][zlib stream]` -> raw bytes of exactly that length.
pub fn engine_decompress(blob: &[u8]) -> Option<Vec<u8>> {
    if blob.len() < 4 {
        return None;
    }
    let n = u32::from_be_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    let out = miniz_oxide::inflate::decompress_to_vec_zlib(&blob[4..]).ok()?;
    // the engine's `uncompress` only requires that the data fit the stated size
    if out.len() > n { return None; }
    Some(out)
}

/// A length-prefixed string blob that may be zlib-compressed: `lenbits` raw length, a compressed
/// flag, then either the raw bytes or `lenbits` compressed length + the compressed bytes.
/// `data` is the image offset of the bytes, `len_off` that of the dword length, `cap` the room.
pub fn blob(c: &mut Codec, data: usize, len_off: usize, lenbits: u32, cap: usize) {
    if c.dec {
        let l = c.fld(len_off, 4, lenbits, 0) as usize;
        let compressed = c.io(1, 0) != 0;
        if l > cap {
            c.fail(format!("string blob of {l} bytes exceeds its {cap}-byte buffer"));
            return;
        }
        if !compressed {
            c.bytes(data, l);
        } else {
            let lc = c.io(lenbits, 0) as usize;
            let mut tmp = vec![0u8; lc];
            for b in tmp.iter_mut() {
                *b = c.io(8, 0) as u8;
            }
            match engine_decompress(&tmp) {
                Some(raw) if raw.len() <= l => {
                    for (k, b) in raw.iter().enumerate() {
                        c.store(data + k, 1, *b as u64);
                    }
                }
                _ => c.fail("string blob: zlib stream does not inflate into the stated length"),
            }
        }
    } else {
        let l = c.load(len_off, 4) as usize;
        c.io(lenbits, l as u64);
        if l > cap || data + l > c.img.len() {
            c.fail("string blob longer than its buffer");
            return;
        }
        let raw = c.img[data..data + l].to_vec();
        let z = if l >= 0x80 { engine_compress(&raw) } else { None };
        match z {
            Some(z) => {
                c.io(1, 1);
                c.io(lenbits, z.len() as u64);
                for b in z { c.io(8, b as u64); }
            }
            None => {
                c.io(1, 0);
                for b in raw { c.io(8, b as u64); }
            }
        }
    }
}

/// `n` strings x 12 languages of offsets, each behind a "present" flag (absent = 0xFFFF), laid out
/// `[language][string]` with `lang_stride` bytes between languages.
fn offset_grid(c: &mut Codec, base: usize, n: usize, lang_stride: usize, bits: u32) {
    if c.dec {
        for l in 0..12 {
            for i in 0..n { c.store(base + l * lang_stride + 2 * i, 2, 0xFFFF); }
        }
    }
    for i in 0..n {
        for l in 0..12 {
            let o = base + l * lang_stride + 2 * i;
            if c.dec {
                if c.io(1, 0) != 0 {
                    let v = c.io(bits, 0);
                    c.store(o, 2, v);
                }
            } else {
                let v = c.load(o, 2);
                if v == 0xFFFF {
                    c.io(1, 0);
                } else {
                    c.io(1, 1);
                    c.io(bits, v);
                }
            }
        }
    }
}

/// The script's string table (`a1+0x5DBC`): up to 112 strings x 12 languages, one shared blob.
pub fn script_strings(c: &mut Codec) {
    let a = A1;
    let n = c.fld(a + 0x5DBC, 4, 7, 0) as usize;
    if n > 112 {
        c.fail(format!("string table claims {n} strings (max 112)"));
        return;
    }
    offset_grid(c, a + 0x738, n, 0xE0, 15);
    if n > 0 {
        blob(c, a + 0x11B8, a + 0x5DB8, 15, 0x4C00);
    }
}

/// A "one string, 12 languages" table (`0x1A0` bytes): count (1 bit), offsets, blob.
/// `off_bits`/`len_bits` are 9 for the name/category tables.
pub fn one_string(c: &mut Codec, t: usize, cnt_off: usize, off_bits: u32, data_off: usize, len_off: usize, len_bits: u32, cap: usize) {
    let n = c.fld(cnt_off, 4, 1, 0) as usize;
    offset_grid(c, t, n, 2, off_bits);
    if n > 0 {
        blob(c, data_off, len_off, len_bits, cap);
    }
    let _ = t;
}

// ===================================================================================================
// Traits (the 0x2C-byte "ScriptActionList" the dll names it by mistake: it is s_player_traits)
// ===================================================================================================

pub fn traits(c: &mut Codec, o: usize) {
    // header: ten small fields
    for (i, w) in [4u32, 3, 4, 3, 4, 4, 2, 3, 2, 2].iter().enumerate() {
        c.fld(o + i, 1, *w, 0);
    }
    if c.dec { c.store(o + 0xA, 2, 0); }
    // record @ +0xC
    let r = o + 0xC;
    c.fld(r, 1, 4, 0);
    c.fld(r + 1, 1, 4, 0);
    c.fld(r + 9, 1, 8, 0);
    c.fld(r + 0xA, 1, 8, 0);
    c.fld(r + 4, 1, 4, 0);
    c.fld(r + 5, 1, 2, 0);
    c.fld(r + 2, 1, 2, 0);
    c.fld(r + 3, 1, 2, 0);
    c.fld(r + 6, 1, 2, 0);
    c.fld(r + 7, 1, 2, 0);
    c.fld(r + 8, 1, 2, 0);
    c.fld(r + 0xB, 1, 8, 0);
    // layout @ +0x18
    let l = o + 0x18;
    c.fld(l, 1, 5, 0);
    c.fld(l + 1, 1, 4, 0);
    c.fld(l + 2, 1, 4, 0);
    c.fld(l + 3, 1, 2, 0);
    // optional 9-bit field behind a "none" flag -> dword at +0x1C ((v+1) <= 0x191 ? v : -1)
    if c.dec {
        let v = if c.io(1, 0) == 0 { 0xFFFF_FFFFu64 } else { c.io(9, 0) };
        c.store(l + 4, 4, v);
    } else {
        let v = c.load(l + 4, 4);
        if v == 0xFFFF_FFFF {
            c.io(1, 0);
        } else {
            c.io(1, 1);
            c.io(9, v);
        }
    }
    // bit lengths @ +0x20
    let b = o + 0x20;
    for (i, w) in [3u32, 2, 2, 3, 4].iter().enumerate() {
        c.fld(b + i, 1, *w, 0);
    }
    // type codes @ +0x28
    let t = o + 0x28;
    c.fld(t, 1, 3, 0);
    c.fld(t + 1, 1, 3, 0);
    c.fld(t + 2, 1, 2, 0);
    if c.dec { c.store(t + 3, 1, 0); }
}

/// `0xE0550`: the 0x3C-byte base traits (a handful of flags + small fields) followed by a list.
pub fn base_traits(c: &mut Codec, o: usize) {
    for b in 0..4 {
        c.flag(o, 1, b);
    }
    c.fld(o + 1, 1, 6, 0);
    c.fld(o + 2, 1, 7, 0);
    c.fld(o + 3, 1, 8, 0);
    c.fld(o + 4, 1, 8, 0);
    c.fld(o + 5, 1, 8, 0);
    c.fld(o + 6, 1, 4, 0);
    c.fld(o + 7, 1, 4, 0);
    c.fld(o + 8, 1, 6, 0);
    traits(c, o + 0xC);
}

// ===================================================================================================
// Metadata (s_content_item_metadata, 0x2B0 bytes at a1+8) -- also what the `chdr` chunk carries
// ===================================================================================================

/// A 16-bit-character string, NUL-terminated on the wire, `cap` characters in the image.
fn wstring(c: &mut Codec, o: usize, cap: usize) {
    if c.dec {
        for i in 0..cap {
            let v = c.io(16, 0);
            c.store(o + 2 * i, 2, v);
            if v == 0 {
                return;
            }
        }
        c.store(o + 2 * (cap - 1), 2, 0);
    } else {
        for i in 0..cap {
            let v = c.load(o + 2 * i, 2);
            c.io(16, v);
            if v == 0 {
                return;
            }
        }
    }
}

/// An 8-bit-character string, `cap` bytes.
fn cstring(c: &mut Codec, o: usize, cap: usize) {
    if c.dec {
        for i in 0..cap {
            let v = c.io(8, 0);
            c.store(o + i, 1, v);
            if v == 0 {
                return;
            }
        }
        c.store(o + cap - 1, 1, 0);
    } else {
        for i in 0..cap {
            let v = c.load(o + i, 1);
            c.io(8, v);
            if v == 0 {
                return;
            }
        }
    }
}

fn u64f(c: &mut Codec, o: usize) {
    if c.dec {
        let v = c.io(64, 0);
        c.store(o, 8, v);
    } else {
        let v = c.load(o, 8);
        c.io(64, v);
    }
}

/// `0x103968`: a creator stamp (timestamp, xuid, 16-char gamertag, "online" bit), 0x24 bytes.
fn stamp(c: &mut Codec, o: usize) {
    u64f(c, o);
    u64f(c, o + 8);
    cstring(c, o + 0x10, 16);
    c.fld(o + 0x20, 1, 1, 0);
}

/// `0x103638`. `m` is the image offset of the metadata (a1 + 8).
pub fn metadata(c: &mut Codec, m: usize) {
    let ty = c.fld(m, 1, 4, -1);
    c.fld(m + 4, 4, 32, 0);
    u64f(c, m + 8);
    u64f(c, m + 0x10);
    u64f(c, m + 0x18);
    u64f(c, m + 0x20);
    let q28 = c.fld(m + 0x28, 1, 3, -1);
    c.fld(m + 0x29, 1, 3, 0);
    c.fld(m + 0x2A, 1, 3, 0);
    c.fld(m + 0x2C, 4, 32, 0);
    c.fld_s(m + 0x30, 1, 8);
    stamp(c, m + 0x38);
    stamp(c, m + 0x5C);
    wstring(c, m + 0x80, 128);
    wstring(c, m + 0x180, 128);
    if ty == 3 || ty == 4 {
        c.fld(m + 0x280, 4, 32, 0);
    } else if ty == 6 {
        c.fld_s(m + 0x280, 1, 8);
    }
    if q28 == 2 {
        c.fld(m + 0x290, 2, 16, 0);
    }
    let q29 = c.load(m + 0x29, 1);
    match q29 {
        1 => {
            c.fld(m + 0x2A0, 1, 8, 0);
            c.fld(m + 0x2A1, 1, 2, 0);
            c.fld(m + 0x2A2, 1, 2, 0);
            c.fld(m + 0x2A3, 1, 8, 0);
        }
        2 => { c.fld(m + 0x2A0, 1, 2, 0); }
        _ => return,
    }
    c.fld(m + 0x2A4, 4, 32, 0);
}

// ===================================================================================================
// The base variant (all engine types) and the Megalo extension
// ===================================================================================================

/// `0xE1148`: everything common to every engine type. `a` is the variant object offset (A1).
pub fn base_variant(c: &mut Codec) {
    let a = A1;
    metadata(c, a + 8);
    c.flag(a + 0x732, 2, 0);
    for b in 0..4 {
        c.flag(a + 0x2B8, 1, b);
    }
    c.fld(a + 0x2BA, 1, 8, 0);
    c.fld(a + 0x2BB, 1, 5, 0);
    c.fld(a + 0x2B9, 1, 4, 0);
    c.fld(a + 0x2BC, 2, 7, -1);
    c.fld(a + 0x2BE, 2, 5, 0);
    base_traits(c, a + 0x2C0);
    c.discard(0);
    c.fld(a + 0x2FA, 2, 2, 0);
    for b in 0..5 {
        c.flag(a + 0x2F8, 2, b);
    }
    c.fld(a + 0x2FC, 1, 6, 0);
    traits(c, a + 0x300);
    c.fld_s(a + 0x32C, 2, 8);
    c.fld_s(a + 0x32E, 2, 8);
    traits(c, a + 0x330);
    traits(c, a + 0x35C);
    traits(c, a + 0x388);
    c.fld(a + 0x3B4, 1, 7, 0);
    c.fld(a + 0x3B5, 1, 7, 0);
    c.fld(a + 0x3B6, 1, 7, 0);
    c.fld(a + 0x734, 2, 3, 0);
    c.fld(a + 0x3B8, 1, 3, 0);
    c.fld(a + 0x3B9, 1, 2, 0);
    // eight teams, 0x58 bytes each, from a1+0x3BC
    for t in 0..8 {
        let b = a + 0x3BC + t * 0x58;
        c.fld(b, 1, 4, 0);
        if c.dec {
            for k in 0..12 { c.store(b + 4 + 2 * k, 2, 0xFFFF); }
        }
        let n = c.fld(b + 0x40, 4, 1, 0) as usize;
        if n > 0 {
            // twelve optional 5-bit words
            for k in 0..12 {
                let o = b + 4 + 2 * k;
                if c.dec {
                    if c.io(1, 0) != 0 {
                        let v = c.io(5, 0);
                        c.store(o, 2, v);
                    }
                } else {
                    let v = c.load(o, 2);
                    if v == 0xFFFF { c.io(1, 0); } else { c.io(1, 1); c.io(5, v); }
                }
            }
            // 6-bit-length name blob: data at +0x1C, dword length at +0x3C
            blob(c, b + 0x1C, b + 0x3C, 6, 0x20);
        }
        c.fld(b + 0x44, 1, 4, -1);
        c.fld(b + 0x45, 1, 1, 0);
        c.fld(b + 0x48, 4, 32, 0);
        c.fld(b + 0x4C, 4, 32, 0);
        c.fld(b + 0x50, 4, 32, 0);
        c.fld(b + 0x54, 2, 5, 0);
    }
    c.fld(a + 0x67C, 1, 2, 0);
    // loadouts: 6 palettes x 5 entries x 6 bytes from a1+0x67D
    for g in 0..6 {
        for e in 0..5 {
            let b = a + 0x67D + g * 0x1E + e * 6;
            c.fld(b, 1, 1, 0);
            // name: flag 1 = none (0xFF) else 7 bits
            c.opt(b + 1, 1, 7);
            c.fld_s(b + 2, 1, 8);
            c.fld_s(b + 3, 1, 8);
            c.fld_s(b + 4, 1, 8);
            c.fld(b + 5, 1, 4, 0);
        }
    }
}

// ===================================================================================================
// The Megalo script context (`0x55044`), at a1+0x63F4
// ===================================================================================================

/// Image offset of the script context.
pub const CTX: usize = A1 + 0x63F4;

fn cap(c: &mut Codec, what: &str, n: usize, max: usize) -> bool {
    if n > max {
        c.fail(format!("{what}: count {n} exceeds capacity {max}"));
        false
    } else {
        true
    }
}

/// One declaration set's four kinds, shared shape. `num` = (count_off, bits, init_off, prio_off,
/// cap), the others likewise; `None` where the scope has no such list.
struct Decl {
    count: usize,
    bits: u32,
    cap: usize,
    /// per-entry fields, in wire order: var6 init at `init + 3i`, team byte at `byte + i` (-1
    /// bias, 4 bits), priority at `prio + i` (2 bits)
    init: Option<usize>,
    team: Option<usize>,
    prio: Option<usize>,
}

fn decl(c: &mut Codec, d: &Decl, what: &str) {
    let n = c.fld(d.count, 4, d.bits, 0) as usize;
    if !cap(c, what, n, d.cap) {
        return;
    }
    for i in 0..n {
        if let Some(o) = d.init { var6(c, o + 3 * i); }
        if let Some(o) = d.team { c.fld(o + i, 1, 4, -1); }
        if let Some(o) = d.prio { c.fld(o + i, 1, 2, 0); }
    }
}

pub fn script_context(c: &mut Codec) {
    let x = CTX;
    // conditions
    let n = c.fld(x + 0xF08, 4, 10, 0) as usize;
    if cap(c, "conditions", n, 512) {
        for i in 0..n { condition(c, x + 0xF10 + 16 * i); }
    }
    // actions
    let n = c.fld(x + 0x2F10, 4, 11, 0) as usize;
    if cap(c, "actions", n, 1024) {
        for i in 0..n { action(c, x + 0x2F18 + 20 * i); }
    }
    // triggers
    let n = c.fld(x, 4, 9, 0) as usize;
    if cap(c, "triggers", n, 320) {
        for i in 0..n {
            let t = x + 8 + 12 * i;
            let k = c.fld(t + 8, 1, 3, 0);
            c.fld(t + 9, 1, 3, 0);
            if k == 5 {
                c.opt(t + 0xA, 1, 4);
            } else if c.dec {
                c.store(t + 0xA, 1, 0xFF);
            }
            p700c(c, t);
        }
    }
    // stats
    let n = c.fld(x + 0x7F18, 4, 3, 0) as usize;
    if cap(c, "stats", n, 4) {
        for i in 0..n {
            let e = x + 0x7F20 + 4 * i;
            c.fld(e, 1, 7, 0);
            c.fld(e + 1, 1, 2, 0);
            c.fld(e + 2, 1, 2, -1);
            c.fld(e + 3, 1, 1, 0);
        }
    }
    // global scope
    decl(c, &Decl { count: x + 0x7F30, bits: 4, cap: 12, init: Some(x + 0x7F38), team: None, prio: Some(x + 0x7F8C) }, "global numbers");
    decl(c, &Decl { count: x + 0x7F5C, bits: 4, cap: 8, init: Some(x + 0x7F64), team: None, prio: None }, "global timers");
    decl(c, &Decl { count: x + 0x7F7C, bits: 4, cap: 8, init: None, team: Some(x + 0x7F84), prio: Some(x + 0x7FB0) }, "global teams");
    decl(c, &Decl { count: x + 0x7FB8, bits: 4, cap: 8, init: None, team: None, prio: Some(x + 0x7F98) }, "global players");
    decl(c, &Decl { count: x + 0x7FBC, bits: 5, cap: 16, init: None, team: None, prio: Some(x + 0x7FA0) }, "global objects");
    // player scope
    decl(c, &Decl { count: x + 0x7FC0, bits: 4, cap: 8, init: Some(x + 0x7FC8), team: None, prio: Some(x + 0x8000) }, "player numbers");
    decl(c, &Decl { count: x + 0x7FE0, bits: 3, cap: 4, init: Some(x + 0x7FE8), team: None, prio: None }, "player timers");
    decl(c, &Decl { count: x + 0x7FF4, bits: 3, cap: 4, init: None, team: Some(x + 0x7FFC), prio: Some(x + 0x8010) }, "player teams");
    decl(c, &Decl { count: x + 0x8014, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x8008) }, "player players");
    decl(c, &Decl { count: x + 0x8018, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x800C) }, "player objects");
    // object scope
    decl(c, &Decl { count: x + 0x801C, bits: 4, cap: 8, init: Some(x + 0x8024), team: None, prio: Some(x + 0x805C) }, "object numbers");
    decl(c, &Decl { count: x + 0x803C, bits: 3, cap: 4, init: Some(x + 0x8044), team: None, prio: None }, "object timers");
    decl(c, &Decl { count: x + 0x8050, bits: 2, cap: 4, init: None, team: Some(x + 0x8058), prio: Some(x + 0x806C) }, "object teams");
    decl(c, &Decl { count: x + 0x8070, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x8064) }, "object players");
    decl(c, &Decl { count: x + 0x8074, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x8068) }, "object objects");
    // team scope
    decl(c, &Decl { count: x + 0x8078, bits: 4, cap: 8, init: Some(x + 0x8080), team: None, prio: Some(x + 0x80B8) }, "team numbers");
    decl(c, &Decl { count: x + 0x8098, bits: 3, cap: 4, init: Some(x + 0x80A0), team: None, prio: None }, "team timers");
    decl(c, &Decl { count: x + 0x80AC, bits: 3, cap: 4, init: None, team: Some(x + 0x80B4), prio: Some(x + 0x80CA) }, "team teams");
    decl(c, &Decl { count: x + 0x80D0, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x80C0) }, "team players");
    decl(c, &Decl { count: x + 0x80D4, bits: 3, cap: 4, init: None, team: None, prio: Some(x + 0x80C4) }, "team objects");
    // HUD widget slots (one nibble each)
    let n = c.fld(x + 0x80D8, 4, 3, 0) as usize;
    if cap(c, "widget slots", n, 4) {
        for i in 0..n { c.fld(x + 0x80E0 + i, 1, 4, 0); }
    }
    // seven trigger indices (-1 = none) and 64 dwords
    for k in 0..7 {
        c.fld(x + 0x81EC + 4 * k, 4, 9, -1);
    }
    for k in 0..64 {
        c.fld(x + 0x8208 + 4 * k, 4, 32, 0);
    }
    // HUD widgets
    let n = c.fld(x + 0x80E4, 4, 5, 0) as usize;
    if cap(c, "hud widgets", n, 16) {
        for i in 0..n {
            let w = x + 0x80EC + 16 * i;
            if c.dec {
                c.store(w, 4, 0xFFFF_FFFF);
                c.store(w + 4, 4, 0xFFFF_FFFF);
                c.store(w + 0xA, 1, 0xFF);
            }
            c.fld(w, 4, 7, -1);
            let f = c.fld(w + 0xB, 1, 3, 0);
            if f & 1 != 0 { i11(c, w + 4); }
            if f & 2 != 0 { c.fld(w + 0xA, 1, 4, -1); }
            if f & 4 != 0 { c.fld_s(w + 8, 2, 16); }
            c.fld(w + 0xC, 2, 7, 0);
        }
    }
}

// ===================================================================================================
// The Megalo variant (`0xDEB40` / `0xDE090`) and engine dispatch
// ===================================================================================================

/// 8-bit float at `a1+0xF858`: the 0..10 quantiser, with the engine's two special codes.
fn f858(c: &mut Codec) {
    let o = A1 + 0xF858;
    if c.dec {
        c.marks.push(("f858", c.rd.pos));
        let v = c.io(8, 0);
        let flag = c.img[A1 + 0xF840] & 1 != 0;
        let k = 0.0393700786f32;
        let k2 = 0.0196850393f32;
        let f = match v {
            0 => 0.0,
            0x19 | 0x1A => if !flag { 1.0 } else { 24.0f32 * k + k2 },
            0xFF => 10.0,
            _ => (v as i32 - 1) as f32 * k + k2,
        };
        c.store(o, 4, f.to_bits() as u64);
    } else {
        let f = f32::from_bits(c.load(o, 4) as u32);
        let v = quant(f, 0.0, 10.0, 8);
        c.io(8, v);
    }
}

/// The Megalo extension of the base variant. Image layout offsets are relative to `a1`.
pub fn megalo_variant(c: &mut Codec) {
    let a = A1;
    let ver = c.fld(a + 0x5DC0, 4, 32, 0);
    if ver != 0x6A && ver != 0x6B {
        c.fail(format!("unsupported script encoding version {ver}"));
        return;
    }
    // the second dword is the writer's program version: -1 ("unspecified") in every saved file
    if c.dec {
        c.marks.push(("program-version", c.rd.pos));
        c.io(32, 0);
    } else {
        c.io(32, 0xFFFF_FFFF);
    }
    base_variant(c);
    // player-trait sets: a name index, a description index and a traits record each
    let n = c.fld(a + 0x5DC4, 4, 5, 0) as usize;
    if cap(c, "player trait sets", n, 16) {
        for i in 0..n {
            let r = a + 0x5DCC + 0x30 * i;
            c.fld(r + 0x2C, 1, 7, 0);
            c.fld(r + 0x2D, 1, 7, 0);
            traits(c, r);
        }
    }
    // user-defined options
    let n = c.fld(a + 0x60CC, 4, 5, 0) as usize;
    if cap(c, "scripted options", n, 16) {
        for i in 0..n {
            let r = a + 0x60D4 + 0x30 * i;
            if c.dec { c.store(r, 2, 0xFFFF); }
            c.fld(r, 1, 7, 0);
            c.fld(r + 1, 1, 7, 0);
            let range = c.fld(r + 2, 1, 1, 0) != 0;
            let cnt;
            if range {
                c.fld_s(r + 4, 2, 10);
                cnt = 2usize;
                c.store(r + 8, 4, 2);
            } else {
                c.fld(r + 4, 2, 3, 0);
                cnt = c.fld(r + 8, 4, 4, 0) as usize;
            }
            if cnt > 8 {
                c.fail("scripted option with more than 8 values");
                return;
            }
            for k in 0..cnt {
                let e = r + 0x10 + 4 * k;
                c.fld_s(e, 2, 10);
                if !range {
                    c.fld(e + 2, 1, 7, 0);
                    c.fld(e + 3, 1, 7, 0);
                } else if c.dec {
                    c.store(e + 2, 1, 0xFF);
                    c.store(e + 3, 1, 0xFF);
                }
            }
            // the option's current value
            let cur = a + 0x63D4 + 2 * i;
            if range { c.fld_s(cur, 2, 10); } else { c.fld(cur, 2, 3, 0); }
        }
    }
    script_strings(c);
    c.fld(a + 0xE6FC, 4, 7, -1);
    one_string(c, a + 0xE848, a + 0xE848 + 0x19C, 9, a + 0xE848 + 0x18, a + 0xE848 + 0x198, 9, 0x180);
    one_string(c, a + 0xE9E8, a + 0xF604, 12, a + 0xEA00, a + 0xF600, 12, 0xC00);
    one_string(c, a + 0xF608, a + 0xF608 + 0x19C, 9, a + 0xF608 + 0x18, a + 0xF608 + 0x198, 9, 0x180);
    c.fld(a + 0xF7AC, 4, 5, -1);
    c.fld(a + 0xF7A8, 4, 5, -1);
    // required-object list: up to 32 signed words then a flag
    if c.dec { c.store(a + 0xF7F8, 1, 1); }
    let stored_n = if c.dec { 0 } else { c.load(a + 0xF7B0, 4) };
    let n = c.io(6, stored_n) as usize;
    if n <= 0x20 {
        if c.dec { c.store(a + 0xF7B0, 4, n as u64); }
        for i in 0..n {
            c.fld_s(a + 0xF7B8 + 2 * i, 2, 16);
        }
        c.fld(a + 0xF7F8, 1, 1, 0);
    } else {
        c.fail("required-object list longer than 32");
        return;
    }
    for k in 0..15 {
        c.fld(a + 0xF7FC + 4 * k, 4, 32, 0);
    }
    c.fld(a + 0xF838, 1, 1, 0);
    c.fld(a + 0xF83C, 2, 16, 0);
    c.fld(a + 0xF83E, 1, 1, 0);
    c.fld(a + 0xF83F, 1, 1, 0);
    for k in 0..40 { c.fld(a + 0xE700 + 4 * k, 4, 32, 0); }
    for k in 0..40 { c.fld(a + 0xE7A0 + 4 * k, 4, 32, 0); }
    c.fld(a + 0xE840, 4, 32, 0);
    c.fld(a + 0xE844, 4, 32, 0);
    script_context(c);
    if ver <= 0x6A {
        return;
    }
    c.fld(a + 0xF840, 4, 32, 0);
    for k in 0..5 {
        c.qfloat(a + 0xF844 + 4 * k, 0.0, 2.0, 8);
    }
    f858(c);
    c.qfloat(a + 0xF85C, 0.0, 10.0, 8);
}

/// Engine types: 0 none, 1 sandbox (Forge), 2 megalo (custom), 3 campaign, 4 survival (Firefight).
pub fn variant_body(c: &mut Codec, engine: u32) {
    match engine {
        0 | 3 => base_variant(c),
        2 => megalo_variant(c),
        1 => {
            megalo_variant(c);
            let a = A1;
            c.fld(a + 0xF860, 1, 2, 0);
            c.fld(a + 0xF861, 1, 2, 0);
            c.fld(a + 0xF862, 2, 6, 0);
            traits(c, a + 0xF864);
        }
        4 => survival_variant(c),
        _ => c.fail(format!("engine type {engine} is not a Reach game variant kind")),
    }
}

/// `0x1806A3D78`: ten small fields (a Firefight wave/skull option block).
fn survival_opts(c: &mut Codec, o: usize) {
    for (i, w) in [3u32, 2, 3, 2, 2, 2, 2, 2, 4, 4].iter().enumerate() {
        c.fld(o + i, 1, *w, 0);
    }
}

/// `0x1800F81B4`: a 15-byte record: two bits, a nibble and twelve bytes stored minus one.
fn survival_rec(c: &mut Codec, o: usize) {
    c.fld(o, 1, 1, 0);
    c.fld(o + 1, 1, 1, 0);
    c.fld(o + 2, 1, 4, 0);
    for k in 0..12 {
        c.fld(o + 3 + k, 1, 8, -1);
    }
}

/// Engine type 4, Firefight ("survival"): the base variant plus lives/bonus rules and four
/// wave-group blocks (`0xF7DBC`).
pub fn survival_variant(c: &mut Codec) {
    let a = A1;
    base_variant(c);
    c.fld(a + 0x738, 1, 5, 0);
    let ver = c.fld(a + 0x73A, 1, 3, 0);
    if ver > 2 {
        c.fail(format!("unsupported survival encoding version {ver}"));
        return;
    }
    c.fld(a + 0x740, 1, 8, 0);
    c.fld(a + 0x741, 1, 4, 0);
    c.fld(a + 0x742, 2, 15, 0);
    c.fld(a + 0x744, 2, 15, 0);
    c.fld(a + 0x9C4, 2, 7, -1);
    c.fld(a + 0x9C6, 2, 7, -1);
    c.fld(a + 0x9C8, 2, 15, 0);
    c.fld(a + 0x9CA, 2, 7, -1);
    c.fld(a + 0x9CC, 2, 2, 0);
    traits(c, a + 0x7FC);
    traits(c, a + 0x828);
    survival_opts(c, a + 0x854);
    for g in 0..3 {
        let b = a + 0x8B8 + g * 0x64;
        traits(c, b - 0x58);
        traits(c, b - 0x2C);
        survival_opts(c, b);
    }
    base_traits(c, a + 0x98C);
    for i in 0..3 {
        let b = a + 0x75B + i * 0x34;
        c.fld(b - 0x13, 4, 18, 0);
        survival_rec(c, b - 0xF);
        survival_rec(c, b);
        survival_rec(c, b + 0xF);
    }
    c.fld(a + 0x7E4, 2, 12, 0);
    c.fld(a + 0x7E8, 4, 18, 0);
    survival_rec(c, a + 0x7EC);
    if ver >= 2 {
        c.fld(a + 0x739, 1, 8, 0);
    } else if c.dec {
        // the engine upgrades an old (version 1) variant in memory: flags 0, encoding version 2,
        // so it writes the extra byte back out
        c.store(a + 0x739, 1, 0);
        c.store(a + 0x73A, 1, 2);
    }
}

/// Decode a `mpvr` bitstream (the bytes after the chunk's 40-byte header) into the 64,516-byte
/// `c_game_variant` image.
pub fn decode_payload(payload: &[u8]) -> Result<Vec<u8>, String> {
    let mut c = Codec::decoder(payload, IMAGE_LEN);
    decode_into(&mut c)?;
    Ok(c.img)
}

fn decode_into(c: &mut Codec) -> Result<(), String> {
    let engine = c.io(4, 0) as u32;
    c.store(0, 4, engine as u64);
    variant_body(c, engine);
    if let Some(e) = &c.err {
        return Err(e.clone());
    }
    if c.rd.overrun {
        return Err("the bitstream ended before the variant did".into());
    }
    Ok(())
}

/// Encode a `c_game_variant` image into the `mpvr` bitstream bytes (and the exact bit count).
pub fn encode_image(img: &[u8]) -> Result<(Vec<u8>, usize), String> {
    if img.len() < IMAGE_LEN {
        return Err(format!("image is {} bytes, expected {}", img.len(), IMAGE_LEN));
    }
    let mut c = Codec::encoder(img[..IMAGE_LEN].to_vec());
    let engine = u32::from_le_bytes([img[0], img[1], img[2], img[3]]);
    if engine > 4 { return Err(format!("engine type {engine} is not a Reach variant kind")); }
    c.io(4, engine as u64);
    variant_body(&mut c, engine);
    if let Some(e) = c.err.take() {
        return Err(e);
    }
    Ok((std::mem::take(&mut c.wr.buf), c.wr.pos))
}

/// How a source bitstream relates to what we re-encode from its decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundTrip {
    /// Byte-identical.
    Exact,
    /// Identical except the writer's "program version" dword (0 from some third-party tools, -1
    /// from the game): it is not part of the variant, the engine ignores it on load.
    ProgramVersion,
    /// Identical except the 8-bit code of the `0xF858` float: the engine maps two codes (0x19 and
    /// 0x1A) to one value, so a file holding 0x1A comes back as the canonical 0x19.
    FloatCode,
    /// A Firefight variant saved with encoding version 1: the engine upgrades it to version 2 in
    /// memory and writes one more byte when it saves again (verified against the engine's encoder).
    FirefightUpgrade,
    /// A real difference, with the bit position of the first differing bit.
    Other(usize),
}

fn bits_equal(a: &[u8], abits: usize, b: &[u8], bbits: usize) -> bool {
    abits == bbits && a[..(abits + 7) / 8] == b[..(abits + 7) / 8]
}

fn put_bits(buf: &mut [u8], pos: usize, n: usize, v: u64) {
    for i in 0..n {
        let bit = (v >> (n - 1 - i)) & 1;
        let p = pos + i;
        if bit == 1 { buf[p / 8] |= 0x80 >> (p % 8); } else { buf[p / 8] &= !(0x80 >> (p % 8)); }
    }
}

/// Decode `src` (a `mpvr` bitstream), re-encode, and say how the two relate.
pub fn classify_roundtrip(src: &[u8]) -> Result<RoundTrip, String> {
    let mut c = Codec::decoder(src, IMAGE_LEN);
    decode_into(&mut c)?;
    let used = c.rd.pos;
    let engine = u32::from_le_bytes([c.img[0], c.img[1], c.img[2], c.img[3]]);
    let (enc, ebits) = encode_image(&c.img)?;
    if bits_equal(&enc, ebits, src, used) {
        return Ok(RoundTrip::Exact);
    }
    if engine == 4 && ebits == used + 8 {
        return Ok(RoundTrip::FirefightUpgrade);
    }
    let marks = c.marks.clone();
    let patched = |name: &str, n: usize, v: u64, from_enc: bool| -> Option<bool> {
        let at = marks.iter().find(|m| m.0 == name)?.1;
        let mut s2 = src.to_vec();
        if s2.len() * 8 < at + n { return None; }
        let v = if from_enc {
            // the code the encoder wrote at that spot
            let mut x = 0u64;
            for i in 0..n { x = (x << 1) | ((enc[(at + i) / 8] >> (7 - (at + i) % 8)) & 1) as u64; }
            x
        } else { v };
        put_bits(&mut s2, at, n, v);
        Some(bits_equal(&enc, ebits, &s2, used))
    };
    if patched("program-version", 32, 0xFFFF_FFFF, false) == Some(true) {
        return Ok(RoundTrip::ProgramVersion);
    }
    if patched("f858", 8, 0, true) == Some(true) {
        return Ok(RoundTrip::FloatCode);
    }
    // both quirks at once
    let (mut both, ok) = (src.to_vec(), marks.len() >= 2);
    if ok {
        for (name, n, from_enc) in [("program-version", 32usize, false), ("f858", 8, true)] {
            if let Some(&(_, at)) = marks.iter().find(|m| m.0 == name) {
                let mut x = 0u64;
                if from_enc { for i in 0..n { x = (x << 1) | ((enc[(at + i) / 8] >> (7 - (at + i) % 8)) & 1) as u64; } } else { x = 0xFFFF_FFFF; }
                put_bits(&mut both, at, n, x);
            }
        }
        if bits_equal(&enc, ebits, &both, used) {
            return Ok(RoundTrip::FloatCode);
        }
    }
    let first = (0..used.min(ebits)).find(|&b| ((enc[b / 8] ^ src[b / 8]) >> (7 - b % 8)) & 1 != 0).unwrap_or(used.min(ebits));
    Ok(RoundTrip::Other(first))
}

// ===================================================================================================
// The BLF container
// ===================================================================================================

/// `_blf` header chunk: identical in every game variant file (0x30 bytes, byte-order mark 0xFFFE).
const BLF_HEADER: [u8; 0x30] = {
    let mut h = [0u8; 0x30];
    h[0] = b'_'; h[1] = b'b'; h[2] = b'l'; h[3] = b'f';
    h[7] = 0x30;
    h[9] = 1; // major 1
    h[11] = 2; // minor 2
    h[12] = 0xFF; h[13] = 0xFE;
    h
};

/// Size of the `chdr` chunk (12 byte header + 4 + the 0x2B0-byte metadata).
pub const CHDR_LEN: usize = 0x2C0;
/// Size of the `mpvr` chunk: 12 header + 20 hash + 4 + 4 length + 0x5000 bitstream.
pub const MPVR_LEN: usize = 0x5028;
/// File offset of the `mpvr` chunk, and of the `_eof` chunk.
pub const MPVR_AT: usize = 0x30 + CHDR_LEN;
pub const EOF_AT: usize = MPVR_AT + MPVR_LEN;
/// A whole game variant file without the optional `_fsm` signature block: 21,289 bytes.
pub const FILE_LEN: usize = EOF_AT + 0x11;

/// What a `chdr` chunk carries in its first four body bytes. MCC writes `word_182DFBE20` then a
/// game-state word there; both read 0xFFFF when nothing is loaded ("saved from outside a match").
pub const CHDR_TAG_UNSET: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

fn be32(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

/// The chunk hash the engine checks: SHA-1 over the four length bytes exactly as stored, then the
/// first `len` bytes of the bitstream.
pub fn mpvr_hash(len_bytes: [u8; 4], data: &[u8]) -> [u8; 20] {
    let mut m = Vec::with_capacity(4 + data.len());
    m.extend_from_slice(&len_bytes);
    m.extend_from_slice(data);
    crate::h4::mvar::sha1(&m)
}

/// Build a complete, loadable `.bin` from a `c_game_variant` image.
pub fn build_bin(image: &[u8], chdr_tag: [u8; 4]) -> Result<Vec<u8>, String> {
    let (bits, nbits) = encode_image(image)?;
    let len = (nbits + 7) / 8;
    if len > MPVR_CAPACITY {
        return Err(format!(
            "the game variant needs {len} bytes of bitstream but a .bin holds at most {MPVR_CAPACITY} (its string tables are too large)"
        ));
    }
    let mut out = Vec::with_capacity(FILE_LEN);
    out.extend_from_slice(&BLF_HEADER);
    // chdr
    out.extend_from_slice(b"chdr");
    out.extend_from_slice(&(CHDR_LEN as u32).to_be_bytes());
    out.extend_from_slice(&[0x00, 0x0A, 0x00, 0x02]);
    out.extend_from_slice(&chdr_tag);
    out.extend_from_slice(&image[A1 + 8..A1 + 8 + 0x2B0]);
    // mpvr
    out.extend_from_slice(b"mpvr");
    out.extend_from_slice(&(MPVR_LEN as u32).to_be_bytes());
    out.extend_from_slice(&[0x00, 0x36, 0x00, 0x01]);
    let lenb = (len as u32).to_be_bytes();
    out.extend_from_slice(&mpvr_hash(lenb, &bits[..len]));
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&lenb);
    let at = out.len();
    out.extend_from_slice(&bits[..len]);
    out.resize(at + MPVR_CAPACITY, 0);
    // _eof
    out.extend_from_slice(b"_eof");
    out.extend_from_slice(&0x11u32.to_be_bytes());
    out.extend_from_slice(&[0x00, 0x01, 0x00, 0x01]);
    out.extend_from_slice(&(EOF_AT as u32).to_be_bytes());
    out.push(0);
    debug_assert_eq!(out.len(), FILE_LEN);
    Ok(out)
}

/// What [`parse_bin`] learned about a file besides the variant itself.
#[derive(Clone, Debug)]
pub struct BinInfo {
    /// The 4 bytes at the start of the `chdr` body.
    pub chdr_tag: [u8; 4],
    /// Bytes of bitstream the chunk declares, and how many bits the decoder actually consumed.
    pub declared_len: usize,
    pub bits_used: usize,
    /// The stored SHA-1 equals the recomputed one.
    pub hash_ok: bool,
    /// An `_fsm` (file-share signature) chunk follows `_eof`.
    pub has_fsm: bool,
    /// `chdr`'s metadata copy is byte-identical to the decoded variant's metadata.
    pub chdr_matches: bool,
    pub engine: u32,
}

/// Parse a game variant `.bin` into its `c_game_variant` image.
pub fn parse_bin(d: &[u8]) -> Result<(Vec<u8>, BinInfo), String> {
    if d.len() < 0x30 + 12 || &d[0..4] != b"_blf" {
        return Err("not a BLF file (no _blf header)".into());
    }
    let mut pos = 0usize;
    let (mut chdr, mut mpvr, mut fsm) = (None, None, false);
    while pos + 12 <= d.len() {
        let size = be32(d, pos + 4) as usize;
        if size < 12 || pos + size > d.len() {
            break;
        }
        match &d[pos..pos + 4] {
            b"chdr" => chdr = Some((pos, size)),
            b"mpvr" => mpvr = Some((pos, size)),
            b"_fsm" => fsm = true,
            _ => {}
        }
        pos += size;
    }
    let (mp, msize) = mpvr.ok_or("no mpvr chunk: this is not a Reach game variant (or is a Halo 3/4 one)")?;
    let ver = u16::from_be_bytes([d[mp + 8], d[mp + 9]]);
    if ver != 0x36 {
        return Err(format!("mpvr chunk version {ver} (this codec reads Reach's version 54)"));
    }
    if msize < 40 {
        return Err("mpvr chunk too small".into());
    }
    let lenb = [d[mp + 36], d[mp + 37], d[mp + 38], d[mp + 39]];
    let be = u32::from_be_bytes(lenb) as usize;
    let le = u32::from_le_bytes(lenb) as usize;
    let avail = msize - 40;
    let len = if be <= avail { be } else if le <= avail { le } else { return Err(format!("mpvr declares {be} bytes of bitstream, chunk holds {avail}")); };
    let data = &d[mp + 40..mp + 40 + len];
    let hash_ok = d[mp + 12..mp + 32] == mpvr_hash(lenb, data);
    let mut c = Codec::decoder(data, IMAGE_LEN);
    decode_into(&mut c)?;
    let bits_used = c.rd.pos;
    let engine = u32::from_le_bytes([c.img[0], c.img[1], c.img[2], c.img[3]]);
    let (chdr_tag, chdr_matches) = match chdr {
        Some((cp, cs)) if cs >= 16 + 0x2B0 => {
            let mut t = [0u8; 4];
            t.copy_from_slice(&d[cp + 12..cp + 16]);
            (t, d[cp + 16..cp + 16 + 0x2B0] == c.img[A1 + 8..A1 + 8 + 0x2B0])
        }
        _ => (CHDR_TAG_UNSET, false),
    };
    Ok((c.img, BinInfo { chdr_tag, declared_len: len, bits_used, hash_ok, has_fsm: fsm, chdr_matches, engine }))
}

// ===================================================================================================
// Readable access to the image (the names the file-extraction report and the tests use)
// ===================================================================================================

/// UTF-16LE string out of the image, up to the NUL.
fn wide(img: &[u8], off: usize, cap: usize) -> String {
    let u: Vec<u16> = (0..cap).map(|i| u16::from_le_bytes([img[off + 2 * i], img[off + 2 * i + 1]])).take_while(|&c| c != 0).collect();
    String::from_utf16_lossy(&u)
}

fn narrow(img: &[u8], off: usize, cap: usize) -> String {
    let b: Vec<u8> = img[off..off + cap].iter().copied().take_while(|&c| c != 0).collect();
    String::from_utf8_lossy(&b).into_owned()
}

/// Title / description / author / editor of a variant image.
pub fn titles(img: &[u8]) -> (String, String, String, String) {
    let m = A1 + 8;
    (
        wide(img, m + 0x80, 128),
        wide(img, m + 0x180, 128),
        narrow(img, m + 0x38 + 0x10, 16),
        narrow(img, m + 0x5C + 0x10, 16),
    )
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// A small but non-trivial Megalo variant image built by poking the engine's own offsets:
    /// metadata, a title, a trait set, one condition, two actions, one trigger, a declaration, a
    /// HUD widget and a one-string name table with a compressible blob.
    pub(super) fn synthetic_image() -> Vec<u8> {
        let mut img = vec![0u8; IMAGE_LEN];
        let w32 = |img: &mut Vec<u8>, o: usize, v: u32| img[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let w16 = |img: &mut Vec<u8>, o: usize, v: u16| img[o..o + 2].copy_from_slice(&v.to_le_bytes());
        w32(&mut img, 0, 2); // megalo
        let a = A1;
        let m = a + 8;
        img[m] = 6; // content type: game variant
        w32(&mut img, m + 4, 0x5329);
        img[m + 0x28] = 0; // byte kinds
        for (i, ch) in "Sabre Duel".encode_utf16().enumerate() {
            w16(&mut img, m + 0x80 + 2 * i, ch);
        }
        for (i, ch) in "A test gametype".encode_utf16().enumerate() {
            w16(&mut img, m + 0x180 + 2 * i, ch);
        }
        for (i, b) in b"Sopitive".iter().enumerate() {
            img[m + 0x38 + 0x10 + i] = *b;
        }
        w32(&mut img, a + 0x5DC0, 107);
        w32(&mut img, a + 0x5DC4, 0); // no trait sets
        img[a + 0x2BA] = 15;
        img[a + 0x2BB] = 3;
        w16(&mut img, a + 0x2BC, 0xFFFF); // stored -1
        // base traits / loadouts left zero except one loadout name "none"
        for g in 0..6 { for e in 0..5 { img[a + 0x67D + g * 0x1E + e * 6 + 1] = 0xFF; } }
        for t in 0..8 { w16(&mut img, a + 0x3BC + t * 0x58 + 4, 0xFFFF); }
        // optional fields that default to "none"
        w32(&mut img, a + 0xE6FC, 0xFFFF_FFFF);
        w32(&mut img, a + 0xF7AC, 0xFFFF_FFFF);
        w32(&mut img, a + 0xF7A8, 0xFFFF_FFFF);
        img[a + 0xF7F8] = 1;
        for k in 0..7 { w32(&mut img, CTX + 0x81EC + 4 * k, 0xFFFF_FFFF); }
        // script strings: 2 strings, English only, 100 bytes of repetitive text (compresses)
        w32(&mut img, a + 0x5DBC, 2);
        for l in 0..12 { for i in 0..112 { w16(&mut img, a + 0x738 + l * 0xE0 + 2 * i, 0xFFFF); } }
        w16(&mut img, a + 0x738, 0);
        w16(&mut img, a + 0x738 + 2, 6);
        let text = b"Hello\0Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello\0";
        img[a + 0x11B8..a + 0x11B8 + text.len()].copy_from_slice(text);
        w32(&mut img, a + 0x5DB8, text.len() as u32);
        // a condition: type 7 (var3a) ; an action: opcode 0x14 ; a trigger
        w32(&mut img, CTX + 0xF08, 1);
        img[CTX + 0xF10 + 0xC] = 7;
        img[CTX + 0xF10 + 2] = 0; // var3a type 0: value stored as v-1
        img[CTX + 0xF10] = 4;
        w32(&mut img, CTX + 0x2F10, 2);
        img[CTX + 0x2F18 + 0x10] = 0x14;
        w32(&mut img, CTX + 0x2F18, 300);
        img[CTX + 0x2F18 + 20 + 0x10] = 0x20; // flag action
        img[CTX + 0x2F18 + 20] = 1;
        w32(&mut img, CTX, 1);
        img[CTX + 8 + 8] = 2;
        img[CTX + 8 + 10] = 0xFF;
        w16(&mut img, CTX + 8, 17);
        // one global number declaration (value type 0 = int16 constant 5), priority 2
        w32(&mut img, CTX + 0x7F30, 1);
        img[CTX + 0x7F38 + 2] = 0;
        w16(&mut img, CTX + 0x7F38, 5);
        img[CTX + 0x7F8C] = 2;
        // a HUD widget
        w32(&mut img, CTX + 0x80E4, 1);
        w32(&mut img, CTX + 0x80EC, 3);
        w32(&mut img, CTX + 0x80F0, 0xFFFF_FFFF);
        img[CTX + 0x80EC + 0xA] = 0xFF;
        img
    }

    #[test]
    fn container_constants() {
        assert_eq!(FILE_LEN, 21289);
        assert_eq!(MPVR_AT, 0x2F0);
        assert_eq!(EOF_AT, 0x5318);
        assert_eq!(IMAGE_LEN, 64516);
    }

    #[test]
    fn bit_io_is_msb_first() {
        let mut w = BitWr::default();
        w.bits(3, 0b101);
        w.bits(12, 0xABC);
        w.bits(1, 1);
        assert_eq!(w.pos, 16);
        assert_eq!(w.buf, vec![0b1011_0101, 0b0111_1001]);
        let mut r = BitRd::new(&w.buf);
        assert_eq!(r.bits(3), 0b101);
        assert_eq!(r.bits(12), 0xABC);
        assert_eq!(r.bits(1), 1);
        assert!(!r.overrun);
        r.bits(1);
        assert!(r.overrun, "reading past the end is flagged");
    }

    #[test]
    fn quantiser_matches_the_engine_formula() {
        // the two exact ends and the cell-centre values in between
        assert_eq!(quant(0.0, 0.0, 2.0, 8), 0);
        assert_eq!(quant(2.0, 0.0, 2.0, 8), 255);
        assert_eq!(dequant(0, 0.0, 2.0, 8), 0.0);
        assert_eq!(dequant(255, 0.0, 2.0, 8), 2.0);
        for code in 1..=254u64 {
            let f = dequant(code, 0.0, 10.0, 8);
            assert_eq!(quant(f, 0.0, 10.0, 8), code, "code {code} -> {f}");
        }
        // 1.0 on the 0..10 scale lands on 0x1A, the code the engine writes for it
        assert_eq!(quant(1.0, 0.0, 10.0, 8), 0x1A);
    }

    #[test]
    fn synthetic_variant_round_trips_and_the_container_verifies() {
        let img = synthetic_image();
        let (bits, n) = encode_image(&img).expect("encode");
        // decode -> encode is the identity on the bitstream
        let back = decode_payload(&bits[..(n + 7) / 8]).expect("decode");
        let (bits2, n2) = encode_image(&back).expect("re-encode");
        assert_eq!((bits.clone(), n), (bits2, n2));
        // fields come back
        assert_eq!(titles(&back).0, "Sabre Duel");
        assert_eq!(titles(&back).2, "Sopitive");
        assert_eq!(&back[CTX + 0xF10 + 0xC..CTX + 0xF10 + 0xD], &[7]);
        // the whole file
        let file = build_bin(&img, CHDR_TAG_UNSET).expect("build");
        assert_eq!(file.len(), FILE_LEN);
        assert_eq!(&file[..4], b"_blf");
        assert_eq!(&file[0x30..0x34], b"chdr");
        assert_eq!(&file[MPVR_AT..MPVR_AT + 4], b"mpvr");
        assert_eq!(&file[EOF_AT..EOF_AT + 4], b"_eof");
        let (img2, info) = parse_bin(&file).expect("parse");
        assert!(info.hash_ok && info.chdr_matches && !info.has_fsm);
        assert_eq!(info.bits_used, n);
        assert_eq!(build_bin(&img2, info.chdr_tag).unwrap(), file, "parse -> build is the identity");
    }

    #[test]
    fn string_blobs_compress_like_the_engine_does() {
        let img = synthetic_image();
        let (bits, n) = encode_image(&img).unwrap();
        // the 119-byte blob is under the 0x80 threshold -> stored raw; push it over
        let mut big = img.clone();
        let a = A1;
        let text: Vec<u8> = b"Press X to capture the flag.\0".iter().cycle().take(600).copied().collect();
        big[a + 0x11B8..a + 0x11B8 + 600].copy_from_slice(&text);
        big[a + 0x5DB8..a + 0x5DB8 + 4].copy_from_slice(&600u32.to_le_bytes());
        let (bits_big, n_big) = encode_image(&big).unwrap();
        assert!(n_big < n + 600 * 8, "a 600-byte repetitive blob must be stored compressed");
        let _ = bits;
        let back = decode_payload(&bits_big[..(n_big + 7) / 8]).unwrap();
        assert_eq!(&back[a + 0x11B8..a + 0x11B8 + 600], &text[..]);
    }

    // ---- saboteurs: each of these must be able to FAIL ----

    #[test]
    fn a_flipped_payload_bit_breaks_the_chunk_hash() {
        let mut file = build_bin(&synthetic_image(), CHDR_TAG_UNSET).unwrap();
        file[MPVR_AT + 40 + 7] ^= 0x10;
        match parse_bin(&file) {
            Ok((_, info)) => assert!(!info.hash_ok, "a flipped bit must break the SHA-1"),
            Err(_) => {} // decoding a corrupted stream may also be refused outright
        }
    }

    #[test]
    fn a_zeroed_hash_is_reported() {
        let mut file = build_bin(&synthetic_image(), CHDR_TAG_UNSET).unwrap();
        for b in &mut file[MPVR_AT + 12..MPVR_AT + 32] { *b = 0; }
        let (_, info) = parse_bin(&file).unwrap();
        assert!(!info.hash_ok);
    }

    #[test]
    fn a_stream_shifted_by_one_bit_does_not_decode_to_the_same_variant() {
        // the "wrong width" saboteur: if any field width in the codec were off by one, every
        // later field would read shifted, exactly like this
        let img = synthetic_image();
        let (bits, n) = encode_image(&img).unwrap();
        let mut w = BitWr::default();
        w.bits(1, 0);
        let mut r = BitRd::new(&bits);
        let mut left = n;
        while left > 0 { let k = left.min(32); w.bits(k as u32, r.bits(k as u32)); left -= k; }
        let shifted = decode_payload(&w.buf);
        match shifted {
            Err(_) => {}
            Ok(other) => assert_ne!(titles(&other).0, "Sabre Duel", "a shifted stream must not look right"),
        }
    }

    #[test]
    fn an_oversized_variant_is_refused_not_truncated() {
        // 112 strings of 12 languages with a huge incompressible blob exceeds the 20,480-byte chunk
        let mut img = synthetic_image();
        let a = A1;
        let mut x = 0x1234_5678u32;
        for k in 0..0x4C00 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            img[a + 0x11B8 + k] = (x >> 24) as u8;
        }
        img[a + 0x5DB8..a + 0x5DB8 + 4].copy_from_slice(&0x4C00u32.to_le_bytes());
        for l in 0..12 { for i in 0..2 { img[a + 0x738 + l * 0xE0 + 2 * i..a + 0x738 + l * 0xE0 + 2 * i + 2].copy_from_slice(&((l * 100 + i) as u16).to_le_bytes()); } }
        // 12 offsets x 2 strings + 19,456 random bytes: 19,456 + header is under 20,480, so grow the
        // other long fields too: force the 15-bit-length string count to its maximum
        img[a + 0x5DBC..a + 0x5DBC + 4].copy_from_slice(&112u32.to_le_bytes());
        for l in 0..12 { for i in 0..112 { img[a + 0x738 + l * 0xE0 + 2 * i..a + 0x738 + l * 0xE0 + 2 * i + 2].copy_from_slice(&((i * 7) as u16).to_le_bytes()); } }
        let e = build_bin(&img, CHDR_TAG_UNSET).expect_err("19 KB of random strings plus 112 x 12 offsets cannot fit a 20,480-byte chunk");
        assert!(e.contains("at most"), "{e}");
    }

    #[test]
    fn a_wrong_engine_kind_is_refused() {
        let mut img = synthetic_image();
        img[0] = 9;
        assert!(encode_image(&img).is_err());
        assert!(parse_bin(b"_blf not a game variant at all............................").is_err());
    }
}

#[cfg(test)]
mod vector_tests {
    use super::*;

    /// Engine-derived test vectors (scratch JSON from the emulation harness): each is the bit string
    /// the real decoder was fed plus the record bytes it produced. The decode must consume exactly
    /// those bits and produce exactly those bytes, and the encode must give the bits back.
    fn run(file: &str, f: fn(&mut Codec, usize), size: usize, opbits: usize) -> (usize, usize, usize) {
        let dir = std::env::var("HMS_GV_VECTORS").unwrap_or_default();
        let path = format!("{dir}/{file}");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("gamevariant: no vector file {path} -- skipped");
            return (0, 0, 0);
        };
        let (mut n, mut bad_dec, mut bad_enc) = (0, 0, 0);
        let mut shown = 0;
        let mut per_op: std::collections::BTreeMap<u32, (u32, u32)> = Default::default();
        let noncanon = 0usize;
        let mut corrupt = 0usize;
        // tiny JSON scan: objects are flat {"op": N, "bits": "...", "widths": [...], "rec": "..."}
        for obj in text.split("{\"op\"").skip(1) {
            let op: u32 = obj.trim_start_matches(|c| c == ':' || c == ' ').split(',').next().unwrap().trim().parse().unwrap();
            let bits = obj.split("\"bits\": \"").nth(1).unwrap().split('"').next().unwrap();
            let rec = obj.split("\"rec\": \"").nth(1).unwrap().split('"').next().unwrap();
            let enc_bits = obj.split("\"enc\": \"").nth(1).unwrap().split('"').next().unwrap();
            let rec: Vec<u8> = (0..rec.len() / 2).map(|i| u8::from_str_radix(&rec[2 * i..2 * i + 2], 16).unwrap()).collect();
            let mut data = vec![0u8; (bits.len() + 7) / 8 + 8];
            for (i, ch) in bits.bytes().enumerate() {
                if ch == b'1' { data[i / 8] |= 0x80 >> (i % 8); }
            }
            n += 1;
            let mut c = Codec::decoder(&data, 256);
            f(&mut c, 0);
            let used = c.rd.pos;
            if c.out_of_domain {
                // the 2-bit operand count holds 0..2; 3 makes the engine's decoder overrun its own
                // record, so what it produced is memory corruption, not a record
                corrupt += 1;
                continue;
            }
            if used != bits.len() || c.img[..size] != rec[..size] {
                bad_dec += 1;
                per_op.entry(op).or_default().0 += 1;
                if shown < 400 && per_op[&op].0 == 1 {
                    shown += 1;
                    eprintln!("DEC MISMATCH op {op:#x}: used {used} of {} bits\n  bits {bits}\n  want {}\n  got  {}", bits.len(),
                        rec.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                        c.img[..size].iter().map(|b| format!("{b:02x}")).collect::<String>());
                }
                continue;
            }
            // an operand-list count of 3 (invalid: the field only holds 0..2) makes the engine's
            // decoder overrun into the opcode byte; its encoder then writes a different opcode.
            // Those vectors describe corrupted memory, not a record: skip them.
            if usize::from_str_radix(&enc_bits[..opbits.min(enc_bits.len())], 2).unwrap_or(op as usize) != op as usize {
                corrupt += 1;
                continue;
            }
            let mut e = Codec::encoder(c.img.clone());
            f(&mut e, 0);
            let mut out = String::new();
            for i in 0..e.wr.pos { let b = (e.wr.buf[i / 8] >> (7 - i % 8)) & 1; out.push(if b == 1 { '1' } else { '0' }); }
            if out != enc_bits {
                bad_enc += 1;
                per_op.entry(op).or_default().1 += 1;
                if shown < 400 && per_op[&op].1 == 1 {
                    shown += 1;
                    eprintln!("ENC MISMATCH op {op:#x}:\n  src  {bits}\n  want {enc_bits}\n  got  {out}\n  rec  {}", rec.iter().map(|b| format!("{b:02x}")).collect::<String>());
                }
            }
        }
        eprintln!("per-op (decode,encode) mismatches: {:?}; skipped as corrupt-memory vectors: {corrupt} (tolerated: {noncanon})", per_op);
        (n, bad_dec, bad_enc)
    }

    #[test]
    fn engine_condition_vectors() {
        let (n, d, e) = run("vec_cond.json", condition, 16, 5);
        eprintln!("condition vectors: {n} decode-mismatch {d} encode-mismatch {e}");
        assert_eq!((d, e), (0, 0));
    }

    #[test]
    fn engine_action_vectors() {
        let (n, d, e) = run("vec_act.json", action, 20, 7);
        eprintln!("action vectors: {n} decode-mismatch {d} encode-mismatch {e}");
        assert_eq!((d, e), (0, 0));
    }
}

#[cfg(test)]
mod real_file_tests {
    use super::*;

    /// Every `*.bin` under the installed MCC `haloreach/{game_variants,hopper_game_variants}`
    /// folders (plus any in `HMS_GV_DIRS`, `:`-separated).
    fn real_files() -> Vec<std::path::PathBuf> {
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        for root in crate::mapcat::mcc_install_roots() {
            for sub in ["game_variants", "hopper_game_variants"] {
                dirs.push(root.join("haloreach").join(sub));
            }
        }
        if let Ok(extra) = std::env::var("HMS_GV_DIRS") {
            dirs.extend(extra.split(':').filter(|d| !d.is_empty()).map(std::path::PathBuf::from));
        }
        let mut out = Vec::new();
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("bin")) {
                    if let Ok(h) = std::fs::read(&p).map(|d| d.get(..4) == Some(b"_blf") && d.len() > 0x2F8 && &d[0x2F0..0x2F4] == b"mpvr") {
                        if h { out.push(p); }
                    }
                }
            }
        }
        out
    }

    /// THE gate: every real Reach gametype on this machine decodes, and re-encodes to the very same
    /// bitstream, except for the three documented ways a source file can be non-canonical. Any
    /// other difference fails the test and names the first differing bit.
    #[test]
    #[ignore]
    fn real_gametypes_round_trip_with_only_documented_differences() {
        let files = real_files();
        if files.is_empty() {
            eprintln!("gamevariant: no installed Reach gametypes found -- real-file gate skipped");
            return;
        }
        let (mut exact, mut pv, mut fc, mut ff, mut refused, mut le_files, mut good_hash) = (0, 0, 0, 0, 0, 0, 0);
        let mut other: Vec<String> = Vec::new();
        for p in &files {
            let d = std::fs::read(p).unwrap();
            let q = MPVR_AT;
            let lenb = [d[q + 36], d[q + 37], d[q + 38], d[q + 39]];
            // the engine's own rule: big-endian length, else native (little-endian)
            let (be, le) = (u32::from_be_bytes(lenb) as usize, u32::from_le_bytes(lenb) as usize);
            let len = if be <= MPVR_CAPACITY { be } else if le <= MPVR_CAPACITY { le_files += 1; le } else { refused += 1; continue; };
            if mpvr_hash(lenb, &d[q + 40..q + 40 + len]) == d[q + 12..q + 32] { good_hash += 1; }
            match classify_roundtrip(&d[q + 40..q + 40 + len]) {
                Ok(RoundTrip::Exact) => exact += 1,
                Ok(RoundTrip::ProgramVersion) => pv += 1,
                Ok(RoundTrip::FloatCode) => fc += 1,
                Ok(RoundTrip::FirefightUpgrade) => ff += 1,
                Ok(RoundTrip::Other(bit)) => other.push(format!("{}: first differing bit {bit}", p.file_name().unwrap().to_string_lossy())),
                Err(_) => refused += 1, // engine itself would refuse (truncated / corrupt / out of capacity)
            }
        }
        eprintln!(
            "real gametypes: {} files: {exact} byte-exact, {pv} differ only in the program-version dword, {fc} only in the 0xF858 float code, {ff} Firefight v1->v2 upgrades, {} unexplained, {refused} refused as corrupt or truncated; {le_files} store the length little-endian (the engine accepts either); {good_hash} carry a SHA-1 our formula reproduces",
            files.len(), other.len()
        );
        assert!(other.is_empty(), "unexplained round-trip differences: {other:?}");
        assert!(exact > 0 || pv + fc + ff > 0, "nothing decoded at all");
        // the SHA-1 formula (length bytes as stored + the stream) must hold for the game's own files
        let decoded = files.len() - refused;
        assert!(good_hash * 10 >= decoded * 8, "the chunk-hash formula reproduces only {good_hash} of {decoded} stored hashes");
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;

    fn scratch() -> Option<String> {
        std::env::var("HMS_GV_SCRATCH").ok()
    }

    /// (path, md5) pairs of the scratch corpus.
    fn corpus() -> Vec<(String, String)> {
        let Some(dir) = scratch() else { return vec![] };
        let t = std::fs::read_to_string(format!("{dir}/corpus54.tsv")).unwrap_or_default();
        t.lines().filter_map(|l| l.split_once('\t')).map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    fn payload_of(d: &[u8]) -> &[u8] {
        let q = 0x2f0;
        let be = u32::from_be_bytes([d[q + 36], d[q + 37], d[q + 38], d[q + 39]]) as usize;
        let le = u32::from_le_bytes([d[q + 36], d[q + 37], d[q + 38], d[q + 39]]) as usize;
        let len = if be <= MPVR_CAPACITY { be } else { le.min(MPVR_CAPACITY) };
        &d[q + 40..q + 40 + len]
    }

    /// Decode every corpus file with the Rust codec; compare against the engine's own decode
    /// (image bytes), against the engine's own ENCODE of that image, and against the source file.
    #[test]
    #[ignore]
    fn decode_and_roundtrip_against_engine() {
        let dir = scratch().expect("HMS_GV_SCRATCH");
        let (mut n, mut dec_fail, mut img_diff, mut eng_ok, mut eng_bad, mut src_ok, mut src_bad) = (0, 0, 0, 0, 0, 0, 0);
        let only = std::env::var("HMS_GV_ONLY").ok();
        let (mut shown_a, mut shown_b, mut shown_c) = (0, 0, 0);
        for (p, md5) in corpus() {
            if let Some(o) = &only { if !p.contains(o.as_str()) { continue; } }
            let d = std::fs::read(&p).unwrap();
            let payload = payload_of(&d);
            let eng_img = std::fs::read(format!("{dir}/eng/{md5}.img")).unwrap();
            let engine = u32::from_le_bytes([eng_img[0], eng_img[1], eng_img[2], eng_img[3]]);
            n += 1;
            let mut c = Codec::decoder(payload, IMAGE_LEN);
            c.trace = Some(Vec::new());
            let r = decode_into(&mut c);
            let name = std::path::Path::new(&p).file_name().unwrap().to_string_lossy().to_string();
            if let Err(e) = r {
                dec_fail += 1;
                if shown_a < 12 { shown_a += 1; eprintln!("DECODE FAIL {name}: {e} (at bit {})", c.rd.pos); }
                continue;
            }
            let mut diffs: Vec<usize> = (0..IMAGE_LEN).filter(|&i| c.written[i] && c.img[i] != eng_img[i]).collect();
            let used = c.rd.pos;
            let want_bits: usize = std::fs::read_to_string(format!("{dir}/eng/{md5}.ev")).unwrap().lines().next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
            if !diffs.is_empty() || used != want_bits {
                img_diff += 1;
                if shown_b < 8 {
                    shown_b += 1;
                    diffs.truncate(12);
                    eprintln!("IMAGE DIFF {name}: used {used} bits (engine {want_bits}); first diffs {:x?}", diffs);
                }
            }
            let (bytes, bits) = match encode_image(&c.img) {
                Ok(x) => x,
                Err(e) => { eng_bad += 1; src_bad += 1; eprintln!("ENCODE FAIL {name}: {e}"); continue; }
            };
            let nbytes = (bits + 7) / 8;
            // against the engine's own encoder
            let enc = std::fs::read(format!("{dir}/eng/{md5}.enc")).unwrap_or_default();
            if enc.len() > 4 {
                let ebits = u32::from_le_bytes([enc[0], enc[1], enc[2], enc[3]]) as usize;
                if ebits == bits && enc[4..4 + nbytes] == bytes[..nbytes] { eng_ok += 1; } else {
                    eng_bad += 1;
                    if shown_c < 10 {
                        shown_c += 1;
                        let first = (0..nbytes.min(enc.len() - 4)).find(|&i| bytes[i] != enc[4 + i]);
                        eprintln!("ENGINE-ENCODE DIFF {name}: mine {bits} bits, engine {ebits}; first differing byte {:?}", first);
                    }
                }
            }
            // against the file
            if bits == used && bytes[..nbytes] == payload[..nbytes.min(payload.len())] { src_ok += 1; } else {
                src_bad += 1;
                let first = (0..bytes.len().min(payload.len())).find(|&i| bytes[i] != payload[i]);
                if let Some(fb) = first {
                    let bitpos = (0..8).map(|k| fb * 8 + k).find(|&b| ((bytes[b / 8] ^ payload[b / 8]) >> (7 - b % 8)) & 1 != 0).unwrap();
                    let mut acc = 0usize;
                    for (w, v, line) in c.trace.as_ref().unwrap() {
                        if acc + *w as usize > bitpos {
                            eprintln!("  source differs: {name}: bit {bitpos} in the field at gamevariant.rs:{line} (width {w}, source value {v:#x})");
                            break;
                        }
                        acc += *w as usize;
                    }
                }
            }
        }
        eprintln!("corpus: {n} files; decode failures {dec_fail}; image/bit-count differences vs engine decode {img_diff}; encode == engine encode {eng_ok} (differ {eng_bad}); encode == source file {src_ok} (differ {src_bad})");
    }

    /// Oracle (b): a film embeds the variant it was recorded with. For each real film whose
    /// gametype also exists on disk (matched by title + author + CONTENT, never by the ambiguous
    /// lineage uid), the bitstream we encode from the film's struct must equal the one the game
    /// saved to disk.
    #[test]
    #[ignore]
    fn film_struct_encodes_to_the_gametype_on_disk() {
        use std::collections::HashMap;
        // disk gametypes: (title, author) -> list of (file name, canonical encoding of its decode, raw payload)
        let mut disk: HashMap<(String, String), Vec<(String, Vec<u8>, Vec<u8>)>> = HashMap::new();
        let mut all: Vec<(String, Vec<u8>)> = Vec::new();
        for (p, _) in corpus() {
            let Ok(d) = std::fs::read(&p) else { continue };
            let Ok((img, _)) = parse_bin(&d) else { continue };
            if u32::from_le_bytes([img[0], img[1], img[2], img[3]]) == 4 { continue; }
            let Ok((enc, bits)) = encode_image(&img) else { continue };
            let enc = enc[..(bits + 7) / 8].to_vec();
            let (t, _, a, _) = titles(&img);
            let name = std::path::Path::new(&p).file_name().unwrap().to_string_lossy().to_string();
            disk.entry((t, a)).or_default().push((name.clone(), enc.clone(), payload_of(&d).to_vec()));
            all.push((name, enc));
        }
        let films = crate::film::spec_corpus();
        let (mut nf, mut exact_canon, mut exact_raw, mut by_content_only, mut title_only, mut none, mut skipped) = (0, 0, 0, 0, 0, 0, 0);
        for fp in films {
            let Ok(f) = crate::film::open(&fp) else { continue };
            let Some(raw) = crate::film_export::game_variant_bytes(&f) else { continue };
            let engine = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            if engine == 4 { skipped += 1; continue; }
            nf += 1;
            let Ok((enc, bits)) = encode_image(raw) else { eprintln!("film encode failed: {}", f.header.game_variant.title); none += 1; continue; };
            let enc = enc[..(bits + 7) / 8].to_vec();
            let (t, _, a, _) = titles(raw);
            let cands = disk.get(&(t.clone(), a.clone())).cloned().unwrap_or_default();
            if let Some((name, _, rawp)) = cands.iter().find(|(_, e, _)| *e == enc) {
                exact_canon += 1;
                if rawp[..] == enc[..] { exact_raw += 1; }
                let _ = name;
            } else if all.iter().any(|(_, e)| *e == enc) {
                by_content_only += 1;
            } else if !cands.is_empty() {
                title_only += 1;
                if title_only <= 40 {
                    // where does the closest candidate first differ?
                    let (name, e, _) = &cands[0];
                    let first = (0..enc.len().min(e.len())).find(|&i| enc[i] != e[i]);
                    eprintln!("film {:?} by {:?}: title+author found on disk ({name}) but the content differs; first differing byte {:?} (film {} B, disk {} B)", t, a, first, enc.len(), e.len());
                    if let Some(fb) = first {
                        let bitpos = (0..8).map(|k| fb * 8 + k).find(|&b| ((enc[b / 8] ^ e[b / 8]) >> (7 - b % 8)) & 1 != 0).unwrap();
                        let mut c = Codec::encoder(raw[..IMAGE_LEN].to_vec());
                        c.trace = Some(Vec::new());
                        let _ = c.io(4, engine as u64);
                        variant_body(&mut c, engine);
                        let mut acc = 0usize;
                        for (w, v, line) in c.trace.as_ref().unwrap() {
                            if acc + *w as usize > bitpos {
                                eprintln!("    first differing bit {bitpos}: field at gamevariant.rs:{line} (width {w}, film value {v:#x})");
                                break;
                            }
                            acc += *w as usize;
                        }
                    }
                }
            } else {
                none += 1;
                eprintln!("film {:?} by {:?}: no gametype of that title+author on disk", t, a);
            }
        }
        eprintln!("films: {nf} (+{skipped} Firefight skipped); bitstream == a disk gametype of the same title+author: {exact_canon} (byte-identical to the file's own payload: {exact_raw}); same content under another name: {by_content_only}; title+author on disk but different content: {title_only}; not on disk: {none}");
    }

    /// Write the `.bin` we build from every film's struct (and from a sample of disk files) into
    /// `$HMS_GV_SCRATCH/built/`, with the source image beside it, so the engine's own file loader
    /// can be run over them in the emulation harness.
    #[test]
    #[ignore]
    fn dump_built_bins_for_the_engine_loader() {
        let dir = format!("{}/built2", scratch().expect("HMS_GV_SCRATCH"));
        let _ = std::fs::create_dir_all(&dir);
        let mut n = 0;
        for (i, fp) in crate::film::spec_corpus().iter().enumerate() {
            let Ok(f) = crate::film::open(fp) else { continue };
            let Some(raw) = crate::film_export::game_variant_bytes(&f) else { continue };
            let Ok(bin) = build_bin(raw, CHDR_TAG_UNSET) else { continue };
            std::fs::write(format!("{dir}/film{i:03}.bin"), &bin).unwrap();
            std::fs::write(format!("{dir}/film{i:03}.src"), raw).unwrap();
            n += 1;
        }
        for (i, (p, _)) in corpus().iter().enumerate().step_by(5) {
            let Ok(d) = std::fs::read(p) else { continue };
            let Ok((img, info)) = parse_bin(&d) else { continue };
            let Ok(bin) = build_bin(&img, info.chdr_tag) else { continue };
            std::fs::write(format!("{dir}/disk{i:03}.bin"), &bin).unwrap();
            std::fs::write(format!("{dir}/disk{i:03}.src"), &img).unwrap();
            n += 1;
        }
        eprintln!("wrote {n} built files to {dir}");
    }
}
