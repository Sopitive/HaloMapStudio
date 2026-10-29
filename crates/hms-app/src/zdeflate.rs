//! zlib "deflate" as the Halo: Reach engine writes it: madler/zlib 1.2.x, level 9, `windowBits` 15,
//! `memLevel` 8, default strategy, one `Z_FINISH` call with all the input available.
//!
//! The engine compresses the string blobs of a game variant this way (`haloreach.dll` `0x778E94`
//! wraps `deflateInit_`; the init code sets `w_bits = 15`, `hash_bits = 15`, `lit_bufsize =
//! 0x4000`, level 9). Any zlib reader inflates *any* valid stream, so a different compressor
//! would still load -- but then a re-encoded gametype could never be byte-identical to the one the
//! game wrote. This is a from-the-algorithm port of `deflate_slow` + `trees.c` (written from the
//! documented zlib algorithm, not from another Rust implementation) and is checked against the
//! reference C library over thousands of inputs (`tests` below, and `gamevariant`'s corpus gate).
//!
//! Inputs above 60,000 bytes are refused (the window would slide; a game variant string blob is at
//! most 19,456 bytes).

const MAX_MATCH: usize = 258;
const MIN_MATCH: usize = 3;
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const W_SIZE: usize = 32768;
const W_MASK: usize = W_SIZE - 1;
const MAX_DIST: usize = W_SIZE - MIN_LOOKAHEAD;
const HASH_BITS: u32 = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const HASH_SHIFT: u32 = (HASH_BITS + MIN_MATCH as u32 - 1) / MIN_MATCH as u32;
const LIT_BUFSIZE: usize = 1 << 14;
const TOO_FAR: usize = 4096;
// level 9 configuration_table entry
const GOOD_MATCH: usize = 32;
const MAX_LAZY: usize = 258;
const NICE_MATCH: usize = 258;
const MAX_CHAIN: usize = 4096;

const LENGTH_CODES: usize = 29;
const LITERALS: usize = 256;
const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;

const EXTRA_LBITS: [u8; LENGTH_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS: [u8; D_CODES] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// `ct_data`: `fc` is freq-or-code, `dl` is dad-or-len (zlib overlays them; the order in which
/// they are overwritten matters, so they are overlaid here too).
#[derive(Clone, Copy, Default)]
struct Ct {
    fc: u16,
    dl: u16,
}

fn bi_reverse(mut code: u32, len: u32) -> u32 {
    let mut res = 0u32;
    for _ in 0..len {
        res |= code & 1;
        code >>= 1;
        res <<= 1;
    }
    res >> 1
}

fn gen_codes(tree: &mut [Ct], max_code: usize, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code = 0u32;
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1] as u32) << 1;
        next_code[bits] = code as u16;
    }
    for n in 0..=max_code {
        let len = tree[n].dl as usize;
        if len == 0 {
            continue;
        }
        tree[n].fc = bi_reverse(next_code[len] as u32, len as u32) as u16;
        next_code[len] = next_code[len].wrapping_add(1);
    }
}

struct Statics {
    ltree: [Ct; L_CODES + 2],
    dtree: [Ct; D_CODES],
    dist_code: [u8; 512],
    length_code: [u8; 256],
    base_length: [u16; LENGTH_CODES],
    base_dist: [u16; D_CODES],
}

fn statics() -> &'static Statics {
    use std::sync::OnceLock;
    static S: OnceLock<Statics> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = Statics {
            ltree: [Ct::default(); L_CODES + 2],
            dtree: [Ct::default(); D_CODES],
            dist_code: [0; 512],
            length_code: [0; 256],
            base_length: [0; LENGTH_CODES],
            base_dist: [0; D_CODES],
        };
        let mut length = 0usize;
        let mut code = 0usize;
        while code < LENGTH_CODES - 1 {
            s.base_length[code] = length as u16;
            for _ in 0..(1 << EXTRA_LBITS[code]) {
                s.length_code[length] = code as u8;
                length += 1;
            }
            code += 1;
        }
        s.length_code[length - 1] = code as u8;
        let mut dist = 0usize;
        code = 0;
        while code < 16 {
            s.base_dist[code] = dist as u16;
            for _ in 0..(1 << EXTRA_DBITS[code]) {
                s.dist_code[dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        dist >>= 7;
        while code < D_CODES {
            s.base_dist[code] = (dist << 7) as u16;
            for _ in 0..(1 << (EXTRA_DBITS[code] - 7)) {
                s.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        let mut bl_count = [0u16; MAX_BITS + 1];
        let mut n = 0;
        while n <= 143 { s.ltree[n].dl = 8; n += 1; bl_count[8] += 1; }
        while n <= 255 { s.ltree[n].dl = 9; n += 1; bl_count[9] += 1; }
        while n <= 279 { s.ltree[n].dl = 7; n += 1; bl_count[7] += 1; }
        while n <= 287 { s.ltree[n].dl = 8; n += 1; bl_count[8] += 1; }
        gen_codes(&mut s.ltree, L_CODES + 1, &bl_count);
        for n in 0..D_CODES {
            s.dtree[n].dl = 5;
            s.dtree[n].fc = bi_reverse(n as u32, 5) as u16;
        }
        s
    })
}

#[derive(Clone, Copy)]
enum Which { L, D, Bl }

struct Desc {
    max_code: i32,
}

struct Def {
    st: &'static Statics,
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    ins_h: usize,
    strstart: usize,
    block_start: i64,
    lookahead: usize,
    prev_length: usize,
    match_length: usize,
    match_start: usize,
    prev_match: usize,
    match_available: bool,
    // input
    input: Vec<u8>,
    in_pos: usize,
    // trees
    dyn_l: [Ct; HEAP_SIZE],
    dyn_d: [Ct; 2 * D_CODES + 1],
    bl: [Ct; 2 * BL_CODES + 1],
    l_desc: Desc,
    d_desc: Desc,
    bl_desc: Desc,
    bl_count: [u16; MAX_BITS + 1],
    heap: [i32; 2 * L_CODES + 1],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; 2 * L_CODES + 1],
    l_buf: Vec<u8>,
    d_buf: Vec<u16>,
    last_lit: usize,
    opt_len: u64,
    static_len: u64,
    // output
    out: Vec<u8>,
    bi_buf: u64,
    bi_valid: u32,
}

impl Def {
    fn new(input: &[u8]) -> Def {
        Def {
            st: statics(),
            window: vec![0; 2 * W_SIZE + MAX_MATCH + 8],
            prev: vec![0; W_SIZE],
            head: vec![0; HASH_SIZE],
            ins_h: 0,
            strstart: 0,
            block_start: 0,
            lookahead: 0,
            prev_length: MIN_MATCH - 1,
            match_length: MIN_MATCH - 1,
            match_start: 0,
            prev_match: 0,
            match_available: false,
            input: input.to_vec(),
            in_pos: 0,
            dyn_l: [Ct::default(); HEAP_SIZE],
            dyn_d: [Ct::default(); 2 * D_CODES + 1],
            bl: [Ct::default(); 2 * BL_CODES + 1],
            l_desc: Desc { max_code: 0 },
            d_desc: Desc { max_code: 0 },
            bl_desc: Desc { max_code: 0 },
            bl_count: [0; MAX_BITS + 1],
            heap: [0; 2 * L_CODES + 1],
            heap_len: 0,
            heap_max: 0,
            depth: [0; 2 * L_CODES + 1],
            l_buf: vec![0; LIT_BUFSIZE],
            d_buf: vec![0; LIT_BUFSIZE],
            last_lit: 0,
            opt_len: 0,
            static_len: 0,
            out: Vec::new(),
            bi_buf: 0,
            bi_valid: 0,
        }
    }

    // ---- bit output (LSB first, like zlib's send_bits) ----
    fn send_bits(&mut self, value: u32, length: u32) {
        self.bi_buf |= (value as u64) << self.bi_valid;
        self.bi_valid += length;
        while self.bi_valid >= 8 {
            self.out.push(self.bi_buf as u8);
            self.bi_buf >>= 8;
            self.bi_valid -= 8;
        }
    }
    fn bi_windup(&mut self) {
        if self.bi_valid > 0 {
            self.out.push(self.bi_buf as u8);
        }
        self.bi_buf = 0;
        self.bi_valid = 0;
    }

    fn tree(&mut self, w: Which) -> &mut [Ct] {
        match w {
            Which::L => &mut self.dyn_l,
            Which::D => &mut self.dyn_d,
            Which::Bl => &mut self.bl,
        }
    }

    fn init_block(&mut self) {
        for n in 0..L_CODES { self.dyn_l[n].fc = 0; }
        for n in 0..D_CODES { self.dyn_d[n].fc = 0; }
        for n in 0..BL_CODES { self.bl[n].fc = 0; }
        self.dyn_l[END_BLOCK].fc = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.last_lit = 0;
    }

    fn smaller(tree: &[Ct], n: usize, m: usize, depth: &[u8]) -> bool {
        tree[n].fc < tree[m].fc || (tree[n].fc == tree[m].fc && depth[n] <= depth[m])
    }

    fn pqdownheap(&mut self, w: Which, mut k: usize) {
        let v = self.heap[k] as usize;
        let mut j = k << 1;
        while j <= self.heap_len {
            let tree: &[Ct] = match w { Which::L => &self.dyn_l, Which::D => &self.dyn_d, Which::Bl => &self.bl };
            if j < self.heap_len && Self::smaller(tree, self.heap[j + 1] as usize, self.heap[j] as usize, &self.depth) {
                j += 1;
            }
            if Self::smaller(tree, v, self.heap[j] as usize, &self.depth) {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v as i32;
    }

    /// (static tree, extra bits, extra base, elems, max length) of a descriptor.
    fn desc_info(&self, w: Which) -> (Option<&'static [Ct]>, &'static [u8], usize, usize, usize) {
        match w {
            Which::L => (Some(&self.st.ltree[..]), &EXTRA_LBITS[..], LITERALS + 1, L_CODES, MAX_BITS),
            Which::D => (Some(&self.st.dtree[..]), &EXTRA_DBITS[..], 0, D_CODES, MAX_BITS),
            Which::Bl => (None, &EXTRA_BLBITS[..], 0, BL_CODES, MAX_BL_BITS),
        }
    }

    fn build_tree(&mut self, w: Which) {
        let (stree, _extra, _base, elems, _maxlen) = self.desc_info(w);
        let _ = stree;
        let mut max_code: i32 = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..elems {
            if self.tree(w)[n].fc != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as i32;
                max_code = n as i32;
                self.depth[n] = 0;
            } else {
                self.tree(w)[n].dl = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node as i32;
            self.tree(w)[node].fc = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(st) = stree {
                self.static_len = self.static_len.wrapping_sub(st[node].dl as u64);
            }
        }
        match w {
            Which::L => self.l_desc.max_code = max_code,
            Which::D => self.d_desc.max_code = max_code,
            Which::Bl => self.bl_desc.max_code = max_code,
        }
        let mut n = self.heap_len / 2;
        while n >= 1 {
            self.pqdownheap(w, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            // pqremove
            let nn = self.heap[1] as usize;
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(w, 1);
            let m = self.heap[1] as usize;
            self.heap_max -= 1;
            self.heap[self.heap_max] = nn as i32;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as i32;
            let f = self.tree(w)[nn].fc.wrapping_add(self.tree(w)[m].fc);
            self.tree(w)[node].fc = f;
            self.depth[node] = self.depth[nn].max(self.depth[m]) + 1;
            self.tree(w)[nn].dl = node as u16;
            self.tree(w)[m].dl = node as u16;
            self.heap[1] = node as i32;
            node += 1;
            self.pqdownheap(w, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(w, max_code);
        let bl_count = self.bl_count;
        gen_codes(self.tree(w), max_code as usize, &bl_count);
    }

    fn gen_bitlen(&mut self, w: Which, max_code: i32) {
        let (stree, extra, base, _elems, max_length) = self.desc_info(w);
        let mut overflow = 0i32;
        for b in 0..=MAX_BITS { self.bl_count[b] = 0; }
        let root = self.heap[self.heap_max] as usize;
        self.tree(w)[root].dl = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h] as usize;
            let dad = self.tree(w)[n].dl as usize;
            let mut bits = self.tree(w)[dad].dl as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            self.tree(w)[n].dl = bits as u16;
            if n as i32 > max_code {
                h += 1;
                continue;
            }
            self.bl_count[bits] += 1;
            let mut xbits = 0usize;
            if n >= base {
                xbits = extra[n - base] as usize;
            }
            let f = self.tree(w)[n].fc as u64;
            self.opt_len = self.opt_len.wrapping_add(f * (bits + xbits) as u64);
            if let Some(st) = stree {
                self.static_len = self.static_len.wrapping_add(f * (st[n].dl as usize + xbits) as u64);
            }
            h += 1;
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        let mut bits = max_length;
        while bits != 0 {
            let mut n = self.bl_count[bits] as i32;
            while n != 0 {
                h -= 1;
                let m = self.heap[h] as usize;
                if m as i32 > max_code {
                    continue;
                }
                if self.tree(w)[m].dl as usize != bits {
                    let f = self.tree(w)[m].fc as u64;
                    let old = self.tree(w)[m].dl as u64;
                    self.opt_len = self.opt_len.wrapping_add((bits as u64).wrapping_sub(old).wrapping_mul(f));
                    self.tree(w)[m].dl = bits as u16;
                }
                n -= 1;
            }
            bits -= 1;
        }
    }

    fn scan_tree(&mut self, w: Which, max_code: i32) {
        let mut prevlen: i32 = -1;
        let mut nextlen = self.tree(w)[0].dl as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        self.tree(w)[(max_code + 1) as usize].dl = 0xffff;
        for n in 0..=max_code as usize {
            let curlen = nextlen;
            nextlen = self.tree(w)[n + 1].dl as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                self.bl[curlen as usize].fc = self.bl[curlen as usize].fc.wrapping_add(count as u16);
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.bl[curlen as usize].fc += 1;
                }
                self.bl[REP_3_6].fc += 1;
            } else if count <= 10 {
                self.bl[REPZ_3_10].fc += 1;
            } else {
                self.bl[REPZ_11_138].fc += 1;
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn send_code_bl(&mut self, c: usize) {
        let t = self.bl[c];
        self.send_bits(t.fc as u32, t.dl as u32);
    }

    fn send_tree(&mut self, w: Which, max_code: i32) {
        let mut prevlen: i32 = -1;
        let mut nextlen = self.tree(w)[0].dl as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        for n in 0..=max_code as usize {
            let curlen = nextlen;
            nextlen = self.tree(w)[n + 1].dl as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                loop {
                    self.send_code_bl(curlen as usize);
                    count -= 1;
                    if count == 0 { break; }
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.send_code_bl(curlen as usize);
                    count -= 1;
                }
                self.send_code_bl(REP_3_6);
                self.send_bits((count - 3) as u32, 2);
            } else if count <= 10 {
                self.send_code_bl(REPZ_3_10);
                self.send_bits((count - 3) as u32, 3);
            } else {
                self.send_code_bl(REPZ_11_138);
                self.send_bits((count - 11) as u32, 7);
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn build_bl_tree(&mut self) -> usize {
        let lm = self.l_desc.max_code;
        let dm = self.d_desc.max_code;
        self.scan_tree(Which::L, lm);
        self.scan_tree(Which::D, dm);
        self.build_tree(Which::Bl);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bl[BL_ORDER[max_blindex]].dl != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.opt_len = self.opt_len.wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        max_blindex
    }

    fn send_all_trees(&mut self, lcodes: usize, dcodes: usize, blcodes: usize) {
        self.send_bits((lcodes - 257) as u32, 5);
        self.send_bits((dcodes - 1) as u32, 5);
        self.send_bits((blcodes - 4) as u32, 4);
        for rank in 0..blcodes {
            let l = self.bl[BL_ORDER[rank]].dl as u32;
            self.send_bits(l, 3);
        }
        self.send_tree(Which::L, lcodes as i32 - 1);
        self.send_tree(Which::D, dcodes as i32 - 1);
    }

    fn d_code(&self, dist: usize) -> usize {
        if dist < 256 { self.st.dist_code[dist] as usize } else { self.st.dist_code[256 + (dist >> 7)] as usize }
    }

    fn compress_block(&mut self, stat: bool) {
        let mut lx = 0usize;
        if self.last_lit != 0 {
            loop {
                let dist = self.d_buf[lx] as usize;
                let mut lc = self.l_buf[lx] as usize;
                lx += 1;
                if dist == 0 {
                    let t = if stat { self.st.ltree[lc] } else { self.dyn_l[lc] };
                    self.send_bits(t.fc as u32, t.dl as u32);
                } else {
                    let code = self.st.length_code[lc] as usize;
                    let t = if stat { self.st.ltree[code + LITERALS + 1] } else { self.dyn_l[code + LITERALS + 1] };
                    self.send_bits(t.fc as u32, t.dl as u32);
                    let extra = EXTRA_LBITS[code] as u32;
                    if extra != 0 {
                        lc -= self.st.base_length[code] as usize;
                        self.send_bits(lc as u32, extra);
                    }
                    let mut d = dist - 1;
                    let code = self.d_code(d);
                    let t = if stat { self.st.dtree[code] } else { self.dyn_d[code] };
                    self.send_bits(t.fc as u32, t.dl as u32);
                    let extra = EXTRA_DBITS[code] as u32;
                    if extra != 0 {
                        d -= self.st.base_dist[code] as usize;
                        self.send_bits(d as u32, extra);
                    }
                }
                if lx >= self.last_lit {
                    break;
                }
            }
        }
        let t = if stat { self.st.ltree[END_BLOCK] } else { self.dyn_l[END_BLOCK] };
        self.send_bits(t.fc as u32, t.dl as u32);
    }

    fn tr_stored_block(&mut self, start: usize, len: usize, last: bool) {
        self.send_bits(last as u32, 3);
        self.bi_windup();
        self.out.push(len as u8);
        self.out.push((len >> 8) as u8);
        self.out.push(!len as u8);
        self.out.push((!len >> 8) as u8);
        let data = self.window[start..start + len].to_vec();
        self.out.extend_from_slice(&data);
    }

    fn tr_flush_block(&mut self, buf_start: Option<usize>, stored_len: usize, last: bool) {
        self.build_tree(Which::L);
        self.build_tree(Which::D);
        let max_blindex = self.build_bl_tree();
        let mut opt_lenb = (self.opt_len + 3 + 7) >> 3;
        let static_lenb = (self.static_len + 3 + 7) >> 3;
        if static_lenb <= opt_lenb {
            opt_lenb = static_lenb;
        }
        if stored_len as u64 + 4 <= opt_lenb && buf_start.is_some() {
            self.tr_stored_block(buf_start.unwrap(), stored_len, last);
        } else if static_lenb == opt_lenb {
            self.send_bits((1 << 1) + last as u32, 3);
            self.compress_block(true);
        } else {
            self.send_bits((2 << 1) + last as u32, 3);
            let (lm, dm) = (self.l_desc.max_code, self.d_desc.max_code);
            self.send_all_trees((lm + 1) as usize, (dm + 1) as usize, max_blindex + 1);
            self.compress_block(false);
        }
        self.init_block();
        if last {
            self.bi_windup();
        }
    }

    fn tally_lit(&mut self, c: u8) -> bool {
        self.d_buf[self.last_lit] = 0;
        self.l_buf[self.last_lit] = c;
        self.last_lit += 1;
        self.dyn_l[c as usize].fc += 1;
        self.last_lit == LIT_BUFSIZE - 1
    }
    fn tally_dist(&mut self, dist: usize, lc: usize) -> bool {
        self.d_buf[self.last_lit] = dist as u16;
        self.l_buf[self.last_lit] = lc as u8;
        self.last_lit += 1;
        let d = dist - 1;
        self.dyn_l[self.st.length_code[lc] as usize + LITERALS + 1].fc += 1;
        let dc = self.d_code(d);
        self.dyn_d[dc].fc += 1;
        self.last_lit == LIT_BUFSIZE - 1
    }

    fn flush_block(&mut self, last: bool) {
        let start = if self.block_start >= 0 { Some(self.block_start as usize) } else { None };
        let len = (self.strstart as i64 - self.block_start) as usize;
        self.tr_flush_block(start, len, last);
        self.block_start = self.strstart as i64;
    }

    // ---- the matcher ----
    fn fill_window(&mut self) {
        // everything fits: no slide. Read all the input on the first call.
        let more = 2 * W_SIZE - self.lookahead - self.strstart;
        let avail = self.input.len() - self.in_pos;
        let n = avail.min(more);
        if n > 0 {
            let (a, b) = (self.strstart + self.lookahead, self.in_pos);
            self.window[a..a + n].copy_from_slice(&self.input[b..b + n]);
            self.in_pos += n;
            self.lookahead += n;
        }
        if self.lookahead >= MIN_MATCH {
            // insert == 0 on the only fill we do
            if self.strstart == 0 {
                self.ins_h = self.window[0] as usize;
                self.ins_h = ((self.ins_h << HASH_SHIFT) ^ self.window[1] as usize) & HASH_MASK;
            }
        }
    }

    fn insert_string(&mut self, s: usize) -> usize {
        self.ins_h = ((self.ins_h << HASH_SHIFT) ^ self.window[s + MIN_MATCH - 1] as usize) & HASH_MASK;
        let head = self.head[self.ins_h] as usize;
        self.prev[s & W_MASK] = head as u16;
        self.head[self.ins_h] = s as u16;
        head
    }

    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = MAX_CHAIN;
        let scan = self.strstart;
        let mut best_len = self.prev_length as usize;
        let mut nice_match = NICE_MATCH;
        let limit = if self.strstart > MAX_DIST { self.strstart - MAX_DIST } else { 0 };
        let w = &self.window;
        let mut scan_end1 = w[scan + best_len - 1];
        let mut scan_end = w[scan + best_len];
        if self.prev_length >= GOOD_MATCH {
            chain_length >>= 2;
        }
        if nice_match > self.lookahead {
            nice_match = self.lookahead;
        }
        loop {
            let m = cur_match;
            if !(w[m + best_len] != scan_end || w[m + best_len - 1] != scan_end1 || w[m] != w[scan] || w[m + 1] != w[scan + 1]) {
                // scan[2] == match[2] is assumed (equal hash keys); compare from index 3
                let mut p = 3usize;
                while p < MAX_MATCH && w[scan + p] == w[m + p] {
                    p += 1;
                }
                // the C loop also compares index 258 (strend) before it stops
                let len = if p >= MAX_MATCH { MAX_MATCH } else { p };
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match {
                        break;
                    }
                    scan_end1 = w[scan + best_len - 1];
                    scan_end = w[scan + best_len];
                }
            }
            cur_match = self.prev[cur_match & W_MASK] as usize;
            if cur_match <= limit {
                break;
            }
            chain_length -= 1;
            if chain_length == 0 {
                break;
            }
        }
        if best_len <= self.lookahead { best_len } else { self.lookahead }
    }

    fn deflate_slow(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0usize;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0 && self.prev_length < MAX_LAZY && self.strstart - hash_head <= MAX_DIST {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5 && (self.match_length == MIN_MATCH && self.strstart - self.match_start > TOO_FAR) {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let bflush = self.tally_dist(self.strstart - 1 - self.prev_match, self.prev_length - MIN_MATCH);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(false);
                }
            } else if self.match_available {
                let c = self.window[self.strstart - 1];
                if self.tally_lit(c) {
                    self.flush_block(false);
                }
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            let c = self.window[self.strstart - 1];
            self.tally_lit(c);
            self.match_available = false;
        }
        self.flush_block(true);
    }
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// zlib-compress `raw` exactly as the engine does; `None` if the stream would not fit in
/// `max_out` bytes (the engine then stores the data raw) or the input is too large.
pub fn compress9(raw: &[u8], max_out: usize) -> Option<Vec<u8>> {
    let z = compress9_unbounded(raw)?;
    if z.len() <= max_out { Some(z) } else { None }
}

/// The full zlib stream (header, deflate data, adler32), no size limit.
pub fn compress9_unbounded(raw: &[u8]) -> Option<Vec<u8>> {
    if raw.len() > 60_000 {
        return None;
    }
    let mut d = Def::new(raw);
    d.init_block();
    let mut out = vec![0x78, 0xDA];
    d.deflate_slow();
    out.extend_from_slice(&d.out);
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflates_back() {
        let mut s = Vec::new();
        for i in 0..1500u32 {
            s.extend_from_slice(format!("string {} of the table; ", i % 97).as_bytes());
        }
        let z = compress9_unbounded(&s).unwrap();
        let back = miniz_oxide::inflate::decompress_to_vec_zlib(&z).unwrap();
        assert_eq!(back, s);
        let tiny = compress9_unbounded(b"a").unwrap();
        assert_eq!(miniz_oxide::inflate::decompress_to_vec_zlib(&tiny).unwrap(), b"a");
        let empty = compress9_unbounded(b"").unwrap();
        assert_eq!(miniz_oxide::inflate::decompress_to_vec_zlib(&empty).unwrap(), b"");
    }

    /// Byte-identical to the reference zlib 1.2.11 (`HMS_GV_ZCASES`: a file of
    /// `u32 raw_len, u32 z_len, raw, z` records made with the C library).
    #[test]
    #[ignore]
    fn matches_reference_zlib() {
        let Ok(p) = std::env::var("HMS_GV_ZCASES") else { eprintln!("no HMS_GV_ZCASES"); return };
        let d = std::fs::read(p).unwrap();
        let (mut pos, mut n, mut bad) = (0usize, 0usize, 0usize);
        while pos + 8 <= d.len() {
            let rl = u32::from_le_bytes(d[pos..pos + 4].try_into().unwrap()) as usize;
            let zl = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let raw = &d[pos + 8..pos + 8 + rl];
            let z = &d[pos + 8 + rl..pos + 8 + rl + zl];
            pos += 8 + rl + zl;
            n += 1;
            if raw.len() > 60_000 { continue; } // refused by design (the window would slide)
            let mine = compress9_unbounded(raw).unwrap();
            if mine != z {
                bad += 1;
                if bad <= 5 {
                    let first = (0..mine.len().min(z.len())).find(|&i| mine[i] != z[i]);
                    eprintln!("MISMATCH case {n}: raw {rl} B, reference {zl} B, mine {} B, first diff at {:?}", mine.len(), first);
                }
            }
        }
        eprintln!("zlib reference: {n} cases, {bad} differ");
        assert_eq!(bad, 0);
    }
}
