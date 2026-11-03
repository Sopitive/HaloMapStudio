/*!
 * hms-mvar.js - read Halo map variants (.mvar) in a browser or in Node. Decode only.
 *
 * Copyright (c) 2026 Sopitive. All rights reserved.
 * Licensed under the PolyForm Noncommercial License 1.0.0, with a separate commercial licence
 * available. This file is a port of the map-variant decoders in Halo Map Studio's own Rust
 * source (crates/hms-app/src/mvar.rs, h3/mvar.rs, h4/mvar.rs) and is covered by the same
 * licence as the rest of Halo Map Studio. See docs/license.html.
 *
 * One file, no dependencies, no network, no eval, no build step.
 *
 * USE
 *   Browser, classic script (works from file:// and from a web server, no import needed):
 *                       <script src="hms-mvar.classic.js"></script>      -> window.HmsMvar
 *                       (hms-mvar.classic.js is GENERATED from this file by build-classic.mjs)
 *   Node, CommonJS:     const HmsMvar = require('./hms-mvar.classic.js');
 *   Browser (module):   <script type="module">
 *                         import HmsMvar from './js/hms-mvar.js';      // also sets window.HmsMvar
 *                         const v = HmsMvar.parse(await file.arrayBuffer());
 *                       </script>
 *   Browser (one line, no import in your own code):
 *                       <script type="module" src="js/hms-mvar.js"></script>   -> window.HmsMvar
 *   Node (ESM):         import HmsMvar from './hms-mvar.js';
 *                       const v = HmsMvar.parse(fs.readFileSync('forge_world.mvar'));
 *
 * API  (all plain objects, numbers and strings; 64-bit ids are 16-digit lower-case hex strings)
 *   HmsMvar.parse(arrayBuffer | Uint8Array | Buffer, opts?) -> parsed          (synchronous)
 *   HmsMvar.parseFile(blob | File)                          -> Promise<parsed> (reads it, then parse)
 *   HmsMvar.Error     class extends Error, .code is one of
 *                     'bad-input' 'not-mvar' 'truncated' 'unsupported' 'corrupt' 'too-large'
 *   HmsMvar.filterObjects(parsed, {paletteIndex, name, label, team, physics, objectType,
 *                         box:{min,max}, near:{x,y,z,radius}, predicate})  -> objects[]
 *   HmsMvar.countBy(objects, 'paletteIndex' | 'placement.physics' | fn)    -> {value: count}
 *   HmsMvar.toCSV(objects, {columns?, safe?})  -> string        (RFC 4180; text cells starting with
 *                         = + - @ get a leading ' unless safe:false)
 *   HmsMvar.toJSON(parsed, space?)             -> string
 *   HmsMvar.withPalette(parsed, paletteJson, {force?})  -> parsed   (attaches .name; see below)
 *   HmsMvar.version, HmsMvar.games = ['reach','halo3','halo4','h2a']
 *
 * parsed = {
 *   game: 'reach'|'halo3'|'halo4'|'h2a',
 *   blf:    { chunks:[{type,offset,size,major,minor}], mvarMajor, compressed, fileSize, hasFsm,
 *             chunkLength, hashKind: 'plain-le'|'salted-be-360'|null, hashOk },
 *   header: { title, description, author, editor, titleIsKey, created:{timestamp,date,xuid,online},
 *             modified:{...}, category, contentType, fileSize, uid, parentUid, rootUid, gameId,
 *             activity, gameMode, engine, hopperId, ... },
 *   map:    { mapId, headerMapId, mccMapId, paletteCrc, version, checksum, builtIn, builtFromXml,
 *             unassignedMapId },
 *   bounds: { xMin,xMax,yMin,yMax,zMin,zMax, min:[x,y,z], max:[x,y,z] },
 *   budget: { max, spent },
 *   quotas: [{ index, min, max, placed, name? ... }],      // index == an object's paletteIndex
 *   labels: [string],
 *   objects:[{ index, slot, paletteIndex, variantIndex, x, y, z, forward:[3], up:[3],
 *              yaw, pitch, roll (degrees, derived from forward/up), team, label, labelIndex, color,
 *              placement:{ raw, physics:'normal'|'fixed'|'phased'|'unknown'|null, symmetry,
 *                          placedAtStart, uniqueSpawn, ... },
 *              spawnSequence, respawnTime, objectType, shape:{type, raw, values}, parentSlot,
 *              flags, extra, name?, raw:{...exact stored fields...} }],
 *   warnings: [string]
 * }
 *
 * Names: the object palette is game data and is NOT embedded. An object's paletteIndex is a
 * position in the base map's palette. withPalette() attaches names from a table you supply:
 *   { "game": "reach", "mapId": 3006, "names": ["Category / Name", ...] }          one map
 *   { "maps": { "3006": { "game": "reach", "names": [...] }, ... } }                 several maps
 *   [ "name0", "name1", ... ]                                                       bare array
 * HMS prints the first form with the script verb  `variant get palette-json`  (see mvar-js.html).
 *
 * Limits: decode only (no encoder); Halo 4 / H2A titles that are localisation keys ("$...")
 * are returned as the stored key; floats are IEEE single precision values held in JS doubles.
 */

const VERSION = '1.0.0';
const MAX_INPUT = 64 * 1024 * 1024;
const f32 = Math.fround;

// ----------------------------------------------------------------------------------------------
// errors
// ----------------------------------------------------------------------------------------------

class HmsMvarError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'HmsMvarError';
    this.code = code;
  }
}

// ----------------------------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------------------------

const _cv = new DataView(new ArrayBuffer(8));
function f32FromBits(u) { _cv.setUint32(0, u >>> 0); return _cv.getFloat32(0); }
function hex(bytes, from, to) {
  let s = '';
  for (let i = from; i < to; i++) s += (bytes[i] < 16 ? '0' : '') + bytes[i].toString(16);
  return s;
}
function hex64(big) { return big.toString(16).padStart(16, '0'); }
function be32(b, o) { return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0; }
function be16(b, o) { return (b[o] << 8) | b[o + 1]; }
const _utf8 = new TextDecoder('utf-8', { ignoreBOM: true });
function utf8Lossy(bytes) { return _utf8.decode(bytes); }
function utf16Lossy(units) {
  let s = '';
  for (let i = 0; i < units.length; i++) {
    const u = units[i];
    if (u >= 0xD800 && u <= 0xDBFF && i + 1 < units.length && units[i + 1] >= 0xDC00 && units[i + 1] <= 0xDFFF) {
      s += String.fromCharCode(u, units[i + 1]); i++;
    } else if (u >= 0xD800 && u <= 0xDFFF) s += '�';
    else s += String.fromCharCode(u);
  }
  return s;
}
function degrees(r) { return r * 180 / Math.PI; }
function isoDate(ts) {
  if (!Number.isFinite(ts) || ts <= 0 || ts > 8.64e12 / 1000) return null;
  return new Date(ts * 1000).toISOString();
}

// ----------------------------------------------------------------------------------------------
// MSB-first bit reader. Reading past the end is a 'truncated' error, never a zero fill.
// ----------------------------------------------------------------------------------------------

class Bits {
  constructor(buf) { this.buf = buf; this.pos = 0; this.nbits = buf.length * 8; }
  bits(n) {                      // n <= 48, returns a non-negative Number
    if (n === 0) return 0;
    let pos = this.pos;
    if (pos + n > this.nbits) throw new HmsMvarError('truncated', 'the variant data ends before all fields were read (bit ' + pos + ' of ' + this.nbits + ')');
    let r = 0, left = n;
    while (left > 0) {
      const avail = 8 - (pos & 7);
      const take = left < avail ? left : avail;
      const v = (this.buf[pos >> 3] >> (avail - take)) & ((1 << take) - 1);
      r = r * (1 << take) + v;
      pos += take; left -= take;
    }
    this.pos = pos;
    return r;
  }
  signed(n) { const r = this.bits(n); return r >= 2 ** (n - 1) ? r - 2 ** n : r; }
  flag() { return this.bits(1) !== 0; }
  u8() { return this.bits(8); }
  i8() { const v = this.bits(8); return v >= 128 ? v - 256 : v; }
  u16() { return this.bits(16); }
  u32() { return this.bits(32); }
  big64() { const hi = this.bits(32), lo = this.bits(32); return (BigInt(hi) << 32n) | BigInt(lo); }
  f32() { return f32FromBits(this.bits(32)); }
  bytes(n) {
    if (this.pos + n * 8 > this.nbits) throw new HmsMvarError('truncated', 'the variant data ends inside a byte block');
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = this.bits(8);
    return out;
  }
  cstr(max) {                    // up to `max` bytes, stops at (and consumes) a 0 byte
    const v = [];
    for (let i = 0; i < max; i++) { const b = this.bits(8); if (b === 0) break; v.push(b); }
    return utf8Lossy(Uint8Array.from(v));
  }
  wstr(max) {                    // up to `max` UTF-16BE units, stops at (and consumes) 0x0000
    const v = [];
    for (let i = 0; i < max; i++) { const u = this.bits(16); if (u === 0) break; v.push(u); }
    return utf16Lossy(v);
  }
  index(bits) { return this.flag() ? null : this.bits(bits); }   // set flag = "none"
}

// ----------------------------------------------------------------------------------------------
// zlib inflate (RFC 1950/1951), bounded output. Used for the compressed label table and for a
// `_cmp` chunk. Throws a plain Error on any malformed stream; callers decide what that means.
// ----------------------------------------------------------------------------------------------

const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CLORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function huff(lengths, n) {
  const count = new Uint16Array(16), symbol = new Uint16Array(n);
  for (let i = 0; i < n; i++) count[lengths[i]]++;
  const offs = new Uint16Array(16);
  for (let l = 1; l < 15; l++) offs[l + 1] = offs[l] + count[l];
  for (let i = 0; i < n; i++) if (lengths[i] !== 0) symbol[offs[lengths[i]]++] = i;
  return { count, symbol };
}
let _fixed = null;
function fixedTables() {
  if (_fixed) return _fixed;
  const l = new Uint8Array(288);
  for (let i = 0; i < 144; i++) l[i] = 8;
  for (let i = 144; i < 256; i++) l[i] = 9;
  for (let i = 256; i < 280; i++) l[i] = 7;
  for (let i = 280; i < 288; i++) l[i] = 8;
  const d = new Uint8Array(30).fill(5);
  _fixed = { lit: huff(l, 288), dist: huff(d, 30) };
  return _fixed;
}

function inflateRaw(src, start, maxOut) {
  let pos = start, bitBuf = 0, bitCnt = 0;
  let out = new Uint8Array(Math.min(Math.max(1024, src.length * 4), maxOut)), op = 0;
  const need = (k) => {
    if (op + k > maxOut) throw new Error('inflate: output limit');
    if (op + k > out.length) {
      const n = new Uint8Array(Math.min(Math.max(out.length * 2, op + k), maxOut));
      n.set(out.subarray(0, op)); out = n;
    }
  };
  const getBits = (n) => {
    while (bitCnt < n) {
      if (pos >= src.length) throw new Error('inflate: input ends early');
      bitBuf |= src[pos++] << bitCnt; bitCnt += 8;
    }
    const v = bitBuf & ((1 << n) - 1);
    bitBuf >>>= n; bitCnt -= n;
    return v;
  };
  const decode = (h) => {
    let code = 0, first = 0, index = 0;
    for (let len = 1; len <= 15; len++) {
      code |= getBits(1);
      const count = h.count[len];
      if (code - count < first) return h.symbol[index + (code - first)];
      index += count; first += count; first <<= 1; code <<= 1;
    }
    throw new Error('inflate: bad code');
  };
  const codes = (lit, dist) => {
    for (;;) {
      let sym = decode(lit);
      if (sym < 256) { need(1); out[op++] = sym; }
      else if (sym === 256) return;
      else {
        sym -= 257;
        if (sym >= 29) throw new Error('inflate: bad length symbol');
        const len = LBASE[sym] + getBits(LEXT[sym]);
        const ds = decode(dist);
        if (ds >= 30) throw new Error('inflate: bad distance symbol');
        const d = DBASE[ds] + getBits(DEXT[ds]);
        if (d > op) throw new Error('inflate: distance too far');
        need(len);
        for (let i = 0; i < len; i++) { out[op] = out[op - d]; op++; }
      }
    }
  };
  let last;
  do {
    last = getBits(1);
    const type = getBits(2);
    if (type === 0) {
      bitBuf = 0; bitCnt = 0;
      if (pos + 4 > src.length) throw new Error('inflate: input ends early');
      const len = src[pos] | (src[pos + 1] << 8), nlen = src[pos + 2] | (src[pos + 3] << 8);
      pos += 4;
      if ((len ^ 0xFFFF) !== nlen) throw new Error('inflate: bad stored block');
      if (pos + len > src.length) throw new Error('inflate: input ends early');
      need(len); out.set(src.subarray(pos, pos + len), op); op += len; pos += len;
    } else if (type === 1) {
      const t = fixedTables(); codes(t.lit, t.dist);
    } else if (type === 2) {
      const nlen = getBits(5) + 257, ndist = getBits(5) + 1, ncode = getBits(4) + 4;
      if (nlen > 286 || ndist > 30) throw new Error('inflate: bad counts');
      const lengths = new Uint8Array(320);
      for (let i = 0; i < ncode; i++) lengths[CLORDER[i]] = getBits(3);
      const lencode = huff(lengths.subarray(0, 19), 19);
      const ll = new Uint8Array(320);
      let idx = 0;
      while (idx < nlen + ndist) {
        let sym = decode(lencode);
        if (sym < 16) ll[idx++] = sym;
        else {
          let prev = 0, rep;
          if (sym === 16) { if (idx === 0) throw new Error('inflate: no previous length'); prev = ll[idx - 1]; rep = 3 + getBits(2); }
          else if (sym === 17) rep = 3 + getBits(3);
          else rep = 11 + getBits(7);
          if (idx + rep > nlen + ndist) throw new Error('inflate: too many lengths');
          while (rep--) ll[idx++] = prev;
        }
      }
      if (ll[256] === 0) throw new Error('inflate: no end of block');
      codes(huff(ll.subarray(0, nlen), nlen), huff(ll.subarray(nlen, nlen + ndist), ndist));
    } else throw new Error('inflate: bad block type');
  } while (!last);
  return out.subarray(0, op);
}
function inflateZlib(src, start, maxOut) {
  if (src.length - start < 2) throw new Error('zlib: too short');
  const cmf = src[start], flg = src[start + 1];
  if ((cmf & 15) !== 8 || ((cmf << 8) | flg) % 31 !== 0 || (flg & 32)) throw new Error('zlib: bad header');
  return inflateRaw(src, start + 2, maxOut);
}

// ----------------------------------------------------------------------------------------------
// SHA-1 (chunk integrity hash)
// ----------------------------------------------------------------------------------------------

function sha1(data) {
  const l = data.length, total = ((l + 9 + 63) >> 6) << 6;
  const buf = new Uint8Array(total);
  buf.set(data); buf[l] = 0x80;
  const dv = new DataView(buf.buffer);
  dv.setUint32(total - 4, (l * 8) >>> 0); dv.setUint32(total - 8, Math.floor(l / 0x20000000));
  let h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
  const w = new Uint32Array(80);
  for (let o = 0; o < total; o += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getUint32(o + i * 4);
    for (let i = 16; i < 80; i++) { const x = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]; w[i] = (x << 1) | (x >>> 31); }
    let a = h0, b = h1, c = h2, d = h3, e = h4;
    for (let i = 0; i < 80; i++) {
      let fn, k;
      if (i < 20) { fn = (b & c) | (~b & d); k = 0x5A827999; }
      else if (i < 40) { fn = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (i < 60) { fn = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else { fn = b ^ c ^ d; k = 0xCA62C1D6; }
      const t = (((a << 5) | (a >>> 27)) + fn + e + k + w[i]) >>> 0;
      e = d; d = c; c = (b << 30) | (b >>> 2); b = a; a = t;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  const out = new Uint8Array(20), od = new DataView(out.buffer);
  [h0, h1, h2, h3, h4].forEach((v, i) => od.setUint32(i * 4, v));
  return out;
}
// The Xbox 360 era "shared" BLF salt, as used by Halo Map Studio's own writer (h4/mvar.rs SALT_360).
const SALT_360 = Uint8Array.from([
  0xED, 0xD4, 0x30, 0x09, 0x66, 0x6D, 0x5C, 0x4A, 0x5C, 0x36, 0x57, 0xFA, 0xB4, 0x0E, 0x02, 0x2F, 0x53,
  0x5A, 0xC6, 0xC9, 0xEE, 0x47, 0x1F, 0x01, 0xF1, 0xA4, 0x47, 0x56, 0xB7, 0x71, 0x4F, 0x1C, 0x36, 0xEC]);
function chunkHash(payload, length, kind) {
  const body = payload.subarray(0, Math.min(length, payload.length));
  const pre = kind === 'plain-le' ? 4 : SALT_360.length + 4;
  const m = new Uint8Array(pre + body.length);
  if (kind === 'plain-le') {
    m[0] = length & 255; m[1] = (length >>> 8) & 255; m[2] = (length >>> 16) & 255; m[3] = (length >>> 24) & 255;
  } else {
    m.set(SALT_360, 0);
    const o = SALT_360.length; m[o] = (length >>> 24) & 255; m[o + 1] = (length >>> 16) & 255; m[o + 2] = (length >>> 8) & 255; m[o + 3] = length & 255;
  }
  m.set(body, pre);
  return sha1(m);
}
function sameBytes(a, ao, b) { for (let i = 0; i < b.length; i++) if (a[ao + i] !== b[i]) return false; return true; }

// ----------------------------------------------------------------------------------------------
// BLF container
// ----------------------------------------------------------------------------------------------

function ascii4(b, o) { return String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]); }

// Inflate every `_cmp` chunk in place (payload: u8 type 0 = zlib, u32 BE size, zlib stream; it
// inflates to an ordinary chunk) and re-stamp `_eof`'s byte offset. Mirrors expand_blf_cmp.
function expandCmp(data, warnings) {
  const parts = [];
  let total = 0, pos = 0, expanded = false;
  const push = (u8) => { parts.push(u8); total += u8.length; };
  while (pos + 12 <= data.length) {
    const magic = ascii4(data, pos), size = be32(data, pos + 4);
    if (size < 12 || pos + size > data.length) break;
    if (magic === '_cmp' && size >= 17) {
      const body = data.subarray(pos + 12, pos + size);
      const ctype = body[0], want = be32(body, 1);
      let v = null;
      if (ctype === 0 && want <= MAX_INPUT) { try { v = inflateZlib(body, 5, MAX_INPUT); } catch (e) { v = null; } }
      if (v && v.length === want && v.length >= 12) { push(v); expanded = true; }
      else { warnings.push('unsupported or damaged _cmp chunk left as it was'); push(data.subarray(pos, pos + size)); }
    } else if (magic === '_eof' && expanded && size >= 16) {
      const e = new Uint8Array(size); e.set(data.subarray(pos, pos + size));
      const before = total;
      e[12] = (before >>> 24) & 255; e[13] = (before >>> 16) & 255; e[14] = (before >>> 8) & 255; e[15] = before & 255;
      push(e);
    } else push(data.subarray(pos, pos + size));
    pos += size;
  }
  if (pos < data.length) push(data.subarray(pos));
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return { data: out, expanded };
}

function walkChunks(data) {
  const chunks = [];
  let pos = 0;
  while (pos + 12 <= data.length) {
    const size = be32(data, pos + 4);
    if (size < 12 || pos + size > data.length) break;
    chunks.push({ type: ascii4(data, pos), offset: pos, size, major: be16(data, pos + 8), minor: be16(data, pos + 10) });
    pos += size;
  }
  return { chunks, end: pos };
}

// ----------------------------------------------------------------------------------------------
// Forge label string table (Reach and Halo 4 share it)
// ----------------------------------------------------------------------------------------------

function readLabels(br, warnings, reachStyle) {
  const count = br.bits(9);
  if (count === 0) return { labels: [], start: br.pos };
  const offsets = [];
  for (let i = 0; i < count; i++) offsets.push(br.flag() ? br.bits(12) : null);
  const bufferSize = br.bits(13);
  let buf;
  if (br.flag()) {
    const clen = br.bits(13);
    const blob = br.bytes(clen);
    buf = new Uint8Array(0);
    if (blob.length > 4) {
      try { buf = inflateZlib(blob, 4, 1 << 20); } catch (e) { warnings.push('the compressed label table could not be inflated (' + e.message + '); labels are empty'); }
    }
  } else buf = br.bytes(bufferSize);
  const labels = offsets.map((o) => {
    if (o === null || o >= buf.length) return '';
    let end = o;
    while (end < buf.length && buf[end] !== 0) end++;
    return utf8Lossy(buf.subarray(o, end));
  });
  return { labels };
}

// ----------------------------------------------------------------------------------------------
// Reach orientation (f64 maths, rounded to f32 on the way out) - port of mvar.rs
// ----------------------------------------------------------------------------------------------

const GLOBAL_UP = [0, 0, 1], GLOBAL_FORWARD = [1, 0, 0], GLOBAL_LEFT = [0, 1, 0];
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize3(v) {
  const m = Math.sqrt(dot3(v, v));
  if (m > 1e-6) { v[0] /= m; v[1] /= m; v[2] /= m; }
  return v;
}
function unitVectorConstants(bitCount) {
  let mask = 0, i = 0, bit = bitCount - 2;
  while (bit >= 0) { if (i % 2 === 1) mask += 2 ** bit; bit--; i++; }
  return [mask, Math.floor(Math.sqrt(mask)) - 1];
}
function dequantizeRealF(q, min, max, qvc, exactMid, exactEnd) {
  let count = qvc;
  if (exactMid) count -= 1;
  if (exactEnd) {
    if (q === 0) return min;
    if (q === count - 1) return max;
    return min + ((max - min) / (count - 2)) * ((q - 1) + 0.5);
  }
  return min + ((max - min) / count) * (q + 0.5);
}
function dequantizeUnitVector3d(value, bitCount) {
  const [apamc, qvc] = unitVectorConstants(bitCount);
  if (apamc <= 0 || qvc <= 0) return GLOBAL_UP.slice();
  const face = Math.max(Math.floor(value / apamc), 0);
  const rem = ((value % apamc) + apamc) % apamc;
  const u = dequantizeRealF(Math.floor(rem / qvc), -1, 1, qvc, true, false);
  const w = dequantizeRealF(rem % qvc, -1, 1, qvc, true, false);
  let v;
  switch (face) {
    case 0: v = [1, u, w]; break;
    case 1: v = [u, 1, w]; break;
    case 2: v = [u, w, 1]; break;
    case 3: v = [-1, u, w]; break;
    case 4: v = [u, -1, w]; break;
    case 5: v = [u, w, -1]; break;
    default: v = GLOBAL_UP.slice();
  }
  return normalize3(v);
}
function angleToAxes(up, angle) {
  const v10 = Math.abs(dot3(up, GLOBAL_FORWARD)), v9 = Math.abs(dot3(up, GLOBAL_LEFT));
  const forward = normalize3(v10 >= v9 ? cross3(GLOBAL_LEFT, up) : cross3(up, GLOBAL_FORWARD));
  let s, c;
  if (angle === Math.PI || angle === -Math.PI) { s = 0; c = -1; } else { s = Math.sin(angle); c = Math.cos(angle); }
  const cr = cross3(up, forward), d = dot3(up, forward), omc = 1 - c;
  return normalize3([
    forward[0] * c + cr[0] * s + up[0] * d * omc,
    forward[1] * c + cr[1] * s + up[1] * d * omc,
    forward[2] * c + cr[2] * s + up[2] * d * omc]);
}
function decodeOrientationReach(upIsGlobal, upQuant, angleQ) {
  const up = upIsGlobal ? GLOBAL_UP.slice() : dequantizeUnitVector3d(upQuant, 20);
  const angle = dequantizeRealF(angleQ, -Math.PI, Math.PI, 16384, false, false);
  const fwd = angleToAxes(up, angle);
  return [fwd.map(f32), up.map(f32)];
}

// Euler-style angles derived from a forward/up pair (pure geometry, degrees): yaw about +Z from
// +X, pitch up from the horizon, roll = turn of `up` about `forward` from the zero-roll up for
// that forward (the zero-roll left is Z x forward).
function eulerFrom(fwd, up) {
  const yaw = degrees(Math.atan2(fwd[1], fwd[0]));
  const pitch = degrees(Math.asin(Math.max(-1, Math.min(1, fwd[2]))));
  let l0 = cross3([0, 0, 1], fwd);
  if (Math.sqrt(dot3(l0, l0)) < 1e-6) l0 = [0, 1, 0];
  normalize3(l0);
  const u0 = cross3(fwd, l0);
  const roll = degrees(Math.atan2(dot3(up, l0), dot3(up, u0)));
  return { yaw, pitch, roll };
}

// ----------------------------------------------------------------------------------------------
// shared description helpers
// ----------------------------------------------------------------------------------------------

const PHYSICS = ['normal', 'fixed', 'unknown', 'phased'];
const SYMMETRY = ['never', 'asymmetric', 'symmetric', 'both'];
function placementReach(raw) {
  return {
    raw,
    physics: PHYSICS[(raw >> 6) & 3],
    symmetry: SYMMETRY[(raw >> 2) & 3],
    placedAtStart: (raw & 0x10) === 0,
    uniqueSpawn: (raw & 0x20) !== 0,
    hideUnlessRequired: (raw & 0x01) !== 0,
    isShortcut: (raw & 0x02) !== 0,
  };
}
// INFERRED from which objects carry only that bit on stock variants (not proven): the other bits are unnamed
const H3_ENGINES = [[0, 'ctf'], [3, 'oddball'], [5, 'koth'], [6, 'assault'], [7, 'territories']];
const SHAPES = ['none', 'sphere', 'cylinder', 'box'];
const SHAPE_COUNT = [0, 1, 3, 4];
function boundsObj(b) {
  return { xMin: b[0], xMax: b[1], yMin: b[2], yMax: b[3], zMin: b[4], zMax: b[5], min: [b[0], b[2], b[4]], max: [b[1], b[3], b[5]] };
}
function stampObj(ts, xuid, online) {
  const timestamp = Number(ts);
  return { timestamp, date: isoDate(timestamp), xuid: hex64(xuid), online };
}
function posOk(o) { return Number.isFinite(o.x) && Number.isFinite(o.y) && Number.isFinite(o.z); }
function withEuler(o, fwd, up) {
  const e = eulerFrom(fwd, up);
  o.yaw = e.yaw; o.pitch = e.pitch; o.roll = e.roll;
}

// ----------------------------------------------------------------------------------------------
// REACH  (mvar chunk v31)
// ----------------------------------------------------------------------------------------------

function reachAxisBits(bitcount, ext) {
  const MIN_UNIT = f32(0.00833333333);
  let minStep = bitcount > 16 ? f32(MIN_UNIT / f32(2 ** (bitcount - 16))) : f32(f32(2 ** (16 - bitcount)) * MIN_UNIT);
  if (minStep < f32(0.0001)) return [26, 26, 26];
  minStep = f32(minStep * 2);
  const out = [];
  for (let i = 0; i < 3; i++) {
    let e = f32(f32(ext[i] / minStep) + f32(0.9999));
    e = Number.isNaN(e) ? 8388608 : Math.min(e, 8388608);
    const edx = e <= -2147483648 ? -2147483648 : Math.trunc(e);
    if (edx <= 0) { out.push(0); continue; }
    const ecx = 31 - Math.clz32(edx);
    const eax = (1 << ecx) - 1;
    out.push(Math.min(ecx + ((edx & eax) !== 0 ? 1 : 0), 26));
  }
  return out;
}

function parseReachHeader(br, warnings) {
  const h = {};
  h.contentType = br.bits(4) - 1;
  h.fileSize = br.bits(32);
  h.uid = br.big64(); h.parentUid = br.big64(); h.rootUid = br.big64(); h.gameId = br.big64();
  h.activity = br.bits(3) - 1;
  h.gameMode = br.bits(3);
  h.engine = br.bits(3);
  h.headerMapId = br.bits(32);
  h.category = br.i8();
  const author = () => { const ts = br.big64(), xuid = br.big64(); const name = br.cstr(16); const online = br.flag(); return { name, ts, xuid, online }; };
  const a = author(), e = author();
  h.author = a.name; h.created = stampObj(a.ts, a.xuid, a.online);
  h.editor = e.name; h.modified = stampObj(e.ts, e.xuid, e.online);
  h.title = br.wstr(128);
  h.description = br.wstr(128);
  if (h.contentType === 3 || h.contentType === 4) br.bits(32);
  else if (h.contentType === 6) br.bits(8);
  h.hopperId = h.activity === 2 ? br.u16() : null;
  if (h.gameMode === 1) { br.bits(8); br.bits(2); br.bits(2); br.bits(8); br.bits(32); }
  else if (h.gameMode === 2) { br.bits(2); br.bits(32); }
  h.version = br.bits(8);
  h.checksum = br.bits(32);
  h.paletteCrc = br.bits(32);
  h.numQuotas = br.bits(9);
  h.mapId = br.bits(32);
  h.builtIn = br.flag();
  h.builtFromXml = br.flag();
  h.bb = [br.f32(), br.f32(), br.f32(), br.f32(), br.f32(), br.f32()];
  h.budgetMax = br.bits(32);
  h.budgetSpent = br.bits(32);
  h.labels = readLabels(br, warnings).labels;
  h.mccMapId = h.version >= 32 ? hex(br.bytes(16), 0, 16) : null;
  return h;
}

function parseReachSlot(br, slotIdx, axis, bb, ext, labels, warnings) {
  if (!br.flag()) return null;
  const flags = br.bits(2);
  const folder = br.flag() ? 0xFFFF : br.bits(8);
  const item = br.flag() ? 0xFF : br.bits(5);
  const inBounds = br.flag();
  let bspIndex = -1;
  if (!inBounds && !br.flag()) bspIndex = br.bits(2);
  const pos = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const q = br.bits(axis[i]);
    pos[i] = f32(f32(f32(0.5 + f32(q)) * f32(ext[i] / f32(2 ** axis[i]))) + bb[2 * i]);
  }
  const upIsGlobal = br.flag();
  const upQuant = upIsGlobal ? 0 : br.bits(20);
  const angleQ = br.bits(14);
  const spawnRel = br.bits(10) - 1;
  const shape = br.bits(2);
  const boundary = [0, 0, 0, 0];
  for (let k = 0; k < SHAPE_COUNT[shape]; k++) boundary[k] = br.bits(11);
  const spawnSeq = br.signed(8);
  const respawn = br.bits(8);
  const mpType = br.bits(5);
  const labelIdx = !br.flag() ? br.bits(8) : 0xFFFF;
  const placement = br.bits(8);
  const teamRaw = br.bits(4) - 1;
  const team = teamRaw < 0 ? 0xFF : teamRaw;
  const color = br.flag() ? -1 : br.bits(3);
  let weaponClips = 0, teleChannel = 0, telePass = 0, locationName = 0xFFFF;
  if (mpType === 1) weaponClips = br.bits(8);
  else if (mpType >= 12 && mpType <= 14) { teleChannel = br.bits(5); telePass = br.bits(5); }
  else if (mpType === 19 && !br.flag()) locationName = br.bits(8);
  if (folder === 0xFFFF || item === 0xFF) return { empty: true };
  const [fwd, up] = decodeOrientationReach(upIsGlobal, upQuant, angleQ);
  const raw = {
    folder, item, pos, team, color, fwd, up, upIsGlobal, upQuant, forwardAngleQ: angleQ, spawnSeq, respawn,
    cachedType: mpType, labelIdx, placement, boundaryShape: shape, flags, inBounds, bspIndex, spawnRel, slot: slotIdx,
    boundary, weaponClips, teleChannel, telePassability: telePass, locationName,
  };
  const extra = {};
  if (mpType === 1) extra.weaponClips = weaponClips;
  if (mpType >= 12 && mpType <= 14) { extra.teleporterChannel = teleChannel; extra.teleporterPassability = telePass; }
  if (mpType === 19) extra.locationNameIndex = locationName === 0xFFFF ? null : locationName;
  const o = {
    slot: slotIdx, paletteIndex: folder, variantIndex: item, x: pos[0], y: pos[1], z: pos[2], forward: fwd, up,
    team: team === 0xFF ? null : team, label: labelIdx === 0xFFFF ? null : (labels[labelIdx] !== undefined ? labels[labelIdx] : null),
    labelIndex: labelIdx === 0xFFFF ? null : labelIdx, color: color < 0 ? null : color,
    placement: placementReach(placement), spawnSequence: spawnSeq, respawnTime: respawn, objectType: mpType,
    shape: { type: SHAPES[shape], raw: boundary.slice(0, SHAPE_COUNT[shape]), values: null },
    parentSlot: spawnRel < 0 ? null : spawnRel, flags, inBounds, extra, raw,
  };
  withEuler(o, fwd, up);
  return o;
}

function parseReach(payload, blf, warnings) {
  const br = new Bits(payload);
  const h = parseReachHeader(br, warnings);
  const bb = h.bb;
  const ext = [f32(bb[1] - bb[0]), f32(bb[3] - bb[2]), f32(bb[5] - bb[4])];
  const axis = reachAxisBits(21, ext);
  const objects = [];
  let empties = 0;
  for (let s = 0; s < 651; s++) {
    const o = parseReachSlot(br, s, axis, bb, ext, h.labels, warnings);
    if (!o) continue;
    if (o.empty) { empties++; continue; }
    objects.push(o);
  }
  const quotas = [];
  for (let i = 0; i < h.numQuotas; i++) {
    if (br.pos + 24 > 229408) break;
    quotas.push({ index: i, min: br.bits(8), max: br.bits(8), placed: br.bits(8) });
  }
  const endBit = br.pos;
  const out = {
    game: 'reach', blf, warnings,
    header: {
      title: h.title, description: h.description, author: h.author, editor: h.editor, titleIsKey: h.title.startsWith('$'),
      created: h.created, modified: h.modified, category: h.category, contentType: h.contentType, fileSize: h.fileSize,
      uid: hex64(h.uid), parentUid: hex64(h.parentUid), rootUid: hex64(h.rootUid), gameId: hex64(h.gameId),
      activity: h.activity, gameMode: h.gameMode, engine: h.engine, hopperId: h.hopperId,
    },
    map: {
      mapId: h.mapId, headerMapId: h.headerMapId, mccMapId: h.mccMapId, paletteCrc: h.paletteCrc.toString(16).padStart(8, '0'),
      version: h.version, checksum: h.checksum, builtIn: h.builtIn, builtFromXml: h.builtFromXml,
      unassignedMapId: h.mapId === 0xFFFFFFFE || h.mapId === 0xFFFFFFFF || h.mapId === 0,
    },
    bounds: boundsObj(bb), budget: { max: h.budgetMax, spent: h.budgetSpent },
    quotas, labels: h.labels, objects, extra: { emptySlots: empties, endBit, numQuotas: h.numQuotas },
  };
  objects.forEach((o, i) => { o.index = i; });
  if (h.numQuotas !== quotas.length) warnings.push('quota table is shorter than its declared count');
  return out;
}

// ----------------------------------------------------------------------------------------------
// HALO 3  (mvar chunk v12) - port of h3/mvar.rs (f32 maths, glam operation order)
// ----------------------------------------------------------------------------------------------

const PI32 = f32(Math.PI);
function h3Dequant(q, min, max, bits, exactMid) {
  let steps = 2 ** bits - 1;
  if (exactMid) steps -= steps % 2;
  if (q === 0) return min;
  if (q >= steps) return max;
  return f32(f32(f32(f32(steps - q) * min) + f32(f32(q) * max)) / f32(steps));
}
const vdot = (a, b) => f32(f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) + f32(a[2] * b[2]));
const vcross = (a, b) => [f32(f32(a[1] * b[2]) - f32(b[1] * a[2])), f32(f32(a[2] * b[0]) - f32(b[2] * a[0])), f32(f32(a[0] * b[1]) - f32(b[0] * a[1]))];
const vscale = (a, s) => [f32(a[0] * s), f32(a[1] * s), f32(a[2] * s)];
const vadd = (a, b) => [f32(a[0] + b[0]), f32(a[1] + b[1]), f32(a[2] + b[2])];
function vnorm0(a) {
  const rcp = f32(1 / f32(Math.sqrt(vdot(a, a))));
  return Number.isFinite(rcp) && rcp > 0 ? vscale(a, rcp) : [0, 0, 0];
}
function h3Seed(up) {
  const v = Math.abs(up[1]) <= Math.abs(up[0]) ? [up[2], 0, -up[0]] : [0, up[2], -up[1]];
  return vnorm0(v);
}
function h3UnitVector(q) {
  const a = h3Dequant((q >> 3) & 0xFF, -1, 1, 8, true), b = h3Dequant((q >> 11) & 0xFF, -1, 1, 8, true);
  let v;
  switch (q & 7) {
    case 0: v = [1, a, b]; break;
    case 1: v = [a, 1, b]; break;
    case 2: v = [a, b, 1]; break;
    case 3: v = [-1, a, b]; break;
    case 4: v = [a, -1, b]; break;
    case 5: v = [a, b, -1]; break;
    default: v = [0, 0, 1];
  }
  return vnorm0(v);
}
function h3Rotate(v, axis, s, c) {
  const t1 = vscale(axis, f32(vdot(axis, v) * f32(1 - c)));
  return vadd(vadd(t1, vscale(v, c)), vscale(vcross(axis, v), s));
}
function h3Axes(br) {
  const upIsGlobal = br.flag();
  let up, upQ = 0;
  if (upIsGlobal) up = [0, 0, 1]; else { upQ = br.bits(19); up = h3UnitVector(upQ); }
  const angleQ = br.u8();
  const angle = h3Dequant(angleQ, -PI32, PI32, 8, true);
  const seed = h3Seed(up);
  let s, c;
  if (angle === PI32 || angle === -PI32) { s = 0; c = -1; } else { s = f32(Math.sin(angle)); c = f32(Math.cos(angle)); }
  const fwd = vnorm0(h3Rotate(seed, up, s, c));
  return { fwd, up, upIsGlobal, upQ, angleQ };
}

function parseHalo3(body, blf, warnings) {
  const br = new Bits(body);
  const h = {};
  h.uniqueId = br.big64();
  h.name = br.wstr(16);
  h.description = br.cstr(128);
  h.author = br.cstr(16);
  h.fileType = br.bits(5) - 1;
  h.authorOnline = br.flag();
  h.authorXuid = br.big64();
  h.sizeBytes = br.big64();
  h.date = br.big64();
  h.lengthSeconds = br.u32();
  h.campaignId = br.u32() | 0;
  h.mapId = br.u32();
  h.gameEngineType = br.bits(4);
  h.campaignDifficulty = br.bits(3) - 1;
  h.hopperId = br.u16();
  h.gameId = br.big64();
  const version = br.u8();
  const checksum = br.u32();
  const numScenario = br.bits(10), numObjects = br.bits(10), numQuotas = br.bits(9);
  if (version > 14) throw new HmsMvarError('unsupported', 'Halo 3 map variant body version ' + version + ' is newer than this reader knows (14)');
  if (numObjects > 640 || numQuotas > 256) throw new HmsMvarError('corrupt', 'counts out of range: ' + numObjects + ' objects / ' + numQuotas + ' quotas');
  const mapId = br.u32();
  const builtIn = br.flag();
  const bounds = [];
  for (let i = 0; i < 6; i++) bounds.push(br.f32());
  const subtype = br.bits(4);
  const budgetMax = br.f32(), budgetSpent = br.f32();
  if (!bounds.every(Number.isFinite)) throw new HmsMvarError('corrupt', 'world bounds are not finite');
  const objects = [], unplaced = [];
  for (let slot = 0; slot < numObjects; slot++) {
    if (!br.flag()) continue;
    const flags = br.u16();
    const quota = br.u32() | 0;
    const parent = br.flag() ? hex64(br.big64()) : null;
    const rec = { slot, flags, quota, parent, placed: false };
    if (!br.flag()) { unplaced.push({ slot, flags, quota, parent, scenarioOwned: (flags & 0x08) !== 0, deleted: (flags & 0x20) !== 0 }); continue; }
    rec.placed = true;
    const posQ = [], pos = [];
    for (let k = 0; k < 3; k++) { const q = br.bits(16); posQ.push(q); pos.push(h3Dequant(q, bounds[2 * k], bounds[2 * k + 1], 16, false)); }
    const ax = h3Axes(br);
    const objectType = br.u8(), mpFlags = br.u8(), engineFlags = br.u16(), sharedStorage = br.u8(), spawnTime = br.u8(), team = br.u8();
    const shape = br.u8();
    const n = shape === 1 ? 1 : shape === 2 ? 3 : shape === 3 ? 4 : 0;
    const bq = [0, 0, 0, 0], bv = [0, 0, 0, 0];
    for (let k = 0; k < n; k++) { const q = br.u16(); bq[k] = q; bv[k] = h3Dequant(q, 0, 60, 16, false); }
    const raw = {
      slot, flags, quota, parent, placed: true, pos, fwd: ax.fwd, up: ax.up, posQ, upIsGlobal: ax.upIsGlobal, upQ: ax.upQ, angleQ: ax.angleQ,
      boundaryQ: bq, objectType, mpFlags, gameEngineFlags: engineFlags, sharedStorage, spawnTime, team, boundaryShape: shape, boundary: bv,
    };
    const o = {
      slot, paletteIndex: quota < 0 ? null : quota, variantIndex: null, x: pos[0], y: pos[1], z: pos[2], forward: ax.fwd, up: ax.up,
      team: team > 8 ? 0 : team, label: null, labelIndex: null, color: null,
      placement: {
        raw: mpFlags, physics: null,
        symmetry: SYMMETRY[(mpFlags >> 2) & 3],                  // bits 2-3; 'never' (0x00) is ambiguous in the engine
        uniqueSpawn: (mpFlags & 1) !== 0,                         // bit 0 = unique object (name INFERRED, bit use PROVEN in halo3.dll)
        placedAtStart: (mpFlags & 2) === 0,                       // bit 1 = NOT placed at start, stored inverted
        engineMask: engineFlags & 0x3FF,                          // 10-bit game-engine mask: bit n = game engine n; 0 reads as all
        engineGamesInferred: H3_ENGINES.filter((e) => (engineFlags & (1 << e[0])) !== 0).map((e) => e[1]),
        engineFlags,
      },
      spawnSequence: null, respawnTime: spawnTime, objectType,
      shape: { type: SHAPES[shape > 3 ? 0 : shape], raw: bq.slice(0, n), values: bv.slice(0, n) },
      parentSlot: null, parentId: parent, flags, scenarioOwned: (flags & 0x08) !== 0, deleted: (flags & 0x20) !== 0,
      extra: { sharedStorage, mpUnreadBits: mpFlags & 0xF0, teamRaw: team }, raw,
    };
    withEuler(o, ax.fwd, ax.up);
    objects.push(o);
  }
  const typeStart = [];
  for (let t = 0; t < 14; t++) typeStart.push(br.bits(9) - 1);
  const quotas = [];
  for (let i = 0; i < numQuotas; i++) {
    const definition = br.u32();
    quotas.push({ index: i, min: br.u8(), max: br.u8(), placed: br.u8(), maxAllowed: br.u8(), price: br.f32(), definition, kind: definition >>> 16, tagIndex: definition & 0xFFFF });
  }
  const mccMapId = version >= 14 ? hex(br.bytes(16), 0, 16) : null;
  objects.forEach((o, i) => { o.index = i; });
  return {
    game: 'halo3', blf, warnings,
    header: {
      title: h.name, description: h.description, author: h.author, editor: null, titleIsKey: false,
      created: { timestamp: Number(h.date), date: isoDate(Number(h.date)), xuid: hex64(h.authorXuid), online: h.authorOnline }, modified: null,
      category: null, contentType: h.fileType, fileSize: Number(h.sizeBytes), uid: hex64(h.uniqueId), parentUid: null, rootUid: null, gameId: hex64(h.gameId),
      activity: null, gameMode: null, engine: h.gameEngineType, hopperId: h.hopperId,
      lengthSeconds: h.lengthSeconds, campaignId: h.campaignId, campaignDifficulty: h.campaignDifficulty,
    },
    map: {
      mapId, headerMapId: h.mapId, mccMapId, paletteCrc: null, version, checksum, builtIn, builtFromXml: null,
      unassignedMapId: mapId === 0xFFFFFFFE || mapId === 0xFFFFFFFF || mapId === 0,
      numScenarioObjects: numScenario, numVariantObjects: numObjects, gameEngineSubtype: subtype, typeStartIndex: typeStart,
    },
    bounds: boundsObj(bounds), budget: { max: budgetMax, spent: budgetSpent },
    quotas, labels: [], objects, extra: { unplacedSlots: unplaced, endBit: br.pos },
  };
}

// ----------------------------------------------------------------------------------------------
// HALO 4 / HALO 2 ANNIVERSARY  (mvar chunk v50 / v52) - port of h4/mvar.rs
// ----------------------------------------------------------------------------------------------

function h4AxisBits(bitcount, ext) {
  const MIN_UNIT = f32(0.0083333338);
  const minStep = bitcount > 16 ? f32(MIN_UNIT / f32(2 ** (bitcount - 16))) : f32(f32(2 ** (16 - bitcount)) * MIN_UNIT);
  if (minStep < f32(0.0001)) return [26, 26, 26];
  const step2 = f32(minStep * 2);
  const out = [];
  for (let i = 0; i < 3; i++) {
    const c = Math.ceil(f32(ext[i] / step2));
    let v = Number.isNaN(c) ? 0 : (c > 0x400000 ? 0x400000 : (c < -(2 ** 53) ? -(2 ** 53) : c));
    if (v <= 0) { out.push(0); continue; }
    const hb = 31 - Math.clz32(v);
    out.push(Math.min(hb + ((v & ((1 << hb) - 1)) !== 0 ? 1 : 0), 26));
  }
  return out;
}
function h4Dequant(q, min, max, bits, exactMid, exactEnd) {
  let count = 2 ** bits;
  if (exactMid) count -= 1;
  let v;
  if (exactEnd) {
    if (q === 0) v = min;
    else if (q === count - 1) v = max;
    else { const step = f32(f32(max - min) / f32(count - 2)); v = f32(min + f32(step * f32(f32(f32(q) - 1) + 0.5))); }
  } else {
    const step = f32(f32(max - min) / f32(count));
    v = f32(min + f32(step * f32(f32(q) + 0.5)));
  }
  return exactMid && 2 * q === count - 1 ? f32(f32(min + max) * 0.5) : v;
}
function skipTraitSet(r) {
  const opt = (n) => { for (let i = 0; i < n; i++) if (r.flag()) r.bits(16); };
  opt(13); r.bits(8); r.bits(2);
  opt(22); r.bits(2); r.bits(5); r.bits(2); r.bits(24); r.bits(40);
  opt(4); r.bits(4); r.bits(2); r.bits(8);
  opt(1); r.bits(3); r.bits(2); r.bits(2); r.bits(3);
  r.bits(25); r.bits(25); r.bits(1); r.bits(8); r.bits(32); r.bits(32); r.bits(2);
  opt(2); r.bits(3); r.bits(14);
}

function parseHalo4(payload, chunkVersion, blf, warnings) {
  const h2a = chunkVersion === 52;
  if (payload.length < 28676) throw new HmsMvarError('truncated', 'the mvar payload is ' + payload.length + ' bytes; a Halo 4 / H2A variant holds 28676');
  const br = new Bits(payload);
  const H = {};
  H.contentType = br.bits(4) - 1;
  H.fileSize = br.u32();
  H.uid = br.big64(); H.parentUid = br.big64(); H.rootUid = br.big64(); H.gameId = br.big64();
  H.activity = br.bits(2);
  H.gameMode = br.bits(3);
  H.engine = br.bits(3);
  H.headerMapId = br.u32();
  H.category = br.i8();
  const authorBlock = () => { const ts = br.big64(), xuid = br.big64(); const name = br.cstr(16); const online = br.flag(); return { name, ts, xuid, online }; };
  const a = authorBlock(), e = authorBlock();
  H.title = br.wstr(128);
  H.description = br.wstr(128);
  if (H.contentType === 3 || H.contentType === 4) br.bits(32);
  else if (H.contentType === 6) br.bits(8);
  H.hopperId = H.activity === 2 ? br.bits(16) : null;
  if (H.engine === 3) { br.bits(8); br.bits(2); br.bits(2); br.bits(8); br.bits(32); }
  else if (H.engine === 4) { br.bits(2); br.bits(32); }
  const version = br.u8();
  if (version < 50 || version > 53) throw new HmsMvarError('unsupported', 'variant body version ' + version + ' (expected 50-53)');
  const checksum = br.u32();
  const paletteCrc = br.u32();
  const numQuotas = br.bits(9);
  if (numQuotas > 256) throw new HmsMvarError('corrupt', 'num_quotas ' + numQuotas + ' > 256');
  const mapId = br.u32();
  const builtIn = br.flag(), builtFromXml = br.flag();
  const bounds = [];
  for (let i = 0; i < 6; i++) bounds.push(br.f32());
  const budgetMax = br.u32(), budgetSpent = br.u32();
  const lab = readLabels(br, warnings);
  const readGuid = h2a ? version >= 53 : version >= 51;
  const mapGuid = readGuid ? hex(br.bytes(16), 0, 16) : null;
  const ext = [f32(bounds[1] - bounds[0]), f32(bounds[3] - bounds[2]), f32(bounds[5] - bounds[4])];
  const axis = h4AxisBits(21, ext);
  const objects = [];
  for (let slot = 0; slot < 651; slot++) {
    const bit = br.pos;
    if (!br.flag()) continue;
    const flags = br.bits(2);
    const quota = br.index(8);
    const variant = br.index(5);
    const inBounds = br.flag();
    const bspRef = h2a && !inBounds ? br.index(2) : null;
    const pos = [0, 0, 0], posQ = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      posQ[i] = br.bits(axis[i]);
      const step = f32(ext[i] / f32(2 ** axis[i]));
      pos[i] = f32(f32(bounds[2 * i] + f32(f32(posQ[i]) * step)) + f32(step * 0.5));
    }
    const upIsGlobal = br.flag();
    const upQuant = upIsGlobal ? 0 : br.bits(20);
    const angleQ = br.bits(14);
    const [fwd, up] = decodeOrientationReach(upIsGlobal, upQuant, angleQ);
    const parent = br.bits(10) - 1;
    const scaleQ = br.bits(6);
    const locked = br.flag();
    const shape = br.bits(2);
    const shapeValues = [0, 0, 0, 0];
    for (let k = 0; k < SHAPE_COUNT[shape]; k++) shapeValues[k] = br.u16();
    const objectType = br.bits(6);
    const placement = br.bits(h2a ? 11 : 10);
    let team = -1, spawnTime = 0, color = null, spawnSeq = 0, userData = 0, h2aExtra = 0;
    let labels4 = [null, null, null, null];
    let td;
    if (objectType === 32) {
      team = br.bits(4) - 1;
      td = { kind: 'initial_ordnance', team, weapon: br.u8(), timing: br.u16() };
    } else if (objectType === 33) {
      const w = []; for (let k = 0; k < 8; k++) w.push(br.u8());
      td = { kind: 'random_ordnance', weapons: w };
    } else if (objectType === 34) {
      const lead = br.u8(); const w = []; for (let k = 0; k < 8; k++) w.push(br.u8());
      td = { kind: 'objective_ordnance', lead, weapons: w };
    } else if (objectType === 35) {
      td = { kind: 'empty' };
    } else {
      team = br.bits(4) - 1;
      spawnTime = br.u8();
      color = br.index(3);
      spawnSeq = br.i8();
      userData = br.i8();
      if (h2a) h2aExtra = br.u8();
      for (let k = 0; k < 4; k++) labels4[k] = br.index(8);
      if (objectType === 1 || objectType === 12) td = { kind: 'byte', value: br.u8() };
      else if (objectType === 11 && h2a) { const x = br.u8(), y = br.u8(); td = { kind: 'pair', a: x, b: y }; }
      else if (objectType === 20) td = { kind: 'index', value: br.u8() - 1 };
      else if (objectType === 31) td = { kind: 'trait_zone', value: br.bits(5) };
      else { const x = br.bits(5), y = br.bits(5); td = { kind: 'pair', a: x, b: y }; }
    }
    const raw = {
      slot, bit, flags, quota, variant, inBounds, pos, posQ, h2aBspRef: bspRef, upIsGlobal, upQuant, forwardAngleQ: angleQ, fwd, up,
      parent, scaleQ, locked, shape, shapeValues, objectType, placement, team, spawnTime, color, spawnSequence: spawnSeq, userData,
      labels: labels4, h2aExtra, typeData: td,
    };
    const labelNames = labels4.filter((l) => l !== null).map((l) => (lab.labels[l] !== undefined ? lab.labels[l] : null)).filter((s) => s !== null);
    const scale = h4Dequant(scaleQ & 0x3F, 0, 10, 6, false, true);
    const extra = { scale, locked, userData, typeData: td };
    if (h2a) extra.h2aExtra = h2aExtra;
    if (bspRef !== null) extra.bspRef = bspRef;
    const physics = PHYSICS[(placement >> 6) & 3];
    const o = {
      slot, paletteIndex: quota, variantIndex: variant, x: pos[0], y: pos[1], z: pos[2], forward: fwd, up,
      team: team < 0 ? null : team, label: labelNames.length ? labelNames[0] : null, labelIndex: labels4.find((l) => l !== null) !== undefined ? labels4.find((l) => l !== null) : null,
      labels: labelNames, labelIndices: labels4, color,
      placement: {
        raw: placement, physics, symmetry: SYMMETRY[(placement >> 2) & 3], placedAtStart: (placement & 0x10) === 0,
        uniqueSpawn: (placement & 0x20) !== 0, hideUnlessRequired: (placement & 0x01) !== 0, isShortcut: (placement & 0x02) !== 0,
      },
      spawnSequence: spawnSeq, respawnTime: spawnTime, objectType,
      shape: { type: SHAPES[shape], raw: shapeValues.slice(0, SHAPE_COUNT[shape]), values: shapeValues.slice(0, SHAPE_COUNT[shape]).map((v) => f32(v / 256)) },
      parentSlot: parent < 0 ? null : parent, flags, inBounds, extra, raw,
    };
    o.raw.scale = scale;
    withEuler(o, fwd, up);
    objects.push(o);
  }
  const objEnd = br.pos;
  const quotas = [];
  for (let i = 0; i < numQuotas; i++) {
    quotas.push(h2a ? { index: i, min: br.bits(10) & 255, max: br.bits(10) & 255, placed: br.bits(10) & 255 }
      : { index: i, min: br.u8(), max: br.u8(), placed: br.u8() });
  }
  const quotaEnd = br.pos;
  for (let i = 0; i < 4; i++) skipTraitSet(br);
  if (br.pos > 28672 * 8) throw new HmsMvarError('corrupt', 'trait records overran the 28672-byte bitstream');
  objects.forEach((o, i) => { o.index = i; });
  const noVariant = objects.filter((o) => o.paletteIndex !== null && o.variantIndex === null).length;
  if (noVariant) warnings.push(noVariant + ' object(s) store no variant index (MCC spawns the wrong object for those)');
  if (H.title.startsWith('$')) warnings.push('the title is a localisation key (' + H.title + '); the game shows its text from its own string tables');
  return {
    game: h2a ? 'h2a' : 'halo4', blf, warnings,
    header: {
      title: H.title, description: H.description, author: a.name, editor: e.name, titleIsKey: H.title.startsWith('$'), descriptionIsKey: H.description.startsWith('$'),
      created: stampObj(a.ts, a.xuid, a.online), modified: stampObj(e.ts, e.xuid, e.online), category: H.category, contentType: H.contentType,
      fileSize: H.fileSize, uid: hex64(H.uid), parentUid: hex64(H.parentUid), rootUid: hex64(H.rootUid), gameId: hex64(H.gameId),
      activity: H.activity, gameMode: H.gameMode, engine: H.engine, hopperId: H.hopperId,
    },
    map: {
      mapId, headerMapId: H.headerMapId, mccMapId: mapGuid, paletteCrc: paletteCrc.toString(16).padStart(8, '0'), version, checksum,
      builtIn, builtFromXml, unassignedMapId: mapId === 0xFFFFFFFE || mapId === 0xFFFFFFFF || mapId === 0,
    },
    bounds: boundsObj(bounds), budget: { max: budgetMax, spent: budgetSpent },
    quotas, labels: lab.labels, objects, extra: { objEndBit: objEnd, quotaEndBit: quotaEnd, endBit: br.pos },
  };
}

// ----------------------------------------------------------------------------------------------
// parse
// ----------------------------------------------------------------------------------------------

function toBytes(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) return new Uint8Array(input);
  throw new HmsMvarError('bad-input', 'parse() takes an ArrayBuffer or a Uint8Array (use parseFile() for a File or Blob)');
}

function parse(input) {
  const orig = toBytes(input);
  if (orig.length > MAX_INPUT) throw new HmsMvarError('too-large', 'the input is larger than ' + MAX_INPUT + ' bytes');
  if (orig.length < 12 || ascii4(orig, 0) !== '_blf') throw new HmsMvarError('not-mvar', 'not a BLF file (no "_blf" header)');
  const warnings = [];
  const { data, expanded } = expandCmp(orig, warnings);
  const { chunks, end } = walkChunks(data);
  const mv = chunks.find((c) => c.type === 'mvar');
  if (!mv) {
    throw new HmsMvarError(end < data.length || !chunks.length ? 'truncated' : 'not-mvar',
      end < data.length ? 'the file is cut short before an "mvar" chunk could be read' : 'this BLF file has no "mvar" chunk (not a map variant)');
  }
  const hasFsm = chunks.some((c) => c.type === '_fsm');
  const blf = {
    chunks, mvarMajor: mv.major, compressed: expanded, fileSize: data.length, hasFsm,
    chunkLength: null, hashKind: null, hashOk: null,
  };
  let result;
  if (mv.major === 12) {
    result = parseHalo3(data.subarray(mv.offset + 12, mv.offset + mv.size), blf, warnings);
  } else if (mv.major === 31 || mv.major === 50 || mv.major === 52) {
    if (mv.size < 36) throw new HmsMvarError('truncated', 'the "mvar" chunk is too small to hold a variant');
    const payload = data.subarray(mv.offset + 36, mv.offset + mv.size);
    const length = be32(data, mv.offset + 32);
    blf.chunkLength = length;
    for (const kind of ['plain-le', 'salted-be-360']) {
      if (length <= payload.length && sameBytes(data, mv.offset + 12, chunkHash(payload, length, kind))) { blf.hashKind = kind; break; }
    }
    blf.hashOk = blf.hashKind === 'plain-le';
    if (blf.hashKind === null) warnings.push('the stored SHA-1 of the mvar chunk does not match its data (the game does not check it when loading)');
    result = mv.major === 31 ? parseReach(payload, blf, warnings) : parseHalo4(payload, mv.major, blf, warnings);
  } else {
    throw new HmsMvarError('unsupported', 'unsupported "mvar" chunk version ' + mv.major + ' (known: 12 Halo 3, 31 Reach, 50 Halo 4, 52 Halo 2 Anniversary)');
  }
  return result;
}

async function parseFile(blob) {
  if (!blob || typeof blob.arrayBuffer !== 'function') throw new HmsMvarError('bad-input', 'parseFile() takes a File or Blob');
  if (blob.size > MAX_INPUT) throw new HmsMvarError('too-large', 'the file is larger than ' + MAX_INPUT + ' bytes');
  return parse(await blob.arrayBuffer());
}

// ----------------------------------------------------------------------------------------------
// palette names
// ----------------------------------------------------------------------------------------------

function withPalette(parsed, table, opts) {
  const force = !!(opts && opts.force);
  let names = null, entry = null;
  if (Array.isArray(table)) names = table;
  else if (table && typeof table === 'object') {
    if (table.maps && typeof table.maps === 'object') {
      entry = table.maps[String(parsed.map.mapId)] || (parsed.map.mccMapId && table.maps[parsed.map.mccMapId]) || null;
      if (!entry) { parsed.warnings.push('palette: no entry for map id ' + parsed.map.mapId); return parsed; }
    } else entry = table;
    names = Array.isArray(entry.names) ? entry.names : null;
  }
  if (!names) { parsed.warnings.push('palette: no "names" array found'); return parsed; }
  if (entry && !force) {
    if (entry.game && entry.game !== parsed.game) { parsed.warnings.push('palette is for ' + entry.game + ', this variant is ' + parsed.game + '; names not applied'); return parsed; }
    if (entry.mapId !== undefined && !parsed.map.unassignedMapId && Number(entry.mapId) !== parsed.map.mapId) {
      parsed.warnings.push('palette is for map id ' + entry.mapId + ', this variant is on ' + parsed.map.mapId + '; names not applied (pass {force:true} to override)');
      return parsed;
    }
  }
  const nm = (i) => (i !== null && i >= 0 && i < names.length && names[i] !== '' ? String(names[i]) : null);
  for (const o of parsed.objects) o.name = nm(o.paletteIndex);
  for (const q of parsed.quotas) q.name = nm(q.index);
  parsed.paletteApplied = true;
  return parsed;
}

// ----------------------------------------------------------------------------------------------
// filtering / counting / export
// ----------------------------------------------------------------------------------------------

function vec3(v) {
  if (Array.isArray(v)) return v;
  return [v.x, v.y, v.z];
}
function asList(x) { return Array.isArray(x) ? x : [x]; }

function filterObjects(parsed, f) {
  f = f || {};
  const objs = Array.isArray(parsed) ? parsed : parsed.objects;
  const has = (k) => f[k] !== undefined && f[k] !== null;
  const pal = has('paletteIndex') ? asList(f.paletteIndex) : null;
  const typ = has('objectType') ? asList(f.objectType) : null;
  const phy = has('physics') ? asList(f.physics) : null;
  const nameQ = has('name') ? (f.name instanceof RegExp ? f.name : String(f.name).toLowerCase()) : null;
  const labelQ = has('label') ? (f.label instanceof RegExp ? f.label : String(f.label).toLowerCase()) : null;
  const teams = f.team !== undefined ? asList(f.team) : null;
  let bmin = null, bmax = null;
  if (f.box) { bmin = vec3(f.box.min); bmax = vec3(f.box.max); }
  let near = null;
  if (f.near) near = { c: [f.near.x, f.near.y, f.near.z], r2: f.near.radius * f.near.radius };
  const test = (q, s) => (q instanceof RegExp ? q.test(s) : s.toLowerCase().includes(q));
  return objs.filter((o) => {
    if (pal && !pal.includes(o.paletteIndex)) return false;
    if (typ && !typ.includes(o.objectType)) return false;
    if (phy && !phy.includes(o.placement && o.placement.physics)) return false;
    if (teams && !teams.includes(o.team)) return false;
    if (nameQ !== null && !(typeof o.name === 'string' && test(nameQ, o.name))) return false;
    if (labelQ !== null) {
      const ls = (o.labels && o.labels.length ? o.labels : (o.label !== null && o.label !== undefined ? [o.label] : []));
      if (!ls.some((s) => test(labelQ, s))) return false;
    }
    if (bmin && !(o.x >= bmin[0] && o.x <= bmax[0] && o.y >= bmin[1] && o.y <= bmax[1] && o.z >= bmin[2] && o.z <= bmax[2])) return false;
    if (near) {
      const dx = o.x - near.c[0], dy = o.y - near.c[1], dz = o.z - near.c[2];
      if (dx * dx + dy * dy + dz * dz > near.r2) return false;
    }
    if (f.predicate && !f.predicate(o)) return false;
    return true;
  });
}

function pathGet(o, key) {
  let v = o;
  for (const k of String(key).split('.')) { if (v === null || v === undefined) return undefined; v = v[k]; }
  return v;
}
function countBy(objects, key) {
  const out = {};
  const fn = typeof key === 'function' ? key : (o) => pathGet(o, key);
  for (const o of objects) {
    const k = String(fn(o));
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}

const CSV_COLUMNS = [
  ['index', (o) => o.index], ['slot', (o) => o.slot], ['paletteIndex', (o) => o.paletteIndex], ['name', (o) => o.name],
  ['variantIndex', (o) => o.variantIndex], ['x', (o) => o.x], ['y', (o) => o.y], ['z', (o) => o.z],
  ['yaw', (o) => o.yaw], ['pitch', (o) => o.pitch], ['roll', (o) => o.roll],
  ['forwardX', (o) => o.forward[0]], ['forwardY', (o) => o.forward[1]], ['forwardZ', (o) => o.forward[2]],
  ['upX', (o) => o.up[0]], ['upY', (o) => o.up[1]], ['upZ', (o) => o.up[2]],
  ['team', (o) => o.team], ['label', (o) => (o.labels && o.labels.length ? o.labels.join('|') : o.label)],
  ['physics', (o) => o.placement && o.placement.physics], ['symmetry', (o) => o.placement && o.placement.symmetry],
  ['placementRaw', (o) => o.placement && o.placement.raw], ['spawnSequence', (o) => o.spawnSequence], ['respawnTime', (o) => o.respawnTime],
  ['objectType', (o) => o.objectType], ['shape', (o) => o.shape && o.shape.type], ['parentSlot', (o) => o.parentSlot],
];
function toCSV(objects, opts) {
  const safe = !(opts && opts.safe === false);
  const cols = (opts && opts.columns) ? CSV_COLUMNS.filter((c) => opts.columns.includes(c[0])) : CSV_COLUMNS;
  const cell = (v) => {
    if (v === null || v === undefined) return '';
    if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
    let s = String(v);
    if (safe && typeof v === 'string' && /^[=+\-@\t\r]/.test(s)) s = "'" + s;
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [cols.map((c) => c[0]).join(',')];
  for (const o of objects) lines.push(cols.map((c) => cell(c[1](o))).join(','));
  return lines.join('\r\n') + '\r\n';
}
function toJSON(parsed, space) { return JSON.stringify(parsed, null, space); }

const HmsMvar = {
  version: VERSION,
  games: ['reach', 'halo3', 'halo4', 'h2a'],
  Error: HmsMvarError,
  parse, parseFile, withPalette, filterObjects, countBy, toCSV, toJSON,
};

if (typeof globalThis !== 'undefined') globalThis.HmsMvar = HmsMvar;

export { HmsMvarError as Error, parse, parseFile, withPalette, filterObjects, countBy, toCSV, toJSON, VERSION as version };
export default HmsMvar;
