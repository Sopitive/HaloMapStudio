//! Vorbis I header handling, for the parts FMOD leaves out of an FSB5 bank.
//!
//! An FSB5 bank stores only the *audio* packets. The three Vorbis header
//! packets are missing: FMOD rebuilds the identification and comment headers
//! from the bank's own fields, and keeps the setup header (the codebooks) in a
//! table inside the game module, keyed by a crc32 "setup id" that each sample
//! carries. See `fvs.rs` for that table.
//!
//! This module therefore provides:
//!   * `identification` / `comment` — synthesise the two trivial headers.
//!   * `setup_packet_len` — a forward, spec-exact parse of a setup header that
//!     reports its true length. That is needed because the size recorded
//!     alongside the pointer in the game module's table is SHORT of the real
//!     packet (measured: 79 bytes short for the stereo header, 72 for the
//!     mono one), so the length has to be recovered by parsing.

use crate::bits::{ilog, BitReader};
use anyhow::{anyhow, bail, Result};

/// Block sizes used by every sample in Reach's banks: 256 and 2048 frames.
///
/// Established by measurement, not assumption: for 400 randomly chosen samples
/// the frame count implied by the per-packet mode bits under 256/2048 exceeded
/// the bank's own declared frame count by 0..1016 frames in every case — always
/// non-negative and always less than one long block, which is exactly the
/// relationship a final-granule trim produces. No other exponent pair in
/// 2^6..2^13 holds that property.
pub const BLOCKSIZE_SHORT_EXP: u8 = 8;
pub const BLOCKSIZE_LONG_EXP: u8 = 11;
pub const BLOCKSIZE_SHORT: u32 = 1 << BLOCKSIZE_SHORT_EXP;
pub const BLOCKSIZE_LONG: u32 = 1 << BLOCKSIZE_LONG_EXP;

/// Build the 30-byte Vorbis identification header.
pub fn identification(channels: u8, sample_rate: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(30);
    v.extend_from_slice(b"\x01vorbis");
    v.extend_from_slice(&0u32.to_le_bytes()); // vorbis_version
    v.push(channels);
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&0i32.to_le_bytes()); // bitrate_maximum
    v.extend_from_slice(&0i32.to_le_bytes()); // bitrate_nominal
    v.extend_from_slice(&0i32.to_le_bytes()); // bitrate_minimum
    v.push(BLOCKSIZE_SHORT_EXP | (BLOCKSIZE_LONG_EXP << 4));
    v.push(0x01); // framing bit
    v
}

/// Build a minimal Vorbis comment header.
pub fn comment() -> Vec<u8> {
    const VENDOR: &[u8] = b"hms-sound";
    let mut v = Vec::with_capacity(16 + VENDOR.len());
    v.extend_from_slice(b"\x03vorbis");
    v.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
    v.extend_from_slice(VENDOR);
    v.extend_from_slice(&0u32.to_le_bytes()); // comment count
    v.push(0x01); // framing bit
    v
}

/// What a setup header says about the stream it describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupHeader {
    /// True length of the setup packet in bytes.
    pub len: usize,
    pub codebooks: u32,
    pub floor_types: Vec<u16>,
    pub residue_types: Vec<u16>,
    /// `blockflag` for each mode, in order: false = short block, true = long.
    pub mode_blockflags: Vec<bool>,
    /// Stereo coupling steps declared by the first mapping. Zero means the
    /// header is channel-count agnostic and may be shared by mono, quad and
    /// 5.1 samples, which is exactly what Reach's banks do.
    pub coupling_steps: u32,
}

impl SetupHeader {
    /// The block size, in frames, selected by a packet's mode number.
    pub fn blocksize_for_mode(&self, mode: usize) -> Option<u32> {
        match self.mode_blockflags.get(mode)? {
            true => Some(BLOCKSIZE_LONG),
            false => Some(BLOCKSIZE_SHORT),
        }
    }

    /// Number of bits used to encode a packet's mode number.
    pub fn mode_number_bits(&self) -> u32 {
        ilog(self.mode_blockflags.len().saturating_sub(1) as u32)
    }
}

/// A codebook has at most 2^24 entries; a per-entry loop over that many is a
/// denial of service on a hostile file, so cap it at something no real
/// codebook approaches. Reach's largest has 6561 entries.
const MAX_CODEBOOK_ENTRIES: u32 = 1 << 20;

fn lookup1_values(entries: u32, dim: u16) -> u32 {
    if dim == 0 {
        return 0;
    }
    let mut n: u32 = 0;
    loop {
        let next = n + 1;
        let mut p: u64 = 1;
        let mut overflow = false;
        for _ in 0..dim {
            p = p.saturating_mul(next as u64);
            if p > entries as u64 {
                overflow = true;
                break;
            }
        }
        if overflow {
            return n;
        }
        n = next;
        if n > entries {
            return entries;
        }
    }
}

fn codebook(r: &mut BitReader) -> Result<()> {
    let sync = r.read(24).ok_or_else(|| anyhow!("codebook: truncated"))?;
    if sync != 0x564342 {
        bail!("codebook sync is 0x{sync:06x}, expected 0x564342");
    }
    let dim = r.read(16).ok_or_else(|| anyhow!("codebook: truncated"))? as u16;
    let entries = r.read(24).ok_or_else(|| anyhow!("codebook: truncated"))?;
    if entries > MAX_CODEBOOK_ENTRIES {
        bail!("codebook declares {entries} entries, refusing");
    }
    let ordered = r.read(1).ok_or_else(|| anyhow!("codebook: truncated"))?;
    if ordered == 0 {
        let sparse = r.read(1).ok_or_else(|| anyhow!("codebook: truncated"))?;
        for _ in 0..entries {
            if sparse == 1 {
                if r.read(1).ok_or_else(|| anyhow!("codebook: truncated"))? == 1 {
                    r.read(5).ok_or_else(|| anyhow!("codebook: truncated"))?;
                }
            } else {
                r.read(5).ok_or_else(|| anyhow!("codebook: truncated"))?;
            }
        }
    } else {
        let mut cur: u32 = 0;
        r.read(5).ok_or_else(|| anyhow!("codebook: truncated"))?;
        while cur < entries {
            let n = r
                .read(ilog(entries - cur))
                .ok_or_else(|| anyhow!("codebook: truncated"))?;
            if n == 0 {
                bail!("ordered codebook makes no progress");
            }
            cur = cur.saturating_add(n);
        }
    }
    let lookup = r.read(4).ok_or_else(|| anyhow!("codebook: truncated"))?;
    match lookup {
        0 => {}
        1 | 2 => {
            r.read(32).ok_or_else(|| anyhow!("codebook: truncated"))?;
            r.read(32).ok_or_else(|| anyhow!("codebook: truncated"))?;
            let value_bits = r.read(4).ok_or_else(|| anyhow!("codebook: truncated"))? + 1;
            r.read(1).ok_or_else(|| anyhow!("codebook: truncated"))?;
            let n = if lookup == 1 {
                lookup1_values(entries, dim)
            } else {
                (entries as u64)
                    .saturating_mul(dim as u64)
                    .min(MAX_CODEBOOK_ENTRIES as u64) as u32
            };
            r.skip(n as usize * value_bits as usize)
                .ok_or_else(|| anyhow!("codebook: truncated lookup table"))?;
        }
        other => bail!("codebook lookup type {other}"),
    }
    Ok(())
}

/// Parse a Vorbis setup header, returning what it describes and its true
/// length. `buf` may be longer than the packet; everything past the framing
/// bit is ignored and the reported `len` is the real packet length.
pub fn parse_setup(buf: &[u8], channels: u8) -> Result<SetupHeader> {
    if buf.len() < 7 || &buf[..7] != b"\x05vorbis" {
        bail!("not a Vorbis setup packet");
    }
    if channels == 0 {
        bail!("channel count of zero");
    }
    let mut r = BitReader::new(buf);
    r.seek_bits(7 * 8);
    let trunc = || anyhow!("setup header: truncated");

    let codebooks = r.read(8).ok_or_else(trunc)? + 1;
    for i in 0..codebooks {
        codebook(&mut r).map_err(|e| anyhow!("codebook {i}: {e}"))?;
    }

    let times = r.read(6).ok_or_else(trunc)? + 1;
    for _ in 0..times {
        if r.read(16).ok_or_else(trunc)? != 0 {
            bail!("time domain transform must be 0");
        }
    }

    let nfloors = r.read(6).ok_or_else(trunc)? + 1;
    let mut floor_types = Vec::with_capacity(nfloors as usize);
    for _ in 0..nfloors {
        let ft = r.read(16).ok_or_else(trunc)? as u16;
        floor_types.push(ft);
        match ft {
            0 => {
                r.skip(8 + 16 + 16 + 6 + 8).ok_or_else(trunc)?;
                let books = r.read(4).ok_or_else(trunc)? + 1;
                r.skip(books as usize * 8).ok_or_else(trunc)?;
            }
            1 => {
                let parts = r.read(5).ok_or_else(trunc)?;
                let mut class_list = Vec::with_capacity(parts as usize);
                for _ in 0..parts {
                    class_list.push(r.read(4).ok_or_else(trunc)? as usize);
                }
                let max_class = class_list.iter().copied().max().map(|m| m + 1).unwrap_or(0);
                let mut dims = vec![0usize; max_class];
                for c in 0..max_class {
                    dims[c] = r.read(3).ok_or_else(trunc)? as usize + 1;
                    let sub = r.read(2).ok_or_else(trunc)?;
                    if sub != 0 {
                        r.read(8).ok_or_else(trunc)?;
                    }
                    r.skip((1usize << sub) * 8).ok_or_else(trunc)?;
                }
                r.read(2).ok_or_else(trunc)?; // multiplier - 1
                let rangebits = r.read(4).ok_or_else(trunc)?;
                for &c in &class_list {
                    let d = *dims.get(c).ok_or_else(|| anyhow!("floor1 class {c}"))?;
                    r.skip(d * rangebits as usize).ok_or_else(trunc)?;
                }
            }
            other => bail!("floor type {other}"),
        }
    }

    let nres = r.read(6).ok_or_else(trunc)? + 1;
    let mut residue_types = Vec::with_capacity(nres as usize);
    for _ in 0..nres {
        let rt = r.read(16).ok_or_else(trunc)? as u16;
        if rt > 2 {
            bail!("residue type {rt}");
        }
        residue_types.push(rt);
        r.skip(24 + 24 + 24).ok_or_else(trunc)?; // begin, end, partition size
        let classes = r.read(6).ok_or_else(trunc)? + 1;
        r.read(8).ok_or_else(trunc)?; // classbook
        let mut cascade = Vec::with_capacity(classes as usize);
        for _ in 0..classes {
            let low = r.read(3).ok_or_else(trunc)?;
            let high = if r.read(1).ok_or_else(trunc)? == 1 {
                r.read(5).ok_or_else(trunc)?
            } else {
                0
            };
            cascade.push(high * 8 + low);
        }
        for c in cascade {
            for j in 0..8 {
                if c & (1 << j) != 0 {
                    r.read(8).ok_or_else(trunc)?;
                }
            }
        }
    }

    let nmaps = r.read(6).ok_or_else(trunc)? + 1;
    let mut coupling_steps = 0;
    for m in 0..nmaps {
        if r.read(16).ok_or_else(trunc)? != 0 {
            bail!("mapping type must be 0");
        }
        let submaps = if r.read(1).ok_or_else(trunc)? == 1 {
            r.read(4).ok_or_else(trunc)? + 1
        } else {
            1
        };
        let steps = if r.read(1).ok_or_else(trunc)? == 1 {
            let steps = r.read(8).ok_or_else(trunc)? + 1;
            let w = ilog(channels as u32 - 1);
            r.skip(steps as usize * 2 * w as usize).ok_or_else(trunc)?;
            steps
        } else {
            0
        };
        if m == 0 {
            coupling_steps = steps;
        }
        if r.read(2).ok_or_else(trunc)? != 0 {
            bail!("mapping reserved bits must be 0");
        }
        if submaps > 1 {
            r.skip(channels as usize * 4).ok_or_else(trunc)?;
        }
        r.skip(submaps as usize * 24).ok_or_else(trunc)?; // unused, floor, residue
    }

    let nmodes = r.read(6).ok_or_else(trunc)? + 1;
    let mut mode_blockflags = Vec::with_capacity(nmodes as usize);
    for _ in 0..nmodes {
        let blockflag = r.read(1).ok_or_else(trunc)? == 1;
        if r.read(16).ok_or_else(trunc)? != 0 || r.read(16).ok_or_else(trunc)? != 0 {
            bail!("mode window/transform type must be 0");
        }
        r.read(8).ok_or_else(trunc)?; // mapping
        mode_blockflags.push(blockflag);
    }
    if r.read(1).ok_or_else(trunc)? != 1 {
        bail!("setup header framing bit is not set");
    }

    Ok(SetupHeader {
        len: (r.bit_pos() + 7) / 8,
        codebooks,
        floor_types,
        residue_types,
        mode_blockflags,
        coupling_steps,
    })
}

/// Length in bytes of the setup packet at the start of `buf`.
pub fn setup_packet_len(buf: &[u8], channels: u8) -> Result<usize> {
    Ok(parse_setup(buf, channels)?.len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identification_is_thirty_bytes_and_carries_the_blocksizes() {
        let id = identification(2, 48_000);
        assert_eq!(id.len(), 30);
        assert_eq!(&id[..7], b"\x01vorbis");
        assert_eq!(id[11], 2); // channels
        assert_eq!(u32::from_le_bytes(id[12..16].try_into().unwrap()), 48_000);
        assert_eq!(id[28], 8 | (11 << 4));
        assert_eq!(id[29], 1);
    }

    #[test]
    fn a_non_setup_packet_is_rejected() {
        assert!(parse_setup(b"\x01vorbisxxxx", 2).is_err());
        assert!(parse_setup(b"", 2).is_err());
        assert!(parse_setup(b"\x05vor", 2).is_err());
    }

    #[test]
    fn zero_channels_is_rejected_rather_than_underflowing() {
        // ilog(channels - 1) would underflow on a 0 channel count.
        let mut blob = b"\x05vorbis".to_vec();
        blob.extend_from_slice(&[0u8; 64]);
        assert!(parse_setup(&blob, 0).is_err());
    }

    #[test]
    fn garbage_after_the_magic_errors_and_does_not_panic() {
        for seed in 0u32..64 {
            let mut blob = b"\x05vorbis".to_vec();
            let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            for _ in 0..512 {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                blob.push((x >> 16) as u8);
            }
            // Must return, either Ok or Err, but never panic or hang.
            let _ = parse_setup(&blob, 2);
        }
    }

    #[test]
    fn blocksize_lookup_follows_the_mode_table() {
        let h = SetupHeader {
            len: 0,
            codebooks: 1,
            floor_types: vec![1],
            residue_types: vec![2],
            mode_blockflags: vec![false, true],
            coupling_steps: 1,
        };
        assert_eq!(h.blocksize_for_mode(0), Some(256));
        assert_eq!(h.blocksize_for_mode(1), Some(2048));
        assert_eq!(h.blocksize_for_mode(2), None);
        assert_eq!(h.mode_number_bits(), 1);
    }
}
