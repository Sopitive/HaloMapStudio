//! FSB5 (FMOD Sample Bank, version 5) container.
//!
//! Layout, as measured on Reach MCC's banks:
//!
//! ```text
//!   0x00  "FSB5"
//!   0x04  u32 version                    (1)
//!   0x08  u32 sample count
//!   0x0c  u32 sample header region size
//!   0x10  u32 name table size            (0 if the bank is unnamed)
//!   0x14  u32 sample data size
//!   0x18  u32 codec                      (15 = FMOD Vorbis)
//!   0x1c  u32 flags
//!   0x20  12 bytes zero
//!   0x2c  16-byte hash
//!   0x3c  sample headers, then the name table, then the sample data
//! ```
//!
//! `header + sample headers + names + data` accounts for the whole file with
//! no slack, which is the check this parser makes.
//!
//! A sample header is a u64 of packed fields followed by an optional chunk
//! list; see `Sample`.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 4] = b"FSB5";
pub const BASE_HEADER_V1: usize = 0x3c;
pub const BASE_HEADER_V0: usize = 0x40;

/// FMOD_SOUND_FORMAT values an FSB5 bank can declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Vorbis,
    FadPcm,
    Pcm16,
    Other(u32),
}

impl Codec {
    pub fn from_raw(v: u32) -> Self {
        match v {
            2 => Codec::Pcm16,
            15 => Codec::Vorbis,
            16 => Codec::FadPcm,
            other => Codec::Other(other),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Codec::Pcm16 => "PCM16",
            Codec::Vorbis => "VORBIS",
            Codec::FadPcm => "FADPCM",
            Codec::Other(_) => "unknown",
        }
    }
}

/// The frequency table the 4-bit sample-header field indexes.
const FREQ_TABLE: [u32; 11] = [
    0, 8000, 11000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
];

/// Chunk kinds that appear in a sample header's chunk list.
pub const CHUNK_CHANNELS: u32 = 1;
pub const CHUNK_FREQUENCY: u32 = 2;
pub const CHUNK_LOOP: u32 = 3;
pub const CHUNK_VORBIS_DATA: u32 = 11;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub kind: u32,
    /// Offset of the chunk's payload within the bank's sample-header region.
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub index: usize,
    /// Channel count as the 2-bit header field encodes it. FMOD maps the four
    /// values to 1, 2, 6 and 8 channels; the `.fsb.info` sidecar carries the
    /// true count and should be preferred when it is available, because a
    /// four-channel sample reports 1 here.
    pub channels: u8,
    pub sample_rate: u32,
    /// Byte offset of this sample's payload within the bank's data region.
    pub data_offset: u64,
    /// Length of the payload in bytes, derived from the next sample's offset.
    pub data_len: u64,
    /// Frame count the bank declares. A decoder must trim its output to this.
    pub frames: u32,
    pub chunks: Vec<Chunk>,
}

impl Sample {
    /// The crc32 "setup id" from this sample's VORBISDATA chunk, which selects
    /// a Vorbis setup header from the game module's table.
    pub fn vorbis_setup_id(&self, header_region: &[u8]) -> Option<u32> {
        let c = self.chunks.iter().find(|c| c.kind == CHUNK_VORBIS_DATA)?;
        if c.size < 4 {
            return None;
        }
        let b = header_region.get(c.offset..c.offset + 4)?;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    }
}

pub struct Bank {
    pub path: PathBuf,
    map: memmap2::Mmap,
    pub version: u32,
    pub codec: Codec,
    pub flags: u32,
    pub hash: [u8; 16],
    pub samples: Vec<Sample>,
    pub names: Vec<String>,
    header_region: Vec<u8>,
    data_start: usize,
    pub data_size: u64,
}

impl std::fmt::Debug for Bank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately a summary: the sample table has tens of thousands of
        // entries and the mapping covers hundreds of megabytes.
        f.debug_struct("Bank")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("codec", &self.codec.name())
            .field("samples", &self.samples.len())
            .field("named", &!self.names.is_empty())
            .field("data_size", &self.data_size)
            .finish()
    }
}

fn rd_u32(b: &[u8], at: usize) -> Result<u32> {
    let s = b
        .get(at..at + 4)
        .with_context(|| format!("truncated reading u32 at 0x{at:x}"))?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn rd_u64(b: &[u8], at: usize) -> Result<u64> {
    let s = b
        .get(at..at + 8)
        .with_context(|| format!("truncated reading u64 at 0x{at:x}"))?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

impl Bank {
    pub fn open(path: &Path) -> Result<Bank> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // The sfx bank is 712 MB; map it rather than reading it in.
        let map = unsafe { memmap2::Mmap::map(&f) }
            .with_context(|| format!("mmap {}", path.display()))?;
        Bank::from_mmap(path.to_path_buf(), map)
    }

    fn from_mmap(path: PathBuf, map: memmap2::Mmap) -> Result<Bank> {
        let d: &[u8] = &map;
        if d.len() < BASE_HEADER_V0 {
            bail!("{}: {} bytes is too small for an FSB5 header", path.display(), d.len());
        }
        if &d[..4] != MAGIC {
            bail!("{}: not an FSB5 bank (magic {:02x?})", path.display(), &d[..4]);
        }
        let version = rd_u32(d, 0x04)?;
        let count = rd_u32(d, 0x08)? as usize;
        let header_size = rd_u32(d, 0x0c)? as usize;
        let name_size = rd_u32(d, 0x10)? as usize;
        let data_size = rd_u32(d, 0x14)? as u64;
        let codec = Codec::from_raw(rd_u32(d, 0x18)?);
        let flags = rd_u32(d, 0x1c)?;
        let mut hash = [0u8; 16];
        hash.copy_from_slice(&d[0x2c..0x3c]);

        if version > 1 {
            bail!("{}: FSB version {version} is not supported", path.display());
        }
        let base = if version == 0 { BASE_HEADER_V0 } else { BASE_HEADER_V1 };

        // Every region must fit, and together they must account for the file.
        let total = (base as u64)
            .checked_add(header_size as u64)
            .and_then(|v| v.checked_add(name_size as u64))
            .and_then(|v| v.checked_add(data_size))
            .with_context(|| format!("{}: region sizes overflow", path.display()))?;
        if total != d.len() as u64 {
            bail!(
                "{}: regions do not account for the file: {base} + {header_size} + {name_size} + \
                 {data_size} = {total}, file is {}",
                path.display(),
                d.len()
            );
        }
        if count == 0 {
            bail!("{}: bank declares no samples", path.display());
        }
        // A sample header is at least 8 bytes.
        if header_size < count.saturating_mul(8) {
            bail!(
                "{}: sample header region is {header_size} bytes, too small for {count} samples",
                path.display()
            );
        }

        // The "regions account for the file" check above already guarantees
        // these fit, but slice through `get` anyway: that check is the only
        // thing standing between a corrupt header and a panic here, and a
        // single guard is not where a parser of untrusted data should rest.
        let header_region = d
            .get(base..base + header_size)
            .with_context(|| format!("{}: sample header region is outside the file", path.display()))?
            .to_vec();
        let name_region = d
            .get(base + header_size..base + header_size + name_size)
            .with_context(|| format!("{}: name table is outside the file", path.display()))?;
        let data_start = base + header_size + name_size;

        let mut samples = Vec::with_capacity(count);
        let mut p = 0usize;
        for index in 0..count {
            let raw = rd_u64(&header_region, p)
                .with_context(|| format!("{}: sample {index} header", path.display()))?;
            p += 8;
            let freq_idx = ((raw >> 1) & 0xf) as usize;
            let sample_rate = *FREQ_TABLE
                .get(freq_idx)
                .filter(|v| **v != 0)
                .with_context(|| format!("{}: sample {index} frequency index {freq_idx}", path.display()))?;
            let channels = match (raw >> 5) & 0x3 {
                0 => 1,
                1 => 2,
                2 => 6,
                _ => 8,
            };
            let data_offset = ((raw >> 7) & 0x07ff_ffff) * 32;
            let frames = ((raw >> 34) & 0x3fff_ffff) as u32;

            let mut chunks = Vec::new();
            let mut more = raw & 1 == 1;
            while more {
                let c = rd_u32(&header_region, p)
                    .with_context(|| format!("{}: sample {index} chunk header", path.display()))?;
                p += 4;
                more = c & 1 == 1;
                let kind = (c >> 25) & 0x7f;
                let size = ((c >> 1) & 0xff_ffff) as usize;
                if p + size > header_region.len() {
                    bail!(
                        "{}: sample {index} chunk kind {kind} of {size} bytes runs past the \
                         sample header region",
                        path.display()
                    );
                }
                chunks.push(Chunk { kind, offset: p, size });
                p += size;
            }
            if data_offset > data_size {
                bail!(
                    "{}: sample {index} starts at 0x{data_offset:x}, past the {data_size}-byte \
                     data region",
                    path.display()
                );
            }
            samples.push(Sample {
                index,
                channels,
                sample_rate,
                data_offset,
                data_len: 0,
                frames,
                chunks,
            });
        }
        if p != header_size {
            bail!(
                "{}: sample headers consumed {p} of {header_size} bytes",
                path.display()
            );
        }

        // Payload length comes from the next sample's offset.
        for i in 0..samples.len() {
            let end = if i + 1 < samples.len() {
                samples[i + 1].data_offset
            } else {
                data_size
            };
            if end < samples[i].data_offset {
                bail!("{}: sample offsets are not monotonic at {i}", path.display());
            }
            samples[i].data_len = end - samples[i].data_offset;
        }

        let mut names = Vec::new();
        if name_size > 0 {
            if name_size < count * 4 {
                bail!(
                    "{}: name table is {name_size} bytes, too small for {count} offsets",
                    path.display()
                );
            }
            for i in 0..count {
                let off = rd_u32(name_region, i * 4)? as usize;
                let tail = name_region
                    .get(off..)
                    .with_context(|| format!("{}: name {i} offset {off} out of range", path.display()))?;
                let end = tail.iter().position(|b| *b == 0).unwrap_or(tail.len());
                names.push(String::from_utf8_lossy(&tail[..end]).into_owned());
            }
        }

        Ok(Bank {
            path,
            map,
            version,
            codec,
            flags,
            hash,
            samples,
            names,
            header_region,
            data_start,
            data_size,
        })
    }

    pub fn sample_header_region(&self) -> &[u8] {
        &self.header_region
    }

    /// The raw, still-encoded payload of one sample.
    pub fn sample_data(&self, index: usize) -> Result<&[u8]> {
        let s = self
            .samples
            .get(index)
            .with_context(|| format!("sample {index} of {}", self.samples.len()))?;
        let a = self.data_start + s.data_offset as usize;
        let b = a + s.data_len as usize;
        self.map
            .get(a..b)
            .with_context(|| format!("sample {index} payload is outside the bank"))
    }

    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(|s| s.as_str())
    }
}

/// Split an FMOD Vorbis payload into its Vorbis packets.
///
/// The payload is a run of `[u16 little-endian length][packet]` records. A
/// zero length, or a length that runs past the end, ends the run: Reach's
/// banks leave up to a few bytes of alignment padding after the last packet.
pub fn split_vorbis_packets(payload: &[u8]) -> (Vec<&[u8]>, usize) {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 2 <= payload.len() {
        let n = u16::from_le_bytes([payload[o], payload[o + 1]]) as usize;
        o += 2;
        if n == 0 || o + n > payload.len() {
            o -= 2;
            break;
        }
        out.push(&payload[o..o + n]);
        o += n;
    }
    (out, o)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the smallest well-formed FSB5 bank: one PCM16 sample, no names.
    fn synthetic() -> Vec<u8> {
        let data: Vec<u8> = (0..64u32).map(|i| i as u8).collect();
        let mut hdr = Vec::new();
        // mode: no chunks, freq index 9 (48 kHz), 2 channels, offset 0, 100 frames
        let raw: u64 = 0 | (9u64 << 1) | (1u64 << 5) | (0u64 << 7) | (100u64 << 34);
        hdr.extend_from_slice(&raw.to_le_bytes());
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes()); // version
        b.extend_from_slice(&1u32.to_le_bytes()); // samples
        b.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // no names
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes()); // PCM16
        b.extend_from_slice(&0u32.to_le_bytes()); // flags
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(&[0xAAu8; 16]);
        assert_eq!(b.len(), BASE_HEADER_V1);
        b.extend_from_slice(&hdr);
        b.extend_from_slice(&data);
        b
    }

    fn open_bytes(b: &[u8]) -> Result<Bank> {
        let dir = std::env::temp_dir().join(format!(
            "hms-sound-test-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir)?;
        let p = dir.join("b.fsb");
        std::fs::write(&p, b)?;
        let r = Bank::open(&p);
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_dir(&dir);
        r
    }

    #[test]
    fn a_well_formed_bank_parses() {
        let bank = open_bytes(&synthetic()).expect("should parse");
        assert_eq!(bank.version, 1);
        assert_eq!(bank.codec, Codec::Pcm16);
        assert_eq!(bank.samples.len(), 1);
        let s = &bank.samples[0];
        assert_eq!(s.sample_rate, 48_000);
        assert_eq!(s.channels, 2);
        assert_eq!(s.frames, 100);
        assert_eq!(s.data_len, 64);
        assert_eq!(bank.sample_data(0).unwrap().len(), 64);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut b = synthetic();
        b[0] = b'X';
        assert!(open_bytes(&b).is_err());
    }

    #[test]
    fn a_truncated_bank_is_rejected_not_panicked_on() {
        let full = synthetic();
        // Every prefix must fail cleanly. The last byte removed already
        // breaks the "regions account for the file" check.
        for cut in 0..full.len() {
            let r = open_bytes(&full[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes should not parse");
        }
    }

    #[test]
    fn a_lying_sample_count_is_rejected() {
        let mut b = synthetic();
        b[0x08..0x0c].copy_from_slice(&1_000_000u32.to_le_bytes());
        let e = open_bytes(&b).expect_err("should reject");
        assert!(
            format!("{e:#}").contains("too small for"),
            "unexpected error: {e:#}"
        );
    }

    #[test]
    fn a_sample_offset_past_the_data_region_is_rejected() {
        let mut b = synthetic();
        // Put the sample's data offset far beyond the 64-byte data region.
        let raw: u64 = 0 | (9u64 << 1) | (1u64 << 5) | (4096u64 << 7) | (100u64 << 34);
        b[BASE_HEADER_V1..BASE_HEADER_V1 + 8].copy_from_slice(&raw.to_le_bytes());
        let e = open_bytes(&b).expect_err("should reject");
        assert!(format!("{e:#}").contains("past the"), "unexpected error: {e:#}");
    }

    #[test]
    fn a_chunk_running_past_the_header_region_is_rejected() {
        // One sample whose header claims a chunk far larger than the region.
        let raw: u64 = 1 | (9u64 << 1) | (1u64 << 5) | (100u64 << 34);
        let chunk: u32 = (CHUNK_VORBIS_DATA << 25) | (0xffff << 1);
        let mut hdr = raw.to_le_bytes().to_vec();
        hdr.extend_from_slice(&chunk.to_le_bytes());
        let data = vec![0u8; 32];
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&15u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(&[0u8; 16]);
        b.extend_from_slice(&hdr);
        b.extend_from_slice(&data);
        let e = open_bytes(&b).expect_err("should reject");
        assert!(
            format!("{e:#}").contains("runs past"),
            "unexpected error: {e:#}"
        );
    }

    #[test]
    fn packet_splitting_stops_at_padding_rather_than_reading_past_the_end() {
        // two packets of 3 bytes, then 2 bytes of padding
        let payload = [3, 0, 1, 2, 3, 3, 0, 4, 5, 6, 0, 0];
        let (pk, consumed) = split_vorbis_packets(&payload);
        assert_eq!(pk.len(), 2);
        assert_eq!(pk[0], &[1, 2, 3]);
        assert_eq!(pk[1], &[4, 5, 6]);
        assert_eq!(consumed, 10);
    }

    #[test]
    fn a_packet_length_that_overruns_the_payload_is_not_read() {
        let payload = [200, 0, 1, 2, 3];
        let (pk, consumed) = split_vorbis_packets(&payload);
        assert!(pk.is_empty());
        assert_eq!(consumed, 0);
    }
}
