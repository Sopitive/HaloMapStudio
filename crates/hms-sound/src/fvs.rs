//! FMOD's Vorbis setup-header table, read out of the game module.
//!
//! An FSB5 bank stores no Vorbis setup headers. Each sample's VORBISDATA
//! chunk instead holds a 32-bit crc32 "setup id", and FMOD looks that up in a
//! table its codec links in statically. In Reach MCC that table lives in
//! `haloreach.dll`: the string `codec_fsbvorbis.cpp` sits immediately before
//! the first setup packet, and `.rdata` holds 33 of them keyed by a flat array
//! of 16-byte records:
//!
//! ```text
//!   +0x00  u32  crc32 setup id
//!   +0x04  u64  virtual address of the setup packet
//!   +0x0c  u32  a length that is SHORT of the real packet
//! ```
//!
//! The recorded length is not usable: measured against a forward parse it is
//! 79 bytes short for the stereo header and 72 short for the mono one, both
//! times stopping inside the residue section. The real length is recovered by
//! parsing the packet (see `vorbis::parse_setup`).
//!
//! HMS therefore ships NO codebook data of its own. The headers are read from
//! the user's own installed `haloreach.dll`, which sits two directories above
//! the banks it decodes.

use crate::vorbis;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Keys below this are not crc32 values; the loose scan picks up a few
/// unrelated 16-byte windows whose +0x04 happens to hold a live pointer, and
/// those always have a small leading u32.
const MIN_PLAUSIBLE_KEY: u32 = 0x1_0000;

/// No real setup packet approaches this. Caps the window handed to the parser.
const MAX_SETUP_LEN: usize = 64 * 1024;

struct Section {
    virtual_address: u64,
    virtual_size: u64,
    raw_offset: usize,
    raw_size: usize,
}

struct Pe {
    image_base: u64,
    sections: Vec<Section>,
}

impl Pe {
    fn parse(d: &[u8]) -> Result<Pe> {
        let lo = |at: usize, n: usize| -> Result<u64> {
            let s = d
                .get(at..at + n)
                .with_context(|| format!("PE: truncated at 0x{at:x}"))?;
            let mut v = 0u64;
            for (i, b) in s.iter().enumerate() {
                v |= (*b as u64) << (8 * i);
            }
            Ok(v)
        };
        if d.len() < 0x40 || &d[..2] != b"MZ" {
            bail!("not a PE image");
        }
        let e = lo(0x3c, 4)? as usize;
        if d.get(e..e + 4) != Some(b"PE\0\0") {
            bail!("PE signature not found");
        }
        let nsec = lo(e + 6, 2)? as usize;
        let optsz = lo(e + 20, 2)? as usize;
        let magic = lo(e + 24, 2)?;
        let image_base = match magic {
            0x20b => lo(e + 24 + 24, 8)?,
            0x10b => lo(e + 24 + 28, 4)?,
            other => bail!("unknown optional header magic 0x{other:x}"),
        };
        let table = e + 24 + optsz;
        if nsec == 0 || nsec > 96 {
            bail!("PE declares {nsec} sections");
        }
        let mut sections = Vec::with_capacity(nsec);
        for i in 0..nsec {
            let o = table + 40 * i;
            sections.push(Section {
                virtual_size: lo(o + 8, 4)?,
                virtual_address: lo(o + 12, 4)?,
                raw_size: lo(o + 16, 4)? as usize,
                raw_offset: lo(o + 20, 4)? as usize,
            });
        }
        Ok(Pe { image_base, sections })
    }

    fn va_to_offset(&self, va: u64) -> Option<usize> {
        let rva = va.checked_sub(self.image_base)?;
        for s in &self.sections {
            let span = s.virtual_size.max(s.raw_size as u64);
            if rva >= s.virtual_address && rva < s.virtual_address + span {
                let delta = (rva - s.virtual_address) as usize;
                if delta < s.raw_size {
                    return Some(s.raw_offset + delta);
                }
            }
        }
        None
    }
}

/// Vorbis setup headers extracted from a game module, keyed by setup id.
pub struct SetupTable {
    headers: HashMap<u32, Vec<u8>>,
}

impl std::fmt::Debug for SetupTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ids: Vec<String> = self.headers.keys().map(|k| format!("0x{k:08X}")).collect();
        ids.sort();
        f.debug_struct("SetupTable").field("ids", &ids).finish()
    }
}

impl SetupTable {
    /// Read the table out of a module such as `haloreach.dll`.
    pub fn from_module(path: &Path) -> Result<SetupTable> {
        let d = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        SetupTable::from_module_bytes(&d).with_context(|| format!("scan {}", path.display()))
    }

    pub fn from_module_bytes(d: &[u8]) -> Result<SetupTable> {
        let pe = Pe::parse(d)?;
        let mut headers: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut conflicts = 0usize;

        for s in &pe.sections {
            let end = s.raw_offset.saturating_add(s.raw_size).min(d.len());
            if end <= s.raw_offset + 16 {
                continue;
            }
            let mut o = s.raw_offset;
            while o + 16 <= end {
                let key = u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
                if key >= MIN_PLAUSIBLE_KEY {
                    let va = u64::from_le_bytes(d[o + 4..o + 12].try_into().unwrap());
                    if let Some(target) = pe.va_to_offset(va) {
                        if d.get(target..target + 7) == Some(b"\x05vorbis") {
                            let window = &d[target..(target + MAX_SETUP_LEN).min(d.len())];
                            // Channel count only affects a mapping that
                            // declares coupling; try stereo then mono.
                            let len = vorbis::setup_packet_len(window, 2)
                                .or_else(|_| vorbis::setup_packet_len(window, 1));
                            if let Ok(len) = len {
                                let blob = window[..len].to_vec();
                                match headers.get(&key) {
                                    Some(prev) if *prev != blob => conflicts += 1,
                                    Some(_) => {}
                                    None => {
                                        headers.insert(key, blob);
                                    }
                                }
                            }
                        }
                    }
                }
                o += 4;
            }
        }
        if headers.is_empty() {
            bail!("no FMOD Vorbis setup headers found in this module");
        }
        let _ = conflicts;
        Ok(SetupTable { headers })
    }

    pub fn get(&self, setup_id: u32) -> Option<&[u8]> {
        self.headers.get(&setup_id).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.headers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.headers.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_pe_is_rejected() {
        assert!(SetupTable::from_module_bytes(b"not a pe at all").is_err());
        assert!(SetupTable::from_module_bytes(&[]).is_err());
    }

    #[test]
    fn an_mz_with_no_pe_header_is_rejected() {
        let mut d = vec![0u8; 0x200];
        d[0] = b'M';
        d[1] = b'Z';
        d[0x3c] = 0x80; // points at zeroes, so no "PE\0\0"
        assert!(SetupTable::from_module_bytes(&d).is_err());
    }

    #[test]
    fn a_pe_with_no_setup_headers_reports_that_rather_than_returning_empty() {
        // Minimal PE32+ with one empty section.
        let mut d = vec![0u8; 0x400];
        d[0] = b'M';
        d[1] = b'Z';
        let e = 0x80usize;
        d[0x3c..0x40].copy_from_slice(&(e as u32).to_le_bytes());
        d[e..e + 4].copy_from_slice(b"PE\0\0");
        d[e + 6..e + 8].copy_from_slice(&1u16.to_le_bytes()); // one section
        d[e + 20..e + 22].copy_from_slice(&0xf0u16.to_le_bytes()); // optional size
        d[e + 24..e + 26].copy_from_slice(&0x20bu16.to_le_bytes());
        d[e + 24 + 24..e + 24 + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes());
        let t = e + 24 + 0xf0;
        d[t + 8..t + 12].copy_from_slice(&0x100u32.to_le_bytes()); // vsize
        d[t + 12..t + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // va
        d[t + 16..t + 20].copy_from_slice(&0x100u32.to_le_bytes()); // raw size
        d[t + 20..t + 24].copy_from_slice(&0x300u32.to_le_bytes()); // raw offset
        let e2 = SetupTable::from_module_bytes(&d).expect_err("should report none found");
        assert!(
            format!("{e2:#}").contains("no FMOD Vorbis setup headers"),
            "unexpected error: {e2:#}"
        );
    }
}
