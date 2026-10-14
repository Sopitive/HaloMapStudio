//! The `.fsb.info` sidecar that ships beside each bank.
//!
//! It is a flat array of fixed 280-byte records, one per bank sample, in bank
//! order. Measured on Reach MCC: `sfx.fsb.info` is 4,525,360 bytes = 16,162 x
//! 280, and the bank declares 16,162 samples; `english.fsb.info` is 8,801,240
//! = 31,433 x 280 against 31,433 samples.
//!
//! ```text
//!   +0x00  u32  subsound id         unique per sample; the key a map's
//!                                   `ugh!` permutation carries
//!   +0x04  u32  source PCM bytes    frames * channels * 2
//!   +0x08  u32  channels            TRUE channel count (1, 2, 4 or 6)
//!   +0x0c  u32  bytes per sample    always 2
//!   +0x10  u32  zero
//!   +0x14  u32  loop/related index  zero for most samples
//!   +0x18  char path[256]           original asset path, NUL padded,
//!                                   e.g. data\sound\weapons\assault_rifle\
//!                                        ar_fire\ar_reach1.aif
//! ```
//!
//! This is what turns a bank into something searchable: the bank's own name
//! table carries only a leaf name (`ar_reach1`), the sidecar carries the path.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub const RECORD_SIZE: usize = 280;
pub const PATH_OFFSET: usize = 0x18;
pub const PATH_CAPACITY: usize = RECORD_SIZE - PATH_OFFSET;

#[derive(Debug, Clone)]
pub struct InfoRecord {
    pub index: usize,
    /// The join key: a map's `fmod bank subsound id hash` equals this.
    pub subsound_id: u32,
    pub source_pcm_bytes: u32,
    pub channels: u32,
    pub bytes_per_sample: u32,
    pub path: String,
}

impl InfoRecord {
    /// Frame count implied by the source size and channel count.
    pub fn frames(&self) -> Option<u32> {
        let per_frame = self.channels.checked_mul(self.bytes_per_sample)?;
        if per_frame == 0 {
            return None;
        }
        Some(self.source_pcm_bytes / per_frame)
    }
}

pub struct BankInfo {
    pub records: Vec<InfoRecord>,
    by_id: HashMap<u32, usize>,
}

impl std::fmt::Debug for BankInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BankInfo")
            .field("records", &self.records.len())
            .finish()
    }
}

impl BankInfo {
    pub fn open(path: &Path) -> Result<BankInfo> {
        let d = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        BankInfo::parse(&d).with_context(|| format!("parse {}", path.display()))
    }

    pub fn parse(d: &[u8]) -> Result<BankInfo> {
        if d.is_empty() {
            bail!("sidecar is empty");
        }
        if d.len() % RECORD_SIZE != 0 {
            bail!(
                "sidecar is {} bytes, not a whole number of {RECORD_SIZE}-byte records \
                 ({} left over)",
                d.len(),
                d.len() % RECORD_SIZE
            );
        }
        let n = d.len() / RECORD_SIZE;
        let mut records = Vec::with_capacity(n);
        let mut by_id = HashMap::with_capacity(n);
        for index in 0..n {
            let r = &d[index * RECORD_SIZE..(index + 1) * RECORD_SIZE];
            let u = |at: usize| u32::from_le_bytes(r[at..at + 4].try_into().unwrap());
            let raw = &r[PATH_OFFSET..];
            let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
            let rec = InfoRecord {
                index,
                subsound_id: u(0x00),
                source_pcm_bytes: u(0x04),
                channels: u(0x08),
                bytes_per_sample: u(0x0c),
                path: String::from_utf8_lossy(&raw[..end]).into_owned(),
            };
            by_id.insert(rec.subsound_id, index);
            records.push(rec);
        }
        Ok(BankInfo { records, by_id })
    }

    /// Resolve a map's `fmod bank subsound id hash` to a bank sample index.
    pub fn index_of_subsound(&self, id: u32) -> Option<usize> {
        self.by_id.get(&id).copied()
    }

    pub fn get(&self, index: usize) -> Option<&InfoRecord> {
        self.records.get(index)
    }

    /// Every record whose path contains `needle`, compared case-insensitively.
    pub fn search(&self, needle: &str) -> Vec<&InfoRecord> {
        let n = needle.to_ascii_lowercase();
        self.records
            .iter()
            .filter(|r| r.path.to_ascii_lowercase().contains(&n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: u32, bytes: u32, channels: u32, path: &str) -> Vec<u8> {
        let mut r = vec![0u8; RECORD_SIZE];
        r[0x00..0x04].copy_from_slice(&id.to_le_bytes());
        r[0x04..0x08].copy_from_slice(&bytes.to_le_bytes());
        r[0x08..0x0c].copy_from_slice(&channels.to_le_bytes());
        r[0x0c..0x10].copy_from_slice(&2u32.to_le_bytes());
        let p = path.as_bytes();
        r[PATH_OFFSET..PATH_OFFSET + p.len()].copy_from_slice(p);
        r
    }

    #[test]
    fn records_parse_and_index() {
        let mut d = record(0x0434B84D, 140_692, 2, r"data\sound\a\b.aif");
        d.extend(record(0x0434B84E, 161_064, 2, r"data\sound\a\c.aif"));
        let info = BankInfo::parse(&d).unwrap();
        assert_eq!(info.records.len(), 2);
        assert_eq!(info.records[0].path, r"data\sound\a\b.aif");
        assert_eq!(info.records[0].frames(), Some(35_173));
        assert_eq!(info.index_of_subsound(0x0434B84E), Some(1));
        assert_eq!(info.index_of_subsound(0xDEAD_BEEF), None);
        assert_eq!(info.search("A\\C").len(), 1);
    }

    #[test]
    fn a_size_that_is_not_a_whole_number_of_records_is_rejected() {
        let mut d = record(1, 4, 1, "x");
        d.push(0); // one byte too many
        let e = BankInfo::parse(&d).expect_err("should reject");
        assert!(
            format!("{e:#}").contains("left over"),
            "unexpected error: {e:#}"
        );
        assert!(BankInfo::parse(&[]).is_err());
        assert!(BankInfo::parse(&[0u8; RECORD_SIZE - 1]).is_err());
    }

    #[test]
    fn a_path_with_no_terminator_is_read_to_the_record_edge_not_past_it() {
        let mut d = vec![0u8; RECORD_SIZE];
        for b in d[PATH_OFFSET..].iter_mut() {
            *b = b'A';
        }
        let info = BankInfo::parse(&d).unwrap();
        assert_eq!(info.records[0].path.len(), PATH_CAPACITY);
    }

    #[test]
    fn a_zero_channel_record_does_not_divide_by_zero() {
        let d = record(1, 1000, 0, "x");
        let info = BankInfo::parse(&d).unwrap();
        assert_eq!(info.records[0].frames(), None);
    }
}
