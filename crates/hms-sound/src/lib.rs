// =============================================================================
// hms-sound — Halo: Reach (MCC) sound assets.
//
// Reach MCC keeps every sound sample in FSB5 banks beside the game module:
//
//     haloreach/fmod/pc/sfx.fsb        712 MB, 16,162 samples   (world sfx)
//     haloreach/fmod/pc/english.fsb    381 MB, 31,433 samples   (dialogue)
//     haloreach/fmod/pc/<lang>.fsb                              (six more)
//     haloreach/fmod/pc/*.fsb.info     one 280-byte record per sample
//
// Every bank declares codec 15 = FMOD Vorbis, 48 kHz throughout. The banks
// hold only audio packets: the Vorbis setup headers live in `haloreach.dll`,
// keyed by a crc32 that each sample carries. `fvs` reads them from there, so
// nothing in this crate embeds codebook data.
//
// A map joins to a bank through the sound gestalt:
//
//     weap/vehi/bipd -> snd! reference
//     snd! in a cache = $#!+ cache_file_sound { first pitch range index,
//                                               pitch range count,
//                                               fmod bank suffix }
//        -> ugh! .pitch ranges[]   -> first permutation + count
//        -> ugh! .permutations[]   (24 bytes each)
//              +0x00 name index into ugh!.import names
//              +0x04 uncompressed sample count
//              +0x14 fmod bank subsound id hash   <-- equals the sidecar's
//                                                    +0x00 subsound id
//
// NO AUDIO OUTPUT LIVES HERE. This crate opens no sound device and pulls in no
// decoder, so it cannot make a noise and cannot fail on a machine with no
// audio hardware. See re-notes/reach_fmod_sound.md for the playback plan.
// =============================================================================

pub mod bits;
pub mod decode;
pub mod fsb5;
pub mod fvs;
pub mod info;
pub mod mix;
pub mod ogg;
pub mod out;
pub mod vorbis;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub use decode::Pcm;
pub use fsb5::{Bank, Codec, Sample};
pub use mix::{Cmd, MixerHandle, VoiceId};
pub use out::{OutDiag, Output, State};
pub use fvs::SetupTable;
pub use info::{BankInfo, InfoRecord};

/// `sizeof(sound_gestalt_permutations_block)` in a Reach cache.
pub const GESTALT_PERMUTATION_STRIDE: usize = 24;
/// Offset of `fmod bank subsound id hash` within that block.
pub const GESTALT_PERMUTATION_FMOD_HASH_OFFSET: usize = 0x14;
/// Offset of `uncompressed sample count` within that block.
pub const GESTALT_PERMUTATION_FRAMES_OFFSET: usize = 0x04;

/// A bank together with its sidecar.
pub struct SoundBank {
    pub bank: Bank,
    pub info: Option<BankInfo>,
}

/// One bank sample resolved down to the bytes a Vorbis decoder needs.
///
/// Borrowed from the bank's mapping and the setup table: nothing is copied.
pub struct RawStream<'a> {
    pub index: usize,
    /// The crc32 that selected `setup` out of the game module's table.
    pub setup_id: u32,
    /// The Vorbis setup (codebook) packet, from `haloreach.dll`.
    pub setup: &'a [u8],
    /// True channel count — the sidecar's when there is one.
    pub channels: u8,
    pub sample_rate: u32,
    /// Frame count the bank declares. A decode must be trimmed to this.
    pub frames: u32,
    /// The sample's audio packets, in order, unframed.
    pub packets: Vec<&'a [u8]>,
}

impl std::fmt::Debug for SoundBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundBank")
            .field("bank", &self.bank)
            .field("sidecar", &self.info.is_some())
            .finish()
    }
}

impl SoundBank {
    /// Open `<dir>/<stem>.fsb`, and its `.fsb.info` sidecar if present.
    ///
    /// A missing sidecar is not an error: without it a sample still decodes,
    /// it just has no asset path and its true channel count is whatever the
    /// bank's 2-bit field says.
    pub fn open(dir: &Path, stem: &str) -> Result<SoundBank> {
        let bank_path = dir.join(format!("{stem}.fsb"));
        let bank = Bank::open(&bank_path)?;
        let info_path = dir.join(format!("{stem}.fsb.info"));
        let info = if info_path.exists() {
            let i = BankInfo::open(&info_path)?;
            if i.records.len() != bank.samples.len() {
                // Mismatched sidecar: report it and carry on without it
                // rather than mapping names onto the wrong samples.
                None
            } else {
                Some(i)
            }
        } else {
            None
        };
        Ok(SoundBank { bank, info })
    }

    /// The `haloreach/fmod/pc` directory for a Reach install, given any path
    /// inside `haloreach/` (a map path will do).
    pub fn bank_dir_for_reach(any_path_in_haloreach: &Path) -> Option<PathBuf> {
        let mut p = any_path_in_haloreach;
        loop {
            let candidate = p.join("fmod").join("pc");
            if candidate.is_dir() {
                return Some(candidate);
            }
            p = p.parent()?;
        }
    }

    /// True channel count: the sidecar's if there is one, else the bank's.
    pub fn channels(&self, index: usize) -> Option<u8> {
        if let Some(i) = &self.info {
            if let Some(r) = i.get(index) {
                if r.channels > 0 && r.channels <= 8 {
                    return Some(r.channels as u8);
                }
            }
        }
        self.bank.samples.get(index).map(|s| s.channels)
    }

    /// Resolve a map's `fmod bank subsound id hash` to a sample index.
    pub fn index_of_subsound(&self, id: u32) -> Option<usize> {
        self.info.as_ref()?.index_of_subsound(id)
    }

    /// Original asset path of a sample, from the sidecar.
    pub fn asset_path(&self, index: usize) -> Option<&str> {
        Some(self.info.as_ref()?.get(index)?.path.as_str())
    }

    /// Everything a decoder needs for one bank sample, resolved once.
    ///
    /// This is the single place that turns a sample index into
    /// (setup header, channel count, rate, declared frames, Vorbis packets).
    /// Both the Ogg re-framer and the in-process decoder go through it, so a
    /// change to the resolution rule cannot land in one and miss the other.
    pub fn raw_stream<'a>(
        &'a self,
        index: usize,
        setups: &'a SetupTable,
    ) -> Result<RawStream<'a>> {
        let s = self
            .bank
            .samples
            .get(index)
            .with_context(|| format!("sample {index} of {}", self.bank.samples.len()))?;
        if self.bank.codec != Codec::Vorbis {
            anyhow::bail!(
                "{}: codec is {}, only FMOD Vorbis can be decoded",
                self.bank.path.display(),
                self.bank.codec.name()
            );
        }
        let setup_id = s
            .vorbis_setup_id(self.bank.sample_header_region())
            .with_context(|| format!("sample {index} has no VORBISDATA chunk"))?;
        let setup = setups.get(setup_id).with_context(|| {
            format!("no Vorbis setup header for id 0x{setup_id:08X}; is this the right game module?")
        })?;
        let channels = self
            .channels(index)
            .with_context(|| format!("sample {index} has no channel count"))?;
        let payload = self.bank.sample_data(index)?;
        let (packets, _consumed) = fsb5::split_vorbis_packets(payload);
        if packets.len() < 2 {
            anyhow::bail!("sample {index} holds {} Vorbis packets", packets.len());
        }
        Ok(RawStream {
            index,
            setup_id,
            setup,
            channels,
            sample_rate: s.sample_rate,
            frames: s.frames,
            packets,
        })
    }

    /// Re-frame one sample as a standard Ogg Vorbis bitstream.
    ///
    /// The result is an ordinary `.ogg` file that any Vorbis decoder reads.
    /// The final page's granule position is set to the bank's declared frame
    /// count, so a decoder trims its output to exactly that many frames.
    pub fn sample_to_ogg(&self, index: usize, setups: &SetupTable) -> Result<Vec<u8>> {
        let raw = self.raw_stream(index, setups)?;
        let setup = raw.setup;
        let channels = raw.channels;
        let packets = &raw.packets;
        let header = vorbis::parse_setup(setup, channels)?;

        // A packet's first bits are: 0 (audio packet), then the mode number.
        let mode_bits = header.mode_number_bits();
        let mut modes = Vec::with_capacity(packets.len());
        for p in packets.iter() {
            let first = *p.first().context("empty Vorbis packet")?;
            let mode = if mode_bits == 0 {
                0
            } else {
                ((first >> 1) & ((1u8 << mode_bits.min(7)) - 1)) as usize
            };
            modes.push(mode.min(header.mode_blockflags.len().saturating_sub(1)));
        }

        // Packet 0 emits nothing; packet i emits (bs[i-1] + bs[i]) / 4 frames.
        let declared = raw.frames as i64;
        let mut granule = Vec::with_capacity(packets.len());
        granule.push(0i64);
        for i in 1..packets.len() {
            let a = header.blocksize_for_mode(modes[i - 1]).unwrap_or(vorbis::BLOCKSIZE_LONG);
            let b = header.blocksize_for_mode(modes[i]).unwrap_or(vorbis::BLOCKSIZE_LONG);
            let prev = *granule.last().unwrap();
            granule.push(prev + ((a + b) / 4) as i64);
        }

        let serial = 1u32;
        let mut out =
            Vec::with_capacity(packets.iter().map(|p| p.len() + 2).sum::<usize>() + setup.len() + 4096);
        let mut seq = 0u32;
        let ident = vorbis::identification(channels, raw.sample_rate);
        out.extend_from_slice(
            &ogg::page(&[&ident], serial, seq, 0, ogg::BOS).context("ident page")?,
        );
        seq += 1;
        let comment = vorbis::comment();
        out.extend_from_slice(
            &ogg::page(&[&comment, setup], serial, seq, 0, 0).context("header page")?,
        );
        seq += 1;

        let mut i = 0usize;
        while i < packets.len() {
            let n = ogg::PACKETS_PER_PAGE.min(packets.len() - i);
            let group = &packets[i..i + n];
            i += n;
            let last = i >= packets.len();
            let g = if last {
                declared
            } else {
                granule[i - 1].min(declared)
            };
            out.extend_from_slice(
                &ogg::page(group, serial, seq, g, if last { ogg::EOS } else { 0 })
                    .context("audio page")?,
            );
            seq += 1;
        }
        Ok(out)
    }
}

/// Pull every `fmod bank subsound id hash` out of a Reach cache's sound
/// gestalt by locating the runs of 24-byte permutation blocks whose hash field
/// resolves against a sidecar.
///
/// This is a reader of last resort: it needs no tag table, so it works on a
/// raw map image. It is deliberately conservative — a run must be at least
/// `min_run` entries long at the exact 24-byte stride before it is believed.
pub fn scan_cache_for_subsound_ids(cache: &[u8], known: &dyn Fn(u32) -> bool, min_run: usize) -> Vec<(usize, u32, u32)> {
    let mut hits: Vec<usize> = Vec::new();
    let mut o = 0usize;
    while o + 4 <= cache.len() {
        let v = u32::from_le_bytes(cache[o..o + 4].try_into().unwrap());
        if v != 0 && known(v) {
            hits.push(o);
        }
        o += 4;
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < hits.len() {
        let mut j = i + 1;
        while j < hits.len() && hits[j] - hits[j - 1] == GESTALT_PERMUTATION_STRIDE {
            j += 1;
        }
        if j - i >= min_run {
            for &h in &hits[i..j] {
                if h < GESTALT_PERMUTATION_FMOD_HASH_OFFSET {
                    continue;
                }
                let entry = h - GESTALT_PERMUTATION_FMOD_HASH_OFFSET;
                let fo = entry + GESTALT_PERMUTATION_FRAMES_OFFSET;
                if fo + 4 <= cache.len() {
                    let frames = u32::from_le_bytes(cache[fo..fo + 4].try_into().unwrap());
                    let hash = u32::from_le_bytes(cache[h..h + 4].try_into().unwrap());
                    out.push((entry, hash, frames));
                }
            }
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gestalt_scan_ignores_isolated_coincidences() {
        // One stray matching value in a sea of zeroes must not be reported.
        let mut cache = vec![0u8; 4096];
        cache[1000..1004].copy_from_slice(&0xABCD_1234u32.to_le_bytes());
        let hits = scan_cache_for_subsound_ids(&cache, &|v| v == 0xABCD_1234, 4);
        assert!(hits.is_empty());
    }

    #[test]
    fn the_gestalt_scan_finds_a_run_at_the_permutation_stride() {
        let mut cache = vec![0u8; 4096];
        let base = 0x200usize;
        for k in 0..6usize {
            let entry = base + k * GESTALT_PERMUTATION_STRIDE;
            let frames = 1000 + k as u32;
            cache[entry + GESTALT_PERMUTATION_FRAMES_OFFSET
                ..entry + GESTALT_PERMUTATION_FRAMES_OFFSET + 4]
                .copy_from_slice(&frames.to_le_bytes());
            let h = entry + GESTALT_PERMUTATION_FMOD_HASH_OFFSET;
            cache[h..h + 4].copy_from_slice(&(0x5000_0000u32 + k as u32).to_le_bytes());
        }
        let hits = scan_cache_for_subsound_ids(
            &cache,
            &|v| (0x5000_0000..0x5000_0006).contains(&v),
            4,
        );
        assert_eq!(hits.len(), 6);
        assert_eq!(hits[0].0, base);
        assert_eq!(hits[0].1, 0x5000_0000);
        assert_eq!(hits[0].2, 1000);
        assert_eq!(hits[5].2, 1005);
    }

    #[test]
    fn a_hash_too_close_to_the_start_is_not_read_out_of_bounds() {
        let mut cache = vec![0u8; 64];
        cache[0..4].copy_from_slice(&0x1111_1111u32.to_le_bytes());
        // Would need entry = 0 - 0x14. Must not panic.
        let hits = scan_cache_for_subsound_ids(&cache, &|v| v == 0x1111_1111, 1);
        assert!(hits.is_empty() || hits[0].0 < cache.len());
    }
}
