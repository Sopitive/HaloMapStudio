//! In-process FMOD Vorbis decode, straight from a bank sample to f32 PCM.
//!
//! An FSB5 bank holds only the *audio* packets of a Vorbis stream — no Ogg
//! framing, no identification header, no comment header, and no setup header.
//! `lib.rs::SoundBank::raw_stream` recovers the three missing pieces (the
//! setup packet out of `haloreach.dll`, the channel count out of the sidecar,
//! and the block sizes from `vorbis::BLOCKSIZE_*`); this module feeds them to
//! `lewton`'s raw-packet API, which is the one Vorbis API shaped for exactly
//! this situation.
//!
//! The decode is not a separate reimplementation of the re-framer: both go
//! through `raw_stream`, so a change to how a sample is resolved reaches both.
//!
//! Two things a caller must know:
//!
//! * **The first audio packet emits nothing.** Vorbis overlaps windows, so the
//!   first packet only primes the right half-window. `lewton` models that and
//!   returns an empty block; the frame accounting here does not special-case
//!   it.
//! * **The stream is longer than the sample.** Vorbis block alignment makes
//!   the last packet overshoot; the bank's declared frame count is the trim
//!   point, exactly as an Ogg final-page granule would be. `decode` applies
//!   it.

use crate::{vorbis, RawStream, SetupTable, SoundBank};
use anyhow::{bail, Context, Result};
use lewton::audio::{read_audio_packet_generic, PreviousWindowRight};
use lewton::header::{read_header_ident, read_header_setup};
use lewton::samples::InterleavedSamples;

/// Decoded PCM for one bank sample.
#[derive(Clone, PartialEq)]
pub struct Pcm {
    /// Interleaved f32, nominally in [-1, 1]. `frames * channels` long.
    pub samples: Vec<f32>,
    pub channels: u8,
    pub sample_rate: u32,
}

impl std::fmt::Debug for Pcm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pcm")
            .field("frames", &self.frames())
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

impl Pcm {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Largest absolute sample value.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |a, s| a.max(s.abs()))
    }

    /// Root-mean-square over every sample of every channel.
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / self.samples.len() as f64).sqrt() as f32
    }

    /// One frame, as a slice of `channels` samples.
    pub fn frame(&self, i: usize) -> Option<&[f32]> {
        let c = self.channels as usize;
        if c == 0 {
            return None;
        }
        self.samples.get(i * c..i * c + c)
    }

    /// Collapse to mono by averaging the channels.
    ///
    /// Averaging rather than summing, so a stereo sample does not come out 2x
    /// louder than a mono one; the gain law in `attenuation` is then the only
    /// thing that sets loudness.
    pub fn to_mono(&self) -> Vec<f32> {
        let c = self.channels as usize;
        if c <= 1 {
            return self.samples.clone();
        }
        self.samples
            .chunks_exact(c)
            .map(|f| f.iter().sum::<f32>() / c as f32)
            .collect()
    }
}

/// Decode one already-resolved stream.
///
/// `trim` is the frame count to cut the result to; pass `stream.frames`.
pub fn decode_stream(stream: &RawStream<'_>) -> Result<Pcm> {
    if stream.channels == 0 {
        bail!("sample {} declares no channels", stream.index);
    }
    // FMOD synthesises the identification header from the bank's own fields;
    // this is the same 30 bytes the Ogg re-framer writes, so both paths agree
    // on the block sizes by construction rather than by a copied constant.
    let ident_bytes = vorbis::identification(stream.channels, stream.sample_rate);
    let ident = read_header_ident(&ident_bytes)
        .map_err(|e| anyhow::anyhow!("synthesised identification header rejected: {e:?}"))?;
    let setup = read_header_setup(
        stream.setup,
        stream.channels,
        (vorbis::BLOCKSIZE_SHORT_EXP, vorbis::BLOCKSIZE_LONG_EXP),
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "Vorbis setup header 0x{:08X} rejected at {} channels: {e:?}",
            stream.setup_id,
            stream.channels
        )
    })?;

    let want = stream.frames as usize * stream.channels as usize;
    let mut out: Vec<f32> = Vec::with_capacity(want);
    let mut pwr = PreviousWindowRight::new();
    for (i, packet) in stream.packets.iter().enumerate() {
        let block: InterleavedSamples<f32> =
            read_audio_packet_generic(&ident, &setup, packet, &mut pwr)
                .map_err(|e| anyhow::anyhow!("sample {} packet {i}: {e:?}", stream.index))?;
        if block.channel_count != stream.channels as usize {
            bail!(
                "sample {} packet {i} decoded {} channels, expected {}",
                stream.index,
                block.channel_count,
                stream.channels
            );
        }
        out.extend_from_slice(&block.samples);
        if out.len() >= want {
            break;
        }
    }
    // The bank's frame count is the authority. Vorbis block alignment makes
    // the packet stream overshoot it by up to one long block (measured: 0 to
    // 1016 frames over 400 samples), which is what a final-page granule
    // position trims in an Ogg file.
    out.truncate(want);
    Ok(Pcm {
        samples: out,
        channels: stream.channels,
        sample_rate: stream.sample_rate,
    })
}

impl SoundBank {
    /// Decode one bank sample to interleaved f32, trimmed to the bank's own
    /// declared frame count.
    pub fn decode_sample(&self, index: usize, setups: &SetupTable) -> Result<Pcm> {
        let raw = self
            .raw_stream(index, setups)
            .with_context(|| format!("resolve sample {index}"))?;
        decode_stream(&raw).with_context(|| format!("decode sample {index}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_statistics_are_over_every_channel() {
        let p = Pcm {
            samples: vec![0.5, -1.0, 0.0, 0.25],
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(p.frames(), 2);
        assert_eq!(p.peak(), 1.0);
        assert!((p.duration_seconds() - 2.0 / 48_000.0).abs() < 1e-12);
        let want = ((0.25f64 + 1.0 + 0.0 + 0.0625) / 4.0).sqrt() as f32;
        assert!((p.rms() - want).abs() < 1e-6, "{} vs {want}", p.rms());
        assert_eq!(p.frame(1), Some(&[0.0f32, 0.25][..]));
        assert_eq!(p.frame(2), None);
        assert_eq!(p.to_mono(), vec![-0.25, 0.125]);
    }

    #[test]
    fn a_zero_channel_pcm_does_not_divide_by_zero() {
        let p = Pcm { samples: vec![1.0], channels: 0, sample_rate: 0 };
        assert_eq!(p.frames(), 0);
        assert_eq!(p.duration_seconds(), 0.0);
        assert_eq!(p.frame(0), None);
    }

    #[test]
    fn mono_is_left_alone_by_the_downmix() {
        let p = Pcm { samples: vec![1.0, -1.0], channels: 1, sample_rate: 48_000 };
        assert_eq!(p.to_mono(), vec![1.0, -1.0]);
    }
}
