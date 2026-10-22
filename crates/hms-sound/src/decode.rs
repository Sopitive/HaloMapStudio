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

/// The top of the band Reach's encoder actually codes. Measured on the real bank (every
/// spectrogram of every class: shots, footsteps, foley, ambience, engine loops): a brick wall at
/// ~16.5 kHz, and above it nothing but isolated rectangles of energy at 17-21 kHz, each one long
/// block wide, 40-60 dB ABOVE the content just under the wall. In a played-back session that band
/// carried more energy than the whole audible band of a footstep, and on an engine loop it swept
/// with the pitch: a steady high-pitched squeal. Nothing the encoder meant to carry is above its
/// own wall, so playback removes what is: a zero-phase FIR, flat below 16.3 kHz, 60 dB down by 17.3.
pub const BAND_LIMIT_HZ: f32 = 16_800.0;
const BAND_LIMIT_TAPS: usize = 255;

fn band_limit_kernel(rate: u32) -> Vec<f32> {
    let m = (BAND_LIMIT_TAPS - 1) / 2;
    let fc = f64::from(BAND_LIMIT_HZ) / f64::from(rate.max(1));
    let mut h: Vec<f64> = (0..BAND_LIMIT_TAPS)
        .map(|k| {
            let x = k as f64 - m as f64;
            let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x) };
            let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (BAND_LIMIT_TAPS - 1) as f64).cos() + 0.08 * (4.0 * std::f64::consts::PI * k as f64 / (BAND_LIMIT_TAPS - 1) as f64).cos();
            sinc * w
        })
        .collect();
    let sum: f64 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    h.into_iter().map(|v| v as f32).collect()
}

/// Remove everything above [`BAND_LIMIT_HZ`] from a decoded sample. The filter is applied
/// PERIODICALLY (the sample wraps), so a looping sample's seam stays seamless; a one-shot's quiet
/// tail wrapping into its head is negligible.
pub fn band_limit(pcm: &mut Pcm) {
    let ch = pcm.channels as usize;
    let n = pcm.frames();
    if ch == 0 || n < 2 || pcm.sample_rate <= 2 * BAND_LIMIT_HZ as u32 {
        return;
    }
    let h = band_limit_kernel(pcm.sample_rate);
    let m = (h.len() - 1) / 2;
    // One channel at a time, padded by `m` frames of wrap-around each side so the inner loop has
    // no index arithmetic; the kernel is symmetric, so each pair of taps shares one multiply.
    let mut ext = vec![0.0f32; n + 2 * m];
    for c in 0..ch {
        for (k, e) in ext.iter_mut().enumerate() {
            let j = (k + n * (m / n + 1) - m) % n;
            *e = pcm.samples[j * ch + c];
        }
        for i in 0..n {
            let centre = i + m;
            let mut acc = h[m] * ext[centre];
            for d in 1..=m {
                acc += h[m + d] * (ext[centre - d] + ext[centre + d]);
            }
            pcm.samples[i * ch + c] = acc;
        }
    }
}

/// Frames of the crossfade that closes a loop (50 ms at 48 kHz).
pub const LOOP_FADE_FRAMES: usize = 2400;

/// A SEAMLESS version of a sample that will be looped on its own. A decoded Reach loop is not
/// sample-exact: the step from its last frame to its first is as large as the loudest step inside
/// it (measured on the Warthog's engine loops once the band limit took the noise out of the
/// interior), which is a tick every loop. The last `F` frames are crossfaded (equal power) into
/// the `F` frames that would follow them, i.e. the sample's own head, and dropped; the result is
/// `F` frames shorter and the wrap is continuous.
pub fn loop_crossfade(p: &Pcm) -> Pcm {
    let ch = p.channels as usize;
    let n = p.frames();
    let f = LOOP_FADE_FRAMES.min(n / 6);
    if ch == 0 || f < 8 || n < 4 * f {
        return p.clone();
    }
    let keep = n - f;
    let mut out = p.samples[..keep * ch].to_vec();
    for i in 0..f {
        let th = (i as f32 + 0.5) / f as f32 * std::f32::consts::FRAC_PI_2;
        let (wh, wt) = (th.sin(), th.cos());
        for c in 0..ch {
            // Head frame `i` and the tail frame that sits `keep` frames after it.
            out[i * ch + c] = p.samples[i * ch + c] * wh + p.samples[(keep + i) * ch + c] * wt;
        }
    }
    Pcm { samples: out, channels: p.channels, sample_rate: p.sample_rate }
}

impl SoundBank {
    /// [`SoundBank::decode_sample`], band-limited for playback (see [`BAND_LIMIT_HZ`]).
    pub fn decode_sample_playback(&self, index: usize, setups: &SetupTable) -> Result<Pcm> {
        let mut p = self.decode_sample(index, setups)?;
        band_limit(&mut p);
        Ok(p)
    }

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

    fn tone(f: f32, n: usize, rate: u32) -> Pcm {
        Pcm { samples: (0..n).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / rate as f32).sin()).collect(), channels: 1, sample_rate: rate }
    }

    /// **THE BAND LIMIT** passes the coded band untouched, removes what is above the encoder's own
    /// wall, and does not break a loop's seam.
    #[test]
    fn the_band_limit_keeps_the_coded_band_and_removes_what_is_above_it() {
        let rms_mid = |p: &Pcm| {
            let v = &p.samples[400..p.samples.len() - 400];
            (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
        };
        for (f, lo, hi) in [(1_000.0f32, 0.70, 0.72), (10_000.0, 0.70, 0.72), (15_500.0, 0.69, 0.72), (18_800.0, 0.0, 0.0007), (21_000.0, 0.0, 0.0007)] {
            let mut p = tone(f, 8_192, 48_000);
            band_limit(&mut p);
            let r = rms_mid(&p);
            assert!(r >= lo && r <= hi, "{f} Hz tone came out at rms {r} (want {lo}..{hi})");
        }
        // A looping 1 kHz tone (an exact number of periods) stays periodic across the seam.
        let mut p = tone(1_000.0, 4_800, 48_000);
        let before = p.samples.clone();
        band_limit(&mut p);
        let worst = p.samples.iter().zip(before.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "the seam moved a 1 kHz loop by {worst}");
    }

    /// **THE LOOP SEAM.** A loop whose last frame does not meet its first (a tick every period)
    /// has a continuous wrap after [`loop_crossfade`].
    #[test]
    fn a_crossfaded_loop_has_no_step_at_the_wrap() {
        // A slow sine with a deliberate step at the seam: 3.37 periods.
        let n = 20_000usize;
        let p = Pcm { samples: (0..n).map(|i| (2.0 * std::f32::consts::PI * 3.37 * i as f32 / n as f32).sin() * 0.5).collect(), channels: 1, sample_rate: 48_000 };
        let seam = (p.samples[0] - p.samples[n - 1]).abs();
        let biggest = |x: &[f32]| x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(seam > 3.0 * biggest(&p.samples), "the test loop must have a step at its seam: {seam}");
        let q = loop_crossfade(&p);
        assert_eq!(q.frames(), n - LOOP_FADE_FRAMES.min(n / 6));
        let seam_q = (q.samples[0] - q.samples[q.frames() - 1]).abs();
        assert!(seam_q <= 2.0 * biggest(&q.samples), "the wrap still steps by {seam_q} (interior {})", biggest(&q.samples));
    }

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
