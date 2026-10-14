//! The mixer: voices in, interleaved stereo-or-more out. **No device here.**
//!
//! This module knows nothing about `cpal`. It is a pure function from a
//! command stream and a set of decoded samples to a block of output frames,
//! which is what makes it testable without making a noise: every gate in
//! `tests` below drives the same `fill` the real audio callback drives.
//!
//! Real-time discipline, because `fill` runs on the audio callback thread:
//!
//! * no locks — commands arrive through an `mpsc` queue that the callback
//!   drains with `try_recv`;
//! * no allocation in `fill` — `voices` is reserved once at construction and
//!   a voice that finishes is *swapped out and handed back* to the control
//!   thread through `retired`, so the `Arc<Pcm>` is dropped (and its buffer
//!   freed) off the audio thread;
//! * a voice count ceiling, so a film that fires a thousand events in a tick
//!   cannot make the callback overrun its deadline.
//!
//! Sample rate: the banks are 48 kHz throughout but a device may not be, so a
//! voice carries a fractional step and is **linearly interpolated**. That is a
//! deliberately simple resampler, not an engine-faithful one; it is also what
//! carries pitch (the gestalt's pitch bounds are in cents, and a cent shift is
//! just a multiplier on the same step).

use crate::decode::Pcm;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

/// Hard ceiling on simultaneous voices. Reach itself caps simultaneous
/// instances per sound class; this is a different thing — a bound on the work
/// the audio callback can be asked to do, so a pathological film cannot make
/// it overrun.
pub const MAX_VOICES: usize = 64;

/// Identifies a playing voice, so a positioned or looping sound can be
/// re-aimed or stopped while it plays.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct VoiceId(pub u64);

/// What the control thread asks the mixer to do.
pub enum Cmd {
    /// Start a voice. `gain` is [left, right], linear, applied to the sample's
    /// own amplitude; `step` is source frames per output frame (rate ratio
    /// times any pitch shift).
    Start {
        id: VoiceId,
        pcm: Arc<Pcm>,
        gain: [f32; 2],
        step: f64,
        looping: bool,
    },
    /// Re-aim a playing voice. Ignored if it has already finished.
    SetGain { id: VoiceId, gain: [f32; 2] },
    /// Change a playing voice's rate (a Doppler or pitch change).
    SetStep { id: VoiceId, step: f64 },
    Stop(VoiceId),
    StopAll,
    /// Master gain, linear. The one knob a volume slider turns.
    Master(f32),
}

struct Voice {
    id: VoiceId,
    pcm: Arc<Pcm>,
    /// Fractional read position in source frames.
    pos: f64,
    step: f64,
    /// Target gain, and the gain actually in use. Gains are ramped rather
    /// than applied instantly: a positioned sound is re-aimed every video
    /// frame, and a stepped gain change is an audible click.
    gain: [f32; 2],
    cur: [f32; 2],
    looping: bool,
}

/// How fast a gain change is reached, as a fraction of the remaining distance
/// per output frame. 1/256 at 48 kHz is a ~5 ms time constant: fast enough to
/// track a moving source, slow enough that no step is audible.
const GAIN_SLEW: f32 = 1.0 / 256.0;

impl Voice {
    /// One output frame's worth of (left, right) from this voice, advancing it.
    /// `None` once it has run out.
    #[inline]
    fn next(&mut self) -> Option<(f32, f32)> {
        let frames = self.pcm.frames();
        if frames == 0 {
            return None;
        }
        if self.pos >= frames as f64 {
            if !self.looping {
                return None;
            }
            // Wrap, keeping the fractional part so a loop does not drift.
            self.pos %= frames as f64;
        }
        let i = self.pos as usize;
        let frac = (self.pos - i as f64) as f32;
        let j = if i + 1 < frames {
            i + 1
        } else if self.looping {
            0
        } else {
            i
        };
        let (a, b) = (sample_lr(&self.pcm, i), sample_lr(&self.pcm, j));
        for c in 0..2 {
            self.cur[c] += (self.gain[c] - self.cur[c]) * GAIN_SLEW;
        }
        self.pos += self.step;
        Some((
            (a.0 + (b.0 - a.0) * frac) * self.cur[0],
            (a.1 + (b.1 - a.1) * frac) * self.cur[1],
        ))
    }
}

/// One source frame folded to (left, right).
///
/// Mono feeds both sides equally. Stereo passes through. Four- and
/// six-channel samples (Reach has 464 and 84 of them, all ambience) are
/// folded: channels 0/2/4 to the left, 1/3/5 to the right, averaged, which
/// keeps a quad ambience bed at the same loudness as a stereo one.
#[inline]
fn sample_lr(p: &Pcm, frame: usize) -> (f32, f32) {
    let c = p.channels as usize;
    let Some(f) = p.frame(frame) else {
        return (0.0, 0.0);
    };
    match c {
        1 => (f[0], f[0]),
        2 => (f[0], f[1]),
        _ => {
            let mut l = 0.0;
            let mut r = 0.0;
            let mut n = 0.0f32;
            for (k, v) in f.iter().enumerate() {
                if k % 2 == 0 {
                    l += *v;
                } else {
                    r += *v;
                }
                n += 0.5;
            }
            let n = n.max(1.0);
            (l / n, r / n)
        }
    }
}

/// The audio-callback side. Owns the playing voices.
pub struct Mixer {
    voices: Vec<Voice>,
    cmds: Receiver<Cmd>,
    retired: Sender<Arc<Pcm>>,
    master: f32,
    /// Voices refused because [`MAX_VOICES`] was reached. A diagnostic.
    pub dropped: u64,
    /// Largest absolute output value since the last `take_peak`. Reach's own
    /// Vorbis streams decode well above full scale, so this is how clipping
    /// is made visible rather than merely inaudible.
    peak: f32,
}

impl Mixer {
    pub fn new(cmds: Receiver<Cmd>, retired: Sender<Arc<Pcm>>) -> Mixer {
        Mixer {
            voices: Vec::with_capacity(MAX_VOICES),
            cmds,
            retired,
            master: 1.0,
            dropped: 0,
            peak: 0.0,
        }
    }

    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    pub fn master(&self) -> f32 {
        self.master
    }

    /// Peak since the last call, and reset.
    pub fn take_peak(&mut self) -> f32 {
        std::mem::replace(&mut self.peak, 0.0)
    }

    /// Drain the command queue. Called at the top of `fill`; separated out so
    /// a test can apply commands and inspect the state without rendering.
    pub fn drain(&mut self) {
        while let Ok(c) = self.cmds.try_recv() {
            self.apply(c);
        }
    }

    fn apply(&mut self, c: Cmd) {
        match c {
            Cmd::Start { id, pcm, gain, step, looping } => {
                if self.voices.len() >= MAX_VOICES {
                    self.dropped += 1;
                    // Hand the buffer straight back; do not free it here.
                    let _ = self.retired.send(pcm);
                    return;
                }
                // A non-finite or non-positive step would hang or NaN the
                // read position, so clamp it rather than trusting the caller.
                let step = if step.is_finite() && step > 0.0 { step.min(64.0) } else { 1.0 };
                let g = [sane(gain[0]), sane(gain[1])];
                self.voices.push(Voice {
                    id,
                    pcm,
                    pos: 0.0,
                    step,
                    gain: g,
                    // Start at the target: a one-shot is too short to ramp
                    // into, and its own attack does the job.
                    cur: g,
                    looping,
                });
            }
            Cmd::SetGain { id, gain } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.gain = [sane(gain[0]), sane(gain[1])];
                }
            }
            Cmd::SetStep { id, step } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    if step.is_finite() && step > 0.0 {
                        v.step = step.min(64.0);
                    }
                }
            }
            Cmd::Stop(id) => self.retire_where(|v| v.id == id),
            Cmd::StopAll => self.retire_where(|_| true),
            Cmd::Master(g) => self.master = sane(g),
        }
    }

    fn retire_where(&mut self, pred: impl Fn(&Voice) -> bool) {
        let mut i = 0;
        while i < self.voices.len() {
            if pred(&self.voices[i]) {
                let v = self.voices.swap_remove(i);
                let _ = self.retired.send(v.pcm);
            } else {
                i += 1;
            }
        }
    }

    /// Render one block. `out` is interleaved `channels`-channel output and is
    /// fully overwritten, so a mixer with no voices writes silence rather than
    /// leaving whatever the device buffer held.
    pub fn fill(&mut self, out: &mut [f32], channels: usize) {
        self.drain();
        if channels == 0 {
            return;
        }
        out.fill(0.0);
        let m = self.master;
        for chunk in out.chunks_mut(channels) {
            let mut l = 0.0f32;
            let mut r = 0.0f32;
            let mut i = 0;
            while i < self.voices.len() {
                match self.voices[i].next() {
                    Some((a, b)) => {
                        l += a;
                        r += b;
                        i += 1;
                    }
                    None => {
                        let v = self.voices.swap_remove(i);
                        let _ = self.retired.send(v.pcm);
                    }
                }
            }
            l *= m;
            r *= m;
            self.peak = self.peak.max(l.abs()).max(r.abs());
            // Clip at the output. Reach's streams decode above full scale
            // (measured: 78% of the sfx bank), so this is load-bearing: a
            // float buffer handed to the device unclamped is not merely loud,
            // on some backends it is a fault.
            let l = l.clamp(-1.0, 1.0);
            let r = r.clamp(-1.0, 1.0);
            match channels {
                1 => chunk[0] = (l + r) * 0.5,
                _ => {
                    chunk[0] = l;
                    chunk[1] = r;
                    // Anything beyond the first pair stays silent: writing a
                    // one-shot into a centre or LFE channel is a guess, and a
                    // guess that doubles its loudness on some setups.
                    for c in chunk[2..].iter_mut() {
                        *c = 0.0;
                    }
                }
            }
        }
    }
}

#[inline]
fn sane(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 64.0)
    } else {
        0.0
    }
}

/// The control side: hands commands to a `Mixer` that may be running on an
/// audio thread, or on no thread at all.
#[derive(Clone)]
pub struct MixerHandle {
    tx: Sender<Cmd>,
    next: Arc<std::sync::atomic::AtomicU64>,
}

impl MixerHandle {
    /// A handle, its mixer, and the channel finished buffers come back on.
    pub fn pair() -> (MixerHandle, Mixer, Receiver<Arc<Pcm>>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let (rtx, rrx) = std::sync::mpsc::channel();
        (
            MixerHandle { tx, next: Arc::new(std::sync::atomic::AtomicU64::new(1)) },
            Mixer::new(rx, rtx),
            rrx,
        )
    }

    /// Allocate a voice id. Monotonic, so an id is never reused and a stale
    /// `SetGain` cannot re-aim somebody else's voice.
    pub fn next_id(&self) -> VoiceId {
        VoiceId(self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    /// Queue a command. Returns false once the mixer is gone, which is how a
    /// caller learns audio has stopped without having to care why.
    pub fn send(&self, c: Cmd) -> bool {
        self.tx.send(c).is_ok()
    }

    pub fn start(&self, pcm: Arc<Pcm>, gain: [f32; 2], step: f64, looping: bool) -> Option<VoiceId> {
        let id = self.next_id();
        if self.send(Cmd::Start { id, pcm, gain, step, looping }) {
            Some(id)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm(ch: u8, frames: usize, v: f32) -> Arc<Pcm> {
        Arc::new(Pcm {
            samples: vec![v; frames * ch as usize],
            channels: ch,
            sample_rate: 48_000,
        })
    }

    #[test]
    fn a_mixer_with_no_voices_writes_silence_over_whatever_was_there() {
        let (_h, mut m, _r) = MixerHandle::pair();
        let mut buf = vec![0.5f32; 64];
        m.fill(&mut buf, 2);
        assert!(buf.iter().all(|s| *s == 0.0), "{buf:?}");
        assert_eq!(m.voice_count(), 0);
    }

    #[test]
    fn a_one_shot_plays_once_at_its_gain_and_then_retires() {
        let (h, mut m, retired) = MixerHandle::pair();
        h.start(pcm(1, 4, 1.0), [0.5, 0.25], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 2 * 8];
        m.fill(&mut buf, 2);
        // Four frames of signal, then silence.
        for f in 0..4 {
            assert!((buf[f * 2] - 0.5).abs() < 1e-3, "frame {f}: {:?}", &buf[..16]);
            assert!((buf[f * 2 + 1] - 0.25).abs() < 1e-3, "frame {f}");
        }
        for f in 4..8 {
            assert_eq!(buf[f * 2], 0.0);
            assert_eq!(buf[f * 2 + 1], 0.0);
        }
        assert_eq!(m.voice_count(), 0, "the voice should have retired");
        // And its buffer came back to the control thread rather than being
        // freed on the audio thread.
        assert_eq!(retired.try_recv().map(|p| p.frames()), Ok(4));
    }

    #[test]
    fn a_looping_voice_never_retires_and_wraps_without_drift() {
        let (h, mut m, _r) = MixerHandle::pair();
        // A 3-frame ramp, looping, read at 1 frame per output frame. The
        // values stay inside full scale so the output clamp cannot be what
        // this test is measuring.
        let p = Arc::new(Pcm { samples: vec![0.1, 0.2, 0.3], channels: 1, sample_rate: 48_000 });
        h.start(p, [1.0, 1.0], 1.0, true).unwrap();
        let mut buf = vec![0.0f32; 9];
        m.fill(&mut buf, 1);
        let got: Vec<f32> = buf.iter().copied().collect();
        assert_eq!(got, vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3], "{got:?}");
        assert_eq!(m.voice_count(), 1);
    }

    #[test]
    fn the_output_is_clamped_because_reach_samples_exceed_full_scale() {
        let (h, mut m, _r) = MixerHandle::pair();
        // 17.78 is ar_reach1's measured peak.
        h.start(pcm(1, 8, 17.78), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 16];
        m.fill(&mut buf, 2);
        assert!(buf.iter().all(|s| *s <= 1.0 && *s >= -1.0), "{buf:?}");
        // ...but the clipping is reported, not hidden.
        assert!(m.take_peak() > 17.0, "peak should record the pre-clip level");
        assert_eq!(m.take_peak(), 0.0, "take_peak resets");
    }

    #[test]
    fn the_master_gain_scales_every_voice() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.send(Cmd::Master(0.25));
        h.start(pcm(1, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 8];
        m.fill(&mut buf, 2);
        assert!((buf[0] - 0.25).abs() < 1e-6, "{:?}", &buf[..4]);
    }

    #[test]
    fn the_voice_ceiling_bounds_the_callbacks_work() {
        let (h, mut m, retired) = MixerHandle::pair();
        for _ in 0..(MAX_VOICES + 10) {
            h.start(pcm(1, 1000, 0.01), [1.0, 1.0], 1.0, true);
        }
        let mut buf = vec![0.0f32; 2];
        m.fill(&mut buf, 2);
        assert_eq!(m.voice_count(), MAX_VOICES);
        assert_eq!(m.dropped, 10);
        // The refused buffers were handed back, not dropped on the audio thread.
        let mut back = 0;
        while retired.try_recv().is_ok() {
            back += 1;
        }
        assert_eq!(back, 10);
    }

    #[test]
    fn a_hostile_step_or_gain_cannot_hang_or_poison_the_mixer() {
        let (h, mut m, _r) = MixerHandle::pair();
        for step in [0.0f64, -1.0, f64::NAN, f64::INFINITY, 1.0e18] {
            h.start(pcm(1, 8, 1.0), [f32::NAN, f32::INFINITY], step, false);
        }
        let mut buf = vec![0.0f32; 64];
        m.fill(&mut buf, 2);
        assert!(buf.iter().all(|s| s.is_finite()), "{buf:?}");
    }

    #[test]
    fn resampling_halves_and_doubles_the_length() {
        let (h, mut m, _r) = MixerHandle::pair();
        // 8 source frames at step 2.0 = 4 output frames.
        h.start(pcm(1, 8, 1.0), [1.0, 1.0], 2.0, false).unwrap();
        let mut buf = vec![0.0f32; 8];
        m.fill(&mut buf, 1);
        assert_eq!(buf.iter().filter(|s| **s > 0.5).count(), 4, "{buf:?}");

        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 8, 1.0), [1.0, 1.0], 0.5, false).unwrap();
        let mut buf = vec![0.0f32; 32];
        m.fill(&mut buf, 1);
        assert_eq!(buf.iter().filter(|s| **s > 0.5).count(), 16, "{buf:?}");
    }

    #[test]
    fn a_voice_can_be_re_aimed_and_stopped_by_id() {
        let (h, mut m, _r) = MixerHandle::pair();
        let id = h.start(pcm(1, 48_000, 1.0), [1.0, 0.0], 1.0, true).unwrap();
        let mut buf = vec![0.0f32; 2];
        m.fill(&mut buf, 2);
        assert!(buf[0] > 0.9 && buf[1] < 0.1, "{buf:?}");
        // Re-aim hard right and let the ramp settle (5 ms at 48 kHz).
        h.send(Cmd::SetGain { id, gain: [0.0, 1.0] });
        let mut long = vec![0.0f32; 2 * 4096];
        m.fill(&mut long, 2);
        let (l, r) = (long[2 * 4095], long[2 * 4095 + 1]);
        assert!(l < 0.05 && r > 0.95, "after the slew: l {l} r {r}");
        h.send(Cmd::Stop(id));
        m.fill(&mut buf, 2);
        assert_eq!(m.voice_count(), 0);
    }

    #[test]
    fn a_stale_set_gain_cannot_re_aim_a_later_voice() {
        let (h, mut m, _r) = MixerHandle::pair();
        let a = h.start(pcm(1, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 2 * 8];
        m.fill(&mut buf, 2); // a retires here
        let b = h.start(pcm(1, 48_000, 1.0), [1.0, 1.0], 1.0, true).unwrap();
        assert_ne!(a, b);
        h.send(Cmd::SetGain { id: a, gain: [0.0, 0.0] });
        m.fill(&mut buf, 2);
        assert!(buf[0] > 0.9, "voice b was silenced by a stale command: {buf:?}");
    }

    #[test]
    fn multichannel_samples_fold_without_getting_louder() {
        for ch in [1u8, 2, 4, 6] {
            let (h, mut m, _r) = MixerHandle::pair();
            h.start(pcm(ch, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
            let mut buf = vec![0.0f32; 8];
            m.fill(&mut buf, 2);
            assert!(
                (buf[0] - 1.0).abs() < 1e-5 && (buf[1] - 1.0).abs() < 1e-5,
                "{ch} channels folded to {} / {}",
                buf[0],
                buf[1]
            );
        }
    }

    #[test]
    fn mono_output_sums_the_two_sides_without_clipping_them_twice() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(2, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 4];
        m.fill(&mut buf, 1);
        assert!((buf[0] - 1.0).abs() < 1e-5, "{buf:?}");
    }

    #[test]
    fn channels_beyond_the_first_pair_are_left_silent() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![9.0f32; 6 * 4];
        m.fill(&mut buf, 6);
        for f in 0..4 {
            assert!(buf[f * 6] > 0.5 && buf[f * 6 + 1] > 0.5);
            for c in 2..6 {
                assert_eq!(buf[f * 6 + c], 0.0, "frame {f} channel {c}");
            }
        }
    }

    #[test]
    fn an_empty_sample_retires_instead_of_spinning() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 0, 0.0), [1.0, 1.0], 1.0, true).unwrap();
        let mut buf = vec![0.0f32; 8];
        m.fill(&mut buf, 2);
        assert_eq!(m.voice_count(), 0, "a zero-length looping sample must not loop forever");
    }
}
