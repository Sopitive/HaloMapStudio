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

/// Extra slots for voices that are fading out after a stop or a steal.
const FADING_HEADROOM: usize = 16;

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
    /// Change a playing voice's rate (a Doppler or pitch change). The voice glides to it
    /// ([`STEP_SLEW`]) rather than jumping, so a rate that is re-aimed once per game frame does
    /// not stair-step.
    SetStep { id: VoiceId, step: f64 },
    /// Tag a voice with a reporting category (see [`CATEGORIES`]); only the level meter reads it.
    SetCat { id: VoiceId, cat: u8 },
    /// Turn a just-started LOOPING voice into a playlist: when it reaches the end of the
    /// sample it moves on to another of `pcms` instead of repeating the same one. This is how
    /// a sound with several permutations loops (an ambience bed of six takes is not a
    /// 3-second loop): `in_order` plays them in sequence, otherwise a random other one.
    Playlist { id: VoiceId, pcms: Vec<Arc<Pcm>>, in_order: bool },
    Stop(VoiceId),
    StopAll,
    /// Master gain, linear. The one knob a volume slider turns.
    Master(f32),
}

struct Voice {
    id: VoiceId,
    /// Reporting category, an index into [`MixStats::cat_sumsq`].
    cat: u8,
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
    /// The rate being glided toward (`step` is the rate in use).
    step_tgt: f64,
    /// Playlist state; empty for an ordinary voice.
    alts: Vec<Arc<Pcm>>,
    idx: usize,
    in_order: bool,
    rng: u32,
    /// Output frames since the voice started: a few-millisecond ramp-in, so a voice that starts
    /// mid-waveform does not click.
    age: u32,
    /// Output frames of fade-out left once the voice has been told to stop (or was stolen);
    /// `None` while it plays on. A hard cut is a click.
    stopping: Option<u32>,
}

/// Ramp-in of every voice, output frames (2 ms at 48 kHz), and the fade a stopped or stolen voice
/// takes to reach silence (5 ms). A one-shot also fades over the last [`TAIL_FRAMES`] SOURCE
/// frames of its sample, so a sample whose decoded tail is not zero does not end in a step.
const RAMP_IN_FRAMES: f32 = 96.0;
const STOP_FRAMES: u32 = 240;
const TAIL_FRAMES: f64 = 192.0;

/// How fast a voice's rate glides to a new target, per output frame: 1/1024 is a ~21 ms time
/// constant, long enough to turn a once-per-game-frame step into a ramp, short enough that a
/// real pitch change is not smeared.
const STEP_SLEW: f64 = 1.0 / 1024.0;

/// Frames either side of a playlist junction over which the voice dips to zero, so two takes
/// that were not cut to join do not click (10 ms at 48 kHz).
const JUNCTION_FRAMES: f64 = 480.0;

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
            if self.alts.len() > 1 {
                // The next take. Carry the overshoot so the junction does not drift.
                let over = self.pos - frames as f64;
                self.idx = if self.in_order {
                    (self.idx + 1) % self.alts.len()
                } else {
                    self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (self.idx + 1 + (self.rng >> 8) as usize % (self.alts.len() - 1)) % self.alts.len()
                };
                self.pcm = self.alts[self.idx].clone();
                self.pos = over;
                return self.next();
            }
            // Wrap, keeping the fractional part so a loop does not drift.
            self.pos %= frames as f64;
        }
        if (self.step_tgt - self.step).abs() > 1e-9 {
            self.step += (self.step_tgt - self.step) * STEP_SLEW;
        }
        let i = self.pos as usize;
        let t = (self.pos - i as f64) as f32;
        // Four-point (Catmull-Rom) interpolation. A pitch or rate change read with a straight
        // line between two samples images high frequencies back into the band: at 48 kHz source
        // to a 44.1 kHz device that is a constant rasp on bright material (gunfire).
        let tap = |k: isize| -> (f32, f32) {
            let n = frames as isize;
            let idx = i as isize + k;
            let idx = if self.looping { idx.rem_euclid(n) } else { idx.clamp(0, n - 1) };
            sample_lr(&self.pcm, idx as usize)
        };
        let (p0, p1, p2, p3) = (tap(-1), tap(0), tap(1), tap(2));
        let cubic = |a: f32, b: f32, c: f32, d: f32| {
            b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)))
        };
        let (ya, yb) = (cubic(p0.0, p1.0, p2.0, p3.0), cubic(p0.1, p1.1, p2.1, p3.1));
        for c in 0..2 {
            self.cur[c] += (self.gain[c] - self.cur[c]) * GAIN_SLEW;
        }
        // Ramp in; fade out when told to stop; fade out over the tail of a one-shot.
        // (Both fades are at most a quarter of the sample, so a very short sample is not all ramp.)
        let ramp_in = RAMP_IN_FRAMES.min((frames as f32 / 4.0).max(1.0));
        let mut ramp = ((self.age + 1) as f32 / ramp_in).min(1.0);
        self.age = self.age.saturating_add(1);
        if let Some(left) = self.stopping.as_mut() {
            ramp *= *left as f32 / STOP_FRAMES as f32;
            *left = left.saturating_sub(1);
        }
        if !self.looping {
            let tail = TAIL_FRAMES.min((frames as f64 / 4.0).max(1.0));
            ramp *= (((frames as f64 - self.pos) / tail).min(1.0)) as f32;
        }
        let a = (ya, yb);
        // A playlist voice dips across its junctions (both sides of the sample's ends).
        let edge = if self.alts.len() > 1 {
            ((self.pos / JUNCTION_FRAMES).min(1.0)).min(((frames as f64 - self.pos) / JUNCTION_FRAMES).min(1.0)) as f32
        } else {
            1.0
        };
        self.pos += self.step;
        if self.stopping == Some(0) {
            return None;
        }
        Some((a.0 * self.cur[0] * edge * ramp, a.1 * self.cur[1] * edge * ramp))
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
    lim: Limiter,
    pub stats: Arc<MixStats>,
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
            lim: Limiter::new(),
            stats: Arc::new(MixStats::default()),
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
                if self.voices.iter().filter(|v| v.stopping.is_none()).count() >= MAX_VOICES {
                    // Steal the oldest one-shot with a FADE (a hard cut is a pop); the newcomer
                    // starts at once. Loops are never stolen: with nothing to steal, the newcomer
                    // is refused. Voices still fading count against a small headroom so the
                    // mixer's work stays bounded.
                    let full = self.voices.len() >= MAX_VOICES + FADING_HEADROOM;
                    let victim = if full {
                        None
                    } else {
                        self.voices.iter_mut().filter(|v| v.stopping.is_none() && !v.looping).max_by_key(|v| v.age)
                    };
                    match victim {
                        Some(v) => v.stopping = Some(STOP_FRAMES),
                        None => {
                            self.dropped += 1;
                            // Hand the buffer straight back; do not free it here.
                            let _ = self.retired.send(pcm);
                            return;
                        }
                    }
                }
                // A non-finite or non-positive step would hang or NaN the
                // read position, so clamp it rather than trusting the caller.
                let step = if step.is_finite() && step > 0.0 { step.min(64.0) } else { 1.0 };
                let g = [sane(gain[0]), sane(gain[1])];
                self.voices.push(Voice {
                    id,
                    cat: 0,
                    pcm,
                    pos: 0.0,
                    step,
                    gain: g,
                    // Start at the target: a one-shot is too short to ramp
                    // into, and its own attack does the job.
                    cur: g,
                    looping,
                    step_tgt: step,
                    alts: Vec::new(),
                    idx: 0,
                    in_order: false,
                    rng: (id.0 as u32).wrapping_mul(2_654_435_761) | 1,
                    age: 0,
                    stopping: None,
                });
            }
            Cmd::Playlist { id, pcms, in_order } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.alts = pcms;
                    v.in_order = in_order;
                    v.looping = true;
                } else {
                    for p in pcms {
                        let _ = self.retired.send(p);
                    }
                }
            }
            Cmd::SetGain { id, gain } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.gain = [sane(gain[0]), sane(gain[1])];
                }
            }
            Cmd::SetStep { id, step } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    if step.is_finite() && step > 0.0 {
                        v.step_tgt = step.min(64.0);
                    }
                }
            }
            Cmd::SetCat { id, cat } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.cat = cat.min(CATEGORIES as u8 - 1);
                }
            }
            Cmd::Stop(id) => self.retire_where(|v| v.id == id),
            Cmd::StopAll => self.retire_where(|_| true),
            Cmd::Master(g) => self.master = sane(g),
        }
    }

    /// Fade out every voice matching `pred` (they retire when the fade ends).
    fn retire_where(&mut self, pred: impl Fn(&Voice) -> bool) {
        for v in self.voices.iter_mut() {
            if v.stopping.is_none() && pred(v) {
                v.stopping = Some(STOP_FRAMES);
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
        let mut cat_ss = [0.0f64; CATEGORIES];
        for chunk in out.chunks_mut(channels) {
            let mut l = 0.0f32;
            let mut r = 0.0f32;
            let mut i = 0;
            while i < self.voices.len() {
                match self.voices[i].next() {
                    Some((a, b)) => {
                        l += a;
                        r += b;
                        let (am, bm) = (a * m, b * m);
                        cat_ss[self.voices[i].cat as usize] += f64::from((am * am + bm * bm) * 0.5);
                        i += 1;
                    }
                    None => {
                        let v = self.voices.swap_remove(i);
                        for a in v.alts {
                            let _ = self.retired.send(a);
                        }
                        let _ = self.retired.send(v.pcm);
                    }
                }
            }
            l *= m;
            r *= m;
            self.peak = self.peak.max(l.abs()).max(r.abs());
            // LIMIT, then soft-clip. Reach's streams decode far above full scale (78% of the sfx
            // bank, loudest +41 dBFS; an assault-rifle shot lands at 2.2x full scale after its own
            // authored gain) and the engine mixes them through a limiter. A LOOK-AHEAD limiter: the
            // gain starts falling a millisecond BEFORE the peak arrives and is smooth, so a loud
            // transient is turned down, not distorted. (The first limiter here reacted to each
            // sample, which put distortion 7 dB below the signal in the shot's upper band, the
            // crackle that was heard on every shot.)
            let (ol, or, g) = self.lim.process(l, r);
            if g < 0.999 {
                self.stats.limited.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            l = ol;
            r = or;
            if l.abs() > 1.0 || r.abs() > 1.0 {
                self.stats.clipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let l = soft_clip(l);
            let r = soft_clip(r);
            self.stats.sumsq_nano.fetch_add((((l * l + r * r) * 0.5) as f64 * 1.0e9) as u64, std::sync::atomic::Ordering::Relaxed);
            self.stats.frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        for (c, v) in cat_ss.iter().enumerate() {
            self.stats.cat_sumsq[c].fetch_add((*v * 1.0e9) as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// The limiter's ceiling, its look-ahead (frames, ~1.3 ms at 48 kHz: also its latency), its
/// release coefficient (~100 ms at 48 kHz), and the knee below which the soft clip is the
/// identity.
const LIMIT_CEILING: f32 = 0.9;
pub const LOOKAHEAD: usize = 64;
const LIMIT_RELEASE: f32 = 0.0002;
const KNEE: f32 = 0.9;

/// A look-ahead peak limiter. The input is delayed by [`LOOKAHEAD`] frames; the gain applied to a
/// delayed frame is the boxcar average of the sliding minimum of the required gains, which is
/// never above the gain that frame itself needs, and ramps down over the look-ahead instead of
/// jumping. Recovery is a slow one-pole (a fast one would modulate low-frequency content).
struct Limiter {
    dl: Vec<(f32, f32)>,
    req: Vec<f32>,
    min: Vec<f32>,
    sum: f32,
    rel: f32,
    pos: usize,
}

impl Limiter {
    fn new() -> Limiter {
        let n = LOOKAHEAD + 1;
        Limiter { dl: vec![(0.0, 0.0); n], req: vec![1.0; n], min: vec![1.0; n], sum: n as f32, rel: 1.0, pos: 0 }
    }

    /// `(left, right, gain)` for the frame that entered [`LOOKAHEAD`] frames ago.
    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32, f32) {
        let n = LOOKAHEAD + 1;
        let i = self.pos;
        let out = self.dl[(i + 1) % n];
        self.dl[i] = (l, r);
        let pk = l.abs().max(r.abs());
        self.req[i] = if pk > LIMIT_CEILING { LIMIT_CEILING / pk } else { 1.0 };
        let m = self.req.iter().fold(1.0f32, |a, b| a.min(*b));
        self.sum += m - self.min[i];
        self.min[i] = m;
        if i == 0 {
            // The running sum drifts by rounding; re-sum once per lap.
            self.sum = self.min.iter().sum();
        }
        let s = (self.sum / n as f32).min(1.0);
        self.rel = if s < self.rel { s } else { self.rel + (s - self.rel) * LIMIT_RELEASE };
        self.pos = (i + 1) % n;
        let g = self.rel;
        (out.0 * g, out.1 * g, g)
    }
}

/// Identity below [`KNEE`]; above it a smooth saturation that never exceeds 1.0.
#[inline]
pub fn soft_clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= KNEE {
        x
    } else {
        let room = 1.0 - KNEE;
        x.signum() * (KNEE + room * ((a - KNEE) / room).tanh())
    }
}

/// Counters the audio callback keeps and anyone may read.
#[derive(Debug, Default)]
pub struct MixStats {
    /// Output frames in which the signal, AFTER the limiter, still exceeded full scale (so the
    /// soft clip had to round it off).
    pub clipped: std::sync::atomic::AtomicU64,
    /// Output frames in which the limiter was pulling the gain down.
    pub limited: std::sync::atomic::AtomicU64,
    /// Running sum of `(l^2 + r^2) / 2` in nano-units, and the frames it covers: the RMS of the
    /// last second is the difference of two readings.
    pub sumsq_nano: std::sync::atomic::AtomicU64,
    pub frames: std::sync::atomic::AtomicU64,
    /// The same sum of squares per voice CATEGORY, before the limiter (what each category would
    /// contribute alone): 0 other, 1 ambience, 2 engine, 3 body, 4 weapon.
    pub cat_sumsq: [std::sync::atomic::AtomicU64; CATEGORIES],
}

/// Reporting categories of a voice.
pub const CATEGORIES: usize = 5;

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
    /// The mixer's own counters (clipping, limiting), shared with the callback.
    pub stats: Arc<MixStats>,
}

impl MixerHandle {
    /// A handle, its mixer, and the channel finished buffers come back on.
    pub fn pair() -> (MixerHandle, Mixer, Receiver<Arc<Pcm>>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let (rtx, rrx) = std::sync::mpsc::channel();
        let mixer = Mixer::new(rx, rtx);
        (
            MixerHandle { tx, next: Arc::new(std::sync::atomic::AtomicU64::new(1)), stats: mixer.stats.clone() },
            mixer,
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

    /// Start a looping playlist of takes (see [`Cmd::Playlist`]).
    pub fn start_playlist(&self, pcms: Vec<Arc<Pcm>>, in_order: bool, gain: [f32; 2], step: f64) -> Option<VoiceId> {
        let first = pcms.first()?.clone();
        let id = self.start(first, gain, step, true)?;
        self.send(Cmd::Playlist { id, pcms, in_order });
        Some(id)
    }

    /// Tag a voice with its reporting category (the level meter's per-category RMS).
    pub fn set_cat(&self, id: VoiceId, cat: u8) {
        self.send(Cmd::SetCat { id, cat });
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

    /// `Mixer::fill` with the limiter's look-ahead latency taken out: the frames returned are the
    /// ones the voices produced, from the first.
    fn fill_al(m: &mut Mixer, buf: &mut [f32], ch: usize) {
        let mut big = vec![0.0f32; buf.len() + LOOKAHEAD * ch];
        m.fill(&mut big, ch);
        buf.copy_from_slice(&big[LOOKAHEAD * ch..]);
    }

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
        fill_al(&mut m, &mut buf, 2);
        assert!(buf.iter().all(|s| *s == 0.0), "{buf:?}");
        assert_eq!(m.voice_count(), 0);
    }

    #[test]
    fn a_one_shot_plays_once_at_its_gain_and_then_retires() {
        let (h, mut m, retired) = MixerHandle::pair();
        h.start(pcm(1, 4, 1.0), [0.5, 0.25], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 2 * 8];
        fill_al(&mut m, &mut buf, 2);
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
        fill_al(&mut m, &mut buf, 1);
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
        fill_al(&mut m, &mut buf, 2);
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
        fill_al(&mut m, &mut buf, 2);
        assert!((buf[0] - 0.25).abs() < 1e-6, "{:?}", &buf[..4]);
    }

    #[test]
    fn the_voice_ceiling_bounds_the_callbacks_work() {
        let (h, mut m, retired) = MixerHandle::pair();
        for _ in 0..(MAX_VOICES + 10) {
            h.start(pcm(1, 1000, 0.01), [1.0, 1.0], 1.0, true);
        }
        let mut buf = vec![0.0f32; 2];
        fill_al(&mut m, &mut buf, 2);
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
        fill_al(&mut m, &mut buf, 2);
        assert!(buf.iter().all(|s| s.is_finite()), "{buf:?}");
    }

    #[test]
    fn resampling_halves_and_doubles_the_length() {
        let (h, mut m, _r) = MixerHandle::pair();
        // 8 source frames at step 2.0 = 4 output frames.
        h.start(pcm(1, 8, 1.0), [1.0, 1.0], 2.0, false).unwrap();
        let mut buf = vec![0.0f32; 8];
        fill_al(&mut m, &mut buf, 1);
        assert_eq!(buf.iter().filter(|s| **s > 0.3).count(), 4, "{buf:?}");

        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 8, 1.0), [1.0, 1.0], 0.5, false).unwrap();
        let mut buf = vec![0.0f32; 32];
        fill_al(&mut m, &mut buf, 1);
        // (16 output frames, the last one inside the one-shot's tail fade.)
        let n = buf.iter().filter(|s| **s > 0.3).count();
        assert!((15..=16).contains(&n), "{n} {buf:?}");
    }

    #[test]
    fn a_voice_can_be_re_aimed_and_stopped_by_id() {
        let (h, mut m, _r) = MixerHandle::pair();
        let id = h.start(pcm(1, 48_000, 1.0), [1.0, 0.0], 1.0, true).unwrap();
        let mut buf = vec![0.0f32; 2 * 256];
        fill_al(&mut m, &mut buf, 2);
        let tail = &buf[2 * 255..];
        assert!(tail[0] > 0.8 && tail[1] < 0.1, "{tail:?}");
        // Re-aim hard right and let the ramp settle (5 ms at 48 kHz).
        h.send(Cmd::SetGain { id, gain: [0.0, 1.0] });
        let mut long = vec![0.0f32; 2 * 4096];
        m.fill(&mut long, 2);
        let (l, r) = (long[2 * 4095], long[2 * 4095 + 1]);
        assert!(l < 0.05 && r > 0.85, "after the slew: l {l} r {r}");
        // A stop FADES (a hard cut is a click): the voice is still there for a moment, falling...
        h.send(Cmd::Stop(id));
        let mut a = vec![0.0f32; 2 * 100];
        m.fill(&mut a, 2);
        assert_eq!(m.voice_count(), 1);
        assert!(a[2 * 99 + 1] < a[1], "falling: {} -> {}", a[1], a[2 * 99 + 1]);
        // ...and gone once the fade (STOP_FRAMES) is over.
        let mut b = vec![0.0f32; 2 * 300];
        m.fill(&mut b, 2);
        assert_eq!(m.voice_count(), 0);
        assert!(b[2 * 299 + 1].abs() < 1e-6);
    }

    #[test]
    fn a_stale_set_gain_cannot_re_aim_a_later_voice() {
        let (h, mut m, _r) = MixerHandle::pair();
        let a = h.start(pcm(1, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 2 * 8];
        fill_al(&mut m, &mut buf, 2); // a retires here
        let b = h.start(pcm(1, 48_000, 1.0), [1.0, 1.0], 1.0, true).unwrap();
        assert_ne!(a, b);
        h.send(Cmd::SetGain { id: a, gain: [0.0, 0.0] });
        let mut long = vec![0.0f32; 2 * 512];
        m.fill(&mut long, 2);
        assert!(long[2 * 511] > 0.5, "voice b was silenced by a stale command: {:?}", &long[2 * 508..]);
    }

    #[test]
    fn multichannel_samples_fold_without_getting_louder() {
        for ch in [1u8, 2, 4, 6] {
            let (h, mut m, _r) = MixerHandle::pair();
            h.start(pcm(ch, 4, 0.5), [1.0, 1.0], 1.0, false).unwrap();
            let mut buf = vec![0.0f32; 8];
            fill_al(&mut m, &mut buf, 2);
            // (frame 1 of 4: past the 1-frame ramp, before the 1-frame tail)
            assert!(
                (buf[2] - 0.5).abs() < 1e-5 && (buf[3] - 0.5).abs() < 1e-5,
                "{ch} channels folded to {} / {}",
                buf[2],
                buf[3]
            );
        }
    }

    #[test]
    fn mono_output_sums_the_two_sides_without_clipping_them_twice() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(2, 4, 0.5), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 4];
        fill_al(&mut m, &mut buf, 1);
        assert!((buf[1] - 0.5).abs() < 1e-5, "{buf:?}");
    }

    #[test]
    fn channels_beyond_the_first_pair_are_left_silent() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 4, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![9.0f32; 6 * 4];
        fill_al(&mut m, &mut buf, 6);
        for f in 0..4 {
            if f == 1 || f == 2 {
                assert!(buf[f * 6] > 0.5 && buf[f * 6 + 1] > 0.5);
            }
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
        fill_al(&mut m, &mut buf, 2);
        assert_eq!(m.voice_count(), 0, "a zero-length looping sample must not loop forever");
    }

    /// A ramp whose value IS its frame index: whatever the mixer outputs, its first difference
    /// is the playback rate in use at that output frame.
    fn ramp(n: usize) -> Arc<Pcm> {
        Arc::new(Pcm { samples: (0..n).map(|i| i as f32 / n as f32 * 0.5).collect(), channels: 1, sample_rate: 48_000 })
    }

    /// A rate that is re-aimed once per game frame must glide, not jump: the largest change of
    /// rate between two consecutive output frames stays far below the size of the step.
    #[test]
    fn a_rate_change_glides_instead_of_jumping() {
        let n = 400_000;
        let (h, mut m, _r) = MixerHandle::pair();
        let id = h.start(ramp(n), [1.0, 1.0], 1.0, true).unwrap();
        let mut buf = vec![0.0f32; 4800];
        fill_al(&mut m, &mut buf, 1);
        h.send(Cmd::SetStep { id, step: 1.4 });
        let mut out = vec![0.0f32; 24_000];
        m.fill(&mut out, 1);
        // Rate over 256-frame blocks (a single frame's difference is below f32 resolution on a
        // ramp this long): rate = (y[i+256]-y[i]) * n / 0.5 / 256.
        let rate: Vec<f64> = (0..out.len() - 256).step_by(256).map(|i| f64::from(out[i + 256] - out[i]) * n as f64 / 0.5 / 256.0).collect();
        let worst = rate.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f64, f64::max);
        assert!((rate[rate.len() - 1] - 1.4).abs() < 0.02, "it must arrive: {}", rate[rate.len() - 1]);
        assert!(rate[0] < 1.2, "and not arrive at once: {}", rate[0]);
        assert!(worst < 0.15, "a 0.4 jump in rate became a step of {worst} between two blocks");
    }

    /// The mixer needs nothing from the control thread to keep a loop going: with the main
    /// thread stalled (no commands for a long run of callbacks) the output is the loop,
    /// continuous, in callback-sized pieces of any size.
    #[test]
    fn a_stalled_control_thread_does_not_interrupt_a_loop() {
        let n = 10_000;
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(ramp(n), [1.0, 1.0], 1.0, true).unwrap();
        let mut all = Vec::new();
        for chunk in [512usize, 1024, 2048, 256, 1024].iter().cycle().take(60) {
            let mut b = vec![0.0f32; *chunk];
            m.fill(&mut b, 1);
            all.extend(b);
        }
        let step = 0.5 / n as f32;
        for w in all[200..].windows(2) {
            let d = w[1] - w[0];
            // Either the normal step, or the one wrap back to the start.
            assert!((d - step).abs() < 2e-5 || (d + 0.5 - step).abs() < 1e-3, "discontinuity {d}");
        }
        assert_eq!(m.voice_count(), 1);
    }

    /// Mixing is cheap enough to run on the audio thread: 48 stereo voices at a 256-frame
    /// callback cost a small fraction of the time the callback has.
    #[test]
    fn the_mixer_costs_a_fraction_of_the_callback_budget() {
        let (h, mut m, _r) = MixerHandle::pair();
        for k in 0..48 {
            h.start(pcm(2, 96_000, 0.1), [0.5, 0.5], 1.0 + k as f64 * 0.003, true).unwrap();
        }
        let mut b = vec![0.0f32; 256 * 2];
        m.fill(&mut b, 2);
        let t = std::time::Instant::now();
        for _ in 0..200 {
            m.fill(&mut b, 2);
        }
        let per = t.elapsed().as_secs_f64() / 200.0;
        let budget = 256.0 / 48_000.0;
        assert!(per < 0.5 * budget, "a 256-frame callback with 48 voices took {:.3} ms of {:.3} ms", per * 1e3, budget * 1e3);
    }

    /// A playlist voice moves on to another take at the end of one (so a bed of six takes is
    /// not a 3-second loop), and the junction does not click.
    #[test]
    fn a_playlist_voice_moves_on_and_the_junction_does_not_click() {
        let (h, mut m, _r) = MixerHandle::pair();
        let a = pcm(1, 4_000, 0.5);
        let b = pcm(1, 4_000, -0.5);
        h.start_playlist(vec![a, b], true, [1.0, 1.0], 1.0).unwrap();
        let mut out = vec![0.0f32; 16_000];
        m.fill(&mut out, 1);
        assert!(out[2_000] > 0.4 && out[6_000] < -0.4 && out[10_000] > 0.4, "A, then B, then A again");
        let worst = out[200..].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.5 / 400.0, "the junction stepped by {worst}");
    }

    /// With ONE take the playlist is just a loop.
    #[test]
    fn a_one_take_playlist_is_an_ordinary_loop() {
        let (h, mut m, _r) = MixerHandle::pair();
        h.start_playlist(vec![pcm(1, 100, 0.25)], false, [1.0, 1.0], 1.0).unwrap();
        let mut out = vec![0.0f32; 1_000];
        m.fill(&mut out, 1);
        assert!(out[200..].iter().all(|v| (v - 0.25).abs() < 1e-3));
    }

    /// A synthetic gunshot: a decaying 3 kHz burst that starts far above full scale, like Reach's
    /// own (the median sfx sample peaks at +9 dBFS and the loudest at +41).
    fn shot(peak: f32) -> Arc<Pcm> {
        let n = 24_000;
        Arc::new(Pcm {
            samples: (0..n).map(|i| peak * (-(i as f32) / 3_000.0).exp() * (i as f32 * 0.39).sin()).collect(),
            channels: 1,
            sample_rate: 48_000,
        })
    }

    /// **CLIPPING.** A held automatic burst (15 shots a second of a sample that peaks at +12 dBFS)
    /// stays inside full scale and does not flat-top: the limiter takes the gain down instead of
    /// the output being sliced off. The slice is a harsh high-pitched crackle.
    #[test]
    fn a_dense_burst_is_limited_not_hard_clipped() {
        let (h, mut m, _r) = MixerHandle::pair();
        let mut out: Vec<f32> = Vec::new();
        for k in 0..45 {
            h.start(shot(4.0), [1.0, 1.0], 1.0, false).unwrap();
            let mut b = vec![0.0f32; 3_200 * 2];
            m.fill(&mut b, 2);
            out.extend(b.chunks_exact(2).map(|c| c[0]));
            let _ = k;
        }
        let peak = out.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let flat = out.iter().filter(|v| v.abs() >= 0.999).count() as f32 / out.len() as f32;
        assert!(peak <= 1.0, "peak {peak}");
        assert!(flat < 0.001, "{:.3}% of the burst is flat-topped at full scale", flat * 100.0);
        // The limiter was working (the input really was far over), and said so.
        assert!(m.stats.limited.load(std::sync::atomic::Ordering::Relaxed) > 1000);
        // No sample-to-sample jump a limiter should not make: the burst's own steepest slope is
        // 4.0 * 0.39 per frame before limiting, so after it nothing exceeds full scale per frame.
        let worst = out.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1.0, "a step of {worst}");
    }

    /// **START / STOP / STEAL POPS.** A voice that starts on a high sample, one that is stopped
    /// while loud, and one stolen at the voice cap all move by a small step per frame, never a jump.
    #[test]
    fn no_voice_starts_stops_or_is_stolen_with_a_click() {
        // Start on a high sample.
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(pcm(1, 48_000, 0.6), [1.0, 1.0], 1.0, true).unwrap();
        let mut b = vec![0.0f32; 400];
        m.fill(&mut b, 1);
        let worst = b.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max).max(b[0].abs());
        assert!(worst < 0.6 / 90.0, "the start stepped by {worst}");
        // Stop while loud.
        let (h, mut m, _r) = MixerHandle::pair();
        let id = h.start(pcm(1, 48_000, 0.6), [1.0, 1.0], 1.0, true).unwrap();
        let mut warm = vec![0.0f32; 4_000];
        m.fill(&mut warm, 1);
        h.send(Cmd::Stop(id));
        let mut b = vec![0.0f32; 600];
        m.fill(&mut b, 1);
        let worst = std::iter::once(warm[3_999]).chain(b.iter().copied()).collect::<Vec<_>>().windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "the stop stepped by {worst} (a hard cut of this voice is 0.6)");
        assert_eq!(m.voice_count(), 0);
        // Steal at the cap: MAX_VOICES quiet one-shots then one more; the stolen one fades.
        let (h, mut m, _r) = MixerHandle::pair();
        for _ in 0..MAX_VOICES {
            h.start(pcm(1, 100_000, 0.01), [1.0, 1.0], 1.0, false).unwrap();
        }
        let mut warm = vec![0.0f32; 2_000];
        m.fill(&mut warm, 1);
        let before = warm[1_999];
        h.start(pcm(1, 100_000, 0.01), [1.0, 1.0], 1.0, false).unwrap();
        let mut b = vec![0.0f32; 600];
        m.fill(&mut b, 1);
        let worst = std::iter::once(before).chain(b.iter().copied()).collect::<Vec<_>>().windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.002, "stealing a voice stepped by {worst} against a per-voice level of 0.01");
        assert_eq!(m.dropped, 0, "the newcomer must be accepted by stealing");
    }

    /// **THE LIMITER DOES NOT DISTORT.** A gunshot-like burst (a decaying 3 kHz tone and noise
    /// peaking at 2.2x full scale, which is where an assault-rifle shot lands after its own
    /// authored gain) comes out as the SAME waveform at a lower level: the part of the output that
    /// is not a scaled copy of the input is far below the signal. A per-sample limiter measured
    /// -16 to -21 dB here, and -7 dB in the band above 8 kHz: the crackle on every shot.
    #[test]
    fn a_loud_transient_is_turned_down_not_distorted() {
        let n = 24_000usize;
        let mut seed = 7u32;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
                let env = (-(i as f32) / 3_000.0).exp();
                2.2 * env * (0.8 * (2.0 * std::f32::consts::PI * 3_000.0 * i as f32 / 48_000.0).sin() + 0.2 * noise)
            })
            .collect();
        let (h, mut m, _r) = MixerHandle::pair();
        h.start(Arc::new(Pcm { samples: x.clone(), channels: 1, sample_rate: 48_000 }), [1.0, 1.0], 1.0, false).unwrap();
        let mut y = vec![0.0f32; n];
        fill_al(&mut m, &mut y, 1);
        // A level that changes slowly is not distortion; a gain that moves within a cycle is. So
        // fit one gain per 64-sample block (1.3 ms) and measure what that cannot explain. (The
        // voice's own 2 ms ramp-in and its tail fade are left out.)
        let (mut er, mut ex) = (0.0f64, 0.0f64);
        let mut alpha = 0.0f64;
        let mut at = 200;
        while at + 64 <= n - 400 {
            let (mut xy, mut xx) = (0.0f64, 0.0f64);
            for i in at..at + 64 {
                xy += f64::from(x[i]) * f64::from(y[i]);
                xx += f64::from(x[i]) * f64::from(x[i]);
            }
            let al = xy / xx.max(1e-30);
            if at == 200 {
                alpha = al;
            }
            for i in at..at + 64 {
                let d = f64::from(y[i]) - al * f64::from(x[i]);
                er += d * d;
                ex += al * al * f64::from(x[i]) * f64::from(x[i]);
            }
            at += 64;
        }
        let db = 10.0 * (er / ex).log10();
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak <= 1.0 && peak > 0.5, "the burst must come out limited, not silent: peak {peak}");
        assert!(db < -30.0, "the limiter distorted the burst: residual {db:.1} dB below the signal (best-fit gain {alpha:.3})");
    }

    /// **RESAMPLING.** A 10 kHz tone played at the ratio of a 48 kHz sample on a 44.1 kHz device
    /// comes out as the same tone: a straight line between neighbouring samples smears it (its
    /// worst error is several percent of the tone), a four-point interpolator does not.
    #[test]
    fn resampling_a_high_tone_is_accurate() {
        let f = 10_000.0f64;
        let n = 48_000usize;
        let src = Arc::new(Pcm { samples: (0..n).map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / 48_000.0).sin() as f32 * 0.5).collect(), channels: 1, sample_rate: 48_000 });
        let (h, mut m, _r) = MixerHandle::pair();
        let step = 48_000.0 / 44_100.0;
        h.start(src, [1.0, 1.0], step, true).unwrap();
        let mut out = vec![0.0f32; 20_000];
        fill_al(&mut m, &mut out, 1);
        let mut worst = 0.0f64;
        for (k, v) in out.iter().enumerate().skip(2_000) {
            let ideal = (2.0 * std::f64::consts::PI * f * (k as f64 * step) / 48_000.0).sin() * 0.5;
            worst = worst.max((f64::from(*v) - ideal).abs());
        }
        assert!(worst < 0.05, "worst resampling error {worst} on a 0.5 tone");
    }
}
