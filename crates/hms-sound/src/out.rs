//! The output device. **The only part of HMS that touches audio hardware.**
//!
//! Everything here is built around one rule: *HMS must start and run on a
//! machine with no working audio device, and must never block startup on one.*
//! So:
//!
//! * nothing opens a device until a caller asks for one ([`Output::open`] is
//!   never called from startup — see `Output::lazy`);
//! * every failure is a returned `Err`, logged once, after which audio is off
//!   for the session and every `play` is a silent no-op;
//! * a device that disappears mid-session (a USB headset unplugged) lands in
//!   cpal's error callback, which records the fault rather than panicking —
//!   a panic in an audio callback would abort the process;
//! * no caller ever waits on the audio thread. The only channel into it is a
//!   non-blocking queue.
//!
//! **Testing this without making a noise.** `Output::silent()` builds the same
//! control surface with no device at all, and [`Output::pump`] renders a block
//! through the real [`crate::mix::Mixer`] into a buffer the caller owns. Every
//! test in this crate uses one of those two, so the suite proves the mixing
//! path without opening the speakers. `Output::open_muted()` additionally
//! opens the real device with the master gain pinned to zero, which is how
//! "the device opens and the callback is fed" is checked.

use crate::decode::Pcm;
use crate::mix::{Cmd, Mixer, MixerHandle, VoiceId};
use anyhow::{anyhow, Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// What a device, or the lack of one, is doing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    /// No device has been asked for yet.
    Unopened,
    /// A device is running: (device name, sample rate, channels).
    Running(String, u32, u16),
    /// Opening failed, or the device faulted. Audio is off for the session;
    /// the string is what to show the user, once.
    Off(String),
}

struct Faults {
    /// Set by cpal's error callback. Read by `state()`.
    faulted: AtomicBool,
    message: std::sync::Mutex<String>,
    callbacks: AtomicU64,
    frames: AtomicU64,
    /// Frames in the last callback's buffer.
    last_buf: AtomicU64,
    /// The longest time between two callback starts, and the longest the mixer took inside one.
    max_gap_us: AtomicU64,
    max_fill_us: AtomicU64,
    /// Callbacks whose mix took longer than the audio it produced (the device ran dry), and
    /// callbacks that arrived more than twice a buffer late (the device thread was starved).
    underruns: AtomicU64,
    late: AtomicU64,
}

/// What the output callback has been doing, for the owner to read without setting anything.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OutDiag {
    pub rate: u32,
    pub callbacks: u64,
    pub buffer_frames: u64,
    pub max_gap_ms: f32,
    pub max_fill_ms: f32,
    pub underruns: u64,
    pub late: u64,
    /// Output frames that still exceeded full scale after the limiter, and frames the limiter was
    /// working in (running totals since the device opened).
    pub clipped: u64,
    pub limited: u64,
    /// Running totals for the output's RMS: `sumsq_nano` over `frames`.
    pub sumsq_nano: u64,
    pub mixed_frames: u64,
    /// Per-category sums of squares (nano-units), see `MixStats::cat_sumsq`.
    pub cat_sumsq_nano: [u64; crate::mix::CATEGORIES],
}

/// f32 -> i16 with SATURATION (and NaN to silence). A bare `as i16` of an out-of-range float
/// saturates in Rust, but an intermediate integer multiply or a different backend's convert would
/// wrap, and a wrapped sample is a full-scale click; the explicit clamp is the contract.
pub fn to_i16(s: f32) -> i16 {
    if s.is_nan() {
        return 0;
    }
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// f32 -> u16 (offset binary), saturating.
pub fn to_u16(s: f32) -> u16 {
    if s.is_nan() {
        return 32768;
    }
    ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u16
}

impl Faults {
    fn new() -> Faults {
        Faults {
            faulted: AtomicBool::new(false),
            message: std::sync::Mutex::new(String::new()),
            callbacks: AtomicU64::new(0),
            frames: AtomicU64::new(0),
            last_buf: AtomicU64::new(0),
            max_gap_us: AtomicU64::new(0),
            max_fill_us: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            late: AtomicU64::new(0),
        }
    }

    /// Bookkeeping for one callback: `buf` frames at `rate`, `last` the previous start, and a
    /// closure-free split into before/after so the mixer call itself is untouched. Lock-free.
    fn begin(&self, last: &mut Option<std::time::Instant>, buf: usize, rate: u32) -> std::time::Instant {
        let now = std::time::Instant::now();
        self.callbacks.fetch_add(1, Ordering::Relaxed);
        self.frames.fetch_add(buf as u64, Ordering::Relaxed);
        self.last_buf.store(buf as u64, Ordering::Relaxed);
        if let Some(prev) = *last {
            let gap = now.duration_since(prev).as_micros() as u64;
            self.max_gap_us.fetch_max(gap, Ordering::Relaxed);
            let buf_us = buf as u64 * 1_000_000 / u64::from(rate.max(1));
            if gap > 2 * buf_us + 2_000 {
                self.late.fetch_add(1, Ordering::Relaxed);
            }
        }
        *last = Some(now);
        now
    }

    fn end(&self, t0: std::time::Instant, buf: usize, rate: u32) {
        let us = t0.elapsed().as_micros() as u64;
        self.max_fill_us.fetch_max(us, Ordering::Relaxed);
        if us > buf as u64 * 1_000_000 / u64::from(rate.max(1)) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A handle on the output. Cheap to clone the control side; dropping the
/// `Output` stops the stream.
pub struct Output {
    handle: MixerHandle,
    /// Kept alive to keep the stream running. `None` for a silent output.
    stream: Option<Box<dyn StreamKeepAlive>>,
    /// Present only for a silent/pumped output: the mixer nobody else drives.
    local: Option<Mixer>,
    retired: Option<std::sync::mpsc::Receiver<Arc<Pcm>>>,
    state: State,
    faults: Arc<Faults>,
    /// Output sample rate, for the rate ratio a voice is played at.
    rate: u32,
}

/// `cpal::Stream` is not `Send`, and nothing here needs it to be; this trait
/// exists only so the stream can be stored without naming its type.
trait StreamKeepAlive {}
impl<T> StreamKeepAlive for T {}

impl std::fmt::Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Output")
            .field("state", &self.state)
            .field("rate", &self.rate)
            .field("device", &self.stream.is_some())
            .finish()
    }
}

impl Output {
    /// An output with **no device**. Sounds are accepted and dropped.
    ///
    /// This is what a machine with no audio hardware gets, and what the tests
    /// use. `rate` is the rate to pretend to run at.
    pub fn silent(rate: u32, why: &str) -> Output {
        let (handle, mixer, retired) = MixerHandle::pair();
        Output {
            handle,
            stream: None,
            local: Some(mixer),
            retired: Some(retired),
            state: if why.is_empty() { State::Unopened } else { State::Off(why.to_string()) },
            faults: Arc::new(Faults::new()),
            rate: rate.max(1),
        }
    }

    /// Open the default output device.
    ///
    /// `start_muted` pins the master gain to zero, so the device and the
    /// callback can be exercised without a sound coming out — which is how
    /// this is verified. The caller raises the gain when it actually wants
    /// to be heard.
    pub fn open(start_muted: bool) -> Result<Output> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))?;
        let name = device.name().unwrap_or_else(|_| "?".into());
        let cfg = device
            .default_output_config()
            .with_context(|| format!("{name}: no default output config"))?;
        let rate = cfg.sample_rate().0;
        let channels = cfg.channels();
        let fmt = cfg.sample_format();

        let (handle, mut mixer, retired) = MixerHandle::pair();
        if start_muted {
            handle.send(Cmd::Master(0.0));
        }
        let faults = Arc::new(Faults::new());
        let errf = faults.clone();
        let err = move |e: cpal::StreamError| {
            // NEVER panic here: this runs on the audio thread, and a panic
            // across that boundary aborts the process.
            if let Ok(mut m) = errf.message.lock() {
                *m = format!("{e}");
            }
            errf.faulted.store(true, Ordering::Relaxed);
        };
        let statf = faults.clone();
        let nch = channels as usize;
        let mut last: Option<std::time::Instant> = None;
        let stream: Box<dyn StreamKeepAlive> = match fmt {
            cpal::SampleFormat::F32 => {
                let s = device
                    .build_output_stream(
                        &cfg.config(),
                        move |out: &mut [f32], _| {
                            let buf = out.len() / nch.max(1);
                            let t0 = statf.begin(&mut last, buf, rate);
                            mixer.fill(out, nch);
                            statf.end(t0, buf, rate);
                        },
                        err,
                        None,
                    )
                    .with_context(|| format!("{name}: build f32 stream"))?;
                s.play().with_context(|| format!("{name}: play"))?;
                Box::new(s)
            }
            cpal::SampleFormat::I16 => {
                let mut scratch: Vec<f32> = Vec::new();
                let s = device
                    .build_output_stream(
                        &cfg.config(),
                        move |out: &mut [i16], _| {
                            let buf = out.len() / nch.max(1);
                            let t0 = statf.begin(&mut last, buf, rate);
                            scratch.resize(out.len(), 0.0);
                            mixer.fill(&mut scratch, nch);
                            for (o, s) in out.iter_mut().zip(scratch.iter()) {
                                *o = to_i16(*s);
                            }
                            statf.end(t0, buf, rate);
                        },
                        err,
                        None,
                    )
                    .with_context(|| format!("{name}: build i16 stream"))?;
                s.play().with_context(|| format!("{name}: play"))?;
                Box::new(s)
            }
            cpal::SampleFormat::U16 => {
                let mut scratch: Vec<f32> = Vec::new();
                let s = device
                    .build_output_stream(
                        &cfg.config(),
                        move |out: &mut [u16], _| {
                            let buf = out.len() / nch.max(1);
                            let t0 = statf.begin(&mut last, buf, rate);
                            scratch.resize(out.len(), 0.0);
                            mixer.fill(&mut scratch, nch);
                            for (o, s) in out.iter_mut().zip(scratch.iter()) {
                                *o = to_u16(*s);
                            }
                            statf.end(t0, buf, rate);
                        },
                        err,
                        None,
                    )
                    .with_context(|| format!("{name}: build u16 stream"))?;
                s.play().with_context(|| format!("{name}: play"))?;
                Box::new(s)
            }
            other => anyhow::bail!("{name}: unsupported sample format {other:?}"),
        };
        Ok(Output {
            handle,
            stream: Some(stream),
            local: None,
            retired: Some(retired),
            state: State::Running(name, rate, channels),
            faults,
            rate,
        })
    }

    /// Open a device, or fall back to a silent output that says why.
    ///
    /// This is the call an application makes: it cannot fail, so no caller has
    /// to handle "there is no sound card" as an error path.
    pub fn open_or_silent(start_muted: bool) -> Output {
        match Output::open(start_muted) {
            Ok(o) => o,
            Err(e) => Output::silent(48_000, &format!("{e:#}")),
        }
    }

    pub fn state(&self) -> State {
        if self.faults.faulted.load(Ordering::Relaxed) {
            let m = self
                .faults
                .message
                .lock()
                .map(|s| s.clone())
                .unwrap_or_else(|_| "device fault".into());
            return State::Off(format!("device fault: {m}"));
        }
        self.state.clone()
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state(), State::Running(..))
    }

    /// Output sample rate. A voice's step is `48000 / rate` times its pitch.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// `(callbacks, frames)` the device has asked for. Zero on a silent
    /// output, and the proof that a real stream is actually being driven.
    pub fn served(&self) -> (u64, u64) {
        (
            self.faults.callbacks.load(Ordering::Relaxed),
            self.faults.frames.load(Ordering::Relaxed),
        )
    }

    /// The callback's own numbers (all zero on a silent output).
    pub fn diag(&self) -> OutDiag {
        let f = &self.faults;
        OutDiag {
            rate: self.rate,
            callbacks: f.callbacks.load(Ordering::Relaxed),
            buffer_frames: f.last_buf.load(Ordering::Relaxed),
            max_gap_ms: f.max_gap_us.load(Ordering::Relaxed) as f32 / 1000.0,
            max_fill_ms: f.max_fill_us.load(Ordering::Relaxed) as f32 / 1000.0,
            underruns: f.underruns.load(Ordering::Relaxed),
            late: f.late.load(Ordering::Relaxed),
            clipped: self.handle.stats.clipped.load(Ordering::Relaxed),
            limited: self.handle.stats.limited.load(Ordering::Relaxed),
            sumsq_nano: self.handle.stats.sumsq_nano.load(Ordering::Relaxed),
            mixed_frames: self.handle.stats.frames.load(Ordering::Relaxed),
            cat_sumsq_nano: std::array::from_fn(|c| self.handle.stats.cat_sumsq[c].load(Ordering::Relaxed)),
        }
    }

    pub fn handle(&self) -> &MixerHandle {
        &self.handle
    }

    /// Master gain, linear. 0.0 is silence.
    pub fn set_master(&self, g: f32) {
        self.handle.send(Cmd::Master(g));
    }

    pub fn stop_all(&self) {
        self.handle.send(Cmd::StopAll);
    }

    /// Start a one-shot or a loop. `pitch` is a multiplier (1.0 = as
    /// recorded); the rate ratio to the device is applied here so callers
    /// never have to know the device's rate.
    pub fn play(&self, pcm: Arc<Pcm>, gain: [f32; 2], pitch: f64, looping: bool) -> Option<VoiceId> {
        if pcm.frames() == 0 {
            return None;
        }
        let step = (pcm.sample_rate as f64 / self.rate as f64) * pitch;
        self.handle.start(pcm, gain, step, looping)
    }

    /// Start a looping playlist of takes: a sound with several permutations does not repeat one
    /// of them for ever. `pitch` as for [`Output::play`].
    pub fn play_playlist(&self, pcms: Vec<Arc<Pcm>>, in_order: bool, gain: [f32; 2], pitch: f64) -> Option<VoiceId> {
        let rate = pcms.first()?.sample_rate;
        let step = (rate as f64 / self.rate as f64) * pitch;
        self.handle.start_playlist(pcms, in_order, gain, step)
    }

    pub fn set_gain(&self, id: VoiceId, gain: [f32; 2]) {
        self.handle.send(Cmd::SetGain { id, gain });
    }

    /// Re-pitch a playing voice: the same rate arithmetic as [`Output::play`], for a source that
    /// is `source_rate` Hz. A looping engine is re-pitched every frame from its RPM.
    pub fn set_pitch(&self, id: VoiceId, source_rate: u32, pitch: f64) {
        let step = (source_rate as f64 / self.rate as f64) * pitch;
        self.handle.send(Cmd::SetStep { id, step });
    }

    pub fn stop(&self, id: VoiceId) {
        self.handle.send(Cmd::Stop(id));
    }

    /// Free the buffers of voices that have finished. Call it occasionally
    /// from the control thread; the audio thread deliberately does not free
    /// them itself.
    pub fn reap(&self) -> usize {
        let Some(r) = &self.retired else { return 0 };
        let mut n = 0;
        while r.try_recv().is_ok() {
            n += 1;
        }
        n
    }

    /// Render a block through the real mixer into `out`. **Only works on a
    /// silent output** — a device-backed output's mixer belongs to the audio
    /// thread. This is how the mixing path is tested and measured without a
    /// device.
    pub fn pump(&mut self, out: &mut [f32], channels: usize) -> bool {
        match &mut self.local {
            Some(m) => {
                m.fill(out, channels);
                true
            }
            None => false,
        }
    }

    /// Peak the mixer has seen since the last call. Only meaningful on a
    /// silent (pumped) output, for the same reason as `pump`.
    pub fn take_peak(&mut self) -> Option<f32> {
        self.local.as_mut().map(|m| m.take_peak())
    }

    /// Voices currently playing. Only meaningful on a silent output.
    pub fn voice_count(&self) -> Option<usize> {
        self.local.as_ref().map(|m| m.voice_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm(frames: usize, v: f32) -> Arc<Pcm> {
        Arc::new(Pcm { samples: vec![v; frames], channels: 1, sample_rate: 48_000 })
    }

    #[test]
    fn a_silent_output_accepts_sounds_and_makes_none() {
        let mut o = Output::silent(44_100, "no device in this test");
        assert!(!o.is_running());
        assert!(matches!(o.state(), State::Off(_)));
        assert_eq!(o.served(), (0, 0));
        assert!(o.play(pcm(100, 1.0), [1.0, 1.0], 1.0, false).is_some());
        // The mixer is real, so the sound is really there — it just has
        // nowhere to go.
        let mut buf = vec![0.0f32; 400];
        assert!(o.pump(&mut buf, 2));
        // (past the limiter's look-ahead latency and the voice's 25-frame ramp-in: a 100-frame
        // sample ramps over a quarter of itself)
        let at = 2 * (crate::mix::LOOKAHEAD + 50);
        assert!(buf[at] > 0.5, "{:?}", &buf[..4]);
    }

    #[test]
    fn the_rate_ratio_is_applied_so_callers_need_not_know_the_device_rate() {
        // 48 kHz source into a 96 kHz device: half a source frame per output
        // frame, so 100 source frames become 200 output frames.
        let mut o = Output::silent(96_000, "");
        o.play(pcm(100, 1.0), [1.0, 1.0], 1.0, false).unwrap();
        let mut buf = vec![0.0f32; 400];
        o.pump(&mut buf, 1);
        let n = buf.iter().filter(|s| **s > 0.5).count();
        assert!((140..=200).contains(&n), "{n} output frames from 100 source frames at 2x");
    }

    #[test]
    fn an_empty_sample_is_refused_rather_than_occupying_a_voice() {
        let o = Output::silent(48_000, "");
        assert!(o.play(pcm(0, 0.0), [1.0, 1.0], 1.0, false).is_none());
        assert_eq!(o.voice_count(), Some(0));
    }

    #[test]
    fn open_or_silent_never_fails_however_hostile_the_machine() {
        // On a machine with no sound card this must still return, muted, and
        // report why. On one with a card it opens it at zero gain, so this
        // test is silent either way.
        let o = Output::open_or_silent(true);
        match o.state() {
            State::Running(name, rate, ch) => {
                assert!(rate >= 8_000 && rate <= 384_000, "rate {rate}");
                assert!(ch >= 1, "channels {ch}");
                eprintln!("opened {name} at {rate} Hz, {ch} ch (muted)");
            }
            State::Off(why) => eprintln!("no device, as expected on a headless box: {why}"),
            State::Unopened => panic!("open_or_silent must not leave the state Unopened"),
        }
    }

    /// The device really opens and its callback really runs — proven by the
    /// callback counter, with the master gain pinned at zero so nothing is
    /// audible. Skips itself when there is no device.
    #[test]
    fn a_real_device_opens_muted_and_its_callback_is_fed() {
        let o = Output::open_or_silent(true);
        if !o.is_running() {
            eprintln!("skipped: no audio device ({:?})", o.state());
            return;
        }
        o.play(pcm(48_000, 1.0), [1.0, 1.0], 1.0, true);
        // Give the device a few callback periods. 300 ms is generous even for
        // a 100 ms buffer and keeps the suite quick.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
        while std::time::Instant::now() < deadline && o.served().0 < 2 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let (cbs, frames) = o.served();
        assert!(cbs >= 1, "the stream callback never ran");
        assert!(frames > 0, "the callback asked for no frames");
        eprintln!("{cbs} callbacks, {frames} frames, state {:?}", o.state());
        // And it is still healthy, not faulted.
        assert!(o.is_running(), "stream faulted: {:?}", o.state());
    }

    #[test]
    fn the_device_conversions_saturate_instead_of_wrapping() {
        assert_eq!(to_i16(0.0), 0);
        assert_eq!(to_i16(1.0), 32767);
        assert_eq!(to_i16(-1.0), -32767);
        // Reach's streams are far above full scale: these must pin, not wrap to the other sign.
        assert_eq!(to_i16(17.78), 32767);
        assert_eq!(to_i16(-17.78), -32767);
        assert_eq!(to_i16(f32::INFINITY), 32767);
        assert_eq!(to_i16(f32::NAN), 0);
        assert_eq!(to_u16(17.78), 65535);
        assert_eq!(to_u16(-17.78), 0);
        assert_eq!(to_u16(0.0), 32768);
    }
}
