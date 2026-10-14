//! The graphics BENCHMARK: a repeatable frame-time measurement of the real application frame.
//!
//! Why in-app and not a separate harness: the thing users complain about is the *editor* frame,
//! which is `update()` (panels, object bookkeeping, particle simulation, overlay rebuilds) plus
//! `SceneRenderer::render()` plus egui's own paint — all on one thread. A standalone renderer
//! loop would measure only the third of that which was never the problem.
//!
//! What it measures, per frame:
//!   * `cpu_ms`  — wall time inside `App::update()`, i.e. everything the UI thread does;
//!   * `frame_ms`— wall time between consecutive `update()` entries, i.e. the real frame period
//!                 (so a GPU-bound frame shows up here even though the CPU was idle);
//!   * `gpu`     — per-pass GPU milliseconds from `wgpu` timestamp queries, in the second phase.
//!
//! It runs in two phases on purpose. Reading timestamps back requires draining the GPU, which
//! serialises CPU and GPU and therefore destroys the frame-period measurement — so phase 1 runs
//! free (wall + CPU) and phase 2 turns the timer on for the per-pass attribution. Vertical sync
//! is also forced off for a benchmark run, otherwise every frame period is the display's refresh
//! interval and nothing can be compared.
//!
//! Driven by environment variables because it is a measurement diagnostic, not a feature:
//!   `HMS_BENCH=<frames>`      run the benchmark (default 300 frames per phase), then exit
//!   `HMS_BENCH_WARMUP=<n>`    frames discarded first (default 90 — shader/pipeline warm-up,
//!                             the post-load object-decode tail and the exposure ramp)
//!   `HMS_BENCH_LABEL=<text>`  a tag printed with the results (the map/preset under test)
//!   `HMS_BENCH_OUT=<path>`    append the result line to this file as well as stderr
//!   `HMS_BENCH_NOGPU=1`       skip phase 2 (adapters without `TIMESTAMP_QUERY`)
//!
//! Everything else about the run — which map, which variant, where the camera is — comes from
//! the existing headless switches (`HMS_MAP`, `HMS_MVAR`, `HMS_CAM`), so a benchmark and a
//! screenshot of the same thing are configured identically.

use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Waiting for the map (and its object-decode tail) to settle.
    Waiting,
    /// Discarding warm-up frames.
    Warmup,
    /// Free-running: wall + CPU frame time.
    Wall,
    /// GPU timestamps on: per-pass attribution.
    Gpu,
    Done,
}

pub struct Bench {
    phase: Phase,
    frames: u32,
    warmup: u32,
    left: u32,
    label: String,
    out: Option<std::path::PathBuf>,
    want_gpu: bool,
    cpu_ms: Vec<f32>,
    frame_ms: Vec<f32>,
    gpu_cpu_ms: Vec<f32>,
    last_entry: Option<Instant>,
    frame_start: Option<Instant>,
    /// Per-frame `update()` stage split (the HMS_FRAMEPROF stages), phase 1 only.
    stages: Vec<[f32; 6]>,
    /// The scene's draw-call census at the moment it settled.
    census: String,
    /// Filled from the renderer's timer at the end of phase 2.
    gpu_report: Vec<(String, f64)>,
    gpu_frame: (f64, f64, f64),
    gpu_available: bool,
}

/// mean / median / p99 / max of a sample set, in milliseconds.
pub fn stats(v: &[f32]) -> (f32, f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = s.iter().sum::<f32>() / s.len() as f32;
    let median = s[s.len() / 2];
    let p99 = s[((s.len() as f32 * 0.99) as usize).min(s.len() - 1)];
    (mean, median, p99, *s.last().unwrap())
}

impl Bench {
    pub fn from_env() -> Option<Self> {
        let frames: u32 = std::env::var("HMS_BENCH").ok()?.trim().parse().unwrap_or(300);
        let frames = frames.clamp(10, 100_000);
        let warmup = std::env::var("HMS_BENCH_WARMUP").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(90u32);
        Some(Self {
            phase: Phase::Waiting,
            frames,
            warmup,
            left: warmup,
            label: std::env::var("HMS_BENCH_LABEL").unwrap_or_default(),
            out: std::env::var("HMS_BENCH_OUT").ok().map(std::path::PathBuf::from),
            want_gpu: std::env::var("HMS_BENCH_NOGPU").is_err(),
            cpu_ms: Vec::with_capacity(frames as usize),
            frame_ms: Vec::with_capacity(frames as usize),
            gpu_cpu_ms: Vec::with_capacity(frames as usize),
            last_entry: None,
            frame_start: None,
            stages: Vec::new(),
            census: String::new(),
            gpu_report: Vec::new(),
            gpu_frame: (0.0, 0.0, 0.0),
            gpu_available: false,
        })
    }

    /// True while the benchmark is driving the app (so the frame loop must keep pumping and the
    /// camera must be left alone).
    pub fn active(&self) -> bool {
        self.phase != Phase::Done
    }

    /// The map has finished loading AND its object tail has drained: start counting.
    pub fn scene_settled(&mut self) {
        if self.phase == Phase::Waiting {
            self.phase = Phase::Warmup;
            self.left = self.warmup;
            eprintln!("HMS_BENCH warm-up: {} frames", self.warmup);
        }
    }

    /// As `scene_settled`, recording the scene's draw-call census for the report (it is the
    /// quantity any encode-time optimisation has to move, so it belongs next to the timings).
    pub fn scene_settled_with(&mut self, census: String) {
        if self.phase == Phase::Waiting {
            self.census = census;
            self.scene_settled();
        }
    }

    /// Call at the top of `update()`. Records the previous frame's period.
    pub fn frame_begin(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_entry {
            let dt = now.duration_since(prev).as_secs_f32() * 1000.0;
            match self.phase {
                Phase::Wall => self.frame_ms.push(dt),
                Phase::Gpu => self.gpu_cpu_ms.push(dt),
                _ => {}
            }
        }
        self.last_entry = Some(now);
        self.frame_start = Some(now);
    }

    /// Call at the END of `update()`. Returns `true` once the whole run is finished (the caller
    /// then prints and exits).
    ///
    /// `wants_gpu_timer` is what the caller should have told the renderer this frame; it is read
    /// back by [`Bench::gpu_timer_wanted`] so the caller can switch the timer on at the phase
    /// boundary without duplicating the state machine.
    pub fn frame_end(&mut self, stages: [f32; 6]) -> bool {
        let cpu = self.frame_start.take().map(|t| t.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
        match self.phase {
            Phase::Waiting | Phase::Done => return self.phase == Phase::Done,
            Phase::Warmup => {
                self.left = self.left.saturating_sub(1);
                if self.left == 0 {
                    self.phase = Phase::Wall;
                    self.left = self.frames;
                    eprintln!("HMS_BENCH phase 1/2 (wall + CPU, {} frames)", self.frames);
                }
            }
            Phase::Wall => {
                self.cpu_ms.push(cpu);
                self.stages.push(stages);
                self.left = self.left.saturating_sub(1);
                if self.left == 0 {
                    if self.want_gpu {
                        self.phase = Phase::Gpu;
                        self.left = self.frames;
                        eprintln!("HMS_BENCH phase 2/2 (GPU timestamps, {} frames)", self.frames);
                    } else {
                        self.phase = Phase::Done;
                    }
                }
            }
            Phase::Gpu => {
                self.left = self.left.saturating_sub(1);
                if self.left == 0 {
                    self.phase = Phase::Done;
                }
            }
        }
        self.phase == Phase::Done
    }

    /// Whether the renderer's per-pass GPU timer should be on right now.
    pub fn gpu_timer_wanted(&self) -> bool {
        self.phase == Phase::Gpu
    }

    pub fn take_gpu_report(&mut self, available: bool, report: Vec<(String, f64)>, frame: (f64, f64, f64)) {
        self.gpu_available = available;
        self.gpu_report = report;
        self.gpu_frame = frame;
    }

    /// Render the result as text, print it to stderr and (optionally) append it to a file.
    pub fn report(&self) -> String {
        let (cm, cmed, c99, cmax) = stats(&self.cpu_ms);
        let (wm, wmed, w99, wmax) = stats(&self.frame_ms);
        let mut s = String::new();
        s.push_str(&format!("=== HMS graphics benchmark {} ===\n", self.label));
        s.push_str(&format!("frames: {} measured ({} warm-up discarded)\n", self.frame_ms.len(), self.warmup));
        if !self.census.is_empty() {
            s.push_str(&format!("{}\n", self.census));
        }
        s.push_str(&format!(
            "frame period : mean {wm:7.2} ms  median {wmed:7.2}  p99 {w99:7.2}  max {wmax:7.2}  ({:.1} fps mean)\n",
            if wm > 0.0 { 1000.0 / wm } else { 0.0 }
        ));
        s.push_str(&format!("update() CPU : mean {cm:7.2} ms  median {cmed:7.2}  p99 {c99:7.2}  max {cmax:7.2}\n"));
        if !self.stages.is_empty() {
            // The same six stages HMS_FRAMEPROF reports, but measured over the benchmark's own
            // frame window instead of a rolling 300 and without the GPU-drain phase mixed in.
            const NAMES: [&str; 6] = ["pre (load/tick/commands)", "panels + properties", "particle sim", "render encode", "exposure meter", "input + rest"];
            let n = self.stages.len() as f32;
            let mut rows: Vec<(f32, &str)> = (0..6)
                .map(|k| (self.stages.iter().map(|r| r[k]).sum::<f32>() / n, NAMES[k]))
                .collect();
            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            s.push_str("update() CPU per stage (mean ms, biggest first):\n");
            for (ms, name) in rows {
                let share = if cm > 0.0 { ms / cm * 100.0 } else { 0.0 };
                s.push_str(&format!("  {name:<28} {ms:7.3}  {share:5.1}%\n"));
            }
        }
        if self.gpu_available {
            let (gm, g99, gmax) = self.gpu_frame;
            s.push_str(&format!("GPU (sum of passes): mean {gm:7.2} ms  p99 {g99:7.2}  max {gmax:7.2}\n"));
            s.push_str("GPU per pass (mean ms, biggest first):\n");
            for (name, ms) in &self.gpu_report {
                let share = if self.gpu_frame.0 > 0.0 { ms / self.gpu_frame.0 * 100.0 } else { 0.0 };
                s.push_str(&format!("  {name:<28} {ms:7.3}  {share:5.1}%\n"));
            }
        } else if self.want_gpu {
            s.push_str("GPU per pass: unavailable (the adapter does not expose TIMESTAMP_QUERY)\n");
        }
        s
    }

    pub fn finish(&self) {
        let text = self.report();
        eprint!("{text}");
        if let Some(p) = &self.out {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = f.write_all(text.as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_are_ordered_and_pick_the_right_tail() {
        let v: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (mean, median, p99, max) = stats(&v);
        assert!((mean - 50.5).abs() < 1e-3);
        assert_eq!(median, 51.0);
        // Index `(n * 0.99) as usize`, the same convention `App::frame_prof_report` already
        // uses — for exactly 100 samples that lands on the last one.
        assert_eq!(p99, 100.0);
        assert_eq!(max, 100.0);
        // an empty sample must not panic or divide by zero
        assert_eq!(stats(&[]), (0.0, 0.0, 0.0, 0.0));
        // a single sample is its own everything
        assert_eq!(stats(&[7.0]), (7.0, 7.0, 7.0, 7.0));
    }

    /// The phase machine must not start measuring before the scene has settled, must discard the
    /// warm-up, and must only ask for the GPU timer during phase 2.
    #[test]
    fn phases_advance_in_order() {
        let mut b = Bench {
            phase: Phase::Waiting,
            frames: 3,
            warmup: 2,
            left: 2,
            label: String::new(),
            out: None,
            want_gpu: true,
            cpu_ms: Vec::new(),
            frame_ms: Vec::new(),
            gpu_cpu_ms: Vec::new(),
            last_entry: None,
            frame_start: None,
            stages: Vec::new(),
            census: String::new(),
            gpu_report: Vec::new(),
            gpu_frame: (0.0, 0.0, 0.0),
            gpu_available: false,
        };
        // while waiting, frames are neither counted nor timed
        for _ in 0..5 {
            b.frame_begin();
            assert!(!b.frame_end([0.0; 6]));
        }
        assert!(b.frame_ms.is_empty() && b.cpu_ms.is_empty());
        assert!(!b.gpu_timer_wanted());
        b.scene_settled();
        // 2 warm-up frames, then 3 measured, then 3 with the timer, then done
        for _ in 0..2 {
            b.frame_begin();
            assert!(!b.frame_end([0.0; 6]));
        }
        assert!(b.frame_ms.is_empty(), "warm-up frames must not be measured");
        for _ in 0..3 {
            b.frame_begin();
            assert!(!b.gpu_timer_wanted(), "phase 1 must run without the GPU timer");
            assert!(!b.frame_end([0.0; 6]));
        }
        assert_eq!(b.cpu_ms.len(), 3);
        assert!(b.gpu_timer_wanted(), "phase 2 must turn the GPU timer on");
        for k in 0..3 {
            b.frame_begin();
            let done = b.frame_end([0.0; 6]);
            assert_eq!(done, k == 2);
        }
        assert!(!b.active());
    }

    /// With the GPU phase disabled the run must finish after phase 1 (adapters with no
    /// timestamp support must not hang the benchmark).
    #[test]
    fn nogpu_finishes_after_phase_one() {
        let mut b = Bench::from_env_for_test(2, 1, false);
        b.scene_settled();
        b.frame_begin();
        assert!(!b.frame_end([0.0; 6])); // warm-up
        b.frame_begin();
        assert!(!b.frame_end([0.0; 6]));
        b.frame_begin();
        assert!(b.frame_end([0.0; 6]), "should be done after {} measured frames", 2);
    }

    impl Bench {
        fn from_env_for_test(frames: u32, warmup: u32, want_gpu: bool) -> Self {
            Self {
                phase: Phase::Waiting,
                frames,
                warmup,
                left: warmup,
                label: String::new(),
                out: None,
                want_gpu,
                cpu_ms: Vec::new(),
                frame_ms: Vec::new(),
                gpu_cpu_ms: Vec::new(),
                last_entry: None,
                frame_start: None,
                stages: Vec::new(),
                census: String::new(),
                gpu_report: Vec::new(),
                gpu_frame: (0.0, 0.0, 0.0),
                gpu_available: false,
            }
        }
    }

    /// The report must be printable with no GPU data and with it.
    #[test]
    fn report_renders_without_gpu_data() {
        let mut b = Bench::from_env_for_test(3, 0, true);
        b.frame_ms = vec![16.0, 17.0, 18.0];
        b.cpu_ms = vec![4.0, 5.0, 6.0];
        let t = b.report();
        assert!(t.contains("frame period"), "{t}");
        assert!(t.contains("update() CPU"), "{t}");
        assert!(t.contains("unavailable"), "{t}");
        b.take_gpu_report(true, vec![("scene-pass".into(), 3.5)], (7.0, 9.0, 11.0));
        let t = b.report();
        assert!(t.contains("scene-pass"), "{t}");
        assert!(t.contains("50.0%"), "the per-pass share must be computed: {t}");
    }
}
