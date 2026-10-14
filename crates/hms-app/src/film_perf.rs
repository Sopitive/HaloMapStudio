//! #film-perf  The Theater frame profiler: a FULL accounting of one playback frame.
//!
//! "FPS is terrible in Theater playback" could not be answered by anything in the app. The
//! existing `HMS_FRAMEPROF` split `update()` into six stages, one of which ("pre") contained the
//! entire film tick — so every Theater cost landed in the same bucket and nothing could say which
//! of the ten things a film frame does was the expensive one. `film perf` now reports every one
//! of them separately, with the distribution rather than the mean, because "laggy" is a p99
//! complaint and a mean hides a hitch.
//!
//! It is ALWAYS ON (not env-gated): the whole cost is [`STAGES`] `Instant::now()` calls per frame,
//! about 400 ns, and a diagnostic that has to be switched on cannot answer a user's complaint
//! about a frame that already happened. Sampling only runs while a film is open.
//!
//! The wall period is sampled separately from the `update()` stages, because a GPU-bound frame
//! shows up only there: `update()` returns quickly and the thread blocks in `present`.

use std::time::Instant;

/// Rolling window, in frames. 600 at 60 Hz is ten seconds of playback — long enough for a p99 to
/// mean something and short enough that it describes what the user is looking at now.
pub const WINDOW: usize = 600;

pub const STAGES: usize = 17;

/// What each stage covers. The order is the order they run in, so the report reads as a timeline.
pub const NAMES: [&str; STAGES] = [
    "pre (menus/windows/cmds)",
    "film clock (advance)",
    "film first-person resolve",
    "film resolve_bipeds",
    "film ground_corpses",
    "film sync (place bodies)",
    "film ragdolls",
    "film appearance",
    "film update_actors (pose)",
    "film camera",
    "film timeline UI",
    "scene tick (Forge + actors)",
    "panels + properties",
    "particle sim",
    "render encode",
    "exposure meter",
    "input + rest",
];

#[derive(Default)]
pub struct FilmProf {
    mark: Option<Instant>,
    entry: Option<Instant>,
    /// The wall period of the frame being measured (previous `update()` entry to this one).
    wall: f32,
    cur: [f32; STAGES],
    /// One row per measured frame: `[0]` = wall period since the previous frame, `[1..]` = stages.
    hist: Vec<[f32; STAGES + 1]>,
    /// Frames seen since the window was last reported (so a report can say how fresh it is).
    pub frames: u64,
}

/// mean / p50 / p95 / p99 / max of a sample set.
pub fn stats(v: &[f32]) -> (f32, f32, f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f32| s[((s.len() as f32 * q) as usize).min(s.len() - 1)];
    (s.iter().sum::<f32>() / s.len() as f32, pick(0.50), pick(0.95), pick(0.99), *s.last().unwrap())
}

impl FilmProf {
    /// Top of `update()`. Starts the stage clock and samples the wall period.
    pub fn frame_begin(&mut self) {
        let now = Instant::now();
        self.wall = self.entry.map(|p| (now - p).as_secs_f32() * 1000.0).unwrap_or(0.0);
        self.entry = Some(now);
        self.mark = Some(now);
        self.cur = [0.0; STAGES];
    }

    /// Charge the time since the previous lap to stage `k`.
    pub fn lap(&mut self, k: usize) {
        let now = Instant::now();
        if let Some(m) = self.mark {
            if let Some(slot) = self.cur.get_mut(k) {
                *slot += (now - m).as_secs_f32() * 1000.0;
            }
        }
        self.mark = Some(now);
    }

    /// End of `update()`. `sampling` is false when no film is open, in which case the frame is
    /// thrown away rather than diluting the window with editor frames.
    pub fn frame_end(&mut self, sampling: bool) {
        if !sampling {
            return;
        }
        self.frames += 1;
        let mut row = [0.0f32; STAGES + 1];
        row[0] = self.wall;
        row[1..].copy_from_slice(&self.cur);
        if self.hist.len() == WINDOW {
            self.hist.remove(0);
        }
        self.hist.push(row);
    }

    pub fn len(&self) -> usize {
        self.hist.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hist.is_empty()
    }

    pub fn clear(&mut self) {
        self.hist.clear();
        self.frames = 0;
    }

    /// The report `film perf` prints: wall period and `update()` CPU as distributions, then every
    /// stage's mean with its share, biggest first, and each stage's own p99 so a stage that only
    /// hitches is visible next to one that is always slow.
    pub fn report(&self) -> String {
        if self.hist.is_empty() {
            return "film perf: no frames sampled yet (open a film; the window fills as it plays)\n".into();
        }
        let n = self.hist.len();
        let wall: Vec<f32> = self.hist.iter().map(|r| r[0]).filter(|w| *w > 0.0).collect();
        let cpu: Vec<f32> = self.hist.iter().map(|r| r[1..].iter().sum()).collect();
        let (wm, w50, w95, w99, wmax) = stats(&wall);
        let (cm, c50, c95, c99, cmax) = stats(&cpu);
        let mut s = String::new();
        s.push_str(&format!("=== film perf: {n} frames ===\n"));
        s.push_str(&format!(
            "frame period : mean {wm:7.2} ms  p50 {w50:7.2}  p95 {w95:7.2}  p99 {w99:7.2}  max {wmax:7.2}   ({:.1} fps mean, {:.1} fps at p99)\n",
            if wm > 0.0 { 1000.0 / wm } else { 0.0 },
            if w99 > 0.0 { 1000.0 / w99 } else { 0.0 }
        ));
        s.push_str(&format!(
            "update() CPU : mean {cm:7.2} ms  p50 {c50:7.2}  p95 {c95:7.2}  p99 {c99:7.2}  max {cmax:7.2}\n"
        ));
        let mut rows: Vec<(f32, f32, f32, &str)> = (0..STAGES)
            .map(|k| {
                let col: Vec<f32> = self.hist.iter().map(|r| r[k + 1]).collect();
                let (m, _, _, p99, max) = stats(&col);
                (m, p99, max, NAMES[k])
            })
            .collect();
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        s.push_str("stage                          mean     share    p99     max\n");
        for (m, p99, max, name) in rows {
            let share = if cm > 0.0 { m / cm * 100.0 } else { 0.0 };
            s.push_str(&format!("  {name:<28} {m:7.3} {share:5.1}% {p99:7.3} {max:7.3}\n"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_pick_the_right_tail() {
        let v: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (mean, p50, p95, p99, max) = stats(&v);
        assert!((mean - 50.5).abs() < 1e-3);
        assert_eq!(p50, 51.0);
        assert_eq!(p95, 96.0);
        assert_eq!(p99, 100.0);
        assert_eq!(max, 100.0);
        assert_eq!(stats(&[]), (0.0, 0.0, 0.0, 0.0, 0.0));
    }

    /// A frame with no film open must not enter the window, and the window must not grow past
    /// `WINDOW` (a 20-minute watch would otherwise hold 72000 rows).
    #[test]
    fn only_film_frames_are_sampled_and_the_window_is_bounded() {
        let mut p = FilmProf::default();
        for _ in 0..5 {
            p.frame_begin();
            p.lap(5);
            p.frame_end(false);
        }
        assert!(p.is_empty(), "editor frames must not be sampled");
        for _ in 0..(WINDOW + 50) {
            p.frame_begin();
            p.lap(5);
            p.frame_end(true);
        }
        assert_eq!(p.len(), WINDOW);
        let t = p.report();
        assert!(t.contains("frame period"), "{t}");
        assert!(t.contains("film sync (place bodies)"), "{t}");
    }

    /// A lap must charge the stage it names, and the report must attribute it there.
    #[test]
    fn a_lap_charges_the_stage_it_names() {
        let mut p = FilmProf::default();
        for _ in 0..10 {
            p.frame_begin();
            std::thread::sleep(std::time::Duration::from_millis(2));
            p.lap(6); // "film ragdolls"
            p.frame_end(true);
        }
        let n = p.len();
        assert_eq!(n, 10);
        let col: Vec<f32> = p.hist.iter().map(|r| r[7]).collect();
        let (m, ..) = stats(&col);
        assert!(m >= 1.5, "the slept 2 ms must land in stage 6, got {m}");
        // and nowhere else
        for k in 0..STAGES {
            if k == 6 {
                continue;
            }
            let other: Vec<f32> = p.hist.iter().map(|r| r[k + 1]).collect();
            let (om, ..) = stats(&other);
            assert!(om < 0.5, "stage {k} should be ~0, got {om}");
        }
    }
}
