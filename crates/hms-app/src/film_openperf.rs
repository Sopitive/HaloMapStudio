//! #film-open-perf  Headless timing of everything a Theater open does on the UI thread.
use std::time::Instant;
use crate::film;
use crate::film_decode as fd;
use crate::film_play::{ActorDriver, Playback};

fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1000.0 }

pub(crate) fn stage_table(path: &std::path::Path) -> Option<Vec<(&'static str, f64)>> {
    let mut rows: Vec<(&'static str, f64)> = Vec::new();
    let t = Instant::now();
    let f = film::open(path).ok()?;
    rows.push(("worker: film::open", ms(t)));
    let t = Instant::now();
    let idx = fd::build_index_film(&f);
    rows.push(("worker: build_index_film", ms(t)));
    let mut idx = idx;
    let t = Instant::now();
    idx.attach_derived();
    rows.push(("worker: (of which) attach_derived", ms(t)));
    // ---- everything below is what the UI thread does ----
    let t = Instant::now();
    let pb = Playback::new(idx);
    rows.push(("UI: Playback::new", ms(t)));
    let t = Instant::now();
    let st = crate::film_stats::compute(&f, &pb.index, &crate::film_stats::NoNames);
    rows.push(("UI: film_compute_stats", ms(t)));
    drop(st);
    let t = Instant::now();
    let names = std::collections::HashMap::new();
    let lines = crate::film_ui::roster_lines(&f, &pb, &names);
    rows.push(("UI: roster_lines", ms(t)));
    let t = Instant::now();
    let _ = crate::film_ui::roster_coverage(&pb, f.present_players().count());
    rows.push(("UI: roster_coverage", ms(t)));
    let _ = lines;
    let t = Instant::now();
    let mut d = ActorDriver::default();
    d.prepare(&pb, &|_, _| None);
    rows.push(("UI: ActorDriver::prepare", ms(t)));
    let t = Instant::now();
    let _ = pb.deaths();
    rows.push(("UI: pb.deaths", ms(t)));
    let t = Instant::now();
    let _ = pb.assassinations();
    rows.push(("UI: pb.assassinations", ms(t)));
    let t = Instant::now();
    let _ = pb.camera_timeline();
    rows.push(("UI: camera_timeline", ms(t)));
    let t = Instant::now();
    let _ = crate::film_anim::recorded_rides(&pb.index);
    rows.push(("UI: recorded_rides", ms(t)));
    let t = Instant::now();
    let _ = crate::film_anim::find_rides(&pb.index);
    rows.push(("UI: find_rides", ms(t)));
    // ---- frame steps ----
    let total = pb.total_ticks().max(2);
    let mut pb = pb;
    let (mut worst, mut sum, mut n) = (0.0f64, 0.0f64, 0u32);
    let (mut w_cap, mut s_cap) = (0.0f64, 0.0f64);
    for k in 0..40u32 {
        pb.tick = (total as f32 - 1.0) * (k as f32 / 39.0);
        let t = Instant::now();
        let (states, tot) = d.states_and_total(&pb);
        let a = ms(t);
        let t = Instant::now();
        let c = d.caption_with(&pb, &states, tot);
        let b = ms(t);
        let t = Instant::now();
        let _ = crate::film_play::solve_camera_at(&pb, d.eye_height(&pb));
        let cc = ms(t);
        let _ = c;
        let frame = a + b + cc;
        worst = worst.max(frame);
        sum += frame;
        w_cap = w_cap.max(b);
        s_cap += b;
        n += 1;
    }
    rows.push(("frame: states+caption+camera mean", sum / n as f64));
    rows.push(("frame: states+caption+camera worst", worst));
    rows.push(("frame: caption worst", w_cap));
    let _ = s_cap;
    rows.push(("meta: ticks", total as f64));
    rows.push(("meta: tracks", pb.index.tracks.len() as f64));
    rows.push(("meta: events", pb.index.events.len() as f64));
    rows.push(("meta: position keys (biped)", pb.index.tracks.iter().filter(|t| t.entity_type == 11).map(|t| t.positions.len()).sum::<usize>() as f64));
    let t = Instant::now();
    let mut n = 0u32;
    for tr in pb.index.tracks.iter().filter(|t| t.entity_type == 11).take(300) {
        for w in tr.positions.windows(2).take(200) {
            let _ = crate::film_anim::derive(tr, w[0].0, None);
            n += 1;
        }
    }
    rows.push(("meta: derive us/call", ms(t) * 1000.0 / n.max(1) as f64));
    Some(rows)
}

#[test]
#[ignore = "diagnostic: prints the open-path stage table"]
fn stage_table_probe() {
    let mut v = film::spec_corpus();
    v.sort();
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    for p in v {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.is_empty() && !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Some(rows) = stage_table(&p) else { continue };
        println!("=== {name}");
        for (k, v) in rows { println!("  {k:<40} {v:>10.2}"); }
    }
}

// ---- the map-backed half of the open path (GPU-free: the native library + an `ActorScene`) ----

pub(crate) struct OpenMap {
    pub native: std::sync::Arc<hms_native::NativeDll>,
    pub cache: hms_native::CacheHandle,
    pub map: std::path::PathBuf,
    /// The suite-wide native-library lock (`reach_actor::tests`): every map-backed gate holds it, so
    /// two of them never drive the one native library at once (the cause of a gate that passed alone
    /// and found no carriers in the parallel suite).
    _guard: std::sync::MutexGuard<'static, ()>,
}

pub(crate) fn open_forge() -> Option<OpenMap> {
    let (guard, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
    let m = crate::mapcat::enumerate().into_iter().find(|c| {
        c.game == crate::mapcat::Game::Reach
            && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
    })?;
    Some(OpenMap { native, cache, map: m.path, _guard: guard })
}

/// What the UI thread does with a MAP once the index lands: names, spawn weapons, the biped
/// resolve, then the first frames of `sync`. Returns `(stage, ms)` rows.
pub(crate) fn map_stage_table(o: &OpenMap, f: &film::Film, idx: fd::FilmIndex) -> Vec<(&'static str, f64)> {
    let mut rows = Vec::new();
    let mut pb = Playback::new(idx);
    let t = Instant::now();
    let names = crate::film_stats::MapNames::build(&o.native, o.cache);
    rows.push(("map: MapNames::build (stats)", ms(t)));
    let t = Instant::now();
    let st = crate::film_stats::compute(f, &pb.index, &names);
    rows.push(("map: film_stats::compute (named)", ms(t)));
    drop(st);
    let t = Instant::now();
    let vn = crate::film_variant::VariantNames::build(&o.native, o.cache);
    let weap = o.native.tags_by_group(o.cache, "weap");
    rows.push(("map: VariantNames+weap tags", ms(t)));
    if let Some(gv) = f.header.gametype.clone() {
        let t = Instant::now();
        let _ = crate::film_variant::apply_spawn_weapons(&mut pb.index.tracks, crate::film_play::BIPED_TYPE, &gv, &vn, &weap);
        rows.push(("map: apply_spawn_weapons", ms(t)));
    }
    let mut d = ActorDriver::default();
    d.cap = 64;
    let mut actors = crate::reach_actor::ActorScene::default();
    let t = Instant::now();
    d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
    rows.push(("map: resolve_bipeds", ms(t)));
    let t = Instant::now();
    d.sync(&pb, &mut actors, &o.native, o.cache);
    rows.push(("map: first sync (tick 0)", ms(t)));
    let total = pb.total_ticks().max(2);
    let (mut worst, mut sum) = (0.0f64, 0.0f64);
    let mut each: Vec<(f64, u32)> = Vec::new();
    for k in 1..=30u32 {
        pb.seek_tick((total as f32 - 1.0) * (k as f32 / 30.0));
        let t = Instant::now();
        d.sync(&pb, &mut actors, &o.native, o.cache);
        let e = ms(t);
        each.push((e, pb.tick as u32));
        worst = worst.max(e);
        sum += e;
    }
    each.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("  slowest seeks (ms, tick): {:?}", &each[..4.min(each.len())]);
    println!("  sync profile: {}", d.prof.report());
    rows.push(("map: sync mean over 30 seeks", sum / 30.0));
    rows.push(("map: sync worst over 30 seeks", worst));
    // steady playback: consecutive ticks from the middle
    pb.seek_tick(total as f32 * 0.5);
    let (mut worst, mut sum) = (0.0f64, 0.0f64);
    for k in 0..120u32 {
        pb.seek_tick(total as f32 * 0.5 + k as f32);
        let t = Instant::now();
        d.sync(&pb, &mut actors, &o.native, o.cache);
        let e = ms(t);
        worst = worst.max(e);
        sum += e;
    }
    rows.push(("map: sync mean, 120 consecutive ticks", sum / 120.0));
    rows.push(("map: sync worst, 120 consecutive ticks", worst));
    rows
}

#[test]
#[ignore = "diagnostic: prints the map-backed open-path stage table"]
fn map_stage_table_probe() {
    let Some(o) = open_forge() else { println!("no forge_halo / native library: skipped"); return };
    let mut v = film::spec_corpus();
    v.sort();
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    for p in v {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.is_empty() && !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Ok(f) = film::open(&p) else { continue };
        if !name.contains("forge_hal") { continue }
        let idx = fd::build_index_film(&f);
        println!("=== {name}");
        for (k, v) in map_stage_table(&o, &f, idx) { println!("  {k:<44} {v:>10.2}"); }
    }
}

// =============================================================================================
// THE GATE  (#film-open-perf)
//
// MEASURED (release, this machine, the two largest corpus films; stage table in CHANGELOG):
//   UI-thread work when the index lands, BEFORE : ~12,700 - 15,100 ms  (film_compute_stats' gait
//                                                  census ran `derive` -- and its ~600-probe
//                                                  `flight_episode` -- on every position key)
//   UI-thread work when the index lands, AFTER  : ~12 - 18 ms   (census + rides folded on the worker)
//   worst single frame step (states + caption + camera) : 1.4 - 1.9 ms
// Budgets are ~6x the measured fixed cost: loose enough for a loaded box, three orders of
// magnitude under the regression, and the saboteur below proves the open budget bites.
// =============================================================================================
#[cfg(test)]
mod gate {
    use super::*;

    /// UI-thread open budget, ms (measured ~12-25 alone, up to ~105 while the whole suite runs in
    /// parallel; the regression was ~12,000+, so 250 still bites by a factor of ~50 and no longer
    /// flakes under load).
    const UI_OPEN_BUDGET_MS: f64 = 250.0;
    /// One frame step's budget, ms (measured worst 1.4-1.9).
    const FRAME_STEP_BUDGET_MS: f64 = 12.0;

    fn largest() -> Option<(String, film::Film)> {
        let mut best: Option<(u64, std::path::PathBuf)> = None;
        for p in film::spec_corpus() {
            let Ok(m) = std::fs::metadata(&p) else { continue };
            if best.as_ref().map_or(true, |(s, _)| m.len() > *s) {
                best = Some((m.len(), p));
            }
        }
        let (_, p) = best?;
        let name = p.file_name()?.to_string_lossy().into_owned();
        Some((name, film::open(&p).ok()?))
    }

    /// Everything the UI thread runs when the decode worker's index lands, in `film_tick` order.
    fn ui_open_ms(f: &film::Film, idx: fd::FilmIndex) -> (f64, Playback) {
        let t = Instant::now();
        let pb = Playback::new(idx);
        let _ = crate::film_stats::compute(f, &pb.index, &crate::film_stats::NoNames);
        let _ = crate::film_ui::roster_lines(f, &pb, &std::collections::HashMap::new());
        let _ = crate::film_ui::roster_coverage(&pb, f.present_players().count());
        let mut d = ActorDriver::default();
        d.prepare(&pb, &|_, _| None);
        let _ = (pb.deaths(), pb.assassinations(), pb.camera_timeline());
        (ms(t), pb)
    }

    fn check(what: &str, measured: f64, budget: f64) -> Result<(), String> {
        if measured <= budget { Ok(()) } else { Err(format!("{what}: {measured:.1} ms over the {budget:.0} ms budget")) }
    }

    /// The open path and a frame step stay inside their budgets on the largest film.
    #[test]
    fn open_and_frame_steps_stay_in_budget() {
        let Some((name, f)) = largest() else { eprintln!("no films on this machine: gate skipped"); return };
        let idx = fd::build_index_film(&f); // the WORKER's job, not the UI thread's
        assert!(idx.derived_valid().is_some(), "the worker must have folded the rides and the census");
        let (open, mut pb) = ui_open_ms(&f, idx);
        let d = ActorDriver::default();
        let mut worst = 0.0f64;
        let total = pb.total_ticks().max(2);
        for k in 0..60u32 {
            pb.tick = (total as f32 - 1.0) * (k as f32 / 59.0);
            let t = Instant::now();
            let (states, tot) = d.states_and_total(&pb);
            let _ = d.caption_with(&pb, &states, tot);
            let _ = crate::film_play::solve_camera_at(&pb, d.eye_height(&pb));
            worst = worst.max(ms(t));
        }
        println!("film {name}: UI open {open:.1} ms (budget {UI_OPEN_BUDGET_MS}), worst frame step {worst:.2} ms (budget {FRAME_STEP_BUDGET_MS})");
        if cfg!(debug_assertions) {
            eprintln!("debug build: budgets are release-only, not asserted");
            return;
        }
        check("UI-thread open", open, UI_OPEN_BUDGET_MS).unwrap();
        check("worst frame step", worst, FRAME_STEP_BUDGET_MS).unwrap();
    }

    /// SABOTEUR: take the folded results away (what the code did before: the census and the ride
    /// detector run on the UI thread when the index lands) and the SAME check must go red.
    #[test]
    fn saboteur_eager_census_on_the_ui_thread_trips_the_budget() {
        if cfg!(debug_assertions) { return }
        let Some((_, f)) = largest() else { return };
        let mut idx = fd::build_index_film(&f);
        idx.derived = None; // the slow path: nothing precomputed
        let (open, _) = ui_open_ms(&f, idx);
        println!("saboteur: eager census took {open:.1} ms against the {UI_OPEN_BUDGET_MS} ms budget");
        let r = check("UI-thread open (saboteur)", open, UI_OPEN_BUDGET_MS);
        assert!(r.is_err(), "the gate failed to bite: the eager path took only {open:.1} ms");
    }

    /// The speed-ups change no answer: the memoised flight search, the per-track table and the
    /// worker-folded census all equal the plain computation.
    #[test]
    fn the_fast_paths_give_the_same_answers() {
        let Some((_, f)) = largest() else { return };
        let idx = fd::build_index_film(&f);
        let mut keys = 0usize;
        for tr in idx.tracks.iter().filter(|t| t.entity_type == 11 && t.positions.len() > 8).take(40) {
            let map = crate::film_anim::MovingMap::build(tr);
            for w in tr.positions.windows(2) {
                for k in [w[0].0, w[0].0 + 1, (w[0].0 + w[1].0) / 2] {
                    assert_eq!(crate::film_anim::flight_episode(tr, k), crate::film_anim::flight_episode_direct(tr, k), "flight_episode memo differs at {k}");
                    assert_eq!(crate::film_anim::derive_mapped(tr, k, None, &map), crate::film_anim::derive(tr, k, None), "derive_mapped differs at {k}");
                    keys += 1;
                }
            }
        }
        assert!(keys > 1000, "the comparison covered only {keys} ticks: vacuous");
        // the census: folded-by-worker == folded-in-place, field for field
        let with = crate::film_stats::compute(&f, &idx, &crate::film_stats::NoNames);
        let mut bare = idx.clone();
        bare.derived = None;
        let without = crate::film_stats::compute(&f, &bare, &crate::film_stats::NoNames);
        assert!(!with.players.is_empty());
        assert_eq!(format!("{:?}", with.players), format!("{:?}", without.players));
        assert_eq!(crate::film_anim::accepted_rides_cached(&idx), crate::film_anim::accepted_rides(&bare));
    }
}

/// #film-knives  Census: every non-biped object, by tag leaf and verdict, with how many state a
/// parent and the type of that parent.
#[test]
#[ignore = "diagnostic"]
fn object_census_probe() {
    let Some(o) = open_forge() else { println!("no map"); return };
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    let mut v: Vec<std::path::PathBuf> = film::spec_corpus();
    v.extend(std::fs::read_dir(std::path::Path::new(&std::env::var("HOME").unwrap()).join("Downloads")).into_iter().flatten().flatten().map(|e| e.path()));
    v.sort();
    for p in v {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.is_empty() && !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Ok(f) = film::open(&p) else { continue };
        let idx = fd::build_index_film(&f);
        let pb = Playback::new(idx);
        let mut d = ActorDriver::default();
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let Some(ot) = d.objects.as_ref() else { println!("=== {name}: no object table"); continue };
        let by_net: std::collections::HashMap<u32, &fd::EntityTrack> = pb.index.tracks.iter().map(|t| (t.net_id, t)).collect();
        let mut rows: std::collections::BTreeMap<String, (usize, usize, usize, u64, std::collections::BTreeMap<u8, usize>)> = Default::default();
        for (net, id) in &ot.objects {
            let Some(t) = by_net.get(net) else { continue };
            let k = format!("{} {} [{}]", id.group, id.name, id.draw.label());
            let e = rows.entry(k).or_default();
            e.0 += 1;
            let att: Vec<_> = t.attachments.iter().filter_map(|(tk, a)| a.map(|a| (*tk, a))).collect();
            if !att.is_empty() {
                e.1 += 1;
                for (_, a) in &att { if let Some(pt) = by_net.get(&a.parent) { *e.4.entry(pt.entity_type).or_default() += 1; } }
            }
            if t.positions.len() > 1 { e.2 += 1; }
            e.3 += (t.last_tick.saturating_sub(t.first_tick)) as u64;
        }
        println!("=== {name}");
        for (k, (n, att, mov, life, ptypes)) in rows {
            if n < 1 { continue }
            println!("  {k:<60} n={n:<5} with_parent={att:<5} moving={mov:<5} mean_life={:<6} parent_types={ptypes:?}", life / n as u64);
        }
    }
}

/// #film-knives  How do loose weapons MOVE, and are they near any biped while they do?
#[test]
#[ignore = "diagnostic"]
fn weapon_motion_probe() {
    let Some(o) = open_forge() else { println!("no map"); return };
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    for p in film::spec_corpus() {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.is_empty() && !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Ok(f) = film::open(&p) else { continue };
        let pb = Playback::new(fd::build_index_film(&f));
        let mut d = ActorDriver::default();
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let Some(ot) = d.objects.as_ref() else { continue };
        let bips: Vec<&fd::EntityTrack> = pb.index.tracks.iter().filter(|t| t.entity_type == 11).collect();
        let (mut tot, mut drifting, mut near, mut long_gap_keys, mut keys) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut worst: Vec<(f32, u32, u32, String)> = Vec::new();
        for t in pb.index.tracks.iter() {
            let Some(id) = ot.objects.get(&t.net_id) else { continue };
            if id.group != "weap" { continue }
            for w in t.positions.windows(2) {
                keys += 1;
                let gap = w[1].0 - w[0].0;
                let disp = ((w[1].1[0]-w[0].1[0]).powi(2)+(w[1].1[1]-w[0].1[1]).powi(2)+(w[1].1[2]-w[0].1[2]).powi(2)).sqrt();
                if gap > 30 && disp > 3.0 { long_gap_keys += 1; worst.push((disp, gap, t.net_id, id.name.clone())); }
            }
            let (a, b) = (t.first_tick, t.last_tick);
            let mut k = a;
            while k <= b {
                if let Some(pos) = t.sample(k) {
                    tot += 1;
                    let v = { let p2 = t.sample(k + 6).unwrap_or(pos); ((p2[0]-pos[0]).powi(2)+(p2[1]-pos[1]).powi(2)+(p2[2]-pos[2]).powi(2)).sqrt() / 0.1 };
                    let nb = bips.iter().filter(|b| k >= b.first_tick && k <= b.last_tick).filter_map(|b| b.sample(k)).map(|q| ((q[0]-pos[0]).powi(2)+(q[1]-pos[1]).powi(2)+(q[2]-pos[2]).powi(2)).sqrt()).fold(f32::MAX, f32::min);
                    if nb < 2.0 { near += 1; }
                    if v > 2.0 && nb > 4.0 { drifting += 1; }
                }
                k += 30;
            }
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("=== {name}: weap samples {tot}, near a biped(<2wu) {near}, MOVING>2wu/s while >4wu from every biped {drifting}; key pairs {keys}, with gap>30t & jump>3wu: {long_gap_keys}; biggest jumps (wu,gap,net,tag) {:?}", &worst[..worst.len().min(5)]);
    }
}

/// #film-knives  For objects that state a parent: how far is the DRAWN position (track.sample, what
/// the debris lane uses) from the parent's position while attached?
#[test]
#[ignore = "diagnostic"]
fn attached_drift_probe() {
    let Some(o) = open_forge() else { println!("no map"); return };
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    let mut v: Vec<std::path::PathBuf> = film::spec_corpus();
    v.extend(std::fs::read_dir(std::path::Path::new(&std::env::var("HOME").unwrap()).join("Downloads")).into_iter().flatten().flatten().map(|e| e.path()));
    for p in v {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.is_empty() && !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Ok(f) = film::open(&p) else { continue };
        let pb = Playback::new(fd::build_index_film(&f));
        let mut d = ActorDriver::default();
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let Some(ot) = d.objects.as_ref() else { continue };
        let by: std::collections::HashMap<u32, &fd::EntityTrack> = pb.index.tracks.iter().map(|t| (t.net_id, t)).collect();
        let mut rows: std::collections::BTreeMap<String, (usize, usize, f32, f32, usize, usize)> = Default::default();
        for t in &pb.index.tracks {
            let Some(id) = ot.objects.get(&t.net_id) else { continue };
            if t.attachments.is_empty() { continue }
            let e = rows.entry(format!("{} {}", id.group, id.name)).or_default();
            let mut sum = 0.0f32; let mut n = 0usize; let mut mx = 0.0f32; let mut far = 0usize;
            for (i, (tk, a)) in t.attachments.iter().enumerate() {
                let Some(a) = a else { continue };
                let end = t.attachments.get(i + 1).map(|x| x.0).unwrap_or(t.last_tick);
                let Some(par) = by.get(&a.parent) else { continue };
                let mut k = *tk;
                while k < end {
                    if let (Some(c), Some(pp)) = (t.sample(k), par.sample(k)) {
                        let dd = ((c[0]-pp[0]).powi(2)+(c[1]-pp[1]).powi(2)+(c[2]-pp[2]).powi(2)).sqrt();
                        sum += dd; n += 1; mx = mx.max(dd); if dd > 3.0 { far += 1; }
                    }
                    k += 20;
                }
            }
            e.0 += 1; e.1 += n; e.2 += sum; e.3 = e.3.max(mx); e.4 += far;
            // stale: how many position keys does it state at all while attached?
            e.5 += t.positions.len();
        }
        println!("=== {name}");
        for (k, (objs, n, sum, mx, far, keys)) in rows {
            println!("  {k:<44} objs={objs:<6} attached_samples={n:<7} mean_dist={:.2} max_dist={mx:.1} samples>3wu={far} pos_keys={keys}", sum / n.max(1) as f32);
        }
    }
}

#[test]
#[ignore = "diagnostic"]
fn attachment_offsets_probe() {
    let Some(o) = open_forge() else { return };
    let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
    let mut v: Vec<std::path::PathBuf> = film::spec_corpus();
    v.extend(std::fs::read_dir(std::path::Path::new(&std::env::var("HOME").unwrap()).join("Downloads")).into_iter().flatten().flatten().map(|e| e.path()));
    for p in v {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !want.iter().any(|w| name.contains(w.as_str())) { continue }
        let Ok(f) = film::open(&p) else { continue };
        let pb = Playback::new(fd::build_index_film(&f));
        let mut d = ActorDriver::default();
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let Some(ot) = d.objects.as_ref() else { continue };
        println!("=== {name}");
        let mut shown = std::collections::HashMap::<String, usize>::new();
        let mut nodes = std::collections::BTreeMap::<(String, u8), usize>::new();
        let (mut cleared, mut set) = (0, 0);
        for t in &pb.index.tracks {
            let Some(id) = ot.objects.get(&t.net_id) else { continue };
            if t.attachments.is_empty() { continue }
            for (tk, a) in &t.attachments {
                match a { Some(a) => { set += 1; *nodes.entry((id.name.clone(), a.node)).or_default() += 1;
                    let n = shown.entry(id.name.clone()).or_default(); *n += 1;
                    if *n <= 4 { println!("  {} net {:#x} @{tk}: parent {:#x} node {} off {:?} mode {} seat {:?}  | life {}..{} pos_keys {} first_pos {:?}", id.name, t.net_id, a.parent, a.node, a.offset, a.mode, a.seat, t.first_tick, t.last_tick, t.positions.len(), t.positions.first().map(|p| p.1)); } }
                    None => cleared += 1 }
            }
        }
        println!("  set {set}, cleared {cleared}; nodes {:?}", nodes);
    }
}

#[cfg(test)]
mod carry_probe {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn carried_weapon_census() {
        let Some(o) = open_forge() else { return };
        let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
        let mut v = film::spec_corpus();
        v.extend(std::fs::read_dir(std::path::Path::new(&std::env::var("HOME").unwrap()).join("Downloads")).into_iter().flatten().flatten().map(|e| e.path()));
        for p in v {
            let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if !want.iter().any(|w| name.contains(w.as_str())) { continue }
            let Ok(f) = film::open(&p) else { continue };
            let pb = Playback::new(fd::build_index_film(&f));
            let mut d = ActorDriver::default();
            d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
            let Some(ot) = d.objects.as_ref() else { continue };
            let weap_nets: std::collections::HashSet<u32> = ot.objects.iter().filter(|(_, i)| i.group == "weap").map(|(n, _)| *n).collect();
            let mut by_kind: std::collections::BTreeMap<u8, usize> = Default::default();
            let (mut pick_w, mut drop_w) = (std::collections::HashSet::new(), std::collections::HashSet::new());
            let mut ex = vec![];
            for e in &pb.index.events {
                if e.kind == 32 || e.kind == 34 || e.kind == crate::film_action::EV_PICKUP {
                    *by_kind.entry(e.kind).or_default() += 1;
                    for n in [e.subject, e.other].into_iter().flatten() {
                        if weap_nets.contains(&n) { if e.kind == 34 { drop_w.insert(n); } else { pick_w.insert(n); } }
                    }
                    if ex.len() < 3 && e.kind == 32 { ex.push((e.tick, e.subject, e.other)); }
                }
            }
            let bip_nets: std::collections::HashSet<u32> = pb.index.tracks.iter().filter(|t| t.entity_type == 11).map(|t| t.net_id).collect();
            let with_weapon_field = pb.index.tracks.iter().filter(|t| t.entity_type == 11 && !t.weapons.is_empty()).count();
            let with_spawn_weapon = pb.index.tracks.iter().filter(|t| t.entity_type == 11 && t.spawn_weapon.is_some()).count();
            println!("=== {name}: weap entities {}, event counts {:?}, weap entities with a pickup {}, with a drop {}; bipeds {} (with weapon field {with_weapon_field}, with spawn weapon {with_spawn_weapon}); sample pickups {:?}; subject is biped for first: {:?}", weap_nets.len(), by_kind, pick_w.len(), drop_w.len(), bip_nets.len(), ex, ex.first().map(|e| (e.1.map_or(false, |n| bip_nets.contains(&n)), e.2.map_or(false, |n| weap_nets.contains(&n)))));
        }
    }

    #[test]
    #[ignore = "diagnostic"]
    fn carry_distance_histogram() {
        let Some(o) = open_forge() else { return };
        let want: Vec<String> = std::env::var("HMS_FILM_SUBSTR").unwrap_or_default().split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect();
        for p in film::spec_corpus() {
            let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if !want.iter().any(|w| name.contains(w.as_str())) { continue }
            let Ok(f) = film::open(&p) else { continue };
            let pb = Playback::new(fd::build_index_film(&f));
            let mut d = ActorDriver::default();
            d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
            let Some(ot) = d.objects.as_ref() else { continue };
            let bips: Vec<&fd::EntityTrack> = pb.index.tracks.iter().filter(|t| t.entity_type == 11).collect();
            let nearest = |k: u32, pos: [f32; 3]| -> Option<(u32, f32)> {
                bips.iter().filter(|b| k >= b.first_tick && k <= b.last_tick && b.destroyed_tick.map_or(true, |dt| k < dt))
                    .filter_map(|b| b.sample(k).map(|q| (b.net_id, ((q[0]-pos[0]).powi(2)+(q[1]-pos[1]).powi(2)+(q[2]-pos[2]).powi(2)).sqrt())))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            };
            let edges = [0.3f32, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 99.0];
            let mut hist = [0usize; 8];
            let (mut samples, mut stable, mut alive_total) = (0usize, 0usize, 0usize);
            let mut stable_hist = [0usize; 8];
            for t in &pb.index.tracks {
                let Some(id) = ot.objects.get(&t.net_id) else { continue };
                if id.group != "weap" { continue }
                let mut k = t.first_tick + 30;
                while k <= t.last_tick {
                    if let Some(pos) = t.sample(k) {
                        alive_total += 1;
                        if let Some((bn, dd)) = nearest(k, pos) {
                            samples += 1;
                            let bi = edges.iter().position(|e| dd <= *e).unwrap();
                            hist[bi] += 1;
                            let same = |kk: u32| t.sample(kk).and_then(|pp| nearest(kk, pp)).map_or(false, |(n2, d2)| n2 == bn && d2 < 1.5);
                            if dd < 1.5 && same(k - 15) && same(k - 30) { stable += 1; stable_hist[bi] += 1; }
                        }
                    }
                    k += 15;
                }
            }
            println!("=== {name}: weapon samples {alive_total}; nearest-biped distance histogram (<=0.3,0.6,0.9,1.2,1.5,2,3,more): {hist:?}; stable carry (same biped <1.5wu over 30 ticks) {stable} -> by distance {stable_hist:?}");
        }
    }
}

// =============================================================================================
// #film-carry  A weapon a Spartan CARRIES is drawn in its hand, not as a loose object beside it.
// =============================================================================================
#[cfg(test)]
mod carry_gate {
    use super::*;

    #[derive(Debug, Default, Clone, Copy)]
    struct Tally {
        /// loose-lane weapons drawn within 1.5 wu of a live body
        loose_near_body: usize,
        /// carried weapons the loose lane STILL drew (must be 0)
        carried_but_loose: usize,
        /// carriers and how many of them hold the derived weapon in the hand actor
        carriers: usize,
        in_hand: usize,
        /// worst distance, hand node -> the carried weapon's recorded position
        worst_hand_wu: f32,
    }

    fn measure(o: &OpenMap, f: &film::Film, derive: bool) -> Tally {
        let mut pb = Playback::new(fd::build_index_film(f));
        pb.moving_only = false;
        let mut d = ActorDriver::default();
        d.cap = 64;
        d.derive_carried = derive;
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let mut actors = crate::reach_actor::ActorScene::default();
        let total = pb.total_ticks();
        let mut t = Tally::default();
        for k in 1..=7u32 {
            pb.seek_tick((total / 8 * k) as f32);
            d.sync(&pb, &mut actors, &o.native, o.cache);
            actors.update_poses_headless(0.0);
            let tick = pb.tick as u32;
            let park = glam::Vec3::from(d.park);
            let Some(ot) = d.objects.as_ref() else { continue };
            // the loose lane
            for (net, id) in d.object_actor.iter() {
                let Some(a) = actors.get(*id) else { continue };
                let c = a.world().w_axis.truncate();
                if (c - park).length() < 1.0 { continue }
                let Some(ident) = ot.ident(*net) else { continue };
                if ident.group != "weap" { continue }
                if d.carry_hidden.contains(net) { t.carried_but_loose += 1; }
                let near = d.body_actor.values().filter_map(|b| actors.get(*b)).any(|b| (b.world().w_axis.truncate() - c).length() < 1.5);
                if near {
                    t.loose_near_body += 1;
                    if std::env::var("HMS_CARRY_WHY").is_ok() {
                        let wt = pb.index.tracks.iter().find(|x| x.net_id == *net).unwrap();
                        let bodies: Vec<&fd::EntityTrack> = pb.index.tracks.iter().filter(|x| x.entity_type == 11).collect();
                        let nr = |k: u32| -> Option<(u32, f32)> { let p = wt.sample(k)?; bodies.iter().filter(|b| k >= b.first_tick && k <= b.last_tick && pb.death_at(b).map_or(true, |dd| k < dd)).filter_map(|b| b.sample(k).map(|q| (b.net_id, ((q[0]-p[0]).powi(2)+(q[1]-p[1]).powi(2)+(q[2]-p[2]).powi(2)).sqrt()))).min_by(|a, b| a.1.total_cmp(&b.1)) };
                        println!("   still loose near body: tick {tick} {} net {:#x} now {:?} -15 {:?} -30 {:?} age {} draw {:?}", ident.name, net, nr(tick), nr(tick.saturating_sub(15)), nr(tick.saturating_sub(30)), tick - wt.first_tick, ident.draw);
                    }
                }
            }
            // the hands
            for (carrier, widx) in d.carry_hand.iter() {
                let Some(bid) = d.body_actor.get(carrier) else { continue };
                let Some(b) = actors.get(*bid) else { continue };
                t.carriers += 1;
                let want = d.weap_tags.get(*widx as usize).copied();
                let Some(w) = b.weapon.as_ref().filter(|w| Some(w.obj_tag) == want) else { continue };
                t.in_hand += 1;
                // hand node in world space
                let hand = (b.world() * b.pose.model.get(w.parent_node.max(0) as usize).copied().unwrap_or(glam::Mat4::IDENTITY) * w.parent_local).w_axis.truncate();
                // the weapon entity this carrier holds: the nearest carried one to the hand
                let mut best = f32::MAX;
                for net in d.carry_hidden.iter() {
                    if let Some(tr) = pb.index.tracks.iter().find(|t| t.net_id == *net) {
                        if let Some(p) = tr.sample(tick) { best = best.min((glam::Vec3::from(p) - hand).length()); }
                    }
                }
                if best < f32::MAX { t.worst_hand_wu = t.worst_hand_wu.max(best); }
                if best > 2.0 { println!("   far hand: tick {tick} carrier {carrier:#x} body_origin {:?} hand {:?} parent_node {} anim {:?} nearest_carried_weapon_dist {best:.2}", b.world().w_axis.truncate(), hand, w.parent_node, b.anim_name); }
            }
        }
        println!("   tags that could not be held: {:?}", d.hold_failed.iter().map(|t| o.native.tag_name(o.cache, *t).unwrap_or_default()).collect::<Vec<_>>());
        t
    }

    fn film_named(sub: &str) -> Option<film::Film> {
        film::spec_corpus().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().contains(sub))).and_then(|p| film::open(&p).ok())
    }
    const CLIPS: [&str; 2] = ["53C14E9D", "2BAB2586"];

    #[test]
    #[ignore = "diagnostic: before/after table"]
    fn carry_table() {
        let Some(o) = open_forge() else { return };
        for c in CLIPS {
            let Some(f) = film_named(c) else { continue };
            println!("{c}\n  before: {:?}\n  after:  {:?}", measure(&o, &f, false), measure(&o, &f, true));
            let _ = ();
        }
    }

    /// Hand-carried weapons are drawn by their carrier, in its hand, within 2 wu of where the
    /// recording put the weapon; the loose lane draws none of them and far fewer sit beside bodies.
    #[test]
    fn carried_weapons_are_in_the_hand_and_not_loose() {
        let Some(o) = open_forge() else { eprintln!("no forge_halo / native library: gate skipped"); return };
        for c in CLIPS {
            let f = film_named(c).unwrap_or_else(|| panic!("fixture film {c} missing: the gate cannot run"));
            let after = measure(&o, &f, true);
            let before = measure(&o, &f, false);
            println!("{c}: before {before:?}\n      after  {after:?}");
            assert!(after.carriers >= 3, "{c}: liveness -- only {} carriers were found", after.carriers);
            assert!(after.in_hand >= 1, "{c}: no carried weapon reached a hand ({after:?})");
            assert_eq!(after.carried_but_loose, 0, "{c}: a carried weapon was also drawn loose");
            assert!(after.worst_hand_wu <= 2.0, "{c}: a hand is {:.2} wu from the weapon it carries", after.worst_hand_wu);
            assert!(before.loose_near_body >= 3, "{c}: liveness -- the old path had only {} loose weapons beside bodies", before.loose_near_body);
            assert!(after.loose_near_body + after.carriers / 2 <= before.loose_near_body, "{c}: loose weapons beside bodies {} -> {} with {} carriers", before.loose_near_body, after.loose_near_body, after.carriers);
        }
    }

    /// SABOTEUR: turn the derivation off and the same conditions must fail.
    #[test]
    fn saboteur_without_the_carry_resolver_the_gate_fails() {
        let Some(o) = open_forge() else { return };
        let f = film_named(CLIPS[0]).expect("fixture film missing");
        let off = measure(&o, &f, false);
        let on = measure(&o, &f, true);
        assert!(off.in_hand == 0 && off.carriers == 0, "saboteur: nothing should be carried when derivation is off: {off:?}");
        assert!(on.loose_near_body + on.carriers / 2 <= off.loose_near_body, "the gate's reduction condition does not separate on {on:?} from off {off:?}");
        // ...and the gate's own conditions reject the off run:
        assert!(off.carriers < 3 && off.in_hand < 1, "the off run would pass the liveness conditions: {off:?}");
    }
}

#[cfg(test)]
mod light_model_probe {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn do_the_light_and_beacon_have_models() {
        let Some(o) = open_forge() else { return };
        let mut actors = crate::reach_actor::ActorScene::default();
        for g in ["bloc", "weap"] {
            for t in o.native.tags_by_group(o.cache, g) {
                let name = o.native.tag_name(o.cache, t).unwrap_or_default();
                if !(name.contains("ff_light") || name.contains("hill_beacon") || name.ends_with("energy_sword") || name.contains("knife") || name.contains("blade")) { continue }
                match actors.add_object(&o.native, o.cache, t, glam::Vec3::ZERO, 0.0) {
                    Ok(id) => { let a = actors.get(id).unwrap(); println!("{g} {name}: drawn, model tag {:#x}, nodes {}, regions {}", a.def.model.mode_tag, a.def.model.skel.nodes.len(), a.def.model.region_names.len()); }
                    Err(e) => println!("{g} {name}: NOT drawn ({e})"),
                }
            }
        }
    }
}

#[cfg(test)]
mod knife_scan {
    use super::*;
    use crate::film_stats::NameSource;

    #[test]
    #[ignore = "diagnostic: HMS_FILM_LIST = a file of film paths"]
    fn find_knives_everywhere() {
        let list = std::env::var("HMS_FILM_LIST").expect("HMS_FILM_LIST");
        let paths: Vec<std::path::PathBuf> = std::fs::read_to_string(list).unwrap().lines().map(std::path::PathBuf::from).collect();
        let mut caches: std::collections::HashMap<String, Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle, crate::film_stats::MapNames)>> = Default::default();
        let (mut films, mut with_knife) = (0usize, 0usize);
        for p in paths {
            let Ok(f) = film::open(&p) else { continue };
            let stem = f.header.scenario_name().to_ascii_lowercase().replace('\\', "/").rsplit('/').next().unwrap_or("").to_string();
            let entry = caches.entry(stem.clone()).or_insert_with(|| {
                let (n, c) = crate::film_action::objective_inventory::open_reach(&stem)?;
                let map = crate::mapcat::enumerate().into_iter().find(|m| m.game == crate::mapcat::Game::Reach && m.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(&stem)))?;
                let names = crate::film_stats::MapNames::build_with_map(&n, c, Some(&map.path));
                Some((std::sync::Arc::new(n), c, names))
            });
            let Some((_, _, names)) = entry else { continue };
            let idx = fd::build_index_film(&f);
            films += 1;
            let mut found: std::collections::BTreeMap<String, (usize, usize, usize)> = Default::default();
            for t in &idx.tracks {
                let Some((_, _, path)) = t.def_wire.and_then(|w| names.definition(w)) else { continue };
                if path.contains("knife") || path.contains("sheath") || path.contains("spartan_k") {
                    let e = found.entry(path.rsplit('\\').next().unwrap_or(&path).to_string()).or_default();
                    e.0 += 1;
                    if t.rides() { e.1 += 1; }
                    if t.positions.len() > 1 { e.2 += 1; }
                }
            }
            let ass = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_ASSASSINATE).count();
            if !found.is_empty() { with_knife += 1; }
            println!("{} [{stem}] ticks {} assassinate_events {ass}: {:?}", p.file_name().unwrap().to_string_lossy(), idx.ticks, found);
        }
        println!("SCANNED {films} films, {with_knife} with knife/sheath tracks");
    }
}

#[cfg(test)]
mod knife_tag_probe {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn who_owns_the_spartan_knife() {
        let Some(o) = open_forge() else { return };
        let knife: Vec<u32> = o.native.tags_by_group(o.cache, "weap").into_iter().filter(|t| o.native.tag_name(o.cache, *t).unwrap_or_default().contains("spartan_knife")).collect();
        println!("knife tags: {:?}", knife.iter().map(|t| o.native.tag_name(o.cache, *t).unwrap_or_default()).collect::<Vec<_>>());
        for g in ["bipd", "weap", "vehi", "bloc", "scen", "eqip", "crea"] {
            for t in o.native.tags_by_group(o.cache, g) {
                for a in o.native.enumerate_attachments(o.cache, t) {
                    let co = a.child_obj;
                    if knife.contains(&co) {
                        let (m, vi, ms) = (a.marker_sid, a.variant_index, a.child_mode);
                        println!("{g} {} has a child/attachment row -> knife: marker sid {m:#x} variant {vi} mode {ms}", o.native.tag_name(o.cache, t).unwrap_or_default());
                    }
                }
            }
        }
        for t in o.native.tags_by_group(o.cache, "bipd") {
            let n = o.native.tag_name(o.cache, t).unwrap_or_default();
            if n.contains("spartan") || n.contains("elite") {
                let rows = o.native.enumerate_attachments(o.cache, t);
                for a in &rows { let (co, cm, vi, vs, ms, pos, rot) = (a.child_obj, a.child_mode, a.variant_index, a.variant_name_sid, a.marker_sid, a.pos, a.rot); println!("   row variant[{vi}] vsid {vs:#x} marker {ms:#x} child '{}' mode {cm:#x} pos {:?} rot {:?}", o.native.tag_name(o.cache, co).unwrap_or_default().rsplit('\\').next().unwrap_or(""), pos, rot); }
                println!("{n}: {} attachment rows: {:?}", rows.len(), rows.iter().map(|a| { let c = a.child_obj; o.native.tag_name(o.cache, c).unwrap_or_default() }).collect::<Vec<_>>());
            }
        }
    }
}

// =============================================================================================
// #knife  A Spartan's hlmt CHILD objects (knife + sheath) ride the PARENT MARKER's node.
// CPU-only: no GPU mesh is read -- the palette the children are skinned by is the production
// function `reach_actor::child_palette_matrix`, applied to the child's bind point.
// =============================================================================================
#[cfg(test)]
mod knife_gate {
    use super::*;
    use glam::{Mat4, Vec3};
    use crate::reach_actor::{child_palette_matrix, Pose};

    struct Rig {
        o: OpenMap,
    }

    fn marker_of(model: &crate::reach_actor::ReachModel, sid: u32) -> Option<crate::reach_skel::Marker> {
        model.markers.iter().find(|g| g.name == sid).and_then(|g| crate::reach_skel::pick_marker(&g.markers, -1, -1))
    }

    /// (name, node count) of the knife/sheath children the Spartan's DEFAULT variant selects.
    fn children_of(o: &OpenMap, tag: u32, vsid: u32) -> Vec<String> {
        let dsid = crate::scene::object_default_variant_sid(&o.native, o.cache, tag);
        let rows: Vec<_> = o.native.enumerate_attachments(o.cache, tag).into_iter().filter(|a| { let c = a.child_mode; c != 0 && c != u32::MAX }).collect();
        let sids: Vec<u32> = rows.iter().map(|a| a.variant_name_sid).collect();
        let chosen = crate::scene::pick_child_variant(&sids, vsid, dsid);
        rows.iter().filter(|a| crate::scene::child_row_selected(a.variant_name_sid, a.variant_index == 0, chosen))
            .map(|a| { let co = a.child_obj; o.native.tag_name(o.cache, co).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string() })
            .collect()
    }

    fn tag_of(o: &OpenMap, suffix: &str) -> Option<u32> {
        o.native.tags_by_group(o.cache, "bipd").into_iter().find(|t| o.native.tag_name(o.cache, *t).unwrap_or_default().to_lowercase().ends_with(suffix))
    }

    /// Per pose: (max |old placement - marker world|, max |new placement - marker world|,
    /// spread of the hip-to-knife distance, node travel from the bind pose), over `poses`.
    struct Report { new_err: f32, old_err: f32, spread: f32, travel: f32, poses: usize }

    fn measure(o: &OpenMap, child_palette_node: Option<i32>) -> Option<Report> {
        let _ = child_palette_node;
        let tag = tag_of(o, "spartans\\spartans")?;
        let mut actors = crate::reach_actor::ActorScene::default();
        let id = actors.add(&o.native, o.cache, "spartans\\spartans", Vec3::ZERO, 0.0).ok()?;
        let (dsid, rows_sid) = (crate::scene::object_default_variant_sid(&o.native, o.cache, tag), 0u32);
        let _ = rows_sid;
        let (marker_sid, n) = crate::scene::child_marker_for(&o.native, o.cache, tag, dsid, dsid)?;
        assert!(n >= 1);
        let a0 = actors.get(id)?;
        let model = a0.def.model.clone();
        let inv = a0.def.inverse_bind.clone();
        let marker = marker_of(&model, marker_sid)?;
        assert!(marker.node >= 0, "the knife marker names a skeleton node");
        let node = marker.node;
        let bind = Pose::bind(&model.skel);
        let p_bind = bind.marker_model(&marker).w_axis.truncate();
        // names: idle, sprint, crouch, a death and an assassination clip, whichever the graph has
        let g = a0.def.graph.clone()?;
        let want = ["combat:rifle:idle", "sprint:rifle:move_front", "crouch:rifle:idle", "crouch:rifle:move_front", "combat:rifle:move_front", "combat:rifle:melee_strike", "death", "assassin", "sync_action", "airborne"];
        let mut found: Vec<usize> = Vec::new();
        for w in want {
            if let Some(i) = g.find(w) { if !found.contains(&i) { found.push(i); } }
        }
        let mut rep = Report { new_err: 0.0, old_err: 0.0, spread: 0.0, travel: 0.0, poses: 0 };
        let mut dists: Vec<f32> = Vec::new();
        for ai in found {
            let frames = g.anims[ai].frame_count.max(1) as f32;
            for k in 0..4 {
                let f = frames * k as f32 / 4.0;
                actors.set_anim_index(&o.native, o.cache, id, ai, None).ok()?;
                let a = actors.actors.iter_mut().find(|a| a.id == id)?;
                a.frame = f;
                a.compute_pose();
                let pose: &Pose = &a.pose;
                let want_p = pose.marker_model(&marker).w_axis.truncate();
                let m = child_palette_matrix(Mat4::IDENTITY, pose, &inv, node);
                let new_p = m.transform_point3(p_bind);
                // the OLD placement: skinned by the body's own bones named by the child's node indices
                let mut old_err = 0.0f32;
                for kn in 0..3usize {
                    let om = pose.model.get(kn).copied().unwrap_or(Mat4::IDENTITY) * inv.get(kn).copied().unwrap_or(Mat4::IDENTITY);
                    old_err = old_err.max((om.transform_point3(p_bind) - want_p).length());
                }
                rep.new_err = rep.new_err.max((new_p - want_p).length());
                rep.old_err = rep.old_err.max(old_err);
                rep.travel = rep.travel.max((pose.model[node as usize].w_axis.truncate() - bind.model[node as usize].w_axis.truncate()).length());
                dists.push((new_p - pose.model[node as usize].w_axis.truncate()).length());
                rep.poses += 1;
            }
        }
        let (lo, hi) = dists.iter().fold((f32::MAX, f32::MIN), |(l, h), d| (l.min(*d), h.max(*d)));
        rep.spread = hi - lo;
        Some(rep)
    }

    /// Variant gating and liveness: the Spartan and the hologram Spartan have a knife and a sheath
    /// (the default variant selects both), the other model variants that author children select the
    /// knife alone, and Elites select none.
    #[test]
    fn the_children_exist_for_the_spartan_and_not_the_elite() {
        let Some(o) = open_forge() else { eprintln!("no forge_halo: gate skipped"); return };
        let sp = tag_of(&o, "spartans\\spartans").expect("the Spartan biped");
        let dsid = crate::scene::object_default_variant_sid(&o.native, o.cache, sp);
        let kids = children_of(&o, sp, dsid);
        println!("Spartan default-variant children: {kids:?}");
        assert!(kids.iter().any(|k| k == "spartan_knife") && kids.iter().any(|k| k == "spartan_knife_sheath"), "{kids:?}");
        let holo = tag_of(&o, "hologram_spartan").expect("hologram_spartan");
        assert!(!children_of(&o, holo, 0).is_empty());
        let elite = tag_of(&o, "elite\\elite").expect("the Elite biped");
        assert!(children_of(&o, elite, 0).is_empty(), "Elites author no knife");
        // a variant that authors only the knife
        let rows: Vec<_> = o.native.enumerate_attachments(o.cache, sp);
        let only_knife = rows.iter().map(|a| a.variant_name_sid).find(|&s| s != 0 && s != 1 && children_of(&o, sp, s).len() == 1);
        assert!(only_knife.is_some(), "some Spartan model variant selects the knife alone");
        // the children have render models
        for k in rows.iter().take(2) {
            let cm = k.child_mode;
            let m = crate::reach_actor::ReachModel::read(&o.native, o.cache, cm).expect("child render model");
            assert!(!m.skel.nodes.is_empty());
        }
    }

    /// The children follow the marker's node in every pose; the old placements do not.
    #[test]
    fn the_knife_follows_the_hip_marker_in_every_pose() {
        let Some(o) = open_forge() else { return };
        let r = measure(&o, None).expect("the Spartan rig");
        println!("knife: {} poses; node travel up to {:.3} wu; NEW error {:.5} wu, hip-distance spread {:.5} wu; OLD (body-bone) error {:.3} wu", r.poses, r.travel, r.new_err, r.spread, r.old_err);
        assert!(r.poses >= 12, "liveness: only {} poses (the graph lacks the clips)", r.poses);
        assert!(r.travel > 0.05, "liveness: the hip node never moved ({:.3} wu), the poses prove nothing", r.travel);
        assert!(r.new_err < 1e-3, "the child strays {:.5} wu from its marker", r.new_err);
        assert!(r.spread < 1e-3, "the knife-to-hip distance varies by {:.5} wu across poses", r.spread);
    }

    /// SABOTEUR: the previous behaviour (children skinned by the body bones their own node indices
    /// name) must fail the same bound -- on this rig by more than the bound by a wide margin.
    #[test]
    fn saboteur_the_old_placement_leaves_the_hip() {
        let Some(o) = open_forge() else { return };
        let r = measure(&o, None).expect("the Spartan rig");
        assert!(r.old_err > 0.02, "the gate does not bite: the old placement is within {:.4} wu", r.old_err);
        assert!(r.old_err > 20.0 * r.new_err.max(1e-4), "old {:.4} vs new {:.5}", r.old_err, r.new_err);
    }
}

#[cfg(test)]
mod race_enum {
    use super::*;

    /// Every actor the draw path would show at spread ticks, by source, with its distance to the
    /// nearest body / vehicle.
    #[test]
    #[ignore = "diagnostic: Reach(1).mov from ~/Downloads"]
    fn enumerate_the_race_clip() {
        let Some(o) = open_forge() else { return };
        let Some(h) = std::env::var_os("HOME") else { return };
        let Ok(f) = film::open(&std::path::Path::new(&h).join("Downloads").join("Reach(1).mov")) else { println!("no Reach(1).mov"); return };
        let mut pb = Playback::new(fd::build_index_film(&f));
        pb.moving_only = false;
        let mut d = ActorDriver::default();
        d.cap = 64;
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let mut actors = crate::reach_actor::ActorScene::default();
        let total = pb.total_ticks();
        #[derive(Default, Debug)]
        struct Agg { samples: usize, far: usize, first_far: Option<u32>, max_far: f32 }
        let mut agg: std::collections::BTreeMap<String, Agg> = Default::default();
        let (mut ticks, mut seated_total, mut bodies_total, mut with_child) = (0usize, 0usize, 0usize, 0usize);
        let mut k = 0u32;
        while k < total {
            pb.seek_tick(k as f32);
            d.sync(&pb, &mut actors, &o.native, o.cache);
            actors.update_poses_headless(0.0);
            ticks += 1;
            let park = glam::Vec3::from(d.park);
            let pos_of = |id: u32| actors.get(id).map(|a| a.world().w_axis.truncate());
            let bodies: Vec<glam::Vec3> = d.body_actor.values().filter_map(|i| pos_of(*i)).filter(|p| (*p - park).length() > 1.0).collect();
            let vehicles: Vec<glam::Vec3> = d.vehicle_actor.values().filter_map(|i| pos_of(*i)).filter(|p| (*p - park).length() > 1.0).collect();
            bodies_total += bodies.len();
            for bid in d.body_actor.values() {
                if let Some(a) = actors.get(*bid) {
                    if a.world_override.is_some() { seated_total += 1; }
                    let tag = a.def.obj_tag;
                    if crate::scene::child_marker_for(&o.native, o.cache, tag, a.variant_sid, crate::scene::object_default_variant_sid(&o.native, o.cache, tag)).is_some() { with_child += 1; }
                }
            }
            let mut note = |kind: &str, id: u32, name: String| {
                let Some(p) = pos_of(id) else { return };
                if (p - park).length() < 1.0 { return }
                let nb = bodies.iter().map(|b| (*b - p).length()).fold(f32::MAX, f32::min);
                let nv = vehicles.iter().map(|b| (*b - p).length()).fold(f32::MAX, f32::min);
                let dist = nb.min(nv);
                let e = agg.entry(format!("{kind}: {name}")).or_default();
                e.samples += 1;
                if dist > 1.5 { e.far += 1; e.first_far.get_or_insert(k); e.max_far = e.max_far.max(if dist == f32::MAX { 0.0 } else { dist }); }
            };
            for (net, id) in d.object_actor.iter() {
                let nm = d.objects.as_ref().and_then(|ot| ot.ident(*net)).map(|i| format!("{} {}", i.group, i.name)).unwrap_or_default();
                note("object", *id, nm);
            }
            for (net, id) in d.vehicle_actor.iter() { let _ = net; note("vehicle", *id, actors.get(*id).map(|a| a.def.name.clone()).unwrap_or_default()); }
            k += 300;
        }
        println!("race clip: {ticks} sampled ticks; mean bodies drawn {:.1}, seated {:.1}, bodies whose model has child rows {:.1}", bodies_total as f32 / ticks as f32, seated_total as f32 / ticks as f32, with_child as f32 / ticks as f32);
        for (k, v) in agg { println!("  {k:<60} samples {:<5} farther than 1.5 wu from any body/vehicle: {:<5} first at tick {:?}, max {:.1}", v.samples, v.far, v.first_far, v.max_far); }
    }
}

#[cfg(test)]
mod size_probe {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn model_extents() {
        let Some(o) = open_forge() else { return };
        for (g, sub) in [("bloc", "mp_hill_beacon"), ("weap", "spartan_knife\\spartan_knife"), ("weap", "spartan_knife_sheath"), ("vehi", "mongoose\\mongoose"), ("bipd", "spartans\\spartans")] {
            for t in o.native.tags_by_group(o.cache, g) {
                let n = o.native.tag_name(o.cache, t).unwrap_or_default();
                if !n.ends_with(sub) { continue }
                let mode = o.native.resolve_mode(o.cache, t);
                let Ok(h) = o.native.open_model(o.cache, mode) else { println!("{n}: no model"); continue };
                let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
                let cnt = o.native.model_section_count(h);
                for i in 0..cnt { if let Some(s) = o.native.model_section(h, i) { if cnt <= 2 { let (nid, vf, fc, vc) = (s.node_index, s.vertex_format, s.format_class, s.vertex_count); println!("   section {i}: node_index {nid} vertex_format {vf} format_class {fc} verts {vc}"); } for k in 0..3 { lo[k] = lo[k].min(s.bounds_min[k]); hi[k] = hi[k].max(s.bounds_max[k]); } } }
                println!("{n}: sections {cnt}, extent {:.3} x {:.3} x {:.3} wu", hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]);
            }
        }
    }
}

#[cfg(test)]
mod marker_probe {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn knife_marker_instances() {
        let Some(o) = open_forge() else { return };
        for suffix in ["spartans\\spartans", "hologram_spartan"] {
            let Some(tag) = o.native.tags_by_group(o.cache, "bipd").into_iter().find(|t| o.native.tag_name(o.cache, *t).unwrap_or_default().to_lowercase().ends_with(suffix)) else { continue };
            let mut actors = crate::reach_actor::ActorScene::default();
            let Ok(id) = actors.add(&o.native, o.cache, suffix, glam::Vec3::ZERO, 0.0) else { continue };
            let a = actors.get(id).unwrap();
            let bind = crate::reach_actor::Pose::bind(&a.def.model.skel);
            let rows = o.native.enumerate_attachments(o.cache, tag);
            let r0 = rows[0];
            let (rp, rr) = (r0.pos, r0.rot);
            println!("{suffix}: row0 composed pos {rp:?} rot {rr:?}");
            for g in a.def.model.markers.iter().filter(|g| g.name == r0.marker_sid) {
                for (i, m) in g.markers.iter().enumerate() {
                    let p = bind.marker_model(m).w_axis.truncate();
                    println!("   marker instance {i}: region {} perm {} node {} ({}) bind pos {:?}", m.region, m.permutation, m.node, a.def.model.node_names.get(m.node.max(0) as usize).cloned().unwrap_or_default(), p);
                }
            }
        }
    }
}

#[cfg(test)]
mod race_beacons {
    use super::*;
    #[test]
    #[ignore = "diagnostic: Reach(1).mov from ~/Downloads"]
    fn why_are_beacons_far_from_bodies() {
        let Some(o) = open_forge() else { return };
        let Some(h) = std::env::var_os("HOME") else { return };
        let Ok(f) = film::open(&std::path::Path::new(&h).join("Downloads").join("Reach(1).mov")) else { return };
        let mut pb = Playback::new(fd::build_index_film(&f));
        pb.moving_only = false;
        let mut d = ActorDriver::default();
        d.cap = 64;
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let mut actors = crate::reach_actor::ActorScene::default();
        let idx2 = fd::build_index_film(&f);
        let by: std::collections::HashMap<u32, &fd::EntityTrack> = idx2.tracks.iter().map(|t| (t.net_id, t)).collect();
        let lookup = |n: u32| by.get(&n).copied();
        let total = pb.total_ticks();
        let mut cats: std::collections::BTreeMap<String, usize> = Default::default();
        let mut shown = 0;
        let mut k = 0u32;
        while k < total {
            pb.seek_tick(k as f32);
            d.sync(&pb, &mut actors, &o.native, o.cache);
            let park = glam::Vec3::from(d.park);
            let bodies: Vec<glam::Vec3> = d.body_actor.values().filter_map(|i| actors.get(*i)).map(|a| a.world().w_axis.truncate()).filter(|p| (*p - park).length() > 1.0).collect();
            for (net, id) in d.object_actor.iter() {
                let Some(a) = actors.get(*id) else { continue };
                let p = a.world().w_axis.truncate();
                if (p - park).length() < 1.0 { continue }
                let nb = bodies.iter().map(|b| (*b - p).length()).fold(f32::MAX, f32::min);
                if nb <= 1.5 { *cats.entry("near a drawn body".into()).or_default() += 1; continue }
                let t = by[net];
                let att = t.attached_at(k);
                let cat = match att {
                    None => {
                        let q = t.sample(k + 60).unwrap_or([p.x, p.y, p.z]);
                        let v = ((q[0] - p.x).powi(2) + (q[1] - p.y).powi(2) + (q[2] - p.z).powi(2)).sqrt();
                        let was_attached = t.attachments.iter().filter(|(kk, _)| *kk <= k).last().map_or(false, |(_, a)| a.is_none());
                        let nxt = t.positions.iter().find(|(pt, _)| *pt > k).map(|(pt, _)| pt - k);
                        let prv = t.positions.iter().rev().find(|(pt, _)| *pt <= k).map(|(pt, _)| k - pt);
                        format!("NOT attached; released-earlier {was_attached}; moves {}; nearest own keys: prev {} / next {} ticks away", if v > 2.0 { ">2wu per second" } else { "<2wu per second (parked)" }, prv.map_or("none".into(), |x| if x > 600 { ">600".to_string() } else { "<=600".to_string() }), nxt.map_or("none".into(), |x| if x > 600 { ">600".to_string() } else { "<=600".to_string() }))
                    }
                    Some(at) => {
                        let par = lookup(at.parent);
                        let alive = par.map_or(false, |p| p.alive_at(k));
                        let drawn = d.body_actor.contains_key(&at.parent);
                        let target = par.and_then(|p| crate::film_objective::follow_position(&lookup, p, k as f32, 1));
                        let dd = target.map(|tg| (tg - p).length()).unwrap_or(-1.0);
                        format!("attached; parent type {} alive {alive} body-drawn {drawn}; drawn-vs-follow-target {}", par.map_or(255, |p| p.entity_type), if dd < 0.0 { "n/a" } else if dd < 1.0 { "<1wu" } else { ">=1wu" })
                    }
                };
                *cats.entry(cat.clone()).or_default() += 1;
                if shown < 6 && att.is_some() { shown += 1; println!("   far beacon sample: tick {k} net {net:#x} dist-to-nearest-body {nb:.1}: {cat}"); }
            }
            k += 300;
        }
        for (c, n) in cats { println!("  {n:>6}  {c}"); }
    }
}

#[cfg(test)]
mod vehicle_children {
    use super::*;
    #[test]
    #[ignore = "diagnostic"]
    fn mongoose_child_rows() {
        let Some(o) = open_forge() else { return };
        for t in o.native.tags_by_group(o.cache, "vehi") {
            let n = o.native.tag_name(o.cache, t).unwrap_or_default();
            if !n.ends_with("mongoose\\mongoose") { continue }
            let rows = o.native.enumerate_attachments(o.cache, t);
            println!("{n}: {} child rows", rows.len());
            for a in rows.iter().take(8) { let (co, cm, vi, ms) = (a.child_obj, a.child_mode, a.variant_index, a.marker_sid); println!("   variant[{vi}] marker {ms:#x} child '{}' mode {cm:#x}", o.native.tag_name(o.cache, co).unwrap_or_default()); }
        }
        // every actor-drawn kind in the race: which object tags (any class) carry child rows?
        for g in ["bipd", "vehi", "weap", "bloc", "scen", "eqip"] {
            let mut with = 0; let mut all = 0;
            for t in o.native.tags_by_group(o.cache, g) { all += 1; if !o.native.enumerate_attachments(o.cache, t).is_empty() { with += 1; } }
            println!("{g}: {with} of {all} tags have hlmt child rows");
        }
    }
}

// =============================================================================================
// #knife  On real clips (the race clip, Castle Wars): every drawn Spartan's knife/sheath child sits
// on its hip marker in the pose the body is DRAWN in -- seated riders, corpses, sprinters alike.
// =============================================================================================
#[cfg(test)]
mod knife_clip_gate {
    use super::*;
    use glam::{Mat4, Vec3};

    /// (pairs checked, worst error of the production placement, worst error of the OLD placement,
    /// worst distance from the body's origin, seated pairs)
    fn run(o: &OpenMap, f: &film::Film, step: u32) -> (usize, f32, f32, f32, usize) {
        let mut pb = Playback::new(fd::build_index_film(f));
        pb.moving_only = false;
        let mut d = ActorDriver::default();
        d.cap = 64;
        d.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
        let mut actors = crate::reach_actor::ActorScene::default();
        let total = pb.total_ticks();
        let (mut n, mut seated, mut new_w, mut old_w, mut far) = (0usize, 0usize, 0.0f32, 0.0f32, 0.0f32);
        let mut k = 0u32;
        while k < total {
            pb.seek_tick(k as f32);
            d.sync(&pb, &mut actors, &o.native, o.cache);
            actors.update_poses_headless(0.0);
            let park = Vec3::from(d.park);
            for id in d.body_actor.values() {
                let Some(a) = actors.get(*id) else { continue };
                let world = a.world();
                if (world.w_axis.truncate() - park).length() < 1.0 { continue }
                let tag = a.def.obj_tag;
                let dsid = crate::scene::object_default_variant_sid(&o.native, o.cache, tag);
                let Some((msid, _)) = crate::scene::child_marker_for(&o.native, o.cache, tag, a.variant_sid, dsid) else { continue };
                let Some(marker) = a.def.model.markers.iter().find(|g| g.name == msid).and_then(|g| crate::reach_skel::pick_marker(&g.markers, -1, -1)) else { continue };
                let bind = crate::reach_actor::Pose::bind(&a.def.model.skel);
                let p_bind = bind.marker_model(&marker).w_axis.truncate();
                let want = world.transform_point3(a.pose.marker_model(&marker).w_axis.truncate());
                // PRODUCTION: the child palette (delta = the actor's world, the actor was never "built" here)
                let m = crate::reach_actor::child_palette_matrix(world, &a.pose, &a.def.inverse_bind, marker.node);
                let got = m.transform_point3(p_bind);
                // OLD: skinned by the body bones named by the knife's own node indices (0..2)
                let mut old = 0.0f32;
                for kn in 0..3usize {
                    let om = world * a.pose.model.get(kn).copied().unwrap_or(Mat4::IDENTITY) * a.def.inverse_bind.get(kn).copied().unwrap_or(Mat4::IDENTITY);
                    old = old.max((om.transform_point3(p_bind) - want).length());
                }
                n += 1;
                if a.world_override.is_some() { seated += 1; }
                new_w = new_w.max((got - want).length());
                old_w = old_w.max(old);
                far = far.max((got - world.w_axis.truncate()).length());
            }
            k += step;
        }
        (n, new_w, old_w, far, seated)
    }

    fn clips() -> Vec<(String, film::Film)> {
        let mut v: Vec<(String, film::Film)> = Vec::new();
        if let Some(h) = std::env::var_os("HOME") {
            if let Ok(f) = film::open(&std::path::Path::new(&h).join("Downloads").join("Reach(1).mov")) { v.push(("Reach(1)".into(), f)); }
        }
        for s in ["53C14E9D", "2BAB2586"] {
            if let Some(p) = film::spec_corpus().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().contains(s))) {
                if let Ok(f) = film::open(&p) { v.push((s.to_string(), f)); }
            }
        }
        v
    }

    #[test]
    fn every_drawn_knife_is_on_its_hip_marker_on_real_clips() {
        let Some(o) = open_forge() else { eprintln!("no forge_halo: gate skipped"); return };
        let (mut n, mut seated, mut worst, mut far) = (0usize, 0usize, 0.0f32, 0.0f32);
        for (name, f) in clips() {
            let (pairs, new_w, old_w, far_w, s) = run(&o, &f, 600);
            println!("{name}: {pairs} body-ticks with a knife ({s} seated); production error {new_w:.5} wu; old placement up to {old_w:.3} wu; farthest from body origin {far_w:.3} wu");
            n += pairs; seated += s; worst = worst.max(new_w); far = far.max(far_w);
        }
        assert!(n >= 300, "liveness: only {n} drawn knives across the clips");
        assert!(worst < 0.01, "a knife is {worst:.4} wu off its hip marker");
        assert!(far < 1.2, "a knife is {far:.2} wu from its body's origin: not on the Spartan");
        let _ = seated;
    }

    /// SABOTEUR: the previous placement fails the same bound by a wide margin on real poses.
    #[test]
    fn saboteur_the_old_placement_floats_on_real_clips() {
        let Some(o) = open_forge() else { return };
        let mut worst_old = 0.0f32;
        let mut n = 0;
        for (_, f) in clips() {
            let (pairs, _, old_w, _, _) = run(&o, &f, 600);
            n += pairs;
            worst_old = worst_old.max(old_w);
        }
        assert!(n >= 300, "liveness");
        assert!(worst_old > 0.02, "the gate does not bite: the old placement is within {worst_old:.4} wu");
    }
}
