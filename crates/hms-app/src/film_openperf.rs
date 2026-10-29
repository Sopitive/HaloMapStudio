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
}

pub(crate) fn open_forge() -> Option<OpenMap> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    let mut d = std::env::current_exe().ok()?.parent().map(|p| p.to_path_buf());
    for _ in 0..3 {
        if let Some(p) = &d {
            roots.push(p.clone());
            d = p.parent().map(|q| q.to_path_buf());
        }
    }
    let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
    let native = std::sync::Arc::new(hms_native::NativeDll::load(&dll).ok()?);
    let m = crate::mapcat::enumerate().into_iter().find(|c| {
        c.game == crate::mapcat::Game::Reach
            && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
    })?;
    if let Some(d) = m.path.parent() {
        native.set_shared_cache_dirs(&[d.to_path_buf()]);
    }
    let cache = native.open_cache(&m.path.to_string_lossy()).ok()?;
    Some(OpenMap { native, cache, map: m.path })
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
