//! Measurement probes for #film-anim. These are `#[test]`s because that is how every other
//! finding in `re-notes/reach_saved_film_layout.md` was measured: run one with `--nocapture`, read
//! the table, and the number either supports a claim or it does not.
//!
//! None of them assert much. Their job is to PRINT, and a probe that prints a clean negative is a
//! result.

use crate::film;
use crate::film_decode::{self, DecodeParams, FilmIndex};
use std::path::PathBuf;

/// Every Reach saved film on this machine, biggest first (the big ones carry entity streams).
pub fn films() -> Vec<PathBuf> {
    // ONE implementation, in `film.rs`; see `film::film_search_dirs`.
    film::find_finished_films()
}

/// A film's basename, which is the ONLY part of the path safe to print: the directory above it
/// contains the machine owner's real first name.
pub fn name(p: &std::path::Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Open a film and build its playback index, or `None` if it has no entity stream.
pub fn indexed(p: &std::path::Path) -> Option<(film::Film, FilmIndex)> {
    let f = film::open(p).ok()?;
    if f.data.records.is_empty() || f.header.map_variant.is_empty() {
        return None;
    }
    let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
    if idx.entity_messages == 0 {
        return None;
    }
    Some((f, idx))
}

// ---------------------------------------------------------------------------------------------
// probe 1 — the entity-type census, with MOTION, so the unidentified types stop being a mystery
// ---------------------------------------------------------------------------------------------

/// What each of the 19 entity types actually does on screen. The brief's histogram
/// (`13:7790 12:1610 17:1126 11:559 ...`) counts MESSAGES; what matters for drawing is how many
/// distinct entities there are, how many move, how far, how fast, and how high off the ground.
#[test]
fn probe_type_motion_census() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    #[derive(Default, Clone)]
    struct Agg {
        entities: usize,
        moving: usize,
        positions: usize,
        path_total: f64,
        max_speed: f32,
        speeds: Vec<f32>,
        z_span: f64,
    }
    let mut per_type: std::collections::BTreeMap<u8, Agg> = Default::default();
    let mut films_used = 0usize;
    for p in films.iter().take(14) {
        let Some((_f, idx)) = indexed(p) else { continue };
        films_used += 1;
        for t in &idx.tracks {
            let a = per_type.entry(t.entity_type).or_default();
            a.entities += 1;
            a.positions += t.positions.len();
            let pl = t.path_length();
            a.path_total += pl as f64;
            if t.moves() {
                a.moving += 1;
            }
            // top speed over the track, ignoring respawn teleports
            let mut zmin = f32::MAX;
            let mut zmax = f32::MIN;
            for w in t.positions.windows(2) {
                let (t0, p0) = w[0];
                let (t1, p1) = w[1];
                if t1 <= t0 {
                    continue;
                }
                let dt = (t1 - t0) as f32 / 60.0;
                let d = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
                let s = d / dt;
                if s < 30.0 {
                    a.max_speed = a.max_speed.max(s);
                    if s > 0.2 {
                        a.speeds.push(s);
                    }
                }
            }
            for (_, q) in &t.positions {
                zmin = zmin.min(q[2]);
                zmax = zmax.max(q[2]);
            }
            if zmax > zmin {
                a.z_span += (zmax - zmin) as f64;
            }
        }
        println!("  {}  ({} tracks)", name(p), idx.tracks.len());
    }
    println!("\n== entity-type motion census over {films_used} films ==");
    println!("type  entities  moving   positions   mean path   p50 spd  p95 spd  max spd  mean z-span");
    for (ty, a) in &per_type {
        let mut sp = a.speeds.clone();
        sp.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |f: f32| if sp.is_empty() { 0.0 } else { sp[((sp.len() - 1) as f32 * f) as usize] };
        println!(
            "{ty:>4}  {:>8}  {:>6}  {:>10}   {:>9.2}   {:>7.2}  {:>7.2}  {:>7.2}  {:>10.2}",
            a.entities,
            a.moving,
            a.positions,
            a.path_total / a.entities.max(1) as f64,
            q(0.50),
            q(0.95),
            a.max_speed,
            a.z_span / a.entities.max(1) as f64,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// probe 2 — the VEHICLE oracle: a biped that rides something
// ---------------------------------------------------------------------------------------------

/// Which entity type is a vehicle, found with no engine knowledge at all.
///
/// The oracle in the brief: **a biped's position becomes coincident with another entity's for a
/// sustained run of ticks, and then they move together.** That is what getting into a vehicle
/// looks like in a replication stream, and nothing else in the game does it — two bipeds standing
/// next to each other never agree to within a fraction of a world unit while both travel tens of
/// world units.
#[test]
fn probe_vehicle_ride_oracle() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    /// How close two entities must be to count as "the same place", in world units.
    const NEAR_WU: f32 = 0.9;
    /// How long the coincidence must last, in ticks (1 s).
    const MIN_RUN: u32 = 60;
    /// How far the pair must travel together over the run, in world units.
    const MIN_TRAVEL: f32 = 4.0;
    /// Sample grid: 10 Hz is plenty to catch a one-second ride.
    const STEP: u32 = 6;

    let mut per_type: std::collections::BTreeMap<u8, (usize, f32, u32)> = Default::default();
    for p in films.iter().take(14) {
        let Some((_f, idx)) = indexed(p) else { continue };
        let bipeds: Vec<usize> = (0..idx.tracks.len())
            .filter(|&i| idx.tracks[i].entity_type == 11 && idx.tracks[i].positions.len() > 10)
            .collect();
        let others: Vec<usize> = (0..idx.tracks.len())
            .filter(|&i| idx.tracks[i].entity_type != 11 && idx.tracks[i].positions.len() > 10 && idx.tracks[i].moves())
            .collect();
        if bipeds.is_empty() || others.is_empty() {
            continue;
        }
        let mut hits: Vec<(u8, u32, u32, f32, u32, u32)> = Vec::new(); // type, run ticks, travel, ...
        for &bi in &bipeds {
            let b = &idx.tracks[bi];
            for &oi in &others {
                let o = &idx.tracks[oi];
                let lo = b.first_tick.max(o.first_tick);
                let hi = b.last_tick.min(o.last_tick);
                if hi.saturating_sub(lo) < MIN_RUN {
                    continue;
                }
                let mut run_start: Option<u32> = None;
                let mut last_near = lo;
                let mut t = lo;
                while t <= hi {
                    let near = match (b.sample(t), o.sample(t)) {
                        (Some(a), Some(c)) => {
                            let d = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2)).sqrt();
                            d < NEAR_WU
                        }
                        _ => false,
                    };
                    if near {
                        run_start.get_or_insert(t);
                        last_near = t;
                    } else if let Some(s) = run_start.take() {
                        let len = last_near - s;
                        if len >= MIN_RUN {
                            let travel = segment_travel(o, s, last_near);
                            if travel >= MIN_TRAVEL {
                                hits.push((o.entity_type, len, travel as u32, travel, s, last_near));
                            }
                        }
                    }
                    t += STEP;
                }
                if let Some(s) = run_start {
                    let len = last_near - s;
                    if len >= MIN_RUN {
                        let travel = segment_travel(o, s, last_near);
                        if travel >= MIN_TRAVEL {
                            hits.push((o.entity_type, len, travel as u32, travel, s, last_near));
                        }
                    }
                }
            }
        }
        if hits.is_empty() {
            println!("  {}: no ride", name(p));
            continue;
        }
        let mut by_ty: std::collections::BTreeMap<u8, (usize, f32)> = Default::default();
        for h in &hits {
            let e = by_ty.entry(h.0).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += h.3;
        }
        let summary: Vec<String> = by_ty.iter().map(|(t, (n, d))| format!("type {t}: {n} rides, {d:.0} wu")).collect();
        println!("  {}: {}", name(p), summary.join("; "));
        // the longest three, for eyeballing
        let mut top = hits.clone();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        for h in top.iter().take(3) {
            println!(
                "      type {} ride {:.1}s travelling {:.0} wu, ticks {}..{}",
                h.0,
                h.1 as f32 / 60.0,
                h.3,
                h.4,
                h.5
            );
        }
        for (t, (n, d)) in by_ty {
            let e = per_type.entry(t).or_insert((0, 0.0, 0));
            e.0 += n;
            e.1 += d;
            e.2 += 1;
        }
    }
    println!("\n== ride oracle: which type do bipeds RIDE ==");
    println!("type   rides   total wu   films");
    for (t, (n, d, f)) in &per_type {
        println!("{t:>4}  {n:>6}  {d:>9.0}  {f:>6}");
    }
}

fn segment_travel(t: &film_decode::EntityTrack, from: u32, to: u32) -> f32 {
    let mut d = 0.0f32;
    for w in t.positions.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t0 < from || t1 > to || t1 <= t0 {
            continue;
        }
        let step = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
        if step / ((t1 - t0) as f32 / 60.0) < 30.0 {
            d += step;
        }
    }
    d
}

// ---------------------------------------------------------------------------------------------
// probe 3 — the SPRINT oracle: is there a speed tier above the run cap?
// ---------------------------------------------------------------------------------------------

/// Reach's movement armour abilities show up in the SPEED distribution before any bit is decoded.
/// A base-trait run is one speed; sprint is faster; armour lock is exactly zero for a second or
/// more; a jetpack climbs. This probe prints the distribution so the later state decode has a
/// target to hit, and tells us whether these films even contain the abilities.
#[test]
fn probe_speed_distribution_and_ability_oracles() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    let mut all: Vec<f32> = Vec::new();
    println!("film                                 samples   p50    p75    p90    p99   max   lock-s  climb-s");
    for p in films.iter().take(14) {
        let Some((_f, idx)) = indexed(p) else { continue };
        let mut sp: Vec<f32> = Vec::new();
        let mut lock_ticks = 0u32; // dead stop while alive
        let mut climb_ticks = 0u32; // sustained positive Z
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            for w in t.positions.windows(2) {
                let (t0, a) = w[0];
                let (t1, b) = w[1];
                if t1 <= t0 || t1 - t0 > 12 {
                    continue;
                }
                let dt = (t1 - t0) as f32 / 60.0;
                let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
                let s = d / dt;
                if s < 30.0 {
                    sp.push(s);
                    if s < 0.05 {
                        lock_ticks += t1 - t0;
                    }
                    let dz = (b[2] - a[2]) / dt;
                    if dz > 1.5 {
                        climb_ticks += t1 - t0;
                    }
                }
            }
        }
        if sp.is_empty() {
            continue;
        }
        sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f32| sp[((sp.len() - 1) as f32 * f) as usize];
        println!(
            "{:<36} {:>7}  {:>5.2}  {:>5.2}  {:>5.2}  {:>5.2} {:>5.2}  {:>6.1}  {:>7.1}",
            name(p).chars().take(36).collect::<String>(),
            sp.len(),
            q(0.50),
            q(0.75),
            q(0.90),
            q(0.99),
            sp[sp.len() - 1],
            lock_ticks as f32 / 60.0,
            climb_ticks as f32 / 60.0,
        );
        all.extend_from_slice(&sp);
    }
    if all.is_empty() {
        return;
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n== pooled biped speed distribution, {} samples ==", all.len());
    // A histogram is what shows a second mode (= sprint) if there is one.
    let max = all[all.len() - 1].min(12.0);
    let bins = 24usize;
    let mut h = vec![0usize; bins + 1];
    for &s in &all {
        h[((s / max * bins as f32) as usize).min(bins)] += 1;
    }
    for (i, c) in h.iter().enumerate() {
        let lo = i as f32 * max / bins as f32;
        let bar = (*c as f64 / all.len() as f64 * 400.0) as usize;
        println!("  {lo:>5.2}+ {:>8}  {}", c, "#".repeat(bar.min(100)));
    }
}

// ---------------------------------------------------------------------------------------------
// probe 4 — the biped DERIVED block: how many bits are there, and what is in them?
// ---------------------------------------------------------------------------------------------

/// The facing was found 2 bits past the base block. What else is in that block? This dumps, for
/// real type-11 messages, how many payload bits remain after the base walk and the 13-bit yaw —
/// i.e. how much room there is for a locomotion state — and the per-bit-offset VALUE HISTOGRAM,
/// which is how a small enum field announces itself.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_biped_derived_block_shape() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    for p in films.iter().take(6) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_frame, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams {
            cv3_axis_bits: w,
            cv3_total_bits: w.iter().sum(),
            angle_always: true,
            update_prefix_bits: 1,
            ..base
        };
        // tail length histogram, and per-bit "fraction set" for the first 64 bits after the base
        let mut tail_hist: std::collections::BTreeMap<i64, usize> = Default::default();
        let mut set_count = vec![0usize; 96];
        let mut seen = 0usize;
        let mut latch_hist: std::collections::BTreeMap<String, usize> = Default::default();
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || !e.is_update || e.dirty_mask == 0 {
                    continue;
                }
                let Some(st) = film_decode::decode_entity_payload(e, &params) else { continue };
                latch_hist
                    .entry(format!(
                        "latch={:?} stop={:?} pos={} mask={:#06x}",
                        st.latched_at,
                        st.stopped_at_unknown_field,
                        st.position_raw.is_some() as u8,
                        e.dirty_mask & 0xFFFF
                    ))
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if st.latched_at != Some(2) || st.position_raw.is_none() {
                    continue;
                }
                let total = e.payload.len() * 8;
                let after = st.bits_after_base;
                tail_hist
                    .entry((total as i64 - after as i64) / 8 * 8)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                seen += 1;
                let mut r = film_decode::BitReader::new(&e.payload);
                if r.skip(after).is_err() {
                    continue;
                }
                for i in 0..96 {
                    match r.bits(1) {
                        Ok(1) => set_count[i] += 1,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
        }
        if seen == 0 {
            continue;
        }
        println!("\n== {} : {seen} latched-at-2 biped updates ==", name(p));
        println!("  message-class histogram (top 10):");
        let mut lh: Vec<_> = latch_hist.into_iter().collect();
        lh.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, v) in lh.iter().take(10) {
            println!("    {v:>8}  {k}");
        }
        println!("  tail bits after the base block (bucketed by 8):");
        let mut th: Vec<_> = tail_hist.into_iter().collect();
        th.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, v) in th.iter().take(10) {
            println!("    {k:>5} bits : {v}");
        }
        println!("  fraction of messages with each derived-block bit SET:");
        for row in 0..6 {
            let s: Vec<String> = (0..16)
                .map(|c| {
                    let i = row * 16 + c;
                    format!("{:>3}", (set_count[i] as f64 / seen as f64 * 100.0).round() as u32)
                })
                .collect();
            println!("    +{:>3}: {}", row * 16, s.join(" "));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// probe 5 — WHAT each entity is: resolve the creation payload's tag reference
// ---------------------------------------------------------------------------------------------

/// Open a Reach cache for the tag-resolving probes, or `None` (and the probe skips).
///
/// ONE library + cache for the whole test binary, behind a lock. The native page cache is not
/// built for several independently `dlopen`ed copies of itself, and probes running in parallel
/// each opening their own crashes the test binary with SIGSEGV -- the same reason
/// `reach_actor::tests` shares one. The guard is returned so the caller holds the lock for as
/// long as it uses the cache.
static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static PROBE_CACHE: std::sync::OnceLock<Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle)>> =
    std::sync::OnceLock::new();

pub fn open_reach(
    stem: &str,
) -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle)> {
    let guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (n, c) = PROBE_CACHE.get_or_init(|| open_reach_inner(stem)).clone()?;
    Some((guard, n, c))
}

fn open_reach_inner(stem: &str) -> Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle)> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..3 {
            if let Some(p) = &d {
                roots.push(p.clone());
                d = p.parent().map(|q| q.to_path_buf());
            }
        }
    }
    let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
    let native = std::sync::Arc::new(hms_native::NativeDll::load(&dll).ok()?);
    let cands = crate::mapcat::enumerate();
    let map = cands.iter().find(|c| {
        c.game == crate::mapcat::Game::Reach
            && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(stem))
    })?;
    if let Some(dir) = map.path.parent() {
        native.set_shared_cache_dirs(&[dir.to_path_buf()]);
    }
    let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
    Some((native, cache))
}

/// **The identity question, answered.** The shared creation reader starts with a 14-bit
/// `TagReference(group = -1)` — the object's DEFINITION tag. Resolve it against the loaded map's
/// tag table and every stream entity names itself.
///
/// This is also the proof that the creation reader is right: a wrong 14 bits would scatter across
/// the whole tag table, whereas the truth is a handful of plausible tags per entity type with the
/// RIGHT GROUP for that type.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_entity_tag_identity() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available -- tag identity not measured");
        return;
    };
    let mut per_type: std::collections::BTreeMap<u8, std::collections::BTreeMap<String, usize>> = Default::default();
    let mut films_used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        films_used += 1;
        if films_used > 6 {
            break;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams {
            cv3_axis_bits: w,
            cv3_total_bits: w.iter().sum(),
            angle_always: true,
            update_prefix_bits: 1,
            ..base
        };
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.is_update {
                    continue;
                }
                // the shared creation reader's FIRST field
                let mut r = film_decode::BitReader::new(&e.payload);
                let Ok(ti) = r.bits(params.tagref_bits) else { continue };
                let ti = ti as u32;
                let label = if ti == (1u32 << params.tagref_bits) - 1 {
                    "<null>".to_string()
                } else {
                    match (native.tag_class(cache, ti), native.tag_name(cache, ti)) {
                        (Some(g), Some(n)) => format!("{g} {n}"),
                        (None, Some(n)) => format!("???? {n}"),
                        _ => format!("<unresolved {ti}>"),
                    }
                };
                *per_type.entry(e.entity_type).or_default().entry(label).or_insert(0) += 1;
            }
        }
    }
    println!("\n== creation tag reference per entity type, over {films_used} forge_halo films ==");
    for (ty, m) in &per_type {
        let total: usize = m.values().sum();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("\ntype {ty}  ({} creations, {} distinct tags)", total, m.len());
        for (k, c) in v.iter().take(12) {
            println!("   {c:>7}  {k}");
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 6 — are list-B kind-2 entries really CREATIONS?
// ---------------------------------------------------------------------------------------------

/// The spec says list B kind 2 is a creation (0.07 per tick) and kind 3 an update (9-16 per
/// tick). But resolving a kind-2 payload's leading 14 bits as a tag index gives shader and bitmap
/// tags, nearly constant per entity type — which is what a MISREAD field looks like. Before
/// hunting a tag index, establish what kind 2 actually is.
#[test]
fn probe_command_kind_census() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    for p in films.iter().take(4) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams {
            cv3_axis_bits: w,
            cv3_total_bits: w.iter().sum(),
            angle_always: true,
            update_prefix_bits: 1,
            ..base
        };
        // per (list, kind): count, payload-length histogram, and the 5-bit leading value
        let mut per: std::collections::BTreeMap<(char, u8), (usize, Vec<usize>, std::collections::BTreeMap<u8, usize>)> =
            Default::default();
        let mut ticks = 0usize;
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            ticks += 1;
            for (i, c) in t.commands.iter().enumerate() {
                let list = if i < t.commands_a { 'A' } else { 'B' };
                let e = per.entry((list, c.kind)).or_default();
                e.0 += 1;
                e.1.push(c.payload.len());
                let ty = (c.payload.first().copied().unwrap_or(0) >> 3) & 0x1F;
                *e.2.entry(ty).or_insert(0) += 1;
            }
        }
        println!("\n== {} : {ticks} ticks ==", name(p));
        println!("list kind    count  per-tick   len p50  len p95   leading 5-bit values (top 6)");
        for ((l, k), (n, mut lens, tys)) in per {
            lens.sort_unstable();
            let q = |f: f32| lens[((lens.len() - 1) as f32 * f) as usize];
            let mut tv: Vec<_> = tys.into_iter().collect();
            tv.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = tv.iter().take(6).map(|(t, c)| format!("{t}:{c}")).collect();
            println!(
                "   {l} {k:>4}  {n:>7}  {:>8.3}  {:>7}  {:>7}   {}",
                n as f64 / ticks.max(1) as f64,
                q(0.5),
                q(0.95),
                top.join(" ")
            );
        }
    }
}

/// Reach object tag groups — the only groups a replicated OBJECT's definition can belong to.
/// A tag-reference field read at the right offset lands in this set; one read at the wrong offset
/// scatters across shaders and bitmaps, which is exactly what the first attempt did.
pub const OBJECT_GROUPS: &[&str] = &[
    "bipd", "vehi", "weap", "eqip", "garb", "proj", "scen", "mach", "ctrl", "crea", "efsc", "term",
    "ssce", "bloc", "obje", "gint", "lifi", "plac", "dgr!", "sinm", "spas",
];

/// Sweep (bit offset, width) for the creation payload's DEFINITION TAG REFERENCE, scored against
/// a sharp oracle: a correct read resolves to an OBJECT tag group, a wrong one does not.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_sweep_creation_tagref() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    // Resolve every tag index ONCE: the sweep asks about the same indices thousands of times.
    let mut class_of: Vec<Option<String>> = vec![None; 0x10000];
    for (i, slot) in class_of.iter_mut().enumerate() {
        *slot = native.tag_class(cache, i as u32);
    }
    let is_obj = |ti: u32| -> bool {
        class_of
            .get(ti as usize)
            .and_then(|o| o.as_deref())
            .map_or(false, |g| OBJECT_GROUPS.contains(&g))
    };

    // Gather creation payloads per entity type from the NORMAL films (one film re-creates its
    // whole Forge scene 8.9 times a second and would drown everything else).
    let mut by_type: std::collections::BTreeMap<u8, Vec<Vec<u8>>> = Default::default();
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams {
            cv3_axis_bits: w,
            cv3_total_bits: w.iter().sum(),
            angle_always: true,
            update_prefix_bits: 1,
            ..base
        };
        let mut creations = 0usize;
        let mut mine: std::collections::BTreeMap<u8, Vec<Vec<u8>>> = Default::default();
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.is_update {
                    continue;
                }
                creations += 1;
                let v = mine.entry(e.entity_type).or_default();
                if v.len() < 4000 {
                    v.push(e.payload.clone());
                }
            }
        }
        let rate = creations as f64 / f.data.records.len().max(1) as f64;
        if rate > 1.0 {
            println!("  SKIP {} ({rate:.2} creations/tick -- pathological)", name(p));
            continue;
        }
        used += 1;
        for (k, v) in mine {
            by_type.entry(k).or_default().extend(v);
        }
        if used >= 6 {
            break;
        }
    }
    println!("\n== creation-tagref sweep over {used} films ==");
    for (ty, pays) in &by_type {
        if pays.len() < 20 {
            continue;
        }
        let mut best: Vec<(f64, usize, u32, usize)> = Vec::new();
        for off in 0..40usize {
            for wid in 12..=17u32 {
                let mut ok = 0usize;
                let mut n = 0usize;
                let mut distinct = std::collections::BTreeSet::new();
                for pay in pays {
                    let mut r = film_decode::BitReader::new(pay);
                    if r.skip(off).is_err() {
                        continue;
                    }
                    let Ok(v) = r.bits(wid) else { continue };
                    n += 1;
                    distinct.insert(v);
                    if is_obj(v as u32) {
                        ok += 1;
                    }
                }
                if n > 0 {
                    best.push((ok as f64 / n as f64, off, wid, distinct.len()));
                }
            }
        }
        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        print!("type {ty:>2} ({:>6} creations): ", pays.len());
        for (s, o, w, d) in best.iter().take(4) {
            print!("  +{o}/{w}b = {:.1}% ({d} distinct)", s * 100.0);
        }
        println!();
    }
    // And for the WINNER of each type, what the tags actually are.
    println!("\n== resolved definitions at the best (offset, width) per type ==");
    for (ty, pays) in &by_type {
        if pays.len() < 20 {
            continue;
        }
        let mut best = (0.0f64, 0usize, 14u32);
        for off in 0..40usize {
            for wid in 12..=17u32 {
                let (mut ok, mut n) = (0usize, 0usize);
                for pay in pays {
                    let mut r = film_decode::BitReader::new(pay);
                    if r.skip(off).is_err() {
                        continue;
                    }
                    let Ok(v) = r.bits(wid) else { continue };
                    n += 1;
                    if is_obj(v as u32) {
                        ok += 1;
                    }
                }
                let s = if n > 0 { ok as f64 / n as f64 } else { 0.0 };
                if s > best.0 {
                    best = (s, off, wid);
                }
            }
        }
        if best.0 < 0.5 {
            println!("\ntype {ty}: no offset resolves to object tags (best {:.1}%)", best.0 * 100.0);
            continue;
        }
        let mut hist: std::collections::BTreeMap<String, usize> = Default::default();
        for pay in pays {
            let mut r = film_decode::BitReader::new(pay);
            if r.skip(best.1).is_err() {
                continue;
            }
            let Ok(v) = r.bits(best.2) else { continue };
            let label = match (native.tag_class(cache, v as u32), native.tag_name(cache, v as u32)) {
                (Some(g), Some(n)) => format!("{g} {n}"),
                _ => format!("<unresolved {v}>"),
            };
            *hist.entry(label).or_insert(0) += 1;
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\ntype {ty} at +{}/{}b ({:.1}% object tags, {} distinct):", best.1, best.2, best.0 * 100.0, v.len());
        for (k, c) in v.iter().take(14) {
            println!("   {c:>7}  {k}");
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 7 — identify a stream entity by WHERE IT SPAWNED
// ---------------------------------------------------------------------------------------------

/// The creation payload's tag reference is not readable at any fixed offset (see
/// `probe_sweep_creation_tagref`, a clean negative). But a Forge-placed object spawns **at its
/// Forge spawn point**, and HMS already knows every one of those: the film embeds the whole map
/// variant, and a slot's quota index resolves through the scenario sandbox palette to a real tag.
///
/// So: take each stream entity's FIRST decoded position, find the nearest present Forge object,
/// and report the distance and the tag group. If type 15 really is the vehicle, type-15 entities
/// spawn on `vehi` palette objects.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_spawn_point_identity() {
    let films = films();
    if films.is_empty() {
        eprintln!("film_anim: no films on this machine");
        return;
    }
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    let Some(scnr) = native.find_scenario(cache) else { return };
    let palette = native.scnr_forge_palette_full(cache, scnr);
    let mut types: Vec<(u32, u32)> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if seen.insert(k) {
                types.push(k);
            }
        }
    }
    // quota index -> (tag group, tag path)
    let resolve = |folder: u16, item: u8| -> Option<(String, String)> {
        let &(pi, ew) = types.get(folder as usize)?;
        let p = palette
            .iter()
            .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
            .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
        Some((
            native.tag_class(cache, p.tag_short).unwrap_or_default(),
            native.tag_name(cache, p.tag_short).unwrap_or_default(),
        ))
    };

    let mut per_type: std::collections::BTreeMap<u8, std::collections::BTreeMap<String, usize>> = Default::default();
    let mut dist_per_type: std::collections::BTreeMap<u8, Vec<f32>> = Default::default();
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        used += 1;
        if used > 6 {
            break;
        }
        let objs: Vec<&film::VariantObject> = f.header.map_variant.present_objects().collect();
        for t in &idx.tracks {
            let Some(&(_, first)) = t.positions.first() else { continue };
            let mut best: Option<(f32, u16, u8)> = None;
            for o in &objs {
                let d = ((o.position[0] - first[0]).powi(2)
                    + (o.position[1] - first[1]).powi(2)
                    + (o.position[2] - first[2]).powi(2))
                .sqrt();
                if best.map_or(true, |b| d < b.0) {
                    best = Some((d, o.quota_index, o.item));
                }
            }
            let Some((d, q, it)) = best else { continue };
            dist_per_type.entry(t.entity_type).or_default().push(d);
            if d < 1.5 {
                let label = resolve(q, it)
                    .map(|(g, n)| format!("{g} {}", n.rsplit('\\').next().unwrap_or(&n)))
                    .unwrap_or_else(|| format!("<quota {q}>"));
                *per_type.entry(t.entity_type).or_default().entry(label).or_insert(0) += 1;
            }
        }
    }
    println!("\n== nearest Forge spawn point to each entity's FIRST position, {used} films ==");
    println!("type   entities   p10 dist  p50 dist   within 1.5 wu");
    for (ty, ds) in &dist_per_type {
        let mut d = ds.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f32| d[((d.len() - 1) as f32 * f) as usize];
        let near = d.iter().filter(|&&x| x < 1.5).count();
        println!(
            "{ty:>4}   {:>8}  {:>9.2}  {:>8.2}   {:>5} ({:.0}%)",
            d.len(),
            q(0.10),
            q(0.50),
            near,
            near as f64 / d.len() as f64 * 100.0
        );
    }
    for (ty, m) in &per_type {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("\ntype {ty} spawned on ({} distinct palette objects):", m.len());
        for (k, c) in v.iter().take(14) {
            println!("   {c:>7}  {k}");
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 8 — the ARMOUR ABILITY: a typed TagReference('eqip') in a biped's creation payload
// ---------------------------------------------------------------------------------------------

/// The spec states, from the disassembly, that a type-11 creation reads a
/// `TagReference('eqip')` and the type-11 dirty block a `TagReference('weap')`. A *typed* group
/// reference is `bits_needed(group_count + 1)` bits wide — far narrower than the 14-bit untyped
/// one — and in Reach the equipment a player carries IS their armour ability.
///
/// So sweep (offset, width) over real type-11 creation payloads and score each candidate by how
/// often it indexes an `eqip` tag that is one of Reach's seven armour abilities. A wrong field
/// scatters over all the equipment in the map; the right one concentrates on seven tags.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_sweep_armour_ability() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    let eqip = native.tags_by_group(cache, "eqip");
    let eq_name: Vec<String> = eqip.iter().map(|&t| native.tag_name(cache, t).unwrap_or_default()).collect();
    // Reach's seven armour abilities, by the tag path the map carries.
    let is_ability = |i: usize| -> bool {
        eq_name.get(i).map_or(false, |n| {
            ["active_camouflage", "armor_lockup", "drop_shield", "evade", "hologram", "jet_pack", "sprint"]
                .iter()
                .any(|a| n.ends_with(&format!("\\{a}")))
        })
    };
    println!("== eqip: {} tags, bits_needed(n+1) = {} ==", eqip.len(), film_decode::ceil_log2(eqip.len() as u32 + 1));

    // Collect type-11 creation payloads, and the type-17 ones for the 'weap' control.
    let mut pays: Vec<Vec<u8>> = Vec::new();
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        let mut n = 0usize;
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.is_update || e.entity_type != 11 {
                    continue;
                }
                n += 1;
                if pays.len() < 20000 {
                    pays.push(e.payload.clone());
                }
            }
        }
        if n > 0 {
            used += 1;
        }
        if used >= 6 {
            break;
        }
    }
    println!("{} type-11 creation payloads from {used} films, median {} bytes", pays.len(),
        { let mut l: Vec<usize> = pays.iter().map(|p| p.len()).collect(); l.sort(); l.get(l.len()/2).copied().unwrap_or(0) });

    let mut scored: Vec<(f64, usize, u32, usize, String)> = Vec::new();
    for off in 0..420usize {
        for wid in 4..=6u32 {
            let mut hit = 0usize;
            let mut n = 0usize;
            let mut hist: std::collections::BTreeMap<u64, usize> = Default::default();
            for pay in &pays {
                let mut r = film_decode::BitReader::new(pay);
                if r.skip(off).is_err() {
                    continue;
                }
                let Ok(v) = r.bits(wid) else { continue };
                n += 1;
                *hist.entry(v).or_insert(0) += 1;
                if is_ability(v as usize) {
                    hit += 1;
                }
            }
            if n < pays.len() / 2 {
                continue;
            }
            let mut hv: Vec<_> = hist.into_iter().collect();
            hv.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = hv.iter().take(8).map(|(v, c)| format!("{v}:{}", c * 100 / n)).collect();
            scored.push((hit as f64 / n as f64, off, wid, hv.len(), top.join(" ")));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\n== sweep: fraction of values naming one of the SEVEN armour abilities ==");
    println!("score   off  wid  distinct  value histogram (% of samples)");
    for (s, o, w, d, h) in scored.iter().take(14) {
        println!("{:.3}  {o:>4}  {w:>3}  {d:>8}  {h}", s);
    }
    if let Some((s, o, w, _, _)) = scored.first() {
        if *s > 0.5 {
            let mut hist: std::collections::BTreeMap<u64, usize> = Default::default();
            for pay in &pays {
                let mut r = film_decode::BitReader::new(pay);
                if r.skip(*o).is_err() {
                    continue;
                }
                if let Ok(v) = r.bits(*w) {
                    *hist.entry(v).or_insert(0) += 1;
                }
            }
            println!("\n== winner +{o}/{w}b resolved against the eqip table ==");
            let mut hv: Vec<_> = hist.into_iter().collect();
            hv.sort_by(|a, b| b.1.cmp(&a.1));
            for (v, c) in hv {
                println!("   {c:>6}  {v:>3}  {}", eq_name.get(v as usize).map(|s| s.as_str()).unwrap_or("<out of range>"));
            }
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 9 — the ability field, scored against the JETPACK oracle instead of a value range
// ---------------------------------------------------------------------------------------------

/// Per-biped behaviour measured from the decoded positions alone — the oracles the brief lists.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyBehaviour {
    pub net_id: u32,
    /// Seconds of sustained climb (dz > 1.5 wu/s): a JETPACK, or at worst a lift.
    pub climb_s: f32,
    /// Seconds at a dead stop while the body still exists: ARMOUR LOCK.
    pub stop_s: f32,
    /// Fastest sustained speed, wu/s: SPRINT shows up here.
    pub top_speed: f32,
    /// Seconds above the 95th percentile of the ordinary run, pooled: sustained sprint.
    pub fast_s: f32,
    pub alive_s: f32,
}

pub fn body_behaviour(t: &film_decode::EntityTrack, fast_cut: f32) -> BodyBehaviour {
    let mut b = BodyBehaviour { net_id: t.net_id, alive_s: (t.last_tick - t.first_tick) as f32 / 60.0, ..Default::default() };
    for w in t.positions.windows(2) {
        let (t0, p0) = w[0];
        let (t1, p1) = w[1];
        if t1 <= t0 || t1 - t0 > 12 {
            continue;
        }
        let dt = (t1 - t0) as f32 / 60.0;
        let d = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        let s = d / dt;
        if s >= 30.0 {
            continue;
        }
        b.top_speed = b.top_speed.max(s);
        if s > fast_cut {
            b.fast_s += dt;
        }
        if s < 0.05 {
            b.stop_s += dt;
        }
        if (p1[2] - p0[2]) / dt > 1.5 {
            b.climb_s += dt;
        }
    }
    b
}

/// Collect, per biped entity, its creation payload and its measured behaviour.
pub fn bipeds_with_payloads(p: &std::path::Path) -> Option<(film::Film, FilmIndex, std::collections::HashMap<u32, Vec<u8>>)> {
    let (f, idx) = indexed(p)?;
    let params = idx.params;
    let mut pay: std::collections::HashMap<u32, Vec<u8>> = Default::default();
    for rec in f.data.records.iter() {
        let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
        for e in &t.entities {
            if !e.is_update && e.entity_type == 11 {
                pay.entry(e.net_id).or_insert_with(|| e.payload.clone());
            }
        }
    }
    Some((f, idx, pay))
}

/// Sweep the type-11 creation payload for a field that SEPARATES jetpackers from everyone else.
///
/// This is the oracle the brief asks for and it cannot be gamed by a constant: a constant field
/// puts every body in one group and separates nothing. The score is the ratio of climb-per-body
/// inside the best-separating value to climb-per-body outside it, so 1.0 = no information.
#[test]
fn probe_sweep_ability_against_jetpack() {
    let films = films();
    if films.is_empty() {
        return;
    }
    // pooled speed cut for "sprinting": p95 of all biped speeds
    let mut rows: Vec<(BodyBehaviour, Vec<u8>)> = Vec::new();
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Some((f, idx, pay)) = bipeds_with_payloads(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        used += 1;
        if used > 6 {
            break;
        }
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11 && t.positions.len() > 30) {
            let Some(pl) = pay.get(&t.net_id) else { continue };
            rows.push((body_behaviour(t, 4.2), pl.clone()));
        }
    }
    let total_climb: f32 = rows.iter().map(|r| r.0.climb_s).sum();
    let total_stop: f32 = rows.iter().map(|r| r.0.stop_s).sum();
    let total_fast: f32 = rows.iter().map(|r| r.0.fast_s).sum();
    println!(
        "\n== {} bipeds with payloads from {used} films: {:.0} s climb, {:.0} s dead-stop, {:.0} s fast ==",
        rows.len(),
        total_climb,
        total_stop,
        total_fast
    );
    if rows.len() < 50 {
        println!("too few bodies to sweep");
        return;
    }
    // Score every (offset, width): how well does the field concentrate the climbing?
    #[derive(Clone)]
    struct Cand {
        score: f64,
        off: usize,
        wid: u32,
        groups: usize,
        best_val: u64,
        detail: String,
    }
    let mut out: Vec<Cand> = Vec::new();
    for off in 0..480usize {
        for wid in 3..=6u32 {
            let mut g: std::collections::BTreeMap<u64, (usize, f32, f32, f32)> = Default::default();
            let mut n = 0usize;
            for (b, pl) in &rows {
                let mut r = film_decode::BitReader::new(pl);
                if r.skip(off).is_err() {
                    continue;
                }
                let Ok(v) = r.bits(wid) else { continue };
                n += 1;
                let e = g.entry(v).or_insert((0, 0.0, 0.0, 0.0));
                e.0 += 1;
                e.1 += b.climb_s;
                e.2 += b.stop_s;
                e.3 += b.fast_s;
            }
            if n < rows.len() / 2 || g.len() < 2 || g.len() > 16 {
                continue;
            }
            // best separation of CLIMB
            let mut best = (1.0f64, 0u64);
            for (&v, &(cnt, climb, _, _)) in &g {
                if cnt < 3 {
                    continue;
                }
                let inside = climb as f64 / cnt as f64;
                let out_cnt = n - cnt;
                let out_climb = (total_climb - climb) as f64;
                if out_cnt == 0 {
                    continue;
                }
                let outside = out_climb / out_cnt as f64;
                let ratio = (inside + 0.01) / (outside + 0.01);
                if ratio > best.0 {
                    best = (ratio, v);
                }
            }
            let mut gv: Vec<_> = g.iter().collect();
            gv.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            let detail: Vec<String> = gv
                .iter()
                .take(8)
                .map(|(v, (c, cl, st, fa))| format!("{v}:n{c}/cl{:.0}/st{:.0}/fa{:.0}", cl, st, fa))
                .collect();
            out.push(Cand { score: best.0, off, wid, groups: g.len(), best_val: best.1, detail: detail.join(" ") });
        }
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    println!("ratio   off  wid  groups  best   per-value  n=bodies cl=climb-s st=stop-s fa=fast-s");
    for c in out.iter().take(16) {
        println!("{:>6.2}  {:>4}  {:>3}  {:>6}  {:>4}   {}", c.score, c.off, c.wid, c.groups, c.best_val, c.detail);
    }
}

// ---------------------------------------------------------------------------------------------
// probe 10 — what the Spartan jmad graph actually offers
// ---------------------------------------------------------------------------------------------

/// Before deciding what to decode, find out what can be DRAWN. The locomotion state is only worth
/// recovering to the resolution the animation graph can express.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_spartan_anim_vocabulary() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    let Some((tag, path)) = crate::reach_actor::ActorScene::list_bipeds(&native, cache)
        .into_iter()
        .find(|(_, n)| n.contains("characters\\spartans\\spartans"))
    else {
        println!("no spartan bipd");
        return;
    };
    let Some(def) = crate::reach_actor::ActorDef::read(&native, cache, tag) else { return };
    let Some(g) = def.graph.as_ref() else {
        println!("spartan has no jmad");
        return;
    };
    println!("== {path}: {} animations ==", g.anims.len());
    let groups = [
        "jump", "airborne", "land", "evade", "armor_lock", "jet_pack", "assassin",
        "melee", "berserk", "throw", "flight:", "sprint:", "warthog", "ghost_", "falcon", "mongoose",
        "enter", "exit", "board", "death", "die", "impulse", "sync",
    ];
    for k in groups {
        let mut hit: Vec<&str> = g
            .anims
            .iter()
            .map(|a| a.name.as_str())
            .filter(|n| n.contains(k))
            .collect();
        hit.sort_unstable();
        println!("\n-- '{k}' : {} --", hit.len());
        for chunk in hit.iter().take(24).collect::<Vec<_>>().chunks(4) {
            println!("   {}", chunk.iter().map(|s| **s).collect::<Vec<_>>().join("  |  "));
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 11 — calibrate the RIDE threshold
// ---------------------------------------------------------------------------------------------

/// How close does a biped's replicated position actually get to the vehicle it is riding, and how
/// close do non-riders get? The gap between those two distributions is the whole of the ride
/// detector's reliability, so it is measured rather than assumed.
#[test]
fn probe_ride_distance_calibration() {
    let films = films();
    if films.is_empty() {
        return;
    }
    // "riding" is now defined by the SHIPPING detector, so the calibration measures the rule that
    // actually runs rather than a stand-in heuristic.
    let mut riding: Vec<f32> = Vec::new();
    let mut riding_xy: Vec<f32> = Vec::new();
    let mut nearby: Vec<f32> = Vec::new();
    let (mut rides_total, mut films_used) = (0usize, 0usize);
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        films_used += 1;
        if films_used > 8 {
            break;
        }
        let rides = crate::film_anim::accepted_rides(&idx);
        rides_total += rides.len();
        println!("  {:<38} {:>3} rides", name(p).chars().take(38).collect::<String>(), rides.len());
        let at = |net: u32, t: u32| idx.tracks.iter().find(|x| x.net_id == net).and_then(|x| x.sample(t));
        for r in &rides {
            let mut t = r.from_tick;
            while t <= r.to_tick {
                if let (Some(a), Some(c)) = (at(r.biped_net_id, t), at(r.vehicle_net_id, t)) {
                    riding.push(((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2)).sqrt());
                    riding_xy.push(((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)).sqrt());
                }
                t += 6;
            }
        }
        // everything else within 4 wu of a moving vehicle: the population the rule must reject
        let veh: Vec<usize> = (0..idx.tracks.len())
            .filter(|&i| idx.tracks[i].entity_type == 15 && idx.tracks[i].moves())
            .collect();
        for &vi in &veh {
            let v = &idx.tracks[vi];
            for b in idx.tracks.iter().filter(|x| x.entity_type == 11 && x.positions.len() > 10) {
                if rides.iter().any(|r| r.biped_net_id == b.net_id && r.vehicle_net_id == v.net_id) {
                    continue;
                }
                let lo = b.first_tick.max(v.first_tick);
                let hi = b.last_tick.min(v.last_tick);
                let mut t = lo;
                while t <= hi {
                    if let (Some(a), Some(c)) = (b.sample(t), v.sample(t)) {
                        let d = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2)).sqrt();
                        if d < 4.0 {
                            nearby.push(d);
                        }
                    }
                    t += 6;
                }
            }
        }
    }
    println!("\n== {rides_total} rides over {films_used} films, under the HORIZONTAL-displacement rule ==");
    for (label, mut v) in [("RIDING (3D gap)", riding), ("RIDING (xy gap)", riding_xy), ("not riding, within 4 wu", nearby)] {
        if v.is_empty() {
            println!("{label}: no samples");
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f32| v[((v.len() - 1) as f32 * f) as usize];
        println!(
            "{label:<26} n={:<8} p05={:.2} p25={:.2} p50={:.2} p75={:.2} p95={:.2}",
            v.len(), q(0.05), q(0.25), q(0.50), q(0.75), q(0.95)
        );
    }
}

#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_vehicle_identification_detail() {
    let want = ["4ED6DDC1_67FF2BA9", "4ED6DDC1_67FF2AFF", "21897D8A_6A95B5EF", "4832BB46_6774B938"];
    let films = films();
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("no forge_halo cache");
        return;
    };
    let Some(scnr) = native.find_scenario(cache) else { return };
    let palette = native.scnr_forge_palette_full(cache, scnr);
    let mut order: Vec<(u32, u32)> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if seen.insert(k) {
                order.push(k);
            }
        }
    }
    println!("palette {} entries, {} distinct types", palette.len(), order.len());
    for p in films.iter() {
        let n = name(p);
        if !want.iter().any(|w| n.contains(w)) {
            continue;
        }
        let Some((f, idx)) = indexed(p) else { continue };
        println!("\n== {n} : {} tracks ==", idx.tracks.len());
        let rides = crate::film_anim::find_rides(&idx);
        let ridden: std::collections::HashSet<u32> = rides.iter().map(|r| r.vehicle_net_id).collect();
        for tr in idx.tracks.iter().filter(|t| t.entity_type == 15 && ridden.contains(&t.net_id)) {
            println!("   RIDDEN {:#010x}: spawned_on {:?}, {} positions, first {:?}",
                tr.net_id, tr.spawned_on, tr.positions.len(), tr.positions.first().map(|p| p.1));
        }
        let mut hist: std::collections::BTreeMap<String, usize> = Default::default();
        let mut errs: Vec<f32> = Vec::new();
        for tr in idx.tracks.iter().filter(|t| t.entity_type == 15) {
            let label = match tr.spawned_on {
                None => "<no position / no spawn match>".to_string(),
                Some((q, it, err)) => {
                    errs.push(err);
                    let r = order.get(q as usize).and_then(|&(pi, ew)| {
                        palette
                            .iter()
                            .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == it as u32)
                            .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))
                    });
                    match r {
                        Some(e) => format!(
                            "err {:.2} -> {} {}",
                            err,
                            native.tag_class(cache, e.tag_short).unwrap_or_default(),
                            native.tag_name(cache, e.tag_short).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string()
                        ),
                        None => format!("err {err:.2} -> quota {q} UNRESOLVED (order len {})", order.len()),
                    }
                }
            };
            *hist.entry(label).or_insert(0) += 1;
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, c) in v.iter().take(20) {
            println!("   {c:>4}  {k}");
        }
        println!("   variant: {} present objects, quota max {:?}",
            f.header.map_variant.present_objects().count(),
            f.header.map_variant.present_objects().map(|o| o.quota_index).max());
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 12 — list-B kind 4 is the entity DESTROY message
// ---------------------------------------------------------------------------------------------

/// Kind 4 carries 5 bytes and its 5-bit leading field has the same entity-type histogram as the
/// creations (`13:7255 12:1542 17:1026 11:523 ...` against `13:7790 12:1610 17:1126 11:559 ...`),
/// which says it is a per-entity message about the same population. If it is a DESTROY, then
/// every kind-4 net id must be an id that was CREATED earlier in the same film, and no id should
/// be destroyed twice without being re-created. Both are checked here.
///
/// This matters because a parked vehicle sends one creation and then nothing: without a
/// destroy tick its track's `last_tick` equals its `first_tick` and it is never "alive".
#[test]
fn probe_kind4_is_destroy() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                 kind4   matched a live id   double-destroy   never-created");
    for p in films.iter().take(8) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        let mut live: std::collections::HashSet<u32> = Default::default();
        let (mut n4, mut ok, mut dbl, mut never) = (0usize, 0usize, 0usize, 0usize);
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if !e.is_update {
                    live.insert(e.net_id);
                }
            }
            for (i, c) in t.commands.iter().enumerate() {
                if i < t.commands_a || c.kind != 4 || c.payload.len() < 4 {
                    continue;
                }
                n4 += 1;
                let mut r = film_decode::BitReader::new(&c.payload);
                let _ty = r.bits(5).unwrap_or(0);
                let hi = r.bits(16).unwrap_or(0);
                let lo = r.bits(11).unwrap_or(0);
                let id = ((hi << 16) | lo) as u32;
                if live.remove(&id) {
                    ok += 1;
                } else if id == 0 {
                    never += 1;
                } else {
                    dbl += 1;
                }
            }
        }
        println!(
            "{:<36} {n4:>6}   {ok:>8} ({:>5.1}%)   {dbl:>10}   {never:>10}",
            name(p).chars().take(36).collect::<String>(),
            ok as f64 / n4.max(1) as f64 * 100.0
        );
    }
}

/// End-to-end: build the real `VehicleTable` the app builds, and report every ride with the
/// vehicle NAMED. This is the shipping path, measured.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_vehicle_table_end_to_end() {
    let films = films();
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("no forge_halo cache");
        return;
    };
    let Some(scnr) = native.find_scenario(cache) else { return };
    let palette = native.scnr_forge_palette_full(cache, scnr);
    let mut order: Vec<(u32, u32)> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if seen.insert(k) {
                order.push(k);
            }
        }
    }
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        used += 1;
        if used > 8 {
            break;
        }
        let vt = crate::film_anim::VehicleTable::build(&idx, |folder, item| {
            let &(pi, ew) = order.get(folder as usize)?;
            let e = palette
                .iter()
                .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
                .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
            if native.tag_class(cache, e.tag_short).as_deref() != Some("vehi") {
                return None;
            }
            Some((e.tag_short, native.tag_name(cache, e.tag_short).unwrap_or_default()))
        });
        let total = idx.tracks.iter().filter(|t| t.entity_type == 15).count();
        println!(
            "\n== {} : {total} vehicles, {} identified, {} not, {} rides ==",
            name(p),
            vt.vehicles.len(),
            vt.unidentified,
            vt.rides.len()
        );
        let mut worst: Vec<&crate::film_anim::VehicleIdent> = vt.vehicles.values().collect();
        worst.sort_by(|a, b| b.spawn_error.partial_cmp(&a.spawn_error).unwrap());
        for v in worst.iter().take(3) {
            println!("   loosest match: {} at {:.2} wu (seat family '{}')", v.name, v.spawn_error, v.seat_family);
        }
        let mut shown = 0;
        for r in &vt.rides {
            let v = vt.vehicles.get(&r.vehicle_net_id);
            let seat = vt.seat_of(r.biped_net_id, (r.from_tick + r.to_tick) / 2);
            println!(
                "   ride {:.1}s from {}: body {:#010x} in {} -> seat {:?}",
                r.seconds(),
                crate::film_play::fmt_time(r.from_tick as f32 / 60.0),
                r.biped_net_id,
                v.map(|v| v.name.as_str()).unwrap_or("<UNIDENTIFIED VEHICLE>"),
                seat.map(|s| s.mode)
            );
            shown += 1;
            if shown >= 10 {
                println!("   ... and {} more rides", vt.rides.len() - shown);
                break;
            }
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

/// What markers a vehicle's render model carries — the engine seats a unit at one of them.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app <name> -- --ignored --nocapture --test-threads=1"]
fn probe_vehicle_seat_markers() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else { return };
    for want in ["mongoose", "warthog\\warthog", "ghost\\ghost", "scorpion", "falcon\\falcon", "revenant\\revenant"] {
        let Some((tag, path)) = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .map(|t| (t, native.tag_name(cache, t).unwrap_or_default()))
            .find(|(_, n)| n.contains(want))
        else {
            println!("no vehi matching {want}");
            continue;
        };
        let Some(def) = crate::reach_actor::ActorDef::read(&native, cache, tag) else {
            println!("{path}: no render model");
            continue;
        };
        let names = def.model.marker_names(&native, cache);
        println!("\n== {path} : {} nodes, {} marker groups ==", def.model.skel.len(), names.len());
        println!("   {}", names.join(", "));
        for n in ["driver", "seat", "gunner", "passenger", "rider"] {
            if let Some(g) = def.model.marker_group(&native, cache, n) {
                for m in &g.markers {
                    println!("   '{n}' -> node {} t ({:.3},{:.3},{:.3})", m.node, m.translation.x, m.translation.y, m.translation.z);
                }
            }
        }
    }
    // the cache is shared for the whole test binary; it is not closed here
}

// ---------------------------------------------------------------------------------------------
// probe 13 — how far into a biped's DERIVED block can a decoder actually get?
// ---------------------------------------------------------------------------------------------

/// The engine's biped dirty reader (`sub_1803B4B88`) walks fields 0x14..0x3B with the gate list
/// `{0x15, 0x1A, 0x1F, 0x24, 0x29, 0x2E}`. Field 0x36 is the OBJECT ACTION record: a 7-bit enum
/// whose names are literals in the binary — `_action_assassinate` (4), `_action_vehicle_enter`
/// (0x10), `_action_jump` (0x20), `_action_melee_ai` (0x2D), `_action_weapon_pickup` (0x2E)…
/// That is the whole combat-event answer, IF a decoder can walk that far.
///
/// It can only walk that far when every gate before it is SET. This probe measures how often the
/// gates pass, which bounds the whole feature before a line of walk code is written.
#[test]
fn probe_biped_derived_gate_reach() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                  updates  latch@2   gate0x14  gate0x15  reach 0x29  reach 0x2E  reach 0x36");
    for p in films.iter().take(6) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        let (mut n, mut latched, mut p14, mut g15, mut r29, mut r2e, mut r36) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || e.dirty_mask == 0 {
                    continue;
                }
                n += 1;
                // The header mask carries the SAME information as the presence bits -- the stream
                // says twice which fields are present -- so the gate question can be answered
                // from the mask alone, with no walk at all.
                let m = e.dirty_mask;
                if m & (1 << 0x14) != 0 {
                    p14 += 1;
                }
                if (m >> 0x15) & 0x1F != 0 {
                    g15 += 1;
                }
                if m & (1u64 << 0x29) != 0 {
                    r29 += 1;
                }
                if m & (1u64 << 0x2E) != 0 {
                    r2e += 1;
                }
                if m & (1u64 << 0x36) != 0 {
                    r36 += 1;
                }
                let Some(st) = film_decode::decode_entity_payload(e, &params) else { continue };
                if st.latched_at == Some(2) {
                    latched += 1;
                }
            }
        }
        println!(
            "{:<36} {n:>8}  {:>6.1}%  {:>8.2}% {:>8.2}%  {:>9.3}%  {:>9.3}%  {:>9.4}%",
            name(p).chars().take(36).collect::<String>(),
            latched as f64 / n.max(1) as f64 * 100.0,
            p14 as f64 / n.max(1) as f64 * 100.0,
            g15 as f64 / n.max(1) as f64 * 100.0,
            r29 as f64 / n.max(1) as f64 * 100.0,
            r2e as f64 / n.max(1) as f64 * 100.0,
            r36 as f64 / n.max(1) as f64 * 100.0,
        );
    }
}

/// The full 64-bit dirty-mask census for bipeds: WHICH derived fields ever appear.
#[test]
fn probe_biped_mask_census() {
    let films = films();
    if films.is_empty() {
        return;
    }
    const FIELD: [(u32, &str); 18] = [
        (0x14, "aim yaw+pitch"),
        (0x19, "weapon0 model"),
        (0x29, "weapon selection"),
        (0x2A, "weapon0 'weap' tag"),
        (0x2E, "grenade counts"),
        (0x2F, "controlling player"),
        (0x30, "body generation"),
        (0x31, "equipment/ability mode"),
        (0x32, "equipment object ref"),
        (0x33, "flashlight"),
        (0x34, "in play / gone"),
        (0x35, "ASSASSINATION"),
        (0x36, "OBJECT ACTION enum"),
        (0x37, "control input record"),
        (0x38, "ai 0x38"),
        (0x39, "ai 0x39"),
        (0x3A, "ai 0x3A"),
        (0x3B, "ai 0x3B"),
    ];
    let mut tot = vec![0usize; 64];
    let mut n = 0usize;
    let mut films_used = 0;
    for p in films.iter().take(8) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        films_used += 1;
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 {
                    continue;
                }
                n += 1;
                for b in 0..64 {
                    if e.dirty_mask & (1u64 << b) != 0 {
                        tot[b] += 1;
                    }
                }
            }
        }
    }
    println!("\n== biped dirty-mask bit frequency over {n} type-11 messages, {films_used} films ==");
    for b in 0..64 {
        if tot[b] == 0 {
            continue;
        }
        let label = FIELD.iter().find(|(i, _)| *i == b as u32).map(|(_, s)| *s).unwrap_or(if b < 0x14 { "base field" } else { "" });
        println!("  bit {b:#04x} ({b:>2}) {:>9}  {:>7.4}%   {label}", tot[b], tot[b] as f64 / n.max(1) as f64 * 100.0);
    }
}

/// What else is set on a message that carries field 0x35 (the assassination record)? If those
/// messages are simple, the walk needed to reach 0x35 is short and the field is reachable.
#[test]
fn probe_assassination_message_shape() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let mut masks: std::collections::BTreeMap<u64, usize> = Default::default();
    let mut lens: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut total = 0usize;
    let mut samples: Vec<(String, u32, Vec<u8>, u64)> = Vec::new();
    for p in films.iter().take(8) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || e.dirty_mask & (1u64 << 0x35) == 0 {
                    continue;
                }
                total += 1;
                *masks.entry(e.dirty_mask).or_insert(0) += 1;
                *lens.entry(e.payload.len()).or_insert(0) += 1;
                if samples.len() < 6 {
                    samples.push((name(p), t.tick, e.payload.clone(), e.dirty_mask));
                }
            }
        }
    }
    println!("\n== {total} biped messages with field 0x35 (assassination) ==");
    let mut m: Vec<_> = masks.into_iter().collect();
    m.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  mask histogram (top 12):");
    for (k, c) in m.iter().take(12) {
        let bits: Vec<String> = (0..64).filter(|b| k & (1u64 << b) != 0).map(|b| format!("{b:#x}")).collect();
        println!("    {c:>6}  {k:#018x}  = {}", bits.join(","));
    }
    println!("  payload length histogram:");
    let mut l: Vec<_> = lens.into_iter().collect();
    l.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, c) in l.iter().take(10) {
        println!("    {k:>4} bytes : {c}");
    }
    for (f, t, pay, m) in &samples {
        println!("  sample {f} tick {t} mask {m:#x} payload {:02x?}", &pay[..pay.len().min(24)]);
    }
}

// ---------------------------------------------------------------------------------------------
// probe 14 — does the biped derived walk CLOSE?
// ---------------------------------------------------------------------------------------------

/// The oracle for `film_anim::unit::decode`. A correct walk must (a) consume the payload to
/// within 7 bits (the writer rounds up to a byte) and (b) rebuild the message's own 64-bit header
/// mask exactly, because the stream states which fields are present TWICE. Both are free.
#[test]
fn probe_unit_walk_oracle() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                 messages   closed   mask ok   BOTH     assassinations   aim");
    let (mut gt, mut gb, mut ga, mut gm) = (0usize, 0usize, 0usize, 0usize);
    for p in films.iter().take(8) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        let (mut n, mut closed, mut maskok, mut both, mut ass, mut aim) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || !e.is_update || e.dirty_mask == 0 {
                    continue;
                }
                n += 1;
                let Some(st) = crate::film_anim::unit::decode(e, &params, 6) else { continue };
                let c = st.stopped_at.is_none() && st.leftover_bits < 8;
                let m = st.rebuilt_mask == e.dirty_mask;
                if c {
                    closed += 1;
                }
                if m {
                    maskok += 1;
                }
                if c && m {
                    both += 1;
                    if st.assassination.is_some() {
                        ass += 1;
                    }
                    if st.aim_yaw_q.is_some() {
                        aim += 1;
                    }
                }
            }
        }
        println!(
            "{:<36} {n:>8}  {:>6.1}%  {:>6.1}%  {:>6.1}%  {ass:>12}  {aim:>10}",
            name(p).chars().take(36).collect::<String>(),
            closed as f64 / n.max(1) as f64 * 100.0,
            maskok as f64 / n.max(1) as f64 * 100.0,
            both as f64 / n.max(1) as f64 * 100.0,
        );
        gt += n;
        gb += both;
        ga += ass;
        gm += maskok;
    }
    println!(
        "\nTOTAL {gt} messages: {:.2}% rebuilt the header mask, {:.2}% passed BOTH oracles, {ga} assassination records decoded",
        gm as f64 / gt.max(1) as f64 * 100.0,
        gb as f64 / gt.max(1) as f64 * 100.0
    );
}

/// Drill into the messages that carry field 0x35, where the walk has to reach the end.
#[test]
fn probe_assassination_walk() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let (mut n, mut maskok, mut got) = (0usize, 0usize, 0usize);
    let mut leftover: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut stopped: std::collections::BTreeMap<String, usize> = Default::default();
    let mut shown = 0;
    for p in films.iter().filter(|p| {
        // The 8 biggest films, PLUS the clip whose gametype is literally called "Assassination."
        name(p).contains("21897D8A_6A95B5EF") || true
    }).take(14) {
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (frame, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || !e.is_update || e.dirty_mask & (1u64 << 0x35) == 0 {
                    continue;
                }
                n += 1;
                let Some(st) = crate::film_anim::unit::decode(e, &params, 6) else { continue };
                if st.rebuilt_mask == e.dirty_mask {
                    maskok += 1;
                }
                *leftover.entry(st.bits_at_0x35.unwrap_or(999)).or_insert(0) += 1;
                *stopped.entry(format!("stop={:?} latch={:?} rebuilt={:#x}", st.stopped_at, st.latched_at, st.rebuilt_mask)).or_insert(0) += 1;
                if let Some(a) = st.assassination {
                    got += 1;
                    let real = a.target.map_or(false, |x| x != 0 && x != 0x7FFF_07FF && x != 0xFFFF_FFFF);
                    if real {
                        *stopped.entry(format!("REAL TARGET {:#010x}", a.target.unwrap())).or_insert(0) += 1;
                    }
                    if shown < 10 && real {
                        let pos = a.position_raw.map(|r| frame.world(r));
                        println!(
                            "  {} tick {:>7}  attacker {:#010x} -> target {:#010x}  at {:?}  leftover {} bits  mask {}",
                            name(p).chars().take(22).collect::<String>(),
                            t.tick, e.net_id, a.target.unwrap(),
                            pos.map(|p| [(p[0] * 100.0).round() / 100.0, (p[1] * 100.0).round() / 100.0, (p[2] * 100.0).round() / 100.0]),
                            st.leftover_bits,
                            if st.rebuilt_mask == e.dirty_mask { "OK" } else { "BAD" }
                        );
                        shown += 1;
                    }
                }
            }
        }
    }
    println!("\n== {n} messages with field 0x35: {maskok} rebuilt the mask ({:.1}%), {got} yielded a record ==",
        maskok as f64 / n.max(1) as f64 * 100.0);
    let mut l: Vec<_> = leftover.into_iter().collect();
    l.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  BITS AVAILABLE AT FIELD 0x35 (999 = never reached): {:?}", &l[..l.len().min(10)]);
    let mut s: Vec<_> = stopped.into_iter().collect();
    s.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, c) in s.iter().take(18) { println!("   {c:>6}  {k}"); }
}

/// Audit every ride the detector claims: how far the pair travelled HORIZONTALLY together, and
/// how much of that travel was level. A co-fall scores zero horizontal travel; a real drive does
/// not. Run after any change to the detector.
#[test]
fn probe_ride_audit() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let want = ["28AC2CC7_680042F8", "4832BB46_6774B938", "53C14E9D_6774AE16", "67FF2AFF"];
    for p in films.iter() {
        let n = name(p);
        if !want.iter().any(|w| n.contains(w)) {
            continue;
        }
        let Some((_f, idx)) = indexed(p) else { continue };
        let rides = crate::film_anim::accepted_rides(&idx);
        println!("\n== {n}: {} rides ==", rides.len());
        let mut flat_total = 0.0f32;
        for (i, r) in rides.iter().enumerate() {
            let v = idx.tracks.iter().find(|x| x.net_id == r.vehicle_net_id).unwrap();
            let (mut flat, mut drop) = (0.0f32, 0.0f32);
            let mut t = r.from_tick;
            while t + 6 <= r.to_tick {
                if let (Some(a), Some(b)) = (v.sample(t), v.sample(t + 6)) {
                    flat += (b[0] - a[0]).hypot(b[1] - a[1]);
                    drop += (b[2] - a[2]).abs();
                }
                t += 6;
            }
            flat_total += flat;
            if i < 12 {
                println!(
                    "   {:>6.1}s  body {:#010x} in vehicle {:#010x}  horizontal {:>7.1} wu   vertical {:>6.1} wu   from {}",
                    r.seconds(), r.biped_net_id, r.vehicle_net_id, flat, drop,
                    crate::film_play::fmt_time(r.from_tick as f32 / 60.0)
                );
            }
        }
        println!("   total horizontal travel while ridden: {flat_total:.0} wu");
    }
}

/// WHY are so many vehicles never drawn? The caption says "no position ever decoded", which reads
/// like a decode failure. Find out whether it is one.
#[test]
fn probe_undrawn_vehicles() {
    let films = films();
    if films.is_empty() {
        return;
    }
    for p in films.iter().take(8) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let (mut with, mut without) = (0usize, 0usize);
        let mut life: Vec<f32> = Vec::new();
        let mut masks: std::collections::BTreeMap<u64, usize> = Default::default();
        let mut msgs: Vec<usize> = Vec::new();
        let mut destroyed = 0usize;
        // count messages per entity in a second pass
        let params = idx.params;
        let mut per_entity: std::collections::HashMap<u32, (usize, u64)> = Default::default();
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 15 {
                    continue;
                }
                let v = per_entity.entry(e.net_id).or_insert((0, 0));
                v.0 += 1;
                v.1 |= e.dirty_mask;
            }
        }
        for tr in idx.tracks.iter().filter(|t| t.entity_type == 15) {
            if tr.positions.is_empty() {
                without += 1;
                let (n, m) = per_entity.get(&tr.net_id).copied().unwrap_or((0, 0));
                msgs.push(n);
                *masks.entry(m).or_insert(0) += 1;
                let end = tr.destroyed_tick.unwrap_or(idx.ticks);
                life.push((end.saturating_sub(tr.first_tick)) as f32 / 60.0);
                if tr.destroyed_tick.is_some() {
                    destroyed += 1;
                }
            } else {
                with += 1;
            }
        }
        if without == 0 {
            continue;
        }
        life.sort_by(|a, b| a.partial_cmp(b).unwrap());
        msgs.sort_unstable();
        let q = |v: &Vec<f32>, f: f32| v[((v.len() - 1) as f32 * f) as usize];
        println!(
            "\n== {} : {with} vehicles with a position, {without} without ==",
            name(p).chars().take(40).collect::<String>()
        );
        println!(
            "   lifetime of the positionless (s): p05 {:.2}  p50 {:.2}  p95 {:.2}   |  {destroyed}/{without} were destroyed",
            q(&life, 0.05), q(&life, 0.50), q(&life, 0.95)
        );
        println!("   messages each: p05 {}  p50 {}  p95 {}", msgs[msgs.len() / 20], msgs[msgs.len() / 2], msgs[msgs.len() * 19 / 20]);
        let mut mv: Vec<_> = masks.into_iter().collect();
        mv.sort_by(|a, b| b.1.cmp(&a.1));
        for (m, c) in mv.iter().take(5) {
            let bits: Vec<String> = (0..64).filter(|b| m & (1u64 << b) != 0).map(|b| format!("{b:#x}")).collect();
            println!("   {c:>4} entities, union of every dirty mask they ever sent = [{}]", bits.join(","));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// probe 15 — how far does a locomotion clip actually TRAVEL?
// ---------------------------------------------------------------------------------------------

/// One global "frames per world unit" cannot be right for every gait: `move_front_slow`,
/// `move_front_fast`, `move_back`, `move_left/right` and `sprint:*` each have their own stride
/// and frame count, so a single constant makes the feet slide on some and over-step on others.
///
/// The right number is each clip's OWN travel per frame. This measures whether that is
/// recoverable from the animation data at all — i.e. whether a Reach locomotion clip translates
/// its root node, or is authored in place with the movement held somewhere this reader does not
/// decode.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app probe_clip_root_motion -- --ignored --nocapture --test-threads=1"]
fn probe_clip_root_motion() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    let Some((tag, _path)) = crate::reach_actor::ActorScene::list_bipeds(&native, cache)
        .into_iter()
        .find(|(_, n)| n.contains("characters\\spartans\\spartans"))
    else {
        return;
    };
    let Some(def) = crate::reach_actor::ActorDef::read(&native, cache, tag) else { return };
    let Some(g) = def.graph.as_ref() else { return };
    println!("clip                                 frames  root dx,dy over the clip   |xy|   per frame");
    for name in [
        "combat:rifle:idle",
        "combat:rifle:move_front_slow",
        "combat:rifle:move_front_fast",
        "combat:rifle:move_back",
        "combat:rifle:move_back_slow",
        "combat:rifle:move_left",
        "combat:rifle:move_right",
        "combat:rifle:turn_left",
        "combat:rifle:turn_right",
        "sprint:rifle:move_front_fast",
        "crouch:rifle:move_front",
    ] {
        let Some(i) = g.find(name) else {
            println!("{name:<36}  <not in the graph>");
            continue;
        };
        let Some(data) = g.load(&native, cache, i) else {
            println!("{name:<36}  <resource load failed>");
            continue;
        };
        let n = data.frame_count.max(1) as usize;
        let ty = g.anims[i].anim_type;
        let at = |f: f32| -> Option<[f32; 3]> {
            let ch = crate::reach_jmad::sample(&data, ty, f);
            ch.first().and_then(|c| c.trans).map(|t| [t.x, t.y, t.z])
        };
        match (at(0.0), at((n - 1) as f32)) {
            (Some(a), Some(b)) => {
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let d = dx.hypot(dy);
                println!(
                    "{name:<36} {n:>6}  ({dx:>7.3}, {dy:>7.3})  {d:>7.3}  {:>8.4}",
                    d / n.max(1) as f32
                );
            }
            _ => println!("{name:<36} {n:>6}  <root node has no translation channel>"),
        }
    }
}

/// Every Reach locomotion clip is authored IN PLACE — `probe_clip_root_motion` measures zero root
/// translation on all eleven — so a clip's stride length is not recoverable from the animation
/// data this reader decodes. What IS recoverable is the speed each gait is actually performed at
/// in the recordings, and that is the number the cycle has to be driven by: a clip authored to
/// loop once per stride at 30 fps covers `v / 30` world units per frame when the body is moving
/// at that gait's own speed `v`.
///
/// So this measures the real speed distribution PER GAIT across the corpus.
#[test]
fn probe_speed_per_gait() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let mut per: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        used += 1;
        if used > 8 {
            break;
        }
        for tr in idx.tracks.iter().filter(|t| t.entity_type == 11 && t.facings.len() > 10) {
            // Sample at the track's own position keys, which is where the derivative is honest.
            for &(tick, _) in tr.positions.iter().step_by(3) {
                let st = crate::film_anim::derive(tr, tick, None);
                if st.provenance != crate::film_anim::Provenance::Derived {
                    continue;
                }
                let key = format!("{:?}/{:?}", st.stance, st.gait);
                let v = per.entry(key).or_default();
                if v.len() < 400_000 {
                    v.push(st.ground_speed);
                }
            }
        }
    }
    println!("\n== ground speed per derived gait, {used} films ==");
    println!("stance/gait                       samples     p25     p50     p75     p90   -> frames/wu at 30fps");
    for (k, mut v) in per {
        if v.len() < 50 {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f32| v[((v.len() - 1) as f32 * f) as usize];
        let med = q(0.50).max(0.01);
        println!(
            "{k:<32} {:>8}  {:>6.2}  {:>6.2}  {:>6.2}  {:>6.2}   {:>8.2}",
            v.len(),
            q(0.25),
            med,
            q(0.75),
            q(0.90),
            30.0 / med
        );
    }
}

// ---------------------------------------------------------------------------------------------
// probe 16 — do paired-melee detections actually concentrate in the assassination clip?
// ---------------------------------------------------------------------------------------------

/// The validation the detector lives or dies by: a clip whose gametype is literally
/// "Assassination." should yield many pairs, and ordinary Slayer clips very few. If the contrast
/// is weak, the detector is finding close-quarters contact rather than assassinations and nothing
/// should be drawn.
#[test]
fn probe_sync_pair_contrast() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                 gametype                 mins  bodies  pairs  per min");
    let mut rows: Vec<(bool, f64, String)> = Vec::new();
    for p in films.iter().take(16) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let pairs = crate::film_anim::find_sync_pairs(&idx);
        let mins = idx.ticks as f64 / 3600.0;
        let gt = f.header.game_variant.title.clone();
        let is_assassination = gt.to_lowercase().contains("assassin");
        let per_min = pairs.len() as f64 / mins.max(0.01);
        println!(
            "{:<36} {:<24} {:>5.1}  {:>6}  {:>5}  {:>7.2}",
            name(p).chars().take(36).collect::<String>(),
            gt.chars().take(24).collect::<String>(),
            mins,
            idx.owner_group_count(),
            pairs.len(),
            per_min
        );
        for q in pairs.iter().take(3) {
            println!(
                "      {:>5.1}s  attacker {:#010x} -> victim {:#010x}  at {}",
                q.seconds(),
                q.attacker_net_id,
                q.victim_net_id,
                crate::film_play::fmt_time(q.from_tick as f32 / 60.0)
            );
        }
        rows.push((is_assassination, per_min, name(p)));
    }
    let a: Vec<f64> = rows.iter().filter(|r| r.0).map(|r| r.1).collect();
    let o: Vec<f64> = rows.iter().filter(|r| !r.0).map(|r| r.1).collect();
    let mean = |v: &Vec<f64>| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    println!(
        "\n== CONTRAST: assassination clips {:.2} pairs/min ({} films), everything else {:.2} ({} films) ==",
        mean(&a),
        a.len(),
        mean(&o),
        o.len()
    );
    if mean(&o) > 0.0 {
        println!("   ratio {:.1}x", mean(&a) / mean(&o));
    }
}

/// Why do the assassination-gametype clips yield no pairs? Either the detector's bounds are wrong
/// or those clips do not carry the positions it needs. This separates the two.
#[test]
fn probe_assassination_clip_density() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                 gametype              bodies  pos/body  closest approach of any two bodies");
    for p in films.iter().take(16) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let gt = f.header.game_variant.title.clone();
        let bip: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && !t.positions.is_empty()).collect();
        if bip.is_empty() {
            println!("{:<36} {:<22} {:>6}  (no bipeds with positions)", name(p).chars().take(36).collect::<String>(), gt.chars().take(22).collect::<String>(), 0);
            continue;
        }
        let total: usize = bip.iter().map(|t| t.positions.len()).sum();
        // closest sustained approach between any two different-owner bodies
        let mut best = f32::MAX;
        let mut best_run = 0u32;
        for (i, a) in bip.iter().enumerate() {
            for b in bip.iter().skip(i + 1) {
                if a.owner_raw.is_some() && a.owner_raw == b.owner_raw {
                    continue;
                }
                let lo = a.first_tick.max(b.first_tick);
                let hi = a.last_tick.min(b.last_tick);
                let (mut t, mut run, mut cur) = (lo, 0u32, 0u32);
                while t <= hi && t < lo + 200_000 {
                    if let (Some(x), Some(y)) = (a.sample(t), b.sample(t)) {
                        let d = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt();
                        best = best.min(d);
                        if d < 1.1 {
                            cur += 6;
                            run = run.max(cur);
                        } else {
                            cur = 0;
                        }
                    }
                    t += 6;
                }
                best_run = best_run.max(run);
            }
        }
        println!(
            "{:<36} {:<22} {:>6}  {:>8.0}  closest {:.2} wu, longest run inside 1.1 wu = {:.1}s",
            name(p).chars().take(36).collect::<String>(),
            gt.chars().take(22).collect::<String>(),
            bip.len(),
            total as f64 / bip.len() as f64,
            best,
            best_run as f32 / 60.0
        );
    }
}

/// WHICH gate rejects the candidate pairs? Counts survivors after each one, per film, so the
/// detector's bounds can be judged instead of guessed at.
#[test]
fn probe_sync_pair_gates() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                              gametype            runs  len-ok  drift-ok  death-ok   drift p50/p90   len p50");
    for p in films.iter().take(16) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let gt = f.header.game_variant.title.clone();
        let bip: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && t.positions.len() > 4).collect();
        let (mut runs, mut len_ok, mut drift_ok, mut death_ok) = (0usize, 0usize, 0usize, 0usize);
        let mut drifts: Vec<f32> = Vec::new();
        let mut lens: Vec<u32> = Vec::new();
        for (i, a) in bip.iter().enumerate() {
            for b in bip.iter().skip(i + 1) {
                if a.owner_raw.is_some() && a.owner_raw == b.owner_raw {
                    continue;
                }
                let lo = a.first_tick.max(b.first_tick);
                let hi = a.last_tick.min(b.last_tick);
                if hi <= lo + 72 {
                    continue;
                }
                let mut start: Option<u32> = None;
                let mut offs: Vec<[f32; 3]> = Vec::new();
                let mut t = lo;
                let mut flush = |s: u32, e: u32, offs: &Vec<[f32; 3]>,
                                 runs: &mut usize, len_ok: &mut usize, drift_ok: &mut usize, death_ok: &mut usize,
                                 drifts: &mut Vec<f32>, lens: &mut Vec<u32>| {
                    *runs += 1;
                    let len = e.saturating_sub(s);
                    lens.push(len);
                    if len < 72 || len > 300 || offs.len() < 4 {
                        return;
                    }
                    *len_ok += 1;
                    let n = offs.len() as f32;
                    let m = [
                        offs.iter().map(|o| o[0]).sum::<f32>() / n,
                        offs.iter().map(|o| o[1]).sum::<f32>() / n,
                        offs.iter().map(|o| o[2]).sum::<f32>() / n,
                    ];
                    let d = offs
                        .iter()
                        .map(|o| ((o[0] - m[0]).powi(2) + (o[1] - m[1]).powi(2) + (o[2] - m[2]).powi(2)).sqrt())
                        .fold(0.0f32, f32::max);
                    drifts.push(d);
                    if d > 0.30 {
                        return;
                    }
                    *drift_ok += 1;
                    let ends = |t: &film_decode::EntityTrack| {
                        let end = t.destroyed_tick.unwrap_or(t.last_tick);
                        end >= s && end <= e + 60 && end + 60 < idx.ticks
                    };
                    if ends(a) ^ ends(b) {
                        *death_ok += 1;
                    }
                };
                while t <= hi {
                    let near = match (a.sample(t), b.sample(t)) {
                        (Some(x), Some(y)) => {
                            let o = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
                            if (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() <= 1.1 {
                                offs.push(o);
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    };
                    if near {
                        start.get_or_insert(t);
                    } else if let Some(s) = start.take() {
                        flush(s, t.saturating_sub(6), &offs, &mut runs, &mut len_ok, &mut drift_ok, &mut death_ok, &mut drifts, &mut lens);
                        offs.clear();
                    }
                    t += 6;
                }
                if let Some(s) = start {
                    flush(s, hi, &offs, &mut runs, &mut len_ok, &mut drift_ok, &mut death_ok, &mut drifts, &mut lens);
                }
            }
        }
        drifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lens.sort_unstable();
        let q = |v: &Vec<f32>, f: f32| if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f32 * f) as usize] };
        println!(
            "{:<33} {:<19} {runs:>5}  {len_ok:>6}  {drift_ok:>8}  {death_ok:>8}   {:>5.2}/{:>5.2}      {:>5}",
            name(p).chars().take(33).collect::<String>(),
            gt.chars().take(19).collect::<String>(),
            q(&drifts, 0.50),
            q(&drifts, 0.90),
            lens.get(lens.len() / 2).copied().unwrap_or(0)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// probe 17 — the weapon in each body's hands
// ---------------------------------------------------------------------------------------------

/// Does the `'weap'` reference width solve, and do the tags it yields resolve to real weapons?
/// Reports the winner AND the runner-up margin, so the answer can be judged.
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app probe_held_weapons -- --ignored --nocapture --test-threads=1"]
fn probe_held_weapons() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    let weap = native.tags_by_group(cache, "weap");
    println!("forge_halo has {} 'weap' tags -> bits_needed(n+1) = {}", weap.len(), film_decode::ceil_log2(weap.len() as u32 + 1));
    let mut used = 0;
    for p in films.iter() {
        if !name(p).contains("forge_hal") {
            continue;
        }
        let Ok(f) = film::open(p) else { continue };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            continue;
        }
        let (base, _) = film_decode::solve_params(&f.bytes, &f.data);
        let (_fr, w) = film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
        let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
        // the sweep, scored by the mask oracle
        const BIT: u64 = 1 << 0x2A;
        let mut scores = [0usize; 10];
        let mut seen = 0usize;
        for rec in f.data.records.iter() {
            if seen >= 400 {
                break;
            }
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            for e in &t.entities {
                if e.entity_type != 11 || !e.is_update || e.dirty_mask & BIT == 0 {
                    continue;
                }
                seen += 1;
                for b in 4..=9u32 {
                    if let Some(u) = crate::film_anim::unit::decode(e, &params, b) {
                        if u.trustworthy(e.dirty_mask) {
                            scores[b as usize] += 1;
                        }
                    }
                }
            }
        }
        let mut rank: Vec<(usize, u32)> = (4..=9u32).map(|b| (scores[b as usize], b)).collect();
        rank.sort_by(|a, b| b.0.cmp(&a.0));
        println!(
            "\n{}: {seen} messages with the field; mask-oracle widths {:?} (margin {}) -- the oracle \
             cannot see this field, so the value itself has to decide",
            name(p).chars().take(40).collect::<String>(),
            &rank[..3],
            rank[0].0 as i64 - rank[1].0 as i64
        );
        // A TYPED TagReference plausibly indexes the GROUP's own instance list, not the global
        // tag table. Try both readings at every width and score by how many land on a real weapon.
        let names: Vec<String> = weap.iter().map(|&t| native.tag_name(cache, t).unwrap_or_default()).collect();
        for b in 4..=9u32 {
            let mut vals: Vec<u32> = Vec::new();
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type != 11 || !e.is_update || e.dirty_mask & BIT == 0 {
                        continue;
                    }
                    if let Some(u) = crate::film_anim::unit::decode(e, &params, b) {
                        if let Some(w) = u.weapon_tag {
                            vals.push(w);
                        }
                    }
                }
                if vals.len() > 2000 {
                    break;
                }
            }
            if vals.is_empty() {
                continue;
            }
            let in_range = vals.iter().filter(|&&v| (v as usize) < names.len()).count();
            // 0 dominates every width, which is what "no reference" looks like. Judge the width
            // on the NON-ZERO values: a hand weapon is plausible, a vehicle turret is not.
            let nz: Vec<u32> = vals.iter().copied().filter(|&v| v != 0).collect();
            if !nz.is_empty() {
                let mut hz: std::collections::BTreeMap<String, usize> = Default::default();
                for &v in &nz {
                    let n = names
                        .get(v as usize)
                        .map(|n| n.rsplit('\\').next().unwrap_or(n).to_string())
                        .unwrap_or_else(|| format!("<oob {v}>"));
                    *hz.entry(n).or_insert(0) += 1;
                }
                let mut t: Vec<_> = hz.into_iter().collect();
                t.sort_by(|a, b| b.1.cmp(&a.1));
                let hand = ["assault_rifle", "dmr", "magnum", "shotgun", "sniper", "rocket", "needle",
                            "plasma_pistol", "plasma_repeater", "plasma_rifle", "energy_sword", "gravity_hammer",
                            "grenade_launcher", "spartan_laser", "concussion", "focus_rifle", "fuel_rod"];
                let handy: usize = t.iter().filter(|(k, _)| hand.iter().any(|h| k.contains(h))).map(|(_, c)| *c).sum();
                println!(
                    "      -> {} non-zero, {} ({:.0}%) are HAND weapons: {}",
                    nz.len(),
                    handy,
                    100.0 * handy as f64 / nz.len() as f64,
                    t.iter().take(7).map(|(k, c)| format!("{k}x{c}")).collect::<Vec<_>>().join(" ")
                );
            }
            let mut h: std::collections::BTreeMap<u32, usize> = Default::default();
            for &v in &vals {
                *h.entry(v).or_insert(0) += 1;
            }
            let mut top: Vec<_> = h.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            let show: Vec<String> = top
                .iter()
                .take(5)
                .map(|(v, c)| {
                    let n = names
                        .get(**v as usize)
                        .map(|n| n.rsplit('\\').next().unwrap_or(n).to_string())
                        .unwrap_or_else(|| "<out of range>".into());
                    format!("{n}x{c}")
                })
                .collect();
            println!(
                "   {b}b: {:>5} values, {:>3}% index a 'weap', {:>3} distinct | {}",
                vals.len(),
                100 * in_range / vals.len(),
                h.len(),
                show.join(" ")
            );
        }
        used += 1;
        if used >= 3 {
            break;
        }
    }
}

/// How many bodies ever get a weapon? The field is rare in the stream, so the question is not
/// "does it decode" but "does it cover enough bodies to be worth drawing".
#[test]
fn probe_weapon_coverage() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                   bodies  with a weapon   reports  distinct weapons");
    for p in films.iter().take(12) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let bodies: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && !t.positions.is_empty()).collect();
        let armed = bodies.iter().filter(|t| !t.weapons.is_empty()).count();
        let reports: usize = bodies.iter().map(|t| t.weapons.len()).sum();
        let distinct: std::collections::BTreeSet<u32> =
            bodies.iter().flat_map(|t| t.weapons.iter().map(|(_, w)| *w)).collect();
        println!(
            "{:<38} {:>6}  {:>6} ({:>3.0}%)  {:>8}  {:>4}",
            name(p).chars().take(38).collect::<String>(),
            bodies.len(),
            armed,
            100.0 * armed as f64 / bodies.len().max(1) as f64,
            reports,
            distinct.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// probe 18 — does first person actually follow the player's look?
// ---------------------------------------------------------------------------------------------

/// The owner reports first person "doesn't follow the player's look vector at all". Two very
/// different faults would look the same: the aim DECODE being wrong, or the aim simply not being
/// reachable for a player the recording never named (an unnamed player has no configuration slot,
/// and `aim_at` is keyed on the slot).
///
/// This separates them. If a NAMED player's aim yaw moves around over the clip, the decode and
/// the lookup are fine and the fault is the unnamed path.
#[test]
fn probe_first_person_aim() {
    let films = films();
    if films.is_empty() {
        return;
    }
    for p in films.iter() {
        let n = name(p);
        if !(n.contains("21897D8A") || n.contains("4ED6DDC1_67FF2AFF")) {
            continue;
        }
        let Some((_f, idx)) = indexed(p) else { continue };
        println!(
            "\n== {n} : {} owner groups, {} named ==",
            idx.owner_group_count(),
            idx.owner_groups().iter().filter(|&&o| idx.slot_of_owner(o).is_some()).count()
        );
        println!("slot  samples   yaw span   yaw sd   ticks yaw CHANGED   pitch span (deg)   body-facing sd");
        for slot in 0..16u8 {
            let v = idx.player_aims.get(slot as usize).cloned().unwrap_or_default();
            if v.len() < 50 {
                continue;
            }
            let yaws: Vec<f32> = v.iter().map(|(_, a)| a.yaw()).collect();
            let pitches: Vec<f32> = v.iter().map(|(_, a)| a.pitch().to_degrees()).collect();
            // circular spread: how much of the circle is visited
            let mut changed = 0usize;
            for w in v.windows(2) {
                if w[0].1.yaw_q != w[1].1.yaw_q {
                    changed += 1;
                }
            }
            let mean = yaws.iter().sum::<f32>() / yaws.len() as f32;
            let sd = (yaws.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / yaws.len() as f32).sqrt();
            let (ymin, ymax) = yaws.iter().fold((f32::MAX, f32::MIN), |a, &y| (a.0.min(y), a.1.max(y)));
            let (pmin, pmax) = pitches.iter().fold((f32::MAX, f32::MIN), |a, &y| (a.0.min(y), a.1.max(y)));
            // the body facing of whatever body this slot owned, for comparison
            let owner = idx.owner_groups().into_iter().find(|&o| idx.slot_of_owner(o) == Some(slot));
            let bsd = owner
                .and_then(|o| {
                    let f: Vec<f32> = idx
                        .tracks
                        .iter()
                        .filter(|t| t.entity_type == 11 && t.owner_raw == Some(o))
                        .flat_map(|t| t.facings.iter().map(|(_, y)| *y))
                        .collect();
                    (f.len() > 10).then(|| {
                        let m = f.iter().sum::<f32>() / f.len() as f32;
                        (f.iter().map(|y| (y - m).powi(2)).sum::<f32>() / f.len() as f32).sqrt()
                    })
                })
                .unwrap_or(f32::NAN);
            println!(
                "{slot:>4}  {:>7}  {:>9.2}  {:>7.2}  {:>9} ({:>3.0}%)   {:>6.0} .. {:<6.0}  {:>8.2}",
                v.len(),
                ymax - ymin,
                sd,
                changed,
                100.0 * changed as f64 / (v.len() - 1).max(1) as f64,
                pmin,
                pmax,
                bsd
            );
        }
    }
}

/// If only the recorder has a real aim, first person for everyone else has to fall back to the
/// body facing. So: how well covered IS the body facing?
#[test]
fn probe_body_facing_coverage() {
    let films = films();
    if films.is_empty() {
        return;
    }
    println!("film                                  bodies  with facings  median facings  median gap (ticks)  alive ticks");
    for p in films.iter().take(10) {
        let Some((f, idx)) = indexed(p) else { continue };
        if idx.entity_messages as f64 / f.data.records.len().max(1) as f64 > 40.0 {
            continue;
        }
        let bodies: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && !t.positions.is_empty()).collect();
        if bodies.is_empty() {
            continue;
        }
        let with = bodies.iter().filter(|t| t.facings.len() > 5).count();
        let mut counts: Vec<usize> = bodies.iter().map(|t| t.facings.len()).collect();
        counts.sort_unstable();
        // median gap between consecutive facing samples, over bodies that have them
        let mut gaps: Vec<u32> = Vec::new();
        let mut alive: Vec<u32> = Vec::new();
        for t in bodies.iter().filter(|t| t.facings.len() > 5) {
            for w in t.facings.windows(2) {
                gaps.push(w[1].0.saturating_sub(w[0].0));
            }
            alive.push(t.last_tick.saturating_sub(t.first_tick));
        }
        gaps.sort_unstable();
        alive.sort_unstable();
        println!(
            "{:<38} {:>6}  {:>6} ({:>3.0}%)  {:>14}  {:>18}  {:>11}",
            name(p).chars().take(38).collect::<String>(),
            bodies.len(),
            with,
            100.0 * with as f64 / bodies.len() as f64,
            counts[counts.len() / 2],
            gaps.get(gaps.len() / 2).copied().unwrap_or(0),
            alive.get(alive.len() / 2).copied().unwrap_or(0),
        );
    }
}

/// Does the rewritten discovery actually find the Linux install's films, including the one the
/// game is recording right now?
#[test]
fn probe_film_discovery() {
    let dirs = film::film_search_dirs();
    println!("== {} film folder(s) discovered ==", dirs.len());
    for d in &dirs {
        let n = std::fs::read_dir(d)
            .map(|rd| rd.flatten().filter(|e| film::is_film_file(&e.path())).count())
            .unwrap_or(0);
        // The folder, with the user component elided: a Proton prefix path embeds the account name.
        let shown: Vec<String> = d
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let safe = shown
            .iter()
            .enumerate()
            .map(|(i, s)| if i > 0 && shown.get(i - 1).map_or(false, |p| p.eq_ignore_ascii_case("users")) { "<user>".to_string() } else { s.clone() })
            .collect::<Vec<_>>()
            .join("/");
        println!("  {n:>4} film(s)  {safe}");
    }
    let films = film::find_films();
    println!("\n== {} film file(s) ==", films.len());
    let mut by_ext: std::collections::BTreeMap<String, usize> = Default::default();
    for f in &films {
        *by_ext.entry(f.extension().unwrap_or_default().to_string_lossy().to_lowercase()).or_insert(0) += 1;
    }
    println!("  by extension: {by_ext:?}");
    for f in films.iter().take(5) {
        let len = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        println!("  {:>10} bytes  {}", len, name(f));
    }
}

// ---------------------------------------------------------------------------------------------
// probe 19 — the instanced-armour tiling
// ---------------------------------------------------------------------------------------------

/// Why do SOME armour pieces stretch between two bones?
///
/// The per-instance vertex copy exists (`scene.rs`, `#instmesh`): each placement gets its own
/// copy of the vertices its index range touches, rebound to that placement's node. The owner's
/// capture shows it is not covering every case — a knee piece floats free with a *tether* of
/// stretched triangles back to the leg, which means ONE mesh's vertices are split between a
/// correct bone and a wrong one, not a whole piece on the wrong node.
///
/// Four things could do that, and this measures all four instead of reasoning about them:
/// placements whose node is negative (skipped, so they keep the mesh's own rigid bone);
/// placements whose node is outside the skeleton; index ranges that OVERLAP (the second copy
/// wins); and indices in the instance mesh covered by NO placement (never rebound, so they keep
/// the belt).
#[test]
#[ignore = "opens a second copy of the native library; run it alone: cargo test --profile quick -p hms-app probe_instanced_armour -- --ignored --nocapture --test-threads=1"]
fn probe_instanced_armour_tiling() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    for want in ["characters\\spartans\\spartans", "characters\\elite\\elite"] {
        let Some(tag) = native
            .tags_by_group(cache, "bipd")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).unwrap_or_default().contains(want))
        else {
            println!("\n{want}: not in this map");
            continue;
        };
        let mode = native.resolve_mode(cache, tag);
        let Some(def) = crate::reach_actor::ActorDef::read(&native, cache, tag) else { continue };
        let nodes = def.model.skel.len();
        let Some((im, insts)) = crate::scene::model_instances(&native, cache, mode) else {
            println!("\n{want}: no instanced geometry");
            continue;
        };
        println!("\n== {want} ==");
        println!("   instance mesh {im}, {} placements, skeleton {nodes} nodes", insts.len());
        let neg = insts.iter().filter(|i| i.node < 0).count();
        let oob = insts.iter().filter(|i| i.node >= 0 && i.node as usize >= nodes).count();
        println!("   placements with node < 0 (SKIPPED, keep the mesh's own bone): {neg}");
        println!("   placements with node >= skeleton length: {oob}");
        // Do the ranges tile without gap or overlap?
        let mut r: Vec<(u32, u32)> = insts.iter().map(|i| (i.index_start, i.index_count)).collect();
        r.sort();
        let (mut gaps, mut overlaps, mut gap_idx, mut ov_idx) = (0usize, 0usize, 0u32, 0u32);
        let mut cursor = r.first().map(|x| x.0).unwrap_or(0);
        if cursor != 0 {
            gaps += 1;
            gap_idx += cursor;
        }
        for (s, c) in &r {
            if *s > cursor {
                gaps += 1;
                gap_idx += s - cursor;
            } else if *s < cursor {
                overlaps += 1;
                ov_idx += cursor - s;
            }
            cursor = cursor.max(s + c);
        }
        println!("   ranges: {gaps} gap(s) totalling {gap_idx} indices, {overlaps} overlap(s) totalling {ov_idx} indices");
        println!("   covered up to index {cursor}");
        // #instmesh How bad was the OLD behaviour? An index no placement owns was emitted
        // carrying the mesh's own rigid node (the belt), while the triangles on either side of
        // it were rebound to whatever attachment owns them. So the stray triangle's corners sat
        // in the belt's frame while its neighbours sat in a helmet's or a knee's: when the
        // skeleton moves, the triangle is stretched by the distance between those two nodes.
        // That distance is the spike/tether length, and it is measurable from the bind pose.
        let rigid = def.model.mesh_rigid_node.get(im).copied().unwrap_or(0) as usize;
        let bind_pos = |n: usize| -> Option<glam::Vec3> {
            let nd = def.model.skel.nodes.get(n)?;
            if !nd.inverse_is_frame() { return None; }
            Some(nd.inverse_bind_authored().inverse().w_axis.truncate())
        };
        let rigid_name = def.model.node_names.get(rigid).cloned().unwrap_or_default();
        let rigid_p = bind_pos(rigid);
        println!("   mesh rigid node = {rigid} '{rigid_name}' (the bone a stray triangle KEPT)");
        // Walk the gaps again, naming the placement on each side.
        let mut pairs: Vec<(u32, u32, usize, usize)> = Vec::new(); // (gap start, len, node before, node after)
        let mut ordered: Vec<&crate::scene::ModelInstance> = insts.iter().collect();
        ordered.sort_by_key(|i| i.index_start);
        let mut cur = 0u32;
        let mut prev_node = usize::MAX;
        for inst in &ordered {
            if inst.index_start > cur {
                pairs.push((cur, inst.index_start - cur, prev_node, inst.node.max(0) as usize));
            }
            cur = cur.max(inst.index_start + inst.index_count);
            if inst.node >= 0 { prev_node = inst.node as usize; }
        }
        let mut worst = 0.0f32;
        let mut dists: Vec<f32> = Vec::new();
        for (start, len, a, b) in &pairs {
            let near = if *b != usize::MAX { *b } else { *a };
            let d = match (rigid_p, bind_pos(near)) {
                (Some(r), Some(n)) => (r - n).length(),
                _ => f32::NAN,
            };
            if d.is_finite() {
                dists.push(d);
                worst = worst.max(d);
            }
            if dists.len() <= 6 {
                println!(
                    "     gap @{start} ({len} idx) sits between '{}' and '{}' — {:.3} wu from '{rigid_name}'",
                    def.model.node_names.get(*a).cloned().unwrap_or_else(|| "<none>".into()),
                    def.model.node_names.get(*b).cloned().unwrap_or_else(|| "<none>".into()),
                    d
                );
            }
        }
        dists.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = dists.get(dists.len() / 2).copied().unwrap_or(f32::NAN);
        println!(
            "   {} unowned run(s): bind-pose distance from the kept bone to the neighbouring attachment median {:.3} wu, worst {:.3} wu",
            pairs.len(), med, worst
        );
    }
}

/// #death How usable is the destroy record as a DEATH, in practice?
///
/// The corpse pass draws a body for a few seconds after its track is destroyed. That is only
/// worth doing if (a) destroys actually happen on a biped often enough to be the thing the owner
/// is watching for, and (b) the destroy tick is close enough to the last reported position that
/// the body does not vanish and reappear somewhere else.
///
/// Prints, per film: biped tracks, how many carry a destroy, the gap `destroyed_tick -
/// last_tick` in ticks, and how many deaths land within 6 s of another one (which is what the
/// actor pool has to cover at once).
#[test]
fn probe_death_census() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let (mut tot_b, mut tot_d, mut tot_gap_bad) = (0usize, 0usize, 0usize);
    let mut gaps: Vec<i64> = Vec::new();
    let mut frozen: Vec<i64> = Vec::new();
    for p in films.iter().take(12) {
        let n = name(p);
        let Some((_f, idx)) = indexed(p) else { continue };
        let bipeds: Vec<_> = idx.tracks.iter().filter(|t| t.entity_type == 11).collect();
        let dead: Vec<_> = bipeds.iter().filter(|t| t.destroyed_tick.is_some()).collect();
        let mut ticks: Vec<u32> = Vec::new();
        for t in &dead {
            let d = t.destroyed_tick.unwrap();
            let g = d as i64 - t.last_tick as i64;
            gaps.push(g);
            frozen.push(t.last_tick as i64 - t.positions.last().map(|x| x.0 as i64).unwrap_or(t.last_tick as i64));
            if g.abs() > 60 {
                tot_gap_bad += 1;
            }
            ticks.push(d);
        }
        ticks.sort();
        // how many deaths share a 6-second window with at least one other
        let mut together = 0usize;
        for (i, &a) in ticks.iter().enumerate() {
            if ticks.iter().enumerate().any(|(j, &b)| j != i && b.abs_diff(a) <= 360) {
                together += 1;
            }
        }
        let secs = idx.ticks as f32 / 60.0;
        // How many corpses would be on screen at once if each lingered 5 s from its track's end?
        let mut peak = 0usize;
        let mut t = idx.first_tick;
        while t < idx.first_tick + idx.ticks {
            let c = dead
                .iter()
                .filter(|x| t > x.last_tick && t <= x.last_tick + 300)
                .count();
            peak = peak.max(c);
            t += 30;
        }
        println!(
            "{n}: {} biped track(s), {} destroyed ({:.1}/min), {together} within 6 s of another, peak {peak} corpse(s) at once over a 5 s linger",
            bipeds.len(),
            dead.len(),
            dead.len() as f32 / (secs / 60.0).max(1e-3)
        );
        tot_b += bipeds.len();
        tot_d += dead.len();
    }
    gaps.sort();
    frozen.sort();
    let med = gaps.get(gaps.len() / 2).copied().unwrap_or(0);
    println!(
        "\nlast POSITION to last message: median {} ticks, p95 {} ticks (how long a destroyed body already stands frozen)",
        frozen.get(frozen.len() / 2).copied().unwrap_or(0),
        frozen.get(frozen.len() * 95 / 100).copied().unwrap_or(0)
    );
    println!(
        "\nTOTAL {tot_d} destroy(s) over {tot_b} biped track(s); destroyed_tick - last_tick median {med} ticks, \
         min {:?}, max {:?}, {tot_gap_bad} more than 1 s apart",
        gaps.first(),
        gaps.last()
    );
}

/// #death WHERE is the moment of death? `last_tick` or `destroyed_tick`?
///
/// `probe_death_census` found the two are a median **974 ticks (16 s)** apart, so they cannot
/// both be it. The oracle is the RESPAWN: a player's next life starts a fixed delay after they
/// died (Reach's default respawn is 5 s, and whatever a given playlist uses is constant within
/// one film). So for consecutive lives of the same owner group, `next.first_tick - death` must
/// be nearly the SAME number every time. Whichever candidate gives the tighter spread is the
/// death instant; the other is something else.
///
/// Reported as the inter-quartile range of that delay, per film, for both candidates, so the
/// winner comes with a margin.
#[test]
fn probe_death_instant_oracle() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let q = |v: &mut Vec<i64>, f: f32| -> i64 {
        if v.is_empty() {
            return 0;
        }
        v.sort();
        v[((v.len() - 1) as f32 * f) as usize]
    };
    let (mut win_last, mut win_dest, mut win_pos) = (0usize, 0usize, 0usize);
    for p in films.iter().take(12) {
        let n = name(p);
        let Some((_f, idx)) = indexed(p) else { continue };
        // Bipeds grouped by owner, in creation order.
        let mut by_owner: std::collections::HashMap<u8, Vec<&crate::film_decode::EntityTrack>> = Default::default();
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            if let Some(o) = t.owner_raw {
                by_owner.entry(o).or_default().push(t);
            }
        }
        let (mut dl, mut dd, mut dp) = (Vec::new(), Vec::new(), Vec::new());
        for (_o, mut lives) in by_owner {
            lives.sort_by_key(|t| t.first_tick);
            for w in lives.windows(2) {
                let (a, b) = (w[0], w[1]);
                let Some(d) = a.destroyed_tick else { continue };
                // only consecutive lives that really are consecutive (no overlap)
                if b.first_tick <= a.first_tick {
                    continue;
                }
                dl.push(b.first_tick as i64 - a.last_tick as i64);
                dd.push(b.first_tick as i64 - d as i64);
                dp.push(b.first_tick as i64 - a.positions.last().map(|x| x.0 as i64).unwrap_or(a.last_tick as i64));
            }
        }
        if dl.len() < 20 {
            continue;
        }
        let (l25, l50, l75) = (q(&mut dl, 0.25), q(&mut dl, 0.5), q(&mut dl, 0.75));
        let (d25, d50, d75) = (q(&mut dd, 0.25), q(&mut dd, 0.5), q(&mut dd, 0.75));
        let (p25, p50, p75) = (q(&mut dp, 0.25), q(&mut dp, 0.5), q(&mut dp, 0.75));
        let (liqr, diqr, piqr) = (l75 - l25, d75 - d25, p75 - p25);
        if piqr <= liqr && piqr <= diqr {
            win_pos += 1;
        } else if liqr < diqr {
            win_last += 1;
        } else {
            win_dest += 1;
        }
        println!(
            "{n}: {} respawn pair(s)\n   next.first - LAST POSITION   median {p50:>7} IQR {piqr:>7} ticks ({:.2} s)\n   next.first - last_tick       median {l50:>7} IQR {liqr:>7} ticks ({:.2} s)\n   next.first - destroyed_tick  median {d50:>7} IQR {diqr:>7} ticks ({:.2} s)",
            dl.len(),
            p50 as f32 / 60.0,
            l50 as f32 / 60.0,
            d50 as f32 / 60.0
        );
    }
    println!("\ntighter spread: LAST POSITION on {win_pos} film(s), last_tick on {win_last}, destroyed_tick on {win_dest}");
}

/// #death What does the END of a biped's track look like — is there a ragdoll in it?
///
/// If the engine replicates the dying body as it falls, the last second of a destroyed biped's
/// track carries a characteristic DROP (standing eye-line to prone is roughly half a world unit)
/// that a body which merely stopped moving cannot have. Compares the final second of tracks that
/// were destroyed against the final second of tracks that were still alive when the film ended.
#[test]
fn probe_death_tail_motion() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let pct = |v: &mut Vec<f32>, f: f32| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * f) as usize]
    };
    let (mut dz_d, mut flat_d, mut dz_a, mut flat_a) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut quiet_d = 0usize;
    for p in films.iter().take(12) {
        let Some((_f, idx)) = indexed(p) else { continue };
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            if t.last_tick < t.first_tick + 60 {
                continue;
            }
            let (a, b) = (t.sample(t.last_tick - 60), t.sample(t.last_tick));
            let (Some(a), Some(b)) = (a, b) else { continue };
            let dz = b[2] - a[2];
            let flat = (b[0] - a[0]).hypot(b[1] - a[1]);
            if t.destroyed_tick.is_some() {
                dz_d.push(dz);
                flat_d.push(flat);
                // how many report NOTHING in their last second (settled well before the end)
                if t.positions.iter().rev().take_while(|(tt, _)| *tt + 60 >= t.last_tick).count() <= 1 {
                    quiet_d += 1;
                }
            } else {
                dz_a.push(dz);
                flat_a.push(flat);
            }
        }
    }
    println!(
        "destroyed bipeds (n={}): last second  dz p05 {:+.2} p50 {:+.2} p95 {:+.2} wu   flat p50 {:.2} p95 {:.2} wu   {quiet_d} silent",
        dz_d.len(), pct(&mut dz_d, 0.05), pct(&mut dz_d, 0.5), pct(&mut dz_d, 0.95),
        pct(&mut flat_d, 0.5), pct(&mut flat_d, 0.95)
    );
    println!(
        "survivors        (n={}): last second  dz p05 {:+.2} p50 {:+.2} p95 {:+.2} wu   flat p50 {:.2} p95 {:.2} wu",
        dz_a.len(), pct(&mut dz_a, 0.05), pct(&mut dz_a, 0.5), pct(&mut dz_a, 0.95),
        pct(&mut flat_a, 0.5), pct(&mut flat_a, 0.95)
    );
}

/// #death #assassination Round two: is a death CLOSE and FROM BEHIND?
///
/// The first attempt looked for two bodies held at a fixed offset for a couple of seconds and
/// failed — it found MORE pairs on ordinary clips than on a gametype built around assassinations
/// (`probe_sync_pair_contrast`). That hypothesis needed a body's position to be replicated densely
/// enough for "locked together" to be measurable, and it is not.
///
/// This one needs only ONE instant. Now that `death_tick` pins the moment a body died, ask: at
/// that instant, was another player's body within `NEAR` world units, and was it BEHIND the victim
/// — within `BACK_DEG` of the victim's own last reported facing, reversed? An assassination comes
/// from behind at contact range, so a clip whose gametype is an assassination gametype must show a
/// HIGHER share of such deaths. Same oracle as before; if it inverts again, it is dead again.
#[test]
fn probe_death_from_behind_contrast() {
    let films = films();
    if films.is_empty() {
        return;
    }
    const NEAR: f32 = 2.0;
    const BACK_DEG: f32 = 70.0;
    println!("{:<40} {:<24} {:>7} {:>8} {:>8} {:>8}", "film", "gametype", "deaths", "near", "behind", "share");
    let (mut a_d, mut a_b, mut o_d, mut o_b) = (0usize, 0usize, 0usize, 0usize);
    for p in films.iter().take(16) {
        let Some((f, idx)) = indexed(p) else { continue };
        let gt = f.header.game_variant.title.clone();
        let is_assn = gt.to_lowercase().contains("assassin");
        let bip: Vec<&crate::film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == 11 && !t.positions.is_empty()).collect();
        let (mut deaths, mut near, mut behind) = (0usize, 0usize, 0usize);
        for v in &bip {
            let Some(d) = crate::film_anim::death_tick(v) else { continue };
            let Some(vp) = v.sample(d) else { continue };
            deaths += 1;
            // the victim's own facing at the moment it died
            let face = v.facing_at(d);
            let mut best: Option<(f32, f32)> = None; // (distance, |bearing from the victim's back|)
            for k in &bip {
                if k.net_id == v.net_id {
                    continue;
                }
                if k.owner_raw.is_some() && k.owner_raw == v.owner_raw {
                    continue;
                }
                if !k.alive_at(d) || crate::film_anim::death_tick(k).is_some_and(|kd| kd < d) {
                    continue;
                }
                let Some(kp) = k.sample(d) else { continue };
                let dist = ((kp[0] - vp[0]).powi(2) + (kp[1] - vp[1]).powi(2) + (kp[2] - vp[2]).powi(2)).sqrt();
                if dist > NEAR {
                    continue;
                }
                let bearing = match face {
                    Some(y) => {
                        let to = (kp[1] - vp[1]).atan2(kp[0] - vp[0]);
                        let back = y + std::f32::consts::PI;
                        let mut dd = to - back;
                        while dd > std::f32::consts::PI {
                            dd -= std::f32::consts::TAU;
                        }
                        while dd < -std::f32::consts::PI {
                            dd += std::f32::consts::TAU;
                        }
                        dd.abs().to_degrees()
                    }
                    None => f32::NAN,
                };
                if best.map_or(true, |(bd, _)| dist < bd) {
                    best = Some((dist, bearing));
                }
            }
            if let Some((_dist, bearing)) = best {
                near += 1;
                if bearing.is_finite() && bearing <= BACK_DEG {
                    behind += 1;
                }
            }
        }
        if deaths == 0 {
            continue;
        }
        if is_assn {
            a_d += deaths;
            a_b += behind;
        } else {
            o_d += deaths;
            o_b += behind;
        }
        println!(
            "{:<40} {:<24} {deaths:>7} {near:>8} {behind:>8} {:>7.1}%{}",
            name(p).chars().take(40).collect::<String>(),
            gt.chars().take(24).collect::<String>(),
            100.0 * behind as f32 / deaths as f32,
            if is_assn { "  <- ASSASSINATION GAMETYPE" } else { "" }
        );
    }
    let sh = |b: usize, d: usize| if d == 0 { f32::NAN } else { 100.0 * b as f32 / d as f32 };
    println!(
        "\nassassination gametypes: {a_b}/{a_d} deaths from behind within {NEAR} wu = {:.1}%\nevery other clip:        {o_b}/{o_d} = {:.1}%",
        sh(a_b, a_d),
        sh(o_b, o_d)
    );
}

/// #elite Is there an ELITE anywhere in this machine's films?
///
/// HMS draws every body with the Spartan biped and the caption says so. Making that per-species
/// needs (a) a field in the file that says which species a player is, and (b) a clip where the
/// answer is known independently, to test any candidate against. (b) is the blocker: in Reach a
/// player's species is not a profile setting, it is decided by the GAMETYPE — Invasion gives one
/// whole team Elites, and Infection's zombies are Elites — so an ordinary Slayer or Forge clip has
/// no Elite in it at all and cannot settle anything.
///
/// This censuses the whole corpus by engine type, and reports the two candidate bytes in the
/// player record (`+0x48`, 0..3, and `+0x4B`, 0..4) against team, which is the only partition an
/// Invasion clip would make obvious.
#[test]
fn probe_elite_species_census() {
    let films = films();
    if films.is_empty() {
        return;
    }
    const ENGINE: [&str; 13] = [
        "none", "ctf", "slayer", "oddball", "king", "juggernaut", "territories", "assault",
        "infection", "vip", "invasion", "stockpile", "race",
    ];
    let mut by_engine: std::collections::BTreeMap<String, usize> = Default::default();
    let mut f48_by_team: std::collections::BTreeMap<(u8, u8), usize> = Default::default();
    let mut f4b_by_team: std::collections::BTreeMap<(u8, u8), usize> = Default::default();
    let mut players = 0usize;
    let mut decisive: Vec<String> = Vec::new();
    for p in &films {
        let Ok(f) = crate::film::open_header(p) else { continue };
        let e = f.header.game_engine_type as usize;
        let label = ENGINE.get(e).copied().unwrap_or("?").to_string();
        *by_engine.entry(format!("{e} {label}")).or_default() += 1;
        let gt = f.header.game_variant.title.to_lowercase();
        let desc = f.content.description.to_lowercase();
        // Reach's "Elite Slayer" is an ordinary slayer variant whose base player traits set the
        // species, so the engine type cannot show it; the title is the only hint available here.
        if label == "invasion" || label == "infection" || gt.contains("elite") || desc.contains("elite") {
            decisive.push(format!("{} ({label}, \"{}\")", name(p), f.header.game_variant.title));
        }
        for pl in f.header.players.iter().filter(|x| x.present()) {
            let Some(a) = pl.appearance() else { continue };
            players += 1;
            *f48_by_team.entry((pl.team, a.f48)).or_default() += 1;
            *f4b_by_team.entry((pl.team, a.f4b)).or_default() += 1;
        }
    }
    println!("{} film(s) by engine type:", films.len());
    for (k, n) in &by_engine {
        println!("   {k:<16} {n}");
    }
    println!("\n{players} present player record(s)");
    println!("   +0x48 by (team, value): {f48_by_team:?}");
    println!("   +0x4B by (team, value): {f4b_by_team:?}");
    if decisive.is_empty() {
        println!(
            "\nNO Invasion clip, NO Infection clip and no variant that names Elites: there is no recording \
             here in which any player is known to be an Elite, so NO candidate field can be tested. The \
             Spartan default stands.\nNOTE: the game variant's General Settings carry a PLAYER SPECIES \
             setting whose values are Player Preference / All Spartans / All Elites / Use Team Species. \
             Under 'Player Preference' the species comes from each player's own customisation, so a \
             per-player preferred-species field may well BE in the player record \u{2014} this sweep \
             simply had no clip in which anyone was an Elite to score it against. Decode the game \
             variant first; it says which clips even permitted one."
        );
    } else {
        println!("\nDECISIVE CLIPS ({}):", decisive.len());
        for d in &decisive {
            println!("   {d}");
        }
    }
}

/// #weapon Is the SPAWN weapon ever sent? Measure when a body's first weapon report arrives.
///
/// Derived field 0x2A reports a unit's `'weap'` on change, and coverage is poor (0-8% of bodies
/// over 12 films, zero on half of them). The open question is whether the weapon a player SPAWNS
/// with is anywhere in the stream. If it were replicated at creation, a body's first weapon report
/// would land at or very near its own `first_tick`. This measures that gap for every body that
/// ever reports one.
#[test]
fn probe_first_weapon_report_delay() {
    let films = films();
    if films.is_empty() {
        return;
    }
    let mut gaps: Vec<i64> = Vec::new();
    let (mut bodies, mut armed) = (0usize, 0usize);
    let mut at_spawn = 0usize;
    for p in films.iter().take(12) {
        let Some((_f, idx)) = indexed(p) else { continue };
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            bodies += 1;
            let Some(&(wt, _)) = t.weapons.first() else { continue };
            armed += 1;
            let g = wt as i64 - t.first_tick as i64;
            if g <= 2 {
                at_spawn += 1;
            }
            gaps.push(g);
        }
    }
    gaps.sort();
    let q = |f: f32| gaps.get(((gaps.len().max(1) - 1) as f32 * f) as usize).copied().unwrap_or(0);
    println!(
        "{armed} of {bodies} biped track(s) ever report a weapon ({:.1}%); {at_spawn} of those within 2 ticks of their own creation",
        100.0 * armed as f32 / bodies.max(1) as f32
    );
    println!(
        "first weapon report - first_tick: p05 {} p25 {} median {} p75 {} p95 {} ticks",
        q(0.05), q(0.25), q(0.5), q(0.75), q(0.95)
    );
    println!(
        "VERDICT: if the spawn weapon were replicated, the median would be ~0. It is {} ticks ({:.1} s).",
        q(0.5),
        q(0.5) as f32 / 60.0
    );
}

/// #instmesh #armour-census Walk EVERY recorded player's armour on EVERY film and check each
/// attachment the body would wear — not one body, by eye.
///
/// The spot-check that preceded this looked at one Spartan in one frame and called the armour
/// fixed; the owner was still seeing floating pieces. One clean body is not evidence that all
/// bodies are clean, so this asks the question of the whole corpus and with no renderer involved.
///
/// For each film and each present player, the recorded appearance is resolved through the
/// renderer's OWN code (`scene::player_customization_table` → `appearance_picks_from` →
/// `set_player_variants_for_audit` → `scene::variant_section_bits`), so what is audited is what
/// would be drawn. Then, for each instance that variant wears:
///
/// * does the placement name a node that exists?
/// * do the instance's own vertices sit near that node in the bind pose? A piece hangs off its
///   node, so its centroid should be within a fraction of a world unit of it. The threshold is
///   `FAR_WU` and the yardstick is the one this work already measured: a stray triangle kept on
///   the belt instead of the helmet was 0.217 wu out of place in the bind pose, and the worst of
///   the 35 unowned runs was 0.290 wu, so anything past half a world unit is a piece drawn
///   somewhere it does not belong.
/// * does any index of the instance fall outside the mesh, or into another instance's range?
///
/// Reports bodies-with-a-fault over bodies-checked, and groups the failures by the armour slots
/// they have in common, which is what names the bug if it is one permutation.
#[test]
#[ignore]
fn probe_armour_census() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else {
        eprintln!("film_anim: forge_halo cache not available");
        return;
    };
    /// Past this, in the BIND pose, a piece is not sitting on the node it hangs off.
    const FAR_WU: f32 = 0.5;

    let Some(tag) = native
        .tags_by_group(cache, "bipd")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).unwrap_or_default().contains("characters\\spartans\\spartans"))
    else {
        println!("no spartan biped in this map");
        return;
    };
    let mode = native.resolve_mode(cache, tag);
    let Some(def) = crate::reach_actor::ActorDef::read(&native, cache, tag) else { return };
    let Some((im, insts)) = crate::scene::model_instances(&native, cache, mode) else { return };
    // The object tag the variant walk is keyed on (the hlmt hangs off the OBJECT, not the model).
    let obj_tag = tag;

    // ---- the geometry of each attachment, once -----------------------------------------------
    let Ok(model) = native.open_model(cache, mode) else {
        println!("could not open the spartan model");
        return;
    };
    let nsec = native.model_section_count(model);
    let Some(sec) = native.model_section(model, im as u32) else { return };
    let Some(geom) = native.model_decode_geometry(model, im as u32) else { return };
    let stride = sec.vertex_stride.max(12) as usize;
    let vcount = geom.vertices.len() / stride.max(1);
    let pos = |vi: usize| -> Option<glam::Vec3> {
        let o = vi * stride;
        let b = geom.vertices.get(o..o + 12)?;
        Some(glam::Vec3::new(
            f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        ))
    };
    let idx_stride = if sec.index_stride == 4 { 4usize } else { 2usize };
    let index_at = |k: usize| -> Option<usize> {
        let o = k * idx_stride;
        let b = geom.indices.get(o..o + idx_stride)?;
        Some(if idx_stride == 4 {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
        } else {
            u16::from_le_bytes([b[0], b[1]]) as usize
        })
    };
    let nidx = geom.indices.len() / idx_stride;
    let bind_pos = |n: usize| -> Option<glam::Vec3> {
        let nd = def.model.skel.nodes.get(n)?;
        if !nd.inverse_is_frame() { return None; }
        Some(nd.inverse_bind_authored().inverse().w_axis.truncate())
    };
    // Per instance: (centroid, radius, distance from its node, why it is bad)
    struct Inst {
        node: i32,
        centre: glam::Vec3,
        radius: f32,
        dist: f32,
        bad: Option<String>,
    }
    let nodes = def.model.skel.len();
    // #instmesh-space The placement ranges are RAW-STRIP offsets and `geom` is the EXPANDED
    // list, so they are translated through the native expansion's own per-triangle origin map —
    // the same call the renderer uses, so this audits what is actually drawn.
    let origins = native.model_section_triangle_origins(model, im as u32);
    if origins.is_none() {
        println!("   !! this native library has no triangle-origin map; the audit below is meaningless");
    }
    let owners = crate::scene::instance_index_ranges(&insts, origins.as_deref());
    let mut info: Vec<Inst> = Vec::with_capacity(insts.len());
    for (ii, inst) in insts.iter().enumerate() {
        let (a, b) = owners.get(ii).copied().flatten().unwrap_or((0, 0));
        let mut sum = glam::Vec3::ZERO;
        let mut n = 0usize;
        let mut oob = 0usize;
        let mut pts: Vec<glam::Vec3> = Vec::new();
        for k in a..b.min(nidx) {
            let Some(vi) = index_at(k) else { continue };
            if vi >= vcount {
                oob += 1;
                continue;
            }
            if let Some(p) = pos(vi) {
                sum += p;
                n += 1;
                if pts.len() < 4096 { pts.push(p); }
            }
        }
        let centre = if n > 0 { sum / n as f32 } else { glam::Vec3::ZERO };
        let radius = pts.iter().map(|p| (*p - centre).length()).fold(0.0f32, f32::max);
        let node_ok = inst.node >= 0 && (inst.node as usize) < nodes;
        let dist = match (node_ok, bind_pos(inst.node.max(0) as usize)) {
            (true, Some(bp)) => (centre - bp).length(),
            _ => f32::NAN,
        };
        let bad = if !node_ok {
            Some(format!("placement names node {} of {nodes}", inst.node))
        } else if owners.get(ii).copied().flatten().is_none() {
            Some("no triangles of the expanded list map to this placement's strip range".into())
        } else if b > nidx {
            Some(format!("index range {a}..{b} runs past the mesh's {nidx} indices"))
        } else if oob > 0 {
            Some(format!("{oob} index/indices point past the mesh's {vcount} vertices"))
        } else if n == 0 {
            Some("no vertices at all".into())
        } else if dist.is_finite() && dist > FAR_WU {
            Some(format!(
                "centroid is {dist:.3} wu from '{}' (radius {radius:.3})",
                def.model.node_names.get(inst.node as usize).cloned().unwrap_or_default()
            ))
        } else {
            None
        };
        if bad.is_some() && ii < 400 {
            // keep them all; the summary prints the first few
        }
        info.push(Inst { node: inst.node, centre, radius, dist, bad });
    }
    native.close_model(model);
    let model_bad: Vec<usize> = (0..info.len()).filter(|&i| info[i].bad.is_some()).collect();
    println!(
        "== spartans: {} placements over {nsec} sections, instance mesh {im} ==",
        insts.len()
    );
    // Do the subpart index ranges and the DECODED index buffer live in the same space? They must,
    // or every attachment is paired with somebody else's triangles.
    println!(
        "   section: vertex_count={} index_count={} index_stride={} vertex_stride={} format_class={} flags={:#x}",
        sec.vertex_count, sec.index_count, sec.index_stride, sec.vertex_stride, sec.format_class, sec.flags
    );
    println!(
        "   decoded: {vcount} vertices, {nidx} indices (subparts cover up to {})",
        insts.iter().map(|x| x.index_start + x.index_count).max().unwrap_or(0)
    );
    println!("   placements that are wrong IN THE MODEL ITSELF: {}", model_bad.len());
    for &i in model_bad.iter().take(12) {
        println!(
            "     [{i}] node {} '{}': {}",
            info[i].node,
            def.model.node_names.get(info[i].node.max(0) as usize).cloned().unwrap_or_default(),
            info[i].bad.as_deref().unwrap_or("")
        );
        // WHICH node does this piece actually sit on? If the answer is a different bone, the
        // placement's node index is being misread; if it is the same bone, the model is authored
        // that way and the engine draws it in the same place.
        let mut near: Vec<(f32, usize)> = (0..nodes)
            .filter_map(|n| bind_pos(n).map(|bp| ((info[i].centre - bp).length(), n)))
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let list = near
            .iter()
            .take(3)
            .map(|(d, n)| format!("'{}' {d:.3}", def.model.node_names.get(*n).cloned().unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(", ");
        println!("          centroid ({:.3}, {:.3}, {:.3}) radius {:.3}; nearest nodes: {list}",
            info[i].centre.x, info[i].centre.y, info[i].centre.z, info[i].radius);
    }
    // The raw placement block around every suspect, names resolved, so the question "is the tag
    // really saying this or is the reader wrong" is answered from the bytes.
    if let Some(m) = native.tag_meta(cache, mode, 0, 0x70) {
        let u = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        let (pcnt, pptr) = (u(&m, 0x20), u(&m, 0x24));
        if let Some(pb) = native.tag_ptr_bytes(cache, pptr, pcnt * 60) {
            for &i in model_bad.iter().take(4) {
                let lo = i.saturating_sub(2);
                for j in lo..(i + 3).min(pcnt as usize) {
                    let e = &pb[j * 60..j * 60 + 60];
                    let nm = native.string_id(cache, u(e, 0)).unwrap_or_default();
                    let node = u(e, 4) as i32;
                    let hex: String = e[..16].iter().map(|b| format!("{b:02x}")).collect();
                    println!(
                        "          placement[{j}] name='{nm}' node={node} '{}' idx[{}..+{}] raw={hex}{}",
                        def.model.node_names.get(node.max(0) as usize).cloned().unwrap_or_default(),
                        insts[j].index_start, insts[j].index_count,
                        if j == i { "   <-- SUSPECT" } else { "" }
                    );
                }
            }
        }
    }
    // A more sensitive test than the distance threshold: is the node a placement NAMES the one
    // its geometry actually sits on? A piece hung off the wrong bone is the whole fault class, and
    // this catches it even when the two bones happen to be close together.
    {
        let nearest = |c: glam::Vec3| -> Option<(usize, f32)> {
            (0..nodes)
                .filter_map(|n| bind_pos(n).map(|bp| (n, (c - bp).length())))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        };
        let mut mismatched: Vec<(usize, usize, f32, f32)> = Vec::new();
        for (i, x) in info.iter().enumerate() {
            if x.node < 0 { continue; }
            let Some((nn, nd)) = nearest(x.centre) else { continue };
            // "names its nearest bone, or one essentially as close" — a fixup/jiggle bone sits on
            // top of its parent, so requiring the exact nearest would flag harmless pairs.
            if nn != x.node as usize && x.dist.is_finite() && x.dist > nd + 0.05 {
                mismatched.push((i, nn, nd, x.dist));
            }
        }
        println!(
            "   placements whose geometry sits on a DIFFERENT bone than the one they name: {} of {}",
            mismatched.len(),
            info.len()
        );
        for (i, nn, nd, dist) in mismatched.iter().take(12) {
            println!(
                "     [{i}] names '{}' ({dist:.3} wu away) but sits on '{}' ({nd:.3} wu)",
                def.model.node_names.get(info[*i].node.max(0) as usize).cloned().unwrap_or_default(),
                def.model.node_names.get(*nn).cloned().unwrap_or_default()
            );
        }
    }
    let mut d: Vec<f32> = info.iter().map(|x| x.dist).filter(|x| x.is_finite()).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| d.get(((d.len().max(1) - 1) as f32 * f) as usize).copied().unwrap_or(f32::NAN);
    println!(
        "   attachment centroid to its own node, bind pose: p50 {:.3} p90 {:.3} p99 {:.3} max {:.3} wu (threshold {FAR_WU})",
        q(0.5), q(0.9), q(0.99), d.last().copied().unwrap_or(f32::NAN)
    );

    // ---- every recorded player on every film --------------------------------------------------
    let table = crate::scene::player_customization_table(&native, cache);
    if table.is_empty() {
        println!("   (no player_model_customization in this cache; the per-body pass is skipped)");
        return;
    }
    let films = films();
    let (mut bodies, mut bad_bodies, mut default_only) = (0usize, 0usize, 0usize);
    let mut worn_hist: Vec<usize> = Vec::new();
    let mut by_slot: std::collections::BTreeMap<(usize, u8), usize> = Default::default();
    let mut bad_by_slot: std::collections::BTreeMap<(usize, u8), usize> = Default::default();
    let mut examples: Vec<String> = Vec::new();
    for p in films.iter() {
        let Ok(f) = crate::film::open_header(p) else { continue };
        for pl in f.header.players.iter().filter(|x| x.present()) {
            let Some(a) = pl.appearance() else { continue };
            bodies += 1;
            for (k, &opt) in a.slots.iter().enumerate() {
                *by_slot.entry((k, opt)).or_default() += 1;
            }
            let picks = crate::scene::appearance_picks_from(&table, &a.slots, &crate::film::SPARTAN_APPEARANCE_SLOTS);
            if picks.is_empty() {
                default_only += 1;
            }
            // Resolve EXACTLY as the renderer does.
            crate::scene::set_player_variants_for_audit(vec![crate::scene::PlayerVariant {
                sid: 0xF1A1_0000,
                picks,
                elite_picks: Vec::new(),
            }]);
            let Some((sids, _, _, inst_bits, _)) =
                crate::scene::variant_section_bits(&native, cache, obj_tag, mode, nsec as usize)
            else {
                continue;
            };
            let Some(v) = sids.iter().position(|&s| s == 0xF1A1_0000) else { continue };
            let worn: Vec<usize> = (0..insts.len())
                .filter(|&i| inst_bits.get(i).copied().unwrap_or(0) & (1u64 << v) != 0)
                .collect();
            worn_hist.push(worn.len());
            let faults: Vec<usize> = worn.iter().copied().filter(|&i| info[i].bad.is_some()).collect();
            if !faults.is_empty() {
                bad_bodies += 1;
                for (k, &opt) in a.slots.iter().enumerate() {
                    *bad_by_slot.entry((k, opt)).or_default() += 1;
                }
                if examples.len() < 10 {
                    examples.push(format!(
                        "{}  slots {:?}  wears {} piece(s), {} bad: {}",
                        pl.gamertag,
                        a.slots,
                        worn.len(),
                        faults.len(),
                        faults
                            .iter()
                            .take(3)
                            .map(|&i| format!("[{i}] {}", info[i].bad.as_deref().unwrap_or("")))
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                }
            }
        }
    }
    crate::scene::set_player_variants_for_audit(Vec::new());
    worn_hist.sort();
    println!(
        "\n{bad_bodies} of {bodies} recorded player(s) wear at least one BAD attachment ({:.1}%); {default_only} resolve to no picks at all",
        100.0 * bad_bodies as f32 / bodies.max(1) as f32
    );
    println!(
        "   attachments worn per body: min {} median {} max {}",
        worn_hist.first().copied().unwrap_or(0),
        worn_hist.get(worn_hist.len() / 2).copied().unwrap_or(0),
        worn_hist.last().copied().unwrap_or(0)
    );
    for e in &examples {
        println!("     {e}");
    }
    if bad_bodies > 0 {
        println!("\n   armour options the FAILING bodies share (slot, option: failing/total):");
        let mut rows: Vec<((usize, u8), usize)> = bad_by_slot.into_iter().collect();
        rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for ((k, opt), n) in rows.into_iter().take(16) {
            let tot = by_slot.get(&(k, opt)).copied().unwrap_or(0);
            println!(
                "     slot {k} ({}) option {opt}: {n}/{tot}",
                crate::film::APPEARANCE_SLOT_NAMES.get(k).copied().unwrap_or("?")
            );
        }
    }
}

/// #instmesh HOW do the instance placements pair with the instance mesh's geometry?
///
/// `probe_armour_census` showed that pairing them 1:1 in block order is WRONG: 74 of the Spartan's
/// 105 placements end up with geometry that sits on a different bone than the placement names, and
/// the mismatch is a shuffle rather than an offset. This dumps the raw blocks so the real key can
/// be read off the bytes instead of guessed.
#[test]
#[ignore]
fn probe_instance_subpart_pairing() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else { return };
    let Some(tag) = native
        .tags_by_group(cache, "bipd")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).unwrap_or_default().contains("characters\\spartans\\spartans"))
    else { return };
    let mode = native.resolve_mode(cache, tag);
    let u = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
    let u16at = |s: &[u8], o: usize| u16::from_le_bytes([s[o], s[o + 1]]);
    let Some(m) = native.tag_meta(cache, mode, 0, 0x70) else { return };
    let inst_mesh = u(&m, 0x1C) as i32;
    let (pcnt, pptr) = (u(&m, 0x20), u(&m, 0x24));
    let (mcnt, mptr) = (u(&m, crate::reach_skel::OFF_MESHES), u(&m, crate::reach_skel::OFF_MESHES + 4));
    println!("instance mesh = {inst_mesh}, placements = {pcnt}, meshes = {mcnt}");
    let Some(mb) = native.tag_ptr_bytes(cache, mptr, mcnt * 92) else { return };
    let me = &mb[inst_mesh as usize * 92..(inst_mesh as usize + 1) * 92];
    let hex: String = me.iter().map(|b| format!("{b:02x}")).collect();
    println!("mesh[{inst_mesh}] raw = {hex}");
    let (prcnt, prptr) = (u(me, 0x04), u(me, 0x08));
    let (scnt, sptr) = (u(me, 0x0C), u(me, 0x10));
    println!("  parts: count={prcnt} ptr={prptr:#x}   subparts: count={scnt} ptr={sptr:#x}");
    // The PART block: in Reach a part is 72 bytes. Dump the first few whatever the stride, by
    // trying the strides that make the block's own count land on a sane size.
    for stride in [72usize, 76, 68, 64] {
        let Some(pb) = native.tag_ptr_bytes(cache, prptr, prcnt * stride as u32) else { continue };
        println!("  -- parts at stride {stride} --");
        for i in 0..prcnt.min(8) as usize {
            let e = &pb[i * stride..i * stride + stride.min(32)];
            let h: String = e.iter().map(|b| format!("{b:02x}")).collect();
            println!("     part[{i}] {h}");
        }
        break;
    }
    let Some(sb) = native.tag_ptr_bytes(cache, sptr, scnt * 16) else { return };
    println!("  -- subparts (16 B) --");
    for i in 0..scnt.min(12) as usize {
        let e = &sb[i * 16..i * 16 + 16];
        let h: String = e.iter().map(|b| format!("{b:02x}")).collect();
        println!(
            "     subpart[{i}] start={} count={} u16@8={} u16@10={} u16@12={} u16@14={} raw={h}",
            u(e, 0), u(e, 4), u16at(e, 8), u16at(e, 10), u16at(e, 12), u16at(e, 14)
        );
    }
    // The NATIVE submesh table for the instance mesh. The per-shader split already proves these
    // ranges are in the DECODED (de-stripped) index space -- they sum to exactly the decoded
    // index count -- so if there is one per placement, this is the translation the raw subpart
    // ranges are missing.
    if let Ok(model) = native.open_model(cache, mode) {
        let n = native.model_submesh_count(model);
        let mine: Vec<_> = (0..n)
            .filter_map(|i| native.model_submesh(model, i))
            .filter(|sm| sm.section_index as i32 == inst_mesh)
            .collect();
        println!("  -- native submeshes of section {inst_mesh}: {} of {n} --", mine.len());
        for (k, sm) in mine.iter().take(12).enumerate() {
            println!("     submesh[{k}] shader={} idx[{}..+{}]", sm.shader_index, sm.index_start, sm.index_length);
        }
        let total: u32 = mine.iter().map(|sm| sm.index_length).sum();
        println!("     total indices in those submeshes: {total}");
        native.close_model(model);
    }
    // Raw index_count vs DECODED index count, per section: are the two in the same space?
    if let Ok(model) = native.open_model(cache, mode) {
        let n = native.model_section_count(model);
        let (mut same, mut diff) = (0usize, 0usize);
        for i in 0..n {
            let Some(sec) = native.model_section(model, i) else { continue };
            let Some(g) = native.model_decode_geometry(model, i) else { continue };
            let stride = if sec.index_stride == 4 { 4 } else { 2 };
            let dec = g.indices.len() / stride;
            if dec == sec.index_count as usize { same += 1; } else {
                diff += 1;
                if diff <= 8 {
                    println!("     section {i}: raw index_count={} decoded={dec} (bytes {}) verts={}  stride={}",
                        sec.index_count, g.indices.len(), sec.vertex_count, stride);
                }
            }
        }
        println!("  -- sections whose decoded index count MATCHES the tag's: {same}; differs: {diff} --");
        native.close_model(model);
    }
    // And the placements, with their names, for the same indices.
    let Some(pb) = native.tag_ptr_bytes(cache, pptr, pcnt * 60) else { return };
    println!("  -- placements (60 B) --");
    for i in 0..pcnt.min(12) as usize {
        let e = &pb[i * 60..i * 60 + 60];
        println!(
            "     placement[{i}] name='{}' node={}",
            native.string_id(cache, u(e, 0)).unwrap_or_default(),
            u(e, 4) as i32
        );
    }
}

/// #instmesh Can the attachment boundaries be recovered from the DECODED list alone?
///
/// The tag's subpart ranges are in STRIP space and HMS renders a de-stripped LIST, which is why
/// 74 of 105 attachments are bound to the wrong bone. Both sequences are in the SAME ORDER, so if
/// the decoded triangles segment cleanly into 105 vertex-disjoint runs, segment *i* is attachment
/// *i* and the mapping is recoverable without the strip. This measures whether that is true.
#[test]
#[ignore]
fn probe_instance_segmentation() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else { return };
    let Some(tag) = native
        .tags_by_group(cache, "bipd")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).unwrap_or_default().contains("characters\\spartans\\spartans"))
    else { return };
    let mode = native.resolve_mode(cache, tag);
    let Some((im, insts)) = crate::scene::model_instances(&native, cache, mode) else { return };
    let Ok(model) = native.open_model(cache, mode) else { return };
    let Some(g) = native.model_decode_geometry(model, im as u32) else { return };
    native.close_model(model);
    let idx: Vec<u32> = g.indices.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect();
    let tris = idx.len() / 3;
    // Walk the triangles in order; a new segment begins when a triangle shares NO vertex with the
    // span of the current one (attachments are vertex-disjoint blocks emitted in order).
    let mut segs: Vec<(usize, usize, u32, u32)> = Vec::new(); // (first tri, tri count, vmin, vmax)
    let (mut lo, mut hi, mut start) = (u32::MAX, 0u32, 0usize);
    for t in 0..tris {
        let (a, b, c) = (idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]);
        let (tmin, tmax) = (a.min(b).min(c), a.max(b).max(c));
        if start < t && (tmin > hi || tmax < lo) {
            segs.push((start, t - start, lo, hi));
            start = t;
            lo = tmin;
            hi = tmax;
        } else {
            lo = lo.min(tmin);
            hi = hi.max(tmax);
        }
    }
    if start < tris { segs.push((start, tris - start, lo, hi)); }
    println!("decoded {tris} triangles -> {} vertex-disjoint segment(s); {} placements", segs.len(), insts.len());
    let mut sizes: Vec<usize> = segs.iter().map(|s| s.1).collect();
    sizes.sort();
    println!(
        "   segment triangle counts: min {} median {} max {}",
        sizes.first().copied().unwrap_or(0),
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0)
    );
    for (i, s) in segs.iter().take(8).enumerate() {
        println!("   seg[{i}] tris {}..+{} verts {}..{}", s.0, s.1, s.2, s.3);
    }
}

// ---------------------------------------------------------------------------------------------
// #film-action  WHY MOST ASSASSINATIONS ARE NOT STAGED, and what else the wire offers
// ---------------------------------------------------------------------------------------------

/// The gap, measured: how many bodies the recording DESTROYS against how many kills its
/// `incident` channel names, and WHOSE kills those are.
///
/// This is the probe that explains the user's "the assassinations do NOT play". It is not a
/// staging bug: on an assassination gametype the recording states only a handful of the kills,
/// and every one of them belongs to the SAME player.
#[test]
fn probe_kill_feed_coverage_against_the_death_count() {
    for p in films() {
        let Some((f, idx)) = indexed(&p) else { continue };
        let deaths = idx
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.destroyed_tick.is_some())
            .count();
        if deaths < 20 {
            continue;
        }
        let kills: Vec<&crate::film_action::ActionEvent> = idx
            .events
            .iter()
            .filter(|e| e.incident_label().is_some_and(|n| crate::film_action::KILL_INCIDENTS.contains(&n)))
            .collect();
        let assn = crate::film_anim::assassinations(&idx).len();
        let mut by_subject: std::collections::BTreeMap<Option<u32>, usize> = Default::default();
        for e in &kills {
            *by_subject.entry(e.subject).or_default() += 1;
        }
        let dmg = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_DAMAGE_AFTERMATH).count();
        let melee = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_MELEE_INITIATE).count();
        println!(
            "{:<42} {:<34} bodies destroyed {deaths:>4}  kill rows {:>3}  assassinations {assn:>3}  \
             distinct killers {:>2}  damage_aftermath {dmg:>5}  melee_initiate {melee:>5}",
            name(&p),
            f.content.title,
            kills.len(),
            by_subject.len(),
        );
        for (s, n) in by_subject.iter().take(6) {
            println!("        killer {:>12} named in {n} kill row(s)", s.map(|v| format!("{v:#x}")).unwrap_or_else(|| "none".into()));
        }
    }
}

/// Does the inference reproduce the assassinations the recording DOES state, and does it stay
/// quiet on clips that are not about assassination?
///
/// The 8 rows the `incident` channel states on the user's assassination clip are a free ORACLE:
/// run [`crate::film_anim::inferred_assassinations`] with nothing excluded and count how many of
/// them it rediscovers (recall). The share of all deaths it fires on, read across assassination
/// gametypes against everything else, is the contrast.
#[test]
fn probe_inferred_assassination_recall_and_contrast() {
    // HOLD-OUT DISCIPLINE: `HMS_PROBE_FILM=<substring>` restricts the probe to the DEVELOPMENT
    // clip, so the rest of the corpus stays unexamined until the parameters are frozen. The
    // headline numbers are the unfiltered run, taken once, after the freeze.
    let only = std::env::var("HMS_PROBE_FILM").unwrap_or_default();
    // ...and the DEVELOPMENT clip is excluded from the held-out run by name, so the headline
    // numbers never include the clip whose internals were inspected.
    let skip = std::env::var("HMS_PROBE_SKIP").unwrap_or_default();
    let mut rows: Vec<(bool, String, usize, usize, usize, usize, usize)> = Vec::new();
    for p in films() {
        if !only.is_empty() && !name(&p).contains(&only) {
            continue;
        }
        if !skip.is_empty() && name(&p).contains(&skip) {
            continue;
        }
        let Some((f, idx)) = indexed(&p) else { continue };
        let deaths = idx
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11 && crate::film_anim::death_tick(t).is_some())
            .count();
        if deaths < 20 {
            continue;
        }
        let stated = crate::film_anim::assassinations(&idx);
        let stated_victims: std::collections::HashSet<u32> = stated.iter().filter_map(|a| a.victim).collect();
        // recall: run with NOTHING excluded, so the stated ones are fair game
        let all = crate::film_anim::inferred_assassinations(&idx, &[]);
        let found: std::collections::HashSet<u32> = all.iter().filter_map(|a| a.victim).collect();
        let recalled = stated_victims.iter().filter(|v| found.contains(v)).count();
        // ...and the rows that would actually be ADDED
        let added = crate::film_anim::inferred_assassinations(&idx, &stated).len();
        // THE CONFOUND, reported rather than inferred: v2 rests entirely on the distance term,
        // and an ordinary melee kill is also close range. How many of the rows v2 fires on would
        // the REJECTED v1 melee test have called a melee kill?
        let melee_overlap = all
            .iter()
            .filter(|a| {
                let Some(k) = a.attacker else { return false };
                idx.events.iter().any(|e| {
                    e.kind == crate::film_action::EV_MELEE_INITIATE
                        && e.subject == Some(k)
                        && e.tick <= a.tick + 30
                        && e.tick + 120 >= a.tick
                })
            })
            .count();
        let title = f.content.title.clone();
        let assn_gametype = title.to_lowercase().contains("assassin");
        println!(
            "{:<6} {:<42} {:<34} deaths {deaths:>4}  stated {:>3}  inferred-total {:>4} ({:>5.2}% of deaths)  recall {recalled}/{}  would add {added}  melee-test overlap {melee_overlap}",
            if assn_gametype { "ASSN" } else { "other" },
            name(&p),
            title,
            stated_victims.len(),
            all.len(),
            100.0 * all.len() as f32 / deaths as f32,
            stated_victims.len(),
        );
        rows.push((assn_gametype, title, deaths, stated_victims.len(), all.len(), recalled, melee_overlap));
    }
    for want in [true, false] {
        let g: Vec<_> = rows.iter().filter(|r| r.0 == want).collect();
        if g.is_empty() {
            continue;
        }
        let d: usize = g.iter().map(|r| r.2).sum();
        let i: usize = g.iter().map(|r| r.4).sum();
        let st: usize = g.iter().map(|r| r.3).sum();
        let rc: usize = g.iter().map(|r| r.5).sum();
        let mo: usize = g.iter().map(|r| r.6).sum();
        println!(
            "POOLED {:<6} {} clip(s): {d} deaths, {i} inferred ({:.2}%), {st} stated, recall {rc}/{st} ({:.0}%), melee-test overlap {mo}/{i} ({:.0}%)",
            if want { "ASSN" } else { "other" },
            g.len(),
            100.0 * i as f32 / d.max(1) as f32,
            100.0 * rc as f32 / st.max(1) as f32,
            100.0 * mo as f32 / i.max(1) as f32,
        );
    }
}

/// WHY the inference missed every assassination the recording states — the per-condition
/// autopsy of a registered failure (recall 0 of 8 on the development clip).
///
/// Diagnosis only: it changes no parameter and asserts nothing. For each of the stated victims
/// it prints whether that body even has a derived death at all, whether any `damage_aftermath`
/// names it, whether the stated killer swung, and the geometry at the moment of death.
#[test]
fn probe_why_the_inference_missed_the_stated_assassinations() {
    let only = std::env::var("HMS_PROBE_FILM").unwrap_or_default();
    for p in films() {
        if !only.is_empty() && !name(&p).contains(&only) {
            continue;
        }
        let Some((f, idx)) = indexed(&p) else { continue };
        let stated = crate::film_anim::assassinations(&idx);
        if stated.is_empty() {
            continue;
        }
        println!("\n== {} {:?}", name(&p), f.content.title);
        for a in &stated {
            let Some(v) = a.victim else { continue };
            let Some(t) = idx.tracks.iter().find(|q| q.net_id == v) else {
                println!("  victim {v:#x} @{} -- NO TRACK AT ALL", a.tick);
                continue;
            };
            let d = crate::film_anim::death_tick(t);
            let dmg: Vec<(u32, Option<u32>)> = idx
                .events
                .iter()
                .filter(|e| e.kind == crate::film_action::EV_DAMAGE_AFTERMATH && e.subject == Some(v))
                .map(|e| (e.tick, e.other))
                .collect();
            let dmg_any_ref = idx
                .events
                .iter()
                .filter(|e| e.kind == crate::film_action::EV_DAMAGE_AFTERMATH)
                .filter(|e| e.subject == Some(v) || e.other == Some(v))
                .count();
            let melee = a.attacker.map(|k| {
                idx.events
                    .iter()
                    .filter(|e| e.kind == crate::film_action::EV_MELEE_INITIATE && e.subject == Some(k))
                    .map(|e| e.tick as i64 - a.tick as i64)
                    .min_by_key(|x| x.abs())
            });
            let geo = a.attacker.and_then(|k| {
                let kt = idx.tracks.iter().find(|q| q.net_id == k)?;
                let kp = kt.sample(a.tick)?;
                let vp = t.sample(a.tick)?;
                let dist = ((kp[0] - vp[0]).powi(2) + (kp[1] - vp[1]).powi(2) + (kp[2] - vp[2]).powi(2)).sqrt();
                let face = t.facing_at(a.tick)?;
                let to = (kp[1] - vp[1]).atan2(kp[0] - vp[0]);
                Some((dist, super::ang_delta(to, face + std::f32::consts::PI).abs().to_degrees()))
            });
            println!(
                "  victim {v:#x} incident@{:>6}  track {}..{} destroyed {:?}  death_tick {:?}  \
                 damage_aftermath naming it: {} (as ref0) / {dmg_any_ref} (either ref)  \
                 nearest killer melee {:?} ticks away  geometry {:?}",
                a.tick,
                t.first_tick,
                t.last_tick,
                t.destroyed_tick,
                d,
                dmg.len(),
                melee.flatten(),
                geo.map(|(a, b)| (((a * 100.0) as i32) as f32 / 100.0, ((b * 10.0) as i32) as f32 / 10.0)),
            );
            if !dmg.is_empty() {
                println!("        its damage rows (tick, responsible): {:?}", &dmg[dmg.len().saturating_sub(6)..]);
            }
        }
        // ...and what the 69 rows the rule DOES fire on actually look like
        let inferred = crate::film_anim::inferred_assassinations(&idx, &[]);
        println!("  rule fires on {} death(s); first 8 victims: {:?}", inferred.len(),
            inferred.iter().take(8).map(|a| (a.tick, a.victim.map(|v| format!("{v:#x}")))).collect::<Vec<_>>());
    }
}

/// #corpse-fall  **How hard does Reach pull a body down?** Fitted off the recorded positions
/// of bodies that are plainly in free fall, so the number a corpse falls at is measured on
/// this engine rather than assumed from Earth gravity and a world-unit conversion.
///
/// The fit: over each biped track, take consecutive reported positions, form the vertical
/// velocity of each pair, and keep a triple of pairs only when the body is unsupported the
/// whole way through -- vz strictly decreasing, no horizontal speed that says teleport, and
/// the samples close enough together (the triple inside 20 ticks) that the quantisation is
/// small against the change. The slope of vz against time over such a triple is `-g`.
#[test]
fn probe_gravity_from_falling_bodies() {
    let films = films();
    let mut all: Vec<f32> = Vec::new();
    for p in films.iter().take(12) {
        let Some((_f, idx)) = indexed(p) else { continue };
        let mut mine: Vec<f32> = Vec::new();
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            for w in t.positions.windows(3) {
                let [(t0, a), (t1, b), (t2, c)] = [w[0], w[1], w[2]];
                if t1 <= t0 || t2 <= t1 || t2 - t0 > 20 {
                    continue;
                }
                let d0 = (t1 - t0) as f32 / 60.0;
                let d1 = (t2 - t1) as f32 / 60.0;
                let g0 = (b[0] - a[0]).hypot(b[1] - a[1]) / d0;
                let g1 = (c[0] - b[0]).hypot(c[1] - b[1]) / d1;
                if g0 > 12.0 || g1 > 12.0 {
                    continue; // a respawn teleport, not a fall
                }
                let v0 = (b[2] - a[2]) / d0;
                let v1 = (c[2] - b[2]) / d1;
                // Strictly falling through BOTH pairs, and fast enough that the 1/256 wu
                // position quantisation is noise against the step.
                if !(v1 < v0 && v1 < -1.0) {
                    continue;
                }
                let dt = (d0 + d1) / 2.0;
                if dt < 1.0 / 60.0 {
                    continue;
                }
                let g = -(v1 - v0) / dt;
                if g.is_finite() && g > 0.5 && g < 20.0 {
                    mine.push(g);
                }
            }
        }
        if mine.is_empty() {
            continue;
        }
        mine.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:<46} {:>7} samples  median g {:6.3} wu/s^2  (p25 {:5.2}, p75 {:5.2})",
            name(p),
            mine.len(),
            mine[mine.len() / 2],
            mine[mine.len() / 4],
            mine[mine.len() * 3 / 4]
        );
        all.extend(mine);
    }
    if all.is_empty() {
        println!("no falling samples");
        return;
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nPOOLED {} samples: median {:.4} wu/s^2, p25 {:.3}, p75 {:.3}, mean {:.3}",
        all.len(),
        all[all.len() / 2],
        all[all.len() / 4],
        all[all.len() * 3 / 4],
        all.iter().sum::<f32>() / all.len() as f32
    );
}

/// #instmesh-parts Does the PART BUILDER slice the instance mesh in the same index space the
/// rebinder does?
///
/// `decode_object_geometry` touches the instance mesh twice. The first pass (the bone rebind)
/// translates each placement's `s_subpart` range out of RAW STRIP space through the native
/// expansion's per-triangle origin map. The second pass — the per-shader part builder — slices
/// `indices` with the SAME raw numbers, untranslated. This probe prints what that costs: how much
/// of the expanded buffer the raw slices reach, how many of the emitted runs start on a triangle
/// boundary, and the longest triangle edge the resulting part contains (in bind-pose model space,
/// world units). An armour attachment is a few centimetres across; a triangle whose edge is a
/// whole world unit is a shard spanning the body.
#[test]
#[ignore]
fn probe_instance_part_slicing() {
    let Some((_lock, native, cache)) = open_reach("forge_halo") else { return };
    let Some(tag) = native
        .tags_by_group(cache, "bipd")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).unwrap_or_default().contains("characters\\spartans\\spartans"))
    else { return };
    let mode = native.resolve_mode(cache, tag);
    let Some((im, insts)) = crate::scene::model_instances(&native, cache, mode) else { return };
    let Ok(model) = native.open_model(cache, mode) else { return };
    let Some(sec) = native.model_section(model, im as u32) else { return };
    let Some(geom) = native.model_decode_geometry(model, im as u32) else { return };
    let origins = native.model_section_triangle_origins(model, im as u32);
    let owners = crate::scene::instance_index_ranges(&insts, origins.as_deref());

    let stride = sec.vertex_stride.max(12) as usize;
    let vcount = geom.vertices.len() / stride.max(1);
    let pos = |vi: usize| -> Option<glam::Vec3> {
        let o = vi * stride;
        let b = geom.vertices.get(o..o + 12)?;
        Some(glam::Vec3::new(
            f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        ))
    };
    let istride = if sec.index_stride == 4 { 4usize } else { 2 };
    let index_at = |k: usize| -> Option<u32> {
        let o = k * istride;
        let b = geom.indices.get(o..o + istride)?;
        Some(if istride == 4 {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u16::from_le_bytes([b[0], b[1]]) as u32
        })
    };
    let nidx = geom.indices.len() / istride;

    // Every submesh of the instance mesh, with its (already LIST-RELATIVE) range.
    let smc = native.model_submesh_count(model);
    let subs: Vec<_> = (0..smc)
        .filter_map(|i| native.model_submesh(model, i))
        .filter(|s| s.section_index as usize == im)
        .collect();
    println!("== instance mesh {im}: {nidx} list indices ({} triangles), {vcount} vertices, {} placements, {} submesh(es) ==",
        nidx / 3, insts.len(), subs.len());
    for s in &subs {
        println!("   submesh shader={} list range {}..{}", s.shader_index, s.index_start, s.index_start + s.index_length);
    }
    println!("   raw strip index_count = {}", sec.index_count);
    println!("   {} of {} placements translate into the expanded list",
        owners.iter().filter(|o| o.is_some()).count(), insts.len());

    // Longest edge of a run of `indices` read as a triangle list.
    let worst_edge = |runs: &[(usize, usize)]| -> (f32, usize, usize) {
        let (mut worst, mut tris, mut unaligned) = (0.0f32, 0usize, 0usize);
        let mut buf: Vec<u32> = Vec::new();
        for &(a, b) in runs {
            if a % 3 != 0 || b % 3 != 0 { unaligned += 1; }
            for k in a..b.min(nidx) {
                if let Some(v) = index_at(k) { buf.push(v); }
            }
        }
        for t in buf.chunks_exact(3) {
            let (Some(p0), Some(p1), Some(p2)) =
                (pos(t[0] as usize), pos(t[1] as usize), pos(t[2] as usize)) else { continue };
            let e = (p1 - p0).length().max((p2 - p1).length()).max((p0 - p2).length());
            if e > worst { worst = e; }
            tris += 1;
        }
        (worst, tris, unaligned)
    };

    for s in &subs {
        let (ss, se) = (s.index_start as usize, (s.index_start + s.index_length) as usize);
        // what the CURRENT part builder does: raw subpart offsets, clipped to the list range
        let raw: Vec<(usize, usize)> = insts.iter().filter_map(|i| {
            let a = (i.index_start as usize).max(ss);
            let b = ((i.index_start as usize) + (i.index_count as usize)).min(se);
            (b > a).then_some((a, b))
        }).collect();
        // what it SHOULD do: the translated ranges, clipped to the same list range
        let fixed: Vec<(usize, usize)> = owners.iter().filter_map(|o| {
            let (ra, rb) = (*o)?;
            let (a, b) = (ra.max(ss), rb.min(se));
            (b > a).then_some((a, b))
        }).collect();
        let (wr, tr, ur) = worst_edge(&raw);
        let (wf, tf, uf) = worst_edge(&fixed);
        let cov = |r: &[(usize, usize)]| r.iter().map(|(a, b)| b - a).sum::<usize>();
        println!("   shader {}: RAW slices {} runs, {} indices ({} triangles), {ur} not 3-aligned, worst edge {wr:.3} wu",
            s.shader_index, raw.len(), cov(&raw), tr, );
        println!("              TRANSLATED {} runs, {} indices ({} triangles), {uf} not 3-aligned, worst edge {wf:.3} wu",
            fixed.len(), cov(&fixed), tf);
    }
    native.close_model(model);
}
