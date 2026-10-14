//! Measurement probes for #film-stats. `#[test]`s because that is how every other finding in
//! `re-notes/reach_saved_film_layout.md` was measured: run one with `--nocapture`, read the
//! table, and the number either supports a claim or it does not.
//!
//! None of them assert much. Their job is to PRINT, and a probe that prints a clean negative is
//! a result.
//!
//! PRIVACY: these print a film's FILE NAME only, never its path. The film folders on this
//! machine sit under a directory named after the owner's real first name.

use crate::film;
use crate::film_decode::{self, FilmIndex};
use std::path::PathBuf;

/// Every Reach saved film on this machine, biggest first (the big ones carry entity streams).
///
/// Deliberately its own copy of the walk rather than a reach into `film_anim::probes`, so these
/// probes cannot be broken by edits to another module's test tree.
pub fn films() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for base in ["/mnt", "/run/media", "/media"] {
        let Ok(rd) = std::fs::read_dir(base) else { continue };
        for e in rd.flatten() {
            roots.push(e.path());
            let Ok(rd2) = std::fs::read_dir(e.path()) else { continue };
            for e2 in rd2.flatten() {
                roots.push(e2.path());
            }
        }
    }
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(h));
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in roots {
        let Ok(rd) = std::fs::read_dir(root.join("Users")) else { continue };
        for u in rd.flatten() {
            let Ok(rd2) = std::fs::read_dir(u.path().join("AppData").join("LocalLow")) else { continue };
            for mcc in rd2.flatten() {
                if !mcc.file_name().to_string_lossy().starts_with("MCC") {
                    continue;
                }
                let t = mcc.path().join("Temporary");
                dirs.push(t.join("UserContent").join("HaloReach").join("Movie"));
                dirs.push(t.join("HaloReach").join("autosave"));
            }
        }
    }
    let mut out: Vec<(u64, PathBuf)> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let ok = p
                .extension()
                .map_or(false, |x| x.eq_ignore_ascii_case("mov") || x.eq_ignore_ascii_case("film"));
            let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            if ok && len > 0x20000 {
                out.push((len, p));
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    out.dedup_by(|a, b| a.1 == b.1);
    out.into_iter().map(|(_, p)| p).collect()
}

/// A film's FILE NAME only — never its path.
pub fn name(p: &std::path::Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().to_string()
}

pub fn index_of(p: &std::path::Path) -> Option<(film::Film, FilmIndex)> {
    let f = film::open(p).ok()?;
    let idx = film_decode::build_index_for(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot());
    Some((f, idx))
}

/// Films that actually carry an entity stream, which is the only kind these statistics apply to.
pub fn streamed_films(limit: usize) -> Vec<(PathBuf, film::Film, FilmIndex)> {
    let mut out = Vec::new();
    for p in films() {
        if out.len() >= limit {
            break;
        }
        let Some((f, idx)) = index_of(&p) else { continue };
        if idx.entity_messages == 0 || idx.owner_group_count() == 0 {
            continue;
        }
        out.push((p, f, idx));
    }
    out
}

/// Where do a player's bodies END, and is there an end-of-recording teardown burst that would
/// make "track ended" over-count deaths? This is what sets [`super::CUT_WINDOW_TICKS`].
#[test]
fn probe_death_shape() {
    let films = streamed_films(10);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films on this machine -- death shape not measured");
        return;
    }
    println!(
        "{:<44} {:>7} {:>5} {:>5} {:>4}   destroys by how much of the recording was LEFT",
        "film", "ticks", "body", "destr", "own"
    );
    for (p, _f, idx) in &films {
        let bodies: Vec<_> = idx.tracks.iter().filter(|t| t.entity_type == 11).collect();
        let end = idx.ticks;
        let destroyed = bodies.iter().filter(|t| t.destroyed_tick.is_some()).count();
        let mut hist = [0usize; 6]; // <=1s, <=5s, <=10s, <=30s, <=60s, more
        for t in bodies.iter().filter_map(|t| t.destroyed_tick) {
            let left = end.saturating_sub(t);
            hist[match left {
                0..=60 => 0,
                61..=300 => 1,
                301..=600 => 2,
                601..=1800 => 3,
                1801..=3600 => 4,
                _ => 5,
            }] += 1;
        }
        println!(
            "{:<44} {:>7} {:>5} {:>5} {:>4}   <=1s {:<4} <=5s {:<4} <=10s {:<4} <=30s {:<4} <=60s {:<4} more {}",
            name(p),
            end,
            bodies.len(),
            destroyed,
            idx.owner_group_count(),
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4],
            hist[5]
        );
    }
}

/// The per-player table itself, over the corpus — the sanity read on Part 1.
#[test]
fn probe_player_table() {
    let films = streamed_films(6);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- player table not measured");
        return;
    }
    for (p, f, idx) in &films {
        let t0 = std::time::Instant::now();
        let s = super::compute(f, idx, &super::NoNames);
        println!("\n=== {} ({:.0} ms to compute) ===", name(p), t0.elapsed().as_secs_f32() * 1000.0);
        print!("{}", s.table());
    }
}

// ---------------------------------------------------------------------------------------------
// Part 2: is there a SCORE or KILL counter anywhere in the stream?
// ---------------------------------------------------------------------------------------------

/// Every message of the engine-globals entity types, re-decoded out of the stream.
///
/// The per-tick PLAYER record is not where a score would live — it is fully walked from the
/// disassembly (`sub_1803C7698`) and every one of its fields is accounted for. The engine's
/// per-player GAME state is a replicated entity of its own: the 19 entity types are named in the
/// binary and type **3** is `game-engine-player` (16 of them, exactly the 16 player slots), type
/// **5** is `game-engine-statborg` and type **4** is `game-engine-megalo-team`. If a score or a
/// kill counter is replicated at all, it is in one of those.
pub struct EngineMsg {
    pub tick: u32,
    pub entity_type: u8,
    pub net_id: u32,
    pub dirty_mask: u64,
    pub payload: Vec<u8>,
    pub is_update: bool,
}

/// Collect every message of the given entity types, plus the tick of every player-body destroy
/// (the independently-known death timeline).
pub fn engine_messages(
    f: &film::Film,
    idx: &FilmIndex,
    types: &[u8],
) -> (Vec<EngineMsg>, Vec<u32>) {
    let mut out = Vec::new();
    let mut deaths: Vec<u32> = Vec::new();
    let mut base: Option<u32> = None;
    for (i, rec) in f.data.records.iter().enumerate() {
        let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
        let b = *base.get_or_insert(t.tick.wrapping_sub(i as u32));
        let tick = t.tick.wrapping_sub(b);
        for &(ty, _net) in &t.destroyed {
            if ty == 11 {
                deaths.push(tick);
            }
        }
        for e in &t.entities {
            if types.contains(&e.entity_type) {
                out.push(EngineMsg {
                    tick,
                    entity_type: e.entity_type,
                    net_id: e.net_id,
                    dirty_mask: e.dirty_mask,
                    payload: e.payload.clone(),
                    is_update: e.is_update,
                });
            }
        }
    }
    (out, deaths)
}

/// What the engine-globals entity types actually send: how many messages, how big, and which
/// dirty masks. This is the "is there anything there at all" read, before any sweep.
#[test]
fn probe_engine_entity_census() {
    let films = streamed_films(4);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- engine entity census not measured");
        return;
    }
    for (p, f, idx) in &films {
        let (msgs, deaths) = engine_messages(f, idx, &[0, 1, 2, 3, 4, 5]);
        println!(
            "\n=== {} | {} ticks | {} body destroys | {} engine-globals messages ===",
            name(p),
            idx.ticks,
            deaths.len(),
            msgs.len()
        );
        for ty in [0u8, 1, 2, 3, 4, 5] {
            let m: Vec<&EngineMsg> = msgs.iter().filter(|m| m.entity_type == ty).collect();
            if m.is_empty() {
                continue;
            }
            let ids: std::collections::BTreeSet<u32> = m.iter().map(|m| m.net_id).collect();
            let mut lens: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut masks: std::collections::BTreeMap<u64, usize> = Default::default();
            for x in &m {
                *lens.entry(x.payload.len()).or_default() += 1;
                *masks.entry(x.dirty_mask).or_default() += 1;
            }
            let mut top: Vec<_> = masks.iter().map(|(k, v)| (*v, *k)).collect();
            top.sort_by(|a, b| b.cmp(a));
            println!(
                "  type {ty:>2}: {:>7} msgs ({} create, {} update)  {:>3} entities  payload lens {:?}",
                m.len(),
                m.iter().filter(|m| !m.is_update).count(),
                m.iter().filter(|m| m.is_update).count(),
                ids.len(),
                lens.iter().take(8).collect::<Vec<_>>()
            );
            for (n, mask) in top.iter().take(6) {
                println!("            mask {mask:#018x} x{n}");
            }
        }
    }
}

/// Read `width` bits at bit `offset`, MSB-first, the same bit order the stream uses.
fn bits_at(p: &[u8], offset: usize, width: usize) -> Option<u32> {
    if width == 0 || width > 32 || offset + width > p.len() * 8 {
        return None;
    }
    let mut v = 0u32;
    for i in 0..width {
        let b = offset + i;
        let bit = (p[b / 8] >> (7 - (b % 8))) & 1;
        v = (v << 1) | bit as u32;
    }
    Some(v)
}

/// One candidate (bit offset, width) inside one dirty-mask class.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub offset: usize,
    pub width: usize,
    /// Fraction of consecutive samples (within one entity) that did not go DOWN.
    pub monotone: f64,
    /// How many positive steps there were in total.
    pub steps: usize,
    /// Sum of the positive steps — the candidate's own "total events" claim.
    pub total: i64,
    /// Fraction of those steps that landed within [`TIMING_WINDOW`] ticks of a player-body
    /// destroy. A kill counter must step when somebody died.
    pub timing: f64,
    /// How many entities' total rise equals the death count of the player slot they map to.
    pub slot_death_hits: usize,
    /// PERMUTATION-INVARIANT version of the same oracle: sort the per-entity rises and sort the
    /// per-player death counts, then count how many positions agree. This needs no assumption
    /// about which type-3 entity belongs to which player slot, which is the one piece of the
    /// mapping that is a guess.
    pub rise_multiset_match: usize,
    /// Entities that carried any sample at all.
    pub entities: usize,
}

/// How close a counter step has to be to a body destroy to count as correlated, ticks.
/// Generous on purpose: a score award is not necessarily the same tick as the destroy.
pub const TIMING_WINDOW: u32 = 30;

/// Sweep every (bit offset, width) inside one dirty-mask class for a COUNTER.
///
/// Why per mask class: the dirty block is a flat presence-bit walk, so a field's bit offset
/// depends on which fields are present in that particular message. Within one dirty mask the
/// layout is fixed, which is the only condition under which a fixed-offset sweep means anything.
pub fn sweep_class(
    msgs: &[&EngineMsg],
    deaths: &[u32],
    slot_deaths: &[usize],
    slot_of_entity: &std::collections::HashMap<u32, usize>,
) -> Vec<Candidate> {
    let len = msgs.first().map(|m| m.payload.len()).unwrap_or(0);
    if len == 0 || msgs.len() < 20 {
        return Vec::new();
    }
    let bits = len * 8;
    // Per entity, in tick order.
    let mut by_entity: std::collections::BTreeMap<u32, Vec<&EngineMsg>> = Default::default();
    for m in msgs {
        by_entity.entry(m.net_id).or_default().push(m);
    }
    for v in by_entity.values_mut() {
        v.sort_by_key(|m| m.tick);
    }
    // `deaths` arrives in tick order, so the "did somebody die near here" test is a binary
    // search, not a linear scan over (on one film) 1,662 ticks per candidate step.
    let near_death = |t: u32| -> bool {
        let i = deaths.partition_point(|&d| d + TIMING_WINDOW < t);
        deaths.get(i).map_or(false, |&d| d <= t + TIMING_WINDOW)
    };
    let mut out = Vec::new();
    for offset in 0..bits {
        for width in 1..=20usize.min(bits - offset) {
            let (mut pairs, mut down, mut steps, mut total, mut timed) = (0usize, 0usize, 0usize, 0i64, 0usize);
            let mut hits = 0usize;
            let mut ents = 0usize;
            let mut rises: Vec<i64> = Vec::new();
            for (net, series) in &by_entity {
                let mut prev: Option<u32> = None;
                let mut rise = 0i64;
                let mut any = false;
                for m in series {
                    let Some(v) = bits_at(&m.payload, offset, width) else { continue };
                    any = true;
                    if let Some(p) = prev {
                        pairs += 1;
                        if v < p {
                            down += 1;
                        } else if v > p {
                            steps += 1;
                            total += (v - p) as i64;
                            rise += (v - p) as i64;
                            if near_death(m.tick) {
                                timed += 1;
                            }
                        }
                    }
                    prev = Some(v);
                }
                if any {
                    ents += 1;
                    rises.push(rise);
                    if let Some(&slot) = slot_of_entity.get(net) {
                        if slot < slot_deaths.len() && rise == slot_deaths[slot] as i64 && rise > 0 {
                            hits += 1;
                        }
                    }
                }
            }
            if steps == 0 || pairs == 0 {
                continue;
            }
            rises.sort_unstable_by(|a, b| b.cmp(a));
            let mut want: Vec<i64> = slot_deaths.iter().map(|&d| d as i64).collect();
            want.sort_unstable_by(|a, b| b.cmp(a));
            let ms = rises
                .iter()
                .zip(want.iter())
                .filter(|(a, b)| *a == *b && **b > 0)
                .count();
            out.push(Candidate {
                rise_multiset_match: ms,
                offset,
                width,
                monotone: 1.0 - down as f64 / pairs as f64,
                steps,
                total,
                timing: timed as f64 / steps as f64,
                slot_death_hits: hits,
                entities: ents,
            });
        }
    }
    out
}

/// THE PART 2 MEASUREMENT: is a score or kill counter replicated anywhere in a Reach film?
///
/// Oracles, in order of sharpness:
///
/// 1. **Per-slot deaths.** This reader already counts a player's deaths independently, from the
///    entity destroys. A DEATH counter's total rise per player must equal that number exactly.
///    That is a 16-way simultaneous constraint and essentially nothing passes it by chance.
/// 2. **Conservation.** In Slayer, the sum of everyone's kills equals the sum of everyone's
///    deaths, and the right-hand side is already known.
/// 3. **Timing.** A kill or score step must land near a body destroy.
/// 4. **Monotonicity.** A score or kill counter never goes down over a match.
#[test]
fn probe_counter_sweep() {
    // The per-slot death oracle only has a LEFT-HAND SIDE on a film whose owner groups are
    // bound to player slots, so those films are swept first.
    let mut films = streamed_films(14);
    films.sort_by_key(|(_, f, idx)| {
        let bound = idx.owner_groups().iter().filter(|&&o| idx.slot_of_owner(o).is_some()).count();
        std::cmp::Reverse((bound, f.data.records.len()))
    });
    films.truncate(4);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- counter sweep not measured");
        return;
    }
    for (p, f, idx) in &films {
        let stats = crate::film_stats::compute(f, idx, &crate::film_stats::NoNames);
        let mut slot_deaths = vec![0usize; 16];
        for pl in &stats.players {
            if let Some(s) = pl.slot {
                if (s as usize) < 16 {
                    slot_deaths[s as usize] = pl.deaths;
                }
            }
        }
        let total_deaths: usize = stats.players.iter().map(|p| p.deaths).sum();
        let (msgs, deaths) = engine_messages(f, idx, &[3, 4, 5]);
        println!(
            "\n=== {} | {} ticks | {} deaths (per-slot known for {} of 16) ===",
            name(p),
            idx.ticks,
            total_deaths,
            slot_deaths.iter().filter(|d| **d > 0).count()
        );
        for ty in [3u8, 4, 5] {
            let of_type: Vec<&EngineMsg> = msgs.iter().filter(|m| m.entity_type == ty).collect();
            if of_type.len() < 50 {
                continue;
            }
            // Map each entity to a player slot by its creation ORDER, which for the 16
            // `game-engine-player` entities is the only candidate ordering available.
            let mut ids: Vec<u32> = of_type.iter().map(|m| m.net_id).collect();
            ids.sort_unstable();
            ids.dedup();
            let slot_of: std::collections::HashMap<u32, usize> =
                ids.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            // Group by (dirty mask, payload length): within one class the presence-bit walk puts
            // every field at a fixed offset, which is what makes a fixed-offset sweep legitimate.
            let mut classes: std::collections::BTreeMap<(u64, usize), Vec<&EngineMsg>> = Default::default();
            for m in &of_type {
                classes.entry((m.dirty_mask, m.payload.len())).or_default().push(m);
            }
            let mut big: Vec<((u64, usize), Vec<&EngineMsg>)> =
                classes.into_iter().filter(|(_, v)| v.len() >= 100).collect();
            big.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
            for ((mask, plen), cls) in big.into_iter().take(5) {
                let mut cands = sweep_class(&cls, &deaths, &slot_deaths, &slot_of);
                if cands.is_empty() {
                    println!("  type {ty} mask {mask:#x} len {plen}: {} msgs, no candidate ever CHANGED value", cls.len());
                    continue;
                }
                // Rank: a counter must be monotone first, then explain the deaths, then be timed.
                cands.sort_by(|a, b| {
                    let key = |c: &Candidate| {
                        (
                            c.slot_death_hits.max(c.rise_multiset_match),
                            (c.monotone * 1e6) as i64,
                            -((c.total - total_deaths as i64).abs()),
                            (c.timing * 1e6) as i64,
                        )
                    };
                    key(b).cmp(&key(a))
                });
                println!(
                    "  type {ty} mask {mask:#018x} len {plen}: {} msgs, {} entities, {} candidates",
                    cls.len(),
                    cls.iter().map(|m| m.net_id).collect::<std::collections::BTreeSet<_>>().len(),
                    cands.len()
                );
                for c in cands.iter().take(4) {
                    println!(
                        "      off {:>3} w {:>2}  mono {:>7.4}  steps {:>7}  total {:>8} (deaths {})  timed {:>6.3}  slot-death {:>2}/{}  multiset {:>2}",
                        c.offset, c.width, c.monotone, c.steps, c.total, total_deaths, c.timing, c.slot_death_hits, c.entities, c.rise_multiset_match
                    );
                }
                // The runner-up margin on the sharpest oracle. A winner with no margin is not a
                // result, so the number is printed whether it is good news or not.
                let key = |c: &Candidate| c.slot_death_hits.max(c.rise_multiset_match);
                let best = key(&cands[0]);
                let runner = cands.get(1).map(key).unwrap_or(0);
                println!(
                    "      -> best death-oracle hits {best}, runner-up {runner} (margin {})",
                    best as i64 - runner as i64
                );
                // The STRICT gate, which is what a real counter would have to pass: never goes
                // down, and the total it claims is within 20% of the deaths the recording
                // independently proves. Printed whether or not anything survives it, because an
                // empty line here IS the result.
                let strict: Vec<&Candidate> = cands
                    .iter()
                    .filter(|c| {
                        c.monotone >= 0.9999
                            && c.total > 0
                            && ((c.total - total_deaths as i64).abs() as f64) <= 0.2 * total_deaths as f64
                    })
                    .collect();
                match strict.first() {
                    Some(c) => println!(
                        "      -> STRICT (never decreases, total within 20% of the deaths): off {} w {} total {} timed {:.3}, {} survivor(s)",
                        c.offset, c.width, c.total, c.timing, strict.len()
                    ),
                    None => println!("      -> STRICT (never decreases, total within 20% of the deaths): NOTHING survives"),
                }
            }
        }
    }
}

/// What are the per-tick SUBSYSTEM CHANNELS, and could a Megalo score be in one?
///
/// Exactly one channel is present on 1.000 ticks of every real film, and the engine has a
/// `game_variant_state_codec` (`haloreach.dll+0x3C8870`) on this path. The envelope decoder
/// SKIPS the channel payload, so this is the last unexplored region of the record and the only
/// remaining candidate home for a score after the engine-globals entities came up empty.
///
/// This probe reports the channel's id and the distribution of its bit length, and sweeps the
/// payload for a per-player counter the same way `sweep_class` does the entity payloads — one
/// length class at a time, since a fixed offset only means anything inside a fixed layout.
#[test]
fn probe_subsystem_channel() {
    let mut films = streamed_films(14);
    films.sort_by_key(|(_, f, idx)| {
        let bound = idx.owner_groups().iter().filter(|&&o| idx.slot_of_owner(o).is_some()).count();
        std::cmp::Reverse((bound, f.data.records.len()))
    });
    films.truncate(3);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- subsystem channel not measured");
        return;
    }
    for (p, f, idx) in &films {
        let stats = crate::film_stats::compute(f, idx, &crate::film_stats::NoNames);
        let mut slot_deaths = vec![0usize; 16];
        for pl in &stats.players {
            if let Some(s) = pl.slot {
                if (s as usize) < 16 {
                    slot_deaths[s as usize] = pl.deaths;
                }
            }
        }
        let total_deaths: usize = stats.players.iter().map(|pl| pl.deaths).sum();
        // id -> (count, bit-length histogram), plus the payloads by length class.
        let mut ids: std::collections::BTreeMap<u32, usize> = Default::default();
        let mut lens: std::collections::BTreeMap<u32, usize> = Default::default();
        let mut by_len: std::collections::BTreeMap<u32, Vec<(u32, Vec<u8>)>> = Default::default();
        let mut deaths: Vec<u32> = Vec::new();
        let mut base: Option<u32> = None;
        let mut used_hist = [0usize; 5];
        for (i, rec) in f.data.records.iter().enumerate() {
            let body = rec.body(&f.bytes);
            let Ok(t) = film_decode::decode_record(body, &idx.params) else { continue };
            let b = *base.get_or_insert(t.tick.wrapping_sub(i as u32));
            let tick = t.tick.wrapping_sub(b);
            for &(ty, _n) in &t.destroyed {
                if ty == 11 {
                    deaths.push(tick);
                }
            }
            used_hist[(t.channels_used as usize).min(4)] += 1;
            for k in 0..t.channels_used as usize {
                let (id, bits, off) = t.channels[k];
                *ids.entry(id).or_default() += 1;
                *lens.entry(bits).or_default() += 1;
                // Copy the payload out, byte-aligned from its own bit offset.
                if bits > 0 && bits <= 4096 {
                    let mut v = vec![0u8; ((bits + 7) / 8) as usize];
                    for j in 0..bits as usize {
                        let bi = off as usize + j;
                        if bi / 8 >= body.len() {
                            break;
                        }
                        let bit = (body[bi / 8] >> (7 - (bi % 8))) & 1;
                        v[j / 8] |= bit << (7 - (j % 8));
                    }
                    by_len.entry(bits).or_default().push((tick, v));
                }
            }
        }
        println!(
            "\n=== {} | {} ticks | {} deaths | channels per tick {:?} ===",
            name(p),
            idx.ticks,
            total_deaths,
            used_hist
        );
        println!("  subsystem ids: {:?}", ids.iter().map(|(k, v)| (format!("{k:#010x}"), *v)).collect::<Vec<_>>());
        let mut top: Vec<(usize, u32)> = lens.iter().map(|(k, v)| (*v, *k)).collect();
        top.sort_by(|a, b| b.cmp(a));
        println!("  bit lengths (count, bits): {:?}", top.iter().take(10).collect::<Vec<_>>());
        // Sweep the biggest length class.
        for (_n, bits) in top.iter().take(2) {
            let Some(cls) = by_len.get(bits) else { continue };
            if cls.len() < 40 || *bits == 0 {
                continue;
            }
            // One pseudo-entity: the channel is per TICK, not per player, so the sweep looks for
            // a monotone counter anywhere in it rather than a per-entity series.
            let msgs: Vec<EngineMsg> = cls
                .iter()
                .map(|(t, v)| EngineMsg {
                    tick: *t,
                    entity_type: 255,
                    net_id: 0,
                    dirty_mask: 0,
                    payload: v.clone(),
                    is_update: true,
                })
                .collect();
            let refs: Vec<&EngineMsg> = msgs.iter().collect();
            let slot_of: std::collections::HashMap<u32, usize> = Default::default();
            let mut cands = sweep_class(&refs, &deaths, &slot_deaths, &slot_of);
            cands.sort_by(|a, b| {
                let key = |c: &Candidate| {
                    (
                        (c.monotone * 1e6) as i64,
                        -((c.total - total_deaths as i64).abs()),
                        (c.timing * 1e6) as i64,
                    )
                };
                key(b).cmp(&key(a))
            });
            println!("  channel class {bits} bits, {} samples, {} candidates", cls.len(), cands.len());
            for c in cands.iter().take(6) {
                println!(
                    "      off {:>4} w {:>2}  mono {:>7.4}  steps {:>7}  total {:>8} (deaths {})  timed {:>6.3}",
                    c.offset, c.width, c.monotone, c.steps, c.total, total_deaths, c.timing
                );
            }
        }
    }
}

/// The SCOREBOARD gate: entity type 5 is `game-engine-statborg`, the engine's own score store,
/// and this is the test that says the decode is right rather than merely plausible.
///
/// Three oracles, all free:
///
/// 1. **The stream states the dirty mask twice** — once as the message's own 64-bit header field
///    and once as the per-field presence bits. A wrong walk desynchronises and the rebuilt mask
///    stops matching.
/// 2. **The payload is consumed to within the writer's byte padding** (0..7 bits left).
/// 3. **The engine clamps every statistic to ±30000**, so a value outside that is a decode error
///    rather than a big score.
///
/// Asserts the pass rate rather than only printing it, because this one is load-bearing: a wrong
/// score is worse than no score.
#[test]
fn the_statborg_score_decodes_against_both_free_oracles() {
    let films = streamed_films(10);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- the statborg decode was not exercised");
        return;
    }
    let (mut seen, mut ok) = (0usize, 0usize);
    for (p, f, idx) in &films {
        let st = crate::film_stats::compute(f, idx, &crate::film_stats::NoNames);
        seen += idx.statborg_messages;
        ok += idx.statborg_decoded;
        println!(
            "{:<46} statborg {:>5} msgs, {:>5} passed both oracles ({:>5.1}%), {} slot(s) scored, scores total {}",
            name(p),
            idx.statborg_messages,
            idx.statborg_decoded,
            100.0 * idx.statborg_decoded as f64 / idx.statborg_messages.max(1) as f64,
            st.coverage.score_slots,
            st.scoreboard.iter().filter(|r| r.kind == "player").map(|r| r.score).sum::<i32>()
        );
        assert_eq!(
            idx.statborg_messages, idx.statborg_decoded,
            "{}: a statborg message failed an oracle -- the score walk is wrong, and a wrong score is worse than none",
            name(p)
        );
    }
    println!("TOTAL statborg {seen} messages, {ok} passed both oracles");
    assert!(seen > 0, "no statborg messages found at all");
    assert_eq!(seen, ok);
}

/// Cross-film summary of what Part 1 can actually report, for the honesty numbers in the UI.
#[test]
fn probe_corpus_coverage() {
    let films = streamed_films(40);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- corpus coverage not measured");
        return;
    }
    let (mut bodies, mut weap_bodies, mut films_with_weap, mut films_with_score) = (0usize, 0usize, 0usize, 0usize);
    println!(
        "{:<46} {:>6} {:>5} {:>4} {:>5} {:>6} {:>6} {:>6}",
        "film", "ticks", "body", "own", "named", "weap%", "class%", "scored"
    );
    for (p, f, idx) in &films {
        let s = crate::film_stats::compute(f, idx, &crate::film_stats::NoNames);
        let c = &s.coverage;
        bodies += c.bodies;
        weap_bodies += c.weapon_bodies;
        if c.weapon_bodies > 0 {
            films_with_weap += 1;
        }
        if c.score_slots > 0 {
            films_with_score += 1;
        }
        let gait: u32 = s.players.iter().map(|x| x.gait_ticks).sum();
        let body: u32 = s.players.iter().map(|x| x.body_ticks).sum();
        println!(
            "{:<46} {:>6} {:>5} {:>4} {:>5} {:>5.1}% {:>5.1}% {:>6}",
            name(p),
            idx.ticks,
            c.bodies,
            c.owner_groups,
            c.named_players,
            100.0 * c.weapon_bodies as f64 / c.bodies.max(1) as f64,
            100.0 * gait as f64 / body.max(1) as f64,
            c.score_slots
        );
    }
    println!(
        "\nTOTAL {} films: {} bodies, {} ({:.1}%) ever named a weapon; {} of {} films name one at all; {} of {} films carry a score",
        films.len(),
        bodies,
        weap_bodies,
        100.0 * weap_bodies as f64 / bodies.max(1) as f64,
        films_with_weap,
        films.len(),
        films_with_score,
        films.len()
    );
}

/// What the statborg fields actually DO across the corpus.
///
/// Three questions, each of which was either claimed too strongly or left unmeasured:
///
/// 1. **Is the per-update score series monotone within a clip?** This is the real behavioural
///    check — "final score equals the number of updates" is not, because a score is a cumulative
///    total replicated as an ABSOLUTE value, so that equality only holds for a clip that happens
///    to start at zero and gain one point per update.
/// 2. **Is field 1 (`carried_score`) ever SENT?** A field that is always zero has the same shape
///    as a misread offset, so the presence bit has to be counted, not just the value.
/// 3. **Is the sign bit of the 16-bit statistics ever exercised?**
#[test]
fn probe_statborg_behaviour() {
    let films = streamed_films(40);
    if films.is_empty() {
        eprintln!("film_stats: no streamed films -- statborg behaviour not measured");
        return;
    }
    // Field presence, counted from the REBUILT mask (which equals the message's own).
    let mut present = [0usize; 4];
    let (mut msgs, mut updates) = (0usize, 0usize);
    // Value ranges.
    let (mut min_round, mut max_round) = (i32::MAX, i32::MIN);
    let (mut min_carry, mut max_carry) = (i32::MAX, i32::MIN);
    let (mut min_won, mut max_won) = (i32::MAX, i32::MIN);
    let mut finals = 0usize;
    // Monotonicity of the per-slot round-score series, within one clip.
    let (mut steps, mut up, mut down, mut flat) = (0usize, 0usize, 0usize, 0usize);
    let mut drops: Vec<(String, u8, i16, i16, u32)> = Vec::new();
    // A drop TO ZERO is a round reset, not a decrement. The two have to be counted apart or the
    // monotonicity answer is meaningless.
    let (mut reset, mut decrement) = (0usize, 0usize);
    let mut carry_films: std::collections::BTreeSet<String> = Default::default();
    // The equality the earlier claim rested on.
    let (mut eq, mut rows) = (0usize, 0usize);
    let mut neg_rows = 0usize;

    for (p, f, idx) in &films {
        let s = crate::film_stats::compute(f, idx, &crate::film_stats::NoNames);
        for r in &s.scoreboard {
            rows += 1;
            if r.score == r.updates as i32 {
                eq += 1;
            }
            if r.score < 0 || r.round_score < 0 || r.carried_score < 0 {
                neg_rows += 1;
            }
        }
        // Re-walk the raw messages so the PRESENCE bits can be counted, not just the folded
        // values -- a field that is never sent and a field that is sent as zero look identical
        // once folded.
        for (i, rec) in f.data.records.iter().enumerate() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
            let _ = i;
            for e in t.entities.iter().filter(|e| e.entity_type == 5) {
                let Some(sb) = film_decode::decode_statborg(e, &idx.params) else { continue };
                if !sb.trustworthy(e.dirty_mask) {
                    continue;
                }
                msgs += 1;
                if e.is_update {
                    updates += 1;
                }
                for b in 0..4 {
                    if sb.rebuilt_mask & (1 << b) != 0 {
                        present[b] += 1;
                    }
                }
                if let Some(v) = sb.round_score {
                    min_round = min_round.min(v as i32);
                    max_round = max_round.max(v as i32);
                }
                if let Some(v) = sb.total_score {
                    min_carry = min_carry.min(v as i32);
                    max_carry = max_carry.max(v as i32);
                    if v != 0 {
                        carry_films.insert(name(p));
                    }
                }
                if let Some(v) = sb.rounds_won {
                    min_won = min_won.min(v as i32);
                    max_won = max_won.max(v as i32);
                }
                if sb.is_final == Some(true) {
                    finals += 1;
                }
            }
        }
        // Monotonicity, per slot, within this clip.
        for (slot, _) in idx.statborg_slots() {
            let Some(series) = idx.statborg.get(slot as usize) else { continue };
            let mut prev: Option<i16> = None;
            for (tick, s) in series {
                let Some(v) = s.round_score else { continue };
                if let Some(q) = prev {
                    steps += 1;
                    match v.cmp(&q) {
                        std::cmp::Ordering::Greater => up += 1,
                        std::cmp::Ordering::Less => {
                            down += 1;
                            if v == 0 {
                                reset += 1;
                            } else {
                                decrement += 1;
                            }
                            if drops.len() < 20 {
                                drops.push((name(p), slot, q, v, *tick));
                            }
                        }
                        std::cmp::Ordering::Equal => flat += 1,
                    }
                }
                prev = Some(v);
            }
        }
    }
    println!("\n--- statborg behaviour over {} films ---", films.len());
    println!("messages that passed every oracle: {msgs} ({updates} updates, {} creations)", msgs - updates);
    println!(
        "field PRESENCE (how often each field was actually SENT):\n  \
         0 round_score   {:>6}  ({:>5.1}%)\n  \
         1 carried_score {:>6}  ({:>5.1}%)\n  \
         2 rounds_won    {:>6}  ({:>5.1}%)\n  \
         3 final         {:>6}  ({:>5.1}%)  of which {finals} set TRUE",
        present[0], 100.0 * present[0] as f64 / msgs.max(1) as f64,
        present[1], 100.0 * present[1] as f64 / msgs.max(1) as f64,
        present[2], 100.0 * present[2] as f64 / msgs.max(1) as f64,
        present[3], 100.0 * present[3] as f64 / msgs.max(1) as f64,
    );
    let rng = |mn: i32, mx: i32| if mn == i32::MAX { "never sent".to_string() } else { format!("{mn} .. {mx}") };
    println!(
        "value ranges: round_score {}   carried_score {}   rounds_won {}",
        rng(min_round, max_round), rng(min_carry, max_carry), rng(min_won, max_won)
    );
    println!(
        "round_score series WITHIN a clip: {steps} consecutive pairs -- {up} up, {flat} unchanged, \
         {down} DOWN ({reset} of them a reset TO ZERO = a round boundary, {decrement} a genuine decrement)"
    );
    println!("films whose carried_score is ever non-zero: {carry_films:?}");
    for (f, slot, a, b, t) in drops.iter().take(10) {
        println!("  drop: {f} slot {slot} {a} -> {b} at tick {t}");
    }
    println!(
        "scoreboard rows: {rows}; score == update count on {eq} ({:.0}%) -- NOT a validation, see the note; rows with any negative value: {neg_rows}",
        100.0 * eq as f64 / rows.max(1) as f64
    );
}
