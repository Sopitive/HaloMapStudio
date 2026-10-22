//! #film-riders  A film's RIDERS: who sits in which seat of which vehicle, and when.
//!
//! **The recording states it.** A seated unit stops sending a position of its own -- its place in
//! the world is its parent's -- and the stream says so through base field 12, the object-to-object
//! attachment record: a 27-bit parent reference, an 8-bit attach point, a local offset and
//! orientation, and a 6-bit SEAT INDEX (`film_decode::BaseAttachment`, `sub_1803D7E58`). Measured
//! over the 69-film corpus (`probes::census_ride_intervals`):
//!
//! * the parent of a seated biped is a type-15 entity (a vehicle, or a Warthog's turret, which is
//!   a `vehi` of its own) in all 49 stated intervals;
//! * the biped reports **almost no position** while it is parented (a handful of samples in a
//!   ride that lasts minutes) and the few it does send are the stale boarding spot, up to 189 wu
//!   from the vehicle by the end. A renderer that trusts the body's own position therefore leaves
//!   the player standing where he stepped up while the vehicle drives away -- the report this
//!   module answers;
//! * the displacement detector (`film_anim::find_rides`) overlapped none of those intervals, for
//!   the same reason: there is no body displacement to match.
//!
//! The seat index is an index into the PARENT's `unit_seat` block (`reach_seats::UnitSeats`), so
//! the seat's own `label` is the jmad mode (`warthog_d`) and its `marker name` is the model
//! marker the body is parented to -- the two things test mode already uses.

use crate::film_anim::Ride;

/// How long after a stated link ends the body is still given the seat's EXIT clip, ticks.
///
/// An upper bound for the search only: the clip's own authored length decides how much of it
/// plays (the shipped `<seat>:exit` clips are 30-60 frames at 30 fps, i.e. 1-2 s).
pub const EXIT_SEARCH_TICKS: u32 = 240;

/// What the recording says about one body's seat at one tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SeatState {
    /// Linked to a vehicle's seat. `age` = ticks since the link began (drives the boarding clip).
    Seated { ride: Ride, age: u32 },
    /// The link ended `age` ticks ago (drives the exit clip). Only a STATED ride leaves one: a
    /// detected ride's end is a guess and gets no exit clip.
    Leaving { ride: Ride, age: u32 },
}

/// The seat state of `biped` at `tick`, from the film's accepted rides. A pure function of the
/// tick, so a seek lands on the correct seated state with no event having to replay.
pub fn state_at(rides: &[Ride], biped: u32, tick: u32) -> Option<SeatState> {
    if let Some(r) = rides.iter().find(|r| r.biped_net_id == biped && r.covers(tick)) {
        return Some(SeatState::Seated { ride: *r, age: tick - r.from_tick });
    }
    rides
        .iter()
        .filter(|r| r.biped_net_id == biped && r.seat.is_some() && r.to_tick < tick)
        .filter(|r| tick - (r.to_tick + 1) < EXIT_SEARCH_TICKS)
        .max_by_key(|r| r.to_tick)
        .map(|r| SeatState::Leaving { ride: *r, age: tick - (r.to_tick + 1) })
}

/// A body's seat, resolved against the open map.
#[derive(Clone, Debug, PartialEq)]
pub struct Rider {
    pub state: SeatState,
    /// `mode` = the seat block's `label` (the jmad mode); `vehicle` = the vehicle's tag leaf.
    pub seat: crate::film_anim::Seat,
    /// The `vehi` tag whose model the seat marker lives in (the turret's, for a gunner).
    pub vehicle_tag: Option<u32>,
    /// The seat's index in that tag's seat block; `None` for a detected ride, which states none.
    pub seat_index: Option<usize>,
}

impl Rider {
    pub fn ride(&self) -> Ride {
        match self.state {
            SeatState::Seated { ride, .. } | SeatState::Leaving { ride, .. } => ride,
        }
    }
    pub fn seated(&self) -> bool {
        matches!(self.state, SeatState::Seated { .. })
    }
}

/// The clips a rider's graph authors for one seat, for the case where the weapon class the body
/// carries has no clip of its own. `play_vehicle::phase_clip_candidates` asks for
/// `<seat>:<class>:enter`, `<seat>:any:enter` and `<seat>:enter`; the shipped graphs author the
/// per-class ones for SOME classes only (`wraith_d` has `rifle` and no `pistol`), and a body whose
/// class is not among them would otherwise be drawn standing at the seat marker. Any authored clip
/// for the seat beats that, and the engine's own animation inheritance resolves a missing class
/// the same way (to the nearest authored one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeatClips {
    pub idle: Option<String>,
    pub enter: Option<String>,
    pub exit: Option<String>,
}

impl SeatClips {
    /// One pass over the graph's animation names. `<seat>:any:<leaf>` and the class-less
    /// `<seat>:<leaf>` win over a named class, so the choice does not depend on name order.
    pub fn find(graph: &crate::reach_actor::AnimGraph, mode: &str) -> SeatClips {
        let prefix = format!("{mode}:");
        let pick = |leaf: &str| -> Option<String> {
            let suffix = format!(":{leaf}");
            let mut best: Option<(u8, &str)> = None;
            for a in &graph.anims {
                let n = a.name.as_str();
                if !n.starts_with(&prefix) || !n.ends_with(&suffix) {
                    continue;
                }
                let rank = if n == format!("{mode}:{leaf}") {
                    0
                } else if n == format!("{mode}:any:{leaf}") {
                    1
                } else {
                    2
                };
                if best.map_or(true, |(r, _)| rank < r) {
                    best = Some((rank, n));
                }
            }
            best.map(|(_, n)| n.to_string())
        };
        SeatClips { idle: pick("idle"), enter: pick("enter"), exit: pick("exit") }
    }
}

/// The first of `cands` that the graph authors EXACTLY, as a BASE clip, with its length in
/// seconds.
///
/// Not `play_vehicle::first_authored_clip`: that goes through `AnimGraph::find`, which falls back
/// to a SUBSTRING match, so `mongoose_d:rifle:exit` -- a clip the Mongoose does not author --
/// resolves to `mongoose_d:rifle:exit_pose`, a one-frame OVERLAY, and the body's exit "lasts" two
/// ticks. A boarding or exit clip is a base clip authored under exactly the name asked for.
pub fn exact_base_clip(graph: &crate::reach_actor::AnimGraph, cands: &[String]) -> Option<(String, f32)> {
    cands.iter().find_map(|c| {
        let a = graph
            .anims
            .iter()
            .find(|a| a.name.eq_ignore_ascii_case(c) && a.anim_type == crate::reach_jmad::AnimType::Base)?;
        Some((a.name.clone(), a.frame_count.max(0) as f32 / crate::film_anim::CLIP_FPS))
    })
}

/// How far through the exit the body is, 0..1, as a smoothstep: the weight that moves the body
/// from the seat marker to the place the recording puts it once the link is gone. A snap at the
/// end of the exit clip would teleport the body by however far the vehicle moved meanwhile.
pub fn exit_blend(age_ticks: f32, len_ticks: f32) -> f32 {
    if len_ticks <= 0.0 {
        return 1.0;
    }
    let t = (age_ticks / len_ticks).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod probes {
    use crate::{film, film_decode};

    #[test]
    #[ignore]
    fn census_ride_intervals() {
        use crate::film_anim;
        let films = film::find_finished_films();
        let (mut ints, mut covered, mut ticks_total, mut ticks_cov) = (0usize, 0usize, 0u64, 0u64);
        let mut seats: std::collections::BTreeMap<(u8, u8), usize> = Default::default();
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let by: std::collections::HashMap<u32, usize> = idx.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
            let rides = film_anim::accepted_rides(&idx);
            let mut nint = 0;
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                for (k, (tick, a)) in t.attachments.iter().enumerate() {
                    let Some(a) = a else { continue };
                    let Some(&vi) = by.get(&a.parent) else { continue };
                    let v = &idx.tracks[vi];
                    if v.entity_type != 15 { continue; }
                    let end = t.attachments.get(k + 1).map(|x| x.0).or(t.destroyed_tick).unwrap_or(t.last_tick);
                    ints += 1; nint += 1;
                    *seats.entry((a.seat.unwrap_or(255), a.node)).or_default() += 1;
                    let own: Vec<_> = t.positions.iter().filter(|(q, _)| *q >= *tick && *q <= end).collect();
                    let mut cov = 0u64;
                    for q in *tick..=end { if rides.iter().any(|r| r.biped_net_id == t.net_id && r.vehicle_net_id == a.parent && r.covers(q)) { cov += 1; } }
                    ticks_total += (end - tick + 1) as u64; ticks_cov += cov;
                    if cov > 0 { covered += 1; }
                    // distance own pos to vehicle pos
                    let mut ds = vec![];
                    for (q, bp) in &own { if let Some(vp) = v.sample(*q) { ds.push(((bp[0]-vp[0]).powi(2)+(bp[1]-vp[1]).powi(2)+(bp[2]-vp[2]).powi(2)).sqrt()); } }
                    ds.sort_by(|a,b| a.partial_cmp(b).unwrap());
                    let med = ds.get(ds.len()/2).copied();
                    // before boarding: own positions in the 120 ticks before
                    let pre: Vec<_> = t.positions.iter().filter(|(q, _)| *q + 120 >= *tick && *q < *tick).collect();
                    println!("  {} biped {} -> veh {} seat {:?} node {} ticks {}..{} ({:.1}s) ownpos={} (pre {}) med|own-veh|={:?} vehpos={} detector_cov={}/{}",
                        p.file_name().unwrap().to_string_lossy().chars().take(24).collect::<String>(), t.net_id, a.parent, a.seat, a.node, tick, end, (end-tick) as f32/60.0, own.len(), pre.len(), med, v.positions.len(), cov, end-tick+1);
                }
            }
            let _ = nint;
        }
        println!("intervals {ints}; with detector overlap {covered}; ticks {ticks_total} covered by detector {ticks_cov}");
        println!("(seat,node) hist {:?}", seats);
    }

    /// Corpus-wide totals of the stated rides (print-only): how many films hold any, how many rides,
    /// which seat indices, how many end without a clearing statement, and how many of the exit
    /// events (kind 19) are backed by a stated link at all.
    #[test]
    #[ignore = "print-only probe"]
    fn census_stated_rides() {
        use crate::film_anim::{accepted_rides, recorded_rides};
        let films = film::find_finished_films();
        let (mut with, mut rides, mut open, mut bodies, mut vehicles, mut ticks, mut overlap) = (0usize, 0usize, 0usize, 0usize, 0usize, 0u64, 0u64);
        let (mut exits, mut exits_backed) = (0usize, 0usize);
        let mut seats: std::collections::BTreeMap<u8, usize> = Default::default();
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            bodies += idx.tracks.iter().filter(|t| t.entity_type == 11).count();
            vehicles += idx.tracks.iter().filter(|t| t.entity_type == 15).count();
            let r = recorded_rides(&idx);
            if r.is_empty() {
                continue;
            }
            with += 1;
            rides += r.len();
            let detected: Vec<_> = accepted_rides(&idx).into_iter().filter(|x| x.seat.is_none()).collect();
            for x in &r {
                *seats.entry(x.seat.unwrap()).or_default() += 1;
                ticks += (x.to_tick - x.from_tick + 1) as u64;
                let b = idx.tracks.iter().find(|t| t.net_id == x.biped_net_id).unwrap();
                open += b.attachments.iter().all(|(k, a)| *k <= x.to_tick || a.is_none()) as usize
                    * (x.to_tick >= b.destroyed_tick.unwrap_or(b.last_tick)) as usize;
                overlap += detected.iter().filter(|d| d.biped_net_id == x.biped_net_id).map(|d| d.to_tick.min(x.to_tick).saturating_sub(d.from_tick.max(x.from_tick)) as u64).sum::<u64>();
            }
            for e in idx.events.iter().filter(|e| e.kind == crate::film_action::EV_EXIT_VEHICLE) {
                exits += 1;
                exits_backed += r.iter().any(|x| Some(x.biped_net_id) == e.subject && e.tick + 120 >= x.from_tick && e.tick <= x.to_tick + 120) as usize;
            }
        }
        println!(
            "{} finished films, {bodies} biped tracks, {vehicles} vehicle tracks; {with} films hold stated rides: {rides} rides ({ticks} ticks, {open} ending with the body), seats {seats:?}; \
             detector rides overlapping them: {overlap} ticks; exit events {exits}, backed by a stated ride within 2 s: {exits_backed}",
            films.len()
        );
    }

    #[test]
    #[ignore]
    fn census_exit_events_vs_links() {
        use crate::film_action as fa;
        let films = film::find_finished_films();
        let mut tot: std::collections::BTreeMap<u8, usize> = Default::default();
        let (mut exits, mut with_link, mut link_any, mut no_track) = (0usize, 0usize, 0usize, 0usize);
        let mut examples = 0;
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for e in &idx.events { if matches!(e.kind, 8 | 19 | 40 | 41) { *tot.entry(e.kind).or_default() += 1; } }
            for e in idx.events.iter().filter(|e| e.kind == fa::EV_EXIT_VEHICLE) {
                exits += 1;
                let Some(subj) = e.subject else { continue };
                let Some(tr) = idx.tracks.iter().find(|t| t.net_id == subj) else { no_track += 1; continue };
                if !tr.attachments.is_empty() { link_any += 1; }
                // last statement at or before the event tick (+ small slack)
                let st = tr.attachments.iter().filter(|(k, _)| *k <= e.tick + 30).last();
                let linked = st.map(|(_, a)| a.map(|a| Some(a.parent) == e.other).unwrap_or(false)).unwrap_or(false);
                // or a link that ended within +-120 ticks of the event
                let near = tr.attachments.windows(2).any(|w| w[0].1.map(|a| Some(a.parent) == e.other).unwrap_or(false) && w[1].0.abs_diff(e.tick) <= 120);
                if linked || near { with_link += 1; } else if examples < 8 {
                    examples += 1;
                    println!("  unlinked exit: film {} t={} subj {} veh {:?} seat {:?} stmts {:?}", p.file_name().unwrap().to_string_lossy().chars().take(30).collect::<String>(), e.tick, subj, e.other, e.seat, tr.attachments.iter().map(|(k,a)| (*k, a.map(|a| a.parent))).take(6).collect::<Vec<_>>());
                }
            }
        }
        println!("event kinds {:?}", tot);
        println!("exit events {exits}: subject track missing {no_track}; subject has ANY parent statement {link_any}; linked to that vehicle at/near the exit {with_link}");
    }

    #[test]
    #[ignore]
    fn census_field12_coverage() {
        let films = film::find_finished_films();
        let (mut msgs, mut b12, mut got12, mut creat12, mut creat_got) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut stops: std::collections::BTreeMap<Option<u32>, u64> = Default::default();
        let mut some_by_update = (0u64, 0u64);
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
                for e in t.entities.iter().filter(|e| e.entity_type == 11) {
                    msgs += 1;
                    if e.dirty_mask & (1 << 12) == 0 { continue; }
                    b12 += 1;
                    if !e.is_update { creat12 += 1; }
                    let Some(st) = film_decode::decode_entity_payload(e, &idx.params) else { *stops.entry(Some(999)).or_default() += 1; continue };
                    if st.rebuilt_mask & (1 << 12) != 0 {
                        got12 += 1;
                        if !e.is_update { creat_got += 1; }
                        if st.attach.is_some() { if e.is_update { some_by_update.0 += 1 } else { some_by_update.1 += 1 } }
                    } else { *stops.entry(st.stopped_at_unknown_field).or_default() += 1; }
                }
            }
        }
        println!("biped msgs {msgs}; bit12 set {b12} (of which creations {creat12}); field 12 walked {got12} (creations {creat_got}); Some(parent): updates {} creations {}", some_by_update.0, some_by_update.1);
        println!("walk stopped before 12, by stopped_at_unknown_field: {:?}", stops);
    }

    #[test]
    #[ignore]
    fn census_gate_latch_vs_field12() {
        let films = film::find_finished_films();
        // (latched_at, mask has bit2, mask has bit12, rebuilt has bit12) -> count, biped updates only
        let mut h: std::collections::BTreeMap<(Option<u32>, bool, bool, bool), u64> = Default::default();
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
                for e in t.entities.iter().filter(|e| e.entity_type == 11 && e.is_update && e.dirty_mask & 0xFFFFC != 0) {
                    let Some(st) = film_decode::decode_entity_payload(e, &idx.params) else { continue };
                    *h.entry((st.latched_at, e.dirty_mask & 4 != 0, e.dirty_mask & (1 << 12) != 0, st.rebuilt_mask & (1 << 12) != 0)).or_default() += 1;
                }
            }
        }
        println!("biped updates whose mask names any base field 2..19: (latched_at, mask bit2, mask bit12, rebuilt bit12) -> n");
        for (k, v) in &h { println!("  {:?} -> {v}", k); }
    }
}

// ---------------------------------------------------------------------------------------------
// gates
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod gates {
    use super::*;
    use crate::film_anim::{accepted_rides, recorded_rides, BIPED_TYPE, VEHICLE_TYPE};
    use crate::film_decode::{
        decode_entity_payload, BaseAttachment, DecodeParams, EntityMessage, EntityTrack, FilmIndex,
    };

    /// Pack a bit list (value, width) into a payload, MSB first -- the stream's own order.
    fn pack(bits: &[(u64, u32)]) -> Vec<u8> {
        let mut v: Vec<bool> = Vec::new();
        for &(val, n) in bits {
            for i in (0..n).rev() {
                v.push((val >> i) & 1 != 0);
            }
        }
        let mut bytes = vec![0u8; (v.len() + 7) / 8];
        for (i, b) in v.iter().enumerate() {
            if *b {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        bytes
    }

    fn params() -> DecodeParams {
        DecodeParams { cv3_axis_bits: [15, 16, 14], cv3_total_bits: 45, tagref_bits: 14, update_prefix_bits: 1, ..Default::default() }
    }

    fn att(parent: u32, seat: Option<u8>) -> BaseAttachment {
        BaseAttachment { parent, seat, ..Default::default() }
    }

    fn vehicle(net: u32) -> EntityTrack {
        EntityTrack { net_id: net, entity_type: VEHICLE_TYPE, first_tick: 0, last_tick: 2000, ..Default::default() }
    }

    fn biped(net: u32, attachments: Vec<(u32, Option<BaseAttachment>)>) -> EntityTrack {
        EntityTrack { net_id: net, entity_type: BIPED_TYPE, first_tick: 0, last_tick: 2000, attachments, ..Default::default() }
    }

    // ---- the base-field walk ----------------------------------------------------------------

    /// A biped update: position, then the field-2 gate CLEAR, then the field-8 gate SET with the
    /// attachment at field 12. `sub_1803B05C8` ASSIGNS its latch at every gate, so the clear gate
    /// at 2 skips fields 2..7 only and gate 8 is still read. The walk used to stop at the first
    /// clear gate, which hid field 12 on 764 of the 1,641 corpus messages that carry it.
    fn riding_update(parent: u32, seat: u64) -> EntityMessage {
        let bits: Vec<(u64, u32)> = vec![
            (0, 1),                 // update prefix
            (1, 1), (1, 1),         // field 0 present, compressed vector flag
            (111, 15), (222, 16), (333, 14),
            (0, 1),                 // field 1 absent
            (0, 1),                 // GATE at 2: CLEAR -> fields 2..7 absent, no presence bits
            (1, 1),                 // GATE at 8: SET
            (0, 1), (0, 1), (0, 1), (0, 1), // fields 8, 9, 10, 11 absent
            (1, 1),                 // field 12 present
            (1, 1),                 // a parent follows
            ((parent >> 16) as u64, 16), ((parent & 0x7FF) as u64, 11),
            (5, 8),                 // the attach point
            (0, 1), (1, 1),         // two flags (the second picks the offset range)
            (0x8000, 16), (0x8000, 16), (0x8000, 16),
            (1, 1), (0, 8),         // orientation: the cheap +Z axis, then the 8-bit angle every film carries
            (0, 1), (seat, 6),      // seat flag CLEAR = a 6-bit seat index follows
            (0, 1),                 // the trailing flag
        ];
        EntityMessage {
            entity_type: 11,
            net_id: 7,
            dirty_mask: (1 << 0) | (1 << 12),
            payload: pack(&bits),
            is_update: true,
        }
    }

    #[test]
    fn a_clear_gate_hides_only_its_own_group_so_the_attachment_behind_it_is_read() {
        let m = riding_update(0x0042_0123, 2);
        let st = decode_entity_payload(&m, &params()).expect("decodes");
        let a = st.attach.unwrap_or_else(|| panic!("field 12 was reached past the clear gate at field 2: {st:?}"));
        assert_eq!(a.parent, 0x0042_0123);
        assert_eq!(a.seat, Some(2));
        assert_eq!(a.node, 5);
        assert_eq!(st.rebuilt_mask, m.dirty_mask, "the rebuilt mask equals the message's own: the walk is aligned");
        assert!(st.position_raw.is_some());
        // What the early stop left behind is preserved exactly, because the facing offset was
        // measured from it: 1 prefix + 1 + 1 + 45 position + 1 (field 1) + 1 (gate 2) = 50.
        assert_eq!(st.latched_at, Some(2));
        assert_eq!(st.bits_after_base, 50, "`bits_after_base` stays where the walk stood after the first clear gate");
    }

    /// The same message with the attachment cleared (`parent` flag clear) reports NO attachment
    /// but still consumes its bits, which is how the engine un-seats a body.
    #[test]
    fn a_cleared_attachment_is_read_as_none_and_keeps_the_walk_aligned() {
        let mut bits: Vec<(u64, u32)> = vec![
            (0, 1), (1, 1), (1, 1), (111, 15), (222, 16), (333, 14), (0, 1), (0, 1), (1, 1),
            (0, 1), (0, 1), (0, 1), (0, 1), (1, 1),
        ];
        bits.extend([(0, 1), (1, 1), (0, 1)]); // no parent; seat flag SET = -1; trailing flag
        let m = EntityMessage { entity_type: 11, net_id: 7, dirty_mask: 1 | (1 << 12), payload: pack(&bits), is_update: true };
        let st = decode_entity_payload(&m, &params()).expect("decodes");
        assert_eq!(st.attach, None);
        assert_eq!(st.rebuilt_mask, m.dirty_mask);
    }

    // ---- the rides --------------------------------------------------------------------------

    #[test]
    fn a_stated_parent_and_seat_is_a_ride_that_ends_the_tick_before_it_is_cleared() {
        let idx = FilmIndex {
            ticks: 2000,
            tracks: vec![
                vehicle(100),
                biped(1, vec![(10, None), (200, Some(att(100, Some(0)))), (260, Some(BaseAttachment { node: 9, ..att(100, Some(0)) })), (400, None)]),
            ],
            ..Default::default()
        };
        let r = recorded_rides(&idx);
        assert_eq!(r.len(), 1, "a restated record (new offset, same parent and seat) is one ride: {r:?}");
        assert_eq!((r[0].biped_net_id, r[0].vehicle_net_id, r[0].from_tick, r[0].to_tick, r[0].seat), (1, 100, 200, 399, Some(0)));
    }

    #[test]
    fn only_a_vehicle_with_a_seat_index_seats_a_body() {
        let idx = FilmIndex {
            ticks: 2000,
            tracks: vec![
                vehicle(100),
                EntityTrack { net_id: 200, entity_type: 13, ..Default::default() },
                biped(1, vec![(50, Some(att(200, Some(0))))]), // parent is not a vehicle
                biped(2, vec![(50, Some(att(100, None)))]),    // a parent but no seat: carried/stuck, not seated
                biped(3, vec![(50, Some(att(999, Some(0))))]), // parent is no entity at all
            ],
            ..Default::default()
        };
        assert!(recorded_rides(&idx).is_empty(), "{:?}", recorded_rides(&idx));
    }

    #[test]
    fn a_body_moved_between_seats_has_two_back_to_back_rides_that_never_overlap() {
        let idx = FilmIndex {
            ticks: 2000,
            tracks: vec![vehicle(100), biped(1, vec![(100, Some(att(100, Some(2)))), (300, Some(att(100, Some(0)))), (500, None)])],
            ..Default::default()
        };
        let r = recorded_rides(&idx);
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].seat, r[0].from_tick, r[0].to_tick), (Some(2), 100, 299));
        assert_eq!((r[1].seat, r[1].from_tick, r[1].to_tick), (Some(0), 300, 499));
        assert!(!(r[0].covers(300) && r[1].covers(300)));
    }

    /// A body whose ride never ended runs to its destroy, or to the film's last report.
    #[test]
    fn a_link_nobody_cleared_ends_with_the_body() {
        let mut b = biped(1, vec![(100, Some(att(100, Some(0))))]);
        b.destroyed_tick = Some(777);
        let idx = FilmIndex { ticks: 2000, tracks: vec![vehicle(100), b], ..Default::default() };
        assert_eq!(recorded_rides(&idx)[0].to_tick, 777);
    }

    /// The detector's ride is dropped for a body that STATED its parent (it overlapped none of the
    /// 49 real ones) and kept for a body the stream was silent about.
    #[test]
    fn the_recording_outranks_the_displacement_detector() {
        let pts = |f: &dyn Fn(u32) -> [f32; 3]| -> Vec<(u32, [f32; 3])> { (0..=600u32).step_by(6).map(|t| (t, f(t))).collect() };
        let veh = EntityTrack { net_id: 100, entity_type: VEHICLE_TYPE, first_tick: 0, last_tick: 600, positions: pts(&|t| [t as f32 / 7.5, 0.0, 0.0]), ..Default::default() };
        let mk = |net: u32, attachments: Vec<(u32, Option<BaseAttachment>)>| EntityTrack {
            net_id: net,
            entity_type: BIPED_TYPE,
            first_tick: 0,
            last_tick: 600,
            positions: pts(&|t| [t as f32 / 7.5 - 0.3, 0.6, 0.4]),
            attachments,
            ..Default::default()
        };
        // silent body: the detector's ride stands
        let idx = FilmIndex { ticks: 601, tracks: vec![veh.clone(), mk(1, vec![])], ..Default::default() };
        let r = accepted_rides(&idx);
        assert_eq!(r.len(), 1, "{r:?}");
        assert_eq!(r[0].seat, None, "a detected ride states no seat");
        // a body that states NO parent: the detector is overruled
        let idx = FilmIndex { ticks: 601, tracks: vec![veh.clone(), mk(1, vec![(0, None)])], ..Default::default() };
        assert!(accepted_rides(&idx).is_empty(), "{:?}", accepted_rides(&idx));
        // a body that states a seat: the recorded ride replaces the detected one
        let idx = FilmIndex { ticks: 601, tracks: vec![veh, mk(1, vec![(0, None), (100, Some(att(100, Some(1)))), (500, None)])], ..Default::default() };
        let r = accepted_rides(&idx);
        assert_eq!(r.len(), 1, "{r:?}");
        assert_eq!((r[0].seat, r[0].from_tick, r[0].to_tick), (Some(1), 100, 499));
    }

    // ---- a rider who dies ---------------------------------------------------------------------

    /// A body that boards stops reporting a position, so "the last reported position" -- the
    /// engine-free definition of the death tick -- is the boarding spot. It must follow the seat
    /// link instead, or a driver killed a minute into a drive dies where he stepped up.
    #[test]
    fn a_body_killed_in_its_seat_dies_when_its_link_ends_not_where_it_boarded() {
        let mk = |attachments: Vec<(u32, Option<BaseAttachment>)>, positions: Vec<(u32, [f32; 3])>, last: u32| EntityTrack {
            net_id: 1,
            entity_type: BIPED_TYPE,
            first_tick: 0,
            last_tick: last,
            destroyed_tick: Some(last + 600),
            positions,
            attachments,
            ..Default::default()
        };
        let walk: Vec<(u32, [f32; 3])> = (0..=100u32).step_by(10).map(|t| (t, [t as f32 / 10.0, 0.0, 0.0])).collect();
        // Boards at 100 and never leaves: the body's own last message (900) is the end.
        let open = mk(vec![(0, None), (100, Some(att(100, Some(0))))], walk.clone(), 900);
        assert_eq!(crate::film_anim::death_tick(&open), Some(900));
        // The link is cleared at 700 and nothing follows: the clear is the death.
        let cleared = mk(vec![(0, None), (100, Some(att(100, Some(0)))), (700, None)], walk.clone(), 705);
        assert_eq!(crate::film_anim::death_tick(&cleared), Some(700));
        // He got out and walked on: the last REPORTED position is still the answer.
        let mut walked_on = walk.clone();
        walked_on.extend((0..10u32).map(|i| (800 + i * 10, [20.0 + i as f32, 0.0, 0.0])));
        let out = mk(vec![(0, None), (100, Some(att(100, Some(0)))), (700, None)], walked_on, 900);
        assert_eq!(crate::film_anim::death_tick(&out), Some(890));
        // A body that never boarded is unchanged.
        let plain = mk(vec![(0, None)], walk, 300);
        assert_eq!(crate::film_anim::death_tick(&plain), Some(100));
    }

    /// The corpse of a rider lies at its VEHICLE, not at the boarding spot the body last reported.
    #[test]
    fn the_corpse_of_a_killed_rider_lies_at_the_vehicle_it_died_in() {
        let veh = EntityTrack {
            net_id: 100,
            entity_type: VEHICLE_TYPE,
            first_tick: 0,
            last_tick: 2000,
            positions: (0..=2000u32).step_by(10).map(|t| (t, [50.0 + t as f32 / 10.0, 0.0, 0.0])).collect(),
            ..Default::default()
        };
        let rider = EntityTrack {
            net_id: 1,
            entity_type: BIPED_TYPE,
            first_tick: 0,
            last_tick: 1000,
            destroyed_tick: Some(1600),
            positions: vec![(0, [0.0, 0.0, 0.0]), (90, [49.0, 0.0, 0.0])],
            attachments: vec![(0, None), (100, Some(att(100, Some(0))))],
            ..Default::default()
        };
        let mut pb = crate::film_play::Playback::new(FilmIndex { ticks: 2001, tracks: vec![veh, rider], ..Default::default() });
        pb.seek_tick(1100.0);
        let c = pb.corpses(8);
        assert_eq!(c.len(), 1, "the rider is a corpse after its death at 1000");
        let want = 50.0 + 1000.0 / 10.0;
        assert!((c[0].pos[0] - want).abs() < 0.01, "the corpse lies at the vehicle ({want}), not the boarding spot (49): {:?}", c[0].pos);
        // and the same body BEFORE it boarded is a corpse nowhere (it is alive)
        pb.seek_tick(50.0);
        assert!(pb.corpses(8).is_empty());
    }

    // ---- the seat state ---------------------------------------------------------------------

    fn ride(from: u32, to: u32, seat: Option<u8>) -> Ride {
        Ride { biped_net_id: 1, vehicle_net_id: 100, from_tick: from, to_tick: to, seat }
    }

    #[test]
    fn the_seat_state_is_a_pure_function_of_the_tick() {
        let rides = [ride(200, 399, Some(0)), ride(900, 1000, None)];
        assert_eq!(state_at(&rides, 1, 199), None);
        assert_eq!(state_at(&rides, 1, 200), Some(SeatState::Seated { ride: rides[0], age: 0 }));
        assert_eq!(state_at(&rides, 1, 399), Some(SeatState::Seated { ride: rides[0], age: 199 }));
        assert_eq!(state_at(&rides, 1, 400), Some(SeatState::Leaving { ride: rides[0], age: 0 }));
        assert_eq!(state_at(&rides, 1, 400 + EXIT_SEARCH_TICKS - 1), Some(SeatState::Leaving { ride: rides[0], age: EXIT_SEARCH_TICKS - 1 }));
        assert_eq!(state_at(&rides, 1, 400 + EXIT_SEARCH_TICKS), None, "the exit window closes");
        assert_eq!(state_at(&rides, 2, 300), None, "another body is untouched");
        // a DETECTED ride's end is a guess, so it earns no exit clip
        assert_eq!(state_at(&rides, 1, 1001), None);
        // Seek safety: asking in any order gives the same answers (no state is carried).
        let a: Vec<_> = [300u32, 150, 450, 300, 5000].iter().map(|&t| state_at(&rides, 1, t)).collect();
        let b: Vec<_> = [5000u32, 300, 450, 150, 300].iter().rev().map(|&t| state_at(&rides, 1, t)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn the_exit_blend_runs_from_the_seat_to_the_recorded_place_without_a_snap() {
        assert_eq!(exit_blend(0.0, 60.0), 0.0);
        assert_eq!(exit_blend(60.0, 60.0), 1.0);
        assert_eq!(exit_blend(500.0, 60.0), 1.0);
        let mut last = 0.0;
        for i in 0..=60 {
            let w = exit_blend(i as f32, 60.0);
            assert!(w >= last, "monotonic");
            assert!(w - last < 0.06, "no step bigger than a frame's worth: {last} -> {w}");
            last = w;
        }
        assert_eq!(exit_blend(3.0, 0.0), 1.0, "a clip of no length is already finished");
    }
}

// ---------------------------------------------------------------------------------------------
// the end-to-end gates: real recordings, a real map, the production `ActorDriver::sync`
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::film_anim::{accepted_rides, VEHICLE_TYPE};
    use crate::film_decode::FilmIndex;
    use crate::film_play::{ActorDriver, Playback};
    use crate::reach_actor::ActorScene;
    use glam::{Mat4, Vec3};
    use hms_native::{CacheHandle, NativeDll};
    use std::path::PathBuf;
    use std::sync::Arc;

    struct Open {
        native: Arc<NativeDll>,
        cache: CacheHandle,
        map: PathBuf,
    }

    fn open_forge() -> Option<Open> {
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
        let native = Arc::new(NativeDll::load(&dll).ok()?);
        let m = crate::mapcat::enumerate().into_iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
        })?;
        if let Some(d) = m.path.parent() {
            native.set_shared_cache_dirs(&[d.to_path_buf()]);
        }
        let cache = native.open_cache(&m.path.to_string_lossy()).ok()?;
        Some(Open { native, cache, map: m.path })
    }

    /// What one actor looked like after one production `sync`.
    #[derive(Clone, Debug)]
    struct Drawn {
        pos: Vec3,
        world: Option<Mat4>,
        anim: String,
    }

    struct Rig {
        pb: Playback,
        driver: ActorDriver,
        actors: ActorScene,
    }

    impl Rig {
        fn new(o: &Open, idx: FilmIndex) -> Rig {
            let mut pb = Playback::new(idx);
            pb.moving_only = false;
            let mut driver = ActorDriver::default();
            // The cap is on bodies drawn at once; the default 16 would drop a rider on a busy clip.
            driver.cap = 512;
            let mut actors = ActorScene::default();
            driver.resolve_bipeds(&pb, &o.native, o.cache, Some(&o.map));
            driver.sync(&pb, &mut actors, &o.native, o.cache);
            Rig { pb, driver, actors }
        }

        /// Seek to `tick`, run the production sync, and report how `net`'s body is drawn.
        fn drawn(&mut self, o: &Open, tick: u32, net: u32) -> Option<Drawn> {
            self.pb.seek_tick(tick as f32);
            self.driver.sync(&self.pb, &mut self.actors, &o.native, o.cache);
            let id = *self.driver.body_actor.get(&net)?;
            let a = self.actors.get(id)?;
            Some(Drawn { pos: a.pos, world: a.world_override, anim: a.anim_name.clone() })
        }
    }

    fn vehicle_pos(idx: &FilmIndex, net: u32, tick: u32) -> Option<Vec3> {
        let tr = idx.tracks.iter().find(|t| t.net_id == net)?;
        crate::film_vehicle::position_at(tr, tick as f32)
    }

    /// Films that hold at least one stated ride whose vehicle has a placement of its own.
    fn films_with_rides(limit: usize) -> Vec<(PathBuf, FilmIndex)> {
        let mut out = Vec::new();
        for p in crate::film::find_finished_films() {
            if !p.file_name().map_or(false, |n| n.to_string_lossy().contains("forge_hal")) {
                continue;
            }
            let Ok(f) = crate::film::open(&p) else { continue };
            let idx = crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let has = accepted_rides(&idx).iter().any(|r| {
                r.seat.is_some() && idx.tracks.iter().any(|t| t.net_id == r.vehicle_net_id && t.entity_type == VEHICLE_TYPE && !t.positions.is_empty())
            });
            if has {
                out.push((p, idx));
            }
            if out.len() >= limit {
                break;
            }
        }
        out
    }

    /// **THE GATE for "players don't seem to actually enter vehicles".** On real recordings, with
    /// a real map, through the production `ActorDriver::sync`:
    ///
    /// * before the link the body is drawn exactly where it reports itself, standing;
    /// * while linked it is drawn IN THE SEAT -- within a few world units of the vehicle's
    ///   recorded placement at that tick, as a whole matrix (`world_override`), playing the seat's
    ///   idle clip -- while the position the body itself reports is, on the rides chosen, many
    ///   wu from the vehicle (the premise: that is the position the old build drew);
    /// * the same holds when the playhead JUMPS into the middle of the ride, with no tick of the
    ///   boarding ever shown (a fresh driver, so nothing carries over);
    /// * a body that never boarded is not touched;
    /// * and the exit never snaps.
    ///
    /// IGNORED BY DEFAULT: it opens a Reach cache and the native parser's global state collides
    /// with other cache tests. `cargo test -p hms-app film_riders -- --include-ignored`.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn a_recorded_rider_is_drawn_in_its_seat_while_the_link_lasts() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let films = crate::film::find_finished_films();
        if films.is_empty() {
            eprintln!("SKIPPED: no Reach saved films on this machine");
            return;
        }
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let (mut n_films, mut n_rides, mut n_far_premise, mut n_vehicles) = (0usize, 0usize, 0usize, 0usize);
        let mut max_seat_to_vehicle = 0.0f32;
        let mut max_exit_step = 0.0f32;
        let mut turret_rides = 0usize;
        let (mut n_died_seated, mut n_died_far, mut n_enter, mut n_exit) = (0usize, 0usize, 0usize, 0usize);
        let mut seats: std::collections::BTreeSet<(String, String, bool)> = Default::default();
        for (path, idx) in films_with_rides(6) {
            n_films += 1;
            n_vehicles += idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE).count();
            let all = accepted_rides(&idx);
            let rides: Vec<Ride> = all
                .iter()
                .copied()
                .filter(|r| r.seat.is_some() && r.to_tick > r.from_tick + 420)
                // standing for the two seconds before, i.e. not coming straight out of another seat
                .filter(|r| state_at(&all, r.biped_net_id, r.from_tick - 120).is_none() && r.from_tick > 130)
                .filter(|r| idx.tracks.iter().any(|t| t.net_id == r.biped_net_id && t.first_tick + 130 <= r.from_tick))
                .filter(|r| vehicle_pos(&idx, r.vehicle_net_id, r.from_tick + 200).is_some())
                .take(3)
                .collect();
            let name = path.file_name().unwrap().to_string_lossy().chars().take(34).collect::<String>();
            let mut rig = Rig::new(&o, idx);
            for r in rides {
                let idx = &rig.pb.index;
                let bt = idx.tracks.iter().find(|t| t.net_id == r.biped_net_id).unwrap().clone();
                let ticks = [r.from_tick + 200, (r.from_tick + r.to_tick) / 2, r.to_tick - 10];
                // -- before: standing where it reports itself ----------------------------------
                let before = rig.drawn(&o, r.from_tick - 120, r.biped_net_id).expect("the body is drawn before boarding");
                let reported = Vec3::from(bt.sample(r.from_tick - 120).unwrap());
                assert!(before.world.is_none(), "{name}: no seat frame before the link");
                assert!(before.pos.distance(reported) < 0.05, "{name}: before boarding the body is where it reports itself");
                // -- seated ------------------------------------------------------------------
                let mut far = false;
                let mut seat_pos = Vec::new();
                for &t in &ticks {
                    let d = rig.drawn(&o, t, r.biped_net_id).expect("the rider is drawn");
                    let v = vehicle_pos(&rig.pb.index, r.vehicle_net_id, t).unwrap();
                    let m = d.world.unwrap_or_else(|| panic!("{name}: tick {t}: a seated body has a seat frame"));
                    let at = m.w_axis.truncate();
                    max_seat_to_vehicle = max_seat_to_vehicle.max(at.distance(v));
                    assert!(at.distance(v) < 5.0, "{name}: tick {t}: seat {at:?} is {:.2} wu from the vehicle at {v:?}", at.distance(v));
                    if let Some(own) = bt.sample(t) {
                        far |= Vec3::from(own).distance(v) > 8.0;
                    }
                    // The seat's own idle -- but only where the rider's graph authors one for that
                    // seat (the Falcon's passenger seats are an example that does not): the
                    // oracle is the graph itself, so a missing clip is reported, not excused.
                    let rider = rig.driver.rider_at(&rig.pb, r.biped_net_id, t, &o.native, o.cache).expect("a rider");
                    let id = *rig.driver.body_actor.get(&r.biped_net_id).unwrap();
                    let graph = &rig.actors.get(id).unwrap().def;
                    let prefix = format!("{}:", rider.seat.mode);
                    let authors_idle = graph
                        .graph
                        .as_ref()
                        .map_or(false, |g| g.anims.iter().any(|a| a.name.starts_with(&prefix) && a.name.ends_with(":idle")));
                    seats.insert((rider.seat.vehicle.clone(), rider.seat.mode.clone(), authors_idle));
                    if authors_idle {
                        assert!(d.anim.starts_with(&prefix) && d.anim.ends_with(":idle"), "{name}: tick {t}: the graph has {prefix}*:idle but the body plays {:?}", d.anim);
                    }
                    seat_pos.push((t, d));
                }
                n_far_premise += far as usize;
                // -- the boarding clip plays when the link starts, the exit clip when it ends ---
                let authors = |suffix: &str| {
                    let rider = rig.driver.rider_at(&rig.pb, r.biped_net_id, ticks[0], &o.native, o.cache).expect("a rider");
                    let id = *rig.driver.body_actor.get(&r.biped_net_id).unwrap();
                    let prefix = format!("{}:", rider.seat.mode);
                    rig.actors.get(id).unwrap().def.graph.as_ref().map_or(false, |g| g.anims.iter().any(|a| a.name.starts_with(&prefix) && a.name.ends_with(suffix)))
                };
                let (has_enter, has_exit) = (authors(":enter"), authors(":exit"));
                if has_enter {
                    let d = rig.drawn(&o, r.from_tick + 8, r.biped_net_id).expect("drawn while boarding");
                    assert!(d.anim.ends_with(":enter"), "{name}: 8 ticks into the link the body plays its boarding clip, not {:?}", d.anim);
                    assert!(d.world.is_some(), "{name}: and is already on the seat marker");
                    n_enter += 1;
                }
                if has_exit && bt.destroyed_tick.map_or(true, |t| t > r.to_tick + 60) && rig.pb.death_at(&bt).map_or(true, |t| t > r.to_tick + 60) {
                    let d = rig.drawn(&o, r.to_tick + 1 + 8, r.biped_net_id).expect("drawn while leaving");
                    let why = rig.driver.rider_at(&rig.pb, r.biped_net_id, r.to_tick + 9, &o.native, o.cache);
                    let exits: Vec<(String, i16, String)> = {
                        let id = *rig.driver.body_actor.get(&r.biped_net_id).unwrap();
                        let prefix = format!("{}:", why.as_ref().unwrap().seat.mode);
                        rig.actors.get(id).unwrap().def.graph.as_ref().map_or(vec![], |g| {
                            g.anims.iter().filter(|a| a.name.starts_with(&prefix) && a.name.contains("exit")).map(|a| (a.name.clone(), a.frame_count, format!("{:?}", a.anim_type))).collect()
                        })
                    };
                    assert!(
                        d.anim.ends_with(":exit"),
                        "{name}: 8 ticks after the link the body plays its exit clip, not {:?} (ride {r:?}; rider {why:?}; drawn {d:?}; exit-ish clips {exits:?})",
                        d.anim
                    );
                    n_exit += 1;
                }
                // -- seek safety: a fresh rig, straight into the middle of the ride -----------
                let mid = ticks[1];
                let mut fresh = Rig::new(&o, rig.pb.index.clone());
                let jumped = fresh.drawn(&o, mid, r.biped_net_id).expect("drawn after a jump");
                let walked = &seat_pos[1].1;
                assert_eq!(jumped.anim, walked.anim, "{name}: a jump lands on the same clip as walking there");
                let (jm, wm) = (jumped.world.expect("seat frame after a jump"), walked.world.unwrap());
                assert!(jm.w_axis.truncate().distance(wm.w_axis.truncate()) < 1e-3, "{name}: a jump lands in the same seat");
                // -- a body that never boarded is untouched -----------------------------------
                let bystander = rig.pb.index.tracks.iter().find(|t| {
                    t.entity_type == crate::film_anim::BIPED_TYPE
                        && t.net_id != r.biped_net_id
                        && t.first_tick + 5 < mid
                        && t.destroyed_tick.map_or(true, |d| d > mid + 5)
                        && t.sample(mid).is_some()
                        && !rig.pb.index.tracks.is_empty()
                        && accepted_rides(&rig.pb.index).iter().all(|q| q.biped_net_id != t.net_id)
                });
                if let Some(b) = bystander {
                    let (net, own) = (b.net_id, Vec3::from(b.sample(mid).unwrap()));
                    if let Some(d) = rig.drawn(&o, mid, net) {
                        assert!(d.world.is_none(), "{name}: a body that never boarded has no seat frame");
                        assert!(d.pos.distance(own) < 0.05, "{name}: and is drawn where it reports itself");
                    }
                }
                // -- the exit: continuous, then standing at the recorded place -----------------
                let end = r.to_tick + 1;
                let mut last: Option<Vec3> = None;
                for t in (end - 2)..(end + 400) {
                    // a body that climbs into another seat is a new boarding, not this exit
                    if t >= end && matches!(state_at(&all, r.biped_net_id, t), Some(SeatState::Seated { .. })) {
                        break;
                    }
                    let Some(d) = rig.drawn(&o, t, r.biped_net_id) else { break };
                    let p = d.world.map(|m| m.w_axis.truncate()).unwrap_or(d.pos);
                    if let Some(l) = last {
                        max_exit_step = max_exit_step.max(p.distance(l));
                    }
                    last = Some(p);
                }
                if let (Some(l), Some(own)) = (last, bt.sample(end + 399)) {
                    // only if the body is on foot again by then (it may have climbed straight into another seat)
                    let on_foot = rig.pb.death_at(&bt).map_or(true, |d| d > end + 399);
                    if on_foot && bt.destroyed_tick.map_or(true, |d| d > end + 399) && state_at(&all, r.biped_net_id, end + 399).is_none() {
                        let after = rig.drawn(&o, end + 399, r.biped_net_id).unwrap();
                        assert!(after.world.is_none(), "{name}: standing once the exit has played");
                        let off = l.distance(Vec3::from(own));
                        assert!(
                            off < 0.5,
                            "{name}: and at the recorded place: {off:.2} wu off ({:?} vs {own:?}); ride {r:?}; death {:?}; destroyed {:?}; last report {:?}; statements {:?}",
                            after.pos,
                            rig.pb.death_at(&bt),
                            bt.destroyed_tick,
                            bt.positions.last(),
                            bt.attachments.iter().map(|(k, a)| (*k, a.map(|a| (a.parent, a.seat)))).collect::<Vec<_>>()
                        );
                    }
                }
                // -- a rider killed in its seat lies at the vehicle, not at the boarding spot -----
                if let Some(d) = rig.pb.death_at(&bt).filter(|&d| d >= r.from_tick && d <= r.to_tick + 2) {
                    if let (Some(corpse), Some(v)) = (rig.drawn(&o, d + 30, r.biped_net_id), vehicle_pos(&rig.pb.index, r.vehicle_net_id, d)) {
                        let xy = ((corpse.pos.x - v.x).powi(2) + (corpse.pos.y - v.y).powi(2)).sqrt();
                        assert!(xy < 3.0, "{name}: the rider died in its seat at {d}, so the corpse is at the vehicle ({xy:.1} wu off: {:?} vs {v:?})", corpse.pos);
                        n_died_seated += 1;
                        if let Some(own) = bt.sample(d) {
                            n_died_far += (Vec3::from(own).distance(v) > 8.0) as usize;
                        }
                    }
                }
                if rig.driver.vehicles.as_ref().map_or(false, |vt| !vt.vehicles.contains_key(&r.vehicle_net_id)) {
                    turret_rides += 1;
                }
                n_rides += 1;
            }
        }
        eprintln!(
            "riders: {n_films} films, {n_vehicles} vehicles, {n_rides} rides checked ({n_far_premise} with the body's own position > 8 wu from the vehicle, {turret_rides} on a vehicle outside the table = turret); \
             max seat-to-vehicle {max_seat_to_vehicle:.2} wu, max exit step {max_exit_step:.2} wu/tick"
        );
        eprintln!("boarding clips checked on {n_enter} ride(s), exit clips on {n_exit}");
        assert!(n_enter >= 3 && n_exit >= 3, "liveness: {n_enter} boarding / {n_exit} exit clips exercised");
        eprintln!("died in the seat: {n_died_seated} ride(s), {n_died_far} of them with the body's own last position > 8 wu from the vehicle");
        eprintln!("seats seen (vehicle, seat label, graph authors an idle for it): {seats:?}");
        assert!(seats.iter().any(|s| s.2), "liveness: no seat with an authored idle was exercised");
        assert!(n_films >= 2 && n_vehicles >= 20, "liveness: the sweep saw {n_films} films / {n_vehicles} vehicles");
        assert!(n_rides >= 4, "liveness: only {n_rides} rides could be checked");
        assert!(n_far_premise >= 2, "premise: the old build's position must be far from the vehicle on some of these rides");
        assert!(max_exit_step < 1.5, "the exit snaps: {max_exit_step} wu in one tick");
    }

    /// The regression sweep: EVERY stated ride in the first films that hold any, sampled at three
    /// ticks, through the production sync. While a link is active the body is drawn in a seat
    /// frame within a few world units of its vehicle's recorded placement -- or the ride is
    /// counted as unplaceable (a vehicle that never replicated a position, e.g. one that never
    /// left its spawn, whose place the Forge scene holds) and the body is left on its own
    /// position. Nothing is silently dropped, and nobody is left at the boarding spot with the
    /// vehicle driving away.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn every_stated_ride_is_either_in_its_seat_or_counted_as_unplaceable() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if crate::film::find_finished_films().is_empty() {
            eprintln!("SKIPPED: no Reach saved films on this machine");
            return;
        }
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let (mut rides_n, mut seated_n, mut unplaceable, mut max_d) = (0usize, 0usize, 0usize, 0.0f32);
        let mut kinds: std::collections::BTreeMap<(String, String), usize> = Default::default();
        let mut worst = String::new();
        for (path, idx) in films_with_rides(10) {
            let name = path.file_name().unwrap().to_string_lossy().chars().take(34).collect::<String>();
            let all = accepted_rides(&idx);
            let mut rig = Rig::new(&o, idx);
            for r in all.iter().filter(|r| r.seat.is_some() && r.to_tick > r.from_tick + 10) {
                rides_n += 1;
                for t in [r.from_tick + 5, (r.from_tick + r.to_tick) / 2, r.to_tick - 1] {
                    let Some(d) = rig.drawn(&o, t, r.biped_net_id) else { continue };
                    // another seat state at this tick (a restated ride) is the other ride's business
                    let Some(rider) = rig.driver.rider_at(&rig.pb, r.biped_net_id, t, &o.native, o.cache).filter(|x| x.ride() == *r) else { continue };
                    *kinds.entry((rider.seat.vehicle.clone(), rider.seat.mode.clone())).or_default() += 1;
                    match (d.world, vehicle_pos(&rig.pb.index, r.vehicle_net_id, t)) {
                        (Some(m), Some(v)) => {
                            seated_n += 1;
                            let dist = m.w_axis.truncate().distance(v);
                            if dist > max_d {
                                max_d = dist;
                                worst = format!("{name} tick {t} {:?}", rider.seat);
                            }
                            assert!(dist < 6.0, "{name}: tick {t}: {:?} is seated {dist:.1} wu from its vehicle", rider.seat);
                        }
                        // A turret has no position of its own: it is placed through its parent.
                        (Some(m), None) => {
                            seated_n += 1;
                            assert!(m.w_axis.truncate().length() > 0.0, "{name}: tick {t}: a seat frame exists");
                        }
                        (None, _) => unplaceable += 1,
                    }
                }
            }
        }
        eprintln!(
            "ride sweep: {rides_n} stated rides, {seated_n} samples drawn in a seat (worst {max_d:.2} wu: {worst}), {unplaceable} unplaceable; seats: {kinds:?}"
        );
        assert!(rides_n >= 20, "liveness: only {rides_n} stated rides were swept");
        assert!(seated_n >= 40, "liveness: only {seated_n} samples were drawn in a seat");
        assert!(unplaceable * 4 <= seated_n, "too many unplaceable rides: {unplaceable} of {}", seated_n + unplaceable);
    }
}
