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

    /// **#film-child-follow  THE GATE for "the Warthog's turret slides around".** Through the
    /// production `ActorDriver::sync` on real recordings: while a child vehicle (the turret) is
    /// parented to a drawn Warthog, its drawn matrix expressed in the hull's frame is the SAME at
    /// every sub-tick sample of a stretch where the hull is really driving. Independent
    /// interpolation of the two tracks (the old lane) drifts by decimetres.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn a_parented_turret_keeps_its_place_on_a_driving_hull() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let (mut windows, mut worst, mut worst_at) = (0usize, 0.0f32, String::new());
        for p in crate::film::find_finished_films() {
            if windows >= 6 {
                break;
            }
            if !p.file_name().map_or(false, |n| n.to_string_lossy().contains("forge_hal")) {
                continue;
            }
            let Ok(f) = crate::film::open(&p) else { continue };
            let idx = crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            // (child, parent, from, to) stretches where the hull moves >= 6 wu in 60 ticks
            let mut spans: Vec<(u32, u32, u32)> = Vec::new();
            for c in idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE) {
                for (k, (t0, a)) in c.attachments.iter().enumerate() {
                    let Some(a) = a else { continue };
                    let t1 = c.attachments.get(k + 1).map_or(c.last_tick, |x| x.0);
                    let Some(h) = idx.tracks.iter().find(|t| t.net_id == a.parent && t.entity_type == VEHICLE_TYPE) else { continue };
                    let mut t = *t0 + 30;
                    while t + 60 < t1 {
                        if let (Some(p0), Some(p1)) = (h.sample(t), h.sample(t + 60)) {
                            if Vec3::from(p0).distance(Vec3::from(p1)) >= 6.0 {
                                spans.push((c.net_id, a.parent, t));
                                break;
                            }
                        }
                        t += 30;
                    }
                }
            }
            if spans.is_empty() {
                continue;
            }
            let name = p.file_name().unwrap().to_string_lossy().chars().take(34).collect::<String>();
            let mut rig = Rig::new(&o, idx);
            for (child, hull, t0) in spans.into_iter().take(3) {
                let mut rels: Vec<Mat4> = Vec::new();
                for k in 0..=60 * 4 {
                    rig.pb.seek_tick(t0 as f32 + k as f32 * 0.25);
                    rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                    let (Some(&ca), Some(&ha)) = (rig.driver.vehicle_actor.get(&child), rig.driver.vehicle_actor.get(&hull)) else { break };
                    let (Some(cw), Some(hw)) = (
                        rig.actors.get(ca).and_then(|a| a.world_override),
                        rig.actors.get(ha).and_then(|a| a.world_override),
                    ) else { break };
                    rels.push(hw.inverse() * cw);
                }
                if rels.len() < 200 {
                    eprintln!("{name}: child {child:#x} not drawn over the stretch ({} samples)", rels.len());
                    continue;
                }
                windows += 1;
                let r0 = rels[0];
                let d = rels
                    .iter()
                    .map(|r| (r.w_axis - r0.w_axis).truncate().length().max((r.x_axis - r0.x_axis).truncate().length()).max((r.z_axis - r0.z_axis).truncate().length()))
                    .fold(0.0f32, f32::max);
                if d > worst {
                    worst = d;
                    worst_at = format!("{name} child {child:#x} tick {t0}");
                }
            }
        }
        eprintln!("turret-on-hull windows {windows}: worst drift in the hull frame {worst:.5} ({worst_at})");
        assert!(windows >= 3, "liveness: only {windows} driving stretches had a drawn turret");
        assert!(worst < 2e-3, "the turret slides on its hull by {worst:.4} wu ({worst_at})");
    }

    /// **#race-riders  THE GATE for "Mongooses drive around by themselves, nobody is in them".**
    /// A race clip (`HT Ignition v2.1`, a downloaded saved film in `~/Downloads/Reach(1).mov`)
    /// starts with 16 players SPAWNED INSIDE their Mongooses: each biped's parent link is in its
    /// own CREATION message, a block the decoder used to skip (no type-11 arm in `creation()`), so
    /// 14 of 24 Mongooses had no recorded driver and the 16 players stood at their spawn point.
    ///
    /// Through the production `ActorDriver::sync`, at every 60th tick of every recorded ride:
    /// the rider is drawn in the vehicle's seat frame (a whole matrix, within 3 wu of the hull's
    /// recorded placement) playing a Mongoose seat clip. Counted: recorded vehicle-ticks vs
    /// rendered ones, and how many of the recorded ones are creation-borne (what the old build
    /// lost). Liveness: the clip has Mongoose drivers, and creation-borne ones.
    #[test]
    #[ignore = "opens a Reach cache and reads a downloaded race clip"]
    fn every_recorded_mongoose_driver_in_the_race_clip_is_drawn_in_the_seat() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let p = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Downloads").join("Reach(1).mov");
        let Ok(f) = crate::film::open(&p) else {
            eprintln!("SKIPPED: the race clip is not on this machine");
            return;
        };
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let rides = accepted_rides(&idx);
        let firsts: std::collections::HashMap<u32, u32> = idx.tracks.iter().map(|t| (t.net_id, t.first_tick)).collect();
        let first_of = |net: u32| firsts.get(&net).copied().unwrap_or(0);
        let mut rig = Rig::new(&o, idx);
        // THE APP'S OWN CAP (`ActorDriver::default().cap`), not the rig's 512: a gate that raises it
        // hides every body the real window drops.
        rig.driver.cap = ActorDriver::default().cap;
        let (mut recorded, mut rendered, mut born, mut born_rendered, mut worst) = (0usize, 0usize, 0usize, 0usize, 0.0f32);
        let (mut mongoose_rides, mut bad_clip) = (0usize, 0usize);
        let mut misses: Vec<String> = Vec::new();
        let mut dead_occupants = 0usize;
        let mut pre_position = 0usize;
        // INDEPENDENT GROUND TRUTH for the creation-borne rides: a rider's own first REPORTED
        // position (not the seed, which sits at the creation tick) lies within 1 wu of the
        // vehicle the ride names. Nothing of the parent link is used to decide this.
        let (mut corroborated, mut uncorroborated) = (0usize, 0usize);
        for r in rides.iter().filter(|r| r.seat.is_some() && r.from_tick <= first_of(r.biped_net_id) + 8) {
            let tr = rig.pb.index.tracks.iter().find(|x| x.net_id == r.biped_net_id);
            let veh = rig.pb.index.tracks.iter().find(|x| x.net_id == r.vehicle_net_id);
            let first = tr.and_then(|t| t.positions.iter().find(|(k, _)| *k > t.first_tick).copied());
            match (first, veh) {
                (Some((k, bp)), Some(v)) if crate::film_vehicle::position_at(v, k as f32).is_some_and(|vp| Vec3::from(bp).distance(vp) < 1.0) => corroborated += 1,
                _ => uncorroborated += 1,
            }
        }
        let mut ticks: Vec<u32> = rides.iter().flat_map(|r| (r.from_tick + 4..r.to_tick.saturating_sub(4)).step_by(60)).collect();
        // The first 400 ticks densely: the riders' own first position reports arrive 15 to 270
        // ticks after the spawn, and that is the window the sparse 60-tick sweep walked straight over.
        ticks.extend((41..400).step_by(3));
        ticks.sort_unstable();
        ticks.dedup();
        for t in ticks {
            rig.pb.seek_tick(t as f32);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            for r in rides.iter().filter(|r| r.seat.is_some() && r.from_tick + 1 <= t && t + 4 < r.to_tick) {
                let Some(v) = rig.pb.index.tracks.iter().find(|x| x.net_id == r.vehicle_net_id) else { continue };
                let Some(hull) = crate::film_vehicle::position_at(v, t as f32) else { continue };
                if !rig.driver.vehicles.as_ref().and_then(|vt| vt.vehicles.get(&r.vehicle_net_id)).is_some_and(|x| x.name.contains("mongoose")) {
                    continue;
                }
                // A rider who has DIED lies where he fell (the death clip owns the body); that is
                // the engine's ragdoll business, not a seat. Counted, not asserted on.
                let dead = rig.driver.body_actor.get(&r.biped_net_id).and_then(|&id| rig.actors.get(id)).is_some_and(|a| a.anim_name.contains("dead"));
                if dead {
                    dead_occupants += 1;
                    continue;
                }
                recorded += 1;
                let creation_borne = r.from_tick <= first_of(r.biped_net_id) + 8;
                born += creation_borne as usize;
                let drawn = rig.driver.body_actor.get(&r.biped_net_id).and_then(|&id| rig.actors.get(id)).map(|a| (a.world_override, a.anim_name.clone()));
                if drawn.as_ref().map_or(true, |d| d.0.is_none()) && misses.len() < 12 {
                    misses.push(format!("t{t} biped {:#x} veh {:#x} ride {}..{} NOT DRAWN IN SEAT ({:?})", r.biped_net_id, r.vehicle_net_id, r.from_tick, r.to_tick, drawn.as_ref().map(|d| d.1.clone())));
                }
                if let Some((Some(m), anim)) = drawn {
                    let d = m.w_axis.truncate().distance(hull);
                    worst = worst.max(d);
                    if d >= 3.0 && misses.len() < 12 {
                        misses.push(format!("t{t} biped {:#x} veh {:#x} ride {}..{} d {d:.1} anim {anim}", r.biped_net_id, r.vehicle_net_id, r.from_tick, r.to_tick));
                    }
                    if d < 3.0 {
                        rendered += 1;
                        born_rendered += creation_borne as usize;
                        if !anim.contains("mongoose") {
                            bad_clip += 1;
                        }
                    }
                }
            }
        }
        mongoose_rides += rides.iter().filter(|r| r.seat.is_some()).count();
        eprintln!(
            "race clip: {mongoose_rides} stated rides; sampled vehicle-ticks with a recorded Mongoose occupant {recorded} (creation-borne {born}); \
             rendered in the seat {rendered} (creation-borne {born_rendered}); dead riders excluded {dead_occupants}; before first position (excluded) {pre_position}; worst seat-to-hull {worst:.2} wu; not a mongoose clip {bad_clip}"
        );
        for m in &misses {
            eprintln!("miss: {m}");
        }
        eprintln!("independent truth: {corroborated} creation-borne rides corroborated by the rider's own first position, {uncorroborated} not");
        assert!(corroborated >= 14, "liveness: only {corroborated} creation-borne rides are corroborated by the riders' own positions");
        assert!(mongoose_rides >= 20 && born >= 100, "liveness: {mongoose_rides} rides / {born} creation-borne Mongoose samples");
        assert_eq!(rendered, recorded, "{} recorded Mongoose driver samples are not drawn in the seat", recorded - rendered);
        assert_eq!(bad_clip, 0, "{bad_clip} drivers are not posed with a Mongoose seat clip");
    }

    /// **#film-wheels  THE GATE for "vehicle wheels don't turn in Theater".** On the race clip's
    /// Mongooses, through the production `ActorDriver::sync`, then the pose the draw path skins
    /// from (`Actor::pose.model`, which `skin_palette` turns into the bone palette):
    ///
    /// * parked at the start (odometer 0): the rear wheel node is unrotated and the steering node
    ///   is centred;
    /// * over a driving stretch, the rear wheel's rotation angle follows the DISTANCE rolled / the
    ///   authored radius, sample by sample (wrapped; the clip is four 90-degree keys, so 0.2 rad);
    /// * the steering node swings with the recorded turning and rests when it is straight;
    /// * seeking away and back to a tick poses the identical wheel.
    #[test]
    #[ignore = "opens a Reach cache and reads a downloaded race clip"]
    fn the_race_clips_mongoose_wheels_turn_with_the_distance_they_roll() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let p = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Downloads").join("Reach(1).mov");
        let Ok(f) = crate::film::open(&p) else {
            eprintln!("SKIPPED: the race clip is not on this machine");
            return;
        };
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let mut rig = Rig::new(&o, idx);
        // The angle one node has turned from its bind orientation, radians, 0..pi.
        let node_angle = |rig: &Rig, net: u32, node: &str| -> Option<f32> {
            let a = rig.actors.get(*rig.driver.vehicle_actor.get(&net)?)?;
            let i = a.def.model.node_names.iter().position(|n| n == node)?;
            let (_, rb, _) = a.def.model.skel.bind_model()[i].to_scale_rotation_translation();
            let (_, rp, _) = a.pose.model[i].to_scale_rotation_translation();
            Some((rb.inverse() * rp).normalize().w.abs().clamp(0.0, 1.0).acos() * 2.0)
        };
        let wrap = |x: f32| (x + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        let mut vi = crate::play_vehicle::VehicleIndex::default();
        let nets: Vec<u32> = rig.pb.index.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE).map(|t| t.net_id).collect();
        let (mut driven, mut worst, mut worst_rest, mut steered, mut straight, mut bad_steer) = (0usize, 0.0f32, 0.0f32, 0usize, 0usize, 0usize);
        let mut seek_checked = 0usize;
        for net in nets {
            let Some(tr) = rig.pb.index.tracks.iter().find(|t| t.net_id == net).cloned() else { continue };
            let Some(ident) = rig.driver.vehicles.as_ref().and_then(|vt| vt.vehicles.get(&net)).cloned() else { continue };
            if !ident.name.contains("mongoose") || tr.positions.len() < 2000 {
                continue;
            }
            let Some(e) = vi.entry(&o.native, o.cache, ident.obj_tag) else { continue };
            let anim = e.anim.clone();
            use crate::reach_vehicle_anim::Role;
            let Some((rear, radius)) = anim.overlays.iter().find_map(|ov| match &ov.role {
                Role::Roll { radius, wheels } => wheels.iter().find(|w| w.contains("back") || w.contains("rear")).map(|w| (w.clone(), *radius)),
                _ => None,
            }) else {
                continue;
            };
            let Some((wb, tl)) = e.drive.as_ref().ok().and_then(|d| Some((d.wheelbase?, d.turning?))) else { continue };
            let steer_node = anim.overlays.iter().find_map(|ov| matches!(ov.role, Role::Steer { .. }).then(|| ov.nodes.first().cloned()).flatten());
            let odo = crate::film_vehicle::Odometer::build(&tr);
            // -- parked at the start: nothing rolled yet ------------------------------------
            let t_rest = tr.first_tick + 3;
            rig.pb.seek_tick(t_rest as f32);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            rig.actors.update_poses_headless(0.0);
            if let Some(a) = node_angle(&rig, net, &rear) {
                worst_rest = worst_rest.max(a.min(std::f32::consts::TAU - a));
            }
            // -- a driving stretch: >= 8 wu in 60 ticks, forward ------------------------------
            let mut t0 = None;
            let mut t = tr.first_tick + 600;
            while t + 80 < tr.last_tick {
                if odo.at((t + 60) as f32) - odo.at(t as f32) >= 8.0 {
                    t0 = Some(t);
                    break;
                }
                t += 240;
            }
            let Some(t0) = t0 else { continue };
            driven += 1;
            for k in 0..30 {
                let ft = t0 as f32 + k as f32 * 2.0;
                rig.pb.seek_tick(ft);
                rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                rig.actors.update_poses_headless(0.0);
                let Some(a) = node_angle(&rig, net, &rear) else { continue };
                let want = wrap(odo.at(ft) / radius).abs();
                let mut err = (a - want).abs();
                if err > std::f32::consts::PI {
                    err = (std::f32::consts::TAU - err).abs();
                }
                worst = worst.max(err);
                if let Some(sn) = &steer_node {
                    let s = crate::film_vehicle::steer_fraction_at(&tr, ft, wb, tl.max_left, tl.max_right);
                    if let Some(sa) = node_angle(&rig, net, sn) {
                        if s.abs() > 0.5 {
                            steered += 1;
                            bad_steer += (sa < 0.03) as usize;
                        } else if s.abs() < 0.05 {
                            straight += 1;
                            bad_steer += (sa > 0.06) as usize;
                        }
                    }
                }
            }
            // -- ticks where the recording is turning hard: the steering node must be bent ----
            if let Some(sn) = &steer_node {
                let mut t = tr.first_tick + 600;
                let mut hits = 0;
                while t + 10 < tr.last_tick && hits < 4 {
                    let ft = t as f32;
                    if crate::film_vehicle::steer_fraction_at(&tr, ft, wb, tl.max_left, tl.max_right).abs() > 0.5 {
                        hits += 1;
                        rig.pb.seek_tick(ft);
                        rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                        rig.actors.update_poses_headless(0.0);
                        if let Some(sa) = node_angle(&rig, net, sn) {
                            steered += 1;
                            bad_steer += (sa < 0.1) as usize;
                        }
                    }
                    t += 20;
                }
            }
            // -- a seek away and back poses the identical wheel --------------------------------
            let probe_t = t0 as f32 + 17.0;
            rig.pb.seek_tick(probe_t);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            rig.actors.update_poses_headless(0.0);
            let first = node_angle(&rig, net, &rear);
            rig.pb.seek_tick((t0 + 4000).min(tr.last_tick - 1) as f32);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            rig.pb.seek_tick(probe_t);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            rig.actors.update_poses_headless(0.0);
            let again = node_angle(&rig, net, &rear);
            assert_eq!(first.is_some(), again.is_some());
            if let (Some(a), Some(b)) = (first, again) {
                assert!((a - b).abs() < 1e-4, "{net:#x}: a seek changed the wheel angle {a} -> {b}");
                seek_checked += 1;
            }
            if driven >= 6 {
                break;
            }
        }
        eprintln!("wheels: {driven} driving stretches, worst |angle - distance/radius| {worst:.3} rad, worst parked angle {worst_rest:.3}, steering samples {steered} bent / {straight} straight ({bad_steer} wrong), seek checks {seek_checked}");
        assert!(driven >= 3 && seek_checked >= 3, "liveness: {driven} stretches / {seek_checked} seek checks");
        assert!(worst_rest < 0.05, "a parked Mongoose's wheel is rotated by {worst_rest} rad");
        assert!(worst < 0.2, "the wheel does not follow distance/radius: worst error {worst} rad");
        assert!(steered >= 3 && straight >= 3, "liveness: steering saw {steered} bent / {straight} straight samples");
        assert_eq!(bad_steer, 0, "{bad_steer} steering samples disagree with the recorded turning");
    }

    /// **#monitor-names  THE GATE for "monitor players have no names in Theater".** Over every film
    /// of the corpus (plus the downloaded race clip) that holds a Forge monitor, through the
    /// production name path (`name_tags_with` over the driver's `monitor_owner`, then
    /// `resolve_head_anchors` on the drawn actor): every monitor alive at the sampled moment has a
    /// tag, it is drawn as an INFERENCE (never `named`), it floats above the monitor's own `head`
    /// marker, and a tag that carries a seat's name belongs to a seat that is NOT also driving a
    /// Spartan at that moment. The race clip's two monitors are two DIFFERENT seats.
    ///
    /// Liveness: at least 10 monitors across films and at least 2 resolved to a seat.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn every_forge_monitor_gets_a_name_tag_above_its_head() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        let mut paths = crate::film::find_finished_films();
        paths.push(PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Downloads").join("Reach(1).mov"));
        let (mut monitors, mut resolved, mut unknown, mut tagged, mut anchored, mut conflicts) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut race_seats: Vec<usize> = Vec::new();
        let mut undrawn = 0usize;
        for p in paths {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let mut rig = Rig::new(&o, idx);
            let mons: Vec<(u32, u32, u32)> = rig
                .pb
                .index
                .tracks
                .iter()
                .filter(|t| t.entity_type == 11 && rig.driver.biped_of.get(&t.net_id).is_some_and(|b| b.contains("monitor")))
                .map(|t| (t.net_id, t.first_tick, t.destroyed_tick.unwrap_or(t.last_tick)))
                .collect();
            let is_race = p.file_name().is_some_and(|n| n == "Reach(1).mov");
            for (net, a, b) in mons {
                let t = (a + (b - a) / 2).max(a + 2);
                rig.pb.seek_tick(t as f32);
                rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                rig.actors.update_poses_headless(0.0);
                let mut tags = crate::film_ui::name_tags_with(&rig.pb, &[], 4096, &rig.driver.monitor_owner);
                crate::film_ui::resolve_name_colors(&mut tags, None, crate::reach_chud::TEAM_NONE, false);
                crate::film_ui::resolve_head_anchors(&mut tags, &rig.actors, &o.native, o.cache);
                // A monitor the recording never placed (0 or 1 position reports) is not DRAWN, so
                // there is nothing to float a name over; it is counted, not asserted on.
                if !rig.pb.bipeds(4096).iter().any(|x| x.net_id == net) {
                    undrawn += 1;
                    continue;
                }
                monitors += 1;
                let Some(tag) = tags.iter().find(|x| x.net_id == net) else { continue };
                tagged += 1;
                assert!(!tag.named, "a monitor's name is an inference, never a stated gamertag");
                anchored += tag.on_marker as usize;
                match rig.driver.monitor_owner.get(&net) {
                    Some(crate::film_roster::MonitorOwner::Seat(si)) => {
                        resolved += 1;
                        // not also playing a Spartan right now
                        let busy = rig.pb.index.tracks.iter().any(|x| {
                            x.entity_type == 11
                                && x.net_id != net
                                && rig.pb.index.seat_index_of_entity(x.net_id) == Some(*si)
                                && x.first_tick + 30 <= t
                                && t + 30 <= x.destroyed_tick.unwrap_or(x.last_tick)
                        });
                        conflicts += busy as usize;
                        if is_race {
                            race_seats.push(*si);
                        }
                    }
                    _ => unknown += 1,
                }
            }
        }
        eprintln!("monitors: {monitors} drawn ({undrawn} never placed, excluded), {tagged} tagged ({anchored} on the head marker), {resolved} resolved to a seat, {unknown} labelled Monitor, {conflicts} conflicts; race clip seats {race_seats:?}");
        assert!(monitors >= 10 && resolved >= 2, "liveness: {monitors} monitors / {resolved} resolved");
        assert_eq!(tagged, monitors, "{} monitors have no name tag", monitors - tagged);
        assert_eq!(anchored, tagged, "{} monitor tags are not on the head marker", tagged - anchored);
        assert_eq!(conflicts, 0, "{conflicts} monitor tags name a seat that is playing a Spartan at that moment");
        assert!(race_seats.len() == 2 && race_seats[0] != race_seats[1], "the race clip's two monitors must be two different seats: {race_seats:?}");
    }

    /// #race-truth  print-only: an INDEPENDENT ground truth for who rides which vehicle, from the
    /// bodies' own reported positions (no parent link, no occupant field).
    #[test]
    #[ignore = "print-only probe"]
    fn probe_race_independent_truth() {
        let p = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov");
        let f = crate::film::open(&p).expect("opens");
        let idx = crate::film_decode::build_index_film(&f);
        let rides = accepted_rides(&idx);
        let vehicles: Vec<&crate::film_decode::EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE).collect();
        let mut near_total = 0usize;
        let mut near_unexplained: Vec<String> = Vec::new();
        let mut ride_far = 0usize;
        let mut ride_samples = 0usize;
        for b in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            for &(t, bp) in &b.positions {
                // nearest vehicle at that tick
                let mut best: Option<(f32, u32)> = None;
                for v in &vehicles {
                    if !v.alive_at(t) {
                        continue;
                    }
                    if let Some(vp) = crate::film_vehicle::position_at(v, t as f32) {
                        let d = Vec3::from(bp).distance(vp);
                        if best.map_or(true, |x| d < x.0) {
                            best = Some((d, v.net_id));
                        }
                    }
                }
                let riding = rides.iter().find(|r| r.biped_net_id == b.net_id && r.from_tick <= t && t <= r.to_tick);
                if let Some(r) = riding {
                    ride_samples += 1;
                    let vp = vehicles.iter().find(|v| v.net_id == r.vehicle_net_id).and_then(|v| crate::film_vehicle::position_at(v, t as f32));
                    if vp.map_or(false, |vp| Vec3::from(bp).distance(vp) > 6.0) {
                        ride_far += 1;
                    }
                }
                if let Some((d, vn)) = best {
                    if d < 2.5 {
                        near_total += 1;
                        if riding.map(|r| r.vehicle_net_id) != Some(vn) && near_unexplained.len() < 40 {
                            near_unexplained.push(format!("biped {:#x} t{t} within {d:.2} wu of vehicle {vn:#x}; decoder ride: {:?}", b.net_id, riding.map(|r| (r.vehicle_net_id, r.from_tick, r.to_tick))));
                        }
                    }
                }
            }
        }
        eprintln!("truth: {near_total} body position reports lie within 2.5 wu of a Mongoose; {} of them are NOT in the decoder's ride for that vehicle", near_unexplained.len());
        for l in &near_unexplained {
            eprintln!("   {l}");
        }
        eprintln!("rides: {ride_samples} reported positions during decoder rides, {ride_far} of them > 6 wu from the ride's vehicle");
        // first reported position vs nearest vehicle for each biped
        for b in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            let Some(&(t, bp)) = b.positions.first() else { continue };
            let mut ranked: Vec<(f32, u32)> = vehicles.iter().filter(|v| v.alive_at(t)).filter_map(|v| Some((Vec3::from(bp).distance(crate::film_vehicle::position_at(v, t as f32)?), v.net_id))).collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let r = rides.iter().find(|r| r.biped_net_id == b.net_id && r.from_tick <= t + 10);
            eprintln!("first {:#x} life {}..{} firstpos@{t}: nearest {:?} | decoder ride at spawn {:?}", b.net_id, b.first_tick, b.last_tick, ranked.iter().take(2).map(|(d, n)| format!("{n:#x}@{d:.2}")).collect::<Vec<_>>(), r.map(|r| (r.vehicle_net_id, r.from_tick, r.to_tick)));
        }
    }

    /// #race-truth  print-only: which of the race clip's vehicles does the FORGE SCENE also draw?
    #[test]
    #[ignore = "print-only probe"]
    fn probe_race_forge_copies() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let p = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov");
        let f = crate::film::open(&p).expect("opens");
        let idx = crate::film_decode::build_index_film(&f);
        let rig = Rig::new(&o, idx);
        let vt = rig.driver.vehicles.as_ref().unwrap();
        let idx = &rig.pb.index;
        eprintln!("variant objects {} | vehicles identified {} (by definition {}, with a forge copy {})", idx.variant_objects.len(), vt.vehicles.len(), rig.driver.veh_by_def.added, rig.driver.veh_by_def.with_forge_copy);
        for v in idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE) {
            let Some(id) = vt.vehicles.get(&v.net_id) else { continue };
            let t0 = v.first_tick + 4;
            let p0 = crate::film_vehicle::position_at(v, t0 as f32);
            let away = |t: u32| crate::film_vehicle::position_at(v, t as f32).map(|p| Vec3::from(id.spawn_pos).distance(p));
            eprintln!(
                "veh {:#x} life {}..{} forge copy at spawn {:?} (err {:.2}) | start {:?} | distance from that spawn at +60/+600/+6000 ticks: {:?} {:?} {:?}",
                v.net_id, v.first_tick, v.last_tick,
                id.spawn_pos.iter().map(|x| if x.is_finite() { format!("{x:.1}") } else { "none".into() }).collect::<Vec<_>>(),
                id.spawn_error, p0.map(|p| format!("({:.1},{:.1},{:.1})", p.x, p.y, p.z)),
                away(t0 + 60).map(|d| d as i32), away(t0 + 600).map(|d| d as i32), away(t0 + 6000).map(|d| d as i32)
            );
        }
    }

    /// #race-truth  print-only, whole corpus + Downloads: biped creations that state a parent
    /// (mask bit 12) versus the ones the decoder actually recovered one for.
    #[test]
    #[ignore = "print-only probe"]
    fn probe_creation_parents_corpus() {
        let mut paths = crate::film::find_finished_films();
        let dl = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads");
        for n in ["Reach(1).mov", "Reach.mov", "asq_mglo-1_ca_forge__A5992DE3_6938D76B.mov"] {
            paths.push(dl.join(n));
        }
        let (mut films, mut with12, mut recovered, mut vehicle_films) = (0usize, 0usize, 0usize, 0usize);
        for p in paths {
            let Ok(f) = crate::film::open(&p) else { eprintln!("cannot open {}", p.file_name().unwrap().to_string_lossy()); continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let mut base_tick: Option<u32> = None;
            let (mut n12, mut shapes) = (0usize, std::collections::BTreeMap::<u64, usize>::new());
            let mut nets = Vec::new();
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = crate::film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
                let _ = *base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
                for e in &t.entities {
                    if e.entity_type == 11 && !e.is_update {
                        *shapes.entry(e.dirty_mask & 0xFFFFF).or_default() += 1;
                        if e.dirty_mask & 0x1000 != 0 {
                            n12 += 1;
                            nets.push(e.net_id);
                        }
                    }
                }
            }
            for n in &nets {
                let ok = idx.tracks.iter().find(|t| t.net_id == *n).is_some_and(|t| t.attachments.first().is_some_and(|a| a.0 <= t.first_tick + 1 && a.1.is_some()));
                if !ok {
                    let mut base_tick: Option<u32> = None;
                    for (i, rec) in f.data.records.iter().enumerate() {
                        let Ok(t) = crate::film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
                        let _ = *base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
                        for e in t.entities.iter().filter(|e| e.net_id == *n && !e.is_update) {
                            let pls = [(e.payload.as_slice(), e.dirty_mask)];
                            let (w, c) = crate::film_decode::solve_creation_dirty_width(&pls, &idx.params);
                            eprintln!("   MISSED creation {n:#x} mask {:#x} payload {} bits; own best width {w} ({c} rebuilt); film width solve {:?}", e.dirty_mask, e.payload.len() * 8, crate::film_decode::creation_biped(&e.payload, &idx.params, 7).map(|c| (c.owner, c.end_bit)));
                        }
                    }
                }
            }
            let got = nets.iter().filter(|n| idx.tracks.iter().find(|t| t.net_id == **n).is_some_and(|t| t.attachments.first().is_some_and(|a| a.0 <= t.first_tick + 1 && a.1.is_some()))).count();
            let veh = idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE).count();
            films += 1;
            with12 += n12;
            recovered += got;
            vehicle_films += (veh > 0) as usize;
            if n12 != got || n12 > 0 {
                eprintln!("film {} | vehicles {veh} | biped creations stating a parent {n12}, recovered {got} | creation shapes {:?}", p.file_name().unwrap().to_string_lossy(), shapes.iter().map(|(m, c)| format!("{m:#x}x{c}")).collect::<Vec<_>>());
            }
        }
        eprintln!("corpus: {films} films ({vehicle_films} with vehicles); {with12} biped creations state a parent, {recovered} recovered");
    }

    /// **#cam-jitter  THE GATE for "the camera is jittery when following a fast vehicle".** On the
    /// race clip, following a rider whose Mongoose is doing ~20 wu/s, through the production camera
    /// solve (`solve_camera_at`, the vehicle camera the engine switches to) and the production
    /// vehicle placement (`sync`), at 30 / 60 / 144 fps and irregular frame times: the hull must stay
    /// put in the view -- high-pass RMS of its projected position under 0.5 px at every rate (the
    /// lag-free camera measured 0.01 px; the rounded-tick camera 7.1 px at 144 fps, 5.7 px irregular).
    /// Saboteur: read the rounded tick again in `context_pose`'s vehicle arm.
    /// #vehicle-flip  **A VEHICLE NEVER TURNS 90 DEGREES IN A TICK.** Reported: Mongooses "randomly
    /// rotate 90 degrees" and the follow camera flashes. Cause: the forward angle is measured from a
    /// reference tangent chosen by comparing |up.x| with |up.y|; for a near-level hull that is a
    /// tie at the precision `up` travels in, so single keys decoded 90 degrees about `up` off
    /// (6,732 of them in the race clip's 17 Mongoose tracks, up to 586 per hull, every one 87-93
    /// degrees). Gate: in every recorded clip, no orientation key of any track turns faster than
    /// 400 deg/s (a Mongoose spins nowhere near that; the broken keys were 5,000 deg/s and up) while
    /// also turning more than 30 degrees; and the tie really occurs (liveness).
    /// Saboteur: make `orientation_world_after` return the engine-rule decode (6,732 -> back).
    #[test]
    #[ignore = "reads the downloaded clips"]
    fn no_recorded_track_turns_ninety_degrees_in_one_tick() {
        let dir = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads");
        let mut clips = 0;
        let mut keys = 0usize;
        let before = crate::film_decode::TIE_BRANCH_TAKEN.load(std::sync::atomic::Ordering::Relaxed);
        for name in ["Reach(1).mov", "Reach.mov", "asq_mglo-1_ca_forge__A5992DE3_6938D76B.mov"] {
            let Ok(f) = crate::film::open(&dir.join(name)) else { continue };
            let idx = crate::film_decode::build_index_film(&f);
            clips += 1;
            let mut bad = Vec::new();
            for tr in &idx.tracks {
                keys += tr.orientations.len();
                for w in tr.orientations.windows(2) {
                    let q = |f: [f32; 3], u: [f32; 3]| glam::Quat::from_mat3(&crate::film_decode::basis_mat3(f, u));
                    let ang = q(w[0].1, w[0].2).angle_between(q(w[1].1, w[1].2)).to_degrees();
                    let dt = (w[1].0 - w[0].0).max(1) as f32 / 60.0;
                    if ang > 30.0 && ang / dt > 400.0 {
                        bad.push((tr.net_id, w[0].0, ang));
                    }
                }
            }
            eprintln!("{name}: {} track(s), {} jump(s) {:?}", idx.tracks.len(), bad.len(), &bad[..bad.len().min(5)]);
            assert!(bad.is_empty(), "{name}: {} orientation key(s) turn >30 deg faster than 400 deg/s", bad.len());
        }
        let taken = crate::film_decode::TIE_BRANCH_TAKEN.load(std::sync::atomic::Ordering::Relaxed) - before;
        eprintln!("{clips} clip(s), {keys} orientation keys, {taken} tie key(s) resolved by continuity");
        assert!(clips >= 1, "no clip found");
        assert!(taken >= 1000 || clips < 2, "the tie never occurred ({taken}): the gate proves nothing");
    }

    /// #vehicle-flip  The follow camera over a long span has no outlier frame: following the
    /// racer through ticks 600..20000 at 144 fps, no frame turns the camera more than 6 degrees or
    /// moves it more than 1.5 wu from the previous one (a Mongoose at 25 wu/s moves 0.17 wu a
    /// frame; the bad keys turned the camera 90 degrees for 1-3 ticks). Reports the count.
    #[test]
    #[ignore = "opens a Reach cache and reads a downloaded race clip"]
    fn the_follow_camera_has_no_outlier_frame_over_a_long_drive() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let p = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov");
        let f = crate::film::open(&p).expect("opens");
        let idx = crate::film_decode::build_index_film(&f);
        let rider: u32 = 0xec0e029b;
        let owner = idx.tracks.iter().find(|t| t.net_id == rider).and_then(|t| t.owner_raw).expect("owner");
        let mut rig = Rig::new(&o, idx);
        let mut worst = (0usize, String::new());
        for context in [true, false] {
            rig.pb.context_cameras = context;
            rig.pb.camera = crate::film_play::FilmCamera::FollowOwner(owner);
            let mut prev: Option<(Vec3, f32, f32)> = None;
            let (mut frames, mut outliers, mut ticks) = (0usize, 0usize, Vec::new());
            let n = ((20000.0 - 600.0) / 60.0 * 144.0) as usize;
            for i in 0..n {
                rig.pb.seek_tick(600.0 + i as f32 * 60.0 / 144.0);
                rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                let Some(pose) = crate::film_play::solve_camera_at(&rig.pb, rig.driver.eye_height(&rig.pb)) else { prev = None; continue };
                let cur = (Vec3::from(pose.pos), pose.yaw, pose.pitch);
                if let Some(pr) = prev {
                    let mut dy = (cur.1 - pr.1).abs();
                    if dy > std::f32::consts::PI { dy = 2.0 * std::f32::consts::PI - dy; }
                    frames += 1;
                    if dy.to_degrees() > 6.0 || (cur.0 - pr.0).length() > 1.5 || (cur.2 - pr.2).abs().to_degrees() > 6.0 {
                        outliers += 1;
                        if ticks.len() < 8 { ticks.push((600.0 + i as f32 * 60.0 / 144.0) as u32); }
                    }
                }
                prev = Some(cur);
            }
            eprintln!("context {context}: {frames} frames, {outliers} outlier(s) at {ticks:?}");
            if outliers > worst.0 { worst = (outliers, format!("context {context}: {ticks:?}")); }
        }
        assert_eq!(worst.0, 0, "outlier camera frames: {}", worst.1);
    }

    /// #fp-jitter  The camera of an ON-FOOT subject, sampled like a display would. Over `secs` seconds
    /// from `t0` at `fps` (optionally with irregular frame times) returns the 95th percentile of the
    /// per-frame second difference of the position (wu) and of the yaw (degrees), and the largest
    /// position second difference. Respawn cuts (a 3 wu jump) are not jitter and are skipped.
    fn on_foot_camera_series(pb: &mut Playback, t0: f32, secs: f32, fps: f32, irregular: bool) -> Option<(f32, f32, f32)> {
        let mut t = 0.0f32;
        let mut rngs = 777u32;
        let mut poses: Vec<(f32, [f32; 3], f32)> = Vec::new();
        while t < secs {
            let dt = if irregular {
                rngs = rngs.wrapping_mul(1664525).wrapping_add(1013904223);
                (1.0 / fps) * (0.4 + 1.2 * ((rngs >> 8) as f32 / 16777216.0))
            } else {
                1.0 / fps
            };
            t += dt;
            pb.seek_tick(t0 + t * 60.0);
            let Some(p) = crate::film_play::solve_camera_at(pb, 0.62) else { continue };
            poses.push((t, p.pos, p.yaw));
        }
        if poses.len() < 20 {
            return None;
        }
        let wrap = |d: f32| (d + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI;
        let (mut ep, mut ey) = (Vec::new(), Vec::new());
        for w in poses.windows(3) {
            if (Vec3::from(w[2].1) - Vec3::from(w[0].1)).length() > 3.0 { continue; }
            // the deviation of the middle frame from the straight line (in TIME) through its neighbours,
            // so uneven frame times are not mistaken for roughness
            let f = (w[1].0 - w[0].0) / (w[2].0 - w[0].0);
            let line = Vec3::from(w[0].1) + (Vec3::from(w[2].1) - Vec3::from(w[0].1)) * f;
            ep.push((Vec3::from(w[1].1) - line).length());
            let (a, c) = (wrap(w[0].2 - w[1].2), wrap(w[2].2 - w[1].2));
            ey.push((-(a + (c - a) * f)).to_degrees().abs());
        }
        let p95 = |v: &mut Vec<f32>| { v.sort_by(|a, b| a.total_cmp(b)); v[((v.len() - 1) as f32 * 0.95) as usize] };
        let worst = ep.iter().cloned().fold(0.0, f32::max);
        Some((p95(&mut ep), p95(&mut ey), worst))
    }

    /// The films the first-person gates can use: the race clip plus the playable corpus.
    fn jitter_films() -> Vec<crate::film_decode::FilmIndex> {
        let mut files: Vec<PathBuf> = vec![PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov")];
        files.extend(crate::film::spec_corpus().into_iter().take(40));
        let mut out = Vec::new();
        for p in files {
            if out.len() >= 5 { break; }
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.data.records.len() < 3000 { continue; }
            let idx = crate::film_decode::build_index_film(&f);
            if idx.tracks.iter().any(|t| t.entity_type == 11 && t.positions.len() > 400 && t.owner_raw.is_some()) {
                out.push(idx);
            }
        }
        eprintln!("{} film(s) for the first-person gates", out.len());
        out
    }

    /// #fp-jitter  **THE VIEW OF A WALKING PLAYER DOES NOT DEPEND ON THE DISPLAY RATE.** Reported: weird
    /// jitter in first person when spectating a player. The camera (and a followed body) were
    /// placed at the ROUNDED tick and the recorded look direction was held between reports, so at
    /// 144 Hz or with irregular frames the view stepped. Now position, aim and facing are all
    /// sampled at the fractional clock, the look turning along the shortest arc between reports a
    /// network-update apart. Gate: for the busiest 8 s of up to two on-foot subjects in each of five
    /// films, in first person and in follow, the 95th-percentile second difference of the camera
    /// at 144 fps / irregular frames stays within 15% / 50% of what 60 fps shows plus a small
    /// slack (60 fps samples the recorded keys themselves, so it is the data's own roughness).
    /// Before: 144 fps was 2.2x 60 fps in position on the race clip. Saboteur: read the rounded tick.
    #[test]
    #[ignore = "reads downloaded clips"]
    fn the_view_of_a_walking_player_does_not_depend_on_the_display_rate() {
        let mut judged = 0usize;
        let mut report = String::new();
        for idx in jitter_films() {
            let mut cands: Vec<(u8, usize)> = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.positions.len() > 400).filter_map(|t| t.owner_raw.map(|o| (o, t.positions.len()))).collect();
            cands.sort_by_key(|c| std::cmp::Reverse(c.1));
            cands.dedup_by_key(|c| c.0);
            report += &format!("film: {} owner candidate(s)\n", cands.len());
            let mut pb = Playback::new(idx);
            pb.context_cameras = false;
            for &(owner, _) in cands.iter().take(2) {
                let tr = pb.index.tracks.iter().filter(|t| t.owner_raw == Some(owner) && t.entity_type == 11).max_by_key(|t| t.positions.len()).unwrap().clone();
                let mut best = (0.0f32, tr.first_tick);
                let mut k = tr.first_tick + 120;
                while k + 600 < tr.last_tick {
                    let d = crate::film_vehicle::position_at(&tr, (k + 480) as f32).zip(crate::film_vehicle::position_at(&tr, k as f32)).map(|(a, b)| (a - b).length()).unwrap_or(0.0);
                    if d > best.0 && d < 200.0 { best = (d, k); }
                    k += 300;
                }
                for fp in [true, false] {
                    pb.camera = if fp { crate::film_play::FilmCamera::FirstPersonOwner(owner) } else { crate::film_play::FilmCamera::FollowOwner(owner) };
                    let at = |fps: f32, irr: bool, pb: &mut Playback| on_foot_camera_series(pb, best.1 as f32, 8.0, fps, irr);
                    let (Some(a), Some(b), Some(c)) = (at(60.0, false, &mut pb), at(144.0, false, &mut pb), at(90.0, true, &mut pb)) else {
                        eprintln!("owner {owner} fp {fp}: no series (window {best:?})");
                        continue;
                    };
                    judged += 1;
                    report += &format!("owner {owner} {} 60: pos {:.4} yaw {:.3} | 144: pos {:.4} yaw {:.3} | irr: pos {:.4} yaw {:.3}\n", if fp { "FP" } else { "FOLLOW" }, a.0, a.1, b.0, b.1, c.0, c.1);
                    assert!(b.0 <= a.0 * 1.15 + 0.004 && c.0 <= a.0 * 1.5 + 0.006, "{report}\nposition step grows with the display rate");
                    assert!(b.1 <= a.1 * 1.15 + 0.05 && c.1 <= a.1 * 1.5 + 0.08, "{report}\nlook step grows with the display rate");
                }
            }
        }
        eprintln!("{report}");
        assert!(judged >= 8, "only {judged} subject views judged: the gate proves nothing");
    }

    /// #fp-jitter  **A STANDING-STILL PLAYER'S VIEW IS PERFECTLY STABLE.** Wherever the recording holds a
    /// body at one position with no new look report for 3 s or more, the first-person camera at
    /// 144 fps does not move: position and yaw second differences are zero (1e-4 wu, 1e-3 deg).
    #[test]
    #[ignore = "reads downloaded clips"]
    fn a_standing_players_first_person_view_is_perfectly_stable() {
        let (mut windows, mut worst_p, mut worst_y) = (0usize, 0f32, 0f32);
        for idx in jitter_films() {
            let mut pb = Playback::new(idx);
            pb.context_cameras = false;
            let tracks: Vec<crate::film_decode::EntityTrack> = pb.index.tracks.iter().filter(|t| t.entity_type == 11 && t.owner_raw.is_some() && t.positions.len() > 50).cloned().collect();
            for tr in tracks.iter().take(40) {
                for w in tr.positions.windows(2) {
                    let ((t0, p0), (t1, p1)) = (w[0], w[1]);
                    if t1 - t0 < 200 || Vec3::from(p1).distance(Vec3::from(p0)) > 0.005 { continue; }
                    if tr.facings.iter().any(|&(k, _)| k > t0 && k < t1) { continue; }
                    // some player must still be the subject (the camera resolves) and not be a recorder with a live aim
                    let owner = tr.owner_raw.unwrap();
                    pb.camera = crate::film_play::FilmCamera::FirstPersonOwner(owner);
                    // a seated rider's view is its vehicle's, which moves: not a standing player
                    if pb.index.derived_valid().is_some_and(|d| d.rides.iter().any(|r| r.biped_net_id == tr.net_id && r.from_tick <= t1 && r.to_tick >= t0)) { continue; }
                    if pb.index.entity_for_owner(owner, t0 + 20).map(|e| e.net_id) != Some(tr.net_id) { continue; }
                    if pb.index.slot_of_owner(owner).is_some_and(|s| pb.aim_live.get(s as usize).copied().unwrap_or(false)) { continue; }
                    // exact: compare every sampled pose with the first, while this body is the subject
                    let mut first: Option<([f32; 3], f32)> = None;
                    let mut t = 0.0f32;
                    let mut seen = 0usize;
                    while t < (t1 - t0 - 40) as f32 / 60.0 {
                        t += 1.0 / 144.0;
                        let k = (t0 + 20) as f32 + t * 60.0;
                        pb.seek_tick(k);
                        let tick = k.round() as u32;
                        if pb.death_at(tr).is_some_and(|d| tick >= d) || pb.index.entity_for_owner(owner, tick).map(|e| e.net_id) != Some(tr.net_id) { continue; }
                        let Some(c) = crate::film_play::solve_camera_at(&pb, 0.62) else { continue };
                        let f0 = *first.get_or_insert((c.pos, c.yaw));
                        let dy = (c.yaw - f0.1 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                        worst_p = worst_p.max(Vec3::from(c.pos).distance(Vec3::from(f0.0)));
                        worst_y = worst_y.max(dy.to_degrees().abs());
                        seen += 1;
                    }
                    if seen < 100 { continue; }
                    windows += 1;
                }
            }
        }
        eprintln!("{windows} standing window(s): worst position drift {worst_p:.6} wu, yaw drift {worst_y:.5} deg");
        assert!(windows >= 3, "only {windows} standing windows found: the gate proves nothing");
        assert!(worst_p < 1e-4 && worst_y < 1e-3, "a standing player's view moves: {worst_p} wu, {worst_y} deg");
    }

    #[test]
    #[ignore = "opens a Reach cache and reads a downloaded race clip"]
    fn probe_follow_camera_jitter() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let p = PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov");
        let f = crate::film::open(&p).expect("opens");
        let idx = crate::film_decode::build_index_film(&f);
        let rider: u32 = 0xec0e029b;
        let veh: u32 = 0xebf0027d;
        let owner = idx.tracks.iter().find(|t| t.net_id == rider).and_then(|t| t.owner_raw).expect("owner");
        let vt = idx.tracks.iter().find(|t| t.net_id == veh).cloned().expect("vehicle");
        // a 4 s window where the hull is fast
        let mut t0 = 3000u32;
        let mut best = (0.0f32, 3000u32);
        while t0 + 300 < vt.last_tick {
            let d = crate::film_vehicle::position_at(&vt, (t0 + 240) as f32).zip(crate::film_vehicle::position_at(&vt, t0 as f32)).map(|(a, b)| a.distance(b)).unwrap_or(0.0);
            if d > best.0 {
                best = (d, t0);
            }
            t0 += 600;
        }
        let t0 = best.1;
        eprintln!("window t{t0}..{}: hull covers {:.0} wu in 4 s ({:.1} wu/s)", t0 + 240, best.0, best.0 / 4.0);
        let mut rig = Rig::new(&o, idx);
        for context in [true, false] {
            rig.pb.context_cameras = context;
            rig.pb.camera = crate::film_play::FilmCamera::FollowOwner(owner);
            for (label, fps, irregular) in [("30 fps", 30.0f32, false), ("60 fps", 60.0, false), ("144 fps", 144.0, false), ("irregular ~90", 90.0, true)] {
                let mut t = 0.0f32;
                let mut rngs = 12345u32;
                let mut screen: Vec<(f32, f32)> = Vec::new();
                let mut cam_lag: Vec<f32> = Vec::new();
                let mut label_seen = String::new();
                let frames = (4.0 * fps) as usize;
                for _ in 0..frames {
                    let dt = if irregular {
                        rngs = rngs.wrapping_mul(1664525).wrapping_add(1013904223);
                        (1.0 / fps) * (0.4 + 1.2 * ((rngs >> 8) as f32 / 16777216.0))
                    } else {
                        1.0 / fps
                    };
                    t += dt;
                    rig.pb.seek_tick(t0 as f32 + t * 60.0);
                    rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                    let Some(pose) = crate::film_play::solve_camera_at(&rig.pb, rig.driver.eye_height(&rig.pb)) else { continue };
                    label_seen = pose.label.clone();
                    let Some(&vid) = rig.driver.vehicle_actor.get(&veh) else { continue };
                    let Some(a) = rig.actors.get(vid) else { continue };
                    let vp = a.world_override.map(|m| m.w_axis.truncate()).unwrap_or(a.pos);
                    let cam = hms_render::Camera { pos: Vec3::from(pose.pos), yaw: pose.yaw, pitch: pose.pitch, fov_y: 60f32.to_radians(), near: 0.05, far: 50_000.0 };
                    let c = cam.view_proj(16.0 / 9.0) * vp.extend(1.0);
                    if c.w > 0.1 {
                        screen.push(((c.x / c.w) * 960.0, (c.y / c.w) * 540.0));
                    }
                    cam_lag.push((Vec3::from(pose.pivot.unwrap_or(pose.pos)) - vp).length());
                }
                let hp = |v: &[(f32, f32)]| -> f32 {
                    let d: Vec<f32> = v.windows(3).map(|w| { let (ex, ey) = (w[1].0 - 0.5 * (w[0].0 + w[2].0), w[1].1 - 0.5 * (w[0].1 + w[2].1)); ex * ex + ey * ey }).collect();
                    (d.iter().sum::<f32>() / d.len().max(1) as f32).sqrt()
                };
                let m = cam_lag.iter().sum::<f32>() / cam_lag.len().max(1) as f32;
                let jit = hp(&screen);
                eprintln!("CAMJ ctx={context} {label:<14}: vehicle on screen high-pass RMS {jit:.2} px | pivot-to-hull distance mean {m:.2} wu | {label_seen}");
                if !context {
                    // the PLAIN follow camera of a seated rider follows the vehicle, not the stale body
                    assert!(m < 3.0, "{label}: the plain follow camera is {m:.1} wu from the hull it is riding");
                    assert!(jit < 0.5, "{label}: the plain follow camera wobbles {jit:.2} px on the hull");
                }
                if context {
                    assert!(label_seen.contains("riding"), "liveness: the vehicle camera is not the one in use ({label_seen})");
                    assert!(screen.len() > 100, "liveness: only {} frames had the hull in view", screen.len());
                    assert!(jit < 0.5, "{label}: the followed hull wobbles {jit:.2} px in the view");
                }
            }
        }
    }

    /// #assassination-timing  print-only: the independent timeline of every stated assassination.
    #[test]
    #[ignore = "print-only probe"]
    fn probe_assassination_timeline() {
        let mut paths = crate::film::find_finished_films();
        paths.push(PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov"));
        let (mut total, mut with_flag) = (0usize, 0usize);
        let mut lead: Vec<i64> = Vec::new();
        for p in paths {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let asn = crate::film_anim::assassinations(&idx);
            if asn.is_empty() {
                continue;
            }
            let who: std::collections::HashSet<u32> = asn.iter().flat_map(|a| [a.attacker, a.victim]).flatten().collect();
            // every 0x35 statement on those bodies, in tick order
            let mut flags: std::collections::HashMap<u32, Vec<(u32, bool)>> = Default::default();
            for (i, rec) in f.data.records.iter().enumerate() {
                let Ok(t) = crate::film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
                for e in &t.entities {
                    if e.entity_type != 11 || !e.is_update || !who.contains(&e.net_id) {
                        continue;
                    }
                    let Some(u) = crate::film_anim::unit::decode(e, &idx.params, crate::film_decode::WEAPREF_BITS) else { continue };
                    if !u.trustworthy(e.dirty_mask) {
                        continue;
                    }
                    if e.dirty_mask & (1u64 << 0x35) != 0 {
                        flags.entry(e.net_id).or_default().push((i as u32, u.assassination.and_then(|a| a.target).is_some()));
                    }
                }
            }
            eprintln!("ASN {} | {} stated assassinations", p.file_name().unwrap().to_string_lossy(), asn.len());
            for a in &asn {
                total += 1;
                let (Some(k), Some(v)) = (a.attacker, a.victim) else { eprintln!("   t{} incident {} (no pair)", a.tick, a.incident); continue };
                let kf = flags.get(&k).cloned().unwrap_or_default();
                let vf = flags.get(&v).cloned().unwrap_or_default();
                let set = kf.iter().rev().find(|(t, on)| *on && *t <= a.tick + 30).copied();
                let clear = set.and_then(|(s, _)| kf.iter().find(|(t, on)| !*on && *t > s).copied());
                if let Some((s, _)) = set {
                    with_flag += 1;
                    lead.push(a.tick as i64 - s as i64);
                }
                eprintln!(
                    "   incident t{} ({}) killer {k:#x} victim {v:#x} | killer 0x35 set {:?} cleared {:?} | victim 0x35 msgs {:?} | request(22) move {:?} | victim last tick {:?}",
                    a.tick, a.incident, set.map(|x| x.0), clear.map(|x| x.0), vf.iter().take(4).collect::<Vec<_>>(), a.action_index,
                    idx.tracks.iter().find(|t| t.net_id == v).map(|t| t.last_tick)
                );
            }
        }
        lead.sort();
        eprintln!("ASN total {total}, killer 0x35 flag found for {with_flag}; incident minus flag-set ticks: {lead:?}");
    }

    /// #assassination-timing  print-only: when do the two bodies actually lock together, relative to
    /// the incident? (killer-victim distance and the victim's own movement, tick by tick)
    #[test]
    #[ignore = "print-only probe"]
    fn probe_assassination_lock_tick() {
        let mut paths = crate::film::find_finished_films();
        paths.push(PathBuf::from(std::env::var("HOME").unwrap()).join("Downloads").join("Reach(1).mov"));
        let mut leads: Vec<i64> = Vec::new();
        let mut ends: Vec<i64> = Vec::new();
        for p in paths {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            for a in crate::film_anim::assassinations(&idx) {
                let (Some(k), Some(v)) = (a.attacker, a.victim) else { continue };
                let (Some(kt), Some(vt)) = (idx.tracks.iter().find(|t| t.net_id == k), idx.tracks.iter().find(|t| t.net_id == v)) else { continue };
                // first tick in [I-240, I] after which killer and victim stay within 1.2 wu through the incident
                let mut lock: Option<u32> = None;
                let from = a.tick.saturating_sub(240);
                let mut t = a.tick;
                while t > from {
                    match (kt.sample(t), vt.sample(t)) {
                        (Some(pk), Some(pv)) if glam::Vec3::from(pk).distance(glam::Vec3::from(pv)) < 1.2 => lock = Some(t),
                        _ => break,
                    }
                    t -= 1;
                }
                // the victim's last replicated position tick and the killer's, after the incident
                let end = vt.last_tick as i64 - a.tick as i64;
                if let Some(l) = lock {
                    leads.push(a.tick as i64 - l as i64);
                }
                ends.push(end);
            }
        }
        leads.sort();
        ends.sort();
        eprintln!("LOCK incident minus first tick of continuous <1.2 wu contact ({} cases): {leads:?}", leads.len());
        eprintln!("LOCK victim last replicated tick minus incident ({} cases): {ends:?}", ends.len());
    }

    /// **#assassination-timing  THE GATE for "assassinations are a little delayed on the animation".**
    /// Through the production `ActorDriver::sync` on real clips with stated assassinations:
    ///
    /// * the killer's and the victim's move starts on the incident tick, on its FIRST frame, and with
    ///   NO cross-fade (`blend_left == 0`): the sync action snaps both bodies onto the shared root;
    /// * a killer who finishes another victim inside the first move's window (kill chains) starts
    ///   the NEW move at the new incident (frame 0 there), not 70 ticks later when the old window ends.
    ///
    /// Liveness: at least 8 plain assassinations and 3 chained ones. Saboteurs: put the 0.18 s
    /// cross-fade back, or pick the oldest covering action again.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn assassination_moves_start_on_their_incident_with_no_fade_and_chains_restart() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let (mut plain, mut chained, mut shown) = (0usize, 0usize, 0usize);
        for p in crate::film::find_finished_films() {
            if plain >= 12 && chained >= 3 {
                break;
            }
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let asn: Vec<_> = crate::film_anim::assassinations(&idx).into_iter().filter(|a| a.attacker.is_some() && a.victim.is_some()).collect();
            if asn.len() < 2 {
                continue;
            }
            let mut rig = Rig::new(&o, idx);
            for (n, a) in asn.iter().enumerate().take(14) {
                let (k, v) = (a.attacker.unwrap(), a.victim.unwrap());
                let chain = n > 0 && asn[n - 1].attacker == Some(k) && a.tick - asn[n - 1].tick < crate::film_anim::SYNC_TICKS;
                // the tick before the incident, then the incident, then a few ticks into the move
                let mut at_incident: Vec<(u32, bool, f32, f32)> = Vec::new();
                for t in [a.tick.saturating_sub(1), a.tick, a.tick + 10] {
                    rig.pb.seek_tick(t as f32);
                    // let any cross-fade armed by earlier ticks finish (the rig has no frame clock)
                    rig.actors.update_poses_headless(1.0);
                    rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                    for id in [k, v] {
                        let Some(&aid) = rig.driver.body_actor.get(&id) else { continue };
                        let Some(ac) = rig.actors.get(aid) else { continue };
                        at_incident.push((t, crate::film_anim::is_sync_clip(&ac.anim_name), ac.frame, ac.blend_left));
                    }
                }
                // entries: (before: killer, victim), (incident: killer, victim), (+10: killer, victim)
                if at_incident.len() < 6 {
                    continue;
                }
                let (inc_k, plus_k) = (at_incident[2], at_incident[4]);
                assert!(inc_k.1, "incident t{}: the killer is not in the assassination clip on the incident tick", a.tick);
                assert!(inc_k.2 <= 1.01, "incident t{}: the move starts on frame {} instead of 0 (chain: {chain})", a.tick, inc_k.2);
                let restarted = asn.iter().any(|b| b.attacker == Some(k) && b.tick > a.tick && b.tick <= a.tick + 10);
                assert!(restarted || (plus_k.2 > 3.0 && plus_k.2 < 8.0), "incident t{}: 10 ticks in, the move is at frame {} (want ~5)", a.tick, plus_k.2);
                assert_eq!(inc_k.3, 0.0, "incident t{}: the killer's move starts through a cross-fade ({} s left)", a.tick, inc_k.3);
                if chain { chained += 1 } else { plain += 1 }
                if shown < 3 {
                    shown += 1;
                    eprintln!("   incident t{} chain={chain}: killer on the clip at frame {:.1}, no fade; +10 ticks frame {:.1}", a.tick, inc_k.2, plus_k.2);
                }
            }
        }
        eprintln!("assassination starts: {plain} plain, {chained} chained");
        assert!(plain >= 8 && chained >= 3, "liveness: {plain} plain / {chained} chained assassinations were exercised");
    }

    /// #melee-timing  print-only: what Theater draws for each recorded melee, and when the hit lands.
    #[test]
    #[ignore = "print-only probe"]
    fn probe_melee_inventory() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let mut by_name: std::collections::BTreeMap<String, (usize, u32, u32)> = Default::default();
        let mut hit_offsets: Vec<i64> = Vec::new();
        let (mut events, mut no_clip) = (0usize, 0usize);
        for p in crate::film::find_finished_films().into_iter().take(14) {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let swings: Vec<(u32, u32)> = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_MELEE_INITIATE).filter_map(|e| Some((e.tick, e.subject?))).collect();
            let hits: Vec<(u32, u32)> = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_MELEE_DAMAGE).filter_map(|e| Some((e.tick, e.subject?))).collect();
            if swings.is_empty() {
                continue;
            }
            for &(t, who) in &swings {
                if let Some(&(ht, _)) = hits.iter().find(|&&(ht, hs)| hs == who && ht >= t && ht <= t + 120) {
                    hit_offsets.push(ht as i64 - t as i64);
                }
            }
            let mut rig = Rig::new(&o, idx);
            for &(t, who) in swings.iter().take(10) {
                events += 1;
                let mut name = String::new();
                let (mut active_until, mut last_frame) = (0u32, 0.0f32);
                for dt in 0..90u32 {
                    rig.pb.seek_tick((t + dt) as f32);
                    rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                    let Some(&aid) = rig.driver.body_actor.get(&who) else { break };
                    let Some(ac) = rig.actors.get(aid) else { break };
                    if let Some((_, fr)) = ac.replacement {
                        name = ac.replacement_name.clone();
                        active_until = dt + 1;
                        last_frame = fr;
                    }
                }
                if name.is_empty() {
                    no_clip += 1;
                    continue;
                }
                let e = by_name.entry(name).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 = e.1.max(active_until);
                e.2 = e.2.max(last_frame as u32);
            }
        }
        hit_offsets.sort();
        eprintln!("MELEE {events} swings sampled, {no_clip} with NO replacement clip drawn");
        for (n, (c, until, lastf)) in &by_name {
            eprintln!("MELEE clip {n}: {c} swings, replacement active for up to {until} ticks, last frame {lastf}");
        }
        eprintln!("MELEE recorded hit lands this many ticks after the swing event ({}): {:?}", hit_offsets.len(), hit_offsets);
    }

    /// #melee-timing  print-only: how are the recorded melee events spaced, and how long after the
    /// swing does the damage event land?
    #[test]
    #[ignore = "print-only probe"]
    fn probe_melee_event_spacing() {
        let (mut gaps, mut hit): (Vec<u32>, Vec<i64>) = (Vec::new(), Vec::new());
        let mut per_kind: std::collections::BTreeMap<u8, usize> = Default::default();
        for p in crate::film::find_finished_films() {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let mut last: std::collections::HashMap<u32, u32> = Default::default();
            for e in &idx.events {
                *per_kind.entry(e.kind).or_default() += (e.kind == crate::film_action::EV_MELEE_INITIATE || e.kind == crate::film_action::EV_MELEE_DAMAGE || e.kind == crate::film_action::EV_MELEE_CLANG) as usize;
                if e.kind == crate::film_action::EV_MELEE_INITIATE {
                    if let Some(s) = e.subject {
                        if let Some(&t0) = last.get(&s) {
                            gaps.push(e.tick - t0);
                        }
                        last.insert(s, e.tick);
                    }
                }
                if e.kind == crate::film_action::EV_MELEE_DAMAGE {
                    if let Some(s) = e.subject.or(e.other) {
                        if let Some(&t0) = last.get(&s) {
                            hit.push(e.tick as i64 - t0 as i64);
                        }
                    }
                }
            }
        }
        gaps.sort();
        hit.sort();
        let q = |v: &[u32], f: f32| v.get(((v.len() as f32) * f) as usize).copied();
        eprintln!("MSP kinds {per_kind:?}; consecutive swing gaps n={} p10 {:?} p50 {:?} p90 {:?}; under 60: {}", gaps.len(), q(&gaps, 0.1), q(&gaps, 0.5), q(&gaps, 0.9), gaps.iter().filter(|g| **g < 60).count());
        eprintln!("MSP damage minus last swing: n={} {:?}", hit.len(), hit.iter().take(40).collect::<Vec<_>>());
    }

    /// #melee-class  print-only: which weapon/class does Theater resolve for each recorded swing?
    #[test]
    #[ignore = "print-only probe"]
    fn probe_melee_weapon_classes() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let mut classes: std::collections::BTreeMap<String, usize> = Default::default();
        let mut tags: std::collections::BTreeMap<String, usize> = Default::default();
        for p in crate::film::find_finished_films().into_iter().take(20) {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let swings: Vec<(u32, u32)> = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_MELEE_INITIATE).filter_map(|e| Some((e.tick, e.subject?))).collect();
            if swings.is_empty() {
                continue;
            }
            let mut rig = Rig::new(&o, idx);
            rig.pb.seek_tick(swings[0].0 as f32);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            for &(t, who) in &swings {
                let Some(tr) = rig.pb.index.tracks.iter().find(|x| x.net_id == who) else { continue };
                let w = tr.weapon_at(t);
                let n = w.and_then(|w| rig.driver.weap_tags.get(w as usize).copied()).and_then(|tag| o.native.tag_name(o.cache, tag)).map(|n| n.rsplit('\\').next().unwrap_or(&n).to_string()).unwrap_or_else(|| "<none>".into());
                *tags.entry(n).or_default() += 1;
                let c = w.and_then(|w| rig.driver.weap_class.get(w as usize).cloned()).flatten().unwrap_or_else(|| "<unresolved>".into());
                *classes.entry(c).or_default() += 1;
            }
        }
        eprintln!("MCL weapons at the swing tick: {tags:?}");
        eprintln!("MCL classes: {classes:?}");
    }

    /// #melee-class  print-only: every melee clip the Spartan graph authors.
    #[test]
    #[ignore = "print-only probe"]
    fn probe_melee_graph_names() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let p = crate::film::find_finished_films().into_iter().next().expect("a film");
        let f = crate::film::open(&p).expect("opens");
        let idx = crate::film_decode::build_index_film(&f);
        let mut rig = Rig::new(&o, idx);
        rig.pb.seek_tick(2000.0);
        rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
        let g = rig.actors.actors.iter().find_map(|a| a.def.graph.clone()).expect("a graph");
        let mut names: Vec<String> = g.anims.iter().filter(|a| a.name.contains("melee") || a.name.contains("lunge")).map(|a| format!("{} [{:?} {}f]", a.name, a.anim_type, a.frame_count)).collect();
        names.sort();
        std::fs::write(std::env::temp_dir().join("hms_melee_names.txt"), names.join("\n")).unwrap();
    }

    /// **#melee  THE GATE for "melee animations look weird in Theater".** On real clips, through the
    /// production `ActorDriver::sync`: an undisturbed swing lasts exactly its AUTHORED length (the clip's
    /// own frame count x 2 ticks, not a fixed 60), fades in (first tick's weight under 0.35) and out (the
    /// last tick's under 0.35) instead of snapping on and off the locomotion clip. A swing cut short by
    /// the body dying / being locked into another action is counted, not judged.
    ///
    /// The weapon CLASS is reported, not fixed: only a few percent of swings have a class the film
    /// states (see the numbers printed). Saboteur: set the weight to 1 everywhere.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn a_swing_uses_the_weapon_in_the_hand_and_fades_in_and_out() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(o) = open_forge() else { return };
        let (mut swings, mut stated, mut judged, mut interrupted, mut bad) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut classes: std::collections::BTreeMap<String, usize> = Default::default();
        let mut shown = 0;
        for p in crate::film::find_finished_films().into_iter().take(30) {
            if judged >= 40 {
                break;
            }
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let ev: Vec<(u32, u32)> = idx.events.iter().filter(|e| e.kind == crate::film_action::EV_MELEE_INITIATE).filter_map(|e| Some((e.tick, e.subject?))).collect();
            if ev.is_empty() {
                continue;
            }
            let mut rig = Rig::new(&o, idx);
            rig.pb.seek_tick(ev[0].0 as f32);
            rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
            for &(t, who) in &ev {
                if let Some(tr) = rig.pb.index.tracks.iter().find(|x| x.net_id == who) {
                    swings += 1;
                    stated += tr.weapon_at(t).and_then(|w| rig.driver.weap_class.get(w as usize).cloned().flatten()).is_some() as usize;
                }
            }
            for &(t, who) in ev.iter().take(10) {
                if ev.iter().any(|&(t2, w2)| w2 == who && t2 > t && t2 < t + 80) {
                    continue; // a following swing restarts the clip
                }
                let mut active: Vec<(u32, f32, String)> = Vec::new();
                let mut n_frames = 0.0f32;
                for dt in 0..80u32 {
                    rig.pb.seek_tick((t + dt) as f32);
                    rig.driver.sync(&rig.pb, &mut rig.actors, &o.native, o.cache);
                    let Some(&aid) = rig.driver.body_actor.get(&who) else { break };
                    let Some(ac) = rig.actors.get_mut(aid) else { break };
                    if ac.replacement.is_some() {
                        let name = ac.replacement_name.clone();
                        if n_frames == 0.0 {
                            n_frames = ac.replacement_clip(&o.native, o.cache, &name).map_or(0.0, |(_, n)| n);
                        }
                        active.push((dt, ac.replacement_weight, name));
                    }
                }
                let (Some(first), Some(last)) = (active.first(), active.last()) else { continue };
                let len = (last.0 - first.0 + 1) as f32;
                // the window ends where the clip does, unless something else took the body
                if (len - n_frames * 2.0).abs() > 1.5 {
                    interrupted += 1;
                    continue;
                }
                judged += 1;
                *classes.entry(first.2.split(':').nth(1).unwrap_or("?").to_string()).or_default() += 1;
                if first.1 > 0.35 || last.1 > 0.35 {
                    bad += 1;
                    if shown < 4 {
                        shown += 1;
                        eprintln!("   bad swing at t{t}: first weight {:.2}, last weight {:.2}, {len} ticks of {}", first.1, last.1, n_frames * 2.0);
                    }
                }
            }
        }
        eprintln!("melee: {swings} swings; film states the weapon class for {stated}; {judged} undisturbed swings judged, {interrupted} cut short, classes {classes:?}, {bad} without a fade");
        assert!(judged >= 20, "liveness: only {judged} undisturbed swings were judged");
        assert_eq!(bad, 0, "{bad} swings snap on or off");
    }
}
