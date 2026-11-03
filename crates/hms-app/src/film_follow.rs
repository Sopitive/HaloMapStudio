//! #film-follow  STREAM a growing `.temp` (or any film) as JSON lines, with NO window, NO GPU
//! device, NO audio and nothing rendered.
//!
//! This module imports only the film readers (`film*`), `serde_json` and `std`. It never touches
//! `wgpu`, `egui`, `eframe` or an audio crate, and `main.rs` dispatches `--film-follow` before any
//! of those are initialised (the same arrangement as `--film-stats`). A gate greps this file's own
//! source for those names, so the property cannot silently regress.
//!
//! # How it follows
//!
//! The game rewrites a `.temp` roughly every 15 s; every rewrite is the same stream with more
//! records at the end (block-preallocated, so the reader stops at the last COMPLETE record). A
//! refresh re-reads the file, re-frames it (94 M records/s) and decodes ONLY the records it has
//! not seen: all state (decode parameters, entity tracks, roster, megalo values, score) lives in
//! [`Follower`] and is advanced record by record, so cost per refresh is proportional to what is
//! new. If the file's identity changes (a different match, or fewer records than before) the
//! follower resets and says so.
//!
//! # Determinism (what makes `--since` safe)
//!
//! Processing is strictly sequential over records, and every decision uses only state built from
//! earlier records, so the events produced for a prefix of the file are exactly a prefix of the
//! events produced for the whole file, however the file was chunked. Event ids are
//! `<tick>-<n>`, `n` counting events within a tick. Decode parameters are solved once from the
//! FIRST [`PARAM_RECORDS`] records; nothing is emitted before that many exist.
//!
//! # Event schema (one JSON object per line; every event has `id`, `tick`, `seconds`, `type`)
//!
//! | type | fields |
//! |------|--------|
//! | `reset` | `reason` |
//! | `player_join` | `slot`, `name`, `team`, `identifier`, `joined_in_progress` |
//! | `player_leave` | `slot`, `name`, `reason` (`left_game` / `removed`) |
//! | `player_swap` | `slot`, `with_slot`, `name` |
//! | `player_update` | `slot`, `name`, `team` (configuration change) |
//! | `incident` | `incident`, `subject`, `other`, `subject_slot/name`, `other_slot/name` (kills, medals) |
//! | `event` | `kind`, `name`, `subject`, `other` (only with `all_events`) |
//! | `score` | `statborg`, `is_team`, `slot`/`team`, `name`, `round_score`, `total_score`, `rounds_won`, `final` |
//! | `megalo` | `scope`, `owner`, `owner_name`, `kind`, `slot`, `value`, `text` |
//! | `game` | `name`, `value` (current_state, game_finished, current_round, round_condition_flags) |
//! | `object_spawn` / `object_delete` | `id`, `entity_type`, `type`, `tag`, `label`, `position`, `team`, `shape` |
//! | `zone_enter` / `zone_exit` | `zone_id`, `zone_label`, `occupant_id`, `occupant_type`, `slot`, `name` |

use crate::film::Film;
use crate::film_decode::{EntityTrack, PositionFrame};
use crate::film_megalo::{self, Key, MegaloDecls, Scope};
use crate::film_objects::{self as fo, Filter, ObjRow, Volume};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Records needed before decoding starts (decode parameters are solved from exactly these).
pub const PARAM_RECORDS: usize = 256;

#[derive(Clone, Debug, Default)]
pub struct Opts {
    pub since: Option<u32>,
    pub watch_labels: Vec<String>,
    pub watch_ids: Vec<u32>,
    /// Sphere radius for a watched object that has no boundary of its own.
    pub watch_radius: Option<f32>,
    pub filter: Filter,
    /// Emit `object_spawn` / `object_delete`.
    pub objects: bool,
    pub all_events: bool,
    /// Also emit the once-a-second round clock.
    pub clock: bool,
}

pub struct Event {
    pub id: String,
    pub tick: u32,
    pub json: Value,
}

struct FObj {
    row: ObjRow,
    tr: EntityTrack,
    announced: bool,
}

#[derive(Default)]
pub struct RefreshStats {
    pub records_total: usize,
    pub records_new: usize,
    pub events: usize,
    pub wall: std::time::Duration,
}

pub struct Follower {
    pub opts: Opts,
    done: usize,
    fingerprint: Option<(String, u64)>,
    params: Option<crate::film_decode::DecodeParams>,
    frame: Option<PositionFrame>,
    base_tick: Option<u32>,
    decls: Option<std::sync::Arc<MegaloDecls>>,
    watched: Vec<u32>,
    occupants: Vec<u32>,
    objs: HashMap<u32, FObj>,
    order: Vec<u32>,
    updates: Vec<(u32, crate::film_roster::PlayerUpdate)>,
    roster: crate::film_roster::Roster,
    teams_enabled: Option<bool>,
    biped_payloads: Vec<(u32, Vec<u8>, u32)>,
    eqip: Option<u32>,
    bodies: HashMap<u32, crate::film_roster::Verdict>,
    statborg_slot: HashMap<u32, u8>,
    statborg: Vec<crate::film_decode::StatborgSample>,
    slot_of: HashMap<u32, u8>,
    team_of: HashMap<u32, u8>,
    mval: HashMap<Key, film_megalo::Value>,
    gval: HashMap<&'static str, i64>,
    zone_inside: HashMap<u32, Vec<u32>>,
    pub last: RefreshStats,
}

fn fp_of(f: &Film) -> (String, u64) {
    let mut h = 0xcbf29ce484222325u64;
    if let Some(r) = f.data.records.first() {
        for b in r.body(&f.bytes).iter().take(64) {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
    }
    (f.header.session_id.clone(), h)
}

impl Follower {
    pub fn new(opts: Opts) -> Follower {
        Follower {
            opts,
            done: 0,
            fingerprint: None,
            params: None,
            frame: None,
            base_tick: None,
            decls: None,
            watched: Vec::new(),
            occupants: Vec::new(),
            objs: HashMap::new(),
            order: Vec::new(),
            updates: Vec::new(),
            roster: Default::default(),
            teams_enabled: None,
            biped_payloads: Vec::new(),
            eqip: None,
            bodies: HashMap::new(),
            statborg_slot: HashMap::new(),
            statborg: vec![Default::default(); crate::film_decode::STATBORG_SLOTS as usize],
            slot_of: HashMap::new(),
            team_of: HashMap::new(),
            mval: HashMap::new(),
            gval: HashMap::new(),
            zone_inside: HashMap::new(),
            last: Default::default(),
        }
    }

    fn reset(&mut self) {
        let opts = std::mem::take(&mut self.opts);
        *self = Follower::new(opts);
    }

    /// Read the file and advance. Returns the NEW events (already filtered by `since`).
    pub fn refresh(&mut self, f: &Film, names: Option<&dyn crate::film_stats::NameSource>) -> Vec<Event> {
        let t0 = std::time::Instant::now();
        let mut out: Vec<Event> = Vec::new();
        let fp = fp_of(f);
        let n = f.data.records.len();
        if self.fingerprint.as_ref().map_or(false, |p| *p != fp) || n < self.done {
            self.reset();
            out.push(Event { id: "reset".into(), tick: 0, json: json!({"id": "reset", "tick": 0, "seconds": 0.0, "type": "reset", "reason": "the file's identity changed or it shrank"}) });
        }
        self.fingerprint = Some(fp);
        self.last.records_total = n;
        if self.params.is_none() {
            if n < PARAM_RECORDS {
                self.last.records_new = 0;
                self.last.wall = t0.elapsed();
                return out;
            }
            let mut fd = f.data.clone();
            fd.records.truncate(PARAM_RECORDS);
            let (base, _) = crate::film_decode::solve_params(&f.bytes, &fd);
            let (frame, w) = crate::film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
            self.params = Some(crate::film_decode::DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base });
            self.frame = if f.header.map_variant.is_empty() { None } else { Some(frame) };
            self.decls = film_megalo::decls_of(f).map(std::sync::Arc::new);
            self.teams_enabled = f.header.gametype.as_ref().map(|g| g.misc.teams_enabled());
            self.roster = crate::film_roster::Roster::build(&f.header.players, &self.updates);
        }
        let params = self.params.unwrap();
        let from = self.done;
        for i in from..n {
            let rec = f.data.records[i];
            let Ok(t) = crate::film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            let b = *self.base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
            let tick = t.tick.wrapping_sub(b);
            let mut seq = 0u32;
            let mut evs: Vec<Value> = Vec::new();
            self.step(f, &t, tick, &params, names, &mut evs);
            for mut e in evs {
                let id = format!("{tick}-{seq}");
                seq += 1;
                e["id"] = json!(id);
                e["tick"] = json!(tick);
                e["seconds"] = json!((tick as f64 / 60.0 * 1000.0).round() / 1000.0);
                if self.opts.since.map_or(true, |s| tick > s) {
                    out.push(Event { id, tick, json: e });
                }
            }
        }
        self.done = n;
        self.last.records_new = n - from;
        self.last.events = out.len();
        self.last.wall = t0.elapsed();
        out
    }

    fn name_of_slot(&self, slot: u8, tick: u32) -> Option<String> {
        self.roster.seat_holding(slot, tick).map(|s| s.name.clone()).filter(|s| !s.is_empty())
    }

    fn who(&self, id: Option<u32>) -> (Option<u8>, Option<String>) {
        match id.and_then(|i| self.bodies.get(&i)) {
            Some(crate::film_roster::Verdict::Named(i, _)) => {
                let s = &self.roster.seats[*i];
                (Some(s.slot), Some(s.name.clone()))
            }
            _ => (None, None),
        }
    }

    fn step(&mut self, f: &Film, t: &crate::film_decode::TickRecord, tick: u32, params: &crate::film_decode::DecodeParams, names: Option<&dyn crate::film_stats::NameSource>, evs: &mut Vec<Value>) {
        use crate::film_roster::UpdateKind;
        // ---- roster (list A kind 9)
        let mut roster_dirty = false;
        for c in t.commands.iter().take(t.commands_a).filter(|c| c.kind == 9) {
            let Some(u) = crate::film_roster::decode_player_update(&c.payload) else { continue };
            let before = self.name_of_slot(u.player, tick);
            let cfg_name = u.config.as_ref().and_then(|c| c.name.clone());
            let cfg_team = u.config.as_ref().and_then(|c| c.team);
            let ev = match u.kind {
                UpdateKind::Add => json!({"type": "player_join", "slot": u.player, "name": cfg_name, "team": cfg_team, "identifier": format!("{:016X}", u.identifier), "joined_in_progress": u.joined_in_progress}),
                UpdateKind::LeftGame => json!({"type": "player_leave", "slot": u.player, "name": before, "reason": "left_game"}),
                UpdateKind::Remove => json!({"type": "player_leave", "slot": u.player, "name": before, "reason": "removed"}),
                UpdateKind::Swap => json!({"type": "player_swap", "slot": u.player, "with_slot": u.swap_with.map(|s| s.0), "name": before}),
                UpdateKind::Configuration => json!({"type": "player_update", "slot": u.player, "name": cfg_name.or(before), "team": cfg_team}),
            };
            evs.push(ev);
            self.updates.push((tick, u));
            roster_dirty = true;
        }
        if roster_dirty {
            self.roster = crate::film_roster::Roster::build(&f.header.players, &self.updates);
        }
        // ---- events
        for c in t.commands.iter().skip(t.commands_a).filter(|c| c.kind == 1) {
            let Some(e) = crate::film_action::fold(tick, &c.payload) else { continue };
            let (ss, sn) = self.who(e.subject);
            let (os, on) = self.who(e.other);
            if e.kind == 14 {
                evs.push(json!({"type": "incident", "incident": e.incident.and_then(crate::film_action::incident_name), "incident_id": e.incident, "subject": e.subject, "other": e.other, "subject_slot": ss, "subject_name": sn, "other_slot": os, "other_name": on}));
            } else if self.opts.all_events {
                evs.push(json!({"type": "event", "kind": e.kind, "name": crate::film_action::event_name(e.kind), "subject": e.subject, "other": e.other, "subject_slot": ss, "subject_name": sn, "other_slot": os, "other_name": on}));
            }
        }
        // ---- destroys come BEFORE zone evaluation (a destroyed body is gone at its destroy tick)
        for &(_ty, net) in &t.destroyed {
            if let Some(o) = self.objs.get_mut(&net) {
                if o.row.destroyed_tick.is_none() {
                    o.row.destroyed_tick = Some(tick);
                    o.tr.destroyed_tick = Some(tick);
                    if self.opts.objects && o.announced {
                        evs.push(object_event("object_delete", &o.row, &o.tr, tick));
                    }
                }
            }
        }
        // ---- entity messages
        let mut created: Vec<u32> = Vec::new();
        for e in &t.entities {
            if e.entity_type == 5 {
                if let Some(sb) = crate::film_decode::decode_statborg(e, params) {
                    if sb.trustworthy(e.dirty_mask) {
                        if let Some(i) = sb.index {
                            self.statborg_slot.insert(e.net_id, i);
                        }
                        if let Some(&slot) = self.statborg_slot.get(&e.net_id) {
                            let cur = &mut self.statborg[slot as usize];
                            let mut next = *cur;
                            if sb.round_score.is_some() { next.round_score = sb.round_score; }
                            if sb.total_score.is_some() { next.total_score = sb.total_score; }
                            if sb.rounds_won.is_some() { next.rounds_won = sb.rounds_won; }
                            if sb.is_final.is_some() { next.is_final = sb.is_final; }
                            if next != *cur {
                                *cur = next;
                                let team = slot >= crate::film_decode::STATBORG_PLAYERS;
                                let name = if team { None } else { self.name_of_slot(slot, tick) };
                                evs.push(json!({"type": "score", "statborg": slot, "is_team": team,
                                    "slot": (!team).then_some(slot), "team": team.then(|| slot - crate::film_decode::STATBORG_PLAYERS),
                                    "name": name, "round_score": next.round_score, "total_score": next.total_score,
                                    "rounds_won": next.rounds_won, "final": next.is_final}));
                            }
                        }
                    }
                }
                continue;
            }
            if let Some(d) = self.decls.clone() {  // Arc clone
                let walk = match e.entity_type {
                    3 => film_megalo::walk_player(e, params, &d),
                    4 => film_megalo::walk_team(e, params, &d),
                    0 | 1 => film_megalo::walk_globals(e, params, &d),
                    _ => None,
                };
                if let Some(w) = walk {
                    if w.trustworthy(e.dirty_mask) {
                        let (scope, owner) = match e.entity_type {
                            3 => {
                                if let Some(o) = w.owner { self.slot_of.insert(e.net_id, o); }
                                (Scope::Player, self.slot_of.get(&e.net_id).copied())
                            }
                            4 => {
                                if let Some(o) = w.owner { self.team_of.insert(e.net_id, o); }
                                (Scope::Team, self.team_of.get(&e.net_id).copied())
                            }
                            _ => (Scope::Global, Some(0)),
                        };
                        if let Some(owner) = owner {
                            for (kind, slot, v) in &w.vars {
                                let key = Key { scope, owner, kind: *kind, slot: *slot };
                                if self.mval.get(&key) != Some(v) {
                                    self.mval.insert(key, *v);
                                    let nm = if scope == Scope::Player { self.name_of_slot(owner, tick) } else { None };
                                    evs.push(json!({"type": "megalo", "scope": scope.name(), "owner": owner, "owner_name": nm, "kind": kind.name(), "slot": slot, "value": v.number(), "text": v.text()}));
                                }
                            }
                        }
                        for (name, v) in &w.game {
                            if *name == "round_timer_seconds" && !self.opts.clock {
                                continue;
                            }
                            if matches!(*name, "sudden_death_seconds" | "grace_period_seconds") && !self.opts.clock {
                                continue;
                            }
                            if self.gval.get(name) != Some(v) {
                                self.gval.insert(*name, *v);
                                evs.push(json!({"type": "game", "name": name, "value": v}));
                            }
                        }
                    }
                }
            }
            if e.entity_type < 11 || e.entity_type > 18 {
                continue;
            }
            // ---- object tracks (same append rules as the batch index)
            if !self.objs.contains_key(&e.net_id) {
                let row = ObjRow {
                    net_id: e.net_id,
                    entity_type: e.entity_type,
                    def_wire: None,
                    tag: None,
                    first_tick: tick,
                    last_tick: tick,
                    destroyed_tick: None,
                    first_pos: None,
                    placed: None,
                    mp: None,
                    parent: None,
                    owner_slot: None,
                    owner_name: None,
                    megalo_ref: false,
                };
                let tr = EntityTrack { net_id: e.net_id, entity_type: e.entity_type, first_tick: tick, last_tick: tick, ..Default::default() };
                self.objs.insert(e.net_id, FObj { row, tr, announced: false });
                self.order.push(e.net_id);
                if matches!(e.entity_type, 11 | 15) {
                    self.occupants.push(e.net_id);
                }
                created.push(e.net_id);
            }
            if !e.is_update {
                if let Some(w) = crate::film_action::creation_def_wire(e) {
                    let o = self.objs.get_mut(&e.net_id).unwrap();
                    o.row.def_wire.get_or_insert(w);
                    if o.row.tag.is_none() {
                        o.row.tag = names.and_then(|n| n.definition(w)).map(|(_, _, p)| p);
                    }
                    o.tr.def_wire.get_or_insert(w);
                }
                if e.entity_type == 11 {
                    self.biped_payloads.push((e.net_id, e.payload.clone(), tick));
                }
            }
            if let Some(st) = crate::film_decode::decode_entity_payload(e, params) {
                let frame = self.frame;
                let o = self.objs.get_mut(&e.net_id).unwrap();
                o.tr.last_tick = tick;
                o.row.last_tick = tick;
                if let (Some(raw), Some(fr)) = (st.position_raw, frame) {
                    let p = fr.world(raw);
                    if p.iter().all(|v| v.is_finite()) {
                        o.tr.positions.push((tick, p));
                        if o.row.first_pos.is_none() {
                            o.row.first_pos = Some(p);
                        }
                    }
                }
                if let Some(or) = st.orientation_raw {
                    let (fw, up) = crate::film_decode::orientation_world_after(or, o.tr.orientations.last().map(|k| (k.1, k.2)));
                    o.tr.orientations.push((tick, fw, up));
                }
                if st.rebuilt_mask & (1 << 12) != 0 {
                    let last = o.tr.attachments.last().map(|(_, a)| *a);
                    if last != Some(st.attach) {
                        o.tr.attachments.push((tick, st.attach));
                    }
                    o.row.parent = o.tr.attachments.iter().find_map(|(_, a)| a.map(|a| a.parent));
                }
                if let Some(m) = st.mp {
                    o.row.mp = Some(m);
                }
            }
        }
        // ---- bodies: lock the 'eqip' width from the first 16 biped creations, then attribute
        if self.eqip.is_none() && self.biped_payloads.len() >= 16 {
            let pls: Vec<Vec<u8>> = self.biped_payloads.iter().take(16).map(|(_, p, _)| p.clone()).collect();
            self.eqip = Some(crate::film_decode::solve_eqip_bits(&pls, params).0);
        }
        if let Some(eq) = self.eqip {
            let pending: Vec<(u32, Vec<u8>, u32)> = std::mem::take(&mut self.biped_payloads);
            for (net, pl, ctick) in pending {
                if let Some(c) = crate::film_decode::creation_biped(&pl, params, eq) {
                    let v = self.roster.attribute(c.look.as_ref(), (c.owner + 1) as u8, ctick, self.teams_enabled);
                    if let Some(o) = self.objs.get_mut(&net) {
                        if let crate::film_roster::Verdict::Named(i, _) = &v {
                            o.row.owner_slot = Some(self.roster.seats[*i].slot);
                            o.row.owner_name = Some(self.roster.seats[*i].name.clone());
                        }
                    }
                    self.bodies.insert(net, v);
                }
            }
        }
        // ---- spawns: join to the Forge placement by first position (end of the record)
        let mv = &f.header.map_variant;
        for net in created {
            let first_pos = self.objs[&net].row.first_pos;
            if let Some(p) = first_pos {
                if matches!(self.objs[&net].row.entity_type, 12 | 13 | 14 | 15 | 17 | 18) {
                    let placements: Vec<&crate::film::VariantObject> = mv.present_objects().collect();
                    let pl = fo::join_placement(&placements, mv, p);
                    self.objs.get_mut(&net).unwrap().row.placed = pl;
                }
            }
            {
                let o = &self.objs[&net];
                let by_id = self.opts.watch_ids.contains(&net);
                let by_label = o.row.label().map_or(false, |l| self.opts.watch_labels.iter().any(|w| l.eq_ignore_ascii_case(w)));
                if by_id || by_label {
                    self.watched.push(net);
                }
            }
            if self.opts.objects {
                let o = &self.objs[&net];
                let skip_proj = o.row.entity_type == 16;
                if !skip_proj && self.opts.filter.matches(&o.row, Some(&o.tr)) {
                    evs.push(object_event("object_spawn", &o.row, &o.tr, tick));
                    self.objs.get_mut(&net).unwrap().announced = true;
                }
            }
        }
        // ---- zones
        self.zones(tick, evs);
    }

    fn zones(&mut self, tick: u32, evs: &mut Vec<Value>) {
        if self.opts.watch_ids.is_empty() && self.opts.watch_labels.is_empty() {
            return;
        }
        let occ_ids: Vec<u32> = self.occupants.clone();
        for z in self.watched.clone() {
            let zo = &self.objs[&z];
            let vol: Option<Volume> = if zo.row.destroyed_tick.map_or(false, |d| tick >= d) {
                None
            } else if let Some(c) = fo::pos_at(&zo.tr, tick) {
                let (fw, up) = fo::basis_at(&zo.tr, tick);
                match zo.row.boundary() {
                    Some((shape, dims, _)) => Some(Volume { center: c, forward: fw, up, shape, dims }),
                    None => self.opts.watch_radius.map(|r| Volume { center: c, forward: fw, up, shape: 1, dims: [r, 0.0, 0.0, 0.0] }),
                }
            } else {
                None
            };
            let now: Vec<u32> = match vol {
                Some(v) => occ_ids
                    .iter()
                    .copied()
                    .filter(|id| {
                        let o = &self.objs[id];
                        fo::occ_alive(&o.row, tick) && fo::pos_at(&o.tr, tick).map_or(false, |p| v.contains(p))
                    })
                    .collect(),
                None => Vec::new(),
            };
            let prev = self.zone_inside.get(&z).cloned().unwrap_or_default();
            let mut moves: Vec<(u32, bool)> = Vec::new();
            for w in &now {
                if !prev.contains(w) {
                    moves.push((*w, true));
                }
            }
            for w in &prev {
                if !now.contains(w) {
                    moves.push((*w, false));
                }
            }
            moves.sort();
            for (w, enter) in moves {
                let o = &self.objs[&w];
                let zl = zo.row.label().map(|s| s.to_string());
                let (slot, name) = (o.row.owner_slot, o.row.owner_name.clone());
                evs.push(json!({"type": if enter { "zone_enter" } else { "zone_exit" }, "zone_id": z, "zone_label": zl,
                    "occupant_id": w, "occupant_type": o.row.type_name(), "slot": slot, "name": name}));
            }
            self.zone_inside.insert(z, now);
        }
    }
}

fn object_event(kind: &str, row: &ObjRow, tr: &EntityTrack, tick: u32) -> Value {
    let pos = fo::pos_at(tr, tick).or(row.first_pos);
    json!({"type": kind, "id_obj": row.net_id, "entity_type": row.entity_type, "object_type": row.type_name(),
        "tag": row.tag, "label": row.label(), "position": pos,
        "team": row.mp.map(|m| m.team).filter(|t| *t >= 0).or(row.placed.as_ref().filter(|p| p.team != 0xFF).map(|p| p.team as i8)),
        "shape": row.boundary().map(|b| fo::shape_name(b.0))})
}

// ---------------------------------------------------------------------------------------------
// the CLI loop
// ---------------------------------------------------------------------------------------------

/// Parse `--film-follow`'s arguments (everything after the flag) and run. Returns the exit code.
/// `out` receives the stream, `err` the status lines.
pub fn run_cli(args: &[String], names: Option<&dyn crate::film_stats::NameSource>, out: &mut dyn std::io::Write, err: &mut dyn std::io::Write) -> i32 {
    let mut path: Option<String> = None;
    let mut opts = Opts::default();
    let mut emit_csv = false;
    let mut poll_ms: u64 = 1000;
    let mut once = false;
    let mut max_secs: Option<u64> = None;
    let mut filters: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--emit" => match val().as_str() {
                "json-lines" | "jsonl" | "json" => emit_csv = false,
                "csv" => emit_csv = true,
                x => {
                    let _ = writeln!(err, "--emit json-lines|csv, got '{x}'");
                    return 2;
                }
            },
            "--poll-ms" => poll_ms = val().parse().unwrap_or(1000).max(20),
            "--since" => match crate::film_data::parse_time(&val()) {
                Ok(t) => opts.since = Some(t),
                Err(e) => {
                    let _ = writeln!(err, "--since: {e}");
                    return 2;
                }
            },
            "--watch-label" => opts.watch_labels.push(val()),
            "--watch-id" => match crate::film_data::parse_obj_id(&val()) {
                Ok(v) => opts.watch_ids.push(v),
                Err(e) => {
                    let _ = writeln!(err, "--watch-id: {e}");
                    return 2;
                }
            },
            "--watch-radius" => opts.watch_radius = val().parse().ok(),
            "--filter" => filters.push(val()),
            "--map" => {
                val();
            }
            "--objects" => opts.objects = true,
            "--all-events" => opts.all_events = true,
            "--clock" => opts.clock = true,
            "--once" => once = true,
            "--max-seconds" => max_secs = val().parse().ok(),
            x if !x.starts_with("--") && path.is_none() => path = Some(x.to_string()),
            x => {
                let _ = writeln!(err, "unknown argument '{x}'");
                return 2;
            }
        }
        i += 1;
    }
    let Some(path) = path else {
        let _ = writeln!(err, "usage: hms-app --film-follow <path.temp|film> [--emit json-lines|csv] [--poll-ms N] [--since T] [--watch-label NAME]... [--watch-id ID]... [--watch-radius R] [--map cache.map] [--filter key=value]... [--objects] [--all-events] [--clock] [--once] [--max-seconds N]");
        return 2;
    };
    match Filter::parse(&filters) {
        Ok(f) => opts.filter = f,
        Err(e) => {
            let _ = writeln!(err, "--filter: {e}");
            return 2;
        }
    }
    let mut fol = Follower::new(opts);
    let started = std::time::Instant::now();
    // #film-reader  ONE handle, kept open across polls: read-only, share READ|WRITE|DELETE, the
    // size always asked of the handle (a path stat of a file MCC is appending to can be stale on
    // NTFS), and a finalise rename that never has to wait for us.
    let mut tail: Option<crate::film_reader::Tail> = None;
    let mut last_err: Option<String> = None;
    let mut header_done = false;
    let mut parsed_once = false;
    let mut announced_final = false;
    let target = std::path::PathBuf::from(&path);
    loop {
        if tail.is_none() {
            match crate::film_reader::Tail::open(&target, &crate::film_reader::Retry::PATIENT, crate::film::same_recording, crate::film::film_search_dirs()) {
                Ok(t) => {
                    tail = Some(t);
                    last_err = None;
                }
                Err(e) => {
                    let m = e.to_string();
                    if last_err.as_deref() != Some(m.as_str()) {
                        let _ = writeln!(err, "waiting: {m} (details are logged to {})", crate::film_reader::LOG_FILE);
                        last_err = Some(m);
                    }
                }
            }
        }
        let mut processed = false;
        let mut dropped = false;
        if let Some(t) = tail.as_mut() {
            match t.poll() {
                Ok(p) => {
                    if p.new_bytes > 0 || p.restarted || !parsed_once {
                        match crate::film::parse(t.bytes()) {
                            Ok(f) => {
                                parsed_once = true;
                                processed = true;
                                let cpu0 = cpu_seconds();
                                let evs = fol.refresh(&f, names);
                                let cpu = cpu_seconds() - cpu0;
                                for e in &evs {
                                    let r = if emit_csv {
                                        if !header_done {
                                            let _ = writeln!(out, "id,tick,seconds,type,data");
                                            header_done = true;
                                        }
                                        writeln!(out, "{},{},{:.3},{},{}", e.id, e.tick, e.tick as f64 / 60.0, e.json["type"].as_str().unwrap_or(""), csv_quote(&e.json.to_string()))
                                    } else {
                                        writeln!(out, "{}", e.json)
                                    };
                                    if r.is_err() {
                                        return 0; // the consumer went away
                                    }
                                }
                                let _ = out.flush();
                                let _ = writeln!(err, "refresh: {} records ({} new), {} events, {:.1} ms wall, {:.1} ms cpu", fol.last.records_total, fol.last.records_new, evs.len(), fol.last.wall.as_secs_f64() * 1e3, cpu * 1e3);
                            }
                            Err(e) => {
                                let _ = writeln!(err, "waiting: {e}");
                            }
                        }
                    }
                    if p.just_finalised && !announced_final {
                        announced_final = true;
                        let name = t.finalised_as.as_ref().and_then(|f| f.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let line = serde_json::json!({"id": "finalised", "tick": fol.last.records_total, "seconds": 0.0, "type": "finalised", "file": name});
                        let _ = if emit_csv { writeln!(out, "finalised,{},0.000,finalised,{}", fol.last.records_total, csv_quote(&line.to_string())) } else { writeln!(out, "{line}") };
                        let _ = out.flush();
                        let _ = writeln!(err, "the recording was finalised as {name}");
                    }
                }
                Err(e) => {
                    let _ = writeln!(err, "read error ({e}); re-opening");
                    dropped = true;
                }
            }
        }
        if dropped {
            tail = None;
        }
        if once && processed {
            return 0;
        }
        if max_secs.map_or(false, |m| started.elapsed().as_secs() >= m) {
            return 0;
        }
        std::thread::sleep(std::time::Duration::from_millis(poll_ms));
    }
}

fn csv_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Process CPU seconds (user + system), read from `/proc/self/stat` where it exists.
pub fn cpu_seconds() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    let Some(rp) = s.rfind(')') else { return 0.0 };
    let f: Vec<&str> = s[rp + 2..].split_whitespace().collect();
    let tick = 100.0; // USER_HZ
    let u: f64 = f.get(11).and_then(|x| x.parse().ok()).unwrap_or(0.0);
    let k: f64 = f.get(12).and_then(|x| x.parse().ok()).unwrap_or(0.0);
    (u + k) / tick
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE0: u32 = 0xEB070194; // team 0's flag base, label ctf_flag_return
    const BASE1: u32 = 0xEB080195;

    fn opts() -> Opts {
        Opts { watch_labels: vec!["ctf_flag_return".into()], watch_radius: Some(3.0), objects: true, ..Default::default() }
    }

    fn zone_rows(evs: &[Event]) -> Vec<(u32, String, u32, u32)> {
        let mut v: Vec<_> = evs
            .iter()
            .filter(|e| e.json["type"].as_str().map_or(false, |t| t.starts_with("zone_")))
            .map(|e| (e.tick, e.json["type"].as_str().unwrap().to_string(), e.json["zone_id"].as_u64().unwrap() as u32, e.json["occupant_id"].as_u64().unwrap() as u32))
            .collect();
        v.sort();
        v
    }

    /// #film-follow  GATE (zones). The follower's ENTER/EXIT stream for the two flag bases must
    /// (a) equal, row for row, what the batch `film inside` verb computes from the finished index,
    /// (b) put the capturing player INSIDE a base on exactly the two scoreboard capture ticks, and
    /// (c) name that player the way the roster does.
    #[test]
    fn follower_zone_events_match_the_batch_verb_and_the_scoreboard() {
        let Some(f) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let mut fol = Follower::new(opts());
        let evs = fol.refresh(&f, None);
        let rows = zone_rows(&evs);
        assert_eq!(rows.iter().filter(|r| r.1 == "zone_enter").count(), 115);
        assert_eq!(rows.iter().filter(|r| r.1 == "zone_exit").count(), 115);
        assert_eq!(rows[0], (1456, "zone_enter".into(), BASE1, 3969384472));

        // (a) the batch verb, same zones, same radius, whole film
        let idx = crate::film_decode::build_index_film(&f);
        let csv = crate::film_data::verb(&f, Some(&idx), "inside", &["label=ctf_flag_return".into(), "r=3".into(), "0..t32303".into()]).unwrap().unwrap();
        let mut batch: Vec<(u32, String, u32, u32)> = csv
            .lines()
            .skip(1)
            .map(|l| {
                let c: Vec<&str> = l.split(',').collect();
                (c[0].parse().unwrap(), if c[2] == "ENTER" { "zone_enter" } else { "zone_exit" }.to_string(), c[3].parse().unwrap(), c[5].parse().unwrap())
            })
            .collect();
        batch.sort();
        assert_eq!(rows, batch, "live and batch zone transitions are the same stream");

        // (b) slot 4 is inside a base while the scoreboard says it captured
        let inside_at = |slot: u64, tick: u32| -> Option<u32> {
            let mut open: HashMap<(u32, u32), bool> = HashMap::new();
            for e in evs.iter().filter(|e| e.tick <= tick && e.json["slot"] == slot) {
                let t = e.json["type"].as_str().unwrap();
                if t.starts_with("zone_") {
                    open.insert((e.json["zone_id"].as_u64().unwrap() as u32, e.json["occupant_id"].as_u64().unwrap() as u32), t == "zone_enter");
                }
            }
            open.into_iter().find(|(_, v)| *v).map(|((z, _), _)| z)
        };
        assert_eq!(inside_at(4, 23905), Some(BASE0), "the first capture happens inside base 0");
        assert_eq!(inside_at(4, 29170), Some(BASE0), "the second capture happens inside base 0");
        assert_eq!(inside_at(4, 25000), None, "and between them slot 4 is elsewhere");
        let scores: Vec<u32> = evs.iter().filter(|e| e.json["type"] == "score" && e.json["is_team"] == true && e.json["team"] == 0).map(|e| e.tick).collect();
        assert!(scores.contains(&23905) && scores.contains(&29170), "{scores:?}");

        // (c) the name is the roster's
        let e = evs.iter().find(|e| e.tick == 23702 && e.json["type"] == "zone_enter" && e.json["slot"] == 4).expect("slot 4 enters at 23702");
        let seat = idx.roster.seat_holding(4, 23702).expect("a seat holds slot 4");
        assert_eq!(e.json["name"].as_str().unwrap(), seat.name);
    }

    /// #film-follow  GATE (growing file). A film truncated at several record boundaries is fed to
    /// ONE follower in turn, as a game rewriting a `.temp` would: the events it emits across all
    /// refreshes must equal the one-shot events of the whole file (no duplicates, nothing lost),
    /// and the one-shot events of every prefix must be a prefix of the whole's.
    #[test]
    fn a_growing_file_yields_a_consistent_prefix_stream() {
        let Some(full) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let n = full.data.records.len();
        let cut_bytes = |k: usize| -> Vec<u8> {
            let r = full.data.records[k - 1];
            full.bytes[..r.body_at + r.len].to_vec()
        };
        let cuts = [PARAM_RECORDS - 1, 300, 4_001, 17_777, n];
        let whole = Follower::new(opts()).refresh(&full, None);
        assert!(whole.len() > 4000);
        let mut inc = Follower::new(opts());
        let mut acc: Vec<Event> = Vec::new();
        let mut new_records = Vec::new();
        for &k in &cuts {
            let f = crate::film::parse(&cut_bytes(k)).expect("a truncated film parses");
            assert_eq!(f.data.records.len(), k);
            acc.extend(inc.refresh(&f, None));
            new_records.push(inc.last.records_new);
            let one_shot = Follower::new(opts()).refresh(&f, None);
            assert!(whole.len() >= one_shot.len());
            for (a, b) in whole.iter().zip(one_shot.iter()) {
                assert_eq!(a.json, b.json, "prefix {k}");
            }
            assert_eq!(acc.len(), one_shot.len(), "cut {k}: incremental total equals the prefix's one-shot");
        }
        assert_eq!(new_records, vec![0, 300, 3701, 13776, n - 17_777], "each refresh decodes only the NEW records (nothing before 256 records exist)");
        assert_eq!(acc.len(), whole.len());
        for (a, b) in acc.iter().zip(whole.iter()) {
            assert_eq!(a.json, b.json);
        }
        let mut ids: Vec<&str> = acc.iter().map(|e| e.id.as_str()).collect();
        let total = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), total, "no duplicate event ids");
        // `--since` resumes: nothing at or before the tick, and the rest identical
        let mut late = Follower::new(Opts { since: Some(20000), ..opts() });
        let tail = late.refresh(&full, None);
        let expect: Vec<&Event> = whole.iter().filter(|e| e.tick > 20000).collect();
        assert_eq!(tail.len(), expect.len());
        assert!(tail.iter().zip(expect.iter()).all(|(a, b)| a.json == b.json));
    }

    /// #film-reader  GATE. A real film appended to a file in chunks that end MID-RECORD, read through
    /// the shared-handle tail, parses to a monotonically growing record count that never exceeds
    /// what has been written, the follower's events over all the refreshes equal the one-shot
    /// events, and `same_recording` tells a truncated copy from a different match.
    #[test]
    fn a_real_film_written_in_odd_chunks_is_followed_to_the_same_events() {
        use std::io::Write;
        let Some(full) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let dir = std::env::temp_dir().join(format!("hms-follow-real-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("live.temp");
        std::fs::write(&p, b"").unwrap();
        let bytes = full.bytes.clone();
        let mut tail = crate::film_reader::Tail::open(&p, &crate::film_reader::Retry::NONE, crate::film::same_recording, vec![]).unwrap();
        let mut fol = Follower::new(opts());
        let mut acc: Vec<Event> = Vec::new();
        let (mut at, mut last_records, mut steps) = (0usize, 0usize, 0u32);
        let mut w = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        while at < bytes.len() {
            // chunk ends are deliberately not on record boundaries
            let n = (1 + (steps as usize * 7919 + 1237) % 400_000).min(bytes.len() - at);
            w.write_all(&bytes[at..at + n]).unwrap();
            w.flush().unwrap();
            at += n;
            steps += 1;
            let r = tail.poll().unwrap();
            assert_eq!(r.new_bytes, n);
            assert_eq!(tail.bytes(), &bytes[..at]);
            let Ok(f) = crate::film::parse(tail.bytes()) else { continue };
            let recs = f.data.records.len();
            assert!(recs >= last_records && recs <= full.data.records.len(), "records only grow and never exceed the whole ({recs})");
            last_records = recs;
            acc.extend(fol.refresh(&f, None));
        }
        assert!(steps > 10);
        assert_eq!(last_records, full.data.records.len());
        let whole = Follower::new(opts()).refresh(&full, None);
        assert_eq!(acc.len(), whole.len());
        assert!(acc.iter().zip(whole.iter()).all(|(a, b)| a.json == b.json));
        // same_recording: a truncated copy is the same recording, a different match is not
        let cut = &bytes[..bytes.len() / 2];
        assert!(crate::film::same_recording(&bytes[..], cut));
        if let Some(other) = crate::film::spec_corpus().into_iter().find(|q| !q.to_string_lossy().contains("53C14E9D")).and_then(|q| std::fs::read(q).ok()) {
            assert!(!crate::film::same_recording(&bytes, &other));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// #film-follow  GATE (headless). The follower must not be able to reach the GPU, the window
    /// toolkit or audio: its non-comment source names none of them.
    #[test]
    fn the_follower_source_names_no_gpu_window_or_audio_crate() {
        let src = include_str!("film_follow.rs");
        let banned = [["wg", "pu"].concat(), ["eg", "ui"].concat(), ["ef", "rame"].concat(), ["ro", "dio"].concat(), ["cp", "al"].concat(), ["win", "it"].concat(), ["fm", "od"].concat()];
        for (n, line) in src.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            for b in &banned {
                assert!(!line.contains(b.as_str()), "film_follow.rs:{} names `{b}`", n + 1);
            }
        }
    }
}

#[cfg(test)]
mod cost {
    use super::*;
    #[test]
    #[ignore = "probe: labels in the corpus and refresh cost"]
    fn corpus_labels_and_cost() {
        let mut biggest: Option<(usize, std::path::PathBuf)> = None;
        for p in crate::film::spec_corpus() {
            let Ok(f) = crate::film::open(&p) else { continue };
            let n = f.data.records.len();
            let labels = &f.header.map_variant.labels;
            if !labels.is_empty() {
                eprintln!("LAB {} records={} labels={:?}", p.file_name().unwrap().to_string_lossy(), n, labels);
            }
            if biggest.as_ref().map_or(true, |b| n > b.0) { biggest = Some((n, p.clone())); }
        }
        let (n, p) = biggest.unwrap();
        eprintln!("COST biggest {} records, {} MB", n, std::fs::metadata(&p).unwrap().len() / 1_000_000);
        let t = std::time::Instant::now(); let c = cpu_seconds();
        let f = crate::film::open(&p).unwrap();
        eprintln!("COST open+frame: {:.0} ms wall {:.0} ms cpu", t.elapsed().as_secs_f64()*1e3, (cpu_seconds()-c)*1e3);
        // one-shot, then pretend the file grew by the last 1500 records (25 s of game)
        let mut full_bytes_cut = |k: usize| { let r = f.data.records[k-1]; f.bytes[..r.body_at + r.len].to_vec() };
        let near = crate::film::parse(&full_bytes_cut(n - 1500)).unwrap();
        let mut fol = Follower::new(Opts::default());
        let t = std::time::Instant::now(); let c = cpu_seconds();
        let a = fol.refresh(&near, None);
        eprintln!("COST one-shot {} records: {:.0} ms wall {:.0} ms cpu, {} events", near.data.records.len(), t.elapsed().as_secs_f64()*1e3, (cpu_seconds()-c)*1e3, a.len());
        let t = std::time::Instant::now(); let c = cpu_seconds();
        let b = fol.refresh(&f, None);
        eprintln!("COST incremental +{} records: {:.1} ms wall {:.1} ms cpu (follower only), {} events", fol.last.records_new, fol.last.wall.as_secs_f64()*1e3, (cpu_seconds()-c)*1e3, b.len());
        let t2 = std::time::Instant::now(); let _ = crate::film::parse(&f.bytes).unwrap();
        eprintln!("COST re-frame whole file again: {:.0} ms (incl. copy {} MB)", t2.elapsed().as_secs_f64()*1e3, f.bytes.len()/1_000_000);
        let _ = t;
    }
}

#[cfg(test)]
mod sample {
    #[test]
    #[ignore = "writes a sample JSON-lines stream to $HMS_SAMPLE_OUT"]
    fn write_sample_stream() {
        let Some(out) = std::env::var_os("HMS_SAMPLE_OUT") else { return };
        let Some(f) = crate::film_megalo::tests_castle_wars() else { return };
        let path = std::env::var("HMS_FILM").unwrap();
        let _ = f;
        let mut buf: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let args: Vec<String> = [path.as_str(), "--once", "--watch-label", "ctf_flag_return", "--watch-radius", "3"].iter().map(|s| s.to_string()).collect();
        let code = super::run_cli(&args, None, &mut buf, &mut err);
        assert_eq!(code, 0);
        std::fs::write(out, buf).unwrap();
        eprintln!("{}", String::from_utf8_lossy(&err));
    }
}
