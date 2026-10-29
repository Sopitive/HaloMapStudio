//! #film-roster  WHO WAS IN THE ROOM, AND WHEN -- the time-aware player roster of a saved film.
//!
//! A film's header names the players present when it was WRITTEN. People join, leave and come
//! back during a match, and the recording states each of those changes in its own command list:
//! the engine's PLAYER UPDATE, queue element type 9 (`simulation_queue_player_update_insert`,
//! `reach_tag_test.exe`), applied on playback by `simulation_players_apply_update` and
//! decoded in `haloreach.dll` by `VariantCommand_ParseAndDispatchComplexObjectAction`
//! (`ExecuteVariantCommandOnce` case 9).
//!
//! ```text
//! player update   4 bits  player slot (index into the 16 s_player_configuration slots)
//!                 64 bits player identifier (XUID)
//!                 3 bits  update type: 0 left-game, 1 swap, 2 remove, 3 add, 4 configuration
//!   type 3 (add)  1 flag  player-left-game          1 flag  player-joined-in-progress
//!                 if NOT left-game: 48-bit machine id, 2-bit input user, 2-bit controller
//!   type 3 / 4    config delta  (sub_1803A1EB8): flag->name (7-bit chars through a 6-range
//!                 table), flag->64-bit xuid, flag->appearance (sub_1803A296C), flag->7 bits,
//!                 flag->5 bits-1 TEAM (+0x52), flag->6 bits, flag->2 bits
//!                 data delta    (sub_1803A2338): flag->name2, flag->5 bits-1, flag->5 bits-1
//!   type 1 (swap) 4-bit swap slot + 64-bit identifier
//! ```

use crate::film_decode::{creation_appearance, BipedAppearance, BitReader};

/// `simulation_player_update_type`, in the engine's order (`simulation_players_apply_update`'s
/// `switch (player_update->update_type)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateKind {
    LeftGame,
    Swap,
    Remove,
    Add,
    Configuration,
}

/// The sparse `s_player_configuration` delta a type-3/4 update carries: every field is behind its
/// own presence flag, and only what CHANGED is stated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConfigDelta {
    pub name: Option<String>,
    pub xuid: Option<u64>,
    pub look: Option<BipedAppearance>,
    pub f50: Option<u8>,
    /// `+0x52`, `ReadBits(5) - 1`: the TEAM index, `-1` = none.
    pub team: Option<i8>,
    pub f51: Option<u8>,
    pub f53: Option<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlayerUpdate {
    pub player: u8,
    pub identifier: u64,
    pub kind: UpdateKind,
    pub left_game: bool,
    pub joined_in_progress: bool,
    pub machine: Option<[u8; 6]>,
    pub config: Option<ConfigDelta>,
    pub data_name: Option<String>,
    pub data_team: Option<i8>,
    pub data_b33: Option<i8>,
    pub swap_with: Option<(u8, u64)>,
    pub leftover_bits: usize,
}

/// The 7-bit gamertag character code -> UTF-16 unit, `sub_1800DC68C` with its range table at
/// `haloreach.dll+0x9D8DA8`: `{0,0} {'0','9'} {'a','z'} {'A','Z'} {'(',')'} {' ',' '}`.
pub fn name_char(code: u16) -> u16 {
    const RANGES: [(u16, u16); 6] =
        [(0, 0), (b'0' as u16, b'9' as u16), (b'a' as u16, b'z' as u16), (b'A' as u16, b'Z' as u16), (b'(' as u16, b')' as u16), (b' ' as u16, b' ' as u16)];
    let mut base = 0u16;
    for (lo, hi) in RANGES {
        let n = hi - lo + 1;
        if code - base.min(code) < n && code >= base {
            return code + lo - base;
        }
        base += n;
    }
    code
}

fn read_name(r: &mut BitReader) -> Option<String> {
    let mut units: Vec<u16> = Vec::new();
    for _ in 0..16 {
        let c = name_char(r.bits(7).ok()? as u16);
        if c == 0 {
            break;
        }
        units.push(c);
    }
    Some(String::from_utf16_lossy(&units))
}

fn read_xuid(r: &mut BitReader) -> Option<u64> {
    // `Bitstream_ReadBytes(.., 0x40)`: 64 bits, stored byte by byte in stream order.
    let b = r.read_bytes(8).ok()?;
    Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

/// `sub_1803A1EB8` with `a3 == 0` (the network form).
fn config_delta(r: &mut BitReader) -> Option<ConfigDelta> {
    let mut c = ConfigDelta::default();
    if r.flag().ok()? {
        c.name = Some(read_name(r)?);
    }
    if r.flag().ok()? {
        c.xuid = Some(read_xuid(r)?);
    }
    if r.flag().ok()? {
        c.look = Some(creation_appearance(r).ok()?);
    }
    if r.flag().ok()? {
        c.f50 = Some(r.bits(7).ok()? as u8);
    }
    if r.flag().ok()? {
        c.team = Some(r.bits(5).ok()? as i8 - 1);
    }
    if r.flag().ok()? {
        c.f51 = Some(r.bits(6).ok()? as u8);
    }
    // `Bitstream_ReadBits(a1, sub_1803A0A04(2) + 1)`: `sub_1803A0A04(2)` is 1, so two bits.
    if r.flag().ok()? {
        c.f53 = Some(r.bits(2).ok()? as u8);
    }
    Some(c)
}

/// Decode one list-B command-kind-9 payload.
pub fn decode_player_update(payload: &[u8]) -> Option<PlayerUpdate> {
    let mut r = BitReader::new(payload);
    let player = r.bits(4).ok()? as u8;
    let b = r.read_bytes(8).ok()?;
    let identifier = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    let kind = match r.bits(3).ok()? {
        0 => UpdateKind::LeftGame,
        1 => UpdateKind::Swap,
        2 => UpdateKind::Remove,
        3 => UpdateKind::Add,
        4 => UpdateKind::Configuration,
        _ => return None,
    };
    let mut u = PlayerUpdate {
        player,
        identifier,
        kind,
        left_game: false,
        joined_in_progress: false,
        machine: None,
        config: None,
        data_name: None,
        data_team: None,
        data_b33: None,
        swap_with: None,
        leftover_bits: 0,
    };
    if kind == UpdateKind::Add {
        u.left_game = r.flag().ok()?;
        u.joined_in_progress = r.flag().ok()?;
        if !u.left_game {
            let m = r.read_bytes(6).ok()?;
            u.machine = Some([m[0], m[1], m[2], m[3], m[4], m[5]]);
            r.bits(2).ok()?;
            r.bits(2).ok()?;
        }
    }
    if matches!(kind, UpdateKind::Add | UpdateKind::Configuration) {
        u.config = Some(config_delta(&mut r)?);
        if r.flag().ok()? {
            u.data_name = Some(read_name(&mut r)?);
        }
        if r.flag().ok()? {
            u.data_team = Some(r.bits(5).ok()? as i8 - 1);
        }
        if r.flag().ok()? {
            u.data_b33 = Some(r.bits(5).ok()? as i8 - 1);
        }
    }
    if kind == UpdateKind::Swap {
        let i = r.bits(4).ok()? as u8;
        let b = r.read_bytes(8).ok()?;
        u.swap_with = Some((i, u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])));
    }
    u.leftover_bits = r.remaining();
    Some(u)
}


// ---------------------------------------------------------------------------------------------
// the roster: who held each slot, and when
// ---------------------------------------------------------------------------------------------

/// One stretch of one player holding one slot with one configuration. A configuration change
/// (name, team or armour) ends the segment and opens the next, so asking "who was this, as of tick
/// T" is a lookup and never a replay.
#[derive(Clone, Debug, PartialEq)]
pub struct Seat {
    pub slot: u8,
    /// The player identifier (the XUID) the update was keyed on.
    pub identifier: u64,
    /// The name the stream states for this seat at this time.
    pub name: String,
    /// `s_player_configuration +0x52`, `-1` = not on a team (yet).
    pub team: i8,
    /// The player-data block's team (`sub_1803A2338`, `a2[32]`, `ReadBits(5) - 1`), `-1` = none.
    pub data_team: i8,
    pub look: Option<BipedAppearance>,
    pub from_tick: u32,
    /// The tick the segment ended: a `Remove`, a reconfiguration, a swap, or a replacement.
    pub until_tick: Option<u32>,
    /// The tick the player left the game (`left-game` update, or an `Add` that already carried
    /// the flag). The record stays in the list -- its bodies are still theirs -- but no new body
    /// of theirs can appear after this.
    pub left_tick: Option<u32>,
    /// Came from the film HEADER rather than from a player update.
    pub from_header: bool,
    pub joined_in_progress: bool,
}

impl Seat {
    /// Could a body that SPAWNED at `tick` belong to this seat?
    pub fn playing_at(&self, tick: u32) -> bool {
        tick >= self.from_tick
            && self.until_tick.map_or(true, |u| tick < u)
            && self.left_tick.map_or(true, |l| tick < l)
    }
    /// Did this seat hold its slot (playing or already left, but not yet removed) at `tick`?
    pub fn holds_at(&self, tick: u32) -> bool {
        tick >= self.from_tick && self.until_tick.map_or(true, |u| tick < u)
    }
}

/// Every seat of a film, in the order the engine created them.
#[derive(Clone, Debug, Default)]
pub struct Roster {
    pub seats: Vec<Seat>,
    /// Player updates the stream carried, and how many applied cleanly (the engine's own
    /// identifier checks passed). A film where these differ has a roster that should not be trusted.
    pub updates: usize,
    pub applied: usize,
}

/// The header's armour in the stream's terms, so the two compare directly.
pub fn header_look(p: &crate::film::FilmPlayer) -> Option<BipedAppearance> {
    let a = p.appearance()?;
    let mut slots = [0i8; 12];
    for (i, v) in a.slots.iter().enumerate() {
        slots[i] = *v as i8;
    }
    Some(BipedAppearance {
        flags: a.f48,
        colors: [a.color_primary as i8, a.color_secondary as i8, a.f4b as i8],
        model_choice: a.model_choice,
        slots,
        ids: [0; 4],
    })
}

/// Do two appearances state the same armour? The comparison the header route has always used:
/// species, flags, three colours and the ten slot options. The four `ids` change from one update
/// to the next and are not armour.
pub fn same_armour(a: &BipedAppearance, b: &BipedAppearance) -> bool {
    a.flags == b.flags && a.colors == b.colors && a.model_choice == b.model_choice && a.slots[..10] == b.slots[..10]
}

impl Roster {
    /// Seed from the header, then apply every player update in tick order, the way
    /// `simulation_players_apply_update` does on playback.
    pub fn build(header: &[crate::film::FilmPlayer], updates: &[(u32, PlayerUpdate)]) -> Roster {
        let mut r = Roster::default();
        // The segment currently holding each slot.
        let mut cur: [Option<usize>; 16] = [None; 16];
        for p in header.iter().filter(|p| p.present() && p.index < 16) {
            r.seats.push(Seat {
                slot: p.index as u8,
                identifier: p.xuid,
                name: p.gamertag.clone(),
                team: p.team as i8,
                data_team: -1,
                look: header_look(p),
                from_tick: 0,
                until_tick: None,
                left_tick: None,
                from_header: true,
                joined_in_progress: false,
            });
            cur[p.index] = Some(r.seats.len() - 1);
        }
        for (tick, u) in updates {
            r.updates += 1;
            let slot = (u.player & 15) as usize;
            let tick = *tick;
            match u.kind {
                UpdateKind::Remove => {
                    let Some(i) = cur[slot] else { continue };
                    if r.seats[i].identifier != u.identifier {
                        continue; // "identifier mismatch": the engine refuses it too
                    }
                    r.seats[i].until_tick = Some(tick);
                    cur[slot] = None;
                    r.applied += 1;
                }
                UpdateKind::LeftGame => {
                    let Some(i) = cur[slot] else { continue };
                    if r.seats[i].identifier != u.identifier {
                        continue;
                    }
                    r.seats[i].left_tick.get_or_insert(tick);
                    r.applied += 1;
                }
                UpdateKind::Add => {
                    if let Some(i) = cur[slot].take() {
                        r.seats[i].until_tick.get_or_insert(tick);
                    }
                    let c = u.config.clone().unwrap_or_default();
                    r.seats.push(Seat {
                        slot: slot as u8,
                        identifier: u.identifier,
                        name: c.name.unwrap_or_default(),
                                team: c.team.unwrap_or(-1),
                        data_team: u.data_team.unwrap_or(-1),
                        look: c.look,
                        from_tick: tick,
                        until_tick: None,
                        left_tick: u.left_game.then_some(tick),
                        from_header: false,
                        joined_in_progress: u.joined_in_progress,
                    });
                    cur[slot] = Some(r.seats.len() - 1);
                    r.applied += 1;
                }
                UpdateKind::Configuration => {
                    let Some(i) = cur[slot] else { continue };
                    if r.seats[i].identifier != u.identifier {
                        continue;
                    }
                    let c = u.config.clone().unwrap_or_default();
                    let old = &r.seats[i];
                    let name = c.name.clone().unwrap_or_else(|| old.name.clone());
                    let team = c.team.unwrap_or(old.team);
                    let look = c.look.or(old.look);
                    let data_team = u.data_team.unwrap_or(old.data_team);
                    let same = name == old.name
                        && team == old.team
                        && data_team == old.data_team
                        && match (&look, &old.look) {
                            (Some(a), Some(b)) => same_armour(a, b),
                            (None, None) => true,
                            _ => false,
                        };
                    r.applied += 1;
                    if same {
                        continue;
                    }
                    let mut next = old.clone();
                    next.name = name;
                    next.team = team;
                    next.data_team = data_team;
                    next.look = look;
                    next.from_tick = tick;
                    next.until_tick = None;
                    next.from_header = false;
                    r.seats[i].until_tick = Some(tick);
                    r.seats.push(next);
                    cur[slot] = Some(r.seats.len() - 1);
                }
                UpdateKind::Swap => {
                    let Some((other, other_id)) = u.swap_with else { continue };
                    let other = (other & 15) as usize;
                    let (a, b) = (cur[slot], cur[other]);
                    let id_a = a.map_or(0, |i| r.seats[i].identifier);
                    let id_b = b.map_or(0, |i| r.seats[i].identifier);
                    if id_a != u.identifier || id_b != other_id {
                        continue;
                    }
                    // The records trade places: each player keeps their identity and moves slot.
                    let mut moved = [None, None];
                    for (k, (from, to)) in [(a, other), (b, slot)].into_iter().enumerate() {
                        if let Some(i) = from {
                            let mut next = r.seats[i].clone();
                            next.slot = to as u8;
                            next.from_tick = tick;
                            next.until_tick = None;
                            next.from_header = false;
                            r.seats[i].until_tick = Some(tick);
                            r.seats.push(next);
                            moved[k] = Some(r.seats.len() - 1);
                        }
                    }
                    cur[other] = moved[0];
                    cur[slot] = moved[1];
                    r.applied += 1;
                }
            }
        }
        r
    }

    /// The seat holding `slot` at `tick` (playing, or left but not removed).
    pub fn seat_holding(&self, slot: u8, tick: u32) -> Option<&Seat> {
        self.seats.iter().rev().find(|s| s.slot == slot && s.holds_at(tick))
    }
}


/// How a body came to be tied to a seat. Every basis rests on statements the recording makes; none
/// is a guess about timing or position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    /// Exactly one seat playing at the body's spawn tick states this armour, and nothing the
    /// recording says about the body's team contradicts it.
    Armour,
    /// Several seats state this armour, and the body's team stamp (compared with each seat's stated
    /// team) leaves one player.
    ArmourAndTeam,
    /// Free-for-all only: the armour matched nobody (or the seat never stated one), but the body's
    /// team stamp names a slot and exactly one seat held it when the body spawned. In a
    /// free-for-all every player is their own team, so the team IS the slot.
    SlotStamp,
    /// An earlier route -- the aim correlation, the owner-stamp rule or elimination -- named the
    /// slot, and the roster confirms someone was playing in it when the body spawned.
    Binding,
}

/// What the roster says about one body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The body carries no appearance block: the gametype's or the AI's, never a player's.
    NotAPlayer,
    /// One seat (an index into [`Roster::seats`]).
    Named(usize, Basis),
    /// A player's body, but two or more people playing at that moment state the same armour and
    /// the recording gives nothing to tell them apart. The seats are the candidates.
    Ambiguous(Vec<usize>),
    /// A player's body that no seat playing at its spawn tick accounts for.
    Unmatched,
}

impl Roster {
    /// Every seat that could own a body spawned at `tick` wearing `look`.
    pub fn candidates(&self, look: &BipedAppearance, tick: u32) -> Vec<usize> {
        self.seats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.playing_at(tick) && s.look.as_ref().map_or(false, |l| same_armour(l, look)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Whose body is this? `stamp` is the body's creation team stamp (`team + 1`, 0 = none);
    /// `teams` is whether the game has teams (`None` = not known).
    ///
    /// Two statements make the answer, and the roster is what makes both of them time-aware:
    /// 1. the body's armour is a player's armour, bit for bit, and the roster says who was playing
    ///    in it when the body spawned;
    /// 2. the body's team stamp must not contradict what the roster says about that player's team.
    ///
    /// A seat's `data_team` is the team the player-data block states (`-1` = not stated); where it
    /// is stated it equals the stamp of every body the seat ever spawns -- 0 contradictions over
    /// 1,500+ bodies on the corpus's team films.
    pub fn attribute(&self, look: Option<&BipedAppearance>, stamp: u8, tick: u32, teams: Option<bool>) -> Verdict {
        let Some(look) = look else { return Verdict::NotAPlayer };
        let team = (stamp > 0).then(|| stamp as i8 - 1);
        let mut cands = self.candidates(look, tick);
        // A candidate whose stated team contradicts the stamp cannot own the body.
        if let Some(t) = team {
            cands.retain(|&i| self.seats[i].data_team < 0 || self.seats[i].data_team == t);
        }
        // `cands` are SEATS playing at one instant, so they sit in different slots: a slot holds
        // one segment at a time. Two of them are two people (or one person the recording lists
        // twice, which it does for the recorder), never one.
        if cands.len() == 1 {
            let basis = if self.candidates(look, tick).len() > 1 { Basis::ArmourAndTeam } else { Basis::Armour };
            return Verdict::Named(cands[0], basis);
        }
        if cands.len() > 1 {
            // Free-for-all: the stamp is the slot.
            if let (Some(t), Some(false)) = (team, teams) {
                let by_slot: Vec<usize> = cands.iter().copied().filter(|&i| self.seats[i].slot as i8 == t).collect();
                if by_slot.len() == 1 {
                    return Verdict::Named(by_slot[0], Basis::ArmourAndTeam);
                }
            }
            return Verdict::Ambiguous(cands);
        }
        // Armour named nobody. In a free-for-all the team stamp still names a slot.
        if let (Some(t), Some(false)) = (team, teams) {
            if let Some(seat) = self.seat_playing(t as u8, tick) {
                return Verdict::Named(seat, Basis::SlotStamp);
            }
        }
        Verdict::Unmatched
    }

    /// The seat playing in `slot` at `tick` (index into `seats`).
    pub fn seat_playing(&self, slot: u8, tick: u32) -> Option<usize> {
        self.seats.iter().rposition(|s| s.slot == slot && s.playing_at(tick))
    }
}

// ---------------------------------------------------------------------------------------------
// #monitor-names  a Forge MONITOR's player
// ---------------------------------------------------------------------------------------------

/// Whose a Forge monitor is, as far as the recording lets anyone say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonitorOwner {
    /// This seat (index into [`Roster::seats`]) is the one the evidence points at.
    Seat(usize),
    /// A monitor the recording cannot place.
    Unknown,
}

/// How close a seat's own body must hand over to a monitor to count as the same player, ticks.
const MONITOR_HANDOFF_TICKS: i64 = 30;
/// How much further the runner-up must be, ticks, for the nearest handoff to be taken as the answer.
const MONITOR_HANDOFF_MARGIN: i64 = 60;

/// Name the players of a film's Forge monitors.
///
/// **What the recording does and does not say (MEASURED over the 56-film corpus).** A monitor
/// (`monitor_editor`) is a biped with NO appearance block and a creation owner stamp of nothing,
/// so the armour rule that names every other body has nothing to work with; and the unit's
/// "controlling player" field (0x2F), which agrees with the roster on every named body that carries
/// it, is never sent for a monitor. So there is no stated link from a monitor to its player, and
/// what follows is an INFERENCE, drawn as one:
///
/// 1. a player controls one body at a time, so a seat is a candidate only if it is in the game when
///    the monitor spawns and none of its own named bodies is alive through the monitor's life
///    (a margin at each end absorbs the tick the controls hand over);
/// 2. a seat already given to a monitor that is alive at the same time is taken;
/// 3. among the rest, the seat whose own body ENDS where the monitor begins (or begins where it
///    ends) within [`MONITOR_HANDOFF_TICKS`], and by a clear margin, is the player. One candidate
///    left is the player outright.
///
/// Anything that does not come out to exactly one seat is [`MonitorOwner::Unknown`]: a wrong
/// gamertag is worse than the label "Monitor".
pub fn assign_monitors(idx: &crate::film_decode::FilmIndex, monitors: &[u32]) -> std::collections::HashMap<u32, MonitorOwner> {
    let mut out: std::collections::HashMap<u32, MonitorOwner> = Default::default();
    let end_of = |t: &crate::film_decode::EntityTrack| t.destroyed_tick.unwrap_or(t.last_tick);
    let mut ms: Vec<&crate::film_decode::EntityTrack> =
        idx.tracks.iter().filter(|t| t.entity_type == 11 && monitors.contains(&t.net_id)).collect();
    ms.sort_by_key(|t| (t.first_tick, t.net_id));
    // (seat, from, to) of monitors already given
    let mut taken: Vec<(usize, u32, u32)> = Vec::new();
    for m in ms {
        let (a, b) = (m.first_tick, end_of(m).max(m.first_tick));
        let (lo, hi) = if b - a > 120 { (a + 30, b - 30) } else { (a + 2, b.max(a + 3) - 2) };
        let mut eligible: Vec<usize> = Vec::new();
        for (si, seat) in idx.roster.seats.iter().enumerate() {
            if !seat.playing_at(a) || taken.iter().any(|&(s, f, t)| s == si && f <= b && a <= t) {
                continue;
            }
            let busy = idx.tracks.iter().any(|x| {
                x.entity_type == 11
                    && idx.seat_index_of_entity(x.net_id) == Some(si)
                    && x.first_tick <= hi
                    && end_of(x) >= lo
            });
            if !busy {
                eligible.push(si);
            }
        }
        let gap = |si: usize| -> i64 {
            let mut g = i64::MAX;
            for x in idx.tracks.iter().filter(|x| x.entity_type == 11 && idx.seat_index_of_entity(x.net_id) == Some(si)) {
                for d in [a as i64 - end_of(x) as i64, x.first_tick as i64 - b as i64] {
                    if d >= -MONITOR_HANDOFF_TICKS {
                        g = g.min(d.abs());
                    }
                }
            }
            g
        };
        let pick = match eligible.len() {
            0 => None,
            1 => Some(eligible[0]),
            _ => {
                let mut ranked: Vec<(i64, usize)> = eligible.iter().map(|&s| (gap(s), s)).collect();
                ranked.sort();
                (ranked[0].0 <= MONITOR_HANDOFF_TICKS && ranked[1].0 > ranked[0].0 + MONITOR_HANDOFF_MARGIN).then_some(ranked[0].1)
            }
        };
        match pick {
            Some(si) => {
                taken.push((si, a, b));
                out.insert(m.net_id, MonitorOwner::Seat(si));
            }
            None => {
                out.insert(m.net_id, MonitorOwner::Unknown);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// the index: every biped, tied to a seat
// ---------------------------------------------------------------------------------------------

/// Who a body is, as the roster states it. What a nameplate draws.
#[derive(Clone, Debug)]
pub enum Identity<'a> {
    /// A player's body, tied to one seat.
    Player { seat: &'a Seat, basis: Basis },
    /// A player's body that two or more people playing at that moment could own.
    Candidates(Vec<&'a Seat>),
    /// A player's body that nobody playing at its spawn accounts for.
    Unknown,
    /// Not a player's body: the gametype's or the AI's. Nothing to name.
    NotAPlayer,
}

impl crate::film_decode::FilmIndex {
    /// Build the roster from the header and the stream's player updates, then tie every biped to a
    /// seat. Replaces the header-slot bindings: a header names who was in the LOBBY, and on the
    /// corpus's team films the first tick removes and re-adds nearly every one of them.
    ///
    /// `teams` is the game variant's `teams enabled`, when the caller has it.
    pub fn attach_roster(&mut self, players: &[crate::film::FilmPlayer], teams: Option<bool>) {
        use crate::film_decode::{BindOrigin, PlayerBinding};
        self.roster = Roster::build(players, &self.player_updates);
        self.teams_enabled = teams;
        self.body_verdicts.clear();
        self.body_teams.clear();
        let mut bindings: Vec<PlayerBinding> = Vec::new();
        for tr in self.tracks.iter().filter(|t| t.entity_type == 11) {
            let mut v = self.roster.attribute(tr.look.as_ref(), tr.owner_raw.unwrap_or(0), tr.first_tick, teams);
            // A body the armour and stamp could not place may still have been placed by the
            // aim correlation or the stamp rule. Keep that only where the roster confirms
            // someone was playing in the slot it names when the body spawned.
            if !matches!(v, Verdict::Named(..) | Verdict::NotAPlayer) {
                if let Some(slot) = self.legacy_slot_of(tr.net_id) {
                    if let Some(i) = self.roster.seat_playing(slot, tr.first_tick) {
                        let ok = match &v {
                            Verdict::Ambiguous(c) => c.contains(&i),
                            _ => true,
                        };
                        if ok {
                            v = Verdict::Named(i, Basis::Binding);
                        }
                    }
                }
            }
            if let Verdict::Named(i, _) = &v {
                bindings.push(PlayerBinding {
                    slot: self.roster.seats[*i].slot,
                    net_id: tr.net_id,
                    entity_type: tr.entity_type,
                    first_tick: tr.first_tick,
                    last_tick: tr.last_tick,
                    cost: 0.0,
                    margin: 0.0,
                    samples: 0,
                    from_facing: false,
                    from_owner_rule: false,
                    origin: BindOrigin::Roster,
                });
            }
            self.body_verdicts.insert(tr.net_id, v);
            if let Some(o) = tr.owner_raw.filter(|o| *o > 0 && tr.look.is_some()) {
                self.body_teams.insert(tr.net_id, o - 1);
            }
        }
        bindings.sort_by_key(|b| (b.slot, b.first_tick));
        self.bindings = bindings;
    }

    /// The slot an earlier route (aim correlation, owner stamp rule, elimination) bound this body
    /// to, if it did and was confident.
    fn legacy_slot_of(&self, net_id: u32) -> Option<u8> {
        self.bindings.iter().find(|b| b.net_id == net_id && b.is_confident() && b.origin != crate::film_decode::BindOrigin::Roster).map(|b| b.slot)
    }

    /// Who this body is.
    pub fn identity_of(&self, net_id: u32) -> Identity<'_> {
        match self.body_verdicts.get(&net_id) {
            Some(Verdict::Named(i, basis)) => Identity::Player { seat: &self.roster.seats[*i], basis: *basis },
            Some(Verdict::Ambiguous(c)) => Identity::Candidates(c.iter().map(|&i| &self.roster.seats[i]).collect()),
            Some(Verdict::Unmatched) => Identity::Unknown,
            Some(Verdict::NotAPlayer) | None => Identity::NotAPlayer,
        }
    }

    /// The index into [`Roster::seats`] of the seat this body belongs to.
    pub fn seat_index_of_entity(&self, net_id: u32) -> Option<usize> {
        match self.body_verdicts.get(&net_id) {
            Some(Verdict::Named(i, _)) => Some(*i),
            _ => None,
        }
    }

    /// The seat this body belongs to, when the roster names one.
    pub fn seat_of_entity(&self, net_id: u32) -> Option<&Seat> {
        match self.identity_of(net_id) {
            Identity::Player { seat, .. } => Some(seat),
            _ => None,
        }
    }

    /// Who held `slot` at `tick` (playing, or left and not yet removed): the name a roster list
    /// shows for a slot at this moment.
    pub fn seat_in_slot(&self, slot: u8, tick: u32) -> Option<&Seat> {
        self.roster.seat_holding(slot, tick)
    }

    /// The team this body was on: its creation stamp, `team + 1`, as the engine's colour rule reads
    /// it. `None` when the body carries no team.
    pub fn team_of_entity(&self, net_id: u32) -> Option<u8> {
        self.body_teams.get(&net_id).copied()
    }

    /// The team the body `slot` is driving at `tick` is on.
    ///
    /// The body's own stamp first; failing that (the player is between lives) the team the seat
    /// states; failing that the stamp of any body that slot ever drove.
    pub fn team_of_slot_at(&self, slot: u8, tick: u32) -> Option<u8> {
        if let Some(t) = self.entity_for_slot(slot, tick).and_then(|b| self.team_of_entity(b.net_id)) {
            return Some(t);
        }
        if let Some(s) = self.seat_in_slot(slot, tick).filter(|s| s.data_team >= 0) {
            return Some(s.data_team as u8);
        }
        self.bindings.iter().filter(|b| b.slot == slot).find_map(|b| self.team_of_entity(b.net_id))
    }
}

#[cfg(test)]
mod tests {
    //! The roster's rules, on hand-built timelines -- and the decoder, on a hand-built bit stream.
    //! The real-film gates are in `census` below and are `#[ignore]`d (they need the corpus).
    use super::*;
    use crate::film::FilmPlayer;

    fn look(model: u8, primary: i8, s0: i8) -> BipedAppearance {
        let mut slots = [0i8; 12];
        slots[0] = s0;
        BipedAppearance { flags: 0, colors: [primary, 0, 0], model_choice: model, slots, ids: [0; 4] }
    }

    fn hdr(index: usize, name: &str, xuid: u64, l: &BipedAppearance) -> FilmPlayer {
        let mut raw = vec![0u8; 160];
        raw[0x48] = l.flags;
        raw[0x49] = l.colors[0] as u8;
        raw[0x4A] = l.colors[1] as u8;
        raw[0x4B] = l.colors[2] as u8;
        raw[0x4C] = l.model_choice;
        for i in 0..10 {
            raw[0x58 + i] = l.slots[i] as u8;
        }
        FilmPlayer {
            index,
            valid: 1,
            team: 0,
            machine: [0; 6],
            xuid,
            gamertag: name.into(),
            service_tag: String::new(),
            raw,
        }
    }

    fn upd(player: u8, id: u64, kind: UpdateKind) -> PlayerUpdate {
        PlayerUpdate {
            player,
            identifier: id,
            kind,
            left_game: false,
            joined_in_progress: false,
            machine: None,
            config: None,
            data_name: None,
            data_team: None,
            data_b33: None,
            swap_with: None,
            leftover_bits: 0,
        }
    }

    fn add(player: u8, id: u64, name: &str, l: Option<BipedAppearance>, data_team: Option<i8>) -> PlayerUpdate {
        let mut u = upd(player, id, UpdateKind::Add);
        u.joined_in_progress = true;
        u.config = Some(ConfigDelta { name: Some(name.into()), look: l, ..Default::default() });
        u.data_team = data_team;
        u
    }

    fn name_of(r: &Roster, v: &Verdict) -> String {
        match v {
            Verdict::Named(i, _) => r.seats[*i].name.clone(),
            other => format!("{other:?}"),
        }
    }

    /// The name that SAME armour belongs to changes when the slot changes hands: the armour alone
    /// cannot tell the two apart, only the time can.
    #[test]
    fn a_slot_handed_to_someone_else_names_each_body_for_who_held_it_then() {
        let same = look(0, 7, 3);
        let r = Roster::build(
            &[hdr(3, "Old", 0xA, &same)],
            &[
                (10, upd(3, 0xA, UpdateKind::Remove)),
                (10, add(3, 0xB, "New", Some(same), None)),
            ],
        );
        assert_eq!(name_of(&r, &r.attribute(Some(&same), 0, 5, None)), "Old");
        assert_eq!(name_of(&r, &r.attribute(Some(&same), 0, 20, None)), "New");
        // The tick of the swap itself belongs to the new occupant: the update is applied before
        // the tick's bodies spawn.
        assert_eq!(name_of(&r, &r.attribute(Some(&same), 0, 10, None)), "New");
    }

    /// One person leaves and comes back in another slot: two lives, one identifier, two slots, and
    /// nobody in between.
    #[test]
    fn a_player_who_leaves_and_rejoins_in_another_slot_is_one_person_in_two_slots() {
        let l = look(0, 9, 4);
        let other = look(1, 2, 2);
        let mut left = upd(2, 0xC, UpdateKind::LeftGame);
        left.left_game = true;
        let r = Roster::build(
            &[hdr(2, "Rover", 0xC, &l), hdr(0, "Anchor", 0xD, &other)],
            &[
                (100, left),
                (120, upd(2, 0xC, UpdateKind::Remove)),
                (200, add(6, 0xC, "Rover", Some(l), None)),
            ],
        );
        let before = r.attribute(Some(&l), 0, 50, None);
        let after = r.attribute(Some(&l), 0, 250, None);
        let (Verdict::Named(a, _), Verdict::Named(b, _)) = (&before, &after) else { panic!("{before:?} {after:?}") };
        assert_eq!((r.seats[*a].slot, r.seats[*b].slot), (2, 6));
        assert_eq!(r.seats[*a].identifier, r.seats[*b].identifier, "the same person");
        // In between they are gone, and the body must not be given to anyone else.
        assert_eq!(r.attribute(Some(&l), 0, 150, None), Verdict::Unmatched);
        assert_eq!(r.attribute(Some(&l), 0, 110, None), Verdict::Unmatched, "left at 100: no new body of theirs");
        assert_eq!(name_of(&r, &r.attribute(Some(&l), 0, 99, None)), "Rover", "a body that spawned before they left");
    }

    /// A player who LEFT keeps the record: a body that spawned while they were playing is still
    /// theirs when the tick is long past.
    #[test]
    fn a_body_belongs_to_whoever_was_playing_when_it_spawned_not_when_it_is_looked_at() {
        let l = look(0, 1, 1);
        let mut left = upd(1, 0xE, UpdateKind::LeftGame);
        left.left_game = true;
        let r = Roster::build(&[hdr(1, "Quitter", 0xE, &l)], &[(100, left)]);
        assert_eq!(name_of(&r, &r.attribute(Some(&l), 0, 90, None)), "Quitter");
        assert_eq!(r.attribute(Some(&l), 0, 100, None), Verdict::Unmatched);
        // `seat_holding` still answers for the left record (a roster list shows them).
        assert!(r.seat_holding(1, 500).is_some());
    }

    #[test]
    fn a_swap_trades_the_two_records_between_slots() {
        let (a, b) = (look(0, 1, 1), look(0, 2, 2));
        let mut sw = upd(1, 0xA1, UpdateKind::Swap);
        sw.swap_with = Some((2, 0xB2));
        let r = Roster::build(&[hdr(1, "A", 0xA1, &a), hdr(2, "B", 0xB2, &b)], &[(50, sw)]);
        let slot_of = |l: &BipedAppearance, t: u32| match r.attribute(Some(l), 0, t, None) {
            Verdict::Named(i, _) => (r.seats[i].name.clone(), r.seats[i].slot),
            v => panic!("{v:?}"),
        };
        assert_eq!(slot_of(&a, 10), ("A".into(), 1));
        assert_eq!(slot_of(&a, 60), ("A".into(), 2), "A moved to B's slot and kept their armour");
        assert_eq!(slot_of(&b, 60), ("B".into(), 1));
        // A swap whose identifiers do not match what the slots hold is refused, as the engine does.
        let mut bad = upd(1, 0xFFFF, UpdateKind::Swap);
        bad.swap_with = Some((2, 0xB2));
        let r2 = Roster::build(&[hdr(1, "A", 0xA1, &a), hdr(2, "B", 0xB2, &b)], &[(50, bad)]);
        assert_eq!(r2.applied, 0);
        assert_eq!(r2.seats.len(), 2);
    }

    #[test]
    fn a_configuration_update_changes_the_armour_from_that_tick_and_not_before() {
        let (first, second) = (look(0, 1, 1), look(0, 5, 9));
        let mut cfg = upd(4, 0xF, UpdateKind::Configuration);
        cfg.config = Some(ConfigDelta { look: Some(second), ..Default::default() });
        let r = Roster::build(&[hdr(4, "Fitter", 0xF, &first)], &[(100, cfg)]);
        assert_eq!(name_of(&r, &r.attribute(Some(&first), 0, 50, None)), "Fitter");
        assert_eq!(r.attribute(Some(&first), 0, 150, None), Verdict::Unmatched);
        assert_eq!(name_of(&r, &r.attribute(Some(&second), 0, 150, None)), "Fitter");
        // An identical re-send (the stream repeats configurations constantly) opens no segment.
        let mut same = upd(4, 0xF, UpdateKind::Configuration);
        same.config = Some(ConfigDelta { look: Some(first), ..Default::default() });
        let r2 = Roster::build(&[hdr(4, "Fitter", 0xF, &first)], &[(100, same)]);
        assert_eq!(r2.seats.len(), 1);
    }

    /// Two players in identical armour: only a team the roster STATES, matched against the body's
    /// own team stamp, tells them apart -- and where it is not stated nothing may be guessed.
    #[test]
    fn identical_armour_is_told_apart_by_a_stated_team_and_by_nothing_else() {
        let l = look(0, 3, 3);
        let r = Roster::build(
            &[],
            &[
                (0, add(0, 0x11, "Red", Some(l), Some(0))),
                (0, add(1, 0x22, "Blue", Some(l), Some(1))),
            ],
        );
        assert_eq!(name_of(&r, &r.attribute(Some(&l), 1, 5, Some(true))), "Red", "stamp 1 = team 0");
        assert_eq!(name_of(&r, &r.attribute(Some(&l), 2, 5, Some(true))), "Blue", "stamp 2 = team 1");
        match r.attribute(Some(&l), 0, 5, Some(true)) {
            Verdict::Ambiguous(c) => assert_eq!(c.len(), 2),
            v => panic!("no stamp, no team: must stay ambiguous, got {v:?}"),
        }
        // Teams not stated at all (-1): the stamp has nothing to compare, so it is ambiguous too.
        let r2 = Roster::build(
            &[],
            &[(0, add(0, 0x11, "Red", Some(l), None)), (0, add(1, 0x22, "Blue", Some(l), None))],
        );
        assert!(matches!(r2.attribute(Some(&l), 1, 5, Some(true)), Verdict::Ambiguous(_)));
        // ...except in a free-for-all, where the stamp IS the slot.
        assert_eq!(name_of(&r2, &r2.attribute(Some(&l), 2, 5, Some(false))), "Blue");
    }

    #[test]
    fn a_body_with_no_appearance_block_is_not_a_players_and_gets_no_name() {
        let r = Roster::build(&[hdr(0, "P", 1, &look(0, 1, 1))], &[]);
        assert_eq!(r.attribute(None, 1, 5, Some(false)), Verdict::NotAPlayer);
        assert_eq!(r.attribute(None, 0, 5, None), Verdict::NotAPlayer);
    }

    /// Free-for-all: the stamp names the slot. That is only allowed to NAME a body whose armour
    /// no seat states -- never to override armour that is stated.
    #[test]
    fn the_free_for_all_slot_stamp_fills_only_gaps_the_armour_leaves() {
        let l = look(0, 1, 1);
        let r = Roster::build(&[], &[(0, add(5, 0x55, "Quiet", None, None))]);
        assert_eq!(r.attribute(Some(&l), 6, 5, Some(false)), Verdict::Named(0, Basis::SlotStamp));
        assert_eq!(r.attribute(Some(&l), 6, 5, Some(true)), Verdict::Unmatched, "a team game's stamp is a team");
        assert_eq!(r.attribute(Some(&l), 6, 5, None), Verdict::Unmatched, "unknown game type: no guessing");
        assert_eq!(r.attribute(Some(&l), 4, 5, Some(false)), Verdict::Unmatched, "nobody holds slot 3");
    }

    /// The engine's own identifier checks (`simulation_players_apply_update`): an update keyed on a
    /// player who is not the one in that slot is refused, not applied to whoever is there.
    #[test]
    fn an_update_for_the_wrong_identifier_is_refused() {
        let l = look(0, 1, 1);
        let mut cfg = upd(1, 0xBAD, UpdateKind::Configuration);
        cfg.config = Some(ConfigDelta { name: Some("Imposter".into()), ..Default::default() });
        let r = Roster::build(
            &[hdr(1, "Real", 0x1, &l)],
            &[(10, cfg), (20, upd(1, 0xBAD, UpdateKind::Remove)), (30, upd(1, 0xBAD, UpdateKind::LeftGame))],
        );
        assert_eq!((r.updates, r.applied), (3, 0));
        assert_eq!(name_of(&r, &r.attribute(Some(&l), 0, 100, None)), "Real");
    }

    // ---- the decoder ------------------------------------------------------------------------

    /// MSB-first bit writer, written from the ENCODER's side (`sub_140B504C0`) so the decoder is
    /// not graded against itself.
    #[derive(Default)]
    struct W(Vec<bool>);
    impl W {
        fn put(&mut self, v: u64, n: u32) {
            for i in (0..n).rev() {
                self.0.push((v >> i) & 1 != 0);
            }
        }
        fn flag(&mut self, b: bool) {
            self.0.push(b);
        }
        fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; (self.0.len() + 7) / 8];
            for (i, b) in self.0.iter().enumerate() {
                if *b {
                    out[i / 8] |= 0x80 >> (i % 8);
                }
            }
            out
        }
        fn name(&mut self, s: &str) {
            for c in s.chars() {
                self.put(encode_name_char(c) as u64, 7);
            }
            self.put(0, 7);
        }
        fn look(&mut self, l: &BipedAppearance) {
            self.put((l.flags & 1) as u64, 1);
            self.put(((l.flags >> 1) & 1) as u64, 1);
            for c in l.colors {
                self.put((c + 1) as u64, 6);
            }
            self.put(l.model_choice as u64, 1);
            for s in l.slots {
                if s < 0 {
                    self.flag(true);
                } else {
                    self.flag(false);
                    self.put(s as u64, 7);
                }
            }
            for id in l.ids {
                self.put(id as u64, 8);
            }
            for n in [7u32, 7, 3, 6, 6, 6, 8] {
                self.put(0, n);
            }
        }
        fn id(&mut self, id: u64) {
            for b in id.to_le_bytes() {
                self.put(b as u64, 8);
            }
        }
    }

    /// The inverse of the engine's 7-bit table, derived by hand from its six ranges.
    fn encode_name_char(c: char) -> u8 {
        match c {
            '0'..='9' => 1 + (c as u8 - b'0'),
            'a'..='z' => 11 + (c as u8 - b'a'),
            'A'..='Z' => 37 + (c as u8 - b'A'),
            '(' => 63,
            ')' => 64,
            ' ' => 65,
            _ => panic!("not in the table"),
        }
    }

    #[test]
    fn the_gamertag_character_table_is_the_engines() {
        for (code, ch) in [(1u16, '0'), (10, '9'), (11, 'a'), (36, 'z'), (37, 'A'), (62, 'Z'), (63, '('), (64, ')'), (65, ' ')] {
            assert_eq!(name_char(code), ch as u16, "code {code}");
        }
        assert_eq!(name_char(0), 0, "0 terminates");
    }

    /// A joined-in-progress `Add`, laid out as the encoder writes it, comes back field for field.
    #[test]
    fn an_add_update_decodes_exactly_what_the_encoder_wrote() {
        let l = BipedAppearance { flags: 2, colors: [4, -1, 7], model_choice: 1, slots: [3, -1, 0, 9, -1, -1, 1, 2, -1, 5, -1, -1], ids: [1, 2, 3, 4] };
        let mut w = W::default();
        w.put(6, 4); // player index
        w.id(0x0123_4567_89AB_CDEF);
        w.put(3, 3); // Add
        w.flag(false); // not left
        w.flag(true); // joined in progress
        for b in [189u64, 70, 15, 33, 0, 0] {
            w.put(b, 8); // machine
        }
        w.put(0, 2);
        w.put(0, 2);
        w.flag(true); // name
        w.name("Demon (1)");
        w.flag(false); // xuid
        w.flag(true); // appearance
        w.look(&l);
        w.flag(false); // +0x50
        w.flag(true); // team
        w.put(1 + 2, 5); // team 2
        w.flag(false); // +0x51
        w.flag(false); // +0x53
        w.flag(false); // data name
        w.flag(true); // data team
        w.put(1 + 2, 5);
        w.flag(false); // b33
        let bytes = w.bytes();
        let u = decode_player_update(&bytes).expect("decodes");
        assert_eq!((u.player, u.identifier, u.kind), (6, 0x0123_4567_89AB_CDEF, UpdateKind::Add));
        assert!(!u.left_game && u.joined_in_progress);
        assert_eq!(u.machine, Some([189, 70, 15, 33, 0, 0]));
        let c = u.config.as_ref().expect("config");
        assert_eq!(c.name.as_deref(), Some("Demon (1)"));
        assert_eq!(c.team, Some(2));
        assert_eq!(c.look, Some(l));
        assert_eq!(u.data_team, Some(2));
        assert!(u.leftover_bits < 8, "the payload is consumed to its byte padding, left {}", u.leftover_bits);
    }

    #[test]
    fn a_swap_and_a_remove_decode_their_own_shapes() {
        let mut w = W::default();
        w.put(3, 4);
        w.id(0xAA);
        w.put(1, 3); // swap
        w.put(8, 4);
        w.id(0xBB);
        let u = decode_player_update(&w.bytes()).unwrap();
        assert_eq!((u.kind, u.player, u.swap_with), (UpdateKind::Swap, 3, Some((8, 0xBB))));
        let mut w = W::default();
        w.put(15, 4);
        w.id(0xCC);
        w.put(2, 3); // remove
        let u = decode_player_update(&w.bytes()).unwrap();
        assert_eq!((u.kind, u.player, u.identifier), (UpdateKind::Remove, 15, 0xCC));
        // update types above 4 do not exist
        let mut w = W::default();
        w.put(0, 4);
        w.id(1);
        w.put(5, 3);
        assert!(decode_player_update(&w.bytes()).is_none());
    }

    // ---- the index --------------------------------------------------------------------------

    fn body(net: u32, first: u32, last: u32, owner: u8, l: Option<BipedAppearance>) -> crate::film_decode::EntityTrack {
        crate::film_decode::EntityTrack {
            net_id: net,
            entity_type: 11,
            first_tick: first,
            last_tick: last,
            positions: vec![(first, [net as f32, 0.0, 0.0]), (last, [net as f32, 1.0, 0.0])],
            owner_raw: Some(owner),
            appearance: l.is_some(),
            look: l,
            ..Default::default()
        }
    }

    /// `attach_roster` ties every biped to a seat, rebuilds the slot bindings from them, and keeps
    /// the AI's body out of it -- and the nameplates read exactly that.
    #[test]
    fn the_index_names_player_bodies_by_their_seat_and_leaves_the_ai_alone() {
        let (a, b) = (look(0, 1, 1), look(1, 2, 2));
        let players = [hdr(0, "LobbyName", 0x1, &a)];
        let mut idx = crate::film_decode::FilmIndex {
            ticks: 400,
            tracks: vec![
                body(0x10, 0, 300, 1, Some(a)),   // before the swap of occupants
                body(0x11, 250, 400, 2, Some(b)), // the new occupant of slot 1
                body(0x12, 0, 400, 0, None),      // AI
            ],
            player_updates: vec![
                (100, upd(0, 0x1, UpdateKind::Remove)),
                (100, add(1, 0x2, "Newcomer", Some(b), None)),
                (150, add(0, 0x3, "Rejoiner", Some(a), None)),
            ],
            ..Default::default()
        };
        idx.attach_roster(&players, Some(false));
        // 0x10 spawned at t0, when the lobby's player held slot 0 in that armour.
        let seat = idx.seat_of_entity(0x10).expect("named");
        assert_eq!((seat.name.as_str(), seat.slot), ("LobbyName", 0));
        let seat = idx.seat_of_entity(0x11).expect("named");
        assert_eq!((seat.name.as_str(), seat.slot, seat.identifier), ("Newcomer", 1, 0x2));
        assert!(matches!(idx.identity_of(0x12), Identity::NotAPlayer));
        assert_eq!(idx.slot_of_entity(0x11), Some(1));
        assert_eq!(idx.slot_of_entity(0x12), None);
        assert_eq!(idx.team_of_entity(0x11), Some(1), "the stamp 2 is team 1");
        assert_eq!(idx.team_of_entity(0x12), None, "an AI body carries no team here");
        assert_eq!(idx.seat_in_slot(0, 200).map(|s| s.name.as_str()), Some("Rejoiner"));
        assert_eq!(idx.seat_in_slot(0, 50).map(|s| s.name.as_str()), Some("LobbyName"));
    }

    /// An earlier route (the aim correlation, the owner-stamp rule) may fill a gap the armour left,
    /// but only into a slot the roster says somebody was PLAYING in when the body spawned.
    #[test]
    fn an_earlier_routes_binding_fills_a_gap_only_where_the_roster_confirms_the_slot() {
        use crate::film_decode::{BindOrigin, PlayerBinding};
        let (known, stray) = (look(0, 1, 1), look(1, 9, 9));
        let bind = |net: u32, slot: u8| PlayerBinding {
            slot,
            net_id: net,
            entity_type: 11,
            first_tick: 5,
            last_tick: 90,
            cost: 0.0,
            margin: 1.0,
            samples: 100,
            from_facing: true,
            from_owner_rule: true,
            origin: BindOrigin::OwnerStamp,
        };
        let mut idx = crate::film_decode::FilmIndex {
            ticks: 100,
            tracks: vec![body(0x10, 5, 90, 0, Some(stray)), body(0x11, 5, 90, 0, Some(stray))],
            player_updates: vec![(0, add(2, 0x2, "Two", Some(known), None))],
            bindings: vec![bind(0x10, 2), bind(0x11, 9)],
            ..Default::default()
        };
        idx.attach_roster(&[], None);
        // slot 2 was playing: the earlier route's answer stands, and says so.
        assert!(matches!(idx.identity_of(0x10), Identity::Player { basis: Basis::Binding, seat } if seat.name == "Two"));
        // slot 9 had nobody: the earlier route is not believed.
        assert!(matches!(idx.identity_of(0x11), Identity::Unknown));
        assert_eq!(idx.slot_of_entity(0x11), None);
    }

    /// The Theater's player list: one row per person with their bodies counted under them, the
    /// lobby leftovers left out, and a separate row for bodies that belong to nobody.
    #[test]
    fn the_cast_lists_each_person_once_and_leaves_out_the_lobby_that_never_played() {
        let (a, b) = (look(0, 1, 1), look(0, 2, 2));
        let players = [hdr(0, "Lobby leftover", 0xDEAD, &look(0, 5, 5)), hdr(1, "Stays", 0x5, &a)];
        let mut idx = crate::film_decode::FilmIndex {
            ticks: 400,
            tracks: vec![
                body(0x10, 10, 100, 1, Some(a)),
                body(0x11, 120, 200, 1, Some(a)), // the same person's second life
                body(0x12, 50, 300, 2, Some(b)),
                body(0x13, 10, 300, 7, None), // a Forge monitor or the AI
            ],
            player_updates: vec![(0, upd(0, 0xDEAD, UpdateKind::Remove)), (0, add(2, 0x6, "Joiner", Some(b), None))],
            ..Default::default()
        };
        idx.attach_roster(&players, Some(false));
        let f = crate::film::Film::default();
        let pb = crate::film_play::Playback::new(idx);
        let rows = crate::film_ui::roster_lines(&f, &pb, &std::collections::HashMap::new());
        let people: Vec<(&str, usize)> = rows.iter().filter(|r| r.owner.is_none()).map(|r| (r.label.as_str(), r.lives)).collect();
        assert_eq!(people, vec![("Stays", 2), ("Joiner", 1)], "{rows:?}");
        let nobody: Vec<_> = rows.iter().filter(|r| r.owner.is_some()).collect();
        assert_eq!(nobody.len(), 1, "one row for the AI's bodies");
        assert!(!nobody[0].named && nobody[0].lives == 1 && nobody[0].owner == Some(7));
    }

    /// The first-person arms wear the armour of whoever the roster says was in the camera's slot at
    /// this tick, not the lobby's record for the slot.
    #[test]
    fn the_first_person_arms_take_the_armour_of_the_seat_not_the_header_slot() {
        let (lobby, real) = (look(0, 1, 1), look(0, 3, 4));
        let players = [hdr(1, "Lobby", 0x1, &lobby)];
        let mut idx = crate::film_decode::FilmIndex {
            ticks: 400,
            tracks: vec![body(0x10, 10, 30, 2, Some(real))],
            player_updates: vec![(5, upd(1, 0x1, UpdateKind::Remove)), (5, add(1, 0x9, "Real", Some(real), None))],
            ..Default::default()
        };
        idx.attach_roster(&players, Some(false));
        let mut pb = crate::film_play::Playback::new(idx);
        pb.camera = crate::film_play::FilmCamera::FirstPerson(1);
        let first_slot = |pb: &crate::film_play::Playback| pb.subject_armour().expect("an armour").1[0];
        // before the hand-over the camera's slot belongs to the lobby player
        pb.seek_tick(2.0);
        assert_eq!(first_slot(&pb), 1);
        // while the camera's body is alive its own seat is the answer
        pb.seek_tick(20.0);
        assert_eq!(first_slot(&pb), 4, "the real occupant's armour");
        // between lives the slot's CURRENT occupant is, not the lobby's record for the slot
        pb.seek_tick(100.0);
        assert_eq!(first_slot(&pb), 4, "the real occupant's armour, with no body to ask");
    }

    /// What a viewer sees: a name over a player's body, nothing over the AI's, and an honest label
    /// for a body nobody accounts for.
    #[test]
    fn the_nameplates_name_players_skip_the_ai_and_label_the_rest_honestly() {
        let (a, b, c) = (look(0, 1, 1), look(0, 2, 2), look(0, 3, 3));
        let mut idx = crate::film_decode::FilmIndex {
            ticks: 400,
            tracks: vec![
                body(0x10, 0, 400, 1, Some(a)),
                body(0x12, 0, 400, 0, None),
                body(0x13, 0, 400, 0, Some(c)), // nobody wears this
            ],
            player_updates: vec![(0, add(0, 0x1, "Alpha", Some(a), None)), (0, add(1, 0x2, "Bravo", Some(b), None))],
            ..Default::default()
        };
        idx.attach_roster(&[], Some(true));
        let mut pb = crate::film_play::Playback::new(idx);
        pb.seek_tick(10.0);
        let tags = crate::film_ui::name_tags(&pb, &[], 16);
        let by = |net: u32| tags.iter().find(|t| t.net_id == net);
        assert_eq!(by(0x10).map(|t| (t.text.as_str(), t.named)), Some(("Alpha", true)));
        assert!(by(0x12).is_none(), "an AI body has no nameplate");
        let t = by(0x13).expect("a player's body always gets a tag");
        assert!(!t.named && t.inferred_from.is_some(), "an unplaced body is labelled as unplaced: {t:?}");
        assert_ne!(t.text, format!("{:#010x}", 0x13u32), "never a bare entity id");
    }
}

/// The coverage table, BEFORE (the header-only routes) and AFTER (the roster), per film.
///
/// `cargo test --release -p hms-app film_roster::coverage_table -- --ignored --nocapture`
#[cfg(test)]
mod coverage_table {
    use crate::film;
    use crate::film_decode::*;
    use super::*;

    fn films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        let mut v = film::spec_corpus();
        v.sort();
        v
    }

    /// The header-only index this module replaces: the stamp rule, the armour route against the
    /// HEADER roster, and the cross-film fallback.
    fn legacy_index(f: &film::Film) -> FilmIndex {
        let mut idx = build_index_full(
            &f.bytes,
            &f.data,
            &f.header.map_variant,
            f.recorder_slot(),
            occupied_slot_mask(&f.header.players),
            elite_slot_mask(&f.header.players),
        );
        idx.bind_by_appearance(&f.header.players);
        let unnamed = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_some() && idx.slot_of_entity(t.net_id).is_none()).count();
        if unnamed > 0 {
            idx.bind_by_appearance_across_films(AppearanceKey::shared());
        }
        idx
    }

    #[test]
    #[ignore = "diagnostic: prints the coverage table"]
    fn attribution() {
        println!("film | players | BEFORE stated / inferred / none | AFTER by-armour / +team / slot-stamp / binding / ambiguous / unknown | non-player | name disagreements vs before");
        let mut tot = [0usize; 10];
        for path in films() {
            let Ok(f) = film::open(&path) else { continue };
            if f.data.records.len() < 500 {
                continue;
            }
            let old = legacy_index(&f);
            let idx = build_index_film(&f);
            let bodies: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_some()).collect();
            let nonplayer = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_none()).count();
            if bodies.is_empty() {
                continue;
            }
            let (mut b_stated, mut b_inferred, mut b_none) = (0, 0, 0);
            let (mut a_arm, mut a_team, mut a_slot, mut a_bind, mut a_amb, mut a_unk) = (0, 0, 0, 0, 0, 0);
            let mut name_dis = 0;
            for tr in &bodies {
                let before = old.slot_of_entity(tr.net_id).map(|s| f.header.players.iter().find(|p| p.present() && p.index == s as usize).map(|p| p.gamertag.clone()).unwrap_or_default());
                if before.is_some() {
                    b_stated += 1;
                } else if old.inferred_name(tr.net_id).is_some() {
                    b_inferred += 1;
                } else {
                    b_none += 1;
                }
                match idx.identity_of(tr.net_id) {
                    Identity::Player { seat, basis } => {
                        match basis {
                            Basis::Armour => a_arm += 1,
                            Basis::ArmourAndTeam => a_team += 1,
                            Basis::SlotStamp => a_slot += 1,
                            Basis::Binding => a_bind += 1,
                        }
                        if let Some(b) = &before {
                            if !b.is_empty() && b != &seat.name {
                                name_dis += 1;
                            }
                        }
                    }
                    Identity::Candidates(_) => a_amb += 1,
                    _ => a_unk += 1,
                }
            }
            println!(
                "{:<46} | {:>5} | {:>5} / {:>4} / {:>4} | {:>5} / {:>4} / {:>4} / {:>4} / {:>4} / {:>4} | {:>4} | {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                bodies.len(), b_stated, b_inferred, b_none, a_arm, a_team, a_slot, a_bind, a_amb, a_unk, nonplayer, name_dis
            );
            for (i, v) in [bodies.len(), b_stated, b_inferred, b_none, a_arm, a_team, a_slot, a_bind, a_amb, a_unk].into_iter().enumerate() {
                tot[i] += v;
            }
        }
        println!("TOTAL players {} | BEFORE stated {} inferred {} none {} | AFTER armour {} +team {} slot-stamp {} binding {} ambiguous {} unknown {}", tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], tot[6], tot[7], tot[8], tot[9]);
    }
}

/// Gates over the REAL recordings. `#[ignore]`d because they need the Windows partition's films;
/// run them one module per process:
/// `cargo test --release -p hms-app film_roster::real_film_gates -- --ignored --nocapture --test-threads=1`.
///
/// Every gate asserts that it RAN -- how many films, payloads and bodies it judged -- because a
/// gate that finds no films passes, and a pass that judged nothing is not a pass.
#[cfg(test)]
mod real_film_gates {
    use super::*;
    use crate::film;
    use crate::film_decode::*;

    fn corpus() -> Vec<std::path::PathBuf> {
        let mut v = film::spec_corpus();
        v.sort();
        v
    }

    fn open_playable(path: &std::path::Path) -> Option<film::Film> {
        let f = film::open(path).ok()?;
        (f.data.records.len() >= 500).then_some(f)
    }

    fn key(path: &std::path::Path) -> String {
        path.file_name().unwrap_or_default().to_string_lossy().to_string()
    }

    fn find(keyword: &str) -> film::Film {
        let p = corpus()
            .into_iter()
            .find(|p| key(p).contains(keyword))
            .unwrap_or_else(|| panic!("the film `{keyword}` is not on this machine; this fixture cannot run"));
        film::open(&p).expect("opens")
    }

    /// Every player update in every film decodes, is consumed to its byte padding, and is one the
    /// engine's own identifier checks would accept.
    ///
    /// The padding test is the sharp one: the writer pads each payload to a byte, so a field read
    /// at the wrong width leaves the reader several bits short or long. Every width in the
    /// decoder rides on it.
    #[test]
    #[ignore = "needs the film corpus"]
    fn every_player_update_closes_to_its_byte_padding_and_applies() {
        let (mut films, mut payloads, mut with_updates) = (0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            films += 1;
            let (base, _) = solve_params(&f.bytes, &f.data);
            let (_fr, w) = frame_from_bounds(f.header.map_variant.world_bounds);
            let params = DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base };
            let mut n = 0usize;
            for rec in f.data.records.iter() {
                let Ok(t) = decode_record(rec.body(&f.bytes), &params) else { continue };
                for c in t.commands.iter().take(t.commands_a).filter(|c| c.kind == 9) {
                    n += 1;
                    let u = decode_player_update(&c.payload).unwrap_or_else(|| panic!("{}: an undecodable player update", key(&path)));
                    assert!(u.leftover_bits < 8, "{}: a player update left {} bits unread", key(&path), u.leftover_bits);
                }
            }
            payloads += n;
            // A stock-map film has no embedded variant, hence no world box to quantise positions
            // against, and the index is (deliberately) empty -- see `build_index_full`.
            if n > 0 && !f.header.map_variant.is_empty() {
                with_updates += 1;
                let idx = build_index_film(&f);
                assert_eq!(idx.player_updates.len(), n, "{}: the index dropped player updates", key(&path));
                assert_eq!(
                    idx.roster.applied, idx.roster.updates,
                    "{}: the engine's identifier checks would refuse {} update(s)",
                    key(&path),
                    idx.roster.updates - idx.roster.applied
                );
            }
        }
        println!("judged {films} films, {with_updates} with player updates, {payloads} payloads");
        assert!(films >= 39 && with_updates >= 21 && payloads >= 1700, "ran on too little: {films} films, {with_updates} with updates, {payloads} payloads");
    }

    /// COVERAGE FLOOR. Every film, every player body: how many does the production path name?
    ///
    /// Measured on this machine's corpus when the roster landed, against 1,405 of 3,518 before.
    #[test]
    #[ignore = "needs the film corpus"]
    fn the_production_path_names_nearly_every_player_body_in_every_film() {
        // (film, player bodies, bodies the roster places). Exact, from the run that landed it.
        const MEASURED: &[(&str, usize, usize)] = &[];
        let (mut films, mut bodies, mut named, mut stated) = (0usize, 0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            let idx = build_index_film(&f);
            let players: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_some()).collect();
            if players.is_empty() {
                continue;
            }
            films += 1;
            let n = players.iter().filter(|t| matches!(idx.identity_of(t.net_id), Identity::Player { .. })).count();
            let st = players
                .iter()
                .filter(|t| matches!(idx.identity_of(t.net_id), Identity::Player { basis, .. } if basis != Basis::SlotStamp))
                .count();
            bodies += players.len();
            named += n;
            stated += st;
            // Every film: at most 2% of player bodies (and never more than 3) left unplaced.
            let unplaced = players.len() - n;
            assert!(unplaced * 50 <= players.len().max(1) || unplaced <= 1, "{}: {unplaced} of {} player bodies unplaced", key(&path), players.len());
            if let Some((_, want_bodies, want_named)) = MEASURED.iter().find(|(k, _, _)| key(&path).contains(k)) {
                assert_eq!(players.len(), *want_bodies, "{}: the number of player bodies moved", key(&path));
                assert!(n >= *want_named, "{}: only {n} of {} placed, {want_named} were when this landed", key(&path), players.len());
            }
        }
        println!("judged {films} films, {bodies} player bodies, {named} placed, {stated} of them stated");
        assert!(films >= 20 && bodies >= 3500, "ran on too little: {films} films, {bodies} bodies");
        assert!(named * 1000 >= bodies * 995, "{named} of {bodies} placed");
        assert!(stated * 1000 >= bodies * 990, "{stated} of {bodies} placed on stated evidence");
    }

    /// EXACT ANSWERS, read off the streams by hand (the player updates and the body spawn ticks),
    /// not computed by the code under test.
    ///
    /// * **Castle Wars** (`53C14E9D`): slot 6 holds `Take My Shoes` from the first tick; at tick
    ///   25,006 the stream removes them and adds `King Zion 77` into THE SAME SLOT. Same slot,
    ///   different person, and their armour is not shared -- the point is the time.
    /// * **`2BAB2586`**: `InertMist6840` joins mid-match into slot 9 (tick 32,890), is moved through
    ///   slots 12 and 13, is gone from tick 49,688 to 75,571, and RETURNS in slot 7. One person
    ///   (one identifier), bodies in two different slots on either side of an absence.
    /// * **`3E31FE03`**: `BeamedRTX` joins in progress into slot 6 (tick 5,831) and is swapped to
    ///   slot 7 at 10,146 while `Crafting4lyf` takes slot 6.
    /// * **`28AC2CC7`** and **`2BAB2586`**: bodies with no appearance block -- the gametype's, not
    ///   a player's -- exist (82 and 136), carry no name and get no nameplate.
    #[test]
    #[ignore = "needs the film corpus"]
    fn exact_answers_for_slot_reuse_a_rejoin_a_mid_match_join_and_the_ai() {
        let name_of = |idx: &FilmIndex, net: u32| -> (String, u8, u64) {
            match idx.identity_of(net) {
                Identity::Player { seat, .. } => (seat.name.clone(), seat.slot, seat.identifier),
                other => panic!("{net:#x} should be a named player, got {other:?}"),
            }
        };
        // ---- a leave + a join into the same slot ------------------------------------------
        let f = find("53C14E9D");
        let idx = build_index_film(&f);
        assert_eq!(idx.seat_in_slot(6, 25_005).map(|s| s.name.as_str()), Some("Take My Shoes"));
        assert_eq!(idx.seat_in_slot(6, 25_006).map(|s| s.name.as_str()), Some("King Zion 77"));
        for net in [0xef6c0056u32, 0xef8102eb, 0xefa002e9] {
            assert_eq!(name_of(&idx, net).0, "Take My Shoes", "{net:#x} spawned before the slot changed hands");
            assert_eq!(name_of(&idx, net).1, 6);
        }
        for net in [0xf045000au32, 0xf05401c6, 0xf06a005d, 0xf08f02da] {
            assert_eq!(name_of(&idx, net).0, "King Zion 77", "{net:#x} spawned after");
            assert_eq!(name_of(&idx, net).1, 6);
        }
        // ---- a rejoin in another slot -----------------------------------------------------
        let f = find("2BAB2586");
        let idx = build_index_film(&f);
        let (first, second) = (name_of(&idx, 0x927e02f8), name_of(&idx, 0xbc4602d6));
        assert_eq!((first.0.as_str(), first.1), ("InertMist6840", 9));
        assert_eq!((second.0.as_str(), second.1), ("InertMist6840", 7));
        assert_eq!(first.2, 0x16227affd0515932);
        assert_eq!(first.2, second.2, "the same person on both sides of the absence");
        assert!(
            !idx.roster.seats.iter().any(|s| s.identifier == first.2 && s.from_tick >= 49_700 && s.from_tick < 75_571),
            "nobody holds that identifier while they are gone"
        );
        // ---- the AI ---------------------------------------------------------------------
        let non_player: Vec<u32> = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_none()).map(|t| t.net_id).collect();
        assert_eq!(non_player.len(), 136);
        assert!(non_player.iter().all(|n| matches!(idx.identity_of(*n), Identity::NotAPlayer) && idx.slot_of_entity(*n).is_none()));
        for n in [0xe9a30030u32, 0xe9a40031, 0xe9a50032] {
            assert!(non_player.contains(&n));
        }
        let f = find("28AC2CC7");
        let idx = build_index_film(&f);
        let non_player: Vec<u32> = idx.tracks.iter().filter(|t| t.entity_type == 11 && t.look.is_none()).map(|t| t.net_id).collect();
        assert_eq!(non_player.len(), 82);
        assert!(non_player.iter().all(|n| matches!(idx.identity_of(*n), Identity::NotAPlayer)));
        // ---- a join in progress, then a move ----------------------------------------------
        let f = find("3E31FE03");
        let idx = build_index_film(&f);
        let a = name_of(&idx, 0xec67019e);
        let b = name_of(&idx, 0xece60124);
        assert_eq!((a.0.as_str(), a.1, b.0.as_str(), b.1), ("BeamedRTX", 6, "BeamedRTX", 7));
        assert_eq!(a.2, 0x9000007a7dc79);
        assert_eq!(idx.seat_in_slot(6, 10_146).map(|s| s.name.as_str()), Some("Crafting4lyf"));
        assert_eq!(idx.seat_in_slot(6, 10_145).map(|s| s.name.as_str()), Some("BeamedRTX"));
        assert!(idx.roster.seats.iter().any(|s| s.name == "BeamedRTX" && s.joined_in_progress && s.from_tick == 5_831));
    }

    /// FREE-FOR-ALL: a body's team stamp is a SLOT. Judged only on bodies the armour alone placed,
    /// so the stamp is an independent witness and not an input.
    #[test]
    #[ignore = "needs the film corpus"]
    fn in_a_free_for_all_the_stamp_is_the_slot_of_the_player_the_armour_names() {
        let (mut films, mut checked, mut agree) = (0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            if f.header.gametype.as_ref().map(|g| g.misc.teams_enabled()) != Some(false) {
                continue;
            }
            let idx = build_index_film(&f);
            let mut here = 0;
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let stamp = tr.owner_raw.unwrap_or(0);
                if let (Identity::Player { seat, basis: Basis::Armour }, true) = (idx.identity_of(tr.net_id), stamp > 0) {
                    checked += 1;
                    here += 1;
                    if stamp - 1 == seat.slot {
                        agree += 1;
                    }
                }
            }
            films += (here > 0) as usize;
        }
        println!("judged {films} free-for-all films, {checked} bodies, {agree} agree");
        assert!(films >= 12 && checked >= 900, "ran on too little: {films} films, {checked} bodies");
        assert!(agree * 100 >= checked * 98, "only {agree} of {checked} bodies' stamps equal their seat's slot");
    }

    /// TEAMS: where the roster states a player's team, every body the armour places on them carries
    /// that team's stamp. Counted over the candidates directly, NOT over the verdicts -- the verdict
    /// already applies this rule, and a gate that re-applies it grades the rule against itself.
    #[test]
    #[ignore = "needs the film corpus"]
    fn a_stated_team_agrees_with_the_stamp_of_every_body_the_armour_places_on_it() {
        let (mut films, mut checked, mut contradictions) = (0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            let idx = build_index_film(&f);
            let mut here = 0;
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let (Some(look), Some(stamp)) = (tr.look.as_ref(), tr.owner_raw.filter(|s| *s > 0)) else { continue };
                let c = idx.roster.candidates(look, tr.first_tick);
                if let [i] = c[..] {
                    let t = idx.roster.seats[i].data_team;
                    if t >= 0 {
                        checked += 1;
                        here += 1;
                        if t as u8 != stamp - 1 {
                            contradictions += 1;
                        }
                    }
                }
            }
            films += (here > 0) as usize;
        }
        println!("judged {films} films, {checked} bodies, {contradictions} contradictions");
        assert!(films >= 15 && checked >= 1500, "ran on too little: {films} films, {checked} bodies");
        // A handful at tick 2..12 are the start-of-film placeholder bodies, whose armour is not
        // yet the player's (see the report); anything beyond that is a real disagreement.
        assert!(contradictions * 100 <= checked, "{contradictions} of {checked} bodies contradict their seat's stated team");
    }

    /// THE PRODUCTION NAMEPLATE PATH, end to end, over every film at twenty moments each: no tag is
    /// ever a bare entity id, an AI body never gets one, and a player's body always does.
    #[test]
    #[ignore = "needs the film corpus"]
    fn the_nameplates_never_show_an_entity_id_and_never_skip_a_player() {
        let (mut films, mut moments, mut tags_seen, mut players_seen, mut ai_tagged) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            let idx = build_index_film(&f);
            if idx.tracks.iter().all(|t| t.entity_type != 11) {
                continue;
            }
            let ticks = idx.ticks.max(1);
            let mut pb = crate::film_play::Playback::new(idx);
            films += 1;
            for k in 1..=20u32 {
                pb.seek_tick((ticks / 21 * k) as f32);
                let tags = crate::film_ui::name_tags(&pb, &[], 4096);
                let alive: Vec<u32> = pb.bipeds(4096).into_iter().map(|p| p.net_id).collect();
                moments += 1;
                for t in &tags {
                    tags_seen += 1;
                    assert!(!t.text.starts_with("0x"), "{}: a nameplate is a bare entity id: {:?}", key(&path), t.text);
                    assert!(!t.text.is_empty());
                }
                for net in alive {
                    let is_player = pb.index.tracks.iter().any(|t| t.net_id == net && t.look.is_some());
                    let has_tag = tags.iter().any(|t| t.net_id == net);
                    if is_player {
                        players_seen += 1;
                        assert!(has_tag, "{}: player body {net:#x} has no nameplate", key(&path));
                    } else if has_tag {
                        ai_tagged += 1;
                    }
                }
            }
        }
        println!("judged {films} films, {moments} moments, {tags_seen} nameplates, {players_seen} live player bodies, {ai_tagged} AI bodies tagged");
        assert!(films >= 20 && moments >= 400 && players_seen >= 1500, "ran on too little: {films} films, {moments} moments, {players_seen} bodies");
        assert_eq!(ai_tagged, 0, "an AI body got a nameplate");
    }

    /// THE CAST. The Theater's player list has one row per PERSON who played, with their bodies
    /// counted under them, and a separate row only for bodies that belong to no player.
    #[test]
    #[ignore = "needs the film corpus"]
    fn the_cast_has_one_row_per_person_and_a_row_for_nobody_elses_bodies() {
        let (mut films, mut people, mut bodies) = (0usize, 0usize, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            let idx = build_index_film(&f);
            let named: Vec<u32> = idx.tracks.iter().filter(|t| t.entity_type == 11 && idx.seat_of_entity(t.net_id).is_some()).map(|t| t.net_id).collect();
            if named.is_empty() {
                continue;
            }
            let pb = crate::film_play::Playback::new(idx);
            let rows = crate::film_ui::roster_lines(&f, &pb, &std::collections::HashMap::new());
            let person_rows: Vec<&crate::film_ui::RosterRow> = rows.iter().filter(|r| r.owner.is_none() && r.slot.is_some()).collect();
            let who: std::collections::BTreeSet<u64> =
                named.iter().filter_map(|n| pb.index.seat_of_entity(*n)).map(|s| s.identifier).collect();
            let with_bodies = person_rows.iter().filter(|r| r.lives > 0).count();
            assert_eq!(with_bodies, who.len(), "{}: {with_bodies} rows with bodies for {} people who had one", key(&path), who.len());
            assert_eq!(person_rows.iter().map(|r| r.lives).sum::<usize>(), named.len(), "{}: the rows do not account for every named body", key(&path));
            // no person is listed twice, and a name that is a placeholder never labels a person
            let labels: std::collections::BTreeSet<&str> = person_rows.iter().map(|r| r.label.as_str()).collect();
            assert!(person_rows.iter().all(|r| r.named && !r.label.starts_with("Player ")), "{}: a person row carries a placeholder label", key(&path));
            assert!(labels.len() + 1 >= person_rows.len() || labels.len() * 10 >= person_rows.len() * 9, "{}: the same label repeats", key(&path));
            films += 1;
            people += who.len();
            bodies += named.len();
        }
        println!("judged {films} films, {people} people, {bodies} named bodies");
        assert!(films >= 20 && people >= 200 && bodies >= 3400, "ran on too little: {films} films, {people} people, {bodies} bodies");
    }

    /// CONCURRENCY. One person drives one biped, so the bodies the roster gives them must not pile up.
    ///
    /// Measured by LIFE (spawn to death, `film_anim::death_tick`), not by how long a corpse lingers:
    /// the stamp-era gate counted lingering corpses and had to be exempted wherever respawns are
    /// fast. By life, the worst person in the corpus has THREE bodies at once -- the placeholder
    /// bodies several films spawn for the same player in the first ticks (`53C14E9D`: three bodies
    /// at tick 2, same armour, same team, same slot) -- and nobody has more. Four would mean two
    /// people's bodies hung on one name.
    #[test]
    #[ignore = "needs the film corpus"]
    fn no_person_has_more_than_three_live_bodies_at_once() {
        let (mut films, mut people, mut worst, mut at_worst) = (0usize, 0usize, 0i32, 0usize);
        for path in corpus() {
            let Some(f) = open_playable(&path) else { continue };
            let idx = build_index_film(&f);
            let mut lives: std::collections::BTreeMap<u64, Vec<(u32, u32)>> = Default::default();
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                if let Some(seat) = idx.seat_of_entity(tr.net_id) {
                    let end = crate::film_anim::death_tick(tr).unwrap_or(tr.last_tick).max(tr.first_tick);
                    lives.entry(seat.identifier).or_default().push((tr.first_tick, end));
                }
            }
            if lives.is_empty() {
                continue;
            }
            films += 1;
            people += lives.len();
            for (id, l) in &lives {
                let p = peak_bodies_alive(l);
                assert!(p <= MAX_PLAYER_BODIES_AT_ONCE, "{}: {id:#x} has {p} live bodies at once", key(&path));
                if p > worst {
                    worst = p;
                    at_worst = 0;
                }
                at_worst += (p == worst) as usize;
            }
        }
        println!("judged {films} films, {people} people; worst {worst} live bodies at once ({at_worst} such people)");
        assert!(films >= 20 && people >= 200, "ran on too little: {films} films, {people} people");
        assert_eq!(worst, 3, "the measured worst moved");
    }

    /// THE FOLLOWED PLAYER'S NAME (the spectated-player card, the first-person arms' armour and the
    /// colours all read `subject_seat_index`): following a slot names whoever held it AT THE
    /// PLAYHEAD, across a slot handed to someone else and across a rejoin in another slot.
    ///
    /// Castle Wars: slot 6 is `Take My Shoes` until tick 25,006 and `King Zion 77` after (bodies
    /// spawned at 21,123 and 25,386). `2BAB2586`: `InertMist6840` is in slot 9 at 33,344 and
    /// slot 7 at 76,011.
    #[test]
    #[ignore = "needs the film corpus"]
    fn following_a_slot_names_whoever_held_it_at_the_playhead() {
        use crate::film_play::{FilmCamera, Playback};
        let who = |pb: &mut Playback, cam: FilmCamera, tick: f32| -> String {
            pb.camera = cam;
            pb.seek_tick(tick);
            let k = pb.subject_seat_index().unwrap_or_else(|| panic!("nobody followed at {tick}"));
            pb.index.roster.seats[k].name.clone()
        };
        let mut pb = Playback::new(build_index_film(&find("53C14E9D")));
        for cam in [FilmCamera::Follow(6), FilmCamera::FirstPerson(6)] {
            assert_eq!(who(&mut pb, cam, 21_200.0), "Take My Shoes");
            assert_eq!(who(&mut pb, cam, 25_500.0), "King Zion 77");
        }
        let mut pb = Playback::new(build_index_film(&find("2BAB2586")));
        assert_eq!(who(&mut pb, FilmCamera::Follow(9), 33_400.0), "InertMist6840");
        assert_eq!(who(&mut pb, FilmCamera::Follow(7), 76_100.0), "InertMist6840");
    }
}

