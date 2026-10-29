//! #film-megalo  The MEGALO VARIABLES a saved film records.
//!
//! A Reach film embeds the whole `c_game_variant` (so the gametype's variable DECLARATIONS) and
//! records the replication stream (so the variable VALUES the engine chose to send). This module
//! decodes both and joins them: [`MegaloDecls`] is what the gametype declares, [`MegaloSeries`]
//! is every value change the recording carries, per scope and owner, as a time series.
//!
//! # Where the facts come from (all PROVEN = read out of `haloreach.dll`, MCC build sha16
//! `738dd2d24ea3aea1`, with ReverseMe; no third-party code was read)
//!
//! * **Declarations.** `GameVariantContext_DecodeFromBitstream` (`sub_180055044`) fills the
//!   script context at `c_game_variant + 0x63F8` in this file's image (the dll's offsets are
//!   object-relative and the object starts at +4, the same shift `film_variant` documents).
//!   Four declaration sets live in it: global `+0x7F30`, player `+0x7FC0`, object `+0x801C`,
//!   team `+0x8078`. Each is a set of counts, the variables' initial values and a 2-bit
//!   **network priority** per number/player/object variable. The very same sets are what the
//!   replicators below index, so the layout is cross-checked by two functions.
//! * **Values.** Three entity types carry them, each through ONE shared per-scope codec:
//!   - type 3 `game-engine-player`, fields `0x18` (priority 1 pass) and `0x19` (priority 2
//!     pass) -> `sub_18014B084(state+0x6C, bs, decl_player, host, pass)`;
//!   - type 4 `game-engine-megalo-team`, fields `0` and `1` -> `sub_18014B34C`;
//!   - type 1 `custom-engine-globals` (and 0 `sandbox-`), fields `12` and `13` ->
//!     `sub_18014B720`.
//!
//!   Per pass the codec writes, in this order: number variables whose declared priority equals
//!   the pass (raw 32 bits), player variables with that priority (flag; clear = 4 bits, set =
//!   none), then -- **in pass 1 only, regardless of any declared priority** -- every team
//!   variable (4 bits, stored minus 1) and every timer variable (14-bit seconds, 5-bit rate,
//!   1-bit at-edge), then object variables with that priority (27-bit reference).
//! * **What is NOT in a film.** Priority `0` (none) and `3` (local) are never serialised: the
//!   codec only tests `== 1` and `== 2`. A variable declared that way is simply absent from every
//!   recording. Likewise per-OBJECT variables: there is no object-scope replicator in the three
//!   entity types above, so the object declaration set is reported but its values are not
//!   recoverable from a film. (See `UNRECOVERABLE`.)
//!
//! Variable NAMES do not exist in a gametype -- a Megalo variable is only ever `scope.kind[n]`
//! (`player.number[2]`), so this module reports slots, not names. The value ORDER (what a
//! variable means) belongs to the script, which is not interpreted here.

use crate::film_decode::{BitReader, DecodeParams, DecodeStop, EntityMessage};

/// Offset of the Megalo script context inside this file's `c_game_variant` image.
pub const CTX: usize = 0x63F8;
const GLOBAL_B: usize = CTX + 0x7F30;
const PLAYER_B: usize = CTX + 0x7FC0;
const OBJECT_B: usize = CTX + 0x801C;
const TEAM_B: usize = CTX + 0x8078;
/// HUD widget count (`ctx + 0x80D8`): the number of `hud-widget` fields the globals entity has.
const WIDGET_COUNT_AT: usize = CTX + 0x80D8;
/// Script-section record counts (INFERRED names: the decode reads them first, 9/10/11 bits).
const TRIGGER_COUNT_AT: usize = CTX;
const ACTION_COUNT_AT: usize = CTX + 0xF08;
const CONDITION_COUNT_AT: usize = CTX + 0x2F10;

pub const UNRECOVERABLE: &str = "Not in a film: variables declared with network priority none (0) or local (3) -- the replication codec only serialises priorities 1 and 2; every per-OBJECT variable (no object-scope replicator exists); variable NAMES (a gametype stores none: variables are `scope.kind[n]`); anything the script computed but never stored in a replicated variable; and the values of a variable before the recording's first update of it (a film that starts mid-match carries each variable's value only once it changes).";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    Global,
    Player,
    Team,
    Object,
}

impl Scope {
    pub fn name(self) -> &'static str {
        match self {
            Scope::Global => "global",
            Scope::Player => "player",
            Scope::Team => "team",
            Scope::Object => "object",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Number,
    Timer,
    Team,
    Player,
    Object,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Number => "number",
            Kind::Timer => "timer",
            Kind::Team => "team",
            Kind::Player => "player",
            Kind::Object => "object",
        }
    }
}

/// A network priority as the declaration stores it: 0 none, 1 = the "low-pri" pass (entity field
/// `0x18` / team field 0 / globals field 12), 2 = the "high-pri" pass (`0x19` / 1 / 13), 3 local.
/// The 1/2 -> low/high naming is INFERRED from the team entity's debug field names
/// (`low-pri-variables`, `high-pri-variables`); that only 1 and 2 are sent is PROVEN.
pub fn priority_name(p: u8) -> &'static str {
    match p {
        0 => "none",
        1 => "low",
        2 => "high",
        3 => "local",
        _ => "invalid",
    }
}

/// One declared variable slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarDecl {
    pub kind: Kind,
    pub slot: u8,
    /// Declared network priority; `None` for timers (a timer has no priority field).
    pub priority: Option<u8>,
    /// Whether the replication codec will EVER send this variable: priority 1 or 2, or (team and
    /// timer variables) always in pass 1.
    pub replicated: bool,
    /// The 3-byte initial value `(a, b, type)`; type 0 = the immediate in `a,b` (i16).
    pub init: (u8, u8, u8),
}

impl VarDecl {
    pub fn init_text(&self) -> String {
        if self.init.2 == 0 {
            format!("{}", i16::from_le_bytes([self.init.0, self.init.1]))
        } else {
            format!("ref(type {}, {}, {})", self.init.2, self.init.0, self.init.1)
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScopeDecl {
    pub vars: Vec<VarDecl>,
}

impl ScopeDecl {
    pub fn count(&self, k: Kind) -> usize {
        self.vars.iter().filter(|v| v.kind == k).count()
    }
    fn of(&self, k: Kind) -> impl Iterator<Item = &VarDecl> {
        self.vars.iter().filter(move |v| v.kind == k)
    }
    /// Variables sent in `pass` (1 or 2), in the codec's own order.
    fn in_pass(&self, pass: u8) -> Vec<&VarDecl> {
        let mut out = Vec::new();
        for k in [Kind::Number, Kind::Player, Kind::Team, Kind::Timer, Kind::Object] {
            for v in self.of(k) {
                let sent = match k {
                    Kind::Team | Kind::Timer => pass == 1,
                    _ => v.priority == Some(pass),
                };
                if sent {
                    out.push(v);
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MegaloDecls {
    pub global: ScopeDecl,
    pub player: ScopeDecl,
    pub team: ScopeDecl,
    pub object: ScopeDecl,
    pub hud_widgets: u32,
    pub triggers: u32,
    pub actions: u32,
    pub conditions: u32,
    /// True when every count was inside its capacity and every priority <= 3.
    pub sane: bool,
}

fn rd32(v: &[u8], o: usize) -> u32 {
    v.get(o..o + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn rd8(v: &[u8], o: usize) -> u8 {
    v.get(o).copied().unwrap_or(0)
}

/// Layout of one declaration set, relative to its base. Global differs from the other three.
struct Layout {
    ints: (usize, usize, usize, usize), // count, entries, max, priority array
    timers: (usize, usize, usize),      // count, entries, max
    teams: (usize, usize, usize),       // count, values, max
    players: (usize, usize, usize),     // count, priority array, max
    objects: (usize, usize, usize),     // count, priority array, max
}

const GLOBAL_LAYOUT: Layout = Layout {
    ints: (0x00, 0x08, 12, 0x5C),
    timers: (0x2C, 0x34, 8),
    teams: (0x4C, 0x54, 8),
    players: (0x88, 0x68, 8),
    objects: (0x8C, 0x70, 16),
};
const PLAYER_LAYOUT: Layout = Layout {
    ints: (0x00, 0x08, 8, 0x40),
    timers: (0x20, 0x28, 4),
    teams: (0x34, 0x3C, 4),
    players: (0x54, 0x48, 4),
    objects: (0x58, 0x4C, 4),
};
const OBJECT_LAYOUT: Layout = PLAYER_LAYOUT;
const TEAM_LAYOUT: Layout = Layout {
    ints: (0x00, 0x08, 8, 0x40),
    timers: (0x20, 0x28, 4),
    teams: (0x34, 0x3C, 4),
    players: (0x58, 0x48, 4),
    objects: (0x5C, 0x4C, 4),
};

fn parse_scope(v: &[u8], base: usize, l: &Layout, sane: &mut bool) -> ScopeDecl {
    let mut vars = Vec::new();
    let mut count = |at: usize, max: usize, sane: &mut bool| -> usize {
        let c = rd32(v, base + at) as usize;
        if c > max {
            *sane = false;
        }
        c.min(max)
    };
    let n = count(l.ints.0, l.ints.2, sane);
    for i in 0..n {
        let e = base + l.ints.1 + i * 3;
        let p = rd8(v, base + l.ints.3 + i);
        if p > 3 {
            *sane = false;
        }
        vars.push(VarDecl { kind: Kind::Number, slot: i as u8, priority: Some(p), replicated: p == 1 || p == 2, init: (rd8(v, e), rd8(v, e + 1), rd8(v, e + 2)) });
    }
    let n = count(l.timers.0, l.timers.2, sane);
    for i in 0..n {
        let e = base + l.timers.1 + i * 3;
        vars.push(VarDecl { kind: Kind::Timer, slot: i as u8, priority: None, replicated: true, init: (rd8(v, e), rd8(v, e + 1), rd8(v, e + 2)) });
    }
    let n = count(l.teams.0, l.teams.2, sane);
    for i in 0..n {
        vars.push(VarDecl { kind: Kind::Team, slot: i as u8, priority: None, replicated: true, init: (rd8(v, base + l.teams.1 + i), 0, 0xFF) });
    }
    let n = count(l.players.0, l.players.2, sane);
    for i in 0..n {
        let p = rd8(v, base + l.players.1 + i);
        if p > 3 {
            *sane = false;
        }
        vars.push(VarDecl { kind: Kind::Player, slot: i as u8, priority: Some(p), replicated: p == 1 || p == 2, init: (0, 0, 0xFF) });
    }
    let n = count(l.objects.0, l.objects.2, sane);
    for i in 0..n {
        let p = rd8(v, base + l.objects.1 + i);
        if p > 3 {
            *sane = false;
        }
        vars.push(VarDecl { kind: Kind::Object, slot: i as u8, priority: Some(p), replicated: p == 1 || p == 2, init: (0, 0, 0xFF) });
    }
    ScopeDecl { vars }
}

impl MegaloDecls {
    /// Parse the declarations out of a raw `c_game_variant` image (0xFC04 bytes). `None` when the
    /// image is too short or is not a Megalo (engine type 2) variant.
    pub fn parse(variant: &[u8]) -> Option<MegaloDecls> {
        if variant.len() < WIDGET_COUNT_AT + 4 || rd32(variant, 0) != 2 {
            return None;
        }
        let mut sane = true;
        let global = parse_scope(variant, GLOBAL_B, &GLOBAL_LAYOUT, &mut sane);
        let player = parse_scope(variant, PLAYER_B, &PLAYER_LAYOUT, &mut sane);
        let object = parse_scope(variant, OBJECT_B, &OBJECT_LAYOUT, &mut sane);
        let team = parse_scope(variant, TEAM_B, &TEAM_LAYOUT, &mut sane);
        let hud = rd32(variant, WIDGET_COUNT_AT);
        if hud > 4 {
            sane = false;
        }
        Some(MegaloDecls {
            global,
            player,
            team,
            object,
            hud_widgets: hud.min(4),
            triggers: rd32(variant, TRIGGER_COUNT_AT),
            actions: rd32(variant, ACTION_COUNT_AT),
            conditions: rd32(variant, CONDITION_COUNT_AT),
            sane,
        })
    }

    pub fn scope(&self, s: Scope) -> &ScopeDecl {
        match s {
            Scope::Global => &self.global,
            Scope::Player => &self.player,
            Scope::Team => &self.team,
            Scope::Object => &self.object,
        }
    }

    /// Machine-readable declaration list: one JSON object per variable slot.
    pub fn to_json(&self) -> String {
        let mut rows = Vec::new();
        for s in [Scope::Global, Scope::Player, Scope::Team, Scope::Object] {
            for v in &self.scope(s).vars {
                rows.push(format!(
                    "{{\"scope\":\"{}\",\"kind\":\"{}\",\"slot\":{},\"priority\":{},\"replicated\":{},\"initial\":\"{}\"}}",
                    s.name(),
                    v.kind.name(),
                    v.slot,
                    v.priority.map_or("null".to_string(), |p| format!("\"{}\"", priority_name(p))),
                    v.replicated && s != Scope::Object,
                    v.init_text()
                ));
            }
        }
        format!(
            "{{\"engine\":\"megalo\",\"sane\":{},\"hud_widgets\":{},\"triggers\":{},\"actions\":{},\"conditions\":{},\"variables\":[{}]}}",
            self.sane,
            self.hud_widgets,
            self.triggers,
            self.actions,
            self.conditions,
            rows.join(",")
        )
    }

    /// CSV: `scope,kind,slot,priority,replicated,initial`.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("scope,kind,slot,priority,replicated,initial\n");
        for sc in [Scope::Global, Scope::Player, Scope::Team, Scope::Object] {
            for v in &self.scope(sc).vars {
                s.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    sc.name(),
                    v.kind.name(),
                    v.slot,
                    v.priority.map_or("-", priority_name),
                    v.replicated && sc != Scope::Object,
                    v.init_text()
                ));
            }
        }
        s
    }
}

/// The image of the embedded `c_game_variant` inside a parsed film.
pub fn variant_image(f: &crate::film::Film) -> Option<&[u8]> {
    let c = f.chunks.iter().find(|c| &c.magic == b"flmh")?;
    let p = c.payload(&f.bytes);
    let at = crate::film::flmh::GAME_OPTIONS + crate::film::game_options::GAME_VARIANT;
    p.get(at..at + crate::film::game_options::GAME_VARIANT_LEN)
}

pub fn decls_of(f: &crate::film::Film) -> Option<MegaloDecls> {
    MegaloDecls::parse(variant_image(f)?)
}

// ---------------------------------------------------------------------------------------------
// values
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Num(i32),
    /// `timer-seconds-left` (14 bits), `timer-enumerated-rate` (5), `timer-at-edge` (1).
    Timer { seconds: u16, rate: u8, at_edge: bool },
    /// A team index, `-1` = none.
    Team(i8),
    /// A player slot (0..15), `-1` = none.
    Player(i8),
    /// An object reference in the net-id namespace; `None` = the null reference.
    Object(Option<u32>),
}

impl Value {
    pub fn text(&self) -> String {
        match self {
            Value::Num(n) => n.to_string(),
            Value::Timer { seconds, rate, at_edge } => format!("{seconds}s rate{rate}{}", if *at_edge { " edge" } else { "" }),
            Value::Team(t) => t.to_string(),
            Value::Player(p) => p.to_string(),
            Value::Object(None) => "null".into(),
            Value::Object(Some(o)) => format!("0x{o:08X}"),
        }
    }
    /// A single number for plotting / CSV `value`: seconds for a timer, the id otherwise.
    pub fn number(&self) -> i64 {
        match self {
            Value::Num(n) => *n as i64,
            Value::Timer { seconds, .. } => *seconds as i64,
            Value::Team(t) => *t as i64,
            Value::Player(p) => *p as i64,
            Value::Object(o) => o.map_or(-1, |x| x as i64),
        }
    }
}

fn read_timer(r: &mut BitReader, seconds_bits: u32) -> Result<Value, DecodeStop> {
    let seconds = r.bits(seconds_bits)? as u16;
    let rate = r.bits(5)? as u8;
    let at_edge = r.bits(1)? != 0;
    Ok(Value::Timer { seconds, rate, at_edge })
}

/// `sub_1803CCC04(dst, bs, 0, host)`: optional object reference.
fn read_object(r: &mut BitReader, host: bool) -> Result<Option<u32>, DecodeStop> {
    if !r.flag()? {
        return Ok(None);
    }
    if host {
        let idx = r.bits(10)?;
        let gen = r.bits(2)?;
        Ok(Some(((gen as u32) << 30) | idx as u32))
    } else {
        let hi = r.bits(16)?;
        let lo = r.bits(11)?;
        Ok(Some(((hi << 16) | lo) as u32))
    }
}

/// One per-scope variable codec pass (`sub_18014B084` / `B34C` / `B720`).
fn read_pass(r: &mut BitReader, d: &ScopeDecl, pass: u8, host: bool, out: &mut Vec<(Kind, u8, Value)>) -> Result<(), DecodeStop> {
    for v in d.in_pass(pass) {
        let val = match v.kind {
            Kind::Number => Value::Num(r.bits(32)? as u32 as i32),
            Kind::Player => {
                if r.flag()? {
                    Value::Player(-1)
                } else {
                    Value::Player(r.bits(4)? as i8)
                }
            }
            Kind::Team => Value::Team(r.bits(4)? as i8 - 1),
            Kind::Timer => read_timer(r, 14)?,
            Kind::Object => Value::Object(read_object(r, host)?),
        };
        out.push((v.kind, v.slot, val));
    }
    Ok(())
}

/// The decode parameters the film index itself uses (so a message cuts exactly where the index's
/// does): solved header widths, the per-axis position widths from the map's world bounds, and
/// the 1-bit update prefix.
pub fn film_params(f: &crate::film::Film) -> DecodeParams {
    let (base, _) = crate::film_decode::solve_params(&f.bytes, &f.data);
    let (_frame, w) = crate::film_decode::frame_from_bounds(f.header.map_variant.world_bounds);
    DecodeParams { cv3_axis_bits: w, cv3_total_bits: w.iter().sum(), angle_always: true, update_prefix_bits: 1, ..base }
}

/// What one entity message yielded.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    /// Creation messages only: the player slot / team index the entity stands for.
    pub owner: Option<u8>,
    pub vars: Vec<(Kind, u8, Value)>,
    /// Engine-state fields of the globals entity (`name`, value), when sent.
    pub game: Vec<(&'static str, i64)>,
    pub rebuilt_mask: u64,
    pub leftover_bits: usize,
    /// The walk reached the end of the entity's field list with every later field absent, so the
    /// payload could be checked against the message's own mask (the free oracle).
    pub closed: bool,
    pub stopped_at: Option<u32>,
    pub failed: bool,
}

impl Walk {
    /// The two free oracles: the mask rebuilt from the presence bits agrees with the message's
    /// own mask (over the fields walked), and a CLOSED walk consumed the payload to within the
    /// writer's byte padding.
    pub fn trustworthy(&self, header_mask: u64) -> bool {
        if self.failed {
            return false;
        }
        match self.stopped_at {
            None => self.closed && self.rebuilt_mask == header_mask && self.leftover_bits <= 7,
            Some(f) => {
                let low = (1u64 << f) - 1;
                self.rebuilt_mask & low == header_mask & low
            }
        }
    }
}

const PLAYER_GATES: [u32; 6] = [0, 2, 8, 0x11, 0x18, 0x1D];
const PLAYER_FIELDS: u32 = 0x2D;

/// Walk a type-3 `game-engine-player` message through field `0x19` (the two variable fields),
/// and on to the end of the field list when nothing past `0x19` is present.
pub fn walk_player(e: &EntityMessage, p: &DecodeParams, d: &MegaloDecls) -> Option<Walk> {
    if e.entity_type != 3 {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut w = Walk::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        let i = r.bits(5).ok()? as u8;
        if i > 0xF {
            return None;
        }
        w.owner = Some(i);
    }
    if e.dirty_mask == 0 && !e.is_update {
        w.closed = true;
        w.leftover_bits = r.remaining();
        return Some(w);
    }
    let mut g = 0usize;
    let mut field = 0u32;
    while field < PLAYER_FIELDS {
        if g < PLAYER_GATES.len() && field == PLAYER_GATES[g] {
            g += 1;
            let Ok(open) = r.flag() else { w.failed = true; return Some(w) };
            if !open {
                field = PLAYER_GATES.get(g).copied().unwrap_or(PLAYER_FIELDS);
                continue;
            }
        }
        let Ok(present) = r.flag() else {
            // The writer stops once every remaining field is absent.
            break;
        };
        let this = field;
        field += 1;
        if !present {
            continue;
        }
        w.rebuilt_mask |= 1u64 << this;
        let res: Result<(), DecodeStop> = (|| {
            match this {
                0 => {
                    r.bits(3)?;
                }
                1 => {
                    r.bits(3)?;
                    r.bits(10)?;
                }
                2 => {
                    for b in [4, 3, 4, 3, 4, 4, 2, 3, 2, 2] {
                        r.bits(b)?;
                    }
                }
                3 => {
                    for b in [4, 4, 8, 8, 4, 2, 2, 2, 2, 2, 2, 8] {
                        r.bits(b)?;
                    }
                }
                4 => {
                    for b in [5, 4, 4, 2] {
                        r.bits(b)?;
                    }
                    if r.flag()? {
                        r.bits(9)?;
                    }
                }
                5 => {
                    for b in [3, 2, 2, 3, 4] {
                        r.bits(b)?;
                    }
                }
                6 => {
                    for b in [3, 3, 2] {
                        r.bits(b)?;
                    }
                }
                7 => {
                    for b in [5, 5, 1] {
                        r.bits(b)?;
                    }
                }
                8 | 0xB | 0xC => {
                    r.bits(16)?;
                }
                9 => {
                    r.bits(10)?;
                }
                0xA => {
                    r.bits(4)?;
                }
                0xD | 0xE => {
                    r.flag()?;
                }
                0xF => {
                    r.bits(3)?;
                    r.bits(3)?;
                }
                0x10 => {
                    read_object(&mut r, false)?;
                }
                0x11 | 0x14 | 0x16 | 0x17 => {
                    r.flag()?;
                }
                0x12 => {
                    r.bits(7)?;
                }
                0x13 => {
                    r.bits(5)?;
                }
                0x15 => {
                    r.bits(19)?;
                }
                0x18 => read_pass(&mut r, &d.player, 1, p.host_authority, &mut w.vars)?,
                0x19 => read_pass(&mut r, &d.player, 2, p.host_authority, &mut w.vars)?,
                _ => return Err(DecodeStop::Overrun),
            }
            Ok(())
        })();
        if res.is_err() {
            if this >= 0x1A {
                w.stopped_at = Some(this);
            } else {
                w.failed = true;
            }
            w.leftover_bits = r.remaining();
            return Some(w);
        }
    }
    w.closed = true;
    w.leftover_bits = r.remaining();
    Some(w)
}

/// Walk a type-4 `game-engine-megalo-team` message: 3 presence flags, then the two variable
/// passes and the `coop-spawning-enabled` bit. Fully closed -- the whole entity is covered.
pub fn walk_team(e: &EntityMessage, p: &DecodeParams, d: &MegaloDecls) -> Option<Walk> {
    if e.entity_type != 4 {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut w = Walk::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    } else {
        // `sub_180458DE0`: 4 bits stored minus 1, valid <= 8.
        let t = r.bits(4).ok()? as i32 - 1;
        if !(0..=8).contains(&t) {
            return None;
        }
        w.owner = Some(t as u8);
    }
    if e.dirty_mask == 0 && !e.is_update {
        w.closed = true;
        w.leftover_bits = r.remaining();
        return Some(w);
    }
    for field in 0..3u32 {
        let Ok(present) = r.flag() else { break };
        if !present {
            continue;
        }
        w.rebuilt_mask |= 1u64 << field;
        let res = match field {
            0 => read_pass(&mut r, &d.team, 1, p.host_authority, &mut w.vars),
            1 => read_pass(&mut r, &d.team, 2, p.host_authority, &mut w.vars),
            _ => r.flag().map(|_| ()),
        };
        if res.is_err() {
            w.failed = true;
            w.leftover_bits = r.remaining();
            return Some(w);
        }
    }
    w.closed = true;
    w.leftover_bits = r.remaining();
    Some(w)
}

/// Walk a type-0/1 globals message (`sub_180466950` fields 0..11 -> `sub_1804558A0` fields
/// 12..17 [-> `sub_180455408` fields 18..21 for type 0]). Reaches the global variable fields and
/// the engine round/clock state; stops at the first HUD widget (variable-length) if one is sent.
pub fn walk_globals(e: &EntityMessage, p: &DecodeParams, d: &MegaloDecls) -> Option<Walk> {
    if e.entity_type != 0 && e.entity_type != 1 {
        return None;
    }
    let mut r = BitReader::new(&e.payload);
    let mut w = Walk::default();
    if e.is_update {
        r.skip(p.update_prefix_bits as usize).ok()?;
    }
    if e.dirty_mask == 0 && !e.is_update {
        w.closed = true;
        return Some(w);
    }
    macro_rules! tryw {
        ($e:expr, $f:expr) => {
            match $e {
                Ok(v) => v,
                Err(_) => {
                    if $f < 12 || $f >= 18 {
                        w.failed = true;
                    } else {
                        w.stopped_at = Some($f);
                    }
                    w.leftover_bits = r.remaining();
                    return Some(w);
                }
            }
        };
    }
    for field in 0..14u32 {
        let present = match r.flag() {
            Ok(b) => b,
            Err(_) => {
                w.closed = true;
                w.leftover_bits = 0;
                return Some(w);
            }
        };
        if !present {
            continue;
        }
        w.rebuilt_mask |= 1u64 << field;
        match field {
            0 => {
                tryw!(r.bits(8), field);
                tryw!(r.bits(9), field);
                let mask = tryw!(r.bits(8), field);
                tryw!(r.bits(8), field);
                tryw!(r.bits(8), field);
                for i in 0..8 {
                    if mask & (1 << i) != 0 {
                        tryw!(r.bits(4), field);
                    }
                }
            }
            1 => {
                for _ in 0..8 {
                    tryw!(r.bits(8), field);
                }
            }
            2 => {
                let v = tryw!(r.bits(3), field);
                w.game.push(("current_state", v as i64));
            }
            3 => {
                let v = tryw!(r.flag(), field);
                w.game.push(("game_finished", v as i64));
            }
            4 => {
                let none = tryw!(r.flag(), field);
                if !none {
                    let v = tryw!(r.bits(10), field);
                    w.game.push(("current_round", v as i64));
                } else {
                    w.game.push(("current_round", -1));
                }
            }
            5 | 6 | 7 => {
                let t = tryw!(read_timer(&mut r, 16), field);
                let name = match field {
                    5 => "round_timer_seconds",
                    6 => "sudden_death_seconds",
                    _ => "grace_period_seconds",
                };
                if let Value::Timer { seconds, .. } = t {
                    w.game.push((name, seconds as i64));
                }
            }
            8 => {
                let v = tryw!(r.bits(8), field);
                w.game.push(("round_condition_flags", v as i64));
            }
            9 => {
                tryw!(r.bits(3), field);
            }
            10 => {
                tryw!(r.bits(32), field);
                tryw!(r.bits(8), field);
                tryw!(r.bits(4), field);
            }
            11 => {
                tryw!(r.bits(1), field);
                tryw!(r.bits(1), field);
            }
            12 => tryw!(read_pass(&mut r, &d.global, 1, p.host_authority, &mut w.vars), field),
            _ => tryw!(read_pass(&mut r, &d.global, 2, p.host_authority, &mut w.vars), field),
        }
    }
    // HUD widgets (variable length): a clear flag costs one bit, a set one stops the walk.
    for k in 0..d.hud_widgets {
        match r.flag() {
            Ok(false) => {}
            Ok(true) => {
                w.stopped_at = Some(14 + k);
                w.leftover_bits = r.remaining();
                return Some(w);
            }
            Err(_) => {
                w.closed = true;
                return Some(w);
            }
        }
    }
    if e.entity_type == 0 {
        // The sandbox tail (fields 18..21) is not walked; a type-0 message is never CLOSED.
        w.stopped_at = Some(18);
        w.leftover_bits = r.remaining();
        return Some(w);
    }
    w.closed = true;
    w.leftover_bits = r.remaining();
    Some(w)
}

// ---------------------------------------------------------------------------------------------
// the time series
// ---------------------------------------------------------------------------------------------

/// A series key: which variable, on which owner. `owner` is the player SLOT (player scope), the
/// team index (team scope) and 0 (global).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    pub scope: Scope,
    pub owner: u8,
    pub kind: Kind,
    pub slot: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub messages: usize,
    pub trusted: usize,
    pub with_vars: usize,
    pub closed_trusted: usize,
}

#[derive(Clone, Debug, Default)]
pub struct MegaloSeries {
    pub decls: Option<MegaloDecls>,
    /// Change events per key, ascending by tick (`tick`, `value`). A value holds until the next.
    pub series: std::collections::BTreeMap<Key, Vec<(u32, Value)>>,
    /// Engine round/clock state changes: (name, [(tick, value)]).
    pub game: std::collections::BTreeMap<&'static str, Vec<(u32, i64)>>,
    pub player: Counts,
    pub team: Counts,
    pub global: Counts,
    pub last_tick: u32,
}

impl MegaloSeries {
    /// Value of `key` at `tick`: the last change at or before it. `None` before the first one
    /// (the recording had not yet been told).
    pub fn value_at(&self, key: &Key, tick: u32) -> Option<Value> {
        let s = self.series.get(key)?;
        let i = s.partition_point(|(t, _)| *t <= tick);
        if i == 0 { None } else { Some(s[i - 1].1) }
    }

    /// Every key's value at `tick` (only keys already stated).
    pub fn snapshot(&self, tick: u32) -> Vec<(Key, Value)> {
        self.series.keys().filter_map(|k| self.value_at(k, tick).map(|v| (*k, v))).collect()
    }

    pub fn total_changes(&self) -> usize {
        self.series.values().map(|s| s.len()).sum()
    }

    /// Build from a film. `roster` is not needed for decoding (slots are stated); it is used by
    /// the text exporters to name the player who held a slot at a tick.
    pub fn build(f: &crate::film::Film) -> MegaloSeries {
        let decls = decls_of(f);
        let mut out = MegaloSeries { decls: decls.clone(), ..Default::default() };
        let Some(d) = decls else { return out };
        let params = film_params(f);
        let mut slot_of: std::collections::HashMap<u32, u8> = Default::default();
        let mut team_of: std::collections::HashMap<u32, u8> = Default::default();
        let mut base_tick: Option<u32> = None;
        for (i, rec) in f.data.records.iter().enumerate() {
            let Ok(t) = crate::film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
            let b = *base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
            let tick = t.tick.wrapping_sub(b);
            out.last_tick = out.last_tick.max(tick);
            for e in &t.entities {
                match e.entity_type {
                    3 => {
                        let Some(w) = walk_player(e, &params, &d) else { continue };
                        out.player.messages += 1;
                        if !w.trustworthy(e.dirty_mask) {
                            continue;
                        }
                        out.player.trusted += 1;
                        if let Some(o) = w.owner {
                            slot_of.insert(e.net_id, o);
                        }
                        let Some(&slot) = slot_of.get(&e.net_id) else { continue };
                        if w.closed {
                            out.player.closed_trusted += 1;
                        }
                        if !w.vars.is_empty() {
                            out.player.with_vars += 1;
                        }
                        out.push(Scope::Player, slot, tick, &w);
                    }
                    4 => {
                        let Some(w) = walk_team(e, &params, &d) else { continue };
                        out.team.messages += 1;
                        if !w.trustworthy(e.dirty_mask) {
                            continue;
                        }
                        out.team.trusted += 1;
                        if let Some(o) = w.owner {
                            team_of.insert(e.net_id, o);
                        }
                        let Some(&team) = team_of.get(&e.net_id) else { continue };
                        if w.closed {
                            out.team.closed_trusted += 1;
                        }
                        if !w.vars.is_empty() {
                            out.team.with_vars += 1;
                        }
                        out.push(Scope::Team, team, tick, &w);
                    }
                    0 | 1 => {
                        let Some(w) = walk_globals(e, &params, &d) else { continue };
                        out.global.messages += 1;
                        if !w.trustworthy(e.dirty_mask) {
                            continue;
                        }
                        out.global.trusted += 1;
                        if w.closed {
                            out.global.closed_trusted += 1;
                        }
                        if !w.vars.is_empty() {
                            out.global.with_vars += 1;
                        }
                        out.push(Scope::Global, 0, tick, &w);
                    }
                    _ => {}
                }
            }
        }
        out
    }

    fn push(&mut self, scope: Scope, owner: u8, tick: u32, w: &Walk) {
        for (kind, slot, v) in &w.vars {
            let s = self.series.entry(Key { scope, owner, kind: *kind, slot: *slot }).or_default();
            if s.last().map(|(_, lv)| lv) != Some(v) {
                s.push((tick, *v));
            }
        }
        for (name, v) in &w.game {
            let s = self.game.entry(name).or_default();
            if s.last().map(|(_, lv)| *lv) != Some(*v) {
                s.push((tick, *v));
            }
        }
    }

    /// CSV of every change: `tick,seconds,scope,owner,owner_name,kind,slot,value,text`.
    /// `name_of(slot, tick)` names the player holding a slot (the time-aware roster).
    pub fn to_csv(&self, name_of: &dyn Fn(u8, u32) -> Option<String>) -> String {
        let mut rows: Vec<(u32, String)> = Vec::new();
        for (k, s) in &self.series {
            for (t, v) in s {
                let nm = if k.scope == Scope::Player { name_of(k.owner, *t).unwrap_or_default() } else { String::new() };
                rows.push((
                    *t,
                    format!(
                        "{},{:.3},{},{},{},{},{},{},{}\n",
                        t,
                        *t as f64 / 60.0,
                        k.scope.name(),
                        k.owner,
                        csv_field(&nm),
                        k.kind.name(),
                        k.slot,
                        v.number(),
                        csv_field(&v.text())
                    ),
                ));
            }
        }
        for (name, s) in &self.game {
            for (t, v) in s {
                rows.push((*t, format!("{},{:.3},game,0,,{},0,{},{}\n", t, *t as f64 / 60.0, name, v, v)));
            }
        }
        rows.sort_by_key(|r| r.0);
        let mut out = String::from("tick,seconds,scope,owner,owner_name,kind,slot,value,text\n");
        for (_, r) in rows {
            out.push_str(&r);
        }
        out
    }

    /// The whole thing as JSON: declarations, coverage counts and every series.
    pub fn to_json(&self, name_of: &dyn Fn(u8, u32) -> Option<String>) -> String {
        let mut series = Vec::new();
        for (k, s) in &self.series {
            let pts: Vec<String> = s.iter().map(|(t, v)| format!("[{},{}]", t, v.number())).collect();
            let nm = if k.scope == Scope::Player { s.first().and_then(|(t, _)| name_of(k.owner, *t)).unwrap_or_default() } else { String::new() };
            series.push(format!(
                "{{\"scope\":\"{}\",\"owner\":{},\"owner_name\":\"{}\",\"kind\":\"{}\",\"slot\":{},\"points\":[{}]}}",
                k.scope.name(),
                k.owner,
                json_escape(&nm),
                k.kind.name(),
                k.slot,
                pts.join(",")
            ));
        }
        let mut game = Vec::new();
        for (name, s) in &self.game {
            let pts: Vec<String> = s.iter().map(|(t, v)| format!("[{t},{v}]")).collect();
            game.push(format!("{{\"name\":\"{name}\",\"points\":[{}]}}", pts.join(",")));
        }
        let c = |c: &Counts| format!("{{\"messages\":{},\"trusted\":{},\"with_variables\":{},\"closed\":{}}}", c.messages, c.trusted, c.with_vars, c.closed_trusted);
        format!(
            "{{\"declarations\":{},\"coverage\":{{\"player\":{},\"team\":{},\"global\":{}}},\"ticks\":{},\"series\":[{}],\"game\":[{}],\"not_in_film\":\"{}\"}}",
            self.decls.as_ref().map_or("null".to_string(), |d| d.to_json()),
            c(&self.player),
            c(&self.team),
            c(&self.global),
            self.last_tick + 1,
            series.join(","),
            game.join(","),
            json_escape(UNRECOVERABLE)
        )
    }
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn json_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn films() -> Vec<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return vec![std::path::PathBuf::from(p)];
        }
        crate::film::spec_corpus()
    }

    #[test]
    #[ignore = "probe: megalo variable census over the corpus"]
    fn probe_megalo_census() {
        for p in films() {
            let Ok(f) = crate::film::open(&p) else { continue };
            let Some(d) = decls_of(&f) else { continue };
            let s = MegaloSeries::build(&f);
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let rep = |sc: &ScopeDecl| sc.vars.iter().filter(|v| v.replicated).count();
            eprintln!(
                "{name}: sane={} g={}/{} p={}/{} t={}/{} o={}/{} hud={} trg={} | P msgs {} ok {} closed {} vars {} | T {} {} {} {} | G {} {} {} {} | changes {}",
                d.sane,
                rep(&d.global), d.global.vars.len(), rep(&d.player), d.player.vars.len(), rep(&d.team), d.team.vars.len(), rep(&d.object), d.object.vars.len(),
                d.hud_widgets, d.triggers,
                s.player.messages, s.player.trusted, s.player.closed_trusted, s.player.with_vars,
                s.team.messages, s.team.trusted, s.team.closed_trusted, s.team.with_vars,
                s.global.messages, s.global.trusted, s.global.closed_trusted, s.global.with_vars,
                s.total_changes()
            );
        }
    }
}


#[cfg(test)]
mod gates {
    use super::*;

    /// Castle Wars (`*53C14E9D*`): a team CTF-style gametype with 5 replicated player variables,
    /// 2 team object variables and a scoreboard with two captures -- the fixture every number
    /// below is read off.
    pub(super) fn castle_wars() -> Option<crate::film::Film> {
        let p = if let Ok(p) = std::env::var("HMS_FILM") {
            std::path::PathBuf::from(p)
        } else {
            crate::film::spec_corpus().into_iter().find(|p| p.to_string_lossy().contains("53C14E9D"))?
        };
        crate::film::open(&p).ok()
    }

    /// #film-megalo  GATE. Every claim is checked against a DIFFERENT decoder of the same film:
    /// the scoreboard (statborg, entity type 5), the roster (list A kind 9), the entity tracks
    /// (creation messages), and the engine clock (60 ticks = one timer second).
    #[test]
    fn megalo_variables_agree_with_the_scoreboard_roster_and_tracks() {
        let Some(f) = castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let d = decls_of(&f).expect("megalo declarations");
        assert!(d.sane, "every declared count within capacity and every priority <= 3");
        assert_eq!(d.triggers, 70);
        let kinds = |sc: &ScopeDecl| sc.vars.iter().map(|v| (v.kind, v.slot, v.priority)).collect::<Vec<_>>();
        assert_eq!(
            kinds(&d.player),
            vec![(Kind::Number, 0, Some(1)), (Kind::Number, 1, Some(1)), (Kind::Timer, 0, None), (Kind::Timer, 1, None), (Kind::Timer, 2, None)]
        );
        assert_eq!(kinds(&d.team), vec![(Kind::Object, 0, Some(1)), (Kind::Object, 1, Some(1))]);
        assert_eq!(d.global.vars.iter().filter(|v| v.replicated && v.kind == Kind::Number).map(|v| v.slot).collect::<Vec<_>>(), vec![2]);

        let s = MegaloSeries::build(&f);
        // Every player/team/global message decodes under the engine's own two oracles.
        assert_eq!((s.player.messages, s.player.trusted), (3619, 3619));
        assert_eq!((s.team.messages, s.team.trusted), (25, 25));
        assert_eq!((s.global.messages, s.global.trusted), (508, 508));

        // (1) The engine clock: a rate-5 countdown timer ticks once per 60 ticks (+-2).
        let key = |scope, owner, kind, slot| Key { scope, owner, kind, slot };
        let t6 = s.series.get(&key(Scope::Player, 6, Kind::Timer, 2)).expect("slot 6 timer[2]");
        let secs: Vec<(u32, u16)> = t6.iter().map(|(t, v)| (*t, if let Value::Timer { seconds, .. } = v { *seconds } else { 999 })).collect();
        assert_eq!(secs, vec![(29, 0), (25006, 5), (25066, 4), (25128, 3), (25186, 2), (25245, 1), (25306, 0)]);
        for w in secs[1..].windows(2) {
            let dt = w[1].0 as i64 - w[0].0 as i64;
            assert!((58..=62).contains(&dt), "a countdown second is 60 ticks, saw {dt}");
        }

        // (2) The roster: slot 6's occupant changes at tick 25006 (list A kind 9, an independent
        // decoder) and that is the tick its variables are re-stated.
        let idx = crate::film_decode::build_index_film(&f);
        let boundary = idx.roster.seats.iter().filter(|st| st.slot == 6).any(|st| st.from_tick == 25006 || st.until_tick == Some(25006));
        assert!(boundary, "the roster has a slot-6 seat boundary at tick 25006");

        // (3) The scoreboard: the two captures (team statborg 16 stepping 7 -> 8 -> 9, then 10 at the round end) land on
        // exactly the ticks the team's flag-object variable is cleared; the globals entity's
        // `game_finished` lands on the scoreboard's `final` tick.
        let sb = &idx.statborg[16];
        let step_ticks: Vec<u32> = sb.windows(2).filter(|w| w[1].1.round_score > w[0].1.round_score).map(|w| w[1].0).collect();
        assert_eq!(step_ticks, vec![23905, 29170, 31839], "two captures, then the round-end award");
        let flag = s.series.get(&key(Scope::Team, 1, Kind::Object, 1)).expect("team 1 object[1]");
        let nulls: Vec<u32> = flag.iter().filter(|(_, v)| *v == Value::Object(None)).map(|(t, _)| *t).collect();
        assert_eq!(nulls, vec![20139, 23905, 29170, 31839]);
        for t in &step_ticks {
            assert!(nulls.contains(t), "capture at {t} must clear the flag variable");
        }
        let fin = idx.statborg[0].iter().find(|(_, x)| x.is_final == Some(true)).map(|(t, _)| *t).expect("final flag");
        assert_eq!(s.game.get("game_finished").map(|v| v.clone()), Some(vec![(fin, 1)]));
        assert_eq!(fin, 31839);

        // (4) The entity tracks: every non-null team object reference names a live entity -- the
        // flag base is a `generic` (13), the flag a `weapon` (17).
        let mut resolved = 0;
        for (k, pts) in &s.series {
            if k.scope != Scope::Team || k.kind != Kind::Object {
                continue;
            }
            for (t, v) in pts {
                if let Value::Object(Some(id)) = v {
                    let tr = idx.tracks.iter().find(|tr| tr.net_id == *id).expect("reference names a track");
                    assert_eq!(tr.entity_type, if k.slot == 0 { 13 } else { 17 }, "object[{}] at {t}", k.slot);
                    assert!(tr.first_tick <= *t);
                    resolved += 1;
                }
            }
        }
        assert_eq!(resolved, 7);
    }
}

/// Test fixture shared with `film_data`'s gates.
#[cfg(test)]
pub(crate) fn tests_castle_wars() -> Option<crate::film::Film> {
    gates::castle_wars()
}
