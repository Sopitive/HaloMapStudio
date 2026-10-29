//! #film-data  Script/CLI READ ACCESS to what a saved film decodes.
//!
//! One pure entry point, [`verb`], shared by both script hosts (`script_host.rs` headless and the
//! GUI's command server in `main.rs`), so a verb cannot exist in one and not the other. Every
//! output is machine-readable: CSV (header row first) or JSON, and the schema of each is listed
//! in [`HELP`] and in `docs/scripting.html`.
//!
//! | verb | output |
//! |------|--------|
//! | `film megalo [info]` | text summary of the gametype's Megalo script + what the film records |
//! | `film megalo decls [csv\|json]` | every declared variable slot: scope, kind, slot, priority, replicated, initial |
//! | `film megalo vars [csv\|json]` | every recorded change of every variable (the whole time series) |
//! | `film megalo at <time> [csv\|json]` | every variable's value at one moment |
//! | `film megalo get <scope> <owner> <kind> <slot> [csv\|json]` | one variable's series |
//! | `film variant [json]` | the embedded game variant: metadata + declared settings |
//! | `film events [csv\|json] [kind]` | the 72-type event channel, on the tick line |
//! | `film seats [csv\|json]` | the time-aware roster: who held which slot, when |
//! | `film score [csv\|json]` | the scoreboard store (statborg) change log |

use crate::film::Film;
use crate::film_decode::FilmIndex;
use crate::film_megalo::{self, Key, Kind, MegaloSeries, Scope};
use std::sync::{Arc, Mutex, OnceLock};

pub const HELP: &str = "film highlight [csv|json] | objects [filter...] [csv|json] | object <id> [csv|json] | inside <id|filter...> [time|a..b] [r=R|cyl=r,top,bot|box=w,l,top,bot] [occupants=players|vehicles|all] [csv|json] | near <id> r=R [time|a..b] | dump [json] | film megalo [info] | megalo decls [csv|json] | megalo vars [csv|json] | megalo at <time> [csv|json] | megalo get <scope> <owner> <kind> <slot> [csv|json] | variant [json] | events [csv|json] [kind] | seats [csv|json] | score [csv|json]\n  time = seconds (12.5), mm:ss (1:30) or ticks (t1234)\n  object filters: type= tag= label= (text|glob|#index) team= owner= slot= alive=<time> id= parent= near=x,y,z,r box=x0,y0,z0,x1,y1,z1 at=<time> megalo=1 zone=1 shape=sphere|cylinder|box ; events filters: type= player= slot= from= to= ; megalo vars filters: scope= kind= var= owner=<index|name> from= to= | schemas: objects = film_objects::OBJECT_COLUMNS | inside = tick,seconds,event,zone_id,zone_label,occupant_id,occupant_type,slot,name | megalo decls = scope,kind,slot,priority,replicated,initial | megalo vars = tick,seconds,scope,owner,owner_name,kind,slot,value,text | events = tick,seconds,kind,name,subject,other,incident,incident_name,seat | seats = slot,identifier,name,team,data_team,from_tick,until_tick,left_tick,from_header,joined_in_progress | score = tick,seconds,statborg,is_team,owner_name,round_score,total_score,rounds_won,final";

fn cache() -> &'static Mutex<Option<((usize, usize, String), Arc<MegaloSeries>)>> {
    static C: OnceLock<Mutex<Option<((usize, usize, String), Arc<MegaloSeries>)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

/// The film's Megalo series, built once per film per process (a long film is ~1 s to walk).
pub fn series_of(f: &Film) -> Arc<MegaloSeries> {
    let key = (f.bytes.len(), f.data.records.len(), f.header.session_id.clone());
    let mut g = cache().lock().unwrap();
    if let Some((k, s)) = g.as_ref() {
        if *k == key {
            return s.clone();
        }
    }
    let s = Arc::new(MegaloSeries::build(f));
    *g = Some((key, s.clone()));
    s
}

/// Parse `12.5`, `12.5s`, `1:30` or `t1234` into a tick.
pub fn parse_time(s: &str) -> Result<u32, String> {
    let bad = || format!("bad time '{s}' (seconds like 12.5, mm:ss like 1:30, or ticks like t1234)");
    if let Some(t) = s.strip_prefix('t').or_else(|| s.strip_prefix("tick")) {
        return t.parse::<u32>().map_err(|_| bad());
    }
    if let Some((m, sec)) = s.split_once(':') {
        let m: f64 = m.parse().map_err(|_| bad())?;
        let sec: f64 = sec.parse().map_err(|_| bad())?;
        return Ok(((m * 60.0 + sec) * 60.0).round().max(0.0) as u32);
    }
    let v: f64 = s.trim_end_matches('s').parse().map_err(|_| bad())?;
    Ok((v * 60.0).round().max(0.0) as u32)
}

fn name_fn<'a>(f: &'a Film, idx: Option<&'a FilmIndex>) -> impl Fn(u8, u32) -> Option<String> + 'a {
    move |slot, tick| {
        if let Some(i) = idx {
            if let Some(s) = i.roster.seat_holding(slot, tick) {
                if !s.name.is_empty() {
                    return Some(s.name.clone());
                }
            }
        }
        f.header.players.iter().find(|p| p.present() && p.index == slot as usize).map(|p| p.gamertag.clone()).filter(|g| !g.is_empty())
    }
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn jstr(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

fn want_json(args: &[String]) -> bool {
    args.iter().any(|a| a.eq_ignore_ascii_case("json"))
}

fn parse_scope(s: &str) -> Result<Scope, String> {
    match s.to_ascii_lowercase().as_str() {
        "global" | "g" => Ok(Scope::Global),
        "player" | "p" => Ok(Scope::Player),
        "team" | "t" => Ok(Scope::Team),
        "object" | "o" => Ok(Scope::Object),
        x => Err(format!("bad scope '{x}' (global|player|team|object)")),
    }
}

fn parse_kind(s: &str) -> Result<Kind, String> {
    match s.to_ascii_lowercase().as_str() {
        "number" | "num" | "n" => Ok(Kind::Number),
        "timer" | "t" => Ok(Kind::Timer),
        "team" => Ok(Kind::Team),
        "player" => Ok(Kind::Player),
        "object" | "obj" => Ok(Kind::Object),
        x => Err(format!("bad kind '{x}' (number|timer|team|player|object)")),
    }
}

/// The host-facing entry: `None` when `sub` is not one of ours; an error when it is but no film
/// is open. Both script hosts call exactly this. The hosts keep a HEADER-ONLY film open (cheap),
/// but these verbs read the recording, so the whole file is opened once per path and cached.
pub fn dispatch(path: Option<&std::path::Path>, f: Option<&Film>, idx: Option<&FilmIndex>, sub: &str, args: &[String]) -> Option<Result<String, String>> {
    if !matches!(sub, "megalo" | "variant" | "gametype" | "events" | "seats" | "score" | "scores" | "scoreboard" | "objects" | "object" | "inside" | "near" | "dump" | "highlight") {
        return None;
    }
    let Some(f) = f else { return Some(Err("film: no film open (film <path>)".into())) };
    if !f.data.records.is_empty() {
        return verb(f, idx, sub, args);
    }
    let Some(path) = path else { return Some(Err("film: the open film has no path to read the recording from".into())) };
    let full = match full_film(path) {
        Ok(f) => f,
        Err(e) => return Some(Err(e)),
    };
    verb(&full, idx, sub, args)
}

/// The whole film at `path`, opened once per path and cached (the hosts hold header-only films).
pub fn full_film(path: &std::path::Path) -> Result<Arc<Film>, String> {
    static FULL: OnceLock<Mutex<Option<(std::path::PathBuf, Arc<Film>)>>> = OnceLock::new();
    let cell = FULL.get_or_init(|| Mutex::new(None));
    let mut g = cell.lock().unwrap();
    match g.as_ref() {
        Some((p, a)) if p == path => Ok(a.clone()),
        _ => {
            let a = Arc::new(crate::film::open(path).map_err(|e| e.to_string())?);
            *g = Some((path.to_path_buf(), a.clone()));
            Ok(a)
        }
    }
}

/// Returns `None` when `sub` is not one of this module's verbs.
pub fn verb(f: &Film, idx: Option<&FilmIndex>, sub: &str, args: &[String]) -> Option<Result<String, String>> {
    match sub {
        "megalo" => Some(megalo(f, idx, args)),
        "variant" | "gametype" => Some(variant(f, args)),
        "events" => Some(events(f, idx, args)),
        "seats" => Some(seats(idx, args)),
        "score" | "scores" | "scoreboard" => Some(score(f, idx, args)),
        "objects" => Some(objects(f, idx, args)),
        "object" => Some(object(f, idx, args)),
        "inside" => Some(inside(f, idx, args, false)),
        "near" => Some(inside(f, idx, args, true)),
        "dump" => Some(dump(f, idx, args)),
        "highlight" | "megaloobjects" => Some(highlight(f, idx, args)),
        _ => None,
    }
}

fn no_megalo(f: &Film) -> String {
    format!(
        "film megalo: this film's game variant is engine type {} ({}), not a Megalo script -- it declares no variables",
        f.header.game_engine_type,
        match f.header.game_engine_type {
            1 => "sandbox / Forge",
            3 => "campaign",
            4 => "survival",
            _ => "none",
        }
    )
}

fn megalo(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("info");
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    let Some(decls) = film_megalo::decls_of(f) else { return Err(no_megalo(f)) };
    let names = name_fn(f, idx);
    match sub {
        "decls" | "declarations" | "vars-declared" => Ok(if want_json(rest) { decls.to_json() } else { decls.to_csv() }),
        "vars" | "series" | "all" => {
            let s = series_of(f);
            let flt = MegaloFilter::parse(rest)?;
            if flt.is_empty() {
                return Ok(if want_json(rest) { s.to_json(&names) } else { s.to_csv(&names) });
            }
            let fs = flt.apply(&s, &names);
            Ok(if want_json(rest) { fs.to_json(&names) } else { fs.to_csv(&names) })
        }
        "at" | "snapshot" => {
            let t = parse_time(rest.first().ok_or("film megalo at <time>  (seconds, mm:ss or t<tick>)")?)?;
            let s = series_of(f);
            let snap = s.snapshot(t);
            if want_json(rest) {
                let rows: Vec<String> = snap
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{{\"scope\":\"{}\",\"owner\":{},\"owner_name\":{},\"kind\":\"{}\",\"slot\":{},\"value\":{},\"text\":{}}}",
                            k.scope.name(),
                            k.owner,
                            jstr(&if k.scope == Scope::Player { names(k.owner, t).unwrap_or_default() } else { String::new() }),
                            k.kind.name(),
                            k.slot,
                            v.number(),
                            jstr(&v.text())
                        )
                    })
                    .collect();
                let game: Vec<String> = s
                    .game
                    .iter()
                    .filter_map(|(n, pts)| {
                        let i = pts.partition_point(|(tt, _)| *tt <= t);
                        (i > 0).then(|| format!("{{\"name\":\"{n}\",\"value\":{}}}", pts[i - 1].1))
                    })
                    .collect();
                Ok(format!("{{\"tick\":{t},\"seconds\":{:.3},\"variables\":[{}],\"game\":[{}]}}", t as f64 / 60.0, rows.join(","), game.join(",")))
            } else {
                let mut out = String::from("tick,seconds,scope,owner,owner_name,kind,slot,value,text\n");
                for (k, v) in &snap {
                    let nm = if k.scope == Scope::Player { names(k.owner, t).unwrap_or_default() } else { String::new() };
                    out.push_str(&format!(
                        "{},{:.3},{},{},{},{},{},{},{}\n",
                        t,
                        t as f64 / 60.0,
                        k.scope.name(),
                        k.owner,
                        csv_field(&nm),
                        k.kind.name(),
                        k.slot,
                        v.number(),
                        csv_field(&v.text())
                    ));
                }
                for (n, pts) in &s.game {
                    let i = pts.partition_point(|(tt, _)| *tt <= t);
                    if i > 0 {
                        out.push_str(&format!("{},{:.3},game,0,,{},0,{},{}\n", t, t as f64 / 60.0, n, pts[i - 1].1, pts[i - 1].1));
                    }
                }
                Ok(out)
            }
        }
        "get" => {
            if rest.len() < 4 {
                return Err("film megalo get <scope> <owner> <kind> <slot> [csv|json]".into());
            }
            let key = Key {
                scope: parse_scope(&rest[0])?,
                owner: rest[1].parse().map_err(|_| format!("bad owner '{}' (player slot / team index / 0)", rest[1]))?,
                kind: parse_kind(&rest[2])?,
                slot: rest[3].parse().map_err(|_| format!("bad slot '{}'", rest[3]))?,
            };
            let s = series_of(f);
            let pts = s.series.get(&key).ok_or_else(|| {
                format!("no recorded values for {}.{}[{}] of owner {} (not declared, not replicated, or never sent)", key.scope.name(), key.kind.name(), key.slot, key.owner)
            })?;
            if want_json(&rest[4..]) {
                let p: Vec<String> = pts.iter().map(|(t, v)| format!("{{\"tick\":{t},\"value\":{},\"text\":{}}}", v.number(), jstr(&v.text()))).collect();
                Ok(format!("{{\"scope\":\"{}\",\"owner\":{},\"kind\":\"{}\",\"slot\":{},\"points\":[{}]}}", key.scope.name(), key.owner, key.kind.name(), key.slot, p.join(",")))
            } else {
                let mut out = String::from("tick,seconds,value,text\n");
                for (t, v) in pts {
                    out.push_str(&format!("{},{:.3},{},{}\n", t, *t as f64 / 60.0, v.number(), csv_field(&v.text())));
                }
                Ok(out)
            }
        }
        "info" | "summary" | "" => {
            let s = series_of(f);
            let rep = |sc: Scope| decls.scope(sc).vars.iter().filter(|v| v.replicated).count();
            let mut o = String::new();
            o.push_str(&format!(
                "Megalo script: {} trigger(s), {} action record(s), {} condition record(s), {} HUD widget(s)\n",
                decls.triggers, decls.actions, decls.conditions, decls.hud_widgets
            ));
            for sc in [Scope::Global, Scope::Player, Scope::Team, Scope::Object] {
                let d = decls.scope(sc);
                o.push_str(&format!(
                    "  {:<6} declares {} number, {} timer, {} team, {} player, {} object variable(s); {} replicated{}\n",
                    sc.name(),
                    d.count(Kind::Number),
                    d.count(Kind::Timer),
                    d.count(Kind::Team),
                    d.count(Kind::Player),
                    d.count(Kind::Object),
                    rep(sc),
                    if sc == Scope::Object { " in principle -- NOT in a film (see below)" } else { "" }
                ));
            }
            o.push_str(&format!(
                "recorded: {} variable series, {} value changes over {} ticks (player msgs {}/{} decoded clean, team {}/{}, global {}/{})\n",
                s.series.len(),
                s.total_changes(),
                s.last_tick + 1,
                s.player.trusted,
                s.player.messages,
                s.team.trusted,
                s.team.messages,
                s.global.trusted,
                s.global.messages
            ));
            o.push_str(&format!("NOT in a film: {}\n", film_megalo::UNRECOVERABLE));
            Ok(o)
        }
        x => Err(format!("film megalo: unknown '{x}'. {HELP}")),
    }
}

fn variant(f: &Film, args: &[String]) -> Result<String, String> {
    let h = &f.header;
    let gt = h.gametype.as_ref();
    let m = film_megalo::decls_of(f);
    if want_json(args) {
        let mut o = format!(
            "{{\"title\":{},\"description\":{},\"author\":{},\"engine_type\":{},\"engine\":{},\"map_id\":{},\"activity\":{},\"game_mode\":{}",
            jstr(&h.game_variant.title),
            jstr(&h.game_variant.description),
            jstr(&h.game_variant.created_by),
            h.game_engine_type,
            jstr(gt.map_or("unknown", |g| g.engine_name())),
            h.game_variant.map_id,
            h.game_variant.activity,
            h.game_variant.game_mode
        );
        if let Some(g) = gt {
            o.push_str(&format!(
                ",\"teams_enabled\":{},\"rounds\":{},\"round_time_limit_min\":{},\"early_victory_wins\":{},\"sudden_death_s\":{},\"grace_period_s\":{},\"lives_per_round\":{},\"respawn_time_s\":{},\"suicide_penalty_s\":{},\"betrayal_penalty_s\":{},\"species\":{}",
                g.misc.teams_enabled(),
                g.misc.number_of_rounds,
                g.misc.round_time_limit,
                g.misc.early_victory_win_count,
                g.misc.sudden_death_limit,
                g.misc.grace_period,
                g.respawn.lives_per_round,
                g.respawn.respawn_time,
                g.respawn.suicide_penalty,
                g.respawn.betrayal_penalty,
                jstr(&format!("{:?}", g.species))
            ));
        }
        if let Some(d) = &m {
            o.push_str(&format!(",\"megalo\":{}", d.to_json()));
        }
        o.push('}');
        Ok(o)
    } else {
        let mut o = format!(
            "title        {}\nauthor       {}\nengine       {} (type {})\nmap id       {}\n",
            h.game_variant.title,
            h.game_variant.created_by,
            gt.map_or("unknown", |g| g.engine_name()),
            h.game_engine_type,
            h.game_variant.map_id
        );
        if let Some(g) = gt {
            o.push_str(&format!(
                "teams        {}\nrounds       {} (time limit {} min, early victory {} wins)\nrespawn      {} s, lives/round {}, suicide penalty {} s, betrayal penalty {} s\nspecies      {:?}\n",
                g.misc.teams_enabled(),
                g.misc.number_of_rounds,
                g.misc.round_time_limit,
                g.misc.early_victory_win_count,
                g.respawn.respawn_time,
                g.respawn.lives_per_round,
                g.respawn.suicide_penalty,
                g.respawn.betrayal_penalty,
                g.species
            ));
        }
        if let Some(d) = &m {
            o.push_str(&format!("megalo       {} trigger(s), {} variable slot(s) declared\n", d.triggers, d.global.vars.len() + d.player.vars.len() + d.team.vars.len() + d.object.vars.len()));
        }
        Ok(o)
    }
}

fn events(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film events: the recording is not decoded yet (film index, then wait)")?;
    let json = want_json(args);
    let table = table_of(f, idx);
    let mut kind_filter: Option<u8> = None;
    let (mut from, mut to) = (0u32, u32::MAX);
    let mut player: Option<String> = None;
    let mut slot: Option<u8> = None;
    for a in args.iter().filter(|a| !a.eq_ignore_ascii_case("json") && !a.eq_ignore_ascii_case("csv")) {
        let (k, v) = a.split_once('=').unwrap_or(("type", a.as_str()));
        match k {
            "type" | "kind" => {
                kind_filter = Some(match v.parse::<u8>() {
                    Ok(n) => n,
                    Err(_) => (0u8..72)
                        .find(|k| crate::film_action::event_name(*k).eq_ignore_ascii_case(v))
                        .ok_or_else(|| format!("film events: unknown event '{v}' (a kind number 0-71 or a name such as incident)"))?,
                })
            }
            "from" => from = parse_time(v)?,
            "to" => to = parse_time(v)?,
            "player" | "name" => player = Some(v.to_ascii_lowercase()),
            "slot" => slot = Some(v.parse().map_err(|_| format!("bad slot '{v}'"))?),
            _ => return Err(format!("film events: unknown filter '{a}' (type= player= slot= from= to=)")),
        }
    }
    let who = |id: Option<u32>| -> (Option<u8>, Option<String>) {
        match id.and_then(|i| table.get(i)) {
            Some(r) => (r.owner_slot, r.owner_name.clone()),
            None => (None, None),
        }
    };
    let rows: Vec<_> = idx
        .events
        .iter()
        .filter(|e| kind_filter.map_or(true, |k| e.kind == k) && e.tick >= from && e.tick <= to)
        .map(|e| (e, who(e.subject), who(e.other)))
        .filter(|(_, s, o)| {
            player.as_ref().map_or(true, |p| [s, o].iter().any(|x| x.1.as_deref().map_or(false, |n| n.to_ascii_lowercase().contains(p.as_str()))))
                && slot.map_or(true, |sl| s.0 == Some(sl) || o.0 == Some(sl))
        })
        .collect();
    if json {
        let r: Vec<String> = rows
            .iter()
            .map(|(e, s, o)| {
                format!(
                    "{{\"tick\":{},\"seconds\":{:.3},\"kind\":{},\"name\":{},\"subject\":{},\"other\":{},\"incident\":{},\"incident_name\":{},\"seat\":{},\"subject_slot\":{},\"subject_name\":{},\"other_slot\":{},\"other_name\":{}}}",
                    e.tick,
                    e.tick as f64 / 60.0,
                    e.kind,
                    jstr(crate::film_action::event_name(e.kind)),
                    e.subject.map_or("null".into(), |v| v.to_string()),
                    e.other.map_or("null".into(), |v| v.to_string()),
                    e.incident.map_or("null".into(), |v| v.to_string()),
                    e.incident.map_or("null".into(), |v| jstr(crate::film_action::incident_name(v).unwrap_or(""))),
                    e.seat.map_or("null".into(), |v| v.to_string()),
                    s.0.map_or("null".into(), |v| v.to_string()),
                    s.1.as_deref().map_or("null".into(), jstr),
                    o.0.map_or("null".into(), |v| v.to_string()),
                    o.1.as_deref().map_or("null".into(), jstr)
                )
            })
            .collect();
        Ok(format!("[{}]", r.join(",")))
    } else {
        let mut o = String::from("tick,seconds,kind,name,subject,other,incident,incident_name,seat,subject_slot,subject_name,other_slot,other_name\n");
        for (e, s, ot) in rows {
            o.push_str(&format!(
                "{},{:.3},{},{},{},{},{},{},{},{},{},{},{}\n",
                e.tick,
                e.tick as f64 / 60.0,
                e.kind,
                crate::film_action::event_name(e.kind),
                e.subject.map_or(String::new(), |v| v.to_string()),
                e.other.map_or(String::new(), |v| v.to_string()),
                e.incident.map_or(String::new(), |v| v.to_string()),
                e.incident.map_or(String::new(), |v| csv_field(crate::film_action::incident_name(v).unwrap_or(""))),
                e.seat.map_or(String::new(), |v| v.to_string()),
                s.0.map_or(String::new(), |v| v.to_string()),
                csv_field(s.1.as_deref().unwrap_or("")),
                ot.0.map_or(String::new(), |v| v.to_string()),
                csv_field(ot.1.as_deref().unwrap_or(""))
            ));
        }
        Ok(o)
    }
}

fn seats(idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film seats: the recording is not decoded yet (film index, then wait)")?;
    let seats = &idx.roster.seats;
    if want_json(args) {
        let r: Vec<String> = seats
            .iter()
            .map(|s| {
                format!(
                    "{{\"slot\":{},\"identifier\":\"{:016X}\",\"name\":{},\"team\":{},\"data_team\":{},\"from_tick\":{},\"until_tick\":{},\"left_tick\":{},\"from_header\":{},\"joined_in_progress\":{}}}",
                    s.slot,
                    s.identifier,
                    jstr(&s.name),
                    s.team,
                    s.data_team,
                    s.from_tick,
                    s.until_tick.map_or("null".into(), |v| v.to_string()),
                    s.left_tick.map_or("null".into(), |v| v.to_string()),
                    s.from_header,
                    s.joined_in_progress
                )
            })
            .collect();
        Ok(format!("[{}]", r.join(",")))
    } else {
        let mut o = String::from("slot,identifier,name,team,data_team,from_tick,until_tick,left_tick,from_header,joined_in_progress\n");
        for s in seats {
            o.push_str(&format!(
                "{},{:016X},{},{},{},{},{},{},{},{}\n",
                s.slot,
                s.identifier,
                csv_field(&s.name),
                s.team,
                s.data_team,
                s.from_tick,
                s.until_tick.map_or(String::new(), |v| v.to_string()),
                s.left_tick.map_or(String::new(), |v| v.to_string()),
                s.from_header,
                s.joined_in_progress
            ));
        }
        Ok(o)
    }
}

fn score(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film score: the recording is not decoded yet (film index, then wait)")?;
    let names = name_fn(f, Some(idx));
    let mut rows: Vec<(u32, u8, crate::film_decode::StatborgSample)> = Vec::new();
    for (slot, v) in idx.statborg.iter().enumerate() {
        for (t, s) in v {
            rows.push((*t, slot as u8, *s));
        }
    }
    rows.sort_by_key(|r| (r.0, r.1));
    let opt = |v: Option<i16>| v.map_or(String::new(), |x| x.to_string());
    if want_json(args) {
        let r: Vec<String> = rows
            .iter()
            .map(|(t, slot, s)| {
                let team = *slot >= crate::film_decode::STATBORG_PLAYERS;
                format!(
                    "{{\"tick\":{t},\"seconds\":{:.3},\"statborg\":{slot},\"is_team\":{team},\"owner_name\":{},\"round_score\":{},\"total_score\":{},\"rounds_won\":{},\"final\":{}}}",
                    *t as f64 / 60.0,
                    jstr(&if team { String::new() } else { names(*slot, *t).unwrap_or_default() }),
                    s.round_score.map_or("null".into(), |v| v.to_string()),
                    s.total_score.map_or("null".into(), |v| v.to_string()),
                    s.rounds_won.map_or("null".into(), |v| v.to_string()),
                    s.is_final.map_or("null".into(), |v| v.to_string())
                )
            })
            .collect();
        Ok(format!("[{}]", r.join(",")))
    } else {
        let mut o = String::from("tick,seconds,statborg,is_team,owner_name,round_score,total_score,rounds_won,final\n");
        for (t, slot, s) in rows {
            let team = slot >= crate::film_decode::STATBORG_PLAYERS;
            o.push_str(&format!(
                "{},{:.3},{},{},{},{},{},{},{}\n",
                t,
                t as f64 / 60.0,
                slot,
                team,
                csv_field(&if team { String::new() } else { names(slot, t).unwrap_or_default() }),
                opt(s.round_score),
                opt(s.total_score),
                opt(s.rounds_won),
                s.is_final.map_or(String::new(), |v| v.to_string())
            ));
        }
        Ok(o)
    }
}

/// Filters for `film megalo vars`: `scope=` `kind=` `var=<slot>` `owner=<index|name substring>`
/// `from=` `to=`. A name filter resolves through the time-aware roster, so it follows the PLAYER
/// across slot changes, not the slot.
#[derive(Default)]
struct MegaloFilter {
    scope: Option<Scope>,
    kind: Option<Kind>,
    var: Option<u8>,
    owner_idx: Option<u8>,
    owner_name: Option<String>,
    from: Option<u32>,
    to: Option<u32>,
}

impl MegaloFilter {
    fn parse(args: &[String]) -> Result<MegaloFilter, String> {
        let mut m = MegaloFilter::default();
        for a in args.iter().filter(|a| a.contains('=')) {
            let (k, v) = a.split_once('=').unwrap();
            match k {
                "scope" => m.scope = Some(parse_scope(v)?),
                "kind" => m.kind = Some(parse_kind(v)?),
                "var" | "slot" => m.var = Some(v.parse().map_err(|_| format!("bad var '{v}'"))?),
                "owner" | "player" => match v.parse::<u8>() {
                    Ok(n) => m.owner_idx = Some(n),
                    Err(_) => m.owner_name = Some(v.to_ascii_lowercase()),
                },
                "from" => m.from = Some(parse_time(v)?),
                "to" => m.to = Some(parse_time(v)?),
                _ => return Err(format!("film megalo vars: unknown filter '{a}' (scope= kind= var= owner= from= to=)")),
            }
        }
        Ok(m)
    }
    fn is_empty(&self) -> bool {
        self.scope.is_none() && self.kind.is_none() && self.var.is_none() && self.owner_idx.is_none() && self.owner_name.is_none() && self.from.is_none() && self.to.is_none()
    }
    fn apply(&self, s: &MegaloSeries, names: &dyn Fn(u8, u32) -> Option<String>) -> MegaloSeries {
        let mut out = MegaloSeries { decls: s.decls.clone(), last_tick: s.last_tick, ..Default::default() };
        for (k, pts) in &s.series {
            if self.scope.map_or(false, |x| x != k.scope) || self.kind.map_or(false, |x| x != k.kind) || self.var.map_or(false, |x| x != k.slot) {
                continue;
            }
            if self.owner_idx.map_or(false, |x| x != k.owner) {
                continue;
            }
            let kept: Vec<(u32, film_megalo::Value)> = pts
                .iter()
                .filter(|(t, _)| self.from.map_or(true, |a| *t >= a) && self.to.map_or(true, |b| *t <= b))
                .filter(|(t, _)| {
                    self.owner_name.as_ref().map_or(true, |n| {
                        k.scope == Scope::Player && names(k.owner, *t).map_or(false, |nm| nm.to_ascii_lowercase().contains(n.as_str()))
                    })
                })
                .cloned()
                .collect();
            if !kept.is_empty() {
                out.series.insert(*k, kept);
            }
        }
        out
    }
}

fn table_cache() -> &'static Mutex<Option<((usize, usize, usize), Arc<crate::film_objects::ObjectTable>)>> {
    static C: OnceLock<Mutex<Option<((usize, usize, usize), Arc<crate::film_objects::ObjectTable>)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

/// The object table for a film + its index, built once per film per process.
pub fn table_of(f: &Film, idx: &FilmIndex) -> Arc<crate::film_objects::ObjectTable> {
    let key = (f.bytes.len(), f.data.records.len(), idx.tracks.len());
    let mut g = table_cache().lock().unwrap();
    if let Some((k, t)) = g.as_ref() {
        if *k == key {
            return t.clone();
        }
    }
    let m = if film_megalo::decls_of(f).is_some() { Some(series_of(f)) } else { None };
    let t = Arc::new(crate::film_objects::ObjectTable::build(f, idx, None, m.as_deref()));
    *g = Some((key, t.clone()));
    t
}

fn track_of<'a>(idx: &'a FilmIndex, id: u32) -> Option<&'a crate::film_decode::EntityTrack> {
    idx.tracks.iter().find(|t| t.net_id == id)
}

fn objects(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film objects: the recording is not decoded yet (film index, then wait)")?;
    let json = want_json(args);
    let rest: Vec<String> = args.iter().filter(|a| !a.eq_ignore_ascii_case("json") && !a.eq_ignore_ascii_case("csv")).cloned().collect();
    let flt = crate::film_objects::Filter::parse(&rest)?;
    let table = table_of(f, idx);
    let by_id: std::collections::HashMap<u32, &crate::film_decode::EntityTrack> = idx.tracks.iter().map(|t| (t.net_id, t)).collect();
    let at = flt.at.or_else(|| flt.preds.iter().find_map(|p| if let crate::film_objects::Pred::Alive(t) = p { Some(*t) } else { None }));
    let rows: Vec<&crate::film_objects::ObjRow> = table.rows.iter().filter(|r| flt.matches(r, by_id.get(&r.net_id).copied())).collect();
    if json {
        let v: Vec<serde_json::Value> = rows.iter().map(|r| crate::film_objects::row_json(r, by_id.get(&r.net_id).copied(), at)).collect();
        Ok(serde_json::to_string(&v).unwrap_or_default())
    } else {
        let mut o = String::from(crate::film_objects::OBJECT_COLUMNS);
        o.push('\n');
        for r in rows {
            o.push_str(&crate::film_objects::row_csv(r, by_id.get(&r.net_id).copied(), at));
            o.push('\n');
        }
        Ok(o)
    }
}

fn object(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film object: the recording is not decoded yet (film index, then wait)")?;
    let id_s = args.iter().find(|a| !a.eq_ignore_ascii_case("json") && !a.eq_ignore_ascii_case("csv")).ok_or("film object <id> [csv|json]  (id = 0x-prefixed hex or decimal, see `film objects`)")?;
    let id = parse_obj_id(id_s)?;
    let table = table_of(f, idx);
    let row = table.get(id).ok_or_else(|| format!("film object: no entity 0x{id:08X} in this recording"))?;
    let tr = track_of(idx, id).ok_or("film object: no track")?;
    let mut ev: Vec<(u32, &'static str, String)> = Vec::new();
    ev.push((row.first_tick, "create", format!("type={} def_wire={}", row.type_name(), row.def_wire.map_or(String::new(), |w| w.to_string()))));
    for (t, p) in &tr.positions {
        ev.push((*t, "position", format!("{:.4},{:.4},{:.4}", p[0], p[1], p[2])));
    }
    for (t, fw, up) in &tr.orientations {
        ev.push((*t, "orientation", format!("fwd={:.4}/{:.4}/{:.4};up={:.4}/{:.4}/{:.4}", fw[0], fw[1], fw[2], up[0], up[1], up[2])));
    }
    for (t, m) in idx.mp_props.get(&id).map(|v| v.as_slice()).unwrap_or(&[]) {
        ev.push((*t, "mp", format!("flags={};team={};shape={};dims={:.3}/{:.3}/{:.3}/{:.3}", m.flags, m.team, crate::film_objects::shape_name(m.shape), m.dims[0], m.dims[1], m.dims[2], m.dims[3])));
    }
    for (t, a) in &tr.attachments {
        ev.push((*t, "attach", a.map_or("none".to_string(), |a| format!("parent={};node={}", a.parent, a.node))));
    }
    if let Some(d) = tr.destroyed_tick {
        ev.push((d, "destroy", String::new()));
    }
    ev.sort_by_key(|e| e.0);
    if want_json(args) {
        let pts: Vec<serde_json::Value> = ev.iter().map(|(t, w, d)| serde_json::json!({"tick": t, "seconds": *t as f64 / 60.0, "what": w, "detail": d})).collect();
        Ok(serde_json::json!({"object": crate::film_objects::row_json(row, Some(tr), None), "series": pts}).to_string())
    } else {
        let mut o = String::from("tick,seconds,what,detail\n");
        for (t, w, d) in ev {
            o.push_str(&format!("{},{:.3},{},{}\n", t, t as f64 / 60.0, w, csv_field(&d)));
        }
        Ok(o)
    }
}

pub fn parse_obj_id(s: &str) -> Result<u32, String> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u32::from_str_radix(h, 16),
        None => s.parse::<u32>(),
    }
    .map_err(|_| format!("bad object id '{s}' (0x-prefixed hex or decimal)"))
}

/// `film inside` / `film near`: occupancy of a zone at a tick, or ENTER/EXIT over a range.
fn inside(f: &Film, idx: Option<&FilmIndex>, args: &[String], near: bool) -> Result<String, String> {
    use crate::film_objects as fo;
    let idx = idx.ok_or("film inside: the recording is not decoded yet (film index, then wait)")?;
    let json = want_json(args);
    let table = table_of(f, idx);
    let by_id: std::collections::HashMap<u32, &crate::film_decode::EntityTrack> = idx.tracks.iter().map(|t| (t.net_id, t)).collect();
    let mut zone_filter: Vec<String> = Vec::new();
    let mut when: Option<(u32, u32)> = None;
    let mut single: Option<u32> = None;
    let mut over: Option<fo::Volume> = None;
    let mut over_spec: Option<(u8, [f32; 4])> = None;
    let mut occ_kind = "players";
    let mut id_target: Option<u32> = None;
    for a in args.iter().filter(|a| !a.eq_ignore_ascii_case("json") && !a.eq_ignore_ascii_case("csv")) {
        if let Some(v) = a.strip_prefix("r=") {
            over_spec = Some((1, [v.parse().map_err(|_| format!("bad radius '{v}'"))?, 0.0, 0.0, 0.0]));
        } else if let Some(v) = a.strip_prefix("cyl=") {
            let n: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            if n.len() != 3 {
                return Err("cyl=radius,top,bottom".into());
            }
            over_spec = Some((2, [n[0], 0.0, n[1], n[2]]));
        } else if let Some(v) = a.strip_prefix("box=") {
            let n: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            if n.len() != 4 {
                return Err("box=width,length,top,bottom".into());
            }
            over_spec = Some((3, [n[0], n[1], n[2], n[3]]));
        } else if let Some(v) = a.strip_prefix("occupants=") {
            occ_kind = match v {
                "players" | "bodies" => "players",
                "vehicles" => "vehicles",
                "all" => "all",
                _ => return Err("occupants=players|vehicles|all".into()),
            };
        } else if let Some((a0, b0)) = a.split_once("..") {
            when = Some((parse_time(a0)?, parse_time(b0)?));
        } else if a.contains('=') {
            zone_filter.push(a.clone());
        } else if a.starts_with("0x") || a.starts_with("0X") || a.chars().all(|c| c.is_ascii_digit()) && a.len() > 6 {
            id_target = Some(parse_obj_id(a)?);
        } else {
            single = Some(parse_time(a)?);
        }
    }
    let _ = &mut over;
    if near && over_spec.map_or(true, |s| s.0 != 1) {
        return Err("film near <id> r=<radius> [time|a..b]".into());
    }
    let flt = fo::Filter::parse(&zone_filter)?;
    let zones: Vec<&fo::ObjRow> = table
        .rows
        .iter()
        .filter(|r| id_target.map_or(true, |i| r.net_id == i) && (id_target.is_some() || !zone_filter.is_empty()))
        .filter(|r| flt.matches(r, by_id.get(&r.net_id).copied()))
        .filter(|r| over_spec.is_some() || r.boundary().is_some())
        .collect();
    if zones.is_empty() {
        return Err("film inside: no zone matched (give an object id, or filters such as label=NAME; an object with no boundary needs r= / cyl= / box=)".into());
    }
    let occ: Vec<(&fo::ObjRow, &crate::film_decode::EntityTrack)> = table
        .rows
        .iter()
        .filter(|r| match occ_kind {
            "players" => r.entity_type == 11,
            "vehicles" => r.entity_type == 15,
            _ => matches!(r.entity_type, 11 | 15 | 12 | 13 | 17 | 18),
        })
        .filter_map(|r| by_id.get(&r.net_id).map(|t| (r, *t)))
        .collect();
    let (lo, hi) = when.unwrap_or((single.unwrap_or(0), single.unwrap_or(idx.ticks.saturating_sub(1))));
    let mut rows: Vec<(u32, String, &fo::ObjRow, u32)> = Vec::new(); // tick, event, zone, who
    for z in &zones {
        let tr = by_id[&z.net_id];
        let vol_at = |t: u32| -> Option<fo::Volume> {
            if !fo::occ_alive(z, t) && z.first_tick > t {
                return None;
            }
            if z.destroyed_tick.map_or(false, |d| t >= d) {
                return None;
            }
            match over_spec {
                Some((shape, dims)) => {
                    let c = fo::pos_at(tr, t)?;
                    let (fw, up) = fo::basis_at(tr, t);
                    Some(fo::Volume { center: c, forward: fw, up, shape, dims })
                }
                None => fo::volume_of(z, tr, t),
            }
        };
        if single.is_some() && when.is_none() {
            if let Some(v) = vol_at(lo) {
                for w in fo::inside_ids(&v, &occ, lo) {
                    rows.push((lo, "INSIDE".into(), z, w));
                }
            }
        } else {
            for tr_ in fo::transitions(z.net_id, &vol_at, &occ, lo, hi) {
                rows.push((tr_.tick, if tr_.enter { "ENTER" } else { "EXIT" }.into(), z, tr_.who));
            }
        }
    }
    rows.sort_by_key(|r| (r.0, r.2.net_id, r.3));
    let line = |t: u32, ev: &str, z: &fo::ObjRow, w: u32| -> (String, serde_json::Value) {
        let o = table.get(w);
        let (ty, slot, name) = o.map_or(("?", None, None), |o| (o.type_name(), o.owner_slot, o.owner_name.clone()));
        let csv = format!(
            "{},{:.3},{},{},{},{},{},{},{}",
            t,
            t as f64 / 60.0,
            ev,
            z.net_id,
            csv_field(z.label().unwrap_or("")),
            w,
            ty,
            slot.map_or(String::new(), |s| s.to_string()),
            csv_field(name.as_deref().unwrap_or(""))
        );
        let js = serde_json::json!({"tick": t, "seconds": t as f64 / 60.0, "event": ev, "zone_id": z.net_id, "zone_label": z.label(), "occupant_id": w, "occupant_type": ty, "slot": slot, "name": name});
        (csv, js)
    };
    if json {
        let v: Vec<serde_json::Value> = rows.iter().map(|(t, e, z, w)| line(*t, e, z, *w).1).collect();
        Ok(serde_json::to_string(&v).unwrap_or_default())
    } else {
        let mut o = String::from("tick,seconds,event,zone_id,zone_label,occupant_id,occupant_type,slot,name\n");
        for (t, e, z, w) in rows {
            o.push_str(&line(t, &e, z, w).0);
            o.push('\n');
        }
        Ok(o)
    }
}

/// `film dump [json]`: one JSON document with everything the other verbs expose.
fn dump(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let _ = args;
    let idx = idx.ok_or("film dump: the recording is not decoded yet (film index, then wait)")?;
    let sub = |s: &str, a: &[&str]| -> serde_json::Value {
        let a: Vec<String> = a.iter().map(|x| x.to_string()).collect();
        verb(f, Some(idx), s, &a)
            .and_then(|r| r.ok())
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .unwrap_or(serde_json::Value::Null)
    };
    Ok(serde_json::json!({
        "schema": "hms-film-dump/1",
        "ticks": idx.ticks,
        "variant": sub("variant", &["json"]),
        "megalo": if film_megalo::decls_of(f).is_some() { sub("megalo", &["vars", "json"]) } else { serde_json::Value::Null },
        "seats": sub("seats", &["json"]),
        "events": sub("events", &["json"]),
        "score": sub("score", &["json"]),
        "objects": sub("objects", &["json"]),
    })
    .to_string())
}

/// `film highlight [csv|json]`: the objects the Theater highlight would outline, with category.
/// Columns: `id,id_hex,category,name,tag,label,team,entity_type,first_tick,destroyed_tick,shape,w_or_r,length,top,bottom`.
fn highlight(f: &Film, idx: Option<&FilmIndex>, args: &[String]) -> Result<String, String> {
    let idx = idx.ok_or("film highlight: the recording is not decoded yet (film index, then wait)")?;
    let m = if film_megalo::decls_of(f).is_some() { Some(series_of(f)) } else { None };
    let t = crate::film_highlight::build(f, idx, &Default::default(), m.as_deref(), &|_| false, &|_, leaf| leaf.to_ascii_lowercase().contains("invisible"));
    if want_json(args) {
        let v: Vec<serde_json::Value> = t
            .items
            .iter()
            .map(|i| serde_json::json!({"id": i.net_id, "category": i.cat.name(), "name": i.name, "label": i.label, "team": i.team, "entity_type": i.entity_type, "first_tick": i.first_tick, "destroyed_tick": i.destroyed_tick, "zone": i.zone.map(|z| serde_json::json!({"shape": crate::film_objects::shape_name(z.0), "dims": z.1}))}))
            .collect();
        return Ok(serde_json::to_string(&v).unwrap_or_default());
    }
    let mut o = String::from("id,id_hex,category,name,label,team,entity_type,first_tick,destroyed_tick,shape,w_or_r,length,top,bottom\n");
    for i in &t.items {
        let z = i.zone.filter(|z| z.0 != 0);
        o.push_str(&format!(
            "{},0x{:08X},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            i.net_id,
            i.net_id,
            i.cat.name(),
            csv_field(&i.name),
            csv_field(i.label.as_deref().unwrap_or("")),
            i.team,
            i.entity_type,
            i.first_tick,
            i.destroyed_tick.map_or(String::new(), |d| d.to_string()),
            z.map_or(String::new(), |z| crate::film_objects::shape_name(z.0).to_string()),
            z.map_or(String::new(), |z| format!("{:.3}", z.1[0])),
            z.map_or(String::new(), |z| format!("{:.3}", z.1[1])),
            z.map_or(String::new(), |z| format!("{:.3}", z.1[2])),
            z.map_or(String::new(), |z| format!("{:.3}", z.1[3])),
        ));
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: &Film, idx: &FilmIndex, sub: &str, args: &[&str]) -> String {
        let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        verb(f, Some(idx), sub, &a).expect("a known verb").expect("verb succeeds")
    }

    /// #film-data  GATE for the megalo verb group, on the Castle Wars fixture. Expected values are
    /// the film's own (the flag object's references and the countdown), and the name column is
    /// checked against the roster rather than typed in.
    #[test]
    fn megalo_verbs_return_the_recorded_values() {
        let Some(f) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        // `get`: one variable's whole series, exact.
        assert_eq!(
            run(&f, &idx, "megalo", &["get", "team", "1", "object", "1"]),
            "tick,seconds,value,text\n29,0.483,3922133076,0xE9C70054\n20139,335.650,-1,null\n20151,335.850,4013686891,0xEF3C006B\n23905,398.417,-1,null\n23918,398.633,4026467024,0xEFFF02D0\n29170,486.167,-1,null\n29183,486.383,4041539695,0xF0E5006F\n31839,530.650,-1,null\n"
        );
        // `at`: slot 6's countdown mid-way (25066 -> 4 s), named by the roster at that tick.
        let at = run(&f, &idx, "megalo", &["at", "t25100"]);
        let name = idx.roster.seat_holding(6, 25100).map(|s| s.name.clone()).expect("a seat holds slot 6");
        assert!(!name.is_empty());
        assert!(at.lines().any(|l| l == format!("25100,418.333,player,6,{name},timer,2,4,4s rate5 edge")), "{at}");
        assert!(at.lines().any(|l| l == "25100,418.333,game,0,,round_timer_seconds,0,8159,8159" || l.starts_with("25100,418.333,game,0,,round_timer_seconds")));
        // `vars`: CSV and JSON carry the same number of changes; `decls` states the priorities.
        let csv = run(&f, &idx, "megalo", &["vars"]);
        let json: serde_json::Value = serde_json::from_str(&run(&f, &idx, "megalo", &["vars", "json"])).expect("json parses");
        let pts: usize = json["series"].as_array().unwrap().iter().map(|s| s["points"].as_array().unwrap().len()).sum();
        let game: usize = json["game"].as_array().unwrap().iter().map(|s| s["points"].as_array().unwrap().len()).sum();
        assert_eq!(csv.lines().count() - 1, pts + game);
        assert_eq!(pts, 395);
        let decls = run(&f, &idx, "megalo", &["decls"]);
        assert!(decls.lines().any(|l| l == "player,number,0,low,true,0"));
        assert!(decls.lines().any(|l| l == "global,number,0,none,false,0"));
        assert!(decls.lines().any(|l| l == "object,number,0,low,false,0"), "object scope is declared but flagged not-replicated");
        // errors are specific
        let e = verb(&f, Some(&idx), "megalo", &["get".into(), "player".into(), "1".into(), "number".into(), "7".into()]).unwrap().unwrap_err();
        assert!(e.contains("no recorded values"), "{e}");
    }

    /// #film-data  GATE for the data verbs (variant / events / seats / score): each one must
    /// agree with the structure it was read from, row for row.
    #[test]
    fn data_verbs_match_the_index_they_read() {
        let Some(f) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let v: serde_json::Value = serde_json::from_str(&run(&f, &idx, "variant", &["json"])).unwrap();
        assert_eq!(v["engine"], "megalo");
        assert_eq!(v["engine_type"], 2);
        assert_eq!(v["teams_enabled"], true);
        assert_eq!(v["megalo"]["variables"].as_array().unwrap().len(), 10 + 5 + 2 + 9);
        // events: CSV rows == index events; the kind filter is exact; names come from the table.
        let all = run(&f, &idx, "events", &[]);
        assert_eq!(all.lines().count() - 1, idx.events.len());
        let inc = run(&f, &idx, "events", &["incident"]);
        assert_eq!(inc.lines().count() - 1, idx.events.iter().filter(|e| e.kind == 14).count());
        assert!(inc.lines().skip(1).all(|l| l.split(',').nth(3) == Some("incident")));
        assert!(inc.lines().nth(1).unwrap().starts_with("170,2.833,14,incident,3945267637,3918200856,7864343,splatter_kill,"));
        // seats: slot 6 is re-seated at tick 25006 -- two consecutive segments meet there.
        let seats = run(&f, &idx, "seats", &[]);
        let s6: Vec<Vec<&str>> = seats.lines().skip(1).map(|l| l.split(',').collect::<Vec<_>>()).filter(|c| c[0] == "6").collect();
        assert!(s6.iter().any(|c| c[6] == "25006"), "a slot-6 seat ends at 25006: {s6:?}");
        assert!(s6.iter().any(|c| c[5] == "25006"), "a slot-6 seat begins at 25006: {s6:?}");
        assert_eq!(seats.lines().count() - 1, idx.roster.seats.len());
        // score: the two captures, exact.
        let score = run(&f, &idx, "score", &[]);
        assert!(score.lines().any(|l| l.starts_with("23905,398.417,16,true,,8,")));
        assert!(score.lines().any(|l| l.starts_with("29170,486.167,16,true,,9,")));
        assert!(score.lines().any(|l| l.starts_with("31839,530.650,0,false,") && l.ends_with(",true")));
        let sj: serde_json::Value = serde_json::from_str(&run(&f, &idx, "score", &["json"])).unwrap();
        assert_eq!(sj.as_array().unwrap().len(), score.lines().count() - 1);
        // unknown verbs are not ours; a missing film is an error
        assert!(dispatch(None, Some(&f), Some(&idx), "records", &[]).is_none());
        assert!(dispatch(None, None, None, "megalo", &[]).unwrap().is_err());
        // a HEADER-ONLY film (what the hosts hold open) is re-opened in full from its path
        let p = std::env::var("HMS_FILM").map(std::path::PathBuf::from).ok().or_else(|| crate::film::spec_corpus().into_iter().find(|p| p.to_string_lossy().contains("53C14E9D"))).unwrap();
        let hdr = crate::film::open_header(&p).unwrap();
        assert!(hdr.data.records.is_empty());
        let via = dispatch(Some(&p), Some(&hdr), Some(&idx), "megalo", &["get".into(), "team".into(), "1".into(), "object".into(), "1".into()]).unwrap().unwrap();
        assert_eq!(via.lines().count(), 9);
    }

    /// #film-data  GATE for the object / zone / filter verb group (Castle Wars). The replicated
    /// boundary must equal the Forge placement's on every object that has both (an independent
    /// decode of the same box), and the zone verb must put the capturing player inside a flag base
    /// on the scoreboard's capture ticks.
    #[test]
    fn object_zone_and_filter_verbs_return_the_recorded_values() {
        let Some(f) = crate::film_megalo::tests_castle_wars() else {
            eprintln!("SKIPPED: Castle Wars fixture (*53C14E9D*) not on this machine");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let table = table_of(&f, &idx);
        // replicated multiplayer properties vs the variant's placement record: two decoders, one answer
        let (mut agree, mut disagree) = (0, 0);
        for r in &table.rows {
            if let (Some(m), Some(p)) = (r.mp, &r.placed) {
                if m.shape == p.shape && (0..4).all(|i| (m.dims[i] - p.dims[i]).abs() < 0.01) { agree += 1 } else { disagree += 1 }
            }
        }
        assert_eq!((agree, disagree), (569, 0));
        // labels come from the variant by position: the two flag bases
        let csv = run(&f, &idx, "objects", &["label=ctf_flag_return"]);
        let rows: Vec<Vec<&str>> = csv.lines().skip(1).map(|l| l.split(',').collect()).collect();
        assert_eq!(rows.len(), 2, "{csv}");
        let col = |name: &str| crate::film_objects::OBJECT_COLUMNS.split(',').position(|c| c == name).unwrap();
        assert_eq!(rows[0][col("id")], "3943104916");
        assert_eq!(rows[1][col("id")], "3943170453");
        assert_eq!(rows[0][col("id_hex")], "0xEB070194");
        assert_eq!(rows[0][col("label")], "ctf_flag_return");
        assert_eq!(rows[0][col("type")], "generic");
        assert!(rows.iter().all(|r| r[col("join_dist")].parse::<f32>().unwrap() < 0.05));
        // filters compose: zone objects only / by type / by region
        let all = run(&f, &idx, "objects", &[]).lines().count() - 1;
        assert_eq!(all, table.rows.len());
        let zones = run(&f, &idx, "objects", &["zone=1"]).lines().count() - 1;
        assert_eq!(zones, table.rows.iter().filter(|r| r.boundary().is_some()).count());
        assert!(zones > 0 && zones < all);
        let near = run(&f, &idx, "objects", &["near=85.829,149.443,40.4,1", "type=generic", "label=ctf*"]);
        assert_eq!(near.lines().count() - 1, 1);
        assert!(verb(&f, Some(&idx), "objects", &["bogus=1".into()]).unwrap().is_err());
        // object series
        let series = run(&f, &idx, "object", &["3943104916"]);
        assert!(series.lines().nth(1).unwrap().starts_with("9,0.150,create,type=generic"), "{series}");
        // zone occupancy at the capture ticks: the capturing body (0xEF380005, slot 4) is inside base 0
        let at = run(&f, &idx, "inside", &["3943104916", "r=3", "t23905"]);
        let slot4: Vec<&str> = at.lines().skip(1).filter(|l| l.split(',').nth(7) == Some("4")).collect();
        assert_eq!(slot4.len(), 1, "{at}");
        assert!(slot4[0].starts_with("23905,398.417,INSIDE,3943104916,ctf_flag_return,4013424645,biped,4,"), "{}", slot4[0]);
        let seat = idx.roster.seat_holding(4, 23905).unwrap().name.clone();
        assert!(slot4[0].ends_with(&format!(",{seat}")));
        // ...and ENTER/EXIT over the window around it
        let win = run(&f, &idx, "inside", &["3943104916", "r=3", "t23000..t24100"]);
        assert!(win.lines().any(|l| l.starts_with("23702,") && l.contains(",ENTER,3943104916,") && l.contains(",4013424645,biped,4,")));
        assert!(win.lines().any(|l| l.starts_with("24006,") && l.contains(",EXIT,3943104916,") && l.contains(",4013424645,biped,4,")));
        // events: filter by player, type and time range; columns name the players
        let ev = run(&f, &idx, "events", &["type=incident", "slot=4"]);
        let manual = idx
            .events
            .iter()
            .filter(|e| e.kind == 14)
            .filter(|e| [e.subject, e.other].iter().any(|o| o.and_then(|i| table.get(i)).map_or(false, |r| r.owner_slot == Some(4))))
            .count();
        assert_eq!(ev.lines().count() - 1, manual);
        assert!(manual > 0);
        let windowed = run(&f, &idx, "events", &["type=incident", "from=t100", "to=t2000"]);
        assert!(windowed.lines().skip(1).all(|l| { let t: u32 = l.split(',').next().unwrap().parse().unwrap(); (100..=2000).contains(&t) }));
        // megalo variables by owner: index and by NAME (which follows the player across the re-seat at 25006)
        let by_idx = run(&f, &idx, "megalo", &["vars", "owner=6", "kind=timer", "var=2"]);
        assert_eq!(by_idx.lines().count() - 1, 7);
        let name = idx.roster.seat_holding(6, 25100).unwrap().name.to_ascii_lowercase();
        let by_name = run(&f, &idx, "megalo", &["vars", &format!("owner={name}"), "kind=timer", "var=2"]);
        assert_eq!(by_name.lines().count() - 1, 6, "{by_name}");
        assert!(by_name.lines().skip(1).all(|l| l.split(',').next().unwrap().parse::<u32>().unwrap() >= 25006));
        // dump: one JSON with every section
        let d: serde_json::Value = serde_json::from_str(&run(&f, &idx, "dump", &[])).unwrap();
        for k in ["variant", "megalo", "seats", "events", "score", "objects"] {
            assert!(!d[k].is_null(), "dump.{k}");
        }
        assert_eq!(d["objects"].as_array().unwrap().len(), table.rows.len());
        assert_eq!(d["events"].as_array().unwrap().len(), idx.events.len());
    }
}

