//! The CROSS-GAME EQUIVALENCE TABLE (#xgame): which object of title B stands in for an object of
//! title A, how confident that is, and by how much the two differ in size.
//!
//! A plain `#`-commented TSV so a user can correct a row in any text editor and reload with no
//! rebuild. One directed row per `(source game, source key) -> target game`:
//!
//! ```text
//! # src_game  src_key                 dst_game  dst_key                     conf     dx,dy,dz   note
//! reach       ff_arena_wall#wall_coliseum  halo4  fw_unsc_c_arena_wall#drw_wall_coliseum  exact  0.00,0.00,0.00  Coliseum Wall
//! reach       spawn_point              halo4     ff_initial_spawn            exact    -          respawn point
//! reach       warthog                  halo3     warthog                     exact    -          1:1
//! ```
//!
//! `conf` is one of `exact` / `close` / `approximate` / `none` (`none` = deliberately DROP: the
//! target title has nothing like it, and the converter must not substitute something wrong).
//! `dst_key` may be `-` together with `conf none`.
//!
//! The table is a CURATION layer: a pair it does not name falls back to [`best_match`], the same
//! matcher that proposed the table in the first place (`--xgame-build-table`). So a fresh install
//! converts sensibly, and a user's edits always win.

use std::collections::HashMap;

use crate::mapcat::Game;

use super::catalog::{Catalog, CatalogObject, Kind};

/// The baked-in table (regenerate the auto rows with `--xgame-build-table`).
pub const BUILTIN_TABLE: &str = include_str!("equivalence.tsv");
/// File name of the user override in the HMS settings dir.
pub const OVERRIDE_FILE: &str = "xgame_equivalence.tsv";

/// How good a mapping is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// The same object (a gameplay role that exists in both titles, or the same shape and size).
    Exact,
    /// Recognisably the same piece; size within ~12 %.
    Close,
    /// The nearest thing the target title has; expect to nudge it by hand.
    Approximate,
    /// Nothing in the target title stands in - the object is DROPPED.
    None,
}

impl Confidence {
    pub fn name(self) -> &'static str {
        match self { Confidence::Exact => "exact", Confidence::Close => "close", Confidence::Approximate => "approximate", Confidence::None => "none" }
    }
    pub fn parse(s: &str) -> Option<Confidence> {
        match s.trim().to_ascii_lowercase().as_str() {
            "exact" => Some(Confidence::Exact),
            "close" => Some(Confidence::Close),
            "approximate" | "approx" => Some(Confidence::Approximate),
            "none" | "drop" => Some(Confidence::None),
            _ => None,
        }
    }
}

/// One directed mapping row.
#[derive(Clone, Debug, PartialEq)]
pub struct EqRow {
    pub src_game: Game,
    pub src_key: String,
    pub dst_game: Game,
    /// Empty when `conf == None` (an explicit drop).
    pub dst_key: String,
    pub conf: Confidence,
    /// Target size minus source size, world units; `[0;3]` when either side has no box.
    pub delta: [f32; 3],
    /// The axis mapping the converter must apply when placing (`perm[i] = j`: the target's local
    /// axis `j` plays the source's axis `i`). Stored in the TSV's `axes` column as `xyz`/`yxz`/...
    pub perm: [usize; 3],
    pub note: String,
    /// True when a human wrote / reviewed the row (the `curated` marker in the note column).
    pub curated: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub rows: Vec<EqRow>,
    /// Lines the parser could not read, with their 1-based line numbers (surfaced in the GUI).
    pub errors: Vec<(usize, String)>,
    index: HashMap<(&'static str, String, &'static str), usize>,
}

impl Table {
    /// The table HMS uses: the settings-dir override when present, else the baked-in copy.
    pub fn load() -> Table {
        if let Some(p) = crate::project::settings_dir().map(|d| d.join(OVERRIDE_FILE)) {
            if p.is_file() {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    let t = Table::parse(&s);
                    log::info!("#xgame: equivalence override {} ({} rows, {} bad lines)", p.display(), t.rows.len(), t.errors.len());
                    return t;
                }
            }
        }
        Table::parse(BUILTIN_TABLE)
    }

    pub fn parse(text: &str) -> Table {
        let mut t = Table::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            // Columns are POSITIONAL: never drop an empty one, or a blank delta would shift the
            // note into its place (a user editing by hand leaves blanks all the time).
            let f: Vec<&str> = line.split('\t').map(|s| s.trim()).collect();
            if f.len() < 5 { t.errors.push((n + 1, format!("needs at least 5 tab-separated columns, got {}", f.len()))); continue; }
            let (Some(sg), Some(dg)) = (super::game_from_str(f[0]), super::game_from_str(f[2])) else {
                t.errors.push((n + 1, format!("unknown game '{}' / '{}'", f[0], f[2]))); continue;
            };
            let Some(conf) = Confidence::parse(f[4]) else { t.errors.push((n + 1, format!("unknown confidence '{}'", f[4]))); continue };
            if f[1].is_empty() { t.errors.push((n + 1, "the source key is empty".into())); continue; }
            // `-` or blank in the target column = an explicit DROP, which must agree with `none`
            let dst_key = if f[3] == "-" || f[3].is_empty() { String::new() } else { f[3].to_ascii_lowercase() };
            if dst_key.is_empty() && conf != Confidence::None {
                t.errors.push((n + 1, format!("no target, but the confidence is '{}' (an empty target needs `none`)", conf.name()))); continue;
            }
            if !dst_key.is_empty() && conf == Confidence::None {
                t.errors.push((n + 1, format!("confidence `none` means DROP, so the target must be `-` (got '{dst_key}')"))); continue;
            }
            let delta = f.get(5).map(|s| parse_delta(s)).unwrap_or([0.0; 3]);
            let note = f.get(6).copied().unwrap_or("").to_string();
            // column 8 (optional, defaults to the identity so older files keep working)
            let perm = f.get(7).map(|s| axes_parse(s)).unwrap_or(AXES_IDENTITY);
            t.rows.push(EqRow {
                src_game: sg, src_key: f[1].to_ascii_lowercase(), dst_game: dg, dst_key,
                conf, delta, perm, curated: note.to_ascii_lowercase().contains("curated"), note,
            });
        }
        t.reindex();
        t
    }

    fn reindex(&mut self) {
        self.index.clear();
        for (i, r) in self.rows.iter().enumerate() {
            // a later row wins, so a user can append a correction instead of editing in place
            self.index.insert((r.src_game.short_name(), r.src_key.clone(), r.dst_game.short_name()), i);
        }
    }

    pub fn row(&self, src: Game, key: &str, dst: Game) -> Option<&EqRow> {
        self.index.get(&(src.short_name(), key.to_ascii_lowercase(), dst.short_name())).and_then(|&i| self.rows.get(i))
    }

    pub fn to_tsv(&self) -> String {
        let mut s = String::from(TSV_HEADER);
        let mut rows: Vec<&EqRow> = self.rows.iter().collect();
        rows.sort_by(|a, b| (a.src_game.short_name(), a.dst_game.short_name(), a.src_key.as_str()).cmp(&(b.src_game.short_name(), b.dst_game.short_name(), b.src_key.as_str())));
        let mut pair = ("", "");
        for r in rows {
            let p = (r.src_game.short_name(), r.dst_game.short_name());
            if p != pair { s.push_str(&format!("\n# ---- {} -> {} ----\n", r.src_game.display_name(), r.dst_game.display_name())); pair = p; }
            let d = if r.delta == [0.0; 3] { "-".to_string() } else { format!("{:.2},{:.2},{:.2}", r.delta[0], r.delta[1], r.delta[2]) };
            s.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                r.src_game.short_name(), r.src_key, r.dst_game.short_name(),
                if r.dst_key.is_empty() { "-" } else { &r.dst_key }, r.conf.name(), d, r.note, axes_str(r.perm)));
        }
        s
    }
}

fn parse_delta(s: &str) -> [f32; 3] {
    if s == "-" { return [0.0; 3]; }
    let mut d = [0.0f32; 3];
    for (i, p) in s.split(',').take(3).enumerate() { d[i] = p.trim().parse().unwrap_or(0.0); }
    d
}

pub const TSV_HEADER: &str = "\
# CROSS-GAME EQUIVALENCE TABLE  (#xgame)  -  Halo Map Studio
#
# Tab-separated. Columns:
#   src_game  src_key  dst_game  dst_key  conf  dx,dy,dz  note  axes
#
#   src_game / dst_game : reach | halo4 | h2a | halo3
#   src_key / dst_key   : the catalog key of the object (`hms-app --xgame-catalog <filter>` lists
#                         them). `-` in dst_key together with conf `none` means DROP the object.
#   conf                : exact | close | approximate | none
#   dx,dy,dz            : target size minus source size in world units (informational), or `-`
#   note                : free text; put the word `curated` in it to mark a row you reviewed
#   axes                : which of the TARGET's local axes plays each of the SOURCE's, as a
#                         permutation of `xyz` (`xyz` = none, `yxz` = the target model is turned
#                         90 degrees about Z relative to the source). Optional; blank = `xyz`.
#
# A LATER row for the same (src_game, src_key, dst_game) wins, so you can append a correction
# instead of editing a line in place. A pair this file does not name falls back to HMS's own
# size + name + category matcher, so an unlisted object still converts.
#
# Regenerate the auto-proposed rows with:  hms-app --xgame-build-table [out.tsv]
# Edit a copy without rebuilding HMS: drop it at <settings dir>/xgame_equivalence.tsv
";

// ---------------------------------------------------------------------------------------------
// the matcher
// ---------------------------------------------------------------------------------------------

/// Noise tokens stripped before comparing names: title / category / class prefixes that say
/// nothing about the SHAPE.
const NOISE: &[&str] = &[
    "ff", "fw", "bb", "drw", "sc", "dom", "aa", "wep", "veh", "gad", "obj", "fx", "unsc", "cov",
    "covenant", "human", "forerunner", "fore", "c", "d", "e", "b", "a", "structure", "structures",
    "gadget", "gadgets", "scenery", "vehicles", "weapons", "forge", "levels", "objects", "thorage",
    "mcc", "island", "fi", "ca", "the", "of", "and", "misc", "default", "generic", "new", "01",
];

/// Synonyms folded to one spelling so the four titles' naming habits line up.
const SYNONYMS: &[(&str, &str)] = &[
    ("arena", "coliseum"), ("colise", "coliseum"), ("coliseumwall", "coliseum"),
    ("incline", "ramp"), ("slope", "ramp"), ("wedge", "ramp"), ("ramps", "ramp"),
    ("walkway", "bridge"), ("catwalk", "bridge"), ("gangway", "bridge"), ("bridges", "bridge"),
    ("plat", "platform"), ("platforms", "platform"), ("deck", "platform"), ("landing", "platform"),
    ("blocks", "block"), ("plate", "flat"), ("thin", "narrow"),
    ("sml", "small"), ("sm", "small"), ("lrg", "large"), ("lg", "large"), ("xlrg", "xlarge"),
    ("med", "medium"), ("mid", "medium"), ("dbl", "double"), ("sgl", "single"),
    ("win", "window"), ("windows", "window"), ("doors", "door"), ("walls", "wall"),
    ("rail", "railing"), ("railings", "railing"), ("guardrail", "railing"),
    ("boulder", "rock"), ("rocks", "rock"), ("stone", "rock"), ("pine", "tree"), ("seq", "tree"),
    ("crates", "crate"), ("container", "crate"), ("box", "crate"), ("barrel", "crate"),
    ("sandbags", "sandbag"), ("barricades", "barricade"), ("cones", "cone"),
    ("mancannon", "mancannons"), ("gravlift", "mancannons"), ("gravitylift", "mancannons"),
    ("teleporters", "teleporter"), ("powerups", "powerup"), ("explosives", "explosive"),
    ("lights", "light"), ("corridors", "corridor"), ("pipes", "pipe"), ("tube", "pipe"),
    ("shortwall", "wall"), ("tall", "high"),
];

/// Normalised SHAPE tokens of a catalog row: its key, entry, variant and display name, split on
/// every non-alphanumeric boundary and on digit/letter boundaries (so `2x2`, `1x1_flat` survive
/// as their own strong tokens), with noise dropped and synonyms folded.
pub fn tokens(o: &CatalogObject) -> Vec<String> {
    let raw = format!("{} {} {} {}", o.key.replace(['#', '@'], " "), o.entry, o.variant, o.display);
    let mut out: Vec<String> = Vec::new();
    for t in raw.split(|c: char| !c.is_ascii_alphanumeric()) {
        let t = t.to_ascii_lowercase();
        if t.is_empty() { continue; }
        // keep NxM size tokens whole (the strongest cross-game signal there is)
        let is_size = t.len() >= 3 && t.contains('x') && t.chars().all(|c| c.is_ascii_digit() || c == 'x');
        let t = if is_size { t } else { SYNONYMS.iter().find(|(a, _)| *a == t).map(|(_, b)| b.to_string()).unwrap_or(t) };
        if NOISE.contains(&t.as_str()) { continue; }
        if !out.contains(&t) { out.push(t); }
    }
    out
}

/// Jaccard overlap of two token sets, with the NxM size tokens weighted triple.
pub fn name_score(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let w = |t: &String| if t.contains('x') && t.chars().all(|c| c.is_ascii_digit() || c == 'x') { 3.0 } else { 1.0 };
    let inter: f32 = a.iter().filter(|t| b.contains(t)).map(w).sum();
    let total: f32 = a.iter().map(w).sum::<f32>() + b.iter().map(w).sum::<f32>() - inter;
    if total <= 0.0 { 0.0 } else { (inter / total).clamp(0.0, 1.0) }
}

/// The identity axis mapping: the target's local X plays the source's local X, and so on.
pub const AXES_IDENTITY: [usize; 3] = [0, 1, 2];
/// The six axis permutations, as `perm[source axis] = target axis`.
const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Worst-axis relative error between `src` and `dst` extents under ONE axis mapping
/// (`perm[i] = j` means the target's local axis `j` stands in for the source's local axis `i`).
fn err_under(src: [f32; 3], dst: [f32; 3], perm: [usize; 3]) -> f32 {
    let mut worst = 0.0f32;
    for i in 0..3 {
        let (x, y) = (src[i], dst[perm[i]]);
        // A ONE-world-unit floor: a sub-wu difference on a thin axis is not something a player
        // sees, and without it a Coliseum Wall's 0.50 vs 0.60 wu panel thickness alone would
        // demote an otherwise identical 10 x 12 wall from `close` to `approximate`.
        worst = worst.max((x - y).abs() / x.max(y).max(1.0));
    }
    worst.min(1.0)
}

/// How much better than the identity mapping a permutation has to be before the converter will
/// TURN the object. Below this the two fits are a wash and an unnecessary rotation is the worse
/// answer (it moves the piece's asymmetries around for no gain).
pub const ROTATE_GAIN: f32 = 0.05;

/// The best axis MAPPING between two boxes, and the worst-axis relative error it achieves.
///
/// This is the fix for "some objects are not rotated the way you might expect": comparing SORTED
/// extents correctly recognises that Reach's 2.02 x 1.02 x 1.00 `block_1x2` and Halo 4's
/// 1.00 x 2.00 x 1.00 `bb_1x2` are the same piece, but the two models are authored along
/// DIFFERENT axes - so placing the Halo 4 block at the Reach block's orientation lays it across
/// the gap instead of along it. Returning the permutation lets the converter apply the
/// compensating rotation ([`crate::xgame::convert`]).
///
/// Ties go to the identity, and a permutation is only reported when it beats the identity by
/// [`ROTATE_GAIN`]. `None` when either side has no measured box.
pub fn align_axes(a: &CatalogObject, b: &CatalogObject) -> Option<([usize; 3], f32)> {
    if !a.has_box() || !b.has_box() { return None; }
    let (src, dst) = (a.dims, b.dims);
    let id_err = err_under(src, dst, AXES_IDENTITY);
    let mut best = (AXES_IDENTITY, id_err);
    for p in PERMS {
        let e = err_under(src, dst, p);
        if e < best.1 - 1e-4 { best = (p, e); }
    }
    if best.0 != AXES_IDENTITY && id_err - best.1 < ROTATE_GAIN { return Some((AXES_IDENTITY, id_err)); }
    Some(best)
}

/// Relative size error between two boxes under the best axis mapping, as the WORST axis.
/// 0 = identical, 1 = hopeless. Returns None when either side has no box.
pub fn size_error(a: &CatalogObject, b: &CatalogObject) -> Option<f32> {
    align_axes(a, b).map(|(_, e)| e)
}

/// Target size minus source size, per axis in SORTED order (so it reads as "the target is this
/// much bigger in its long / medium / short axis").
pub fn size_delta(a: &CatalogObject, b: &CatalogObject) -> [f32; 3] {
    if !a.has_box() || !b.has_box() { return [0.0; 3]; }
    let (x, y) = (a.sorted_dims(), b.sorted_dims());
    [y[0] - x[0], y[1] - x[1], y[2] - x[2]]
}

/// A scored candidate.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub key: String,
    pub score: f32,
    pub conf: Confidence,
    pub delta: [f32; 3],
    /// `perm[i] = j`: the target's local axis `j` stands in for the source's local axis `i`.
    /// [`AXES_IDENTITY`] when the two models are authored the same way round.
    pub perm: [usize; 3],
    pub why: String,
}

/// `[1,0,2]` -> `"yxz"`: the axis mapping as the TSV stores it (position i = the target axis that
/// plays the source's axis i).
pub fn axes_str(p: [usize; 3]) -> String { p.iter().map(|&j| ['x', 'y', 'z'][j]).collect() }
/// `"yxz"` -> `[1,0,2]`; anything unreadable (or `-`) is the identity.
pub fn axes_parse(s: &str) -> [usize; 3] {
    let mut out = AXES_IDENTITY;
    let b: Vec<usize> = s.trim().chars().filter_map(|c| "xyz".find(c)).collect();
    if b.len() == 3 && b[0] != b[1] && b[1] != b[2] && b[0] != b[2] { out = [b[0], b[1], b[2]]; }
    out
}

/// How compatible two kinds are: 1.0 identical, 0.45 same group, None incompatible.
///
/// This is a HARD GATE, not a score: a GAMEPLAY kind only ever matches itself (a respawn zone
/// must never become a block), and the STRUCTURE / PROP line is never crossed in either
/// direction - a Forge platform cannot become an artillery emplacement, a crate or a rock, no
/// matter how well the boxes happen to line up.
pub fn kind_fit(a: Kind, b: Kind) -> Option<f32> {
    if a == b { return Some(1.0); }
    if a.is_structural() != b.is_structural() { return None; }
    if a.is_prop() != b.is_prop() { return None; }
    if a.is_gameplay() || b.is_gameplay() {
        // the one deliberate exception: the three teleporter roles substitute for each other
        use Kind::*;
        let tele = |k: Kind| matches!(k, TeleporterSender | TeleporterReceiver | TeleporterTwoWay);
        if tele(a) && tele(b) { return Some(0.6); }
        let spawn = |k: Kind| matches!(k, SpawnInitial | SpawnRespawn);
        if spawn(a) && spawn(b) { return Some(0.5); }
        // A respawn zone and a WEAK respawn zone differ only in strength, so they substitute.
        // An ANTI-respawn zone does the OPPOSITE: turning one into a plain zone would send players
        // exactly where the author kept them away from, so it only ever matches another anti zone.
        let zone = |k: Kind| matches!(k, RespawnZone | RespawnZoneWeak);
        if zone(a) && zone(b) { return Some(0.5); }
        if a == RespawnZoneAnti || b == RespawnZoneAnti { return None; }
        let safe = |k: Kind| matches!(k, SafeZoneSoft | SafeZoneHard);
        if safe(a) && safe(b) { return Some(0.6); }
        let kill = |k: Kind| matches!(k, KillZoneSoft | KillZoneHard);
        if kill(a) && kill(b) { return Some(0.6); }
        return None;
    }
    if a.group().contains(&b) { Some(0.45) } else { None }
}

/// Score `b` as a stand-in for `a`. None when their kinds are incompatible.
pub fn score(a: &CatalogObject, b: &CatalogObject) -> Option<Candidate> {
    let fit = kind_fit(a.kind(), b.kind())?;
    let ns = name_score(&tokens(a), &tokens(b));
    let aligned = align_axes(a, b);
    let err = aligned.map(|(_, e)| e);
    let perm = aligned.map_or(AXES_IDENTITY, |(p, _)| p);
    // A GADGET is matched on its role for the same reason a gameplay object is: a man cannon's
    // marker model is not what the player experiences, so refusing one because its box is 40 %
    // off would drop a working grav lift for no reason.
    let gameplay = a.kind().is_gameplay() || a.kind().is_gadget();
    let (sc, conf, why) = if gameplay {
        // a role, not a shape: the kind IS the answer, the name only breaks ties
        let conf = if a.kind() == b.kind() { Confidence::Exact } else { Confidence::Close };
        (0.75 * fit + 0.25 * ns, conf, format!("role {} (name {:.2})", b.kind.as_str(), ns))
    } else {
        match err {
            Some(e) => {
                // THE CEILING. Past it the piece no longer does the job it was placed to do - a
                // 4 wu floor plate becoming a 2.6 wu one leaves a hole to fall through - so it is
                // REFUSED outright rather than offered as "approximate" and silently substituted.
                if e > MAX_SIZE_ERR { return None; }
                // A large but allowable error needs CORROBORATION: either the two names agree on
                // something (a shared `2x2`, `ramp`, `corner` token) or they are the same kind.
                // Without that, a shape that merely happens to be the right size is a coincidence.
                if e > CORROBORATE_ABOVE && ns < MIN_NAME_AGREEMENT && a.kind() != b.kind() { return None; }
                let ss = 1.0 - e;
                let conf = if e <= EXACT_ERR && a.kind() == b.kind() { Confidence::Exact }
                    else if e <= CLOSE_ERR { Confidence::Close }
                    else { Confidence::Approximate };
                let turn = if perm == AXES_IDENTITY { String::new() } else { format!(", axes {}", axes_str(perm)) };
                (0.58 * ss + 0.27 * ns, conf, format!("size err {:.0}% (name {:.2}, kind {:.2}{turn})", e * 100.0, ns, fit))
            }
            // no box on either side: names and kind only, never better than approximate
            None => {
                if ns < MIN_NAME_AGREEMENT && a.kind() != b.kind() { return None; }
                (0.55 * ns + 0.2 * fit, Confidence::Approximate, format!("no box; name {ns:.2}"))
            }
        }
    };
    // Tie-break towards the GENERIC piece: when two identically-sized candidates exist (the
    // Forge World `fw_unsc_c_arena_wall` and the Forge Island `fi_unsc_c_arena_wall`), prefer the
    // one that appears on more canvases - that is the one a user thinks of as "the" piece.
    let spread = (b.canvases.len().min(10) as f32) / 10.0;
    let sc = sc + 0.15 * fit + 0.03 * spread;
    Some(Candidate { key: b.key.clone(), score: sc, conf, delta: size_delta(a, b), perm, why })
}

/// The lowest total score that still counts as a mapping - below this the object is DROPPED
/// rather than turned into something unrecognisable.
pub const MIN_SCORE: f32 = 0.33;

// --- the size-error ladder --------------------------------------------------------------------
// MEASURED, not invented. Over the 5 625 proposed rows, split by whether the two names agree
// (>= 0.50 token overlap = almost certainly the same piece in two titles):
//
//   name agrees strongly  n=1466   p50  2%   p75  9%   p90 17%   p95 20%   p99 27%
//   weak / no agreement   n=1082   p50 14%   p75 21%   p90 26%   p95 29%
//
//   * 3 % - the "same model, remeasured" cluster: 758 of the 1466 strongly-named rows sit at or
//     under it (Coliseum Wall 10.00/0.50/12.00 in Reach vs 10.00/0.60/11.80 in Halo 4 = 2 %;
//     every `bb_*` / `block_*` and `inc_ramp_*` / `ramp_*` pair lands here).
//   * 12 % - the TROUGH in that histogram (11 % has 18 rows, 12 % has 8, 13 % has 25): the
//     natural break between "the same piece" and "a piece one step along the title's own size
//     grid". The bump at 16-18 % is the Reach-vs-Halo-4 bridge family, genuinely 0.4 wu wider,
//     which a player does notice - so it reads `approximate`, which is the point.
//   * 30 % - past the 95th percentile of even the UNCORROBORATED matches, i.e. beyond here a
//     size agreement is not evidence of anything. It is also where a piece stops doing its job:
//     a 4 wu floor plate replaced by a 2.8 wu one leaves a hole to fall through. The converter
//     REFUSES rather than substituting - that is the difference between the old "100 % coverage"
//     and an honest number.
/// At or under this the two pieces are the same object in two titles.
pub const EXACT_ERR: f32 = 0.03;
/// At or under this they are one step apart on the same size grid.
pub const CLOSE_ERR: f32 = 0.12;
/// Above this a match needs CORROBORATION - either the names agree on something or the two kinds
/// are identical. Set to [`CLOSE_ERR`] deliberately: a `close` match stands on its size alone, but
/// once a piece is more than one grid step off, size agreement on its own is a coincidence. (It
/// is what stopped Reach's `wtf_is_this_thing` becoming Halo 4's "Vertigo Catwalk" on an 18 %
/// size match with ZERO shared name tokens and a different kind.)
pub const CORROBORATE_ABOVE: f32 = CLOSE_ERR;
/// Above this there is NO match: the object is dropped and reported, never substituted.
pub const MAX_SIZE_ERR: f32 = 0.30;
/// The token overlap a large-error match needs to be believed (one shared meaningful token).
pub const MIN_NAME_AGREEMENT: f32 = 0.12;

/// Every acceptable stand-in for `src` among `pool`, best first.
pub fn candidates<'a>(src: &CatalogObject, pool: &[&'a CatalogObject]) -> Vec<Candidate> {
    let mut v: Vec<Candidate> = pool.iter().filter_map(|b| score(src, b)).filter(|c| c.score >= MIN_SCORE).collect();
    v.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal).then(x.key.cmp(&y.key)));
    v
}

/// The single best stand-in, or None.
pub fn best_match<'a>(src: &CatalogObject, pool: &[&'a CatalogObject]) -> Option<Candidate> {
    candidates(src, pool).into_iter().next()
}

/// What a lookup resolved to.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub dst_key: String,
    pub conf: Confidence,
    pub delta: [f32; 3],
    /// The axis mapping to apply when placing (see [`EqRow::perm`]).
    pub perm: [usize; 3],
    /// `table` (a row named it) or `auto` (the matcher chose it), plus the reason.
    pub source: String,
    pub why: String,
}

/// Resolve one object: the table row first when its target is placeable on `canvas`, else the
/// matcher over the canvas's own palette. `None` = drop (with a reason in the report).
pub fn resolve(
    table: &Table,
    cat: &Catalog,
    src_game: Game,
    src_key: &str,
    dst_game: Game,
    canvas_pool: &[&CatalogObject],
) -> Result<Resolved, String> {
    let Some(src) = cat.get(src_game, src_key) else { return Err(format!("'{src_key}' is not in the {} catalog", src_game.display_name())) };
    if let Some(r) = table.row(src_game, src_key, dst_game) {
        if r.conf == Confidence::None {
            return Err(format!("the table drops it{}", if r.note.is_empty() { String::new() } else { format!(": {}", r.note) }));
        }
        match canvas_pool.iter().find(|o| o.key == r.dst_key) {
            Some(o) => return Ok(Resolved {
                dst_key: o.key.clone(), conf: r.conf, delta: if r.delta == [0.0; 3] { size_delta(src, o) } else { r.delta },
                // a curated row's own `axes` wins; an auto row re-derives it from the boxes, so a
                // rebuilt catalog corrects the rotation without anyone editing the table
                perm: if r.curated && r.perm != AXES_IDENTITY { r.perm } else { align_axes(src, o).map_or(AXES_IDENTITY, |(p, _)| p) },
                source: if r.curated { "table (curated)".into() } else { "table".into() }, why: r.note.clone(),
            }),
            // the row names an object this canvas cannot hold - fall through to the matcher
            None => {}
        }
    }
    match best_match(src, canvas_pool) {
        Some(c) => Ok(Resolved { dst_key: c.key, conf: c.conf, delta: c.delta, perm: c.perm, source: "auto".into(), why: c.why }),
        None => Err(format!("no {} object of kind '{}' within tolerance", dst_game.display_name(), src.kind)),
    }
}

/// Auto-propose the whole table from the catalog: for every source object, the best stand-in in
/// each other title (over that title's WHOLE palette, not one canvas).
pub fn propose(cat: &Catalog) -> Table {
    let mut t = Table::default();
    for &sg in &super::GAMES {
        let srcs: Vec<&CatalogObject> = cat.of_game(sg).collect();
        if srcs.is_empty() { continue; }
        for &dg in &super::GAMES {
            if dg == sg { continue; }
            let pool: Vec<&CatalogObject> = cat.of_game(dg).filter(|o| o.places.iter().any(|p| !p.hidden)).collect();
            if pool.is_empty() { continue; }
            for s in &srcs {
                match best_match(s, &pool) {
                    Some(c) => t.rows.push(EqRow {
                        src_game: sg, src_key: s.key.clone(), dst_game: dg, dst_key: c.key, conf: c.conf,
                        delta: c.delta, perm: c.perm, note: format!("auto: {}", c.why), curated: false,
                    }),
                    None => t.rows.push(EqRow {
                        src_game: sg, src_key: s.key.clone(), dst_game: dg, dst_key: String::new(), conf: Confidence::None,
                        delta: [0.0; 3], perm: AXES_IDENTITY,
                        note: format!("auto: nothing in the {} palette is within {:.0}% of this {}", dg.short_name(), MAX_SIZE_ERR * 100.0, s.kind), curated: false,
                    }),
                }
            }
        }
    }
    t.reindex();
    t
}

/// One-line-per-row listing (`--xgame-table [filter]`).
pub fn describe(t: &Table, cat: &Catalog, filter: Option<&str>) -> String {
    let f = filter.map(|s| s.to_ascii_lowercase());
    let mut s = String::new();
    if !t.errors.is_empty() {
        s.push_str(&format!("{} unreadable line(s):\n", t.errors.len()));
        for (n, e) in &t.errors { s.push_str(&format!("  line {n}: {e}\n")); }
    }
    let mut by_pair: std::collections::BTreeMap<(&str, &str), Vec<&EqRow>> = Default::default();
    for r in &t.rows {
        if let Some(n) = &f {
            if !r.src_key.contains(n.as_str()) && !r.dst_key.contains(n.as_str()) && !r.note.to_ascii_lowercase().contains(n.as_str()) { continue; }
        }
        by_pair.entry((r.src_game.short_name(), r.dst_game.short_name())).or_default().push(r);
    }
    for ((a, b), rows) in by_pair {
        let n = |c: Confidence| rows.iter().filter(|r| r.conf == c).count();
        s.push_str(&format!("\n=== {a} -> {b}: {} rows ({} exact, {} close, {} approximate, {} dropped) ===\n",
            rows.len(), n(Confidence::Exact), n(Confidence::Close), n(Confidence::Approximate), n(Confidence::None)));
        for r in rows {
            let d = if r.delta == [0.0; 3] { "        -".to_string() } else { format!("{:+5.1}/{:+5.1}/{:+5.1}", r.delta[0], r.delta[1], r.delta[2]) };
            let kind = cat.get(r.src_game, &r.src_key).map(|o| o.kind.clone()).unwrap_or_default();
            s.push_str(&format!("  {:<42} -> {:<42} {:<12} {d}  [{kind}] {}{}\n",
                r.src_key, if r.dst_key.is_empty() { "(drop)" } else { &r.dst_key }, r.conf.name(),
                if r.curated { "CURATED " } else { "" }, r.note));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked-in table parses with no unreadable lines, and every row names a real game pair.
    #[test]
    fn builtin_table_parses() {
        let t = Table::parse(BUILTIN_TABLE);
        assert!(t.errors.is_empty(), "{:?}", t.errors);
        for r in &t.rows {
            assert_ne!(r.src_game, r.dst_game, "a row must cross titles: {r:?}");
            assert!(!r.src_key.is_empty());
            assert_eq!(r.dst_key.is_empty(), r.conf == Confidence::None, "an empty target must be a drop: {r:?}");
        }
    }

    /// Round-trip: write the table out, read it back, get the same rows.
    #[test]
    fn tsv_round_trips() {
        let t = Table::parse("reach\tfoo\thalo4\tbar\texact\t1.00,2.00,3.00\tcurated note\nhalo3\tbaz\treach\t-\tnone\t-\tnothing like it\n");
        assert_eq!(t.rows.len(), 2);
        assert!(t.rows[0].curated);
        assert_eq!(t.rows[0].delta, [1.0, 2.0, 3.0]);
        assert_eq!(t.rows[1].conf, Confidence::None);
        let back = Table::parse(&t.to_tsv());
        assert_eq!(back.rows.len(), 2);
        assert!(back.errors.is_empty());
        for (a, b) in t.rows.iter().zip(back.rows.iter().chain(back.rows.iter())) {
            if a.src_key == b.src_key { assert_eq!(a.conf, b.conf); assert_eq!(a.dst_key, b.dst_key); }
        }
    }

    /// A LATER row for the same triple wins, so a user can append a correction.
    #[test]
    fn later_row_wins() {
        let t = Table::parse("reach\tfoo\thalo4\tbar\tapproximate\t-\tauto\nreach\tfoo\thalo4\tbetter\texact\t-\tcurated\n");
        let r = t.row(crate::mapcat::Game::Reach, "foo", crate::mapcat::Game::Halo4).unwrap();
        assert_eq!(r.dst_key, "better");
        assert!(r.curated);
    }

    #[allow(clippy::needless_update)]
    fn obj(game: &str, key: &str, kind: Kind, dims: [f32; 3]) -> CatalogObject {
        CatalogObject { game: game.into(), key: key.into(), entry: key.into(), display: key.into(), dims,
            box_source: "render".into(), kind: kind.name().into(), parts: 1,
            places: vec![super::super::catalog::PaletteSpot { canvas: "c".into(), ..Default::default() }],
            canvases: vec!["c".into()], ..Default::default() }
    }

    /// REGRESSION (user-reported): a Reach Forge PLATFORM was being converted into Halo 4's
    /// Covenant artillery emplacement ("FF Dom Turretpad"), because the old classifier sent every
    /// unnamed `bloc` tag through `block_or_wall` and the kind group was a soft score. The
    /// structure / prop line is now a HARD gate in both directions.
    #[test]
    fn a_platform_can_never_become_an_emplacement() {
        use Kind::*;
        // the gate itself
        for (a, b) in [(Block, Scenery), (Platform, Scenery), (Block, Crate), (Ramp, NaturalRock),
                       (Bridge, Decorative), (Wall, Cover), (Platform, Explosive), (Building, Scenery)] {
            assert!(kind_fit(a, b).is_none(), "{a:?} must not match {b:?}");
            assert!(kind_fit(b, a).is_none(), "{b:?} must not match {a:?}");
        }
        // and the two real objects, at their real measured sizes
        let plat = obj("reach", "ff_plat_1x4#block_1x4", Block, [4.02, 1.02, 1.02]);
        let pad = obj("halo4", "cov_artillery_gun#ff_dom_turretpad", Scenery, [2.09, 0.79, 1.63]);
        assert!(score(&plat, &pad).is_none(), "the 1x4 platform must not score against the turret pad");
        assert!(best_match(&plat, &[&pad]).is_none(), "with only the pad on offer the platform must be DROPPED");
    }

    /// The shipped table is AUDITED by the test suite, not just by eye: no row may pair a
    /// structural piece with a prop, and no row may exceed the size-error ceiling.
    #[test]
    fn the_shipped_table_has_no_structural_to_prop_rows() {
        let cat = Catalog::load();
        let t = Table::load();
        if cat.objects.is_empty() || t.rows.is_empty() { return; }
        let mut bad = Vec::new();
        for r in &t.rows {
            if r.conf == Confidence::None { continue; }
            let (Some(a), Some(b)) = (cat.get(r.src_game, &r.src_key), cat.get(r.dst_game, &r.dst_key)) else { continue };
            if a.kind().is_structural() != b.kind().is_structural() || a.kind().is_prop() != b.kind().is_prop() {
                bad.push(format!("{} {} [{}] -> {} {} [{}]", r.src_game.short_name(), r.src_key, a.kind,
                    r.dst_game.short_name(), r.dst_key, b.kind));
            }
            // A CURATED row is a deliberate human override and may exceed the ceiling - that is
            // what curation is for (the Reach Coliseum Wall onto Halo 3, say, where the honest
            // answer is "nothing is this big, here is the nearest, expect to tile it"). Only the
            // AUTOMATIC rows are held to it.
            if !r.curated && !a.kind().is_gameplay() && !a.kind().is_gadget() {
                if let Some(e) = size_error(a, b) {
                    assert!(e <= MAX_SIZE_ERR + 1e-3, "{} {} -> {} {}: size error {:.0}% is past the {:.0}% ceiling",
                        r.src_game.short_name(), r.src_key, r.dst_game.short_name(), r.dst_key, e * 100.0, MAX_SIZE_ERR * 100.0);
                }
            }
        }
        assert!(bad.is_empty(), "{} structure/prop rows, e.g.\n  {}", bad.len(), bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n  "));
        // and specifically: no `ff_plat_*` row may leave the structural family
        for r in t.rows.iter().filter(|r| r.src_key.starts_with("ff_plat_") && r.conf != Confidence::None) {
            let Some(b) = cat.get(r.dst_game, &r.dst_key) else { continue };
            assert!(b.kind().is_structural(), "ff_plat row {} -> {} {} is a {}", r.src_key, r.dst_game.short_name(), r.dst_key, b.kind);
        }
    }

    /// The AXIS MAPPING, pinned on the catalog's own measured boxes.
    ///
    /// The Coliseum Wall is the control: Reach measures 10.00 x 0.50 x 12.00 and Halo 4
    /// 10.00 x 0.60 x 11.80 - BOTH thin in Y - so the correct answer is the IDENTITY and the
    /// converter must not turn it. The 1x2 block is the live case: Reach's is 2.02 x 1.02 x 1.00
    /// (long in X) and Halo 4's is 1.00 x 2.00 x 1.00 (long in Y), so the target has to be yawed
    /// 90 degrees or it lies across the gap the source spanned.
    #[test]
    fn axis_mapping_on_the_real_coliseum_wall_and_the_1x2_block() {
        let cat = Catalog::load();
        if cat.objects.is_empty() { return; }
        let find = |g: Game, pre: &str| cat.of_game(g).find(|o| o.key.starts_with(pre));
        if let (Some(r), Some(h)) = (find(Game::Reach, "ff_arena_wall"), find(Game::Halo4, "fw_unsc_c_arena_wall")) {
            let (perm, err) = align_axes(r, h).expect("both have boxes");
            assert_eq!(perm, AXES_IDENTITY, "the Coliseum Wall is thin in Y in BOTH titles: {:?} vs {:?}", r.dims, h.dims);
            assert!(err <= CLOSE_ERR, "coliseum wall size error {:.2}", err);
        }
        if let (Some(r), Some(h)) = (find(Game::Reach, "ff_plat_1x2#"), find(Game::Halo4, "fw_unsc_c_plat_1x2#")) {
            let (perm, err) = align_axes(r, h).expect("both have boxes");
            assert_eq!(perm, [1, 0, 2], "the 1x2 block is long in X in Reach {:?} and in Y in Halo 4 {:?}", r.dims, h.dims);
            assert!(err <= EXACT_ERR, "with the axes lined up the 1x2 blocks are the same piece ({err:.3})");
            // and without the permutation it would look like a 50 % size error
            assert!(err_under(r.dims, h.dims, AXES_IDENTITY) > 0.4);
        }
    }

    /// A thin-in-X wall against a thin-in-Y wall is recognised as the same piece AND reported as
    /// needing a turn - the machinery the real data only exercises on some pieces.
    #[test]
    fn a_wall_authored_on_a_different_axis_is_matched_and_flagged() {
        let src = obj("reach", "wall_a", Kind::Wall, [10.0, 0.5, 12.0]);
        let dst = obj("halo4", "wall_b", Kind::Wall, [0.5, 10.0, 12.0]);
        let (perm, err) = align_axes(&src, &dst).unwrap();
        assert_eq!(perm, [1, 0, 2]);
        assert!(err < 0.01, "{err}");
        let c = score(&src, &dst).unwrap();
        assert_eq!(c.perm, [1, 0, 2]);
        assert_eq!(c.conf, Confidence::Exact);
        assert_eq!(axes_str(c.perm), "yxz");
        assert_eq!(axes_parse("yxz"), [1, 0, 2]);
        assert_eq!(axes_parse("-"), AXES_IDENTITY);
    }

    /// A near-tie must NOT rotate: turning a piece for a 1 % gain only moves its asymmetries
    /// around.
    #[test]
    fn a_near_cube_is_left_alone() {
        let a = obj("reach", "c1", Kind::Block, [2.0, 2.02, 2.0]);
        let b = obj("halo4", "c2", Kind::Block, [2.02, 2.0, 2.0]);
        assert_eq!(align_axes(&a, &b).unwrap().0, AXES_IDENTITY);
    }

    /// The CEILING: past `MAX_SIZE_ERR` there is no match at all, so the object is dropped and
    /// reported instead of being silently replaced by the nearest wrong thing.
    #[test]
    fn a_match_past_the_ceiling_is_refused_not_approximated() {
        let a = obj("reach", "block_4x4", Kind::Block, [4.0, 4.0, 1.0]);
        let ok = obj("halo4", "bb_4x4", Kind::Block, [4.0, 4.0, 1.0]);
        let too_small = obj("halo4", "bb_1x1", Kind::Block, [1.0, 1.0, 1.0]);
        assert_eq!(score(&a, &ok).unwrap().conf, Confidence::Exact);
        assert!(score(&a, &too_small).is_none(), "75 % smaller is not a match");
        assert!(best_match(&a, &[&too_small]).is_none());
        // and a big-but-allowable error needs the names to agree
        let unrelated = obj("halo4", "silo_cap", Kind::Block, [4.0, 4.0, 1.35]);
        let e = size_error(&a, &unrelated).unwrap();
        assert!(e > CORROBORATE_ABOVE && e <= MAX_SIZE_ERR, "{e}");
        assert!(score(&a, &unrelated).is_some(), "same kind corroborates on its own");
        let other_kind = obj("halo4", "silo_cap", Kind::Building, [4.0, 4.0, 1.35]);
        assert!(score(&a, &other_kind).is_none(), "a different kind with no shared name token is a coincidence");
    }

    /// The matcher must never turn a gameplay object into a structure, and must pick the
    /// same-sized wall over a bigger one.
    #[test]
    fn matcher_gates_on_kind_and_picks_the_closest_size() {
        let src = obj("reach", "ff_arena_wall#wall_coliseum", Kind::Wall, [10.0, 0.5, 12.0]);
        let good = obj("halo4", "fw_unsc_c_arena_wall#drw_wall_coliseum", Kind::Wall, [10.0, 0.5, 12.0]);
        let big = obj("halo4", "fw_unsc_c_wall_2x4", Kind::Wall, [20.0, 1.0, 8.0]);
        let zone = obj("halo4", "respawn_zone", Kind::RespawnZone, [0.0, 0.0, 0.0]);
        let pool = [&good, &big, &zone];
        let best = best_match(&src, &pool).expect("a wall must match");
        assert_eq!(best.key, "fw_unsc_c_arena_wall#drw_wall_coliseum");
        assert_eq!(best.conf, Confidence::Exact);
        assert_eq!(best.delta, [0.0, 0.0, 0.0]);
        // a respawn zone can never become a wall
        let rz = obj("reach", "respawn_zone", Kind::RespawnZone, [0.0; 3]);
        assert!(score(&rz, &good).is_none());
        assert!(score(&rz, &zone).is_some());
        // and the kind gate holds the other way round too
        assert!(score(&src, &zone).is_none());
    }

    /// Size tolerance drives the confidence ladder.
    #[test]
    fn confidence_ladder() {
        let a = obj("reach", "block_2x2", Kind::Block, [8.0, 8.0, 4.0]);
        let same = obj("halo4", "bb_2x2", Kind::Block, [8.0, 8.0, 4.0]);
        let near = obj("halo4", "bb_2x2b", Kind::Block, [8.5, 8.0, 4.0]);
        let step = obj("halo4", "bb_2x2_tall", Kind::Block, [8.0, 8.0, 5.5]);   // one grid step up
        let far = obj("halo4", "bb_5x5", Kind::Block, [20.0, 20.0, 2.0]);       // nothing like it
        assert_eq!(score(&a, &same).unwrap().conf, Confidence::Exact);
        assert_eq!(score(&a, &near).unwrap().conf, Confidence::Close);
        assert_eq!(score(&a, &step).unwrap().conf, Confidence::Approximate);
        assert!(score(&a, &far).is_none(), "past the ceiling there is no match at all");
    }

    /// The NxM token is the strongest name signal: 2x2 beats 3x3 when the sizes are both wrong.
    #[test]
    fn size_tokens_dominate_names() {
        let a = obj("reach", "block_2x2", Kind::Block, [0.0; 3]);
        let two = obj("halo4", "bb_2x2", Kind::Block, [0.0; 3]);
        let three = obj("halo4", "bb_3x3", Kind::Block, [0.0; 3]);
        let s2 = score(&a, &two).unwrap().score;
        let s3 = score(&a, &three).unwrap().score;
        assert!(s2 > s3, "2x2 {s2} should beat 3x3 {s3}");
    }

    /// `resolve` prefers the table, falls back to the matcher, and honours an explicit drop.
    #[test]
    fn resolve_prefers_the_table_then_the_matcher() {
        let src = obj("reach", "src", Kind::Block, [8.0, 8.0, 4.0]);
        let a = obj("halo4", "a", Kind::Block, [8.0, 8.0, 4.0]);
        let b = obj("halo4", "b", Kind::Block, [8.2, 8.0, 4.0]);
        let cat = Catalog { note: String::new(), objects: vec![src.clone(), a.clone(), b.clone()] };
        let pool: Vec<&CatalogObject> = vec![&a, &b];
        // no table row -> the matcher picks the exact-size one
        let t = Table::default();
        let r = resolve(&t, &cat, Game::Reach, "src", Game::Halo4, &pool).unwrap();
        assert_eq!(r.dst_key, "a");
        assert_eq!(r.source, "auto");
        // a row overrides it
        let t = Table::parse("reach\tsrc\thalo4\tb\tclose\t-\tcurated by hand\n");
        let r = resolve(&t, &cat, Game::Reach, "src", Game::Halo4, &pool).unwrap();
        assert_eq!(r.dst_key, "b");
        assert!(r.source.contains("table"));
        // a drop row is honoured
        let t = Table::parse("reach\tsrc\thalo4\t-\tnone\t-\tcurated: nothing like it\n");
        assert!(resolve(&t, &cat, Game::Reach, "src", Game::Halo4, &pool).is_err());
        // a row naming an object this canvas cannot hold falls back to the matcher
        let t = Table::parse("reach\tsrc\thalo4\tnot_on_this_canvas\texact\t-\t\n");
        let r = resolve(&t, &cat, Game::Reach, "src", Game::Halo4, &pool).unwrap();
        assert_eq!(r.source, "auto");
    }
}
