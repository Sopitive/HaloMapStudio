//! CROSS-GAME map-variant conversion (#xgame): take a Forge variant authored in one Halo title
//! and re-create it on a Forge canvas of another, mapping every placed object to the nearest
//! equivalent in the target game's palette.
//!
//! Three layers, each a plain data structure so a user can inspect and correct it:
//!
//!   1. [`catalog`] - the CROSS-GAME OBJECT CATALOG. Every Forge palette object of every
//!      installed game with its render-model bounding box in WORLD UNITS, its Forge category /
//!      menu name / tag, its multiplayer object type and the canvases it appears on. Built by
//!      `--xgame-build-catalog` from the installed caches and shipped as
//!      `crates/hms-app/src/xgame/catalog.json` (baked in with `include_str!`), overridable by
//!      dropping a `xgame_catalog.json` into the HMS settings dir.
//!   2. [`table`] - the EQUIVALENCE TABLE. Directed `(source game, source key) -> (target game,
//!      target key)` rows with a confidence and the size delta, as a `#`-commented TSV
//!      (`equivalence.tsv`, same baked-in + settings-dir-override scheme). Auto-proposed by the
//!      matcher in `table::propose`, then hand-curated; a pair the table does not name falls back
//!      to the matcher at run time, so the file is a CURATION layer, never a hard requirement.
//!   3. [`convert`] - the CONVERTER. Reads a variant of any game into the game-neutral
//!      [`convert::XObject`], maps each object through the table, re-centres the cloud inside the
//!      target canvas, enforces the target palette's quotas / budget / slot cap and writes the
//!      result through the existing byte-exact encoder of the target game.
//!
//! AXES: every Halo engine here uses the same right-handed Z-up world-unit frame (`.mvar`
//! positions are `[x, y, z]` world units, `up` defaults to `[0,0,1]`, `fwd` to `[1,0,0]` in
//! Reach `mvar.rs`, `h4/mvar.rs` and `h3/mvar.rs` alike), so a pose carries across titles
//! unchanged - only the canvas ORIGIN differs, which is what the re-centring offset is for.
//!
//! SCALE: Halo 4 records carry a 6-bit object scale (datum +0x2C) that HMS writes, but an in-game
//! test showed no visible effect, so NOTHING here compensates size differences with it. The
//! converter picks the closest-sized palette object and REPORTS the residual delta.

pub mod catalog;
pub mod convert;
pub mod gui;
pub mod table;

use crate::mapcat::Game;

/// The four titles, in the one spelling every data file uses.
pub const GAMES: [Game; 4] = [Game::Reach, Game::Halo4, Game::H2A, Game::Halo3];

/// `"reach" | "halo4" | "h2a" | "halo3"` -> [`Game`] (the spelling used in the data files).
pub fn game_from_str(s: &str) -> Option<Game> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "reach" | "haloreach" | "halo reach" | "hr" => Some(Game::Reach),
        "halo4" | "h4" | "halo 4" => Some(Game::Halo4),
        "h2a" | "halo2a" | "groundhog" | "halo 2 anniversary" | "h2" => Some(Game::H2A),
        "halo3" | "h3" | "halo 3" => Some(Game::Halo3),
        _ => None,
    }
}

/// Leaf of a tag path (`objects\levels\forge\ff_arena_wall\ff_arena_wall` -> `ff_arena_wall`).
pub fn tag_leaf(path: &str) -> String {
    path.rsplit(['\\', '/']).next().unwrap_or(path).to_ascii_lowercase()
}

/// The catalog KEY of a palette object: the object's own identity, never the map's.
/// `<tag leaf>` plus `#<palette variant name>` when the entry has named variants and
/// `@v<hlmt variant sid>` when the palette row selects a model variant - the three things that
/// together pick exactly one placeable shape.
pub fn object_key(tag_path: &str, variant_name: &str, variant_sid: u32) -> String {
    let mut k = tag_leaf(tag_path);
    let v = variant_name.trim();
    if !v.is_empty() && !v.eq_ignore_ascii_case(&k) {
        k.push('#');
        k.push_str(&v.to_ascii_lowercase());
    }
    if variant_sid != 0 {
        k.push_str(&format!("@v{variant_sid:08x}"));
    }
    k
}

/// The ONE parser behind both `--xgame-convert` and the `convertvariant` script verb, so a
/// scripted conversion and a CLI conversion cannot drift:
///
/// ```text
/// <src.mvar> <reach|halo4|halo3> <canvas> [as <out.mvar>] [offset <x,y,z>] [yaw <deg>]
///            [scale <s>] [keeporigin] [minconf exact|close|approximate] [noquota] [nobudget]
///            [template <file.mvar>] [report] [bykind] [dry]
/// ```
///
/// Returns the report text (the summary, or the full listing with `report`). `dry` converts and
/// reports without writing anything.
pub fn convert_cli(args: &[String]) -> Result<String, String> {
    use std::path::PathBuf;
    const USAGE: &str = "convertvariant <src.mvar> <reach|halo4|halo3> <canvas> [as <out.mvar>] \
                         [offset x,y,z] [yaw <deg>] [scale <s>] [keeporigin] \
                         [minconf exact|close|approximate] [noquota] [nobudget] \
                         [template <file.mvar>] [report] [bykind] [dry]";
    let pos: Vec<&String> = args.iter().take_while(|a| !is_kw(a)).collect();
    if pos.len() < 3 { return Err(USAGE.into()); }
    let src_path = PathBuf::from(pos[0]);
    if !src_path.is_file() { return Err(format!("{}: no such file\n{USAGE}", src_path.display())); }
    let dst_game = game_from_str(pos[1]).ok_or_else(|| format!("unknown game '{}'\n{USAGE}", pos[1]))?;
    let canvas = pos[2].clone();

    let mut opt = convert::Options::default();
    let mut out: Option<PathBuf> = None;
    let mut full = false;
    let mut bykind = false;
    let mut dry = false;
    let mut i = pos.len();
    while i < args.len() {
        let a = args[i].to_ascii_lowercase();
        let mut next = |need: &str| -> Result<String, String> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| format!("{a} needs {need}"))
        };
        match a.as_str() {
            "as" | "out" | "to" => out = Some(PathBuf::from(next("a path")?)),
            "offset" => {
                let v = next("x,y,z")?;
                let p: Vec<f32> = v.split([',', ' ']).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
                if p.len() != 3 { return Err("offset needs x,y,z".into()); }
                opt.offset = [p[0], p[1], p[2]];
            }
            "yaw" => opt.yaw_deg = next("degrees")?.parse().map_err(|_| "yaw needs a number".to_string())?,
            "scale" => opt.layout_scale = next("a factor")?.parse().map_err(|_| "scale needs a number".to_string())?,
            "keeporigin" => opt.keep_origin = true,
            "minconf" => opt.min_confidence = table::Confidence::parse(&next("a confidence")?).ok_or("minconf: exact | close | approximate")?,
            "noquota" => opt.enforce_quotas = false,
            "nobudget" => opt.enforce_budget = false,
            "template" => opt.template = Some(PathBuf::from(next("a .mvar")?)),
            "report" | "full" => full = true,
            "bykind" => bykind = true,
            "dry" | "dryrun" => dry = true,
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let cat = catalog::Catalog::load();
    if cat.objects.is_empty() { return Err("the object catalog is empty - run `hms-app --xgame-build-catalog` once".into()); }
    let tab = table::Table::load();
    let src = convert::read_variant(&cat, &src_path)?;
    let out = out.unwrap_or_else(|| {
        let stem = src_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "converted".into());
        src_path.with_file_name(format!("{stem}_{}_{}.mvar", dst_game.short_name(), canvas))
    });
    let rep = if dry {
        convert::convert(&cat, &tab, &src, dst_game, &canvas, &opt)?.report
    } else {
        convert::convert_to_file(&cat, &tab, &src, dst_game, &canvas, &out, &opt)?
    };
    let mut s = if full { rep.describe() } else { rep.summary() };
    if bykind { s.push('\n'); s.push_str(&rep.by_kind()); }
    if !dry { s.push_str(&format!("\nwrote {}", out.display())); }
    Ok(s)
}

/// The option keywords of [`convert_cli`] (so the positional scan knows where to stop).
fn is_kw(a: &str) -> bool {
    matches!(a.to_ascii_lowercase().as_str(),
        "as" | "out" | "to" | "offset" | "yaw" | "scale" | "keeporigin" | "minconf" | "noquota"
        | "nobudget" | "template" | "report" | "full" | "bykind" | "dry" | "dryrun")
}

/// Round-trip GATE for a converted file (`--xgame-verify <file.mvar>`): read it back with the
/// TARGET title's own reader, re-encode it with that title's own encoder and require the bytes to
/// come back identical. This is the check that says "the file HMS just wrote is a file HMS can
/// read, and writing what it read changes nothing" - the same property the per-title shipped
/// `.mvar` gates assert on 333 Halo 3, 388 Halo 4 and every Reach file.
pub fn verify_variant(path: &std::path::Path) -> Result<String, String> {
    let data = crate::mvar::read_expanded(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let game = convert::variant_game(path).ok_or_else(|| format!("{}: not a map variant", path.display()))?;
    match game {
        Game::Halo4 | Game::H2A => {
            let r = crate::h4::mvar::roundtrip_report(path, None).map_err(|e| e.to_string())?;
            if !r.contains("BYTE-IDENTICAL") { return Err(format!("{}: re-encode DIFFERS\n{r}", path.display())); }
            Ok(format!("{}: {r}", path.display()))
        }
        Game::Halo3 => {
            let v = crate::h3::mvar::parse_h3_variant(path).map_err(|e| e.to_string())?;
            let out = crate::h3::mvar::rebuild_h3_file(&data, &v, &v.objects, None).map_err(|e| e.to_string())?;
            if out != data { return Err(format!("{}: Halo 3 re-encode DIFFERS ({} vs {} bytes)", path.display(), out.len(), data.len())); }
            let placed = v.placed().count();
            Ok(format!("{}: halo3 body v{}, {} slots ({placed} placed), {} quotas, budget {}/{}, re-encode BYTE-IDENTICAL ({} bytes)",
                path.display(), v.version, v.objects.len(), v.quotas.len(), v.spent_budget, v.maximum_budget, data.len()))
        }
        Game::Reach => {
            let v = crate::mvar::parse_variant(path).ok_or_else(|| format!("{}: cannot parse", path.display()))?;
            let (out, written) = crate::mvar::rebuild_file(&data, &v.objects, None)?;
            if out != data { return Err(format!("{}: Reach re-encode DIFFERS ({} vs {} bytes)", path.display(), out.len(), data.len())); }
            Ok(format!("{}: reach v{}, {} objects ({written} written), map {}, re-encode BYTE-IDENTICAL ({} bytes)",
                path.display(), v.globals.version, v.objects.len(), v.map_id, data.len()))
        }
    }
}
