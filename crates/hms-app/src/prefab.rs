//! PREFABS (#prefab, GitHub issue #5): a saved group of Forge objects you can drop into any
//! variant again and again, in any of the four games.
//!
//! A prefab is ONE self-describing JSON file (`*.hmsprefab`) so it can be shared: mailed, posted,
//! put in a repo, hand-edited in a text editor. The format is documented here and by the field
//! comments on [`Prefab`] / [`PrefabObject`]; nothing in it is a pointer into a particular install.
//!
//! ```text
//! {
//!   "format": "hms-prefab", "version": 1,
//!   "name": "Double tower", "game": "reach", "canvas": "ca_forge_ravine",
//!   "created": "2026-10-09T11:22:33Z", "created_unix": 1791545753, "hms": "1.3.1",
//!   "anchor": "footprint-centre-floor",
//!   "objects": [ { "key": "ff_arena_wall#wall_coliseum", "pos": [..], "fwd": [..], ... } ]
//! }
//! ```
//!
//! IDENTITY. Each object records FOUR independent ways to find the same piece again, strongest
//! first, because none of them survives every move on its own:
//!   * `spot` - the palette coordinates on the canvas it was saved from (exact, same canvas only);
//!   * `key` - the cross-game catalog key (`xgame::object_key` = tag leaf + variant), which is
//!     what the equivalence resolver speaks and what carries the object to another canvas;
//!   * `tag` + `variant` + `variant_sid` - the raw tag path and model variant, so a prefab is
//!     still readable (and matchable) with no catalog at all;
//!   * `display` - what to call it in a report when nothing resolves.
//!
//! POSE. Positions are stored RELATIVE to the prefab's own anchor, never to the map origin, so a
//! prefab is a shape rather than a location. The anchor is the FOOTPRINT CENTRE AND FLOOR of the
//! captured set: the XY centre of the AABB of the object positions, and the lowest z among them
//! (`anchor` = `"footprint-centre-floor"`). Dropping a prefab therefore puts the middle of its
//! footprint under the cursor with its lowest piece at the cursor's height - the same thing
//! placing a single object does. Orientations are stored verbatim (`fwd`/`up`, the engine's own
//! object basis); every title uses the same right-handed Z-up world-unit frame, which is why a
//! pose carries across games unchanged (see `xgame::mod`'s AXES note).
//!
//! PLACEMENT is a rigid motion of that relative cloud: optional mirror across a world axis plane
//! through the anchor, then a yaw about +Z, then a translation to the chosen origin. Mirroring
//! reflects both basis vectors, exactly as the Construct tool's Mirror does, so the copy faces the
//! way a real mirror image would.
//!
//! HONESTY. Same-game placement either resolves every object exactly or says which ones it could
//! not. Cross-game placement routes each object through the existing `xgame` equivalence resolver
//! and reports, per object, whether the match was exact, close or approximate - and an object with
//! no acceptable match is DROPPED and listed, never silently swapped for a wildly wrong piece.
//! ALL OR NOTHING. Capacity (object slots / per-entry quota / canvas budget) is checked BEFORE
//! anything is created, so a prefab that does not fit places nothing at all and names the limit it
//! hit; and if a creation still fails half way through, everything already made is rolled back
//! ([`PrefabScene::pf_remove`]). Half a structure is never left behind.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::mapcat::Game;
use crate::xgame::catalog::Catalog;
use crate::xgame::table::{Confidence, Table};

/// `format` of every file this module writes; a file without it is not a prefab.
pub const FORMAT: &str = "hms-prefab";
/// Current `version`. A reader accepts anything it knows (1), and says so when it does not.
pub const VERSION: u32 = 1;
/// File extension (so the browser can filter, and a shared file is recognisable).
pub const EXT: &str = "hmsprefab";
/// The one anchor rule version 1 uses (see the module docs).
pub const ANCHOR_RULE: &str = "footprint-centre-floor";

/// Hard ceiling on what a prefab file may contain. A prefab is a BUILD, not a map: the biggest
/// variant any of these games can hold is 651 objects, so a file claiming more is either corrupt
/// or hostile and is rejected before anything is allocated for it.
pub const MAX_OBJECTS: usize = 651;
/// Hard ceiling on a prefab file's size on disk, for the same reason (a 651-object prefab is
/// ~200 KB of JSON).
pub const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

// =================================================================================================
// the file format
// =================================================================================================

/// One saved object inside a prefab. Everything is optional except the pose and the identity, so
/// a hand-written prefab can be three lines long and a future field cannot break an old file.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct PrefabObject {
    /// Cross-game catalog key (`xgame::object_key`): the strongest portable identity.
    #[serde(default)]
    pub key: String,
    /// Human name for reports and the preview list.
    #[serde(default)]
    pub display: String,
    /// Full object (`obje`) tag path, e.g. `objects\levels\forge\ff_arena_wall\ff_arena_wall`.
    #[serde(default)]
    pub tag: String,
    /// Palette variant name (`wall_coliseum`), "" on a single-variant entry.
    #[serde(default)]
    pub variant: String,
    /// `hlmt` model-variant string id the palette row selects (0 = the default model).
    #[serde(default)]
    pub variant_sid: u32,
    /// Palette coordinates on the SOURCE canvas: `[category index, quota, variant index]`.
    /// Used only when a prefab is placed back onto the same canvas of the same game.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot: Option<[u32; 3]>,

    /// Position RELATIVE to the prefab anchor, world units.
    pub pos: [f32; 3],
    /// Object basis, verbatim (engine `fwd` = local +X).
    pub fwd: [f32; 3],
    /// Object basis, verbatim (engine `up` = local +Z).
    pub up: [f32; 3],

    /// Forge team: -1 = none, 0..7 the team colours, 8 = neutral.
    #[serde(default = "team_default")]
    pub team: i16,
    /// Forge colour override (-1 / absent = inherit the team colour).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<u8>,
    /// Respawn time, seconds (0 = the palette default).
    #[serde(default)]
    pub spawn_time: u8,
    /// Spawn sequence / spawn order. Also the SCALE encoding on a `scale`-labelled object.
    #[serde(default)]
    pub spawn_sequence: i32,
    /// Forge "user data" byte (Halo 4 only; 0 elsewhere).
    #[serde(default)]
    pub user_data: i32,
    /// Placement flags byte (symmetry / at-start / unique / physics).
    #[serde(default)]
    pub placement: u8,
    /// Halo 4 placement bits 8-9.
    #[serde(default)]
    pub placement_hi: u8,
    /// Boundary shape: 0 none, 1 sphere, 2 cylinder, 3 box.
    #[serde(default)]
    pub shape: u8,
    /// Boundary size in WORLD UNITS - the only spelling all four titles agree on
    /// (sphere `[r]`, cylinder `[r, top, bottom]`, box `[w, l, top, bottom]`).
    #[serde(default)]
    pub shape_wu: [f32; 4],
    /// Gametype label strings (resolved). Label 0 first; the rest are Halo 4 only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// Teleporter channel / passability.
    #[serde(default)]
    pub tele_channel: u8,
    #[serde(default)]
    pub tele_passability: u8,
    /// Spare clips (weapons).
    #[serde(default)]
    pub weapon_clips: u8,
    /// Multiplayer object type, for the report only (the target's comes from its own palette).
    #[serde(default)]
    pub object_type: u8,
    /// Halo 4 record `variant-object-scale` quantum (datum +0x2C), when it was not the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_q: Option<u8>,
    /// Halo 4 Forge object LOCK (a locked object cannot be grabbed in Forge).
    #[serde(default, skip_serializing_if = "is_false")]
    pub locked: bool,
    /// Halo 4 trait-zone index (object type 31).
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub trait_zone: u8,

    // --- the HMS pseudo-flags: project state, never in a .mvar, so the prefab is where they live
    /// SCALED override (`None` = follow the derived default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaled: Option<bool>,
    /// SHADOW (sun-shadow caster) override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<bool>,
    /// Displayed damage state (1 minor .. 4 destroyed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage: Option<u8>,
}

fn team_default() -> i16 {
    8
}
fn is_false(b: &bool) -> bool {
    !*b
}
fn is_zero_u8(v: &u8) -> bool {
    *v == 0
}

impl PrefabObject {
    /// What to call this object in a report: the display name, else the key, else the tag leaf.
    pub fn label(&self) -> String {
        if !self.display.is_empty() {
            return self.display.clone();
        }
        if !self.key.is_empty() {
            return self.key.clone();
        }
        crate::xgame::tag_leaf(&self.tag)
    }
}

/// A saved prefab: a named, dated, game-tagged group of objects with poses relative to its own
/// anchor. See the module docs for the format.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Prefab {
    /// Always [`FORMAT`]. Checked on read.
    pub format: String,
    /// Always [`VERSION`] for files this build writes.
    pub version: u32,
    /// The user's name for it (also the file stem, sanitised).
    pub name: String,
    /// Source game, in the one spelling every HMS data file uses: `reach|halo4|h2a|halo3`.
    pub game: String,
    /// Canvas stem it was captured on (`ca_forge_ravine`, `forge_halo`, `sandbox`).
    #[serde(default)]
    pub canvas: String,
    /// When it was captured, UTC, RFC-3339-ish, for the list.
    #[serde(default)]
    pub created: String,
    /// The same instant as unix seconds (what the list actually sorts on).
    #[serde(default)]
    pub created_unix: u64,
    /// The HMS version that wrote it.
    #[serde(default)]
    pub hms: String,
    /// Free-text note the user can add (or hand-edit into the file).
    #[serde(default)]
    pub notes: String,
    /// Which anchor rule the relative positions use; [`ANCHOR_RULE`] in version 1.
    #[serde(default)]
    pub anchor: String,
    pub objects: Vec<PrefabObject>,
}

impl Default for Prefab {
    fn default() -> Self {
        Prefab {
            format: FORMAT.into(),
            version: VERSION,
            name: String::new(),
            game: String::new(),
            canvas: String::new(),
            created: String::new(),
            created_unix: 0,
            hms: env!("CARGO_PKG_VERSION").into(),
            notes: String::new(),
            anchor: ANCHOR_RULE.into(),
            objects: Vec::new(),
        }
    }
}

impl Prefab {
    pub fn game_of(&self) -> Option<Game> {
        crate::xgame::game_from_str(&self.game)
    }
    /// AABB of the relative positions (min, max). None when empty.
    pub fn extent(&self) -> Option<([f32; 3], [f32; 3])> {
        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        let mut n = 0;
        for o in &self.objects {
            if o.pos.iter().any(|v| !v.is_finite()) {
                continue;
            }
            for k in 0..3 {
                mn[k] = mn[k].min(o.pos[k]);
                mx[k] = mx[k].max(o.pos[k]);
            }
            n += 1;
        }
        (n > 0).then_some((mn, mx))
    }
    /// One line for a list row.
    pub fn summary(&self) -> String {
        let g = self.game_of().map(|g| g.display_name().to_string()).unwrap_or_else(|| self.game.clone());
        let size = self
            .extent()
            .map(|(a, b)| format!("{:.0} x {:.0} x {:.0} wu", b[0] - a[0], b[1] - a[1], b[2] - a[2]))
            .unwrap_or_else(|| "empty".into());
        format!(
            "{} - {} object(s), {} ({}{}){}",
            self.name,
            self.objects.len(),
            g,
            if self.canvas.is_empty() { "canvas unknown".into() } else { self.canvas.clone() },
            if self.created.is_empty() { String::new() } else { format!(", {}", self.created) },
            format!(", {size}")
        )
    }
    /// A per-kind count for the preview ("12 x Coliseum Wall, 4 x Block 2x2…"), most first.
    pub fn contents(&self) -> Vec<(String, usize)> {
        let mut m: HashMap<String, usize> = HashMap::new();
        for o in &self.objects {
            *m.entry(o.label()).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// VALIDATE an untrusted file. Everything a malformed or hostile prefab could do - claim a
    /// wrong format, an unknown version, a silly object count, non-finite poses, a name that walks
    /// out of the prefabs folder - is caught here, with a message instead of a panic.
    ///
    /// Returns the warnings worth showing; a hard problem is an `Err`.
    pub fn validate(&mut self) -> Result<Vec<String>, String> {
        if self.format.trim() != FORMAT {
            return Err(format!("not an HMS prefab (format '{}', expected '{FORMAT}')", self.format));
        }
        if self.version == 0 || self.version > VERSION {
            return Err(format!(
                "prefab format version {} is newer than this build understands (up to {VERSION}) - update HMS",
                self.version
            ));
        }
        if self.objects.is_empty() {
            return Err("the prefab holds no objects".into());
        }
        if self.objects.len() > MAX_OBJECTS {
            return Err(format!(
                "the prefab claims {} objects; a map variant holds at most {MAX_OBJECTS}, so this file is not usable",
                self.objects.len()
            ));
        }
        let mut warn = Vec::new();
        if self.game_of().is_none() {
            warn.push(format!("unknown source game '{}' - cross-game placement is not possible", self.game));
        }
        if !self.anchor.is_empty() && self.anchor != ANCHOR_RULE {
            warn.push(format!("unknown anchor rule '{}' - treated as '{ANCHOR_RULE}'", self.anchor));
        }
        // Per object: a non-finite or absurd pose is repaired to something harmless and reported,
        // because dropping the object silently would change the build without saying so.
        let mut fixed_pose = 0usize;
        let mut fixed_basis = 0usize;
        let mut no_identity = 0usize;
        for o in self.objects.iter_mut() {
            if o.pos.iter().any(|v| !v.is_finite() || v.abs() > 1.0e6) {
                o.pos = [0.0; 3];
                fixed_pose += 1;
            }
            let (f, u) = (Vec3::from(o.fwd), Vec3::from(o.up));
            if !f.is_finite() || !u.is_finite() || f.length_squared() < 1e-9 || u.length_squared() < 1e-9 || f.normalize().cross(u.normalize()).length_squared() < 1e-9 {
                o.fwd = [1.0, 0.0, 0.0];
                o.up = [0.0, 0.0, 1.0];
                fixed_basis += 1;
            }
            if o.key.is_empty() && o.tag.is_empty() && o.spot.is_none() {
                no_identity += 1;
            }
            o.team = o.team.clamp(-1, 8);
            o.shape = o.shape.min(3);
            for v in o.shape_wu.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
                *v = v.clamp(0.0, 10_000.0);
            }
            if let Some(d) = o.damage {
                if d > 4 {
                    o.damage = None;
                }
            }
            o.labels.retain(|l| !l.trim().is_empty());
            o.labels.truncate(4);
            for l in o.labels.iter_mut() {
                l.truncate(64);
            }
        }
        self.name = clean_name(&self.name);
        self.notes.truncate(2000);
        if self.name.is_empty() {
            self.name = "prefab".into();
        }
        if fixed_pose > 0 {
            warn.push(format!("{fixed_pose} object(s) had an unusable position and were moved to the prefab origin"));
        }
        if fixed_basis > 0 {
            warn.push(format!("{fixed_basis} object(s) had an unusable orientation and were reset to the default facing"));
        }
        if no_identity > 0 {
            warn.push(format!("{no_identity} object(s) carry no identity at all and cannot be placed"));
        }
        Ok(warn)
    }
}

// =================================================================================================
// storage: one folder under the settings dir, one file per prefab
// =================================================================================================

/// Where prefabs live: `<settings dir>/prefabs`. Created on first save.
pub fn prefabs_dir() -> Option<PathBuf> {
    crate::project::settings_dir().map(|d| d.join("prefabs"))
}

/// A prefab NAME reduced to something that is safe as a file stem AND safe as a display string.
///
/// This is the one gate that keeps an imported prefab from writing outside the prefabs folder: a
/// path separator, a drive letter, `..`, a NUL and every control character are dropped, so the
/// result can only ever name a file directly inside the folder.
pub fn clean_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => out.push('-'),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {}
            c => out.push(c),
        }
    }
    let out = out.trim().trim_matches('.').trim().to_string();
    let mut out: String = out.chars().take(80).collect();
    if out.is_empty() || out.chars().all(|c| c == '-') {
        out = "prefab".into();
    }
    out
}

/// Full path of the prefab called `name` inside the prefabs folder. `None` when there is no
/// settings dir; the name is always sanitised first, so the result never escapes the folder.
pub fn path_for(name: &str) -> Option<PathBuf> {
    prefabs_dir().map(|d| d.join(format!("{}.{EXT}", clean_name(name))))
}

/// One row of the prefabs list.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub game: String,
    pub canvas: String,
    pub objects: usize,
    pub created: String,
    pub created_unix: u64,
    /// `Some(reason)` when the file is present but unreadable - shown in the list rather than
    /// hidden, so a broken share is visible instead of silently missing.
    pub problem: Option<String>,
}

/// Every prefab in the folder, newest first. Never fails: an unreadable file becomes a row with a
/// `problem` instead of taking the listing down.
pub fn list() -> Vec<Entry> {
    let Some(dir) = prefabs_dir() else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().map_or(true, |x| !x.eq_ignore_ascii_case(EXT)) {
            continue;
        }
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        match load_file(&p) {
            Ok((pf, _)) => out.push(Entry {
                name: if pf.name.is_empty() { stem } else { pf.name.clone() },
                path: p,
                game: pf.game.clone(),
                canvas: pf.canvas.clone(),
                objects: pf.objects.len(),
                created: pf.created.clone(),
                created_unix: pf.created_unix,
                problem: None,
            }),
            Err(why) => out.push(Entry {
                name: stem,
                path: p,
                game: String::new(),
                canvas: String::new(),
                objects: 0,
                created: String::new(),
                created_unix: 0,
                problem: Some(why),
            }),
        }
    }
    out.sort_by(|a, b| b.created_unix.cmp(&a.created_unix).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    out
}

/// Read and validate a prefab file from ANY path. Returns the prefab plus any warnings.
///
/// The file is untrusted input: the size is checked before it is read, the JSON is parsed with
/// `serde_json` (which errors rather than panics), and [`Prefab::validate`] then repairs or
/// refuses everything else.
pub fn load_file(path: &Path) -> Result<(Prefab, Vec<String>), String> {
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    if md.len() > MAX_FILE_BYTES {
        return Err(format!("{}: {} bytes is far too large for a prefab (limit {MAX_FILE_BYTES})", path.display(), md.len()));
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse + validate prefab JSON (the testable half of [`load_file`]).
pub fn parse(text: &str) -> Result<(Prefab, Vec<String>), String> {
    let mut pf: Prefab = serde_json::from_str(text).map_err(|e| format!("not readable as a prefab: {e}"))?;
    let warn = pf.validate()?;
    Ok((pf, warn))
}

/// Serialise a prefab (pretty, so a shared file is readable and diffable).
pub fn to_json(p: &Prefab) -> String {
    serde_json::to_string_pretty(p).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

/// Write a prefab to an explicit path.
pub fn save_file(p: &Prefab, path: &Path) -> Result<PathBuf, String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(path, to_json(p)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path.to_path_buf())
}

/// Store a prefab in the prefabs folder under its own (sanitised) name.
pub fn store(p: &Prefab) -> Result<PathBuf, String> {
    let path = path_for(&p.name).ok_or("no settings folder to keep prefabs in")?;
    save_file(p, &path)
}

/// Find a prefab by name (case-insensitive exact first, then a unique substring).
pub fn find(name: &str) -> Result<Entry, String> {
    let all = list();
    if all.is_empty() {
        return Err("no prefabs saved yet (select some objects and use Tools > Prefabs > Save selection)".into());
    }
    let n = name.trim().to_lowercase();
    if let Some(e) = all.iter().find(|e| e.name.to_lowercase() == n) {
        return Ok(e.clone());
    }
    let hits: Vec<&Entry> = all.iter().filter(|e| e.name.to_lowercase().contains(&n)).collect();
    match hits.len() {
        1 => Ok(hits[0].clone()),
        0 => Err(format!("no prefab called '{name}' (have: {})", all.iter().map(|e| e.name.clone()).collect::<Vec<_>>().join(", "))),
        _ => Err(format!("'{name}' matches {} prefabs: {}", hits.len(), hits.iter().map(|e| e.name.clone()).collect::<Vec<_>>().join(", "))),
    }
}

/// Load a prefab by name from the folder.
pub fn load_named(name: &str) -> Result<(Prefab, Vec<String>), String> {
    let e = find(name)?;
    load_file(&e.path)
}

/// Delete a prefab by name. Only ever removes a `.hmsprefab` directly inside the folder.
pub fn delete(name: &str) -> Result<PathBuf, String> {
    let e = find(name)?;
    let dir = prefabs_dir().ok_or("no prefabs folder")?;
    if e.path.parent() != Some(dir.as_path()) || e.path.extension().map_or(true, |x| !x.eq_ignore_ascii_case(EXT)) {
        return Err(format!("{} is not inside the prefabs folder", e.path.display()));
    }
    std::fs::remove_file(&e.path).map_err(|err| format!("{}: {err}", e.path.display()))?;
    Ok(e.path)
}

/// Rename a prefab (the file and the `name` inside it both move).
pub fn rename(old: &str, new: &str) -> Result<PathBuf, String> {
    let e = find(old)?;
    let (mut pf, _) = load_file(&e.path)?;
    let clean = clean_name(new);
    if clean.eq_ignore_ascii_case(&e.name) {
        return Ok(e.path);
    }
    let dst = path_for(&clean).ok_or("no prefabs folder")?;
    if dst.exists() {
        return Err(format!("a prefab called '{clean}' already exists"));
    }
    pf.name = clean;
    let out = save_file(&pf, &dst)?;
    let _ = std::fs::remove_file(&e.path);
    Ok(out)
}

/// IMPORT a prefab someone shared: validate it, then copy it into the prefabs folder under a free
/// name. The source may be anywhere; the destination is always inside the folder (the name goes
/// through [`clean_name`], so a crafted `name` cannot redirect the write).
pub fn import_from(path: &Path) -> Result<(PathBuf, Prefab, Vec<String>), String> {
    let (mut pf, warn) = load_file(path)?;
    // Prefer the file's own stem when the embedded name is a placeholder, so two imports of
    // different files do not both land on "prefab".
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if pf.name == "prefab" && !stem.is_empty() {
        pf.name = clean_name(&stem);
    }
    let dir = prefabs_dir().ok_or("no settings folder to keep prefabs in")?;
    let base = clean_name(&pf.name);
    let mut name = base.clone();
    let mut n = 2;
    while dir.join(format!("{name}.{EXT}")).exists() {
        name = format!("{base} ({n})");
        n += 1;
        if n > 999 {
            return Err("too many prefabs with that name".into());
        }
    }
    pf.name = name;
    let out = store(&pf)?;
    Ok((out, pf, warn))
}

// =================================================================================================
// geometry: the anchor and the placement transform (pure, so it is testable on its own)
// =================================================================================================

/// The prefab anchor of a cloud of object positions: the XY centre of their AABB and the lowest z
/// (see [`ANCHOR_RULE`]). `[0,0,0]` for an empty cloud.
pub fn anchor_of(positions: &[[f32; 3]]) -> [f32; 3] {
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut n = 0usize;
    for p in positions {
        if p.iter().any(|v| !v.is_finite()) {
            continue;
        }
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
        n += 1;
    }
    if n == 0 {
        return [0.0; 3];
    }
    [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, mn[2]]
}

/// How a prefab is dropped into the world: a mirror, a yaw and a translation, in that order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xform {
    /// Where the prefab's anchor lands, world units.
    pub origin: [f32; 3],
    /// Yaw about +Z, degrees, applied about the anchor.
    pub yaw_deg: f32,
    /// Mirror across the world plane with this axis as its normal (0 = X, 1 = Y, 2 = Z), through
    /// the anchor. `None` = no mirror.
    pub mirror: Option<usize>,
}

impl Default for Xform {
    fn default() -> Self {
        Xform { origin: [0.0; 3], yaw_deg: 0.0, mirror: None }
    }
}

impl Xform {
    /// Apply the transform to one stored (relative) pose -> the world pose to place at.
    ///
    /// Mirroring reflects the position AND both basis vectors, which is what the Construct tool's
    /// Mirror does (`construct::mirror_point` / `mirror_dir`): it flips handedness, exactly as a
    /// real mirror image would, and the engine rebuilds local +Y from `cross(up, fwd)` so the
    /// result is still a basis the record can hold.
    pub fn apply(&self, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let mut p = Vec3::from(pos);
        let mut f = Vec3::from(fwd);
        let mut u = Vec3::from(up);
        if let Some(ax) = self.mirror {
            let n = [Vec3::X, Vec3::Y, Vec3::Z][ax.min(2)];
            // about the prefab's own anchor, which is the local origin of the stored poses
            p = crate::construct::mirror_point(p, Vec3::ZERO, n);
            f = crate::construct::mirror_dir(f, n);
            u = crate::construct::mirror_dir(u, n);
        }
        let yaw = self.yaw_deg.to_radians();
        if yaw != 0.0 {
            let (s, c) = (yaw.sin(), yaw.cos());
            let rot = |v: Vec3| Vec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
            p = rot(p);
            f = rot(f);
            u = rot(u);
        }
        p += Vec3::from(self.origin);
        let f = f.normalize_or_zero();
        let u = u.normalize_or_zero();
        let (f, u) = if f.length_squared() < 1e-9 || u.length_squared() < 1e-9 || f.cross(u).length_squared() < 1e-9 {
            (Vec3::X, Vec3::Z)
        } else {
            (f, u)
        };
        (p.to_array(), f.to_array(), u.to_array())
    }
}

// =================================================================================================
// resolving a prefab's objects against the OPEN canvas's palette
// =================================================================================================

/// How an object was matched to a row of the target palette. Ordered weakest-last so a report can
/// sort on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchKind {
    /// The very same palette object (same game, found by palette slot or by catalog key).
    Exact,
    /// The equivalence resolver called it an exact equivalent in the other game.
    Equivalent,
    /// Close: the resolver's size/name agreement was good but not exact.
    Close,
    /// Approximate: the nearest the target palette can offer. CHECK THESE.
    Approximate,
    /// Nothing acceptable - the object is dropped and listed.
    Unmatched,
}

impl MatchKind {
    pub fn name(self) -> &'static str {
        match self {
            MatchKind::Exact => "exact",
            MatchKind::Equivalent => "equivalent",
            MatchKind::Close => "close",
            MatchKind::Approximate => "APPROXIMATE",
            MatchKind::Unmatched => "NO MATCH",
        }
    }
    /// Exact / equivalent / close are the matches nobody has to go and check.
    pub fn is_solid(self) -> bool {
        matches!(self, MatchKind::Exact | MatchKind::Equivalent | MatchKind::Close)
    }
    pub fn placed(self) -> bool {
        self != MatchKind::Unmatched
    }
}

/// One row of the target palette, as the resolver needs to see it. Filled by the host from
/// whichever game is open (`App::prefab_palette`), so the resolver itself is game-agnostic.
#[derive(Clone, Debug, Default)]
pub struct PalRow {
    /// Palette index - what the host's own "place this row" call takes.
    pub index: usize,
    /// Cross-game catalog key, "" when the catalog does not cover this row.
    pub key: String,
    /// Display text for the report.
    pub display: String,
    /// Object tag path (the no-catalog fallback identity).
    pub tag: String,
    /// Palette variant NAME of the row ("" on a single-variant entry) - recorded in the prefab so
    /// the file says in words which permutation it means, not only inside the key.
    pub variant: String,
    /// `hlmt` model-variant sid of the row.
    pub variant_sid: u32,
    /// Palette coordinates on this canvas: `[category index, quota, variant index]`.
    pub spot: [u32; 3],
    /// Per-instance price (0 when the title's palette is not priced offline, i.e. Reach).
    pub price: f64,
    /// `Maximum Allowed` for the row's ENTRY (<= 0 = unlimited).
    pub max_allowed: i32,
}

/// What the open variant can still take. The host fills it; the resolver only reads it.
#[derive(Clone, Debug, Default)]
pub struct Capacity {
    pub slots_used: usize,
    pub slots_max: usize,
    pub budget_spent: f64,
    pub budget_max: f64,
    /// False for Reach, whose Forge palette carries no offline prices - the report then says so
    /// rather than printing a meaningless number.
    pub budget_priced: bool,
    /// Current instance count per palette QUOTA (entry), so all variants of an entry share it.
    pub per_quota: HashMap<u32, u32>,
}

/// What happened to one prefab object.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub index: usize,
    pub src: String,
    pub dst: String,
    pub kind: MatchKind,
    /// Which of the four identity routes found it (or why none did).
    pub via: String,
    /// Target size minus source size (sorted axes), world units; `[0;3]` when not comparable.
    pub delta: [f32; 3],
    /// Axis permutation the resolver asked for: the target model's local axis that plays each of
    /// the source's (see `xgame::convert::turn_to_axes`). Identity unless a cross-game structural
    /// piece is authored along different axes.
    pub perm: [usize; 3],
    /// The palette row to place, when there is one.
    pub pal: Option<usize>,
    /// World pose the object would take.
    pub pos: [f32; 3],
    pub note: String,
    /// Datum, once it is actually placed.
    pub datum: Option<u32>,
}

/// The whole placement - the thing both the panel and the script verb print.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub prefab: String,
    pub src_game: String,
    pub src_canvas: String,
    pub dst_game: String,
    pub dst_canvas: String,
    pub cross_game: bool,
    pub xform: Option<Xform>,
    pub objects: Vec<Outcome>,
    /// Set when NOTHING was placed. A refusal is always whole: a prefab either goes in or it does
    /// not, so a half-built structure can never appear.
    pub refused: Option<String>,
    pub placed: Vec<u32>,
    /// Labels the target variant has no entry for (dropped, listed).
    pub labels_dropped: Vec<String>,
    pub warnings: Vec<String>,
    pub capacity: Capacity,
    /// Budget the placement would spend (0 when the title is not priced).
    pub budget_add: f64,
    /// True when this was a PREVIEW: everything was resolved and checked, nothing was created.
    pub dry_run: bool,
}

impl Report {
    pub fn solid(&self) -> usize {
        self.objects.iter().filter(|o| o.kind.is_solid()).count()
    }
    pub fn approximate(&self) -> usize {
        self.objects.iter().filter(|o| o.kind == MatchKind::Approximate).count()
    }
    pub fn unmatched(&self) -> usize {
        self.objects.iter().filter(|o| o.kind == MatchKind::Unmatched).count()
    }
    /// One screenful. The headline count deliberately EXCLUDES the approximate matches - an
    /// approximate match is the nearest the palette can offer and is often not the right piece,
    /// so counting it as covered is how a conversion that needed hand work came to report 100%.
    pub fn summary(&self) -> String {
        if let Some(why) = &self.refused {
            return format!("prefab '{}' NOT placed: {why}", self.prefab);
        }
        let x = self.xform.unwrap_or_default();
        format!(
            "prefab '{}' ({} {}) -> {} {}: {} {} of {} object(s) - {} solid, {} APPROXIMATE (check these), {} with no match{}\n\
             at ({:.1}, {:.1}, {:.1}) yaw {:.0} deg{}   slots {}/{}   budget {}{}",
            self.prefab,
            self.src_game,
            if self.src_canvas.is_empty() { "?" } else { &self.src_canvas },
            self.dst_game,
            self.dst_canvas,
            if self.dry_run { "PREVIEW: would place" } else { "placed" },
            // A dry run creates nothing, so reporting `placed.len()` (0) beside "4 solid" reads
            // like a failure. Say what it WOULD place instead.
            if self.dry_run { self.objects.iter().filter(|o| o.kind.placed()).count() } else { self.placed.len() },
            self.objects.len(),
            self.solid(),
            self.approximate(),
            self.unmatched(),
            if self.cross_game { "  [CROSS-GAME: every piece went through the equivalence resolver]" } else { "" },
            x.origin[0],
            x.origin[1],
            x.origin[2],
            x.yaw_deg,
            match x.mirror {
                Some(a) => format!(" mirrored across {}", ["X", "Y", "Z"][a.min(2)]),
                None => String::new(),
            },
            self.capacity.slots_used + if self.dry_run { self.objects.iter().filter(|o| o.kind.placed()).count() } else { self.placed.len() },
            self.capacity.slots_max,
            if self.capacity.budget_priced {
                format!("{:.0}/{:.0}", self.capacity.budget_spent + self.budget_add, self.capacity.budget_max)
            } else {
                "not priced by this title's palette".into()
            },
            if self.labels_dropped.is_empty() {
                String::new()
            } else {
                format!("\ngametype labels this variant has no entry for (dropped): {}", self.labels_dropped.join(", "))
            }
        )
    }
    /// The full per-object listing - the honesty that matters for a cross-game placement.
    pub fn describe(&self) -> String {
        let mut s = self.summary();
        for w in &self.warnings {
            s.push_str(&format!("\n  ! {w}"));
        }
        s.push_str("\nper object:\n");
        for o in &self.objects {
            let d = if o.delta == [0.0; 3] {
                "            ".to_string()
            } else {
                format!("{:+5.1}/{:+5.1}/{:+5.1}", o.delta[0], o.delta[1], o.delta[2])
            };
            s.push_str(&format!(
                "  [{:>3}] {:<12} {:<30} -> {:<30} {d}  at ({:>8.2},{:>8.2},{:>8.2}){}  {}{}\n",
                o.index,
                o.kind.name(),
                o.src,
                if o.dst.is_empty() { "-" } else { &o.dst },
                o.pos[0],
                o.pos[1],
                o.pos[2],
                match o.datum { Some(d) => format!(" 0x{d:08X}"), None => "           ".into() },
                o.via,
                if o.note.is_empty() { String::new() } else { format!(" ({})", o.note) }
            ));
        }
        s
    }
}

/// How to place a prefab.
#[derive(Clone, Debug)]
pub struct PlaceOptions {
    pub xform: Xform,
    /// Refuse anything the equivalence resolver rates worse than this (cross-game only).
    pub min_confidence: Confidence,
    /// Honour the target palette's per-entry `Maximum Allowed`.
    pub enforce_quotas: bool,
    /// Honour the target canvas's budget.
    pub enforce_budget: bool,
    /// Allow the CROSS-GAME route (step two). With this off, a prefab from another game is
    /// refused outright rather than resolved - same-game placement never depends on it.
    pub allow_cross_game: bool,
    /// Resolve and report WITHOUT creating anything (what the panel's preview uses).
    pub dry_run: bool,
}

impl Default for PlaceOptions {
    fn default() -> Self {
        PlaceOptions {
            xform: Xform::default(),
            min_confidence: Confidence::Approximate,
            enforce_quotas: true,
            enforce_budget: true,
            allow_cross_game: true,
            dry_run: false,
        }
    }
}

/// Resolve every prefab object against a target palette. PURE: no scene, no files, so the whole
/// matching policy (and its honesty) is testable on its own.
///
/// `same_canvas` says the palette belongs to the very canvas the prefab was captured on, which is
/// what makes the recorded `spot` usable.
pub fn resolve_objects(
    pf: &Prefab,
    pal: &[PalRow],
    dst_game: Game,
    same_canvas: bool,
    cat: &Catalog,
    table: &Table,
    opt: &PlaceOptions,
) -> Vec<Outcome> {
    let src_game = pf.game_of();
    let cross = src_game.map_or(true, |g| g != dst_game);
    // index the palette the three ways identity can arrive
    let by_spot: HashMap<[u32; 3], usize> = pal.iter().enumerate().map(|(i, r)| (r.spot, i)).collect();
    let mut by_key: HashMap<&str, usize> = HashMap::new();
    for (i, r) in pal.iter().enumerate() {
        if !r.key.is_empty() {
            by_key.entry(r.key.as_str()).or_insert(i);
        }
    }
    let mut by_tag: HashMap<(String, u32), usize> = HashMap::new();
    for (i, r) in pal.iter().enumerate() {
        if !r.tag.is_empty() {
            by_tag.entry((crate::xgame::tag_leaf(&r.tag), r.variant_sid)).or_insert(i);
        }
    }
    // the catalog pool for the equivalence resolver: only objects that really are on this canvas
    let pool: Vec<&crate::xgame::catalog::CatalogObject> =
        pal.iter().filter_map(|r| (!r.key.is_empty()).then(|| cat.get(dst_game, &r.key)).flatten()).collect();

    let mut out = Vec::with_capacity(pf.objects.len());
    for (i, o) in pf.objects.iter().enumerate() {
        let mut r = Outcome {
            index: i,
            src: o.label(),
            dst: String::new(),
            kind: MatchKind::Unmatched,
            via: String::new(),
            delta: [0.0; 3],
            perm: crate::xgame::table::AXES_IDENTITY,
            pal: None,
            pos: o.pos,
            note: String::new(),
            datum: None,
        };
        let hit = |r: &mut Outcome, idx: usize, kind: MatchKind, via: &str| {
            r.pal = Some(idx);
            r.kind = kind;
            r.via = via.into();
            r.dst = pal[idx].display.clone();
        };
        // --- step one: the SAME object, same game. This path never touches the resolver, which
        // is why same-game placement works whatever state the cross-game matcher is in.
        if !cross {
            if same_canvas {
                if let Some(&idx) = o.spot.as_ref().and_then(|s| by_spot.get(s)) {
                    hit(&mut r, idx, MatchKind::Exact, "same palette slot");
                    out.push(r);
                    continue;
                }
            }
            if let Some(&idx) = by_key.get(o.key.as_str()).filter(|_| !o.key.is_empty()) {
                hit(&mut r, idx, MatchKind::Exact, "same catalog object on this canvas");
                out.push(r);
                continue;
            }
            if let Some(&idx) = by_tag.get(&(crate::xgame::tag_leaf(&o.tag), o.variant_sid)).filter(|_| !o.tag.is_empty()) {
                hit(&mut r, idx, MatchKind::Exact, "same object tag on this canvas");
                out.push(r);
                continue;
            }
            // same game but this canvas's palette does not carry the piece: fall through to the
            // resolver, which is the honest answer ("the nearest thing this canvas has").
        }
        // --- step two: CROSS-GAME (or a canvas that lacks the piece) through the existing
        // equivalence resolver. Clearly a second step: it is only reached when step one could not
        // answer, and it is reported as such.
        if cross && !opt.allow_cross_game {
            r.via = "cross-game placement is off".into();
            r.note = format!(
                "the prefab was saved in {} and this is {} - turn cross-game placement on to route it through the equivalence table",
                pf.game,
                dst_game.short_name()
            );
            out.push(r);
            continue;
        }
        let Some(sg) = src_game else {
            r.via = "unknown source game".into();
            r.note = format!("'{}' is not a game HMS knows, so the object cannot be resolved", pf.game);
            out.push(r);
            continue;
        };
        if o.key.is_empty() {
            r.via = "no catalog key".into();
            r.note = "this object carries no portable identity, so it cannot be matched in another game".into();
            out.push(r);
            continue;
        }
        if pool.is_empty() {
            r.via = "no catalog for this canvas".into();
            r.note = "the cross-game object catalog has no entry for this canvas (run --xgame-build-catalog once)".into();
            out.push(r);
            continue;
        }
        match crate::xgame::table::resolve(table, cat, sg, &o.key, dst_game, &pool) {
            Ok(res) if res.conf <= opt.min_confidence => {
                let idx = pal.iter().position(|p| p.key == res.dst_key);
                match idx {
                    Some(idx) => {
                        // `Exact` is reserved for the three step-one routes - the very same
                        // palette object. Anything the RESOLVER chose is at best `Equivalent`,
                        // even when its confidence is "exact": we only get here because this
                        // canvas does not carry the piece, so whatever was found is a stand-in,
                        // however good. Overstating that is how a report comes to read 100%.
                        let kind = match res.conf {
                            Confidence::Exact => MatchKind::Equivalent,
                            Confidence::Close => MatchKind::Close,
                            _ => MatchKind::Approximate,
                        };
                        hit(&mut r, idx, kind, if cross { "equivalence resolver" } else { "nearest piece on this canvas" });
                        r.delta = res.delta;
                        r.note = format!("{}: {}", res.source, res.why);
                        // Turn the pose so the TARGET model lies the way the source one did. A
                        // gameplay marker keeps its own facing (its model box says nothing about
                        // which way a player should look) - the same rule the converter uses.
                        let gameplay = cat.get(dst_game, &res.dst_key).map_or(true, |c| c.kind().is_gameplay());
                        if !gameplay && res.perm != crate::xgame::table::AXES_IDENTITY {
                            r.perm = res.perm;
                            r.note = format!("{} [turned: target axes {}]", r.note, crate::xgame::table::axes_str(res.perm));
                        }
                    }
                    None => {
                        r.via = "resolver".into();
                        r.note = format!("the resolver chose '{}', which is not on this canvas's palette", res.dst_key);
                    }
                }
            }
            Ok(res) => {
                r.via = "resolver".into();
                r.note = format!("best match '{}' is only {} (below the '{}' floor)", res.dst_key, res.conf.name(), opt.min_confidence.name());
            }
            Err(e) => {
                r.via = "resolver".into();
                r.note = e;
            }
        }
        out.push(r);
    }
    out
}

/// Would this set of outcomes FIT? `Some(reason)` = no, and nothing must be created.
///
/// Three limits, each the one the open title actually enforces: the variant's object SLOTS, the
/// palette entry's `Maximum Allowed`, and the canvas BUDGET. A refusal names the limit and the
/// numbers, so "it does not fit" is never a mystery.
pub fn capacity_refusal(outcomes: &[Outcome], pal: &[PalRow], cap: &Capacity, opt: &PlaceOptions) -> Option<String> {
    let want: Vec<&PalRow> = outcomes.iter().filter(|o| o.kind.placed()).filter_map(|o| o.pal.and_then(|i| pal.get(i))).collect();
    if want.is_empty() {
        return Some(format!(
            "none of its {} object(s) resolved to anything this canvas can place{}",
            outcomes.len(),
            match outcomes.iter().find(|o| !o.note.is_empty()) {
                Some(o) => format!(" (first reason: {})", o.note),
                None => String::new(),
            }
        ));
    }
    if cap.slots_max > 0 && cap.slots_used + want.len() > cap.slots_max {
        return Some(format!(
            "it needs {} object slot(s) and only {} of the variant's {} are free",
            want.len(),
            cap.slots_max.saturating_sub(cap.slots_used),
            cap.slots_max
        ));
    }
    if opt.enforce_quotas {
        let mut add: HashMap<u32, u32> = HashMap::new();
        for r in &want {
            *add.entry(r.spot[1]).or_default() += 1;
        }
        let mut over: Vec<String> = Vec::new();
        for (quota, n) in add {
            let row = want.iter().find(|r| r.spot[1] == quota).unwrap();
            if row.max_allowed <= 0 {
                continue; // unlimited
            }
            let have = cap.per_quota.get(&quota).copied().unwrap_or(0);
            if have + n > row.max_allowed as u32 {
                over.push(format!("{} ({} placed + {} wanted > max {})", row.display, have, n, row.max_allowed));
            }
        }
        if !over.is_empty() {
            return Some(format!("it would pass this canvas's per-object limit for {}", over.join("; ")));
        }
    }
    if opt.enforce_budget && cap.budget_priced && cap.budget_max > 0.0 {
        let add: f64 = want.iter().map(|r| r.price).sum();
        if cap.budget_spent + add > cap.budget_max {
            return Some(format!(
                "it costs {add:.0} of budget and only {:.0} of the canvas's {:.0} is left",
                cap.budget_max - cap.budget_spent,
                cap.budget_max
            ));
        }
    }
    None
}

// =================================================================================================
// the host surface
// =================================================================================================

/// What prefabs need from whichever editor is open. `App` implements it for all four games; the
/// headless script host implements it for the games it can place in, so the script verbs work
/// without a window.
pub trait PrefabScene {
    /// The open canvas: `(game, canvas stem)`. `None` = nothing to place into.
    fn pf_target(&self) -> Option<(Game, String)>;
    /// The open canvas's palette, one row per placeable item (see [`PalRow`]).
    fn pf_palette(&self) -> Vec<PalRow>;
    /// What the open variant can still take.
    fn pf_capacity(&self) -> Capacity;
    /// The current multi-selection, in order.
    fn pf_selection(&self) -> Vec<u32>;
    fn pf_set_selection(&mut self, datums: Vec<u32>);
    /// Read a placed object as a prefab object: its palette row (when known) plus the WORLD pose
    /// and every property worth keeping. `None` = not a placement this editor owns.
    fn pf_read(&self, datum: u32) -> Option<(Option<usize>, PrefabObject)>;
    /// Create one object from palette row `pal` with the given prefab object's pose and
    /// properties already in WORLD space. Returns its datum.
    fn pf_place(&mut self, pal: usize, o: &PrefabObject) -> Result<u32, String>;
    /// Delete objects this editor created - the ROLLBACK that keeps "a prefab lands whole or not
    /// at all" true even when a creation fails half way through.
    fn pf_remove(&mut self, datums: &[u32]);
    /// The variant's gametype label table (a label the target has no entry for is dropped).
    fn pf_labels(&self) -> Vec<String>;
    /// Take one undo snapshot before a placement (so a whole prefab undoes in one step).
    fn pf_push_undo(&mut self) {}
    /// Where a prefab lands by default: the point the user is looking at.
    fn pf_drop_point(&self) -> [f32; 3];
    /// Mark the scene as needing a rebuild after objects changed.
    fn pf_invalidate(&mut self) {}
}

/// CAPTURE the current selection as a prefab called `name`.
pub fn capture(scene: &dyn PrefabScene, name: &str) -> Result<Prefab, String> {
    let (game, canvas) = scene.pf_target().ok_or("open a map and a variant first - a prefab is captured from placed objects")?;
    let sel = scene.pf_selection();
    if sel.is_empty() {
        return Err("select the objects to save first (click, then shift-click to add; or box-select)".into());
    }
    if sel.len() > MAX_OBJECTS {
        return Err(format!("{} objects selected; a prefab holds at most {MAX_OBJECTS}", sel.len()));
    }
    let pal = scene.pf_palette();
    let mut objs: Vec<PrefabObject> = Vec::with_capacity(sel.len());
    let mut skipped = 0usize;
    for d in &sel {
        match scene.pf_read(*d) {
            Some((pi, mut o)) => {
                if let Some(row) = pi.and_then(|i| pal.get(i)) {
                    // Identity comes from the PALETTE row, which is where the catalog key and the
                    // canvas spot both live - never from the map's own datum numbering.
                    if o.key.is_empty() {
                        o.key = row.key.clone();
                    }
                    if o.display.is_empty() {
                        o.display = row.display.clone();
                    }
                    if o.tag.is_empty() {
                        o.tag = row.tag.clone();
                    }
                    if o.variant.is_empty() {
                        o.variant = row.variant.clone();
                    }
                    o.spot = Some(row.spot);
                    o.variant_sid = row.variant_sid;
                }
                objs.push(o);
            }
            None => skipped += 1,
        }
    }
    if objs.is_empty() {
        return Err("none of the selected objects is a Forge placement that can be saved in a prefab".into());
    }
    // Relative poses, about the selection's OWN anchor.
    let anchor = anchor_of(&objs.iter().map(|o| o.pos).collect::<Vec<_>>());
    for o in objs.iter_mut() {
        for k in 0..3 {
            o.pos[k] -= anchor[k];
        }
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut pf = Prefab {
        name: clean_name(name),
        game: game.short_name().into(),
        canvas,
        created: utc_stamp(now),
        created_unix: now,
        objects: objs,
        ..Default::default()
    };
    if skipped > 0 {
        pf.notes = format!("{skipped} selected item(s) were not Forge placements and were not saved");
    }
    pf.validate()?;
    Ok(pf)
}

/// `1760000000` -> `"2025-10-09T07:33:20Z"`. A tiny civil-date conversion, so a prefab carries a
/// readable date without pulling in a date crate.
pub fn utc_stamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    // days since 1970-01-01 -> civil date (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", tod / 3600, (tod % 3600) / 60, tod % 60)
}

/// PLACE a prefab into the open variant.
///
/// Two phases, deliberately: everything is resolved and the capacity checked FIRST, and only when
/// the whole prefab fits is a single undo step pushed and the objects created. A prefab therefore
/// either lands complete or does not land at all.
pub fn place(scene: &mut dyn PrefabScene, pf: &Prefab, opt: &PlaceOptions) -> Report {
    let mut rep = Report {
        prefab: pf.name.clone(),
        src_game: pf.game.clone(),
        src_canvas: pf.canvas.clone(),
        xform: Some(opt.xform),
        capacity: Capacity::default(),
        dry_run: opt.dry_run,
        ..Default::default()
    };
    let Some((dst_game, canvas)) = scene.pf_target() else {
        rep.refused = Some("no map variant is open to place into".into());
        return rep;
    };
    rep.dst_game = dst_game.short_name().into();
    rep.dst_canvas = canvas.clone();
    rep.cross_game = pf.game_of().map_or(true, |g| g != dst_game);
    let pal = scene.pf_palette();
    if pal.is_empty() {
        rep.refused = Some("the open map has no Forge palette to place from".into());
        return rep;
    }
    let cap = scene.pf_capacity();
    rep.capacity = cap.clone();

    let cat = Catalog::load();
    let table = Table::load();
    if rep.cross_game && cat.objects.is_empty() {
        rep.refused = Some("cross-game placement needs the object catalog (run `hms-app --xgame-build-catalog` once)".into());
        return rep;
    }
    let same_canvas = !rep.cross_game && canvas.eq_ignore_ascii_case(&pf.canvas);
    let mut outcomes = resolve_objects(pf, &pal, dst_game, same_canvas, &cat, &table, opt);

    // World poses (needed by the report whether or not anything is created). A cross-game piece
    // whose target model is authored along different local axes is TURNED first, so it lies the
    // way the source one did - the same rule the variant converter uses.
    let mut world: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(pf.objects.len());
    for (o, out) in pf.objects.iter().zip(outcomes.iter()) {
        let (fwd, up) = crate::xgame::convert::turn_to_axes(o.fwd, o.up, out.perm);
        world.push(opt.xform.apply(o.pos, fwd, up));
    }
    for (out, w) in outcomes.iter_mut().zip(world.iter()) {
        out.pos = w.0;
    }
    rep.budget_add = outcomes
        .iter()
        .filter(|o| o.kind.placed())
        .filter_map(|o| o.pal.and_then(|i| pal.get(i)).map(|r| r.price))
        .sum();

    if let Some(why) = capacity_refusal(&outcomes, &pal, &cap, opt) {
        rep.objects = outcomes;
        rep.refused = Some(why);
        return rep;
    }
    if rep.unmatched() > 0 {
        rep.warnings.push(format!(
            "{} object(s) had no acceptable match and were left out - nothing was substituted for them",
            rep.unmatched()
        ));
    }
    if opt.dry_run {
        rep.objects = outcomes;
        return rep;
    }

    // --- phase two: create. One undo snapshot for the whole prefab.
    scene.pf_push_undo();
    let labels = scene.pf_labels();
    let mut dropped: Vec<String> = Vec::new();
    let mut placed = Vec::new();
    for (i, out) in outcomes.iter_mut().enumerate() {
        let Some(pi) = out.pal else { continue };
        let src = &pf.objects[i];
        let (pos, fwd, up) = world[i];
        let mut o = src.clone();
        o.pos = pos;
        o.fwd = fwd;
        o.up = up;
        // Keep only the labels the target variant already has an entry for; a label it does not
        // know is reported rather than invented (adding to the table is a variant edit, not a
        // placement). The SCALED / SHADOW pseudo-flags ride along as explicit overrides, so a
        // scaled piece still renders scaled even where its `scale` label was dropped.
        let before = o.labels.len();
        o.labels.retain(|l| labels.iter().any(|k| k.eq_ignore_ascii_case(l)));
        if o.labels.len() != before {
            for l in src.labels.iter().filter(|l| !labels.iter().any(|k| k.eq_ignore_ascii_case(l))) {
                if !dropped.contains(l) {
                    dropped.push(l.clone());
                }
            }
        }
        match scene.pf_place(pi, &o) {
            Ok(d) => {
                out.datum = Some(d);
                placed.push(d);
            }
            Err(e) => {
                // ALL OR NOTHING: a creation that fails half way through would leave part of a
                // structure behind, which is exactly what this feature promises not to do. Undo
                // what has been made so far and refuse the whole placement.
                scene.pf_remove(&placed);
                scene.pf_invalidate();
                for o in outcomes.iter_mut() {
                    o.datum = None;
                }
                rep.refused = Some(format!(
                    "object {} of {} ('{}') could not be created ({e}), so nothing was placed",
                    i + 1,
                    pf.objects.len(),
                    src.label()
                ));
                rep.objects = outcomes;
                return rep;
            }
        }
    }
    scene.pf_invalidate();
    // Selectable as a group straight away, so the whole prefab can be nudged / rotated at once.
    scene.pf_set_selection(placed.clone());
    rep.placed = placed;
    rep.labels_dropped = dropped;
    rep.objects = outcomes;
    rep
}

// =================================================================================================
// the script verb: `prefab save|list|show|place|rename|delete|import|where`
// =================================================================================================

/// `prefab <subcommand> …` - the whole verb, parsed and executed here so the grammar is in one
/// place and the GUI, the headless host and the TCP server cannot drift.
pub const USAGE: &str = "\
prefab save <name>                         save the current selection as a prefab
prefab list [filter]                       what is saved (objects, game, canvas, date)
prefab show <name>                         what one prefab contains
prefab place <name> [at X Y Z] [yaw DEG] [mirror x|y|z] [report] [dry]
                                           [nocrossgame] [noquota] [nobudget]
                                           [minconf exact|close|approximate]
prefab rename <old> <new>                  rename a prefab
prefab delete <name>                       delete a prefab
prefab import <path>                       copy a shared prefab file into the prefabs folder
prefab export <name> <path>                write a prefab to a file you can hand to someone
prefab where                               the prefabs folder";

/// Execute one `prefab …` command against any host.
pub fn command(scene: &mut dyn PrefabScene, args: &str) -> Result<String, String> {
    let t: Vec<String> = args.split_whitespace().map(|s| s.to_string()).collect();
    let sub = t.first().map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| "list".into());
    let rest = &t[t.len().min(1)..];
    match sub.as_str() {
        "where" | "dir" | "folder" => Ok(prefabs_dir().map(|d| d.display().to_string()).unwrap_or_else(|| "no settings folder".into())),
        "list" | "ls" => {
            let f = rest.join(" ").to_lowercase();
            let all = list();
            let rows: Vec<&Entry> = all.iter().filter(|e| f.is_empty() || e.name.to_lowercase().contains(&f)).collect();
            if rows.is_empty() {
                return Ok(format!(
                    "no prefabs{} in {}",
                    if f.is_empty() { String::new() } else { format!(" matching '{f}'") },
                    prefabs_dir().map(|d| d.display().to_string()).unwrap_or_default()
                ));
            }
            let mut s = format!("{} prefab(s) in {}:\n", rows.len(), prefabs_dir().map(|d| d.display().to_string()).unwrap_or_default());
            for e in rows {
                match &e.problem {
                    Some(why) => s.push_str(&format!("  {:<28} UNREADABLE: {why}\n", e.name)),
                    None => s.push_str(&format!(
                        "  {:<28} {:>4} object(s)  {:<6} {:<24} {}\n",
                        e.name, e.objects, e.game, e.canvas, e.created
                    )),
                }
            }
            Ok(s)
        }
        "save" | "add" => {
            let name = rest.join(" ");
            if name.trim().is_empty() {
                return Err("prefab save <name>".into());
            }
            let pf = capture(scene, &name)?;
            let path = store(&pf)?;
            Ok(format!("saved prefab '{}' ({} objects) -> {}", pf.name, pf.objects.len(), path.display()))
        }
        "show" | "info" | "get" => {
            let name = rest.join(" ");
            if name.trim().is_empty() {
                return Err("prefab show <name>".into());
            }
            let (pf, warn) = load_named(&name)?;
            let mut s = pf.summary();
            if !pf.notes.is_empty() {
                s.push_str(&format!("\nnote: {}", pf.notes));
            }
            for w in warn {
                s.push_str(&format!("\n  ! {w}"));
            }
            s.push_str("\ncontents:\n");
            for (k, n) in pf.contents() {
                s.push_str(&format!("  {n:>4} x {k}\n"));
            }
            Ok(s)
        }
        "place" | "drop" => {
            let (name, opt, full) = parse_place(rest, scene)?;
            let (pf, warn) = load_named(&name)?;
            let rep = place(scene, &pf, &opt);
            let mut s = if full { rep.describe() } else { rep.summary() };
            for w in warn {
                s.push_str(&format!("\n  ! {w}"));
            }
            if rep.refused.is_some() {
                return Err(s);
            }
            Ok(s)
        }
        "rename" | "mv" => {
            if rest.len() < 2 {
                return Err("prefab rename <old> <new>".into());
            }
            let p = rename(&rest[0], &rest[1..].join(" "))?;
            Ok(format!("renamed -> {}", p.display()))
        }
        "delete" | "del" | "remove" | "rm" => {
            let name = rest.join(" ");
            if name.trim().is_empty() {
                return Err("prefab delete <name>".into());
            }
            let p = delete(&name)?;
            Ok(format!("deleted {}", p.display()))
        }
        "import" => {
            let p = rest.join(" ");
            if p.trim().is_empty() {
                return Err("prefab import <path to a .hmsprefab>".into());
            }
            let (out, pf, warn) = import_from(Path::new(p.trim()))?;
            let mut s = format!("imported '{}' ({} objects, {}) -> {}", pf.name, pf.objects.len(), pf.game, out.display());
            for w in warn {
                s.push_str(&format!("\n  ! {w}"));
            }
            Ok(s)
        }
        "export" => {
            if rest.len() < 2 {
                return Err("prefab export <name> <path>".into());
            }
            let (pf, _) = load_named(&rest[0])?;
            let mut dst = PathBuf::from(rest[1..].join(" "));
            if dst.is_dir() {
                dst = dst.join(format!("{}.{EXT}", clean_name(&pf.name)));
            }
            let p = save_file(&pf, &dst)?;
            Ok(format!("wrote {}", p.display()))
        }
        other => Err(format!("unknown prefab subcommand '{other}'\n{USAGE}")),
    }
}

/// `prefab place` option grammar (also used by the panel, so the two cannot drift).
fn parse_place(t: &[String], scene: &dyn PrefabScene) -> Result<(String, PlaceOptions, bool), String> {
    const KW: [&str; 10] = ["at", "yaw", "mirror", "report", "full", "dry", "dryrun", "nocrossgame", "noquota", "nobudget"];
    let is_kw = |s: &String| {
        let l = s.to_ascii_lowercase();
        KW.contains(&l.as_str()) || l == "minconf" || l == "camera"
    };
    let name: Vec<&String> = t.iter().take_while(|a| !is_kw(a)).collect();
    if name.is_empty() {
        return Err(format!("prefab place <name> [options]\n{USAGE}"));
    }
    let mut opt = PlaceOptions::default();
    opt.xform.origin = scene.pf_drop_point();
    let mut full = false;
    let mut i = name.len();
    while i < t.len() {
        let a = t[i].to_ascii_lowercase();
        match a.as_str() {
            "at" => {
                let v: Vec<f32> = t
                    .get(i + 1..i + 4)
                    .ok_or("at needs X Y Z")?
                    .iter()
                    .map(|s| s.trim_end_matches(',').parse::<f32>().map_err(|_| "at needs three numbers".to_string()))
                    .collect::<Result<_, _>>()?;
                opt.xform.origin = [v[0], v[1], v[2]];
                i += 3;
            }
            "camera" => opt.xform.origin = scene.pf_drop_point(),
            "yaw" => {
                opt.xform.yaw_deg = t.get(i + 1).ok_or("yaw needs degrees")?.parse().map_err(|_| "yaw needs a number")?;
                i += 1;
            }
            "mirror" => {
                let ax = t.get(i + 1).ok_or("mirror needs x, y or z")?.to_ascii_lowercase();
                opt.xform.mirror = Some(match ax.as_str() {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    _ => return Err("mirror needs x, y or z".into()),
                });
                i += 1;
            }
            "minconf" => {
                let v = t.get(i + 1).ok_or("minconf needs exact|close|approximate")?;
                opt.min_confidence = Confidence::parse(v).ok_or("minconf: exact | close | approximate")?;
                i += 1;
            }
            "report" | "full" => full = true,
            "dry" | "dryrun" => {
                opt.dry_run = true;
                full = true;
            }
            "nocrossgame" => opt.allow_cross_game = false,
            "noquota" => opt.enforce_quotas = false,
            "nobudget" => opt.enforce_budget = false,
            other => return Err(format!("unknown prefab place option '{other}'\n{USAGE}")),
        }
        i += 1;
    }
    Ok((name.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "), opt, full))
}

// =================================================================================================
// tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pos: [f32; 3], fwd: [f32; 3], up: [f32; 3], key: &str) -> PrefabObject {
        // `spot` matches `pal_rows()` below, so the same-canvas route has a slot to find.
        let spot = match key { "block_1x1" => [0, 0, 0], "block_2x2" => [0, 1, 0], _ => [1, 2, 0] };
        PrefabObject { key: key.into(), display: key.into(), tag: format!("objects\\a\\{key}"), spot: Some(spot), pos, fwd, up, team: 8, ..Default::default() }
    }

    fn sample() -> Prefab {
        Prefab {
            name: "tower".into(),
            game: "reach".into(),
            canvas: "ca_forge_ravine".into(),
            objects: vec![
                obj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], "block_1x1"),
                obj([4.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], "block_2x2"),
                obj([0.0, 3.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], "ff_arena_wall"),
            ],
            ..Default::default()
        }
    }

    /// The file format round-trips: write it, read it back, get the same prefab.
    #[test]
    fn round_trip_through_the_file_format() {
        let mut p = sample();
        p.objects[1].team = -1;
        p.objects[1].color = Some(3);
        p.objects[1].scaled = Some(true);
        p.objects[1].spawn_sequence = -40;
        p.objects[2].shape = 3;
        p.objects[2].shape_wu = [12.0, 8.0, 4.0, 2.0];
        p.objects[2].labels = vec!["scale".into()];
        p.objects[2].damage = Some(2);
        p.objects[2].scale_q = Some(11);
        p.created_unix = 1_760_000_000;
        p.created = utc_stamp(p.created_unix);
        let text = to_json(&p);
        let (back, warn) = parse(&text).expect("re-read");
        assert!(warn.is_empty(), "{warn:?}");
        assert_eq!(back, p, "a prefab must survive a write + read unchanged");
        // and it is human-readable JSON, which is the point of the format
        assert!(text.contains("\"format\": \"hms-prefab\""));
        assert!(text.contains("\"key\": \"block_1x1\""));
        assert_eq!(utc_stamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_stamp(1_760_000_000), "2025-10-09T08:53:20Z"); // 20370 days + 32000 s
    }

    /// The anchor is the footprint centre and floor, and capture/restore is EXACT under a pure
    /// translation: every object comes back at the same offset from the drop point.
    #[test]
    fn relative_pose_survives_translation() {
        let world = [[10.0, 20.0, 5.0], [14.0, 20.0, 7.0], [10.0, 23.0, 10.0]];
        let a = anchor_of(&world);
        assert_eq!(a, [12.0, 21.5, 5.0], "XY centre of the AABB, lowest z");
        let rel: Vec<[f32; 3]> = world.iter().map(|p| [p[0] - a[0], p[1] - a[1], p[2] - a[2]]).collect();
        let x = Xform { origin: [-100.0, 7.5, 33.0], ..Default::default() };
        for (r, w) in rel.iter().zip(world.iter()) {
            let (p, _, _) = x.apply(*r, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
            for k in 0..3 {
                assert!((p[k] - (w[k] - a[k] + x.origin[k])).abs() < 1e-4, "{p:?} vs {w:?}");
            }
        }
    }

    /// A placed prefab is GEOMETRICALLY IDENTICAL to the captured one: under translation and yaw
    /// every pairwise distance and every relative orientation is preserved exactly; under a
    /// mirror the distances still are and the handedness flips (which is what a mirror means).
    #[test]
    fn placement_is_a_rigid_motion() {
        let p = sample();
        let dist = |a: &[([f32; 3], [f32; 3], [f32; 3])], i: usize, j: usize| {
            (Vec3::from(a[i].0) - Vec3::from(a[j].0)).length()
        };
        let base: Vec<_> = p.objects.iter().map(|o| (o.pos, o.fwd, o.up)).collect();
        for x in [
            Xform { origin: [100.0, -40.0, 12.0], yaw_deg: 0.0, mirror: None },
            Xform { origin: [-3.0, 9.0, 0.0], yaw_deg: 90.0, mirror: None },
            Xform { origin: [0.0, 0.0, 0.0], yaw_deg: 37.5, mirror: None },
            Xform { origin: [5.0, 5.0, 5.0], yaw_deg: 180.0, mirror: Some(0) },
            Xform { origin: [5.0, 5.0, 5.0], yaw_deg: 0.0, mirror: Some(1) },
        ] {
            let moved: Vec<_> = p.objects.iter().map(|o| x.apply(o.pos, o.fwd, o.up)).collect();
            for i in 0..base.len() {
                for j in 0..base.len() {
                    let d0 = (Vec3::from(base[i].0) - Vec3::from(base[j].0)).length();
                    let d1 = dist(&moved, i, j);
                    assert!((d0 - d1).abs() < 1e-3, "{x:?}: distance {i}-{j} changed {d0} -> {d1}");
                }
                // angles between every pair of basis vectors are preserved by both a rotation
                // and a reflection
                let b = (Vec3::from(base[i].1), Vec3::from(base[i].2));
                let m = (Vec3::from(moved[i].1), Vec3::from(moved[i].2));
                assert!((b.0.dot(b.1) - m.0.dot(m.1)).abs() < 1e-4, "{x:?}: basis angle changed");
                assert!((m.0.length() - 1.0).abs() < 1e-4 && (m.1.length() - 1.0).abs() < 1e-4);
            }
            // handedness of the LAYOUT: preserved by a yaw, flipped by a mirror
            let tri = |a: &Vec<([f32; 3], [f32; 3], [f32; 3])>| {
                let u = Vec3::from(a[1].0) - Vec3::from(a[0].0);
                let v = Vec3::from(a[2].0) - Vec3::from(a[0].0);
                u.cross(v).z
            };
            let s0 = tri(&base.iter().cloned().collect());
            let s1 = tri(&moved);
            if x.mirror.is_some() {
                assert!(s0 * s1 <= 1e-4, "{x:?}: a mirror must flip the layout's handedness ({s0} -> {s1})");
            } else {
                assert!(s0 * s1 >= -1e-4, "{x:?}: a yaw must not flip handedness");
            }
        }
    }

    /// A mirror about an axis plane through the anchor maps the cloud onto itself negated on that
    /// axis - checked component-wise so "mirror" cannot quietly become "rotate".
    #[test]
    fn mirror_negates_exactly_one_axis() {
        let x = Xform { origin: [0.0; 3], yaw_deg: 0.0, mirror: Some(1) };
        let (p, f, u) = x.apply([3.0, 4.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(p, [3.0, -4.0, 5.0]);
        assert_eq!(f, [0.0, -1.0, 0.0]);
        assert_eq!(u, [0.0, 0.0, 1.0]);
    }

    fn pal_rows() -> Vec<PalRow> {
        vec![
            PalRow { index: 0, key: "block_1x1".into(), display: "Block 1x1".into(), tag: "objects\\a\\block_1x1".into(), spot: [0, 0, 0], price: 10.0, max_allowed: 4, ..Default::default() },
            PalRow { index: 1, key: "block_2x2".into(), display: "Block 2x2".into(), tag: "objects\\a\\block_2x2".into(), spot: [0, 1, 0], price: 10.0, max_allowed: 0, ..Default::default() },
            PalRow { index: 2, key: "ff_arena_wall".into(), display: "Coliseum Wall".into(), tag: "objects\\a\\ff_arena_wall".into(), spot: [1, 2, 0], price: 50.0, max_allowed: 0, ..Default::default() },
        ]
    }

    /// Same-game, same-canvas placement resolves by the recorded palette slot and is EXACT for
    /// every object - with an empty catalog and an empty equivalence table, which is the point:
    /// same-game placement must not depend on the cross-game matcher at all.
    #[test]
    fn same_game_resolution_needs_no_catalog() {
        let pf = sample();
        let cat = Catalog::default();
        let table = Table::default();
        let out = resolve_objects(&pf, &pal_rows(), Game::Reach, true, &cat, &table, &PlaceOptions::default());
        assert_eq!(out.len(), 3);
        for o in &out {
            assert_eq!(o.kind, MatchKind::Exact, "{o:?}");
            assert_eq!(o.via, "same palette slot");
        }
        assert_eq!(out.iter().filter_map(|o| o.pal).collect::<Vec<_>>(), vec![0, 1, 2]);
        // a DIFFERENT canvas of the same game still resolves, by catalog key / tag
        let out = resolve_objects(&pf, &pal_rows(), Game::Reach, false, &cat, &table, &PlaceOptions::default());
        for o in &out {
            assert_eq!(o.kind, MatchKind::Exact);
            assert_eq!(o.via, "same catalog object on this canvas");
        }
        // and a prefab whose spot is stale but whose tag is right still lands
        let mut pf2 = pf.clone();
        for o in pf2.objects.iter_mut() {
            o.spot = Some([9, 9, 9]);
            o.key.clear();
        }
        let out = resolve_objects(&pf2, &pal_rows(), Game::Reach, true, &cat, &table, &PlaceOptions::default());
        for o in &out {
            assert_eq!(o.via, "same object tag on this canvas", "{o:?}");
        }
    }

    /// Cross-game placement is a clearly-labelled SECOND step: with it off a foreign prefab is
    /// refused whole and says why, and same-game placement is untouched.
    #[test]
    fn cross_game_is_a_second_step() {
        let pf = sample(); // a Reach prefab
        let cat = Catalog::default();
        let table = Table::default();
        let opt = PlaceOptions { allow_cross_game: false, ..Default::default() };
        let out = resolve_objects(&pf, &pal_rows(), Game::Halo4, false, &cat, &table, &opt);
        assert!(out.iter().all(|o| o.kind == MatchKind::Unmatched));
        assert!(out[0].note.contains("cross-game placement on"), "{}", out[0].note);
        let refusal = capacity_refusal(&out, &pal_rows(), &Capacity::default(), &opt).expect("refused");
        assert!(refusal.contains("none of its 3 object(s) resolved"), "{refusal}");
    }

    /// Cross-game placement through the REAL catalog + table, with the report asserted. Skipped
    /// (with a message) when the installed catalog has no Reach -> Halo 4 overlap to test with,
    /// so the suite still runs on a machine with no games installed.
    #[test]
    fn cross_game_report_is_honest() {
        let cat = Catalog::load();
        let table = Table::load();
        let reach: Vec<&crate::xgame::catalog::CatalogObject> = cat.of_game(Game::Reach).collect();
        let h4_canvas = cat.canvases(Game::Halo4).into_iter().next();
        let (Some(canvas), false) = (h4_canvas, reach.is_empty()) else {
            eprintln!("cross_game_report_is_honest: no catalog installed, nothing to assert");
            return;
        };
        let pool = cat.on_canvas(Game::Halo4, &canvas);
        if pool.is_empty() {
            eprintln!("cross_game_report_is_honest: Halo 4 canvas '{canvas}' has no palette, nothing to assert");
            return;
        }
        let pal: Vec<PalRow> = pool
            .iter()
            .enumerate()
            .map(|(i, c)| PalRow {
                index: i,
                key: c.key.clone(),
                display: c.display.clone(),
                tag: c.tag.clone(),
                variant: c.variant.clone(),
                variant_sid: 0,
                spot: c.spot_on(&canvas).map(|s| [s.category_index, s.quota, s.variant]).unwrap_or([0; 3]),
                price: c.price as f64,
                max_allowed: 0,
            })
            .collect();
        // a prefab of the first few Reach structural pieces that the table/matcher can speak for
        let picks: Vec<&crate::xgame::catalog::CatalogObject> =
            reach.iter().filter(|c| c.kind().is_structural() && c.has_box()).take(6).cloned().collect();
        if picks.is_empty() {
            eprintln!("cross_game_report_is_honest: no structural Reach objects in the catalog");
            return;
        }
        let pf = Prefab {
            name: "xgame probe".into(),
            game: "reach".into(),
            canvas: "ca_forge_ravine".into(),
            objects: picks
                .iter()
                .enumerate()
                .map(|(i, c)| PrefabObject {
                    key: c.key.clone(),
                    display: c.display.clone(),
                    tag: c.tag.clone(),
                    pos: [i as f32 * 4.0, 0.0, 0.0],
                    fwd: [1.0, 0.0, 0.0],
                    up: [0.0, 0.0, 1.0],
                    team: 8,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let out = resolve_objects(&pf, &pal, Game::Halo4, false, &cat, &table, &PlaceOptions::default());
        assert_eq!(out.len(), pf.objects.len(), "one outcome per object, always");
        // THE HONESTY PROPERTY: every outcome is labelled, every placed one names a real palette
        // row and carries the reason it was chosen, and nothing is placed without a match.
        for o in &out {
            assert!(!o.src.is_empty());
            match o.kind {
                MatchKind::Unmatched => {
                    assert!(o.pal.is_none(), "an unmatched object must never be given a piece: {o:?}");
                    assert!(!o.note.is_empty() || !o.via.is_empty(), "an unmatched object must say why: {o:?}");
                }
                k => {
                    let pi = o.pal.expect("a matched object names a palette row");
                    assert!(pi < pal.len());
                    assert!(!o.dst.is_empty(), "a matched object names what it became: {o:?}");
                    assert_eq!(o.via, "equivalence resolver", "cross-game matches must say they came from the resolver");
                    assert!(!o.note.is_empty(), "a resolver match must carry its reason: {o:?}");
                    // never a WILDLY wrong piece: an approximate match is allowed, a match whose
                    // size is out by more than the matcher's own ceiling is not
                    if k != MatchKind::Approximate {
                        assert!(o.delta.iter().all(|d| d.abs() < 50.0), "{o:?}");
                    }
                }
            }
        }
        eprintln!(
            "cross-game probe: {} object(s) -> {} solid, {} approximate, {} unmatched",
            out.len(),
            out.iter().filter(|o| o.kind.is_solid()).count(),
            out.iter().filter(|o| o.kind == MatchKind::Approximate).count(),
            out.iter().filter(|o| o.kind == MatchKind::Unmatched).count()
        );
    }

    /// Capacity: slots, per-entry quota and budget each refuse the WHOLE placement, by name.
    #[test]
    fn quota_and_budget_refuse_the_whole_prefab() {
        let pf = sample();
        let pal = pal_rows();
        let cat = Catalog::default();
        let table = Table::default();
        let opt = PlaceOptions::default();
        let out = resolve_objects(&pf, &pal, Game::Reach, true, &cat, &table, &opt);
        let ok = Capacity { slots_used: 10, slots_max: 651, budget_spent: 0.0, budget_max: 1000.0, budget_priced: true, per_quota: HashMap::new() };
        assert!(capacity_refusal(&out, &pal, &ok, &opt).is_none(), "a prefab that fits is not refused");

        // slots
        let full = Capacity { slots_used: 650, ..ok.clone() };
        let why = capacity_refusal(&out, &pal, &full, &opt).expect("slots");
        assert!(why.contains("3 object slot(s)") && why.contains("only 1"), "{why}");

        // per-entry quota: block_1x1 allows 4 and 4 are already placed
        let mut q = ok.clone();
        q.per_quota.insert(0, 4);
        let why = capacity_refusal(&out, &pal, &q, &opt).expect("quota");
        assert!(why.contains("Block 1x1") && why.contains("max 4"), "{why}");
        assert!(capacity_refusal(&out, &pal, &q, &PlaceOptions { enforce_quotas: false, ..opt.clone() }).is_none(), "noquota lifts it");

        // budget: the three pieces cost 70 and only 50 is left
        let b = Capacity { budget_spent: 950.0, ..ok.clone() };
        let why = capacity_refusal(&out, &pal, &b, &opt).expect("budget");
        assert!(why.contains("costs 70") && why.contains("only 50"), "{why}");
        assert!(capacity_refusal(&out, &pal, &b, &PlaceOptions { enforce_budget: false, ..opt.clone() }).is_none(), "nobudget lifts it");
        // an unpriced title (Reach) is never refused on budget, and says so instead
        let unpriced = Capacity { budget_priced: false, ..b.clone() };
        assert!(capacity_refusal(&out, &pal, &unpriced, &opt).is_none());
    }

    /// A malformed prefab file is REJECTED cleanly - a message, never a panic, and never a write
    /// outside the prefabs folder.
    #[test]
    fn malformed_files_are_rejected_cleanly() {
        for (text, want) in [
            ("", "not readable"),
            ("not json at all", "not readable"),
            ("{}", "not readable"), // missing the required object list
            (r#"{"format":"something-else","version":1,"name":"x","game":"reach","objects":[]}"#, "not an HMS prefab"),
            (r#"{"format":"hms-prefab","version":99,"name":"x","game":"reach","objects":[]}"#, "newer than this build"),
            (r#"{"format":"hms-prefab","version":1,"name":"x","game":"reach","objects":[]}"#, "no objects"),
        ] {
            let e = parse(text).err().unwrap_or_else(|| panic!("{text:?} should have been rejected"));
            assert!(e.to_lowercase().contains(&want.to_lowercase()), "{text:?} -> {e}");
        }
        // too many objects
        let many = Prefab { name: "x".into(), game: "reach".into(), objects: vec![obj([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], "a"); MAX_OBJECTS + 1], ..Default::default() };
        let e = parse(&to_json(&many)).err().expect("too many");
        assert!(e.contains("at most"), "{e}");

        // A coordinate that is not a number at all cannot survive JSON (serde writes NaN as
        // `null` and refuses to read it back). That is still a CLEAN rejection - a message, not a
        // panic - which is what a shared file deserves.
        for text in [
            r#"{"format":"hms-prefab","version":1,"name":"x","game":"reach","objects":[{"pos":[null,0,1],"fwd":[1,0,0],"up":[0,0,1]}]}"#,
            r#"{"format":"hms-prefab","version":1,"name":"x","game":"reach","objects":[{"pos":"over there","fwd":[1,0,0],"up":[0,0,1]}]}"#,
            r#"{"format":"hms-prefab","version":1,"name":"x","game":"reach","objects":[{"fwd":[1,0,0],"up":[0,0,1]}]}"#,
            r#"{"format":"hms-prefab","version":1,"name":"x","game":"reach","objects":"lots"}"#,
        ] {
            let e = parse(text).err().unwrap_or_else(|| panic!("{text} should have been rejected"));
            assert!(e.starts_with("not readable as a prefab"), "{text} -> {e}");
        }

        // garbage FIELDS that CAN be written down are repaired and reported, not fatal. Checked
        // through `validate` directly, because a non-finite float never reaches a file.
        let mut bad = sample();
        bad.objects[0].pos = [f32::NAN, 0.0, 1e30];
        bad.objects[1].fwd = [0.0, 0.0, 0.0];
        bad.objects[2].team = 99;
        bad.objects[2].damage = Some(77);
        bad.objects[2].shape = 200;
        bad.objects[2].labels = vec!["  ".into(), "ok".into()];
        bad.game = "halo9".into();
        bad.anchor = "something-else".into();
        bad.name = "../../escape".into();
        let warn = bad.validate().expect("repairable");
        assert_eq!(bad.objects[0].pos, [0.0; 3]);
        assert_eq!(bad.objects[1].fwd, [1.0, 0.0, 0.0]);
        assert_eq!(bad.objects[1].up, [0.0, 0.0, 1.0]);
        assert_eq!(bad.objects[2].team, 8);
        assert_eq!(bad.objects[2].damage, None);
        assert_eq!(bad.objects[2].shape, 3);
        assert_eq!(bad.objects[2].labels, vec!["ok".to_string()], "a blank label is dropped");
        // the name is sanitised IN PLACE - assert the safety property, not a brittle spelling
        assert!(!bad.name.contains('/') && !bad.name.contains('\\') && bad.name != "..", "{}", bad.name);
        assert!(path_for(&bad.name).map_or(true, |p| p.parent() == prefabs_dir().as_deref()));
        assert!(warn.iter().any(|w| w.contains("unusable position")), "{warn:?}");
        assert!(warn.iter().any(|w| w.contains("unusable orientation")), "{warn:?}");
        assert!(warn.iter().any(|w| w.contains("unknown source game")), "{warn:?}");
        assert!(warn.iter().any(|w| w.contains("unknown anchor rule")), "{warn:?}");
        // and what it repaired now survives a real write + read
        let (again, w2) = parse(&to_json(&bad)).expect("the repaired prefab is writable");
        assert!(w2.iter().any(|w| w.contains("unknown source game")), "{w2:?}");
        assert_eq!(again.objects[0].pos, [0.0; 3]);
    }

    /// An imported prefab can NEVER write outside the prefabs folder, whatever its `name` says.
    #[test]
    fn a_name_can_never_escape_the_prefabs_folder() {
        for evil in [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\evil",
            "/etc/shadow",
            "C:\\Windows\\win.ini",
            "a/b/c",
            "\u{0}boom",
            "....",
            "   ",
            "",
        ] {
            let clean = clean_name(evil);
            assert!(!clean.contains('/') && !clean.contains('\\') && !clean.contains(':'), "{evil:?} -> {clean:?}");
            assert!(!clean.contains('\u{0}'), "{evil:?} -> {clean:?}");
            assert!(!clean.is_empty(), "{evil:?} -> empty");
            assert_ne!(clean, "..", "{evil:?}");
            // and the path it produces is a direct child of the folder
            if let Some(dir) = prefabs_dir() {
                let p = dir.join(format!("{clean}.{EXT}"));
                assert_eq!(p.parent(), Some(dir.as_path()), "{evil:?} -> {}", p.display());
            }
        }
        assert_eq!(clean_name("Double Tower"), "Double Tower");
        assert_eq!(clean_name("a/b"), "a-b");
    }

    /// The `prefab place` option grammar, since the panel and the verb share it.
    #[test]
    fn place_options_parse() {
        struct Nothing;
        impl PrefabScene for Nothing {
            fn pf_target(&self) -> Option<(Game, String)> {
                None
            }
            fn pf_palette(&self) -> Vec<PalRow> {
                Vec::new()
            }
            fn pf_capacity(&self) -> Capacity {
                Capacity::default()
            }
            fn pf_selection(&self) -> Vec<u32> {
                Vec::new()
            }
            fn pf_set_selection(&mut self, _: Vec<u32>) {}
            fn pf_read(&self, _: u32) -> Option<(Option<usize>, PrefabObject)> {
                None
            }
            fn pf_place(&mut self, _: usize, _: &PrefabObject) -> Result<u32, String> {
                Err("nothing".into())
            }
            fn pf_remove(&mut self, _: &[u32]) {}
            fn pf_labels(&self) -> Vec<String> {
                Vec::new()
            }
            fn pf_drop_point(&self) -> [f32; 3] {
                [1.0, 2.0, 3.0]
            }
        }
        let tok = |s: &str| s.split_whitespace().map(|x| x.to_string()).collect::<Vec<_>>();
        let (name, o, full) = parse_place(&tok("Double Tower at 10 20 30 yaw 90 mirror y report noquota"), &Nothing).unwrap();
        assert_eq!(name, "Double Tower");
        assert_eq!(o.xform.origin, [10.0, 20.0, 30.0]);
        assert_eq!(o.xform.yaw_deg, 90.0);
        assert_eq!(o.xform.mirror, Some(1));
        assert!(full && !o.enforce_quotas && o.enforce_budget);
        // no `at` -> the drop point (the camera)
        let (_, o, _) = parse_place(&tok("tower"), &Nothing).unwrap();
        assert_eq!(o.xform.origin, [1.0, 2.0, 3.0]);
        assert!(parse_place(&tok("tower mirror q"), &Nothing).is_err());
        assert!(parse_place(&tok(""), &Nothing).is_err());
        // A word that is not an option keyword belongs to the NAME - prefab names have spaces
        // in them ("Double Tower"), so the positional scan must not stop at the first space. A
        // mistyped option therefore becomes part of the name and fails later, by name, which is
        // a clearer message than "unknown option" would be.
        let (name, _, _) = parse_place(&tok("Double Tower noquata"), &Nothing).unwrap();
        assert_eq!(name, "Double Tower noquata");
        // but a real keyword still ends the name, wherever it appears
        let (name, o, _) = parse_place(&tok("Double Tower yaw 45"), &Nothing).unwrap();
        assert_eq!((name.as_str(), o.xform.yaw_deg), ("Double Tower", 45.0));
        // `place` refuses with no variant open, and says so
        let mut n = Nothing;
        let rep = place(&mut n, &sample(), &PlaceOptions::default());
        assert!(rep.refused.as_deref().unwrap_or_default().contains("no map variant is open"), "{rep:?}");
        assert!(rep.summary().contains("NOT placed"));
    }

    /// Any installed Reach variant with room to grow, for the save-and-reopen gate below.
    fn a_reach_variant_with_room() -> Option<(std::path::PathBuf, Vec<u8>, crate::mvar::Variant)> {
        for d in crate::mapcat::variant_dirs_of(Game::Reach) {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut files: Vec<std::path::PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")))
                .collect();
            files.sort();
            for p in files {
                if crate::h4::mvar::is_h4_variant(&p) || crate::h3::mvar::is_h3_variant(&p) {
                    continue;
                }
                let Some(v) = crate::mvar::parse_variant(&p) else { continue };
                let Ok(data) = crate::mvar::read_expanded(&p) else { continue };
                // enough objects to make a prefab from, and enough free slots for two copies
                if v.objects.len() >= 4 && v.objects.len() + 8 <= 651 {
                    return Some((p, data, v));
                }
            }
        }
        None
    }

    /// THE END-TO-END GATE, through HMS's own byte-exact Reach codec: build a prefab out of real
    /// objects in a real shipped variant, place it TWICE at different positions and rotations
    /// (one of them mirrored), write the file with the real encoder, READ IT BACK with the real
    /// reader, and assert numerically that each copy is geometrically identical to the original.
    ///
    /// Skipped with a message when no Reach install is present, so the suite still runs dry.
    #[test]
    fn reach_prefab_survives_a_real_mvar_save_and_reopen() {
        let Some((src_path, data, v)) = a_reach_variant_with_room() else {
            eprintln!("reach_prefab_survives_a_real_mvar_save_and_reopen: no installed Reach variant, nothing to assert");
            return;
        };
        // --- capture: four real objects become a prefab, anchored on their own footprint
        let picked: Vec<crate::mvar::PlacedObject> = v.objects.iter().take(4).cloned().collect();
        let anchor = anchor_of(&picked.iter().map(|o| o.pos).collect::<Vec<_>>());
        let pf = Prefab {
            name: "roundtrip".into(),
            game: "reach".into(),
            canvas: "test".into(),
            objects: picked
                .iter()
                .map(|o| PrefabObject {
                    key: format!("q{}v{}", o.folder, o.item),
                    display: format!("quota {}.{}", o.folder, o.item),
                    pos: [o.pos[0] - anchor[0], o.pos[1] - anchor[1], o.pos[2] - anchor[2]],
                    fwd: o.fwd,
                    up: o.up,
                    team: if o.team == 0xFF { -1 } else { o.team.min(8) as i16 },
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        // it goes through the FILE, like a shared prefab would
        let (pf, warn) = parse(&to_json(&pf)).expect("the prefab must survive its own format");
        assert!(warn.is_empty(), "{warn:?}");

        // --- place it twice. The second copy is yawed AND mirrored, which is the case most
        // likely to go wrong (a mirror flips handedness; a yaw must still compose with it).
        let shots = [
            Xform { origin: [anchor[0] + 30.0, anchor[1], anchor[2]], yaw_deg: 0.0, mirror: None },
            Xform { origin: [anchor[0] - 30.0, anchor[1] + 15.0, anchor[2] + 2.0], yaw_deg: 90.0, mirror: Some(1) },
        ];
        let mut objects = v.objects.clone();
        let mut expect: Vec<Vec<[f32; 3]>> = Vec::new();
        for x in shots {
            let mut want = Vec::new();
            for (src, po) in picked.iter().zip(pf.objects.iter()) {
                let (pos, fwd, up) = x.apply(po.pos, po.fwd, po.up);
                // exactly what `pf_place` does, minus the editor: the SAME palette piece, the
                // transformed pose, a fresh spawnable record.
                let mut o = crate::mvar::new_placed_object(src.folder, src.item, pos, fwd, up, src.team, src.color);
                o.cached_type = src.cached_type;
                want.push(o.pos);
                objects.push(o);
            }
            expect.push(want);
        }

        // --- write it with the real encoder, read it back with the real reader
        let (bytes, written) = crate::mvar::rebuild_file(&data, &objects, None).expect("re-encode");
        assert_eq!(written, objects.len(), "every object must have reached a slot");
        let dir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join("hms-prefab-gate");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let out = dir.join("prefab_roundtrip.mvar");
        std::fs::write(&out, &bytes).expect("write");
        let back = crate::mvar::parse_variant(&out).expect("HMS must be able to read back what it wrote");
        assert_eq!(back.map_id, v.map_id, "the variant still targets its own map");
        assert_eq!(back.objects.len(), objects.len(), "{} objects went in, {} came back", objects.len(), back.objects.len());

        // --- THE ASSERTION: each copy is the original, rigidly moved.
        let n = picked.len();
        let base = back.objects.len() - 2 * n;
        for (c, want) in expect.iter().enumerate() {
            let copy = &back.objects[base + c * n..base + (c + 1) * n];
            for (i, o) in copy.iter().enumerate() {
                // the same palette piece, still spawnable
                assert_eq!((o.folder, o.item), (picked[i].folder, picked[i].item), "copy {c} object {i} changed piece");
                assert_eq!(o.flags & 1, 1, "copy {c} object {i} is not an occupied slot");
                assert_eq!(o.placement, crate::mvar::PLACEMENT_DEFAULT_SPAWNABLE, "copy {c} object {i} would not spawn");
                // the position the file holds is the intended one to within ONE quantisation
                // step of the variant's own position grid (which is what any save costs)
                for k in 0..3 {
                    assert!((o.pos[k] - want[i][k]).abs() < 0.05, "copy {c} object {i} axis {k}: {} vs wanted {}", o.pos[k], want[i][k]);
                }
            }
            // GEOMETRY: every pairwise distance inside the copy equals the original's
            for i in 0..n {
                for j in 0..n {
                    let a = (Vec3::from(picked[i].pos) - Vec3::from(picked[j].pos)).length();
                    let b = (Vec3::from(copy[i].pos) - Vec3::from(copy[j].pos)).length();
                    assert!((a - b).abs() < 0.12, "copy {c}: distance {i}-{j} is {b}, the original is {a}");
                }
            }
        }
        eprintln!(
            "reach save/reopen gate: {} -> {} ({} objects, {n} per copy x 2 copies) read back and geometrically identical",
            src_path.file_name().unwrap_or_default().to_string_lossy(),
            out.display(),
            back.objects.len()
        );
    }

    /// A whole capture -> file -> place cycle over a fake scene: the placed objects come back at
    /// the right world poses, as one selection, and the prefab is byte-stable on disk.
    #[test]
    fn capture_and_place_over_a_fake_scene() {
        struct Fake {
            placed: Vec<(usize, PrefabObject)>,
            sel: Vec<u32>,
            objs: Vec<(u32, usize, PrefabObject)>,
            undos: usize,
            /// Fail the Nth creation (1-based), to prove the rollback.
            fail_on: Option<usize>,
            removed: Vec<u32>,
        }
        impl PrefabScene for Fake {
            fn pf_target(&self) -> Option<(Game, String)> {
                Some((Game::Reach, "ca_forge_ravine".into()))
            }
            fn pf_palette(&self) -> Vec<PalRow> {
                pal_rows()
            }
            fn pf_capacity(&self) -> Capacity {
                Capacity { slots_used: 3, slots_max: 651, budget_priced: false, ..Default::default() }
            }
            fn pf_selection(&self) -> Vec<u32> {
                self.sel.clone()
            }
            fn pf_set_selection(&mut self, d: Vec<u32>) {
                self.sel = d;
            }
            fn pf_read(&self, datum: u32) -> Option<(Option<usize>, PrefabObject)> {
                // A real adapter leaves the catalog identity blank and lets `capture` fill it
                // from the palette row; mirror that so the test exercises the same path.
                self.objs.iter().find(|(d, _, _)| *d == datum).map(|(_, i, o)| {
                    let mut o = o.clone();
                    o.key.clear();
                    o.display.clear();
                    o.spot = None;
                    (Some(*i), o)
                })
            }
            fn pf_place(&mut self, pal: usize, o: &PrefabObject) -> Result<u32, String> {
                if self.fail_on == Some(self.placed.len() + 1) {
                    return Err("no render model".into());
                }
                self.placed.push((pal, o.clone()));
                Ok(0xD800_0000 + self.placed.len() as u32)
            }
            fn pf_remove(&mut self, datums: &[u32]) {
                // The fake hands out datums in creation order, so a rollback drops exactly that
                // many from the tail of what it made.
                self.removed.extend_from_slice(datums);
                let keep = self.placed.len().saturating_sub(datums.len());
                self.placed.truncate(keep);
            }
            fn pf_labels(&self) -> Vec<String> {
                vec!["scale".into()]
            }
            fn pf_push_undo(&mut self) {
                self.undos += 1;
            }
            fn pf_drop_point(&self) -> [f32; 3] {
                [0.0; 3]
            }
        }
        // three objects in the world, 10 wu up and 50 wu out
        let mut world = sample();
        for o in world.objects.iter_mut() {
            o.pos = [o.pos[0] + 50.0, o.pos[1] - 20.0, o.pos[2] + 10.0];
        }
        world.objects[2].labels = vec!["scale".into(), "ctf".into()];
        world.objects[2].scaled = Some(true);
        let mut fake = Fake {
            placed: Vec::new(),
            sel: vec![1, 2, 3],
            objs: world.objects.iter().enumerate().map(|(i, o)| (i as u32 + 1, i, o.clone())).collect(),
            undos: 0,
            fail_on: None,
            removed: Vec::new(),
        };
        let pf = capture(&fake, "Double Tower").expect("capture");
        assert_eq!(pf.name, "Double Tower");
        assert_eq!(pf.game, "reach");
        assert_eq!(pf.objects.len(), 3);
        // anchored on its own footprint, not on the map
        assert_eq!(anchor_of(&pf.objects.iter().map(|o| o.pos).collect::<Vec<_>>()), [0.0, 0.0, 0.0]);
        // identity came from the palette rows
        assert_eq!(pf.objects[2].spot, Some([1, 2, 0]));
        assert_eq!(pf.objects[2].display, "Coliseum Wall");
        // it survives the file
        let (pf, _) = parse(&to_json(&pf)).expect("file");

        let opt = PlaceOptions { xform: Xform { origin: [-200.0, 0.0, 4.0], yaw_deg: 90.0, mirror: None }, ..Default::default() };
        let rep = place(&mut fake, &pf, &opt);
        assert!(rep.refused.is_none(), "{}", rep.describe());
        assert_eq!(rep.placed.len(), 3);
        assert_eq!(fake.undos, 1, "one undo step for the whole prefab");
        assert_eq!(fake.sel, rep.placed, "the prefab is selected as a group right after placement");
        assert_eq!(rep.solid(), 3);
        assert_eq!(rep.approximate(), 0);
        assert_eq!(rep.unmatched(), 0);
        // the unknown label was dropped and REPORTED; the known one stayed
        assert_eq!(rep.labels_dropped, vec!["ctf".to_string()]);
        assert_eq!(fake.placed[2].1.labels, vec!["scale".to_string()]);
        assert_eq!(fake.placed[2].1.scaled, Some(true), "the pseudo-flags ride along");
        // GEOMETRY: the placed cloud is the captured cloud, rigidly moved
        for i in 0..3 {
            for j in 0..3 {
                let a = (Vec3::from(pf.objects[i].pos) - Vec3::from(pf.objects[j].pos)).length();
                let b = (Vec3::from(fake.placed[i].1.pos) - Vec3::from(fake.placed[j].1.pos)).length();
                assert!((a - b).abs() < 1e-3, "distance {i}-{j}: {a} vs {b}");
            }
        }
        // a 90 degree yaw about +Z: (x, y) -> (-y, x), then the origin
        let e = &fake.placed[1].1;
        assert!((e.pos[0] - (-(pf.objects[1].pos[1]) + -200.0)).abs() < 1e-3, "{e:?}");
        assert!((e.pos[1] - (pf.objects[1].pos[0] + 0.0)).abs() < 1e-3, "{e:?}");
        assert!((e.pos[2] - (pf.objects[1].pos[2] + 4.0)).abs() < 1e-3, "{e:?}");
        assert_eq!(rep.summary().contains("yaw 90"), true, "{}", rep.summary());

        // placing it again MIRRORED gives the mirror image, still rigid, still one undo step
        fake.placed.clear();
        let opt = PlaceOptions { xform: Xform { origin: [0.0; 3], yaw_deg: 0.0, mirror: Some(0) }, ..Default::default() };
        let rep = place(&mut fake, &pf, &opt);
        assert!(rep.refused.is_none(), "{}", rep.describe());
        for i in 0..3 {
            assert!((fake.placed[i].1.pos[0] + pf.objects[i].pos[0]).abs() < 1e-4);
            assert!((fake.placed[i].1.pos[1] - pf.objects[i].pos[1]).abs() < 1e-4);
        }
        assert!(rep.summary().contains("mirrored across X"), "{}", rep.summary());

        // and a dry run changes nothing
        fake.placed.clear();
        let before = fake.undos;
        let rep = place(&mut fake, &pf, &PlaceOptions { dry_run: true, ..Default::default() });
        assert!(fake.placed.is_empty() && fake.undos == before, "a dry run must not touch the scene");
        assert_eq!(rep.objects.len(), 3);

        // ALL OR NOTHING: a creation that fails half way through rolls the rest back, refuses the
        // whole placement and says which object failed. Half a structure must never be left behind.
        fake.placed.clear();
        fake.removed.clear();
        fake.fail_on = Some(3);
        let rep = place(&mut fake, &pf, &PlaceOptions::default());
        let why = rep.refused.as_deref().unwrap_or_default();
        assert!(why.contains("object 3 of 3") && why.contains("nothing was placed"), "{why}");
        assert_eq!(fake.removed.len(), 2, "the two objects already created must be removed");
        assert!(fake.placed.is_empty(), "nothing may survive a refused placement");
        assert!(rep.objects.iter().all(|o| o.datum.is_none()), "the report must not claim datums that were rolled back");
        assert!(rep.summary().contains("NOT placed"), "{}", rep.summary());
    }
}
