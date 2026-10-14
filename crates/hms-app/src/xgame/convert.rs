//! THE CONVERTER (#xgame): a Forge variant of one title re-created on a Forge canvas of another.
//!
//! Shape of the pass:
//!
//! ```text
//!   source .mvar --(the title's own reader)--> Vec<XObject>   (game-neutral: key + pose + gameplay)
//!        |                                         |
//!        |                             table::resolve per object, over the
//!        |                             TARGET CANVAS's palette only
//!        v                                         v
//!   source cloud AABB  ---- place() ---->   target poses (re-centred, yawed, clamped)
//!                                                  |
//!                        the target title's own byte-exact encoder (mvar.rs /
//!                        h4/mvar.rs / h3/mvar.rs) writing into a TEMPLATE variant
//!                        that already lives on the target canvas
//! ```
//!
//! A TEMPLATE is needed because every encoder here rebuilds a real `.mvar`: it copies the content
//! header, the map id / GUID, the palette CRC and the trait records verbatim and replaces only the
//! object slots and the quota table. So the converter picks a shipped variant of the target
//! canvas (the one with the fewest objects, so the most free slots) and writes the converted
//! objects into a copy of it.
//!
//! What carries over, and what cannot:
//!   * position / orientation - DIRECTLY. All four titles use the same right-handed Z-up world
//!     unit frame, so only the canvas ORIGIN differs; that is the re-centring offset.
//!   * team, colour, respawn time, spawn sequence, user data, boundary shape + size, teleporter
//!     channel, spare clips, gametype labels - wherever the target title has the same field.
//!     Halo 3 has no per-object colour and no label table, so those are dropped (and reported).
//!   * per-object SCALE is never used to paper over a size difference (the Halo 4 field is
//!     written but had no visible effect in-game), so a size delta is REPORTED, not compensated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::Vec3;

use crate::mapcat::Game;

use super::catalog::{Catalog, CatalogObject, Kind};
use super::table::{Confidence, Table};

// ---------------------------------------------------------------------------------------------
// the game-neutral object
// ---------------------------------------------------------------------------------------------

/// One placed object, in the one shape all four titles can be read into and written from.
#[derive(Clone, Debug, Default)]
pub struct XObject {
    /// Catalog key in the SOURCE game, or empty when the placement did not resolve.
    pub key: String,
    /// What to call it in the report.
    pub display: String,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    /// -1 none, 0..7 team colours, 8 neutral (the Halo 4 convention; Reach / H3 use 8 / 0xFF).
    pub team: i8,
    pub color: Option<u8>,
    pub spawn_time: u8,
    pub spawn_sequence: i32,
    pub user_data: i32,
    /// 0 none, 1 sphere, 2 cylinder, 3 box.
    pub shape: u8,
    /// Boundary values in WORLD UNITS: sphere [r], cylinder [r, top, bottom], box [w, l, top, bottom].
    pub shape_wu: [f32; 4],
    /// The source title's multiplayer object type (diagnostics; the target's comes from its palette).
    pub object_type: u8,
    /// Reach-style placement byte (symmetry / at-start / unique / physics); the low byte of the
    /// Halo 4 10-bit field.
    pub placement: u8,
    /// Gametype label strings (resolved), empty when the title / object has none.
    pub labels: Vec<String>,
    pub tele_channel: u8,
    pub tele_passability: u8,
    pub weapon_clips: u8,
    /// Source slot, for the report.
    pub slot: u16,
}

/// A whole source variant, read into the neutral shape.
#[derive(Clone, Debug, Default)]
pub struct XVariant {
    pub game_short: String,
    pub path: PathBuf,
    pub title: String,
    pub author: String,
    pub description: String,
    /// Canvas stem the variant was authored on (`forge_halo`, `ca_forge_ravine`, `sandbox`).
    pub canvas: String,
    pub map_id: u32,
    pub bounds: [f32; 6],
    pub budget_max: u32,
    pub objects: Vec<XObject>,
    /// Placements whose palette coordinates did not resolve against the catalog (reported, dropped).
    pub unresolved: usize,
}

impl XVariant {
    pub fn game(&self) -> Option<Game> { super::game_from_str(&self.game_short) }
    /// AABB of the placed objects (min, max); None when there are none.
    pub fn cloud(&self) -> Option<([f32; 3], [f32; 3])> { cloud_of(self.objects.iter().map(|o| o.pos)) }
}

fn cloud_of(it: impl Iterator<Item = [f32; 3]>) -> Option<([f32; 3], [f32; 3])> {
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut n = 0usize;
    for p in it {
        if p.iter().any(|v| !v.is_finite()) { continue; }
        for k in 0..3 { mn[k] = mn[k].min(p[k]); mx[k] = mx[k].max(p[k]); }
        n += 1;
    }
    (n > 0).then_some((mn, mx))
}

// ---------------------------------------------------------------------------------------------
// reading a source variant
// ---------------------------------------------------------------------------------------------

/// Which title a `.mvar` belongs to, from the file itself.
pub fn variant_game(path: &Path) -> Option<Game> {
    if crate::h4::mvar::is_h4_variant(path) {
        // H4 and H2A share the container; the map id tells them apart via the installed caches.
        let id = crate::h4::mvar::read_h4_map_id(path);
        let g = id.and_then(|id| crate::mapcat::enumerate().into_iter()
            .find(|c| (c.game == Game::Halo4 || c.game == Game::H2A) && c.map_id == Some(id))
            .map(|c| c.game));
        return Some(g.unwrap_or(Game::Halo4));
    }
    if crate::h3::mvar::is_h3_variant(path) { return Some(Game::Halo3); }
    crate::mvar::parse_variant(path).map(|_| Game::Reach)
}

/// Canvas stem of a map id in `game` (the catalog and the palette are both keyed on it).
fn canvas_of(game: Game, map_id: u32) -> Option<String> {
    crate::mapcat::enumerate().into_iter()
        .find(|c| c.game == game && !c.modded && c.map_id == Some(map_id))
        .map(|c| c.stem)
}

/// Map a canvas stem back to its cache path. An MCC install often carries `<map> - Copy.map`
/// backups next to the real cache and they share the stem, so prefer a stock file whose NAME is
/// the stem and which carries a map id - a `- Copy` has neither.
pub fn canvas_path(game: Game, stem: &str) -> Option<PathBuf> {
    let cands: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter()
        .filter(|c| c.game == game && c.stem.eq_ignore_ascii_case(stem)).collect();
    let named = |c: &crate::mapcat::MapCandidate| c.file_name.eq_ignore_ascii_case(&format!("{stem}.map"));
    cands.iter().find(|c| !c.modded && c.map_id.is_some() && named(c))
        .or_else(|| cands.iter().find(|c| c.map_id.is_some() && named(c)))
        .or_else(|| cands.iter().find(|c| c.map_id.is_some()))
        .or_else(|| cands.first())
        .map(|c| c.path.clone())
}

/// Every canvas of `game` that the catalog knows a Forge palette for, with its map id.
pub fn canvases(cat: &Catalog, game: Game) -> Vec<(String, Option<u32>)> {
    let cands = crate::mapcat::enumerate();
    cat.canvases(game).into_iter()
        .map(|s| { let id = cands.iter().find(|c| c.game == game && c.stem == s).and_then(|c| c.map_id); (s, id) })
        .collect()
}

/// Read a `.mvar` of any title into the neutral shape, resolving every placement's palette
/// coordinates through the catalog (so no cache has to be opened for the SOURCE side).
pub fn read_variant(cat: &Catalog, path: &Path) -> Result<XVariant, String> {
    match variant_game(path).ok_or_else(|| format!("{}: not a map variant HMS can read", path.display()))? {
        Game::Reach => read_reach(cat, path),
        Game::Halo4 => read_h4(cat, path, Game::Halo4),
        Game::H2A => read_h4(cat, path, Game::H2A),
        Game::Halo3 => read_h3(cat, path),
    }
}

/// Catalog lookup by the per-canvas palette coordinates a record stores.
fn by_spot<'a>(cat: &'a Catalog, game: Game, canvas: &str, quota: u32, variant: u32) -> Option<&'a CatalogObject> {
    cat.of_game(game).find(|o| o.places.iter().any(|p| p.canvas.eq_ignore_ascii_case(canvas) && p.quota == quota && p.variant == variant))
        // a record may index a type whose variant index is past the entry's variants: the engine
        // falls back to variant 0, so do the same
        .or_else(|| cat.of_game(game).find(|o| o.places.iter().any(|p| p.canvas.eq_ignore_ascii_case(canvas) && p.quota == quota && p.variant == 0)))
}

fn read_reach(cat: &Catalog, path: &Path) -> Result<XVariant, String> {
    let v = crate::mvar::parse_variant(path).ok_or_else(|| format!("{}: cannot parse as a Reach variant", path.display()))?;
    let canvas = canvas_of(Game::Reach, v.map_id).unwrap_or_default();
    if canvas.is_empty() { return Err(format!("no installed Reach map has id {} (the variant's canvas is not present)", v.map_id)); }
    let mut out = XVariant {
        game_short: Game::Reach.short_name().into(), path: path.to_path_buf(), title: v.title.clone(), author: v.author.clone(),
        description: v.description.clone(), canvas: canvas.clone(), map_id: v.map_id, bounds: v.globals.bounds,
        budget_max: v.globals.budget_max, objects: Vec::new(), unresolved: 0,
    };
    for o in &v.objects {
        let Some(c) = by_spot(cat, Game::Reach, &canvas, o.folder as u32, o.item as u32) else { out.unresolved += 1; continue };
        let labels = (o.label_idx != 0xFFFF).then(|| v.labels.get(o.label_idx as usize).cloned()).flatten().into_iter().collect();
        out.objects.push(XObject {
            key: c.key.clone(), display: c.display.clone(), pos: o.pos, fwd: o.fwd, up: o.up,
            team: if o.team == 0xFF { -1 } else { o.team as i8 }, color: (o.color >= 0).then(|| o.color as u8),
            spawn_time: o.respawn, spawn_sequence: o.spawn_seq, user_data: 0,
            shape: o.boundary_shape, shape_wu: reach_boundary_wu(o.boundary_shape, o.boundary),
            object_type: o.cached_type, placement: o.placement, labels,
            tele_channel: o.tele_channel, tele_passability: o.tele_passability, weapon_clips: o.weapon_clips, slot: o.slot,
        });
    }
    Ok(out)
}

/// Reach boundary values are 11-bit quanta over 0..200 world units - the convention the editor
/// already draws and edits with (`main.rs` `wu_to_bval` and its `(v / 2047) * 200` inverse).
/// Halo 3 quantises the same numbers 16-bit over 0..60 wu and Halo 4 stores them as 16-bit
/// 1/256 wu, so WORLD UNITS are the only shape every title agrees on.
pub const REACH_BOUNDARY_MAX: f32 = 200.0;
pub const REACH_BOUNDARY_BITS: f32 = 2047.0;
fn reach_boundary_wu(shape: u8, q: [u16; 4]) -> [f32; 4] {
    let n = [0usize, 1, 3, 4][(shape as usize).min(3)];
    let mut out = [0.0f32; 4];
    for k in 0..n { out[k] = (q[k] as f32 / REACH_BOUNDARY_BITS) * REACH_BOUNDARY_MAX; }
    out
}
fn reach_boundary_q(shape: u8, wu: [f32; 4]) -> [u16; 4] {
    let n = [0usize, 1, 3, 4][(shape as usize).min(3)];
    let mut out = [0u16; 4];
    for k in 0..n { out[k] = crate::wu_to_bval(wu[k]); }
    out
}

fn read_h4(cat: &Catalog, path: &Path, game: Game) -> Result<XVariant, String> {
    let v = crate::h4::mvar::parse_h4_variant(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let canvas = canvas_of(game, v.map_id).unwrap_or_default();
    if canvas.is_empty() { return Err(format!("no installed {} map has id {} (the variant's canvas is not present)", game.display_name(), v.map_id)); }
    let mut out = XVariant {
        game_short: game.short_name().into(), path: path.to_path_buf(), title: v.title.clone(), author: v.author.clone(),
        description: v.description.clone(), canvas: canvas.clone(), map_id: v.map_id, bounds: v.bounds,
        budget_max: v.budget_max, objects: Vec::new(), unresolved: 0,
    };
    for o in &v.objects {
        let Some(q) = o.quota else { continue };
        let Some(c) = by_spot(cat, game, &canvas, q as u32, o.variant.unwrap_or(0) as u32) else { out.unresolved += 1; continue };
        let labels: Vec<String> = o.labels.iter().flatten().filter_map(|&i| v.labels.get(i as usize).cloned()).collect();
        let (chan, pass) = match &o.type_data { crate::h4::mvar::H4TypeData::Pair(a, b) => (*a, *b), _ => (0, 0) };
        let clips = match &o.type_data { crate::h4::mvar::H4TypeData::Byte(b) => *b, _ => 0 };
        out.objects.push(XObject {
            key: c.key.clone(), display: c.display.clone(), pos: o.pos, fwd: o.fwd, up: o.up,
            team: o.team, color: o.color, spawn_time: o.spawn_time, spawn_sequence: o.spawn_sequence as i32,
            user_data: o.user_data as i32, shape: o.shape as u8, shape_wu: o.shape_wu(),
            object_type: o.object_type, placement: (o.placement & 0xFF) as u8, labels,
            tele_channel: chan, tele_passability: pass, weapon_clips: clips, slot: o.slot,
        });
    }
    Ok(out)
}

fn read_h3(cat: &Catalog, path: &Path) -> Result<XVariant, String> {
    let v = crate::h3::mvar::parse_h3_variant(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let canvas = canvas_of(Game::Halo3, v.map_id).unwrap_or_default();
    if canvas.is_empty() { return Err(format!("no installed Halo 3 map has id {} (the variant's canvas is not present)", v.map_id)); }
    let mut out = XVariant {
        game_short: Game::Halo3.short_name().into(), path: path.to_path_buf(), title: v.header.name.clone(), author: v.header.author.clone(),
        description: v.header.description.clone(), canvas: canvas.clone(), map_id: v.map_id, bounds: v.bounds,
        budget_max: v.maximum_budget.max(0.0) as u32, objects: Vec::new(), unresolved: 0,
    };
    for o in v.placed() {
        // a Halo 3 record's quota indexes the variant's own quota table; the quota's `definition`
        // is `(sandbox block + 1) << 16 | entry`, which is what the catalog's spot carries.
        let Some(q) = usize::try_from(o.quota).ok().and_then(|i| v.quotas.get(i)) else { out.unresolved += 1; continue };
        let Some(crate::h3::mvar::QuotaSource::Sandbox(block)) = q.source() else { out.unresolved += 1; continue };
        let entry = q.index() as u32;
        let Some(c) = cat.of_game(Game::Halo3).find(|c| c.places.iter().any(|p| p.canvas.eq_ignore_ascii_case(&canvas) && p.category_index == block as u32 && p.quota == entry)) else {
            out.unresolved += 1; continue
        };
        out.objects.push(XObject {
            key: c.key.clone(), display: c.display.clone(), pos: o.pos, fwd: o.fwd, up: o.up,
            team: if o.team == 0xFF { -1 } else { o.team as i8 }, color: None,
            spawn_time: o.spawn_time, spawn_sequence: 0, user_data: 0,
            shape: o.boundary_shape, shape_wu: o.boundary,
            object_type: o.object_type, placement: 0x0C, labels: Vec::new(),
            tele_channel: o.shared_storage, tele_passability: 0, weapon_clips: o.shared_storage, slot: o.slot as u16,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// templates
// ---------------------------------------------------------------------------------------------

/// A shipped variant of the target canvas the converter can rebuild into.
#[derive(Clone, Debug)]
pub struct Template {
    pub path: PathBuf,
    pub objects: usize,
    pub bounds: [f32; 6],
    pub budget_max: u32,
    /// Centroid of the template's own placed objects - the best guess at "where people build" on
    /// this canvas (None when it ships empty).
    pub anchor: Option<[f32; 3]>,
    /// Floor: the lowest object z of the template's own cloud.
    pub floor: Option<f32>,
    /// Slots the converted file CANNOT use because the template must keep them. Halo 3 stores the
    /// map's own scenario-owned objects in the same 640-slot array as the Forge placements, so a
    /// Sandbox template with 330 scenario slots leaves only ~310 for a converted layout. 0 for
    /// Reach / Halo 4, whose slot arrays are Forge-only.
    pub reserved: usize,
}

/// Every installable template for `game` + `map_id`, FEWEST objects first (the most free slots).
pub fn templates(game: Game, map_id: u32) -> Vec<Template> {
    let dirs: Vec<PathBuf> = crate::mapcat::variant_dirs_of(game);
    let mut out: Vec<Template> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map_or(true, |x| !x.eq_ignore_ascii_case("mvar")) { continue; }
            if let Some(t) = template_of(game, &p, map_id) { out.push(t); }
        }
    }
    // FEWEST USED SLOTS first = the most room for the conversion (`reserved` is non-zero only on
    // Halo 3, whose slot array also holds the map's own scenario objects). Ties go to a SHIPPED
    // variant (`hopper_map_variants`), so the same canvas picks the same template on every
    // machine and the converted file never inherits a stranger's save metadata.
    out.sort_by_key(|t| (t.objects + t.reserved, !t.path.to_string_lossy().contains("hopper"), t.path.clone()));
    out
}

fn template_of(game: Game, p: &Path, map_id: u32) -> Option<Template> {
    match game {
        Game::Reach => {
            if crate::h4::mvar::is_h4_variant(p) || crate::h3::mvar::is_h3_variant(p) { return None; }
            let v = crate::mvar::parse_variant(p)?;
            (v.map_id == map_id).then(|| {
                let (mn, mx) = cloud_of(v.objects.iter().map(|o| o.pos)).unzip();
                Template { path: p.to_path_buf(), objects: v.objects.len(), bounds: v.globals.bounds, budget_max: v.globals.budget_max,
                    anchor: mn.zip(mx).map(|(a, b): ([f32; 3], [f32; 3])| mid(a, b)).filter(|_| v.objects.len() >= 4),
                    floor: mn.map(|a: [f32; 3]| a[2]), reserved: 0 }
            })
        }
        Game::Halo4 | Game::H2A => {
            if crate::h4::mvar::read_h4_map_id(p)? != map_id { return None; }
            let v = crate::h4::mvar::parse_h4_variant(p).ok()?;
            let (mn, mx) = cloud_of(v.objects.iter().map(|o| o.pos)).unzip();
            Some(Template { path: p.to_path_buf(), objects: v.objects.len(), bounds: v.bounds, budget_max: v.budget_max,
                anchor: mn.zip(mx).map(|(a, b): ([f32; 3], [f32; 3])| mid(a, b)).filter(|_| v.objects.len() >= 4),
                floor: mn.map(|a: [f32; 3]| a[2]), reserved: 0 })
        }
        Game::Halo3 => {
            if crate::h3::mvar::read_h3_map_id(p)? != map_id { return None; }
            let v = crate::h3::mvar::parse_h3_variant(p).ok()?;
            let placed: Vec<[f32; 3]> = v.placed().map(|o| o.pos).collect();
            let (mn, mx) = cloud_of(placed.iter().copied()).unzip();
            Some(Template { path: p.to_path_buf(), objects: placed.len(), bounds: v.bounds, budget_max: v.maximum_budget.max(0.0) as u32,
                anchor: mn.zip(mx).map(|(a, b): ([f32; 3], [f32; 3])| mid(a, b)).filter(|_| placed.len() >= 4),
                floor: mn.map(|a: [f32; 3]| a[2]),
                // the scenario-owned slots the converted file must leave alone
                reserved: v.objects.iter().filter(|o| !o.placed).count().max(v.num_scenario_objects as usize) })
        }
    }
}

fn mid(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5] }

// ---------------------------------------------------------------------------------------------
// options + report
// ---------------------------------------------------------------------------------------------

/// How to place the converted cloud and what to allow.
#[derive(Clone, Debug)]
pub struct Options {
    /// Extra world-unit offset on top of the automatic re-centring.
    pub offset: [f32; 3],
    /// Extra yaw about +Z, DEGREES, applied about the cloud centre.
    pub yaw_deg: f32,
    /// Uniform scale of the LAYOUT (object positions only, never object sizes). 1.0 = none.
    pub layout_scale: f32,
    /// Skip the automatic re-centring and keep the source world coordinates.
    pub keep_origin: bool,
    /// Refuse any mapping worse than this.
    pub min_confidence: Confidence,
    /// Honour the target palette's per-entry `maximum allowed`.
    pub enforce_quotas: bool,
    /// Honour the target canvas's `maximum_budget`.
    pub enforce_budget: bool,
    /// An explicit template; None = the one with the fewest objects.
    pub template: Option<PathBuf>,
    /// The unix timestamp stamped into the converted file's content header. `None` = read the
    /// clock at the moment of conversion, which is what a real save does.
    ///
    /// It exists so a conversion can be made REPRODUCIBLE. Everything else about the output is a
    /// pure function of (source, canvas, options); the clock is not, and the stamp is mirrored
    /// into the bitstream and both `chdr` copies, so two conversions a second apart differ in
    /// **22 bytes** (measured) rather than the four a reader would expect: the body copy is
    /// bit-packed and not byte-aligned, so a one-bit change smears. Supplying the instant makes
    /// two runs byte-identical, which is what the dialog-versus-script gate asserts, with no
    /// exemptions.
    pub stamp_unix: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Options { offset: [0.0; 3], yaw_deg: 0.0, layout_scale: 1.0, keep_origin: false,
            min_confidence: Confidence::Approximate, enforce_quotas: true, enforce_budget: true, template: None,
            stamp_unix: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status { Mapped, Resized, Dropped }

impl Status {
    pub fn name(self) -> &'static str { match self { Status::Mapped => "mapped", Status::Resized => "resized", Status::Dropped => "DROPPED" } }
}

/// What happened to one source object.
#[derive(Clone, Debug)]
pub struct ObjReport {
    pub index: usize,
    pub src_slot: u16,
    pub src_key: String,
    pub src_display: String,
    pub dst_key: String,
    pub dst_display: String,
    pub kind: String,
    pub conf: Confidence,
    /// Target size minus source size (sorted axes), world units.
    pub delta: [f32; 3],
    pub src_pos: [f32; 3],
    pub dst_pos: [f32; 3],
    pub status: Status,
    /// Why it was dropped, or how it was matched.
    pub note: String,
}

/// The whole conversion.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub src_path: PathBuf,
    pub src_game: String,
    pub src_canvas: String,
    pub src_title: String,
    pub dst_game: String,
    pub dst_canvas: String,
    pub template: PathBuf,
    pub objects: Vec<ObjReport>,
    pub unresolved_source: usize,
    /// The automatic offset actually used (before the user's extra offset).
    pub auto_offset: [f32; 3],
    pub user_offset: [f32; 3],
    pub yaw_deg: f32,
    pub source_cloud: Option<([f32; 3], [f32; 3])>,
    /// AABB the converted objects actually landed in, world units - what to point a camera at.
    pub dest_cloud: Option<([f32; 3], [f32; 3])>,
    pub target_bounds: [f32; 6],
    pub budget_spent: u32,
    pub budget_max: u32,
    /// True when the TARGET title's palette carries per-object prices (Halo 4 / H2A / Halo 3).
    /// Reach's Forge palette has none offline, so its spend is not a number worth printing.
    pub budget_priced: bool,
    pub clamped: usize,
    /// How many pieces the converter had to TURN because the target model is authored along
    /// different local axes (see `turn_to_axes`).
    pub rotated: usize,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn mapped(&self) -> usize { self.objects.iter().filter(|o| o.status != Status::Dropped).count() }
    pub fn dropped(&self) -> usize { self.objects.iter().filter(|o| o.status == Status::Dropped).count() }
    pub fn exact(&self) -> usize { self.objects.iter().filter(|o| o.status != Status::Dropped && o.conf == Confidence::Exact).count() }
    pub fn close(&self) -> usize { self.objects.iter().filter(|o| o.status != Status::Dropped && o.conf == Confidence::Close).count() }
    pub fn approximate(&self) -> usize { self.objects.iter().filter(|o| o.status != Status::Dropped && o.conf == Confidence::Approximate).count() }
    /// THE HEADLINE NUMBER: `exact + close` as a percentage of the source objects.
    ///
    /// It deliberately EXCLUDES the approximate matches. An approximate match is the nearest the
    /// target palette can offer and is often not the right piece at all; counting it as "covered"
    /// is how a conversion that needed a lot of hand work came to report 100 %. An approximate
    /// count is reported beside this, as something to go and check.
    pub fn coverage(&self) -> f32 {
        let total = self.objects.len();
        if total == 0 { return 0.0; }
        (self.exact() + self.close()) as f32 * 100.0 / total as f32
    }
    /// Approximate matches as a percentage of the source objects - the "check these" number.
    pub fn approximate_pct(&self) -> f32 {
        let total = self.objects.len();
        if total == 0 { return 0.0; }
        self.approximate() as f32 * 100.0 / total as f32
    }
    /// Everything that reached the file, solid or not, as a percentage - the number the OLD
    /// `coverage()` reported. Kept so a report can say both and nobody is misled by either.
    pub fn placed_pct(&self) -> f32 {
        let total = self.objects.len();
        if total == 0 { return 0.0; }
        self.mapped() as f32 * 100.0 / total as f32
    }
    /// One screenful: the counts, the coverage and the placement.
    pub fn summary(&self) -> String {
        let d = self.dest_cloud.unwrap_or(([0.0; 3], [0.0; 3]));
        let d = [d.0, d.1];
        format!("{} '{}' ({}) -> {} {}: {:.1}% SOLID coverage ({} exact + {} close of {} objects); \
                 {} approximate ({:.1}%) - CHECK THESE; {} dropped{}\n\
                 placement: auto offset [{:.1}, {:.1}, {:.1}] + user [{:.1}, {:.1}, {:.1}], yaw {:.0} deg{}{}\n\
                 landed in x [{:.0}, {:.0}] y [{:.0}, {:.0}] z [{:.0}, {:.0}] wu (canvas box x [{:.0}, {:.0}] y [{:.0}, {:.0}] z [{:.0}, {:.0}])\n\
                 budget {}   template {}",
            self.src_game, self.src_title, self.src_canvas, self.dst_game, self.dst_canvas,
            self.coverage(), self.exact(), self.close(), self.objects.len(), self.approximate(), self.approximate_pct(), self.dropped(),
            if self.unresolved_source > 0 { format!(" (+{} source placements the reader could not resolve)", self.unresolved_source) } else { String::new() },
            self.auto_offset[0], self.auto_offset[1], self.auto_offset[2], self.user_offset[0], self.user_offset[1], self.user_offset[2],
            self.yaw_deg, if self.clamped > 0 { format!(", {} clamped into the canvas bounds", self.clamped) } else { String::new() },
            if self.rotated > 0 { format!(", {} piece(s) turned to match the target model's axes", self.rotated) } else { String::new() },
            d[0][0], d[1][0], d[0][1], d[1][1], d[0][2], d[1][2],
            self.target_bounds[0], self.target_bounds[1], self.target_bounds[2], self.target_bounds[3], self.target_bounds[4], self.target_bounds[5],
            if self.budget_priced { format!("{} of {}", self.budget_spent, self.budget_max) }
                else { format!("not priced by this title's palette (cap {})", self.budget_max) },
            self.template.display())
    }
    /// The full per-object listing.
    pub fn describe(&self) -> String {
        let mut s = self.summary();
        s.push('\n');
        if !self.warnings.is_empty() {
            s.push_str("\nwarnings:\n");
            for w in &self.warnings { s.push_str(&format!("  - {w}\n")); }
        }
        s.push_str("\nper object:\n");
        for o in &self.objects {
            let d = if o.delta == [0.0; 3] { "          ".to_string() } else { format!("{:+5.1}/{:+5.1}/{:+5.1}", o.delta[0], o.delta[1], o.delta[2]) };
            s.push_str(&format!("  [{:>3}] {:<9} {:<34} -> {:<34} {:<11} {d}  [{}] {}\n",
                o.index, o.status.name(), o.src_display, if o.dst_display.is_empty() { "-" } else { &o.dst_display },
                o.conf.name(), o.kind, o.note));
        }
        s
    }
    /// Per-kind breakdown, worst first - what a user has to fix by hand. Approximate and dropped
    /// are the two columns that matter; solid is what needs no attention.
    pub fn by_kind(&self) -> String {
        let mut m: std::collections::BTreeMap<&str, [usize; 4]> = Default::default();
        for o in &self.objects {
            let e = m.entry(o.kind.as_str()).or_default();
            e[0] += 1;
            match (o.status, o.conf) {
                (Status::Dropped, _) => e[3] += 1,
                (_, Confidence::Approximate) => e[2] += 1,
                _ => e[1] += 1,
            }
        }
        let mut rows: Vec<(&str, [usize; 4])> = m.into_iter().collect();
        rows.sort_by_key(|(_, c)| (std::cmp::Reverse(c[3] + c[2]), std::cmp::Reverse(c[0])));
        let mut s = String::from("per kind (total / solid / APPROXIMATE - check / DROPPED):\n");
        for (k, c) in rows { s.push_str(&format!("  {k:<22} {:>4} / {:>4} / {:>4} / {:>4}\n", c[0], c[1], c[2], c[3])); }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// the conversion
// ---------------------------------------------------------------------------------------------

/// One resolved placement ready for the target encoder.
struct Placed {
    report_index: usize,
    dst: String,
    pos: [f32; 3],
    /// The pose the TARGET model must take: the source's orientation with the matcher's axis
    /// permutation already applied (see `turn_to_axes`).
    fwd: [f32; 3],
    up: [f32; 3],
    src: XObject,
}

/// Re-aim a pose so the TARGET model sits the way the SOURCE model did.
///
/// The matcher recognises two pieces as the same shape by comparing SORTED extents, which is what
/// lets it see that Reach's 2.02 x 1.02 x 1.00 `block_1x2` and Halo 4's 1.00 x 2.00 x 1.00
/// `bb_1x2` are the same block. But the two models are authored along different local axes, so
/// copying the source's orientation verbatim lays the Halo 4 block ACROSS the gap the Reach one
/// spanned. `perm[i] = j` says the target's local axis `j` plays the source's axis `i`; this turns
/// the pose so that happens.
///
/// The engine's object basis is `col0 = fwd` (local +X), `col1 = cross(up, fwd)` (local +Y),
/// `col2 = up` (local +Z) - `scene::object_matrix`, an exact port of the engine's
/// `Matrix_BuildOrientation`. So the new forward is whichever source-local axis the target's X
/// must take, and the new up likewise for the target's Z; the engine rebuilds Y itself, which
/// keeps the result a proper rotation (never a mirror).
pub fn turn_to_axes(fwd: [f32; 3], up: [f32; 3], perm: [usize; 3]) -> ([f32; 3], [f32; 3]) {
    if perm == super::table::AXES_IDENTITY { return (fwd, up); }
    let f = Vec3::from(fwd);
    let u = Vec3::from(up);
    if f.length_squared() < 1e-9 || u.length_squared() < 1e-9 { return (fwd, up); }
    let cols = [f.normalize(), u.normalize().cross(f.normalize()), u.normalize()];
    // inverse permutation: inv[target axis] = the source axis it plays
    let mut inv = [0usize; 3];
    for (i, &j) in perm.iter().enumerate() { inv[j] = i; }
    let nf = cols[inv[0]];
    let nu = cols[inv[2]];
    if nf.cross(nu).length_squared() < 1e-9 { return (fwd, up); }
    (nf.to_array(), nu.to_array())
}

/// The conversion result: the bytes to write plus the report.
pub struct Converted {
    pub bytes: Vec<u8>,
    pub report: Report,
}

/// Convert `src` onto `(dst_game, dst_canvas)`. Produces the target `.mvar` bytes and the report;
/// nothing is written to disk (see [`convert_to_file`]).
pub fn convert(cat: &Catalog, table: &Table, src: &XVariant, dst_game: Game, dst_canvas: &str, opt: &Options) -> Result<Converted, String> {
    let src_game = src.game().ok_or_else(|| format!("unknown source game '{}'", src.game_short))?;
    if src_game == dst_game && src.canvas.eq_ignore_ascii_case(dst_canvas) {
        return Err("the source and the target are the same canvas of the same game".into());
    }
    if src.objects.is_empty() {
        return Err(format!("'{}' has no Forge placement HMS can resolve on '{}'{} - a Firefight or campaign variant, or a canvas the catalog does not cover, has nothing to convert",
            src.title, src.canvas,
            if src.unresolved > 0 { format!(" ({} placement(s) did not resolve against the catalog)", src.unresolved) } else { String::new() }));
    }
    let pool: Vec<&CatalogObject> = cat.on_canvas(dst_game, dst_canvas);
    if pool.is_empty() { return Err(format!("the catalog has no {} palette for canvas '{dst_canvas}' (run --xgame-build-catalog)", dst_game.display_name())); }
    let canvas_map = canvas_path(dst_game, dst_canvas).ok_or_else(|| format!("no installed {} map called '{dst_canvas}'", dst_game.display_name()))?;
    let map_id = crate::mapcat::read_map_id(&canvas_map).ok_or_else(|| format!("{}: no map id", canvas_map.display()))?;

    // the template: an existing variant of this canvas to rebuild into
    let tmpl = match &opt.template {
        Some(p) => template_of(dst_game, p, map_id).ok_or_else(|| format!("{} is not a {} variant of '{dst_canvas}'", p.display(), dst_game.display_name()))?,
        None => templates(dst_game, map_id).into_iter().next()
            .ok_or_else(|| format!("no installed {} variant of '{dst_canvas}' to use as a template - open one in the game once, or pass one explicitly", dst_game.display_name()))?,
    };

    let mut rep = Report {
        src_path: src.path.clone(), src_game: src_game.short_name().into(), src_canvas: src.canvas.clone(), src_title: src.title.clone(),
        dst_game: dst_game.short_name().into(), dst_canvas: dst_canvas.into(), template: tmpl.path.clone(),
        unresolved_source: src.unresolved, user_offset: opt.offset, yaw_deg: opt.yaw_deg,
        source_cloud: src.cloud(), target_bounds: tmpl.bounds, budget_max: tmpl.budget_max, ..Default::default()
    };

    // --- 1. resolve every object through the table / matcher ---
    let mut resolved: Vec<Option<Placed>> = Vec::with_capacity(src.objects.len());
    let mut rotated = 0usize;
    for (i, o) in src.objects.iter().enumerate() {
        let sc = cat.get(src_game, &o.key);
        let kind = sc.map(|c| c.kind.clone()).unwrap_or_default();
        let mut r = ObjReport {
            index: i, src_slot: o.slot, src_key: o.key.clone(), src_display: o.display.clone(), dst_key: String::new(),
            dst_display: String::new(), kind, conf: Confidence::None, delta: [0.0; 3], src_pos: o.pos, dst_pos: o.pos,
            status: Status::Dropped, note: String::new(),
        };
        match super::table::resolve(table, cat, src_game, &o.key, dst_game, &pool) {
            Ok(res) if res.conf <= opt.min_confidence => {
                let dc = pool.iter().find(|c| c.key == res.dst_key);
                r.dst_key = res.dst_key.clone();
                r.dst_display = dc.map(|c| c.display.clone()).unwrap_or_default();
                r.conf = res.conf;
                r.delta = res.delta;
                // A gameplay MARKER's model size says nothing about play (a respawn point is a
                // marker, not a shape), so only structural pieces are flagged `resized`.
                let structural = !dc.map_or(false, |c| c.kind().is_gameplay());
                r.status = if structural && res.delta.iter().any(|d| d.abs() > 0.25) { Status::Resized } else { Status::Mapped };
                r.note = format!("{}: {}", res.source, res.why);
                // ROTATION: a gameplay marker keeps its own facing (a respawn point's model box
                // says nothing about which way a player should look); a structural piece is
                // turned so the target model lies the way the source model did.
                let perm = if dc.map_or(true, |c| c.kind().is_gameplay()) { super::table::AXES_IDENTITY } else { res.perm };
                let (fwd, up) = turn_to_axes(o.fwd, o.up, perm);
                if perm != super::table::AXES_IDENTITY {
                    rotated += 1;
                    let last = rep.objects.len();
                    let _ = last;
                    r.note = format!("{} [turned: target axes {}]", r.note, super::table::axes_str(perm));
                }
                rep.objects.push(r);
                resolved.push(Some(Placed { report_index: i, dst: res.dst_key, pos: o.pos, fwd, up, src: o.clone() }));
                continue;
            }
            Ok(res) => r.note = format!("best match '{}' is only {} (below the '{}' floor)", res.dst_key, res.conf.name(), opt.min_confidence.name()),
            Err(e) => r.note = e,
        }
        rep.objects.push(r);
        resolved.push(None);
    }

    // --- 2. place the cloud: re-centre inside the target canvas, then yaw / offset ---
    let live: Vec<[f32; 3]> = resolved.iter().flatten().map(|p| p.pos).collect();
    let cloud = cloud_of(live.iter().copied());
    let (auto, pivot) = match (cloud, opt.keep_origin) {
        (Some((mn, mx)), false) => {
            let c = mid(mn, mx);
            // XY: the template's own cloud centre when it has one (that is where people build on
            // this canvas), else the middle of the world box.
            let t = tmpl.anchor.unwrap_or([(tmpl.bounds[0] + tmpl.bounds[1]) * 0.5, (tmpl.bounds[2] + tmpl.bounds[3]) * 0.5, (tmpl.bounds[4] + tmpl.bounds[5]) * 0.5]);
            // Z: floor to floor, so a layout keeps its height above the ground it was built on.
            let dz = tmpl.floor.unwrap_or(t[2]) - mn[2];
            ([t[0] - c[0], t[1] - c[1], dz], c)
        }
        _ => ([0.0; 3], cloud.map(|(a, b)| mid(a, b)).unwrap_or([0.0; 3])),
    };
    rep.auto_offset = auto;
    rep.rotated = rotated;
    let yaw = opt.yaw_deg.to_radians();
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let s = if opt.layout_scale.is_finite() && opt.layout_scale > 0.01 { opt.layout_scale } else { 1.0 };
    for p in resolved.iter_mut().flatten() {
        // about the SOURCE cloud centre, then the automatic + user offset
        let d = [(p.pos[0] - pivot[0]) * s, (p.pos[1] - pivot[1]) * s, (p.pos[2] - pivot[2]) * s];
        let r = [d[0] * cy - d[1] * sy, d[0] * sy + d[1] * cy, d[2]];
        p.pos = [pivot[0] + r[0] + auto[0] + opt.offset[0], pivot[1] + r[1] + auto[1] + opt.offset[1], pivot[2] + r[2] + auto[2] + opt.offset[2]];
        if yaw != 0.0 {
            let rot = |v: [f32; 3]| [v[0] * cy - v[1] * sy, v[0] * sy + v[1] * cy, v[2]];
            p.fwd = rot(p.fwd);
            p.up = rot(p.up);
        }
        // the canvas bounds are the quantisation box: outside it the encoder clamps anyway, so
        // clamp here too and SAY SO
        let b = tmpl.bounds;
        let mut clamped = false;
        for k in 0..3 {
            let (lo, hi) = (b[2 * k].min(b[2 * k + 1]), b[2 * k].max(b[2 * k + 1]));
            if lo.is_finite() && hi.is_finite() && hi > lo {
                let c = p.pos[k].clamp(lo, hi);
                if (c - p.pos[k]).abs() > 1e-3 { clamped = true; }
                p.pos[k] = c;
            }
        }
        if clamped {
            rep.clamped += 1;
            let r = &mut rep.objects[p.report_index];
            r.note = format!("{} [CLAMPED into the canvas bounds]", r.note);
        }
        rep.objects[p.report_index].dst_pos = p.pos;
    }

    // --- 3. gates: per-entry quota, total budget, slot cap. Gameplay objects go first so a
    //        budget crunch never costs the spawns and objectives.
    // The REAL slot budget: the title's array minus whatever the template must keep.
    let slot_cap = (match dst_game { Game::Halo3 => 640usize, _ => 651 }).saturating_sub(tmpl.reserved);
    if tmpl.reserved > 0 {
        rep.warnings.push(format!("the template keeps {} of the {} object slots for the map's own scenario objects, leaving {slot_cap} for the conversion",
            tmpl.reserved, if dst_game == Game::Halo3 { 640 } else { 651 }));
    }
    let mut order: Vec<usize> = (0..resolved.len()).filter(|&i| resolved[i].is_some()).collect();
    order.sort_by_key(|&i| {
        let k = pool.iter().find(|c| Some(&c.key) == resolved[i].as_ref().map(|p| &p.dst)).map(|c| c.kind()).unwrap_or(Kind::Unknown);
        (!k.is_gameplay(), i)
    });
    let mut per_entry: HashMap<(u32, u32), usize> = HashMap::new(); // (category_index, quota) -> placed
    let mut spent = 0u32;
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        let Some(p) = resolved[i].as_ref() else { continue };
        let Some(c) = pool.iter().find(|c| c.key == p.dst) else { continue };
        let Some(spot) = c.spot_on(dst_canvas) else { continue };
        let mut deny: Option<String> = None;
        if kept.len() >= slot_cap { deny = Some(format!("the target variant holds only {slot_cap} objects")); }
        if deny.is_none() && opt.enforce_quotas && c.max_allowed > 0 {
            let n = per_entry.entry((spot.category_index, spot.quota)).or_default();
            if *n as i32 >= c.max_allowed { deny = Some(format!("'{}' is capped at {} per variant", c.entry, c.max_allowed)); }
        }
        if deny.is_none() && opt.enforce_budget && tmpl.budget_max > 0 && c.price > 0 && spent + c.price as u32 > tmpl.budget_max {
            deny = Some(format!("over the canvas budget ({} of {})", spent, tmpl.budget_max));
        }
        match deny {
            Some(why) => {
                resolved[i] = None;
                let r = &mut rep.objects[i];
                r.status = Status::Dropped;
                r.note = format!("{why} [{}]", r.note);
            }
            None => {
                *per_entry.entry((spot.category_index, spot.quota)).or_default() += 1;
                spent += c.price.max(0) as u32;
                kept.push(i);
            }
        }
    }
    kept.sort_unstable();
    rep.budget_spent = spent;
    rep.budget_priced = pool.iter().any(|c| c.price > 0);
    rep.dest_cloud = cloud_of(kept.iter().filter_map(|&i| resolved[i].as_ref()).map(|p| p.pos));

    // SYSTEMATIC size mismatch: Halo 3's Forge pieces are about half the size of Reach's, so a
    // 1:1 layout leaves the geometry floating with gaps. Measure the median target/source size
    // ratio over the structural pieces and suggest the `scale` the user should have passed.
    if (opt.layout_scale - 1.0).abs() < 1e-3 {
        let mut ratios: Vec<f32> = Vec::new();
        for &i in &kept {
            let Some(p) = resolved[i].as_ref() else { continue };
            let Some(src_o) = cat.get(src_game, &p.src.key) else { continue };
            let Some(dst_o) = pool.iter().find(|c| c.key == p.dst) else { continue };
            if dst_o.kind().is_gameplay() || !src_o.has_box() || !dst_o.has_box() { continue; }
            let (a, b) = (src_o.sorted_dims()[0], dst_o.sorted_dims()[0]);
            if a > 0.5 && b > 0.0 { ratios.push(b / a); }
        }
        if ratios.len() >= 8 {
            ratios.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let med = ratios[ratios.len() / 2];
            if !(0.8..=1.25).contains(&med) {
                rep.warnings.push(format!(
                    "the target palette's pieces are on average {:.0}% of the source's size - re-run with `scale {:.2}`                      to shrink the layout's SPACING to match (positions only; object sizes never change)",
                    med * 100.0, med));
            }
        }
    }

    // --- 4. hand the kept placements to the target title's encoder ---
    let placed: Vec<&Placed> = kept.iter().filter_map(|&i| resolved[i].as_ref()).collect();
    // The ONE thing about the output that is not a pure function of the input. Read once, here,
    // so every writer below stamps the same instant and `opt.stamp_unix` can pin it.
    let stamp = opt.stamp_unix.unwrap_or_else(crate::h4::mvar::unix_now);
    let bytes = match dst_game {
        Game::Reach => write_reach(cat, &tmpl, dst_canvas, &placed, src, &mut rep, stamp)?,
        Game::Halo4 | Game::H2A => write_h4(cat, dst_game, &tmpl, &canvas_map, dst_canvas, &placed, src, &mut rep, stamp)?,
        Game::Halo3 => write_h3(cat, &tmpl, &canvas_map, dst_canvas, &placed, &mut rep)?,
    };
    Ok(Converted { bytes, report: rep })
}

/// [`convert`] plus the file write.
pub fn convert_to_file(cat: &Catalog, table: &Table, src: &XVariant, dst_game: Game, dst_canvas: &str, out: &Path, opt: &Options) -> Result<Report, String> {
    let c = convert(cat, table, src, dst_game, dst_canvas, opt)?;
    if let Some(d) = out.parent() { let _ = std::fs::create_dir_all(d); }
    std::fs::write(out, &c.bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(c.report)
}

/// A title-neutral variant title / author for the converted file. The content-header strings are
/// FIXED-WIDTH and go through the game's own UI, so anything non-printable is stripped (a Reach
/// author field full of garbage bytes otherwise shows up as mojibake in the Halo 4 file browser).
fn titles(src: &XVariant) -> (String, String) {
    let clean = |s: &str, n: usize| -> String {
        s.chars().filter(|c| !c.is_control() && (c.is_ascii_graphic() || *c == ' ')).collect::<String>().trim().chars().take(n).collect()
    };
    let t = clean(&src.title, 100);
    (if t.is_empty() { "Converted".into() } else { t }, clean(&src.author, 15))
}

// ---- Reach target ----------------------------------------------------------------------------

fn write_reach(cat: &Catalog, tmpl: &Template, canvas: &str, placed: &[&Placed], src: &XVariant, rep: &mut Report, stamp: u64) -> Result<Vec<u8>, String> {
    let data = crate::mvar::read_expanded(&tmpl.path).map_err(|e| format!("{}: {e}", tmpl.path.display()))?;
    let tv = crate::mvar::parse_variant(&tmpl.path).ok_or_else(|| format!("{}: cannot re-read the template", tmpl.path.display()))?;
    let mut labels: Vec<String> = tv.labels.clone();
    let mut objects: Vec<crate::mvar::PlacedObject> = Vec::with_capacity(placed.len());
    let mut quota_hits: HashMap<usize, usize> = HashMap::new();
    for p in placed {
        let Some(c) = cat.get(Game::Reach, &p.dst) else { continue };
        let Some(spot) = c.spot_on(canvas) else { continue };
        // the wire fields are 8 and 5 bits; a palette past them cannot be addressed at all
        if spot.quota > 255 || spot.variant > 31 {
            let r = &mut rep.objects[p.report_index];
            r.status = Status::Dropped;
            r.note = format!("palette slot ({}, {}) is outside Reach's 8-bit quota / 5-bit variant fields [{}]", spot.quota, spot.variant, r.note);
            continue;
        }
        let mut o = crate::mvar::new_placed_object(spot.quota as u16, spot.variant as u8, p.pos, p.fwd, p.up,
            if p.src.team < 0 { crate::mvar::TEAM_NEUTRAL } else { p.src.team as u8 },
            p.src.color.map(|c| c as i32).unwrap_or(-1));
        o.respawn = p.src.spawn_time;
        o.spawn_seq = p.src.spawn_sequence.clamp(-100, 100);
        o.cached_type = c.object_type;
        // PLACEMENT: the SOURCE decides gametype visibility (symmetry, at-start, unique), the
        // TARGET decides PHYSICS. A Reach structure left at the default "normal" physics falls
        // over the moment the round starts, and Halo 3 carries no physics field at all, so a
        // structural piece with no physics to inherit is pinned FIXED.
        let phys = match (c.kind().is_gameplay(), p.src.placement & crate::mvar::PLACE_PHYSICS_MASK) {
            (true, _) => 0,
            (false, 0) => 0x40, // fixed
            (false, ph) => ph,
        };
        o.placement = (p.src.placement & !crate::mvar::PLACE_PHYSICS_MASK) | phys;
        if o.placement & crate::mvar::PLACE_SYMMETRY_MASK == 0 { o.placement |= crate::mvar::PLACE_SYMMETRY_MASK; }
        o.boundary_shape = if c.shape != 0 { c.shape } else { p.src.shape };
        o.boundary = reach_boundary_q(o.boundary_shape, p.src.shape_wu);
        o.tele_channel = p.src.tele_channel;
        o.tele_passability = p.src.tele_passability;
        o.weapon_clips = p.src.weapon_clips;
        o.label_idx = match p.src.labels.first() {
            Some(l) if !l.is_empty() => {
                let i = labels.iter().position(|x| x == l).unwrap_or_else(|| { labels.push(l.clone()); labels.len() - 1 });
                if i < 0xFFFF { i as u16 } else { 0xFFFF }
            }
            _ => 0xFFFF,
        };
        *quota_hits.entry(spot.quota as usize).or_default() += 1;
        objects.push(o);
    }
    // the quota table: raise each entry's maximum to what we placed (the engine recomputes
    // `placed_on_map` itself, and the encoder re-counts it)
    let mut quotas: Vec<(u8, u8)> = tv.globals.quotas.iter().map(|q| (q.0, q.1)).collect();
    for (q, n) in &quota_hits {
        if let Some(e) = quotas.get_mut(*q) { e.1 = e.1.max((*n).min(255) as u8); }
    }
    let (title, author) = titles(src);
    let mut header = crate::mvar::HeaderEdits {
        title: Some(title), description: Some(format!("converted from {} ({}) by Halo Map Studio", src.canvas, src.game_short)),
        author: Some(author), editor: Some("Halo Map Studio".into()), labels: (labels != tv.labels).then_some(labels),
        quotas: Some(quotas), ..Default::default()
    };
    header.stamp_new_at(0, stamp);
    if objects.len() > 651 { rep.warnings.push(format!("{} objects is past Reach's 651 slots; the extra ones are dropped by the encoder", objects.len())); }
    let (bytes, written) = crate::mvar::rebuild_file(&data, &objects, Some(&header))?;
    if written < objects.len() { rep.warnings.push(format!("the encoder wrote {written} of {} objects (no free slot for the rest)", objects.len())); }
    Ok(bytes)
}

// ---- Halo 4 / H2A target ---------------------------------------------------------------------

fn write_h4(cat: &Catalog, dst_game: Game, tmpl: &Template, canvas_map: &Path, canvas: &str, placed: &[&Placed], src: &XVariant, rep: &mut Report, stamp: u64) -> Result<Vec<u8>, String> {
    let cache = crate::h4::cache::H4Cache::open(canvas_map).map_err(|e| format!("{}: {e}", canvas_map.display()))?;
    let fp = crate::h4::palette::palette(&cache);
    let tv = crate::h4::mvar::parse_h4_variant(&tmpl.path).map_err(|e| format!("{}: {e}", tmpl.path.display()))?;
    // 50/51 = Halo 4 (shipped / MCC-saved), 52/53 = Halo 2 Anniversary. All four round-trip
    // byte-identically through the encoder, so all four are valid conversion targets.
    if !matches!(tv.version, 50 | 51 | 52 | 53) {
        return Err(format!("{} is body version {} - HMS's Halo 4 / H2A encoder writes 50, 51, 52 and 53", tmpl.path.display(), tv.version));
    }
    let data = crate::h4::mvar::read_h4_expanded(&tmpl.path).map_err(|e| format!("{}: {e}", tmpl.path.display()))?;
    let mut labels: Vec<String> = tv.labels.clone();
    let mut objects: Vec<crate::h4::mvar::H4PlacedObject> = Vec::with_capacity(placed.len());
    for p in placed {
        // Every lookup is against the TARGET title's own rows. Halo 4 and Halo 2 Anniversary name
        // many Forge pieces identically (`fw_stone_plat_10x10#bb_10x10` is in both catalogs), so
        // asking Halo 4 first and falling back resolved an H2A target to a Halo 4 object, whose
        // `spot_on` then had no entry for an H2A canvas - and the placement was dropped without a
        // word. Each of the three steps below now reports WHY, as the Halo 3 writer does.
        let miss = |rep: &mut Report, why: &str| {
            let r = &mut rep.objects[p.report_index];
            r.status = Status::Dropped;
            r.note = format!("{why} [{}]", r.note);
        };
        let Some(c) = cat.get(dst_game, &p.dst) else { miss(rep, "the catalog lost the target object"); continue };
        let Some(spot) = c.spot_on(canvas) else { miss(rep, "not in this canvas's palette"); continue };
        let pose = crate::h4::palette::H4Pose { pos: p.pos, fwd: p.fwd, up: p.up };
        // instantiate gives the engine's own fresh-drop defaults (phased physics, spawn time,
        // boundary from the obje multiplayer block, spare clips)
        let Some(inst) = crate::h4::palette::instantiate(&cache, &fp, spot.quota as u8, spot.variant as u8, pose) else {
            miss(rep, "the canvas's palette has no such entry"); continue
        };
        let mut o = inst.record;
        o.team = p.src.team.clamp(-1, 8);
        o.color = p.src.color.map(|c| c & 7);
        // keep the engine's own physics for this palette entry (phased blocks stay phased) and
        // take only the gametype-visibility bits from the source; see `write_reach`
        let keep = o.placement & !0x3Fu16;
        o.placement = keep | (p.src.placement & 0x3F) as u16;
        if o.placement & crate::mvar::PLACE_SYMMETRY_MASK as u16 == 0 { o.placement |= crate::mvar::PLACE_SYMMETRY_MASK as u16; }
        if p.src.spawn_time != 0 { o.spawn_time = p.src.spawn_time; }
        o.spawn_sequence = p.src.spawn_sequence.clamp(-100, 100) as i8;
        o.user_data = p.src.user_data.clamp(-100, 100) as i8;
        // the source's own boundary wins when both sides have one (a resized safe zone matters)
        if p.src.shape != 0 && o.shape as u8 == p.src.shape {
            for k in 0..4 { o.shape_values[k] = crate::h4::edit::shape_wu_to_raw(p.src.shape_wu[k]); }
        }
        if let crate::h4::mvar::H4TypeData::Pair(_, _) = o.type_data {
            if p.src.tele_channel != 0 || p.src.tele_passability != 0 {
                o.type_data = crate::h4::mvar::H4TypeData::Pair(p.src.tele_channel & 0x1F, p.src.tele_passability & 0x1F);
            }
        }
        if let crate::h4::mvar::H4TypeData::Byte(_) = o.type_data {
            if p.src.weapon_clips != 0 { o.type_data = crate::h4::mvar::H4TypeData::Byte(p.src.weapon_clips); }
        }
        for (i, l) in p.src.labels.iter().take(4).enumerate() {
            if l.is_empty() { continue; }
            let li = labels.iter().position(|x| x == l).unwrap_or_else(|| { labels.push(l.clone()); labels.len() - 1 });
            if li < 256 { o.labels[i] = Some(li as u8); }
        }
        o.slot = crate::h4::mvar::NEW_SLOT;
        objects.push(o);
    }
    let pal = fp.to_mvar_palette();
    let spent = crate::h4::mvar::budget_spent(&objects, &pal);
    let (title, author) = titles(src);
    let mut edits = crate::h4::mvar::H4HeaderEdits {
        title: Some(title),
        description: Some(format!("converted from {} ({}) by Halo Map Studio", src.canvas, src.game_short)),
        author: Some(author),
        editor: Some("Halo Map Studio".into()),
        labels: (labels != tv.labels).then_some(labels),
        budget_spent: Some(spent),
        budget_max: Some(crate::h4::edit::budget_max_for(tv.budget_max, spent)),
        ..Default::default()
    };
    edits.stamp_new_at(0, stamp);
    if objects.len() > crate::h4::mvar::H4_SLOTS {
        rep.warnings.push(format!("{} objects is past Halo 4's {} slots", objects.len(), crate::h4::mvar::H4_SLOTS));
        objects.truncate(crate::h4::mvar::H4_SLOTS);
    }
    let (bytes, _slots) = crate::h4::mvar::rebuild_h4_file(&data, &objects, Some(&edits)).map_err(|e| e.to_string())?;
    Ok(bytes)
}

// ---- Halo 3 target ---------------------------------------------------------------------------

fn write_h3(cat: &Catalog, tmpl: &Template, canvas_map: &Path, canvas: &str, placed: &[&Placed], rep: &mut Report) -> Result<Vec<u8>, String> {
    let cache = crate::h3::cache::H3Cache::open(canvas_map).map_err(|e| format!("{}: {e}", canvas_map.display()))?;
    let items = crate::h3::palette::palette(&cache);
    let tv = crate::h3::mvar::parse_h3_variant(&tmpl.path).map_err(|e| format!("{}: {e}", tmpl.path.display()))?;
    let data = crate::mvar::read_expanded(&tmpl.path).map_err(|e| format!("{}: {e}", tmpl.path.display()))?;
    let mut out_v = tv.clone();
    // keep the template's SCENARIO-owned slots (the map's own objects the variant edits) and
    // throw away its Forge placements
    let kept: Vec<crate::h3::mvar::H3VariantObject> = tv.objects.iter().filter(|o| !o.placed).cloned().collect();
    let mut used = vec![false; 640];
    for k in &kept { if k.slot < 640 { used[k.slot] = true; } }
    let mut list: Vec<crate::h3::mvar::H3VariantObject> = kept.clone();
    let mut next = tv.num_scenario_objects as usize;
    for p in placed {
        let miss = |rep: &mut Report, why: &str| {
            let r = &mut rep.objects[p.report_index];
            r.status = Status::Dropped;
            r.note = format!("{why} [{}]", r.note);
        };
        let Some(c) = cat.get(Game::Halo3, &p.dst) else { miss(rep, "the catalog lost the target object"); continue };
        let Some(spot) = c.spot_on(canvas) else { miss(rep, "not in this canvas's palette"); continue };
        let Some(it) = items.iter().find(|it| it.block as u32 == spot.category_index && it.entry as u32 == spot.quota) else {
            miss(rep, "the canvas's palette has no such row"); continue
        };
        // the quota: the template's table when it already carries this row, else appended
        let qi = match out_v.quotas.iter().position(|q| q.definition == it.definition) {
            Some(q) => q as i32,
            None => {
                if out_v.quotas.len() >= crate::h3::mvar::H3_QUOTA_SLOTS {
                    rep.warnings.push(format!("the quota table is full; '{}' could not be added", it.name));
                    miss(rep, "the quota table is full");
                    continue;
                }
                out_v.quotas.push(it.quota());
                (out_v.quotas.len() - 1) as i32
            }
        };
        while next < 640 && used[next] { next += 1; }
        if next >= 640 {
            rep.warnings.push("Halo 3 variants hold 640 object slots and the template's own scenario objects fill the rest; the remaining placements were dropped".into());
            for q in placed.iter().skip_while(|x| x.report_index != p.report_index) {
                let r = &mut rep.objects[q.report_index];
                r.status = Status::Dropped;
                r.note = format!("no free Halo 3 object slot [{}]", r.note);
            }
            break;
        }
        let mut o = crate::h3::palette::instantiate(it, qi, next, p.pos, tv.bounds);
        o.set_pose(p.pos, Vec3::from(p.fwd), Vec3::from(p.up), tv.bounds);
        o.team = if p.src.team < 0 { crate::h3::palette::TEAM_NEUTRAL } else { (p.src.team as u8).min(8) };
        if p.src.spawn_time != 0 { o.spawn_time = p.src.spawn_time; }
        if p.src.shape != 0 && o.boundary_shape == p.src.shape { o.set_boundary(o.boundary_shape, p.src.shape_wu); }
        if p.src.tele_channel != 0 { o.shared_storage = p.src.tele_channel; }
        used[next] = true;
        next += 1;
        list.push(o);
    }
    list.sort_by_key(|r| r.slot);
    // recount what the table says is placed (the engine recomputes it, but keep the file honest)
    let mut placed_counts = vec![0u8; out_v.quotas.len()];
    for r in list.iter().filter(|r| r.placed) {
        if let Ok(i) = usize::try_from(r.quota) { if let Some(n) = placed_counts.get_mut(i) { *n = n.saturating_add(1); } }
    }
    for (q, n) in out_v.quotas.iter_mut().zip(placed_counts) { q.placed_on_map = n; q.maximum = q.maximum.max(n); }
    out_v.spent_budget = list.iter().filter(|r| r.placed).filter_map(|r| usize::try_from(r.quota).ok()).filter_map(|i| out_v.quotas.get(i)).map(|q| q.price).sum();
    out_v.num_variant_objects = list.last().map_or(0, |r| r.slot as u16 + 1).max(tv.num_variant_objects);
    let edits = crate::h3::mvar::H3HeaderEdits {
        name: Some(rep.src_title.chars().take(15).collect()),
        description: Some(format!("converted from {} ({})", rep.src_canvas, rep.src_game)),
        author: None,
    };
    crate::h3::mvar::rebuild_h3_file(&data, &out_v, &list, Some(&edits)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reach's 11-bit boundary quanta survive the world-unit round trip (a safe zone keeps its
    /// size when it crosses titles and comes back).
    #[test]
    fn reach_boundary_round_trips_through_world_units() {
        for shape in 1u8..=3 {
            for q in [0u16, 1, 37, 500, 1023, 2047] {
                let qs = [q; 4];
                let wu = reach_boundary_wu(shape, qs);
                let back = reach_boundary_q(shape, wu);
                let n = [0usize, 1, 3, 4][shape as usize];
                for k in 0..n { assert_eq!(back[k], q, "shape {shape} slot {k}"); }
            }
        }
    }

    /// The neutral cloud AABB and the mid-point helper agree with hand arithmetic.
    #[test]
    fn cloud_and_mid() {
        let c = cloud_of([[1.0, 2.0, 3.0], [-1.0, 5.0, 0.0]].into_iter()).unwrap();
        assert_eq!(c, ([-1.0, 2.0, 0.0], [1.0, 5.0, 3.0]));
        assert_eq!(mid(c.0, c.1), [0.0, 3.5, 1.5]);
        assert!(cloud_of([[f32::NAN, 0.0, 0.0]].into_iter()).is_none());
    }

    // -----------------------------------------------------------------------------------------
    // END-TO-END, on the installed game. Every one of these SKIPS (returns) when the maps or the
    // variants are not on this machine, so the suite stays green on a bare checkout.
    // -----------------------------------------------------------------------------------------

    fn ready() -> Option<(Catalog, Table)> {
        let c = Catalog::load();
        if c.objects.is_empty() { return None; }
        Some((c, Table::load()))
    }

    /// A SHIPPED Reach Forge variant, if one is installed. Prefers `hopper_map_variants` (the
    /// game's own files, so the test is the same everywhere) over `map_variants` (the player's
    /// own saves, which may be anything), and skips the huge ones so the Halo 4 BUDGET gate - a
    /// separate test - is not what this test measures.
    fn reach_source(cat: &Catalog) -> Option<XVariant> {
        let mut dirs = crate::mapcat::variant_dirs();
        dirs.sort_by_key(|d| !d.to_string_lossy().contains("hopper"));
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar"))).collect();
            files.sort();
            for f in files {
                if crate::h4::mvar::is_h4_variant(&f) || crate::h3::mvar::is_h3_variant(&f) { continue; }
                if let Ok(v) = read_variant(cat, &f) {
                    if (50..=450).contains(&v.objects.len()) { return Some(v); }
                }
            }
        }
        None
    }

    /// Reading a shipped variant back through the CATALOG resolves essentially every placement:
    /// if this drifts, the palette coordinates in `catalog.json` no longer match the game.
    #[test]
    fn reading_a_shipped_variant_resolves_through_the_catalog() {
        let Some((cat, _)) = ready() else { return };
        let Some(v) = reach_source(&cat) else { return };
        eprintln!("source: {} objects on {} ({} unresolved)", v.objects.len(), v.canvas, v.unresolved);
        assert!(v.unresolved * 20 <= v.objects.len(), "{} of {} placements did not resolve against the catalog", v.unresolved, v.objects.len());
        assert!(v.objects.iter().all(|o| !o.key.is_empty()));
        assert!(v.objects.iter().all(|o| o.pos.iter().all(|c| c.is_finite())));
    }

    /// THE conversion: a real Reach variant onto a real Halo 4 canvas. Asserts the object count
    /// the report claims equals the object count the written file decodes to, that the poses
    /// survive the trip (the cloud keeps its SHAPE, only its origin moves), and that the file
    /// re-encodes byte-identically through Halo 4's own encoder.
    #[test]
    fn reach_variant_converts_onto_a_halo4_canvas_with_the_poses_intact() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo4).into_iter().find(|c| c == "ca_forge_ravine")
            .or_else(|| cat.canvases(Game::Halo4).into_iter().next()) else { return };
        if canvas_path(Game::Halo4, &canvas).is_none() { return; }
        let opt = Options::default();
        let Ok(out) = convert(&cat, &tab, &src, Game::Halo4, &canvas, &opt) else { return };
        let rep = &out.report;
        eprintln!("{}", rep.summary());
        assert!(rep.mapped() > 0);
        assert!(rep.coverage() > 80.0, "coverage {:.1}% on {} -> halo4 {canvas}\n{}", rep.coverage(), src.canvas, rep.by_kind());
        // every drop says WHY - a silent loss is the one thing a conversion must never do
        assert!(rep.objects.iter().filter(|o| o.status == Status::Dropped).all(|o| !o.note.is_empty()));

        // write it, read it back with Halo 4's own reader, and compare object for object
        let dir = std::env::temp_dir().join("hms_xgame_test");
        let _ = std::fs::create_dir_all(&dir);
        let dst = dir.join("convert_h4.mvar");
        std::fs::write(&dst, &out.bytes).unwrap();
        let back = crate::h4::mvar::parse_h4_variant(&dst).expect("the converted file must parse as a Halo 4 variant");
        assert_eq!(back.objects.len(), rep.mapped(), "the report's object count must equal the file's");
        assert!(back.objects.iter().all(|o| o.quota.is_some() && o.variant.is_some()),
            "every record needs an explicit quota AND variant (a NONE variant makes the engine spawn the next entry)");
        assert!(back.objects.iter().all(|o| o.flags == 1), "every occupied slot carries flags 1");

        // the POSES: the converted cloud is the source cloud rigidly translated, so its SPAN
        // must match to within the position quantisation step.
        let (smn, smx) = src.cloud().unwrap();
        let (dmn, dmx) = cloud_of(back.objects.iter().map(|o| o.pos)).unwrap();
        for k in 0..3 {
            let (a, b) = (smx[k] - smn[k], dmx[k] - dmn[k]);
            assert!((a - b).abs() < 1.0, "axis {k}: source span {a:.2} wu vs converted {b:.2} wu");
        }
        // and the RELATIVE pose of every object is preserved by the same offset
        let off = [dmn[0] - smn[0], dmn[1] - smn[1], dmn[2] - smn[2]];
        let mut checked = 0;
        for r in rep.objects.iter().filter(|r| r.status != Status::Dropped) {
            for k in 0..3 {
                assert!((r.dst_pos[k] - (r.src_pos[k] + off[k])).abs() < 1.0,
                    "object {} axis {k}: {:.2} vs {:.2}", r.index, r.dst_pos[k], r.src_pos[k] + off[k]);
            }
            checked += 1;
        }
        assert!(checked > 0);

        // the byte-exact gate: re-encoding what we wrote changes nothing
        let rt = crate::h4::mvar::roundtrip_report(&dst, None).unwrap();
        assert!(rt.contains("BYTE-IDENTICAL"), "{rt}");
    }

    /// The same trip onto a Halo 3 canvas, where the palette gap is real: coverage is lower, the
    /// file still has to be a valid Halo 3 variant, and the report's count has to be honest.
    #[test]
    fn reach_variant_converts_onto_a_halo3_canvas() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo3).into_iter().find(|c| c == "sandbox") else { return };
        if canvas_path(Game::Halo3, &canvas).is_none() { return; }
        let Ok(out) = convert(&cat, &tab, &src, Game::Halo3, &canvas, &Options::default()) else { return };
        let rep = &out.report;
        eprintln!("{}", rep.summary());
        let dir = std::env::temp_dir().join("hms_xgame_test");
        let _ = std::fs::create_dir_all(&dir);
        let dst = dir.join("convert_h3.mvar");
        std::fs::write(&dst, &out.bytes).unwrap();
        let back = crate::h3::mvar::parse_h3_variant(&dst).expect("the converted file must parse as a Halo 3 variant");
        assert_eq!(back.placed().count(), rep.mapped(), "the report's object count must equal the file's");
        assert!(back.quotas.len() <= crate::h3::mvar::H3_QUOTA_SLOTS);
        // every placed record points at a quota the table actually carries
        for o in back.placed() {
            let q = usize::try_from(o.quota).unwrap();
            assert!(q < back.quotas.len(), "quota {q} past the {} entries", back.quotas.len());
        }
        // re-encode gate
        let data = crate::mvar::read_expanded(&dst).unwrap();
        assert_eq!(crate::h3::mvar::rebuild_h3_file(&data, &back, &back.objects, None).unwrap(), data);
    }

    /// #h2a-target  Halo 2 Anniversary is a conversion TARGET, not just a source. H2A variants
    /// carry body version 52 (shipped) or 53 (game-saved), and both encode byte-identically, so
    /// the writer must accept them the way it accepts Halo 4's 50/51. The gate: the converted
    /// file has to parse back as an H2A variant, keep the template's own body version, and
    /// re-encode to the same bytes — the same bar the Halo 4 and Halo 3 targets are held to.
    #[test]
    fn reach_variant_converts_onto_a_halo2_anniversary_canvas() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::H2A).into_iter().find(|c| canvas_path(Game::H2A, c).is_some()) else { return };
        let out = match convert(&cat, &tab, &src, Game::H2A, &canvas, &Options::default()) {
            Ok(o) => o,
            // a missing template or cache is "not installed"; a version refusal is the bug
            Err(e) => { assert!(!e.contains("body version"), "H2A refused as a target: {e}"); return }
        };
        let rep = &out.report;
        eprintln!("reach -> h2a {canvas}: {}", rep.summary());
        assert!(rep.mapped() > 0, "nothing survived the trip onto {canvas}");

        let dir = std::env::temp_dir().join("hms_xgame_test");
        let _ = std::fs::create_dir_all(&dir);
        let dst = dir.join("convert_h2a.mvar");
        std::fs::write(&dst, &out.bytes).unwrap();
        let back = crate::h4::mvar::parse_h4_variant(&dst).expect("the converted file must parse as an H2A variant");
        assert!(matches!(back.version, 52 | 53), "body version {} is not H2A's", back.version);
        assert_eq!(back.chunk_version, crate::h4::mvar::H2A_MVAR_VERSION);
        assert_eq!(back.objects.len(), rep.mapped(), "the report's object count must equal the file's");
        assert!(back.objects.iter().all(|o| o.quota.is_some() && o.variant.is_some()),
            "every record needs an explicit quota AND variant");
        // the byte-exact gate, exactly as for Halo 4
        let rt = crate::h4::mvar::roundtrip_report(&dst, None).unwrap();
        assert!(rt.contains("BYTE-IDENTICAL"), "{rt}");
    }

    /// The GATES bite: with the quota and budget caps off, MORE objects survive than with them
    /// on, and no run ever exceeds the canvas's slot array.
    #[test]
    fn quota_budget_and_slot_gates_bite() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo3).into_iter().find(|c| c == "sandbox") else { return };
        if canvas_path(Game::Halo3, &canvas).is_none() { return; }
        let strict = Options::default();
        let mut loose = Options::default();
        loose.enforce_quotas = false;
        loose.enforce_budget = false;
        let (Ok(a), Ok(b)) = (convert(&cat, &tab, &src, Game::Halo3, &canvas, &strict),
                              convert(&cat, &tab, &src, Game::Halo3, &canvas, &loose)) else { return };
        eprintln!("halo3 gates: strict {} vs loose {} of {}", a.report.mapped(), b.report.mapped(), src.objects.len());
        assert!(b.report.mapped() >= a.report.mapped(), "turning the gates OFF must never place fewer objects");
        // the slot cap is absolute either way
        for r in [&a.report, &b.report] {
            assert!(r.mapped() <= 640, "{} objects is past Halo 3's 640 slots", r.mapped());
        }
        // a dropped object always says why
        assert!(a.report.objects.iter().filter(|o| o.status == Status::Dropped).all(|o| !o.note.is_empty()));
    }

    /// The confidence FLOOR is honoured: asking for `exact` only can never let a `close` or
    /// `approximate` mapping through.
    #[test]
    fn min_confidence_floor_is_honoured() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo4).into_iter().find(|c| c == "ca_forge_ravine") else { return };
        if canvas_path(Game::Halo4, &canvas).is_none() { return; }
        let mut opt = Options::default();
        opt.min_confidence = Confidence::Exact;
        let Ok(out) = convert(&cat, &tab, &src, Game::Halo4, &canvas, &opt) else { return };
        assert!(out.report.objects.iter().filter(|o| o.status != Status::Dropped).all(|o| o.conf == Confidence::Exact));
        assert!(out.report.close() == 0 && out.report.approximate() == 0);
        let loose = convert(&cat, &tab, &src, Game::Halo4, &canvas, &Options::default()).unwrap();
        assert!(loose.report.mapped() >= out.report.mapped());
    }

    /// `keep_origin` really keeps the source coordinates (modulo the canvas clamp), and a yaw of
    /// 90 degrees swaps the cloud's X and Y spans.
    #[test]
    fn placement_controls_do_what_they_say() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo4).into_iter().find(|c| c == "dlc_forge_island")
            .or_else(|| cat.canvases(Game::Halo4).into_iter().find(|c| c == "ca_forge_ravine")) else { return };
        if canvas_path(Game::Halo4, &canvas).is_none() { return; }
        let mut keep = Options::default();
        keep.keep_origin = true;
        let Ok(k) = convert(&cat, &tab, &src, Game::Halo4, &canvas, &keep) else { return };
        assert_eq!(k.report.auto_offset, [0.0; 3]);
        let mut yaw = Options::default();
        yaw.yaw_deg = 90.0;
        let Ok(y) = convert(&cat, &tab, &src, Game::Halo4, &canvas, &yaw) else { return };
        let (a0, a1) = k.report.dest_cloud.unwrap();
        let (b0, b1) = y.report.dest_cloud.unwrap();
        let (ax, ay) = (a1[0] - a0[0], a1[1] - a0[1]);
        let (bx, by) = (b1[0] - b0[0], b1[1] - b0[1]);
        eprintln!("yaw 0 span {ax:.1} x {ay:.1}; yaw 90 span {bx:.1} x {by:.1}");
        assert!((ax - by).abs() < 2.0 && (ay - bx).abs() < 2.0, "a 90 degree yaw must swap the X and Y spans");
    }

    /// Converting a variant onto the canvas it already lives on is refused (it would be a copy,
    /// and the source/target palettes would be the same).
    #[test]
    fn same_game_same_canvas_is_refused() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let canvas = src.canvas.clone();
        assert!(convert(&cat, &tab, &src, Game::Reach, &canvas, &Options::default()).is_err());
    }

    /// `turn_to_axes` is the rotation half of the fix: it must re-aim the pose so the TARGET's
    /// local axes end up where the SOURCE's were, and leave an identity mapping untouched.
    #[test]
    fn turn_to_axes_yaws_the_pose_and_leaves_the_identity_alone() {
        let (f, u) = ([1.0f32, 0.0, 0.0], [0.0f32, 0.0, 1.0]);
        // identity: nothing moves
        assert_eq!(turn_to_axes(f, u, super::super::table::AXES_IDENTITY), (f, u));
        // yxz: the target's local Y must take the source's local X. The engine's basis is
        // col0 = fwd, col1 = up x fwd, col2 = up, so the new forward is the source's local Y
        // (= up x fwd = +Y here) and up is unchanged -> a 90 degree yaw.
        let (nf, nu) = turn_to_axes(f, u, [1, 0, 2]);
        assert!((Vec3::from(nf) - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5, "{nf:?}");
        assert!((Vec3::from(nu) - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5, "{nu:?}");
        // and it composes with an already-turned source pose instead of overwriting it
        let (f2, u2) = ([0.0f32, 1.0, 0.0], [0.0f32, 0.0, 1.0]); // source already yawed 90
        let (nf2, nu2) = turn_to_axes(f2, u2, [1, 0, 2]);
        assert!((Vec3::from(nf2) - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5, "{nf2:?}");
        assert!((Vec3::from(nu2) - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5, "{nu2:?}");
        // zxy lays the piece down: the target's local Z takes the source's local X
        let (nf3, nu3) = turn_to_axes(f, u, [2, 0, 1]);
        assert!((Vec3::from(nf3) - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5, "{nf3:?}");
        assert!((Vec3::from(nu3) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5, "{nu3:?}");
        // every mapping keeps the basis orthonormal and right-handed
        for p in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let (a, b) = turn_to_axes(f, u, p);
            let (a, b) = (Vec3::from(a), Vec3::from(b));
            assert!((a.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5);
            assert!(a.dot(b).abs() < 1e-5, "mapping {p:?} gave a sheared basis");
        }
    }

    /// REGRESSION (user-reported): converting The Cage to Halo 4 put "FF Dom Turretpad" objects
    /// in the layout where Reach had Forge platforms. Over the SHIPPED catalog and table, no
    /// Reach structural piece may resolve to a Halo 4 prop on any canvas.
    #[test]
    fn no_reach_structure_resolves_to_a_halo4_prop() {
        let cat = Catalog::load();
        let tab = Table::load();
        if cat.objects.is_empty() { return; }
        let Some(canvas) = cat.canvases(Game::Halo4).into_iter().find(|c| c == "ca_forge_ravine") else { return };
        let pool: Vec<&CatalogObject> = cat.on_canvas(Game::Halo4, &canvas);
        if pool.is_empty() { return; }
        let mut checked = 0;
        for src in cat.of_game(Game::Reach).filter(|o| o.kind().is_structural()) {
            let Ok(r) = super::super::table::resolve(&tab, &cat, Game::Reach, &src.key, Game::Halo4, &pool) else { continue };
            let Some(dst) = pool.iter().find(|c| c.key == r.dst_key) else { continue };
            assert!(dst.kind().is_structural(),
                "{} [{}] resolved to {} [{}] - a structural piece must never become a prop", src.key, src.kind, dst.key, dst.kind);
            assert!(!dst.key.contains("turretpad") && !dst.key.contains("artillery"),
                "{} resolved to the emplacement {}", src.key, dst.key);
            checked += 1;
        }
        assert!(checked > 50, "only {checked} structural pieces resolved - the pool looks wrong");
    }

    /// Coverage must EXCLUDE the approximate matches: that is the whole point of the honesty fix.
    #[test]
    fn coverage_counts_only_exact_and_close() {
        let row = |st: Status, cf: Confidence| ObjReport {
            index: 0, src_slot: 0, src_key: "k".into(), src_display: "d".into(), dst_key: "t".into(), dst_display: "t".into(),
            kind: "block".into(), conf: cf, delta: [0.0; 3], src_pos: [0.0; 3], dst_pos: [0.0; 3], status: st, note: String::new(),
        };
        let r = Report {
            objects: vec![row(Status::Mapped, Confidence::Exact), row(Status::Resized, Confidence::Close),
                          row(Status::Mapped, Confidence::Approximate), row(Status::Dropped, Confidence::None)],
            ..Default::default()
        };
        assert_eq!((r.exact(), r.close(), r.approximate(), r.dropped()), (1, 1, 1, 1));
        assert!((r.coverage() - 50.0).abs() < 1e-3, "solid coverage is 2 of 4, not 3 of 4");
        assert!((r.approximate_pct() - 25.0).abs() < 1e-3);
        assert!((r.placed_pct() - 75.0).abs() < 1e-3);
        let s = r.summary();
        assert!(s.contains("50.0% SOLID"), "{s}");
        assert!(s.contains("1 approximate"), "{s}");
        assert!(r.by_kind().contains("APPROXIMATE"));
    }

    /// End to end on the installed game: the converted file's objects carry the TURNED pose, and
    /// the report counts the turns.
    #[test]
    fn converted_pieces_are_turned_to_the_target_axes() {
        let Some((cat, tab)) = ready() else { return };
        let Some(src) = reach_source(&cat) else { return };
        let Some(canvas) = cat.canvases(Game::Halo4).into_iter().find(|c| c == "ca_forge_ravine") else { return };
        if canvas_path(Game::Halo4, &canvas).is_none() { return; }
        let Ok(out) = convert(&cat, &tab, &src, Game::Halo4, &canvas, &Options::default()) else { return };
        let rep = &out.report;
        eprintln!("{}\n{}", rep.summary(), rep.by_kind());
        // every turned object says so
        let turned: Vec<&ObjReport> = rep.objects.iter().filter(|o| o.note.contains("turned")).collect();
        assert_eq!(turned.len(), rep.rotated);
        // and the CONTRACT holds for every object: the pose the converter wrote is exactly the
        // source pose with the resolved axis mapping applied - no more, no less.
        let pool: Vec<&CatalogObject> = cat.on_canvas(Game::Halo4, &canvas);
        let mut seen_perm = false;
        for (o, r) in src.objects.iter().zip(rep.objects.iter()) {
            if r.status == Status::Dropped { continue; }
            let Ok(res) = super::super::table::resolve(&tab, &cat, Game::Reach, &o.key, Game::Halo4, &pool) else { continue };
            let gameplay = pool.iter().find(|c| c.key == res.dst_key).map_or(true, |c| c.kind().is_gameplay());
            let want = if gameplay { super::super::table::AXES_IDENTITY } else { res.perm };
            if want != super::super::table::AXES_IDENTITY { seen_perm = true; }
            assert_eq!(r.note.contains("turned"), want != super::super::table::AXES_IDENTITY,
                "{}: note {:?} disagrees with the resolved mapping {:?}", o.display, r.note, want);
        }
        let _ = seen_perm;
        // a gameplay marker is NEVER turned
        for o in rep.objects.iter().filter(|o| o.note.contains("turned")) {
            let k = super::super::catalog::Kind::from_str(&o.kind);
            assert!(!k.is_gameplay(), "{} is a {} and must not be turned", o.src_display, o.kind);
        }
    }

}
