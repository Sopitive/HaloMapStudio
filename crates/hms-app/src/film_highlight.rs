//! #film-highlight  Theater: HIGHLIGHT the objects a gametype's script is responsible for, and give
//! the ones with nothing to see (an `invisible_cube_of_derek`, a hill-marker zone, a script
//! helper) a drawn placeholder of their real extents.
//!
//! # What counts as a "Megalo object" (what the recording actually carries)
//!
//! A recording does not mark objects as scripted. This module uses only what it does carry, in
//! this priority order (first rule that fits wins; see [`classify`]):
//!
//! | category | rule | evidence used |
//! |---|---|---|
//! | [`Category::Objective`] | the object lane's own objective verdict (flag, ball, bomb, skull ...) | resolved definition (`film_objective::Draw::Objective`) |
//! | [`Category::AttachedMarker`] | a generic object (type 13/14) that rides something AND is a marker (states a zone shape or non-default properties, has no model, or a variable refers to it) | base field 12: it states a parent (the race clip's hill markers glued to players) |
//! | [`Category::Zone`] | it states a boundary shape (sphere / cylinder / box) | base field 11 replicated shape, else the Forge placement's |
//! | [`Category::ScriptSpawned`] | a generic object (type 13) with NO placement in the embedded map variant, carrying non-default multiplayer properties (a team, flags or a shape) | the entity exists but the variant never placed it |
//! | [`Category::LabelledForge`] | a placed object with a Forge label or a spawn sequence | the variant's placement record |
//! | [`Category::MegaloRef`] | any other object a replicated Megalo variable refers to | the player/team/global variable series (`film_megalo`) |
//! | [`Category::Invisible`] | none of the above, but the definition has no drawable render model | the tag: 4 or fewer render triangles (the engine's invisible-wall nub) or an `invisible_*` name |
//!
//! Players and bodies are NOT highlighted. The entities that CARRY variables (player, team, global)
//! are listed separately by [`variable_carriers`]. Units, vehicles and projectiles are only ever
//! highlighted through a variable that refers to them.
//!
//! What this cannot know (UNRESOLVED): objects that a script spawns with default properties
//! and never refers to in a replicated variable are indistinguishable from any other generic
//! object and are not classified.

use crate::film::Film;
use crate::film_decode::FilmIndex;
use crate::film_objects::{ObjRow, ObjectTable};
use eframe::egui;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    Objective,
    Zone,
    AttachedMarker,
    ScriptSpawned,
    LabelledForge,
    MegaloRef,
    /// Nothing else about it is notable, but it has no drawable model (an `invisible_*` cube, a
    /// script helper): drawn as a placeholder so it is not simply absent.
    Invisible,
}

impl Category {
    pub const ALL: [Category; 7] = [
        Category::Objective,
        Category::Zone,
        Category::AttachedMarker,
        Category::ScriptSpawned,
        Category::LabelledForge,
        Category::MegaloRef,
        Category::Invisible,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Category::Objective => "objective / flag",
            Category::Zone => "zone / hill",
            Category::AttachedMarker => "attached marker",
            Category::ScriptSpawned => "script-spawned",
            Category::LabelledForge => "labelled Forge",
            Category::MegaloRef => "script variable",
            Category::Invisible => "invisible / no model",
        }
    }
    pub fn bit(self) -> u8 {
        1 << (self as u8)
    }
    /// Linear-ish RGB used for the wire lane; chosen to be distinct on both bright and dark maps.
    pub fn colour(self) -> [f32; 3] {
        match self {
            Category::Objective => [1.0, 0.25, 0.25],
            Category::Zone => [1.0, 0.8, 0.1],
            Category::AttachedMarker => [0.2, 0.9, 1.0],
            Category::ScriptSpawned => [0.9, 0.3, 1.0],
            Category::LabelledForge => [0.4, 1.0, 0.4],
            Category::MegaloRef => [1.0, 0.55, 0.15],
            Category::Invisible => [0.85, 0.85, 0.95],
        }
    }
}

/// Everything [`classify`] looks at, as plain values so the rule is a pure function.
#[derive(Clone, Debug, Default)]
pub struct Facts<'a> {
    pub entity_type: u8,
    /// Joined to a placement in the film's embedded map variant.
    pub placed: bool,
    pub label: Option<&'a str>,
    pub spawn_seq: i8,
    /// Boundary shape (replicated, else the placement's): 0 none.
    pub zone_shape: u8,
    /// Replicated multiplayer properties exist and are non-default (a team, flags, or a shape).
    pub mp_non_default: bool,
    /// States a parent at some point.
    pub rides: bool,
    pub megalo_ref: bool,
    pub objective: bool,
    /// The definition has no drawable render model (see [`hull_info`]).
    pub no_model: bool,
}

pub fn classify(f: &Facts) -> Option<Category> {
    if !matches!(f.entity_type, 12 | 13 | 14 | 15 | 17 | 18) {
        return None;
    }
    if f.objective {
        return Some(Category::Objective);
    }
    // A marker, not decoration: the race clip also glues 23,000 plain lights to things.
    if f.rides && matches!(f.entity_type, 13 | 14) && (f.zone_shape != 0 || f.mp_non_default || f.no_model || f.megalo_ref) {
        return Some(Category::AttachedMarker);
    }
    // Equipment (type 12) carries default shapes of its own (a drop shield's bubble): not a zone.
    if f.zone_shape != 0 && matches!(f.entity_type, 13 | 14 | 18) {
        return Some(Category::Zone);
    }
    if !f.placed && f.entity_type == 13 && f.mp_non_default {
        return Some(Category::ScriptSpawned);
    }
    if f.placed && (f.label.is_some() || f.spawn_seq != 0) && matches!(f.entity_type, 12 | 13 | 18) {
        return Some(Category::LabelledForge);
    }
    if f.megalo_ref {
        return Some(Category::MegaloRef);
    }
    if f.no_model && matches!(f.entity_type, 12 | 13 | 14 | 18) {
        return Some(Category::Invisible);
    }
    None
}

/// The facts of one object row.
pub fn facts_of<'a>(r: &'a ObjRow, rides: bool, objective: bool, no_model: bool) -> Facts<'a> {
    Facts {
        entity_type: r.entity_type,
        placed: r.placed.is_some(),
        label: r.label(),
        spawn_seq: r.placed.as_ref().map_or(0, |p| p.spawn_seq),
        zone_shape: r.boundary().map_or(0, |b| b.0),
        mp_non_default: r.mp.map_or(false, |m| m.team >= 0 || m.shape != 0 || (m.flags & !0x10) != 0),
        rides,
        megalo_ref: r.megalo_ref,
        objective,
        no_model,
    }
}

/// The entities that carry Megalo VARIABLES (not highlighted): `(entity type, count)` per scope.
pub fn variable_carriers(idx: &FilmIndex) -> [(&'static str, usize); 3] {
    let n = |t: u8| idx.tracks.iter().filter(|x| x.entity_type == t).count();
    [("player (game-engine-player)", n(3)), ("team (game-engine-megalo-team)", n(4)), ("global (custom-engine-globals)", n(0) + n(1))]
}

// ---------------------------------------------------------------------------------------------
// the per-film table
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Item {
    pub net_id: u32,
    pub entity_type: u8,
    pub cat: Category,
    /// Label if the Forge variant has one, else the tag leaf, else the entity type.
    pub name: String,
    pub tag_leaf: String,
    pub label: Option<String>,
    pub team: i16,
    pub first_tick: u32,
    pub destroyed_tick: Option<u32>,
    pub obj_tag: Option<u32>,
    /// `(shape, [w_or_r, length, top, bottom])` in world units.
    pub zone: Option<(u8, [f32; 4])>,
}

impl Item {
    pub fn alive_at(&self, t: u32) -> bool {
        t >= self.first_tick && self.destroyed_tick.map_or(true, |d| t < d)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub items: Vec<Item>,
    pub counts: [usize; 7],
}

/// `defs`: wire -> (tag id, group, tag path), for every distinct definition (the driver's
/// `def_tags`). `objective`: the object lane's objective verdict for a net id.
pub fn build(
    f: &Film,
    idx: &FilmIndex,
    defs: &HashMap<u32, (u32, String, String)>,
    megalo: Option<&crate::film_megalo::MegaloSeries>,
    objective: &dyn Fn(u32) -> bool,
    no_model: &dyn Fn(Option<u32>, &str) -> bool,
) -> Table {
    let table = ObjectTable::build(f, idx, None, megalo);
    let by: HashMap<u32, &crate::film_decode::EntityTrack> = idx.tracks.iter().map(|t| (t.net_id, t)).collect();
    let mut out = Table::default();
    for r in &table.rows {
        let Some(tr) = by.get(&r.net_id) else { continue };
        let def = r.def_wire.and_then(|w| defs.get(&w));
        let tag_leaf = def.map(|d| d.2.rsplit('\\').next().unwrap_or(&d.2).to_string()).unwrap_or_default();
        let facts = facts_of(r, tr.rides(), objective(r.net_id), no_model(def.map(|d| d.0), &tag_leaf));
        let Some(cat) = classify(&facts) else { continue };
        let label = r.label().map(|s| s.to_string());
        let name = label.clone().filter(|s| !s.is_empty()).or_else(|| (!tag_leaf.is_empty()).then(|| tag_leaf.clone())).unwrap_or_else(|| r.type_name().to_string());
        out.counts[cat as usize] += 1;
        out.items.push(Item {
            net_id: r.net_id,
            entity_type: r.entity_type,
            cat,
            name,
            tag_leaf,
            label,
            team: r.mp.map(|m| m.team as i16).filter(|t| *t >= 0).or(r.placed.as_ref().filter(|p| p.team != 0xFF).map(|p| p.team as i16)).unwrap_or(-1),
            first_tick: r.first_tick,
            destroyed_tick: r.destroyed_tick,
            obj_tag: def.map(|d| d.0),
            zone: r.boundary().map(|b| (b.0, b.1)),
        });
    }
    out
}

/// What the panel/overlay shows. `mask` has [`Category::bit`] set per shown category.
#[derive(Clone, Debug)]
pub struct Filter {
    pub mask: u8,
    pub text: String,
}

impl Default for Filter {
    fn default() -> Self {
        Filter { mask: 0x7F, text: String::new() }
    }
}

impl Filter {
    pub fn matches(&self, it: &Item) -> bool {
        if self.mask & it.cat.bit() == 0 {
            return false;
        }
        if self.text.is_empty() {
            return true;
        }
        let t = self.text.to_ascii_lowercase();
        it.name.to_ascii_lowercase().contains(&t)
            || it.tag_leaf.to_ascii_lowercase().contains(&t)
            || it.label.as_deref().map_or(false, |l| l.to_ascii_lowercase().contains(&t))
    }
}

// ---------------------------------------------------------------------------------------------
// extents
// ---------------------------------------------------------------------------------------------

/// What a tag has to show: whether it has a render model, and the local-space bounds of its
/// collision hull (else its physics hull).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HullInfo {
    pub has_model: bool,
    pub bounds: Option<([f32; 3], [f32; 3])>,
    /// 0 none, 1 collision, 2 physics.
    pub source: u8,
}

/// Triangles an object's render model would draw (0 when it has none or cannot be opened).
pub fn render_triangles(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, mode_tag: u32) -> u32 {
    let Ok(h) = native.open_model(cache, mode_tag) else { return 0 };
    let mut bytes = 0usize;
    for i in 0..native.model_submesh_count(h) {
        if let Some(g) = native.model_decode_geometry(h, i) {
            bytes += g.indices.len();
        }
    }
    native.close_model(h);
    (bytes / 6) as u32
}

/// `name`: the tag path (an `invisible_*` object is placeholder-drawn whatever its model says).
/// A render model with 4 triangles or fewer is the engine's own "invisible collision wall"
/// placeholder nub (`scene.rs`: `render_empty || render_ext < 0.25`), so it counts as no model.
pub fn hull_info(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, obj_tag: u32, name: &str) -> HullInfo {
    let mode = native.resolve_mode(cache, obj_tag);
    let drawable = mode != 0 && mode != 0xFFFF_FFFF && render_triangles(native, cache, mode) > 4 && !name.to_ascii_lowercase().contains("invisible");
    let mut h = HullInfo { has_model: drawable, ..Default::default() };
    let bounds = |v: &[f32]| -> Option<([f32; 3], [f32; 3])> {
        if v.len() < 9 {
            return None;
        }
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in v.chunks_exact(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo.iter().all(|x| x.is_finite()) && hi.iter().all(|x| x.is_finite())).then_some((lo, hi))
    };
    let coll = native.resolve_coll(cache, obj_tag);
    if coll != 0 && coll != 0xFFFF_FFFF {
        if let Some((v, _)) = native.coll_geometry(cache, coll) {
            if let Some(b) = bounds(&v) {
                h.bounds = Some(b);
                h.source = 1;
                return h;
            }
        }
    }
    let phmo = native.resolve_phmo(cache, obj_tag);
    if phmo != 0 && phmo != 0xFFFF_FFFF {
        if let Some((v, _)) = native.phmo_geometry(cache, phmo) {
            if let Some(b) = bounds(&v) {
                h.bounds = Some(b);
                h.source = 2;
            }
        }
    }
    h
}

/// Half size of the stand-in cube drawn when an object has no zone and no hull at all.
pub const FALLBACK_HALF: f32 = 0.3;

/// A live item at one tick: where it is and how it is turned.
#[derive(Clone, Copy, Debug)]
pub struct Live {
    pub item: usize,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
}

/// The items that exist at `tick` and pass the filter, placed by the recording (riders follow
/// their parent, see `film_objective::follow_position`).
pub fn live_at(table: &Table, flt: &Filter, idx: &FilmIndex, track_of: &HashMap<u32, usize>, tick: u32) -> Vec<Live> {
    let lookup = |n: u32| track_of.get(&n).and_then(|&i| idx.tracks.get(i));
    let mut out = Vec::new();
    for (i, it) in table.items.iter().enumerate() {
        if !it.alive_at(tick) || !flt.matches(it) {
            continue;
        }
        let Some(tr) = lookup(it.net_id) else { continue };
        let Some(p) = crate::film_objective::follow_position(&lookup, tr, tick as f32, 0) else { continue };
        let (fwd, up) = tr.orientation_at(tick).unwrap_or(([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]));
        out.push(Live { item: i, pos: p.to_array(), fwd, up });
    }
    out
}

pub type Segs = Vec<([f32; 3], [f32; 3], [f32; 3])>;
pub type Tris = Vec<([f32; 3], [f32; 4])>;

fn bval(v: f32) -> u16 {
    crate::wu_to_bval(v)
}

/// Wire + translucent fill for one live item, in world space.
/// * a zone shape -> the engine-style zone volume (`boundary_geometry`), recoloured by category;
/// * else the oriented bounds of the collision/physics hull, wire always and a fill when the
///   object has no render model (`has_model == false`: the placeholder case);
/// * else a small cube.
pub fn item_geometry(it: &Item, l: &Live, hull: HullInfo, segs: &mut Segs, tris: &mut Tris) {
    let col = it.cat.colour();
    let pos = glam::Vec3::from(l.pos);
    let (fwd, up) = (glam::Vec3::from(l.fwd), glam::Vec3::from(l.up));
    if let Some((shape, d)) = it.zone.filter(|z| z.0 != 0) {
        let b = match shape {
            1 => [bval(d[0]), 0, 0, 0],
            2 => [bval(d[0]), bval(d[2]), bval(d[3]), 0],
            _ => [bval(d[0]), bval(d[1]), bval(d[2]), bval(d[3])],
        };
        let (s, t) = crate::boundary_geometry(shape, b, 0, pos, fwd, up);
        segs.extend(s.into_iter().map(|(a, b, _)| (a, b, col)));
        tris.extend(t.into_iter().map(|(p, c)| (p, [col[0], col[1], col[2], c[3].max(0.10)])));
        return;
    }
    let (lo, hi, filled) = match hull.bounds {
        Some((lo, hi)) => (glam::Vec3::from(lo), glam::Vec3::from(hi), !hull.has_model),
        None => (glam::Vec3::splat(-FALLBACK_HALF), glam::Vec3::splat(FALLBACK_HALF), true),
    };
    let m = crate::film_decode::basis_mat3(l.fwd, l.up);
    let c = |x: f32, y: f32, z: f32| -> glam::Vec3 { pos + m * glam::Vec3::new(x, y, z) };
    let cs = [
        c(lo.x, lo.y, lo.z), c(hi.x, lo.y, lo.z), c(hi.x, hi.y, lo.z), c(lo.x, hi.y, lo.z),
        c(lo.x, lo.y, hi.z), c(hi.x, lo.y, hi.z), c(hi.x, hi.y, hi.z), c(lo.x, hi.y, hi.z),
    ];
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
        segs.push((cs[a].into(), cs[b].into(), col));
    }
    if filled {
        let fill = [col[0], col[1], col[2], 0.18];
        for &(a, b, c2, d) in &[(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)] {
            for k in [a, b, c2, a, c2, d] {
                tris.push((cs[k].into(), fill));
            }
        }
    }
    let _ = up;
}

/// Short on-screen text for an item: `name #id [team]`.
pub fn label_text(it: &Item) -> String {
    let mut s = it.name.clone();
    s.push_str(&format!(" #{:X}", it.net_id & 0xFFFF));
    if it.team >= 0 {
        s.push_str(&format!(" t{}", it.team));
    }
    s
}

// ---------------------------------------------------------------------------------------------
// persistence
// ---------------------------------------------------------------------------------------------

const SETTING_FILE: &str = "film_highlight.txt";

/// `(on, category mask)`. Absent / garbage = off with every category.
pub fn parse_setting(s: &str) -> (bool, u8) {
    let mut it = s.split_whitespace();
    let on = it.next() == Some("1");
    let mask = it.next().and_then(|m| m.parse::<u8>().ok()).map_or(0x7F, |m| m & 0x7F);
    (on, mask)
}

pub fn load_setting() -> (bool, u8) {
    crate::project::settings_dir()
        .and_then(|d| std::fs::read_to_string(d.join(SETTING_FILE)).ok())
        .map_or((false, 0x7F), |s| parse_setting(&s))
}

pub fn save_setting(on: bool, mask: u8) {
    if let Some(d) = crate::project::settings_dir() {
        let _ = std::fs::create_dir_all(&d);
        let _ = std::fs::write(d.join(SETTING_FILE), format!("{} {}", on as u8, mask));
    }
}

// ---------------------------------------------------------------------------------------------
// the Theater state + panel
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Row {
    pub net_id: u32,
    pub cat: Category,
    pub text: String,
    pub pos: [f32; 3],
    /// Drawn as a placeholder (no drawable model).
    pub placeholder: bool,
}

#[derive(Clone, Debug)]
pub struct Label {
    pub pos: [f32; 3],
    pub text: String,
    pub colour: [f32; 3],
}

/// Everything the Theater overlay remembers. `on` and the category mask persist.
pub struct State {
    pub on: bool,
    pub flt: Filter,
    pub table: Option<std::sync::Arc<Table>>,
    pub key: (usize, usize),
    pub hulls: HashMap<u32, HullInfo>,
    pub rows: Vec<Row>,
    pub labels: Vec<Label>,
    /// A row the user clicked this frame (consumed by the app).
    pub click: Option<u32>,
    pub focused: Option<u32>,
    pub follow: Option<u32>,
    pub follow_off: glam::Vec3,
    /// The lanes currently hold OUR geometry.
    pub drawn: bool,
    pub status: String,
    pub carriers: Vec<(&'static str, usize)>,
}

impl State {
    pub fn load() -> State {
        let (on, mask) = load_setting();
        State {
            on,
            flt: Filter { mask, text: String::new() },
            table: None,
            key: (0, 0),
            hulls: HashMap::new(),
            rows: Vec::new(),
            labels: Vec::new(),
            click: None,
            focused: None,
            follow: None,
            follow_off: glam::Vec3::ZERO,
            drawn: false,
            status: String::new(),
            carriers: Vec::new(),
        }
    }
    pub fn save(&self) {
        save_setting(self.on, self.flt.mask);
    }
}

pub fn colour32(c: [f32; 3]) -> egui::Color32 {
    egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

/// The Theater panel section: the switch, the legend (which is also the category filter), the
/// text filter and the list. Returns true when a persisted setting changed.
pub fn panel_ui(ui: &mut egui::Ui, st: &mut State) -> bool {
    let mut changed = false;
    ui.separator();
    if ui
        .checkbox(&mut st.on, "Highlight script objects")
        .on_hover_text("Outline the objects the gametype's script is responsible for (zones, hill markers, script-spawned and labelled objects, objectives) and draw a placeholder for ones with no visible model. Off by default.")
        .changed()
    {
        changed = true;
    }
    if st.on {
        ui.horizontal_wrapped(|ui| {
            for c in Category::ALL {
                let mut on = st.flt.mask & c.bit() != 0;
                let n = st.table.as_ref().map_or(0, |t| t.counts[c as usize]);
                let txt = egui::RichText::new(format!("{} ({n})", c.name())).color(colour32(c.colour()));
                if ui.checkbox(&mut on, txt).changed() {
                    st.flt.mask = if on { st.flt.mask | c.bit() } else { st.flt.mask & !c.bit() };
                    changed = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Filter");
            ui.text_edit_singleline(&mut st.flt.text).on_hover_text("Forge label or tag name contains...");
        });
        if !st.status.is_empty() {
            ui.small(&st.status);
        }
        if !st.carriers.is_empty() {
            ui.small(format!(
                "Variable-carrying entities (not highlighted): {}",
                st.carriers.iter().map(|(n, c)| format!("{n} x{c}")).collect::<Vec<_>>().join(", ")
            ));
        }
        ui.small("Click a row to move the camera to it; click it again to follow it.");
        egui::ScrollArea::vertical().id_salt("film_hl_rows").max_height(220.0).show(ui, |ui| {
            for r in &st.rows {
                let mark = if st.follow == Some(r.net_id) { "\u{25B6} " } else if st.focused == Some(r.net_id) { "\u{25CF} " } else { "" };
                let txt = egui::RichText::new(format!("{mark}{}{}", r.text, if r.placeholder { "  [placeholder]" } else { "" })).color(colour32(r.cat.colour()));
                if ui.selectable_label(st.focused == Some(r.net_id), txt).clicked() {
                    st.click = Some(r.net_id);
                }
            }
        });
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f() -> Facts<'static> {
        Facts { entity_type: 13, ..Default::default() }
    }

    #[test]
    fn classification_is_a_priority_list() {
        // nothing special: not a Megalo object
        assert_eq!(classify(&f()), None);
        // a placed, unlabelled, default Forge block
        assert_eq!(classify(&Facts { placed: true, ..f() }), None);
        // priority: objective beats everything, attached beats zone, zone beats spawned
        assert_eq!(classify(&Facts { objective: true, rides: true, zone_shape: 2, ..f() }), Some(Category::Objective));
        // a plain object riding something is decoration, not a marker
        assert_eq!(classify(&Facts { rides: true, ..f() }), None);
        assert_eq!(classify(&Facts { rides: true, no_model: true, ..f() }), Some(Category::AttachedMarker));
        // equipment never counts as a zone
        assert_eq!(classify(&Facts { entity_type: 12, zone_shape: 1, ..Default::default() }), None);
        assert_eq!(classify(&Facts { rides: true, zone_shape: 2, ..f() }), Some(Category::AttachedMarker));
        assert_eq!(classify(&Facts { zone_shape: 2, placed: true, ..f() }), Some(Category::Zone));
        assert_eq!(classify(&Facts { mp_non_default: true, ..f() }), Some(Category::ScriptSpawned));
        // a placed object that is merely "non default" is not script-spawned
        assert_eq!(classify(&Facts { mp_non_default: true, placed: true, ..f() }), None);
        assert_eq!(classify(&Facts { placed: true, label: Some("race_flag"), ..f() }), Some(Category::LabelledForge));
        assert_eq!(classify(&Facts { placed: true, spawn_seq: 3, ..f() }), Some(Category::LabelledForge));
        assert_eq!(classify(&Facts { placed: true, megalo_ref: true, ..f() }), Some(Category::MegaloRef));
        // lowest priority: a placed, plain, model-less object is only a placeholder case
        assert_eq!(classify(&Facts { placed: true, no_model: true, ..f() }), Some(Category::Invisible));
        assert_eq!(classify(&Facts { placed: true, no_model: true, label: Some("x"), ..f() }), Some(Category::LabelledForge));
        assert_eq!(classify(&Facts { entity_type: 17, no_model: true, ..Default::default() }), None);
    }

    #[test]
    fn players_projectiles_and_held_weapons_are_never_highlighted() {
        for t in [11u8, 16, 3, 4, 5, 0, 1] {
            assert_eq!(classify(&Facts { entity_type: t, objective: true, rides: true, zone_shape: 3, megalo_ref: true, ..Default::default() }), None, "type {t}");
        }
        // a weapon riding a hand is not an attached MARKER (only generic objects are)
        assert_eq!(classify(&Facts { entity_type: 17, rides: true, ..Default::default() }), None);
        // ...but a weapon a variable refers to is
        assert_eq!(classify(&Facts { entity_type: 17, rides: true, megalo_ref: true, ..Default::default() }), Some(Category::MegaloRef));
    }

    #[test]
    fn filter_and_setting_round_trip() {
        let it = Item { net_id: 1, entity_type: 13, cat: Category::Zone, name: "speed_limit".into(), tag_leaf: "mp_hill_beacon".into(), label: Some("speed_limit".into()), team: -1, first_tick: 0, destroyed_tick: None, obj_tag: None, zone: None };
        let mut fl = Filter::default();
        assert!(fl.matches(&it));
        fl.text = "BEACON".into();
        assert!(fl.matches(&it));
        fl.text = "pit".into();
        assert!(!fl.matches(&it));
        fl.text.clear();
        fl.mask = Category::AttachedMarker.bit();
        assert!(!fl.matches(&it));
        assert_eq!(parse_setting("1 5"), (true, 5));
        assert_eq!(parse_setting(""), (false, 0x7F));
        assert_eq!(parse_setting("garbage"), (false, 0x7F));
        assert_eq!(Category::ALL.iter().fold(0u8, |m, c| m | c.bit()), 0x7F);
    }
}

#[cfg(test)]
mod gates {
    use super::*;

    fn race_clip() -> Option<crate::film::Film> {
        let p = std::env::var_os("HMS_FILM").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Downloads").join("Reach(1).mov")))?;
        let f = crate::film::open(&p).ok()?;
        (f.header.game_variant.title.contains("Ignition")).then_some(f)
    }

    fn items(f: &Film, idx: &FilmIndex) -> Table {
        let m = crate::film_megalo::decls_of(f).map(|_| crate::film_megalo::MegaloSeries::build(f));
        build(f, idx, &Default::default(), m.as_ref(), &|_| false, &|_, leaf| leaf.contains("invisible"))
    }

    /// #film-highlight  GATE (what counts as a Megalo object, on the race clip). Without a map
    /// cache (so no tag names): the 37 hill markers the script glues to players, the placed zones,
    /// and the variable-referenced vehicles; and none of them is a player.
    #[test]
    fn the_race_clips_script_objects_are_classified_by_what_the_recording_carries() {
        let Some(f) = race_clip() else {
            eprintln!("SKIPPED: race clip (HT Ignition v2.1) not on this machine");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let t = items(&f, &idx);
        let n = |c: Category| t.counts[c as usize];
        assert_eq!(n(Category::AttachedMarker), 37, "the hill markers glued to players");
        assert_eq!(n(Category::Zone), 75, "placed zones: 5 race_flag + 16 speed_limit + 6 Wear Zone + 14 pit_stop + 2 spectator + 32 plain hill beacons");
        assert_eq!(n(Category::MegaloRef), 24, "vehicles a script variable refers to");
        for (label, want) in [("race_flag", 5), ("speed_limit", 16), ("Wear Zone", 6), ("pit_stop", 14), ("Spectator", 2)] {
            assert_eq!(t.items.iter().filter(|i| i.cat == Category::Zone && i.label.as_deref() == Some(label)).count(), want, "{label}");
        }
        assert!(t.items.iter().all(|i| i.entity_type != 11 && i.entity_type != 16), "no body or projectile is ever highlighted");
        let carriers = variable_carriers(&idx);
        assert_eq!(carriers[0].1, 16, "16 player entities carry variables, listed separately");
        // the category filter and the text filter select exactly their rows
        let mut fl = Filter { mask: Category::AttachedMarker.bit(), text: String::new() };
        assert_eq!(t.items.iter().filter(|i| fl.matches(i)).count(), 37);
        fl = Filter { mask: 0x7F, text: "speed_limit".into() };
        assert_eq!(t.items.iter().filter(|i| fl.matches(i)).count(), 16);
    }

    // ---- a CPU rasteriser over the exact geometry the overlay lanes receive ----------------------

    fn raster(segs: &Segs, tris: &Tris, vp: glam::Mat4, w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut img = vec![[0u8; 3]; w * h];
        let proj = |p: [f32; 3]| -> Option<(f32, f32)> {
            let c = vp * glam::Vec4::new(p[0], p[1], p[2], 1.0);
            if c.w <= 1e-4 { return None; }
            Some(((c.x / c.w * 0.5 + 0.5) * w as f32, (1.0 - (c.y / c.w * 0.5 + 0.5)) * h as f32))
        };
        let q = |c: [f32; 3]| [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8];
        for t in tris.chunks_exact(3) {
            let (Some(a), Some(b), Some(c)) = (proj(t[0].0), proj(t[1].0), proj(t[2].0)) else { continue };
            let col = q([t[0].1[0], t[0].1[1], t[0].1[2]]);
            let (x0, x1) = (a.0.min(b.0).min(c.0).floor().max(0.0) as usize, (a.0.max(b.0).max(c.0).ceil() as usize).min(w - 1));
            let (y0, y1) = (a.1.min(b.1).min(c.1).floor().max(0.0) as usize, (a.1.max(b.1).max(c.1).ceil() as usize).min(h - 1));
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area.abs() < 1e-6 { continue; }
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((b.0 - px) * (c.1 - py) - (c.0 - px) * (b.1 - py)) / area;
                    let w1 = ((c.0 - px) * (a.1 - py) - (a.0 - px) * (c.1 - py)) / area;
                    if w0 >= 0.0 && w1 >= 0.0 && w0 + w1 <= 1.0 { img[y * w + x] = col; }
                }
            }
        }
        for (a, b, c) in segs {
            let (Some(pa), Some(pb)) = (proj(*a), proj(*b)) else { continue };
            let n = ((pb.0 - pa.0).abs().max((pb.1 - pa.1).abs()) * 2.0).ceil().max(1.0) as usize;
            for k in 0..=n {
                let f = k as f32 / n as f32;
                let (x, y) = (pa.0 + (pb.0 - pa.0) * f, pa.1 + (pb.1 - pa.1) * f);
                if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h { img[y as usize * w + x as usize] = q(*c); }
            }
        }
        img
    }

    /// #film-highlight  GATE (invisible objects reach the screen). `invisible_cube_of_alarming_2`
    /// (the invisible-cube family of `objects\multi\invisible_cube_of_derek`) has a render model
    /// of no triangles and a 0.167 wu physics cube. With the highlight on, the placeholder must
    /// put its category colour on the pixels of that cube: wire at the 8 projected corners, fill at
    /// its centre, nothing outside its projected bounds. Rasterised on the CPU from the geometry
    /// the overlay lanes are handed, through a real perspective projection (see the module note
    /// on what this does and does not prove).
    #[test]
    #[ignore = "opens the native library: map-backed gate, run one module per process with --ignored --test-threads=1"]
    fn an_invisible_object_gets_a_placeholder_on_the_pixels_of_its_real_extents() {
        placeholder_on_pixels("4ED6DDC1_67FF2BA9", "invisible_cube_of_alarming_2", 10);
        // the object the request named, in the clip that has it
        placeholder_on_pixels("EFC42DFE_6A951540", "invisible_cube_of_derek", 7);
    }

    /// The part of the placeholder gate that needs no map: the geometry the overlay lanes get for a
    /// model-less object lands, coloured, on the pixels of its real extents.
    fn placeholder_pixels(it: &Item, l: &Live, hull: HullInfo) {
        let (lo, hi) = hull.bounds.expect("a physics cube");
        let (mut segs, mut tris) = (Vec::new(), Vec::new());
        item_geometry(it, l, hull, &mut segs, &mut tris);
        assert_eq!(segs.len(), 12);
        assert_eq!(tris.len(), 36);
        // a camera 0.9 wu away, looking at the cube's centre
        let m = crate::film_decode::basis_mat3(l.fwd, l.up);
        let centre = glam::Vec3::from(l.pos) + m * ((glam::Vec3::from(lo) + glam::Vec3::from(hi)) * 0.5);
        let eye = centre + glam::Vec3::new(0.5, 0.5, 0.4);
        let (w, h) = (320usize, 240usize);
        let vp = glam::Mat4::perspective_rh(1.0, w as f32 / h as f32, 0.05, 100.0) * glam::Mat4::look_at_rh(eye, centre, glam::Vec3::Z);
        let img = raster(&segs, &tris, vp, w, h);
        let col = { let c = it.cat.colour(); [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8] };
        let proj = |p: glam::Vec3| { let c = vp * p.extend(1.0); (((c.x / c.w * 0.5 + 0.5) * w as f32) as i32, ((1.0 - (c.y / c.w * 0.5 + 0.5)) * h as f32) as i32) };
        let at = |x: i32, y: i32| img.get(y as usize * w + x as usize).copied();
        let near = |x: i32, y: i32| (-1..=1).any(|dy| (-1..=1).any(|dx| at(x + dx, y + dy) == Some(col)));
        let (cx, cy) = proj(centre);
        assert_eq!(at(cx, cy), Some(col), "the fill covers the cube's centre pixel");
        let mut xs = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for k in 0..8u8 {
            let corner = glam::Vec3::from(l.pos) + m * glam::Vec3::new(if k & 1 == 0 { lo[0] } else { hi[0] }, if k & 2 == 0 { lo[1] } else { hi[1] }, if k & 4 == 0 { lo[2] } else { hi[2] });
            let (x, y) = proj(corner);
            assert!(near(x, y), "corner {k} at ({x},{y}) is coloured");
            xs = (xs.0.min(x), xs.1.max(x), xs.2.min(y), xs.3.max(y));
        }
        let lit = img.iter().filter(|p| **p == col).count();
        assert!(lit > 1500, "the placeholder covers a real area ({lit} px)");
        let outside = img.iter().enumerate().filter(|(i, p)| **p != [0, 0, 0] && { let (x, y) = ((i % w) as i32, (i / w) as i32); x < xs.0 - 2 || x > xs.1 + 2 || y < xs.2 - 2 || y > xs.3 + 2 }).count();
        assert_eq!(outside, 0, "nothing is drawn outside the projected extents");
    }

    /// #film-highlight  GATE (default suite, no map or film): the placeholder for a model-less
    /// object with an authored 0.1672 wu cube hull, turned and placed arbitrarily, lands on the
    /// pixels of its projected extents. Bites if the fill or the hull bounds are dropped.
    #[test]
    fn a_synthetic_invisible_cube_rasterises_on_its_projected_extents() {
        let it = Item { net_id: 7, entity_type: 13, cat: Category::AttachedMarker, name: "invisible_cube_of_derek".into(), tag_leaf: "invisible_cube_of_derek".into(), label: None, team: -1, first_tick: 0, destroyed_tick: None, obj_tag: None, zone: None };
        let hull = HullInfo { has_model: false, bounds: Some(([-0.0836; 3], [0.0836; 3])), source: 2 };
        let l = Live { item: 0, pos: [10.0, 20.0, 3.0], fwd: [0.6, 0.8, 0.0], up: [0.0, 0.0, 1.0] };
        placeholder_pixels(&it, &l, hull);
        // a drawable model gets the wire only: the centre pixel is not filled
        let (mut s, mut t) = (Vec::new(), Vec::new());
        item_geometry(&it, &l, HullInfo { has_model: true, ..hull }, &mut s, &mut t);
        assert_eq!((s.len(), t.len()), (12, 0));
    }

    fn placeholder_on_pixels(clip: &str, leaf: &str, min_items: usize) {
        let Some((native, cache)) = crate::film_action::objective_inventory::open_reach("forge_halo") else {
            eprintln!("SKIPPED: forge_halo not available");
            return;
        };
        let Some(path) = crate::film::spec_corpus().into_iter().find(|p| p.to_string_lossy().contains(clip)) else {
            eprintln!("SKIPPED: clip with invisible_cube objects not on this machine");
            return;
        };
        let map = crate::mapcat::enumerate().into_iter().find(|c| c.game == crate::mapcat::Game::Reach && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))).unwrap();
        let names = crate::film_stats::MapNames::build_with_map(&native, cache, Some(&map.path));
        use crate::film_stats::NameSource;
        let f = crate::film::open(&path).unwrap();
        let idx = crate::film_decode::build_index_film(&f);
        let mut defs = HashMap::new();
        for t in &idx.tracks {
            if let Some(w) = t.def_wire { if let Some(d) = names.definition(w) { defs.insert(w, d); } }
        }
        let tab = build(&f, &idx, &defs, None, &|_| false, &|tag, leaf| tag.map_or(false, |t| !hull_info(&native, cache, t, leaf).has_model));
        let cubes: Vec<&Item> = tab.items.iter().filter(|i| i.tag_leaf == leaf).collect();
        assert!(cubes.len() >= min_items, "liveness: the clip has invisible cubes ({})", cubes.len());
        let track_of: HashMap<u32, usize> = idx.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
        // the attached cube objects (no zone shape: they take the hull path)
        let it = cubes.iter().find(|i| i.zone.is_none() && i.cat == Category::AttachedMarker).expect("an attached invisible cube");
        let tick = (it.first_tick + 5).min(idx.ticks - 1);
        let live = live_at(&tab, &Filter::default(), &idx, &track_of, tick);
        let l = live.iter().find(|l| tab.items[l.item].net_id == it.net_id).expect("alive just after it spawns");
        let hull = hull_info(&native, cache, it.obj_tag.unwrap(), &it.tag_leaf);
        assert!(!hull.has_model, "the model draws nothing");
        let (lo, hi) = hull.bounds.expect("a physics cube");
        assert!(((hi[0] - lo[0]) - 0.1672).abs() < 1e-3, "authored cube edge");
        placeholder_pixels(it, l, hull);
        // OFF means nothing: a filter with every category cleared yields no live items
        assert!(live_at(&tab, &Filter { mask: 0, text: String::new() }, &idx, &track_of, tick).is_empty());
        // cost: a frame's geometry for the 400 nearest items
        let all = live_at(&tab, &Filter::default(), &idx, &track_of, tick);
        let t0 = std::time::Instant::now();
        let (mut s2, mut t2) = (Vec::new(), Vec::new());
        for l in all.iter().take(400) { item_geometry(&tab.items[l.item], l, hull, &mut s2, &mut t2); }
        eprintln!("HIGHLIGHT geometry for {} items: {:.2} ms, {} segs, {} tris", all.len().min(400), t0.elapsed().as_secs_f64() * 1e3, s2.len(), t2.len());
    }
}
