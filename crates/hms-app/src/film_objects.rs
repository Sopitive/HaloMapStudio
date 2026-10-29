//! #film-objects  What a saved film says about its OBJECTS: identity, lifetime, motion, the
//! multiplayer properties the engine replicates (team, flags, boundary shape and size), and the
//! Forge placement record -- label, spawn sequence, respawn -- the embedded map variant adds.
//!
//! # Where each property comes from (PROVEN = `haloreach.dll` / `reach_tag_test.exe` + measured)
//!
//! * **Replicated, per object, in the per-tick stream** (base field 11, `multiplayer-properties`
//!   in `c_simulation_object_entity_definition::object_update_decode`): a 15-bit **data flags**
//!   word, a 4-bit **team designator** (stored minus 1), then a **boundary shape** (2 bits: 0 none,
//!   1 sphere, 2 cylinder, 3 box) with 11-bit quantised dimensions in `[0, 200]` with exact
//!   endpoints -- sphere: radius; cylinder: radius, +height, -height; box: width, length,
//!   +height, -height (`sub_140E27850`). PROVEN.
//! * **NOT replicated**: the Forge LABEL, the spawn sequence, the respawn time, the colour. The
//!   engine keeps those in the map variant only. The film embeds the whole map variant, so HMS
//!   JOINS an entity to its placement by position (the entity's first replicated position equals
//!   the placement's, to within `JOIN_TOLERANCE`). Joined labels are INFERRED-by-position and
//!   carry their `join_dist`; an object the gametype spawned at runtime has no placement and so no
//!   label.
//! * Object TYPE: the engine entity type (biped / weapon / vehicle / generic ...) is stated; the
//!   object DEFINITION (tag) is the 14-bit network tag reference in the creation message, resolved
//!   to a tag path only when a map cache is supplied.

use crate::film::Film;
use crate::film_decode::{BitReader, DecodeStop, EntityTrack, FilmIndex};
use std::collections::{HashMap, HashSet};

/// How close an entity's first position must be to a variant placement to be joined to it (world
/// units). Replicated positions are quantised to a fraction of a unit.
pub const JOIN_TOLERANCE: f32 = 0.25;

pub const ENTITY_TYPE_NAMES: [&str; 19] = [
    "sandbox_globals",
    "custom_globals",
    "survival_globals",
    "player",
    "megalo_team",
    "statborg",
    "breakable_surface_group",
    "map_variant",
    "game_ai",
    "debug_globals",
    "campaign_metagame",
    "biped",
    "item",
    "generic",
    "generic_garbage",
    "vehicle",
    "projectile",
    "weapon",
    "device",
];

pub fn entity_type_name(t: u8) -> &'static str {
    ENTITY_TYPE_NAMES.get(t as usize).copied().unwrap_or("?")
}

pub fn shape_name(s: u8) -> &'static str {
    match s {
        0 => "none",
        1 => "sphere",
        2 => "cylinder",
        3 => "box",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------------------------
// replicated multiplayer properties
// ---------------------------------------------------------------------------------------------

/// Base field 11. `dims` is `[width_or_radius, length, top, bottom]`: sphere uses `[0]`, cylinder
/// `[0]`, `[2]`, `[3]`, box all four.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MpProps {
    pub flags: u16,
    /// `team designator - 1`: -1 = none.
    pub team: i8,
    pub shape: u8,
    pub dims: [f32; 4],
}

/// `sub_1403C4B70(.., min, max, bits, 0, exact_endpoints = 1)` -> `Bitstream_SkipBits`: raw 0 is
/// the minimum, raw `2^bits - 1` the maximum, anything between sits on a `(2^bits - 2)`-step grid.
pub fn qreal_exact_endpoints(raw: u32, bits: u32, lo: f32, hi: f32) -> f32 {
    let n = 1u32 << bits;
    if raw == 0 {
        lo
    } else if raw == n - 1 {
        hi
    } else {
        let step = (hi - lo) / (n - 2) as f32;
        (raw - 1) as f32 * step + step * 0.5 + lo
    }
}

pub fn read_mp(r: &mut BitReader) -> Result<MpProps, DecodeStop> {
    let flags = r.bits(15)? as u16;
    let team = r.bits(4)? as i8 - 1;
    let shape = r.bits(2)? as u8;
    let mut q = |r: &mut BitReader| -> Result<f32, DecodeStop> { Ok(qreal_exact_endpoints(r.bits(11)? as u32, 11, 0.0, 200.0)) };
    let mut dims = [0.0f32; 4];
    match shape {
        1 => dims[0] = q(r)?,
        2 => {
            dims[0] = q(r)?;
            dims[2] = q(r)?;
            dims[3] = q(r)?;
        }
        3 => {
            dims[0] = q(r)?;
            dims[1] = q(r)?;
            dims[2] = q(r)?;
            dims[3] = q(r)?;
        }
        _ => {}
    }
    Ok(MpProps { flags, team, shape, dims })
}

// ---------------------------------------------------------------------------------------------
// the boundary volume
// ---------------------------------------------------------------------------------------------

/// An oriented boundary volume. `forward`/`up` are the object's recorded basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Volume {
    pub center: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub shape: u8,
    pub dims: [f32; 4],
}

impl Volume {
    /// Is `p` inside? Local frame: x = forward, z = up. Sphere: distance <= radius. Cylinder:
    /// horizontal distance <= radius and `-bottom <= z <= top`. Box: `|x| <= width`, `|y| <= length` (HALF
    /// extents, width along forward -- the editor's zone draw convention), `-bottom <= z <= top`.
    pub fn contains(&self, p: [f32; 3]) -> bool {
        if self.shape == 0 {
            return false;
        }
        let m = crate::film_decode::basis_mat3(self.forward, self.up);
        let d = glam::Vec3::from(p) - glam::Vec3::from(self.center);
        let l = m.transpose() * d;
        match self.shape {
            1 => d.length() <= self.dims[0],
            2 => (l.x * l.x + l.y * l.y).sqrt() <= self.dims[0] && l.z >= -self.dims[3] && l.z <= self.dims[2],
            // Box: width along the object's FORWARD axis, length along its RIGHT axis, both HALF
            // extents -- the convention the editor's own zone draw (`boundary_geometry`) uses and the
            // owner has verified on screen.
            3 => l.x.abs() <= self.dims[0] && l.y.abs() <= self.dims[1] && l.z >= -self.dims[3] && l.z <= self.dims[2],
            _ => false,
        }
    }
}


// ---------------------------------------------------------------------------------------------
// the object table
// ---------------------------------------------------------------------------------------------

/// The Forge placement an entity was joined to.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    pub slot: usize,
    pub quota_index: u16,
    pub item: u8,
    pub label_idx: u16,
    pub label: Option<String>,
    pub team: u8,
    pub spawn_seq: i8,
    pub respawn: u8,
    pub shape: u8,
    pub dims: [f32; 4],
    pub placement: u8,
    pub cached_type: u8,
    pub color: i8,
    pub join_dist: f32,
}

#[derive(Clone, Debug)]
pub struct ObjRow {
    pub net_id: u32,
    pub entity_type: u8,
    pub def_wire: Option<u32>,
    pub tag: Option<String>,
    pub first_tick: u32,
    pub last_tick: u32,
    pub destroyed_tick: Option<u32>,
    pub first_pos: Option<[f32; 3]>,
    pub placed: Option<Placed>,
    /// The latest replicated multiplayer properties.
    pub mp: Option<MpProps>,
    pub parent: Option<u32>,
    pub owner_slot: Option<u8>,
    pub owner_name: Option<String>,
    pub megalo_ref: bool,
}

impl ObjRow {
    pub fn type_name(&self) -> &'static str {
        entity_type_name(self.entity_type)
    }
    pub fn end_tick(&self) -> u32 {
        self.destroyed_tick.unwrap_or(self.last_tick)
    }
    pub fn alive_at(&self, t: u32) -> bool {
        t >= self.first_tick && t <= self.end_tick()
    }
    /// The label text (the joined placement's), if any.
    pub fn label(&self) -> Option<&str> {
        self.placed.as_ref().and_then(|p| p.label.as_deref())
    }
    /// The boundary the object states: its replicated one if it ever sent one with a shape, else
    /// the placement's.
    pub fn boundary(&self) -> Option<(u8, [f32; 4], &'static str)> {
        if let Some(m) = self.mp.filter(|m| m.shape != 0) {
            return Some((m.shape, m.dims, "replicated"));
        }
        self.placed.as_ref().filter(|p| p.shape != 0).map(|p| (p.shape, p.dims, "placement"))
    }
}

pub struct ObjectTable {
    pub rows: Vec<ObjRow>,
    by_id: HashMap<u32, usize>,
}

/// Lookup of the open tag names; see `film_stats::NameSource`.
pub type Names<'a> = Option<&'a dyn crate::film_stats::NameSource>;

impl ObjectTable {
    pub fn build(f: &Film, idx: &FilmIndex, names: Names, megalo: Option<&crate::film_megalo::MegaloSeries>) -> ObjectTable {
        let mv = &f.header.map_variant;
        let placements: Vec<&crate::film::VariantObject> = mv.present_objects().collect();
        let mut refs: HashSet<u32> = HashSet::new();
        if let Some(m) = megalo {
            for pts in m.series.values() {
                for (_, v) in pts {
                    if let crate::film_megalo::Value::Object(Some(id)) = v {
                        refs.insert(*id);
                    }
                }
            }
        }
        let mut rows = Vec::new();
        let mut by_id = HashMap::new();
        for tr in &idx.tracks {
            if tr.entity_type > 18 {
                continue;
            }
            let first_pos = tr.positions.first().map(|(_, p)| *p);
            let placed = if matches!(tr.entity_type, 12 | 13 | 14 | 15 | 17 | 18) { first_pos.and_then(|p| join_placement(&placements, mv, p)) } else { None };
            let (owner_slot, owner_name) = if tr.entity_type == 11 {
                match idx.identity_of(tr.net_id) {
                    crate::film_roster::Identity::Player { seat, .. } => (Some(seat.slot), Some(seat.name.clone())),
                    _ => (None, None),
                }
            } else {
                (None, None)
            };
            let tag = tr.def_wire.and_then(|w| names.and_then(|n| n.definition(w))).map(|(_, _, p)| p);
            by_id.insert(tr.net_id, rows.len());
            rows.push(ObjRow {
                net_id: tr.net_id,
                entity_type: tr.entity_type,
                def_wire: tr.def_wire,
                tag,
                first_tick: tr.first_tick,
                last_tick: tr.last_tick,
                destroyed_tick: tr.destroyed_tick,
                first_pos,
                placed,
                mp: idx.mp_props.get(&tr.net_id).and_then(|v| v.last()).map(|(_, m)| *m),
                parent: tr.attachments.iter().find_map(|(_, a)| a.map(|a| a.parent)),
                owner_slot,
                owner_name,
                megalo_ref: refs.contains(&tr.net_id),
            });
        }
        ObjectTable { rows, by_id }
    }

    pub fn get(&self, id: u32) -> Option<&ObjRow> {
        self.by_id.get(&id).map(|&i| &self.rows[i])
    }
}

/// The nearest present placement within [`JOIN_TOLERANCE`] of `p`.
pub fn join_placement(placements: &[&crate::film::VariantObject], mv: &crate::film::EmbeddedMapVariant, p: [f32; 3]) -> Option<Placed> {
    let mut best: Option<(f32, &crate::film::VariantObject)> = None;
    for o in placements {
        let d = ((o.position[0] - p[0]).powi(2) + (o.position[1] - p[1]).powi(2) + (o.position[2] - p[2]).powi(2)).sqrt();
        if d <= JOIN_TOLERANCE && best.map_or(true, |(bd, _)| d < bd) {
            best = Some((d, o));
        }
    }
    best.map(|(d, o)| Placed {
        slot: o.slot,
        quota_index: o.quota_index,
        item: o.item,
        label_idx: o.label_idx,
        label: (o.label_idx != 0xFFFF).then(|| mv.labels.get(o.label_idx as usize).cloned()).flatten(),
        team: o.team,
        spawn_seq: o.spawn_seq,
        respawn: o.respawn,
        shape: o.boundary_shape,
        dims: o.boundary,
        placement: o.placement,
        cached_type: o.cached_type,
        color: o.color,
        join_dist: d,
    })
}

/// The last sample at or before `tick` (HOLD, not interpolate: the stream's own samples are the
/// only thing a live reader has, so batch and live agree).
pub fn hold<T: Copy>(v: &[(u32, T)], tick: u32) -> Option<T> {
    let i = v.partition_point(|(t, _)| *t <= tick);
    if i == 0 { None } else { Some(v[i - 1].1) }
}

pub fn pos_at(tr: &EntityTrack, tick: u32) -> Option<[f32; 3]> {
    hold(&tr.positions, tick)
}

pub fn basis_at(tr: &EntityTrack, tick: u32) -> ([f32; 3], [f32; 3]) {
    let i = tr.orientations.partition_point(|(t, _, _)| *t <= tick);
    if i == 0 { ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) } else { (tr.orientations[i - 1].1, tr.orientations[i - 1].2) }
}

/// The zone volume of `row` at `tick`, from its replicated boundary (else the placement's).
pub fn volume_of(row: &ObjRow, tr: &EntityTrack, tick: u32) -> Option<Volume> {
    let (shape, dims, _) = row.boundary()?;
    let center = pos_at(tr, tick).or(row.first_pos)?;
    let (forward, up) = basis_at(tr, tick);
    Some(Volume { center, forward, up, shape, dims })
}

// ---------------------------------------------------------------------------------------------
// filters
// ---------------------------------------------------------------------------------------------

pub fn glob(pat: &str, s: &str) -> bool {
    let (p, s) = (pat.to_ascii_lowercase(), s.to_ascii_lowercase());
    if !p.contains('*') {
        return s.contains(&p);
    }
    let parts: Vec<&str> = p.split('*').collect();
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match s[pos..].find(part) {
            Some(k) => {
                if i == 0 && !p.starts_with('*') && k != 0 {
                    return false;
                }
                pos += k + part.len();
            }
            None => return false,
        }
    }
    if !p.ends_with('*') {
        if let Some(last) = parts.last() {
            if !last.is_empty() && !s.ends_with(last) {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Debug)]
pub enum Pred {
    Type(String),
    Tag(String),
    Label(String),
    LabelId(u16),
    Team(i16),
    Owner(String),
    Slot(u8),
    Alive(u32),
    Id(u32),
    Near([f32; 3], f32),
    Box([f32; 3], [f32; 3]),
    Megalo(bool),
    Zone(bool),
    Shape(u8),
    Parent(u32),
}

#[derive(Clone, Debug, Default)]
pub struct Filter {
    pub preds: Vec<Pred>,
    /// Time at which `near`/`box` are evaluated (`at=`); default = the alive time, else first position.
    pub at: Option<u32>,
}

fn parse_id(s: &str) -> Result<u32, String> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u32::from_str_radix(h, 16),
        None => s.parse::<u32>(),
    }
    .map_err(|_| format!("bad object id '{s}' (0x-prefixed hex or decimal)"))
}

fn floats(s: &str, n: usize) -> Result<Vec<f32>, String> {
    let v: Result<Vec<f32>, _> = s.split(',').map(|x| x.trim().parse::<f32>()).collect();
    let v = v.map_err(|_| format!("bad number list '{s}'"))?;
    if v.len() != n {
        return Err(format!("expected {n} comma-separated numbers, got '{s}'"));
    }
    Ok(v)
}

impl Filter {
    /// `key=value` tokens. Keys: `type` `tag` `label` (text, glob, or `#<index>`) `team` `owner`
    /// (name substring) `slot` `alive` (time) `id` `parent` `near` (`x,y,z,r`) `box`
    /// (`x0,y0,z0,x1,y1,z1`) `at` (time) `megalo` `zone` `shape`. Unknown tokens are an error.
    pub fn parse(args: &[String]) -> Result<Filter, String> {
        let mut f = Filter::default();
        for a in args {
            let Some((k, v)) = a.split_once('=') else {
                return Err(format!("bad filter '{a}' (expected key=value; keys: type tag label team owner slot alive id parent near box at megalo zone shape)"));
            };
            let on = |v: &str| !matches!(v, "0" | "false" | "no" | "off");
            match k.to_ascii_lowercase().as_str() {
                "type" => f.preds.push(Pred::Type(v.to_string())),
                "tag" => f.preds.push(Pred::Tag(v.to_string())),
                "label" => match v.strip_prefix('#') {
                    Some(n) => f.preds.push(Pred::LabelId(n.parse().map_err(|_| format!("bad label index '{v}'"))?)),
                    None => f.preds.push(Pred::Label(v.to_string())),
                },
                "team" => f.preds.push(Pred::Team(match v.to_ascii_lowercase().as_str() {
                    "none" | "-1" => -1,
                    "red" => 0,
                    "blue" => 1,
                    "green" => 2,
                    "orange" => 3,
                    "purple" => 4,
                    "gold" | "yellow" => 5,
                    "brown" => 6,
                    "pink" => 7,
                    x => x.parse::<i16>().map_err(|_| format!("bad team '{v}'"))?,
                })),
                "owner" | "player" => f.preds.push(Pred::Owner(v.to_string())),
                "slot" => f.preds.push(Pred::Slot(v.parse().map_err(|_| format!("bad slot '{v}'"))?)),
                "alive" => f.preds.push(Pred::Alive(crate::film_data::parse_time(v)?)),
                "at" => f.at = Some(crate::film_data::parse_time(v)?),
                "id" => f.preds.push(Pred::Id(parse_id(v)?)),
                "parent" => f.preds.push(Pred::Parent(parse_id(v)?)),
                "near" => {
                    let n = floats(v, 4)?;
                    f.preds.push(Pred::Near([n[0], n[1], n[2]], n[3]));
                }
                "box" => {
                    let n = floats(v, 6)?;
                    f.preds.push(Pred::Box([n[0].min(n[3]), n[1].min(n[4]), n[2].min(n[5])], [n[0].max(n[3]), n[1].max(n[4]), n[2].max(n[5])]));
                }
                "megalo" => f.preds.push(Pred::Megalo(on(v))),
                "zone" => f.preds.push(Pred::Zone(on(v))),
                "shape" => f.preds.push(Pred::Shape(match v.to_ascii_lowercase().as_str() {
                    "none" => 0,
                    "sphere" => 1,
                    "cylinder" => 2,
                    "box" => 3,
                    _ => return Err(format!("bad shape '{v}' (none|sphere|cylinder|box)")),
                })),
                x => return Err(format!("unknown filter key '{x}'")),
            }
        }
        Ok(f)
    }

    pub fn matches(&self, r: &ObjRow, tr: Option<&EntityTrack>) -> bool {
        let alive_t = self.preds.iter().find_map(|p| if let Pred::Alive(t) = p { Some(*t) } else { None });
        let at = self.at.or(alive_t);
        self.preds.iter().all(|p| match p {
            Pred::Type(s) => glob(s, r.type_name()) || r.tag.as_deref().map_or(false, |t| glob(s, t)),
            Pred::Tag(s) => r.tag.as_deref().map_or(false, |t| glob(s, t)),
            Pred::Label(s) => r.label().map_or(false, |l| if s.contains('*') { glob(s, l) } else { l.eq_ignore_ascii_case(s) }),
            Pred::LabelId(n) => r.placed.as_ref().map_or(false, |p| p.label_idx == *n),
            Pred::Team(t) => {
                let team = r.mp.map(|m| m.team as i16).or(r.placed.as_ref().map(|p| if p.team == 0xFF { -1 } else { p.team as i16 }));
                team == Some(*t)
            }
            Pred::Owner(s) => r.owner_name.as_deref().map_or(false, |n| n.to_ascii_lowercase().contains(&s.to_ascii_lowercase())),
            Pred::Slot(s) => r.owner_slot == Some(*s),
            Pred::Alive(t) => r.alive_at(*t),
            Pred::Id(i) => r.net_id == *i,
            Pred::Parent(i) => r.parent == Some(*i),
            Pred::Megalo(b) => r.megalo_ref == *b,
            Pred::Zone(b) => r.boundary().is_some() == *b,
            Pred::Shape(s) => r.boundary().map_or(0, |b| b.0) == *s,
            Pred::Near(c, rad) => self.pos(r, tr, at).map_or(false, |p| {
                ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() <= *rad
            }),
            Pred::Box(lo, hi) => self.pos(r, tr, at).map_or(false, |p| (0..3).all(|i| p[i] >= lo[i] && p[i] <= hi[i])),
        })
    }

    fn pos(&self, r: &ObjRow, tr: Option<&EntityTrack>, at: Option<u32>) -> Option<[f32; 3]> {
        match (tr, at) {
            (Some(tr), Some(t)) => pos_at(tr, t),
            _ => r.first_pos,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// output
// ---------------------------------------------------------------------------------------------

fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or(String::new(), |x| x.to_string())
}

fn csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub const OBJECT_COLUMNS: &str = "id,entity_type,type,tag,first_tick,last_tick,destroyed_tick,x,y,z,fwd_x,fwd_y,fwd_z,up_x,up_y,up_z,parent,owner_slot,owner_name,megalo_ref,placement_slot,quota_index,item,label_idx,label,placed_team,spawn_seq,respawn_s,cached_type,placement_flags,join_dist,placed_shape,placed_w_or_r,placed_length,placed_top,placed_bottom,mp_flags,mp_team,mp_shape,mp_w_or_r,mp_length,mp_top,mp_bottom,zone_source,id_hex";

pub fn row_csv(r: &ObjRow, tr: Option<&EntityTrack>, at: Option<u32>) -> String {
    let pos = match (tr, at) {
        (Some(tr), Some(t)) => pos_at(tr, t),
        _ => r.first_pos,
    };
    let (f, u) = match (tr, at) {
        (Some(tr), Some(t)) => basis_at(tr, t),
        (Some(tr), None) => basis_at(tr, r.first_tick),
        _ => ([f32::NAN; 3], [f32::NAN; 3]),
    };
    let p = r.placed.as_ref();
    let fl = |v: f32| if v.is_nan() { String::new() } else { format!("{v:.4}") };
    let mut c: Vec<String> = vec![
        r.net_id.to_string(),
        r.entity_type.to_string(),
        r.type_name().to_string(),
        csv(r.tag.as_deref().unwrap_or("")),
        r.first_tick.to_string(),
        r.last_tick.to_string(),
        opt(r.destroyed_tick),
    ];
    match pos {
        Some(p) => c.extend(p.iter().map(|v| format!("{v:.4}"))),
        None => c.extend(["".into(), "".into(), "".into()]),
    }
    c.extend(f.iter().map(|v| fl(*v)));
    c.extend(u.iter().map(|v| fl(*v)));
    c.push(opt(r.parent));
    c.push(opt(r.owner_slot));
    c.push(csv(r.owner_name.as_deref().unwrap_or("")));
    c.push(r.megalo_ref.to_string());
    c.push(opt(p.map(|p| p.slot)));
    c.push(opt(p.map(|p| p.quota_index)));
    c.push(opt(p.map(|p| p.item)));
    c.push(opt(p.filter(|p| p.label_idx != 0xFFFF).map(|p| p.label_idx)));
    c.push(csv(r.label().unwrap_or("")));
    c.push(opt(p.filter(|p| p.team != 0xFF).map(|p| p.team)));
    c.push(opt(p.map(|p| p.spawn_seq)));
    c.push(opt(p.map(|p| p.respawn)));
    c.push(opt(p.map(|p| p.cached_type)));
    c.push(opt(p.map(|p| p.placement)));
    c.push(opt(p.map(|p| format!("{:.4}", p.join_dist))));
    c.push(p.map_or(String::new(), |p| shape_name(p.shape).to_string()));
    for i in 0..4 {
        c.push(p.filter(|p| p.shape != 0).map_or(String::new(), |p| format!("{:.3}", p.dims[i])));
    }
    c.push(opt(r.mp.map(|m| m.flags)));
    c.push(opt(r.mp.map(|m| m.team)));
    c.push(r.mp.map_or(String::new(), |m| shape_name(m.shape).to_string()));
    for i in 0..4 {
        c.push(r.mp.filter(|m| m.shape != 0).map_or(String::new(), |m| format!("{:.3}", m.dims[i])));
    }
    c.push(r.boundary().map_or(String::new(), |b| b.2.to_string()));
    c.push(format!("0x{:08X}", r.net_id));
    c.join(",")
}

pub fn row_json(r: &ObjRow, tr: Option<&EntityTrack>, at: Option<u32>) -> serde_json::Value {
    use serde_json::json;
    let pos = match (tr, at) {
        (Some(tr), Some(t)) => pos_at(tr, t),
        _ => r.first_pos,
    };
    json!({
        "id": r.net_id,
        "id_hex": format!("0x{:08X}", r.net_id),
        "entity_type": r.entity_type,
        "type": r.type_name(),
        "def_wire": r.def_wire,
        "tag": r.tag,
        "first_tick": r.first_tick,
        "last_tick": r.last_tick,
        "destroyed_tick": r.destroyed_tick,
        "position": pos,
        "parent": r.parent,
        "owner_slot": r.owner_slot,
        "owner_name": r.owner_name,
        "megalo_ref": r.megalo_ref,
        "label": r.label(),
        "placement": r.placed.as_ref().map(|p| json!({
            "slot": p.slot, "quota_index": p.quota_index, "item": p.item,
            "label_idx": (p.label_idx != 0xFFFF).then_some(p.label_idx),
            "team": (p.team != 0xFF).then_some(p.team), "spawn_seq": p.spawn_seq, "respawn_s": p.respawn,
            "cached_type": p.cached_type, "flags": p.placement, "color": p.color,
            "shape": shape_name(p.shape), "dims": p.dims, "join_dist": p.join_dist,
        })),
        "mp": r.mp.map(|m| json!({"flags": m.flags, "team": m.team, "shape": shape_name(m.shape), "dims": m.dims})),
        "zone_source": r.boundary().map(|b| b.2),
    })
}


/// Is `row` a live occupant candidate at `t`? A destroyed object is gone AT its destroy tick; one
/// never destroyed lives to the end of the recording.
pub fn occ_alive(row: &ObjRow, t: u32) -> bool {
    t >= row.first_tick && row.destroyed_tick.map_or(true, |d| t < d)
}

/// Which occupants (net ids) are inside `vol` at `tick`, by HOLD position. Only entities that have
/// ever stated a position can be inside anything.
pub fn inside_ids(vol: &Volume, occ: &[(&ObjRow, &EntityTrack)], tick: u32) -> Vec<u32> {
    occ.iter()
        .filter(|(r, tr)| occ_alive(r, tick) && pos_at(tr, tick).map_or(false, |p| vol.contains(p)))
        .map(|(r, _)| r.net_id)
        .collect()
}

/// One zone-occupancy transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub tick: u32,
    pub enter: bool,
    pub zone: u32,
    pub who: u32,
}

/// Replay `[from, to]` one tick at a time for one zone, emitting ENTER/EXIT. `vol_at` gives the
/// zone's volume at a tick (`None` = the zone does not exist then, which empties it).
pub fn transitions(
    zone: u32,
    vol_at: &dyn Fn(u32) -> Option<Volume>,
    occ: &[(&ObjRow, &EntityTrack)],
    from: u32,
    to: u32,
) -> Vec<Transition> {
    let mut out = Vec::new();
    let mut inside: Vec<u32> = Vec::new();
    for t in from..=to {
        let now: Vec<u32> = match vol_at(t) {
            Some(v) => inside_ids(&v, occ, t),
            None => Vec::new(),
        };
        for w in &now {
            if !inside.contains(w) {
                out.push(Transition { tick: t, enter: true, zone, who: *w });
            }
        }
        for w in &inside {
            if !now.contains(w) {
                out.push(Transition { tick: t, enter: false, zone, who: *w });
            }
        }
        inside = now;
    }
    out
}

#[cfg(test)]
mod hill_marker_gates {
    use super::*;

    /// The race clip the "drifting hill markers" report came from (`HT Ignition v2.1`, 16
    /// players, 24 Mongooses). Not in the repo: `$HMS_FILM`, else `~/Downloads/Reach(1).mov`.
    fn race_clip() -> Option<crate::film::Film> {
        let p = std::env::var_os("HMS_FILM").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Downloads").join("Reach(1).mov")))?;
        let f = crate::film::open(&p).ok()?;
        (f.header.game_variant.title.contains("Ignition")).then_some(f)
    }

    /// #film-hill-markers  GATE. The race gametype glues a hill-marker zone to every player (and the
    /// player rides a Mongoose). The recording states the parent and then no position of the
    /// marker's own, so interpolating its own samples slid it across the map. The independent
    /// check: at each RELEASE of a marker (attachment cleared) the marker's first own position
    /// sample is where it really was, and that sample must sit at the parent chain's position
    /// (marker -> body -> Mongoose) to within the authored attach offset (~0.8 wu), not 15 wu away.
    #[test]
    fn hill_markers_follow_the_player_they_are_attached_to() {
        let Some(f) = race_clip() else {
            eprintln!("SKIPPED: race clip (HT Ignition v2.1) not on this machine");
            return;
        };
        let idx = crate::film_decode::build_index_film(&f);
        let by: HashMap<u32, &EntityTrack> = idx.tracks.iter().map(|x| (x.net_id, x)).collect();
        let lookup = |n: u32| by.get(&n).copied();
        let d = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        // hill markers = generic objects with a replicated boundary that ride something
        let markers: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| t.entity_type == 13 && t.rides() && idx.mp_props.get(&t.net_id).map_or(false, |v| v.iter().any(|(_, m)| m.shape != 0))).collect();
        assert!(markers.len() == 37, "liveness: the clip has hill markers that ride players ({})", markers.len());
        let (mut fixed, mut old, mut releases, mut still_static) = (Vec::new(), Vec::new(), 0usize, 0usize);
        for tr in &markers {
            if !tr.moves() { still_static += 1; }
            let mut parent: Option<u32> = None;
            let mut start: Option<u32> = None;
            for (k, a) in &tr.attachments {
                match a {
                    Some(a) => {
                        if parent != Some(a.parent) || start.is_none() { start = Some(*k); }
                        parent = Some(a.parent)
                    }
                    None => {
                        releases += 1;
                        let Some(par) = parent.take().and_then(|p| by.get(&p)) else { continue };
                        let Some((st, sp)) = tr.positions.iter().find(|(pt, _)| pt >= k) else { continue };
                        let want = crate::film_objective::follow_position(&lookup, par, *st as f32, 1).unwrap();
                        fixed.push(d(*sp, want.to_array()));
                        let start = start.take();
                        // what the lane drew in the MIDDLE of the attachment: its own interpolated samples
                        if let Some(a0) = start {
                            if *k > a0 + 120 {
                                let mid = (a0 + *k) / 2;
                                let drawn_old = tr.sample(mid).unwrap();
                                let chain = crate::film_objective::follow_position(&lookup, par, mid as f32, 1).unwrap();
                                old.push(d(drawn_old, chain.to_array()));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(releases, 1393);
        assert_eq!(fixed.len(), 1393);
        fixed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        old.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("HILL release error (new): median {:.2} p90 {:.2} max {:.2} over {}; old mid-attachment error: median {:.2} p90 {:.2} p99 {:.2} max {:.2} over {} stretches", fixed[fixed.len()/2], fixed[fixed.len()*9/10], fixed.last().unwrap(), fixed.len(), old[old.len()/2], old[old.len()*9/10], old[old.len()*99/100], old.last().unwrap(), old.len());
        assert!(fixed[fixed.len() / 2] > 0.7 && fixed[fixed.len() / 2] < 0.85, "median release error {}", fixed[fixed.len() / 2]);
        assert!(*fixed.last().unwrap() < 1.0, "worst release error {}", fixed.last().unwrap());
        assert!(old[old.len() / 2] > 5.0 && old[old.len() * 9 / 10] > 15.0, "the old interpolation was far off ({})", old[old.len() / 2]);
        // markers that sit glued to a player without ever stating a position of their own: `moves()`
        // called them static, which sent them to the "already in the scene" bucket
        assert!(still_static >= 10, "{still_static} glued markers have no path of their own");
        // the attach offsets decode to the authored scale (<= ~1.1 wu), not 33x that
        let worst = markers.iter().flat_map(|t| t.attachments.iter()).filter_map(|(_, a)| *a).map(|a| a.offset.iter().fold(0.0f32, |m, v| m.max(v.abs()))).fold(0.0f32, f32::max);
        assert!(worst < 1.2, "attach offset scale ({worst})");
        let clean = markers.iter().flat_map(|t| t.attachments.iter()).filter_map(|(_, a)| *a).filter(|a| a.node == 0).any(|a| (a.offset[0] - 1.0).abs() < 0.01 && a.offset[1].abs() < 0.01);
        assert!(clean, "a node-0 attachment reads exactly (1.0, 0, 0)");
    }

    /// #film-hill-markers  GATE (the draw lane's inputs). A marker glued to a player never states a
    /// position after attaching, so `moves()` called it static and the lane left it to the Forge
    /// scene (frozen at the pit); and the lane placed the rest from their own interpolated samples.
    /// Through the REAL `ObjectTable` (definitions resolved from forge_halo): every ridden marker is
    /// drawn from the stream, and at any tick it is drawn within 1 wu of the thing it rides.
    #[test]
    #[ignore = "opens the native library: map-backed gate, run one module per process with --ignored --test-threads=1"]
    fn the_object_lane_draws_hill_markers_on_their_riders() {
        let Some(f) = race_clip() else {
            eprintln!("SKIPPED: race clip (HT Ignition v2.1) not on this machine");
            return;
        };
        let Some((native, cache)) = crate::film_action::objective_inventory::open_reach("forge_halo") else {
            eprintln!("SKIPPED: forge_halo not available");
            return;
        };
        let map = crate::mapcat::enumerate().into_iter().find(|c| c.game == crate::mapcat::Game::Reach && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))).unwrap();
        let names = crate::film_stats::MapNames::build_with_map(&native, cache, Some(&map.path));
        let idx = crate::film_decode::build_index_film(&f);
        use crate::film_stats::NameSource;
        let table = crate::film_objective::ObjectTable::build(&idx, |w| names.definition(w), |_| None, &[]);
        let by: HashMap<u32, &EntityTrack> = idx.tracks.iter().map(|x| (x.net_id, x)).collect();
        let lookup = |n: u32| by.get(&n).copied();
        let beacons: Vec<&EntityTrack> = idx.tracks.iter().filter(|t| table.objects.get(&t.net_id).map_or(false, |o| o.name == "mp_hill_beacon") && t.rides()).collect();
        assert_eq!(beacons.len(), 37);
        assert!(beacons.iter().all(|t| table.objects[&t.net_id].draw == crate::film_objective::Draw::Debris), "every ridden marker is drawn from the stream");
        let (mut checked, mut worst) = (0usize, 0.0f32);
        for tick in (2_000..idx.ticks.min(60_000)).step_by(997) {
            for (o, pos, _) in table.alive_at(&idx, tick) {
                let Some(tr) = by.get(&o.net_id) else { continue };
                if o.name != "mp_hill_beacon" || !tr.rides() { continue; }
                let Some(a) = tr.attached_at(tick) else { continue };
                let Some(par) = by.get(&a.parent) else { continue };
                let Some(want) = crate::film_objective::follow_position(&lookup, par, tick as f32, 1) else { continue };
                checked += 1;
                worst = worst.max(((pos[0] - want.x).powi(2) + (pos[1] - want.y).powi(2) + (pos[2] - want.z).powi(2)).sqrt());
            }
        }
        assert!(checked > 100, "liveness: {checked} marker placements checked");
        assert!(worst < 1.0, "worst lane-vs-rider distance {worst}");
    }


    #[test]
    #[ignore = "probe"]
    fn scan_invisible_cube() {
        let (native, cache) = crate::film_action::objective_inventory::open_reach("forge_halo").unwrap();
        let map = crate::mapcat::enumerate().into_iter().find(|c| c.game == crate::mapcat::Game::Reach && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))).unwrap();
        let names = crate::film_stats::MapNames::build_with_map(&native, cache, Some(&map.path));
        use crate::film_stats::NameSource;
        let mut paths = crate::film::spec_corpus();
        if let Some(h) = std::env::var_os("HOME") {
            if let Ok(rd) = std::fs::read_dir(std::path::Path::new(&h).join("Downloads")) { for e in rd.flatten() { let p = e.path(); if p.extension().map_or(false, |x| x == "mov" || x == "film" || x == "temp") { paths.push(p); } } }
        }
        paths.sort(); paths.dedup();
        for p in paths {
            let Ok(f) = crate::film::open(&p) else { continue };
            if !f.header.scenario_name().to_ascii_lowercase().contains("forge_halo") { continue; }
            let idx = crate::film_decode::build_index_film(&f);
            let n = idx.tracks.iter().filter(|t| t.def_wire.and_then(|w| names.definition(w)).map_or(false, |(_, _, path)| path.contains("invisible_cube"))).count();
            let b = idx.tracks.iter().filter(|t| t.def_wire.and_then(|w| names.definition(w)).map_or(false, |(_, _, path)| path.contains("hill_beacon"))).count();
            eprintln!("SC {} records {} invisible_cube tracks {} beacons {}", p.file_name().unwrap().to_string_lossy(), f.data.records.len(), n, b);
        }
    }
}
