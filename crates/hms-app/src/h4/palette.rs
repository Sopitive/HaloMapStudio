//! Halo 4 Forge palette: the scnr "Map Variant Palettes" block (+0x2C4) with its
//! categories, entries (quota slots), object variants, localized display names, prices / limits,
//! the multiplayer-object defaults of every object, and `instantiate` - a NEW placement expressed
//! as (palette entry, variant, pose) that renders through the existing H4 object path and carries
//! the record the `.mvar` encoder (mvar.rs) needs. Layout + evidence: docs/halo4_forge_palette.md.
//!
//! Engine facts (halo4.dll, IDA on the shipped build; all checked against the cache data):
//!   * quota index = sum of the entry counts of the palettes before it + entry index
//!     (`sub_1800B3550`; its inverse `sub_1800B3614` walks the palettes subtracting counts).
//!   * a scenario object is matched to a palette variant by TAG DATUM (variant +0x10) and by the
//!     hlmt model variant index (`sub_1800B5C90` -> `object_get_variant_index` sub_1805D912C):
//!     the variant's name sid @0x14 names a hlmt variant, 0 = the object's default (obje +0x60).
//!   * the runtime object datum stores the quota as u16 and the variant index as u8 (76-byte
//!     `s_variant_object_datum`, mvar.rs) - the same two indices every shipped .mvar carries.
//!
//! Display names: `ui\strings\forge` (unic.rs) - every palette / entry / variant name sid of
//! every MP map resolves there (tested). The Assembly Halo4MCC scnr plugin names the fields;
//! the values were cross-checked on the caches (hidden palette = the superforge block set,
//! thorage = the MCC additions, `max <= 0` = unlimited).

use std::sync::atomic::{AtomicU32, Ordering};

use glam::Vec3;

use super::cache::{ByteRead, H4Cache};
use super::mvar::{H4PaletteEntry, H4PaletteVariant, H4PlacedObject, H4Shape, H4TypeData, H4Variant};
use super::objects::{
    default_change_colors, default_variant_sid, model_of, mp_object_defaults, object_variant_index, H4Placement, MpObjectDefaults,
};
use super::unic::{LocaleTable, LANG_ENGLISH};

pub const OFF_SCNR_FORGE_PALETTE: usize = 0x2C4;
/// Palette element (0x14): sid @0, u8 flags @4 (bit 0 hidden = superforge only), u8 @5, u8
/// thorage @6, u8 @7, entries block @8 (count @8, ptr @0xC, pad @0x10).
pub const FORGE_PALETTE_ELEM: usize = 0x14;
/// Entry element (0x1C): sid @0, variants block @4, i32 maximum allowed @0x10 (<= 0 = unlimited),
/// i32 price per instance @0x14, u8 thorage @0x18, 3 x u8 @0x19.
pub const FORGE_ENTRY_ELEM: usize = 0x1C;
/// Variant element (0x48): display sid @0, object tag ref @4 (datum @0x10), hlmt variant sid
/// @0x14, then four 12-byte blocks of u32 tag / resource datums: attached resource owners @0x18,
/// top-level resource owners (the object itself) @0x24, attached resources @0x30, orphaned @0x3C.
pub const FORGE_VARIANT_ELEM: usize = 0x48;
pub const PALETTE_FLAG_HIDDEN: u8 = 1;
/// The unic tag carrying the Forge UI strings.
pub const FORGE_STRINGS_TAG: &str = "ui\\strings\\forge";

#[derive(Clone, Debug)]
pub struct ForgeCategory {
    /// Index in the scnr palette block.
    pub index: usize,
    pub sid: u32,
    /// Raw string id name (`ff_weapons_human`).
    pub name: String,
    /// Localized (`Weapons, Human`); the raw name when no string resolves.
    pub display: String,
    pub flags: u8,
    /// Superforge-only palette (flag bit 0): the duplicate `bb_*` block set on every Forge map.
    /// The Forge highlight predicate (halo4.dll sub_1800BD8C8, quota ->
    /// category via sub_1800B3614, `palette[cat].flags & 1`) rejects every object of such a
    /// category: they are FIXED in-game - the monitor can never grab one.
    pub hidden: bool,
    /// MCC "thorage" addition (byte @6).
    pub thorage: bool,
    /// Range of this category's entries in `ForgePalette::entries` (= its quota indices).
    pub first_entry: usize,
    pub entry_count: usize,
}

#[derive(Clone, Debug)]
pub struct ForgeVariant {
    /// Index inside the entry (the variant record's `variant` field).
    pub index: usize,
    pub sid: u32,
    /// Raw display-name sid (`gad_fusion_coil`), empty on single-variant entries.
    pub name: String,
    /// Localized (`Fusion Coil`); the entry's display name when the variant names none.
    pub display: String,
    /// Object tag (scen / bloc / vehi / weap / eqip / mach / dspn ...), None for a null ref.
    pub tag: Option<usize>,
    pub class: Option<[u8; 4]>,
    pub tag_path: String,
    /// hlmt model variant this palette variant spawns (sid @0x14; 0 = the object's default).
    pub variant_sid: u32,
    pub variant_name: String,
    /// The resolved hlmt variant index (`object_get_variant_index`), None = model has none.
    pub model_variant: Option<usize>,
    /// `mode` render model (obje -> hlmt -> mode), None = marker without a drawable model.
    pub mode: Option<usize>,
    /// obje +0x154 multiplayer object defaults (type, shape, spawn time, phased physics ...).
    pub mp: Option<MpObjectDefaults>,
    /// Sizes of the four resource-prediction blocks (attached owners, self, attached, orphaned).
    pub prediction: [usize; 4],
}

impl ForgeVariant {
    pub fn class_str(&self) -> String { self.class.map(|c| String::from_utf8_lossy(&c).into_owned()).unwrap_or_default() }
    /// Multiplayer object type (0 ordinary when the object authors no MP block).
    pub fn object_type(&self) -> u8 { self.mp.map(|m| m.object_type).unwrap_or(0) }
}

#[derive(Clone, Debug)]
pub struct ForgeEntry {
    /// Flat quota index = the variant record's `quota` field.
    pub quota: u8,
    pub category: usize,
    pub sid: u32,
    pub name: String,
    pub display: String,
    /// `Maximum Allowed` (<= 0 = unlimited up to the code maximum).
    pub max_allowed: i32,
    pub price: i32,
    pub thorage: bool,
    pub variants: Vec<ForgeVariant>,
}

impl ForgeEntry {
    /// The variant a record's `variant` index (None = 0) selects.
    pub fn variant(&self, index: Option<u8>) -> Option<&ForgeVariant> { self.variants.get(index.unwrap_or(0) as usize) }
}

#[derive(Clone, Debug, Default)]
pub struct ForgePalette {
    pub map_name: String,
    pub categories: Vec<ForgeCategory>,
    /// Flattened in scnr order: `entries[q]` is quota index `q`.
    pub entries: Vec<ForgeEntry>,
    /// True when `ui\strings\forge` resolved at least one name (false = raw sids shown).
    pub localized: bool,
}

impl ForgePalette {
    pub fn entry(&self, quota: u8) -> Option<&ForgeEntry> { self.entries.get(quota as usize) }
    /// Resolve a variant record's (quota, variant) pair.
    pub fn resolve(&self, quota: Option<u8>, variant: Option<u8>) -> Option<(&ForgeEntry, &ForgeVariant)> {
        let e = self.entry(quota?)?;
        Some((e, e.variant(variant)?))
    }
    /// First entry / variant whose raw name or display name equals `name` (case-insensitive).
    pub fn find(&self, name: &str) -> Option<(u8, u8)> {
        for e in &self.entries {
            for v in &e.variants {
                if v.name.eq_ignore_ascii_case(name) || v.display.eq_ignore_ascii_case(name) { return Some((e.quota, v.index as u8)); }
            }
            if e.name.eq_ignore_ascii_case(name) || e.display.eq_ignore_ascii_case(name) { return Some((e.quota, 0)); }
        }
        None
    }
    /// The mvar.rs view of the same data (quota index / prices / variant tags), for the codec.
    pub fn to_mvar_palette(&self) -> Vec<H4PaletteEntry> {
        self.entries.iter().map(|e| H4PaletteEntry {
            quota: e.quota,
            palette: self.categories.get(e.category).map(|c| c.name.clone()).unwrap_or_default(),
            name: e.name.clone(),
            max: e.max_allowed,
            price: e.price,
            variants: e.variants.iter().map(|v| H4PaletteVariant { name: v.name.clone(), variant_name: v.variant_name.clone(), tag: v.tag }).collect(),
        }).collect()
    }
    /// Entries visible in the normal Forge menu (not the hidden superforge palettes).
    pub fn visible_entries(&self) -> impl Iterator<Item = &ForgeEntry> {
        self.entries.iter().filter(move |e| !self.categories.get(e.category).map_or(false, |c| c.hidden))
    }
    /// Human-readable listing (HMS_H4_PALETTE_DUMP / --dump-h4-palette).
    pub fn describe(&self) -> String {
        let mut s = format!("h4 forge palette '{}': {} categories, {} entries (quota 0..{}), {} object variants, names {}\n",
            self.map_name, self.categories.len(), self.entries.len(), self.entries.len().saturating_sub(1),
            self.entries.iter().map(|e| e.variants.len()).sum::<usize>(), if self.localized { "localized (ui\\strings\\forge)" } else { "RAW (no string table)" });
        for c in &self.categories {
            s.push_str(&format!("[{:2}] {:<26} '{}'{}{} entries {}..{} ({})\n", c.index, c.name, c.display,
                if c.hidden { " HIDDEN(superforge)" } else { "" }, if c.thorage { " thorage" } else { "" },
                c.first_entry, c.first_entry + c.entry_count.saturating_sub(1), c.entry_count));
            for e in &self.entries[c.first_entry..c.first_entry + c.entry_count] {
                s.push_str(&format!("  {:3} {:<28} '{}' max {:3} price {:4}{}\n", e.quota, e.name, e.display, e.max_allowed, e.price, if e.thorage { " thorage" } else { "" }));
                for v in &e.variants {
                    s.push_str(&format!("      .{} {:<30} '{}' {} {}{} type {} shape {} spawn {}s{} hlmt-variant {}{}\n", v.index, v.name, v.display, v.class_str(),
                        v.tag_path.rsplit('\\').next().unwrap_or(&v.tag_path),
                        if v.mode.is_some() { "" } else { " [no render model]" },
                        v.object_type(), v.mp.map(|m| m.shape).unwrap_or(0), v.mp.map(|m| m.spawn_time).unwrap_or(0),
                        if v.mp.map_or(false, |m| m.phased_in_forge()) { " phased" } else { "" },
                        v.model_variant.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                        if v.variant_sid != 0 { format!(" ({})", v.variant_name) } else { String::new() }));
                }
            }
        }
        s
    }
}

/// Read the map's Forge palette (categories, entries, variants, localized names, MP defaults).
pub fn palette(c: &H4Cache) -> ForgePalette {
    let d = c.data();
    let mut out = ForgePalette { map_name: c.map_name.clone(), ..Default::default() };
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let Some((np, po)) = c.block(sm + OFF_SCNR_FORGE_PALETTE) else { return out };
    // English strings of `ui\strings\forge`, then anywhere in the language table
    let loc = LocaleTable::load(c, LANG_ENGLISH);
    let forge = loc.as_ref().and_then(|l| l.window_of(c, FORGE_STRINGS_TAG));
    let mut display = |sid: u32, fallback: &str| -> String {
        match loc.as_ref().and_then(|l| l.get_in(forge, sid)) {
            Some(s) if sid != 0 => { out.localized = true; s.to_string() }
            _ => fallback.to_string(),
        }
    };
    let mut categories = Vec::new();
    let mut entries: Vec<ForgeEntry> = Vec::new();
    for i in 0..np.min(64) {
        let pe = po + i * FORGE_PALETTE_ELEM;
        let sid = d.u32_at(pe);
        let name = c.sid(sid);
        let flags = d.u8_at(pe + 4);
        let mut cat = ForgeCategory {
            index: i, sid, display: display(sid, &name), name, flags, hidden: flags & PALETTE_FLAG_HIDDEN != 0,
            thorage: d.u8_at(pe + 6) != 0, first_entry: entries.len(), entry_count: 0,
        };
        if let Some((ne, eo)) = c.block(pe + 8) {
            for k in 0..ne.min(256) {
                if entries.len() >= 256 { break; }
                let e = eo + k * FORGE_ENTRY_ELEM;
                let esid = d.u32_at(e);
                let ename = c.sid(esid);
                let edisplay = display(esid, &ename);
                let mut variants = Vec::new();
                if let Some((nv, vo)) = c.block(e + 4) {
                    for v in 0..nv.min(32) {
                        let va = vo + v * FORGE_VARIANT_ELEM;
                        let vsid = d.u32_at(va);
                        let vname = c.sid(vsid);
                        let tag = c.tag_ref(va + 4).map(|(_, t)| t);
                        let variant_sid = match d.u32_at(va + 0x14) { 0xFFFF_FFFF => 0, x => x };
                        let prediction = [0x18usize, 0x24, 0x30, 0x3C].map(|o| c.block(va + o).map(|(n, _)| n).unwrap_or(0));
                        variants.push(ForgeVariant {
                            index: v,
                            sid: vsid,
                            display: if vsid != 0 { display(vsid, &vname) } else { edisplay.clone() },
                            name: vname,
                            class: tag.and_then(|t| c.tag_class(t)),
                            tag_path: tag.map(|t| c.tag_name(t).to_string()).unwrap_or_default(),
                            variant_name: c.sid(variant_sid),
                            model_variant: tag.and_then(|t| object_variant_index(c, t, variant_sid)),
                            mode: tag.and_then(|t| model_of(c, t)),
                            mp: tag.and_then(|t| mp_object_defaults(c, t)),
                            prediction,
                            tag,
                            variant_sid,
                        });
                    }
                }
                entries.push(ForgeEntry {
                    quota: entries.len() as u8, category: i, sid: esid, name: ename, display: edisplay,
                    max_allowed: d.i32_at(e + 0x10), price: d.i32_at(e + 0x14), thorage: d.u8_at(e + 0x18) != 0, variants,
                });
                cat.entry_count += 1;
            }
        }
        categories.push(cat);
    }
    out.categories = categories;
    out.entries = entries;
    out
}

// ---------------------------------------------------------------------------------------------
// placements
// ---------------------------------------------------------------------------------------------

/// Where and how a new object stands: world position + unit forward / up (the variant record's
/// own orientation model; `H4Placement::basis` consumes the same pair).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H4Pose {
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
}

impl H4Pose {
    /// Upright, facing +X.
    pub fn at(pos: [f32; 3]) -> Self { H4Pose { pos, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] } }
    /// Upright, yawed about +Z (radians, 0 = +X).
    pub fn yawed(pos: [f32; 3], yaw: f32) -> Self { H4Pose { pos, fwd: [yaw.cos(), yaw.sin(), 0.0], up: [0.0, 0.0, 1.0] } }
}

impl Default for H4Pose {
    fn default() -> Self { H4Pose::at([0.0; 3]) }
}

/// A Forge object HMS placed (or re-instantiated from a record): a stable identity for the
/// editor, the resolved tag / model / hlmt variant / default change colours for the renderer, and
/// the `.mvar` record for the encoder.
#[derive(Clone, Debug)]
pub struct H4ForgeInstance {
    /// Process-unique datum-like id (`next_id`), never reused; NOT the .mvar slot.
    pub id: u32,
    pub quota: u8,
    pub variant: u8,
    pub tag: usize,
    pub class: [u8; 4],
    /// `mode` render model (None = marker without one; still saved, not drawn).
    pub mode: Option<usize>,
    /// hlmt model variant the renderer shows (objects.rs `variant_meshes`).
    pub model_variant: Option<usize>,
    pub variant_sid: u32,
    /// Authored default change colours [primary, secondary, tertiary, quaternary] (linear RGB,
    /// white when unauthored). A Forge placement's PRIMARY is overwritten by the team / colour
    /// rule at render time (`h4::tint::primary_change_color`); the other three ride these. #h4-veh
    pub change_colors: [[f32; 3]; 4],
    /// `entry.display` / `variant.display` for the UI.
    pub display: String,
    /// The variant record (mvar.rs): quota, variant, pose, type, defaults - what `save_objects`
    /// writes. `slot` is `NEW_SLOT` until the encoder assigns one.
    pub record: H4PlacedObject,
}

impl H4ForgeInstance {
    /// The render-path placement (same struct the scenario / variant objects use).
    pub fn placement(&self) -> H4Placement {
        H4Placement {
            class: self.class,
            palette_tag: self.tag,
            name: self.display.clone(),
            pos: self.record.pos,
            rot: [0.0; 3],
            scale: 1.0,
            flags: self.record.placement as u32,
            basis: Some((Vec3::from(self.record.fwd), Vec3::from(self.record.up))),
            mp: Some((self.record.team, self.record.color)),
            model_variant: self.model_variant,
        }
    }
    pub fn pose(&self) -> H4Pose { H4Pose { pos: self.record.pos, fwd: self.record.fwd, up: self.record.up } }
    /// Move / turn the object (re-quantised through the engine's orientation codec).
    pub fn set_pose(&mut self, pose: H4Pose) {
        self.record.pos = pose.pos;
        self.record.set_orientation(pose.fwd, pose.up);
    }
}

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// A fresh process-unique instance id (starts at 1).
pub fn next_id() -> u32 { NEXT_ID.fetch_add(1, Ordering::Relaxed) }

/// Instantiate palette entry `quota` / variant `variant` at `pose` with the defaults a Halo 4
/// Forge placement gets (the modal values of the 146 647 shipped objects, see
/// docs/halo4_forge_palette.md section 5): slot flags 1, team neutral (8), scale q 7, placement
/// 0x0C (symmetric + asymmetric, normal physics) or 0xCC (phased) when the object's MP block
/// sets "phased physics in Forge", the MP block's default spawn time / boundary shape + size
/// (1/256 wu units), spare clips 2 on weapons, 255 on dominion pads. None when the entry /
/// variant does not exist or references a null tag.
pub fn instantiate(c: &H4Cache, pal: &ForgePalette, quota: u8, variant: u8, pose: H4Pose) -> Option<H4ForgeInstance> {
    let (e, v) = pal.resolve(Some(quota), Some(variant))?;
    let tag = v.tag?;
    let object_type = v.object_type();
    // the variant index is written explicitly even when it is 0 (the game's own
    // records always carry one; a NONE index makes the engine spawn the NEXT palette entry)
    let mut record = H4PlacedObject::new(quota, Some(variant), object_type, pose.pos, pose.fwd, pose.up);
    if let Some(mp) = v.mp {
        if mp.phased_in_forge() { record.placement = 0xCC; }
        record.spawn_time = mp.spawn_time.clamp(0, 255) as u8;
        record.shape = match mp.shape { 1 => H4Shape::Sphere, 2 => H4Shape::Cylinder, 3 => H4Shape::Box, _ => H4Shape::None };
        let q = |wu: f32| (wu.max(0.0) * 256.0).round().min(65535.0) as u16;
        record.shape_values = match record.shape {
            H4Shape::None => [0; 4],
            H4Shape::Sphere => [q(mp.boundary[0]), 0, 0, 0],
            H4Shape::Cylinder => [q(mp.boundary[0]), q(mp.boundary[2]), q(mp.boundary[3]), 0],
            H4Shape::Box => [q(mp.boundary[0]), q(mp.boundary[1]), q(mp.boundary[2]), q(mp.boundary[3])],
        };
    }
    record.type_data = match (object_type, record.type_data) {
        (1, H4TypeData::Byte(_)) => H4TypeData::Byte(2),
        (12, H4TypeData::Byte(_)) => H4TypeData::Byte(255),
        (_, t) => t,
    };
    let vsid = if v.variant_sid != 0 { v.variant_sid } else { default_variant_sid(c, tag) };
    Some(H4ForgeInstance {
        id: next_id(),
        quota,
        variant,
        tag,
        class: v.class.unwrap_or(*b"scen"),
        mode: v.mode,
        model_variant: v.model_variant,
        variant_sid: vsid,
        change_colors: default_change_colors(c, tag, vsid),
        display: if v.sid != 0 { format!("{} / {}", e.display, v.display) } else { e.display.clone() },
        record,
    })
}

/// Re-instantiate a decoded variant record (keeps its slot / properties; None when its quota or
/// variant is outside the palette or the tag is null).
pub fn from_record(c: &H4Cache, pal: &ForgePalette, record: &H4PlacedObject) -> Option<H4ForgeInstance> {
    let (e, v) = pal.resolve(record.quota, record.variant)?;
    let tag = v.tag?;
    let vsid = if v.variant_sid != 0 { v.variant_sid } else { default_variant_sid(c, tag) };
    Some(H4ForgeInstance {
        id: next_id(),
        quota: e.quota,
        variant: v.index as u8,
        tag,
        class: v.class.unwrap_or(*b"scen"),
        mode: v.mode,
        model_variant: v.model_variant,
        variant_sid: vsid,
        change_colors: default_change_colors(c, tag, vsid),
        display: if v.sid != 0 { format!("{} / {}", e.display, v.display) } else { e.display.clone() },
        record: record.clone(),
    })
}

/// Every object of a variant as instances (records whose quota / variant / tag do not resolve
/// are skipped and counted).
pub fn instances_of(c: &H4Cache, pal: &ForgePalette, v: &H4Variant) -> (Vec<H4ForgeInstance>, usize) {
    let mut out = Vec::with_capacity(v.objects.len());
    let mut unresolved = 0;
    for o in &v.objects {
        match from_record(c, pal, o) { Some(i) => out.push(i), None => unresolved += 1 }
    }
    (out, unresolved)
}

/// `HMS_H4_PLACE` spec: `<quota>[.<variant>][:<team>][@x,y,z]` items separated by `;`. Items
/// without a position are placed in a row along +X from `origin` (3 wu apart), like Reach's
/// HMS_PLACE. `:<team>` overrides the placement's team byte (0..7 a real team, 8 neutral), which is
/// what the engine resolves the object's primary CHANGE COLOUR from - `#h4-veh`, so a colour-change
/// Forge piece can be rendered on either team from the headless path.
pub fn parse_place_spec(spec: &str) -> Vec<(u8, u8, Option<[f32; 3]>, Option<i8>)> {
    let mut out = Vec::new();
    for item in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (sel, pos) = match item.split_once('@') { Some((a, b)) => (a.trim(), Some(b)), None => (item, None) };
        let (sel, team) = match sel.split_once(':') { Some((a, t)) => (a.trim(), t.trim().parse::<i8>().ok()), None => (sel, None) };
        let (q, v) = match sel.split_once('.') { Some((q, v)) => (q.trim(), v.trim()), None => (sel, "0") };
        let (Ok(q), Ok(v)) = (q.parse::<u8>(), v.parse::<u8>()) else { continue };
        let pos = pos.and_then(|p| {
            let f: Vec<f32> = p.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (f.len() == 3).then(|| [f[0], f[1], f[2]])
        });
        out.push((q, v, pos, team));
    }
    out
}

/// Diag: instantiate an `HMS_H4_PLACE` spec (see `parse_place_spec`); `log` receives one line
/// per item, resolved or not.
pub fn place_spec(c: &H4Cache, pal: &ForgePalette, spec: &str, origin: [f32; 3], log: &mut dyn FnMut(String)) -> Vec<H4ForgeInstance> {
    let mut out = Vec::new();
    let mut row = 0usize;
    for (q, v, pos, team) in parse_place_spec(spec) {
        let pos = pos.unwrap_or_else(|| { let p = [origin[0] + row as f32 * 3.0, origin[1], origin[2]]; row += 1; p });
        match instantiate(c, pal, q, v, H4Pose::at(pos)) {
            Some(mut i) => {
                if let Some(t) = team { i.record.team = t; }
                let cc = super::tint::primary_change_color(c, i.tag, i.variant_sid, Some((i.record.team, i.record.color)));
                log(format!("HMS_H4_PLACE: quota {q}.{v} '{}' ({} {}) at ({:.2},{:.2},{:.2}) type {} placement {:#05x} hlmt-variant {:?} team {} authored change colours {:?} resolved primary {:?} id {}{}",
                    i.display, String::from_utf8_lossy(&i.class), c.tag_name(i.tag).rsplit('\\').next().unwrap_or(""), pos[0], pos[1], pos[2],
                    i.record.object_type, i.record.placement, i.model_variant, i.record.team, i.change_colors[0], [cc[0], cc[1], cc[2]], i.id, if i.mode.is_none() { " [no render model - marker]" } else { "" }));
                out.push(i);
            }
            None => log(format!("HMS_H4_PLACE: quota {q}.{v} does not resolve on '{}' ({} entries)", pal.map_name, pal.entries.len())),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::mvar::{all_variant_files, map_id_of_cache, parse_h4_variant};
    use std::collections::{BTreeMap, HashMap};

    fn open(name: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(format!("{name}.map"));
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Ravine: 17 categories / 125 entries, every name localized, the flattening equals mvar.rs's.
    #[test]
    fn ravine_palette() {
        let Some(c) = open("ca_forge_ravine") else { return };
        let p = palette(&c);
        assert_eq!(p.categories.len(), 17);
        assert_eq!(p.entries.len(), 125);
        assert!(p.localized);
        assert_eq!(p.categories[0].display, "Weapons, Human");
        assert_eq!(p.entries[0].display, "Magnum");
        assert_eq!(p.entries[39].variants[0].display, "Fusion Coil");
        assert_eq!(p.entries[91].display, "Building Blocks");
        assert!(p.categories[12].hidden && p.categories[12].name == "ff_structure");
        assert!(p.categories[13..].iter().all(|c| c.thorage));
        // every entry and variant display name resolved (no raw sid left over)
        for e in &p.entries {
            assert_ne!(e.display, e.name, "entry {} not localized", e.name);
            for v in &e.variants { if v.sid != 0 { assert_ne!(v.display, v.name, "variant {} not localized", v.name); } }
        }
        // mvar.rs view agrees (quota order, tags, prices)
        let m = crate::h4::mvar::load_forge_palette(&c);
        assert_eq!(m.len(), p.entries.len());
        for (a, b) in m.iter().zip(&p.entries) {
            assert_eq!((a.quota, a.price, a.max), (b.quota, b.price, b.max_allowed));
            assert_eq!(a.variants.len(), b.variants.len());
            for (x, y) in a.variants.iter().zip(&b.variants) { assert_eq!(x.tag, y.tag); }
        }
        // object types come from the MP block: weapons 1, spawns 16, respawn zones 17
        assert_eq!(p.entries[0].variants[0].object_type(), 1);
        assert_eq!(p.entries[48].variants[0].object_type(), 16);
        assert_eq!(p.entries[51].variants[0].object_type(), 17);
        assert!(p.entries[91].variants[0].mp.unwrap().phased_in_forge());
        assert_eq!(p.find("Fusion Coil"), Some((39, 0)));
        assert_eq!(p.find("bb_2x2"), Some((91, 8)));
        let _ = p.describe();
    }

    /// Every shipped variant (388) on its base map: every object resolves to a palette entry +
    /// variant + non-null tag through `from_record` (0 unresolved), and the object type stored
    /// in the record equals the palette object's MP-block type.
    #[test]
    fn every_shipped_variant_resolves() {
        let files = all_variant_files();
        if files.is_empty() { eprintln!("skip: no halo4 variants"); return }
        let Some(dir) = maps_dir() else { return };
        let mut ids: HashMap<u32, String> = HashMap::new();
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x == "map") && !p.to_string_lossy().contains(" - Copy") {
                if let Some(id) = map_id_of_cache(&p) { ids.insert(id, p.file_stem().unwrap().to_string_lossy().into_owned()); }
            }
        }
        let mut caches: HashMap<String, (H4Cache, ForgePalette)> = HashMap::new();
        let (mut n_files, mut n_objects, mut n_unresolved, mut n_type_mismatch, mut n_models) = (0, 0, 0, 0, 0);
        let mut per_map: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for f in &files {
            let v = parse_h4_variant(f).unwrap();
            let Some(stem) = ids.get(&v.map_id) else { continue };
            if !caches.contains_key(stem) {
                let c = H4Cache::open(&dir.join(format!("{stem}.map"))).unwrap();
                let p = palette(&c);
                caches.insert(stem.clone(), (c, p));
            }
            let (c, p) = &caches[stem];
            let (inst, unresolved) = instances_of(c, p, &v);
            n_files += 1;
            n_objects += v.objects.len();
            n_unresolved += unresolved;
            for (i, o) in inst.iter().zip(v.objects.iter().filter(|o| o.quota.is_some())) {
                let (_, pv) = p.resolve(Some(i.quota), Some(i.variant)).unwrap();
                if pv.object_type() != o.object_type {
                    // 52 shipped records (grenades / two weapons / one spawn point) store type 0
                    // where the object's MP block says 1 / 2 / 16 - stale authoring, never the
                    // other way round; the game still treats them by their palette object
                    assert_eq!(o.object_type, 0, "{}: {} record type {} vs palette {}", f.display(), pv.name, o.object_type, pv.object_type());
                    n_type_mismatch += 1;
                }
                if i.mode.is_some() { n_models += 1; }
                let pl = i.placement();
                assert_eq!(pl.pos, o.pos);
            }
            let e = per_map.entry(stem.clone()).or_default();
            e.0 += 1;
            e.1 += v.objects.len();
        }
        for (m, (files, objs)) in &per_map { eprintln!("  {m}: {files} variants, {objs} objects, palette {} entries", caches[m].1.entries.len()); }
        eprintln!("{n_files} variants, {n_objects} objects, {n_unresolved} unresolved, {n_type_mismatch} type mismatches, {n_models} with a render model");
        assert!(n_files > 0);
        assert_eq!(n_unresolved, 0);
        assert!(n_type_mismatch <= 52, "{n_type_mismatch} stale-type records (52 known)");
    }

    /// Every MP map: the palette parses, every visible entry has a localized display name and at
    /// least one variant with a non-null tag.
    #[test]
    fn every_mp_map_palette() {
        let Some(dir) = maps_dir() else { return };
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if !p.extension().map_or(false, |x| x == "map") || p.to_string_lossy().contains(" - Copy") { continue; }
            let Ok(c) = H4Cache::open(&p) else { continue };
            let pal = palette(&c);
            if pal.entries.is_empty() { continue; }
            let cats: Vec<String> = pal.categories.iter().map(|c| format!("{}={}", c.display, c.entry_count)).collect();
            eprintln!("{}: {} entries; {}", c.map_name, pal.entries.len(), cats.join(", "));
            assert!(pal.localized, "{}: no localized names", c.map_name);
            for e in pal.visible_entries() {
                assert_ne!(e.display, e.name, "{}: entry {} not localized", c.map_name, e.name);
                assert!(e.variants.iter().any(|v| v.tag.is_some()), "{}: entry {} has no object", c.map_name, e.name);
            }
            n += 1;
        }
        eprintln!("{n} maps with a forge palette");
        assert!(n > 0);
    }

    /// Instantiate a few Ravine entries: defaults follow the MP block, the record round-trips
    /// through the placement matrix, ids are unique.
    #[test]
    fn instantiate_defaults() {
        let Some(c) = open("ca_forge_ravine") else { return };
        let p = palette(&c);
        let a = instantiate(&c, &p, 91, 8, H4Pose::at([1.0, 2.0, 3.0])).unwrap(); // Block, 2X2
        let b = instantiate(&c, &p, 0, 0, H4Pose::yawed([4.0, 5.0, 6.0], 1.0)).unwrap(); // Magnum
        let z = instantiate(&c, &p, 51, 0, H4Pose::at([0.0; 3])).unwrap(); // Respawn Zone
        assert_ne!(a.id, b.id);
        assert_eq!(a.record.placement, 0xCC);
        assert_eq!(a.record.quota, Some(91));
        assert_eq!(a.record.variant, Some(8));
        assert_eq!(a.record.object_type, 0);
        assert!(a.mode.is_some());
        assert_eq!(b.record.placement, 0x0C);
        assert_eq!(b.record.object_type, 1);
        assert_eq!(b.record.spawn_time, 30);
        assert_eq!(b.record.type_data, H4TypeData::Byte(2));
        assert_eq!(b.record.variant, Some(0)); // explicit, like every game-written record
        assert_eq!(z.record.object_type, 17);
        assert_eq!(z.record.shape, H4Shape::Cylinder);
        assert_eq!(z.record.shape_values, [768, 512, 512, 0]);
        assert!(z.mode.is_none() || z.mode.is_some());
        let pl = b.placement();
        assert!((Vec3::from(b.record.fwd).dot(Vec3::new(1.0f32.cos(), 1.0f32.sin(), 0.0)) - 1.0).abs() < 1e-3);
        assert_eq!(pl.pos, [4.0, 5.0, 6.0]);
        assert!(instantiate(&c, &p, 200, 0, H4Pose::default()).is_none());
        assert_eq!(parse_place_spec("91.8@1,2,3; 0 ;bad;5.1"), vec![(91, 8, Some([1.0, 2.0, 3.0]), None), (0, 0, None, None), (5, 1, None, None)]);
        // #h4-veh the optional `:team` suffix (before the position)
        assert_eq!(parse_place_spec("87:1@4,5,6;83.2:8"), vec![(87, 0, Some([4.0, 5.0, 6.0]), Some(1)), (83, 2, None, Some(8))]);
    }
}

#[cfg(test)]
mod grab_ingame {
    //! #h4-grab-ingame regression: what the game does with an HMS-written record.
    use super::*;
    use crate::h4::cache::maps_dir;

    fn open(name: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(format!("{name}.map"));
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Why `instantiate` always writes an explicit variant index: the palette entries' variant blocks are laid out
    /// back to back in the scnr tag data, and `c_map_variant::copy_and_validate`
    /// (halo4.dll sub_1800B254C) clamps a record's variant byte to `min(v, count)` before
    /// `create_object` (sub_1800B6184) indexes the block with it UNCHECKED - so a NONE index
    /// (0xFF) resolves to `block[count]` = the NEXT
    /// entry's first variant (a Magnum placed in HMS spawned as an Assault Rifle) or, past the
    /// last contiguous block, a non-object tag. `instantiate` therefore always writes Some(v).
    #[test]
    fn variant_none_would_spawn_the_next_palette_entry() {
        for map in ["ca_forge_ravine", "dlc_forge_island"] {
            let Some(c) = open(map) else { continue };
            let d = c.data();
            let scnr = c.find_tags(b"scnr")[0];
            let sm = c.tag_meta(scnr).unwrap();
            let (np, po) = c.block(sm + OFF_SCNR_FORGE_PALETTE).unwrap();
            let mut blocks: Vec<(String, usize, usize)> = Vec::new();
            for i in 0..np {
                let pe = po + i * FORGE_PALETTE_ELEM;
                let Some((ne, eo)) = c.block(pe + 8) else { continue };
                for k in 0..ne {
                    let e = eo + k * FORGE_ENTRY_ELEM;
                    let (nv, vo) = c.block(e + 4).unwrap_or((0, 0));
                    blocks.push((c.sid(d.u32_at(e)), vo, nv));
                }
            }
            // quota 0 (Magnum, one variant): block[count] is quota 1's variant 0 (Assault Rifle)
            let (n0, vo0, nv0) = &blocks[0];
            let (n1, vo1, _) = &blocks[1];
            assert_eq!((n0.as_str(), *nv0), ("wep_magnum", 1));
            assert_eq!(vo0 + nv0 * FORGE_VARIANT_ELEM, *vo1, "{map}: the variant blocks are contiguous");
            let (_, next_tag) = c.tag_ref(vo1 + 4).expect("quota 1 variant 0 tag");
            assert_eq!(n1, "wep_assault_rifle");
            assert!(c.tag_name(next_tag).ends_with("storm_assault_rifle"), "{map}: {}", c.tag_name(next_tag));
            let contiguous = blocks.windows(2).filter(|w| w[0].2 > 0 && w[0].1 + w[0].2 * FORGE_VARIANT_ELEM == w[1].1).count();
            eprintln!("{map}: {} of {} consecutive palette entries have contiguous variant blocks", contiguous, blocks.len() - 1);
            assert!(contiguous >= 30, "{map}: the weapon / armour-ability / vehicle entries are contiguous (NONE lands on the next real object): {contiguous}");
            // and what HMS writes: an explicit index on every placement, 0 included
            let p = palette(&c);
            for (q, e) in p.entries.iter().enumerate() {
                for v in &e.variants {
                    if v.tag.is_none() { continue; }
                    let inst = instantiate(&c, &p, q as u8, v.index as u8, H4Pose::at([0.0; 3])).unwrap_or_else(|| panic!("{map}: quota {q} variant {} instantiates", v.index));
                    assert_eq!(inst.record.variant, Some(v.index as u8), "{map}: quota {q} '{}' variant {} is stored explicitly", e.name, v.index);
                    assert_eq!(inst.record.quota, Some(q as u8));
                }
            }
        }
    }

    /// A freshly instantiated record equals a GAME-placed one in every field the Forge grab
    /// path reads (halo4.dll sub_1800BD8C8: variant index present, quota valid, parent none,
    /// not locked; plus the placement / flags / scale / user data the object is created with).
    /// Reference values: the objects the game itself wrote into the user's 'Firebase Sandbox'
    /// (dlc_forge_island, MCC-saved, --dump-h4-mvar): ff_grid `flags 1 inb 1 parent -1 scale_q 7
    /// locked 0 color None user 0 q Some(98)/Some(0) place 0xCC team 8 spawn 0`, veh_mongoose
    /// `place 0x0C`, sp_initial_spawn `place 0x0C type 16`.
    #[test]
    fn new_record_matches_a_game_placed_one() {
        let Some(c) = open("dlc_forge_island") else { return };
        let p = palette(&c);
        let by_name = |n: &str| p.entries.iter().position(|e| e.name == n).unwrap_or_else(|| panic!("{n} in the palette")) as u8;
        for (name, placement, object_type) in [("ff_grid", 0xCCu16, 0u8), ("veh_mongoose", 0x0C, 7), ("sp_initial_spawn", 0x0C, 16)] {
            let q = by_name(name);
            let r = instantiate(&c, &p, q, 0, H4Pose::at([200.0, 100.0, -20.0])).unwrap().record;
            assert_eq!(r.flags, 1, "{name}: slot flags");
            assert!(r.in_bounds, "{name}: in bounds");
            assert_eq!(r.parent, -1, "{name}: no parent (copy_and_validate DROPS records with one)");
            assert_eq!(r.scale_q, crate::h4::mvar::SCALE_Q_DEFAULT, "{name}: scale q 7");
            assert!(!r.locked, "{name}: not locked");
            assert_eq!(r.quota, Some(q), "{name}: quota");
            assert_eq!(r.variant, Some(0), "{name}: EXPLICIT variant 0 like the game writes");
            assert_eq!(r.placement, placement, "{name}: placement");
            assert_eq!(r.object_type, object_type, "{name}: object type");
            assert_eq!(r.team, 8, "{name}: neutral");
            assert_eq!(r.color, None, "{name}: no colour override");
            assert_eq!(r.user_data, 0, "{name}: user data");
            assert_eq!(r.labels, [None; 4], "{name}: no labels");
            assert_eq!(r.type_data, H4TypeData::Pair(0, 0), "{name}: type tail");
            assert_eq!(r.shape, H4Shape::None, "{name}: no shape");
        }
        // the hidden (superforge) category on this map is the fixed `bb_*` block set: the grab
        // predicate rejects any object whose quota resolves to a category with flag bit 0
        let hidden: Vec<&ForgeCategory> = p.categories.iter().filter(|c| c.hidden).collect();
        assert!(!hidden.is_empty());
        for cat in hidden {
            for e in &p.entries[cat.first_entry..cat.first_entry + cat.entry_count] {
                assert!(e.name.starts_with("bb_"), "hidden category '{}' holds {}", cat.name, e.name);
            }
        }
    }
}
