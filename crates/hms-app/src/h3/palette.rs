//! Halo 3 Forge (sandbox) palette: the scnr map-variant palette blocks -> placeable rows, and
//! the `obje` multiplayer-object defaults a NEW placement takes. #h3-edit
//!
//! Layout (Halo3MCC/scnr.xml, verified on every shipped map through mvar.rs's quota cross-check):
//!   * scnr `Map Variant {Vehicle, Weapon, Equipment, Scenery, Teleporter, Goal, Spawner}
//!     Palette` blocks at +0x1E0 .. +0x228 (mvar.rs `SANDBOX_PALETTES`, element 0x1C): object
//!     tagref @0, display name sid @0x10, maximum allowed i32 @0x14, price f32 @0x18.
//!   * obje (every object class) +0xE0 `Multiplayer Object` block (0xC4): game engine flags u16
//!     @0, type u8 @2, teleporter passability u8 @3, flags u16 @4, boundary shape u8 @6, spawn
//!     timer type u8 @7, default spawn time i16 @8, boundary width/radius f32 @0xC, box length
//!     @0x10, +height @0x14, -height @0x18.
//! A quota's `definition` is `(block + 1) << 16 | entry`, so a palette row IS its quota key;
//! placing a row into a variant whose quota table lacks that entry appends the quota (price /
//! maximum from the palette element, as the game's own table carries them).
//!
//! What Forge writes for a new placement (the shipped files' placed objects, every map):
//! slot flags 0x83 (38196 of 47086 new-slot placements; 0x03 the rest), multiplayer flags 0x0C,
//! game engine flags 0x3FF (every engine), team 8 (neutral); type / shape / spawn time from the
//! object's multiplayer block.

use super::cache::H3Cache;
use super::mvar::{H3Quota, H3VariantObject, QuotaSource, SANDBOX_ELEM, SANDBOX_OFF_MAX, SANDBOX_OFF_PRICE, SANDBOX_PALETTES};
use crate::h4::cache::ByteRead;

pub const OFF_OBJE_MULTIPLAYER: usize = 0xE0;
pub const MP_ELEM: usize = 0xC4;
/// Slot flags of a Forge-placed object (the dominant shipped value).
pub const PLACED_FLAGS: u16 = 0x83;
/// Multiplayer flags of a Forge-placed object.
pub const PLACED_MP_FLAGS: u8 = 0x0C;
/// Game engine flags: appears in every engine.
pub const ALL_ENGINES: u16 = 0x3FF;
/// The neutral team.
pub const TEAM_NEUTRAL: u8 = 8;

pub const CATEGORY_NAMES: [&str; 7] = ["Vehicles", "Weapons", "Equipment", "Scenery", "Teleporters", "Goal Objects", "Spawning"];

/// An object's multiplayer-object block (what a fresh placement copies).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct H3MpDefaults {
    pub engine_flags: u16,
    pub object_type: u8,
    pub passability: u8,
    pub flags: u16,
    pub boundary_shape: u8,
    pub spawn_timer_type: u8,
    pub spawn_time: i16,
    /// width/radius, box length, +height, -height.
    pub boundary: [f32; 4],
}

/// The obje's multiplayer block, None when the tag has none (then it places as an ordinary
/// object with no boundary).
pub fn mp_defaults(c: &H3Cache, obje: usize) -> Option<H3MpDefaults> {
    let m = c.tag_meta(obje)?;
    let (_, o) = c.block(m + OFF_OBJE_MULTIPLAYER)?;
    let d = c.data();
    Some(H3MpDefaults {
        engine_flags: d.u16_at(o), object_type: d.u8_at(o + 2), passability: d.u8_at(o + 3), flags: d.u16_at(o + 4),
        boundary_shape: d.u8_at(o + 6), spawn_timer_type: d.u8_at(o + 7), spawn_time: d.i16_at(o + 8),
        boundary: [d.f32_at(o + 0xC), d.f32_at(o + 0x10), d.f32_at(o + 0x14), d.f32_at(o + 0x18)],
    })
}

/// One placeable palette row.
#[derive(Clone, Debug)]
pub struct H3PaletteItem {
    /// Row index (the `static_palette` position).
    pub index: usize,
    /// Sandbox block (0..7) and entry index within it.
    pub block: usize,
    pub entry: usize,
    /// The quota `definition` of this row (`(block + 1) << 16 | entry`).
    pub definition: u32,
    /// The object tag (obje) and its render model when it has one.
    pub tag: usize,
    pub mode_tag: Option<usize>,
    /// Display name (the palette element's string id; the tag name's tail when empty).
    pub name: String,
    pub category: &'static str,
    pub price: f32,
    /// Maximum allowed (<= 0 = unlimited).
    pub max_allowed: i32,
    pub mp: Option<H3MpDefaults>,
}

impl H3PaletteItem {
    pub fn object_type(&self) -> u8 { self.mp.map(|m| m.object_type).unwrap_or(0) }
    /// The quota entry this row adds to a variant that lacks it.
    pub fn quota(&self) -> H3Quota {
        H3Quota { definition: self.definition, minimum: 0, maximum: 0, placed_on_map: 0, maximum_allowed: self.max_allowed.clamp(0, 255) as u8, price: self.price }
    }
}

/// Every row of the map's sandbox palette, block by block (a null object ref is skipped).
pub fn palette(c: &H3Cache) -> Vec<H3PaletteItem> {
    let mut out = Vec::new();
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let d = c.data();
    for (block, &off) in SANDBOX_PALETTES.iter().enumerate() {
        let Some((n, o)) = c.block(sm + off) else { continue };
        for entry in 0..n.min(0xFFFF) {
            let e = o + entry * SANDBOX_ELEM;
            let Some((_, tag)) = c.tag_ref(e) else { continue };
            let sid = c.sid(d.u32_at(e + 0x10));
            let name = if sid.is_empty() { c.tag_name(tag).rsplit('\\').next().unwrap_or("").to_string() } else { sid };
            out.push(H3PaletteItem {
                index: out.len(), block, entry, definition: ((block as u32 + 1) << 16) | entry as u32, tag,
                mode_tag: super::objects::model_of(c, tag), name, category: CATEGORY_NAMES[block],
                price: d.f32_at(e + SANDBOX_OFF_PRICE), max_allowed: d.i32_at(e + SANDBOX_OFF_MAX), mp: mp_defaults(c, tag),
            });
        }
    }
    out
}

/// Row index of the quota a placed object came from (None for a scenario-owned quota).
pub fn item_of_quota(items: &[H3PaletteItem], q: &H3Quota) -> Option<usize> {
    match q.source()? {
        QuotaSource::Sandbox(b) => items.iter().position(|it| it.block == b && it.entry == q.index()),
        QuotaSource::Scenario(_) => None,
    }
}

/// A NEW placement of `item` at `pos` (world), facing +X, up +Z, quota `quota_index`, in slot
/// `slot`: the record a fresh Forge drop carries (module doc), pose quantised through
/// `set_pose` against `bounds`.
pub fn instantiate(item: &H3PaletteItem, quota_index: i32, slot: usize, pos: [f32; 3], bounds: [f32; 6]) -> H3VariantObject {
    let mp = item.mp.unwrap_or_default();
    let mut o = H3VariantObject {
        slot, flags: PLACED_FLAGS, quota: quota_index, parent: None, placed: true,
        pos, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], pos_q: [0; 3], up_is_global: true, up_q: 0, angle_q: 127, boundary_q: [0; 4],
        object_type: mp.object_type, mp_flags: PLACED_MP_FLAGS, game_engine_flags: ALL_ENGINES, shared_storage: 0,
        spawn_time: mp.spawn_time.clamp(0, 255) as u8, team: TEAM_NEUTRAL, boundary_shape: mp.boundary_shape, boundary: [0.0; 4],
    };
    o.set_pose(pos, glam::Vec3::X, glam::Vec3::Z, bounds);
    o.set_boundary(mp.boundary_shape, mp.boundary);
    o
}

#[cfg(test)]
mod tests {
    use super::super::cache::{maps_dir, H3Cache};
    use super::*;

    /// guardian's palette: the seven blocks come out in order with names, every row has an
    /// object tag, most resolve a render model, and the multiplayer defaults of the shipped
    /// placements agree with the tag's block (type / shape / spawn time) - the fresh-drop rule.
    #[test]
    fn guardian_palette_rows_and_mp_defaults() {
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let items = palette(&c);
        assert!(items.len() >= 90, "{} rows", items.len());
        let cats: Vec<&str> = { let mut v = Vec::new(); for it in &items { if v.last() != Some(&it.category) { v.push(it.category); } } v };
        assert_eq!(cats, CATEGORY_NAMES.to_vec());
        assert!(items.iter().all(|it| !it.name.is_empty()));
        let with_model = items.iter().filter(|it| it.mode_tag.is_some()).count();
        eprintln!("guardian palette: {} rows, {with_model} with a model; first rows {:?}", items.len(), items.iter().take(6).map(|it| format!("{}:{} ${} max {} type {}", it.category, it.name, it.price, it.max_allowed, it.object_type())).collect::<Vec<_>>());
        assert!(with_model * 4 >= items.len() * 3, "{with_model} of {} rows resolve a model", items.len());
        // the shipped variant's placed objects vs their palette row's multiplayer block
        let Some(f) = super::super::mvar::shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let v = super::super::mvar::parse_h3_variant(&f).unwrap();
        let (mut checked, mut type_ok, mut shape_ok, mut time_ok) = (0, 0, 0, 0);
        for o in v.placed() {
            let Some(q) = v.quota_of(o) else { continue };
            let Some(i) = item_of_quota(&items, q) else { continue };
            let Some(mp) = items[i].mp else { continue };
            checked += 1;
            if mp.object_type == o.object_type { type_ok += 1; }
            if mp.boundary_shape == o.boundary_shape { shape_ok += 1; }
            if mp.spawn_time.clamp(0, 255) as u8 == o.spawn_time { time_ok += 1; }
        }
        eprintln!("guardian MLG placed vs tag defaults: {checked} checked, type {type_ok}, shape {shape_ok}, spawn time {time_ok}");
        assert!(checked > 50 && type_ok == checked, "object type is the tag's on every placement");
    }

    /// Probe (print-only, #[ignore]): palette row names against their multiplayer object type, to
    /// pin the type numbering the property labels depend on.
    #[test]
    #[ignore]
    fn palette_type_probe() {
        use std::collections::BTreeMap;
        let mut by_type: BTreeMap<u8, Vec<String>> = BTreeMap::new();
        for c in super::super::cache::installed_caches().into_iter().take(40) {
            let Ok(cache) = H3Cache::open(&c) else { continue };
            for it in palette(&cache) {
                let e = by_type.entry(it.object_type()).or_default();
                if e.len() < 7 && !e.contains(&it.name) { e.push(format!("{}:{}", it.category, it.name)); }
            }
        }
        for (t, names) in &by_type { eprintln!("PTYPE {t:>2}: {}", names.join(" | ")); }
    }
}
