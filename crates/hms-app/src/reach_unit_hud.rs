//! #seat-hud  WHICH `chud` a unit sitting in a seat sees, and what each piece of it is for.
//!
//! The owner's report: *"we also need the crosshair for the vehicle, they will have a different
//! one depending on the vehicle, it should be in the seat block for the vehicle."* It is.
//!
//! # What the engine does (PROVEN, `reach_tag_test.exe`, which carries the symbols)
//!
//! `sub_1408BED90` (`chud.cpp`, the per-player snapshot the HUD manager builds every tick) fills
//! eleven `chud definition` SLOTS, 0x60 bytes apart from `+0x20` of the snapshot. A slot is a
//! `(chdt tag, source object)` pair and EVERY non-empty slot is drawn
//! (`sub_1408C7EB0` walks slots 0..10 and updates each in turn). The slots this module cares
//! about:
//!
//! ```text
//!   0 / 3   the unit's primary weapon     tag = weap +0x3C4 `chud interface` (datum at +0x3D0)
//!   1       the unit's second weapon      (same, from the +0x34B inventory index)
//!   2       the BACKPACK weapon           (same)
//!   4       the weapon of the VEHICLE the unit sits in, when the seat is a GUNNER seat
//!   5       the player's own HUD          globals `player representation` -> spartan / elite
//!   6       THE SEAT'S HUD                `sub_140D87780(parent unit tag, seat index)`
//! ```
//!
//! `sub_140D87780` is three lines: take the seat element (`unit +0x444`, stride 0x12C), read its
//! `unit hud interface` block at `+0xE8` (`unit_hud_reference_block`, 16-byte elements, max
//! `NUMBER_OF_UNIT_HUD_TYPES` = 2, each one a single `chud interface` tag reference), and return
//! ELEMENT 0's tag datum (`+0x0C`), or NONE when the block is empty. That is the whole answer to
//! "where is the vehicle's crosshair": a vehicle's reticule is a collection of the `chud` its
//! seat names, exactly as a weapon's is a collection of the `chud` the weapon names.
//!
//! Which of the unit's OWN weapon slots survive a seat is decided by the seat's flags
//! (`seat +0x00`, read in the same function from the snapshot of the unit's parent):
//!
//! * bit 5 `allows weapons` SET: slots 0/3, 1 and 2 are added as on foot (`LABEL_1` in the
//!   decompile, then the backpack weapon). CLEAR: none of them. A driver of a jeep sees NO
//!   biped-weapon HUD at all, ammo and reticule both.
//! * bit 3 `gunner` SET: slot 4 is added from the vehicle the unit sits in
//!   (`weapon = inventory[ vehicle +0x34A ]`, falling back to the first ancestor that forwards
//!   its weapons), whatever the allows-weapons bit says.
//!
//! The seat HUD (slot 6) is added whenever the unit has a parent and a valid seat index, with no
//! flag test at all.
//!
//! # What this module is
//!
//! The decision, with the tag reads behind it, and nothing else: [`seat_hud`] (the
//! `sub_140D87780` reader), [`vehicle_weapon`], and [`SeatHuds::for_seat`] which applies the
//! flag rules above. Painting is [`crate::reach_hud`]'s: the same widget decode and the same
//! `paint_collection`.

use hms_native::{CacheHandle, NativeDll};

use crate::reach_seats::{Seat, S_HUD_INTERFACE, SEAT_MAX, SEAT_STRIDE, UNIT_SEATS};

/// `unit_hud_reference_block` element size (`sizeof(struct unit_hud_reference)`), straight off
/// the block's definition.
pub const HUD_REFERENCE_STRIDE: usize = 16;

/// `unit +0x42C` `weapons` (`unit_weapon_block`, max 4, 20 bytes: `weapon` tag reference at +0,
/// `position` enum at +0x10). `tagdefs` prints it at `+?394`; 0x98 is the same runtime-installed
/// shift every `?` offset from `+?348` onward carries (see [`crate::reach_seats`]).
pub const UNIT_WEAPONS: usize = 0x42C;
pub const UNIT_WEAPON_STRIDE: usize = 20;
pub const UNIT_WEAPON_MAX: usize = 4;

/// `seat flags` bit 3, the one `sub_1408BED90` tests (`v57 & 8`) before it adds the vehicle's
/// weapon. NOT [`crate::reach_seats::FLAG_GUNNER_PLAYER_ONLY`]: the engine does not look at it.
pub const FLAG_GUNNER: u32 = 1 << 3;

/// `unit +0x168` `unit_flags`, and its bit 25 `controlled by parent gunner` (enum member index 25 of
/// the 30 in `unit_flags`, read from the tool build's `.data`). A child unit carrying it is the
/// parent's gun turret: the engine looks for the gunner's weapon there when the parent itself
/// holds none (`sub_1408BED90`, the sibling loop over `object +0x10` / `+0x0C`).
pub const UNIT_FLAGS: usize = 0x168;
pub const UNIT_FLAG_CONTROLLED_BY_PARENT_GUNNER: u32 = 1 << 25;

fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// A tag reference's datum, as a tag index, or `None` for a null one. `+0x0C` of a 16-byte
/// reference, the same read `reach_hud::tag_ref` does.
fn datum_of(b: &[u8], o: usize) -> Option<u32> {
    let d = rd_u32(b, o + 0x0C);
    (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
}

/// `sub_140D87780(unit, seat)`: the `chud interface` a seat names, element 0 of its `unit hud
/// interface` block. `None` when the unit has no such seat or the seat names no HUD.
pub fn seat_hud(native: &NativeDll, cache: CacheHandle, unit_tag: u32, seat_index: usize) -> Option<u32> {
    let h = native.tag_meta(cache, unit_tag, 0, (UNIT_SEATS + 8) as u32)?;
    let count = rd_u32(&h, UNIT_SEATS) as usize;
    if seat_index >= count || count > SEAT_MAX {
        return None;
    }
    let seats = native.tag_ptr_bytes(cache, rd_u32(&h, UNIT_SEATS + 4), (count * SEAT_STRIDE) as u32)?;
    let e = &seats[seat_index * SEAT_STRIDE..(seat_index + 1) * SEAT_STRIDE];
    let n = rd_u32(e, S_HUD_INTERFACE) as usize;
    if n == 0 || n > 2 {
        return None;
    }
    let refs = native.tag_ptr_bytes(cache, rd_u32(e, S_HUD_INTERFACE + 4), (n * HUD_REFERENCE_STRIDE) as u32)?;
    datum_of(&refs, 0)
}

/// `unit +0x42C` `weapons`, element `index`: the `weap` a vehicle starts with.
pub fn unit_weapon(native: &NativeDll, cache: CacheHandle, unit_tag: u32, index: usize) -> Option<u32> {
    let h = native.tag_meta(cache, unit_tag, 0, (UNIT_WEAPONS + 8) as u32)?;
    let count = rd_u32(&h, UNIT_WEAPONS) as usize;
    if index >= count || count > UNIT_WEAPON_MAX {
        return None;
    }
    let b = native.tag_ptr_bytes(cache, rd_u32(&h, UNIT_WEAPONS + 4), (count * UNIT_WEAPON_STRIDE) as u32)?;
    datum_of(&b[index * UNIT_WEAPON_STRIDE..], 0)
}

/// The `weap` a gunner seat of `unit_tag` fires, the way the engine finds it for slot 4: the
/// vehicle's own first weapon, else the first weapon of a CHILD unit flagged `controlled by
/// parent gunner` (the Scorpion's cannon, the Wraith's mortar, the Falcon's chin gun are children
/// of a hull whose own `weapons` block is empty).
///
/// "First" is attachment order, which is the order the engine creates (and so lists) the
/// children in; HMS has no switch for a vehicle's second weapon, so index 0 is always the active
/// one.
pub fn vehicle_weapon(native: &NativeDll, cache: CacheHandle, unit_tag: u32) -> Option<u32> {
    if let Some(w) = unit_weapon(native, cache, unit_tag, 0) {
        return Some(w);
    }
    let vehis = native.tags_by_group(cache, "vehi");
    native.enumerate_attachments(cache, unit_tag).iter().find_map(|a| {
        let c = a.child_obj;
        if !vehis.contains(&c) {
            return None;
        }
        let f = rd_u32(&native.tag_meta(cache, c, UNIT_FLAGS as u32, 4)?, 0);
        (f & UNIT_FLAG_CONTROLLED_BY_PARENT_GUNNER != 0).then(|| unit_weapon(native, cache, c, 0)).flatten()
    })
}

/// Which HUDs a unit sees from one seat, in the engine's own slot terms (module comment).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeatHuds {
    /// Slot 6: the `chud` the seat names (`seat +0xE8`, element 0).
    pub seat: Option<u32>,
    /// Slot 4: the `weap` the vehicle itself carries, present only in a seat flagged `gunner`.
    /// `unit +0x42C` element 0 -- the engine's `+0x34A` "current weapon" starts there, and test
    /// mode has no weapon-switch for a vehicle, so element 1 (the Banshee's bomb, the Sabre's
    /// missiles) is never the active one.
    pub vehicle_weapon: Option<u32>,
    /// That weapon's own `chud interface` (`weap +0x3C4`), the tag slot 4 draws.
    pub vehicle_weapon_hud: Option<u32>,
    /// Slots 0..2: does the unit's OWN weapon HUD (reticule and ammo) survive this seat?
    pub biped_weapons: bool,
    /// The seat's camera is the chase (third-person) one: seat flag bit 4 `third person camera`.
    pub third_person: bool,
}

impl SeatHuds {
    /// On foot: the biped's own weapon HUD and nothing from a vehicle.
    pub fn on_foot() -> SeatHuds {
        SeatHuds { biped_weapons: true, ..Default::default() }
    }

    /// Apply `sub_1408BED90`'s rules to one seat of `unit_tag`.
    pub fn for_seat(native: &NativeDll, cache: CacheHandle, unit_tag: u32, seat: &Seat) -> SeatHuds {
        let vehicle_weapon =
            (seat.flags & FLAG_GUNNER != 0).then(|| vehicle_weapon(native, cache, unit_tag)).flatten();
        SeatHuds {
            seat: seat_hud(native, cache, unit_tag, seat.index),
            vehicle_weapon,
            vehicle_weapon_hud: vehicle_weapon.and_then(|w| crate::reach_chud::weapon_chud(native, cache, w)),
            biped_weapons: seat.allows_weapons(),
            third_person: seat.third_person(),
        }
    }

    /// Does the biped's own weapon HUD (reticule, ammo block) draw? `weapon_usable` is the ride's
    /// own phase-aware answer ([`crate::play_mode::PlayMode::weapon_usable`]); the seat's flag has
    /// to agree. `None` = on foot.
    pub fn biped_hud(weapon_usable: bool, seat: Option<&SeatHuds>) -> bool {
        weapon_usable && seat.map_or(true, |h| h.biped_weapons)
    }

    /// Does this seat put ANY vehicle HUD on screen?
    pub fn any_vehicle_hud(&self) -> bool {
        self.seat.is_some() || self.vehicle_weapon_hud.is_some()
    }
}

/// What [`paint_seat_huds`] put on screen, as numbers a gate can read.
#[derive(Clone, Debug, Default)]
pub struct SeatPaint {
    /// One line per decision, for the HUD panel.
    pub lines: Vec<String>,
    /// The vehicle weapon's reticule: the union of its parts' screen rectangles, `None` = none drawn.
    pub crosshair: Option<eframe::egui::Rect>,
    pub crosshair_parts: usize,
    /// The seat `chud`'s own reticule (a Warthog driver's `v_driver`), same meaning.
    pub seat_crosshair: Option<eframe::egui::Rect>,
    pub seat_crosshair_parts: usize,
    /// Widgets of the seat's own `chud` drawn / left out, and the collections that drew any.
    pub seat_drawn: usize,
    pub seat_skipped: usize,
    pub seat_collections: Vec<String>,
}

/// Paint the vehicle's part of the HUD: slot 4 (the vehicle weapon's reticule) and slot 6 (every
/// collection of the seat's own `chud`). The biped's slots 0..2 and the player's slot 5 are the
/// caller's, exactly as on foot.
///
/// `values` is what the seat `chud` is fed. The reticule is the same composed-part painter a
/// handheld weapon's uses (`reach_chud::paint_crosshair`), at the `crosshair` anchor
/// (`chud_anchor_type_enum` 17) which for these views is the centre of the viewport.
pub fn paint_seat_huds(
    ctx: &eframe::egui::Context,
    painter: &eframe::egui::Painter,
    vs: &crate::reach_hud::VirtualScreen,
    native: &NativeDll,
    cache: CacheHandle,
    huds: &SeatHuds,
    skin: Option<&crate::reach_chud::ChudSkin>,
    values: &crate::reach_hud::HudValues,
    crosshair_on: bool,
    hud_on: bool,
) -> SeatPaint {
    let name = |t: u32| native.tag_name(cache, t).unwrap_or_else(|| format!("tag {t:#x}"));
    let mut out = SeatPaint::default();

    // ---- slot 4: the vehicle weapon's reticule ---------------------------------------------
    match (huds.vehicle_weapon, crosshair_on) {
        (Some(w), true) => match (crate::reach_chud::crosshair_for(ctx, native, cache, w), skin) {
            (Some(art), Some(sk)) => {
                out.crosshair = Some(crate::reach_chud::paint_crosshair(painter, vs.viewport, &art, sk, vs.vpx));
                out.crosshair_parts = art.crosshair.parts.len();
                out.lines.push(format!(
                    "Vehicle crosshair: {} authored part(s) from {} (the weapon {})",
                    out.crosshair_parts,
                    name(art.crosshair.chud_tag),
                    name(w)
                ));
            }
            (None, _) => out.lines.push(format!(
                "Vehicle crosshair: {} authors no reticule that draws at rest (or names no `chud interface`)",
                name(w)
            )),
            (_, None) => out.lines.push("Vehicle crosshair: the chud globals carry no skin to colour it with".into()),
        },
        (Some(w), false) => out.lines.push(format!("Vehicle crosshair: {} is switched off (Crosshair)", name(w))),
        (None, _) => {}
    }

    // ---- slot 6: the seat's own chud --------------------------------------------------------
    //
    // Its `crosshair` collection is anchored at 255 (no placement of its own) with the widgets at
    // anchor 17, which `reach_hud::paint_collection` cannot place -- that is why a handheld
    // weapon's reticule has its own painter, and why a seat's goes through the same one.
    if let Some(t) = huds.seat {
        if crosshair_on {
            if let (Some(art), Some(sk)) = (crate::reach_chud::crosshair_for_chud(ctx, native, cache, t), skin) {
                out.seat_crosshair = Some(crate::reach_chud::paint_crosshair(painter, vs.viewport, &art, sk, vs.vpx));
                out.seat_crosshair_parts = art.crosshair.parts.len();
                out.lines.push(format!("Seat crosshair: {} authored part(s) from {}", out.seat_crosshair_parts, name(t)));
            }
        }
        match crate::reach_hud::hud_art(ctx, native, cache, t).filter(|_| hud_on) {
            Some(a) => {
                for i in 0..a.chud.collections.len() {
                    let (d, s) = crate::reach_hud::paint_collection(ctx, painter, vs, &a, i, skin, values, 0);
                    if d > 0 {
                        out.seat_collections.push(a.chud.collections[i].base.name.clone());
                    }
                    out.seat_drawn += d;
                    out.seat_skipped += s;
                }
                out.lines.push(format!(
                    "Seat HUD: {} - {} widget(s) drawn from [{}], {} left out for want of a value",
                    name(t),
                    out.seat_drawn,
                    out.seat_collections.join(", "),
                    out.seat_skipped
                ));
            }
            None if hud_on => out.lines.push(format!("Seat HUD: '{}' did not decode", name(t))),
            None => {}
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod gates {
    //! #seat-hud  The seat -> HUD rules, measured against the shipped `forge_halo` cache.
    //!
    //! `cargo test -p hms-app --release reach_unit_hud -- --ignored --nocapture --test-threads=1`

    use super::*;
    use crate::reach_hud as rh;
    use crate::reach_seats::UnitSeats;

    type Open = (std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle);

    fn open() -> Option<Open> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    fn name(native: &NativeDll, cache: CacheHandle, t: Option<u32>) -> String {
        t.and_then(|t| native.tag_name(cache, t)).unwrap_or_else(|| "-".into())
    }

    /// The `vehi` whose path ends with `\<tail>`.
    fn vehi(native: &NativeDll, cache: CacheHandle, tail: &str) -> u32 {
        let want = format!("\\{tail}");
        native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(&want)))
            .unwrap_or_else(|| panic!("forge_halo has no vehi ending {want}"))
    }

    /// Every seat of every vehicle, with the HUDs the production code resolves for it.
    fn every_seat(native: &NativeDll, cache: CacheHandle) -> Vec<(u32, Seat, SeatHuds)> {
        let mut out = Vec::new();
        for v in native.tags_by_group(cache, "vehi") {
            let Some(us) = UnitSeats::read(native, cache, v) else { continue };
            for s in us.seats {
                let h = SeatHuds::for_seat(native, cache, v, &s);
                out.push((v, s, h));
            }
        }
        out
    }

    fn huds_of(native: &NativeDll, cache: CacheHandle, tail: &str, seat: usize) -> SeatHuds {
        let v = vehi(native, cache, tail);
        let us = UnitSeats::read(native, cache, v).expect("seats");
        SeatHuds::for_seat(native, cache, v, &us.seats[seat])
    }

    /// **THE SEAT BLOCK IS WHERE THE HUD COMES FROM.**
    ///
    /// Liveness first: 16 of the 138 seats in `forge_halo` name a `chud` and the gate demands the
    /// reader finds them (a reader aimed at the wrong offset finds zero). Then the INDEPENDENT
    /// route: scan the raw 300-byte seat element for ANY (count, pointer) pair whose elements
    /// reference a `chdt`, which does not know `S_HUD_INTERFACE`. Over all 138 seats that scan has
    /// to hit at exactly one offset and name the same tag the reader names.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn a_seat_names_its_hud_in_the_seat_block() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let chdts: Vec<u32> = native.tags_by_group(cache, "chdt");
        let rows = every_seat(native, cache);
        assert!(rows.len() >= 100, "only {} seats read; the walk is not reaching the vehicles", rows.len());
        let resolved: Vec<_> = rows.iter().filter(|r| r.2.seat.is_some()).collect();
        assert!(resolved.len() >= 12, "only {} of {} seats resolved a HUD", resolved.len(), rows.len());
        for (_, s, h) in &resolved {
            let t = h.seat.unwrap();
            assert!(chdts.contains(&t), "seat {} resolved tag {t:#x}, which is not a chdt", s.label);
            assert!(name(native, cache, Some(t)).starts_with("ui\\chud\\"), "seat {} HUD {:?}", s.label, name(native, cache, Some(t)));
        }

        // ---- the independent scan ----
        let mut offsets = std::collections::BTreeSet::new();
        let mut scanned = 0usize;
        let mut mismatched = Vec::new();
        for v in native.tags_by_group(cache, "vehi") {
            let hdr = native.tag_meta(cache, v, 0, (crate::reach_seats::UNIT_SEATS + 8) as u32);
            let Some(hdr) = hdr else { continue };
            let count = rd_u32(&hdr, crate::reach_seats::UNIT_SEATS) as usize;
            if count == 0 || count > crate::reach_seats::SEAT_MAX {
                continue;
            }
            let Some(b) = native.tag_ptr_bytes(
                cache,
                rd_u32(&hdr, crate::reach_seats::UNIT_SEATS + 4),
                (count * SEAT_STRIDE) as u32,
            ) else {
                continue;
            };
            for i in 0..count {
                let e = &b[i * SEAT_STRIDE..(i + 1) * SEAT_STRIDE];
                scanned += 1;
                let mut found: Vec<(usize, u32)> = Vec::new();
                for o in (0..SEAT_STRIDE - 8).step_by(4) {
                    let n = rd_u32(e, o) as usize;
                    if !(1..=2).contains(&n) {
                        continue;
                    }
                    if let Some(refs) = native.tag_ptr_bytes(cache, rd_u32(e, o + 4), (n * HUD_REFERENCE_STRIDE) as u32) {
                        if let Some(d) = datum_of(&refs, 0).filter(|d| chdts.contains(d)) {
                            found.push((o, d));
                        }
                    }
                }
                for (o, _) in &found {
                    offsets.insert(*o);
                }
                let mine = rows.iter().find(|r| r.0 == v && r.1.index == i).and_then(|r| r.2.seat);
                if found.first().map(|f| f.1) != mine {
                    mismatched.push((name(native, cache, Some(v)), i, found.clone(), mine));
                }
            }
        }
        assert!(scanned >= 100, "the independent scan read only {scanned} seats");
        assert!(mismatched.is_empty(), "the raw scan disagrees with the reader: {mismatched:?}");
        assert_eq!(
            offsets.into_iter().collect::<Vec<_>>(),
            vec![S_HUD_INTERFACE],
            "the scan must find a chdt reference at the seat's `unit hud interface` offset and nowhere else"
        );
    }

    /// **THE HUD DIFFERS BY VEHICLE AND SEAT, AND IS THE SEAT'S OWN.**
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn each_vehicle_seat_gets_its_own_hud() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let n = |t: Option<u32>| name(native, cache, t);
        let pair = |h: &SeatHuds| (n(h.seat), n(h.vehicle_weapon_hud));
        let warthog_gunner = huds_of(native, cache, "warthog_chaingun\\warthog_chaingun", 0);
        let ghost_driver = huds_of(native, cache, "ghost\\ghost", 0);
        let falcon_driver = huds_of(native, cache, "falcon\\falcon", 0);
        let warthog_driver = huds_of(native, cache, "warthog\\warthog", 0);
        let mongoose_driver = huds_of(native, cache, "mongoose\\mongoose", 0);
        let banshee_driver = huds_of(native, cache, "banshee\\banshee", 0);
        let scorpion_driver = huds_of(native, cache, "scorpion\\scorpion", 0);
        eprintln!("warthog gunner : {:?}", pair(&warthog_gunner));
        eprintln!("warthog driver : {:?}", pair(&warthog_driver));
        eprintln!("mongoose driver: {:?}", pair(&mongoose_driver));
        eprintln!("ghost driver   : {:?}", pair(&ghost_driver));
        eprintln!("banshee driver : {:?}", pair(&banshee_driver));
        eprintln!("falcon driver  : {:?}", pair(&falcon_driver));
        eprintln!("scorpion driver: {:?}", pair(&scorpion_driver));
        // The three the owner's report names, each authored differently.
        assert_eq!(pair(&warthog_gunner), ("-".into(), "ui\\chud\\v_chaingun".into()));
        assert_eq!(pair(&ghost_driver), ("ui\\chud\\v_boost_meter".into(), "ui\\chud\\v_ghost".into()));
        assert_eq!(pair(&warthog_driver).0, "ui\\chud\\v_driver");
        assert_eq!(pair(&mongoose_driver).0, "ui\\chud\\v_mongoose");
        assert_eq!(pair(&falcon_driver).0, "ui\\chud\\v_falcon");
        let all = [&warthog_gunner, &ghost_driver, &falcon_driver, &warthog_driver, &mongoose_driver];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(pair(a), pair(b), "two different seats resolved the same HUD pair");
            }
        }
        // A hull whose own `weapons` block is empty takes the gun from the CHILD turret flagged
        // `controlled by parent gunner`: the Scorpion's cannon, found through its attachments.
        assert!(
            scorpion_driver.vehicle_weapon.is_some(),
            "the Scorpion's gunner weapon is on a child turret and the attachment search did not find it"
        );
        assert!(falcon_driver.vehicle_weapon.is_some(), "the Falcon's chin gun is a child turret too");
    }

    /// **THE FLAGS DECIDE WHICH SLOTS SURVIVE.**
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_seat_flags_decide_which_huds_survive() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let rows = every_seat(native, cache);
        // Structure, over every seat: the biped's HUD survives exactly where the seat allows
        // weapons, and a vehicle weapon is only ever offered to a `gunner` (bit 3) seat.
        let (mut kept, mut dropped, mut armed) = (0, 0, 0);
        for (_, s, h) in &rows {
            assert_eq!(h.biped_weapons, s.allows_weapons(), "seat {}", s.label);
            if h.vehicle_weapon.is_some() {
                assert!(s.flags & FLAG_GUNNER != 0, "seat {} was given a vehicle weapon without bit 3", s.label);
                armed += 1;
            }
            if h.biped_weapons { kept += 1 } else { dropped += 1 }
        }
        assert!(kept >= 20 && dropped >= 40 && armed >= 15, "kept {kept} dropped {dropped} armed {armed}: the census is not reaching the cache");

        let warthog_driver = huds_of(native, cache, "warthog\\warthog", 0);
        assert!(!warthog_driver.biped_weapons, "a Warthog driver stows the biped's weapon");
        assert_eq!(warthog_driver.vehicle_weapon, None, "the horn is not a gunner weapon: bit 3 is clear");
        let warthog_passenger = huds_of(native, cache, "warthog\\warthog", 1);
        assert!(warthog_passenger.biped_weapons, "a Warthog passenger keeps the biped's weapon");
        assert!(!warthog_passenger.any_vehicle_hud(), "and sees no vehicle HUD: the seat names none");
        let pelican = huds_of(native, cache, "pelican\\pelican", 4);
        assert!(pelican.biped_weapons && !pelican.any_vehicle_hud(), "a Pelican troop seat: the biped's HUD alone");
        let ghost = huds_of(native, cache, "ghost\\ghost", 0);
        assert!(!ghost.biped_weapons && ghost.seat.is_some() && ghost.vehicle_weapon_hud.is_some());
        // The pure rule main.rs applies, including the ride's own phase-aware veto.
        assert!(SeatHuds::biped_hud(true, None), "on foot");
        assert!(!SeatHuds::biped_hud(true, Some(&warthog_driver)));
        assert!(SeatHuds::biped_hud(true, Some(&warthog_passenger)));
        assert!(!SeatHuds::biped_hud(false, Some(&warthog_passenger)), "the ride says the weapon is not usable yet");
        assert_eq!(SeatHuds::on_foot(), SeatHuds { biped_weapons: true, ..Default::default() });
        assert!(!SeatHuds::on_foot().any_vehicle_hud());

        // `PlayHud` keeps it per boarding and clears it on foot.
        let v = vehi(native, cache, "warthog\\warthog");
        let mut hud = crate::play_hud::PlayHud::default();
        hud.bind_seat(native, cache, Some((v, 0)));
        assert_eq!(hud.seat_huds.as_ref().map(|(_, h)| h.clone()), Some(warthog_driver));
        hud.bind_seat(native, cache, Some((v, 1)));
        assert_eq!(hud.seat_huds.as_ref().map(|(_, h)| h.clone()), Some(warthog_passenger));
        hud.bind_seat(native, cache, None);
        assert!(hud.seat_huds.is_none(), "getting out returns to the on-foot HUD");
    }

    /// One row of the authored crosshair of `chud_tag`: `(widget, bitmap tag, submap, size vu,
    /// scale, offset)` for every bitmap widget of the collection named `crosshair` at the
    /// crosshair anchor (17), read through `read_chud` -- NOT through `weapon_crosshair`.
    fn authored(native: &NativeDll, cache: CacheHandle, chud_tag: u32) -> Vec<(String, u32, u16, [f32; 2], [f32; 2], [f32; 2], i32, [f32; 2], u32)> {
        let Some(c) = rh::read_chud(native, cache, chud_tag) else { return Vec::new() };
        let mut out = Vec::new();
        for coll in c.collections.iter().filter(|c| c.base.name == "crosshair") {
            for w in &coll.widgets {
                let (rh::WidgetArt::Bitmap { sprite, flags, .. }, Some(p)) = (&w.art, w.base.placement()) else { continue };
                if p.anchor == rh::ANCHOR_CROSSHAIR {
                    out.push((w.base.name.clone(), sprite.bitmap_tag, sprite.bitmap_index, sprite.size, p.scale, p.offset, w.base.blend_mode, p.origin, *flags));
                }
            }
        }
        out
    }

    /// The screen rectangle an authored crosshair widget covers, built from the tag alone: the
    /// viewport centre plus the offset, the sprite pivoted by its `widget origin`, and the
    /// reflections `mirror horizontal` / `mirror vertical` (bitmap flags 0 / 1) add about the point.
    fn expected(vs: &rh::VirtualScreen, a: &(String, u32, u16, [f32; 2], [f32; 2], [f32; 2], i32, [f32; 2], u32)) -> eframe::egui::Rect {
        use eframe::egui::{pos2, vec2, Rect};
        let c = vs.viewport.center() + vec2(a.5[0], a.5[1]) * vs.vpx;
        let sz = vec2(a.3[0] * a.4[0].abs(), a.3[1] * a.4[1].abs()) * vs.vpx;
        let mut u = Rect::from_center_size(c + vec2((0.5 - a.7[0]) * sz.x, (0.5 - a.7[1]) * sz.y), sz);
        if a.8 & 1 != 0 {
            u = u.union(Rect::from_min_max(pos2(2.0 * c.x - u.max.x, u.min.y), pos2(2.0 * c.x - u.min.x, u.max.y)));
        }
        if a.8 & 2 != 0 {
            u = u.union(Rect::from_min_max(pos2(u.min.x, 2.0 * c.y - u.max.y), pos2(u.max.x, 2.0 * c.y - u.min.y)));
        }
        u
    }

    /// **THE DRAW DATA.** Run the production painter headlessly at 1920x1080 for each seat and
    /// read what it emitted: the seat `chud`'s crosshair widgets through the widget trace, the
    /// vehicle weapon's reticule through its returned rectangle. Both are compared to rectangles
    /// built from the AUTHORED placement (anchor 17 = the viewport centre, the widget's own
    /// offset, sprite size x scale x points-per-virtual-unit), which `read_chud` decodes on its own.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_vehicle_crosshair_is_drawn_where_the_widget_authors_it() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = eframe::egui::Context::default();
        let vp = eframe::egui::Rect::from_min_size(eframe::egui::pos2(0.0, 0.0), eframe::egui::vec2(1920.0, 1080.0));
        let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned());
        let vs = rh::VirtualScreen::from_skin(vp, skin.as_ref().expect("chud globals carry a skin"));
        let near = |a: eframe::egui::Rect, b: eframe::egui::Rect| {
            (a.min - b.min).abs().max_elem() < 0.6 && (a.max - b.max).abs().max_elem() < 0.6
        };
        println!("\n| vehicle | seat | HUD tag | crosshair bitmaps (tag, submap) | size pt | centre pt |");
        println!("|---|---|---|---|---|---|");
        let mut seen_seat_crosshair = 0;
        let mut seen_weapon_crosshair = 0;
        for (tail, seat) in [
            ("warthog\\warthog", 0),
            ("mongoose\\mongoose", 0),
            ("ghost\\ghost", 0),
            ("banshee\\banshee", 0),
            ("warthog_chaingun\\warthog_chaingun", 0),
            ("warthog_gauss\\warthog_gauss", 0),
            ("falcon\\falcon", 0),
            ("scorpion\\scorpion", 0),
            ("scorpion_anti_infantry\\scorpion_anti_infantry", 0),
            ("wraith_anti_infantry\\wraith_anti_infantry", 0),
            ("sabre\\sabre", 0),
        ] {
            let huds = huds_of(native, cache, tail, seat);
            let mut vv = rh::HudValues::default();
            vv.third_person = huds.third_person;
            let mut sp = SeatPaint::default();
            let _ = ctx.run(Default::default(), |ctx| {
                let painter = ctx.layer_painter(eframe::egui::LayerId::new(eframe::egui::Order::Foreground, eframe::egui::Id::new(tail)));
                sp = paint_seat_huds(ctx, &painter, &vs, native, cache, &huds, skin.as_ref(), &vv, true, true);
            });
            let label = format!("{tail} seat {seat}");

            // ---- the seat chud's own crosshair ----
            if let Some(t) = huds.seat {
                let want = authored(native, cache, t);
                if !want.is_empty() {
                    let r = sp.seat_crosshair.unwrap_or_else(|| panic!("{label}: the seat chud authors a crosshair and none was drawn"));
                    assert_eq!(sp.seat_crosshair_parts, want.len(), "{label}: drawn parts vs authored");
                    let mut all: Option<eframe::egui::Rect> = None;
                    for a in &want {
                        let rr = expected(&vs, a);
                        all = Some(all.map_or(rr, |u| u.union(rr)));
                        println!(
                            "| {tail} | {seat} | {} | {} ({:#x}, {}) blend {} | {:.1}x{:.1} | ({:.1},{:.1}) |",
                            name(native, cache, huds.seat), a.0, a.1, a.2, rh::blend_mode_name(a.6), rr.width(), rr.height(), rr.center().x, rr.center().y
                        );
                    }
                    let all = all.unwrap();
                    assert!(near(r, all), "{label}: seat reticule {r:?}, authored {all:?}");
                    seen_seat_crosshair += 1;
                }
            }
            // ---- the vehicle weapon's reticule ----
            if let Some(wh) = huds.vehicle_weapon_hud {
                let r = sp.crosshair.unwrap_or_else(|| panic!("{label}: the vehicle weapon authors a reticule and none was drawn"));
                let want = authored(native, cache, wh);
                assert!(sp.crosshair_parts >= 1 && sp.crosshair_parts <= want.len(), "{label}: {} parts of {} authored", sp.crosshair_parts, want.len());
                // Every drawn part is one the tag authors, so the union lies inside the union of
                // ALL authored rects, and the first part is centred where its widget says.
                let mut all: Option<eframe::egui::Rect> = None;
                for a in &want {
                    let rr = expected(&vs, a);
                    all = Some(all.map_or(rr, |u| u.union(rr)));
                }
                let all = all.unwrap().expand(0.6);
                assert!(all.contains_rect(r), "{label}: reticule {r:?} is outside the authored {all:?}");
                assert!(r.width() > 8.0 && r.height() > 8.0, "{label}: reticule is {:?}", r.size());
                seen_weapon_crosshair += 1;
                println!(
                    "| {tail} | {seat} | {} (weapon {}) | {} part(s) | {:.0}x{:.0} | ({:.0},{:.0}) |",
                    name(native, cache, Some(wh)),
                    name(native, cache, huds.vehicle_weapon),
                    sp.crosshair_parts, r.width(), r.height(), r.center().x, r.center().y
                );
            }
            println!("<!-- {label}: {} -->", sp.lines.join(" | "));
        }
        assert!(seen_seat_crosshair >= 4, "only {seen_seat_crosshair} seat-chud crosshair widgets were checked");
        assert!(seen_weapon_crosshair >= 5, "only {seen_weapon_crosshair} vehicle-weapon reticules were checked");
    }

    /// A seat that names no HUD and carries no gun paints NOTHING of its own -- exactly the
    /// engine's empty slots 4 and 6 -- and the on-foot reticule path stays the caller's.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn a_seat_with_no_hud_paints_nothing_of_its_own() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = eframe::egui::Context::default();
        let vp = eframe::egui::Rect::from_min_size(eframe::egui::pos2(0.0, 0.0), eframe::egui::vec2(1920.0, 1080.0));
        let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned());
        let vs = rh::VirtualScreen::from_skin(vp, skin.as_ref().unwrap());
        for huds in [huds_of(native, cache, "warthog\\warthog", 1), SeatHuds::on_foot()] {
            assert!(!huds.any_vehicle_hud());
            rh::set_widget_log(true);
            let mut sp = SeatPaint::default();
            let _ = ctx.run(Default::default(), |ctx| {
                let painter = ctx.layer_painter(eframe::egui::LayerId::new(eframe::egui::Order::Foreground, eframe::egui::Id::new("none")));
                sp = paint_seat_huds(ctx, &painter, &vs, native, cache, &huds, skin.as_ref(), &rh::HudValues::default(), true, true);
            });
            let log = rh::take_widget_log();
            rh::set_widget_log(false);
            assert!(log.is_empty() && sp.crosshair.is_none() && sp.seat_drawn == 0 && sp.lines.is_empty());
        }
    }
}
