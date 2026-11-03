//! #vehicle-weapons  What a vehicle FIRES, from the tags: `unit +0x42C weapons` and which seat uses them.
//!
//! A `vehi` is a `unit`; its `weapons` block (`unit_weapon_block`, max 4, 20 bytes: `weapon` tag
//! reference +0x00, `position` enum +0x10) names the `weap` tags the vehicle is built with, and the
//! unit's `right_hand_node` (+0x340) / `left_hand_node` (+0x344) name the vehicle-model markers the
//! primary / secondary weapon is attached to. Offsets: `reach_seats` module comment (the +0x98 shift
//! after `unit +0x348`) puts `weapons` at `tagdefs`' `+?394 + 0x98 = +0x42C`.
//!
//! Which SEAT fires them is not a field of its own: a seat flagged `gunner` (bit 3) or `gunner
//! (player only)` (bit 17) works the vehicle's weapons, and a driver seat that is not also a
//! gunner seat does not (it drives). The gate [`gates`] checks that rule against every vehicle.

use hms_native::{CacheHandle, NativeDll};

use crate::reach_actor::tag_block;
use crate::reach_seats::{Seat, UnitSeats};

pub const UNIT_WEAPONS: usize = 0x42C;
pub const UNIT_WEAPON_SIZE: usize = 20;
pub const UNIT_WEAPON_MAX: i32 = 4;
pub const UNIT_RIGHT_HAND_NODE: usize = 0x340;
pub const UNIT_LEFT_HAND_NODE: usize = 0x344;

/// `initial_weapon_position`: 0 primary, 1 secondary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VehicleWeapon {
    pub weapon: u32,
    pub position: i32,
}

#[derive(Clone, Debug, Default)]
pub struct VehicleWeapons {
    pub unit_tag: u32,
    pub weapons: Vec<VehicleWeapon>,
    /// Marker the PRIMARY weapon is attached to on the vehicle model (`right_hand_node`).
    pub primary_marker: String,
    pub secondary_marker: String,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

impl VehicleWeapons {
    pub fn read(native: &NativeDll, cache: CacheHandle, unit_tag: u32) -> VehicleWeapons {
        let b = tag_block(native, cache, unit_tag, UNIT_WEAPONS, UNIT_WEAPON_SIZE, UNIT_WEAPON_MAX);
        let weapons = b
            .chunks_exact(UNIT_WEAPON_SIZE)
            .filter_map(|e| {
                let t = crate::reach_projectile::tag_ref(e, 0)?;
                Some(VehicleWeapon { weapon: t, position: rd_u32(e, 0x10) as i32 })
            })
            .collect();
        let head = native.tag_meta(cache, unit_tag, 0, 0x348).unwrap_or_default();
        let name = |o: usize| {
            let sid = rd_u32(&head, o);
            if sid == 0 || sid == u32::MAX { String::new() } else { native.string_id(cache, sid).unwrap_or_default() }
        };
        VehicleWeapons { unit_tag, weapons, primary_marker: name(UNIT_RIGHT_HAND_NODE), secondary_marker: name(UNIT_LEFT_HAND_NODE) }
    }

    /// Does the occupant of `seat` work this vehicle's weapons?
    pub fn seat_fires(seat: &Seat) -> bool {
        seat.is_gunner()
    }

    /// The weapon in position `position` (0 primary, 1 secondary), if the vehicle has one.
    pub fn at(&self, position: i32) -> Option<VehicleWeapon> {
        self.weapons.iter().copied().find(|w| w.position == position)
    }
}

/// What a seat works: the `weap` tag, the unit that carries it and the marker it sits on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatWeapon {
    pub weapon: u32,
    pub owner: u32,
    pub marker: String,
}

/// The weapon the occupant of seat `seat` of `vehi_tag` fires, or None when that seat drives only
/// (or the vehicle has nothing to fire). The vehicle's own primary weapon (position 0, else the
/// first); a gunner seat whose vehicle carries none fires the weapon of the seatless CHILD vehicle
/// attached to the placement (`wraith_mortar`, `scorpion_cannon`, `falcon_chin_gun`, ...): the
/// shipped tags put those weapons on a child object, not on the hull.
pub fn weapon_for_seat(native: &NativeDll, cache: CacheHandle, vehi_tag: u32, placement_tag: u32, seat: usize) -> Option<SeatWeapon> {
    let seats = UnitSeats::read(native, cache, vehi_tag)?;
    if !seats.seats.get(seat).map_or(false, VehicleWeapons::seat_fires) {
        return None;
    }
    let own = VehicleWeapons::read(native, cache, vehi_tag);
    if let Some(w) = own.at(0).or_else(|| own.weapons.first().copied()) {
        return Some(SeatWeapon { weapon: w.weapon, owner: vehi_tag, marker: own.primary_marker });
    }
    for a in native.enumerate_attachments(cache, placement_tag) {
        let child = VehicleWeapons::read(native, cache, a.child_obj);
        let seatless = UnitSeats::read(native, cache, a.child_obj).map_or(true, |u| u.seats.is_empty());
        if seatless {
            if let Some(w) = child.at(0).or_else(|| child.weapons.first().copied()) {
                return Some(SeatWeapon { weapon: w.weapon, owner: a.child_obj, marker: child.primary_marker });
            }
        }
    }
    None
}

#[cfg(test)]
mod gates {
    use super::*;

    /// Every vehicle: its weapons and its seats, the way the tags say.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_vehicle_weapons() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let nm = |t: u32| native.tag_name(cache, t).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string();
        for v in native.tags_by_group(cache, "vehi") {
            let vw = VehicleWeapons::read(native, cache, v);
            let seats = UnitSeats::read(native, cache, v).map(|u| u.seats).unwrap_or_default();
            let ws: Vec<String> = vw.weapons.iter().map(|w| format!("{}@{}", nm(w.weapon), w.position)).collect();
            let ss: Vec<String> = seats.iter().map(|s| format!("{}[{}{}{}]", s.label, if s.is_driver() { "D" } else { "" }, if s.is_gunner() { "G" } else { "" }, if s.allows_weapons() { "w" } else { "" })).collect();
            println!("{:<30} weapons {:?} markers '{}'/'{}' seats {:?}", nm(v), ws, vw.primary_marker, vw.secondary_marker, ss);
        }
        for w in native.tags_by_group(cache, "weap") {
            let n = native.tag_name(cache, w).unwrap_or_default();
            if n.to_lowercase().contains("horn") {
                println!("HORN weap: {n}");
            }
        }
        for g in ["snd!", "lsnd", "effe", "vehi", "scen", "jmad", "foot", "ligh"] {
            for t in native.tags_by_group(cache, g) {
                let n = native.tag_name(cache, t).unwrap_or_default();
                if n.to_lowercase().contains("horn") {
                    println!("HORN {g}: {n}");
                }
            }
        }
    }

    /// The four horn weapons, and the three shapes any vehicle weapon is made of: how they are triggered,
    /// what they fire, and what their sound attachments are scaled by.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_horn_weapons() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        for leaf in ["warthog_horn", "mongoose_horn", "pickup_horn", "truck_cab_horn", "warthog_chaingun", "ghost_dual_cannon", "banshee_bomb_launcher", "scorpion_cannon", "wraith_mortar", "falcon_machinegun"] {
            let Some(w) = native.tags_by_group(cache, "weap").into_iter().find(|&t| native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))) else { println!("{leaf}: not in the cache"); continue };
            let def = crate::reach_weapon::WeaponDef::read(native, cache, w);
            println!("== {leaf}: {}", def.as_ref().map_or("unreadable".to_string(), |d| crate::play_fire::describe(d)));
            if let Some(d) = &def {
                for (i, t) in d.triggers.iter().enumerate() {
                    println!("   trigger {i}: {:?}", t);
                }
            }
            for f in crate::play_sound::ObjFn::read_all(native, cache, w) {
                println!("   fn import '{}' export '{}' off_with '{}' scale_by '{}' flags {:#x} min {}", f.import, f.export, f.off_with, f.scale_by, f.flags, f.min_value);
            }
            let at = tag_block_attachments(native, cache, w);
            for a in at {
                println!("   attach {a}");
            }
            for b in crate::reach_projectile::weapon_barrels(native, cache, w) {
                println!("   barrel: proj {:?} fx {:?} rps {:?}", b.projectile.map(|p| native.tag_name(cache, p)), b.firing_effect.map(|p| native.tag_name(cache, p)), b.rounds_per_second);
            }
        }
    }

    fn tag_block_attachments(native: &NativeDll, cache: CacheHandle, tag: u32) -> Vec<String> {
        let b = tag_block(native, cache, tag, crate::reach_projectile::OBJE_ATTACHMENTS, crate::reach_projectile::OBJE_ATTACHMENT_SIZE, 24);
        b.chunks_exact(32)
            .map(|e| {
                let cls: String = e[0..4].iter().rev().map(|&c| c as char).collect();
                let t = crate::reach_projectile::tag_ref(e, 0).and_then(|t| native.tag_name(cache, t)).unwrap_or_default();
                let sid = |o: usize| { let v = rd_u32(e, o); if v == 0 || v == u32::MAX { String::new() } else { native.string_id(cache, v).unwrap_or_default() } };
                format!("{cls} {t} marker '{}' scale '{}'", sid(0x10), sid(0x18))
            })
            .collect()
    }

    fn named(native: &NativeDll, cache: CacheHandle, group: &str, leaf: &str) -> u32 {
        native
            .tags_by_group(cache, group)
            .into_iter()
            .find(|&t| native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf)))
            .unwrap_or_else(|| panic!("no {group} {leaf}"))
    }

    /// Which weapon each family's driver / gunner works, from the real tags. LITERAL expectations
    /// (read off the probe above); a seat that only rides must fire nothing.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn each_seat_works_its_authored_weapon() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let leaf = |t: u32| native.tag_name(cache, t).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string();
        // (vehicle, seat index, weapon leaf it must fire)
        let want = [
            ("ghost", 0usize, "ghost_dual_cannon"),
            ("warthog", 0, "warthog_horn"),
            ("mongoose", 0, "mongoose_horn"),
            ("banshee", 0, "banshee_dual_cannon"),
        ];
        let mut checked = 0;
        for (v, seat, w) in want {
            let vt = native
                .tags_by_group(cache, "vehi")
                .into_iter()
                .find(|&t| leaf(t) == v)
                .unwrap_or_else(|| panic!("no vehicle {v}"));
            let got = weapon_for_seat(native, cache, vt, vt, seat).unwrap_or_else(|| panic!("{v} seat {seat} fires nothing"));
            assert_eq!(leaf(got.weapon), w, "{v} seat {seat}");
            checked += 1;
        }
        assert_eq!(checked, 4);
        // a seat that is not a gunner's fires nothing: the warthog's passenger seat
        let warthog = native.tags_by_group(cache, "vehi").into_iter().find(|&t| leaf(t) == "warthog").unwrap();
        let seats = UnitSeats::read(native, cache, warthog).unwrap().seats;
        let passenger = seats.iter().position(|s| !s.is_gunner()).expect("the warthog has a seat that only rides");
        assert!(weapon_for_seat(native, cache, warthog, warthog, passenger).is_none(), "a passenger seat fires nothing");
        // a seatless child turret's weapon is reached through the placement (the wraith's mortar)
        let wraith = native.tags_by_group(cache, "vehi").into_iter().find(|&t| leaf(t) == "wraith").unwrap();
        let mortar = weapon_for_seat(native, cache, wraith, wraith, 0);
        eprintln!("wraith driver seat -> {:?}", mortar.as_ref().map(|m| leaf(m.weapon)));
    }

    /// The ghost cannon's machine, held for a second, fires the authored bolt; released, nothing.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_held_trigger_fires_the_vehicles_authored_projectile() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let weap = named(native, cache, "weap", "ghost_dual_cannon");
        let def = crate::reach_weapon::WeaponDef::read(native, cache, weap).expect("readable");
        let bolt = named(native, cache, "proj", "ghost_dual_cannon_bolt");
        let mut wf = crate::play_fire::WeaponFire::new(std::sync::Arc::new(def), vec![0; 4]);
        let input = |held: bool| crate::play_fire::FireInput { held, alt_held: false, usable: true, reloading: false, has_reserve: true, crouched: false, melee: false };
        let mut shots = 0;
        for _ in 0..60 {
            for ev in wf.advance(1.0 / 60.0, &input(true)) {
                if let crate::play_fire::FireEvent::Shot { barrel, .. } = ev {
                    assert_eq!(wf.def.barrels[barrel].projectile, Some(bolt), "the barrel fires the authored bolt");
                    shots += 1;
                }
            }
        }
        assert!((6..=8).contains(&shots), "7.5 rps for one second is 7-8 shots: {shots}");
        // the round already in the barrel's recovery may still leave; after that, nothing
        for _ in 0..30 {
            wf.advance(1.0 / 60.0, &input(false));
        }
        let mut after = 0;
        for _ in 0..60 {
            after += wf.advance(1.0 / 60.0, &input(false)).iter().filter(|e| matches!(e, crate::play_fire::FireEvent::Shot { .. })).count();
        }
        assert_eq!(after, 0, "released: no shots");
    }

    /// #seat-weapons  Which seats allow the rider's own weapon, per family, LITERAL from the real tags
    /// (engine rule: `unit_can_use_weapon_in_seat` tests seat flag bit 5; driver and gunner seats
    /// author it clear, passenger seats set). Also: a recorded rider in a no-weapon seat holds
    /// nothing in Theater, and in a weapon seat keeps what the recording says.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn seats_allow_the_riders_weapon_exactly_where_the_engine_does() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let leaf = |t: u32| native.tag_name(cache, t).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string();
        // (vehicle, seat label, allows weapons)
        let table: [(&str, &str, bool); 28] = [
            ("warthog", "warthog_d", false),
            ("warthog", "warthog_p", true),
            ("warthog", "warthog_b_d", false),
            ("warthog_chaingun", "warthog_g", false),
            ("warthog_troop", "warthog_p_l", true),
            ("mongoose", "mongoose_d", false),
            ("mongoose", "mongoose_p", true),
            ("ghost", "ghost_d", false),
            ("banshee", "banshee_d", false),
            ("falcon", "falcon_d", false),
            ("falcon", "falcon_p_l1", false),
            ("falcon", "falcon_p_l1_wpn", true),
            ("falcon", "falcon_p_bench_wpn", true),
            ("scorpion", "scorpion_d", false),
            ("scorpion", "scorpion_p_lf", true),
            ("scorpion_anti_infantry", "scorpion_g", false),
            ("wraith", "wraith_d", false),
            ("wraith", "wraith_b_l", false),
            ("wraith_anti_infantry", "wraith_g", false),
            ("revenant", "revenant_d", false),
            ("revenant", "revenant_p", true),
            ("pelican", "pelican_d", false),
            ("pelican", "pelican_p_l01", true),
            ("pelican", "pelican_g", false),
            ("pelican", "pelican_e", true),
            ("sabre", "sabre_p", false),
            ("truck_cab_large", "truck_p", false),
            ("phantom_mp", "phantom_pr_lf", true),
        ];
        let mut seen = 0;
        for (v, label, allows) in table {
            let Some(vt) = native.tags_by_group(cache, "vehi").into_iter().find(|&t| leaf(t) == v) else { continue };
            let seats = UnitSeats::read(native, cache, vt).unwrap().seats;
            let (i, s) = seats.iter().enumerate().find(|(_, s)| s.label == label).unwrap_or_else(|| panic!("{v} has no seat {label}"));
            assert_eq!(s.allows_weapons(), allows, "{v} seat {label}");
            // Theater: the held weapon the recording names survives only where the seat allows it
            let mut memo = std::collections::HashMap::new();
            let held = crate::film_play::weapon_in_seat(&mut memo, native, cache, Some(vt), Some(i), Some(0x1234));
            assert_eq!(held, if allows { Some(0x1234) } else { None }, "{v} seat {label}: rider's weapon");
            seen += 1;
        }
        assert!(seen >= 20, "liveness: only {seen} of the table's seats exist in forge_halo");
        // a ride whose seat the recording does not state keeps its weapon
        let mut memo = std::collections::HashMap::new();
        assert_eq!(crate::film_play::weapon_in_seat(&mut memo, native, cache, None, None, Some(7)), Some(7));
    }
}
