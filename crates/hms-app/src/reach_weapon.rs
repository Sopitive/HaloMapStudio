//! #weapon-fire  The Reach `weap` FIRING MODEL: what the trigger, the barrel and the magazine
//! author, and the engine's own per-tick state machine that turns them into shots.
//!
//! # Where this comes from
//!
//! `reach_tag_test.exe` (the MCC TOOL build, symbols kept for the asserts) and the field sets it
//! links. Everything below is read from decompiled engine code unless a line says INFERRED:
//!
//! * `weapon_barrel_definition_fields` (the CURRENT field set, `.data` +0x214AAA0, 0x184 bytes) —
//!   its nested structs (`firing` at +0x04, `firing error` at +0x50, `accuracy penalties` at +0x7C)
//!   walked recursively, because `reverseme tagdefs` prints the whole block as `?` (the runtime-
//!   installed field-set trap; see `reach-runtime-installed-field-sets`). The one `eh` struct
//!   (+0x124) has a NULL field pointer and is a 16-byte hole fixed by the offsets on both sides.
//! * `sub_140DF7640` — the weapon object's per-tick update: magazines, then triggers, then barrels.
//! * `sub_140DF64E0` — one TRIGGER's tick: reads the input bit, applies the `behavior`.
//! * `sub_140DEAB40` — one BARREL's tick: the state machine (idle / firing / locked recovery /
//!   soft recovery), the rate-of-fire cycle, the heat and error decays.
//! * `sub_140DE37A0` — "is this barrel allowed to fire now" (shots-per-fire window + RoF interval).
//! * `sub_140DE86B0` — one fire: rounds, heat, age, the projectile request, the empty click.
//! * `sub_140DE3F80` — the recovery length, `sub_140DE4120` — the rate-of-fire interval.
//! * `weapon_magazine_can_fire_weapon` — the ammunition gate.
//!
//! # The engine's clock
//!
//! The simulation ticks at 60 Hz in MCC (`game_time_globals + 6`, read by `sub_1400BD300`), but
//! the authored times are in the 30 Hz grid: `sub_1400BD120(s) = s * 30`, and
//! `sub_1400BD360() = tick_rate / 30 = 2`. Every constant below that looks odd (`2 *`, `+ 2`)
//! is that conversion, not a fudge. See [`recovery_ticks`] and [`rate_interval_ticks`].

use hms_native::{CacheHandle, NativeDll};

use crate::reach_actor::{block_at, tag_block};

// ---------------------------------------------------------------------------------------------
// layouts
// ---------------------------------------------------------------------------------------------

pub const WEAP_FLAGS: usize = 0x204;
pub const WEAP_SECONDARY_FLAGS: usize = 0x208;
pub const WEAP_READY_TIME: usize = 0x214;
pub const WEAP_HEAT_RECOVERY_THRESHOLD: usize = 0x238;
pub const WEAP_OVERHEATED_THRESHOLD: usize = 0x23C;
pub const WEAP_HEAT_DETONATION_THRESHOLD: usize = 0x240;
pub const WEAP_HEAT_DETONATION_FRACTION: usize = 0x244;
pub const WEAP_HEAT_LOSS_PER_SECOND: usize = 0x248;
pub const WEAP_HEAT_LOSS_FN: usize = 0x24C;
pub const WEAP_HEAT_VENTING_TIME: usize = 0x254;
pub const WEAP_HEAT_VENTING_EXIT_HEAT: usize = 0x258;
pub const WEAP_OVERHEATED_HEAT_LOSS_FN: usize = 0x268;
pub const WEAP_OVERHEATED_HEAT_LOSS_PER_SECOND: usize = 0x264;
pub const WEAP_TRIGGERS: usize = 0x3FC;
pub const WEAP_BARRELS: usize = 0x408;
pub const WEAP_WEAPON_TYPE: usize = 0x3B4;
pub const WEAP_READY_FOR_USE_TIME: usize = 0x4A4;
/// `weapon_types` index of the Spartan laser (the one the heat code special-cases).
pub const WEAPON_TYPE_SPLASER: i16 = 7;

pub const TRIGGER_SIZE: usize = 0x8C;
/// `charging` struct (+0x1C) `fire fractions` block header.
pub const TRIGGER_FIRE_FRACTIONS: usize = 0x80;
pub const TRIGGER_MAX: i32 = 2;
pub const BARREL_SIZE: usize = 0x184;
pub const BARREL_MAX: i32 = 2;

// weap flags
pub const WF_CANNOT_FIRE_AT_MAX_AGE: u32 = 1 << 6;
/// `weapon ages with each kill` -- the age comes from kills, not rounds.
pub const WF_AGES_WITH_EACH_KILL: u32 = 1 << 20;
pub const WF_MELEE_ONLY: u32 = 1 << 22;
pub const WF_CANNOT_FIRE_WHILE_BOOSTING: u32 = 1 << 13;
// weap secondary flags
pub const WSF_NEVER_OVERHEATS: u32 = 1 << 6;

// barrel flags
pub const BF_CAN_FIRE_WITH_PARTIAL_AMMO: u32 = 1 << 1;
pub const BF_EJECTS_DURING_CHAMBER: u32 = 1 << 3;
pub const BF_CANT_FIRE_WHEN_OTHERS_FIRING: u32 = 1 << 7;
pub const BF_CANT_FIRE_WHEN_OTHERS_RECOVERING: u32 = 1 << 8;
pub const BF_DONT_CLEAR_FIRE_BIT_AFTER_RECOVERING: u32 = 1 << 9;
/// `can fire at maximum age`.
pub const BF_CAN_FIRE_AT_MAX_AGE: u32 = 1 << 11;
pub const BF_USE_ONE_FIRING_EFFECT_PER_BURST: u32 = 1 << 12;

/// `weapon_trigger_behaviors`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behavior {
    Spew,
    Latch,
    LatchAutofire,
    LatchTether,
    Charge,
    LatchZoom,
    SpewCharge,
    SwordCharge,
}

impl Behavior {
    fn from_i16(v: i16) -> Option<Behavior> {
        Some(match v {
            0 => Behavior::Spew,
            1 => Behavior::Latch,
            2 => Behavior::LatchAutofire,
            3 => Behavior::LatchTether,
            4 => Behavior::Charge,
            5 => Behavior::LatchZoom,
            6 => Behavior::SpewCharge,
            7 => Behavior::SwordCharge,
            _ => return None,
        })
    }

    pub fn word(self) -> &'static str {
        match self {
            Behavior::Spew => "spew",
            Behavior::Latch => "latch",
            Behavior::LatchAutofire => "latch-autofire",
            Behavior::LatchTether => "latch-tether",
            Behavior::Charge => "charge",
            Behavior::LatchZoom => "latch-zoom",
            Behavior::SpewCharge => "spew-charge",
            Behavior::SwordCharge => "sword-charge",
        }
    }
}

/// One `weapon_triggers` element (140 bytes, `weap +0x3FC`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriggerDef {
    pub flags: u32,
    /// `weapon_trigger_inputs`: 0 right trigger, 1 left trigger, 2 melee (the three `case`s of
    /// `sub_140DF64E0`).
    pub input: i16,
    pub behavior: Option<Behavior>,
    /// `primary barrel` / `secondary barrel` block indices (0xFFFF = none).
    pub primary_barrel: Option<usize>,
    pub secondary_barrel: Option<usize>,
    pub autofire_time: f32,
    pub autofire_throw: f32,
    pub charging_time: f32,
    pub charged_time: f32,
    pub overcharged_action: i8,
    pub charging_flags: u8,
    /// `cancelled trigger throw` -- the analog depth a release must exceed to count as a pull.
    pub cancelled_trigger_throw: i16,
    /// `autofire` `secondary action` / `primary action` (`weapon_trigger_autofire_actions`:
    /// 0 fire, 1 charge, 2 fire other).
    pub autofire_secondary_action: i16,
    pub autofire_primary_action: i16,
    /// `fire fractions`: the charge fractions at which a discharge adds a round.
    pub fire_fractions: Vec<f32>,
}

/// One `weapon_barrels` element (388 bytes, `weap +0x408`), with every field that gates a shot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BarrelDef {
    pub flags: u32,
    /// `rounds per second` [initial, final]; the cyclic rate ramps between them.
    pub rounds_per_second: [f32; 2],
    /// Seconds of continuous fire to reach the final rate / idle seconds to fall back.
    pub acceleration_time: f32,
    pub deceleration_time: f32,
    /// `shots per fire` (min, max): a burst. (0, 0) = no authored window.
    pub shots_per_fire: [i16; 2],
    /// `fire recovery time:seconds` and the `soft recovery fraction` of it that may queue a shot.
    pub fire_recovery_time: f32,
    pub soft_recovery_fraction: f32,
    pub magazine: Option<usize>,
    pub rounds_per_shot: i16,
    pub minimum_rounds_loaded: i16,
    pub rounds_between_tracers: i16,
    pub marker_sid: u32,
    pub firing_noise: i16,
    /// `distribution function` (0 point, 1 horizontal fan), `projectiles per shot`,
    /// `distribution angle` radians.
    pub distribution_function: i16,
    pub projectiles_per_shot: i16,
    pub distribution_angle: f32,
    /// `minimum error` and `error angle` bounds, radians.
    pub minimum_error: f32,
    pub error_angle: [f32; 2],
    /// `deceleration time` of the `firing error` struct, seconds.
    pub error_deceleration_time: f32,
    pub projectile: Option<u32>,
    pub heat_per_round: f32,
    pub age_per_round: f32,
    pub overload_time: f32,
    /// The BAKED runtime rates the compiler derives from the authored times (`+0x16C` rate of fire
    /// acceleration, `+0x170` deceleration, `+0x174` error deceleration), per second. A time of 0
    /// bakes to 30 (one 30 Hz step); otherwise `1 / time` (needler 0.8 s -> 1.25).
    pub acceleration_rate: f32,
    pub deceleration_rate: f32,
    pub error_deceleration_rate: f32,
    /// `firing penalty function` (single wield) and its crouched variant: the `mapping_function`
    /// blobs `sub_140DE2A60` evaluates at the current error to get the error one round adds.
    /// Sampled at error = 0, 0.1 ... 1.0 (the engine evaluates the function; eleven samples and a
    /// linear interpolation are enough for a curve the tags author with 2-3 knots).
    pub firing_penalty: Option<[f32; 11]>,
    pub firing_crouched_penalty: Option<[f32; 11]>,
    /// The `rate of fire acceleration` / `deceleration` object FUNCTIONS (`+0x0C` / `+0x14` string
    /// ids), sampled over `heat`. When one is authored the cyclic rate is not a ramp over a time
    /// but `f(heat)`: the plasma repeater's `heat_driven_rate_of_fire` slows it from 15 to 3 rps as
    /// it heats up. `heat generated per round` can be a function the same way.
    pub rate_of_fire_acceleration_fn: Option<[f32; 11]>,
    pub rate_of_fire_deceleration_fn: Option<[f32; 11]>,
    pub heat_per_round_fn: Option<[f32; 11]>,
}

/// One `magazines` element's fields that gate a shot (the rest lives in `play_hud::Magazine`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeaponDef {
    pub weap: u32,
    pub flags: u32,
    pub secondary_flags: u32,
    pub weapon_type: i16,
    pub ready_time: f32,
    pub heat_recovery_threshold: f32,
    pub overheated_threshold: f32,
    pub heat_detonation_threshold: f32,
    pub heat_loss_per_second: f32,
    pub overheated_heat_loss_per_second: f32,
    /// `heat loss` / `overheated heat loss` object functions over `heat` (the repeater's
    /// `heat_loss` is 0.25 /s cold and 0.06 /s hot).
    pub heat_loss_fn: Option<[f32; 11]>,
    pub overheated_heat_loss_fn: Option<[f32; 11]>,
    /// `heat venting time` and `heat venting exit heat`: how long a weapon that cannot cool by
    /// itself (the repeater: recovery threshold 0) vents, and the heat it ends at.
    pub heat_venting_time: f32,
    pub heat_venting_exit_heat: f32,
    pub triggers: Vec<TriggerDef>,
    pub barrels: Vec<BarrelDef>,
}

fn rd_f32(b: &[u8], o: usize) -> f32 {
    match b.get(o..o + 4) {
        Some(s) => {
            let v = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            if v.is_finite() { v } else { 0.0 }
        }
        None => 0.0,
    }
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_i16(b: &[u8], o: usize) -> i16 {
    b.get(o..o + 2).map_or(0, |s| i16::from_le_bytes([s[0], s[1]]))
}

fn block_index(v: i16) -> Option<usize> {
    (v >= 0).then_some(v as usize)
}

/// The weapon's own `functions` block (`obje +0xEC`, 64-byte elements), reduced to what the firing
/// model reads: the functions whose IMPORT is `heat`, sampled at eleven points, by EXPORT name.
/// A function over any other import (`age`, `is_venting`, the animation drivers) is left out --
/// nothing in the firing model feeds them -- so a weapon that names one falls back to its plain
/// float rather than evaluating a wrong input.
#[derive(Clone, Debug, Default)]
pub struct ObjectFns {
    over_heat: Vec<(u32, [f32; 11])>,
}

impl ObjectFns {
    pub fn read(native: &NativeDll, cache: CacheHandle, weap: u32) -> ObjectFns {
        let b = tag_block(native, cache, weap, 0xEC, 0x40, 256);
        let mut over_heat = Vec::new();
        for e in b.chunks_exact(0x40) {
            let import = rd_u32(e, 4);
            if import == 0 || native.string_id(cache, import).as_deref() != Some("heat") {
                continue;
            }
            let (size, addr) = (rd_u32(e, 0x18), rd_u32(e, 0x24));
            if !(0x0C..=0x400).contains(&size) || addr == 0 || addr == u32::MAX {
                continue;
            }
            let Some(blob) = native.tag_ptr_bytes(cache, addr, size) else { continue };
            let Some(f) = crate::scene::MapFn::decode(&blob) else { continue };
            let mut curve = [0.0f32; 11];
            for (i, c) in curve.iter_mut().enumerate() {
                *c = f.eval(i as f32 / 10.0);
            }
            over_heat.push((rd_u32(e, 8), curve));
        }
        ObjectFns { over_heat }
    }

    /// The `heat -> value` curve exported under `export_sid`, if the weapon has one.
    pub fn over_heat(&self, export_sid: u32) -> Option<[f32; 11]> {
        if export_sid == 0 {
            return None;
        }
        self.over_heat.iter().find(|(s, _)| *s == export_sid).map(|(_, c)| *c)
    }
}

impl TriggerDef {
    pub fn decode(native: &NativeDll, cache: CacheHandle, e: &[u8]) -> TriggerDef {
        let fractions = block_at(native, cache, e, TRIGGER_FIRE_FRACTIONS, 4, 16);
        TriggerDef {
            flags: rd_u32(e, 0x00),
            input: rd_i16(e, 0x04),
            behavior: Behavior::from_i16(rd_i16(e, 0x06)),
            primary_barrel: block_index(rd_i16(e, 0x08)),
            secondary_barrel: block_index(rd_i16(e, 0x0A)),
            autofire_time: rd_f32(e, 0x10),
            autofire_throw: rd_f32(e, 0x14),
            charging_time: rd_f32(e, 0x1C),
            charged_time: rd_f32(e, 0x20),
            overcharged_action: e.get(0x24).map_or(0, |&b| b as i8),
            charging_flags: e.get(0x25).copied().unwrap_or(0),
            cancelled_trigger_throw: rd_i16(e, 0x26),
            autofire_secondary_action: rd_i16(e, 0x18),
            autofire_primary_action: rd_i16(e, 0x1A),
            fire_fractions: fractions.chunks_exact(4).map(|c| rd_f32(c, 0)).collect(),
        }
    }
}

impl BarrelDef {
    pub fn decode(native: &NativeDll, cache: CacheHandle, e: &[u8], fns: &ObjectFns) -> BarrelDef {
        let function = |at: usize| -> Option<[f32; 11]> {
            let el = block_at(native, cache, e, at, 0x14, 1);
            // The element is a `data` field: size at +0, address at +0x0C.
            let (size, addr) = (rd_u32(&el, 0), rd_u32(&el, 0x0C));
            let blob = (size >= 0x0C && size <= 0x400 && addr != 0 && addr != u32::MAX)
                .then(|| native.tag_ptr_bytes(cache, addr, size))
                .flatten()?;
            let f = crate::scene::MapFn::decode(&blob)?;
            let mut curve = [0.0f32; 11];
            for (i, c) in curve.iter_mut().enumerate() {
                *c = f.eval(i as f32 / 10.0);
            }
            Some(curve)
        };
        BarrelDef {
            flags: rd_u32(e, 0x00),
            rounds_per_second: [rd_f32(e, 0x04), rd_f32(e, 0x08)],
            acceleration_time: rd_f32(e, 0x10),
            deceleration_time: rd_f32(e, 0x18),
            shots_per_fire: [rd_i16(e, 0x24), rd_i16(e, 0x26)],
            fire_recovery_time: rd_f32(e, 0x28),
            soft_recovery_fraction: rd_f32(e, 0x2C),
            magazine: block_index(rd_i16(e, 0x38)),
            rounds_per_shot: rd_i16(e, 0x3A),
            minimum_rounds_loaded: rd_i16(e, 0x3C),
            rounds_between_tracers: rd_i16(e, 0x3E),
            marker_sid: rd_u32(e, 0x40),
            firing_noise: rd_i16(e, 0x44),
            distribution_function: rd_i16(e, 0x68),
            projectiles_per_shot: rd_i16(e, 0x6A),
            distribution_angle: rd_f32(e, 0x6C),
            minimum_error: rd_f32(e, 0x70),
            error_angle: [rd_f32(e, 0x74), rd_f32(e, 0x78)],
            error_deceleration_time: rd_f32(e, 0x50),
            projectile: crate::reach_projectile::tag_ref(e, 0x104),
            heat_per_round: rd_f32(e, 0x150),
            age_per_round: rd_f32(e, 0x158),
            overload_time: rd_f32(e, 0x160),
            acceleration_rate: rd_f32(e, 0x16C),
            deceleration_rate: rd_f32(e, 0x170),
            error_deceleration_rate: rd_f32(e, 0x174),
            firing_penalty: function(0x8C),
            firing_crouched_penalty: function(0x98),
            rate_of_fire_acceleration_fn: fns.over_heat(rd_u32(e, 0x0C)),
            rate_of_fire_deceleration_fn: fns.over_heat(rd_u32(e, 0x14)),
            heat_per_round_fn: fns.over_heat(rd_u32(e, 0x154)),
        }
    }
}

impl WeaponDef {
    pub fn read(native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<WeaponDef> {
        let head = native.tag_meta(cache, weap, 0x200, 0x70)?;
        let at = |o: usize| o - 0x200;
        let triggers = tag_block(native, cache, weap, WEAP_TRIGGERS, TRIGGER_SIZE, TRIGGER_MAX);
        let barrels = tag_block(native, cache, weap, WEAP_BARRELS, BARREL_SIZE, BARREL_MAX);
        let fns = ObjectFns::read(native, cache, weap);
        Some(WeaponDef {
            weap,
            flags: rd_u32(&head, at(WEAP_FLAGS)),
            secondary_flags: rd_u32(&head, at(WEAP_SECONDARY_FLAGS)),
            weapon_type: native.tag_meta(cache, weap, WEAP_WEAPON_TYPE as u32, 2).map_or(0, |b| rd_i16(&b, 0)),
            ready_time: rd_f32(&head, at(WEAP_READY_TIME)),
            heat_recovery_threshold: rd_f32(&head, at(WEAP_HEAT_RECOVERY_THRESHOLD)),
            overheated_threshold: rd_f32(&head, at(WEAP_OVERHEATED_THRESHOLD)),
            heat_detonation_threshold: rd_f32(&head, at(WEAP_HEAT_DETONATION_THRESHOLD)),
            heat_loss_per_second: rd_f32(&head, at(WEAP_HEAT_LOSS_PER_SECOND)),
            overheated_heat_loss_per_second: rd_f32(&head, at(WEAP_OVERHEATED_HEAT_LOSS_PER_SECOND)),
            heat_loss_fn: fns.over_heat(rd_u32(&head, at(WEAP_HEAT_LOSS_FN))),
            overheated_heat_loss_fn: fns.over_heat(rd_u32(&head, at(WEAP_OVERHEATED_HEAT_LOSS_FN))),
            heat_venting_time: rd_f32(&head, at(WEAP_HEAT_VENTING_TIME)),
            heat_venting_exit_heat: rd_f32(&head, at(WEAP_HEAT_VENTING_EXIT_HEAT)),
            triggers: triggers.chunks_exact(TRIGGER_SIZE).map(|e| TriggerDef::decode(native, cache, e)).collect(),
            barrels: barrels.chunks_exact(BARREL_SIZE).map(|e| BarrelDef::decode(native, cache, e, &fns)).collect(),
        })
    }
}

#[cfg(test)]
mod gates {
    //! #weapon-fire  **The layout gates.** `weapon_barrels` is a runtime-installed field set
    //! `reverseme tagdefs` prints as `?`, so every offset above was recovered from the field
    //! arrays in `reach_tag_test.exe` and the decompiled users of the block. These read the real
    //! `forge_halo` weapons at those offsets and compare with values the DESIGNERS wrote down:
    //! a rate of fire is a round number, a recovery time a multiple of a 30 Hz tick, a magazine a
    //! known size. A wrong offset reads garbage, and garbage is not 10.0.
    use super::*;

    fn weapons() -> Option<(std::sync::MutexGuard<'static, ()>, Vec<(String, WeaponDef)>)> {
        let (g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let mut tags = native.tags_by_group(cache, "weap");
        tags.sort_unstable();
        let mut out = Vec::new();
        for t in tags {
            let name = native.tag_name(cache, t).unwrap_or_default();
            if let Some(d) = WeaponDef::read(&native, cache, t) {
                out.push((name.rsplit('\\').next().unwrap_or(&name).to_string(), d));
            }
        }
        Some((g, out))
    }

    fn get<'a>(all: &'a [(String, WeaponDef)], leaf: &str) -> &'a WeaponDef {
        &all.iter().find(|w| w.0 == leaf).unwrap_or_else(|| panic!("{leaf} missing")).1
    }

    /// Values read off the shipped tags by eye in the tag editor's terms.
    #[test]
    fn the_barrel_and_trigger_layout_reads_the_designers_numbers() {
        let Some((_g, all)) = weapons() else { return eprintln!("reach_weapon: forge_halo not available") };
        let near = |a: f32, b: f32| (a - b).abs() < 1.0e-4;
        let ar = get(&all, "assault_rifle");
        assert_eq!(ar.triggers[0].behavior, Some(Behavior::Spew));
        assert_eq!(ar.triggers[0].primary_barrel, Some(0));
        let b = &ar.barrels[0];
        assert!(near(b.rounds_per_second[0], 10.0) && near(b.rounds_per_second[1], 10.0));
        assert!(near(b.fire_recovery_time, 0.05) && near(b.soft_recovery_fraction, 0.0));
        assert_eq!((b.magazine, b.rounds_per_shot, b.projectiles_per_shot, b.shots_per_fire), (Some(0), 1, 1, [0, 0]));
        assert!(b.projectile.is_some());
        assert!(b.error_angle[0] < b.error_angle[1] && b.error_angle[1] > 0.01, "the rifle's error widens: {:?}", b.error_angle);
        assert!(b.firing_penalty.is_some(), "the rifle authors a firing-penalty curve");

        let dmr = get(&all, "dmr");
        assert_eq!(dmr.triggers[0].behavior, Some(Behavior::Latch));
        let b = &dmr.barrels[0];
        assert!(near(b.rounds_per_second[0], 3.0) && near(b.fire_recovery_time, 0.33) && near(b.soft_recovery_fraction, 0.2));
        assert_eq!(b.shots_per_fire, [1, 1]);
        assert_eq!(b.flags & BF_DONT_CLEAR_FIRE_BIT_AFTER_RECOVERING, BF_DONT_CLEAR_FIRE_BIT_AFTER_RECOVERING);

        let magnum = &get(&all, "magnum").barrels[0];
        assert!(near(magnum.fire_recovery_time, 0.19) && near(magnum.soft_recovery_fraction, 0.5));
        let sniper = &get(&all, "sniper_rifle").barrels[0];
        assert!(near(sniper.fire_recovery_time, 0.75) && near(sniper.soft_recovery_fraction, 0.2));
        let needler = &get(&all, "needler").barrels[0];
        assert!(near(needler.rounds_per_second[0], 8.0) && near(needler.rounds_per_second[1], 12.0));
        assert!(near(needler.acceleration_time, 0.8) && near(needler.deceleration_time, 0.1));

        let pistol = get(&all, "plasma_pistol");
        let t = &pistol.triggers[0];
        assert_eq!(t.behavior, Some(Behavior::LatchAutofire));
        assert_eq!((t.primary_barrel, t.secondary_barrel), (Some(0), Some(1)));
        assert!(near(t.autofire_time, 0.2) && near(t.charging_time, 0.8));
        assert_eq!((t.autofire_primary_action, t.autofire_secondary_action), (0, 1), "tap fires, hold charges");
        assert!(near(pistol.barrels[0].heat_per_round, 0.14));
        let launcher = get(&all, "plasma_launcher");
        assert_eq!(launcher.triggers[0].behavior, Some(Behavior::Charge));
        assert!(near(launcher.triggers[0].charging_time, 3.0) && near(launcher.triggers[0].charged_time, 2.0));
        assert_eq!(launcher.triggers[0].fire_fractions, vec![0.25, 0.5, 0.75, 1.0]);
        let laser = get(&all, "spartan_laser");
        assert_eq!(laser.triggers[0].behavior, Some(Behavior::SpewCharge));
        assert!(near(laser.triggers[0].charging_time, 2.5));
        assert_eq!(laser.weapon_type, WEAPON_TYPE_SPLASER);
        assert!(near(get(&all, "spike_rifle").ready_time, 1.3));
        assert!(get(&all, "energy_sword").barrels.is_empty(), "a sword has no barrel to fire");
    }

    /// The compiler bakes the runtime rates from the authored times: 1 / time, and 30 for a time of
    /// 0 (a rate of one 30 Hz step). If the offsets were wrong these would not agree.
    #[test]
    fn the_baked_runtime_rates_are_the_reciprocals_of_the_authored_times() {
        let Some((_g, all)) = weapons() else { return eprintln!("reach_weapon: forge_halo not available") };
        let mut checked = 0;
        for (name, d) in &all {
            for b in &d.barrels {
                for (t, rate) in [(b.acceleration_time, b.acceleration_rate), (b.deceleration_time, b.deceleration_rate)] {
                    let expect = if t > 0.0 { 1.0 / t } else { 30.0 };
                    assert!((rate - expect).abs() < 1.0e-3 * expect, "{name}: time {t} baked to rate {rate}, expected {expect}");
                    checked += 1;
                }
            }
        }
        assert!(checked >= 40, "only {checked} rates checked");
    }

    /// Object functions over heat: the repeater is the one hand weapon that has them, and its rate
    /// of fire really is a function of its heat.
    #[test]
    fn the_plasma_repeaters_rate_of_fire_is_a_function_of_heat() {
        let Some((_g, all)) = weapons() else { return eprintln!("reach_weapon: forge_halo not available") };
        let d = get(&all, "plasma_repeater");
        let b = &d.barrels[0];
        let rof = b.rate_of_fire_acceleration_fn.expect("heat_driven_rate_of_fire");
        assert_eq!(rof[0], 0.0);
        assert!((rof[10] - 1.0).abs() < 1.0e-3, "fully hot = the final rate");
        assert!(rof.windows(2).all(|w| w[1] >= w[0]), "monotone in heat: {rof:?}");
        assert!(b.heat_per_round_fn.is_some() && d.heat_loss_fn.is_some());
        assert!(d.heat_loss_fn.unwrap()[0] > d.heat_loss_fn.unwrap()[10], "it cools slower the hotter it is");
        // and no other hand weapon has a heat-driven rate
        for (name, d) in &all {
            if name != "plasma_repeater" && !name.contains("turret") {
                assert!(d.barrels.iter().all(|b| b.rate_of_fire_acceleration_fn.is_none()), "{name}");
            }
        }
    }
}
