//! The GAME VARIANT a Reach saved film was recorded under (#gamevariant).
//!
//! A film embeds the whole `c_game_variant` as a raw memory image (`flmh` -> `s_game_options`
//! +0x14D8, 0xFC04 bytes). [`crate::film`] already reads its content metadata — title, author,
//! engine type. This module decodes the part that was left undecoded: the **declared settings**
//! — round and respawn options, social and map options, the four player-trait sets, the eight
//! team configurations and the six loadout palettes.
//!
//! Two user-visible things were blocked on exactly this:
//!
//! * **The weapon a player spawns with.** The replication stream names a unit's weapon only when
//!   it CHANGES (0.8% of bodies ever, median 64 s after the body appears), so almost nobody was
//!   drawn holding anything. A spawn weapon is a declared setting: the base player traits'
//!   *initial primary weapon*, overridden by the chosen loadout's *primary weapon*.
//! * **Species.** *Player Species* is a declared setting too (the team options' model override
//!   type), and it says whether an Elite was even possible.
//!
//! The Megalo SCRIPT is deliberately not interpreted — see the notes. This is the declared data
//! only, which is what carries the value.
//!
//! # Where the layout came from
//!
//! The same way every other format in this repo was established, and deliberately NOT by reading
//! anyone's source:
//!
//! * **`haloreach.dll`** (MCC, sha16 `738dd2d24ea3aea1`), with ReverseMe. The decisive function
//!   is the base-variant bitstream decode at **`+0xE1148`**: it reads each field with an explicit
//!   bit width and stores it at an explicit struct offset, which gives the memory layout and the
//!   value ranges in one pass. The loadout accessor at **`+0x4A5A0`** and
//!   `PlayerInteraction_IsEnabled` (**`+0x131FD8`**) independently confirm the loadout array's
//!   base and stride, and `+0x37334` confirms where the variant object starts.
//! * **`reach_tag_test.exe`** tag reflection data (`reverseme tagdefs ... wezr`), which NAMES
//!   every field: `game_engine_settings_definition` holds the very same
//!   miscellaneous/respawn/social/map-override/team/loadout option structs, because the tag is
//!   where the built-in variants' defaults are authored.
//!
//! The two agree field for field. The clinching check is the player-trait set: all 26 of its
//! fields have a bit width in `haloreach.dll` exactly equal to the cardinality of the enum the
//! tag definition names for that field, in tag order (4 bits for a 13-value enum, 3 for a
//! 7-value one, 2 for a 3-value one, ...). That is two independent engine artefacts agreeing on
//! 26 fields, which is why no offset in this module is a guess.
//!
//! Value LABELS for the enums come from two places: the engine's own reflection strings where
//! they survive de-duplication, and otherwise ReachVariantTool's **CC0 (public-domain)
//! documentation** — see `THIRD-PARTY-NOTICES.md`. No code from that project was read, copied or
//! translated; its program code is GPLv3 and this file is an independent implementation from the
//! binary facts above.

use hms_native::{CacheHandle, NativeDll};

// ---------------------------------------------------------------------------------------------
// offsets — all relative to the start of `c_game_variant`
// ---------------------------------------------------------------------------------------------

/// Offsets into `c_game_variant`. The variant OBJECT (a polymorphic `c_game_engine_base_variant`)
/// begins at +0x04 with an 8-byte vtable pointer, so every object-relative offset in
/// `haloreach.dll+0xE1148` is 4 higher here. `engine type` at +0x00 is outside the object.
pub mod off {
    /// u32 game engine type: 0 none, **1 sandbox (Forge)**, **2 megalo**, 3 campaign,
    /// 4 survival. Measured: equals `chdr`'s `engine` byte on **65 of 65** real Reach films.
    pub const ENGINE_TYPE: usize = 0x000;
    /// `s_content_item_metadata` (0x2B0). Content type 6 = game variant; anything else means
    /// the film carries no variant.
    pub const METADATA: usize = 0x00C;
    /// `game_engine_miscellaneous_options`, 8 bytes.
    pub const MISC: usize = 0x2BC;
    /// `game_engine_respawn_options` + the respawn player traits, 0x38 bytes.
    pub const RESPAWN: usize = 0x2C4;
    /// `game_engine_social_options`, 4 bytes.
    pub const SOCIAL: usize = 0x2FC;
    /// `game_engine_map_override_options` flags byte; the rest of the block follows.
    pub const MAP_FLAGS: usize = 0x300;
    /// Base player traits (`map override options` -> `player traits`), 0x2C bytes.
    pub const BASE_TRAITS: usize = 0x304;
    /// i16 weapon set index (-2 map default), i16 vehicle set index.
    pub const WEAPON_SET: usize = 0x330;
    pub const VEHICLE_SET: usize = 0x332;
    pub const RED_TRAITS: usize = 0x334;
    pub const BLUE_TRAITS: usize = 0x360;
    pub const YELLOW_TRAITS: usize = 0x38C;
    /// Three i8 powerup durations, seconds (red, blue, yellow).
    pub const POWERUP_DURATIONS: usize = 0x3B8;
    /// `game_engine_team_options`: model override type (= PLAYER SPECIES), designator switch
    /// type, then the eight team records.
    pub const TEAM_MODEL_OVERRIDE: usize = 0x3BC;
    pub const TEAM_DESIGNATOR_SWITCH: usize = 0x3BD;
    pub const TEAMS: usize = 0x3C0;
    pub const TEAM_STRIDE: usize = 0x58;
    pub const TEAM_COUNT: usize = 8;
    /// `loadout_flags_definition`: bit 0 spartan loadouts enabled, bit 1 elite loadouts enabled.
    pub const LOADOUT_FLAGS: usize = 0x680;
    /// The loadout palettes: 6 tiers x 5 loadouts x 6 bytes, tier-major. The array ends at
    /// +0x735 and the next field is the u16 at +0x736, which is exactly the one
    /// `InitializeGameVariantContext` (`haloreach.dll+0x37334`) clears bit 0 of — the layout
    /// closes with no slack.
    pub const LOADOUTS: usize = 0x681;
    pub const LOADOUT_STRIDE: usize = 6;
    pub const LOADOUT_TIERS: usize = 6;
    pub const LOADOUTS_PER_TIER: usize = 5;
    /// u16, bit 0 set by the FIRST flag the bitstream carries (before the misc options).
    /// **[unknown]**: zero on all 65 real films. The one social flag the memory image has no
    /// home for is `observers enabled`, which is the obvious candidate, but nothing proves it.
    pub const TAIL_FLAGS: usize = 0x736;
    /// u16, a 3-bit value the decode rejects above 3, read between the map options and the team
    /// options. **[unknown]**; zero on all 65 real films.
    pub const TAIL_ENUM: usize = 0x738;
    /// `sizeof(c_game_variant)`, from `game_variant_clear` at `haloreach.dll+0x42D4C`.
    pub const SIZE: usize = 0xFC04;
}

/// A loadout / trait weapon index that means "leave whatever the player already had".
pub const WEAPON_UNCHANGED: i8 = -3;
/// "spawn with no weapon in that slot".
pub const WEAPON_NONE: i8 = -1;
/// "whatever the map's own default is" — the engine reads it from a globals tag, not the variant.
pub const WEAPON_MAP_DEFAULT: i8 = -2;
/// "pick one at random" — unknowable from a recording.
pub const WEAPON_RANDOM: i8 = -4;

// ---------------------------------------------------------------------------------------------
// the decoded settings
// ---------------------------------------------------------------------------------------------

/// One `s_player_trait_shield_vitality` (10 used bytes). Field names are the engine's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShieldTraits {
    pub damage_resistance: u8,
    pub body_multiplier: u8,
    pub body_recharge_rate: u8,
    pub shield_multiplier: u8,
    pub shield_recharge_rate: u8,
    pub overshield_recharge_rate: u8,
    pub headshot_immunity: u8,
    pub vampirism: u8,
    pub assassination_immunity: u8,
}

/// One `s_player_trait_weapons`. **This is where a spawn weapon lives.**
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeaponTraits {
    pub damage_modifier: u8,
    pub melee_damage_modifier: u8,
    pub recharging_grenades: u8,
    pub weapon_pickup: u8,
    /// `player_trait_grenade_count_setting`: see [`grenade_count_label`].
    pub grenade_count: u8,
    pub infinite_ammo: u8,
    pub equipment_usage: u8,
    pub drop_equipment_on_death: u8,
    pub infinite_equipment_energy: u8,
    /// Index into `motl`'s `weapons` collection, or one of the `WEAPON_*` sentinels.
    pub primary_weapon: i8,
    pub secondary_weapon: i8,
    /// Index into `motl`'s `equipment` collection (the Armor Ability), or a sentinel.
    pub equipment: i8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MovementTraits {
    pub speed_multiplier: u8,
    pub gravity_multiplier: u8,
    pub vehicle_usage: u8,
    pub double_jump: u8,
    /// Percent; -1 = unchanged, 0 = cannot jump, 100 = normal.
    pub jump_multiplier: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppearanceTraits {
    pub active_camo: u8,
    pub waypoint: u8,
    pub gamertag: u8,
    pub aura: u8,
    pub forced_change_color: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SensorTraits {
    pub motion_tracker: u8,
    pub motion_tracker_range: u8,
    pub directional_damage: u8,
}

/// `s_game_engine_player_traits`, 0x2C bytes: the five sub-blocks in tag order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerTraits {
    pub shield: ShieldTraits,
    pub weapons: WeaponTraits,
    pub movement: MovementTraits,
    pub appearance: AppearanceTraits,
    pub sensors: SensorTraits,
}

impl PlayerTraits {
    pub const SIZE: usize = 0x2C;

    fn parse(b: &[u8]) -> PlayerTraits {
        let g = |o: usize| b.get(o).copied().unwrap_or(0);
        let s = |o: usize| b.get(o).copied().unwrap_or(0) as i8;
        PlayerTraits {
            shield: ShieldTraits {
                damage_resistance: g(0x00),
                body_multiplier: g(0x01),
                body_recharge_rate: g(0x02),
                shield_multiplier: g(0x03),
                shield_recharge_rate: g(0x04),
                overshield_recharge_rate: g(0x05),
                headshot_immunity: g(0x06),
                vampirism: g(0x07),
                assassination_immunity: g(0x08),
            },
            weapons: WeaponTraits {
                damage_modifier: g(0x0C),
                melee_damage_modifier: g(0x0D),
                recharging_grenades: g(0x0E),
                weapon_pickup: g(0x0F),
                grenade_count: g(0x10),
                infinite_ammo: g(0x11),
                equipment_usage: g(0x12),
                drop_equipment_on_death: g(0x13),
                infinite_equipment_energy: g(0x14),
                primary_weapon: s(0x15),
                secondary_weapon: s(0x16),
                equipment: s(0x17),
            },
            movement: MovementTraits {
                speed_multiplier: g(0x18),
                gravity_multiplier: g(0x19),
                vehicle_usage: g(0x1A),
                double_jump: g(0x1B),
                jump_multiplier: i32le(b, 0x1C),
            },
            appearance: AppearanceTraits {
                active_camo: g(0x20),
                waypoint: g(0x21),
                gamertag: g(0x22),
                aura: g(0x23),
                forced_change_color: g(0x24),
            },
            sensors: SensorTraits {
                motion_tracker: g(0x28),
                motion_tracker_range: g(0x29),
                directional_damage: g(0x2A),
            },
        }
    }

    /// Whether every field is "unchanged" (0) and every index a sentinel — i.e. the set does
    /// nothing. The engine's own default for an untouched trait set.
    pub fn is_unchanged(&self) -> bool {
        *self == PlayerTraits::default_unchanged()
    }

    fn default_unchanged() -> PlayerTraits {
        PlayerTraits {
            movement: MovementTraits { jump_multiplier: -1, ..Default::default() },
            weapons: WeaponTraits {
                primary_weapon: WEAPON_UNCHANGED,
                secondary_weapon: WEAPON_UNCHANGED,
                equipment: WEAPON_UNCHANGED,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// `game_engine_miscellaneous_options`. Flag bits are the engine's own names, in tag order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MiscOptions {
    /// bit 0 teams enabled, 1 round reset players, 2 round reset map, 3 perfection enabled.
    pub flags: u8,
    pub early_victory_win_count: i8,
    /// Minutes. 0 = no limit.
    pub round_time_limit: i8,
    /// 0 = no limit.
    pub number_of_rounds: i8,
    /// Seconds; 0 disabled, -1 infinite (the engine's own comment).
    pub sudden_death_limit: i16,
    /// Seconds; 0 disabled.
    pub grace_period: i16,
}

impl MiscOptions {
    pub fn teams_enabled(&self) -> bool { self.flags & 1 != 0 }
    pub fn perfection_enabled(&self) -> bool { self.flags & 8 != 0 }
}

/// `game_engine_respawn_options`, plus the respawn player traits the memory image embeds where
/// the tag keeps only a name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RespawnOptions {
    /// bit 0 inherit respawn time (RVT: "synchronize with team"), 1 respawn with teammate,
    /// 2 respawn at location, 3 respawn on kills.
    pub flags: u8,
    /// 0 = unlimited.
    pub lives_per_round: i8,
    pub team_lives_per_round: i8,
    /// Seconds.
    pub respawn_time: i8,
    pub suicide_penalty: i8,
    pub betrayal_penalty: i8,
    pub respawn_growth: i8,
    /// "initial loadout selection time", seconds — RVT calls it Loadout Camera Time.
    pub loadout_camera_time: i8,
    pub traits_duration: i8,
    pub traits: PlayerTraits,
}

/// `game_engine_social_options`. The memory image splits the tag's 8-bit flag word into a 2-bit
/// `team changing` enum and five flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SocialOptions {
    /// Five bits. The tag's flag list minus the two team-changing bits and `observers enabled`
    /// leaves exactly these five, in this order: friendly fire, betrayal booting, enemy voice,
    /// open channel voice, dead player voice. **The bit ORDER is [inferred] from tag order**;
    /// the five fields themselves are [code].
    pub flags: u16,
    /// 0 disabled, 1 enabled, 2 balancing only.
    pub team_changing: u16,
}

impl SocialOptions {
    pub fn friendly_fire(&self) -> bool { self.flags & 0x01 != 0 }
    pub fn betrayal_booting(&self) -> bool { self.flags & 0x02 != 0 }
}

/// `game_engine_map_override_options` — the parts that are not trait sets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapOptions {
    /// bit 0 grenades on map, 1 shortcuts on map, 2 equipment on map, 3 powerups on map,
    /// 4 turrets on map, 5 indestructible vehicles.
    pub flags: u8,
    /// -2 = map default.
    pub weapon_set: i16,
    pub vehicle_set: i16,
    /// Seconds, red / blue / yellow.
    pub powerup_duration: [i8; 3],
}

/// One of the eight team configurations (0x58 bytes). The localized name table inside it is not
/// decoded here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TeamConfig {
    /// bit 0 = the team is enabled; bits 1..3 = override colour / ui text tint / ui bitmap tint.
    pub flags: u8,
    /// `initial team designator`; -1 = none.
    pub initial_designator: i8,
    /// `game_engine_team_options_player_model_choice`: 0 spartan, 1 elite. Only consulted when
    /// [`PlayerSpecies::SetByTeam`] is in force.
    pub model_override: u8,
    pub color_override: u32,
    pub ui_text_tint: u32,
    pub ui_bitmap_tint: u32,
    pub number_of_fireteams: u16,
}

impl TeamConfig {
    pub fn enabled(&self) -> bool { self.flags & 1 != 0 }
}

/// `game_engine_team_options_model_override_type` — what the Reach UI calls **Player Species**.
/// The five value names below are the engine's own, read out of `reach_tag_test.exe`'s
/// reflection strings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayerSpecies {
    /// "none (player choice)" — each player spawns as the biped their customisation selects.
    #[default]
    PlayerChoice,
    ForceSpartan,
    ForceElite,
    /// Each team's own `model override` decides.
    SetByTeam,
    /// Defence spartan, offence elite (RVT calls it "Spartans vs Elites").
    SetByDesignator,
    Unknown(u8),
}

impl PlayerSpecies {
    pub fn from_raw(v: u8) -> PlayerSpecies {
        match v {
            0 => PlayerSpecies::PlayerChoice,
            1 => PlayerSpecies::ForceSpartan,
            2 => PlayerSpecies::ForceElite,
            3 => PlayerSpecies::SetByTeam,
            4 => PlayerSpecies::SetByDesignator,
            v => PlayerSpecies::Unknown(v),
        }
    }

    /// The engine's own name for the value.
    pub fn label(&self) -> String {
        match self {
            PlayerSpecies::PlayerChoice => "none (player choice)".into(),
            PlayerSpecies::ForceSpartan => "force spartan".into(),
            PlayerSpecies::ForceElite => "force elite".into(),
            PlayerSpecies::SetByTeam => "set by team".into(),
            PlayerSpecies::SetByDesignator => "set by designator".into(),
            PlayerSpecies::Unknown(v) => format!("unknown ({v})"),
        }
    }

    /// Whether an Elite could possibly appear under this setting. `PlayerChoice` depends on each
    /// player's own customisation, which the film does not carry, so it is `true` but unproven.
    pub fn elites_possible(&self) -> bool {
        !matches!(self, PlayerSpecies::ForceSpartan)
    }
}

/// One entry of a loadout palette (6 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Loadout {
    /// bit 0 = visible in game.
    pub flags: u8,
    /// Index into the `lgtd` tag's `loadout names` block; -1 = none.
    pub name: i8,
    /// Index into `motl`'s `weapons` collection, or a `WEAPON_*` sentinel.
    pub primary_weapon: i8,
    pub secondary_weapon: i8,
    /// Index into `motl`'s `equipment` collection (Armor Ability), or a sentinel.
    pub armor_ability: i8,
    /// `player_trait_grenade_count_setting`; see [`grenade_count_label`].
    pub grenade_count: u8,
}

impl Loadout {
    pub fn visible(&self) -> bool { self.flags & 1 != 0 }
    /// A loadout that sets nothing at all — the shape an unused palette slot has.
    pub fn is_empty(&self) -> bool {
        self.name < 0
            && self.primary_weapon == WEAPON_UNCHANGED
            && self.secondary_weapon == WEAPON_UNCHANGED
            && self.armor_ability == WEAPON_UNCHANGED
    }
}

/// The whole decoded `c_game_variant`'s declared settings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameVariant {
    /// 0 none, 1 sandbox (Forge), 2 megalo, 3 campaign, 4 survival.
    pub engine_type: u32,
    pub misc: MiscOptions,
    pub respawn: RespawnOptions,
    pub social: SocialOptions,
    pub map: MapOptions,
    pub base_traits: PlayerTraits,
    pub red_traits: PlayerTraits,
    pub blue_traits: PlayerTraits,
    pub yellow_traits: PlayerTraits,
    pub species: PlayerSpecies,
    pub designator_switch: u8,
    pub teams: Vec<TeamConfig>,
    /// bit 0 spartan loadouts enabled, bit 1 elite loadouts enabled.
    pub loadout_flags: u8,
    /// 6 tiers x 5 loadouts, tier-major (`tier * 5 + n`).
    pub loadouts: Vec<Loadout>,
    /// `+0x736` / `+0x738`: the two fields the decode writes that nothing has been able to name.
    /// Zero on every real film measured.
    pub unknown_tail: (u16, u16),
}

impl GameVariant {
    /// Decode the raw `c_game_variant` image. `None` when this is not a Reach game variant: the
    /// embedded metadata's content type must be 6. All 65 real Reach films on this machine carry
    /// one; the films that return `None` are another TITLE's (nine Halo 2 Anniversary films share
    /// the BLF container but not this layout, and read as all zero here), which is exactly the
    /// case this check exists for.
    pub fn parse(v: &[u8]) -> Option<GameVariant> {
        if v.len() < off::LOADOUTS + off::LOADOUT_STRIDE * 30 {
            return None;
        }
        // Content type 6 == game variant. Anything else (0 on a film with no variant) means the
        // image is not a variant and every field below would be noise.
        if u32le(v, off::METADATA) != 6 {
            return None;
        }
        let mut teams = Vec::with_capacity(off::TEAM_COUNT);
        for i in 0..off::TEAM_COUNT {
            let t = off::TEAMS + i * off::TEAM_STRIDE;
            teams.push(TeamConfig {
                flags: byte(v, t),
                initial_designator: i8at(v, t + 0x44),
                model_override: byte(v, t + 0x45),
                color_override: u32le(v, t + 0x48),
                ui_text_tint: u32le(v, t + 0x4C),
                ui_bitmap_tint: u32le(v, t + 0x50),
                number_of_fireteams: u16le(v, t + 0x54),
            });
        }
        let mut loadouts = Vec::with_capacity(off::LOADOUT_TIERS * off::LOADOUTS_PER_TIER);
        for i in 0..off::LOADOUT_TIERS * off::LOADOUTS_PER_TIER {
            let r = off::LOADOUTS + i * off::LOADOUT_STRIDE;
            loadouts.push(Loadout {
                flags: byte(v, r),
                name: i8at(v, r + 1),
                primary_weapon: i8at(v, r + 2),
                secondary_weapon: i8at(v, r + 3),
                armor_ability: i8at(v, r + 4),
                grenade_count: byte(v, r + 5),
            });
        }
        let r = off::RESPAWN;
        Some(GameVariant {
            engine_type: u32le(v, off::ENGINE_TYPE),
            misc: MiscOptions {
                flags: byte(v, off::MISC),
                early_victory_win_count: i8at(v, off::MISC + 1),
                round_time_limit: i8at(v, off::MISC + 2),
                number_of_rounds: i8at(v, off::MISC + 3),
                sudden_death_limit: i16le(v, off::MISC + 4),
                grace_period: i16le(v, off::MISC + 6),
            },
            respawn: RespawnOptions {
                flags: byte(v, r),
                lives_per_round: i8at(v, r + 1),
                team_lives_per_round: i8at(v, r + 2),
                respawn_time: i8at(v, r + 3),
                suicide_penalty: i8at(v, r + 4),
                betrayal_penalty: i8at(v, r + 5),
                respawn_growth: i8at(v, r + 6),
                loadout_camera_time: i8at(v, r + 7),
                traits_duration: i8at(v, r + 8),
                traits: PlayerTraits::parse(slice(v, r + 0x0C, PlayerTraits::SIZE)),
            },
            social: SocialOptions { flags: u16le(v, off::SOCIAL), team_changing: u16le(v, off::SOCIAL + 2) },
            map: MapOptions {
                flags: byte(v, off::MAP_FLAGS),
                weapon_set: i16le(v, off::WEAPON_SET),
                vehicle_set: i16le(v, off::VEHICLE_SET),
                powerup_duration: [
                    i8at(v, off::POWERUP_DURATIONS),
                    i8at(v, off::POWERUP_DURATIONS + 1),
                    i8at(v, off::POWERUP_DURATIONS + 2),
                ],
            },
            base_traits: PlayerTraits::parse(slice(v, off::BASE_TRAITS, PlayerTraits::SIZE)),
            red_traits: PlayerTraits::parse(slice(v, off::RED_TRAITS, PlayerTraits::SIZE)),
            blue_traits: PlayerTraits::parse(slice(v, off::BLUE_TRAITS, PlayerTraits::SIZE)),
            yellow_traits: PlayerTraits::parse(slice(v, off::YELLOW_TRAITS, PlayerTraits::SIZE)),
            species: PlayerSpecies::from_raw(byte(v, off::TEAM_MODEL_OVERRIDE)),
            designator_switch: byte(v, off::TEAM_DESIGNATOR_SWITCH),
            teams,
            loadout_flags: byte(v, off::LOADOUT_FLAGS),
            loadouts,
            unknown_tail: (u16le(v, off::TAIL_FLAGS), u16le(v, off::TAIL_ENUM)),
        })
    }

    pub fn engine_name(&self) -> &'static str {
        match self.engine_type {
            0 => "none",
            1 => "sandbox (Forge)",
            2 => "megalo",
            3 => "campaign",
            4 => "survival (Firefight)",
            _ => "unknown",
        }
    }

    pub fn loadout(&self, tier: usize, n: usize) -> Option<&Loadout> {
        self.loadouts.get(tier * off::LOADOUTS_PER_TIER + n)
    }

    /// Every loadout that is visible in game, as `(tier, slot, loadout)`.
    pub fn visible_loadouts(&self) -> impl Iterator<Item = (usize, usize, &Loadout)> {
        self.loadouts.iter().enumerate().filter(|(_, l)| l.visible()).map(|(i, l)| {
            (i / off::LOADOUTS_PER_TIER, i % off::LOADOUTS_PER_TIER, l)
        })
    }

    /// The weapon a player with this loadout spawns holding — the engine's own precedence
    /// (`haloreach.dll+0x4A5A0`): the loadout overrides the player's traits, and only a field
    /// that is NOT "unchanged" overrides.
    pub fn primary_for(&self, tier: usize, n: usize) -> i8 {
        match self.loadout(tier, n) {
            Some(l) if l.primary_weapon != WEAPON_UNCHANGED => l.primary_weapon,
            _ => self.base_traits.weapons.primary_weapon,
        }
    }

    /// What every player spawns holding, when the variant says the same thing for all of them.
    ///
    /// The film does NOT record which loadout a player picked (it is player state, not a
    /// replicated entity field), so this is only answerable when the answer does not depend on
    /// the choice. Two cases do:
    ///
    /// * the base player traits name a primary weapon — it applies to everyone, and a loadout
    ///   can only override it if that loadout sets its own, which is then reported as
    ///   [`SpawnWeapon::varies`];
    /// * no trait weapon, but every visible loadout names the SAME primary.
    pub fn spawn_primary(&self) -> SpawnWeapon {
        let base = self.base_traits.weapons.primary_weapon;
        let mut vis: Vec<i8> = self.visible_loadouts().map(|(t, n, _)| self.primary_for(t, n)).collect();
        vis.sort_unstable();
        vis.dedup();
        match (base, vis.len()) {
            // Nothing visible to choose between: the traits decide outright.
            (b, 0) => SpawnWeapon { index: b, varies: false, from_loadout: false },
            (_, 1) => SpawnWeapon {
                index: vis[0],
                varies: false,
                from_loadout: vis[0] != base,
            },
            (b, _) => SpawnWeapon { index: b, varies: true, from_loadout: false },
        }
    }
}

/// What species a recorded player's body should be DRAWN as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DrawnSpecies {
    Spartan,
    Elite,
    /// The variant defers to something the film does not carry — each player's own customisation
    /// (*player choice*), or a designator role this project has not established. A Spartan is
    /// the right thing to draw, but the caption has to say it is a DEFAULT.
    #[default]
    NotRecorded,
}

impl GameVariant {
    /// The species a player on `team` spawned as, as far as the variant can say.
    ///
    /// *player choice* — which is 42 of the 43 real variants on this machine — means the biped
    /// comes from the player's own customisation options, and that is profile data: it is not in
    /// the variant and it is not in the player record either (the sweep that looked for it there
    /// is in the notes). So the honest answer there is [`DrawnSpecies::NotRecorded`].
    pub fn species_for_team(&self, team: u8) -> DrawnSpecies {
        match self.species {
            PlayerSpecies::ForceSpartan => DrawnSpecies::Spartan,
            PlayerSpecies::ForceElite => DrawnSpecies::Elite,
            PlayerSpecies::SetByTeam => match self.teams.get(team as usize) {
                // `game_engine_team_options_player_model_choice`: 0 spartan, 1 elite.
                Some(t) if t.model_override == 1 => DrawnSpecies::Elite,
                Some(_) => DrawnSpecies::Spartan,
                None => DrawnSpecies::NotRecorded,
            },
            // "set by designator" is defence spartan / offence elite, but which designator VALUE
            // is the defence role is not established here and no film on this machine exercises
            // it, so this does not guess.
            PlayerSpecies::SetByDesignator => DrawnSpecies::NotRecorded,
            PlayerSpecies::PlayerChoice | PlayerSpecies::Unknown(_) => DrawnSpecies::NotRecorded,
        }
    }

    /// #team-colour  The team a body belongs to, from the owner stamp its creation carries.
    ///
    /// The stamp is the object's `owner team` (5 bits, `index + 1`, 0 = nobody), which for a
    /// player's biped is the player's `m_multiplayer_team` -- the same value the engine's
    /// `player_refresh_object_appearance_full` hands the colour builder. It is only a TEAM when the
    /// variant has teams (in a free-for-all every player is a team of one, which is why it looks
    /// like a player index there), so `None` unless this variant's teams bit is set, and `None`
    /// for a stamp naming a team the variant does not enable: never guessed.
    pub fn team_of_owner(&self, owner_raw: u8) -> Option<u8> {
        if !self.misc.teams_enabled() || owner_raw == 0 {
            return None;
        }
        let team = owner_raw - 1;
        self.teams.get(team as usize).filter(|t| t.enabled()).map(|_| team)
    }

    /// #team-colour  The team of a recorded BODY: the owner-stamp rule of
    /// [`GameVariant::team_of_owner`], for a PLAYER's biped only -- the one whose creation carries
    /// an appearance block. The gametype's and the AI's bodies keep their authored colours.
    pub fn team_of_body(&self, has_appearance: bool, owner_raw: Option<u8>) -> Option<u8> {
        if !has_appearance {
            return None;
        }
        self.team_of_owner(owner_raw?)
    }

    /// #team-colour  The variant's own colour for `team`, when it overrides the default.
    ///
    /// `sub_1404D1F00` (`game_engine_team.cpp`): the variant must have teams enabled, the team
    /// record must be enabled, and its flag bit 1 (`override color`) must be set; the colour is then
    /// the record's packed `0x00RRGGBB` at +0x48, each byte times `0.00392156886`, unconverted.
    pub fn team_color_override(&self, team: u8) -> Option<[f32; 3]> {
        if !self.misc.teams_enabled() {
            return None;
        }
        let t = self.teams.get(team as usize).filter(|t| t.enabled() && t.flags & 2 != 0)?;
        let c = t.color_override;
        let ch = |shift: u32| ((c >> shift) & 0xFF) as f32 * 0.003_921_568_6;
        Some([ch(16), ch(8), ch(0)])
    }

    /// #team-colour  The `[primary, secondary]` change colours a body on `team` wears.
    ///
    /// `sub_1403AC310`: with a team the engine builds the customisation's colours first and then
    /// OVERWRITES the primary and the secondary slot with the team colour -- the variant's override
    /// if it has one, else the multiplayer globals' default table entry (`defaults`, `mulg`
    /// universal `team colors`). The tertiary and quaternary slots (the visor, the lights) stay the
    /// player's own, and Theater only draws the first two. `None` when neither source has an
    /// entry: the caller keeps the customisation rather than invent a colour.
    pub fn team_skin(&self, team: u8, defaults: &[[f32; 3]]) -> Option<[[f32; 3]; 2]> {
        let c = self.team_color_override(team).or_else(|| defaults.get(team as usize).copied())?;
        Some([c, c])
    }

    /// Whether any ENABLED team is set to elite AND the species setting actually consults it.
    /// A team can carry `model override = elite` under *player choice*, where it is inert — 8 of
    /// the 26 elite team records in this machine's corpus are exactly that.
    pub fn any_team_is_elite(&self) -> bool {
        self.species == PlayerSpecies::SetByTeam
            && self.teams.iter().any(|t| t.enabled() && t.model_override == 1)
    }
}

/// #team-colour  The multiplayer globals' default TEAM colours: `mulg` -> `universal` block
/// (element 76 B) -> `team colors` block at +0x20, 12-byte linear RGB per team (`sub_140AC50F0`).
/// Indexed by team. Empty when the cache has no readable `mulg`. Same read shape as
/// `SceneController::profile_colors`, and the same linear-float space.
pub fn default_team_colors(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> Vec<[f32; 3]> {
    let Some(mulg) = native.tags_by_group(cache, "mulg").first().copied() else { return Vec::new() };
    let u = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
    let Some(h) = native.tag_meta(cache, mulg, 0, 8) else { return Vec::new() };
    if u(&h, 0) == 0 {
        return Vec::new();
    }
    let Some(univ) = native.tag_ptr_bytes(cache, u(&h, 4), 0x2C) else { return Vec::new() };
    let (cnt, ptr) = (u(&univ, 0x20), u(&univ, 0x24));
    if cnt == 0 || cnt > 32 {
        return Vec::new();
    }
    let Some(b) = native.tag_ptr_bytes(cache, ptr, cnt * 12) else { return Vec::new() };
    (0..cnt as usize)
        .map(|i| {
            let f = |k: usize| f32::from_le_bytes([b[i * 12 + k * 4], b[i * 12 + k * 4 + 1], b[i * 12 + k * 4 + 2], b[i * 12 + k * 4 + 3]]);
            let c = [f(0), f(1), f(2)];
            if c.iter().all(|v| v.is_finite() && *v >= 0.0) { c } else { [1.0; 3] }
        })
        .collect()
}

/// The answer to "what is in this player's hands the moment they spawn".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpawnWeapon {
    /// A `motl` weapons-collection index, or one of the `WEAPON_*` sentinels.
    pub index: i8,
    /// True when the visible loadouts disagree, so the answer depends on a choice the film does
    /// not record. `index` is then the trait fallback only.
    pub varies: bool,
    /// True when the value came from the loadout palette rather than the base player traits.
    pub from_loadout: bool,
}

impl SpawnWeapon {
    /// Whether this names a real weapon (rather than unchanged / none / random / map default).
    pub fn is_concrete(&self) -> bool { self.index >= 0 && !self.varies }
}

/// Which rule produced the weapon a body is DRAWN holding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpawnWeaponRule {
    /// The variant says the same thing for every player: either the base player traits name the
    /// weapon and no visible loadout overrides it, or every visible loadout names the same one.
    /// Not a guess.
    Stated,
    /// The visible loadouts disagree, so the weapon depends on a choice the film does not
    /// record, and this is the **most common** primary weapon among them (ties to the lowest
    /// tier and slot). Right for the plurality of players and wrong for the rest; the caption
    /// says so.
    MostCommonLoadout,
    #[default]
    None,
}

impl GameVariant {
    /// The weapon to DRAW in every body's hands, which is a slightly different question from
    /// [`GameVariant::spawn_primary`].
    ///
    /// `spawn_primary` refuses to answer when the visible loadouts disagree, which is the right
    /// thing to report. But a body that spawned certainly held SOMETHING, and in a gametype whose
    /// ten selectable loadouts carry an energy sword eight times, drawing nothing is less true
    /// than drawing the sword. So the drawing path falls back to the **modal** primary weapon of
    /// the visible loadouts — the maximum-likelihood answer under a choice the film does not
    /// record — and reports [`SpawnWeaponRule::MostCommonLoadout`] so the caption can say it is
    /// the gametype's most common loadout rather than this player's.
    pub fn spawn_primary_drawable(&self) -> (i8, SpawnWeaponRule) {
        let sw = self.spawn_primary();
        if !sw.varies {
            // A SENTINEL is not a statement. "unchanged", "map default", "random" and "none"
            // all mean the variant declines to name a weapon, so there is nothing to draw and
            // the caption must not claim the gametype stated one for every player.
            let rule = if sw.index >= 0 { SpawnWeaponRule::Stated } else { SpawnWeaponRule::None };
            return (sw.index, rule);
        }
        // Count the primaries the visible loadouts actually give, in tier/slot order so a tie
        // resolves to the first selectable loadout rather than to hash order.
        let mut counts: Vec<(i8, usize)> = Vec::new();
        for (t, n, _) in self.visible_loadouts() {
            let v = self.primary_for(t, n);
            match counts.iter_mut().find(|(w, _)| *w == v) {
                Some((_, c)) => *c += 1,
                None => counts.push((v, 1)),
            }
        }
        match counts.iter().copied().enumerate().max_by_key(|&(i, (_, c))| (c, usize::MAX - i)) {
            Some((_, (v, _))) => (v, SpawnWeaponRule::MostCommonLoadout),
            None => (sw.index, SpawnWeaponRule::None),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// value labels
// ---------------------------------------------------------------------------------------------

/// `player_trait_grenade_count_setting`, all fifteen values. Engine-named, and corroborated by
/// the two engine helpers that split the value into a frag and a plasma count
/// (`haloreach.dll+0x108EF4` / `+0x108F44`): 3..6 give 1..4 frags, 7..10 give 1..4 plasmas and
/// 11..14 give 1..4 of each, which is exactly this order.
pub fn grenade_count_label(v: u8) -> String {
    const N: [&str; 15] = [
        "unchanged", "none", "map default", "1 frag", "2 frag", "3 frag", "4 frag", "1 plasma",
        "2 plasma", "3 plasma", "4 plasma", "1 each", "2 each", "3 each", "4 each",
    ];
    N.get(v as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("{v}"))
}

/// `player_trait_damage_resistance`, all thirteen values (engine reflection strings).
pub fn damage_resistance_label(v: u8) -> String {
    const N: [&str; 13] = [
        "unchanged", "10%", "50%", "90%", "100%", "110%", "150%", "200%", "300%", "500%",
        "1000%", "2000%", "invulnerable",
    ];
    N.get(v as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("{v}"))
}

/// `player_trait_boolean`: unchanged / disabled / enabled.
pub fn boolean_label(v: u8) -> &'static str {
    match v {
        0 => "unchanged",
        1 => "disabled",
        2 => "enabled",
        _ => "?",
    }
}

/// `player_trait_movement_vehicle_usage` (engine reflection strings).
pub fn vehicle_usage_label(v: u8) -> String {
    const N: [&str; 9] = [
        "unchanged", "none", "full", "passenger only", "driver only", "gunner only",
        "not passenger", "not driver", "not gunner",
    ];
    N.get(v as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("{v}"))
}

/// How a weapon/equipment index prints when it is not a real index.
pub fn weapon_sentinel(v: i8) -> Option<&'static str> {
    match v {
        WEAPON_UNCHANGED => Some("unchanged"),
        WEAPON_MAP_DEFAULT => Some("map default"),
        WEAPON_NONE => Some("none"),
        WEAPON_RANDOM => Some("random"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// resolving the indices against the open map
// ---------------------------------------------------------------------------------------------

mod motl {
    /// `multiplayer_object_type_list`: `object types` block, 20-byte entries
    /// `{ string id name; tag reference object; }`.
    pub const OBJECT_TYPES: usize = 0x000;
    pub const OBJECT_TYPE_ELEM: usize = 20;
    /// `weapons` / `equipment` collections, 16-byte entries
    /// `{ long block index object type; string id description; real weight; long sprite; }`.
    pub const WEAPONS: usize = 0x00C;
    pub const EQUIPMENT: usize = 0x030;
    pub const COLLECTION_ELEM: usize = 16;
    pub const MAX: i32 = 2048;
}

mod lgtd {
    /// `loadout_globals_definition`: `loadout names`, 4-byte entries holding one string id.
    pub const LOADOUT_NAMES: usize = 0x018;
    pub const NAME_ELEM: usize = 4;
    pub const MAX: i32 = 128;
}

/// Everything the OPEN MAP can say about the indices a variant carries: what weapon a loadout's
/// index means, and what a loadout is called.
///
/// Both tables are plain tag data — `multiplayer_object_type_list` ('motl') and
/// `loadout_globals_definition` ('lgtd') — and the engine resolves a loadout the same way
/// (`haloreach.dll+0x132E10` indexes `motl`'s `weapons` block; `+0x1F0644` indexes `lgtd`'s
/// `loadout names`).
#[derive(Clone, Debug, Default)]
pub struct VariantNames {
    /// Per `motl` weapons-collection index: the `weap` tag id and its tag path.
    pub weapons: Vec<(u32, String)>,
    /// Per `motl` equipment-collection index: the `eqip` tag id and its tag path.
    pub equipment: Vec<(u32, String)>,
    /// Per `lgtd` loadout-name index: the resolved display string.
    pub loadout_names: Vec<String>,
}

impl VariantNames {
    /// Read `motl` and `lgtd` out of the open cache. Empty vectors when the map has neither,
    /// which is not an error — the caller then prints indices.
    pub fn build(native: &NativeDll, cache: CacheHandle) -> VariantNames {
        let mut out = VariantNames::default();
        if let Some(&tag) = native.tags_by_group(cache, "motl").first() {
            let types = crate::reach_actor::tag_block(
                native, cache, tag, motl::OBJECT_TYPES, motl::OBJECT_TYPE_ELEM, motl::MAX,
            );
            let resolve = |slot: i32| -> (u32, String) {
                let o = slot.max(0) as usize * motl::OBJECT_TYPE_ELEM;
                if slot < 0 || o + motl::OBJECT_TYPE_ELEM > types.len() {
                    return (0xFFFF_FFFF, String::new());
                }
                // A Reach tag reference is 16 bytes with the tag id at +0x0C (low 16 bits);
                // the reference itself sits at +0x04 of the entry.
                let id = u32le(&types, o + 0x04 + 0x0C);
                if id == 0xFFFF_FFFF {
                    return (0xFFFF_FFFF, String::new());
                }
                let id = id & 0xFFFF;
                (id, native.tag_name(cache, id).unwrap_or_default())
            };
            for (off, dst) in [(motl::WEAPONS, 0usize), (motl::EQUIPMENT, 1)] {
                let col = crate::reach_actor::tag_block(
                    native, cache, tag, off, motl::COLLECTION_ELEM, motl::MAX,
                );
                let v: Vec<(u32, String)> = col
                    .chunks_exact(motl::COLLECTION_ELEM)
                    .map(|e| resolve(i32le(e, 0)))
                    .collect();
                if dst == 0 { out.weapons = v } else { out.equipment = v }
            }
        }
        if let Some(&tag) = native.tags_by_group(cache, "lgtd").first() {
            let names = crate::reach_actor::tag_block(
                native, cache, tag, lgtd::LOADOUT_NAMES, lgtd::NAME_ELEM, lgtd::MAX,
            );
            out.loadout_names = names
                .chunks_exact(lgtd::NAME_ELEM)
                .map(|e| native.string_id(cache, u32le(e, 0)).unwrap_or_default())
                .collect();
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.weapons.is_empty() && self.equipment.is_empty() && self.loadout_names.is_empty()
    }

    /// A weapon index as text: the sentinel's name, the tag's leaf name, or the bare index.
    pub fn weapon_label(&self, v: i8) -> String {
        if let Some(s) = weapon_sentinel(v) { return s.into() }
        match self.weapons.get(v.max(0) as usize) {
            Some((_, p)) if !p.is_empty() => leaf(p),
            _ => format!("#{v}"),
        }
    }

    pub fn equipment_label(&self, v: i8) -> String {
        if let Some(s) = weapon_sentinel(v) { return s.into() }
        match self.equipment.get(v.max(0) as usize) {
            Some((_, p)) if !p.is_empty() => leaf(p),
            _ => format!("#{v}"),
        }
    }

    pub fn loadout_name(&self, v: i8) -> String {
        if v < 0 { return String::new() }
        self.loadout_names.get(v as usize).cloned().unwrap_or_else(|| format!("#{v}"))
    }

    /// The `weap` TAG ID a weapon index names, if the map has it.
    pub fn weapon_tag(&self, v: i8) -> Option<u32> {
        if v < 0 { return None }
        match self.weapons.get(v as usize) {
            Some(&(id, _)) if id != 0xFFFF_FFFF => Some(id),
            _ => None,
        }
    }
}

/// `objects\weapons\rifle\dmr\dmr` -> `dmr`.
fn leaf(p: &str) -> String {
    p.rsplit(|c| c == '\\' || c == '/').next().unwrap_or(p).to_string()
}

// ---------------------------------------------------------------------------------------------
// the spawn weapon, wired through to the drawn bodies
// ---------------------------------------------------------------------------------------------

/// What [`apply_spawn_weapons`] did, so the caller can label it honestly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpawnWeaponApplied {
    /// The index into the map's `weap` group list that every body now starts with.
    pub weap_index: Option<u32>,
    /// The resolved tag path, for the caption.
    pub tag_name: String,
    /// How many biped tracks it was applied to.
    pub bodies: usize,
    /// Which rule produced it.
    pub rule: SpawnWeaponRule,
    /// Why nothing was applied, when `weap_index` is `None`.
    pub why: &'static str,
}

/// Give every recorded body the weapon the GAME VARIANT says its player spawned with.
///
/// `weap_tags` is the map's `weap` group list in the order the playback indexes it, so the value
/// written is in the same space as the `'weap'` reports the stream carries and a real pickup
/// still overrides it from the tick it arrives.
///
/// Returns what it did. It applies nothing when the variant's answer depends on a loadout choice
/// the film does not record ([`SpawnWeapon::varies`]), when the variant leaves the weapon
/// "unchanged"/"random"/"map default", or when the open map has no `motl` to resolve the index
/// through — in each case drawing nothing is the honest outcome.
pub fn apply_spawn_weapons(
    tracks: &mut [crate::film_decode::EntityTrack],
    biped_type: u8,
    variant: &GameVariant,
    names: &VariantNames,
    weap_tags: &[u32],
) -> SpawnWeaponApplied {
    let (index, rule) = variant.spawn_primary_drawable();
    let sw = SpawnWeapon { index, ..variant.spawn_primary() };
    if let Some(s) = weapon_sentinel(sw.index) {
        return SpawnWeaponApplied { why: match s {
            "unchanged" => "the variant leaves the spawn weapon unchanged",
            "none" => "the variant spawns players with no weapon",
            "map default" => "the variant defers to the map's default weapon, which is not in the film",
            _ => "the variant picks the spawn weapon at random",
        }, ..Default::default() };
    }
    let Some(tag) = names.weapon_tag(sw.index) else {
        return SpawnWeaponApplied { why: "the open map has no multiplayer object type list to resolve the index", ..Default::default() };
    };
    let Some(idx) = weap_tags.iter().position(|&t| t == tag) else {
        return SpawnWeaponApplied { why: "the weapon the variant names is not in this map", ..Default::default() };
    };
    let mut bodies = 0usize;
    for t in tracks.iter_mut().filter(|t| t.entity_type == biped_type) {
        t.spawn_weapon = Some(idx as u32);
        bodies += 1;
    }
    SpawnWeaponApplied {
        weap_index: Some(idx as u32),
        tag_name: names.weapon_label(sw.index),
        bodies,
        rule,
        why: match rule {
            SpawnWeaponRule::MostCommonLoadout =>
                "the gametype's selectable loadouts disagree, so this is the most common one; a \
                 player's own choice is not recorded",
            _ => "",
        },
    }
}

// ---------------------------------------------------------------------------------------------
// a one-screen dump, for `--film-stats` and the Theater card
// ---------------------------------------------------------------------------------------------

impl GameVariant {
    /// A compact human summary. `names` may be empty (no map open), in which case indices print
    /// as `#n`.
    pub fn summary(&self, names: &VariantNames) -> String {
        let mut s = String::new();
        let v = &self.misc;
        s.push_str(&format!("engine           {} ({})\n", self.engine_name(), self.engine_type));
        s.push_str(&format!(
            "rounds           {} round(s), {} min each, {} to win, sudden death {} s, grace {} s\n",
            if v.number_of_rounds == 0 { "no limit".into() } else { v.number_of_rounds.to_string() },
            if v.round_time_limit == 0 { "no limit".into() } else { v.round_time_limit.to_string() },
            v.early_victory_win_count, v.sudden_death_limit, v.grace_period,
        ));
        s.push_str(&format!(
            "teams            {}, species {}, {} enabled\n",
            if v.teams_enabled() { "on" } else { "off" },
            self.species.label(),
            self.teams.iter().filter(|t| t.enabled()).count(),
        ));
        let r = &self.respawn;
        s.push_str(&format!(
            "respawn          {} s (+{} growth), suicide +{}, betrayal +{}, lives {}, loadout cam {} s\n",
            r.respawn_time, r.respawn_growth, r.suicide_penalty, r.betrayal_penalty,
            if r.lives_per_round == 0 { "unlimited".into() } else { r.lives_per_round.to_string() },
            r.loadout_camera_time,
        ));
        s.push_str(&format!(
            "social           friendly fire {}, betrayal booting {}, team changing {}\n",
            if self.social.friendly_fire() { "on" } else { "off" },
            if self.social.betrayal_booting() { "on" } else { "off" },
            self.social.team_changing,
        ));
        let t = &self.base_traits;
        s.push_str(&format!(
            "base traits      primary {}, secondary {}, ability {}, grenades {}, damage resistance {}\n",
            names.weapon_label(t.weapons.primary_weapon),
            names.weapon_label(t.weapons.secondary_weapon),
            names.equipment_label(t.weapons.equipment),
            grenade_count_label(t.weapons.grenade_count),
            damage_resistance_label(t.shield.damage_resistance),
        ));
        let (drawn, rule) = self.spawn_primary_drawable();
        s.push_str(&format!(
            "spawn weapon     {}{}\n",
            names.weapon_label(drawn),
            match rule {
                SpawnWeaponRule::MostCommonLoadout =>
                    "  (the gametype's most common selectable loadout; a player's own choice is not recorded)",
                _ => "",
            },
        ));
        s.push_str(&format!(
            "loadouts         {} visible of {}, spartan {}, elite {}\n",
            self.visible_loadouts().count(),
            self.loadouts.len(),
            if self.loadout_flags & 1 != 0 { "on" } else { "off" },
            if self.loadout_flags & 2 != 0 { "on" } else { "off" },
        ));
        for (tier, n, l) in self.visible_loadouts() {
            s.push_str(&format!(
                "  tier {tier} #{n}        {:<22} {} / {} / {} / {}\n",
                names.loadout_name(l.name),
                names.weapon_label(l.primary_weapon),
                names.weapon_label(l.secondary_weapon),
                names.equipment_label(l.armor_ability),
                grenade_count_label(l.grenade_count),
            ));
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// little endian readers (bounds-checked; a short buffer reads zero)
// ---------------------------------------------------------------------------------------------

fn byte(b: &[u8], o: usize) -> u8 { b.get(o).copied().unwrap_or(0) }
fn i8at(b: &[u8], o: usize) -> i8 { byte(b, o) as i8 }
fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([byte(b, o), byte(b, o + 1)]) }
fn i16le(b: &[u8], o: usize) -> i16 { u16le(b, o) as i16 }
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([byte(b, o), byte(b, o + 1), byte(b, o + 2), byte(b, o + 3)])
}
fn i32le(b: &[u8], o: usize) -> i32 { u32le(b, o) as i32 }
fn slice(b: &[u8], o: usize, n: usize) -> &[u8] { b.get(o..o + n).unwrap_or(&[]) }

// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic variant image built from the spec, to pin every offset this module reads.
    fn synthetic() -> Vec<u8> {
        let mut v = vec![0u8; off::SIZE];
        v[off::ENGINE_TYPE] = 2;
        v[off::METADATA] = 6; // content type: game variant
        v[off::MISC] = 0b1001; // teams enabled + perfection enabled
        v[off::MISC + 1] = 3; // early victory win count
        v[off::MISC + 2] = 15; // round time limit, minutes
        v[off::MISC + 3] = 2; // number of rounds
        v[off::MISC + 4..off::MISC + 6].copy_from_slice(&(-1i16).to_le_bytes());
        v[off::MISC + 6..off::MISC + 8].copy_from_slice(&30i16.to_le_bytes());
        v[off::RESPAWN] = 0b0001;
        v[off::RESPAWN + 1] = 5; // lives
        v[off::RESPAWN + 3] = 7; // respawn time
        v[off::RESPAWN + 7] = 10; // loadout camera time
        v[off::RESPAWN + 0x0C + 0x15] = 9u8; // respawn traits: primary weapon 9
        v[off::SOCIAL] = 0b0000_0011;
        v[off::SOCIAL + 2] = 2;
        v[off::MAP_FLAGS] = 0b11_1111;
        v[off::WEAPON_SET..off::WEAPON_SET + 2].copy_from_slice(&(-2i16).to_le_bytes());
        v[off::POWERUP_DURATIONS] = 5;
        v[off::POWERUP_DURATIONS + 1] = 30;
        v[off::POWERUP_DURATIONS + 2] = 90;
        // base traits: damage resistance, a primary weapon and a grenade count.
        v[off::BASE_TRAITS] = 12; // invulnerable
        v[off::BASE_TRAITS + 0x10] = 4; // grenade count = 2 frag
        v[off::BASE_TRAITS + 0x15] = 6; // primary weapon index 6
        v[off::BASE_TRAITS + 0x16] = (-1i8) as u8; // secondary: none
        v[off::BASE_TRAITS + 0x1C..off::BASE_TRAITS + 0x20].copy_from_slice(&(-1i32).to_le_bytes());
        v[off::TEAM_MODEL_OVERRIDE] = 2; // force elite
        v[off::TEAM_DESIGNATOR_SWITCH] = 1;
        v[off::TEAMS] = 1; // team 0 enabled
        v[off::TEAMS + 0x44] = 0;
        v[off::TEAMS + 0x45] = 1; // elite
        v[off::TEAMS + 0x54] = 3; // fireteams
        v[off::TEAMS + off::TEAM_STRIDE] = 1; // team 1 enabled
        v[off::LOADOUT_FLAGS] = 3;
        // tier 0 slot 0: visible, name 7, primary 11, secondary unchanged, AA 2, 4 each.
        let r = off::LOADOUTS;
        v[r] = 1;
        v[r + 1] = 7;
        v[r + 2] = 11;
        v[r + 3] = WEAPON_UNCHANGED as u8;
        v[r + 4] = 2;
        v[r + 5] = 14;
        // every other slot: empty, as a real unused palette slot is.
        for i in 1..off::LOADOUT_TIERS * off::LOADOUTS_PER_TIER {
            let r = off::LOADOUTS + i * off::LOADOUT_STRIDE;
            v[r + 1] = (-1i8) as u8;
            v[r + 2] = WEAPON_UNCHANGED as u8;
            v[r + 3] = WEAPON_UNCHANGED as u8;
            v[r + 4] = WEAPON_UNCHANGED as u8;
        }
        v
    }

    #[test]
    fn every_field_lands_where_the_spec_says() {
        let gv = GameVariant::parse(&synthetic()).expect("a content-type-6 image parses");
        assert_eq!(gv.engine_type, 2);
        assert_eq!(gv.engine_name(), "megalo");
        assert!(gv.misc.teams_enabled());
        assert!(gv.misc.perfection_enabled());
        assert_eq!(gv.misc.early_victory_win_count, 3);
        assert_eq!(gv.misc.round_time_limit, 15);
        assert_eq!(gv.misc.number_of_rounds, 2);
        assert_eq!(gv.misc.sudden_death_limit, -1);
        assert_eq!(gv.misc.grace_period, 30);
        assert_eq!(gv.respawn.lives_per_round, 5);
        assert_eq!(gv.respawn.respawn_time, 7);
        assert_eq!(gv.respawn.loadout_camera_time, 10);
        assert_eq!(gv.respawn.traits.weapons.primary_weapon, 9);
        assert!(gv.social.friendly_fire());
        assert!(gv.social.betrayal_booting());
        assert_eq!(gv.social.team_changing, 2);
        assert_eq!(gv.map.weapon_set, -2);
        assert_eq!(gv.map.powerup_duration, [5, 30, 90]);
        assert_eq!(gv.base_traits.shield.damage_resistance, 12);
        assert_eq!(damage_resistance_label(12), "invulnerable");
        assert_eq!(gv.base_traits.weapons.primary_weapon, 6);
        assert_eq!(gv.base_traits.weapons.secondary_weapon, WEAPON_NONE);
        assert_eq!(gv.base_traits.weapons.grenade_count, 4);
        assert_eq!(grenade_count_label(4), "2 frag");
        assert_eq!(gv.base_traits.movement.jump_multiplier, -1);
        assert_eq!(gv.species, PlayerSpecies::ForceElite);
        assert_eq!(gv.species.label(), "force elite");
        assert_eq!(gv.teams.len(), 8);
        assert!(gv.teams[0].enabled() && gv.teams[1].enabled() && !gv.teams[2].enabled());
        assert_eq!(gv.teams[0].model_override, 1);
        assert_eq!(gv.teams[0].number_of_fireteams, 3);
        assert_eq!(gv.loadout_flags, 3);
        assert_eq!(gv.loadouts.len(), 30);
        let l = gv.loadout(0, 0).copied().unwrap();
        assert!(l.visible());
        assert_eq!((l.name, l.primary_weapon, l.armor_ability, l.grenade_count), (7, 11, 2, 14));
        assert_eq!(grenade_count_label(14), "4 each");
        assert_eq!(gv.visible_loadouts().count(), 1);
    }

    /// The loadout overrides the traits, and only when it is not "unchanged" — the engine's own
    /// precedence at `haloreach.dll+0x4A5A0`.
    #[test]
    fn the_loadout_overrides_the_traits_only_when_it_sets_something() {
        let gv = GameVariant::parse(&synthetic()).unwrap();
        // tier 0 slot 0 sets primary 11, so it wins over the traits' 6.
        assert_eq!(gv.primary_for(0, 0), 11);
        // tier 0 slot 1 leaves primary unchanged, so the base traits' 6 stands.
        assert_eq!(gv.primary_for(0, 1), 6);
        // One visible loadout, so the spawn weapon is unambiguous and comes from it.
        let sw = gv.spawn_primary();
        assert_eq!(sw.index, 11);
        assert!(sw.is_concrete() && sw.from_loadout && !sw.varies);
    }

    /// When the selectable loadouts disagree the DRAWING path takes the most common primary,
    /// and reports that it did. Modelled on the real `"Assassination."` gametype, whose ten
    /// visible loadouts carry an energy sword eight times.
    #[test]
    fn the_drawn_weapon_falls_back_to_the_most_common_loadout() {
        let mut v = synthetic();
        // Four visible loadouts: primary 11, 11, 11, 19.
        for (i, prim) in [(0usize, 11i8), (1, 11), (2, 11), (3, 19)] {
            let r = off::LOADOUTS + i * off::LOADOUT_STRIDE;
            v[r] = 1;
            v[r + 2] = prim as u8;
        }
        let gv = GameVariant::parse(&v).unwrap();
        assert!(gv.spawn_primary().varies, "they disagree, so the stated answer refuses");
        assert_eq!(gv.spawn_primary_drawable(), (11, SpawnWeaponRule::MostCommonLoadout));

        // Make them agree and the rule goes back to Stated.
        let r = off::LOADOUTS + 3 * off::LOADOUT_STRIDE;
        v[r + 2] = 11;
        let gv = GameVariant::parse(&v).unwrap();
        assert_eq!(gv.spawn_primary_drawable(), (11, SpawnWeaponRule::Stated));
    }

    /// Two visible loadouts that disagree cannot be resolved from a film: the choice is not
    /// recorded, so the answer must say so rather than pick one.
    #[test]
    fn disagreeing_visible_loadouts_are_reported_as_varying() {
        let mut v = synthetic();
        let r = off::LOADOUTS + off::LOADOUT_STRIDE;
        v[r] = 1; // visible
        v[r + 2] = 19; // a different primary
        let gv = GameVariant::parse(&v).unwrap();
        let sw = gv.spawn_primary();
        assert!(sw.varies);
        assert!(!sw.is_concrete());
        assert_eq!(sw.index, 6, "the trait fallback is still reported");
    }

    /// The spawn weapon reaches the drawn bodies, and a real `'weap'` report still overrides it.
    ///
    /// This is the whole wiring in one test: `apply_spawn_weapons` writes the resolved index onto
    /// every biped track, `EntityTrack::weapon_at` returns it before the first report, and the
    /// first report takes over from its own tick.
    #[test]
    fn the_spawn_weapon_reaches_the_tracks_and_a_pickup_overrides_it() {
        use crate::film_decode::EntityTrack;
        let gv = GameVariant::parse(&synthetic()).unwrap();
        // One visible loadout naming primary 11; the map's `motl` entry 11 is tag 0x700, which
        // sits at index 2 of this map's `weap` group list.
        let mut names = VariantNames::default();
        names.weapons = (0..12).map(|i| (0x6F0 + i as u32, format!("objects\\weapons\\w{i}"))).collect();
        let weap_tags: Vec<u32> = vec![0x6EE, 0x6EF, 0x6FB, 0x700];
        let mut tracks = vec![
            EntityTrack { net_id: 1, entity_type: 11, ..Default::default() },
            // A vehicle: must NOT be given a hand weapon.
            EntityTrack { net_id: 2, entity_type: 15, ..Default::default() },
            // A biped that picks something up at tick 300.
            EntityTrack { net_id: 3, entity_type: 11, weapons: vec![(300, 1)], ..Default::default() },
        ];
        let got = apply_spawn_weapons(&mut tracks, 11, &gv, &names, &weap_tags);
        assert_eq!(got.rule, SpawnWeaponRule::Stated);
        assert_eq!(got.bodies, 2, "both bipeds, neither vehicle");
        assert_eq!(got.weap_index, Some(2), "motl weapon 11 is tag 0x6FB, weap-list index 2");
        assert_eq!(tracks[0].spawn_weapon, Some(2));
        assert_eq!(tracks[1].spawn_weapon, None);
        // Before any report, the spawn weapon; from the report's own tick, the report.
        assert_eq!(tracks[0].weapon_at(0), Some(2));
        assert_eq!(tracks[0].weapon_at(9999), Some(2));
        assert_eq!(tracks[2].weapon_at(299), Some(2), "spawned with the loadout's weapon");
        assert_eq!(tracks[2].weapon_at(300), Some(1), "the pickup overrides it");
        assert_eq!(tracks[2].weapon_at(5000), Some(1));
    }

    /// Nothing is applied when the variant cannot say, and the reason is reported rather than a
    /// weapon being invented.
    #[test]
    fn an_unstated_spawn_weapon_applies_nothing_and_says_why() {
        use crate::film_decode::EntityTrack;
        let mut v = synthetic();
        // No visible loadout, and the base traits leave the primary unchanged.
        v[off::LOADOUTS] = 0;
        v[off::BASE_TRAITS + 0x15] = WEAPON_UNCHANGED as u8;
        let gv = GameVariant::parse(&v).unwrap();
        let mut tracks = vec![EntityTrack { net_id: 1, entity_type: 11, ..Default::default() }];
        let got = apply_spawn_weapons(&mut tracks, 11, &gv, &VariantNames::default(), &[]);
        assert_eq!(got.weap_index, None);
        assert_eq!(got.bodies, 0);
        assert!(got.why.contains("unchanged"), "{}", got.why);
        assert_eq!(tracks[0].spawn_weapon, None);
        assert_eq!(tracks[0].weapon_at(0), None);
    }

    /// A sentinel is not a statement: a variant that leaves the weapon "unchanged" or defers to
    /// the map must NOT be reported as having stated one, or the card claims a fact it does not
    /// have. Measured on this machine's corpus: 3 of the 24 films checked by hand were labelled
    /// "stated by the variant for every player: unchanged", which is a contradiction.
    #[test]
    fn a_sentinel_spawn_weapon_is_not_reported_as_stated() {
        for sentinel in [WEAPON_UNCHANGED, WEAPON_MAP_DEFAULT, WEAPON_NONE, WEAPON_RANDOM] {
            let mut v = synthetic();
            v[off::LOADOUTS] = 0; // nothing visible to choose between
            v[off::BASE_TRAITS + 0x15] = sentinel as u8;
            let gv = GameVariant::parse(&v).unwrap();
            let (index, rule) = gv.spawn_primary_drawable();
            assert_eq!(index, sentinel);
            assert_eq!(
                rule,
                SpawnWeaponRule::None,
                "{:?} is not a weapon the variant stated",
                weapon_sentinel(sentinel)
            );
        }
        // A real index still is stated.
        let gv = GameVariant::parse(&synthetic()).unwrap();
        assert_eq!(gv.spawn_primary_drawable().1, SpawnWeaponRule::Stated);
    }

    /// A film with no variant: the image is all zero and its metadata content type is not 6.
    #[test]
    fn an_absent_variant_is_none_not_zeroes() {
        assert!(GameVariant::parse(&vec![0u8; off::SIZE]).is_none());
        assert!(GameVariant::parse(&[]).is_none());
    }

    /// Species per player: only `force *` and `set by team` can answer it, and `set by team`
    /// reads the team's own model override. Modelled on the one real film that uses it, where
    /// the odd teams are elite and every recorded player is on team 0.
    #[test]
    fn species_per_team_follows_the_setting() {
        let mut v = synthetic();
        v[off::TEAM_MODEL_OVERRIDE] = 3; // set by team
        v[off::TEAMS + 0x45] = 0; // team 0 spartan
        v[off::TEAMS + off::TEAM_STRIDE + 0x45] = 1; // team 1 elite
        let gv = GameVariant::parse(&v).unwrap();
        assert_eq!(gv.species, PlayerSpecies::SetByTeam);
        assert_eq!(gv.species_for_team(0), DrawnSpecies::Spartan);
        assert_eq!(gv.species_for_team(1), DrawnSpecies::Elite);
        assert!(gv.any_team_is_elite());

        // Under player choice the same team records are INERT.
        v[off::TEAM_MODEL_OVERRIDE] = 0;
        let gv = GameVariant::parse(&v).unwrap();
        assert_eq!(gv.species_for_team(1), DrawnSpecies::NotRecorded);
        assert!(!gv.any_team_is_elite());
        assert!(gv.species.elites_possible(), "player choice cannot rule an Elite out");

        // Forced spartan rules it out outright.
        v[off::TEAM_MODEL_OVERRIDE] = 1;
        let gv = GameVariant::parse(&v).unwrap();
        assert_eq!(gv.species_for_team(1), DrawnSpecies::Spartan);
        assert!(!gv.species.elites_possible());
    }

    // -----------------------------------------------------------------------------------------
    // gates that run against whatever real films this machine has, and skip loudly when it has
    // none. The synthetic tests above pin the offsets; these say the offsets are right about
    // real data.
    // -----------------------------------------------------------------------------------------

    /// Every real film's variant decodes, and **every value lands inside the range the engine
    /// itself declares for that field**.
    ///
    /// This is the corpus gate, and it is a strong one: the trait set packs thirty small enums
    /// into 0x2C bytes, so a one-byte misalignment anywhere in it would put at least one of them
    /// over its cardinality, and there are five trait sets per film. Measured 2026-10-11 over the
    /// 64 films `find_finished_films` sees on this machine: every one decodes, the engine type
    /// agrees with `chdr` on all 64, and there were **0 out-of-range values** in 9,600 enum
    /// checks plus every loadout, team and option field.
    #[test]
    fn real_films_decode_inside_every_declared_range() {
        let films = crate::film::find_finished_films();
        if films.is_empty() {
            eprintln!("SKIPPED: no Reach saved films on this machine");
            return;
        }
        // (field name, value, how many values the tag definition declares for it). The
        // cardinalities come from `reverseme tagdefs reach_tag_test.exe wezr`.
        let check_traits = |t: &PlayerTraits, where_: &str, bad: &mut Vec<String>| {
            let sh = &t.shield;
            let w = &t.weapons;
            let m = &t.movement;
            let a = &t.appearance;
            let se = &t.sensors;
            for (name, v, card) in [
                ("damage resistance", sh.damage_resistance, 13u8),
                ("body multiplier", sh.body_multiplier, 7),
                ("body recharge rate", sh.body_recharge_rate, 15),
                ("shield multiplier", sh.shield_multiplier, 7),
                ("shield recharge rate", sh.shield_recharge_rate, 15),
                ("overshield recharge rate", sh.overshield_recharge_rate, 15),
                ("headshot immunity", sh.headshot_immunity, 3),
                ("vampirism", sh.vampirism, 6),
                ("assassination immunity", sh.assassination_immunity, 3),
                ("damage modifier", w.damage_modifier, 13),
                ("melee damage modifier", w.melee_damage_modifier, 13),
                ("recharging grenades", w.recharging_grenades, 3),
                ("weapon pickup", w.weapon_pickup, 3),
                ("grenade count", w.grenade_count, 15),
                ("infinite ammo", w.infinite_ammo, 4),
                ("equipment usage", w.equipment_usage, 4),
                ("drop equipment on death", w.drop_equipment_on_death, 3),
                ("infinite equipment energy", w.infinite_equipment_energy, 3),
                ("speed multiplier", m.speed_multiplier, 18),
                ("gravity multiplier", m.gravity_multiplier, 14),
                ("vehicle usage", m.vehicle_usage, 9),
                ("double jump", m.double_jump, 4),
                ("active camo", a.active_camo, 6),
                ("waypoint", a.waypoint, 4),
                ("gamertag", a.gamertag, 4),
                ("aura", a.aura, 5),
                ("forced change color", a.forced_change_color, 14),
                ("motion tracker", se.motion_tracker, 5),
                ("motion tracker range", se.motion_tracker_range, 8),
                ("directional damage", se.directional_damage, 3),
            ] {
                if v >= card {
                    bad.push(format!("{where_}.{name} = {v}, max {}", card - 1));
                }
            }
            for (name, v) in [
                ("primary weapon", w.primary_weapon),
                ("secondary weapon", w.secondary_weapon),
                ("equipment", w.equipment),
            ] {
                if !(-4..=63).contains(&v) {
                    bad.push(format!("{where_}.{name} = {v}"));
                }
            }
            if !(-1..=400).contains(&m.jump_multiplier) {
                bad.push(format!("{where_}.jump multiplier = {}", m.jump_multiplier));
            }
        };
        let (mut present, mut absent, mut engine_agree) = (0usize, 0usize, 0usize);
        let mut species: std::collections::BTreeMap<String, usize> = Default::default();
        let mut bad: Vec<String> = Vec::new();
        for p in &films {
            let Ok(f) = crate::film::open(p) else { continue };
            // The file NAME only: the directory above it carries the machine owner's real name.
            let n = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let Some(gv) = f.header.gametype.as_ref() else {
                absent += 1;
                continue;
            };
            present += 1;
            if gv.engine_type == f.content.engine as u32 {
                engine_agree += 1;
            } else {
                bad.push(format!("{n}: engine {} but chdr says {}", gv.engine_type, f.content.engine));
            }
            *species.entry(gv.species.label()).or_default() += 1;
            for (name, t) in [
                ("base", &gv.base_traits),
                ("respawn", &gv.respawn.traits),
                ("red", &gv.red_traits),
                ("blue", &gv.blue_traits),
                ("yellow", &gv.yellow_traits),
            ] {
                check_traits(t, &format!("{n}.{name}"), &mut bad);
            }
            if gv.misc.flags > 0x0F { bad.push(format!("{n}: misc flags {:#x}", gv.misc.flags)) }
            if gv.social.flags > 0x1F { bad.push(format!("{n}: social flags {:#x}", gv.social.flags)) }
            if gv.social.team_changing > 2 { bad.push(format!("{n}: team changing {}", gv.social.team_changing)) }
            if gv.map.flags > 0x3F { bad.push(format!("{n}: map flags {:#x}", gv.map.flags)) }
            if gv.loadout_flags > 3 { bad.push(format!("{n}: loadout flags {}", gv.loadout_flags)) }
            if matches!(gv.species, PlayerSpecies::Unknown(_)) { bad.push(format!("{n}: species {:?}", gv.species)) }
            if gv.designator_switch > 2 { bad.push(format!("{n}: designator switch {}", gv.designator_switch)) }
            if !(0..=60).contains(&gv.misc.round_time_limit) { bad.push(format!("{n}: round time {}", gv.misc.round_time_limit)) }
            if !(0..=31).contains(&gv.misc.number_of_rounds) { bad.push(format!("{n}: rounds {}", gv.misc.number_of_rounds)) }
            if !(0..=5).contains(&gv.misc.early_victory_win_count) { bad.push(format!("{n}: rounds to win {}", gv.misc.early_victory_win_count)) }
            if !(-1..=126).contains(&gv.misc.sudden_death_limit) { bad.push(format!("{n}: sudden death {}", gv.misc.sudden_death_limit)) }
            if !(0..=31).contains(&gv.misc.grace_period) { bad.push(format!("{n}: grace {}", gv.misc.grace_period)) }
            for d in gv.map.powerup_duration {
                if !(0..=120).contains(&d) { bad.push(format!("{n}: powerup duration {d}")) }
            }
            for (i, t) in gv.teams.iter().enumerate() {
                if t.flags > 0x0F { bad.push(format!("{n}: team {i} flags {:#x}", t.flags)) }
                if t.model_override > 1 { bad.push(format!("{n}: team {i} model {}", t.model_override)) }
                if !(-1..=8).contains(&t.initial_designator) { bad.push(format!("{n}: team {i} designator {}", t.initial_designator)) }
                if t.number_of_fireteams > 31 { bad.push(format!("{n}: team {i} fireteams {}", t.number_of_fireteams)) }
            }
            for (i, l) in gv.loadouts.iter().enumerate() {
                if l.flags > 1 { bad.push(format!("{n}: loadout {i} flags {}", l.flags)) }
                if l.grenade_count > 14 { bad.push(format!("{n}: loadout {i} grenades {}", l.grenade_count)) }
                for v in [l.primary_weapon, l.secondary_weapon, l.armor_ability] {
                    if !(-4..=63).contains(&v) { bad.push(format!("{n}: loadout {i} index {v}")) }
                }
                if !(-1..=127).contains(&l.name) { bad.push(format!("{n}: loadout {i} name {}", l.name)) }
            }
        }
        eprintln!(
            "#gamevariant corpus: {} film(s), {present} carry a variant, {absent} do not; \
             engine type agrees with chdr on {engine_agree}/{present}; species {species:?}",
            films.len()
        );
        assert!(present > 0, "no film on this machine carries a game variant");
        assert!(bad.is_empty(), "{} out-of-range value(s):\n  {}", bad.len(), bad.join("\n  "));
        assert_eq!(engine_agree, present, "engine type must match chdr wherever a variant exists");
    }

    /// An INDEPENDENT check on the death-instant oracle, and on the respawn-time offset.
    ///
    /// `#death` placed the moment of death at a body's last reported position because that is the
    /// candidate whose gap to the player's NEXT life is most nearly constant — median +3.1 s to
    /// +8.9 s across twelve films, against "Reach's respawn is 5 s" as the only sanity check
    /// available at the time. The variant now STATES the configured respawn delay, so the two can
    /// be compared per film, which tests the oracle and `off::RESPAWN + 3` at once.
    ///
    /// Reported, never asserted beyond a sign check: the real delay is the configured time plus
    /// the loadout/respawn UI and any growth or penalty the player had accrued, so the measured
    /// median should sit AT or ABOVE the configured time, not on it.
    #[test]
    fn configured_respawn_time_brackets_the_measured_respawn_delay() {
        let films = crate::film::find_finished_films();
        if films.is_empty() {
            eprintln!("SKIPPED: no Reach saved films on this machine");
            return;
        }
        let mut rows: Vec<(String, i8, f32, usize)> = Vec::new();
        for p in films.iter().take(12) {
            let Ok(f) = crate::film::open(p) else { continue };
            let Some(gv) = f.header.gametype.clone() else { continue };
            if f.data.records.is_empty() || f.header.map_variant.is_empty() { continue }
            let idx = crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.entity_messages == 0 { continue }
            let mut by_owner: std::collections::HashMap<u8, Vec<&crate::film_decode::EntityTrack>> =
                Default::default();
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                if let Some(o) = t.owner_raw {
                    by_owner.entry(o).or_default().push(t);
                }
            }
            let mut d: Vec<i64> = Vec::new();
            for (_o, mut lives) in by_owner {
                lives.sort_by_key(|t| t.first_tick);
                for w in lives.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if a.destroyed_tick.is_none() || b.first_tick <= a.first_tick { continue }
                    let Some(&(last_pos, _)) = a.positions.last() else { continue };
                    let gap = b.first_tick as i64 - last_pos as i64;
                    if (0..3600).contains(&gap) { d.push(gap) }
                }
            }
            if d.len() < 20 { continue }
            d.sort_unstable();
            let median = d[d.len() / 2] as f32 / 60.0;
            rows.push((
                p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                gv.respawn.respawn_time,
                median,
                d.len(),
            ));
        }
        if rows.is_empty() {
            eprintln!("SKIPPED: no film with enough paired lives to measure a respawn delay");
            return;
        }
        eprintln!("#gamevariant respawn cross-check (configured vs measured, seconds):");
        for (n, cfg, med, pairs) in &rows {
            eprintln!("  {n:<44} configured {cfg:>3} s   measured median {med:>5.1} s   ({pairs} pairs)");
        }
        // The measured delay can never be NEGATIVE if the death instant is where #death put it.
        for (n, _, med, _) in &rows {
            assert!(*med > 0.0, "{n}: measured respawn delay {med} s is not positive");
        }
    }

    #[test]
    fn sentinels_print_as_words() {
        assert_eq!(weapon_sentinel(WEAPON_UNCHANGED), Some("unchanged"));
        assert_eq!(weapon_sentinel(WEAPON_NONE), Some("none"));
        assert_eq!(weapon_sentinel(WEAPON_MAP_DEFAULT), Some("map default"));
        assert_eq!(weapon_sentinel(WEAPON_RANDOM), Some("random"));
        assert_eq!(weapon_sentinel(0), None);
        let names = VariantNames::default();
        assert_eq!(names.weapon_label(WEAPON_UNCHANGED), "unchanged");
        assert_eq!(names.weapon_label(4), "#4");
    }

    // -----------------------------------------------------------------------------------------
    // #team-colour
    // -----------------------------------------------------------------------------------------

    /// The synthetic variant with team 1 carrying a colour override (`flags` bit 1, packed
    /// `0x00RRGGBB` at +0x48) and team 2 enabled WITHOUT one. Team 0 is enabled with no override.
    fn team_variant(teams_on: bool) -> GameVariant {
        let mut v = synthetic();
        v[off::MISC] = if teams_on { 0b1001 } else { 0b1000 };
        v[off::TEAMS + off::TEAM_STRIDE] = 0b11; // team 1: enabled + override colour
        v[off::TEAMS + off::TEAM_STRIDE + 0x48..off::TEAMS + off::TEAM_STRIDE + 0x4C]
            .copy_from_slice(&0x0033_66FFu32.to_le_bytes());
        v[off::TEAMS + 2 * off::TEAM_STRIDE] = 0b01; // team 2: enabled, no override
        // a colour on team 2 that must be IGNORED because its override bit is clear
        v[off::TEAMS + 2 * off::TEAM_STRIDE + 0x48..off::TEAMS + 2 * off::TEAM_STRIDE + 0x4C]
            .copy_from_slice(&0x00FF_0000u32.to_le_bytes());
        GameVariant::parse(&v).expect("parses")
    }

    /// `sub_1404D1F00` + `sub_1403AC310`: the variant's override is used only when teams are on,
    /// the team is enabled AND flag bit 1 is set; the bytes are `R = bits 16..23` and so on, times
    /// 1/255; both change colours take it; and otherwise the globals' table answers.
    #[test]
    fn a_team_wears_the_variants_override_else_the_globals_default() {
        let defaults = [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]];
        let gv = team_variant(true);
        let c = gv.team_color_override(1).expect("team 1 overrides");
        assert!((c[0] - 0x33 as f32 / 255.0).abs() < 1e-6 && (c[1] - 0x66 as f32 / 255.0).abs() < 1e-6 && (c[2] - 1.0).abs() < 1e-6, "{c:?}");
        assert_eq!(gv.team_skin(1, &defaults), Some([c, c]), "primary AND secondary take the team colour");
        assert_eq!(gv.team_color_override(2), None, "a colour with the override bit clear is not used");
        assert_eq!(gv.team_skin(2, &defaults), Some([defaults[2], defaults[2]]), "no override: the globals' table");
        assert_eq!(gv.team_skin(0, &defaults), Some([defaults[0], defaults[0]]));
        assert_eq!(gv.team_skin(5, &defaults), None, "no entry in either source: keep the customisation");
    }

    /// The swap is driven by the variant's TEAMS BIT -- never by what the gametype is called. The
    /// same bytes with that one bit cleared must stop answering, and a team the variant has not
    /// enabled is not a team.
    #[test]
    fn the_owner_stamp_is_a_team_only_when_the_variant_has_teams() {
        let on = team_variant(true);
        let off_ = team_variant(false);
        for stamp in 0..=8u8 {
            assert_eq!(off_.team_of_owner(stamp), None, "teams off: stamp {stamp} is a player, not a team");
            assert_eq!(off_.team_color_override(1), None);
        }
        assert_eq!(on.team_of_owner(0), None, "wire 0 is nobody");
        assert_eq!(on.team_of_owner(1), Some(0));
        assert_eq!(on.team_of_owner(2), Some(1));
        assert_eq!(on.team_of_owner(3), Some(2));
        assert_eq!(on.team_of_owner(4), None, "team 3 is not enabled in this variant");
        assert_eq!(on.team_of_owner(9), None, "past the eight team records");
    }

    /// **THE GATE for "teams should wear the team's colours".** Every real film, the real `mulg`
    /// team table, the production rule (`team_of_body` -> `team_skin` -> `choose_skin`):
    ///
    /// * in a TEAM game every player body whose owner stamp names an enabled team resolves to that
    ///   team's `[colour, colour]` and NOT to the customisation; team 0 and team 1 differ, and
    ///   (unless the variant overrides them) are the table's red and blue;
    /// * in a free-for-all every body keeps the customisation, whatever the stamp says;
    /// * the swap follows the variant's teams BIT: clearing it on a team film's variant puts every
    ///   body back on the customisation, so it cannot be keyed on anything else.
    ///
    /// IGNORED BY DEFAULT: it opens a Reach cache for `mulg`.
    #[test]
    #[ignore = "opens a Reach cache and reads the saved-film corpus"]
    fn team_games_wear_the_teams_colours_and_free_for_alls_keep_their_own() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let films = crate::film::find_finished_films();
        if films.is_empty() {
            eprintln!("SKIPPED: no Reach saved films on this machine");
            return;
        }
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            let mut d = exe.parent().map(|p| p.to_path_buf());
            for _ in 0..3 {
                if let Some(p) = &d {
                    roots.push(p.clone());
                    d = p.parent().map(|q| q.to_path_buf());
                }
            }
        }
        let Some(dll) = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file()) else {
            eprintln!("SKIPPED: no native payload next to the test binary");
            return;
        };
        let Ok(native) = hms_native::NativeDll::load(&dll) else { return };
        let Some(m) = crate::mapcat::enumerate().into_iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
        }) else {
            eprintln!("SKIPPED: forge_halo is not available");
            return;
        };
        if let Some(d) = m.path.parent() {
            native.set_shared_cache_dirs(&[d.to_path_buf()]);
        }
        let Ok(cache) = native.open_cache(&m.path.to_string_lossy()) else { return };
        let defaults = default_team_colors(&native, cache);
        eprintln!("mulg team colours: {defaults:?}");
        assert!(defaults.len() >= 2, "liveness: the globals' team table has {} entries", defaults.len());
        let (r, b) = (defaults[0], defaults[1]);
        assert!(r[0] > r[1] * 2.0 && r[0] > r[2] * 2.0, "team 0 is the table's red: {r:?}");
        assert!(b[2] > b[0] * 2.0 && b[2] > b[1], "team 1 is the table's blue: {b:?}");

        // Marks a customisation that must NOT win in a team game, and must win in a free-for-all.
        let own: [[f32; 3]; 2] = [[9.0; 3], [9.0; 3]];
        let (mut team_films, mut ffa_films, mut team_bodies, mut ffa_bodies, mut unknown) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut teams_seen = std::collections::BTreeSet::new();
        for p in &films {
            let Ok(f) = crate::film::open(p) else { continue };
            let Some(gv) = f.header.gametype.clone() else { continue };
            if f.header.map_variant.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let skins: Vec<Option<[[f32; 3]; 2]>> = (0..off::TEAM_COUNT as u8).map(|t| gv.team_skin(t, &defaults)).collect();
            let mut off_bit = gv.clone();
            off_bit.misc.flags &= !1;
            let skins_off: Vec<Option<[[f32; 3]; 2]>> = (0..off::TEAM_COUNT as u8).map(|t| off_bit.team_skin(t, &defaults)).collect();
            let team_game = gv.misc.teams_enabled();
            let mut counted = false;
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11 && t.appearance) {
                let table = team_game.then_some(&skins[..]);
                let got = crate::film_play::choose_skin(table, gv.team_of_body(t.appearance, t.owner_raw), Some(own));
                // the same body under the same variant with ONLY the teams bit cleared
                let cleared = crate::film_play::choose_skin(
                    (off_bit.misc.teams_enabled()).then_some(&skins_off[..]),
                    off_bit.team_of_body(t.appearance, t.owner_raw),
                    Some(own),
                );
                assert_eq!(cleared, Some(own), "{}: with the teams bit clear a body keeps its customisation", p.display().to_string().rsplit('/').next().unwrap());
                counted = true;
                if team_game {
                    match gv.team_of_body(t.appearance, t.owner_raw) {
                        Some(team) => {
                            team_bodies += 1;
                            teams_seen.insert(team);
                            let want = gv.team_skin(team, &defaults);
                            assert!(want.is_some(), "team {team} has a colour in the variant or the table");
                            assert_eq!(got, want, "team {team}'s body wears the team colours, not the customisation");
                            assert_ne!(got, Some(own));
                        }
                        None => {
                            unknown += 1;
                            assert_eq!(got, Some(own), "a body whose team is not stated keeps the customisation");
                        }
                    }
                } else {
                    ffa_bodies += 1;
                    assert_eq!(got, Some(own), "a free-for-all body keeps its customisation");
                }
            }
            if counted {
                if team_game { team_films += 1 } else { ffa_films += 1 }
            }
        }
        eprintln!(
            "team colours: {team_films} team film(s) / {team_bodies} bodies on teams {teams_seen:?} ({unknown} with no stated team), {ffa_films} free-for-all film(s) / {ffa_bodies} bodies"
        );
        assert!(team_films >= 1 && team_bodies >= 50, "liveness: {team_films} team films / {team_bodies} bodies");
        assert!(teams_seen.len() >= 2, "both sides of a team game were seen: {teams_seen:?}");
        assert!(ffa_films >= 1 && ffa_bodies >= 30, "liveness: {ffa_films} free-for-all films / {ffa_bodies} bodies");
    }
}
