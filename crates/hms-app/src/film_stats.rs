//! #film-stats — player statistics pulled out of a Reach (MCC) Theater saved film.
//!
//! This module is **data extraction**, not a panel. It is a pure function from a film's bytes to
//! a structure of numbers: nothing in here touches a renderer, a camera, a loaded scene or a
//! frame. The GUI panel, the script verb and the GPU-free `--film-stats` CLI verb are all thin
//! callers of the same [`compute`].
//!
//! ## Every statistic carries its provenance
//!
//! The rest of HMS's Theater surface labels what it shows **decoded** / **derived** /
//! **synthesised**, and so does this. A number here is one of:
//!
//! * **decoded** — read out of the film's stream and validated against an oracle. The entity
//!   DESTROY (list B kind 4) is decoded: across 8 films and 356,957 kind-4 messages, 100.0% name
//!   an entity created earlier in the same film and not yet destroyed, with zero double-destroys.
//! * **derived** — computed from decoded quantities. Deaths are derived, because attributing a
//!   destroyed body to a *player* goes through the owner stamp; distance is derived, because the
//!   stream replicates a position only on some ticks and the path between keys is interpolated.
//! * **not available** — the recording does not carry it. Kills, assists, betrayals, suicides,
//!   medals and damage are in this class, and [`SCORE_NEGATIVE`] says why.
//!
//! ## The score IS here; the kills are not
//!
//! Entity type 5 is the engine's own `game-engine-statborg` score store — 24 slots, of which
//! 0..15 are the player configuration slots — and HMS decodes it exactly, so a clip's scoreboard
//! comes out of the recording with no inference at all. See [`decode_statborg`].
//!
//! **Kills, deaths as the game counts them, assists, betrayals, suicides and medals are not in a
//! film at all.** The engine keeps them in a separate per-player table that no replicated entity
//! carries, so nothing sends them to a client. HMS counts deaths itself from the entity destroys
//! and refuses to invent the rest; [`SCORE_NEGATIVE`] and `probes::probe_counter_sweep` carry the
//! sweep that confirmed it from the file side before the engine confirmed it from the code side.

#[allow(unused_imports)]
use crate::film_decode::decode_statborg;

use crate::film_decode::{EntityTrack, FilmIndex};
use eframe::egui;
use serde::Serialize;

/// The entity type of a player's body (`biped` — the 19 types are named in the binary).
pub const BIPED: u8 = crate::film_anim::BIPED_TYPE;

/// Every statistic's provenance, as the UI and the JSON both print it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Prov {
    /// Read out of the stream and oracle-checked.
    Decoded,
    /// Computed from decoded quantities — honest and reproducible, but this reader's inference.
    Derived,
    /// The recording does not carry it at all.
    NotAvailable,
}

impl Prov {
    pub fn label(self) -> &'static str {
        match self {
            Prov::Decoded => "decoded",
            Prov::Derived => "derived",
            Prov::NotAvailable => "not available",
        }
    }
}

/// How late in the recording a body's destroy stops being read as a death.
///
/// The worry this answers: if a match ending tore every entity down at once, "a body's track
/// ending" would hand every player one spurious death. Destroys inside this window are therefore
/// reported separately as `deaths_at_cut` rather than silently folded in either direction.
///
/// **Measured, and the worry is unfounded** (`probes::probe_death_shape`): over 10 films and
/// 3,530 body destroys, **zero** fell in the last second of the recording. So the column
/// correctly reports 0 everywhere and essentially every destroy is a real death. It is kept
/// because the next film might not be like these ten, and a number that says 0 is worth more
/// than an assumption that it would be.
pub const CUT_WINDOW_TICKS: u32 = 60; // one second

/// How far apart two replicated positions may be for the ticks between them to be CLASSIFIED,
/// ticks.
///
/// This gate is load-bearing and the census is wrong without it. A biped replicates a position
/// on only some ticks (the client is expected to integrate velocity, and masks carrying velocity
/// without position are common), and `film_anim::derive` HOLDS the last known position between
/// keys — so a body whose positions arrive once every ten seconds derives a velocity of zero for
/// the nine seconds in between and reads as standing still. Measured without the gate: a body
/// with 2% position coverage was reported "99% stationary", which is an artefact of the gap, not
/// a player who did not move. With it, only the ticks the recording actually bracketed are
/// classified and the rest are counted as `unclassified_ticks`.
pub const MAX_KEY_GAP: u32 = 30; // half a second

/// Above this horizontal speed a body is not running, it is being launched, thrown or teleported
/// (`film_anim::cuts::SPRINT_MAX`). `top_foot_speed_wu_s` excludes it; `top_speed_wu_s` does not.
pub const FOOT_SPEED_MAX: f32 = crate::film_anim::cuts::SPRINT_MAX;

/// How many of a player's rides the human-readable table lists (longest first). The JSON carries
/// all of them.
pub const RIDES_LISTED: usize = 6;

// ---------------------------------------------------------------------------------------------
// what a caller can supply from a loaded map (still GPU-free)
// ---------------------------------------------------------------------------------------------

/// Names that need an open map cache. Everything else in this module works from the film bytes
/// alone, which is what lets `--film-stats <file>` run with no `--map` and nothing installed.
///
/// Implemented by the CLI (a `SceneController` opened through `cli_scene_controller`, no window
/// and no wgpu device) and by the GUI. When no source is supplied the ids are still reported and
/// the output says the names were not resolved — it never drops the row.
pub trait NameSource {
    /// `(quota index, variant item)` of a Forge spawn point → `(tag id, tag path)`, for `vehi`
    /// tags only. This is how a stream vehicle is identified: by the vehicle spawn point it first
    /// appeared on.
    fn vehicle(&self, quota: u16, item: u8) -> Option<(u32, String)>;
    /// A `'weap'` group index (what a unit's dirty block carries) → that weapon's tag path.
    fn weapon(&self, index: u32) -> Option<String>;
    /// #gamevariant  The open map's `motl` / `lgtd` tables, which turn the game variant's
    /// loadout and trait INDICES into weapon names. `None` without a map, in which case the
    /// gametype block still reports the indices.
    fn variant_names(&self) -> Option<&crate::film_variant::VariantNames> {
        None
    }
    /// #film-action  A creation message's 14-bit network tag reference (`EntityTrack::def_wire`)
    /// → `(tag id, 4-char group, tag path)`: **the object definition the entity was spawned
    /// from**. `None` without a map, or when the map's own `.map` file could not be read for the
    /// per-tag datum salts the table has to be sorted by.
    fn definition(&self, _wire: u32) -> Option<(u32, String, String)> {
        None
    }
}

/// A `NameSource` that resolves nothing — the no-`--map` case, spelled out rather than an
/// `Option<&dyn ...>` sprinkled through the computation.
pub struct NoNames;
impl NameSource for NoNames {
    fn vehicle(&self, _q: u16, _i: u8) -> Option<(u32, String)> {
        None
    }
    fn weapon(&self, _i: u32) -> Option<String> {
        None
    }
}

/// Vehicle and weapon names resolved out of an OPEN MAP CACHE — and nothing else.
///
/// Built once from the native parser's own tables, so it holds no borrow of the cache and can be
/// handed to [`compute`] freely. There is **no GPU anywhere on this path**: the CLI opens the
/// cache through `cli_scene_controller`, which loads only the native library.
pub struct MapNames {
    /// `(palette_index, entry_within)` in QUOTA order — the index a variant object's `folder`
    /// field is.
    order: Vec<(u32, u32)>,
    /// `(palette_index, entry_within, variant_within, tag id, tag class, tag path)`.
    entries: Vec<(u32, u32, u32, u32, String, String)>,
    /// `'weap'` group tag paths, in group-index order: exactly the index a unit's dirty block
    /// carries.
    weap: Vec<String>,
    /// #gamevariant  `motl` weapons / equipment and `lgtd` loadout names.
    variant: crate::film_variant::VariantNames,
    /// #film-action  The engine's network tag-reference table for this cache, and the resolved
    /// `(tag id, group, path)` of every entry. `None` when the map's `.map` file was not readable
    /// (the salts live in it, not in any native export).
    reftable: Option<(crate::film_action::RefTable, Vec<(u32, String, String)>)>,
}

impl MapNames {
    /// Read the tables out of an open cache. `native` + `cache` come from a `SceneController`
    /// that opened a map — in the CLI, one built by `cli_scene_controller()` with no window.
    pub fn build(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> MapNames {
        MapNames::build_with_map(native, cache, None)
    }

    /// As [`MapNames::build`], with the path of the cache's own `.map` file so the network
    /// tag-reference table can be built — which is what turns a stream entity's creation
    /// message into `bipd elite` rather than an index.
    pub fn build_with_map(
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        map_path: Option<&std::path::Path>,
    ) -> MapNames {
        let palette = native
            .find_scenario(cache)
            .map(|scnr| native.scnr_forge_palette_full(cache, scnr))
            .unwrap_or_default();
        let mut order: Vec<(u32, u32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u32, u32)> = Default::default();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if seen.insert(k) {
                order.push(k);
            }
        }
        let entries = palette
            .iter()
            .map(|e| {
                (
                    e.palette_index,
                    e.entry_within,
                    e.variant_within,
                    e.tag_short,
                    native.tag_class(cache, e.tag_short).unwrap_or_default(),
                    native.tag_name(cache, e.tag_short).unwrap_or_default(),
                )
            })
            .collect();
        let weap = native
            .tags_by_group(cache, "weap")
            .into_iter()
            .map(|t| native.tag_name(cache, t).unwrap_or_default())
            .collect();
        let variant = crate::film_variant::VariantNames::build(native, cache);
        // #film-action  The 9-group network tag-reference table. Costs one enumeration per
        // concrete group plus a read of the map file's 8-byte-per-tag instance array.
        let reftable = map_path
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|bytes| {
                let mut tags: Vec<(u32, String)> = Vec::new();
                for g in crate::film_action::RefTable::CONCRETE_GROUPS {
                    for t in native.tags_by_group(cache, g) {
                        tags.push((t, (*g).to_string()));
                    }
                }
                let t = crate::film_action::RefTable::for_map(&bytes, &tags)?;
                let resolved = t
                    .flat
                    .iter()
                    .map(|&id| {
                        (
                            id,
                            native.tag_class(cache, id).unwrap_or_default(),
                            native.tag_name(cache, id).unwrap_or_default(),
                        )
                    })
                    .collect();
                Some((t, resolved))
            });
        MapNames { order, entries, weap, variant, reftable }
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty() && self.weap.is_empty()
    }
}

impl NameSource for MapNames {
    fn vehicle(&self, quota: u16, item: u8) -> Option<(u32, String)> {
        let &(pi, ew) = self.order.get(quota as usize)?;
        let e = self
            .entries
            .iter()
            .find(|e| e.0 == pi && e.1 == ew && e.2 == item as u32)
            .or_else(|| self.entries.iter().find(|e| e.0 == pi && e.1 == ew))?;
        // Only VEHICLES: anything else resolving would put scenery in the vehicle table.
        if e.4 != "vehi" {
            return None;
        }
        Some((e.3, e.5.clone()))
    }
    fn weapon(&self, index: u32) -> Option<String> {
        self.weap.get(index as usize).cloned().filter(|s| !s.is_empty())
    }
    fn definition(&self, wire: u32) -> Option<(u32, String, String)> {
        let (_t, resolved) = self.reftable.as_ref()?;
        if wire == 0 {
            return None;
        }
        resolved.get(wire as usize - 1).cloned()
    }
    fn variant_names(&self) -> Option<&crate::film_variant::VariantNames> {
        Some(&self.variant)
    }
}

// ---------------------------------------------------------------------------------------------
// the output
// ---------------------------------------------------------------------------------------------

/// One ride, as the per-player table reports it.
#[derive(Clone, Debug, Serialize)]
pub struct RideRow {
    /// The vehicle's network entity id.
    pub vehicle_net_id: u32,
    /// The vehicle's tag leaf name (`warthog`, `mongoose`, …), or `null` with no map open.
    pub vehicle: Option<String>,
    pub from_tick: u32,
    pub to_tick: u32,
    pub seconds: f32,
}

/// One weapon a player's body was recorded holding.
#[derive(Clone, Debug, Serialize)]
pub struct WeaponRow {
    /// The `'weap'` group index out of the unit's dirty block.
    pub weap_index: u32,
    /// Its tag path, or `null` with no map open.
    pub weapon: Option<String>,
    /// The first tick a body of this player was reported holding it.
    pub first_tick: u32,
}

/// One row of the recorded SCOREBOARD.
///
/// This comes out of entity type 5, `game-engine-statborg` — the engine's own score store, 24
/// slots of which 0..15 are the player configuration slots and 16..23 the teams. Its index IS
/// the player slot, so a score needs no owner-group bridge and lands on a named player directly.
///
/// A film only carries the score UPDATES the recording was sent, so a clip whose gametype never
/// awarded a point has no rows at all. That is the recording's limit, not a decode failure, and
/// a missing row is reported as "the recording never said" rather than as zero.
#[derive(Clone, Debug, Serialize)]
pub struct ScoreRow {
    /// Statborg index: 0..15 is a player configuration slot, 16..23 is a team.
    pub statborg_index: u8,
    /// `"player"` or `"team"`.
    pub kind: &'static str,
    /// The player slot (for a player row) or the team index (for a team row).
    pub slot: u8,
    pub gamertag: Option<String>,
    /// The score the game would DISPLAY: the carried-in total plus the live round, or the
    /// carried-in total alone once `final` is set. That is the engine's own rule.
    pub score: i32,
    /// `round_score` — statistic index 0, the one Megalo sets.
    pub round_score: i16,
    /// Score carried in from COMPLETED rounds.
    pub carried_score: i16,
    pub rounds_won: i16,
    /// The engine's `final` flag: the round is over and the total is authoritative.
    pub is_final: bool,
    /// How many score updates the recording carried for this slot.
    pub updates: usize,
}

/// Everything this module can say about one player.
///
/// Keyed by **owner group** — one player's bodies across every respawn — because that is the one
/// identity a film states outright (4 bits at bit offset 272 of a type-11 creation payload). The
/// configuration `slot` and the `gamertag` are present only where the recording names them, and
/// a film's header roster is who was present when the header was written, not everyone who
/// played. A player with no gamertag is still a complete row.
#[derive(Clone, Debug, Serialize)]
pub struct PlayerStats {
    /// The raw owner stamp, 1..15. This is the row's stable key.
    pub owner: u8,
    /// The film-header configuration slot, where this film's recorder anchor corroborated
    /// `slot = owner - 1`. `null` means the recording holds no name for this player.
    pub slot: Option<u8>,
    pub gamertag: Option<String>,
    pub service_tag: Option<String>,
    pub team: Option<u8>,
    /// The entity id of this player's first body — a usable label when there is no gamertag.
    pub first_body_net_id: Option<u32>,
    /// #film-action  The BIPED DEFINITION this player's bodies were spawned from, as the stream
    /// states it in every creation message — `characters\spartans\spartans`,
    /// `characters\elite\elite`, `vehicles\monitor\monitor_editor`. `None` without a map
    /// (`--map`), in which case `biped_wire` still carries the raw reference. This is what makes
    /// an Elite draw as an Elite instead of being assumed a Spartan.
    pub biped: Option<String>,
    /// The raw 14-bit network tag reference, `index + 1` into the engine's 9-group table.
    pub biped_wire: Option<u32>,
    /// True when the owner stamp is 0, which means **nobody**: the stamp is 1-based and 0 is the
    /// engine's "no player". These bodies are real and fully measured, but they are not
    /// attributed to a player, so they are excluded from the named-player fraction.
    pub unowned: bool,

    /// Bodies this player had: one per spawn. **derived** (the owner grouping).
    pub lives: usize,
    /// Bodies destroyed before the final cut. **derived** — the destroy itself is decoded.
    pub deaths: usize,
    /// Bodies destroyed inside the last [`CUT_WINDOW_TICKS`] of the recording. These are the
    /// end-of-match teardown and are NOT counted as deaths; reported so the two add up.
    pub deaths_at_cut: usize,
    /// Bodies still alive when the recording stopped.
    pub bodies_alive_at_end: usize,
    pub first_spawn_tick: Option<u32>,
    pub last_death_tick: Option<u32>,
    /// Union of this player's bodies' lifetimes, ticks (overlapping corpses counted once).
    pub alive_ticks: u32,
    /// Sum of the bodies' lifetimes, ticks. Differs from `alive_ticks` exactly where a corpse
    /// lingered past the respawn, and it is the right denominator for `position_ticks`.
    pub body_ticks: u32,
    /// The longest single life, ticks.
    pub longest_life_ticks: u32,

    /// Distance travelled, world units. **derived**, and a LOWER BOUND: the stream replicates a
    /// position on only some ticks, so the path between keys cuts corners.
    pub distance_wu: f32,
    /// How many of this player's alive ticks carried a replicated position. The honesty number
    /// for `distance_wu` and `top_speed_wu_s`.
    pub position_ticks: u32,
    /// Top horizontal speed, wu/s, over the classified samples. A respawn teleport is rejected
    /// (the derivative gate drops anything over 40 wu/s), but a man-cannon launch or a
    /// teleporter is NOT — that is what `top_foot_speed_wu_s` is for.
    pub top_speed_wu_s: f32,
    /// Top horizontal speed below [`FOOT_SPEED_MAX`]: the fastest this player actually moved on
    /// foot, with launches and teleports excluded.
    pub top_foot_speed_wu_s: f32,

    /// Ticks the gait census could classify at all: bracketed by two replicated positions no
    /// more than [`MAX_KEY_GAP`] apart, with a decoded body facing. Every `*_ticks` figure below
    /// is out of THIS, not out of `alive_ticks`.
    pub gait_ticks: u32,
    /// Ticks the recording does not support a classification for. `gait_ticks +
    /// unclassified_ticks == body_ticks`, so the two always add up.
    pub unclassified_ticks: u32,
    pub sprinting_ticks: u32,
    pub airborne_ticks: u32,
    pub stationary_ticks: u32,
    pub running_ticks: u32,
    pub walking_ticks: u32,
    pub seated_ticks: u32,

    /// Ticks spent in a vehicle, from the ride detector.
    pub vehicle_ticks: u32,
    pub rides: Vec<RideRow>,

    /// Weapons the recording named in this player's hands. See [`Coverage::weapon_bodies`] —
    /// the field reports only on CHANGE, never at spawn, so this is empty for most players and
    /// empty for every player on half of all films. An empty list does not mean unarmed.
    pub weapons: Vec<WeaponRow>,
    /// How many of this player's bodies ever named a weapon.
    pub weapon_bodies: usize,

    /// Per-tick aim records this player's configuration slot carried. **decoded** (13-bit yaw,
    /// 12-bit pitch). Zero when the player has no slot.
    pub aim_samples: usize,

    /// The player's final SCORE, out of the engine's own score store (`game-engine-statborg`).
    /// **decoded.** `None` means the recording carried no score update for this slot — which is
    /// not the same as zero, and is the usual case on a social clip whose gametype never scored.
    pub score: Option<i32>,
    pub round_score: Option<i16>,
    pub rounds_won: Option<i16>,
    /// How many score updates the recording carried for this player.
    pub score_updates: usize,
}

impl PlayerStats {
    /// The label a human-readable row should use: the gamertag when the recording stored one, the
    /// first body's entity id otherwise.
    pub fn label(&self) -> String {
        match (&self.gamertag, self.first_body_net_id) {
            (Some(g), _) if !g.is_empty() => g.clone(),
            (_, Some(n)) => format!("{n:#010x}"),
            _ => format!("owner {}", self.owner),
        }
    }
    /// Deaths per minute of time alive — the one rate that needs no kill counter.
    pub fn deaths_per_minute(&self) -> f32 {
        let m = self.alive_ticks as f32 / 3600.0;
        if m <= 0.0 {
            0.0
        } else {
            self.deaths as f32 / m
        }
    }
}

/// Clip-wide coverage: what fraction of the thing each statistic is computed from actually
/// decoded. A statistic that is only true for some players has to say for how many.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Coverage {
    pub records: usize,
    pub records_decoded: usize,
    pub entity_messages: usize,
    /// Player bodies (type-11 entities) in the clip.
    pub bodies: usize,
    /// Bodies carrying a readable owner stamp — the ones a player row can be built from.
    pub bodies_with_owner: usize,
    /// Bodies that ever replicated a position.
    pub bodies_with_position: usize,
    /// Bodies that ever reported a body facing (the gait census needs one).
    pub bodies_with_facing: usize,
    /// Bodies that ever named the weapon in their hands.
    pub weapon_bodies: usize,
    /// Owner groups whose player could be NAMED from the film header.
    pub named_players: usize,
    /// Player slots the film header names at all.
    pub header_players: usize,
    /// Player slots that carried per-tick records. An INDEPENDENT count of how many players were
    /// in the game, which neither the header roster nor the owner stamp supplies — so a film
    /// where this and `owner_groups` disagree badly is one where the owner grouping should not be
    /// trusted, and the output says so.
    pub slots_with_records: usize,
    /// Owner groups found in the stream (rows in `players`, including an unowned one).
    pub owner_groups: usize,
    /// Type-11 destroy messages in the clip (deaths before attribution).
    pub body_destroys: usize,
    /// …of which fell inside the final [`CUT_WINDOW_TICKS`].
    pub body_destroys_at_cut: usize,
    /// `game-engine-statborg` messages in the clip, and how many passed BOTH free oracles (the
    /// rebuilt presence mask equal to the message's own, and the payload consumed to within the
    /// writer's byte padding). A gap between the two means the score walk desynchronised and the
    /// scores should not be trusted.
    pub statborg_messages: usize,
    pub statborg_decoded: usize,
    /// Statborg slots that ever reported a score.
    pub score_slots: usize,
}

/// The clip's own identity, so a JSON row is self-describing without the file name (which, on
/// this machine, is under a path that must not be reproduced).
#[derive(Clone, Debug, Default, Serialize)]
pub struct ClipInfo {
    pub title: String,
    pub gametype: String,
    /// `s_content_item_metadata+0x2A` — 0 none, 1 CTF, 2 slayer, …
    pub engine_type: u8,
    pub map_id: u32,
    pub scenario: String,
    pub ticks: u32,
    pub seconds: f32,
    pub recorded_unix: u64,
    /// The configuration slot of the player who recorded the film.
    pub recorder_slot: Option<u8>,
    /// True when this film's recorder anchor corroborated `slot = owner - 1`, which is what
    /// licenses putting a gamertag on every owner group rather than only on the recorder's.
    pub owner_rule_corroborated: bool,
    /// WHICH anchor did it: `author_record` (proven, every finished film) or `live_aim`
    /// (**derived**, the only anchor an in-progress `.temp` has). A consumer that only wants
    /// stated facts should filter on this.
    pub owner_anchor: crate::film_decode::OwnerAnchor,
    /// True when the names on this clip came from a DERIVED anchor and are therefore unverified.
    pub names_unverified: bool,
    /// The one player slot whose aim is LIVE — the recording player. Present on every film,
    /// author record or not, and the only way an in-progress `.temp` identifies its own recorder.
    pub live_aim_slot: Option<u8>,
    /// A film records either entity replication or player ACTIONS, never both. An action-only
    /// film has no entity stream, so there are no bodies and no statistics to pull.
    pub has_entity_stream: bool,
}

/// #gamevariant  One loadout of a palette, as the JSON carries it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LoadoutRow {
    /// 0..5. The engine calls these loadout PALETTES; a gametype script assigns one to a player.
    pub tier: usize,
    /// 0..4 within the tier.
    pub slot: usize,
    pub visible: bool,
    /// Resolved through the map's `lgtd` tag; empty without a map.
    pub name: String,
    pub primary_weapon: String,
    pub secondary_weapon: String,
    pub armor_ability: String,
    pub grenades: String,
    /// The raw indices, so a consumer is never stuck with a label.
    pub raw: [i8; 4],
}

/// #gamevariant  The DECLARED settings of the game variant the film was recorded under.
///
/// Every field here is **decoded** — read out of the `c_game_variant` image the film embeds, at
/// offsets taken from the engine's own bitstream decode. The only derived field is
/// `spawn_primary_weapon`, which applies the engine's loadout-over-traits precedence; it is
/// reported as not available when the answer depends on a loadout choice the film never records.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GametypeInfo {
    pub engine_type: u32,
    pub engine: String,
    /// `none (player choice)` / `force spartan` / `force elite` / `set by team` /
    /// `set by designator` — the engine's own names.
    pub player_species: String,
    pub species_allows_elites: bool,
    /// Which teams the variant actually spawns as Elites. Empty unless `player_species` is
    /// `set by team` — a team's own `model override` is inert under any other setting.
    pub elite_teams: Vec<usize>,
    /// Why the species is or is not knowable for this clip, in plain words.
    pub species_note: String,
    pub teams_enabled: bool,
    pub teams_active: usize,
    pub perfection_enabled: bool,
    pub number_of_rounds: i8,
    pub round_time_limit_minutes: i8,
    pub rounds_to_win: i8,
    pub sudden_death_seconds: i16,
    pub grace_period_seconds: i16,
    pub respawn_time_seconds: i8,
    pub respawn_growth_seconds: i8,
    pub suicide_penalty_seconds: i8,
    pub betrayal_penalty_seconds: i8,
    pub lives_per_round: i8,
    pub team_lives_per_round: i8,
    pub loadout_camera_time_seconds: i8,
    pub respawn_traits_duration_seconds: i8,
    pub friendly_fire: bool,
    pub betrayal_booting: bool,
    /// 0 disabled, 1 enabled, 2 balancing only.
    pub team_changing: u16,
    pub weapon_set: i16,
    pub vehicle_set: i16,
    pub powerup_duration_seconds: [i8; 3],
    pub spartan_loadouts_enabled: bool,
    pub elite_loadouts_enabled: bool,
    /// The base player traits that apply to everybody, in the fields that matter most.
    pub base_primary_weapon: String,
    pub base_secondary_weapon: String,
    pub base_armor_ability: String,
    pub base_grenades: String,
    pub base_damage_resistance: String,
    pub base_shield_multiplier: u8,
    pub base_headshot_immunity: String,
    pub base_assassination_immunity: String,
    pub base_jump_multiplier: i32,
    /// What every player spawns holding, when the variant says the same thing for all of them.
    /// Empty when it does not — `spawn_primary_weapon_note` then says why.
    pub spawn_primary_weapon: String,
    pub spawn_primary_weapon_note: String,
    /// What HMS actually DRAWS in every body's hands, which is answerable more often than
    /// `spawn_primary_weapon`: when the selectable loadouts disagree this is the most common one
    /// among them. `drawn_primary_weapon_rule` says which rule produced it.
    pub drawn_primary_weapon: String,
    pub drawn_primary_weapon_rule: &'static str,
    pub loadouts: Vec<LoadoutRow>,
    /// False when the weapon and loadout names are indices because no map was open.
    pub names_resolved: bool,
}

impl GametypeInfo {
    fn build(gv: &crate::film_variant::GameVariant, names: &dyn NameSource) -> GametypeInfo {
        use crate::film_variant as fv;
        let empty = fv::VariantNames::default();
        let n = names.variant_names().unwrap_or(&empty);
        let t = &gv.base_traits;
        let sw = gv.spawn_primary();
        let drawn = gv.spawn_primary_drawable();
        let loadouts = gv
            .loadouts
            .iter()
            .enumerate()
            .filter(|(_, l)| l.visible() || !l.is_empty())
            .map(|(i, l)| LoadoutRow {
                tier: i / fv::off::LOADOUTS_PER_TIER,
                slot: i % fv::off::LOADOUTS_PER_TIER,
                visible: l.visible(),
                name: n.loadout_name(l.name),
                primary_weapon: n.weapon_label(l.primary_weapon),
                secondary_weapon: n.weapon_label(l.secondary_weapon),
                armor_ability: n.equipment_label(l.armor_ability),
                grenades: fv::grenade_count_label(l.grenade_count),
                raw: [l.primary_weapon, l.secondary_weapon, l.armor_ability, l.grenade_count as i8],
            })
            .collect();
        GametypeInfo {
            engine_type: gv.engine_type,
            engine: gv.engine_name().to_string(),
            player_species: gv.species.label(),
            species_allows_elites: gv.species.elites_possible(),
            elite_teams: if gv.any_team_is_elite() {
                gv.teams
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.enabled() && t.model_override == 1)
                    .map(|(i, _)| i)
                    .collect()
            } else {
                Vec::new()
            },
            species_note: match gv.species {
                fv::PlayerSpecies::ForceSpartan =>
                    "Every player spawned as a Spartan: the variant forces it.".to_string(),
                fv::PlayerSpecies::ForceElite =>
                    "Every player spawned as an ELITE: the variant forces it.".to_string(),
                fv::PlayerSpecies::SetByTeam => format!(
                    "Species came from each player's TEAM. Elite team(s): {}. A body is drawn as \
                     an Elite only for those teams.",
                    if gv.any_team_is_elite() { "see elite_teams" } else { "none" }
                ),
                fv::PlayerSpecies::SetByDesignator =>
                    "Species came from each team's DESIGNATOR (defence spartan, offence elite). \
                     Which designator value is the defence role is not established here, so the \
                     species is reported as not available rather than guessed."
                        .to_string(),
                _ => "Species came from each player's own CUSTOMISATION options, which is \
                      profile data: the SETTING defers to it, so neither the game variant nor the \
                      film's 160-byte player records can answer which species a player picked. \
                      The RECORDING answers it per body instead: every spawn message names the \
                      object definition the body was spawned from, so each body is drawn as the \
                      Spartan, Elite or Monitor the stream says it was. See the `bipeds` census \
                      and each player's `biped`."
                    .to_string(),
            },
            teams_enabled: gv.misc.teams_enabled(),
            teams_active: gv.teams.iter().filter(|t| t.enabled()).count(),
            perfection_enabled: gv.misc.perfection_enabled(),
            number_of_rounds: gv.misc.number_of_rounds,
            round_time_limit_minutes: gv.misc.round_time_limit,
            rounds_to_win: gv.misc.early_victory_win_count,
            sudden_death_seconds: gv.misc.sudden_death_limit,
            grace_period_seconds: gv.misc.grace_period,
            respawn_time_seconds: gv.respawn.respawn_time,
            respawn_growth_seconds: gv.respawn.respawn_growth,
            suicide_penalty_seconds: gv.respawn.suicide_penalty,
            betrayal_penalty_seconds: gv.respawn.betrayal_penalty,
            lives_per_round: gv.respawn.lives_per_round,
            team_lives_per_round: gv.respawn.team_lives_per_round,
            loadout_camera_time_seconds: gv.respawn.loadout_camera_time,
            respawn_traits_duration_seconds: gv.respawn.traits_duration,
            friendly_fire: gv.social.friendly_fire(),
            betrayal_booting: gv.social.betrayal_booting(),
            team_changing: gv.social.team_changing,
            weapon_set: gv.map.weapon_set,
            vehicle_set: gv.map.vehicle_set,
            powerup_duration_seconds: gv.map.powerup_duration,
            spartan_loadouts_enabled: gv.loadout_flags & 1 != 0,
            elite_loadouts_enabled: gv.loadout_flags & 2 != 0,
            base_primary_weapon: n.weapon_label(t.weapons.primary_weapon),
            base_secondary_weapon: n.weapon_label(t.weapons.secondary_weapon),
            base_armor_ability: n.equipment_label(t.weapons.equipment),
            base_grenades: fv::grenade_count_label(t.weapons.grenade_count),
            base_damage_resistance: fv::damage_resistance_label(t.shield.damage_resistance),
            base_shield_multiplier: t.shield.shield_multiplier,
            base_headshot_immunity: fv::boolean_label(t.shield.headshot_immunity).to_string(),
            base_assassination_immunity: fv::boolean_label(t.shield.assassination_immunity).to_string(),
            base_jump_multiplier: t.movement.jump_multiplier,
            spawn_primary_weapon: if sw.varies { String::new() } else { n.weapon_label(sw.index) },
            spawn_primary_weapon_note: if sw.varies {
                "The visible loadouts name different primary weapons and a film does not record \
                 which loadout a player chose, so the spawn weapon is not available. The base \
                 player traits' weapon is reported as `base_primary_weapon`."
                    .to_string()
            } else if sw.from_loadout {
                "From the loadout palette: every visible loadout names this weapon, so the \
                 player's choice cannot change it."
                    .to_string()
            } else {
                "From the base player traits, which apply to every player. A loadout that names \
                 its own primary weapon would override it; none of the visible ones does."
                    .to_string()
            },
            drawn_primary_weapon: n.weapon_label(drawn.0),
            drawn_primary_weapon_rule: match drawn.1 {
                fv::SpawnWeaponRule::Stated => "stated by the variant for every player",
                fv::SpawnWeaponRule::MostCommonLoadout =>
                    "the most common selectable loadout; a player's own choice is not recorded",
                // Not a statement: say WHICH way the variant declined, so a consumer is not
                // left reading "not available" as "we failed to decode it".
                fv::SpawnWeaponRule::None => match fv::weapon_sentinel(drawn.0) {
                    Some("unchanged") =>
                        "not stated: the gametype leaves every player's weapon unchanged",
                    Some("map default") =>
                        "not stated: the gametype defers to the map's own default weapon, which \
                         is not in the film",
                    Some("none") => "the gametype spawns players with no weapon at all",
                    Some("random") =>
                        "not stated: the gametype picks each spawn weapon at random",
                    _ => "not available",
                },
            },
            loadouts,
            names_resolved: names.variant_names().map_or(false, |v| !v.is_empty()),
        }
    }
}

/// #film-action  One row of THE KILL FEED, read off the stream's `incident` events.
///
/// The engine broadcasts a Reach medal/incident as list B command kind 1, event type 14, with
/// a 32-bit string id and up to two object references: the player who earned it and the other
/// party. So who killed whom, with what kind of kill and at which tick, is **decoded**, not
/// derived -- which is new, and is the one thing the old "kills are not in a film" note was
/// wrong about (it was looking at the per-player tables, which really are absent).
#[derive(Clone, Debug, Serialize)]
pub struct KillRow {
    pub tick: u32,
    pub seconds: f32,
    /// The incident's name from the engine's own string table, e.g. `kill`, `melee_kill`,
    /// `finishing_move_kill`, `splatter_kill`, `headshot_kill`, `suicide`.
    pub incident: String,
    /// Raw string id, for an incident this build's table does not name.
    pub incident_id: u32,
    /// The killer's network entity, and their player slot / gamertag when the owner bridge
    /// named one.
    pub killer_net_id: Option<u32>,
    pub killer_slot: Option<u8>,
    pub killer: Option<String>,
    pub victim_net_id: Option<u32>,
    pub victim_slot: Option<u8>,
    pub victim: Option<String>,
    /// True when the incident is one of the ASSASSINATION ids. Measured: on the three gametypes
    /// whose titles are about assassination, 34 of 34 kills are `finishing_move_kill`, against
    /// 68 of 1,904 (3.6%) on every other clip.
    pub assassination: bool,
}

/// #film-action  WHO KILLED EACH BODY, and from which channel — the coverage number that
/// explains why Theater can only stage a handful of the kills it is asked about.
///
/// **The kill feed covers the RECORDER and nobody else.** Measured on `Chill Assassinations.`:
/// 163 bodies destroyed, 16 kill-feed rows, and all 16 naming the same killer — a player with
/// one life and no deaths, i.e. whoever recorded the film. Medals are personal and a film is one
/// client's view, so a kill the recorder was not part of never reaches the `incident` channel.
///
/// **The DAMAGE record covers nearly everyone.** `damage_aftermath` carries two object
/// references and the generic kind-1 header reader already decodes both: reference 0 is the
/// object damaged and reference 1 is the responsible unit. The last one naming a body before it
/// died is a killer for a death the feed never mentions.
///
/// The two provenances are kept apart on purpose and `agree` is the cross-check between them:
/// the feed is **decoded**, the damage attribution is **derived from the damage record**.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DeathAttribution {
    /// Bodies whose life ended (`film_anim::death_tick`).
    pub deaths: usize,
    /// ...that the kill feed names a killer for. DECODED.
    pub from_kill_feed: usize,
    /// ...that only the damage record names a killer for. DERIVED.
    pub from_damage_record: usize,
    /// ...that neither names anyone for.
    pub unattributed: usize,
    /// Distinct killers each channel names.
    pub feed_killers: usize,
    pub damage_killers: usize,
    /// Deaths both channels name a killer for, and whether they agree. This is the whole
    /// evidence that the damage record's second reference really is the killer.
    pub agree: usize,
    pub disagree: usize,
}

/// #film-action  A biped getting into or out of a vehicle, with the SEAT index -- which the
/// stream states (a bare 6-bit field, `sub_1803B5644`) rather than HMS having to guess it from
/// geometry.
#[derive(Clone, Debug, Serialize)]
pub struct VehicleTransitionRow {
    pub tick: u32,
    pub seconds: f32,
    /// `biped_enter_vehicle`, `biped_board_vehicle`, `biped_exit_vehicle` or
    /// `biped_vehicle_evict`.
    pub event: &'static str,
    pub biped_net_id: Option<u32>,
    pub slot: Option<u8>,
    pub player: Option<String>,
    pub vehicle_net_id: Option<u32>,
    pub vehicle: Option<String>,
    pub seat: Option<u8>,
}

/// #film-action  Every event type the recording carried, by name, with its count. The whole
/// channel, so a consumer can see what is there without re-deriving it.
#[derive(Clone, Debug, Serialize)]
pub struct ActionChannel {
    /// Kind-1 commands seen.
    pub commands: usize,
    /// How many the decoder accepted (the engine's own range checks: type <= 0x47, refs <= 2).
    pub decoded: usize,
    /// Event name → count, every type present.
    pub by_event: std::collections::BTreeMap<String, u64>,
    /// Incident name (or `0x…` for an unnamed id) → count.
    pub by_incident: std::collections::BTreeMap<String, u64>,
    /// `biped_assassinate` events. These are the engine's full synchronised-action record (the
    /// action index, the position, two orientation vectors and a damage record) but they are a
    /// client→host REQUEST, so only a film recorded BY THE HOST carries them: 15 of the corpus's
    /// 16 are in one such film, which carries no replicated bipeds at all.
    pub assassinate_events: usize,
}

/// A whole film's statistics.
#[derive(Clone, Debug, Serialize)]
pub struct FilmStats {
    /// Schema version of the JSON this serialises to. Bumped when a field changes meaning.
    pub schema: u32,
    pub clip: ClipInfo,
    /// #gamevariant  The declared settings of the game variant this film was recorded under.
    /// `None` when the film carries no variant (a pure Forge recording).
    pub gametype: Option<GametypeInfo>,
    pub coverage: Coverage,
    pub players: Vec<PlayerStats>,
    /// The recorded SCOREBOARD, player rows then team rows, highest score first. Independent of
    /// `players`: a slot can have a score with no body in the clip, and a body can have no slot.
    pub scoreboard: Vec<ScoreRow>,
    /// #film-action  Every BIPED DEFINITION the recording spawned a body from, with the body
    /// count — the census that answers "were there any Elites in this clip". Keys are
    /// `group path` with a map open and `wire <n>` without one.
    pub bipeds: std::collections::BTreeMap<String, u64>,
    /// #film-action  THE KILL FEED, in tick order. Decoded from the stream's `incident` events.
    pub kills: Vec<KillRow>,
    /// #film-action  How many deaths a killer can be named for, and from which channel.
    pub death_attribution: DeathAttribution,
    /// #film-action  Vehicle entries and exits, with the stated seat index.
    pub vehicle_transitions: Vec<VehicleTransitionRow>,
    /// #film-action  The whole replicated event channel, by name.
    pub actions: ActionChannel,
    /// Statistic name → provenance, so a consumer can filter on confidence without reading this
    /// file. Includes the statistics that are **not available**, with the reason in
    /// [`FilmStats::notes`].
    pub provenance: std::collections::BTreeMap<String, &'static str>,
    /// Honesty notes: the coverage caveats and the measured negatives, in plain words.
    pub notes: Vec<String>,
    pub compute_ms: f32,
}

/// The schema version of [`FilmStats`]'s JSON.
pub const SCHEMA: u32 = 2;

/// #film-action  How many kill-feed rows a `FilmStats` keeps. A 47-minute clip carries 1,500+
/// incidents; the JSON is a report, not an archive, and the full channel is still summarised in
/// [`ActionChannel::by_incident`].
pub const KILLS_KEPT: usize = 400;
/// As `KILLS_KEPT`, for vehicle entries and exits.
pub const VEHICLE_TRANSITIONS_KEPT: usize = 400;
/// How many kill-feed / vehicle rows the human-readable table prints.
pub const KILLS_SHOWN: usize = 24;

/// What the stream does NOT carry, and the measurement that says so. Printed by the CLI and
/// surfaced in the panel, because a consumer asking "where are the kills" deserves the numbers
/// rather than a missing field.
pub const SCORE_NEGATIVE: &str = "\
The SCORE is in the recording and is decoded exactly: entity type 5 is the engine's own \
`game-engine-statborg` score store, 24 slots of which 0..15 are the player slots, carrying the \
round score, the score carried in from completed rounds, the rounds won and the `final` flag. \
PER-PLAYER TOTALS -- kills, deaths as the game counts them, assists, betrayals, suicides, \
medals -- are NOT: the engine keeps them in a separate per-player table that no replicated \
entity carries, pushed by a non-simulation path, so nothing sends them to a client and nothing \
records them in a film. HMS counts deaths itself, from the entity destroys. \
THE KILL FEED, HOWEVER, IS IN THE RECORDING and is decoded (#film-action): the engine \
broadcasts each Reach medal as an `incident` event on list B command kind 1, naming the player \
who earned it, the other party and the medal's own engine string id -- so who killed whom, with \
what kind of kill and at which tick, is read off the wire. It is not a total: the engine awards \
several incidents for one event (`kill` and `finishing_move_kill` fire together on an \
assassination), so counting rows is a different number from the game's own kill count and HMS \
does not present it as one. There is no kill count, and a derived one \
was refused: every candidate counter in the replicated per-player state was swept against four \
independent oracles (never decreases, total equals the independently-known deaths, steps when \
somebody actually died, matches each player's own death count) and nothing survived the first \
two. See `film_stats::probes` for the sweep and its margins.";

/// #film-action  The coverage limit of the kill feed, and what covers the rest. Printed with the
/// attribution block, because a reader who sees 16 kill rows against 163 deaths will otherwise
/// assume the decode is broken.
pub const KILL_FEED_COVERAGE: &str = "THE KILL FEED COVERS ONLY THE RECORDER. A Reach medal is personal and a film is one client's view, so an `incident` reaches the recording only when the player who recorded it earned it. Measured on `Chill Assassinations.`: 163 bodies destroyed, 49 incidents, 16 kill rows -- and all 16 naming the SAME killer, a player with one life and no deaths, i.e. whoever was recording. That is why Theater can stage only a handful of a clip's assassinations: for the rest the recording states nothing, and a body simply falls with its death clip. THE DAMAGE RECORD COVERS NEARLY EVERYONE, and both of its references are already decoded: `damage_aftermath` names the object damaged and the responsible unit, so the last one before a body died gives that death a killer. It is DERIVED (the references are decoded; choosing the last one and calling that unit the killer is this reader's inference), and the cross-check is printed above: on the deaths both channels name, they agree. Two attempts to turn this into an ASSASSINATION detector were pre-registered and both failed, and the numbers are kept in `film_anim::inferred_assassinations` so they are not repeated: requiring `biped_melee_initiate` scored 0/8 recall (the killer of all eight stated assassinations never swung once in the whole clip), and requiring the two bodies to interpenetrate scored 1.58x contrast against the 1.8x geometry test this project already rejected, with 87-90% of its rows overlapping the melee test.";

// ---------------------------------------------------------------------------------------------
// the computation
// ---------------------------------------------------------------------------------------------

/// Statistics from an already-decoded film. Pure: no renderer, no scene, no frame.
///
/// `names` is only consulted for vehicle and weapon TAG NAMES; pass [`NoNames`] and every row is
/// still produced, with ids in place of names.
pub fn compute(film: &crate::film::Film, idx: &FilmIndex, names: &dyn NameSource) -> FilmStats {
    let t0 = std::time::Instant::now();
    let cut_from = idx.ticks.saturating_sub(CUT_WINDOW_TICKS);

    // ---- clip identity -----------------------------------------------------------------------
    let clip = ClipInfo {
        title: film.content.title.clone(),
        gametype: film.header.game_variant.title.clone(),
        engine_type: film.content.engine,
        map_id: film.content.map_id,
        scenario: film.header.scenario_path.clone(),
        ticks: idx.ticks,
        seconds: idx.ticks as f32 / 60.0,
        recorded_unix: film.content.created_unix,
        recorder_slot: idx.recorder_slot,
        owner_rule_corroborated: idx.owner_rule_ok,
        owner_anchor: idx.owner_anchor,
        names_unverified: idx.owner_anchor.is_derived(),
        live_aim_slot: idx.live_aim_slot(),
        has_entity_stream: idx.entity_messages > 0,
    };

    // ---- the rides, which need no map at all -------------------------------------------------
    // `accepted_rides` keys on matched HORIZONTAL displacement, so it is a property of the
    // stream; only the vehicle's NAME needs the map's sandbox palette.
    let rides = crate::film_anim::accepted_rides(idx);
    let vehicle_name: std::collections::HashMap<u32, String> = idx
        .tracks
        .iter()
        .filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE)
        .filter_map(|t| {
            let (q, item, d) = t.spawned_on?;
            if d > crate::film_anim::SPAWN_MATCH_WU {
                return None;
            }
            let (_tag, path) = names.vehicle(q, item)?;
            Some((t.net_id, path.rsplit('\\').next().unwrap_or(&path).to_string()))
        })
        .collect();

    // ---- per-owner accumulation --------------------------------------------------------------
    let mut cov = Coverage {
        records: idx.records,
        records_decoded: idx.records_decoded,
        entity_messages: idx.entity_messages,
        header_players: film.present_players().count(),
        slots_with_records: idx.player_aims.iter().filter(|v| !v.is_empty()).count(),
        statborg_messages: idx.statborg_messages,
        statborg_decoded: idx.statborg_decoded,
        score_slots: idx.statborg_slots().len(),
        ..Default::default()
    };
    for t in idx.tracks.iter().filter(|t| t.entity_type == BIPED) {
        cov.bodies += 1;
        if t.owner_raw.is_some() {
            cov.bodies_with_owner += 1;
        }
        if !t.positions.is_empty() {
            cov.bodies_with_position += 1;
        }
        if !t.facings.is_empty() {
            cov.bodies_with_facing += 1;
        }
        if !t.weapons.is_empty() {
            cov.weapon_bodies += 1;
        }
        if let Some(d) = t.destroyed_tick {
            cov.body_destroys += 1;
            if d >= cut_from {
                cov.body_destroys_at_cut += 1;
            }
        }
    }

    let mut players: Vec<PlayerStats> = Vec::new();
    for owner in idx.owner_groups() {
        let bodies: Vec<&EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == BIPED && t.owner_raw == Some(owner)).collect();
        let slot = idx.slot_of_owner(owner);
        let hdr = slot.and_then(|s| film.header.players.iter().find(|p| p.present() && p.index == s as usize));

        let mut st = PlayerStats {
            owner,
            slot,
            gamertag: hdr.map(|p| p.gamertag.clone()).filter(|g| !g.is_empty()),
            service_tag: hdr.map(|p| p.service_tag.clone()).filter(|g| !g.is_empty()),
            team: hdr.map(|p| p.team),
            first_body_net_id: bodies.iter().min_by_key(|t| t.first_tick).map(|t| t.net_id),
            // #film-action  The definition the stream names for this player's bodies. The modal
            // value across their lives, so one unreadable creation cannot change the answer.
            biped_wire: {
                let mut c: std::collections::BTreeMap<u32, usize> = Default::default();
                for b in bodies.iter() {
                    if let Some(w) = b.def_wire {
                        *c.entry(w).or_default() += 1;
                    }
                }
                c.into_iter().max_by_key(|&(_, n)| n).map(|(w, _)| w)
            },
            biped: None,
            unowned: owner == 0,
            lives: bodies.len(),
            deaths: 0,
            deaths_at_cut: 0,
            bodies_alive_at_end: 0,
            first_spawn_tick: bodies.iter().map(|t| t.first_tick).min(),
            last_death_tick: None,
            alive_ticks: 0,
            body_ticks: 0,
            longest_life_ticks: 0,
            distance_wu: 0.0,
            position_ticks: 0,
            top_speed_wu_s: 0.0,
            top_foot_speed_wu_s: 0.0,
            gait_ticks: 0,
            unclassified_ticks: 0,
            sprinting_ticks: 0,
            airborne_ticks: 0,
            stationary_ticks: 0,
            running_ticks: 0,
            walking_ticks: 0,
            seated_ticks: 0,
            vehicle_ticks: 0,
            rides: Vec::new(),
            weapons: Vec::new(),
            weapon_bodies: 0,
            aim_samples: slot
                .and_then(|s| idx.player_aims.get(s as usize))
                .map(|v| v.len())
                .unwrap_or(0),
            // The SCORE, straight out of the engine's own score store. Keyed on the player's
            // configuration SLOT, which the statborg index simply IS -- no owner bridge needed,
            // so a player HMS can name is a player whose score is exact.
            score: slot.and_then(|sl| idx.score_at(sl, u32::MAX)),
            round_score: slot.and_then(|sl| idx.statborg_at(sl, u32::MAX)).map(|t| t.0),
            rounds_won: slot.and_then(|sl| idx.statborg_at(sl, u32::MAX)).map(|t| t.2),
            score_updates: slot.and_then(|sl| idx.statborg_at(sl, u32::MAX)).map(|t| t.4).unwrap_or(0),
        };

        // Lifetimes, as a union of intervals: a corpse that lingers past the respawn overlaps the
        // next body, and counting both would inflate time alive.
        let mut spans: Vec<(u32, u32)> = Vec::with_capacity(bodies.len());
        for b in &bodies {
            let end = match b.destroyed_tick {
                Some(d) if d >= cut_from => {
                    st.deaths_at_cut += 1;
                    d
                }
                Some(d) => {
                    st.deaths += 1;
                    st.last_death_tick = Some(st.last_death_tick.map_or(d, |p: u32| p.max(d)));
                    d
                }
                None => {
                    st.bodies_alive_at_end += 1;
                    idx.ticks.saturating_sub(1).max(b.last_tick)
                }
            };
            let end = end.max(b.first_tick);
            spans.push((b.first_tick, end));
            st.body_ticks += end - b.first_tick;
            st.longest_life_ticks = st.longest_life_ticks.max(end - b.first_tick);
            st.distance_wu += b.distance_to(u32::MAX);
            st.position_ticks += b.positions.len() as u32;
            if !b.weapons.is_empty() {
                st.weapon_bodies += 1;
            }
        }
        spans.sort_unstable();
        {
            let mut cur: Option<(u32, u32)> = None;
            for (a, b) in spans {
                match cur {
                    Some((s, e)) if a <= e => cur = Some((s, e.max(b))),
                    Some((s, e)) => {
                        st.alive_ticks += e - s;
                        cur = Some((a, b));
                    }
                    None => cur = Some((a, b)),
                }
            }
            if let Some((s, e)) = cur {
                st.alive_ticks += e - s;
            }
        }

        // The gait census, walked over the body's own POSITION KEYS rather than over every
        // tick. Two things come out of that choice:
        //
        //  * it is honest — a tick is classified only when the recording bracketed it with two
        //    positions less than [`MAX_KEY_GAP`] apart, so a ten-second gap in replication is
        //    counted as `unclassified` instead of being read as ten seconds of standing still
        //    (which is what the per-tick version did: 99% "stationary" on a body with 2%
        //    position coverage);
        //  * it is cheap — O(keys) instead of O(ticks), which took a 47-minute 1,689-body clip
        //    from 647 ms to well under a tenth of that.
        //
        // The classifier itself is `film_anim::derive`, the same one the drawn body uses, so the
        // numbers here and the picture on screen can never disagree.
        for b in &bodies {
            use crate::film_anim::{Gait, Provenance, Stance};
            let mut classified = 0u32;
            for w in b.positions.windows(2) {
                let (t0, t1) = (w[0].0, w[1].0);
                if t1 <= t0 || t1 - t0 > MAX_KEY_GAP {
                    continue;
                }
                let span = t1 - t0;
                let seat = rides
                    .iter()
                    .find(|r| r.biped_net_id == b.net_id && r.covers(t0))
                    .map(|r| crate::film_anim::Seat {
                        vehicle_net_id: r.vehicle_net_id,
                        mode: String::new(),
                        vehicle: String::new(),
                    });
                if seat.is_some() {
                    st.seated_ticks += span;
                    classified += span;
                    continue;
                }
                let d = crate::film_anim::derive(b, t0, None);
                if d.provenance != Provenance::Derived {
                    continue; // no decoded facing at this tick: nothing to classify
                }
                classified += span;
                st.top_speed_wu_s = st.top_speed_wu_s.max(d.ground_speed);
                if d.ground_speed <= FOOT_SPEED_MAX {
                    st.top_foot_speed_wu_s = st.top_foot_speed_wu_s.max(d.ground_speed);
                }
                match d.stance {
                    Stance::Sprint => st.sprinting_ticks += span,
                    Stance::Flight => st.airborne_ticks += span,
                    _ => match d.gait {
                        Gait::Idle | Gait::TurnLeft | Gait::TurnRight => st.stationary_ticks += span,
                        Gait::WalkFront | Gait::BackSlow => st.walking_ticks += span,
                        _ => st.running_ticks += span,
                    },
                }
            }
            st.gait_ticks += classified;
        }
        st.unclassified_ticks = st.body_ticks.saturating_sub(st.gait_ticks);

        // Rides, and the weapons the recording named.
        let own: std::collections::HashSet<u32> = bodies.iter().map(|t| t.net_id).collect();
        // No extra gate here: `accepted_rides` IS the validated detector (42 rides across 8
        // films after the gravity-false-positive fix, horizontal displacement only), and adding
        // a second threshold on top of it would report a number nothing has been measured
        // against.
        for r in rides.iter().filter(|r| own.contains(&r.biped_net_id)) {
            st.vehicle_ticks += r.to_tick - r.from_tick;
            st.rides.push(RideRow {
                vehicle_net_id: r.vehicle_net_id,
                vehicle: vehicle_name.get(&r.vehicle_net_id).cloned(),
                from_tick: r.from_tick,
                to_tick: r.to_tick,
                seconds: r.seconds(),
            });
        }
        let mut seen: std::collections::BTreeMap<u32, u32> = Default::default();
        for b in &bodies {
            for &(tick, w) in &b.weapons {
                seen.entry(w).and_modify(|t| *t = (*t).min(tick)).or_insert(tick);
            }
        }
        st.rides.sort_by(|a, b| b.seconds.total_cmp(&a.seconds));
        st.weapons = seen
            .into_iter()
            .map(|(w, tick)| WeaponRow { weap_index: w, weapon: names.weapon(w), first_tick: tick })
            .collect();
        st.weapons.sort_by_key(|w| w.first_tick);

        st.biped = st.biped_wire.and_then(|w| names.definition(w)).map(|(_, g, path)| format!("{g} {path}"));

        players.push(st);
    }
    // ---- the recorded scoreboard -------------------------------------------------------------
    let mut scoreboard: Vec<ScoreRow> = Vec::new();
    for (i, is_team) in idx.statborg_slots() {
        let Some((r, t, w, f, n)) = idx.statborg_at(i, u32::MAX) else { continue };
        let slot = if is_team { i - crate::film_decode::STATBORG_PLAYERS } else { i };
        scoreboard.push(ScoreRow {
            statborg_index: i,
            kind: if is_team { "team" } else { "player" },
            slot,
            gamertag: (!is_team)
                .then(|| {
                    film.header
                        .players
                        .iter()
                        .find(|p| p.present() && p.index == slot as usize)
                        .map(|p| p.gamertag.clone())
                        .filter(|g| !g.is_empty())
                })
                .flatten(),
            score: if f { t as i32 } else { t as i32 + r as i32 },
            round_score: r,
            carried_score: t,
            rounds_won: w,
            is_final: f,
            updates: n,
        });
    }
    scoreboard.sort_by(|a, b| a.kind.cmp(b.kind).then(b.score.cmp(&a.score)));

    players.sort_by_key(|p| p.owner);
    cov.named_players = players.iter().filter(|p| p.gamertag.is_some()).count();
    cov.owner_groups = players.len();

    // ---- #film-action  THE EVENT CHANNEL: the kill feed, vehicle transitions, the histogram ---
    // Everything here is read off list B command kind 1 (`crate::film_action`). No geometry, no
    // inference: the stream states the event type, the two parties and the seat.
    let mut by_event: std::collections::BTreeMap<String, u64> = Default::default();
    let mut by_incident: std::collections::BTreeMap<String, u64> = Default::default();
    let mut kills: Vec<KillRow> = Vec::new();
    let mut vehicle_transitions: Vec<VehicleTransitionRow> = Vec::new();
    let mut assassinate_events = 0usize;
    // net id -> (slot, gamertag), through the owner bridge the index already corroborated.
    let name_of = |net: Option<u32>| -> (Option<u8>, Option<String>) {
        let Some(net) = net else { return (None, None) };
        let Some(tr) = idx.tracks.iter().find(|t| t.net_id == net) else { return (None, None) };
        let slot = idx
            .slot_of_entity(net)
            .or_else(|| tr.owner_raw.and_then(|o| idx.slot_of_owner(o)));
        let tag = slot.and_then(|sl| {
            film.header
                .players
                .iter()
                .find(|p| p.present() && p.index == sl as usize)
                .map(|p| p.gamertag.clone())
                .filter(|g| !g.is_empty())
        });
        (slot, tag)
    };
    for ev in &idx.events {
        *by_event.entry(ev.name().to_string()).or_default() += 1;
        if ev.kind == crate::film_action::EV_ASSASSINATE {
            assassinate_events += 1;
        }
        if let Some(id) = ev.incident {
            let key = ev.incident_label().map(|s| s.to_string()).unwrap_or_else(|| format!("{id:#010x}"));
            *by_incident.entry(key.clone()).or_default() += 1;
            // A kill incident names the earner first and the other party second. `death` and
            // `suicide` are kept too: they are the only rows for a kill with no killer.
            let is_kill = ev.is_kill() || key == "death" || key == "suicide" || key == "fell_to_death";
            if is_kill && kills.len() < KILLS_KEPT {
                let (ks, kn) = name_of(ev.subject);
                let (vs, vn) = name_of(ev.other);
                kills.push(KillRow {
                    tick: ev.tick,
                    seconds: ev.tick as f32 / 60.0,
                    incident: key,
                    incident_id: id,
                    killer_net_id: ev.subject,
                    killer_slot: ks,
                    killer: kn,
                    victim_net_id: ev.other,
                    victim_slot: vs,
                    victim: vn,
                    assassination: ev.is_assassination(),
                });
            }
        }
        if matches!(
            ev.kind,
            crate::film_action::EV_ENTER_VEHICLE
                | crate::film_action::EV_BOARD_VEHICLE
                | crate::film_action::EV_EXIT_VEHICLE
                | crate::film_action::EV_VEHICLE_EVICT
        ) && vehicle_transitions.len() < VEHICLE_TRANSITIONS_KEPT
        {
            let (sl, pl) = name_of(ev.subject);
            vehicle_transitions.push(VehicleTransitionRow {
                tick: ev.tick,
                seconds: ev.tick as f32 / 60.0,
                event: ev.name(),
                biped_net_id: ev.subject,
                slot: sl,
                player: pl,
                vehicle_net_id: ev.other,
                vehicle: ev.other.and_then(|v| vehicle_name.get(&v).cloned()),
                seat: ev.seat,
            });
        }
    }
    let actions = ActionChannel {
        commands: idx.event_commands,
        decoded: idx.events.len(),
        by_event,
        by_incident,
        assassinate_events,
    };

    // #film-action  WHO KILLED EACH BODY. Two channels, kept apart: the `incident` kill feed,
    // which is decoded but covers only the recorder, and `damage_aftermath`'s responsible unit,
    // which is derived from a decoded reference but covers nearly every death. `agree` is the
    // cross-check that says the second reference really is the killer.
    let mut death_attribution = DeathAttribution::default();
    {
        // victim -> killer, from EVERY kill incident (not the capped `kills` list).
        let mut feed: std::collections::HashMap<u32, u32> = Default::default();
        // victim -> the damage rows naming it, in tick order.
        let mut dmg: std::collections::HashMap<u32, Vec<(u32, u32)>> = Default::default();
        for ev in &idx.events {
            if ev.kind == crate::film_action::EV_DAMAGE_AFTERMATH {
                if let (Some(v), Some(k)) = (ev.subject, ev.other) {
                    dmg.entry(v).or_default().push((ev.tick, k));
                }
            }
            if ev.is_kill() {
                if let (Some(v), Some(k)) = (ev.other, ev.subject) {
                    feed.entry(v).or_insert(k);
                }
            }
        }
        for rows in dmg.values_mut() {
            rows.sort_unstable();
        }
        let (mut fk, mut dk) = (
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        );
        for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
            let Some(d) = crate::film_anim::death_tick(t) else { continue };
            death_attribution.deaths += 1;
            let f = feed.get(&t.net_id).copied();
            // the last damage within [-2 s, +1 s] of the death that names someone else
            let k = dmg
                .get(&t.net_id)
                .and_then(|rows| {
                    rows.iter()
                        .filter(|&&(tk, who)| who != t.net_id && tk <= d + 60 && tk + 120 >= d)
                        .max_by_key(|&&(tk, _)| tk)
                        .map(|&(_, who)| who)
                });
            if let Some(f) = f {
                fk.insert(f);
            }
            if let Some(k) = k {
                dk.insert(k);
            }
            match (f, k) {
                (Some(f), Some(k)) => {
                    death_attribution.from_kill_feed += 1;
                    if f == k {
                        death_attribution.agree += 1;
                    } else {
                        death_attribution.disagree += 1;
                    }
                }
                (Some(_), None) => death_attribution.from_kill_feed += 1,
                (None, Some(_)) => death_attribution.from_damage_record += 1,
                (None, None) => death_attribution.unattributed += 1,
            }
        }
        death_attribution.feed_killers = fk.len();
        death_attribution.damage_killers = dk.len();
    }

    // #film-action  The body census, over EVERY biped track (not just the owned ones).
    let mut bipeds: std::collections::BTreeMap<String, u64> = Default::default();
    for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
        let Some(w) = tr.def_wire else { continue };
        let key = names.definition(w).map(|(_, g, p)| format!("{g} {p}")).unwrap_or_else(|| format!("wire {w}"));
        *bipeds.entry(key).or_default() += 1;
    }

    let mut out = FilmStats {
        schema: SCHEMA,
        clip,
        // #gamevariant  The film's own embedded game variant, decoded. Costs nothing: the
        // header was already parsed when the film was opened.
        gametype: film.header.gametype.as_ref().map(|gv| GametypeInfo::build(gv, names)),
        coverage: cov,
        players,
        scoreboard,
        bipeds,
        kills,
        death_attribution,
        vehicle_transitions,
        actions,
        provenance: provenance_table(),
        notes: Vec::new(),
        compute_ms: 0.0,
    };
    out.notes = notes_for(&out);
    out.compute_ms = t0.elapsed().as_secs_f32() * 1000.0;
    out
}

/// Open a film and compute its statistics. The whole headless path, with no map.
pub fn from_path(path: &std::path::Path) -> Result<FilmStats, String> {
    from_path_with(path, &NoNames)
}

/// As [`from_path`], with a name source for the vehicle and weapon tag names.
pub fn from_path_with(path: &std::path::Path, names: &dyn NameSource) -> Result<FilmStats, String> {
    let f = crate::film::open(path).map_err(|e| e.to_string())?;
    let idx = crate::film_decode::build_index_players(&f.bytes, &f.data, &f.header.map_variant, f.recorder_slot(), &f.header.players);
    Ok(compute(&f, &idx, names))
}

/// Statistic name → provenance label. The statistics that are **not available** are listed too,
/// so a consumer sees that they were considered and refused rather than forgotten.
pub fn provenance_table() -> std::collections::BTreeMap<String, &'static str> {
    let mut m = std::collections::BTreeMap::new();
    for (k, v) in [
        // decoded: straight out of the stream, oracle-checked.
        ("lives", Prov::Decoded),
        ("first_spawn_tick", Prov::Decoded),
        ("aim_samples", Prov::Decoded),
        ("kill_feed", Prov::Decoded),
        // derived: decoded quantities plus this reader's inference.
        ("deaths", Prov::Derived),
        // #film-action  The REFERENCES are decoded (`damage_aftermath` names the damaged object
        // and the responsible unit); picking the last one before a death and calling that unit
        // the killer is this reader's inference, so the attribution is derived.
        ("death_killer_from_damage_record", Prov::Derived),
        ("deaths_at_cut", Prov::Derived),
        ("last_death_tick", Prov::Derived),
        ("alive_ticks", Prov::Derived),
        ("longest_life_ticks", Prov::Derived),
        ("distance_wu", Prov::Derived),
        ("top_speed_wu_s", Prov::Derived),
        ("sprinting_ticks", Prov::Derived),
        ("airborne_ticks", Prov::Derived),
        ("stationary_ticks", Prov::Derived),
        ("running_ticks", Prov::Derived),
        ("walking_ticks", Prov::Derived),
        ("seated_ticks", Prov::Derived),
        ("vehicle_ticks", Prov::Derived),
        ("rides", Prov::Derived),
        ("gamertag", Prov::Derived),
        ("slot", Prov::Derived),
        ("armour", Prov::Derived),
        ("team", Prov::Derived),
        // decoded but sparse — the coverage block says how sparse.
        ("weapons", Prov::Decoded),
        // THE SCORE IS IN THE STREAM. `game-engine-statborg` (entity type 5) is the engine's own
        // score store and its index is the player slot, so this is decoded and attributed
        // without any inference at all. It is only ever as complete as the recording: a clip
        // whose gametype never awarded a point carries no updates.
        ("score", Prov::Decoded),
        ("round_score", Prov::Decoded),
        ("rounds_won", Prov::Decoded),
        // #gamevariant  The game variant is DECLARED DATA, embedded in the film whole, so its
        // settings are decoded outright -- loadout palettes, player traits, respawn and round
        // options, player species. The spawn weapon applies the engine's own
        // loadout-over-traits precedence to them, which makes it derived.
        ("gametype", Prov::Decoded),
        // #film-action  Two different questions, and the table has to keep them apart. The
        // gametype's *setting* is declared data and is decoded outright, but under *player
        // choice* it defers to profile data a film does not carry, so it is not DECISIVE about
        // any particular body. The body's own definition reference is both.
        ("gametype.player_species", Prov::Decoded),
        ("gametype.loadouts", Prov::Decoded),
        ("gametype.respawn_time_seconds", Prov::Decoded),
        ("gametype.base_primary_weapon", Prov::Decoded),
        ("gametype.spawn_primary_weapon", Prov::Derived),
        ("gametype.drawn_primary_weapon", Prov::Derived),
        // #film-action  The replicated EVENT channel -- list B command kind 1, 72 named event
        // types. Everything here is stated on the wire: the event type, its up-to-two object
        // references and its per-type parameters. The KILL FEED comes out of the `incident`
        // event, whose leading 32 bits are an engine string id, so who killed whom and with what
        // kind of kill is decoded outright. Per-player kill TOTALS are still not available:
        // the engine's per-player statistics table is not replicated (see SCORE_NEGATIVE), and
        // counting incidents is not the same number -- the engine awards several at once
        // (`kill` and `finishing_move_kill` fire together on an assassination).
        ("actions", Prov::Decoded),
        ("kills_feed", Prov::Decoded),
        ("kills_feed.incident", Prov::Decoded),
        ("kills_feed.killer", Prov::Derived),
        ("kills_feed.victim", Prov::Derived),
        ("kills_feed.assassination", Prov::Decoded),
        ("vehicle_transitions", Prov::Decoded),
        ("vehicle_transitions.seat", Prov::Decoded),
        ("bipeds", Prov::Decoded),
        ("biped", Prov::Decoded),
        // refused.
        ("loadout_chosen_per_player", Prov::NotAvailable),
        ("kills", Prov::NotAvailable),
        ("assists", Prov::NotAvailable),
        ("medals", Prov::NotAvailable),
        ("damage", Prov::NotAvailable),
        ("shots_fired", Prov::NotAvailable),
        ("betrayals", Prov::NotAvailable),
        ("suicides", Prov::NotAvailable),
    ] {
        m.insert(k.to_string(), v.label());
    }
    m
}

/// The honesty notes, generated from this clip's own coverage so they are never boilerplate.
fn notes_for(s: &FilmStats) -> Vec<String> {
    let c = &s.coverage;
    let mut v = Vec::new();
    if !s.clip.has_entity_stream {
        v.push(
            "This film records player INPUT, not entity replication: it carries no bodies, so \
             there are no per-player statistics to pull. Both kinds decode fully; a film has one \
             or the other, never both."
                .into(),
        );
        return v;
    }
    // #film-action
    if s.actions.decoded > 0 {
        let assn = s.kills.iter().filter(|k| k.assassination).count();
        v.push(format!(
            "THE ACTION CHANNEL IS DECODED. List B command kind 1 is the engine's own event \
             channel -- 72 named types, each with up to two object references and its own \
             parameters -- and this clip carries {} of them ({} accepted). That is where the \
             discrete actions ride: `biped_melee_initiate`, `biped_throw_initiate`, \
             `weapon_reload`, `unit_zoom`, `biped_enter_vehicle` / `biped_exit_vehicle` (with \
             the stated SEAT index), and `incident` -- Reach's medal/kill feed, whose leading 32 \
             bits are an engine string id. So the kill feed is read, not guessed: {} row(s) here, \
             {assn} of them an assassination. The unit's own derived action field 0x36, which \
             carries `_action_assassinate`, is essentially never replicated (2 messages in \
             7,367,920 across 57 films), so this channel is the whole answer.",
            s.actions.commands,
            s.actions.decoded,
            s.kills.len()
        ));
    }
    if !s.bipeds.is_empty() {
        v.push(format!(
            "Every body names the OBJECT DEFINITION it was spawned from: a 14-bit network tag \
             reference at bit 0 of its creation message. This clip spawned bodies from {} \
             definition(s){}. HMS no longer assumes a Spartan.",
            s.bipeds.len(),
            if s.bipeds.keys().any(|k| k.starts_with("wire ")) {
                " (open the map with --map to resolve them to tag names)"
            } else {
                ""
            }
        ));
    }
    v.push(format!(
        "Deaths are a body's track ending — list B kind 4, the entity destroy, which is DECODED \
         (100.0% of 356,957 messages name a live entity, zero double-destroys). Attributing one \
         to a player goes through the owner stamp, so the per-player figure is derived. \
         Measured on this corpus, there is no end-of-recording teardown burst: {} of {} body \
         destroys fell in the last {:.0}s, so essentially every destroy is a real death and \
         bodies still alive at the end are simply not counted.",
        c.body_destroys_at_cut,
        c.body_destroys,
        CUT_WINDOW_TICKS as f32 / 60.0
    ));
    v.push(format!(
        "Distance is a LOWER BOUND: the stream replicates a position on only some ticks ({} of \
         {} bodies ever sent one), and a straight line between two sparse keys cuts the corners \
         off the real path.",
        c.bodies_with_position, c.bodies
    ));
    let gait = s.players.iter().map(|p| p.gait_ticks).sum::<u32>();
    let body = s.players.iter().map(|p| p.body_ticks).sum::<u32>();
    v.push(format!(
        "The sprint / airborne / stationary census classifies a tick only where the recording \
         bracketed it with two positions less than {:.1}s apart AND the body reported a facing: \
         {:.1}% of all body-ticks in this clip ({} of {}). The rest is counted as unclassified \
         rather than as standing still — a body that stops replicating is not the same thing as \
         a player who stopped moving.",
        MAX_KEY_GAP as f32 / 60.0,
        100.0 * gait as f32 / body.max(1) as f32,
        gait,
        body
    ));
    v.push(
        "The classified sample is BIASED TOWARDS MOTION, and the share figures have to be read \
         that way. A biped replicates a position when it moves, so the ticks dense enough to \
         classify are disproportionately the ticks the player was moving: a low `still` share \
         means \"rarely still while moving\", not \"rarely still\". The shares are honest \
         relative to each other and across players; they are not shares of the match."
            .to_string(),
    );
    // #gamevariant  The gametype is the ONLY place a spawn weapon or a species exists, so the
    // note is generated from what this film's own variant says rather than being boilerplate.
    v.push(match &s.gametype {
        None => "This recording carries NO game variant: the embedded `c_game_variant` image is \
                 all zero and its content type is not 6, which is what a pure Forge session \
                 looks like. Every gametype figure is therefore not available for this clip, \
                 including the spawn weapon and the player species."
            .to_string(),
        Some(g) if g.spawn_primary_weapon.is_empty() => format!(
            "The SPAWN WEAPON is not available for this clip. {} Species is `{}`.",
            g.spawn_primary_weapon_note, g.player_species
        ),
        Some(g) => format!(
            "The SPAWN WEAPON is `{}`. {} It is DECLARED data, not recorded data: a unit reports \
             the weapon in its hands only when it CHANGES, so the recording itself never names a \
             starting weapon. Species is `{}`{}.",
            g.spawn_primary_weapon,
            g.spawn_primary_weapon_note,
            g.player_species,
            if g.species_allows_elites { "" } else { ", so every player was a Spartan" }
        ),
    });
    v.push(if c.weapon_bodies == 0 {
        "No body in this clip ever named the weapon in its hands. The weapon field reports only \
         on CHANGE, never at spawn, so an empty weapons list means the recording never said — \
         NOT that the player carried nothing."
            .to_string()
    } else {
        format!(
            "{} of {} bodies ({:.1}%) ever named the weapon in their hands. The field reports \
             only on CHANGE, never at spawn, so an empty weapons list means the recording never \
             said — NOT that the player carried nothing.",
            c.weapon_bodies,
            c.bodies,
            100.0 * c.weapon_bodies as f32 / c.bodies.max(1) as f32
        )
    });
    v.push(format!(
        "{} of {} player row(s) could be NAMED. The film header names {} slot(s) and {} slot(s) \
         carried per-tick records. A header roster is who was present when the header was \
         written, so a player it never named still gets a full row, keyed by owner group and \
         labelled with their entity id.{}",
        c.named_players,
        s.players.len(),
        c.header_players,
        c.slots_with_records,
        match s.clip.owner_anchor {
            crate::film_decode::OwnerAnchor::AuthorRecord =>
                " Names were promoted by `slot = owner - 1`, corroborated against this film's own \
                 author record — a stated fact.",
            crate::film_decode::OwnerAnchor::LiveAim =>
                " This recording carries NO author record, which is what an in-progress `.temp` \
                 looks like: those fields are only written when a film is finalised. The recorder \
                 was identified instead from the one player slot whose aim is LIVE (a player's own \
                 input is captured locally; every other slot's aim is zero-filled), and \
                 `slot = owner - 1` was corroborated against that. The names, armour and colours \
                 below are therefore DERIVED and unverified, and were promoted only onto header \
                 slots that are actually occupied.",
            crate::film_decode::OwnerAnchor::None =>
                " This film's recorder anchor did not corroborate `slot = owner - 1`, so no name \
                 was promoted from the rule — by design, since a wrong name is worse than none.",
        }
    ));
    // The one cross-check on the owner grouping itself. Two independent counts of "how many
    // players were in this game" that disagree badly mean the grouping is not to be trusted, and
    // saying so is the whole point of printing both.
    let groups_real = s.players.iter().filter(|p| !p.unowned).count();
    if c.slots_with_records > 0 && groups_real * 2 < c.slots_with_records {
        v.push(format!(
            "WARNING: {} owner group(s) against {} player slot(s) carrying per-tick records. \
             These are two independent counts of how many players were in this game and they \
             disagree, so the per-player split on this clip should not be trusted — the clip \
             totals still are.",
            groups_real, c.slots_with_records
        ));
    }
    v.push(if c.score_slots == 0 {
        "The recording carries NO score update at all, so there is no scoreboard. That is not \
         zero and it is not a decode failure: a film only carries the score updates the \
         recording was sent, and a social or Forge clip whose gametype never awards a point \
         sends none."
            .to_string()
    } else {
        format!(
            "The scoreboard is DECODED, out of the engine's own `game-engine-statborg` store: \
             {} slot(s) reported, over {} of {} statborg messages that passed both free oracles \
             (the rebuilt presence mask equal to the message's own, and the payload consumed to \
             within the writer's byte padding). A slot with no row was never sent a score \
             update, which is not the same as zero.",
            c.score_slots, c.statborg_decoded, c.statborg_messages
        )
    });
    if c.score_slots > 0 {
        let sc: i32 = s.scoreboard.iter().filter(|r| r.kind == "player").map(|r| r.score).sum();
        v.push(format!(
            "The scores total {sc} against {} death(s) counted here, and the two are NOT expected \
             to balance. A statborg score is the player's CUMULATIVE match total, carried in from \
             before the recording started; the deaths are only the ones inside this clip. On a \
             clip that covers a whole match the scores come in under the deaths (measured: 77 \
             against 95 on a 16-player assassination match, the gap being falls and suicides); on \
             a short clip they come in far over it. So this is a sanity read, not a conservation \
             check.",
            c.body_destroys - c.body_destroys_at_cut
        ));
    }
    v.push(SCORE_NEGATIVE.to_string());
    v.push(KILL_FEED_COVERAGE.to_string());
    v
}

// ---------------------------------------------------------------------------------------------
// rendering it for a human
// ---------------------------------------------------------------------------------------------

pub fn fmt_time(seconds: f32) -> String {
    let s = seconds.max(0.0).round() as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

fn pct(n: u32, d: u32) -> String {
    if d == 0 {
        "   -".into()
    } else {
        format!("{:3.0}%", 100.0 * n as f32 / d as f32)
    }
}

impl FilmStats {
    /// The default CLI output: a readable table on stdout.
    pub fn table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{}\n  gametype {:?}  engine {}  map id {}  {}  {} ticks ({})\n",
            if self.clip.title.is_empty() { "(untitled clip)" } else { self.clip.title.as_str() },
            self.clip.gametype,
            self.clip.engine_type,
            self.clip.map_id,
            self.clip.scenario,
            self.clip.ticks,
            fmt_time(self.clip.seconds)
        ));
        let c = &self.coverage;
        s.push_str(&format!(
            "  records {}/{} decoded  {} entity messages  {} bodies ({} with an owner, {} with a position, {} with a facing)\n               player names anchored on: {}\n",
            c.records_decoded, c.records, c.entity_messages, c.bodies, c.bodies_with_owner, c.bodies_with_position, c.bodies_with_facing,
            self.clip.owner_anchor.label()
        ));
        // #gamevariant  The gametype's real settings, not just its title.
        match &self.gametype {
            None => s.push_str("\n  GAMETYPE SETTINGS: this recording carries no game variant\n"),
            Some(g) => {
                s.push_str(&format!(
                    "\n  GAMETYPE SETTINGS (decoded from the embedded game variant{})\n",
                    if g.names_resolved { "" } else { "; no map open, so weapons read as indices" }
                ));
                s.push_str(&format!(
                    "    species {}   teams {} ({} active)   rounds {}   round time {}   sudden death {} s\n",
                    g.player_species,
                    if g.teams_enabled { "on" } else { "off" },
                    g.teams_active,
                    if g.number_of_rounds == 0 { "no limit".into() } else { g.number_of_rounds.to_string() },
                    if g.round_time_limit_minutes == 0 { "no limit".into() } else { format!("{} min", g.round_time_limit_minutes) },
                    g.sudden_death_seconds,
                ));
                s.push_str(&format!(
                    "    respawn {} s (+{} growth, suicide +{}, betrayal +{})   lives {}   loadout cam {} s\n",
                    g.respawn_time_seconds, g.respawn_growth_seconds, g.suicide_penalty_seconds,
                    g.betrayal_penalty_seconds,
                    if g.lives_per_round == 0 { "unlimited".into() } else { g.lives_per_round.to_string() },
                    g.loadout_camera_time_seconds,
                ));
                s.push_str(&format!(
                    "    base traits: {} / {} / {}, grenades {}, damage resistance {}, headshot immunity {}\n",
                    g.base_primary_weapon, g.base_secondary_weapon, g.base_armor_ability,
                    g.base_grenades, g.base_damage_resistance, g.base_headshot_immunity,
                ));
                s.push_str(&format!(
                    "    SPAWN WEAPON: {}   (drawn: {} -- {})\n",
                    if g.spawn_primary_weapon.is_empty() { "not available".into() } else { g.spawn_primary_weapon.clone() },
                    g.drawn_primary_weapon,
                    g.drawn_primary_weapon_rule,
                ));
                if g.loadouts.is_empty() {
                    s.push_str("    loadouts: none authored\n");
                } else {
                    s.push_str(&format!(
                        "    loadouts ({} authored, spartan {}, elite {}):\n",
                        g.loadouts.len(),
                        if g.spartan_loadouts_enabled { "on" } else { "off" },
                        if g.elite_loadouts_enabled { "on" } else { "off" },
                    ));
                    for l in &g.loadouts {
                        s.push_str(&format!(
                            "      tier {} #{} {} {:<24} {} / {} / {} / {}\n",
                            l.tier, l.slot, if l.visible { "*" } else { " " }, l.name,
                            l.primary_weapon, l.secondary_weapon, l.armor_ability, l.grenades,
                        ));
                    }
                }
            }
        }
        if self.players.is_empty() {
            s.push_str("\n  no player bodies in this clip\n");
            for n in &self.notes {
                s.push_str(&format!("\n  note: {n}\n"));
            }
            return s;
        }
        if self.scoreboard.is_empty() {
            s.push_str("\n  scoreboard: the recording carries no score update (not zero -- never sent)\n");
        } else {
            s.push_str("\n  SCOREBOARD (decoded from the engine's own game-engine-statborg store)\n");
            for r in &self.scoreboard {
                s.push_str(&format!(
                    "    {:<6} {:>2}  {:<20} score {:>6}  (round {:>5} + carried {:>5}){}  rounds won {:>3}  {} update(s)\n",
                    r.kind,
                    r.slot,
                    r.gamertag.clone().unwrap_or_else(|| "(not named by this recording)".into()),
                    r.score,
                    r.round_score,
                    r.carried_score,
                    if r.is_final { "  FINAL" } else { "" },
                    r.rounds_won,
                    r.updates
                ));
            }
        }
        s.push('\n');
        s.push_str(&format!(
            "  {:<20} {:>3} {:>6} {:>4} {:>6} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}\n",
            "player", "own", "score", "life", "deaths", "alive", "longest", "dist wu", "foot", "peak", "sprnt", "air", "still", "veh", "clas"
        ));
        for p in &self.players {
            s.push_str(&format!(
                "  {:<20} {:>3} {:>6} {:>4} {:>6} {:>7} {:>7} {:>7.0} {:>5.1} {:>5.1} {:>5} {:>5} {:>5} {:>5} {:>5}\n",
                p.label(),
                p.owner,
                match p.score {
                    Some(v) => v.to_string(),
                    None => "-".into(),
                },
                p.lives,
                p.deaths,
                fmt_time(p.alive_ticks as f32 / 60.0),
                fmt_time(p.longest_life_ticks as f32 / 60.0),
                p.distance_wu,
                p.top_foot_speed_wu_s,
                p.top_speed_wu_s,
                pct(p.sprinting_ticks, p.gait_ticks),
                pct(p.airborne_ticks, p.gait_ticks),
                pct(p.stationary_ticks, p.gait_ticks),
                fmt_time(p.vehicle_ticks as f32 / 60.0),
                pct(p.gait_ticks, p.body_ticks),
            ));
            if p.unowned {
                s.push_str("      owner stamp 0 = NOBODY: these bodies are in the recording but belong to no player\n");
            }
            for r in p.rides.iter().take(RIDES_LISTED) {
                s.push_str(&format!(
                    "      rode {} for {:.1}s from {}\n",
                    r.vehicle.clone().unwrap_or_else(|| format!("vehicle {:#010x} (not named: no map open)", r.vehicle_net_id)),
                    r.seconds,
                    fmt_time(r.from_tick as f32 / 60.0)
                ));
            }
            if p.rides.len() > RIDES_LISTED {
                s.push_str(&format!("      ... and {} more ride(s)\n", p.rides.len() - RIDES_LISTED));
            }
            for w in &p.weapons {
                s.push_str(&format!(
                    "      held {} from {}\n",
                    w.weapon.clone().unwrap_or_else(|| format!("'weap' index {} (not named: no map open)", w.weap_index)),
                    fmt_time(w.first_tick as f32 / 60.0)
                ));
            }
        }
        s.push_str(
            "\n  own = the owner stamp (this row's key)   score = the recorded score, '-' = never sent\n  \
             life / deaths = bodies spawned / destroyed\n  \
             alive = union of this player's bodies' lifetimes   foot = top speed on foot, wu/s\n  \
             peak = top speed including man-cannon launches and teleports, wu/s\n  \
             sprnt / air / still = share of the ticks that could be CLASSIFIED\n  \
             clas = what share of this player's body-ticks those were; the rest is replication\n  \
             the recording never bracketed with two positions close enough to differentiate.\n",
        );

        // #film-action  the replicated EVENT channel
        if self.actions.decoded > 0 {
            s.push_str(&format!(
                "\n  ACTIONS (decoded -- list B command kind 1, the engine's 72-type event channel): \
                 {}/{} commands\n",
                self.actions.decoded, self.actions.commands
            ));
            let mut ev: Vec<_> = self.actions.by_event.iter().collect();
            ev.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            let line: Vec<String> = ev.iter().map(|(k, c)| format!("{k} {c}")).collect();
            for chunk in line.chunks(4) {
                s.push_str(&format!("    {}\n", chunk.join("   ")));
            }
        }
        if !self.bipeds.is_empty() {
            s.push_str("\n  BODIES, by the object definition the stream names in their creation message (decoded)\n");
            let mut b: Vec<_> = self.bipeds.iter().collect();
            b.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            for (k, c) in b {
                s.push_str(&format!("    {c:>6}  {k}\n"));
            }
        }
        if !self.kills.is_empty() {
            let assn = self.kills.iter().filter(|k| k.assassination).count();
            s.push_str(&format!(
                "\n  KILL FEED (decoded from the stream's `incident` events): {} row(s), {assn} assassination(s)\n",
                self.kills.len()
            ));
            for k in self.kills.iter().take(KILLS_SHOWN) {
                s.push_str(&format!(
                    "    {:>8}  {:<24} {} -> {}{}\n",
                    fmt_time(k.seconds),
                    k.incident,
                    k.killer.clone().unwrap_or_else(|| k.killer_net_id.map(|v| format!("{v:#010x}")).unwrap_or_else(|| "-".into())),
                    k.victim.clone().unwrap_or_else(|| k.victim_net_id.map(|v| format!("{v:#010x}")).unwrap_or_else(|| "-".into())),
                    if k.assassination { "   [ASSASSINATION]" } else { "" }
                ));
            }
            if self.kills.len() > KILLS_SHOWN {
                s.push_str(&format!("    ... and {} more\n", self.kills.len() - KILLS_SHOWN));
            }
        }
        let a = &self.death_attribution;
        if a.deaths > 0 {
            s.push_str(&format!(
                "\n  WHO KILLED EACH BODY: {} death(s) \u{2014} {} named by the KILL FEED (decoded, \
                 {} distinct killer(s)), {} more named only by the DAMAGE RECORD (derived from \
                 `damage_aftermath`'s responsible unit, {} distinct killer(s)), {} unattributed\n",
                a.deaths, a.from_kill_feed, a.feed_killers, a.from_damage_record, a.damage_killers, a.unattributed
            ));
            if a.agree + a.disagree > 0 {
                s.push_str(&format!(
                    "    cross-check: both channels name a killer for {} death(s) and they AGREE on {} ({:.0}%)\n",
                    a.agree + a.disagree,
                    a.agree,
                    100.0 * a.agree as f32 / (a.agree + a.disagree) as f32
                ));
            }
        }
        if !self.vehicle_transitions.is_empty() {
            s.push_str(&format!(
                "\n  VEHICLE ENTRIES / EXITS (decoded, with the stated seat index): {} row(s)\n",
                self.vehicle_transitions.len()
            ));
            for v in self.vehicle_transitions.iter().take(KILLS_SHOWN) {
                s.push_str(&format!(
                    "    {:>8}  {:<22} {} seat {}  {}\n",
                    fmt_time(v.seconds),
                    v.event,
                    v.player.clone().unwrap_or_else(|| v.biped_net_id.map(|b| format!("{b:#010x}")).unwrap_or_else(|| "-".into())),
                    v.seat.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
                    v.vehicle.clone().unwrap_or_else(|| v.vehicle_net_id.map(|b| format!("{b:#010x}")).unwrap_or_else(|| "-".into()))
                ));
            }
        }

        for n in &self.notes {
            s.push_str(&format!("\n  note: {n}\n"));
        }
        s
    }

    /// The compact per-player lines the GUI panel and the script verb share.
    pub fn lines(&self) -> Vec<String> {
        self.players
            .iter()
            .map(|p| {
                format!(
                    "{:<18} {}{:>2} life(s) {:>2} death(s)  alive {}  {:>5.0} wu  top {:>4.1} wu/s{}",
                    p.label(),
                    match p.score {
                        Some(v) => format!("score {v:>5}  "),
                        None => String::new(),
                    },
                    p.lives,
                    p.deaths,
                    fmt_time(p.alive_ticks as f32 / 60.0),
                    p.distance_wu,
                    p.top_foot_speed_wu_s,
                    if p.vehicle_ticks > 0 {
                        format!("  veh {}", fmt_time(p.vehicle_ticks as f32 / 60.0))
                    } else {
                        String::new()
                    }
                )
            })
            .collect()
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

// ---------------------------------------------------------------------------------------------
// the headless CLI verb
// ---------------------------------------------------------------------------------------------

/// `hms-app --film-stats <file.mov|.film|.temp|directory>... [--json] [--map <cache.map>]`
///
/// The PRIMARY interface, and deliberately not behind the script host: `--script` needs a GPU
/// (it fails with "Parent device is lost" when another instance holds the device), which makes
/// it useless for someone who only wants numbers out of a file, in CI, or over SSH. This verb
/// opens no window and creates no wgpu device. With no `--map` it needs nothing installed at
/// all — only the film.
///
/// Discipline: **data on stdout, diagnostics on stderr**, and a non-zero exit only when nothing
/// could be read. A per-file failure names the file and the reason and the run carries on.
///
/// Returns the process exit code.
pub fn cli(paths: &[String], json: bool, names: &dyn NameSource) -> i32 {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for a in paths {
        let p = std::path::PathBuf::from(a);
        if p.is_dir() {
            match std::fs::read_dir(&p) {
                Ok(rd) => {
                    let mut batch: Vec<std::path::PathBuf> = rd
                        .flatten()
                        .map(|e| e.path())
                        .filter(|q| {
                            q.extension().map_or(false, |x| {
                                x.eq_ignore_ascii_case("mov")
                                    || x.eq_ignore_ascii_case("film")
                                    || x.eq_ignore_ascii_case("temp")
                            })
                        })
                        .collect();
                    batch.sort();
                    if batch.is_empty() {
                        eprintln!("{}: no .mov / .film / .temp files in this folder", p.display());
                    }
                    files.extend(batch);
                }
                Err(e) => eprintln!("{}: {e}", p.display()),
            }
        } else {
            files.push(p);
        }
    }
    if files.is_empty() {
        eprintln!(
            "--film-stats: nothing to read.\n\
             usage: hms-app --film-stats <film.mov | film.film | folder>... [--json] [--map <cache.map>]\n\
             With no --map, vehicle and weapon TAG NAMES are not resolved; every other statistic\n\
             comes from the film alone."
        );
        return 2;
    }
    let mut ok = 0usize;
    let mut out: Vec<FilmStats> = Vec::new();
    for p in &files {
        match std::fs::metadata(p) {
            // A zero-byte autosave stub is the common case in the autosave folder (87 of 110 in
            // one directory), so it is reported as what it is rather than as a parse failure.
            Ok(m) if m.len() == 0 => {
                eprintln!("{}: empty file (MCC leaves zero-byte autosave stubs)", p.display());
                continue;
            }
            Ok(_) => {}
            // Distinct from "empty": a path that is not there reads as a zero length if the
            // error is swallowed, which turns "you typed the wrong name" into "this film is
            // empty" and sends the reader looking in the wrong place.
            Err(e) => {
                eprintln!("{}: {e}", p.display());
                continue;
            }
        }
        match from_path_with(p, names) {
            Ok(s) => {
                ok += 1;
                if json {
                    out.push(s);
                } else {
                    // The file NAME, never the path: on this machine the film folders sit under a
                    // directory named after the owner.
                    println!(
                        "\n=== {} ===",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    );
                    print!("{}", s.table());
                }
            }
            Err(e) => eprintln!("{}: {e}", p.display()),
        }
    }
    if json {
        #[derive(serde::Serialize)]
        struct Batch<'a> {
            schema: u32,
            films: &'a [FilmStats],
        }
        match serde_json::to_string_pretty(&Batch { schema: SCHEMA, films: &out }) {
            Ok(j) => println!("{j}"),
            Err(e) => {
                eprintln!("--film-stats: could not serialise: {e}");
                return 1;
            }
        }
    }
    eprintln!("--film-stats: {ok} of {} file(s) read", files.len());
    if ok == 0 {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------------------------
// the panel
// ---------------------------------------------------------------------------------------------

/// Fixed height of the statistics table, so numbers changing never move what is below it.
pub const PANEL_HEIGHT: f32 = 220.0;

/// The statistics section of the Theater side panel. One collapsing header, closed by default,
/// reached through a SINGLE call from the panel's own code — this module owns everything inside
/// it.
///
/// `stats` is computed ONCE per film and cached by the caller. Nothing here is per frame: a
/// `None` simply means it has not been computed yet. `force_open` is the script hook
/// (`film stats open`), so the section can be captured without a mouse.
pub fn panel_ui(ui: &mut egui::Ui, stats: Option<&FilmStats>, force_open: bool) {
    egui::CollapsingHeader::new("Player statistics")
        .id_salt("film_stats_panel")
        .default_open(false)
        // `film stats open` forces it, so the section can be verified from a script without a
        // mouse; left `None` the viewer keeps control of it.
        .open(force_open.then_some(true))
        .show(ui, |ui| {
            let Some(s) = stats else {
                ui.label(egui::RichText::new("Not computed yet — the recording is still decoding.").weak());
                return;
            };
            if s.players.is_empty() {
                ui.label(
                    egui::RichText::new(if s.clip.has_entity_stream {
                        "No player bodies in this clip."
                    } else {
                        "This film records player INPUT, not entity replication, so it carries no \
                         bodies and there are no per-player statistics to pull."
                    })
                    .weak(),
                );
                return;
            }
            egui::ScrollArea::vertical()
                .id_salt("film_stats_scroll")
                .min_scrolled_height(PANEL_HEIGHT)
                .max_height(PANEL_HEIGHT)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Grid::new("film_stats_grid").num_columns(8).spacing([8.0, 2.0]).striped(true).show(ui, |ui| {
                        for h in ["player", "score", "lives", "deaths", "alive", "dist wu", "top wu/s", "veh"] {
                            ui.label(egui::RichText::new(h).strong().small());
                        }
                        ui.end_row();
                        for p in &s.players {
                            let lbl = if p.gamertag.is_some() {
                                egui::RichText::new(p.label())
                            } else {
                                egui::RichText::new(p.label()).monospace().weak()
                            };
                            ui.label(lbl).on_hover_text(if p.unowned {
                                "The owner stamp on these bodies is 0, which means NOBODY: the stamp is \
                                 1-based. They are in the recording but belong to no player."
                                    .to_string()
                            } else if p.gamertag.is_some() {
                                format!(
                                    "owner group {}, player slot {:?}{}",
                                    p.owner,
                                    p.slot,
                                    // #film-action  the character the stream names for these bodies
                                    match (&p.biped, p.biped_wire) {
                                        (Some(b), _) => format!("\nbody: {b} (decoded from the spawn message)"),
                                        (None, Some(w)) => format!("\nbody: definition reference {w} (open a map to name it)"),
                                        _ => String::new(),
                                    }
                                )
                            } else {
                                format!(
                                    "owner group {} — this player joined after the film header was written, \
                                     so the recording never stored their gamertag. Every number here is \
                                     still theirs.",
                                    p.owner
                                )
                            });
                            match p.score {
                                Some(v) => {
                                    ui.label(format!("{v}")).on_hover_text(format!(
                                        "Decoded from the engine's own game-engine-statborg score store \
                                         (no inference: its index IS this player's slot). Round score {}, \
                                         rounds won {}, over {} recorded update(s).",
                                        p.round_score.unwrap_or(0),
                                        p.rounds_won.unwrap_or(0),
                                        p.score_updates
                                    ));
                                }
                                None => {
                                    ui.label(egui::RichText::new("-").weak()).on_hover_text(
                                        "The recording carries no score update for this player. That is not \
                                         zero -- a film only carries the updates it was sent, and a clip \
                                         whose gametype never awards a point sends none.",
                                    );
                                }
                            }
                            ui.label(format!("{}", p.lives));
                            ui.label(format!("{}", p.deaths)).on_hover_text(
                                "A body's track ending: the entity DESTROY message, which is decoded. \
                                 Attributing it to a player goes through the owner stamp, so the \
                                 per-player figure is derived.",
                            );
                            ui.label(fmt_time(p.alive_ticks as f32 / 60.0));
                            ui.label(format!("{:.0}", p.distance_wu)).on_hover_text(
                                "A LOWER BOUND. The recording replicates a position on only some ticks, \
                                 so a straight line between two keys cuts the corners off the real path.",
                            );
                            ui.label(format!("{:.1}", p.top_foot_speed_wu_s)).on_hover_text(format!(
                                "Top speed on foot. Including man-cannon launches and teleports it is {:.1} wu/s.",
                                p.top_speed_wu_s
                            ));
                            ui.label(if p.vehicle_ticks > 0 {
                                fmt_time(p.vehicle_ticks as f32 / 60.0)
                            } else {
                                "-".into()
                            })
                            .on_hover_text(if p.rides.is_empty() {
                                "No ride detected. The detector keys on matched HORIZONTAL displacement \
                                 between a body and a vehicle, and refuses a hop-in-hop-out that never \
                                 went anywhere."
                                    .to_string()
                            } else {
                                p.rides
                                    .iter()
                                    .take(RIDES_LISTED)
                                    .map(|r| {
                                        format!(
                                            "{} for {:.1}s from {}",
                                            r.vehicle.clone().unwrap_or_else(|| "(vehicle not named)".into()),
                                            r.seconds,
                                            fmt_time(r.from_tick as f32 / 60.0)
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            });
                            ui.end_row();
                        }
                    });
                });
            ui.add_space(4.0);
            // The honesty block. Every statistic above is labelled, and the ones a Reach film
            // does NOT carry are named outright so nobody goes looking for a kill feed.
            egui::CollapsingHeader::new("Where these numbers come from")
                .id_salt("film_stats_prov")
                .default_open(false)
                .show(ui, |ui| {
                    for n in &s.notes {
                        ui.label(egui::RichText::new(n).weak().small());
                        ui.add_space(2.0);
                    }
                });
            ui.label(
                egui::RichText::new(
                    "Per-player kill TOTALS are not in a Reach saved film; the score, and the kill \
                     FEED, are.",
                )
                .weak()
                .small(),
            )
            .on_hover_text(SCORE_NEGATIVE);
        });

    // #film-action  The replicated EVENT channel, as its own card: the kill feed, the bodies
    // named by the definition they were spawned from, and vehicle entries and exits with the
    // seat the stream states.
    let Some(s) = stats else { return };
    if s.actions.decoded == 0 && s.bipeds.is_empty() {
        return;
    }
    egui::CollapsingHeader::new("What happened (read off the recording)")
        .id_salt("film_action_panel")
        .default_open(false)
        .open(force_open.then_some(true))
        .show(ui, |ui| {
            if !s.bipeds.is_empty() {
                ui.label(egui::RichText::new("bodies, by the character the recording names").strong().small());
                let mut b: Vec<_> = s.bipeds.iter().collect();
                b.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                for (k, c) in b {
                    ui.label(egui::RichText::new(format!("   {c:>5}  {k}")).small()).on_hover_text(
                        "Every spawn message in the recording opens with a 14-bit reference to the \
                         object definition the body was spawned from, so this is DECODED -- HMS no \
                         longer assumes a Spartan. Open a map to resolve the reference to a name.",
                    );
                }
                ui.add_space(4.0);
            }
            if s.kills.is_empty() {
                ui.label(egui::RichText::new("This clip's event channel carried no kill incidents.").weak().small());
            } else {
                let assn = s.kills.iter().filter(|k| k.assassination).count();
                ui.label(
                    egui::RichText::new(format!("kill feed — {} row(s), {assn} assassination(s)", s.kills.len()))
                        .strong()
                        .small(),
                )
                .on_hover_text(
                    "Decoded. The engine broadcasts each Reach medal as an `incident` event naming \
                     the player who earned it and the other party, and its leading 32 bits are the \
                     engine's own string id -- so the kind of kill is read, not guessed. An \
                     assassination is `finishing_move_kill`: on the three gametypes whose titles \
                     are about assassination, 34 of 34 kills are one, against 68 of 1,904 \
                     elsewhere.",
                );
                egui::ScrollArea::vertical()
                    .id_salt("film_action_kills")
                    .min_scrolled_height(PANEL_HEIGHT * 0.6)
                    .max_height(PANEL_HEIGHT * 0.6)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for k in &s.kills {
                            let who = |n: &Option<String>, id: Option<u32>| {
                                n.clone().unwrap_or_else(|| id.map(|v| format!("{v:#010x}")).unwrap_or_else(|| "-".into()))
                            };
                            let line = format!(
                                "{:>8}  {:<22} {} -> {}",
                                fmt_time(k.seconds),
                                k.incident,
                                who(&k.killer, k.killer_net_id),
                                who(&k.victim, k.victim_net_id)
                            );
                            let t = if k.assassination {
                                egui::RichText::new(format!("{line}   ASSASSINATION")).small().strong()
                            } else {
                                egui::RichText::new(line).small()
                            };
                            ui.label(t).on_hover_text(format!(
                                "incident {:#010x} at tick {}. The names come through the owner bridge, \
                                 which is derived; the incident and the two parties are decoded.",
                                k.incident_id, k.tick
                            ));
                        }
                    });
            }
            if !s.vehicle_transitions.is_empty() {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("vehicle entries / exits — {} row(s)", s.vehicle_transitions.len()))
                        .strong()
                        .small(),
                )
                .on_hover_text(
                    "Decoded, including the SEAT index -- the stream states it as a bare 6-bit \
                     field, so this is no longer worked out from two bodies moving together.",
                );
                for v in s.vehicle_transitions.iter().take(KILLS_SHOWN) {
                    ui.label(
                        egui::RichText::new(format!(
                            "   {:>8}  {:<20} {} seat {}  {}",
                            fmt_time(v.seconds),
                            v.event,
                            v.player.clone().unwrap_or_else(|| "(unnamed)".into()),
                            v.seat.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
                            v.vehicle.clone().unwrap_or_else(|| "(vehicle not named)".into())
                        ))
                        .small(),
                    );
                }
            }
            ui.add_space(4.0);
            let mut ev: Vec<_> = s.actions.by_event.iter().collect();
            ev.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            ui.label(
                egui::RichText::new(format!(
                    "the whole channel: {} event(s) over {} type(s)",
                    s.actions.decoded,
                    s.actions.by_event.len()
                ))
                .weak()
                .small(),
            )
            .on_hover_text(
                ev.iter().map(|(k, c)| format!("{k}  {c}")).collect::<Vec<_>>().join("\n"),
            );
        });
}

#[cfg(test)]
mod probes;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_table_marks_the_refused_statistics() {
        let t = provenance_table();
        // The point of the table: a consumer can see that kill TOTALS were considered and
        // refused -- which is still true, and is a different claim from the kill FEED.
        for k in ["kills", "medals", "assists", "damage", "betrayals", "suicides", "shots_fired"] {
            assert_eq!(t.get(k).copied(), Some("not available"), "{k}");
        }
        // #film-action  The FEED, the seat index and the body's own character are decoded, and
        // the table has to say so or a consumer will keep believing the old blanket negative.
        for k in ["actions", "kills_feed", "kills_feed.incident", "vehicle_transitions.seat", "bipeds", "biped"] {
            assert_eq!(t.get(k).copied(), Some("decoded"), "{k}");
        }
        // Putting a NAME on a party still goes through the owner bridge.
        assert_eq!(t.get("kills_feed.killer").copied(), Some("derived"));
        // The SCORE is a different story: it is in the stream, in the engine's own statborg
        // store, and it is decoded rather than inferred.
        for k in ["score", "round_score", "rounds_won", "lives", "weapons"] {
            assert_eq!(t.get(k).copied(), Some("decoded"), "{k}");
        }
        assert_eq!(t.get("deaths").copied(), Some("derived"));
    }

    /// The union, not the sum: a corpse that lingers past the respawn overlaps the next body, and
    /// adding both lifetimes would inflate time alive past the length of the clip.
    #[test]
    fn overlapping_lives_are_counted_once() {
        let mut idx = FilmIndex { ticks: 1000, ..Default::default() };
        // Two bodies of one player, overlapping 0..600 and 500..900.
        for (first, last, destroy) in [(0u32, 600u32, Some(600u32)), (500, 900, Some(900))] {
            idx.tracks.push(EntityTrack {
                net_id: first + 1,
                entity_type: BIPED,
                first_tick: first,
                last_tick: last,
                owner_raw: Some(3),
                destroyed_tick: destroy,
                moves_cached: Some(false),
                ..Default::default()
            });
        }
        let film = crate::film::Film {
            bytes: Vec::new(),
            chunks: Vec::new(),
            container_name: String::new(),
            chdr_leading: 0,
            content: Default::default(),
            author: Default::default(),
            header: Default::default(),
            data: Default::default(),
        };
        // `compute` needs an entity stream to produce rows.
        idx.entity_messages = 2;
        let s = compute(&film, &idx, &NoNames);
        assert_eq!(s.players.len(), 1);
        let p = &s.players[0];
        assert_eq!(p.lives, 2);
        // 0..900 as one span, not 600 + 400.
        assert_eq!(p.alive_ticks, 900);
        assert_eq!(p.longest_life_ticks, 600);
        assert_eq!(p.deaths, 2, "both destroys are well clear of the final cut");
    }

    /// A destroy in the last second of the recording is the end-of-match teardown, not a death.
    #[test]
    fn the_final_teardown_is_not_a_death() {
        let mut idx = FilmIndex { ticks: 1000, entity_messages: 2, ..Default::default() };
        idx.tracks.push(EntityTrack {
            net_id: 1,
            entity_type: BIPED,
            first_tick: 0,
            last_tick: 300,
            owner_raw: Some(1),
            destroyed_tick: Some(300),
            moves_cached: Some(false),
            ..Default::default()
        });
        idx.tracks.push(EntityTrack {
            net_id: 2,
            entity_type: BIPED,
            first_tick: 310,
            last_tick: 999,
            owner_raw: Some(1),
            destroyed_tick: Some(999),
            moves_cached: Some(false),
            ..Default::default()
        });
        let film = crate::film::Film {
            bytes: Vec::new(),
            chunks: Vec::new(),
            container_name: String::new(),
            chdr_leading: 0,
            content: Default::default(),
            author: Default::default(),
            header: Default::default(),
            data: Default::default(),
        };
        let s = compute(&film, &idx, &NoNames);
        let p = &s.players[0];
        assert_eq!(p.deaths, 1);
        assert_eq!(p.deaths_at_cut, 1);
        assert_eq!(p.lives, 2);
    }
}
