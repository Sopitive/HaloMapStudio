//! The CROSS-GAME OBJECT CATALOG (#xgame): every Forge palette object of every installed title,
//! with the render-model bounding box in WORLD UNITS that makes a Coliseum Wall recognisable
//! whatever it is called.
//!
//! One row per placeable SHAPE = one `(Forge category, palette entry, palette variant)` triple.
//! The row is keyed on the OBJECT ([`crate::xgame::object_key`] = tag leaf + variant), never on
//! the map, because a title's palette differs per canvas: `canvases` lists the canvases the
//! object appears on and `places` carries the per-canvas palette coordinates the converter needs.
//!
//! Where the box comes from (no GPU in any path):
//!   * Reach - the native parser: `obje -> mode`, then the union of every render section's
//!     COMPRESSION bounds (`ZhModelSection::bounds_min/max`, the snorm dequantisation range, so
//!     already world units in model space). Hidden blocks have no render geometry, so their box
//!     falls back to the collision (`coll`) then physics (`phmo`) hull.
//!   * Halo 4 / H2A - `h4::objects::load_model` and the union of the variant's meshes'
//!     `MeshDef::bounds`.
//!   * Halo 3 - `h3::objects::load_model` and its compression table (`H3Model::mesh_bounds`).
//!
//! Build it with `hms-app --xgame-build-catalog [out.json]`, read it with
//! `--xgame-catalog [filter]`. The shipped copy is baked in with `include_str!`; a
//! `xgame_catalog.json` in the HMS settings dir overrides it with no rebuild.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::mapcat::Game;

use super::{object_key, tag_leaf};

/// The baked-in catalog (regenerate with `--xgame-build-catalog`).
pub const BUILTIN_CATALOG: &str = include_str!("catalog.json");
/// File name of the user override in the HMS settings dir.
pub const OVERRIDE_FILE: &str = "xgame_catalog.json";

/// Where one object sits in ONE canvas's palette - what the converter needs to emit a record.
///
/// Serialized as ONE compact string `canvas:category:quota:variant[:h]` (`h` = a hidden
/// superforge category), because a catalog carries ~15 000 of these and the file is baked into
/// the executable - the object form cost 3x the bytes for nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaletteSpot {
    /// Canvas map stem (`forge_halo`, `ca_forge_ravine`, `sandbox`).
    pub canvas: String,
    /// Reach / H4: the Forge CATEGORY index in the scnr palette block. H3: the sandbox block.
    pub category_index: u32,
    /// Reach: `folder` (the flat quota index = position in `forge_type_order`).
    /// Halo 4 / H2A: the flat palette `quota` index. Halo 3: the palette ROW index.
    pub quota: u32,
    /// Reach: `item`. Halo 4: the variant index inside the entry. Halo 3: always 0.
    pub variant: u32,
    /// True when this spot is in a HIDDEN (superforge) category - the monitor can never grab it,
    /// so the converter never places into it.
    pub hidden: bool,
}

impl std::fmt::Display for PaletteSpot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}:{}:{}:{}{}", self.canvas, self.category_index, self.quota, self.variant, if self.hidden { ":h" } else { "" })
    }
}

impl std::str::FromStr for PaletteSpot {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let p: Vec<&str> = s.split(':').collect();
        if p.len() < 4 { return Err(format!("palette spot '{s}': want canvas:category:quota:variant[:h]")); }
        let n = |i: usize| p[i].parse::<u32>().map_err(|_| format!("palette spot '{s}': field {i} is not a number"));
        Ok(PaletteSpot { canvas: p[0].to_string(), category_index: n(1)?, quota: n(2)?, variant: n(3)?, hidden: p.get(4) == Some(&"h") })
    }
}

impl Serialize for PaletteSpot {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { s.collect_str(self) }
}

impl<'de> Deserialize<'de> for PaletteSpot {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// One placeable shape of one title.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CatalogObject {
    /// `"reach" | "halo4" | "h2a" | "halo3"`.
    pub game: String,
    /// Stable cross-canvas identity (see [`crate::xgame::object_key`]).
    pub key: String,
    /// Forge menu category, raw name (`ff_structure`, `STRUCTURES`, `Scenery`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub category: String,
    /// Palette entry raw name (`structure_doors`, `ff_doors_win_wall`).
    pub entry: String,
    /// Palette variant raw name (`wall_coliseum`, `drw_wall_coliseum`); empty on single-variant.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub variant: String,
    /// Localized Forge menu name (`Wall, Coliseum`) - the raw name when no string table resolves.
    pub display: String,
    /// Object tag path, and its 4CC class (`bloc`, `scen`, `weap`, `vehi`, `eqip`, `mach`).
    pub tag: String,
    pub class: String,
    /// Render-model tag path; empty when the object has no drawable model (markers, zones).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// Render-model AABB SIZE in world units `[x, y, z]`, model space. `[0,0,0]` = no box found.
    pub dims: [f32; 3],
    /// Centre of that AABB relative to the object's origin (world units), so a converter can see
    /// that e.g. a Reach block is centred while its Halo 4 twin sits on its base.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub center: [f32; 3],
    /// Where the box came from: `render` / `collision` / `physics` / `none`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub box_source: String,
    /// Render part count (a shape discriminator: a railing has many parts, a block has one).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub parts: u32,
    /// Multiplayer object type of the title (Reach `cached_type`, H4 6-bit type, H3 `object_type`).
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub object_type: u8,
    /// Multiplayer boundary shape: 0 none, 1 sphere, 2 cylinder, 3 box.
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub shape: u8,
    /// Default respawn/spawn time in seconds authored on the object.
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub spawn_time: u16,
    /// Forge price per instance and `maximum allowed` (<= 0 = unlimited).
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub price: i32,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub max_allowed: i32,
    /// The shape FAMILY this row was classified into - the matcher's hard gate ([`Kind`]).
    pub kind: String,
    /// Canvas stems this object appears on (sorted, deduped). DERIVED from `places`, so it is
    /// never stored in the file (`Catalog::from_json` rebuilds it).
    #[serde(skip)]
    pub canvases: Vec<String>,
    /// Per-canvas palette coordinates (sorted by canvas).
    pub places: Vec<PaletteSpot>,
}

fn is_zero3(v: &[f32; 3]) -> bool { v.iter().all(|&x| x == 0.0) }
fn is_zero_u32(v: &u32) -> bool { *v == 0 }
fn is_zero_u8(v: &u8) -> bool { *v == 0 }
fn is_zero_u16(v: &u16) -> bool { *v == 0 }
fn is_zero_i32(v: &i32) -> bool { *v == 0 }

impl CatalogObject {
    /// `dims` sorted descending - the rotation-independent size signature a wall is recognised by.
    pub fn sorted_dims(&self) -> [f32; 3] {
        let mut d = self.dims;
        d.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        d
    }
    pub fn has_box(&self) -> bool { self.dims.iter().any(|&v| v > 1e-4) }
    pub fn kind(&self) -> Kind { Kind::from_str(&self.kind) }
    /// The non-hidden palette spot on `canvas`, else any spot on it.
    pub fn spot_on(&self, canvas: &str) -> Option<&PaletteSpot> {
        self.places.iter().find(|p| p.canvas.eq_ignore_ascii_case(canvas) && !p.hidden)
            .or_else(|| self.places.iter().find(|p| p.canvas.eq_ignore_ascii_case(canvas)))
    }
    pub fn volume(&self) -> f32 { self.dims[0].max(0.0) * self.dims[1].max(0.0) * self.dims[2].max(0.0) }
}

/// The whole catalog.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Catalog {
    /// Free-form note, written by the builder (host, date, map count).
    #[serde(default)]
    pub note: String,
    pub objects: Vec<CatalogObject>,
}

impl Catalog {
    pub fn from_json(s: &str) -> Result<Catalog, String> {
        let mut c: Catalog = serde_json::from_str(s).map_err(|e| format!("catalog json: {e}"))?;
        for o in &mut c.objects {
            o.canvases = o.places.iter().map(|p| p.canvas.clone()).collect();
            o.canvases.sort();
            o.canvases.dedup();
        }
        Ok(c)
    }
    /// COMPACT json (one line per object): the file is baked into the executable with
    /// `include_str!`, so pretty-printing it would cost megabytes for nothing.
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\n\"note\": {},\n\"objects\": [\n", serde_json::to_string(&self.note).unwrap_or_default());
        for (i, o) in self.objects.iter().enumerate() {
            s.push_str(&serde_json::to_string(o).unwrap_or_default());
            if i + 1 < self.objects.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("]}\n");
        s
    }

    /// The catalog HMS uses: the settings-dir override when present, else the baked-in copy.
    /// Never fails - a broken override logs and falls back.
    pub fn load() -> Catalog {
        if let Some(p) = crate::project::settings_dir().map(|d| d.join(OVERRIDE_FILE)) {
            if p.is_file() {
                match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Catalog::from_json(&s)) {
                    Ok(c) => { log::info!("#xgame: catalog override {} ({} objects)", p.display(), c.objects.len()); return c; }
                    Err(e) => log::warn!("#xgame: ignoring {}: {e}", p.display()),
                }
            }
        }
        Catalog::from_json(BUILTIN_CATALOG).unwrap_or_else(|e| { log::error!("#xgame: baked-in catalog is broken: {e}"); Catalog::default() })
    }

    pub fn get(&self, game: Game, key: &str) -> Option<&CatalogObject> {
        let g = game.short_name();
        self.objects.iter().find(|o| o.game == g && o.key == key)
    }
    pub fn of_game(&self, game: Game) -> impl Iterator<Item = &CatalogObject> {
        let g = game.short_name();
        self.objects.iter().filter(move |o| o.game == g)
    }
    /// Objects of `game` placeable on `canvas` (a non-hidden palette spot there).
    pub fn on_canvas(&self, game: Game, canvas: &str) -> Vec<&CatalogObject> {
        let c = canvas.to_string();
        self.of_game(game).filter(|o| o.places.iter().any(|p| p.canvas.eq_ignore_ascii_case(&c) && !p.hidden)).collect()
    }
    pub fn canvases(&self, game: Game) -> Vec<String> {
        let mut v: Vec<String> = self.of_game(game).flat_map(|o| o.canvases.iter().cloned()).collect();
        v.sort();
        v.dedup();
        v
    }
}

// ---------------------------------------------------------------------------------------------
// shape families
// ---------------------------------------------------------------------------------------------

/// The shape FAMILY of a palette object. The matcher only ever pairs objects whose kinds are
/// compatible ([`Kind::group`]), so a Coliseum Wall can never become a Warthog however the
/// numbers fall. Gameplay kinds are the 1:1 rows - they exist in every title and carry over
/// exactly; structural kinds are matched by SIZE within their group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    // --- gameplay: 1:1 across titles ---
    SpawnInitial,
    SpawnRespawn,
    SpawnCamera,
    RespawnZone,
    RespawnZoneWeak,
    RespawnZoneAnti,
    SafeZoneSoft,
    SafeZoneHard,
    KillZoneSoft,
    KillZoneHard,
    FlagStand,
    HillMarker,
    CapturePlate,
    BallSpawn,
    LocationName,
    TeleporterSender,
    TeleporterReceiver,
    TeleporterTwoWay,
    Weapon,
    Grenade,
    Equipment,
    ArmorAbility,
    Powerup,
    Vehicle,
    Turret,
    OrdnanceDrop,
    Objective,
    // --- structure / scenery: matched by size ---
    Block,
    Wall,
    Window,
    Door,
    Ramp,
    Bridge,
    Platform,
    Corner,
    Railing,
    Cover,
    Building,
    Pipe,
    Crate,
    NaturalRock,
    NaturalTree,
    Light,
    ShieldDoor,
    OneWayDoor,
    ManCannon,
    Explosive,
    Fx,
    Decorative,
    Scenery,
    Unknown,
}

impl Kind {
    pub fn name(self) -> &'static str {
        use Kind::*;
        match self {
            SpawnInitial => "spawn_initial", SpawnRespawn => "spawn_respawn", SpawnCamera => "spawn_camera",
            RespawnZone => "respawn_zone", RespawnZoneWeak => "respawn_zone_weak", RespawnZoneAnti => "respawn_zone_anti",
            SafeZoneSoft => "safe_zone_soft", SafeZoneHard => "safe_zone_hard",
            KillZoneSoft => "kill_zone_soft", KillZoneHard => "kill_zone_hard",
            FlagStand => "flag_stand", HillMarker => "hill_marker", CapturePlate => "capture_plate",
            BallSpawn => "ball_spawn", LocationName => "location_name",
            TeleporterSender => "teleporter_sender", TeleporterReceiver => "teleporter_receiver", TeleporterTwoWay => "teleporter_two_way",
            Weapon => "weapon", Grenade => "grenade", Equipment => "equipment", ArmorAbility => "armor_ability",
            Powerup => "powerup", Vehicle => "vehicle", Turret => "turret", OrdnanceDrop => "ordnance_drop", Objective => "objective",
            Block => "block", Wall => "wall", Window => "window", Door => "door", Ramp => "ramp", Bridge => "bridge",
            Platform => "platform", Corner => "corner", Railing => "railing", Cover => "cover", Building => "building",
            Pipe => "pipe", Crate => "crate", NaturalRock => "natural_rock", NaturalTree => "natural_tree",
            Light => "light", ShieldDoor => "shield_door", OneWayDoor => "one_way_door", ManCannon => "man_cannon",
            Explosive => "explosive", Fx => "fx", Decorative => "decorative", Scenery => "scenery", Unknown => "unknown",
        }
    }
    pub fn from_str(s: &str) -> Kind {
        use Kind::*;
        for k in ALL_KINDS { if k.name() == s { return k; } }
        Unknown
    }
    /// Kinds that name a GAMEPLAY role: these must match EXACTLY across titles (no size test).
    pub fn is_gameplay(self) -> bool {
        use Kind::*;
        matches!(self, SpawnInitial | SpawnRespawn | SpawnCamera | RespawnZone | RespawnZoneWeak | RespawnZoneAnti
            | SafeZoneSoft | SafeZoneHard | KillZoneSoft | KillZoneHard | FlagStand | HillMarker | CapturePlate
            | BallSpawn | LocationName | TeleporterSender | TeleporterReceiver | TeleporterTwoWay | Weapon | Grenade
            | Equipment | ArmorAbility | Powerup | Vehicle | Turret | OrdnanceDrop | Objective)
    }
    /// True when this kind is a FORGE BUILDING PIECE - something a player walks on, through or
    /// behind. The hard gate in `table::kind_fit` never lets one of these pair with a prop.
    pub fn is_structural(self) -> bool {
        use Kind::*;
        matches!(self, Block | Platform | Wall | Window | Door | Corner | Bridge | Ramp | Building | Pipe | Railing)
    }
    /// True when this kind is an EFFECT EMITTER whose render box is incidental - a man cannon's
    /// push, a light's colour and an FX orb's filter are what the object is for, and its model is
    /// a small marker that says nothing about the gameplay. These match on ROLE like the gameplay
    /// kinds, not on size.
    pub fn is_gadget(self) -> bool {
        use Kind::*;
        matches!(self, ManCannon | Light | Fx)
    }
    /// True when this kind is set-dressing: a crate, a rock, a generator, an emplacement.
    pub fn is_prop(self) -> bool {
        use Kind::*;
        matches!(self, Crate | Cover | Decorative | Scenery | Explosive | NaturalRock | NaturalTree)
    }

    /// The kinds this one may substitute for when the exact kind is absent. The matcher prefers an
    /// identical kind and only falls back INSIDE this list, which never crosses the
    /// structure / prop line: a Reach 1x4 platform can reach a Halo 4 block and nothing else.
    pub fn group(self) -> &'static [Kind] {
        use Kind::*;
        const DECK: &[Kind] = &[Block, Platform, Building];           // things you stand on
        const SLOPE: &[Kind] = &[Ramp, Bridge];                        // things you walk up / across
        const SPAN: &[Kind] = &[Bridge, Ramp, Platform, Block];
        const THIN: &[Kind] = &[Wall, Window, Door, Corner];           // things you walk through / behind
        const RAIL: &[Kind] = &[Railing, Cover, Wall, Decorative];     // waist-high barriers
        const TUBE: &[Kind] = &[Pipe, Decorative, Building];
        const PROP: &[Kind] = &[Crate, Cover, Decorative, Scenery, Explosive];
        const NATURAL: &[Kind] = &[NaturalRock, NaturalTree, Scenery];
        const BARRIER: &[Kind] = &[ShieldDoor, OneWayDoor, Window];
        match self {
            Block | Platform | Building => DECK,
            Ramp => SLOPE,
            Bridge => SPAN,
            Wall | Window | Door | Corner => THIN,
            Railing => RAIL,
            Pipe => TUBE,
            Crate | Cover | Decorative | Scenery | Explosive => PROP,
            NaturalRock | NaturalTree => NATURAL,
            ShieldDoor | OneWayDoor => BARRIER,
            // Gameplay kinds have no fallback group: they match their own kind EXACTLY or not at
            // all (`table::kind_fit`). Light / ManCannon / Fx / Turret are likewise exact-only.
            _ => &[],
        }
    }
}

pub const ALL_KINDS: [Kind; 50] = {
    use Kind::*;
    [SpawnInitial, SpawnRespawn, SpawnCamera, RespawnZone, RespawnZoneWeak, RespawnZoneAnti, SafeZoneSoft, SafeZoneHard,
     KillZoneSoft, KillZoneHard, FlagStand, HillMarker, CapturePlate, BallSpawn, LocationName, TeleporterSender,
     TeleporterReceiver, TeleporterTwoWay, Weapon, Grenade, Equipment, ArmorAbility, Powerup, Vehicle, Turret,
     OrdnanceDrop, Objective, Block, Wall, Window, Door, Ramp, Bridge, Platform, Corner, Railing, Cover, Building,
     Pipe, Crate, NaturalRock, NaturalTree, Light, ShieldDoor, OneWayDoor, ManCannon, Explosive, Fx, Decorative, Scenery]
};

/// Classify a palette row into a [`Kind`] from everything the palette says about it. Names are
/// the primary signal (they are remarkably consistent across the four titles); the multiplayer
/// object type and the tag class break ties, and the box shape separates a flat block from a wall.
/// Classify a palette row into a [`Kind`].
///
/// `tag` is the object's TAG LEAF (`fw_unsc_c_wall_2x1_door`, `cov_artillery_gun`) and carries
/// more signal than anything else: a palette entry may be called `ff_thorage_covenant / blue`
/// while the tag says exactly what the model is. Leaving it out is how Reach's
/// `ff_reactor_door_blue` - whose palette row is only named "blue" - came out a block instead of
/// a door.
pub fn classify(tag: &str, category: &str, entry: &str, variant: &str, display: &str, class: &str, object_type: u8, game: Game, dims: [f32; 3]) -> Kind {
    use Kind::*;
    // The tag's TITLE / THEME PREFIX says which Forge set the piece belongs to, never what it is,
    // and reading it as a description is actively wrong: Halo 2 Anniversary names its ENTIRE
    // Forge set `fw_stone_*`, which made all 167 of its structures read as rock.
    let bare = strip_tag_prefix(tag);
    let hay = format!("{bare} {category} {entry} {variant} {display}").to_ascii_lowercase();
    let has = |n: &str| hay.contains(n);
    let any = |ns: &[&str]| ns.iter().any(|n| hay.contains(n));
    // WORD test: a substring test alone mistakes `chill_viewing_area` for a hill and
    // `fw_stone_terrain_hill_lg` for a KOTH marker, so the role checks use whole words.
    let words: Vec<&str> = hay.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let word = |n: &str| words.iter().any(|w| *w == n);

    // --- gameplay first: a name that names a role wins over any shape reasoning ---
    if any(&["initial spawn", "spawning_initial", "ff_initial_spawn", "initial_spawn", "starting_point", "spawn_point_initial"]) { return SpawnInitial; }
    if any(&["respawn point", "spawning_respawn", "respawn_point", "spawn_point", "ff_respawn_point"]) && !has("zone") { return SpawnRespawn; }
    if any(&["spawning_camera", "loadout camera", "spawn camera", "respawn_camera", "loadout_camera"]) { return SpawnCamera; }
    if has("respawn_zone_anti") || (has("anti") && has("respawn")) { return RespawnZoneAnti; }
    if has("respawn_zone_weak") || (has("weak") && has("respawn")) { return RespawnZoneWeak; }
    if any(&["respawn_zone", "respawn zone", "respawn area"]) { return RespawnZone; }
    if has("safe") && (has("hard") || has("boundary")) && !has("soft") { return SafeZoneHard; }
    if has("soft safe") || (has("safe") && has("soft")) { return SafeZoneSoft; }
    if any(&["safe_volume", "safe area", "safe boundary", "spawning_boundaries_safe", "safe_zone"]) { return SafeZoneHard; }
    // a hazard BALL is a prop, not a boundary: keep it out of the kill-zone family so a Reach
    // kill BOUNDARY can never be converted into Halo 4's Kill Ball (or Halo 3's damage sphere)
    if any(&["kill ball", "kill_ball", "damage_sphere", "damage sphere"]) { return Explosive; }
    if has("soft kill") || (has("kill") && has("soft")) { return KillZoneSoft; }
    if any(&["kill_volume", "kill ball", "kill area", "kill boundary", "spawning_boundaries_kill", "kill_zone"]) { return KillZoneHard; }
    if any(&["flag_base", "flag stand", "flag_stand", "capture_flag", "ctf_flag", "flag return", "flag_return"]) { return FlagStand; }
    // a KOTH marker, not a mound of terrain called "hill": the role names only.
    if any(&["hill_marker", "hill marker", "king of the hill", "koth_hill", "zone_koth"]) || word("koth") { return HillMarker; }
    if any(&["obj_plate", "capture plate", "territor", "capture_plate", "oddball_plate",
             "goal_area", "goal area", "destination_static", "vip_boundary", "_zone_static"]) { return CapturePlate; }
    if any(&["ball spawn", "ball_spawn", "oddball", "skull"]) { return BallSpawn; }
    if any(&["obj_location_name", "location name", "location_name", "named location"]) { return LocationName; }
    if has("sender") { return TeleporterSender; }
    if has("receiver") || has("reciever") { return TeleporterReceiver; }
    if any(&["teleporter", "teleport", "two-way", "two way node", "gadget_teleporters"]) { return TeleporterTwoWay; }
    if any(&["oddball_ball_spawn", "ball_spawn_point", "bomb_spawn_point", "flag_spawn_point"]) { return BallSpawn; }
    if any(&["ordnance", "drop pod", "care package"]) { return OrdnanceDrop; }
    if any(&["overshield", "over shield", "active camo", "camouflage", "damage boost", "speed boost", "custom powerup", "powerup", "power-up", "gadget_powerups"]) { return Powerup; }
    if has("armor abilit") || has("armour abilit") || category.to_ascii_lowercase().contains("armor_abilities") || entry.starts_with("arm_") || entry.starts_with("aa_") { return ArmorAbility; }
    if any(&["grenade", "_nade"]) { return Grenade; }
    if any(&["turret", "machinegun", "machine gun", "plasma cannon", "plasmaturret", "shade"]) && class == "vehi" { return Turret; }
    if class == "weap" { return Weapon; }
    if class == "vehi" { return Vehicle; }
    if class == "eqip" { return Equipment; }
    // the multiplayer object type says what the engine will treat it as, whatever it is called
    match (game, object_type) {
        (Game::Halo4, 1) | (Game::H2A, 1) => return Weapon,
        (Game::Halo4, 2) | (Game::H2A, 2) => return Grenade,
        (Game::Halo4, 5) | (Game::H2A, 5) => return Equipment,
        (Game::Halo4, 7..=9) | (Game::H2A, 7..=9) => return Vehicle,
        (Game::Halo4, 13) | (Game::H2A, 13) => return TeleporterSender,
        (Game::Halo4, 14) | (Game::H2A, 14) => return TeleporterReceiver,
        (Game::Halo4, 15) | (Game::H2A, 15) => return TeleporterTwoWay,
        (Game::Halo4, 16) | (Game::H2A, 16) => return SpawnRespawn,
        (Game::Halo4, 17) | (Game::H2A, 17) => return RespawnZone,
        (Game::Halo4, 26) | (Game::H2A, 26) => return SafeZoneHard,
        (Game::Halo4, 27) | (Game::H2A, 27) => return KillZoneHard,
        (Game::Halo4, 28) | (Game::H2A, 28) => return SpawnCamera,
        (Game::Reach, 1) => return Weapon,
        (Game::Reach, 12) => return TeleporterTwoWay,
        (Game::Reach, 13) => return TeleporterSender,
        (Game::Reach, 14) => return TeleporterReceiver,
        (Game::Reach, 19) => return LocationName,
        _ => {}
    }

    // --- gadgets / fx ---
    if any(&["man cannon", "man_cannon", "mancannon", "grav lift", "gravity lift", "gravlift", "grav_lift", "gravity_lift"]) { return ManCannon; }
    if any(&["shield door", "door_shield", "shield_door", "shield wall"]) { return ShieldDoor; }
    if any(&["one way", "oneway", "door_oneway"]) { return OneWayDoor; }
    if any(&["light", "lamp", "spotlight"]) && !has("lightrifle") && !has("hardlight") { return Light; }
    if any(&["fusion coil", "fusion_coil", "explosive", "propane", "plasma battery", "gadget_explosive"]) { return Explosive; }
    if any(&["gadget_sfx", "screen fx", "sfx", "fx_", "effect",
             // the Halo 3 Forge FX palette, by its tag names
             "nova", "juicy", "colorblind", "color_blind", "gloomy", "olde_timey", "pen_and_ink",
             "next_gen", "old_timey", "purple_fx", "greenscreen"]) { return Fx; }

    // --- natural --- (whole words: `stone_` as a substring matched every H2A Forge tag)
    if any(&["rock", "boulder", "cliff"]) || word("stone") || word("rocks") { return NaturalRock; }
    if word("tree") || any(&["pine", "sequoia", "seq_tree"]) || word("bush") || word("shrub") { return NaturalTree; }

    // --- structure ---
    // A Forge BUILDING PIECE has to EARN that family. The old code sent every unclassified
    // `bloc`/`scen` tag through `block_or_wall`, which is how a Covenant artillery emplacement
    // (`ff_thorage_scenery / ff_thorage_covenant / ff_dom_turretpad`, a 2.1 x 0.8 x 1.6 bloc)
    // became a `block` and then stood in for a Reach 1x4 platform. Structural kinds are now gated
    // on the palette CATEGORY / ENTRY naming - the Forge menu's own "Structure" section in each
    // title - so a prop can never be offered as a floor to walk on.
    let structural = any(&[
        // Reach
        "ff_structure", "structure_blocks", "structure_bridges", "structure_inclines",
        "structure_doors", "structure_buildings", "structure_decorative", "structure_platforms",
        "block_", "plat_", "bridge_", "ramp_", "bank_", "wall_", "window_", "corner_", "room_",
        // Halo 4 / H2A (the ff_* Structure categories and their bb_/bp_/inc_/drw_/dec_ prefixes)
        "ff_building_blocks", "ff_bridges_platforms", "ff_inclines", "ff_doors_win_wall",
        "ff_decorative", "ff_buildings", "bb_", "bp_", "inc_", "drw_", "dec_",
        // Halo 3 (Sandbox / Foundry Forge pieces)
        "sand_", "fore_",
    ]) || class == "bloc" && any(&["ff_ravine", "ff_erosion", "ff_bonanza", "ff_dropoff", "ff_creeper", "ff_skyline"]);

    if any(&["coliseum", "arena_wall", "arena_window"]) {
        return if has("window") { Window } else { Wall };
    }
    if any(&["window"]) { return Window; }
    if any(&["door"]) { return Door; }
    if any(&["railing", "rail_", "guard rail", "handrail", "barrier"]) { return Railing; }
    if any(&["ramp", "incline", "slope", "wedge"]) { return Ramp; }
    if any(&["bridge", "walkway", "catwalk", "gangway"]) { return Bridge; }
    if any(&["corner"]) { return Corner; }
    if any(&["pipe", "tube", "conduit"]) { return Pipe; }
    if any(&["crate", "box_", "container", "ammo", "case", "dumpster", "pallet", "barrel", "canister",
             "drum", "cabinet", "stool", "locker", "radio", "tray", "spool", "cart_", "crashcart",
             "forklift", "dingy", "tote", "server", "phonebox", "golf", "missile", "missle",
             "battery", "garbage", "medical", "fish_", "blitzcan", "chair", "bucket", "vending",
             "generator", "keg", "tank_", "fuel"]) { return Crate; }
    if any(&["sandbag", "barricade", "cover", "shield_barrier", "street cone", "cone"]) { return Cover; }
    // "ff_building_blocks" is the BLOCK category, not a building - test the block names first.
    if !any(&["building_blocks", "building blocks"])
        && any(&["building", "tower", "bunker", "silo", "corridor", "room", "base_", "station", "plateau", "hut", "shelter"]) { return Building; }
    // BLOCK FIRST. Halo 4 and H2A name every building block `..._plat_NxM`, so testing for a
    // bare `plat_` ahead of this made `fw_unsc_c_plat_2x2` ("Block, 2X2") a Platform.
    if any(&["block", "bb_", "wall_s", "wall_m", "wall_l", "wall_xl"]) || entry.starts_with("block_") {
        // "flat" blocks are plates; a tall thin one reads as a wall
        return block_or_wall(dims);
    }
    if any(&["platform", "landing", "deck", "bp_plat", "plat_y", "plat_lg", "plat_tower"]) { return Platform; }
    if any(&["wall"]) { return Wall; }
    if any(&["decorative", "antenna", "monitor", "console", "column", "pillar", "grid", "banner", "sign"]) { return Decorative; }
    // The catch-all: a structural palette entry with no better name is a block; EVERYTHING ELSE is
    // scenery - a prop, which the matcher will only ever swap for another prop.
    if structural { return block_or_wall(dims); }
    if class == "bloc" || class == "scen" || class == "mach" || class == "ctrl" || class == "dspn" { return Scenery; }
    Unknown
}

/// Forge tag-name prefixes that name the SET, not the object: strip them before reading a tag
/// as a description. Longest first so `fw_stone_` wins over `fw_`.
const TAG_PREFIXES: [&str; 14] = [
    "fw_unsc_c_", "fw_unsc_d_", "fi_unsc_c_", "fw_stone_", "fw_bonanza_", "fw_ravine_",
    "fw_island_", "fw_warhouse_", "ca_dropoff_", "ca_creeper_", "storm_", "fw_", "fi_", "ff_",
];

/// The tag leaf with its set prefix removed (`fw_stone_wall_arena` -> `wall_arena`,
/// `fw_unsc_c_plat_2x2` -> `plat_2x2`). Halo 3's `sand_` / `fore_` are deliberately KEPT: there
/// they identify the Sandbox / Foundry Forge sets, which is real structural evidence.
pub fn strip_tag_prefix(tag: &str) -> String {
    let t = tag.to_ascii_lowercase();
    for p in TAG_PREFIXES {
        if let Some(rest) = t.strip_prefix(p) {
            if !rest.is_empty() { return rest.to_string(); }
        }
    }
    t
}

/// A boxy structural piece with no name hint: thin + tall = a wall, otherwise a block.
fn block_or_wall(dims: [f32; 3]) -> Kind {
    let mut d = dims;
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (thin, _mid, long) = (d[0], d[1], d[2]);
    if long > 1e-3 && thin / long < 0.12 && dims[2] > 2.0 { Kind::Wall } else { Kind::Block }
}

// ---------------------------------------------------------------------------------------------
// building the catalog
// ---------------------------------------------------------------------------------------------

/// A row under construction, merged across canvases by key.
#[derive(Default)]
struct Acc {
    obj: CatalogObject,
}

/// Build the catalog from every installed map of every title. `log` receives one line per map.
pub fn build(log: &mut dyn FnMut(String)) -> Catalog {
    let cands = crate::mapcat::enumerate();
    let mut rows: BTreeMap<(String, String), Acc> = BTreeMap::new();
    let mut maps = 0usize;

    // Reach: the native parser (one cache open per canvas).
    let reach: Vec<_> = cands.iter().filter(|c| c.game == Game::Reach && !c.modded).collect();
    if !reach.is_empty() {
        match reach_builder() {
            Some(dll) => {
                for c in &reach {
                    let stem = c.stem.clone();
                    let n = build_reach_map(&dll, &c.path, &stem, &mut rows);
                    if n > 0 { maps += 1; log(format!("reach {stem}: {n} palette rows")); }
                }
            }
            None => log("reach: native parser unavailable - no Reach rows".into()),
        }
    }

    // Halo 4 + H2A: the pure-Rust reader.
    for c in cands.iter().filter(|c| (c.game == Game::Halo4 || c.game == Game::H2A) && !c.modded && c.h4_type == Some(1)) {
        match crate::h4::cache::H4Cache::open(&c.path) {
            Ok(cache) => {
                let n = build_h4_map(&cache, c.game, &c.stem, &mut rows);
                if n > 0 { maps += 1; log(format!("{} {}: {n} palette rows", c.game.short_name(), c.stem)); }
            }
            Err(e) => log(format!("{}: {e}", c.path.display())),
        }
    }

    // Halo 3: the pure-Rust reader (every MP cache carries sandbox palettes).
    for c in cands.iter().filter(|c| c.game == Game::Halo3 && !c.modded) {
        match crate::h3::cache::H3Cache::open(&c.path) {
            Ok(cache) => {
                let n = build_h3_map(&cache, &c.stem, &mut rows);
                if n > 0 { maps += 1; log(format!("halo3 {}: {n} palette rows", c.stem)); }
            }
            Err(_) => {}
        }
    }

    let mut objects: Vec<CatalogObject> = rows.into_values().map(|mut a| {
        a.obj.canvases.sort();
        a.obj.canvases.dedup();
        a.obj.places.sort_by(|x, y| (x.canvas.as_str(), x.hidden).cmp(&(y.canvas.as_str(), y.hidden)));
        a.obj.places.dedup();
        a.obj
    }).collect();
    objects.sort_by(|a, b| (a.game.as_str(), a.kind.as_str(), a.key.as_str()).cmp(&(b.game.as_str(), b.kind.as_str(), b.key.as_str())));
    Catalog {
        note: format!("#xgame object catalog: {} objects from {maps} canvases, built by `hms-app --xgame-build-catalog`", objects.len()),
        objects,
    }
}

fn merge(rows: &mut BTreeMap<(String, String), Acc>, mut o: CatalogObject, spot: PaletteSpot) {
    let k = (o.game.clone(), o.key.clone());
    let e = rows.entry(k).or_insert_with(Acc::default);
    if e.obj.key.is_empty() {
        o.canvases.push(spot.canvas.clone());
        o.places.push(spot);
        e.obj = o;
    } else {
        // Prefer a localized display name and a real box if a later canvas has one.
        if (e.obj.display.is_empty() || e.obj.display == e.obj.entry) && !o.display.is_empty() { e.obj.display = o.display.clone(); }
        if !e.obj.has_box() && o.has_box() {
            e.obj.dims = o.dims; e.obj.center = o.center; e.obj.parts = o.parts;
            e.obj.box_source = o.box_source.clone(); e.obj.kind = o.kind.clone();
        }
        e.obj.canvases.push(spot.canvas.clone());
        e.obj.places.push(spot);
    }
}

// ---- Reach -----------------------------------------------------------------------------------

/// The native parser next to the executable, or None (then Reach contributes no rows).
fn reach_builder() -> Option<hms_native::NativeDll> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
    hms_native::NativeDll::load(&exe_dir.join(crate::PAYLOAD_DLL)).ok()
}

fn build_reach_map(dll: &hms_native::NativeDll, path: &std::path::Path, stem: &str, rows: &mut BTreeMap<(String, String), Acc>) -> usize {
    let dirs = crate::mapcat::shared_cache_dirs(path);
    dll.set_shared_cache_dirs(&dirs);
    let Ok(cache) = dll.open_cache(&path.to_string_lossy()) else { return 0 };
    let n = (|| {
        let Some(scnr) = dll.find_scenario(cache) else { return 0 };
        let palette = dll.scnr_forge_palette_full(cache, scnr);
        if palette.is_empty() { return 0; }
        // `folder` is a flat index into the distinct (palette_index, entry_within) order.
        let mut types: Vec<(u32, u32)> = Vec::new();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if !types.contains(&k) { types.push(k); }
        }
        let mut count = 0usize;
        for e in &palette {
            if e.tag_short == 0 || e.tag_short == 0xFFFF_FFFF { continue; }
            let tag = dll.tag_name(cache, e.tag_short).unwrap_or_default();
            if tag.is_empty() { continue; }
            let class = dll.tag_class(cache, e.tag_short).unwrap_or_default();
            let mode = dll.resolve_mode(cache, e.tag_short);
            let model = if mode != 0 && mode != 0xFFFF_FFFF { dll.tag_name(cache, mode).unwrap_or_default() } else { String::new() };
            let (dims, center, src, parts) = reach_box(dll, cache, e.tag_short, mode);
            let (object_type, shape, spawn_time, _boundary) = reach_mp_defaults(dll, cache, e.tag_short);
            let key = object_key(&tag, &e.variant_name, e.variant_name_sid);
            let kind = classify(&tag_leaf(&tag), &e.category_name, &e.name, &e.variant_name, "", &class, object_type, Game::Reach, dims);
            let quota = types.iter().position(|&k| k == (e.palette_index, e.entry_within)).unwrap_or(0) as u32;
            let o = CatalogObject {
                game: Game::Reach.short_name().into(), key, category: e.category_name.clone(), entry: e.name.clone(),
                variant: e.variant_name.clone(),
                display: if e.variant_name.is_empty() { e.name.clone() } else { e.variant_name.clone() },
                tag, class, model, dims, center, box_source: src.into(), parts,
                object_type, shape, spawn_time, price: 0, max_allowed: 0, kind: kind.name().into(),
                canvases: Vec::new(), places: Vec::new(),
            };
            merge(rows, o, PaletteSpot { canvas: stem.into(), category_index: e.palette_index, quota, variant: e.variant_within, hidden: false });
            count += 1;
        }
        count
    })();
    dll.close_cache(cache);
    n
}

/// Reach render-model box: the union of every section's compression bounds, else the collision,
/// else the physics hull (hidden blocks have no render geometry at all).
fn reach_box(dll: &hms_native::NativeDll, cache: hms_native::CacheHandle, obj_tag: u32, mode: u32) -> ([f32; 3], [f32; 3], &'static str, u32) {
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    let mut parts = 0u32;
    if mode != 0 && mode != 0xFFFF_FFFF {
        if let Ok(h) = dll.open_model(cache, mode) {
            let n = dll.model_section_count(h);
            for i in 0..n {
                if n > 1 && !dll.model_section_allowed(h, i) { continue; }
                let Some(s) = dll.model_section(h, i) else { continue };
                if s.vertex_count == 0 { continue; }
                let (a, b) = (s.bounds_min, s.bounds_max);
                if (0..3).any(|k| !a[k].is_finite() || !b[k].is_finite()) || (0..3).all(|k| b[k] <= a[k]) { continue; }
                for k in 0..3 { mn[k] = mn[k].min(a[k]); mx[k] = mx[k].max(b[k]); }
                parts += 1;
            }
            dll.close_model(h);
        }
    }
    if parts > 0 { return (size_of(mn, mx), center_of(mn, mx), "render", parts); }
    for (tag, src) in [(dll.resolve_coll(cache, obj_tag), "collision"), (dll.resolve_phmo(cache, obj_tag), "physics")] {
        if tag == 0 || tag == 0xFFFF_FFFF { continue; }
        let g = if src == "collision" { dll.coll_geometry(cache, tag) } else { dll.phmo_geometry(cache, tag) };
        let Some((v, _)) = g else { continue };
        if v.len() < 3 { continue; }
        let (mut a, mut b) = ([f32::MAX; 3], [f32::MIN; 3]);
        for c in v.chunks_exact(3) { for k in 0..3 { a[k] = a[k].min(c[k]); b[k] = b[k].max(c[k]); } }
        if (0..3).any(|k| b[k] <= a[k]) { continue; }
        return (size_of(a, b), center_of(a, b), src, 0);
    }
    ([0.0; 3], [0.0; 3], "none", 0)
}

/// REACH's `obje` multiplayer-object block, from the TOOL build's own reflection data
/// (`reverseme tagdefs reach_tag_test.exe obje`): the block is at obje **+0x140** (Halo 4 moved
/// it to +0x154) and its element is 188 bytes laid out
///
/// ```text
///   +0x00 byte flags  game engine flags        +0x14 char enum boundary shape
///   +0x01 char enum   type   <- NOT +2         +0x15 char enum spawn timer type
///   +0x02 byte flags  teleporter passability   +0x16 short     default spawn time [s]
///   +0x04 real        boundary width/radius    +0x18 short     default abandonment time
///   +0x08 real        boundary box length      +0x1A word      flags
///   +0x0C real        boundary +height
///   +0x10 real        boundary -height
/// ```
///
/// Reading it from the tag is what makes a converted Reach object carry the right
/// `cached_type` (1 weapon, 12/13/14 teleporter, 19 location ...) - the field the engine
/// switches on. Returns (type, boundary shape, default spawn time, boundary in world units).
pub const OFF_REACH_OBJE_MULTIPLAYER: u32 = 0x140;
fn reach_mp_defaults(dll: &hms_native::NativeDll, cache: hms_native::CacheHandle, obj_tag: u32) -> (u8, u8, u16, [f32; 4]) {
    let Some(b) = dll.tag_meta(cache, obj_tag, OFF_REACH_OBJE_MULTIPLAYER, 12) else { return (0, 0, 0, [0.0; 4]) };
    if b.len() < 12 { return (0, 0, 0, [0.0; 4]); }
    let count = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if count == 0 || count > 1 { return (0, 0, 0, [0.0; 4]); }
    let ptr = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let Some(e) = dll.tag_ptr_bytes(cache, ptr, 0x1C) else { return (0, 0, 0, [0.0; 4]) };
    if e.len() < 0x1C { return (0, 0, 0, [0.0; 4]); }
    let f = |o: usize| f32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]]);
    let spawn = i16::from_le_bytes([e[0x16], e[0x17]]).max(0) as u16;
    (e[1], e[0x14], spawn, [f(4), f(8), f(0xC), f(0x10)])
}

fn size_of(mn: [f32; 3], mx: [f32; 3]) -> [f32; 3] { [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]] }
fn center_of(mn: [f32; 3], mx: [f32; 3]) -> [f32; 3] { [(mx[0] + mn[0]) * 0.5, (mx[1] + mn[1]) * 0.5, (mx[2] + mn[2]) * 0.5] }

// ---- Halo 4 / H2A ----------------------------------------------------------------------------

fn build_h4_map(cache: &crate::h4::cache::H4Cache, game: Game, stem: &str, rows: &mut BTreeMap<(String, String), Acc>) -> usize {
    let pal = crate::h4::palette::palette(cache);
    if pal.entries.is_empty() { return 0; }
    let mut count = 0usize;
    for e in &pal.entries {
        let cat = pal.categories.get(e.category);
        let hidden = cat.map_or(false, |c| c.hidden);
        let cat_name = cat.map(|c| c.name.clone()).unwrap_or_default();
        for v in &e.variants {
            let Some(tag) = v.tag else { continue };
            let (dims, center, src, parts) = h4_box(cache, tag, v.mode, v.model_variant);
            let key = object_key(&v.tag_path, &v.name, v.variant_sid);
            let mp = v.mp;
            let object_type = v.object_type();
            let kind = classify(&tag_leaf(&v.tag_path), &cat_name, &e.name, &v.name, &v.display, &v.class_str(), object_type, game, dims);
            let o = CatalogObject {
                game: game.short_name().into(), key, category: cat_name.clone(), entry: e.name.clone(), variant: v.name.clone(),
                display: if v.display.is_empty() { e.display.clone() } else { v.display.clone() },
                tag: v.tag_path.clone(), class: v.class_str(), model: v.mode.map(|m| cache.tag_name(m).to_string()).unwrap_or_default(),
                dims, center, box_source: src.into(), parts, object_type,
                shape: mp.map(|m| m.shape).unwrap_or(0), spawn_time: mp.map(|m| m.spawn_time.max(0) as u16).unwrap_or(0),
                price: e.price, max_allowed: e.max_allowed, kind: kind.name().into(), canvases: Vec::new(), places: Vec::new(),
            };
            merge(rows, o, PaletteSpot { canvas: stem.into(), category_index: e.category as u32, quota: e.quota as u32, variant: v.index as u32, hidden });
            count += 1;
        }
    }
    count
}

fn h4_box(cache: &crate::h4::cache::H4Cache, obje: usize, mode: Option<usize>, variant: Option<usize>) -> ([f32; 3], [f32; 3], &'static str, u32) {
    let Some(mode) = mode else { return ([0.0; 3], [0.0; 3], "none", 0) };
    // EXACT first: the decoded vertices of this palette variant's meshes (CPU only, no GPU). The
    // `mode` compression bounds are a single MODEL-WIDE element, so they over-cover a variant.
    if let Ok(g) = crate::h4::scene::decode_object_geom(cache, obje, mode, variant) {
        let parts: u32 = g.meshes.iter().map(|m| m.parts.len().max(1) as u32).sum();
        if g.tri_count() > 0 && (0..3).any(|k| g.max[k] > g.min[k]) {
            return (size_of(g.min.into(), g.max.into()), center_of(g.min.into(), g.max.into()), "render", parts);
        }
    }
    let Ok(model) = crate::h4::objects::load_model(cache, mode) else { return ([0.0; 3], [0.0; 3], "none", 0) };
    let meshes = crate::h4::objects::variant_meshes(cache, obje, &model, variant);
    let list: Vec<usize> = if meshes.is_empty() { model.lod0_meshes.clone() } else { meshes };
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut parts = 0u32;
    for &mi in &list {
        let Some(m) = model.meshes.get(mi) else { continue };
        if !m.has_geometry() { continue; }
        let (a, b) = (m.bounds.pos_min, m.bounds.pos_max);
        if (0..3).any(|k| !a[k].is_finite() || !b[k].is_finite()) || (0..3).all(|k| b[k] <= a[k]) { continue; }
        for k in 0..3 { mn[k] = mn[k].min(a[k]); mx[k] = mx[k].max(b[k]); }
        parts += m.parts.len().max(1) as u32;
    }
    if parts == 0 {
        // No render geometry at all (a marker, or a hidden block): fall back to the physics hull.
        for (hull, src) in [(crate::h4::collision::object_collision_hull(cache, obje), "collision"),
                            (crate::h4::collision::object_physics_hull(cache, obje), "physics")] {
            if let Some((a, b)) = hull.and_then(|h| h.bounds()) {
                if (0..3).any(|k| b[k] > a[k]) { return (size_of(a.into(), b.into()), center_of(a.into(), b.into()), src, 0); }
            }
        }
        return ([0.0; 3], [0.0; 3], "none", 0);
    }
    (size_of(mn, mx), center_of(mn, mx), "bounds", parts)
}

// ---- Halo 3 ----------------------------------------------------------------------------------

fn build_h3_map(cache: &crate::h3::cache::H3Cache, stem: &str, rows: &mut BTreeMap<(String, String), Acc>) -> usize {
    let items = crate::h3::palette::palette(cache);
    if items.is_empty() { return 0; }
    let mut count = 0usize;
    for it in &items {
        let tag = cache.tag_name(it.tag).to_string();
        let class = cache.tag_class(it.tag).map(|c| String::from_utf8_lossy(&c).into_owned()).unwrap_or_default();
        let (dims, center, src, parts) = h3_box(cache, it.mode_tag);
        let key = object_key(&tag, "", 0);
        let object_type = it.object_type();
        let kind = classify(&tag_leaf(&tag), it.category, &it.name, "", &it.name, &class, object_type, Game::Halo3, dims);
        let o = CatalogObject {
            game: Game::Halo3.short_name().into(), key, category: it.category.into(), entry: it.name.clone(), variant: String::new(),
            display: it.name.clone(), tag, class, model: it.mode_tag.map(|m| cache.tag_name(m).to_string()).unwrap_or_default(),
            dims, center, box_source: src.into(), parts, object_type,
            shape: it.mp.map(|m| m.boundary_shape).unwrap_or(0), spawn_time: it.mp.map(|m| m.spawn_time.max(0) as u16).unwrap_or(0),
            price: it.price.round() as i32, max_allowed: it.max_allowed, kind: kind.name().into(), canvases: Vec::new(), places: Vec::new(),
        };
        merge(rows, o, PaletteSpot { canvas: stem.into(), category_index: it.block as u32, quota: it.entry as u32, variant: 0, hidden: false });
        count += 1;
    }
    count
}

fn h3_box(cache: &crate::h3::cache::H3Cache, mode: Option<usize>) -> ([f32; 3], [f32; 3], &'static str, u32) {
    let Some(mode) = mode else { return ([0.0; 3], [0.0; 3], "none", 0) };
    if let Ok(g) = crate::h3::scene::decode_model_geom(cache, mode) {
        let parts: u32 = g.meshes.iter().map(|m| m.parts.len().max(1) as u32).sum();
        if g.tri_count() > 0 && (0..3).any(|k| g.max[k] > g.min[k]) {
            return (size_of(g.min.into(), g.max.into()), center_of(g.min.into(), g.max.into()), "render", parts);
        }
    }
    let Ok(model) = crate::h3::objects::load_model(cache, mode) else { return ([0.0; 3], [0.0; 3], "none", 0) };
    let list: Vec<usize> = if model.lod0_meshes.is_empty() { (0..model.meshes.len()).collect() } else { model.lod0_meshes.clone() };
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut parts = 0u32;
    for &mi in &list {
        let Some(m) = model.meshes.get(mi) else { continue };
        if !m.has_geometry() { continue; }
        let b = model.mesh_bounds(mi).copied().unwrap_or(m.bounds);
        let (a, z) = (b.pos_min, b.pos_max);
        if (0..3).any(|k| !a[k].is_finite() || !z[k].is_finite()) || (0..3).all(|k| z[k] <= a[k]) { continue; }
        for k in 0..3 { mn[k] = mn[k].min(a[k]); mx[k] = mx[k].max(z[k]); }
        parts += m.parts.len().max(1) as u32;
    }
    if parts == 0 { return ([0.0; 3], [0.0; 3], "none", 0) }
    (size_of(mn, mx), center_of(mn, mx), "bounds", parts)
}

// ---------------------------------------------------------------------------------------------
// listing
// ---------------------------------------------------------------------------------------------

/// Human-readable dump (`--xgame-catalog [substring]`).
pub fn describe(c: &Catalog, filter: Option<&str>) -> String {
    let f = filter.map(|s| s.to_ascii_lowercase());
    let mut s = String::new();
    s.push_str(&format!("{}\n", c.note));
    for g in super::GAMES {
        let rows: Vec<&CatalogObject> = c.of_game(g)
            .filter(|o| f.as_ref().map_or(true, |n| {
                o.key.to_ascii_lowercase().contains(n.as_str()) || o.display.to_ascii_lowercase().contains(n.as_str())
                    || o.entry.to_ascii_lowercase().contains(n.as_str()) || o.kind.contains(n.as_str()) || o.tag.to_ascii_lowercase().contains(n.as_str())
            }))
            .collect();
        if rows.is_empty() { continue; }
        let boxed = rows.iter().filter(|o| o.has_box()).count();
        s.push_str(&format!("\n=== {} - {} objects ({boxed} with a box), canvases: {} ===\n",
            g.display_name(), rows.len(), c.canvases(g).len()));
        let mut by_kind: BTreeMap<&str, Vec<&&CatalogObject>> = BTreeMap::new();
        for o in &rows { by_kind.entry(o.kind.as_str()).or_default().push(o); }
        for (k, list) in by_kind {
            s.push_str(&format!("  [{k}] {} objects\n", list.len()));
            for o in list {
                s.push_str(&format!("    {:<40} {:>7.2} x {:>6.2} x {:>6.2} wu  {:<5} parts {:<3} type {:<3} '{}'  {} canvas(es)\n",
                    o.key, o.dims[0], o.dims[1], o.dims[2], o.box_source, o.parts, o.object_type, o.display, o.canvases.len()));
            }
        }
    }
    s
}

/// `tag_leaf` re-exported for the table's name matcher.
pub fn leaf(path: &str) -> String { tag_leaf(path) }

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked-in catalog parses, carries all four titles' worth of rows, and every row has a
    /// game / key / kind. (The DATA is only as good as the machine that generated it; this gate
    /// is about the file being readable and self-consistent.)
    #[test]
    fn builtin_catalog_decodes() {
        let c = Catalog::from_json(BUILTIN_CATALOG).expect("baked-in catalog must parse");
        if c.objects.is_empty() { return; } // a tree built before the first --xgame-build-catalog
        assert!(c.objects.iter().all(|o| !o.game.is_empty() && !o.key.is_empty() && !o.kind.is_empty()));
        // keys are unique per game
        let mut seen = std::collections::HashSet::new();
        for o in &c.objects { assert!(seen.insert((o.game.clone(), o.key.clone())), "duplicate {} {}", o.game, o.key); }
        // every place has a canvas and the canvas list covers the places
        for o in &c.objects {
            for p in &o.places { assert!(!p.canvas.is_empty()); assert!(o.canvases.contains(&p.canvas), "{}: place {} not in canvases", o.key, p.canvas); }
        }
    }

    /// The Coliseum Wall is the shape the whole idea rests on: Reach's and Halo 4's must both be
    /// in the catalog, classified as a wall, and the same size to within a tenth of a world unit.
    #[test]
    fn coliseum_wall_matches_across_reach_and_halo4() {
        let c = Catalog::load();
        if c.objects.is_empty() { return; }
        let r = c.of_game(Game::Reach).find(|o| o.key.starts_with("ff_arena_wall"));
        let h = c.of_game(Game::Halo4).find(|o| o.key.starts_with("fw_unsc_c_arena_wall"));
        let (Some(r), Some(h)) = (r, h) else { return }; // not every machine installs both titles
        assert_eq!(r.kind(), Kind::Wall, "reach coliseum wall kind");
        assert_eq!(h.kind(), Kind::Wall, "halo 4 coliseum wall kind");
        let (a, b) = (r.sorted_dims(), h.sorted_dims());
        for k in 0..3 { assert!((a[k] - b[k]).abs() < 0.2, "coliseum wall axis {k}: reach {:?} vs h4 {:?}", r.dims, h.dims); }
    }

    #[test]
    fn classify_knows_the_gameplay_families() {
        let d = [1.0, 1.0, 1.0];
        assert_eq!(classify("initial_spawn_point", "ff_spawning", "spawning_initial", "", "Initial Spawn Point", "scen", 0, Game::Reach, d), Kind::SpawnInitial);
        assert_eq!(classify("respawn_zone", "ff_spawning", "respawn_zone", "", "Respawn Zone", "scen", 0, Game::Reach, d), Kind::RespawnZone);
        assert_eq!(classify("weak_anti_respawn_zone", "ff_spawning", "respawn_zone_anti", "", "Anti Respawn Zone", "scen", 0, Game::Reach, d), Kind::RespawnZoneAnti);
        assert_eq!(classify("soft_kill_volume", "ff_spawning", "spawning_boundaries_kill", "soft", "Soft Kill Boundary", "scen", 0, Game::Reach, d), Kind::KillZoneSoft);
        assert_eq!(classify("mp_flag_base", "ff_objectives", "obj_flag_base", "", "Flag Stand", "scen", 0, Game::Reach, d), Kind::FlagStand);
        assert_eq!(classify("mp_hill_beacon", "ff_objectives", "obj_hill_marker", "", "Hill Marker", "scen", 0, Game::Reach, d), Kind::HillMarker);
        assert_eq!(classify("powerup_blue", "ff_gadgets", "gadget_powerups", "overshield", "Overshield", "eqip", 0, Game::Reach, d), Kind::Powerup);
        assert_eq!(classify("storm_dmr", "ff_weapons_human", "wep_dmr", "", "DMR", "weap", 1, Game::Halo4, d), Kind::Weapon);
        assert_eq!(classify("warthog", "ff_vehicles", "veh_warthog", "", "Warthog", "vehi", 8, Game::Halo4, d), Kind::Vehicle);
        assert_eq!(classify("fw_unsc_c_arena_wall", "ff_doors_win_wall", "drw_wall_coliseum", "", "Wall, Coliseum", "bloc", 0, Game::Halo4, [10.0, 0.5, 12.0]), Kind::Wall);
        assert_eq!(classify("fw_unsc_c_plat_2x2", "ff_structure", "ff_building_blocks", "bb_2x2", "Block, 2X2", "bloc", 0, Game::Halo4, [4.0, 4.0, 2.0]), Kind::Block);
        assert_eq!(classify("fi_boulder_a", "ff_natural", "fw_island_rock01", "", "Rock 1", "bloc", 0, Game::Halo4, [6.0, 6.0, 4.0]), Kind::NaturalRock);
        // the two the TAG name is the only evidence for
        assert_eq!(classify("cov_artillery_gun", "ff_thorage_scenery", "ff_thorage_covenant", "ff_dom_turretpad", "", "bloc", 0, Game::Halo4, [2.1, 0.8, 1.6]), Kind::Scenery);
        assert_eq!(classify("ff_reactor_door_blue", "ff_thorage_structure", "ff_thorage_doors", "blue", "", "bloc", 0, Game::Reach, [2.2, 0.3, 2.7]), Kind::Door);
    }

    #[test]
    fn object_key_is_stable_and_discriminating() {
        assert_eq!(super::super::object_key("objects\\levels\\forge\\ff_arena_wall\\ff_arena_wall", "wall_coliseum", 0), "ff_arena_wall#wall_coliseum");
        assert_eq!(super::super::object_key("objects\\vehicles\\warthog\\warthog", "", 0), "warthog");
        assert_ne!(super::super::object_key("a\\warthog", "", 1), super::super::object_key("a\\warthog", "", 2));
    }
}
