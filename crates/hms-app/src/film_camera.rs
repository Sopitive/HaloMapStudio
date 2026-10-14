//! #film-camera  The ENGINE'S CAMERA MODES, decoded, and HMS's Theater camera driven into them.
//!
//! The question this module answers is the one the user asked: *when a player is assassinated, or
//! gets into a vehicle, or dies and waits to respawn, the game does not keep the plain
//! first-person view — it moves the camera into a context camera.* So Theater must too, and it
//! must do it from the engine's own rules rather than from taste.
//!
//! # What the engine actually has (RE'd, `haloreach_debug.dll`)
//!
//! There is **one camera-mode enum with thirteen members**, and it is named in the binary. The
//! name table is a plain array of thirteen `const char*` at **`.data` RVA `0x180FEC0`**, the
//! strings live in `camera.cpp`'s literal pool, and the count is confirmed independently by
//! `director_set_camera_mode`'s own range check — `sub_180784D50` rejects `camera_mode > 0xC`
//! under the assert `"VALID_INDEX(camera_mode, k_number_of_camera_modes)"` (`director.cpp`), so
//! `k_number_of_camera_modes == 13`. See [`ENGINE_CAMERA_MODE_NAMES`].
//!
//! The setter is **`director_set_camera_mode`, `haloreach_debug.dll+0x786DA0`**:
//! `(director*, camera_mode, float transition_seconds, bool force)`. It switches on the mode and
//! constructs the matching camera object in place at `director+8` (thirteen distinct
//! constructors, one per mode — which is how the enum ordering above is cross-checked: case 3
//! builds the object whose assert string is `first_person_camera.cpp:188`, case 4 the one whose
//! vtable is the `dead` camera's). The **transition time** it is handed is stored at
//! `director+0x17C`, and that is the only blending the engine does between two modes.
//!
//! # The selection, in the engine's own order
//!
//! `unit_should_be_third_person` (**`+0x781D40`**, `(object_index, int* out_reason) -> bool`) is
//! the first/third-person decision, and it reports *why*:
//!
//! | reason | the gate | what it means |
//! |---|---|---|
//! | 0 | `+0x4E68D0(unit)` | nothing special; the unit's own default |
//! | 1 | `+0x534870(unit)` | the unit's animation mode is in table A |
//! | 2 | — | the seat offers no camera, or the parent is not a unit |
//! | 3 | `+0x534920(unit)` | the unit's animation mode is in table B |
//! | 4 | `+0x512940(weapon)` | **the WEAPON forces third person** |
//!
//! * **The vehicle case.** When the unit has a parent (`unit+0x14` != NONE) that is a biped or a
//!   vehicle, the engine reads the unit's seat index from `unit+0x336` and indexes the PARENT's
//!   tag: seat block count at **`+0x444`**, address at **`+0x448`**, element stride **`0x12C`**
//!   (the code computes `seat_index * 0x4B` in dwords). The seat's first dword is its flags, and
//!   **bit 4** is the third-person bit (`flags >> 4 & 1` is returned as the answer) while
//!   **bit 6** clear means the seat offers no camera at all (reason 2).
//! * **The weapon case.** `+0x512940` fetches the unit's current weapon
//!   (`unit_get_weapon(unit, unit+0x34A)`) and returns **bit 15 of the dword at its object
//!   definition `+0x204`**. `object_struct_definition` is `0x168` and `item_struct_definition`
//!   `0x9C`, so `0x168 + 0x9C = 0x204` is exactly the first field of
//!   `weapon_struct_definition` — i.e. the weapon's own flags word. This is the user's "some
//!   weapons have cameras as well they change to", and it is a per-weapon authored bit, not a
//!   special case for turrets: a detached turret is simply a weapon that sets it.
//! * **The animation cases.** `+0x534870` and `+0x534920` both read the unit's current animation
//!   mode (`unit + *(i16*)(unit+0x18E) + 0x204`) and compare it against a five-entry table:
//!   table A at `.data 0x17C97E0` = `{0x57, 0x27A, 0x27B, 0x27C, 0x27D}`, table B at
//!   `0x17C97F8` = `{0x58, 0x27E, 0x27F, 0x280, 0x281}`. **A synchronised action is one of
//!   these** — it is the only thing that puts a biped into a whole-body animation mode — which is
//!   why being assassinated takes your view out of your own eyes without any camera-mode change
//!   being broadcast.
//!
//! `game_director::update` (**`+0xA7F720`**) is the per-frame driver, and the death transition is
//! explicit in it:
//!
//! ```text
//! if mode not in {flying, scripted, authored} and watched_player != NONE:
//!     if player->unit (+0x28) == NONE and player->corpse (+0x2C) != NONE:
//!         if mode in {first person, following}:  set_camera_mode(dead, 0.0); target = corpse
//!     else if observer.target != player->unit:   set_camera_mode(preferred, 0.0); target = unit
//! if mode in {dead, respawning}:
//!     if director_choose_dead_camera_mode(+0x9218C0) == orbiting:
//!         set_camera_mode(orbiting, 0.0); target = the body; orbit distance = globals camera +0x98
//! ```
//!
//! So the death chain is **`first person` or `following` -> `dead` -> `orbiting` around the
//! corpse -> `respawning` -> back to the preferred mode**, and every step is handed a transition
//! time of **0.0**: the engine CUTS to the dead camera, it does not ease into it.
//! `+0x9218C0` gates `dead -> orbiting` on the ticks elapsed since the unit's own death tick
//! (`unit+0x320`) against three authored seconds in the globals camera block.
//!
//! # The numbers, measured out of a real map
//!
//! Every one of those camera-block fields is loaded by **one** function,
//! `+0xA70960`, and it loads them into a flat run of globals with a perfectly linear mapping:
//! `global(0x1818583A0 + k) <- camera_block[0] + 0x24 + k`, over 48 consecutive stores. The
//! mapping is anchored by a field HMS already relied on — `+0x24` is `real field of view
//! [degrees]`, and `global 0x1818583A0` is exactly where the loader puts it. So every float the
//! camera code reads can be named by its camera-block offset, and then read out of a shipped map.
//! The values below come from `forge_halo.map`'s own `globals\globals` (camera block element 0 at
//! file offset `0xD42C60C`; the anchor reads 78.0 there, as HMS already documented).

use crate::film_play::{CameraPose, FollowRig, Playback};

// -------------------------------------------------------------------------------------------
// the enum
// -------------------------------------------------------------------------------------------

/// The engine's thirteen camera modes, **in the engine's own order**.
///
/// Source: the `const char*` array at `haloreach_debug.dll` `.data` RVA `0x180FEC0`, thirteen
/// slots, strings from `camera.cpp`. The count is corroborated by `director_set_camera_mode`'s
/// range check (`sub_180784D50` rejects `> 0xC`) and the ordering by the thirteen-case switch in
/// `director_set_camera_mode` itself, whose case 3 constructs the object that asserts in
/// `first_person_camera.cpp`.
pub const ENGINE_CAMERA_MODE_NAMES: [&str; 13] = [
    "following",      // 0
    "orbiting",       // 1
    "flying",         // 2
    "first person",   // 3
    "dead",           // 4
    "static",         // 5
    "scripted",       // 6
    "authored",       // 7
    "respawning",     // 8
    "animated trick", // 9
    "tracked target", // 10
    "sandbox",        // 11
    "space_fighter",  // 12
];

/// `k_number_of_camera_modes`, from `director_set_camera_mode`'s own `camera_mode > 0xC` reject.
pub const NUMBER_OF_CAMERA_MODES: usize = 13;

/// One of the engine's camera modes. Only the ones a Theater playback can ever be in are
/// reachable from [`resolve`]; the rest exist so the enum is the engine's and not a subset of it,
/// and so `film camera mode` can report an honest "the engine has this, a film never shows it".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EngineCameraMode {
    Following = 0,
    Orbiting = 1,
    Flying = 2,
    FirstPerson = 3,
    Dead = 4,
    Static = 5,
    Scripted = 6,
    Authored = 7,
    Respawning = 8,
    AnimatedTrick = 9,
    TrackedTarget = 10,
    Sandbox = 11,
    SpaceFighter = 12,
}

impl EngineCameraMode {
    pub fn index(self) -> u8 {
        self as u8
    }
    pub fn name(self) -> &'static str {
        ENGINE_CAMERA_MODE_NAMES[self as usize]
    }
    pub fn from_index(i: u8) -> Option<EngineCameraMode> {
        use EngineCameraMode::*;
        Some(match i {
            0 => Following,
            1 => Orbiting,
            2 => Flying,
            3 => FirstPerson,
            4 => Dead,
            5 => Static,
            6 => Scripted,
            7 => Authored,
            8 => Respawning,
            9 => AnimatedTrick,
            10 => TrackedTarget,
            11 => Sandbox,
            12 => SpaceFighter,
            _ => return None,
        })
    }
    /// True for the modes that do NOT look out of the subject's eyes, i.e. the ones that need a
    /// pivot and a stand-off cast.
    pub fn is_third_person(self) -> bool {
        !matches!(self, EngineCameraMode::FirstPerson)
    }
}

/// `unit_should_be_third_person`'s own reason code (`+0x781D40` writes it through its second
/// argument). Kept because it is the difference between "the game put you in third person
/// because of your SEAT" and "...because of your WEAPON", and the viewer should be told which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThirdPersonReason {
    /// Reason 0 — nothing forced it; the player's own preference stands.
    Default,
    /// Reason 1 — the unit's animation mode is in table A (`.data 0x17C97E0`).
    AnimationA,
    /// Reason 2 — the seat offers no camera, or the parent is not a unit.
    NoSeatCamera,
    /// Reason 3 — the unit's animation mode is in table B (`.data 0x17C97F8`).
    AnimationB,
    /// Reason 4 — the unit's current weapon sets bit 15 of `weap +0x204`.
    Weapon,
    /// The vehicle SEAT's own third-person bit (`seat flags` bit 4). The engine folds this into
    /// the return value rather than into the reason code, so this variant is HMS's own name for
    /// it; it is reported separately because it is the case the user asked about by name.
    Seat,
}

impl ThirdPersonReason {
    pub fn label(self) -> &'static str {
        match self {
            ThirdPersonReason::Default => "the player's own preference",
            ThirdPersonReason::AnimationA => "a whole-body animation mode (table A)",
            ThirdPersonReason::NoSeatCamera => "the seat offers no camera",
            ThirdPersonReason::AnimationB => "a whole-body animation mode (table B)",
            ThirdPersonReason::Weapon => "the weapon forces third person (weap +0x204 bit 15)",
            ThirdPersonReason::Seat => "the vehicle seat forces third person (seat flags bit 4)",
        }
    }
}

// -------------------------------------------------------------------------------------------
// the engine's own numbers
// -------------------------------------------------------------------------------------------

/// The engine's own name for each camera-block field this module uses, so a reader can audit the
/// constants below against the tag rather than against this comment.
///
/// The block is `camera_block`, **216 (0xD8) bytes, max 1**, at `matg +0x3FC`; the field names
/// come from `reach_tag_test.exe`'s own reflection data and the values from `forge_halo.map`'s
/// `globals\globals` (element 0 at file offset `0xD42C60C`, where `+0x24` reads 78.0 — the anchor
/// HMS already relied on).
///
/// | offset | field | forge_halo |
/// |---|---|---|
/// | `+0x024` | `field of view [degrees]` | 78.0 |
/// | `+0x03C` | **dead camera transition time (s)** | 1.5 |
/// | `+0x040` | falling death transition time (s) | 0.7 |
/// | `+0x044` | **dead cam initial distance [wu]** | 3.2 |
/// | `+0x048` | **dead cam final distance [wu]** | 1.6 |
/// | `+0x04C` | **dead cam z offset [wu]** | 0.45 |
/// | `+0x050` | dead cam maximum elevation (rad) | 0.26 |
/// | `+0x054` | dead cam movement delay (s) | 0.3 |
/// | `+0x058` | **time_to_auto_switch_to_orbiting (s)** | 2.0 |
/// | `+0x064` | dead camera minimum falling velocity | 0.05 |
/// | `+0x094` | **orbit cam minimum distance [wu]** | 1.0 |
/// | `+0x098` | **orbit cam maximum distance [wu]** | 2.5 |
/// | `+0x09C` | orbit cam movement delay (s) | 0.3 |
/// | `+0x0A0` | **orbit cam z offset [wu]** | 0.45 |
/// | `+0x0A4` / `+0x0A8` | orbit cam minimum / maximum elevation (rad) | -0.1 / 0.78 |
/// | `+0x0AC` | max playback speed (saved films) | 5.0 |
/// | `+0x0B8` | **enter vehicle transition time (s)** | 0.55 |
/// | `+0x0BC` | exit vehicle transition time (s) | 0.5 |
///
/// Two of these close the loop on the disassembly independently, which is why the whole table can
/// be trusted: `+0x058` is the field `director_choose_dead_camera_mode` compares the ticks since
/// death against, and it is literally named `time_to_auto_switch_to_orbiting`; and `+0x098` is
/// the one `game_director::update` hands to the orbiting camera's distance setter, and it is
/// named `orbit cam maximum distance`.
pub const CAMERA_BLOCK_TAG_FIELD: &str = "matg +0x3FC -> camera_block (0xD8 bytes)";

/// Where the **dead** camera starts and settles, world units — `camera block +0x044` and
/// `+0x048`, read **3.2** and **1.6** in `forge_halo.map`. The engine eases between them over
/// [`DEAD_CAMERA_TRANSITION_SECONDS`], which is what makes a death read as the view falling away
/// from you and then closing back in on your body.
pub const DEAD_CAM_INITIAL_DIST_WU: f32 = 3.2;
pub const DEAD_CAM_FINAL_DIST_WU: f32 = 1.6;
/// How high above the body's root the dead camera's pivot sits, world units — `camera block
/// +0x04C`, **0.45**. Note it is almost exactly HMS's own invented follow pivot of 0.40, so the
/// framing barely moves; it is just no longer a guess.
pub const DEAD_CAM_Z_OFFSET_WU: f32 = 0.45;
/// The dead camera's elevation ceiling, radians — `camera block +0x050`, **0.26** (14.9 degrees).
pub const DEAD_CAM_MAX_ELEVATION_RAD: f32 = 0.26;
/// How long the dead camera takes to go from its initial to its final distance, seconds —
/// `camera block +0x03C`, **1.5**. (`+0x040`, the *falling* death transition, is 0.7; HMS does
/// not distinguish the two because a film does not record whether a death was a fall.)
pub const DEAD_CAMERA_TRANSITION_SECONDS: f32 = 1.5;

/// How far the **orbiting** camera may sit from the body it is watching, world units — `camera
/// block +0x094` and `+0x098`, read **1.0** and **2.5**. `game_director::update` hands the
/// maximum straight into the orbiting camera's distance setter (`+0xA81A80`) the moment `dead`
/// becomes `orbiting`, so 2.5 is where a spectated body is framed from.
///
/// HMS's own follow rig already sat at 2.4 wu, 4% short of the engine's number — close enough
/// that nobody would have noticed, and no longer a guess either way.
pub const ORBIT_CAM_MIN_DIST_WU: f32 = 1.0;
pub const ORBIT_CAM_MAX_DIST_WU: f32 = 2.5;
/// The orbiting camera's pivot height above the body's root node, world units — `camera block
/// +0x0A0`, **0.45**.
pub const ORBIT_CAM_Z_OFFSET_WU: f32 = 0.45;
/// The orbiting camera's elevation limits, radians — `camera block +0x0A4` / `+0x0A8`, read
/// **-0.1** and **0.78** (-5.7 and 44.7 degrees). HMS's orbit handle is clamped to these while a
/// context camera is in force, so a spectate shot cannot be dragged somewhere the game could not
/// put it.
pub const ORBIT_CAM_MIN_ELEVATION_RAD: f32 = -0.1;
pub const ORBIT_CAM_MAX_ELEVATION_RAD: f32 = 0.78;


/// How long the camera stays in `dead` before it becomes `orbiting`, SECONDS.
///
/// `director_choose_dead_camera_mode` (`+0x9218C0`) measures `now - unit->death_tick`
/// (`unit+0x320`) and compares it against the globals camera block's **`+0x58`**, which the tag
/// names outright: **`time_to_auto_switch_to_orbiting`**. Read **2.0** in `forge_halo.map`. (The
/// function reads `+0x5C` and `+0x60` — `ignore_stick_time` and `ignore_button_time`, both also
/// 2.0 — as well, but those gate a LIVE player's input and a film has no input to ignore.)
pub const DEAD_TO_ORBIT_SECONDS: f32 = 2.0;

/// The transition time the engine uses when it swaps a live player between first and third
/// person, SECONDS. Globals camera block **`+0xB8`**, read **0.55** in `forge_halo.map`.
///
/// The tag's own name for it is **`enter vehicle transition time`**, and that is not a
/// coincidence or a misread: entering a vehicle IS the first-to-third-person switch, so the one
/// non-zero transition `game_director::update` ever passes to `director_set_camera_mode` is the
/// one the tag authored for boarding. `+0xBC`, `exit vehicle transition time`, is 0.5 — see
/// [`EXIT_VEHICLE_TRANSITION_SECONDS`].
pub const ENTER_VEHICLE_TRANSITION_SECONDS: f32 = 0.55;
/// Camera block **`+0xBC`**, `exit vehicle transition time`, **0.5** in `forge_halo.map`.
pub const EXIT_VEHICLE_TRANSITION_SECONDS: f32 = 0.5;


/// **The death transition is a CUT.** Every `set_camera_mode` on the death chain in
/// `game_director::update` passes a transition time of literal `0.0` — into `dead`, into
/// `orbiting` and back out to the preferred mode. The easing a death DOES have is the dead
/// camera's own distance ramp ([`DEAD_CAMERA_TRANSITION_SECONDS`]), not a mode blend. Named
/// rather than left as a bare zero so that nobody "improves" it into an ease.
pub const DEATH_SWITCH_SECONDS: f32 = 0.0;

/// `weapon_struct_definition` flags **bit 15 = `uses 3rd person camera`**, and `bit 8 = `support
/// weapon``. Three independent reads agree: `+0x512940` returns
/// `*(u32*)(weapon_tag + 0x204) >> 0xF & 1`; the segment arithmetic puts
/// `weapon_struct_definition +0x00` at exactly `0x168 + 0x9C = 0x204`; and the reflection data
/// names bit 15 `uses 3rd person camera`.
///
/// This is the user's "some weapons have cameras as well they change to", and the **detached
/// turret** case is the same flag rather than a special case — a portable turret is a weapon
/// that sets it, and `weap +0x498 support third person camera range` (degrees, "0 straight, -90
/// down, 90 up") is the authored pitch range it is framed within.
pub const WEAPON_FLAGS_TAG_FIELD: &str = "weap +0x204 flags, bit 15 'uses 3rd person camera'";

/// The first-person camera's pitch limit, radians — **85.5 degrees**.
/// `first_person_camera::update` (`+0x96F0D0`) clamps the accumulated pitch to
/// `+-1.49225652`.
pub const FP_PITCH_LIMIT_RAD: f32 = 1.49225652;

/// Ticks per second, so the authored seconds above can be compared against a film tick.
pub const TICK_HZ: f32 = 60.0;

// -------------------------------------------------------------------------------------------
// what the film says, as film-constant tables
// -------------------------------------------------------------------------------------------

/// The tag-authored bits the resolver needs, read ONCE per film from the loaded cache.
///
/// It is kept separate from [`resolve`] on purpose: the resolver has to be a pure function of
/// `(film, body, tick)` so that scrubbing backwards gives the identical camera, and tag data is
/// film-constant rather than tick-varying. An empty table is the honest no-map case and
/// [`CameraTags::seat_is_third_person`] says what it falls back to.
#[derive(Clone, Debug, Default)]
pub struct CameraTags {
    /// `vehicle net id -> that vehicle's seat flags`, one dword per seat, in seat order.
    /// `vehi +0x444` count / `+0x448` address, stride `0x12C`, first dword of each element.
    pub seat_flags: std::collections::HashMap<u32, Vec<u32>>,
    /// Weapon tag indices whose `weap +0x204` has bit 15 set, i.e. which force third person.
    pub third_person_weapons: std::collections::HashSet<u32>,
}

/// `unit_seat_block` flags, `vehi +0x444[seat] +0x00`, **by the engine's own names**.
///
/// The three bits this module uses are each corroborated twice — once by what the code does with
/// them and once by the name `reach_tag_test.exe`'s reflection data gives them, and the two agree
/// in every case:
///
/// | bit | name | what the code does with it |
/// |---|---|---|
/// | 4 | **`third person camera`** | `unit_should_be_third_person` RETURNS `flags >> 4 & 1` |
/// | 6 | **`third person on enter`** | clear -> reason 2 |
/// | 7 | **`first person camera slaved to gun.`** | set -> `first_person_camera_build` takes the camera's position, forward and up straight off a MARKER on the parent (`object_get_marker(parent, 0x101, ...)`) |
///
/// (For the record, the rest of the run in `.rdata` is `gunner`(3), `allows weapons`(5),
/// `not valid without driver`(8), `allow AI noncombatants`(9), `boarding seat`(10) ...)
pub const SEAT_FLAG_THIRD_PERSON: u32 = 1 << 4;
/// `seat flags` bit 6, `third person on enter`. Clear is `unit_should_be_third_person`'s reason 2.
pub const SEAT_FLAG_THIRD_PERSON_ON_ENTER: u32 = 1 << 6;
/// `seat flags` bit 7, `first person camera slaved to gun.` — the seat's camera comes straight
/// off a marker on the vehicle.
///
/// **HMS decodes and reports this bit but cannot apply it.** The marker is animated on the
/// vehicle's own skeleton and a film does not sample vehicle node poses — only the object's root
/// basis — so there is nothing to put the camera on. Saying so is the honest answer; substituting
/// the root would be a different camera wearing this one's name.
pub const SEAT_FLAG_CAMERA_SLAVED_TO_GUN: u32 = 1 << 7;
/// `weap +0x204` bit 15, `uses 3rd person camera`, from `+0x512940`.
pub const WEAPON_FLAG_THIRD_PERSON: u32 = 1 << 15;
/// `weap +0x204` bit 8, `support weapon` — the class a detached/portable turret belongs to.
pub const WEAPON_FLAG_SUPPORT: u32 = 1 << 8;

impl CameraTags {
    /// Does riding seat `seat` of this vehicle put the camera in third person?
    ///
    /// With the seat flags decoded this is the engine's own bit. Without them — no map open, or a
    /// vehicle whose model HMS could not identify — it answers **true**, and that is a stated
    /// fallback rather than a decode: every land vehicle a Reach player drives is third person,
    /// so "a rider is in third person" is right far more often than it is wrong, and
    /// [`CameraContext::decoded`] reports which of the two happened so nothing downstream can
    /// claim the bit was read when it was not.
    pub fn seat_is_third_person(&self, vehicle: u32, seat: Option<u8>) -> (bool, bool) {
        let Some(flags) = self.seat_flags.get(&vehicle) else { return (true, false) };
        let Some(s) = seat else { return (true, false) };
        let Some(&f) = flags.get(s as usize) else { return (true, false) };
        (f & SEAT_FLAG_THIRD_PERSON != 0, true)
    }
    /// Does this seat offer a camera at all (`seat flags` bit 6)?
    pub fn seat_has_camera(&self, vehicle: u32, seat: Option<u8>) -> bool {
        match (self.seat_flags.get(&vehicle), seat) {
            (Some(flags), Some(s)) => flags.get(s as usize).map_or(true, |f| f & SEAT_FLAG_THIRD_PERSON_ON_ENTER != 0),
            _ => true,
        }
    }
    /// Is this seat's camera slaved to a marker on the vehicle (`seat flags` bit 7)? Reported so
    /// the viewer is told that the shot they are looking at is an orbit standing in for an
    /// authored marker camera HMS cannot follow — see [`SEAT_FLAG_CAMERA_SLAVED_TO_GUN`].
    pub fn seat_camera_is_slaved(&self, vehicle: u32, seat: Option<u8>) -> bool {
        match (self.seat_flags.get(&vehicle), seat) {
            (Some(flags), Some(s)) => {
                flags.get(s as usize).is_some_and(|f| f & SEAT_FLAG_CAMERA_SLAVED_TO_GUN != 0)
            }
            _ => false,
        }
    }
}

/// Read a `bipd`/`vehi`'s SEAT FLAGS straight out of the loaded cache: one dword per seat, in
/// seat order.
///
/// `unit_struct_definition`'s `seats` block is at absolute **`+0x444`** (count) / **`+0x448`**
/// (address) in both `bipd` and `vehi` — both chain `object_struct_definition` (0x168) then
/// `unit_struct_definition`, so the offset is the same in each — with element stride **`0x12C`**
/// and a cap of 64 (`MAXIMUM_SEATS_PER_UNIT_DEFINITION`). Each element's first dword is its
/// flags. Every one of those four numbers is corroborated twice: `unit_should_be_third_person`
/// computes `seat_index * 0x4B` in dwords off `parent_tag + 0x448` and bounds it against
/// `+0x444`, and the reflection data gives `unit +0x2DC` (= absolute 0x444) `seats`, element 300.
pub fn decode_seat_flags(
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    obj_tag: u32,
) -> Option<Vec<u32>> {
    let u = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
    let h = native.tag_meta(cache, obj_tag, 0, 0x44C)?;
    let (cnt, ptr) = (u(&h, 0x444), u(&h, 0x448));
    if cnt == 0 || cnt > 64 {
        return None;
    }
    let b = native.tag_ptr_bytes(cache, ptr, cnt * 0x12C)?;
    Some((0..cnt as usize).map(|i| u(&b, i * 0x12C)).collect())
}

/// Does this `weap` set `uses 3rd person camera`? `weap +0x204` bit 15 — see
/// [`WEAPON_FLAGS_TAG_FIELD`].
pub fn decode_weapon_third_person(
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    weap_tag: u32,
) -> Option<bool> {
    // Read from offset 0 so the buffer's indices ARE the tag's offsets, which is how every other
    // tag read in this tree is written; a read at `offset = 0x204` returns a buffer whose index 0
    // is that field, and mixing the two conventions is how an off-by-0x204 gets in.
    let b = native.tag_meta(cache, weap_tag, 0, 0x208)?;
    let f = u32::from_le_bytes([b[0x204], b[0x205], b[0x206], b[0x207]]);
    Some(f & WEAPON_FLAG_THIRD_PERSON != 0)
}

/// How far the **dead** camera is from the body, `secs` seconds after the death.
///
/// The engine eases from `dead cam initial distance` to `dead cam final distance` over
/// `dead camera transition time` — 3.2 wu to 1.6 wu over 1.5 s in `forge_halo.map`. A pure
/// function of the elapsed time, so it rewinds exactly.
pub fn dead_cam_distance(secs: f32) -> f32 {
    let t = (secs / DEAD_CAMERA_TRANSITION_SECONDS).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    DEAD_CAM_INITIAL_DIST_WU + (DEAD_CAM_FINAL_DIST_WU - DEAD_CAM_INITIAL_DIST_WU) * s
}

// -------------------------------------------------------------------------------------------
// the context
// -------------------------------------------------------------------------------------------

/// What put the camera where it is, beyond the mode itself.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextDetail {
    /// Nothing is overriding the player's preference.
    Plain,
    /// The body is in a vehicle seat.
    VehicleSeat {
        vehicle: u32,
        /// `seat flags` bit 7, `first person camera slaved to gun.` — the engine would take this
        /// seat's camera straight off a MARKER on the vehicle, which a film cannot drive (see
        /// [`SEAT_FLAG_CAMERA_SLAVED_TO_GUN`]). Carried so the read-out can say that the orbit
        /// the viewer is looking at is standing in for an authored camera, rather than leaving
        /// them to assume it is the shot the game framed.
        slaved: bool,
        /// The seat index, when the recording STATES it. `biped_exit_vehicle` carries a 6-bit
        /// seat index and nothing else in a film does, so this is `Some` only for a ride that
        /// ended in a stated exit — see [`seat_index_of_ride`].
        seat: Option<u8>,
    },
    /// The body is holding a weapon that sets `uses 3rd person camera` — a detached turret, or
    /// anything else authored with the flag.
    WeaponThirdPerson {
        /// The `weap` tag index, so the viewer can be told WHICH weapon did it.
        weapon: u32,
    },
    /// The body is in a synchronised action — an assassination.
    SyncAction {
        /// True for the one performing it, false for the one it is performed on.
        attacker: bool,
        partner: Option<u32>,
        /// The tick the action started, so the age is `tick - started`.
        started: u32,
    },
    /// The body has just died and the camera is cutting to its corpse.
    Dead { died_at: u32 },
    /// The camera is orbiting the corpse while the player waits to respawn.
    Orbiting { died_at: u32 },
    /// The player is being put back into the world.
    Respawning { died_at: u32 },
}

/// Which camera the engine would be in, for one body at one tick.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraContext {
    pub mode: EngineCameraMode,
    pub reason: ThirdPersonReason,
    /// The body the camera is ON. For the dead chain this is still the dead body's own net id —
    /// a film has no separate corpse entity, the track simply carries on past the death — but it
    /// is named explicitly because the engine retargets the observer at `player+0x2C` and a
    /// reader should not have to infer that the two coincide here.
    pub subject: u32,
    pub detail: ContextDetail,
    /// How far into the transition INTO this mode, 0..1. A pure function of the tick: it is
    /// `(tick - entered) / transition_ticks`, so scrubbing backwards rewinds it exactly.
    pub blend: f32,
    /// The tick this mode was entered.
    pub entered: u32,
    /// False when the MODE or its window is a stated fallback rather than something read out of
    /// the recording or the tags. Nothing may describe this context as decoded while it is false.
    pub decoded: bool,
    /// Whether the POSE is the engine's own authored geometry, which is a different question from
    /// [`CameraContext::decoded`] and has to be asked separately.
    ///
    /// `true` for the death chain, where every distance, pivot height, elevation limit and
    /// duration comes from the globals camera block. `false` for the assassination and the
    /// vehicle seat, where the engine's shot is an authored `trak` camera track or a camera
    /// marker on the vehicle, neither of which a film can drive — so HMS orbits instead and says
    /// so. Keeping the two flags apart is what stops "the mode transition is decoded" being read
    /// as "this is the shot the game framed".
    pub pose_is_authored: bool,
}

impl CameraContext {
    /// A one-line account, honest about what is and is not decoded.
    pub fn label(&self) -> String {
        let what = match &self.detail {
            ContextDetail::Plain => String::new(),
            ContextDetail::VehicleSeat { vehicle, seat, slaved } => format!(
                ", riding 0x{vehicle:08X} {}{}",
                match seat {
                    Some(s) => format!("seat {s}"),
                    None => "(seat not stated: only a stated EXIT ever names one)".to_string(),
                },
                if *slaved {
                    " — that seat's camera is `first person camera slaved to gun.` (flags bit 7),                      which rides a marker on the vehicle's own skeleton; a film samples only the                      vehicle's root, so this orbit stands in for it"
                } else {
                    ""
                }
            ),
            ContextDetail::SyncAction { attacker, partner, started } => format!(
                ", assassination as the {} of 0x{:08X}, {} ticks in \
                 (the engine's own shot is the unit's authored `sync action camera` track, \
                 which a film cannot drive -- this is an orbit of the shared root standing in \
                 for it)",
                if *attacker { "attacker" } else { "victim" },
                partner.unwrap_or(0),
                self.entered.max(*started) - *started
            ),
            ContextDetail::WeaponThirdPerson { weapon } => format!(
                ", holding weapon tag 0x{weapon:08X}, which sets `uses 3rd person camera` \
                 (weap +0x204 bit 15)"
            ),
            ContextDetail::Dead { died_at } => format!(", died at tick {died_at}"),
            ContextDetail::Orbiting { died_at } => {
                format!(", orbiting the body at {ORBIT_CAM_MAX_DIST_WU} wu since tick {died_at}")
            }
            ContextDetail::Respawning { died_at } => format!(", respawning after tick {died_at}"),
        };
        format!(
            "engine camera mode {} ({}){}{}{}{}",
            self.mode.index(),
            self.mode.name(),
            what,
            if self.reason == ThirdPersonReason::Default {
                String::new()
            } else {
                format!(" [{}]", self.reason.label())
            },
            if self.decoded { "" } else { " [the MODE is not fully decoded: a fallback is in play]" },
            if self.pose_is_authored { "" } else { " [the POSE is HMS's, standing in for an authored camera a film cannot drive]" }
        )
    }
}

// -------------------------------------------------------------------------------------------
// the timeline
// -------------------------------------------------------------------------------------------

/// One stretch during which the engine would have taken a body's camera OUT of the player's
/// preferred mode.
///
/// The timeline holds only the OVERRIDES, never the base mode, because the base is the viewer's
/// own choice (`film camera first` vs `film camera follow`) and must not be baked into a table
/// that is built once per film.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeSegment {
    pub from_tick: u32,
    /// Exclusive.
    pub to_tick: u32,
    pub mode: EngineCameraMode,
    pub reason: ThirdPersonReason,
    pub detail: ContextDetail,
    /// The transition time into this segment, seconds — the engine's own third argument to
    /// `director_set_camera_mode`.
    pub transition: f32,
    pub decoded: bool,
}

impl ModeSegment {
    pub fn covers(&self, tick: u32) -> bool {
        tick >= self.from_tick && tick < self.to_tick
    }
}

/// Every body's camera overrides for a whole film, built ONCE.
///
/// Why a table and not a per-frame scan: the same mistake has been made twice in this tree
/// already (`corpses()` walking the whole event channel per track per frame, and the ragdoll
/// catch-up re-simulating on every seek). A film can carry 50,000 events and 2,000 tracks, so
/// anything the draw path asks per frame has to be a lookup.
#[derive(Clone, Debug, Default)]
pub struct CameraTimeline {
    /// `net id -> its override segments, in tick order, non-overlapping`.
    pub per_body: std::collections::HashMap<u32, Vec<ModeSegment>>,
}

impl CameraTimeline {
    /// The override in force for this body at this tick, if any.
    pub fn at(&self, net_id: u32, tick: u32) -> Option<&ModeSegment> {
        let v = self.per_body.get(&net_id)?;
        // Segments are tick-ordered and disjoint, so the last one that starts at or before `tick`
        // is the only candidate.
        let i = v.partition_point(|s| s.from_tick <= tick);
        v[..i].last().filter(|s| s.covers(tick))
    }
    pub fn segments(&self, net_id: u32) -> &[ModeSegment] {
        self.per_body.get(&net_id).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn total_segments(&self) -> usize {
        self.per_body.values().map(|v| v.len()).sum()
    }
}

/// The seat index the recording STATES for a ride, or `None`.
///
/// `biped_enter_vehicle` fires **0** times in the whole corpus — boarding rides the vehicle's own
/// seat state, not an event — so the only place a film ever names a seat is
/// `biped_exit_vehicle` / `biped_vehicle_evict`, which carry a 6-bit seat index (404 of them
/// across the corpus). An exit names the seat the body was *in*, so an exit that falls at or just
/// after the end of a ride states that ride's seat and nothing else does.
pub fn seat_index_of_ride(idx: &crate::film_decode::FilmIndex, ride: &crate::film_anim::Ride) -> Option<u8> {
    idx.events
        .iter()
        .filter(|e| {
            matches!(e.kind, crate::film_action::EV_EXIT_VEHICLE | crate::film_action::EV_VEHICLE_EVICT)
                && e.subject == Some(ride.biped_net_id)
                && e.other == Some(ride.vehicle_net_id)
                // The ride detector cannot see the step-off tick (a body that just dismounted is
                // still beside the vehicle), so the exit can land either side of `to_tick`. One
                // second of slack either way is the window `film_play::seat_at` already works to.
                && e.tick + 60 >= ride.from_tick
                && e.tick <= ride.to_tick + 60
        })
        .min_by_key(|e| e.tick.abs_diff(ride.to_tick))
        .and_then(|e| e.seat)
}

/// Build the whole film's camera overrides.
///
/// The three sources, each already decoded elsewhere in HMS and each a statement of the
/// recording rather than an inference:
///
/// * **the dead chain** from [`Playback::death_at`] and the corpse window — the engine's
///   `first person|following -> dead -> orbiting -> respawning`;
/// * **the sync actions** from [`Playback::assassinations`], which are `incident` rows and
///   `biped_assassinate` events on the wire;
/// * **the rides** from [`crate::film_anim::accepted_rides`], cut by stated exits.
///
/// Precedence, and it matters: a body that is assassinated is dead at the same tick, and the
/// sync action has to win for its own duration or the victim's camera would cut to the dead
/// camera in the middle of the move. The engine has the same precedence for the same reason —
/// `unit_should_be_third_person`'s animation-mode gate is evaluated on a unit that is still
/// alive, and the player's unit does not go to NONE until the action completes.
pub fn build_timeline(pb: &Playback, tags: &CameraTags) -> CameraTimeline {
    let idx = &pb.index;
    let mut per: std::collections::HashMap<u32, Vec<ModeSegment>> = Default::default();

    // --- rides -> the vehicle camera -------------------------------------------------------
    for ride in crate::film_anim::accepted_rides(idx) {
        let seat = seat_index_of_ride(idx, &ride);
        let (third, from_tags) = tags.seat_is_third_person(ride.vehicle_net_id, seat);
        if !third {
            // The seat is authored first person, so the player's preference stands and there is
            // nothing to override. Recording it as a segment would be wrong, not merely noisy:
            // `at()` would then report a mode the engine never switched to.
            continue;
        }
        per.entry(ride.biped_net_id).or_default().push(ModeSegment {
            from_tick: ride.from_tick,
            to_tick: ride.to_tick + 1,
            mode: EngineCameraMode::Following,
            reason: if tags.seat_has_camera(ride.vehicle_net_id, seat) {
                ThirdPersonReason::Seat
            } else {
                ThirdPersonReason::NoSeatCamera
            },
            detail: ContextDetail::VehicleSeat {
                vehicle: ride.vehicle_net_id,
                seat,
                slaved: tags.seat_camera_is_slaved(ride.vehicle_net_id, seat),
            },
            transition: ENTER_VEHICLE_TRANSITION_SECONDS,
            decoded: from_tags && seat.is_some(),
        });
    }

    // --- sync actions -> the assassination camera ------------------------------------------
    for a in pb.assassinations() {
        for (net, attacker) in
            [(a.attacker, true), (a.victim, false)].into_iter().filter_map(|(n, w)| n.map(|n| (n, w)))
        {
            per.entry(net).or_default().push(ModeSegment {
                from_tick: a.tick,
                to_tick: a.tick + crate::film_anim::SYNC_TICKS,
                mode: EngineCameraMode::Following,
                reason: ThirdPersonReason::AnimationA,
                detail: ContextDetail::SyncAction {
                    attacker,
                    partner: if attacker { a.victim } else { a.attacker },
                    started: a.tick,
                },
                transition: ENTER_VEHICLE_TRANSITION_SECONDS,
                // The ACTION is stated; which of the two animation-mode tables it lands in is
                // not, because a film never carries the unit's animation mode (bit 0x36 appears
                // twice in 7.37 million messages).
                decoded: a.stated,
            });
        }
    }

    // --- weapons that force third person ---------------------------------------------------
    //
    // `EntityTrack::weapons` is the recording's own statement of what a body is holding: a unit
    // message carries a `'weap'` tag reference and HMS already decodes it (it is what dresses the
    // hands). So "is this player holding a weapon the engine would take out of first person for"
    // is a decoded fact per tick, and the only outside input is the flag itself.
    //
    // Without a map open `third_person_weapons` is empty and this contributes nothing — the
    // honest answer, because the flag lives in the tag and nothing in a film states it.
    if !tags.third_person_weapons.is_empty() {
        for tr in &idx.tracks {
            if tr.entity_type != crate::film_anim::BIPED_TYPE {
                continue;
            }
            for (i, &(t, w)) in tr.weapons.iter().enumerate() {
                if !tags.third_person_weapons.contains(&w) {
                    continue;
                }
                // The run lasts until the body is reported holding something else, or to the end
                // of its life.
                let end = tr.weapons.get(i + 1).map(|&(t2, _)| t2).unwrap_or(tr.last_tick + 1);
                if end <= t {
                    continue;
                }
                per.entry(tr.net_id).or_default().push(ModeSegment {
                    from_tick: t,
                    to_tick: end,
                    mode: EngineCameraMode::Following,
                    reason: ThirdPersonReason::Weapon,
                    detail: ContextDetail::WeaponThirdPerson { weapon: w },
                    transition: ENTER_VEHICLE_TRANSITION_SECONDS,
                    decoded: true,
                });
            }
        }
    }

    // --- the dead chain --------------------------------------------------------------------
    let orbit_at = (DEAD_TO_ORBIT_SECONDS * TICK_HZ).round() as u32;
    for tr in &idx.tracks {
        let Some(died) = pb.death_at(tr) else { continue };
        // A victim of a decoded assassination goes down only when the clip ends, which is what
        // `film_play::corpse_window` already does for the body itself; the camera has to agree or
        // the view would cut to the corpse while the body is still being drawn mid-move.
        let assassinated = pb
            .deaths()
            .get(&tr.net_id)
            .is_some_and(|&(_, inc, _)| crate::film_action::ASSASSINATION_INCIDENTS.contains(&inc));
        let down = died + if assassinated { crate::film_anim::SYNC_TICKS } else { 0 };
        let gone = tr.destroyed_tick.unwrap_or(tr.last_tick).max(down + 1);
        let v = per.entry(tr.net_id).or_default();
        v.push(ModeSegment {
            from_tick: down,
            to_tick: (down + orbit_at).min(gone.max(down + 1)),
            mode: EngineCameraMode::Dead,
            reason: ThirdPersonReason::Default,
            detail: ContextDetail::Dead { died_at: down },
            transition: DEATH_SWITCH_SECONDS,
            decoded: true,
        });
        if gone > down + orbit_at {
            v.push(ModeSegment {
                from_tick: down + orbit_at,
                to_tick: gone,
                mode: EngineCameraMode::Orbiting,
                reason: ThirdPersonReason::Default,
                detail: ContextDetail::Orbiting { died_at: down },
                transition: DEATH_SWITCH_SECONDS,
                decoded: true,
            });
        }
    }

    // Sort and resolve overlaps once, so `at()` can binary-search. The order of the pushes above
    // is the precedence: a sync action must beat the dead chain over its own window, so it is
    // kept and the dead segment is pushed past it.
    for v in per.values_mut() {
        *v = resolve_overlaps(std::mem::take(v));
    }
    per.retain(|_, v| !v.is_empty());
    CameraTimeline { per_body: per }
}

/// Flatten a body's overlapping override segments into a disjoint, tick-ordered list, keeping the
/// HIGHEST-PRIORITY segment over every tick.
///
/// A **sweep over the boundary points**, not a pairwise clip, and the difference is a real bug the
/// pairwise version had: clipping only the segment already in hand loses the TAIL of a
/// lower-priority segment that a higher-priority one sits strictly inside. With the way
/// `build_timeline` currently lays segments out that case cannot arise — a victim's sync window
/// ends exactly where their dead segment begins, and the dead chain always runs to the end of the
/// track — but "cannot arise today" is how a trap gets left behind, and the sweep is no harder to
/// read than the clip was.
///
/// Adjacent runs of the same mode and detail are merged, so the result does not report a
/// transition where nothing changed.
fn resolve_overlaps(segs: Vec<ModeSegment>) -> Vec<ModeSegment> {
    if segs.len() < 2 {
        return segs.into_iter().filter(|s| s.from_tick < s.to_tick).collect();
    }
    let mut bounds: Vec<u32> = Vec::with_capacity(segs.len() * 2);
    for s in &segs {
        if s.from_tick < s.to_tick {
            bounds.push(s.from_tick);
            bounds.push(s.to_tick);
        }
    }
    bounds.sort_unstable();
    bounds.dedup();
    let mut out: Vec<ModeSegment> = Vec::new();
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        // The winner over [a, b): highest priority, and among equals the one that started first,
        // so the answer does not depend on the input order.
        let Some(win) = segs
            .iter()
            .filter(|s| s.from_tick <= a && s.to_tick >= b)
            .max_by_key(|s| (priority(&s.detail), std::cmp::Reverse(s.from_tick)))
        else {
            continue;
        };
        match out.last_mut() {
            Some(last)
                if last.to_tick == a && last.mode == win.mode && last.detail == win.detail =>
            {
                last.to_tick = b;
            }
            _ => out.push(ModeSegment { from_tick: a, to_tick: b, ..win.clone() }),
        }
    }
    out
}

/// Which override beats which, where two cover the same tick. Higher wins.
fn priority(d: &ContextDetail) -> u8 {
    match d {
        ContextDetail::SyncAction { .. } => 4,
        ContextDetail::Dead { .. } | ContextDetail::Orbiting { .. } | ContextDetail::Respawning { .. } => 3,
        ContextDetail::VehicleSeat { .. } => 2,
        // STRICTLY below the seat, and this is the engine's order rather than a preference:
        // `unit_should_be_third_person` resolves the PARENT path (reasons 1 and 2, the seat) and
        // only falls through to the weapon (reason 4) when the unit has no parent. A rider's seat
        // therefore decides, whatever they are holding.
        //
        // These two were both 1 at first, with this very comment above them. Equal priority does
        // not mean "the seat wins": the sweep's tie-break is which segment STARTED first, so a
        // player who picked the turret up before boarding kept the turret's camera through the
        // whole ride. `a_vehicle_seat_beats_a_third_person_weapon` is the gate that caught it.
        ContextDetail::WeaponThirdPerson { .. } => 1,
        ContextDetail::Plain => 0,
    }
}

// -------------------------------------------------------------------------------------------
// the resolver
// -------------------------------------------------------------------------------------------

/// Which camera the engine would be in for `net_id` at `tick`, given the viewer's preferred mode.
///
/// **A pure function of `(timeline, preference, net_id, tick)`.** No state carries between calls
/// and nothing is remembered from the last tick, so seeking backwards produces the identical
/// answer — which is the determinism rule the ragdoll work had to learn the hard way.
pub fn resolve(
    timeline: &CameraTimeline,
    preferred: EngineCameraMode,
    net_id: u32,
    tick: u32,
) -> CameraContext {
    match timeline.at(net_id, tick) {
        Some(s) => {
            let ticks = (s.transition * TICK_HZ).round() as u32;
            let blend = if ticks == 0 {
                1.0
            } else {
                ((tick - s.from_tick) as f32 / ticks as f32).clamp(0.0, 1.0)
            };
            CameraContext {
                mode: s.mode,
                reason: s.reason,
                subject: net_id,
                detail: s.detail.clone(),
                blend,
                entered: s.from_tick,
                decoded: s.decoded,
                pose_is_authored: matches!(
                    s.detail,
                    ContextDetail::Dead { .. }
                        | ContextDetail::Orbiting { .. }
                        | ContextDetail::Respawning { .. }
                ),
            }
        }
        None => {
            // No override: the player's own preference, and the engine's transition back out of
            // whatever was in force. `entered` is the end of the previous override so the blend
            // back is a function of the tick too.
            let entered = timeline
                .segments(net_id)
                .iter()
                .filter(|s| s.to_tick <= tick)
                .map(|s| s.to_tick)
                .max()
                .unwrap_or(0);
            let ticks = (ENTER_VEHICLE_TRANSITION_SECONDS * TICK_HZ).round() as u32;
            let blend = if ticks == 0 || entered == 0 {
                1.0
            } else {
                ((tick - entered) as f32 / ticks as f32).clamp(0.0, 1.0)
            };
            CameraContext {
                mode: preferred,
                reason: ThirdPersonReason::Default,
                subject: net_id,
                detail: ContextDetail::Plain,
                blend,
                entered,
                decoded: true,
                // The plain camera is the viewer's own, so there is no authored shot to match.
                pose_is_authored: true,
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
// the pose
// -------------------------------------------------------------------------------------------

/// Where a context camera sits.
///
/// `plain` is the pose the viewer's own preferred camera would have produced at this same tick;
/// the context pose is blended from it over the engine's transition time, so a switch eases the
/// way the engine's does and a scrub lands on the identical frame.
pub fn context_pose(
    pb: &Playback,
    ctx: &CameraContext,
    tick: u32,
    rig: FollowRig,
    plain: &CameraPose,
) -> Option<CameraPose> {
    let target = match &ctx.detail {
        // Coming back OUT of a context camera. The engine eases this too — it passes a transition
        // time when it returns a player to their preferred mode — so the symmetric thing is to
        // blend from the shot that just ended toward the plain pose, rather than snapping.
        //
        // The outgoing shot is still computable past its own end: the orbit formulas do not care
        // that the segment is over, and `timeline.at` is not consulted, so this stays a pure
        // function of the tick and the scrub gate still holds. Where there is nothing to come
        // back from (`entered == 0`, or the previous segment has gone) the plain pose stands.
        ContextDetail::Plain => {
            if ctx.blend >= 1.0 || ctx.entered == 0 {
                return Some(plain.clone());
            }
            let prev = pb
                .camera_timeline()
                .segments(ctx.subject)
                .iter()
                .find(|s| s.to_tick == ctx.entered)
                .cloned();
            let Some(prev) = prev else { return Some(plain.clone()) };
            // Re-enter the outgoing context AT ITS OWN mode, fully blended in, and ask for its
            // pose at this tick.
            let outgoing = CameraContext {
                mode: prev.mode,
                reason: prev.reason,
                subject: ctx.subject,
                detail: prev.detail.clone(),
                blend: 1.0,
                entered: prev.from_tick,
                decoded: prev.decoded,
                pose_is_authored: false,
            };
            let from = context_pose(pb, &outgoing, tick, rig, plain)?;
            let mut out = blend_pose(&from, plain, ctx.blend);
            out.label = format!(
                "{} [leaving {} ({}), {:.0}% back]",
                plain.label,
                prev.mode.name(),
                prev.mode.index(),
                ctx.blend * 100.0
            );
            return Some(out);
        }

        // The vehicle camera frames the VEHICLE, not the rider. That is the whole visible
        // difference, and the recording supports it: a type-15 entity replicates a full
        // (forward, up) basis on 96.2% of its position updates, so the camera can sit behind the
        // vehicle's own facing rather than behind the player's aim.
        ContextDetail::VehicleSeat { vehicle, .. } => {
            let tr = pb.index.tracks.iter().find(|t| t.net_id == *vehicle)?;
            let p = tr.sample(tick)?;
            let yaw = tr
                .facing_at(tick)
                .or_else(|| tr.sample_orientation(tick).map(|(f, _)| f[1].atan2(f[0])))
                .unwrap_or(plain.yaw);
            orbit(p, crate::film_play::FOLLOW_PIVOT_WU, yaw + rig.yaw_off, rig.elev, rig.dist)
        }

        // The assassination. Both halves of a Reach sync action are authored around ONE root and
        // HMS already draws them that way (`ActorDriver::sync_at` puts the victim on the
        // attacker's transform), so the camera orbits that same root. Framing the victim's own
        // replicated position would not work: a body stops replicating almost as soon as the
        // action starts, which is the reason the shared root exists.
        ContextDetail::SyncAction { attacker, partner, .. } => {
            let root = if *attacker { Some(ctx.subject) } else { *partner };
            let tr = pb.index.tracks.iter().find(|t| Some(t.net_id) == root)?;
            let p = tr.sample(ctx.entered).or_else(|| tr.sample(tick))?;
            let yaw = tr.facing_at(ctx.entered).unwrap_or(plain.yaw);
            // **The framing here is HMS's, and it has to be said plainly.** The engine's
            // sync-action camera is `unit +0x0A8 sync action camera` (`bipd +0x210`), and the
            // Spartan authors it as a `camera tracks` block with ONE entry — i.e. a `trak`
            // camera track, a list of (position, quaternion) control points. HMS cannot evaluate
            // it: a `trak` is authored in the sync action's own local space and the recording
            // samples an object's ROOT basis and not its node poses, so there is nothing to hang
            // the track on. So this is an orbit of the shared root, swung a quarter turn off the
            // attacker's facing so the move is side-on rather than looking at the attacker's
            // back, at HMS's own follow distance. It is a stand-in for the authored camera, not
            // that camera, and `CameraContext::label` does not claim otherwise.
            orbit(
                p,
                crate::film_play::FOLLOW_PIVOT_WU,
                yaw + rig.yaw_off + std::f32::consts::FRAC_PI_4,
                rig.elev,
                rig.dist,
            )
        }

        // A weapon that forces third person. The engine's shot here is the ordinary `following`
        // camera on the unit -- there is no separate authored camera for it, only the flag that
        // says to leave first person -- so HMS's follow rig IS the right shape.
        // `weap +0x498 support third person camera range` is the pitch range it would be framed
        // within: decoded, reported, NOT applied, because a film does not record the view pitch
        // for anyone but the recorder.
        ContextDetail::WeaponThirdPerson { .. } => {
            let tr = pb.index.tracks.iter().find(|t| t.net_id == ctx.subject)?;
            let p = tr.sample(tick)?;
            let yaw = tr
                .facing_at(tick)
                .or_else(|| tr.sample_orientation(tick).map(|(f, _)| f[1].atan2(f[0])))
                .unwrap_or(plain.yaw);
            orbit(p, crate::film_play::FOLLOW_PIVOT_WU, yaw + rig.yaw_off, rig.elev, rig.dist)
        }

        // The DEAD camera, with the engine's own geometry: a pivot `dead cam z offset` above the
        // body's root, easing from `dead cam initial distance` out to `dead cam final distance`
        // over `dead camera transition time`, elevation capped at `dead cam maximum elevation`.
        // That ramp is why a Reach death reads as the view dropping away from you and then
        // closing back in on your body, and it is the one easing the death chain has: the MODE
        // switch itself is a cut (see `DEATH_SWITCH_SECONDS`).
        ContextDetail::Dead { died_at } => {
            let tr = pb.index.tracks.iter().find(|t| t.net_id == ctx.subject)?;
            let p = tr.sample(tick).or_else(|| tr.sample(ctx.entered))?;
            let secs = tick.saturating_sub(*died_at) as f32 / TICK_HZ;
            orbit(
                p,
                DEAD_CAM_Z_OFFSET_WU,
                plain.yaw + rig.yaw_off,
                rig.elev.min(DEAD_CAM_MAX_ELEVATION_RAD),
                dead_cam_distance(secs),
            )
        }
        // The ORBITING camera — what the player actually spectates their own body from while the
        // respawn counts down. `orbit cam maximum distance` is the distance
        // `game_director::update` hands the orbiting camera the moment `dead` ends, and the
        // elevation is clamped to the orbit camera's own authored limits so a drag of HMS's
        // orbit handle cannot put the shot somewhere the game could not.
        ContextDetail::Orbiting { .. } | ContextDetail::Respawning { .. } => {
            let tr = pb.index.tracks.iter().find(|t| t.net_id == ctx.subject)?;
            let p = tr.sample(tick).or_else(|| tr.sample(ctx.entered))?;
            orbit(
                p,
                ORBIT_CAM_Z_OFFSET_WU,
                plain.yaw + rig.yaw_off,
                rig.elev.clamp(ORBIT_CAM_MIN_ELEVATION_RAD, ORBIT_CAM_MAX_ELEVATION_RAD),
                rig.dist.clamp(ORBIT_CAM_MIN_DIST_WU, ORBIT_CAM_MAX_DIST_WU),
            )
        }
    };
    let mut pose = target;
    pose.label = ctx.label();
    if ctx.blend < 1.0 {
        pose = blend_pose(plain, &pose, ctx.blend);
        pose.label = format!("{} [{:.0}% in]", ctx.label(), ctx.blend * 100.0);
    }
    Some(pose)
}

/// A camera orbiting a pivot above a point, looking at it. The same rig the plain follow camera
/// uses, so a context camera and the viewer's own orbit handle agree.
fn orbit(feet: [f32; 3], pivot_up: f32, yaw: f32, elev: f32, dist: f32) -> CameraPose {
    let pivot = [feet[0], feet[1], feet[2] + pivot_up];
    let (ce, se) = (elev.cos(), elev.sin());
    CameraPose {
        pos: [
            pivot[0] - yaw.cos() * ce * dist,
            pivot[1] - yaw.sin() * ce * dist,
            pivot[2] + se * dist,
        ],
        yaw,
        pitch: -elev,
        pivot: Some(pivot),
        label: String::new(),
    }
}

/// Ease between two poses. `t` is the engine's transition progress, and it is applied to a
/// smoothstep rather than linearly so the cut does not have a visible corner at either end.
fn blend_pose(a: &CameraPose, b: &CameraPose, t: f32) -> CameraPose {
    let t = t.clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    let l = |x: f32, y: f32| x + (y - x) * s;
    CameraPose {
        pos: [l(a.pos[0], b.pos[0]), l(a.pos[1], b.pos[1]), l(a.pos[2], b.pos[2])],
        // Yaw is an angle: interpolate the SHORT way round or a camera crossing +-pi spins all
        // the way through the back of the map.
        yaw: a.yaw + wrap_pi(b.yaw - a.yaw) * s,
        pitch: l(a.pitch, b.pitch),
        pivot: b.pivot.or(a.pivot),
        label: b.label.clone(),
    }
}

fn wrap_pi(mut x: f32) -> f32 {
    while x > std::f32::consts::PI {
        x -= std::f32::consts::TAU;
    }
    while x < -std::f32::consts::PI {
        x += std::f32::consts::TAU;
    }
    x
}

// -------------------------------------------------------------------------------------------
// the setting
// -------------------------------------------------------------------------------------------

/// #film-camera  Whether Theater follows the engine into its context cameras, remembered between
/// sessions exactly the way `spectate_card.txt` and `first_person_fov.txt` are.
///
/// Default **on**: it is what the game does, and the whole point of a Theater mode is to show
/// what the game showed. The toggle exists because a viewer lining up a shot wants the camera to
/// stay where they put it.
const CONTEXT_SETTING_FILE: &str = "context_cameras.txt";

pub fn load_context_setting() -> bool {
    crate::project::settings_dir().map(|d| load_context_from_dir(&d)).unwrap_or(true)
}

pub fn load_context_from_dir(dir: &std::path::Path) -> bool {
    match std::fs::read_to_string(dir.join(CONTEXT_SETTING_FILE)) {
        Ok(s) => !matches!(s.trim(), "0" | "off" | "false"),
        Err(_) => true,
    }
}

pub fn save_context_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() {
        save_context_to_dir(&dir, on);
    }
}

pub fn save_context_to_dir(dir: &std::path::Path, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(CONTEXT_SETTING_FILE), if on { "1" } else { "0" });
}

// -------------------------------------------------------------------------------------------
// tests
// -------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's own enum, and the two independent facts that pin it: the thirteen-slot name
    /// array and `director_set_camera_mode`'s `> 0xC` reject.
    #[test]
    fn engine_camera_mode_enum_matches_the_binary() {
        assert_eq!(ENGINE_CAMERA_MODE_NAMES.len(), NUMBER_OF_CAMERA_MODES);
        assert_eq!(NUMBER_OF_CAMERA_MODES, 13);
        // Every index round-trips, and no index outside the enum resolves.
        for i in 0..13u8 {
            let m = EngineCameraMode::from_index(i).expect("every index 0..12 is a mode");
            assert_eq!(m.index(), i);
            assert_eq!(m.name(), ENGINE_CAMERA_MODE_NAMES[i as usize]);
        }
        assert!(EngineCameraMode::from_index(13).is_none(), "k_number_of_camera_modes is 13");
        // The four the camera work turns on, by position. If anybody reorders the array these
        // fail, which is the point: the order is the binary's, not a convenience.
        assert_eq!(EngineCameraMode::FirstPerson.index(), 3);
        assert_eq!(EngineCameraMode::Dead.index(), 4);
        assert_eq!(EngineCameraMode::Orbiting.index(), 1);
        assert_eq!(EngineCameraMode::Following.index(), 0);
        assert_eq!(ENGINE_CAMERA_MODE_NAMES[3], "first person");
        assert_eq!(ENGINE_CAMERA_MODE_NAMES[4], "dead");
    }

    /// The seat and weapon bits, as `unit_should_be_third_person` reads them.
    #[test]
    fn seat_and_weapon_flag_bits_are_the_engine_s() {
        assert_eq!(SEAT_FLAG_THIRD_PERSON, 0x10, "flags >> 4 & 1");
        assert_eq!(SEAT_FLAG_THIRD_PERSON_ON_ENTER, 0x40, "flags & 0x40");
        assert_eq!(SEAT_FLAG_CAMERA_SLAVED_TO_GUN, 0x80, "flags & 0x80 in +0x96EA90");
        assert_eq!(WEAPON_FLAG_THIRD_PERSON, 0x8000, "value >> 0xF & 1");
        let mut t = CameraTags::default();
        t.seat_flags.insert(7, vec![SEAT_FLAG_THIRD_PERSON_ON_ENTER, SEAT_FLAG_THIRD_PERSON_ON_ENTER | SEAT_FLAG_THIRD_PERSON]);
        // Seat 0 is authored FIRST person, so the bit must say so and must say it was decoded.
        assert_eq!(t.seat_is_third_person(7, Some(0)), (false, true));
        assert_eq!(t.seat_is_third_person(7, Some(1)), (true, true));
        // An unknown vehicle, and a known vehicle with an unstated seat, both fall back to third
        // person AND must report that the bit was not read.
        assert_eq!(t.seat_is_third_person(99, Some(0)), (true, false));
        assert_eq!(t.seat_is_third_person(7, None), (true, false));
    }

    /// The transition times, which are the only blending the engine does.
    #[test]
    fn the_death_transition_is_a_cut_and_the_person_switch_is_not() {
        assert_eq!(DEATH_SWITCH_SECONDS, 0.0, "game_director passes literal 0.0 on the dead chain");
        assert!((ENTER_VEHICLE_TRANSITION_SECONDS - 0.55).abs() < 1e-6, "globals camera +0xB8");
        assert!((ORBIT_CAM_MAX_DIST_WU - 2.5).abs() < 1e-6, "globals camera +0x98");
        assert!((DEAD_TO_ORBIT_SECONDS - 2.0).abs() < 1e-6, "globals camera +0x58/+0x5C/+0x60");
    }

    fn seg(from: u32, to: u32, d: ContextDetail, transition: f32) -> ModeSegment {
        ModeSegment {
            from_tick: from,
            to_tick: to,
            mode: EngineCameraMode::Following,
            reason: ThirdPersonReason::Default,
            detail: d,
            transition,
            decoded: true,
        }
    }

    fn timeline(v: Vec<ModeSegment>) -> CameraTimeline {
        let mut t = CameraTimeline::default();
        t.per_body.insert(1, v);
        t
    }

    /// A lookup must land on the segment that covers the tick and on nothing else, including at
    /// both edges — `to_tick` is exclusive and a half-open bug here would show as a camera that
    /// sticks one tick past a death.
    #[test]
    fn timeline_lookup_is_half_open_and_exact() {
        let t = timeline(vec![
            seg(100, 220, ContextDetail::Dead { died_at: 100 }, 0.0),
            seg(220, 400, ContextDetail::Orbiting { died_at: 100 }, 0.0),
        ]);
        assert!(t.at(1, 99).is_none(), "before the first segment");
        assert!(matches!(t.at(1, 100).map(|s| &s.detail), Some(ContextDetail::Dead { .. })));
        assert!(matches!(t.at(1, 219).map(|s| &s.detail), Some(ContextDetail::Dead { .. })));
        assert!(
            matches!(t.at(1, 220).map(|s| &s.detail), Some(ContextDetail::Orbiting { .. })),
            "to_tick is EXCLUSIVE: 220 belongs to the orbit, not the dead camera"
        );
        assert!(matches!(t.at(1, 399).map(|s| &s.detail), Some(ContextDetail::Orbiting { .. })));
        assert!(t.at(1, 400).is_none(), "after the last segment");
        assert!(t.at(2, 150).is_none(), "another body has no segments");
    }

    /// **Determinism.** The whole resolver has to be a pure function of the tick, because the
    /// viewer scrubs. Walking the timeline forwards and then backwards must produce byte-identical
    /// contexts at every tick.
    #[test]
    fn resolving_is_a_pure_function_of_the_tick() {
        let t = timeline(vec![
            seg(100, 160, ContextDetail::SyncAction { attacker: false, partner: Some(2), started: 100 }, ENTER_VEHICLE_TRANSITION_SECONDS),
            seg(160, 280, ContextDetail::Dead { died_at: 160 }, 0.0),
            seg(280, 500, ContextDetail::Orbiting { died_at: 160 }, 0.0),
        ]);
        let fwd: Vec<CameraContext> =
            (0..600).map(|k| resolve(&t, EngineCameraMode::FirstPerson, 1, k)).collect();
        let back: Vec<CameraContext> =
            (0..600).rev().map(|k| resolve(&t, EngineCameraMode::FirstPerson, 1, k)).collect();
        for (k, (f, b)) in fwd.iter().zip(back.iter().rev()).enumerate() {
            assert_eq!(f, b, "tick {k} must resolve identically scrubbing either way");
        }
        // And the transition itself is monotone in the tick rather than wall clock.
        assert_eq!(fwd[100].blend, 0.0, "the sync action is 0% in on its first tick");
        let ticks = (ENTER_VEHICLE_TRANSITION_SECONDS * TICK_HZ).round() as u32;
        assert!((fwd[(100 + ticks) as usize].blend - 1.0).abs() < 1e-6, "fully in after {ticks} ticks");
        assert_eq!(fwd[160].blend, 1.0, "the dead camera is a CUT, so it is 100% in immediately");
    }

    /// Coming back OUT of a context camera is eased, not snapped — the engine passes a
    /// transition time when it returns a player to their preferred mode, so the symmetric thing
    /// is to blend back. The blend has to be a function of the TICK like every other, so this
    /// checks it rises monotonically from 0 at the tick the override ends to 1 one transition
    /// later, and stays there.
    #[test]
    fn leaving_a_context_camera_eases_back_as_a_function_of_the_tick() {
        let t = timeline(vec![seg(
            100,
            200,
            ContextDetail::VehicleSeat { vehicle: 9, seat: Some(0), slaved: false },
            0.0,
        )]);
        let at = |k| resolve(&t, EngineCameraMode::FirstPerson, 1, k);
        // Inside the override there is nothing to come back from.
        assert_eq!(at(150).detail, ContextDetail::VehicleSeat { vehicle: 9, seat: Some(0), slaved: false });
        // The tick it ends: back to the preference, 0% of the way back.
        assert_eq!(at(200).detail, ContextDetail::Plain);
        assert_eq!(at(200).blend, 0.0, "the ease-out starts at 0");
        assert_eq!(at(200).entered, 200, "and it is anchored on the tick the override ENDED");
        let ticks = (ENTER_VEHICLE_TRANSITION_SECONDS * TICK_HZ).round() as u32;
        // Monotone in the tick, and never backwards.
        let mut last = -1.0f32;
        for k in 200..=(200 + ticks) {
            let b = at(k).blend;
            assert!(b >= last, "tick {k}: blend went backwards, {last} -> {b}");
            last = b;
        }
        assert!((at(200 + ticks).blend - 1.0).abs() < 1e-6, "fully back after {ticks} ticks");
        assert_eq!(at(200 + ticks + 60).blend, 1.0, "and it stays there");
        // A body that was never in an override has nothing to ease out of, so it is simply plain.
        let empty = CameraTimeline::default();
        let c = resolve(&empty, EngineCameraMode::FirstPerson, 1, 500);
        assert_eq!(c.detail, ContextDetail::Plain);
        assert_eq!(c.entered, 0, "no previous override");
        assert_eq!(c.blend, 1.0, "so no ease-out is in progress");
    }

    /// The precedence that matters: a body that is assassinated is dead at the same tick, and the
    /// sync action has to hold its own window or the victim's camera cuts to the corpse mid-move.
    #[test]
    fn a_sync_action_beats_the_dead_chain_over_its_own_window() {
        let t = timeline(resolve_overlaps(vec![
            seg(100, 300, ContextDetail::Dead { died_at: 100 }, 0.0),
            seg(100, 200, ContextDetail::SyncAction { attacker: false, partner: Some(2), started: 100 }, 0.0),
        ]));
        assert!(
            matches!(t.at(1, 150).map(|s| &s.detail), Some(ContextDetail::SyncAction { .. })),
            "mid-assassination the camera must still be on the action, got {:?}",
            t.at(1, 150)
        );
        assert!(
            matches!(t.at(1, 250).map(|s| &s.detail), Some(ContextDetail::Dead { .. })),
            "once the action is over the dead camera takes it, got {:?}",
            t.at(1, 250)
        );
    }

    /// **The sweep must not lose the TAIL** of a lower-priority segment that a higher-priority one
    /// sits strictly inside. This is the exact bug the pairwise clip had: with a sync action from
    /// 150 to 200 inside a dead segment from 100 to 300, the clip emitted dead [100,150) and sync
    /// [150,200) and then simply dropped dead [200,300), so the camera returned to the viewer's
    /// own mode in the middle of a death.
    #[test]
    fn the_sweep_keeps_the_tail_of_a_segment_it_interrupts() {
        let out = resolve_overlaps(vec![
            seg(100, 300, ContextDetail::Dead { died_at: 100 }, 0.0),
            seg(150, 200, ContextDetail::SyncAction { attacker: true, partner: Some(2), started: 150 }, 0.0),
        ]);
        let spans: Vec<(u32, u32, bool)> = out
            .iter()
            .map(|s| (s.from_tick, s.to_tick, matches!(s.detail, ContextDetail::SyncAction { .. })))
            .collect();
        assert_eq!(
            spans,
            vec![(100, 150, false), (150, 200, true), (200, 300, false)],
            "the interrupted segment must resume; got {spans:?}"
        );
        // Disjoint, ordered, and gapless across the whole covered range.
        for w in out.windows(2) {
            assert_eq!(w[0].to_tick, w[1].from_tick, "no gap and no overlap: {:?}", spans);
        }
        let t = timeline(out);
        assert!(matches!(t.at(1, 250).map(|s| &s.detail), Some(ContextDetail::Dead { .. })));
    }

    /// Adjacent pieces of the SAME mode and detail are merged, so the timeline does not report a
    /// transition where nothing changed — and `resolve`'s blend therefore does not restart.
    #[test]
    fn the_sweep_merges_runs_that_did_not_actually_change() {
        let out = resolve_overlaps(vec![
            seg(100, 200, ContextDetail::Dead { died_at: 100 }, 0.0),
            seg(150, 300, ContextDetail::Dead { died_at: 100 }, 0.0),
        ]);
        assert_eq!(out.len(), 1, "one continuous dead segment, got {out:?}");
        assert_eq!((out[0].from_tick, out[0].to_tick), (100, 300));
    }

    /// The orbit rig: the camera must sit the stated distance from its pivot and look back at it,
    /// at every angle. A sign error here is invisible in one shot and obvious in a pan.
    #[test]
    fn the_orbit_sits_at_its_distance_and_looks_at_the_pivot() {
        for &yaw in &[-3.0f32, -1.0, 0.0, 0.7, 2.5, 3.1] {
            for &elev in &[-0.5f32, 0.0, 0.2, 0.9] {
                let p = orbit([10.0, -4.0, 2.0], 0.4, yaw, elev, ORBIT_CAM_MAX_DIST_WU);
                let pivot = p.pivot.expect("an orbit has a pivot");
                let d = ((p.pos[0] - pivot[0]).powi(2)
                    + (p.pos[1] - pivot[1]).powi(2)
                    + (p.pos[2] - pivot[2]).powi(2))
                .sqrt();
                assert!(
                    (d - ORBIT_CAM_MAX_DIST_WU).abs() < 1e-3,
                    "yaw {yaw} elev {elev}: distance {d}, want {ORBIT_CAM_MAX_DIST_WU}"
                );
                // Looking from the camera back at the pivot is exactly (yaw, -elev).
                let to = [pivot[0] - p.pos[0], pivot[1] - p.pos[1], pivot[2] - p.pos[2]];
                let want_yaw = to[1].atan2(to[0]);
                assert!(
                    wrap_pi(p.yaw - want_yaw).abs() < 1e-3,
                    "yaw {yaw} elev {elev}: camera yaw {} does not point at the pivot ({want_yaw})",
                    p.yaw
                );
                assert!((p.pitch + elev).abs() < 1e-6);
            }
        }
    }

    /// Blending an angle must take the SHORT way round. The naive lerp sends a camera crossing
    /// +-pi all the way through the back of the map, which is a full spin on screen.
    #[test]
    fn the_blend_takes_the_short_way_round_pi() {
        let a = CameraPose { pos: [0.0; 3], yaw: 3.0, pitch: 0.0, pivot: None, label: String::new() };
        let b = CameraPose { pos: [0.0; 3], yaw: -3.0, pitch: 0.0, pivot: None, label: String::new() };
        let mid = blend_pose(&a, &b, 0.5);
        // The short way from 3.0 to -3.0 is +0.283 rad through pi, so halfway is past pi.
        assert!(
            mid.yaw > 3.0,
            "halfway between 3.0 and -3.0 must go THROUGH pi (got {}), not back through 0",
            mid.yaw
        );
        assert!(wrap_pi(mid.yaw).abs() > 3.0, "and it must still be near pi: {}", mid.yaw);
        // The ends are exact.
        assert!((blend_pose(&a, &b, 0.0).yaw - 3.0).abs() < 1e-6);
        assert!(wrap_pi(blend_pose(&a, &b, 1.0).yaw - (-3.0)).abs() < 1e-5);
    }

    /// **The corpus gate.** Build the camera timeline for EVERY film on the machine and check
    /// the STRUCTURE of what comes out, not an arithmetic identity: that a body's segments are
    /// tick-ordered and never overlap, that every segment is non-empty, and that a lookup at
    /// every segment boundary lands on exactly the segment that owns it.
    ///
    /// Overlapping segments is the specific bug this is here to catch, and it is not
    /// hypothetical: a body that is assassinated is dead at the same tick, so the sync-action
    /// window and the dead chain are built over the same ticks by construction, and the
    /// resolution pass is the only thing separating them. If it ever stops separating them,
    /// `at()`'s binary search silently returns whichever segment happens to sort first and the
    /// victim's camera cuts to their corpse in the middle of the kill. Arithmetic on distances
    /// could not see that; this can.
    ///
    /// It also prints the census, which is where the report's real numbers come from.
    #[test]
    fn camera_timelines_are_well_formed_on_every_real_film() {
        let films = crate::film::find_finished_films();
        if films.is_empty() {
            eprintln!("no films on this machine; the corpus gate is vacuous here");
            return;
        }
        let (mut nfilms, mut nbodies, mut nsegs) = (0usize, 0usize, 0usize);
        let mut by_kind: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut seat_stated = 0usize;
        let mut checked_boundaries = 0usize;
        for p in &films {
            let Ok(f) = crate::film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            if idx.tracks.is_empty() {
                continue;
            }
            let pb = Playback::new(idx);
            let tl = pb.camera_timeline();
            nfilms += 1;
            nbodies += tl.per_body.len();
            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            for (net, segs) in &tl.per_body {
                nsegs += segs.len();
                let mut prev_end = 0u32;
                for s in segs {
                    assert!(
                        s.from_tick < s.to_tick,
                        "{stem}: body 0x{net:08X} has an EMPTY segment {s:?}"
                    );
                    assert!(
                        s.from_tick >= prev_end,
                        "{stem}: body 0x{net:08X} segments OVERLAP -- {s:?} starts before {prev_end}"
                    );
                    prev_end = s.to_tick;
                    *by_kind
                        .entry(match &s.detail {
                            ContextDetail::Plain => "plain",
                            ContextDetail::VehicleSeat { seat, .. } => {
                                if seat.is_some() {
                                    seat_stated += 1;
                                }
                                "vehicle seat"
                            }
                            ContextDetail::WeaponThirdPerson { .. } => "weapon 3rd person",
                            ContextDetail::SyncAction { .. } => "sync action",
                            ContextDetail::Dead { .. } => "dead",
                            ContextDetail::Orbiting { .. } => "orbiting",
                            ContextDetail::Respawning { .. } => "respawning",
                        })
                        .or_default() += 1;
                    // Every boundary resolves to exactly the segment that owns it. This is the
                    // half-open check, run against real data rather than a fixture.
                    let got = tl.at(*net, s.from_tick).map(|g| (g.from_tick, g.to_tick));
                    assert_eq!(
                        got,
                        Some((s.from_tick, s.to_tick)),
                        "{stem}: body 0x{net:08X} tick {} resolved to {got:?}, not its own segment",
                        s.from_tick
                    );
                    if s.to_tick > s.from_tick {
                        let last = tl.at(*net, s.to_tick - 1).map(|g| (g.from_tick, g.to_tick));
                        assert_eq!(
                            last,
                            Some((s.from_tick, s.to_tick)),
                            "{stem}: body 0x{net:08X} last tick {} resolved to {last:?}",
                            s.to_tick - 1
                        );
                    }
                    checked_boundaries += 2;
                }
            }
        }
        println!(
            "\n#film-camera corpus: {nfilms} film(s), {nbodies} bodies with overrides, {nsegs} \
             segment(s), {checked_boundaries} boundary lookups checked"
        );
        for (k, n) in &by_kind {
            println!("  {k:<14} {n}");
        }
        println!(
            "  vehicle-seat segments whose SEAT the recording states: {seat_stated} of {}",
            by_kind.get("vehicle seat").copied().unwrap_or(0)
        );
        assert!(nfilms > 0, "no film produced a decodable index");
        assert!(nsegs > 0, "not one film produced a single camera override -- the resolver is inert");
    }

    /// **THE END-TO-END GATE.** Step across a real death on a real film and assert that the
    /// camera `solve_camera_at` actually produces changes mode and ends up framing the corpse.
    ///
    /// This is a different assertion from every other gate here, and the distinction is the whole
    /// point: the others check the TIMELINE, which is a table. A table can be perfectly correct
    /// while the camera never moves, because the pose is produced by a different function that
    /// can refuse before it ever consults the table. Reported behaviour — "on death it does not
    /// change to the third-person spectate" — was exactly that, and no structural gate could have
    /// seen it.
    ///
    /// It also prints the diagnosis, because the reachability of the dead window is the quantity
    /// that decides whether the camera can work at all: a body's `last_tick` is its last
    /// replicated MESSAGE, which a dead body stops sending almost at once, while the dead and
    /// orbiting segments run on until the recording DESTROYS the corpse.
    #[test]
    fn the_camera_actually_moves_to_the_corpse_on_death() {
        use crate::film_play::{FilmCamera, Playback};
        let films = crate::film::find_films();
        if films.is_empty() {
            eprintln!("no films on this machine; the end-to-end gate is vacuous here");
            return;
        }
        // The counters. `tested` is the well-posed set: a death where the camera, asked for that
        // player, actually lands on THAT body. A player who has already respawned by the tick
        // under test is a different question and is counted separately rather than asserted on —
        // the first cut of this gate conflated the two and reported failures that were really its
        // own ambiguity.
        let (mut deaths, mut tested, mut no_pose, mut third_person, mut on_body) = (0, 0, 0, 0, 0);
        let (mut other_body, mut orbit_tested, mut orbit_ok) = (0, 0, 0);
        let mut unreachable: Vec<i64> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        let mut capturable: Vec<String> = Vec::new();
        for p in films.iter().take(8) {
            let Ok(f) = crate::film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            if idx.tracks.is_empty() {
                continue;
            }
            let mut pb = Playback::new(idx);
            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let stem = stem.rsplit('_').next().unwrap_or(&stem).to_string();
            // Every (owner, body, death) the film states.
            let cases: Vec<(u8, u32, u32)> = pb
                .index
                .tracks
                .iter()
                .filter_map(|t| {
                    let o = t.owner_raw?;
                    let d = pb.death_at(t)?;
                    Some((o, t.net_id, d))
                })
                .collect();
            for (owner, net, died) in cases {
                deaths += 1;
                pb.camera = FilmCamera::FirstPersonOwner(owner);
                let tl_end = pb
                    .camera_timeline()
                    .segments(net)
                    .iter()
                    .map(|s| s.to_tick)
                    .max()
                    .unwrap_or(died);
                let last = pb.index.tracks.iter().find(|t| t.net_id == net).map_or(died, |t| t.last_tick);
                unreachable.push(tl_end as i64 - last as i64);
                // A tick squarely inside the DEAD segment of THIS body.
                let Some(seg) = pb
                    .camera_timeline()
                    .segments(net)
                    .iter()
                    .find(|s| matches!(s.detail, ContextDetail::Dead { .. }))
                    .cloned()
                else {
                    continue;
                };
                // The LAST tick of the dead window, not its middle.
                //
                // This is where the bug lived and the midpoint barely touched it. A body keeps
                // replicating for a moment after its death tick, so halfway through the dead
                // segment most bodies are still inside their own `last_tick` and the old
                // `solve_plain_camera_at(..)?` had not yet begun to refuse. Measured: with the
                // bug deliberately restored, sampling the MIDPOINT caught it on only 13 of 1,170
                // deaths -- it failed, but by a margin thin enough to be luck. The end of the
                // window is past `last_tick` for far more of them, which is the whole point:
                // probe where the failure is, not where the feature is.
                let t_dead = seg.to_tick.saturating_sub(1).max(seg.from_tick);
                pb.tick = t_dead as f32;
                // Well-posed only if the camera, asked for this player, lands on this body.
                match crate::film_play::camera_subject_or_corpse(&pb, t_dead) {
                    Some((s, _)) if s == net => {}
                    _ => {
                        other_body += 1;
                        continue;
                    }
                }
                tested += 1;
                // A capturable triple, so a shot can be aimed at a death that is KNOWN reachable
                // instead of hunting for one through the app. Printed for the first few only.
                if capturable.len() < 8 {
                    capturable.push(format!(
                        "{stem}  film camera followbody {owner}  /  film seek {:.2}   (body 0x{net:08X}, dead {}..{})",
                        t_dead as f32 / TICK_HZ,
                        seg.from_tick,
                        seg.to_tick
                    ));
                }
                let Some(pose) = crate::film_play::solve_camera_at(&pb, crate::film_play::EYE_HEIGHT_WU)
                else {
                    no_pose += 1;
                    if failures.len() < 6 {
                        failures.push(format!(
                            "{stem} 0x{net:08X}: NO POSE at tick {t_dead}, inside its own dead \
                             segment {}..{} (died {died}, last message {last})",
                            seg.from_tick, seg.to_tick
                        ));
                    }
                    continue;
                };
                // Third person means a pivot, and the pivot must be ON this body.
                let Some(pivot) = pose.pivot else {
                    if failures.len() < 6 {
                        failures.push(format!(
                            "{stem} 0x{net:08X}: pose at tick {t_dead} has NO PIVOT -- still a \
                             first-person view, not the spectate"
                        ));
                    }
                    continue;
                };
                third_person += 1;
                let tr = pb.index.tracks.iter().find(|t| t.net_id == net).expect("the body");
                if let Some(body) = tr.sample(t_dead) {
                    let d = ((pivot[0] - body[0]).powi(2) + (pivot[1] - body[1]).powi(2)).sqrt();
                    if d < 0.3 {
                        on_body += 1;
                    } else if failures.len() < 6 {
                        failures.push(format!(
                            "{stem} 0x{net:08X}: pivot is {d:.2} wu from the body horizontally"
                        ));
                    }
                }
                // And deeper still, the ORBITING segment, where the engine settles.
                if let Some(orb) = pb
                    .camera_timeline()
                    .segments(net)
                    .iter()
                    .find(|s| matches!(s.detail, ContextDetail::Orbiting { .. }))
                    .cloned()
                {
                    let t_orb = orb.from_tick + (orb.to_tick - orb.from_tick) / 2;
                    pb.tick = t_orb as f32;
                    if matches!(crate::film_play::camera_subject_or_corpse(&pb, t_orb), Some((s, _)) if s == net) {
                        orbit_tested += 1;
                        if let Some(op) = crate::film_play::solve_camera_at(&pb, crate::film_play::EYE_HEIGHT_WU) {
                            if let (Some(piv), Some(body)) = (op.pivot, tr.sample(t_orb)) {
                                let dist = ((op.pos[0] - piv[0]).powi(2)
                                    + (op.pos[1] - piv[1]).powi(2)
                                    + (op.pos[2] - piv[2]).powi(2))
                                .sqrt();
                                let off = ((piv[0] - body[0]).powi(2) + (piv[1] - body[1]).powi(2)).sqrt();
                                // The engine's orbit: `orbit cam maximum distance` from a pivot
                                // `orbit cam z offset` above the body.
                                if off < 0.3 && (dist - ORBIT_CAM_MAX_DIST_WU).abs() < 0.3 {
                                    orbit_ok += 1;
                                } else if failures.len() < 6 {
                                    failures.push(format!(
                                        "{stem} 0x{net:08X}: orbit at tick {t_orb} is {dist:.2} wu \
                                         from a pivot {off:.2} wu off the body (want \
                                         {ORBIT_CAM_MAX_DIST_WU} and 0)"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        unreachable.sort_unstable();
        let med = unreachable.get(unreachable.len() / 2).copied().unwrap_or(0);
        println!(
            "\n#film-camera END-TO-END: {deaths} death(s); {tested} well-posed ({other_body} \
             skipped because that player had already respawned onto another body).\n  \
             of the {tested}: {no_pose} got NO pose, {third_person} went third person, \
             {on_body} framed the body.\n  \
             orbit: {orbit_ok} of {orbit_tested} sat at {ORBIT_CAM_MAX_DIST_WU} wu on the body.\n  \
             median dead-window ticks PAST the body's last replicated message: {med} ({:.1} s) \
             -- this is the camera the old `solve_plain_camera_at(..)?` could not reach at all.",
            med as f32 / TICK_HZ
        );
        for e in &failures {
            println!("    {e}");
        }
        println!("  capturable deaths (film / verbs / body):");
        for c in &capturable {
            println!("    {c}");
        }
        // A gate that measures NOTHING must fail, not pass — that is the whole difference between
        // this and the structural gates that were green while the camera never moved. But the
        // message has to say WHERE to look, because the commonest way for this to go to zero is
        // not a camera fault at all: `cases` is built from `t.owner_raw?`, and `owner_raw` is
        // written in exactly one place (`film_decode.rs`, the owner-stamp walk) behind a guard
        // that refuses any creation payload which did not walk to its end. When that walk stops
        // closing, every body loses its owner, this gate sees no deaths, and the fault is two
        // modules away. `FilmIndex::creations_closed` is the number that tells the two apart.
        assert!(
            deaths > 0,
            "no film produced a death to watch, so this gate measured NOTHING. Before suspecting \
             the camera, check `FilmIndex::creations_closed`: it is the share of biped creation \
             payloads whose walk reached the end, and `owner_raw` -- which this gate selects on -- \
             is only written for those. At 0% every body is ownerless and no death is visible here, \
             whatever the camera does. `film_decode`'s own \
             `the_walked_owner_groups_behave_like_players_on_every_real_film` reports that figure."
        );
        assert!(tested > 0, "no death was well-posed; the gate measured nothing");
        assert_eq!(
            no_pose, 0,
            "the camera solver refused inside {no_pose} body's OWN dead segment -- a dying \
             player's view stops being solved, so it can never reach the spectate"
        );
        assert_eq!(
            third_person, tested,
            "only {third_person} of {tested} deaths produced a third-person pose"
        );
        assert_eq!(on_body, tested, "only {on_body} of {tested} framed the body itself");
        assert_eq!(
            orbit_ok, orbit_tested,
            "only {orbit_ok} of {orbit_tested} orbits sat where the engine's camera block says"
        );
    }

    /// **The view must stop being first person when a context camera takes it out**, and the
    /// three things that key off that must agree: the subject's body stops being hidden, the
    /// first-person arms stop being drawn, and the arms' armour stops being installed.
    ///
    /// All three asked `FilmCamera::is_first_person`, which reports the viewer's chosen MODE and
    /// stays true right through a death. The visible result was a spectate camera orbiting an
    /// INVISIBLE corpse with a pair of floating hands in front of it — on screen that is
    /// indistinguishable from the camera never having moved, which is exactly how it was
    /// reported.
    #[test]
    fn a_context_camera_takes_the_view_out_of_first_person() {
        use crate::film_play::{FilmCamera, Playback};
        let films = crate::film::find_films();
        if films.is_empty() {
            eprintln!("no films on this machine; this gate is vacuous here");
            return;
        }
        let mut checked = 0usize;
        for p in films.iter().take(6) {
            let Ok(f) = crate::film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            if idx.tracks.is_empty() {
                continue;
            }
            let mut pb = Playback::new(idx);
            let cases: Vec<(u8, u32)> = pb
                .index
                .tracks
                .iter()
                .filter_map(|t| Some((t.owner_raw?, t.net_id)))
                .collect();
            for (owner, net) in cases {
                let Some(seg) = pb
                    .camera_timeline()
                    .segments(net)
                    .iter()
                    .find(|s| matches!(s.detail, ContextDetail::Dead { .. } | ContextDetail::Orbiting { .. }))
                    .cloned()
                else {
                    continue;
                };
                let t = seg.from_tick + (seg.to_tick - seg.from_tick) / 2;
                pb.camera = FilmCamera::FirstPersonOwner(owner);
                pb.tick = t as f32;
                if !matches!(crate::film_play::camera_subject_or_corpse(&pb, t), Some((s, _)) if s == net) {
                    continue;
                }
                checked += 1;
                pb.context_cameras = true;
                assert!(
                    !pb.view_is_first_person(),
                    "0x{net:08X} tick {t}: the {} camera still reports a first-person VIEW",
                    seg.mode.name()
                );
                assert_eq!(
                    pb.hidden_entity(),
                    None,
                    "0x{net:08X} tick {t}: the {} camera is orbiting this body and would HIDE it",
                    seg.mode.name()
                );
                // And with the context cameras off, the plain first-person view is unchanged —
                // the toggle has to leave the old behaviour exactly alone.
                pb.context_cameras = false;
                assert!(
                    pb.view_is_first_person(),
                    "0x{net:08X}: with context cameras OFF the view must still be first person"
                );
                if checked > 40 {
                    break;
                }
            }
            if checked > 40 {
                break;
            }
        }
        println!("#film-camera  {checked} dead/orbiting moment(s) checked for a first-person leak");
        assert!(checked > 0, "no dead or orbiting moment was reachable; the gate measured nothing");
    }

    /// **The named-film probe.** Print every camera transition the resolver produces on the
    /// films the camera work is meant to show, so the claim "the assassination camera fires here"
    /// can be checked against tick numbers rather than taken on trust.
    ///
    /// A probe rather than a gate: which films exist is a property of this machine, so asserting
    /// on their contents would make the suite fail somewhere else for no reason. The gate that
    /// bites on every machine is `camera_timelines_are_well_formed_on_every_real_film`.
    #[test]
    fn probe_camera_transitions_on_the_named_films() {
        // `find_films`, not `find_finished_films`: the densest assassination clip on this machine
        // is a `.temp` autosave, and a `.temp` is byte-identical in container format to a `.mov`
        // (same `_blf` header, same `chdr`). `find_finished_films` excludes it on purpose,
        // because the container GATES assert invariants that only hold once a film is finalised
        // -- but nothing here reads the container, only the decoded stream, so the exclusion
        // would cost this probe its best data for no reason.
        let films = crate::film::find_films();
        for p in &films {
            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            // The clips the round is aimed at, by the ids the task named. Matched on the id alone
            // so the user's own name, which is in the path, never reaches this output.
            if !["21897D8A", "53C14E9D", "4832BB46", "8917571E"].iter().any(|k| stem.contains(k)) {
                continue;
            }
            let Ok(f) = crate::film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let ticks = idx.ticks;
            let pb = Playback::new(idx);
            let tl = pb.camera_timeline();
            let mut n: std::collections::BTreeMap<&str, usize> = Default::default();
            let mut sync_examples: Vec<String> = Vec::new();
            let mut seat_examples: Vec<String> = Vec::new();
            let (mut seat_stated, mut seat_total) = (0usize, 0usize);
            for (net, segs) in &tl.per_body {
                for s in segs {
                    let k = match &s.detail {
                        ContextDetail::Plain => "plain",
                        ContextDetail::VehicleSeat { vehicle, seat, .. } => {
                            seat_total += 1;
                            if seat.is_some() {
                                seat_stated += 1;
                            }
                            if seat_examples.len() < 6 {
                                seat_examples.push(format!(
                                    "body 0x{net:08X} in 0x{vehicle:08X} seat {} from tick {} to {} ({:.1} s)",
                                    seat.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                                    s.from_tick,
                                    s.to_tick,
                                    (s.to_tick - s.from_tick) as f32 / TICK_HZ
                                ));
                            }
                            "vehicle seat"
                        }
                        ContextDetail::SyncAction { attacker, partner, .. } => {
                            if sync_examples.len() < 12 {
                                sync_examples.push(format!(
                                    "body 0x{net:08X} is the {} of 0x{:08X}, ticks {}..{}",
                                    if *attacker { "ATTACKER" } else { "victim  " },
                                    partner.unwrap_or(0),
                                    s.from_tick,
                                    s.to_tick
                                ));
                            }
                            "sync action"
                        }
                        ContextDetail::WeaponThirdPerson { .. } => "weapon 3rd person",
                        ContextDetail::Dead { .. } => "dead",
                        ContextDetail::Orbiting { .. } => "orbiting",
                        ContextDetail::Respawning { .. } => "respawning",
                    };
                    *n.entry(k).or_default() += 1;
                }
            }
            println!(
                "\n=== {} === {:.1} s, {} tracks, {} bodies with camera overrides",
                stem.rsplit('_').next().unwrap_or(&stem),
                ticks as f32 / TICK_HZ,
                pb.index.tracks.len(),
                tl.per_body.len()
            );
            for (k, c) in &n {
                println!("    {k:<14} {c}");
            }
            if seat_total > 0 {
                println!("    seat index STATED on {seat_stated} of {seat_total} vehicle segments");
                for e in &seat_examples {
                    println!("      {e}");
                }
            }
            for e in &sync_examples {
                println!("      {e}");
            }
            // And walk one real sync action tick by tick, so the TRANSITION itself is on the
            // record: which mode, how far into the blend, at what tick.
            if let Some((net, seg)) = tl.per_body.iter().find_map(|(net, segs)| {
                segs.iter().find(|s| matches!(s.detail, ContextDetail::SyncAction { .. })).map(|s| (*net, s))
            }) {
                println!("    the first sync action on body 0x{net:08X}, tick by tick:");
                let span = (seg.to_tick - seg.from_tick) as i64;
                for d in [-30i64, -1, 0, 5, 16, 33, 60, span / 2, span - 1, span, span + 30] {
                    let t = (seg.from_tick as i64 + d).max(0) as u32;
                    let c = resolve(tl, EngineCameraMode::FirstPerson, net, t);
                    println!(
                        "      tick {t:>7} ({d:+5}) mode {} {:<14} blend {:>5.1}%  {:?}",
                        c.mode.index(),
                        c.mode.name(),
                        c.blend * 100.0,
                        c.detail
                    );
                }
            }
        }
    }

    /// The seat must beat the weapon where both apply, because that is the engine's own order:
    /// `unit_should_be_third_person` resolves the PARENT path first and only falls through to the
    /// weapon when the unit has no parent. A gunner holding a turret is framed by their seat.
    #[test]
    fn a_vehicle_seat_beats_a_third_person_weapon() {
        let out = resolve_overlaps(vec![
            seg(100, 400, ContextDetail::WeaponThirdPerson { weapon: 0x55 }, 0.0),
            seg(200, 300, ContextDetail::VehicleSeat { vehicle: 9, seat: Some(1), slaved: false }, 0.0),
        ]);
        let t = timeline(out);
        assert!(matches!(t.at(1, 150).map(|s| &s.detail), Some(ContextDetail::WeaponThirdPerson { .. })));
        assert!(
            matches!(t.at(1, 250).map(|s| &s.detail), Some(ContextDetail::VehicleSeat { .. })),
            "in a seat the SEAT decides, got {:?}",
            t.at(1, 250)
        );
        assert!(
            matches!(t.at(1, 350).map(|s| &s.detail), Some(ContextDetail::WeaponThirdPerson { .. })),
            "and the weapon resumes on dismount, got {:?}",
            t.at(1, 350)
        );
        // A sync action still beats both.
        let out2 = resolve_overlaps(vec![
            seg(100, 400, ContextDetail::WeaponThirdPerson { weapon: 0x55 }, 0.0),
            seg(200, 300, ContextDetail::VehicleSeat { vehicle: 9, seat: Some(1), slaved: false }, 0.0),
            seg(220, 260, ContextDetail::SyncAction { attacker: true, partner: Some(2), started: 220 }, 0.0),
        ]);
        let t2 = timeline(out2);
        assert!(matches!(t2.at(1, 240).map(|s| &s.detail), Some(ContextDetail::SyncAction { .. })));
    }

    /// `decoded` and `pose_is_authored` must stay DIFFERENT questions. The death chain's shot is
    /// the engine's own geometry; the assassination's and the vehicle's are HMS's stand-ins for
    /// an authored camera a film cannot drive. A context that reported the second as authored
    /// would be claiming the game framed a shot HMS chose.
    #[test]
    fn a_decoded_mode_does_not_make_the_pose_authored() {
        let t = timeline(vec![
            seg(100, 200, ContextDetail::SyncAction { attacker: true, partner: Some(2), started: 100 }, 0.0),
            seg(200, 300, ContextDetail::VehicleSeat { vehicle: 9, seat: Some(0), slaved: false }, 0.0),
            seg(300, 400, ContextDetail::Dead { died_at: 300 }, 0.0),
            seg(400, 500, ContextDetail::Orbiting { died_at: 300 }, 0.0),
        ]);
        let at = |k| resolve(&t, EngineCameraMode::FirstPerson, 1, k);
        // All four have a decoded MODE -- the fixture says so.
        for k in [150u32, 250, 350, 450] {
            assert!(at(k).decoded, "tick {k}: the mode is decoded");
        }
        // Only the death chain has an authored POSE.
        assert!(!at(150).pose_is_authored, "the assassination shot is HMS's, not the engine's trak");
        assert!(!at(250).pose_is_authored, "the vehicle shot is HMS's, not the seat's camera marker");
        assert!(at(350).pose_is_authored, "the dead camera IS the globals camera block's geometry");
        assert!(at(450).pose_is_authored, "so is the orbit");
        // And the label says which, in words, for the two that are stand-ins.
        assert!(at(150).label().contains("POSE is HMS's"), "{}", at(150).label());
        assert!(!at(350).label().contains("POSE is HMS's"), "{}", at(350).label());
    }

    /// **Every camera constant, read back out of a shipped map.**
    ///
    /// This is the gate that makes the numbers above facts rather than comments. It opens
    /// `forge_halo.map`, checks the ANCHOR first — camera block element 0 `+0x24`
    /// `field of view [degrees]` must read 78.0, which is the field HMS already relied on and the
    /// one that pins the whole linear mapping `global(0x1818583A0 + k) <- camblk + 0x24 + k` — and
    /// then checks each constant against its own camera-block offset.
    ///
    /// If the anchor fails, the gate FAILS rather than skipping: a map that is present but whose
    /// camera block is not where this says it is means the offsets are wrong, and quietly passing
    /// would be exactly the "gate that cannot bite" this tree has been burned by before. Only a
    /// genuinely absent map file is a skip.
    #[test]
    fn the_camera_constants_are_what_the_map_authors() {
        const MAP: &str = "/mnt/990pro/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/maps/forge_halo.map";
        /// Camera block element 0's file offset in `forge_halo.map` — the address HMS's own
        /// `REACH_FIELD_OF_VIEW_DEG` note already records.
        const CAMBLK: u64 = 0xD42C60C;
        let Ok(mut f) = std::fs::File::open(MAP) else {
            eprintln!("{MAP} is not on this machine; the map-readback gate is vacuous here");
            return;
        };
        use std::io::{Read, Seek, SeekFrom};
        let mut buf = [0u8; 0xD8];
        f.seek(SeekFrom::Start(CAMBLK)).expect("seek");
        f.read_exact(&mut buf).expect("read the camera block");
        let at = |o: usize| f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);

        // THE ANCHOR. Everything else is only meaningful if this holds.
        assert!(
            (at(0x24) - 78.0).abs() < 1e-4,
            "camera block +0x24 reads {} not 78.0 -- the block is not at {CAMBLK:#X} and every \
             offset below is therefore unverified",
            at(0x24)
        );

        for (off, want, name) in [
            (0x03C, DEAD_CAMERA_TRANSITION_SECONDS, "dead camera transition time"),
            (0x044, DEAD_CAM_INITIAL_DIST_WU, "dead cam initial distance"),
            (0x048, DEAD_CAM_FINAL_DIST_WU, "dead cam final distance"),
            (0x04C, DEAD_CAM_Z_OFFSET_WU, "dead cam z offset"),
            (0x050, DEAD_CAM_MAX_ELEVATION_RAD, "dead cam maximum elevation"),
            (0x058, DEAD_TO_ORBIT_SECONDS, "time_to_auto_switch_to_orbiting"),
            (0x094, ORBIT_CAM_MIN_DIST_WU, "orbit cam minimum distance"),
            (0x098, ORBIT_CAM_MAX_DIST_WU, "orbit cam maximum distance"),
            (0x0A0, ORBIT_CAM_Z_OFFSET_WU, "orbit cam z offset"),
            (0x0A4, ORBIT_CAM_MIN_ELEVATION_RAD, "orbit cam minimum elevation"),
            (0x0A8, ORBIT_CAM_MAX_ELEVATION_RAD, "orbit cam maximum elevation"),
            (0x0B8, ENTER_VEHICLE_TRANSITION_SECONDS, "enter vehicle transition time"),
            (0x0BC, EXIT_VEHICLE_TRANSITION_SECONDS, "exit vehicle transition time"),
        ] {
            let got = at(off);
            assert!(
                (got - want).abs() < 1e-4,
                "camera block +{off:#05X} `{name}` reads {got} but the constant says {want}"
            );
        }

        // And the biped's own camera heights, which the first-person camera stands on. Same
        // method, different tag: `objects\characters\spartans\spartans` meta in this map.
        const SPARTAN_BIPD: u64 = 0x12C5FE5C;
        let mut bb = [0u8; 4];
        f.seek(SeekFrom::Start(SPARTAN_BIPD + 0x524)).expect("seek");
        f.read_exact(&mut bb).expect("read");
        let standing = f32::from_le_bytes(bb);
        assert!(
            (standing - crate::film_play::EYE_HEIGHT_WU).abs() < 1e-5,
            "bipd +0x524 `standing camera height` reads {standing}, not {}",
            crate::film_play::EYE_HEIGHT_WU
        );
        // The SYNC ACTION CAMERA exists and the Spartan authors a camera track in it. This is the
        // structural half of the honest claim in `context_pose`: the engine's assassination shot
        // IS an authored `trak`, so HMS's orbit is a stand-in and not a reconstruction.
        let mut sb = [0u8; 0x78];
        f.seek(SeekFrom::Start(SPARTAN_BIPD + 0x210)).expect("seek");
        f.read_exact(&mut sb).expect("read");
        let tracks = i32::from_le_bytes([sb[0x14], sb[0x15], sb[0x16], sb[0x17]]);
        // #film-camera  THE FIELD OF VIEW, and why no context camera changes it.
        //
        // A `unit_camera_struct` carries its own `override fov` at `+0x38`, and the unit authors
        // TWO of them -- `unit camera` (`bipd +0x198`) and `sync action camera` (`bipd +0x210`).
        // If either were non-zero, the assassination camera would render at a different field of
        // view from the ordinary one and HMS would have to follow. Both read **0.0** for the
        // Spartan, as does the biped's own `override fov` (`bipd +0x1D0`, checked above), so
        // every mode in the death/assassination/vehicle chain falls through to the globals
        // `field of view` -- the 78 degrees HMS already renders at. The FOV is therefore
        // CONSTANT across these cameras as a decoded fact, not as an omission, and this gate is
        // what keeps that claim honest if a map ever authors one.
        let mut ub = [0u8; 0x78];
        f.seek(SeekFrom::Start(SPARTAN_BIPD + 0x198)).expect("seek");
        f.read_exact(&mut ub).expect("read");
        for (base, label, buf) in
            [(0x198u64, "unit camera", &ub), (0x210u64, "sync action camera", &sb)]
        {
            let fov = f32::from_le_bytes([buf[0x38], buf[0x39], buf[0x3A], buf[0x3B]]);
            assert_eq!(
                fov, 0.0,
                "bipd +{base:#X} `{label}` -> `override fov` reads {fov}, not 0 -- this camera \
                 renders at its OWN field of view and the context cameras can no longer be \
                 treated as sharing the globals 78 degrees"
            );
        }
        assert_eq!(
            tracks, 1,
            "bipd +0x210 `sync action camera` -> `camera tracks` should hold exactly one authored \
             track for the Spartan, got {tracks}"
        );
        println!(
            "#film-camera map readback OK: fov {}, dead {}->{} wu over {} s, orbit {} wu, \
             switch at {} s, enter-vehicle {} s, spartan eye {} wu, sync-action camera tracks {}",
            at(0x24), at(0x44), at(0x48), at(0x3C), at(0x98), at(0x58), at(0xB8), standing, tracks
        );
    }

    /// The setting round-trips through a directory, like the other two Theater settings.
    #[test]
    fn the_context_camera_setting_round_trips_and_defaults_on() {
        let d = std::env::temp_dir().join(format!("hms-ctxcam-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        assert!(load_context_from_dir(&d), "default is ON: it is what the game does");
        save_context_to_dir(&d, false);
        assert!(!load_context_from_dir(&d));
        save_context_to_dir(&d, true);
        assert!(load_context_from_dir(&d));
        let _ = std::fs::remove_dir_all(&d);
    }
}
