//! Playing a Reach saved film: the clock, the scrub, and what to draw at a given moment.
//!
//! The expensive part (decoding the whole film into entity tracks) happens once in
//! `film_decode::build_index` and takes well under a second even on the longest film measured, so
//! this is a pure state machine over an index that is already in memory. There is no progressive
//! streaming and no disk cache because the measurement ruled both out.
//!
//! HONESTY RULES baked in here:
//! * only entity types whose payload actually decodes are offered for drawing;
//! * a biped is placed at its RECORDED position and facing, and its animation is NOT driven by
//!   the film (the stream's animation state is not decoded), so the UI must say "idle pose";
//! * entities whose type HMS cannot place are counted and reported, never guessed at.

use crate::film_decode::{EntityTrack, FilmIndex};
use std::collections::HashMap;

/// Reach simulates, and films record, at 60 ticks per second.
pub const TICK_HZ: f32 = 60.0;

/// One entity to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub net_id: u32,
    pub entity_type: u8,
    pub pos: [f32; 3],
    /// Ground speed at this moment, world units per second (clamped; a respawn teleport is not
    /// a sprint). Drives which locomotion animation is chosen.
    pub speed: f32,
    /// Distance travelled along the track so far, which drives the cycle's phase.
    pub distance: f32,
    /// Degrees about +Z, the `Actor::yaw` convention (0 = +X).
    pub yaw: f32,
    /// False when the track has no orientation yet, so `yaw` is a carried-over default.
    pub has_yaw: bool,
    /// #death `Some(ticks since the body died)` when this is a CORPSE rather than a live body.
    /// See `film_anim::death_tick` — the film states where a body's track ends, never that it
    /// died or how, so the fall itself is synthesised and the caption says so.
    pub dead_ticks: Option<u32>,
}

/// #melee / #fire  Which kind of ONE-SHOT clip a body is playing: a death fall, a two-party
/// synchronised action, or a melee swing. All three run once from a start tick and then hold the
/// last frame, which is the opposite of a cycle; the code only differs in what it counts.
pub const ONESHOT_DEATH: u8 = 0;
pub const ONESHOT_SYNC: u8 = 1;
pub const ONESHOT_MELEE: u8 = 2;
/// #jump  The LANDING, `combat:<class>:land_hard` — a body that has just touched down. Same
/// one-shot rule as the three above: driven by the age of the touchdown, holding the last frame
/// until the window closes and the body goes back to its locomotion cycle.
pub const ONESHOT_LAND: u8 = 3;
/// #jump-loop  The AIRBORNE clip, `combat:<class>:airborne_arc`. 45 frames against a jump that
/// lasts as long as the body is in the air, so it must advance once and HOLD: driven from the
/// locomotion phase (which wraps) it replayed from the start over and over, which is "the
/// airborne animation is looping, almost like a player is jumping repeatedly".
pub const ONESHOT_AIR: u8 = 4;

/// #fp-shake  A clock-driven clip's FRAME at a playhead position, in ticks.
///
/// Reach's animations are 30 fps against the film's 60 Hz ticks, so a frame is half a tick.
///
/// **The playhead must be passed as the float it is.** `Playback::tick` is advanced by real frame
/// time, and rounding it to a whole tick before computing a frame makes the animation step at the
/// TICK rate rather than the display rate — so above 60 Hz, or below 1x speed, the clip holds for
/// some rendered frames and jumps on others. Measured on the rounded form: at 144 Hz the frame
/// advances `0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5`, and at 0.25x `0, 0, 0.5, 0, 0, 0.5`; off the
/// float it advances 0.208 and 0.125 every frame. That judder is "the FP model arms shake or have
/// some sort of waviness", and it is invisible at exactly 60 Hz 1x, which is why it survived.
///
/// Still a pure function of the playhead, so scrubbing to a position reproduces the pose.
pub fn clock_frame(playhead_ticks: f32, frames: f32) -> f32 {
    let n = frames.max(1.0);
    (playhead_ticks.max(0.0) * 0.5 % n).max(0.0)
}

/// Which ONE-SHOT clip, if any, owns a body's whole skeleton this tick — and with what age.
///
/// A one-shot is driven by an age in ticks and HOLDS its last frame; everything else is driven by
/// the wrapping locomotion phase. Getting a clip onto the wrong one of those two lanes is the
/// shape of two user-visible defects at once, so the precedence is a function rather than a match
/// buried in `sync`:
///
/// * `sync` — a two-party assassination owns both parties outright and beats even death: the
///   victim plays its half first and only lies down afterwards (`#film-action`).
/// * `dead` — the death fall, counted from the landing (`#corpse-fall`).
/// * `land` — `combat:<class>:land_hard`, 13 frames from touchdown (`#jump`).
/// * `air`  — `combat:<class>:airborne_arc`, 45 frames from leaving the ground (`#jump-loop`).
///   LAST, so the landing takes over the moment it fires: the two overlap for a few ticks because
///   `velocity_at` is smoothed, and the landing is the one that should win there. That ordering is
///   what makes the sequence airborne -> land -> gait run without a restart in the middle.
///
/// A melee swing is deliberately NOT here: it is a Replacement clip animating 34 of 82 nodes, so
/// it rides the replacement lane over the locomotion clip instead of owning the skeleton.
pub fn oneshot_for(
    sync: Option<u32>,
    dead: Option<u32>,
    land: Option<u32>,
    air: Option<u32>,
) -> Option<(u32, u8)> {
    if let Some(age) = sync { return Some((age, ONESHOT_SYNC)); }
    if let Some(age) = dead { return Some((age, ONESHOT_DEATH)); }
    if let Some(age) = land { return Some((age, ONESHOT_LAND)); }
    air.map(|age| (age, ONESHOT_AIR))
}

/// #fire  How long a firing overlay runs, in film ticks. The shipped `fire_1` clips are 10-19
/// frames at 30 fps; the clip's own length clamps it.
pub const FIRE_TICKS: u32 = 40;

/// #melee  How long a swing is held, in film ticks. One melee clip: the shipped strikes are
/// 24-33 frames at 30 fps against the film's 60 Hz tick line.
pub const MELEE_TICKS: u32 = 60;

/// Which entity types this build knows how to place. Type 11 is the unit/biped — the one that
/// matters for watching a match. The others decode positions too and are offered separately so
/// the user can see the whole picture without HMS pretending to know what they are.
pub const BIPED_TYPE: u8 = 11;

/// #corpse-fall  How hard this engine pulls a falling body down, **wu/s²**.
///
/// **Fitted off the recordings themselves**, not converted from Earth gravity and a world-unit
/// guess. `film_anim::probes::probe_gravity_from_falling_bodies` takes every consecutive triple
/// of reported positions on a biped track where the body is unmistakably unsupported — vertical
/// speed strictly decreasing and past 1 wu/s, no horizontal speed that says respawn teleport,
/// samples within 20 ticks — and reads the slope of vz against time. Over **1,447 such triples
/// on 12 films the median is 10.4576 wu/s², with the whole interquartile range landing on two
/// discrete values**, which is what a constant looks like through a quantised position stream.
///
/// #gravity  **One definition, in [`crate::reach_physics`].** This is an alias so that the
/// ragdolls, test mode's jump and the editor's settle cannot drift apart again — the codebase
/// held three unrelated gravities differing by up to 3.25x, two of them Earth's and one a
/// digit-transposition of the other. The number is unchanged, so a corpse falls exactly as it
/// did.
pub const CORPSE_GRAVITY: f32 = crate::reach_physics::rigid_body_fall_acceleration();

/// #corpse-fall  A body within this of the surface under it is already lying on it, wu.
/// The measured floor noise is ±0.05 wu (the position stream's own quantisation).
pub const GROUND_EPS: f32 = 0.06;

/// #corpse-fall  How far below a death HMS will look for a floor, wu. Past this the body went
/// over the edge of the map and there is nothing honest to land it on, so it is left where the
/// recording last put it.
pub const GROUND_REACH: f32 = 60.0;

/// #corpse-fall  Where a body that went down at `pos` comes to rest, and how long it takes.
///
/// Returns `None` when the body is already on the floor (so nothing moves) or when no floor was
/// found within [`GROUND_REACH`]. Otherwise `(resting world Z, ticks of free fall)`.
pub fn corpse_fall(pos: [f32; 3], floor_z: Option<f32>) -> Option<(f32, u32)> {
    let z = floor_z?;
    let drop = pos[2] - z;
    if !(drop > GROUND_EPS) {
        return None;
    }
    // t = sqrt(2h/g), in 60 Hz ticks. At least one tick so the division downstream is safe.
    let secs = (2.0 * drop / CORPSE_GRAVITY).sqrt();
    Some((z, ((secs * 60.0).ceil() as u32).max(1)))
}

/// #corpse-fall  One dead body's whole recorded afterlife — see [`Playback::corpse_traces`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorpseTrace {
    pub net_id: u32,
    /// The tick the body goes down (the stated kill, or the derived moment where there is none).
    pub death_tick: u32,
    /// Where a corpse used to be drawn: the body's position at `death_tick`.
    pub at_death: [f32; 3],
    /// The LAST position the recording reported for this body, and when.
    pub last_tick: u32,
    pub last_pos: [f32; 3],
    /// How many positions the recording reported AFTER the body went down. These are the engine
    /// replicating its own ragdoll, and they are the honest answer to where the body came to
    /// rest — drawing the corpse at `at_death` throws every one of them away.
    pub reports_after: usize,
    /// How far the body fell between the two, wu (positive = downwards).
    pub fell: f32,
    /// The body was off the ground when it went down (decoded vertical speed).
    pub airborne: bool,
}

/// The playback clock over a decoded film.
#[derive(Clone, Debug)]
pub struct Playback {
    pub index: FilmIndex,
    /// #film-action  The decoded assassinations, folded ONCE. `sync_bodies` runs on the draw
    /// path, and re-scanning a 47-minute film's 50,000 events every frame is exactly the kind of
    /// per-frame walk this module has already been rescued from once.
    sync: std::cell::OnceCell<Vec<crate::film_anim::Assassination>>,
    /// #death `net id -> (the tick an incident named this body as the party killed, the
    /// incident, the killer)`. Folded ONCE, for the same reason `sync` is: `corpses()` used to
    /// call `film_anim::death_from_events` for every track on every frame, which is one pass over
    /// the whole event channel per track per frame.
    deaths: std::cell::OnceCell<std::collections::HashMap<u32, (u32, &'static str, Option<u32>)>>,
    /// Current position in ticks (fractional, so playback is smooth between the 60 Hz keys).
    pub tick: f32,
    pub playing: bool,
    /// 0.25 .. 4.0; 1.0 is real time.
    pub speed: f32,
    /// Draw non-biped moving entities as well.
    pub show_all_types: bool,
    /// Only draw entities that actually move (the static Forge scene is already drawn from the
    /// map variant, so redrawing 600 motionless objects on top of it is pure noise).
    pub moving_only: bool,
    /// Where the viewer is watching from.
    pub camera: FilmCamera,
    /// #nametag-mode  How much of the floating-name display is on: always, only over the body the
    /// viewer is LOOKING AT (the engine's own second name display, gated by the held weapon's
    /// authored autoaim cone), or nothing. Persisted by
    /// [`crate::reach_hud::load_name_tag_mode`] / [`crate::reach_hud::save_name_tag_mode`].
    pub name_tags: crate::reach_hud::NameTagMode,
    /// #spectate  Draw Theater's own spectated-player card while the camera is on one body.
    /// Persisted by [`load_spectate_card_setting`] / [`save_spectate_card_setting`].
    pub spectate_card: bool,
    /// #crosshair  Draw the held weapon's own authored reticule in first person.
    pub crosshair: bool,
    /// #reach-hud  Draw the authored first-person HUD (the weapon's ammo block, the grenade
    /// counts, the shield and health meters, and a seated player's vehicle HUD).
    pub hud: bool,
    /// #respawn  Draw the respawn countdown over a body that is waiting to come back.
    pub respawn_widget: bool,
    /// Where the follow camera sits relative to its subject. Driven by the mouse in the viewport
    /// and consumed by `solve_camera`.
    pub rig: FollowRig,
    /// The subject the rig was last set up for, so changing subject recenters it rather than
    /// leaving the camera pointing at where the last one was.
    pub rig_subject: Option<u8>,
    /// #film-camera  Follow the engine into its CONTEXT CAMERAS — the assassination camera, the
    /// vehicle camera, the third-person spectate on death. Default on, because it is what the
    /// game does; the toggle exists because a viewer lining up a shot wants the camera to stay
    /// where they put it. Persisted by [`crate::film_camera::save_context_setting`].
    pub context_cameras: bool,
    /// #film-camera  Every body's camera overrides, folded ONCE. The same reason `sync` and
    /// `deaths` are `OnceCell`s: building it walks the whole event channel and every track, and
    /// the draw path asks for it every frame.
    timeline: std::cell::OnceCell<crate::film_camera::CameraTimeline>,
    /// #film-camera  The tag-authored bits the camera resolver needs (seat flags, which weapons
    /// force third person), filled in by [`ActorDriver`] once a map is open. Film-constant, so it
    /// does not disturb the resolver's purity.
    ///
    /// **Private, and written only through [`Playback::set_camera_tags`]**, which drops the
    /// cached timeline. `timeline` is derived from this field, and this tree has already been
    /// bitten once by two writers sharing one field: a `pub` here would let a map opened after
    /// the first frame leave a timeline built from the empty fallbacks in place for ever.
    camera_tags: crate::film_camera::CameraTags,
    /// Per configuration slot: does that player's recorded AIM actually move?
    ///
    /// Measured once when the film is opened, because the answer is startling and it decides
    /// where a first-person camera gets its look direction. **Only ONE slot per film has a live
    /// aim — the player who recorded it.** On two 14- and 16-player clips every other slot
    /// carries the *same* `yaw_q` on all ~25,000 of its samples with the pitch exactly level,
    /// i.e. the field is zero-filled for everyone but the recorder, whose own input is captured
    /// locally. Trusting `aim_at` for those players pointed first person at a fixed angle
    /// forever, which is what "first person doesn't follow the look vector at all" was.
    pub aim_live: [bool; 16],
}

/// Theater's camera modes, over a decoded film.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilmCamera {
    /// HMS's ordinary free camera; playback does not touch it.
    #[default]
    Free,
    /// Through a player's eyes, aimed along that player's recorded AIM angles. Keyed by the
    /// film header's configuration SLOT, so it only works for a player the header names.
    FirstPerson(u8),
    /// Behind and above a player's biped, looking along their aim.
    Follow(u8),
    /// Through the eyes of an OWNER GROUP — one player's bodies across every respawn, named or
    /// not.
    ///
    /// This is what makes every body on the map watchable. A film's header roster is who was
    /// present when the header was written, so on one measured clip it names 3 players while 14
    /// owner groups had a body; keying the camera on the slot made 11 of those 14 unwatchable for
    /// no better reason than that the recording never stored their gamertag. Aiming a camera does
    /// not need a name.
    FirstPersonOwner(u8),
    /// Behind and above an owner group's current body.
    FollowOwner(u8),
}

impl FilmCamera {
    /// The owner group this mode is watching, resolving a slot through its binding.
    pub fn owner(self, idx: &FilmIndex) -> Option<u8> {
        match self {
            FilmCamera::Free => None,
            FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => Some(o),
            FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => idx.owner_of_slot(s),
        }
    }
    pub fn is_first_person(self) -> bool {
        matches!(self, FilmCamera::FirstPerson(_) | FilmCamera::FirstPersonOwner(_))
    }
}

/// Where the camera should be this tick.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPose {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    /// The point on the subject the camera is looking at, when it is orbiting one. The caller
    /// casts from here out to the camera to keep the view out of walls — from the subject, so a
    /// blocked shot is pulled IN along the same line and stays framed on the player.
    pub pivot: Option<[f32; 3]>,
    /// What to tell the user about this pose, including when it is NOT available.
    pub label: String,
}

/// A SPARTAN's first-person camera height above its feet, world units (1 wu = 10 feet). The stream
/// gives the biped's ORIGIN, which sits at the feet, so a first-person camera has to be lifted.
///
/// **The engine's own number**, not a guess any more: `bipd` +0x524 `standing camera height
/// [world units]`, which reads `9A 99 19 3F` = 0.600000024 for
/// `objects\characters\spartans\spartans` in `forge_halo.map` (meta at 0x12C5FE5C, so the field is
/// at file offset 0x12C60380; identical in `m05`, `20_sword_slayer` and `52_ivory_tower`). The
/// engine adds it to the biped origin along the biped's up axis and scales it by the object's
/// scale -- `biped_calculate_camera_position`, `bipeds.cpp:1750` -- and reads it through
/// `biped_get_camera_height`, which interpolates between +0x524 and +0x52C `crouching camera
/// height` by the crouch bit.
///
/// HMS used 0.55, which was 0.05 wu (about 15 cm) low.
///
/// Measured and deliberately NOT applied: the CROUCHING heights (Spartan 0.42, Elite 0.51),
/// because a Reach film does not record the crouch bit at all; the per-weapon-class `camera
/// heights` block at +0x54C, whose only non-identity row is the `finishing_move_kill` one
/// (0.65/0.75); and `camera offset` at +0x560 (Spartan forward 0.13, down 0.05), which the engine
/// applies as the view pitches down and whose code was not traced.
pub const EYE_HEIGHT_WU: f32 = 0.60;

/// An ELITE's first-person camera height — the Elites are taller. `bipd` +0x524 of
/// `objects\characters\elite\elite` reads `D7 A3 30 3F` = 0.689999998 (`forge_halo.map` meta at
/// 0x12F853A0, field at 0x12F858C4). See [`EYE_HEIGHT_WU`] for the engine path.
pub const ELITE_EYE_HEIGHT_WU: f32 = 0.69;

/// #fp-fov  The authored field of view Reach plays at, DEGREES and HORIZONTAL.
///
/// `globals` -> `camera` block (`matg` +0x3FC) -> +0x024 `real field of view [degrees]`. Read out
/// of `forge_halo.map`'s own `globals\globals` (tag 5): the camera block's first element sits at
/// file offset 0xD42C60C and +0x024 holds `00 00 9C 42` = **78.0**. The same 78.0 is in `m05.map`
/// and `20_sword_slayer.map`, and `haloreach.dll`'s invalid-FOV fallback is the matching
/// `1.36135685f` = 78.000 degrees. MCC's own FOV slider defaults to the same number and is stored
/// in whole degrees; a slider value of 0 means "unset" and the engine keeps this tag value.
///
/// The player BIPEDS author no override: `bipd` +0x1D0 `override fov` ("if non-zero, overrides the
/// FOV set in the unit or globals") reads `00 00 00 00` for both
/// `objects\characters\spartans\spartans` and `objects\characters\elite\elite`.
pub const REACH_FIELD_OF_VIEW_DEG: f32 = 78.0;

/// #fp-fov  The aspect the authored FOV is referenced to — **hardcoded 4:3 in the engine**, not
/// the window's. `observer_update` writes `0x3FAAAAAB` (1.3333334f) into the observer's
/// `aspect_ratio` on the straight-line path (`haloreach_debug.dll+0x74EA81`), and the view-matrix
/// builder repeats the literal `1.33333337`.
pub const REACH_FOV_REFERENCE_ASPECT: f32 = 4.0 / 3.0;

/// #fp-fov  The factor the render camera applies to the observer's FOV (`render_cameras.cpp`,
/// `haloreach.dll+0x287DFC`): `camera->vertical_field_of_view = observer->vertical_field_of_view *
/// 0.785`, the literal at `.rdata haloreach.dll+0xA8AE44` (`C3 F5 48 3F`).
pub const REACH_FOV_SCALE: f32 = 0.785;

/// #fp-fov  Reach draws the first-person MODEL through its OWN camera, at 88% of the world
/// camera's vertical FOV — i.e. zoomed IN relative to the world.
///
/// `render_first_person_view.cpp:61` (`haloreach.dll+0x286C6C`) copies the world camera into the
/// first-person pass and then does `scale = 0.690800011 / fp_camera->vfov_scale; fp_camera->vfov
/// *= scale`, publishing `scale` to the debug global `render_first_person_fov_scale`
/// (`haloreach.dll+0xB408E0` names `0x180B445F4`, and that is its only writer). With the ordinary
/// `vfov_scale` of 0.785 the product is exactly `0.690800011 / 0.785 = 0.88`.
///
/// **HMS DOES do this, and this comment used to say it did not.** The stale sentence cost a wrong
/// hypothesis: asked why a user's first-person model looks "wavy" at their MCC field of view of
/// 120, two agents in a row read this paragraph and concluded the viewmodel was being drawn
/// through the world projection. It is not. `hms-render` has a second camera buffer and bind
/// group (`SceneRenderer::fp_camera_buf` / `fp_camera_bg`), `render` writes it every frame as the
/// world camera with `fov_y *= fp_fov_scale`, the five `fp_*` mesh lanes are drawn through it,
/// and the scale is installed by `SceneRenderer::set_first_person_fov_scale` from
/// [`first_person_fov_scale`] at **three** call sites — `main.rs` (test mode entry and the
/// window's own refresh), `headless.rs` and `script_host.rs`. Gated by
/// [`the_first_person_rig_is_drawn_narrower_than_the_world`], which also records why that gate
/// cannot reach the renderer itself.
///
/// What the factor buys, for the record: the rig comes out `tan(43.205/2) / tan(49.097/2)` =
/// **13.3% larger** than the world projection would draw it at the authored 78 degrees, and
/// 16.4% at 120. Nothing else about the framing moves — both projections share the view and
/// differ only by a scale of clip-space x and y. Doing it by scaling the rig instead WOULD shear
/// its normals, because the difference is an anisotropic scale in view space and not a uniform
/// one, which is why it is a second projection rather than a transform.
pub const REACH_FIRST_PERSON_FOV_SCALE: f32 = 0.88;

/// #fp-fov The factor to draw the first-person RIG's own camera at. The engine's
/// [`REACH_FIRST_PERSON_FOV_SCALE`] unless `HMS_FP_FOV_SCALE` overrides it, which exists so the
/// before/after of this change is one environment variable rather than a rebuild: `=1` draws the
/// rig through the WORLD camera, which is what HMS did before and what makes the arms splay as
/// the FOV slider widens.
pub fn first_person_fov_scale() -> f32 {
    std::env::var("HMS_FP_FOV_SCALE")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.25..=4.0).contains(v))
        .unwrap_or(REACH_FIRST_PERSON_FOV_SCALE)
}

/// #fp-fov  The narrowest and widest field of view the control offers, in the engine's own units.
///
/// The engine itself accepts 1 to 150 degrees (`valid_field_of_view`, `haloreach.dll+0xCAFF4`,
/// which rejects anything below 0.0174532924 rad or above 2.61799383 and substitutes
/// `1.36135685f` = 78 degrees). MCC's own slider is much narrower than that; this keeps to a range
/// that stays a FIRST-PERSON view rather than a fish-eye.
pub const FOV_MIN_DEG: f32 = 60.0;
pub const FOV_MAX_DEG: f32 = 120.0;

/// #fp-fov  The persisted first-person field of view, in the SAME number MCC's own slider shows:
/// degrees of HORIZONTAL field of view referenced to 4:3, which is what the engine's
/// `observer->horizontal_field_of_view` is set from. Default [`REACH_FIELD_OF_VIEW_DEG`], i.e. the
/// value the globals tag authors, so the app starts out framing exactly like the game.
const FOV_SETTING_FILE: &str = "first_person_fov.txt";

/// #spectate  Whether Theater's spectated-player card is drawn, remembered between sessions the
/// same way the first-person FOV is. Default on: it is what the game shows while you spectate.
const SPECTATE_CARD_SETTING_FILE: &str = "spectate_card.txt";

pub fn load_spectate_card_setting() -> bool {
    crate::project::settings_dir().map(|d| load_spectate_card_from_dir(&d)).unwrap_or(true)
}

pub fn load_spectate_card_from_dir(dir: &std::path::Path) -> bool {
    match std::fs::read_to_string(dir.join(SPECTATE_CARD_SETTING_FILE)) {
        Ok(s) => !matches!(s.trim(), "0" | "off" | "false"),
        Err(_) => true,
    }
}

pub fn save_spectate_card_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() {
        save_spectate_card_to_dir(&dir, on);
    }
}

pub fn save_spectate_card_to_dir(dir: &std::path::Path, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(SPECTATE_CARD_SETTING_FILE), if on { "1" } else { "0" });
}

pub fn load_fov_setting() -> f32 {
    crate::project::settings_dir().map(|d| load_fov_from_dir(&d)).unwrap_or(REACH_FIELD_OF_VIEW_DEG)
}

pub fn load_fov_from_dir(dir: &std::path::Path) -> f32 {
    std::fs::read_to_string(dir.join(FOV_SETTING_FILE))
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| (FOV_MIN_DEG..=FOV_MAX_DEG).contains(v))
        .unwrap_or(REACH_FIELD_OF_VIEW_DEG)
}

pub fn save_fov_setting(deg: f32) {
    if let Some(dir) = crate::project::settings_dir() {
        save_fov_to_dir(&dir, deg);
    }
}

pub fn save_fov_to_dir(dir: &std::path::Path, deg: f32) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(FOV_SETTING_FILE), format!("{:.1}", deg.clamp(FOV_MIN_DEG, FOV_MAX_DEG)));
}

/// #fp-fov  Reach's VERTICAL field of view, radians — what a first-person Theater camera renders
/// at. **49.097 degrees** (78.153 degrees horizontal at 16:9).
///
/// Reach is **Hor+**: the vertical FOV is derived ONCE from the authored horizontal FOV at the
/// hardcoded 4:3 reference and then never changes with the window, and the projection grows the
/// horizontal with the real aspect. `observer.cpp:2289` (`sub_18074FEE0`) is the derivation —
/// `vfov = 2*atan(tan(hfov/2) / aspect_ratio)` — and `render_cameras.cpp:580`
/// (`haloreach.dll` `sub_1800E7D60`) is the projection: `y_scale = 1/(tan(vfov/2) * …)`,
/// `x_scale = 1/(aspect * tan(vfov/2) * …)`. `hms_render::Camera::view_proj` holds `fov_y` and
/// derives the horizontal from the aspect in exactly the same way, so this is a drop-in.
///
/// HMS's own default camera is 70 degrees VERTICAL (about 102 horizontal at 16:9) — far wider than
/// the game, which is why a first-person Theater shot never looked like one.
pub fn reach_vertical_fov() -> f32 {
    reach_vertical_fov_for(REACH_FIELD_OF_VIEW_DEG)
}

/// #fp-fov  [`reach_vertical_fov`] for any authored/slider horizontal FOV in degrees — the engine's
/// own chain, so the number the user types is the number MCC's slider would mean.
pub fn reach_vertical_fov_for(horizontal_deg: f32) -> f32 {
    let h = horizontal_deg.clamp(FOV_MIN_DEG, FOV_MAX_DEG).to_radians();
    2.0 * ((h * 0.5).tan() / REACH_FOV_REFERENCE_ASPECT).atan() * REACH_FOV_SCALE
}

/// #fp-fov  The vertical FOV Reach draws the first-person MODEL at, radians — **43.205 degrees**.
/// See [`REACH_FIRST_PERSON_FOV_SCALE`]; recorded for the record, not yet applied.
pub fn reach_first_person_model_fov() -> f32 {
    reach_first_person_model_fov_for(REACH_FIELD_OF_VIEW_DEG)
}

/// #fp-fov  The first-person MODEL pass's vertical FOV for a given world FOV.
pub fn reach_first_person_model_fov_for(horizontal_deg: f32) -> f32 {
    reach_vertical_fov_for(horizontal_deg) * REACH_FIRST_PERSON_FOV_SCALE
}

/// The point on the body the follow camera orbits and looks at: roughly chest height above the
/// feet.
pub const FOLLOW_PIVOT_WU: f32 = 0.40;
/// Default distance from that pivot, world units.
pub const FOLLOW_DIST_WU: f32 = 2.4;
/// Default elevation above the pivot, degrees. Small on purpose.
///
/// The old rig sat at `EYE_HEIGHT + FOLLOW_UP` = **1.10 wu above the feet and only 1.4 wu back**.
/// A Spartan is about 0.7 wu tall, so that was ~0.4 wu above his head looking down about 28
/// degrees — it read as hovering over him rather than standing behind him. At 2.3 wu back and 6
/// degrees of elevation the camera sits ~0.69 wu above the feet, just under head height, which is
/// a conventional over-the-shoulder framing.
pub const FOLLOW_ELEV_DEG: f32 = 3.6;
/// Orbit distance limits, world units: closer than this is inside the body, further is a dot in
/// the distance.
pub const FOLLOW_DIST_MIN: f32 = 1.0;
pub const FOLLOW_DIST_MAX: f32 = 8.0;
/// Orbit pitch limit, degrees — short of vertical so the view cannot flip over the top.
pub const FOLLOW_PITCH_LIMIT: f32 = 80.0;

/// Deprecated: kept so nothing outside this module breaks on the name.
#[deprecated(note = "the follow rig is FOLLOW_PIVOT_WU / FOLLOW_DIST_WU / FOLLOW_ELEV_DEG")]
#[allow(dead_code)]
pub const FOLLOW_BACK_WU: f32 = 1.4;
#[deprecated(note = "the follow rig is FOLLOW_PIVOT_WU / FOLLOW_DIST_WU / FOLLOW_ELEV_DEG")]
#[allow(dead_code)]
pub const FOLLOW_UP_WU: f32 = 0.55;

/// Where the follow camera is, relative to the subject — the orbit the viewer drives with the
/// mouse.
///
/// It lives HERE, on the playback, and not on `App::camera`, because `App::film_tick` overwrites
/// the app camera's position, yaw and pitch from `solve_camera` every frame while a film camera
/// mode is active. Ordinary mouse-look on the app camera would therefore be wiped on the next
/// frame; the orbit has to be state that `solve_camera` CONSUMES.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollowRig {
    /// Yaw offset from the subject's own facing, radians. 0 = directly behind them.
    pub yaw_off: f32,
    /// Elevation above the pivot, radians. Positive looks down at the subject.
    pub elev: f32,
    /// Distance from the pivot, world units.
    pub dist: f32,
}

impl Default for FollowRig {
    fn default() -> Self {
        FollowRig { yaw_off: 0.0, elev: FOLLOW_ELEV_DEG.to_radians(), dist: FOLLOW_DIST_WU }
    }
}

impl FollowRig {
    /// Apply a mouse drag, in radians.
    pub fn drag(&mut self, dyaw: f32, delev: f32) {
        self.yaw_off = wrap_pi(self.yaw_off + dyaw);
        let lim = FOLLOW_PITCH_LIMIT.to_radians();
        self.elev = (self.elev + delev).clamp(-lim, lim);
    }
    /// Apply a scroll-wheel zoom.
    pub fn zoom(&mut self, delta: f32) {
        self.dist = (self.dist * (1.0 - delta * 0.1)).clamp(FOLLOW_DIST_MIN, FOLLOW_DIST_MAX);
    }
    /// Put the camera back behind the subject at the default framing.
    pub fn recenter(&mut self) {
        *self = FollowRig::default();
    }
}

fn wrap_pi(mut a: f32) -> f32 {
    while a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    while a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

impl Playback {
    pub fn new(index: FilmIndex) -> Playback {
        let aim_live = Self::aim_live_of(&index);
        Playback {
            index,
            sync: std::cell::OnceCell::new(),
            deaths: std::cell::OnceCell::new(),
            tick: 0.0,
            playing: false,
            speed: 1.0,
            show_all_types: false,
            moving_only: true,
            camera: FilmCamera::Free,
            name_tags: crate::reach_hud::load_name_tag_mode(),
            spectate_card: load_spectate_card_setting(),
            crosshair: crate::reach_hud::load_crosshair_setting(),
            hud: crate::reach_hud::load_hud_setting(),
            respawn_widget: crate::reach_hud::load_respawn_setting(),
            rig: FollowRig::default(),
            rig_subject: None,
            context_cameras: crate::film_camera::load_context_setting(),
            timeline: std::cell::OnceCell::new(),
            camera_tags: Default::default(),
            aim_live,
        }
    }

    /// Which slots have a RECORDED look direction, from the aim samples alone.
    ///
    /// Factored out because [`Playback::replace_index`] has to answer the same question again
    /// when a live recording grows, and two copies of this would drift.
    fn aim_live_of(index: &FilmIndex) -> [bool; 16] {
        // One pass over the aim samples: a slot whose quantised yaw never changes has no recorded
        // look direction, however many samples it has.
        let mut aim_live = [false; 16];
        for (slot, v) in index.player_aims.iter().enumerate().take(16) {
            if v.len() < 30 {
                continue;
            }
            let first = v[0].1.yaw_q;
            // A handful of stray changes is noise, not a player looking around; require a real
            // share of the samples to differ.
            let changed = v.iter().filter(|(_, a)| a.yaw_q != first).count();
            aim_live[slot] = changed * 20 > v.len();
        }
        aim_live
    }

    /// #film-live  Swap in a freshly decoded index WITHOUT disturbing the view.
    ///
    /// For following a recording that is still being written: MCC writes the match to a `.temp`
    /// in its autosave folder, that file is byte-identical in format to a finished `.mov`, and
    /// this reader already frames the stream forward and stops at the last complete record. So
    /// re-decoding a grown file and transplanting the result is all that "live" needs.
    ///
    /// Everything the viewer chose is kept BY CONSTRUCTION -- the camera, the follow rig, the
    /// playhead, the speed, the name-tag mode, the HUD and crosshair toggles -- because this
    /// replaces only the index and the caches derived FROM the index, rather than rebuilding a
    /// `Playback` and copying fields back. Copying fields back is what would rot: every setting
    /// added later would have to be remembered here, and the one that was forgotten would
    /// silently reset itself every two seconds.
    ///
    /// The playhead is clamped rather than moved, so a viewer parked in the past stays parked and
    /// one sitting at the live edge keeps advancing.
    pub fn replace_index(&mut self, index: FilmIndex) {
        self.aim_live = Self::aim_live_of(&index);
        self.index = index;
        // Every cache below is a pure function of the index, so all of them have to go.
        self.sync = std::cell::OnceCell::new();
        self.deaths = std::cell::OnceCell::new();
        self.timeline = std::cell::OnceCell::new();
        let end = self.total_ticks().saturating_sub(1) as f32;
        self.tick = self.tick.min(end).max(0.0);
    }

    /// #film-camera  Install the tag-authored camera bits, DROPPING the derived timeline so the
    /// next query rebuilds it against them. The only writer of `camera_tags`.
    pub fn set_camera_tags(&mut self, tags: crate::film_camera::CameraTags) {
        if self.camera_tags.seat_flags == tags.seat_flags
            && self.camera_tags.third_person_weapons == tags.third_person_weapons
        {
            return;
        }
        self.camera_tags = tags;
        self.timeline = std::cell::OnceCell::new();
    }

    pub fn camera_tags(&self) -> &crate::film_camera::CameraTags {
        &self.camera_tags
    }

    /// #film-camera  Every body's camera overrides for this film, built once.
    pub fn camera_timeline(&self) -> &crate::film_camera::CameraTimeline {
        self.timeline.get_or_init(|| crate::film_camera::build_timeline(self, &self.camera_tags))
    }

    /// #film-camera  **Is the view ACTUALLY inside the subject's head at this tick?**
    ///
    /// This is the question almost every "are we in first person" test in this module really
    /// wanted, and `FilmCamera::is_first_person` is not it. That one reports the viewer's chosen
    /// MODE, which stays `FirstPerson`/`FirstPersonOwner` all the way through a death — what
    /// changes is the engine camera the context resolves to. Three separate places asked the
    /// mode and got "yes" while the dead camera was orbiting the body from six feet away: the
    /// body was hidden, the first-person arms were drawn, and the arms' armour was installed.
    /// All three are invisible in the code and glaring on screen.
    pub fn view_is_first_person(&self) -> bool {
        if !self.camera.is_first_person() {
            return false;
        }
        if !self.context_cameras {
            return true;
        }
        self.camera_context()
            .map_or(true, |c| c.mode == crate::film_camera::EngineCameraMode::FirstPerson)
    }

    /// #film-camera  Which engine camera mode the viewer's current subject is in at this tick,
    /// and why. `None` in free camera or where the subject has no body.
    pub fn camera_context(&self) -> Option<crate::film_camera::CameraContext> {
        let tick = self.tick.round().max(0.0) as u32;
        // #film-camera  The SAME resolver the render path uses. They diverged once -- this
        // reported a `dead` context while `solve_camera_at` refused to produce a pose at all --
        // and a diagnostic that disagrees with the thing it is diagnosing is worse than none.
        let (subject, _) = camera_subject_or_corpse(self, tick)?;
        let preferred = if self.camera.is_first_person() {
            crate::film_camera::EngineCameraMode::FirstPerson
        } else {
            crate::film_camera::EngineCameraMode::Following
        };
        Some(crate::film_camera::resolve(self.camera_timeline(), preferred, subject, tick))
    }
    pub fn total_ticks(&self) -> u32 {
        self.index.ticks
    }
    pub fn total_seconds(&self) -> f32 {
        self.index.ticks as f32 / TICK_HZ
    }
    pub fn seconds(&self) -> f32 {
        self.tick / TICK_HZ
    }
    /// #blade-anim  The ANIMATION CLOCK a film frame must be rendered at: the playback position
    /// in seconds, and nothing else.
    ///
    /// Every animated material in the scene — the energy sword's scrolling plasma, a flashing
    /// Forge light, water, a shield's churning noise — is a function of this one number, so
    /// feeding it the wall clock makes the picture at a given tick depend on WHEN you looked at
    /// it: scrub away and back and the blade is at a different phase, and two captures of the
    /// same tick differ. Taken from the tick, a frame is a pure function of the playback
    /// position, which is the same rule the ragdolls, the death clips and the landing lane are
    /// already held to. It is the playback position rather than `tick.round()` so the animation
    /// stays smooth between ticks while playing.
    pub fn anim_seconds(&self) -> f32 {
        self.seconds()
    }
    pub fn seek_tick(&mut self, t: f32) {
        self.tick = t.clamp(0.0, self.total_ticks().saturating_sub(1) as f32);
    }
    pub fn seek_seconds(&mut self, s: f32) {
        self.seek_tick(s * TICK_HZ);
    }
    /// Step one tick (1/60 s) in either direction.
    pub fn step(&mut self, ticks: i32) {
        self.seek_tick(self.tick + ticks as f32);
    }
    /// Advance by real seconds. Stops (and clears `playing`) at the end rather than looping — a
    /// silent loop makes a short film look like it is still running.
    pub fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let end = self.total_ticks().saturating_sub(1) as f32;
        self.tick += dt * TICK_HZ * self.speed;
        if self.tick >= end {
            self.tick = end;
            self.playing = false;
        }
    }
    /// Does this track qualify for drawing under the current filters?
    ///
    /// **`moving_only` does NOT apply to a body**, and that exclusion is the fix for "some of the
    /// bipeds are not rendering, I see floating knives but not the full biped". The filter is
    /// there to keep the 370,000 static Forge tracks of a long clip out of the entity list, and
    /// it was being applied to players too: a body whose whole life moves it less than half a
    /// world unit -- a player shot the moment they spawned, which on a 0-second-respawn gametype
    /// is a great many of them -- was dropped from `frame()`, and therefore drawn neither alive
    /// nor as a corpse. **Its WEAPON was still drawn**, because the object lane has no such
    /// filter and a dropped sword is a `weap` track like any other, so what was left on screen
    /// was an energy sword standing where the body should have been. Measured on the 12:16 clip:
    /// 4 of its 64 bodies never move, and the caption did not even count them as missing, because
    /// it counts what `frame()` returned.
    ///
    /// A body is a body whether or not it walked anywhere, so the motion test now only decides
    /// the OTHER types, which is what it was written for.
    fn wanted(&self, t: &EntityTrack) -> bool {
        if t.entity_type == BIPED_TYPE {
            return true;
        }
        if self.moving_only && !t.moves() {
            return false;
        }
        self.show_all_types
    }
    /// Everything to draw at the current moment.
    pub fn frame(&self) -> Vec<Placement> {
        let tick = self.tick.round().max(0.0) as u32;
        let mut out = Vec::new();
        for t in &self.index.tracks {
            if !self.wanted(t) {
                continue;
            }
            if tick < t.first_tick || tick > t.last_tick {
                continue;
            }
            // #death A body whose life ENDED stops being drawn alive at the moment it died, not
            // at its last message. The two are a median 4 s apart (p95 22 s), and that gap is
            // exactly why a killed player used to stand frozen on the spot and then vanish.
            // `death_at` is the SAME moment the corpse starts from, so the two can never both
            // draw the same body.
            if self.death_at(t).is_some_and(|d| tick > d) {
                continue;
            }
            let Some(pos) = t.sample(tick) else { continue };
            // A biped reports its body yaw in its own field, not the generic orientation, so
            // prefer that; without it a player would be drawn facing +X for their whole life.
            let (yaw, has_yaw) = match t.facing_at(tick) {
                Some(y) => (y.to_degrees(), true),
                None => match t.sample_orientation(tick) {
                    Some((fwd, _up)) => (fwd[1].atan2(fwd[0]).to_degrees(), true),
                    None => (0.0, false),
                },
            };
            out.push(Placement {
                net_id: t.net_id,
                entity_type: t.entity_type,
                pos,
                yaw,
                has_yaw,
                speed: t.speed_at(tick).unwrap_or(0.0).clamp(0.0, 8.0),
                distance: t.distance_to(tick),
                dead_ticks: None,
            });
        }
        out
    }
    /// Bipeds to draw, capped — the actor pool is finite and a long film accumulates far more
    /// biped tracks (respawns) than are ever alive at once.
    pub fn bipeds(&self, cap: usize) -> Vec<Placement> {
        // In first person the camera sits inside the subject's head, so drawing their own body
        // fills most of the screen with the inside of their helmet. The engine hides the player's
        // own body in first person; so does this. The follow camera keeps it, because the whole
        // point of a follow camera is to watch that body.
        let hide = self.hidden_entity();
        let mut v: Vec<Placement> = self
            .frame()
            .into_iter()
            .filter(|p| p.entity_type == BIPED_TYPE)
            .filter(|p| Some(p.net_id) != hide)
            .collect();
        v.sort_by_key(|p| p.net_id);
        v.truncate(cap);
        v
    }

    /// #death The CORPSES at this moment: bodies that have died and that the recording has not
    /// yet DESPAWNED, placed where they fell. See [`Playback::corpse_window`].
    ///
    /// Capped, and the cap is separate from the living cap so a busy firefight can never push a
    /// live player off screen to make room for a body. Sorted by net id and then truncated, so
    /// which corpses make the cut is a pure function of the tick — scrubbing backwards restores
    /// exactly the same set.
    pub fn corpses(&self, cap: usize) -> Vec<Placement> {
        if cap == 0 {
            return Vec::new();
        }
        let tick = self.tick.round().max(0.0) as u32;
        let hide = self.hidden_entity();
        let mut out: Vec<Placement> = Vec::new();
        for t in &self.index.tracks {
            if t.entity_type != BIPED_TYPE || !self.wanted(t) || Some(t.net_id) == hide {
                continue;
            }
            // #death When this body goes down and when the recording DELETES it. Both come out
            // of `corpse_window`, so `frame()` and this agree on the moment of death and the
            // body lies there as long as the stream says it did.
            let Some((d, gone)) = self.corpse_window(t) else { continue };
            if tick <= d || tick > gone {
                continue;
            }
            // The stated moment can land after the body's last reported position, so take the
            // last position at or before it rather than demanding one exactly there.
            let Some(pos) = t.sample(d).or_else(|| t.positions.last().map(|&(_, p)| p)) else { continue };
            let (yaw, has_yaw) = match t.facing_at(d).or_else(|| t.facings.last().map(|&(_, y)| y)) {
                Some(y) => (y.to_degrees(), true),
                None => match t.sample_orientation(d) {
                    Some((fwd, _up)) => (fwd[1].atan2(fwd[0]).to_degrees(), true),
                    None => (0.0, false),
                },
            };
            out.push(Placement {
                net_id: t.net_id,
                entity_type: t.entity_type,
                pos,
                yaw,
                has_yaw,
                speed: 0.0,
                distance: t.distance_to(d),
                dead_ticks: Some(tick - d),
            });
        }
        out.sort_by_key(|p| p.net_id);
        out.truncate(cap);
        out
    }

    /// #corpse-fall  What the recording itself says about where a dead body ENDED UP.
    ///
    /// Print-only, and it exists because "the ragdolls float midair" has two possible causes that
    /// look identical on a still: the body is drawn at a position that was never on the ground,
    /// or the clip is played at the wrong height. This measures the first one directly.
    pub fn corpse_traces(&self) -> Vec<CorpseTrace> {
        let mut out = Vec::new();
        for t in &self.index.tracks {
            if t.entity_type != BIPED_TYPE {
                continue;
            }
            let Some((d, _gone)) = self.corpse_window(t) else { continue };
            let Some(at_death) = t.sample(d) else { continue };
            let Some(&(last_tick, last_pos)) = t.positions.last() else { continue };
            out.push(CorpseTrace {
                net_id: t.net_id,
                death_tick: d,
                at_death,
                last_tick,
                last_pos,
                reports_after: t.positions.iter().filter(|&&(k, _)| k > d).count(),
                fell: at_death[2] - last_pos[2],
                airborne: crate::film_anim::died_airborne(t, d),
            });
        }
        out.sort_by_key(|c| c.death_tick);
        out
    }

    /// #film-action  The decoded assassinations, cached for the life of this playback.
    pub fn assassinations(&self) -> &[crate::film_anim::Assassination] {
        self.sync.get_or_init(|| crate::film_anim::assassinations(&self.index))
    }

    /// #film-action  How long BEFORE the kill lands to park the playhead when jumping to one.
    ///
    /// The incident fires on the killing blow, so seeking straight to it starts the clip with the
    /// move already finishing. 45 ticks is 0.75 s at the recorded 60 Hz -- enough to see the
    /// attacker close the distance, short enough not to land before the previous kill.
    pub const ASSASSINATION_LEAD_TICKS: u32 = 45;

    /// #film-action  Jump to the next (`+1`) or previous (`-1`) assassination and FRAME it.
    ///
    /// The user's complaint was "the assassination animations still need to be shown properly,
    /// they are like too quick or something so you basically never see them". The rate turned out
    /// to be correct -- the clips are authored at 75 frames over the 150-tick window and HMS plays
    /// them at exactly that -- so the real fault was that there is no way to FIND one: a 7-minute
    /// clip holds three, each lasting 2.5 s, anywhere in 25,000 ticks, and most of the time the
    /// camera is pointing somewhere else entirely. Seeking alone is not enough for the same
    /// reason; the camera has to go with it.
    ///
    /// So this does both: parks the playhead [`ASSASSINATION_LEAD_TICKS`] before the kill and puts
    /// the camera on the VICTIM's owner group in third person, which is the angle that shows the
    /// move -- the attacker comes into frame behind them. First person would show the victim's own
    /// view, where the move happens off-camera behind their head.
    ///
    /// Returns the assassination it moved to, so the caller can say which one and whether the
    /// recording STATED it or HMS derived it. `None` means there are none in this film, or none
    /// further in that direction -- it does not wrap, because silently looping to the start of a
    /// 47-minute clip reads as the button having done nothing.
    pub fn jump_to_assassination(&mut self, dir: i32) -> Option<crate::film_anim::Assassination> {
        let here = self.tick.round().max(0.0) as u32;
        // Compare against the point each jump PARKS at, not the kill tick, so that pressing
        // "next" and then "previous" returns to where you were instead of to the same kill: after
        // a jump the playhead sits at `tick - lead`, which is still before the kill.
        let park = |a: &crate::film_anim::Assassination| a.tick.saturating_sub(Self::ASSASSINATION_LEAD_TICKS);
        let mut marks: Vec<crate::film_anim::Assassination> = self.assassinations().to_vec();
        marks.sort_by_key(|a| a.tick);
        let found = if dir >= 0 {
            marks.into_iter().find(|a| park(a) > here)
        } else {
            marks.into_iter().rev().find(|a| park(a) < here)
        }?;
        self.playing = false;
        self.seek_tick(park(&found) as f32);
        // Put the camera on the victim where the stream names one. A kill the stream gives no
        // victim for still gets the seek -- a jump that moves the playhead and leaves the camera
        // alone is worth more than no jump at all -- so this is deliberately not a failure.
        if let Some(owner) = found
            .victim
            .and_then(|v| self.index.tracks.iter().find(|t| t.net_id == v))
            .and_then(|t| t.owner_raw)
        {
            self.camera = FilmCamera::FollowOwner(owner);
        }
        Some(found)
    }

    /// #death Every body the event channel says was KILLED, with the tick, the incident and the
    /// killer. One pass over the events for the whole film; see the field's own note.
    pub fn deaths(&self) -> &std::collections::HashMap<u32, (u32, &'static str, Option<u32>)> {
        self.deaths.get_or_init(|| {
            let mut out: std::collections::HashMap<u32, (u32, &'static str, Option<u32>)> = Default::default();
            for e in &self.index.events {
                let Some(victim) = e.other else { continue };
                let Some(n) = e.incident_label() else { continue };
                if !(crate::film_action::KILL_INCIDENTS.contains(&n)
                    || matches!(n, "death" | "suicide" | "fell_to_death"))
                {
                    continue;
                }
                // The FIRST such incident: a body appears in later rows as a corpse (`avenged`).
                match out.get(&victim) {
                    Some(&(t, _, _)) if t <= e.tick => {}
                    _ => {
                        out.insert(victim, (e.tick, n, e.subject));
                    }
                }
            }
            out
        })
    }

    /// #death The ONE moment this body's life ended, used by every path that has to agree on it.
    ///
    /// `frame()` and `corpses()` both need it and they have to give the same answer: while they
    /// did not, a body whose stated kill landed before its last reported position was drawn ALIVE
    /// and as a CORPSE at the same tick, two actors on one net id.
    ///
    /// The rules, in order:
    /// * a track the recording never DESTROYED is a life that had not ended when the recording
    ///   stopped, so it has no death — whatever an incident says. (A recycled net id splices two
    ///   lives into one track, and that is the case this rule protects: `film_anim::death_tick`
    ///   rejects it outright, and an incident from the earlier life must not be allowed to bury
    ///   the later one.)
    /// * otherwise the moment is the recording's own statement — the `incident` that names this
    ///   body as the party killed — clamped to no later than the last position the body
    ///   reported, which is where it is drawn.
    pub fn death_at(&self, t: &EntityTrack) -> Option<u32> {
        let derived = crate::film_anim::death_tick(t)?;
        let stated = self
            .deaths()
            .get(&t.net_id)
            .map(|&(tk, _, _)| tk)
            .filter(|&tk| tk >= t.first_tick && tk <= t.last_tick);
        Some(stated.map_or(derived, |c| c.min(derived)))
    }

    /// #death The window a body lies on the ground for: `(the tick it goes down, the tick it
    /// goes away)`.
    ///
    /// **The despawn is in the recording.** The entity-DESTROY message is the corpse being
    /// deleted, not the death — it lands a median 974 ticks (16 s) after the body's last message
    /// — so a body should lie there from its death until that destroy, which is what the engine
    /// did and what the user sees in the game. HMS used to clear it after a flat 5 s, which is
    /// why "the dead players have their bipeds disappear".
    ///
    /// A victim of a decoded assassination goes down only when the two-party clip ENDS: it plays
    /// its half first (see `sync_bodies`), and without the offset it would be a corpse and
    /// mid-assassination at the same tick. The 5 s is kept as a FLOOR, so a body can never be
    /// swept away before its own death clip has finished.
    fn corpse_window(&self, t: &EntityTrack) -> Option<(u32, u32)> {
        let d0 = self.death_at(t)?;
        let assn = self
            .deaths()
            .get(&t.net_id)
            .is_some_and(|&(_, inc, _)| crate::film_action::ASSASSINATION_INCIDENTS.contains(&inc));
        let down = d0 + if assn { crate::film_anim::SYNC_TICKS } else { 0 };
        let gone = t
            .destroyed_tick
            .unwrap_or(t.last_tick)
            .max(down + crate::film_anim::CORPSE_TICKS);
        Some((down, gone))
    }

    /// #film-action  The bodies locked into a decoded assassination at this moment.
    ///
    /// They have to be a THIRD source alongside the living and the corpses, because the victim of
    /// an assassination typically stops replicating a position the moment the action starts — so
    /// `frame()` has already dropped it and the corpse window has not opened yet. Without this
    /// the victim simply vanished for the length of the performance, which is exactly the
    /// "the victim disappears" the user reported.
    ///
    /// Both parties are placed on the ATTACKER's recorded transform at the moment of the kill,
    /// which is the shared root both halves of the clip are authored around.
    pub fn sync_bodies(&self) -> Vec<Placement> {
        let tick = self.tick.round().max(0.0) as u32;
        let hide = self.hidden_entity();
        let mut out: Vec<Placement> = Vec::new();
        for a in self.assassinations() {
            if !a.covers(tick) {
                continue;
            }
            let root = a
                .attacker
                .and_then(|id| self.index.tracks.iter().find(|t| t.net_id == id))
                .and_then(|tr| {
                    let p = tr.sample(a.tick).or_else(|| tr.positions.last().map(|&(_, p)| p))?;
                    Some((p, tr.facing_at(a.tick).or_else(|| tr.facings.last().map(|&(_, y)| y))))
                });
            let Some((pos, yaw)) = root else { continue };
            for id in [a.attacker, a.victim].into_iter().flatten() {
                if Some(id) == hide {
                    continue;
                }
                let Some(tr) = self.index.tracks.iter().find(|t| t.net_id == id) else { continue };
                if !self.wanted(tr) {
                    continue;
                }
                out.push(Placement {
                    net_id: id,
                    entity_type: BIPED_TYPE,
                    pos,
                    yaw: yaw.map(|y| y.to_degrees()).unwrap_or(0.0),
                    has_yaw: yaw.is_some(),
                    speed: 0.0,
                    distance: 0.0,
                    dead_ticks: None,
                });
            }
        }
        out.sort_by_key(|p| p.net_id);
        out
    }

    /// Every body an actor has to be found for at this moment: the living, then the corpses.
    pub fn bodies(&self, cap: usize, corpse_cap: usize) -> Vec<Placement> {
        self.bodies_from(self.frame(), cap, corpse_cap)
    }

    /// #film-perf  The same, from a frame the caller has ALREADY computed.
    ///
    /// `frame()` walks every track in the index, and `sync` needs both the drawable bodies and
    /// the total count of bipeds this moment has — which used to be two separate `frame()` calls
    /// per frame of playback.
    pub fn bodies_from(&self, frame: Vec<Placement>, cap: usize, corpse_cap: usize) -> Vec<Placement> {
        let hide = self.hidden_entity();
        let mut v: Vec<Placement> = frame
            .into_iter()
            .filter(|p| p.entity_type == BIPED_TYPE)
            .filter(|p| Some(p.net_id) != hide)
            .collect();
        v.sort_by_key(|p| p.net_id);
        v.truncate(cap);
        v.extend(self.corpses(corpse_cap));
        // #film-action  The two parties of a decoded assassination, which neither of the other
        // two sources covers: the victim has stopped replicating and is not yet a corpse.
        // Deduplicated, with the sync placement winning — it is the one that puts the pair
        // together.
        let sync = self.sync_bodies();
        if !sync.is_empty() {
            let ids: std::collections::HashSet<u32> = sync.iter().map(|p| p.net_id).collect();
            v.retain(|p| !ids.contains(&p.net_id));
            v.extend(sync);
        }
        v
    }

    /// The entity that must NOT be drawn at this moment: the first-person camera's own body.
    ///
    /// #film-camera  **A context camera that has taken the view out of first person hides
    /// nothing**, and getting this wrong is invisible in the code and glaring on screen. The
    /// viewer's MODE is still `FirstPersonOwner` all the way through a death — what changes is the
    /// engine camera the context resolves to — so the old `is_first_person()` test went on hiding
    /// the subject's body while the dead and orbiting cameras were orbiting that exact body. The
    /// spectate shot would have framed an invisible corpse, which is indistinguishable on screen
    /// from the camera not having moved at all.
    ///
    /// The test is therefore on the resolved CAMERA, not on the viewer's chosen mode: a body is
    /// hidden only while the camera is genuinely inside its head.
    pub fn hidden_entity(&self) -> Option<u32> {
        if !self.view_is_first_person() {
            return None;
        }
        let tick = self.tick.round().max(0.0) as u32;
        match self.camera {
            FilmCamera::FirstPerson(slot) => self.index.entity_for_slot(slot, tick).map(|b| b.net_id),
            FilmCamera::FirstPersonOwner(o) => self.index.entity_for_owner(o, tick).map(|t| t.net_id),
            _ => None,
        }
    }
    /// #fp-arms  WHICH PLAYER the camera is riding, as a configuration slot.
    ///
    /// The body's own slot binding first, because that is the one the recording states; the
    /// CAMERA's own subject second, because the window's camera list is OWNER-keyed and an owner
    /// group whose bodies were never bound to a slot would otherwise answer "nobody". That second
    /// route is the whole fix for "the first-person arms do not wear my armour": the armour is
    /// looked up by slot, and a `None` here installed nothing at all and left the arms in the
    /// model's own default black.
    pub fn subject_slot(&self) -> Option<u8> {
        if let Some(s) = self.hidden_entity().and_then(|net| self.index.slot_of_entity(net)) {
            return Some(s);
        }
        match self.camera {
            FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => Some(s),
            FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => self.index.slot_of_owner(o),
            FilmCamera::Free => None,
        }
    }

    /// A one-line status for the timeline window.
    pub fn status(&self) -> String {
        let f = self.frame();
        let bipeds = f.iter().filter(|p| p.entity_type == BIPED_TYPE).count();
        format!(
            "{} / {}  —  {} entities here ({bipeds} bipeds)",
            fmt_time(self.seconds()),
            fmt_time(self.total_seconds()),
            f.len()
        )
    }
}

/// `m:ss.t`, the scale a Theater scrub works at.
pub fn fmt_time(s: f32) -> String {
    let s = s.max(0.0);
    format!("{}:{:04.1}", (s / 60.0) as u32, s % 60.0)
}

/// A fixed pool of actors reused across the film: assigning a different entity to an actor only
/// changes its placement, which needs NO mesh rebuild, whereas adding and removing actors does.
#[derive(Default, Debug)]
pub struct ActorPool {
    /// `actor id -> the net id it is currently showing`.
    pub assigned: HashMap<u32, u32>,
}

impl ActorPool {
    /// Match the pool's actors to this frame's bipeds. Returns, for each pool slot, the placement
    /// it should take (or `None` to park it out of sight). Pure, so the mapping is testable.
    pub fn assign(&mut self, slots: &[u32], bipeds: &[Placement]) -> Vec<(u32, Option<Placement>)> {
        // Keep an actor on the same entity across frames where possible, so a biped does not
        // teleport between actors (and so its pose stays with it once animation lands).
        let mut out: Vec<(u32, Option<Placement>)> = Vec::with_capacity(slots.len());
        let mut taken: Vec<bool> = vec![false; bipeds.len()];
        let mut keep: HashMap<u32, usize> = HashMap::new();
        for (i, b) in bipeds.iter().enumerate() {
            keep.insert(b.net_id, i);
        }
        // pass 1: stay with the same entity
        let mut chosen: HashMap<u32, usize> = HashMap::new();
        for &slot in slots {
            if let Some(prev) = self.assigned.get(&slot) {
                if let Some(&i) = keep.get(prev) {
                    if !taken[i] {
                        taken[i] = true;
                        chosen.insert(slot, i);
                    }
                }
            }
        }
        // pass 2: fill the rest in order
        let mut next = 0usize;
        for &slot in slots {
            if chosen.contains_key(&slot) {
                continue;
            }
            while next < bipeds.len() && taken[next] {
                next += 1;
            }
            if next < bipeds.len() {
                taken[next] = true;
                chosen.insert(slot, next);
                next += 1;
            }
        }
        self.assigned.clear();
        for &slot in slots {
            match chosen.get(&slot) {
                Some(&i) => {
                    self.assigned.insert(slot, bipeds[i].net_id);
                    out.push((slot, Some(bipeds[i])));
                }
                None => out.push((slot, None)),
            }
        }
        out
    }
}

/// The locomotion ladder, chosen from the MEASURED ground speed. The thresholds come from the
/// real distribution of player speeds over 254,414 samples in three films (median 2.64 wu/s,
/// p75 3.56, p90 3.91), not from guesswork.
///
/// This is SYNTHESISED motion: the film's own animation state is not decoded, so the cycle is
/// picked from the recorded speed rather than replayed. Aiming, crouching, jumping and weapon
/// state are not drawn at all.
pub fn locomotion_for(speed: f32) -> &'static str {
    match speed {
        s if s < 0.5 => "combat:rifle:idle",
        s if s < 1.8 => "combat:rifle:walk_front_slow",
        s if s < 3.2 => "combat:rifle:move_front_slow",
        _ => "combat:rifle:move_front_fast",
    }
}

/// **Retired.** One global frames-per-world-unit for every clip, chosen so a run at ~4 wu/s
/// played at about its authored rate. It could only ever be right for one gait: at 7.5 a walking
/// body's cycle advanced barely a third as fast as its feet travelled, which is the stiff,
/// gliding look this replaced. See `film_anim::cycles_per_wu`, which is per gait and measured.
#[deprecated(note = "use film_anim::cycle_phase, which is per gait")]
#[allow(dead_code)]
pub const CYCLE_FRAMES_PER_WU: f32 = 7.5;

/// #fp-arms One `@player representation` row: the two bare `mode` tags one species sees when it
/// is looking through its own eyes, and which species that is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FpRep {
    pub hands: u32,
    pub body: u32,
    /// True for the ELITE row. Decided the way `scene::elite_customization_regions` decides it
    /// everywhere else — the model carries `armor_pieces`, the region the globals' own
    /// `elite selections` block overrides — and never from the tag path.
    pub elite: bool,
}

/// #film-perf  The five phases of [`ActorDriver::sync`], as a rolling distribution.
///
/// Printed by `film perf`. Five `Instant::now()` calls per frame; see [`SYNC_NAMES`].
#[derive(Debug, Default)]
pub struct SyncProf {
    mark: Option<std::time::Instant>,
    cur: [f32; SYNC_STAGES],
    hist: Vec<[f32; SYNC_STAGES]>,
}

pub const SYNC_STAGES: usize = 5;
pub const SYNC_NAMES: [&str; SYNC_STAGES] = [
    "once-per-film tables",
    "frame() index walk",
    "vehicles",
    "objects (weapons/flags)",
    "bodies (pool + anim pick)",
];

impl SyncProf {
    fn begin(&mut self) {
        self.mark = Some(std::time::Instant::now());
        self.cur = [0.0; SYNC_STAGES];
    }
    fn lap(&mut self, k: usize) {
        let now = std::time::Instant::now();
        if let Some(m) = self.mark {
            if let Some(slot) = self.cur.get_mut(k) {
                *slot += (now - m).as_secs_f32() * 1000.0;
            }
        }
        self.mark = Some(now);
    }
    fn end(&mut self) {
        if self.hist.len() == crate::film_perf::WINDOW {
            self.hist.remove(0);
        }
        self.hist.push(self.cur);
    }
    pub fn clear(&mut self) {
        self.hist.clear();
    }
    /// One line per phase: mean, share of the whole `sync`, p99 and max.
    pub fn report(&self) -> String {
        if self.hist.is_empty() {
            return String::new();
        }
        let n = self.hist.len() as f32;
        let total: f32 = self.hist.iter().map(|r| r.iter().sum::<f32>()).sum::<f32>() / n;
        let mut out = format!("film sync ({} frames, mean {total:.3} ms) by phase:\n", self.hist.len());
        let mut rows: Vec<(f32, f32, f32, &str)> = (0..SYNC_STAGES)
            .map(|k| {
                let col: Vec<f32> = self.hist.iter().map(|r| r[k]).collect();
                let (m, _, _, p99, max) = crate::film_perf::stats(&col);
                (m, p99, max, SYNC_NAMES[k])
            })
            .collect();
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (m, p99, max, name) in rows {
            let share = if total > 0.0 { m / total * 100.0 } else { 0.0 };
            out.push_str(&format!("  {name:<28} {m:7.3} {share:5.1}% {p99:7.3} {max:7.3}\n"));
        }
        out
    }
}

/// Drives a pool of `reach_actor` actors from the playback clock.
///
/// Moving an actor is just `pos`/`yaw` — the bone palette folds the placement in — so a frame of
/// playback costs no mesh rebuild. Only CREATING the pool does, which happens once.
#[derive(Debug)]
pub struct ActorDriver {
    /// #film-perf  Where the time inside [`ActorDriver::sync`] goes, as a rolling distribution.
    ///
    /// `film perf` named `sync` as a third of a Theater frame (7.6 ms mean, 274 ms worst
    /// measured) but could not say which of its five phases that was, so it gets the same
    /// treatment the frame itself got: five laps and the distribution, not a mean.
    pub prof: SyncProf,
    pub pool: ActorPool,
    /// Actor ids in the pool, in creation order.
    pub slots: Vec<u32>,
    /// The `bipd` the pool is built from, as a path fragment matched against the cache's bipeds.
    /// The film does not yet tell us which biped an entity actually is, so every player is drawn
    /// with the same one and the UI says so.
    pub biped: String,
    /// Maximum simultaneous bipeds drawn (Reach's own player cap).
    pub cap: usize,
    /// #ragdoll  Pose dead bodies from the articulated solver. ON, because Reach's own death
    /// variety IS the ragdoll; off falls back to the one-shot death clip plus the vertical fall.
    pub ragdoll: bool,
    /// #ragdoll  Per `bipd` tag, the ragdoll read from its `phmo` (shared by every body of that
    /// species). `None` = that biped has no articulated physics model.
    ragdoll_defs: HashMap<u32, Option<std::sync::Arc<crate::ragdoll::Ragdoll>>>,
    /// #ragdoll  Per dead body, its simulation. A CACHE: the pose is a pure function of the seed
    /// and the tick, so clearing this changes nothing that is drawn.
    corpse_sims: HashMap<u32, CorpseSim>,
    /// #ragdoll  Bodies posed by the solver at the last sync.
    pub ragdolls_shown: usize,
    /// #ragdoll  The furthest any one body has been simulated, in ticks.
    pub ragdoll_steps: u64,
    /// #ragdoll-budget  Bodies that ran out of this frame's catch-up budget and will advance
    /// again next frame. Non-zero only just after a long seek.
    pub ragdoll_catching_up: usize,
    /// #ragdoll  Why a dead body could not be simulated, if one could not. Reported, never silent.
    pub ragdoll_missing: Option<String>,
    /// #ragdoll  Per simulated body at the last sync: `(net id, pelvis world position, age in
    /// ticks)`. Diagnostic only -- it is what lets a scripted capture aim a camera at a ragdoll
    /// instead of guessing which player to follow.
    pub ragdoll_bodies: Vec<(u32, [f32; 3], u32)>,
    /// #melee  `(nodes the last installed swing animates, nodes in the skeleton)`.
    pub melee_nodes: (usize, usize),
    /// #melee  Where each body swinging this frame IS, and the clip it is swinging. Diagnostic
    /// only, so a scripted capture can aim a camera at a swing instead of hunting for one.
    pub melee_bodies: Vec<([f32; 3], String)>,
    /// #ragdoll  Per placed-object datum, its physics hull as world triangles.
    ///
    /// `ObjectScene::physics_world_tris` resolves the `phmo` and reads its geometry through the
    /// native tag reader on EVERY call, and a ragdoll asks the world what is around it every tick
    /// -- so without this a dozen corpses near a Forge structure would re-decode every hull sixty
    /// times a second. A film's Forge scene never moves, so the memo is valid for the whole film
    /// and is dropped with it.
    obj_tris: HashMap<u32, std::sync::Arc<Vec<glam::Vec3>>>,
    /// Where an unused actor waits: far below the map, out of every shot.
    pub park: [f32; 3],
    /// Why the last sync could not place anything, if it could not.
    pub last_error: Option<String>,
    /// Drive a locomotion cycle from the recorded motion instead of standing still.
    pub animate: bool,
    /// `actor id -> the animation name currently loaded`, so `set_anim` (which loads a resource)
    /// only runs when the state actually changes.
    pub anim_of: HashMap<u32, String>,
    /// `net id -> track index`, built once per index so `sync` can reach a biped's TRACK (and so
    /// its decoded facing and recorded velocity) without scanning 300,000 tracks per frame.
    pub track_of: HashMap<u32, usize>,
    /// Every vehicle in the film, identified by the Forge object it spawned on, plus every ride.
    /// Built once, on the first sync after a film is indexed. See `film_anim::VehicleTable`.
    pub vehicles: Option<crate::film_anim::VehicleTable>,
    /// Draw the vehicles the film recorded.
    pub show_vehicles: bool,
    /// `vehicle net id -> actor id`. A vehicle's actor is created on demand and kept, because
    /// creating one rebuilds meshes while moving one does not.
    pub vehicle_actor: HashMap<u32, u32>,
    /// Vehicles whose model could not be placed, with the reason — reported, never silent.
    pub vehicle_error: Option<String>,
    /// #film-objects  Every OTHER object the film recorded — the flag, the dropped weapons, the
    /// debris — identified through its own 14-bit definition reference. Built once per
    /// (cache, film) by `resolve_objects`; `None` until a map is open to resolve against.
    pub objects: Option<crate::film_objective::ObjectTable>,
    /// #film-objects  Draw them. On by default: without it the Castle Wars clip the report came
    /// from shows no flag at all. The user asked for a toggle rather than being forced, so this
    /// is it, and it is the ONE switch for the whole lane.
    pub show_objects: bool,
    /// #film-objects  The most objects the stream may draw at one moment. An actor is a GPU mesh
    /// build, and a nine-minute clip accumulates 447 `weap` tracks, so there has to be a ceiling;
    /// the draw list is ordered objectives-first so a cap can never drop the flag.
    ///
    /// **96 is above the measured peak, not a guess at one.** Sampled across the Castle Wars clip
    /// (`film_objective::probes::probe_real_table_over_a_film`) the stream wants to draw 35 to 73
    /// objects at a time, peaking at 73; the first draft's 48 was below that and clipped six of
    /// the twelve sampled moments. `objects_capped` says when the ceiling bites, so a clip that
    /// exceeds it reports the fact instead of quietly dropping weapons.
    pub object_cap: usize,
    /// #film-objects  `object net id -> actor id`. Created on demand and kept, because creating
    /// an actor rebuilds meshes while moving one does not.
    pub object_actor: HashMap<u32, u32>,
    /// #film-objects  Objects drawn at the last sync, those of them that are objectives, and the
    /// ones left to the Forge scene because they are still on their own spawn point.
    pub objects_shown: usize,
    pub objectives_shown: usize,
    pub objects_on_spawn: usize,
    /// #film-objects  Objects the stream wanted to draw at the last sync but `object_cap` cut.
    pub objects_capped: usize,
    /// #film-vehid  Each DISTINCT definition reference a type-15 track states, resolved once to
    /// `(tag id, 4-char group, tag path)`.
    ///
    /// Filled by [`ActorDriver::resolve_objects`], which already holds the reference table, and
    /// consumed by [`crate::film_vehicle::identify_by_definition`] in `sync`, which is where the
    /// vehicle table exists. Two entries on the measured clip (`warthog`, `warthog_chaingun`) —
    /// it is keyed by the wire value precisely so it stays that small.
    pub vehicle_defs: HashMap<u32, (u32, String, String)>,
    /// #film-vehid  Every present object in the film's own map variant as
    /// `(resolved tag id, position)` — kept from `resolve_objects` so the vehicle lane can look
    /// for a Forge copy of its OWN definition instead of the nearest vehicle of any kind.
    pub variant_spawns: Vec<(u32, [f32; 3])>,
    /// #film-vehid  What the definition pass added to the vehicle table, for the caption.
    pub veh_by_def: crate::film_vehicle::DefIdentified,
    /// #film-vehid  Whether that pass has run for this film. It is a one-shot: the inputs are
    /// per-film and the result is an insert into the table.
    pub veh_by_def_done: bool,
    /// #film-objects  Objects whose model could not be placed, with the reason.
    pub object_error: Option<String>,
    /// #film-objects  What `resolve_objects` found, in words, for the caption.
    pub object_status: String,
    /// #film-objects  Guards `resolve_objects` so it runs once per (cache, film).
    object_cache: Option<u64>,
    /// #vehicle-damage  Per `vehi` object tag, the damage-response table read out of its model's
    /// `hlmt` (`None` = that model authors no damage states at all). Read ONCE per film; a tag
    /// walk, so never per frame.
    veh_responses: HashMap<u32, Option<crate::film_vehicle::DamageResponses>>,
    /// #vehicle-damage  Per vehicle net id, the recording's own damage history and the wreck
    /// window: `(timeline, wreck rule)`. Both are pure functions of the tick.
    veh_damage: HashMap<u32, (crate::film_vehicle::DamageTimeline, crate::film_vehicle::WreckRule)>,
    /// #vehicle-damage  Vehicles drawn in some damage state at the last sync, and of those, how
    /// many are wrecks.
    pub veh_damaged_now: usize,
    pub wrecks_now: usize,
    /// #vehicle-rot  Vehicles whose recorded basis is more than 5 degrees off level at the last
    /// sync — i.e. the ones the old yaw-only placement was drawing flat.
    pub veh_tilted: usize,
    /// #vehicle-damage  Whole-clip figures for the status line: how many vehicles the recording
    /// ever damaged, how many it destroyed, how many `damage_section_response` events named a
    /// vehicle, and how many vehicles have a model that authors damage states at all.
    pub veh_damage_totals: (usize, usize, usize, usize),
    /// #vehicle-damage  Set once the two tables above have been built for the open film.
    veh_damage_ready: bool,
    /// #film-camera  The tag-authored camera bits, decoded off the loaded map once per film:
    /// which seats of which vehicle force third person, and which weapons do. Handed to the
    /// playback by `main.rs`, which is the only writer of `Playback::camera_tags`.
    cam_tags: crate::film_camera::CameraTags,
    /// The map's `'weap'` tags in group order. A film names a unit's weapon as an INDEX into this
    /// list, so it has to be resolved once and kept.
    pub weap_tags: Vec<u32>,
    /// Why a weapon could not be placed, if one could not.
    pub weapon_error: Option<String>,
    /// Bodies drawn holding a weapon the recording named, at the last sync.
    pub weapons_shown: usize,
    /// #wclass The ANIMATION CLASS of each entry of `weap_tags`, in the same order — `rifle`,
    /// `pistol`, `missile`, `sword`, `hammer`, `ball`, `turret`, `flag`, `unarmed`, or `fixed`
    /// for the vehicle-mounted guns nobody carries. Read once per map off `weap` +0x3AC.
    pub weap_class: Vec<Option<String>>,
    /// #wclass Distinct animation classes seen, so `clip_frames` can be keyed on one cheaply.
    /// Index 0 is always [`crate::film_anim::DEFAULT_WEAPON_CLASS`].
    class_ids: Vec<String>,
    /// #anim-rate  The biped tag PATHS seen, so `clip_frames` can be keyed on the graph as
    /// cheaply as on the class. Index 0 is the default pool's biped.
    graph_ids: Vec<String>,
    /// #wclass Bodies drawn on their weapon's OWN animation class at the last sync, and bodies
    /// whose class the graph had no clip for so the generic set was used instead.
    pub class_bodies: usize,
    pub class_fallbacks: usize,
    /// `(graph, class, stance, gait) -> the loaded clip's own frame count`, learned as clips
    /// load, so the rate is keyed on the CLIP rather than on the gait's assumed rifle-set length.
    ///
    /// #wclass The class is part of the key because the sets have different authored lengths: a
    /// rocket launcher's walk is not the rifle walk's 24 frames, and one shared entry would drive
    /// one of them at the other's stride.
    ///
    /// #anim-rate  The GRAPH is part of it for the same reason, and that one bites hard now that
    /// Elites draw: the Spartan's `combat:<class>:idle` is **156** frames and the Elite's
    /// `combat:<class>:idle:var1` is **50**. With one shared entry whichever species loaded last
    /// drove the other, so an Elite idled 3.1x too fast or a Spartan 3.1x too slow depending on
    /// the order bodies happened to be assigned.
    pub clip_frames: HashMap<(u16, u16, u8, u8), f32>,
    /// #film-perf  Per drawn body, the cycle-phase PREFIX SUM over its whole past
    /// ([`crate::film_anim::cycle_phase_prefix`]) and the inputs it was built from.
    ///
    /// `cycle_phase_with` walks every position sample before the tick, once per body per frame,
    /// which made one Theater frame cost more the further into the clip it was: measured on the
    /// 7-minute test clip, `sync` took 9.0 ms a frame at 1:00 and 36.8 ms at 7:00, and 97% of
    /// that was this. The prefix turns the per-frame answer into a binary search plus one
    /// segment, and it is accumulated in the same order as the walk, so the phase is the same
    /// `f32` -- see `the_prefix_phase_is_bit_identical_to_the_walk`.
    ///
    /// The key is every input the rates depend on: the body's own graph (`gid`), the animation
    /// class it is carrying (`cid`), how many clip frame counts have been learnt so far (the map
    /// fills in lazily as graphs load) and how many samples the track holds (a film that is still
    /// being written GROWS). Any change to any of them rebuilds the prefix rather than trusting
    /// it, because a stale phase would pose a body wrongly and silently.
    pub phase_prefix: HashMap<u32, ((u16, u16, usize, usize), Vec<f32>)>,
    /// #death How many CORPSES may be on screen at once, on top of `cap` living bodies.
    ///
    /// Measured at a 5 s linger: a peak of 17 bodies on the busiest clip and 3-10 on the rest.
    /// A body now lies there until the recording despawns it (a median 16 s), so the default is
    /// 14 rather than 8; past that the oldest-numbered bodies are left out, which is still a
    /// pure function of the tick. It never competes with a live player for an actor.
    pub corpse_cap: usize,
    /// #death Bodies lying dead at the last sync.
    pub corpses_shown: usize,
    /// #corpse-fall  `net id -> (the world Z this body comes to rest at, the ticks it falls)`,
    /// or `None` for a body that was already on the floor or that went over the edge.
    ///
    /// A property of the DEATH, not of the tick — exactly like `death_company` — so the cast is
    /// paid once per body however long it lies there, and the fall stays a pure function of the
    /// playback tick, so scrubbing backwards lifts the body again exactly as it fell.
    corpse_rest: HashMap<u32, Option<(f32, u32)>>,
    /// #corpse-fall  Bodies that were drawn ABOVE the floor at the last sync and are now falling
    /// to it, and bodies whose floor could not be found at all. Both reported, never silent.
    pub corpses_falling: usize,
    pub corpses_no_floor: usize,
    /// #death `net id -> who was standing next to that body when it died`. A property of the
    /// DEATH, not of the tick, so it is worked out once per body however long the body lies there
    /// — `death_company` scans the index, and the panel asks for it on every frame a corpse is on
    /// screen. Cleared with the rest of the per-film tables.
    death_company: std::cell::RefCell<HashMap<u32, Option<crate::film_anim::DeathCompany>>>,
    /// #fp Reach's FIRST-PERSON arm models, ONE ROW PER SPECIES.
    ///
    /// Handed down by the app rather than read here: they live on `matg` +0x474
    /// `@player representation` (+0x14 hands, +0x24 body), overridable at `scnr` +0x794, which is
    /// a scene read, and their section picks have to be installed on the scene before the first
    /// mesh build. Empty until the app has resolved them.
    ///
    /// #fp-arms  **There is a row per playable species and they are not interchangeable.**
    /// `forge_halo` carries two — `objects\characters\spartans\fp\fp` (47 nodes, regions
    /// `arms` + `flair_forearm`) and `objects\characters\elite\fp\fp` (41 nodes, regions
    /// `armor_pieces` + `body`) — and this used to take the FIRST one, so an Elite in first
    /// person looked down at Spartan hands.
    pub fp_reps: Vec<FpRep>,
    /// #crosshair  The `weap` tag the first-person subject is holding this frame, or `None` when
    /// the camera is not first person or the body is empty-handed. Published by
    /// `sync_first_person`, which already resolves it; the crosshair needs the same weapon so the
    /// reticule can never disagree with the gun on screen.
    pub fp_weapon_tag: Option<u32>,
    /// #fp `render model -> the actor drawing it`, for the first-person rig: the arms, the body,
    /// and one per first-person weapon model the clip has needed so far. PARKED when not in
    /// shot rather than destroyed, because destroying an actor rebuilds every mesh in the scene.
    fp_actor_of: HashMap<u32, u32>,
    /// #fp What the first-person rig is showing right now, in words, for the caption.
    pub fp_status: String,
    /// #fp-arms  Whether the body the first-person camera is riding is an ELITE, from the last
    /// frame placed. Read by `dress_first_person`, which runs on the host's own schedule.
    pub fp_subject_elite: bool,
    /// #fp `weap tag -> (its first-person MODEL, its first-person ANIMATION GRAPH)`, resolved on
    /// demand from `weap` +0x3B8 `player interface` -> `first person`.
    /// #fp-species  `(weap tag, species row) -> (first person model, first person graph)`.
    fp_weapon_of: HashMap<(u32, usize), (Option<u32>, Option<u32>)>,
    /// #skin  Per player slot, `(synthetic armour variant sid, [primary, secondary] colours)`.
    /// Built once per film; `None` until the open map can answer.
    pub appearance: Option<HashMap<u8, (u32, Option<[[f32; 3]; 2]>)>>,
    /// #film-action  Every assassination the recording STATES, decoded from the event channel
    /// (`film_anim::assassinations`). Built once per film in `prepare`.
    pub sync: Vec<crate::film_anim::Assassination>,
    /// #melee  Per body (net id), the ticks at which the recording says it SWUNG, sorted.
    /// Straight off the decoded event channel (`biped_melee_initiate`, kind 29, ref 0 = the
    /// biped), so nothing here is inferred.
    pub melee_of: HashMap<u32, Vec<u32>>,
    /// #fire  Per body, the ticks at which the recording says it FIRED (`action_weapon_fire`,
    /// kind 26, ref 0 = the biped), sorted.
    pub fire_of: HashMap<u32, Vec<u32>>,
    /// #aim  The one player slot whose aim is LIVE in this recording, if the test is unambiguous.
    ///
    /// A player's own input is captured locally and written exactly; every other slot's aim field
    /// is replicated and in practice zero-filled (measured: on 14- and 16-player clips every
    /// non-recorder slot carries the same `yaw_q` on all ~25,000 of its samples with the pitch
    /// exactly level). So this is the only body whose look direction the recording states, and
    /// the only one that can be posed with a real aim overlay. See
    /// `FilmIndex::live_aim_slot`.
    pub live_aim: Option<u8>,
    /// #aim  Bodies posed with a real aim overlay this frame, and bodies whose graph had no aim
    /// clip to pose with -- for the caption.
    pub aim_shown: usize,
    pub aim_missing: usize,
    /// #melee / #fire  Bodies playing a melee swing / carrying a firing overlay this frame.
    pub melee_shown: usize,
    pub fire_shown: usize,
    /// #melee  Some graph here has no melee clip for a body that swung, so it kept walking.
    pub melee_clip_missing: bool,
    /// #film-action  How many bodies are mid-assassination at the last sync, and whether the
    /// graph actually had the two-party clip (if it did not, the pair is standing in an idle and
    /// the caption has to say so rather than let a viewer believe they watched one).
    pub sync_shown: usize,
    pub sync_clip_missing: bool,
    /// #film-action  Exactly which clip each body in an action is playing, and at which frame —
    /// the caption prints it, because "it is performing an assassination" is a claim a viewer
    /// should be able to check against the graph.
    pub sync_clips: Vec<String>,
    /// #film-action  `net id -> the biped tag PATH the stream says that body was spawned from`.
    /// Handed down by the app, like `fp_models`: resolving a creation message's 14-bit network
    /// tag reference needs the loaded map's tag table AND the cache's per-tag datum salts, which
    /// the decode worker does not have. Empty means "nobody told us", and every body then draws
    /// with `biped`.
    pub biped_of: HashMap<u32, String>,
    /// #film-action  One actor pool per NON-default biped the recording spawned, so an Elite
    /// draws as an Elite. Keyed by tag path; `slots` remains the pool for `biped`.
    species_slots: HashMap<String, Vec<u32>>,
    species_pool: HashMap<String, ActorPool>,
    /// #film-action  Bodies drawn with a species other than `biped` at the last sync.
    pub species_shown: usize,
    /// #film-action  Guards `resolve_bipeds` so it runs once per (cache, film), not per frame.
    biped_cache: Option<u64>,
    /// #film-action  What `resolve_bipeds` found, in words, for the caption.
    pub biped_status: String,
    /// #projectile  Recorded projectiles drawn as a MODEL at the last sync, and those whose
    /// definition carries no render model at all (a bullet): the engine draws nothing of those
    /// either, so they are left to their attached effect rather than reported as an error.
    pub projectiles_shown: usize,
    pub projectiles_no_model: usize,
    /// #projectile  `body net id -> the actor slot drawing it`, written by every `sync`.
    ///
    /// The muzzle-flash lane needs the POSED actor of the body that fired, and the pool's
    /// assignment is the only thing that knows which one that is (an actor is re-assigned as
    /// players come and go). Recorded where the assignment is made rather than re-derived, so the
    /// flash can never end up on a different body than the one the pose belongs to.
    pub body_actor: HashMap<u32, u32>,
    /// #projectile  The map's weapon/projectile tag table: what each `weap` launches and plays,
    /// and what each `proj` plays. Built once per (cache, film) in `sync`; `None` until a map is
    /// open to read it from.
    pub projectiles: Option<crate::film_projectile::ProjectileTable>,
    /// #projectile  Play the firing effects, the projectile trails and the detonations. On by
    /// default -- without it a Theater clip is a gunfight with no muzzle flashes and no
    /// explosions, which is the report this lane exists for.
    pub show_projectile_fx: bool,
    /// #projectile  What the last [`ActorDriver::film_effects`] asked for, split by kind.
    pub fx_muzzle: usize,
    pub fx_trail: usize,
    pub fx_detonation: usize,
    /// #projectile  Firing events at this tick whose weapon the recording never named, so there
    /// is no `weap` to take a firing effect from. Counted, not guessed at.
    pub fx_unarmed: usize,
    /// #projectile  Firing events whose weapon has a barrel but no authored firing effect.
    pub fx_no_effect: usize,
    /// #projectile  Recorded projectiles alive at this tick that carry NO render model
    /// (`resolve_mode` -> none): a bullet. They are drawn by their attached effect alone, which
    /// is all the engine draws of them either.
    pub fx_modelless: usize,
    /// #death Set once if a corpse ever ended up on something that is NOT one of the biped's
    /// `landing_dead` clips — i.e. this biped's graph has no death animation and the bodies are
    /// lying in a standing pose. A PROPERTY of the model, not a per-frame count: the caption is
    /// computed between frames, so a count taken during the last sync would be a tick stale.
    pub death_clip_missing: bool,
    /// #death Exactly which clip each body lying dead is on, and at which frame — the panel
    /// prints it, so "that body is playing its death" is a claim a viewer can check against the
    /// graph, and a body that got NO death clip says so by name instead of silently standing.
    pub death_clips: Vec<String>,
}

impl Default for ActorDriver {
    fn default() -> Self {
        ActorDriver {
            prof: SyncProf::default(),
            pool: ActorPool::default(),
            slots: Vec::new(),
            // The FULL path, so the name match cannot land on
            // `equipment\\hologram\\bipeds\\hologram_spartan` (which also ends in "spartan").
            biped: "characters\\spartans\\spartans".into(),
            cap: 16,
            park: [0.0, 0.0, -10_000.0],
            last_error: None,
            animate: true,
            anim_of: HashMap::new(),
            track_of: HashMap::new(),
            vehicles: None,
            show_vehicles: true,
            corpse_cap: 14,
            ragdoll: true,
            ragdoll_defs: HashMap::new(),
            corpse_sims: HashMap::new(),
            ragdolls_shown: 0,
            ragdoll_steps: 0,
            ragdoll_catching_up: 0,
            ragdoll_missing: None,
            ragdoll_bodies: Vec::new(),
            melee_nodes: (0, 0),
            melee_bodies: Vec::new(),
            obj_tris: HashMap::new(),
            corpses_shown: 0,
            corpse_rest: HashMap::new(),
            corpses_falling: 0,
            corpses_no_floor: 0,
            death_company: Default::default(),
            death_clip_missing: false,
            death_clips: Vec::new(),
            projectiles_shown: 0,
            projectiles_no_model: 0,
            body_actor: HashMap::new(),
            projectiles: None,
            show_projectile_fx: true,
            fx_muzzle: 0,
            fx_trail: 0,
            fx_detonation: 0,
            fx_unarmed: 0,
            fx_no_effect: 0,
            fx_modelless: 0,
            vehicle_actor: HashMap::new(),
            vehicle_error: None,
            objects: None,
            show_objects: true,
            object_cap: 96,
            object_actor: HashMap::new(),
            objects_shown: 0,
            objectives_shown: 0,
            objects_on_spawn: 0,
            objects_capped: 0,
            vehicle_defs: HashMap::new(),
            variant_spawns: Vec::new(),
            veh_by_def: Default::default(),
            veh_by_def_done: false,
            object_error: None,
            object_status: String::new(),
            object_cache: None,
            veh_responses: HashMap::new(),
            veh_damage: HashMap::new(),
            veh_damaged_now: 0,
            wrecks_now: 0,
            veh_tilted: 0,
            veh_damage_totals: (0, 0, 0, 0),
            veh_damage_ready: false,
            cam_tags: Default::default(),
            weap_tags: Vec::new(),
            weapon_error: None,
            weapons_shown: 0,
            fp_reps: Vec::new(),
            fp_weapon_tag: None,
            fp_actor_of: HashMap::new(),
            fp_status: String::new(),
            fp_subject_elite: false,
            fp_weapon_of: HashMap::new(),
            appearance: None,
            weap_class: Vec::new(),
            class_ids: vec![crate::film_anim::DEFAULT_WEAPON_CLASS.to_string()],
            graph_ids: vec!["characters\\spartans\\spartans".to_string()],
            class_bodies: 0,
            class_fallbacks: 0,
            clip_frames: HashMap::new(),
            phase_prefix: HashMap::new(),
            sync: Vec::new(),
            melee_of: HashMap::new(),
            fire_of: HashMap::new(),
            live_aim: None,
            aim_shown: 0,
            aim_missing: 0,
            melee_shown: 0,
            fire_shown: 0,
            melee_clip_missing: false,
            sync_shown: 0,
            sync_clip_missing: false,
            sync_clips: Vec::new(),
            biped_of: HashMap::new(),
            species_slots: HashMap::new(),
            species_pool: HashMap::new(),
            species_shown: 0,
            biped_cache: None,
            biped_status: String::new(),
        }
    }
}

/// What one sync did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncReport {
    pub shown: usize,
    pub parked: usize,
    pub created: usize,
    /// Bipeds this moment has that the pool could not cover (more players than actors).
    pub wanted: usize,
    /// Vehicles drawn this frame.
    pub vehicles: usize,
    /// Bodies drawn holding the weapon the recording named.
    pub armed: usize,
    /// Bodies riding a vehicle this frame.
    pub riders: usize,
}

impl ActorDriver {
    /// Build the per-film side tables: the net-id index, and the vehicle catalog.
    ///
    /// Done here rather than in `film_decode::build_index` because identifying a vehicle needs
    /// the loaded map's scenario sandbox palette, which the decode worker thread does not have.
    fn prepare(&mut self, pb: &Playback, scene_palette: &dyn Fn(u16, u8) -> Option<(u32, String)>) {
        self.track_of = pb.index.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
        self.vehicles = Some(crate::film_anim::VehicleTable::build(&pb.index, |q, i| scene_palette(q, i)));
        // #film-action  The assassinations the recording states. Decoded, so this is a read of
        // the event channel and not a detector -- see `film_anim::assassinations`.
        self.sync = crate::film_anim::assassinations(&pb.index);
        // #melee / #fire  The two per-body action channels, folded once per film. Both are
        // ordinary decoded events with the acting biped in reference 0.
        let (melee, fire) = Self::action_tables(&pb.index);
        self.melee_of = melee;
        self.fire_of = fire;
        // #aim  Which slot's aim the recording actually states. See the field's own note.
        self.live_aim = pb.index.live_aim_slot();
    }

    /// #melee / #fire  The two per-body action channels, folded off the decoded event channel:
    /// `(melee_of, fire_of)`, each `net id -> sorted ticks`.
    ///
    /// Extracted from `prepare` so the gates can build the SAME tables the app runs on instead of
    /// a hand-written copy of them -- a gate that re-derives part of a rule disagrees with the
    /// rule the moment the rule gains a condition.
    pub fn action_tables(idx: &crate::film_decode::FilmIndex) -> (HashMap<u32, Vec<u32>>, HashMap<u32, Vec<u32>>) {
        let mut melee: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut fire: HashMap<u32, Vec<u32>> = HashMap::new();
        for ev in &idx.events {
            let Some(net) = ev.subject else { continue };
            match ev.kind {
                crate::film_action::EV_MELEE_INITIATE => melee.entry(net).or_default().push(ev.tick),
                crate::film_action::EV_WEAPON_FIRE => fire.entry(net).or_default().push(ev.tick),
                _ => {}
            }
        }
        for v in melee.values_mut().chain(fire.values_mut()) {
            v.sort_unstable();
            v.dedup();
        }
        (melee, fire)
    }

    /// #melee  Ticks since this body's most recent swing, when that is still inside `window`.
    fn action_age(table: &HashMap<u32, Vec<u32>>, net: u32, tick: u32, window: u32) -> Option<u32> {
        let v = table.get(&net)?;
        let i = v.partition_point(|&t| t <= tick);
        if i == 0 {
            return None;
        }
        let age = tick - v[i - 1];
        (age < window).then_some(age)
    }

    /// #melee  Fill in the DECODED discrete actions on a derived state: the melee swing.
    ///
    /// A swing replaces the whole pose, so it must not be raised for a body that is dead or
    /// inside a synchronised action — those already own the skeleton and the recording's own
    /// ordering says so (a Reach assassination raises no melee event at all).
    ///
    /// ONE implementation, called from both the draw path and the caption path, so the panel can
    /// never disagree with the picture about who just swung.
    fn apply_actions(&self, st: &mut crate::film_anim::BodyState, net: u32, tick: u32) {
        if st.sync.is_some() || st.stance == crate::film_anim::Stance::Dead {
            return;
        }
        // The window is one melee clip: the shipped swings are 24-33 frames at 30 fps, i.e.
        // 48-66 ticks of the film's 60 Hz line, and the clip's own length clamps the frame
        // anyway.
        if let Some(age) = Self::action_age(&self.melee_of, net, tick, MELEE_TICKS) {
            st.melee_ticks = Some(age);
            st.melee_variant = ((tick - age) % 3) as u8;
        }
    }

    /// #aim  The look direction to pose this body with: `(yaw offset from its own facing, pitch)`
    /// in radians, or `None` when the recording states no aim for it.
    ///
    /// **Only the recorder's slot has one.** For every other body the aim field is zero-filled,
    /// and posing a body from a constant junk yaw would be worse than posing nothing — so this
    /// returns `None` and the overlay is not applied. That is the recording's limit, and the
    /// caption says so rather than letting a viewer believe they are watching 16 players look
    /// around.
    fn aim_offset(&self, pb: &Playback, net: u32, tick: u32, body_yaw: Option<f32>) -> Option<(f32, f32)> {
        let slot = pb.index.slot_of_entity(net)?;
        if self.live_aim != Some(slot) {
            return None;
        }
        let aim = pb.index.aim_at(slot, tick)?;
        let dy = match body_yaw {
            Some(b) => crate::film_anim::ang_delta_pub(aim.yaw(), b),
            None => 0.0,
        };
        Some((dy, aim.pitch()))
    }

    /// #film-action  Resolve WHICH BIPED each recorded body was spawned from, and remember it.
    ///
    /// Every creation message opens with a 14-bit network tag reference naming the object
    /// definition (`EntityTrack::def_wire`). Turning that into a tag needs two things the decode
    /// worker does not have: the loaded map's tag table, and the cache's per-tag datum SALTS,
    /// which live in the `.map` file itself and in no native export. So it is done here, once per
    /// (cache, film), and both hosts — the GUI and the script runner — call this one function
    /// rather than keeping two copies of it.
    ///
    /// A body whose reference does not resolve to a `bipd` is left alone: it then draws with
    /// `biped`, and the caption says how many bodies were named.
    pub fn resolve_bipeds(
        &mut self,
        pb: &Playback,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        map_path: Option<&std::path::Path>,
    ) {
        let key = cache.0 ^ ((pb.index.tracks.len() as u64) << 32) ^ pb.index.entity_messages as u64;
        if self.biped_cache == Some(key) {
            return;
        }
        self.biped_cache = Some(key);
        self.biped_of.clear();
        self.biped_status = "no map is open to resolve the definition each spawn message names".into();
        // #film-objects  The object lane is resolved off the SAME reference table, so it shares
        // every one of these early exits and its own status line says which one it took.
        self.objects = None;
        self.object_cache = Some(key);
        self.object_status = self.biped_status.clone();
        let Some(mp) = map_path else { return };
        let Ok(bytes) = std::fs::read(mp) else {
            self.biped_status = format!("the map file {} could not be read", mp.display());
            self.object_status = self.biped_status.clone();
            return;
        };
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) {
                tags.push((t, (*g).to_string()));
            }
        }
        let Some(table) = crate::film_action::RefTable::for_map(&bytes, &tags) else {
            self.biped_status = "this map is not a Reach MCC cache, so it carries no tag table to resolve against".into();
            self.object_status = self.biped_status.clone();
            return;
        };
        // #film-objects  One walk of the reference table serves both lanes: building it enumerates
        // every tag in the cache and sorts them by datum handle, so doing it twice per film would
        // be the same work for nothing.
        self.resolve_objects(pb, native, cache, &table);
        let mut unresolved = 0usize;
        for tr in pb.index.tracks.iter().filter(|t| t.entity_type == BIPED_TYPE) {
            let Some(w) = tr.def_wire else {
                unresolved += 1;
                continue;
            };
            let named = table.resolve(w).filter(|&tag| native.tag_class(cache, tag).as_deref() == Some("bipd"));
            match named.and_then(|tag| native.tag_name(cache, tag)) {
                Some(path) => {
                    self.biped_of.insert(tr.net_id, path);
                }
                None => unresolved += 1,
            }
        }
        self.biped_status = if self.biped_of.is_empty() {
            format!("the map is open but no body's definition reference resolved to a biped ({unresolved} tried)")
        } else {
            let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
            for b in self.biped_of.values() {
                *kinds.entry(b.rsplit('\\').next().unwrap_or(b)).or_default() += 1;
            }
            format!(
                "{}{}",
                kinds.iter().map(|(k, n)| format!("{n} x {k}")).collect::<Vec<_>>().join(", "),
                if unresolved > 0 { format!("; {unresolved} unresolved") } else { String::new() }
            )
        };
    }

    /// #film-objects  Identify every OTHER object the film recorded: the objective objects (the
    /// flag), the loose and dropped weapons, and the physics debris.
    ///
    /// The report this exists for was "I am looking at the castle wars clip, and I see no flag at
    /// all". The flag was in the recording the whole time — 33 tracks, 28 of which move — and the
    /// draw filter dropped it, because [`Playback::wanted`] keeps only `BIPED_TYPE` and only
    /// vehicles had been given a lane of their own. This is that lane for everything else.
    ///
    /// Takes the reference table rather than building one: it is called from the middle of
    /// [`ActorDriver::resolve_bipeds`], which has already built it, so each film pays for one
    /// walk of the cache's tag list instead of two.
    ///
    /// Two outside things it resolves on the way, both once per film and never per frame:
    /// * each definition's **`weap` +0x3AC weapon class** (`reach_actor::weapon_class`), which
    ///   corroborates the objective list — see `film_objective::OBJECTIVE_LEAVES`;
    /// * every present object in the film's own embedded MAP VARIANT, through the scenario
    ///   sandbox palette, so a weapon that nobody ever picked up can be left to the Forge scene
    ///   that is already drawing it rather than z-fighting with it.
    fn resolve_objects(
        &mut self,
        pb: &Playback,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        refs: &crate::film_action::RefTable,
    ) {
        // The variant's own placements, resolved to real tags the same way the vehicle lane does.
        let palette = native
            .find_scenario(cache)
            .map(|scnr| native.scnr_forge_palette_full(cache, scnr))
            .unwrap_or_default();
        let mut order: Vec<(u32, u32)> = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for e in &palette {
                let k = (e.palette_index, e.entry_within);
                if seen.insert(k) {
                    order.push(k);
                }
            }
        }
        let spawns: Vec<(u32, [f32; 3])> = pb
            .index
            .variant_objects
            .iter()
            .filter_map(|&(folder, item, pos)| {
                let &(pi, ew) = order.get(folder as usize)?;
                let e = palette
                    .iter()
                    .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
                    .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
                Some((e.tag_short, pos))
            })
            .collect();
        // One class read per DISTINCT definition, memoised across the walk.
        let mut classes: HashMap<u32, Option<String>> = HashMap::new();
        let table = crate::film_objective::ObjectTable::build(
            &pb.index,
            |wire| {
                let tag = refs.resolve(wire)?;
                Some((tag, native.tag_class(cache, tag)?, native.tag_name(cache, tag)?))
            },
            |tag| {
                classes
                    .entry(tag)
                    .or_insert_with(|| crate::reach_actor::weapon_class(native, cache, tag))
                    .clone()
            },
            &spawns,
        );
        // #film-vehid  The same reference table identifies a VEHICLE, which the spawn-point
        // search cannot do on a gametype that spawns its own: `warthog` and `warthog_chaingun`
        // are stated outright in every creation message. Resolved here because this is where the
        // table is; applied in `sync`, where the vehicle table is. Keyed by the WIRE value, so
        // the map holds one entry per distinct definition (two on the measured clip) rather than
        // one per track.
        self.variant_spawns = spawns.clone();
        self.vehicle_defs.clear();
        for tr in pb.index.tracks.iter().filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE) {
            let Some(w) = tr.def_wire else { continue };
            if self.vehicle_defs.contains_key(&w) {
                continue;
            }
            let Some(tag) = refs.resolve(w) else { continue };
            let (Some(g), Some(n)) = (native.tag_class(cache, tag), native.tag_name(cache, tag)) else { continue };
            self.vehicle_defs.insert(w, (tag, g, n));
        }
        self.veh_by_def_done = false;
        self.veh_by_def = Default::default();
        let mut kinds: std::collections::BTreeMap<(&'static str, String), usize> = Default::default();
        for o in table.objects.values() {
            *kinds.entry((o.draw.label(), o.name.clone())).or_default() += 1;
        }
        self.object_status = if table.objects.is_empty() {
            format!(
                "the map is open but no recorded object resolved to something to draw \
                 ({} in the scene already, {} with no world presence, {} unresolved)",
                table.by_draw[3], table.by_draw[4], table.unresolved
            )
        } else {
            format!(
                "{}; {} already in the scene, {} tokens, {} unresolved",
                kinds.iter().map(|((k, n), c)| format!("{c} x {n} ({k})")).collect::<Vec<_>>().join(", "),
                table.by_draw[3],
                table.by_draw[4],
                table.unresolved
            )
        };
        self.objects = Some(table);
    }

    /// Which seat this body is in, with the recording's own say overriding the derived ride.
    ///
    /// #film-action  The ride detector is geometric (matched horizontal displacement between a
    /// body and a vehicle) and it cannot see the moment a player steps off, because a body that
    /// has just dismounted is still next to the vehicle. The stream states that moment:
    /// `biped_exit_vehicle` (and `biped_vehicle_evict`) name the biped, the vehicle and the
    /// 6-bit SEAT index outright. So a derived ride is CUT at a stated exit.
    ///
    /// Entry is the other way round and the corpus says so: `biped_enter_vehicle` fires **0**
    /// times in 57 films, because the engine replicates boarding through the vehicle's own seat
    /// state rather than as an event. So there is nothing to override there, and the derived ride
    /// still opens the seat.
    fn seat_at(&self, pb: &Playback, net_id: u32, tick: u32) -> Option<crate::film_anim::Seat> {
        let seat = self.vehicles.as_ref().and_then(|v| v.seat_of(net_id, tick))?;
        // The last stated exit for this body at or before now. If the ride detector still thinks
        // the body is seated after it, believe the recording.
        let exited = pb
            .index
            .events
            .iter()
            .filter(|e| {
                matches!(e.kind, crate::film_action::EV_EXIT_VEHICLE | crate::film_action::EV_VEHICLE_EVICT)
                    && e.subject == Some(net_id)
                    && e.tick <= tick
            })
            .map(|e| (e.tick, e.other))
            .max_by_key(|&(t, _)| t);
        match exited {
            // Only cut the seat when the exit names the vehicle the detector has it in, so a
            // player who got out of one vehicle and into another keeps the second seat.
            Some((_, Some(v))) if v == seat.vehicle_net_id => None,
            Some((_, None)) => None,
            _ => Some(seat),
        }
    }

    /// #film-action  Is this body inside a decoded assassination at `tick`, and on which side?
    ///
    /// Returns the state plus the SHARED ROOT the pair should be drawn on — the attacker's own
    /// recorded transform at the moment the kill landed, held for the whole performance. Both
    /// halves of a Reach sync action are authored around one root (the engine attaches both units
    /// to a `sync_action_marker` object), so putting the two bodies on the same transform is what
    /// makes them meet; drawing each on its own replicated position would not, because a body
    /// stops replicating a position almost as soon as the action starts.
    fn sync_at(
        &self,
        pb: &Playback,
        net_id: u32,
        tick: u32,
    ) -> Option<(crate::film_anim::SyncState, Option<([f32; 3], f32)>)> {
        use crate::film_anim::{SyncSide, SyncState};
        let a = self.sync.iter().find(|a| {
            a.covers(tick) && (a.attacker == Some(net_id) || a.victim == Some(net_id))
        })?;
        let side = if a.attacker == Some(net_id) { SyncSide::Attacker } else { SyncSide::Victim };
        let elite = |id: Option<u32>| {
            id.and_then(|i| self.biped_of.get(&i)).map(|p| crate::film_anim::is_elite(p)).unwrap_or(false)
        };
        // The move: off the wire when the film carried the host's own record, otherwise chosen
        // from the PAIR's ids so it is stable across re-watches and the same for both halves.
        let set_len = crate::film_anim::sync_move_set(elite(a.attacker), elite(a.victim)).len();
        let move_index = match a.action_index {
            Some(i) => i as usize % set_len,
            None => {
                let k = a.attacker.unwrap_or(0) ^ a.victim.unwrap_or(0) ^ a.tick;
                (k.rotate_right(7) as usize) % set_len
            }
        };
        let root = a
            .attacker
            .and_then(|id| self.track_of.get(&id))
            .and_then(|&i| pb.index.tracks.get(i))
            .and_then(|tr| tr.sample(a.tick).map(|p| (p, tr.facing_at(a.tick).unwrap_or(0.0).to_degrees())));
        Some((
            SyncState {
                side,
                age: a.age(tick),
                move_index,
                attacker_elite: elite(a.attacker),
                victim_elite: elite(a.victim),
                move_decoded: a.move_decoded,
                stated: a.stated,
                incident: a.incident,
                partner: if side == SyncSide::Attacker { a.victim } else { a.attacker },
            },
            root,
        ))
    }

    /// #film-camera  The tag-authored camera bits this driver decoded off the loaded map.
    pub fn camera_tags(&self) -> &crate::film_camera::CameraTags {
        &self.cam_tags
    }

    /// #vehicle-damage  Build the per-film vehicle damage tables: each vehicle model's authored
    /// damage responses, and the history the recording's own `damage_section_response` events
    /// make of them.
    ///
    /// Runs once per film. It reads tags (the `hlmt` damage-section / instant-response /
    /// region-transition blocks) and walks the whole event list, so it must never be per frame —
    /// and it cannot live in [`ActorDriver::prepare`], which has no cache.
    fn prepare_vehicle_damage(
        &mut self,
        pb: &Playback,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
    ) {
        if self.veh_damage_ready {
            return;
        }
        let Some(vt) = self.vehicles.as_ref() else { return };
        // Owned first: the loop below writes `self.veh_responses`, so the table cannot stay
        // borrowed across it.
        let vehicles: Vec<(u32, u32)> = vt.vehicles.values().map(|v| (v.net_id, v.obj_tag)).collect();
        self.veh_damage_ready = true;
        // #film-camera  The same walk answers the camera question: a vehicle's SEAT FLAGS say
        // whether riding it puts the view in third person, and that is the engine's own bit
        // rather than HMS's "every land vehicle is third person" fallback. Decoded here because
        // this is the one place that has both the vehicle list and a cache.
        for (net, obj_tag) in &vehicles {
            if let Some(flags) = crate::film_camera::decode_seat_flags(native, cache, *obj_tag) {
                self.cam_tags.seat_flags.insert(*net, flags);
            }
        }
        // #film-camera  And WHICH WEAPONS take the view out of first person -- `weap +0x204` bit
        // 15, `uses 3rd person camera`. Asked of every weapon the recording ever reports a body
        // holding, which is a small set (a film names a handful of weapon tags, not the map's
        // whole `weap` group), so this is one tag read per distinct weapon per film.
        {
            let mut seen: std::collections::HashSet<u32> = Default::default();
            for tr in &pb.index.tracks {
                for &(_, w) in &tr.weapons {
                    seen.insert(w);
                }
            }
            for w in seen {
                if crate::film_camera::decode_weapon_third_person(native, cache, w) == Some(true) {
                    self.cam_tags.third_person_weapons.insert(w);
                }
            }
        }
        // The event list is already in tick order, so each object's slice is too -- which is what
        // makes the running maximum in `DamageTimeline::build` correct.
        let mut per: HashMap<u32, Vec<crate::film_action::ActionEvent>> = HashMap::new();
        for e in pb.index.events.iter().filter(|e| e.kind == crate::film_action::EV_DAMAGE_SECTION_RESPONSE) {
            if let Some(n) = e.subject {
                per.entry(n).or_default().push(*e);
            }
        }
        let (mut damaged, mut destroyed, mut evs, mut authored) = (0usize, 0usize, 0usize, 0usize);
        for (net, obj_tag) in vehicles {
            let resp = match self.veh_responses.get(&obj_tag) {
                Some(r) => r.clone(),
                None => {
                    let r = crate::film_vehicle::DamageResponses::decode(native, cache, obj_tag);
                    self.veh_responses.insert(obj_tag, r.clone());
                    r
                }
            };
            // A model with no authored damage states cannot be shown damaged, and saying so is
            // the honest answer -- not substituting some other state.
            let Some(resp) = resp else { continue };
            authored += 1;
            let despawn = self
                .track_of
                .get(&net)
                .and_then(|&i| pb.index.tracks.get(i))
                .and_then(|t| t.destroyed_tick);
            let empty: Vec<crate::film_action::ActionEvent> = Vec::new();
            let tl = crate::film_vehicle::DamageTimeline::build(&resp, per.get(&net).unwrap_or(&empty).iter());
            evs += tl.state_events;
            if !tl.steps.is_empty() {
                damaged += 1;
            }
            let w = crate::film_vehicle::WreckRule { destroyed_at: tl.destroyed_at(), despawn };
            if w.destroyed_at.is_some() {
                destroyed += 1;
            }
            self.veh_damage.insert(net, (tl, w));
        }
        self.veh_damage_totals = (damaged, destroyed, evs, authored);
    }

    /// #vehicle-damage  The damage state this vehicle is drawn in at `tick` (0 = undamaged), and
    /// whether it is a WRECK -- destroyed and not yet collected by the engine.
    pub fn vehicle_damage_at(&self, net_id: u32, tick: u32) -> (u8, bool) {
        match self.veh_damage.get(&net_id) {
            Some((tl, w)) => (tl.state_at(tick), w.is_wreck_at(tick)),
            None => (0, false),
        }
    }

    /// #vehicle-damage  How long the recording kept this wreck before destroying the entity, in
    /// ticks. `None` when the recording does not state both ends.
    pub fn wreck_persistence(&self, net_id: u32) -> Option<u32> {
        self.veh_damage.get(&net_id).and_then(|(_, w)| w.persisted())
    }

    /// Where a rider's body belongs: the vehicle's placement times its SEAT MARKER.
    ///
    /// This is the engine's own composition, and both halves of it are authored data, not a
    /// guess. The seat markers on a Reach vehicle's render model are named exactly like the
    /// animation modes a rider plays — the mongoose carries `mongoose_d` and `mongoose_p`, the
    /// ghost `ghost_d`, the scorpion `scorpion_d`, the falcon `falcon_d` and `falcon_p_l1`… — so
    /// the same string picks the seat and the pose. (The Warthog is the exception: its seats are
    /// `driver` / `passenger` / `turret`, with `warthog_b_d` reserved for the boarding animation,
    /// so those are tried too.) The seat animation is authored relative to that marker, which is
    /// why composing the two is right and placing the body at the vehicle's ORIGIN is not —
    /// measured by capture, a rider drawn at the mongoose's origin sinks below its seat.
    ///
    /// Returns `(position, yaw degrees)`, or `None` when the vehicle is not drawn or has no
    /// marker by any of those names — in which case the caller leaves the body at its own
    /// recorded position rather than inventing one.
    fn seat_world(
        &self,
        seat: &crate::film_anim::Seat,
        pb: &Playback,
        tick: f32,
        actors: &crate::reach_actor::ActorScene,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
    ) -> Option<(glam::Vec3, f32)> {
        let tr = self.track_of.get(&seat.vehicle_net_id).and_then(|&i| pb.index.tracks.get(i))?;
        // #film-jitter  The rider rides the vehicle's placement at the SAME fractional clock the
        // vehicle itself is drawn at. Sampling the two at different moments would slide the body
        // in its own seat every frame.
        let pos = crate::film_vehicle::position_at(tr, tick)?.to_array();
        // #vehicle-rot  The rider rides the vehicle's REAL placement, basis and all. This used to
        // rebuild a yaw-only matrix, so a rider in a banking Falcon or a rolling Warthog sat
        // bolt upright while the vehicle around him turned.
        let orient = crate::film_vehicle::orientation_at(tr, tick);
        let vehicle_world = crate::film_vehicle::vehicle_basis(glam::Vec3::from(pos), orient).unwrap_or_else(|| {
            let yaw = orient.map(|(f, _)| f[1].atan2(f[0])).unwrap_or(0.0);
            glam::Mat4::from_translation(glam::Vec3::from(pos)) * glam::Mat4::from_rotation_z(yaw)
        });
        // The vehicle's own actor carries the render model the marker lives on.
        let id = self.vehicle_actor.get(&seat.vehicle_net_id)?;
        let a = actors.actors.iter().find(|a| a.id == *id)?;
        let fallbacks = ["driver", "passenger"];
        let mut names: Vec<&str> = vec![seat.mode.as_str()];
        names.extend_from_slice(&fallbacks);
        let g = names.iter().find_map(|n| a.def.model.marker_group(native, cache, n))?;
        let m = g.markers.first()?;
        let world = vehicle_world * a.pose.marker_model(m);
        let (_s, r, t) = world.to_scale_rotation_translation();
        let fwd = r * glam::Vec3::X;
        Some((t, fwd.y.atan2(fwd.x).to_degrees()))
    }

    /// Place the pool from the playback's current moment. Grows the pool on demand up to `cap`.
    /// #corpse-fall  Find the floor under every body that has just gone down.
    ///
    /// **Why this exists.** The user's report was "the ragdolls for all bipeds make bipeds just
    /// float midair", and the measurement behind the fix is in `film corpses trace`: a film
    /// reports **no position at all after a body goes down** (0 reports after the death on every
    /// one of 1,662 corpse records on the development clip), so the engine's own ragdoll fall is
    /// simply not on the wire. The body is drawn at its last reported position, and the position
    /// stream is sparse — so a player shot while jumping, falling, or between two sparse samples
    /// is left hanging exactly where the last packet put him, which on that clip is **up to
    /// 1.0 wu (three metres) above the floor**, for the whole twenty seconds the corpse now
    /// lies there.
    ///
    /// So the landing is DERIVED, and derived with HMS's own instrument rather than a guess: one
    /// downward cast through [`crate::scene::SceneController::raycast_solid`] — the BSP *and* the
    /// placed objects, the same primitive the camera stand-off solver and physics settle trust —
    /// and then a free fall at the engine's own measured [`CORPSE_GRAVITY`].
    ///
    /// `floor` is handed in rather than taken, because the collision world belongs to the scene
    /// and the driver only has the cache. It must answer "the world Z of the first solid surface
    /// below this point", or `None`.
    /// #projectile  Every effect the FILM lane should be playing at this moment: the weapon
    /// firing effects, the recorded projectiles' travelling effects and the detonations.
    ///
    /// Called from the host right after `ActorScene::update` has written the pose, next to
    /// `ActorScene::particle_sources` and for the same reason: a muzzle flash sits on a marker of
    /// the gun in a hand, so it can only be placed once this frame's pose exists. The result is
    /// handed to [`crate::film_projectile::install`], which resolves the `effe` and installs the
    /// lane.
    ///
    /// **A pure function of the tick.** Nothing here is integrated frame to frame — every birth
    /// time comes from the event's own film tick and every position from the recorded track — so
    /// scrubbing to a moment twice plays the same thing, which is the rule the ragdolls, the
    /// death clips and the camera are already held to.
    ///
    /// What it deliberately does NOT do: simulate a rifle round. A Reach film records one
    /// player's look direction and zero-fills every other slot's (see [`ActorDriver::live_aim`]),
    /// so for all but one body the recording does not state which way the shot went. Firing a
    /// tracer along the body's yaw would be an invention, and an invention that would be visibly
    /// horizontal. The bullet's own `proj` definition is read and available
    /// ([`crate::film_projectile::WeaponShot::projectile`]) for the day the direction is; until
    /// then the muzzle flash is what the recording can honestly support.
    pub fn film_effects(
        &mut self,
        pb: &Playback,
        actors: &crate::reach_actor::ActorScene,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
    ) -> Vec<crate::film_projectile::FilmEffect> {
        use crate::film_projectile::{FilmEffect, FILM_FX_SEED, TICK_SECONDS};
        self.fx_muzzle = 0;
        self.fx_trail = 0;
        self.fx_detonation = 0;
        self.fx_unarmed = 0;
        self.fx_no_effect = 0;
        self.fx_modelless = 0;
        if !self.show_projectile_fx {
            return Vec::new();
        }
        let Some(pt) = self.projectiles.as_ref() else { return Vec::new() };
        let tick = pb.tick.round().max(0.0) as u32;
        let ftick = pb.tick.max(0.0);
        let mut out: Vec<FilmEffect> = Vec::new();

        // ---- 1. the weapon FIRING EFFECTS ------------------------------------------------
        //
        // `action_weapon_fire` (event 26) names the body that fired and the tick. The weapon is
        // the one the recording says was in its hands at that tick, and the effect is that
        // weapon's `barrels[0].firing effects[0].firing effect^`. It is played as a ONE-SHOT born
        // at the firing tick, on the gun's own model, so the effect's `primary_trigger` /
        // `muzzle_flash` / `primary_ejection` locations land on the muzzle.
        for (&net, ticks) in self.fire_of.iter() {
            // Only shots still inside the firing window; `FIRE_TICKS` is the same window the
            // firing ANIMATION overlay uses, so the flash and the recoil clip agree by
            // construction.
            let lo = tick.saturating_sub(FIRE_TICKS);
            let from = ticks.partition_point(|&t| t < lo);
            let to = ticks.partition_point(|&t| t <= tick);
            if from >= to {
                continue;
            }
            let Some(&slot) = self.body_actor.get(&net) else { continue };
            let Some(a) = actors.actors.iter().find(|a| a.id == slot) else { continue };
            let Some(w) = a.weapon.as_ref() else {
                self.fx_unarmed += to - from;
                continue;
            };
            let Some(shot) = pt.by_weapon.get(&w.obj_tag) else {
                self.fx_unarmed += to - from;
                continue;
            };
            let Some(effe) = shot.firing_effect else {
                self.fx_no_effect += to - from;
                continue;
            };
            // The gun's model->world matrix for THIS pose -- the same expression
            // `ActorScene::describe` prints as "where the HELD WEAPON actually ends up".
            let world = a.world() * w.world(glam::Mat4::IDENTITY, &a.pose);
            if !world.w_axis.truncate().is_finite() {
                continue;
            }
            for &ft in &ticks[from..to] {
                out.push(FilmEffect {
                    // One clock per (body, shot): a second shot from the same gun must re-arm
                    // rather than inherit the first one's spent clock.
                    seed: FILM_FX_SEED ^ 0x01 ^ ((net as u64) << 24) ^ ((ft as u64) << 4),
                    effe,
                    mode: shot.mode,
                    world,
                    birth: ft as f32 * TICK_SECONDS,
                    path: None,
                    label: format!("muzzle #{net:#x} tick {ft}"),
                });
                self.fx_muzzle += 1;
            }
        }

        // ---- 2 and 3. the recorded PROJECTILES: their trail and their detonation ----------
        if let Some(ot) = self.objects.as_ref() {
            for o in ot.objects.values() {
                if o.draw != crate::film_objective::Draw::Projectile {
                    continue;
                }
                let Some(fx) = pt.by_projectile.get(&o.obj_tag) else { continue };
                let Some(&ti) = self.track_of.get(&o.net_id) else { continue };
                let Some(tr) = pb.index.tracks.get(ti) else { continue };
                if tr.positions.is_empty() {
                    continue;
                }
                let born = tr.first_tick as f32 * TICK_SECONDS;
                let died = tr.last_tick as f32 * TICK_SECONDS;
                let alive = ftick >= tr.first_tick as f32 && ftick <= tr.last_tick as f32 + 1.0;
                if alive {
                    if fx.mode == 0 || fx.mode == 0xFFFF_FFFF {
                        self.fx_modelless += 1;
                    }
                    if let Some(world) = crate::film_projectile::projectile_world(tr, ftick) {
                        let path = crate::film_projectile::recorded_path(tr);
                        for (i, &effe) in fx.attached.iter().enumerate() {
                            out.push(FilmEffect {
                                seed: FILM_FX_SEED ^ 0x02 ^ ((o.net_id as u64) << 24) ^ ((i as u64) << 8),
                                effe,
                                mode: fx.mode,
                                world,
                                birth: born,
                                // A MOVING source: the recorded flight, so the trail is left in
                                // the air behind the projectile instead of dragged along with it.
                                path: Some(path.clone()),
                                label: format!("trail #{:#x} {}", o.net_id, fx.name),
                            });
                            self.fx_trail += 1;
                        }
                    }
                }
                // The DETONATION. A projectile's track ends when the engine destroyed the
                // projectile, which for anything that authors a detonation effect is the
                // detonation -- and the last recorded position is where it happened. The effect
                // is played only when the `proj` authors one, so a bullet (which authors none)
                // plays nothing rather than being given an explosion it never had.
                let Some(det) = fx.detonation else { continue };
                let end = tr.positions[tr.positions.len() - 1].1;
                if ftick >= tr.last_tick as f32 && ftick <= tr.last_tick as f32 + 240.0 {
                    out.push(FilmEffect {
                        seed: FILM_FX_SEED ^ 0x03 ^ ((o.net_id as u64) << 24),
                        effe: det,
                        // A detonation is not on the projectile's model: it is an explosion in the
                        // world, and its systems name no marker.
                        mode: 0,
                        world: glam::Mat4::from_translation(glam::Vec3::from(end)),
                        birth: died,
                        path: None,
                        label: format!("detonation #{:#x} {}", o.net_id, fx.name),
                    });
                    self.fx_detonation += 1;
                }
            }
        }
        let _ = (native, cache);
        out
    }

    pub fn ground_corpses(&mut self, pb: &Playback, floor: &dyn Fn([f32; 3]) -> Option<f32>) {
        let mut no_floor = 0usize;
        for p in pb.corpses(self.corpse_cap) {
            if !self.corpse_rest.contains_key(&p.net_id) {
                let rest = corpse_fall(p.pos, floor(p.pos));
                self.corpse_rest.insert(p.net_id, rest);
            }
            // A body with no entry at all was already on the floor; one whose cast found nothing
            // is over the edge of the map. Only the second is worth reporting, so the two are
            // told apart by re-asking the floor for the bodies that have no resting height.
            if self.corpse_rest.get(&p.net_id).copied().flatten().is_none() && floor(p.pos).is_none() {
                no_floor += 1;
            }
        }
        self.corpses_no_floor = no_floor;
    }

    /// #corpse-fall  Where [`ActorDriver::ground_corpses`] decided this body comes to rest, and
    /// how long it falls to get there. `None` = already on the floor, or no floor under it.
    pub fn corpse_rest_of(&self, net_id: u32) -> Option<(f32, u32)> {
        self.corpse_rest.get(&net_id).copied().flatten()
    }

    /// #corpse-fall  The world Z a corpse is actually DRAWN at, `age` ticks after it went down.
    /// The same expression `sync` places it with, so the diagnostic cannot drift from the draw.
    pub fn corpse_drawn_z(&self, net_id: u32, death_pos: [f32; 3], age: u32) -> f32 {
        match self.corpse_rest_of(net_id) {
            Some((rest_z, ft)) => {
                let t = age.min(ft) as f32 / 60.0;
                (death_pos[2] - 0.5 * CORPSE_GRAVITY * t * t).max(rest_z)
            }
            None => death_pos[2],
        }
    }

    pub fn sync(
        &mut self,
        pb: &Playback,
        actors: &mut crate::reach_actor::ActorScene,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
    ) -> SyncReport {
        self.prof.begin(); // #film-perf  see `SyncProf`
        // The scenario sandbox palette, which turns a Forge quota index into a real object tag.
        // Read ONCE per film, not per frame: `scnr_forge_palette_full` walks 348 entries out of
        // the cache and the resolve then asks the cache for a class and a name per entry. Doing
        // that every frame was most of the cost of merely having a film open with nothing drawn.
        if self.vehicles.is_none() {
            let palette = native
                .find_scenario(cache)
                .map(|scnr| native.scnr_forge_palette_full(cache, scnr))
                .unwrap_or_default();
            let mut order: Vec<(u32, u32)> = Vec::new();
            {
                let mut seen = std::collections::HashSet::new();
                for e in &palette {
                    let k = (e.palette_index, e.entry_within);
                    if seen.insert(k) {
                        order.push(k);
                    }
                }
            }
            let resolve = |folder: u16, item: u8| -> Option<(u32, String)> {
                let &(pi, ew) = order.get(folder as usize)?;
                let e = palette
                    .iter()
                    .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
                    .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
                // Only VEHICLES are wanted here; anything else resolving would put scenery in the
                // vehicle table.
                if native.tag_class(cache, e.tag_short).as_deref() != Some("vehi") {
                    return None;
                }
                Some((e.tag_short, native.tag_name(cache, e.tag_short).unwrap_or_default()))
            };
            self.prepare(pb, &resolve);
        } else if self.track_of.len() != pb.index.tracks.len() {
            // A different film: drop the tables and rebuild them on the next frame.
            self.track_of.clear();
            self.death_company.borrow_mut().clear();
            self.corpse_rest.clear();
            self.corpse_sims.clear();
            self.ragdoll_defs.clear();
            self.obj_tris.clear();
            self.vehicles = None;
            self.veh_damage.clear();
            self.veh_responses.clear();
            self.veh_damage_ready = false;
            // #film-objects  Same reason as the vehicle table: these are keyed by this film's
            // net ids. `resolve_objects` rebuilds on the next `resolve_bipeds`, whose guard key
            // already includes the track count.
            self.objects = None;
            self.object_cache = None;
            // #projectile  The projectile table is per (cache, film) like the object table: the
            // learned projectiles come from THIS film's recorded tracks.
            self.projectiles = None;
            // #film-vehid  Both inputs to the definition pass are per-film too.
            self.vehicle_defs.clear();
            self.variant_spawns.clear();
            self.veh_by_def_done = false;
            self.veh_by_def = Default::default();
        self.cam_tags = Default::default();
        }
        // #film-vehid  Fill in the vehicles the Forge spawn-point search could not identify from
        // the definition their own creation message states. A one-shot per film: the inputs are
        // per-film and the result is an insert into the table. It runs here rather than in
        // `resolve_objects` because the two halves land in different passes -- the reference
        // table arrives with the map, the vehicle table with the sandbox palette -- and this is
        // the one place both are in hand.
        if !self.veh_by_def_done && !self.vehicle_defs.is_empty() && self.vehicles.is_some() {
            self.veh_by_def_done = true;
            let defs = std::mem::take(&mut self.vehicle_defs);
            let spawns = std::mem::take(&mut self.variant_spawns);
            let mut got = None;
            if let Some(vt) = self.vehicles.as_mut() {
                got = Some(crate::film_vehicle::identify_by_definition(&pb.index, vt, &spawns, |w| {
                    defs.get(&w).cloned()
                }));
            }
            self.vehicle_defs = defs;
            self.variant_spawns = spawns;
            if let Some(g) = got {
                self.veh_by_def = g;
            }
        }
        // #projectile  What each weapon launches and plays, and what each projectile plays. Read
        // ONCE per (cache, film): it walks every `weap`'s barrels block and every `proj` those
        // name, which is 63 and 52 tags on `forge_halo` and must never be per-frame work.
        if self.projectiles.is_none() && !self.weap_tags.is_empty() {
            let mut t = crate::film_projectile::ProjectileTable::build(native, cache, &self.weap_tags, |p| {
                crate::reach_projectile::object_attachment_effects(native, cache, p)
                    .into_iter()
                    .map(|(e, _, _)| e)
                    .collect()
            });
            // A projectile the recording REPLICATED whose launcher is not in the weapon list (a
            // thrown grenade comes from the unit, not from a `weap` barrel) still needs its
            // definition, so learn it from the recorded tracks.
            if let Some(ot) = self.objects.as_ref() {
                let projs: Vec<u32> = ot
                    .objects
                    .values()
                    .filter(|o| o.draw == crate::film_objective::Draw::Projectile)
                    .map(|o| o.obj_tag)
                    .collect();
                for p in projs {
                    t.learn_projectile(native, cache, p, |q| {
                        crate::reach_projectile::object_attachment_effects(native, cache, q)
                            .into_iter()
                            .map(|(e, _, _)| e)
                            .collect()
                    });
                }
            }
            self.projectiles = Some(t);
        }
        if self.weap_tags.is_empty() {
            self.weap_tags = native.tags_by_group(cache, "weap");
            // #wclass  Each weapon's animation class, read ONCE per map (63 tags on
            // `forge_halo`), never per frame. `weap` +0x3AC; see `reach_actor::weapon_class`.
            self.weap_class = self
                .weap_tags
                .iter()
                .map(|&t| crate::reach_actor::weapon_class(native, cache, t))
                .collect();
        }

        let tick = pb.tick.round().max(0.0) as u32;
        // #film-jitter  The clock itself, unrounded. Which objects EXIST at this moment is a
        // question about a tick; WHERE each one is drawn is a question about the clock, and
        // rounding the second one is what made a moving vehicle stall for a frame and then catch
        // up. See `film_vehicle::position_at`.
        let ftick = pb.tick.max(0.0);
        // #death The living first, then the corpses — the two caps are separate so a body on the
        // ground can never take an actor a live player needs.
        // #film-perf  ONE walk of the index per sync: the frame is computed here and both the
        // drawable bodies and `wanted` come out of it.
        self.prof.lap(0); // #film-perf  the once-per-film table guards
        let frame = pb.frame();
        let mut rep = SyncReport::default();
        rep.wanted = frame.iter().filter(|p| p.entity_type == BIPED_TYPE).count();
        let bipeds = pb.bodies_from(frame, self.cap, self.corpse_cap);
        self.prof.lap(1); // #film-perf  the walks of every track in the recording

        // ---- the vehicles the film recorded -------------------------------------------------
        if self.show_vehicles {
            // #vehicle-damage  Per-film, per-vehicle: what its model's damage responses can do,
            // and which of them the recording fired. Built ONCE (it reads tags), right here
            // rather than in `prepare`, because `prepare` has no cache.
            self.prepare_vehicle_damage(pb, native, cache);
            // #vehicle-rot  The FULL recorded basis, not a compass heading. A vehicle's base
            // dirty block carries field 2 -- a 19-bit up axis plus an 8-bit angle about it, which
            // the engine's own decode turns into an orthonormal (forward, up) pair -- on a large
            // share of its updates, so its pitch and ROLL are decoded facts. Collapsing that to
            // `atan2(fwd.y, fwd.x)` is what used to draw a man-cannoned Warthog flat.
            let live: Vec<(u32, u32, [f32; 3], f32, Option<glam::Mat4>, u8)> = match &self.vehicles {
                Some(vt) => vt
                    .to_draw_at_with(&pb.index, tick, &|net| {
                        self.veh_damage.get(&net).map_or(false, |(_, w)| w.is_wreck_at(tick))
                    })
                    .into_iter()
                    .map(|(v, pos, orient)| {
                        // #film-jitter  WHICH vehicles are drawn is decided on the whole tick --
                        // a lifetime is a tick -- but WHERE each one is drawn is re-sampled at
                        // the fractional clock. `pb.tick` is an f32 and the placement used to
                        // round it, so a vehicle stood still for a whole display frame and then
                        // jumped, which is the jitter on a flying vehicle. Equal to the rounded
                        // sampler on a whole tick, so every determinism gate still bites.
                        let (pos, orient) = match self.track_of.get(&v.net_id).and_then(|&i| pb.index.tracks.get(i)) {
                            Some(tr) => (
                                crate::film_vehicle::position_at(tr, ftick).map(|p| p.to_array()).unwrap_or(pos),
                                crate::film_vehicle::orientation_at(tr, ftick).or(orient),
                            ),
                            None => (pos, orient),
                        };
                        let yaw = orient.map(|(f, _)| f[1].atan2(f[0]).to_degrees()).unwrap_or(0.0);
                        let world = crate::film_vehicle::vehicle_basis(glam::Vec3::from(pos), orient);
                        let state = self.veh_damage.get(&v.net_id).map_or(0, |(tl, _)| tl.state_at(tick));
                        (v.net_id, v.obj_tag, pos, yaw, world, state)
                    })
                    .collect(),
                None => Vec::new(),
            };
            self.veh_tilted = live
                .iter()
                .filter(|(_, _, _, _, w, _)| {
                    w.map_or(false, |m| crate::film_vehicle::tilt_degrees(m.z_axis.truncate().to_array()) > 5.0)
                })
                .count();
            self.veh_damaged_now = live.iter().filter(|(_, _, _, _, _, st)| *st > 0).count();
            self.wrecks_now = live
                .iter()
                .filter(|(_, _, _, _, _, st)| *st >= crate::film_vehicle::STATE_DESTROYED)
                .count();
            let mut seen: std::collections::HashSet<u32> = Default::default();
            for (net, obj_tag, pos, yaw, world, state) in live {
                seen.insert(net);
                let id = match self.vehicle_actor.get(&net).copied() {
                    Some(id) => id,
                    None => match actors.add_object(native, cache, obj_tag, glam::Vec3::from(pos), yaw) {
                        Ok(id) => {
                            self.vehicle_actor.insert(net, id);
                            rep.created += 1;
                            id
                        }
                        Err(e) => {
                            self.vehicle_error = Some(e);
                            continue;
                        }
                    },
                };
                if let Some(a) = actors.get_mut(id) {
                    a.pos = glam::Vec3::from(pos);
                    a.yaw = yaw;
                    // #vehicle-rot  `Actor::world_override` replaces the pos/yaw/pitch placement
                    // with a whole matrix, which is the only shape that can express ROLL. Written
                    // every sync from the tick, so it stays a pure function of the moment and
                    // scrubbing backwards undoes the tumble. `None` (a vehicle the recording has
                    // not oriented yet) falls back to the yaw placement rather than to identity.
                    a.world_override = world;
                    rep.vehicles += 1;
                }
                // #vehicle-damage  The authored damage permutations, driven by the recording.
                actors.set_damage(id, state);
            }
            // Park the vehicles that are not alive at this tick rather than destroying them: a
            // destroy would rebuild every mesh in the scene on the next scrub.
            for (net, id) in self.vehicle_actor.clone() {
                if !seen.contains(&net) {
                    if let Some(a) = actors.get_mut(id) {
                        a.pos = glam::Vec3::from(self.park);
                        // #vehicle-rot  The override is a whole PLACEMENT, so a parked actor that
                        // kept the one it had last frame would still be drawn where it was --
                        // `pos` alone would never move it again.
                        a.world_override = None;
                    }
                }
            }
        }

        self.prof.lap(2); // #film-perf  the vehicle lane
        // ---- #film-objects  the objective objects, the loose weapons and the debris ---------
        //
        // The flag reaches the screen here. Same shape as the vehicle lane above, deliberately:
        // a `weap` is an ordinary object, so it is placed by `add_object` with the SAME recorded
        // `(forward, up)` basis -- a flag planted in the ground, dropped on its side or carried
        // at an angle is all one rule. Nothing here is weapon-specific except the classification,
        // and that happened once per film in `resolve_objects`.
        self.objects_shown = 0;
        self.objectives_shown = 0;
        self.objects_on_spawn = 0;
        self.objects_capped = 0;
        if self.show_objects {
            let live: Vec<(u32, u32, [f32; 3], f32, Option<glam::Mat4>, bool, bool)> = match &self.objects {
                Some(ot) => {
                    self.objects_on_spawn = ot.parked_on_spawn_at(&pb.index, tick);
                    let want = ot.to_draw_at(&pb.index, tick).len();
                    self.objects_capped = want.saturating_sub(self.object_cap);
                    ot.to_draw_capped(&pb.index, tick, self.object_cap)
                        .into_iter()
                        .map(|(o, pos, orient)| {
                            // #film-jitter  Re-sampled at the fractional clock, exactly as the
                            // vehicle lane above: a thrown garbage tire and a man-cannoned
                            // Warthog are the same placement rule and have the same judder.
                            let (pos, orient) = match self.track_of.get(&o.net_id).and_then(|&i| pb.index.tracks.get(i)) {
                                Some(tr) => (
                                    crate::film_vehicle::position_at(tr, ftick).map(|p| p.to_array()).unwrap_or(pos),
                                    crate::film_vehicle::orientation_at(tr, ftick).or(orient),
                                ),
                                None => (pos, orient),
                            };
                            let yaw = orient.map(|(f, _)| f[1].atan2(f[0]).to_degrees()).unwrap_or(0.0);
                            let world = crate::film_objective::object_basis(glam::Vec3::from(pos), orient);
                            (
                                o.net_id,
                                o.obj_tag,
                                pos,
                                yaw,
                                world,
                                o.draw == crate::film_objective::Draw::Objective,
                                o.draw == crate::film_objective::Draw::Projectile,
                            )
                        })
                        .collect()
                }
                None => Vec::new(),
            };
            let mut seen: std::collections::HashSet<u32> = Default::default();
            self.projectiles_shown = 0;
            self.projectiles_no_model = 0;
            for (net, obj_tag, pos, yaw, world, is_objective, is_projectile) in live {
                seen.insert(net);
                let id = match self.object_actor.get(&net).copied() {
                    Some(id) => id,
                    None => match actors.add_object(native, cache, obj_tag, glam::Vec3::from(pos), yaw) {
                        Ok(id) => {
                            self.object_actor.insert(net, id);
                            rep.created += 1;
                            id
                        }
                        Err(e) => {
                            // #projectile  A BULLET has no render model at all (`resolve_mode`
                            // returns none for `assault_rifle_bullet`, `sniper_rifle_bullet`,
                            // `falcon_machinegun_bullet`, measured on `forge_halo`), so failing
                            // to build one is not an error to report -- the engine draws nothing
                            // of it either and its tracer is the attached EFFECT. Counted so the
                            // caption can say so.
                            if is_projectile {
                                self.projectiles_no_model += 1;
                            } else {
                                self.object_error = Some(e);
                            }
                            continue;
                        }
                    },
                };
                if let Some(a) = actors.get_mut(id) {
                    a.pos = glam::Vec3::from(pos);
                    a.yaw = yaw;
                    // The whole recorded placement, basis and all -- the only shape that can
                    // express a flag lying on its side. Written every sync FROM THE TICK, so it
                    // is a pure function of the moment and a scrub backwards reproduces it.
                    a.world_override = world;
                    self.objects_shown += 1;
                    if is_objective {
                        self.objectives_shown += 1;
                    }
                    if is_projectile {
                        self.projectiles_shown += 1;
                    }
                }
            }
            // Park what is not alive now rather than destroying it: a destroy rebuilds every
            // mesh in the scene on the next scrub.
            for (net, id) in self.object_actor.clone() {
                if !seen.contains(&net) {
                    if let Some(a) = actors.get_mut(id) {
                        a.pos = glam::Vec3::from(self.park);
                        a.world_override = None;
                    }
                }
            }
        } else {
            for (_, id) in self.object_actor.clone() {
                if let Some(a) = actors.get_mut(id) {
                    a.pos = glam::Vec3::from(self.park);
                    a.world_override = None;
                }
            }
        }

        self.prof.lap(3); // #film-perf  the loose-object lane
        // ---- the bodies ---------------------------------------------------------------------
        // Reset the per-sync counters BEFORE anything increments them.
        self.sync_shown = 0;
        self.species_shown = 0;
        self.corpses_falling = 0;
        self.sync_clips.clear();
        self.death_clips.clear();
        // #film-action  Bodies are grouped by the BIPED the recording says they were spawned
        // from, and each species gets its own actor pool. An actor's mesh is built from one
        // model, so an Elite cannot borrow a Spartan actor -- and until this landed every body
        // drew as a Spartan because there was only ever one pool.
        //
        // The empty key is the DEFAULT pool (`self.biped`): that is every body the recording
        // names as the Spartan, and every body it names nothing for.
        let mut grouped: std::collections::BTreeMap<String, Vec<Placement>> = Default::default();
        for p in bipeds {
            // `biped` is a SUFFIX (`characters\\spartans\\spartans`) and the resolved paths are
            // full (`objects\\characters\\spartans\\spartans`), so the default is matched by
            // ending rather than by equality -- otherwise every Spartan would be routed into a
            // second, duplicate pool and the default one would sit empty.
            let key = match self.biped_of.get(&p.net_id) {
                Some(b) if !b.ends_with(self.biped.as_str()) => b.clone(),
                _ => String::new(),
            };
            grouped.entry(key).or_default().push(p);
        }
        let budget = self.cap + self.corpse_cap;
        // Grow each pool to cover its own group, and remember the keys in a stable order.
        let keys: Vec<String> = grouped.keys().cloned().collect();
        for key in &keys {
            let want = grouped.get(key).map(|g| g.len()).unwrap_or(0).min(budget);
            let path = if key.is_empty() { self.biped.clone() } else { key.clone() };
            loop {
                let have = if key.is_empty() {
                    self.slots.len()
                } else {
                    self.species_slots.get(key).map(|v| v.len()).unwrap_or(0)
                };
                if have >= want {
                    break;
                }
                let park = glam::Vec3::from(self.park);
                match actors.add(native, cache, &path, park, 0.0) {
                    Ok(id) => {
                        if key.is_empty() {
                            self.slots.push(id);
                        } else {
                            self.species_slots.entry(key.clone()).or_default().push(id);
                        }
                        rep.created += 1;
                    }
                    Err(e) => {
                        // A species whose model will not load must not break the others: record
                        // it and let that group fall through to the default pool next frame.
                        self.last_error = Some(e);
                        break;
                    }
                }
            }
        }
        if !self.slots.is_empty() {
            self.last_error = None;
        }
        // One combined assignment: each group against its own pool, plus every pool whose group
        // is absent this frame so its actors get parked rather than left where they were.
        let mut assigned: Vec<(u32, Option<Placement>)> = Vec::new();
        let empty: Vec<Placement> = Vec::new();
        {
            let group = grouped.get("").unwrap_or(&empty).clone();
            let slots = self.slots.clone();
            assigned.extend(self.pool.assign(&slots, &group));
        }
        let species_keys: Vec<String> = self.species_slots.keys().cloned().collect();
        for key in species_keys {
            let group = grouped.get(&key).unwrap_or(&empty).clone();
            let slots = self.species_slots.get(&key).cloned().unwrap_or_default();
            let pool = self.species_pool.entry(key.clone()).or_default();
            let a = pool.assign(&slots, &group);
            if !key.is_empty() {
                self.species_shown += a.iter().filter(|(_, p)| p.is_some()).count();
            }
            assigned.extend(a);
        }
        // The last slot is the ONE-SHOT channel: `(age in ticks, is a sync action)`. A death clip
        // and a two-party sync action are both played once from a start tick and then held, which
        // is the opposite of a cycle; the bool only decides which caption counts it.
        let mut want_anim: Vec<(u32, Vec<String>, f32, bool, (u16, u16, u8, u8), Option<(u32, u8)>)> = Vec::new();
        // #aim / #fire  The OVERLAY requests for this frame: `(slot, aim clip candidates, aim
        // offset, fire clip candidates, ticks since the shot)`. Resolved after the base clips,
        // because an overlay composes onto whatever the base left behind.
        let mut want_overlay: Vec<(u32, Vec<String>, Option<(f32, f32)>, Vec<String>, Option<u32>)> = Vec::new();
        // #melee  Per slot, the REPLACEMENT clip to lay over the locomotion clip and how old the
        // swing is in film ticks. `None` = nothing to replace, which is the ordinary case.
        let mut want_replace: Vec<(u32, Option<(Vec<String>, u32)>)> = Vec::new();
        let mut weapons: Vec<(u32, Option<u32>)> = Vec::new();
        self.body_actor.clear();
        for (slot, place) in assigned {
            let Some(a) = actors.get_mut(slot) else { continue };
            match place {
                Some(p) => {
                    // #projectile  Which actor is drawing this body, for the muzzle-flash lane.
                    self.body_actor.insert(p.net_id, slot);
                    a.pos = glam::Vec3::from(p.pos);
                    if p.has_yaw {
                        a.yaw = p.yaw;
                    }
                    // #armour-fx The body's own stable identity, which is what its armour
                    // effect's randoms are seeded from. The ACTOR SLOT cannot be: the pool hands
                    // out whichever slot is free, so it depends on the order bodies arrived and
                    // not on the playhead — scrub away and back and the same body can come back
                    // on a different slot. The recorded net id is a function of the recording.
                    a.fx_seed = p.net_id;
                    rep.shown += 1;
                    // Derive what this body is DOING from its own track: the recorded position
                    // and the decoded body yaw together give the direction of travel in the
                    // body's frame, which is what picks between the forward, back, left and right
                    // cycles the graph has always had.
                    let seat = self.seat_at(pb, p.net_id, tick);
                    let tr = self.track_of.get(&p.net_id).and_then(|&i| pb.index.tracks.get(i));
                    // #corpse-fall  Where this body comes to rest, and how far into the fall
                    // it is at this tick. `None` = it was already on the floor (the common case)
                    // or no floor was found under it.
                    let fall = p.dead_ticks.and_then(|_| self.corpse_rest.get(&p.net_id).copied().flatten());
                    let falling = matches!((p.dead_ticks, fall), (Some(age), Some((_, ft))) if age < ft);
                    let mut st = match p.dead_ticks {
                        // #death Which death clip this body falls with is decided by its OWN
                        // graph (the Elite's death clips are not named like the Spartan's) and by
                        // whether it was off the ground when it died.
                        //
                        // #corpse-fall  The clip's age is counted from the LANDING, not from the
                        // death: `landing_dead` is authored as a body arriving at the floor, so a
                        // body still dropping towards it plays the `airborne_dead` flail and only
                        // starts its landing when it actually lands.
                        Some(age) => crate::film_anim::dead_state(
                            p.net_id,
                            match fall {
                                Some((_, ft)) if age >= ft => age - ft,
                                _ => age,
                            },
                            self.biped_of.get(&p.net_id).is_some_and(|b| crate::film_anim::is_elite(b)),
                            falling
                                || tr.and_then(|tr| pb.death_at(tr).map(|d| crate::film_anim::died_airborne(tr, d)))
                                    .unwrap_or(false),
                        ),
                        None => match tr {
                            Some(tr) => crate::film_anim::derive(tr, tick, seat),
                            None => crate::film_anim::BodyState::default(),
                        },
                    };
                    // #jump  How long ago this body touched down, filled HERE rather than inside
                    // `derive` because `cycle_phase` calls `derive` once per segment and the
                    // lookback must not ride that path. A seated or dead body never lands.
                    if st.seat.is_none() && p.dead_ticks.is_none() {
                        st.land_ticks = tr.and_then(|tr| crate::film_anim::land_age(tr, tick));
                        // #jump-loop  ...and how long it has been in the air, so the airborne
                        // clip advances once and holds instead of wrapping.
                        st.air_ticks = tr.and_then(|tr| crate::film_anim::airborne_age(tr, tick));
                        // #jump-apex  The STANCE has to agree with it, or a body at the top of its
                        // own jump is posed on a ground gait while the airborne clip runs. `derive`
                        // decides the stance from an instantaneous `|vz| > cut`, which is false at
                        // an apex; the apex-bridged test lives here because `cycle_phase` calls
                        // `derive` once per SEGMENT and must not pay for the extra samples.
                        // The gait mirrors what `derive` picks for a flight.
                        if st.air_ticks.is_some()
                            && st.seat.is_none()
                            && st.sync.is_none()
                            && st.stance != crate::film_anim::Stance::Flight
                        {
                            st.stance = crate::film_anim::Stance::Flight;
                            st.gait = if st.ground_speed > crate::film_anim::cuts::WALK {
                                crate::film_anim::Gait::RunFront
                            } else {
                                crate::film_anim::Gait::Idle
                            };
                            st.because = "airborne: still in the flight its own motion began (apex bridged)";
                        }
                    }
                    // #melee  The decoded discrete actions, on both the draw and caption paths.
                    self.apply_actions(&mut st, p.net_id, tick);
                    // #corpse-fall  LAND THE BODY. The film states no position after a body goes
                    // down, so a corpse left at its last reported position hangs in the air
                    // wherever that packet happened to be -- which is what "the ragdolls just
                    // float midair" is. It falls, from the death position to the floor found
                    // under it, at the engine's own measured gravity, as a pure function of the
                    // tick. A body already on the floor (the common case) has no entry and is
                    // untouched; a body with no floor within reach went over the edge and is
                    // left exactly where the recording put it, and the panel counts both.
                    if let (Some(age), Some((_, ft))) = (p.dead_ticks, fall) {
                        // ONE implementation, shared with the `film corpses trace` diagnostic and
                        // with the gate that asserts the fall -- so a test of the number is a
                        // test of what is drawn.
                        let z = self.corpse_drawn_z(p.net_id, p.pos, age);
                        if let Some(a) = actors.get_mut(slot) {
                            a.pos.z = z;
                        }
                        if age < ft {
                            self.corpses_falling += 1;
                        }
                    }
                    // #film-action  A decoded assassination overrides the derived gait AND the
                    // corpse: for the length of the clip both bodies are locked together on the
                    // attacker's own transform, and the victim only becomes a corpse after.
                    let mut sync_root: Option<([f32; 3], f32)> = None;
                    if let Some((sy, root)) = self.sync_at(pb, p.net_id, tick) {
                        st.stance = crate::film_anim::Stance::SyncAction;
                        st.gait = crate::film_anim::Gait::Idle;
                        st.seat = None;
                        st.dead_ticks = 0;
                        // The EVENT is decoded; the pose of the chosen move is only as decoded as
                        // the move is, which `describe` spells out.
                        // The EVENT is decoded only when the recording states it. An inferred
                        // one is DERIVED all the way down and must never print as decoded.
                        st.provenance = match (sy.stated, sy.move_decoded) {
                            (true, true) => crate::film_anim::Provenance::Decoded,
                            (true, false) => crate::film_anim::Provenance::Derived,
                            (false, _) => crate::film_anim::Provenance::Derived,
                        };
                        st.because = if sy.stated {
                            "the recording states this kill, both players and the tick"
                        } else {
                            "the recording does not state this kill: it is inferred from the damage and melee channels"
                        };
                        sync_root = root;
                        st.sync = Some(sy);
                        self.sync_shown += 1;
                        if let (Some(a), Some((pos, yaw))) = (actors.get_mut(slot), sync_root) {
                            a.pos = glam::Vec3::from(pos);
                            a.yaw = yaw;
                        }
                    }
                    if st.seat.is_some() {
                        rep.riders += 1;
                        // A rider is drawn on the VEHICLE's transform, not on its own replicated
                        // position. That is what the engine does — a seated unit's world matrix is
                        // the vehicle's times the seat marker, and the seat animation is authored
                        // in that frame — and the recorded position cannot substitute for it: a
                        // rider's own replicated origin sits 0.62-1.55 wu from the vehicle origin
                        // (measured, p05-p95 over 5,152 ride samples) because it is the body's
                        // feet, so drawing there puts the Spartan beside the mongoose rather than
                        // on it. Verified by capture.
                        if let Some(seat) = st.seat.clone() {
                            if let Some(w) = self.seat_world(&seat, pb, ftick, actors, native, cache) {
                                if let Some(a) = actors.get_mut(slot) {
                                    a.pos = w.0;
                                    a.yaw = w.1;
                                }
                            }
                        }
                    }
                    // The weapon the recording says is in this body's hands. Rare in the stream
                    // (about 0.13% of unit messages, i.e. when a player picks something up), so
                    // this changes a handful of times per clip and `set_weapon_tag` is a no-op
                    // when it has not.
                    let want_index = self
                        .track_of
                        .get(&p.net_id)
                        .and_then(|&i| pb.index.tracks.get(i))
                        .and_then(|tr| tr.weapon_at(tick))
                        .map(|w| w as usize);
                    let want_weapon = want_index.and_then(|w| self.weap_tags.get(w).copied());
                    weapons.push((slot, want_weapon));
                    if want_weapon.is_some() {
                        rep.armed += 1;
                    }
                    // #wclass  The animation set this body should move in. Known only when the
                    // recording named a weapon; otherwise the generic set, and the caption says
                    // which. Works the same whichever way the weapon arrived (a change-of-hands
                    // event or, later, a loadout out of the game variant), because both land in
                    // the track's own weapon field.
                    let want_class: Option<&str> =
                        want_index.and_then(|w| self.weap_class.get(w)).and_then(|c| c.as_deref());
                    if self.animate {
                        // The normalised cycle phase, integrated per gait along this body's own
                        // track -- NOT a distance times one global constant. See
                        // `film_anim::cycle_phase`.
                        let cls = want_class.unwrap_or(crate::film_anim::DEFAULT_WEAPON_CLASS);
                        let cid = match self.class_ids.iter().position(|c| c == cls) {
                            Some(i) => i as u16,
                            None => {
                                self.class_ids.push(cls.to_string());
                                (self.class_ids.len() - 1) as u16
                            }
                        };
                        // #anim-rate  Which GRAPH will be asked for the clip. Two species'
                        // clips of the same name have different authored lengths, so the frame
                        // count this body's rate is driven by has to be keyed on both.
                        let gpath: &str = self.biped_of.get(&p.net_id).map(|b| b.as_str()).unwrap_or(self.biped.as_str());
                        let gid = match self.graph_ids.iter().position(|g| g == gpath) {
                            Some(i) => i as u16,
                            None => {
                                self.graph_ids.push(gpath.to_string());
                                (self.graph_ids.len() - 1) as u16
                            }
                        };
                        let frames = &self.clip_frames;
                        // The phase is integrated over the body's WHOLE past with the class it is
                        // carrying NOW. A mid-life weapon swap therefore re-reads the earlier
                        // stretch at the new set's stride; the alternative (re-deriving which
                        // weapon was held at every past segment) would put a per-segment lookup
                        // back on the per-frame path, which is exactly the quadratic this was
                        // rescued from. The error is a phase offset at the swap, and the
                        // cross-fade covers it.
                        // #film-perf  Through the PREFIX, which is the same arithmetic in the
                        // same order (see `phase_prefix`). The walk is still what builds it, so
                        // there is one expression for the phase and not two.
                        let phase = match self.track_of.get(&p.net_id).and_then(|&i| pb.index.tracks.get(i)) {
                            Some(tr) => {
                                let frames_of = |st: crate::film_anim::Stance, ga: crate::film_anim::Gait| {
                                    frames.get(&(gid, cid, st as u8, ga as u8)).copied()
                                };
                                let key = (gid, cid, frames.len(), tr.positions.len());
                                let fresh = self.phase_prefix.get(&p.net_id).is_some_and(|(k, _)| *k == key);
                                if !fresh {
                                    let pre = crate::film_anim::cycle_phase_prefix(tr, &frames_of);
                                    self.phase_prefix.insert(p.net_id, (key, pre));
                                }
                                match self.phase_prefix.get(&p.net_id) {
                                    Some((_, pre)) => crate::film_anim::cycle_phase_at(tr, tick, pre, &frames_of)
                                        // A prefix that does not match the track is refused by
                                        // `cycle_phase_at`; fall back to the walk rather than pose
                                        // the body from a number nothing checked.
                                        .unwrap_or_else(|| crate::film_anim::cycle_phase_with(tr, tick, &frames_of)),
                                    None => crate::film_anim::cycle_phase_with(tr, tick, &frames_of),
                                }
                            }
                            None => 0.0,
                        };
                        want_anim.push((
                            slot,
                            crate::film_anim::anim_candidates(&st, cls),
                            phase,
                            st.seat.is_some(),
                            (gid, cid, st.stance as u8, st.gait as u8),
                            oneshot_for(
                                st.sync.as_ref().map(|sy| sy.age),
                                p.dead_ticks.map(|age| match fall {
                                    Some((_, ft)) if age >= ft => age - ft,
                                    _ => age,
                                }),
                                st.land_ticks,
                                st.air_ticks,
                            ),
                        ));
                        // #aim / #fire  What to compose on top. The aim is only ever stated for
                        // ONE body per recording (see `aim_offset`); the firing overlay is stated
                        // for every body, because `action_weapon_fire` is replicated.
                        want_overlay.push((
                            slot,
                            crate::film_anim::aim_candidates(&st, cls),
                            self.aim_offset(pb, p.net_id, tick, tr.and_then(|t| t.facing_at(tick))),
                            crate::film_anim::fire_candidates(cls),
                            Self::action_age(&self.fire_of, p.net_id, tick, FIRE_TICKS),
                        ));
                        // #melee  The swing, in the REPLACEMENT lane. A death or a synchronised
                        // action already owns the whole skeleton, so a swing never applies there.
                        want_replace.push((
                            slot,
                            match (&st.sync, p.dead_ticks, st.melee_ticks) {
                                (None, None, Some(age)) => Some((
                                    crate::film_anim::melee_candidates(cls, st.stance == crate::film_anim::Stance::Flight, st.melee_variant),
                                    age,
                                )),
                                _ => None,
                            },
                        ));
                    }
                }
                None => {
                    a.pos = glam::Vec3::from(self.park);
                    rep.parked += 1;
                }
            }
        }
        self.weapons_shown = weapons.iter().filter(|(_, t)| t.is_some()).count();
        // Put the recorded weapon in each body's hands. `set_weapon_tag` returns early when the
        // actor is already holding it, so this costs nothing on a steady frame.
        for (slot, tag) in weapons {
            if let Err(e) = actors.set_weapon_tag(native, cache, slot, tag) {
                self.weapon_error = Some(e);
            }
        }
        // Load an animation only when it changes (set_anim reads a resource), then drive the
        // cycle's PHASE from the distance travelled so the feet match the ground.
        self.corpses_shown = 0;
        self.class_bodies = 0;
        self.class_fallbacks = 0;
        for (slot, cands, phase, seated, key, dead) in want_anim {
            let current = self.anim_of.get(&slot).cloned();
            if current.as_deref() != cands.first().map(|s| s.as_str()) {
                // Try the candidates in order: seat sets are named inconsistently across
                // vehicles (`warthog_d:any:idle` but `ghost_d:idle`), so the first name the
                // graph actually has wins.
                for c in &cands {
                    if actors.set_anim(native, cache, slot, Some(c)).is_ok() {
                        self.anim_of.insert(slot, cands[0].clone());
                        break;
                    }
                }
            }
            // #wclass  Did the body actually get its own weapon's set, or did the graph have no
            // clip for that class and the generic tail win? Counted, never assumed.
            let asked = self.class_ids.get(key.1 as usize).cloned().unwrap_or_default();
            if let Some(a) = actors.get(slot) {
                let got = crate::film_anim::class_of_clip(&a.anim_name).unwrap_or("");
                if key.1 != 0 {
                    if got == asked {
                        self.class_bodies += 1;
                    } else {
                        self.class_fallbacks += 1;
                    }
                }
            }
            if let Some(a) = actors.get_mut(slot) {
                a.playing = false; // the phase is ours, not a free-running clock
                let n = a.frame_count().max(1.0);
                // #death A death clip plays ONCE and then HOLDS its last frame — it is not a
                // cycle, so wrapping its phase would make the body fall over again and again.
                // The frame comes straight from the age of the death in ticks (60 Hz ticks ->
                // 30 fps frames), which makes the whole fall a pure function of the playback
                // tick: scrub backwards and the body stands up again exactly as it fell.
                if let Some((age, kind)) = dead {
                    if kind == ONESHOT_SYNC {
                        // #film-action  The same one-shot rule, for the two-party clip.
                        if !crate::film_anim::is_sync_clip(&a.anim_name) {
                            self.sync_clip_missing = true;
                        }
                        self.sync_clips.push(format!("{} @{:.0}/{:.0}", a.anim_name, a.frame, n));
                    } else if kind == ONESHOT_DEATH {
                        self.corpses_shown += 1;
                        let got = crate::film_anim::is_death_clip(&a.anim_name);
                        if !got {
                            self.death_clip_missing = true;
                        }
                        self.death_clips.push(format!(
                            "{} @{:.0}/{:.0}{}",
                            if a.anim_name.is_empty() { "<no clip>" } else { a.anim_name.as_str() },
                            a.frame,
                            n,
                            if got { "" } else { "  <-- NOT a death clip: this graph has none of them" }
                        ));
                    }
                    // #fp-shake  The sub-tick part of the playhead carries into the one-shot's
                    // age, so a death fall, an assassination, a landing and a jump arc all
                    // advance at the DISPLAY rate rather than stepping at the tick rate. Off the
                    // rounded tick these judder above 60 Hz and below 1x speed exactly as the
                    // first-person arms did.
                    a.frame = ((age as f32 + (pb.tick - tick as f32)) * 0.5).min(n - 0.001).max(0.0);
                    continue;
                }
                // Learn the clip's REAL length for this state, so the stride rate is keyed on the
                // clip rather than on the gait's assumed rifle-set frame count.
                self.clip_frames.insert(key, n);
                // A seated pose has no stride to drive it: hold frame 0. Otherwise the NORMALISED
                // phase is scaled into whatever frame count the clip that is actually loaded has,
                // which is what carries a foot mid-stride across a change of gait.
                a.frame = if seated { 0.0 } else { (phase.rem_euclid(1.0) * n).min(n - 0.001).max(0.0) };
            }
        }
        // #melee  Install the REPLACEMENT lane, AFTER every base clip is chosen and framed: the
        // swing overwrites the arms and torso of the pose the locomotion clip produced and leaves
        // the legs on it, which is the whole point of the lane (see `Actor::replacement`).
        self.melee_shown = 0;
        self.melee_bodies.clear();
        // Clear every slot first: a body that is not swinging this frame must have no replacement
        // left on it, and that has to hold even for a slot `want_replace` has no entry for (a
        // parked actor, or any body at all while the locomotion switch is off).
        let all_slots: Vec<u32> =
            self.slots.iter().copied().chain(self.species_slots.values().flatten().copied()).collect();
        for slot in all_slots {
            if let Some(a) = actors.get_mut(slot) {
                a.replacement = None;
                a.replacement_name.clear();
            }
        }
        for (slot, want) in want_replace {
            let Some(a) = actors.get_mut(slot) else { continue };
            let Some((cands, age)) = want else {
                a.replacement = None;
                a.replacement_name.clear();
                continue;
            };
            let mut got = false;
            for c in &cands {
                // The lane only accepts a type-3 REPLACEMENT clip, so a graph whose melee is
                // authored some other way is reported rather than posed wrongly.
                if let Some((anim, n)) = a.replacement_clip(native, cache, c) {
                    // 60 Hz film ticks -> 30 fps frames, played once and then held, exactly like
                    // a death clip: scrub back and the swing un-swings.
                    a.replacement = Some((anim, (age as f32 / 2.0).min(n - 0.001).max(0.0)));
                    a.replacement_name = c.clone();
                    // How much of the skeleton the swing actually carries -- the number the bug
                    // was about, measured off the clip rather than assumed. A replacement that
                    // covered the whole body would not need this lane at all.
                    let animates = a.clip_animates_nodes(anim);
                    self.melee_nodes = (animates.iter().filter(|b| **b).count(), animates.len());
                    got = crate::film_anim::is_melee_clip(c);
                    break;
                }
            }
            if got {
                self.melee_shown += 1;
                if let Some(a) = actors.get(slot) {
                    self.melee_bodies.push((a.pos.to_array(), a.replacement_name.clone()));
                }
            } else {
                a.replacement = None;
                a.replacement_name.clear();
                self.melee_clip_missing = true;
            }
        }
        // #aim / #fire  Compose the overlays, AFTER every base clip is chosen and framed: an
        // overlay is added to the pose the locomotion clip produced, which is what makes a look
        // direction a look direction rather than a pose that cancels the stride.
        self.aim_shown = 0;
        self.aim_missing = 0;
        self.fire_shown = 0;
        for (slot, aim_cands, aim, fire_cands, fire_age) in want_overlay {
            let Some(a) = actors.get_mut(slot) else { continue };
            if !a.overlays.is_empty() {
                a.overlays.clear();
            }
            // A body with no aim CANDIDATES has nothing to pose and is not a failure: a dead
            // body, a rider and a body inside a synchronised action all return an empty list
            // because the clip they are playing already owns the whole skeleton.
            if let (Some((dy, dp)), false) = (aim, aim_cands.is_empty()) {
                let mut got = false;
                for c in &aim_cands {
                    if let Some((anim, grid)) = a.aim_grid(native, cache, c) {
                        let cells = grid.cells(dy, dp);
                        if !cells.is_empty() {
                            a.overlays.push((anim, cells));
                            got = true;
                        }
                        break;
                    }
                }
                if got {
                    self.aim_shown += 1;
                } else {
                    self.aim_missing += 1;
                }
            }
            if let Some(age) = fire_age {
                for c in &fire_cands {
                    if let Some((anim, n)) = a.clock_overlay(native, cache, c) {
                        let f = age as f32 / 2.0;
                        if f < n {
                            a.overlays.push((anim, vec![(f, 1.0)]));
                            self.fire_shown += 1;
                        }
                        break;
                    }
                }
            }
        }
        // #fp The first-person rig, placed on the camera this same frame.
        self.sync_first_person(pb, actors, native, cache, tick);
        self.prof.lap(4); // #film-perf  the bodies, their animation picks and the rig
        self.prof.end();
        rep
    }

    /// #skin  Give every drawn body the armour and the colours its player actually had.
    ///
    /// ONE implementation, on the driver, because BOTH hosts need it and they used to disagree:
    /// this lived only in the GUI, so a headless capture of a film drew every body in its DEFAULT
    /// armour while the window drew the recorded one. A difference like that makes the scripted
    /// path useless for checking what the user is actually looking at -- which is exactly how an
    /// "Elites are invisible" report could not be reproduced from a script.
    ///
    /// Two halves, both per player slot:
    ///   * the ten appearance bytes become a synthetic model VARIANT (region -> permutation picks
    ///     out of `globals\player_model_customization`), installed on the scene once; the drawn
    ///     actor then names that variant and the ordinary variant machinery selects its sections
    ///     and its instanced attachments;
    ///   * the two colour bytes index `matg` "profile colors" and become the actor's primary and
    ///     secondary change colours.
    ///
    /// A body the recording could not tie to a player keeps the model's own defaults -- the
    /// picture never invents an appearance for someone it cannot name. Returns whether anything
    /// changed, i.e. whether the actor meshes need rebuilding.
    pub fn apply_recorded_appearance(
        &mut self,
        pb: &Playback,
        film: &crate::film::Film,
        sc: &mut crate::scene::SceneController,
        actors: &mut crate::reach_actor::ActorScene,
    ) -> bool {
        // Build the per-slot table once per film: the synthetic variants go to the scene (which
        // drops the decodes that depend on them), the colours stay here.
        if self.appearance.is_none() {
            // The table is read out of the open map (`globals\player_model_customization` and
            // `matg` "profile colors"). While the map is still loading both come back empty, and
            // caching that would leave every body in its default armour for the rest of the
            // session -- so wait for a readable cache instead of remembering the empty answer.
            if sc.profile_colors().is_empty() {
                return false;
            }
            let mut variants: Vec<crate::scene::PlayerVariant> = Vec::new();
            let mut table: HashMap<u8, (u32, Option<[[f32; 3]; 2]>)> = Default::default();
            for p in film.header.players.iter().filter(|p| p.present()) {
                let Some(a) = p.appearance() else { continue };
                let sid = crate::scene::player_variant_sid(p.index as u8);
                // #elite BOTH species' pick lists: a player has a Spartan loadout and an Elite
                // loadout, and the render model the body is drawn from decides which one applies.
                let v = sc.player_variant(sid, &a.slots);
                let worn = v.worn();
                if worn {
                    variants.push(v);
                }
                let cols = sc.armour_colors(a.color_primary, a.color_secondary);
                // sid 0 = "no synthetic variant", which leaves the model's own default on.
                table.insert(p.index as u8, (if worn { sid } else { 0 }, cols));
            }
            sc.set_player_variants(variants);
            self.appearance = Some(table);
        }
        let Some(table) = self.appearance.as_ref() else { return false };
        let mut dirty = false;
        for (&actor_id, &net) in self.pool.assigned.iter() {
            let want = pb.index.slot_of_entity(net).and_then(|s| table.get(&s)).copied();
            let Some(a) = actors.get_mut(actor_id) else { continue };
            let (vsid, skin) = want.unwrap_or((0, None));
            // #skin  An actor is RECYCLED between bodies, so "this body has no recorded
            // appearance" has to CLEAR the previous occupant's armour rather than leave it on --
            // that is how one player's armour could follow a pooled actor onto somebody else's
            // body, or onto a body of the other species.
            //
            // But it must only clear a PLAYER variant. `build_actor_meshes` legitimately writes
            // the model's own default variant into `variant_sid` whenever it finds it unset, so
            // comparing against a bare 0 made the two fight: this cleared it, the build resolved
            // it again, which marked the actors dirty, which rebuilt every actor mesh -- EVERY
            // FRAME, for as long as any body had no recorded appearance. That is a full mesh
            // rebuild per frame and it is what made Theater playback crawl. The player variants
            // are a reserved range (`player_variant_sid` = 0xF1A1_0000 | slot), so "is the actor
            // wearing somebody's recorded armour" is a question with an exact answer.
            let wearing_recorded = a.variant_sid & 0xFFFF_0000 == 0xF1A1_0000;
            if vsid != 0 {
                if a.variant_sid != vsid {
                    a.variant_sid = vsid;
                    a.variant_name = "recorded armour".into();
                    dirty = true;
                }
            } else if wearing_recorded {
                a.variant_sid = 0;
                a.variant_name.clear();
                dirty = true;
            }
            if a.skin != skin {
                a.skin = skin;
                dirty = true;
            }
        }
        // #fp-arms  AND THE FIRST-PERSON ARMS, in the same player's colours.
        //
        // The arms are not pooled bodies, so the loop above never reached them and they were drawn
        // in the arms model's own authored default -- a plain black glove under a third-person
        // body wearing that player's tan and green. The colours are the ones already resolved for
        // this slot (`matg` "profile colors" through `SceneController::armour_colors`), applied
        // through the render model's own `four_change_color` masks exactly as a body's are, so
        // there is one table and one rule for both. The arms' PIECES are a separate install
        // (`dress_first_person`): these are bare `mode` tags with no `hlmt`, so the variant walk
        // cannot start on them and the selections go in as a section mask instead.
        let fp_arms = self
            .fp_reps
            .iter()
            .find(|r| r.elite == self.fp_subject_elite)
            .or_else(|| self.fp_reps.first())
            .and_then(|r| self.fp_actor_of.get(&r.hands).copied());
        if let Some(id) = fp_arms {
            let skin = pb.subject_slot().and_then(|s| table.get(&s)).and_then(|&(_, c)| c);
            if let Some(a) = actors.get_mut(id) {
                if a.skin != skin {
                    a.skin = skin;
                    dirty = true;
                }
            }
        }
        dirty
    }

    /// #fp-arms  Resolve the cache's `@player representation` rows and install each arms model's
    /// section pick. Call once per cache; a no-op after that.
    ///
    /// **ONE implementation, here, because there are two hosts.** The GUI and the script runner
    /// each keep their own `ActorDriver`, and every step installed on only one of them silently
    /// changes the answer — this one was GUI-only, so a scripted first-person seek reported "this
    /// cache has no `@player representation`" while the GUI drew arms.
    ///
    /// Returns true when the rows landed, so the caller can stop asking.
    pub fn resolve_first_person(&mut self, sc: &mut crate::scene::SceneController) -> bool {
        let Some(cache) = sc.cache_handle() else { return false };
        let rows = sc.first_person_models();
        if rows.is_empty() {
            return false;
        }
        let native = sc.native_dll();
        let mut reps: Vec<FpRep> = Vec::new();
        let mut masks: Vec<(u32, Vec<bool>)> = Vec::new();
        for (_, h, b) in rows {
            reps.push(FpRep { hands: h, body: b, elite: sc.model_is_elite(h) });
            for t in [h, b].into_iter().filter(|t| *t != 0xFFFF_FFFF) {
                let m = crate::reach_actor::default_section_mask(native, cache, t);
                if !m.is_empty() {
                    masks.push((t, m));
                }
            }
        }
        self.fp_reps = reps;
        for (t, m) in masks {
            sc.set_section_pick(t, m);
        }
        true
    }

    /// #fp-arms  Dress the first-person arms in ONE recorded player's own armour, and say whether
    /// anything changed.
    ///
    /// The arms are bare `mode` tags with no `hlmt`, so the variant walk `build_actor_meshes`
    /// uses cannot start on them and the selections have to be installed as an explicit section
    /// mask keyed by render model — which means one player at a time, exactly right for a view
    /// through one player's eyes. `already` is the `(hands tag, slot)` the caller last installed,
    /// so a steady frame costs one comparison.
    pub fn dress_first_person(
        &self,
        sc: &mut crate::scene::SceneController,
        slot: u8,
        appearance: &[u8],
        already: Option<(u32, u8)>,
    ) -> Option<(u32, u8)> {
        let cache = sc.cache_handle()?;
        let elite = self.fp_subject_is_elite_slot(slot);
        let rep = *self.fp_reps.iter().find(|r| r.elite == elite).or_else(|| self.fp_reps.first())?;
        if already == Some((rep.hands, slot)) {
            return already;
        }
        let v = sc.player_variant(crate::scene::player_variant_sid(slot), appearance);
        let picks = if rep.elite { v.elite_picks } else { v.picks };
        let mask = crate::reach_actor::section_mask_with_picks(sc.native_dll(), cache, rep.hands, &picks);
        if mask.is_empty() {
            return already;
        }
        sc.set_section_pick(rep.hands, mask);
        Some((rep.hands, slot))
    }

    /// #fp-arms  Is the body this player slot is driving an ELITE? From the biped the recording
    /// names for it, never from the slot.
    fn fp_subject_is_elite_slot(&self, slot: u8) -> bool {
        let _ = slot;
        self.fp_subject_elite
    }

    /// #fp-anim  Choose the first-person BASE clip, its one-shot age, and the OVERLAYS to
    /// compose on it, from one body's decoded state. Pure, so it can be gated against the real
    /// graph for every combination of inputs.
    ///
    /// The split is the tag's, not a preference. In a Reach first-person graph
    /// `first_person:idle`, `:sprint_loop`, `:fire_1:var*`, `:melee_strike_*`, `:reload_*` and
    /// `:ready` are animation type 1 (BASE) while `:moving`, `:fall`, `:jump`, `:land_hard`,
    /// `:land_soft`, `:look` and `:acceleration_screens` are type 2 (OVERLAY) -- measured on the
    /// Spartan and Elite graph of every weapon in `forge_halo`. An overlay composes onto whatever
    /// pose it is given, so handing one to `set_anim` as a body's ONLY clip composes it onto the
    /// skeleton's authored default and draws the arms' bind pose, which is both arms straight
    /// out. That is what this used to do for `:moving`, and a player in a 16-player clip is
    /// moving nearly every frame.
    ///
    /// The two DECODED actions outrank the gait, in the engine's own precedence: a swing and a
    /// shot each replace the arms outright (first-person `fire_1` is a BASE clip here, unlike the
    /// third-person `combat:<class>:fire_1`, which is an overlay on a body that still has to
    /// walk). Both are one-shots played from the recorded event tick, so scrubbing undoes them.
    pub(crate) fn fp_clips(
        melee: Option<u32>,
        fired: Option<u32>,
        sprinting: bool,
        airborne: bool,
        moving: bool,
    ) -> (Vec<&'static str>, Option<u32>, Vec<&'static str>) {
        // A CANDIDATE LIST, best first, because not every weapon authors every action: the
        // energy sword's first-person graph has no `fire_1` at all -- a sword's trigger IS its
        // lunge (`first_person:melee_lunge`) -- so a single hard-coded name silently failed to
        // load on every melee weapon and left the arms on whatever they were already playing.
        // The caller takes the first candidate the open graph actually has.
        let (base, oneshot): (Vec<&'static str>, Option<u32>) = match (melee, fired) {
            // Which of the two authored swings, derived from the event's own tick so it replays
            // identically -- the same rule the third-person swing uses.
            (Some(age), _) => (
                if (age / 7) % 2 == 0 {
                    vec!["first_person:melee_strike_1", "first_person:melee_strike_2", "first_person:melee_lunge"]
                } else {
                    vec!["first_person:melee_strike_2", "first_person:melee_strike_1", "first_person:melee_lunge"]
                },
                Some(age),
            ),
            (None, Some(age)) => (
                vec!["first_person:fire_1:var1", "first_person:melee_lunge", "first_person:melee_strike_1"],
                Some(age),
            ),
            (None, None) if sprinting => (vec!["first_person:sprint_loop", "first_person:idle"], None),
            _ => (vec!["first_person:idle"], None),
        };
        let mut overlays: Vec<&'static str> = Vec::new();
        // Not while a one-shot owns the arms: the engine does not walk a swing.
        if oneshot.is_none() {
            if airborne {
                overlays.push("first_person:fall");
            } else if moving && !sprinting {
                overlays.push("first_person:moving");
            }
        }
        (base, oneshot, overlays)
    }

    /// #fp Put Reach's FIRST-PERSON ARMS on the camera, or park them.
    ///
    /// The engine does not show you the inside of your own helmet: a player looking through their
    /// own eyes sees a dedicated arms model, and the weapon in their hands is a different tag
    /// from the one on the ground. Both are wired here.
    ///
    /// * The arms are `matg` +0x474 `@player representation` -> `first person hands` / `first
    ///   person body`, two bare `mode` tags with no object, no `hlmt` and no animation graph.
    /// * The weapon is `weap` +0x3B8 `player interface` -> `first person` -> `first person model`,
    ///   authored at first-person scale (which is the real reason a Reach gun looks smaller in
    ///   your own hands; it is not the ground model scaled down).
    ///
    /// **They are drawn in BIND POSE, and the caption says so.** The first-person animations
    /// belong to the WEAPON (`first person animations`, the tagref beside the model), so a body
    /// whose weapon the recording never named has no authored first-person pose to play at all —
    /// and that is almost every body, because a film reports a weapon only when it changes hands.
    ///
    /// The rig is mounted on the solved camera, so it inherits whatever that view honestly is:
    /// the recorder's own aim where there is one, the body's decoded facing where there is not,
    /// and `look_source` names which. Nothing here invents a look direction.
    ///
    /// No per-frame mesh work: an actor's meshes are built once and its bone palette folds the
    /// placement delta in, so following the camera costs a matrix multiply.
    fn sync_first_person(
        &mut self,
        pb: &Playback,
        actors: &mut crate::reach_actor::ActorScene,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        tick: u32,
    ) {
        let park = glam::Vec3::from(self.park);
        // What each model needs: where it goes (position, yaw deg, pitch deg, model yaw deg),
        // which graph poses it, and which clip of that graph. Empty unless a first-person camera
        // is live AND it has solved a pose.
        struct Fp {
            mode: u32,
            place: (glam::Vec3, f32, f32, f32),
            graph: Option<u32>,
        }
        let mut want: Vec<Fp> = Vec::new();
        let mut weapon_note = String::new();
        let mut clip: Option<&'static str> = None;
        // #fp-anim  The BASE clip candidates (best first; the first one the open graph has wins),
        // the OVERLAY clips composed on top, and the age in ticks of a one-shot base (a swing or
        // a shot) when one owns the arms.
        let mut fp_bases: Vec<&'static str> = Vec::new();
        let mut fp_overlays: Vec<&'static str> = Vec::new();
        let mut fp_oneshot: Option<u32> = None;
        // #film-camera  The ARMS follow the resolved camera, not the chosen mode: a dead or
        // orbiting shot is a third-person view and must not have a pair of hands mounted on it.
        let live = pb.view_is_first_person();
        // #fp-arms  WHICH species' arms. The camera's subject is a specific recorded body and the
        // biped the recording named for it says what it is, so an Elite gets the Elite row.
        let subject_elite = pb
            .hidden_entity()
            .and_then(|net| self.biped_of.get(&net))
            .is_some_and(|b| crate::film_anim::is_elite(b));
        self.fp_subject_elite = subject_elite;
        let rep = self
            .fp_reps
            .iter()
            .find(|r| r.elite == subject_elite)
            .or_else(|| self.fp_reps.first())
            .copied();
        // #crosshair  Cleared every frame and republished below only while a first-person view is
        // actually being posed, so a stale weapon can never outlive the camera that chose it.
        self.fp_weapon_tag = None;
        if live {
            if let (Some(FpRep { hands, body, .. }), Some(p)) = (rep, solve_camera_at(pb, self.eye_height(pb))) {
                let eye = glam::Vec3::from(p.pos);
                let view = (eye, p.yaw.to_degrees(), p.pitch.to_degrees(), crate::reach_actor::FP_ARMS_MODEL_YAW);
                let subject = pb.hidden_entity().and_then(|net| {
                    self.track_of.get(&net).and_then(|&i| pb.index.tracks.get(i))
                });
                // The weapon this body is carrying, and with it the FIRST-PERSON graph that poses
                // both the arms and the gun. In Reach the first-person "model" of a weapon is the
                // weapon's OWN render model -- measured, see the note -- and what makes the view
                // first person is the separate `first person animations` graph beside it.
                let held = subject
                    .and_then(|tr| tr.weapon_at(tick))
                    .and_then(|w| self.weap_tags.get(w as usize).copied());
                // #fp-species  Row 0 of the weapon's `player interface` is the Spartan's
                // first-person model and graph, row 1 the Elite's; the body the recording names
                // says which. Cached per (weapon, species), because the same weapon in an Elite's
                // hands is a different graph entirely.
                // #crosshair  Publish it: the reticule is authored on this very weapon.
                self.fp_weapon_tag = held;
                let species = usize::from(subject_elite);
                let fp = match held {
                    Some(w) => {
                        match self.fp_weapon_of.get(&(w, species)) {
                            Some(v) => *v,
                            None => {
                                let v = crate::reach_actor::first_person_weapon(native, cache, w, species);
                                self.fp_weapon_of.insert((w, species), v);
                                v
                            }
                        }
                    }
                    None => (None, None),
                };
                let (fp_model, fp_graph) = fp;
                // #fp-anim  Which first-person clip, and WHICH KIND of clip.
                //
                // A Reach first-person graph is a BASE clip plus OVERLAYS, and the tag says which
                // each one is: `first_person:idle`, `:sprint_loop`, `:fire_1:var*`,
                // `:melee_strike_*`, `:reload_*`, `:ready` are type 1 (BASE), while
                // `first_person:moving`, `:fall`, `:jump`, `:land_hard`, `:land_soft`, `:look`
                // and `:acceleration_screens` are type 2 (OVERLAY). Measured on `forge_halo`,
                // on both the Spartan and the Elite graph of every weapon.
                //
                // That distinction is the whole bug. This used to hand `first_person:moving`
                // (an OVERLAY) to `set_anim` as the body's only clip, and `clip_trs` composes an
                // overlay onto the skeleton's AUTHORED DEFAULT -- so the view became the arms'
                // bind pose, which is both arms straight out, plus a small delta. A player in a
                // 16-player clip is moving almost all the time, so almost every frame drew that
                // T-pose: "the arms just stick straight out, none of the animations are working".
                // The base is now always a real BASE clip and `moving` / `fall` ride on top of it
                // as the overlays they are, exactly as the engine composes them.
                //
                // The two DECODED actions take the base over the gait, in the engine's own
                // precedence: a melee swing replaces the arms outright, and so does a shot --
                // first-person `fire_1` is a type 1 BASE clip here, unlike the third-person
                // `combat:<class>:fire_1`, which is an overlay on a body that still has to walk.
                let st = subject.map(|tr| crate::film_anim::derive(tr, tick, None));
                let net = pb.hidden_entity();
                let melee = net.and_then(|n| Self::action_age(&self.melee_of, n, tick, MELEE_TICKS));
                let fired = net.and_then(|n| Self::action_age(&self.fire_of, n, tick, FIRE_TICKS));
                let sprinting = matches!(st.as_ref().map(|s| s.stance), Some(crate::film_anim::Stance::Sprint));
                let airborne = matches!(st.as_ref().map(|s| s.stance), Some(crate::film_anim::Stance::Flight));
                let moving = st.as_ref().is_some_and(|s| s.gait != crate::film_anim::Gait::Idle);
                // `(clip, the one-shot age in ticks)`. A one-shot is played from its event tick
                // and then holds, so scrubbing backwards un-plays it, exactly like the
                // third-person swing.
                let (bases, oneshot, overlays) = Self::fp_clips(melee, fired, sprinting, airborne, moving);
                fp_bases = bases;
                fp_oneshot = oneshot;
                fp_overlays = overlays;
                // Only draw the arms when something can POSE them. Their bind pose is a T-pose
                // with both arms straight out along +Y, which mounted on the camera fills the
                // whole view with forearm -- worse than drawing nothing, and a claim the
                // recording does not support. The pose comes from the weapon's first-person
                // graph, so no weapon means no arms, and the caption says exactly that.
                if hands != 0xFFFF_FFFF && fp_graph.is_some() {
                    want.push(Fp { mode: hands, place: view, graph: fp_graph });
                    // #fp-gun  And the WEAPON, on the same rig. A Reach weapon's `first person
                    // model` is its ordinary render model, and the first-person graph lists that
                    // model's own nodes (`b_handle`/`b_blades` for the sword, `b_gun`/
                    // `b_magazine`/`b_ophandle`/`b_safety`/`b_switch` for a rifle) beside the
                    // arms' -- one clip poses both, in one space. So the gun mounts exactly where
                    // the hands do, with the same graph, clip and frame, and needs no marker
                    // attach at all. It used to come out as a blade stretched across the view
                    // because the graph-to-model node match was scored against the GRAPH's node
                    // count: a weapon model resolves ~100% of its own handful of nodes but only a
                    // tenth of the graph's, failed the threshold, and fell back to matching by
                    // node POSITION, which on these two lists drives every bone with the wrong
                    // channel. With the match scored on either side, the weapon maps and the
                    // ordinary skinned path draws it.
                    if let Some(m) = fp_model {
                        want.push(Fp { mode: m, place: view, graph: fp_graph });
                    }
                }
                // #fp The first-person BODY is resolved and deliberately NOT drawn, and that is
                // a measured decision rather than an omission. It is an ordinary biped model
                // standing at the player's feet (origin at the feet, +X forward, pelvis at
                // z=0.380) and the first-person camera sits 0.55 wu up, INSIDE its chest; nothing
                // poses it either, because it has no part in the weapon's first-person graph and
                // its own locomotion lives on the third-person biped. Drawn, it put a blown-out
                // wall of unposed torso across the middle of every shot. The third-person body is
                // already hidden for the camera's own subject, so the view is honest without it.
                let _ = body;
                weapon_note = match (held, fp_model, fp_graph) {
                    (Some(_), Some(m), Some(_)) => format!(
                        ", holding {} \u{2014} the gun and the hands are posed by that weapon's \
                         OWN first-person graph, which lists both models' nodes and drives them \
                         from one clip, so the grip needs no marker attach",
                        native.tag_name(cache, m).unwrap_or_default()
                    ),
                    (Some(w), _, _) => format!(
                        ", but {} authors no first-person interface, so there is no authored pose \
                         for the arms",
                        native.tag_name(cache, w).unwrap_or_default()
                    ),
                    _ => ", but nothing names a weapon for this body, so there is no authored pose \
                          for the arms"
                        .into(),
                };
            }
        }
        // #fp-anim  What each first-person model ended up playing: (model tag, clip name, frame,
        // frame count, live node-map entries, model node count).
        let mut fp_live: Vec<(u32, String, f32, f32, Option<usize>, usize)> = Vec::new();
        // Create what is missing, place what is wanted, park the rest. NEVER destroy: destroying
        // an actor rebuilds every mesh in the scene.
        let mut failed: Option<String> = None;
        for f in &want {
            if self.fp_actor_of.contains_key(&f.mode) {
                continue;
            }
            match actors.add_render_model(native, cache, f.mode, f.graph, park, 0.0) {
                Ok(id) => {
                    self.fp_actor_of.insert(f.mode, id);
                }
                Err(e) => {
                    failed = Some(format!("{}: {e}", native.tag_name(cache, f.mode).unwrap_or_default()));
                }
            }
        }
        let placed: Vec<(u32, u32, Option<u32>, (glam::Vec3, f32, f32, f32))> = self
            .fp_actor_of
            .iter()
            .map(|(&m, &id)| {
                let w = want.iter().find(|f| f.mode == m);
                (m, id, w.and_then(|f| f.graph), w.map(|f| f.place).unwrap_or((park, 0.0, 0.0, 0.0)))
            })
            .collect();
        for (m, id, graph, place) in placed {
            let wanted = want.iter().any(|f| f.mode == m);
            if wanted {
                // The graph can change under a model (a weapon swap), and swapping it rebuilds
                // nothing: the model is the same, only what poses it changes.
                actors.set_model_graph(native, cache, id, graph);
                // #fp-anim  The first candidate THIS graph actually has. Resolved per actor,
                // because the arms and the weapon ride the same graph but a graph that lacks the
                // clip must fall back rather than leave the rig on a stale pose.
                let picked = actors
                    .get(id)
                    .and_then(|a| a.def.graph.as_ref())
                    .and_then(|g| fp_bases.iter().copied().find(|c| g.find(c).is_some()));
                if clip.is_none() {
                    clip = picked;
                }
                if let Some(c) = picked.filter(|_| graph.is_some()) {
                    if actors.get(id).map(|a| a.anim_name.as_str()) != Some(c) {
                        let _ = actors.set_anim(native, cache, id, Some(c));
                    }
                }
            }
            if let Some(a) = actors.get_mut(id) {
                if wanted {
                    a.pos = place.0;
                    a.yaw = place.1;
                    a.pitch = place.2;
                    a.model_yaw = place.3;
                    // #fp-anim  Report what the rig is ACTUALLY doing, read back off the actor
                    // itself rather than off the intent above. "The arms stick straight out" is a
                    // bind pose, and a bind pose means either no clip loaded (`set_anim`'s error
                    // is swallowed, because a missing clip must not stop the frame) or a node map
                    // that reaches none of this model's bones. Both are invisible unless stated.
                    // #fp-anim  Compose the gait OVERLAYS on the base clip. `clock_overlay`
                    // REFUSES a clip the tag does not mark as an overlay, so a base clip can
                    // never be installed here by mistake; each one runs on the same playback
                    // clock as the base, so scrubbing undoes it too.
                    a.overlays.clear();
                    for name in &fp_overlays {
                        if let Some((anim, frames)) = a.clock_overlay(native, cache, name) {
                            let f = clock_frame(pb.tick, frames);
                            a.overlays.push((anim, vec![(f, 1.0)]));
                        }
                    }

                    // The first-person clips are not stride-driven, so they run on the CLOCK --
                    // but the clock has to be the PLAYBACK tick, or scrubbing back would not undo
                    // the pose. Frames are 30 Hz against the film's 60 Hz ticks.
                    a.playing = false;
                    let n = a.frame_count().max(1.0);
                    // #fp-shake  Off the FLOAT playhead, not the rounded tick — see
                    // `clock_frame`. The one-shot carries the same sub-tick offset so a swing is
                    // as smooth as the walk it is laid over.
                    let sub = pb.tick - tick as f32;
                    a.frame = match fp_oneshot {
                        // #fp-anim  A swing or a shot plays ONCE from the tick the recording says
                        // it happened and then HOLDS on its last frame, so scrubbing backwards
                        // un-plays it. Same rule as the third-person one-shots.
                        Some(age) => ((age as f32 + sub) * 0.5).min(n - 1.0).max(0.0),
                        None => clock_frame(pb.tick, n),
                    };
                    // #fp-anim  Report what the rig is ACTUALLY doing, read back off the actor
                    // AFTER it has been posed rather than off the intent above. "The arms stick
                    // straight out" is a bind pose, and a bind pose means either no clip loaded
                    // (`set_anim`'s error is swallowed, because a missing clip must not stop the
                    // frame) or a node map that reaches none of this model's bones.
                    fp_live.push((
                        m,
                        a.anim_name.clone(),
                        a.frame,
                        n,
                        a.def.anim_node_map.as_ref().map(|v| v.iter().filter(|&&x| x >= 0).count()),
                        a.def.model.node_names.len(),
                    ));
                } else {
                    a.pos = park;
                }
            }
        }
        // #fp-attach  PARENT THE GUN TO THE HAND, the way the graph itself says to.
        //
        // A Reach first-person graph is ONE skeleton over BOTH models: the arms' 41-47 nodes and
        // the weapon's handful (`b_handle`/`b_blades`, `b_gun`/`b_magazine`/...). In that tree the
        // weapon's root is a CHILD of a hand node, so the channel the clip plays for it is a
        // transform LOCAL to the hand -- a 0.03 wu grip offset, not a rig-space position. HMS
        // composed it through the WEAPON model's own two-node skeleton, where that root has no
        // parent at all, so every first-person weapon was drawn at the rig's origin: measured on
        // the energy sword, `b_handle` landed 0.033 wu from the eye (inside the player's head)
        // while the `r_hand` holding it was 0.158 wu out, and the blade stood up across the middle
        // of the screen. The parent index is the graph's own (`s_animation_graph_node` +0x08), and
        // `ActorScene::update` resolves it after the arms are posed.
        //
        // The resolution itself is `ActorScene::fp_grip_of`, shared with test mode: it was this
        // closure, Theater-only, and that is exactly why the same rig held its gun correctly in a
        // film and left it floating in test mode.
        let arms_id = rep.and_then(|r| self.fp_actor_of.get(&r.hands).copied());
        let ids: Vec<u32> = self.fp_actor_of.values().copied().collect();
        for id in ids {
            let attach = arms_id.and_then(|arms| actors.fp_grip_of(arms, id));
            let mut became_rig = false;
            if let Some(a) = actors.get_mut(id) {
                a.fp_attach = attach;
                // #fp-fov Everything in the first-person set IS the rig -- the arms and the gun
                // alike -- and the rig is drawn through its own, narrower camera. Set here, where
                // the set is known, and never cleared: an actor only ever enters `fp_actor_of`
                // because it is a first-person model, and a parked one is still one. The lane a
                // mesh belongs to is decided at BUILD time, so a model that has just become part
                // of the rig has to be rebuilt into the right one -- `set_first_person` says so
                // and dirties it.
                became_rig = a.set_first_person(true);
                if attach.is_none() {
                    // Never leave a stale placement behind: an actor that is no longer attached
                    // (parked, or a weapon the graph does not parent) goes back to its own
                    // pos/yaw/pitch, which for a parked one is the park point.
                    a.world_override = None;
                }
            }
            if became_rig { actors.rebuild = true; }
        }
        // #fp-arms  Whether the arms are wearing the subject's OWN armour, and in whose name.
        // "The arms don't share the customization" is invisible otherwise: a rig can be perfectly
        // posed and still be drawn in the arms model's authored default. Computed before the
        // status string so it borrows nothing the assignment needs.
        let armour_note: String = match pb.subject_slot() {
            None => ", in the arms model's DEFAULT armour \u{2014} this recording never bound the \
                     camera's subject to a player slot, so there is no appearance to wear"
                .to_string(),
            // The table is built by `apply_recorded_appearance`, which runs AFTER this on the
            // first frame of a film -- so "not resolved yet" and "this player recorded none" are
            // different answers and the caption must not state the second when it means the first.
            Some(sl) if self.appearance.is_none() => format!(
                ", in the arms model's DEFAULT armour for this frame \u{2014} the recorded \
                 appearances are not resolved yet (player slot {sl})"
            ),
            Some(sl) => match self.appearance.as_ref().and_then(|a| a.get(&sl)) {
                None => format!(
                    ", in the arms model's DEFAULT armour \u{2014} player slot {sl} has no recorded \
                     appearance in this film"
                ),
                Some((_, cols)) => format!(
                    ", wearing player slot {sl}'s own armour{}",
                    match cols {
                        Some(c) => format!(
                            " and their own colours (primary {:.2},{:.2},{:.2})",
                            c[0][0], c[0][1], c[0][2]
                        ),
                        None => " (no colours recorded for them)".to_string(),
                    }
                ),
            },
        };
        self.fp_status = if !live {
            String::new()
        } else if rep.is_none() {
            "first person: this cache has no `@player representation`, so no first-person arms \
             model could be resolved"
                .into()
        } else if want.is_empty() {
            format!(
                "FIRST-PERSON ARMS: not drawn here{weapon_note}. The arms model carries no \
                 animations of its own \u{2014} in Reach the first-person animation belongs to the \
                 WEAPON (`weap` +0x3B8 `player interface` -> `first person animations`) \u{2014} and \
                 drawing them in their bind pose would put a T-pose forearm across the whole view"
            )
        } else {
            let mut t = format!(
                "FIRST-PERSON ARMS drawn from the globals' own `@player representation` \
                 (`matg` +0x474) \u{2014} the {} row, chosen from the biped the recording \
                 names for this body \u{2014} and mounted on the view{weapon_note}",
                if subject_elite { "ELITE" } else { "SPARTAN" }
            );
            t.push_str(&armour_note);
            if let Some(c) = clip {
                t.push_str(&format!(
                    ", playing the BASE clip `{c}`{}{}",
                    match fp_oneshot {
                        Some(age) => format!(" once from the recorded event {} tick(s) ago", age),
                        None => String::new(),
                    },
                    if fp_overlays.is_empty() {
                        String::new()
                    } else {
                        format!(" with the OVERLAY(s) {} composed on top", fp_overlays.join(", "))
                    }
                ));
            }
            for (m, name, frame, n, live, nodes) in &fp_live {
                t.push_str(&format!(
                    ". {}: {} at frame {:.1}/{:.0}, node map {}",
                    native.tag_name(cache, *m).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string(),
                    if name.is_empty() { "BIND POSE (no clip loaded)" } else { name.as_str() },
                    frame, n,
                    match live {
                        Some(k) => format!("{k} of this model's {nodes} node(s) driven"),
                        None => format!("positional ({nodes} node(s))"),
                    }
                ));
            }
            if let Some(e) = failed {
                t.push_str(&format!(". One first-person model could not be placed \u{2014} {e}"));
            }
            t
        };
    }

    /// Drop the pool (on closing a film or loading another map).
    pub fn clear(&mut self, actors: &mut crate::reach_actor::ActorScene) {
        if !self.slots.is_empty() || !self.vehicle_actor.is_empty() || !self.object_actor.is_empty() {
            actors.clear();
        }
        self.slots.clear();
        self.pool.assigned.clear();
        self.anim_of.clear();
        self.track_of.clear();
        self.death_company.borrow_mut().clear();
        self.corpse_rest.clear();
        self.corpse_sims.clear();
        self.ragdoll_defs.clear();
        self.obj_tris.clear();
        self.ragdoll_missing = None;
        self.vehicles = None;
        self.vehicle_actor.clear();
        self.vehicle_error = None;
        // #film-objects  A stale `object net id -> actor id` entry would point at an actor built
        // from the PREVIOUS film's tag, so a new film's flag would be drawn as whatever used to
        // hold that id. The actor scene itself has just been cleared above.
        self.objects = None;
        self.object_actor.clear();
        self.object_error = None;
        self.object_cache = None;
        self.object_status.clear();
        self.objects_shown = 0;
        self.objectives_shown = 0;
        self.objects_on_spawn = 0;
        self.objects_capped = 0;
        self.veh_damage.clear();
        self.veh_responses.clear();
        self.veh_damage_ready = false;
        self.cam_tags = Default::default();
        self.weap_tags.clear();
        self.weap_class.clear();
        self.fp_reps.clear();
        self.fp_actor_of.clear();
        self.fp_weapon_of.clear();
        self.appearance = None;
        self.fp_status.clear();
        self.clip_frames.clear();
        self.phase_prefix.clear(); // #film-perf  keyed on `clip_frames`, so it goes with it
    }


    /// #fp-frame  **Print-only diagnostic.** Where the first-person rig actually lands IN THE
    /// FRAME, measured off the posed actors rather than off the intent.
    ///
    /// Every node of every first-person actor is taken to WORLD space (`actor.world()` times its
    /// posed model-space frame), then into the camera's own basis, and reported as the angle off
    /// the view axis and as the normalised device coordinate it projects to. A rig that is posed
    /// perfectly but sits below the bottom of the frustum reads here as `ndc_y` below -1, which
    /// is exactly the failure a "the arms are driven, 47 of 47 nodes" report cannot see.
    ///
    /// Nothing in the shipping render path reads this.
    pub fn fp_framing(
        &self,
        actors: &crate::reach_actor::ActorScene,
        cam: &hms_render::Camera,
        aspect: f32,
        verbose: bool,
    ) -> String {
        let mut out = String::new();
        if self.fp_actor_of.is_empty() {
            return "first-person framing: no first-person actor is placed\n".into();
        }
        let fwd = cam.forward();
        // The render's own basis: `Mat4::look_at_rh(pos, pos+forward, +Z)`.
        let right = fwd.cross(glam::Vec3::Z).normalize();
        let up = right.cross(fwd).normalize();
        let th = (cam.fov_y * 0.5).tan();
        out.push_str(&format!(
            "first-person framing: fov_y {:.1}\u{00B0} (={:.1}\u{00B0} horizontal at {:.3}), near {:.3} wu\n",
            cam.fov_y.to_degrees(),
            2.0 * (th * aspect).atan().to_degrees(),
            aspect,
            cam.near
        ));
        for (&mode, &id) in &self.fp_actor_of {
            let Some(a) = actors.get(id) else { continue };
            if (a.pos - glam::Vec3::from(self.park)).length() < 0.01 {
                out.push_str(&format!("  model {mode:#010x}: PARKED\n"));
                continue;
            }
            let w = a.world();
            let bind = a.def.model.skel.bind_model();
            let (mut lo_y, mut hi_y, mut lo_x, mut hi_x) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            let (mut inside, mut total, mut nearclip) = (0usize, 0usize, 0usize);
            let mut rows = String::new();
            for (i, m) in a.pose.model.iter().enumerate() {
                let p = (w * *m).w_axis.truncate();
                let d = p - cam.pos;
                let (f, r, u) = (d.dot(fwd), d.dot(right), d.dot(up));
                if f <= 0.0 {
                    total += 1;
                    continue;
                }
                if f < cam.near { nearclip += 1; }
                let (nx, ny) = (r / (f * th * aspect), u / (f * th));
                lo_y = lo_y.min(ny); hi_y = hi_y.max(ny);
                lo_x = lo_x.min(nx); hi_x = hi_x.max(nx);
                total += 1;
                if nx.abs() <= 1.0 && ny.abs() <= 1.0 { inside += 1; }
                if verbose {
                    let ms = m.w_axis.truncate();
                    let bs = bind.get(i).map(|b: &glam::Mat4| b.w_axis.truncate()).unwrap_or_default();
                    rows.push_str(&format!(
                        "      {:<22} model ({:+6.3},{:+6.3},{:+6.3}) bind ({:+6.3},{:+6.3},{:+6.3})  depth {:6.3}  right {:+6.3}  up {:+6.3}  |  {:+6.1}\u{00B0} h {:+6.1}\u{00B0} v  |  ndc ({:+6.2}, {:+6.2})\n",
                        a.def.model.node_names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                        ms.x, ms.y, ms.z, bs.x, bs.y, bs.z,
                        f, r, u,
                        (r / f).atan().to_degrees(), (u / f).atan().to_degrees(),
                        nx, ny
                    ));
                }
            }
            out.push_str(&format!(
                "  model {mode:#010x} '{}' clip '{}' frame {:.1}: {inside}/{total} nodes in frame, \
                 ndc_x {lo_x:+.2}..{hi_x:+.2}  ndc_y {lo_y:+.2}..{hi_y:+.2}{}\n",
                a.def.model.node_names.first().map(|s| s.as_str()).unwrap_or(""),
                a.anim_name, a.frame,
                if nearclip > 0 { format!(", {nearclip} node(s) nearer than the near plane") } else { String::new() }
            ));
            // #fp-cam  The `camera_control` node, which is the one the ENGINE drives the view
            // from. Reported so a rig that swings out of frame can be told apart from a view that
            // is supposed to swing with it.
            if let Some(ci) = a.def.model.node_names.iter().position(|n| n.eq_ignore_ascii_case("camera_control")) {
                if let Some(m) = a.pose.model.get(ci) {
                    let (t, f, u) = (m.w_axis.truncate(), m.x_axis.truncate(), m.z_axis.truncate());
                    out.push_str(&format!(
                        "    camera_control: t ({:+.3},{:+.3},{:+.3})  fwd ({:+.3},{:+.3},{:+.3}) = {:.1}\u{00B0} off model +X  up ({:+.3},{:+.3},{:+.3})\n",
                        t.x, t.y, t.z, f.x, f.y, f.z,
                        f.normalize_or_zero().dot(glam::Vec3::X).clamp(-1.0, 1.0).acos().to_degrees(),
                        u.x, u.y, u.z
                    ));
                }
            }
            out.push_str(&rows);
        }
        out
    }

    /// The camera height of the body the first-person camera is on — the Elite row where the
    /// recording names an Elite, the Spartan row otherwise. One implementation, so the view and
    /// the rig mounted on it can never disagree about where the eye is.
    pub fn eye_height(&self, pb: &Playback) -> f32 {
        let elite = pb
            .hidden_entity()
            .and_then(|net| self.biped_of.get(&net))
            .is_some_and(|b| crate::film_anim::is_elite(b));
        if elite { ELITE_EYE_HEIGHT_WU } else { EYE_HEIGHT_WU }
    }

    /// What every body on screen is doing AT `pb`'s current tick, computed fresh.
    ///
    /// Pure, and deliberately NOT a read of whatever the last rendered frame left behind: the
    /// scripted path (`film seek` then `film show`) asks for the caption between frames, so a
    /// cached answer is one tick stale — it printed the new tick's header over the previous
    /// tick's per-body list. The GUI was consistent only by accident of ordering. Recomputing
    /// costs one `derive` per drawn body, which is a few dozen floating-point operations.
    pub fn states_at(&self, pb: &Playback) -> Vec<(u32, crate::film_anim::BodyState)> {
        self.states_and_total(pb).0
    }

    /// #wclass The animation class of the weapon a body is carrying at `tick`, from the map's
    /// own `weap` tags — `None` when the recording never named one for this body.
    fn class_at(&self, tr: Option<&crate::film_decode::EntityTrack>, tick: u32) -> Option<&str> {
        let w = tr?.weapon_at(tick)? as usize;
        self.weap_class.get(w)?.as_deref()
    }

    /// #weapon  How many of the bodies drawn at `pb`'s tick are holding a weapon THE CACHE
    /// RESOLVED, counted fresh from the recording.
    ///
    /// **Not `self.weapons_shown`.** That counter is written by `sync`, i.e. by the last DRAWN
    /// frame, and the scripted path asks for the caption between frames (`film seek` then
    /// `film show`): before the first frame at a new tick it is still 0, so the panel printed
    /// "NOBODY is holding a weapon" over a scene in which everybody was. Measured on
    /// `21897D8A_...`: `weapons_shown` read 0 immediately after a seek and 27 one frame later,
    /// on the same tick. Same rule as `sync`'s own — the track's weapon index at this tick has to
    /// resolve to a `weap` tag in the open cache — so the two can never disagree about a body.
    pub fn armed_at(&self, pb: &Playback) -> usize {
        let tick = pb.tick.round().max(0.0) as u32;
        pb.bodies_from(pb.frame(), self.cap, self.corpse_cap)
            .into_iter()
            .filter(|p| {
                self.track_of
                    .get(&p.net_id)
                    .and_then(|&i| pb.index.tracks.get(i))
                    .and_then(|tr| tr.weapon_at(tick))
                    .and_then(|w| self.weap_tags.get(w as usize).copied())
                    .is_some()
            })
            .count()
    }

    /// #film-perf  The same, plus how many bipeds this moment has in total — both out of ONE walk
    /// of the index. The caption needs the total ("14 of 15 drawn") and used to call `frame()` a
    /// second time for it.
    pub fn states_and_total(&self, pb: &Playback) -> (Vec<(u32, crate::film_anim::BodyState)>, usize) {
        let tick = pb.tick.round().max(0.0) as u32;
        let frame = pb.frame();
        let total = frame.iter().filter(|p| p.entity_type == BIPED_TYPE).count();
        let states = pb
            .bodies_from(frame, self.cap, self.corpse_cap)
            .into_iter()
            .map(|p| {
                let seat = self.vehicles.as_ref().and_then(|v| v.seat_of(p.net_id, tick));
                let tr = self.track_of.get(&p.net_id).and_then(|&i| pb.index.tracks.get(i));
                let st = match (p.dead_ticks, tr) {
                    (Some(age), tr) => {
                        let mut st = crate::film_anim::dead_state(
                            p.net_id,
                            age,
                            self.biped_of.get(&p.net_id).is_some_and(|b| crate::film_anim::is_elite(b)),
                            tr.and_then(|tr| pb.death_at(tr).map(|d| crate::film_anim::died_airborne(tr, d)))
                                .unwrap_or(false),
                        );
                        // #death The one extra thing the caption can say about a death: who was
                        // standing next to the body, and from what direction. Computed HERE and
                        // not in `sync`, because it scans the index and only the panel shows it.
                        st.death_company = match self.death_company.borrow_mut().entry(p.net_id) {
                            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                            std::collections::hash_map::Entry::Vacant(e) => *e.insert(tr.and_then(|tr| {
                                pb.death_at(tr).and_then(|d| crate::film_anim::death_company(&pb.index.tracks, tr, d))
                            })),
                        };
                        st
                    }
                    (None, Some(tr)) => crate::film_anim::derive(tr, tick, seat),
                    (None, None) => crate::film_anim::BodyState::default(),
                };
                // #wclass  Which animation set this body is really being driven on.
                let mut st = st;
                st.weapon_class = self.class_at(tr, tick).map(|c| c.to_string());
                // #melee  The decoded discrete actions, by the same rule the draw path uses.
                self.apply_actions(&mut st, p.net_id, tick);
                (p.net_id, st)
            })
            .collect();
        (states, total)
    }

    /// The caption the UI must show. It is PER BODY, and it says for each one where its pose came
    /// from, because the three provenances genuinely differ:
    ///
    /// * the position and the body yaw are **decoded** and oracle-checked;
    /// * the gait, the stance and the seat are **derived** from those two — which is what the
    ///   engine itself does, since the biped's 40-field derived dirty block carries no animation
    ///   state at all (read off `haloreach.dll`, `sub_1803B4B88`: locomotion is reconstructed on
    ///   the receiver from the control record and the aiming vector);
    /// * where a body has no decoded facing at this tick, nothing can be derived and the pose is
    ///   a standing default — **synthesised** — and the line says so for that body.
    pub fn caption(&self, pb: &Playback) -> String {
        let (states, total) = self.states_and_total(pb);
        self.caption_with(pb, &states, total)
    }

    /// #film-perf  The same, from states the caller already has.
    pub fn caption_with(&self, pb: &Playback, states: &[(u32, crate::film_anim::BodyState)], bipeds: usize) -> String {
        // `states` is exactly the set `sync` places this frame, so it is the honest count even
        // when the question is asked between frames; the actor pool's own size is not.
        let dead = states.iter().filter(|(_, st)| st.stance == crate::film_anim::Stance::Dead).count();
        let live = states.len() - dead;
        let mut s = format!("{live} of {bipeds} player(s) drawn at their RECORDED position and facing. ");
        let derived = states
            .iter()
            .filter(|(_, st)| st.stance != crate::film_anim::Stance::Dead)
            .filter(|(_, st)| st.provenance == crate::film_anim::Provenance::Derived)
            .count();
        let blind = live - derived;
        if self.animate {
            s.push_str(&format!(
                "{derived} posed from their own recorded motion (gait, strafe direction, airborne and sprint are DERIVED from the \
                 decoded position + body yaw \u{2014} the stream carries no animation state, the engine derives it too)"
            ));
            if blind > 0 {
                s.push_str(&format!("; {blind} standing in a default pose (no decoded facing at this tick)"));
            }
            // #weapon Say the honest thing EVERY time, including — especially — when the answer
            // is "none". Leaving this clause out when no body is armed reads as a broken feature,
            // and it is not: a film reports a unit's weapon only when it CHANGES hands.
            // #weapon  Counted at THIS tick, not read off the last drawn frame -- see `armed_at`.
            let armed = states.len().min(self.armed_at(pb));
            if armed > 0 {
                // #wclass  The animation set is now the weapon's OWN class, read off the weapon
                // tag (`weap` +0x3AC `weapon class`), so a sword is carried like a sword. Say how
                // many bodies that reached, and how many fell back to the generic set because
                // this biped's graph has no clips for their class.
                s.push_str(&format!(
                    "; {armed} holding the weapon the recording names, {} of them moving in that \
                     weapon's OWN animation class (read from the weapon tag's `weapon class` \
                     field){}",
                    self.class_bodies,
                    if self.class_fallbacks > 0 {
                        format!(
                            "; {} on the generic {} set because this biped's graph has no clips for their class",
                            self.class_fallbacks,
                            crate::film_anim::DEFAULT_WEAPON_CLASS
                        )
                    } else {
                        String::new()
                    }
                ));
            } else if live > 0 {
                // #weapon  Rare now that the game variant's LOADOUTS are decoded and seeded onto
                // the tracks (`film_variant::apply_spawn_weapons`), so when it does print it is a
                // real statement about this recording rather than a standing disclaimer. The old
                // text here ended "which HMS does not decode yet", which stopped being true.
                s.push_str(
                    "; NOBODY is holding a weapon at this tick, and that is the recording rather \
                     than this program: a film reports a unit's weapon only when it CHANGES \
                     hands, and the weapon a player SPAWNS with is a loadout out of the game \
                     variant -- so a body names nothing at all only when the variant states no \
                     loadout for it and it has not picked anything up yet",
                );
            }
            // #aim  WHERE THEY ARE LOOKING, and the recording's own limit on it. Said every
            // time, including when the answer is "one body", because a viewer watching sixteen
            // players must not be left to assume all sixteen are being aimed.
            let aim_live = self.live_aim;
            s.push_str(&match aim_live {
                Some(slot) => format!(
                    "; the LOOK DIRECTION is posed with the biped's own aim OVERLAY \
                     (`combat:<class>:aim_<state>_up`, blended onto the locomotion clip the way \
                     the engine composes an overlay) for {} of them \u{2014} and for only that many \
                     because a Reach film records ONE player's aim exactly, the recorder's \
                     (slot {slot}); every other slot's aim field is replicated and zero-filled, \
                     so their bodies are posed level, which is what the recording says{}",
                    self.aim_shown,
                    if self.aim_missing > 0 {
                        format!(
                            ", and {} had an aim to pose but no aim clip in their own graph",
                            self.aim_missing
                        )
                    } else {
                        String::new()
                    }
                ),
                None => "; NO look direction is posed: this recording has no slot whose aim \
                         field actually moves, so there is no aim to apply and every body is \
                         posed level"
                    .to_string(),
            });
            // #melee  Swings, which are DECODED rather than derived.
            if self.melee_shown > 0 {
                s.push_str(&format!(
                    "; {} swinging a MELEE strike \u{2014} the recording's own \
                     `biped_melee_initiate` event, played once from the tick it fires. The clips \
                     are type 3, REPLACEMENT, and carry only {} of the skeleton's {} nodes, so \
                     the swing is laid OVER the locomotion clip and the legs stay on it{}",
                    self.melee_shown,
                    self.melee_nodes.0,
                    self.melee_nodes.1,
                    if self.melee_clip_missing {
                        ", and one graph here has no type-3 melee clip for the class its body is carrying"
                    } else {
                        ""
                    }
                ));
            }
            // #fire  Shots.
            if self.fire_shown > 0 {
                s.push_str(&format!(
                    "; {} FIRING \u{2014} the recording's `action_weapon_fire` event, composed as \
                     the weapon class's own `fire_1` recoil overlay on top of whatever else the \
                     body is doing",
                    self.fire_shown
                ));
            }
            if dead > 0 {
                s.push_str(&format!(
                    "; {dead} lying dead, falling with that BIPED'S OWN death clip{} -- the \
                     ground landing, or the `airborne_dead` flail where the recorded vertical \
                     speed says the body was off the ground when it died. The MOMENT of death is \
                     the recording's own (the `incident` that names the body as the party \
                     killed), and the body lies there until the recording DESTROYS it, which is \
                     the despawn the engine performed -- a median 16 s later, never a fixed \
                     window. WHICH RAGDOLL actually played is NOT recorded: a Reach death is a \
                     physics ragdoll, and the one thing on the wire that could narrow it (the \
                     damage direction inside `damage_aftermath`'s 122-bit parameter block) is \
                     undecoded, so the clip is rotated deterministically from the entity id and \
                     replays the same way every time",
                    if self.death_clip_missing {
                        " where it has one (some graph here has none, so a body lies in a standing pose)"
                    } else {
                        ""
                    }
                ));
                // #corpse-fall  A film states NO position for a body after it goes down (0
                // reports after the death on all 1,662 corpse records of the development clip),
                // so a corpse left at its last reported position hangs wherever that packet was
                // -- measured at up to 1.0 wu above the floor. The landing is therefore DERIVED,
                // with one downward cast through HMS's own collision and a free fall at the
                // engine's own measured gravity, and this says so.
                // #ragdoll  A body the SOLVER owns does not use the one-shot vertical fall at
                // all, so the caption must not claim it does: say which rule each body is under.
                let all_ragdolls = self.ragdolls_shown > 0 && self.ragdolls_shown >= self.corpses_shown;
                if !all_ragdolls {
                    s.push_str(&format!(
                        " A body the solver does not own is LANDED on the floor under it -- a \
                         downward cast through the map's own collision, then a free fall at the \
                         {:.2} wu/s\u{b2} fitted off these recordings; the film itself reports no \
                         position at all once a body is down, so where it comes to rest is \
                         derived, not decoded.",
                        CORPSE_GRAVITY
                    ));
                    if self.corpses_falling > 0 {
                        s.push_str(&format!(" {} still falling.", self.corpses_falling));
                    }
                }
                if self.corpses_no_floor > 0 {
                    s.push_str(&format!(
                        " {} with no floor within {:.0} wu -- they went over the edge, so they are \
                         left exactly where the recording put them.",
                        self.corpses_no_floor, GROUND_REACH
                    ));
                }
                // #ragdoll  What a dead body actually IS now: a simulation, not a clip.
                if self.ragdolls_shown > 0 {
                    s.push_str(&format!(
                        " {} of them is a RAGDOLL -- the biped's own physics model, its rigid                          bodies joined by the constraints the tag authors, colliding with the map                          and the Forge objects. Seeded from the recording: the death tick, the                          body's recorded velocity and the killer the kill feed names; only the                          initial spin is derived from the entity id, because the film does not                          state the damage impulse.",
                        if self.ragdolls_shown == 1 { "One".into() } else { format!("{}", self.ragdolls_shown) }
                    ));
                } else if let Some(why) = &self.ragdoll_missing {
                    s.push_str(&format!(" No body could be simulated: {why}."));
                }
                if !self.death_clips.is_empty() {
                    // #ragdoll  When the solver owns the body the clip is only the SEED POSE the
                    // simulation starts from, not what is drawn -- so it must not read as the
                    // thing on screen.
                    s.push_str(&format!(
                        " {}: {}.",
                        if self.ragdolls_shown > 0 { "Seeded from" } else { "Lying on" },
                        self.death_clips.join(", ")
                    ));
                }
            }
            // #film-action  ASSASSINATIONS ARE DECODED. The old caption here said the opposite,
            // and it was wrong in a specific way worth not repeating: the engine's per-unit
            // action field really is absent from a film, but discrete actions ride a different
            // channel entirely (list B command kind 1, 72 named event types), and its `incident`
            // event names the kill, both players and the tick.
            let total_sync = pb.assassinations().len();
            if self.sync_shown > 0 {
                let decoded_move = states
                    .iter()
                    .filter_map(|(_, st)| st.sync.as_ref())
                    .any(|sy| sy.move_decoded);
                s.push_str(&format!(
                    "; {} body(ies) mid-ASSASSINATION, which the recording STATES \u{2014} the kill, \
                     both players and the tick come off the stream's own event channel. {}{}",
                    self.sync_shown,
                    if decoded_move {
                        "The move itself is decoded too: this film carries the host's own \
                         `biped_assassinate` record. "
                    } else {
                        "WHICH of the eight moves is not decoded here \u{2014} that rides a \
                         client\u{2192}host request only a host-recorded film carries \u{2014} so the move \
                         is chosen from the pair's own ids and replays the same way every time. "
                    },
                    if self.sync_clip_missing {
                        "This biped's graph has no two-party clip for that species pair, so the \
                         pair is standing rather than performing one."
                    } else {
                        "Both halves come from the biped's own `sync_actions` set."
                    }
                ));
                if !self.sync_clips.is_empty() {
                    s.push_str(&format!(" Playing: {}.", self.sync_clips.join(", ")));
                }
            } else if total_sync > 0 {
                s.push_str(&format!(
                    "; this clip contains {total_sync} DECODED assassination(s), none at this tick"
                ));
            }
            // #aim  Say what a body's ORIENTATION actually is, because "they don't seem to be
            // looking in the right direction" is a fair reading of what used to be drawn. Where
            // the recording states an aim, the body is now turned by its replicated BODY YAW
            // *and* posed by the graph's own aim OVERLAY on top; where it does not, the body is
            // turned by the body yaw alone, which is all the recording says. The per-tick clause
            // above gives the count either way, so this one only names what is still missing.
            s.push_str(
                ". Crouch and the armour ability are NOT decoded and NOT drawn. ",
            );
            // #film-action  The species, which is also decoded now.
            if self.biped_of.is_empty() {
                s.push_str(&format!(
                    "Every body is drawn with the default biped: {}. ",
                    if self.biped_status.is_empty() { "the bodies have not been resolved yet" } else { self.biped_status.as_str() }
                ));
            } else {
                s.push_str(&format!(
                    "Each body is drawn as the character the recording NAMES in its spawn message \
                     ({}){}. ",
                    self.biped_status,
                    if self.species_shown > 0 {
                        format!("; {} on screen now are not Spartans", self.species_shown)
                    } else {
                        String::new()
                    }
                ));
            }
        } else {
            s.push_str("Idle pose (animation off). ");
        }
        match &self.vehicles {
            None => s.push_str("Vehicles: not scanned yet (the film is still being prepared). "),
            Some(vt) if vt.total == 0 => s.push_str("No vehicles in this clip. "),
            Some(vt) => {
                // TWO SCOPES, said separately, because mixing them makes the figures look wrong:
                // what is on screen at THIS tick, and what the whole clip contains. Each sentence
                // adds up on its own.
                let tick = pb.tick.round().max(0.0) as u32;
                let drawn = vt.to_draw_at(&pb.index, tick).len();
                let parked = vt.parked_on_spawn_at(&pb.index, tick);
                s.push_str(&format!(
                    "Vehicles here now: {} \u{2014} {drawn} drawn from the recording, {parked} parked on their own spawn point where \
                     the film's Forge scene already draws them. ",
                    drawn + parked
                ));
                // #film-vehid  Say which of the two identifications found each vehicle. The
                // spawn-point search can find none at all on a gametype that SPAWNS its vehicles
                // rather than having Forge place them (0 of 66 on the measured clip), and the
                // definition every creation message states finds them exactly.
                s.push_str(&format!(
                    "In the whole clip: {} vehicle(s) \u{2014} {} identified, {} of those by the spawn point they appeared on and \
                     {} by the object DEFINITION their own creation message states; {} whose model could not be worked out. ",
                    vt.total,
                    vt.vehicles.len(),
                    vt.vehicles.len().saturating_sub(self.veh_by_def.added),
                    self.veh_by_def.added,
                    vt.unidentified
                ));
                // #film-attach  The ones with no position of their own are NOT "parked where the
                // Forge scene draws them" -- on the measured clip the variant places no vehicle
                // at all, and 30 of 33 Warthog turrets are in this group because they RIDE their
                // Warthog. Saying so is the honest version, and it names what is missing.
                if vt.no_position > 0 {
                    s.push_str(&format!(
                        "{} more never replicated a position of their own: a child object riding a PARENT reports no position, \
                         because its place in the world is its parent's. That parent reference is base field 12 of the object \
                         block and it is not decoded yet, so those are not drawn. ",
                        vt.no_position
                    ));
                }
                s.push_str(&format!(
                    "{} rider(s) seated now, {} ride(s) in the clip. A vehicle is placed with its FULL recorded basis (the stream's \
                     own 19-bit up axis + 8-bit angle about it), so pitch and roll are drawn \u{2014} {} tilted more than 5 degrees \
                     off level at this moment. ",
                    states.iter().filter(|(_, st)| st.seat.is_some()).count(),
                    vt.rides.len(),
                    self.veh_tilted
                ));
                // #vehicle-damage  What the recording states about damage, and what it does not.
                let (dmg, dest, evs, authored) = self.veh_damage_totals;
                if authored == 0 {
                    s.push_str(
                        "No vehicle in this clip has a model that authors damage states, so none can be shown damaged. ",
                    );
                } else {
                    s.push_str(&format!(
                        "Damage: {dmg} of {authored} vehicle(s) whose model authors damage states were damaged in this clip and \
                         {dest} destroyed, from {evs} decoded damage_section_response event(s); {} drawn damaged now, {} of them \
                         wreck(s) waiting for the despawn the recording itself performed. ",
                        self.veh_damaged_now, self.wrecks_now
                    ));
                }
            }
        }
        // #film-objects  The flag, the dropped weapons and the debris. Two scopes again, said
        // separately: what is on screen now, and what the whole recording holds. The clip-wide
        // figures are the ones that answer "is the flag even in this film" -- which is the
        // question the report that opened this lane was really asking.
        if !self.show_objects {
            s.push_str("Recorded objects (the flag, dropped weapons, debris): hidden. ");
        } else {
            match &self.objects {
                None => s.push_str(
                    "Recorded objects: not resolved yet \u{2014} a map has to be open to turn each spawn message's \
                     14-bit definition reference into a tag. ",
                ),
                Some(ot) if ot.objects.is_empty() => s.push_str(&format!(
                    "No recorded object in this clip is drawn from the stream: {}. ",
                    self.object_status
                )),
                Some(ot) => {
                    s.push_str(&format!(
                        "Objects here now: {} drawn from the recording ({} objective), {} left to the film's own Forge scene \
                         because they are still on their own spawn point{}. ",
                        self.objects_shown,
                        self.objectives_shown,
                        self.objects_on_spawn,
                        if self.objects_capped > 0 {
                            format!(", {} more past the {} the pool draws at once", self.objects_capped, self.object_cap)
                        } else {
                            String::new()
                        }
                    ));
                    s.push_str(&format!(
                        "In the whole clip: {} objective object(s), {} loose weapon(s) and {} piece(s) of debris are drawn from \
                         the stream; {} more are Forge pieces the scene already draws and {} are tokens with no world presence \
                         (an armour ability a loadout granted), {} whose definition did not resolve. Each one is placed with its \
                         FULL recorded basis, so a flag lying on its side is drawn on its side. ",
                        ot.by_draw[0], ot.by_draw[1], ot.by_draw[2], ot.by_draw[3], ot.by_draw[4], ot.unresolved
                    ));
                    // #projectile  The projectile lane, said in full, because every part of it is
                    // a claim a viewer should be able to check.
                    if ot.by_draw[5] > 0 || self.fx_muzzle > 0 || self.fx_detonation > 0 {
                        s.push_str(&format!(
                            "PROJECTILES: {} in the whole clip are replicated by the recording (engine entity type 16 \u{2014} \
                             grenades, rockets, missiles, needles, Falcon and turret bullets), of which {} are drawn as a model \
                             here now and {} carry no render model at all (a rifle bullet; the engine draws nothing of those \
                             either, so they are left to their trail). Here now: {} muzzle flash(es), {} travelling effect(s) \
                             and {} detonation(s). ",
                            ot.by_draw[5], self.projectiles_shown, self.projectiles_no_model,
                            self.fx_muzzle, self.fx_trail, self.fx_detonation,
                        ));
                        s.push_str(&format!(
                            "A muzzle flash is the weapon's own `weap` +0x408 `barrels[0].firing effects[0].firing effect^`, \
                             played once at the `action_weapon_fire` tick on the gun in the body's hands; the effect names its \
                             own markers (`primary_trigger`, `muzzle_flash`, `primary_ejection`) and a named marker the model \
                             does not have is SKIPPED, never moved to the weapon origin. A detonation is the projectile's own \
                             `proj` `detonation effect (ground)` at the last position its track reports. {}{}",
                            if self.fx_unarmed > 0 {
                                format!("{} shot(s) here have no named weapon in the recording, so no flash is played for them. ", self.fx_unarmed)
                            } else { String::new() },
                            if self.fx_no_effect > 0 {
                                format!("{} shot(s) came from a weapon whose barrel authors no firing effect. ", self.fx_no_effect)
                            } else { String::new() },
                        ));
                        s.push_str(
                            "A RIFLE ROUND IS NOT SIMULATED, and the reason is the recording: a Reach film captures one \
                             player's look direction and zero-fills every other slot's, so for all but one body it does not \
                             state which way the shot went. Firing a tracer along the body's yaw would be an invention, and a \
                             visibly horizontal one. The bullet's `proj` definition is read and in hand (its initial velocity, \
                             its gravity scale, its maximum range) for the day the direction is. ",
                        );
                    }
                    if ot.by_draw[0] > 0 {
                        // Name them, because "1 objective object" does not tell a viewer whether
                        // the thing they cannot see is the flag.
                        let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
                        for o in ot.objectives() {
                            *kinds.entry(o.name.as_str()).or_default() += 1;
                        }
                        s.push_str(&format!(
                            "The objective object(s) in this clip: {}. ",
                            kinds.iter().map(|(n, c)| format!("{c} x {n}")).collect::<Vec<_>>().join(", ")
                        ));
                        s.push_str(
                            "An objective is drawn WHERE THE RECORDING PUTS IT, carried or not \u{2014} the film keeps reporting \
                             its position while a player holds it, so nothing is derived from a hand marker. Its waypoint is \
                             coloured from the carrier's team where the recording names one, and `navpt neutral` otherwise, \
                             because a film states neither the gametype's waypoint colour nor which side a flag belongs to. ",
                        );
                    }
                }
            }
        }
        // #fp What the first-person rig is showing, when one is live.
        if !self.fp_status.is_empty() {
            s.push_str(&format!(" {}. ", self.fp_status));
        }
        if let Some(e) = &self.last_error {
            s.push_str(&format!(" Could not place a biped: {e}"));
        }
        if let Some(e) = &self.vehicle_error {
            s.push_str(&format!(" Could not place a vehicle: {e}"));
        }
        if let Some(e) = &self.weapon_error {
            s.push_str(&format!(" Could not place a weapon: {e}"));
        }
        if let Some(e) = &self.object_error {
            s.push_str(&format!(" Could not place a recorded object: {e}"));
        }
        s
    }

    /// One line per drawn body: what it is doing and where that came from. This is what replaces
    /// the old blanket disclaimer.
    pub fn body_lines(&self, pb: &Playback) -> Vec<String> {
        Self::body_lines_from(&self.states_at(pb))
    }

    /// #film-perf  The same, from states the caller already has. `states_at` is the expensive
    /// part, and the panel needs both the caption and these lines from the SAME tick.
    pub fn body_lines_from(states: &[(u32, crate::film_anim::BodyState)]) -> Vec<String> {
        states
            .iter()
            .map(|(net, st)| {
                format!(
                    "0x{net:08X}  {}  [{}: {}]",
                    crate::film_anim::describe(st),
                    st.provenance.label(),
                    st.because
                )
            })
            .collect()
    }
}


/// Solve the camera for the current tick.
///
/// Returns `None` in `Free` mode. In the player modes it returns a pose whenever the player's
/// biped is known at this tick, and otherwise a pose-less reason in `label` is impossible to
/// express, so it returns `None` and the caller keeps the free camera and shows `camera_status`.
pub fn solve_camera(pb: &Playback) -> Option<CameraPose> {
    solve_camera_at(pb, EYE_HEIGHT_WU)
}

/// [`solve_camera`] with the subject's OWN camera height — an Elite's eye is 0.09 wu higher than a
/// Spartan's, and the two species are both on screen in the same film. Callers that place the
/// view or the first-person rig go through [`ActorDriver::eye_height`] to get it; the ones that
/// only print a status line can use the Spartan default.
pub fn solve_camera_at(pb: &Playback, eye_height: f32) -> Option<CameraPose> {
    if pb.camera == FilmCamera::Free {
        return None;
    }
    let plain = solve_plain_camera_at(pb, eye_height);
    // #film-camera  Follow the engine into its context cameras -- the assassination camera, the
    // vehicle camera, the third-person spectate on death. Off, the plain pose stands and the
    // camera is exactly where the viewer put it.
    if !pb.context_cameras {
        return plain;
    }
    let tick = pb.tick.round().max(0.0) as u32;
    // #film-camera  **The context is resolved BEFORE a plain pose is required, and this ordering
    // is the whole of the death camera.**
    //
    // It used to open `let plain = solve_plain_camera_at(..)?`, and that `?` was a bug with
    // exactly one visible symptom: on death the view never changed to the third-person
    // spectate. `solve_plain_camera_at` resolves its subject through `entity_for_slot` /
    // `entity_for_owner`, both of which require `tick <= track.last_tick` -- the last tick the
    // body replicated a MESSAGE. A dead body stops replicating almost at once, while the `dead`
    // and `orbiting` segments run on until the recording DESTROYS the corpse. So the one case
    // that needs the context was the one case that returned None before `resolve` was ever
    // called, and no amount of correctness in the timeline could have helped.
    //
    // `camera_subject_or_corpse` therefore finds the body the viewer is watching even after it
    // has stopped replicating, and a dead/orbiting pose stands on its own when there is no live
    // body left to anchor to.
    let Some((subject, _)) = camera_subject_or_corpse(pb, tick) else { return plain };
    let preferred = if pb.camera.is_first_person() {
        crate::film_camera::EngineCameraMode::FirstPerson
    } else {
        crate::film_camera::EngineCameraMode::Following
    };
    let ctx = crate::film_camera::resolve(pb.camera_timeline(), preferred, subject, tick);
    if ctx.detail == crate::film_camera::ContextDetail::Plain && ctx.blend >= 1.0 {
        return plain;
    }
    // The pose the context blends out of. Where the body is still live this is its own
    // first-person or follow pose; where it is not, there is nothing to blend from, so the
    // context's own shot stands alone and we hand it an anchor built from the body itself.
    let anchor = match plain.clone() {
        Some(p) => p,
        None => corpse_anchor(pb, subject, tick, eye_height)?,
    };
    crate::film_camera::context_pose(pb, &ctx, tick, pb.rig, &anchor).or(plain)
}

/// #film-camera  The body the viewer's camera is riding, **including after it has stopped
/// replicating** — which is the state a corpse is in for almost all of the time the engine
/// spends orbiting it.
///
/// Live bodies go through [`camera_subject`] unchanged. When that finds nothing, the viewer is
/// watching a player whose body has gone quiet, and the question becomes "which of that player's
/// bodies does the camera timeline still have a segment for at this tick" — which is precisely
/// the dead/orbiting window. Asking the timeline rather than inventing a rule keeps the two in
/// agreement by construction: a body is reachable here exactly while it has a camera.
pub fn camera_subject_or_corpse(pb: &Playback, tick: u32) -> Option<(u32, Option<u8>)> {
    if let Some(s) = camera_subject(pb, tick) {
        return Some(s);
    }
    let (owner, slot) = match pb.camera {
        FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => (pb.index.owner_of_slot(s)?, Some(s)),
        FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => (o, pb.index.slot_of_owner(o)),
        FilmCamera::Free => return None,
    };
    // #film-perf  This walk runs only when the LIVE lookup already failed, i.e. inside a death
    // window, and the `owner_raw` test short-circuits before the timeline is consulted — so on a
    // 6,959-track film it is ~7k integer comparisons plus a hashmap-and-binary-search for the few
    // dozen bodies that player ever had. That is tens of microseconds, and it is deliberately not
    // memoised: a per-tick cache would be mutable state on the playback, and the one rule this
    // module cannot break is that the camera is a pure function of the tick.
    let tl = pb.camera_timeline();
    pb.index
        .tracks
        .iter()
        .filter(|t| t.owner_raw == Some(owner) && tl.at(t.net_id, tick).is_some())
        .max_by_key(|t| t.first_tick)
        .map(|t| (t.net_id, slot))
}

/// #film-camera  An anchor pose built from a body that is no longer replicating, for a context
/// camera that has no live pose to blend out of.
///
/// The position is the body's last reported one — `EntityTrack::sample` holds it past the end of
/// the track, which is what makes a corpse addressable at all — and the facing is its last
/// decoded one. **Deliberately the RECORDED position and not the drawn one:** a corpse that has
/// been dropped onto the floor by `ActorDriver::ground_corpses` sits slightly below where it was
/// last reported, but that correction lives on the driver and this function has only the
/// playback. The difference is the height of a fall, the camera orbits a pivot roughly half a
/// body above it either way, and taking the recording's own number keeps this a pure function of
/// the film.
fn corpse_anchor(pb: &Playback, net_id: u32, tick: u32, eye_height: f32) -> Option<CameraPose> {
    let tr = pb.index.tracks.iter().find(|t| t.net_id == net_id)?;
    let feet = tr.sample(tick).or_else(|| tr.positions.last().map(|&(_, p)| p))?;
    let yaw = tr
        .facing_at(tick)
        .or_else(|| tr.sample_orientation(tick).map(|(f, _)| f[1].atan2(f[0])))
        .unwrap_or(0.0);
    Some(CameraPose {
        pos: [feet[0], feet[1], feet[2] + eye_height],
        yaw,
        pitch: 0.0,
        pivot: None,
        label: format!("entity 0x{net_id:08X}, no longer replicating (its last reported position)"),
    })
}

/// #film-camera  WHICH BODY the viewer's chosen camera is riding at this tick, and the
/// configuration slot behind it.
///
/// Factored out of [`solve_plain_camera_at`] because the context resolver needs the same answer
/// and the two must never disagree: a context resolved against one body while the pose was built
/// from another would put the assassination camera on the wrong player.
pub fn camera_subject(pb: &Playback, tick: u32) -> Option<(u32, Option<u8>)> {
    match pb.camera {
        FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => {
            Some((pb.index.entity_for_slot(s, tick)?.net_id, Some(s)))
        }
        FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => {
            Some((pb.index.entity_for_owner(o, tick)?.net_id, pb.index.slot_of_owner(o)))
        }
        FilmCamera::Free => None,
    }
}

/// [`solve_camera_at`] WITHOUT the engine's context cameras: the viewer's own chosen mode and
/// nothing else. Kept as its own function because the context cameras blend out of it, so the
/// plain pose has to be available at the same tick.
pub fn solve_plain_camera_at(pb: &Playback, eye_height: f32) -> Option<CameraPose> {
    if pb.camera == FilmCamera::Free {
        return None;
    }
    let first_person = pb.camera.is_first_person();
    let tick = pb.tick.round().max(0.0) as u32;
    // Resolve the subject to a BODY. A slot goes through its binding; an owner group goes
    // straight to whichever of its bodies is alive at this tick.
    let (tr, slot) = match pb.camera {
        FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => {
            let b = pb.index.entity_for_slot(s, tick)?;
            (pb.index.tracks.iter().find(|t| t.net_id == b.net_id)?, Some(s))
        }
        FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => {
            (pb.index.entity_for_owner(o, tick)?, pb.index.slot_of_owner(o))
        }
        FilmCamera::Free => return None,
    };
    let feet = tr.sample(tick)?;
    // The AIM angles are the player's own look direction, recorded once per tick. They are what
    // makes first person genuinely first person rather than "wherever the body happened to point".
    // A body with no named slot has no aim record, so it falls back to its own decoded facing --
    // which is still the recording's own statement about where that player was looking.
    // Only a slot whose aim actually moves is worth using; see `Playback::aim_live`. For
    // everyone else the body's own decoded yaw IS the recorded look direction (it tracks the aim
    // to 5.7-7.3 degrees where both exist), just without pitch.
    let aim = slot.filter(|&s| pb.aim_live.get(s as usize).copied().unwrap_or(false)).and_then(|s| pb.index.aim_at(s, tick));
    let (yaw, pitch) = match aim {
        Some(a) => (a.yaw(), a.pitch()),
        None => (
            tr.facing_at(tick)
                .or_else(|| tr.sample_orientation(tick).map(|(f, _)| f[1].atan2(f[0])))
                .unwrap_or(0.0),
            0.0,
        ),
    };
    if first_person {
        let pos = [feet[0], feet[1], feet[2] + eye_height];
        let who = match slot {
            Some(s) => format!("slot {s}"),
            None => "an unnamed player".to_string(),
        };
        let label = format!(
            "first person: {who}, entity 0x{:08X}, {}",
            tr.net_id,
            look_source(pb, tr, slot)
        );
        return Some(CameraPose { pos, yaw, pitch, pivot: None, label });
    }
    // FOLLOW. The camera orbits a pivot on the BODY and looks AT it; it does not copy the
    // player's own pitch. Reusing the recorded aim for pitch (which is what it used to do) made
    // the camera tip up and down every time the player glanced at the sky.
    //
    // Yaw still comes from the player's aim, so the camera stays behind them as they turn, with
    // the viewer's own orbit offset added on top.
    let rig = pb.rig;
    let cyaw = yaw + rig.yaw_off;
    let pivot = [feet[0], feet[1], feet[2] + FOLLOW_PIVOT_WU];
    let (ce, se) = (rig.elev.cos(), rig.elev.sin());
    let pos = [
        pivot[0] - cyaw.cos() * ce * rig.dist,
        pivot[1] - cyaw.sin() * ce * rig.dist,
        pivot[2] + se * rig.dist,
    ];
    // Looking from the camera back at the pivot is exactly (cyaw, -elev).
    let (yaw, pitch) = (cyaw, -rig.elev);
    let who = match slot {
        Some(s) => format!("slot {s}"),
        None => "an unnamed player".to_string(),
    };
    let label = format!(
        "follow: {who}, entity 0x{:08X}, {:.1} wu back, {:.0}\u{00B0} up, {:.0}\u{00B0} round",
        tr.net_id,
        rig.dist,
        rig.elev.to_degrees(),
        rig.yaw_off.to_degrees()
    );
    Some(CameraPose { pos, yaw, pitch, pivot: Some(pivot), label })
}

/// Where a first-person view's look direction is coming from, and how good it is — in words,
/// because the three cases are genuinely different and the viewer should be able to tell.
pub fn look_source(pb: &Playback, tr: &crate::film_decode::EntityTrack, slot: Option<u8>) -> String {
    if slot.map_or(false, |s| pb.aim_live.get(s as usize).copied().unwrap_or(false)) {
        return "along their own recorded aim (this is the player who saved the clip)".into();
    }
    // The body yaw is a real decoded look direction, but it is reported sparsely: on one clip
    // only 36% of bodies have any at all and the median body has ONE sample for its whole life.
    match tr.facings.len() {
        0 => "look direction NOT recorded for this player \u{2014} the view is fixed".into(),
        1..=4 => format!(
            "along the body facing, but it was only recorded {} time(s) in this life, so the view barely turns",
            tr.facings.len()
        ),
        n => format!("along the body facing ({n} samples; only the player who saved the clip has a recorded aim)"),
    }
}

/// A one-line, honest account of what the camera is doing, including why it is not following.
pub fn camera_status(pb: &Playback) -> String {
    let tick = pb.tick.round().max(0.0) as u32;
    // #film-camera  Where a context camera has taken the view, SAY SO rather than reporting the
    // viewer's chosen mode. The line below would otherwise read "first person on player slot 9"
    // over an orbit of that player's corpse, which is the same mode-versus-camera confusion that
    // hid the death camera not working at all.
    if pb.context_cameras {
        if let Some(c) = pb.camera_context() {
            if c.mode != crate::film_camera::EngineCameraMode::FirstPerson
                || c.detail != crate::film_camera::ContextDetail::Plain
            {
                return format!("{} \u{2014} {}", plain_camera_status(pb, tick), c.label());
            }
        }
    }
    plain_camera_status(pb, tick)
}

/// The viewer's chosen camera mode in words, with no context applied. Split out so
/// [`camera_status`] can prefix it with the engine camera the context resolved to.
fn plain_camera_status(pb: &Playback, tick: u32) -> String {
    match pb.camera {
        FilmCamera::Free => "free camera".into(),
        FilmCamera::FirstPerson(s) | FilmCamera::Follow(s) => {
            let what = if pb.camera.is_first_person() { "first person" } else { "follow" };
            if pb.index.entity_for_slot(s, tick).is_some() {
                format!("{what} on player slot {s}")
            } else if pb.index.lives_of_slot(s).is_empty() {
                format!("{what} on slot {s}: no biped could be matched to that player in this film")
            } else {
                let next = pb
                    .index
                    .lives_of_slot(s)
                    .into_iter()
                    .find(|b| b.first_tick > tick)
                    .map(|b| format!(", next at {}", fmt_time(b.first_tick as f32 / 60.0)));
                format!(
                    "{what} on slot {s}: that player has no biped at {}{}",
                    fmt_time(tick as f32 / 60.0),
                    next.unwrap_or_default()
                )
            }
        }
        FilmCamera::FirstPersonOwner(o) | FilmCamera::FollowOwner(o) => {
            let what = if pb.camera.is_first_person() { "first person" } else { "follow" };
            let who = match pb.index.slot_of_owner(o) {
                Some(s) => format!("player slot {s}"),
                None => "a player this recording never named".to_string(),
            };
            match pb.index.entity_for_owner(o, tick) {
                Some(t) => {
                    if pb.camera.is_first_person() {
                        format!("{what} on {who} \u{2014} {}", look_source(pb, t, pb.index.slot_of_owner(o)))
                    } else {
                        format!("{what} on {who} (entity {:#010x})", t.net_id)
                    }
                }
                None => {
                    let next = pb
                        .index
                        .tracks
                        .iter()
                        .filter(|t| t.entity_type == 11 && t.owner_raw == Some(o) && t.first_tick > tick)
                        .map(|t| t.first_tick)
                        .min()
                        .map(|t| format!(", next at {}", fmt_time(t as f32 / 60.0)));
                    format!(
                        "{what} on {who}: no body at {}{}",
                        fmt_time(tick as f32 / 60.0),
                        next.unwrap_or_default()
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_decode::{EntityTrack, FilmIndex};

    fn track(net_id: u32, ty: u8, pts: &[(u32, [f32; 3])]) -> EntityTrack {
        EntityTrack {
            net_id,
            entity_type: ty,
            first_tick: pts.first().map(|p| p.0).unwrap_or(0),
            last_tick: pts.last().map(|p| p.0).unwrap_or(0),
            positions: pts.to_vec(),
            orientations: vec![(pts.first().map(|p| p.0).unwrap_or(0), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])],
            facings: Vec::new(),
            owner_raw: None,
            appearance: false,
            look: None,
            attachments: Vec::new(),
            moves_cached: None,
            weapons: Vec::new(),
            spawn_weapon: None,
            def_wire: None,
            spawned_on: None,
            destroyed_tick: None,
        }
    }

    fn index() -> FilmIndex {
        FilmIndex {
            ticks: 600,
            tracks: vec![
                // a biped that walks 10 wu along +X over 300 ticks
                track(1, 11, &[(0, [0.0, 0.0, 0.0]), (300, [10.0, 0.0, 0.0])]),
                // a second biped that appears halfway
                track(2, 11, &[(300, [5.0, 5.0, 0.0]), (600, [5.0, 15.0, 0.0])]),
                // a motionless object
                track(3, 13, &[(0, [1.0, 2.0, 3.0]), (600, [1.0, 2.0, 3.0])]),
            ],
            ..Default::default()
        }
    }

    /// A cast of three players: one the recording names (slot 2), two it never did. All three
    /// had bodies, and all three must be watchable.
    fn cast_index() -> FilmIndex {
        use crate::film_decode::PlayerBinding;
        let mut mk = |net: u32, owner: u8, from: u32, to: u32, x: f32| {
            let mut t = track(net, 11, &[(from, [x, 0.0, 0.0]), (to, [x + 4.0, 0.0, 0.0])]);
            t.owner_raw = Some(owner);
            t.facings = vec![(from, 0.0), (to, 0.0)];
            t
        };
        FilmIndex {
            ticks: 600,
            tracks: vec![
                mk(0x11, 1, 0, 300, 0.0),   // owner 1, first life
                mk(0x12, 1, 320, 600, 20.0), // owner 1, respawn
                mk(0x21, 5, 0, 600, 40.0),   // owner 5 -- never named
                mk(0x31, 9, 100, 600, 60.0), // owner 9 -- never named
            ],
            bindings: vec![PlayerBinding {
                slot: 2,
                net_id: 0x11,
                entity_type: 11,
                first_tick: 0,
                last_tick: 300,
                cost: 0.05,
                margin: 1.0,
                samples: 500,
                from_facing: true,
                from_owner_rule: false,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// The defect this replaces: the cast was built from the film HEADER, so a player the
    /// recording never named had no row and could not be followed at all. Every body must be
    /// watchable — aiming a camera needs an entity id, not a gamertag.
    #[test]
    fn every_player_who_had_a_body_can_be_followed_named_or_not() {
        let idx = cast_index();
        assert_eq!(idx.owner_groups(), vec![1, 5, 9], "three players had a body");
        assert_eq!(idx.slot_of_owner(1), Some(2), "owner 1 is the one the recording names");
        assert_eq!(idx.slot_of_owner(5), None);
        assert_eq!(idx.slot_of_owner(9), None);
        // and every one of them resolves to a body, which is what the camera needs
        for o in idx.owner_groups() {
            let mid = idx.owner_tick_range(o).map(|(a, b)| (a + b) / 2).unwrap();
            assert!(idx.entity_for_owner(o, mid).is_some(), "owner {o} has no body at {mid}");
        }
    }

    /// Following an UNNAMED player must produce a real camera pose, not nothing.
    #[test]
    fn the_camera_follows_a_player_the_recording_never_named() {
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(400.0);
        pb.camera = FilmCamera::FollowOwner(5);
        let pose = solve_camera(&pb).expect("an unnamed player is still followable");
        assert!(pose.label.contains("unnamed"), "{}", pose.label);
        assert!(pose.label.contains("0x00000021"), "{}", pose.label);
        // first person on the same player hides that player's own body
        pb.camera = FilmCamera::FirstPersonOwner(5);
        assert_eq!(pb.hidden_entity(), Some(0x21));
        assert!(camera_status(&pb).contains("never named"), "{}", camera_status(&pb));
    }

    /// Following a player across a RESPAWN has to pick up the new body, because a player gets a
    /// new network entity every life.
    #[test]
    fn following_a_player_follows_them_through_a_respawn() {
        let idx = cast_index();
        assert_eq!(idx.entity_for_owner(1, 100).map(|t| t.net_id), Some(0x11));
        assert_eq!(idx.entity_for_owner(1, 500).map(|t| t.net_id), Some(0x12));
        assert_eq!(idx.owner_lives(1), 2);
        // between the two lives this player has no body, and the status says so rather than
        // silently doing nothing
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(310.0);
        pb.camera = FilmCamera::FollowOwner(1);
        assert!(solve_camera(&pb).is_none());
        let st = camera_status(&pb);
        assert!(st.contains("no body at") && st.contains("next at"), "{st}");
    }

    /// The follow rig's GEOMETRY, asserted rather than eyeballed.
    ///
    /// The old rig sat 1.10 wu above the feet and 1.4 wu back — about 0.4 wu over the head of a
    /// 0.7 wu Spartan, looking down ~28 degrees, which read as hovering above him.
    #[test]
    fn the_follow_camera_sits_behind_the_player_not_above_him() {
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(400.0);
        pb.camera = FilmCamera::FollowOwner(5);
        let feet = pb.index.entity_for_owner(5, 400).unwrap().sample(400).unwrap();
        let p = solve_camera(&pb).expect("pose");
        let above = p.pos[2] - feet[2];
        let back = (p.pos[0] - feet[0]).hypot(p.pos[1] - feet[1]);
        // The old rig put the camera 1.10 wu up and 1.4 wu back, looking down ~28 degrees.
        assert!((0.45..0.65).contains(&above), "camera {above:.2} wu above the feet, wanted about eye height");
        assert!((2.2..2.6).contains(&back), "camera {back:.2} wu back, wanted 2.2-2.5");
        let look_down = -p.pitch.to_degrees();
        assert!((1.0..10.0).contains(&look_down), "looking down {look_down:.1} degrees, wanted a gentle angle");
        // and it looks AT the body: the pivot is on the subject
        let pivot = p.pivot.expect("a follow pose has a pivot");
        assert!((pivot[0] - feet[0]).abs() < 1e-3 && (pivot[1] - feet[1]).abs() < 1e-3);
    }

    /// **The first-person defect.** Only the player who SAVED a clip has a recorded aim; every
    /// other slot's aim field is zero-filled, so trusting it pointed the view at a fixed angle
    /// forever. A slot whose aim never moves must fall through to the body facing.
    #[test]
    fn a_dead_aim_record_is_not_used_as_a_look_direction() {
        use crate::film_decode::PlayerAim;
        let mut idx = cast_index();
        idx.player_aims = vec![Vec::new(); 16];
        // slot 2 is "recorded": its yaw moves
        idx.player_aims[2] = (0..600u32)
            .map(|t| (t, PlayerAim { slot: 2, yaw_q: (t as u16 * 7) % 8192, pitch_q: 2048, f08: None, f40: None }))
            .collect();
        // slot 3 is zero-filled: thousands of samples, one value
        idx.player_aims[3] = (0..600u32)
            .map(|t| (t, PlayerAim { slot: 3, yaw_q: 1234, pitch_q: 2048, f08: None, f40: None }))
            .collect();
        let pb = Playback::new(idx);
        assert!(pb.aim_live[2], "a moving aim is live");
        assert!(!pb.aim_live[3], "an aim that never changes is NOT a look direction");
        // and the live one is actually used
        let mut pb2 = pb.clone();
        pb2.seek_tick(100.0);
        pb2.camera = FilmCamera::FirstPerson(2);
        let a = solve_camera(&pb2).expect("slot 2 has a body");
        assert!(a.label.contains("recorded aim"), "{}", a.label);
    }

    /// The UI has to be able to say which of the three cases a view is in.
    #[test]
    fn the_look_source_is_described_honestly() {
        let idx = cast_index();
        let pb = Playback::new(idx);
        let tr = pb.index.entity_for_owner(5, 400).unwrap();
        // this body has 2 facing samples, which is the sparse case
        let s = look_source(&pb, tr, None);
        assert!(s.contains("barely turns"), "{s}");
        let mut t2 = tr.clone();
        t2.facings.clear();
        assert!(look_source(&pb, &t2, None).contains("NOT recorded"));
        t2.facings = (0..200u32).map(|i| (i * 3, i as f32 * 0.01)).collect();
        assert!(look_source(&pb, &t2, None).contains("200 samples"));
    }

    /// Follow must NOT copy the player's pitch — the camera would tip up and down every time the
    /// player glanced at the sky — while first person must.
    #[test]
    fn follow_pitch_comes_from_the_rig_and_first_person_from_the_aim() {
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(400.0);
        pb.camera = FilmCamera::FollowOwner(5);
        let a = solve_camera(&pb).unwrap().pitch;
        pb.rig.elev = 40f32.to_radians();
        let b = solve_camera(&pb).unwrap().pitch;
        assert!((b - -40f32.to_radians()).abs() < 1e-3, "follow pitch must track the RIG: {b}");
        assert!((a - b).abs() > 0.1);
        // first person ignores the rig entirely
        pb.camera = FilmCamera::FirstPersonOwner(5);
        let fp = solve_camera(&pb).unwrap();
        assert!(fp.pivot.is_none(), "first person does not orbit a pivot");
        assert_eq!(fp.pitch, 0.0, "this body has no recorded aim, so first person is level");
    }

    /// Orbiting swings the camera around the subject and keeps it the same distance away.
    #[test]
    fn orbiting_moves_around_the_player_at_a_constant_distance() {
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(400.0);
        pb.camera = FilmCamera::FollowOwner(5);
        let behind = solve_camera(&pb).unwrap();
        pb.rig.drag(std::f32::consts::FRAC_PI_2, 0.0); // 90 degrees round
        let side = solve_camera(&pb).unwrap();
        let pivot = behind.pivot.unwrap();
        let r = |p: &CameraPose| {
            (p.pos[0] - pivot[0]).powi(2) + (p.pos[1] - pivot[1]).powi(2) + (p.pos[2] - pivot[2]).powi(2)
        };
        assert!((r(&behind) - r(&side)).abs() < 1e-2, "orbiting must not change the distance");
        assert!((side.yaw - behind.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
        // and the camera really moved to the side
        assert!((side.pos[1] - behind.pos[1]).abs() > 1.0);
    }

    /// The orbit must stay inside sane bounds however hard it is driven.
    #[test]
    fn the_orbit_is_clamped_so_it_cannot_flip_or_escape() {
        let mut r = FollowRig::default();
        for _ in 0..200 {
            r.drag(0.3, 0.3);
            r.zoom(1.0);
        }
        assert!(r.elev <= FOLLOW_PITCH_LIMIT.to_radians() + 1e-4, "pitch escaped: {}", r.elev);
        assert!(r.dist >= FOLLOW_DIST_MIN - 1e-4, "distance escaped: {}", r.dist);
        for _ in 0..200 {
            r.drag(-0.3, -0.3);
            r.zoom(-1.0);
        }
        assert!(r.elev >= -FOLLOW_PITCH_LIMIT.to_radians() - 1e-4);
        assert!(r.dist <= FOLLOW_DIST_MAX + 1e-4, "distance escaped: {}", r.dist);
        assert!(r.yaw_off.abs() <= std::f32::consts::PI + 1e-4, "yaw must stay wrapped");
        r.recenter();
        assert_eq!(r, FollowRig::default());
    }

    /// #corpse-fall  A body left hanging above the floor FALLS to it, and the fall is a pure
    /// function of the tick.
    ///
    /// This is the gate for "the ragdolls for all bipeds make bipeds just float midair". It bites
    /// on all three cases at once, because the first implementation of this could easily have
    /// grounded everything: a body already on the floor must not move at all, a body with no
    /// floor under it must not move at all, and only the body that is genuinely hanging falls.
    #[test]
    fn a_body_left_hanging_above_the_floor_falls_onto_it() {
        // already down: nothing to do, whatever the floor says
        assert_eq!(corpse_fall([0.0, 0.0, 10.0], Some(10.0)), None);
        assert_eq!(corpse_fall([0.0, 0.0, 10.0], Some(9.99)), None, "inside the floor noise");
        // over the edge of the map: no floor, so the body is left exactly where it is
        assert_eq!(corpse_fall([0.0, 0.0, 10.0], None), None);
        // hanging 1 wu up: it lands on the floor, after the time a 1 wu free fall really takes
        let (rest, ticks) = corpse_fall([0.0, 0.0, 11.0], Some(10.0)).expect("this one falls");
        assert_eq!(rest, 10.0);
        // #gravity  The expected time is derived from the MEASURED acceleration as a LITERAL, not
        // from `CORPSE_GRAVITY`. Deriving it from the constant under test makes the assertion
        // vacuous: it passes for any value, which was measured -- pointing `CORPSE_GRAVITY` at the
        // engine's authored 4.1712594 (a 2.5x change) left this gate green. 1 wu at
        // 10.4576 wu/s^2 takes sqrt(2/10.4576) = 0.4374 s = 26.2 ticks at 60 Hz.
        const MEASURED_FALL: f32 = 10.4576;
        assert_eq!(
            CORPSE_GRAVITY, MEASURED_FALL,
            "the corpse lane must fall at the acceleration fitted from the recordings"
        );
        let want = (2.0f32 / MEASURED_FALL).sqrt() * 60.0;
        assert!(
            (ticks as f32 - want).abs() <= 1.0,
            "a 1 wu fall at {MEASURED_FALL} wu/s^2 takes {want:.2} ticks, got {ticks}"
        );

        // ...and the drawn height follows that fall and then HOLDS, never sinking through.
        let mut d = ActorDriver::default();
        let pb = Playback::new(index());
        d.corpse_rest.insert(7, Some((10.0, ticks)));
        let z = |age: u32| d.corpse_drawn_z(7, [0.0, 0.0, 11.0], age);
        assert_eq!(z(0), 11.0, "at the moment of death the body has not moved");
        assert!(z(ticks / 2) < 11.0 && z(ticks / 2) > 10.0, "mid-fall it is between the two");
        assert!((z(ticks) - 10.0).abs() < 1e-3, "it arrives exactly on the floor");
        assert_eq!(z(ticks * 10), 10.0, "and stays there rather than sinking");
        // strictly falling, so scrubbing backwards lifts it again by the same amounts
        for a in 1..ticks {
            assert!(z(a) < z(a - 1), "the fall must be monotone at tick {a}");
        }
        // a body with no entry at all is untouched
        assert_eq!(d.corpse_drawn_z(99, [0.0, 0.0, 11.0], 500), 11.0);
        let _ = pb;
    }

    /// #corpse-fall  The recording's own answer to "where did the body end up" is NOTHING: the
    /// trace must report zero reports after the death, which is what forced the derived landing.
    #[test]
    fn the_recording_reports_no_position_once_a_body_is_down() {
        let mut idx = index();
        // a body that walks, then is destroyed one tick after its last reported position
        let mut t = track(42, 11, &[(0, [0.0, 0.0, 5.0]), (100, [1.0, 0.0, 6.0])]);
        t.destroyed_tick = Some(200);
        idx.tracks.push(t);
        let pb = Playback::new(idx);
        let c = pb.corpse_traces().into_iter().find(|c| c.net_id == 42).expect("a corpse record");
        assert_eq!(c.death_tick, 100, "the body goes down at its last reported position");
        assert_eq!(c.reports_after, 0);
        assert_eq!(c.last_pos, [1.0, 0.0, 6.0]);
        assert_eq!(c.at_death, [1.0, 0.0, 6.0]);
    }

    /// The slot-keyed camera still works, so the `film camera first <slot>` command and anything
    /// else that used it is unchanged.
    #[test]
    fn the_slot_keyed_camera_still_resolves() {
        let mut pb = Playback::new(cast_index());
        pb.seek_tick(100.0);
        pb.camera = FilmCamera::FirstPerson(2);
        assert!(solve_camera(&pb).is_some());
        assert_eq!(pb.camera.owner(&pb.index), Some(1), "a slot resolves to its owner group");
        assert!(pb.camera.is_first_person());
    }

    #[test]
    fn the_clock_runs_in_real_time_and_stops_at_the_end() {
        let mut p = Playback::new(index());
        assert_eq!(p.total_ticks(), 600);
        assert!((p.total_seconds() - 10.0).abs() < 1e-5);
        p.playing = true;
        p.advance(1.0);
        assert!((p.seconds() - 1.0).abs() < 1e-4, "one second of wall clock is one second of film");
        p.speed = 2.0;
        p.advance(1.0);
        assert!((p.seconds() - 3.0).abs() < 1e-4, "2x speed covers two seconds");
        // running past the end stops rather than looping
        p.advance(100.0);
        assert!(!p.playing);
        assert!((p.seconds() - (599.0 / 60.0)).abs() < 1e-3);
        // a paused clock does not move
        let before = p.tick;
        p.advance(5.0);
        assert_eq!(p.tick, before);
    }

    #[test]
    fn seeking_and_stepping_stay_in_range() {
        let mut p = Playback::new(index());
        p.seek_seconds(5.0);
        assert!((p.tick - 300.0).abs() < 1e-3);
        p.seek_seconds(-10.0);
        assert_eq!(p.tick, 0.0);
        p.seek_seconds(1000.0);
        assert_eq!(p.tick, 599.0);
        p.step(-1);
        assert_eq!(p.tick, 598.0);
        p.step(10_000);
        assert_eq!(p.tick, 599.0);
        p.seek_tick(0.0);
        p.step(-5);
        assert_eq!(p.tick, 0.0);
    }

    #[test]
    fn a_scrubbed_moment_reports_what_is_there() {
        let mut p = Playback::new(index());
        p.moving_only = true;
        p.show_all_types = false;
        // at tick 0 only biped 1 is alive
        p.seek_tick(0.0);
        let f = p.frame();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].net_id, 1);
        assert_eq!(f[0].pos, [0.0, 0.0, 0.0]);
        // halfway through its walk it is interpolated, not snapped
        p.seek_tick(150.0);
        let f = p.frame();
        assert!((f[0].pos[0] - 5.0).abs() < 0.01, "interpolated to {:?}", f[0].pos);
        // at 300 both bipeds are alive
        p.seek_tick(300.0);
        assert_eq!(p.frame().len(), 2);
        // the motionless object is filtered out by `moving_only`, and included when asked for
        p.show_all_types = true;
        assert_eq!(p.frame().len(), 2, "a motionless track is not motion");
        p.moving_only = false;
        assert_eq!(p.frame().len(), 3);
    }

    /// **The gate the old `wanted` filter fails.** A body that never moves is still a body -- a
    /// player shot the instant they spawned -- and the motion filter used to drop it from the
    /// frame entirely, while the weapon in its hands stayed on screen as its own `weap` track.
    /// That is "I see floating knives essentially, but not the full biped".
    #[test]
    fn a_body_that_never_moves_is_still_drawn() {
        let mut idx = index();
        // A player who spawns and is killed on the spot: two reports 0.2 wu apart, under the
        // half-world-unit `moves()` threshold.
        idx.tracks.push(track(4, BIPED_TYPE, &[(0, [7.0, 7.0, 0.0]), (600, [7.1, 7.1, 0.0])]));
        // ... and the energy sword they spawned holding, which does not move either.
        idx.tracks.push(track(5, 17, &[(0, [7.0, 7.0, 0.0]), (600, [7.1, 7.1, 0.0])]));
        let mut p = Playback::new(idx);
        p.moving_only = true;
        p.show_all_types = false;
        assert!(
            !p.index.tracks.iter().find(|t| t.net_id == 4).unwrap().moves(),
            "this fixture body is supposed to be motionless; the test is not checking what it claims"
        );
        p.seek_tick(300.0);
        let f = p.frame();
        assert!(
            f.iter().any(|b| b.net_id == 4),
            "a motionless BODY was filtered out of the frame: {:?}",
            f.iter().map(|b| b.net_id).collect::<Vec<_>>()
        );
        assert_eq!(f.len(), 3, "all three bodies alive at tick 300 must be in the frame");
        // The filter still does its real job on everything that is NOT a body, which is what it
        // was written for: the motionless `weap` and the motionless type-13 object stay out.
        assert!(!f.iter().any(|b| b.net_id == 5 || b.net_id == 3));
        p.show_all_types = true;
        assert_eq!(p.frame().len(), 3, "a motionless non-body is still not motion");
        p.moving_only = false;
        assert_eq!(p.frame().len(), 5, "with the filter off, every track alive at this tick is there");
    }

    /// #death A body whose track ENDS is drawn alive up to its death, lies on the ground until
    /// the tick the recording DESTROYS it, and then goes. Every one of those is a pure function
    /// of the tick, so scrubbing backwards brings the body back.
    #[test]
    fn a_destroyed_body_lies_where_it_fell_and_scrubbing_restores_it() {
        let mut idx = index();
        // The biped walks to (10,0,0) by tick 300, keeps sending until 500, is destroyed at 1000.
        idx.ticks = 2000;
        idx.tracks[0].last_tick = 500;
        idx.tracks[0].destroyed_tick = Some(1000);
        let death = crate::film_anim::death_tick(&idx.tracks[0]).expect("a destroyed track has a death");
        // The death is the LAST POSITION (300), not the last message (500) and not the destroy.
        assert_eq!(death, 300, "the death is the last reported position");
        let mut p = Playback::new(idx);
        // alive right up to the death...
        p.seek_tick(299.0);
        assert!(p.bipeds(16).iter().any(|b| b.net_id == 1), "alive before the death");
        assert!(p.corpses(8).is_empty(), "not a corpse yet");
        // ...and NOT drawn alive afterwards, which is the frozen standing body the old build left
        p.seek_tick(400.0);
        assert!(!p.bipeds(16).iter().any(|b| b.net_id == 1), "a dead body is not drawn alive");
        let c = p.corpses(8);
        assert_eq!(c.len(), 1, "it is a corpse");
        assert_eq!(c[0].dead_ticks, Some(100));
        assert_eq!(c[0].pos, [10.0, 0.0, 0.0], "it lies where it fell");
        // #death STILL THERE at 10 s, which the old flat 5 s window had already cleared: the
        // recording destroys this body at 1000 and the destroy is the DESPAWN, so that is when
        // the body goes.
        p.seek_tick(900.0);
        assert_eq!(p.corpses(8).len(), 1, "the body lies there until the recording deletes it");
        p.seek_tick(1000.0);
        assert_eq!(p.corpses(8).len(), 1, "up to and including the destroy tick");
        p.seek_tick(1001.0);
        assert!(p.corpses(8).is_empty(), "and is gone after it");
        // and scrubbing BACK restores exactly the same corpse
        p.seek_tick(400.0);
        assert_eq!(p.corpses(8), c, "scrubbing backwards restores the body");
        // the corpse cap never steals an actor from the living
        assert_eq!(p.bodies(16, 0).len(), p.bipeds(16).len(), "corpse cap 0 draws no bodies");
    }

    /// #film-action  THE WHOLE ASSASSINATION STAGING, end to end, from the recording's own
    /// `incident` row to the two placements the draw path needs.
    ///
    /// Every link in the chain has its own failure mode and the user has seen at least one of
    /// them ("the victim just flops on the ground"), so each is asserted separately rather than
    /// through one "it looks right" check:
    ///
    /// * the incident is recognised and the pair comes off it the right way round;
    /// * the victim is NOT drawn alive after the kill, and is NOT a corpse during the action;
    /// * both bodies ARE placed, on ONE shared root, for the whole 150-tick window;
    /// * the victim lies down only when the action ends;
    /// * and the clip names asked for are the two-party `sync_actions` pair, one per side.
    #[test]
    fn a_stated_assassination_stages_both_bodies_on_one_root_for_the_whole_window() {
        use crate::film_anim::SYNC_TICKS;
        const KILL: u32 = 500;
        let attacker = track(0x11, BIPED_TYPE, &[(0, [0.0, 0.0, 0.0]), (KILL, [10.0, 0.0, 0.0]), (900, [14.0, 0.0, 0.0])]);
        let mut victim = track(0x12, BIPED_TYPE, &[(0, [40.0, 0.0, 0.0]), (KILL, [10.6, 0.0, 0.0])]);
        victim.last_tick = KILL + 20;
        victim.destroyed_tick = Some(2000);
        let idx = FilmIndex {
            ticks: 3000,
            tracks: vec![attacker, victim],
            events: vec![crate::film_action::ActionEvent {
                tick: KILL,
                kind: crate::film_action::EV_INCIDENT,
                subject: Some(0x11),
                other: Some(0x12),
                incident: Some(0x0078_0015), // finishing_move_kill
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut p = Playback::new(idx);
        // 1. the recording's own statement, read the right way round
        let a = p.assassinations().to_vec();
        assert_eq!(a.len(), 1);
        assert_eq!((a[0].tick, a[0].attacker, a[0].victim), (KILL, Some(0x11), Some(0x12)));
        assert_eq!(a[0].incident, "finishing_move_kill");

        // 2. mid-action: BOTH bodies placed, on ONE root, and the victim is not a corpse
        for age in [1u32, SYNC_TICKS / 2, SYNC_TICKS - 1] {
            p.seek_tick((KILL + age) as f32);
            assert!(
                !p.frame().iter().any(|q| q.net_id == 0x12),
                "the victim must not be drawn alive {age} ticks into the action"
            );
            assert!(p.corpses(8).is_empty(), "and must not be a corpse yet at +{age}");
            let v = p.bodies_from(p.frame(), 16, 8);
            let att = v.iter().find(|q| q.net_id == 0x11).expect("attacker staged");
            let vic = v.iter().find(|q| q.net_id == 0x12).expect("victim staged");
            assert_eq!(att.pos, vic.pos, "both halves share one root at +{age}");
            assert_eq!(att.pos, [10.0, 0.0, 0.0], "which is the attacker's own transform at the kill");
            assert_eq!(v.iter().filter(|q| q.net_id == 0x12).count(), 1, "exactly one victim");
        }

        // 3. the window ENDS and the victim lies down -- and only then
        p.seek_tick((KILL + SYNC_TICKS) as f32);
        let v = p.bodies_from(p.frame(), 16, 8);
        assert!(!v.iter().any(|q| q.net_id == 0x12), "the action is over");
        p.seek_tick((KILL + SYNC_TICKS + 1) as f32);
        let c = p.corpses(8);
        assert_eq!(c.len(), 1, "now a corpse");
        assert_eq!(c[0].net_id, 0x12);
        // ...and it lies there until the recording DESTROYS it, not for a fixed 5 s
        p.seek_tick(1999.0);
        assert_eq!(p.corpses(8).len(), 1, "still there a long way past the old 5 s window");
        p.seek_tick(2001.0);
        assert!(p.corpses(8).is_empty(), "gone at the despawn");

        // 4. the clip names each side is asked for are the two-party pair, not an idle
        for (side, suffix) in [
            (crate::film_anim::SyncSide::Attacker, "_attacker"),
            (crate::film_anim::SyncSide::Victim, "_victim"),
        ] {
            let c = crate::film_anim::sync_candidates(false, false, side, 0);
            assert!(crate::film_anim::is_sync_clip(&c[0]), "{c:?}");
            assert!(c[0].ends_with(suffix), "{c:?}");
        }
    }

    /// #death The 5 s minimum is a FLOOR, not the window: a body the recording deletes almost
    /// immediately still lies there long enough for its death clip to play out.
    #[test]
    fn the_corpse_window_never_falls_below_the_death_clip() {
        let mut idx = index();
        idx.ticks = 2000;
        idx.tracks[0].last_tick = 310;
        // destroyed one second after the death
        idx.tracks[0].destroyed_tick = Some(360);
        let mut p = Playback::new(idx);
        p.seek_tick(350.0);
        assert_eq!(p.corpses(8).len(), 1);
        // the floor, not the destroy, decides
        p.seek_tick((300 + crate::film_anim::CORPSE_TICKS) as f32);
        assert_eq!(p.corpses(8).len(), 1, "the body lies there for at least the floor");
        p.seek_tick((300 + crate::film_anim::CORPSE_TICKS + 1) as f32);
        assert!(p.corpses(8).is_empty());
    }

    /// #death A track the recording never destroyed was still alive when the film stopped — it is
    /// not a death, and must not grow a corpse.
    #[test]
    fn a_body_the_film_never_destroyed_is_not_a_death() {
        let idx = index();
        assert!(crate::film_anim::death_tick(&idx.tracks[0]).is_none());
        let mut p = Playback::new(idx);
        p.seek_tick(500.0);
        assert!(p.corpses(8).is_empty());
        // ...and neither is a non-biped that was destroyed.
        let mut idx2 = index();
        idx2.tracks[2].destroyed_tick = Some(400);
        assert!(crate::film_anim::death_tick(&idx2.tracks[2]).is_none(), "only bipeds die");
    }

    /// #death The death clip plays ONCE and holds: the frame rises with the age of the death and
    /// never wraps, because a wrapped death clip makes the body fall over again and again.
    #[test]
    fn the_death_clip_advances_once_and_holds() {
        // the mapping sync applies: 60 Hz ticks -> 30 fps frames, clamped to the clip
        let frames_of = |age: u32, n: f32| (age as f32 / 2.0).min(n - 0.001).max(0.0);
        let n = 40.0;
        assert_eq!(frames_of(0, n), 0.0);
        assert_eq!(frames_of(60, n), 30.0);
        assert!(frames_of(120, n) > frames_of(60, n), "it advances");
        assert_eq!(frames_of(240, n), n - 0.001, "and then HOLDS the last frame");
        assert_eq!(frames_of(crate::film_anim::CORPSE_TICKS, n), n - 0.001);
    }

    /// #death The fall a body is given is a function of its entity id alone, so the same clip
    /// re-watched shows the same death — and the SET it is drawn from is this body's own graph's.
    #[test]
    fn the_death_variant_is_deterministic_and_covers_the_set() {
        let v = |id: u32| crate::film_anim::dead_state(id, 0, false, false).death_variant;
        assert_eq!(v(0x949d031d), v(0x949d031d));
        let seen: std::collections::HashSet<u8> = (0u32..4096).map(|i| v(i.wrapping_mul(0x9E3779B1))).collect();
        assert!(seen.len() >= 2, "the variants are not all the same clip: {seen:?}");
        for id in [0u32, 1, 0xFFFF_FFFF, 0x949d031d] {
            for (elite, airborne) in [(false, false), (false, true), (true, false), (true, true)] {
                let c = crate::film_anim::death_candidates(v(id), "any", elite, airborne);
                assert!(
                    crate::film_anim::is_death_clip(&c[0]),
                    "the first candidate is a death clip: {c:?}"
                );
                // every death clip of both graphs, in this body's own order, then the fallback
                assert_eq!(c.len(), 12, "eleven death clips plus the standing fallback: {c:?}");
                assert!(c.last().is_some_and(|l| l.ends_with(":idle")), "the fallback is last: {c:?}");
                // the body's OWN species leads, and the wanted orientation leads within it
                let want = match (elite, airborne) {
                    (false, false) => "combat:any:landing_dead",
                    (false, true) => "combat:any:airborne_dead",
                    (true, false) => "combat:landing_dead",
                    (true, true) => "combat:airborne_dead",
                };
                assert!(
                    c[0].starts_with(want) || c[..4].iter().any(|n| n.starts_with(want)),
                    "the own-graph set leads for ({elite}, {airborne}): {c:?}"
                );
            }
        }
        // #death An ELITE dies on a clip the Spartan list never had. This is the whole of
        // "elites don't ragdoll at all when they die": `combat:landing_dead` has no `any:`
        // field, so none of the three Spartan names matched and the body fell through to idle.
        let e = crate::film_anim::death_candidates(0, "any", true, false);
        assert_eq!(e[0], "combat:landing_dead");
        assert!(
            !crate::film_anim::death_candidates(0, "any", false, false)
                .iter()
                .take(3)
                .any(|n| n == "combat:landing_dead"),
            "and the Spartan's own set does not lead with it"
        );
    }

    #[test]
    fn bipeds_are_capped_and_stable() {
        let mut p = Playback::new(index());
        p.seek_tick(300.0);
        assert_eq!(p.bipeds(16).len(), 2);
        assert_eq!(p.bipeds(1).len(), 1);
        assert_eq!(p.bipeds(0).len(), 0);
        // the cap keeps the SAME entities frame to frame (sorted by id, not by chance)
        let a = p.bipeds(1);
        let b = p.bipeds(1);
        assert_eq!(a, b);
    }

    #[test]
    fn the_actor_pool_keeps_an_entity_on_its_own_actor() {
        let mut pool = ActorPool::default();
        let slots = [10u32, 11, 12];
        let mk = |id: u32| Placement { net_id: id, entity_type: 11, pos: [id as f32, 0.0, 0.0], yaw: 0.0, has_yaw: true, speed: 0.0, distance: 0.0, dead_ticks: None };
        // first frame: two bipeds, three slots -> one parked
        let a = pool.assign(&slots, &[mk(1), mk(2)]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.iter().filter(|(_, p)| p.is_some()).count(), 2);
        let slot_of_1 = a.iter().find(|(_, p)| p.map(|p| p.net_id) == Some(1)).unwrap().0;
        // next frame the order flips: entity 1 must stay on the SAME actor
        let b = pool.assign(&slots, &[mk(2), mk(1)]);
        let slot_of_1b = b.iter().find(|(_, p)| p.map(|p| p.net_id) == Some(1)).unwrap().0;
        assert_eq!(slot_of_1, slot_of_1b, "an entity hopped between actors");
        // more bipeds than slots: exactly `slots.len()` are shown, none twice
        let c = pool.assign(&slots, &[mk(1), mk(2), mk(3), mk(4), mk(5)]);
        let shown: Vec<u32> = c.iter().filter_map(|(_, p)| p.map(|p| p.net_id)).collect();
        assert_eq!(shown.len(), 3);
        let mut uniq = shown.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 3, "an entity was shown on two actors at once");
        // no bipeds at all: every slot parks
        let d = pool.assign(&slots, &[]);
        assert!(d.iter().all(|(_, p)| p.is_none()));
    }

    #[test]
    fn locomotion_ladder_follows_the_measured_speeds() {
        // Thresholds come from the real distribution (median 2.64 wu/s, p75 3.56, p90 3.91).
        assert_eq!(locomotion_for(0.0), "combat:rifle:idle");
        assert_eq!(locomotion_for(0.4), "combat:rifle:idle");
        assert_eq!(locomotion_for(1.0), "combat:rifle:walk_front_slow");
        assert_eq!(locomotion_for(2.64), "combat:rifle:move_front_slow");
        assert_eq!(locomotion_for(3.9), "combat:rifle:move_front_fast");
        assert_eq!(locomotion_for(8.0), "combat:rifle:move_front_fast");
        // the ladder is monotone: faster never picks a slower cycle
        let order = ["combat:rifle:idle", "combat:rifle:walk_front_slow", "combat:rifle:move_front_slow", "combat:rifle:move_front_fast"];
        let rank = |n: &str| order.iter().position(|o| *o == n).unwrap();
        let mut last = 0usize;
        for i in 0..=80 {
            let r = rank(locomotion_for(i as f32 * 0.1));
            assert!(r >= last, "speed {} went backwards down the ladder", i as f32 * 0.1);
            last = r;
        }
    }

    #[test]
    fn a_frame_carries_the_speed_that_drives_the_cycle() {
        let mut p = Playback::new(index());
        p.seek_tick(150.0);
        let f = p.frame();
        // the walking biped covers 10 wu over 300 ticks = 5 s -> 2 wu/s
        let b = f.iter().find(|q| q.net_id == 1).unwrap();
        assert!((b.speed - 2.0).abs() < 0.01, "speed {}", b.speed);
        assert!(b.distance > 0.0 && b.distance <= 10.0, "distance {}", b.distance);
        // a teleport must not read as a sprint
        assert!(f.iter().all(|q| q.speed <= 8.0));
    }

    #[test]
    fn times_format_for_a_scrub_bar() {
        assert_eq!(fmt_time(0.0), "0:00.0");
        assert_eq!(fmt_time(4.9), "0:04.9");
        assert_eq!(fmt_time(65.25), "1:05.2");
        assert_eq!(fmt_time(431.1), "7:11.1");
        assert_eq!(fmt_time(-5.0), "0:00.0");
    }

    #[test]
    fn status_mentions_the_time_and_the_count() {
        let mut p = Playback::new(index());
        p.seek_tick(300.0);
        let s = p.status();
        assert!(s.contains("0:05.0"), "{s}");
        assert!(s.contains("2 bipeds"), "{s}");
    }

    /// A biped walking a circle while a player's recorded aim follows the same circle: the
    /// correlation must pick that player's slot, and a decoy slot aiming the opposite way must
    /// lose. Nothing in a film states the player/entity link, so this is the test that the
    /// derivation actually derives it.
    fn aim_index() -> FilmIndex {
        use crate::film_decode::PlayerAim;
        let mut idx = FilmIndex { ticks: 600, ..Default::default() };
        let mut tr = EntityTrack { net_id: 0x55, entity_type: 11, first_tick: 0, last_tick: 599, ..Default::default() };
        let mut slot0 = Vec::new();
        let mut slot1 = Vec::new();
        for t in 0..600u32 {
            let a = t as f32 * 0.01;
            tr.positions.push((t, [a.cos(), a.sin(), 0.0]));
            tr.orientations.push((t, [a.cos(), a.sin(), 0.0], [0.0, 0.0, 1.0]));
            let q = |ang: f32| ((ang.rem_euclid(std::f32::consts::TAU)) / std::f32::consts::TAU * 8192.0) as u16;
            // 2148 is 100 steps ABOVE level (2048), i.e. a modest upward aim.
            slot0.push((t, PlayerAim { slot: 0, yaw_q: q(a), pitch_q: 2148, ..Default::default() }));
            slot1.push((t, PlayerAim {
                slot: 1,
                yaw_q: q(a + std::f32::consts::PI),
                pitch_q: crate::film_decode::PITCH_LEVEL_Q,
                ..Default::default()
            }));
        }
        tr.last_tick = 599;
        idx.tracks.push(tr);
        idx.player_aims = vec![Vec::new(); 16];
        idx.player_aims[0] = slot0;
        idx.player_aims[1] = slot1;
        idx.bind_players();
        idx
    }

    #[test]
    fn the_player_driving_a_biped_is_derived_from_the_aim() {
        let idx = aim_index();
        assert_eq!(idx.bindings.len(), 1, "one biped, one binding: {:?}", idx.bindings);
        let b = idx.bindings[0];
        assert_eq!(b.slot, 0, "the slot whose aim matches must win, not the one aiming backwards");
        assert_eq!(b.net_id, 0x55);
        assert!(b.cost < 0.05, "a facing that tracks the aim exactly should cost almost nothing: {}", b.cost);
        assert!(b.margin > 2.0, "the decoy aiming 180 degrees away must lose by a lot: {}", b.margin);
        assert!(b.from_facing, "this track has orientations, so the facing should have been used");
        assert!(b.is_confident());
        assert_eq!(idx.slot_of_entity(0x55), Some(0));
        assert!(idx.entity_for_slot(0, 300).is_some());
        assert!(idx.entity_for_slot(1, 300).is_none(), "slot 1 drove nothing");
    }

    #[test]
    fn first_person_sits_at_eye_height_and_looks_along_the_recorded_aim() {
        let mut p = Playback::new(aim_index());
        p.seek_tick(0.0);
        p.camera = FilmCamera::FirstPerson(0);
        let pose = solve_camera(&p).expect("slot 0 has a biped at tick 0");
        // tick 0: the biped is at (1, 0, 0) and the aim yaw is 0.
        assert!((pose.pos[0] - 1.0).abs() < 1e-3 && pose.pos[1].abs() < 1e-3, "{:?}", pose.pos);
        assert!(
            (pose.pos[2] - EYE_HEIGHT_WU).abs() < 1e-4,
            "the stream gives the FEET, so the eye must be lifted: {:?}",
            pose.pos
        );
        assert!(pose.yaw.abs() < 0.01, "yaw should be the recorded aim: {}", pose.yaw);
        assert!(
            pose.pitch > 0.1 && pose.pitch < 0.2,
            "100 steps above level is a modest upward aim, not the sky: {}",
            pose.pitch
        );
        assert!(pose.label.contains("recorded aim"), "{}", pose.label);
    }

    #[test]
    fn follow_sits_behind_the_biped_along_its_aim() {
        let mut p = Playback::new(aim_index());
        p.seek_tick(0.0);
        p.camera = FilmCamera::Follow(0);
        let pose = solve_camera(&p).expect("slot 0 has a biped at tick 0");
        // Looking along +X from (1,0,0), so the camera is directly behind on -X, at the rig's
        // own distance scaled by its elevation.
        let back = FOLLOW_DIST_WU * FOLLOW_ELEV_DEG.to_radians().cos();
        assert!((pose.pos[0] - (1.0 - back)).abs() < 1e-3, "{:?}", pose.pos);
        // It now sits at about the player's own eye height, not over his head: the old rig was
        // at EYE_HEIGHT + FOLLOW_UP = 1.10 wu, roughly 0.4 wu above a 0.7 wu Spartan.
        assert!(
            (pose.pos[2] - EYE_HEIGHT_WU).abs() < 0.08,
            "the follow camera should sit at about eye height: {:?}",
            pose.pos
        );
    }

    #[test]
    fn the_free_camera_is_never_moved_and_an_unmatched_player_says_so() {
        let mut p = Playback::new(aim_index());
        p.camera = FilmCamera::Free;
        assert!(solve_camera(&p).is_none(), "free mode must leave the user's camera alone");
        assert_eq!(camera_status(&p), "free camera");
        // Slot 1 drove nothing in this film: no pose, and the status explains why rather than
        // silently doing nothing.
        p.camera = FilmCamera::FirstPerson(1);
        assert!(solve_camera(&p).is_none());
        let st = camera_status(&p);
        assert!(st.contains("no biped could be matched"), "{st}");
    }

    #[test]
    fn a_player_with_no_biped_yet_is_reported_with_the_time_it_spawns() {
        // One life, starting at tick 300: before that the camera has nothing to sit on, and the
        // status must say when it will -- players do not exist at the start of a round.
        let mut idx = aim_index();
        for b in idx.bindings.iter_mut() {
            b.first_tick = 300;
        }
        idx.tracks[0].first_tick = 300;
        let mut p = Playback::new(idx);
        p.camera = FilmCamera::FirstPerson(0);
        p.seek_tick(60.0);
        assert!(solve_camera(&p).is_none());
        let st = camera_status(&p);
        assert!(st.contains("no biped at"), "{st}");
        assert!(st.contains("next at 0:05.0"), "{st}");
    }

    #[test]
    fn first_person_does_not_draw_its_own_body() {
        let mut p = Playback::new(aim_index());
        p.seek_tick(10.0);
        p.moving_only = false;
        let all = p.bipeds(16).len();
        assert_eq!(all, 1, "the fixture has one biped");
        assert_eq!(p.hidden_entity(), None, "nothing is hidden with a free camera");
        p.camera = FilmCamera::FirstPerson(0);
        assert_eq!(p.hidden_entity(), Some(0x55), "the subject's own body is hidden");
        assert!(p.bipeds(16).is_empty(), "and it is not handed to the actor pool");
        // A follow camera is third person, so the body stays.
        p.camera = FilmCamera::Follow(0);
        assert_eq!(p.hidden_entity(), None);
        assert_eq!(p.bipeds(16).len(), 1);
        // Another player's first person does not hide this body.
        p.camera = FilmCamera::FirstPerson(1);
        assert_eq!(p.hidden_entity(), None, "slot 1 drives nothing in the fixture");
        assert_eq!(p.bipeds(16).len(), 1);
    }

    /// #fp-anim  EVERY clip the first-person selector can pick as a BASE is a BASE clip in the
    /// shipped graph, and every clip it composes is an OVERLAY.
    ///
    /// This is the gate on the bug the user reported as "the arms just stick straight out, none
    /// of the animations are working". `clip_trs` composes an OVERLAY onto the skeleton's
    /// authored default, so a selector that returns `first_person:moving` (type 2) as a body's
    /// only clip draws the arms' bind pose -- both arms straight out -- and that is what it
    /// returned for any body that was moving, which in a 16-player clip is nearly every body on
    /// nearly every frame. Driving the selector over every combination of its inputs and checking
    /// each name's ANIMATION TYPE in a real weapon graph fails on that selector and passes on
    /// this one.
    ///
    /// IGNORED BY DEFAULT: it opens a Reach cache, and the native parser's process-global state
    /// collides with `reach_actor`'s own cache tests. Run it alone:
    ///
    /// ```text
    /// cargo test -p hms-app first_person_base -- --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "opens a Reach cache; the native parser's global state collides with reach_actor's cache tests"]
    fn a_first_person_base_clip_is_a_base_clip_and_an_overlay_is_an_overlay() {
        use crate::reach_actor::{AnimGraph, first_person_weapon};
        use crate::reach_jmad::AnimType;
        // Open any shipped Reach cache, exactly as `reach_widget`'s own cache test does.
        let open_any_reach = || -> Option<(hms_native::NativeDll, hms_native::CacheHandle)> {
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
            roots.push(std::path::PathBuf::from("target/release"));
            let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
            let native = hms_native::NativeDll::load(&dll).ok()?;
            let map = crate::mapcat::enumerate().into_iter().find(|c| c.game == crate::mapcat::Game::Reach)?;
            native.set_shared_cache_dirs(&[map.path.parent()?.to_path_buf()]);
            let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
            Some((native, cache))
        };
        let Some((native, cache)) = open_any_reach() else {
            eprintln!("no Reach cache on this machine - skipped");
            return;
        };
        // Every combination of the selector's inputs, including the ones that cannot co-occur:
        // the function must be total, and a future edit must not be able to slip an overlay in.
        let mut bases: Vec<Vec<&'static str>> = Vec::new();
        let mut overlays: Vec<&'static str> = Vec::new();
        for melee in [None, Some(0u32), Some(7), Some(14)] {
            for fired in [None, Some(0u32), Some(9)] {
                for sprinting in [false, true] {
                    for airborne in [false, true] {
                        for moving in [false, true] {
                            let (b, _, ov) = super::ActorDriver::fp_clips(melee, fired, sprinting, airborne, moving);
                            if !bases.contains(&b) { bases.push(b); }
                            for o in ov { if !overlays.contains(&o) { overlays.push(o); } }
                        }
                    }
                }
            }
        }
        assert!(bases.len() >= 4, "the selector collapsed to {bases:?}");
        assert!(!overlays.is_empty(), "the selector composes no overlays at all");
        // Both species' graphs, on every weapon that authors one.
        let mut checked = 0usize;
        // Per overlay name, how many graphs do NOT have it -- a typo would make this ALL of them,
        // which is the failure a "missing is fine" rule would otherwise hide.
        let mut missing: std::collections::BTreeMap<&str, usize> = Default::default();
        for t in native.tags_by_group(cache, "weap") {
            let (_, Some(j)) = first_person_weapon(&native, cache, t, 0) else { continue };
            let Some(g) = AnimGraph::read(&native, cache, j) else { continue };
            let path = native.tag_name(cache, j).unwrap_or_default();
            // A graph that has none of these names is not a first-person graph; skip it rather
            // than failing on an unrelated tag.
            if g.find("first_person:idle").is_none() { continue; }
            checked += 1;
            for cands in &bases {
                // At least ONE candidate must exist in this graph -- a weapon that authors none
                // of them would leave the rig on a stale pose -- and whichever one the resolver
                // would pick must be a BASE clip, because an overlay composed alone draws the
                // bind pose (both arms straight out), which is the bug this gate exists for.
                let Some(i) = cands.iter().find_map(|c| g.find(c)) else {
                    panic!(
                        "{path}: none of the selector's base candidates {cands:?} exist in this graph"
                    );
                };
                assert_eq!(
                    g.anims[i].anim_type, AnimType::Base,
                    "{path}: '{}' is chosen as the BASE clip but the tag marks it {:?} - composing it alone draws the bind pose",
                    g.anims[i].name, g.anims[i].anim_type
                );
            }
            for o in &overlays {
                // A graph MAY legitimately lack one: the oddball's first-person set has no
                // `first_person:fall`, and `clock_overlay` declines a clip that is not there, so
                // the gait simply is not composed. What must never happen is composing something
                // the tag does not mark as an OVERLAY -- that is the bind-pose bug in reverse.
                let Some(i) = g.find(o) else {
                    *missing.entry(*o).or_insert(0usize) += 1;
                    continue;
                };
                assert_eq!(
                    g.anims[i].anim_type, AnimType::Overlay,
                    "{path}: '{o}' is composed as an overlay but the tag marks it {:?}",
                    g.anims[i].anim_type
                );
            }
        }
        assert!(checked >= 10, "only {checked} first-person graph(s) checked - the cache lookup is not doing its job");
        for o in &overlays {
            let miss = missing.get(o).copied().unwrap_or(0);
            assert!(
                miss * 2 < checked,
                "overlay '{o}' is absent from {miss} of {checked} first-person graphs - that is a \
                 misspelt clip name, not a weapon that happens not to author it"
            );
        }
        eprintln!(
            "{checked} first-person graph(s): bases {bases:?} all resolve to a BASE clip; \
             overlays {overlays:?} all OVERLAY where present (absent: {missing:?})"
        );
    }

    // ---------------------------------------------------------------------------------------
    // #fp-frame  THE FIRST-PERSON FRAMING GATE
    //
    // "The arms are drawn, skinned, animated and 47 of 47 nodes are driven" was TRUE for a whole
    // day while the user saw no arms at all, because every check asked whether the rig was POSED
    // and none asked whether it was IN FRAME. These project the rig through the real placement
    // and the real projection and assert where it lands on screen.
    // ---------------------------------------------------------------------------------------

    /// The MEASURED posed model-space positions of the first-person rig, `forge_halo`, Spartan
    /// arms + energy sword on `first_person:idle` (the capture these tests were written from;
    /// `film fp verbose` reprints them). Model space is the rig's own: **+X forward**, +Y left,
    /// +Z up — see [`crate::reach_actor::FP_ARMS_MODEL_YAW`].
    const FP_POSED_ARMS: &[(&str, [f32; 3])] = &[
        ("l_hand", [0.199, 0.105, -0.050]),
        ("r_hand", [0.158, -0.113, -0.097]),
        ("l_forearm", [0.125, 0.152, -0.114]),
        ("l_index_tip", [0.219, 0.076, 0.005]),
    ];
    /// The sword's own two nodes in the WEAPON model's space — i.e. LOCAL to the hand the graph
    /// parents them to, which is the whole point of `#fp-attach`.
    const FP_POSED_WEAPON_LOCAL: &[(&str, [f32; 3])] = &[
        ("b_handle", [0.033, 0.010, -0.002]),
        ("b_blades", [0.150, 0.007, 0.002]),
    ];
    /// The same two nodes where the attach actually puts them, in the RIG's space — measured off
    /// the running app (`film fp verbose`, which prints them camera-relative as depth / right / up;
    /// with the rig mounted on the view those are +X / -Y / +Z).
    const FP_POSED_WEAPON_RIG: &[(&str, [f32; 3])] = &[
        ("b_handle", [0.184, -0.100, -0.079]),
        ("b_blades", [0.263, -0.087, 0.008]),
    ];

    /// Project a world point through `hms_render::Camera`'s own view-projection and return its
    /// normalised device coordinates plus its view-space depth.
    fn project(cam: &hms_render::Camera, aspect: f32, p: glam::Vec3) -> (f32, f32, f32) {
        let clip = cam.view_proj(aspect) * p.extend(1.0);
        (clip.x / clip.w, clip.y / clip.w, clip.w)
    }

    /// A first-person camera the way `solve_camera` builds one, plus Reach's own field of view.
    fn fp_camera() -> hms_render::Camera {
        hms_render::Camera {
            pos: glam::Vec3::new(10.0, 20.0, 1.0 + EYE_HEIGHT_WU),
            yaw: 0.7,
            pitch: -0.15,
            fov_y: reach_vertical_fov(),
            ..Default::default()
        }
    }

    /// Reach's field of view, from the globals tag through the engine's own derivation.
    ///
    /// Gate: HMS used its editor default of 70 degrees VERTICAL (about 102 horizontal at 16:9) for
    /// a first-person Theater view, which is nothing like the game.
    #[test]
    fn the_first_person_field_of_view_is_reachs_own() {
        assert!(
            (REACH_FIELD_OF_VIEW_DEG - 78.0).abs() < 1e-6,
            "the authored globals FOV is 78 degrees (matg camera block +0x24)"
        );
        let v = reach_vertical_fov().to_degrees();
        assert!(
            (v - 49.0969).abs() < 0.01,
            "Reach's vertical FOV is 2*atan(tan(78/2)/(4/3))*0.785 = 49.097 degrees, got {v:.4}"
        );
        let fp = reach_first_person_model_fov().to_degrees();
        assert!(
            (fp - 43.2053).abs() < 0.01,
            "the first-person model pass is 0.88 of that = 43.205 degrees, got {fp:.4}"
        );
        // Hor+: the vertical is fixed and the horizontal grows with the window, so the authored
        // 78 reappears as the TRUE horizontal at 16:9.
        for (a, want) in [(16.0 / 9.0, 78.153), (4.0 / 3.0, 62.683), (21.0 / 9.0, 93.64)] {
            let h = 2.0 * ((a * (reach_vertical_fov() * 0.5).tan()).atan()).to_degrees();
            assert!((h - want).abs() < 0.05, "horizontal FOV at {a:.3} should be {want}, got {h:.3}");
        }
        assert!(
            reach_vertical_fov() < hms_render::Camera::default().fov_y * 0.8,
            "Reach is far narrower than the editor camera; if this ever passes by accident the \
             editor default has moved and the first-person path needs looking at"
        );
    }

    /// #fp-fov **THE GATE for "the first-person arms look really weird at 120 FOV".**
    ///
    /// Reach draws the first-person MODEL through its own camera, at a narrower VERTICAL field of
    /// view than the world — confirmed off the machine code, not remembered: the one reference to
    /// the literal `0.690800011` in `haloreach.dll` is at `+0x286CFC`, inside the function that
    /// copies the world camera into the first-person camera and then does
    ///
    /// ```text
    ///   v3 = 0.690800011 / fmaxf(fp_camera.vfov_scale, 1e-4);   // +0x2C, the world camera's 0.785
    ///   render_first_person_fov_scale = v3;                     // the debug global, +0xB445F4
    ///   fp_camera.vertical_fov *= v3;                           // +0x28
    /// ```
    ///
    /// So it is a CONSTANT RATIO on the world camera's vertical FOV — `0.6908 / 0.785` = 0.88 —
    /// and not a fixed model FOV. HMS drew the rig through the world camera, which is what makes
    /// a viewmodel splay and shrink as the FOV slider widens, and this pins the whole chain across
    /// the slider rather than at one setting: the ERROR a shared camera makes grows from 14.6% at
    /// 60, through 15.3% at the authored 78, to **19.3% at 120** — which is the complaint.
    #[test]
    fn the_first_person_model_is_drawn_at_the_engines_own_narrower_fov() {
        assert!((REACH_FIRST_PERSON_FOV_SCALE - 0.6908 / REACH_FOV_SCALE).abs() < 1e-4,
            "the scale is the engine's `0.6908 / vfov_scale`, got {REACH_FIRST_PERSON_FOV_SCALE}");
        // The whole slider, as the engine's chain computes it. `size` is how much bigger a point a
        // fixed distance in front of the eye is drawn by the rig camera than by the world camera
        // — i.e. exactly how much too SMALL the arms were.
        let rows: [(f32, f32, f32, f32); 4] = [
            // (slider horizontal deg, world vertical deg, rig vertical deg, size ratio)
            (60.0, 36.759, 32.348, 1.1456),
            (78.0, 49.097, 43.205, 1.1535),
            (100.0, 65.612, 57.738, 1.1692),
            (120.0, 82.285, 72.411, 1.1935),
        ];
        let mut last = 0.0f32;
        for (h, want_world, want_fp, want_size) in rows {
            let w = reach_vertical_fov_for(h).to_degrees();
            let f = reach_first_person_model_fov_for(h).to_degrees();
            assert!((w - want_world).abs() < 0.01, "world vertical FOV at {h}: want {want_world}, got {w:.3}");
            assert!((f - want_fp).abs() < 0.01, "rig vertical FOV at {h}: want {want_fp}, got {f:.3}");
            // The scale is on the VERTICAL FOV, so the ratio is of the half-angle TANGENTS. If it
            // were ever applied to the horizontal, or before the 4:3 derivation, these move.
            let size = (w.to_radians() * 0.5).tan() / (f.to_radians() * 0.5).tan();
            assert!((size - want_size).abs() < 0.002, "rig size ratio at {h}: want {want_size}, got {size:.4}");
            assert!(size > last, "the error a SHARED camera makes must grow with the FOV ({h} deg)");
            last = size;
        }
        // ...and it is a RATIO, not a fixed model FOV: the rig's FOV has to move with the slider,
        // or the arms would be identical at 60 and 120 while the world around them changed.
        assert!(reach_first_person_model_fov_for(120.0) > reach_first_person_model_fov_for(60.0) * 1.5,
            "the rig's own FOV tracks the slider");
        // The ratio itself never moves (a constant factor is what makes this safe to apply
        // everywhere on the slider rather than a correction tuned at one end).
        for h in [FOV_MIN_DEG, 70.0, 90.0, FOV_MAX_DEG] {
            let r = reach_first_person_model_fov_for(h) / reach_vertical_fov_for(h);
            assert!((r - REACH_FIRST_PERSON_FOV_SCALE).abs() < 1e-5, "ratio at {h}: {r}");
        }
    }

    /// The first-person camera sits at the biped's OWN authored camera height, and the two species
    /// differ.
    ///
    /// Gate: HMS lifted every first-person view by a guessed 0.55 wu, 0.05 short for a Spartan and
    /// 0.14 short for an Elite.
    #[test]
    fn the_first_person_eye_is_at_the_bipeds_authored_camera_height() {
        assert!(
            (EYE_HEIGHT_WU - 0.60).abs() < 1e-6,
            "`bipd` +0x524 of objects\\characters\\spartans\\spartans is 0.600000024"
        );
        assert!(
            (ELITE_EYE_HEIGHT_WU - 0.69).abs() < 1e-6,
            "`bipd` +0x524 of objects\\characters\\elite\\elite is 0.689999998"
        );
        assert!(
            ELITE_EYE_HEIGHT_WU > EYE_HEIGHT_WU + 0.05,
            "an Elite's eye is meaningfully higher than a Spartan's; if these ever collapse to one \
             number the species lane has been lost"
        );
        // And the solver actually uses what it is handed, rather than a constant of its own.
        let mut idx = FilmIndex { ticks: 60, ..Default::default() };
        let mut tr = EntityTrack { net_id: 7, entity_type: 11, first_tick: 0, last_tick: 59, ..Default::default() };
        for t in 0..60u32 {
            tr.positions.push((t, [1.0, 2.0, 3.0]));
        }
        tr.owner_raw = Some(4);
        idx.tracks.push(tr);
        let mut pb = Playback::new(idx);
        pb.camera = FilmCamera::FirstPersonOwner(4);
        pb.tick = 30.0;
        for h in [EYE_HEIGHT_WU, ELITE_EYE_HEIGHT_WU] {
            let p = solve_camera_at(&pb, h).expect("a first-person pose");
            assert!(
                (p.pos[2] - (3.0 + h)).abs() < 1e-5,
                "the eye should be {h} wu above the feet, it is at {:.3}",
                p.pos[2]
            );
        }
    }

    /// #fp-fov  **THE FIRST-PERSON RIG IS DRAWN NARROWER THAN THE WORLD**, by the engine's own
    /// factor, at every field of view the control offers — including the 120 this user plays at.
    ///
    /// Written because the doc comment on [`REACH_FIRST_PERSON_FOV_SCALE`] claimed for a while
    /// that HMS did not apply it, and that claim outlived the feature and misled two agents into
    /// blaming a wide viewmodel for something else. A gate cannot be misled by a comment.
    ///
    /// **What this gate CANNOT reach, stated rather than implied:** it exercises the FOV CHAIN,
    /// not the renderer. `SceneRenderer::set_first_person_fov_scale` and the second camera bind
    /// group need a GPU device, so "the rig's meshes are bound to the narrow camera" is checked
    /// by reading the three call sites (`main.rs`, `headless.rs`, `script_host.rs`) and by the
    /// renderer's own `first_person_view()` readout in the live window, not here. The env
    /// override exists for exactly that: `HMS_FP_FOV_SCALE=1` draws the rig through the world
    /// camera, and the difference is visible immediately.
    #[test]
    fn the_first_person_rig_is_drawn_narrower_than_the_world() {
        // 1. The factor is the engine's, unconditionally — the user double-clicks the app and
        //    never sets an environment variable, so the DEFAULT has to be right.
        assert_eq!(
            REACH_FIRST_PERSON_FOV_SCALE, 0.88,
            "the engine's `render_first_person_view` does `vfov *= 0.6908 / 0.785` = 0.88"
        );
        assert!(
            (0.690_800_011f32 / REACH_FOV_SCALE - REACH_FIRST_PERSON_FOV_SCALE).abs() < 1e-6,
            "0.6908 / {REACH_FOV_SCALE} must be {REACH_FIRST_PERSON_FOV_SCALE}"
        );
        // 2. ...and nothing in the environment of a plain run changes it.
        assert_eq!(first_person_fov_scale(), REACH_FIRST_PERSON_FOV_SCALE);
        // 3. The rig is NARROWER than the world at every setting the control offers, by exactly
        //    that factor. Worked out from the chain, not read back: at the authored 78 the world
        //    is 49.097 degrees vertical and the rig 43.205; at this user's 120 the world is
        //    82.3 and the rig 72.4.
        for deg in [FOV_MIN_DEG, REACH_FIELD_OF_VIEW_DEG, 105.0, FOV_MAX_DEG] {
            let world = reach_vertical_fov_for(deg);
            let rig = reach_first_person_model_fov_for(deg);
            assert!(
                (rig - world * REACH_FIRST_PERSON_FOV_SCALE).abs() < 1e-6,
                "at {deg} deg the rig's vfov {rig} is not {REACH_FIRST_PERSON_FOV_SCALE} of the world's {world}"
            );
            assert!(rig < world, "at {deg} deg the rig must be narrower than the world");
        }
        // 4. The two numbers, as literals, so a change to the chain has to be deliberate.
        assert!(
            (reach_vertical_fov_for(78.0).to_degrees() - 49.097).abs() < 0.01,
            "the world vfov at 78 should be 49.097 deg, got {}",
            reach_vertical_fov_for(78.0).to_degrees()
        );
        assert!(
            (reach_first_person_model_fov_for(78.0).to_degrees() - 43.205).abs() < 0.01,
            "the rig vfov at 78 should be 43.205 deg, got {}",
            reach_first_person_model_fov_for(78.0).to_degrees()
        );
        assert!(
            (reach_first_person_model_fov_for(120.0).to_degrees() - 72.42).abs() < 0.05,
            "the rig vfov at the user's 120 should be 72.4 deg, got {}",
            reach_first_person_model_fov_for(120.0).to_degrees()
        );
    }

    /// #fp-fov  The user-facing field-of-view control: the game's own units, the game's own
    /// default, and it persists.
    #[test]
    fn the_first_person_fov_control_speaks_reachs_own_units() {
        // The default IS the authored value, so the app starts out framing like the game.
        assert!((REACH_FIELD_OF_VIEW_DEG - 78.0).abs() < 1e-6);
        assert!(
            (reach_vertical_fov_for(REACH_FIELD_OF_VIEW_DEG) - reach_vertical_fov()).abs() < 1e-9,
            "the default of the control must be exactly the engine value, not a near miss"
        );
        // Wider in means wider out, and the engine's own chain all the way.
        let (a, b) = (reach_vertical_fov_for(78.0), reach_vertical_fov_for(100.0));
        assert!(b > a, "a larger FOV must widen the view");
        for h in [70.0f32, 78.0, 90.0, 110.0] {
            let want = 2.0 * ((h.to_radians() * 0.5).tan() / (4.0 / 3.0)).atan() * 0.785;
            assert!((reach_vertical_fov_for(h) - want).abs() < 1e-6, "{h}: engine chain");
        }
        // Out of range is clamped rather than obeyed -- the engine itself refuses anything
        // outside 1..150 and substitutes 78.
        assert_eq!(reach_vertical_fov_for(5.0), reach_vertical_fov_for(FOV_MIN_DEG));
        assert_eq!(reach_vertical_fov_for(400.0), reach_vertical_fov_for(FOV_MAX_DEG));
        // And it persists: the control would be useless to someone who double-clicks the app.
        let dir = std::env::temp_dir().join(format!("hms-fov-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert!(
            (load_fov_from_dir(&dir) - REACH_FIELD_OF_VIEW_DEG).abs() < 1e-6,
            "with nothing saved the control reads the engine default"
        );
        save_fov_to_dir(&dir, 95.0);
        assert!((load_fov_from_dir(&dir) - 95.0).abs() < 0.05, "a saved FOV must come back");
        save_fov_to_dir(&dir, 999.0);
        assert!(
            (load_fov_from_dir(&dir) - FOV_MAX_DEG).abs() < 0.05,
            "an out-of-range save is clamped, not stored"
        );
        std::fs::write(dir.join("first_person_fov.txt"), "not a number").unwrap();
        assert!(
            (load_fov_from_dir(&dir) - REACH_FIELD_OF_VIEW_DEG).abs() < 1e-6,
            "a corrupt setting falls back to the engine default rather than to zero"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **The gate that would have caught it.** The first-person rig, mounted the way
    /// `sync_first_person` mounts it and projected through Reach's own field of view, has to be IN
    /// FRONT of the camera everywhere and ON SCREEN where the measurement says it is.
    ///
    /// With the old `FP_ARMS_MODEL_YAW` of -90 the whole rig was turned 90 degrees to the right:
    /// `l_hand` projected to `ndc_x` +2.3, the sword to +17, and `r_hand` went BEHIND the camera
    /// entirely. Not one node of either model was inside the frame, which is exactly what the user
    /// saw. Both halves of that are checked here, because either on its own can be got right by
    /// accident.
    #[test]
    fn the_first_person_rig_lands_inside_the_viewport_at_every_aspect() {
        use crate::reach_actor::{actor_world, FP_ARMS_MODEL_YAW};
        let cam = fp_camera();
        // The mount `sync_first_person` applies: the solved camera's own position and angles.
        let mount = actor_world(cam.pos, cam.yaw.to_degrees(), cam.pitch.to_degrees(), FP_ARMS_MODEL_YAW);
        // The arms' own nodes, plus the weapon's where `#fp-attach` puts them.
        let every: Vec<(&str, glam::Vec3)> = FP_POSED_ARMS
            .iter()
            .chain(FP_POSED_WEAPON_RIG.iter())
            .map(|(n, p)| (*n, mount.transform_point3(glam::Vec3::from(*p))))
            .collect();
        // The nodes the measurement puts ON SCREEN. `r_hand` is deliberately NOT one of them: on
        // this clip it is the hand gripping the hilt and it sits just below the bottom edge
        // (ndc_y -1.34), which is where Reach puts it too.
        // `l_forearm` is not one either: it is the elbow, close enough to the eye that it
        // projects past the edge while the limb it starts still crosses the frame.
        //
        // The aspects are 4:3 and wider. Reach is Hor+, so the VERTICAL framing is the same at all
        // of them and narrower-than-4:3 windows crop the sides -- in the engine too.
        const ON_SCREEN: &[&str] = &["l_hand", "l_index_tip", "b_handle", "b_blades"];
        for &aspect in &[16.0 / 9.0_f32, 1100.0 / 750.0, 4.0 / 3.0] {
            for (name, w) in &every {
                let (x, y, d) = project(&cam, aspect, *w);
                assert!(
                    d > cam.near,
                    "{name} is behind the near plane at aspect {aspect:.3} (depth {d:.3}) -- the \
                     rig is not facing the way the view is"
                );
                if ON_SCREEN.contains(name) {
                    assert!(
                        x.abs() <= 1.0 && y.abs() <= 1.0,
                        "the first-person rig is OUT OF FRAME: {name} projects to ndc ({x:+.2}, \
                         {y:+.2}) at aspect {aspect:.3} -- it has to be inside [-1, 1]"
                    );
                }
            }
        }
    }

    /// The rig is in the LOWER part of the view and straddles it, the way a pair of first-person
    /// arms does -- not clinging to one edge.
    ///
    /// Gate: the broken mount put every node on the right-hand side (`ndc_x` +0.42 .. +8.40, never
    /// once negative), which this rejects even if some of it had stayed on screen.
    #[test]
    fn the_first_person_rig_frames_like_a_pair_of_arms() {
        use crate::reach_actor::{actor_world, FP_ARMS_MODEL_YAW};
        let cam = fp_camera();
        let mount = actor_world(cam.pos, cam.yaw.to_degrees(), cam.pitch.to_degrees(), FP_ARMS_MODEL_YAW);
        let aspect = 16.0 / 9.0;
        let ndc: Vec<(&str, f32, f32)> = FP_POSED_ARMS
            .iter()
            .map(|(n, p)| {
                let (x, y, _) = project(&cam, aspect, mount.transform_point3(glam::Vec3::from(*p)));
                (*n, x, y)
            })
            .collect();
        assert!(
            ndc.iter().any(|(_, x, _)| *x < -0.1) && ndc.iter().any(|(_, x, _)| *x > 0.1),
            "a left hand and a right hand straddle the view; this rig is all on one side: {ndc:?}"
        );
        assert!(
            ndc.iter().all(|(_, _, y)| *y < 0.25),
            "the arms belong in the LOWER part of the frame, not across the middle: {ndc:?}"
        );
        let (_, _, hand_y) = *ndc.iter().find(|(n, _, _)| *n == "l_hand").expect("l_hand");
        assert!(
            (-0.95..=-0.2).contains(&hand_y),
            "the near hand should sit in the lower third of the frame; it is at ndc_y {hand_y:+.2}"
        );
    }

    /// #fp-attach  The weapon's MESH is baked where the gun is HELD, not where the rig is mounted.
    ///
    /// Gate: `build_actor_meshes` took the mesh BASIS from the actor's anchor but the mesh ORIGIN
    /// from `pos`. Those are the same point for everything placed in the world, but a first-person
    /// weapon is parented to the hand while its `pos` is still the eye — so every rebuild (a weapon
    /// swap, an armour install) baked the gun at the CAMERA and the palette's `anchor^-1 * world`
    /// fold-in came out as identity, leaving it there: "the weapon isn't held properly, it just
    /// floats there", enormous because it was centimetres from the lens. The first build of a fresh
    /// scene happens BEFORE the attach resolves, which is why a one-shot scripted capture passed.
    #[test]
    fn a_first_person_weapons_mesh_is_baked_where_it_is_held() {
        use crate::reach_actor::{actor_mesh_frame, actor_world, FP_ARMS_MODEL_YAW};
        let cam = fp_camera();
        let mount = actor_world(cam.pos, cam.yaw.to_degrees(), cam.pitch.to_degrees(), FP_ARMS_MODEL_YAW);
        let r_hand = glam::Vec3::from(FP_POSED_ARMS.iter().find(|(n, _)| *n == "r_hand").unwrap().1);
        // What `ActorScene::update` resolves for an attached weapon, and what `anchor` becomes.
        let world = mount * glam::Mat4::from_translation(r_hand);
        let (origin, fwd, up) = actor_mesh_frame(world);
        let hand = mount.transform_point3(r_hand);
        assert!(
            (origin - hand).length() < 1e-5,
            "the mesh origin must be the anchor's own translation (the hand), it is {origin:?}"
        );
        assert!(
            (fwd - world.x_axis.truncate()).length() < 1e-6 && (up - world.z_axis.truncate()).length() < 1e-6,
            "origin and basis must come off the SAME matrix"
        );
        // And the value it used to take instead -- the actor's `pos`, which for a rig mounted on
        // the view is the EYE -- is far enough away that this gate genuinely separates the two.
        assert!(
            (origin - cam.pos).length() > 0.15,
            "a held gun is measurably off the camera; it is only {:.3} wu away, so this test \
             could not tell the two apart",
            (origin - cam.pos).length()
        );
    }

    /// #fp-attach  ...and the CALL SITE takes it from there, not from the actor's position.
    ///
    /// A source gate, because the consequence only shows on a GPU: the helper above can be perfect
    /// and the builder can still pass something else. The pre-fix form is a hard failure.
    #[test]
    fn the_actor_mesh_builder_takes_its_origin_from_the_anchor() {
        let src = include_str!("reach_actor.rs");
        assert!(
            src.contains("let (origin, bfwd, bup) = actor_mesh_frame(a.anchor);"),
            "build_actor_meshes must take the mesh origin AND basis from one call to \
             `actor_mesh_frame` on the actor's anchor"
        );
        assert!(
            !src.contains("a.def.obj_tag, a.variant_sid, a.pos, bfwd, bup"),
            "the pre-fix form -- mesh ORIGIN from `a.pos` while the BASIS comes from `anchor` -- \
             baked a first-person weapon on the camera and is a hard failure"
        );
    }

    /// #fp-arms  The camera's subject resolves to a player SLOT even when the body it is riding
    /// has no binding of its own.
    ///
    /// Gate: the armour install asked only `slot_of_entity`, and the window's camera list is
    /// OWNER-keyed. A body whose own life was never bound to a slot answered `None`, nothing was
    /// installed, and the arms stayed in the model's default black under a third-person body
    /// wearing that player's colours.
    #[test]
    fn the_first_person_subject_resolves_to_a_slot_through_its_owner() {
        use crate::film_decode::PlayerBinding;
        let mut idx = FilmIndex { ticks: 600, ..Default::default() };
        // Two lives of ONE owner group. Only the first is bound to a slot.
        for (net, first, last) in [(0x11u32, 0u32, 199u32), (0x22, 200, 599)] {
            let mut tr = EntityTrack {
                net_id: net,
                entity_type: 11,
                first_tick: first,
                last_tick: last,
                owner_raw: Some(4),
                ..Default::default()
            };
            for t in first..=last {
                tr.positions.push((t, [1.0, 2.0, 3.0]));
            }
            idx.tracks.push(tr);
        }
        idx.bindings.push(PlayerBinding {
            slot: 9,
            net_id: 0x11,
            entity_type: 11,
            first_tick: 0,
            last_tick: 199,
            from_owner_rule: true,
            ..Default::default()
        });
        let mut pb = Playback::new(idx);
        pb.camera = FilmCamera::FirstPersonOwner(4);
        pb.tick = 400.0;
        let net = pb.hidden_entity().expect("the owner has a body at this tick");
        assert_eq!(net, 0x22, "the SECOND life is the one alive at tick 400");
        assert_eq!(
            pb.index.slot_of_entity(net),
            None,
            "this life has no binding of its own -- that is the case the fix is for"
        );
        assert_eq!(
            pb.subject_slot(),
            Some(9),
            "the owner group names the player even when this body does not"
        );
        // And the bound life still answers directly.
        pb.tick = 100.0;
        assert_eq!(pb.subject_slot(), Some(9));
    }

    /// #fp-attach  The first-person WEAPON is parented to the hand, so its root lands IN the hand.
    ///
    /// Gate: HMS composed the weapon's clip channel through the weapon model's OWN two-node
    /// skeleton, where its root has no parent, so the gun was drawn at the rig's origin — 0.21 wu
    /// from the hand holding it and all but on the eye. The sword then stood up across the middle
    /// of the screen.
    ///
    /// The check is rotation-free on purpose: the hilt's offset from the hand is the clip's own
    /// LOCAL translation turned by whatever the hand is doing, so its LENGTH is what the two
    /// measurements have to agree on, and they do to 0.001 wu.
    #[test]
    fn the_first_person_weapon_is_held_in_the_hand() {
        let v = |t: &[(&str, [f32; 3])], n: &str| -> glam::Vec3 {
            glam::Vec3::from(t.iter().find(|(k, _)| *k == n).unwrap_or_else(|| panic!("{n}")).1)
        };
        let r_hand = v(FP_POSED_ARMS, "r_hand");
        for (name, _) in FP_POSED_WEAPON_LOCAL {
            let local = v(FP_POSED_WEAPON_LOCAL, name).length();
            let placed = (v(FP_POSED_WEAPON_RIG, name) - r_hand).length();
            assert!(
                (placed - local).abs() < 0.006,
                "{name} is a child of r_hand, so it must sit exactly its own local offset \
                 ({local:.3} wu) from the hand; the attach puts it {placed:.3} wu away"
            );
        }
        // And the composition the bug produced -- the weapon's local channel treated as a
        // rig-space position -- has to be far enough off that this gate genuinely separates them.
        let adrift = (v(FP_POSED_WEAPON_LOCAL, "b_handle") - r_hand).length();
        assert!(
            adrift > 0.15,
            "the un-parented composition must be measurably wrong for this to bite; it is only \
             {adrift:.3} wu out"
        );
        // Un-parented, the hilt sat 0.035 wu from the rig origin -- which is the eye.
        assert!(
            v(FP_POSED_WEAPON_LOCAL, "b_handle").length() < 0.06,
            "the un-parented hilt sat all but on the eye; if that has changed, re-measure"
        );
    }

    /// #skin  A body with no recorded appearance must not make the actor meshes dirty FOREVER.
    ///
    /// This is the gate on "FPS is terrible in theater playback". `build_actor_meshes` writes the
    /// model's own default variant into `Actor::variant_sid` whenever it finds it unset, and the
    /// appearance pass used to compare that against a bare 0 and clear it again — so the two
    /// fought, every frame marked the actor scene dirty, and every frame rebuilt every actor
    /// mesh. The player variants live in a reserved range, so the question "is this actor wearing
    /// somebody's RECORDED armour" has an exact answer and a model default must be left alone.
    ///
    /// Pure, so it needs no cache and no GPU: it drives the same decision the loop makes.
    #[test]
    fn a_model_default_variant_is_not_mistaken_for_recorded_armour() {
        // The decision the appearance pass makes, in the same form.
        fn clears(actor_variant_sid: u32, wanted: u32) -> bool {
            let wearing_recorded = actor_variant_sid & 0xFFFF_0000 == 0xF1A1_0000;
            wanted == 0 && wearing_recorded
        }
        fn sets(actor_variant_sid: u32, wanted: u32) -> bool {
            wanted != 0 && actor_variant_sid != wanted
        }
        // A model DEFAULT variant (any ordinary string id) with nothing recorded: leave it be.
        // Before the fix this returned true and the scene was rebuilt on every single frame.
        for default_sid in [0x0000_0001u32, 0x1234_5678, 0xDEAD_BEEF, 0x0000_FFFF] {
            assert!(
                !clears(default_sid, 0),
                "a model default variant {default_sid:#x} was treated as recorded armour, which \
                 rebuilds every actor mesh every frame"
            );
            assert!(!sets(default_sid, 0), "nothing to set when no appearance is recorded");
        }
        // Somebody's recorded armour, and this body has none: that MUST be cleared, or one
        // player's armour follows a pooled actor onto another player's body.
        for slot in 0u8..16 {
            let sid = crate::scene::player_variant_sid(slot);
            assert_eq!(sid & 0xFFFF_0000, 0xF1A1_0000, "the reserved range moved: {sid:#x}");
            assert!(clears(sid, 0), "slot {slot}'s recorded armour was not cleared");
        }
        // A real appearance is installed, once, and then left alone.
        let a = crate::scene::player_variant_sid(3);
        let b = crate::scene::player_variant_sid(9);
        assert!(sets(0, a), "a fresh actor takes the recorded variant");
        assert!(sets(b, a), "a recycled actor takes the new body's variant");
        assert!(!sets(a, a), "an actor already wearing it is not touched again");
        assert!(!clears(a, a), "an actor already wearing it is not cleared");
    }
}

// ------------------------------------------------------------------------------------------
// #ragdoll  Theater ragdolls
// ------------------------------------------------------------------------------------------

/// #ragdoll  How often the simulation keeps a keyframe of a body's trajectory, in film ticks.
///
/// The simulation is a PURE FUNCTION of (seed, tick) -- the cache only decides how much of it has
/// to be re-run. Scrubbing forward costs one step per tick; scrubbing BACKWARDS restarts from the
/// nearest keyframe at or before the target and re-runs at most this many steps, which is what
/// makes a ragdoll undo itself exactly when the user drags the timeline back. Dropping the whole
/// cache changes nothing that is drawn.
pub const RAGDOLL_KEY_TICKS: u32 = 15;

/// #ragdoll-budget  The most simulation ticks one `simulate_ragdolls` call may run, over ALL
/// corpses together.
///
/// Catching a ragdoll up is ~1 ms per tick per body, and a seek can land a long way past a death:
/// measured on the test film, twelve rewind-seeks cost **336 ms each** with the solver on against
/// **124 ms** with it off, so the catch-up alone was ~212 ms of every seek. Dragging the timeline
/// fires several seeks a second, and the window stopped responding.
///
/// The simulation stays a pure function of (seed, tick) -- this only bounds how much of it may be
/// re-run in ONE call. A body that needs more keeps its last simulated pose and advances again on
/// the next frame, so a long seek settles over a few frames instead of blocking the UI for a third
/// of a second. 120 ticks is two seconds of fall at 60 Hz and about 8 ms of work shared across
/// every corpse on screen.
pub const RAGDOLL_CATCHUP_BUDGET: u32 = 120;

/// #ragdoll  The death blow the solver seeds a corpse with, wu/s, along the direction from the
/// killer to the victim.
///
/// The recording does NOT state the damage impulse -- it states the KILL (`incident`, the engine's
/// own medal/kill feed) and both parties, so the direction is a decoded fact and only the
/// magnitude is chosen. One value per kind of kill, ordered the way the engine's own damage is:
/// a melee finisher throws a body harder than a bullet does, and a splatter hardest.
pub fn death_blow(incident: &str) -> f32 {
    match incident {
        "assassination_kill" | "finishing_move_kill" => 1.5,
        "melee_kill" | "yoink" => 4.0,
        "splatter_kill" => 9.0,
        "headshot_kill" => 2.5,
        _ => 2.0,
    }
}

/// #ragdoll  One dead body's ragdoll simulation.
///
/// Everything in here is derived from the seed (the death tick, the recorded position and
/// velocity, the recorded killer, and the entity id) plus the tick asked for, so two scrubs to
/// the same moment give the same pose to the bit.
#[derive(Debug)]
struct CorpseSim {
    rd: std::sync::Arc<crate::ragdoll::Ragdoll>,
    /// The actor placement the pose is expressed relative to (the death position and facing) and
    /// its inverse.
    root: glam::Mat4,
    root_inv: glam::Mat4,
    /// The death clip's per-node LOCAL matrices: what the nodes no rigid body drives keep.
    locals: Vec<glam::Mat4>,
    parents: Vec<i16>,
    order: Vec<usize>,
    /// Keyframes `(age in film ticks, state)`, every [`RAGDOLL_KEY_TICKS`], age 0 first.
    keys: Vec<(u32, crate::ragdoll::RagdollState)>,
    /// The age `state` is at.
    age: u32,
    state: crate::ragdoll::RagdollState,
    /// Ticks stepped since the sim was created, for the diagnostics.
    pub stepped: u64,
}

impl CorpseSim {
    /// Rewind to at or before `age`: the nearest keyframe when the timeline went BACKWARDS,
    /// nothing at all when it did not. Separate from the stepping so the caller can gather the
    /// collision geometry around where the body actually is before each chunk of steps -- a
    /// rewind that stepped forward on its own would have to do it with the wrong geometry.
    fn rewind_to(&mut self, age: u32) {
        if age >= self.age { return; }
        let (k, st) = self
            .keys
            .iter()
            .rev()
            .find(|(t, _)| *t <= age)
            .map(|(t, s)| (*t, s.clone()))
            .unwrap_or_else(|| (0, self.keys[0].1.clone()));
        self.age = k;
        self.state = st;
        self.keys.retain(|(t, _)| *t <= k);
    }

    /// Rewind then step forward: the whole "what is the pose at this age" operation, exactly as
    /// the driver performs it. The driver splits the two so it can re-gather collision geometry
    /// between chunks; a test (and any caller with one fixed world) can use this.
    fn advance_to(&mut self, age: u32, world: &dyn crate::physics_settle::StaticWorld, p: &crate::ragdoll::SimParams) {
        self.rewind_to(age);
        while self.age < age {
            let before = self.age;
            self.step_to(age.min(self.age + RAGDOLL_KEY_TICKS), world, p);
            if self.age <= before { break; }
        }
    }

    /// Step FORWARD to `target` (never backwards), keeping a keyframe every
    /// [`RAGDOLL_KEY_TICKS`]. Once the body is asleep every later tick is the same state, so the
    /// cursor jumps to `target` and no more work is done.
    fn step_to(&mut self, target: u32, world: &dyn crate::physics_settle::StaticWorld, p: &crate::ragdoll::SimParams) {
        while self.age < target {
            if self.state.asleep {
                self.age = target;
                break;
            }
            crate::ragdoll::step(&self.rd, &mut self.state, world, p);
            self.age += 1;
            self.stepped += 1;
            if self.age % RAGDOLL_KEY_TICKS == 0 && self.keys.last().map(|(t, _)| *t) != Some(self.age) {
                self.keys.push((self.age, self.state.clone()));
            }
        }
    }
}

/// #ragdoll  A `StaticWorld` over the open scene: the level's collision triangles plus the
/// physics hulls of the Forge objects near the body. The level grid is persistent and immutable
/// after load, so this is cheap enough to query every tick.
struct SceneRagdollWorld<'a> {
    scene: &'a dyn crate::objscene::ObjectScene,
    extra: Vec<[glam::Vec3; 3]>,
}

impl crate::physics_settle::StaticWorld for SceneRagdollWorld<'_> {
    fn gather(&self, mn: glam::Vec3, mx: glam::Vec3, out: &mut Vec<[glam::Vec3; 3]>) {
        self.scene.static_tris_in_aabb(mn, mx, out);
        crate::physics_settle::StaticWorld::gather(self.extra.as_slice(), mn, mx, out);
    }
}

impl ActorDriver {
    /// #ragdoll  Pose every dead body from the ARTICULATED SOLVER instead of from a death clip.
    ///
    /// Reach's death variety is not a clip library -- the Spartan graph holds exactly SEVEN death
    /// clips for 1,393 animations -- it is ragdoll physics, so "the dead bipeds need proper
    /// variety" is a request for a simulation. This runs one: the biped's own `phmo` rigid bodies
    /// joined by its own authored constraints ([`crate::ragdoll`]), colliding with the level and
    /// with the Forge objects, seeded from the recording.
    ///
    /// **Deterministic under scrubbing**, which is the hard requirement: the state at tick T is a
    /// pure function of the seed and T, and the seed is entirely recorded -- the death tick, the
    /// body's recorded position and velocity, the killer the `incident` event names, and the
    /// entity id. Scrubbing forward steps; scrubbing BACKWARDS rewinds to the nearest keyframe and
    /// re-simulates, so the body stands back up through exactly the fall it came down by. The
    /// cache is a speedup and nothing else: throw it away and the same tick redraws the same pose.
    ///
    /// Call AFTER [`ActorDriver::sync`] -- it reads the death clip `sync` chose as its seed pose
    /// and then overrides the pose that clip would have drawn.
    pub fn simulate_ragdolls(
        &mut self,
        pb: &Playback,
        actors: &mut crate::reach_actor::ActorScene,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        scene: &dyn crate::objscene::ObjectScene,
        objects: &[hms_ipc::ObjectInfo],
    ) {
        use crate::ragdoll;
        self.ragdolls_shown = 0;
        self.ragdoll_steps = 0;
        self.ragdoll_bodies.clear();
        self.ragdoll_catching_up = 0;
        // #ragdoll-budget  Shared across every corpse this frame, so N bodies cannot multiply the
        // cost of a seek.
        let mut budget = RAGDOLL_CATCHUP_BUDGET;
        // net id -> actor slot, as `sync` assigned them this frame.
        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        for (&slot, &net) in &self.pool.assigned {
            slot_of.insert(net, slot);
        }
        if !self.ragdoll {
            for (&_net, &slot) in &slot_of {
                if let Some(a) = actors.get_mut(slot) {
                    a.pose_override = None;
                }
            }
            self.corpse_sims.clear();
            return;
        }
        let corpses = pb.corpses(self.corpse_cap);
        let alive: std::collections::HashSet<u32> = corpses.iter().map(|p| p.net_id).collect();
        self.corpse_sims.retain(|k, _| alive.contains(k));
        // Any body that is NOT a corpse this frame must have its override cleared, or a respawned
        // player would keep a dead pose.
        for (&net, &slot) in &slot_of {
            if !alive.contains(&net) {
                if let Some(a) = actors.get_mut(slot) {
                    a.pose_override = None;
                }
            }
        }
        let deaths = pb.deaths();
        let sim_params = ragdoll::SimParams {
            // The film's own tick, so the fall rate is the one measured off the recordings.
            dt: 1.0 / 60.0,
            gravity: CORPSE_GRAVITY,
            ..Default::default()
        };
        for p in &corpses {
            let Some(age) = p.dead_ticks else { continue };
            let Some(&slot) = slot_of.get(&p.net_id) else { continue };
            // ---- the definition, once per biped tag --------------------------------------
            let obj_tag = match actors.get(slot) {
                Some(a) => a.def.obj_tag,
                None => continue,
            };
            if !self.ragdoll_defs.contains_key(&obj_tag) {
                let bind = actors.get(slot).map(|a| a.def.model.skel.bind_model()).unwrap_or_default();
                let name = self.biped_of.get(&p.net_id).cloned().unwrap_or_default();
                let rd = ragdoll::ragdoll_for(native, cache, obj_tag, &bind, name).map(std::sync::Arc::new);
                if rd.is_none() {
                    self.ragdoll_missing = Some(format!(
                        "0x{obj_tag:08X} has no articulated physics model (its phmo has fewer than two rigid bodies with a shape, or no constraints between them)"
                    ));
                }
                self.ragdoll_defs.insert(obj_tag, rd);
            }
            let Some(Some(rd)) = self.ragdoll_defs.get(&obj_tag).cloned() else { continue };

            // ---- the seed, once per body -------------------------------------------------
            if !self.corpse_sims.contains_key(&p.net_id) {
                // Pin the placement to the DEATH pose: the whole fall lives in the pose, so the
                // body's world motion is the solver's and nothing else moves it.
                // WHICH death clip seeds the pose is recomputed from the recording, never read
                // off the actor's `anim_name`. That field is whatever `sync` last installed, so
                // it depends on where the timeline has BEEN -- and a seed that depends on history
                // is exactly what breaks scrubbing: measured on the test film, a body reached at
                // tick 251 by playing forward and the same body reached by dragging back to 249
                // and forward again settled 0.29 wu apart, because the two routes had different
                // clips on the actor when the simulation was seeded.
                let elite = self.biped_of.get(&p.net_id).is_some_and(|b| crate::film_anim::is_elite(b));
                let tr = self.track_of.get(&p.net_id).and_then(|&i| pb.index.tracks.get(i));
                let airborne = tr
                    .and_then(|tr| pb.death_at(tr).map(|d| crate::film_anim::died_airborne(tr, d)))
                    .unwrap_or(false);
                let variant = (p.net_id.rotate_right(8) % 12) as u8;
                // the weapon class AT THE DEATH TICK, not at the tick being drawn: the seed must
                // not change as the timeline moves
                let death_tick = deaths
                    .get(&p.net_id)
                    .map(|&(t, _, _)| t)
                    .or_else(|| tr.map(|t| t.last_tick))
                    .unwrap_or(0);
                let cls = self.class_at(tr, death_tick).unwrap_or(crate::film_anim::DEFAULT_WEAPON_CLASS).to_string();
                let cands = crate::film_anim::death_candidates(variant, &cls, elite, airborne);
                let (clip, root, parents, order, seed) = {
                    let Some(a) = actors.get_mut(slot) else { continue };
                    a.pos = glam::Vec3::from(p.pos);
                    a.yaw = p.yaw;
                    let root = a.world();
                    let parents: Vec<i16> = a.def.model.skel.nodes.iter().map(|n| n.parent).collect();
                    let order = a.def.model.skel.topo_order();
                    // the first candidate this graph actually has, in the fixed candidate order
                    let mut clip = String::new();
                    let mut seed = None;
                    for c in &cands {
                        if let Some(s) = a.clip_pose_model(native, cache, c, 0.0) {
                            clip = c.clone();
                            seed = Some(s);
                            break;
                        }
                    }
                    (clip, root, parents, order, seed)
                };
                let Some((model, locals)) = seed else {
                    let _ = clip;
                    self.ragdoll_missing =
                        Some(format!("none of this biped's death clips could be sampled for a seed pose ({cands:?})"));
                    continue;
                };
                let node_world: Vec<glam::Mat4> = model.iter().map(|m| root * *m).collect();
                let mut state = rd.state_from_pose(&node_world);
                // ---- the recorded seed velocities ----------------------------------------
                let dtick = Some(death_tick);
                let v0 = self
                    .track_of
                    .get(&p.net_id)
                    .and_then(|&i| pb.index.tracks.get(i))
                    .and_then(|tr| {
                        crate::film_anim::velocity_at(tr, dtick.unwrap_or(tr.last_tick).saturating_sub(6))
                    })
                    .map(glam::Vec3::from)
                    .filter(|v| v.is_finite() && v.length() < 40.0)
                    .unwrap_or(glam::Vec3::ZERO);
                // The death BLOW: direction from the killer the `incident` event names, magnitude
                // by the kind of kill. A kill with no killer on the wire gets none.
                let blow = match deaths.get(&p.net_id) {
                    Some((_, incident, Some(killer))) => {
                        let kp = self
                            .track_of
                            .get(killer)
                            .and_then(|&i| pb.index.tracks.get(i))
                            .and_then(|tr| tr.sample(dtick.unwrap_or(tr.last_tick)))
                            .map(glam::Vec3::from);
                        kp.map(|kp| {
                            let mut d = glam::Vec3::from(p.pos) - kp;
                            d.z = 0.0;
                            let d = d.normalize_or_zero();
                            d * crate::film_play::death_blow(incident)
                        })
                        .unwrap_or(glam::Vec3::ZERO)
                    }
                    _ => glam::Vec3::ZERO,
                };
                // The spin: NOT recorded, so it is derived from the entity id the same way the
                // death clip variant always has been -- deterministic, and the only part of the
                // seed that is invented. The caption says so.
                let h = p.net_id.wrapping_mul(2654435761);
                let spin = glam::Vec3::new(
                    ((h & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
                    ((h >> 8 & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
                    ((h >> 16 & 0xFF) as f32 / 255.0 - 0.5) * 2.0,
                ) * 3.0;
                for s in state.links.iter_mut() {
                    s.v = v0 + blow;
                    s.w = spin;
                }
                let keys = vec![(0u32, state.clone())];
                self.corpse_sims.insert(
                    p.net_id,
                    CorpseSim { rd: rd.clone(), root, root_inv: root.inverse(), locals, parents, order, keys, age: 0, state, stepped: 0 },
                );
            }

            // ---- step and pose ----------------------------------------------------------
            //
            // In CHUNKS, re-gathering the Forge objects' physics hulls around the body each time.
            // The level's own triangles are queried per link per substep with the correct swept
            // box (`SceneRagdollWorld` delegates straight to the scene), but the OBJECT hulls are
            // pre-filtered to a box, and a single box taken at the start of a long catch-up -- a
            // seek two seconds past a death -- would not contain where the body ends up, so it
            // would fall through a Forge block it should have landed on. A chunk of
            // `RAGDOLL_KEY_TICKS` is at most a few wu of travel, which `HULL_PAD` covers.
            const HULL_PAD: f32 = 8.0;
            if let Some(sim) = self.corpse_sims.get_mut(&p.net_id) {
                sim.rewind_to(age);
            }
            loop {
                let Some(sim) = self.corpse_sims.get(&p.net_id) else { break };
                if sim.age >= age { break; }
                // #ragdoll-budget  Stop when this frame's shared budget is spent; the body keeps
                // its last simulated pose and catches up on the next frame.
                if budget == 0 {
                    self.ragdoll_catching_up += 1;
                    break;
                }
                let target = age.min(sim.age + RAGDOLL_KEY_TICKS).min(sim.age + budget);
                // #film-perf  An ASLEEP body does no physics at all: `CorpseSim::step_to` tests
                // `state.asleep` first and jumps the cursor to `target` without touching the
                // world or the extra triangles. Gathering them was therefore pure waste -- and it
                // is not a small waste: `nearby_object_tris` walks the WHOLE placed-object table
                // and re-filters every triangle of every nearby hull, once per corpse, EVERY
                // frame, for the rest of the clip. A settled body never wakes (nothing in a
                // recording moves the scenery under it), so with fourteen corpses on the ground
                // that was fourteen full object-table scans a frame for a result nothing read.
                //
                // Exactly equivalent, not an approximation: the value skipped is the argument to
                // a call that provably does not read it.
                let extra_tris = if sim.state.asleep {
                    Vec::new()
                } else {
                    let (mn, mx) = sim.rd.aabb(&sim.state);
                    self.nearby_object_tris(
                        scene,
                        objects,
                        mn - glam::Vec3::splat(HULL_PAD),
                        mx + glam::Vec3::splat(HULL_PAD),
                    )
                };
                let Some(sim) = self.corpse_sims.get_mut(&p.net_id) else { break };
                let world = SceneRagdollWorld { scene, extra: extra_tris };
                let before = sim.age;
                sim.step_to(target, &world, &sim_params);
                budget = budget.saturating_sub(sim.age.saturating_sub(before));
                // `step_to` only ever moves forward, so it must make progress; stop rather than
                // spin if it ever does not.
                if sim.age <= before { break; }
            }
            let Some((model, steps, root)) = ({
                let Some(sim) = self.corpse_sims.get(&p.net_id) else { continue };
                let model = sim.rd.pose_model(&sim.state, sim.root_inv, &sim.parents, &sim.locals, &sim.order);
                Some((model, sim.stepped, sim.root))
            }) else { continue };
            if let Some(a) = actors.get_mut(slot) {
                // The placement never moves once the body is dead: the pose carries the fall.
                let (_, _, t) = root.to_scale_rotation_translation();
                a.pos = t;
                a.pose_override = Some(model);
                self.ragdolls_shown += 1;
                self.ragdoll_steps = self.ragdoll_steps.max(steps);
                // Where the body actually IS: the first link's world centre of mass, which for a
                // biped phmo is a real limb, not the (deliberately unmoving) placement.
                if let Some(sim) = self.corpse_sims.get(&p.net_id) {
                    if let Some(l) = sim.state.links.first() {
                        self.ragdoll_bodies.push((p.net_id, l.com.to_array(), age));
                    }
                }
            }
        }
        if self.ragdolls_shown > 0 {
            self.ragdoll_missing = None;
        }
    }

    /// #ragdoll  The physics hulls of the placed objects whose box is anywhere near `[mn, mx]`,
    /// as world triangles. The same hulls the editor's settle collides with
    /// (`ObjectScene::physics_world_tris`), restricted by the cheap position reject the settle
    /// uses, so a ragdoll lands on a Forge block rather than through it.
    fn nearby_object_tris(
        &mut self,
        scene: &dyn crate::objscene::ObjectScene,
        objects: &[hms_ipc::ObjectInfo],
        mn: glam::Vec3,
        mx: glam::Vec3,
    ) -> Vec<[glam::Vec3; 3]> {
        const NEAR: f32 = 12.0;
        let mut out = Vec::new();
        let pad = glam::Vec3::splat(0.5);
        for o in objects {
            let p = glam::Vec3::from(o.pos);
            if p.x < mn.x - NEAR || p.x > mx.x + NEAR || p.y < mn.y - NEAR || p.y > mx.y + NEAR || p.z < mn.z - NEAR || p.z > mx.z + NEAR {
                continue;
            }
            let tris = match self.obj_tris.get(&o.datum) {
                Some(t) => t.clone(),
                None => {
                    let flat: Vec<glam::Vec3> = scene.physics_world_tris(o).into_iter().flatten().collect();
                    let a = std::sync::Arc::new(flat);
                    self.obj_tris.insert(o.datum, a.clone());
                    a
                }
            };
            for c in tris.chunks_exact(3) {
                let t = [c[0], c[1], c[2]];
                if crate::physics_settle::tri_overlaps(&t, mn - pad, mx + pad) {
                    out.push(t);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod ragdoll_tests {
    use super::*;
    use crate::physics_settle::{ConvexShape, RigidBodyDef, ShapeMaterial};
    use crate::ragdoll;
    use glam::{Mat3, Mat4, Vec3};

    fn plane(z: f32, half: f32) -> Vec<[Vec3; 3]> {
        let a = Vec3::new(-half, -half, z);
        let b = Vec3::new(half, -half, z);
        let c = Vec3::new(half, half, z);
        let d = Vec3::new(-half, half, z);
        vec![[a, b, c], [a, c, d]]
    }

    /// A two-link biped stand-in: a torso pill and a leg pill, joined by a ragdoll constraint
    /// whose pivots agree at the bind pose (which is what a real phmo's do).
    fn two_link() -> (ragdoll::Ragdoll, Vec<Mat4>, Vec<i16>, Vec<usize>) {
        let mat = ShapeMaterial { friction: 0.6, restitution: 0.1, mass: 1.0 };
        let mk = |node: i16, len: f32| RigidBodyDef {
            shapes: vec![ConvexShape::capsule(Vec3::ZERO, Vec3::new(0.0, 0.0, -len), 0.09, mat).unwrap()],
            mass: 1.0, com: None, inertia: None, inertia_scale: Vec3::ONE,
            lin_damping: 0.0, ang_damping: 0.1, motion_type: 2, flags: 0, node,
        };
        // node 0 at z 1.0, node 1 at z 0.6: the joint is 0.4 below node 0 and at node 1's origin
        let bind = vec![
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.6)),
        ];
        let c = crate::phmo_shapes::ConstraintDef {
            name: 1,
            node_a: 0,
            node_b: 1,
            scale_a: 1.0,
            frame_a: Mat3::IDENTITY,
            pivot_a: Vec3::new(0.0, 0.0, -0.4),
            scale_b: 1.0,
            frame_b: Mat3::IDENTITY,
            pivot_b: Vec3::ZERO,
            edge: 0,
            kind: crate::phmo_shapes::ConstraintKind::Ragdoll {
                twist: (-0.4, 0.4),
                cone: (-0.8, 0.8),
                plane: (-0.5, 0.5),
                friction_torque: 0.2,
            },
        };
        assert!(c.pivot_residual(&bind) < 1e-6, "the stand-in's own pivots agree at bind");
        let rd = ragdoll::build(&[mk(0, 0.4), mk(1, 0.4)], &[c], &bind, "stand-in".into()).unwrap();
        let parents = vec![-1i16, 0];
        let order = vec![0usize, 1];
        (rd, bind, parents, order)
    }

    fn sim_of(rd: &ragdoll::Ragdoll, bind: &[Mat4], parents: &[i16], order: &[usize], seed_v: Vec3) -> CorpseSim {
        let root = Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0));
        let node_world: Vec<Mat4> = bind.iter().map(|m| root * *m).collect();
        let mut state = rd.state_from_pose(&node_world);
        for s in state.links.iter_mut() {
            s.v = seed_v;
            s.w = Vec3::new(0.5, -0.3, 0.9);
        }
        CorpseSim {
            rd: std::sync::Arc::new(rd.clone()),
            root,
            root_inv: root.inverse(),
            locals: bind.to_vec(),
            parents: parents.to_vec(),
            order: order.to_vec(),
            keys: vec![(0, state.clone())],
            age: 0,
            state,
            stepped: 0,
        }
    }

    fn params() -> ragdoll::SimParams {
        ragdoll::SimParams { dt: 1.0 / 60.0, gravity: CORPSE_GRAVITY, ..Default::default() }
    }

    /// #ragdoll THE SCRUBBING GATE. The pose at a tick must be the same however the timeline got
    /// there -- played straight through, jumped to, or reached by dragging backwards and forwards
    /// again. Anything less and a ragdoll would differ every time the user scrubs past it.
    #[test]
    fn a_ragdoll_is_the_same_at_a_tick_however_the_timeline_reached_it() {
        let (rd, bind, parents, order) = two_link();
        let world = plane(0.0, 20.0);
        let p = params();
        // (a) straight through to tick 150
        let mut a = sim_of(&rd, &bind, &parents, &order, Vec3::new(2.0, 0.0, 0.0));
        for t in 1..=150u32 { a.advance_to(t, &world, &p); }
        // (b) jumped straight to 150
        let mut b = sim_of(&rd, &bind, &parents, &order, Vec3::new(2.0, 0.0, 0.0));
        b.advance_to(150, &world, &p);
        // (c) forward to 150, scrubbed back to 40, forward to 150 again
        let mut c = sim_of(&rd, &bind, &parents, &order, Vec3::new(2.0, 0.0, 0.0));
        c.advance_to(150, &world, &p);
        c.advance_to(40, &world, &p);
        c.advance_to(150, &world, &p);
        assert_eq!(a.state, b.state, "stepped and jumped agree");
        assert_eq!(a.state, c.state, "and a scrub back and forward returns to the same pose");
        // and scrubbing back really does UNDO it
        let mut d = sim_of(&rd, &bind, &parents, &order, Vec3::new(2.0, 0.0, 0.0));
        d.advance_to(150, &world, &p);
        d.advance_to(0, &world, &p);
        let fresh = sim_of(&rd, &bind, &parents, &order, Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(d.state, fresh.state, "scrubbed back to the death tick, the body is un-dead");
        // a DIFFERENT seed gives a different ragdoll: that is the variety
        let mut e = sim_of(&rd, &bind, &parents, &order, Vec3::new(-2.0, 1.0, 0.0));
        e.advance_to(150, &world, &p);
        assert_ne!(a.state.links[1].com, e.state.links[1].com, "a different death gives a different body");
    }

    /// #ragdoll A corpse must come to rest ON the floor, not through it and not in orbit.
    #[test]
    fn a_ragdoll_lands_on_the_floor_and_sleeps() {
        let (rd, bind, parents, order) = two_link();
        let world = plane(0.0, 20.0);
        let p = params();
        let mut s = sim_of(&rd, &bind, &parents, &order, Vec3::new(3.0, 1.0, 0.0));
        for t in 1..=crate::film_anim::CORPSE_TICKS.min(600) {
            s.advance_to(t, &world, &p);
            if s.state.asleep { break; }
        }
        assert!(s.state.asleep, "came to rest after {} ticks", s.age);
        for (i, l) in s.state.links.iter().enumerate() {
            assert!(l.com.is_finite(), "link {i} finite");
            assert!(l.com.z > -0.2, "link {i} did not sink through the floor: {:?}", l.com);
            assert!(l.com.z < 1.0, "link {i} is lying down, not standing: {:?}", l.com);
            assert!(glam::Vec2::new(l.com.x, l.com.y).length() < 25.0, "link {i} did not fly off the map: {:?}", l.com);
        }
        // the joint still holds
        let (pa, pb) = {
            let j = &s.rd.joints[0];
            (
                s.rd.link_world(j.a, &s.state).transform_point3(j.pivot_a),
                s.rd.link_world(j.b, &s.state).transform_point3(j.pivot_b),
            )
        };
        assert!((pa - pb).length() < 0.03, "the limb is still attached: {:.4} wu apart", (pa - pb).length());
    }

    /// #ragdoll The pose handed to the skeleton is MODEL space relative to the actor placement,
    /// and the nodes no rigid body drives follow their driven parent rather than collapsing.
    #[test]
    fn the_pose_read_back_is_model_space_and_undriven_nodes_follow_their_parent() {
        let (rd, bind, parents, order) = two_link();
        // add a third node (a "finger") with no rigid body, a child of node 1
        let parents = vec![parents[0], parents[1], 1i16];
        let order = vec![0usize, 1, 2];
        let mut locals = bind.clone();
        locals.push(Mat4::from_translation(Vec3::new(0.0, 0.0, -0.3)));
        let world = plane(0.0, 20.0);
        let p = params();
        let mut s = sim_of(&rd, &bind, &parents, &order, Vec3::ZERO);
        s.locals = locals;
        s.parents = parents.clone();
        s.order = order.clone();
        // at tick 0 nothing has moved, so the pose must be the bind pose in model space
        let m0 = s.rd.pose_model(&s.state, s.root_inv, &s.parents, &s.locals, &s.order);
        assert_eq!(m0.len(), 3);
        for (i, b) in bind.iter().enumerate() {
            let d = (m0[i].w_axis - b.w_axis).length();
            assert!(d < 1e-4, "node {i} starts at its bind position in model space: {d}");
        }
        // the undriven finger is its parent times its own local
        let want = m0[1] * s.locals[2];
        assert!((m0[2].w_axis - want.w_axis).length() < 1e-5, "the finger rides node 1");
        // and after the body has fallen the finger still rides it
        s.advance_to(90, &world, &p);
        let m1 = s.rd.pose_model(&s.state, s.root_inv, &s.parents, &s.locals, &s.order);
        let want = m1[1] * s.locals[2];
        assert!((m1[2].w_axis - want.w_axis).length() < 1e-5, "still riding after the fall");
        assert!((m1[0].w_axis - m0[0].w_axis).length() > 0.1, "and the body really did move");
    }
}

/// #rewind  "If I rewind, a lot of players seem to be invisible."
///
/// This gate splits that report in two. The DECODE half -- which bodies a tick has, and where --
/// is a pure function of the tick and the index, so arriving at a tick by rewinding must give a
/// byte-identical answer to arriving at it by playing forward. If this gate ever fails, the bug
/// is in playback. While it passes, playback is exonerated and anything still invisible after a
/// rewind is the stateful ACTOR POOL failing to re-show a body it parked, which is a different
/// piece of code and needs a render to see.
///
/// `#[ignore]`d: it reads a film off the user's drive.
/// `HMS_FILM=<film> cargo test -p hms-app rewinding -- --ignored --nocapture`
#[cfg(test)]
mod rewind_tests {
    #[test]
    #[ignore]
    fn rewinding_to_a_tick_gives_the_same_bodies_as_playing_to_it() {
        let Ok(path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM");
            return;
        };
        let f = crate::film::open(std::path::Path::new(&path)).expect("film opens");
        let idx = crate::film_decode::build_index_players(
            &f.bytes,
            &f.data,
            &f.header.map_variant,
            f.recorder_slot(),
            &f.header.players,
        );
        let total = idx.ticks;
        assert!(total > 600, "need a clip of some length");
        let mut pb = super::Playback::new(idx);
        const CAP: usize = 16;
        const CORPSE_CAP: usize = 8;

        // Ten probe ticks spread over the clip.
        let probes: Vec<u32> = (1..=10).map(|k| total * k / 11).collect();

        // Forward: walk the clip in order, recording each probe.
        let mut forward: Vec<Vec<super::Placement>> = Vec::new();
        for &t in &probes {
            pb.tick = t as f32;
            forward.push(pb.bodies(CAP, CORPSE_CAP));
        }
        // Rewound: jump to the END, then visit the SAME probes in reverse.
        pb.tick = (total - 1) as f32;
        let _ = pb.bodies(CAP, CORPSE_CAP);
        let mut rewound: Vec<Vec<super::Placement>> = vec![Vec::new(); probes.len()];
        for (i, &t) in probes.iter().enumerate().rev() {
            pb.tick = t as f32;
            rewound[i] = pb.bodies(CAP, CORPSE_CAP);
        }

        let mut worst = 0usize;
        for (i, &t) in probes.iter().enumerate() {
            let (a, b) = (&forward[i], &rewound[i]);
            let ids_a: Vec<u32> = a.iter().map(|p| p.net_id).collect();
            let ids_b: Vec<u32> = b.iter().map(|p| p.net_id).collect();
            if ids_a != ids_b {
                worst += 1;
                println!(
                    "  tick {t}: forward {} bodies, rewound {} -- MISMATCH\n     fwd {ids_a:?}\n     rew {ids_b:?}",
                    a.len(),
                    b.len()
                );
            }
            assert_eq!(
                a.len(),
                b.len(),
                "tick {t}: playing forward gives {} bodies, rewinding gives {} -- playback is not \
                 a pure function of the tick, which is the bug behind \"players are invisible \
                 after a rewind\"",
                a.len(),
                b.len()
            );
            assert_eq!(ids_a, ids_b, "tick {t}: the same bodies must be drawn either way");
            for (pa, pb_) in a.iter().zip(b.iter()) {
                assert_eq!(pa.pos, pb_.pos, "tick {t}: body {:#010x} moved", pa.net_id);
            }
        }
        println!(
            "  {} probe tick(s) over {total} ticks: identical body sets either way ({} mismatch)",
            probes.len(),
            worst
        );
        let max = forward.iter().map(|v| v.len()).max().unwrap_or(0);
        assert!(max > 0, "the probes have to find SOME bodies or this gate proves nothing");
        println!("  busiest probe drew {max} bodies");
    }
}

/// #film-action  THE ASSASSINATION JUMP, on real recordings.
///
/// **Why this gate is here and not next to the other playback tests.** The user's report was
/// "the assassination animations still need to be shown properly, they are like too quick or
/// something so you basically never see them". The playback rate was measured and found CORRECT
/// -- the shipped moves are authored at 75 frames and HMS plays them across the full 150-tick
/// window -- so the thing that was actually missing was any way to *find* one: three kills of
/// 2.5 s each, scattered through 25,000 ticks, with the camera usually pointing elsewhere.
///
/// A synthetic index cannot check that. The whole claim is about where the kills really are in a
/// real recording and whether the jump lands on them, so this opens real films through the same
/// entry point the app uses and asserts the behaviour a viewer would see.
#[cfg(test)]
mod assassination_jump_gates {
    use super::*;
    use crate::film;

    /// Open real films through the production path and keep the ones with at least two kills.
    fn clips_with_kills() -> Vec<Playback> {
        let mut out = Vec::new();
        for path in film::spec_corpus() {
            let Ok(f) = film::open(&path) else { continue };
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let pb = Playback::new(idx);
            if pb.assassinations().len() >= 2 {
                out.push(pb);
            }
            if out.len() >= 4 {
                break;
            }
        }
        out
    }

    /// Walking "next" visits every kill in order, never repeats one, and stops at the end.
    ///
    /// The no-repeat part is the one that needed care: the jump parks the playhead BEFORE the kill
    /// it is showing, so a "next" that compared the playhead against the kill tick would find the
    /// same kill again and the button would appear to do nothing. Comparing against the point each
    /// jump parks at is what makes the walk advance.
    #[test]
    fn walking_forward_visits_every_kill_once_and_stops_at_the_end() {
        let clips = clips_with_kills();
        if clips.is_empty() {
            eprintln!("film_play: no real film here has two assassinations -- jump not measured");
            return;
        }
        let mut checked = 0usize;
        for mut pb in clips {
            let mut expected: Vec<u32> = pb.assassinations().iter().map(|a| a.tick).collect();
            expected.sort_unstable();
            expected.dedup();
            pb.seek_tick(0.0);
            let mut seen: Vec<u32> = Vec::new();
            // Bounded by the count, so a jump that failed to advance ends the loop with a short
            // list and fails the comparison below rather than spinning.
            for _ in 0..expected.len() + 2 {
                let Some(a) = pb.jump_to_assassination(1) else { break };
                assert!(
                    !seen.contains(&a.tick),
                    "jumped to the kill at tick {} twice -- the walk does not advance",
                    a.tick
                );
                // The playhead must sit BEFORE the kill, or the move is already over on arrival.
                let park = pb.tick.round() as u32;
                assert!(
                    park < a.tick && a.tick - park <= Playback::ASSASSINATION_LEAD_TICKS,
                    "parked at {park} for a kill at {} -- not within the lead-in",
                    a.tick
                );
                // ...and inside the window the two bodies are locked together, so the move is
                // actually playing a moment later.
                assert!(
                    a.covers(a.tick),
                    "the kill at {} does not cover its own tick",
                    a.tick
                );
                seen.push(a.tick);
            }
            assert_eq!(
                seen, expected,
                "walking forward did not visit every kill in order"
            );
            // At the last kill there is nothing further, and it must NOT wrap: a silent jump back
            // to the start of a 47-minute clip reads as the button having done nothing.
            assert!(
                pb.jump_to_assassination(1).is_none(),
                "the jump wrapped around instead of stopping at the last kill"
            );
            checked += 1;
        }
        eprintln!("film_play: the assassination walk checked on {checked} real clip(s)");
    }

    /// The jump moves the CAMERA, not just the playhead -- which is the whole point, since a seek
    /// that leaves the camera where it was still shows the viewer nothing.
    ///
    /// Only asserted where the stream names a victim with an owner group; a kill the stream gives
    /// no victim for still gets the seek, deliberately, and that case is counted rather than
    /// skipped silently.
    #[test]
    fn the_jump_puts_the_camera_on_the_victim() {
        let clips = clips_with_kills();
        if clips.is_empty() {
            eprintln!("film_play: no real film here has two assassinations -- camera not measured");
            return;
        }
        let (mut framed, mut no_victim) = (0usize, 0usize);
        for mut pb in clips {
            pb.seek_tick(0.0);
            // BOUNDED on purpose. Under a broken search the jump returns the same kill for ever
            // and an unbounded `while let` HANGS -- which is strictly worse than failing, because
            // a hung test reports nothing. Proven: with the park comparison sabotaged, this loop
            // span until it was killed while the gate above failed in milliseconds.
            let budget = pb.assassinations().len() + 2;
            let mut seen_ticks: Vec<u32> = Vec::new();
            for _ in 0..budget {
                let Some(a) = pb.jump_to_assassination(1) else { break };
                assert!(
                    !seen_ticks.contains(&a.tick),
                    "the jump returned the kill at tick {} twice -- it is not advancing",
                    a.tick
                );
                seen_ticks.push(a.tick);
                let owner = a
                    .victim
                    .and_then(|v| pb.index.tracks.iter().find(|t| t.net_id == v))
                    .and_then(|t| t.owner_raw);
                match owner {
                    Some(o) => {
                        assert_eq!(
                            pb.camera,
                            FilmCamera::FollowOwner(o),
                            "the camera was not put on the victim's owner group for the kill at {}",
                            a.tick
                        );
                        framed += 1;
                    }
                    None => no_victim += 1,
                }
            }
        }
        eprintln!(
            "film_play: {framed} kill(s) framed on the victim, {no_victim} with no owned victim \
             in the stream"
        );
        // The gate must have a case, or it proves nothing about the camera.
        assert!(
            framed > 0,
            "every kill on every clip here lacks an owned victim, so this gate proved nothing \
             about the camera"
        );
    }
}
