//! #reach-hud  Reach's first-person HUD, drawn from the game's own `chud_definition` tags.
//!
//! `reach_chud.rs` already decodes the two pieces this file needs a sibling for: the `chgd`
//! globals (the palette and the VIRTUAL SCREEN everything is laid out in) and one weapon's
//! reticule. This file generalises that to **any** `chud_definition`, so the authored ammo block,
//! the grenade counts, the shield and health meters and a vehicle's own HUD can be drawn from the
//! tags rather than re-invented, and it carries the honest bookkeeping of which of those fields a
//! saved film can actually fill.
//!
//! WHAT A `chdt` IS, at the byte level (`reverseme tagdefs reach_tag_test.exe chdt`):
//!
//! ```text
//! chud_definition
//!   +0x000 block  widget collections -> chud_widget_collection_block [max 32]  (220-byte elements)
//!            +0x000 struct chud_widget_base_struct (124 bytes)
//!                     +0x000 string id  artist name          (the collection's NAME)
//!                     +0x007 char       sort layer
//!                     +0x00C struct     state data           (+0x01C block, 56-byte element)
//!                              +0x000 block Active State  (12-byte AND terms of 8-byte ORs)
//!                              +0x00C block Flash State
//!                              +0x018 block Hidden State
//!                     +0x028 struct     placement data       (+0x038 block, 28-byte elements [max 5])
//!                              +0x000 byte flags  window state   (chud_curvature_res_flags)
//!                              +0x001 char enum   anchor type    (chud_anchor_type_enum)
//!                              +0x004 point2d     widget origin
//!                              +0x00C point2d     origin offset
//!                              +0x014 point2d     widget scale
//!                     +0x060 struct     render data          (+0x070 block, 120-byte element)
//!                              +0x004 argb custom color A .. +0x034 custom color D
//!                              +0x054 short enum color output A .. +0x05E output F
//!            +0x07C struct datasource (+0x08C block, 32-byte element)
//!                     +0x004 short enum type   (chud_datasource_type_enum)
//!                     +0x006 short     render maximum
//!            +0x098 block  bitmap widgets -> chud_widget_bitmap_block [max 32]  (156-byte elements)
//!                     +0x000 base (124)   +0x07C long flags   +0x080 tag ref bitmap
//!                     +0x090 char sequence index
//!            +0x0A4 block  text widgets   -> chud_widget_text_block [max 60]    (132-byte elements)
//!                     +0x000 base (124)   +0x07C word flags (chud_widget_text_flags)
//!                     +0x07E short enum font type (global_font_id)   +0x080 string id input string
//! ```
//!
//! and the three blocks a widget hangs off its 124-byte base — state, placement, render — are at
//! the SAME offsets inside a collection, a bitmap widget and a text widget, because all three
//! embed that one base struct. [`WidgetBase`] reads it once and every caller reuses it.
//!
//! THE LAYOUT RULE. The chud never converts a size on the CPU: the shader maps the authored
//! virtual screen (1280 x 720 on every shipped skin) linearly onto the window's SAFE FRAME, so
//! one virtual unit is `safeFrameHeightPx / virtualHeight` points
//! ([`reach_chud::px_per_virtual`]) and every anchor is a corner or an edge midpoint of that same
//! safe frame ([`VirtualScreen`]). A widget's point is `anchor + origin_offset * vpx`, with
//! `+x` right and `+y` down — read off the shipped tags: `ui\chud\assault_rifle`'s `ammo_area` is
//! anchored `top right` at `(+12, +31)` and its children sit at `-37 .. -85` in x, i.e. leftward
//! and downward into the screen from the top-right corner, which is exactly where Reach draws the
//! ammo block.
//!
//! WHAT THIS DOES NOT CLAIM. `widget origin` is used here as the ALIGNMENT PIVOT of a text
//! widget's own measured box (0 = left/top, 1 = right/bottom), which is what the shipped tags
//! read like — the assault rifle's `reserve_count` authors (0.5, 0.5) and is a centred number,
//! its `shematic` authors (1.0, 0.5) and is a right-aligned glyph. Bitmap widgets are centred on
//! their point, the same convention [`reach_chud::paint_crosshair`] already ships for the
//! reticule, because a sprite's registration is baked into its sheet rectangle. Neither reading
//! was confirmed against the engine's own quad builder.

use eframe::egui;
use hms_native::{CacheHandle, NativeDll};

use crate::reach_chud::{px_per_virtual, ChudSkin, NavpointSprite, COLOR_OUTPUT_BASE, SKIN_COLORS};

// ---------------------------------------------------------------------------------------------
// #nametag-mode  show always / only when looked at / hidden
// ---------------------------------------------------------------------------------------------

/// How much of the floating-name display is on.
///
/// In game there are TWO name displays and they are not the same thing (see
/// `film_ui::paint_name_tags`): the **ally waypoint** (chevron + gamertag, teammates only) and
/// the **look-at name** (`haloreach.dll+0x18ACC8` / `+0x18AE04`, no chevron, coloured by
/// `are_enemies`, lifted 0.050 wu rather than 0.075). [`NameTagMode::LookedAt`] is the second of
/// those, gated the engine's own way — by the viewer's weapon's authored autoaim cone
/// ([`AimAssist`]) rather than by an invented angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NameTagMode {
    /// Every body HMS draws carries its name, all the time. HMS's own default: a film has no
    /// local player and watching a clip is the point.
    #[default]
    Always,
    /// Only the body the viewer is looking at, by the engine's own targeting rule.
    LookedAt,
    /// No names at all.
    Hidden,
}

impl NameTagMode {
    pub fn is_on(self) -> bool {
        !matches!(self, NameTagMode::Hidden)
    }
    /// The word the setting file and the script verb use.
    pub fn keyword(self) -> &'static str {
        match self {
            NameTagMode::Always => "always",
            NameTagMode::LookedAt => "lookedat",
            NameTagMode::Hidden => "hidden",
        }
    }
    /// The label the panel shows.
    pub fn label(self) -> &'static str {
        match self {
            NameTagMode::Always => "Always",
            NameTagMode::LookedAt => "Only when looked at",
            NameTagMode::Hidden => "Hidden",
        }
    }
    /// Every spelling the script verb and the setting file accept. `on`/`off` are kept because
    /// `film nametags on|off` is what the verb took when the setting was a bool.
    pub fn parse(s: &str) -> Option<NameTagMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "always" | "all" | "on" | "1" | "true" => Some(NameTagMode::Always),
            "lookedat" | "looked-at" | "look" | "lookat" | "looking" | "2" => Some(NameTagMode::LookedAt),
            "hidden" | "hide" | "none" | "off" | "0" | "false" => Some(NameTagMode::Hidden),
            _ => None,
        }
    }
    pub const ALL: [NameTagMode; 3] = [NameTagMode::Always, NameTagMode::LookedAt, NameTagMode::Hidden];
}

/// Remembered between sessions the same way the spectated-player card is.
const NAME_TAG_SETTING_FILE: &str = "name_tags.txt";
/// #crosshair  Whether the weapon's authored reticule is drawn in first person.
const CROSSHAIR_SETTING_FILE: &str = "crosshair.txt";
/// #reach-hud  Whether the authored first-person HUD is drawn.
const HUD_SETTING_FILE: &str = "first_person_hud.txt";
/// #respawn  Whether the respawn countdown widget is drawn over a dead body.
const RESPAWN_SETTING_FILE: &str = "respawn_widget.txt";

pub fn load_name_tag_mode() -> NameTagMode {
    crate::project::settings_dir().map(|d| load_name_tag_mode_from_dir(&d)).unwrap_or_default()
}

pub fn load_name_tag_mode_from_dir(dir: &std::path::Path) -> NameTagMode {
    std::fs::read_to_string(dir.join(NAME_TAG_SETTING_FILE))
        .ok()
        .and_then(|s| NameTagMode::parse(&s))
        .unwrap_or_default()
}

pub fn save_name_tag_mode(m: NameTagMode) {
    if let Some(dir) = crate::project::settings_dir() {
        save_name_tag_mode_to_dir(&dir, m);
    }
}

pub fn save_name_tag_mode_to_dir(dir: &std::path::Path, m: NameTagMode) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(NAME_TAG_SETTING_FILE), m.keyword());
}

/// A plain on/off setting file, defaulting to ON — the three HUD toggles all behave this way.
fn load_flag_from_dir(dir: &std::path::Path, file: &str) -> bool {
    match std::fs::read_to_string(dir.join(file)) {
        Ok(s) => !matches!(s.trim(), "0" | "off" | "false"),
        Err(_) => true,
    }
}

fn save_flag_to_dir(dir: &std::path::Path, file: &str, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(file), if on { "1" } else { "0" });
}

pub fn load_crosshair_setting() -> bool {
    crate::project::settings_dir().map(|d| load_flag_from_dir(&d, CROSSHAIR_SETTING_FILE)).unwrap_or(true)
}
pub fn save_crosshair_setting(on: bool) {
    if let Some(d) = crate::project::settings_dir() {
        save_flag_to_dir(&d, CROSSHAIR_SETTING_FILE, on);
    }
}
pub fn load_hud_setting() -> bool {
    crate::project::settings_dir().map(|d| load_flag_from_dir(&d, HUD_SETTING_FILE)).unwrap_or(true)
}
pub fn save_hud_setting(on: bool) {
    if let Some(d) = crate::project::settings_dir() {
        save_flag_to_dir(&d, HUD_SETTING_FILE, on);
    }
}
pub fn load_respawn_setting() -> bool {
    crate::project::settings_dir().map(|d| load_flag_from_dir(&d, RESPAWN_SETTING_FILE)).unwrap_or(true)
}
pub fn save_respawn_setting(on: bool) {
    if let Some(d) = crate::project::settings_dir() {
        save_flag_to_dir(&d, RESPAWN_SETTING_FILE, on);
    }
}

// ---------------------------------------------------------------------------------------------
// #lookat  the engine's own "am I looking at that player" rule
// ---------------------------------------------------------------------------------------------

/// `weap` +0x2D4 `weapon aim assist` (`aim_assist_struct`, 60 bytes).
///
/// The tool build's own field names carry their units:
///
/// ```text
/// +0x2D4 angle  autoaim angle:degrees#the maximum angle that autoaim works at full strength
/// +0x2D8 real   autoaim range:world units#the maximum distance that autoaim works at full strength
/// +0x2DC real   autoaim falloff range:world units
/// +0x2E0 real   autoaim near falloff range:world units
/// +0x2E4 angle  magnetism angle:degrees
/// +0x2E8 real   magnetism range:world units
/// +0x2EC real   magnetism falloff range:world units
/// +0x2F0 real   magnetism near falloff range:world units
/// +0x2F4 angle  deviation angle:degrees
/// ```
///
/// **The `:degrees` suffix is the TOOL's display unit, not the storage unit. These are RADIANS.**
/// Measured, not assumed: over all 51 shipped Reach caches, every one of the 4,349 non-zero
/// `angle` values in a `weap` lands on an exact multiple of 0.05 degrees when read as radians and
/// on none when read as degrees. The magnum authors `0.039270` (2.25 deg) autoaim and `0.078540`
/// (4.50 deg) magnetism; the sniper rifle `0.013963` (0.80 deg) and `0.027925` (1.60 deg); the
/// assault rifle `0.087266` (5.00 deg) and `0.174533` (10.00 deg).
///
/// **The discriminator is the field TYPE, not the name.** `angle` (tag-definition type id 8) is
/// radians on disk; a `real` whose name merely ends `:degrees` is literally degrees — the same
/// `weap` holds both, e.g. `+0x498 support third person camera range:degrees` is a `real bounds`
/// reading (-60.0, +75.0) on the machinegun turret. The Reach tag definitions carry 126 `angle`
/// fields with no `:degrees` suffix and 111 `real` fields that ARE degrees, so the suffix decides
/// nothing. [`tests::aim_assist_angles_are_radians_not_degrees`] pins this against the shipped
/// cache, because reading a stored angle in the wrong unit is exactly the mistake that nearly
/// wrecked the ragdoll work, where `phmo` joint limits turned out to be the other way round.
///
/// **This block is NOT what gates the look-at name.** See [`LookAtCone`]: the HUD's own
/// nearest-target search reads none of these fields. The block is decoded and reported because it
/// drives the aim-assist lane that fills `unit+0x2B8`, which is the look-at name's *primary*
/// source — a lane that has not been decompiled, so nothing here is built on it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AimAssist {
    /// `autoaim angle`, RADIANS as stored.
    pub autoaim_angle: f32,
    /// `autoaim range`, world units.
    pub autoaim_range: f32,
    pub autoaim_falloff_range: f32,
    pub autoaim_near_falloff: f32,
    /// `magnetism angle`, RADIANS as stored.
    pub magnetism_angle: f32,
    pub magnetism_range: f32,
    pub magnetism_falloff_range: f32,
    pub magnetism_near_falloff: f32,
    /// `deviation angle`, RADIANS as stored.
    pub deviation_angle: f32,
}

/// Is `deg` a multiple of the authoring grid, within single-precision slack?
///
/// The slack has to scale with the value: a 45-degree angle stored as an f32 radian and converted
/// back loses more absolute precision than a 0.8-degree one.
pub fn on_angle_grid(deg: f32) -> bool {
    let n = (deg / ANGLE_AUTHORING_GRID_DEG).round();
    let tol = (deg.abs() * 1e-5).max(1e-4);
    (deg - n * ANGLE_AUTHORING_GRID_DEG).abs() <= tol
}

/// Byte offset of `weapon aim assist` inside a `weap` tag — `weapon_struct_definition` +0xD0 on a
/// base of 0x204 (`object_struct_definition` 0x168 + `item_struct_definition` 0x9C), the same base
/// `handle node` at struct +0x1A4 confirms by landing on the +0x3A8 `reach_actor` already uses.
pub const WEAPON_AIM_ASSIST: usize = 0x2D4;

/// The AUTHORING GRID every Reach `angle` lands on, in degrees.
///
/// This is the structural test that settles radians-vs-degrees, and it is far sharper than "is
/// the number small". Over all 51 shipped Reach caches there are 4,349 non-zero `angle` values in
/// `weap` tags; read as radians, **4,349 of 4,349 land on an exact multiple of 0.05 degrees**,
/// and read as degrees, **none of them do**. The editor authors these in degrees on a 0.05 grid
/// and the compiler converts to radians, so the grid survives in the stored value and nowhere
/// else.
///
/// A size threshold would NOT do: the ghost's dual cannon authors a `deviation angle` of
/// `0.7853982`, which is a perfectly ordinary 45.0 degrees in radians and would trip any
/// "too big to be radians" rule.
pub const ANGLE_AUTHORING_GRID_DEG: f32 = 0.05;

impl AimAssist {
    /// Read one weapon's block. `None` when the tag is unreadable.
    pub fn read(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> Option<AimAssist> {
        let meta = native.tag_meta(cache, weap_tag, 0, (WEAPON_AIM_ASSIST + 0x3C) as u32)?;
        let f = |o: usize| rd_f32(&meta, WEAPON_AIM_ASSIST + o).unwrap_or(0.0);
        Some(AimAssist {
            autoaim_angle: f(0x00),
            autoaim_range: f(0x04),
            autoaim_falloff_range: f(0x08),
            autoaim_near_falloff: f(0x0C),
            magnetism_angle: f(0x10),
            magnetism_range: f(0x14),
            magnetism_falloff_range: f(0x18),
            magnetism_near_falloff: f(0x1C),
            deviation_angle: f(0x20),
        })
    }
    /// The three `angle` fields, in storage order.
    pub fn angles(&self) -> [f32; 3] {
        [self.autoaim_angle, self.magnetism_angle, self.deviation_angle]
    }
    /// Does every non-zero angle in this block land on the authoring grid when read as RADIANS?
    ///
    /// See [`ANGLE_AUTHORING_GRID_DEG`]: this is the property that distinguishes the two readings,
    /// and it is a property of the whole corpus rather than of any one weapon.
    pub fn angles_are_radians(&self) -> bool {
        self.angles().iter().filter(|a| **a != 0.0).all(|a| on_angle_grid(a.to_degrees()))
    }
    /// ...and the same question of the DEGREE reading, which must come back false. A gate that
    /// only checked one direction would pass on a value that happens to satisfy both.
    pub fn angles_are_degrees(&self) -> bool {
        let nz: Vec<f32> = self.angles().iter().copied().filter(|a| *a != 0.0).collect();
        !nz.is_empty() && nz.iter().all(|a| on_angle_grid(*a))
    }
    /// Is this block authored at all? A weapon with no aim assist zeroes the whole struct.
    pub fn is_authored(&self) -> bool {
        self.autoaim_angle > 0.0 && self.autoaim_range > 0.0
    }
}

/// `weap` +0x2CA `magnification levels` / +0x2CC `magnification range` — the zoom block, which is
/// the ONLY tag-sourced quantity the engine's look-at search reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZoomLevels {
    /// `magnification levels` (short). 0 = this weapon does not scope.
    pub levels: i16,
    /// `magnification range` (real bounds) — `(min, max)`.
    pub range: (f32, f32),
}

/// Byte offsets of the zoom block inside a `weap` tag, on the same 0x204 base.
pub const WEAPON_MAGNIFICATION_LEVELS: usize = 0x2CA;
pub const WEAPON_MAGNIFICATION_RANGE: usize = 0x2CC;

impl ZoomLevels {
    pub fn read(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> Option<ZoomLevels> {
        let b = native.tag_meta(cache, weap_tag, WEAPON_MAGNIFICATION_LEVELS as u32, 0x0A)?;
        Some(ZoomLevels {
            levels: i16::from_le_bytes([b[0], b[1]]),
            range: (rd_f32(&b, 2).unwrap_or(1.0), rd_f32(&b, 6).unwrap_or(1.0)),
        })
    }
    /// `[RE] sub_1804856B8`: the magnification at `zoom_level`, which is
    /// `pow(max / min, z / (n - 1)) * min` for more than one level and 1.0 otherwise. An
    /// unzoomed weapon (`zoom_level < 0`) is always 1.0.
    pub fn magnification(&self, zoom_level: i8) -> f32 {
        if zoom_level < 0 || self.levels <= 0 || self.range.0 <= 0.0 {
            return 1.0;
        }
        if self.levels == 1 {
            return self.range.0.max(1.0);
        }
        let t = (zoom_level as f32) / (self.levels as f32 - 1.0);
        ((self.range.1 / self.range.0).powf(t) * self.range.0).max(1.0)
    }
}

/// `[RE] sub_18010F034` `HUD_FindNearestTarget`'s own half-angle: the literal
/// `flt_180A8AB78 = 0.052359879`, i.e. **3.0000 degrees**, hard-coded in `.rdata`.
pub const LOOKAT_HALF_ANGLE_RAD: f32 = 0.052_359_879;
/// ...and its own range: the literal `dword_180A8B3F4 = 40.0` world units, multiplied by the held
/// weapon's current zoom magnification.
pub const LOOKAT_RANGE_WU: f32 = 40.0;

/// The cone a LOOK-AT name is gated by.
///
/// **This is NOT the weapon's aim-assist block.** The engine's HUD look-at name comes from
/// `sub_1802CB01C`, which takes the unit's own aim target (`unit+0x2B8`, filled by the aim-assist
/// system) and, when that is empty, falls back to `HUD_FindNearestTarget` (`sub_18010F034`). That
/// fallback was decompiled field by field and **reads none of `weap` +0x2D4..+0x2F4**: its cone is
/// a hard-coded **3.0 degree half-angle** out to a hard-coded **40.0 world units times the held
/// weapon's zoom magnification** (the only tag it reads is the zoom block at +0x2CA, through
/// `sub_1804856B8`). The test is a squared dot product against `cos(3 deg)`, with the candidate's
/// bounding-sphere radius inflating the cone, and it picks the **nearest** hit rather than the
/// most on-axis one. [`AimAssist`] is still decoded and reported, because it is the block the
/// *other* lane (`unit+0x2B8`) is driven by — but that lane was not decompiled, so nothing here
/// pretends it is.
///
/// WHAT HMS DOES NOT COPY, and why:
/// * **The bounding-sphere inflation.** The engine adds the candidate object's own radius to both
///   ends of the test (`r*r + t*t + 2*r*sin(3deg)*t >= |c-eye|^2 * cos(3deg)^2`). A film gives HMS
///   no object bounding sphere, so the test here is against the body's own anchor point — which
///   makes HMS's cone very slightly NARROWER than the engine's, never wider.
/// * **The line-of-sight raycast.** `HUD_FindNearestTarget` does one per candidate
///   (`sub_180112364`), but the *producer* path that actually writes the name record
///   (`sub_18018ACC8` / `sub_18018AE04`) has none, and the raycast is skipped entirely whenever
///   `unit+0x2B8` already holds a target. Names read through walls in game, and they do here.
/// * **The aim direction.** The engine's cone starts at the camera position (`observer+0x154`) but
///   points along the player's own **aim** direction from the control state (`sub_180062E00`), not
///   along the camera forward. In a Theater first-person view those are the same thing, since the
///   camera IS the recorded aim; with a free camera HMS uses the camera, and says so.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAtCone {
    /// Half-angle of the cone, RADIANS.
    pub angle: f32,
    /// Maximum distance, world units.
    pub range: f32,
    /// The zoom magnification the range was scaled by, for the diagnostic.
    pub magnification: f32,
}

/// The engine's own unzoomed cone — 3 degrees out to 40 world units. It needs no weapon at all,
/// which is why an empty-handed viewer gets exactly the same rule rather than an HMS invention.
pub const UNARMED_LOOKAT: LookAtCone =
    LookAtCone { angle: LOOKAT_HALF_ANGLE_RAD, range: LOOKAT_RANGE_WU, magnification: 1.0 };

impl LookAtCone {
    /// The cone for a viewer holding `zoom` at scope level `zoom_level` (-1 = not scoped).
    pub fn for_weapon(zoom: Option<ZoomLevels>, zoom_level: i8) -> LookAtCone {
        let m = zoom.map(|z| z.magnification(zoom_level)).unwrap_or(1.0);
        LookAtCone { angle: LOOKAT_HALF_ANGLE_RAD, range: LOOKAT_RANGE_WU * m, magnification: m }
    }
    /// Is `target` inside this cone, looking from `eye` along `forward`?
    pub fn contains(&self, eye: glam::Vec3, forward: glam::Vec3, target: glam::Vec3) -> bool {
        let to = target - eye;
        let d = to.length();
        if d <= 1e-4 || d > self.range {
            return false;
        }
        let f = forward.normalize_or_zero();
        if f.length_squared() < 0.5 {
            return false;
        }
        f.dot(to / d) >= self.angle.cos()
    }
    /// How far off-axis `target` is, radians — for the diagnostic to report a near miss.
    pub fn off_axis(eye: glam::Vec3, forward: glam::Vec3, target: glam::Vec3) -> f32 {
        let to = (target - eye).normalize_or_zero();
        let f = forward.normalize_or_zero();
        if to.length_squared() < 0.5 || f.length_squared() < 0.5 {
            return std::f32::consts::PI;
        }
        f.dot(to).clamp(-1.0, 1.0).acos()
    }
}

/// Keep only the ONE body the viewer is looking at.
///
/// Returns the index into `anchors` that won, or `None` when nothing is in the cone. The engine
/// holds one target (`unit+0x2B8` is one object) and `HUD_FindNearestTarget` picks the **NEAREST**
/// candidate inside the cone, not the most on-axis one — so that is what this does.
pub fn looked_at(cone: &LookAtCone, eye: glam::Vec3, forward: glam::Vec3, anchors: &[glam::Vec3]) -> Option<usize> {
    anchors
        .iter()
        .enumerate()
        .filter(|(_, p)| cone.contains(eye, forward, **p))
        .min_by(|a, b| {
            let (x, y) = ((*a.1 - eye).length_squared(), (*b.1 - eye).length_squared());
            x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------------------------
// the virtual screen every chud widget is placed in
// ---------------------------------------------------------------------------------------------

/// The authored virtual screen mapped onto one viewport: the safe frame, and how many screen
/// points one virtual unit is.
#[derive(Clone, Copy, Debug)]
pub struct VirtualScreen {
    pub viewport: egui::Rect,
    /// The safe frame, in screen points — the rectangle the virtual screen maps onto.
    pub safe: egui::Rect,
    /// Screen points per chud virtual unit.
    pub vpx: f32,
}

impl VirtualScreen {
    /// From the open cache's own skin. `safe frame` is a FRACTION of the window inset by the
    /// authored "ding" scaled by `realHeight / 1080` (`+0x2D94BC` / `+0x2D846C`), centred.
    pub fn from_skin(viewport: egui::Rect, skin: &ChudSkin) -> VirtualScreen {
        let cv = skin.fullscreen_wide();
        let (sfh, sfv) = match cv {
            Some(c) => (c.safe_frame[0], c.safe_frame[1]),
            None => (crate::reach_chud::SAFE_FRAME_V_FALLBACK, crate::reach_chud::SAFE_FRAME_V_FALLBACK),
        };
        let (dh, dv) = match cv {
            Some(c) => (c.ding[0], c.ding[1]),
            None => (0.0, 0.0),
        };
        let k = viewport.height() / 1080.0;
        let w = (viewport.width() * sfh - 2.0 * dh * k).max(1.0);
        let h = (viewport.height() * sfv - 2.0 * dv * k).max(1.0);
        VirtualScreen {
            viewport,
            safe: egui::Rect::from_center_size(viewport.center(), egui::vec2(w, h)),
            vpx: px_per_virtual(viewport.height(), skin.virtual_height(), sfv, dv),
        }
    }
    /// With no cache to ask: Reach's own authored numbers.
    pub fn fallback(viewport: egui::Rect) -> VirtualScreen {
        let sf = crate::reach_chud::SAFE_FRAME_V_FALLBACK;
        VirtualScreen {
            viewport,
            safe: egui::Rect::from_center_size(
                viewport.center(),
                egui::vec2(viewport.width() * sf, viewport.height() * sf),
            ),
            vpx: px_per_virtual(viewport.height(), crate::reach_chud::VIRTUAL_HEIGHT_FALLBACK, sf, 0.0),
        }
    }
    /// Where a `chud_anchor_type_enum` value sits, in screen points.
    ///
    /// Only the anchors a first-person HUD actually uses are answered. Everything else — a
    /// tracked target, a territory, a fireteam member, a `<player>` — is a RUNTIME anchor the
    /// engine resolves against live game state, and a film gives HMS nothing to resolve it
    /// against, so it returns `None` and the widget is skipped rather than drawn in the wrong
    /// place.
    pub fn anchor(&self, a: u8) -> Option<egui::Pos2> {
        let s = self.safe;
        Some(match a {
            ANCHOR_TOP_LEFT => s.left_top(),
            ANCHOR_TOP_CENTER => egui::pos2(s.center().x, s.top()),
            ANCHOR_TOP_RIGHT => s.right_top(),
            ANCHOR_CENTER => s.center(),
            ANCHOR_BOTTOM_LEFT => s.left_bottom(),
            ANCHOR_BOTTOM_CENTER => egui::pos2(s.center().x, s.bottom()),
            ANCHOR_BOTTOM_RIGHT => s.right_bottom(),
            // The aim point. In a first-person view that is the centre of the viewport, not of
            // the safe frame.
            ANCHOR_CROSSHAIR => self.viewport.center(),
            ANCHOR_TOP_CENTER_SAVEDFILM => egui::pos2(s.center().x, s.top()),
            _ => return None,
        })
    }
}

/// `chud_anchor_type_enum`, the members a first-person HUD uses.
pub const ANCHOR_PARENT: u8 = 0;
pub const ANCHOR_TOP_LEFT: u8 = 1;
pub const ANCHOR_TOP_CENTER: u8 = 2;
pub const ANCHOR_TOP_RIGHT: u8 = 3;
pub const ANCHOR_CENTER: u8 = 4;
pub const ANCHOR_BOTTOM_LEFT: u8 = 5;
pub const ANCHOR_BOTTOM_CENTER: u8 = 6;
pub const ANCHOR_BOTTOM_RIGHT: u8 = 7;
pub const ANCHOR_CROSSHAIR: u8 = 17;
pub const ANCHOR_TOP_CENTER_SAVEDFILM: u8 = 50;

// ---------------------------------------------------------------------------------------------
// decoding one chud_definition
// ---------------------------------------------------------------------------------------------

/// The `chud_state_enum` members this file names. The whole enum is 330 long; these are the ones
/// a saved film can answer or the ones that decide whether a widget belongs on screen at all.
pub const STATE_FULLSCREEN_WIDE: i32 = 1;
pub const STATE_FIRST_PERSON: i32 = 24;
pub const STATE_THIRD_PERSON: i32 = 25;
pub const STATE_SINGLE_WIELDING: i32 = 43;
pub const STATE_UNARMED: i32 = 42;
pub const STATE_ZOOMED: i32 = 83;
pub const STATE_UNZOOMED: i32 = 84;
pub const STATE_WEAPON_IS_RIGHT_HAND: i32 = 97;
pub const STATE_WEAPON_IS_BACKPACKED: i32 = 99;
pub const STATE_CROSSHAIR_NORMAL: i32 = 100;
pub const STATE_READY_TO_SPAWN: i32 = 298;
pub const STATE_WAITING_TO_SPAWN: i32 = 299;
pub const STATE_MULTI_FFA: i32 = 263;
pub const STATE_MULTI_TEAM: i32 = 264;

/// `chud_state_enum`'s own value names, as `reverseme tagdefs reach_tag_test.exe chdt` prints
/// them. Only used by the diagnostic, so a condition can be read rather than guessed at.
pub fn state_name(v: i32) -> String {
    const NAMES: [(i32, &str); 24] = [
        (1, "fullscreen wide"),
        (21, "spartan"),
        (22, "elite"),
        (24, "first person"),
        (25, "third person"),
        (32, "shield available"),
        (42, "unarmed"),
        (43, "single wielding"),
        (45, "support weapon"),
        (83, "zoomed"),
        (84, "unzoomed"),
        (92, "grenades unusable"),
        (93, "<grenade> selected"),
        (96, "<grenade> empty"),
        (97, "weapon is right hand"),
        (99, "weapon is backpacked"),
        (100, "crosshair normal"),
        (107, "ammo loaded low"),
        (108, "ammo loaded empty"),
        (109, "ammo reserve low"),
        (263, "multi ffa"),
        (264, "multi team"),
        (298, "status ready to spawn"),
        (299, "status waiting to spawn"),
    ];
    match NAMES.iter().find(|(k, _)| *k == v) {
        Some((_, n)) => format!("{v} '{n}'"),
        None => format!("{v}"),
    }
}

/// One `chud_widget_placement_data_base_block` element.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// `window state` — which `chud_curvature_res_flags` layouts this entry describes.
    pub res_flags: u8,
    /// `anchor type`, a `chud_anchor_type_enum` value.
    pub anchor: u8,
    /// `widget origin` — the alignment pivot, 0 = left/top .. 1 = right/bottom.
    pub origin: [f32; 2],
    /// `origin offset`, virtual units, `+x` right and `+y` down from the anchor.
    pub offset: [f32; 2],
    /// `widget scale`. A negative component mirrors on that axis.
    pub scale: [f32; 2],
}

/// The 124-byte `chud_widget_base_struct` every collection and widget starts with.
#[derive(Clone, Debug)]
pub struct WidgetBase {
    /// `artist name` — the widget's own name in the tag.
    pub name: String,
    /// `sort layer`; lower is drawn first.
    pub sort: i8,
    /// Every authored placement, in tag order.
    pub placements: Vec<Placement>,
    /// `Active State` — an **AND of ORs**, kept as the tree the tag authors rather than
    /// flattened. The outer `Vec` is the `chud_widget_state_and_block` terms, each inner `Vec`
    /// that term's `chud_widget_state_or_block` conditions.
    ///
    /// The nesting is not cosmetic. The magnum's reticule halves author ONE term with the two
    /// conditions `97 'weapon is right hand'` and `98 'weapon is left hand'` — an OR, true
    /// whichever hand holds it. Flattened into "any of these" that reads the same, but its
    /// `headshot` pip authors `103 'crosshair enemy headshot'` and `97` and the two readings
    /// disagree outright: as an OR the pip is on all the time, as an AND it is on only while the
    /// reticule is over a head. Keeping the tree is what lets [`WidgetBase::on`] answer either
    /// shape correctly.
    pub active: Vec<Vec<StateCond>>,
    /// `Flash State` (+0x00C) — what the engine PULSES the widget with. A widget with a flash
    /// state and no active state is the flash itself and shows nothing at rest; see
    /// [`flash_only`].
    pub flash: Vec<Vec<StateCond>>,
    /// `Hidden State`, same shape: each term is an AND of ORs that SUPPRESSES the widget.
    pub hidden: Vec<Vec<StateCond>>,
    /// `color output A` as an index into [`SKIN_COLORS`], when it names a skin colour.
    pub color: Option<usize>,
    /// `color output A`, raw, so a non-skin output can be told apart from "no render data".
    pub output_a: i32,
    /// `custom color A` .. `custom color D`, each `[a, r, g, b]`. All four are kept because
    /// `color output A` can name any of them — reading every non-skin output as custom A was
    /// wrong for every widget that authors its colour in B, C or D.
    pub custom: [[f32; 4]; 4],
    /// Does `animation data` authorize an ACTIVE animation for this widget (+0x054 -> +0x14)?
    pub anim_active: bool,
    /// ...and a FLASHING one (+0x28)?
    pub anim_flashing: bool,
    /// `scalar output A` (`chud_render_scalar_output_enum`). 8 is `flash animation` and 9
    /// `scalar animation B`: the widget's intensity is driven by an animation, which at rest is
    /// ZERO. HMS runs no chud animations, so such a widget must not be drawn at full strength.
    pub scalar_a: i32,
    /// #hud-shader  `shader type` — which of the twenty-one chud shaders draws this widget. See
    /// [`shader_type_name`]. `-1` = no render data.
    pub shader_type: i32,
    /// #hud-shader  `alpha blend mode`. See [`blend_mode_name`]. `-1` = no render data.
    pub blend_mode: i32,
}

/// #hud-shader  `chud_shader_type_enum`, all 21 members, read out of the tool build's own
/// reflection data (`chud_shader_type_enum` at `.data +0x2088A50`).
///
/// **This is the field that says what a widget IS**, and nothing in HMS read it until now. A chud
/// widget is not "a sprite tinted by `color output A`" — it is a sprite run through one of these
/// twenty-one shaders, and **seven of them are METERS**. So "is this widget a meter?" is an
/// authored fact, not something to infer from its name, and the colour outputs mean different
/// things per shader (a gradient meter has a ramp; a shield meter has its own law).
pub fn shader_type_name(v: i32) -> &'static str {
    const NAMES: [&str; 21] = [
        "really simple{simple}",
        "anti-alias",
        "meter",
        "text simple",
        "meter shield",
        "meter gradient",
        "crosshair",
        "d-dmg (don't use)",
        "solid (don't use)",
        "sensor (don't use)",
        "meter single-color",
        "navpoints",
        "medal",
        "texture cam",
        "meter chapter",
        "meter double gradient",
        "meter radial gradient",
        "distortion and blur",
        "emblem",
        "ddamage apply",
        "damage tracker (don't use)",
    ];
    NAMES.get(v as usize).copied().unwrap_or("<out of range>")
}

/// The `chud_shader_type_enum` members that are METERS — the ones that take a fill fraction.
///
/// Seven of twenty-one, and the list is the enum's own: `meter` (2), `meter shield` (4),
/// `meter gradient` (5), `meter single-color` (10), `meter chapter` (14),
/// `meter double gradient` (15), `meter radial gradient` (16).
pub const METER_SHADERS: [i32; 7] = [2, 4, 5, 10, 14, 15, 16];

/// `chud_shader_type_enum` 0 — a plain tinted sprite, the only shader HMS actually reproduces.
pub const SHADER_REALLY_SIMPLE: i32 = 0;
/// ...2 `meter`, the simplest of the meter family.
pub const SHADER_METER: i32 = 2;
/// ...4 `meter shield`, which is what Reach's own shield bar authors.
pub const SHADER_METER_SHIELD: i32 = 4;
/// ...6 `crosshair`, the reticule's own shader.
pub const SHADER_CROSSHAIR: i32 = 6;

/// Is this shader one of the seven meters?
pub fn shader_is_meter(v: i32) -> bool {
    METER_SHADERS.contains(&v)
}

/// #hud-shader  `render_blend_mode`, all 18 members, from `.data +0x208DB10`.
///
/// The other field nothing read. HMS paints every widget as one textured quad through egui's
/// single pipeline, which is premultiplied alpha-over — so a widget authored `additive`,
/// `multiply` or `maximum` is drawn in the wrong blend and looks wrong however right its colour,
/// its sprite and its rectangle are.
pub fn blend_mode_name(v: i32) -> &'static str {
    const NAMES: [&str; 18] = [
        "opaque",
        "additive",
        "multiply",
        "alpha blend",
        "double multiply",
        "pre multiplied alpha",
        "maximum",
        "multiply add",
        "add src times dstalpha",
        "add src times srcalpha",
        "inverse alpha blend",
        "motion blur static",
        "motion blur inhibit",
        "apply shadow to shadow mask",
        "alpha blend constant",
        "overdraw apply",
        "wet effect",
        "minimum",
    ];
    NAMES.get(v as usize).copied().unwrap_or("<out of range>")
}

/// `render_blend_mode` 0 `opaque`, 3 `alpha blend`, 5 `pre multiplied alpha` — reproduced by an
/// ordinary alpha-over quad, which is what egui draws.
pub const BLEND_REPRODUCIBLE: [i32; 4] = [0, 1, 3, 5];
/// ...1 `additive`.
pub const BLEND_ADDITIVE: i32 = 1;

/// Can HMS draw this blend mode as it is authored? (`additive` included — see [`blend_tint`].)
pub fn blend_is_reproducible(v: i32) -> bool {
    BLEND_REPRODUCIBLE.contains(&v)
}

/// #hud-shader  The vertex colour that makes egui's one pipeline draw `mode` CORRECTLY.
///
/// **This is exact, not an approximation.** egui blends with `src_factor = ONE` and
/// `dst_factor = ONE_MINUS_SRC_ALPHA` over PREMULTIPLIED vertex colours, i.e. the framebuffer
/// gets `src.rgb + dst.rgb * (1 - src.a)`. So:
///
/// * `alpha blend` / `opaque` / `pre multiplied alpha` — premultiply by alpha, the ordinary case;
/// * **`additive` — premultiplied RGB with `a = 0`**, which leaves `dst` untouched and ADDS
///   `src.rgb`. That is additive blending identically, through the same pipeline.
///
/// Why it matters, measured: Reach's shield bar authors `shader 4 'meter shield'` in
/// `alpha blend`, and its BORDER authors `shader 1 'anti-alias'` in **`additive`** at the
/// identical 249x57 pt rectangle with a higher sort layer. Drawn additive the border brightens the
/// bar's frame; drawn alpha-over — which is what HMS did — an opaque dark blue plate covers the
/// bright bar completely, which is what "the shields and health don't look right" was.
///
/// A mode this cannot express returns `None` and the caller must not pretend.
pub fn blend_tint(color: egui::Color32, mode: i32) -> Option<egui::Color32> {
    let (r, g, b, a) = (color.r(), color.g(), color.b(), color.a());
    match mode {
        // Additive: carry the colour, claim no coverage.
        BLEND_ADDITIVE => Some(egui::Color32::from_rgba_premultiplied(r, g, b, 0)),
        // The alpha-over family, and "no render data" (-1), which is the ordinary case.
        -1 | 0 | 3 | 5 => Some(color),
        _ => None,
    }
}

/// `chud_widget_render_data_base_block_struct` +0x000 `shader type` (a `char enum`).
pub const RENDER_SHADER_TYPE: usize = 0x00;
/// ...+0x074 `alpha blend mode` (a `short enum`).
pub const RENDER_ALPHA_BLEND_MODE: usize = 0x74;

/// `chud_render_scalar_output_enum` 8 `flash animation` and 9 `scalar animation B`.
pub const SCALAR_FLASH_ANIMATION: i32 = 8;
pub const SCALAR_ANIMATION_B: i32 = 9;

impl WidgetBase {
    /// The placement for the full-screen widescreen window state, else the first authored one.
    pub fn placement(&self) -> Option<&Placement> {
        self.placements
            .iter()
            .find(|p| p.res_flags & crate::reach_chud::RES_FULLSCREEN_WIDE != 0)
            .or_else(|| self.placements.first())
    }
    /// Does the engine draw this with no condition on it?
    pub fn unconditional(&self) -> bool {
        self.active.is_empty() && self.hidden.is_empty()
    }
    /// `custom color A`.
    pub fn custom_a(&self) -> [f32; 4] {
        self.custom[0]
    }
    /// #hud-shader  Is this widget a METER by its own authored `shader type`?
    ///
    /// **The tag's answer, replacing a name heuristic.** Whether a widget takes a fill fraction was
    /// being inferred from its NAME (`w.starts_with("meter") && !w.starts_with("meter_border")`),
    /// which is exactly the shape of rule this project has been bitten by before. `shader type`
    /// states it: seven of the twenty-one shaders are meters.
    pub fn is_meter_shader(&self) -> bool {
        shader_is_meter(self.shader_type)
    }

    /// #hud-shader  Can HMS draw this widget in the blend mode it authors?
    pub fn blend_reproducible(&self) -> bool {
        self.blend_mode < 0 || blend_is_reproducible(self.blend_mode)
    }

    /// Is this widget's intensity driven by an animation HMS does not run?
    ///
    /// `+0x060 scalar output A` names what multiplies the widget: `flash animation` and
    /// `scalar animation B` are both zero at rest, so the engine draws nothing until something
    /// triggers them. The shield's `flahsy` blob (authored `crosshair enemy`, a solid red) and
    /// every `*_flash` layer are exactly these, and drawing them unconditionally paints the HUD
    /// with damage flashes that are not happening.
    pub fn animation_driven(&self) -> bool {
        matches!(self.scalar_a, SCALAR_FLASH_ANIMATION | SCALAR_ANIMATION_B)
    }
    /// Every `chud_state_enum` value this widget's conditions mention, for the diagnostic.
    pub fn mentions(&self) -> Vec<i32> {
        let mut v: Vec<i32> =
            self.active.iter().chain(self.hidden.iter()).flatten().map(|c| c.cond).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    /// Would this widget be on, given the set of states HMS can actually assert?
    ///
    /// `known` is every `chud_state_enum` value HMS is prepared to say is TRUE this frame. The
    /// tag's own boolean shape is honoured: a state block holds `chud_widget_state_and_block`
    /// terms, every one of which must hold, and each term holds
    /// `chud_widget_state_or_block` conditions, any one of which satisfies it. An empty Active
    /// block is unconditional; a satisfied Hidden block suppresses the widget outright.
    ///
    /// A condition HMS cannot evaluate counts as FALSE, so an unknowable widget stays off rather
    /// than being drawn on a guess. That is deliberately asymmetric and it is the safe direction:
    /// a missing piece of HUD is a gap, a wrongly-drawn one is a lie.
    pub fn on(&self, known: &[i32]) -> bool {
        if satisfied(&self.hidden, known) && !self.hidden.is_empty() {
            return false;
        }
        satisfied(&self.active, known)
    }
}

/// Resolve a bitmap widget's art, whether the `bitm` is a SPRITE SHEET or a plain bitmap.
///
/// `reach_chud::read_sprite` answers only for a sheet: it walks `bitm` +0x070 `sequences` to the
/// requested sequence and then that sequence's +0x034 `sprites` block. Reach's reticule sheets are
/// sprite sheets and that is all the crosshair ever needed — but **most of the HUD is not**. A
/// `campaign_shield_meter`, a `campaign_health_meter`, a `ballistic_meters_2_rows`, a
/// `grenade_scematics`: each sequence of those names a whole submap and authors no sprite
/// rectangle at all, so the sprite walk comes back `None` and the widget is silently dropped.
/// That is why the shield block first decoded with ONE widget (its `warning` text) and the health
/// block with NONE.
///
/// The fallback is the engine's own other path: a sequence with no sprites names its submap
/// through `+0x020 first bitmap index`, and the drawn size is that submap's own
/// `+0x000 width` x `+0x002 height` with the same flag adjustment the sprite path uses — bit 0
/// halves the stored dimensions and bit 4 divides them by three, for the 2x and 3x supersampled
/// sheets.
pub fn read_widget_art(
    native: &NativeDll,
    cache: CacheHandle,
    tag: u32,
    sequence: usize,
) -> Option<(NavpointSprite, ArtRoute)> {
    if let Some(sp) = crate::reach_chud::read_sprite(native, cache, tag, sequence, 0) {
        return Some((sp, ArtRoute::Sheet));
    }
    let meta = native.tag_meta(cache, tag, 0, 0xC0)?;
    // The sequence's own submap, through `+0x020 first bitmap index`.
    let seqs = block(native, cache, &meta, 0x70, 64, 256);
    let named = seqs
        .chunks_exact(64)
        .nth(sequence)
        .and_then(|s| Some(i16::from_le_bytes(s.get(0x20..0x22)?.try_into().ok()?)))
        .filter(|v| *v >= 0)
        .map(|v| v as usize);
    let route = if named.is_some() { ArtRoute::FirstBitmapIndex } else { ArtRoute::Unresolved };
    let bitmap_index = named.unwrap_or(0);
    let bitmaps = block(native, cache, &meta, 0x7C, 56, 65536);
    let bm = bitmaps.chunks_exact(56).nth(bitmap_index)?;
    let mut w = i16::from_le_bytes(bm.get(0x00..0x02)?.try_into().ok()?) as i32;
    let mut h = i16::from_le_bytes(bm.get(0x02..0x04)?.try_into().ok()?) as i32;
    let flags = *bm.get(0x05)?;
    if flags & 0x01 != 0 {
        w >>= 1;
        h >>= 1;
    } else if flags & 0x10 != 0 {
        w = (w as f32 / 3.0).floor() as i32;
        h = (h as f32 / 3.0).floor() as i32;
    }
    if w <= 0 || h <= 0 {
        return None;
    }
    Some((
        NavpointSprite {
            bitmap_tag: tag,
            bitmap_index: bitmap_index as u16,
            uv: [0.0, 1.0, 0.0, 1.0],
            size: [w as f32, h as f32],
        },
        route,
    ))
}

/// One `chud_widget_state_or_block` clause: a `chud_state_enum` condition, possibly NEGATED.
///
/// `chud_widget_state_flags` has exactly one member and it is **bit 0 `Invert condition` — "Check
/// if the condition is false"**. Ignoring it inverts the meaning of real widgets: the Spartan
/// HUD's whole `grenades` collection is authored `NOT 92 'grenades unusable'`, which read without
/// the flag says "draw the grenades only when grenades are unusable" and means the block never
/// appears at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCond {
    pub cond: i32,
    /// `chud_widget_state_flags` bit 0.
    pub invert: bool,
}

impl StateCond {
    pub fn is(cond: i32) -> StateCond {
        StateCond { cond, invert: false }
    }
    pub fn not(cond: i32) -> StateCond {
        StateCond { cond, invert: true }
    }
    /// Does this clause hold, given the states that are TRUE and the states HMS can DECIDE at all?
    ///
    /// A condition outside `decidable` is unknown, and an unknown clause never holds — **not even
    /// an inverted one**. That asymmetry is deliberate and it is load-bearing. A plain closed
    /// world ("anything not listed is false") would make `NOT <anything HMS has never thought
    /// about>` come out TRUE, and the first real widget it reaches is the Spartan HUD's compass:
    /// `NOT 169 'watermark compass on'`, a 1024-unit strip that would be painted across the top of
    /// every frame on the strength of a state nobody evaluated. A missing piece of HUD is a gap;
    /// a confidently drawn wrong one is a lie.
    pub fn holds(&self, known: &[i32], decidable: &[i32]) -> bool {
        decidable.contains(&self.cond) && (known.contains(&self.cond) != self.invert)
    }
}

/// `chud_widget_state_flags` bit 0 — `Invert condition`.
pub const STATE_FLAG_INVERT: u8 = 1 << 0;

/// Evaluate one `chud` state block: an AND over terms, each term an OR over clauses.
///
/// `known` is a **CLOSED WORLD**: every `chud_state_enum` value in it is true and every value not
/// in it is false. That has to be stated, because it is what an inverted clause depends on —
/// `NOT 92` holds precisely because 92 is absent from `known`. [`RESTING_STATES`] is built to be
/// exactly that complete truth set for a resting first-person HUD, and
/// [`HudValues::known_states`] for a live frame.
///
/// An EMPTY block is satisfied — which for an `Active State` means "unconditional" and is why
/// [`WidgetBase::on`] checks `hidden` for emptiness separately before letting it suppress
/// anything.
pub fn satisfied(block: &[Vec<StateCond>], known: &[i32]) -> bool {
    let decidable = decidable_states(known);
    block.iter().all(|term| term.is_empty() || term.iter().any(|c| c.holds(known, &decidable)))
}

/// The `chud_state_enum` values HMS is prepared to decide AT ALL: the ones it asserts are true
/// (`known`) plus the ones it asserts are FALSE.
///
/// Everything else is unknown, and [`StateCond::holds`] says why that matters.
pub fn decidable_states(known: &[i32]) -> Vec<i32> {
    let mut v = known.to_vec();
    v.extend(KNOWN_FALSE_STATES);
    v
}

/// The states HMS positively asserts are FALSE for a living, armed, unscoped player seen in first
/// person on a full-screen widescreen display.
///
/// Only an INVERTED clause can read these, and listing them is what lets the grenade block draw:
/// its gate is `NOT 92 'grenades unusable'`, which needs HMS to be willing to say 92 is false.
pub const KNOWN_FALSE_STATES: [i32; 16] = [
    2, 3, 4, 5, // the other window layouts
    25,  // third person
    31,  // shielded critical
    33, 34, 35, // overshield 2/3/4
    42,  // unarmed -- replaced by STATE_UNARMED in `known` when the body holds nothing
    46,  // in sabre
    49,  // binoculars enabled
    83,  // zoomed
    92,  // grenades unusable
    98,  // weapon is left hand -- Reach has no dual wielding
    99,  // weapon is backpacked
];

/// The `chud_state_enum` values that are TRUE for a living, armed, unscoped player seen in first
/// person on a full-screen widescreen display — the RESTING state of the HUD.
///
/// This is the set a reticule has to be evaluated against, and getting it wrong is why 34 of
/// Reach's 64 weapons looked as though they authored no crosshair at all. They author one; every
/// piece of it is gated. The magnum's five reticule parts are each `97 'weapon is right hand'` OR
/// `98 'weapon is left hand'`, so a filter that demanded an EMPTY Active state rejected all of
/// them and the decode reported "no reticule" for the whole player sandbox — magnum, DMR, assault
/// rifle, shotgun, sniper, needle rifle, plasma pistol, rockets, sword, hammer — while the 29 it
/// did decode were the vehicle and turret weapons, whose reticules happen to be unconditional.
///
/// `98 'weapon is left hand'` is NOT in this set: Reach has no dual wielding, so the engine puts
/// the one held weapon in the right hand, and a part gated on the left hand alone genuinely never
/// draws. It still reaches the screen through the OR it shares with `97`.
pub const RESTING_STATES: [i32; 9] = [
    STATE_FULLSCREEN_WIDE,        // 1   the window layout HMS maps the virtual screen onto
    STATE_FIRST_PERSON,           // 24  this is the first-person HUD
    STATE_SHIELD_AVAILABLE,       // 32  a living Spartan has shields
    STATE_SINGLE_WIELDING,        // 43  one weapon in hand
    STATE_BINOCULARS_DISABLED,    // 50  no Spartan in multiplayer carries binoculars
    STATE_UNZOOMED,               // 84  the resting optic
    STATE_WEAPON_IS_RIGHT_HAND,   // 97  where Reach puts it
    STATE_CROSSHAIR_NORMAL,       // 100 nothing under the reticule
    STATE_MULTI_FFA,              // 263 a saved film is a multiplayer game
];

/// `chud_state_enum` 50 `binoculars disabled`. The Spartan HUD's whole `health_meter` collection
/// is gated on it, so without it the health bar never draws at all.
pub const STATE_BINOCULARS_DISABLED: i32 = 50;

/// `chud_state_enum` 32 `shield available`.
pub const STATE_SHIELD_AVAILABLE: i32 = 32;
/// `chud_state_enum` 92 `grenades unusable`, 93 `<grenade> selected`, 94 `<grenade> unselected`.
///
/// The angle brackets mark a DATASOURCE-RELATIVE state: a datasource-driven collection is drawn
/// once per element and `<grenade> selected` is true on the repeat for the type the player
/// currently has up. So these cannot live in a frame-wide truth set — they are added per repeat
/// by [`repeat_states`].
pub const STATE_GRENADES_UNUSABLE: i32 = 92;
pub const STATE_GRENADE_SELECTED: i32 = 93;
pub const STATE_GRENADE_UNSELECTED: i32 = 94;

/// `chud_datasource_type_enum` 0 `grenades`.
pub const DATASOURCE_GRENADES: i32 = 0;

/// The extra states that are true for ONE repeat of a datasource-driven collection.
///
/// Without these the grenade block draws its frame and nothing inside it: every one of its four
/// widgets is gated on `<grenade> selected` or `<grenade> unselected`, which are properties of
/// the repeat rather than of the frame. HMS has no recorded "which grenade type is up", so it
/// takes repeat 0 to be the selected one — the engine's own element order, frag then plasma —
/// and says so.
pub fn repeat_states(datasource: Option<i32>, repeat: usize) -> Vec<i32> {
    match datasource {
        Some(DATASOURCE_GRENADES) if repeat == 0 => vec![STATE_GRENADE_SELECTED],
        Some(DATASOURCE_GRENADES) => vec![STATE_GRENADE_UNSELECTED],
        _ => Vec::new(),
    }
}

/// Read one widget's two state blocks out of its 124-byte base, without decoding the rest of it.
///
/// Exposed so `reach_chud::weapon_crosshair` can ask the same question this file asks, rather
/// than keeping a second, differently-wrong copy of the rule.
pub fn read_states(
    native: &NativeDll,
    cache: CacheHandle,
    widget_base: &[u8],
) -> (Vec<Vec<StateCond>>, Vec<Vec<StateCond>>, Vec<Vec<StateCond>>) {
    let sd = block(native, cache, widget_base, 0x0C + 0x10, 56, 1);
    let conds = |off: usize| -> Vec<Vec<StateCond>> {
        let Some(sd) = sd.chunks_exact(56).next() else { return Vec::new() };
        block(native, cache, sd, off, 12, 10)
            .chunks_exact(12)
            .map(|and| {
                block(native, cache, and, 0x00, 8, 10)
                    .chunks_exact(8)
                    .filter_map(|or| {
                        // `chud_widget_state_or_block`: +0x000 byte flags, +0x004 long enum
                        // condition. Bit 0 of the flags NEGATES it.
                        Some(StateCond { cond: rd_i32(or, 0x04)?, invert: or[0] & STATE_FLAG_INVERT != 0 })
                    })
                    .collect::<Vec<StateCond>>()
            })
            .filter(|t: &Vec<StateCond>| !t.is_empty())
            .collect()
    };
    // `chud_widget_state_data_base_block_struct`: +0x000 Active, +0x00C FLASH, +0x018 Hidden.
    // The middle one was never read, and it is what the shield's damage blob hangs on.
    (conds(0x00), conds(0x0C), conds(0x18))
}

/// Does this widget draw in the HUD's RESTING state?
///
/// The question `reach_chud::weapon_crosshair` actually wants: not "is it unconditional" but "is
/// it on when a living player is holding this weapon and looking at nothing in particular".
pub fn widget_draws_at_rest(native: &NativeDll, cache: CacheHandle, widget_base: &[u8]) -> bool {
    let (active, flash, hidden) = read_states(native, cache, widget_base);
    if satisfied(&hidden, &RESTING_STATES) && !hidden.is_empty() {
        return false;
    }
    satisfied(&active, &RESTING_STATES)
}

/// Is this widget nothing but a FLASH — on only while something is flashing?
///
/// **NOT a usable rule, and kept only as the record of why.** `Flash State` does not gate a
/// widget, it PULSES one: the Spartan HUD's `health_meter` — the health bar itself — authors no
/// active state and `FLASH when (26 OR 27 OR 28)`, i.e. it flashes while you are unshielded and is
/// drawn the rest of the time. Suppressing flash-only widgets therefore deleted the health bar.
#[allow(dead_code)]
///
/// `Flash State` is the third block of a widget's state data and it is what the engine pulses the
/// widget with. A widget that authors a flash state and NO active state is not an always-on piece
/// of HUD: it is the flash itself, and at rest the engine shows nothing. The shield block's
/// `flahsy` is exactly that — a 202x54 unit blob of `ui\chud\bitmaps\blob` coloured
/// `crosshair enemy`, i.e. SOLID RED, authored with an empty Active State so every
/// "is it unconditional" test says yes.
///
/// Drawing it painted a red bar across the shield meter on every frame: "the shields and grenades
/// stuff looks wrong as well". HMS runs no chud animations and a film states no damage flash, so
/// a flash-only widget is not drawn.
pub fn flash_only(active: &[Vec<StateCond>], flash: &[Vec<StateCond>]) -> bool {
    active.is_empty() && !flash.is_empty()
}

/// #hud-art  Which of `bitm`'s two addressing schemes answered for one widget.
///
/// A `chud` bitmap widget names a `bitm` and a `sequence index`, and a `bitm` can be addressed two
/// ways: a SPRITE SHEET, where the sequence carries a `sprites` block of sub-rectangles, or a plain
/// sequence, which names a whole submap through `first bitmap index`. Reach's HUD uses both, and
/// **which one answered is the difference between a grenade icon and the wrong grenade icon**, so
/// it is recorded rather than lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtRoute {
    /// The sequence's own `sprites` block gave a sub-rectangle.
    Sheet,
    /// The sequence named a whole submap through `+0x020 first bitmap index`.
    FirstBitmapIndex,
    /// Neither: the sequence index does not address a sequence this `bitm` has. **Submap 0 is NOT
    /// substituted** — see [`read_widget_art`].
    Unresolved,
}

impl ArtRoute {
    pub fn word(self) -> &'static str {
        match self {
            ArtRoute::Sheet => "sprite sheet",
            ArtRoute::FirstBitmapIndex => "first bitmap index",
            ArtRoute::Unresolved => "UNRESOLVED",
        }
    }
}

/// What a widget draws.
#[derive(Clone, Debug)]
pub enum WidgetArt {
    /// A sprite out of a `bitm` sheet.
    Bitmap {
        sprite: NavpointSprite,
        flags: u32,
        /// #hud-art  The widget's own authored `sequence index` (`chud_widget_bitmap_block`
        /// +0x090). **Printed, because it is the field a silent fallback throws away**: if the
        /// sprite walk cannot answer, [`read_widget_art`] used to drop to submap 0 regardless of
        /// what this says, so every widget of a sheet drew the SAME picture.
        sequence: i8,
        /// Which route resolved the art: the sprite-sheet walk, the sequence's own
        /// `first bitmap index`, or the submap-0 fallback.
        route: ArtRoute,
    },
    /// A string, in one of Reach's eleven fonts.
    Text {
        font: Option<crate::reach_font::FontId>,
        /// `input string` — a string id. For an ammo block this is either `chud_out_a` (a runtime
        /// NUMBER the engine substitutes) or a literal like `assault_rifle`, which in
        /// `tv_nord_hud-15` is the weapon's SCHEMATIC glyph rather than letters.
        input: String,
        /// `chud_widget_text_flags`.
        flags: u16,
    },
}

/// One decoded widget: its base, and what it draws.
#[derive(Clone, Debug)]
pub struct HudWidget {
    pub base: WidgetBase,
    pub art: WidgetArt,
}

/// One `chud_widget_collection_block` element.
#[derive(Clone, Debug)]
pub struct HudCollection {
    pub base: WidgetBase,
    /// `datasource` type, a `chud_datasource_type_enum` value, or `None` for `<none>` (-1).
    pub datasource: Option<i32>,
    /// `render maximum` — how many repeats of the collection the datasource may draw.
    pub datasource_max: i32,
    /// Per-repeat `origin offset` / `widget scale`, "as offsets from the previous one". The
    /// grenade block uses this to lay its two types out.
    pub positions: Vec<([f32; 2], [f32; 2])>,
    pub widgets: Vec<HudWidget>,
}

impl HudCollection {
    pub fn get(&self, name: &str) -> Option<&HudWidget> {
        self.widgets.iter().find(|w| w.base.name == name)
    }
}

/// One decoded `chud_definition`.
#[derive(Clone, Debug)]
pub struct HudChud {
    pub tag: u32,
    pub path: String,
    pub collections: Vec<HudCollection>,
}

impl HudChud {
    pub fn get(&self, name: &str) -> Option<&HudCollection> {
        self.collections.iter().find(|c| c.base.name == name)
    }
}

/// `chud_widget_text_flags` bit 0 — the string is a NUMBER the engine substitutes.
pub const TEXT_IS_NUMBER: u16 = 1 << 0;
/// bit 12 `format as time`.
pub const TEXT_FORMAT_AS_TIME: u16 = 1 << 12;
/// The `input string` every runtime-number text widget carries.
pub const CHUD_OUT_A: &str = "chud_out_a";

/// Read a whole `chud_definition`.
pub fn read_chud(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<HudChud> {
    let meta = native.tag_meta(cache, tag, 0, 0x28)?;
    let path = native.tag_name(cache, tag).unwrap_or_default();
    let sid = |v: u32| native.string_id(cache, v).unwrap_or_default();
    let colls = block(native, cache, &meta, 0x00, 220, 32);
    let mut collections = Vec::new();
    for coll in colls.chunks_exact(220) {
        let base = read_base(native, cache, coll, &sid);
        let ds = block(native, cache, coll, 0x7C + 0x10, 32, 1);
        let (datasource, datasource_max, positions) = match ds.chunks_exact(32).next() {
            Some(d) => {
                let ty = i16::from_le_bytes([d[4], d[5]]) as i32;
                let max = i16::from_le_bytes([d[6], d[7]]) as i32;
                let pos = block(native, cache, d, 0x14, 16, 32)
                    .chunks_exact(16)
                    .map(|p| {
                        (
                            [rd_f32(p, 0).unwrap_or(0.0), rd_f32(p, 4).unwrap_or(0.0)],
                            [rd_f32(p, 8).unwrap_or(1.0), rd_f32(p, 12).unwrap_or(1.0)],
                        )
                    })
                    .collect();
                ((ty >= 0).then_some(ty), max, pos)
            }
            None => (None, 0, Vec::new()),
        };
        let mut widgets = Vec::new();
        for w in block(native, cache, coll, 0x98, 156, 32).chunks_exact(156) {
            let b = read_base(native, cache, w, &sid);
            let flags = rd_u32(w, 0x7C).unwrap_or(0);
            let seq = w[0x90] as i8;
            let Some(bt) = tag_ref(w, 0x80) else { continue };
            if seq < 0 {
                continue;
            }
            let Some((sprite, route)) = read_widget_art(native, cache, bt, seq as usize) else { continue };
            widgets.push(HudWidget { base: b, art: WidgetArt::Bitmap { sprite, flags, sequence: seq, route } });
        }
        for w in block(native, cache, coll, 0xA4, 132, 60).chunks_exact(132) {
            let b = read_base(native, cache, w, &sid);
            let flags = u16::from_le_bytes([w[0x7C], w[0x7D]]);
            let font = crate::reach_font::FontId::from_index(i16::from_le_bytes([w[0x7E], w[0x7F]]) as i32);
            let input = rd_u32(w, 0x80).map(sid).unwrap_or_default();
            widgets.push(HudWidget { base: b, art: WidgetArt::Text { font, input, flags } });
        }
        widgets.sort_by_key(|w| w.base.sort);
        collections.push(HudCollection { base, datasource, datasource_max, positions, widgets });
    }
    Some(HudChud { tag, path, collections })
}

/// Read the 124-byte base struct every collection and widget shares.
fn read_base(native: &NativeDll, cache: CacheHandle, buf: &[u8], sid: &dyn Fn(u32) -> String) -> WidgetBase {
    let name = rd_u32(buf, 0).map(sid).unwrap_or_default();
    let sort = buf.get(7).copied().unwrap_or(0) as i8;
    let placements = block(native, cache, buf, 0x28 + 0x10, 28, 5)
        .chunks_exact(28)
        .map(|p| Placement {
            res_flags: p[0],
            anchor: p[1],
            origin: [rd_f32(p, 0x04).unwrap_or(0.0), rd_f32(p, 0x08).unwrap_or(0.0)],
            offset: [rd_f32(p, 0x0C).unwrap_or(0.0), rd_f32(p, 0x10).unwrap_or(0.0)],
            scale: [rd_f32(p, 0x14).unwrap_or(1.0), rd_f32(p, 0x18).unwrap_or(1.0)],
        })
        .collect();
    let (active, flash, hidden) = read_states(native, cache, buf);
    let render = block(native, cache, buf, 0x60 + 0x10, 120, 1);
    let r = render.chunks_exact(120).next();
    let color = r.and_then(|r| {
        let v = i16::from_le_bytes([r[0x54], r[0x55]]) as i32;
        (v >= COLOR_OUTPUT_BASE).then(|| (v - COLOR_OUTPUT_BASE) as usize).filter(|k| *k < SKIN_COLORS.len())
    });
    // `custom color A` .. `D` are four consecutive `real argb color`s from +0x004.
    let mut custom = [[1.0f32; 4]; 4];
    if let Some(r) = r {
        for (i, c) in custom.iter_mut().enumerate() {
            let b = 0x04 + i * 0x10;
            *c = [
                rd_f32(r, b).unwrap_or(0.0),
                rd_f32(r, b + 4).unwrap_or(0.0),
                rd_f32(r, b + 8).unwrap_or(0.0),
                rd_f32(r, b + 12).unwrap_or(0.0),
            ];
        }
    }
    // `animation data` (+0x044 struct, +0x054 block, 100-byte element): five sub-structs of
    // `{byte flags, char input type, pad2, tag reference animation}` at +0x00 initializing,
    // +0x14 active, +0x28 flashing, +0x3C readying, +0x50 unreadying.
    let anim = block(native, cache, buf, 0x44 + 0x10, 100, 2);
    let anim_el = anim.chunks_exact(100).next();
    let has_anim = |off: usize| -> bool { anim_el.is_some_and(|a| tag_ref(a, off + 0x04).is_some()) };
    let (anim_active, anim_flashing) = (has_anim(0x14), has_anim(0x28));
    let output_a = r.map(|r| i16::from_le_bytes([r[0x54], r[0x55]]) as i32).unwrap_or(-1);
    let scalar_a = r.map(|r| i16::from_le_bytes([r[0x60], r[0x61]]) as i32).unwrap_or(-1);
    // #hud-shader  The two fields that say what the widget IS and how it is composited.
    let shader_type = r.map(|r| r[RENDER_SHADER_TYPE] as i8 as i32).unwrap_or(-1);
    let blend_mode = r
        .map(|r| i16::from_le_bytes([r[RENDER_ALPHA_BLEND_MODE], r[RENDER_ALPHA_BLEND_MODE + 1]]) as i32)
        .unwrap_or(-1);
    WidgetBase {
        name, sort, placements, active, flash, hidden, color, output_a, custom, scalar_a,
        anim_active, anim_flashing, shader_type, blend_mode,
    }
}

// ---------------------------------------------------------------------------------------------
// the live values, and which of them a film can give
// ---------------------------------------------------------------------------------------------

/// How a HUD field got its value — printed in the panel so a viewer is never shown a number
/// without being told where it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prov {
    /// Straight out of the recording or out of the open cache's tags.
    Recorded,
    /// Computed from things the recording states (a magazine counted down from the authored size
    /// by the fire and reload events, a countdown from the variant's respawn time and a death
    /// tick). Honest arithmetic on recorded facts — not a replicated field.
    Derived,
    /// The recording does not carry it. The field is NOT drawn.
    Absent,
}

impl Prov {
    pub fn word(self) -> &'static str {
        match self {
            Prov::Recorded => "recorded",
            Prov::Derived => "derived",
            Prov::Absent => "not in the recording",
        }
    }
}

/// One value plus where it came from. `Absent` always carries `None`.
#[derive(Clone, Copy, Debug)]
pub struct Field<T> {
    pub value: Option<T>,
    pub prov: Prov,
}

impl<T> Field<T> {
    pub fn recorded(v: T) -> Field<T> {
        Field { value: Some(v), prov: Prov::Recorded }
    }
    pub fn derived(v: T) -> Field<T> {
        Field { value: Some(v), prov: Prov::Derived }
    }
    pub fn absent() -> Field<T> {
        Field { value: None, prov: Prov::Absent }
    }
}

impl<T> Default for Field<T> {
    fn default() -> Self {
        Field::absent()
    }
}

/// Everything the first-person HUD needs this frame, with each field's provenance attached.
#[derive(Clone, Debug, Default)]
pub struct HudValues {
    /// The `weap` the subject is holding, and its `chud interface` — the tag that authors the
    /// ammo block and the reticule.
    pub weapon_tag: Option<u32>,
    /// Rounds in the magazine.
    ///
    /// **A saved film does not carry this, and it cannot be counted either.** The unit's derived
    /// block does have the fields — `0x15` is a 9-bit "rounds loaded" and `0x16` an 11-bit
    /// reserve, both already costed and thrown away in `film_anim` — but the writer's gate for
    /// that whole weapon-slot group is never set: a census over **eight films and 5,438,363
    /// unit messages found zero occurrences of bits 0x15..0x28.** Nor can it be reconstructed
    /// from the event channel: `weapon_reload` (kind 27, the densest weapon event there is at
    /// 26,254 in the corpus) carries **four flag bits and nothing else** — no magazine size, no
    /// round count — so there is no number to count down from. The field stays `Absent` and the
    /// number is not drawn.
    pub loaded: Field<i32>,
    /// The magazine size the meter needs as a denominator: `magazines[0]`'s own
    /// `rounds loaded maximum`.
    ///
    /// **RECOVERED.** The tool build's `weap` definition carries the `magazines` block's offset
    /// (+0x3F0) but prints its element struct as `0 bytes` — one of the runtime-installed field
    /// sets. The layout is now read out of the binary three independent ways and lives in
    /// [`crate::play_hud`]: 96-byte elements (`weapon_magazine_fields_v2`) with
    /// `rounds loaded maximum` at +0x0A. Theater still leaves this `None`, because a film states
    /// no magazine CONTENTS to put over the denominator; test mode fills both.
    pub loaded_max: Option<i32>,
    /// Rounds in reserve — `Absent` for the same reason as [`HudValues::loaded`].
    pub reserve: Field<i32>,
    /// Frag / plasma grenades. **This one the recording does carry**: unit derived field `0x2E`
    /// is sixteen bits, one byte per type, and it arrives 7,412 times in 5.4 M messages — i.e.
    /// whenever a count changes.
    pub frags: Field<i32>,
    pub plasmas: Field<i32>,
    /// Shield and health as a FRACTION. The recording carries no vitality channel; what it does
    /// carry is whether a body is alive, so these are `Derived` 1.0 for a living body and 0.0 for
    /// a dead one — a full bar or an empty one, never an invented middle.
    pub shield: Field<f32>,
    pub health: Field<f32>,
    /// Is the subject scoped? From the event channel's `unit_zoom` (kind 16).
    pub zoomed: Field<bool>,
    /// `(vehicle chud path, seat index)` while the subject is seated.
    pub vehicle: Option<(String, u8)>,
    /// Seconds until the subject respawns, and the whole-second number the widget prints.
    pub respawn: Field<f32>,
    /// Why the respawn countdown reads what it does, for the panel.
    pub respawn_note: String,
    /// #play-hud  The label of the key the viewer has bound to the action button, for a BUTTON
    /// PROMPT — a string that resolves to words with a private-use glyph embedded in them, like
    /// `"Hold \u{E446} to drive Warthog"`.
    ///
    /// **`None` in Theater, and that is not an oversight.** A recording has no button for the
    /// viewer to hold, so drawing the instruction would invite an action that does nothing; the
    /// prompt is reported rather than drawn (see [`paint_collection`]). In test mode there IS a
    /// button, so the glyph is replaced with the player's own bound key and the prompt is drawn —
    /// which is strictly better than the game's own line, because it names *their* key.
    pub button_label: Option<String>,
}

impl HudValues {
    /// The `chud_state_enum` values HMS is prepared to assert this frame. Everything else counts
    /// as FALSE, which keeps a widget HMS cannot evaluate off the screen.
    pub fn known_states(&self) -> Vec<i32> {
        let mut v = vec![STATE_FULLSCREEN_WIDE, STATE_FIRST_PERSON, STATE_BINOCULARS_DISABLED];
        if self.weapon_tag.is_some() {
            v.push(STATE_SINGLE_WIELDING);
            v.push(STATE_WEAPON_IS_RIGHT_HAND);
        } else {
            v.push(STATE_UNARMED);
        }
        match self.zoomed.value {
            Some(true) => v.push(STATE_ZOOMED),
            // Unzoomed is the engine's own resting state and the one every reticule is authored
            // against, so an unknown zoom is treated as unzoomed rather than as neither.
            _ => v.push(STATE_UNZOOMED),
        }
        if self.shield.value.unwrap_or(0.0) > 0.0 {
            v.push(32); // shield available
        }
        v.push(STATE_CROSSHAIR_NORMAL);
        v
    }
    /// One line per field for the panel, naming the provenance of each.
    pub fn report(&self) -> Vec<String> {
        let n = |f: &Field<i32>, what: &str| match f.value {
            Some(v) => format!("{what}: {v} ({})", f.prov.word()),
            None => format!("{what}: {}", f.prov.word()),
        };
        let frac = |f: &Field<f32>, what: &str| match f.value {
            Some(v) => format!("{what}: {:.0}% ({})", v * 100.0, f.prov.word()),
            None => format!("{what}: {}", f.prov.word()),
        };
        let mut out = vec![
            n(&self.loaded, "Magazine"),
            n(&self.reserve, "Reserve"),
            n(&self.frags, "Frag grenades"),
            n(&self.plasmas, "Plasma grenades"),
            frac(&self.shield, "Shield"),
            frac(&self.health, "Health"),
        ];
        out.push(match self.zoomed.value {
            Some(z) => format!("Zoom: {} ({})", if z { "scoped" } else { "unscoped" }, self.zoomed.prov.word()),
            None => format!("Zoom: {}", self.zoomed.prov.word()),
        });
        if let Some((p, seat)) = &self.vehicle {
            out.push(format!("Vehicle HUD: {p} (seat {seat})"));
        }
        if !self.respawn_note.is_empty() {
            out.push(self.respawn_note.clone());
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// #respawn  the countdown
// ---------------------------------------------------------------------------------------------

/// The embedded game variant's respawn options, as much of them as a countdown needs.
///
/// These come out of the film's own `c_game_variant` (`film_variant::RespawnOptions`), so the
/// base time and both penalties are RECORDED, not guessed. What the recording does **not** carry
/// is a per-death countdown value: no event in the 72-type channel is a respawn timer. So the
/// widget counts down from these numbers off the recorded death tick, which is derivation from
/// recorded facts and is labelled as such.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RespawnRules {
    /// `respawn time`, seconds.
    pub base: i8,
    /// `respawn growth`, seconds added per death.
    pub growth: i8,
    /// `suicide penalty`, seconds added when the victim killed themselves.
    pub suicide: i8,
    /// `betrayal penalty`, seconds added when the killer was on the victim's own team.
    pub betrayal: i8,
}

/// Which penalty a death earns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeathKind {
    #[default]
    Normal,
    Suicide,
    Betrayal,
}

impl RespawnRules {
    /// How long this death keeps the player out, seconds.
    ///
    /// `deaths_before` is how many times they had already died this round, which is what
    /// `respawn growth` multiplies. The two penalties are additive on top of the base, which is
    /// how the variant presents them (`respawn 3 s (+0 growth, suicide +5, betrayal +5)`).
    pub fn seconds(&self, deaths_before: u32, kind: DeathKind) -> f32 {
        let mut s = self.base as f32 + self.growth as f32 * deaths_before as f32;
        s += match kind {
            DeathKind::Normal => 0.0,
            DeathKind::Suicide => self.suicide as f32,
            DeathKind::Betrayal => self.betrayal as f32,
        };
        s.max(0.0)
    }
    /// Seconds LEFT on the clock at `now`, given a death at `died_at` (both in seconds), or
    /// `None` once the countdown has run out.
    pub fn remaining(&self, died_at: f32, now: f32, deaths_before: u32, kind: DeathKind) -> Option<f32> {
        let total = self.seconds(deaths_before, kind);
        let left = total - (now - died_at);
        (left > 0.0 && now >= died_at).then_some(left)
    }
}

/// The countdown for one dead body, and where each end of it came from.
///
/// **The better clock is the recording's own.** A player gets a brand-new network entity every
/// time they respawn, so the NEXT life's first tick is its creation-message tick — exact to one
/// tick. The uncertainty is entirely at the other end: the death instant is the body's last
/// reported position (±~0.3 s, measured IQR 39 ticks over twelve films), or the stated `incident`
/// tick where the kill feed reached this recording at all.
///
/// So the widget prefers `next_life` and falls back to the variant's own respawn arithmetic when
/// the recording has no next life to point at — the last death of a round, or a body whose owner
/// never came back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RespawnClock {
    /// `(death seconds, respawn seconds)` — both ends off the recording.
    NextLife { died_at: f32, respawn_at: f32 },
    /// The variant's own numbers, counted off the death.
    FromRules { died_at: f32, rules: RespawnRules, deaths_before: u32, kind: DeathKind },
}

impl RespawnClock {
    /// Seconds still to wait at `now`, or `None` when the countdown is not running.
    pub fn remaining(&self, now: f32) -> Option<f32> {
        match self {
            RespawnClock::NextLife { died_at, respawn_at } => {
                let left = respawn_at - now;
                (left > 0.0 && now >= *died_at).then_some(left)
            }
            RespawnClock::FromRules { died_at, rules, deaths_before, kind } => {
                rules.remaining(*died_at, now, *deaths_before, *kind)
            }
        }
    }
    /// One line saying where this countdown comes from — the panel prints it rather than letting
    /// a number stand on its own.
    pub fn note(&self) -> String {
        match self {
            RespawnClock::NextLife { died_at, respawn_at } => format!(
                "Respawn: counted to the recording's own next life for this player ({:.1} s after the death at {:.1} s)",
                respawn_at - died_at,
                died_at
            ),
            RespawnClock::FromRules { rules, deaths_before, kind, .. } => format!(
                "Respawn: the recording has no next life for this player, so the countdown is the game variant's own \
                 {} s (+{} growth x {deaths_before} deaths, {kind:?}) = {:.0} s",
                rules.base,
                rules.growth,
                rules.seconds(*deaths_before, *kind)
            ),
        }
    }
    /// The whole number the widget prints: the engine shows a countdown as the number of whole
    /// seconds still to wait, so 2.3 s left reads "3".
    pub fn displayed(left: f32) -> i32 {
        left.ceil().max(0.0) as i32
    }
}

// ---------------------------------------------------------------------------------------------
// painting
// ---------------------------------------------------------------------------------------------

/// A sprite uploaded for one widget.
pub struct WidgetTex {
    pub tex: egui::TextureHandle,
    pub uv: egui::Rect,
}

/// One `chdt` with every bitmap it draws uploaded, cached per (cache, tag).
pub struct HudArt {
    pub chud: HudChud,
    /// Parallel to `chud.collections[i].widgets[j]`, keyed `(i, j)`; a text widget has no entry.
    pub tex: std::collections::HashMap<(usize, usize), WidgetTex>,
    /// #reach-strings  The cache's localized text, so an authored `string_id` can become the
    /// words the game prints. Loaded once per cache beside the art, because that is the lifetime
    /// it shares: one `tags_by_group("unic")` plus a name each.
    pub strings: crate::reach_strings::ReachStrings,
}

thread_local! {
    static HUD_ART: std::cell::RefCell<std::collections::HashMap<(u64, u32), Option<std::rc::Rc<HudArt>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Decode and upload one `chdt`, once per open cache.
pub fn hud_art(
    ctx: &egui::Context,
    native: &NativeDll,
    cache: CacheHandle,
    tag: u32,
) -> Option<std::rc::Rc<HudArt>> {
    HUD_ART.with(|m| {
        if let Some(v) = m.borrow().get(&(cache.0, tag)) {
            return v.clone();
        }
        let made = (|| {
            let chud = read_chud(native, cache, tag)?;
            let mut tex = std::collections::HashMap::new();
            for (ci, c) in chud.collections.iter().enumerate() {
                for (wi, w) in c.widgets.iter().enumerate() {
                    let WidgetArt::Bitmap { sprite, .. } = &w.art else { continue };
                    let Some(t) = upload_sprite(ctx, native, cache, sprite, &w.base) else { continue };
                    tex.insert((ci, wi), t);
                }
            }
            let strings = crate::reach_strings::ReachStrings::load(native, cache);
            Some(std::rc::Rc::new(HudArt { chud, tex, strings }))
        })();
        m.borrow_mut().insert((cache.0, tag), made.clone());
        made
    })
}

/// Upload one sprite, honouring the authored scale's sign as a UV flip — the same rule the
/// reticule already uses.
fn upload_sprite(
    ctx: &egui::Context,
    native: &NativeDll,
    cache: CacheHandle,
    sp: &NavpointSprite,
    base: &WidgetBase,
) -> Option<WidgetTex> {
    let (bgra, w, h) = native.decode_bitmap_keep_alpha(cache, sp.bitmap_tag, sp.bitmap_index as u32, 0)?;
    if w == 0 || h == 0 || bgra.len() < (w as usize * h as usize * 4) {
        return None;
    }
    let px: Vec<egui::Color32> = bgra
        .chunks_exact(4)
        .take(w as usize * h as usize)
        .map(|c| egui::Color32::from_rgba_unmultiplied(c[2], c[1], c[0], c[3]))
        .collect();
    let img = egui::ColorImage { size: [w as usize, h as usize], pixels: px };
    let tex = ctx.load_texture(format!("reach-hud-{}-{}", sp.bitmap_tag, sp.bitmap_index), img, egui::TextureOptions::LINEAR);
    let s = base.placement().map(|p| p.scale).unwrap_or([1.0, 1.0]);
    let (u0, u1) = if s[0] < 0.0 { (sp.uv[1], sp.uv[0]) } else { (sp.uv[0], sp.uv[1]) };
    let (v0, v1) = if s[1] < 0.0 { (sp.uv[3], sp.uv[2]) } else { (sp.uv[2], sp.uv[3]) };
    Some(WidgetTex { tex, uv: egui::Rect::from_min_max(egui::pos2(u0, v0), egui::pos2(u1, v1)) })
}

/// The screen colour a widget's `color output A` resolves to.
/// Is this `real argb color` authored at all, or is it an untouched slot?
///
/// An all-zero entry is not "black": it is a colour the artist never set. The shield meter's
/// border authors exactly that in `custom color A` and its real colour in `custom color B`, and
/// taking A literally paints the border solid black.
fn authored(c: [f32; 4]) -> bool {
    c.iter().any(|v| *v > 0.0)
}

/// The screen colour a widget's `color output A` resolves to.
///
/// `chud_render_color_output_enum`: 0..3 are `custom color A`..`D`, 4 and 5 are colour
/// ANIMATIONS, and 18 and up are the skin palette (`COLOR_OUTPUT_BASE`). Three things were wrong
/// here and all three showed on screen:
///
/// * **every non-skin output was read as custom A**, so a widget that authors its colour in B, C
///   or D drew in whatever happened to be in A;
/// * **an unauthored custom A was drawn as opaque black**, which is what the shield and health
///   meter borders did;
/// * a colour ANIMATION output has no resting colour at all, so there is nothing to read from it.
///
/// For the last two the fallback is the widget's own palette: the first custom colour it actually
/// authors. That is an approximation of a multi-output shader HMS does not run, and it is stated
/// as one — but it is drawn from the widget's own numbers rather than invented, and it cannot
/// produce a colour the artist never put on that widget.
pub fn widget_color(base: &WidgetBase, skin: Option<&ChudSkin>) -> egui::Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let rgb = |c: [f32; 4]| egui::Color32::from_rgb(q(c[1]), q(c[2]), q(c[3]));
    // The authored alpha of a SKIN colour is not the widget's opacity (several are authored
    // a = 0.0 and are plainly visible in game), so a skin colour is taken opaque — the same
    // reading `film_ui::NamePlateRules` documents for the navpoint palette.
    if let (Some(k), Some(s)) = (base.color, skin) {
        return rgb(s.colors[k]);
    }
    // A custom output names one of the four, and it is used only when the artist set it.
    if (0..4).contains(&base.output_a) {
        let c = base.custom[base.output_a as usize];
        if authored(c) {
            return rgb(c);
        }
    }
    // Unauthored, or an animation output: the first colour this widget does author.
    match base.custom.iter().find(|c| authored(**c)) {
        Some(c) => rgb(*c),
        // Nothing authored anywhere. White is the neutral that leaves a sprite's own art alone,
        // which is what a HUD bitmap is tinted against.
        None => egui::Color32::WHITE,
    }
}

/// Where one widget's point lands, in screen points, and the scale its art is drawn at.
///
/// `parent` is the collection's own resolved point for a widget whose anchor is
/// `chud_anchor_type_enum` 0 `parent`, which is what every child of an authored block uses.
pub fn widget_point(vs: &VirtualScreen, p: &Placement, parent: Option<egui::Pos2>) -> Option<(egui::Pos2, [f32; 2])> {
    let a = if p.anchor == ANCHOR_PARENT { parent? } else { vs.anchor(p.anchor)? };
    Some((egui::pos2(a.x + p.offset[0] * vs.vpx, a.y + p.offset[1] * vs.vpx), p.scale))
}

/// The number a `chud_out_a` text widget prints, formatted the way its own
/// `chud_widget_text_flags` say to.
pub fn format_number(v: i32, flags: u16) -> String {
    // bit 1 `force 2-digit number`, bit 2 `force 3-digit number`, bit 12 `format as time`.
    if flags & TEXT_FORMAT_AS_TIME != 0 {
        let (m, s) = (v / 60, (v % 60).abs());
        return format!("{m}:{s:02}");
    }
    if flags & (1 << 2) != 0 {
        return format!("{v:03}");
    }
    if flags & (1 << 1) != 0 {
        return format!("{v:02}");
    }
    v.to_string()
}

/// Draw one text widget: `text` at its placement, pivoted by `widget origin`.
///
/// The baseline is placed so the box's TOP is `origin.y` of the line height above the point, i.e.
/// `origin = (0.5, 0.5)` centres the box on the point — which is what a centred ammo number
/// needs — and `(1.0, 0.5)` right-aligns it.
#[allow(clippy::too_many_arguments)]
pub fn paint_text(
    ctx: &egui::Context,
    painter: &egui::Painter,
    at: egui::Pos2,
    scale: [f32; 2],
    origin: [f32; 2],
    font: Option<crate::reach_font::FontId>,
    text: &str,
    color: egui::Color32,
    vpx: f32,
) -> Option<egui::Rect> {
    if text.is_empty() {
        return None;
    }
    let s = vpx * scale[0].abs().max(0.01);
    let atlas = font.and_then(|f| crate::reach_font::atlas_for(ctx, f));
    match atlas {
        Some(a) => {
            let w = a.font.measure(text) * s;
            let lh = a.font.line_height() as f32 * s;
            let x = at.x - w * origin[0];
            // `dy` of a glyph is measured up from the baseline, so the baseline sits
            // `ascender * s` below the box's top.
            let top = at.y - lh * origin[1];
            a.paint(painter, egui::pos2(x, top + a.font.ascender as f32 * s), text, color, s);
            Some(egui::Rect::from_min_size(egui::pos2(x, top), egui::vec2(w, lh)))
        }
        None => {
            // No Reach fonts on this machine: same place, same SIZE (the authored line height in
            // virtual units), the app's own glyphs.
            let size = crate::reach_chud::HUD_FONT_LINE_HEIGHT_VU * s;
            let g = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(size), color);
            let p = egui::pos2(at.x - g.size().x * origin[0], at.y - g.size().y * origin[1]);
            let sz = g.size();
            painter.galley(p, g, color);
            Some(egui::Rect::from_min_size(p, sz))
        }
    }
}

/// Draw one bitmap widget, centred on its point and sized by the sprite times the authored scale.
///
/// `fill` is the horizontal fill fraction: 1.0 draws the whole sprite, less clips it from the
/// LEFT, which is how a chud meter reads a level.
#[allow(clippy::too_many_arguments)]
pub fn paint_bitmap(
    painter: &egui::Painter,
    at: egui::Pos2,
    scale: [f32; 2],
    sprite: &NavpointSprite,
    tex: &WidgetTex,
    color: egui::Color32,
    vpx: f32,
    fill: f32,
    blend: i32,
) -> Option<egui::Rect> {
    // #hud-shader  The authored blend mode, reproduced through egui's own pipeline. A mode that
    // cannot be expressed draws NOTHING rather than drawing in the wrong one.
    let color = blend_tint(color, blend)?;
    let w = (sprite.size[0] * scale[0]).abs() * vpx;
    let h = (sprite.size[1] * scale[1]).abs() * vpx;
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let f = fill.clamp(0.0, 1.0);
    if f <= 0.0 {
        return None;
    }
    let full = egui::Rect::from_center_size(at, egui::vec2(w, h));
    let rect = egui::Rect::from_min_size(full.min, egui::vec2(w * f, h));
    let uv = egui::Rect::from_min_max(
        tex.uv.min,
        egui::pos2(tex.uv.min.x + (tex.uv.max.x - tex.uv.min.x) * f, tex.uv.max.y),
    );
    let mut mesh = egui::Mesh::with_texture(tex.tex.id());
    mesh.add_rect_with_uv(rect, uv, color);
    painter.add(egui::Shape::mesh(mesh));
    Some(rect)
}

/// What a collection's `chud_out_a` number and its meter fill should read, for one HUD frame.
///
/// Returned per COLLECTION name, because that is how the tags organise it: `ammo_area`'s
/// `reserve_count` is the reserve and its `meter` is the magazine, `backpack_ammo_area`'s
/// `ammo_count` is the stowed weapon's, `grenades`' two repeats are the two types.
pub struct CollectionValues<'a> {
    pub v: &'a HudValues,
}

impl CollectionValues<'_> {
    /// The number for a `chud_out_a` widget of `collection`/`widget`, or `None` to draw nothing.
    ///
    /// The collection names are not uniform across the shipped weapons — the assault rifle
    /// authors `ammo_area`, the magnum `ammo_area_right`, and a stowed weapon's block is always
    /// `backpack_ammo_area` — so the ammo blocks are matched by PREFIX. Inside one, `ammo_count`
    /// is the magazine and `reserve_count` the reserve.
    pub fn number(&self, collection: &str, widget: &str, repeat: usize) -> Option<i32> {
        if collection.starts_with("ammo_area") {
            return match widget {
                "ammo_count" => self.v.loaded.value,
                "reserve_count" | "reserve_count_kudos" => self.v.reserve.value,
                _ => None,
            };
        }
        // A stowed weapon's ammo is the same never-sent field, and HMS does not even know WHICH
        // weapon is stowed (the derived block's slot-1..3 references never arrive either).
        if collection == "backpack_ammo_area" {
            return None;
        }
        match (collection, widget) {
            ("grenades", "grenade_count_hilite") | ("grenades", "grenade_count_backpack") => {
                if repeat == 0 { self.v.frags.value } else { self.v.plasmas.value }
            }
            _ => None,
        }
    }
    /// Is this bitmap a METER, and if so how full is it?
    ///
    /// `None` = not a meter, draw the whole sprite. `Some(None)` = a meter whose value the
    /// recording does not give, which is **not drawn at all** — a meter left at full would be a
    /// claim, and an empty one would be a different claim. `Some(Some(f))` = draw `f` of it.
    pub fn meter(&self, collection: &str, widget: &str) -> Option<Option<f32>> {
        self.meter_for(collection, widget, None)
    }

    /// #hud-shader  As [`CollectionValues::meter`], with the widget's own authored `shader type`.
    ///
    /// **`shader` is what decides whether this is a meter at all.** It used to be inferred from
    /// the NAME — `w.starts_with("meter") && !w.starts_with("meter_border")` — and a name rule is
    /// the shape of thing this project has been bitten by: a shield NAME heuristic once drew a
    /// solid barricade see-through. The tag states it outright, and seven of the twenty-one chud
    /// shaders are meters ([`METER_SHADERS`]).
    ///
    /// The two readings AGREE on the shipped player HUD, which is the cross-check worth keeping:
    /// `shield_meter/meter` is `shader 4 'meter shield'`, its three `meter_sup*` siblings are the
    /// same shader (the overshield layers), `health_meter/health_meter` is `shader 2 'meter'`, and
    /// `meter_border` — which the name rule had to special-case — is `shader 1 'anti-alias'`, not
    /// a meter at all. The name rule was right here and is still the wrong instrument.
    ///
    /// `shader` of `None` keeps the old name-based reading, so a caller that has not got the field
    /// is no worse off than before.
    pub fn meter_for(&self, collection: &str, widget: &str, shader: Option<i32>) -> Option<Option<f32>> {
        let is_meter = match shader {
            Some(sh) => shader_is_meter(sh),
            None => {
                (collection.starts_with("ammo_area") && widget == "meter")
                    || (collection == "shield_meter"
                        && widget.starts_with("meter")
                        && !widget.starts_with("meter_border"))
                    || (collection == "health_meter" && widget == "health_meter")
            }
        };
        if !is_meter {
            return None;
        }
        // WHICH meter it is still comes from the collection, because that is what the tags
        // organise by: an `ammo_area*` meter is the magazine, the shield block's is the shield.
        if collection.starts_with("ammo_area") {
            return Some(match (self.v.loaded.value, self.v.loaded_max) {
                (Some(l), Some(m)) if m > 0 => Some(l as f32 / m as f32),
                _ => None,
            });
        }
        match collection {
            "shield_meter" => Some(self.v.shield.value),
            "health_meter" => Some(self.v.health.value),
            // A meter by its shader, in a collection HMS has no value for.
            _ => Some(None),
        }
    }
    /// The 0..1 fill a meter should draw at, for the gates to read.
    pub fn fill(&self, collection: &str, widget: &str) -> Option<f32> {
        self.meter(collection, widget).flatten()
    }
}

// ---------------------------------------------------------------------------------------------
// #hud-trace  what the painter ACTUALLY did, per widget
// ---------------------------------------------------------------------------------------------

/// One widget the painter considered, and what it did with it.
///
/// **This exists because the chud overlay cannot be captured headlessly.** A headless `shot`
/// renders the 3D scene only — measured: three captures at one tick with the HUD toggled came back
/// byte-identical — so "the shields and health don't look right" cannot be turned into a defect by
/// looking. It can be turned into one by reading what the painter emitted: which sprite each
/// widget resolved to, the rectangle it painted at, whether it was treated as a meter, and what
/// colour came out. Every one of those is a number, and a wrong one is visible in a table.
#[derive(Clone, Debug)]
pub struct WidgetDraw {
    pub collection: String,
    pub widget: String,
    /// Which repeat of a datasource-driven collection this was.
    pub repeat: usize,
    /// `"bitmap"` or `"text"`.
    pub kind: &'static str,
    /// `(bitm tag, submap index, size in VIRTUAL UNITS)` for a bitmap.
    pub sprite: Option<(u32, u16, [f32; 2])>,
    /// #hud-art  `(authored sequence index, which decode route answered)` for a bitmap.
    pub art_route: Option<(i8, ArtRoute)>,
    /// #hud-shader  `shader type` and `alpha blend mode`, as authored.
    pub shader_type: i32,
    pub blend_mode: i32,
    /// The string a text widget drew, or the reason it drew none.
    pub text: Option<String>,
    /// `chud_anchor_type_enum` of this widget's own placement.
    pub anchor: u8,
    /// The authored `origin offset`, virtual units.
    pub offset: [f32; 2],
    pub origin: [f32; 2],
    pub scale: [f32; 2],
    /// Where it landed and how big, in screen points. `None` when it never got that far.
    pub rect: Option<egui::Rect>,
    /// `None` = not a meter. `Some(None)` = a meter with no level, not drawn. `Some(Some(f))` = fill.
    pub meter: Option<Option<f32>>,
    pub color: Option<egui::Color32>,
    /// `"drawn"`, or why not.
    pub outcome: String,
}

impl WidgetDraw {
    /// One line, for the panel and for the headless dump.
    pub fn line(&self, vs: Option<&VirtualScreen>) -> String {
        let where_ = match (self.rect, vs) {
            (Some(r), Some(vs)) => format!(
                "at ({:.0},{:.0}) {:.0}x{:.0} pt = ({:.2},{:.2}) of the viewport",
                r.center().x,
                r.center().y,
                r.width(),
                r.height(),
                (r.center().x - vs.viewport.min.x) / vs.viewport.width().max(1.0),
                (r.center().y - vs.viewport.min.y) / vs.viewport.height().max(1.0),
            ),
            (Some(r), None) => format!("at ({:.0},{:.0}) {:.0}x{:.0} pt", r.center().x, r.center().y, r.width(), r.height()),
            _ => "no rect".into(),
        };
        let art = match (&self.sprite, &self.text) {
            (Some((t, i, sz)), _) => format!(
                "bitm {t:#x} submap {i} {:.0}x{:.0} vu  seq {} via {}",
                sz[0],
                sz[1],
                self.art_route.map(|(q, _)| q.to_string()).unwrap_or_else(|| "?".into()),
                self.art_route.map(|(_, r)| r.word()).unwrap_or("?")
            ),
            (_, Some(txt)) => format!("text {txt:?}"),
            _ => "no art".into(),
        };
        let meter = match self.meter {
            None => "not a meter".to_string(),
            Some(None) => "METER, no level -> not drawn".to_string(),
            Some(Some(f)) => format!("METER fill {:.3}", f),
        };
        let col = match self.color {
            Some(c) => format!("#{:02X}{:02X}{:02X} a{:02X}", c.r(), c.g(), c.b(), c.a()),
            None => "-".into(),
        };
        format!(
            "{}/{}{} {} | shader {} '{}'{} | blend {} '{}'{} | {art} | anchor {} off ({:+.1},{:+.1}) origin ({:.2},{:.2}) scale ({:+.2},{:+.2}) | {where_} | {meter} | {col} | {}",
            self.collection,
            self.widget,
            if self.repeat > 0 { format!("#{}", self.repeat) } else { String::new() },
            self.kind,
            self.shader_type,
            shader_type_name(self.shader_type),
            if shader_is_meter(self.shader_type) { " METER" } else { "" },
            self.blend_mode,
            blend_mode_name(self.blend_mode),
            if blend_is_reproducible(self.blend_mode) { "" } else { "  <-- HMS CANNOT REPRODUCE THIS BLEND" },
            self.anchor,
            self.offset[0],
            self.offset[1],
            self.origin[0],
            self.origin[1],
            self.scale[0],
            self.scale[1],
            self.outcome
        )
    }
}

/// #hud-trace  Which drawn widgets a LATER drawn widget buries, as `(index, the widget that
/// covers it)`.
///
/// **"Drawn" is not "visible", and the difference cost a round trip with the user.** The shield
/// bar painted correctly at (960,65) 249x57 pt and then `meter_border` painted an opaque plate over
/// exactly the same rectangle; the draw count said 6 and the bar was not on screen. A trace that
/// only counts draws cannot say that, so this says it.
///
/// The test is deliberately conservative — a widget counts as covered only when a strictly later
/// one, in a blend that OBSCURES (`opaque`, `alpha blend`, `pre multiplied alpha`, `multiply`),
/// contains its rectangle. An `additive` or `maximum` widget never obscures, which is the whole
/// point of the shield border being additive, so one is never reported as covering anything.
pub fn occlusions(log: &[WidgetDraw]) -> Vec<(usize, String)> {
    /// The blend modes in which a drawn quad can hide what is under it.
    fn obscures(mode: i32) -> bool {
        matches!(mode, -1 | 0 | 2 | 3 | 5)
    }
    let mut out = Vec::new();
    for (i, a) in log.iter().enumerate() {
        if a.outcome != "drawn" {
            continue;
        }
        let Some(ra) = a.rect else { continue };
        if ra.width() <= 0.0 || ra.height() <= 0.0 {
            continue;
        }
        for b in log.iter().skip(i + 1) {
            if b.outcome != "drawn" || !obscures(b.blend_mode) {
                continue;
            }
            let Some(rb) = b.rect else { continue };
            // `contains_rect` with a hair of slack: the two rects are computed from the same
            // authored numbers, so an exact cover differs only by float noise.
            let pad = egui::Rect::from_min_max(
                rb.min - egui::vec2(0.5, 0.5),
                rb.max + egui::vec2(0.5, 0.5),
            );
            if pad.contains_rect(ra) {
                out.push((i, b.widget.clone()));
                break;
            }
        }
    }
    out
}

thread_local! {
    static WIDGET_LOG: std::cell::RefCell<Vec<WidgetDraw>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Off by default, so an ordinary frame costs nothing and THEATER IS UNAFFECTED.
    static WIDGET_LOG_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Turn the per-widget trace on or off. Off is the default and the only state Theater ever uses.
pub fn set_widget_log(on: bool) {
    WIDGET_LOG_ON.with(|f| f.set(on));
    if !on {
        WIDGET_LOG.with(|l| l.borrow_mut().clear());
    }
}

/// Is the trace recording?
pub fn widget_log_on() -> bool {
    WIDGET_LOG_ON.with(|f| f.get())
}

/// Take everything the last traced frame recorded, and clear it.
pub fn take_widget_log() -> Vec<WidgetDraw> {
    WIDGET_LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

fn log_widget(d: WidgetDraw) {
    if WIDGET_LOG_ON.with(|f| f.get()) {
        WIDGET_LOG.with(|l| l.borrow_mut().push(d));
    }
}

/// Paint one authored collection.
///
/// Returns how many widgets were drawn and how many were skipped because their state could not
/// be evaluated — the panel reports both rather than claiming a complete HUD.
///
/// #hud-trace  With [`set_widget_log`] on, every widget considered is recorded with the sprite it
/// resolved to, the rect it painted at and whether it was treated as a meter. Off by default.
#[allow(clippy::too_many_arguments)]
pub fn paint_collection(
    ctx: &egui::Context,
    painter: &egui::Painter,
    vs: &VirtualScreen,
    art: &HudArt,
    ci: usize,
    skin: Option<&ChudSkin>,
    values: &HudValues,
    repeat: usize,
) -> (usize, usize) {
    let Some(c) = art.chud.collections.get(ci) else { return (0, 0) };
    // The frame's truth set, plus the states that belong to THIS repeat of a datasource-driven
    // collection (`<grenade> selected` and friends).
    let mut known = values.known_states();
    known.extend(repeat_states(c.datasource, repeat));
    if !c.base.on(&known) {
        return (0, 0);
    }
    let Some(cp) = c.base.placement() else { return (0, 0) };
    let Some((mut parent, _)) = widget_point(vs, cp, None) else { return (0, 0) };
    // The datasource lays repeats out "as offsets from the previous one".
    for k in 0..repeat.min(c.positions.len()) {
        parent.x += c.positions[k].0[0] * vs.vpx;
        parent.y += c.positions[k].0[1] * vs.vpx;
    }
    let cv = CollectionValues { v: values };
    let (mut drawn, mut skipped) = (0, 0);
    // String ids this collection could not resolve to text, for the panel to name.
    let mut unresolved_strings: Vec<String> = Vec::new();
    for (wi, w) in c.widgets.iter().enumerate() {
        // #hud-trace  One record per widget considered, filled in as the decisions are taken and
        // pushed at whichever exit this widget takes. Recording only; nothing below branches on it.
        let mut tr = WidgetDraw {
            collection: c.base.name.clone(),
            widget: w.base.name.clone(),
            repeat,
            kind: match &w.art {
                WidgetArt::Bitmap { .. } => "bitmap",
                WidgetArt::Text { .. } => "text",
            },
            sprite: match &w.art {
                WidgetArt::Bitmap { sprite, .. } => {
                    Some((sprite.bitmap_tag, sprite.bitmap_index, sprite.size))
                }
                _ => None,
            },
            art_route: match &w.art {
                WidgetArt::Bitmap { sequence, route, .. } => Some((*sequence, *route)),
                _ => None,
            },
            text: match &w.art {
                WidgetArt::Text { input, .. } => Some(format!("<{input}>")),
                _ => None,
            },
            shader_type: w.base.shader_type,
            blend_mode: w.base.blend_mode,
            anchor: w.base.placement().map(|p| p.anchor).unwrap_or(255),
            offset: w.base.placement().map(|p| p.offset).unwrap_or([0.0, 0.0]),
            origin: w.base.placement().map(|p| p.origin).unwrap_or([0.0, 0.0]),
            scale: w.base.placement().map(|p| p.scale).unwrap_or([1.0, 1.0]),
            rect: None,
            meter: None,
            color: None,
            outcome: String::new(),
        };
        if !w.base.on(&known) {
            skipped += 1;
            tr.outcome = format!(
                "NOT DRAWN: its Active/Hidden state is not satisfied by what HMS can assert.                  Mentions {:?}",
                w.base.mentions()
            );
            log_widget(tr);
            continue;
        }
        // The player HUD's EFFECT overlays -- the damage flashes, the boot blob, the compass --
        // are authored unconditional and HMS cannot drive any of them. See `PLAYER_WIDGETS`.
        if !is_drawn_player_widget(&c.base.name, &w.base.name) {
            skipped += 1;
            tr.outcome = "NOT DRAWN: an effect overlay HMS cannot drive (PLAYER_WIDGETS)".into();
            log_widget(tr);
            continue;
        }
        // A widget that authors NO colour anywhere -- no skin output and not one of its four
        // custom slots set -- is one HMS cannot colour at all; painting it WHITE put two white
        // plates over the health bar. Drawing nothing is the honest answer.
        // #hud-shader  ...**unless its SHADER supplies its own colour.** A meter is composited by
        // its shader out of its own art, so leaving `custom color A` unauthored is the normal case
        // for one, not a gap: measured over a shipped cache, **158 of the 294 widgets whose
        // `color output A` is 0 author no colour at all**. Refusing those took the assault rifle's
        // whole magazine BAR off the screen — and that chud authors no `ammo_count` widget, so the
        // bar is the only magazine display there is. A meter is therefore drawn untinted, which
        // leaves its own pixels alone and is the smallest claim available.
        if w.base.color.is_none()
            && !w.base.custom.iter().any(|c| c.iter().any(|v| *v > 0.0))
            && !w.base.is_meter_shader()
        {
            skipped += 1;
            tr.outcome = format!(
                "NOT DRAWN: authors no colour anywhere (color output A = {}, shader {} '{}')",
                w.base.output_a,
                w.base.shader_type,
                shader_type_name(w.base.shader_type)
            );
            log_widget(tr);
            continue;
        }
        // A widget whose `scalar output A` is a flash or scalar ANIMATION is multiplied by a
        // value that is zero at rest, so the engine shows nothing until the flash is triggered.
        // HMS runs no chud animations, and drawing these at full strength paints the HUD with
        // damage flashes that are not happening — the shield's `flahsy` is a solid red blob
        // authored `crosshair enemy`, and every `*_flash` layer is the same.
        if w.base.animation_driven() {
            skipped += 1;
            tr.outcome = format!("NOT DRAWN: scalar output A {} is an animation, zero at rest", w.base.scalar_a);
            log_widget(tr);
            continue;
        }
        let Some(p) = w.base.placement() else {
            tr.outcome = "NOT DRAWN: authors no placement".into();
            log_widget(tr);
            continue;
        };
        let Some((at, scale)) = widget_point(vs, p, Some(parent)) else {
            skipped += 1;
            tr.outcome = format!("NOT DRAWN: anchor {} is a runtime anchor HMS cannot resolve", p.anchor);
            log_widget(tr);
            continue;
        };
        let col = widget_color(&w.base, skin);
        tr.color = Some(col);
        match &w.art {
            WidgetArt::Bitmap { sprite, .. } => {
                let m = cv.meter_for(&c.base.name, &w.base.name, Some(w.base.shader_type));
                tr.meter = m;
                let Some(tex) = art.tex.get(&(ci, wi)) else {
                    tr.outcome = "NOT DRAWN: its bitmap did not upload".into();
                    log_widget(tr);
                    continue;
                };
                let fill = match m {
                    // A meter with no recorded level: drawn neither full nor empty.
                    Some(None) => {
                        skipped += 1;
                        tr.outcome = "NOT DRAWN: a meter with no level".into();
                        log_widget(tr);
                        continue;
                    }
                    Some(Some(f)) => f,
                    None => 1.0,
                };
                let r = paint_bitmap(painter, at, scale, sprite, tex, col, vs.vpx, fill, w.base.blend_mode);
                tr.rect = r;
                tr.outcome = if r.is_some() {
                    "drawn".into()
                } else if !w.base.blend_reproducible() {
                    format!(
                        "NOT DRAWN: blend mode {} '{}' cannot be reproduced",
                        w.base.blend_mode,
                        blend_mode_name(w.base.blend_mode)
                    )
                } else {
                    "NOT DRAWN: degenerate rect".to_string()
                };
                log_widget(tr);
                drawn += 1;
            }
            WidgetArt::Text { font, input, flags } => {
                let text = if input == CHUD_OUT_A || flags & TEXT_IS_NUMBER != 0 {
                    match cv.number(&c.base.name, &w.base.name, repeat) {
                        Some(n) => format_number(n, *flags),
                        None => {
                            skipped += 1;
                            tr.outcome = "NOT DRAWN: a runtime number HMS has no value for".into();
                            log_widget(tr);
                            continue;
                        }
                    }
                } else {
                    // #reach-strings  A LITERAL string id. HMS can now resolve it to the words
                    // the game prints, and there are THREE outcomes, not two.
                    //
                    // `input string` is a `string_id` the engine looks up in a localized table.
                    // The key MCC uses is `<unic tag path, backslashes, lowercased>_<string id>`,
                    // uppercased then CRC-32'd — recovered and proven against 36 of 36 authored
                    // vehicle-seat prompts. Before that was known, EVERY one of these was skipped
                    // and reported, which is why "assault_rifle" once appeared as words across
                    // the ammo block.
                    //
                    // The three outcomes, and why the last two still draw nothing:
                    //
                    // 1. ORDINARY TEXT -> drawn.
                    // 2. ICON-ONLY: the text is nothing but private-use codepoints, i.e. it IS a
                    //    glyph rather than words. The weapon `shematic` is this: its one codepoint
                    //    maps into MCC's `font_package_icon*.bin`, which HMS does not have, so
                    //    there is no text underneath to draw. Substituting a label here would
                    //    print "[?]" where the weapon's outline belongs.
                    // 3. A BUTTON PROMPT: words with a button glyph embedded ("Hold <glyph> to
                    //    drive Warthog"). Correct for an interactive mode, WRONG here — Theater
                    //    is a recording, there is no button for the viewer to hold, and drawing
                    //    the instruction would invite an action that does nothing. Test mode draws
                    //    these with the player's own bound key; a film does not draw them at all.
                    //
                    // Each case is reported with its reason rather than lumped together, because
                    // "no reader" and "it is an icon" are different facts about the same widget.
                    match art.strings.resolve(input, Some("ui\\hud\\hud_messages")) {
                        None => {
                            skipped += 1;
                            tr.outcome = "NOT DRAWN: no localized text under this string id".into();
                            log_widget(tr);
                            unresolved_strings
                                .push(format!("{input} (no localized text under this id)"));
                            continue;
                        }
                        Some(r) => {
                            // Derived from `substitute_glyphs` rather than re-implemented: drop
                            // every private-use codepoint and see whether any words remain.
                            let (bare, glyphs) = crate::reach_strings::substitute_glyphs(&r.text, "");
                            if bare.trim().is_empty() {
                                skipped += 1;
                                tr.outcome = format!(
                                    "NOT DRAWN: an ICON, not text ({} glyph(s) in MCC's icon font)",
                                    glyphs.len()
                                );
                                log_widget(tr);
                                unresolved_strings.push(format!(
                                    "{input} (an ICON, not text: {} glyph(s) in MCC's icon font)",
                                    glyphs.len()
                                ));
                                continue;
                            }
                            if !glyphs.is_empty() {
                                // #play-hud  A BUTTON PROMPT. Drawn only when the caller has a
                                // real button to name: `button_label` is `Some` in test mode and
                                // `None` over a recording, which is the whole distinction.
                                match &values.button_label {
                                    Some(key) => {
                                        crate::reach_strings::substitute_glyphs(&r.text, key).0
                                    }
                                    None => {
                                        skipped += 1;
                                        tr.outcome = "NOT DRAWN: a button prompt, and this caller named no key".into();
                                        log_widget(tr);
                                        unresolved_strings.push(format!(
                                            "{input} (a button prompt: {:?} — not drawn over a recording)",
                                            r.text.trim()
                                        ));
                                        continue;
                                    }
                                }
                            } else {
                                r.text
                            }
                        }
                    }
                };
                let r = paint_text(ctx, painter, at, scale, p.origin, *font, &text, col, vs.vpx);
                tr.text = Some(text.clone());
                tr.rect = r;
                tr.outcome = "drawn".into();
                log_widget(tr);
                drawn += 1;
            }
        }
    }
    if !unresolved_strings.is_empty() {
        UNRESOLVED.with(|u| u.borrow_mut().extend(unresolved_strings));
    }
    (drawn, skipped)
}

thread_local! {
    /// Every authored string id the last painted frame could not turn into text, so the
    /// panel can say WHICH pieces are missing rather than leaving silent gaps.
    static UNRESOLVED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Take the string ids the last frame could not draw, and clear the list.
pub fn take_unresolved_strings() -> Vec<String> {
    UNRESOLVED.with(|u| {
        let mut v = u.borrow_mut();
        let mut out = std::mem::take(&mut *v);
        out.sort();
        out.dedup();
        out
    })
}

// ---------------------------------------------------------------------------------------------
// #respawn  drawing the countdown
// ---------------------------------------------------------------------------------------------

/// The `chdt` that carries Reach's multiplayer status slots.
pub const MP_CHUD: &str = "ui\\chud\\scoreboard";
/// The slot the countdown is drawn in: `spawning_on_you`.
///
/// **Chosen from the tag, not by eye, and the first choice was wrong.** There is no chud widget
/// named for a respawn, so the slot has to be inferred — and the inference that matters is which
/// authored collection the engine's own SPAWN STATUS states drive. Over every `chdt` in a shipped
/// cache there is exactly one: `ui\chud\scoreboard`'s `spawning_on_you`, gated on
/// `297 'teammate attempting to spawn'`, the neighbour of `298 'status ready to spawn'` and
/// `299 'status waiting to spawn'` in the same enum run.
///
/// It is anchored at `chud_anchor_type_enum` **17 `crosshair`** — the centre of the screen — at
/// `(0, -53)` virtual units, with its own text `you_are_last_man` at `(+2.6, +185.1)` from there,
/// i.e. **132 virtual units BELOW the screen centre**. That is where Reach puts a spawn message.
///
/// The first attempt used `omniwidget_high_center` instead, which is anchored `top center` at
/// `(0, +209)` and lands at about a fifth of the way down the screen — "the respawn thing seems to
/// work BUT it's in the wrong spot". The omniwidgets are MEGALO script slots (all four are gated
/// on `290 '<megalo omni widget> available'`), not the engine's status display, so they were never
/// the right family.
pub const RESPAWN_SLOT: &str = "spawning_on_you";
/// The text widget inside it whose authored placement, font and colour the countdown borrows.
pub const RESPAWN_TEXT_WIDGET: &str = "you_are_last_man";
/// How far under the label the number sits, in multiples of the label's own line height. The tag
/// authors one string here, not two, so this is HMS's own stacking and is kept to exactly one line
/// so the pair reads as one message.
pub const RESPAWN_VALUE_LINES: f32 = 1.0;

/// `game_engine_status_enum_definition` — the enum the engine's own respawn UI keys off.
pub const STATUS_RESPAWNING_SOON: i32 = 2;
pub const STATUS_WAITING_TO_SPAWN: i32 = 29;

/// The English the countdown reads. **HMS's own words**, not the game's.
///
/// There is no chud widget named for a respawn anywhere in `chdt` or `chgd` — every widget, state
/// and datasource name was searched and none mentions one. The countdown is GENERIC chud text
/// driven by `game_engine_status_enum_definition` (2 `respawning soon`, 29 `waiting to spawn`),
/// whose strings live in `smdt +0x04C state responses` as an ffa message and a team message per
/// state. HMS draws the authored SLOT — `omniwidget_high_center`'s own placement, fonts and
/// colours out of the tag — and supplies this text, rather than pretending to have resolved the
/// localized string.
pub const RESPAWN_LABEL: &str = "RESPAWNING IN";

/// Paint the respawn countdown into the authored centre-screen status slot.
///
/// Returns the slot's resolved point, so a caller can report where it went.
pub fn paint_respawn(
    ctx: &egui::Context,
    painter: &egui::Painter,
    vs: &VirtualScreen,
    art: &HudArt,
    skin: Option<&ChudSkin>,
    seconds_left: f32,
) -> Option<egui::Pos2> {
    let c = art.chud.get(RESPAWN_SLOT)?;
    let cp = c.base.placement()?;
    let (parent, _) = widget_point(vs, cp, None)?;
    let w = c.get(RESPAWN_TEXT_WIDGET)?;
    let WidgetArt::Text { font, .. } = &w.art else { return None };
    let p = w.base.placement()?;
    let (at, scale) = widget_point(vs, p, Some(parent))?;
    let col = widget_color(&w.base, skin);
    // The line height this widget's own font draws at, so the number stacks by a real line rather
    // than by a guessed number of points.
    let line = font
        .and_then(|f| crate::reach_font::atlas_for(ctx, f))
        .map(|a| a.font.line_height() as f32)
        .unwrap_or(crate::reach_chud::HUD_FONT_LINE_HEIGHT_VU)
        * vs.vpx
        * scale[1].abs().max(0.01);
    paint_text(ctx, painter, at, scale, p.origin, *font, RESPAWN_LABEL, col, vs.vpx);
    let below = egui::pos2(at.x, at.y + line * RESPAWN_VALUE_LINES);
    paint_text(
        ctx,
        painter,
        below,
        scale,
        p.origin,
        *font,
        &RespawnClock::displayed(seconds_left).to_string(),
        col,
        vs.vpx,
    );
    Some(at)
}

// ---------------------------------------------------------------------------------------------
// which chdt to draw
// ---------------------------------------------------------------------------------------------

/// `ui\chud\spartan` / `ui\chud\elite` — the per-species player HUD: shield, health, motion
/// tracker, grenades.
pub const PLAYER_CHUD_SPARTAN: &str = "ui\\chud\\spartan";
pub const PLAYER_CHUD_ELITE: &str = "ui\\chud\\elite";

/// The collections of the player HUD a Theater viewer can honestly be shown.
///
/// The motion tracker needs every nearby object's velocity and the boot sequence, commendations,
/// player training and out-of-bounds timer are all runtime state a film has nothing to say about.
pub const PLAYER_COLLECTIONS: [&str; 3] = ["shield_meter", "health_meter", "grenades"];

/// The widgets of those collections HMS draws: the BARS, their borders, and the grenade icons.
///
/// **A whitelist, and it is deliberate.** The shield and health blocks are not just two bars —
/// each carries a set of EFFECT overlays the engine drives from events: `flahsy` (a damage flash;
/// the shield's is `campaign_shield_flash` coloured `crosshair enemy`, i.e. a solid red blob, and
/// the health's is a red `blob`), `meter_border_flash`, `shield_boot_blob` (part of the HUD's
/// power-on sequence) and the `compass` pair. Every one of them is authored to draw with no
/// condition HMS can refuse, so they were all being painted at full strength on every frame —
/// a red bar across the shield and two white plates over the health: "the shields and grenades
/// stuff looks wrong as well".
///
/// Three different tag fields were tried as a general rule for "this is an effect, not a bar", and
/// none of them separates these widgets from the real ones:
/// * `scalar output A` — none of them names `flash animation`;
/// * `Flash State` — `health_meter`, the health BAR itself, authors `FLASH when (26 OR 27 OR 28)`
///   and no active state, so suppressing flash-only widgets deleted the health bar outright;
/// * the `animation data` block — `meter` and `health_meter` carry `active`+`flashing` animations
///   exactly like `flahsy` does.
///
/// So rather than ship a rule that is wrong in a way the tags do not reveal, HMS draws the pieces
/// it can actually drive and says so. The three it keeps are the three it has a value for.
///
/// The two grenade COUNTS are in the list even though nothing draws them today: they are real HUD
/// elements, not effects, and the only reason they are blank is that HMS does not yet decode the
/// unit's grenade field (`0x2E`, two bytes, on the wire and arriving 7,412 times in 5.4 M
/// messages). Keeping them here means they appear the moment that lands, instead of being
/// excluded by a list nobody remembers to update.
pub const PLAYER_WIDGETS: [&str; 7] = [
    "meter",
    "meter_border",
    "health_meter",
    "hilight",
    "unhilight",
    "grenade_count_hilite",
    "grenade_count_backpack",
];

/// Is this a widget of the player HUD that HMS draws? See [`PLAYER_WIDGETS`].
pub fn is_drawn_player_widget(collection: &str, widget: &str) -> bool {
    if !PLAYER_COLLECTIONS.contains(&collection) {
        return true; // not one of the curated blocks; the tag's own states decide
    }
    PLAYER_WIDGETS.contains(&widget)
}

/// The prefix of the weapon-HUD collections that belong on screen in first person. The shipped
/// weapons are not consistent: `ammo_area` on the assault rifle, `ammo_area_right` on the magnum.
/// `crosshair` is drawn separately, by `reach_chud::paint_crosshair`, so it is not here, and
/// `backpack_ammo_area` is left out because HMS does not know which weapon is stowed.
pub const WEAPON_AMMO_PREFIX: &str = "ammo_area";

/// Find a `chdt` by path.
pub fn chud_by_path(native: &NativeDll, cache: CacheHandle, path: &str) -> Option<u32> {
    native
        .tags_by_group(cache, "chdt")
        .into_iter()
        .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.eq_ignore_ascii_case(path)))
}

/// A seated player's own HUD: the `vehi`'s `chud interface`.
///
/// `vehicle` tags carry the same `player interface` struct a weapon does, so a vehicle's HUD is
/// found the same way — but at the VEHICLE's own base, not the weapon's. The offset is resolved
/// by searching the tag for a `chdt` reference rather than hard-coding one, because the `vehi`
/// chain's base differs from `weap`'s and the tool build's `vehi` definition does not expose the
/// `player interface` struct at a single fixed offset the way `weap` does.
pub fn vehicle_chud(native: &NativeDll, cache: CacheHandle, vehi_tag: u32) -> Option<u32> {
    let meta = native.tag_meta(cache, vehi_tag, 0, 0x900)?;
    let chdts: Vec<u32> = native.tags_by_group(cache, "chdt");
    // A tag reference is 16 bytes with the tag datum at +0x0C; scan every aligned reference and
    // take the first that names a `chdt`.
    (0..meta.len().saturating_sub(16))
        .step_by(4)
        .filter_map(|o| tag_ref(&meta, o))
        .find(|t| chdts.contains(t))
}

// ---------------------------------------------------------------------------------------------
// print-only diagnostic
// ---------------------------------------------------------------------------------------------

/// `--dump-hud <map> [chdt substring]`: every authored widget with its STATE conditions, which
/// `--dump-chdt` does not print, plus the two things a first-person HUD needs out of `weap`: the
/// aim-assist block (and whether its angles read as radians or degrees) and the raw `magazines`
/// block, whose element layout the tool build's own definition does not carry.
pub fn dump_hud(native: &NativeDll, cache: CacheHandle, filter: &str) {
    println!("# every chud widget with its Active/Hidden STATE conditions and RESOLVED colour\n");
    let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned());
    println!(
        "# skin: {}\n",
        skin.as_ref().map(|s| format!("'{}' (chud_skin_type_enum {})", s.type_name(), s.skin_type))
            .unwrap_or_else(|| "NONE — colours fall back to the authored custom A".into())
    );
    let mut tags = native.tags_by_group(cache, "chdt");
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        let Some(c) = read_chud(native, cache, t) else { continue };
        println!("================ chdt '{}'  ({} collection(s))", c.path, c.collections.len());
        for coll in &c.collections {
            let ds = coll.datasource.map(|d| format!("datasource {d} max {}", coll.datasource_max)).unwrap_or_default();
            println!(
                "  [{}] anchor {} off ({:+.1},{:+.1}) origin ({:.2},{:.2}) {ds}",
                coll.base.name,
                coll.base.placement().map(|p| p.anchor).unwrap_or(255),
                coll.base.placement().map(|p| p.offset[0]).unwrap_or(0.0),
                coll.base.placement().map(|p| p.offset[1]).unwrap_or(0.0),
                coll.base.placement().map(|p| p.origin[0]).unwrap_or(0.0),
                coll.base.placement().map(|p| p.origin[1]).unwrap_or(0.0),
            );
            print_states("      ", &coll.base);
            for w in &coll.widgets {
                let what = match &w.art {
                    WidgetArt::Bitmap { sprite, .. } => format!(
                        "BITMAP {:>6.1}x{:<6.1} vu",
                        sprite.size[0], sprite.size[1]
                    ),
                    WidgetArt::Text { font, input, flags } => format!(
                        "TEXT  font {:<28} '{input}' flags {flags:#06x}",
                        font.map(|f| f.file_stem()).unwrap_or("<none>")
                    ),
                };
                // #hud-colour  The RESOLVED colour beside the authored output, so "the colours
                // are wrong" can be checked against the tag one widget at a time.
                let c = widget_color(&w.base, skin.as_ref());
                // #hud-colour  WHERE the colour came from, the raw ARGB of that slot, whether
                // the slot was authored at all, and whether the widget is drawn — so the table
                // can be read beside what is on screen.
                let src = match w.base.color {
                    Some(k) => format!("skin '{}'", SKIN_COLORS[k]),
                    None if (0..4).contains(&w.base.output_a) => {
                        format!("custom {}", [b'A', b'B', b'C', b'D'][w.base.output_a as usize] as char)
                    }
                    None => format!("output {}", w.base.output_a),
                };
                let raw = match w.base.color {
                    Some(k) => skin.as_ref().map(|sk| sk.colors[k]).unwrap_or([0.0; 4]),
                    None if (0..4).contains(&w.base.output_a) => w.base.custom[w.base.output_a as usize],
                    None => [0.0; 4],
                };
                let set = if w.base.color.is_some() || raw.iter().any(|v| *v > 0.0) {
                    ""
                } else {
                    " UNSET, fell back"
                };
                let anim = match (w.base.anim_active, w.base.anim_flashing) {
                    (true, true) => " anim:active+flash",
                    (true, false) => " anim:active",
                    (false, true) => " anim:flash",
                    _ => "",
                };
                let mut rest = RESTING_STATES.to_vec();
                rest.extend(repeat_states(coll.datasource, 0));
                let why = if !is_drawn_player_widget(&coll.base.name, &w.base.name) {
                    "  NOT DRAWN: an effect overlay HMS cannot drive (see PLAYER_WIDGETS)"
                } else if !w.base.on(&rest) {
                    "  NOT DRAWN at rest"
                } else if w.base.animation_driven() {
                    "  NOT DRAWN: driven by an animation"
                } else {
                    ""
                };
                let swatch = format!(
                    "{src} argb({:.2},{:.2},{:.2},{:.2}){set}{anim} -> #{:02X}{:02X}{:02X}{why}",
                    raw[0], raw[1], raw[2], raw[3], c.r(), c.g(), c.b()
                );
                println!(
                    "      {:<32} sort {:>3}  {what}  anchor {} off ({:+.1},{:+.1}) origin ({:.2},{:.2}) scale ({:+.3},{:+.3})  [{swatch}]",
                    w.base.name,
                    w.base.sort,
                    w.base.placement().map(|p| p.anchor).unwrap_or(255),
                    w.base.placement().map(|p| p.offset[0]).unwrap_or(0.0),
                    w.base.placement().map(|p| p.offset[1]).unwrap_or(0.0),
                    w.base.placement().map(|p| p.origin[0]).unwrap_or(0.0),
                    w.base.placement().map(|p| p.origin[1]).unwrap_or(0.0),
                    w.base.placement().map(|p| p.scale[0]).unwrap_or(1.0),
                    w.base.placement().map(|p| p.scale[1]).unwrap_or(1.0),
                );
                print_states("          ", &w.base);
                let _ = &rest;
            }
        }
    }
    dump_weapon_numbers(native, cache, filter);
    // #play-hud  The recovered `magazines` element layout read off real weapons under BOTH
    // candidate strides, plus every playable biped's authored vitality and grenades — the
    // map-backed half of the test-mode HUD's value sources.
    crate::play_hud::dump_ammo_and_vitality(native, cache, filter);
    // #hud-trace  ...and what the painter actually EMITS for each widget, which is the only way to
    // see "the shields and health don't look right" without a window.
    crate::play_hud::dump_painted_hud(native, cache, filter);
}

/// Print a state block the way the tag means it: AND of ORs, with the resting verdict beside it,
/// so "this widget is missing" can be read rather than guessed at.
fn print_states(pad: &str, b: &WidgetBase) {
    let show = |block: &[Vec<StateCond>]| -> String {
        block
            .iter()
            .map(|term| {
                let ors = term
                    .iter()
                    .map(|c| if c.invert { format!("NOT {}", state_name(c.cond)) } else { state_name(c.cond) })
                    .collect::<Vec<_>>()
                    .join(" OR ");
                if term.len() > 1 { format!("({ors})") } else { ors }
            })
            .collect::<Vec<_>>()
            .join(" AND ")
    };
    if !b.active.is_empty() {
        println!("{pad}ACTIVE when: {}", show(&b.active));
    }
    if !b.flash.is_empty() {
        println!("{pad}FLASH when: {}", show(&b.flash));
    }
    if !b.hidden.is_empty() {
        println!("{pad}HIDDEN when: {}", show(&b.hidden));
    }
    if !(b.active.is_empty() && b.flash.is_empty() && b.hidden.is_empty()) {
        println!("{pad}  -> at rest: {}", if b.on(&RESTING_STATES) { "DRAWN" } else { "off" });
    }
}

/// The two `weap` numbers a HUD needs, read off real weapons: the aim-assist block (with both
/// readings of every angle printed, so radians-vs-degrees is settled by looking rather than by
/// assuming) and the raw bytes of the `magazines` block, whose element layout the tool build's
/// own tag definition leaves empty.
pub fn dump_weapon_numbers(native: &NativeDll, cache: CacheHandle, filter: &str) {
    println!("\n# weap +{WEAPON_AIM_ASSIST:#05x} 'weapon aim assist' and +0x3F0 'magazines'\n");
    let mut tags = native.tags_by_group(cache, "weap");
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        let short = name.rsplit('\\').next().unwrap_or(&name).to_string();
        let Some(a) = AimAssist::read(native, cache, t) else { continue };
        println!(
            "  {short:<34} autoaim {:9.5} raw = {:6.2} deg (as rad) / {:6.4} rad (as deg)   range {:6.2}+{:.2}",
            a.autoaim_angle,
            a.autoaim_angle.to_degrees(),
            a.autoaim_angle.to_radians(),
            a.autoaim_range,
            a.autoaim_falloff_range
        );
        println!(
            "  {:<34} magnetism {:8.5} = {:6.2} deg   deviation {:8.5} = {:6.2} deg   {}",
            "",
            a.magnetism_angle,
            a.magnetism_angle.to_degrees(),
            a.deviation_angle,
            a.deviation_angle.to_degrees(),
            if a.angles_are_radians() { "(radian reading is on the 0.05-deg grid)" } else { "<-- NOT on the grid" }
        );
        // The magazines block, raw. The tool build's definition gives count and max but no
        // element size, so print the header and the first 48 bytes the pointer actually holds.
        if let Some(meta) = native.tag_meta(cache, t, 0, 0x400) {
            let n = rd_i32(&meta, 0x3F0).unwrap_or(0);
            let ptr = rd_u32(&meta, 0x3F4).unwrap_or(0);
            if n > 0 && ptr != 0 && ptr != u32::MAX {
                if let Some(b) = native.tag_ptr_bytes(cache, ptr, 48) {
                    let shorts: Vec<String> =
                        b.chunks_exact(2).take(16).map(|c| i16::from_le_bytes([c[0], c[1]]).to_string()).collect();
                    println!("  {:<34} magazines[{n}] shorts: {}", "", shorts.join(" "));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// tiny byte readers, local so this file never reaches into another's privates
// ---------------------------------------------------------------------------------------------

fn rd_i32(b: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

/// A tag block's payload: `{i32 count, u32 address, u32 definition}` at `off` inside `buf`.
fn block(native: &NativeDll, cache: CacheHandle, buf: &[u8], off: usize, elem: usize, max: i32) -> Vec<u8> {
    let (Some(n), Some(ptr)) = (rd_i32(buf, off), rd_u32(buf, off + 4)) else { return Vec::new() };
    if n <= 0 || n > max || ptr == 0 || ptr == u32::MAX {
        return Vec::new();
    }
    native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
}

fn tag_ref(buf: &[u8], off: usize) -> Option<u32> {
    let datum = rd_u32(buf, off + 0x0C)?;
    (datum != u32::MAX).then_some(datum & 0xFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE MODE GATE.** Every spelling the three-way setting has to survive: the GUI writes a
    /// keyword, the script verb reads one back, and the file written in one session is read in the
    /// next. A round trip that loses `LookedAt` would silently turn the new mode into the old
    /// boolean, which is exactly the bug the bool had.
    #[test]
    fn the_name_tag_mode_round_trips_through_its_setting_file() {
        let dir = std::env::temp_dir().join(format!("hms-nametag-{}", std::process::id()));
        for m in NameTagMode::ALL {
            save_name_tag_mode_to_dir(&dir, m);
            assert_eq!(load_name_tag_mode_from_dir(&dir), m, "{:?} did not survive the file", m);
            // and the keyword it wrote parses back to itself
            assert_eq!(NameTagMode::parse(m.keyword()), Some(m));
        }
        // the OLD boolean spellings still work, so a settings folder from before the three-way
        // mode keeps meaning what it meant
        assert_eq!(NameTagMode::parse("1"), Some(NameTagMode::Always));
        assert_eq!(NameTagMode::parse("on"), Some(NameTagMode::Always));
        assert_eq!(NameTagMode::parse("0"), Some(NameTagMode::Hidden));
        assert_eq!(NameTagMode::parse("off"), Some(NameTagMode::Hidden));
        assert_eq!(NameTagMode::parse("nonsense"), None);
        // a missing file is the default, not a panic
        let empty = dir.join("does-not-exist");
        assert_eq!(load_name_tag_mode_from_dir(&empty), NameTagMode::Always);
        assert!(NameTagMode::Always.is_on() && NameTagMode::LookedAt.is_on() && !NameTagMode::Hidden.is_on());
    }

    /// **THE LOOK-AT GATE.** The cone has to be the ENGINE's cone — the hard-coded 3.0 degree
    /// half-angle and 40 world units out of `HUD_FindNearestTarget` — it has to behave like a
    /// cone, and the winner has to be the NEAREST body inside it rather than the most on-axis
    /// one, because that is the tie-break the engine uses.
    ///
    /// 3 degrees is narrow on purpose: at 20 wu a body 1.5 wu to the side is already outside,
    /// which is exactly what makes the mode read as "when you look AT them".
    #[test]
    fn the_look_at_cone_is_the_engines_cone_and_picks_the_nearest_body() {
        // the two engine constants, pinned to their decompiled values
        assert!(
            (LOOKAT_HALF_ANGLE_RAD.to_degrees() - 3.0).abs() < 0.0005,
            "flt_180A8AB78 is 3.0000 degrees, got {}",
            LOOKAT_HALF_ANGLE_RAD.to_degrees()
        );
        assert_eq!(LOOKAT_RANGE_WU, 40.0, "dword_180A8B3F4");
        let cone = UNARMED_LOOKAT;
        let eye = glam::Vec3::ZERO;
        let fwd = glam::Vec3::X;
        assert!(cone.contains(eye, fwd, glam::vec3(10.0, 0.0, 0.0)), "dead ahead");
        // 20 wu out: 0.5 wu off axis = 1.43 deg (in), 1.5 wu = 4.29 deg (out).
        assert!(cone.contains(eye, fwd, glam::vec3(20.0, 0.5, 0.0)));
        assert!(!cone.contains(eye, fwd, glam::vec3(20.0, 1.5, 0.0)), "4.3 deg is outside a 3 deg cone");
        assert!(!cone.contains(eye, fwd, glam::vec3(-10.0, 0.0, 0.0)), "behind");
        assert!(!cone.contains(eye, fwd, glam::vec3(41.0, 0.0, 0.0)), "past the range");
        assert!(!cone.contains(eye, fwd, glam::vec3(0.0, 0.0, 0.0)), "at the eye");
        // THE NEAREST wins, not the most on-axis. The far body here is dead centre and the near
        // one is 1.1 degrees off, and the engine still takes the near one.
        let anchors = [glam::vec3(30.0, 0.0, 0.0), glam::vec3(10.0, 0.2, 0.0)];
        assert!(cone.contains(eye, fwd, anchors[0]) && cone.contains(eye, fwd, anchors[1]));
        assert!(
            LookAtCone::off_axis(eye, fwd, anchors[1]) > LookAtCone::off_axis(eye, fwd, anchors[0]),
            "the near body really is the more off-axis of the two"
        );
        assert_eq!(looked_at(&cone, eye, fwd, &anchors), Some(1), "nearest, not most on-axis");
        assert_eq!(looked_at(&cone, eye, fwd, &[glam::vec3(10.0, 9.0, 0.0)]), None);
        assert_eq!(looked_at(&cone, eye, fwd, &[]), None);
        // ZOOM stretches the range and nothing else. `sub_1804856B8`: one level is the min
        // outright, several interpolate geometrically between min and max.
        let unscoped = LookAtCone::for_weapon(Some(ZoomLevels { levels: 2, range: (4.0, 10.0) }), -1);
        assert_eq!(unscoped.range, 40.0, "not scoped: magnification 1.0");
        let z0 = LookAtCone::for_weapon(Some(ZoomLevels { levels: 2, range: (4.0, 10.0) }), 0);
        assert!((z0.magnification - 4.0).abs() < 1e-4, "{}", z0.magnification);
        assert!((z0.range - 160.0).abs() < 1e-3);
        let z1 = LookAtCone::for_weapon(Some(ZoomLevels { levels: 2, range: (4.0, 10.0) }), 1);
        assert!((z1.magnification - 10.0).abs() < 1e-4, "{}", z1.magnification);
        let one = LookAtCone::for_weapon(Some(ZoomLevels { levels: 1, range: (3.0, 3.0) }), 0);
        assert!((one.magnification - 3.0).abs() < 1e-4);
        // a weapon that does not scope, and no weapon at all, are both the plain cone
        let none = LookAtCone::for_weapon(Some(ZoomLevels { levels: 0, range: (0.0, 0.0) }), 0);
        assert_eq!(none, UNARMED_LOOKAT);
        assert_eq!(LookAtCone::for_weapon(None, -1), UNARMED_LOOKAT);
        // the angle never changes with zoom — only the range does
        for z in [-1i8, 0, 1] {
            assert_eq!(
                LookAtCone::for_weapon(Some(ZoomLevels { levels: 2, range: (4.0, 10.0) }), z).angle,
                LOOKAT_HALF_ANGLE_RAD
            );
        }
    }

    /// **THE RESPAWN GATE.** The countdown is arithmetic on the variant's own four numbers, and
    /// every one of them has to be in play: the base, the growth per death, and the two
    /// penalties, with the printed number the number of whole seconds still to wait.
    #[test]
    fn the_respawn_countdown_uses_the_variants_own_numbers() {
        // the shape `--film-stats` prints: `respawn 3 s (+0 growth, suicide +5, betrayal +5)`
        let r = RespawnRules { base: 3, growth: 0, suicide: 5, betrayal: 5 };
        assert_eq!(r.seconds(0, DeathKind::Normal), 3.0);
        assert_eq!(r.seconds(7, DeathKind::Normal), 3.0, "no growth means deaths do not add");
        assert_eq!(r.seconds(0, DeathKind::Suicide), 8.0);
        assert_eq!(r.seconds(0, DeathKind::Betrayal), 8.0);
        // growth multiplies the number of PREVIOUS deaths
        let g = RespawnRules { base: 5, growth: 2, suicide: 0, betrayal: 3 };
        assert_eq!(g.seconds(0, DeathKind::Normal), 5.0);
        assert_eq!(g.seconds(3, DeathKind::Normal), 11.0);
        assert_eq!(g.seconds(3, DeathKind::Betrayal), 14.0);
        // the clock
        assert_eq!(r.remaining(100.0, 100.0, 0, DeathKind::Normal), Some(3.0));
        assert_eq!(r.remaining(100.0, 102.5, 0, DeathKind::Normal), Some(0.5));
        assert_eq!(r.remaining(100.0, 103.0, 0, DeathKind::Normal), None, "run out");
        assert_eq!(r.remaining(100.0, 99.0, 0, DeathKind::Normal), None, "before the death");
        // the printed number: whole seconds still to wait
        assert_eq!(RespawnClock::displayed(3.0), 3);
        assert_eq!(RespawnClock::displayed(2.3), 3);
        assert_eq!(RespawnClock::displayed(0.1), 1);
        assert_eq!(RespawnClock::displayed(0.0), 0);

        // THE TWO CLOCKS, and which one wins. The recording's own next life is exact at its far
        // end; the variant's arithmetic is the fallback when there is no next life to point at.
        let live = RespawnClock::NextLife { died_at: 100.0, respawn_at: 103.1 };
        let near = |a: Option<f32>, b: f32| matches!(a, Some(v) if (v - b).abs() < 1e-3);
        assert!(near(live.remaining(100.0), 3.1), "{:?}", live.remaining(100.0));
        assert!(near(live.remaining(103.0), 0.1), "{:?}", live.remaining(103.0));
        assert_eq!(live.remaining(103.1), None, "the countdown ends ON the respawn");
        assert_eq!(live.remaining(99.0), None, "before the death there is no countdown");
        assert!(live.note().contains("next life"), "{}", live.note());
        let fb =
            RespawnClock::FromRules { died_at: 10.0, rules: r, deaths_before: 0, kind: DeathKind::Betrayal };
        assert_eq!(fb.remaining(10.0), Some(8.0), "base 3 + betrayal 5");
        assert_eq!(fb.remaining(18.0), None);
        assert!(fb.note().contains("game variant's own"), "{}", fb.note());
    }

    /// **THE STATE GATE, and it is a gate on the decoded STRUCTURE.**
    ///
    /// The boolean shape of a `chud` state block is AND-of-ORs, and the two readings disagree on
    /// real tags. This is the bug that made 34 of Reach's 64 weapons look as though they authored
    /// no crosshair: the magnum's five reticule parts are each `(97 right hand OR 98 left hand)`,
    /// a filter that demanded an EMPTY active block threw all five away, and the dump reported "no
    /// chud interface" for the entire player sandbox. Meanwhile its `headshot` pip is
    /// `103 AND 97` — an AND — and reading that as an OR would paint a headshot marker on screen
    /// all the time.
    ///
    /// So: AND of ORs, a state HMS cannot assert is FALSE, and a satisfied Hidden block beats
    /// everything.
    #[test]
    fn a_widgets_state_is_an_and_of_ors_and_what_hms_cannot_assert_is_false() {
        let base = |active: Vec<Vec<StateCond>>, hidden: Vec<Vec<StateCond>>| WidgetBase {
            name: "w".into(),
            sort: 0,
            placements: Vec::new(),
            active,
            flash: Vec::new(),
            hidden,
            color: None,
            output_a: 0,
            custom: [[1.0; 4]; 4],
            scalar_a: 0,
            anim_active: false,
            anim_flashing: false,
            shader_type: 0,
            blend_mode: 3,
        };
        let is = StateCond::is;
        let not = StateCond::not;
        let known = [STATE_FULLSCREEN_WIDE, STATE_FIRST_PERSON, STATE_UNZOOMED];
        assert!(base(vec![], vec![]).on(&known), "no conditions = always on");
        assert!(base(vec![vec![is(STATE_UNZOOMED)]], vec![]).on(&known));
        assert!(!base(vec![vec![is(STATE_ZOOMED)]], vec![]).on(&known), "a state HMS cannot assert is FALSE");
        assert!(!base(vec![vec![is(STATE_WAITING_TO_SPAWN)]], vec![]).on(&known));
        assert!(!base(vec![], vec![vec![is(STATE_FIRST_PERSON)]]).on(&known), "hidden beats unconditional");
        assert!(
            !base(vec![vec![is(STATE_UNZOOMED)]], vec![vec![is(STATE_FIRST_PERSON)]]).on(&known),
            "hidden beats an asserted active state"
        );
        // THE OR: one term, two clauses, only one of them true -> ON.
        assert!(
            base(vec![vec![is(STATE_ZOOMED), is(STATE_UNZOOMED)]], vec![]).on(&known),
            "an OR needs only one of its clauses"
        );
        // THE AND: two terms, one of them false -> OFF. Read as a flat OR this would be ON, which
        // is exactly the magnum's `headshot` pip drawn permanently.
        assert!(
            !base(vec![vec![is(STATE_UNZOOMED)], vec![is(STATE_ZOOMED)]], vec![]).on(&known),
            "every AND term must hold"
        );
        assert!(base(vec![vec![is(STATE_UNZOOMED)], vec![is(STATE_FIRST_PERSON)]], vec![]).on(&known));
        // THE INVERT FLAG. `chud_widget_state_flags` bit 0 is "Check if the condition is false",
        // and the Spartan HUD's whole `grenades` collection is authored `NOT 92 grenades
        // unusable`. Drop the flag and that block never draws -- which is exactly what happened
        // before it was decoded.
        let dec = decidable_states(&known);
        assert!(not(92).holds(&known, &dec), "NOT 92 holds: 92 is decidable and false");
        assert!(!is(92).holds(&known, &dec));
        // AN UNDECIDABLE STATE NEVER HOLDS, inverted or not. 169 `watermark compass on` is the
        // real case: a plain closed world would make `NOT 169` true and paint a 1024-unit compass
        // strip across every frame on a state nobody evaluated.
        assert!(!is(169).holds(&known, &dec), "169 is not asserted");
        assert!(!not(169).holds(&known, &dec), "and NOT 169 must not hold either -- 169 is unknown");
        assert!(!satisfied(&[vec![not(169)]], &known), "the compass stays off");
        assert!(KNOWN_FALSE_STATES.contains(&92), "the grenade gate needs 92 to be decidable");
        assert!(!KNOWN_FALSE_STATES.contains(&169) && !RESTING_STATES.contains(&169));
        assert!(base(vec![vec![not(92)]], vec![]).on(&known), "the grenade block's own gate");
        assert!(!base(vec![vec![not(STATE_UNZOOMED)]], vec![]).on(&known), "NOT a TRUE state is false");
        assert!(
            base(vec![vec![not(STATE_UNZOOMED), is(STATE_FIRST_PERSON)]], vec![]).on(&known),
            "an inverted clause still only has to satisfy its OR"
        );
        // an inverted HIDDEN clause suppresses too
        assert!(!base(vec![], vec![vec![not(STATE_ZOOMED)]], ).on(&known), "HIDDEN when NOT zoomed");
        // the magnum's own two shapes, with the RESTING set the crosshair is evaluated against
        let magnum_half = base(vec![vec![is(STATE_WEAPON_IS_RIGHT_HAND), is(98)]], vec![]);
        assert!(magnum_half.on(&RESTING_STATES), "the magnum's reticule halves draw at rest");
        let magnum_pip = base(vec![vec![is(103)], vec![is(STATE_WEAPON_IS_RIGHT_HAND)]], vec![]);
        assert!(!magnum_pip.on(&RESTING_STATES), "its headshot pip does NOT");
        // and the resting set is the one the engine's own HUD rests in
        assert!(RESTING_STATES.contains(&STATE_WEAPON_IS_RIGHT_HAND), "Reach holds the one weapon right-handed");
        assert!(RESTING_STATES.contains(&STATE_UNZOOMED) && !RESTING_STATES.contains(&STATE_ZOOMED));
        assert!(!RESTING_STATES.contains(&98), "there is no left-hand weapon in Reach");
        assert!(base(vec![], vec![]).unconditional());
        assert!(!base(vec![vec![is(STATE_ZOOMED)]], vec![]).unconditional());
        assert_eq!(base(vec![vec![is(5), not(3)]], vec![vec![is(9)]]).mentions(), vec![3, 5, 9]);
        // `satisfied` on its own: an empty block is satisfied, which is what makes an empty
        // ACTIVE block unconditional and why `on` has to check `hidden` for emptiness separately.
        assert!(satisfied(&[], &known));
        assert!(!satisfied(&[vec![is(STATE_ZOOMED)]], &known));
        assert_eq!(STATE_FLAG_INVERT, 1, "chud_widget_state_flags bit 0");

        // THE PER-REPEAT STATES. A datasource-driven collection is drawn once per element and
        // `<grenade> selected` is true only on the repeat whose type is up -- a property of the
        // repeat, not of the frame. Without them the grenade block draws its frame and nothing
        // inside it, because all four of its widgets are gated on one or the other.
        assert_eq!(repeat_states(Some(DATASOURCE_GRENADES), 0), vec![STATE_GRENADE_SELECTED]);
        assert_eq!(repeat_states(Some(DATASOURCE_GRENADES), 1), vec![STATE_GRENADE_UNSELECTED]);
        assert!(repeat_states(None, 0).is_empty(), "a collection with no datasource has no repeat states");
        assert!(repeat_states(Some(7), 0).is_empty(), "and nor does a `scripted objects` one");
        let mut rest = RESTING_STATES.to_vec();
        rest.extend(repeat_states(Some(DATASOURCE_GRENADES), 0));
        assert!(
            base(vec![vec![is(STATE_GRENADE_SELECTED)]], vec![]).on(&rest),
            "the selected grenade's icon and count draw on repeat 0"
        );
        assert!(
            !base(vec![vec![is(STATE_GRENADE_SELECTED)]], vec![]).on(&RESTING_STATES),
            "...and not without the repeat's own states"
        );
        // the block's own gate is the inverted one, and it holds in both repeats
        for r in 0..2 {
            let mut k = RESTING_STATES.to_vec();
            k.extend(repeat_states(Some(DATASOURCE_GRENADES), r));
            assert!(base(vec![vec![not(STATE_GRENADES_UNUSABLE)]], vec![vec![is(STATE_ZOOMED)]]).on(&k));
        }
    }

    /// #hud-shader  **THE BLEND GATE, ON THE OBSERVABLE.**
    ///
    /// A gate asserting "we read blend mode 1 for `meter_border`" would have been GREEN before the
    /// fix too, because the field was read into a struct and then ignored downstream. So this
    /// asserts the thing that actually matters: **composite an additive widget over a bright one
    /// through egui's own blend equation and check the bright one survives.**
    ///
    /// egui's pipeline is `src_factor = ONE`, `dst_factor = ONE_MINUS_SRC_ALPHA`, over
    /// premultiplied vertex colours, so the framebuffer gets `src.rgb + dst.rgb * (1 - src.a)`.
    /// That equation is transcribed here — not re-derived from the code under test — and run on
    /// the two colours the shipped Spartan HUD actually authors.
    #[test]
    fn an_additive_widget_does_not_paint_out_what_is_under_it() {
        /// egui's blend, as the GPU does it. `src` is already premultiplied.
        fn composite(src: egui::Color32, dst: egui::Color32) -> [u16; 3] {
            let inv = 255 - src.a() as u16;
            [
                src.r() as u16 + dst.r() as u16 * inv / 255,
                src.g() as u16 + dst.g() as u16 * inv / 255,
                src.b() as u16 + dst.b() as u16 * inv / 255,
            ]
        }
        // The two colours measured off `ui\chud\spartan`: the shield BAR resolves to #2495FF
        // (skin `highlight`) and its BORDER to #004CB5, and the border is drawn second.
        let bar = egui::Color32::from_rgb(0x24, 0x95, 0xFF);
        let border = egui::Color32::from_rgb(0x00, 0x4C, 0xB5);

        // THE BUG: the border taken as alpha-over. An opaque tint has a = 255, so `inv` is 0 and
        // the destination contributes NOTHING — the bar is gone.
        let wrong = blend_tint(border, 3).expect("alpha blend is expressible");
        assert_eq!(wrong.a(), 255, "an opaque alpha-blend tint covers whatever is under it");
        let over = composite(wrong, bar);
        assert_eq!(over, [0x00, 0x4C, 0xB5], "alpha-over leaves only the border — the bar is painted out");
        assert!(
            over[1] < bar.g() as u16,
            "and it is DARKER than the bar, which is what 'the shields don't look right' looked like"
        );

        // THE FIX: the border in its authored ADDITIVE mode. Premultiplied with a = 0, so `inv` is
        // 255, the destination survives in full and the border's colour is ADDED to it.
        let right = blend_tint(border, BLEND_ADDITIVE).expect("additive is expressible");
        assert_eq!(right.a(), 0, "additive claims no coverage");
        let add = composite(right, bar);
        assert_eq!(
            add,
            [0x24, 0x95 + 0x4C, 0xFF + 0xB5],
            "additive ADDS to the bar instead of replacing it"
        );
        for k in 0..3 {
            assert!(
                add[k] >= [bar.r(), bar.g(), bar.b()][k] as u16,
                "no channel may come out darker than the bar it was drawn over"
            );
        }
        // The bar is still there, which is the whole claim.
        assert!(add[1] > over[1] && add[2] > over[2], "additive preserves the bar, alpha-over destroys it");

        // AND THE MODES THAT CANNOT BE EXPRESSED MUST REFUSE, so nothing is drawn in the wrong one.
        assert_eq!(blend_tint(border, 2), None, "`multiply` is not expressible as one alpha-over quad");
        assert_eq!(blend_tint(border, 6), None, "nor is `maximum`");
        assert_eq!(blend_tint(border, 17), None, "nor `minimum`");
        assert!(blend_tint(border, -1).is_some(), "no render data is the ordinary alpha-over case");
        // The names, so a diagnostic cannot mislabel what it refused.
        assert_eq!(blend_mode_name(0), "opaque");
        assert_eq!(blend_mode_name(BLEND_ADDITIVE), "additive");
        assert_eq!(blend_mode_name(3), "alpha blend");
        assert_eq!(blend_mode_name(6), "maximum");
        assert_eq!(blend_mode_name(99), "<out of range>");
    }

    /// #hud-shader  **THE SHADER GATE.** "Is this a meter" is an authored field now, and the
    /// seven-member meter family has to be exactly the enum's own.
    #[test]
    fn whether_a_widget_is_a_meter_comes_from_its_authored_shader() {
        // The enum, pinned. These are the member names out of the tool build's reflection data.
        assert_eq!(shader_type_name(SHADER_REALLY_SIMPLE), "really simple{simple}");
        assert_eq!(shader_type_name(1), "anti-alias");
        assert_eq!(shader_type_name(SHADER_METER), "meter");
        assert_eq!(shader_type_name(3), "text simple");
        assert_eq!(shader_type_name(SHADER_METER_SHIELD), "meter shield");
        assert_eq!(shader_type_name(SHADER_CROSSHAIR), "crosshair");
        assert_eq!(shader_type_name(16), "meter radial gradient");
        assert_eq!(shader_type_name(21), "<out of range>", "the enum has exactly 21 members");
        // Seven of the twenty-one are meters, and the set is the enum's.
        assert_eq!(METER_SHADERS.len(), 7);
        for m in METER_SHADERS {
            assert!(shader_type_name(m).contains("meter"), "{m} '{}' is not a meter", shader_type_name(m));
        }
        for v in 0..21 {
            assert_eq!(
                shader_is_meter(v),
                shader_type_name(v).contains("meter"),
                "{v} '{}' is classified wrongly",
                shader_type_name(v)
            );
        }
        // THE WIDGETS THE PLAYER HUD ACTUALLY DRAWS, by shader, as measured on `ui\chud\spartan`.
        // The shield bar and the health bar are meters; the BORDER is not, and that is the case the
        // deleted name rule had to special-case by prefix.
        assert!(shader_is_meter(SHADER_METER_SHIELD), "shield_meter/meter is `meter shield`");
        assert!(shader_is_meter(SHADER_METER), "health_meter/health_meter is `meter`");
        assert!(!shader_is_meter(1), "shield_meter/meter_border is `anti-alias`, not a meter");
        assert!(!shader_is_meter(3), "a text widget is not a meter");

        // THE CROSS-CHECK WORTH KEEPING: on the shipped player HUD the authored shader and the
        // deleted name rule AGREE. They are kept side by side here because a reader will ask why
        // both ever existed — and because if they ever disagree, the tag wins.
        let cv = CollectionValues {
            v: &HudValues { shield: Field::recorded(0.5), health: Field::recorded(0.25), ..Default::default() },
        };
        for (coll, widget, shader) in [
            ("shield_meter", "meter", SHADER_METER_SHIELD),
            ("shield_meter", "meter_sup2", SHADER_METER_SHIELD),
            ("health_meter", "health_meter", SHADER_METER),
            ("shield_meter", "meter_border", 1),
        ] {
            assert_eq!(
                cv.meter_for(coll, widget, Some(shader)).is_some(),
                cv.meter_for(coll, widget, None).is_some(),
                "{coll}/{widget}: the authored shader and the old name rule disagree"
            );
        }
        // ...and the shader rule answers a case the name rule could not: a meter whose widget is
        // not called `meter` at all.
        assert_eq!(cv.meter_for("shield_meter", "anything", Some(SHADER_METER_SHIELD)), Some(Some(0.5)));
        assert_eq!(cv.meter_for("shield_meter", "anything", None), None, "the name rule misses it");
        // A meter in a collection with no value is a meter with no level: NOT drawn, rather than
        // drawn full.
        assert_eq!(cv.meter_for("boost_meter", "meter", Some(SHADER_METER)), Some(None));
        assert_eq!(cv.meter_for("boost_meter", "meter", Some(SHADER_REALLY_SIMPLE)), None, "not a meter at all");
    }

    /// #hud-trace  **THE OCCLUSION GATE.** "Drawn" is not "visible": the shield bar was painted
    /// and then buried, and the trace has to say so.
    #[test]
    fn the_trace_reports_a_widget_that_was_painted_and_then_covered() {
        let at = |x: f32, y: f32, w: f32, h: f32, blend: i32, name: &str, outcome: &str| WidgetDraw {
            collection: "shield_meter".into(),
            widget: name.into(),
            repeat: 0,
            kind: "bitmap",
            sprite: None,
            art_route: None,
            shader_type: 0,
            blend_mode: blend,
            text: None,
            anchor: 0,
            offset: [0.0, 0.0],
            origin: [0.0, 0.0],
            scale: [1.0, 1.0],
            rect: Some(egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h))),
            meter: None,
            color: None,
            outcome: outcome.into(),
        };
        // The real case: the bar, then an ALPHA-OVER border at the identical rect. Covered.
        let bad = vec![
            at(0.0, 0.0, 249.0, 57.0, 3, "meter", "drawn"),
            at(0.0, 0.0, 249.0, 57.0, 3, "meter_border", "drawn"),
        ];
        let occ = occlusions(&bad);
        assert_eq!(occ.len(), 1, "the bar must be reported as covered: {occ:?}");
        assert_eq!(occ[0].0, 0, "it is the FIRST widget that is buried");
        assert_eq!(occ[0].1, "meter_border", "and the border is what buried it");

        // The FIXED case: the same geometry with the border ADDITIVE. Nothing is covered, because
        // an additive quad cannot hide what is under it.
        let good = vec![
            at(0.0, 0.0, 249.0, 57.0, 3, "meter", "drawn"),
            at(0.0, 0.0, 249.0, 57.0, BLEND_ADDITIVE, "meter_border", "drawn"),
        ];
        assert!(occlusions(&good).is_empty(), "an additive border buries nothing");

        // ORDER MATTERS: a widget drawn FIRST cannot bury one drawn after it.
        let reversed = vec![
            at(0.0, 0.0, 249.0, 57.0, 3, "meter_border", "drawn"),
            at(0.0, 0.0, 249.0, 57.0, 3, "meter", "drawn"),
        ];
        assert_eq!(occlusions(&reversed)[0].1, "meter", "the later one is always the one that covers");

        // A PARTIAL overlap is not a cover, and a widget that was never drawn is not either.
        let partial = vec![
            at(0.0, 0.0, 249.0, 57.0, 3, "meter", "drawn"),
            at(100.0, 0.0, 249.0, 57.0, 3, "meter_border", "drawn"),
        ];
        assert!(occlusions(&partial).is_empty(), "a partial overlap leaves the bar visible");
        let skipped = vec![
            at(0.0, 0.0, 249.0, 57.0, 3, "meter", "drawn"),
            at(0.0, 0.0, 249.0, 57.0, 3, "meter_border", "NOT DRAWN: whatever"),
        ];
        assert!(occlusions(&skipped).is_empty(), "a widget that did not draw cannot cover one that did");
        assert!(occlusions(&[]).is_empty());
    }

    /// **THE NUMBER-FORMAT GATE.** `chud_widget_text_flags` decides how `chud_out_a` prints, and
    /// getting it wrong is how an ammo count becomes a clock.
    #[test]
    fn the_authored_text_flags_format_the_number() {
        assert_eq!(format_number(7, TEXT_IS_NUMBER), "7");
        assert_eq!(format_number(7, TEXT_IS_NUMBER | (1 << 1)), "07");
        assert_eq!(format_number(7, TEXT_IS_NUMBER | (1 << 2)), "007");
        assert_eq!(format_number(125, TEXT_IS_NUMBER | TEXT_FORMAT_AS_TIME), "2:05");
        assert_eq!(format_number(0, TEXT_IS_NUMBER), "0");
    }

    /// **THE PROVENANCE GATE.** The panel must never print a number without saying where it came
    /// from, and an absent field must stay empty rather than defaulting to zero.
    #[test]
    fn an_absent_field_reports_itself_as_absent() {
        let mut v = HudValues::default();
        let r = v.report();
        assert!(r.iter().all(|l| l.contains("not in the recording")), "{r:?}");
        assert!(v.reserve.value.is_none());
        // A METER WITH NO VALUE IS NOT DRAWN. `Some(None)` is the "it is a meter and I have no
        // level for it" answer, and it has to stay distinct from both `None` (not a meter, draw
        // the whole sprite) and `Some(Some(1.0))` (full). Defaulting an unknown meter to full is
        // a claim about the recording, and it is the exact shape of mistake this file exists to
        // avoid: an ammo bar reading FULL on a clip that carries no ammo at all.
        {
            let cv = CollectionValues { v: &v };
            assert_eq!(cv.meter("ammo_area", "meter"), Some(None), "an unknown magazine is not drawn");
            assert_eq!(cv.meter("ammo_area_right", "meter"), Some(None), "...under either authored name");
            assert_eq!(cv.meter("shield_meter", "meter_sup3"), Some(None), "an unknown shield is not drawn");
            assert_eq!(cv.meter("health_meter", "health_meter"), Some(None), "an unknown health is not drawn");
            assert_eq!(cv.meter("shield_meter", "meter_border"), None, "the border is not a meter at all");
            assert_eq!(cv.meter("ammo_area", "shematic"), None, "nor is the schematic glyph");
            assert_eq!(cv.fill("health_meter", "health_meter"), None);
        }
        v.loaded = Field::derived(12);
        v.loaded_max = Some(32);
        v.shield = Field::derived(1.0);
        let r = v.report();
        assert!(r.iter().any(|l| l == "Magazine: 12 (derived)"), "{r:?}");
        assert!(r.iter().any(|l| l == "Shield: 100% (derived)"), "{r:?}");
        assert!(r.iter().any(|l| l == "Reserve: not in the recording"), "{r:?}");
        // the meter denominators come through
        let cv = CollectionValues { v: &v };
        assert_eq!(cv.fill("ammo_area", "meter"), Some(12.0 / 32.0));
        assert_eq!(cv.fill("ammo_area", "nothing"), None);
        assert_eq!(cv.fill("shield_meter", "meter_sup2"), Some(1.0));
        assert_eq!(cv.fill("shield_meter", "meter_border"), None, "the border is not a meter");
        assert_eq!(cv.number("ammo_area", "reserve_count", 0), None, "reserve is absent");
    }

    /// **THE VIRTUAL-SCREEN GATE.** Every anchor has to land on the right corner of the safe
    /// frame, and a runtime anchor HMS cannot resolve has to come back `None` rather than (0,0).
    #[test]
    fn the_anchors_land_on_the_safe_frame() {
        let vp = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1920.0, 1080.0));
        let vs = VirtualScreen::fallback(vp);
        // Reach's own 0.87 safe frame on a 1080-line window is 939.6 points tall, and one virtual
        // unit is 939.6 / 720 = 1.305 points.
        assert!((vs.vpx - 1.305).abs() < 0.001, "{}", vs.vpx);
        assert!((vs.safe.height() - 1080.0 * 0.87).abs() < 0.01);
        assert_eq!(vs.anchor(ANCHOR_CENTER), Some(vs.safe.center()));
        assert_eq!(vs.anchor(ANCHOR_TOP_LEFT), Some(vs.safe.left_top()));
        assert_eq!(vs.anchor(ANCHOR_TOP_RIGHT), Some(vs.safe.right_top()));
        assert_eq!(vs.anchor(ANCHOR_BOTTOM_RIGHT), Some(vs.safe.right_bottom()));
        // the aim point is the VIEWPORT's centre, not the safe frame's
        assert_eq!(vs.anchor(ANCHOR_CROSSHAIR), Some(vp.center()));
        // a runtime anchor: a tracked target, a territory, a <player>
        assert_eq!(vs.anchor(15), None);
        assert_eq!(vs.anchor(25), None);
        assert_eq!(vs.anchor(34), None);
        // `parent` is not a screen anchor at all
        assert_eq!(vs.anchor(ANCHOR_PARENT), None);
        // the ammo block: `top right` at (+12, +31) virtual units
        let p = Placement {
            res_flags: 1,
            anchor: ANCHOR_TOP_RIGHT,
            origin: [0.5, 0.5],
            offset: [12.0, 31.0],
            scale: [1.0, 1.0],
        };
        let (at, _) = widget_point(&vs, &p, None).expect("top right resolves");
        assert!((at.x - (vs.safe.right() + 12.0 * vs.vpx)).abs() < 0.01);
        assert!((at.y - (vs.safe.top() + 31.0 * vs.vpx)).abs() < 0.01);
        // a child of it, at `parent` -37 in x
        let child =
            Placement { res_flags: 1, anchor: ANCHOR_PARENT, origin: [1.0, 0.5], offset: [-37.0, 27.0], scale: [1.0, 1.0] };
        assert!(widget_point(&vs, &child, None).is_none(), "a parent anchor with no parent resolves to nothing");
        let (c, _) = widget_point(&vs, &child, Some(at)).expect("with a parent it resolves");
        assert!(c.x < at.x, "the schematic sits LEFT of the block's point");
        assert!(c.y > at.y, "and below it");
    }

    /// The window-state pick: a widget authors up to five placements and the full-screen
    /// widescreen one is the one a single player on a widescreen display gets.
    #[test]
    fn the_fullscreen_widescreen_placement_wins() {
        let mk = |res: u8, x: f32| Placement {
            res_flags: res,
            anchor: ANCHOR_CENTER,
            origin: [0.5, 0.5],
            offset: [x, 0.0],
            scale: [1.0, 1.0],
        };
        let b = |ps: Vec<Placement>| WidgetBase {
            name: "w".into(),
            sort: 0,
            placements: ps,
            active: Vec::new(),
            flash: Vec::new(),
            hidden: Vec::new(),
            color: None,
            output_a: 0,
            custom: [[1.0; 4]; 4],
            scalar_a: 0,
            anim_active: false,
            anim_flashing: false,
            shader_type: 0,
            blend_mode: 3,
        };
        // res 0b00100 is `halfscreen`, 0b00001 is `fullscreen wide`
        let w = b(vec![mk(0b00100, 1.0), mk(0b00001, 2.0), mk(0b00000, 3.0)]);
        assert_eq!(w.placement().map(|p| p.offset[0]), Some(2.0));
        // nothing marked fullscreen-wide: the first authored one, which is what the shipped tags
        // rely on (most author res flags 0 on every entry)
        let w = b(vec![mk(0b00000, 9.0), mk(0b00100, 1.0)]);
        assert_eq!(w.placement().map(|p| p.offset[0]), Some(9.0));
        assert!(b(Vec::new()).placement().is_none());
    }

    // ---- gates that need the shipped cache -------------------------------------------------

    /// **THE UNITS GATE**, and the one this whole file hangs on: are `weap`'s aim-assist angles
    /// stored in RADIANS or in DEGREES?
    ///
    /// The tool build's own field name says `autoaim angle:degrees`, but that is the EDITOR's
    /// display unit — every Halo `angle` compiles to radians in the tag. This asserts it against
    /// the shipped cache rather than taking either reading on trust, because the `phmo` joint
    /// limits in this same tree turned out the other way round and reading them as radians left
    /// every joint unlimited.
    ///
    /// Bounded on purpose: it reads at most 40 weapons' 60-byte block, so it cannot be the sweep
    /// that took the test binary to 15.7 GB.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn aim_assist_angles_are_radians_not_degrees() {
        let Some((native, cache)) = open_reach("forge_halo") else { return };
        let mut tags = native.tags_by_group(cache, "weap");
        tags.sort_unstable();
        let mut authored = 0usize;
        let mut widest = 0.0f32;
        let mut degree_reading_rejected = 0usize;
        for t in tags.into_iter().take(40) {
            let Some(a) = AimAssist::read(&native, cache, t) else { continue };
            if !a.is_authored() {
                continue;
            }
            authored += 1;
            let name = native.tag_name(cache, t).unwrap_or_default();
            eprintln!(
                "reach_hud: {:<40} autoaim {:.6} ({:.2} deg)  magnetism {:.6} ({:.2} deg)  range {:.1}",
                name.rsplit('\\').next().unwrap_or(&name),
                a.autoaim_angle,
                a.autoaim_angle.to_degrees(),
                a.magnetism_angle,
                a.magnetism_angle.to_degrees(),
                a.autoaim_range
            );
            widest = widest.max(a.autoaim_angle.max(a.magnetism_angle));
            // THE UNITS, both directions. The radian reading must land on the 0.05-degree
            // authoring grid and the degree reading must NOT -- a one-directional check would
            // pass on a value that happens to satisfy both.
            assert!(
                a.angles_are_radians(),
                "{name}: autoaim {} / magnetism {} / deviation {} do NOT land on the 0.05-degree \
                 authoring grid when read as radians ({:.4} / {:.4} / {:.4} deg), so either they \
                 are degrees -- making every angle in this file wrong by a factor of 57 -- or the \
                 offset is not the aim-assist block at all",
                a.autoaim_angle,
                a.magnetism_angle,
                a.deviation_angle,
                a.autoaim_angle.to_degrees(),
                a.magnetism_angle.to_degrees(),
                a.deviation_angle.to_degrees()
            );
            assert!(
                !a.angles_are_degrees(),
                "{name}: these angles land on the authoring grid read as DEGREES too, so this \
                 gate cannot tell the two readings apart and proves nothing"
            );
            degree_reading_rejected += 1;
            assert!(a.autoaim_range > 0.0 && a.autoaim_range < 1000.0, "{name}: autoaim range {}", a.autoaim_range);
        }
        assert!(authored >= 5, "only {authored} weapons authored an aim-assist block — the offset is wrong");
        assert_eq!(degree_reading_rejected, authored, "every authored block must have been checked both ways");
        eprintln!(
            "reach_hud: {authored} weapons authored an aim-assist block; all of them land on the \
             0.05-degree grid as RADIANS and none of them does as degrees. Widest cone {:.2} deg.",
            widest.to_degrees()
        );
        // The TARGETING cones (autoaim and magnetism, not the projectile `deviation angle`,
        // which the ghost authors at a legitimate 45 degrees) stay in single figures.
        assert!(
            widest.to_degrees() < 25.0,
            "the widest authored autoaim/magnetism cone is {:.1} degrees, which is not a targeting cone",
            widest.to_degrees()
        );
    }

    /// Open the shipped Reach cache, the same way every other module's cache gates do. Returns
    /// `None` — and says why — on a machine with no Reach install, so the suite never depends on
    /// one.
    fn open_reach(stem: &str) -> Option<(std::sync::Arc<NativeDll>, CacheHandle)> {
        static CACHE: std::sync::OnceLock<Option<(std::sync::Arc<NativeDll>, CacheHandle)>> =
            std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| {
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
                let native = std::sync::Arc::new(NativeDll::load(&dll).ok()?);
                let cands = crate::mapcat::enumerate();
                let map = cands.iter().find(|c| {
                    c.game == crate::mapcat::Game::Reach
                        && c.path.file_stem().is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case(stem))
                })?;
                native.set_shared_cache_dirs(&[map.path.parent()?.to_path_buf()]);
                let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
                Some((native, cache))
            })
            .clone()
    }

    /// **THE AUTHORED-HUD GATE.** The ammo block, the grenade block and the shield and health
    /// meters have to decode out of the shipped tags with the names this file draws by. A rename
    /// or a wrong offset turns the HUD into an empty screen, and this is what catches that.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_authored_hud_blocks_decode() {
        let Some((native, cache)) = open_reach("forge_halo") else { return };
        // the player HUD
        let t = chud_by_path(&native, cache, PLAYER_CHUD_SPARTAN).expect("ui\\chud\\spartan");
        let c = read_chud(&native, cache, t).expect("spartan chud decodes");
        eprintln!("reach_hud: {} has {} collections", c.path, c.collections.len());
        for want in PLAYER_COLLECTIONS {
            let coll = c.get(want).unwrap_or_else(|| panic!("ui\\chud\\spartan has no '{want}' collection"));
            eprintln!(
                "  {want:<14} anchor {} datasource {:?} widgets {}",
                coll.base.placement().map(|p| p.anchor).unwrap_or(255),
                coll.datasource,
                coll.widgets.len()
            );
            assert!(!coll.widgets.is_empty(), "'{want}' authors no widgets");
            assert!(coll.base.placement().is_some(), "'{want}' authors no placement");
            // EVERY BITMAP MUST RESOLVE TO A REAL SIZE. This is the gate that caught the shield
            // and health meters being dropped outright: their `bitm`s are plain bitmaps, not
            // sprite sheets, so the sprite walk answered `None` and `read_chud` skipped them --
            // 8 widgets decoded as 1 and 6 as 0. A zero size means the same failure has come
            // back in a form that no longer shows up as a missing widget.
            for w in &coll.widgets {
                if let WidgetArt::Bitmap { sprite, .. } = &w.art {
                    assert!(
                        sprite.size[0] > 0.5 && sprite.size[1] > 0.5,
                        "{want}/{}: bitmap resolved to {:?} virtual units",
                        w.base.name,
                        sprite.size
                    );
                    assert!(
                        sprite.size[0] < 2000.0 && sprite.size[1] < 2000.0,
                        "{want}/{}: bitmap is {:?} virtual units, far wider than the 1280x720 \
                         virtual screen -- the supersample flags were not applied",
                        w.base.name,
                        sprite.size
                    );
                }
            }
            eprintln!(
                "    bitmaps: {}",
                coll.widgets
                    .iter()
                    .filter_map(|w| match &w.art {
                        WidgetArt::Bitmap { sprite, .. } =>
                            Some(format!("{} {:.0}x{:.0}", w.base.name, sprite.size[0], sprite.size[1])),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        // the grenade block is the one datasource-driven collection HMS draws, and it has to be
        // the `grenades` datasource with room for the two types
        let g = c.get("grenades").unwrap();
        assert_eq!(g.datasource, Some(0), "the grenade block's datasource is `grenades`");
        assert_eq!(g.datasource_max, 2, "two grenade types");
        assert!(g.get("grenade_count_hilite").is_some(), "the selected type's count");
        // the shield meter: a meter plus a border, and the border is NOT a meter
        let s = c.get("shield_meter").unwrap();
        assert!(s.get("meter").is_some() && s.get("meter_border").is_some());

        // a weapon's own ammo block
        let wt = chud_by_path(&native, cache, "ui\\chud\\assault_rifle").expect("ui\\chud\\assault_rifle");
        let w = read_chud(&native, cache, wt).expect("assault rifle chud decodes");
        let ammo = w.get("ammo_area").expect("the AR authors an ammo_area");
        let sch = ammo.get("shematic").expect("the schematic glyph");
        match &sch.art {
            WidgetArt::Text { font, input, .. } => {
                eprintln!("  AR schematic: font {:?} string '{input}'", font.map(|f| f.file_stem()));
                assert_eq!(*font, Some(crate::reach_font::FontId::FullScreenHudMessage), "the schematic is a hud glyph");
                assert!(!input.is_empty() && input != CHUD_OUT_A, "the schematic is a LITERAL string id");
            }
            _ => panic!("the schematic is a text widget"),
        }
        let res = ammo.get("reserve_count").expect("the reserve count");
        match &res.art {
            WidgetArt::Text { input, flags, .. } => {
                assert_eq!(input, CHUD_OUT_A, "a runtime number");
                assert!(flags & TEXT_IS_NUMBER != 0, "and it is flagged as a number");
            }
            _ => panic!("the reserve count is a text widget"),
        }
        assert!(ammo.get("meter").is_some(), "the magazine meter");

        // and the crosshair collection is still there, so `paint_crosshair` has not been orphaned
        assert!(w.get("crosshair").is_some(), "the AR authors a crosshair collection");
    }

    /// **THE LAYOUT GATE — where the HUD actually lands on screen.**
    ///
    /// Everything else here checks that the tags DECODE. This checks that the decoded numbers put
    /// the HUD where Reach puts it, which is the one thing a reader of the code cannot verify by
    /// reading it and the one thing a wrong sign or a swapped axis destroys silently. Reach's own
    /// layout, from the shipped tags: the **ammo block top right**, the **grenades top left**, the
    /// **shield and health meters top centre**, and the **reticule dead centre of the viewport**
    /// (not of the safe frame).
    ///
    /// It resolves every widget of every block HMS draws against a 1920x1080 viewport and asserts
    /// the quadrant, so a flipped `origin offset`, an anchor resolved to the wrong corner, or a
    /// virtual-unit scale that has lost its factor all fail here rather than on screen.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_hud_lands_where_reach_puts_it() {
        let Some((native, cache)) = open_reach("forge_halo") else { return };
        let skin = crate::reach_chud::read_globals(&native, cache)
            .and_then(|g| g.default_skin().cloned())
            .expect("ui\\chud\\globals");
        let vp = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1920.0, 1080.0));
        let vs = VirtualScreen::from_skin(vp, &skin);
        eprintln!(
            "reach_hud: 1920x1080 -> safe frame {:.0}x{:.0} at ({:.0},{:.0}), one virtual unit = {:.3} pt",
            vs.safe.width(),
            vs.safe.height(),
            vs.safe.min.x,
            vs.safe.min.y,
            vs.vpx
        );
        assert!(vs.safe.width() < vp.width() && vs.safe.height() < vp.height(), "the safe frame insets");
        assert!((vs.safe.center() - vp.center()).length() < 0.01, "and it is centred");

        // Where every widget of a block ends up, as a fraction of the viewport.
        let spread = |a: &HudArt, coll: &str| -> Vec<(String, f32, f32)> {
            let Some(c) = a.chud.get(coll) else { return Vec::new() };
            let Some(cp) = c.base.placement() else { return Vec::new() };
            let Some((parent, _)) = widget_point(&vs, cp, None) else { return Vec::new() };
            c.widgets
                .iter()
                .filter_map(|w| {
                    let p = w.base.placement()?;
                    let (at, _) = widget_point(&vs, p, Some(parent))?;
                    Some((
                        w.base.name.clone(),
                        (at.x - vp.min.x) / vp.width(),
                        (at.y - vp.min.y) / vp.height(),
                    ))
                })
                .collect()
        };
        let ctx = egui::Context::default();

        // ---- the player HUD: grenades top LEFT, the two meters top CENTRE ------------------
        let pt = chud_by_path(&native, cache, PLAYER_CHUD_SPARTAN).expect("ui\\chud\\spartan");
        let pa = hud_art(&ctx, &native, cache, pt).expect("the player chud uploads");
        for (coll, want) in [("grenades", "left"), ("shield_meter", "centre"), ("health_meter", "centre")] {
            let ws = spread(&pa, coll);
            assert!(!ws.is_empty(), "'{coll}' resolved no widget positions");
            for (n, fx, fy) in &ws {
                eprintln!("reach_hud: {coll}/{n:<26} at {:.3}, {:.3} of the viewport", fx, fy);
                assert!(*fy < 0.45, "{coll}/{n} is at y {fy:.3} -- the whole top HUD must be in the TOP half");
                match want {
                    "left" => assert!(*fx < 0.35, "{coll}/{n} is at x {fx:.3}, not on the LEFT"),
                    _ => assert!(
                        (0.2..0.8).contains(fx),
                        "{coll}/{n} is at x {fx:.3}, not near the CENTRE"
                    ),
                }
            }
        }

        // ---- a weapon's ammo block: top RIGHT ---------------------------------------------
        let wt = crate::reach_chud::weapon_chud(
            &native,
            cache,
            native
                .tags_by_group(cache, "weap")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with("\\assault_rifle")))
                .expect("the assault rifle"),
        )
        .expect("the assault rifle references a chud interface");
        let wa = hud_art(&ctx, &native, cache, wt).expect("the weapon chud uploads");
        let ammo: Vec<_> = wa
            .chud
            .collections
            .iter()
            .filter(|c| c.base.name.starts_with(WEAPON_AMMO_PREFIX))
            .flat_map(|c| spread(&wa, &c.base.name))
            .collect();
        assert!(!ammo.is_empty(), "the assault rifle's ammo block resolved nothing");
        for (n, fx, fy) in &ammo {
            eprintln!("reach_hud: ammo/{n:<30} at {fx:.3}, {fy:.3} of the viewport");
            assert!(*fx > 0.6, "ammo/{n} is at x {fx:.3}, not on the RIGHT");
            assert!(*fy < 0.45, "ammo/{n} is at y {fy:.3}, not at the TOP");
            assert!(*fx < 1.0 && *fy > 0.0, "ammo/{n} is off screen at {fx:.3}, {fy:.3}");
        }

        // THE SIGN OF THE OFFSET, which a quadrant test cannot see. A block anchored to a corner
        // hangs INWARD from it: the ammo block is anchored top right and every child authors a
        // NEGATIVE x, so each child must sit LEFT of the block's own point, while the grenade
        // block is anchored top left and its icon and count author POSITIVE x and must sit RIGHT
        // of theirs. Flip the sign of `origin offset` and this is what fails -- the quadrant
        // assertions above all still pass, because the blocks merely move further into their own
        // corner.
        let parent_of = |a: &HudArt, coll: &str| -> egui::Pos2 {
            let c = a.chud.get(coll).unwrap_or_else(|| panic!("no '{coll}'"));
            widget_point(&vs, c.base.placement().expect("placement"), None).expect("resolves").0
        };
        let ammo_coll = wa
            .chud
            .collections
            .iter()
            .find(|c| c.base.name.starts_with(WEAPON_AMMO_PREFIX))
            .map(|c| c.base.name.clone())
            .expect("an ammo block");
        let ap = parent_of(&wa, &ammo_coll);
        for (n, fx, _) in spread(&wa, &ammo_coll) {
            let x = vp.min.x + fx * vp.width();
            assert!(
                x <= ap.x + 0.01,
                "ammo/{n} sits RIGHT of its own block point ({x:.1} > {:.1}) -- the x offset's \
                 sign is inverted, so the block hangs off the edge of the screen instead of \
                 inward from the corner",
                ap.x
            );
        }
        let gp = parent_of(&pa, "grenades");
        let gx: Vec<(String, f32)> = spread(&pa, "grenades")
            .into_iter()
            .filter(|(n, _, _)| n == "hilight" || n == "grenade_count_hilite")
            .map(|(n, fx, _)| (n, vp.min.x + fx * vp.width()))
            .collect();
        assert!(!gx.is_empty(), "the grenade block's selected icon and count did not resolve");
        for (n, x) in gx {
            assert!(
                x >= gp.x - 0.01,
                "grenades/{n} sits LEFT of its own block point ({x:.1} < {:.1}) -- the x offset's \
                 sign is inverted",
                gp.x
            );
        }

        // ---- the reticule: dead centre of the VIEWPORT -------------------------------------
        let xh = wa.chud.get("crosshair").expect("the AR authors a crosshair collection");
        for w in &xh.widgets {
            let Some(p) = w.base.placement() else { continue };
            assert_eq!(p.anchor, ANCHOR_CROSSHAIR, "a reticule part must use the crosshair anchor");
            let (at, _) = widget_point(&vs, p, None).expect("the crosshair anchor resolves");
            let off = (at - vp.center()).length();
            eprintln!("reach_hud: crosshair/{:<24} {:.1} pt off the viewport centre", w.base.name, off);
            // The spread bars sit 15 virtual units out; nothing is more than 40 units off.
            assert!(off < 40.0 * vs.vpx, "crosshair/{} is {off:.1} pt off centre", w.base.name);
        }

        // ---- the respawn countdown: CENTRED, and BELOW the middle --------------------------
        //
        // This line is the one that was missing, and its absence is why the widget shipped in the
        // wrong place: "the respawn thing seems to work BUT its in the wrong spot on the screen".
        // The slot is `spawning_on_you`, the one collection in any shipped `chdt` that the
        // engine's own spawn-status states drive, and it hangs from anchor 17 `crosshair` -- the
        // SCREEN CENTRE -- at (0,-53) with its text at (+2.6,+185.1), i.e. 132 virtual units
        // below the middle. The first attempt used `omniwidget_high_center`, a MEGALO script slot
        // anchored `top center`, which lands about a fifth of the way down.
        let mt = chud_by_path(&native, cache, MP_CHUD).expect("ui\\chud\\scoreboard");
        let ma = hud_art(&ctx, &native, cache, mt).expect("the multiplayer chud uploads");
        let slot = ma.chud.get(RESPAWN_SLOT).expect("the spawn-status slot");
        assert_eq!(
            slot.base.placement().map(|p| p.anchor),
            Some(ANCHOR_CROSSHAIR),
            "the spawn-status slot must hang from the screen centre, not from an edge"
        );
        let text = slot.get(RESPAWN_TEXT_WIDGET).expect("its text widget");
        let (parent, _) = widget_point(&vs, slot.base.placement().unwrap(), None).unwrap();
        let (at, _) = widget_point(&vs, text.base.placement().unwrap(), Some(parent)).unwrap();
        let (fx, fy) = ((at.x - vp.min.x) / vp.width(), (at.y - vp.min.y) / vp.height());
        eprintln!("reach_hud: respawn/{RESPAWN_TEXT_WIDGET:<22} at {fx:.3}, {fy:.3} of the viewport");
        assert!((0.45..0.55).contains(&fx), "the respawn countdown is at x {fx:.3}, not CENTRED");
        assert!(
            (0.55..0.80).contains(&fy),
            "the respawn countdown is at y {fy:.3} -- it belongs just BELOW the middle of the \
             screen, which is where the engine's own spawn-status slot sits"
        );
    }

    /// **THE DRAW GATE — what the painter actually emits.**
    ///
    /// Every other gate here checks a number on the way to the screen. This one runs the REAL
    /// painting code — [`paint_collection`], [`crate::reach_chud::paint_crosshair`],
    /// [`paint_respawn`] — into a real `egui::Context` and reads back the shapes it produced.
    ///
    /// It exists because a headless capture cannot see any of this: `shot` renders the 3D scene
    /// only, so the chud overlay lives solely in a window, and this session is not allowed to open
    /// one. Inspecting the draw commands is as close to looking at the picture as can be got
    /// without one, and it is strictly sharper than the arithmetic gates: it catches a widget that
    /// decodes and positions perfectly and then emits nothing, which is exactly how the crosshair
    /// managed to be "landed" while drawing nothing at all.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_painter_emits_the_hud() {
        let Some((native, cache)) = open_reach("forge_halo") else { return };
        let skin = crate::reach_chud::read_globals(&native, cache)
            .and_then(|g| g.default_skin().cloned())
            .expect("ui\\chud\\globals");
        let vp = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1920.0, 1080.0));
        let ctx = egui::Context::default();
        // A living, armed player: what the HUD rests on.
        let mut v = HudValues { weapon_tag: None, ..Default::default() };
        v.shield = Field::derived(1.0);
        v.health = Field::derived(1.0);

        // Run one frame and collect every shape the HUD emitted, with its bounding box.
        let run = |what: &str, draw: &mut dyn FnMut(&egui::Context, &egui::Painter, &VirtualScreen)| -> Vec<egui::Rect> {
            let mut draw = std::cell::RefCell::new(draw);
            let out = ctx.run(Default::default(), |ctx| {
                let painter =
                    ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new(what)));
                let vs = VirtualScreen::from_skin(vp, &skin);
                (draw.borrow_mut())(ctx, &painter, &vs);
            });
            let rects: Vec<egui::Rect> = out
                .shapes
                .iter()
                .map(|s| s.shape.visual_bounding_rect())
                .filter(|r| r.is_finite() && r.width() > 0.0 && r.height() > 0.0)
                .collect();
            eprintln!("reach_hud: {what} emitted {} drawable shape(s)", rects.len());
            rects
        };

        // ---- the player HUD: shield, health, grenades --------------------------------------
        let pt = chud_by_path(&native, cache, PLAYER_CHUD_SPARTAN).expect("ui\\chud\\spartan");
        let pa = hud_art(&ctx, &native, cache, pt).expect("the player chud uploads");
        for want in PLAYER_COLLECTIONS {
            let i = pa.chud.collections.iter().position(|c| c.base.name == want).expect(want);
            let reps = if pa.chud.collections[i].datasource == Some(DATASOURCE_GRENADES) { 2 } else { 1 };
            let mut total = 0usize;
            let rects = run(want, &mut |c, p, vs| {
                for r in 0..reps {
                    let (d, _) = paint_collection(c, p, vs, &pa, i, Some(&skin), &v, r);
                    total += d;
                }
            });
            assert!(!rects.is_empty(), "'{want}' drew NOTHING — it decodes and positions but emits no shape");
            let union = rects.iter().skip(1).fold(rects[0], |a, r| a.union(*r));
            eprintln!(
                "  {want:<14} {} shape(s) over {:.0}x{:.0} pt at ({:.0},{:.0})",
                rects.len(),
                union.width(),
                union.height(),
                union.min.x,
                union.min.y
            );
            assert!(total > 0, "'{want}' reported drawing no widgets");
            // On screen, and a sane size: a HUD block is tens to a few hundred points, never a
            // single pixel and never the whole window.
            assert!(vp.contains_rect(union.intersect(vp)), "'{want}' drew outside the viewport");
            assert!(union.width() > 8.0 && union.height() > 4.0, "'{want}' is {union:?} — degenerate");
            assert!(union.width() < vp.width() * 0.9, "'{want}' spans almost the whole window: {union:?}");
        }
        // The grenade block must draw its CONTENTS, not just its frame: all four of its widgets
        // are gated on `<grenade> selected` / `<grenade> unselected`, which only hold once the
        // per-repeat states are supplied. Without `repeat_states` this is 0.
        let gi = pa.chud.collections.iter().position(|c| c.base.name == "grenades").unwrap();
        let mut grenade_widgets = 0usize;
        run("grenades (both repeats)", &mut |c, p, vs| {
            for r in 0..2 {
                let (d, _) = paint_collection(c, p, vs, &pa, gi, Some(&skin), &v, r);
                grenade_widgets += d;
            }
        });
        assert!(
            grenade_widgets >= 2,
            "the grenade block drew {grenade_widgets} widget(s) — its contents are gated on the \
             per-repeat `<grenade> selected` states and they are not being supplied"
        );

        // ---- the crosshair, through its own painter ----------------------------------------
        for weapon in ["magnum", "assault_rifle", "dmr", "energy_sword"] {
            let t = native
                .tags_by_group(cache, "weap")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(&format!("\\{weapon}"))));
            let Some(t) = t else { continue };
            let Some(art) = crate::reach_chud::crosshair_for(&ctx, &native, cache, t) else {
                panic!("{weapon} decodes no reticule — the state evaluation has regressed")
            };
            let mut union = egui::Rect::NOTHING;
            let rects = run(weapon, &mut |_, p, vs| {
                union = crate::reach_chud::paint_crosshair(p, vp, &art, &skin, vs.vpx);
            });
            assert!(!rects.is_empty(), "{weapon}'s reticule emitted no shape");
            assert_eq!(rects.len(), art.crosshair.parts.len(), "{weapon}: one mesh per authored part");
            let off = (union.center() - vp.center()).length();
            eprintln!(
                "  {weapon:<14} {} part(s), {:.0}x{:.0} pt, centre {off:.1} pt off the viewport centre",
                art.crosshair.parts.len(),
                union.width(),
                union.height()
            );
            assert!(off < 30.0, "{weapon}'s reticule is {off:.1} pt off centre");
            assert!(union.width() > 10.0 && union.width() < 400.0, "{weapon}'s reticule is {union:?}");
        }

        // ---- the respawn countdown ----------------------------------------------------------
        let mt = chud_by_path(&native, cache, MP_CHUD).expect("ui\\chud\\scoreboard");
        let ma = hud_art(&ctx, &native, cache, mt).expect("the multiplayer chud uploads");
        let mut at = None;
        let rects = run("respawn countdown", &mut |c, p, vs| {
            at = paint_respawn(c, p, vs, &ma, Some(&skin), 2.4);
        });
        assert!(at.is_some(), "the respawn slot did not resolve");
        assert!(!rects.is_empty(), "the respawn countdown emitted no shape — the text did not draw");
        let union = rects.iter().skip(1).fold(rects[0], |a, r| a.union(*r));
        eprintln!("  respawn        {} shape(s) over {:?}", rects.len(), union);
        assert!(union.center().x > vp.width() * 0.3 && union.center().x < vp.width() * 0.7, "not centred: {union:?}");
        // BELOW the middle, which is where the engine's own spawn-status slot sits. This
        // assertion used to demand the UPPER half, and it passed happily on the wrong slot — "the
        // respawn thing seems to work BUT its in the wrong spot on the screen". It now rejects
        // exactly the placement that was shipped.
        assert!(
            union.center().y > vp.height() * 0.55,
            "the respawn countdown is at {union:?} — it belongs BELOW the middle of the screen, \
             not at the top"
        );
        assert!(union.max.y < vp.height() * 0.85, "...but not down with the ammo either: {union:?}");
        // 2.4 s left must print as "3" -- whole seconds still to wait.
        assert_eq!(RespawnClock::displayed(2.4), 3);

        // ---- WHICH FONT THE TEXT ACTUALLY DREW WITH ----------------------------------------
        //
        // This is the one question behind "the names are too large and seemingly bold". It cannot
        // be answered from a screenshot in this session -- a headless `shot` renders the 3D scene
        // only, so the whole chud overlay is absent from it (measured: three captures at one tick
        // with the name tags set to `always`, `always` and `hidden` came back byte-identical) --
        // and opening a window is not allowed here. So it is answered from the DRAW COMMANDS,
        // which is sharper anyway.
        //
        // The discriminator is exact and needs no pixel: Reach's own atlas paints a string as ONE
        // textured `Shape::Mesh` (`FontAtlas::paint` builds a mesh of glyph quads), while the
        // no-Reach-font fallback lays a galley out and emits `Shape::Text`. So if any text shape
        // here is `Shape::Text`, the game's font was NOT used.
        let hud_font = crate::reach_font::hud_atlas(&ctx);
        let num_font = crate::reach_font::atlas_for(&ctx, crate::reach_font::FontId::HudNumber);
        eprintln!(
            "reach_hud: hud_atlas = {:?}, hud_number = {:?}",
            hud_font.as_ref().map(|f| f.font.name.clone()),
            num_font.as_ref().map(|f| f.font.name.clone())
        );
        if hud_font.is_none() {
            eprintln!("reach_hud: no Reach fonts on this machine — the text falls back to egui's own");
            return;
        }
        let ammo_art = crate::reach_chud::weapon_chud(
            &native,
            cache,
            native
                .tags_by_group(cache, "weap")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with("\\assault_rifle")))
                .expect("the assault rifle"),
        )
        .and_then(|t| hud_art(&ctx, &native, cache, t))
        .expect("the assault rifle's chud");
        let out = ctx.run(Default::default(), |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("fontcheck")));
            let vs = VirtualScreen::from_skin(vp, &skin);
            paint_respawn(ctx, &painter, &vs, &ma, Some(&skin), 2.4);
            // ...and a weapon's schematic glyph, which is the other string the HUD draws.
            if let Some(i) = ammo_art.chud.collections.iter().position(|c| c.base.name.starts_with(WEAPON_AMMO_PREFIX))
            {
                paint_collection(ctx, &painter, &vs, &ammo_art, i, Some(&skin), &v, 0);
            }
        });
        let mut meshes = 0usize;
        for s in &out.shapes {
            match &s.shape {
                egui::Shape::Mesh(_) => meshes += 1,
                egui::Shape::Text(t) => panic!(
                    "the HUD drew a string as an egui galley ({:?}) rather than as a mesh of \
                     Reach's own glyphs — `reach_font::atlas_for` returned None at DRAW time even \
                     though the font decodes, so every string on this HUD is in the app's font, \
                     not the game's",
                    t.galley.text()
                ),
                _ => {}
            }
        }
        assert!(meshes > 0, "no text or sprite meshes at all");
        eprintln!("reach_hud: every HUD string drew as a Reach-font MESH ({meshes} mesh shapes, 0 egui galleys)");

        // ---- AND THE NAME TAGS, which is where the complaint started ------------------------
        //
        // "The names are STILL way too large and seemingly bold, but in game the letters are
        // thinner for sure." A bold-looking sans is exactly what egui's proportional fallback
        // draws, so the first question is simply whether `paint_name_tags` is taking it. Same
        // discriminator: Reach's atlas emits a `Shape::Mesh`, the fallback a `Shape::Text`.
        let tag = crate::film_ui::NameTag {
            text: "Sopitive".into(),
            pos: [0.0, 0.0, 0.0],
            named: true,
            inferred_from: None,
            team: 0,
            net_id: 1,
            feet: [0.0, 0.0, 0.0],
            on_marker: true,
            slot: Some(0),
            show_icon: false,
            color_index: 0,
        };
        let rules = crate::film_ui::NamePlateRules::from_chud(&skin);
        let out = ctx.run(Default::default(), |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("nametag")));
            crate::film_ui::paint_name_tags(
                ctx,
                &painter,
                vp,
                std::slice::from_ref(&tag),
                &rules,
                None,
                glam::vec3(0.0, -20.0, 0.0),
                &|_| Some((vp.center().x, vp.center().y)),
            );
        });
        let mut tag_meshes = 0usize;
        for s in &out.shapes {
            match &s.shape {
                egui::Shape::Mesh(_) => tag_meshes += 1,
                egui::Shape::Text(t) => panic!(
                    "the name tag drew as an egui galley ({:?}) rather than Reach's own glyphs — \
                     THIS is the \"too large and bold\" font, and it means `reach_font::hud_atlas` \
                     returned None at draw time",
                    t.galley.text()
                ),
                _ => {}
            }
        }
        assert_eq!(tag_meshes, 1, "one name, one mesh");
        // ...and at the authored SIZE: 21 virtual units of line height at the engine's own
        // distance scale, which at 1080 lines is about 27 points, not 50.
        let f = hud_font.as_ref().unwrap();
        let scale = crate::film_ui::nameplate_scale(20.0, &rules) * rules.px_per_virtual(vp.height());
        let drawn = f.font.line_height() as f32 * scale;
        eprintln!(
            "reach_hud: a name at 20 wu draws {:.1} pt tall in '{}' (ascender {} + descender {} = {} virtual units)",
            drawn,
            f.font.name,
            f.font.ascender,
            f.font.descender,
            f.font.line_height()
        );
        assert!((15.0..45.0).contains(&drawn), "a 1080-line name tag is {drawn:.1} pt tall");
    }

    /// **THE COLOUR GATE.** Every HUD piece HMS draws must resolve to a colour the TAG authors,
    /// and no piece may resolve to one it does not.
    ///
    /// This is the gate the colour bugs got past. Three separate faults shipped: every non-skin
    /// output read as custom A (so a widget authoring its colour in B/C/D drew the wrong one), an
    /// UNSET custom A drawn as opaque black (the shield and health borders), and the player HUD's
    /// effect overlays — damage flashes, the boot blob, the compass — painted at full strength
    /// because nothing in the tag refuses them.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn every_drawn_hud_piece_has_an_authored_colour() {
        let Some((native, cache)) = open_reach("forge_halo") else { return };
        let skin = crate::reach_chud::read_globals(&native, cache)
            .and_then(|g| g.default_skin().cloned())
            .expect("ui\\chud\\globals");
        let t = chud_by_path(&native, cache, PLAYER_CHUD_SPARTAN).expect("ui\\chud\\spartan");
        let chud = read_chud(&native, cache, t).expect("the player chud");
        let mut drawn = 0usize;
        for coll in chud.collections.iter().filter(|c| PLAYER_COLLECTIONS.contains(&c.base.name.as_str())) {
            let mut rest = RESTING_STATES.to_vec();
            rest.extend(repeat_states(coll.datasource, 0));
            for w in &coll.widgets {
                let keep = is_drawn_player_widget(&coll.base.name, &w.base.name)
                    && (w.base.color.is_some() || w.base.custom.iter().any(|c| c.iter().any(|v| *v > 0.0)));
                let col = widget_color(&w.base, Some(&skin));
                eprintln!(
                    "reach_hud: {:<14} {:<24} {} #{:02X}{:02X}{:02X}",
                    coll.base.name,
                    w.base.name,
                    if keep { "DRAWN " } else { "skipped" },
                    col.r(),
                    col.g(),
                    col.b()
                );
                if !keep {
                    continue;
                }
                drawn += 1;
                // Never BLACK. An unauthored custom A is all-zero, and reading it literally is
                // what painted the meter borders black.
                assert!(
                    col != egui::Color32::BLACK,
                    "{}/{} draws BLACK — an unset colour slot is being taken literally",
                    coll.base.name,
                    w.base.name
                );
                // Never the fallback WHITE either: that means nothing was authored anywhere, and
                // such a widget must have been skipped rather than tinted.
                assert!(
                    col != egui::Color32::WHITE,
                    "{}/{} draws WHITE — it authors no colour at all and should not be drawn",
                    coll.base.name,
                    w.base.name
                );
            }
        }
        assert!(drawn >= 4, "only {drawn} player-HUD widgets draw — the whitelist has gone stale");
        // And the effect overlays specifically must NOT be in it, by name, because those are the
        // ones the user saw: a solid red blob on the shield and two white plates on the health.
        for bad in ["flahsy", "shield_boot_blob", "meter_border_flash", "compass", "compass_additive"] {
            assert!(!PLAYER_WIDGETS.contains(&bad), "{bad} is an effect overlay HMS cannot drive");
        }
        // The shield border must keep the blue it authors in custom B, which is the fix for the
        // black border and the thing a regression would undo.
        let sb = chud.get("shield_meter").unwrap().get("meter_border").unwrap();
        let c = widget_color(&sb.base, Some(&skin));
        eprintln!("reach_hud: the shield border resolves to #{:02X}{:02X}{:02X}", c.r(), c.g(), c.b());
        assert!(c.b() > c.r() && c.b() > 100, "the shield border is not blue: {c:?}");
    }

    /// **THE FONT GATE.** The whole "the names are too large and bold" complaint is one
    /// question: does this machine's Reach font actually get used? This answers it without a GPU
    /// — the search is a filesystem walk and the decode is pure CPU — and it also pins the thing
    /// that can defeat a decoded font at DRAW time: `FontAtlas` packs every glyph into ONE row,
    /// and a strip wider than a GPU will allocate silently draws nothing.
    #[test]
    fn the_reach_hud_font_resolves_and_fits_one_texture() {
        let Some(src) = crate::reach_font::find_font_source(None) else {
            eprintln!("reach_hud: no Reach font source on this machine — the name tags fall back to egui's own font");
            return;
        };
        eprintln!("reach_hud: font source = {}", src.display());
        for id in [
            crate::reach_font::FontId::FullScreenHudMessage,
            crate::reach_font::FontId::HudNumber,
            crate::reach_font::FontId::SplitScreenHudMessage,
        ] {
            let f = src.load(id).unwrap_or_else(|| panic!("{} did not decode", id.file_stem()));
            // `FontAtlas::build`'s own packing, recomputed through the public accessor: one row,
            // one pixel of padding per glyph.
            let pad = 1usize;
            let glyphs: Vec<_> = (0u32..0x1_0000).filter_map(|cp| f.glyph(cp)).collect();
            let w: usize = glyphs.iter().map(|g| g.w as usize + pad).sum::<usize>() + pad;
            let h = glyphs.iter().map(|g| g.h as usize).max().unwrap_or(1) + pad * 2;
            eprintln!(
                "reach_hud: {:<22} {} glyphs, ascender {} descender {}, atlas strip {w} x {h}",
                id.file_stem(),
                glyphs.len(),
                f.ascender,
                f.descender
            );
            assert!(glyphs.len() > 60, "{}: only {} glyphs", id.file_stem(), glyphs.len());
            // wgpu's own default `max_texture_dimension_2d`. A one-row atlas wider than this is
            // rejected at upload and every glyph drawn from it is invisible — which looks exactly
            // like "the font did not load" while `hud_atlas` happily returns `Some`.
            assert!(
                w <= 8192,
                "{}: a one-row glyph atlas is {w} px wide, past wgpu's 8192 limit — it will not upload, \
                 and the HUD text will draw nothing at all",
                id.file_stem()
            );
        }
    }
}

/// #reach-strings  THE THREE OUTCOMES for an authored `string_id` on a chud text widget.
///
/// Theater draws the first and refuses the other two, for different reasons, and the reasons are
/// the point: before the localized-key convention was recovered, all three were one bucket called
/// "HMS has no string reader", and the weapon schematic once printed the words "assault_rifle"
/// across the ammo block.
///
/// The classification is derived from `reach_strings::substitute_glyphs` rather than restated, so
/// it cannot drift from the function that does the substituting.
#[cfg(test)]
mod chud_text_outcome_gates {
    use crate::reach_strings::{substitute_glyphs, GLYPH_ACTION};

    /// What `paint_collection` decides, as a pure function of the resolved text.
    #[derive(Debug, PartialEq)]
    enum Outcome {
        Draw(String),
        /// Nothing but glyphs: it IS an icon, and the icon font is MCC's.
        Icon(usize),
        /// Words with a button glyph in them: right for an interactive mode, wrong over a film.
        ButtonPrompt,
    }

    fn classify(resolved: &str) -> Outcome {
        let (bare, glyphs) = substitute_glyphs(resolved, "");
        if bare.trim().is_empty() {
            return Outcome::Icon(glyphs.len());
        }
        if !glyphs.is_empty() {
            return Outcome::ButtonPrompt;
        }
        Outcome::Draw(resolved.to_string())
    }

    #[test]
    fn plain_text_is_drawn_and_a_glyph_is_not() {
        // 1. Ordinary words -> drawn, unchanged.
        assert_eq!(classify("Reloading"), Outcome::Draw("Reloading".into()));
        assert_eq!(classify("12"), Outcome::Draw("12".into()));

        // 2. The weapon schematic: one private-use codepoint and NOTHING else. Drawing a
        //    substituted label here would put "[?]" where the weapon's outline belongs.
        assert_eq!(classify("\u{E203}"), Outcome::Icon(1));
        assert_eq!(classify("  \u{E203}\u{E204} "), Outcome::Icon(2), "whitespace is not words");

        // 3. A real vehicle prompt, verbatim from the shipped strings. Correct for test mode with
        //    the player's own key substituted; wrong over a recording, where there is no button.
        let prompt = format!("Hold {GLYPH_ACTION} to drive Warthog");
        assert_eq!(classify(&prompt), Outcome::ButtonPrompt);
        // The detach prompts carry a SECOND glyph (melee, U+E448) -- a bool would have hidden it.
        assert_eq!(classify("Press \u{E448} to detach turret"), Outcome::ButtonPrompt);
    }

    /// The boundary that decides outcome 2 against outcome 3 is "are there any words left", and
    /// it has to be exactly that. A text of one glyph plus one letter is a prompt, not an icon.
    #[test]
    fn one_letter_is_enough_to_make_it_a_prompt_not_an_icon() {
        assert_eq!(classify(&format!("{GLYPH_ACTION}x")), Outcome::ButtonPrompt);
        assert_eq!(classify(&format!("{GLYPH_ACTION}")), Outcome::Icon(1));
        // ...and an empty string is neither drawable nor a prompt: it is nothing to draw.
        assert_eq!(classify(""), Outcome::Icon(0));
    }
}
