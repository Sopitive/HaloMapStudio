//! #reach-chud  Halo: Reach's HUD authoring, decoded from `chud_globals_definition` (`chgd`).
//!
//! Everything the floating player name over a head looks like is authored in one place: the
//! `chgd` tag's `skins` block. The engine never holds a second copy of those numbers — the
//! per-local-player runtime HUD state IS a `chud_skin_info_block` element, 1332 bytes
//! (`0x14D` dwords), which is why every offset the disassembly uses is a *tag* offset:
//!
//! ```text
//! [RE] haloreach.dll+0x2E36B8  chud_color_from_index(index, out[4])
//!        index 0 -> skin+0x074   1 -> +0x084   2 -> +0x094   3 -> +0x0A4   4 -> +0x0B4
//!              5 -> skin+0x184   6 -> +0x194   7 -> +0x1A4   8 -> +0x1B4
//!              9 -> skin+0x0C4  10 -> +0x0D4  11 -> +0x0E4   else -> 0
//! [RE] haloreach.dll+0x2CB01C  scale = min(skin+0x48C, max(20.0 / distance, skin+0x488))
//! ```
//!
//! and `reverseme tagdefs reach_tag_test.exe chgd` names those offsets exactly:
//! `+0x074 navpt normal`, `+0x0A4 navpt enemy`, `+0x488 waypoint minimum distance scale`,
//! `+0x48C waypoint maximum distance scale`. So [`SKIN_COLORS`] below is the tool build's own
//! field list, not a guess, and [`color_from_index`] is the engine's own switch.
//!
//! LAYOUT (`chgd`, 528 bytes, from `reverseme tagdefs reach_tag_test.exe chgd`):
//!
//! ```text
//! chud_globals_definition
//!   +0x000 block  skins -> chud_skin_info_block [max 3]   (element 1332 = 0x534 bytes)
//!            +0x000 long enum  type -> chud_skin_type_enum  (0 default / 1 dervish / 2 monitor)
//!            +0x004 .. +0x203  THIRTY-TWO real_argb_color  (16 B each: alpha, red, green, blue)
//!            +0x3C0 tag reference  navpoints reference     (the chdt that draws them)
//!            +0x400 tag reference  saved-film reference
//!            +0x488 real  waypoint minimum distance scale
//!            +0x48C real  waypoint maximum distance scale
//!   +0x018 block  player training data
//! ```
//!
//! `chud_skin_type_enum` is `default / dervish / monitor` — a Halo 3 holdover. It is NOT
//! Spartan-vs-Elite and it is NOT per team, so there is exactly ONE authored navpoint palette in
//! a multiplayer game however many skins the tag carries.

use hms_native::{CacheHandle, NativeDll};

/// One `chud_skin_info_block` element is this many bytes — and the engine's per-local-player HUD
/// state is one of these, which is why `0x14D * 4` turns up all over the disassembly.
pub const SKIN_ELEMENT: usize = 1332;

/// The thirty-two `real_argb_color` fields inside a skin, in the tag's own order, as
/// `reverseme tagdefs reach_tag_test.exe chgd` prints them. Index `i` sits at
/// `skin + 0x04 + i * 0x10`.
pub const SKIN_COLORS: [&str; 32] = [
    "primary background",
    "secondary background",
    "highlight background",
    "flash background",
    "crosshair normal",
    "crosshair enemy",
    "crosshair friendly",
    "navpt normal",
    "navpt ally",
    "navpt neutral",
    "navpt enemy",
    "navpt dead",
    "navpt laser point unlocked",
    "navpt laser point locked",
    "navpt text",
    "msg flash self",
    "msg flash friend",
    "msg flash enemy",
    "msg flash neutral",
    "invincible shield",
    "player navpoint (firing)",
    "player navpoint (taking damage)",
    "player navpoint (speaking)",
    "player navpoint (coop spawning)",
    "custom 0",
    "custom 1",
    "custom 2",
    "neutral territory",
    "fire team triangle (close)",
    "fire team triangle (far)",
    "fire team triangle border (close)",
    "fire team triangle border (far)",
];

/// Byte offset of skin colour `i` inside a skin element.
pub const fn skin_color_offset(i: usize) -> usize {
    0x04 + i * 0x10
}

/// The colour a navpoint record's `color index` field selects, as the engine's own switch
/// (`haloreach.dll+0x2E36B8`) does it: a dense 0..11 index onto NON-contiguous skin colours.
///
/// Returns the index into [`SKIN_COLORS`], or `None` for an out-of-range index (which the engine
/// answers with a zeroed colour).
pub const fn color_from_index(index: i32) -> Option<usize> {
    // offset -> SKIN_COLORS index: 0x74 = 7, 0x84 = 8, 0x94 = 9, 0xA4 = 10, 0xB4 = 11,
    // 0x184 = 24, 0x194 = 25, 0x1A4 = 26, 0x1B4 = 27, 0xC4 = 12, 0xD4 = 13, 0xE4 = 14.
    match index {
        0 => Some(7),   // skin+0x074  navpt normal
        1 => Some(8),   // skin+0x084  navpt ally
        2 => Some(9),   // skin+0x094  navpt neutral
        3 => Some(10),  // skin+0x0A4  navpt enemy
        4 => Some(11),  // skin+0x0B4  navpt dead
        5 => Some(24),  // skin+0x184  custom 0
        6 => Some(25),  // skin+0x194  custom 1
        7 => Some(26),  // skin+0x1A4  custom 2
        8 => Some(27),  // skin+0x1B4  neutral territory
        9 => Some(12),  // skin+0x0C4  navpt laser point unlocked
        10 => Some(13), // skin+0x0D4  navpt laser point locked
        11 => Some(14), // skin+0x0E4  navpt text
        _ => None,
    }
}

/// The colour a navpoint record's STATE selects, overriding its colour index.
///
/// `[RE] haloreach.dll+0x2E3AD8` (the TEXT colour) switches on the record's state at +0x7C:
/// 1 -> skin+0x144, 2 -> skin+0x154, 3 -> skin+0x164, anything else -> the colour index.
/// `+0x2E39D4` (the ICON colour) adds 8 -> skin+0x174 and has no override.
pub const fn color_from_state(state: i32) -> Option<usize> {
    match state {
        1 => Some(20), // player navpoint (firing)
        2 => Some(21), // player navpoint (taking damage)
        3 => Some(22), // player navpoint (speaking)
        _ => None,
    }
}

/// `[RE] haloreach.dll+0x2CB76C` builds a teammate's name record with THIS colour index,
/// unconditionally — the team is used only to decide who gets a record at all.
pub const PLAYER_NAME_COLOR_INDEX: i32 = 0;

/// The index into [`SKIN_COLORS`] that [`PLAYER_NAME_COLOR_INDEX`] resolves to.
pub const PLAYER_NAME_COLOR: usize = 7; // navpt normal

/// HMS's "no team known" sentinel, the same one `film_ui::NameTag::team` uses.
pub const TEAM_NONE: u8 = 0xFF;

/// The engine's own team comparison for a navpoint over a PLAYER, as a colour index for
/// [`color_from_index`].
///
/// `[RE] haloreach.dll+0x14844C` — the one place in the whole navpoint system that compares two
/// teams. With `v2` the viewer's team and `v10` the subject's (8 = no team at all):
///
/// ```text
/// v11   = (v2 == 8 || v10 == 8) ? 0 : ((v2 == -1 || v10 == -1) ? 2 : (v2 != v10) + 1)
/// index = (v11 == 0) ? 2 : (v11 == 1 ? 0 : 3)
/// ```
///
/// which is: **same team -> 0 `navpt normal`; different team -> 3 `navpt enemy`; no team on
/// either side -> 2 `navpt neutral`.** A free-for-all is therefore ALL `navpt enemy`, because
/// Reach gives every player in a teams-off game their own team index.
pub fn navpoint_color_index(viewer_team: u8, subject_team: u8) -> i32 {
    if viewer_team == TEAM_NONE || subject_team == TEAM_NONE {
        2 // navpt neutral
    } else if viewer_team == subject_team {
        0 // navpt normal
    } else {
        3 // navpt enemy
    }
}

/// `[RE] haloreach.dll+0x2CB76C`: the marker the engine resolves on the unit's render model to
/// anchor the floating name, passed to `Object_BuildWorldTransform` as a `string_id` literal.
///
/// `[RE] haloreach.dll+0x489FD4`: `Object_BuildWorldTransform(unit, 0xC2, out, 1, 0, 1)` and the
/// caller reads the resulting matrix's translation.
pub const NAME_ANCHOR_MARKER_SID: u32 = 0xC2;

/// `[RE] haloreach.dll+0x2CB76C`: after the marker is resolved the engine lifts the record's z by
/// this, in world units — the only constant in the whole path (`0.075000003f`).
pub const NAME_ANCHOR_UP_WU: f32 = 0.075;

/// `[RE] haloreach.dll+0x18AE04` lifts the LOOK-AT record by this instead — the literal
/// `0.0500000007f`, 0.025 wu lower than the ally waypoint's.
///
/// The marker is the same one: that filler calls `Object_GetWorldPositionAndOrientation(unit, …)`
/// (`+0x48A06C`), which is `Object_BuildWorldTransform(unit, 0xC2, …)` — the identical `head`
/// marker the waypoint producer resolves through `sub_180489FD4`. Only the lift differs.
pub const LOOKAT_ANCHOR_UP_WU: f32 = 0.05;

/// One entry of a skin's `curvature infos` block (skin+0x2DC, 268-byte elements) — the authored
/// VIRTUAL SCREEN the whole HUD is laid out in, per resolution.
///
/// `chud_curvature_res_flags` (+0x000): bit 0 `fullscreen wide` (720p fullscreen), 1 `fullscreen
/// standard`, 2 `halfscreen`, 3 `quarterscreen wide`, 4 `quarterscreen standard`. `virtual width`
/// is a long at +0x094 and `virtual height` a long at +0x098
/// (`reverseme tagdefs reach_tag_test.exe chgd`).
#[derive(Clone, Copy, Debug)]
pub struct ChudCurvature {
    /// `res flags` — which resolution/window layouts this entry describes.
    pub res_flags: u8,
    pub virtual_width: i32,
    pub virtual_height: i32,
    /// `global safe frame horiz.` / `vert.` (+0x0D0 / +0x0D4), as a fraction.
    pub safe_frame: [f32; 2],
    /// `safe frame horizontal ding` / `vertical ding` (+0x0D8 / +0x0DC), in 1080-line pixels.
    pub ding: [f32; 2],
}

/// `chud_curvature_res_flags` bit 0 — the 16:9 full-screen authoring, which is the one a single
/// player on a widescreen display gets and the only one HMS has a use for.
pub const RES_FULLSCREEN_WIDE: u8 = 1 << 0;

/// Reach authors its HUD for a 720-line virtual screen; this is the fallback when the tag cannot
/// be read, and the shipped `ui\chud\globals` authors exactly this (asserted by the cache gate).
pub const VIRTUAL_HEIGHT_FALLBACK: f32 = 720.0;
/// `global safe frame vert.` as the shipped `ui\chud\globals` authors it, for the no-cache path.
pub const SAFE_FRAME_V_FALLBACK: f32 = 0.870;

/// How many real pixels one chud VIRTUAL unit is, vertically, in a window `window_h` pixels tall.
///
/// `[RE]` The chud never converts a size on the CPU: `sub_1802E62D8` hands the GPU vertices in
/// virtual units and `sub_1802D8720` hands the shader `{realW, realH, virtualW, virtualH}` plus
/// the safe-frame rect `{offX, halfW, offY, halfH}`, so virtual maps linearly onto the SAFE FRAME:
/// `px_per_virtual_y = safeFrameHeightPx / virtualHeight`. The safe frame itself is
/// `realH * fraction` inset by the authored "ding" scaled by `realH / 1080`
/// (`sub_1802D94BC` / `sub_1802D846C`).
///
/// This is why an authored 15-point font is not 15 screen pixels: at 1080p with Reach's own
/// 720-line virtual screen and 0.87 safe frame it is `15 * 1080 * 0.87 / 720` = 19.6 px, and at
/// 1440p it is 26.1. A HUD size that ignores the window is wrong at every size but one.
pub fn px_per_virtual(window_h: f32, virtual_height: f32, safe_frame_v: f32, ding_v: f32) -> f32 {
    if window_h <= 0.0 || virtual_height <= 1.0 {
        return 1.0;
    }
    let safe_px = window_h * safe_frame_v - 2.0 * ding_v * (window_h / 1080.0);
    (safe_px / virtual_height).max(0.01)
}

/// One decoded skin: the whole authored palette plus the two waypoint scale clamps.
#[derive(Clone, Debug)]
pub struct ChudSkin {
    /// `chud_skin_type_enum`: 0 default, 1 dervish, 2 monitor.
    pub skin_type: i32,
    /// The thirty-two `real_argb_color`s, each `[a, r, g, b]` in the tag's own order.
    pub colors: [[f32; 4]; 32],
    /// `waypoint minimum distance scale`, skin +0x488.
    pub waypoint_min_scale: f32,
    /// `waypoint maximum distance scale`, skin +0x48C.
    pub waypoint_max_scale: f32,
    /// `navpoints reference` (+0x3C0) — the `chdt` that draws them. Tag index, for provenance.
    pub navpoints_tag: Option<u32>,
    /// `saved-film reference` (+0x400) — Theater's own `chdt`.
    pub saved_film_tag: Option<u32>,
    /// `curvature infos` (+0x2DC) — the authored virtual screen per resolution.
    pub curvatures: Vec<ChudCurvature>,
}

impl ChudSkin {
    /// A colour by its name in [`SKIN_COLORS`].
    pub fn color(&self, name: &str) -> Option<[f32; 4]> {
        SKIN_COLORS.iter().position(|n| *n == name).map(|i| self.colors[i])
    }
    /// The colour the engine gives a plain teammate's floating name.
    pub fn player_name_color(&self) -> [f32; 4] {
        self.colors[PLAYER_NAME_COLOR]
    }
    /// The `curvature infos` entry a full-screen widescreen window uses.
    pub fn fullscreen_wide(&self) -> Option<&ChudCurvature> {
        self.curvatures
            .iter()
            .find(|c| c.res_flags & RES_FULLSCREEN_WIDE != 0)
            .or_else(|| self.curvatures.first())
    }
    /// The virtual-screen HEIGHT the HUD is authored against for a full-screen widescreen
    /// window, in the tag's own units. Every authored pixel size — a font's included — is a size
    /// in THIS space, which the engine then scales to the framebuffer.
    pub fn virtual_height(&self) -> f32 {
        self.fullscreen_wide().map(|c| c.virtual_height as f32).filter(|h| *h > 1.0).unwrap_or(VIRTUAL_HEIGHT_FALLBACK)
    }
    /// `global safe frame vert.`, the fraction of the real window the HUD is mapped onto.
    pub fn safe_frame_v(&self) -> f32 {
        self.fullscreen_wide().map(|c| c.safe_frame[1]).filter(|f| *f > 0.01).unwrap_or(SAFE_FRAME_V_FALLBACK)
    }
    /// `safe frame vertical ding`, an extra inset the engine scales by `realHeight / 1080`.
    pub fn safe_frame_ding_v(&self) -> f32 {
        self.fullscreen_wide().map(|c| c.ding[1]).unwrap_or(0.0)
    }
    pub fn type_name(&self) -> &'static str {
        match self.skin_type {
            0 => "default",
            1 => "dervish",
            2 => "monitor",
            _ => "<unknown>",
        }
    }
}

/// The open cache's `chgd`, decoded.
#[derive(Clone, Debug)]
pub struct ChudGlobals {
    pub path: String,
    pub skins: Vec<ChudSkin>,
}

impl ChudGlobals {
    /// The skin a multiplayer game draws with: `chud_skin_type_enum` 0 `default`.
    ///
    /// `[RE] haloreach.dll+0x2E36B8` indexes the skins array by a per-local-player index
    /// (`+0x1EAC` of the HUD player state); the enum is `default / dervish / monitor`, so a
    /// Spartan or Elite in a multiplayer game is `default` either way.
    pub fn default_skin(&self) -> Option<&ChudSkin> {
        self.skins.iter().find(|s| s.skin_type == 0).or_else(|| self.skins.first())
    }
}

fn rd_i32(b: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

/// Read a tag block's payload: `{i32 count, u32 address, u32 definition}` at `off` inside `buf`.
fn block(native: &NativeDll, cache: CacheHandle, buf: &[u8], off: usize, elem: usize, max: i32) -> Vec<u8> {
    let (Some(n), Some(ptr)) = (rd_i32(buf, off), rd_u32(buf, off + 4)) else { return Vec::new() };
    if n <= 0 || n > max || ptr == 0 || ptr == u32::MAX {
        return Vec::new();
    }
    native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
}

/// A tag reference's tag index, from the 16-byte reference at `off`.
fn tag_ref(buf: &[u8], off: usize) -> Option<u32> {
    let datum = rd_u32(buf, off + 0x0C)?;
    (datum != u32::MAX).then_some(datum & 0xFFFF)
}

/// Decode the open cache's `chud_globals_definition`.
pub fn read_globals(native: &NativeDll, cache: CacheHandle) -> Option<ChudGlobals> {
    let tag = native.tags_by_group(cache, "chgd").first().copied()?;
    let meta = native.tag_meta(cache, tag, 0, 0x210)?;
    let skins_raw = block(native, cache, &meta, 0x00, SKIN_ELEMENT, 3);
    let mut skins = Vec::new();
    for s in skins_raw.chunks_exact(SKIN_ELEMENT) {
        let mut colors = [[0.0f32; 4]; 32];
        for (i, c) in colors.iter_mut().enumerate() {
            let o = skin_color_offset(i);
            *c = [
                rd_f32(s, o).unwrap_or(0.0),
                rd_f32(s, o + 4).unwrap_or(0.0),
                rd_f32(s, o + 8).unwrap_or(0.0),
                rd_f32(s, o + 12).unwrap_or(0.0),
            ];
        }
        let curvatures = block(native, cache, s, 0x2DC, 268, 64)
            .chunks_exact(268)
            .map(|c| ChudCurvature {
                res_flags: c[0],
                virtual_width: rd_i32(c, 0x094).unwrap_or(0),
                virtual_height: rd_i32(c, 0x098).unwrap_or(0),
                safe_frame: [rd_f32(c, 0x0D0).unwrap_or(0.0), rd_f32(c, 0x0D4).unwrap_or(0.0)],
                ding: [rd_f32(c, 0x0D8).unwrap_or(0.0), rd_f32(c, 0x0DC).unwrap_or(0.0)],
            })
            .collect();
        skins.push(ChudSkin {
            skin_type: rd_i32(s, 0x00).unwrap_or(-1),
            colors,
            waypoint_min_scale: rd_f32(s, 0x488).unwrap_or(0.0),
            waypoint_max_scale: rd_f32(s, 0x48C).unwrap_or(0.0),
            navpoints_tag: tag_ref(s, 0x3C0),
            saved_film_tag: tag_ref(s, 0x400),
            curvatures,
        });
    }
    if skins.is_empty() {
        return None;
    }
    Some(ChudGlobals { path: native.tag_name(cache, tag).unwrap_or_default(), skins })
}

/// The nominal LINE HEIGHT of `full_screen_hud_message_font` (`tv_nord_hud-15`), in chud virtual
/// units — `ascender 17 + descender 4` out of the `.font` header, which is what the engine's
/// layout accumulates (`sub_180268C80`: `lineAdvance = glyphHeight * scale / font[+0x160]`, and
/// `font[+0x160]` is the package supersample factor, 1 in the shipped 1x package).
///
/// Only a fallback: where HMS has the real asset it uses that font's own header instead.
pub const HUD_FONT_LINE_HEIGHT_VU: f32 = 21.0;

// ---------------------------------------------------------------------------------------------
// the navpoint icon sprite
// ---------------------------------------------------------------------------------------------

/// `[RE] haloreach.dll+0x2E3BAC` draws the navpoint's main icon with
/// `sub_1802E3600(0, 1, R+0x04, 0.0, scale, flags, ...)`, which is
/// `sub_1802E7684(skin+0x3CC, 1, R+0x04, 0.5, 1.0, ...)` -> `sub_1802DE308(bitmap, 1, R+0x04, out)`.
/// So the SEQUENCE is the literal 1 and the SPRITE is the record's own `R+0x04`, which the
/// teammate-name producer (`+0x2CB76C`) writes as 0.
pub const NAVPOINT_SEQUENCE: usize = 1;
/// `R+0x04` for a plain player-name record.
pub const NAVPOINT_SPRITE: usize = 0;

/// One resolved sprite out of the navpoint sheet: where it is and how big the engine draws it.
#[derive(Clone, Copy, Debug)]
pub struct NavpointSprite {
    /// The `bitm` the skin's `navpoints reference` (+0x3C0) points at.
    pub bitmap_tag: u32,
    /// Which submap of that `bitm` the sprite lives in.
    pub bitmap_index: u16,
    /// `[left, right, top, bottom]` texture coordinates, after the spacing inset.
    pub uv: [f32; 4],
    /// `[width, height]` in CHUD VIRTUAL UNITS — which, as `+0x2DE308` computes it, is simply the
    /// sprite's size in sheet pixels.
    pub size: [f32; 2],
}

/// Resolve the navpoint icon sprite out of the skin's `navpoints reference` `bitm`.
///
/// `[RE] haloreach.dll+0x2DE308` reads the post-processed `bitm` at these byte offsets, which are
/// the tag's own (`reverseme tagdefs reach_tag_test.exe bitm`):
///
/// ```text
/// bitmap_group  +0x006 short  sprite spacing
///               +0x019 char   max mipmap level   (picks the default spacing: 1 when it is 1, else 4)
///               +0x070 block  sequences [max 256]           (64-byte elements)
///                        +0x020 short  first bitmap index
///                        +0x034 block  sprites [max 128]    (32-byte elements)
///                                 +0x000 short  bitmap index
///                                 +0x008 real   left    +0x00C right
///                                 +0x010 real   top     +0x014 bottom
///               +0x07C block  bitmaps [max 65536]           (56-byte elements)
///                        +0x000 short width   +0x002 short height   +0x005 byte flags
/// ```
///
/// and then: inset each edge by `spacing / width` (resp. height), and the drawn size is
/// `(right - left) * width` by `(bottom - top) * height`. Flags bit 0 halves the stored
/// dimensions and bit 4 divides them by three (the 2x / 3x supersampled sheets).
pub fn read_navpoint_sprite(native: &NativeDll, cache: CacheHandle, skin: &ChudSkin) -> Option<NavpointSprite> {
    read_sprite(native, cache, skin.navpoints_tag?, NAVPOINT_SEQUENCE, NAVPOINT_SPRITE)
}

/// One sprite of any sequence of a `bitm` sprite sheet, by the rule above.
pub fn read_sprite(native: &NativeDll, cache: CacheHandle, tag: u32, sequence: usize, sprite: usize) -> Option<NavpointSprite> {
    let meta = native.tag_meta(cache, tag, 0, 0xC0)?;
    let spacing = i16::from_le_bytes(meta.get(0x06..0x08)?.try_into().ok()?);
    let seqs = block(native, cache, &meta, 0x70, 64, 256);
    let seq = seqs.chunks_exact(64).nth(sequence)?;
    let sprites = block(native, cache, seq, 0x34, 32, 128);
    let sp = sprites.chunks_exact(32).nth(sprite)?;
    let bitmap_index = i16::from_le_bytes(sp.get(0x00..0x02)?.try_into().ok()?).max(0) as u16;
    let (mut l, mut r, mut t, mut b) = (rd_f32(sp, 0x08)?, rd_f32(sp, 0x0C)?, rd_f32(sp, 0x10)?, rd_f32(sp, 0x14)?);

    let bitmaps = block(native, cache, &meta, 0x7C, 56, 65536);
    let bm = bitmaps.chunks_exact(56).nth(bitmap_index as usize)?;
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
    // `+0x2DE308`: spacing > 0 is used as-is, 0 takes the usage default, and a NEGATIVE spacing
    // means none at all.
    let inset = if spacing > 0 {
        spacing as f32
    } else if spacing < 0 {
        0.0
    } else if *meta.get(0x19)? != 1 {
        4.0
    } else {
        1.0
    };
    if inset > 0.0 {
        let (du, dv) = (inset / w as f32, inset / h as f32);
        l += du;
        r -= du;
        t += dv;
        b -= dv;
    }
    Some(NavpointSprite {
        bitmap_tag: tag,
        bitmap_index,
        uv: [l, r, t, b],
        size: [(r - l) * w as f32, (b - t) * h as f32],
    })
}

thread_local! {
    /// One decoded `chgd` per open cache. The tag comes from the shared UI paging, so every
    /// Reach map carries the same one, but keying on the handle keeps a reopened map honest.
    static GLOBALS: std::cell::RefCell<std::collections::HashMap<u64, Option<std::rc::Rc<ChudGlobals>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The open cache's `chgd`, decoded once.
pub fn globals_for(native: &NativeDll, cache: CacheHandle) -> Option<std::rc::Rc<ChudGlobals>> {
    GLOBALS.with(|m| {
        m.borrow_mut()
            .entry(cache.0)
            .or_insert_with(|| read_globals(native, cache).map(std::rc::Rc::new))
            .clone()
    })
}

// ---------------------------------------------------------------------------------------------
// print-only diagnostic
// ---------------------------------------------------------------------------------------------

/// `--dump-chud <map>`: everything that authors the floating player name, from the open cache.
pub fn dump(native: &NativeDll, cache: CacheHandle) {
    println!("# the floating player name ('chud navpoint'), as Reach authors it\n");

    // 1. the anchor marker, resolved against the cache's own string table.
    let sid = native.string_id(cache, NAME_ANCHOR_MARKER_SID);
    println!("ANCHOR  [RE haloreach.dll+0x2CB76C -> +0x489FD4]");
    println!("  Object_BuildWorldTransform(unit, string_id 0x{NAME_ANCHOR_MARKER_SID:02X}, ...) then z += {NAME_ANCHOR_UP_WU}");
    println!("  string_id 0x{:02X} resolves to {:?}", NAME_ANCHOR_MARKER_SID, sid);
    let want = sid.clone().unwrap_or_default();
    let mut found = 0;
    for t in native.tags_by_group(cache, "bipd") {
        let mode = native.resolve_mode(cache, t);
        let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) else { continue };
        let bind = m.skel.bind_model();
        let path = native.tag_name(cache, t).unwrap_or_default();
        // The engine's marker and the one HMS used to use, side by side, in the model's own BIND
        // pose — which is what makes the height difference a number rather than an impression.
        for name in [want.as_str(), "chud_nav_point"] {
            if name.is_empty() {
                continue;
            }
            let Some(g) = m.markers.iter().find(|g| native.string_id(cache, g.name).as_deref() == Some(name)) else {
                continue;
            };
            if name == want {
                found += 1;
            }
            for mk in &g.markers {
                let node = if mk.node >= 0 { m.node_names.get(mk.node as usize).cloned().unwrap_or_default() } else { "<model>".into() };
                let world = mk.node.try_into().ok().and_then(|n: usize| bind.get(n).copied()).unwrap_or(glam::Mat4::IDENTITY) * mk.local();
                println!(
                    "    bipd '{path}' marker '{name}' node {} '{node}' local=({:.3},{:.3},{:.3}) bind-pose z={:.3} wu (+{NAME_ANCHOR_UP_WU} = {:.3})",
                    mk.node,
                    mk.translation.x,
                    mk.translation.y,
                    mk.translation.z,
                    world.w_axis.z,
                    world.w_axis.z + NAME_ANCHOR_UP_WU
                );
            }
        }
    }
    println!("  '{want}' is carried by {found} bipd(s)");

    // 2. the chgd skins.
    let Some(g) = read_globals(native, cache) else {
        println!("\n(no chgd in this cache)");
        return;
    };
    println!("\nchgd '{}' — {} skin(s)", g.path, g.skins.len());
    for (i, s) in g.skins.iter().enumerate() {
        println!(
            "\n== skin[{i}] type {} ({})   waypoint scale min {} max {}",
            s.skin_type,
            s.type_name(),
            s.waypoint_min_scale,
            s.waypoint_max_scale
        );
        println!(
            "   navpoints reference -> {:?}   saved-film reference -> {:?}",
            s.navpoints_tag.and_then(|t| native.tag_name(cache, t)),
            s.saved_film_tag.and_then(|t| native.tag_name(cache, t))
        );
        println!("   virtual screen (curvature infos, {} entry/entries):", s.curvatures.len());
        for c in &s.curvatures {
            println!(
                "     res flags {:#07b}  virtual {} x {}  safe frame {:.3} x {:.3}",
                c.res_flags, c.virtual_width, c.virtual_height, c.safe_frame[0], c.safe_frame[1]
            );
        }
        println!(
            "   -> full-screen widescreen virtual height = {}, safe frame v = {}, ding v = {}",
            s.virtual_height(),
            s.safe_frame_v(),
            s.safe_frame_ding_v()
        );
        for wh in [720.0f32, 1080.0, 1440.0] {
            let k = px_per_virtual(wh, s.virtual_height(), s.safe_frame_v(), s.safe_frame_ding_v());
            println!(
                "      a {wh:.0}-px window: 1 virtual unit = {k:.4} px, so the {} font's {} units draw {:.1}..{:.1} px \
                 over the {}..{} distance scale",
                crate::reach_font::FontId::FullScreenHudMessage.file_stem(),
                HUD_FONT_LINE_HEIGHT_VU,
                HUD_FONT_LINE_HEIGHT_VU * k * s.waypoint_min_scale,
                HUD_FONT_LINE_HEIGHT_VU * k * s.waypoint_max_scale,
                s.waypoint_min_scale,
                s.waypoint_max_scale
            );
        }
        match read_navpoint_sprite(native, cache, s) {
            Some(sp) => println!(
                "   navpoint icon: {} sequence {NAVPOINT_SEQUENCE} sprite {NAVPOINT_SPRITE} -> submap {} \
                 uv ({:.4},{:.4})..({:.4},{:.4}) size {:.1} x {:.1} virtual units",
                native.tag_name(cache, sp.bitmap_tag).unwrap_or_default(),
                sp.bitmap_index,
                sp.uv[0],
                sp.uv[2],
                sp.uv[1],
                sp.uv[3],
                sp.size[0],
                sp.size[1]
            ),
            None => println!("   navpoint icon: could not be read"),
        }
        for (j, c) in s.colors.iter().enumerate() {
            let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            println!(
                "   [{j:2}] +0x{:03X} {:34} argb=({:.3},{:.3},{:.3},{:.3})  #{:02X}{:02X}{:02X} a={:02X}",
                skin_color_offset(j),
                SKIN_COLORS[j],
                c[0],
                c[1],
                c[2],
                c[3],
                u8c(c[1]),
                u8c(c[2]),
                u8c(c[3]),
                u8c(c[0])
            );
        }
    }

    // 3. the engine's index map, resolved on the default skin.
    if let Some(s) = g.default_skin() {
        println!("\nCOLOUR INDEX -> colour  [RE haloreach.dll+0x2E36B8], on the 'default' skin");
        for i in 0..13 {
            match color_from_index(i) {
                Some(k) => {
                    let c = s.colors[k];
                    let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    println!(
                        "   index {i:2} -> skin+0x{:03X} {:34} #{:02X}{:02X}{:02X} a={:02X}",
                        skin_color_offset(k),
                        SKIN_COLORS[k],
                        u8c(c[1]),
                        u8c(c[2]),
                        u8c(c[3]),
                        u8c(c[0])
                    );
                }
                None => println!("   index {i:2} -> (none: the engine returns a zeroed colour)"),
            }
        }
        println!("\n   a teammate's NAME is built with colour index {PLAYER_NAME_COLOR_INDEX} and colour override -1");
        println!("   [RE haloreach.dll+0x2CB76C], so it draws in '{}'", SKIN_COLORS[PLAYER_NAME_COLOR]);
        for st in 1..4 {
            if let Some(k) = color_from_state(st) {
                let c = s.colors[k];
                let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                println!(
                    "   state {st} -> skin+0x{:03X} {:34} #{:02X}{:02X}{:02X} a={:02X}",
                    skin_color_offset(k),
                    SKIN_COLORS[k],
                    u8c(c[1]),
                    u8c(c[2]),
                    u8c(c[3]),
                    u8c(c[0])
                );
            }
        }
    }

    // 4. mulg team colours, for comparison.
    println!("\nmulg team colours (for comparison — NOT what the engine uses here)");
    for (i, c) in crate::reach_widget::team_colors(native, cache).iter().enumerate() {
        println!("   team {i:2} #{:02X}{:02X}{:02X}", c.r(), c.g(), c.b());
    }
}

/// `--dump-chud <map> --icon-png <dir>`: write the whole navpoint sprite sheet and every sprite of
/// every sequence out as PNGs, so the sheet's layout can be LOOKED AT rather than inferred from
/// its numbers. Print-only in the sense that matters — it touches nothing the app reads.
pub fn dump_navpoint_sheet(native: &NativeDll, cache: CacheHandle, skin: &ChudSkin, dir: &str) {
    let Some(tag) = skin.navpoints_tag else {
        println!("no navpoints reference");
        return;
    };
    let name = native.tag_name(cache, tag).unwrap_or_default();
    let Some(meta) = native.tag_meta(cache, tag, 0, 0xC0) else { return };
    let seqs = block(native, cache, &meta, 0x70, 64, 256);
    println!("\n# {name}: {} sequence(s)", seqs.len() / 64);
    let _ = std::fs::create_dir_all(dir);
    // the whole sheet first
    for submap in 0..4u32 {
        let Some((bgra, w, h)) = native.decode_bitmap_keep_alpha(cache, tag, submap, 0) else { continue };
        let rgba: Vec<u8> = bgra.chunks_exact(4).flat_map(|c| [c[2], c[1], c[0], c[3]]).collect();
        let path = format!("{dir}/sheet{submap}.png");
        match crate::headless::write_png(&path, &rgba, w, h) {
            Ok(()) => println!("  submap {submap}: {w} x {h} -> {path}"),
            Err(e) => println!("  submap {submap}: {e:#}"),
        }
    }
    for (si, seq) in seqs.chunks_exact(64).enumerate() {
        let sprites = block(native, cache, seq, 0x34, 32, 128);
        let first = i16::from_le_bytes(seq[0x20..0x22].try_into().unwrap_or_default());
        let count = i16::from_le_bytes(seq[0x22..0x24].try_into().unwrap_or_default());
        println!("  sequence {si}: first bitmap {first}, bitmap count {count}, {} sprite(s)", sprites.len() / 32);
        for (pi, _) in sprites.chunks_exact(32).enumerate() {
            let Some(sp) = read_sprite(native, cache, tag, si, pi) else { continue };
            let Some((bgra, w, h)) = native.decode_bitmap_keep_alpha(cache, tag, sp.bitmap_index as u32, 0) else { continue };
            let (x0, x1) = ((sp.uv[0] * w as f32) as i32, (sp.uv[1] * w as f32) as i32);
            let (y0, y1) = ((sp.uv[2] * h as f32) as i32, (sp.uv[3] * h as f32) as i32);
            let (cw, ch) = ((x1 - x0).max(1) as u32, (y1 - y0).max(1) as u32);
            let mut crop = vec![0u8; (cw * ch * 4) as usize];
            for y in 0..ch {
                for x in 0..cw {
                    let (sx, sy) = (x0 + x as i32, y0 + y as i32);
                    if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                        continue;
                    }
                    let si2 = ((sy as u32 * w + sx as u32) * 4) as usize;
                    let di = ((y * cw + x) * 4) as usize;
                    if si2 + 3 < bgra.len() {
                        crop[di] = bgra[si2 + 2];
                        crop[di + 1] = bgra[si2 + 1];
                        crop[di + 2] = bgra[si2];
                        crop[di + 3] = bgra[si2 + 3];
                    }
                }
            }
            let path = format!("{dir}/seq{si}_sprite{pi}.png");
            let _ = crate::headless::write_png(&path, &crop, cw, ch);
            println!(
                "    sprite {pi}: submap {} uv ({:.4},{:.4})..({:.4},{:.4}) size {:.1} x {:.1} vu -> {path}",
                sp.bitmap_index, sp.uv[0], sp.uv[2], sp.uv[1], sp.uv[3], sp.size[0], sp.size[1]
            );
        }
    }
}

/// The second half of `--dump-chud`: one film's own players, their teams, and the colour each
/// name resolves to under [`navpoint_color_index`] with the viewer the film was recorded by.
pub fn dump_film(path: &std::path::Path, skin: Option<&ChudSkin>) {
    let film = match crate::film::open_header(path) {
        Ok(f) => f,
        Err(e) => {
            println!("\n{}: {e:#}", path.display());
            return;
        }
    };
    let viewer = film.recorder_slot();
    let h = &film.header;
    println!("\n# {} — the viewer and every name's resolved colour", path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default());
    println!(
        "  teams: {}   recorder: {:?} (slot {:?})",
        h.gametype
            .as_ref()
            .map_or("<no game variant>".to_string(), |g| if g.misc.teams_enabled() { "on".into() } else { "off".into() }),
        film.content.created_by,
        viewer
    );
    let teams = h.gametype.as_ref().is_some_and(|g| g.misc.teams_enabled());
    let vt = viewer.and_then(|i| h.players.iter().find(|p| p.index == i as usize)).map(|p| p.team).unwrap_or(TEAM_NONE);
    println!("  viewer team = {vt}");
    // Through the SHIPPED function, so this prints what the app draws rather than a second
    // implementation of the same rule.
    let mut tags: Vec<crate::film_ui::NameTag> = h
        .players
        .iter()
        .filter(|p| p.present())
        .map(|p| crate::film_ui::NameTag {
            text: p.gamertag.clone(),
            pos: [0.0; 3],
            named: !p.gamertag.is_empty(),
            inferred_from: None,
            team: p.team,
            net_id: 0,
            feet: [0.0; 3],
            on_marker: false,
            slot: Some(p.index as u8),
            color_index: 2,
            show_icon: false,
        })
        .collect();
    crate::film_ui::resolve_name_colors(&mut tags, viewer, vt, teams);
    for (p, t) in h.players.iter().filter(|p| p.present()).zip(tags.iter()) {
        let idx = t.color_index;
        let k = color_from_index(idx);
        let hex = match (skin, k) {
            (Some(s), Some(k)) => {
                let c = s.colors[k];
                let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                format!("#{:02X}{:02X}{:02X}", u8c(c[1]), u8c(c[2]), u8c(c[3]))
            }
            _ => "<no chgd>".into(),
        };
        println!(
            "   slot {:2} team {:3}  {:20} -> colour index {idx} {:34} {hex}  {}",
            p.index,
            p.team,
            p.gamertag,
            k.map(|k| SKIN_COLORS[k]).unwrap_or("<none>"),
            if t.show_icon { "+ chevron (ally waypoint)" } else { "name only (look-at display)" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #spectate The card's subtree walk: everything under the item, minus the four components
    /// HMS cannot or should not draw, with origins made relative to the item.
    ///
    /// Built on a synthetic screen so it gates the WALK rather than the tag: a regression that
    /// drew the whole screen (the spawn strings, the flying-camera label) or that forgot to
    /// rebase the origins fails here.
    #[test]
    fn the_spectate_walk_takes_the_item_subtree_only() {
        use crate::reach_widget::{Component, Props, Screen};
        let comp = |name: &str, parent: &str, kind: &str, left: f32, top: f32| Component {
            name: name.into(),
            kind: kind.into(),
            parent: parent.into(),
            props: Props {
                reals: vec![("prop_left".into(), left), ("prop_top".into(), top)],
                ..Default::default()
            },
        };
        let screen = Screen {
            path: String::new(),
            bindings: Vec::new(),
            virtual_size: [0.0, 0.0],
            components: vec![
                comp("control_pad_container", "", "container_widget", 437.0, 576.0),
                comp("film_control_pad_list", "control_pad_container", "list_widget", 0.0, 23.0),
                comp(SPECTATE_ITEM, "film_control_pad_list", "list_item_widget", 0.0, 4.0),
                comp("gray_field", SPECTATE_ITEM, "bitmap_widget", 3.0, 0.0),
                comp("text_player_name", SPECTATE_ITEM, "text_widget", 36.0, 4.0),
                comp("player_bar_mini_emblem", SPECTATE_ITEM, "emblem_widget", 6.0, 3.0),
                comp("animated_dpad_up_text", SPECTATE_ITEM, "text_widget", 252.0, 4.0),
                // NOT under the item: the rest of the Theater screen.
                comp("flying_cam_text_container", "control_pad_container", "container_widget", 0.0, 0.0),
                comp("flying_cam_text", "flying_cam_text_container", "text_widget", 0.0, 0.0),
            ],
        };
        let got: Vec<(String, [f32; 2])> =
            spectate_components(&screen).into_iter().map(|(c, o)| (c.name.clone(), o)).collect();
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec![SPECTATE_ITEM, "gray_field", "text_player_name"], "{names:?}");
        // the item is its own origin, children are relative to it
        assert_eq!(got[0].1, [0.0, 0.0]);
        assert_eq!(got[1].1, [3.0, 0.0]);
        assert_eq!(got[2].1, [36.0, 4.0]);
        // and the absolute place is the authored one: 437 + 0 + 0, 576 + 23 + 4
        assert_eq!(screen.origin_of(screen.get(SPECTATE_ITEM).unwrap()), [437.0, 603.0]);
    }

    /// The engine's own switch, byte for byte. `haloreach.dll+0x2E36B8` loads a 16-byte colour
    /// from these exact skin offsets for these exact indices — this is that table, so a
    /// renumbering of [`SKIN_COLORS`] or of [`color_from_index`] cannot pass silently.
    #[test]
    fn color_index_offsets_match_the_engine_switch() {
        // (colour index, the skin byte offset the disassembly loads from)
        const ENGINE: [(i32, usize); 12] = [
            (0, 0x074),
            (1, 0x084),
            (2, 0x094),
            (3, 0x0A4),
            (4, 0x0B4),
            (5, 0x184),
            (6, 0x194),
            (7, 0x1A4),
            (8, 0x1B4),
            (9, 0x0C4),
            (10, 0x0D4),
            (11, 0x0E4),
        ];
        for (index, off) in ENGINE {
            let k = color_from_index(index).unwrap_or_else(|| panic!("colour index {index} resolves to nothing"));
            assert_eq!(skin_color_offset(k), off, "colour index {index} -> {} at the wrong offset", SKIN_COLORS[k]);
        }
        // Out of range is a zeroed colour in the engine, i.e. no entry here.
        assert_eq!(color_from_index(12), None);
        assert_eq!(color_from_index(-1), None);
    }

    /// The three STATE colours, likewise: `+0x2E3AD8` loads +0x144 / +0x154 / +0x164 for states
    /// 1 / 2 / 3 and nothing else.
    #[test]
    fn state_colors_match_the_engine_switch() {
        for (state, off) in [(1, 0x144usize), (2, 0x154), (3, 0x164)] {
            let k = color_from_state(state).unwrap();
            assert_eq!(skin_color_offset(k), off);
        }
        assert_eq!(color_from_state(0), None);
        assert_eq!(color_from_state(8), None); // the ICON path has one; the TEXT path does not
    }

    /// The names come from the tool build's own reflection data
    /// (`reverseme tagdefs reach_tag_test.exe chgd`), and the ones this module keys off are
    /// pinned to their authored offsets.
    #[test]
    fn skin_color_names_sit_at_their_authored_offsets() {
        assert_eq!(SKIN_COLORS.len(), 32);
        for (name, off) in [
            ("navpt normal", 0x074usize),
            ("navpt ally", 0x084),
            ("navpt neutral", 0x094),
            ("navpt enemy", 0x0A4),
            ("navpt text", 0x0E4),
            ("player navpoint (firing)", 0x144),
            ("player navpoint (taking damage)", 0x154),
            ("player navpoint (speaking)", 0x164),
        ] {
            let i = SKIN_COLORS.iter().position(|n| *n == name).unwrap_or_else(|| panic!("no '{name}'"));
            assert_eq!(skin_color_offset(i), off, "'{name}'");
        }
        assert_eq!(PLAYER_NAME_COLOR, SKIN_COLORS.iter().position(|n| *n == "navpt normal").unwrap());
        assert_eq!(skin_color_offset(PLAYER_NAME_COLOR), 0x74);
        assert_eq!(SKIN_ELEMENT, 1332); // == 0x14D * 4, the engine's per-skin stride
    }

    /// `+0x14844C`'s team comparison, as a truth table. Same team is `navpt normal`, a different
    /// team is `navpt enemy`, and an unknown team on either side is `navpt neutral`.
    #[test]
    fn navpoint_color_index_is_the_engine_team_rule() {
        assert_eq!(navpoint_color_index(0, 0), 0);
        assert_eq!(navpoint_color_index(3, 3), 0);
        assert_eq!(navpoint_color_index(0, 1), 3);
        assert_eq!(navpoint_color_index(1, 0), 3);
        assert_eq!(navpoint_color_index(TEAM_NONE, 0), 2);
        assert_eq!(navpoint_color_index(0, TEAM_NONE), 2);
        assert_eq!(navpoint_color_index(TEAM_NONE, TEAM_NONE), 2);
        // and each of those three resolves to the colour it is named for
        assert_eq!(color_from_index(0).map(|k| SKIN_COLORS[k]), Some("navpt normal"));
        assert_eq!(color_from_index(2).map(|k| SKIN_COLORS[k]), Some("navpt neutral"));
        assert_eq!(color_from_index(3).map(|k| SKIN_COLORS[k]), Some("navpt enemy"));
    }

    /// The skin accessor reads the field it is named for, out of a synthetic skin — so the
    /// decoder cannot quietly return a constant.
    #[test]
    fn skin_lookup_reads_the_named_field() {
        let mut colors = [[0.0f32; 4]; 32];
        for (i, c) in colors.iter_mut().enumerate() {
            *c = [1.0, i as f32 / 100.0, 0.5, 0.25];
        }
        let s = ChudSkin {
            skin_type: 0,
            colors,
            waypoint_min_scale: 0.8,
            waypoint_max_scale: 1.3,
            navpoints_tag: None,
            saved_film_tag: None,
            curvatures: Vec::new(),
        };
        assert_eq!(s.color("navpt normal").unwrap()[1], 0.07);
        assert_eq!(s.color("navpt enemy").unwrap()[1], 0.10);
        assert_eq!(s.player_name_color()[1], 0.07);
        assert_eq!(s.type_name(), "default");
        assert!(s.color("no such colour").is_none());
    }
}

// ---------------------------------------------------------------------------------------------
// chud_definition (`chdt`) — the widget trees the HUD is actually built from
// ---------------------------------------------------------------------------------------------

/// `chud_datasource_type_enum` (22 values), in index order
/// (`reverseme tagdefs reach_tag_test.exe chdt`). A collection with a datasource is STAMPED once
/// per item of it, which is how one authored widget becomes one per player.
pub const DATASOURCE_TYPES: [&str; 22] = [
    "grenades",
    "players",
    "metagame players",
    "fireteam members",
    "campaign fireteam members",
    "medals",
    "progression toasts",
    "scripted objects",
    "skulls",
    "tracking objects",
    "omni top left",
    "omni top center",
    "omni top right",
    "omni high left",
    "omni high center",
    "omni high right",
    "omni low left",
    "omni low center",
    "omni low right",
    "omni bottom left",
    "omni bottom center",
    "omni bottom right",
];

/// `chud_anchor_type_enum` (51 values), in index order. 25 `<player>` is the one that puts a
/// widget over a PLAYER's head.
pub const ANCHOR_TYPES: [&str; 51] = [
    "parent",
    "top left",
    "top center",
    "top right",
    "center",
    "bottom left",
    "bottom center",
    "bottom right",
    "motion sensor",
    "d-damge",
    "messaging",
    "state msg left",
    "state msg right",
    "msg bottom state",
    "msg bottom prim",
    "tracked target",
    "<tracking object>",
    "crosshair",
    "backpack weapon",
    "grenade",
    "equipment",
    "weapon target",
    "ghost reticule",
    "hologram target",
    "airstrike target",
    "<player>",
    "<scripted object>",
    "metagame bar",
    "metagame p1",
    "metagame p2",
    "metagame p3",
    "metagame p4",
    "s.b. friendly",
    "s.b. enemy",
    "territory 1",
    "territory 2",
    "territory 3",
    "territory 4",
    "territory 5",
    "state msg fireteam",
    "fireteam possible action object",
    "fireteam pending target object",
    "fireteam current target object",
    "fireteam pending directive unit",
    "fireteam current directive unit",
    "enemy objective object",
    "friendly objective object",
    "neutral objective object",
    "lasing target object",
    "<campaign fireteam member>",
    "top center savedfilm",
];

/// `chud_render_color_output_enum` value `v` for `v >= 18` is skin colour `v - 18`; below that it
/// is a per-widget or per-datum colour that no tag field holds.
pub const COLOR_OUTPUT_BASE: i32 = 18;
pub const COLOR_OUTPUT_NAMES: [&str; 18] = [
    "custom color A",
    "custom color B",
    "custom color C",
    "custom color D",
    "color animation A",
    "color animation B",
    "scoreboard friendly",
    "scoreboard enemy",
    "arming meter",
    "metagame p1 color",
    "metagame p2 color",
    "metagame p3 color",
    "metagame p4 color",
    "<metagame player> color",
    "commendation game type color",
    "commendation callout level color",
    "<scripted object> primary color",
    "<scripted object> secondary color",
];

/// Name a `color output` enum value: the skin colour it selects, or the per-widget source.
pub fn color_output_name(v: i32) -> String {
    if v >= COLOR_OUTPUT_BASE {
        let k = (v - COLOR_OUTPUT_BASE) as usize;
        match SKIN_COLORS.get(k) {
            Some(n) => format!("skin '{n}' (+0x{:03X})", skin_color_offset(k)),
            None => format!("<out of range {v}>"),
        }
    } else {
        COLOR_OUTPUT_NAMES.get(v as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("<{v}>"))
    }
}

/// `--dump-chdt <map> [substring]`: every `chud_definition`'s widget tree — the datasource that
/// stamps it, each widget's anchor, scale, font, colour outputs and bitmap. This is how the
/// over-the-head name display is told apart from the navpoint waypoint.
///
/// LAYOUT (`reverseme tagdefs reach_tag_test.exe chdt`):
///
/// ```text
/// chud_definition                      40 bytes
///   +0x000 block  widget collections [max 32]       (220 B)
///            +0x000 struct base -> chud_widget_base_struct  (124 B)
///            +0x07C struct datasource                        (28 B: tag ref, then block @+0x10)
///            +0x098 block  bitmap widgets [max 32]           (156 B)
///            +0x0A4 block  text widgets   [max 60]           (132 B)
///            +0x0B0 block  object highlight widgets [max 4]  (128 B)
/// chud_widget_base_struct              124 bytes
///   +0x000 string id artist name
///   +0x007 char      sort layer
///   +0x00C struct    state data      (28 B: tag ref, block @+0x10, 56-B elements)
///   +0x028 struct    placement data  (28 B: tag ref, block @+0x10, 28-B elements)
///   +0x060 struct    render data     (28 B: tag ref, block @+0x10, 120-B elements)
/// placement element  28 bytes: +0x00 res flags, +0x01 anchor type, +0x02 anchor flags,
///                              +0x04 origin (2 reals), +0x0C origin offset, +0x14 scale
/// render element    120 bytes: +0x00 shader type, +0x04/14/24/34 custom colours A..D,
///                              +0x54..+0x5E color output A..F, +0x74 alpha blend mode
/// datasource element 32 bytes: +0x04 type, +0x06 render maximum
/// text widget       132 bytes: base, +0x07C flags, +0x07E font id, +0x080 input string id
/// bitmap widget     156 bytes: base, +0x07C flags, +0x080 bitmap tag ref, +0x090 sequence index
/// ```
/// `--list-ui <map> [substring]`: every UI tag's path, by group. Finding a widget starts with
/// knowing what the cache actually holds.
pub fn list_ui(native: &NativeDll, cache: CacheHandle, filter: &str) {
    for group in ["cusc", "chdt", "chgd", "wgtz", "cust", "wigl"] {
        let tags = native.tags_by_group(cache, group);
        let mut names: Vec<String> = tags.iter().filter_map(|&t| native.tag_name(cache, t)).collect();
        names.sort();
        let shown: Vec<&String> =
            names.iter().filter(|n| filter.is_empty() || n.to_ascii_lowercase().contains(&filter.to_ascii_lowercase())).collect();
        println!("\n== {group}: {} tag(s), {} shown", names.len(), shown.len());
        for n in shown {
            println!("   {n}");
        }
    }
}

pub fn dump_chdt(native: &NativeDll, cache: CacheHandle, filter: &str) {
    let sid = |v: u32| -> String {
        if v == 0 || v == u32::MAX { String::new() } else { native.string_id(cache, v).unwrap_or_default() }
    };
    let anchor_name = |i: u8| ANCHOR_TYPES.get(i as usize).copied().unwrap_or("<?>");
    for tag in native.tags_by_group(cache, "chdt") {
        let name = native.tag_name(cache, tag).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        let Some(meta) = native.tag_meta(cache, tag, 0, 0x28) else { continue };
        let colls = block(native, cache, &meta, 0x00, 220, 32);
        println!("\n================ chdt '{name}'  ({} widget collection(s))", colls.len() / 220);
        for (ci, coll) in colls.chunks_exact(220).enumerate() {
            let ds = block(native, cache, coll, 0x7C + 0x10, 32, 1);
            // `type` is a SHORT enum at +0x04 and `render maximum` the short right after it,
            // so a 32-bit read here silently packs the two together.
            let ds_type = if ds.len() >= 8 { i16::from_le_bytes(ds[4..6].try_into().unwrap_or_default()) as i32 } else { -1 };
            let ds_max = if ds.len() >= 8 { i16::from_le_bytes(ds[6..8].try_into().unwrap_or_default()) } else { 0 };
            let ds_name = DATASOURCE_TYPES.get(ds_type as usize).copied().unwrap_or("<none>");
            let place = block(native, cache, coll, 0x28 + 0x10, 28, 5);
            let anchors: Vec<String> = place
                .chunks_exact(28)
                .map(|p| {
                    format!(
                        "res {:#07b} anchor {} '{}' origin ({:.1},{:.1}) off ({:.1},{:.1}) scale ({:.3},{:.3})",
                        p[0],
                        p[1],
                        anchor_name(p[1]),
                        rd_f32(p, 0x04).unwrap_or(0.0),
                        rd_f32(p, 0x08).unwrap_or(0.0),
                        rd_f32(p, 0x0C).unwrap_or(0.0),
                        rd_f32(p, 0x10).unwrap_or(0.0),
                        rd_f32(p, 0x14).unwrap_or(0.0),
                        rd_f32(p, 0x18).unwrap_or(0.0)
                    )
                })
                .collect();
            println!(
                "  [{ci}] '{}' sort {}  datasource {ds_type} '{ds_name}'{}",
                sid(rd_u32(coll, 0).unwrap_or(0)),
                coll[0x07] as i8,
                if ds.is_empty() { String::new() } else { format!(" max {ds_max}") }
            );
            for a in &anchors {
                println!("       placement: {a}");
            }
            let dump_base = |what: &str, wi: usize, b: &[u8], extra: String| {
                let place = block(native, cache, b, 0x28 + 0x10, 28, 5);
                let render = block(native, cache, b, 0x60 + 0x10, 120, 1);
                println!("     {what}[{wi}] '{}' sort {}{extra}", sid(rd_u32(b, 0).unwrap_or(0)), b[0x07] as i8);
                for p in place.chunks_exact(28) {
                    println!(
                        "        placement: res {:#07b} anchor {} '{}' origin ({:.1},{:.1}) off ({:.1},{:.1}) scale ({:.3},{:.3})",
                        p[0],
                        p[1],
                        anchor_name(p[1]),
                        rd_f32(p, 0x04).unwrap_or(0.0),
                        rd_f32(p, 0x08).unwrap_or(0.0),
                        rd_f32(p, 0x0C).unwrap_or(0.0),
                        rd_f32(p, 0x10).unwrap_or(0.0),
                        rd_f32(p, 0x14).unwrap_or(0.0),
                        rd_f32(p, 0x18).unwrap_or(0.0)
                    );
                }
                for r in render.chunks_exact(120) {
                    let out: Vec<String> = (0..6)
                        .map(|k| {
                            let v = i16::from_le_bytes(r[0x54 + k * 2..0x56 + k * 2].try_into().unwrap_or_default()) as i32;
                            color_output_name(v)
                        })
                        .collect();
                    println!(
                        "        render: shader {} blend {}  colour outputs A..F: {}",
                        r[0x00],
                        i16::from_le_bytes(r[0x74..0x76].try_into().unwrap_or_default()),
                        out.join(" | ")
                    );
                    for (k, off) in [("A", 0x04usize), ("B", 0x14), ("C", 0x24), ("D", 0x34)] {
                        let c = [rd_f32(r, off).unwrap_or(0.0), rd_f32(r, off + 4).unwrap_or(0.0), rd_f32(r, off + 8).unwrap_or(0.0), rd_f32(r, off + 12).unwrap_or(0.0)];
                        if c.iter().any(|v| *v != 0.0) {
                            println!("          custom colour {k} argb ({:.3},{:.3},{:.3},{:.3})", c[0], c[1], c[2], c[3]);
                        }
                    }
                }
            };
            for (wi, b) in block(native, cache, coll, 0xA4, 132, 60).chunks_exact(132).enumerate() {
                let font = i16::from_le_bytes(b[0x7E..0x80].try_into().unwrap_or_default());
                let extra = format!(
                    "  TEXT font {font} '{}' string '{}' flags {:#06x}",
                    crate::reach_font::FontId::from_index(font as i32).map(|f| f.file_stem()).unwrap_or("<?>"),
                    sid(rd_u32(b, 0x80).unwrap_or(0)),
                    u16::from_le_bytes(b[0x7C..0x7E].try_into().unwrap_or_default())
                );
                dump_base("text", wi, b, extra);
            }
            for (wi, b) in block(native, cache, coll, 0x98, 156, 32).chunks_exact(156).enumerate() {
                let bt = tag_ref(b, 0x80);
                let extra = format!(
                    "  BITMAP {:?} sequence {} flags {:#010x}",
                    bt.and_then(|t| native.tag_name(cache, t)),
                    b[0x90] as i8,
                    rd_u32(b, 0x7C).unwrap_or(0)
                );
                dump_base("bitmap", wi, b, extra);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// #spectate  Theater's spectated-player card, from `ui\cui\in_game\film_playback\film_cameras`
// ---------------------------------------------------------------------------------------------

/// The `cui_screen` Theater draws its camera bar from. One list item, stamped once
/// (`prop_item_count` = 1), showing WHOSE camera you are watching.
pub const FILM_CAMERAS_SCREEN: &str = "ui\\cui\\in_game\\film_playback\\film_cameras";

/// The component whose subtree is the card itself. Everything else in the screen is the
/// surrounding Theater UI (the spawn/out-of-lives strings, the flying-camera label).
pub const SPECTATE_ITEM: &str = "film_control_pad_list_item";

/// Components of that subtree HMS deliberately does NOT draw, and why.
///
/// * `emblem_field` / `player_bar_mini_emblem` — the player's emblem. Reach composes an emblem
///   from `mlib` layers that MCC does not even run (see the note on `emblem_widget` in
///   `film_ui`), so drawing one would be inventing art. The 28 x 28 slot is simply left empty,
///   exactly as the scoreboard leaves it.
/// * `animated_dpad_down_text` / `animated_dpad_up_text` — the controller hints that tell you
///   which way to cycle cameras. They name a physical d-pad HMS does not have.
pub const SPECTATE_SKIP: [&str; 4] =
    ["emblem_field", "player_bar_mini_emblem", "animated_dpad_down_text", "animated_dpad_up_text"];

/// The virtual screen a `cui_screen` is authored in for `resolution_widescreen`.
///
/// The screen's own `screen` component carries no size here (only `prop_category` /
/// `prop_enabled`), so this is the documented 16:9 authoring space. The card's own numbers
/// corroborate it: `control_pad_container` is `left 437, width 277`, whose centre is 575.5 —
/// dead centre of 1152 — and `top 576, height 75` sits on the bottom edge of 720.
pub const CUI_VIRTUAL: [f32; 2] = [1152.0, 720.0];

/// What HMS puts in the card: the spectated player's name and service tag.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectateText<'a> {
    /// `text_player_name` — `prop_font_id` 7, box 159.66 x 22 at (36, 4) inside the item.
    pub name: &'a str,
    /// `text_payer_tag` (the engine's own spelling) — font 7, right-justified, 54 x 22 at (196, 4).
    pub service_tag: &'a str,
}

thread_local! {
    /// One decoded camera screen per open cache. `None` = tried and could not be read, in which
    /// case the caller draws nothing rather than approximating the card.
    static SPECTATE_SCREEN: std::cell::RefCell<std::collections::HashMap<u64, Option<std::rc::Rc<crate::reach_widget::Screen>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The open cache's Theater camera screen, decoded once.
pub fn spectate_screen(native: &NativeDll, cache: CacheHandle) -> Option<std::rc::Rc<crate::reach_widget::Screen>> {
    SPECTATE_SCREEN.with(|m| {
        if let Some(v) = m.borrow().get(&cache.0) {
            return v.clone();
        }
        let made = native
            .tags_by_group(cache, "cusc")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.eq_ignore_ascii_case(FILM_CAMERAS_SCREEN)))
            .and_then(|t| {
                crate::reach_widget::read_screen(
                    native,
                    cache,
                    t,
                    crate::reach_widget::OVERLAY_RESOLUTION,
                    crate::reach_widget::OVERLAY_THEME,
                )
            })
            .filter(|s| s.get(SPECTATE_ITEM).is_some())
            .map(std::rc::Rc::new);
        m.borrow_mut().insert(cache.0, made.clone());
        made
    })
}

/// Every component of the card's subtree, in tag order, with its origin RELATIVE to the item.
///
/// `prop_visible` on the item itself is 0 in the tag — Theater turns it on through a property
/// binding when a camera is selected — so this walk deliberately ignores the item's own hidden
/// flag. HMS only calls it when it IS spectating someone, which is the same condition.
pub fn spectate_components(screen: &crate::reach_widget::Screen) -> Vec<(&crate::reach_widget::Component, [f32; 2])> {
    let Some(item) = screen.get(SPECTATE_ITEM) else { return Vec::new() };
    let base = screen.origin_of(item);
    screen
        .components
        .iter()
        .filter(|c| is_under(screen, c, SPECTATE_ITEM) && !SPECTATE_SKIP.contains(&c.name.as_str()))
        .map(|c| {
            let o = screen.origin_of(c);
            (c, [o[0] - base[0], o[1] - base[1]])
        })
        .collect()
}

/// Is `c` the named component or a descendant of it?
fn is_under(screen: &crate::reach_widget::Screen, c: &crate::reach_widget::Component, ancestor: &str) -> bool {
    if c.name == ancestor {
        return true;
    }
    let mut p = c.parent.clone();
    for _ in 0..16 {
        if p.is_empty() {
            return false;
        }
        if p == ancestor {
            return true;
        }
        match screen.get(&p) {
            Some(q) => p = q.parent.clone(),
            None => return false,
        }
    }
    false
}

/// The card's DRAWN extent in virtual units — the far corner of its children, not the item's own
/// box.
///
/// The list item is authored 350 x 42 because that is the template's nominal cell, but the art
/// stops at `right_edge` (275 + 2 = 277) and is 28 tall. 277 is the width that matters: the
/// item sits at x 437, so the drawn bar's centre is 575.5 of [`CUI_VIRTUAL`]'s 1152 — dead
/// centre, which is where Theater puts it. Using 350 would place the centre at 612 and make the
/// bar look 36 units right of where the game has it.
pub fn spectate_size(screen: &crate::reach_widget::Screen) -> [f32; 2] {
    let drawn = spectate_components(screen);
    let far = drawn
        .iter()
        .filter(|(c, _)| c.name != SPECTATE_ITEM)
        .fold([0.0f32, 0.0f32], |a, (c, o)| [a[0].max(o[0] + c.width()), a[1].max(o[1] + c.height())]);
    if far[0] > 1.0 && far[1] > 1.0 {
        far
    } else {
        screen.get(SPECTATE_ITEM).map(|c| [c.width(), c.height()]).unwrap_or([277.0, 28.0])
    }
}

/// Paint the spectated-player card where Theater puts it, from the tag's own geometry.
///
/// The screen is authored in [`CUI_VIRTUAL`] and HMS maps it the way a 16:9-authored screen has
/// to be mapped onto an arbitrary viewport: fit the HEIGHT, centre the width. That keeps the
/// card's authored centring (its container's centre is 575.5 of 1152, dead centre) true at any
/// window shape, and makes one virtual unit `viewport_height / 720` points — the same kind of
/// rule the chud nameplate uses, with this screen's own virtual height instead of the chud's.
///
/// Returns the card's rectangle on screen.
pub fn paint_spectate_card(
    ctx: &eframe::egui::Context,
    painter: &eframe::egui::Painter,
    viewport: eframe::egui::Rect,
    screen: &crate::reach_widget::Screen,
    text: SpectateText<'_>,
    cache: CacheHandle,
    decode: &dyn Fn(u32) -> Option<(Vec<u8>, u32, u32)>,
    // #row-colour  The spectated player's own colour, `[a, r, g, b]`, or `None` when the
    // recording holds no appearance for them. The whole card carries it, not just its end caps --
    // the same rule the scoreboard row follows, and for the same reason: the big plate authors a
    // flat grey background that the engine colours from elsewhere.
    player_color: Option<[f32; 4]>,
) -> eframe::egui::Rect {
    use eframe::egui;
    let scale = (viewport.height() / CUI_VIRTUAL[1]).max(0.01);
    let x0 = viewport.center().x - CUI_VIRTUAL[0] * 0.5 * scale;
    let Some(item) = screen.get(SPECTATE_ITEM) else { return egui::Rect::NOTHING };
    let base = screen.origin_of(item);
    let at = |v: [f32; 2]| egui::pos2(x0 + (base[0] + v[0]) * scale, viewport.min.y + (base[1] + v[1]) * scale);
    let size = spectate_size(screen);
    let card = egui::Rect::from_min_size(at([0.0, 0.0]), egui::vec2(size[0] * scale, size[1] * scale));

    // `english_body_text_font` = `tv_nord_condensed-15`: `prop_font_id` 7 on both text boxes.
    let font = crate::reach_font::ui_atlas(ctx);
    for (c, ro) in spectate_components(screen) {
        let rect = egui::Rect::from_min_size(at(ro), egui::vec2(c.width() * scale, c.height() * scale));
        let alpha = c.opacity().clamp(0.0, 1.0);
        // The card's BODY plates take the player's colour over their authored grey; its text,
        // its icons and its end caps keep what the tag says.
        let name = c.name.to_ascii_lowercase();
        let is_plate = name.contains("gray_field") || name.contains("gray_dpad_field") || name.contains("_base_field");
        let tint = match player_color.filter(|_| is_plate) {
            Some(p) => p,
            None => c.tint().unwrap_or([1.0, 1.0, 1.0, 1.0]),
        };
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let col = egui::Color32::from_rgba_unmultiplied(q(tint[1]), q(tint[2]), q(tint[3]), q(tint[0] * alpha));
        match c.kind.as_str() {
            "bitmap_widget" => {
                let Some(tag) = c.bitmap() else { continue };
                let Some(tex) = crate::reach_widget::bitmap_texture(ctx, cache, tag, decode) else { continue };
                let mut mesh = egui::Mesh::with_texture(tex.id());
                mesh.add_rect_with_uv(rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), col);
                painter.add(egui::Shape::mesh(mesh));
            }
            "text_widget" => {
                // Only the two strings the RECORDING can fill. Any other text box in this screen
                // names a localized `prop_text_id` that belongs to the live Theater UI.
                let s = match c.name.as_str() {
                    "text_player_name" => text.name,
                    "text_payer_tag" => text.service_tag,
                    _ => continue,
                };
                if s.is_empty() {
                    continue;
                }
                match font.as_deref() {
                    Some(f) => {
                        let w = f.font.measure(s) * scale;
                        // Centre the line box in the authored box, then put the baseline an
                        // ascender below its top -- the box is 22 units for a 15-point font.
                        let line = f.font.line_height() * scale;
                        let bl = rect.min.y + (rect.height() - line) * 0.5 + f.font.ascender as f32 * scale;
                        let x = match c.justification() {
                            2 => rect.max.x - w,
                            1 => rect.center().x - w * 0.5,
                            _ => rect.min.x,
                        };
                        f.paint(painter, egui::pos2(x, bl), s, col, scale);
                    }
                    None => {
                        let g = painter.layout_no_wrap(s.to_string(), egui::FontId::proportional(15.0 * scale), col);
                        let x = match c.justification() {
                            2 => rect.max.x - g.size().x,
                            1 => rect.center().x - g.size().x * 0.5,
                            _ => rect.min.x,
                        };
                        painter.galley(egui::pos2(x, rect.center().y - g.size().y * 0.5), g, col);
                    }
                }
            }
            _ => {}
        }
    }
    card
}

/// `--dump-spectate <map>`: the spectated-player card exactly as the tag authors it.
pub fn dump_spectate(native: &NativeDll, cache: CacheHandle) {
    let Some(screen) = spectate_screen(native, cache) else {
        println!("no '{FILM_CAMERAS_SCREEN}' in this cache (or it has no '{SPECTATE_ITEM}')");
        return;
    };
    let item = screen.get(SPECTATE_ITEM).expect("checked by spectate_screen");
    let base = screen.origin_of(item);
    println!("# {FILM_CAMERAS_SCREEN}");
    println!("  virtual screen {:?} (the screen component authors no size; see CUI_VIRTUAL)", CUI_VIRTUAL);
    println!("  '{SPECTATE_ITEM}' at ({:.1},{:.1}) size {:.1} x {:.1}  visible={}", base[0], base[1], item.width(), item.height(), item.visible());
    let drawn_size = spectate_size(&screen);
    println!(
        "  DRAWN extent {:.1} x {:.1}, so its centre is {:.1} of {:.1}",
        drawn_size[0],
        drawn_size[1],
        base[0] + drawn_size[0] * 0.5,
        CUI_VIRTUAL[0]
    );
    println!("\n  DRAWN (origins relative to the item):");
    for (c, ro) in spectate_components(&screen) {
        let extra = match c.kind.as_str() {
            "bitmap_widget" => format!(
                "  bitmap {:?} tint {:?}",
                c.bitmap().and_then(|t| native.tag_name(cache, t)),
                c.tint()
            ),
            "text_widget" => format!(
                "  font {:?} justification {} text_id {:?}",
                c.font_id().map(|f| f.file_stem()),
                c.justification(),
                c.text_id()
            ),
            _ => String::new(),
        };
        println!(
            "    {:<28} {:<16} at ({:6.1},{:5.1}) size {:6.1} x {:5.1}{extra}",
            c.name, c.kind, ro[0], ro[1], c.width(), c.height()
        );
    }
    println!("\n  NOT DRAWN (and why -- see SPECTATE_SKIP):");
    for n in SPECTATE_SKIP {
        match screen.get(n) {
            Some(c) => println!("    {:<28} {:<16} size {:.1} x {:.1}", c.name, c.kind, c.width(), c.height()),
            None => println!("    {n}: NOT IN THIS TAG (the skip list is stale)"),
        }
    }
    let s = spectate_size(&screen);
    for vh in [720.0f32, 1080.0, 1440.0] {
        println!("  a {vh:.0}-point viewport draws the card {:.0} x {:.0} points", s[0] * vh / CUI_VIRTUAL[1], s[1] * vh / CUI_VIRTUAL[1]);
    }
}

// ---------------------------------------------------------------------------------------------
// #crosshair  the weapon's own reticule
// ---------------------------------------------------------------------------------------------

/// `weap` +0x3B8 `player interface` +0x00C `chud interface` — the `chud_definition` that authors
/// this weapon's HUD, reticule included. (`+0x01C` is `alt state chud interface`, the zoomed
/// variant; HMS draws the resting one.) Offsets from
/// `reverseme tagdefs reach_tag_test.exe weap`.
pub const WEAPON_CHUD_INTERFACE: usize = 0x3B8 + 0x0C;

/// `weap` +0x3D4 `alt state chud interface` — the ZOOMED variant. Every shipped weapon leaves it
/// null (measured: the reference reads group `0xFFFFFFFF` with datum `0xFFFFFFFF` on all of
/// them), so the zoomed HUD is the resting one with its `zoomed` widgets switched on, not a
/// second tag.
pub const WEAPON_ALT_CHUD_INTERFACE: usize = 0x3B8 + 0x1C;

/// The `chdt` a weapon authors its HUD in, or `None` when it references none.
///
/// Read at the reference's OWN offset rather than out of a 0x400-byte slab, because
/// [`NativeDll::tag_meta`] returns `None` unless the whole requested length is readable — so a
/// wide read is a silent all-or-nothing failure for a field near the end of a tag.
pub fn weapon_chud(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> Option<u32> {
    let r = native.tag_meta(cache, weap_tag, WEAPON_CHUD_INTERFACE as u32, 0x10)?;
    tag_ref(&r, 0)
}

/// The widget collection a weapon's reticule lives in, and the anchor its widgets use
/// (`chud_anchor_type_enum` 17 `crosshair`).
pub const CROSSHAIR_COLLECTION: &str = "crosshair";
pub const ANCHOR_CROSSHAIR: u8 = 17;

/// One bitmap widget of a reticule.
#[derive(Clone, Debug)]
pub struct CrosshairPart {
    /// The widget's own name in the tag (`ar_reticule`, `carbine_crosshair_l`, `headshot`, …).
    pub widget: String,
    /// The sprite: which sheet, where in it, and how big in chud virtual units.
    pub sprite: NavpointSprite,
    /// `widget scale` from the placement block for the full-screen widescreen window state. A
    /// NEGATIVE component means the sprite is mirrored on that axis — the spartan laser authors
    /// `-1.05` to reuse one half of its reticule for both sides.
    pub scale: [f32; 2],
    /// `origin offset` from that same placement, in virtual units.
    pub offset: [f32; 2],
    /// `color output A` — the index into [`SKIN_COLORS`] this part draws in.
    pub color: Option<usize>,
    /// `widget origin`: the PIVOT, the point of the sprite (0..1) that sits on the placement point.
    pub origin: [f32; 2],
    /// `chud_widget_bitmap_flags` bits 0 / 1, `mirror horizontal` / `mirror vertical`: the sprite is
    /// ALSO drawn reflected about the placement point. The assault rifle's ring is one quadrant
    /// authored with both set and its corner as the pivot.
    pub mirror: [bool; 2],
}

impl CrosshairPart {
    /// This part's drawn size in chud virtual units, sprite size times the authored scale. The
    /// magnitude: a mirrored part is the same size, drawn flipped.
    pub fn size_vu(&self) -> [f32; 2] {
        [(self.sprite.size[0] * self.scale[0]).abs(), (self.sprite.size[1] * self.scale[1]).abs()]
    }
}

/// One weapon's resting reticule: every bitmap widget of its `crosshair` collection that the
/// engine draws with no condition on it.
///
/// A reticule is COMPOSED, which is why taking one widget is wrong: the needle rifle authors
/// `carbine_crosshair_l` and `_r` as two halves, the magnum a ring plus a `headshot` pip, the
/// assault rifle one `ar_reticule`. All of those carry no Active State, so all of them are always
/// on screen, and HMS draws the same set.
#[derive(Clone, Debug)]
pub struct Crosshair {
    /// The `chdt` it was authored in, for the diagnostic to name.
    pub chud_tag: u32,
    pub parts: Vec<CrosshairPart>,
}

/// A decoded reticule plus one uploaded sprite per part.
pub struct CrosshairArt {
    pub crosshair: Crosshair,
    /// Parallel to `crosshair.parts`: that part's texture and its (possibly mirrored) UV rect.
    pub parts: Vec<(eframe::egui::TextureHandle, eframe::egui::Rect)>,
}

impl Crosshair {
    /// The whole reticule's extent in virtual units, for the diagnostic and the gates.
    pub fn size_vu(&self) -> [f32; 2] {
        self.parts.iter().fold([0.0f32, 0.0f32], |a, p| {
            let s = p.size_vu();
            [a[0].max(p.offset[0].abs() * 2.0 + s[0]), a[1].max(p.offset[1].abs() * 2.0 + s[1])]
        })
    }
}

/// Does this widget draw in the HUD's RESTING state — a living player holding this weapon,
/// unscoped, with nothing in particular under the reticule?
///
/// `chud_widget_base_struct` +0x00C is the `state data` struct; its block lives at +0x01C with
/// 56-byte elements, and inside one of those `+0x000 Active State` is a block of AND terms, each
/// term a block of OR conditions (`+0x018 Hidden State` is the same shape and suppresses the
/// widget). [`crate::reach_hud::widget_draws_at_rest`] evaluates that tree against
/// [`crate::reach_hud::RESTING_STATES`]; this is a thin alias so there is only ever ONE copy of
/// the rule.
///
/// **This used to ask "is it unconditional", and that was wrong for most of the game.** A
/// `--dump-crosshair` over `forge_halo` reported a reticule for 29 of 64 weapons and none for the
/// other 34 — and the 34 were the entire player sandbox (magnum, DMR, assault rifle, shotgun,
/// sniper rifle, needle rifle, plasma pistol, rocket launcher, concussion rifle, focus rifle,
/// energy sword, gravity hammer...) while the 29 were vehicle and turret weapons. The tags were
/// never the problem: `ui\chud\magnum` authors a six-part reticule, and every part of it is gated
/// on `97 'weapon is right hand'` OR `98 'weapon is left hand'`. Demanding an EMPTY Active state
/// threw all six away. The reference at `weap` +0x3C4 was correct all along — measured, the
/// magnum's reads group `'chdt'` with a live datum — and so was the offset.
pub fn widget_is_unconditional(native: &NativeDll, cache: CacheHandle, widget_base: &[u8]) -> bool {
    crate::reach_hud::widget_draws_at_rest(native, cache, widget_base)
}

/// Is any of this widget's placements anchored at `chud_anchor_type_enum` 17 `crosshair`?
pub fn widget_uses_crosshair_anchor(native: &NativeDll, cache: CacheHandle, widget_base: &[u8]) -> bool {
    block(native, cache, widget_base, 0x28 + 0x10, 28, 5).chunks_exact(28).any(|p| p[1] == ANCHOR_CROSSHAIR)
}

/// Read a weapon's resting reticule out of its own `chud interface`.
///
/// The collection is the one named `crosshair`, else the first whose placement uses anchor 17.
/// Within it HMS takes EVERY bitmap widget the engine draws unconditionally, because a reticule
/// is composed of several; the `l_`/`r_`/`t_`/`b_crosshair` spread bars carry an Active State and
/// are skipped, since a film replicates no spread for them to open by.
pub fn weapon_crosshair(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> Option<Crosshair> {
    let meta = native.tag_meta(cache, weap_tag, 0, 0x400)?;
    let chud = tag_ref(&meta, WEAPON_CHUD_INTERFACE)?;
    chud_crosshair(native, cache, chud)
}

/// The same read, from a `chdt` directly -- which is what a SEAT names (`reach_unit_hud`): a
/// Warthog driver's reticule is the `crosshair` collection of `ui\chud\v_driver`, with no weapon
/// anywhere in the chain.
pub fn chud_crosshair(native: &NativeDll, cache: CacheHandle, chud: u32) -> Option<Crosshair> {
    let cmeta = native.tag_meta(cache, chud, 0, 0x28)?;
    let colls = block(native, cache, &cmeta, 0x00, 220, 32);
    let sid = |v: u32| native.string_id(cache, v).unwrap_or_default();
    // Every collection the engine would draw as a reticule AT REST: `crosshair`, and the weapons
    // that split it (`crosshair_locking` on the rocket launcher, `crosshair_fullscreen` /
    // `_halfscreen` on the plasma turret) -- each judged by its OWN states, so the half-screen twin
    // of a full-screen reticle is not drawn on top of it.
    let mut chosen: Vec<&[u8]> = Vec::new();
    for coll in colls.chunks_exact(220) {
        let n = sid(rd_u32(coll, 0)?);
        if (n == CROSSHAIR_COLLECTION || n.starts_with("crosshair_")) && crate::reach_hud::widget_draws_at_rest(native, cache, coll) {
            chosen.push(coll);
        }
    }
    if chosen.is_empty() {
        chosen.extend(colls.chunks_exact(220).find(|c| {
            block(native, cache, c, 0x98, 156, 32)
                .chunks_exact(156)
                .any(|b| block(native, cache, b, 0x28 + 0x10, 28, 5).chunks_exact(28).any(|p| p[1] == ANCHOR_CROSSHAIR))
        }));
    }
    if chosen.is_empty() {
        return None;
    }
    let bitmaps: Vec<u8> = chosen.iter().flat_map(|coll| block(native, cache, coll, 0x98, 156, 32)).collect();
    // EVERY bitmap widget the engine draws with no condition on it, not just the first.
    //
    // A reticule is COMPOSED. Taking "the first" picks whatever leads in tag order, which on the
    // shipped weapons is often the wrong thing: the magnum authors a ring PLUS a `headshot` pip,
    // the needle rifle `carbine_crosshair_l` and `_r` as two halves of one reticule, the assault
    // rifle a single `ar_reticule`. A widget carrying an Active State (the four `l_`/`r_`/`t_`/
    // `b_crosshair` spread bars that open with the weapon's bloom, the grenade launcher's
    // 186-unit trajectory arc) is a STATE widget a film cannot drive, and is skipped.
    let part_of = |w: &[u8]| -> Option<CrosshairPart> {
        let bitmap_tag = tag_ref(w, 0x80)?;
        let sequence = w[0x90] as i8;
        if sequence < 0 {
            return None;
        }
        // The placement for the full-screen widescreen window state, else the first authored one.
        let place = block(native, cache, w, 0x28 + 0x10, 28, 5);
        let p = place
            .chunks_exact(28)
            .find(|p| p[0] & RES_FULLSCREEN_WIDE != 0)
            .or_else(|| place.chunks_exact(28).next())?;
        let render = block(native, cache, w, 0x60 + 0x10, 120, 1);
        let color = render.chunks_exact(120).next().and_then(|r| {
            let v = i16::from_le_bytes(r[0x54..0x56].try_into().ok()?) as i32;
            (v >= COLOR_OUTPUT_BASE).then(|| (v - COLOR_OUTPUT_BASE) as usize).filter(|k| *k < SKIN_COLORS.len())
        });
        Some(CrosshairPart {
            widget: sid(rd_u32(w, 0)?),
            sprite: read_sprite(native, cache, bitmap_tag, sequence as usize, 0)?,
            scale: [rd_f32(p, 0x14).unwrap_or(1.0), rd_f32(p, 0x18).unwrap_or(1.0)],
            offset: [rd_f32(p, 0x0C).unwrap_or(0.0), rd_f32(p, 0x10).unwrap_or(0.0)],
            color,
            origin: [rd_f32(p, 0x04).unwrap_or(0.5), rd_f32(p, 0x08).unwrap_or(0.5)],
            mirror: {
                let f = rd_u32(w, 0x7C).unwrap_or(0);
                [f & 1 != 0, f & 2 != 0]
            },
        })
    };
    let anchored: Vec<CrosshairPart> = bitmaps
        .chunks_exact(156)
        .filter(|w| widget_is_unconditional(native, cache, w) && widget_uses_crosshair_anchor(native, cache, w))
        .filter_map(part_of)
        .collect();
    // A collection chosen by NAME may hold its widgets without the anchor; fall back to every
    // unconditional widget in it rather than drawing nothing.
    let parts = if anchored.is_empty() {
        bitmaps
            .chunks_exact(156)
            .filter(|w| widget_is_unconditional(native, cache, w))
            .filter_map(part_of)
            .collect()
    } else {
        anchored
    };
    (!parts.is_empty()).then_some(Crosshair { chud_tag: chud, parts })
}

thread_local! {
    /// One decoded reticule per (cache, weapon). `None` = that weapon authors none.
    static CROSSHAIRS: std::cell::RefCell<std::collections::HashMap<(u64, u32), Option<std::rc::Rc<CrosshairArt>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The same per (cache, chdt): the reticule a SEAT's own `chud` authors.
    static CHUD_CROSSHAIRS: std::cell::RefCell<std::collections::HashMap<(u64, u32), Option<std::rc::Rc<CrosshairArt>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The open cache's reticule for one weapon, decoded and uploaded once.
pub fn crosshair_for(
    ctx: &eframe::egui::Context,
    native: &NativeDll,
    cache: CacheHandle,
    weap_tag: u32,
) -> Option<std::rc::Rc<CrosshairArt>> {
    CROSSHAIRS.with(|m| {
        if let Some(v) = m.borrow().get(&(cache.0, weap_tag)) {
            return v.clone();
        }
        let made = weapon_crosshair(native, cache, weap_tag).and_then(|ch| upload_crosshair(ctx, native, cache, ch, weap_tag));
        m.borrow_mut().insert((cache.0, weap_tag), made.clone());
        made
    })
}

/// [`crosshair_for`], for the reticule a `chdt` authors directly (a seat's HUD).
pub fn crosshair_for_chud(
    ctx: &eframe::egui::Context,
    native: &NativeDll,
    cache: CacheHandle,
    chud: u32,
) -> Option<std::rc::Rc<CrosshairArt>> {
    CHUD_CROSSHAIRS.with(|m| {
        if let Some(v) = m.borrow().get(&(cache.0, chud)) {
            return v.clone();
        }
        let made = chud_crosshair(native, cache, chud).and_then(|ch| upload_crosshair(ctx, native, cache, ch, chud));
        m.borrow_mut().insert((cache.0, chud), made.clone());
        made
    })
}

/// Upload one decoded reticule's sprites.
fn upload_crosshair(
    ctx: &eframe::egui::Context,
    native: &NativeDll,
    cache: CacheHandle,
    ch: Crosshair,
    key: u32,
) -> Option<std::rc::Rc<CrosshairArt>> {
    use eframe::egui;
    // One texture + UV PER PART: a composed reticule's halves can come from different
    // sheets, and even on one sheet each part is its own sprite rectangle.
    let mut art: Vec<(egui::TextureHandle, egui::Rect)> = Vec::with_capacity(ch.parts.len());
    for (i, part) in ch.parts.iter().enumerate() {
        let sp = part.sprite;
        // The reticule sheet is white coverage like the navpoint one: the ALPHA is the shape.
        let (bgra, w, h) = native.decode_bitmap_keep_alpha(cache, sp.bitmap_tag, sp.bitmap_index as u32, 0)?;
        if w == 0 || h == 0 || bgra.len() < (w as usize * h as usize * 4) {
            return None;
        }
        let px: Vec<egui::Color32> = bgra
            .chunks_exact(4)
            .take(w as usize * h as usize)
            .map(|c| egui::Color32::from_rgba_unmultiplied(255, 255, 255, c[3]))
            .collect();
        let img = egui::ColorImage { size: [w as usize, h as usize], pixels: px };
        let tex = ctx.load_texture(format!("reach-reticule-{key}-{i}"), img, egui::TextureOptions::LINEAR);
        // A NEGATIVE authored scale mirrors the sprite on that axis (the spartan laser
        // reuses one half for both sides), which is a flipped UV rectangle.
        let (u0, u1) = if part.scale[0] < 0.0 { (sp.uv[1], sp.uv[0]) } else { (sp.uv[0], sp.uv[1]) };
        let (v0, v1) = if part.scale[1] < 0.0 { (sp.uv[3], sp.uv[2]) } else { (sp.uv[2], sp.uv[3]) };
        art.push((tex, egui::Rect::from_min_max(egui::pos2(u0, v0), egui::pos2(u1, v1))));
    }
    Some(std::rc::Rc::new(CrosshairArt { crosshair: ch, parts: art }))
}

/// Paint the resting reticule at the crosshair anchor, scaled the way the chud scales everything.
///
/// `chud_anchor_type_enum` 17 `crosshair` is the aim point, which for a first-person view is the
/// centre of the viewport; the widget's own `origin offset` is (0, 0) on every weapon's main
/// reticule, so it sits dead centre. Its size is the sprite's virtual size times the authored
/// `widget scale` times [`px_per_virtual`] — the same virtual-screen rule the nameplate uses, so
/// the reticule grows with the window exactly as the game's does.
pub fn paint_crosshair(
    painter: &eframe::egui::Painter,
    viewport: eframe::egui::Rect,
    art: &CrosshairArt,
    skin: &ChudSkin,
    vpx: f32,
) -> eframe::egui::Rect {
    use eframe::egui;
    let mut union: Option<egui::Rect> = None;
    for (part, (tex, uv)) in art.crosshair.parts.iter().zip(art.parts.iter()) {
        let size = part.size_vu();
        let c = egui::pos2(
            viewport.center().x + part.offset[0] * vpx,
            viewport.center().y + part.offset[1] * vpx,
        );
        let (w, h) = (size[0] * vpx, size[1] * vpx);
        // `widget origin` is the pivot: the point (0..1) of the sprite that sits on `c`.
        let rect = egui::Rect::from_center_size(
            egui::pos2(c.x + (0.5 - part.origin[0]) * w, c.y + (0.5 - part.origin[1]) * h),
            egui::vec2(w, h),
        );
        // `color output A` is the resting colour -- `crosshair normal` on every shipped weapon,
        // which the skin authors opaque (a = 1.0). The enemy/friendly outputs are a
        // target-under-the-reticule state a film does not replicate.
        let col = match part.color.map(|k| skin.colors[k]) {
            Some(c) => {
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                egui::Color32::from_rgba_unmultiplied(q(c[1]), q(c[2]), q(c[3]), q(c[0]))
            }
            None => egui::Color32::WHITE,
        };
        // The sprite, then its reflections about `c` when `mirror horizontal` / `vertical` are set.
        let mut copies = vec![(rect, *uv)];
        for (axis, on) in part.mirror.iter().enumerate() {
            if !*on {
                continue;
            }
            let more: Vec<(egui::Rect, egui::Rect)> = copies
                .iter()
                .map(|(r, u)| {
                    if axis == 0 {
                        (
                            egui::Rect::from_min_max(egui::pos2(2.0 * c.x - r.max.x, r.min.y), egui::pos2(2.0 * c.x - r.min.x, r.max.y)),
                            egui::Rect::from_min_max(egui::pos2(u.max.x, u.min.y), egui::pos2(u.min.x, u.max.y)),
                        )
                    } else {
                        (
                            egui::Rect::from_min_max(egui::pos2(r.min.x, 2.0 * c.y - r.max.y), egui::pos2(r.max.x, 2.0 * c.y - r.min.y)),
                            egui::Rect::from_min_max(egui::pos2(u.min.x, u.max.y), egui::pos2(u.max.x, u.min.y)),
                        )
                    }
                })
                .collect();
            copies.extend(more);
        }
        for (r, u) in copies {
            let mut mesh = egui::Mesh::with_texture(tex.id());
            mesh.add_rect_with_uv(r, u, col);
            painter.add(egui::Shape::mesh(mesh));
            union = Some(match union {
                Some(un) => un.union(r),
                None => r,
            });
        }
    }
    union.unwrap_or_else(|| egui::Rect::from_center_size(viewport.center(), egui::Vec2::ZERO))
}

/// `--dump-crosshair <map> [weapon substring]`: every weapon's authored reticule.
pub fn dump_crosshair(native: &NativeDll, cache: CacheHandle, filter: &str) {
    let skin = read_globals(native, cache).and_then(|g| g.default_skin().cloned());
    let mut tags = native.tags_by_group(cache, "weap");
    tags.sort_unstable();
    println!("# every weapon's resting reticule, from weap +0x{WEAPON_CHUD_INTERFACE:03X} 'chud interface'\n");
    let mut found = 0;
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        match weapon_crosshair(native, cache, t) {
            Some(ch) => {
                found += 1;
                let s = ch.size_vu();
                // A reticule is COMPOSED, so print every part: the magnum's ring plus its
                // `headshot` pip, the needle rifle's two halves, the assault rifle's single
                // `ar_reticule`.
                println!(
                    "  {:<46} {} part(s), {:5.1}x{:5.1} vu overall",
                    name.rsplit('\\').next().unwrap_or(&name),
                    ch.parts.len(),
                    s[0],
                    s[1]
                );
                for part in &ch.parts {
                    let col = part.color.map(|k| SKIN_COLORS[k]).unwrap_or("<none>");
                    let hex = match (&skin, part.color) {
                        (Some(sk), Some(k)) => {
                            let c = sk.colors[k];
                            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            format!("#{:02X}{:02X}{:02X} a={:02X}", q(c[1]), q(c[2]), q(c[3]), q(c[0]))
                        }
                        _ => String::new(),
                    };
                    let ps = part.size_vu();
                    println!(
                        "      {:<24} {} seq {:<3} sprite {:5.1}x{:5.1} scale ({:+.3},{:+.3}) offset ({:+.1},{:+.1}) -> {:5.1}x{:5.1} vu  {col} {hex}",
                        part.widget,
                        native.tag_name(cache, part.sprite.bitmap_tag).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string(),
                        part.sprite.bitmap_index,
                        part.sprite.size[0],
                        part.sprite.size[1],
                        part.scale[0],
                        part.scale[1],
                        part.offset[0],
                        part.offset[1],
                        ps[0],
                        ps[1]
                    );
                }
            }
            None => println!("  {:<46} (no chud interface, or it authors no reticule)", name.rsplit('\\').next().unwrap_or(&name)),
        }
    }
    println!("\n  {found} weapon(s) with a reticule");
    if let Some(sk) = &skin {
        println!("\n  at a 1080-point viewport one virtual unit is {:.4} points", px_per_virtual(1080.0, sk.virtual_height(), sk.safe_frame_v(), sk.safe_frame_ding_v()));
    }
}
