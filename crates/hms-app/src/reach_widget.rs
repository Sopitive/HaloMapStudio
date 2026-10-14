//! #reach-widget  Halo: Reach's OWN UI widgets, decoded from `cui_screen` (`cusc`) tags.
//!
//! Reach's in-game screens are not drawn by hand: each one is a `cusc` tag holding a tree of named
//! COMPONENTS (containers, bitmaps, text, emblems, progress indicators) and, per
//! resolution/theme OVERLAY, a bag of PROPERTIES for each of them — `prop_left`, `prop_top`,
//! `prop_width`, `prop_height`, `prop_font_id`, `prop_justification`, `prop_text_id`,
//! `prop_bitmap_reference`, `prop_tint_color`, `prop_opacity`. The scoreboard's player list is
//! three of these tags:
//!
//! | tag | what it is |
//! |---|---|
//! | `ui\cui\in_game\scoreboard\scoreboard` | the screen: the panel, the column headers, the list |
//! | `…\templates\scoreboard_list_item_player` | one player's row |
//! | `…\templates\scoreboard_list_item_team` | one team header row |
//!
//! All of the geometry below is read out of those tags at run time; none of it is typed in here.
//! Co-ordinates are in the screen's own virtual space — the `screen` component of each overlay
//! declares it, and for `resolution_widescreen` it is 1152 x 720.
//!
//! LAYOUT (`cusc`, from `reverseme tagdefs reach_tag_test.exe cusc`):
//!
//! ```text
//! cui_screen                                   124 bytes
//!   +0x000 tag reference  string list
//!   +0x010 block          template instantiations  (16 B: a tag reference to another cusc)
//!   +0x01C block          components               (16 B)
//!            +0x00 string id type     (container_widget / bitmap_widget / text_widget / …)
//!            +0x04 string id name
//!            +0x08 string id parent   (empty = a root)
//!            +0x0C word      flags
//!            +0x0E short     template instantiation index
//!   +0x028 block          component indices        (8 B)
//!   +0x034 block          overlays                 (32 B)
//!            +0x00 string id resolution  (resolution_widescreen, …_standard, …_half, …)
//!            +0x04 string id theme       (theme_default / theme_spartan / theme_elite)
//!            +0x08 block     components            (76 B: string id name, then 72 B of properties)
//!            +0x14 block     animations
//!   +0x04C block          property bindings        (20 B)
//! properties_struct                            72 bytes, six parallel property blocks
//!   +0x00 long        (8 B)   {string id name, i32   value}
//!   +0x0C real        (8 B)   {string id name, f32   value}
//!   +0x18 string id   (8 B)   {string id name, string id value}
//!   +0x24 tag ref    (20 B)   {string id name, tag reference value}
//!   +0x30 text      (260 B)   {string id name, char[256] value}
//!   +0x3C argb color (20 B)   {string id name, 4 x f32}
//! ```
//!
//! A property can be authored as either a long or a real (`prop_height` is `30` on one component
//! and `30.976` on another), so [`Props::real`] reads both blocks.

use std::collections::HashMap;

use hms_native::{CacheHandle, NativeDll};

/// The three tags the player list is built from.
pub const SCOREBOARD_SCREEN: &str = "ui\\cui\\in_game\\scoreboard\\scoreboard";
pub const SCOREBOARD_ROW_PLAYER: &str = "ui\\cui\\in_game\\scoreboard\\templates\\scoreboard_list_item_player";
pub const SCOREBOARD_ROW_TEAM: &str = "ui\\cui\\in_game\\scoreboard\\templates\\scoreboard_list_item_team";

/// The overlay HMS reads: the 16:9 base authoring, with the neutral theme.
pub const OVERLAY_RESOLUTION: &str = "resolution_widescreen";
pub const OVERLAY_THEME: &str = "theme_default";

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

/// One component's properties, with the authored names kept so nothing is silently dropped.
#[derive(Clone, Debug, Default)]
pub struct Props {
    pub longs: Vec<(String, i32)>,
    pub reals: Vec<(String, f32)>,
    pub sids: Vec<(String, String)>,
    /// `(property name, group, tag index)` — the tag index is the datum's low 16 bits.
    pub tags: Vec<(String, String, Option<u32>)>,
    pub texts: Vec<(String, String)>,
    pub colors: Vec<(String, [f32; 4])>,
}

impl Props {
    /// A numeric property, from whichever of the two blocks authored it.
    pub fn real(&self, name: &str) -> Option<f32> {
        if let Some((_, v)) = self.reals.iter().find(|(n, _)| n == name) {
            return Some(*v);
        }
        self.longs.iter().find(|(n, _)| n == name).map(|(_, v)| *v as f32)
    }
    pub fn int(&self, name: &str) -> Option<i32> {
        if let Some((_, v)) = self.longs.iter().find(|(n, _)| n == name) {
            return Some(*v);
        }
        self.reals.iter().find(|(n, _)| n == name).map(|(_, v)| *v as i32)
    }
    pub fn sid(&self, name: &str) -> Option<&str> {
        self.sids.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
    pub fn color(&self, name: &str) -> Option<[f32; 4]> {
        self.colors.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
    /// The tag index of a tag-reference property, e.g. `prop_bitmap_reference`.
    pub fn tag(&self, name: &str) -> Option<u32> {
        self.tags.iter().find(|(n, _, _)| n == name).and_then(|(_, _, t)| *t)
    }
}

/// One component of a screen: what it is, where it sits, and everything authored about it.
#[derive(Clone, Debug, Default)]
pub struct Component {
    pub name: String,
    /// `container_widget`, `bitmap_widget`, `text_widget`, `emblem_widget`, …
    pub kind: String,
    pub parent: String,
    pub props: Props,
}

impl Component {
    pub fn left(&self) -> f32 { self.props.real("prop_left").unwrap_or(0.0) }
    pub fn top(&self) -> f32 { self.props.real("prop_top").unwrap_or(0.0) }
    pub fn width(&self) -> f32 { self.props.real("prop_width").unwrap_or(0.0) }
    pub fn height(&self) -> f32 { self.props.real("prop_height").unwrap_or(0.0) }
    pub fn opacity(&self) -> f32 { self.props.real("prop_opacity").unwrap_or(1.0) }
    /// `prop_visible` 0 hides a component and everything under it.
    pub fn visible(&self) -> bool { self.props.int("prop_visible").unwrap_or(1) != 0 }
    pub fn bitmap(&self) -> Option<u32> { self.props.tag("prop_bitmap_reference") }
    pub fn tint(&self) -> Option<[f32; 4]> { self.props.color("prop_tint_color") }
    pub fn font_id(&self) -> Option<crate::reach_font::FontId> {
        self.props.int("prop_font_id").and_then(crate::reach_font::FontId::from_index)
    }
    /// 0 = left, 1 = centre, 2 = right (the values the shipped scoreboard authors).
    pub fn justification(&self) -> i32 { self.props.int("prop_justification").unwrap_or(0) }
    /// The localisation key for a text widget's content, e.g. `players`, `score`, `service_tag`.
    pub fn text_id(&self) -> Option<&str> { self.props.sid("prop_text_id") }
    /// `prop_text_case` 1 = upper case.
    pub fn upper_case(&self) -> bool { self.props.int("prop_text_case").unwrap_or(0) == 1 }
}

/// One `property binding` of a `cusc`: the tag's own wiring from one component's property to
/// another's. `cusc` +0x4C, 20-byte element — `{word flags, short conversion function,
/// string id source component, source property, target component, target property}`
/// (`s_cui_property_binding_definition`, from `reach_tag_test.exe`).
///
/// These are how the scoreboard says what colour a row's plates are: nothing authors
/// `roster_bitmap`'s `prop_tint_color`, a binding feeds it.
#[derive(Clone, Debug, Default)]
pub struct Binding {
    pub src_comp: String,
    pub src_prop: String,
    pub dst_comp: String,
    pub dst_prop: String,
}

/// One decoded `cusc`, for one resolution/theme overlay.
#[derive(Clone, Debug, Default)]
pub struct Screen {
    #[allow(dead_code)] // kept so a diagnostic can say which tag a layout came from
    pub path: String,
    pub components: Vec<Component>,
    /// The tag's own property wiring. Read, not discarded: a plate's colour lives here.
    pub bindings: Vec<Binding>,
    /// The `screen` component's own size — the virtual space every co-ordinate is in.
    pub virtual_size: [f32; 2],
}

impl Screen {
    pub fn get(&self, name: &str) -> Option<&Component> {
        self.components.iter().find(|c| c.name == name)
    }
    /// A component's origin in virtual-screen space, by walking its parent chain.
    ///
    /// Reach's components are authored as offsets from their parent (`prop_left` / `prop_top`),
    /// which is why a row template can be stamped anywhere without touching its contents.
    pub fn origin_of(&self, c: &Component) -> [f32; 2] {
        let (mut x, mut y) = (c.left(), c.top());
        let mut parent = c.parent.clone();
        // The tree is a handful of levels deep; the bound stops a malformed cycle dead.
        for _ in 0..16 {
            if parent.is_empty() {
                break;
            }
            let Some(p) = self.get(&parent) else { break };
            x += p.left();
            y += p.top();
            parent = p.parent.clone();
        }
        [x, y]
    }
    /// What the tag wires into `component.property`, if anything: `(source component, source
    /// property)`.
    pub fn bound_from(&self, component: &str, property: &str) -> Option<(&str, &str)> {
        self.bindings
            .iter()
            .find(|b| b.dst_comp == component && b.dst_prop == property)
            .map(|b| (b.src_comp.as_str(), b.src_prop.as_str()))
    }
    /// True when this component or any ancestor is `prop_visible` 0.
    pub fn hidden(&self, c: &Component) -> bool {
        if !c.visible() {
            return true;
        }
        let mut parent = c.parent.clone();
        for _ in 0..16 {
            if parent.is_empty() {
                break;
            }
            let Some(p) = self.get(&parent) else { break };
            if !p.visible() {
                return true;
            }
            parent = p.parent.clone();
        }
        false
    }
}

/// Read one `cusc` tag's components and the properties of one overlay.
pub fn read_screen(native: &NativeDll, cache: CacheHandle, tag: u32, resolution: &str, theme: &str) -> Option<Screen> {
    let meta = native.tag_meta(cache, tag, 0, 124)?;
    let sid = |s: u32| -> String {
        if s == 0 || s == u32::MAX { String::new() } else { native.string_id(cache, s).unwrap_or_default() }
    };
    let mut out = Screen { path: native.tag_name(cache, tag).unwrap_or_default(), ..Default::default() };
    // components
    let comps = block(native, cache, &meta, 0x1C, 16, 8192);
    for e in comps.chunks_exact(16) {
        out.components.push(Component {
            kind: sid(rd_u32(e, 0)?),
            name: sid(rd_u32(e, 4)?),
            parent: sid(rd_u32(e, 8)?),
            props: Props::default(),
        });
    }
    // the chosen overlay's per-component properties
    let overlays = block(native, cache, &meta, 0x34, 32, 24);
    let mut chosen: Option<&[u8]> = None;
    for ov in overlays.chunks_exact(32) {
        let (r, t) = (sid(rd_u32(ov, 0)?), sid(rd_u32(ov, 4)?));
        if r == resolution && t == theme {
            chosen = Some(ov);
            break;
        }
    }
    // the property bindings — the tag's own wiring, which the drawing code needs
    for e in block(native, cache, &meta, 0x4C, 20, 8192).chunks_exact(20) {
        out.bindings.push(Binding {
            src_comp: sid(rd_u32(e, 0x04)?),
            src_prop: sid(rd_u32(e, 0x08)?),
            dst_comp: sid(rd_u32(e, 0x0C)?),
            dst_prop: sid(rd_u32(e, 0x10)?),
        });
    }
    let ov = chosen?;
    let props = block(native, cache, &ov[..], 0x08, 76, 8192);
    for e in props.chunks_exact(76) {
        let name = sid(rd_u32(e, 0)?);
        let p = &e[4..];
        let mut out_props = Props::default();
        for (_, v) in block(native, cache, p, 0x00, 8, 8192).chunks_exact(8).map(|c| (0, c)) {
            out_props.longs.push((sid(rd_u32(v, 0)?), rd_i32(v, 4)?));
        }
        for v in block(native, cache, p, 0x0C, 8, 8192).chunks_exact(8) {
            out_props.reals.push((sid(rd_u32(v, 0)?), rd_f32(v, 4)?));
        }
        for v in block(native, cache, p, 0x18, 8, 8192).chunks_exact(8) {
            out_props.sids.push((sid(rd_u32(v, 0)?), sid(rd_u32(v, 4)?)));
        }
        for v in block(native, cache, p, 0x24, 20, 8192).chunks_exact(20) {
            // tag reference: group fourcc (byte-reversed on disk) at +0x04, datum index at +0x10.
            let group: String = v[4..8].iter().rev().map(|&b| b as char).collect();
            let datum = rd_u32(v, 4 + 0x0C)?;
            let t = (datum != u32::MAX).then(|| datum & 0xFFFF);
            out_props.tags.push((sid(rd_u32(v, 0)?), group, t));
        }
        for v in block(native, cache, p, 0x30, 260, 8192).chunks_exact(260) {
            let end = v[4..].iter().position(|&c| c == 0).unwrap_or(256);
            out_props.texts.push((sid(rd_u32(v, 0)?), String::from_utf8_lossy(&v[4..4 + end]).to_string()));
        }
        for v in block(native, cache, p, 0x3C, 20, 8192).chunks_exact(20) {
            out_props
                .colors
                .push((sid(rd_u32(v, 0)?), [rd_f32(v, 4)?, rd_f32(v, 8)?, rd_f32(v, 12)?, rd_f32(v, 16)?]));
        }
        // `screen` is the overlay's own frame rather than a component of the tree.
        if name == "screen" {
            out.virtual_size = [out_props.real("prop_width").unwrap_or(0.0), out_props.real("prop_height").unwrap_or(0.0)];
        }
        if let Some(c) = out.components.iter_mut().find(|c| c.name == name) {
            c.props = out_props;
        }
    }
    Some(out)
}

/// The three screens the player list needs, read once per map.
#[derive(Clone, Debug)]
pub struct Scoreboard {
    pub screen: Screen,
    pub player_row: Screen,
    #[allow(dead_code)] // decoded and ready; team header rows are not drawn yet
    pub team_row: Screen,
}

impl Scoreboard {
    /// The list's own frame on the screen, in virtual units: `(origin, size)` of
    /// `list_scoreboard`, with the header block above it.
    #[allow(dead_code)]
    pub fn list_rect(&self) -> ([f32; 2], [f32; 2]) {
        match self.screen.get("list_scoreboard") {
            Some(c) => (self.screen.origin_of(c), [c.width(), c.height()]),
            None => ([0.0, 60.0], [349.0, 560.0]),
        }
    }
    /// One row's height in virtual units, from the row template's own root.
    pub fn row_height(&self) -> f32 {
        self.player_row.get("list_item_player").map(|c| c.height()).filter(|h| *h > 0.0).unwrap_or(30.0)
    }
    pub fn row_width(&self) -> f32 {
        self.player_row.get("list_item_player").map(|c| c.width()).filter(|w| *w > 0.0).unwrap_or(350.0)
    }
}

/// Find and decode the scoreboard of the open Reach cache.
pub fn read_scoreboard(native: &NativeDll, cache: CacheHandle) -> Option<Scoreboard> {
    let find = |path: &str| -> Option<u32> {
        native
            .tags_by_group(cache, "cusc")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).map_or(false, |n| n.eq_ignore_ascii_case(path)))
    };
    Some(Scoreboard {
        screen: read_screen(native, cache, find(SCOREBOARD_SCREEN)?, OVERLAY_RESOLUTION, OVERLAY_THEME)?,
        player_row: read_screen(native, cache, find(SCOREBOARD_ROW_PLAYER)?, OVERLAY_RESOLUTION, OVERLAY_THEME)?,
        team_row: read_screen(native, cache, find(SCOREBOARD_ROW_TEAM)?, OVERLAY_RESOLUTION, OVERLAY_THEME)?,
    })
}

/// [tag] Reach's OWN UI PALETTE: `ui\cui\ui_shared_globals` (`wigl`,
/// `user_interface_shared_globals_definition`, 372 bytes).
///
/// This is what a `cui` text widget's `prop_color_preset` names. The engine resolves it in
/// `cui_color_preset.cpp:18` — reject `NONE`, then a linear scan of the `color presets` block at
/// `wigl` +0x94 (20-byte elements, `{string id name, real_argb_color}`) comparing the name, and
/// on a miss the widget KEEPS THE DEFAULT it pre-loaded, which is `wigl` +0x74 `text color`.
/// So a text widget with no preset is not white — it is whatever that tag says.
///
/// The same tag holds the four UI TINT lists at +0xA0, which is where
/// `prop_player_bitmap_tint_color` comes from: `text player`, `text team`, `bitmap player`,
/// `bitmap team`, each a list of `real_argb_color` (16 bytes). These are the UI's own per-team
/// and per-player colours and they are NOT `mulg`'s in-world team colours.
///
/// LAYOUT (`reverseme tagdefs reach_tag_test.exe wigl`):
///
/// ```text
/// wigl +0x074 real argb color  text color          <- the default a text widget starts from
///      +0x084 real argb color  shadow color
///      +0x094 block            color presets [64]  20 B {string id name, real argb color}
///      +0x0A0 block            tint colors [1]     48 B, four parallel lists:
///               +0x00 block  text player     16 B real argb color
///               +0x0C block  text team
///               +0x18 block  bitmap player
///               +0x24 block  bitmap team
///      +0x0AC short           primary emblem count    (80)
///      +0x0AE short           secondary emblem count  (53)
/// ```
#[derive(Clone, Debug, Default)]
pub struct UiGlobals {
    /// +0x74 `text color` — the colour a text widget draws in when it names no preset.
    pub text_color: [f32; 4],
    /// +0x84 `shadow color`.
    pub shadow_color: [f32; 4],
    /// +0x94 `color presets`, in tag order.
    pub presets: Vec<(String, [f32; 4])>,
    /// +0xA0 the four tint lists, in tag order: text player, text team, bitmap player, bitmap team.
    pub tint_text_player: Vec<[f32; 4]>,
    pub tint_text_team: Vec<[f32; 4]>,
    pub tint_bitmap_player: Vec<[f32; 4]>,
    pub tint_bitmap_team: Vec<[f32; 4]>,
    /// +0xAC / +0xAE — how many foreground and background emblem designs the game has.
    pub primary_emblem_count: i16,
    pub secondary_emblem_count: i16,
}

impl UiGlobals {
    /// A named colour preset, or `None` when the tag has no such name — which is exactly when
    /// the engine keeps its default.
    pub fn preset(&self, name: &str) -> Option<[f32; 4]> {
        self.presets.iter().find(|(n, _)| n == name).map(|(_, c)| *c)
    }
    /// The colour a text widget draws in: its preset when the tag defines one, else the tag's
    /// own `text color` default. This is the engine's rule, not a guess.
    pub fn text_preset_or_default(&self, preset: Option<&str>) -> [f32; 4] {
        preset.and_then(|p| self.preset(p)).unwrap_or(self.text_color)
    }
    /// One entry of the `bitmap team` tint list, which is what the scoreboard's colour bars take.
    pub fn bitmap_team(&self, team: u8) -> Option<[f32; 4]> {
        self.tint_bitmap_team.get(team as usize).copied()
    }
    pub fn bitmap_player(&self, slot: u8) -> Option<[f32; 4]> {
        self.tint_bitmap_player.get(slot as usize).copied()
    }
}

fn argb(b: &[u8], o: usize) -> [f32; 4] {
    [
        rd_f32(b, o).unwrap_or(0.0),
        rd_f32(b, o + 4).unwrap_or(0.0),
        rd_f32(b, o + 8).unwrap_or(0.0),
        rd_f32(b, o + 12).unwrap_or(0.0),
    ]
}

/// Read the open cache's `wigl`.
pub fn read_ui_globals(native: &NativeDll, cache: CacheHandle) -> Option<UiGlobals> {
    let tag = native.tags_by_group(cache, "wigl").first().copied()?;
    let meta = native.tag_meta(cache, tag, 0, 372)?;
    let mut out = UiGlobals {
        text_color: argb(&meta, 0x74),
        shadow_color: argb(&meta, 0x84),
        primary_emblem_count: i16::from_le_bytes(meta.get(0xAC..0xAE)?.try_into().ok()?),
        secondary_emblem_count: i16::from_le_bytes(meta.get(0xAE..0xB0)?.try_into().ok()?),
        ..Default::default()
    };
    for e in block(native, cache, &meta, 0x94, 20, 64).chunks_exact(20) {
        let name = rd_u32(e, 0).unwrap_or(0);
        let name = if name == 0 || name == u32::MAX { String::new() } else { native.string_id(cache, name).unwrap_or_default() };
        out.presets.push((name, argb(e, 4)));
    }
    let tints = block(native, cache, &meta, 0xA0, 48, 1);
    let list = |off: usize| -> Vec<[f32; 4]> {
        block(native, cache, &tints, off, 16, 32).chunks_exact(16).map(|c| argb(c, 0)).collect()
    };
    if !tints.is_empty() {
        out.tint_text_player = list(0x00);
        out.tint_text_team = list(0x0C);
        out.tint_bitmap_player = list(0x18);
        out.tint_bitmap_team = list(0x24);
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// the game's own bitmaps, as egui textures
// ---------------------------------------------------------------------------------------------

use eframe::egui;

/// [tag] Reach's own TEAM COLOURS: `multiplayer\multiplayer_globals` (`mulg`) +0x00 `universal`
/// (76-byte element) +0x20 `team colors`, eleven `real_rgb_color`s — red, blue, green, orange,
/// purple, gold, brown, pink, white, black, olive, in the engine's team order.
pub fn team_colors(native: &NativeDll, cache: CacheHandle) -> Vec<egui::Color32> {
    let Some(tag) = native.tags_by_group(cache, "mulg").first().copied() else { return Vec::new() };
    let Some(meta) = native.tag_meta(cache, tag, 0, 0x60) else { return Vec::new() };
    let uni = block(native, cache, &meta, 0x00, 76, 1);
    let cols = block(native, cache, &uni, 0x20, 12, 32);
    cols.chunks_exact(12)
        .filter_map(|c| {
            let f = |o: usize| rd_f32(c, o).unwrap_or(0.0).clamp(0.0, 1.0);
            Some(egui::Color32::from_rgb(
                (f(0) * 255.0 + 0.5) as u8,
                (f(4) * 255.0 + 0.5) as u8,
                (f(8) * 255.0 + 0.5) as u8,
            ))
        })
        .collect()
}

thread_local! {
    /// One decoded scoreboard per open cache. The tags are identical on every Reach map (they
    /// come from the shared UI paging), but keying on the handle keeps a reopened map honest.
    static SCOREBOARD: std::cell::RefCell<HashMap<u64, Option<std::rc::Rc<Scoreboard>>>> =
        std::cell::RefCell::new(HashMap::new());
    /// `(cache, bitmap tag) -> texture`. `None` = tried and could not be decoded.
    static BITMAPS: std::cell::RefCell<HashMap<(u64, u32), Option<egui::TextureHandle>>> =
        std::cell::RefCell::new(HashMap::new());
    /// One decoded `wigl` palette per open cache.
    static UI_GLOBALS: std::cell::RefCell<HashMap<u64, Option<std::rc::Rc<UiGlobals>>>> =
        std::cell::RefCell::new(HashMap::new());
    /// `matg` `profile colors` per open cache, in LINEAR RGB, read once.
    static PROFILE_COLORS: std::cell::RefCell<HashMap<u64, std::rc::Rc<Vec<[f32; 3]>>>> =
        std::cell::RefCell::new(HashMap::new());
    /// `(cache, bitmap tag) -> the decoded bitmap's mean colour`, alpha-weighted.
    static BITMAP_MEANS: std::cell::RefCell<HashMap<(u64, u32), Option<[f32; 3]>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Reach's 30-entry ARMOUR COLOUR palette (`matg` +0x498 `profile colors`), in LINEAR RGB,
/// read once per cache.
///
/// The read itself stays in `SceneController::profile_colors` -- there is one implementation of
/// it and this only caches the answer, because the scoreboard is painted every frame and that
/// function walks the tag table looking for `matg` each time it is called.
pub fn profile_colors_for(cache: CacheHandle, read: &dyn Fn() -> Vec<[f32; 3]>) -> std::rc::Rc<Vec<[f32; 3]>> {
    PROFILE_COLORS.with(|m| {
        if let Some(v) = m.borrow().get(&cache.0) {
            if !v.is_empty() {
                return v.clone();
            }
        }
        // An empty answer means the map is still loading, so it is NOT cached: caching it would
        // leave every row grey for the rest of the session.
        let v = std::rc::Rc::new(read());
        if !v.is_empty() {
            m.borrow_mut().insert(cache.0, v.clone());
        }
        v
    })
}

/// LINEAR RGB -> the sRGB value a screen has to be given to SHOW that colour.
///
/// `matg`'s `profile colors` are linear floats (the tag stores them that way, and the 3D body is
/// tinted with them directly in a linear renderer), while a `cui` colour is an sRGB value written
/// straight to the frame. Putting a linear colour into a UI widget without encoding it is what
/// makes a player's armour colour come out far darker on the scoreboard than on their body:
/// Reach's `Steel` is linear 0.127, which is a mid grey (#646464) on screen and near-black
/// (#202020) if the encode is skipped.
pub fn srgb_from_linear(c: [f32; 3]) -> [f32; 3] {
    let f = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    };
    [f(c[0]), f(c[1]), f(c[2])]
}

/// The MEAN colour of one of the game's UI bitmaps, alpha-weighted, decoded once.
///
/// Needed because a `cui` bitmap widget's painted colour is its tint times its ART, and the
/// scoreboard's plates are not white: `player_base_bitmap` is a single pixel of 180,180,180 and
/// `player_base_servicetag_bitmap` is 213,213,213. Deciding anything about what a plate looks
/// like from its tint alone gets the answer wrong -- a 0.694 tint reads as light, but through a
/// 180-grey plate it paints as a mid 0.49.
pub fn bitmap_mean(cache: CacheHandle, tag: u32, decode: &dyn Fn(u32) -> Option<(Vec<u8>, u32, u32)>) -> Option<[f32; 3]> {
    BITMAP_MEANS.with(|m| {
        if let Some(v) = m.borrow().get(&(cache.0, tag)) {
            return *v;
        }
        let made = decode(tag).and_then(|(bgra, w, h)| {
            let n = (w as usize * h as usize).min(bgra.len() / 4);
            if n == 0 {
                return None;
            }
            let mut sum = [0.0f64; 3];
            let mut wsum = 0.0f64;
            for px in bgra.chunks_exact(4).take(n) {
                let a = px[3] as f64 / 255.0;
                sum[0] += px[2] as f64 / 255.0 * a;
                sum[1] += px[1] as f64 / 255.0 * a;
                sum[2] += px[0] as f64 / 255.0 * a;
                wsum += a;
            }
            // Fully transparent art contributes nothing and must not divide by zero.
            if wsum <= 0.0 {
                return Some([0.0, 0.0, 0.0]);
            }
            Some([(sum[0] / wsum) as f32, (sum[1] / wsum) as f32, (sum[2] / wsum) as f32])
        });
        m.borrow_mut().insert((cache.0, tag), made);
        made
    })
}

/// The open cache's UI palette, decoded once. See [`UiGlobals`].
pub fn ui_globals_for(native: &NativeDll, cache: CacheHandle) -> Option<std::rc::Rc<UiGlobals>> {
    UI_GLOBALS.with(|m| {
        m.borrow_mut()
            .entry(cache.0)
            .or_insert_with(|| read_ui_globals(native, cache).map(std::rc::Rc::new))
            .clone()
    })
}

/// The open cache's scoreboard, decoded once.
pub fn scoreboard_for(native: &NativeDll, cache: CacheHandle) -> Option<std::rc::Rc<Scoreboard>> {
    SCOREBOARD.with(|m| {
        m.borrow_mut()
            .entry(cache.0)
            .or_insert_with(|| read_scoreboard(native, cache).map(std::rc::Rc::new))
            .clone()
    })
}

/// One of Reach's UI bitmaps as an egui texture, decoded once.
///
/// `ui\cui\common\white_pix` is exactly what its name says — a single white pixel — so drawing
/// it tinted IS the authored appearance, not a substitute; the rest (`player_base_bitmap`,
/// `player_base_servicetag_bitmap`, `roster_gradient_bitmap`, …) are real art decoded through the
/// same path the 3D scene uses. `None` means the decode failed and the caller must leave the slot
/// empty rather than invent something.
pub fn bitmap_texture(
    ctx: &egui::Context,
    cache: CacheHandle,
    tag: u32,
    decode: &dyn Fn(u32) -> Option<(Vec<u8>, u32, u32)>,
) -> Option<egui::TextureHandle> {
    BITMAPS.with(|m| {
        if let Some(t) = m.borrow().get(&(cache.0, tag)) {
            return t.clone();
        }
        let made = decode(tag).and_then(|(bgra, w, h)| {
            if w == 0 || h == 0 || bgra.len() < (w as usize * h as usize * 4) {
                return None;
            }
            let px: Vec<egui::Color32> = bgra
                .chunks_exact(4)
                .take(w as usize * h as usize)
                .map(|c| egui::Color32::from_rgba_unmultiplied(c[2], c[1], c[0], c[3]))
                .collect();
            let img = egui::ColorImage { size: [w as usize, h as usize], pixels: px };
            Some(ctx.load_texture(format!("reach-ui-{tag}"), img, egui::TextureOptions::LINEAR))
        });
        m.borrow_mut().insert((cache.0, tag), made.clone());
        made
    })
}


// ---------------------------------------------------------------------------------------------
// print-only diagnostic
// ---------------------------------------------------------------------------------------------

/// [tag] Reach's EMBLEM LIBRARY: `mlib` (`emblem_library`, 116 bytes), reached from
/// `rasg` (`rasterizer_globals`) +0x88 `Emblem Library`.
///
/// ```text
/// mlib +0x004 real           bitmap resolution [pixels]
///      +0x008 real           antialias sharpen
///      +0x00C tag reference  emblem bitmaps        (bitm, 16 B)
///      +0x01C tag reference  emblem bitmaps hi rez (bitm, 16 B)
///      +0x02C block          bitmaps        [256]  12 B
///      +0x038 block          shapes                48 B
///      +0x044 block          front emblems        184 B
///      +0x050 block          back emblems          92 B
///      +0x05C block          runtime front        220 B
///      +0x068 block          runtime back         108 B
/// ```
///
/// The tag POSTPROCESS compiles `front emblems`/`back emblems` into `runtime front`/`runtime
/// back` and then resizes the four authored blocks to zero, so in a BUILT cache only the two
/// runtime blocks and the two bitmap references survive.
#[derive(Clone, Debug, Default)]
pub struct EmblemLibrary {
    pub bitmaps_tag: Option<u32>,
    pub bitmaps_hi_tag: Option<u32>,
    pub resolution: f32,
    pub front_count: usize,
    pub back_count: usize,
    pub front_raw: Vec<u8>,
    pub back_raw: Vec<u8>,
}

/// Read the open cache's emblem library, through `rasg`.
pub fn read_emblem_library(native: &NativeDll, cache: CacheHandle) -> Option<EmblemLibrary> {
    let rasg = native.tags_by_group(cache, "rasg").first().copied()?;
    let rmeta = native.tag_meta(cache, rasg, 0, 0x200)?;
    let datum = rd_u32(&rmeta, 0x88 + 0x0C)?;
    let mlib = if datum == u32::MAX { return None } else { datum & 0xFFFF };
    let meta = native.tag_meta(cache, mlib, 0, 116)?;
    let tref = |o: usize| -> Option<u32> {
        let d = rd_u32(&meta, o + 0x0C)?;
        (d != u32::MAX).then_some(d & 0xFFFF)
    };
    let fn_ = rd_i32(&meta, 0x5C).unwrap_or(0).max(0) as usize;
    let bn = rd_i32(&meta, 0x68).unwrap_or(0).max(0) as usize;
    Some(EmblemLibrary {
        bitmaps_tag: tref(0x0C),
        bitmaps_hi_tag: tref(0x1C),
        resolution: rd_f32(&meta, 0x04).unwrap_or(0.0),
        front_count: fn_,
        back_count: bn,
        front_raw: block(native, cache, &meta, 0x5C, 220, 512),
        back_raw: block(native, cache, &meta, 0x68, 108, 512),
    })
}

/// Print the emblem library: the bitmap tags it points at and the raw runtime blocks, so the
/// baked per-emblem bitmap indices can be read off rather than guessed.
pub fn dump_emblem_library(native: &NativeDll, cache: CacheHandle) {
    let Some(l) = read_emblem_library(native, cache) else {
        println!("mlib: not reachable from rasg +0x88 in this cache");
        return;
    };
    println!("\n================ mlib (emblem_library)");
    println!("  bitmap resolution {} px", l.resolution);
    for (what, t) in [("emblem bitmaps", l.bitmaps_tag), ("emblem bitmaps hi rez", l.bitmaps_hi_tag)] {
        match t {
            Some(t) => println!("  {what}: bitm {t:#06x} '{}'", native.tag_name(cache, t).unwrap_or_default()),
            None => println!("  {what}: NONE"),
        }
    }
    println!("  runtime front: {} entries ({} bytes read)", l.front_count, l.front_raw.len());
    for (i, e) in l.front_raw.chunks_exact(220).enumerate().take(6) {
        println!("    front[{i}] primary layer byte +0xD8 = {}", e[0xD8]);
        for (lay, off) in [("layer0", 0usize), ("layer1", 0x6C)] {
            let w = &e[off..off + 0x6C];
            println!(
                "      {lay} i16s: {}",
                w.chunks_exact(2).take(12).map(|c| format!("{}", i16::from_le_bytes([c[0], c[1]]))).collect::<Vec<_>>().join(" ")
            );
            println!("      {lay} raw : {}", w[..32].iter().map(|v| format!("{v:02x}")).collect::<Vec<_>>().join(" "));
        }
    }
    println!("  runtime back: {} entries ({} bytes read)", l.back_count, l.back_raw.len());
    for (i, e) in l.back_raw.chunks_exact(108).enumerate().take(6) {
        println!(
            "    back[{i}] i16s: {}",
            e.chunks_exact(2).take(12).map(|c| format!("{}", i16::from_le_bytes([c[0], c[1]]))).collect::<Vec<_>>().join(" ")
        );
        println!("    back[{i}] raw : {}", e[..32].iter().map(|v| format!("{v:02x}")).collect::<Vec<_>>().join(" "));
    }
    // The i16 at each stride position, across every entry: a BAKED BITMAP INDEX is a small
    // non-negative number that varies entry to entry and never exceeds the bitmap's image count.
    for (what, raw, stride) in [("front", &l.front_raw, 220usize), ("back", &l.back_raw, 108)] {
        let n = raw.len() / stride;
        if n == 0 {
            continue;
        }
        println!("  {what}: per-i16-slot range over all {n} entries (slot, min, max, distinct)");
        for sl in 0..stride / 2 {
            let mut v: Vec<i16> = (0..n).map(|k| i16::from_le_bytes([raw[k * stride + sl * 2], raw[k * stride + sl * 2 + 1]])).collect();
            let (lo, hi) = (*v.iter().min().unwrap(), *v.iter().max().unwrap());
            v.sort_unstable();
            v.dedup();
            if v.len() > 1 && lo >= 0 && hi < 4096 {
                println!("    +0x{:03X} i16: {lo}..{hi}, {} distinct", sl * 2, v.len());
            }
        }
    }
}

/// Print the `wigl` palette: the default text colour, every colour preset and the four UI tint
/// lists, with their authored floats and the 8-bit colours they become.
pub fn dump_ui_globals(native: &NativeDll, cache: CacheHandle) {
    let Some(g) = read_ui_globals(native, cache) else {
        println!("wigl: NOT IN THIS CACHE");
        return;
    };
    let f = |c: [f32; 4]| {
        format!(
            "a {:.5} r {:.5} g {:.5} b {:.5}  -> rgb8 {:>3} {:>3} {:>3} @ a {:>3}",
            c[0],
            c[1],
            c[2],
            c[3],
            (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (c[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        )
    };
    println!("
================ wigl (user_interface_shared_globals)");
    println!("  +0x74 text color    {}", f(g.text_color));
    println!("  +0x84 shadow color  {}", f(g.shadow_color));
    println!("  +0x94 color presets ({}):", g.presets.len());
    for (i, (n, c)) in g.presets.iter().enumerate() {
        println!("    [{i:2}] {n:<14} {}", f(*c));
    }
    for (what, l) in [
        ("text player", &g.tint_text_player),
        ("text team", &g.tint_text_team),
        ("bitmap player", &g.tint_bitmap_player),
        ("bitmap team", &g.tint_bitmap_team),
    ] {
        println!("  +0xA0 tint {what} ({}):", l.len());
        for (i, c) in l.iter().enumerate() {
            println!("    [{i:2}] {}", f(*c));
        }
    }
    println!("  emblem counts: primary {} secondary {}", g.primary_emblem_count, g.secondary_emblem_count);
}

/// Print every `cui_static_data` (`cust`) tag whose name matches, with its declarations and the
/// argb property table that backs them — which is where `prop_color_preset` names resolve.
pub fn dump_static_data(native: &NativeDll, cache: CacheHandle, filter: &str) {
    let sid = |s: u32| -> String {
        if s == 0 || s == u32::MAX { String::new() } else { native.string_id(cache, s).unwrap_or_default() }
    };
    for tag in native.tags_by_group(cache, "cust") {
        let name = native.tag_name(cache, tag).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        println!("\n================ cust {tag:#06x} '{name}'");
        let Some(meta) = native.tag_meta(cache, tag, 0, 256) else { continue };
        for (i, row) in meta.chunks(16).take(8).enumerate() {
            println!("  meta {:04x}  {}", i * 16, row.iter().map(|v| format!("{v:02x}")).collect::<Vec<_>>().join(" "));
        }
        // +0x00: the DECLARATIONS — {string id name, i32 property type}. Type 7 is an argb color
        // (the same code `prop_property_type` carries on a decision_maker whose then/else are argb).
        let decls = block(native, cache, &meta, 0x00, 8, 4096);
        println!("  declarations ({}):", decls.len() / 8);
        for (i, e) in decls.chunks_exact(8).enumerate() {
            println!("    [{i:3}] {:<28} type {}", sid(rd_u32(e, 0).unwrap_or(0)), rd_i32(e, 4).unwrap_or(-1));
        }
        // +0x0C: one properties_struct instance, whose six parallel blocks hold the values.
        let inst = block(native, cache, &meta, 0x0C, 76, 64);
        println!("  instances ({}):", inst.len() / 76);
        for (i, e) in inst.chunks_exact(76).enumerate() {
            println!("    [{i:3}] {}", sid(rd_u32(e, 0).unwrap_or(0)));
            let p = &e[4..];
            for v in block(native, cache, p, 0x3C, 20, 4096).chunks_exact(20) {
                println!(
                    "          argb {:<28} = a {:.4} r {:.4} g {:.4} b {:.4}   -> rgb8 {:>3} {:>3} {:>3}",
                    sid(rd_u32(v, 0).unwrap_or(0)),
                    rd_f32(v, 4).unwrap_or(0.0),
                    rd_f32(v, 8).unwrap_or(0.0),
                    rd_f32(v, 12).unwrap_or(0.0),
                    rd_f32(v, 16).unwrap_or(0.0),
                    (rd_f32(v, 8).unwrap_or(0.0).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    (rd_f32(v, 12).unwrap_or(0.0).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    (rd_f32(v, 16).unwrap_or(0.0).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                );
            }
            for v in block(native, cache, p, 0x18, 8, 4096).chunks_exact(8) {
                println!("          sid  {:<28} = {}", sid(rd_u32(v, 0).unwrap_or(0)), sid(rd_u32(v, 4).unwrap_or(0)));
            }
            for v in block(native, cache, p, 0x0C, 8, 4096).chunks_exact(8) {
                println!("          real {:<28} = {}", sid(rd_u32(v, 0).unwrap_or(0)), rd_f32(v, 4).unwrap_or(0.0));
            }
        }
    }
}

/// `hms-app --dump-scoreboard <map.map> [tag-substring]` — print EVERYTHING the scoreboard
/// `cusc` tags author, in tag order: every component with its kind/name/parent, every property
/// of every kind with its authored value, the referenced `bitm` tag names, the overlay list and
/// the raw `component indices` block. Print-only; it changes nothing.
pub fn dump_scoreboard(native: &NativeDll, cache: CacheHandle, filter: &str) {
    let all = native.tags_by_group(cache, "cusc");
    println!("cusc tags in cache: {}", all.len());
    let wanted: Vec<u32> = if filter.is_empty() {
        [SCOREBOARD_SCREEN, SCOREBOARD_ROW_PLAYER, SCOREBOARD_ROW_TEAM]
            .iter()
            .filter_map(|p| {
                all.iter().copied().find(|&t| native.tag_name(cache, t).map_or(false, |n| n.eq_ignore_ascii_case(p)))
            })
            .collect()
    } else {
        all.iter()
            .copied()
            .filter(|&t| {
                native.tag_name(cache, t).map_or(false, |n| n.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()))
            })
            .collect()
    };
    for tag in wanted {
        dump_one(native, cache, tag);
    }
}

fn dump_one(native: &NativeDll, cache: CacheHandle, tag: u32) {
    let sid = |s: u32| -> String {
        if s == 0 || s == u32::MAX { String::new() } else { native.string_id(cache, s).unwrap_or_default() }
    };
    let name = native.tag_name(cache, tag).unwrap_or_default();
    println!("\n================ cusc {tag:#06x} '{name}'");
    let Some(meta) = native.tag_meta(cache, tag, 0, 124) else {
        println!("  (meta unreadable)");
        return;
    };
    // the raw block headers, so a count/pointer that looks wrong is visible
    for (off, what, elem) in
        [(0x10usize, "template instantiations", 16usize), (0x1C, "components", 16), (0x28, "component indices", 8), (0x34, "overlays", 32), (0x4C, "property bindings", 20)]
    {
        println!(
            "  block +{off:#04x} {what:<24} count {:>4} ptr {:#010x} (elem {elem})",
            rd_i32(&meta, off).unwrap_or(-1),
            rd_u32(&meta, off + 4).unwrap_or(0)
        );
    }
    // components, in tag order
    let comps = block(native, cache, &meta, 0x1C, 16, 8192);
    println!("  components ({}), IN TAG ORDER:", comps.len() / 16);
    let mut names: Vec<String> = Vec::new();
    for (i, e) in comps.chunks_exact(16).enumerate() {
        let (k, n, par) = (sid(rd_u32(e, 0).unwrap_or(0)), sid(rd_u32(e, 4).unwrap_or(0)), sid(rd_u32(e, 8).unwrap_or(0)));
        let flags = u16::from_le_bytes([e[12], e[13]]);
        let inst = i16::from_le_bytes([e[14], e[15]]);
        println!("    [{i:3}] {k:<22} {n:<42} parent={par:<34} flags={flags:#06x} inst={inst}");
        names.push(n);
    }
    // component indices: 8 bytes each, printed raw AND as two i32 so a draw order would show
    let ci = block(native, cache, &meta, 0x28, 8, 8192);
    println!("  component indices ({}):", ci.len() / 8);
    for (i, e) in ci.chunks_exact(8).enumerate() {
        let a = rd_i32(e, 0).unwrap_or(0);
        let b = rd_i32(e, 4).unwrap_or(0);
        let who = names.get(a as usize).cloned().unwrap_or_default();
        println!("    [{i:3}] {a:>6} {b:>6}   raw {} -> comp[{a}] = {who}", e.iter().map(|v| format!("{v:02x}")).collect::<Vec<_>>().join(" "));
    }
    // property bindings
    let pb = block(native, cache, &meta, 0x4C, 20, 8192);
    println!("  property bindings ({}):", pb.len() / 20);
    for (i, e) in pb.chunks_exact(20).enumerate() {
        println!(
            "    [{i:3}] {}",
            (0..5).map(|k| format!("{}={}", k, sid(rd_u32(e, k * 4).unwrap_or(0)))).collect::<Vec<_>>().join(" ")
        );
    }
    // overlays
    let overlays = block(native, cache, &meta, 0x34, 32, 24);
    println!("  overlays ({}):", overlays.len() / 32);
    for ov in overlays.chunks_exact(32) {
        println!(
            "    {:<26} {:<16} components {:>4} animations {:>4}",
            sid(rd_u32(ov, 0).unwrap_or(0)),
            sid(rd_u32(ov, 4).unwrap_or(0)),
            rd_i32(ov, 0x08).unwrap_or(-1),
            rd_i32(ov, 0x14).unwrap_or(-1)
        );
    }
    // the chosen overlay's properties, per component, in the order the overlay lists them
    for ov in overlays.chunks_exact(32) {
        let (r, t) = (sid(rd_u32(ov, 0).unwrap_or(0)), sid(rd_u32(ov, 4).unwrap_or(0)));
        if r != OVERLAY_RESOLUTION || t != OVERLAY_THEME {
            continue;
        }
        println!("  --- overlay {r} / {t} properties, in overlay order:");
        let props = block(native, cache, &ov[..], 0x08, 76, 8192);
        for (i, e) in props.chunks_exact(76).enumerate() {
            let cname = sid(rd_u32(e, 0).unwrap_or(0));
            let p = &e[4..];
            println!("    [{i:3}] {cname}");
            for v in block(native, cache, p, 0x00, 8, 8192).chunks_exact(8) {
                println!("          long   {:<34} = {}", sid(rd_u32(v, 0).unwrap_or(0)), rd_i32(v, 4).unwrap_or(0));
            }
            for v in block(native, cache, p, 0x0C, 8, 8192).chunks_exact(8) {
                println!("          real   {:<34} = {}", sid(rd_u32(v, 0).unwrap_or(0)), rd_f32(v, 4).unwrap_or(0.0));
            }
            for v in block(native, cache, p, 0x18, 8, 8192).chunks_exact(8) {
                println!("          sid    {:<34} = {}", sid(rd_u32(v, 0).unwrap_or(0)), sid(rd_u32(v, 4).unwrap_or(0)));
            }
            for v in block(native, cache, p, 0x24, 20, 8192).chunks_exact(20) {
                let group: String = v[4..8].iter().rev().map(|&b| b as char).collect();
                let datum = rd_u32(v, 4 + 0x0C).unwrap_or(u32::MAX);
                let idx = (datum != u32::MAX).then(|| datum & 0xFFFF);
                println!(
                    "          tagref {:<34} = {group} {:?} '{}'",
                    sid(rd_u32(v, 0).unwrap_or(0)),
                    idx,
                    idx.and_then(|t| native.tag_name(cache, t)).unwrap_or_default()
                );
            }
            for v in block(native, cache, p, 0x30, 260, 8192).chunks_exact(260) {
                let end = v[4..].iter().position(|&c| c == 0).unwrap_or(256);
                println!(
                    "          text   {:<34} = {:?}",
                    sid(rd_u32(v, 0).unwrap_or(0)),
                    String::from_utf8_lossy(&v[4..4 + end])
                );
            }
            for v in block(native, cache, p, 0x3C, 20, 8192).chunks_exact(20) {
                println!(
                    "          argb   {:<34} = a {:.4} r {:.4} g {:.4} b {:.4}",
                    sid(rd_u32(v, 0).unwrap_or(0)),
                    rd_f32(v, 4).unwrap_or(0.0),
                    rd_f32(v, 8).unwrap_or(0.0),
                    rd_f32(v, 12).unwrap_or(0.0),
                    rd_f32(v, 16).unwrap_or(0.0)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Open any shipped Reach cache, or `None` when the game is not installed here.
    fn open_any_reach() -> Option<(NativeDll, CacheHandle)> {
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
        let native = NativeDll::load(&dll).ok()?;
        let map = crate::mapcat::enumerate().into_iter().find(|c| c.game == crate::mapcat::Game::Reach)?;
        native.set_shared_cache_dirs(&[map.path.parent()?.to_path_buf()]);
        let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
        Some((native, cache))
    }

    /// The real scoreboard tags of a shipped Reach cache.
    ///
    /// IGNORED BY DEFAULT, and not because it is flaky: it has to LOAD THE NATIVE PARSER AND OPEN
    /// A CACHE, and that parser keeps process-global state — doing it while `reach_actor`'s own
    /// cache tests are running in another thread makes their tag lookups come back empty. Those
    /// tests serialise against each other with a lock private to their module, which this one
    /// cannot take. Run it on its own:
    ///
    /// ```text
    /// cargo test -p hms-app reach_widget -- --ignored
    /// ```
    #[test]
    #[ignore = "opens a second cache; the native parser's global state collides with reach_actor's cache tests"]
    fn shipped_scoreboard_decodes() {
        let Some((native, cache)) = open_any_reach() else {
            eprintln!("no Reach cache on this machine - skipped");
            return;
        };
        let sb = read_scoreboard(&native, cache).expect("scoreboard tags");
        // The screen declares the virtual space every co-ordinate is in.
        assert_eq!(sb.player_row.virtual_size, [1152.0, 720.0]);
        // One row is 350 x 30 and its five columns sit where the tag says.
        assert_eq!((sb.row_width(), sb.row_height()), (350.0, 30.0));
        let name = sb.player_row.get("text_player_name").expect("name column");
        assert_eq!((name.left(), name.top(), name.width(), name.height()), (58.0, 4.0, 187.0, 22.0));
        assert_eq!(name.font_id(), Some(crate::reach_font::FontId::EnglishBodyText));
        let tag_col = sb.player_row.get("text_player_designation").expect("service tag column");
        assert_eq!((tag_col.left(), tag_col.width()), (246.0, 54.0));
        assert_eq!(tag_col.text_id(), Some("service_tag"));
        assert_eq!(tag_col.justification(), 2);
        let score = sb.player_row.get("text_player_score").expect("score column");
        assert_eq!((score.left(), score.width()), (300.0, 50.0));
        // Every bitmap column resolves to a real bitm tag.
        for n in ["roster_position_bitmap", "roster_bitmap", "roster_color_end_bitmap", "roster_score_bitmap"] {
            assert!(sb.player_row.get(n).and_then(|c| c.bitmap()).is_some(), "{n} has no bitmap");
        }
        // The dark plate behind the rank number is white_pix tinted nearly black.
        let pos = sb.player_row.get("roster_position_bitmap").unwrap();
        let t = pos.tint().expect("tint");
        assert!(t[1] < 0.2 && t[2] < 0.2 && t[3] < 0.2, "rank plate tint {t:?}");
        // The header row lives on the screen, not the template.
        let hdr = sb.screen.get("text_header_players").expect("players header");
        assert_eq!(hdr.text_id(), Some("players"));
        assert_eq!(sb.screen.get("text_header_score").and_then(|c| c.text_id()), Some("score"));

        // #reach-widget  THE PROPERTY BINDINGS, which used to be read past and thrown away --
        // and which are the only thing that says what colour the player name's plate is.
        assert_eq!(sb.player_row.bindings.len(), 4, "the row template authors four bindings");
        assert_eq!(
            sb.player_row.bound_from("roster_bitmap", "prop_tint_color"),
            Some(("bitmap_color_decision_maker", "prop_result")),
            "the wide name plate takes its colour from the decision maker"
        );
        assert_eq!(
            sb.player_row.bound_from("bitmap_color_decision_maker", "prop_then"),
            Some(("roster_color_start_bitmap", "prop_tint_color")),
            "and the decision maker's `then` is the player's own bar colour"
        );
        // The screen end of the same chain: where that bar colour comes from, and what decides
        // which branch of the decision maker a row takes.
        assert_eq!(
            sb.screen.bound_from("roster_color_start_bitmap", "prop_tint_color"),
            Some(("data_player_scoreboard", "prop_player_bitmap_tint_color"))
        );
        assert_eq!(
            sb.screen.bound_from("bitmap_color_decision_maker", "prop_if"),
            Some(("data_player_scoreboard", "prop_player_left_game"))
        );
        // The row template authors NO tint on the two bars the bindings feed. That is why a
        // white default was being painted, and why this has to stay true for the fix to matter.
        for n in ["roster_bitmap", "roster_color_start_bitmap", "roster_color_end_bitmap"] {
            assert!(sb.player_row.get(n).unwrap().tint().is_none(), "{n} authors a tint after all");
        }
        // The emblem slot the row really does have, so a reader can see it is not invented.
        let em = sb.player_row.get("bitmap_player_scoreboard_emblem").expect("emblem component");
        assert_eq!(em.kind, "emblem_widget");
        assert_eq!((em.left(), em.top(), em.width(), em.height()), (27.0, 2.0, 24.0, 24.0));
        assert_eq!(
            sb.screen.bound_from("bitmap_data_player_scoreboard_emblem", "prop_emblem_info"),
            Some(("data_player_scoreboard", "prop_emblem_info")),
            "the engine feeds the emblem slot from the recorded player data"
        );
    }

    /// Reach's OWN UI PALETTE, out of a shipped cache: the text colour a widget with no preset
    /// draws in, the twelve colour presets by name, and the four tint lists.
    ///
    /// This is what replaced two RGB triples that had been "mapped by eye", and the numbers are
    /// pinned here so a regression to invented colours cannot pass. See the module docs on
    /// [`UiGlobals`] for where each field lives.
    #[test]
    #[ignore = "opens a second cache; the native parser's global state collides with reach_actor's cache tests"]
    fn shipped_ui_globals_palette() {
        let Some((native, cache)) = open_any_reach() else {
            eprintln!("no Reach cache on this machine - skipped");
            return;
        };
        let g = read_ui_globals(&native, cache).expect("wigl");
        let near = |a: f32, b: f32| (a - b).abs() < 2.0e-4;
        // +0x74 `text color`: this, not white, is what `text_player_name` draws in.
        assert!(near(g.text_color[0], 1.0) && near(g.text_color[1], 0.92157), "text color {:?}", g.text_color);
        // +0x84 `shadow color`: half-alpha black.
        assert!(near(g.shadow_color[0], 0.5) && near(g.shadow_color[1], 0.0), "shadow {:?}", g.shadow_color);
        // The twelve presets, by name. `white` is 0.847 and `body` is DARK.
        assert_eq!(g.presets.len(), 12);
        let w = g.preset("white").expect("white preset");
        assert!(near(w[1], 0.84706), "white {w:?}");
        let b = g.preset("body").expect("body preset");
        assert!(near(b[1], 0.35), "body {b:?}");
        assert!(b[1] < 0.5, "body is the dark preset, not a light one");
        assert!(g.preset("transparent").map_or(false, |c| c[0] == 0.0));
        for n in ["hilite", "unhilite", "dim", "semi_dim", "orange", "red", "breadcrumb", "button_key", "ice"] {
            assert!(g.preset(n).is_some(), "preset {n} missing");
        }
        // The tint lists. `bitmap team` is the one the scoreboard's bars take, it has one entry
        // per Reach team, and every one of them is DARK -- this is a UI palette, not `mulg`'s
        // in-world team colours (team red here is 0.53/0.08/0.08, not 1/0/0).
        assert_eq!(g.tint_bitmap_team.len(), 8, "bitmap team");
        assert_eq!(g.tint_bitmap_player.len(), 30, "bitmap player is one entry per armour colour");
        assert_eq!(g.tint_text_player.len(), 30);
        assert_eq!(g.tint_text_team.len(), 8);
        let red = g.bitmap_team(0).expect("team 0");
        assert!(near(red[1], 0.53218) && near(red[2], 0.08139), "team red {red:?}");
        // These are UI BARS, not the in-world team colours: six of the eight are dark (red
        // 0.177, blue 0.196, green 0.219, purple 0.201, brown 0.210, orange 0.319 in relative
        // luminance), and the two bright ones -- pink 0.552 and GOLD 0.725 -- are the worst case
        // a name has to stay readable on. `mulg`'s team red, by contrast, is 1/0/0.
        let lum: Vec<f32> =
            g.tint_bitmap_team.iter().map(|c| 0.2126 * c[1] + 0.7152 * c[2] + 0.0722 * c[3]).collect();
        assert!(lum.iter().filter(|l| **l < 0.35).count() >= 6, "bitmap team is not a dark palette: {lum:?}");
        assert!(lum.iter().all(|l| *l < 0.75), "a team bar is brighter than the palette allows: {lum:?}");
        // And the emblem counts, which say how wide a recorded emblem index may be.
        assert_eq!((g.primary_emblem_count, g.secondary_emblem_count), (80, 53));
    }

    /// The emblem art Reach ships, and the library that indexes it. Read so the claim "HMS does
    /// not composite this" is a statement about a known shape rather than a shrug.
    #[test]
    #[ignore = "opens a second cache; the native parser's global state collides with reach_actor's cache tests"]
    fn shipped_emblem_library() {
        let Some((native, cache)) = open_any_reach() else {
            eprintln!("no Reach cache on this machine - skipped");
            return;
        };
        let l = read_emblem_library(&native, cache).expect("mlib via rasg +0x88");
        assert_eq!((l.front_count, l.back_count), (80, 53), "runtime front / back");
        assert_eq!(l.resolution, 64.0, "each emblem shape is a 64 px vector map");
        let t = l.bitmaps_tag.expect("emblem bitmaps");
        assert_eq!(
            native.tag_name(cache, t).unwrap_or_default().to_ascii_lowercase(),
            "ui\\cui\\common\\emblems\\emblems"
        );
        // The authored blocks a tool would want are compiled away at build time, which is the
        // whole reason the composition is not reproduced here: the runtime blocks carry baked
        // bitmap indices and transforms, and no names.
        assert_eq!(l.front_raw.len(), 80 * 220);
        assert_eq!(l.back_raw.len(), 53 * 108);
    }
}
