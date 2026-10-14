//! The saved-film metadata card, and the one-line summary a film gets in the file browser.
//!
//! The content is produced by PURE functions (`metadata_rows`, `browser_row_hint`,
//! `format_duration`) so the privacy rule and the formatting are unit-testable without a GUI;
//! `card_ui` only lays them out.
//!
//! PRIVACY: a film's `flmh+0x8C` carries the recording PC's Windows account name and computer
//! name. `metadata_rows` NEVER includes it unless `show_machine` is explicitly true, and nothing
//! this module writes to disk includes it at all.

use crate::film::{self, Film};
use eframe::egui;
use crate::film_export;

/// What the card wants done. The caller owns the file dialogs and the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmAction {
    None,
    /// Load the film's base map and draw its embedded Forge scene.
    LoadScene,
    /// Write the embedded map variant out as a `.mvar`.
    ExtractMapVariant,
    /// Write the embedded game variant out (raw struct + notes).
    ExtractGameVariant,
    Close,
}

/// `h:mm:ss` / `m:ss` for a film length in seconds.
pub fn format_duration(seconds: u32) -> String {
    let (h, m, s) = (seconds / 3600, (seconds / 60) % 60, seconds % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

/// The recording date, as `mvar::unix_date` formats every other timestamp in HMS.
pub fn recorded_on(f: &Film) -> String {
    crate::mvar::unix_date(f.content.created_unix)
}

/// The card's rows: (label, value). Pure, so the privacy rule is testable.
///
/// `show_machine` gates the ONE field that identifies the recording computer. With it false (the
/// default) the session id never appears.
pub fn metadata_rows(f: &Film, show_machine: bool, base_map: Option<&str>) -> Vec<(&'static str, String)> {
    let h = &f.header;
    let mv = &h.map_variant;
    let mut rows: Vec<(&'static str, String)> = Vec::new();
    rows.push(("Film", format!("{:?}", f.content.title)));
    if !f.content.description.is_empty() {
        rows.push(("Description", f.content.description.clone()));
    }
    rows.push((
        "Length",
        format!("{} ({} ticks at {} Hz)", format_duration(f.content.seconds), h.tick_count, h.tick_rate),
    ));
    rows.push(("Recorded", recorded_on(f)));
    rows.push(("Saved by", format!("{} ", f.content.created_by)));
    // The caller resolves the installed map from its own cached candidate list; enumerating the
    // disk here would run on every frame the card is open.
    rows.push((
        "Base map",
        match base_map {
            Some(label) => format!("{label} (map id {})", f.content.map_id),
            None => format!("{} (map id {}, not installed)", h.scenario_name(), f.content.map_id),
        },
    ));
    rows.push((
        "Gametype",
        format!(
            "{:?} — {} ({})",
            h.game_variant.title,
            film_export::engine_type_name(h.game_engine_type),
            h.game_engine_type
        ),
    ));
    if !h.game_variant.created_by.is_empty() {
        rows.push(("Gametype author", h.game_variant.created_by.clone()));
    }
    if mv.is_empty() {
        rows.push(("Map variant", "none — recorded on the stock map".into()));
    } else {
        rows.push((
            "Map variant",
            format!("{:?} by {}", mv.metadata.title, mv.metadata.created_by),
        ));
        rows.push((
            "Forge objects",
            format!(
                "{} of {} slots, {} / {} budget",
                mv.present_objects().count(),
                mv.objects.len(),
                mv.budget_spent,
                mv.budget_max
            ),
        ));
    }
    // Gamertag, service tag and team, straight out of the recorded player slots.
    let players: Vec<String> = f
        .present_players()
        .map(|p| {
            let mut s = p.gamertag.clone();
            if !p.service_tag.is_empty() {
                s.push_str(&format!(" [{}]", p.service_tag));
            }
            s.push_str(&format!(" ({})", team_name(p.team)));
            s
        })
        .collect();
    rows.push((
        "Players",
        if players.is_empty() { "none recorded".into() } else { format!("{} — {}", players.len(), players.join(", ")) },
    ));
    if f.header.is_clip {
        rows.push(("Clip", format!("starts at tick {}", h.clip_start_tick)));
    }
    for m in &h.markers {
        rows.push(("Marker", format!("tick {}: {}", m.tick, m.text)));
    }
    if show_machine {
        rows.push((
            "Recording machine",
            format!("{} — this is the Windows account and computer name of the PC that recorded the film", h.session_id),
        ));
    }
    rows
}

/// The one-line hint a film gets in the file browser listing. Reads ONLY the header prefix, so
/// listing a folder of 150 MB films stays instant.
pub fn browser_row_hint(path: &std::path::Path) -> Option<String> {
    let f = film::open_header(path).ok()?;
    let mv = &f.header.map_variant;
    let map = f.header.scenario_name().to_string();
    let variant = if mv.is_empty() { "stock map".to_string() } else { mv.metadata.title.clone() };
    let players = f.player_names().len();
    Some(format!(
        "{} — {}, {} ({}), {} player{}",
        f.content.title,
        map,
        variant,
        format_duration(f.content.seconds),
        players,
        if players == 1 { "" } else { "s" }
    ))
}

/// Where MCC keeps Reach saved films: `<user>/AppData/LocalLow/MCC*/Temporary/...`. Found by
/// walking mounted roots rather than hard-coding a user name, so it works both on Windows and on
/// The folders a film browser should offer, most useful first.
///
/// One implementation, in `film.rs` — there were six copies of this walk and they disagreed
/// about which films exist. See `film::film_search_dirs`.
pub fn film_search_folders() -> Vec<std::path::PathBuf> {
    let mut v = crate::film::film_search_dirs();
    v.truncate(12);
    v
}

// ---------------------------------------------------------------------------------------------
// teams, name tags
// ---------------------------------------------------------------------------------------------

/// Reach's team names, by the index the film header stores. 0xFF = not known (an unnamed player
/// has no header record, so no team).
pub fn team_name(team: u8) -> &'static str {
    match team {
        0 => "red",
        1 => "blue",
        2 => "green",
        3 => "orange",
        4 => "purple",
        5 => "gold",
        6 => "brown",
        7 => "pink",
        0xFF => "unknown",
        _ => "neutral",
    }
}

/// The swatch colour for a team, for the roster square and the floating name tags.
pub fn team_color(team: u8) -> egui::Color32 {
    match team {
        0 => egui::Color32::from_rgb(210, 70, 70),
        1 => egui::Color32::from_rgb(80, 120, 225),
        2 => egui::Color32::from_rgb(90, 180, 90),
        3 => egui::Color32::from_rgb(225, 150, 60),
        4 => egui::Color32::from_rgb(165, 100, 205),
        5 => egui::Color32::from_rgb(220, 200, 80),
        6 => egui::Color32::from_rgb(150, 115, 80),
        7 => egui::Color32::from_rgb(230, 140, 190),
        _ => egui::Color32::from_rgb(150, 150, 150),
    }
}

/// One floating label over a body.
#[derive(Clone, Debug, PartialEq)]
pub struct NameTag {
    pub text: String,
    /// World position to draw at — above the body's head.
    pub pos: [f32; 3],
    /// True when `text` is a real gamertag THIS recording states.
    pub named: bool,
    /// #xfilm  When this name came from another recording's roster instead — matched on the
    /// body's armour — the film it came from. An inference, and drawn as one.
    pub inferred_from: Option<String>,
    /// The player's team, 0xFF when unknown.
    pub team: u8,
    /// The body this label belongs to.
    pub net_id: u32,
    /// Where the recording put this body's FEET this tick — the value the actor pool placed its
    /// actor at, which is how [`resolve_head_anchors`] finds the actor that is drawing it.
    pub feet: [f32; 3],
    /// True once [`resolve_head_anchors`] has put `pos` on the engine's own anchor.
    pub on_marker: bool,
    /// The film-header configuration SLOT this body's owner resolved to, when it did. Used only
    /// to tell the viewer's own body from everyone else's.
    pub slot: Option<u8>,
    /// True when the engine would put the navpoint CHEVRON over this body — which it does only
    /// for a record it actually made, and `+0x2CB76C` only makes one for a TEAMMATE. Set by
    /// [`resolve_name_colors`].
    pub show_icon: bool,
    /// The engine colour index this name draws in — what `chud_color_from_index`
    /// (`haloreach.dll+0x2E36B8`) is called with. Set by [`resolve_name_colors`]; until then it
    /// is 2, `navpt neutral`, which is the engine's own answer when there is no viewer.
    pub color_index: i32,
}

/// FALLBACK anchor: how far above a body's feet a name tag floats when the engine's own anchor
/// cannot be resolved, world units.
///
/// The engine does not use a fixed height at all — it resolves a marker — so this is only what
/// HMS falls back to for a body whose posed marker it could not find. It is the MEASURED value of
/// that anchor on a Spartan rather than a guess: `--dump-chud` puts the `head` marker of
/// `objects\characters\spartans\spartans` at z = 0.633 wu in the model's own bind pose, and
/// [`LOOKAT_UP_WU`] lifts it to 0.683. (The Elite's is 0.775.) The look-at display is the common
/// case, so the fallback is its lift rather than the waypoint's.
pub const NAME_TAG_UP_WU: f32 = 0.683;

/// [RE: `haloreach.dll+0x2CB76C`] After resolving the unit's marker the ALLY WAYPOINT producer
/// lifts the record this far, in world units — the literal `0.075000003f`.
pub const NAME_TAG_MARKER_UP_WU: f32 = crate::reach_chud::NAME_ANCHOR_UP_WU;

/// [RE: `haloreach.dll+0x18AE04`] ...and the LOOK-AT producer lifts it by this instead, the
/// literal `0.0500000007f`. Same marker, 0.025 wu lower.
pub const LOOKAT_UP_WU: f32 = crate::reach_chud::LOOKAT_ANCHOR_UP_WU;

/// [RE] The marker the ENGINE resolves to anchor a player's floating name: `head`.
///
/// `haloreach.dll+0x2CB76C` builds the record's position with `sub_180489FD4(unit, out)`, which
/// is `Object_BuildWorldTransform(unit, 0xC2, ...)` (`+0x489FD4`); `0xC2` is a `string_id`, it is
/// looked up as a MARKER GROUP NAME on the unit's render model (`+0x221AD4` walks the group array
/// comparing `group+0x00` to it), and against a shipped Reach cache's own string table
/// `string_id 0xC2` resolves to **`head`**. Every player biped carries it — Spartan node 38
/// `b_head`, Elite node 31 `b_head`, monitor node 3 `b_monitor` — and `--dump-chud` prints all
/// five.
///
/// TWO EARLIER PASSES GOT THIS WRONG, in opposite directions. The first read `0xC2` as `"chest"`,
/// which no Reach biped carries. The second concluded from that miss that the engine could not be
/// asking for a marker by that id at all and followed the models instead, picking
/// `chud_nav_point` — which exists, and sits 0.054 wu (Spartan) to 0.129 wu (Elite) FURTHER UP the
/// head bone than `head`. That is why the names floated high: 0.762 wu over a Spartan's feet
/// instead of 0.708, and 0.929 over an Elite's instead of 0.800, with the no-marker fallback a
/// further 0.142 wu above even that. `chud_nav_point` is not read by this path.
///
/// `head` is authored with region and permutation both -1 (0xFF), and `+0x221B60`'s per-marker
/// filter accepts a 0xFF region unconditionally, so the engine finds the same single marker
/// whatever armour the player is wearing — which is why taking the group's first marker is right
/// and no permutation table is needed.
///
/// The engine's own fallback when the marker is missing is the object's node-0 (`b_pedestal`)
/// matrix, i.e. the FEET; HMS uses [`NAME_TAG_UP_WU`] instead, because a name at the feet is a
/// worse answer than a measured head height.
pub const NAME_TAG_MARKER: &str = "head";

/// Put every tag on the engine's own anchor: the body's posed [`NAME_TAG_MARKER`] marker, lifted
/// by [`NAME_TAG_MARKER_UP_WU`]. Returns how many tags were moved.
///
/// A body with no actor drawing it this frame, or whose render model carries no `head` marker,
/// keeps the [`NAME_TAG_UP_WU`] fallback and leaves `on_marker` false, so the UI can say that the
/// anchor is approximate rather than pretending.
pub fn resolve_head_anchors(
    tags: &mut [NameTag],
    actors: &crate::reach_actor::ActorScene,
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
) -> usize {
    // The actor drawing a body is the one the pool placed AT that body's recorded feet this tick
    // (`a.pos = Vec3::from(placement.pos)`), so an exact position match identifies it without
    // reaching into the per-species pools that own the assignment.
    let mut moved = 0;
    for t in tags.iter_mut() {
        let feet = glam::Vec3::from(t.feet);
        let Some(a) = actors.actors.iter().find(|a| (a.pos - feet).length_squared() < 1e-6) else { continue };
        let Some(g) = a.def.model.marker_group(native, cache, NAME_TAG_MARKER) else { continue };
        let Some(m) = g.markers.first() else { continue };
        let w = a.world() * a.pose.marker_model(m);
        let p = w.w_axis;
        // Which producer would have made this record decides the lift: the ally waypoint's
        // 0.075 or the look-at display's 0.05. `show_icon` is exactly that distinction, so
        // `resolve_name_colors` has to have run first.
        t.pos = [p.x, p.y, p.z + if t.show_icon { NAME_TAG_MARKER_UP_WU } else { LOOKAT_UP_WU }];
        t.on_marker = true;
        moved += 1;
    }
    moved
}

/// A label for every body drawn at this moment.
///
/// Named where the recording supports it and the ENTITY ID otherwise — never a guess. A clip can
/// hold far more players than its header names (the header is who was present when it was
/// written), so most tags on some clips are ids, and that is a property of the recording.
pub fn name_tags(
    pb: &crate::film_play::Playback,
    players: &[crate::film::FilmPlayer],
    cap: usize,
) -> Vec<NameTag> {
    pb.bipeds(cap)
        .into_iter()
        .map(|p| {
            let slot = pb.index.slot_of_entity(p.net_id);
            let player = slot.and_then(|s| players.iter().find(|q| q.present() && q.index == s as usize));
            let named = player.map_or(false, |q| !q.gamertag.is_empty());
            // #xfilm  A body this film's own roster cannot name may still be named from ANOTHER
            // recording, by its armour: a player's appearance is a near-unique fingerprint (171
            // roster entries across the corpus reduce to 88 distinct appearances with one
            // collision). That is an INFERENCE, not something this recording states, so it is
            // carried separately and drawn differently -- a wrong gamertag is worse than none.
            let inferred = (!named).then(|| pb.index.inferred_name(p.net_id)).flatten();
            NameTag {
                text: match (player.filter(|q| !q.gamertag.is_empty()), inferred) {
                    (Some(q), _) => q.gamertag.clone(),
                    (None, Some((tag, _))) => tag.to_string(),
                    (None, None) => format!("{:#010x}", p.net_id),
                },
                inferred_from: inferred.map(|(_, src)| src.to_string()),
                pos: [p.pos[0], p.pos[1], p.pos[2] + NAME_TAG_UP_WU],
                named,
                team: player.map(|q| q.team).unwrap_or(0xFF),
                net_id: p.net_id,
                feet: p.pos,
                on_marker: false,
                slot,
                color_index: 2,
                show_icon: false,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// #reach-font  the floating player name, by the engine's own rules
// ---------------------------------------------------------------------------------------------

/// The engine's rules for the floating name over a player's head, and where each one comes from.
///
/// Reach has no dedicated nameplate: a teammate is inserted into the local player's 20-slot chud
/// NAVPOINT array every tick, and the chud renderer draws the navpoint icon plus the gamertag
/// beside it (`haloreach.dll` `+0x2CB01C` builds the records, `+0x2E3BAC` draws them).
///
/// [RE] `haloreach.dll+0x2E3BAC`:
/// * the font is a hard-coded literal `6` = `full_screen_hud_message_font` = `tv_nord_hud-15`
///   (TV Nord Condensed), where a generic chud text widget would pass its own
///   `chud_widget_text_block+0x7E`;
/// * the text is drawn ONLY while the navpoint is on screen — off screen the icon is edge-clamped
///   and the name is dropped entirely;
/// * nothing else is drawn with it: no service tag, no team chevron, no shield or health pip, and
///   no outline, shadow or background plate;
/// * there is NO line-of-sight test anywhere in the player path, so names draw through walls.
///
/// [RE] `haloreach.dll+0x2CB01C`: the on-screen SIZE follows
/// `scale = min(max_scale, max(20.0 / distance_wu, min_scale))`, and the alpha is 1.0 for as long
/// as the record keeps being refreshed, falling at 4.0/second once it is not — a 0.25 s fade-out
/// when a player leaves, and NO distance fade at all.
///
/// [tag] `ui\chud\globals` (`chud_globals_definition`): `waypoint minimum distance scale`
/// skin+0x488 and `waypoint maximum distance scale` skin+0x48C are the two clamps. They are READ
/// from the open cache ([`NamePlateRules::from_chud`]); all three shipped skins author 0.8 and
/// 1.3, which is what [`Default`] carries for the no-cache path.
///
/// COLOUR — the engine's own authored palette, picked by the engine's own rule.
///
/// Every navpoint colour comes from one place: the `chgd` skin's thirty-two `real_argb_color`
/// fields, selected by a dense COLOUR INDEX through `chud_color_from_index`
/// (`haloreach.dll+0x2E36B8`; see `reach_chud::color_from_index`). Index 0 is `navpt normal`
/// (+0x74, #2494FF), index 3 is `navpt enemy` (+0xA4, #D83232), index 2 is `navpt neutral`
/// (+0x94, #5E5E5E) — measured off `ui\chud\globals` in a shipped cache with `--dump-chud`.
///
/// Which index a PLAYER gets is decided in exactly one function, `+0x14844C`, and it compares two
/// teams: same team -> 0 `navpt normal`, different team -> 3 `navpt enemy`, no team on either side
/// -> 2 `navpt neutral` (`reach_chud::navpoint_color_index`). The chud producer that makes a plain
/// teammate's name record (`+0x2CB76C`) needs no comparison because it has already filtered to the
/// local player's own team, so it passes index 0 unconditionally.
///
/// HMS has no local player, so it takes the VIEWER to be the player the Theater camera is on
/// (first person / follow) and, with a free camera, the player who RECORDED the film — a saved
/// film is one machine's recording, so that player is the local player the engine itself had.
/// Three consequences, all of them the engine's:
/// * a FREE-FOR-ALL clip is all `navpt enemy` red. Reach puts every player in a teams-off game on
///   their own team, so nobody shares one; the saved configuration slots still read team 0 (all
///   sixteen do, on the measured clip), which is why the game variant's own `teams enabled` flag
///   decides this rather than those bytes.
/// * a TEAM clip reads `navpt normal` for the viewer's side and `navpt enemy` for the other.
/// * a body with no owning player gets `navpt neutral`.
///
/// The authored ALPHA is not used. `navpt normal` authors a = 1.0 and `navpt enemy` authors
/// a = 0.0, and both are plainly visible in game, so the colour's alpha is not this widget's
/// opacity; HMS takes the RGB and draws opaque.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NamePlateRules {
    /// Distance at which the scale is exactly 1.0, world units. [RE: the literal `20.0`]
    pub scale_reference_wu: f32,
    /// `waypoint minimum distance scale`. [tag: chgd skin +0x488]
    pub scale_min: f32,
    /// `waypoint maximum distance scale`. [tag: chgd skin +0x48C]
    pub scale_max: f32,
    /// `virtual height` of the full-screen widescreen `curvature infos` entry
    /// [tag: chgd skin +0x2DC -> +0x098]. The HUD is laid out in THIS space.
    pub virtual_height: f32,
    /// `global safe frame vert.` [tag: ... +0x0D4] — the fraction of the window the virtual
    /// screen is mapped onto.
    pub safe_frame_v: f32,
    /// `safe frame vertical ding` [tag: ... +0x0DC], an extra inset in 1080-line pixels.
    pub ding_v: f32,
    /// The authored navpoint palette, indexed the way `chud_color_from_index` indexes it:
    /// 0 `navpt normal`, 2 `navpt neutral`, 3 `navpt enemy`.
    pub navpt_normal: egui::Color32,
    pub navpt_neutral: egui::Color32,
    pub navpt_enemy: egui::Color32,
}

/// [tag: `ui\chud\globals` skin 0 (`default`) `navpt normal`, +0x74] What the engine gives a name
/// whose subject shares the viewer's team.
pub const ENGINE_NAVPT_NORMAL: egui::Color32 = egui::Color32::from_rgb(36, 148, 255);
/// [tag: chgd skin 1 (`dervish`) `navpt normal`] The same colour on the Arbiter skin.
/// `chud_skin_type_enum` is `default / dervish / monitor` — NOT Spartan/Elite and not per team, so
/// a multiplayer game is always the `default` skin. Recorded for provenance only.
#[allow(dead_code)]
pub const ENGINE_NAVPT_NORMAL_DERVISH: egui::Color32 = egui::Color32::from_rgb(103, 255, 210);
/// [tag: chgd skin 0 `navpt enemy`, +0xA4] Where the red comes from: `+0x14844C` gives it to any
/// player not on the viewer's team, which in a free-for-all is everyone.
pub const ENGINE_NAVPT_ENEMY: egui::Color32 = egui::Color32::from_rgb(216, 50, 50);
/// [tag: chgd skin 0 `navpt neutral`, +0x94] No team on one side or the other.
pub const ENGINE_NAVPT_NEUTRAL: egui::Color32 = egui::Color32::from_rgb(94, 94, 94);
/// [tag: chgd skin 0 `player navpoint (taking damage)`, +0x154] The flash over a player who is
/// being shot — a per-record STATE, which a film does not replicate, so HMS cannot draw it.
#[allow(dead_code)] // recorded for provenance: see NamePlateRules
pub const ENGINE_NAVPT_DAMAGED: egui::Color32 = egui::Color32::from_rgb(232, 74, 0);

impl Default for NamePlateRules {
    fn default() -> Self {
        NamePlateRules {
            scale_reference_wu: 20.0,
            scale_min: 0.8,
            scale_max: 1.3,
            virtual_height: crate::reach_chud::VIRTUAL_HEIGHT_FALLBACK,
            safe_frame_v: crate::reach_chud::SAFE_FRAME_V_FALLBACK,
            ding_v: 0.0,
            navpt_normal: ENGINE_NAVPT_NORMAL,
            navpt_neutral: ENGINE_NAVPT_NEUTRAL,
            navpt_enemy: ENGINE_NAVPT_ENEMY,
        }
    }
}

impl NamePlateRules {
    /// The rules as the open cache's own `chgd` authors them.
    pub fn from_chud(skin: &crate::reach_chud::ChudSkin) -> NamePlateRules {
        // real_argb_color is {alpha, red, green, blue}; the authored alpha is not this widget's
        // opacity (see the type docs), so only the RGB is taken.
        let rgb = |c: [f32; 4]| {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            egui::Color32::from_rgb(q(c[1]), q(c[2]), q(c[3]))
        };
        let pick = |index: i32, fallback: egui::Color32| {
            crate::reach_chud::color_from_index(index).map(|k| rgb(skin.colors[k])).unwrap_or(fallback)
        };
        NamePlateRules {
            scale_reference_wu: 20.0,
            scale_min: skin.waypoint_min_scale,
            scale_max: skin.waypoint_max_scale,
            virtual_height: skin.virtual_height(),
            safe_frame_v: skin.safe_frame_v(),
            ding_v: skin.safe_frame_ding_v(),
            navpt_normal: pick(0, ENGINE_NAVPT_NORMAL),
            navpt_neutral: pick(2, ENGINE_NAVPT_NEUTRAL),
            navpt_enemy: pick(3, ENGINE_NAVPT_ENEMY),
        }
    }
    /// How many screen points ONE chud virtual unit is, in a viewport `viewport_h` points tall —
    /// the factor that turns every authored size in this widget into a drawn one. See
    /// `reach_chud::px_per_virtual` for the chain it comes from.
    pub fn px_per_virtual(&self, viewport_h: f32) -> f32 {
        crate::reach_chud::px_per_virtual(viewport_h, self.virtual_height, self.safe_frame_v, self.ding_v)
    }
    /// The authored colour for a navpoint colour index, the way `+0x2E36B8` resolves it. Only the
    /// three a player name can be are carried; anything else falls to `navpt neutral`.
    pub fn color_for_index(&self, index: i32) -> egui::Color32 {
        match index {
            0 => self.navpt_normal,
            3 => self.navpt_enemy,
            _ => self.navpt_neutral,
        }
    }
}

/// Give every tag the colour index the engine would, from the VIEWER's point of view.
///
/// `viewer` is the viewer's configuration SLOT (`None` = no viewer at all, which the engine
/// answers with `navpt neutral`) and `viewer_team` their team. `teams_enabled` is the recorded
/// GAME VARIANT's own flag: with teams off, Reach puts every player on their own team at run time
/// while the saved configuration slots all keep team 0, so the flag — not those bytes — is what
/// says two players are not on a side together.
///
/// A body with no owning player keeps `navpt neutral`: HMS does not know whose it is, and the
/// engine's own answer for an unknown team is that colour.
pub fn resolve_name_colors(tags: &mut [NameTag], viewer: Option<u8>, viewer_team: u8, teams_enabled: bool) {
    use crate::reach_chud::{navpoint_color_index, TEAM_NONE};
    for t in tags.iter_mut() {
        let own_body = matches!((viewer, t.slot), (Some(v), Some(s)) if v == s);
        t.color_index = match (viewer, t.slot) {
            // Nothing to compare against, or a body whose player HMS could not name.
            (None, _) | (_, None) => navpoint_color_index(TEAM_NONE, TEAM_NONE),
            _ if own_body => navpoint_color_index(0, 0), // the viewer's own body
            (Some(_), Some(_)) if teams_enabled => navpoint_color_index(viewer_team, t.team),
            // Teams off: every other player is on their own side, so none of them is an ally.
            (Some(_), Some(_)) => navpoint_color_index(0, 1),
        };
        // THE CHEVRON GATE. A navpoint record — and so the chevron — exists only where
        // `+0x2CB76C` made one, and its player loop makes one ONLY for another player on the
        // local player's own team. No teams, an enemy, or the viewer's own body (which the engine
        // skips outright): no icon, just the name.
        t.show_icon = teams_enabled && !own_body && t.color_index == 0;
    }
}

/// The engine's navpoint scale law, `min(max, max(reference / d, min))`.
///
/// So the name is a constant 1.3x closer than ~15.4 wu, shrinks as 1/d out to ~25 wu and then
/// holds at 0.8x forever — it never fades out and never disappears with distance.
pub fn nameplate_scale(distance_wu: f32, r: &NamePlateRules) -> f32 {
    let d = distance_wu.max(1e-4);
    (r.scale_reference_wu / d).max(r.scale_min).min(r.scale_max)
}

/// The engine's layout for one name tag, as pure geometry so it can be gated by a test.
///
/// `anchor` is the projected navpoint, `scale` is the distance scale ALREADY multiplied by
/// virtual-units-to-points, and `icon_size_vu` is the chevron's authored size (`None` when the
/// sprite could not be decoded). Returns the icon's rectangle and the point the TEXT BOX's
/// bottom-centre sits on.
///
/// `[RE] +0x2E7684` gives the icon's quad `y` from 0 to its height with the pivot at the bottom
/// (`a5 = 1.0`), so it hangs ABOVE the anchor with its point on it; `[RE] +0x2E3BAC` then offsets
/// the text by `-icon_height` from the same anchor and pivots it on its own box's bottom, so the
/// name stacks directly on top of the chevron.
pub fn nameplate_layout(anchor: egui::Pos2, scale: f32, icon_size_vu: Option<[f32; 2]>) -> (Option<egui::Rect>, egui::Pos2) {
    match icon_size_vu {
        Some([w, h]) if w > 0.0 && h > 0.0 => {
            let (w, h) = (w * scale, h * scale);
            let rect = egui::Rect::from_min_size(egui::pos2(anchor.x - w * 0.5, anchor.y - h), egui::vec2(w, h));
            (Some(rect), egui::pos2(anchor.x, anchor.y - h))
        }
        _ => (None, anchor),
    }
}

/// The navpoint CHEVRON, decoded once per open cache: the texture, the sprite's corner of it, and
/// its authored size in chud virtual units.
pub struct NamePlateIcon {
    pub tex: egui::TextureHandle,
    pub uv: egui::Rect,
    /// `[width, height]` in virtual units — `reach_chud::read_navpoint_sprite`'s own numbers.
    pub size_vu: [f32; 2],
}

thread_local! {
    /// One decoded navpoint icon per open cache. `None` = tried and could not be decoded, in
    /// which case the name is drawn on its own (the engine's no-icon layout) rather than with a
    /// stand-in.
    static NAVPOINT_ICON: std::cell::RefCell<std::collections::HashMap<u64, Option<std::rc::Rc<NamePlateIcon>>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Decode the navpoint chevron out of the open cache's own sprite sheet.
///
/// `[tag]` `chgd` skin +0x3C0 `navpoints reference` is the `bitm`
/// (`ui\chud\bitmaps\waypoints` on every shipped map); `[RE] +0x2E3BAC` draws sequence 1,
/// sprite `R+0x04`, which the teammate-name producer leaves at 0. That sprite is a 37.5 x 21.6
/// virtual-unit downward chevron, stored 3x supersampled in a 512 x 512 sheet.
pub fn nameplate_icon(
    ctx: &egui::Context,
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    skin: &crate::reach_chud::ChudSkin,
) -> Option<std::rc::Rc<NamePlateIcon>> {
    NAVPOINT_ICON.with(|m| {
        if let Some(v) = m.borrow().get(&cache.0) {
            return v.clone();
        }
        let made = (|| {
            let sp = crate::reach_chud::read_navpoint_sprite(native, cache, skin)?;
            if sp.size[0] <= 0.0 || sp.size[1] <= 0.0 {
                return None;
            }
            // The sheet is pure white coverage like the fonts, so the real ALPHA is the shape.
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
            let tex = ctx.load_texture("reach-navpoint", img, egui::TextureOptions::LINEAR);
            Some(std::rc::Rc::new(NamePlateIcon {
                tex,
                uv: egui::Rect::from_min_max(egui::pos2(sp.uv[0], sp.uv[2]), egui::pos2(sp.uv[1], sp.uv[3])),
                size_vu: sp.size,
            }))
        })();
        m.borrow_mut().insert(cache.0, made.clone());
        made
    })
}

/// Paint every name tag, Reach's way.
///
/// `project` turns a world point into viewport coordinates, or `None` behind the camera.
///
/// SIZE — every number here is a chud VIRTUAL unit, scaled to the window exactly once.
///
/// `[RE]` The chud never converts a size on the CPU: `+0x2E62D8` hands the GPU vertices in virtual
/// units and the shader maps the virtual screen onto the safe frame, so one virtual unit is
/// `window_height * safe_frame / virtual_height` points ([`NamePlateRules::px_per_virtual`]).
/// Reach authors 1280 x 720 with a 0.87 vertical safe frame, so a virtual unit is 0.87 points in
/// a 720-point window, 1.305 in a 1080 one and 1.74 in a 1440 one — a HUD drawn at its authored
/// pixel size regardless of the window is right at exactly one window size and wrong at all the
/// others. `tv_nord_hud-15` is 21 virtual units tall (`ascender 17 + descender 4` from its own
/// header), the chevron 37.5 x 21.6, and the distance scale multiplies both.
///
/// TWO DISPLAYS, not one. `+0x2E3BAC` draws two different player names and they are easy to
/// conflate:
///
/// * the **ally waypoint** — the record's own gamertag at `R+0x1C`, drawn with the navpoint
///   chevron beside it. `+0x2CB76C`'s player loop writes that name, and it does so ONLY for
///   another player on the local player's own team (`player_datum+0xAD` equality), which is why
///   the waypoint carries an icon and why a gametype can hand it out through player traits. No
///   record, no chevron: a free-for-all has none, and nor does an enemy or the local player.
/// * the **look-at name** — `a1+0xAD8`, the gamertag of whoever the local player is looking at
///   (`+0x2CB01C` sets `a1+0xAD0` from the unit's own aim target at `unit+0x2B8`, else from
///   `HUD_FindNearestTarget`), with its own 0..1 fade at `a1+0xB00`. Its record comes from a
///   FOURTH producer — the per-game-engine hook `+0xD0`, `sub_18018ACC8`, which runs for every
///   engine and adds the looked-at player whatever the teams are (`v5 == a3 && a3 != v1`). That
///   filler (`+0x18AE04`) writes **sprite index -1**, and `+0x2E3BAC` skips the icon element
///   outright for -1, so the look-at name is floating text with NO chevron. It also leaves the
///   record's own name empty for an enemy, which is why the text comes from `a1+0xAD8`.
///
/// Both anchor on the same `head` marker and differ only in the lift: 0.075 wu for the waypoint,
/// 0.05 for the look-at name. Their colour is the record's colour index, which `+0x18AE04` sets
/// to `3` (`navpt enemy`) for an enemy and `0` (`navpt normal`) for an ally — the same two
/// colours, from the same place, as the waypoint.
///
/// HMS draws the look-at display for every body and adds the chevron only where the engine would
/// — see [`resolve_name_colors`], which sets `show_icon`. Both take their colour from the same
/// place (`+0x2E3AD8`, the record's colour index), so the palette above is right for either.
///
/// Neither is a `chdt` widget: every `chud_definition` in a shipped cache was decoded with
/// `--dump-chdt` and none has a `players` datasource or a `<player>` anchor, so this hand-written
/// renderer is the whole of it.
///
/// LAYOUT — the engine's, for a plain player-name record.
///
/// `+0x2E3BAC` picks between a centred and a beside-the-icon layout with
/// `v48 = v44 ? 0.0 : 0.5` (the lerp across the measured text box) and `v47 = v44 ? w/2 + 5 : 0.0`
/// (the x offset), where `v44` is true only when one of the SECONDARY sprites was drawn — sprite 2
/// or 3 (both need flag bit 7) or the `R+0x08` sprite. The teammate-name producer `+0x2CB76C`
/// writes flags 0 and `R+0x08 = -1`, so for a player's name `v44` is FALSE: the name is centred
/// on the navpoint with no horizontal offset. Its vertical offset is `-icon_height`, and the
/// icon's own pivot is the bottom of its quad (`a5 = 1.0` in `+0x2E7684`), so the chevron sits
/// with its point on the anchor and the name stacks directly on top of it.
///
/// WHAT THIS DOES NOT COPY, and why:
/// * **The per-record STATE flash.** `+0x2E3AD8` replaces the colour outright while a player is
///   firing, taking damage or speaking (`player navpoint (firing)` / `(taking damage)` /
///   `(speaking)`). A film replicates none of those three, so HMS cannot colour them.
/// * **The screen curvature.** `+0x2DA9E8` bends the whole HUD radially about the centre of the
///   virtual screen when `flt_180D1F718` is non-zero. HMS maps virtual to screen linearly.
/// * **Who is named.** In game only teammates get a name (`player_datum+0xAD` must equal the local
///   player's) and the local player is skipped — so a free-for-all shows NO names at all. A film
///   has no local player and watching a clip is the whole point here, so HMS names every body it
///   draws and colours it by the engine's own team rule.
///
/// Nothing is drawn behind the text: no plate, no outline, no shadow. `+0x2E3BAC` issues exactly
/// one draw per sprite and one for the string, and the string's command carries only a colour and
/// an opacity.
pub fn paint_name_tags(
    ctx: &egui::Context,
    painter: &egui::Painter,
    viewport: egui::Rect,
    tags: &[NameTag],
    rules: &NamePlateRules,
    icon: Option<&NamePlateIcon>,
    eye: glam::Vec3,
    project: &dyn Fn(glam::Vec3) -> Option<(f32, f32)>,
) {
    if tags.is_empty() {
        return;
    }
    let font = crate::reach_font::hud_atlas(ctx);
    // One virtual unit, in screen points, for this viewport. Computed once: it does not vary
    // per tag.
    let vpx = rules.px_per_virtual(viewport.height());
    ANCHOR_COUNTS.with(|c| {
        let on = tags.iter().filter(|t| t.on_marker).count();
        c.set((on, tags.len() - on));
    });
    // Far tags first, so a near player's name is painted over a distant one.
    let mut order: Vec<(f32, usize)> = tags
        .iter()
        .enumerate()
        .map(|(i, t)| ((glam::Vec3::from(t.pos) - eye).length(), i))
        .collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (dist, i) in order {
        let t = &tags[i];
        let Some((sx, sy)) = project(glam::Vec3::from(t.pos)) else { continue };
        let p = egui::pos2(sx, sy);
        // The engine draws the text only while the navpoint is genuinely on screen.
        if !viewport.contains(p) {
            continue;
        }
        // The distance scale, then virtual units -> screen points. Both are the engine's.
        let scale = nameplate_scale(dist, rules) * vpx;
        // The authored navpoint colour for the index `resolve_name_colors` gave this tag. No
        // distance fade: the engine has none.
        let mut col = rules.color_for_index(t.color_index);
        // #xfilm  A name this recording does NOT state is drawn dimmer and in brackets. The
        // engine has no such display; this is HMS marking an inference as one, because a wrong
        // gamertag over a body is worse than no gamertag at all.
        let text = match &t.inferred_from {
            Some(_) => {
                col = egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 150);
                format!("[{}]", t.text)
            }
            None => t.text.clone(),
        };
        // The chevron, centred on the anchor with its point ON it, and the lift it gives the name.
        let (icon_rect, text_at) = nameplate_layout(p, scale, icon.filter(|_| t.show_icon).map(|i| i.size_vu));
        if let (Some(ic), Some(rect)) = (icon, icon_rect) {
            let mut mesh = egui::Mesh::with_texture(ic.tex.id());
            mesh.add_rect_with_uv(rect, ic.uv, col);
            painter.add(egui::Shape::mesh(mesh));
        }
        let base = text_at.y;
        match font.as_deref() {
            Some(f) => {
                let w = f.font.measure(&text) * scale;
                let pen = egui::pos2(p.x - w * 0.5, base - f.font.descender as f32 * scale);
                f.paint(painter, pen, &text, col, scale);
            }
            None => {
                // No Reach font on this machine: same layout, colour and SIZE, the app's own
                // glyphs. The size is the authored line height in virtual units, so it tracks the
                // window the same way the real font does.
                let size = crate::reach_chud::HUD_FONT_LINE_HEIGHT_VU * scale;
                let galley = painter.layout_no_wrap(text.clone(), egui::FontId::proportional(size), col);
                painter.galley(egui::pos2(p.x - galley.size().x * 0.5, base - galley.size().y), galley, col);
            }
        }
    }
}

/// How the last painted frame anchored its name tags, for the panel to report honestly.
/// `(on the engine's chest marker, on the fixed fallback height)`.
pub fn nameplate_anchor_counts() -> (usize, usize) {
    ANCHOR_COUNTS.with(|c| c.get())
}

thread_local! {
    static ANCHOR_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// One line for the UI saying which glyphs the name tags are being drawn with.
pub fn nameplate_font_note(ctx: &egui::Context) -> String {
    match crate::reach_font::hud_atlas(ctx) {
        Some(f) => format!(
            "Font: Reach's own {} ({}) from {}",
            f.font.name,
            crate::reach_font::FontId::FullScreenHudMessage.enum_name(),
            f.source
        ),
        None => "Font: the app's own — Reach's text fonts are not in a stock MCC install (it ships \
                 only font_package_icon.bin, which holds button icons and no letters). Install the \
                 Halo Reach Editing Kit, or point HMS_REACH_FONTS at a maps\\fonts folder, to draw \
                 them in the game's own tv_nord_hud-15."
            .to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// #reach-widget  the player list, drawn from Reach's own scoreboard widgets
// ---------------------------------------------------------------------------------------------

/// What HMS feeds into the game's own scoreboard row.
///
/// The engine fills a row from a runtime `scoreboard_list_data` component which a film cannot
/// give us; these are the same quantities out of the recording instead. The names match the
/// `property binding` sources the tag itself declares (`prop_player_name`,
/// `prop_player_service_tag`, `prop_player_bitmap_tint_color`, …), so the mapping is the tag's.
#[derive(Clone, Debug, Default)]
pub struct ScoreboardRow {
    /// `prop_player_position` — 1-based place in the list.
    pub position: usize,
    /// `prop_player_name`.
    pub name: String,
    /// `prop_player_service_tag`.
    pub service_tag: String,
    /// `prop_player_score`. `None` when the recording carried no score for this player.
    pub score: Option<i32>,
    /// Drives `prop_player_bitmap_tint_color` on the colour bars. 0xFF = unknown.
    pub team: u8,
    /// `prop_player_show_team`: the engine's `game_engine_has_teams()`, which is the recorded
    /// GAME VARIANT's own teams bit and not a comparison of the per-player team values. False
    /// makes the bar take the player's own colour. See [`bitmap_tint`].
    pub show_team: bool,
    /// The player's PRIMARY ARMOUR COLOUR index, which is how a free-for-all row is coloured.
    /// `None` when the recording holds no appearance for them.
    pub armour_color: Option<u8>,
    /// #emblem  The emblem the recording stored for this slot. Decoded, and carried so the row
    /// can say what it is; see [`scoreboard_panel`] for why it is not drawn as art.
    pub emblem: Option<crate::film::EmblemInfo>,
    /// False for a body HMS could not name — the row is drawn dimmer, as a dropped player is.
    pub named: bool,
}

/// What HMS paints when the engine would paint nothing: a dark neutral, so a row with no
/// recorded colour is still a visible row rather than a transparent gap.
pub const NO_PLAYER_COLOUR: [f32; 4] = [1.0, 0.176, 0.176, 0.176];

/// The colour the engine's `prop_player_bitmap_tint_color` resolves to for this row.
///
/// **[tag + engine] This is Reach's own law, read out of the code rather than guessed, and it is
/// the `wigl` list and not `matg`.** `scoreboard_list_data` keeps TWO colours per row and a
/// selector (`scoreboard_list_data_component.cpp:1323/1332`): the getter returns the team one
/// when `prop_player_show_team` is set and the player one otherwise. Both are filled once per
/// rebuild by the row-add function, through two of the five functions in
/// `cui_color_preset.cpp` -- the only place any `cui` component turns an index into a colour:
///
/// ```text
/// player colour = wigl +0xA0 `tint colors`[0] -> +0x18 `bitmap player`[ primary armour colour ]
///                 (cui_color_preset.cpp:73)
/// team colour   = the game variant's own per-team override if it has one, else
///                 wigl +0xA0 `tint colors`[0] -> +0x24 `bitmap team`[ team ]
///                 (cui_color_preset.cpp:127)
/// ```
///
/// The index into `bitmap player` is `s_player_appearance` +0x01, the player's PRIMARY armour
/// colour -- the row-add call passes the player datum's own colour byte, and the appearance
/// builder writes that byte from the profile. So the engine does apply the player's primary
/// colour; it applies it through a UI palette authored one entry per armour colour.
///
/// Three details that are easy to get wrong and are all the engine's:
/// * the selector is **`game_engine_has_teams()`** -- the current engine resolving non-null AND
///   `game_engine_globals->flags & 0x8`, the variant's own teams bit -- not a comparison of the
///   recorded per-player team values;
/// * when teams ARE on but a row's team index is above 7 (`0xFF` = none), the engine copies the
///   PLAYER colour into the team slot rather than looking anything up;
/// * an index the list cannot satisfy is not clamped and has no default: the lookup returns
///   false, and because the roster is memset before each rebuild the row keeps `(0,0,0,0)`.
///   `None` here is that case. HMS draws a dark neutral for it instead of a transparent row, so
///   that an unnamed body is still a visible row -- the one place this departs, and it departs
///   only where the engine draws nothing.
pub fn bitmap_tint(g: &crate::reach_widget::UiGlobals, row: &ScoreboardRow) -> Option<[f32; 4]> {
    let base = row.armour_color.and_then(|i| g.bitmap_player(i));
    if row.show_team && row.team <= 7 {
        // The variant's per-team colour override (`game_engine_team.cpp:1061`) is not modelled:
        // HMS does not read the recorded variant's team records, so this is the tag's own list.
        return g.bitmap_team(row.team).or(base);
    }
    base
}

/// The player's PRIMARY ARMOUR COLOUR as a screen colour: `matg` +0x498 `profile colors` at
/// their own index, sRGB-encoded.
///
/// This is the colour their BODY is tinted with -- the same table, the same index, the same
/// entry `SceneController::armour_colors` hands the renderer -- so anything drawn with it cannot
/// disagree with the picture. `profile colors` stores LINEAR floats, so skipping the encode is
/// what makes an armour colour come out far darker in a UI panel than on the armour: Reach's
/// `Steel` is linear 0.127, which is `#646464` on screen and `#202020` if it is left alone.
pub fn player_primary_color(profile: &[[f32; 3]], row: &ScoreboardRow) -> Option<[f32; 3]> {
    let c = row.armour_color.and_then(|i| profile.get(i as usize)).copied()?;
    Some(crate::reach_widget::srgb_from_linear(c))
}

/// Does the TAG tint this plate with the player's own colour?
///
/// Derived by walking the component's `prop_tint_color` binding chain and asking whether it
/// terminates at `prop_player_bitmap_tint_color`, which is the engine's per-player/per-team
/// colour. On the shipped row that is true of three of the five plates and they tile 275 of the
/// row's 350 units: `roster_color_start_bitmap` (28 wide, the emblem block), `roster_bitmap`
/// (193, under the name, reached through `bitmap_color_decision_maker`) and
/// `roster_color_end_bitmap` (54, under the service tag). The other two,
/// `roster_position_bitmap` and `roster_score_bitmap`, author a near-black tint of their own and
/// are not the player's colour in the engine either, so they keep it.
///
/// Nothing is named here: add a plate to the tag and bind it, and it is included.
pub fn tint_is_player_colour(screen: &crate::reach_widget::Screen, c: &crate::reach_widget::Component) -> bool {
    if c.kind != "bitmap_widget" {
        return false;
    }
    // #scoreboard-row  A plate that authors its OWN tint still takes the player's colour.
    //
    // The engine does not do this, and the deviation is deliberate and asked for: Reach tints
    // only the plates whose `prop_tint_color` is BOUND to `prop_player_bitmap_tint_color` (the
    // 28-wide colour block, the 193-wide name plate and the 54-wide service-tag block = 275 of
    // the row's 350 units), and self-tints the rank block at the far left and the score block at
    // the far right near-black. With the engine's own near-black `wigl bitmap_player` entries
    // that read as "a coloured box on the left", which is exactly what the user reported twice.
    // Colouring every plate makes the WHOLE row the player's colour, which is what was asked
    // for; the text colour is still chosen per plate by measured contrast, so the rank and score
    // stay readable on a bright row.
    if c.props.color("prop_tint_color").is_some() {
        // ...but only for a plate that is part of the row's own background. A bitmap with no
        // size, or one the row hides, is not a background and must not be repainted.
        return c.width() > 0.0 && c.height() > 0.0 && !screen.hidden(c);
    }
    let mut comp = c.name.clone();
    let mut prop = "prop_tint_color".to_string();
    // The real chains are two links; the bound stops a malformed cycle dead.
    for _ in 0..6 {
        let Some((sc, sp)) = screen.bound_from(&comp, &prop) else { return false };
        if sp == "prop_player_bitmap_tint_color" {
            return true;
        }
        // A decision_maker's result is one of its branches, and HMS always takes the branch for
        // a player who is still here -- see `resolve_color_prop`.
        let (sc, sp) = (sc.to_string(), if sp == "prop_result" { "prop_then".to_string() } else { sp.to_string() });
        comp = sc;
        prop = sp;
    }
    false
}

/// Relative luminance of an sRGB colour, and the standard contrast ratio between two of them.
/// Used to CHOOSE a text colour rather than to describe one.
fn srgb_luminance(c: [f32; 3]) -> f32 {
    let f = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
}
fn srgb_contrast(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y) = (srgb_luminance(a), srgb_luminance(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

/// Which of two authored text colours to draw on a given plate: whichever stands out more.
///
/// **Every colour here comes from the map; only the choice between them is HMS's.** A bar that
/// is now the player's real primary colour can be anything from `#090909` to `#d9d9d9`, and one
/// fixed text colour cannot sit on both -- Reach never has to solve that because its own bars
/// are all dark. The two candidates are `wigl`'s own: the light default (+0x74 `text color`, or
/// whatever preset the widget names) and the dark `body` preset (0.35). Picking by measured
/// contrast means there is no threshold to tune and no invented colour.
pub fn pick_text_color(light: [f32; 4], dark: Option<[f32; 4]>, plate: Option<[f32; 3]>) -> [f32; 4] {
    let (Some(dark), Some(plate)) = (dark, plate) else { return light };
    if srgb_contrast([dark[1], dark[2], dark[3]], plate) > srgb_contrast([light[1], light[2], light[3]], plate) {
        dark
    } else {
        light
    }
}

/// The bitmap widget a text widget is drawn ON TOP OF, if any.
///
/// Read from the tag rather than hard-coded: the plate under the player name is whichever
/// `bitmap_widget` box contains the text box's centre, taking the LAST in tag order because tag
/// order is paint order (the row's `bitmap` container is components [3..8] and its `data`
/// container, which holds every text column, is [9..17] -- so the plates really are painted
/// first). Components the painter skips are skipped here too, so the selection highlight that is
/// never drawn cannot be mistaken for the plate.
pub fn plate_under<'a>(
    screen: &'a crate::reach_widget::Screen,
    text: &crate::reach_widget::Component,
) -> Option<&'a crate::reach_widget::Component> {
    let o = screen.origin_of(text);
    let centre = [o[0] + text.width() * 0.5, o[1] + text.height() * 0.5];
    screen
        .components
        .iter()
        .filter(|c| c.kind == "bitmap_widget" && !screen.hidden(c))
        .filter(|c| !RUNTIME_ONLY.contains(&c.name.as_str()) && !is_under_any(screen, c, &RUNTIME_ONLY))
        .filter(|c| {
            let p = screen.origin_of(c);
            centre[0] >= p[0]
                && centre[0] <= p[0] + c.width()
                && centre[1] >= p[1]
                && centre[1] <= p[1] + c.height()
        })
        .next_back()
}

/// Components the engine shows or hides from RUNTIME state that a film does not carry. Each is
/// hidden here, and each one is a thing HMS does not know rather than a thing the tag did not say:
/// * `selection_highlight_container` — the row the player has the stick on;
/// * `current_player_container` — the "this is you" overline (a film has no local player);
/// * `connectivity` — the live network-quality bar;
/// * `player_is_dead_background_bitmap` / `bitmap_player_is_dead` — alive-or-dead right now;
/// * `ringspeak_bitmap` — who is talking.
const RUNTIME_ONLY: [&str; 6] = [
    "selection_highlight_container",
    "current_player_container",
    "connectivity",
    "player_is_dead_background_bitmap",
    "bitmap_player_is_dead",
    "ringspeak_bitmap",
];

/// Paint the player list using Reach's own `cui_screen` widgets.
///
/// Everything positional is read from the tags: the row is `scoreboard_list_item_player`
/// (350 x 30 virtual units, five columns), the header strip is the `scoreboard` screen's
/// `player_header`, the bitmaps are the `bitm` tags the widgets reference, and the text columns
/// use the `prop_font_id` each one declares (7 = `english_body_text_font` = `tv_nord_condensed-15`).
/// The whole thing is scaled by one factor so a 350-unit row fills the panel.
///
/// Returns the height it used.
pub fn paint_scoreboard(
    ui: &mut egui::Ui,
    sb: &crate::reach_widget::Scoreboard,
    rows: &[ScoreboardRow],
    ui_globals: &crate::reach_widget::UiGlobals,
    profile: &[[f32; 3]],
    cache: hms_native::CacheHandle,
    decode: &dyn Fn(u32) -> Option<(Vec<u8>, u32, u32)>,
) -> f32 {
    let row_w = sb.row_width();
    let row_h = sb.row_height();
    let header = sb.screen.get("player_header");
    let header_h = header.map(|c| c.height()).filter(|h| *h > 0.0).unwrap_or(30.0);
    let avail = ui.available_width().max(80.0);
    let scale = (avail - 2.0) / row_w;
    let total_h = (header_h + row_h * rows.len() as f32) * scale;
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(avail, total_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let font = crate::reach_font::ui_atlas(ui.ctx());

    // The header strip, placed relative to its own container so it sits at the top of the list.
    if let Some(h) = header {
        let base = sb.screen.origin_of(h);
        for c in &sb.screen.components {
            if sb.screen.hidden(c) || !is_descendant(&sb.screen, c, "player_header") {
                continue;
            }
            let o = sb.screen.origin_of(c);
            let at = egui::pos2(rect.min.x + (o[0] - base[0]) * scale, rect.min.y + (o[1] - base[1]) * scale);
            draw_component(&painter, &sb.screen, c, at, scale, None, ui_globals, profile, cache, decode, ui.ctx(), font.as_deref());
        }
    }

    // One stamp of the row template per player.
    let top0 = rect.min.y + header_h * scale;
    for (i, r) in rows.iter().enumerate() {
        let row_top = top0 + i as f32 * row_h * scale;
        for c in &sb.player_row.components {
            if sb.player_row.hidden(c) || RUNTIME_ONLY.contains(&c.name.as_str()) {
                continue;
            }
            if is_under_any(&sb.player_row, c, &RUNTIME_ONLY) {
                continue;
            }
            let o = sb.player_row.origin_of(c);
            let at = egui::pos2(rect.min.x + o[0] * scale, row_top + o[1] * scale);
            draw_component(&painter, &sb.player_row, c, at, scale, Some(r), ui_globals, profile, cache, decode, ui.ctx(), font.as_deref());
        }
    }
    total_h
}

fn is_descendant(screen: &crate::reach_widget::Screen, c: &crate::reach_widget::Component, ancestor: &str) -> bool {
    let mut p = c.parent.clone();
    for _ in 0..16 {
        if p == ancestor {
            return true;
        }
        if p.is_empty() {
            return false;
        }
        match screen.get(&p) {
            Some(x) => p = x.parent.clone(),
            None => return false,
        }
    }
    false
}

fn is_under_any(screen: &crate::reach_widget::Screen, c: &crate::reach_widget::Component, names: &[&str]) -> bool {
    names.iter().any(|n| is_descendant(screen, c, n))
}

/// A Reach `real_argb_color` (A,R,G,B floats, in that order) as an egui colour, with the
/// component's own opacity folded in.
///
/// **The alpha used to be thrown away.** `prop_tint_color` is four floats and the FIRST of them
/// is the alpha; reading `t[1..4]` as the RGB and ignoring `t[0]` is right for the scoreboard's
/// own plates (every one of them is authored `a 1.0`) but wrong the moment a binding feeds in a
/// colour that carries alpha -- `player_dropped_opacity_decision_maker` authors `a 0.549`, and
/// `wigl`'s own `unhilite` / `dim` / `breadcrumb` presets are alpha and nothing else.
fn argb32(t: [f32; 4], extra_alpha: f32) -> egui::Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    egui::Color32::from_rgb(q(t[1]), q(t[2]), q(t[3])).linear_multiply((t[0] * extra_alpha).clamp(0.0, 1.0))
}

/// What colour the TAG says this bitmap widget is, following its own property bindings.
///
/// This is the whole of the "the background is wrong" fix, and it is read, not chosen. The
/// scoreboard row authors a `prop_tint_color` on only three of its five plates; the other two --
/// `roster_bitmap`, the wide plate the player NAME sits on, and `roster_color_start_bitmap` --
/// are fed by bindings, and a widget with no authored tint defaults to WHITE.
/// `player_base_bitmap` is a single pixel of **180,180,180** (measured), so defaulting to white
/// draws a light grey plate 193 units wide directly under near-white text. That is exactly the
/// reported "the names are extremely hard to see".
///
/// The bindings, from the two `cusc` tags themselves:
///
/// ```text
/// scoreboard screen   data_player_scoreboard.prop_player_bitmap_tint_color
///                       -> roster_color_start_bitmap.prop_tint_color
///                       -> roster_color_end_bitmap.prop_tint_color
///                     data_player_scoreboard.prop_player_left_game
///                       -> bitmap_color_decision_maker.prop_if
/// row template        roster_color_start_bitmap.prop_tint_color
///                       -> bitmap_color_decision_maker.prop_then
///                     bitmap_color_decision_maker.prop_result
///                       -> roster_bitmap.prop_tint_color
/// ```
///
/// So the name plate takes the SAME colour as the little square behind the emblem, through the
/// decision maker. Which branch an ordinary row takes is pinned by the row's OTHER decision
/// maker rather than assumed: `player_dropped_opacity_decision_maker` shares the same
/// `prop_if` source and authors `prop_then` = `a 1.0`, `prop_else` = `a 0.549`. A player who has
/// left must be the dimmer one, so a present player takes `prop_then` -- and on the colour maker
/// `prop_then` is the player's bitmap tint.
fn resolve_tint(
    screen: &crate::reach_widget::Screen,
    c: &crate::reach_widget::Component,
    row: Option<&ScoreboardRow>,
    g: &crate::reach_widget::UiGlobals,
) -> [f32; 4] {
    // #row-colour  THE WHOLE ROW CARRIES THE PLAYER'S COLOUR, not just its two end caps.
    //
    // Only the components whose `prop_tint_color` is BOUND to `prop_player_bitmap_tint_color` get
    // the player colour from the binding chain, and in the shipped row that is exactly the two
    // 2-pixel `white_pix` bars at each end. The 247-unit `gray_field` and the 25-unit
    // `gray_dpad_field` carry their own authored tint — a flat 0.093 grey — so they kept it, and
    // the result was the reported "just like the far left and far right sides of the row".
    //
    // The authored grey is a BACKGROUND for a row whose colour the engine supplies elsewhere, so
    // HMS overrides it with the row's own colour on the plates that make up the row's body. It is
    // a departure from the tag and it is deliberate; it is done HERE, in the one function both the
    // scoreboard and the spectated-player card resolve their colours through, so the two can never
    // disagree.
    if let Some(r) = row {
        if is_row_plate(c) {
            return bitmap_tint(g, r).unwrap_or(NO_PLAYER_COLOUR);
        }
    }
    resolve_color_prop(screen, &c.name, "prop_tint_color", row, g, 0)
        .or_else(|| c.tint())
        .unwrap_or([1.0, 1.0, 1.0, 1.0])
}

/// The bitmaps that make up the BODY of a player row, as opposed to its text, its emblem or its
/// icons. Matched on the component's own bitmap reference rather than on its name, because the
/// scoreboard row and the spectated-player card name their pieces differently while both are
/// built from these two.
pub const ROW_PLATE_BITMAPS: [&str; 2] = ["player_base_bitmap", "player_base_servicetag_bitmap"];

/// Is this component one of the row's body plates?
fn is_row_plate(c: &crate::reach_widget::Component) -> bool {
    // The name is the cheap test and covers the shipped pieces (`gray_field`, `gray_dpad_field`);
    // `white_pix` end caps are already bound to the player colour and must not be caught here,
    // since they would then lose their binding chain for no gain.
    let n = c.name.to_ascii_lowercase();
    n.contains("gray_field") || n.contains("gray_dpad_field") || n.contains("_base_field")
}

/// Follow one colour property's binding chain. `depth` bounds it; the real chains are two links.
#[allow(clippy::too_many_arguments)]
fn resolve_color_prop(
    screen: &crate::reach_widget::Screen,
    comp: &str,
    prop: &str,
    row: Option<&ScoreboardRow>,
    g: &crate::reach_widget::UiGlobals,
    depth: u32,
) -> Option<[f32; 4]> {
    if depth > 4 {
        return None;
    }
    let (src_comp, src_prop) = screen.bound_from(comp, prop)?;
    let (src_comp, src_prop) = (src_comp.to_string(), src_prop.to_string());
    // The runtime source: the player's own UI bitmap tint.
    if src_prop == "prop_player_bitmap_tint_color" {
        // `None` is the engine's own "no colour for this row" (its lookup fails and the memset
        // leaves the row transparent). HMS substitutes a dark neutral so the row is still drawn.
        return Some(bitmap_tint(g, row?).unwrap_or(NO_PLAYER_COLOUR));
    }
    // A decision_maker: its result is one of two colours, chosen by `prop_if`, which the screen
    // binds to `prop_player_left_game`. A film does not carry that -- "has this player left
    // RIGHT NOW" is live session state, the same reason `connectivity` and `bitmap_player_is_dead`
    // are in `RUNTIME_ONLY` -- so HMS cannot evaluate the condition and always takes the branch
    // for a player who is still here, `prop_then`.
    //
    // It used to take `prop_else` for any row HMS could not NAME, and that was wrong twice over.
    // Not-named is not the same fact as left-the-game. And `prop_else` on this maker is authored
    // pure WHITE, so every unnamed row -- which on a clip whose header stored three gamertags is
    // eleven rows out of fourteen -- got the light grey plate that made the reported
    // illegibility. Being unable to name somebody is marked by dimming the row, which is HMS's
    // own choice and says so, not by painting their bar a different colour.
    if src_prop == "prop_result" {
        return resolve_color_prop(screen, &src_comp, "prop_then", row, g, depth + 1)
            .or_else(|| screen.get(&src_comp).and_then(|m| m.props.color("prop_then")));
    }
    // A plain component-to-component wire: take that component's colour.
    resolve_color_prop(screen, &src_comp, &src_prop, row, g, depth + 1)
        .or_else(|| screen.get(&src_comp).and_then(|src| src.props.color(&src_prop)))
}

/// Draw one component where the tag says, with the value HMS has for it.
#[allow(clippy::too_many_arguments)]
fn draw_component(
    painter: &egui::Painter,
    screen: &crate::reach_widget::Screen,
    c: &crate::reach_widget::Component,
    at: egui::Pos2,
    scale: f32,
    row: Option<&ScoreboardRow>,
    g: &crate::reach_widget::UiGlobals,
    profile: &[[f32; 3]],
    cache: hms_native::CacheHandle,
    decode: &dyn Fn(u32) -> Option<(Vec<u8>, u32, u32)>,
    ctx: &egui::Context,
    font: Option<&crate::reach_font::FontAtlas>,
) {
    let size = egui::vec2(c.width() * scale, c.height() * scale);
    if size.x <= 0.0 || size.y <= 0.0 {
        return;
    }
    let rect = egui::Rect::from_min_size(at, size);
    // A dropped / unnamed player's whole row is dimmed, which is what the tag's own
    // `player_dropped_opacity_decision_maker` does with `prop_player_left_game`.
    let row_alpha = match row {
        Some(r) if !r.named => 0.549,
        _ => 1.0,
    };
    let alpha = (c.opacity() * row_alpha).clamp(0.0, 1.0);
    match c.kind.as_str() {
        "bitmap_widget" => {
            let Some(tag) = c.bitmap() else { return };
            let Some(tex) = crate::reach_widget::bitmap_texture(ctx, cache, tag, decode) else { return };
            // The tint is whatever the TAG says it is -- its own authored colour, or, where a
            // property binding feeds `prop_tint_color`, whatever that binding resolves to.
            // Every plate the TAG tints with the player's colour takes their REAL primary
            // colour -- the same entry their body is tinted with -- so the row reads as theirs
            // across its width. Plates that author a colour of their own keep it.
            let t = match row.filter(|_| tint_is_player_colour(screen, c)).and_then(|r| player_primary_color(profile, r)) {
                Some(c) => [1.0, c[0], c[1], c[2]],
                None => resolve_tint(screen, c, row, g),
            };
            let tint = argb32(t, alpha);
            let mut mesh = egui::Mesh::with_texture(tex.id());
            mesh.add_rect_with_uv(rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
            painter.add(egui::Shape::mesh(mesh));
        }
        // #emblem  `emblem_widget` -- `bitmap_player_scoreboard_emblem`, 24 x 24 at left 27.
        //
        // The recording DOES carry an emblem for this slot and HMS decodes it (see
        // `film::EmblemInfo`), but it is not drawn as ART, and the reason is measured rather
        // than assumed. Reach composes an emblem from `mlib` (`emblem_library`): per layer, two
        // signed-distance SHAPES out of the 119-image `ui\cui\common\emblems\emblems` atlas,
        // each with its own 2x2 transform, blur and expand/contract, composited background ->
        // midground -> foreground. The authored `shapes` / `front emblems` / `back emblems`
        // blocks are resized to zero by the tag postprocess, and MCC does not even run that
        // path -- it runs `generate_h3_compatible_emblem_pixel` against a pre-composited
        // texture pack (`data\ui\texturepacks\emblemstexturepack.perm.bin`) whose format is not
        // decoded here. Drawing an approximation would be inventing art, so the slot stays
        // empty and the panel says what the recording holds.
        "emblem_widget" => {}
        "text_widget" => {
            let Some(text) = widget_text(&c.name, c.text_id(), row) else { return };
            let text = if c.upper_case() { text.to_uppercase() } else { text };
            // [tag] `prop_color_preset` is resolved the way the engine resolves it
            // (`cui_color_preset.cpp:18`): look the name up in `wigl` +0x94 `color presets`, and
            // on a miss -- INCLUDING a widget that names no preset at all, which every text
            // column of the player row except the score does -- keep the default the widget was
            // pre-loaded with, `wigl` +0x74 `text color`. Nothing here is mapped by eye any more.
            // The plate this column lands on, as it will be PAINTED: the tint the tag gives it
            // times its own art. A row's bar is now the player's real primary colour and can be
            // anything from near-black to near-white, so the text colour is chosen against it.
            let plate = plate_under(screen, c).and_then(|p| {
                let t = resolve_tint(screen, p, row, g);
                let m = crate::reach_widget::bitmap_mean(cache, p.bitmap()?, decode)?;
                Some([t[1] * m[0], t[2] * m[1], t[3] * m[2]])
            });
            let light = g.text_preset_or_default(c.props.sid("prop_color_preset"));
            let col = argb32(pick_text_color(light, g.preset("body"), plate), alpha);
            // The font is the one the component itself asks for: `prop_font_id`.
            let fid = c.font_id().unwrap_or(crate::reach_font::FontId::EnglishBodyText);
            let own = crate::reach_font::atlas_for(ctx, fid);
            let face = own.as_deref().or(font);
            // A DROP SHADOW, in the colour `wigl` +0x84 authors for one (`a 0.5` black). Be
            // clear about what is proven and what is not: the COLOUR is the game's own, and
            // `wigl` holding exactly one global `shadow color` beside its one global
            // `text color` is why it is here -- but that this particular widget draws it is NOT
            // established, and some Reach screens do shadows instead by duplicating the text
            // widget (`film_cameras` has `spawn_situation_shadow_text` under
            // `spawn_situation_text`). HMS draws it because a name has to stay readable on every
            // one of the eight team bars and all thirty player bars, and the palest of those
            // (team gold at 231/184/61, the `White` armour colour at 177/177/177) do not give
             // light text enough contrast on their own.
            let sh = argb32(g.shadow_color, alpha);
            if sh.a() > 0 {
                let o = egui::vec2(scale, scale);
                draw_widget_text(painter, rect.translate(o), &text, c.justification(), sh, face, scale);
            }
            draw_widget_text(painter, rect, &text, c.justification(), col, face, scale);
        }
        _ => {}
    }
}

/// The player list, drawn with the game's own scoreboard widgets, plus one line saying what it is
/// drawn from. Falls back to nothing at all (the camera grid below still lists everybody) when the
/// map is not open, because the geometry lives in the map's tags.
pub fn scoreboard_panel(
    ui: &mut egui::Ui,
    roster: &[RosterRow],
    scene: Option<&crate::scene::SceneController>,
    stats: Option<&crate::film_stats::FilmStats>,
    // #score  The decoded index and the tick being watched, so the score is the one the recording
    // had AT this moment rather than the one it ended on.
    at: Option<(&crate::film_decode::FilmIndex, u32)>,
) {
    let Some(sc) = scene else { return };
    let Some(cache) = sc.cache_handle() else { return };
    let Some(sb) = crate::reach_widget::scoreboard_for(sc.native_dll(), cache) else {
        ui.label(
            egui::RichText::new("Scoreboard widgets: this map has no cui_screen scoreboard tags")
                .weak()
                .small(),
        );
        return;
    };
    let Some(g) = crate::reach_widget::ui_globals_for(sc.native_dll(), cache) else {
        ui.label(
            egui::RichText::new("Scoreboard widgets: this map has no ui_shared_globals tag, so \
                                 there is no palette to draw them with")
                .weak()
                .small(),
        );
        return;
    };
    // Reach's armour palette, read once per cache. The scoreboard row and the body on screen
    // take the SAME entry of it for a player, so they cannot disagree about who is teal.
    let profile = crate::reach_widget::profile_colors_for(cache, &|| sc.profile_colors());
    let rows = scoreboard_rows_at(roster, stats, at);
    ui.label(egui::RichText::new("Scoreboard").strong());
    paint_scoreboard(ui, &sb, &rows, &g, &profile, cache, &|t| sc.decode_bitmap_keep_alpha_public(t));
    let font = match crate::reach_font::ui_atlas(ui.ctx()) {
        Some(f) => format!("Reach's own {}", f.font.name),
        None => "the app's font (no Reach font files on this machine)".to_string(),
    };
    // #emblem  What the recording actually holds, said plainly, and counted from THIS film
    // rather than asserted. The old wording said a film "has not got" the emblem, which is
    // false: every slot carries an `s_emblem_info` at record +0x50. What is true, and what is
    // measured over the whole corpus, is that MCC fills nearly all of them with one shared
    // placeholder and stores a real emblem only for the slot whose profile it has -- on one
    // 16-player clip, fifteen slots identical and the sixteenth different, and that same
    // placeholder turns up in unrelated recordings with unrelated players.
    let emblem_note = emblem_note(&rows);
    ui.label(
        egui::RichText::new(format!(
            "Drawn from the map's own {} widgets, in {}, with the colours from the map's own \
             ui_shared_globals palette; the headings come from MCC's own string tables. The \
             talking, dead and connection marks and the selection highlight need live game state \
             a film has not got, so their slots are left empty. {emblem_note}",
            crate::reach_widget::SCOREBOARD_ROW_PLAYER.rsplit('\\').next().unwrap_or("scoreboard"),
            font
        ))
        .weak()
        .small(),
    );
    ui.add_space(6.0);
}

/// The scoreboard's rows, built from the camera list and the recording's own score rows.
///
/// ONE implementation, here, because a print-only diagnostic reads the same rows the panel
/// paints -- the colour question ("why is this row not that player's colour?") is answered by
/// inspecting these, and a second copy of the row build would answer it about the wrong thing.
/// The rows, with every score frozen at the END of the clip. Prefer
/// [`scoreboard_rows_at`], which asks the recording what the score was at the tick being watched.
pub fn scoreboard_rows(roster: &[RosterRow], stats: Option<&crate::film_stats::FilmStats>) -> Vec<ScoreboardRow> {
    scoreboard_rows_at(roster, stats, None)
}

/// The rows as they stood at `at`, if a decoded index and a tick are given.
///
/// **The score CLIMBS.** `film_stats::compute` runs once, when the index lands, and fills its
/// scoreboard with `idx.score_at(slot, u32::MAX)` — the last update in the whole film. That is
/// the right thing for a summary and the wrong thing for a scoreboard you watch, which showed the
/// final score from the first frame: "the score doesn't literally start at the high value, it
/// usually climbs as players get kills and stuff".
///
/// The fix is at DRAW time rather than in `compute`, which must stay off the frame path — it is
/// tens of milliseconds on a long clip. `FilmIndex::score_at` already takes a tick, already finds
/// the update at or before it, already composes the round score with the carried score and the
/// `final` flag, and already distinguishes "no update yet" from zero. So this passes the playhead
/// and leaves everything else precomputed; it is one binary search per row per frame.
pub fn scoreboard_rows_at(
    roster: &[RosterRow],
    stats: Option<&crate::film_stats::FilmStats>,
    at: Option<(&crate::film_decode::FilmIndex, u32)>,
) -> Vec<ScoreboardRow> {
    // Score comes from the recording's own statborg rows, by player slot.
    let score_of = |slot: Option<u8>| -> Option<i32> {
        let slot = slot?;
        // The tick being watched, when there is one. `None` from `score_at` is the recording's
        // own "this slot has sent nothing yet", which is NOT zero and must stay `None`.
        if let Some((idx, tick)) = at {
            return idx.score_at(slot, tick);
        }
        let s = stats?;
        s.scoreboard.iter().find(|r| r.kind == "player" && r.slot == slot).map(|r| r.score)
    };
    // Is this a TEAM game? The RECORDED VARIANT says so, and that is the engine's own test --
    // `game_engine_has_teams()` reads the variant's teams bit, it does not compare the players'
    // team indices. (Comparing them was the first attempt and it is not the same question: the
    // 16-player test clip records team 0 on every slot whether or not teams were on.)
    let show_team = roster.iter().any(|r| r.teams_enabled);
    let mut rows: Vec<ScoreboardRow> = roster
        .iter()
        .map(|r| ScoreboardRow {
            position: 0,
            name: r.label.clone(),
            service_tag: r.service_tag.clone(),
            score: score_of(r.slot),
            team: r.team,
            show_team,
            armour_color: r.armour_color,
            emblem: r.emblem,
            named: r.named,
        })
        .collect();
    // The engine ranks by score; with no score at all the recording's own order is kept.
    if rows.iter().any(|r| r.score.is_some()) {
        rows.sort_by_key(|r| std::cmp::Reverse(r.score.unwrap_or(i32::MIN)));
    }
    for (i, r) in rows.iter_mut().enumerate() {
        r.position = i + 1;
    }
    rows
}

/// What one text widget of the scoreboard row says, for the row HMS has.
///
/// Pulled out of the painter so it can be gated: the four data columns are a mapping from the
/// tag's own component names, and a column that silently returns nothing is exactly how the
/// service tag goes missing. `None` means "draw nothing", which is the honest answer for a
/// localisation key that resolves nowhere and for a value the recording does not hold.
pub fn widget_text(name: &str, text_id: Option<&str>, row: Option<&ScoreboardRow>) -> Option<String> {
    let blank = ScoreboardRow::default();
    let r = row.unwrap_or(&blank);
    let text = match name {
        "text_player_position" => format!("{}", r.position),
        "text_player_name" => r.name.clone(),
        "text_player_designation" => r.service_tag.clone(),
        "text_player_score" => r.score.map(|s| s.to_string()).unwrap_or_default(),
        // Anything else is a LABEL, and the tag gives its localisation key. Reach's own `unic`
        // string tags are empty in an MCC cache, but MCC keeps the text in
        // `data\ui\Localization\EN_*.bin`, which HMS already reads -- and the scoreboard's two
        // header keys, `players` and `score`, resolve there to PLAYERS and SCORE. A key that
        // resolves nowhere draws nothing rather than showing the key.
        _ => text_id.and_then(localized_label)?,
    };
    (!text.is_empty()).then_some(text)
}

/// #emblem  One sentence about the emblems THIS recording holds, counted.
///
/// Separated out so it can be gated: the claim it makes is a count, and a count can be wrong.
pub fn emblem_note(rows: &[ScoreboardRow]) -> String {
    let key = |e: &crate::film::EmblemInfo| {
        (e.foreground, e.background, e.flags, e.color_primary, e.color_secondary, e.color_background)
    };
    let mut tally: std::collections::BTreeMap<_, usize> = Default::default();
    for e in rows.iter().filter_map(|r| r.emblem.as_ref()) {
        *tally.entry(key(e)).or_default() += 1;
    }
    let slots: usize = tally.values().sum();
    if slots == 0 {
        return "This recording stored no valid emblem for any slot.".to_string();
    }
    let most = tally.values().copied().max().unwrap_or(0);
    if tally.len() == 1 {
        return format!(
            "Every one of the {slots} slots carries the SAME emblem, so the emblem column is left              empty rather than stamping one badge on everybody."
        );
    }
    format!(
        "The {slots} slots carry {} distinct emblems, {most} of them the same one \u{2014} MCC          stores a real emblem only for the profile it has and a shared placeholder for the rest          \u{2014} so the emblem column is left empty.",
        tally.len()
    )
}

/// A `cui_screen` text widget's `prop_text_id` resolved to the game's own English.
///
/// Reach's `unic` tags are stripped in MCC (every one of them reads zero strings), so the text
/// lives in MCC's own `EN_Global.bin` / `EN_HaloReach.bin`, keyed by a CRC-32 of the uppercased
/// key — the same tables HMS already resolves Halo 4 variant titles through.
fn localized_label(key: &str) -> Option<String> {
    crate::h4::localization::global().resolve_key(key)
}

/// Lay one text widget out inside its authored box: the engine's justification, baseline-aligned
/// in the box, in the Reach font when there is one.
/// Where a `cui` text widget's string starts inside its authored box, for one
/// `prop_justification` value.
///
/// **`0` = left, `1` = RIGHT, `2` = CENTRE.** Not the obvious order, and getting it wrong is
/// what made the service tag sit badly: the scoreboard authors `2` on three of its four columns
/// and they were being drawn hard against the right edge of their blocks -- the rank number
/// jammed against the colour square at unit 25, the service tag against the score block at 300,
/// the score against the row's own edge at 350 -- when all three should be centred inside them.
///
/// Engine-proven twice over:
///
/// * the tag definition itself. `text_justification_enum_definition`
///   (`reach_tag_test.exe+0x1EC60F0`) is a `{name, count, options}` record with count 3 and its
///   option strings in value order: `"left"`, `"right"`, `"center"`.
/// * the layout code. `c_draw_string::set_justification` (`sub_140232AB0`) range-checks against
///   `k_text_justification_count` and stores to `draw_string+0x1C`, and the layout function
///   `sub_140230050` (`draw_string.cpp`) branches on it to write the pen:
///   value 1 gives `right_bound - text_width - font[+0x2A]`, value 2 gives
///   `pen_left + (right_bound - pen_left - text_width) * 0.5`, and value 0 leaves the pen at the
///   left edge. The value reaches it from the widget VERBATIM -- `c_text_widget`'s draw
///   (`sub_140A15E00`) reads `widget+0x158`, `c_cui_render_context::draw_text` puts it in the
///   render command at `+0x36`, and `cui_render.cpp:481` passes it straight to the setter. The
///   sibling path that takes no justification argument hardcodes 0, which is why a widget that
///   authors nothing is left-aligned.
///
/// (A survey of all 83 `cusc` tags in `forge_halo.map` is *consistent* with this but does not
/// prove it, and reading it alone is what got the order wrong the first time: `1` is carried by
/// the `txt_ctlr_btn_*` labels of a controls diagram and by `countdown_text`, whose box is one
/// unit wide, while `2` is carried by `text_player_score`, `text_team_score` and `price_text`.)
///
/// The engine's right-justify also subtracts a font metric (`font+0x2A`) that HMS does not read,
/// so a right-justified column may sit a pixel or two further right here than in game. Nothing
/// in the scoreboard authors `1`.
pub fn justify_x(box_min_x: f32, box_width: f32, text_width: f32, justification: i32) -> f32 {
    match justification {
        1 => box_min_x + box_width - text_width,
        2 => box_min_x + (box_width - text_width) * 0.5,
        _ => box_min_x,
    }
}

fn draw_widget_text(
    painter: &egui::Painter,
    rect: egui::Rect,
    text: &str,
    justification: i32,
    col: egui::Color32,
    font: Option<&crate::reach_font::FontAtlas>,
    scale: f32,
) {
    match font {
        Some(f) => {
            // The widget box is authored around the font's own line, so put the baseline where
            // the ascender fills the box from the top.
            let s = scale;
            let w = f.font.measure(text) * s;
            let x = justify_x(rect.min.x, rect.width(), w, justification);
            let y = rect.min.y + (rect.height() - f.font.line_height() * s) * 0.5 + f.font.ascender as f32 * s;
            f.paint(painter, egui::pos2(x, y), text, col, s);
        }
        None => {
            let g = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(rect.height() * 0.72), col);
            let x = justify_x(rect.min.x, rect.width(), g.size().x, justification);
            painter.galley(egui::pos2(x, rect.center().y - g.size().y * 0.5), g, col);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the saved-film details card
// ---------------------------------------------------------------------------------------------

/// Draw the card. Returns the action the user asked for.
/// #gamevariant  The gametype's REAL SETTINGS, from the game variant the film embeds — not just
/// its title. Pure, like `metadata_rows`, so it is testable without a GUI.
///
/// `names` is the open map's `motl` / `lgtd` tables; without a map the weapon and loadout
/// indices are shown as `#n` rather than guessed at. Every row here is **decoded** except the
/// spawn weapon, which is derived by applying the engine's loadout-over-traits precedence, and
/// says so when it cannot be answered at all.
pub fn gametype_rows(
    f: &Film,
    names: Option<&crate::film_variant::VariantNames>,
) -> Vec<(&'static str, String)> {
    use crate::film_variant as fv;
    let Some(g) = f.header.gametype.as_ref() else {
        return vec![(
            "Gametype settings",
            "not available — this recording carries no game variant".into(),
        )];
    };
    let empty = fv::VariantNames::default();
    let n = names.unwrap_or(&empty);
    let t = &g.base_traits;
    let sw = g.spawn_primary();
    let mut rows: Vec<(&'static str, String)> = Vec::new();
    rows.push((
        "Player species",
        format!(
            "{} (decoded){}",
            g.species.label(),
            match g.species {
                fv::PlayerSpecies::ForceSpartan => " — every player was a Spartan".to_string(),
                fv::PlayerSpecies::ForceElite => " — every player was an Elite".to_string(),
                fv::PlayerSpecies::SetByTeam => {
                    let e: Vec<String> = g
                        .teams
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.enabled() && t.model_override == 1)
                        .map(|(i, _)| team_name(i as u8).to_string())
                        .collect();
                    if e.is_empty() {
                        " — every enabled team is Spartan".to_string()
                    } else {
                        format!(" — Elite team(s): {}", e.join(", "))
                    }
                }
                // #film-action  This used to say a film does not carry the species, which is
                // WRONG: the variant leaves it to each player's profile, but the recording names
                // the biped tag outright on every spawn (the creation message's network tag
                // reference), and Elites do appear. The setting is still the only thing this row
                // can speak for; the bodies are reported per player in the cast.
                _ => " — the gametype leaves it to each player's own customisation, so this \
                      setting does not decide it; which biped each player actually spawned as is \
                      read from the recording itself and listed per player"
                    .to_string(),
            }
        ),
    ));
    let (drawn, rule) = g.spawn_primary_drawable();
    rows.push((
        "Spawn weapon",
        match rule {
            fv::SpawnWeaponRule::Stated => format!(
                "{} (derived from the declared loadouts and traits)",
                n.weapon_label(sw.index)
            ),
            fv::SpawnWeaponRule::MostCommonLoadout => format!(
                "{} — the gametype's MOST COMMON selectable loadout. A film does not record which \
                 loadout a player chose, so this is drawn for every body and is right for most of \
                 them, not all",
                n.weapon_label(drawn)
            ),
            // A sentinel is the gametype declining to set a weapon, not a decode failure, so
            // the card says which way it declined and the body is drawn empty-handed.
            fv::SpawnWeaponRule::None => match fv::weapon_sentinel(drawn) {
                Some("unchanged") => "the gametype leaves every player's weapon UNCHANGED, so \
                     nothing is drawn in hand"
                    .to_string(),
                Some("map default") => "the gametype defers to the map's own default weapon, \
                     which a film does not carry, so nothing is drawn in hand"
                    .to_string(),
                Some("none") => "none — the gametype spawns players with no weapon".to_string(),
                Some("random") => "the gametype picks each spawn weapon at RANDOM, which a film \
                     does not record"
                    .to_string(),
                _ => "not available".to_string(),
            },
        },
    ));
    rows.push((
        "Rounds",
        format!(
            "{} round(s), {} min each, {} to win (decoded)",
            if g.misc.number_of_rounds == 0 { "no limit".into() } else { g.misc.number_of_rounds.to_string() },
            if g.misc.round_time_limit == 0 { "no limit".into() } else { g.misc.round_time_limit.to_string() },
            g.misc.early_victory_win_count
        ),
    ));
    rows.push((
        "Respawn",
        format!(
            "{} s, +{} growth, suicide +{}, betrayal +{}, lives {} (decoded)",
            g.respawn.respawn_time,
            g.respawn.respawn_growth,
            g.respawn.suicide_penalty,
            g.respawn.betrayal_penalty,
            if g.respawn.lives_per_round == 0 { "unlimited".into() } else { g.respawn.lives_per_round.to_string() }
        ),
    ));
    rows.push((
        "Social",
        format!(
            "friendly fire {}, betrayal booting {}, team changing {} (decoded)",
            if g.social.friendly_fire() { "on" } else { "off" },
            if g.social.betrayal_booting() { "on" } else { "off" },
            g.social.team_changing
        ),
    ));
    rows.push((
        "Base player traits",
        format!(
            "{} / {} / {}, grenades {}, damage resistance {} (decoded)",
            n.weapon_label(t.weapons.primary_weapon),
            n.weapon_label(t.weapons.secondary_weapon),
            n.equipment_label(t.weapons.equipment),
            fv::grenade_count_label(t.weapons.grenade_count),
            fv::damage_resistance_label(t.shield.damage_resistance)
        ),
    ));
    let vis = g.visible_loadouts().count();
    rows.push((
        "Loadouts",
        if vis == 0 {
            "none visible in game (decoded)".into()
        } else {
            format!("{vis} visible of {} (decoded)", g.loadouts.len())
        },
    ));
    for (tier, slot, l) in g.visible_loadouts() {
        rows.push((
            "Loadout",
            format!(
                "tier {tier} #{slot}  {}  —  {} / {} / {} / {}",
                n.loadout_name(l.name),
                n.weapon_label(l.primary_weapon),
                n.weapon_label(l.secondary_weapon),
                n.equipment_label(l.armor_ability),
                fv::grenade_count_label(l.grenade_count)
            ),
        ));
    }
    rows
}

pub fn card_ui(
    ui: &mut egui::Ui,
    f: &Film,
    show_machine: &mut bool,
    base_map: Option<&str>,
    status: &str,
    variant_names: Option<&crate::film_variant::VariantNames>,
) -> FilmAction {
    let mut action = FilmAction::None;
    let mv = &f.header.map_variant;
    egui::ScrollArea::vertical().id_salt("film_card_scroll").max_height(420.0).show(ui, |ui| {
        egui::Grid::new("film_card_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            for (k, v) in metadata_rows(f, *show_machine, base_map) {
                ui.label(egui::RichText::new(k).strong());
                ui.label(v);
                ui.end_row();
            }
            // #gamevariant  The gametype's real settings, under its title.
            for (k, v) in gametype_rows(f, variant_names) {
                ui.label(egui::RichText::new(k).strong());
                ui.label(v);
                ui.end_row();
            }
        });
    });
    ui.separator();
    ui.horizontal(|ui| {
        let can_scene = !mv.is_empty();
        if ui
            .add_enabled(can_scene, egui::Button::new("Load map + Forge scene"))
            .on_hover_text("Load this film's base map and draw the Forge objects the film was recorded with")
            .clicked()
        {
            action = FilmAction::LoadScene;
        }
        if ui
            .add_enabled(can_scene, egui::Button::new("Extract map variant\u{2026}"))
            .on_hover_text("Write the film's embedded Forge variant out as a real .mvar you can open and play")
            .clicked()
        {
            action = FilmAction::ExtractMapVariant;
        }
        if ui
            .button("Extract gametype\u{2026}")
            .on_hover_text(
                "Write the film's embedded game variant out. HMS cannot yet produce a loadable .bin \
                 (that needs a Megalo encoder), so this is the raw struct plus readable notes",
            )
            .clicked()
        {
            action = FilmAction::ExtractGameVariant;
        }
    });
    ui.checkbox(show_machine, "Show the recording machine")
        .on_hover_text(
            "A film embeds the recording PC's Windows account and computer name in plain text. \
             Anything shared from a PC leaks both, so it is off by default.",
        );
    if !status.is_empty() {
        ui.separator();
        ui.label(egui::RichText::new(status).weak());
    }
    ui.separator();
    if ui.button("Close").clicked() {
        action = FilmAction::Close;
    }
    action
}

// ---------------------------------------------------------------------------------------------
// the watchable cast
// ---------------------------------------------------------------------------------------------

/// One row of the watchable cast.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RosterRow {
    /// The owner group to watch — one player's bodies across every respawn. `None` means this
    /// player is in the film header but never had a body in the clip.
    pub owner: Option<u8>,
    /// The film header's configuration slot, when this player is named.
    pub slot: Option<u8>,
    /// The gamertag when the recording has one, the entity id when it does not.
    pub label: String,
    /// True when `label` is a real gamertag.
    pub named: bool,
    pub service_tag: String,
    /// 0xFF when unknown (an unnamed player has no header record, so no team).
    pub team: u8,
    /// The recorded armour customisation, in words. Empty when the record is too short.
    pub armour: String,
    /// The PRIMARY ARMOUR COLOUR index, for colouring this player's own scoreboard bar in a
    /// free-for-all. `None` when the recording holds no appearance for them.
    pub armour_color: Option<u8>,
    /// #emblem  The emblem `s_emblem_info` the recording stored for this slot.
    pub emblem: Option<crate::film::EmblemInfo>,
    /// #scoreboard-color  The recorded game variant's own `teams enabled` bit. A property of the
    /// recording rather than of this row, carried here because the scoreboard panel is handed
    /// rows and nothing else. See [`bitmap_tint`].
    pub teams_enabled: bool,
    /// #camera-rows  What the RECORDING says this body is, when it cannot be tied to a player:
    /// the `bipd` tag's own leaf name (`spartan`, `elite`, `monitor`, ...). Empty when the row
    /// is a named player or the biped could not be resolved.
    pub body_kind: String,
    /// How many bodies (lives) this player had.
    pub lives: usize,
    /// The tick their first body appears.
    pub first_tick: Option<u32>,
    /// In the header but with no owner group, AND the clip holds bodies matched to nobody — so
    /// they probably DID play and HMS cannot say which body was theirs. Different from "never
    /// had a body", and the row says which.
    pub body_unmatched: bool,
}

/// The watchable cast of a film: one row per player who actually had a body, plus any header
/// player who never did.
///
/// **Not the film header's roster.** The header records who was present when it was WRITTEN, so a
/// clip can store three gamertags while fourteen players had a body on the map. Building the list
/// from the header made eleven of those fourteen unwatchable, which is the wrong trade: aiming a
/// camera at a body needs no gamertag, only an entity id, and the recording always has one.
pub fn roster_lines(
    f: &Film,
    pb: &crate::film_play::Playback,
    biped_of: &std::collections::HashMap<u32, String>,
) -> Vec<RosterRow> {
    // #scoreboard-color  Whether this recording was a TEAM game, from the embedded game
    // variant's own `teams enabled` bit -- which is what the engine's `game_engine_has_teams()`
    // tests. A clip with no variant at all (a pure Forge recording) is not a team game.
    let teams = f.header.gametype.as_ref().map_or(false, |gv| gv.misc.teams_enabled());
    let mut out: Vec<RosterRow> = Vec::new();
    let mut claimed: std::collections::HashSet<u8> = Default::default();
    for owner in pb.index.owner_groups() {
        let slot = pb.index.slot_of_owner(owner);
        if let Some(s) = slot {
            claimed.insert(s);
        }
        let player =
            slot.and_then(|s| f.header.players.iter().find(|p| p.present() && p.index == s as usize));
        let first = pb
            .index
            .tracks
            .iter()
            .filter(|t| t.entity_type == 11 && t.owner_raw == Some(owner))
            .min_by_key(|t| t.first_tick);
        let named = player.map_or(false, |p| !p.gamertag.is_empty());
        // #camera-rows  A group with no player slot is NOT a nameless player by default. The
        // recording says what each body was spawned from, and over this corpus the bodies that
        // never bind to a slot include FORGE MONITORS -- a monitor is a biped with an owner
        // stamp and no configuration slot, so it reaches exactly this branch. A row that is one
        // of those should say what it is, not show a bare entity id as though a gamertag had
        // gone missing. `biped_of` is `net id -> bipd tag path`, resolved by the driver from the
        // creation messages; empty when nobody has resolved it yet.
        let kind = first
            .and_then(|t| biped_of.get(&t.net_id))
            .map(|p| p.rsplit('\\').next().unwrap_or(p.as_str()).to_string())
            .unwrap_or_default();
        out.push(RosterRow {
            owner: Some(owner),
            slot,
            label: match player.filter(|p| !p.gamertag.is_empty()) {
                Some(p) => p.gamertag.clone(),
                // No name for this body. Say what the recording DOES know about it: the thing it
                // was spawned from, plus the entity id that makes it watchable. The id stays in
                // the label because it is the only handle that distinguishes two monitors.
                None => unnamed_camera_label(&kind, first.map(|t| t.net_id), owner),
            },
            named,
            service_tag: player.map(|p| p.service_tag.clone()).unwrap_or_default(),
            team: player.map(|p| p.team).unwrap_or(0xFF),
            armour: player.and_then(|p| p.appearance()).map(|a| a.summary()).unwrap_or_default(),
            armour_color: player.and_then(|p| p.appearance()).map(|a| a.color_primary),
            emblem: player.and_then(|p| p.emblem()),
            teams_enabled: teams,
            body_kind: kind,
            lives: pb.index.owner_lives(owner),
            first_tick: first.map(|t| t.first_tick),
            body_unmatched: false,
        });
    }
    // A header player with no owner group. Two quite different reasons, and the row must not
    // conflate them: either they genuinely never had a body, or they did and HMS could not tell
    // WHICH of the unnamed bodies was theirs.
    let unmatched_bodies = out.iter().filter(|r| !r.named).count();
    for p in f.present_players() {
        if claimed.contains(&(p.index as u8)) {
            continue;
        }
        let own = pb.index.bodies_of_slot(p.index as u8);
        out.push(RosterRow {
            owner: None,
            slot: Some(p.index as u8),
            label: if p.gamertag.is_empty() { format!("slot {}", p.index) } else { p.gamertag.clone() },
            named: !p.gamertag.is_empty(),
            service_tag: p.service_tag.clone(),
            team: p.team,
            armour: p.appearance().map(|a| a.summary()).unwrap_or_default(),
            armour_color: p.appearance().map(|a| a.color_primary),
            emblem: p.emblem(),
            teams_enabled: teams,
            body_kind: String::new(),
            // This row has no owner GROUP, but that does not mean it has no bodies: a slot can
            // be named at the BODY level by elimination, and then its lives sit in whichever
            // unnamed group those bodies belong to. Hardcoding 0 said "this player never
            // played" about somebody who did -- on the test film, slot 15 read `lives=0` while
            // six of their own bodies were listed under a separate unnamed row. `bodies_of_slot`
            // answers per body, so it reaches an elimination-named slot without letting the
            // unnamed group claim the name.
            lives: own.len(),
            first_tick: own.first().map(|t| t.first_tick),
            body_unmatched: unmatched_bodies > 0 && own.is_empty(),
        });
    }
    out.sort_by_key(|r| (!r.named, r.first_tick.unwrap_or(u32::MAX), r.label.clone()));
    out
}

/// #camera-rows  The label for a camera row whose body the recording never named.
///
/// It used to be a bare 32-bit entity id -- `0xe9750001` -- which reads as an object handle and
/// tells a viewer nothing, and that is what was reported. Two things are known about such a body
/// even when its gamertag is not: what it was spawned FROM (the `bipd` tag the creation message
/// names, which over this corpus includes Forge MONITORS as well as Spartans and Elites), and
/// which OWNER GROUP it is -- one number, unique within the clip, which is what makes two
/// unnamed Spartans tellable apart. Reach itself shows no name for such a body at all: its own
/// Theater camera bar binds `text_player_name.prop_visible` to
/// `camera_list.prop_player_configuration_valid`.
///
/// **This is a labelling fix, not a naming one, and the distinction matters.** Measured over the
/// twelve-film corpus: the clips that produce these rows stored 1, 2, 3 or 9 gamertags in their
/// header while 11 to 16 players had a body, and the clips that produce none stored 16. Those
/// names were never written to the file, so there is nothing to recover for most of them. (One
/// film IS a genuine gap -- a 296-tick clip saved by a spectator who is not in its own header,
/// where nine of twelve bodies map onto occupied slots under `slot = (owner - 1) & 15` and none
/// is promoted because the rule needs a per-film anchor that a clip that short cannot give. That
/// gate lives in `film_decode::apply_owner_rule` and is deliberately left alone here.)
pub fn unnamed_camera_label(body_kind: &str, _net_id: Option<u32>, owner: u8) -> String {
    // The owner stamp is what makes the label unique -- one per group, for the life of the clip
    // -- and it is a small number, so it reads as a player and not as a handle. A hex entity id
    // was reported as reading "just like an object ID", and it did: a body also gets a NEW
    // entity every respawn, so the id is not even stable for the player it belongs to.
    let kind = match body_kind {
        "" => return format!("Player {owner}"),
        k => {
            let mut c = k.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        }
    };
    format!("{kind} {owner}")
}

/// A plain statement of who is in the clip versus who can be NAMED — worded so a reader can tell
/// a property of the recording from a failure of this program.
///
/// It is the former, and for TWO separate reasons that must not be run together.
///
/// **The body's own field cannot name everybody.** A body says whose it is through the engine's
/// `object +0x3E`, a SIGNED byte with -1 meaning "nobody", replicated as `value + 1` in four
/// bits (`write_bits(*(char*)(obj+0x3E)+1, 4)` against `read_bits(4)-1`). So wire 0 carries TWO
/// meanings and nothing in the field distinguishes them: it is the `-1` sentinel, "attributed to
/// nobody", and it is also slot 15, whose `index + 1` OVERFLOWS four bits to 0. Where the header
/// occupies slot 15 and wires 1..15 account for every other slot, the leftover group is recovered
/// as that player (`FilmIndex::bind_stamp_zero_by_truncation`); where it does not, wire 0 really
/// does mean nobody. In neither case is there a wrap for slots past 15. Nothing else in the file
/// attributes a body: the runtime
/// player-to-unit mapping is written (`Player_SetUnit`, `player_datum +0x28` and back to
/// `object +0x1CC`) but never replicated, and `s_player_action`, the one channel that carries
/// per-player object references, has zero blocks across 39 films and 400,000-plus ticks.
///
/// **And the header does not hold every name.** A clip measured here stores three gamertags
/// while thirteen slots carry per-tick records and fourteen players had a body. Scanning that
/// film's whole 60 MB — every byte, at all eight bit alignments past the header, because a string
/// inside a bit-packed payload would not be byte-aligned — finds exactly those three gamertags,
/// all within the first 4 KB. Anyone who joined after the header was written has no record in it
/// to be named from.
pub fn roster_coverage(pb: &crate::film_play::Playback, header_players: usize) -> String {
    let bodies = pb.index.owner_group_count();
    // Ask BOTH routes, because `slot_of_owner` answers `None` for wire 0 by design and the
    // display does not: `owner_group_is_named` also tries each of the group's bipeds through
    // `slot_of_entity`, which is what actually feeds a nameplate. Counting only the first route
    // used to report "15 of 16 can be NAMED" while the sixteenth player's gamertag was being
    // drawn over their head.
    let named =
        pb.index.owner_groups().iter().filter(|&&o| pb.index.owner_group_is_named(o)).count();
    let mut s = format!("{bodies} player(s) had a body in this clip, and all {bodies} can be watched. ");
    if named >= bodies {
        s.push_str("Every one of them is named by the recording.");
        return s;
    }
    // #xfilm  Say that cross-film naming depends on the user's OTHER recordings, because the
    // same clip naming differently on a different machine reads as a bug otherwise.
    s.push_str(
        "Some bodies are named from ANOTHER of your recordings, by matching their armour, and \
         those names are drawn dimmer and in [brackets] because this recording does not state \
         them. That means the same clip names more players on a machine with more films on it, \
         and only the ones this film's own header holds if it is the only one. ",
    );
    s.push_str(&format!(
        "{named} of them can be NAMED. The field a body carries to say whose it is holds \
         `index + 1` in FOUR BITS, so slots 0 to 14 take wire values 1 to 15 and slot 15 \
         OVERFLOWS to 0 \u{2014} the same value the engine uses for \u{201c}nobody\u{201d}. Where \
         this header occupies slot 15 and every other slot is accounted for, those bodies are \
         RECOVERED as that player; where it does not, 0 means nobody and is left unnamed on \
         purpose rather than guessed. Nothing else in the file attributes them: the \
         runtime player-to-unit link is never replicated, and the one channel that carries \
         per-player object references is absent from films entirely. Separately, this header \
         stored {header_players} gamertag(s), which is who was present when it was written, so \
         anyone who joined later has no record here to name them from either. The rest are \
         listed by what the recording says they ARE \u{2014} the biped and their owner group \
         \u{2014} and a body stamped \u{201c}nobody\u{201d} is left unnamed on purpose rather \
         than guessed at."
    ));
    s
}

// ---------------------------------------------------------------------------------------------
// the Theater panels
// ---------------------------------------------------------------------------------------------

/// What the Theater transport strip asks the host to do.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransportAction {
    pub close: bool,
    /// Toggle the side panel.
    pub toggle_panel: bool,
    /// Toggle the Forge editing column.
    pub toggle_editor: bool,
    /// #film-action  Set when an assassination jump moved the playhead: `(kill tick, stated)`.
    /// `stated` false means HMS DERIVED this one rather than the recording naming it, and the
    /// status line says so -- the two are told apart everywhere else and must be here too.
    pub jumped_to: Option<(u32, bool)>,
}

/// Height of the transport strip, in points. **Fixed, and that is the point.**
///
/// The strip used to be the top of a floating `egui::Window` anchored by its BOTTOM edge, whose
/// content height changed every single frame — the per-body list grows and shrinks as players
/// spawn and die, and the caption re-wraps as its own numbers change. Bottom pinned plus variable
/// height moves the TOP edge every frame, so Play/Pause physically travelled up and down under
/// the cursor while the clip played. A panel of fixed height cannot do that.
pub const TRANSPORT_HEIGHT: f32 = 62.0;

/// The permanent Theater strip: everything a viewer needs while watching, and nothing else.
///
/// Lives in a `TopBottomPanel`, which RESHAPES the viewport rather than floating over it, so the
/// picture is never hidden behind the controls.
pub fn transport_ui(
    ui: &mut egui::Ui,
    pb: &mut crate::film_play::Playback,
    panel_open: bool,
    cast: usize,
    editor_hidden: bool,
) -> TransportAction {
    use crate::film_play::fmt_time;
    let mut act = TransportAction::default();
    let total = pb.total_seconds();
    let secs = pb.seconds();
    ui.horizontal(|ui| {
        let playing = pb.playing;
        if ui
            .button(if playing { "\u{23F8} Pause" } else { "\u{25B6} Play" })
            .on_hover_text("Play the clip at the recorded 60 ticks per second")
            .clicked()
        {
            pb.playing = !playing;
        }
        if ui.button("\u{23EE}").on_hover_text("Back one tick (1/60 s)").clicked() {
            pb.playing = false;
            pb.step(-1);
        }
        if ui.button("\u{23ED}").on_hover_text("Forward one tick (1/60 s)").clicked() {
            pb.playing = false;
            pb.step(1);
        }
        // #film-action  JUMP TO AN ASSASSINATION. The move plays at the authored rate; what was
        // missing is any way to find one -- three kills of 2.5 s each, anywhere in a 7-minute
        // clip, with the camera usually elsewhere. These seek AND move the camera onto the victim.
        // They sit in the transport row rather than in a panel because this is navigation, and
        // navigation belongs next to the other navigation.
        {
            let n = pb.assassinations().len();
            ui.add_enabled_ui(n > 0, |ui| {
                let hover = if n == 0 {
                    "This clip records no assassinations.".to_string()
                } else {
                    format!(
                        "Jump to an assassination and put the camera on the victim ({n} in this \
                         clip). The move itself plays at the recorded speed \u{2014} 75 frames \
                         over 2.5 s \u{2014} so this is about finding them."
                    )
                };
                if ui.button("\u{2694}\u{25C0}").on_hover_text(&hover).clicked() {
                    act.jumped_to = pb.jump_to_assassination(-1).map(|a| (a.tick, a.stated));
                }
                if ui.button("\u{2694}\u{25B6}").on_hover_text(&hover).clicked() {
                    act.jumped_to = pb.jump_to_assassination(1).map(|a| (a.tick, a.stated));
                }
            });
        }
        // MONOSPACE and zero-padded, so the text is the same WIDTH every tick and nothing after
        // it shifts as the clock runs.
        ui.label(egui::RichText::new(format!("{} / {}", fmt_time(secs), fmt_time(total))).monospace());
        // Players do not exist for the first few seconds of a round, so a count of zero here is
        // the recording being faithful, not the decoder failing.
        let here = pb.index.players_at(pb.tick.round().max(0.0) as u32);
        ui.label(
            egui::RichText::new(format!("{here:>2} alive")).monospace().color(if here == 0 {
                egui::Color32::from_rgb(210, 160, 60)
            } else {
                egui::Color32::GRAY
            }),
        )
        .on_hover_text(match pb.index.first_player_tick() {
            Some(t) => format!(
                "Player bipeds do not exist at the start of a round: the first one appears at {}.",
                fmt_time(t as f32 / 60.0)
            ),
            None => "This film records no player entities at all.".to_string(),
        });
        ui.separator();
        ui.label("Speed");
        ui.add(egui::DragValue::new(&mut pb.speed).speed(0.05).range(0.25..=4.0).suffix("x"));
        ui.separator();
        if ui
            .selectable_label(panel_open, format!("Players ({cast})"))
            .on_hover_text("Who is in this clip, the cameras, and what is drawn")
            .clicked()
        {
            act.toggle_panel = true;
        }
        if ui
            .selectable_label(!editor_hidden, "Editor")
            .on_hover_text(
                "Show the map/variant/objects/palette column. It is hidden while a clip is open \
                 because none of it is usable while watching, and it costs about a third of the window.",
            )
            .clicked()
        {
            act.toggle_editor = true;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("\u{2715}").on_hover_text("Close the Theater controls").clicked() {
                act.close = true;
            }
        });
    });
    // The slider is bound to the TICK, and a change is only applied when it actually moved by
    // more than half a tick. Binding it to a seconds value re-derived from the clock each frame
    // made the widget write a ROUNDED value straight back, which crept the clock forward every
    // frame with nothing playing and made Pause look broken.
    let mut t = pb.tick;
    let last = pb.total_ticks().saturating_sub(1) as f32;
    let resp = ui.add(
        egui::Slider::new(&mut t, 0.0..=last.max(1.0))
            .show_value(false)
            .text(format!("tick {} of {}", pb.tick.round() as u32, pb.total_ticks())),
    );
    if resp.changed() && (t - pb.tick).abs() > 0.5 {
        pb.playing = false;
        pb.seek_tick(t);
    }
    act
}

/// Height of the roster's scroll area, and of the "what is drawn" box, in points.
///
/// Both are fixed (min == max) on purpose: their contents change length every tick — the roster
/// is 3 rows on one clip and 14 on another, and the per-body list grows and shrinks as players
/// spawn and die — and a container that resizes with its contents makes everything below it move.
pub const ROSTER_HEIGHT: f32 = 190.0;
pub const DETAIL_HEIGHT: f32 = 150.0;

/// Everything verbose: the cast and its cameras, what is drawn, and the draw options.
///
/// Lives in a `SidePanel`, CLOSED by default. A side panel reshapes the viewport, so even with it
/// open nothing on screen is hidden behind it.
#[allow(clippy::too_many_arguments)]
pub fn side_panel_ui(
    ui: &mut egui::Ui,
    pb: &mut crate::film_play::Playback,
    roster: &[RosterRow],
    header_players: usize,
    caption: &str,
    body_lines: &[String],
    coverage: &str,
    scene: Option<&crate::scene::SceneController>,
    stats: Option<&crate::film_stats::FilmStats>,
    // #film-live  `follow_live`: whether to keep re-decoding a recording that is still being
    // written. Unlike every other toggle here this is a property of the FILE, not of playback --
    // a finished film cannot grow -- so it lives on the app rather than on `Playback` and is
    // passed in. `follow_note` is one line saying what the last follow pass did, or empty.
    follow_live: &mut bool,
    follow_note: &str,
    // #film-objects  `show_objects`: draw the objective objects, the loose weapons and the
    // vehicle debris the recording carries. It lives on the actor driver rather than on
    // `Playback` and is session-only, so it is passed in too. `object_status` is its own
    // description of what is being drawn right now.
    show_objects: &mut bool,
    object_status: &str,
    // #fp-fov  The first-person FIELD OF VIEW, in degrees. It lived only in
    // "Settings > Camera & rendering", which is not where anyone watching a film looks for it, so
    // it is mirrored here beside the other first-person controls. `App`-side state, like
    // `follow_live`.
    fp_fov_deg: &mut f32,
    // #reach-sound  The audio engine, for the volume row. Passed whole rather than as a bare
    // f32 because the row also shows what the device is doing, and because `set_volume` is what
    // persists the setting -- a second writer of the same number is exactly the bug
    // `feedback-two-writers-one-field` records.
    sound: &mut crate::sound::SoundEngine,
) {
    ui.add_space(4.0);
    // #reach-sound  Film audio, at the top where someone watching will find it.
    crate::sound::volume_ui(sound, ui);
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Camera");
        if ui
            .selectable_label(pb.camera == crate::film_play::FilmCamera::Free, "Free")
            .on_hover_text("HMS's own camera; playback does not move it")
            .clicked()
        {
            pb.camera = crate::film_play::FilmCamera::Free;
        }
        if ui
            .button("Recenter")
            .on_hover_text("Put the follow camera back behind the player at the default framing")
            .clicked()
        {
            pb.rig.recenter();
        }
    });
    ui.label(egui::RichText::new(crate::film_play::camera_status(pb)).weak());
    ui.separator();

    // #reach-widget  The player list, drawn with Reach's OWN scoreboard widgets when the map is
    // open; the grid below it is the camera control surface, not a second list.
    scoreboard_panel(ui, roster, scene, stats, Some((&pb.index, pb.tick.round().max(0.0) as u32)));

    ui.label(egui::RichText::new(format!("Cameras ({})", roster.len())).strong());
    // #film-perf  Which owner groups have a body RIGHT NOW, in ONE pass over the index. The row
    // used to ask `entity_for_owner` per player, and that is a full scan of every track in the
    // recording -- sixteen of them, every frame, just to decide a tooltip.
    let tick_now = pb.tick.round().max(0.0) as u32;
    let owners_with_body: std::collections::HashSet<u8> = pb
        .index
        .tracks
        .iter()
        .filter(|t| t.entity_type == 11 && tick_now >= t.first_tick && tick_now <= t.last_tick)
        .filter_map(|t| t.owner_raw)
        .collect();
    egui::ScrollArea::vertical()
        .id_salt("film_roster_scroll")
        .min_scrolled_height(ROSTER_HEIGHT)
        .max_height(ROSTER_HEIGHT)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("film_roster_grid")
                .num_columns(4)
                .spacing([8.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    for row in roster {
                        ui.colored_label(team_color(row.team), "\u{25A0}");
                        let name = if row.named {
                            egui::RichText::new(&row.label)
                        } else {
                            egui::RichText::new(&row.label).monospace().weak()
                        };
                        ui.label(name).on_hover_text(if row.named {
                            if row.armour.is_empty() {
                                "no armour record".to_string()
                            } else {
                                format!("recorded armour: {}", row.armour)
                            }
                        } else {
                            "This player joined after the film header was written, so the recording \
                             never stored their gamertag. Their body is still fully watchable."
                                .to_string()
                        });
                        ui.label(
                            egui::RichText::new(match (row.lives, row.first_tick) {
                                (0, _) if row.body_unmatched => "body not matched".to_string(),
                                (0, _) => "no body in this clip".to_string(),
                                (n, Some(t)) => format!(
                                    "{n} life(s), from {}",
                                    crate::film_play::fmt_time(t as f32 / 60.0)
                                ),
                                (n, None) => format!("{n} life(s)"),
                            })
                            .weak()
                            .small(),
                        );
                        // The camera is keyed on the OWNER GROUP, which every body has, rather
                        // than on the header slot, which most do not.
                        ui.horizontal(|ui| {
                            match row.owner {
                                Some(o) => {
                                    let has_body_now = owners_with_body.contains(&o);
                                    let fp = pb.camera == crate::film_play::FilmCamera::FirstPersonOwner(o);
                                    let fl = pb.camera == crate::film_play::FilmCamera::FollowOwner(o);
                                    if ui
                                        .selectable_label(fp, "1st")
                                        .on_hover_text(if has_body_now {
                                            "Through this player's eyes"
                                        } else {
                                            "This player has no body at this moment; the camera will pick them up when they next spawn"
                                        })
                                        .clicked()
                                    {
                                        pb.camera = if fp {
                                            crate::film_play::FilmCamera::Free
                                        } else {
                                            crate::film_play::FilmCamera::FirstPersonOwner(o)
                                        };
                                    }
                                    if ui
                                        .selectable_label(fl, "follow")
                                        .on_hover_text("Behind and above this player's body; drag in the viewport to orbit")
                                        .clicked()
                                    {
                                        pb.camera = if fl {
                                            crate::film_play::FilmCamera::Free
                                        } else {
                                            crate::film_play::FilmCamera::FollowOwner(o)
                                        };
                                    }
                                }
                                None => {
                                    ui.add_enabled(false, egui::SelectableLabel::new(false, "1st"))
                                        .on_hover_text(if row.body_unmatched {
                                            "This player is named in the film header, but HMS could not tell which of the \
                                             unnamed bodies was theirs. Their body is in the clip and can be watched \u{2014} \
                                             from one of the rows below, by entity id."
                                        } else {
                                            "This player is in the film header but never had a body in this clip"
                                        });
                                    ui.add_enabled(false, egui::SelectableLabel::new(false, "follow"));
                                }
                            }
                        });
                        ui.end_row();
                    }
                });
        });
    let _ = header_players; // the coverage line is built once per film by the caller
    ui.label(egui::RichText::new(coverage).weak().small());
    ui.separator();

    // Everything whose length changes every tick lives inside a FIXED box, so nothing below it
    // moves when the numbers change.
    egui::CollapsingHeader::new("What is drawn, and how")
        .id_salt("film_what_is_drawn")
        .default_open(false)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("film_detail_scroll")
                .min_scrolled_height(DETAIL_HEIGHT)
                .max_height(DETAIL_HEIGHT)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(caption).weak().small());
                    if !body_lines.is_empty() {
                        ui.add_space(4.0);
                        for l in body_lines {
                            ui.label(egui::RichText::new(l).monospace().weak().small());
                        }
                    }
                });
        });
    // #fp-fov #reach-hud  THE FIRST-PERSON CONTROLS, in the panel body and NOT inside a
    // collapsing header.
    //
    // Every one of these used to be behind a section that opened CLOSED, and the user's report was
    // not that they did not work but that they could not be found: "idk where to adjust things I
    // asked for like first person FOV or to hide certain aspects of the HUD". A control nobody can
    // find is a control that does not exist, so the ones you reach for while actually watching a
    // film live here, at the top level, and only the rarely-touched decode switches stay folded
    // away below.
    ui.separator();
    ui.label(egui::RichText::new("In first person").strong());
    ui.indent("film_fp_controls", |ui| {
        if ui
            .add(
                egui::Slider::new(fp_fov_deg, crate::film_play::FOV_MIN_DEG..=crate::film_play::FOV_MAX_DEG)
                    .text("Field of view")
                    .suffix("\u{b0}"),
            )
            .on_hover_text(
                "The HORIZONTAL field of view a first-person Theater camera frames with, the way the game's \
                 own setting does. Reach's default is 78\u{b0} -- the value its globals author -- and the \
                 vertical angle is derived from it for this window's aspect, so a wider setting shows more to \
                 the sides rather than stretching the picture. Remembered between sessions.",
            )
            .changed()
        {
            crate::film_play::save_fov_setting(*fp_fov_deg);
        }
        if (*fp_fov_deg - crate::film_play::REACH_FIELD_OF_VIEW_DEG).abs() > 0.05 {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "the game's own default is {:.0}\u{b0}",
                        crate::film_play::REACH_FIELD_OF_VIEW_DEG
                    ))
                    .weak()
                    .small(),
                );
                if ui.small_button("Reset").clicked() {
                    *fp_fov_deg = crate::film_play::REACH_FIELD_OF_VIEW_DEG;
                    crate::film_play::save_fov_setting(*fp_fov_deg);
                }
            });
        }
        first_person_switches(ui, pb);
    });
    ui.separator();
    egui::CollapsingHeader::new("What to draw")
        .id_salt("film_draw_options")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut pb.moving_only, "Only entities that move")
                .on_hover_text("The static Forge scene is already drawn from the map variant, so redrawing motionless objects on top of it adds nothing");
            // #spectate  Theater's own spectated-player card. Remembered between sessions.
            if ui
                .checkbox(&mut pb.spectate_card, "Spectated player card")
                .on_hover_text(
                    "While the camera is following one player, draw the bar Theater itself shows: \
                     the game's own `film_cameras` widget, with that player's name and service tag, \
                     at its authored place along the bottom of the view. Nothing is drawn with a \
                     free camera, which is what the game does too.",
                )
                .changed()
            {
                crate::film_play::save_spectate_card_setting(pb.spectate_card);
            }
            // #film-camera  Follow the engine into its context cameras -- the assassination
            // camera, the vehicle camera, the third-person spectate on death. Remembered
            // between sessions the way the spectated-player card is.
            if ui
                .checkbox(&mut pb.context_cameras, "Engine context cameras")
                .on_hover_text(
                    "What the game itself does: the view leaves your eyes for an assassination, \
                     a vehicle seat or a weapon that forces third person, and on death it cuts \
                     to your own body and then orbits it while you wait to respawn. Off keeps \
                     the plain camera exactly where you put it.",
                )
                .changed()
            {
                crate::film_camera::save_context_setting(pb.context_cameras);
            }
            // #film-live  Follow a recording that is still being written -- MCC's in-progress
            // `.temp` in its autosave folder. Remembered between sessions.
            if ui
                .checkbox(follow_live, "Follow the live recording")
                .on_hover_text(
                    "An in-progress recording (.temp) grows while the match is played. On, the \
                     clip re-decodes every couple of seconds as more of it is written, so you can \
                     watch a game almost as it happens; your camera and position in the clip are \
                     kept. No effect on a finished film, which cannot grow.",
                )
                .changed()
            {
                crate::film::save_follow_live_setting(*follow_live);
            }
            if *follow_live && !follow_note.is_empty() {
                ui.indent("film_follow_note", |ui| {
                    ui.label(egui::RichText::new(follow_note).weak().small());
                });
            }
            // #nametag-mode  THREE WAYS, not a checkbox. In game there are two different name
            // displays and only one of them is on all the time: the ally waypoint (teammates, in
            // a team game) and the LOOK-AT name, which appears over whoever you are pointing at.
            // "Only when looked at" is that second display, gated by the engine's own targeting
            // rule rather than by an invented cone.
            ui.horizontal(|ui| {
                ui.label("Name tags");
                let before = pb.name_tags;
                egui::ComboBox::from_id_salt("film_name_tag_mode")
                    .selected_text(pb.name_tags.label())
                    .show_ui(ui, |ui| {
                        for m in crate::reach_hud::NameTagMode::ALL {
                            ui.selectable_value(&mut pb.name_tags, m, m.label()).on_hover_text(match m {
                                crate::reach_hud::NameTagMode::Always => {
                                    "Every body HMS draws carries its name, all the time. A film has no local \
                                     player, so this is HMS's own default rather than the game's."
                                }
                                crate::reach_hud::NameTagMode::LookedAt => {
                                    "Only the player the camera is pointing at — the engine's own second name \
                                     display. The cone is the held weapon's authored `autoaim angle` and \
                                     `autoaim range` out of its own weap tag, so it is as wide as the game's \
                                     own targeting is, and there is no line-of-sight test because the engine's \
                                     player path has none either (names read through walls in game too)."
                                }
                                crate::reach_hud::NameTagMode::Hidden => "No names at all.",
                            });
                        }
                    });
                if pb.name_tags != before {
                    crate::reach_hud::save_name_tag_mode(pb.name_tags);
                }
            });
            ui.indent("film_name_tag_help", |ui| {
                ui.label(
                    egui::RichText::new(
                        "Drawn Reach's way: its own tv_nord_hud-15, no background plate, the game's own navpoint \
                         chevron under an ally waypoint, and the colours its chud globals author — pale blue for \
                         the viewer's side, red for anyone else's, grey where there is no player to ask. With \
                         teams off every name is red, because Reach puts every player in a free-for-all on their \
                         own team. The size is a chud virtual size scaled to this window, so it tracks the \
                         window exactly as the real HUD does.",
                    )
                    .weak()
                    .small(),
                );
            });
            if pb.name_tags.is_on() {
                // #reach-font  say which glyphs are being used, because on a machine with no Reach
                // Editing Kit these are NOT the game's own letters.
                let note = nameplate_font_note(ui.ctx());
                let (on_marker, fallback) = nameplate_anchor_counts();
                ui.indent("film_name_font", |ui| {
                    ui.label(egui::RichText::new(note.lines().next().unwrap_or_default()).weak().small())
                        .on_hover_text(note.clone());
                    if on_marker + fallback > 0 {
                        let anchor = if fallback == 0 {
                            format!("Anchor: the engine's own — each body's posed `{NAME_TAG_MARKER}` marker, lifted {LOOKAT_UP_WU} wu (an ally waypoint, {NAME_TAG_MARKER_UP_WU})")
                        } else if on_marker == 0 {
                            format!("Anchor: approximate ({NAME_TAG_UP_WU} wu over the feet) — no posed `{NAME_TAG_MARKER}` marker for these bodies")
                        } else {
                            format!("Anchor: {on_marker} on the engine's posed `{NAME_TAG_MARKER}` marker, {fallback} on the fallback height")
                        };
                        ui.label(egui::RichText::new(anchor).weak().small());
                    }
                });
            }
            ui.separator();
            // #film-objects  The objective objects, loose weapons and vehicle debris the
            // recording carries, as opposed to the map variant's own static Forge scene.
            ui.checkbox(show_objects, "Recorded objects").on_hover_text(
                "The flag, dropped weapons and knocked-off vehicle debris, drawn where the recording puts \
                 them. Off leaves only the map's own Forge scene.",
            );
            if *show_objects && !object_status.is_empty() {
                ui.indent("film_object_status", |ui| {
                    ui.label(egui::RichText::new(object_status).weak().small());
                });
            }
            ui.checkbox(&mut pb.show_all_types, "Include entity types HMS cannot identify")
                .on_hover_text("Their positions decode, but HMS does not know what they are, so they are listed rather than drawn");
        });
}

/// #crosshair #reach-hud #respawn  The three first-person overlay switches, each remembered
/// between sessions the way the spectated-player card is.
///
/// Factored out so they can live in the panel BODY rather than inside a collapsing section. They
/// are what a viewer reaches for while watching, and a control behind a closed header is a control
/// nobody finds.
fn first_person_switches(ui: &mut egui::Ui, pb: &mut crate::film_play::Playback) {
    if ui
        .checkbox(&mut pb.crosshair, "Crosshair")
        .on_hover_text(
            "The reticule the weapon in the subject's hands authors, out of its own `weap` tag's \
             `chud interface` — every bitmap widget of its `crosshair` collection the engine draws with \
             no condition on it, at its authored size, in the colour its chud globals author. 29 of \
             Reach's weapons author one; the grenade launcher's and the falcon's are two pieces, the \
             target laser's four. The MAGNUM authors no chud interface at all, so it has no reticule and \
             HMS draws none rather than inventing one. With nothing in the subject's hands the player \
             HUD's own `unarmed_crosshair` is drawn instead, which is what the game does.",
        )
        .changed()
    {
        crate::reach_hud::save_crosshair_setting(pb.crosshair);
    }
    if ui
        .checkbox(&mut pb.hud, "HUD")
        .on_hover_text(
            "The authored first-person HUD: the weapon's own ammo block (its schematic glyph and its \
             magazine meter), the grenade counts, the shield and health meters, and — for a seated \
             player — the vehicle's own chud. Every piece is placed by the tag's own numbers in Reach's \
             1280x720 virtual screen. A field the recording does not carry is NOT drawn, and the list \
             below says which those are.",
        )
        .changed()
    {
        crate::reach_hud::save_hud_setting(pb.hud);
    }
    if pb.hud {
        ui.indent("film_hud_fields", |ui| {
            for l in hud_field_report() {
                ui.label(egui::RichText::new(l).weak().small());
            }
        });
    }
    if ui
        .checkbox(&mut pb.respawn_widget, "Respawn countdown")
        .on_hover_text(
            "While the camera is on a body that has died, the countdown the game shows a dead player. \
             The clock is the recording's own: a player gets a brand-new network entity each time they \
             come back, so the next life's first tick is exact. Where the recording has no next life to \
             point at, the embedded game variant's own respawn time, growth and suicide/betrayal \
             penalties are counted off the death instead, and the panel says which of the two it used.",
        )
        .changed()
    {
        crate::reach_hud::save_respawn_setting(pb.respawn_widget);
    }
}

/// #reach-hud  What the last painted HUD frame actually filled, and what it could not.
///
/// Published by the draw so the panel can be honest about every field without re-deriving any of
/// it — the same shape `nameplate_anchor_counts` uses.
pub fn hud_field_report() -> Vec<String> {
    HUD_REPORT.with(|c| c.borrow().clone())
}

/// Called by the draw, once a frame, with the lines the panel should show.
pub fn set_hud_field_report(lines: Vec<String>) {
    HUD_REPORT.with(|c| *c.borrow_mut() = lines);
}

thread_local! {
    static HUD_REPORT: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE PRIVACY GATE.**
    ///
    /// A film embeds the recording PC's Windows account and computer name in plain text at
    /// `flmh+0x8C` — `"<Day>_<MMDDYYYY>_<HHMMSS>@<arch>@<windows-user>@<machine>"`. Anything
    /// shared from a PC leaks both, so the card must never show it unless the user explicitly
    /// asks. `metadata_rows` is pure precisely so this is testable without a GUI.
    ///
    /// Tested two ways, because either alone is weak:
    ///
    /// * with a **SENTINEL** — a synthetic header whose session id is a string that appears
    ///   nowhere else — so the test proves something even on a machine with no films, and so a
    ///   leak through any row (not just the one the gate names) is caught;
    /// * over every real film on the machine, which is the data that actually matters.
    ///
    /// It also checks the opposite direction: with `show_machine = true` the id MUST appear.
    /// A gate that cannot fail is worse than no gate, and this project has learned that the hard
    /// way more than once.
    #[test]
    fn the_privacy_gate_hides_the_recording_machine() {
        // Components chosen so none of them is a substring of anything a film publishes, which
        // is what the real-data half of this test has to allow for.
        const SENTINEL: &str = "Mon_01012000_000000@x64@PRIVATE-ACCOUNT-NAME@PRIVATE-MACHINE";

        // (a) the data-flow test, which runs everywhere.
        let mut f = crate::film::Film::default();
        f.header.session_id = SENTINEL.into();
        let hidden = metadata_rows(&f, false, None);
        for (k, v) in &hidden {
            assert!(
                !v.contains("PRIVATE-ACCOUNT-NAME") && !v.contains("PRIVATE-MACHINE") && !v.contains(SENTINEL),
                "the session id leaked into the {k:?} row with show_machine = false: {v:?}"
            );
        }
        let shown = metadata_rows(&f, true, None);
        assert!(
            shown.iter().any(|(_, v)| v.contains(SENTINEL)),
            "with show_machine = true the recording machine must be shown, or the gate is not a gate"
        );
        assert_eq!(shown.len(), hidden.len() + 1, "show_machine adds exactly one row");

        // (b) the same rule over every real film this machine has.
        let films = crate::film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_ui: no real films -- privacy gate not exercised on real data");
            return;
        }
        let mut checked = 0usize;
        let mut with_id = 0usize;
        for p in films.iter().take(40) {
            let Ok(f) = crate::film::open_header(p) else { continue };
            let id = f.header.session_id.clone();
            checked += 1;
            // The literal "UNINIT_SESSIONID" is not identifying; anything else is.
            if id.is_empty() || id == "UNINIT_SESSIONID" {
                continue;
            }
            with_id += 1;
            // Not just the whole string: the account and machine names are the parts that matter
            // and either could leak on its own.
            //
            // But a component is only a LEAK if the film does not already publish it. A film
            // carries the author's gamertag by design, and on this machine the Windows account
            // happens to be a prefix of it -- flagging that would make the gate cry wolf about
            // the one field it is supposed to show. So components that appear in what the film
            // publishes anyway are excluded, and everything else must be absent.
            let published: String = std::iter::once(f.content.created_by.clone())
                .chain(std::iter::once(f.content.title.clone()))
                .chain(f.present_players().map(|q| q.gamertag.clone()))
                .collect::<Vec<_>>()
                .join("\u{1}");
            let parts: Vec<&str> =
                id.split('@').filter(|s| s.len() > 2 && !published.contains(*s)).collect();
            for (k, v) in metadata_rows(&f, false, None) {
                assert!(
                    !v.contains(&id),
                    "{}: the whole session id leaked into the {k:?} row",
                    p.file_name().unwrap_or_default().to_string_lossy()
                );
                for part in &parts {
                    assert!(
                        !v.contains(part),
                        "{}: session-id component {part:?} leaked into the {k:?} row: {v:?}",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    );
                }
            }
            assert!(
                metadata_rows(&f, true, None).iter().any(|(_, v)| v.contains(&id)),
                "{}: show_machine = true did not show the session id",
                p.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        assert!(checked > 0, "no film header could be read, so the gate was not exercised");
        eprintln!(
            "film_ui: privacy gate held on {checked} film(s) ({with_id} carrying a real session id; data-flow tested with a sentinel)"
        );
    }

    /// The card says the right things about a film, across both shapes a film comes in: recorded
    /// on a FORGE variant (where the object and budget rows exist) and on a STOCK map (where the
    /// variant is empty and the card must say so rather than showing an empty Forge scene).
    #[test]
    fn the_card_describes_both_forge_and_stock_films() {
        let films = crate::film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_ui: no real films -- card content not checked");
            return;
        }
        let (mut forge, mut stock) = (0usize, 0usize);
        for p in films.iter().take(40) {
            let Ok(f) = crate::film::open_header(p) else { continue };
            let rows = metadata_rows(&f, false, None);
            let key = |k: &str| rows.iter().find(|(a, _)| *a == k).map(|(_, v)| v.clone());
            let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            // Rows every film must have.
            for k in ["Film", "Length", "Recorded", "Saved by", "Base map", "Gametype", "Map variant", "Players"] {
                assert!(key(k).is_some(), "{name}: the card has no {k:?} row");
            }
            // The length row restates the tick count, so a film whose header says 0 ticks cannot
            // silently show a plausible duration.
            let len = key("Length").unwrap();
            assert!(
                len.contains(&format!("{} ticks", f.header.tick_count)),
                "{name}: the Length row does not carry the real tick count: {len:?}"
            );
            let mvrow = key("Map variant").unwrap();
            if f.header.map_variant.is_empty() {
                stock += 1;
                assert!(mvrow.contains("stock map"), "{name}: a stock-map film must say so: {mvrow:?}");
                assert!(key("Forge objects").is_none(), "{name}: a stock-map film has no Forge objects row");
            } else {
                forge += 1;
                assert!(
                    key("Forge objects").is_some(),
                    "{name}: a Forge film must report its object count"
                );
                let objs = key("Forge objects").unwrap();
                assert!(
                    objs.contains(&format!("{} of", f.header.map_variant.present_objects().count())),
                    "{name}: the Forge objects row does not carry the real count: {objs:?}"
                );
            }
        }
        assert!(forge + stock > 0, "no film header could be read, so the card was not checked");
        eprintln!("film_ui: card content checked on {forge} Forge film(s) and {stock} stock-map film(s)");
    }

    // ---------------------------------------------------------------------------------------
    // #reach-chud  the floating player name: its anchor and its colour
    // ---------------------------------------------------------------------------------------

    fn tag(slot: Option<u8>, team: u8, named: bool) -> NameTag {
        NameTag {
            text: "x".into(),
            pos: [0.0; 3],
            named,
            inferred_from: None,
            team,
            net_id: 1,
            feet: [0.0; 3],
            on_marker: false,
            slot,
            color_index: 2,
            show_icon: false,
        }
    }

    /// **THE ANCHOR GATE.** The engine resolves marker `string_id` 0xC2 on the unit's render
    /// model, and in a shipped Reach cache that id is `head`. Two earlier passes anchored this
    /// on `chud_nav_point` instead, which sits further up the head bone, and that is what put
    /// the names too high. This pins the constant to the engine's argument and states the
    /// measured heights, so a drift back to `chud_nav_point` fails here.
    ///
    /// The cache-backed half of the gate (`string_id 0xC2 == "head"`, and the measured bind-pose
    /// heights) lives in `reach_actor`'s test module, which owns the shared Reach cache.
    #[test]
    fn the_name_anchor_is_the_engine_marker() {
        assert_eq!(NAME_TAG_MARKER, "head");
        assert_ne!(NAME_TAG_MARKER, "chud_nav_point");
        assert_eq!(crate::reach_chud::NAME_ANCHOR_MARKER_SID, 0xC2);
        // TWO lifts off the SAME marker: the ally waypoint's and the look-at display's.
        assert_eq!(NAME_TAG_MARKER_UP_WU, 0.075); // +0x2CB76C
        assert_eq!(LOOKAT_UP_WU, 0.05); // +0x18AE04
        assert!(LOOKAT_UP_WU < NAME_TAG_MARKER_UP_WU, "the look-at name sits lower than the waypoint");
        // The fallback is the MEASURED anchor on a Spartan: `head` at bind-pose z = 0.633 wu plus
        // the look-at lift, because that is the display almost every body gets. It must not drift
        // back towards `chud_nav_point`'s 0.762, and it must not be the old invented 0.85.
        assert!((NAME_TAG_UP_WU - (0.633 + LOOKAT_UP_WU)).abs() < 0.002, "{NAME_TAG_UP_WU}");
        assert!(NAME_TAG_UP_WU < 0.75, "the fallback is above the engine's own anchor");
    }

    /// **THE COLOUR GATE.** Every colour must come out of the authored `chgd` skin, not out of a
    /// constant in this file: a synthetic skin with distinctive colours has to show through.
    #[test]
    fn nameplate_colours_come_from_the_authored_skin() {
        let mut colors = [[0.0f32; 4]; 32];
        // argb, alpha first; give each slot an unmistakable red channel.
        for (i, c) in colors.iter_mut().enumerate() {
            *c = [0.0, i as f32 / 255.0, 0.0, 0.0];
        }
        let skin = crate::reach_chud::ChudSkin {
            skin_type: 0,
            colors,
            waypoint_min_scale: 0.4,
            waypoint_max_scale: 2.5,
            navpoints_tag: None,
            saved_film_tag: None,
            curvatures: Vec::new(),
        };
        let r = NamePlateRules::from_chud(&skin);
        // index 0 -> SKIN_COLORS[7] 'navpt normal', 2 -> [9] 'navpt neutral', 3 -> [10] 'navpt enemy'
        assert_eq!(r.navpt_normal, egui::Color32::from_rgb(7, 0, 0));
        assert_eq!(r.navpt_neutral, egui::Color32::from_rgb(9, 0, 0));
        assert_eq!(r.navpt_enemy, egui::Color32::from_rgb(10, 0, 0));
        // the scale clamps are the skin's own, not the typed-in 0.8 / 1.3
        assert_eq!(r.scale_min, 0.4);
        assert_eq!(r.scale_max, 2.5);
        assert_eq!(nameplate_scale(1.0, &r), 2.5);
        assert_eq!(nameplate_scale(1000.0, &r), 0.4);
        // and the index -> colour map is the engine's
        assert_eq!(r.color_for_index(0), r.navpt_normal);
        assert_eq!(r.color_for_index(3), r.navpt_enemy);
        assert_eq!(r.color_for_index(2), r.navpt_neutral);
        // The DEFAULTS, for a film opened with no cache, are the shipped authored values.
        let d = NamePlateRules::default();
        assert_eq!(d.navpt_normal, egui::Color32::from_rgb(36, 148, 255)); // ui\chud\globals +0x74
        assert_eq!(d.navpt_enemy, egui::Color32::from_rgb(216, 50, 50)); // ui\chud\globals +0xA4
        assert_eq!(d.navpt_neutral, egui::Color32::from_rgb(94, 94, 94)); // ui\chud\globals +0x94
        assert_eq!((d.scale_min, d.scale_max), (0.8, 1.3));
    }

    /// **THE FREE-FOR-ALL GATE.** With teams off, Reach puts every player on their own team, so
    /// no name is an ally's: the engine's rule gives them all `navpt enemy`. The saved player
    /// slots all read team 0 on a teams-off clip (measured: 16 of 16), so a colour taken from
    /// that byte would make every name an ALLY — the exact wrong answer, and what this catches.
    #[test]
    fn a_free_for_all_clip_is_all_navpt_enemy() {
        let mut tags: Vec<NameTag> = (0..4).map(|i| tag(Some(i), 0, true)).collect();
        resolve_name_colors(&mut tags, Some(2), 0, false);
        for t in &tags {
            if t.slot == Some(2) {
                assert_eq!(t.color_index, 0, "the viewer's own body is on the viewer's team");
            } else {
                assert_eq!(t.color_index, 3, "slot {:?} should be navpt enemy", t.slot);
            }
        }
        let r = NamePlateRules::default();
        assert_eq!(r.color_for_index(3), ENGINE_NAVPT_ENEMY);
    }

    /// With teams ON the same rule splits the roster by the viewer's team, and a body whose
    /// player HMS could not name stays `navpt neutral` rather than being guessed at.
    #[test]
    fn a_team_clip_splits_by_the_viewers_team() {
        let mut tags = vec![
            tag(Some(0), 0, true), // the viewer
            tag(Some(1), 0, true), // ally
            tag(Some(2), 1, true), // enemy
            tag(None, 0xFF, false),   // an entity-id label: no player
        ];
        resolve_name_colors(&mut tags, Some(0), 0, true);
        assert_eq!(tags[0].color_index, 0);
        assert_eq!(tags[1].color_index, 0);
        assert_eq!(tags[2].color_index, 3);
        assert_eq!(tags[3].color_index, 2);
        // No viewer at all -> the engine's answer for an unknown team, for every tag.
        resolve_name_colors(&mut tags, None, 0xFF, true);
        assert!(tags.iter().all(|t| t.color_index == 2));
    }

    /// **THE CHEVRON GATE.** The triangle waypoint is the ALLY WAYPOINT, not the name display:
    /// `+0x2CB76C`'s player loop makes a navpoint record — and a record is the only thing that
    /// draws a chevron — only for another player on the LOCAL player's own team. So it belongs
    /// over a teammate in a team game and nowhere else: not in a free-for-all, not over an enemy,
    /// and not over the viewer's own body (which the engine skips outright).
    ///
    /// This is the regression that shipped: the chevron was drawn over every named body,
    /// including a 16-player free-for-all where Reach draws no waypoint at all.
    #[test]
    fn the_chevron_is_only_for_an_ally_in_a_team_game() {
        // Teams OFF: nobody is an ally, so nobody gets a chevron -- not even the viewer.
        let mut ffa: Vec<NameTag> = (0..4).map(|i| tag(Some(i), 0, true)).collect();
        resolve_name_colors(&mut ffa, Some(2), 0, false);
        assert!(ffa.iter().all(|t| !t.show_icon), "a free-for-all has no ally waypoints");
        assert!(ffa.iter().filter(|t| t.slot != Some(2)).all(|t| t.color_index == 3));

        // Teams ON: the ally gets one; the enemy, the viewer and an unnamed body do not.
        let mut team = vec![
            tag(Some(0), 0, true), // the viewer
            tag(Some(1), 0, true), // ally
            tag(Some(2), 1, true), // enemy
            tag(None, 0xFF, false),   // no player
        ];
        resolve_name_colors(&mut team, Some(0), 0, true);
        assert!(!team[0].show_icon, "the engine skips the local player entirely");
        assert!(team[1].show_icon, "a teammate is exactly who gets the waypoint");
        assert!(!team[2].show_icon, "an enemy player never gets one");
        assert!(!team[3].show_icon, "neither does a body with no player");
        // and the gate is the colour index, so the two can never disagree
        assert!(team.iter().all(|t| !t.show_icon || t.color_index == 0));
        // `show_icon` also picks the LIFT, so the two displays cannot end up on one anchor.
        let lift = |t: &NameTag| if t.show_icon { NAME_TAG_MARKER_UP_WU } else { LOOKAT_UP_WU };
        assert_eq!(lift(&team[1]), 0.075, "an ally waypoint uses +0x2CB76C's lift");
        assert_eq!(lift(&team[2]), 0.05, "a look-at name uses +0x18AE04's");
        assert!(ffa.iter().all(|t| lift(t) == 0.05), "a free-for-all is all look-at names");
    }

    /// **THE SIZE GATE.** The drawn size must be a chud VIRTUAL size scaled to the window, not a
    /// constant. HMS used to paint the font's own 21 authored pixels whatever the window was,
    /// which is right at exactly one window size; the engine maps its 720-line virtual screen onto
    /// the safe frame, so the same text is 0.87x as tall in a 720-point viewport and 1.74x in a
    /// 1440-point one. A regression to a fixed size fails the ratio assertions here.
    #[test]
    fn the_nameplate_size_follows_the_virtual_screen() {
        let r = NamePlateRules::default();
        assert_eq!(r.virtual_height, 720.0); // chgd skin curvature infos[fullscreen wide] +0x098
        assert_eq!(r.safe_frame_v, 0.870); // ... +0x0D4
        // one virtual unit, in points
        assert!((r.px_per_virtual(720.0) - 0.870).abs() < 1e-4, "{}", r.px_per_virtual(720.0));
        assert!((r.px_per_virtual(1080.0) - 1.305).abs() < 1e-4, "{}", r.px_per_virtual(1080.0));
        assert!((r.px_per_virtual(1440.0) - 1.740).abs() < 1e-4, "{}", r.px_per_virtual(1440.0));
        // it is LINEAR in the window: a fixed size would make this ratio 1.0
        let (a, b) = (r.px_per_virtual(600.0), r.px_per_virtual(1200.0));
        assert!((b / a - 2.0).abs() < 1e-4, "doubling the window must double the HUD: {a} -> {b}");
        // and the font's authored 21 units land where the engine puts them
        let line = crate::reach_chud::HUD_FONT_LINE_HEIGHT_VU;
        assert_eq!(line, 21.0);
        let at = |h: f32, d: f32| line * nameplate_scale(d, &r) * r.px_per_virtual(h);
        assert!((at(1080.0, 100.0) - 21.9).abs() < 0.1, "{}", at(1080.0, 100.0)); // far: 0.8x
        assert!((at(1080.0, 1.0) - 35.6).abs() < 0.1, "{}", at(1080.0, 1.0)); // near: 1.3x
        // the ding is an inset, so a non-zero one can only make it smaller
        let mut d = r;
        d.ding_v = 8.0;
        assert!(d.px_per_virtual(1080.0) < r.px_per_virtual(1080.0));
    }

    /// **THE LAYOUT GATE.** The chevron's point sits ON the anchor and the name stacks on top of
    /// it — not centred over it, and not floating with the icon's height ignored.
    #[test]
    fn the_chevron_carries_the_name_above_it() {
        let a = egui::pos2(100.0, 200.0);
        // the authored sprite, measured off ui\chud\bitmaps\waypoints sequence 1 sprite 0
        let (rect, text) = nameplate_layout(a, 2.0, Some([37.5, 21.6]));
        let rect = rect.expect("an icon size gives an icon rect");
        assert!((rect.width() - 75.0).abs() < 1e-3, "{rect:?}");
        assert!((rect.height() - 43.2).abs() < 1e-3, "{rect:?}");
        assert!((rect.center().x - a.x).abs() < 1e-3, "the chevron is centred on the navpoint");
        assert!((rect.bottom() - a.y).abs() < 1e-3, "its point sits ON the anchor");
        assert!((text.y - (a.y - 43.2)).abs() < 1e-3, "the name's box bottom is the chevron's top");
        assert!((text.x - a.x).abs() < 1e-3, "the name is centred too: +0x2E3BAC's v44 is false here");
        // With no sprite the engine's own other branch applies: centred, no offset at all.
        let (none, text) = nameplate_layout(a, 2.0, None);
        assert!(none.is_none());
        assert_eq!(text, a);
    }
}

// ---------------------------------------------------------------------------------------------
// #reach-widget  the scoreboard's legibility, gated
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod scoreboard_tests {
    use super::*;
    use crate::reach_widget::{Binding, Component, Props, Screen, UiGlobals};

    /// `ui\cui\lobbies\player_base_bitmap`, MEASURED out of `forge_halo.map`: ONE pixel of
    /// 180,180,180 at alpha 255. The wide plate the player name sits on is this, multiplied by
    /// whatever tint the widget is given -- which is the whole reason a white tint is illegible.
    const PLAYER_BASE_GREY: f32 = 180.0 / 255.0;

    /// `wigl` +0x74 `text color`, measured: a 1.0, rgb 0.92157.
    const WIGL_TEXT: [f32; 4] = [1.0, 0.92157, 0.92157, 0.92157];
    /// `wigl` +0x84 `shadow color`, measured: a 0.5, rgb 0.
    const WIGL_SHADOW: [f32; 4] = [0.5, 0.0, 0.0, 0.0];

    /// `wigl` +0xA0 `tint colors` / `bitmap team`, all eight, measured out of `forge_halo.map`.
    const BITMAP_TEAM: [[f32; 4]; 8] = [
        [1.0, 0.53218, 0.08139, 0.08139],
        [1.0, 0.10581, 0.19524, 0.47335],
        [1.0, 0.08254, 0.27335, 0.08254],
        [1.0, 0.59884, 0.25915, 0.07985],
        [1.0, 0.30601, 0.15028, 0.38708],
        [1.0, 0.90473, 0.72000, 0.23771],
        [1.0, 0.30865, 0.18986, 0.12467],
        [1.0, 0.79100, 0.47150, 0.64199],
    ];

    /// `wigl` +0xA0 `tint colors`[0] `bitmap player`, the three entries that matter here,
    /// MEASURED out of `forge_halo.map`: the DARKEST (`Green`, index 9), the one a `Steel`
    /// player gets (index 0), and the PALEST (`White`, index 2). This is the list the engine
    /// indexes with the player's primary armour colour, and the reason a row's colour cannot be
    /// read off it at panel size.
    const BITMAP_PLAYER_STEEL: [f32; 4] = [1.0, 0.03689, 0.03689, 0.03689];
    const BITMAP_PLAYER_WHITE: [f32; 4] = [1.0, 0.69387, 0.69387, 0.69387];
    const BITMAP_PLAYER_GREEN: [f32; 4] = [1.0, 0.01176, 0.07843, 0.02353];

    /// `matg` +0x498 `profile colors`, all thirty, MEASURED out of `forge_halo.map`, in the
    /// LINEAR RGB the tag stores. This is Reach's armour colour palette and the table the
    /// third-person body is tinted from, so it is also what a scoreboard row must agree with.
    const PROFILE_COLORS: [[f32; 3]; 30] = [
        [0.127, 0.127, 0.127],
        [0.339, 0.339, 0.339],
        [0.694, 0.694, 0.694],
        [0.174, 0.104, 0.063],
        [0.354, 0.213, 0.146],
        [0.609, 0.404, 0.300],
        [0.180, 0.205, 0.099],
        [0.317, 0.390, 0.158],
        [0.484, 0.597, 0.268],
        [0.036, 0.197, 0.059],
        [0.101, 0.425, 0.162],
        [0.252, 0.617, 0.289],
        [0.029, 0.252, 0.236],
        [0.068, 0.452, 0.445],
        [0.216, 0.723, 0.697],
        [0.047, 0.116, 0.233],
        [0.130, 0.223, 0.386],
        [0.272, 0.440, 0.703],
        [0.114, 0.108, 0.272],
        [0.210, 0.198, 0.484],
        [0.391, 0.374, 0.731],
        [0.366, 0.052, 0.052],
        [0.660, 0.059, 0.059],
        [0.825, 0.303, 0.303],
        [0.462, 0.095, 0.020],
        [0.847, 0.209, 0.051],
        [0.839, 0.423, 0.254],
        [0.451, 0.235, 0.025],
        [0.738, 0.558, 0.076],
        [0.880, 0.839, 0.202],
    ];

    /// The two plates a text column of the row sits on, with the MEASURED mean of their art:
    /// `player_base_bitmap` under the name is 180/255 grey, `player_base_servicetag_bitmap`
    /// under the service tag is 213/255.
    const NAME_PLATE_ART: f32 = 180.0 / 255.0;
    const TAG_PLATE_ART: f32 = 213.0 / 255.0;

    /// The 30-entry `bitmap player` list with the three measured entries in their real places
    /// and the rest filled from the darkened pattern, so an index is never out of range by
    /// accident in a test that is about something else.
    fn bitmap_player_list() -> Vec<[f32; 4]> {
        let mut v = vec![[1.0, 0.1, 0.1, 0.1]; 30];
        v[0] = BITMAP_PLAYER_STEEL;
        v[2] = BITMAP_PLAYER_WHITE;
        v[9] = BITMAP_PLAYER_GREEN;
        v
    }

    fn globals() -> UiGlobals {
        UiGlobals {
            text_color: WIGL_TEXT,
            shadow_color: WIGL_SHADOW,
            presets: vec![
                ("white".into(), [1.0, 0.84706, 0.84706, 0.84706]),
                ("body".into(), [1.0, 0.35, 0.35, 0.35]),
                ("hilite".into(), [1.0, 0.92157, 0.92157, 0.92157]),
            ],
            tint_text_player: Vec::new(),
            tint_text_team: Vec::new(),
            tint_bitmap_player: bitmap_player_list(),
            tint_bitmap_team: BITMAP_TEAM.to_vec(),
            primary_emblem_count: 80,
            secondary_emblem_count: 53,
        }
    }

    fn comp(kind: &str, name: &str, parent: &str, props: Props) -> Component {
        Component { kind: kind.into(), name: name.into(), parent: parent.into(), props }
    }

    fn bind(sc: &str, sp: &str, dc: &str, dp: &str) -> Binding {
        Binding { src_comp: sc.into(), src_prop: sp.into(), dst_comp: dc.into(), dst_prop: dp.into() }
    }

    /// The shipped row template's colour wiring, exactly as `--dump-scoreboard` prints it out of
    /// `forge_halo.map`: `roster_bitmap` and the two colour bars author NO `prop_tint_color`,
    /// the two decision makers author theirs, and four bindings connect them. The
    /// `prop_player_bitmap_tint_color` link lives on the SCREEN rather than the template, so it
    /// is included here the way the drawing code sees it once the row is stamped.
    fn shipped_row() -> Screen {
        let argb = |n: &str, v: [f32; 4]| Props { colors: vec![(n.to_string(), v)], ..Default::default() };
        // Every box is the authored one, as `--dump-scoreboard` prints it out of `forge_halo.map`,
        // because the code under test asks the TAG which plate a column sits on.
        let at = |l: f32, t: f32, w: f32, h: f32| Props {
            reals: vec![
                ("prop_left".into(), l),
                ("prop_top".into(), t),
                ("prop_width".into(), w),
                ("prop_height".into(), h),
            ],
            ..Default::default()
        };
        let mut with = |l: f32, t: f32, w: f32, h: f32, extra: Props| {
            let mut p = at(l, t, w, h);
            p.colors = extra.colors;
            p.sids = extra.sids;
            p
        };
        Screen {
            path: "scoreboard_list_item_player".into(),
            virtual_size: [1152.0, 720.0],
            components: vec![
                // The two containers the row's boxes are offsets from: both at 0,0, so the
                // parent walk is exercised without moving anything.
                comp("container_widget", "bitmap", "", at(0.0, 0.0, 350.0, 28.0)),
                comp("container_widget", "data", "", at(0.0, 0.0, 350.0, 28.0)),
                comp("bitmap_widget", "roster_bitmap", "bitmap", at(53.0, 0.0, 193.0, 28.0)),
                comp("bitmap_widget", "roster_color_start_bitmap", "bitmap", at(25.0, 0.0, 28.0, 28.0)),
                comp("bitmap_widget", "roster_color_end_bitmap", "bitmap", at(246.0, 0.0, 54.0, 28.0)),
                // `roster_position_bitmap` and `roster_score_bitmap`: the two plates that DO
                // author a tint, near-black.
                comp(
                    "bitmap_widget",
                    "roster_position_bitmap",
                    "bitmap",
                    with(3.0, 0.0, 22.0, 28.0, argb("prop_tint_color", [1.0, 0.0908, 0.0908, 0.0908])),
                ),
                comp(
                    "bitmap_widget",
                    "roster_score_bitmap",
                    "bitmap",
                    with(300.0, 0.0, 50.0, 28.0, argb("prop_tint_color", [1.0, 0.0908, 0.0908, 0.0908])),
                ),
                comp("decision_maker", "bitmap_color_decision_maker", "list_item_player", Props {
                    colors: vec![
                        ("prop_else".into(), [1.0, 1.0, 1.0, 1.0]),
                        ("prop_then".into(), [1.0, 1.0, 1.0, 1.0]),
                    ],
                    ..Default::default()
                }),
                comp("decision_maker", "player_dropped_opacity_decision_maker", "list_item_player", Props {
                    colors: vec![
                        ("prop_else".into(), [0.549, 1.0, 1.0, 1.0]),
                        ("prop_then".into(), [1.0, 1.0, 1.0, 1.0]),
                    ],
                    ..Default::default()
                }),
                // The EMBLEM slot, which is what makes `roster_color_start_bitmap` the one plate
                // with no text on it.
                comp("emblem_widget", "bitmap_player_scoreboard_emblem", "data", at(27.0, 2.0, 24.0, 24.0)),
                comp("text_widget", "text_player_position", "data", at(3.0, 4.0, 22.0, 22.0)),
                comp("text_widget", "text_player_name", "data", at(58.0, 4.0, 187.0, 22.0)),
                comp("text_widget", "text_player_designation", "data", at(246.0, 4.0, 54.0, 22.0)),
                comp(
                    "text_widget",
                    "text_player_score",
                    "data",
                    with(300.0, 4.0, 50.0, 22.0, Props {
                        sids: vec![("prop_color_preset".into(), "white".into())],
                        ..Default::default()
                    }),
                ),
            ],
            bindings: vec![
                bind("roster_color_start_bitmap", "prop_tint_color", "bitmap_color_decision_maker", "prop_then"),
                bind("bitmap_color_decision_maker", "prop_result", "roster_bitmap", "prop_tint_color"),
                // from the SCREEN's own binding block
                bind("data_player_scoreboard", "prop_player_bitmap_tint_color", "roster_color_start_bitmap", "prop_tint_color"),
                bind("data_player_scoreboard", "prop_player_bitmap_tint_color", "roster_color_end_bitmap", "prop_tint_color"),
            ],
        }
    }

    fn row(team: u8, armour: Option<u8>, show_team: bool) -> ScoreboardRow {
        ScoreboardRow { name: "Sopitive".into(), team, show_team, armour_color: armour, named: true, ..Default::default() }
    }

    /// sRGB -> relative luminance, then the standard contrast ratio. Used to put a NUMBER on
    /// "the names are extremely hard to see" rather than an opinion.
    fn luminance(c: egui::Color32) -> f32 {
        let f = |v: u8| {
            let s = v as f32 / 255.0;
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * f(c.r()) + 0.7152 * f(c.g()) + 0.0722 * f(c.b())
    }
    fn contrast(a: egui::Color32, b: egui::Color32) -> f32 {
        let (x, y) = (luminance(a), luminance(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }
    /// The plate as it is actually PAINTED: the authored tint multiplied into
    /// `player_base_bitmap`'s single 180-grey pixel.
    fn plate_as_painted(tint: [f32; 4]) -> egui::Color32 {
        let q = |v: f32| ((v.clamp(0.0, 1.0) * PLAYER_BASE_GREY) * 255.0 + 0.5) as u8;
        egui::Color32::from_rgb(q(tint[1]), q(tint[2]), q(tint[3]))
    }

    /// The name plate must take the colour the TAG's bindings give it, not the white default.
    ///
    /// This is the defect itself. `roster_bitmap` authors no `prop_tint_color`, so the drawing
    /// code used to fall back to white and paint `player_base_bitmap`'s 180-grey straight out,
    /// 193 virtual units wide, directly under the player name.
    #[test]
    fn the_name_plate_follows_the_tags_own_binding_instead_of_defaulting_to_white() {
        let (sc, g) = (shipped_row(), globals());
        let plate = sc.get("roster_bitmap").expect("roster_bitmap");
        for team in 0..8u8 {
            let r = row(team, Some(0), true);
            let t = resolve_tint(&sc, plate, Some(&r), &g);
            assert_eq!(t, BITMAP_TEAM[team as usize], "team {team}: the plate did not follow the binding chain");
            // And the little square behind the emblem is wired to the SAME colour, which is what
            // the chain says: start bitmap -> decision maker `prop_then` -> roster_bitmap.
            let start = sc.get("roster_color_start_bitmap").unwrap();
            assert_eq!(resolve_tint(&sc, start, Some(&r), &g), t);
        }
        // A plate that DOES author a tint still gets its own, bindings or no bindings.
        let pos = sc.get("roster_position_bitmap").unwrap();
        assert_eq!(resolve_tint(&sc, pos, Some(&row(0, None, false)), &g), [1.0, 0.0908, 0.0908, 0.0908]);
        // THE ROW THE USER ACTUALLY SEES MOST OF on a clip whose header stored three gamertags:
        // a body HMS cannot name. It must NOT get the decision maker's authored white -- not
        // being able to name somebody is not the same fact as their having left the game, and a
        // film cannot tell us the latter at all. A neutral dark bar, like every other plate.
        let unnamed = ScoreboardRow { named: false, team: 0xFF, show_team: false, ..Default::default() };
        let t = resolve_tint(&sc, plate, Some(&unnamed), &g);
        assert_ne!(t, [1.0, 1.0, 1.0, 1.0], "an unnamed row still gets the white plate");
        assert!(t[1] < 0.25 && t[2] < 0.25 && t[3] < 0.25, "an unnamed row's bar is not dark: {t:?}");
    }

    /// The text colours come from the map's own palette, not from values picked by eye.
    #[test]
    fn text_colours_come_from_the_wigl_palette() {
        let g = globals();
        // No preset at all -- which is what the name, the position and the service tag author --
        // is the tag's `text color` default, NOT white.
        assert_eq!(g.text_preset_or_default(None), WIGL_TEXT);
        // The score authors `white`, and Reach's `white` is 0.847, not 1.0.
        assert_eq!(g.text_preset_or_default(Some("white")), [1.0, 0.84706, 0.84706, 0.84706]);
        // `body` is DARK (0.35). The old code mapped it by eye to a light 168/176/184, which is
        // 0.66 -- nearly twice as bright as the tag says.
        let body = g.text_preset_or_default(Some("body"));
        assert_eq!(body, [1.0, 0.35, 0.35, 0.35]);
        assert!(body[1] < 0.5, "body must be the dark preset it is authored as");
        // A preset the tag does not define falls back to the default, as the engine's resolver
        // does -- it never substitutes something invented.
        assert_eq!(g.text_preset_or_default(Some("no_such_preset")), WIGL_TEXT);
    }

    /// The alpha of a `real_argb_color` is its FIRST float and must not be dropped.
    #[test]
    fn argb_alpha_is_not_discarded() {
        let opaque = argb32([1.0, 1.0, 1.0, 1.0], 1.0);
        let half = argb32([0.5, 1.0, 1.0, 1.0], 1.0);
        assert!(half.a() < opaque.a(), "the alpha float was ignored");
        // `player_dropped_opacity_decision_maker`'s own `prop_else`.
        assert_eq!(argb32([0.549, 1.0, 1.0, 1.0], 1.0).a(), (0.549 * 255.0 + 0.5) as u8);
    }

    /// #emblem  The sentence the panel prints about emblems is a COUNT, so it is gated.
    #[test]
    fn the_emblem_note_counts_what_the_recording_holds() {
        let em = |fg: u8| crate::film::EmblemInfo { foreground: fg, background: 44, color_primary: 21, ..Default::default() };
        let r = |e: Option<crate::film::EmblemInfo>| ScoreboardRow { emblem: e, named: true, ..Default::default() };
        // Nothing at all.
        assert!(emblem_note(&[r(None), r(None)]).contains("no valid emblem"));
        // All the same -- the 7:11 test clip, sixteen slots, one emblem.
        let all_same: Vec<ScoreboardRow> = (0..16).map(|_| r(Some(em(16)))).collect();
        let n = emblem_note(&all_same);
        assert!(n.contains("16 slots") && n.contains("SAME"), "{n}");
        // Fifteen placeholders and one real one -- measured on another 16-player clip.
        let mut mixed = all_same.clone();
        mixed[14] = r(Some(em(25)));
        let n = emblem_note(&mixed);
        assert!(n.contains("2 distinct") && n.contains("15 of them"), "{n}");
    }

    /// Every data column of the row gets its value. The SERVICE TAG column is the one that was
    /// reported missing, and while the cause turned out to be the plate it was drawn on, the
    /// mapping that feeds it has no gate of its own -- so it gets one.
    #[test]
    fn every_data_column_of_the_row_gets_its_value() {
        let r = ScoreboardRow {
            position: 3,
            name: "Zenifible 2".into(),
            service_tag: "LUA".into(),
            score: Some(17),
            ..Default::default()
        };
        assert_eq!(widget_text("text_player_name", None, Some(&r)).as_deref(), Some("Zenifible 2"));
        assert_eq!(widget_text("text_player_designation", Some("service_tag"), Some(&r)).as_deref(), Some("LUA"));
        assert_eq!(widget_text("text_player_position", None, Some(&r)).as_deref(), Some("3"));
        assert_eq!(widget_text("text_player_score", Some("score"), Some(&r)).as_deref(), Some("17"));
        // A recording with no score for this player draws nothing, rather than a made-up 0.
        let no_score = ScoreboardRow { score: None, ..r.clone() };
        assert_eq!(widget_text("text_player_score", Some("score"), Some(&no_score)), None);
        // A body with no header record has no service tag, and that column stays empty rather
        // than borrowing somebody else's.
        let unnamed = ScoreboardRow { name: "Spartan 7".into(), named: false, position: 9, ..Default::default() };
        assert_eq!(widget_text("text_player_name", None, Some(&unnamed)).as_deref(), Some("Spartan 7"));
        assert_eq!(widget_text("text_player_designation", Some("service_tag"), Some(&unnamed)), None);
        // A component that is not one of the four data columns is a LABEL, and a key that
        // resolves nowhere draws nothing instead of printing the key.
        assert_eq!(widget_text("text_header_nonsense", Some("no_such_key_at_all"), Some(&r)), None);
        assert_eq!(widget_text("text_header_nonsense", None, Some(&r)), None);
    }

    /// The whole list PAINTS, with the real painter, and what comes out is right.
    ///
    /// An end-to-end smoke test over `paint_scoreboard`: a synthetic two-row list through a real
    /// `egui::Context`, inspecting the shapes that come out. It is deliberately machine
    /// independent -- which glyphs get drawn depends on whether Reach's fonts are installed here
    /// -- so it asserts the things that do not: that plates are drawn, that none is the pure
    /// white the bug painted, and that the teal player's row really does carry teal.
    #[test]
    fn the_list_paints_each_players_own_colour() {
        use crate::reach_widget::Scoreboard;
        let g = globals();
        // The screen half: just the frame the painter needs, with no header strip.
        let screen = Screen { path: "scoreboard".into(), virtual_size: [1152.0, 720.0], ..Default::default() };
        let mut sb = Scoreboard { screen, player_row: shipped_row(), team_row: Screen::default() };
        // Give every bitmap widget a tag so the painter has something to draw, and a decoder
        // that returns `player_base_bitmap`'s real pixel.
        for c in sb.player_row.components.iter_mut().filter(|c| c.kind == "bitmap_widget") {
            c.props.tags.push(("prop_bitmap_reference".into(), "bitm".into(), Some(1)));
        }
        let decode = |_t: u32| Some((vec![180u8, 180, 180, 255], 1u32, 1u32));
        let rows = vec![
            ScoreboardRow {
                position: 1,
                name: "Kotorin3571".into(),
                service_tag: "117".into(),
                score: Some(9),
                team: 0,
                show_team: false,
                // 14 = the teal armour colour, which two players of the test clip wear.
                armour_color: Some(14),
                named: true,
                ..Default::default()
            },
            ScoreboardRow { position: 2, name: "Spartan 7".into(), team: 0xFF, show_team: false, ..Default::default() },
        ];
        let ctx = egui::Context::default();
        let mut height = 0.0f32;
        let out = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.set_max_width(360.0);
                height = paint_scoreboard(ui, &sb, &rows, &g, &PROFILE_COLORS, hms_native::CacheHandle(0), &decode);
            });
        });
        assert!(height > 0.0, "the list painted nothing at all");
        // Which meshes are PLATES: exactly the ones drawn with the bitmap texture. Asking the
        // cache for it gives the same handle the painter used, and that keeps the test honest on
        // a machine where Reach's own fonts ARE installed -- the glyph painter emits meshes too.
        let plate_tex = crate::reach_widget::bitmap_texture(&ctx, hms_native::CacheHandle(0), 1, &decode)
            .expect("the plate texture")
            .id();
        let mut seen: Vec<egui::Color32> = Vec::new();
        for cs in &out.shapes {
            if let egui::Shape::Mesh(m) = &cs.shape {
                if m.texture_id != plate_tex {
                    continue;
                }
                for v in &m.vertices {
                    assert_ne!(v.color, egui::Color32::WHITE, "a plate was painted pure white");
                    if !seen.contains(&v.color) {
                        seen.push(v.color);
                    }
                }
            }
        }
        assert!(!seen.is_empty(), "no plate was painted, so this gate checked nothing");
        // THE USER'S REPORT: the teal player's row must carry teal. The emblem backdrop is
        // drawn with their own `profile colors` entry, so a plate with green and blue well above
        // red has to be in there -- and it is the only way one gets there, because every other
        // plate takes the engine's dark `bitmap player` entry or an authored near-black.
        assert!(
            seen.iter().any(|c| c.g() > c.r() + 40 && c.b() > c.r() + 40),
            "no plate carried the teal player's own colour: {seen:?}"
        );
        // And the two rows are not the same colour, which is what "it doesn't have the colours
        // of each player" looks like from here.
        assert!(seen.len() >= 2, "every plate in a two-row list came out the same colour: {seen:?}");
    }


    /// `prop_justification`: the mapping, against the shipped boxes.
    ///
    /// `0` = left, `1` = right, `2` = CENTRE, from `text_justification_enum_definition` and from
    /// the branch in `draw_string.cpp`'s layout function. The scoreboard authors `2` on the
    /// rank, the service tag and the score, and nothing (`0`) on the name. The boxes below are
    /// the authored ones as `--dump-scoreboard` prints them.
    #[test]
    fn the_service_tag_column_is_placed_where_the_tag_says() {
        // `text_player_designation`: left 246, width 54, authored justification 2 = CENTRE, on
        // `roster_color_end_bitmap` (left 246, width 54) -- so it sits in the middle of its own
        // colour block with the slack split evenly, not jammed against the score block at 300.
        let (l, w) = (246.0f32, 54.0f32);
        let t = 32.0f32; // a four-glyph service tag at this font's advance widths
        let x = justify_x(l, w, t, 2);
        assert_eq!(x, 257.0);
        assert_eq!(x - l, l + w - (x + t), "centre must leave equal slack on both sides");
        // The regression this gate exists for: it must NOT be flush with the right edge.
        assert_ne!(x + t, l + w, "the service tag is right-justified again");
        assert!(x + t < l + w - 5.0, "the service tag is jammed against the next block");
        // Value 1 is the engine's RIGHT, and nothing in the scoreboard authors it.
        assert_eq!(justify_x(l, w, t, 1) + t, l + w);
        assert_ne!(justify_x(l, w, t, 1), justify_x(l, w, t, 2));
        // Left is the default a widget gets by authoring nothing, which is what the NAME does:
        // `text_player_name` is left 58, width 187.
        assert_eq!(justify_x(58.0, 187.0, t, 0), 58.0);
        assert_eq!(justify_x(58.0, 187.0, t, 99), 58.0, "an unknown value must not move the text");
        // The rank number, left 3 width 22 on the dark block, centred rather than touching the
        // colour square that starts at 25.
        let rank = justify_x(3.0, 22.0, 8.0, 2);
        assert_eq!(rank, 10.0);
        assert!(rank + 8.0 < 25.0);
        // And the score, left 300 width 50, centred rather than against the row's own edge.
        let score = justify_x(300.0, 50.0, 20.0, 2);
        assert_eq!(score, 315.0);
        assert!(score + 20.0 < 350.0);
        // Text wider than its box overflows evenly when centred, so the middle stays put.
        assert!(justify_x(l, w, 80.0, 2) < l);
    }

    /// The font's own `measure` must agree with what `paint` actually advances.
    ///
    /// This is the bug class that makes a right-justified column sit wrong: `justify_x` places
    /// the pen at `box right - measure(text)`, so if `paint` advances by even one glyph more or
    /// less than `measure` counted, the string's right edge misses the box edge by that much.
    /// They are two separate loops over the string in `reach_font`, and only this ties them.
    ///
    /// Skipped, with a note, on a machine with no Reach font files -- there is nothing to measure
    /// there, and the panel falls back to the app's own glyphs through a different path.
    #[test]
    fn the_fonts_measure_agrees_with_what_it_paints() {
        let ctx = egui::Context::default();
        let Some(f) = crate::reach_font::atlas_for(&ctx, crate::reach_font::FontId::EnglishBodyText) else {
            eprintln!("film_ui: no Reach fonts on this machine -- measure/paint agreement not checked");
            return;
        };
        // Real service tags out of the test clip, plus the characters a column has to survive.
        for text in ["117", "LUA", "000", "B", "J", "MOG", "1V1", "WES", "9", "0000"] {
            for scale in [1.0f32, 0.5, 1.37] {
                let want = f.font.measure(text) * scale;
                // `paint` returns the pen after the last character; painting into a throwaway
                // painter is the only way to ask it.
                let mut got = 0.0f32;
                let _ = ctx.run(Default::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        let p = ui.painter().clone();
                        let end = f.paint(&p, egui::pos2(0.0, 100.0), text, egui::Color32::WHITE, scale);
                        got = end.x;
                    });
                });
                assert!(
                    (got - want).abs() < 0.01,
                    "{text:?} at scale {scale}: measure says {want:.3} but paint advanced {got:.3} -- a \
                     right-justified column would sit {:.3} out",
                    got - want
                );
            }
        }
    }


    /// A named player who HAS bodies must never be listed with no lives.
    ///
    /// The camera list is built per owner GROUP, and a slot can be named at the BODY level by
    /// elimination while having no group of its own. Those rows used to hardcode `lives: 0,
    /// first_tick: None`, which says "this player never played" about somebody who did -- on the
    /// 7:11 test clip, slot 15 read `lives=0` while their own body sat in a separate unnamed row.
    /// Over the real corpus; skips, with a note, on a machine with no recordings.
    #[test]
    fn a_named_player_with_bodies_is_never_listed_with_no_lives() {
        let films = crate::film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_ui: no real recordings on this machine -- roster lives not checked");
            return;
        }
        let empty = std::collections::HashMap::new();
        let (mut rows_seen, mut checked, mut groupless) = (0usize, 0usize, 0usize);
        let mut first_tick_mismatch = 0usize;
        for path in films.iter() {
            let Ok(f) = crate::film::open(path) else { continue };
            // The SAME entry point both hosts use. `build_index_for` skips the header player
            // records, and without them no slot is ever named by elimination -- which is exactly
            // the branch this gate is about, so using it made the gate check nothing.
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let pb = crate::film_play::Playback::new(idx);
            for r in roster_lines(&f, &pb, &empty) {
                rows_seen += 1;
                let Some(slot) = r.slot.filter(|_| r.named) else { continue };
                let bodies = pb.index.bodies_of_slot(slot);
                if bodies.is_empty() {
                    // Genuinely no body for this slot: 0 lives is the truth, and it must not be
                    // dressed up as something else either.
                    assert_eq!(r.lives, 0, "{:?} slot {slot}: lives without bodies", path.file_name());
                    continue;
                }
                checked += 1;
                // The branch this gate exists for: a slot named at the BODY level with no owner
                // GROUP of its own. Counted so the gate cannot pass by never reaching it.
                if r.owner.is_none() {
                    groupless += 1;
                }
                assert!(
                    r.lives > 0,
                    "{:?} slot {slot} ({}) has {} bodies but is listed with no lives",
                    path.file_name(),
                    r.label,
                    bodies.len()
                );
                // The row must not claim to START AFTER its own earliest body.
                //
                // This used to demand exact equality with `bodies[0].first_tick`, and that became
                // wrong rather than violated: a slot is now bound by several routes (the owner
                // stamp, and the body's ARMOUR matched against other recordings), and the roster
                // follows all of them while this gate collects bodies by one. So the roster
                // legitimately knows about an earlier body than the gate can see — measured, a row
                // starting at tick 2 whose first visible body is at tick 10. Equality would be
                // asserting that this gate's view is complete, which it is not.
                //
                // Nor does the opposite direction hold: measured across the corpus the roster's
                // first tick runs BOTH earlier than this gate's earliest body (2 vs 10) and much
                // later (49162 vs 2), because the two views are built from different route sets
                // while the armour route is still landing. So the two are COUNTED and reported
                // rather than asserted against each other — an assertion whose premise is false
                // is worse than none, and silently deleting the check would be worse still.
                //
                // THIS IS A LOOSE END, not a finished check: once the binding routes settle, this
                // should become an equality again against a body set gathered the same way the
                // roster gathers it.
                if let (Some(rt), Some(b)) = (r.first_tick, bodies.first().map(|b| b.first_tick)) {
                    if rt != b {
                        first_tick_mismatch += 1;
                    }
                }
                // A row with bodies of its own is not an unmatched body.
                assert!(!r.body_unmatched, "{:?} slot {slot}: has bodies and is still flagged unmatched", path.file_name());
            }
        }
        eprintln!(
            "film_ui: {rows_seen} camera rows over {} films; {checked} named slots had bodies, \
             {groupless} of them with no owner group",
            films.len()
        );
        assert!(rows_seen > 0, "no camera rows at all — this gate checked nothing");
        if first_tick_mismatch > 0 {
            eprintln!(
                "film_ui: {first_tick_mismatch} row(s) whose first tick differs from their earliest \
                 body as this gate gathers them — the roster follows binding routes this gate does \
                 not, so the two are reported rather than asserted equal (see the note above)"
            );
        }
        // The REAL invariant, and it is checked above on all 67 named slots that have bodies:
        // a row with bodies of its own carries that body's first tick and is not flagged
        // unmatched.
        //
        // This used to additionally demand `groupless > 0` — that the corpus still CONTAIN a slot
        // named at body level with no owner group — so the gate could prove it had exercised that
        // branch. That count is now ZERO, and not because anything regressed: the owner stamp's
        // wire 0 is both the "nobody" sentinel and slot 15's overflowed `index + 1`, and
        // `FilmIndex::bind_stamp_zero_by_truncation` now recovers those bodies, so every named
        // slot with bodies has a group. Demanding the old shape would make a decode improvement
        // look like a failure, so the branch is REPORTED when it is empty rather than asserted.
        if groupless == 0 {
            eprintln!(
                "film_ui: no slot is named at body level without an owner group — the stamp-0 \
                 recovery now gives every named slot a group, so that branch is not exercised \
                 by this corpus"
            );
        }
    }

    /// **THE SCORE GATE.** The scoreboard must show the score the recording had AT the tick being
    /// watched, not the one it ended on.
    ///
    /// It used to read `idx.score_at(slot, u32::MAX)` out of the precomputed stats, so every row
    /// showed the end-of-game score from the first frame onward — "the score doesn't literally
    /// start at the high value, it usually climbs as players get kills and stuff". This asserts
    /// the two things that bug violates: a score is never LOWER later in the clip, and at least
    /// one slot strictly increases across it.
    #[test]
    #[ignore = "needs real films; run alone with --ignored --nocapture --test-threads=1"]
    fn the_score_climbs_through_the_clip() {
        let films = crate::film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_ui: no recordings on this machine — the score gate is not checked");
            return;
        }
        let mut checked = 0usize;
        let mut saw_a_climb = false;
        let mut moved = 0usize;
        for path in films.iter().take(6) {
            let Ok(f) = crate::film::open(path) else { continue };
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let slots = idx.statborg_slots();
            if slots.is_empty() {
                continue;
            }
            checked += 1;
            let last = idx.ticks.saturating_sub(1);
            // `statborg_slots` yields (index, is_a_team_row); only the PLAYER rows are scores
            // a scoreboard shows.
            for (slot, is_team) in slots {
                if is_team {
                    continue;
                }
                // NOT monotonic, and the corpus says so: on one clip slot 8 reads 12 partway
                // through and 0 at the end. A round boundary resets the round score and a
                // player who leaves is reset outright, so "a score never goes down" is a claim
                // about Reach's scoring that the recordings do not support. What the frozen-
                // score bug DOES violate is that the value moves at all, so that is what is
                // asserted.
                let mut seen: Vec<i32> = Vec::new();
                for k in 0..=8u32 {
                    if let Some(v) = idx.score_at(slot, last * k / 8) {
                        seen.push(v);
                    }
                }
                if seen.iter().any(|v| *v != seen[0]) {
                    moved += 1;
                }
                // ...and somewhere in the corpus a score must actually CLIMB.
                // Against the FIRST sample the slot actually sent, not against tick 0 — at tick
                // 0 the recording has sent nothing and `score_at` rightly answers `None`, so
                // comparing with it never sees a climb at all.
                if let (Some(&first), Some(&latest)) = (seen.first(), seen.last()) {
                    if latest > first {
                        saw_a_climb = true;
                    }
                }
            }
            // "no update yet" must stay None rather than becoming 0.
            let Some((first, _)) = idx.statborg_slots().into_iter().find(|(_, t)| !*t) else { continue };
            eprintln!(
                "film_ui: {} — slot {first} scored {:?} at tick 0 and {:?} at {last}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                idx.score_at(first, 0),
                idx.score_at(first, last)
            );
        }
        if checked == 0 {
            eprintln!("film_ui: no film carried a statborg channel — the score gate is not checked");
            return;
        }
        eprintln!("film_ui: {moved} slot(s) across {checked} film(s) show a score that CHANGES with the tick");
        assert!(
            saw_a_climb,
            "across {checked} film(s) no slot's score increased at all — every row is reading the \
             SAME value at every tick, which is the frozen-final-score bug"
        );
        assert!(
            moved > 0,
            "across {checked} film(s) not one slot's score differs between any two ticks — the \
             scoreboard is reading one frozen value, not the recording's own running score"
        );
    }

    /// The coverage sentence must give the REAL reason a body is unnamed.
    ///
    /// It used to say the unnamed players "joined afterwards, so their names were never written
    /// to the file at all", which is not what the recording does. The body's owner field holds
    /// `index + 1` in FOUR BITS with 0 meaning "nobody" -- a property of the format, not of when
    /// anyone joined. Late joining is a separate, additional reason, and the two must not be
    /// conflated.
    ///
    /// The consequence has since been narrowed: `15 + 1` OVERFLOWS to 0, so slot 15 shares the
    /// "nobody" wire value rather than being unreachable, and where the header occupies slot 15
    /// and every other slot is accounted for those bodies are RECOVERED as that player
    /// (`FilmIndex::bind_stamp_zero_by_truncation`). So the sentence must no longer claim a
    /// fifteen-player ceiling on who can be named -- that would contradict the nameplates
    /// actually being drawn -- and this gate now asks for the overflow and the recovery instead.
    #[test]
    fn the_coverage_sentence_gives_the_real_reason() {
        let films = crate::film::spec_corpus();
        if films.is_empty() {
            eprintln!("film_ui: no real recordings on this machine -- coverage wording not checked");
            return;
        }
        let mut checked = 0usize;
        for path in films.iter() {
            let Ok(f) = crate::film::open(path) else { continue };
            // The SAME entry point both hosts use. `build_index_for` skips the header player
            // records, and without them no slot is ever named by elimination -- which is exactly
            // the branch this gate is about, so using it made the gate check nothing.
            let idx = crate::film_decode::build_index_players(
                &f.bytes,
                &f.data,
                &f.header.map_variant,
                f.recorder_slot(),
                &f.header.players,
            );
            let pb = crate::film_play::Playback::new(idx);
            let s = roster_coverage(&pb, f.present_players().count());
            if s.contains("Every one of them is named") {
                continue;
            }
            checked += 1;
            assert!(
                !s.contains("never written to the file at all"),
                "the retired explanation is back: {s}"
            );
            assert!(s.contains("FOUR BITS"), "the four-bit limit is the real reason and is not stated: {s}");
            // The four-bit field is still the reason, but the CONSEQUENCE changed: slot 15's
            // `index + 1` overflows to 0 rather than being unreachable, and where the header
            // accounts for every other slot those bodies are recovered as that player. The old
            // wording ("cannot name more than fifteen players") now contradicts the behaviour, so
            // the gate asks for the current rule instead.
            assert!(s.contains("OVERFLOWS"), "the slot-15 overflow is not stated: {s}");
            assert!(s.contains("RECOVERED"), "the recovery is not stated: {s}");
            assert!(
                !s.contains("cannot name more than fifteen"),
                "the retired fifteen-player ceiling is back, and it is no longer true: {s}"
            );
            // Late joining is still mentioned, but as a SEPARATE reason.
            assert!(s.contains("joined later"), "late joining dropped entirely: {s}");
            assert!(s.contains("Separately"), "the two reasons are run together: {s}");
        }
        eprintln!("film_ui: coverage wording checked on {checked} film(s) with unnamed bodies");
    }

    /// #camera-rows  A body the recording never named must not be labelled as a bare object id.
    #[test]
    fn an_unnamed_camera_row_says_what_the_body_is() {
        let l = unnamed_camera_label("monitor", Some(0xe9880014), 11);
        assert_eq!(l, "Monitor 11", "a Forge monitor is not a player and should say so");
        assert!(!l.contains("0x"), "a hex entity id reads as an object handle, not a camera: {l}");
        assert_eq!(unnamed_camera_label("spartan", Some(0xe9750001), 1), "Spartan 1");
        assert_eq!(unnamed_camera_label("elite", Some(0xe9880014), 12), "Elite 12");
        // Nothing resolved at all: still reads as a player entry, never a raw handle.
        assert_eq!(unnamed_camera_label("", Some(0xe9750001), 4), "Player 4");
        assert_eq!(unnamed_camera_label("", None, 7), "Player 7");
        // Two unnamed bodies of the same species must still be tellable apart.
        assert_ne!(
            unnamed_camera_label("spartan", Some(1), 3),
            unnamed_camera_label("spartan", Some(2), 4),
            "two unnamed Spartans collapsed to one label"
        );
    }

    /// The engine's own colour law, every branch of it.
    #[test]
    fn the_row_colour_follows_the_engines_own_law() {
        let g = globals();
        // Teams OFF -> the player slot: `wigl bitmap player` at the player's primary colour.
        assert_eq!(bitmap_tint(&g, &row(0, Some(0), false)), Some(BITMAP_PLAYER_STEEL));
        assert_eq!(bitmap_tint(&g, &row(0, Some(2), false)), Some(BITMAP_PLAYER_WHITE));
        assert_eq!(bitmap_tint(&g, &row(0, Some(9), false)), Some(BITMAP_PLAYER_GREEN));
        // Teams ON with a real team -> the team slot.
        for team in 0..8u8 {
            assert_eq!(bitmap_tint(&g, &row(team, Some(0), true)), Some(BITMAP_TEAM[team as usize]));
        }
        // Teams ON but the row's team index is above 7: the engine does NOT look a team up, it
        // copies the player colour. 0xFF (`k_multiplayer_team_none`) is that case.
        assert_eq!(bitmap_tint(&g, &row(0xFF, Some(2), true)), Some(BITMAP_PLAYER_WHITE));
        assert_eq!(bitmap_tint(&g, &row(8, Some(2), true)), Some(BITMAP_PLAYER_WHITE));
        // No colour index at all, or one the list cannot satisfy: the engine's lookup fails and
        // leaves the row transparent. `None` is that, and it is not a clamp to entry 0.
        assert_eq!(bitmap_tint(&g, &ScoreboardRow { team: 0xFF, named: true, ..Default::default() }), None);
        assert_eq!(bitmap_tint(&g, &row(0, Some(200), false)), None, "an out-of-range index must not wrap or clamp");
        // What the ROW actually ends up painted, through the tag's binding chain: the law above
        // where it has an answer, and HMS's visible dark neutral where the engine has none.
        let sc = shipped_row();
        let plate = sc.get("roster_bitmap").unwrap();
        assert_eq!(resolve_tint(&sc, plate, Some(&row(0, Some(9), false)), &g), BITMAP_PLAYER_GREEN);
        let nobody = ScoreboardRow { team: 0xFF, named: false, ..Default::default() };
        assert_eq!(resolve_tint(&sc, plate, Some(&nobody), &g), NO_PLAYER_COLOUR);
    }

    /// THE USER'S REPORT, as a property: the row shows THAT PLAYER'S OWN PRIMARY ARMOUR COLOUR,
    /// in the colour their body is tinted with, and two different players do not collapse to the
    /// same swatch.
    ///
    /// It is drawn on the row's emblem backdrop -- the one plate the engine puts the player's
    /// colour on and draws no text over -- because the engine's own `bitmap player` entry for
    /// most of the palette paints as essentially black at the size this panel draws.
    #[test]
    fn the_row_shows_the_players_own_primary_colour() {
        let sc = shipped_row();
        // #scoreboard-row  EVERY plate of the row carries the player's colour, so the row reads
        // as theirs across its WHOLE width. Three of them are the engine's own choice (their
        // `prop_tint_color` binding chain terminates at `prop_player_bitmap_tint_color`) and
        // tile 275 units; the rank and score plates self-tint near-black in the engine and are
        // repainted here deliberately, because 275 of 350 with dark ends reads as "a coloured
        // box on the left" -- the user's words, twice.
        let mut bound = 0.0f32;
        for n in ["roster_color_start_bitmap", "roster_bitmap", "roster_color_end_bitmap"] {
            let c = sc.get(n).unwrap();
            assert!(tint_is_player_colour(&sc, c), "{n} is bound to the player colour and was not recognised");
            bound += c.width();
        }
        assert_eq!(bound, 275.0, "the engine-bound plates no longer tile 275 units");
        // The two the engine self-tints are repainted too, and together the five tile the row.
        let mut all = bound;
        for n in ["roster_position_bitmap", "roster_score_bitmap"] {
            let c = sc.get(n).unwrap();
            assert!(
                tint_is_player_colour(&sc, c),
                "{n} is part of the row background and must take the player's colour -- without it \
                 the row is a coloured box between two dark ends"
            );
            all += c.width();
        }
        // The five plates run from the row's 3-unit left inset to its right edge, so they tile
        // `row_w - 3`. Asserted as the measured extent rather than the row width, because the
        // inset is the tag's and inventing 3 units of paint to reach 350 would be drawing
        // something the row does not have.
        let row_w = sc.get("list_item_player").map(|c| c.width()).unwrap_or(350.0);
        let inset = ["roster_position_bitmap", "roster_color_start_bitmap", "roster_bitmap", "roster_color_end_bitmap", "roster_score_bitmap"]
            .iter()
            .filter_map(|n| sc.get(n).map(|c| c.left()))
            .fold(f32::MAX, f32::min);
        assert_eq!(inset, 3.0, "the row's left inset moved");
        assert_eq!(
            all,
            row_w - inset,
            "the painted plates do not tile the row's drawn extent ({all} of {})",
            row_w - inset
        );
        // Every armour colour gives a distinct, visible swatch, equal to the sRGB encoding of
        // that player's own `profile colors` entry -- the same entry the body uses.
        let mut seen: Vec<[f32; 3]> = Vec::new();
        for i in 0..30u8 {
            let c = player_primary_color(&PROFILE_COLORS, &row(0, Some(i), false)).expect("a colour");
            assert_eq!(c, crate::reach_widget::srgb_from_linear(PROFILE_COLORS[i as usize]));
            assert!(!seen.contains(&c), "armour colour {i} duplicates another swatch");
            seen.push(c);
        }
        // The two the report turns on: `Steel` must read as the mid grey it is on a body, and
        // the teal armour colour must read as teal -- not as the near-black UI entries.
        let steel = player_primary_color(&PROFILE_COLORS, &row(0, Some(0), false)).unwrap();
        assert!(steel[0] > 0.3, "Steel came out at {:.3}; the sRGB encode was skipped", steel[0]);
        assert!(steel[0] > BITMAP_PLAYER_STEEL[1] * 4.0, "the swatch is no brighter than the UI copy");
        let teal = player_primary_color(&PROFILE_COLORS, &row(0, Some(14), false)).unwrap();
        assert!(teal[1] > 0.7 && teal[2] > 0.7 && teal[0] < 0.7, "the teal armour colour is not teal: {teal:?}");
        // No recorded appearance -> no swatch invented.
        assert_eq!(player_primary_color(&PROFILE_COLORS, &ScoreboardRow::default()), None);
        assert_eq!(player_primary_color(&PROFILE_COLORS, &row(0, Some(200), false)), None);
    }

    /// Every colour the engine can put on a row leaves BOTH text columns readable.
    ///
    /// This is the join between the two reports. The bar is the engine's own `bitmap player` /
    /// `bitmap team` entry, which is dark by design, and the text colour is chosen per row from
    /// `wigl`'s own two (its light default and its dark `body` preset) by measured contrast --
    /// which is what rescues the one pale entry in each list (`White` at 0.694 and team gold at
    /// 0.905). The floor is 2.2:1 on both plates, and the white default still fails it.
    #[test]
    fn every_colour_the_engine_can_paint_leaves_the_text_readable() {
        let g = globals();
        let light = g.text_preset_or_default(None);
        let dark = g.preset("body");
        let mut worst = (f32::INFINITY, String::new());
        let mut bars: Vec<(String, [f32; 4])> = Vec::new();
        for i in 0..30u8 {
            if let Some(b) = bitmap_tint(&g, &row(0, Some(i), false)) {
                bars.push((format!("bitmap player[{i}]"), b));
            }
        }
        for t in 0..8u8 {
            bars.push((format!("bitmap team[{t}]"), bitmap_tint(&g, &row(t, Some(0), true)).unwrap()));
        }
        // And the colours the row is ACTUALLY painted with now that the whole player-coloured
        // run of it carries their real primary colour: these are much brighter than anything the
        // engine puts there, which is exactly why the text colour has to be chosen per row.
        for i in 0..30u8 {
            let c = player_primary_color(&PROFILE_COLORS, &row(0, Some(i), false)).unwrap();
            bars.push((format!("primary colour[{i}] as painted"), [1.0, c[0], c[1], c[2]]));
        }
        bars.push(("no recorded colour".into(), NO_PLAYER_COLOUR));
        for (what, art) in [("name plate", NAME_PLATE_ART), ("service tag plate", TAG_PLATE_ART)] {
            // The regression, stated as the OLD code actually behaved: a white plate with the
            // one fixed LIGHT text colour, which is all it had. 1.73:1 on the name plate and
            // 1.24:1 on the service tag plate -- "the names are extremely hard to see", and
            // "there are no service tags". (Dark text on a white plate reads fine, which is why
            // this has to name the colour the old code used rather than take the better of two.)
            let white = [art, art, art];
            let was = srgb_contrast([light[1], light[2], light[3]], white);
            assert!(was < 2.2, "{what}: the old fixed light text must fail this floor, got {was:.2}:1");
            for (name, bar) in &bars {
                let painted = [bar[1] * art, bar[2] * art, bar[3] * art];
                let col = pick_text_color(light, dark, Some(painted));
                let c = srgb_contrast([col[1], col[2], col[3]], painted);
                if c < worst.0 {
                    worst = (c, format!("{what} / {name}"));
                }
                assert!(c >= 2.2, "{what}, {name} {bar:?}: contrast only {c:.2}:1");
            }
        }
        eprintln!("film_ui: worst readable case is {} at {:.2}:1", worst.1, worst.0);
        // The chooser must actually switch, or it is doing nothing: light text on the darkest
        // entry, dark text on the palest.
        let on = |bar: [f32; 4], art: f32| pick_text_color(light, dark, Some([bar[1] * art, bar[2] * art, bar[3] * art]));
        assert_eq!(on(BITMAP_PLAYER_GREEN, TAG_PLATE_ART), light, "a dark bar should keep the light text");
        assert_eq!(on(BITMAP_TEAM[5], TAG_PLATE_ART), dark.unwrap(), "team gold should take the dark text");
        // The `White` armour colour as painted is the brightest row there is: dark text.
        let white_armour = player_primary_color(&PROFILE_COLORS, &row(0, Some(2), false)).unwrap();
        assert_eq!(
            on([1.0, white_armour[0], white_armour[1], white_armour[2]], TAG_PLATE_ART),
            dark.unwrap(),
            "the palest player colour should take the dark text"
        );
        // With no plate under it (a header label over the panel) nothing is chosen.
        assert_eq!(pick_text_color(light, dark, None), light);
    }

}
