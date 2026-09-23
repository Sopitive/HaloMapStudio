//! Hovering an entry of the properties panel's TEAM / COLOR dropdowns PREVIEWS the
//! change on the selected object(s): the selection wireframe is hidden and the hovered team /
//! colour is pushed into the render path transiently (no `ObjMeta` mutation, no undo step).
//!
//! The preview only happens when picking that entry would actually CHANGE the object's effective
//! change colour. The engine rule (scene::forge_tint): an explicit colour override (0..7) wins;
//! otherwise a real team (0..7) supplies the colour; otherwise the object is neutral (no tint).
//! So with the override set to red, hovering the team list previews nothing (teams cannot
//! influence the colour); with the override on "inherit", every team entry previews.
//!
//! Hide rule (no flicker in the gaps between entries): while a popup is OPEN and the pointer is
//! anywhere inside its rect, the wireframe stays hidden as long as at least one entry of THAT popup
//! would change the effective colour; the previewed colour is the LAST hovered entry and is kept
//! while the pointer crosses the padding. Both revert when the pointer leaves the popup rect or the
//! popup closes (`PopupState::step`).
//!
//! This module is the PURE part (the predicate, the preview map, the popup state machine); `App`
//! owns the transient state (`hover_preview`), merges the map into `forge_colors` in `tick_scene`,
//! and hides the wireframe through `SceneRenderer::set_highlight_hidden`.

use std::collections::HashMap;

/// A dropdown entry under the cursor.
/// `Team` carries the raw .mvar team byte (0..7 = colour teams, 8 = neutral, 0xFF = none);
/// `Color` the .mvar colour override (-1 = inherit the team colour, 0..7 = explicit palette index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoverEntry {
    Team(u8),
    Color(i32),
}

/// Which dropdown a hover belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Popup {
    Team,
    Color,
}

impl Popup {
    /// Every entry the popup lists, in menu order (team: neutral, the 8 colours, none; colour:
    /// inherit, the 8 colours) -- what "any entry would change the colour" is evaluated over.
    pub fn entries(self) -> Vec<HoverEntry> {
        match self {
            Popup::Team => std::iter::once(HoverEntry::Team(8))
                .chain((0..8u8).map(HoverEntry::Team))
                .chain(std::iter::once(HoverEntry::Team(0xFF)))
                .collect(),
            Popup::Color => std::iter::once(HoverEntry::Color(-1)).chain((0..8i32).map(HoverEntry::Color)).collect(),
        }
    }
}

/// The per-frame popup/hover state: which popup the pointer is inside (None = no open popup under
/// the pointer) and the entry whose colour is being previewed (the last hovered one of that popup).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PopupState {
    pub inside: Option<Popup>,
    pub entry: Option<HoverEntry>,
}

impl PopupState {
    /// Advance by one frame. `inside` = the open popup whose rect contains the pointer; `hovered` =
    /// the entry under the pointer this frame (None in the gaps / padding). Leaving the popup (or
    /// it closing) drops the entry; a gap keeps the last hovered entry; a new entry replaces it.
    /// Returns true when `inside` changed (the caller re-evaluates the hide rule then).
    pub fn step(&mut self, inside: Option<Popup>, hovered: Option<HoverEntry>) -> bool {
        let entered = inside != self.inside;
        self.entry = match inside {
            None => None,
            Some(p) => match hovered.filter(|h| h.popup() == p) {
                Some(h) => Some(h),
                None if entered => None,
                None => self.entry.filter(|e| e.popup() == p),
            },
        };
        self.inside = inside;
        entered
    }
}

/// Would ANY entry of `popup` change the effective colour of at least one selected object that
/// visibly responds to it? Decides whether the wireframe hides while the pointer is inside that
/// popup (override set to red + team popup -> false: no team entry can change the colour).
pub fn popup_would_change(
    sel: &[(u32, u8, i32)],
    popup: Popup,
    responds: impl Fn(u32) -> Option<bool>,
) -> bool {
    popup.entries().into_iter().any(|e| !preview_map(sel.iter().copied(), e, &responds).is_empty())
}

impl HoverEntry {
    /// The dropdown this entry lives in.
    pub fn popup(self) -> Popup {
        match self {
            HoverEntry::Team(_) => Popup::Team,
            HoverEntry::Color(_) => Popup::Color,
        }
    }

    /// `team red` / `color 3` / `color inherit` for the status line.
    pub fn describe(self) -> String {
        match self {
            HoverEntry::Team(0xFF) => "team none".to_string(),
            HoverEntry::Team(8) => "team neutral".to_string(),
            HoverEntry::Team(t) => format!("team {}", crate::scene::FORGE_COLOR_NAMES[(t as usize).min(10)]),
            HoverEntry::Color(c) if c < 0 => "color inherit".to_string(),
            HoverEntry::Color(c) => format!("color {}", crate::scene::FORGE_COLOR_NAMES[(c as usize).min(10)]),
        }
    }
}

/// The palette index the render resolves for a placement: the explicit colour override wins, else
/// a real team (0..7), else `None` = neutral (the self-illum keeps its authored colour).
pub fn effective_change_color(team: u8, color: i32) -> Option<u8> {
    if (0..8).contains(&color) {
        Some(color as u8)
    } else if team < 8 {
        Some(team)
    } else {
        None
    }
}

/// The (team, color) pair a placement would carry after picking `entry`.
pub fn apply_entry(team: u8, color: i32, entry: HoverEntry) -> (u8, i32) {
    match entry {
        HoverEntry::Team(t) => (t, color),
        HoverEntry::Color(c) => (team, c),
    }
}

/// Would picking `entry` change this placement's EFFECTIVE colour? `Some(new (team, color))` when
/// it would, `None` when the render would come out identical (nothing to preview).
pub fn preview_for(team: u8, color: i32, entry: HoverEntry) -> Option<(u8, i32)> {
    let (t, c) = apply_entry(team, color, entry);
    (effective_change_color(t, c) != effective_change_color(team, color)).then_some((t, c))
}

/// The .mvar colour byte the render path keys on (`forge_colors` values): -1 inherit -> 0xFF.
pub fn color_byte(color: i32) -> u8 {
    if color < 0 { 0xFF } else { color as u8 }
}

/// Build the transient preview map for a selection: `datum -> (team, colour byte)` for every
/// selected object (`sel` yields `(datum, team, color)`) whose effective colour `entry` would change
/// AND whose render model visibly uses the change colour (`responds(datum)`: `Some(false)` = no
/// change-colour parts, `None` = not decoded yet -> no preview either). An EMPTY map means "nothing
/// would change": keep the wireframe, preview nothing.
pub fn preview_map(
    sel: impl IntoIterator<Item = (u32, u8, i32)>,
    entry: HoverEntry,
    responds: impl Fn(u32) -> Option<bool>,
) -> HashMap<u32, (u8, u8)> {
    let mut out = HashMap::new();
    for (datum, team, color) in sel {
        let Some((t, c)) = preview_for(team, color, entry) else { continue };
        if responds(datum) != Some(true) {
            continue;
        }
        out.insert(datum, (t, color_byte(c)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: u8 = 0;
    const BLUE: u8 = 1;
    const NEUTRAL: u8 = 8;
    const NONE: u8 = 0xFF;

    #[test]
    fn effective_colour_rule() {
        assert_eq!(effective_change_color(RED, -1), Some(0), "inherit -> team colour");
        assert_eq!(effective_change_color(RED, 3), Some(3), "override wins over the team");
        assert_eq!(effective_change_color(NEUTRAL, -1), None, "neutral team, inherit -> no tint");
        assert_eq!(effective_change_color(NONE, -1), None, "no team, inherit -> no tint");
        assert_eq!(effective_change_color(NONE, 5), Some(5), "override on an unset team still tints");
    }

    #[test]
    fn team_hover_only_previews_when_the_colour_inherits() {
        // colour override = red: hovering any team must NOT preview (teams cannot change the colour).
        for t in [RED, BLUE, 7, NEUTRAL, NONE] {
            assert_eq!(preview_for(BLUE, 0, HoverEntry::Team(t)), None, "override red, hover team {t}");
        }
        // colour = inherit: hovering a DIFFERENT team previews the team colour.
        assert_eq!(preview_for(RED, -1, HoverEntry::Team(BLUE)), Some((BLUE, -1)));
        assert_eq!(preview_for(RED, -1, HoverEntry::Team(NEUTRAL)), Some((NEUTRAL, -1)), "red -> neutral loses the tint");
        assert_eq!(preview_for(NEUTRAL, -1, HoverEntry::Team(BLUE)), Some((BLUE, -1)));
        // ... but hovering the team it already has, or neutral<->none (both untinted), changes nothing.
        assert_eq!(preview_for(RED, -1, HoverEntry::Team(RED)), None);
        assert_eq!(preview_for(NEUTRAL, -1, HoverEntry::Team(NONE)), None);
        assert_eq!(preview_for(NONE, -1, HoverEntry::Team(NEUTRAL)), None);
    }

    #[test]
    fn colour_hover_previews_when_it_differs_from_the_effective_colour() {
        assert_eq!(preview_for(RED, -1, HoverEntry::Color(1)), Some((RED, 1)), "inherit red -> explicit blue");
        assert_eq!(preview_for(RED, -1, HoverEntry::Color(0)), None, "explicit red == inherited red: no visible change");
        assert_eq!(preview_for(RED, 3, HoverEntry::Color(-1)), Some((RED, -1)), "back to inherit previews the team colour");
        assert_eq!(preview_for(RED, 0, HoverEntry::Color(-1)), None, "inherit would resolve to the same red");
        assert_eq!(preview_for(NEUTRAL, -1, HoverEntry::Color(4)), Some((NEUTRAL, 4)));
        assert_eq!(preview_for(NEUTRAL, 4, HoverEntry::Color(-1)), Some((NEUTRAL, -1)), "inherit on a neutral team drops the tint");
        assert_eq!(preview_for(NEUTRAL, 4, HoverEntry::Color(4)), None);
    }

    #[test]
    fn preview_map_filters_by_change_and_by_model_response() {
        let sel = vec![
            (1u32, RED, -1),      // inherits -> previews blue
            (2u32, RED, 0),       // override red -> team hover changes nothing
            (3u32, BLUE, -1),     // already blue -> nothing
            (4u32, NEUTRAL, -1),  // previews blue, but the model has no change-colour parts
            (5u32, NONE, -1),     // previews blue, but the model is not decoded yet
        ];
        let responds = |d: u32| match d {
            4 => Some(false),
            5 => None,
            _ => Some(true),
        };
        let m = preview_map(sel.clone(), HoverEntry::Team(BLUE), responds);
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(&1), Some(&(BLUE, 0xFF)));
        // Same selection, hovering the colour list: the override object DOES respond to a colour.
        let m = preview_map(sel, HoverEntry::Color(2), responds);
        assert_eq!(m.get(&1), Some(&(RED, 2)));
        assert_eq!(m.get(&2), Some(&(RED, 2)));
        assert_eq!(m.get(&3), Some(&(BLUE, 2)));
        assert_eq!(m.len(), 3);
        // Nothing hovered would change -> empty map -> the wireframe stays.
        let m = preview_map(vec![(2u32, RED, 0)], HoverEntry::Team(BLUE), responds);
        assert!(m.is_empty());
    }

    #[test]
    fn popup_would_change_follows_the_override_rule() {
        let yes = |_: u32| Some(true);
        // colour override red: NO team entry can change the colour -> the team popup never hides.
        assert!(!popup_would_change(&[(1, BLUE, 0)], Popup::Team, yes));
        // ... but the colour popup can (blue, inherit->blue team, ...).
        assert!(popup_would_change(&[(1, BLUE, 0)], Popup::Color, yes));
        // inherit: the team popup hides (any other team changes it), so does the colour popup.
        assert!(popup_would_change(&[(1, RED, -1)], Popup::Team, yes));
        assert!(popup_would_change(&[(1, RED, -1)], Popup::Color, yes));
        // a model with no change-colour parts never hides anything.
        assert!(!popup_would_change(&[(1, RED, -1)], Popup::Team, |_| Some(false)));
        assert!(!popup_would_change(&[(1, RED, -1)], Popup::Color, |_| None));
        // empty selection: nothing to change.
        assert!(!popup_would_change(&[], Popup::Team, yes));
    }

    #[test]
    fn popup_state_machine_keeps_the_preview_across_gaps() {
        let (a, b) = (HoverEntry::Team(BLUE), HoverEntry::Team(2));
        let mut st = PopupState::default();
        // enter the popup over the padding: inside, no entry yet -> (wireframe hides, colour real)
        assert!(st.step(Some(Popup::Team), None));
        assert_eq!(st, PopupState { inside: Some(Popup::Team), entry: None });
        // hover A -> preview A
        assert!(!st.step(Some(Popup::Team), Some(a)));
        assert_eq!(st.entry, Some(a));
        // move to a gap -> still inside, still A (no flicker, no revert)
        assert!(!st.step(Some(Popup::Team), None));
        assert_eq!(st, PopupState { inside: Some(Popup::Team), entry: Some(a) });
        // hover B -> preview B
        st.step(Some(Popup::Team), Some(b));
        assert_eq!(st.entry, Some(b));
        // leave the popup rect (or it closes) -> restore everything
        assert!(st.step(None, None));
        assert_eq!(st, PopupState::default());
        // re-entering starts clean (the old B must not come back)
        st.step(Some(Popup::Team), None);
        assert_eq!(st.entry, None);
        // a stale hover from the OTHER popup is ignored
        st.step(Some(Popup::Color), Some(a));
        assert_eq!(st, PopupState { inside: Some(Popup::Color), entry: None });
        st.step(Some(Popup::Color), Some(HoverEntry::Color(3)));
        assert_eq!(st.entry, Some(HoverEntry::Color(3)));
    }

    #[test]
    fn popup_entries_cover_the_menus() {
        assert_eq!(Popup::Team.entries().len(), 10);
        assert_eq!(Popup::Team.entries()[0], HoverEntry::Team(8));
        assert_eq!(Popup::Team.entries()[9], HoverEntry::Team(0xFF));
        assert_eq!(Popup::Color.entries().len(), 9);
        assert_eq!(Popup::Color.entries()[0], HoverEntry::Color(-1));
    }
}
