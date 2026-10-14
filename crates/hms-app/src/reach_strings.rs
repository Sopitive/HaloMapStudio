//! #reach-strings  Reach's LOCALIZED strings, resolved for the first time.
//!
//! Reach authors every piece of in-game text as a `string_id` inside a
//! `multilingual_unicode_string_list` (`unic`) tag — `ui\hud\hud_messages`,
//! `globals\game_engine_text`, `ui\cui\strings\forge_strings` and twenty-five more in a shipped
//! cache. **MCC strips the text out of those tags** and serves it from
//! `<MCC>/data/ui/Localization/<LANG>_<Title>.bin` instead, which is why
//! [`crate::reach_hud`] has until now refused to draw any chud text widget and reported the
//! string id as unresolved.
//!
//! # The key convention, measured
//!
//! A `.bin` table is keyed by the CRC-32 of the UPPERCASED key string
//! ([`crate::h4::localization::key_hash`]), and the key MCC uses for a Reach string is
//!
//! ```text
//!     <unic tag path, with backslashes>  '_'  <string_id name>
//! ```
//!
//! so the Warthog driver's seat prompt is `ui\hud\hud_messages_warthog_enter_driver`. The four
//! that pinned the convention down, found by taking the English out of `EN_HaloReach.bin` first
//! and hunting the key that hashes to it:
//!
//! ```text
//!   ui\hud\hud_messages_warthog_enter_driver     0x6871118E  "Hold  to drive Warthog"
//!   ui\hud\hud_messages_warthog_enter_passenger  0x6090A24A  "Hold  to ride in Warthog"
//!   ui\hud\hud_messages_warthog_enter_gunner     0xB1277F31  "Hold  to operate Warthog turret"
//!   ui\hud\hud_messages_warthog_board            0xD0CE2A52  "Hold  to board enemy Warthog"
//! ```
//!
//! and the string ids on the left are exactly the `enter seat string` values
//! [`crate::reach_seats`] reads out of the `vehi` tags — they are not invented, and the same ids
//! appear verbatim in HREK's own
//! `tags/ui/hud/hud_messages.multilingual_unicode_string_list`.
//!
//! **The tag path is read from the CACHE, not hardcoded.** [`ReachStrings::load`] takes every
//! `unic` tag's own name and tries each one, so a string in `globals\game_engine_text` or
//! `ui\cui\strings\forge_strings` resolves by the same rule with no table of paths in HMS.
//!
//! # The glyphs
//!
//! The prompts embed PRIVATE-USE codepoints (`U+E446` is the action button, `U+E108` a d-pad
//! direction). Those map into MCC's own `font_package_icon*.bin`, which HMS does not read — the
//! same gap that stops the weapon schematic drawing. [`substitute_glyphs`] replaces each one with
//! a caller-supplied label (the player's actual bound key, from [`crate::keybinds`]) and RETURNS
//! the codepoints it replaced, so a caller can say which glyph it stood in for instead of
//! silently printing a tofu box.

use hms_native::{CacheHandle, NativeDll};

/// The Unicode private-use area. Every glyph MCC's icon font supplies sits in here.
pub const PRIVATE_USE: std::ops::RangeInclusive<u32> = 0xE000..=0xF8FF;

/// `U+E446` — the action / "X" button glyph every vehicle-entry prompt opens with.
pub const GLYPH_ACTION: char = '\u{E446}';

/// One resolved string: where it came from and what it says.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    /// The `unic` tag whose name built the key, e.g. `ui\hud\hud_messages`.
    pub unic: String,
    /// The localization key, e.g. `ui\hud\hud_messages_warthog_enter_driver`.
    pub key: String,
    /// The text, exactly as the table holds it — private-use glyphs and all.
    pub text: String,
}

/// The `unic` tag names a cache carries, in cache order. Cheap to build and cheap to hold.
#[derive(Clone, Debug, Default)]
pub struct ReachStrings {
    /// Every `unic` tag's path, lower-cased, backslash-separated, as the cache states it.
    pub unics: Vec<String>,
}

impl ReachStrings {
    /// Take the cache's own `unic` tag names. No path is written into HMS.
    pub fn load(native: &NativeDll, cache: CacheHandle) -> ReachStrings {
        let mut unics: Vec<String> = native
            .tags_by_group(cache, "unic")
            .into_iter()
            .filter_map(|t| native.tag_name(cache, t))
            .map(|n| n.replace('/', "\\").to_ascii_lowercase())
            .collect();
        unics.sort();
        unics.dedup();
        ReachStrings { unics }
    }

    /// Resolve a `string_id` NAME (as [`NativeDll::string_id`] gives it) against MCC's tables.
    ///
    /// `preferred` names a `unic` to try first — pass `ui\hud\hud_messages` for a HUD prompt so
    /// a name that happens to exist in two lists resolves to the HUD's copy. Everything after
    /// that is the cache's own list, in order; `None` when no `unic`'s key hits any table.
    pub fn resolve(&self, sid: &str, preferred: Option<&str>) -> Option<Resolved> {
        if sid.is_empty() {
            return None;
        }
        let loc = crate::h4::localization::global();
        if loc.is_empty() {
            return None;
        }
        let order = preferred
            .map(|p| p.replace('/', "\\").to_ascii_lowercase())
            .into_iter()
            .chain(self.unics.iter().cloned());
        let mut seen: Vec<String> = Vec::new();
        for unic in order {
            if seen.contains(&unic) {
                continue;
            }
            let key = key_for(&unic, sid);
            if let Some(text) = loc.resolve_key(&key) {
                return Some(Resolved { unic, key, text });
            }
            seen.push(unic);
        }
        None
    }
}

/// The key MCC uses for a string in a given `unic`. Exposed because the convention is the
/// finding, and a caller may want to state the key it tried.
pub fn key_for(unic: &str, sid: &str) -> String {
    format!("{}_{}", unic.replace('/', "\\").to_ascii_lowercase(), sid)
}

/// **Is this string an ICON rather than text?** True when it carries at least one character and
/// every non-whitespace character is a private-use codepoint.
///
/// This is the distinction between the two kinds of chud string, and it is why
/// [`substitute_glyphs`] must not be applied blind:
///
/// * `"Hold \u{E446} to drive Warthog"` is TEXT with one glyph embedded. Substituting the
///   player's own bound key for the glyph gives them something better than the game's own line,
///   because it names *their* key.
/// * a weapon's `shematic` widget resolves to a SINGLE private-use codepoint and nothing else.
///   There is no text under it: the codepoint IS the weapon outline, drawn from MCC's
///   `font_package_icon*.bin`, which HMS does not read. Substituting `"[E]"` there would print a
///   key name where a gun silhouette belongs.
///
/// So an icon-only string must keep being refused — now for a stated reason rather than for want
/// of a resolver.
pub fn is_icon_only(text: &str) -> bool {
    let mut any = false;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        any = true;
        if !PRIVATE_USE.contains(&(c as u32)) {
            return false;
        }
    }
    any
}

/// Replace every private-use codepoint with `label`, and report which ones were replaced.
///
/// HMS has no icon font, so the alternative is a tofu box. A caller passes the player's own bound
/// key (`"E"`, `"F"`, …) and can then say in a diagnostic that it stood in for `U+E446`.
pub fn substitute_glyphs(text: &str, label: &str) -> (String, Vec<char>) {
    let mut out = String::with_capacity(text.len());
    let mut replaced = Vec::new();
    for c in text.chars() {
        if PRIVATE_USE.contains(&(c as u32)) {
            if !replaced.contains(&c) {
                replaced.push(c);
            }
            out.push_str(label);
        } else {
            out.push(c);
        }
    }
    (out, replaced)
}

#[cfg(test)]
mod gates {
    //! #reach-strings  `cargo test -p hms-app --profile quick reach_strings:: -- --ignored --nocapture`

    use super::*;

    /// The key convention, asserted by HASH against the four strings that pinned it down. This
    /// one needs no map and no MCC install: it is arithmetic over the convention, so it fails
    /// the moment the convention is written down wrong.
    #[test]
    fn the_key_convention_hashes_to_the_strings_it_was_derived_from() {
        use crate::h4::localization::key_hash;
        let k = |sid: &str| key_hash(&key_for("ui\\hud\\hud_messages", sid));
        assert_eq!(k("warthog_enter_driver"), 0x6871_118E, "\"Hold [x] to drive Warthog\"");
        assert_eq!(k("warthog_enter_passenger"), 0x6090_A24A, "\"Hold [x] to ride in Warthog\"");
        assert_eq!(k("warthog_enter_gunner"), 0xB127_7F31, "\"Hold [x] to operate Warthog turret\"");
        assert_eq!(k("warthog_board"), 0xD0CE_2A52, "\"Hold [x] to board enemy Warthog\"");
        // The convention is case-insensitive and slash-agnostic, because the hash uppercases and
        // `key_for` normalises; a forward-slash tag name must give the same key.
        assert_eq!(key_for("UI/Hud/Hud_Messages", "warthog_board"), "ui\\hud\\hud_messages_warthog_board");
        // And a DIFFERENT separator must NOT hash to it -- otherwise the assertion above is
        // satisfied by any convention at all.
        assert_ne!(key_hash("ui\\hud\\hud_messages\\warthog_enter_driver"), 0x6871_118E);
        assert_ne!(key_hash("warthog_enter_driver"), 0x6871_118E);
    }

    #[test]
    fn an_icon_only_string_is_not_text_and_a_prompt_is() {
        // the weapon schematic: one private-use codepoint, nothing under it
        assert!(is_icon_only("\u{E001}"));
        assert!(is_icon_only("  \u{E001}\u{E002} "));
        // the vehicle prompt: text with a glyph embedded
        assert!(!is_icon_only("Hold \u{E446} to drive Warthog"));
        assert!(!is_icon_only("12"));
        // empty is not an icon -- there is nothing to draw either way, and saying "icon" would
        // make a caller refuse for the wrong reason
        assert!(!is_icon_only(""));
        assert!(!is_icon_only("   "));
    }

    #[test]
    fn glyph_substitution_reports_what_it_stood_in_for() {
        let (s, g) = substitute_glyphs("Hold \u{E446} to drive Warthog", "E");
        assert_eq!(s, "Hold E to drive Warthog");
        assert_eq!(g, vec![GLYPH_ACTION]);
        let (s, g) = substitute_glyphs("plain text", "E");
        assert_eq!(s, "plain text");
        assert!(g.is_empty());
    }

    /// **THE LIVE GATE.** Every `enter seat string` and `in-seat string` every `vehi` in
    /// `forge_halo` authors must resolve to English through the cache's own `unic` names.
    #[test]
    #[ignore = "opens the native library and the MCC localization tables; run alone with --ignored"]
    fn every_authored_seat_prompt_resolves_to_english() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("reach_strings gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        if crate::h4::localization::global().is_empty() {
            eprintln!("reach_strings live gate SKIPPED: no MCC localization folder found");
            return;
        }
        let strings = ReachStrings::load(native, cache);
        assert!(strings.unics.len() > 10, "forge_halo should carry many unic tags, found {}", strings.unics.len());
        let hud = "ui\\hud\\hud_messages";
        assert!(strings.unics.iter().any(|u| u == hud), "the cache does not carry {hud}");

        let mut ids: Vec<String> = Vec::new();
        for v in native.tags_by_group(cache, "vehi") {
            let Some(u) = crate::reach_seats::UnitSeats::read(native, cache, v) else { continue };
            for s in &u.seats {
                for id in [&s.enter_seat_string, &s.in_seat_string] {
                    if !id.is_empty() && !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
            }
        }
        assert!(ids.len() > 20, "expected many distinct seat prompt ids, found {}", ids.len());
        ids.sort();
        let mut unresolved: Vec<String> = Vec::new();
        let mut from_hud = 0usize;
        for id in &ids {
            match strings.resolve(id, Some(hud)) {
                Some(r) => {
                    if r.unic == hud {
                        from_hud += 1;
                    }
                    let (shown, glyphs) = substitute_glyphs(&r.text, "[ACTION]");
                    println!(
                        "  {id:<28} {:<22} {shown}{}",
                        r.unic,
                        if glyphs.is_empty() { String::new() } else { format!("   (glyphs {glyphs:?})") }
                    );
                }
                None => unresolved.push(id.clone()),
            }
        }
        println!("{} of {} seat prompt ids resolved, {from_hud} from {hud}", ids.len() - unresolved.len(), ids.len());
        assert!(
            unresolved.is_empty(),
            "{} of {} authored seat prompt ids did not resolve: {unresolved:?}",
            unresolved.len(),
            ids.len()
        );
    }
}
