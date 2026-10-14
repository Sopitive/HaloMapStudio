//! #pickup  **Picking a weapon up off the map, in test mode.**
//!
//! Reported: *"I also can't pick up any of the weapons on the map"*. `Play ▸ Give weapon` could
//! put any of the map's weapons in your hands from a menu, but the weapons the map itself has
//! PLACED — the ones lying on the ground where the author put them — could not be touched, which
//! is the thing a player actually tries first.
//!
//! **This is the same shape as getting into a vehicle, and deliberately not a second
//! implementation of it.** `play_vehicle::offer` finds the nearest seat you could enter, resolves
//! the engine's own authored prompt string, and substitutes the button glyph with your bound key;
//! this does the same for a weapon, through the same [`reach_strings`] helpers, and hands the
//! result to the SAME `give_weapon` the menu uses. Nothing about holding a weapon is duplicated:
//! a pickup is `give_weapon` with a different trigger.
//!
//! # What is authored and what is ours
//!
//! **Authored, read from the tag:**
//!
//! * the prompt — `weap`'s `pickup message` at **+0x16C**, a `string_id`, resolved through the
//!   same `ui\hud\hud_messages` preference the seat prompt uses (`swap message` at **+0x170** is
//!   the one the engine shows when your hands are full, and that is the one shown then);
//! * which placed objects are weapons at all — the map's own `weap` tag set, the same list
//!   `Play ▸ Give weapon` is built from, so a placement is a candidate exactly when its tag is
//!   one the map holds.
//!
//! **OURS, in one place, pending the engine's:** [`PICKUP_REACH_WU`]. Searched for and did not
//! find: there is **no authored pickup or interact range for the player** anywhere in the 238 tag
//! groups of `reach_tag_test.exe`'s definitions. The only two near misses are
//! `character_interact_block`'s *"default maximum object interact range"*, which is an **AI**
//! field on `char` and not the player's, and `device`'s *"automatic activation radius"*, which is
//! how close you have to be for a door to open itself. The vehicle side's `SEARCH_RADIUS = 3.0`
//! **is** authored — the `unit_seat_block`'s own note says the player searches 3 wu for a vehicle
//! — but that note is about vehicles, so borrowing it here would be presenting a guess as a
//! measurement. So the range is a single constant with this comment on it, and when the physics
//! campaign reaches `unit_pickup_object` it is the one line that changes.
//!
//! # The geometry of the test
//!
//! Distance is measured from the player's **capsule axis**, not from their eye and not from their
//! feet. A weapon on the floor 0.20 wu in front of a Spartan's toes is **0.20 wu from the axis**
//! and **0.59 wu from a standing eye** — and, the part that decides it, **0.42 wu from a
//! crouching eye**, because the authored eye heights differ (0.600 standing, 0.420 crouched). So
//! measuring from the eye would make **crouching change what you can reach by 40%**, for no
//! reason a player could see. The axis does not move when you crouch, and the gate below asserts
//! exactly that pair of facts.
//!
//! There is deliberately **no facing cone**. The vehicle side has one because a seat has an entry
//! marker with an authored direction; a weapon lying on the ground has no front, and inventing a
//! cone for it would make the mode refuse pickups for a reason nothing authored.

use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};

use crate::reach_strings::{self, ReachStrings};

/// `weap` field offsets. A weapon's `item` fields begin at +0x168, so these are absolute within
/// the tag — the same convention `reach_damage`'s weapon offsets use.
const WEAP_PICKUP_MESSAGE: usize = 0x16C;
/// What the engine prints instead when your hands are already full.
const WEAP_SWAP_MESSAGE: usize = 0x170;

/// How close the player has to be to pick a weapon up, world units, measured from the capsule
/// axis to the placement's origin.
///
/// **This is HMS's number, not the engine's**, for the reason set out in the module docs: Reach
/// authors no player pickup or interact range in any tag. 1.0 wu is a little over 3 m at Reach's
/// scale and a little under a Spartan's own height, which is about as far as a player expects to
/// be able to reach — but it is a choice, and it is a choice in exactly one place so that the
/// measured value can replace it.
pub const PICKUP_REACH_WU: f32 = 1.0;

/// How far out to consider placements at all. Only bounds how many objects are examined; the
/// reach test above is what decides. Generous, because a placement's ORIGIN can sit well away
/// from the model a player is standing next to.
const SEARCH_WU: f32 = PICKUP_REACH_WU + 4.0;

/// One weapon the player could pick up right now, with the prompt the game would print.
#[derive(Clone, Debug, PartialEq)]
pub struct PickupOffer {
    /// The placed object's datum.
    pub datum: u32,
    /// Its `weap` tag — what [`crate::play_mode::PlayMode::give_weapon`] is handed.
    pub weap_tag: u32,
    pub path: String,
    /// Distance from the player's capsule axis to the placement's origin, world units.
    pub distance: f32,
    /// The authored `string_id` — `pickup message` with empty hands, `swap message` when the
    /// player is already holding something.
    pub prompt_id: String,
    /// The engine's own prompt with the button glyph replaced by the player's bound key.
    /// **`None` when the string did not resolve, and a caller must then draw nothing** — same
    /// rule as the seat prompt, for the same reason.
    pub prompt: Option<String>,
    /// The private-use codepoints that were substituted, so a caption can say so.
    pub substituted: Vec<char>,
    /// True when this would REPLACE what the player is holding rather than fill empty hands.
    pub swap: bool,
}

impl PickupOffer {
    /// The leaf name a player recognises.
    pub fn leaf(&self) -> &str {
        self.path.rsplit(['\\', '/']).next().unwrap_or(&self.path)
    }

    /// What the player would see, or the reason there is nothing to show.
    pub fn caption(&self) -> String {
        match &self.prompt {
            Some(p) => p.clone(),
            None if self.prompt_id.is_empty() => {
                format!("{} authors no {} string", self.leaf(), if self.swap { "swap" } else { "pickup" })
            }
            None => format!("prompt string '{}' did not resolve; nothing drawn", self.prompt_id),
        }
    }
}

/// The authored prompt `string_id` for a weapon tag: `swap message` when the player's hands are
/// full, `pickup message` when they are empty. Empty when the tag authors neither.
pub fn prompt_id(native: &NativeDll, cache: CacheHandle, weap: u32, holding: bool) -> String {
    let off = if holding { WEAP_SWAP_MESSAGE } else { WEAP_PICKUP_MESSAGE };
    let read = |o: usize| -> String {
        native
            .tag_meta(cache, weap, o as u32, 4)
            .map(|m| u32::from_le_bytes([m[0], m[1], m[2], m[3]]))
            .filter(|v| *v != 0)
            .and_then(|v| native.string_id(cache, v))
            .unwrap_or_default()
    };
    let s = read(off);
    // A weapon that authors no swap string still authors a pickup one, and showing that is better
    // than showing nothing — the engine's choice of which to print is not worth refusing a
    // pickup over.
    if s.is_empty() && holding {
        read(WEAP_PICKUP_MESSAGE)
    } else {
        s
    }
}

/// Distance from the player's capsule AXIS to a point — see the module docs for why the axis and
/// not the eye.
fn axis_distance(point: Vec3, feet: Vec3, height: f32) -> f32 {
    let z = point.z.clamp(feet.z, feet.z + height.max(0.0));
    (point - Vec3::new(feet.x, feet.y, z)).length()
}

/// The nearest weapon the player could pick up, or `None`.
///
/// `weapons` is the map's own `weap` tag set — `(tag, leaf)` pairs, exactly what `Play ▸ Give
/// weapon` lists — so a placement counts as a weapon when its tag is in it. `key_label` is the
/// player's own bound key, substituted for the engine's button glyph. `holding` picks which
/// authored string is resolved.
#[allow(clippy::too_many_arguments)]
pub fn offer(
    native: &NativeDll,
    cache: CacheHandle,
    strings: &ReachStrings,
    objects: &[ObjectInfo],
    weapons: &[(u32, String)],
    feet: Vec3,
    height: f32,
    holding: bool,
    key_label: &str,
) -> Option<PickupOffer> {
    let mut best: Option<PickupOffer> = None;
    for o in objects {
        let pos = Vec3::from(o.pos);
        // Cheap reject on the origin first, so a map with thousands of placements is not a
        // per-frame tag read.
        if (pos - feet).length() > SEARCH_WU {
            continue;
        }
        if !weapons.iter().any(|(t, _)| *t == o.primary_tag) {
            continue;
        }
        let distance = axis_distance(pos, feet, height);
        if distance > PICKUP_REACH_WU {
            continue;
        }
        if best.as_ref().is_some_and(|b| b.distance <= distance) {
            continue;
        }
        let path = weapons
            .iter()
            .find(|(t, _)| *t == o.primary_tag)
            .map(|(_, p)| p.clone())
            .or_else(|| native.tag_name(cache, o.primary_tag))
            .unwrap_or_default();
        let id = prompt_id(native, cache, o.primary_tag, holding);
        let resolved = if id.is_empty() {
            None
        } else {
            strings.resolve(&id, Some(crate::play_vehicle::HUD_MESSAGES_UNIC))
        };
        let (prompt, substituted) = match resolved {
            // An ICON-only string is a glyph, not text: substituting a key label there would
            // print a key name where a picture belongs. The seat prompt makes the same call.
            Some(r) if reach_strings::is_icon_only(&r.text) => (None, Vec::new()),
            Some(r) => {
                let (t, g) = reach_strings::substitute_glyphs(&r.text, key_label);
                (Some(t), g)
            }
            None => (None, Vec::new()),
        };
        best = Some(PickupOffer {
            datum: o.datum,
            weap_tag: o.primary_tag,
            path,
            distance,
            prompt_id: id,
            prompt,
            substituted,
            swap: holding,
        });
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #pickup-gate  The reach is measured from the CAPSULE AXIS, so crouching does not change
    /// what you can pick up — which is the whole reason it is not measured from the eye.
    #[test]
    fn the_reach_is_measured_from_the_body_not_the_eye() {
        let feet = Vec3::new(0.0, 0.0, 0.0);
        // A weapon lying on the floor just in front of the player's toes -- the case the choice
        // of origin actually decides. (An earlier version of this gate put it 0.5 wu out, where
        // the HORIZONTAL distance dominates and the eye is only 1.5x further: the assertion
        // failed at 0.743 vs 0.5 and it was the fixture that was wrong, not the rule. A gate
        // whose geometry does not exercise the thing it claims is a gate that cannot fail for
        // the right reason.)
        let gun = Vec3::new(0.20, 0.0, 0.05);
        // The AUTHORED capsule and eye heights for a Spartan, standing and crouched.
        let (h_stand, h_crouch) = (0.650f32, 0.500f32);
        let (eye_stand, eye_crouch) = (0.600f32, 0.420f32);
        let standing = axis_distance(gun, feet, h_stand);
        let crouching = axis_distance(gun, feet, h_crouch);
        assert!(
            (standing - crouching).abs() < 1e-6,
            "crouching must not change the reach: {standing} vs {crouching}"
        );
        assert!(standing < PICKUP_REACH_WU, "a gun at your toes must be in reach, {standing} wu");
        // What the eye would have given instead, standing and crouched -- the two numbers that
        // made the axis the right origin.
        let eye_up = (gun - Vec3::new(0.0, 0.0, eye_stand)).length();
        let eye_down = (gun - Vec3::new(0.0, 0.0, eye_crouch)).length();
        assert!(
            eye_up > standing * 2.5,
            "the eye must be the materially further measurement ({eye_up} vs {standing})"
        );
        // THE ONE THAT DECIDES IT: measured from the eye, crouching changes your reach. Measured
        // from the axis (asserted above) it does not.
        assert!(
            (eye_up - eye_down).abs() > 0.1,
            "the eye measurement must MOVE when you crouch, or there would be nothing to choose \
             between the two origins ({eye_up} standing vs {eye_down} crouched)"
        );
        // Above and below the body still measure to the nearest point of the axis, not to a
        // single point on it.
        let high = axis_distance(Vec3::new(0.0, 0.0, 2.0), feet, h_stand);
        assert!((high - (2.0 - h_stand)).abs() < 1e-6, "above the head measures to the top, got {high}");
    }

    /// #pickup-gate  Out of reach is out of reach, and the boundary is the ONE constant.
    #[test]
    fn the_reach_is_the_one_constant_and_nothing_else() {
        let feet = Vec3::ZERO;
        let h = 0.65;
        let just_in = axis_distance(Vec3::new(PICKUP_REACH_WU - 0.01, 0.0, 0.3), feet, h);
        let just_out = axis_distance(Vec3::new(PICKUP_REACH_WU + 0.01, 0.0, 0.3), feet, h);
        assert!(just_in <= PICKUP_REACH_WU, "inside the reach must be inside it");
        assert!(just_out > PICKUP_REACH_WU, "outside it must be outside it");
    }

    /// #pickup-gate  The script verb reaches the production executor through the REAL parser.
    ///
    /// `play pickup` reports and `play pickup take` acts, and both go through
    /// `script::parse_line` into `EditorCommand::Play` -- the same one `play_command` dispatches
    /// -- so a scripted run drives the function the key calls and not a copy of it.
    #[test]
    fn the_pickup_verb_parses_through_the_real_parser() {
        use crate::script::{parse_line, EditorCommand};
        for (line, want) in [
            ("play pickup", vec!["pickup"]),
            ("play pickup take", vec!["pickup", "take"]),
            ("play pick up", vec!["pick", "up"]),
        ] {
            match parse_line(line) {
                Ok(Some(EditorCommand::Play(args))) => assert_eq!(args, want, "for '{line}'"),
                other => panic!("'{line}' must parse to a Play command, got {other:?}"),
            }
        }
    }

    /// #pickup-gate  The caption never invents a prompt: a string that did not resolve says so,
    /// and a weapon that authors none says that instead. Same rule as the seat prompt — a caller
    /// that drew `prompt` unconditionally would print an empty box.
    #[test]
    fn a_prompt_that_did_not_resolve_is_reported_not_faked() {
        let mut o = PickupOffer {
            datum: 1,
            weap_tag: 2,
            path: "objects\\weapons\\rifle\\assault_rifle\\assault_rifle".into(),
            distance: 0.4,
            prompt_id: String::new(),
            prompt: None,
            substituted: Vec::new(),
            swap: false,
        };
        assert_eq!(o.leaf(), "assault_rifle");
        assert!(o.caption().contains("authors no pickup string"), "got '{}'", o.caption());
        o.swap = true;
        assert!(o.caption().contains("authors no swap string"), "got '{}'", o.caption());
        o.prompt_id = "weapon_pickup".into();
        assert!(o.caption().contains("did not resolve"), "got '{}'", o.caption());
        o.prompt = Some("Hold E to pick up Assault Rifle".into());
        assert_eq!(o.caption(), "Hold E to pick up Assault Rifle");
    }
}
