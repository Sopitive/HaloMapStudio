//! #play-hud  The whole test-mode HUD as ONE function over an egui painter, so the app draws it
//! and the headless rasteriser (`hud_raster`) draws the identical shapes into a PNG.
//!
//! This is the body that used to live inside `App::play_first_person_hud`. It owns no state: the
//! value feed is [`crate::play_hud::PlayHud`], the widget decode and painter are
//! [`crate::reach_hud`] / [`crate::reach_chud`], the seat rules are [`crate::reach_unit_hud`].

use hms_native::{CacheHandle, NativeDll};

use crate::{play_hud::PlayHud, reach_chud, reach_hud, reach_unit_hud};

/// Everything the draw needs that is not in [`PlayHud`].
#[derive(Clone, Debug, Default)]
pub struct HudFrame {
    pub elite: bool,
    /// The held `weap` (0 = empty hands).
    pub weapon_tag: u32,
    /// The ride's phase-aware answer to "may the biped's weapon be used" (true on foot).
    pub weapon_usable: bool,
    pub seat_huds: Option<reach_unit_hud::SeatHuds>,
    /// `(seat index, caption)` while riding.
    pub riding: Option<(usize, String)>,
    pub hud_on: bool,
    pub crosshair_on: bool,
    /// The label of the key bound to the action button, for a button prompt.
    pub key_label: String,
    /// #use-hold  The action prompt on offer ("Hold E to drive Warthog"), resolved with the player's
    /// own key; `None` when no seat or weapon is in range. Drawn through the player HUD's own
    /// `player_training` collection, where the engine prints runtime messages.
    pub prompt: Option<String>,
}

/// Paint the HUD for one frame; returns the provenance / decision report `play hud` prints.
pub fn paint_play_hud(
    ctx: &eframe::egui::Context,
    painter: &eframe::egui::Painter,
    viewport: eframe::egui::Rect,
    native: &NativeDll,
    cache: CacheHandle,
    hud: &PlayHud,
    f: &HudFrame,
) -> Vec<String> {
    let skin = reach_chud::globals_for(native, cache).and_then(|g| g.default_skin().cloned());
    let vs = match &skin {
        Some(s) => reach_hud::VirtualScreen::from_skin(viewport, s),
        None => reach_hud::VirtualScreen::fallback(viewport),
    };
    let mut lines: Vec<String> = Vec::new();

    // #seat-hud  WHICH HUDs the seat puts on screen is the SEAT BLOCK's answer
    // (`reach_unit_hud`, the engine's `sub_1408BED90` slots): the `chud` the seat names, the
    // vehicle weapon's own `chud` in a gunner seat, and the biped's weapon HUD only where the
    // seat allows weapons.
    let seat_huds = f.seat_huds.clone();
    let biped_hud = reach_unit_hud::SeatHuds::biped_hud(f.weapon_usable, seat_huds.as_ref());
    let vehicle = f.riding.as_ref().and_then(|(seat, _)| {
        seat_huds
            .as_ref()
            .and_then(|h| h.seat)
            .and_then(|t| native.tag_name(cache, t))
            .map(|n| (n, (*seat).min(255) as u8))
    });
    let weapon = if biped_hud { f.weapon_tag } else { 0 };
    let mut v = hud.values(weapon, vehicle.clone(), Some(f.key_label.clone()));
    // #seat-hud  A chase-camera seat is states 25, not 24 (see `HudValues::third_person`).
    let seat_tp = seat_huds.as_ref().is_some_and(|h| h.third_person);
    v.third_person = seat_tp;
    v.training_text = f.prompt.clone();
    let (mut drawn, mut skipped) = (0usize, 0usize);
    let player_path = if f.elite { reach_hud::PLAYER_CHUD_ELITE } else { reach_hud::PLAYER_CHUD_SPARTAN };

    // ---- the held weapon's reticule ---------------------------------------------------
    //
    // NOT in a seat that stows the biped's weapon: the vehicle's reticule (below) is the one the
    // engine shows there, and drawing this as well would put two on screen (and, with the weapon
    // stowed, this one would be the UNARMED reticule).
    if f.crosshair_on && biped_hud {
        let art = (weapon != 0).then(|| reach_chud::crosshair_for(ctx, native, cache, weapon)).flatten();
        match (&art, &skin) {
            (Some(a), Some(sk)) => {
                reach_chud::paint_crosshair(painter, viewport, a, sk, vs.vpx);
                lines.push(format!(
                    "Crosshair: {} authored part(s) from {}",
                    a.crosshair.parts.len(),
                    native.tag_name(cache, a.crosshair.chud_tag).unwrap_or_default()
                ));
            }
            _ => {
                // The engine's own empty-handed reticule, out of the player HUD.
                let drew = reach_hud::chud_by_path(native, cache, player_path)
                    .and_then(|t| reach_hud::hud_art(ctx, native, cache, t))
                    .and_then(|a| {
                        let i = a.chud.collections.iter().position(|c| c.base.name == "unarmed_crosshair")?;
                        Some(reach_hud::paint_collection(ctx, painter, &vs, &a, i, skin.as_ref(), &v, 0))
                    });
                lines.push(match (weapon, drew) {
                    (0, Some((d, _))) if d > 0 => {
                        "Crosshair: empty hands, so the player HUD's own `unarmed_crosshair` is drawn".into()
                    }
                    (0, _) => "Crosshair: empty hands, and this HUD authors no unarmed reticule".into(),
                    (_, _) => "Crosshair: this weapon authors no reticule that draws at rest".into(),
                });
            }
        }
    }

    // ---- #seat-hud  the vehicle's part: slot 4 (its weapon's reticule) and slot 6 (the seat's
    // own `chud`). The player-owned fields are cleared for the seat `chud` -- see
    // `PlayHud::vehicle_values` -- so its meters take nothing from the driver's shield.
    if let Some(h) = seat_huds.clone() {
        let mut vv = hud.vehicle_values(vehicle.clone(), Some(f.key_label.clone()));
        vv.third_person = seat_tp;
        let sp = reach_unit_hud::paint_seat_huds(ctx, painter, &vs, native, cache, &h, skin.as_ref(), &vv, f.crosshair_on, f.hud_on);
        drawn += sp.seat_drawn;
        skipped += sp.seat_skipped;
        lines.extend(sp.lines);
    }

    // ---- #use-hold  the action prompt, in the player HUD's `player_training` collection ----------
    //
    // Outside `hud_on`: it is the game telling the player what the button does, not a meter.
    if f.prompt.is_some() {
        match reach_hud::chud_by_path(native, cache, player_path).and_then(|t| reach_hud::hud_art(ctx, native, cache, t)) {
            Some(a) => {
                if let Some(i) = a.chud.collections.iter().position(|c| c.base.name == reach_hud::TRAINING_COLLECTION) {
                    let (d, s) = reach_hud::paint_collection(ctx, painter, &vs, &a, i, skin.as_ref(), &v, 0);
                    drawn += d;
                    skipped += s;
                    lines.push(format!("Action prompt: {d} widget(s) of `{}` drawn", reach_hud::TRAINING_COLLECTION));
                }
            }
            None => lines.push(format!("Action prompt: this cache carries no '{player_path}'")),
        }
    }

    if f.hud_on {
        // ---- the player HUD: shields, health, grenades -------------------------------
        match reach_hud::chud_by_path(native, cache, player_path).and_then(|t| reach_hud::hud_art(ctx, native, cache, t)) {
            Some(a) => {
                for want in reach_hud::PLAYER_COLLECTIONS {
                    let Some(i) = a.chud.collections.iter().position(|c| c.base.name == want) else { continue };
                    // The grenade block is datasource-driven: one repeat per grenade TYPE.
                    let reps =
                        if a.chud.collections[i].datasource == Some(reach_hud::DATASOURCE_GRENADES) { 2 } else { 1 };
                    for r in 0..reps {
                        let (d, s) = reach_hud::paint_collection(ctx, painter, &vs, &a, i, skin.as_ref(), &v, r);
                        drawn += d;
                        skipped += s;
                    }
                }
            }
            None => lines.push(format!("Player HUD: this cache carries no '{player_path}'")),
        }

        // ---- the held weapon's own ammo block ----------------------------------------
        //
        // Matched by PREFIX: the shipped weapons are not consistent (`ammo_area` on the assault
        // rifle, `ammo_area_right` on the magnum, `backpack_ammo_area` for a stowed one), which
        // is why `reach_hud::WEAPON_AMMO_PREFIX` exists.
        if weapon != 0 {
            match reach_chud::weapon_chud(native, cache, weapon).and_then(|t| reach_hud::hud_art(ctx, native, cache, t)) {
                Some(a) => {
                    let mut blocks = 0usize;
                    for (i, c) in a.chud.collections.iter().enumerate() {
                        if !c.base.name.starts_with(reach_hud::WEAPON_AMMO_PREFIX) {
                            continue;
                        }
                        blocks += 1;
                        let (d, s) = reach_hud::paint_collection(ctx, painter, &vs, &a, i, skin.as_ref(), &v, 0);
                        drawn += d;
                        skipped += s;
                    }
                    if blocks == 0 {
                        lines.push("Ammo block: this weapon's chud authors no `ammo_area*` collection".into());
                    }
                }
                None => lines.push("Ammo block: this weapon references no `chud interface`".into()),
            }
        } else if f.riding.is_some() && !biped_hud {
            lines.push(
                "Ammo block: not drawn - this seat does not allow weapons (seat flag `allows weapons` \
                 clear), so the engine adds none of the biped's weapon HUD slots"
                    .into(),
            );
        }

        if let Some((_, cap)) = &f.riding {
            lines.push(format!("Riding: {cap}"));
        }
        lines.push(format!("HUD: {drawn} authored widget(s) drawn, {skipped} left out for want of a value"));
    }

    // The motion tracker, named rather than silently missing.
    lines.push(
        "Motion tracker: NOT drawn. Test mode has one body on the map and no other players, so every blip \
         would be invented; an empty radar frame would be a claim of its own."
            .into(),
    );
    let unresolved = reach_hud::take_unresolved_strings();
    if !unresolved.is_empty() {
        lines.push(format!(
            "Not drawn ({} authored string(s) HMS cannot resolve to a glyph): {}",
            unresolved.len(),
            unresolved.join(", ")
        ));
    }
    lines.extend(hud.report());
    lines.extend(v.report());
    lines
}
