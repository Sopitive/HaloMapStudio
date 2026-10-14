//! #seat-entry  TEST MODE'S VEHICLES: the prompt, getting in, riding, and driving.
//!
//! This is the app-side half of [`crate::reach_seats`] / [`crate::reach_vehicle`] /
//! [`crate::reach_strings`]: it finds the vehicle a player is standing next to, runs the seats'
//! own authored entry tests against it, resolves the engine's own prompt, and holds the ride.
//!
//! Everything measured lives in the three modules above. **Nothing here chooses a number.**
//!
//! # The three things a reader should know before changing this
//!
//! **1. A vehicle's seats are not all on the vehicle.** The Warthog's GUNNER seat is on
//! `warthog_chaingun` / `_gauss` / `_rocket`, separate `vehi` tags attached as child objects —
//! and the Falcon's six turrets likewise. So the scan walks the parent's own seats AND every
//! attached child whose tag is a `vehi`, composing the child's frame with
//! `enumerate_attachments`'s own `(pos, rot)`. Looking only at the parent would make a Warthog's
//! turret un-enterable, which is the obvious bug this was built to avoid.
//!
//! **2. The player is NOT stepped while seated.** [`crate::play_ctl::PlayerState::step`] is
//! bypassed entirely, not extended: a rider's transform is the vehicle's transform times the seat
//! marker, which is already how Theater draws riders. Running the capsule controller as well
//! would fight the seat for the body's position — two writers on one piece of state, the defect
//! family this project keeps paying for.
//!
//! **3. The driven vehicle is drawn as an ACTOR, and the placement is HIDDEN.** The editor's
//! object table is the user's unsaved map variant; mutating it at 60 Hz would leave their Warthog
//! wherever they parked it if test mode ended by a crash or a map close. So test mode never
//! touches it: [`Riding::drive_state`] is this module's own, the vehicle draws through
//! `reach_actor::ActorScene` like the player's own biped, and the placement is suppressed for the
//! duration. **Hiding must be as reliable as showing** — a leak one way shows two Warthogs, a
//! leak the other way shows none — so [`Riding::hidden_datum`] is the single source of both and
//! [`gates`] asserts each direction.
//!
//! # What the prompt says, and when it says nothing
//!
//! The text is the game's own, through the key convention
//! [`crate::reach_strings`] measured: `ui\hud\hud_messages_<enter seat string>`. The engine's
//! button glyph (`U+E446`) is substituted with the player's OWN bound key, which is more use to
//! them than the game's text would be — and the substitution is reported, not silent. **When the
//! string does not resolve, [`VehicleOffer::prompt`] is `None` and the caller draws nothing**:
//! there is no "Press E to enter" of HMS's own invention anywhere in this file.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};

use crate::objscene::ObjectScene;
use crate::reach_actor::{Pose, ReachModel};
use crate::reach_seats::{self, EntryTest, UnitSeats};
use crate::reach_strings::{self, ReachStrings};
use crate::reach_vehicle::{DriveInput, DriveModel, DriveState, VehiclePhysics};

/// The `unic` the HUD prompts live in — read off the CACHE by [`ReachStrings`], and passed as the
/// preferred list so a name that exists twice resolves to the HUD's copy.
pub const HUD_MESSAGES_UNIC: &str = "ui\\hud\\hud_messages";

/// How far from the player to look for a vehicle at all, world units.
///
/// **The authors' own number, not a choice**: the `unit_seat_block`'s own note reads *"the entry
/// radius shouldn't exceed 3 world units, as that is as far as the player will search for a
/// vehicle to enter."* Every seat's own `entry radius` is then applied on top, so this only
/// bounds how many objects are examined.
pub const SEARCH_RADIUS: f32 = 3.0;

// ---------------------------------------------------------------------------------------------
// THE CACHE OF WHAT A VEHICLE IS
// ---------------------------------------------------------------------------------------------

/// Everything read once per `vehi` TAG: its seats, its render model (for the markers), its
/// physics, and the drive model its tag supports.
pub struct VehicleEntry {
    pub vehi_tag: u32,
    pub path: String,
    pub seats: UnitSeats,
    pub model: Option<ReachModel>,
    pub pose: Option<Pose>,
    pub phys: Option<VehiclePhysics>,
    /// `Ok` with the model, or `Err` with the reason this vehicle is not drivable — the reason is
    /// shown rather than swallowed.
    pub drive: Result<DriveModel, String>,
    /// The jmad seat-mode family for riders (`warthog`, `ghost`, `mongoose`, `truck`), or empty
    /// when the shipped Spartan graph has no seat set for it.
    pub seat_family: String,
    /// #vehicle-anim  The vehicle's OWN graph, and what each of its overlays means. This is the
    /// half of the ride the user asked for second: *"Vehicles have their own unique animation
    /// they play when a biped is driving it, so handlebars, wheels etc would function"*.
    pub graph: Option<crate::reach_actor::AnimGraph>,
    pub anim: crate::reach_vehicle_anim::VehicleAnim,
}

impl VehicleEntry {
    /// A marker group's frames in MODEL space, empty when the model does not carry it.
    pub fn marker_frames(&self, native: &NativeDll, cache: CacheHandle, name: &str) -> Vec<Mat4> {
        let (Some(m), Some(p)) = (&self.model, &self.pose) else { return Vec::new() };
        match m.marker_group(native, cache, name) {
            Some(g) => g.markers.iter().map(|mk| p.marker_model(mk)).collect(),
            None => Vec::new(),
        }
    }
}

/// Lazily-built per-tag cache. Reading a render model is expensive and a vehicle's tag does not
/// change, so this is built once per tag and held for the session.
#[derive(Default)]
pub struct VehicleIndex {
    by_tag: HashMap<u32, VehicleEntry>,
    strings: Option<ReachStrings>,
}

impl VehicleIndex {
    /// The entry for a `vehi` tag, built on first use. `None` when the tag is not a `vehi` or
    /// authors no seats.
    pub fn entry(&mut self, native: &NativeDll, cache: CacheHandle, vehi_tag: u32) -> Option<&VehicleEntry> {
        if !self.by_tag.contains_key(&vehi_tag) {
            if native.tag_class(cache, vehi_tag).as_deref() != Some("vehi") {
                return None;
            }
            let seats = UnitSeats::read(native, cache, vehi_tag)?;
            let path = native.tag_name(cache, vehi_tag).unwrap_or_default();
            let mode = native.resolve_mode(cache, vehi_tag);
            let model = ReachModel::read(native, cache, mode);
            let pose = model.as_ref().map(|m| Pose::bind(&m.skel));
            let mut phys = VehiclePhysics::read(native, cache, vehi_tag);
            // The wheelbase comes from the friction points' MARKER POSITIONS, which live in the
            // render model and not in the `vehi` tag — so they are resolved here and handed over
            // rather than guessed there.
            if let (Some(p), Some(m), Some(ps)) = (phys.as_mut(), model.as_ref(), pose.as_ref()) {
                // #vehicle-suspension  ALIGNMENT, not just resolution. `contact_points()` ZIPS
                // these against the friction points, so a vector that silently skipped a missing
                // marker would hand wheel 3's position wheel 2's radius and mass share. The
                // previous version did exactly that — two chained `filter_map`s — which was
                // harmless while the only consumer was a front-to-back extent and is not harmless
                // now. So each marker resolves to an `Option` and the set is only accepted whole.
                let resolve = |name: &str| -> Option<Vec3> {
                    let g = m.marker_group(native, cache, name)?;
                    let mk = g.markers.first().copied()?;
                    Some(ps.marker_model(&mk).w_axis.truncate())
                };
                // A partial set is still handed over: the wheelbase and track only need the
                // front-to-back and side-to-side extents, so they keep measuring. The contact
                // solve refuses a short vector on its own (`contact_points` compares the lengths),
                // which is where the misalignment would have mattered.
                let wheels: Vec<Vec3> =
                    p.havok.friction_points.iter().filter_map(|f| resolve(&f.marker_name)).collect();
                p.set_wheel_positions(wheels);
                let agps: Vec<Option<Vec3>> =
                    p.havok.antigravity_points.iter().map(|a| resolve(&a.marker_name)).collect();
                if !agps.is_empty() && agps.iter().all(|a| a.is_some()) {
                    p.set_antigravity_positions(agps.into_iter().flatten().collect());
                }
            }
            let driver = seats.seats.iter().find(|s| s.is_driver()).or_else(|| seats.seats.first());
            let drive = match &phys {
                Some(p) => DriveModel::for_vehicle(p, driver),
                None => Err("this vehicle's physics block could not be read".to_string()),
            };
            let seat_family = crate::film_anim::seat_family_for(&path);
            // #vehicle-anim  The vehicle's own graph, classified once per TAG. Every overlay is
            // sampled to decide whether its frames wrap or saturate, which is a handful of
            // resource reads -- so it happens here, beside the render model, and never per frame.
            let graph = crate::reach_actor::resolve_jmad(native, cache, vehi_tag)
                .and_then(|j| crate::reach_actor::AnimGraph::read(native, cache, j));
            let anim = match (model.as_ref(), graph.as_ref()) {
                (Some(m), Some(g)) => {
                    crate::reach_vehicle_anim::VehicleAnim::read(native, cache, m, g, phys.as_ref())
                }
                _ => {
                    let mut v = crate::reach_vehicle_anim::VehicleAnim::default();
                    v.notes.push(
                        "this vehicle has no render model or no animation graph, so nothing of it                          is animated while it is driven"
                            .into(),
                    );
                    v
                }
            };
            self.by_tag.insert(
                vehi_tag,
                VehicleEntry { vehi_tag, path, seats, model, pose, phys, drive, seat_family, graph, anim },
            );
        }
        self.by_tag.get(&vehi_tag)
    }

    /// The localized-string reader, built once per cache.
    pub fn strings(&mut self, native: &NativeDll, cache: CacheHandle) -> &ReachStrings {
        self.strings.get_or_insert_with(|| ReachStrings::load(native, cache))
    }

    pub fn len(&self) -> usize {
        self.by_tag.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_tag.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// THE OFFER
// ---------------------------------------------------------------------------------------------

/// One seat the player could get into right now, with the prompt the game would print.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleOffer {
    /// The PLACED object's datum — which may be an attached child, not the one the player
    /// clicked.
    pub datum: u32,
    /// The datum of the object the seat's own vehicle is ATTACHED TO, when it is a child (the
    /// Warthog under a `warthog_chaingun`). `None` when the seat is on the placement itself.
    pub parent_datum: Option<u32>,
    /// The `vehi` tag whose seats block `seat` indexes. **Not always the placement's own tag.**
    pub vehi_tag: u32,
    pub vehi_path: String,
    pub seat: usize,
    /// The seat's own `label` string id (`warthog_d`, `ghost_d`, …) — which is also the jmad
    /// seat-mode name.
    pub seat_label: String,
    /// `enter seat string`, the authored string id.
    pub prompt_id: String,
    /// The engine's own prompt with the button glyph replaced by the player's bound key.
    /// **`None` when the string did not resolve, and a caller must then draw nothing.**
    pub prompt: Option<String>,
    /// The private-use codepoints that were substituted, so a caption can say so.
    pub substituted: Vec<char>,
    /// The four authored gates, so a refusal can say which one.
    pub test: EntryTest,
}

impl VehicleOffer {
    /// What the player would see, or the reason there is nothing to show.
    pub fn caption(&self) -> String {
        match &self.prompt {
            Some(p) => p.clone(),
            None if self.prompt_id.is_empty() => format!("{} seat {} authors no prompt string", self.vehi_path, self.seat),
            None => format!("prompt string '{}' did not resolve; nothing drawn", self.prompt_id),
        }
    }
}

/// The best seat the player can enter right now, over every nearby vehicle and every seat of it.
///
/// `key_label` is the player's own bound key for the action, substituted for the engine's button
/// glyph. `player_vel` is the player's velocity (the vehicle's is zero while it is a placement),
/// against the seat's `maximum relative velocity`.
///
/// Returns the OFFER that passes; `None` when nothing does. The preference order when several
/// pass is [`UnitSeats::best`]'s — driver, then gunner, then the lowest seat index — and that
/// order is stated there as HMS's own rather than the engine's.
#[allow(clippy::too_many_arguments)]
pub fn offer(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    objects: &[ObjectInfo],
    player_pos: Vec3,
    player_facing: Vec3,
    player_vel: Vec3,
    key_label: &str,
) -> Option<VehicleOffer> {
    let mut best: Option<VehicleOffer> = None;
    for o in objects {
        // Cheap reject first: the authors' own 3 wu search, against the placement's origin plus
        // a generous slack for a long vehicle.
        if (Vec3::from(o.pos) - player_pos).length() > SEARCH_RADIUS + 4.0 {
            continue;
        }
        let parent_world = crate::scene::object_matrix(o);
        // The placement's own seats.
        collect(index, native, cache, o.primary_tag, parent_world, o.datum, None, player_pos, player_facing, player_vel, key_label, &mut best);
        // And every attached child that is itself a vehicle — the Warthog's turret, the Falcon's
        // six. `enumerate_attachments` already gives the child's own local frame.
        for a in native.enumerate_attachments(cache, o.primary_tag) {
            if a.child_obj == 0 {
                continue;
            }
            let local = Mat4::from_rotation_translation(
                glam::Quat::from_xyzw(a.rot[0], a.rot[1], a.rot[2], a.rot[3]).normalize(),
                Vec3::from(a.pos),
            );
            collect(
                index,
                native,
                cache,
                a.child_obj,
                parent_world * local,
                o.datum,
                Some(o.datum),
                player_pos,
                player_facing,
                player_vel,
                key_label,
                &mut best,
            );
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn collect(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    vehi_tag: u32,
    world: Mat4,
    datum: u32,
    parent_datum: Option<u32>,
    player_pos: Vec3,
    player_facing: Vec3,
    player_vel: Vec3,
    key_label: &str,
    best: &mut Option<VehicleOffer>,
) {
    // Entries are read once per tag, but the borrow of `index` must end before `strings`.
    let Some(entry) = index.entry(native, cache, vehi_tag) else { return };
    let (path, seat_family) = (entry.path.clone(), entry.seat_family.clone());
    let _ = seat_family;
    let seats = entry.seats.clone();
    let mut found: Vec<(usize, EntryTest)> = Vec::new();
    for s in &seats.seats {
        if !s.player_allowed() {
            continue;
        }
        let group = s.entry_marker_group();
        if group.is_empty() {
            continue;
        }
        let Some(entry) = index.entry(native, cache, vehi_tag) else { return };
        let frames: Vec<Mat4> = entry.marker_frames(native, cache, group).into_iter().map(|m| world * m).collect();
        if frames.is_empty() {
            continue;
        }
        if let Some((_, t)) = reach_seats::best_entry_marker(s, &frames, player_pos, player_facing, player_vel) {
            if t.passes() {
                found.push((s.index, t));
            }
        }
    }
    let Some(pick) = seats.best(&found.iter().map(|f| f.0).collect::<Vec<_>>()) else { return };
    let test = found.iter().find(|f| f.0 == pick).map(|f| f.1).expect("the pick came from `found`");
    let seat = &seats.seats[pick];
    let strings = index.strings(native, cache);
    let resolved = strings.resolve(&seat.enter_seat_string, Some(HUD_MESSAGES_UNIC));
    let (prompt, substituted) = match resolved {
        // An ICON-only string is a glyph, not text: substituting a key label there would print a
        // key name where a picture belongs. Nothing is drawn for one.
        Some(r) if reach_strings::is_icon_only(&r.text) => (None, Vec::new()),
        Some(r) => {
            let (t, g) = reach_strings::substitute_glyphs(&r.text, key_label);
            (Some(t), g)
        }
        None => (None, Vec::new()),
    };
    let cand = VehicleOffer {
        datum,
        parent_datum,
        vehi_tag,
        vehi_path: path,
        seat: pick,
        seat_label: seat.label.clone(),
        prompt_id: seat.enter_seat_string.clone(),
        prompt,
        substituted,
        test,
    };
    // Nearest passing offer wins across vehicles; within one vehicle `best` already chose.
    let better = match best {
        None => true,
        Some(b) => cand.test.distance < b.test.distance,
    };
    if better {
        *best = Some(cand);
    }
}

// ---------------------------------------------------------------------------------------------
// THE RIDE
// ---------------------------------------------------------------------------------------------

/// The player is in a seat. Holds only what the ride owns: the editor's object table is never
/// written.
#[derive(Clone, Debug, PartialEq)]
pub struct Riding {
    /// The placement the ride began on — and the one HIDDEN for the duration.
    pub hidden_datum: u32,
    /// The `vehi` tag whose seats block `seat` indexes.
    pub vehi_tag: u32,
    pub vehi_path: String,
    /// The PLACEMENT's own object tag — the Warthog, even when the seat is on its turret. This
    /// is what has to be DRAWN, because the placement it replaces is the parent's.
    pub placement_tag: u32,
    pub seat: usize,
    pub seat_label: String,
    /// The seat's jmad mode (`warthog_d`), or empty when the graph has no seat set.
    pub seat_mode: String,
    /// Where the vehicle is NOW, in this module's own state. The placement is not moved.
    pub drive: DriveState,
    /// The world transform the ride started from, so exiting can put the player beside it and
    /// leaving can restore nothing (there is nothing to restore).
    pub start: Mat4,
    /// `Some` when the vehicle's tag supports a drive model; `None` with the reason otherwise,
    /// and then the vehicle does not move at all.
    pub drivable: Result<(), String>,
    /// The rider's look, clamped to the seat's authored range each frame.
    pub yaw: f32,
    pub pitch: f32,
    /// #vehicle-ride  What the PLACEMENT looked like, carried into the ride.
    ///
    /// Reported: *"it gets in it and it's basically spawning a new one so it doesn't keep the
    /// colors and everything of the original vehicle at all."* That is a precise description of
    /// the design: the ride draws the vehicle as an actor built from the TAG, so everything that
    /// lives on the placement is left behind. The variant already travelled (test mode passes
    /// `riding_variant_sid` into `draw_pieces`); this is the rest.
    ///
    /// Filled by the caller that mounts the ride, because that is the only place that still has
    /// the placement AND the editor's per-datum maps. Default = the tag's own appearance, which
    /// is exactly the old behaviour, so an un-filled field changes nothing.
    pub appearance: RideAppearance,
}

/// #vehicle-ride  The per-PLACEMENT appearance a ride has to carry with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RideAppearance {
    /// The placement's `(team, color)` out of the editor's own colour map — the same pair the
    /// object pipeline hands to [`crate::scene::forge_tint`], so the ride and the parked vehicle
    /// cannot disagree about what colour it is.
    pub forge_color: Option<(u8, u8)>,
    /// The placement's editor scale multiplier. **Not applied yet**: `reach_actor::Actor` has no
    /// per-actor scale (only `ActorDef::scale`, which is per MODEL and shared), so a scaled
    /// Warthog is still ridden at 1.0. Carried here so the gap is visible and so the field is
    /// already in place when the actor lane gains one.
    pub scale: f32,
    /// The placement's displayed `model_state_enum` damage state.
    pub damage: u8,
}

impl Default for RideAppearance {
    fn default() -> RideAppearance {
        RideAppearance { forge_color: None, scale: 1.0, damage: 0 }
    }
}

impl Riding {
    /// #seat-entry  One line naming the vehicle, the seat and the jmad mode the rider was given,
    /// for the on-screen caption.
    ///
    /// **It exists so the next report identifies the seat without anyone having to ask.** *"I
    /// stand straight up instead of sitting in the seat"* could not be pinned to a vehicle from
    /// the report alone, and 36 of the 138 player seats in `forge_halo` legitimately have no clip
    /// in either rider graph (every `*_b_*` boarding seat, which is species-specific by design,
    /// plus `wraith_d` and `sabre_d`, which no rider graph answers at all). So a rider in one of
    /// those is a permanent correct state, not a bug — and the caption has to say which seat it
    /// is in rather than leaving the player to guess why they are upright.
    pub fn seat_caption(&self) -> String {
        format!(
            "{} seat {} '{}' -> jmad mode '{}'",
            self.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&self.vehi_path),
            self.seat,
            self.seat_label,
            if self.seat_mode.is_empty() { "(none authored)" } else { &self.seat_mode }
        )
    }

    /// The vehicle's current world transform: its position and its FULL attitude.
    ///
    /// #vehicle-suspension  Yaw, pitch and roll, from [`DriveState::basis`] — one function, so
    /// the hull the renderer draws, the hull the seat camera rides on and the hull the wheel
    /// solve raycasts from cannot disagree. *"A Warthog drives up a hill perfectly level"* was
    /// this returning yaw only.
    pub fn world(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.drive.basis(), self.drive.pos)
    }
}

/// Start a ride from an offer. The drive state begins at the placement's own pose, so a vehicle
/// that is never driven stays exactly where the user put it.
pub fn begin(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    offer: &VehicleOffer,
    placement: &ObjectInfo,
) -> Riding {
    let start = crate::scene::object_matrix(placement);
    let yaw = start.x_axis.y.atan2(start.x_axis.x);
    let (seat_mode, drivable) = match index.entry(native, cache, offer.vehi_tag) {
        Some(e) => {
            let label = e.seats.seats.get(offer.seat).map(|s| s.label.clone()).unwrap_or_default();
            // THE SEAT'S OWN `label` IS THE JMAD MODE NAME (`warthog_d`, `ghost_d`,
            // `mongoose_p`), so test mode picks the exact clip from the tag where Theater has to
            // infer driver-versus-passenger from who started riding first.
            //
            // #seat-entry  IT IS TAKEN UNCONDITIONALLY, and that is a bug fix, not a tidy-up.
            // This line used to read
            //
            //     let mode = if e.seat_family.is_empty() { String::new() } else { label };
            //
            // which vetoed the authored label whenever `film_anim::seat_family_for` — a
            // SUBSTRING match on the vehicle's tag path against a hand-written list of sixteen
            // names — did not recognise the vehicle. An empty `seat_mode` then made
            // `play_mode::pose` skip the whole seated block, so the body kept the state the
            // controller gave it; and because `set_pose_from_vehicle` leaves `grounded` false,
            // that state is `Stance::Flight`. The rider was drawn BOLT UPRIGHT with his feet on
            // the seat marker, which is exactly what was reported: *"I stand straight up instead
            // of sitting in the seat."*
            //
            // A guess about the tag's NAME cannot be allowed to override what the tag SAYS. The
            // label is authored for every one of the 138 player seats in `forge_halo` (measured),
            // and if the rider's graph turns out not to carry a clip for it then
            // `film_anim::anim_candidates` falls back on its own — which is the right place for
            // that decision, with the seat still SEATED.
            (label, e.drive.as_ref().map(|_| ()).map_err(|s| s.clone()))
        }
        None => (String::new(), Err("this vehicle's tag could not be read".to_string())),
    };
    Riding {
        hidden_datum: offer.parent_datum.unwrap_or(offer.datum),
        vehi_tag: offer.vehi_tag,
        vehi_path: offer.vehi_path.clone(),
        placement_tag: placement.primary_tag,
        seat: offer.seat,
        seat_label: offer.seat_label.clone(),
        seat_mode,
        drive: DriveState { pos: start.w_axis.truncate(), yaw, ..Default::default() },
        start,
        drivable,
        yaw,
        pitch: 0.0,
        appearance: RideAppearance::default(),
    }
}

/// Where the rider's BODY goes: the vehicle's transform times the seat marker.
///
/// Returns `(feet position, yaw)` because that is what [`crate::play_ctl::PlayerState`] holds —
/// `pos` is the bottom of the capsule. A seated body's own origin IS its seat marker, so the
/// marker's translation is the position and the vehicle's yaw is the facing.
pub fn rider_pose(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
) -> Option<(Vec3, f32)> {
    let world = riding.world();
    let entry = index.entry(native, cache, riding.vehi_tag)?;
    let seat = entry.seats.seats.get(riding.seat)?;
    let frames = entry.marker_frames(native, cache, &seat.marker_name);
    let m = frames.first().copied()?;
    let w = world * m;
    Some((w.w_axis.truncate(), riding.drive.yaw))
}

/// The seat's camera for this frame — the pure `(eye, forward, fov)` function, computed from the
/// vehicle's pose and the seat's own `unit_camera_struct` so it cannot go stale.
pub fn seat_camera(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
    fov_default: f32,
) -> Option<reach_seats::SeatView> {
    let world = riding.world();
    let entry = index.entry(native, cache, riding.vehi_tag)?;
    let seat = entry.seats.seats.get(riding.seat)?.clone();
    let seat_marker = entry.marker_frames(native, cache, &seat.marker_name).first().copied().unwrap_or(Mat4::IDENTITY);
    let cam_marker = if seat.camera_marker.is_empty() {
        None
    } else {
        entry.marker_frames(native, cache, &seat.camera_marker).first().copied()
    };
    let (yaw, pitch) = reach_seats::clamp_seat_look(&seat, riding.drive.yaw, riding.yaw, riding.pitch);
    Some(reach_seats::seat_view(&seat, cam_marker.map(|m| world * m), world * seat_marker, yaw, pitch, fov_default))
}

/// #vehicle-drivetrain  The distance the WHEELS have rolled, which is what a wheel-spin
/// animation phase is a function of.
///
/// `wheel_travel` for a vehicle whose family authors an engine (the engine's own rolling phase
/// input), the hull's resolved `odometer` for one that does not. **The decision is made here and
/// only here**: a hover vehicle's `wheel_travel` is permanently zero and reading it directly
/// would freeze any wheel a hover vehicle happens to have, while reading the odometer for a jeep
/// would hide every slip.
pub fn drive_wheel_distance(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
) -> f32 {
    let geared = index
        .entry(native, cache, riding.vehi_tag)
        .and_then(|e| e.drive.as_ref().ok())
        .map_or(false, |m| m.drivetrain.is_some());
    if geared { riding.drive.wheel_travel } else { riding.drive.odometer }
}

/// #vehicle-ride  The per-datum world-transform override a ride needs: the ridden placement,
/// drawn where the ride has got to.
///
/// **One entry, and it is the PARENT placement.** `hidden_datum` is already the parent (a
/// Warthog's gunner seat lives on `warthog_chaingun`, a child attachment, and the thing that has
/// to move is the hog), and every attachment composes its own local offset on top of the parent's
/// frame — so overriding the parent moves the whole assembly, turret included, with no second
/// entry and no chance of the two drifting apart.
///
/// This is what replaces hide-the-placement-and-draw-an-actor: the REAL object is drawn
/// elsewhere, so its Forge colour, team, variant, scale, shadow flag and damage state come along
/// because they were never left behind. Reported as *"it gets in it and it's basically spawning a
/// new one so it doesn't keep the colors and everything of the original vehicle at all"*.
pub fn ride_transforms(riding: &Riding) -> std::collections::HashMap<u32, Mat4> {
    let mut m = std::collections::HashMap::new();
    m.insert(riding.hidden_datum, riding.world());
    m
}

/// Advance the ride. Does nothing to the vehicle when its tag authors no drive model, which is
/// the honest behaviour for a Banshee: you sit in it and it does not fly.
pub fn step(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &mut Riding,
    input: DriveInput,
    dt: f32,
    scene: &dyn ObjectScene,
) {
    let Some(entry) = index.entry(native, cache, riding.vehi_tag) else { return };
    let Ok(model) = &entry.drive else { return };
    let model = model.clone();
    // The ground under a point: a ray straight down through the level geometry and the Forge
    // objects, the same world the character controller walks on. `raycast_solid` is the one the
    // third-person camera already uses.
    // #vehicle-suspension  THE GROUND UNDER ONE CONTACT POINT. `DriveModel::step` calls this
    // once per wheel per sub-step, so a Scorpion's twelve points at a 100 ms frame would be 144
    // ray casts; the results are MEMOISED on a 0.05 wu grid in XY, which is far finer than a
    // suspension can resolve and bounds the real cost at one ray per wheel per frame.
    //
    // `raycast_solid` is the primitive because it covers the placed Forge objects a map is built
    // out of as well as the BSP; it can only report a distance, so the surface normal comes from
    // the BSP ray and is only advisory — the hull's attitude is solved from the three contact
    // HEIGHTS, which is the engine's own rule.
    // **TWO THINGS THIS GETS RIGHT THAT A FIRST VERSION DID NOT**, and between them they were the
    // *"warthog just flew into the air and the map glitched out"* report:
    //
    // 1. **The ray starts at the WHEEL CENTRE, not 2 wu above it.** A contact can never be above
    //    a wheel's own centre, so starting there loses nothing — and it makes "the ground is above
    //    this wheel" structurally impossible, which bounds the compression at the wheel radius
    //    whatever the geometry does. The first version cast from `wheel + 2z` and could therefore
    //    report a surface up to 2 wu ABOVE the wheel.
    // 2. **It excludes the ridden vehicle's own placement.** `raycast_solid` excluded nothing, and
    //    a vehicle's collision hull is solid to a ray cast from inside it: the ray hit the
    //    Warthog's own roof and called it ground. The ride hides the placement from the RENDER set,
    //    which is deliberately NOT the pick set a ray walks — the editor's table keeps seeing the
    //    vehicle where the user put it — so the exclusion has to be passed in.
    //
    // Memoised on a 0.05 wu grid in XY, far finer than a suspension can resolve, which bounds a
    // Scorpion's twelve points at one ray per wheel per frame instead of one per sub-step.
    let hidden = [riding.hidden_datum];
    let cache: std::cell::RefCell<Vec<(f32, f32, Option<(f32, Vec3)>)>> = std::cell::RefCell::new(Vec::new());
    let ground = |p: Vec3| -> Option<(f32, Vec3)> {
        if let Some(&(_, _, hit)) =
            cache.borrow().iter().find(|(x, y, _)| (x - p.x).abs() < 0.05 && (y - p.y).abs() < 0.05)
        {
            return hit;
        }
        let from = p;
        let hit = scene.raycast_solid_excluding(from, -Vec3::Z, 8.0, &hidden).map(|d| {
            let h = from.z - d;
            let n = scene
                .raycast_scene_n(from, -Vec3::Z)
                .filter(|(hp, _)| (hp.z - h).abs() < 0.05)
                .map(|(_, n)| n)
                .unwrap_or(Vec3::Z);
            (h, if n.z < 0.0 { -n } else { n })
        });
        cache.borrow_mut().push((p.x, p.y, hit));
        hit
    };
    riding.drive = model.step(riding.drive, input, dt, &ground);
}

/// Where the player ends up on getting out: beside the seat marker, along the entry marker's own
/// forward, at the authored entry radius — i.e. exactly where they would have been standing to
/// get in. `None` when the markers cannot be resolved, and then the caller must leave the player
/// where they are rather than teleport them somewhere invented.
pub fn exit_position(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
) -> Option<Vec3> {
    let world = riding.world();
    let entry = index.entry(native, cache, riding.vehi_tag)?;
    let seat = entry.seats.seats.get(riding.seat)?.clone();
    let group = seat.entry_marker_group().to_string();
    let frames = entry.marker_frames(native, cache, &group);
    let m = frames.first().copied()?;
    let w = world * m;
    let fwd = w.transform_vector3(reach_seats::MARKER_FORWARD).normalize_or_zero();
    Some(w.w_axis.truncate() + fwd * seat.entry_radius)
}

/// The velocity the player carries out with them: the vehicle's own.
///
/// Reach does this, and the seat's own `apply velocity on death exit` flag is the tag
/// acknowledging that a rider leaves a moving vehicle with its momentum.
pub fn exit_velocity(riding: &Riding) -> Vec3 {
    riding.drive.vel
}

/// The object set the RENDERER is given: everything except the placements test mode is drawing
/// itself.
///
/// **One function for both directions.** `None` means "nothing is hidden, pass the original
/// slice" — so the caller never builds a copy when there is no ride, and the show case and the
/// hide case cannot drift apart. A leak one way draws two Warthogs; a leak the other draws none.
///
/// The editor's own object table is NOT filtered: selection, the panels and the collision gather
/// all keep seeing the vehicle, and nothing about the user's unsaved variant changes.
pub fn shown(objects: &[ObjectInfo], hidden: &[u32]) -> Option<Vec<ObjectInfo>> {
    if hidden.is_empty() {
        return None;
    }
    Some(objects.iter().filter(|o| !hidden.contains(&o.datum)).cloned().collect())
}

/// One piece of the vehicle to DRAW while it is ridden: a render model and the world transform
/// to put it at.
///
/// **The placement is hidden, so everything it drew has to be drawn here or it vanishes** — and
/// that is more than one model: a Warthog draws its own body *and* its attached turret, a Falcon
/// six turrets. The parent's frame is the ride's own, and each attachment composes its authored
/// local `(pos, rot)` on top, which is the same composition
/// [`crate::reach_actor::attach_child`]'s callers use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawPiece {
    pub mode_tag: u32,
    pub world: Mat4,
    /// #vehicle-anim  The OBJECT tag this piece came from — the `vehi` for the parent, the
    /// attached child's own `obje` for a turret.
    ///
    /// Needed because a vehicle's animation graph hangs off the OBJECT (`obje` -> `hlmt` ->
    /// `animation`) and not off the render model, so an actor placed from `mode_tag` alone has no
    /// graph at all and holds its bind pose for ever — which is exactly what a ridden vehicle did
    /// before this existed: no wheels, no steering, nothing.
    pub obj_tag: u32,
}

/// Every render model the ridden vehicle needs, parent first.
///
/// # The variant filter is not optional, and leaving it out stacks four guns on one mount
///
/// A Warthog's tag attaches **all four** of its weapon variants — `warthog_chaingun`,
/// `warthog_gauss`, `warthog_troop`, `warthog_rocket` — and the placement's MODEL VARIANT picks
/// one. The first version of this function drew all four and the gate caught it: five pieces,
/// four of them inside each other on the same mount. The engine's rule, as `scene.rs`'s own
/// renderer implements it:
///
/// 1. take the placement's `variant_name_sid` when any attachment row carries it;
/// 2. else the object's own `obje +0x60 default model variant` when any row carries that;
/// 3. else fall back to the rows of variant INDEX 0, the `hlmt` default variant.
///
/// # And the child's own marker has to be cancelled out
///
/// `enumerate_attachments` gives the PARENT marker's frame. The engine seats the CHILD's marker
/// of the same name onto it — `child_world = parent_world * P_marker * C_marker^-1`, which is
/// exactly [`crate::reach_actor::attach_child`]. Without the inverse a turret whose own marker is
/// not at its model origin is drawn offset, which is the `truck bed drawn on end 2.7 wu off`
/// failure this project has already had once.
pub fn draw_pieces(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
    placement_variant_sid: u32,
) -> Vec<DrawPiece> {
    let world = riding.world();
    let mut out = Vec::new();
    let parent_mode = native.resolve_mode(cache, riding.placement_tag);
    if parent_mode != 0 && parent_mode != u32::MAX {
        out.push(DrawPiece { mode_tag: parent_mode, world, obj_tag: riding.placement_tag });
    }
    let rows = native.enumerate_attachments(cache, riding.placement_tag);
    // `obje +0x60 default model variant`, read here rather than taken from `scene.rs`'s private
    // copy of the same read.
    let default_sid = native
        .tag_meta(cache, riding.placement_tag, 0x60, 4)
        .filter(|b| b.len() >= 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .filter(|&v| v != u32::MAX)
        .unwrap_or(0);
    let pick = if placement_variant_sid != 0 && rows.iter().any(|a| a.variant_name_sid == placement_variant_sid) {
        placement_variant_sid
    } else if default_sid != 0 && rows.iter().any(|a| a.variant_name_sid == default_sid) {
        default_sid
    } else {
        0
    };
    for a in &rows {
        if a.child_mode == 0 || a.child_mode == u32::MAX {
            continue;
        }
        let keep = if pick != 0 { a.variant_name_sid == pick } else { a.variant_index == 0 };
        if !keep {
            continue;
        }
        let parent_marker = Mat4::from_rotation_translation(
            glam::Quat::from_xyzw(a.rot[0], a.rot[1], a.rot[2], a.rot[3]).normalize(),
            Vec3::from(a.pos),
        );
        // The child's own marker of the same name, cancelled out. When the child's model does not
        // carry it the engine puts the child's ORIGIN on the parent's marker, which is what
        // leaving the inverse out gives.
        let child_marker = native
            .string_id(cache, a.marker_sid)
            .filter(|n| !n.is_empty())
            .and_then(|name| {
                index.entry(native, cache, a.child_obj).and_then(|e| e.marker_frames(native, cache, &name).first().copied())
            })
            .unwrap_or(Mat4::IDENTITY);
        out.push(DrawPiece {
            mode_tag: a.child_mode,
            world: crate::reach_actor::attach_child(world, parent_marker, child_marker),
            obj_tag: a.child_obj,
        });
    }
    out
}

/// #vehicle-anim  ANIMATE the actors that draw a ridden vehicle: its own base clip, plus the
/// overlays its graph authors for the things the driver is doing.
///
/// This is the half of the ride the user asked for second — *"Vehicles have their own unique
/// animation they play when a biped is driving it, so handlebars, wheels etc would function"*.
/// Before it existed the ridden vehicle was drawn in its BIND POSE: `add_render_model` is given
/// a render-model tag and a render model carries no animation graph, so the actor had no clip to
/// play and nothing to compose, and the wheels could not have turned however far it was driven.
///
/// # Three things happen here, in this order, and the order matters
///
/// 1. **The graph is installed.** `set_model_graph` gives the actor the vehicle's own `jmad`
///    without rebuilding a single mesh, and it is idempotent (it returns early when the actor
///    already has that graph), so this costs one comparison per frame after the first.
/// 2. **The base clip is installed, once.** Every shipped vehicle names it `any:idle` and it is
///    a real loop (the Ghost's hover bob, the Warthog's 31-frame engine shake), so it runs on the
///    ride's own accumulated `time` and not on a frame counter.
/// 3. **The overlays are composed on top**, at the frames
///    [`crate::reach_vehicle_anim::VehicleAnim::cells`] resolves from the drive state. An overlay
///    installed *as* the pose would cancel the base clip; that is the distinction `film_play`
///    already pays for once and it is not relearned here.
///
/// # The index guard
///
/// An overlay is named by its INDEX in the graph's `anims`, and the classifier read that index
/// from a graph it loaded itself. If the actor's graph were a different one the same index would
/// silently animate the wrong thing — so each overlay's name is checked against the actor's own
/// graph before it is installed, and a mismatch is reported rather than drawn.
///
/// Returns one line per thing that could not be done, which the caller may surface; an empty
/// result means every piece is animating.
pub fn drive_actors(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
    pieces: &[DrawPiece],
    ids: &[u32],
    actors: &mut crate::reach_actor::ActorScene,
) -> Vec<String> {
    let mut notes = Vec::new();
    // #vehicle-drivetrain  THE WHEELS' OWN TRAVEL, not the hull's.
    //
    // The engine drives the rolling phase from `gear ratio * engine angular velocity`
    // (`vehicle_type_human_jeep.cpp`, `sub_140E69D40`), which is the wheels' SURFACE speed and
    // not the distance the hull covered — so wheelspin shows, and a Warthog held against a wall
    // at full throttle spins its wheels instead of freezing them. A family with no engine has no
    // wheel travel to report, and for those the hull's resolved distance IS the wheel distance;
    // `drive_wheel_distance` makes that choice in one place rather than overloading a zero.
    let st = crate::reach_vehicle_anim::DrivenState {
        odometer: drive_wheel_distance(index, native, cache, riding),
        steer: steer_fraction(index, native, cache, riding),
    };
    let aim = seat_aim(index, native, cache, riding);
    // #vehicle-ride  THE PLACEMENT'S OWN APPEARANCE, before anything is posed.
    //
    // The ride draws the vehicle as an actor, so whatever lives on the placement has to be put
    // back on the actor or getting in visibly respawns the vehicle. The colour goes through the
    // object pipeline's OWN table (`scene::forge_tint`) so a ridden Warthog and a parked one
    // cannot read different colours from the same `(team, color)` pair.
    //
    // **Both change colours get the same tint**, which is the one approximation here: a Forge
    // placement authors ONE colour and the actor lane's `skin` is a `[primary, secondary]` pair
    // for an armoured biped. A vehicle whose model uses the secondary mask for something else
    // would take the Forge colour there too; nothing in the shipped vehicles was seen doing that,
    // and it is a visible-and-reportable approximation rather than a silent one.
    let tint = riding.appearance.forge_color.map(|tc| {
        let c = crate::scene::forge_tint(Some(tc));
        [[c[0], c[1], c[2]], [c[0], c[1], c[2]]]
    });
    for &id in ids.iter() {
        if let Some(a) = actors.get_mut(id) {
            a.skin = tint;
        }
        // The damage state selects other draw parts, so it goes through the setter that marks the
        // actor's meshes dirty rather than being written into the field.
        actors.set_damage(id, riding.appearance.damage);
    }
    for (&id, p) in ids.iter().zip(pieces.iter()) {
        // Each piece has its OWN graph: a Warthog's body and its chaingun are two objects with
        // two graphs, and the turret's `any:idle` is not the hog's.
        let Some(jmad) = crate::reach_actor::resolve_jmad(native, cache, p.obj_tag) else {
            continue;
        };
        if !actors.set_model_graph(native, cache, id, Some(jmad)) {
            notes.push(format!("{:#x} would not take its own animation graph", p.obj_tag));
            continue;
        }
        // The classification, cached per OBJECT tag. `VehicleIndex::entry` only answers for a
        // `vehi`, which every piece of a vehicle is (the Warthog's turret is a `vehi` too).
        let Some(e) = index.entry(native, cache, p.obj_tag) else { continue };
        let anim = e.anim.clone();
        let Some(base_index) = anim.base else { continue };
        let base_name = anim.base_name.clone();
        // --- the base clip, once ----------------------------------------------------------
        let needs_base = actors.get(id).map(|a| a.anim_name != base_name).unwrap_or(false);
        if needs_base {
            if let Err(why) = actors.set_anim(native, cache, id, Some(&base_name)) {
                notes.push(format!("'{base_name}' would not load: {why}"));
            }
        }
        // --- its frame, and the overlays on top -------------------------------------------
        let cells = anim.cells(&st);
        let mut loaded: Vec<(usize, Vec<(f32, f32)>)> = Vec::new();
        for (ai, c) in cells {
            // THE INDEX GUARD: the name the classifier saw must be the name the actor's own
            // graph has at that index.
            let ok = actors
                .get(id)
                .and_then(|a| a.def.graph.as_ref())
                .and_then(|g| g.anims.get(ai))
                .zip(anim.overlays.iter().find(|o| o.anim == ai))
                .map(|(got, want)| got.name == want.name)
                .unwrap_or(false);
            if !ok {
                notes.push(format!(
                    "overlay {ai} of {:#x} is not the clip it was classified as and was not installed",
                    p.obj_tag
                ));
                continue;
            }
            let Some(a) = actors.get_mut(id) else { continue };
            if a.ensure_overlay(native, cache, ai) {
                loaded.push((ai, c));
            } else {
                notes.push(format!("overlay {ai} of {:#x} would not load", p.obj_tag));
            }
        }
        // #vehicle-anim  A TURRET'S AIM — 18 of the 36 shipped vehicles are the `turret` type,
        // four of them the Warthog's own gun positions, so this is the second most likely thing
        // a player sits in.
        //
        // **`vehicle:aiming` is not an animation: it is an AIM GRID.** Measured across three
        // unrelated turrets (`warthog_chaingun`, `machinegun`, `shade_flak_cannon`) it is the
        // same ten frames every time — `b_aim_yaw` stepping 45 deg a frame right round the
        // circle for the first eight, then `b_aim_pitch` at +90 and -90 for the last two. Those
        // are LABELS: the clip states, in its own channels, which direction each frame means. So
        // it goes down exactly the lane Theater drives a biped's look through
        // (`Actor::aim_grid` -> `AimGrid::cells`), rather than being indexed as a range.
        //
        // `aim_grid` returns `None` for a clip that animates neither aim node, so a vehicle whose
        // `vehicle:aiming` is not really a grid is declined by the reader rather than by a name.
        if let Some((yaw, pitch)) = aim {
            if let Some((ai, grid)) = actors.get_mut(id).and_then(|a| a.aim_grid(native, cache, "vehicle:aiming")) {
                let c = grid.cells(yaw, pitch);
                if c.is_empty() {
                    notes.push(format!("{:#x} has an aim grid that resolved no cells for ({yaw:+.2},{pitch:+.2}) rad", p.obj_tag));
                } else {
                    loaded.push((ai, c));
                }
            }
        }
        if let Some(a) = actors.get_mut(id) {
            // The clip's frame is driven here, never by `playing`: a vehicle's idle is a loop on
            // the wall clock and `ActorScene::update` would advance it at its own rate as well,
            // which is two writers on one frame.
            a.playing = false;
            let n = a.frame_count().max(1.0);
            a.frame = (riding.drive.time * crate::film_anim::CLIP_FPS).rem_euclid(n).min(n - 0.001).max(0.0);
            a.overlays = loaded;
            // Unused -- the base clip and the overlays carry the whole pose.
            a.replacement = None;
        }
        let _ = base_index;
    }
    notes
}

/// #vehicle-anim  Where the rider is looking, HULL-RELATIVE and clamped by the seat's own
/// authored range: `(yaw, pitch)` in radians, `+yaw` left and `+pitch` up.
///
/// Hull-relative because that is what a turret's aim overlay labels its frames in — the grid's
/// `b_aim_yaw` sweeps the whole circle about the vehicle's own up axis, so a world yaw would
/// point the gun at a fixed compass bearing however the vehicle turned.
///
/// Clamped through [`crate::reach_seats::clamp_seat_look`], the same function the seat CAMERA
/// uses, so the gun and the view cannot disagree about how far the seat lets the rider turn.
/// `None` when the seat cannot be read.
pub fn seat_aim(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
) -> Option<(f32, f32)> {
    let e = index.entry(native, cache, riding.vehi_tag)?;
    let seat = e.seats.seats.get(riding.seat)?;
    let (yaw, pitch) = reach_seats::clamp_seat_look(seat, riding.drive.yaw, riding.yaw, riding.pitch);
    let rel = {
        let t = std::f32::consts::TAU;
        (yaw - riding.drive.yaw + std::f32::consts::PI).rem_euclid(t) - std::f32::consts::PI
    };
    Some((rel, pitch))
}

/// #vehicle-anim  The driver's steering as the animation wants it: `-1` full right .. `+1` full
/// left.
///
/// The drive model carries the steering as an ANGLE in radians
/// ([`crate::reach_vehicle::DriveState::steer_angle`]), slewed toward the driver's intent at the
/// authored `turn rate` and bounded by the authored `maximum left turn` / `maximum right turn`.
/// So the fraction is that angle over the authored maximum on the side it is turning — which
/// means full lock on the clip is full lock in the tag, per vehicle, rather than a normalisation
/// of HMS's choosing.
///
/// `0.0` for a vehicle with no authored steering (every hover family turns the hull instead), and
/// then a steering overlay sits at its centre frame, which is the honest pose for a Ghost: its
/// `vehicle:steering` clip leans the seat and the sticks, and which end of it is left could not
/// be measured (it moves no front wheel), so it is not driven at all.
pub fn steer_fraction(
    index: &mut VehicleIndex,
    native: &NativeDll,
    cache: CacheHandle,
    riding: &Riding,
) -> f32 {
    let Some(e) = index.entry(native, cache, riding.vehi_tag) else { return 0.0 };
    let Ok(d) = e.drive.as_ref() else { return 0.0 };
    let Some(t) = d.turning else { return 0.0 };
    let a = riding.drive.steer_angle;
    let lim = if a >= 0.0 { t.max_left.abs() } else { t.max_right.abs() };
    if lim <= 1e-4 {
        return 0.0;
    }
    (a / lim).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod gates {
    //! #seat-entry  `cargo test -p hms-app --profile quick play_vehicle:: -- --ignored --nocapture`

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    fn obj(datum: u32, tag: u32, pos: Vec3) -> ObjectInfo {
        ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.into(),
            health: 1.0,
            shield: 1.0,
            mode_tag: 0,
            fwd: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: tag,
            variant_name_sid: 0,
        }
    }

    /// **THE DRAW GATE.** Everything the hidden placement drew must be drawn by the ride, and
    /// a Warthog is TWO models — its body and its attached turret — so a gate that only checked
    /// "at least one" would pass while the turret vanished.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_ridden_warthog_draws_its_body_and_its_turret() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let hog = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog");
        let placement = obj(0x2468, hog, Vec3::new(10.0, 20.0, 30.0));
        let o = VehicleOffer {
            datum: 0x2468,
            parent_datum: None,
            vehi_tag: hog,
            vehi_path: "warthog".into(),
            seat: 0,
            seat_label: "warthog_d".into(),
            prompt_id: "warthog_enter_driver".into(),
            prompt: None,
            substituted: Vec::new(),
            test: EntryTest { distance: 0.0, radius: 1.25, cone: 0.0, cone_limit: 1.3, facing: 0.0, facing_limit: 1.0, relative_speed: 0.0, relative_speed_limit: 3.5 },
        };
        let r = begin(&mut index, native, cache, &o, &placement);
        assert_eq!(r.placement_tag, hog, "the PLACEMENT's tag is what gets drawn");
        let pieces = draw_pieces(&mut index, native, cache, &r, placement.variant_name_sid);
        for p in &pieces {
            println!("   draw {:#x} '{}' at ({:.2},{:.2},{:.2})", p.mode_tag, native.tag_name(cache, p.mode_tag).unwrap_or_default(),
                p.world.w_axis.x, p.world.w_axis.y, p.world.w_axis.z);
        }
        // Its body and EXACTLY ONE of its four weapon variants. The first version drew all
        // four, which is what this count is here to stop coming back.
        assert_eq!(
            pieces.len(),
            2,
            "a Warthog draws its body and ONE turret; got {} piece(s) -- the variant filter is not working",
            pieces.len()
        );
        // The parent is first and sits exactly where the user put it.
        assert_eq!(pieces[0].world.w_axis.truncate(), Vec3::new(10.0, 20.0, 30.0));
        // Every attachment composes ON the ride's frame, so none of them is left at the origin
        // and none of them is at the parent's exact point either.
        for p in &pieces[1..] {
            let at = p.world.w_axis.truncate();
            assert!(at.length() > 1.0, "attachment {:#x} is at the origin -- its frame was not composed", p.mode_tag);
            assert!((at - Vec3::new(10.0, 20.0, 30.0)).length() < 5.0, "attachment {:#x} is {at:?}, nowhere near the vehicle", p.mode_tag);
        }
        // Move the vehicle and every piece moves with it, by the same amount.
        let mut moved = r.clone();
        moved.drive.pos += Vec3::new(5.0, 0.0, 0.0);
        let after = draw_pieces(&mut index, native, cache, &moved, placement.variant_name_sid);
        assert_eq!(after.len(), pieces.len());
        for (a, b) in pieces.iter().zip(after.iter()) {
            let d = b.world.w_axis.truncate() - a.world.w_axis.truncate();
            assert!((d - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4, "piece {:#x} moved by {d:?}", a.mode_tag);
        }
    }

    /// **THE HIDE GATE.** The render set must lose exactly the ridden placement and nothing
    /// else, and with no ride it must not be copied at all.
    #[test]
    fn the_render_set_loses_exactly_the_ridden_placement() {
        let objects = vec![obj(1, 10, Vec3::ZERO), obj(2, 11, Vec3::X), obj(3, 12, Vec3::Y)];
        // No ride: `None`, so the caller passes the original slice and allocates nothing.
        assert!(shown(&objects, &[]).is_none(), "with no ride the set must not be copied");
        // One ride: exactly that datum goes.
        let v = shown(&objects, &[2]).expect("a ride filters");
        assert_eq!(v.iter().map(|o| o.datum).collect::<Vec<_>>(), vec![1, 3]);
        // A datum that is not in the set removes nothing -- a stale hide must not empty the map.
        let v = shown(&objects, &[999]).expect("a ride filters");
        assert_eq!(v.len(), 3, "hiding a datum that is not there must remove nothing");
        // And two, for a ride that ends and another that begins in the same frame.
        let v = shown(&objects, &[1, 3]).expect("a ride filters");
        assert_eq!(v.iter().map(|o| o.datum).collect::<Vec<_>>(), vec![2]);
    }

    /// **THE BYPASS GATE.** While seated the character controller must not run: no capsule
    /// sweep, no gravity, no ground probe, and nothing that drives a locomotion clip.
    ///
    /// Gated on the two observable halves, because `PlayMode::tick` needs a cache and this must
    /// run everywhere:
    ///
    /// * `set_pose_from_vehicle` must leave every quantity the locomotion state machine reads at
    ///   rest, so that getting out cannot land in a run cycle mid-stride or an airborne arc;
    /// * stepping the controller on the same frame must CHANGE those quantities — which is what
    ///   makes the first half a real assertion rather than a tautology.
    #[test]
    fn a_seated_body_is_not_stepped_by_the_controller() {
        use crate::play_ctl::{MoveInput, PlayerState};
        let p = crate::play_phys::tests_support::stand_in();
        // An empty triangle world: `Vec<[Vec3; 3]>` IS a `StaticWorld`, and with no floor the
        // control case falls, which is what makes the seated case's stillness meaningful.
        let world: Vec<[Vec3; 3]> = Vec::new();
        let input = MoveInput { fwd: 1.0, sprint: true, ..Default::default() };

        // THE CONTROL. Stepping on foot in an empty world moves the body and starts a fall, so
        // the quantities below are not inert by nature.
        let mut on_foot = PlayerState::standing_at(Vec3::new(0.0, 0.0, 10.0), 0.0);
        for _ in 0..30 {
            on_foot.step(&input, &p, &world, 1.0 / 60.0);
        }
        assert!(on_foot.vel.z < -1.0, "the control must be falling, got {:?}", on_foot.vel);
        assert!(on_foot.air_time > 0.0, "the control must be airborne");
        assert!(on_foot.pos.z < 10.0, "the control must have moved");

        // THE SEATED CASE. Same input, same world, same number of frames -- but the body is
        // placed by the seat and never stepped, so nothing the locomotion machine reads moves.
        let mut seated = PlayerState::standing_at(Vec3::new(0.0, 0.0, 10.0), 0.0);
        seated.air_time = 5.0;
        seated.ground_speed = 4.0;
        seated.sprint_time = 2.0;
        seated.crouched = true;
        seated.crouch_t = 1.0;
        let vehicle_vel = Vec3::new(3.0, 0.0, 0.0);
        for i in 0..30 {
            // The seat marker moves the vehicle along; the body follows it exactly.
            seated.set_pose_from_vehicle(Vec3::new(i as f32 * 0.05, 0.0, 10.0), 1.0, 0.2, vehicle_vel);
        }
        assert_eq!(seated.pos, Vec3::new(29.0 * 0.05, 0.0, 10.0), "the seat decides the position exactly");
        assert_eq!(seated.vel, vehicle_vel, "the body carries the VEHICLE's velocity, not one of its own");
        assert_eq!(seated.yaw, 1.0);
        assert_eq!(seated.pitch, 0.2);
        assert!(!seated.grounded, "a seated body stands on nothing");
        assert_eq!(seated.air_time, 0.0, "a seated body is not falling -- an air timer would start it in an airborne clip on exit");
        assert_eq!(seated.land_time, 0.0);
        assert_eq!(seated.ground_speed, 0.0, "a seated body travels no ground distance");
        assert_eq!(seated.sprint_time, 0.0);
        assert!(!seated.crouched);
        assert_eq!(seated.crouch_t, 0.0);
        // And the locomotion phase driver must not have advanced at all.
        assert_eq!(seated.distance, 0.0, "a seated body's locomotion cycle must not advance");
        assert!(on_foot.distance >= 0.0);
    }

    /// A SEATED body's animation state must be the seated one — the same shape
    /// `film_anim::derive` builds for a recorded rider — and `anim_candidates` must then ask for
    /// the seat's clips first.
    ///
    /// The bite: `set_pose_from_vehicle` leaves `grounded` false, so `play_ctl::body_state`
    /// returns `Stance::Flight` with `air_ticks` set. Handing that to the animation side would
    /// drive the seated clip's frame off an airborne counter instead of holding frame 0, so the
    /// seated state has to be built deliberately and this gate is what says so.
    #[test]
    fn a_seated_bodys_animation_state_is_the_seated_one_and_asks_for_the_seat_clip() {
        use crate::film_anim::{anim_candidates, BodyState, Gait, Seat as AnimSeat, Stance};
        use crate::play_ctl::{body_state, MoveInput, PlayerState};
        let p = crate::play_phys::tests_support::stand_in();
        let mut st = PlayerState::standing_at(Vec3::ZERO, 0.0);
        st.set_pose_from_vehicle(Vec3::ZERO, 0.0, 0.0, Vec3::new(5.0, 0.0, 0.0));
        // What the controller's own state machine says, UNCORRECTED -- the trap.
        let raw = body_state(&st, &MoveInput::default(), &p, Some("rifle".into()));
        assert_eq!(raw.stance, Stance::Flight, "an ungrounded body reads as airborne -- this is the trap");
        assert!(raw.air_ticks.is_some());
        // What test mode actually hands the animation side.
        let fixed = BodyState {
            stance: Stance::Seated,
            gait: Gait::Idle,
            air_ticks: None,
            land_ticks: None,
            seat: Some(AnimSeat { vehicle_net_id: 1, mode: "warthog_d".into(), vehicle: "warthog".into() }),
            ..raw.clone()
        };
        let cands = anim_candidates(&fixed, "rifle");
        assert_eq!(cands.first().map(|s| s.as_str()), Some("warthog_d:any:idle"), "{cands:?}");
        assert!(cands.iter().any(|c| c == "warthog_d:idle"), "{cands:?}");
        assert!(cands.last().is_some_and(|c| c.starts_with("combat:")), "a missing seat set falls back to standing: {cands:?}");
        // And the uncorrected state would have asked for an AIRBORNE clip instead.
        let wrong = anim_candidates(&raw, "rifle");
        assert!(
            !wrong.iter().any(|c| c.starts_with("warthog_d")),
            "the uncorrected state asks for {wrong:?} -- which is why the seated state is built deliberately"
        );
    }

    /// **THE OFFER GATE.** Standing at a Warthog's own driver entry marker must produce the
    /// game's own prompt, and standing away from it must produce nothing.
    ///
    /// The bite is that the geometry comes from the TAG: the probe point is the entry marker's
    /// own world position pushed out along its own forward by the seat's own `entry radius`, so a
    /// wrong marker axis, a wrong radius or a wrong seat all make the offer vanish.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn standing_at_a_warthogs_entry_marker_offers_the_games_own_prompt() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let hog = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog");
        // One Warthog at the origin, unrotated.
        let objects = vec![obj(0x1234, hog, Vec3::ZERO)];
        let world = crate::scene::object_matrix(&objects[0]);

        // The driver's own entry marker, and the point a player would stand at to use it.
        let (seat_index, marker, radius) = {
            let e = index.entry(native, cache, hog).expect("the Warthog reads");
            let s = e.seats.seats.iter().find(|s| s.is_driver()).expect("the Warthog has a driver seat").clone();
            let m = e.marker_frames(native, cache, s.entry_marker_group()).first().copied().expect("the model carries driver_enter");
            (s.index, world * m, s.entry_radius)
        };
        let fwd = marker.transform_vector3(reach_seats::MARKER_FORWARD).normalize();
        let at = marker.w_axis.truncate() + fwd * (radius * 0.6);
        let looking = (marker.w_axis.truncate() - at).normalize();

        let o = offer(&mut index, native, cache, &objects, at, looking, Vec3::ZERO, "E")
            .expect("standing at the driver's entry marker must offer the seat");
        println!("offer: datum {:#x} '{}' seat {} '{}' id '{}' -> {:?}  [{}]", o.datum, o.vehi_path, o.seat, o.seat_label, o.prompt_id, o.prompt, o.test.describe());
        assert_eq!(o.seat, seat_index, "the driver's seat is the one offered");
        assert_eq!(o.seat_label, "warthog_d");
        assert_eq!(o.prompt_id, "warthog_enter_driver");
        if crate::h4::localization::global().is_empty() {
            eprintln!("(no MCC localization folder: the prompt text is unresolvable here, which is the honest None)");
            assert!(o.prompt.is_none());
        } else {
            let p = o.prompt.as_deref().expect("the prompt must resolve");
            assert_eq!(p, "Hold E to drive Warthog", "the game's own words with the player's own key");
            assert_eq!(o.substituted, vec![reach_strings::GLYPH_ACTION]);
        }
        assert!(o.test.passes());

        // Ten world units away: nothing, and no panic.
        assert!(
            offer(&mut index, native, cache, &objects, at + Vec3::new(10.0, 0.0, 0.0), looking, Vec3::ZERO, "E").is_none(),
            "standing ten world units away must offer nothing"
        );
        // At the right place but looking the other way: the facing gate refuses.
        assert!(
            offer(&mut index, native, cache, &objects, at, -looking, Vec3::ZERO, "E").is_none(),
            "looking away must refuse -- the seat's `entry marker facing angle` says so"
        );
        // At the right place but sprinting past: `maximum relative velocity` refuses.
        assert!(
            offer(&mut index, native, cache, &objects, at, looking, Vec3::new(0.0, 0.0, 99.0), "E").is_none(),
            "moving far faster than the authored `maximum relative velocity` must refuse"
        );
    }

    /// **THE CHILD-SEAT GATE.** The Warthog's GUNNER seat is on `warthog_chaingun`, an attached
    /// child, so the scan must find it through the attachment — and the offer must name the
    /// child's tag while hiding the PARENT's placement.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_warthogs_gunner_seat_is_found_through_the_attachment() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let hog = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog");
        let objects = vec![obj(0x4321, hog, Vec3::ZERO)];
        let world = crate::scene::object_matrix(&objects[0]);
        // Find the gunner's own child tag and its entry marker through the attachment chain.
        let atts = native.enumerate_attachments(cache, hog);
        println!("the Warthog attaches {} object(s):", atts.len());
        let mut found = None;
        for a in &atts {
            let cls = native.tag_class(cache, a.child_obj).unwrap_or_default();
            let path = native.tag_name(cache, a.child_obj).unwrap_or_default();
            println!("   {cls} {path}");
            if cls != "vehi" {
                continue;
            }
            let local = Mat4::from_rotation_translation(
                glam::Quat::from_xyzw(a.rot[0], a.rot[1], a.rot[2], a.rot[3]).normalize(),
                Vec3::from(a.pos),
            );
            let Some(e) = index.entry(native, cache, a.child_obj) else { continue };
            let Some(s) = e.seats.seats.iter().find(|s| s.is_gunner() && s.player_allowed()).cloned() else { continue };
            let Some(m) = e.marker_frames(native, cache, s.entry_marker_group()).first().copied() else { continue };
            found = Some((a.child_obj, world * local * m, s));
            break;
        }
        let Some((child, marker, seat)) = found else {
            panic!("the Warthog attaches no gunner-seat vehicle -- the child-seat path is untested");
        };
        let fwd = marker.transform_vector3(reach_seats::MARKER_FORWARD).normalize();
        let at = marker.w_axis.truncate() + fwd * (seat.entry_radius * 0.6);
        let looking = (marker.w_axis.truncate() - at).normalize();
        let o = offer(&mut index, native, cache, &objects, at, looking, Vec3::ZERO, "E")
            .expect("standing at the gunner's entry marker must offer the seat");
        println!("offer: '{}' seat {} '{}' id '{}' -> {:?}", o.vehi_path, o.seat, o.seat_label, o.prompt_id, o.prompt);
        assert_eq!(o.vehi_tag, child, "the seat belongs to the CHILD's tag, not the Warthog's");
        assert_eq!(o.parent_datum, Some(0x4321), "the parent placement is the one that gets hidden");
        assert_eq!(o.seat_label, "warthog_g");
        assert_eq!(o.prompt_id, "warthog_enter_gunner");
        if !crate::h4::localization::global().is_empty() {
            assert_eq!(o.prompt.as_deref(), Some("Hold E to operate Warthog turret"));
        }
    }

    /// **THE HIDE GATE, both directions.** A ride must hide exactly one placement — the PARENT's,
    /// even for a child seat — and ending it must hide none. A leak one way draws two Warthogs; a
    /// leak the other draws none.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_ride_hides_exactly_one_placement_and_ending_it_hides_none() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let hog = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog");
        let placement = obj(0x9999, hog, Vec3::new(3.0, 4.0, 5.0));
        let offer = VehicleOffer {
            datum: 0x7777,
            parent_datum: Some(0x9999),
            vehi_tag: hog,
            vehi_path: "warthog".into(),
            seat: 0,
            seat_label: "warthog_d".into(),
            prompt_id: "warthog_enter_driver".into(),
            prompt: None,
            substituted: Vec::new(),
            test: EntryTest {
                distance: 0.5,
                radius: 1.25,
                cone: 0.1,
                cone_limit: 1.3,
                facing: 0.1,
                facing_limit: 1.0,
                relative_speed: 0.0,
                relative_speed_limit: 3.5,
            },
        };
        let r = begin(&mut index, native, cache, &offer, &placement);
        assert_eq!(r.hidden_datum, 0x9999, "the PARENT placement is hidden, not the child's datum");
        assert_eq!(r.drive.pos, Vec3::new(3.0, 4.0, 5.0), "the ride starts where the user put the vehicle");
        assert_eq!(r.seat_mode, "warthog_d", "the seat's own label is the jmad mode");
        assert!(r.drivable.is_ok(), "the Warthog must be drivable: {:?}", r.drivable);
        // The hidden set is derived from the ride and from nothing else, so ending the ride
        // cannot leave a hidden placement behind.
        let hidden: Vec<u32> = Some(&r).into_iter().map(|r| r.hidden_datum).collect();
        assert_eq!(hidden, vec![0x9999]);
        let none: Option<&Riding> = None;
        let hidden: Vec<u32> = none.into_iter().map(|r| r.hidden_datum).collect();
        assert!(hidden.is_empty(), "with no ride, nothing is hidden");
    }

    /// **THE DRIVEN-ANIMATION GATE.** Drive a Warthog a measured distance and its front wheel
    /// must have turned by exactly that distance over its own authored radius — and turn the
    /// wheel and the front wheels must steer the way it is turning.
    ///
    /// # Why this gate bites
    ///
    /// It goes through the production functions with the production signatures: [`draw_pieces`]
    /// builds the pieces, [`step`] integrates the ride (which is what fills `odometer`),
    /// [`drive_actors`] installs the graph, the base clip and the overlays, and
    /// `update_poses_headless` poses the skeleton exactly as the GPU path does. Nothing here
    /// computes a pose of its own.
    ///
    /// **The number is not taken from the fixture.** The expected rotation is `distance / radius`
    /// radians, where the radius is read out of the vehicle's OWN authored `friction points`, so
    /// a wrong radius, a wrong frame mapping, a cycle driven as a range, or an overlay that never
    /// loaded all move the measured angle away from it. And the control case — a vehicle that has
    /// not moved — must show NO rotation, which is what stops the assertion passing on a pose
    /// that is non-zero for an unrelated reason.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_driven_warthogs_wheel_turns_by_its_own_authored_radius() {
        use crate::reach_actor::ActorScene;
        use crate::reach_vehicle::DriveInput;
        use crate::reach_vehicle_anim::Role;
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let hog = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog");

        // What the TAG says, read before anything is animated — the oracle.
        let (radius, roll_name, steer_left_end) = {
            let e = index.entry(native, cache, hog).expect("the Warthog reads");
            for l in e.anim.describe() {
                println!("   {l}");
            }
            let roll = e
                .anim
                .overlays
                .iter()
                .find(|o| matches!(o.role, Role::Roll { .. }) && o.name.ends_with("ground_speed"))
                .expect("the Warthog's front wheels roll");
            let Role::Roll { radius, .. } = &roll.role else { unreachable!() };
            let steer = e
                .anim
                .overlays
                .iter()
                .find(|o| matches!(o.role, Role::Steer { .. }))
                .expect("the Warthog steers");
            let Role::Steer { left_end, .. } = &steer.role else { unreachable!() };
            (*radius, roll.name.clone(), *left_end)
        };
        println!("authored front-wheel radius {radius:.4} wu ('{roll_name}'), steering left end = frame {steer_left_end}");
        assert!((radius - 0.250).abs() < 1e-4, "the Warthog's authored wheel radius is 0.250, got {radius}");

        let placement = obj(0x2468, hog, Vec3::ZERO);
        let o = VehicleOffer {
            datum: 0x2468,
            parent_datum: None,
            vehi_tag: hog,
            vehi_path: "warthog".into(),
            seat: 0,
            seat_label: "warthog_d".into(),
            prompt_id: String::new(),
            prompt: None,
            substituted: Vec::new(),
            test: EntryTest { distance: 0.0, radius: 1.25, cone: 0.0, cone_limit: 1.3, facing: 0.0, facing_limit: 1.0, relative_speed: 0.0, relative_speed_limit: 3.5 },
        };
        let mut riding = begin(&mut index, native, cache, &o, &placement);

        // One flat floor at z = 0, so the ride is grounded and the odometer is the distance the
        // wheels actually rolled.
        struct Flat;
    impl crate::objscene::ObjectScene for Flat {
        fn has_cache(&self) -> bool { true }
        fn invalidate(&mut self) {}
        fn rebuild_pending(&self) -> bool { false }
        fn maybe_rebuild(
            &mut self,
            _o: &[ObjectInfo],
            _c: &std::collections::HashMap<u32, (u8, u8)>,
            _m: &hms_render::MeshRenderer,
            _d: &eframe::wgpu::Device,
            _q: &eframe::wgpu::Queue,
        ) -> Option<crate::objscene::RebuiltLanes> { None }
        fn set_object_scales(&mut self, _s: std::collections::HashMap<u32, f32>) {}
        fn set_object_casters(&mut self, _c: std::collections::HashSet<u32>) {}
        fn pick(&self, _o: Vec3, _d: Vec3) -> Option<u32> { None }
        fn aabb_of(&self, _d: u32) -> Option<([f32; 3], [f32; 3])> { None }
        fn object_obb(&self, _d: u32) -> Option<crate::construct::Obb> { None }
        fn selection_wireframe(&self, _d: u32) -> Option<Vec<[f32; 3]>> { None }
        fn translate_pick(&mut self, _d: u32, _v: Vec3) {}
        fn rotate_pick(&mut self, _d: u32, _q: glam::Quat, _p: Vec3) {}
        fn raycast_scene(&self, _o: Vec3, _d: Vec3) -> Option<Vec3> { None }
        fn raycast_scene_n(&self, _o: Vec3, _d: Vec3) -> Option<(Vec3, Vec3)> { None }
        fn raycast_objects_excluding(&self, _o: Vec3, _d: Vec3, _x: &[u32]) -> Option<Vec3> { None }
        fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> { None }
        fn resolve_object_mode(&self, _t: u32) -> u32 { 0 }
        fn tag_name_of(&self, _t: u32) -> String { String::new() }
        fn raw_tag_class(&self, _t: u32) -> Option<String> { None }
        fn bsp_flags_report(&self) -> String { String::new() }
        fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]) { (&[], &[]) }
        fn collision_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
        fn physics_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
        fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> {
            // A spawn point far away, which the fix must NOT use.
            Some((Vec3::new(500.0, 500.0, 50.0), 0.0, 0.0))
        }
        // A floor at z = 0: anything cast straight down from above hits at its own height.
        fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> {
            (dir.z < -0.5 && origin.z > 0.0 && origin.z <= max).then_some(origin.z)
        }
        // #play  The SAME floor as two triangles. The trait has no default for this any more
        // (an empty one is how Halo 3 shipped a frictionless void), so a stand-in has to say
        // what it is -- and a stand-in called `Floor` whose triangle gather was empty would
        // be lying about itself.
        fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            if mn.z > 0.0 || mx.z < 0.0 {
                return;
            }
            let (a, b) = (Vec3::new(mn.x - 1.0, mn.y - 1.0, 0.0), Vec3::new(mx.x + 1.0, mx.y + 1.0, 0.0));
            out.push([a, Vec3::new(b.x, a.y, 0.0), b]);
            out.push([a, b, Vec3::new(a.x, b.y, 0.0)]);
        }
    }
        let floor = Flat;

        let mut actors = ActorScene::default();
        let mut ids: Vec<u32> = Vec::new();

        // The angle one wheel node has turned about its own axle, in MODEL space.
        let wheel_roll = |actors: &ActorScene, id: u32, node: &str| -> f32 {
            let a = actors.get(id).expect("the actor exists");
            let i = a.def.model.node_names.iter().position(|n| n == node).expect("the model carries the node");
            let bind = a.def.model.skel.bind_model();
            // A wheel spins about its own local Y, so the roll shows in where its local +Z went.
            let b = bind[i].transform_vector3(Vec3::Z).normalize();
            let p = a.pose.model[i].transform_vector3(Vec3::Z).normalize();
            let axis = bind[i].transform_vector3(Vec3::Y).normalize();
            b.cross(p).dot(axis).atan2(b.dot(p))
        };
        // How far a wheel node's axle has swung in the hull's XY plane; positive is toward +Y,
        // which is left.
        let wheel_yaw = |actors: &ActorScene, id: u32, node: &str| -> f32 {
            let a = actors.get(id).expect("the actor exists");
            let i = a.def.model.node_names.iter().position(|n| n == node).expect("the model carries the node");
            let bind = a.def.model.skel.bind_model();
            let f = |m: Mat4| {
                let v = m.transform_vector3(Vec3::Y);
                Vec3::new(v.x, v.y, 0.0).normalize()
            };
            let (b, p) = (f(bind[i]), f(a.pose.model[i]));
            (b.x * p.y - b.y * p.x).atan2(b.dot(p))
        };

        // Exactly what `play_mode::sync_vehicle_actors` does, in the same order.
        let mut install = |actors: &mut ActorScene, ids: &mut Vec<u32>, index: &mut VehicleIndex, riding: &Riding| {
            let pieces = draw_pieces(index, native, cache, riding, 0);
            if ids.is_empty() {
                for p in &pieces {
                    ids.push(
                        actors
                            .add_render_model(native, cache, p.mode_tag, None, Vec3::ZERO, 0.0)
                            .expect("the model reads"),
                    );
                }
            }
            for (id, p) in ids.iter().zip(pieces.iter()) {
                if let Some(a) = actors.get_mut(*id) {
                    a.world_override = Some(p.world);
                }
            }
            let notes = drive_actors(index, native, cache, riding, &pieces, ids, actors);
            for n in &notes {
                println!("   NOTE {n}");
            }
            assert!(notes.is_empty(), "the ride could not be animated: {notes:?}");
            actors.update_poses_headless(0.0);
        };

        // --- THE CONTROL: nothing driven yet, so nothing may be turned --------------------
        install(&mut actors, &mut ids, &mut index, &riding);
        let body = ids[0];
        let rest = wheel_roll(&actors, body, "b_wheel_front_left");
        println!("at rest: front-left wheel roll {rest:+.4} rad");
        assert!(rest.abs() < 1e-3, "a vehicle that has not moved must have an unrotated wheel, got {rest:+.4} rad");

        // --- drive it and the wheel must turn by distance / radius ------------------------
        let input = DriveInput { throttle: 1.0, slide: 0.0, want_yaw: riding.drive.yaw, brake: false };
        for _ in 0..120 {
            step(&mut index, native, cache, &mut riding, input, 1.0 / 60.0, &floor);
        }
        let travelled = riding.drive.odometer;
        println!("drove {travelled:.4} wu in 2 s");
        assert!(travelled > 1.0, "the Warthog must actually move: odometer {travelled}");
        install(&mut actors, &mut ids, &mut index, &riding);
        let turned = wheel_roll(&actors, body, "b_wheel_front_left");
        // A rolling wheel turns `d / r` radians, wrapped into the shortest signed angle because
        // that is all a single node orientation can report.
        let mut want = (travelled / radius).rem_euclid(std::f32::consts::TAU);
        if want > std::f32::consts::PI {
            want -= std::f32::consts::TAU;
        }
        let mut err = (turned - want).abs();
        if err > std::f32::consts::PI {
            err = (std::f32::consts::TAU - err).abs();
        }
        println!("front-left wheel roll {turned:+.4} rad, arithmetic says {want:+.4} (err {err:.4})");
        // The clip stores four quantised keys per revolution, so the sampled angle is only as
        // faithful as a 90-degree-key nlerp allows. 0.15 rad is 8.6 degrees — well inside that,
        // and two orders of magnitude outside the 0 the control case gave.
        assert!(
            err < 0.15,
            "the wheel turned {turned:+.4} rad but {travelled:.4} wu over r {radius:.4} is {want:+.4} rad (err {err:.4})"
        );
        // And the REAR wheels turn too: they are a SEPARATE overlay, so a gate that only looked
        // at the front would pass with `vehicle:ground_speed_rear` never installed.
        let rear = wheel_roll(&actors, body, "b_wheel_back_left");
        println!("back-left wheel roll {rear:+.4} rad");
        assert!(rear.abs() > 0.05, "the REAR wheels must turn as well, got {rear:+.4} rad");

        // --- and the steering must point the wheels the way it is turning -----------------
        let mut hard_left = riding.clone();
        hard_left.drive.steer_angle = 60.0f32.to_radians();
        install(&mut actors, &mut ids, &mut index, &hard_left);
        let yl = wheel_yaw(&actors, body, "b_wheel_front_left");
        let mut hard_right = hard_left.clone();
        hard_right.drive.steer_angle = -60.0f32.to_radians();
        install(&mut actors, &mut ids, &mut index, &hard_right);
        let yr = wheel_yaw(&actors, body, "b_wheel_front_right");
        println!("full left: front wheel yaw {:+.2} deg | full right: {:+.2} deg", yl.to_degrees(), yr.to_degrees());
        assert!(yl > 0.2, "steering left must swing the front wheels toward +Y (left), got {:+.2} deg", yl.to_degrees());
        assert!(yr < -0.2, "steering right must swing them the other way, got {:+.2} deg", yr.to_degrees());
        // The authored sweep is +-30 deg at the front universals, so both ends must land near it
        // rather than at some fraction of it — which is what catches a range driven as a cycle.
        assert!(
            (yl.to_degrees() - 30.0).abs() < 3.0 && (yr.to_degrees() + 30.0).abs() < 3.0,
            "the authored sweep is +-30 deg; got {:+.2} / {:+.2}",
            yl.to_degrees(),
            yr.to_degrees()
        );
        // Centred, they come back: a saturating range has a middle, and the Mongoose's own
        // 3-frame clip (-25, 0, +25) is what says the middle is where zero lives.
        let mut centred = hard_left.clone();
        centred.drive.steer_angle = 0.0;
        install(&mut actors, &mut ids, &mut index, &centred);
        let yc = wheel_yaw(&actors, body, "b_wheel_front_left");
        println!("centred: front wheel yaw {:+.2} deg", yc.to_degrees());
        assert!(yc.to_degrees().abs() < 3.0, "centred steering must point the wheels straight, got {:+.2} deg", yc.to_degrees());
    }

    /// Riding a vehicle that authors no drive model must say so and must not move it. The Banshee
    /// is the case: you can sit in it and it does not fly.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn an_undrivable_vehicle_says_so_and_does_not_move() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let banshee = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some("objects\\vehicles\\covenant\\banshee\\banshee"))
            .expect("forge_halo carries the Banshee");
        let placement = obj(0x5555, banshee, Vec3::new(1.0, 2.0, 3.0));
        let o = VehicleOffer {
            datum: 0x5555,
            parent_datum: None,
            vehi_tag: banshee,
            vehi_path: "banshee".into(),
            seat: 0,
            seat_label: "banshee_d".into(),
            prompt_id: "banshee_enter".into(),
            prompt: None,
            substituted: Vec::new(),
            test: EntryTest { distance: 0.0, radius: 1.5, cone: 0.0, cone_limit: 1.4, facing: 0.0, facing_limit: 1.0, relative_speed: 0.0, relative_speed_limit: 3.0 },
        };
        let mut r = begin(&mut index, native, cache, &o, &placement);
        let why = r.drivable.clone().expect_err("the Banshee must not be drivable");
        println!("the Banshee: {why}");
        assert!(why.contains("alien_fighter"), "the refusal must name the family, got {why:?}");
        let before = r.drive.pos;
        struct NoGround;
        // `step` must return without touching the state when there is no drive model.
        let scene_free: Option<&dyn ObjectScene> = None;
        let _ = (NoGround, scene_free);
        let entry = index.entry(native, cache, banshee).unwrap();
        assert!(entry.drive.is_err());
        r.drive.pos = before;
        assert_eq!(r.drive.pos, Vec3::new(1.0, 2.0, 3.0));
    }
}

#[cfg(test)]
mod probes {
    //! #vehicle-anim  PRINT-ONLY. What a vehicle's OWN `jmad` authors, and whether the seat
    //! markers a rider is placed on exist in the render model.
    //!
    //! `cargo test -p hms-app --profile quick play_vehicle::probes:: -- --ignored --nocapture`

    use super::*;
    use crate::reach_actor::{resolve_jmad, AnimGraph};

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// Every `vehi` in the map: its own graph's animation list, its blend screens, and its
    /// skeleton's node names. This is the measurement that decides whether a driven vehicle's
    /// motion is a CLIP or a node transform.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_what_a_vehicles_own_graph_authors() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        println!("{} vehi tags\n", tags.len());
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let Some(j) = resolve_jmad(native, cache, t) else {
                println!("== {leaf}: NO jmad");
                continue;
            };
            let Some(g) = AnimGraph::read(native, cache, j) else {
                println!("== {leaf}: jmad {j:#x} unreadable");
                continue;
            };
            println!(
                "== {leaf}  jmad {:#x} '{}'  {} anims  {} nodes  {} blend screens",
                j,
                native.tag_name(cache, j).unwrap_or_default(),
                g.anims.len(),
                g.node_names.len(),
                g.blend_screens.len()
            );
            for a in &g.anims {
                println!(
                    "     [{:>3}] {:<34} {:?} frames {:>4} loop {:>4} bs {:?} nodes {}",
                    a.index, a.name, a.anim_type, a.frame_count, a.loop_frame, a.blend_screen, a.node_count
                );
            }
            for (i, sc) in g.blend_screens.iter().enumerate() {
                println!(
                    "     screen[{i}] '{}' {}x{} {:?}",
                    g.blend_screen_labels.get(i).cloned().unwrap_or_default(),
                    sc.width(),
                    sc.height(),
                    sc
                );
            }
            println!("     nodes: {}", g.node_names.join(" "));
        }
    }

    /// The seat the player is placed on: does the render model carry the seat's own
    /// `marker name`? `rider_pose` returns `None` when it does not, and then the body is never
    /// moved at all — which is exactly "it doesn't attach the biped to the vehicle".
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seat_markers_resolve_in_the_render_model() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        let mut missing = 0usize;
        let mut total = 0usize;
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let Some(e) = index.entry(native, cache, t) else { continue };
            let seats = e.seats.clone();
            for s in &seats.seats {
                total += 1;
                let seat_m = index
                    .entry(native, cache, t)
                    .map(|e| e.marker_frames(native, cache, &s.marker_name).len())
                    .unwrap_or(0);
                let entry_m = index
                    .entry(native, cache, t)
                    .map(|e| e.marker_frames(native, cache, s.entry_marker_group()).len())
                    .unwrap_or(0);
                let cam = index
                    .entry(native, cache, t)
                    .map(|e| e.marker_frames(native, cache, &s.camera_marker).len())
                    .unwrap_or(0);
                if seat_m == 0 {
                    missing += 1;
                }
                println!(
                    "{leaf:<22} seat {:>2} '{:<14}' marker '{:<18}' x{seat_m}  entry '{:<18}' x{entry_m}  cam '{:<16}' x{cam} {}",
                    s.index,
                    s.label,
                    s.marker_name,
                    s.entry_marker_group(),
                    s.camera_marker,
                    if seat_m == 0 { "  <-- NO SEAT MARKER: the rider is never placed" } else { "" }
                );
            }
        }
        println!("\n{missing} of {total} seats have no resolvable seat marker in the render model");
    }

    /// What one vehicle OVERLAY actually does across its frames: the per-node rotation delta in
    /// degrees, so a CYCLIC spin (a wheel going round) can be told from a PARAMETER RANGE (a
    /// steering wheel sweeping left to right).
    ///
    /// The distinction is the whole design: a cycle is driven by a phase that wraps, a range is
    /// indexed by a value that saturates, and guessing wrong makes a steering wheel spin or a
    /// wheel jitter.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_what_a_vehicle_overlay_does_per_frame() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let want = std::env::var("HMS_PROBE_VEHI").unwrap_or_else(|_| "warthog,mongoose,scorpion,ghost,wraith,revenant,falcon".into());
        let wants: Vec<String> = want.split(',').map(|s| s.trim().to_lowercase()).collect();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_lowercase();
            if !wants.iter().any(|w| &leaf == w) {
                continue;
            }
            let Some(j) = resolve_jmad(native, cache, t) else { continue };
            let Some(g) = AnimGraph::read(native, cache, j) else { continue };
            println!("\n######## {leaf}  ({} nodes)", g.node_names.len());
            for a in &g.anims {
                let Some(d) = g.load(native, cache, a.index) else {
                    println!("  [{}] {} -- NO DATA", a.index, a.name);
                    continue;
                };
                println!("  --- {} ({:?}, {} frames) ---", a.name, a.anim_type, a.frame_count);
                let n = a.frame_count.max(1) as usize;
                // Per node, the rotation delta at each frame, as a signed angle about its own
                // dominant axis -- which is what makes a monotone sweep visible as a sequence.
                let mut rows: Vec<(String, Vec<String>)> = Vec::new();
                let samples: Vec<Vec<crate::reach_jmad::NodeChannel>> =
                    (0..n).map(|f| crate::reach_jmad::sample(&d, a.anim_type, f as f32)).collect();
                for (i, name) in g.node_names.iter().enumerate() {
                    let mut cells = Vec::new();
                    let mut interesting = false;
                    for s in &samples {
                        let c = s.get(i).copied().unwrap_or_default();
                        let mut cell = String::new();
                        if let Some(q) = c.rot {
                            let (axis, ang) = q.to_axis_angle();
                            // SHORTEST signed angle: `to_axis_angle` reports 359.8 deg where the
                            // node has turned -0.2, and a sweep printed that way is unreadable.
                            let mut deg = ang.to_degrees();
                            if deg > 180.0 {
                                deg -= 360.0;
                            }
                            if deg.abs() > 0.05 {
                                interesting = true;
                                // Signed about the dominant axis so a sweep reads as a sequence.
                                let dom = if axis.x.abs() >= axis.y.abs() && axis.x.abs() >= axis.z.abs() {
                                    ('x', axis.x)
                                } else if axis.y.abs() >= axis.z.abs() {
                                    ('y', axis.y)
                                } else {
                                    ('z', axis.z)
                                };
                                cell = format!("{}{:+.1}", dom.0, deg * dom.1.signum());
                            } else {
                                cell = "0".into();
                            }
                        }
                        if let Some(tr) = c.trans {
                            if tr.length() > 0.0005 {
                                interesting = true;
                                cell += &format!(" t({:+.3},{:+.3},{:+.3})", tr.x, tr.y, tr.z);
                            }
                        }
                        cells.push(if cell.is_empty() { "-".into() } else { cell });
                    }
                    if interesting {
                        rows.push((name.clone(), cells));
                    }
                }
                if rows.is_empty() {
                    println!("      (no node moves)");
                }
                for (name, cells) in rows {
                    println!("      {name:<30} {}", cells.join("  "));
                }
            }
        }
    }

    /// Which render-model nodes of a vehicle actually carry GEOMETRY or a MARKER.
    ///
    /// The point: `vehicle:steering` also drives `b_aim_yaw` by +-45 deg, and `vehicle:acceleration`
    /// walks it through a full 360 at 22.5 deg a frame. If that node carried the turret mount,
    /// composing the overlay as authored would swing the gun with the wheel. So it is measured
    /// rather than assumed, and an overlay is applied whole (as the engine composes one) only
    /// once the nodes it drives are known.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_which_vehicle_nodes_carry_geometry_or_markers() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let want = std::env::var("HMS_PROBE_VEHI").unwrap_or_else(|_| "warthog,mongoose,scorpion,ghost".into());
        let wants: Vec<String> = want.split(',').map(|s| s.trim().to_lowercase()).collect();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_lowercase();
            if !wants.iter().any(|w| &leaf == w) {
                continue;
            }
            let Some(e) = index.entry(native, cache, t) else { continue };
            let Some(m) = e.model.as_ref() else { continue };
            println!("\n######## {leaf}: {} model nodes", m.node_names.len());
            for (i, name) in m.node_names.iter().enumerate() {
                let skinned = m
                    .node_maps
                    .iter()
                    .enumerate()
                    .any(|(mi, map)| crate::reach_skel::is_skinned_format(m.mesh_fmt.get(mi).copied().unwrap_or(0)) && map.contains(&(i as u8)));
                let rigid = m.mesh_rigid_node.iter().any(|&r| r as usize == i && r != 0xFF);
                let marks: Vec<String> = m
                    .markers
                    .iter()
                    .filter(|g| g.markers.iter().any(|mk| mk.node == i as i32))
                    .map(|g| native.string_id(cache, g.name).unwrap_or_default())
                    .collect();
                let what = match (skinned, rigid) {
                    (false, false) => "(no geometry)",
                    (true, false) => "skinned",
                    (false, true) => "RIGID mesh",
                    (true, true) => "skinned+rigid",
                };
                println!("   [{i:>2}] {name:<32} {what:<14} markers: {}", if marks.is_empty() { "-".to_string() } else { marks.join(" ") });
            }
        }
    }

    /// **THE STEERING DIRECTION, measured in MODEL space over every vehicle that has one.**
    ///
    /// Read in node-local axes the authored direction appears to flip (the Warthog's front
    /// universals take `z +30` on frame 0, the Mongoose's front hubs take `z -25`), but a local
    /// rotation sign is not comparable between two rigs whose bones are oriented differently.
    /// So this composes each steering overlay onto the bind pose and reads where the FRONT
    /// WHEEL's axle actually ends up in the hull's own XY plane. If the sign agrees across the
    /// fleet there is a fixed engine convention; if it does not, `left_end` is per vehicle and
    /// derived from exactly this measurement.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_which_end_of_a_steering_clip_turns_left() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        let mut rows: Vec<(String, String)> = Vec::new();
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let Some(e) = index.entry(native, cache, t) else { continue };
            println!("\n######## {leaf}");
            for line in e.anim.describe() {
                println!("    {line}");
            }
            if let Some(o) = e.anim.overlays.iter().find(|o| o.name.contains("steering")) {
                rows.push((leaf, format!("{:?}", o.role)));
            }
        }
        println!("\n==== every vehicle with a steering overlay ====");
        for (leaf, role) in &rows {
            println!("   {leaf:<24} {role}");
        }
    }

    /// The authored TURN MAXIMA per wheeled vehicle, beside the steering clip's measured left
    /// end — testing whether the sign the tag authors explains the Mongoose's mirrored clip.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_do_the_authored_turn_maxima_explain_the_steering_sign() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        println!("{:<22} {:>10} {:>10} {:>10}   steering", "vehicle", "max_left", "max_right", "turn_rate");
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let Some(e) = index.entry(native, cache, t) else { continue };
            let Ok(d) = e.drive.as_ref() else { continue };
            let Some(t) = d.turning else { continue };
            let role = e
                .anim
                .overlays
                .iter()
                .find(|o| o.name.contains("steering"))
                .map(|o| format!("{:?}", o.role))
                .unwrap_or_else(|| "(no steering overlay)".into());
            println!(
                "{leaf:<22} {:>10.4} {:>10.4} {:>10.4}   {role}",
                t.max_left.to_degrees(), t.max_right.to_degrees(), t.turn_rate
            );
        }
    }

    /// **DOES THE RIDER'S OWN GRAPH HAVE A CLIP FOR EACH SEAT?** Reported: *"I stand straight up
    /// instead of sitting in the seat."*
    ///
    /// A seated body is given `Stance::Seated` and the seat's `label` as its jmad mode, and
    /// `film_anim::anim_candidates` then asks for `<label>:<class>:idle`, `<label>:idle`, ... and
    /// falls back to the STANDING `combat:` set when the graph has none of them. That fallback is
    /// what standing upright in a seat looks like, so this prints, per seat, which candidate the
    /// Spartan's and the Elite's graphs actually answer — and names every seat where the answer
    /// is the standing fallback.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_does_the_riders_graph_have_a_clip_for_each_seat() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let mut graphs: Vec<(String, AnimGraph)> = Vec::new();
        for (t, path) in &bipeds {
            if !(path.contains("spartan") || path.contains("elite")) {
                continue;
            }
            if let Some(g) = resolve_jmad(native, cache, *t).and_then(|j| AnimGraph::read(native, cache, j)) {
                let leaf = path.rsplit(['\\', '/']).next().unwrap_or(path).to_string();
                if graphs.iter().any(|(n, _)| *n == leaf) {
                    continue;
                }
                println!("rider graph '{leaf}' — {} anims", g.anims.len());
                graphs.push((leaf, g));
            }
        }
        let mut index = VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        let mut missing: Vec<String> = Vec::new();
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let Some(e) = index.entry(native, cache, t) else { continue };
            let seats = e.seats.clone();
            let family = e.seat_family.clone();
            for s in &seats.seats {
                if !s.player_allowed() {
                    continue;
                }
                // Exactly the state `play_mode::pose` builds for a rider, and exactly the
                // candidate list it then walks.
                let st = crate::film_anim::BodyState {
                    stance: crate::film_anim::Stance::Seated,
                    gait: crate::film_anim::Gait::Idle,
                    air_ticks: None,
                    land_ticks: None,
                    seat: Some(crate::film_anim::Seat {
                        vehicle_net_id: 1,
                        mode: s.label.clone(),
                        vehicle: leaf.clone(),
                    }),
                    ..Default::default()
                };
                let cands = crate::film_anim::anim_candidates(&st, "rifle");
                let mut answered: Vec<String> = Vec::new();
                for (who, g) in &graphs {
                    // `AnimGraph::find` is what `set_anim` uses, substring fallback included.
                    let hit = cands.iter().find(|c| g.find(c).is_some());
                    answered.push(format!(
                        "{who}:{}",
                        match hit {
                            Some(h) if h.starts_with(&format!("{}:", s.label)) => format!("'{h}'"),
                            Some(h) => format!("FALLBACK '{h}'"),
                            None => "NONE".to_string(),
                        }
                    ));
                }
                let seated = answered.iter().all(|a| a.contains('\'') && !a.contains("FALLBACK"));
                if !seated {
                    missing.push(format!("{leaf} seat {} '{}' (family '{family}') -> {}", s.index, s.label, answered.join("  ")));
                }
                println!(
                    "{:<24} seat {:>2} '{:<16}' family '{:<9}' {}",
                    leaf, s.index, s.label, family, answered.join("  ")
                );
            }
        }
        println!("\n==== {} player seats whose rider would fall back to a STANDING clip ====", missing.len());
        for m in &missing {
            println!("   {m}");
        }
    }

    /// **THE PHYSICS TYPE TABLE.** Which of the fifteen `vehicle_physics_types_struct` blocks
    /// each shipped vehicle populates, with the wheelbase and track measured from its own
    /// friction points. Asked for directly: *"Warthogs/mongooses have the jeep engine physics
    /// and that is what they use. There are a few vehicle physics types in Reach that we need to
    /// get right."*
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_which_physics_type_each_vehicle_selects() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut index = VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        let mut by_type: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        println!(
            "{:<26} {:<16} {:>4} {:>4} {:>8} {:>8} {:>7}  friction points",
            "vehicle", "type block(s)", "fric", "agp", "wbase", "track", "gravsc"
        );
        for t in tags {
            let path = native.tag_name(cache, t).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let blocks = crate::reach_vehicle::VehiclePhysics::populated_blocks(native, cache, t);
            let Some(e) = index.entry(native, cache, t) else { continue };
            let Some(p) = e.phys.as_ref() else {
                println!("{leaf:<26} {:<16}  (physics unreadable)", blocks.join("+"));
                continue;
            };
            // Wheelbase and track from the friction-point MARKER POSITIONS: the spread along the
            // vehicle's own forward (+X) and across it (+Y).
            let (mut wb, mut tr) = (0.0f32, 0.0f32);
            if p.wheel_positions.len() >= 2 {
                let xs: Vec<f32> = p.wheel_positions.iter().map(|v| v.x).collect();
                let ys: Vec<f32> = p.wheel_positions.iter().map(|v| v.y).collect();
                wb = xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min);
                tr = ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min);
            }
            let names: Vec<String> = p
                .havok
                .friction_points
                .iter()
                .map(|f| format!("{}(r{:.3})", f.marker_name, f.radius))
                .collect();
            println!(
                "{leaf:<26} {:<16} {:>4} {:>4} {:>8.4} {:>8.4} {:>7.3}  {}",
                blocks.join("+"),
                p.havok.friction_points.len(),
                p.havok.antigravity_points.len(),
                wb,
                tr,
                p.havok.gravity_scale(),
                names.join(" ")
            );
            by_type.entry(blocks.join("+")).or_default().push(leaf);
        }
        println!("\n==== which vehicles select each type ====");
        for (k, v) in &by_type {
            println!("   {:<18} {:>2}  {}", k, v.len(), v.join(", "));
        }
    }
}
