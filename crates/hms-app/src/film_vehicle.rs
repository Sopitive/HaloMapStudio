//! #vehicle-rot #vehicle-damage  What a saved film says about a VEHICLE beyond where it is: how
//! it is turned, how damaged it is, and when the engine collected the wreck.
//!
//! Three separate questions, three separate answers, each with its own measurement:
//!
//! 1. **Rotation is in the recording.** A vehicle's base-class dirty block carries field 2 — a
//!    19-bit up axis plus an 8-bit angle about it, which the engine's own decode turns into an
//!    orthonormal `(forward, up)` pair. That is a FULL BASIS: pitch and roll included. Theater
//!    used to collapse it to `atan2(fwd.y, fwd.x)` and place the vehicle level, so the
//!    man-cannon tumble every Castle Wars clip is full of was thrown away one line after it was
//!    decoded. [`vehicle_basis`] builds the engine's matrix from the pair instead.
//!
//! 2. **The damage STATE is in the recording**, through the event channel rather than the entity
//!    stream. `damage_section_response` (list B kind 1, event type **1**) names the object and
//!    the `(damage section, instant response)` pair that fired; the object's own `hlmt` says what
//!    that response does to the model — its `region_transitions` block carries `(region,
//!    new state)` with `new state` the five-value `model_state_enum` HMS's DAMAGE dropdown
//!    already draws. So the chain recording -> authored tag -> drawn permutation is the engine's,
//!    not an invention. See [`DamageResponses`].
//!
//! 3. **The despawn is in the recording** — it always was. A kind-4 destroy ends the track
//!    (`EntityTrack::destroyed_tick`), and that is the moment the engine collected the object,
//!    exactly as Theater already treats a body. What was missing is that between the destroying
//!    response and that despawn the wreck must be drawn DESTROYED, and must be drawn by the
//!    stream even while it is sitting on the Forge spawn point whose pristine copy the scene
//!    draws. [`WreckRule`] is that pair of conditions in one place.

use glam::{Mat4, Vec3};

/// Reach's `model_state_enum`: 0 default, 1 minor, 2 medium, 3 major, 4 destroyed.
pub const STATE_DESTROYED: u8 = 4;
/// How many values that enum has.
pub const STATES: usize = 5;

// ---------------------------------------------------------------------------------------------
// 1. rotation
// ---------------------------------------------------------------------------------------------

/// #vehicle-rot  A vehicle's placement matrix from the recorded `(forward, up)` pair.
///
/// `col0 = forward`, `col1 = cross(up, forward)`, `col2 = up` — the engine's
/// `Matrix_BuildOrientation`, which is the same rule every Forge object is placed with
/// (`scene::object_matrix`). `None` for a vehicle the recording has not oriented yet, in which
/// case the caller keeps its old yaw-only placement rather than inventing a level one.
pub fn vehicle_basis(pos: Vec3, orient: Option<([f32; 3], [f32; 3])>) -> Option<Mat4> {
    let (f, u) = orient?;
    let m = crate::film_decode::basis_mat3(f, u);
    Some(Mat4::from_translation(pos) * Mat4::from_mat3(m))
}

/// #vehicle-rot  How far off level a recorded basis is, degrees: the angle between its up axis
/// and world +Z. 0 = the vehicle is drawn flat, 180 = upside down. This is the quantity the old
/// yaw-only placement discarded, and the number the rotation test asserts on.
pub fn tilt_degrees(up: [f32; 3]) -> f32 {
    let u = Vec3::from(up);
    if u.length_squared() < 1e-9 {
        return 0.0;
    }
    u.normalize().dot(Vec3::Z).clamp(-1.0, 1.0).acos().to_degrees()
}

// ---------------------------------------------------------------------------------------------
// 2. the damage states a model's responses can produce
// ---------------------------------------------------------------------------------------------

/// #vehicle-damage  One object model's damage-response table, reduced to the only thing a
/// renderer can show: per `(damage section, instant response)`, the worst `model_state_enum`
/// value that response's `region_transitions` name.
///
/// ## The tag layout this reads, and where every offset comes from
///
/// `tagdefs hlmt` (`reach_tag_test.exe` reflection data) gives:
///
/// ```text
/// hlmt +0x0CC  struct   damage info            (s_new_model_damage_info, 60 B)
///        +0x14 block    damage sections        -> new_global_damage_section_block, 192 B, max 16
///               +0x84 block  instant responses -> new_instantaneous_damage_response_block,
///                                                 176 B, max 16
///                      +0x58 block  region_transitions -> s_region_transition, 8 B, max 16
///                             +0x00 string id  region
///                             +0x04 short enum new state  (model_state_enum: 0..4)
/// ```
///
/// The reflection data leaves the response's own interior offsets uncertain past +0x34 because
/// two of its fields are runtime-defined structs (`ires`, `irem`) whose size it does not carry.
/// They are **solved from the element size**, which closes with nothing left over: the named
/// fields after them sum to 104 bytes, the element is 176, and 176 - 0x34 - 104 = 20 = a
/// `tag reference` (16, the damage effect) plus a `string id` (4, its marker). That fixes
/// `looping effect` at +0x48 and `region_transitions` at **+0x58**.
///
/// **And the engine confirms all four numbers independently.**
/// `PlayerInteraction_DamageSectionResponse` (`haloreach.dll` 0x180464F10) walks the same tables
/// to apply one of these events, in DWORD-indexed cache addressing:
///
/// ```text
///   *(float *)(hlmt + 0xD0)              the damage info's `maximum vitality`  = 0xCC + 0x04
///   *(int   *)(hlmt + 0xE0) / + 0xE4     the damage SECTIONS count / pointer   = 0xCC + 0x14/0x18
///   (3*section << 4)                     the section stride, 48 dwords         = 192 bytes
///   *(int *)(section + 0x84) / + 0x88    the instant RESPONSES count / pointer
///   (response * 0x2C)                    the response stride, 0x2C dwords      = 176 bytes
///   *(int *)(response + 0x88) == -1      "this response has no delay effect"
/// ```
///
/// That last one is the check that pins the interior: a `tag reference` in a cache carries its
/// tag index at +0x0C, so a `delay effect` index at +0x88 puts the reference itself at +0x7C —
/// which is exactly where the solved layout puts it, counting forward from
/// `region_transitions` at +0x58 through `response damage transfers` (+0x64),
/// `destroy instance group` (+0x70), `custom response behavior` (+0x72),
/// `custom response label` (+0x74) and `response delay` (+0x78).
///
/// [`DamageResponses::decode`]'s own sanity checks (a block count of 0..16 and a state in 0..4)
/// fail loudly rather than inventing a state if any of it ever turns out to be wrong.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DamageResponses {
    /// `[section][response]` -> the worst state that response transitions a region to.
    pub sections: Vec<Vec<u8>>,
}

impl DamageResponses {
    /// Read the table for `obj_tag`'s model. `None` when the object has no model, no damage info
    /// or no section that transitions any region — i.e. a model with no authored damage states,
    /// which is most Forge props.
    pub fn decode(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, obj_tag: u32) -> Option<Self> {
        let hlmt = crate::scene::object_hlmt(native, cache, obj_tag)?;
        let u = |s: &[u8], o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
        // hlmt through the damage-info struct: +0xCC + 0x14 is the damage sections block.
        let h = native.tag_meta(cache, hlmt, 0, 0xEC)?;
        let (scnt, sptr) = (u(&h, 0xCC + 0x14), u(&h, 0xCC + 0x18));
        if scnt == 0 || scnt > 16 {
            return None;
        }
        let sb = native.tag_ptr_bytes(cache, sptr, scnt * 192)?;
        let mut out = DamageResponses { sections: Vec::with_capacity(scnt as usize) };
        let mut any = false;
        for s in 0..scnt as usize {
            let se = &sb[s * 192..(s + 1) * 192];
            let (rcnt, rptr) = (u(se, 0x84), u(se, 0x88));
            let mut row: Vec<u8> = Vec::new();
            if rcnt > 0 && rcnt <= 16 {
                if let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * 176) {
                    for r in 0..rcnt as usize {
                        let re = &rb[r * 176..(r + 1) * 176];
                        let (tcnt, tptr) = (u(re, 0x58), u(re, 0x5C));
                        let mut worst = 0u8;
                        if tcnt > 0 && tcnt <= 16 {
                            if let Some(tb) = native.tag_ptr_bytes(cache, tptr, tcnt * 8) {
                                for t in 0..tcnt as usize {
                                    let st = i16::from_le_bytes([tb[t * 8 + 4], tb[t * 8 + 5]]);
                                    if (0..STATES as i16).contains(&st) {
                                        worst = worst.max(st as u8);
                                    }
                                }
                            }
                        }
                        any |= worst > 0;
                        row.push(worst);
                    }
                }
            }
            out.sections.push(row);
        }
        any.then_some(out)
    }

    /// The state one `damage_section_response` puts the model into — **0 for the two thirds of
    /// the event's own vocabulary that are not a damage state at all.**
    ///
    /// Only an event that names a real instant response (index >= 0) *and* carries a mode other
    /// than 1 actually runs that response. Both exclusions are the engine's own branches, read
    /// out of the receiver, not cautious defaults:
    ///
    /// `PlayerInteraction_DamageSectionResponse` (0x180464F10) splits on the response index:
    ///
    /// ```text
    ///   if (response < 0 || response >= section_response_count) {
    ///       if (response != -1)  return;              // out of range: dropped
    ///       if (mode == 2)       return sub_1804E1C80(object, section);
    ///       if (mode == 3)       return sub_1804E1DA4(object, section);
    ///       return;                                   // any other mode: dropped
    ///   }
    ///   ... PlayerInteraction_DispatchConditionalChildOrScriptCommand(obj, section, response, _, mode, _)
    /// ```
    ///
    /// * **`response == -1` is a SHIELD RECHARGE effect, not damage.** Those two leaves index the
    ///   damage SECTION (the same 192-byte stride) rather than a response: `sub_1804E1C80` reads
    ///   +0x54/+0x58 and `sub_1804E1DA4` reads +0x68/+0x6C, which in
    ///   `new_global_damage_section_block` are the tag index + marker of **`pre recharge effect`**
    ///   (the reference at +0x48) and of **`pre recharge abort effect`** (at +0x5C). A shield
    ///   about to come back, and a recharge interrupted. Neither touches a region — and taking
    ///   the section's worst transition for them, which an earlier draft of this did, would have
    ///   reported a vehicle as wrecked the first time its shield blinked.
    /// * **`mode == 1` is the DELAY EFFECT starting, not the response firing.** The dispatcher
    ///   it reaches ends:
    ///
    ///   ```text
    ///     if (mode != 1) return ScriptCommand_ProcessPlayerInteraction(...);   // the response RUNS
    ///     return PlayerInteraction_DispatchNetworkEvent(response + 0x88, ..., response + 0x8C);
    ///   ```
    ///
    ///   and `response + 0x88 / + 0x8C` are the tag index and marker of **`delay effect`** (the
    ///   reference at +0x7C) — "plays while the timer is counting down". A delayed response fires
    ///   later, and when it does the engine sends mode 0, so honouring mode 1 as well would apply
    ///   the damage `response delay` seconds early and twice.
    pub fn state_of(&self, section: u8, response: i8, mode: u8) -> u8 {
        if response < 0 || mode == 1 {
            return 0;
        }
        let Some(row) = self.sections.get(section as usize) else { return 0 };
        row.get(response as usize).copied().unwrap_or(0)
    }

    /// Which states this model's responses can ever reach (ascending, without 0) — the honest
    /// answer to "can this vehicle show damage at all".
    pub fn reachable(&self) -> Vec<u8> {
        let mut v: Vec<u8> = self.sections.iter().flatten().copied().filter(|&s| s > 0).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// #vehicle-damage  One object's damage history in a film: the ticks at which its displayed
/// state got worse, in order.
///
/// Monotone on purpose. A damage state in Reach is a one-way transition on a region — the engine
/// never puts a blown-off fender back — and the recording never restates an object's state, so
/// taking the running maximum of the responses that have fired is the only reading that does not
/// flicker. A respawned vehicle is a NEW network entity with its own timeline, so nothing has to
/// be reset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DamageTimeline {
    /// `(tick, state)`, strictly increasing in both.
    pub steps: Vec<(u32, u8)>,
    /// How many `damage_section_response` events named this object, including the ones whose
    /// response transitions nothing (a shield knocked down, a stun) — so the UI can say how much
    /// of the channel actually moved the picture.
    pub events: usize,
    /// Of those, how many actually RAN a response — index >= 0 and mode != 1, so neither the
    /// shield pre-recharge / abort branch nor a delay effect starting. See
    /// [`DamageResponses::state_of`].
    pub state_events: usize,
}

impl DamageTimeline {
    /// Fold one object's `damage_section_response` events, in tick order, through its model's
    /// response table.
    pub fn build<'a>(
        responses: &DamageResponses,
        events: impl Iterator<Item = &'a crate::film_action::ActionEvent>,
    ) -> Self {
        let mut out = DamageTimeline::default();
        let mut worst = 0u8;
        for e in events {
            if e.kind != crate::film_action::EV_DAMAGE_SECTION_RESPONSE {
                continue;
            }
            out.events += 1;
            let Some(sec) = e.section else { continue };
            let r = e.response.unwrap_or(-1);
            // A truncated parameter block yields no mode. Treat that as "cannot tell" and skip
            // it rather than defaulting to the one mode that changes the picture.
            let Some(m) = e.response_mode else { continue };
            if r >= 0 && m != 1 {
                out.state_events += 1;
            }
            let st = responses.state_of(sec, r, m);
            if st > worst {
                worst = st;
                out.steps.push((e.tick, st));
            }
        }
        out
    }

    /// The state to DRAW at `tick`. A pure function of the tick, so scrubbing backwards undoes
    /// the damage.
    pub fn state_at(&self, tick: u32) -> u8 {
        let i = self.steps.partition_point(|&(t, _)| t <= tick);
        if i == 0 { 0 } else { self.steps[i - 1].1 }
    }

    /// The tick this object was first drawn DESTROYED, if it ever was.
    pub fn destroyed_at(&self) -> Option<u32> {
        self.steps.iter().find(|&&(_, s)| s >= STATE_DESTROYED).map(|&(t, _)| t)
    }
}

// ---------------------------------------------------------------------------------------------
// 3. the wreck
// ---------------------------------------------------------------------------------------------

/// #vehicle-damage  Whether a vehicle is a WRECK at `tick`, and therefore whether the stream has
/// to keep drawing it.
///
/// The same shape Theater already uses for a corpse: the thing stays until the recording
/// DESTROYS it, which is the despawn the engine itself performed, never a fixed window. The one
/// extra condition a vehicle needs is the spawn-point override — a vehicle parked on its own
/// Forge spawn point is normally left to the embedded Forge scene to draw (otherwise two
/// identical models z-fight), but the Forge scene's copy is PRISTINE, so a wreck that happens to
/// have come to rest where it spawned must be drawn by the stream anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WreckRule {
    /// The tick the recording's own `damage_section_response` put it in the destroyed state.
    pub destroyed_at: Option<u32>,
    /// The tick a kind-4 destroy removed it, i.e. when the engine collected it. `None` = it was
    /// still there when the recording stopped.
    pub despawn: Option<u32>,
}

impl WreckRule {
    pub fn is_wreck_at(&self, tick: u32) -> bool {
        match self.destroyed_at {
            Some(d) => tick >= d && self.despawn.map_or(true, |x| tick <= x),
            None => false,
        }
    }
    /// Ticks the wreck persisted for before the engine collected it, when the recording states
    /// both ends.
    pub fn persisted(&self) -> Option<u32> {
        Some(self.despawn?.saturating_sub(self.destroyed_at?))
    }
}

// ---------------------------------------------------------------------------------------------
// 4. #film-jitter  sampling a track at the FRACTIONAL tick
// ---------------------------------------------------------------------------------------------

/// #film-jitter  A track's recorded position at a **fractional** tick.
///
/// ## Why this exists
///
/// `Playback::tick` is an `f32` and `Playback::advance` moves it by `dt * 60 * speed`, so the
/// playback clock is continuous — but every placement in Theater read it back as
/// `pb.tick.round() as u32`. The drawn transform was therefore quantised to whole ticks while the
/// clock driving it was not, which is a judder generator:
///
/// * at 1x on a 60 Hz display the rounded tick usually advances by one per frame, but the frame
///   interval never divides the tick interval exactly, so every few seconds one tick is drawn
///   twice and the next is skipped;
/// * at any other refresh rate, and at any other playback speed, it is much worse. At 0.25x —
///   the speed a Theater viewer actually uses to look at something — one tick covers several
///   display frames, so a vehicle at 10 wu/s stands still for four frames and then jumps.
///
/// Interpolating between the bracketing POSITION KEYS was already being done
/// ([`crate::film_decode::EntityTrack::sample`]); the fraction being thrown away was the one
/// between TICKS, and it is the same lerp.
///
/// **Equal to [`crate::film_decode::EntityTrack::sample`] at an integer tick**, which is what
/// keeps every existing determinism gate meaningful, and still a pure function of the clock:
/// scrubbing to the same `t` reproduces the same point exactly, forwards or backwards.
pub fn position_at(tr: &crate::film_decode::EntityTrack, t: f32) -> Option<Vec3> {
    let p = &tr.positions;
    let i = p.partition_point(|&(k, _)| (k as f32) <= t);
    if i == 0 {
        return None;
    }
    let (t0, p0) = p[i - 1];
    match p.get(i) {
        Some(&(t1, p1)) if t1 > t0 => {
            let w = ((t - t0 as f32) / (t1 - t0) as f32).clamp(0.0, 1.0);
            Some(Vec3::from(p0) + (Vec3::from(p1) - Vec3::from(p0)) * w)
        }
        _ => Some(Vec3::from(p0)),
    }
}

/// #film-jitter  A track's recorded `(forward, up)` basis at a **fractional** tick, slerped
/// between the bracketing keys exactly as
/// [`crate::film_decode::EntityTrack::orientation_at`] does — and equal to it at an integer tick.
pub fn orientation_at(tr: &crate::film_decode::EntityTrack, t: f32) -> Option<([f32; 3], [f32; 3])> {
    let o = &tr.orientations;
    let i = o.partition_point(|&(k, _, _)| (k as f32) <= t);
    if i == 0 {
        return None;
    }
    let (t0, f0, u0) = o[i - 1];
    match o.get(i) {
        Some(&(t1, f1, u1)) if t1 > t0 => {
            let w = ((t - t0 as f32) / (t1 - t0) as f32).clamp(0.0, 1.0);
            let q = |f: [f32; 3], u: [f32; 3]| glam::Quat::from_mat3(&crate::film_decode::basis_mat3(f, u));
            let m = glam::Mat3::from_quat(q(f0, u0).slerp(q(f1, u1), w));
            Some((m.x_axis.to_array(), m.z_axis.to_array()))
        }
        _ => Some((f0, u0)),
    }
}

/// #film-jitter  One recorded object's whole placement at a fractional tick: [`position_at`] and
/// [`orientation_at`] through [`vehicle_basis`], which is the one rule every recorded object is
/// placed with.
///
/// `None` when the recording has not placed this object yet. The MATRIX may be `None` on its own,
/// which is a vehicle the recording has not oriented yet — the caller then keeps its yaw-only
/// fallback, the contract [`vehicle_basis`] already has.
pub fn placement_at(tr: &crate::film_decode::EntityTrack, t: f32) -> Option<(Vec3, Option<Mat4>)> {
    let p = position_at(tr, t)?;
    Some((p, vehicle_basis(p, orientation_at(tr, t))))
}

// ---------------------------------------------------------------------------------------------
// 5. #film-vehid  identifying a vehicle by the definition it STATES
// ---------------------------------------------------------------------------------------------

/// #film-vehid  The `spawn_pos` given to a vehicle the film's own Forge variant does not place.
///
/// [`crate::film_anim::VehicleTable::on_its_spawn`] exists so the stream's copy of a parked
/// vehicle does not z-fight the Forge scene's copy — but a vehicle the variant never placed has
/// no Forge copy to fight with, so it must always be drawn. There is no "absent" in the
/// `[f32; 3]` that field is, so this is the value that makes the test it feeds
/// (`distance <= 0.25 wu`) false for EVERY input: the distance from a finite point is infinite,
/// and the distance from the sentinel to itself is `NaN` -- and neither `inf <= 0.25` nor
/// `NaN <= 0.25` is true. `f32::MAX` was tried first and the gate caught it: the sentinel is
/// within 0 wu of itself, so a vehicle at that (impossible) point would have been skipped.
///
/// `a_vehicle_with_no_forge_copy_is_always_drawn` gates exactly that.
pub const NO_FORGE_SPAWN: [f32; 3] = [f32::INFINITY, f32::INFINITY, f32::INFINITY];

/// #film-vehid  What [`identify_by_definition`] did, for the caption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DefIdentified {
    /// Type-15 tracks that the spawn-point search could not identify and the stated definition
    /// could.
    pub added: usize,
    /// Of those, how many the film's own Forge variant also places a copy of (so the
    /// spawn-coincident skip still applies to them).
    pub with_forge_copy: usize,
    /// Type-15 tracks still unidentified: no definition, or one that is not a `vehi`.
    pub still_unknown: usize,
    /// Type-15 tracks that never replicated a position, so there is nothing to place yet. These
    /// are the ones that ride a PARENT — see `#film-attach`.
    pub no_position: usize,
}

/// #film-vehid  Identify the vehicles the Forge spawn-point search could not, from the
/// **definition the creation message states**.
///
/// ## Why the spawn-point search is not enough, measured
///
/// [`crate::film_anim::VehicleTable::build`] identifies a vehicle by the nearest `vehi` spawn
/// point in the film's own embedded map variant. That was the only identification available when
/// it was written, and the note on it is explicit that the creation payload's tag reference "is
/// not readable at any fixed offset". **That is no longer true** — `#objdef` solved it: every
/// creation message opens with a 14-bit reference into the engine's nine-group network tag table,
/// `EntityTrack::def_wire` carries it raw, and `film_action::RefTable` resolves it exactly.
///
/// And the spawn-point rule fails completely on a gametype that SPAWNS its vehicles. On the
/// 12:16 clip `asq_mglo-1_forge_hal_CAC29F16` the variant "Xenon // Castle" places no vehicle at
/// all, so the search has nothing to match against and identifies **0 of 66** type-15 tracks —
/// while their own definitions resolve to `vehi warthog` (33 tracks, 16,873 recorded positions)
/// and `vehi warthog_chaingun` (33) with no ambiguity whatever. Thirty-three Warthogs drove
/// around that clip for twelve minutes and not one of them was drawn.
///
/// ## What this does
///
/// Fills in, for every type-15 track the table left out, the `VehicleIdent` its stated definition
/// gives — and only when that definition really is a `vehi`, so an entity type that is not a
/// vehicle cannot sneak into the vehicle lane. `spawn_pos` is the nearest variant object of the
/// SAME definition (not the nearest vehicle of any kind, which is what the distance-only search
/// had to settle for), or [`NO_FORGE_SPAWN`] when the variant places no copy at all.
///
/// A track with no recorded position is left alone and counted: there is nothing to place it at
/// until its parent is decoded (`#film-attach`), and claiming it as drawable would make the
/// caption's own arithmetic lie.
///
/// `resolve_def` turns a track's `def_wire` into `(tag id, 4-char group, tag path)` — the caller
/// supplies it because it needs the loaded cache AND the map file's datum salts, exactly as
/// [`crate::film_objective::ObjectTable::build`] does. `variant_spawns` is every present object
/// in the film's own map variant as `(resolved tag id, position)`.
pub fn identify_by_definition(
    idx: &crate::film_decode::FilmIndex,
    vt: &mut crate::film_anim::VehicleTable,
    variant_spawns: &[(u32, [f32; 3])],
    mut resolve_def: impl FnMut(u32) -> Option<(u32, String, String)>,
) -> DefIdentified {
    let mut out = DefIdentified::default();
    for tr in idx.tracks.iter().filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE) {
        if vt.vehicles.contains_key(&tr.net_id) {
            continue;
        }
        let Some(&(_, first)) = tr.positions.first() else {
            out.no_position += 1;
            continue;
        };
        let named = tr.def_wire.and_then(&mut resolve_def).filter(|(_, g, _)| g == "vehi");
        let Some((obj_tag, _group, path)) = named else {
            out.still_unknown += 1;
            continue;
        };
        // The nearest placed copy of THIS SAME definition, so a stream Warthog is never matched
        // against some other vehicle's spawn point.
        let near = variant_spawns
            .iter()
            .filter(|(tag, _)| *tag == obj_tag)
            .map(|&(_, p)| {
                let d = (p[0] - first[0]).powi(2) + (p[1] - first[1]).powi(2) + (p[2] - first[2]).powi(2);
                (p, d.sqrt())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
        let (spawn_pos, spawn_error) = match near {
            Some((p, d)) => {
                out.with_forge_copy += 1;
                (p, d)
            }
            None => (NO_FORGE_SPAWN, f32::INFINITY),
        };
        vt.vehicles.insert(
            tr.net_id,
            crate::film_anim::VehicleIdent {
                net_id: tr.net_id,
                obj_tag,
                seat_family: crate::film_anim::seat_family_for(&path),
                name: leaf,
                spawn_error,
                spawn_pos,
            },
        );
        out.added += 1;
        // Keep the table's own arithmetic true: `total == vehicles.len() + no_position +
        // unidentified` is what lets the caption print figures that add up.
        vt.unidentified = vt.unidentified.saturating_sub(1);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// 6. #film-attach  composing a CHILD object's placement from its PARENT's
// ---------------------------------------------------------------------------------------------

/// #film-attach  One object's attachment to another, as the recording states it.
///
/// Base field 12 of the shared object dirty block (`sub_1803D7E58`, gate-group index 0xC) is the
/// attachment record: a 27-bit parent object reference in the same namespace as a message's own
/// net id, an 8-bit attach point on the parent, a 16-bit-per-axis local offset and a local
/// `(forward, up)` basis read by the SAME decoder base field 2 uses, followed by an optional
/// 6-bit seat index. See `re-notes/reach_saved_film_layout.md` #film-attach for the full layout
/// and the oracle it was proven against.
///
/// Carried here, rather than assumed, because a child that rides a parent **stops replicating a
/// position of its own**: on the 12:16 clip 30 of 33 `warthog_chaingun` tracks never send one
/// position in their whole life, over as many as 1,364 messages each. Their place in the world is
/// entirely their parent's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Attachment {
    /// The parent's net id.
    pub parent: u32,
    /// The attach point on the parent (the 8-bit field).
    pub node: u8,
    /// The offset from the parent, in the parent's own frame, world units.
    pub offset: Vec3,
    /// The child's orientation in the parent's frame, as the recorded `(forward, up)` pair.
    /// `None` = aligned with the parent.
    pub basis: Option<([f32; 3], [f32; 3])>,
    /// The 6-bit seat index the record carries, `None` when it states -1.
    pub seat: Option<u8>,
}

/// #film-attach  A child's world placement: its parent's placement times its own local one.
///
/// `child_world = parent_world * translate(offset) * basis(local)` — the engine's own
/// composition, and the same `Matrix_BuildOrientation` rule [`vehicle_basis`] places every
/// recorded object with, so a turret on a rolling Warthog rolls with it.
///
/// A pure function of its two inputs, so if the parent's placement is a pure function of the tick
/// then so is this.
pub fn attached_world(parent_world: Mat4, a: &Attachment) -> Mat4 {
    let local = match a.basis {
        Some((f, u)) => Mat4::from_mat3(crate::film_decode::basis_mat3(f, u)),
        None => Mat4::IDENTITY,
    };
    parent_world * Mat4::from_translation(a.offset) * local
}

// ---------------------------------------------------------------------------------------------
// tests + measurement probes
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_decode::EntityTrack;

    /// A track with one recorded basis: rolled 90 degrees about its own forward axis, which is
    /// what a vehicle flung through a man cannon looks like half a second later.
    fn rolled_track(roll_deg: f32) -> EntityTrack {
        let r = glam::Quat::from_rotation_x(roll_deg.to_radians());
        let f = r * Vec3::X;
        let u = r * Vec3::Z;
        EntityTrack {
            net_id: 1,
            entity_type: 15,
            first_tick: 0,
            last_tick: 100,
            positions: vec![(0, [1.0, 2.0, 3.0]), (100, [1.0, 2.0, 3.0])],
            orientations: vec![(0, f.to_array(), u.to_array())],
            ..Default::default()
        }
    }

    /// **The gate the old code would have passed.** A test that only checks the heading is blind
    /// to the bug this fixed, so this one asserts on the part that was thrown away: a vehicle
    /// whose recorded basis is rolled must NOT be drawn level.
    #[test]
    fn a_rolled_vehicle_is_not_drawn_level() {
        for roll in [30.0f32, 90.0, 150.0] {
            let tr = rolled_track(roll);
            let (f, u) = tr.orientation_at(0).expect("the track has a recorded basis");
            // The recording itself states the roll.
            assert!(
                (tilt_degrees(u) - roll).abs() < 1.0,
                "roll {roll}: the decoded up axis is {:.1} degrees off level, expected {roll}",
                tilt_degrees(u)
            );
            let m = vehicle_basis(Vec3::from(tr.sample(0).unwrap()), Some((f, u))).expect("a basis");
            // ... and the placement must carry it. `Mat4::from_rotation_z(yaw)` cannot: its up
            // column is always +Z, so this is exactly the assertion the old placement fails.
            let drawn_up = m.z_axis.truncate().normalize();
            assert!(
                (tilt_degrees(drawn_up.to_array()) - roll).abs() < 1.0,
                "roll {roll}: the PLACEMENT's up axis is {:.1} degrees off level, so the vehicle is drawn level",
                tilt_degrees(drawn_up.to_array())
            );
            // And the old expression is still in the picture: the heading must not have moved.
            let yaw_old = f[1].atan2(f[0]).to_degrees();
            let level = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)) * Mat4::from_rotation_z(yaw_old.to_radians());
            assert!(
                (level.z_axis.truncate() - Vec3::Z).length() < 1e-5,
                "the yaw-only placement is supposed to be level; this test is not comparing what it claims"
            );
            assert!(
                (m.z_axis.truncate() - level.z_axis.truncate()).length() > 0.1,
                "roll {roll}: the new placement is indistinguishable from the yaw-only one"
            );
            // A rotation, not a shear: columns unit and mutually perpendicular.
            let (x, y, z) = (m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate());
            for (n, v) in [("x", x), ("y", y), ("z", z)] {
                assert!((v.length() - 1.0).abs() < 1e-4, "roll {roll}: {n} column is not unit ({})", v.length());
            }
            assert!(x.dot(y).abs() < 1e-4 && y.dot(z).abs() < 1e-4 && z.dot(x).abs() < 1e-4, "roll {roll}: basis is sheared");
            assert!(x.cross(y).dot(z) > 0.9, "roll {roll}: basis is mirrored");
            assert_eq!(m.w_axis.truncate(), Vec3::new(1.0, 2.0, 3.0), "the placement moved the vehicle");
        }
    }

    /// Interpolation is a slerp between the bracketing keys and a PURE FUNCTION of the tick, so
    /// scrubbing backwards reproduces the same basis.
    #[test]
    fn orientation_interpolates_and_is_pure() {
        let q0 = glam::Quat::IDENTITY;
        let q1 = glam::Quat::from_rotation_x(90.0f32.to_radians());
        let tr = EntityTrack {
            net_id: 1,
            entity_type: 15,
            orientations: vec![
                (0, (q0 * Vec3::X).to_array(), (q0 * Vec3::Z).to_array()),
                (60, (q1 * Vec3::X).to_array(), (q1 * Vec3::Z).to_array()),
            ],
            ..Default::default()
        };
        let tilt_at = |t: u32| tilt_degrees(tr.orientation_at(t).unwrap().1);
        assert!(tilt_at(0) < 0.5, "at the first key the tilt should be 0, got {:.2}", tilt_at(0));
        assert!((tilt_at(30) - 45.0).abs() < 1.0, "half way between the keys should be 45 degrees, got {:.2}", tilt_at(30));
        assert!((tilt_at(60) - 90.0).abs() < 0.5, "at the second key the tilt should be 90, got {:.2}", tilt_at(60));
        // Held past the last key.
        assert!((tilt_at(600) - 90.0).abs() < 0.5);
        // PURE FUNCTION OF THE TICK: sample a sweep, scrub forward past the end and back to the
        // start, then sample the same sweep again and demand the identical bases. This is the
        // property that makes scrubbing undo a tumble rather than accumulate one.
        let sweep: Vec<u32> = vec![0, 7, 15, 30, 45, 59, 60, 120, 600];
        let first: Vec<Option<([f32; 3], [f32; 3])>> = sweep.iter().map(|&t| tr.orientation_at(t)).collect();
        for &t in sweep.iter().rev().chain(sweep.iter()) {
            let _ = tr.orientation_at(t);
        }
        let again: Vec<Option<([f32; 3], [f32; 3])>> = sweep.iter().map(|&t| tr.orientation_at(t)).collect();
        assert_eq!(first, again, "the orientation at a tick changed after scrubbing away and back");
        assert!(first[0].is_some() && first.iter().all(|o| o.is_some()));
    }

    /// The damage timeline is monotone, scrubs backwards, and never invents a state the model
    /// does not author.
    #[test]
    fn damage_timeline_is_monotone_and_scrubs() {
        // section 0: response 0 does nothing, response 1 -> major, response 2 -> destroyed
        let resp = DamageResponses { sections: vec![vec![0, 3, 4], vec![1]] };
        assert_eq!(resp.reachable(), vec![1, 3, 4]);
        assert_eq!(resp.state_of(0, 1, 0), 3);
        assert_eq!(
            resp.state_of(0, -1, 2),
            0,
            "a -1 response is the engine's shield pre-recharge / abort branch and must change nothing"
        );
        assert_eq!(
            resp.state_of(0, 2, 1),
            0,
            "mode 1 is the delay effect starting, not the response firing, so it must change nothing"
        );
        assert_eq!(resp.state_of(9, 0, 0), 0, "a section the model does not have must not produce a state");
        let ev = |tick: u32, section: u8, response: i8| crate::film_action::ActionEvent {
            tick,
            kind: crate::film_action::EV_DAMAGE_SECTION_RESPONSE,
            subject: Some(7),
            section: Some(section),
            response: Some(response),
            response_mode: Some(0),
            ..Default::default()
        };
        let evs = vec![ev(100, 1, 0), ev(200, 0, 1), ev(250, 1, 0), ev(300, 0, 2)];
        let tl = DamageTimeline::build(&resp, evs.iter());
        assert_eq!(tl.events, 4);
        assert_eq!(tl.state_events, 4, "every event in this fixture names a real response");
        assert_eq!(tl.steps, vec![(100, 1), (200, 3), (300, 4)], "a worse state is a step, an equal or lesser one is not");
        assert_eq!(tl.state_at(0), 0);
        assert_eq!(tl.state_at(99), 0);
        assert_eq!(tl.state_at(100), 1);
        assert_eq!(tl.state_at(299), 3);
        assert_eq!(tl.state_at(10_000), 4);
        // scrubbing back undoes it
        assert_eq!(tl.state_at(150), 1);
        assert_eq!(tl.destroyed_at(), Some(300));
    }

    /// The wreck is present from the destroying response until the recording's own despawn, and
    /// gone after it.
    #[test]
    fn the_wreck_persists_until_the_recorded_despawn() {
        let w = WreckRule { destroyed_at: Some(300), despawn: Some(1260) };
        assert!(!w.is_wreck_at(299), "a vehicle is not a wreck before it is destroyed");
        assert!(w.is_wreck_at(300), "the wreck must be there on the tick the response fired");
        assert!(w.is_wreck_at(1260), "the wreck must still be there on the despawn tick");
        assert!(!w.is_wreck_at(1261), "the wreck must be gone once the recording destroyed the entity");
        assert_eq!(w.persisted(), Some(960), "16 seconds at 60 ticks");
        // a film that ends with the wreck still lying there states no despawn
        let open = WreckRule { destroyed_at: Some(300), despawn: None };
        assert!(open.is_wreck_at(1_000_000));
        assert_eq!(open.persisted(), None);
        // and a vehicle the recording never destroyed is never a wreck
        let alive = WreckRule { destroyed_at: None, despawn: Some(500) };
        assert!(!alive.is_wreck_at(400) && !alive.is_wreck_at(600));
    }

    // ------------------------------------------------------------------------------------------
    // #film-jitter  the fractional-tick sampler
    // ------------------------------------------------------------------------------------------

    /// A track with keys 10 ticks apart, moving along +X at 1 wu per tick and rolling as it goes.
    fn sparse_track() -> EntityTrack {
        let mut tr = EntityTrack { net_id: 7, entity_type: 15, first_tick: 0, last_tick: 100, ..Default::default() };
        for k in 0..=10u32 {
            let t = k * 10;
            tr.positions.push((t, [t as f32, 0.0, 0.0]));
            let r = glam::Quat::from_rotation_x((t as f32 * 3.0).to_radians());
            tr.orientations.push((t, (r * Vec3::X).to_array(), (r * Vec3::Z).to_array()));
        }
        tr
    }

    /// **The gate the rounded-tick placement fails.** Between two whole ticks the drawn point has
    /// to MOVE — that is the whole of the jitter fix — and at a whole tick it has to be exactly
    /// what the integer sampler gives, or every determinism gate in Theater is comparing two
    /// different functions.
    #[test]
    fn the_fractional_sampler_moves_between_ticks_and_agrees_on_them() {
        let tr = sparse_track();
        // 1. Agreement at every integer tick, EXACTLY -- including before the first key, past the
        //    last one, and on a key itself.
        for t in 0..=110u32 {
            let a = tr.sample(t).map(Vec3::from);
            let b = position_at(&tr, t as f32);
            assert_eq!(a, b, "tick {t}: the fractional sampler disagrees with EntityTrack::sample");
            let c = tr.orientation_at(t);
            let d = orientation_at(&tr, t as f32);
            assert_eq!(c, d, "tick {t}: the fractional orientation disagrees with EntityTrack::orientation_at");
        }
        // 2. It MOVES inside a tick. The rounded placement cannot: round(3.0), round(3.2) and
        //    round(3.4) are the same integer, so all three draw the same point.
        let (a, b, c) = (3.0f32, 3.2, 3.4);
        let round = |t: f32| tr.sample(t.round() as u32).map(Vec3::from).unwrap();
        assert_eq!(round(a), round(c), "this test is not comparing what it claims: the rounded sampler already moved");
        let (pa, pb, pc) = (position_at(&tr, a).unwrap(), position_at(&tr, b).unwrap(), position_at(&tr, c).unwrap());
        assert!((pb - pa).length() > 0.15, "the fractional sampler did not move between tick 3.0 and 3.2");
        assert!((pc - pb).length() > 0.15, "the fractional sampler did not move between tick 3.2 and 3.4");
        // ... and it moves by the RIGHT amount: this track travels 1 wu per tick.
        assert!(((pc - pa).length() - 0.4).abs() < 1e-3, "0.4 of a tick should be 0.4 wu, got {}", (pc - pa).length());
        // 3. The orientation interpolates inside a tick too.
        let ta = tilt_degrees(orientation_at(&tr, 0.0).unwrap().1);
        let tb = tilt_degrees(orientation_at(&tr, 5.0).unwrap().1);
        assert!((tb - ta).abs() > 5.0, "the basis did not turn between tick 0 and tick 5 ({ta:.1} -> {tb:.1} deg)");
        // 4. A PURE FUNCTION OF THE CLOCK, forwards and backwards. This is the rule every Theater
        //    placement obeys, and a fractional clock must not weaken it.
        let sweep: Vec<f32> = (0..=400).map(|i| i as f32 * 0.27).collect();
        let first: Vec<Option<Vec3>> = sweep.iter().map(|&t| position_at(&tr, t)).collect();
        for &t in sweep.iter().rev() {
            let _ = position_at(&tr, t);
        }
        let again: Vec<Option<Vec3>> = sweep.iter().map(|&t| position_at(&tr, t)).collect();
        assert_eq!(first, again, "the drawn position at a tick changed after scrubbing away and back");
        // 5. Monotone along the path: no sample may ever go backwards on a track that only goes
        //    forwards. A sampler that picked the wrong bracket would.
        for w in first.windows(2) {
            if let (Some(a), Some(b)) = (w[0], w[1]) {
                assert!(b.x >= a.x - 1e-4, "the drawn point went backwards: {} -> {}", a.x, b.x);
            }
        }
    }

    /// `placement_at` is the two samplers through `vehicle_basis`, and must agree with doing it
    /// by hand — so there is one placement rule, not two.
    #[test]
    fn placement_at_is_the_two_samplers_through_vehicle_basis() {
        let tr = sparse_track();
        for t in [0.0f32, 3.7, 25.5, 99.9, 500.0] {
            let (p, m) = placement_at(&tr, t).expect("the track is placed");
            assert_eq!(p, position_at(&tr, t).unwrap());
            assert_eq!(m, vehicle_basis(p, orientation_at(&tr, t)));
            let m = m.expect("this track has a basis");
            assert_eq!(m.w_axis.truncate(), p, "the placement does not sit at the sampled position");
        }
        // A track with a position and NO orientation yields a position and no matrix, so the
        // caller keeps its yaw fallback rather than being handed a level one.
        let flat = EntityTrack { net_id: 1, entity_type: 15, positions: vec![(0, [1.0, 2.0, 3.0])], ..Default::default() };
        let (p, m) = placement_at(&flat, 0.0).unwrap();
        assert_eq!(p, Vec3::new(1.0, 2.0, 3.0));
        assert!(m.is_none(), "a vehicle the recording never oriented must not be handed a fabricated basis");
        // And a track the recording has not placed yet yields nothing at all.
        assert!(placement_at(&flat, -5.0).is_none());
    }

    // ------------------------------------------------------------------------------------------
    // #film-vehid  identification by the stated definition
    // ------------------------------------------------------------------------------------------

    fn veh_track(net: u32, first: [f32; 3], positions: usize, def: Option<u32>) -> EntityTrack {
        let mut tr = EntityTrack {
            net_id: net,
            entity_type: crate::film_anim::VEHICLE_TYPE,
            first_tick: 0,
            last_tick: 100,
            def_wire: def,
            ..Default::default()
        };
        for k in 0..positions {
            tr.positions.push((k as u32, [first[0] + k as f32, first[1], first[2]]));
        }
        tr
    }

    /// **The gate the spawn-point-only identification fails.** Four type-15 tracks on a film
    /// whose Forge variant places NO vehicle: the distance search can identify none of them, and
    /// their stated definitions identify every one that is a `vehi` and refuse the one that is
    /// not.
    #[test]
    fn a_vehicle_is_identified_by_the_definition_it_states() {
        let idx = crate::film_decode::FilmIndex {
            tracks: vec![
                veh_track(0x10, [0.0, 0.0, 0.0], 50, Some(1)),   // a warthog that drove
                veh_track(0x11, [50.0, 0.0, 0.0], 20, Some(1)),  // another
                veh_track(0x12, [0.0, 0.0, 0.0], 0, Some(2)),    // a turret: NO position at all
                veh_track(0x13, [9.0, 9.0, 9.0], 5, Some(3)),    // a definition that is not a vehi
                veh_track(0x14, [1.0, 1.0, 1.0], 5, None),       // no definition at all
            ],
            ..Default::default()
        };
        // The variant places nothing, which is the measured case on the 12:16 clip.
        let mut vt = crate::film_anim::VehicleTable {
            total: 5,
            unidentified: 4,
            no_position: 1,
            ..Default::default()
        };
        let resolve = |w: u32| match w {
            1 => Some((0xAAAA, "vehi".to_string(), "objects\\vehicles\\warthog\\warthog".to_string())),
            2 => Some((0xBBBB, "vehi".to_string(), "objects\\vehicles\\warthog\\warthog_chaingun".to_string())),
            3 => Some((0xCCCC, "bloc".to_string(), "objects\\levels\\shared\\fusion_coil".to_string())),
            _ => None,
        };
        let r = identify_by_definition(&idx, &mut vt, &[], resolve);
        assert_eq!(r.added, 2, "the two Warthogs that reported a position must be identified");
        assert_eq!(r.no_position, 1, "the turret has no position to place it at and must be counted, not claimed");
        assert_eq!(r.still_unknown, 2, "a `bloc` and a track with no definition must NOT enter the vehicle lane");
        assert_eq!(r.with_forge_copy, 0, "this variant places nothing");
        assert_eq!(vt.vehicles.len(), 2);
        assert_eq!(vt.vehicles[&0x10].name, "warthog");
        assert_eq!(vt.vehicles[&0x10].obj_tag, 0xAAAA);
        assert_eq!(vt.vehicles[&0x10].seat_family, "warthog", "the seat family must come from the resolved path");
        assert!(!vt.vehicles.contains_key(&0x13), "a fusion coil was admitted to the vehicle lane");
        // The table's own arithmetic stays true, which is what lets the caption add up.
        assert_eq!(vt.total, vt.vehicles.len() + vt.no_position + vt.unidentified);
    }

    /// **The gate that makes the identification worth anything:** a vehicle the film's own Forge
    /// variant does not place must be DRAWN, at every position in the world, and one it does
    /// place must still be left to the scene while it is parked on it.
    #[test]
    fn a_vehicle_with_no_forge_copy_is_always_drawn() {
        let ident = |spawn: [f32; 3]| crate::film_anim::VehicleIdent {
            net_id: 1,
            obj_tag: 1,
            name: "warthog".into(),
            seat_family: "warthog".into(),
            spawn_error: 0.0,
            spawn_pos: spawn,
        };
        let no_copy = ident(NO_FORGE_SPAWN);
        for p in [[0.0f32, 0.0, 0.0], [1e4, -1e4, 400.0], [-0.01, 0.0, 0.0], NO_FORGE_SPAWN] {
            assert!(
                !crate::film_anim::VehicleTable::on_its_spawn(&no_copy, p),
                "a vehicle with no Forge copy was skipped at {p:?} -- the stream's copy is the only one there is"
            );
        }
        // ... and the ordinary case is untouched: a vehicle parked on a real spawn point is still
        // left to the scene, and the same vehicle driven off it is drawn.
        let parked = ident([10.0, 20.0, 30.0]);
        assert!(crate::film_anim::VehicleTable::on_its_spawn(&parked, [10.0, 20.0, 30.1]));
        assert!(!crate::film_anim::VehicleTable::on_its_spawn(&parked, [10.0, 20.5, 30.0]));
    }

    // ------------------------------------------------------------------------------------------
    // #film-attach  the child/parent composition
    // ------------------------------------------------------------------------------------------

    /// A child rides its parent: whatever the parent does, the child's drawn position keeps a
    /// CONSTANT distance from it — which is exactly the oracle the objective lane used to prove a
    /// carried flag, and exactly what 30 of the 12:16 clip's 33 turrets cannot do today, because
    /// they are drawn at a position they never reported.
    #[test]
    fn an_attached_child_keeps_a_constant_offset_from_its_parent() {
        let a = Attachment {
            parent: 0x1234,
            node: 3,
            offset: Vec3::new(0.0, 0.0, 0.6),
            basis: None,
            seat: Some(1),
        };
        // The parent drives a circle, pitching and rolling as it goes.
        let mut dists: Vec<f32> = Vec::new();
        let mut local: Vec<Vec3> = Vec::new();
        for k in 0..240 {
            let t = k as f32 * 0.05;
            let pos = Vec3::new(20.0 * t.cos(), 20.0 * t.sin(), 5.0 + t.sin());
            let r = glam::Quat::from_rotation_z(t) * glam::Quat::from_rotation_x(0.4 * t.sin());
            let parent = Mat4::from_rotation_translation(r, pos);
            let child = attached_world(parent, &a);
            dists.push((child.w_axis.truncate() - pos).length());
            // and in the PARENT's frame the child never moves at all
            local.push(parent.inverse().transform_point3(child.w_axis.truncate()));
        }
        let (min, max) = dists.iter().fold((f32::MAX, f32::MIN), |(a, b), &d| (a.min(d), b.max(d)));
        assert!(
            (max - min) < 1e-3,
            "the child's distance from its parent drifted by {:.4} wu over the drive",
            max - min
        );
        assert!((min - 0.6).abs() < 1e-3, "the offset is 0.6 wu, drawn at {min:.4}");
        for p in &local {
            assert!((*p - a.offset).length() < 1e-3, "the child moved in its parent's own frame: {p:?}");
        }
        // A child with its own local basis is TURNED by the parent, not merely carried.
        let turned = Attachment {
            basis: Some(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])),
            ..a
        };
        let parent = Mat4::from_rotation_z(90.0f32.to_radians());
        let w = attached_world(parent, &turned);
        let fwd = w.x_axis.truncate();
        assert!(
            (fwd - Vec3::NEG_X).length() < 1e-4,
            "a child turned 90 deg on a parent turned 90 deg should face -X, got {fwd:?}"
        );
        // ... and an identity parent leaves the child exactly where its local placement puts it.
        let w = attached_world(Mat4::IDENTITY, &a);
        assert_eq!(w.w_axis.truncate(), a.offset);
    }
}

#[cfg(test)]
mod probes {
    //! Measurement probes for #vehicle-rot / #vehicle-damage. Their job is to PRINT: run one with
    //! `--nocapture` and the number either supports a claim in this module's documentation or it
    //! does not. A probe that prints a clean negative is a result.

    use super::*;
    use crate::film;
    use crate::film_decode::{self, FilmIndex};

    /// `forge_halo` is the only map the vehicle corpus is on, so one cache serves every probe.
    static OWN_CACHE: std::sync::OnceLock<Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle)>> =
        std::sync::OnceLock::new();

    fn open_reach(stem: &str) -> Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle)> {
        OWN_CACHE
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
                let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
                let native = std::sync::Arc::new(hms_native::NativeDll::load(&dll).ok()?);
                let cands = crate::mapcat::enumerate();
                let map = cands.iter().find(|c| {
                    c.game == crate::mapcat::Game::Reach
                        && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(stem))
                })?;
                if let Some(dir) = map.path.parent() {
                    native.set_shared_cache_dirs(&[dir.to_path_buf()]);
                }
                let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
                Some((native, cache))
            })
            .clone()
    }

    /// A film's basename, which is the ONLY part of the path safe to print: the directory above
    /// it carries the machine owner's real first name.
    fn name(p: &std::path::Path) -> String {
        p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    }

    fn indexed(p: &std::path::Path) -> Option<(film::Film, FilmIndex)> {
        let f = film::open(p).ok()?;
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            return None;
        }
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        (idx.entity_messages > 0).then_some((f, idx))
    }

    /// The scenario sandbox palette resolver [`crate::film_anim::VehicleTable::build`] needs, so a
    /// probe identifies vehicles the exact way Theater does.
    fn vehicle_table(
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        idx: &FilmIndex,
    ) -> crate::film_anim::VehicleTable {
        let palette = native
            .find_scenario(cache)
            .map(|scnr| native.scnr_forge_palette_full(cache, scnr))
            .unwrap_or_default();
        let mut order: Vec<(u32, u32)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for e in &palette {
            let k = (e.palette_index, e.entry_within);
            if seen.insert(k) {
                order.push(k);
            }
        }
        crate::film_anim::VehicleTable::build(idx, |folder, item| {
            let &(pi, ew) = order.get(folder as usize)?;
            let e = palette
                .iter()
                .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
                .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
            if native.tag_class(cache, e.tag_short).as_deref() != Some("vehi") {
                return None;
            }
            Some((e.tag_short, native.tag_name(cache, e.tag_short).unwrap_or_default()))
        })
    }

    /// #vehicle-rot  **Does the recording carry a vehicle's ROTATION, and how densely?**
    ///
    /// The claim being tested is that base field 2 is on the wire for vehicles with real pitch
    /// and roll in it — not just a heading. So: per film, how many orientation samples a vehicle
    /// gets per position sample, the longest gap between them, and how far off level they get.
    #[test]
    fn probe_vehicle_orientation_density() {
        let films = film::find_finished_films();
        if films.is_empty() {
            eprintln!("film_vehicle: no films on this machine");
            return;
        }
        let mut tot_pos = 0usize;
        let mut tot_ori = 0usize;
        let mut tracks = 0usize;
        let mut tilted = [0usize; 4]; // samples over 5 / 15 / 45 / 90 degrees off level
        let mut all = 0usize;
        let mut worst: Option<(String, u32, f32)> = None;
        println!(
            "{:<44} {:>4} {:>8} {:>8} {:>6} {:>7} {:>7} {:>7} {:>7}",
            "film", "veh", "pos", "orient", "o/pos", "maxgap", ">5deg", ">45deg", "maxtilt"
        );
        for p in films.iter() {
            let Some((_f, idx)) = indexed(&p) else { continue };
            let vs: Vec<&film_decode::EntityTrack> =
                idx.tracks.iter().filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE).collect();
            if vs.is_empty() {
                continue;
            }
            let (mut fp, mut fo, mut gap, mut over5, mut over45, mut maxtilt) = (0usize, 0usize, 0u32, 0usize, 0usize, 0.0f32);
            for t in &vs {
                tracks += 1;
                fp += t.positions.len();
                fo += t.orientations.len();
                for w in t.orientations.windows(2) {
                    gap = gap.max(w[1].0.saturating_sub(w[0].0));
                }
                for &(tick, _f, u) in &t.orientations {
                    let d = tilt_degrees(u);
                    all += 1;
                    for (k, lim) in [5.0f32, 15.0, 45.0, 90.0].iter().enumerate() {
                        if d > *lim {
                            tilted[k] += 1;
                        }
                    }
                    if d > 5.0 {
                        over5 += 1;
                    }
                    if d > 45.0 {
                        over45 += 1;
                    }
                    if d > maxtilt {
                        maxtilt = d;
                        if worst.as_ref().map_or(true, |w| d > w.2) {
                            worst = Some((name(&p), tick, d));
                        }
                    }
                }
            }
            tot_pos += fp;
            tot_ori += fo;
            println!(
                "{:<44} {:>4} {:>8} {:>8} {:>6.2} {:>7} {:>7} {:>7} {:>6.1}",
                name(&p),
                vs.len(),
                fp,
                fo,
                if fp > 0 { fo as f32 / fp as f32 } else { 0.0 },
                gap,
                over5,
                over45,
                maxtilt
            );
        }
        println!(
            "\nTOTAL {tracks} vehicle track(s): {tot_pos} positions, {tot_ori} orientations ({:.1}% as many)",
            if tot_pos > 0 { 100.0 * tot_ori as f64 / tot_pos as f64 } else { 0.0 }
        );
        println!(
            "of {all} recorded bases: {} over 5 deg off level ({:.1}%), {} over 15 ({:.1}%), {} over 45 ({:.1}%), {} over 90 ({:.1}%)",
            tilted[0], 100.0 * tilted[0] as f64 / all.max(1) as f64,
            tilted[1], 100.0 * tilted[1] as f64 / all.max(1) as f64,
            tilted[2], 100.0 * tilted[2] as f64 / all.max(1) as f64,
            tilted[3], 100.0 * tilted[3] as f64 / all.max(1) as f64,
        );
        if let Some((f, t, d)) = worst {
            println!("worst tilt {d:.1} deg at tick {t} of {f}");
        }
    }

    /// #vehicle-damage  **The hlmt read is GATED, not merely printed.**
    ///
    /// A wrong offset in [`DamageResponses::decode`] would not crash: it would read some other
    /// field, find a count out of range, and silently return `None` for every vehicle — which
    /// reads exactly like "this model authors no damage states" and would quietly turn the whole
    /// feature off. So this asserts on the shipped Reach vehicles themselves: a Warthog, a
    /// Mongoose, a Ghost and a Banshee all have authored damage states, their responses must
    /// reach the DESTROYED state, and every state the table yields must be a legal
    /// `model_state_enum` value.
    #[test]
    fn shipped_reach_vehicles_author_a_destroyed_response() {
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("film_vehicle: forge_halo is not on this machine; skipping");
            return;
        };
        // The sandbox vehicle palette, so this is every vehicle Forge can place rather than a
        // hand-written list of tag paths.
        let scnr = native.find_scenario(cache).expect("forge_halo has a scenario");
        let palette = native.scnr_forge_palette_full(cache, scnr);
        let mut seen: std::collections::BTreeMap<String, Option<Vec<u8>>> = Default::default();
        for e in &palette {
            if native.tag_class(cache, e.tag_short).as_deref() != Some("vehi") {
                continue;
            }
            let path = native.tag_name(cache, e.tag_short).unwrap_or_default();
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            let r = DamageResponses::decode(&native, cache, e.tag_short).map(|d| d.reachable());
            seen.entry(leaf).or_insert(r);
        }
        for (k, v) in &seen {
            eprintln!("  {k:<24} {v:?}");
        }
        assert!(!seen.is_empty(), "forge_halo's sandbox palette holds no `vehi` entry at all");
        for (k, v) in &seen {
            let Some(states) = v else { continue };
            assert!(
                states.iter().all(|&s| (1..STATES as u8).contains(&s)),
                "{k}: the response table yielded {states:?}, which is not a model_state_enum value"
            );
        }
        // The four that must be there. Reach authors a destroyed state for every drivable
        // vehicle (the warthog loses its windshield and fenders, the banshee its whole hull), so
        // a `None` or a table with no 4 in it means the offsets moved.
        for want in ["warthog", "mongoose", "ghost", "banshee"] {
            let hit = seen.iter().find(|(k, _)| k.starts_with(want));
            let Some((k, v)) = hit else {
                eprintln!("film_vehicle: forge_halo's palette has no '{want}'; not asserting on it");
                continue;
            };
            let states = v.as_ref().unwrap_or_else(|| {
                panic!("{k}: DamageResponses::decode found no damage info at all - the hlmt offsets are wrong")
            });
            assert!(
                states.contains(&STATE_DESTROYED),
                "{k}: its damage responses reach {states:?}, with no destroyed state - the region_transitions offset is wrong"
            );
        }
    }

    /// #vehicle-damage  **Does the recording state a vehicle's damage, and its despawn?**
    ///
    /// Needs the map, because the chain is recording -> the vehicle's own `hlmt` damage responses
    /// -> a `model_state_enum` value. Prints, per vehicle model, which states its responses can
    /// reach; then per damaged vehicle the timeline and the gap between the destroying response
    /// and the kind-4 destroy that collected it.
    #[test]
    fn probe_vehicle_damage_and_despawn() {
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("film_vehicle: forge_halo is not on this machine; skipping");
            return;
        };
        let films = film::find_finished_films();
        let mut models: std::collections::BTreeMap<String, Vec<u8>> = Default::default();
        let mut persisted: Vec<u32> = Vec::new();
        let mut films_used = 0usize;
        let (mut evt, mut evt_veh, mut damaged, mut destroyed, mut vehicles) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut state_evt = 0usize;
        // (mode, response >= 0) -> count, over every vehicle-named event: the proof that the
        // mode/response branches above are not a theory about an empty set.
        let mut modes: std::collections::BTreeMap<(u8, bool), usize> = Default::default();
        for p in films.iter() {
            let Some((_f, idx)) = indexed(&p) else { continue };
            let vt = vehicle_table(&native, cache, &idx);
            if vt.vehicles.is_empty() {
                continue;
            }
            films_used += 1;
            let mut per: std::collections::HashMap<u32, Vec<crate::film_action::ActionEvent>> = Default::default();
            for e in idx.events.iter().filter(|e| e.kind == crate::film_action::EV_DAMAGE_SECTION_RESPONSE) {
                evt += 1;
                if let Some(n) = e.subject {
                    per.entry(n).or_default().push(*e);
                }
            }
            let mut lines: Vec<String> = Vec::new();
            for v in vt.vehicles.values() {
                vehicles += 1;
                let Some(resp) = DamageResponses::decode(&native, cache, v.obj_tag) else {
                    models.entry(format!("{} (no damage info)", v.name)).or_default();
                    continue;
                };
                models.insert(v.name.clone(), resp.reachable());
                let empty = Vec::new();
                let evs = per.get(&v.net_id).unwrap_or(&empty);
                evt_veh += evs.len();
                for e in evs {
                    *modes
                        .entry((e.response_mode.unwrap_or(0), e.response.unwrap_or(-1) >= 0))
                        .or_default() += 1;
                }
                let tl = DamageTimeline::build(&resp, evs.iter());
                state_evt += tl.state_events;
                if tl.steps.is_empty() {
                    continue;
                }
                damaged += 1;
                let despawn = idx
                    .tracks
                    .iter()
                    .find(|t| t.net_id == v.net_id)
                    .and_then(|t| t.destroyed_tick);
                let w = WreckRule { destroyed_at: tl.destroyed_at(), despawn };
                if w.destroyed_at.is_some() {
                    destroyed += 1;
                    if let Some(k) = w.persisted() {
                        persisted.push(k);
                    }
                }
                lines.push(format!(
                    "    {:#010x} {:<14} {:>3} event(s) -> {:?}  destroyed_at {:?} despawn {:?} persisted {:?} ticks",
                    v.net_id,
                    v.name,
                    tl.events,
                    tl.steps,
                    w.destroyed_at,
                    despawn,
                    w.persisted()
                ));
            }
            if !lines.is_empty() {
                println!("{}  ({} vehicle(s))", name(&p), vt.vehicles.len());
                for l in lines.iter().take(14) {
                    println!("{l}");
                }
                if lines.len() > 14 {
                    println!("    ... and {} more", lines.len() - 14);
                }
            }
        }
        println!("\nevent shape over every vehicle-named damage_section_response: (mode, names a response) -> count");
        for ((m, named), n) in &modes {
            let what = match (m, named) {
                (1, true) => "a delay effect starting (NO state change)",
                (_, true) => "the response RUNNING (state change)",
                (2, false) => "pre recharge effect (NO state change)",
                (3, false) => "pre recharge abort effect (NO state change)",
                (_, false) => "dropped by the engine's own range check",
            };
            println!("  mode {m}, response {:>3} -> {n:>6}   {what}", if *named { "yes" } else { "-1" });
        }
        println!("\nmodels and the damage states their responses can reach:");
        for (m, st) in &models {
            println!("  {m:<30} {st:?}");
        }
        println!(
            "\n{films_used} film(s) with vehicles, {vehicles} vehicle(s); {evt} damage_section_response event(s) in total, \
             {evt_veh} of them naming a vehicle, {state_evt} of THOSE naming a real instant response (the rest are the \
             shield pre-recharge / abort branch); {damaged} vehicle(s) reached a damage state, {destroyed} were destroyed"
        );
        if !persisted.is_empty() {
            let mut v = persisted.clone();
            v.sort_unstable();
            println!(
                "wreck persistence, destroying response -> kind-4 destroy: n={} median {} ticks ({:.1} s), min {} max {}",
                v.len(),
                v[v.len() / 2],
                v[v.len() / 2] as f32 / 60.0,
                v[0],
                v[v.len() - 1]
            );
        } else {
            println!("no wreck in the corpus states BOTH a destroying response and a despawn");
        }
    }

    // -----------------------------------------------------------------------------------------
    // #film-attach  measurement probes for the four 2026-10-13 reports
    // -----------------------------------------------------------------------------------------

    /// The `forge_halo` cache's own `.map` bytes, which [`crate::film_action::RefTable::for_map`]
    /// needs for the per-tag datum salts.
    fn forge_halo_map_bytes() -> Option<Vec<u8>> {
        let m = crate::mapcat::enumerate().into_iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
        })?;
        std::fs::read(&m.path).ok()
    }

    /// Every track's definition resolved to `(group, tag leaf)`, the way the object lane does it.
    fn def_names(
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        idx: &FilmIndex,
    ) -> std::collections::HashMap<u32, (u32, String, String)> {
        let mut out = Default::default();
        let Some(bytes) = forge_halo_map_bytes() else { return out };
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) {
                tags.push((t, (*g).to_string()));
            }
        }
        let Some(table) = crate::film_action::RefTable::for_map(&bytes, &tags) else { return out };
        for tr in &idx.tracks {
            let Some(tag) = tr.def_wire.and_then(|w| table.resolve(w)) else { continue };
            let Some(g) = native.tag_class(cache, tag) else { continue };
            let Some(n) = native.tag_name(cache, tag) else { continue };
            let leaf = n.rsplit('\\').next().unwrap_or(&n).to_string();
            out.insert(tr.net_id, (tag, g, leaf));
        }
        out
    }

    /// The film named by `HMS_FILM`, or the biggest one on the machine.
    fn probe_film() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HMS_FILM") {
            return Some(std::path::PathBuf::from(p));
        }
        film::find_finished_films().into_iter().next()
    }

    /// #film-attach  **What is in the recording, by resolved definition, and what each draw lane
    /// would do with it.** The user's report is "there were warthogs and warthog turrets being
    /// spawned ... somehow that didn't get decoded"; this answers whether the tracks are there.
    #[test]
    #[ignore]
    fn probe_recorded_object_census() {
        let Some(p) = probe_film() else {
            eprintln!("no film");
            return;
        };
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("no forge_halo");
            return;
        };
        let f = film::open(&p).expect("film opens");
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        let names = def_names(&native, cache, &idx);
        println!(
            "{} : {} ticks, {} entity messages, {} tracks, {} variant objects",
            name(&p),
            idx.ticks,
            idx.entity_messages,
            idx.tracks.len(),
            idx.variant_objects.len()
        );

        // Per-track dirty-mask union, which is the only way to see what a track that never
        // reports a position DOES report.
        let mut union: std::collections::HashMap<u32, u64> = Default::default();
        let mut msgs: std::collections::HashMap<u32, usize> = Default::default();
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
            for e in &t.entities {
                *union.entry(e.net_id).or_default() |= e.dirty_mask;
                *msgs.entry(e.net_id).or_default() += 1;
            }
        }

        #[derive(Default)]
        struct Row {
            tracks: usize,
            with_pos: usize,
            with_ori: usize,
            moves: usize,
            positions: usize,
            orientations: usize,
            msgs: usize,
            union: u64,
        }
        let mut inv: std::collections::BTreeMap<(u8, String, String), Row> = Default::default();
        for tr in &idx.tracks {
            let key = match names.get(&tr.net_id) {
                Some((_, g, leaf)) => (tr.entity_type, g.clone(), leaf.clone()),
                None => (tr.entity_type, "????".to_string(), "(unresolved)".to_string()),
            };
            let r = inv.entry(key).or_default();
            r.tracks += 1;
            r.positions += tr.positions.len();
            r.orientations += tr.orientations.len();
            r.msgs += msgs.get(&tr.net_id).copied().unwrap_or(0);
            r.union |= union.get(&tr.net_id).copied().unwrap_or(0);
            if !tr.positions.is_empty() {
                r.with_pos += 1;
            }
            if !tr.orientations.is_empty() {
                r.with_ori += 1;
            }
            if tr.moves() {
                r.moves += 1;
            }
        }
        println!(
            "\n{:<4} {:<5} {:<30} {:>6} {:>7} {:>7} {:>6} {:>9} {:>9} {:>9}  mask union",
            "type", "group", "definition", "tracks", "w/pos", "w/ori", "moves", "positions", "orients", "msgs"
        );
        for ((ty, g, leaf), r) in &inv {
            println!(
                "{ty:<4} {g:<5} {leaf:<30} {:>6} {:>7} {:>7} {:>6} {:>9} {:>9} {:>9}  {:#018x}",
                r.tracks, r.with_pos, r.with_ori, r.moves, r.positions, r.orientations, r.msgs, r.union
            );
        }

        // What the two existing lanes would make of it.
        let vt = vehicle_table(&native, cache, &idx);
        println!(
            "\nVehicleTable (spawn-point identification): {} of {} type-15 track(s) identified, \
             {} with no position at all, {} with a position but no vehicle spawn within reach",
            vt.vehicles.len(),
            vt.total,
            vt.no_position,
            vt.unidentified
        );
        for tr in idx.tracks.iter().filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE) {
            let who = names.get(&tr.net_id).map(|(_, g, l)| format!("{g} {l}")).unwrap_or_else(|| "(unresolved)".into());
            let verdict = if vt.vehicles.contains_key(&tr.net_id) {
                "DRAWN (spawn matched)"
            } else if tr.positions.is_empty() {
                "not drawn: no_position"
            } else {
                "not drawn: unidentified"
            };
            println!(
                "  {:#010x} {:<28} {:>5} pos {:>5} ori {:>6} msg  ticks {}..{} destroy {:?}  mask {:#014x}  {verdict}",
                tr.net_id,
                who,
                tr.positions.len(),
                tr.orientations.len(),
                msgs.get(&tr.net_id).copied().unwrap_or(0),
                tr.first_tick,
                tr.last_tick,
                tr.destroyed_tick,
                union.get(&tr.net_id).copied().unwrap_or(0)
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // #film-attach  BASE FIELD 12 IS THE OBJECT-TO-OBJECT ATTACHMENT RECORD
    // -----------------------------------------------------------------------------------------
    //
    // Read out of `sub_1803D7E58` (the base-class dirty walk) at gate-group index 0xC:
    //
    // ```text
    //   if (ReadFlag()) {                              // 0 = this object has NO parent
    //       sub_1803D2D84(stream, &state[0x120]);      // bits(16)<<16 | bits(11): a 27-bit
    //                                                  //   OBJECT REFERENCE, the same shape and
    //                                                  //   the same namespace as a message's own
    //                                                  //   net id -- so it is directly
    //                                                  //   comparable to EntityTrack::net_id
    //       state[0x124] = ReadBits(8);                // the attach point on the parent
    //       state[0x126] = ReadFlag();
    //       state[0x127] = ReadFlag();
    //       sub_1803C6CE4(stream, &state[0x128], state[0x127] ? 0 : 2, 0x10);
    //                                                  // 3 x 16 bits: the LOCAL OFFSET, dequantised
    //                                                  //   against a per-mode (min,max) triple at
    //                                                  //   `unk_180A48E30 + 24*mode`
    //       sub_1803D9330(stream, 0, &state[0x140], &state[0x134]);
    //                                                  // the LOCAL ORIENTATION -- the SAME decoder
    //                                                  //   base field 2 uses, so the same 19-bit
    //                                                  //   axis + 8-bit angle
    //   } else {
    //       state[0x11C] = state[0x120] = -1;          // the cleared record
    //   }
    //   state[0x14C] = ReadFlag() ? -1 : ReadBits(6);  // a 6-bit SEAT INDEX, -1 = none
    //   state[0x14E] = ReadFlag();
    // ```
    //
    // The 6-bit seat is the same width `biped_enter_vehicle` / `biped_exit_vehicle` carry
    // (`sub_1803B5644`, "a bare 6-bit seat index"), which corroborates the reading.
    //
    // Everything below REIMPLEMENTS the walk inside the probe module rather than in
    // `film_decode`, deliberately: this is the measurement that says the layout is right, and the
    // field belongs in `film_decode::base_dirty_walk` once it is.

    /// One decoded base-field-12 record.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct AttachRaw {
        /// The 27-bit parent object reference, in the net-id namespace. `None` = the cleared
        /// record, i.e. this object has no parent.
        pub parent: Option<u32>,
        /// The 8-bit attach point on the parent.
        pub node: Option<u8>,
        /// The three raw 16-bit local-offset axes.
        pub offset_q: Option<[u32; 3]>,
        /// The local orientation, in base field 2's own raw `(axis, angle)` form.
        pub orient_raw: Option<(Option<u32>, u32)>,
        /// The 6-bit seat index, `None` when the record states -1.
        pub seat: Option<u8>,
    }

    /// The base-class walk, fields 0..0x13, with GATE-SKIP semantics (a clear gate skips that
    /// gate's group, it does not end the walk) and with field 12 costed.
    ///
    /// Returns `(rebuilt mask, the field-12 record, the field the walk gave up on)`.
    fn base_walk_to_12(
        r: &mut film_decode::BitReader,
        p: &film_decode::DecodeParams,
    ) -> (u64, Option<AttachRaw>, Option<u32>) {
        const GATES: [u32; 2] = [2, 8];
        let mut g = 0usize;
        let mut field = 0u32;
        let mut mask = 0u64;
        let mut attach = None;
        macro_rules! t {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(_) => return (mask, attach, Some(field)),
                }
            };
        }
        while field < 0x14 {
            if g < GATES.len() && field == GATES[g] {
                g += 1;
                if !t!(r.flag()) {
                    field = GATES.get(g).copied().unwrap_or(0x14);
                    continue;
                }
            }
            let present = t!(r.flag());
            let this = field;
            field += 1;
            if !present {
                continue;
            }
            mask |= 1u64 << this;
            match this {
                0 => {
                    if !t!(r.flag()) {
                        return (mask, attach, Some(this));
                    }
                    let w = p.cv3_axis_bits;
                    for b in w {
                        t!(r.bits(b));
                    }
                }
                1 => {
                    if !t!(r.flag()) {
                        t!(r.bits(19));
                        t!(r.bits(10));
                    }
                }
                2 => {
                    if !t!(r.flag()) {
                        t!(r.bits(19));
                    }
                    t!(r.bits(8));
                }
                3 => {
                    if !t!(r.flag()) {
                        t!(r.bits(19));
                        t!(r.bits(8));
                    }
                }
                4 => {
                    t!(r.bits(8));
                    t!(r.bits(1));
                }
                5 => {}
                6 | 7 => {
                    t!(r.bits(8));
                    t!(r.bits(4));
                }
                8 => {
                    t!(r.bits(5));
                    t!(r.bits(48));
                }
                9 => {
                    t!(r.bits(5));
                    for _ in 0..16 {
                        if t!(r.flag()) {
                            t!(r.bits(7));
                        }
                    }
                }
                10 => {
                    let n = t!(r.bits(5)) as u32;
                    if n != 0 {
                        t!(r.bits(n));
                        t!(r.bits(n));
                    }
                }
                // Field 11: 15 bits, 4 bits, then `sub_1804EC888` -- 2 mode bits and then
                // {0, 1, 3, 4} eleven-bit quantised reals for modes {0, 1, 2, 3}.
                11 => {
                    t!(r.bits(15));
                    t!(r.bits(4));
                    let mode = t!(r.bits(2));
                    let n = [0u32, 1, 3, 4][mode as usize];
                    for _ in 0..n {
                        t!(r.bits(11));
                    }
                }
                12 => {
                    let mut a = AttachRaw::default();
                    if t!(r.flag()) {
                        let hi = t!(r.bits(16));
                        let lo = t!(r.bits(11));
                        a.parent = Some(((hi << 16) | lo) as u32);
                        a.node = Some(t!(r.bits(8)) as u8);
                        t!(r.flag());
                        t!(r.flag());
                        a.offset_q =
                            Some([t!(r.bits(16)) as u32, t!(r.bits(16)) as u32, t!(r.bits(16)) as u32]);
                        let axis = if t!(r.flag()) { None } else { Some(t!(r.bits(19)) as u32) };
                        a.orient_raw = Some((axis, t!(r.bits(8)) as u32));
                    }
                    if !t!(r.flag()) {
                        a.seat = Some(t!(r.bits(6)) as u8);
                    }
                    t!(r.flag());
                    attach = Some(a);
                }
                _ => return (mask, attach, Some(this)),
            }
        }
        (mask, attach, None)
    }

    /// The shared creation reader `sub_1803D5A8C`, costed so a CREATION's dirty block can be
    /// reached. A child object is created already attached, so this is where its first
    /// attachment record is.
    fn creation_block(r: &mut film_decode::BitReader, ty: u8, p: &film_decode::DecodeParams) -> bool {
        let shared = |r: &mut film_decode::BitReader| -> bool {
            if r.bits(p.tagref_bits).is_err() || r.bits(13).is_err() {
                return false;
            }
            match r.flag() {
                Ok(true) => {
                    if r.bits(10).is_err() {
                        return false;
                    }
                }
                Ok(false) => {}
                Err(_) => return false,
            }
            if r.bits(4).is_err() {
                return false;
            }
            // NOTE the INVERTED sense: a SET flag means 0xFF and reads nothing.
            match r.flag() {
                Ok(false) => {
                    if r.bits(4).is_err() {
                        return false;
                    }
                }
                Ok(true) => {}
                Err(_) => return false,
            }
            r.bits(7).is_ok() && r.bits(32).is_ok()
        };
        match ty {
            12 | 13 | 14 | 15 | 18 => shared(r),
            17 => shared(r) && r.bits(12).is_ok() && r.bits(7).is_ok(),
            _ => false,
        }
    }

    /// Decode one entity message's base block as far as field 12. Walks UPDATES and the CREATION
    /// types whose creation block is costed.
    fn attach_of(e: &film_decode::EntityMessage, p: &film_decode::DecodeParams) -> (u64, Option<AttachRaw>, Option<u32>) {
        let mut r = film_decode::BitReader::new(&e.payload);
        if e.is_update {
            if r.skip(p.update_prefix_bits as usize).is_err() {
                return (0, None, Some(0));
            }
        } else if !creation_block(&mut r, e.entity_type, p) {
            return (0, None, Some(0));
        }
        base_walk_to_12(&mut r, p)
    }

    /// #film-attach  **Is there a parent reference in the entity stream, and does it name a real
    /// object?**
    ///
    /// The oracle is the one the kind-4 destroy was proven with and costs nothing: a 27-bit
    /// object reference read at the wrong bit offset is noise in a 134-million-value space, so if
    /// the decoded parent is a net id that EXISTS in this film and is ALIVE at that tick on
    /// essentially every message, the field is read correctly. A wrong offset cannot score.
    #[test]
    #[ignore]
    fn probe_attachment_parent_reference() {
        let Some(p) = probe_film() else {
            eprintln!("no film");
            return;
        };
        let f = film::open(&p).expect("film opens");
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        let names = match open_reach("forge_halo") {
            Some((native, cache)) => def_names(&native, cache, &idx),
            None => Default::default(),
        };
        let lookup: std::collections::HashMap<u32, usize> =
            idx.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
        let who = |net: u32| -> String {
            names.get(&net).map(|(_, g, l)| format!("{g} {l}")).unwrap_or_else(|| "?".into())
        };

        let (mut with12, mut named, mut alive, mut cleared, mut stopped) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut maskbad = 0usize;
        // per (child net id) -> (parent net id -> count), node histogram, seat histogram
        let mut per_child: std::collections::BTreeMap<u32, std::collections::BTreeMap<u32, usize>> = Default::default();
        let mut nodes: std::collections::BTreeMap<(String, u8), usize> = Default::default();
        let mut seats: std::collections::BTreeMap<(String, Option<u8>), usize> = Default::default();
        let mut pairs: std::collections::BTreeMap<(String, String), usize> = Default::default();
        let mut base_tick: Option<u32> = None;
        for (i, rec) in f.data.records.iter().enumerate() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
            let b = *base_tick.get_or_insert(t.tick.wrapping_sub(i as u32));
            let tick = t.tick.wrapping_sub(b);
            for e in &t.entities {
                if e.dirty_mask & (1 << 12) == 0 {
                    continue;
                }
                with12 += 1;
                let (mask, a, stop) = attach_of(e, &idx.params);
                // A walk that gave up at a field PAST 12 still read 12 correctly; only one that
                // never reached it is a miss.
                if a.is_none() {
                    stopped += 1;
                    continue;
                }
                // The free half of the oracle: the walk must rebuild the message's own mask over
                // the base fields it covers. Only checkable on a walk that ran to the end.
                if stop.is_none() && mask != e.dirty_mask & 0xFFFFF {
                    maskbad += 1;
                }
                let Some(a) = a else { continue };
                let child = who(e.net_id);
                *seats.entry((child.clone(), a.seat)).or_default() += 1;
                match a.parent {
                    None => cleared += 1,
                    Some(par) => {
                        if let Some(n) = a.node {
                            *nodes.entry((child.clone(), n)).or_default() += 1;
                        }
                        *per_child.entry(e.net_id).or_default().entry(par).or_default() += 1;
                        if let Some(&ti) = lookup.get(&par) {
                            named += 1;
                            if idx.tracks[ti].alive_at(tick) {
                                alive += 1;
                            }
                            *pairs.entry((child, who(par))).or_default() += 1;
                        }
                    }
                }
            }
        }
        println!(
            "{}: {with12} message(s) carry base field 12; {stopped} the walk could not reach, \
             {maskbad} did not rebuild their own base mask",
            name(&p)
        );
        println!(
            "of the {} decoded records: {cleared} state NO parent, {} state one -- \
             {named} ({:.2}%) name a net id THIS FILM CREATED, {alive} ({:.2}%) one that is ALIVE at that tick",
            with12 - stopped,
            with12 - stopped - cleared,
            100.0 * named as f64 / (with12 - stopped - cleared).max(1) as f64,
            100.0 * alive as f64 / (with12 - stopped - cleared).max(1) as f64,
        );
        println!("\nchild definition -> parent definition (messages):");
        for ((c, pa), n) in &pairs {
            println!("  {c:<28} -> {pa:<28} {n:>7}");
        }
        println!("\nthe 8-bit attach point, by child definition:");
        for ((c, nd), n) in &nodes {
            println!("  {c:<28} node {nd:>3}  {n:>7}");
        }
        println!("\nthe 6-bit seat index, by child definition:");
        for ((c, s), n) in &seats {
            println!("  {c:<28} seat {:>5}  {n:>7}", s.map(|v| v as i32).unwrap_or(-1));
        }
        println!("\nhow many DISTINCT parents one child ever names (a stable parent is one):");
        let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
        for (_, m) in &per_child {
            *hist.entry(m.len()).or_default() += 1;
        }
        for (k, v) in &hist {
            println!("  {k} parent(s): {v} child object(s)");
        }
        // And the geometry oracle: for every (child, parent) pair the stream states, how far
        // apart are their RECORDED positions? A genuinely attached pair that both report should
        // hold a near-constant distance.
        println!("\nchild/parent recorded-position distance, where BOTH report a position:");
        for (cnet, m) in per_child.iter().take(40) {
            let Some(&ci) = lookup.get(cnet) else { continue };
            let Some((&pnet, _)) = m.iter().max_by_key(|(_, &n)| n) else { continue };
            let Some(&pi) = lookup.get(&pnet) else { continue };
            let (c, pa) = (&idx.tracks[ci], &idx.tracks[pi]);
            let mut ds: Vec<f32> = Vec::new();
            for &(tk, cp) in &c.positions {
                if let Some(pp) = pa.sample(tk) {
                    ds.push(
                        ((cp[0] - pp[0]).powi(2) + (cp[1] - pp[1]).powi(2) + (cp[2] - pp[2]).powi(2)).sqrt(),
                    );
                }
            }
            if ds.is_empty() {
                println!(
                    "  {cnet:#010x} {:<24} -> {pnet:#010x} {:<24} the child never reports a position",
                    who(*cnet),
                    who(pnet)
                );
                continue;
            }
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  {cnet:#010x} {:<24} -> {pnet:#010x} {:<24} n={} min {:.3} median {:.3} max {:.3} wu",
                who(*cnet),
                who(pnet),
                ds.len(),
                ds[0],
                ds[ds.len() / 2],
                ds[ds.len() - 1]
            );
        }
    }

    /// Percentiles of a sample set, for the cadence and jitter tables.
    fn pct(v: &mut Vec<f64>) -> (f64, f64, f64, f64) {
        if v.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
        (at(0.5), at(0.9), at(0.99), v[v.len() - 1])
    }

    /// #film-jitter  **How often does a vehicle report, and how much of the roughness in its
    /// drawn motion is the TICK QUANTISATION rather than the recording?**
    ///
    /// The measurement is a simulated render. For a display rate `R` and a playback speed `S`,
    /// frame `k` falls at playback tick `t = k * (60/R) * S`, and the question is how far the
    /// drawn point moves between consecutive frames. A smooth pass has an almost constant
    /// per-frame step; judder shows up as the SECOND difference
    /// `|p[k+1] - 2p[k] + p[k-1]|` — zero for constant velocity, large wherever the drawn point
    /// stalls for a frame and then catches up.
    ///
    /// Two samplers are compared on the same frames and the same data:
    ///
    /// * **round** — what every Theater placement did: `EntityTrack::sample(t.round() as u32)`.
    /// * **frac** — [`position_at`], the same lerp with the tick's own fraction kept.
    ///
    /// `repeat` is how many frames drew the EXACT same point as the frame before, which is the
    /// stall itself rather than its consequence.
    #[test]
    #[ignore]
    fn probe_vehicle_cadence_and_jitter() {
        let films: Vec<std::path::PathBuf> = match std::env::var("HMS_FILM") {
            Ok(p) => vec![std::path::PathBuf::from(p)],
            Err(_) => film::find_finished_films(),
        };
        // (display Hz, playback speed)
        let cases = [(60.0f32, 1.0f32), (144.0, 1.0), (60.0, 0.25), (144.0, 0.25)];
        println!(
            "{:<40} {:>5} {:>7} {:>7} {:>7}   position-key gaps (ticks)",
            "film", "veh", "keys", "p50", "p99"
        );
        // Everything accumulated over the corpus, per case.
        let mut jit_round: Vec<Vec<f64>> = vec![Vec::new(); cases.len()];
        let mut jit_frac: Vec<Vec<f64>> = vec![Vec::new(); cases.len()];
        let mut rep_round = vec![0usize; cases.len()];
        let mut rep_frac = vec![0usize; cases.len()];
        let mut frames = vec![0usize; cases.len()];
        let mut gaps_all: Vec<f64> = Vec::new();
        let mut ogaps_all: Vec<f64> = Vec::new();
        let mut fgaps_all: Vec<f64> = Vec::new();
        let mut worst: Option<(String, u32, f64)> = None;
        for p in films.iter() {
            let Some((_f, idx)) = indexed(p) else { continue };
            let vs: Vec<&film_decode::EntityTrack> = idx
                .tracks
                .iter()
                .filter(|t| t.entity_type == crate::film_anim::VEHICLE_TYPE && t.positions.len() >= 100)
                .collect();
            // The biped lane's own cadence, for the same reason.
            for t in idx.tracks.iter().filter(|t| t.entity_type == crate::film_play::BIPED_TYPE) {
                for w in t.facings.windows(2) {
                    fgaps_all.push(w[1].0.saturating_sub(w[0].0) as f64);
                }
            }
            if vs.is_empty() {
                continue;
            }
            let mut gaps: Vec<f64> = Vec::new();
            for t in &vs {
                for w in t.positions.windows(2) {
                    gaps.push(w[1].0.saturating_sub(w[0].0) as f64);
                }
                for w in t.orientations.windows(2) {
                    ogaps_all.push(w[1].0.saturating_sub(w[0].0) as f64);
                }
            }
            gaps_all.extend(gaps.iter().copied());
            let (g50, _g90, g99, _gmax) = pct(&mut gaps.clone());
            println!(
                "{:<40} {:>5} {:>7} {:>7.1} {:>7.1}",
                name(p),
                vs.len(),
                gaps.len() + vs.len(),
                g50,
                g99
            );
            for (ci, &(hz, speed)) in cases.iter().enumerate() {
                let step = (60.0 / hz) * speed;
                for t in &vs {
                    let first = t.positions.first().map(|&(k, _)| k).unwrap_or(0) as f32;
                    let last = t.positions.last().map(|&(k, _)| k).unwrap_or(0) as f32;
                    if last <= first + 1.0 {
                        continue;
                    }
                    let n = ((last - first) / step) as usize;
                    if n < 4 {
                        continue;
                    }
                    let mut pr: Vec<Vec3> = Vec::with_capacity(n);
                    let mut pf: Vec<Vec3> = Vec::with_capacity(n);
                    for k in 0..n {
                        let tk = first + k as f32 * step;
                        pr.push(t.sample(tk.round().max(0.0) as u32).map(Vec3::from).unwrap_or(Vec3::ZERO));
                        pf.push(position_at(t, tk).unwrap_or(Vec3::ZERO));
                    }
                    frames[ci] += n;
                    for k in 1..n - 1 {
                        let a = (pr[k + 1] - 2.0 * pr[k] + pr[k - 1]).length() as f64;
                        let b = (pf[k + 1] - 2.0 * pf[k] + pf[k - 1]).length() as f64;
                        jit_round[ci].push(a);
                        jit_frac[ci].push(b);
                        if ci == 1 && worst.as_ref().map_or(true, |w| a > w.2) {
                            worst = Some((name(p), t.net_id, a));
                        }
                    }
                    for k in 1..n {
                        if pr[k] == pr[k - 1] {
                            rep_round[ci] += 1;
                        }
                        if pf[k] == pf[k - 1] {
                            rep_frac[ci] += 1;
                        }
                    }
                }
            }
        }
        let (p50, p90, p99, pmax) = pct(&mut gaps_all);
        println!(
            "\nPOSITION keys: {} gap(s), p50 {p50:.0} p90 {p90:.0} p99 {p99:.0} max {pmax:.0} ticks",
            gaps_all.len()
        );
        let (o50, o90, o99, omax) = pct(&mut ogaps_all);
        println!("ORIENTATION keys: {} gap(s), p50 {o50:.0} p90 {o90:.0} p99 {o99:.0} max {omax:.0} ticks", ogaps_all.len());
        let (f50, f90, f99, fmax) = pct(&mut fgaps_all);
        println!("BIPED FACING keys: {} gap(s), p50 {f50:.0} p90 {f90:.0} p99 {f99:.0} max {fmax:.0} ticks", fgaps_all.len());
        println!(
            "\n{:<16} {:>9} {:>34} {:>34}",
            "case", "frames", "ROUNDED tick (today)", "FRACTIONAL tick"
        );
        println!(
            "{:<16} {:>9} {:>9} {:>8} {:>8} {:>7} {:>9} {:>8} {:>8} {:>7}",
            "", "", "p50", "p99", "max", "repeat", "p50", "p99", "max", "repeat"
        );
        for (ci, &(hz, speed)) in cases.iter().enumerate() {
            let (a50, _, a99, amax) = pct(&mut jit_round[ci]);
            let (b50, _, b99, bmax) = pct(&mut jit_frac[ci]);
            println!(
                "{:<16} {:>9} {:>9.4} {:>8.4} {:>8.4} {:>6.1}% {:>9.4} {:>8.4} {:>8.4} {:>6.1}%",
                format!("{hz:.0} Hz @ {speed}x"),
                frames[ci],
                a50,
                a99,
                amax,
                100.0 * rep_round[ci] as f64 / frames[ci].max(1) as f64,
                b50,
                b99,
                bmax,
                100.0 * rep_frac[ci] as f64 / frames[ci].max(1) as f64,
            );
        }
        println!(
            "\n(the numbers are the per-frame SECOND DIFFERENCE of the drawn position, in world units: \
             0 = constant velocity, larger = a stall followed by a catch-up. `repeat` = frames that \
             drew the identical point to the previous frame.)"
        );
        if let Some((f, net, d)) = worst {
            println!("worst single frame at 144 Hz @ 1x: {d:.4} wu on entity {net:#010x} of {f}");
        }
    }

    /// #film-jitter  **How many recorded BODIES never move, and so were never drawn?**
    ///
    /// `Playback::wanted` applied the `moving_only` filter to bipeds, so a body whose whole life
    /// moves it less than `moves()`'s half a world unit was dropped from the frame — while the
    /// weapon it was holding stayed on screen as its own `weap` track. This counts them over the
    /// whole corpus, which is the honest measure of how much that cost.
    #[test]
    #[ignore]
    fn probe_motionless_bodies() {
        let films: Vec<std::path::PathBuf> = match std::env::var("HMS_FILM") {
            Ok(p) => vec![std::path::PathBuf::from(p)],
            Err(_) => film::find_finished_films(),
        };
        let (mut tot, mut still, mut films_used) = (0usize, 0usize, 0usize);
        let mut worst: Option<(String, usize, usize)> = None;
        println!("{:<44} {:>8} {:>10} {:>8}", "film", "bodies", "motionless", "share");
        for p in films.iter() {
            let Some((_f, idx)) = indexed(p) else { continue };
            let bodies: Vec<&film_decode::EntityTrack> =
                idx.tracks.iter().filter(|t| t.entity_type == crate::film_play::BIPED_TYPE).collect();
            if bodies.is_empty() {
                continue;
            }
            films_used += 1;
            let n = bodies.len();
            let s = bodies.iter().filter(|t| !t.moves()).count();
            tot += n;
            still += s;
            if worst.as_ref().map_or(true, |w| s * w.1 > w.2 * n) {
                worst = Some((name(p), n, s));
            }
            println!("{:<44} {n:>8} {s:>10} {:>7.1}%", name(p), 100.0 * s as f64 / n as f64);
        }
        println!(
            "\n{films_used} film(s): {still} of {tot} recorded bodies ({:.1}%) never move more than \
             half a world unit in their whole life, and were therefore drawn NEITHER alive NOR as a corpse",
            100.0 * still as f64 / tot.max(1) as f64
        );
        if let Some((f, n, s)) = worst {
            println!("worst clip: {s} of {n} ({:.1}%) on {f}", 100.0 * s as f64 / n as f64);
        }
    }

    /// #film-attach  **WHAT does a mounted turret actually replicate, message by message?**
    ///
    /// The user's report is "the turrets also get velocity data as well though they don't move at
    /// all". A base-class object's VELOCITY is field 1 and its position is field 0; this counts,
    /// per definition, how many of its messages set each dirty bit, so the question is answered
    /// with a histogram rather than with a union.
    ///
    /// The bits worth naming, from the engine:
    ///
    /// | bit | field | what |
    /// |---|---|---|
    /// | 0 | base 0 | position (`compressed_vector3`) |
    /// | 1 | base 1 | translational VELOCITY (19-bit direction + 10-bit log magnitude) |
    /// | 2 | base 2 | orientation (19-bit up axis + 8-bit angle) |
    /// | 3 | base 3 | angular velocity |
    /// | 12 | base 12 | **the attachment record** (`#film-attach`) |
    /// | 20..24 | vehicle derived 0x14..0x18 | `sub_180460334` |
    /// | **21** | **vehicle 0x15** | **a 19-bit unit VECTOR** -- the engine validates it with
    ///   `abs(x*x + y*y + z*z - 1) < 0.001`, so it is a DIRECTION, and on a mounted turret it is
    ///   the only thing that ever changes |
    /// | 23 | vehicle 0x17 | `sub_1803C94B8`, the same selection reader a unit uses for its weapon |
    #[test]
    #[ignore]
    fn probe_dirty_bit_histogram() {
        let Some(p) = probe_film() else {
            eprintln!("no film");
            return;
        };
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("no forge_halo");
            return;
        };
        let f = film::open(&p).expect("film opens");
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        let names = def_names(&native, cache, &idx);
        let key_of: std::collections::HashMap<u32, String> = idx
            .tracks
            .iter()
            .map(|t| {
                let n = names
                    .get(&t.net_id)
                    .map(|(_, g, l)| format!("{} {g} {l}", t.entity_type))
                    .unwrap_or_else(|| format!("{} ????", t.entity_type));
                (t.net_id, n)
            })
            .collect();
        let mut hist: std::collections::BTreeMap<String, (usize, [usize; 32])> = Default::default();
        for rec in f.data.records.iter() {
            let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &idx.params) else { continue };
            for e in &t.entities {
                let Some(k) = key_of.get(&e.net_id) else { continue };
                let row = hist.entry(k.clone()).or_insert((0, [0; 32]));
                row.0 += 1;
                for b in 0..32 {
                    if e.dirty_mask & (1 << b) != 0 {
                        row.1[b] += 1;
                    }
                }
            }
        }
        println!("per definition, the share of its messages that set each dirty bit (blank = never):");
        print!("{:<34} {:>8}", "definition", "msgs");
        for b in [0u32, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 20, 21, 22, 23, 24] {
            print!(" {b:>5}");
        }
        println!();
        for (k, (n, bits)) in &hist {
            if *n < 50 {
                continue;
            }
            print!("{k:<34} {n:>8}");
            for b in [0usize, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 20, 21, 22, 23, 24] {
                if bits[b] == 0 {
                    print!("     .");
                } else {
                    print!(" {:>4.0}%", 100.0 * bits[b] as f64 / *n as f64);
                }
            }
            println!();
        }
    }
}
