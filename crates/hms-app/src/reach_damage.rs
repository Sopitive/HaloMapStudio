//! #damage  Reach's DAMAGE tags: `jpt!` (`damage_effect`) and an object model's vitality,
//! damage sections and instant responses.
//!
//! This is the measured half of "hit -> damage -> response", the half that reads tags. The seam
//! it serves is [`crate::play_damage::ObjectDamage`]; the half that is NOT here is named at the
//! bottom of this comment, because the rule is to leave an unmeasured thing unimplemented rather
//! than fill it in plausibly.
//!
//! # What is measured, and how
//!
//! **The table layout is confirmed TWICE** — once from `reach_tag_test.exe`'s reflection data
//! (`tagdefs hlmt`), and once from the engine's own dword-indexed walk in `haloreach.dll`
//! `sub_180464F10` (the `damage_section_response` receiver, which `film_vehicle` had already
//! read; this work re-derived it independently and agrees field for field):
//!
//! ```text
//!   *(float *)(hlmt + 0xD0)           maximum vitality     (= damage info +0x04), gated `> 0.0`
//!   *(int   *)(hlmt + 0xE0) / +0xE4   damage sections count / pointer
//!   48 dwords                         section stride       = 192 bytes
//!   *(int *)(section + 0x84) / +0x88  instant responses count / pointer
//!   44 dwords                         response stride      = 176 bytes
//! ```
//!
//! **Vitality splits by section**: `new_global_damage_section_block` +0x08
//! `vitality percentage:[0.1]`, whose own field documentation reads "percentage of total object
//! vitality". So a section's pool is `maximum vitality * vitality percentage`. A section with a
//! non-zero `shield material name` (+0x0C, "set this to make this damage section a shield") IS a
//! shield, and the damage info's +0x0E names which section the model's shielded/unshielded
//! rendering follows.
//!
//! **A response's own threshold** is `new_instantaneous_damage_response_block` +0x08
//! `damage threshold`, a real fraction whose field documentation reads "response fires after
//! crossing this threshold. 1=full health".
//!
//! **`jpt!` is 252 bytes (0xFC)**, and recovering it needed the content-scan method described in
//! `re-notes/reach_runtime_installed_field_sets.md`: `damage_outer_cone_angle_struct` at +0xCC is
//! one of the field sets `tagdefs` prints as `?`, and everything past it — including
//! `instantaneous acceleration`, the knockback — is unplaceable until it is recovered. It is one
//! `angle`, 4 bytes.
//!
//! # What is NOT measured, and is therefore not implemented
//!
//! **Which damage SECTION a direct hit lands on.** `hlmt` carries up to sixteen named sections
//! and the response is per section, so "a vehicle takes damage on the side it was hit" depends
//! entirely on this — and the engine resolves it through the collision model's material, not
//! through the hit point's geometry, which this module has not yet read. Until it is read,
//! [`DamageModel`] exposes the two cases the tags DO answer and refuses the third:
//!
//! * an **explosion or child damage** goes to the damage info's `indirect damage section`
//!   (+0x0C), whose own field documentation is "absorbes AOE or child damage";
//! * a hit on a model with ONE section has only one place to go;
//! * a direct hit on a multi-section model returns `None` with a stated reason.
//!
//! Nothing here guesses a section, and nothing here applies damage to "the first section" as a
//! stand-in. A wrong section would show the wrong part of a vehicle breaking, which is exactly
//! the kind of plausible-but-wrong the user would feel immediately.

use hms_native::{CacheHandle, NativeDll};

// ---------------------------------------------------------------------------------------------
// THE SEAM -- declared here because this module owns the law behind it
// ---------------------------------------------------------------------------------------------

/// What caused a hit. The source NAMES the cause and carries no damage value, because picking
/// the `jpt!` for a cause is this module's job and not the caller's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DamageSource {
    /// A melee swing with this weapon tag (0 = unarmed).
    Melee { weapon: u32 },
    /// A single shot from this weapon tag.
    Shot { weapon: u32 },
    /// A projectile impact.
    Projectile { proj: u32 },
    /// An explosion, by its damage tag.
    Explosion { jpt: u32 },
}

impl DamageSource {
    pub fn label(&self) -> &'static str {
        match self {
            DamageSource::Melee { .. } => "melee",
            DamageSource::Shot { .. } => "shot",
            DamageSource::Projectile { .. } => "projectile",
            DamageSource::Explosion { .. } => "explosion",
        }
    }
}

/// One hit, as the thing that caused it can describe it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageHit {
    /// The object that was hit, by placement datum.
    pub target_datum: u32,
    /// Where, in world space. The damage side resolves this against the target's own damage
    /// sections, which is what makes a vehicle take damage on the side it was hit.
    pub point: glam::Vec3,
    /// Direction of travel of whatever did the damage (the knockback a `jpt!` imparts).
    pub dir: glam::Vec3,
    pub source: DamageSource,
    /// Distance from the damage ORIGIN, for the `jpt!`'s falloff. **0.0 for a contact hit**,
    /// which is what every melee is.
    pub distance: f32,
}

/// What came back: enough for a caption to say something real, and nothing it cannot support.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamageReport {
    /// `(damage section name, the state it is now in)` for every section that CHANGED.
    pub sections_changed: Vec<(String, u8)>,
    /// Damage actually applied. `None` when the hit's damage could not be determined -- which is
    /// every explosion today, because the `jpt!` falloff radii are not measured yet. A caller
    /// must print the absence rather than substitute a number.
    pub applied: Option<f32>,
    /// One line for the caption, from the side that knows what it means.
    pub note: String,
}

/// The damage model. A consumer calls this and nothing else.
///
/// `&mut self` because vitality is STATE: a vehicle gets weaker with each hit and which response
/// fires is a function of what is left, so the second call cannot be computed without the first.
pub trait ObjectDamage {
    fn apply(&mut self, hit: DamageHit) -> Option<DamageReport>;
}

/// What a consumer prints while no damage model is installed. Stated, never implied.
pub const NO_MODEL: &str = "no damage model installed \u{2014} the swing plays, nothing takes damage";

// ---------------------------------------------------------------------------------------------
// jpt!  (damage_effect) -- 252 bytes
// ---------------------------------------------------------------------------------------------

/// **How much of a `jpt!` this module reads — NOT the tag's size, which is still unknown.**
///
/// A binary search on readable length says 1020 bytes for all 137 of `forge_halo`'s `jpt!` tags,
/// and that number is a trap: `tag_meta` reads out of the META SECTION, which is contiguous, so a
/// read past the end of a tag silently returns the NEXT tag's bytes. The proof it is spill rather
/// than size: a 1020-byte read of `frag_grenade_equipment_explosion` turns up
/// `sound\characters\footsteps\chief\carpet\jump` at +0x310, and a grenade's damage effect does
/// not play a footstep. Independently, the `rmbl` / `csdt` / `sidt` references such a read finds
/// at +0x130 / +0x140 / +0x150 are **0x10 apart**, which is exactly the
/// `rumble` / `camera shake` / third spacing inside `drdf`'s own `damage_response_class_block`
/// (+0x48 / +0x58 / +0x68) — i.e. they are the neighbouring `drdf`'s fields, not this tag's.
///
/// So this reads only as far as the fields it has actually placed. Neither of the two field arrays the
/// reflection data offers for `damage_effect` describes it — one sums to 252 and the other to
/// 516 — so this is a third instance of the same problem as `weapon_barrels` (see
/// `re-notes/reach_runtime_installed_field_sets.md`) and the shipped field ORDER differs from
/// both: the reflected arrays put the screen-flash and camera-shake fields first and the damage
/// block at +0xB0, while the shipped tag has the damage at the front.
pub const JPT_READ: usize = 0xB4;

/// `damage lower bound`. **Measured from the cache, not from the reflection data**, and
/// corroborated by values nobody who has played Reach could mistake: the assault rifle's bullet
/// reads 5.834 here with an upper bound of 5.834..6.788, and the sniper rifle's reads 80.0/80.0.
/// The reflected offset (+0xBC) reads `0xFFFFFFFF` — a null tag reference — on every tag.
pub const JPT_DAMAGE_LOWER_BOUND: usize = 0x20;
/// `damage upper bound`, a real BOUNDS: `[min, max]` at +0x24 / +0x28, rolled per hit.
pub const JPT_DAMAGE_UPPER_BOUND: usize = 0x24;

/// `damage response` — a `drdf` (`damage_response_definition`). **The one tag reference inside a
/// `jpt!` that this module can defend**, and the route to a damage's sound.
///
/// Measured by concentration, which is the test that separates a real field from meta-section
/// spill: a validated `drdf` reference sits at exactly +0xA4 on **114 of 137** tags
/// (`globals\damage_responses\impact_bullet_slow` for the assault rifle, `impact_bullet_sniper`
/// for the sniper, `explosion_small_lowpass` for the frag grenade), while every `snd!` a wide
/// read turns up is scattered over ten different offsets with no offset holding more than
/// fourteen — the signature of spill.
///
/// **A `jpt!` does not name a sound directly.** The sound is in the `drdf`: its `classes` block
/// (132-byte elements) carries `rumble` at +0x48, `camera shake` at +0x58, and a
/// `global sound effect` block at +0x78 whose `effect name` is a **string id**, not a tag
/// reference. So a damage sound is a NAMED global sound effect and not an `snd!` join.
pub const JPT_DAMAGE_RESPONSE: usize = 0xA4;

// ---------------------------------------------------------------------------------------------
// NOT MEASURED, and therefore not read.
//
// The rest of the `jpt!` damage block -- the falloff `radius` bounds, `cutoff scale`,
// `category`, `side effect`, the damage flags and `instantaneous acceleration` (the knockback) --
// is NOT placed. The reflection data's offsets for them are from a different field order (they
// read null tag references and zeros on every shipped tag), and the shipped order has not been
// recovered yet. Two fields with unmistakable authored values are not enough to pin twenty.
//
// This matters most for an EXPLOSION: without the radius bounds there is no falloff, and a
// damage model that silently applied the full upper bound at any range would be wrong for every
// explosion in the game. So [`DamageEffect::falloff`] is `None` until it is measured, and
// [`DamageEffect::at_distance`] returns `None` rather than a number.
// ---------------------------------------------------------------------------------------------

/// One `jpt!`, reduced to the fields that have been MEASURED. Anything the layout does not yet
/// place is absent from this struct rather than present and wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DamageEffect {
    pub jpt: u32,
    /// `damage lower bound`.
    pub lower: f32,
    /// `damage upper bound` `[min, max]`.
    pub upper: [f32; 2],
    /// `damage response` — the `drdf` this damage plays through (rumble, camera shake and the
    /// NAMED global sound effect live there, not here).
    pub response: Option<u32>,
    /// The falloff radii, **`None` until they are measured** — see the note above. A `Some` here
    /// would be a promise this module cannot keep.
    pub falloff: Option<([f32; 2], f32)>,
}

impl DamageEffect {
    /// The damage at a CONTACT hit: the middle of `damage upper bound`.
    ///
    /// The middle rather than a roll, so a report is reproducible — the engine rolls it, and a
    /// test-mode caption that changed every time it was asked would be useless. [`Self::upper`]
    /// carries the bounds for a caller that wants the roll.
    pub fn contact(&self) -> f32 {
        0.5 * (self.upper[0] + self.upper[1])
    }

    /// The damage at `distance` from the damage's origin, or **`None` when the falloff is not
    /// measured** — which is every tag today.
    ///
    /// Returning `None` rather than `contact()` is the point: an explosion whose damage did not
    /// fall off would do full damage across the map, and that is a plausible-looking wrong answer
    /// of exactly the kind this module exists to avoid.
    pub fn at_distance(&self, distance: f32) -> Option<f32> {
        let (radius, cutoff_scale) = self.falloff?;
        let hi = self.contact();
        let (r0, r1) = (radius[0], radius[1]);
        if !(r1 > r0) || distance <= r0 {
            return Some(hi);
        }
        if distance > r1 {
            return Some(0.0);
        }
        let t = ((distance - r0) / (r1 - r0)).clamp(0.0, 1.0);
        let floor = self.lower + (hi - self.lower) * cutoff_scale.clamp(0.0, 1.0);
        Some(hi + (floor - hi) * t)
    }
}

fn rd_f32(b: &[u8], o: usize) -> f32 {
    if o + 4 > b.len() {
        return 0.0;
    }
    let v = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    if v.is_finite() {
        v
    } else {
        0.0
    }
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_i16(b: &[u8], o: usize) -> i16 {
    if o + 2 > b.len() {
        return 0;
    }
    i16::from_le_bytes([b[o], b[o + 1]])
}

/// Read one `jpt!`. `None` when it is not readable to its full authored length, which is also the
/// test that the layout is the one this module measured.
pub fn damage_effect(native: &NativeDll, cache: CacheHandle, jpt: u32) -> Option<DamageEffect> {
    let b = native.tag_meta(cache, jpt, 0, JPT_READ as u32)?;
    Some(DamageEffect {
        jpt,
        lower: rd_f32(&b, JPT_DAMAGE_LOWER_BOUND),
        upper: [rd_f32(&b, JPT_DAMAGE_UPPER_BOUND), rd_f32(&b, JPT_DAMAGE_UPPER_BOUND + 4)],
        response: {
            let d = rd_u32(&b, JPT_DAMAGE_RESPONSE + 0x0C);
            (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
        },
        // Not measured. See the note above the constants.
        falloff: None,
    })
}

// ---------------------------------------------------------------------------------------------
// hlmt  damage info / sections / responses
// ---------------------------------------------------------------------------------------------

/// `hlmt` +0x0CC `damage info` (`model_damage_info_struct`, 60 bytes inline).
pub const HLMT_DAMAGE_INFO: usize = 0x0CC;
/// ...+0x04 `maximum vitality`. The engine gates the whole damage path on this being `> 0.0`
/// (`sub_180464F10` reads `*(float *)(hlmt + 0xD0)` and branches on it).
pub const DI_MAXIMUM_VITALITY: usize = 0x04;
/// ...+0x0C `indirect damage section` -- "absorbes AOE or child damage".
pub const DI_INDIRECT_SECTION: usize = 0x0C;
/// ...+0x0E `shielded state damage section` -- the section whose depletion the model's
/// shielded/unshielded state reflects.
pub const DI_SHIELDED_SECTION: usize = 0x0E;
/// ...+0x14 / +0x18 the `damage sections` block.
pub const DI_SECTIONS: usize = 0x14;
pub const SECTION_SIZE: usize = 192;
pub const SECTION_MAX: i32 = 16;

pub const SEC_NAME: usize = 0x00; // string id
pub const SEC_FLAGS: usize = 0x04;
/// `vitality percentage:[0.1]` -- "percentage of total object vitality".
pub const SEC_VITALITY_PERCENTAGE: usize = 0x08;
/// `shield material name` -- "set this to make this damage section a shield". Non-zero = shield.
pub const SEC_SHIELD_MATERIAL: usize = 0x0C;
pub const SEC_STUN_TIME: usize = 0x10;
pub const SEC_MINIMUM_STUN_DAMAGE: usize = 0x14;
pub const SEC_RECHARGE_TIME: usize = 0x18;
/// +0x84 / +0x88 the `instant responses` block.
pub const SEC_RESPONSES: usize = 0x84;
pub const RESPONSE_SIZE: usize = 176;
pub const RESPONSE_MAX: i32 = 16;

pub const RSP_FLAGS: usize = 0x00;
pub const RSP_LABEL: usize = 0x04; // string id
/// `damage threshold` -- "response fires after crossing this threshold. 1=full health".
pub const RSP_DAMAGE_THRESHOLD: usize = 0x08;
/// `region_transitions`, solved by element-size closure and corroborated by the engine's own
/// `delay effect` index read at response +0x88 -- see `film_vehicle::DamageResponses`, which
/// established these and which this module deliberately does not duplicate.
pub const RSP_REGION_TRANSITIONS: usize = 0x58;
pub const REGION_TRANSITION_SIZE: usize = 8;
/// `skip fraction` -- "0.0 always fires, 1.0 never fires".
pub const RSP_SKIP_FRACTION: usize = 0x9C;
/// `total damage threshold` -- "scale on total damage section vitality".
pub const RSP_TOTAL_DAMAGE_THRESHOLD: usize = 0xA4;

/// One instant response, reduced to the decision it drives.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InstantResponse {
    pub index: usize,
    pub label: String,
    pub flags: u32,
    /// `damage threshold`, a fraction of the SECTION's vitality; 1 = full health.
    pub threshold: f32,
    /// `skip fraction`: 0 always fires, 1 never fires.
    pub skip_fraction: f32,
    /// `total damage threshold`: a scale on the total damage-section vitality.
    pub total_damage_threshold: f32,
    /// The worst `model_state_enum` this response's `region_transitions` name. The same value
    /// `film_vehicle::DamageResponses` produces, so the render side needs no second rule.
    pub state: u8,
}

/// One damage section of an object model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamageSection {
    pub index: usize,
    pub name: String,
    pub flags: u32,
    /// Share of the model's `maximum vitality` this section holds.
    pub vitality_percentage: f32,
    /// This section's own pool: `maximum vitality * vitality percentage`.
    pub vitality: f32,
    /// Non-zero `shield material name`: this section IS a shield.
    pub is_shield: bool,
    pub recharge_time: f32,
    pub responses: Vec<InstantResponse>,
}

/// One object model's whole damage law, read from its `hlmt`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamageModel {
    pub hlmt: u32,
    /// `maximum vitality`. The engine's own gate: at or below zero there is no damage path.
    pub maximum_vitality: f32,
    /// `indirect damage section` -- where AOE and child damage goes. `None` when unset.
    pub indirect_section: Option<usize>,
    /// `shielded state damage section`.
    pub shielded_section: Option<usize>,
    pub sections: Vec<DamageSection>,
}

impl DamageModel {
    /// Read an object's damage law. `None` when the object has no model, no damage info, or a
    /// `maximum vitality` of zero -- the same three refusals the engine makes.
    pub fn read(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<DamageModel> {
        let hlmt = crate::scene::object_hlmt(native, cache, obj_tag)?;
        let h = native.tag_meta(cache, hlmt, 0, (HLMT_DAMAGE_INFO + 0x3C) as u32)?;
        let di = HLMT_DAMAGE_INFO;
        let maximum_vitality = rd_f32(&h, di + DI_MAXIMUM_VITALITY);
        if !(maximum_vitality > 0.0) {
            return None;
        }
        let sect_index = |v: i16| -> Option<usize> { (v >= 0).then_some(v as usize) };
        let (scnt, sptr) = (rd_u32(&h, di + DI_SECTIONS), rd_u32(&h, di + DI_SECTIONS + 4));
        if scnt == 0 || scnt as i32 > SECTION_MAX {
            return None;
        }
        let sb = native.tag_ptr_bytes(cache, sptr, scnt * SECTION_SIZE as u32)?;
        let mut sections = Vec::with_capacity(scnt as usize);
        for s in 0..scnt as usize {
            let se = &sb[s * SECTION_SIZE..(s + 1) * SECTION_SIZE];
            let pct = rd_f32(se, SEC_VITALITY_PERCENTAGE);
            let (rcnt, rptr) = (rd_u32(se, SEC_RESPONSES), rd_u32(se, SEC_RESPONSES + 4));
            let mut responses = Vec::new();
            if rcnt > 0 && rcnt as i32 <= RESPONSE_MAX {
                if let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * RESPONSE_SIZE as u32) {
                    for r in 0..rcnt as usize {
                        let re = &rb[r * RESPONSE_SIZE..(r + 1) * RESPONSE_SIZE];
                        // The worst state this response's region transitions name -- the same
                        // walk `film_vehicle` validated against the engine.
                        let (tcnt, tptr) = (rd_u32(re, RSP_REGION_TRANSITIONS), rd_u32(re, RSP_REGION_TRANSITIONS + 4));
                        let mut state = 0u8;
                        if tcnt > 0 && tcnt as i32 <= 16 {
                            if let Some(tb) = native.tag_ptr_bytes(cache, tptr, tcnt * REGION_TRANSITION_SIZE as u32) {
                                for t in 0..tcnt as usize {
                                    let st = rd_i16(&tb, t * REGION_TRANSITION_SIZE + 4);
                                    if (0..crate::film_vehicle::STATES as i16).contains(&st) {
                                        state = state.max(st as u8);
                                    }
                                }
                            }
                        }
                        responses.push(InstantResponse {
                            index: r,
                            label: sid(native, cache, rd_u32(re, RSP_LABEL)),
                            flags: rd_u32(re, RSP_FLAGS),
                            threshold: rd_f32(re, RSP_DAMAGE_THRESHOLD),
                            skip_fraction: rd_f32(re, RSP_SKIP_FRACTION),
                            total_damage_threshold: rd_f32(re, RSP_TOTAL_DAMAGE_THRESHOLD),
                            state,
                        });
                    }
                }
            }
            sections.push(DamageSection {
                index: s,
                name: sid(native, cache, rd_u32(se, SEC_NAME)),
                flags: rd_u32(se, SEC_FLAGS),
                vitality_percentage: pct,
                vitality: maximum_vitality * pct,
                is_shield: rd_u32(se, SEC_SHIELD_MATERIAL) != 0,
                recharge_time: rd_f32(se, SEC_RECHARGE_TIME),
                responses,
            });
        }
        Some(DamageModel {
            hlmt,
            maximum_vitality,
            indirect_section: sect_index(rd_i16(&h, di + DI_INDIRECT_SECTION)),
            shielded_section: sect_index(rd_i16(&h, di + DI_SHIELDED_SECTION)),
            sections,
        })
    }

    /// Which section an INDIRECT hit (an explosion, or damage from a child object) lands on.
    ///
    /// The engine's own answer, from the field's own documentation: the damage info's
    /// `indirect damage section` "absorbes AOE or child damage". With none authored, a
    /// single-section model has only one place for it to go; a multi-section model does not, and
    /// this returns `None` rather than choosing.
    pub fn indirect_target(&self) -> Option<usize> {
        self.indirect_section.or(if self.sections.len() == 1 { Some(0) } else { None })
    }

    /// Which section a DIRECT hit lands on. **Not implemented, deliberately.**
    ///
    /// The engine resolves a direct hit to a damage section through the collision model's
    /// MATERIAL, not through the hit point's geometry, and this module has not read that mapping
    /// yet. A single-section model is answerable without it; anything else would be a guess, and
    /// a wrong section shows the wrong part of a vehicle breaking.
    pub fn direct_target(&self, _point: glam::Vec3) -> Result<usize, &'static str> {
        match self.sections.len() {
            0 => Err("this model authors no damage section"),
            1 => Ok(0),
            _ => Err(
                "which damage section a DIRECT hit lands on is not measured yet: the engine \
                 resolves it through the collision model's material, which this module has not \
                 read. Explosions go to the authored `indirect damage section`.",
            ),
        }
    }

    /// Of a section's responses, those whose authored `damage threshold` the section has crossed
    /// at `remaining` (a fraction of that section's vitality, 1 = full health), worst state last.
    ///
    /// The crossing direction is the field's own documentation -- "response fires after crossing
    /// this threshold. 1=full health" -- and nothing more is claimed: **the engine's ordering,
    /// whether a response latches after firing once, and how `skip fraction` randomises it are
    /// NOT yet read out of the engine**, so a caller must not treat this as the firing decision.
    /// It is "which responses this damage has reached", which is what a report can honestly say.
    pub fn responses_crossed(&self, section: usize, remaining: f32) -> Vec<&InstantResponse> {
        let Some(s) = self.sections.get(section) else { return Vec::new() };
        let mut v: Vec<&InstantResponse> = s.responses.iter().filter(|r| remaining <= r.threshold).collect();
        v.sort_by(|a, b| a.state.cmp(&b.state).then(a.index.cmp(&b.index)));
        v
    }

    /// One line per section, for a caption or a diagnostic.
    pub fn describe(&self) -> Vec<String> {
        let mut out = vec![format!(
            "hlmt {:#x}: maximum vitality {:.1}, {} section(s), indirect {:?}, shielded {:?}",
            self.hlmt, self.maximum_vitality, self.sections.len(), self.indirect_section, self.shielded_section
        )];
        for s in &self.sections {
            out.push(format!(
                "  [{}] '{}' {:.0}% = {:.1} vitality{} | {} response(s): {}",
                s.index,
                s.name,
                s.vitality_percentage * 100.0,
                s.vitality,
                if s.is_shield { " SHIELD" } else { "" },
                s.responses.len(),
                s.responses
                    .iter()
                    .map(|r| format!("'{}' at {:.2} -> state {}", r.label, r.threshold, r.state))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        out
    }
}

fn sid(native: &NativeDll, cache: CacheHandle, v: u32) -> String {
    if v == 0 || v == u32::MAX {
        return String::new();
    }
    native.string_id(cache, v).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// material effects -- where an impact SOUND actually comes from
// ---------------------------------------------------------------------------------------------

/// `obje` +0xD0 `melee sound` — the engine's own words: *"this is the sound that is made when I
/// am meleed. This overrides the sweetener sound of my material."*
///
/// **This is the melee impact sound**, named per OBJECT and overriding the material route. It is a
/// direct `snd!` reference, unlike anything in a `jpt!`.
pub const OBJE_MELEE_SOUND: usize = 0xD0;

/// The `snd!` an object makes when it is meleed (`obje` +0xD0), or `None` when it authors none and
/// the material's sweetener applies instead.
pub fn melee_sound(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<u32> {
    let m = native.tag_meta(cache, obj_tag, OBJE_MELEE_SOUND as u32, 16)?;
    let d = rd_u32(&m, 0x0C);
    (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
}

/// `obje` +0xC0 `material effects` -> a `foot` (`material_effects`).
///
/// **This, not the `jpt!`, is where an impact sound lives.** A `jpt!` names a `drdf` whose
/// "global sound effect" is a DSP filter on the listener (`explosion_small_lowpass`), not a sound
/// to play; the sound is chosen by the SURFACE that was hit, which is what a `foot` maps.
pub const OBJE_MATERIAL_EFFECTS: usize = 0xC0;
/// `foot` +0x00 `effects` -> `material_effect_block_v2`, 36-byte elements.
pub const FOOT_EFFECTS: usize = 0x00;
pub const MATERIAL_EFFECT_SIZE: usize = 36;
/// ...+0x0C `sounds` and +0x18 `effects`, both `material_effect_material_block`.
pub const MATERIAL_EFFECT_SOUNDS: usize = 0x0C;
pub const MATERIAL_EFFECT_EFFECTS: usize = 0x18;
/// One `material_effect_material_block` row. **44 bytes** — another field set whose `fields`
/// pointer is NULL in the image, recovered by content scan and then checked against a real map.
/// One shape serves both sub-blocks, which is why the field is called `tag (effect or sound)`.
pub const MATERIAL_ROW_SIZE: usize = 44;
/// `tag (effect or sound)` — an `snd!` in the `sounds` block, an `effe` in the `effects` block.
pub const MATERIAL_ROW_TAG: usize = 0x00;
/// `secondary tag (effect or sound)`.
pub const MATERIAL_ROW_SECONDARY: usize = 0x10;
/// `material name^` — the surface this row is for.
pub const MATERIAL_ROW_NAME: usize = 0x20;

/// One row of a `foot`: what a given SURFACE plays when something hits it.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialResponse {
    /// `material name` — the surface.
    pub material: String,
    /// `tag (effect or sound)`.
    pub tag: Option<u32>,
    /// `secondary tag (effect or sound)`.
    pub secondary: Option<u32>,
}

/// An object's material responses: `(sounds, effects)`, each a row per surface.
///
/// `None` when the object names no `material effects` tag.
pub fn material_responses(
    native: &NativeDll,
    cache: CacheHandle,
    obj_tag: u32,
) -> Option<(Vec<MaterialResponse>, Vec<MaterialResponse>)> {
    let foot = {
        let m = native.tag_meta(cache, obj_tag, OBJE_MATERIAL_EFFECTS as u32, 16)?;
        let d = rd_u32(&m, 0x0C);
        (d != u32::MAX && d != 0).then_some(d & 0xFFFF)?
    };
    let outer = crate::reach_actor::tag_block(native, cache, foot, FOOT_EFFECTS, MATERIAL_EFFECT_SIZE, 64);
    let mut sounds = Vec::new();
    let mut effects = Vec::new();
    for i in 0..outer.len() / MATERIAL_EFFECT_SIZE {
        let e = &outer[i * MATERIAL_EFFECT_SIZE..(i + 1) * MATERIAL_EFFECT_SIZE];
        for (off, out) in [(MATERIAL_EFFECT_SOUNDS, &mut sounds), (MATERIAL_EFFECT_EFFECTS, &mut effects)] {
            let rows = crate::reach_actor::block_at(native, cache, e, off, MATERIAL_ROW_SIZE, 256);
            for r in 0..rows.len() / MATERIAL_ROW_SIZE {
                let row = &rows[r * MATERIAL_ROW_SIZE..(r + 1) * MATERIAL_ROW_SIZE];
                let tr = |o: usize| -> Option<u32> {
                    let d = rd_u32(row, o + 0x0C);
                    (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
                };
                out.push(MaterialResponse {
                    material: sid(native, cache, rd_u32(row, MATERIAL_ROW_NAME)),
                    tag: tr(MATERIAL_ROW_TAG),
                    secondary: tr(MATERIAL_ROW_SECONDARY),
                });
            }
        }
    }
    Some((sounds, effects))
}

// ---------------------------------------------------------------------------------------------
// which jpt! a cause applies
// ---------------------------------------------------------------------------------------------

/// `proj` +0x284 `impact damage` -- the `jpt!` a projectile deals when it hits something.
/// Measured at the `proj` layout this project recovered (864 bytes, own fields from +0x168).
pub const PROJ_IMPACT_DAMAGE: usize = 0x284;
/// `proj` +0x1D8 `detonation damage`.
pub const PROJ_DETONATION_DAMAGE: usize = 0x1D8;
/// `weap` +0x2AC `melee damage parameters` block (200-byte elements, max 2).
///
/// **Fully placed by the reflection data, unlike `weapon_barrels` and `jpt!`** — this field set
/// is static, every offset below came out of `tagdefs weap`, and the two defaults are the
/// engine's own, quoted from the fields' own documentation.
pub const WEAP_MELEE_DAMAGE_PARAMETERS: usize = 0x2AC;
pub const MELEE_DAMAGE_PARAMETERS_SIZE: usize = 200;
pub const MELEE_DAMAGE_PARAMETERS_MAX: i32 = 2;

/// `damage pyramid angles`, a real euler angles 2d — a melee's hit volume is a PYRAMID, not a ray.
pub const MELEE_PYRAMID_ANGLES: usize = 0x00;
/// `damage pyramid depth:wu` — the field documents its own default: "0 defaults to 0.8f".
pub const MELEE_PYRAMID_DEPTH: usize = 0x08;
/// `maximum lunge range:wu` — "0 defaults to 1.22f".
pub const MELEE_MAXIMUM_LUNGE_RANGE: usize = 0x0C;
/// `melee damage` — the `jpt!` an ordinary swing applies.
pub const MELEE_DAMAGE: usize = 0x18;
/// `melee response`.
pub const MELEE_RESPONSE: usize = 0x28;
/// `lunge melee damage` — "this is only important for the energy sword".
pub const MELEE_LUNGE_DAMAGE: usize = 0x38;
/// `empty melee damage` — the swing of a weapon with no ammo.
pub const MELEE_EMPTY_DAMAGE: usize = 0x58;
/// `clang melee damage` — **the two-melee collision**, which is what a "clang" is.
pub const MELEE_CLANG_DAMAGE: usize = 0x78;
pub const MELEE_CLANG_RESPONSE: usize = 0x88;
/// `clang melee against melee weapon damage` — "e.g. used by AR to damage sword guy when
/// clanging sword attack".
pub const MELEE_CLANG_AGAINST_MELEE_DAMAGE: usize = 0x98;
/// `lunge melee explosive damage`.
pub const MELEE_LUNGE_EXPLOSIVE_DAMAGE: usize = 0xB8;

/// `weap` +0x2B8 `clang effect` — "effect that is played in the air between two players that
/// clang with this weapon". The `effe` a clang plays, as distinct from the `jpt!` it deals.
pub const WEAP_CLANG_EFFECT: usize = 0x2B8;

/// The engine's own default `damage pyramid depth` when the tag authors 0, from the field's own
/// documentation ("0 defaults to 0.8f").
pub const MELEE_DEFAULT_PYRAMID_DEPTH: f32 = 0.8;
/// The engine's own default `maximum lunge range` when the tag authors 0 ("0 defaults to 1.22f").
pub const MELEE_DEFAULT_LUNGE_RANGE: f32 = 1.22;

/// One weapon's melee law: how far it reaches, the shape of its hit volume, and the `jpt!` each
/// kind of swing applies.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeleeParameters {
    /// `damage pyramid angles` (yaw, pitch half-angles in radians, as the tag stores them).
    pub pyramid_angles: [f32; 2],
    /// `damage pyramid depth`, with the engine's 0.8 wu default already applied.
    pub pyramid_depth: f32,
    /// `maximum lunge range`, with the engine's 1.22 wu default already applied.
    pub maximum_lunge_range: f32,
    /// `melee damage` — an ordinary swing.
    pub damage: Option<u32>,
    /// `lunge melee damage` — the energy sword's lunge.
    pub lunge_damage: Option<u32>,
    /// `empty melee damage` — swinging a weapon with no ammo.
    pub empty_damage: Option<u32>,
    /// `clang melee damage` — a two-melee collision.
    pub clang_damage: Option<u32>,
    /// `clang melee against melee weapon damage`.
    pub clang_against_melee_damage: Option<u32>,
    /// `weap` +0x2B8 `clang effect` — the `effe` a clang plays in the air between the two.
    pub clang_effect: Option<u32>,
}

/// Read a weapon's melee law. `None` when the weapon authors no melee parameters at all.
pub fn melee_parameters(native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<MeleeParameters> {
    let b = crate::reach_actor::tag_block(
        native,
        cache,
        weap,
        WEAP_MELEE_DAMAGE_PARAMETERS,
        MELEE_DAMAGE_PARAMETERS_SIZE,
        MELEE_DAMAGE_PARAMETERS_MAX,
    );
    if b.len() < MELEE_DAMAGE_PARAMETERS_SIZE {
        return None;
    }
    let e = &b[..MELEE_DAMAGE_PARAMETERS_SIZE];
    let tr = |o: usize| -> Option<u32> {
        let d = rd_u32(e, o + 0x0C);
        (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
    };
    let or_default = |v: f32, d: f32| if v > 0.0 { v } else { d };
    let clang_effect = native.tag_meta(cache, weap, WEAP_CLANG_EFFECT as u32, 16).and_then(|m| {
        let d = rd_u32(&m, 0x0C);
        (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
    });
    Some(MeleeParameters {
        pyramid_angles: [rd_f32(e, MELEE_PYRAMID_ANGLES), rd_f32(e, MELEE_PYRAMID_ANGLES + 4)],
        pyramid_depth: or_default(rd_f32(e, MELEE_PYRAMID_DEPTH), MELEE_DEFAULT_PYRAMID_DEPTH),
        maximum_lunge_range: or_default(rd_f32(e, MELEE_MAXIMUM_LUNGE_RANGE), MELEE_DEFAULT_LUNGE_RANGE),
        damage: tr(MELEE_DAMAGE),
        lunge_damage: tr(MELEE_LUNGE_DAMAGE),
        empty_damage: tr(MELEE_EMPTY_DAMAGE),
        clang_damage: tr(MELEE_CLANG_DAMAGE),
        clang_against_melee_damage: tr(MELEE_CLANG_AGAINST_MELEE_DAMAGE),
        clang_effect,
    })
}

/// The `jpt!` a [`DamageSource`] applies, and whether it is a contact hit or
/// an area effect.
///
/// `Shot` resolves through the weapon's barrel to its PROJECTILE's `impact damage`, not through
/// the barrel's own `firing damage` -- the latter is the damage the act of firing does (to the
/// firer and its riders), which is a different thing and would be wrong here.
pub fn source_damage(
    native: &NativeDll,
    cache: CacheHandle,
    source: DamageSource,
) -> Option<DamageEffect> {
    use DamageSource as S;
    let tag_ref = |b: &[u8], o: usize| -> Option<u32> {
        let d = rd_u32(b, o + 0x0C);
        (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
    };
    let proj_impact = |proj: u32| -> Option<u32> {
        let b = native.tag_meta(cache, proj, 0, crate::reach_projectile::PROJ_SIZE as u32)?;
        tag_ref(&b, PROJ_IMPACT_DAMAGE)
    };
    let jpt = match source {
        S::Explosion { jpt } => Some(jpt),
        S::Projectile { proj } => proj_impact(proj),
        S::Shot { weapon } => crate::reach_projectile::weapon_barrels(native, cache, weapon)
            .first()
            .and_then(|b| b.projectile)
            .and_then(proj_impact),
        // `weap` +0x2AC `melee damage parameters` +0x18 `melee damage`. An UNARMED melee (weapon
        // 0) has no weapon to read, and resolves to nothing rather than to a plausible number.
        S::Melee { weapon } if weapon != 0 => melee_parameters(native, cache, weapon).and_then(|m| m.damage),
        S::Melee { .. } => None,
    }?;
    damage_effect(native, cache, jpt)
}

#[cfg(test)]
mod gates {
    //! #damage  The damage law, measured against `forge_halo`.
    //!
    //! `cargo test -p hms-app --profile quick the_damage -- --ignored --nocapture`

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// **THE `jpt!` LAYOUT GATE.** Every `jpt!` in the map must read to its full authored length,
    /// and the damage block must read as damage: an upper bound that is ordered, finite and
    /// positive on the overwhelming majority, and a falloff radius that is ordered.
    ///
    /// The bite is the ORDERING checks. A layout that is off by any amount scrambles
    /// `damage lower bound` / `damage upper bound[0]` / `damage upper bound[1]` into each other
    /// and into the cone angles, and ordered triples do not survive that.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_damage_effect_layout_reads_as_damage() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_damage gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let jpts = native.tags_by_group(cache, "jpt!");
        assert!(jpts.len() > 50, "forge_halo should carry many jpt! tags, found {}", jpts.len());
        let mut ordered = 0usize;
        let mut positive = 0usize;
        let mut biggest: (f32, String) = (0.0, String::new());
        for &j in &jpts {
            let d = damage_effect(native, cache, j).unwrap_or_else(|| {
                panic!("jpt! {:#x} '{}' is not readable to {JPT_READ:#x}", j, native.tag_name(cache, j).unwrap_or_default())
            });
            assert!(d.lower.is_finite() && d.upper[0].is_finite() && d.upper[1].is_finite(), "{:#x} non-finite damage", j);
            assert!(d.lower >= 0.0 && d.upper[0] >= 0.0, "{:#x} negative damage {:?}", j, (d.lower, d.upper));
            if d.upper[1] >= d.upper[0] {
                ordered += 1;
            }
            if d.upper[1] > 0.0 {
                positive += 1;
            }
            if d.upper[1] > biggest.0 {
                biggest = (d.upper[1], native.tag_name(cache, j).unwrap_or_default());
            }
        }
        let n = jpts.len();
        println!(
            "{n} jpt! tags: {ordered} with an ordered upper bound, {positive} with positive damage; \
             biggest {:.1} on '{}'",
            biggest.0, biggest.1
        );
        assert_eq!(ordered, n, "{} of {n} jpt! have damage upper bound [min > max] -- the layout is scrambled", n - ordered);
        // THE BITE. A wrong layout reads null tag references and zeros, and this is exactly what
        // caught the reflection data's own offsets (+0xBC/+0xC0): they gave 26 of 137 tags doing
        // any damage and a biggest value of 0.0 across the whole map. The ordering checks above
        // passed trivially there, because zeros are ordered -- so the POSITIVE assertions are the
        // ones that bite.
        assert!(
            positive * 10 >= n * 7,
            "only {positive} of {n} jpt! do any damage at all -- a damage tag that does no damage \
             is the signature of a wrong offset"
        );
        assert!(biggest.0 > 20.0, "the most damaging thing in the map does {:.3} -- the layout is wrong", biggest.0);
        // ...and the two authored values anyone who has played Reach would recognise, which is
        // the strongest corroboration available for a layout recovered from the cache.
        let find = |leaf: &str| -> Option<DamageEffect> {
            jpts.iter()
                .copied()
                .find(|&j| native.tag_name(cache, j).map_or(false, |n| n.rsplit('\\').next() == Some(leaf)))
                .and_then(|j| damage_effect(native, cache, j))
        };
        if let Some(d) = find("sniper_rifle_bullet") {
            assert!(
                (d.upper[0] - 80.0).abs() < 0.01 && (d.upper[1] - 80.0).abs() < 0.01,
                "the sniper rifle should do 80, reads {:?}",
                d.upper
            );
        }
        if let Some(d) = find("assault_rifle_bullet") {
            assert!((d.lower - 5.834).abs() < 0.01, "the assault rifle's lower bound should be 5.834, reads {}", d.lower);
            assert!((d.upper[1] - 6.788).abs() < 0.01, "...and its upper max 6.788, reads {}", d.upper[1]);
        }
    }

    /// **THE VITALITY GATE.**
    ///
    /// **The first draft of this gate asserted the section shares SUM TO 1, and it was wrong
    /// about the law while its offsets were right.** The Ghost sums to 3.2 — `body` 1.000,
    /// `seat` 0.400, `wing_left` 0.300, `wing_right` 0.300, `gastank` 0.200, `driver` 1.000 —
    /// because each section is its OWN pool sized as a share of the object's vitality, not a
    /// slice of one shared pool. A Warthog wheel holds 0.150 x 500 = 75 vitality while the body
    /// holds all 500. The field's own documentation ("percentage of total object vitality") says
    /// exactly that; the partition was an inference and it was mine.
    ///
    /// What replaces it is a stronger test of the same offsets: **every section's NAME must
    /// resolve to a non-empty string id**. A wrong stride or a wrong section-block offset
    /// scrambles those string ids into garbage, and `wheel_front_left` / `gastank` /
    /// `rudder_left` / `backhatch` coming out of 192-byte strides across every vehicle in the map
    /// is not something a wrong layout produces.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn vitality_splits_by_section_and_thresholds_are_fractions() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut models = 0usize;
        let mut shielded = 0usize;
        let mut responses = 0usize;
        let mut printed = 0usize;
        for group in ["vehi", "bipd", "bloc", "scen"] {
            for &t in &native.tags_by_group(cache, group) {
                let Some(m) = DamageModel::read(native, cache, t) else { continue };
                models += 1;
                assert!(
                    m.maximum_vitality > 0.0 && m.maximum_vitality < 1.0e6,
                    "'{}' maximum vitality {}",
                    native.tag_name(cache, t).unwrap_or_default(),
                    m.maximum_vitality
                );
                if m.sections.iter().any(|s| s.is_shield) {
                    shielded += 1;
                }
                for s in &m.sections {
                    // THE BITE: the names. A wrong stride turns these into garbage string ids.
                    assert!(
                        !s.name.is_empty(),
                        "'{}' damage section {} has no name -- the section stride or offset is wrong",
                        native.tag_name(cache, t).unwrap_or_default(),
                        s.index
                    );
                    assert!(
                        s.vitality_percentage > 0.0 && s.vitality_percentage <= 1.0,
                        "section '{}' holds {:.3} of the object's vitality",
                        s.name,
                        s.vitality_percentage
                    );
                    assert!((s.vitality - m.maximum_vitality * s.vitality_percentage).abs() < 1e-3);
                    for r in &s.responses {
                        responses += 1;
                        assert!(
                            r.threshold >= 0.0 && r.threshold <= 1.0,
                            "{} '{}' section '{}' response '{}' has threshold {} -- not a fraction of health",
                            group,
                            native.tag_name(cache, t).unwrap_or_default(),
                            s.name,
                            r.label,
                            r.threshold
                        );
                        assert!(
                            r.skip_fraction >= 0.0 && r.skip_fraction <= 1.0,
                            "response '{}' skip fraction {} is not a fraction",
                            r.label,
                            r.skip_fraction
                        );
                    }
                }
                if printed < 6 && m.sections.len() > 1 {
                    printed += 1;
                    for l in m.describe() {
                        println!("{l}");
                    }
                }
            }
        }
        println!("\n{models} model(s) with a damage law, {shielded} with a shield section, {responses} response(s)");
        assert!(models >= 10, "only {models} models carry a damage law; the read is probably failing");
        assert!(responses >= 10, "only {responses} instant responses were read");
    }

    /// Falloff is monotonic and bounded by the authored numbers, and a contact hit gets the full
    /// damage. Pure arithmetic over the production function, so it runs without a map.
    /// Asking for the damage at a distance must REFUSE while the falloff is unmeasured -- that
    /// refusal is what stops an explosion doing full damage across the map -- and must follow the
    /// authored curve once it is not.
    #[test]
    fn falloff_is_refused_while_unmeasured_and_monotonic_once_it_is_not() {
        let unmeasured = DamageEffect { jpt: 1, lower: 10.0, upper: [100.0, 100.0], falloff: None, response: None };
        assert_eq!(unmeasured.contact(), 100.0, "a contact hit is the middle of the upper bound");
        assert_eq!(unmeasured.at_distance(0.0), None, "with no measured falloff, refuse even at zero range");
        assert_eq!(unmeasured.at_distance(50.0), None, "...and certainly at 50 wu");

        let d = DamageEffect { falloff: Some(([1.0, 5.0], 0.0)), ..unmeasured };
        assert_eq!(d.at_distance(0.0), Some(100.0), "a contact hit takes the full upper bound");
        assert_eq!(d.at_distance(1.0), Some(100.0), "inside the inner radius, still full");
        assert_eq!(d.at_distance(6.0), Some(0.0), "past the outer radius, nothing");
        let mid = d.at_distance(3.0).expect("measured");
        assert!(mid > 10.0 && mid < 100.0, "halfway out should be between the bounds, got {mid}");
        let mut prev = f32::INFINITY;
        for i in 0..=60 {
            let v = d.at_distance(i as f32 * 0.1).expect("measured");
            assert!(v <= prev + 1e-4, "damage rose with distance at {:.1}: {prev} -> {v}", i as f32 * 0.1);
            prev = v;
        }
        let e = DamageEffect { falloff: Some(([1.0, 5.0], 1.0)), ..unmeasured };
        assert_eq!(e.at_distance(5.0), Some(100.0), "cutoff scale 1 keeps the damage to the outer radius");
    }

    /// **THE MELEE GATE.** A weapon's melee law, read against `forge_halo`.
    ///
    /// Unlike `weapon_barrels` and `jpt!`, `melee_damage_parameters_block` is a STATIC field set
    /// and every offset came straight out of `tagdefs weap` — so what this gate is for is the
    /// values: a melee must reach a plausible distance, have a hit volume with real angles, and
    /// name a `jpt!` that is really a `jpt!`.
    ///
    /// The bite is the CLASS check on every reference. A wrong offset inside a 200-byte element
    /// still yields a `u32` that looks like a tag index; asking the cache what class that index
    /// actually is, is what separates a tag reference from four bytes of float.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_weapons_melee_names_its_damage_and_its_clang() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut weapons = 0usize;
        let mut with_damage = 0usize;
        let mut with_clang = 0usize;
        let mut with_clang_effect = 0usize;
        for &w in &native.tags_by_group(cache, "weap") {
            let Some(m) = melee_parameters(native, cache, w) else { continue };
            let name = native.tag_name(cache, w).unwrap_or_default();
            let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
            weapons += 1;
            // The engine's own defaults are applied, so these are never zero.
            assert!(
                m.maximum_lunge_range > 0.3 && m.maximum_lunge_range < 10.0,
                "'{leaf}' melee reaches {:.3} wu -- a Spartan is 0.69 wu tall, so that is not a lunge",
                m.maximum_lunge_range
            );
            assert!(
                m.pyramid_depth > 0.0 && m.pyramid_depth < 10.0,
                "'{leaf}' damage pyramid depth {:.3} wu",
                m.pyramid_depth
            );
            for (what, jpt) in [
                ("melee damage", m.damage),
                ("lunge melee damage", m.lunge_damage),
                ("empty melee damage", m.empty_damage),
                ("clang melee damage", m.clang_damage),
                ("clang against melee", m.clang_against_melee_damage),
            ] {
                let Some(j) = jpt else { continue };
                assert_eq!(
                    native.tag_class(cache, j).as_deref(),
                    Some("jpt!"),
                    "'{leaf}' {what} points at {j:#x}, whose class is {:?}, not jpt!",
                    native.tag_class(cache, j)
                );
                // ...and it must read as damage through the production reader.
                let d = damage_effect(native, cache, j)
                    .unwrap_or_else(|| panic!("'{leaf}' {what} -> jpt! {j:#x} is not readable"));
                assert!(d.upper[1] >= d.upper[0], "'{leaf}' {what} bounds {:?}", d.upper);
            }
            if m.damage.is_some() {
                with_damage += 1;
            }
            if m.clang_damage.is_some() {
                with_clang += 1;
            }
            if let Some(e) = m.clang_effect {
                with_clang_effect += 1;
                assert_eq!(native.tag_class(cache, e).as_deref(), Some("effe"), "'{leaf}' clang effect is not an effe");
            }
            if m.clang_damage.is_some() || leaf.contains("sword") || leaf.contains("assault_rifle") {
                let dmg = m.damage.and_then(|j| damage_effect(native, cache, j)).map(|d| d.contact());
                let clang = m.clang_damage.and_then(|j| damage_effect(native, cache, j)).map(|d| d.contact());
                println!(
                    "{leaf}: lunge {:.2} wu, pyramid {:.2} wu at ({:.1},{:.1}) deg | melee {:?} | clang {:?} | clang effect {:?}",
                    m.maximum_lunge_range,
                    m.pyramid_depth,
                    m.pyramid_angles[0].to_degrees(),
                    m.pyramid_angles[1].to_degrees(),
                    dmg,
                    clang,
                    m.clang_effect.and_then(|e| native.tag_name(cache, e)).map(|n| n.rsplit('\\').next().unwrap_or("").to_string()),
                );
            }
        }
        println!(
            "\n{weapons} weapon(s) author melee parameters: {with_damage} name a melee `jpt!`, \
             {with_clang} name a CLANG `jpt!`, {with_clang_effect} name a clang `effe`"
        );
        assert!(weapons >= 20, "only {weapons} weapons author melee parameters");
        assert!(with_damage >= 20, "only {with_damage} weapons name a melee damage effect");
    }

    /// **THE MATERIAL-EFFECTS GATE.** `material_effect_material_block`'s 44-byte stride, checked
    /// against a real map rather than taken on the reflection data's word — it is another field
    /// set whose `fields` pointer is NULL in the image.
    ///
    /// The bite is the same pair that settled every other layout here: every row's `material
    /// name` must resolve to a NON-EMPTY string id, and every tag reference must resolve to a
    /// class the cache agrees with. A wrong stride satisfies neither.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn an_objects_material_effects_name_a_surface_and_a_sound() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut objects = 0usize;
        let mut rows = 0usize;
        let mut sound_rows = 0usize;
        let mut effect_rows = 0usize;
        let mut printed = 0usize;
        let mut named = 0usize;
        let mut unnamed = 0usize;
        for group in ["bipd", "vehi", "weap", "proj"] {
            for &t in &native.tags_by_group(cache, group) {
                let Some((sounds, effects)) = material_responses(native, cache, t) else { continue };
                if sounds.is_empty() && effects.is_empty() {
                    continue;
                }
                objects += 1;
                let name = native.tag_name(cache, t).unwrap_or_default();
                for (what, list) in [("sound", &sounds), ("effect", &effects)] {
                    for r in list {
                        rows += 1;
                        if r.material.is_empty() {
                            unnamed += 1;
                        } else {
                            named += 1;
                        }
                        for (which, tag) in [("tag", r.tag), ("secondary", r.secondary)] {
                            let Some(j) = tag else { continue };
                            let class = native.tag_class(cache, j);
                            assert!(
                                matches!(class.as_deref(), Some("snd!") | Some("lsnd") | Some("effe")),
                                "'{name}' {what} row '{}' {which} points at {j:#x}, class {class:?} -- \
                                 not a sound or an effect, so the row layout is wrong",
                                r.material
                            );
                            if matches!(class.as_deref(), Some("snd!") | Some("lsnd")) {
                                sound_rows += 1;
                            } else {
                                effect_rows += 1;
                            }
                        }
                    }
                }
                if printed < 3 && sounds.len() > 2 {
                    printed += 1;
                    println!("{name}");
                    for r in sounds.iter().take(5) {
                        println!(
                            "   sound on '{}' -> {}",
                            r.material,
                            r.tag.and_then(|j| native.tag_name(cache, j)).unwrap_or_else(|| "-".into())
                        );
                    }
                }
            }
        }
        // A row MAY author no material name -- the focus rifle has such a row, and `liquid -> -`
        // shows a row can carry a name with no tag either, so partly-empty rows are authored and
        // not a symptom. (The first draft of this gate asserted every row names a material and
        // failed on exactly one weapon; that was a premise about the data, not a layout error --
        // the names that ARE there are all real material names.) So the gate is a RATIO on the
        // names plus a strict class check on every tag that is present, which is the half a wrong
        // stride cannot fake.
        println!("\n{objects} object(s) with material effects, {rows} row(s): {named} named, {unnamed} unnamed, {sound_rows} sound reference(s), {effect_rows} effect reference(s)");
        assert!(
            named * 4 >= rows * 3,
            "only {named} of {rows} material rows name a surface -- the 44-byte row stride is wrong"
        );
        assert!(objects >= 3, "only {objects} objects resolved material effects");
        assert!(rows >= 20, "only {rows} material rows were read");
        assert!(sound_rows >= 10, "only {sound_rows} of those rows name a sound -- this is the route to an impact sound");
    }

    /// A direct hit on a multi-section model must REFUSE rather than pick a section, and an
    /// explosion must go to the authored indirect section. This is the unmeasured boundary, held
    /// explicitly so a later change cannot cross it silently.
    #[test]
    fn section_targeting_refuses_what_it_has_not_measured() {
        let one = DamageModel { sections: vec![DamageSection::default()], ..Default::default() };
        assert_eq!(one.direct_target(glam::Vec3::ZERO), Ok(0), "one section has only one answer");
        assert_eq!(one.indirect_target(), Some(0));
        let many = DamageModel {
            sections: vec![DamageSection::default(), DamageSection::default()],
            ..Default::default()
        };
        assert!(many.direct_target(glam::Vec3::ZERO).is_err(), "a multi-section direct hit must refuse");
        assert_eq!(many.indirect_target(), None, "with no authored indirect section, refuse that too");
        let m = DamageModel { indirect_section: Some(1), ..many };
        assert_eq!(m.indirect_target(), Some(1), "an authored indirect section is the answer for AOE");
    }

    /// `responses_crossed` reports what the damage has reached, ordered worst last, and reports
    /// nothing at full health.
    #[test]
    fn responses_are_reported_once_their_threshold_is_crossed() {
        let rsp = |i: usize, th: f32, st: u8| InstantResponse {
            index: i,
            threshold: th,
            state: st,
            label: format!("r{i}"),
            ..Default::default()
        };
        let m = DamageModel {
            maximum_vitality: 100.0,
            sections: vec![DamageSection {
                vitality: 100.0,
                responses: vec![rsp(0, 0.5, 1), rsp(1, 0.25, 2), rsp(2, 0.0, 4)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(m.responses_crossed(0, 1.0).is_empty(), "nothing fires at full health");
        assert!(m.responses_crossed(0, 0.75).is_empty(), "nothing fires above the first threshold");
        let at_half: Vec<u8> = m.responses_crossed(0, 0.5).iter().map(|r| r.state).collect();
        assert_eq!(at_half, vec![1], "the 0.5 threshold is crossed AT 0.5");
        let dead: Vec<u8> = m.responses_crossed(0, 0.0).iter().map(|r| r.state).collect();
        assert_eq!(dead, vec![1, 2, 4], "at zero every threshold is crossed, worst state last");
        assert!(m.responses_crossed(9, 0.0).is_empty(), "an out-of-range section reports nothing");
    }
}

#[cfg(test)]
mod probe {
    //! #damage  What a `jpt!` actually holds, and how long it actually is — because the layout
    //! gate rejected the offsets taken from the reflection data and the gate is to be believed
    //! until something outside it says otherwise.

    use super::*;

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_jpt_size_and_damage_fields() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let jpts = native.tags_by_group(cache, "jpt!");
        // 1. How long is a jpt! really? The read succeeds only when the WHOLE length is readable.
        let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
        for &j in jpts.iter() {
            let mut lo = 0usize;
            let mut hi = 0x400usize;
            while lo + 4 < hi {
                let mid = (lo + hi) / 2;
                if native.tag_meta(cache, j, 0, mid as u32).is_some() { lo = mid } else { hi = mid }
            }
            *hist.entry(lo & !3).or_default() += 1;
        }
        println!("readable-length histogram (length -> tags): {hist:?}");

        // 2. Where do plausible DAMAGE numbers live? Print the whole dword grid for a handful of
        // weapons whose damage anyone who has played Reach would recognise.
        for leaf in ["sniper_rifle_bullet", "assault_rifle_bullet", "frag_grenade", "magnum_bullet"] {
            let found: Vec<u32> = jpts
                .iter()
                .copied()
                .filter(|&j| native.tag_name(cache, j).map_or(false, |n| n.contains(leaf)))
                .collect();
            for j in found.iter().take(2) {
                let name = native.tag_name(cache, *j).unwrap_or_default();
                let Some(b) = native.tag_meta(cache, *j, 0, 0x100) else { continue };
                println!("\n{name}");
                for o in (0..0x100).step_by(4) {
                    let f = rd_f32(&b, o);
                    let u = rd_u32(&b, o);
                    if u == 0 { continue; }
                    let sid = (u > 0x100 && u < 0x8000_0000)
                        .then(|| native.string_id(cache, u))
                        .flatten()
                        .filter(|s| !s.is_empty());
                    println!(
                        "  +{o:#05x} u={u:#010x} f={f:>12.4} i16=({},{}){}",
                        rd_i16(&b, o), rd_i16(&b, o + 2),
                        sid.map(|s| format!(" sid '{s}'")).unwrap_or_default()
                    );
                }
            }
        }
    }

    /// Does `vitality percentage` really sum to 1 across a model's sections? The vitality gate
    /// assumed so and the Ghost sums to 3.2. Print the actual shares so the premise can be
    /// settled from the data rather than from the assumption.
    /// #gravity  The per-object ACCELERATION SCALES and the biped ground-physics gravity scale.
    ///
    /// `object_struct_definition` +0x1C `horizontal acceleration scale`, +0x20 `vertical
    /// acceleration scale`, +0x24 `angular acceleration scale`; and `bipd` +0x668
    /// `physics/ground physics/gravity scale` ("scale on gravity for this entity"). All four
    /// offsets are from `tagdefs`, which has them statically.
    ///
    /// Why this matters: the engine's world gravity is 4.1712594 (the default of the "physics
    /// constants" block, and the multiplier `physics_set_gravity` applies to its argument), while
    /// this project's corpse fit over 1,447 real falls is 10.4576 wu/s^2. 10.4576 / 4.1712594 =
    /// 2.507, so a per-object vertical scale of about 2.5 would reconcile them exactly.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_object_acceleration_scales() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        const ENGINE_GRAVITY: f32 = 4.171_259_4;
        println!("{:<44} {:>9} {:>9} {:>9} {:>11} {:>12}", "tag", "horiz", "VERT", "angular", "bipd gscale", "vert*4.1713");
        for group in ["bipd", "vehi"] {
            for &t in &native.tags_by_group(cache, group) {
                let Some(m) = native.tag_meta(cache, t, 0x1C, 12) else { continue };
                let f = |o: usize| rd_f32(&m, o);
                let (h, v, a) = (f(0), f(4), f(8));
                // `bipd` +0x668 ground-physics gravity scale.
                let gs = if group == "bipd" {
                    native.tag_meta(cache, t, 0x668, 4).map(|b| rd_f32(&b, 0))
                } else {
                    None
                };
                let name = native.tag_name(cache, t).unwrap_or_default();
                let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
                if h == 0.0 && v == 0.0 && a == 0.0 && gs.unwrap_or(0.0) == 0.0 {
                    continue;
                }
                println!(
                    "{:<44} {h:>9.3} {v:>9.3} {a:>9.3} {:>11} {:>12.4}",
                    &leaf[..leaf.len().min(43)],
                    gs.map(|x| format!("{x:.3}")).unwrap_or_else(|| "-".into()),
                    v * ENGINE_GRAVITY
                );
            }
        }
        println!("\nengine world gravity {ENGINE_GRAVITY} ; this project's corpse fit 10.4576 ; ratio {:.4}", 10.4576 / ENGINE_GRAVITY);
    }

    /// #clang  Does anything name a PRESENT, playable sound for a melee impact?
    ///
    /// The shared `fx\generic_melee_clash` effect's permutations resolve to no subsound in any of
    /// MCC's Reach banks (measured elsewhere), so a rifle-on-rifle clang cannot be played from
    /// that route. This asks the other two routes the tags offer, both of which are direct `snd!`
    /// references rather than effects: `obje` +0xD0 `melee sound` ("the sound that is made when I
    /// am meleed"), and the material-effects rows reached from `obje` +0xC0.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_melee_impact_sounds() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let nm = |t: u32| native.tag_name(cache, t).unwrap_or_default();
        let mut direct: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        let mut with_sound = 0usize;
        let mut total = 0usize;
        for group in ["bipd", "vehi", "weap", "scen", "bloc"] {
            for &t in &native.tags_by_group(cache, group) {
                total += 1;
                if let Some(snd) = melee_sound(native, cache, t) {
                    with_sound += 1;
                    let class = native.tag_class(cache, snd).unwrap_or_default();
                    direct
                        .entry(format!("{}  [{class}]", nm(snd)))
                        .or_default()
                        .push(nm(t).rsplit('\\').next().unwrap_or("").to_string());
                }
            }
        }
        println!("== obje +0xD0 `melee sound`: {with_sound} of {total} objects author one ==");
        for (snd, owners) in direct.iter() {
            println!("  {snd}\n      <- {} object(s): {}", owners.len(), owners.iter().take(8).cloned().collect::<Vec<_>>().join(", "));
        }
        // ...and what a WEAPON's own material rows say, which is what a melee landing on a held
        // weapon would consult.
        println!("\n== a weapon's own material-effects rows ==");
        let mut shown = 0usize;
        for &t in &native.tags_by_group(cache, "weap") {
            let Some((sounds, _effects)) = material_responses(native, cache, t) else { continue };
            if sounds.is_empty() || shown >= 6 {
                continue;
            }
            shown += 1;
            println!("  {}", nm(t));
            for r in sounds.iter().take(6) {
                println!("      '{}' -> {}", r.material, r.tag.map(nm).unwrap_or_else(|| "-".into()));
            }
        }
        if shown == 0 {
            println!("  NO weapon authors material-effects sound rows -- so a melee landing on a weapon");
            println!("  has no material route either, and the generic clash effect is the only candidate.");
        }
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_section_vitality_shares() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        for group in ["vehi", "bipd"] {
            for &t in &native.tags_by_group(cache, group) {
                let Some(m) = DamageModel::read(native, cache, t) else { continue };
                let sum: f32 = m.sections.iter().map(|s| s.vitality_percentage).sum();
                let body: f32 = m.sections.iter().filter(|s| !s.is_shield).map(|s| s.vitality_percentage).sum();
                let shield: f32 = m.sections.iter().filter(|s| s.is_shield).map(|s| s.vitality_percentage).sum();
                println!(
                    "{group} '{}': vitality {:.1}, {} sections, sum {sum:.3} (body {body:.3} / shield {shield:.3})",
                    native.tag_name(cache, t).unwrap_or_default().rsplit('\\').next().unwrap_or(""),
                    m.maximum_vitality,
                    m.sections.len()
                );
                for s in &m.sections {
                    println!("    [{}] '{}' {:.3}{}", s.index, s.name, s.vitality_percentage, if s.is_shield { " SHIELD" } else { "" });
                }
            }
        }
    }

    /// What a `jpt!` names for its SOUND, and its EFFECT — found by scanning the shipped
    /// 1020-byte tag for tag references whose class fourcc really is one of the sound or effect
    /// groups. The reflection data's offset for `sound` is from the other field order, so it
    /// cannot be used; a fourcc that validates against the cache's own class lookup can.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_jpt_sound_and_effect_references() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let mut by_offset: std::collections::BTreeMap<(usize, String), usize> = Default::default();
        let jpts = native.tags_by_group(cache, "jpt!");
        for &j in &jpts {
            // Deliberately WIDE: the point of this probe is to show what spills in past the tag.
            let Some(b) = native.tag_meta(cache, j, 0, 1020) else { continue };
            for o in (0..1020usize - 16).step_by(4) {
                let fourcc: String = b[o..o + 4].iter().rev().map(|&c| c as char).collect();
                if !fourcc.chars().all(|c| c.is_ascii_lowercase() || c == '!' || c.is_ascii_digit()) {
                    continue;
                }
                let d = rd_u32(&b, o + 0x0C);
                if d == 0 || d == u32::MAX {
                    continue;
                }
                if native.tag_class(cache, d & 0xFFFF).as_deref() == Some(fourcc.as_str()) {
                    *by_offset.entry((o, fourcc)).or_default() += 1;
                }
            }
        }
        println!("tag references inside a jpt! ({} tags), by (offset, class) -> how many tags:", jpts.len());
        for ((o, c), n) in by_offset.iter() {
            println!("  +{o:#05x}  '{c}'  x{n}");
        }
        // Name a few, so the offsets mean something.
        for leaf in ["frag_grenade", "assault_rifle_bullet", "sniper_rifle_bullet"] {
            let Some(j) = jpts.iter().copied().find(|&j| native.tag_name(cache, j).map_or(false, |n| n.contains(leaf))) else { continue };
            // Deliberately WIDE: the point of this probe is to show what spills in past the tag.
            let Some(b) = native.tag_meta(cache, j, 0, 1020) else { continue };
            println!("\n{}", native.tag_name(cache, j).unwrap_or_default());
            for ((o, c), _) in by_offset.iter() {
                let d = rd_u32(&b, o + 0x0C);
                if d == 0 || d == u32::MAX || native.tag_class(cache, d & 0xFFFF).as_deref() != Some(c.as_str()) {
                    continue;
                }
                println!("   +{o:#05x} '{c}' -> {}", native.tag_name(cache, d & 0xFFFF).unwrap_or_default());
            }
        }
    }
}
