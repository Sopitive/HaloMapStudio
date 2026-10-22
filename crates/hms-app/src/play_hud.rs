//! #play-hud  TEST MODE's first-person HUD: the live player state the authored `chud` widgets
//! need, and nothing else.
//!
//! [`crate::reach_hud`] already decodes and paints every authored widget of a `chud_definition`
//! and already carries the honest bookkeeping of which fields it has a value for. What it had no
//! source for was the values themselves: in Theater almost all of [`crate::reach_hud::HudValues`]
//! is `Absent`, because a saved film records neither ammo (zero occurrences of the relevant bits
//! over eight films and 5,438,363 unit messages) nor any vitality channel at all.
//!
//! **In test mode HMS owns the player, so those fields can finally be STATED.** That is all this
//! file is: a vitality model, an ammo model and a grenade count, each read out of the tags and
//! each carrying where it came from — plus [`PlayHud::values`], which hands them to the existing
//! painter.
//!
//! The provenance discipline is unchanged and is the point. A field this file cannot source from a
//! tag or from live play state stays [`crate::reach_hud::Prov::Absent`] and is not drawn.
//!
//! # 1. THE `weap` `magazines` ELEMENT LAYOUT — the one real RE gap, now closed
//!
//! `reverseme tagdefs reach_tag_test.exe weap` places the block —
//! `+0x3F0 block magazines -> magazines [max 2]` — and then prints
//! `struct magazines // 0 bytes (no static field list: a runtime-defined struct)`. It is one of
//! the runtime-installed field sets described in `re-notes/reach_runtime_installed_field_sets.md`:
//! the field-set record's `fields` pointer and element size are both NULL in the image and are
//! copied in by a C++ dynamic initializer at startup.
//!
//! Recovered by that note's own method, and the answer is corroborated by **three independent
//! routes that agree**:
//!
//! **(a) The content scan.** A `.data` scan for the 8-byte pointer to the field-name string
//! `"rounds loaded maximum"` (`.rdata +0x1A212E0`) finds exactly **two** candidate field arrays —
//! `+0x21486C0` and `+0x21488C0` — and no third, so the current layout must be one of them.
//!
//! **(b) The declarations name them, and the thunk picks one.**
//! `init_weapon_magazine_field_set` (`reach_tag_test.exe+0x303C0`) declares both:
//!
//! ```text
//!   declare_legacy_field_set(.., &unk_1421486C0, "weapon_magazine_fields_v1", 0x6C, 0, -1, sub_140D8A880)
//!   declare_legacy_field_set(.., &unk_1421488C0, "weapon_magazine_fields_v2", 0x60, 0, -1, 0)
//! ```
//!
//! copying each 280-byte record into `.bss` at `+0x5B83150` (v1) and `+0x5B83268` (v2). The
//! installer thunk `sub_14002FE10` then copies `+0x5B83298` — which is **`v2` + 0x30, the v2
//! record's own `fields` pointer** — into the `magazines` field-set record at `.data +0x214C9F0`.
//! So the live layout is **`weapon_magazine_fields_v2`, element size 0x60 = 96 bytes**.
//!
//! That reasoning was validated end to end against an answer this project had already settled by
//! other means: the same pattern for `weapon_barrels` declares `..._v1` (0x138), `..._v2` (0xFC)
//! and an unsuffixed `weapon_barrel_definition_fields` (**0x184 = 388**) at `+0x214AAA0`, and its
//! thunk — `sub_140030370`, the exact address `re-notes` already names — reads that record's
//! `+0x30`. 388 is the stride [`crate::reach_projectile`] measured independently, so the method
//! reproduces a known-correct result before being trusted with an unknown one.
//!
//! **(c) THE ENGINE READS IT.** `weapon_magazine_execute_reload_networking`
//! (`reach_tag_test.exe+0xDF10A0`, asserts tagged `weapons.cpp`) fetches the element with the
//! stride as a literal argument and then reads two fields out of it:
//!
//! ```text
//!   element = tag_block_get_element(weap + 0x3F0, magazine_index, 0x60);   // <- the STRIDE
//!   if (*(u8 *)element & 1)  rounds_loaded = 0;                           // +0x00 flags bit 0
//!   new_loaded = min(*(i16 *)(element + 0x0E), inventory) + rounds_loaded; // +0x0E rounds reloaded
//!   new_loaded = min(new_loaded, *(i16 *)(element + 0x0A));                // +0x0A rounds loaded maximum
//!   assert(new_loaded <= MASK(k_weapon_update_rounds_loaded_bits));        // 9 bits, i.e. <= 0x1FF
//!   if (...) weapon_take_inventory_rounds(weapon, magazine_index, new_loaded - rounds_loaded);
//! ```
//!
//! `+0x0E` is `rounds reloaded` in **v2 only** (v1 puts it at `+0x14`), and flags bit 0 is
//! `"wastes rounds when reloaded"`, which is precisely what zeroing the loaded rounds before the
//! refill does. `0x1FF` is also the same 9-bit width `film_anim` costed for the recorded
//! `rounds loaded` field. Three facts, one layout.
//!
//! And **`0x60` is a literal in three separate engine functions**, not one:
//! `weapon_magazine_execute_reload_networking`, `weapon_set_predicted_rounds_inventory`
//! (`+0xDF4390`) and `weapon_set_predicted_rounds_loaded` (`+0xDF4480`) all call the tag-block
//! element accessor as `(weap_definition + 0x3F0, magazine_index, 0x60)`. The second of those also
//! fixes what `+0x0C` MEANS, which the field name alone would not: it reads
//! `*(i16 *)(element + 0x0C)` and clamps the runtime inventory to it, so that field is the
//! reserve's ceiling. See [`MAG_RUNTIME_INVENTORY_MAXIMUM`].
//!
//! So:
//!
//! ```text
//! weapon_magazine_fields_v2  (96 bytes, weap +0x3F0, max 2)
//!   +0x00  long flags       (bit 0 "wastes rounds when reloaded",
//!                            bit 1 "every round must be chambered")
//!   +0x04  short            rounds recharged:per second
//!   +0x06  short            rounds total initial
//!   +0x08  short            rounds total maximum
//!   +0x0A  short            rounds loaded maximum        <- THE AMMO METER'S DENOMINATOR
//!   +0x0C  short            runtime rounds inventory maximum
//!   +0x0E  short            rounds reloaded
//!   +0x10  real             reload dialogue time:seconds
//!   +0x14  tag reference    reloading effect
//!   +0x24  tag reference    reloading damage effect
//!   +0x34  tag reference    chambering effect
//!   +0x44  tag reference    chambering damage effect
//!   +0x54  block            magazine objects -> magazine_objects [max 8]
//! ```
//!
//! THE TRAPS, and what was done about each:
//!
//! * **Only element 1 or later discriminates a stride.** Both candidates share their first five
//!   fields, so element 0 reads `rounds loaded maximum` at `+0x0A` either way — which is lucky for
//!   the HUD and useless as a test. [`gates::the_magazine_layout_is_v2_against_the_shipped_tags`]
//!   therefore tests the places the two candidates DISAGREE (`+0x0E` vs `+0x14`, `+0x10` vs
//!   `+0x18`, and the nested block at `+0x54` vs `+0x60`) and, for any weapon that authors two
//!   magazines, element 1 itself.
//! * **`tag_meta` reads silently past a tag's end**, so no conclusion here rests on a readable
//!   length. Every read is of a stated field at a stated offset.
//! * **The current array is not always the newest-looking one.** It is whichever one the thunk
//!   installs, which is why (b) reads the thunk rather than preferring a name.
//!
//! # 2. THE PLAYER'S SHIELD AND HEALTH
//!
//! Already authored and already recovered: a playable biped's vitality is its `hlmt`'s
//! `damage info` block, which [`crate::reach_damage::DamageModel`] reads — `maximum vitality` at
//! `hlmt +0xD0`, the `damage sections` block at `+0xE0`, 192-byte elements, each with a
//! `vitality percentage` share at `+0x08` and a `shield material name` at `+0x0C` whose own field
//! help is *"set this to make this damage section a shield"*. So the shield pool and the health
//! pool are two of the biped's own authored sections, with real numbers, and nothing here invents
//! a maximum.
//!
//! **Which section is which is decided by a refusal ladder, not by a guess** — see
//! [`Vitality::read`]. A biped whose sections this file cannot classify leaves the corresponding
//! meter `Absent`, exactly as Theater does.
//!
//! # 3. GRENADES
//!
//! `unit` authors `grenade type` and `grenade count`, at `+0x41C` and `+0x41E` of the tag — that
//! is `tagdefs`'s `+?384` / `+?386` plus the 152 bytes of the two runtime-installed structs in
//! front of them, the same `0x98` shift [`crate::reach_seats`] established and corroborated three
//! ways. The per-type MAXIMUM is global: `matg +0x444 grenades` (56-byte elements) authors
//! `maximum count` at `+0x00`, one element per `global_grenade_type_enum` member.
//!
//! A biped authors ONE type and ONE count, so in test mode that type's count is `Recorded` and the
//! other type's is a DERIVED zero — see [`Grenades::count_of`]. "N of type T" is a statement that
//! the unit carries none of any other type, the same way "two frags" says "no plasmas"; it is read
//! off the singular shape of the authored pair rather than out of a field of its own, so it is
//! labelled `Derived` and not `Recorded`.
//!
//! # WHAT THIS FILE DOES NOT CLAIM
//!
//! * **It is not the engine's firing law.** One trigger pull spends one round, which is the
//!   authored `rounds per shot` of every barrel this file has looked at but is read off no tag
//!   here; see [`ROUNDS_PER_SHOT_ASSUMED`].
//! * **Nothing damages the player yet.** `play_damage` is a one-way seam — the player damages
//!   objects, not the reverse — so the vitality pools sit at their authored maxima until something
//!   calls [`Vitality::hurt`]. That is why they are reported as the pool they are rather than as
//!   "full": the number is a live model's state, not a stand-in for one.
//! * **It runs no chud animation and draws no motion tracker.** See the notes on
//!   [`crate::reach_hud::PLAYER_COLLECTIONS`].

use hms_native::{CacheHandle, NativeDll};

use crate::reach_hud::{Field, HudValues};

// ---------------------------------------------------------------------------------------------
// THE `weap` MAGAZINES LAYOUT
// ---------------------------------------------------------------------------------------------

/// `weap +0x3F0` — the `magazines` block descriptor: `{i32 count, u32 address, u32 definition}`.
///
/// Corroborated by the engine twice over: `weapon_magazine_can_fire_weapon` and
/// `weapon_magazine_execute_reload_networking` both bounds-check an index against
/// `*(int *)(weapon_definition + 0x3F0)` under the assert
/// `"magazine_index>=0 && magazine_index<weapon_definition->weapon.magazines.count"`.
pub const WEAP_MAGAZINES: usize = 0x3F0;

/// `MAXIMUM_NUMBER_OF_MAGAZINES_PER_WEAPON`, straight off the block definition's own max-count
/// macro name at `.data +0x2148A80 +0x18`.
pub const MAGAZINE_MAX: i32 = 2;

/// The element stride: `weapon_magazine_fields_v2`, **96 bytes**.
///
/// Read as a literal third argument to the engine's own tag-block element accessor inside
/// `weapon_magazine_execute_reload_networking`, and declared with that size by
/// `init_weapon_magazine_field_set`. See the module comment.
pub const MAGAZINE_SIZE: usize = 0x60;

/// `weapon_magazine_fields_v1`, **108 bytes** — the other candidate the static data offers, kept
/// so the gates can assert it is REJECTED rather than merely unmentioned.
pub const MAGAZINE_SIZE_V1: usize = 0x6C;

/// Every element size the image offers for this block, newest-installed first.
pub const MAGAZINE_SIZE_CANDIDATES: [usize; 2] = [MAGAZINE_SIZE, MAGAZINE_SIZE_V1];

/// `flags` (`magazine_flags`). Bit 0 is `"wastes rounds when reloaded"` — the engine zeroes the
/// loaded rounds before refilling when it is set — and bit 1 `"every round must be chambered"`.
pub const MAG_FLAGS: usize = 0x00;
/// `rounds recharged:per second` — a battery weapon's regeneration, 0 on an ammunition one.
pub const MAG_ROUNDS_RECHARGED: usize = 0x04;
/// `rounds total initial`.
pub const MAG_ROUNDS_TOTAL_INITIAL: usize = 0x06;
/// `rounds total maximum`.
pub const MAG_ROUNDS_TOTAL_MAXIMUM: usize = 0x08;
/// `rounds loaded maximum` — **the magazine size, and the ammo meter's denominator.** The engine
/// clamps a reload to it: `new_loaded = min(new_loaded, *(i16 *)(element + 0x0A))`.
pub const MAG_ROUNDS_LOADED_MAXIMUM: usize = 0x0A;
/// `runtime rounds inventory maximum` — **the engine's own CEILING on the reserve**, and the field
/// that settles whether `rounds total` counts the loaded magazine.
///
/// Its meaning is not inferred from the name. `weapon_set_predicted_rounds_inventory`
/// (`reach_tag_test.exe+0xDF4390`) fetches the element at this stride and then does
///
/// ```text
///   v6 = *(i16 *)(element + 0x0C);
///   *(u16 *)(weapon + 0x2C6 + magazine_index * 0x1A) = clamp(requested, 0, v6);
/// ```
///
/// i.e. it clamps the runtime inventory to it. So the reserve a HUD prints can never exceed this,
/// whichever reading of `rounds total` is right — which is why [`Ammo::new`] clamps to it, and why
/// the inclusive/exclusive question is bounded rather than open-ended.
pub const MAG_RUNTIME_INVENTORY_MAXIMUM: usize = 0x0C;
/// The runtime magazine state's stride inside a weapon OBJECT (not the tag): the engine indexes
/// `weapon + 0x2C6 + magazine_index * 0x1A` for the inventory and `+0x2CA` for the rounds loaded.
/// Recorded for the record — HMS reads no live object — because it is what identifies the two
/// pools as separate and makes the reload arithmetic above readable.
pub const RUNTIME_MAGAZINE_STRIDE: usize = 0x1A;
/// `rounds reloaded` — how many rounds ONE reload moves. The engine:
/// `new_loaded = min(rounds_reloaded, inventory) + rounds_loaded`.
pub const MAG_ROUNDS_RELOADED: usize = 0x0E;
/// `reload dialogue time:seconds`.
pub const MAG_RELOAD_DIALOGUE_TIME: usize = 0x10;
/// `reloading effect` (tag reference).
pub const MAG_RELOADING_EFFECT: usize = 0x14;
/// `reloading damage effect` (tag reference).
pub const MAG_RELOADING_DAMAGE_EFFECT: usize = 0x24;
/// `chambering effect` (tag reference).
pub const MAG_CHAMBERING_EFFECT: usize = 0x34;
/// `chambering damage effect` (tag reference).
pub const MAG_CHAMBERING_DAMAGE_EFFECT: usize = 0x44;
/// `magazine objects` (block, `MAXIMUM_NUMBER_OF_MAGAZINE_OBJECTS_PER_MAGAZINE` = 8).
pub const MAG_MAGAZINE_OBJECTS: usize = 0x54;

/// Where `weapon_magazine_fields_v1` — the REJECTED candidate — puts `rounds reloaded`.
pub const MAG_V1_ROUNDS_RELOADED: usize = 0x14;
/// ...and its `chamber time:seconds`, where v2 has the second word of `reloading effect`.
pub const MAG_V1_CHAMBER_TIME: usize = 0x18;
/// ...and its `magazine objects` block, 12 bytes past the end of a v2 element.
pub const MAG_V1_MAGAZINE_OBJECTS: usize = 0x60;

/// `k_weapon_update_rounds_loaded_bits` — the 9-bit ceiling the engine asserts a reload against
/// (`"new_rounds_loaded <= MASK(k_weapon_update_rounds_loaded_bits)"`, `weapons.cpp:7310`), and
/// the same width `film_anim` costed for the recorded field.
pub const ROUNDS_LOADED_MASK: i32 = 0x1FF;

/// One `magazines` element, as the tags author it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Magazine {
    pub index: usize,
    pub flags: u32,
    pub rounds_recharged: i32,
    pub rounds_total_initial: i32,
    pub rounds_total_maximum: i32,
    /// `rounds loaded maximum` — the magazine size.
    pub rounds_loaded_maximum: i32,
    pub runtime_inventory_maximum: i32,
    /// `rounds reloaded` — rounds moved by one reload. 0 = this magazine authors none.
    pub rounds_reloaded: i32,
}

/// `magazine_flags` bit 0 — `"wastes rounds when reloaded"`.
pub const MAG_FLAG_WASTES_ROUNDS: u32 = 1 << 0;
/// bit 1 — `"every round must be chambered"`.
pub const MAG_FLAG_MUST_CHAMBER: u32 = 1 << 1;

impl Magazine {
    /// How many magazine elements this weapon authors, and the block's meta address.
    pub fn block(native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<(i32, u32)> {
        let b = native.tag_meta(cache, weap, WEAP_MAGAZINES as u32, 8)?;
        let n = i32::from_le_bytes(b.get(0..4)?.try_into().ok()?);
        let ptr = u32::from_le_bytes(b.get(4..8)?.try_into().ok()?);
        if n <= 0 || n > MAGAZINE_MAX || ptr == 0 || ptr == u32::MAX {
            return None;
        }
        Some((n, ptr))
    }

    /// The raw bytes of `index` at a GIVEN stride — what the discriminating gate reads both
    /// candidates through, so neither reading is privileged in the test.
    pub fn raw_at(
        native: &NativeDll,
        cache: CacheHandle,
        weap: u32,
        index: usize,
        stride: usize,
    ) -> Option<Vec<u8>> {
        let (n, ptr) = Magazine::block(native, cache, weap)?;
        if index >= n as usize {
            return None;
        }
        // Read the whole run and slice, so a stride that is wrong cannot read a short buffer and
        // look plausible by truncation.
        let all = native.tag_ptr_bytes(cache, ptr, (n as usize * stride) as u32)?;
        all.get(index * stride..(index + 1) * stride).map(|s| s.to_vec())
    }

    /// Decode one element from bytes read at [`MAGAZINE_SIZE`].
    pub fn decode(index: usize, e: &[u8]) -> Option<Magazine> {
        if e.len() < MAGAZINE_SIZE {
            return None;
        }
        let s = |o: usize| i16::from_le_bytes([e[o], e[o + 1]]) as i32;
        Some(Magazine {
            index,
            flags: u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
            rounds_recharged: s(MAG_ROUNDS_RECHARGED),
            rounds_total_initial: s(MAG_ROUNDS_TOTAL_INITIAL),
            rounds_total_maximum: s(MAG_ROUNDS_TOTAL_MAXIMUM),
            rounds_loaded_maximum: s(MAG_ROUNDS_LOADED_MAXIMUM),
            runtime_inventory_maximum: s(MAG_RUNTIME_INVENTORY_MAXIMUM),
            rounds_reloaded: s(MAG_ROUNDS_RELOADED),
        })
    }

    /// Read one of a weapon's magazines. `None` when the weapon authors none.
    pub fn read(native: &NativeDll, cache: CacheHandle, weap: u32, index: usize) -> Option<Magazine> {
        let e = Magazine::raw_at(native, cache, weap, index, MAGAZINE_SIZE)?;
        Magazine::decode(index, &e)
    }

    /// Does the authored `rounds total` count the loaded magazine, or is it the reserve on its own?
    ///
    /// **Answered by the tag's own arithmetic, not by preference.** The compiler fills in
    /// `runtime rounds inventory maximum`, so comparing it with the two authored totals says which
    /// pool "total" means:
    ///
    /// * `runtime == total maximum - rounds loaded maximum` -> "total" INCLUDES the magazine, so
    ///   the reserve a HUD prints is `total - a full magazine`;
    /// * `runtime == total maximum` -> "total" IS the reserve, and the magazine is extra.
    ///
    /// `None` when the weapon's numbers satisfy neither (or both, which happens when
    /// `rounds loaded maximum` is 0) — and then [`Ammo`] leaves the reserve `Absent` rather than
    /// choosing. The engine keeps the two pools strictly separate, which is what makes the question
    /// a real one: `weapon_take_inventory_rounds` debits the inventory by exactly the rounds a
    /// reload moved into the magazine, never by a whole magazine.
    pub fn total_counts_the_magazine(&self) -> Option<bool> {
        let r = self.runtime_inventory_maximum;
        let (t, m) = (self.rounds_total_maximum, self.rounds_loaded_maximum);
        let inclusive = r == t - m;
        let exclusive = r == t;
        match (inclusive, exclusive) {
            (true, false) => Some(true),
            (false, true) => Some(false),
            // Indistinguishable (m == 0) or neither: refuse.
            _ => None,
        }
    }

    /// The most the reserve can hold: `runtime rounds inventory maximum`, which is the value the
    /// engine clamps the live inventory to. A magazine that authors 0 here holds no reserve at all.
    pub fn reserve_max(&self) -> i32 {
        self.runtime_inventory_maximum.max(0)
    }

    /// Is this a battery weapon — one whose rounds come back on their own?
    pub fn recharges(&self) -> bool {
        self.rounds_recharged > 0
    }

    /// The engine's own reload, as [`weapon_magazine_execute_reload_networking`] computes it:
    /// `min(min(rounds_reloaded, inventory) + loaded, rounds_loaded_maximum)`, with the loaded
    /// rounds zeroed first when flags bit 0 is set. Returns `(new_loaded, rounds_taken)`.
    ///
    /// `rounds_reloaded` of 0 means the magazine authors no per-reload step; the engine's `min`
    /// would then move nothing at all, so a magazine that authors none reloads WHOLE, which is
    /// what every ammunition weapon in Reach does.
    pub fn reload(&self, loaded: i32, inventory: i32) -> (i32, i32) {
        let start = if self.flags & MAG_FLAG_WASTES_ROUNDS != 0 { 0 } else { loaded.max(0) };
        let step = if self.rounds_reloaded > 0 {
            self.rounds_reloaded
        } else {
            // A whole magazine: the room left in it.
            (self.rounds_loaded_maximum - start).max(0)
        };
        let moved = step.min(inventory.max(0));
        let new = (moved + start).min(self.rounds_loaded_maximum).min(ROUNDS_LOADED_MASK);
        (new, (new - start).max(0))
    }

    /// One line for the diagnostic.
    pub fn describe(&self) -> String {
        format!(
            "magazine[{}] loaded max {}, total {}..{}, runtime inventory max {}, reloaded {}, \
             recharged {}/s, flags {:#x}{}{} | total counts the magazine: {}",
            self.index,
            self.rounds_loaded_maximum,
            self.rounds_total_initial,
            self.rounds_total_maximum,
            self.runtime_inventory_maximum,
            self.rounds_reloaded,
            self.rounds_recharged,
            self.flags,
            if self.flags & MAG_FLAG_WASTES_ROUNDS != 0 { " WASTES" } else { "" },
            if self.flags & MAG_FLAG_MUST_CHAMBER != 0 { " CHAMBER" } else { "" },
            match self.total_counts_the_magazine() {
                Some(true) => "yes",
                Some(false) => "no",
                None => "indeterminate",
            }
        )
    }
}

// ---------------------------------------------------------------------------------------------
// THE LIVE AMMO MODEL
// ---------------------------------------------------------------------------------------------

/// One trigger pull spends one round.
///
/// **Stated as an assumption, because it is one.** `weapon_barrels` authors
/// `"rounds per shot#the number of rounds expended to create a single firing effect"`, and
/// [`crate::reach_projectile`] has the barrel stride (388) but not that field's offset, so this
/// file does not read it. Every Reach player weapon spends one round per shot, so the assumption
/// is invisible in practice — but it IS an assumption and the report says so, rather than letting
/// a shotgun-style multi-round barrel silently count wrong.
pub const ROUNDS_PER_SHOT_ASSUMED: i32 = 1;

/// `weap +0x3B8` `player interface` -> `+0x000 block first person` (32-byte elements), whose
/// `+0x010 first person animations` is the `jmad` the arms and the gun are posed from.
pub const WEAP_FIRST_PERSON: usize = 0x3B8;
/// ...element `+0x10`, a tag reference whose datum is at `+0x0C` of the reference.
pub const FIRST_PERSON_ANIMATIONS: usize = 0x10;
pub const FIRST_PERSON_ELEM: usize = 32;

/// The clip names a reload's DURATION is taken from, in the order the engine would pick them: the
/// empty-magazine reload first, because that is the one an auto-reload runs.
pub const RELOAD_CLIPS: [&str; 3] = ["reload-empty", "reload-full", "reload"];

/// How long a reload takes, and which authored clip said so.
///
/// **The magazine does NOT carry a reload duration.** Its only time field is
/// `reload dialogue time:seconds#the length of time we wait before playing the reload dialogue` —
/// a dialogue delay, not the reload — so reading it as the duration would be a plausible-looking
/// mistake. In Reach the reload lasts exactly as long as the weapon's own first-person reload
/// ANIMATION, so that is where this reads it from: the `weap`'s `first person animations` graph,
/// at the engine's own 30 fps (`frame_count * 0.033333335`, [`crate::reach_jmad::FRAME_RATE`]).
///
/// `None` when the weapon names no first-person graph or the graph holds no reload clip — and then
/// the reload completes on the next frame and the report says it was instant, rather than a made-up
/// second and a half being presented as the game's.
pub fn reload_seconds(native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<(f32, String)> {
    let h = native.tag_meta(cache, weap, WEAP_FIRST_PERSON as u32, 8)?;
    let n = i32::from_le_bytes(h.get(0..4)?.try_into().ok()?);
    let ptr = u32::from_le_bytes(h.get(4..8)?.try_into().ok()?);
    if n <= 0 || n > 2 || ptr == 0 || ptr == u32::MAX {
        return None;
    }
    let b = native.tag_ptr_bytes(cache, ptr, (n as usize * FIRST_PERSON_ELEM) as u32)?;
    let e = b.get(0..FIRST_PERSON_ELEM)?;
    let datum = u32::from_le_bytes(
        e.get(FIRST_PERSON_ANIMATIONS + 0x0C..FIRST_PERSON_ANIMATIONS + 0x10)?.try_into().ok()?,
    );
    if datum == u32::MAX {
        return None;
    }
    let graph = crate::reach_actor::AnimGraph::read(native, cache, datum & 0xFFFF)?;
    for want in RELOAD_CLIPS {
        if let Some(i) = graph.anims.iter().position(|a| a.name.eq_ignore_ascii_case(want)) {
            let a = &graph.anims[i];
            if a.frame_count > 0 {
                return Some((a.duration_seconds(), a.name.clone()));
            }
        }
    }
    // Nothing exactly named: a clip whose name CONTAINS "reload" is still the weapon's own reload.
    let i = graph.anims.iter().position(|a| a.name.to_ascii_lowercase().contains("reload") && a.frame_count > 0)?;
    Some((graph.anims[i].duration_seconds(), graph.anims[i].name.clone()))
}

/// A held weapon's live ammunition.
#[derive(Clone, Debug, PartialEq)]
pub struct Ammo {
    /// The `weap` these numbers belong to. A different weapon means a different [`Ammo`].
    pub weapon: u32,
    pub mag: Magazine,
    /// Rounds in the magazine.
    pub loaded: i32,
    /// Rounds in reserve, or `None` when the tags do not settle which pool "total" means.
    pub reserve: Option<i32>,
    /// Shots this model has already accounted for — [`crate::play_mode::PlayMode::shots`] at the
    /// last update. **The magazine has exactly one writer and it is this delta**: the fire gate
    /// only ever REFUSES a shot, it never spends a round, so a refused pull cannot double-count.
    pub shots_seen: u32,
    /// How many trigger pulls were refused for an empty magazine, for the report.
    pub dry_pulls: u32,
    /// How many reloads have run.
    pub reloads: u32,
}

impl Ammo {
    /// A freshly drawn weapon: a full magazine and the authored reserve.
    pub fn new(native: &NativeDll, cache: CacheHandle, weapon: u32, shots_seen: u32) -> Option<Ammo> {
        let mag = Magazine::read(native, cache, weapon, 0)?;
        if mag.rounds_loaded_maximum <= 0 {
            return None;
        }
        let loaded = mag.rounds_loaded_maximum.min(ROUNDS_LOADED_MASK);
        // The reserve follows the tag's own answer to "does `total` count the magazine", and is
        // then CLAMPED to the ceiling the engine itself clamps the runtime inventory to
        // (`runtime rounds inventory maximum`; see that constant). The clamp is what bounds the
        // remaining uncertainty: a reading that came out one magazine high cannot print a reserve
        // the engine would refuse to hold.
        let reserve = mag.total_counts_the_magazine().map(|inclusive| {
            let raw = if inclusive {
                mag.rounds_total_initial - mag.rounds_loaded_maximum
            } else {
                mag.rounds_total_initial
            };
            raw.clamp(0, mag.reserve_max())
        });
        Some(Ammo { weapon, mag, loaded, reserve, shots_seen, dry_pulls: 0, reloads: 0 })
    }

    /// Is there a round to fire?
    pub fn can_fire(&self) -> bool {
        self.loaded >= ROUNDS_PER_SHOT_ASSUMED
    }

    /// Account for shots that have actually left the barrel.
    ///
    /// `shots` is `PlayMode::shots`, the count of pulls the mode itself accepted — so this spends
    /// a round for a shot that happened and never for one the rate limiter swallowed.
    pub fn account_shots(&mut self, shots: u32) {
        let fired = shots.saturating_sub(self.shots_seen);
        self.shots_seen = shots;
        if fired == 0 {
            return;
        }
        self.loaded = (self.loaded - fired as i32 * ROUNDS_PER_SHOT_ASSUMED).max(0);
    }

    /// Run the engine's own reload. Returns whether anything moved.
    pub fn reload(&mut self) -> bool {
        let Some(reserve) = self.reserve else { return false };
        if self.loaded >= self.mag.rounds_loaded_maximum {
            return false;
        }
        let (new, taken) = self.mag.reload(self.loaded, reserve);
        if taken <= 0 && new == self.loaded {
            return false;
        }
        self.loaded = new;
        self.reserve = Some((reserve - taken).max(0));
        self.reloads += 1;
        true
    }

    /// A battery weapon's rounds coming back, at the authored `rounds recharged:per second`.
    ///
    /// Kept apart from the reload because it is a different authored field and a different pool:
    /// a recharging magazine takes nothing out of a reserve.
    pub fn recharge(&mut self, dt: f32, carry: &mut f32) {
        if !self.mag.recharges() || self.loaded >= self.mag.rounds_loaded_maximum {
            return;
        }
        *carry += dt * self.mag.rounds_recharged as f32;
        let whole = carry.floor();
        if whole >= 1.0 {
            *carry -= whole;
            self.loaded = (self.loaded + whole as i32).min(self.mag.rounds_loaded_maximum);
        }
    }

    /// Fraction of the magazine left, for the meter — the same quantity
    /// [`crate::reach_hud::CollectionValues::meter`] divides out.
    pub fn loaded_fraction(&self) -> Option<f32> {
        (self.mag.rounds_loaded_maximum > 0)
            .then(|| self.loaded as f32 / self.mag.rounds_loaded_maximum as f32)
    }
}

// ---------------------------------------------------------------------------------------------
// THE PLAYER'S VITALITY
// ---------------------------------------------------------------------------------------------

/// Which rule picked a biped's health section — printed, because it is an inference and the
/// alternative was to pick one silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthRule {
    /// Exactly one non-shield section: there is nowhere else for health to be.
    OnlySection,
    /// Several, and the damage info's own `indirect damage section` names one of them —
    /// "absorbes AOE or child damage", i.e. the body.
    IndirectSection,
}

impl HealthRule {
    pub fn word(self) -> &'static str {
        match self {
            HealthRule::OnlySection => "the biped's only non-shield damage section",
            HealthRule::IndirectSection => "the damage info's own `indirect damage section`",
        }
    }
}

/// A playable biped's shield and health, as pools with authored maxima.
///
/// # THE SHARES DO NOT SUM TO 1.0, and must never be asserted to
///
/// `vitality percentage` is documented as "percentage of total object vitality", which reads like a
/// partition and is not one. `forge_halo`'s Spartan authors five sections at
/// **0.64 + 1.0 + 1.0 + 1.0 + 1.0**, and every one of those numbers is right:
///
/// * `body` is 0.643 of 70 = **45**, the health pool;
/// * `head` is **1.0** because a headshot is lethal against the whole pool, not against a share of
///   it;
/// * `shield`, `armor_lock_shield_loop` and `armor_lock_shield_post` are each **1.0** because they
///   are the same 70-point shield seen three ways — the base shield and the two armour-lock states.
///
/// An Elite adds `engineer_shield` and makes it six. So each share is a fraction OF the whole, not
/// a slice of it, and a gate asserting that they add to one would be green on a single-section
/// model and wrong on every playable biped. What [`gates`] asserts instead is the property the
/// field actually has: every share is in `(0, 1]`, and every section's pool is exactly its share of
/// the model's `maximum vitality`.
#[derive(Clone, Debug, PartialEq)]
pub struct Vitality {
    pub model: crate::reach_damage::DamageModel,
    /// The section whose `shield material name` is set, when the biped authors one.
    pub shield_section: Option<usize>,
    /// The section health lives in, and which rule found it.
    pub health_section: Option<(usize, HealthRule)>,
    /// Current pools, in the sections' own units (not fractions).
    pub shield: f32,
    pub health: f32,
    /// What could not be classified, for the report. Empty when everything resolved.
    pub notes: Vec<String>,
}

impl Vitality {
    /// Read a playable biped's vitality out of its `hlmt`.
    ///
    /// **The classification is a refusal ladder.** The shield is the section the engine's own test
    /// marks as one (`shield material name` non-zero; with more than one, the damage info's
    /// `shielded state damage section` breaks the tie). Health is then the only remaining section,
    /// or the one `indirect damage section` names — and if neither answers, `health_section` is
    /// `None`, the health meter stays `Absent`, and the reason is in `notes`. Nothing picks "the
    /// first section" as a stand-in.
    pub fn read(native: &NativeDll, cache: CacheHandle, biped: u32) -> Option<Vitality> {
        Some(Vitality::from_model(crate::reach_damage::DamageModel::read(native, cache, biped)?))
    }

    /// The classification on its own, as a pure function of an authored model.
    ///
    /// Split out of [`Vitality::read`] so the gates can drive the RULE without a cache — including
    /// the branches a shipped cache does not exhibit, which is the only way to see the refusal
    /// paths go red. Both pools start at their maxima; call [`Vitality::restore`] again after
    /// editing a model.
    pub fn from_model(model: crate::reach_damage::DamageModel) -> Vitality {
        let mut notes = Vec::new();
        let shields: Vec<usize> = model.sections.iter().filter(|s| s.is_shield).map(|s| s.index).collect();
        let pool = |i: usize| model.sections.get(i).map(|s| s.vitality).unwrap_or(0.0);
        let shield_section = match shields.len() {
            0 => {
                notes.push("this biped authors no shield damage section".into());
                None
            }
            1 => Some(shields[0]),
            // SEVERAL SECTIONS ARE MARKED AS SHIELDS, and this is the normal case rather than an
            // oddity: a Spartan authors `shield`, `armor_lock_shield_loop` and
            // `armor_lock_shield_post`; an Elite adds `engineer_shield`. The damage info's
            // `shielded state damage section` is UNSET (-1) on both, so it cannot break the tie.
            //
            // **The tie does not need breaking.** Every one of those sections authors
            // `vitality percentage` = 1.0, so they all hold the same pool — the model's own
            // `maximum vitality` — and there is exactly one shield NUMBER whichever section the
            // meter is nominally about. So the rule is: take any of them once their agreement has
            // been CHECKED, and refuse if they disagree.
            //
            // The alternative was to match the section NAMED `shield`, and that was rejected on
            // this project's own evidence: a shield NAME heuristic is what drew a solid barricade
            // see-through once already. A name is an artist's label; an authored share is a number.
            _ => match model.shielded_section.filter(|i| shields.contains(i)) {
                Some(i) => Some(i),
                None => {
                    let first = pool(shields[0]);
                    let agree = shields
                        .iter()
                        .all(|i| (pool(*i) - first).abs() <= first.abs() * 1e-4 + 1e-4);
                    if agree && first > 0.0 {
                        notes.push(format!(
                            "{} sections are marked as shields ({}) and the damage info names none \
                             of them — but all {} author the same pool of {first:.0}, so the shield \
                             meter is unambiguous",
                            shields.len(),
                            shields
                                .iter()
                                .map(|i| model.sections.get(*i).map(|s| s.name.clone()).unwrap_or_default())
                                .collect::<Vec<_>>()
                                .join(", "),
                            shields.len()
                        ));
                        Some(shields[0])
                    } else {
                        notes.push(format!(
                            "{} sections are marked as shields, the damage info names none of them \
                             and they author DIFFERENT pools, so there is no one shield number and \
                             the meter is not drawn",
                            shields.len()
                        ));
                        None
                    }
                }
            },
        };
        let others: Vec<usize> = model
            .sections
            .iter()
            .filter(|s| !s.is_shield && Some(s.index) != shield_section)
            .map(|s| s.index)
            .collect();
        let health_section = if others.len() == 1 {
            Some((others[0], HealthRule::OnlySection))
        } else {
            match model.indirect_section.filter(|i| others.contains(i)) {
                Some(i) => Some((i, HealthRule::IndirectSection)),
                None => {
                    notes.push(format!(
                        "{} non-shield damage sections and the damage info's `indirect damage \
                         section` names none of them, so the health meter is not drawn",
                        others.len()
                    ));
                    None
                }
            }
        };
        let cur = |s: Option<usize>| -> f32 {
            s.and_then(|i| model.sections.get(i)).map(|s| s.vitality).unwrap_or(0.0)
        };
        let (shield, health) = (cur(shield_section), cur(health_section.map(|(i, _)| i)));
        Vitality { model, shield_section, health_section, shield, health, notes }
    }

    /// The shield section's authored pool, when there is one.
    pub fn shield_max(&self) -> Option<f32> {
        self.shield_section.and_then(|i| self.model.sections.get(i)).map(|s| s.vitality).filter(|v| *v > 0.0)
    }
    /// ...and the health section's.
    pub fn health_max(&self) -> Option<f32> {
        self.health_section
            .and_then(|(i, _)| self.model.sections.get(i))
            .map(|s| s.vitality)
            .filter(|v| *v > 0.0)
    }
    /// Shield as the 0..1 fraction the meter draws, or `None` when there is no shield to draw.
    pub fn shield_fraction(&self) -> Option<f32> {
        self.shield_max().map(|m| (self.shield / m).clamp(0.0, 1.0))
    }
    /// Health as the 0..1 fraction the meter draws.
    pub fn health_fraction(&self) -> Option<f32> {
        self.health_max().map(|m| (self.health / m).clamp(0.0, 1.0))
    }
    /// Is the player alive? The engine's own gate is the health pool, not the shield.
    pub fn alive(&self) -> bool {
        self.health_max().is_none_or(|_| self.health > 0.0)
    }

    /// Apply damage the engine's way round: the shield absorbs first, the remainder reaches health.
    ///
    /// **Nothing in HMS calls this yet** — `play_damage` carries hits from the player OUT to
    /// objects and there is no path back — so it exists to be called by whoever lands the
    /// player-damage lane, and to let a gate prove the meters move. The overflow order is the one
    /// `hlmt` encodes by having a shield section at all; what is NOT modelled here is the shield's
    /// own `recharge time` delay or the `stun time` that gates it, both of which the section
    /// authors and neither of which this file drives.
    pub fn hurt(&mut self, amount: f32) -> (f32, f32) {
        if !(amount > 0.0) {
            return (0.0, 0.0);
        }
        let to_shield = amount.min(self.shield.max(0.0));
        self.shield = (self.shield - to_shield).max(0.0);
        let rest = amount - to_shield;
        let to_health = rest.min(self.health.max(0.0));
        self.health = (self.health - to_health).max(0.0);
        (to_shield, to_health)
    }

    /// Put both pools back to their authored maxima.
    pub fn restore(&mut self) {
        self.shield = self.shield_max().unwrap_or(0.0);
        self.health = self.health_max().unwrap_or(0.0);
    }

    /// Lines for the report, naming the pool AND the rule that found it.
    pub fn describe(&self) -> Vec<String> {
        let mut out = Vec::new();
        match (self.shield_section, self.shield_max()) {
            (Some(i), Some(m)) => out.push(format!(
                "Shield: {:.0} / {:.0} (hlmt section [{}] '{}', {:.0}% of the model's {:.0} maximum vitality)",
                self.shield,
                m,
                i,
                self.model.sections.get(i).map(|s| s.name.as_str()).unwrap_or(""),
                self.model.sections.get(i).map(|s| s.vitality_percentage * 100.0).unwrap_or(0.0),
                self.model.maximum_vitality
            )),
            _ => out.push("Shield: this biped authors no shield damage section".into()),
        }
        match (self.health_section, self.health_max()) {
            (Some((i, rule)), Some(m)) => out.push(format!(
                "Health: {:.0} / {:.0} (hlmt section [{}] '{}' — picked as {})",
                self.health,
                m,
                i,
                self.model.sections.get(i).map(|s| s.name.as_str()).unwrap_or(""),
                rule.word()
            )),
            _ => out.push("Health: no damage section could be classified as health".into()),
        }
        out.extend(self.notes.iter().cloned());
        out
    }
}

// ---------------------------------------------------------------------------------------------
// GRENADES
// ---------------------------------------------------------------------------------------------

/// `unit +0x41C` `grenade type` (`global_grenade_type_enum`) — `tagdefs`'s `+?384` plus the
/// `0x98` of runtime-installed structs [`crate::reach_seats`] established.
pub const UNIT_GRENADE_TYPE: usize = 0x41C;
/// `unit +0x41E` `grenade count`.
pub const UNIT_GRENADE_COUNT: usize = 0x41E;

/// `matg +0x444` `grenades` block, 56-byte elements, one per `global_grenade_type_enum` member.
pub const MATG_GRENADES: usize = 0x444;
pub const GRENADE_BLOCK_SIZE: usize = 56;
pub const GRENADE_BLOCK_MAX: i32 = 4;
/// ...+0x00 `maximum count`.
pub const GRENADE_MAXIMUM_COUNT: usize = 0x00;
/// ...+0x28 `projectile` — the tag reference that NAMES which type an element is.
pub const GRENADE_PROJECTILE: usize = 0x28;

/// What a biped is authored to carry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grenades {
    /// `grenade type`, a `global_grenade_type_enum` index, or `None` when the biped authors none.
    pub kind: Option<usize>,
    /// `grenade count` for that type.
    pub count: i32,
    /// Per-type `maximum count` out of `matg`, in enum order, with each element's `projectile`
    /// tag name so the enum order can be READ rather than assumed.
    pub maxima: Vec<(i32, String)>,
}

impl Grenades {
    /// Read the biped's own authored type and count, and the global maxima.
    pub fn read(native: &NativeDll, cache: CacheHandle, biped: u32) -> Grenades {
        let mut g = Grenades::default();
        if let Some(b) = native.tag_meta(cache, biped, UNIT_GRENADE_TYPE as u32, 4) {
            let ty = i16::from_le_bytes([b[0], b[1]]);
            let n = i16::from_le_bytes([b[2], b[3]]) as i32;
            if ty >= 0 {
                g.kind = Some(ty as usize);
            }
            g.count = n.max(0);
        }
        g.maxima = Grenades::global_maxima(native, cache);
        g
    }

    /// `matg`'s own `grenades` block: `(maximum count, projectile tag name)` per enum member.
    pub fn global_maxima(native: &NativeDll, cache: CacheHandle) -> Vec<(i32, String)> {
        let Some(matg) = native.tags_by_group(cache, "matg").first().copied() else { return Vec::new() };
        let Some(h) = native.tag_meta(cache, matg, MATG_GRENADES as u32, 8) else { return Vec::new() };
        let n = i32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        let ptr = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
        if n <= 0 || n > GRENADE_BLOCK_MAX || ptr == 0 || ptr == u32::MAX {
            return Vec::new();
        }
        let Some(b) = native.tag_ptr_bytes(cache, ptr, (n as usize * GRENADE_BLOCK_SIZE) as u32) else {
            return Vec::new();
        };
        b.chunks_exact(GRENADE_BLOCK_SIZE)
            .map(|e| {
                let max = i16::from_le_bytes([e[GRENADE_MAXIMUM_COUNT], e[GRENADE_MAXIMUM_COUNT + 1]]) as i32;
                // A tag reference's datum is at +0x0C of the 16-byte reference.
                let datum = u32::from_le_bytes(
                    e[GRENADE_PROJECTILE + 0x0C..GRENADE_PROJECTILE + 0x10].try_into().unwrap_or([0; 4]),
                );
                let name = (datum != u32::MAX)
                    .then(|| native.tag_name(cache, datum & 0xFFFF).unwrap_or_default())
                    .unwrap_or_default();
                (max, name)
            })
            .collect()
    }

    /// The count for one `global_grenade_type_enum` member: `Some` only for the type the biped
    /// actually authors. Every other type is unknown, NOT zero — zero would be the claim "you are
    /// out of plasmas", which the tags do not make.
    ///
    /// **Clamped to `matg`'s own `maximum count`**, because the two tags disagree and the global
    /// one has to win: `forge_halo`'s Elite authors `grenade count` 4 while `matg` authors a
    /// `maximum count` of 2 for both types. 2 is what Reach actually gives a player, and clamping
    /// is the engine's own pattern for exactly this shape of conflict — see
    /// [`MAG_RUNTIME_INVENTORY_MAXIMUM`], where `weapon_set_predicted_rounds_inventory` clamps a
    /// requested round count to the authored ceiling rather than honouring it. The unclamped value
    /// is kept in `count` and both numbers are printed by [`Grenades::describe`], so the clamp is
    /// visible rather than silent.
    pub fn count_of(&self, kind: usize) -> Option<i32> {
        // THE OTHER TYPE IS ZERO, and that is the field's own meaning rather than an invention.
        // `unit` authors ONE `grenade type` and ONE `grenade count`: "this unit carries N of type
        // T" says it carries none of any other type, the same way "two frags" says "no plasmas".
        // Reported as `Derived` rather than `Recorded`, because it is read off the singular shape
        // of the authored pair and not out of a field of its own.
        //
        // It was `Absent` first, on the grounds that the tag does not state it. That was too
        // strict and it showed: the grenade block draws TWO slots, so the selected type got a
        // number and the other got a bare icon with nothing beside it — which is worse than either
        // drawing both or drawing neither, and is what "the grenades aren't right" was partly
        // about.
        // A biped that authors NO grenade type states nothing at all — not even a zero. Only a
        // biped that authors ONE type implies none of the others.
        let authored = self.kind?;
        if authored != kind {
            return Some(0);
        }
        Some(match self.maximum_of(kind) {
            Some(max) => self.count.min(max),
            // No `matg` ceiling to clamp against: the biped's own number, unmodified.
            None => self.count,
        })
    }

    /// `matg`'s authored `maximum count` for one type, when the block holds that many elements.
    pub fn maximum_of(&self, kind: usize) -> Option<i32> {
        self.maxima.get(kind).map(|(m, _)| *m).filter(|m| *m > 0)
    }

    /// The HUD field for one grenade type, with the right provenance on each of the three cases.
    ///
    /// * the type the biped authors -> `Recorded`, its count clamped to `matg`'s own maximum;
    /// * any OTHER type -> `Derived(0)`. "N of type T" states that the unit carries none of any
    ///   other type, the same way "two frags" says "no plasmas". It is read off the singular shape
    ///   of the authored pair rather than out of a field of its own, so it is `Derived`.
    /// * a biped that authors NO grenade type at all -> `Absent`. Nothing is known then, and zero
    ///   really would be a claim.
    ///
    /// The last two are a genuine distinction, not pedantry: the grenade block draws TWO slots, so
    /// leaving the unauthored type absent gave the selected type a number and the other a bare
    /// icon with nothing beside it — worse than drawing both or neither.
    pub fn field(&self, kind: usize) -> Field<i32> {
        match self.kind {
            Some(k) if k == kind => Field::recorded(self.count_of(kind).unwrap_or(self.count)),
            Some(_) => Field::derived(0),
            None => Field::absent(),
        }
    }

    /// Does the biped's own authored count exceed the global maximum? Reported, not hidden.
    pub fn clamped(&self) -> bool {
        self.kind.is_some_and(|k| self.maximum_of(k).is_some_and(|m| self.count > m))
    }

    /// The frag count as the value feed would carry it — exposed so a gate can prove the clamp
    /// survives the whole way to [`PlayHud::values`] rather than only inside this struct.
    pub fn values_frags(&self) -> Option<i32> {
        self.count_of(GRENADE_TYPE_FRAG)
    }

    /// One line for the report.
    pub fn describe(&self) -> String {
        let types: Vec<String> = self
            .maxima
            .iter()
            .enumerate()
            .map(|(i, (m, p))| {
                format!("[{i}] max {m} '{}'", p.rsplit(['\\', '/']).next().unwrap_or(p))
            })
            .collect();
        match self.kind {
            Some(k) => format!(
                "Grenades: the biped authors type {k} x{} (unit +{UNIT_GRENADE_TYPE:#05x}/+{UNIT_GRENADE_COUNT:#05x}), \
                 drawn as {}{}; matg types: {}. Every OTHER type is a DERIVED zero: one authored \
                 type and one count means none of the rest.",
                self.count,
                self.count_of(k).map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
                if self.clamped() {
                    format!(
                        " (CLAMPED to matg's own maximum of {})",
                        self.maximum_of(k).unwrap_or(0)
                    )
                } else {
                    String::new()
                },
                types.join(", ")
            ),
            None => format!(
                "Grenades: this biped authors no grenade type, so no count is drawn. matg types: {}",
                types.join(", ")
            ),
        }
    }
}

/// `global_grenade_type_enum` 0 and 1, as the `matg` block's own element order states them. The
/// NAMES are read off each element's `projectile`, so [`Grenades::describe`] can print what the
/// order actually is rather than this file asserting it.
pub const GRENADE_TYPE_FRAG: usize = 0;
pub const GRENADE_TYPE_PLASMA: usize = 1;

// ---------------------------------------------------------------------------------------------
// THE WHOLE TEST-MODE HUD STATE
// ---------------------------------------------------------------------------------------------

/// Everything test mode's HUD needs that test mode does not already hold.
///
/// Owned by `App` rather than by `PlayMode`, deliberately: the ammo and the vitality are HUD
/// state, the mode owns movement and animation, and keeping them apart means the HUD's arrival
/// changed nothing about how the player walks.
#[derive(Clone, Debug, Default)]
pub struct PlayHud {
    /// The biped this state belongs to; a species change rebuilds it.
    pub biped: u32,
    pub vitality: Option<Vitality>,
    pub grenades: Grenades,
    /// The held weapon's ammunition, or `None` for empty hands / a weapon with no magazine.
    pub ammo: Option<Ammo>,
    /// Fractional rounds a battery weapon has recharged but not yet turned into a whole round.
    recharge_carry: f32,
    /// Why the ammo is absent, when it is — so the report can say "this weapon authors no
    /// magazine" rather than leaving a silent gap.
    pub ammo_note: String,
    /// Seconds a reload in progress has been running.
    pub reloading: Option<f32>,
    /// How long a reload takes: the held weapon's own first-person reload CLIP, at the engine's
    /// 30 fps. See [`reload_seconds`]. 0 = no clip was found, so a reload completes at once.
    pub reload_duration: f32,
    /// Which authored clip gave [`PlayHud::reload_duration`], for the report. Empty = none did.
    pub reload_clip: String,
    /// Trigger pulls refused this session because the magazine was empty.
    pub dry_pulls: u32,
    /// #seat-hud  The HUDs the seat the player is in puts on screen, keyed by the `(vehi tag,
    /// seat index)` they were read for. `None` on foot.
    pub seat_huds: Option<((u32, usize), crate::reach_unit_hud::SeatHuds)>,
}

impl PlayHud {
    /// #seat-hud  Bind to the seat being ridden (`None` = on foot). A no-op while the seat is
    /// unchanged, so it costs one comparison per frame; the tag reads happen once per boarding.
    pub fn bind_seat(&mut self, native: &NativeDll, cache: CacheHandle, riding: Option<(u32, usize)>) {
        let Some(key) = riding else {
            self.seat_huds = None;
            return;
        };
        if self.seat_huds.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let huds = crate::reach_seats::UnitSeats::read(native, cache, key.0)
            .and_then(|u| u.seats.get(key.1).map(|s| crate::reach_unit_hud::SeatHuds::for_seat(native, cache, key.0, s)))
            .unwrap_or_default();
        self.seat_huds = Some((key, huds));
    }

    /// Bind to a biped: read its vitality and its authored grenades. Called on entering test mode
    /// and whenever the biped changes.
    pub fn bind(&mut self, native: &NativeDll, cache: CacheHandle, biped: u32) {
        if self.biped == biped && self.vitality.is_some() {
            return;
        }
        self.biped = biped;
        self.vitality = Vitality::read(native, cache, biped).map(|mut v| {
            v.restore();
            v
        });
        self.grenades = Grenades::read(native, cache, biped);
    }

    /// Bind to a held weapon. A change of weapon is a change of magazine, so the ammo is rebuilt
    /// from the tag — which is also what a pickup should do.
    pub fn set_weapon(&mut self, native: &NativeDll, cache: CacheHandle, weapon: u32, shots: u32) {
        if weapon == 0 {
            self.ammo = None;
            self.ammo_note = "empty hands: no magazine to count".into();
            self.reloading = None;
            self.reload_duration = 0.0;
            self.reload_clip.clear();
            return;
        }
        if self.ammo.as_ref().is_some_and(|a| a.weapon == weapon) {
            return;
        }
        self.reloading = None;
        self.recharge_carry = 0.0;
        match reload_seconds(native, cache, weapon) {
            Some((s, clip)) => {
                self.reload_duration = s;
                self.reload_clip = clip;
            }
            None => {
                self.reload_duration = 0.0;
                self.reload_clip.clear();
            }
        }
        match Ammo::new(native, cache, weapon, shots) {
            Some(a) => {
                self.ammo_note = a.mag.describe();
                self.ammo = Some(a);
            }
            None => {
                self.ammo = None;
                self.ammo_note =
                    "this weapon authors no `magazines` element with a loaded maximum, so no ammo \
                     is drawn"
                        .into();
            }
        }
    }

    /// Would a trigger pull fire? **This only ever REFUSES** — it never spends a round, so it
    /// cannot double-count against [`Ammo::account_shots`]. An empty magazine starts a reload,
    /// which is the engine's own behaviour when you pull on an empty weapon.
    pub fn before_fire(&mut self) -> bool {
        let Some(a) = self.ammo.as_mut() else { return true }; // nothing to count: do not block
        if self.reloading.is_some() {
            return false;
        }
        if a.can_fire() {
            return true;
        }
        a.dry_pulls += 1;
        self.dry_pulls += 1;
        self.start_reload();
        false
    }

    /// #weapon-fire  The weapon's barrel made an EMPTY CLICK (a pull with too little ammunition,
    /// decided by [`crate::play_fire`] from the barrel's own `rounds per shot` / `minimum rounds
    /// loaded`). Counted exactly as [`PlayHud::before_fire`] counts a dry pull -- the dry-fire sound
    /// and the report read it -- and the engine's reaction to it, a reload, is started.
    pub fn note_dry_pull(&mut self) {
        if let Some(a) = self.ammo.as_mut() {
            a.dry_pulls += 1;
        }
        self.dry_pulls += 1;
        self.start_reload();
    }

    /// #weapon-fire  Take the weapon machine's exact magazine count. The machine spends precisely
    /// the barrel's `rounds per shot` for every round it fires, which the "one round per pull"
    /// accounting of [`Ammo::account_shots`] could only assume; `shots` is [`crate::play_mode::PlayMode::shots`]
    /// so that accounting does not count the same rounds a second time.
    pub fn adopt_rounds(&mut self, loaded: i32, shots: u32) {
        if let Some(a) = self.ammo.as_mut() {
            a.loaded = loaded.clamp(0, a.mag.rounds_loaded_maximum.max(0));
            a.shots_seen = shots;
        }
    }

    /// Begin a reload, if one is possible. The duration is the magazine's authored
    /// `reload dialogue time`.
    pub fn start_reload(&mut self) -> bool {
        let Some(a) = self.ammo.as_ref() else { return false };
        if self.reloading.is_some() || a.loaded >= a.mag.rounds_loaded_maximum {
            return false;
        }
        if a.reserve.unwrap_or(0) <= 0 {
            return false;
        }
        self.reloading = Some(0.0);
        true
    }

    /// One frame. `shots` is [`crate::play_mode::PlayMode::shots`].
    pub fn tick(&mut self, dt: f32, shots: u32) {
        if let Some(a) = self.ammo.as_mut() {
            a.account_shots(shots);
            a.recharge(dt, &mut self.recharge_carry);
        }
        if let Some(t) = self.reloading.as_mut() {
            *t += dt;
            if *t >= self.reload_duration {
                self.reloading = None;
                if let Some(a) = self.ammo.as_mut() {
                    a.reload();
                }
            }
        }
        // A magazine that has run dry with rounds left in reserve reloads on its own, which is
        // what the engine does the moment the last round leaves the barrel.
        if self.reloading.is_none()
            && self.ammo.as_ref().is_some_and(|a| !a.can_fire() && a.reserve.unwrap_or(0) > 0)
        {
            self.start_reload();
        }
    }

    /// Build the value feed the existing painter consumes.
    ///
    /// `weapon` is the held `weap` (0 = empty hands), `vehicle` the `(chud path, seat)` of a ride,
    /// and `button_label` the key the player has bound to the action button — which is what lets a
    /// button prompt be drawn with their own key instead of being refused.
    pub fn values(
        &self,
        weapon: u32,
        vehicle: Option<(String, u8)>,
        button_label: Option<String>,
    ) -> HudValues {
        let mut v = HudValues {
            weapon_tag: (weapon != 0).then_some(weapon),
            vehicle,
            button_label,
            ..Default::default()
        };
        if let Some(a) = &self.ammo {
            // STATED, not derived: this is the model's own count of a magazine HMS owns, and its
            // denominator is the authored `rounds loaded maximum`.
            v.loaded = Field::recorded(a.loaded);
            v.loaded_max = Some(a.mag.rounds_loaded_maximum);
            v.reserve = match a.reserve {
                Some(r) => Field::recorded(r),
                // The tags did not settle which pool `rounds total` means, so there is no honest
                // reserve to print. The magazine still draws.
                None => Field::absent(),
            };
        }
        if let Some(vit) = &self.vitality {
            v.shield = match vit.shield_fraction() {
                Some(f) => Field::recorded(f),
                None => Field::absent(),
            };
            v.health = match vit.health_fraction() {
                Some(f) => Field::recorded(f),
                None => Field::absent(),
            };
        }
        v.frags = self.grenades.field(GRENADE_TYPE_FRAG);
        v.plasmas = self.grenades.field(GRENADE_TYPE_PLASMA);
        v
    }

    /// The value feed for a RIDDEN VEHICLE's own `chud`.
    ///
    /// **The player-owned fields are cleared, deliberately.** A vehicle `chud` authors its own
    /// `shield_meter` and `health_meter`, and [`crate::reach_hud::CollectionValues`] matches a
    /// meter by COLLECTION NAME — so handing it the same values would draw the player's shield
    /// fraction on the Warthog's meter and claim the vehicle's condition from the driver's. HMS
    /// models no vehicle vitality in test mode (`play_damage` is a one-way seam and keeps no
    /// per-placement pool) and no vehicle-weapon ammunition, so both stay `Absent` and the
    /// vehicle's meters are not drawn. What does draw is everything the tag's own states allow:
    /// its frames, its icons and its reticule.
    pub fn vehicle_values(
        &self,
        vehicle: Option<(String, u8)>,
        button_label: Option<String>,
    ) -> HudValues {
        HudValues { vehicle, button_label, ..Default::default() }
    }

    /// The lines the HUD panel prints — every number with where it came from.
    pub fn report(&self) -> Vec<String> {
        let mut out = Vec::new();
        match &self.vitality {
            Some(v) => out.extend(v.describe()),
            None => out.push(
                "Shield / health: this biped's `hlmt` carries no usable `damage info`, so neither \
                 meter is drawn"
                    .into(),
            ),
        }
        out.push(self.grenades.describe());
        match &self.ammo {
            Some(a) => {
                out.push(format!(
                    "Ammo: {} loaded of {} (stated — test mode owns the magazine), reserve {}",
                    a.loaded,
                    a.mag.rounds_loaded_maximum,
                    match a.reserve {
                        Some(r) => r.to_string(),
                        None =>
                            "unknown: the tag's `runtime rounds inventory maximum` matches neither \
                             reading of `rounds total`"
                                .into(),
                    }
                ));
                out.push(format!(
                    "  {} | {} shot(s) accounted, {} dry pull(s), {} reload(s); one round per shot \
                     is ASSUMED (`rounds per shot` lives in `weapon_barrels`, whose offset is not read here)",
                    self.ammo_note, a.shots_seen, a.dry_pulls, a.reloads
                ));
            }
            None => out.push(format!("Ammo: {}", self.ammo_note)),
        }
        out.push(if self.reload_clip.is_empty() {
            "Reload time: this weapon's first-person graph holds no reload clip, so a reload \
             completes at once (the magazine authors no reload duration — its only time field is \
             `reload dialogue time`, which is a dialogue delay)"
                .into()
        } else {
            format!(
                "Reload time: {:.2} s, the authored clip '{}' at 30 fps",
                self.reload_duration, self.reload_clip
            )
        });
        if let Some(t) = self.reloading {
            out.push(format!("Reloading: {t:.2} s of {:.2}", self.reload_duration));
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// print-only diagnostic
// ---------------------------------------------------------------------------------------------

/// Printed by `--dump-hud`: the recovered `magazines` layout read off real weapons under BOTH
/// candidate strides, and every playable biped's vitality and grenades.
///
/// This is the map-backed half of the recovery. The static evidence says 96; this prints what the
/// shipped tags read at 96 and at 108 side by side, so the answer can be checked by looking.
pub fn dump_ammo_and_vitality(native: &NativeDll, cache: CacheHandle, filter: &str) {
    println!("\n# #play-hud  weap +{WEAP_MAGAZINES:#05x} `magazines`, element size {MAGAZINE_SIZE} (weapon_magazine_fields_v2)\n");
    let mut tags = native.tags_by_group(cache, "weap");
    tags.sort_unstable();
    let mut two_mag = 0usize;
    let mut authored = 0usize;
    let mut v2_evidence = (0usize, 0usize); // (a valid reloading-effect ref at +0x14, a plausible reload time at +0x10)
    let mut v1_evidence = (0usize, 0usize); // (a plausible chamber time at +0x18, a valid block at +0x60)
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
            continue;
        }
        let Some((n, _)) = Magazine::block(native, cache, t) else { continue };
        let short = name.rsplit('\\').next().unwrap_or(&name).to_string();
        authored += 1;
        if n > 1 {
            two_mag += 1;
        }
        for i in 0..n as usize {
            let Some(e) = Magazine::raw_at(native, cache, t, i, MAGAZINE_SIZE) else { continue };
            let Some(m) = Magazine::decode(i, &e) else { continue };
            println!("  {short:<34} {}", m.describe());
            // THE DISCRIMINATION, printed per weapon so it can be read rather than trusted.
            let f = |o: usize| f32::from_le_bytes(e[o..o + 4].try_into().unwrap_or([0; 4]));
            let ref_at = |o: usize| -> u32 {
                u32::from_le_bytes(e[o + 0x0C..o + 0x10].try_into().unwrap_or([0; 4]))
            };
            let plausible = |t: f32| t > 0.0 && t < 30.0;
            let reloading = ref_at(MAG_RELOADING_EFFECT);
            if reloading != u32::MAX && reloading != 0 {
                v2_evidence.0 += 1;
            }
            if plausible(f(MAG_RELOAD_DIALOGUE_TIME)) {
                v2_evidence.1 += 1;
            }
            if plausible(f(MAG_V1_CHAMBER_TIME)) {
                v1_evidence.0 += 1;
            }
            println!(
                "  {:<34}   v2: reload dialogue time {:.3}s, reloading effect datum {reloading:#x}  |  \
                 v1 would read: rounds reloaded {}, chamber time {:.3}s",
                "",
                f(MAG_RELOAD_DIALOGUE_TIME),
                i16::from_le_bytes([e[MAG_V1_ROUNDS_RELOADED], e[MAG_V1_ROUNDS_RELOADED + 1]]),
                f(MAG_V1_CHAMBER_TIME)
            );
        }
        // Only element 1 discriminates a stride, so a two-magazine weapon is printed both ways.
        if n > 1 {
            for stride in MAGAZINE_SIZE_CANDIDATES {
                let Some(e) = Magazine::raw_at(native, cache, t, 1, stride) else { continue };
                let Some(m) = Magazine::decode(1, &e) else { continue };
                println!("  {:<34}   ELEMENT 1 at stride {stride}: {}", "", m.describe());
            }
        }
    }
    println!(
        "\n  {authored} weapon(s) author a magazines block, {two_mag} of them author TWO.\n  \
         v2 evidence: {} valid `reloading effect` references at +{MAG_RELOADING_EFFECT:#04x} \
         (a PAD under v1), {} plausible `reload dialogue time`s at +{MAG_RELOAD_DIALOGUE_TIME:#04x} \
         (pad tail under v1).\n  v1 evidence: {} plausible `chamber time`s at \
         +{MAG_V1_CHAMBER_TIME:#04x} (inside a tag reference under v2).",
        v2_evidence.0, v2_evidence.1, v1_evidence.0
    );

    println!("\n# #play-hud  playable bipeds: vitality and authored grenades\n");
    let mut bipeds = native.tags_by_group(cache, "bipd");
    bipeds.sort_unstable();
    for b in bipeds {
        let name = native.tag_name(cache, b).unwrap_or_default();
        println!("  bipd '{name}'");
        match Vitality::read(native, cache, b) {
            Some(mut v) => {
                v.restore();
                for l in v.model.describe() {
                    println!("      {l}");
                }
                for l in v.describe() {
                    println!("    {l}");
                }
                // The damage law, exercised: a shield that absorbs and a health pool that then
                // takes the rest, so the printed numbers are a model and not a pair of constants.
                let before = (v.shield, v.health);
                let (s, h) = v.hurt(v.shield + 10.0);
                println!(
                    "    a {:.0} damage hit: {s:.0} off the shield and {h:.0} off health -> {:.0}/{:.0}",
                    before.0 + 10.0,
                    v.shield,
                    v.health
                );
            }
            None => println!("    no usable `damage info` on this biped's hlmt"),
        }
        println!("    {}", Grenades::read(native, cache, b).describe());
    }
}

/// #hud-trace  **THE HEADLESS PICTURE.** Run the REAL painter into an off-screen `egui::Context`
/// and print, per widget, the sprite it resolved to, the rect it landed on and whether it was
/// treated as a meter.
///
/// This is how "the shields and health don't look right" becomes a defect rather than an
/// impression. The chud overlay cannot be captured headlessly, but the painter's own decisions can
/// be read, and a widget in the wrong place or at the wrong size is a wrong NUMBER here.
///
/// `--dump-hud` calls it. It needs no GPU: `egui::Context::default()` accepts texture uploads and
/// the shapes are inspected rather than rasterised.
pub fn dump_painted_hud(native: &NativeDll, cache: CacheHandle, weapon_filter: &str) {
    use crate::reach_hud as rh;
    let ctx = eframe::egui::Context::default();
    let vp = eframe::egui::Rect::from_min_size(
        eframe::egui::pos2(0.0, 0.0),
        eframe::egui::vec2(1920.0, 1080.0),
    );
    let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned());
    let vs = match &skin {
        Some(sk) => rh::VirtualScreen::from_skin(vp, sk),
        None => rh::VirtualScreen::fallback(vp),
    };
    println!("\n# #hud-trace  WHAT THE PAINTER ACTUALLY EMITS, at 1920x1080\n");
    println!(
        "  virtual screen: safe frame {:.0}x{:.0} at ({:.0},{:.0}), one virtual unit = {:.4} pt",
        vs.safe.width(),
        vs.safe.height(),
        vs.safe.min.x,
        vs.safe.min.y,
        vs.vpx
    );
    // THE SKIN PALETTE WITH ITS ALPHAS. `widget_color` takes a skin colour OPAQUE on the stated
    // grounds that several are authored a = 0.0 and are plainly visible in game, while
    // `reach_chud::paint_crosshair` multiplies the authored alpha in. Both cannot be right, and
    // this prints the numbers that decide it.
    if let Some(sk) = &skin {
        println!("\n  skin '{}' colours (a, r, g, b):", sk.type_name());
        for (i, name) in crate::reach_chud::SKIN_COLORS.iter().enumerate() {
            let c = sk.colors[i];
            println!(
                "    [{i:>2}] {name:<28} a={:.3} r={:.3} g={:.3} b={:.3}{}",
                c[0],
                c[1],
                c[2],
                c[3],
                if c[0] <= 0.001 { "   <-- ALPHA ZERO" } else { "" }
            );
        }
    }

    // The player's own state, exactly as test mode would present it: a full Spartan with the
    // filtered weapon in hand.
    let biped = native
        .tags_by_group(cache, "bipd")
        .into_iter()
        .find(|&t| {
            native
                .tag_name(cache, t)
                .is_some_and(|n| n.to_ascii_lowercase().starts_with("objects\\characters\\spartans"))
        });
    let weapon = native.tags_by_group(cache, "weap").into_iter().find(|&t| {
        native
            .tag_name(cache, t)
            .is_some_and(|n| weapon_filter.is_empty() || n.to_ascii_lowercase().contains(&weapon_filter.to_ascii_lowercase()))
    });
    let mut hud = PlayHud::default();
    if let Some(b) = biped {
        hud.bind(native, cache, b);
    }
    if let Some(w) = weapon {
        hud.set_weapon(native, cache, w, 0);
    }
    let v = hud.values(weapon.unwrap_or(0), None, Some("E".into()));
    println!("\n  values fed to the painter:");
    for l in v.report() {
        println!("    {l}");
    }
    println!("  known chud states asserted: {:?}", v.known_states());

    let run = |label: &str, path_or_tag: Option<u32>| {
        let Some(t) = path_or_tag else {
            println!("\n  -- {label}: no tag");
            return;
        };
        let Some(art) = rh::hud_art(&ctx, native, cache, t) else {
            println!("\n  -- {label}: did not decode");
            return;
        };
        println!(
            "\n  -- {label}: {} ({} collection(s))",
            native.tag_name(cache, t).unwrap_or_default(),
            art.chud.collections.len()
        );
        // #hud-origin  THE COLLECTION's own placement, and what the two readings of a child's
        // `widget origin` would give. HMS centres a bitmap on its point and applies `origin` only
        // to TEXT, and the module says plainly that neither reading was confirmed against the
        // engine's quad builder. The numbers are printed so the question can be settled by looking
        // rather than argued: a reading that puts an authored block OUTSIDE the safe frame is wrong.
        for c in &art.chud.collections {
            let Some(cp) = c.base.placement() else { continue };
            let p = rh::widget_point(&vs, cp, None);
            println!(
                "       [{}] anchor {} off ({:+.1},{:+.1}) origin ({:.2},{:.2}) -> {} | datasource {:?} max {} positions {:?}",
                c.base.name,
                cp.anchor,
                cp.offset[0],
                cp.offset[1],
                cp.origin[0],
                cp.origin[1],
                match p {
                    Some((at, _)) => format!("point ({:.0},{:.0})", at.x, at.y),
                    None => "UNRESOLVED anchor".into(),
                },
                c.datasource,
                c.datasource_max,
                c.positions.iter().map(|q| q.0).collect::<Vec<_>>()
            );
        }
        rh::set_widget_log(true);
        let out = ctx.run(Default::default(), |ctx| {
            let painter = ctx.layer_painter(eframe::egui::LayerId::new(
                eframe::egui::Order::Foreground,
                eframe::egui::Id::new(label),
            ));
            for (i, c) in art.chud.collections.iter().enumerate() {
                let reps = if c.datasource == Some(rh::DATASOURCE_GRENADES) { 2 } else { 1 };
                for r in 0..reps {
                    rh::paint_collection(ctx, &painter, &vs, &art, i, skin.as_ref(), &v, r);
                }
            }
            let _ = &art;
        });
        let log = rh::take_widget_log();
        rh::set_widget_log(false);
        let drawn = log.iter().filter(|d| d.outcome == "drawn").count();
        // #hud-trace  DRAWN IS NOT VISIBLE. The shield defect was a widget that painted perfectly
        // and was then covered by the next one at the same rectangle in an alpha-over blend — a
        // draw count of N with the bar missing from the screen. So say which of the drawn widgets
        // a LATER one buries, rather than leaving the next report to find it by eye.
        let occluded = rh::occlusions(&log);
        for d in &log {
            let hidden = occluded
                .iter()
                .find(|(i, _)| log[*i].widget == d.widget && log[*i].repeat == d.repeat)
                .map(|(_, by)| format!("   <-- PAINTED THEN COVERED by '{by}'"))
                .unwrap_or_default();
            println!("     {}{hidden}", d.line(Some(&vs)));
        }
        println!(
            "     => {} widget(s) considered, {drawn} drawn ({} of them painted then covered), {} shape(s) emitted",
            log.len(),
            occluded.len(),
            out.shapes.len()
        );
        // #hud-origin  For every drawn bitmap whose authored `origin` is NOT (0.5, 0.5), the rect
        // each reading gives and whether it falls inside the authored safe frame. A block that only
        // fits under one of the two readings is evidence for that one.
        let mut asym = 0usize;
        for d in log.iter().filter(|d| d.outcome == "drawn" && d.kind == "bitmap") {
            if (d.origin[0] - 0.5).abs() < 0.01 && (d.origin[1] - 0.5).abs() < 0.01 {
                continue;
            }
            let Some(r) = d.rect else { continue };
            // The CENTRED reading is what was drawn. The PIVOT reading shifts by
            // (0.5 - origin) * size, i.e. it puts `origin` of the box on the point instead.
            let shift = eframe::egui::vec2(
                (0.5 - d.origin[0]) * r.width(),
                (0.5 - d.origin[1]) * r.height(),
            );
            let pivot = r.translate(shift);
            let inside = |q: eframe::egui::Rect| vs.safe.contains_rect(q);
            asym += 1;
            println!(
                "       #hud-origin {}/{} origin ({:.2},{:.2}): CENTRED {:.0}..{:.0} x {:.0}..{:.0} {} | PIVOT {:.0}..{:.0} x {:.0}..{:.0} {}",
                d.collection,
                d.widget,
                d.origin[0],
                d.origin[1],
                r.min.x, r.max.x, r.min.y, r.max.y,
                if inside(r) { "inside safe frame" } else { "OUTSIDE safe frame" },
                pivot.min.x, pivot.max.x, pivot.min.y, pivot.max.y,
                if inside(pivot) { "inside safe frame" } else { "OUTSIDE safe frame" },
            );
        }
        if asym == 0 {
            println!("       #hud-origin every drawn bitmap authors a centred origin, so the two readings agree here");
        }
    };

    run("PLAYER HUD (spartan)", rh::chud_by_path(native, cache, rh::PLAYER_CHUD_SPARTAN));
    if let Some(w) = weapon {
        run(
            &format!("WEAPON HUD ({})", native.tag_name(cache, w).unwrap_or_default()),
            crate::reach_chud::weapon_chud(native, cache, w),
        );
        // AND THE RETICULE, which goes through a DIFFERENT painter
        // (`reach_chud::paint_crosshair`) and so is not in the trace above.
        match (crate::reach_chud::crosshair_for(&ctx, native, cache, w), &skin) {
            (Some(a), Some(sk)) => {
                let mut union = eframe::egui::Rect::NOTHING;
                let out = ctx.run(Default::default(), |ctx| {
                    let painter = ctx.layer_painter(eframe::egui::LayerId::new(
                        eframe::egui::Order::Foreground,
                        eframe::egui::Id::new("xh"),
                    ));
                    union = crate::reach_chud::paint_crosshair(&painter, vp, &a, sk, vs.vpx);
                });
                println!("\n  -- RETICULE: {} authored part(s)", a.crosshair.parts.len());
                for part in &a.crosshair.parts {
                    let sz = part.size_vu();
                    let col = part.color.map(|k| sk.colors[k]);
                    println!(
                        "     {:<24} {:.1}x{:.1} vu at offset ({:+.1},{:+.1}) -> {:.0}x{:.0} pt | colour {:?}{}",
                        part.widget,
                        sz[0],
                        sz[1],
                        part.offset[0],
                        part.offset[1],
                        sz[0] * vs.vpx,
                        sz[1] * vs.vpx,
                        col,
                        match col {
                            Some(c) if c[0] <= 0.001 =>
                                "   <-- AUTHORED ALPHA 0: paint_crosshair multiplies this in, so it draws INVISIBLE",
                            _ => "",
                        }
                    );
                }
                println!(
                    "     union {:.0}x{:.0} pt, centre {:.1} pt off the viewport centre, {} shape(s)",
                    union.width(),
                    union.height(),
                    (union.center() - vp.center()).length(),
                    out.shapes.len()
                );
            }
            _ => println!("\n  -- RETICULE: this weapon decodes none at rest"),
        }
    }
}

/// #hud-art  **THE ART AND COLOUR AUDIT.** For every bitmap widget of every `chdt`: which `bitm`
/// it names, how that `bitm` is addressable (sequences, sprites per sequence, submaps), the
/// authored `sequence index`, and where its colour comes from.
///
/// Three questions the per-widget trace raised and could not answer, each decided by the corpus
/// rather than by preference:
///
/// 1. **Does a grenade sheet hold more than one sprite?** If it does, drawing sprite 0 for both
///    grenade slots is why "the grenades aren't right".
/// 2. **Is `color output A` = 0 with an unauthored `custom color A` COMMON?** If it is, 0 is the
///    enum's "no override / use the default", not "custom color A", and refusing to draw those
///    widgets is suppressing real HUD. The assault rifle's ammo METER is one of them.
/// 3. **Are the meter bitmaps single-submap?** If so, resolving them to submap 0 is right and the
///    `UNRESOLVED` route is benign rather than a bug.
pub fn dump_hud_art_audit(native: &NativeDll, cache: CacheHandle) {
    use crate::reach_hud as rh;
    println!("\n# #hud-art  every chud bitmap widget: its bitm's shape, and where its colour comes from\n");
    // `bitm` block offsets, the same ones `reach_chud::read_sprite` walks.
    let bitm_shape = |t: u32| -> (usize, usize, usize) {
        // (sequences, sprites in sequence 0, submaps)
        let Some(meta) = native.tag_meta(cache, t, 0, 0xC0) else { return (0, 0, 0) };
        let rd = |o: usize| -> (usize, u32) {
            let n = i32::from_le_bytes(meta[o..o + 4].try_into().unwrap_or([0; 4]));
            let p = u32::from_le_bytes(meta[o + 4..o + 8].try_into().unwrap_or([0; 4]));
            (n.max(0) as usize, p)
        };
        let (nseq, pseq) = rd(0x70);
        let (nbm, _) = rd(0x7C);
        let mut nsp = 0usize;
        if nseq > 0 && pseq != 0 && pseq != u32::MAX {
            if let Some(b) = native.tag_ptr_bytes(cache, pseq, (nseq * 64) as u32) {
                if let Some(seq0) = b.chunks_exact(64).next() {
                    nsp = i32::from_le_bytes(seq0[0x34..0x38].try_into().unwrap_or([0; 4])).max(0) as usize;
                }
            }
        }
        (nseq, nsp, nbm)
    };
    let mut out0_unauthored = 0usize;
    let mut out0_total = 0usize;
    let mut skin_coloured = 0usize;
    let mut custom_coloured = 0usize;
    let mut multi_sprite = Vec::new();
    let mut tags = native.tags_by_group(cache, "chdt");
    tags.sort_unstable();
    for t in tags {
        let path = native.tag_name(cache, t).unwrap_or_default();
        let Some(chud) = rh::read_chud(native, cache, t) else { continue };
        let mut printed_header = false;
        for coll in &chud.collections {
            for w in &coll.widgets {
                let rh::WidgetArt::Bitmap { sprite, sequence, route, .. } = &w.art else { continue };
                let (nseq, nsp, nbm) = bitm_shape(sprite.bitmap_tag);
                let bname = native.tag_name(cache, sprite.bitmap_tag).unwrap_or_default();
                let authored: Vec<&str> = (0..4)
                    .map(|i| if w.base.custom[i].iter().any(|v| *v > 0.0) { "X" } else { "." })
                    .collect();
                let no_colour =
                    w.base.color.is_none() && !w.base.custom.iter().any(|c| c.iter().any(|v| *v > 0.0));
                if w.base.color.is_some() {
                    skin_coloured += 1;
                } else if !no_colour {
                    custom_coloured += 1;
                }
                if w.base.output_a == 0 {
                    out0_total += 1;
                    if no_colour {
                        out0_unauthored += 1;
                    }
                }
                if nsp > 1 {
                    multi_sprite.push(format!(
                        "{}/{}  {} seq{} sprites{}",
                        coll.base.name,
                        w.base.name,
                        bname.rsplit('\\').next().unwrap_or(&bname),
                        nseq,
                        nsp
                    ));
                }
                // Print only the interesting rows: a sheet with several sprites, or a widget whose
                // colour HMS refuses.
                if nsp > 1 || no_colour {
                    if !printed_header {
                        println!("  chdt '{path}'");
                        printed_header = true;
                    }
                    println!(
                        "    {:<22}/{:<24} bitm '{}' seq {nseq} sprites(seq0) {nsp} submaps {nbm} | authored seq {sequence} via {} | output A {:>3} skin {:?} custom [{}]{}",
                        coll.base.name,
                        w.base.name,
                        bname.rsplit('\\').next().unwrap_or(&bname),
                        route.word(),
                        w.base.output_a,
                        w.base.color,
                        authored.join(""),
                        if no_colour { "  <-- HMS REFUSES: no colour anywhere" } else { "" }
                    );
                }
            }
        }
    }
    println!(
        "\n  COLOUR: {skin_coloured} widget(s) take a SKIN colour, {custom_coloured} an authored CUSTOM one, \
         and {out0_unauthored} of {out0_total} with `color output A` = 0 author NO colour at all.\n  \
         If that last count is large, 0 is the enum's default rather than `custom color A`, and \
         refusing those widgets suppresses real HUD."
    );
    println!("\n  MULTI-SPRITE SHEETS ({}):", multi_sprite.len());
    for l in &multi_sprite {
        println!("    {l}");
    }
}

#[cfg(test)]
mod gates {
    //! #play-hud  `cargo test -p hms-app --profile quick play_hud::`
    //!
    //! Everything that needs the shipped cache is `#[ignore]`d and named in its own message, the
    //! same arrangement every other map-backed module here uses.

    use super::*;

    /// A magazine as the shipped assault rifle authors one, built from stated numbers rather than
    /// read back out of the code under test.
    fn ar_like() -> Magazine {
        Magazine {
            index: 0,
            flags: 0,
            rounds_recharged: 0,
            rounds_total_initial: 288,
            rounds_total_maximum: 288,
            rounds_loaded_maximum: 32,
            runtime_inventory_maximum: 256,
            rounds_reloaded: 0,
        }
    }

    /// **THE RELOAD GATE.** The arithmetic is the engine's own, transcribed from
    /// `weapon_magazine_execute_reload_networking`, and every clause of it has to be in play: the
    /// `min` against the reserve, the `min` against `rounds loaded maximum`, the per-reload step
    /// when one is authored, and the `wastes rounds when reloaded` flag that throws the partial
    /// magazine away first.
    #[test]
    fn the_reload_is_the_engines_own_arithmetic() {
        // A whole magazine: `rounds reloaded` is 0, so the room left in the magazine is moved.
        let m = ar_like();
        assert_eq!(m.reload(0, 256), (32, 32), "an empty magazine fills");
        assert_eq!(m.reload(20, 256), (32, 12), "a partial one tops up and pays for 12");
        assert_eq!(m.reload(32, 256), (32, 0), "a full one moves nothing");
        // THE RESERVE IS THE CEILING. `min(rounds_reloaded, inventory)`.
        assert_eq!(m.reload(0, 5), (5, 5), "only five rounds left to take");
        assert_eq!(m.reload(0, 0), (0, 0), "an empty reserve reloads nothing");
        // A PER-RELOAD STEP, which is what a shotgun authors: one shell at a time.
        let shot = Magazine { rounds_loaded_maximum: 6, rounds_reloaded: 1, ..m };
        assert_eq!(shot.reload(0, 24), (1, 1), "one shell");
        assert_eq!(shot.reload(5, 24), (6, 1), "and the sixth fills it");
        assert_eq!(shot.reload(6, 24), (6, 0));
        // THE CEILING IS `rounds loaded maximum`, not the step: a step bigger than the magazine
        // must not overfill it.
        let big = Magazine { rounds_loaded_maximum: 4, rounds_reloaded: 99, ..m };
        assert_eq!(big.reload(0, 999), (4, 4), "min against rounds loaded maximum");
        // `wastes rounds when reloaded`: the engine zeroes the loaded rounds FIRST, so a partial
        // magazine reload costs a whole magazine and the partial rounds are gone.
        let waste = Magazine { flags: MAG_FLAG_WASTES_ROUNDS, ..m };
        assert_eq!(waste.reload(20, 256), (32, 32), "the 20 were thrown away and 32 paid for");
        assert_eq!(m.reload(20, 256).1, 12, "...which is NOT what the flag-less magazine does");
        // And the 9-bit assert the engine itself makes.
        let huge = Magazine { rounds_loaded_maximum: 4096, rounds_reloaded: 4096, ..m };
        assert_eq!(huge.reload(0, 99_999).0, ROUNDS_LOADED_MASK, "clamped to 9 bits");
        assert_eq!(ROUNDS_LOADED_MASK, 0x1FF, "k_weapon_update_rounds_loaded_bits");
    }

    /// **THE `rounds total` GATE.** Whether the authored total counts the loaded magazine is
    /// decided by the tag compiler's own `runtime rounds inventory maximum`, and the two readings
    /// must be told apart — including the case where neither holds, which has to come back `None`
    /// rather than defaulting to either.
    #[test]
    fn which_pool_rounds_total_means_is_read_from_the_tag() {
        // inventory max = total - magazine  ->  "total" includes the magazine
        let inc = Magazine { rounds_total_maximum: 288, rounds_loaded_maximum: 32, runtime_inventory_maximum: 256, ..ar_like() };
        assert_eq!(inc.total_counts_the_magazine(), Some(true));
        // inventory max = total            ->  "total" IS the reserve
        let exc = Magazine { rounds_total_maximum: 288, rounds_loaded_maximum: 32, runtime_inventory_maximum: 288, ..ar_like() };
        assert_eq!(exc.total_counts_the_magazine(), Some(false));
        // neither  ->  refuse
        let odd = Magazine { rounds_total_maximum: 288, rounds_loaded_maximum: 32, runtime_inventory_maximum: 100, ..ar_like() };
        assert_eq!(odd.total_counts_the_magazine(), None);
        // AND THE DEGENERATE CASE: a zero magazine makes the two readings identical, so the
        // question is not answerable and must not be answered.
        let zero = Magazine { rounds_total_maximum: 50, rounds_loaded_maximum: 0, runtime_inventory_maximum: 50, ..ar_like() };
        assert_eq!(
            zero.total_counts_the_magazine(),
            None,
            "with no magazine the two readings coincide and nothing is proved"
        );
        // AND THE CEILING, whose meaning the engine fixes rather than its name: it is the value
        // `weapon_set_predicted_rounds_inventory` clamps the live inventory to, so it is the most
        // a reserve can ever read — which is what bounds the reading above.
        assert_eq!(inc.reserve_max(), 256);
        assert_eq!(Magazine { runtime_inventory_maximum: -1, ..inc }.reserve_max(), 0, "never negative");
        assert_eq!(RUNTIME_MAGAZINE_STRIDE, 0x1A, "weapon + 0x2C6 + i * 0x1A");
    }

    /// **THE AMMO-ACCOUNTING GATE, and the trap it is here to hold shut.**
    ///
    /// The magazine has exactly ONE writer — the delta of `PlayMode::shots`, which counts pulls the
    /// mode actually accepted. [`PlayHud::before_fire`] may only refuse. Two writers on one field
    /// is a mistake this project has already paid for once, and here it would show as a weapon
    /// that spends two rounds a trigger pull.
    #[test]
    fn the_magazine_has_exactly_one_writer() {
        let mut a = Ammo {
            weapon: 7,
            mag: ar_like(),
            loaded: 32,
            reserve: Some(256),
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        };
        a.account_shots(1);
        assert_eq!(a.loaded, 31, "one shot, one round");
        a.account_shots(1);
        assert_eq!(a.loaded, 31, "the SAME shot count spends nothing more");
        a.account_shots(4);
        assert_eq!(a.loaded, 28, "three more shots");
        // A shot counter that went backwards (a fresh mode) must not refill the magazine.
        a.account_shots(0);
        assert_eq!(a.loaded, 28, "a lower count is not a refill");
        // AND THE REFUSAL PATH SPENDS NOTHING. `before_fire` on an empty magazine returns false
        // and starts a reload; the round count must be untouched by it.
        let mut h = PlayHud::default();
        h.ammo = Some(Ammo { loaded: 0, ..a.clone() });
        assert!(!h.before_fire(), "an empty magazine refuses the pull");
        assert_eq!(h.ammo.as_ref().unwrap().loaded, 0, "and spends nothing");
        assert_eq!(h.dry_pulls, 1);
        assert!(h.reloading.is_some(), "an empty pull starts the reload the engine would");
        // ...and a full one allows it and still spends nothing: the SHOT spends the round.
        let mut h2 = PlayHud::default();
        h2.ammo = Some(a.clone());
        assert!(h2.before_fire());
        assert_eq!(h2.ammo.as_ref().unwrap().loaded, 28, "before_fire never writes the magazine");
        // With no magazine at all the gate must not block firing — an unauthored weapon still
        // shoots, it just has no HUD number.
        let mut h3 = PlayHud::default();
        assert!(h3.before_fire(), "no ammo model is not the same as an empty magazine");
    }

    /// **THE RELOAD-CLOCK GATE.** A reload takes time, fills on completion and not before, and
    /// an empty magazine with rounds in reserve reloads on its own — which is what the engine does
    /// the moment the last round leaves the barrel.
    #[test]
    fn a_reload_fills_the_magazine_when_it_finishes_and_not_before() {
        let mut h = PlayHud { reload_duration: 1.0, reload_clip: "reload-empty".into(), ..Default::default() };
        h.ammo = Some(Ammo {
            weapon: 7,
            mag: ar_like(),
            loaded: 0,
            reserve: Some(256),
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        });
        // The auto-reload fires on the first tick, because the magazine is dry and the reserve is not.
        h.tick(0.1, 0);
        assert!(h.reloading.is_some(), "a dry magazine reloads itself");
        assert_eq!(h.ammo.as_ref().unwrap().loaded, 0, "and nothing has arrived yet");
        h.tick(0.5, 0);
        assert_eq!(h.ammo.as_ref().unwrap().loaded, 0, "still mid-reload at 0.6 s of 1.0");
        h.tick(0.5, 0);
        assert_eq!(h.ammo.as_ref().unwrap().loaded, 32, "finished");
        assert_eq!(h.ammo.as_ref().unwrap().reserve, Some(224), "and the reserve paid for 32");
        assert!(h.reloading.is_none());
        assert_eq!(h.ammo.as_ref().unwrap().reloads, 1);
        // With the reserve exhausted the magazine stays as it is and no reload spins for ever.
        let mut e = PlayHud { reload_duration: 1.0, ..Default::default() };
        e.ammo = Some(Ammo {
            weapon: 7,
            mag: ar_like(),
            loaded: 0,
            reserve: Some(0),
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        });
        for _ in 0..20 {
            e.tick(0.1, 0);
        }
        assert!(e.reloading.is_none(), "no reserve, no reload");
        assert_eq!(e.ammo.as_ref().unwrap().loaded, 0);
        // A BATTERY WEAPON takes nothing out of a reserve: `rounds recharged:per second`.
        let mut b = PlayHud { reload_duration: 1.0, ..Default::default() };
        b.ammo = Some(Ammo {
            weapon: 7,
            mag: Magazine { rounds_recharged: 4, rounds_loaded_maximum: 100, rounds_reloaded: 0, ..ar_like() },
            loaded: 50,
            reserve: Some(0),
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        });
        b.tick(0.5, 0);
        assert_eq!(b.ammo.as_ref().unwrap().loaded, 52, "4 per second for half a second");
        assert_eq!(b.ammo.as_ref().unwrap().reserve, Some(0), "and the reserve is untouched");
        assert_eq!(b.ammo.as_ref().unwrap().reloads, 0, "a recharge is not a reload");
    }

    /// **THE PROVENANCE GATE, and it is the one that matters most.**
    ///
    /// In Theater these fields are `Absent` and are not drawn. In test mode they are STATED. What
    /// must not happen is the third thing: a plausible middle value for something HMS cannot
    /// source. So every field this builds is checked both ways — present when there is a model,
    /// and `Absent` (not zero, not full) when there is not.
    #[test]
    fn an_unsourced_field_stays_absent_and_is_not_drawn() {
        use crate::reach_hud::{CollectionValues, Prov};
        // Nothing bound at all: every field absent, and the meters refuse to draw.
        let empty = PlayHud::default();
        let v = empty.values(0, None, None);
        assert_eq!(v.loaded.prov, Prov::Absent);
        assert_eq!(v.shield.prov, Prov::Absent);
        assert_eq!(v.health.prov, Prov::Absent);
        assert_eq!(v.frags.prov, Prov::Absent);
        assert_eq!(v.plasmas.prov, Prov::Absent);
        assert_eq!(v.loaded_max, None);
        {
            let cv = CollectionValues { v: &v };
            // `Some(None)` is "it is a meter and I have no level": drawn neither full nor empty.
            assert_eq!(cv.meter("ammo_area", "meter"), Some(None));
            assert_eq!(cv.meter("shield_meter", "meter"), Some(None));
            assert_eq!(cv.meter("health_meter", "health_meter"), Some(None));
            assert_eq!(cv.number("ammo_area", "ammo_count", 0), None);
        }
        // A bound magazine: the number and the denominator are both STATED, so the meter draws.
        let mut h = PlayHud::default();
        h.ammo = Some(Ammo {
            weapon: 7,
            mag: ar_like(),
            loaded: 24,
            reserve: Some(200),
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        });
        let v = h.values(7, None, None);
        assert_eq!(v.loaded.value, Some(24));
        assert_eq!(v.loaded.prov, Prov::Recorded, "test mode OWNS this number");
        assert_eq!(v.loaded_max, Some(32), "the authored `rounds loaded maximum`");
        assert_eq!(v.reserve.value, Some(200));
        {
            let cv = CollectionValues { v: &v };
            assert_eq!(cv.fill("ammo_area", "meter"), Some(24.0 / 32.0));
            // The ammo collection names are NOT uniform and are matched by prefix.
            assert_eq!(cv.fill("ammo_area_right", "meter"), Some(24.0 / 32.0));
            assert_eq!(cv.number("ammo_area_right", "ammo_count", 0), Some(24));
            assert_eq!(cv.number("ammo_area", "reserve_count", 0), Some(200));
        }
        // A weapon whose `rounds total` reading is indeterminate: the MAGAZINE still draws and the
        // RESERVE does not. Two fields, two answers, which is the whole point of `Field`.
        let mut ind = PlayHud::default();
        ind.ammo = Some(Ammo {
            weapon: 7,
            mag: ar_like(),
            loaded: 24,
            reserve: None,
            shots_seen: 0,
            dry_pulls: 0,
            reloads: 0,
        });
        let v = ind.values(7, None, None);
        assert_eq!(v.loaded.value, Some(24), "the magazine is still known");
        assert_eq!(v.reserve.prov, Prov::Absent, "the reserve is not");
        assert!(ind.report().iter().any(|l| l.contains("matches neither reading")), "{:?}", ind.report());
        // GRENADES: the authored type is stated and the OTHER type is absent, never zero.
        let mut g = PlayHud::default();
        g.grenades = Grenades { kind: Some(GRENADE_TYPE_FRAG), count: 2, maxima: vec![(3, "frag".into()), (3, "plasma".into())] };
        let v = g.values(0, None, None);
        assert_eq!(v.frags.value, Some(2));
        assert_eq!(v.frags.prov, Prov::Recorded, "the authored type is stated");
        // THE OTHER TYPE IS A DERIVED ZERO, not absent and not recorded. `unit` authors one type
        // and one count, so "none of any other" is what the pair MEANS — and the provenance has to
        // say it was read off that shape rather than out of a field of its own.
        assert_eq!(v.plasmas.value, Some(0));
        assert_eq!(v.plasmas.prov, Prov::Derived, "zero is DERIVED from the singular authored pair");
        assert_eq!(g.grenades.count_of(GRENADE_TYPE_PLASMA), Some(0));
        // ...and a biped that authors NO grenade type keeps BOTH fields absent, because then
        // nothing is known and zero really would be a claim.
        let silent = PlayHud::default();
        assert_eq!(silent.grenades.field(GRENADE_TYPE_FRAG).prov, Prov::Absent);
        assert_eq!(silent.grenades.field(GRENADE_TYPE_PLASMA).prov, Prov::Absent);
        assert!(g.grenades.describe().contains("DERIVED zero"), "{}", g.grenades.describe());
        // THE CLAMP. `forge_halo`'s Elite authors `grenade count` 4 while `matg` authors a
        // `maximum count` of 2, and 2 is what Reach gives a player — so the global ceiling wins and
        // the conflict is PRINTED rather than resolved silently.
        let over = Grenades {
            kind: Some(GRENADE_TYPE_FRAG),
            count: 4,
            maxima: vec![(2, "frag".into()), (2, "plasma".into())],
        };
        assert_eq!(over.count_of(GRENADE_TYPE_FRAG), Some(2), "clamped to matg's maximum");
        assert_eq!(over.count, 4, "and the authored value is kept, not overwritten");
        assert!(over.clamped());
        assert!(over.describe().contains("CLAMPED"), "{}", over.describe());
        assert_eq!(over.values_frags(), Some(2));
        // Under the ceiling, nothing is clamped and nothing is reported as clamped.
        let under = Grenades { count: 1, ..over.clone() };
        assert_eq!(under.count_of(GRENADE_TYPE_FRAG), Some(1));
        assert!(!under.clamped() && !under.describe().contains("CLAMPED"), "{}", under.describe());
        // No matg block at all: the biped's own number stands, because there is no ceiling to
        // clamp against and inventing one would be worse than honouring the only authored value.
        let nomax = Grenades { kind: Some(GRENADE_TYPE_FRAG), count: 4, maxima: Vec::new() };
        assert_eq!(nomax.count_of(GRENADE_TYPE_FRAG), Some(4));
        assert!(!nomax.clamped());
        assert_eq!(nomax.maximum_of(GRENADE_TYPE_FRAG), None);
        // A biped that authors no grenades at all: nothing is stated for either type.
        let none = PlayHud::default();
        assert_eq!(none.grenades.count_of(GRENADE_TYPE_FRAG), None, "no authored type states nothing");
        assert_eq!(none.grenades.count_of(GRENADE_TYPE_PLASMA), None, "...for EITHER type");
        assert_eq!(none.grenades.field(GRENADE_TYPE_FRAG).prov, Prov::Absent);
        // THE VEHICLE and the BUTTON LABEL ride through untouched.
        let v = empty.values(0, Some(("ui\\chud\\warthog".into(), 0)), Some("E".into()));
        assert_eq!(v.vehicle, Some(("ui\\chud\\warthog".into(), 0)));
        assert_eq!(v.button_label.as_deref(), Some("E"));
    }

    /// **THE VITALITY GATE.** The shield absorbs first and health takes the remainder, the
    /// fractions are of the sections' OWN authored pools, and a model this file cannot classify
    /// leaves the meter absent rather than guessing a section.
    #[test]
    fn the_shield_absorbs_before_health_and_an_unclassifiable_model_is_refused() {
        use crate::reach_damage::{DamageModel, DamageSection};
        let sec = |index: usize, name: &str, pct: f32, shield: bool, max: f32| DamageSection {
            index,
            name: name.into(),
            flags: 0,
            vitality_percentage: pct,
            vitality: max * pct,
            is_shield: shield,
            recharge_time: 0.0,
            responses: Vec::new(),
        };
        // A Spartan-shaped model: one shield section and one body section.
        let model = DamageModel {
            hlmt: 1,
            maximum_vitality: 100.0,
            indirect_section: Some(1),
            shielded_section: Some(0),
            sections: vec![sec(0, "shield", 0.7, true, 100.0), sec(1, "body", 0.3, false, 100.0)],
        };
        let mut v = Vitality {
            model: model.clone(),
            shield_section: Some(0),
            health_section: Some((1, HealthRule::OnlySection)),
            shield: 0.0,
            health: 0.0,
            notes: Vec::new(),
        };
        v.restore();
        // `100.0 * 0.3` is 30.000002 in f32, so the shares are compared with slack — one part in
        // 10^5 of the pool, far tighter than any difference an artist could author.
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(near(v.shield, 70.0) && near(v.health, 30.0), "the authored shares of 100: {v:?}");
        assert_eq!(v.shield_fraction(), Some(1.0));
        assert_eq!(v.health_fraction(), Some(1.0));
        assert!(v.alive());
        // Half the shield.
        let (s0, h0) = v.hurt(35.0);
        assert!(near(s0, 35.0) && h0 == 0.0, "half the shield: {s0} / {h0}");
        assert!(near(v.shield_fraction().unwrap(), 0.5), "{:?}", v.shield_fraction());
        assert_eq!(v.health_fraction(), Some(1.0), "health is untouched while the shield holds");
        // Through the shield and into health.
        let (s1, h1) = v.hurt(45.0);
        assert!(near(s1, 35.0) && near(h1, 10.0), "through the shield: {s1} / {h1}");
        assert_eq!(v.shield, 0.0);
        assert!((v.health_fraction().unwrap() - 20.0 / 30.0).abs() < 1e-4, "{:?}", v.health_fraction());
        assert!(v.alive());
        // Dead, and not negative.
        let (s2, h2) = v.hurt(999.0);
        assert!(s2 == 0.0 && near(h2, 20.0), "the rest of health: {s2} / {h2}");
        assert_eq!((v.shield, v.health), (0.0, 0.0));
        assert!(!v.alive(), "an empty health pool is dead");
        assert_eq!(v.health_fraction(), Some(0.0), "and the meter draws EMPTY, not absent");
        // Zero damage changes nothing.
        v.restore();
        assert_eq!(v.hurt(0.0), (0.0, 0.0));
        assert_eq!(v.hurt(-5.0), (0.0, 0.0));
        // THE MULTI-SHIELD CASE, which is every playable Reach biped: several sections marked as
        // shields, `shielded state damage section` UNSET, and all of them authoring the same pool.
        // The rule must resolve it — and must refuse when they disagree, which is the branch a
        // rule that just took the first shield section would never show.
        let spartan_like = |pcts: &[f32]| DamageModel {
            hlmt: 3,
            maximum_vitality: 70.0,
            indirect_section: Some(0),
            shielded_section: None,
            sections: vec![
                sec(0, "body", 0.6428571, false, 70.0),
                sec(1, "head", 1.0, false, 70.0),
                sec(2, "shield", pcts[0], true, 70.0),
                sec(3, "armor_lock_shield_loop", pcts[1], true, 70.0),
                sec(4, "armor_lock_shield_post", pcts[2], true, 70.0),
            ],
        };
        // All three shields agree -> resolved, and recorded as resolved by agreement.
        let ok = Vitality {
            model: spartan_like(&[1.0, 1.0, 1.0]),
            shield_section: None,
            health_section: None,
            shield: 0.0,
            health: 0.0,
            notes: Vec::new(),
        };
        let classify = Vitality::from_model;
        let resolved = classify(ok.model.clone());
        assert_eq!(resolved.shield_section, Some(2), "the first shield-marked section answers");
        assert!(near(resolved.shield_max().unwrap(), 70.0), "{:?}", resolved.shield_max());
        assert_eq!(resolved.health_section.map(|(i, r)| (i, r)), Some((0, HealthRule::IndirectSection)));
        assert!(near(resolved.health_max().unwrap(), 45.0), "{:?}", resolved.health_max());
        assert!(resolved.notes.iter().any(|n| n.contains("author the same pool")), "{:?}", resolved.notes);
        // One of them DISAGREES -> refused, with a reason. Taking the first section regardless
        // would have printed 70 on a model where the shield is genuinely ambiguous.
        let bad = classify(spartan_like(&[1.0, 0.5, 1.0]));
        assert_eq!(bad.shield_section, None, "disagreeing shield pools must be refused");
        assert_eq!(bad.shield_fraction(), None, "and the meter not drawn");
        assert!(bad.notes.iter().any(|n| n.contains("DIFFERENT pools")), "{:?}", bad.notes);
        // ...while health still resolves, because the two questions are independent.
        assert!(bad.health_max().is_some(), "a refused shield must not take health with it");
        // And when the damage info DOES name a shielded section, that wins outright — no
        // agreement check is needed and a disagreement is irrelevant.
        let mut named = spartan_like(&[1.0, 0.5, 1.0]);
        named.shielded_section = Some(3);
        let n = classify(named);
        assert_eq!(n.shield_section, Some(3), "the damage info's own choice wins");
        assert!(near(n.shield_max().unwrap(), 35.0), "{:?}", n.shield_max());

        // AN UNCLASSIFIABLE MODEL IS REFUSED, not guessed. Three non-shield sections and an
        // `indirect damage section` that names none of them.
        let messy = Vitality {
            model: DamageModel {
                hlmt: 2,
                maximum_vitality: 500.0,
                indirect_section: None,
                shielded_section: None,
                sections: vec![
                    sec(0, "a", 0.4, false, 500.0),
                    sec(1, "b", 0.3, false, 500.0),
                    sec(2, "c", 0.3, false, 500.0),
                ],
            },
            shield_section: None,
            health_section: None,
            shield: 0.0,
            health: 0.0,
            notes: vec!["unclassifiable".into()],
        };
        assert_eq!(messy.shield_fraction(), None, "no shield section: the meter is not drawn");
        assert_eq!(messy.health_fraction(), None, "and nor is health");
        let mut h = PlayHud::default();
        h.vitality = Some(messy);
        let vals = h.values(0, None, None);
        assert_eq!(vals.shield.prov, crate::reach_hud::Prov::Absent);
        assert_eq!(vals.health.prov, crate::reach_hud::Prov::Absent);
        assert!(
            h.report().iter().any(|l| l.contains("no damage section could be classified")),
            "{:?}",
            h.report()
        );
    }

    /// **THE OFFSETS, pinned as constants.** The recovery is the finding; a typo in it is silent.
    #[test]
    fn the_recovered_magazine_offsets_are_the_ones_the_engine_reads() {
        // The two the engine reads by hand in `weapon_magazine_execute_reload_networking`.
        assert_eq!(MAG_ROUNDS_RELOADED, 0x0E, "*(i16 *)(element + 0x0E)");
        assert_eq!(MAG_ROUNDS_LOADED_MAXIMUM, 0x0A, "*(i16 *)(element + 0x0A)");
        assert_eq!(MAG_FLAGS, 0x00, "*(u8 *)element & 1 -- `wastes rounds when reloaded`");
        // The stride, and the candidate it beat.
        assert_eq!(MAGAZINE_SIZE, 96, "weapon_magazine_fields_v2");
        assert_eq!(MAGAZINE_SIZE_V1, 108, "weapon_magazine_fields_v1");
        assert_ne!(MAGAZINE_SIZE, MAGAZINE_SIZE_V1, "a test between two identical strides proves nothing");
        assert_eq!(MAGAZINE_SIZE_CANDIDATES[0], MAGAZINE_SIZE);
        // The v2 layout has to ADD UP to its own element size, which is the check that catches a
        // field written down at the wrong offset: 12 shorts-and-a-real of header, four 16-byte tag
        // references and a 12-byte block.
        assert_eq!(MAG_RELOADING_EFFECT, 0x14);
        assert_eq!(MAG_RELOADING_DAMAGE_EFFECT, MAG_RELOADING_EFFECT + 0x10);
        assert_eq!(MAG_CHAMBERING_EFFECT, MAG_RELOADING_DAMAGE_EFFECT + 0x10);
        assert_eq!(MAG_CHAMBERING_DAMAGE_EFFECT, MAG_CHAMBERING_EFFECT + 0x10);
        assert_eq!(MAG_MAGAZINE_OBJECTS, MAG_CHAMBERING_DAMAGE_EFFECT + 0x10);
        assert_eq!(MAG_MAGAZINE_OBJECTS + 12, MAGAZINE_SIZE, "the element ends on the block");
        // ...and the v1 offsets that DISAGREE, which is what makes the map-backed gate able to
        // tell the two apart at all.
        assert_ne!(MAG_V1_ROUNDS_RELOADED, MAG_ROUNDS_RELOADED);
        assert_eq!(MAG_V1_MAGAZINE_OBJECTS, MAGAZINE_SIZE, "v1's block starts where a v2 element ENDS");
        assert_eq!(MAG_V1_MAGAZINE_OBJECTS + 12, MAGAZINE_SIZE_V1);
        // The unit fields, which are `tagdefs`'s `?` offsets plus the settled 0x98.
        assert_eq!(UNIT_GRENADE_TYPE, 0x384 + 0x98);
        assert_eq!(UNIT_GRENADE_COUNT, 0x386 + 0x98);
        assert_eq!(
            UNIT_GRENADE_COUNT + 2,
            0x420,
            "`powered seats` starts right after, which is what `reach_seats` places at +0x420"
        );
        assert!(
            UNIT_GRENADE_TYPE < crate::reach_seats::UNIT_SEATS,
            "the grenade fields come before the seats block in `unit`"
        );
    }

    /// **THE MAP-BACKED GATE: the magazine layout is v2, and v1 is rejected.**
    ///
    /// The static evidence (three independent routes, see the module comment) says 96 bytes. This
    /// is the measurement that makes it a fact about the shipped tags rather than about the
    /// binary, and it tests the places the two candidates DISAGREE — because element 0 reads
    /// `rounds loaded maximum` identically under both, and a test that only read element 0's first
    /// twelve bytes would pass on either and prove nothing. That is the exact trap that put
    /// `weapon_barrels` at the wrong stride for 47 of 52 references.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_magazine_layout_is_v2_against_the_shipped_tags() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_hud gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut authored = 0usize;
        let mut v2_refs = 0usize;
        let mut v1_refs = 0usize;
        let mut plausible_v2_time = 0usize;
        let mut plausible_v1_time = 0usize;
        let mut element1 = 0usize;
        let mut element1_ok = 0usize;
        let mut sane_magazines = 0usize;
        for t in native.tags_by_group(cache, "weap") {
            let Some((n, _)) = Magazine::block(native, cache, t) else { continue };
            let name = native.tag_name(cache, t).unwrap_or_default();
            authored += 1;
            let Some(e) = Magazine::raw_at(native, cache, t, 0, MAGAZINE_SIZE) else { continue };
            let m = Magazine::decode(0, &e).expect("a 96-byte element decodes");
            // THE VALUES SOMEBODY WHO HAS PLAYED REACH WOULD PREDICT. This is the corroboration
            // the project's own RE notes call the strongest available check on a recovered layout.
            if m.rounds_loaded_maximum > 0 {
                sane_magazines += 1;
                assert!(
                    m.rounds_loaded_maximum <= ROUNDS_LOADED_MASK,
                    "{name}: `rounds loaded maximum` {} is past the engine's own 9-bit ceiling, so \
                     this is not that field",
                    m.rounds_loaded_maximum
                );
                assert!(
                    m.rounds_total_maximum >= m.rounds_loaded_maximum,
                    "{name}: total {} < magazine {}",
                    m.rounds_total_maximum,
                    m.rounds_loaded_maximum
                );
                assert!(
                    m.rounds_total_initial <= m.rounds_total_maximum,
                    "{name}: initial {} > maximum {}",
                    m.rounds_total_initial,
                    m.rounds_total_maximum
                );
            }
            eprintln!("play_hud: {:<34} {}", name.rsplit('\\').next().unwrap_or(&name), m.describe());
            // THE DISCRIMINATOR. v2 puts a tag reference at +0x14 where v1 puts `rounds reloaded`
            // and a 2-byte pad; v2 puts a `real` at +0x10 where v1 puts the tail of an 8-byte pad.
            let f = |o: usize| f32::from_le_bytes(e[o..o + 4].try_into().unwrap());
            let datum = |o: usize| u32::from_le_bytes(e[o + 0x0C..o + 0x10].try_into().unwrap());
            let plausible = |t: f32| t.is_finite() && t > 0.0 && t < 30.0;
            for o in [MAG_RELOADING_EFFECT, MAG_RELOADING_DAMAGE_EFFECT] {
                let d = datum(o);
                if d != u32::MAX && d != 0 && native.tag_name(cache, d & 0xFFFF).is_some() {
                    v2_refs += 1;
                }
            }
            for o in [MAG_CHAMBERING_EFFECT, MAG_CHAMBERING_DAMAGE_EFFECT] {
                let d = datum(o);
                if d != u32::MAX && d != 0 && native.tag_name(cache, d & 0xFFFF).is_some() {
                    v1_refs += 1;
                }
            }
            if plausible(f(MAG_RELOAD_DIALOGUE_TIME)) {
                plausible_v2_time += 1;
            }
            if plausible(f(MAG_V1_CHAMBER_TIME)) {
                plausible_v1_time += 1;
            }
            // ELEMENT 1, the only thing that can test a stride directly.
            if n > 1 {
                element1 += 1;
                let a = Magazine::raw_at(native, cache, t, 1, MAGAZINE_SIZE)
                    .and_then(|b| Magazine::decode(1, &b));
                let b = Magazine::raw_at(native, cache, t, 1, MAGAZINE_SIZE_V1)
                    .and_then(|b| Magazine::decode(1, &b));
                eprintln!("play_hud:   {name} authors TWO magazines; element 1 at 96 = {a:?}, at 108 = {b:?}");
                if a.is_some_and(|m| m.rounds_loaded_maximum > 0 && m.rounds_loaded_maximum <= ROUNDS_LOADED_MASK) {
                    element1_ok += 1;
                }
            }
        }
        eprintln!(
            "play_hud: {authored} weapon(s) author a magazines block ({sane_magazines} with a \
             non-zero magazine); {element1} author two ({element1_ok} of which read sanely at 96). \
             v2-only tag references at +0x14/+0x24: {v2_refs}. Shared references at +0x34/+0x44: \
             {v1_refs}. Plausible `reload dialogue time` at +0x10: {plausible_v2_time}. Plausible \
             v1 `chamber time` at +0x18: {plausible_v1_time}."
        );
        assert!(
            authored >= 15 && sane_magazines >= 12,
            "{authored} weapons author a magazines block and {sane_magazines} have a non-zero \
             magazine — +{WEAP_MAGAZINES:#05x} is not the magazines block"
        );

        // ---- WHAT THIS GATE CAN AND CANNOT DISCRIMINATE, stated rather than implied -------
        //
        // **Three KNOWN ZEROS.** Every field where the two candidate layouts disagree reads zero
        // on every shipped Reach weapon: no magazine authors a reloading or chambering effect, and
        // neither candidate's time field is populated. So none of those is a discriminator HERE,
        // and asserting "the v1 reading is implausible" would have PASSED on zero evidence — a
        // vacuous assertion dressed as a measurement. They are asserted as zeros instead, so the
        // day a modded or later cache authors one, this reddens and the discrimination becomes
        // available for real.
        assert_eq!(v2_refs, 0, "a shipped Reach magazine authors no reloading effect; if one now does, +0x14/+0x24 can discriminate and this gate should start using it");
        assert_eq!(v1_refs, 0, "...and none authors a chambering effect either");
        assert_eq!(plausible_v2_time, 0, "no magazine authors a `reload dialogue time`");
        assert_eq!(plausible_v1_time, 0, "...and none reads a plausible v1 `chamber time` either");
        // **And element 1 is unavailable**: not one shipped weapon authors two magazines, so the
        // stride cannot be tested the way `weapon_barrels` was. That is why the stride rests on the
        // BINARY — `0x60` is a literal argument to the tag-block element accessor in three separate
        // engine functions, and the field-set installer copies v2's record into place — and this
        // asserts the zero so nobody later believes the cache settled it.
        assert_eq!(
            element1, 0,
            "a weapon now authors TWO magazines: element 1 can finally discriminate 96 from 108, \
             and this gate should be extended to do it"
        );
        assert_eq!(element1_ok, element1);

        // ---- THE CORROBORATION THAT DOES BITE: the authored VALUES ------------------------
        //
        // The project's own RE notes call this the strongest available check on a layout recovered
        // by content — "a recovered layout should read out numbers somebody who has played the game
        // would predict". These are Reach's published magazine and reserve sizes, taken from the
        // GAME and not from this file: read at the wrong stride or the wrong offset, not one of
        // them lands. `rounds total maximum` must also equal magazine + reserve maximum, which is
        // an identity the layout either satisfies everywhere or nowhere.
        let expect: [(&str, i32, i32, i32); 7] = [
            // (weapon, rounds loaded maximum, reserve maximum, rounds reloaded)
            ("assault_rifle", 32, 288, 32),
            ("magnum", 8, 40, 8),
            ("dmr", 15, 60, 15),
            ("sniper_rifle", 4, 20, 4),
            ("rocket_launcher", 2, 6, 2),
            ("needler", 24, 96, 24),
            // The shotgun is the one that cannot be a coincidence: it reloads ONE shell at a time,
            // and `rounds reloaded` has to read 1 while its magazine reads 6.
            ("shotgun", 6, 30, 1),
        ];
        let mut checked = 0usize;
        for (want, mag, reserve, reloaded) in expect {
            let Some(t) = native
                .tags_by_group(cache, "weap")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(&format!("\\{want}"))))
            else {
                continue;
            };
            let m = Magazine::read(native, cache, t, 0).expect("a player weapon authors a magazine");
            checked += 1;
            eprintln!("play_hud: {want:<16} expected {mag}/{reserve} reload {reloaded}, read {}", m.describe());
            assert_eq!(m.rounds_loaded_maximum, mag, "{want}: magazine size");
            assert_eq!(m.reserve_max(), reserve, "{want}: reserve maximum");
            assert_eq!(m.rounds_reloaded, reloaded, "{want}: rounds moved by one reload");
            // THE IDENTITY: `rounds total maximum` is the magazine plus the reserve ceiling, which
            // is also what makes `total_counts_the_magazine` answer `true` rather than guessing.
            assert_eq!(
                m.rounds_total_maximum,
                mag + reserve,
                "{want}: `rounds total maximum` must be the magazine plus the reserve ceiling"
            );
            assert_eq!(m.total_counts_the_magazine(), Some(true), "{want}: `total` counts the magazine");
            // ...and the spawn reserve the HUD prints follows from it.
            let a = Ammo::new(native, cache, t, 0).expect("an ammo model");
            assert_eq!(a.loaded, mag, "{want}: a drawn weapon starts with a full magazine");
            assert_eq!(
                a.reserve,
                Some(m.rounds_total_initial - mag),
                "{want}: the spawn reserve is `rounds total initial` less the magazine"
            );
        }
        assert_eq!(checked, expect.len(), "only {checked} of the named player weapons were found");
    }

    /// #hud-trace  **THE HEADLESS PICTURE**, printed. Not an assertion-heavy gate: it runs the
    /// real painter and prints what every widget resolved to, which is the only way to inspect a
    /// HUD that cannot be captured. The assertions are the few things that are absurd whatever the
    /// art says.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn what_the_painter_emits_for_every_widget() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_hud gates SKIPPED: no forge_halo cache");
            return;
        };
        dump_painted_hud(&native, cache, "assault_rifle");
        dump_hud_art_audit(&native, cache);
    }

    /// **THE PLAYER-VITALITY GATE.** A Spartan and an Elite must each resolve a shield pool and a
    /// health pool out of their own `hlmt`, by the refusal ladder and not by position.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn a_playable_biped_authors_a_shield_and_a_health_pool() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_hud gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut found = 0usize;
        for b in native.tags_by_group(cache, "bipd") {
            let name = native.tag_name(cache, b).unwrap_or_default();
            let lower = name.to_ascii_lowercase();
            // The PLAYABLE bipeds only. `hologram_spartan` and `hologram_elite` are the holographic
            // decoy's own bodies: one section, marked as a shield, no body section at all — a
            // perfectly correct model for a decoy and not a thing a player can be.
            if !lower.starts_with("objects\\characters\\") {
                continue;
            }
            if !(lower.contains("spartan") || lower.contains("elite")) {
                continue;
            }
            let Some(mut v) = Vitality::read(native, cache, b) else {
                eprintln!("play_hud: {name} has no usable damage info");
                continue;
            };
            v.restore();
            found += 1;
            // The WHOLE authored model first, so a classification failure can be read rather than
            // guessed at: every section, its share, its pool and whether its `shield material
            // name` is set.
            for l in v.model.describe() {
                eprintln!("play_hud: {name}: {l}");
            }
            for l in v.describe() {
                eprintln!("play_hud: {name}: {l}");
            }
            assert!(
                v.model.maximum_vitality > 0.0,
                "{name}: the engine's own gate is `maximum vitality > 0`"
            );
            assert!(
                v.shield_max().is_some(),
                "{name}: no shield pool resolved — the biped marks {} section(s) as shields",
                v.model.sections.iter().filter(|s| s.is_shield).count()
            );
            assert!(v.health_max().is_some(), "{name}: no section could be classified as health");
            assert_eq!(v.shield_fraction(), Some(1.0), "{name}: a restored shield is full");
            assert_eq!(v.health_fraction(), Some(1.0));
            assert!(v.alive());
            // The pools are DIFFERENT sections: a model where one section answered both would make
            // the shield and the health meter the same bar.
            assert_ne!(
                v.shield_section,
                v.health_section.map(|(i, _)| i),
                "{name}: the shield and the health pool resolved to the SAME damage section"
            );
            // REACH'S OWN PUBLISHED NUMBERS, taken from the game and not from this file: a
            // Spartan is 70 shield / 45 health and an Elite 79 / 50. Read at a wrong offset, a
            // wrong section or a wrong share, not one of the four lands.
            let (want_shield, want_health) =
                if lower.contains("elite") { (79.0f32, 50.0f32) } else { (70.0f32, 45.0f32) };
            assert!(
                (v.shield_max().unwrap() - want_shield).abs() < 0.5,
                "{name}: shield pool reads {:.1}, and Reach's own is {want_shield}",
                v.shield_max().unwrap()
            );
            assert!(
                (v.health_max().unwrap() - want_health).abs() < 0.5,
                "{name}: health pool reads {:.1}, and Reach's own is {want_health}",
                v.health_max().unwrap()
            );
            // The shield pool IS the model's own `maximum vitality` — every shield-marked section
            // authors a 100% share — and the health pool is the body's share of it.
            assert!(
                (v.model.maximum_vitality - want_shield).abs() < 0.5,
                "{name}: `maximum vitality` is {:.1}, not the shield pool",
                v.model.maximum_vitality
            );
            // THE RULE ACTUALLY EXERCISED: more than one section is marked as a shield, the damage
            // info names none of them, and they agree. If a cache ever stops agreeing this reddens.
            let marked = v.model.sections.iter().filter(|s| s.is_shield).count();
            assert!(marked >= 3, "{name}: only {marked} section(s) marked as shields — the ability shields are gone");
            assert_eq!(v.model.shielded_section, None, "{name}: `shielded state damage section` is unset on both playable bipeds");
            let pools: Vec<f32> =
                v.model.sections.iter().filter(|s| s.is_shield).map(|s| s.vitality).collect();
            assert!(
                pools.windows(2).all(|w| (w[0] - w[1]).abs() < 0.5),
                "{name}: the shield-marked sections author DIFFERENT pools {pools:?} — the \
                 agreement the classification rests on has gone, and the meter must be refused"
            );
            assert!(
                v.notes.iter().any(|n| n.contains("author the same pool")),
                "{name}: the rule must record that it resolved by checked agreement: {:?}",
                v.notes
            );
            assert_eq!(
                v.health_section.map(|(_, r)| r),
                Some(HealthRule::IndirectSection),
                "{name}: health must come from the damage info's own `indirect damage section`"
            );
            // And the shares add up to the model's own maximum vitality, which is what
            // `vitality percentage` means ("percentage of total object vitality").
            //
            // **NOT a sum.** The shares do NOT add to 1.0 here and must not be asserted to: a
            // Spartan's five sections author 0.64 + 1.0 + 1.0 + 1.0 + 1.0. They are each a share
            // of the whole, not a partition of it — `head` is 100% because a headshot is lethal
            // against the whole pool, and the three shield sections are 100% because they are the
            // same shield seen three ways. What IS asserted is that every share is a FRACTION, in
            // (0, 1], which is the property `vitality percentage:[0.1]` actually claims.
            for sec in &v.model.sections {
                assert!(
                    sec.vitality_percentage > 0.0 && sec.vitality_percentage <= 1.0001,
                    "{name}: section '{}' authors a share of {:.3}, which is not a fraction — so \
                     either +0x08 is not `vitality percentage` or it is not a share at all",
                    sec.name,
                    sec.vitality_percentage
                );
                assert!(
                    (sec.vitality - v.model.maximum_vitality * sec.vitality_percentage).abs() < 0.01,
                    "{name}: section '{}' pool {:.2} is not its share of the model's maximum",
                    sec.name,
                    sec.vitality
                );
            }
            // The whole live model, exercised against authored numbers.
            let shield = v.shield_max().unwrap();
            let (took, spill) = v.hurt(shield);
            assert!(
                (took - shield).abs() < 1e-3 && spill == 0.0,
                "{name}: the shield must absorb its own pool exactly; it took {took} of {shield} \
                 and spilled {spill} onto health"
            );
            assert_eq!(v.shield_fraction(), Some(0.0));
            assert_eq!(v.health_fraction(), Some(1.0), "{name}: and health is untouched");
            // Grenades, read off the same biped.
            let g = Grenades::read(native, cache, b);
            eprintln!("play_hud: {name}: {}", g.describe());
            assert!(!g.maxima.is_empty(), "matg authors no grenades block");
            assert!(
                g.maxima.iter().any(|(m, _)| *m > 0),
                "every matg grenade type authors a maximum count of 0"
            );
            // The enum ORDER is read, not assumed: element 0's projectile must be the frag and
            // element 1's the plasma, which is what `GRENADE_TYPE_FRAG`/`_PLASMA` claim.
            assert!(
                g.maxima[0].1.contains("frag"),
                "`global_grenade_type_enum` 0 names '{}', not the frag grenade",
                g.maxima[0].1
            );
            if let Some((_, p1)) = g.maxima.get(1) {
                assert!(p1.contains("plasma"), "enum 1 names '{p1}', not the plasma grenade");
            }
            // The biped authors ONE type, and the other type's count stays unknown.
            let k = g.kind.unwrap_or_else(|| panic!("{name} authors no grenade type"));
            assert_eq!(k, GRENADE_TYPE_FRAG, "{name}: a playable biped starts on frags");
            assert_eq!(g.count_of(GRENADE_TYPE_PLASMA), Some(0), "{name}: no plasmas, derived from the singular authored pair");
            assert_eq!(
                g.field(GRENADE_TYPE_PLASMA).prov,
                crate::reach_hud::Prov::Derived,
                "{name}: and it must be labelled DERIVED, not recorded"
            );
            // Reach gives a player TWO grenades, and the drawn count has to be two however the
            // biped's own field reads — `forge_halo`'s Elite authors 4 and matg caps it at 2.
            assert_eq!(
                g.count_of(k),
                Some(2),
                "{name}: the drawn grenade count must be Reach's own 2; biped authors {} and matg \
                 caps at {:?}",
                g.count,
                g.maximum_of(k)
            );
        }
        assert!(found >= 1, "forge_halo carries no Spartan or Elite biped");
    }

    /// **THE AMMO GATE, end to end on real tags.** A real weapon's magazine must build an [`Ammo`]
    /// whose numbers come out of the tag and whose meter has a denominator — which is the field
    /// Theater had to leave `None` and the reason the ammo meter could not be drawn there at all.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn a_real_weapon_builds_a_drawable_ammo_meter() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_hud gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut built = 0usize;
        let mut with_reserve = 0usize;
        for want in ["assault_rifle", "magnum", "dmr", "shotgun", "sniper_rifle", "plasma_pistol"] {
            let Some(t) = native
                .tags_by_group(cache, "weap")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(&format!("\\{want}"))))
            else {
                continue;
            };
            let Some(a) = Ammo::new(native, cache, t, 0) else {
                eprintln!("play_hud: {want} authors no magazine");
                continue;
            };
            built += 1;
            eprintln!(
                "play_hud: {want:<16} {} loaded of {}, reserve {:?}  |  {}",
                a.loaded,
                a.mag.rounds_loaded_maximum,
                a.reserve,
                a.mag.describe()
            );
            assert!(a.can_fire(), "{want}: a freshly drawn weapon has a round in it");
            assert_eq!(a.loaded_fraction(), Some(1.0), "{want}: and a full magazine");
            if a.reserve.is_some() {
                with_reserve += 1;
            }
            // THE DENOMINATOR REACHES THE PAINTER. This is the field the module comment is about.
            let mut h = PlayHud::default();
            h.ammo = Some(a.clone());
            let v = h.values(t, None, None);
            assert_eq!(v.loaded_max, Some(a.mag.rounds_loaded_maximum));
            let cv = crate::reach_hud::CollectionValues { v: &v };
            assert_eq!(cv.fill("ammo_area", "meter"), Some(1.0), "{want}: the meter has a level");
            // Spend the magazine and watch it move; the meter must track it.
            let mut h2 = h.clone();
            let mag = a.mag.rounds_loaded_maximum;
            h2.ammo.as_mut().unwrap().account_shots(1);
            let v2 = h2.values(t, None, None);
            let cv2 = crate::reach_hud::CollectionValues { v: &v2 };
            assert_eq!(cv2.fill("ammo_area", "meter"), Some((mag - 1) as f32 / mag as f32));
        }
        assert!(built >= 3, "only {built} of the named weapons built an ammo model");
        eprintln!("play_hud: {built} weapons built an ammo model, {with_reserve} with a determinate reserve");

        // ---- AND THE DIAGNOSTIC ITSELF RUNS ------------------------------------------------
        //
        // `play_hud_call_site_gates::the_dump_verb_reaches_the_recovered_layout` proves `--dump-hud`
        // still CALLS this, and the gates above prove every number it prints. Neither of them runs
        // the printer, so a slice index in it would reach the user as a crash on a diagnostic flag
        // with every gate green. One call, on a narrow filter so it stays cheap.
        dump_ammo_and_vitality(native, cache, "assault_rifle");
    }
}
