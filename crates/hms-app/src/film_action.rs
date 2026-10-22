//! #film-action  Reach Theater: the SYNCHRONISED ACTION channel, and a unit's object DEFINITION.
//!
//! Two questions this module exists to answer with numbers rather than opinion:
//!
//! 1. **Is the action (assassination, vehicle entry, ...) on the wire?** The engine has an
//!    89-entry object-action enum (`_action_death_ping` = 0 ... `_action_assassinate` = 4 ...
//!    `_action_secondary_weapon_overheat_exit` = 0x58) and a biped carries it as derived
//!    replication field **0x36**, a 7-bit value. The stream states which fields a message
//!    carries *twice* — once as the message's own 64-bit header mask and once as per-field
//!    presence bits — so the first question needs **no field walk at all**: census the header
//!    mask.
//! 2. **Which object definition was an entity spawned from?** `sub_1803D5A8C`, the creation
//!    reader shared by types 11/12/13/14/15/17/18, begins with
//!    `TagReference_BitstreamDecode(group = -1)` — 14 bits at bit offset 0 of every creation
//!    payload. That value is **not** a tag index: `sub_1803CD34C` uses it as `value - 1` into a
//!    flat table of network-referenceable tags (`qword_184E38D48`), built by `sub_1803CD400`
//!    over nine group codes in a fixed order. See `object_ref_table`.
//!
//! Everything here is a probe or a decoder; nothing in it guesses.

use crate::film_decode::{BitReader, EntityMessage};

// ---------------------------------------------------------------------------------------------
// The engine's object-action enum (`_action_*`), read out of the 16-byte-stride name table at
// `haloreach.dll` + 0xA51AA0: {const char* name, u32 priority, u16 flags}. 89 entries, 0..0x58,
// which is why the wire field is 7 bits.
// ---------------------------------------------------------------------------------------------

/// `_action_*` names in enum order. Index == the 7-bit wire value.
pub const ACTION_NAMES: [&str; 89] = [
    "death_ping",                        // 0x00
    "custom_animation",                  // 0x01
    "stop_custom_animation",             // 0x02
    "sync_container",                    // 0x03
    "assassinate",                       // 0x04
    "get_infected",                      // 0x05
    "resurrect",                         // 0x06
    "hard_ping",                         // 0x07
    "ping",                              // 0x08
    "infect",                            // 0x09
    "posture",                           // 0x0A
    "posture_exit",                      // 0x0B
    "vehicle_exit",                      // 0x0C
    "vehicle_exit_and_detach",           // 0x0D
    "vehicle_ejection",                  // 0x0E
    "vehicle_board",                     // 0x0F
    "vehicle_enter",                     // 0x10
    "vehicle_animated_trick",            // 0x11
    "equipment",                         // 0x12
    "equipment_animation",               // 0x13
    "player_dodge",                      // 0x14
    "vehicle_exit_immediate",            // 0x15
    "vehicle_flip",                      // 0x16
    "climb_attach",                      // 0x17
    "climb_detach",                      // 0x18
    "bunker_close",                      // 0x19
    "bunker_open",                       // 0x1A
    "token",                             // 0x1B  (the table's own `action_token`)
    "takeoff",                           // 0x1C
    "perch",                             // 0x1D
    "perch_jump",                        // 0x1E
    "land",                              // 0x1F
    "jump",                              // 0x20
    "animated_jump",                     // 0x21
    "leap_anticipation",                 // 0x22
    "leap",                              // 0x23
    "leap_recovery",                     // 0x24
    "emit_ai",                           // 0x25
    "berserk",                           // 0x26
    "flurry",                            // 0x27
    "deploy_turret",                     // 0x28
    "smash",                             // 0x29
    "vault",                             // 0x2A
    "hoist",                             // 0x2B
    "ai_impulse",                        // 0x2C
    "melee_ai",                          // 0x2D
    "weapon_pickup",                     // 0x2E
    "weapon_trade",                      // 0x2F
    "primary_weapon_drop",               // 0x30
    "secondary_weapon_drop",             // 0x31
    "surge",                             // 0x32
    "phase",                             // 0x33
    "stow",                              // 0x34
    "draw",                              // 0x35
    "juke",                              // 0x36
    "kamikaze",                          // 0x37
    "primary_weapon_switch",             // 0x38
    "secondary_weapon_switch",           // 0x39
    "grenade",                           // 0x3A
    "grenade_pickup",                    // 0x3B
    "equipment_pickup",                  // 0x3C
    "equipment_drop",                    // 0x3D
    "melee_player",                      // 0x3E
    "primary_weapon_primary_reload",     // 0x3F
    "primary_weapon_secondary_reload",   // 0x40
    "secondary_weapon_primary_reload",   // 0x41
    "secondary_weapon_secondary_reload", // 0x42
    "primary_weapon_vent",               // 0x43
    "secondary_weapon_vent",             // 0x44
    "use_health_station",                // 0x45
    "soft_ping",                         // 0x46
    "primary_weapon_primary_recoil",     // 0x47
    "primary_weapon_secondary_recoil",   // 0x48
    "primary_weapon_primary_chamber",    // 0x49
    "primary_weapon_secondary_chamber",  // 0x4A
    "primary_weapon_primary_charged",    // 0x4B
    "primary_weapon_secondary_charged",  // 0x4C
    "secondary_weapon_primary_recoil",   // 0x4D
    "secondary_weapon_secondary_recoil", // 0x4E
    "secondary_weapon_primary_chamber",  // 0x4F
    "secondary_weapon_secondary_chamber", // 0x50
    "secondary_weapon_primary_charged",  // 0x51
    "secondary_weapon_secondary_charged", // 0x52
    "primary_weapon_overheat_enter",     // 0x53
    "primary_weapon_overheat_loop",      // 0x54
    "primary_weapon_overheat_exit",      // 0x55
    "secondary_weapon_overheat_enter",   // 0x56
    "secondary_weapon_overheat_loop",    // 0x57
    "secondary_weapon_overheat_exit",    // 0x58
];

/// `_action_assassinate`, the value `PlayerInteraction_ExecuteAssassinationAction` stamps.
pub const ACTION_ASSASSINATE: u8 = 0x04;

pub fn action_name(v: u8) -> &'static str {
    ACTION_NAMES.get(v as usize).copied().unwrap_or("?")
}

// ---------------------------------------------------------------------------------------------
// The network tag-reference table (`sub_1803CD400` / `sub_1803CD34C`)
// ---------------------------------------------------------------------------------------------

/// The nine group codes `sub_1803CD400` builds the network tag-reference table from, **in the
/// order the engine lists them**, read off the inlined 36-byte literal at `+0x3CD42E`
/// (`"paewjorpmetidamjejbo!tpjeffednsl!dns"`, nine little-endian 4CCs).
///
/// Order matters twice over:
/// * `sub_1803CD5AC(group)` returns the ordinal of the FIRST listed group that the queried
///   group equals, or whose parent or grandparent it is — so `'weap'` resolves to 0 even
///   though a weapon is also an `'item'` and an `'obje'`.
/// * the `group == -1` ("any") form indexes the nine per-group lists CONCATENATED in this
///   order, which is why a biped's creation index lands inside the `'obje'` span.
pub const REF_GROUPS: [&str; 9] = ["weap", "proj", "item", "jmad", "obje", "jpt!", "effe", "lsnd", "snd!"];

/// Reach's object/tag group hierarchy as the engine registers it: `(group, parent)`. Only the
/// groups that can appear under one of `REF_GROUPS` need to be here; the matcher walks at most
/// two levels up, exactly as `sub_180031D64` does (`group`, `parent`, `grandparent`).
const GROUP_PARENT: &[(&str, &str)] = &[
    // obje
    ("obje", ""),
    ("devi", "obje"),
    ("item", "obje"),
    ("unit", "obje"),
    ("proj", "obje"),
    ("scen", "obje"),
    ("bloc", "obje"),
    ("crea", "obje"),
    ("efsc", "obje"),
    ("ssce", "obje"),
    ("gint", "obje"),
    ("term", "obje"),
    ("plac", "obje"),
    // devi
    ("ctrl", "devi"),
    ("mach", "devi"),
    ("lifi", "devi"),
    // item
    ("eqip", "item"),
    ("weap", "item"),
    ("garb", "item"),
    // unit
    ("bipd", "unit"),
    ("vehi", "unit"),
];

fn parent_of(g: &str) -> &'static str {
    GROUP_PARENT.iter().find(|(a, _)| *a == g).map(|(_, p)| *p).unwrap_or("")
}

/// Does `group` match the reference-table slot `slot` the way `sub_180031D64` matches —
/// `group == slot`, or `parent(group) == slot`, or `grandparent(group) == slot`?
pub fn group_matches(group: &str, slot: &str) -> bool {
    if group == slot {
        return true;
    }
    let p = parent_of(group);
    if p == slot {
        return true;
    }
    !p.is_empty() && parent_of(p) == slot
}

/// The flat network tag-reference table: for each of `REF_GROUPS` in order, every tag in the
/// cache whose group (or parent, or grandparent) is that code, **sorted ascending by the
/// engine's tag datum handle** (`sub_1801092A8` is `*a > *b` over the u32 the iterator yields,
/// `index | salt << 16`).
///
/// A wire value `v` decodes as `table[v - 1]`; `v == 0` is the null reference.
#[derive(Clone, Debug, Default)]
pub struct RefTable {
    /// Concatenated per-group spans. Each entry is a cache tag id.
    pub flat: Vec<u32>,
    /// `group_start[i]` is where `REF_GROUPS[i]`'s span begins in `flat`.
    pub group_start: [usize; 9],
}

impl RefTable {
    /// Build it for a loaded Reach cache: `map_bytes` is the map's own `.map` file (for the
    /// salts) and `tags` is `(tag id, 4-char group)` for every tag in it.
    ///
    /// Returns `None` when the map is not a Reach MCC cache or carries no tag table.
    pub fn for_map(map_bytes: &[u8], tags: &[(u32, String)]) -> Option<RefTable> {
        let salts = reach_tag_salts(map_bytes).ok()?;
        Some(RefTable::build(tags, |id| datum_handle(&salts, id)))
    }

    /// Every concrete tag group that can sit under one of [`REF_GROUPS`] — what to enumerate out
    /// of the cache to feed [`RefTable::for_map`]. The native enumerator is exact-group, so the
    /// abstract parents (`obje`, `item`, `unit`, `devi`) return nothing and the hierarchy is
    /// applied here instead.
    pub const CONCRETE_GROUPS: &'static [&'static str] = &[
        "weap", "proj", "eqip", "garb", "bipd", "vehi", "scen", "bloc", "crea", "efsc", "ssce", "gint", "term",
        "ctrl", "mach", "lifi", "plac", "jmad", "jpt!", "effe", "lsnd", "snd!",
    ];

    /// Build it from a tag enumeration: `(tag_id, 4-char group code)` for every tag in the
    /// cache. `sort_key` turns a tag id into the value the engine sorts on.
    pub fn build(tags: &[(u32, String)], sort_key: impl Fn(u32) -> u64) -> RefTable {
        let mut flat = Vec::new();
        let mut group_start = [0usize; 9];
        for (i, slot) in REF_GROUPS.iter().enumerate() {
            group_start[i] = flat.len();
            let mut span: Vec<u32> = tags.iter().filter(|(_, g)| group_matches(g, slot)).map(|(t, _)| *t).collect();
            span.sort_by_key(|&t| sort_key(t));
            flat.extend(span);
        }
        RefTable { flat, group_start }
    }

    pub fn resolve(&self, wire: u32) -> Option<u32> {
        if wire == 0 {
            return None;
        }
        self.flat.get(wire as usize - 1).copied()
    }

    /// Which `REF_GROUPS` span a wire value falls in — a sanity label, not a decode.
    pub fn span_of(&self, wire: u32) -> Option<&'static str> {
        if wire == 0 {
            return None;
        }
        let i = wire as usize - 1;
        let mut best = None;
        for (k, &s) in self.group_start.iter().enumerate() {
            if i >= s {
                best = Some(REF_GROUPS[k]);
            }
        }
        best
    }
}

/// Every tag's **datum-handle salt**, in tag-index order, read out of a Reach (MCC, cache
/// version 13) `.map`. `d` is the whole file.
///
/// This is the engine's sort key and nothing else will do. The network tag-reference table is
/// sorted ascending by the tag's datum handle, `(salt << 16) | index` (`sub_1801092A8` is
/// `*a > *b`, and `sub_1806898CC` is an ascending quicksort), and in a real Reach cache the salt
/// is an **arbitrary** authoring-time allocation counter: bijective over the live tags, always in
/// `0x8000..=0xFFFF`, and NOT monotone in the tag index — `forge_halo` has 352 descents, it wraps
/// `0xFFFF -> 0x8000`, and the resulting order has **zero** fixed points against index order.
/// Sorting by the tag index instead puts a biped's wire value on a random `bloc`.
///
/// Cache header (little-endian; the header is 0xA000 bytes, `daeh` at 0, `toof` at 0x9FFC):
/// ```text
/// 0x020  i32      tag count (cross-check)
/// 0x2E0  u64      virtual base of the tag-data partitions
/// 0x2E8  u64      virtual address of the tag-index header
/// 0x4CC  u32 x 4  section_offsets[4]
/// 0x4DC  {u32 address, u32 size} x 4   sections[4]   (section 2 is the tag data)
/// ```
/// `file_offset(section k) = sections[k].address + section_offsets[k]`, and
/// `file_offset(va) = va - (vbase - file_offset(section 2))`. These are H4's own header fields
/// shifted by +8 (`crate::h4::cache` uses 728/736/1220/1236).
///
/// Tag-index header: `+0x10` i32 tag count, `+0x18` u64 instance-array va, `+0x48` the `sgat`
/// signature. A tag instance is 8 bytes: `{u16 group ordinal, u16 SALT, u32 paged offset}`, and a
/// null tag has `0xFFFF` in both u16s — the engine's own `v7 != -1 && *v6 != -1` test.
///
/// `shared.map` and `campaign.map` are pure resource caches (`vbase == 0`), so they carry no tag
/// table and no salts; every tag comes from the map's own cache.
pub fn reach_tag_salts(d: &[u8]) -> Result<Vec<u16>, &'static str> {
    fn u16_at(d: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([d[o], d[o + 1]])
    }
    fn u32_at(d: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
    }
    fn u64_at(d: &[u8], o: usize) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&d[o..o + 8]);
        u64::from_le_bytes(b)
    }
    if d.len() < 0xA000 || &d[0..4] != b"daeh" || u32_at(d, 4) != 13 {
        return Err("not a Reach MCC (cache version 13) map");
    }
    if &d[0x9FFC..0xA000] != b"toof" {
        return Err("bad cache footer");
    }
    let sec2_file = u32_at(d, 0x4DC + 2 * 8).wrapping_add(u32_at(d, 0x4CC + 2 * 4)) as u64;
    let vbase = u64_at(d, 0x2E0);
    let tag_index_va = u64_at(d, 0x2E8);
    if vbase == 0 || tag_index_va == 0 {
        return Err("cache has no tag table (shared.map / campaign.map)");
    }
    let magic = vbase.wrapping_sub(sec2_file);
    let idx = tag_index_va.wrapping_sub(magic) as usize;
    if idx.saturating_add(0x4C) > d.len() {
        return Err("tag index header out of range");
    }
    if &d[idx + 0x48..idx + 0x4C] != b"sgat" {
        return Err("tag index signature mismatch");
    }
    let n = u32_at(d, idx + 0x10) as usize;
    if n != u32_at(d, 0x20) as usize {
        return Err("tag count disagrees with the header");
    }
    let base = u64_at(d, idx + 0x18).wrapping_sub(magic) as usize;
    if base.saturating_add(n * 8) > d.len() {
        return Err("tag instance array out of range");
    }
    Ok((0..n).map(|i| u16_at(d, base + i * 8 + 2)).collect())
}

/// The engine's sort key for the network tag-reference table: the tag's datum handle.
pub fn datum_handle(salts: &[u16], tag_index: u32) -> u64 {
    let salt = salts.get(tag_index as usize).copied().unwrap_or(0xFFFF) as u64;
    (salt << 16) | tag_index as u64
}

// ---------------------------------------------------------------------------------------------
// The one field every creation payload starts with
// ---------------------------------------------------------------------------------------------

/// The 14-bit network tag-reference at **bit offset 0** of a creation payload — the object
/// DEFINITION the entity was spawned from, as an index into `RefTable` (`+ 1`; 0 = null).
///
/// `sub_1803D5A8C`, the shared creation reader for types 11 (`biped`), 12 (`item`),
/// 13/14 (`generic`), 15 (`vehicle`), 17 (`weapon`) and 18 (`device`), opens with
/// `sub_1803CD34C(..., group = -1)`, and for `group == -1` that function uses a **hard-coded 14
/// bits** (`or ebx,-1; lea eax,[rbx+0xf]`). The reader then FAILS the whole message if the
/// resolved reference is null, so a valid creation always names a definition.
pub fn creation_def_wire(e: &EntityMessage) -> Option<u32> {
    if e.is_update {
        return None; // an update carries no creation block
    }
    let mut r = BitReader::new(&e.payload);
    r.bits(14).ok().map(|v| v as u32)
}

// ---------------------------------------------------------------------------------------------
// The EVENT channel — list B command kind 1 (`sub_18043D9A8`)
// ---------------------------------------------------------------------------------------------

/// One decoded game event: list B's command **kind 1**.
///
/// `sub_18043D9A8` reads, out of the command's own byte payload (MSB-first, like everything
/// else):
///
/// ```text
/// 7 bits   event type, REJECTED by the engine above 0x47  -> 72 types
/// 2 bits   object-reference count, REJECTED above 2
/// n x    [ 1 bit present ][ 27-bit object reference ]      (absent = the null handle)
///          per-type parameters, read by the handler's own vtbl+0x58
/// ```
///
/// The 72 handlers are registered as `a1[34 + type]` by `sub_180419618` — the same function
/// that registers the 19 entity types as `a1[1 + type]` and reports their counts (19 and 72)
/// through its two out-parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameEvent {
    /// 0..71.
    pub kind: u8,
    /// The 0, 1 or 2 object references, `None` where the engine wrote the null handle.
    pub refs: [Option<u32>; 2],
    pub ref_count: u8,
    /// Bits of per-type parameter payload left after the header (including the writer's
    /// round-up-to-a-byte padding, so it is an upper bound on the real parameter width).
    pub param_bits: usize,
    /// The parameter bits, left-aligned in the first byte read after the header.
    pub params: Vec<u8>,
}

/// Decode a list-B kind-1 payload. `None` when it fails the engine's own two range checks.
pub fn decode_event(payload: &[u8]) -> Option<GameEvent> {
    let mut r = BitReader::new(payload);
    let kind = r.bits(7).ok()? as u8;
    if kind > 0x47 {
        return None;
    }
    let n = r.bits(2).ok()? as u8;
    if n > 2 {
        return None;
    }
    let mut refs = [None, None];
    for i in 0..n as usize {
        if r.flag().ok()? {
            let hi = r.bits(16).ok()?;
            let lo = r.bits(11).ok()?;
            refs[i] = Some(((hi << 16) | lo) as u32);
        }
    }
    let param_bits = r.remaining();
    let mut params = Vec::new();
    let mut rr = r;
    while rr.remaining() >= 8 {
        params.push(rr.bits(8).ok()? as u8);
    }
    if rr.remaining() > 0 {
        let k = rr.remaining() as u32;
        params.push((rr.bits(k).ok()? << (8 - k)) as u8);
    }
    Some(GameEvent { kind, refs, ref_count: n, param_bits, params })
}

/// The 72 event types, in registration order (`a1[34 + type]` in `sub_180419618`), named from
/// each handler's own vtable slot 1 — the same method that names the 19 entity types, which is
/// what validates the method: slot 1 of `a1[1 + type]` returns `biped` for 11, `vehicle` for 15
/// and `weapon` for 17, exactly matching the entity table already in `re-notes`.
pub const EVENT_NAMES: [&str; 72] = [
    "damage_aftermath",                         // 0
    "damage_section_response",                  // 1
    "breakable_surface_break",                  // 2
    "breakable_surface_damage_aoe",             // 3
    "projectile_attached",                      // 4
    "projectile_detonate",                      // 5
    "projectile_impact_effect",                 // 6
    "projectile_object_impact_effect",          // 7
    "biped_board_vehicle",                      // 8
    "biped_pickup",                             // 9
    "weapon_effect",                            // 10
    "weapon_empty_click",                       // 11
    "biped_melee_clang",                        // 12
    "player_account_balance_notification",      // 13
    "incident",                                 // 14
    "player_money_grant",                       // 15
    "unit_zoom",                                // 16
    "megalo_chud_message",                      // 17
    "authority_ignored_predicted_position",     // 18
    "biped_exit_vehicle",                       // 19
    "device_touch",                             // 20
    "biped_debug_teleport",                     // 21
    "biped_assassinate",                        // 22
    "biped_equipment_activation",               // 23
    "vehicle_auto_turret_choose_target",        // 24
    "request_weapon_fire",                      // 25
    "action_weapon_fire",                       // 26
    "weapon_reload",                            // 27
    "biped_throw_initiate",                     // 28
    "biped_melee_initiate",                     // 29
    "vehicle_trick",                            // 30
    "biped_dodge",                              // 31
    "weapon_pickup",                            // 32
    "weapon_put_away",                          // 33
    "weapon_drop",                              // 34
    "weapon_trade",                             // 35
    "weapon_tether_request",                    // 36
    "vehicle_flip",                             // 37
    "biped_throw_release",                      // 38
    "biped_melee_damage",                       // 39
    "biped_enter_vehicle",                      // 40
    "biped_vehicle_evict",                      // 41
    "game_engine_request_boot_player",          // 42
    "request_projectile_attach",                // 43
    "biped_pickup_item_request",                // 44
    "projectile_supercombine_request",          // 45
    "object_refresh",                           // 46
    "player_editor_request",                    // 47
    "player_purchase_request",                  // 48
    "player_loadout_request",                   // 49
    "biped_laser_designation",                  // 50
    "player_set_respawn_target_transform",      // 51
    "player_set_orbiting_camera_target",        // 52
    "player_call_for_help_request",             // 53
    "player_force_base_respawn",                // 54
    "networked_ai_action",                      // 55
    "scripted_sound_trigger",                   // 56
    "scripted_sound_naked",                     // 57
    "scripted_sound_start",                     // 58
    "scripted_sound_start_unattached",          // 59
    "scripted_sound_stop",                      // 60
    "scripted_looping_sound_start",             // 61
    "scripted_looping_sound_stop",              // 62
    "scripted_looping_sound_set_scale",         // 63
    "scripted_looping_sound_set_alternate",     // 64
    "scripted_sound_impulse_unattached",        // 65
    "scripted_sound_impulse_unattached_object", // 66
    "scripted_sound_impulse_attached",          // 67
    "scripted_sound_impulse_attached_from_marker", // 68
    "AIDialog",                                 // 69
    "insert_saved_film_marker",                 // 70
    "game_campaign_metagame_points_awarded",    // 71
];

pub fn event_name(k: u8) -> &'static str {
    EVENT_NAMES.get(k as usize).copied().unwrap_or("?")
}

/// Event kinds HMS keeps. The rest (the 14 `scripted_*` sound events, the Megalo message
/// channel, the editor/purchase requests) are recorded but not worth a row.
pub const EV_DAMAGE_AFTERMATH: u8 = 0;
/// #vehicle-damage  Event 1 — the engine's own "this object's damage section fired this instant
/// response" channel, which is where a visible damage state comes from.
pub const EV_DAMAGE_SECTION_RESPONSE: u8 = 1;
pub const EV_BOARD_VEHICLE: u8 = 8;
pub const EV_PICKUP: u8 = 9;
pub const EV_MELEE_CLANG: u8 = 12;
pub const EV_INCIDENT: u8 = 14;
pub const EV_ZOOM: u8 = 16;
pub const EV_EXIT_VEHICLE: u8 = 19;
pub const EV_ASSASSINATE: u8 = 22;
pub const EV_EQUIPMENT_ACTIVATION: u8 = 23;
pub const EV_WEAPON_FIRE: u8 = 26;
pub const EV_RELOAD: u8 = 27;
pub const EV_THROW_INITIATE: u8 = 28;
pub const EV_MELEE_INITIATE: u8 = 29;
pub const EV_DODGE: u8 = 31;
pub const EV_THROW_RELEASE: u8 = 38;
pub const EV_MELEE_DAMAGE: u8 = 39;
pub const EV_ENTER_VEHICLE: u8 = 40;
pub const EV_VEHICLE_EVICT: u8 = 41;

// ---------------------------------------------------------------------------------------------
// Incident names — Reach's medal / kill-feed identifiers
// ---------------------------------------------------------------------------------------------

/// The `incident` event's leading 32-bit field is a **string id**, and these are the values the
/// corpus actually carries, resolved through `forge_halo`'s own string table
/// (`probe_incident_names`). The table is baked in because a string id is a property of the GAME
/// BUILD, not of a map — so `--film-stats` names a kill feed even with no cache to hand — and
/// `incident_name` still falls back to the cache when a value is not listed here.
///
/// `0x0078xxxx` are the engine's own incidents (`sub_18010A998` is called with these as
/// immediates from the gameplay code; `0x00780045` is the one
/// `PlayerInteraction_ExecuteAssassinationAction` fires, and it resolves to
/// `terminal_velocity_assassination`). The rest are Megalo/medal ids from the same table.
pub const INCIDENTS: &[(u32, &str)] = &[
    (0x0000066a, "multikill_x2"),
    (0x0000066b, "multikill_x3"),
    (0x0000066c, "multikill_x4"),
    (0x00000690, "sword_5x"),
    (0x00004525, "wingman_5x"),
    (0x000045b2, "yoink"),
    (0x00006961, "pain18"),
    (0x00006a43, "dirt_walk5"),
    (0x00007926, "5_in_a_row"),
    (0x00007929, "10_in_a_row"),
    (0x0000792d, "15_in_a_row"),
    (0x00007935, "25_in_a_row"),
    (0x00007939, "30_in_a_row"),
    (0x0000793d, "35_in_a_row"),
    (0x00007941, "40_in_a_row"),
    (0x00007986, "fell_to_death"),
    (0x000079d2, "flagcarrier_kill"),
    (0x000079e0, "flag_grabbed"),
    (0x000079ef, "flag_dropped"),
    (0x000079f2, "flag_scored"),
    (0x000079f4, "flag_reset"),
    (0x000079f6, "flag_recovered"),
    (0x00007a68, "custom_game_start"),
    (0x00007a6a, "fiesta_game_start"),
    (0x00007a70, "ctf_game_start"),
    (0x00007f9f, "k_ps_sheet_fade_factors0"),
    (0x000080ef, "juggernaut_new"),
    (0x000080fb, "game_start_slayer"),
    (0x0000814f, "dlc_achieve_13"),
    (0x0000815b, "dlc_achieve_19"),
    (0x00780000, "kill"),
    (0x00780001, "assist"),
    (0x00780002, "death"),
    (0x00780003, "suicide"),
    (0x00780004, "kill_betrayal"),
    (0x00780005, "guardian_kill"),
    (0x00780009, "player_joined"),
    (0x0078000a, "player_rejoined"),
    (0x0078000b, "player_switched_team"),
    (0x00780011, "highjack"),
    (0x00780013, "killjoy"),
    (0x00780015, "finishing_move_kill"),
    (0x00780016, "assassination_kill"),
    (0x00780017, "splatter_kill"),
    (0x00780018, "melee_kill"),
    (0x00780019, "headshot_kill"),
    (0x0078001a, "revenge_kill"),
    (0x0078001b, "round_over"),
    (0x00780022, "game_over"),
    (0x00780023, "player_booted_player"),
    (0x00780024, "first_blood"),
    (0x0078002d, "teleporter_used"),
    (0x0078002f, "respawn_tick"),
    (0x00780031, "gained_team_lead"),
    (0x00780036, "tied_leader"),
    (0x0078003e, "close_call"),
    (0x00780041, "avenged"),
    (0x00780042, "avenger"),
    (0x00780045, "terminal_velocity_assassination"),
    (0x00780046, "recharge_health"),
    (0x00780061, "pre_game_over"),
];

pub fn incident_name(id: u32) -> Option<&'static str> {
    INCIDENTS.binary_search_by_key(&id, |(k, _)| *k).ok().map(|i| INCIDENTS[i].1)
}

/// Incidents that mean "this player killed that player". `death` and `suicide` carry no killer.
pub const KILL_INCIDENTS: &[&str] = &[
    "kill",
    "melee_kill",
    "assassination_kill",
    "finishing_move_kill",
    "splatter_kill",
    "headshot_kill",
    "guardian_kill",
    "flagcarrier_kill",
    "revenge_kill",
    "kill_betrayal",
];

/// Incidents that mean the kill was an **assassination** — the animated, two-party finisher.
/// `assassination_kill` is the medal; `finishing_move_kill` is the same move's other id (the
/// engine awards both, and the corpus carries far more of the second); `terminal_velocity_` is
/// the ledge variant `PlayerInteraction_ExecuteAssassinationAction` fires directly.
pub const ASSASSINATION_INCIDENTS: &[&str] =
    &["assassination_kill", "finishing_move_kill", "terminal_velocity_assassination"];

// ---------------------------------------------------------------------------------------------
// What a consumer gets
// ---------------------------------------------------------------------------------------------

/// One event, folded onto the film's tick line, with the fields HMS can use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ActionEvent {
    pub tick: u32,
    pub kind: u8,
    /// First object reference: the SUBJECT (the unit acting, the object damaged, the player who
    /// earned the incident).
    pub subject: Option<u32>,
    /// Second object reference: the OTHER party (the victim, the vehicle, the responsible unit).
    pub other: Option<u32>,
    /// `incident` only: the string id. Name it with [`incident_name`].
    pub incident: Option<u32>,
    /// `biped_enter_vehicle` / `biped_board_vehicle` / `biped_exit_vehicle`: the 6-bit SEAT
    /// index the handler reads (`sub_1803B5644`).
    pub seat: Option<u8>,
    /// `biped_assassinate` only: the 19-bit value the engine passes to
    /// `PlayerInteraction_ExecuteAssassinationAction` as its third argument — WHICH synchronised
    /// action is playing. Stored raw; HMS uses it only as `value % set_len` to pick a move, which
    /// is a stable per-kill selector whatever the value turns out to mean.
    ///
    /// **It is a sync action NAME (a string id), not an index into a tag block.** An earlier note
    /// here called it an index into "a 282-byte-per-element block off both units' definitions";
    /// that was wrong and is corrected rather than deleted so the same guess is not made twice.
    /// No tag block or struct anywhere in Reach's reflection data has an element size of 282
    /// (0x11A) — all 2014 field sets plus the 50 versioned ones were enumerated; the nearest are
    /// 276, 280, 288, 296, 300 and 304, and `seats` is 300 (older versions 276 and 328, so no
    /// shipped version is 282 either). The debug build names it outright: `"Invalid
    /// sync_action_name 0x%08X '%s'"`, and the writer's own field names are `sync-action-name`,
    /// `sync-action-forward`, `sync-action-up`. Sync actions themselves are authored in `jmad`
    /// (`sync actions groups` 16 B -> `sync actions` 28 B -> participants 48 B / 20 B) and carry
    /// **no camera fields at all**; the assassination camera comes from the unit's own `sync
    /// action camera` (`bipd`/`vehi` +0x210, a full `unit_camera_struct`). Written up in
    /// `re-notes/reach_saved_film_layout.md` under `# THE ENGINE'S CAMERA MODES (#film-camera)`.
    pub action_index: Option<u32>,
    /// #vehicle-damage  `damage_section_response` only: the 4-bit DAMAGE SECTION index — a row of
    /// the object model's `damage sections` block (`hlmt` +0xCC damage info +0x14, max 16, which
    /// is why 4 bits suffice).
    pub section: Option<u8>,
    /// #vehicle-damage  `damage_section_response` only: the INSTANT RESPONSE index within that
    /// section (`instant responses`, section +0x84, max 16), or `-1` when the writer set the
    /// "no specific response" flag.
    pub response: Option<i8>,
    /// #vehicle-damage  `damage_section_response` only: the 2-bit response MODE the engine
    /// rejects above 3 (`PlayerInteraction_DamageSectionResponse` branches on it: 2 and 3 are the
    /// whole-section forms that carry no response index, 0 and 1 dispatch the named response).
    pub response_mode: Option<u8>,
}

impl ActionEvent {
    pub fn name(&self) -> &'static str {
        event_name(self.kind)
    }
    /// The incident's name, from the baked table.
    pub fn incident_label(&self) -> Option<&'static str> {
        self.incident.and_then(incident_name)
    }
    pub fn is_kill(&self) -> bool {
        self.incident_label().map(|n| KILL_INCIDENTS.contains(&n)).unwrap_or(false)
    }
    pub fn is_assassination(&self) -> bool {
        self.kind == EV_ASSASSINATE
            || self.incident_label().map(|n| ASSASSINATION_INCIDENTS.contains(&n)).unwrap_or(false)
    }
}

/// Fold one list-B kind-1 command into an [`ActionEvent`]. `None` when the payload fails the
/// engine's own two range checks (event type <= 0x47, reference count <= 2).
pub fn fold(tick: u32, payload: &[u8]) -> Option<ActionEvent> {
    let ev = decode_event(payload)?;
    let mut out = ActionEvent {
        tick,
        kind: ev.kind,
        subject: ev.refs[0],
        other: ev.refs[1],
        ..Default::default()
    };
    let mut r = BitReader::new(payload);
    // re-walk the header so the parameter reader starts bit-exact
    r.bits(7).ok()?;
    let n = r.bits(2).ok()?;
    for _ in 0..n {
        if r.flag().ok()? {
            r.bits(27).ok()?;
        }
    }
    match ev.kind {
        // `sub_1803B5644`: a bare 6-bit seat index.
        EV_ENTER_VEHICLE | EV_BOARD_VEHICLE | EV_EXIT_VEHICLE => {
            out.seat = r.bits(6).ok().map(|v| v as u8);
        }
        // `sub_1803B5BD4`: 19 bits - 1, then a position, then two 19-bit unit vectors, then a
        // full damage record (`sub_1803B6FBC`). Only the leading index is taken: everything
        // after it needs this film's `compressed_vector3` widths, and nothing consumes them yet.
        EV_ASSASSINATE => {
            out.action_index = r.bits(19).ok().map(|v| (v as u32).wrapping_sub(1));
        }
        EV_INCIDENT => {
            out.incident = r.bits(32).ok().map(|v| v as u32);
        }
        // #vehicle-damage  `sub_180464E94` (the handler's own vtbl+0x58 parameter reader, vtable
        // `0x180A50070`, whose vtbl+0x68 is `PlayerInteraction_DamageSectionResponse`):
        //
        //   4 bits   damage SECTION index              -> params +0x00
        //   1 flag   CLEAR = read 4 more bits (the instant RESPONSE index), SET = -1  -> +0x04
        //   2 bits   response MODE (the apply returns early above 3)                  -> +0x08
        //   1 flag   one bit folded into +0x0C bit 23
        //
        // 12 bits in the dense case, which closes the measured budget exactly: one object
        // reference makes the header 7 + 2 + 1 + 27 = 37 bits, 37 + 12 = 49 -> 7 bytes, and the
        // corpus measures 56 - 37 = 19 "parameter bits" (12 real + 7 of the writer's padding).
        EV_DAMAGE_SECTION_RESPONSE => {
            out.section = r.bits(4).ok().map(|v| v as u8);
            out.response = Some(if r.flag().ok()? { -1 } else { r.bits(4).ok()? as i8 });
            out.response_mode = r.bits(2).ok().map(|v| v as u8);
        }
        _ => {}
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// probes
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod probes {
    use super::*;
    use std::collections::BTreeMap;

    use crate::film;
    use crate::film_decode::{self, DecodeParams};

    #[allow(unused_imports)]
    use film_decode::BitReader as _Br;

    fn films() -> Vec<std::path::PathBuf> {
        film::find_finished_films()
    }

    /// Our own copy of the cache opener. `film_anim::probes` has one but its module is private,
    /// and that module belongs to another owner; a second `OnceLock` here keeps the two apart.
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

    fn label(p: &std::path::Path, f: &film::Film) -> String {
        let t = f.header.game_variant.title.trim();
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if t.is_empty() { stem } else { format!("{stem}  [{t}]") }
    }

    /// THE decisive census for "is the action on the wire?". The 64-bit header mask of every
    /// entity message, per entity type, over EVERY film on the machine — no field walk, so no
    /// field width can be blamed for a zero.
    #[test]
    fn probe_header_mask_census_every_film() {
        let films = films();
        if films.is_empty() {
            eprintln!("film_action: no films on this machine");
            return;
        }
        // [entity_type][bit] -> count, plus a per-type message count.
        let mut tot = vec![[0u64; 64]; 19];
        let mut msgs = [0u64; 19];
        let mut films_used = 0usize;
        let mut hits_0x36: Vec<String> = Vec::new();
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut local = vec![[0u64; 64]; 19];
            let mut lmsg = [0u64; 19];
            let mut ok = 0usize;
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                ok += 1;
                for e in &t.entities {
                    let ty = (e.entity_type as usize).min(18);
                    lmsg[ty] += 1;
                    let mut m = e.dirty_mask;
                    while m != 0 {
                        let b = m.trailing_zeros() as usize;
                        local[ty][b] += 1;
                        m &= m - 1;
                    }
                }
            }
            if ok == 0 {
                continue;
            }
            films_used += 1;
            let n11: u64 = lmsg[11];
            let a = local[11][0x36];
            let s = local[11][0x35];
            if a > 0 {
                hits_0x36.push(format!("{}: 0x36 x{a} of {n11}", label(p, &f)));
            }
            println!(
                "{:<58} recs {:>7}  t11 {:>8}  0x35 {:>6}  0x36 {:>6}  0x37 {:>6}",
                label(p, &f),
                ok,
                n11,
                s,
                a,
                local[11][0x37]
            );
            for ty in 0..19 {
                msgs[ty] += lmsg[ty];
                for b in 0..64 {
                    tot[ty][b] += local[ty][b];
                }
            }
        }
        println!("\n== header-mask bit frequency, {films_used} films ==");
        for ty in 0..19 {
            if msgs[ty] == 0 {
                continue;
            }
            let bits: Vec<String> = (0..64)
                .filter(|&b| tot[ty][b] > 0)
                .map(|b| format!("{b:#02x}:{}", tot[ty][b]))
                .collect();
            println!("type {ty:>2} ({:>9} msgs)  {}", msgs[ty], bits.join(" "));
        }
        println!("\nfilms where biped bit 0x36 (the object-action enum) ever appears: {}", hits_0x36.len());
        for h in &hits_0x36 {
            println!("  {h}");
        }
    }

    /// What list A carries. It is a different command namespace from list B (list B's kind 2/3/4
    /// are entity create/update/destroy), and nothing has ever been decoded out of it.
    #[test]
    fn probe_list_a_census() {
        let films = films();
        if films.is_empty() {
            return;
        }
        // kind -> (count, length histogram)
        let mut a: BTreeMap<u8, (u64, BTreeMap<usize, u64>)> = BTreeMap::new();
        let mut b: BTreeMap<u8, (u64, BTreeMap<usize, u64>)> = BTreeMap::new();
        let mut sample: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
        let mut recs = 0u64;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                recs += 1;
                for (i, c) in t.commands.iter().enumerate() {
                    let tgt = if i < t.commands_a { &mut a } else { &mut b };
                    let e = tgt.entry(c.kind).or_default();
                    e.0 += 1;
                    *e.1.entry(c.payload.len()).or_default() += 1;
                    if i < t.commands_a {
                        sample.entry(c.kind).or_insert_with(|| c.payload.clone());
                    }
                }
            }
        }
        println!("\n== command census over {recs} records ==");
        for (name, m) in [("LIST A", &a), ("LIST B", &b)] {
            println!("{name}:");
            for (k, (n, lens)) in m.iter() {
                let mut v: Vec<_> = lens.iter().collect();
                v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let top: Vec<String> = v.iter().take(6).map(|(l, c)| format!("{l}B x{c}")).collect();
                println!("  kind {k:>2}  {n:>9}  lens: {}", top.join(", "));
            }
        }
        println!("\nlist A first payloads:");
        for (k, s) in sample.iter() {
            println!("  kind {k:>2}: {}", s.iter().take(24).map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
        }
    }

    /// List B's kinds 1, 5, 6 and 11 have never been decoded. Kind 6 is 62,388 entries of
    /// EXACTLY 5 bytes across the corpus -- the same width as kind 4 (the destroy: a 5-bit
    /// entity type plus a 27-bit object reference) with ~8 bits to spare, which is the shape a
    /// per-object EVENT would have. Test it the way kind 4 was tested: does the reference name
    /// an entity that exists, and does the tail look like the 7-bit object-action enum?
    #[test]
    fn probe_list_b_unknown_kinds() {
        let films = films();
        if films.is_empty() {
            return;
        }
        for p in films.iter().take(24) {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            // net_id -> entity type, for liveness
            let mut alive: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            let mut k6_tail: BTreeMap<u32, u64> = BTreeMap::new();
            let mut k6_type: BTreeMap<u8, u64> = BTreeMap::new();
            let mut k6_known = 0u64;
            let mut k6_tot = 0u64;
            let mut k5: BTreeMap<u32, u64> = BTreeMap::new();
            let mut k11: Vec<Vec<u8>> = Vec::new();
            let mut k1_lead: BTreeMap<(usize, u8), u64> = BTreeMap::new();
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    alive.insert(e.net_id, e.entity_type);
                }
                for (i, c) in t.commands.iter().enumerate() {
                    if i < t.commands_a {
                        continue;
                    }
                    match c.kind {
                        6 if c.payload.len() == 5 => {
                            let mut r = BitReader::new(&c.payload);
                            let ty = r.bits(5).unwrap_or(0) as u8;
                            let hi = r.bits(16).unwrap_or(0);
                            let lo = r.bits(11).unwrap_or(0);
                            let id = ((hi << 16) | lo) as u32;
                            let tail = r.bits(8).unwrap_or(0) as u32;
                            k6_tot += 1;
                            if alive.get(&id) == Some(&ty) {
                                k6_known += 1;
                            }
                            *k6_tail.entry(tail).or_default() += 1;
                            *k6_type.entry(ty).or_default() += 1;
                        }
                        5 if c.payload.len() == 4 => {
                            let mut r = BitReader::new(&c.payload);
                            *k5.entry(r.bits(32).unwrap_or(0) as u32).or_default() += 1;
                        }
                        11 => {
                            if k11.len() < 6 {
                                k11.push(c.payload.clone());
                            }
                        }
                        1 => {
                            let lead = c.payload.first().copied().unwrap_or(0);
                            *k1_lead.entry((c.payload.len(), lead)).or_default() += 1;
                        }
                        _ => {}
                    }
                }
            }
            if k6_tot == 0 && k5.is_empty() && k11.is_empty() {
                continue;
            }
            println!("\n{}", label(p, &f));
            if k6_tot > 0 {
                println!(
                    "  kind 6: {k6_tot} msgs, {k6_known} ({:.1}%) name a LIVE entity of the same type",
                    k6_known as f64 / k6_tot as f64 * 100.0
                );
                println!("    types: {:?}", k6_type);
                let mut v: Vec<_> = k6_tail.iter().collect();
                v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let top: Vec<String> = v
                    .iter()
                    .take(14)
                    .map(|(t, c)| format!("{t:#04x}({}/{})x{c}", action_name((**t >> 1) as u8), action_name(**t as u8)))
                    .collect();
                println!("    tail8: {}", top.join(" "));
            }
            if !k5.is_empty() {
                let mut v: Vec<_> = k5.iter().collect();
                v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                println!("  kind 5 top: {:?}", v.iter().take(6).map(|(k, c)| (format!("{k:#x}"), **c)).collect::<Vec<_>>());
            }
            for s in &k11 {
                println!("  kind 11: {}", s.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
            }
            let mut v: Vec<_> = k1_lead.iter().collect();
            v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            println!("  kind 1 (len,lead) top: {:?}", v.iter().take(8).map(|((l, b), c)| (*l, format!("{b:02x}"), **c)).collect::<Vec<_>>());
        }
    }

    /// Print-only: where and when each decoded assassination happens on one clip, so a capture
    /// can be aimed at one without a human at the keyboard.
    #[test]
    fn probe_assassination_camera_targets() {
        for p in films() {
            let n = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            if !n.contains("706C7553_6ACC3BF7") {
                continue;
            }
            let Ok(f) = film::open(&p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for a in crate::film_anim::assassinations(&idx) {
                let tr = a.attacker.and_then(|i| idx.tracks.iter().find(|t| t.net_id == i));
                let pos = tr.and_then(|t| t.sample(a.tick).or_else(|| t.positions.last().map(|&(_, q)| q)));
                let yaw = tr.and_then(|t| t.facing_at(a.tick).or_else(|| t.facings.last().map(|&(_, y)| y)));
                println!(
                    "tick {:>7} ({:>6.2}s) {:<22} attacker {:?} victim {:?} pos {:?} yaw {:?}",
                    a.tick,
                    a.tick as f32 / 60.0,
                    a.incident,
                    a.attacker.map(|v| format!("{v:#x}")),
                    a.victim.map(|v| format!("{v:#x}")),
                    pos.map(|q| [(q[0] * 100.0).round() / 100.0, (q[1] * 100.0).round() / 100.0, (q[2] * 100.0).round() / 100.0]),
                    yaw.map(|y| (y.to_degrees() * 10.0).round() / 10.0)
                );
            }
        }
    }

    /// WHICH reference is the killer. The kill-feed rows on one clip all name the same entity
    /// in slot 0, which one biped cannot be, so the slot's entity TYPE has to be checked rather
    /// than assumed.
    #[test]
    fn probe_incident_reference_types() {
        let films = films();
        if films.is_empty() {
            return;
        }
        for p in films.iter().take(14) {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut etype: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            // (ref slot, entity type) -> count, over KILL incidents only
            let mut t: BTreeMap<(usize, String), u64> = BTreeMap::new();
            let mut distinct: [std::collections::BTreeSet<u32>; 2] = Default::default();
            let mut rows = 0u64;
            for rec in f.data.records.iter() {
                let Ok(tr) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &tr.entities {
                    etype.insert(e.net_id, e.entity_type);
                }
                for c in tr.commands.iter().skip(tr.commands_a).filter(|c| c.kind == 1) {
                    let Some(ev) = fold(0, &c.payload) else { continue };
                    if !ev.is_kill() {
                        continue;
                    }
                    rows += 1;
                    for (k, r) in [ev.subject, ev.other].into_iter().enumerate() {
                        match r {
                            None => *t.entry((k, "null".into())).or_default() += 1,
                            Some(id) => {
                                distinct[k].insert(id);
                                let lab = match etype.get(&id) {
                                    Some(&ty) => format!("type {ty}"),
                                    None => "unknown".into(),
                                };
                                *t.entry((k, lab)).or_default() += 1;
                            }
                        }
                    }
                }
            }
            if rows == 0 {
                continue;
            }
            let lab = label(p, &f);
            println!(
                "{:<46} {rows:>5} kill rows  r0 {} distinct, r1 {} distinct   {:?}",
                &lab[..lab.len().min(45)],
                distinct[0].len(),
                distinct[1].len(),
                t
            );
        }
    }

    /// Which incident marks an ASSASSINATION. `assassination_kill` and `finishing_move_kill`
    /// are both candidates; the clips whose gametype is literally about assassination decide it.
    #[test]
    fn probe_assassination_incident_contrast() {
        let films = films();
        if films.is_empty() {
            return;
        }
        const INT: [&str; 8] = [
            "kill",
            "melee_kill",
            "assassination_kill",
            "finishing_move_kill",
            "splatter_kill",
            "headshot_kill",
            "death",
            "suicide",
        ];
        println!("{:<46} {:>8} {}", "film [gametype]", "events", INT.iter().map(|n| format!("{:>12}", &n[..n.len().min(12)])).collect::<Vec<_>>().join(" "));
        let mut agg: BTreeMap<bool, [u64; 8]> = BTreeMap::new();
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let title = f.header.game_variant.title.to_lowercase();
            let assn = title.contains("assassin");
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut n = [0u64; 8];
            let mut tot = 0u64;
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for c in t.commands.iter().skip(t.commands_a).filter(|c| c.kind == 1) {
                    let Some(ev) = fold(0, &c.payload) else { continue };
                    tot += 1;
                    if let Some(lab) = ev.incident_label() {
                        if let Some(i) = INT.iter().position(|x| *x == lab) {
                            n[i] += 1;
                        }
                    }
                }
            }
            if n.iter().all(|&x| x == 0) {
                continue;
            }
            let e = agg.entry(assn).or_insert([0u64; 8]);
            for i in 0..8 {
                e[i] += n[i];
            }
            let lab = label(p, &f);
            println!("{:<46} {tot:>8} {}", &lab[..lab.len().min(45)], n.iter().map(|v| format!("{v:>12}")).collect::<Vec<_>>().join(" "));
        }
        println!("\n{:<46} {:>8} {}", "AGGREGATE", "", INT.iter().map(|n| format!("{:>12}", &n[..n.len().min(12)])).collect::<Vec<_>>().join(" "));
        for (assn, n) in agg.iter() {
            let tot: u64 = n[0] + n[1] + n[2] + n[3] + n[4] + n[5];
            println!(
                "{:<46} {:>8} {}",
                if *assn { "assassination gametypes" } else { "every other clip" },
                tot,
                n.iter().map(|v| format!("{v:>12}")).collect::<Vec<_>>().join(" ")
            );
        }
    }

    #[test]
    fn the_incident_table_is_sorted_and_the_gate_bites() {
        assert!(INCIDENTS.windows(2).all(|w| w[0].0 < w[1].0), "INCIDENTS must be sorted for binary_search");
        assert_eq!(incident_name(0x0078_0016), Some("assassination_kill"));
        // shift the key by one and the gate must fail
        assert_ne!(incident_name(0x0078_0017), Some("assassination_kill"));
        assert_eq!(incident_name(0x0078_0017), Some("splatter_kill"));
        assert_eq!(event_name(22), "biped_assassinate");
        assert_ne!(event_name(23), "biped_assassinate");
        assert_eq!(ACTION_NAMES[ACTION_ASSASSINATE as usize], "assassinate");
    }

    /// The EVENT channel, censused. For each of the 72 event types: how often it fires, how
    /// many object references it carries, how wide its parameter payload is, and what KIND of
    /// entity its references name. A two-reference event between two bipeds is what a
    /// synchronised action looks like on the wire.
    #[test]
    fn probe_event_channel_census() {
        let films = films();
        if films.is_empty() {
            return;
        }
        #[derive(Default, Clone)]
        struct Agg {
            n: u64,
            refs: [u64; 3],
            bits: BTreeMap<usize, u64>,
            reftype: BTreeMap<(usize, u8), u64>,
            unresolved: u64,
            same: u64,
        }
        let mut glob: BTreeMap<u8, Agg> = BTreeMap::new();
        let mut bad = 0u64;
        let mut tot = 0u64;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut etype: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            let mut local: BTreeMap<u8, Agg> = BTreeMap::new();
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    etype.insert(e.net_id, e.entity_type);
                }
                for (i, c) in t.commands.iter().enumerate() {
                    if i < t.commands_a || c.kind != 1 {
                        continue;
                    }
                    tot += 1;
                    let Some(ev) = decode_event(&c.payload) else {
                        bad += 1;
                        continue;
                    };
                    let a = local.entry(ev.kind).or_default();
                    a.n += 1;
                    a.refs[ev.ref_count.min(2) as usize] += 1;
                    *a.bits.entry(ev.param_bits).or_default() += 1;
                    for (k, r) in ev.refs.iter().enumerate() {
                        if let Some(id) = r {
                            match etype.get(id) {
                                Some(&ty) => *a.reftype.entry((k, ty)).or_default() += 1,
                                None => a.unresolved += 1,
                            }
                        }
                    }
                    if ev.refs[0].is_some() && ev.refs[0] == ev.refs[1] {
                        a.same += 1;
                    }
                }
            }
            for (k, v) in local {
                let g = glob.entry(k).or_default();
                g.n += v.n;
                g.unresolved += v.unresolved;
                g.same += v.same;
                for i in 0..3 {
                    g.refs[i] += v.refs[i];
                }
                for (b, c) in v.bits {
                    *g.bits.entry(b).or_default() += c;
                }
                for (b, c) in v.reftype {
                    *g.reftype.entry(b).or_default() += c;
                }
            }
        }
        println!("\n== event channel (list B kind 1): {tot} commands, {bad} rejected by the engine's own range checks ==");
        println!("{:>4} {:>9}  {:<18} {:<28} {}", "type", "count", "refs(0/1/2)", "param bits (top)", "ref entity types");
        for (k, a) in glob.iter() {
            let mut bb: Vec<_> = a.bits.iter().collect();
            bb.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            let bits: Vec<String> = bb.iter().take(4).map(|(b, c)| format!("{b}x{c}")).collect();
            let rt: Vec<String> = a.reftype.iter().map(|((k, ty), c)| format!("r{k}:t{ty}x{c}")).collect();
            println!(
                "{:>4} {:>9}  {:<18} {:<28} {}{}",
                k,
                a.n,
                format!("{}/{}/{}", a.refs[0], a.refs[1], a.refs[2]),
                bits.join(" "),
                rt.join(" "),
                if a.unresolved > 0 { format!("  [unresolved {}]", a.unresolved) } else { String::new() }
            );
        }
    }

    /// Per film: how often the events that matter for a two-party action fire, and the
    /// `incident` payload's leading 32-bit identifier.
    #[test]
    fn probe_action_events_per_film() {
        let films = films();
        if films.is_empty() {
            return;
        }
        const WATCH: [(u8, &str); 10] = [
            (22, "biped_assassinate"),
            (14, "incident"),
            (29, "biped_melee_initiate"),
            (39, "biped_melee_damage"),
            (12, "biped_melee_clang"),
            (0, "damage_aftermath"),
            (40, "biped_enter_vehicle"),
            (8, "biped_board_vehicle"),
            (19, "biped_exit_vehicle"),
            (41, "biped_vehicle_evict"),
        ];
        let mut inc: BTreeMap<u32, u64> = BTreeMap::new();
        let mut assn: Vec<String> = Vec::new();
        println!("{:<44} {}", "film", WATCH.iter().map(|(_, n)| format!("{:>10}", &n[..n.len().min(10)])).collect::<Vec<_>>().join(" "));
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut n = [0u64; 10];
            let mut m36 = 0u64;
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.entity_type == 11 && e.dirty_mask & (1u64 << 0x36) != 0 {
                        m36 += 1;
                    }
                }
                for (i, c) in t.commands.iter().enumerate() {
                    if i < t.commands_a || c.kind != 1 {
                        continue;
                    }
                    let Some(ev) = decode_event(&c.payload) else { continue };
                    if let Some(k) = WATCH.iter().position(|(x, _)| *x == ev.kind) {
                        n[k] += 1;
                    }
                    if ev.kind == 14 && ev.params.len() >= 4 {
                        *inc.entry(u32::from_be_bytes([ev.params[0], ev.params[1], ev.params[2], ev.params[3]])).or_default() += 1;
                    }
                    if ev.kind == 22 {
                        assn.push(format!(
                            "    {} t{} refs {:?} {:?} bits {} params {}",
                            label(p, &f),
                            ev.ref_count,
                            ev.refs[0].map(|v| format!("{v:#x}")),
                            ev.refs[1].map(|v| format!("{v:#x}")),
                            ev.param_bits,
                            ev.params.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("")
                        ));
                    }
                }
            }
            if n.iter().all(|&x| x == 0) && m36 == 0 {
                continue;
            }
            let lab = label(p, &f);
            println!(
                "{:<44} {}  [mask0x36 {m36}]",
                lab.chars().take(43).collect::<String>(), // char-safe: a gamertag can hold a multi-byte emoji
                n.iter().map(|v| format!("{v:>10}")).collect::<Vec<_>>().join(" ")
            );
        }
        println!("\nincident leading u32 (top 30 of {} distinct):", inc.len());
        let mut v: Vec<_> = inc.iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (k, c) in v.iter().take(30) {
            println!("  {k:#010x}  x{c}");
        }
        println!("\nbiped_assassinate events ({}):", assn.len());
        for a in &assn {
            println!("{a}");
        }
    }

    /// The `incident` event's leading 32-bit field, resolved through the CACHE's string-id
    /// table. If it is a string id, every value names itself and the assassination incident
    /// stops being a guess.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_incident_names() {
        let films = films();
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("film_action: forge_halo cache not available");
            return;
        };
        let mut inc: BTreeMap<u32, u64> = BTreeMap::new();
        let mut pairs: BTreeMap<u32, (u64, u64)> = BTreeMap::new(); // id -> (2-ref, biped-pair)
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            let mut etype: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    etype.insert(e.net_id, e.entity_type);
                }
                for (i, c) in t.commands.iter().enumerate() {
                    if i < t.commands_a || c.kind != 1 {
                        continue;
                    }
                    let Some(ev) = decode_event(&c.payload) else { continue };
                    if ev.kind != 14 || ev.params.len() < 4 {
                        continue;
                    }
                    let id = u32::from_be_bytes([ev.params[0], ev.params[1], ev.params[2], ev.params[3]]);
                    *inc.entry(id).or_default() += 1;
                    let e = pairs.entry(id).or_default();
                    if ev.ref_count == 2 {
                        e.0 += 1;
                        let a = ev.refs[0].and_then(|r| etype.get(&r).copied());
                        let b = ev.refs[1].and_then(|r| etype.get(&r).copied());
                        if a == Some(11) && b == Some(11) {
                            e.1 += 1;
                        }
                    }
                }
            }
        }
        println!("\n== incident ids, resolved through the cache string table ({} distinct) ==", inc.len());
        let mut v: Vec<_> = inc.iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (id, c) in v {
            let nm = native.string_id(cache, *id).unwrap_or_else(|| "<unresolved>".into());
            let (two, bp) = pairs.get(id).copied().unwrap_or_default();
            println!("  {id:#010x}  x{c:<6}  2refs {two:<6} biped-pair {bp:<6}  {nm}");
        }
        println!("\nengine-named: 0x00780045 is the id PlayerInteraction_ExecuteAssassinationAction fires");
        println!("  resolves to: {:?}", native.string_id(cache, 0x0078_0045));
    }

    /// The network tag-reference table, built from the cache and scored. A wire value decodes as
    /// `table[v - 1]`; if the table is right, type 11 (`biped`) resolves to `bipd` tags, 15 to
    /// `vehi`, 17 to `weap`, 12 to `eqip`, 13 to scenery/blocs — and the whole thing falls apart
    /// if the group ORDER or the sort key is wrong, which is the gate.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_object_definition_identity() {
        let films = films();
        let Some((native, cache)) = open_reach("forge_halo") else {
            eprintln!("film_action: forge_halo cache not available");
            return;
        };
        // Enumerate every concrete group that can sit under one of REF_GROUPS.
        const CONCRETE: &[&str] = &[
            "weap", "proj", "eqip", "garb", "bipd", "vehi", "scen", "bloc", "crea", "efsc", "ssce", "gint", "term",
            "ctrl", "mach", "lifi", "plac", "jmad", "jpt!", "effe", "lsnd", "snd!", "item", "unit", "devi", "obje",
            "crat",
        ];
        let mut tags: Vec<(u32, String)> = Vec::new();
        print!("group counts:");
        for g in CONCRETE {
            let v = native.tags_by_group(cache, g);
            if !v.is_empty() {
                print!(" {g}={}", v.len());
            }
            for t in v {
                tags.push((t, (*g).to_string()));
            }
        }
        println!();
        println!("sample ids: {:?}", tags.iter().take(8).map(|(t, g)| format!("{g}:{t:#x}")).collect::<Vec<_>>());
        // The engine's key: the tag's datum handle, from the cache's own tag-instance array.
        let map_path = crate::mapcat::enumerate()
            .into_iter()
            .find(|c| {
                c.game == crate::mapcat::Game::Reach
                    && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
            })
            .map(|c| c.path);
        let salts: Vec<u16> = match map_path.as_ref().and_then(|p| std::fs::read(p).ok()) {
            Some(bytes) => match reach_tag_salts(&bytes) {
                Ok(v) => {
                    println!("salts: {} tags, first {:?}", v.len(), &v[..8.min(v.len())]);
                    v
                }
                Err(e) => {
                    println!("salt read failed: {e}");
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        for (label, key) in [("datum handle (salt<<16|index)", 2u8), ("tag index", 0u8)] {
            let t = RefTable::build(&tags, |id| if key == 2 { datum_handle(&salts, id) } else { id as u64 });
            println!("\n--- sort key: {label} --- flat {} entries, group starts {:?}", t.flat.len(), t.group_start);
            let mut per: BTreeMap<u8, BTreeMap<String, u64>> = BTreeMap::new();
            let mut used = 0usize;
            for p in films.iter() {
                if !name_contains(p, "forge_hal") {
                    continue;
                }
                let Ok(f) = film::open(p) else { continue };
                if f.data.records.is_empty() {
                    continue;
                }
                used += 1;
                if used > 6 {
                    break;
                }
                let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
                for rec in f.data.records.iter() {
                    let Ok(tr) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                    for e in &tr.entities {
                        let Some(w) = creation_def_wire(e) else { continue };
                        let lab = match t.resolve(w) {
                            None => format!("<null {w}>"),
                            Some(tag) => match (native.tag_class(cache, tag), native.tag_name(cache, tag)) {
                                (Some(g), Some(n)) => format!("{g} {}", n.rsplit('\\').next().unwrap_or(&n)),
                                _ => format!("<unresolved tag {tag:#x} span {:?}>", t.span_of(w)),
                            },
                        };
                        *per.entry(e.entity_type).or_default().entry(lab).or_default() += 1;
                    }
                }
            }
            for (ty, m) in &per {
                let tot: u64 = m.values().sum();
                let mut v: Vec<_> = m.iter().collect();
                v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let top: Vec<String> = v.iter().take(6).map(|(k, c)| format!("{k} x{c}")).collect();
                println!("  type {ty:>2} ({tot:>6}, {:>3} distinct): {}", m.len(), top.join(" | "));
            }
        }
    }

    fn name_contains(p: &std::path::Path, s: &str) -> bool {
        p.file_stem().map(|x| x.to_string_lossy().contains(s)).unwrap_or(false)
    }

    /// The creation payload's leading 14-bit network tag reference. If it really is the object
    /// definition, the number of DISTINCT values per entity type per film must look like the
    /// number of distinct object definitions of that kind in play: one biped on an all-Spartan
    /// Slayer clip, several on Firefight (Elite, Grunt, Jackal, Brute ...).
    #[test]
    fn probe_creation_definition_index() {
        let films = films();
        if films.is_empty() {
            return;
        }
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            // entity type -> value -> count
            let mut per: BTreeMap<u8, BTreeMap<u32, u64>> = BTreeMap::new();
            let mut creations = 0u64;
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for e in &t.entities {
                    if e.is_update {
                        continue;
                    }
                    creations += 1;
                    if let Some(v) = creation_def_wire(e) {
                        *per.entry(e.entity_type).or_default().entry(v).or_default() += 1;
                    }
                }
            }
            if creations == 0 {
                continue;
            }
            println!("\n{}  ({creations} creations)", label(p, &f));
            for (ty, vals) in per.iter() {
                let mut v: Vec<_> = vals.iter().collect();
                v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let top: Vec<String> = v.iter().take(10).map(|(k, c)| format!("{k}x{c}")).collect();
                println!("  type {ty:>2}: {:>4} distinct, top: {}", vals.len(), top.join(" "));
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod objective_inventory {
    //! #film-objects  What NON-BIPED objects a recording actually contains, by resolved tag.
    //!
    //! Written to answer one question with evidence rather than assumption: a Castle Wars clip
    //! shows no flag, and the playback draw filter keeps only `BIPED_TYPE`, so it was not known
    //! whether the flag is MISSING FROM THE RECORDING or merely never drawn. This resolves every
    //! track's 14-bit definition reference, not just the bipeds', and prints the inventory.
    //!
    //! `#[ignore]`d: it opens a map and a film, far too heavy for the normal suite. Run it as
    //! `HMS_FILM=<film> cargo test -p hms-app print_non_biped_inventory -- --ignored --nocapture`.

    pub(crate) fn open_reach(stem: &str) -> Option<(hms_native::NativeDll, hms_native::CacheHandle)> {
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            let mut d = exe.parent().map(|p| p.to_path_buf());
            for _ in 0..6 {
                if let Some(p) = &d {
                    roots.push(p.clone());
                    d = p.parent().map(|q| q.to_path_buf());
                }
            }
        }
        let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
        let native = hms_native::NativeDll::load(&dll).ok()?;
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
    }

    #[test]
    #[ignore]
    fn print_non_biped_inventory() {
        let Ok(film_path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM to the film to inventory");
            return;
        };
        let stem = std::env::var("HMS_MAP_STEM").unwrap_or_else(|_| "forge_halo".to_string());
        let film = crate::film::open(std::path::Path::new(&film_path)).expect("film opens");
        let idx = crate::film_decode::build_index(&film.bytes, &film.data, &film.header.map_variant);
        let Some((native, cache)) = open_reach(&stem) else {
            eprintln!("no Reach cache named {stem} on this machine");
            return;
        };
        let map = crate::mapcat::enumerate()
            .into_iter()
            .find(|c| {
                c.game == crate::mapcat::Game::Reach
                    && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(&stem))
            })
            .expect("map found twice");
        let bytes = std::fs::read(&map.path).expect("map reads");
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in super::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) {
                tags.push((t, (*g).to_string()));
            }
        }
        let table = super::RefTable::for_map(&bytes, &tags).expect("ref table builds");

        let mut inv: std::collections::BTreeMap<(u8, String, String), (usize, usize)> = Default::default();
        let mut unresolved: std::collections::BTreeMap<u8, usize> = Default::default();
        for tr in &idx.tracks {
            if tr.entity_type == crate::film_play::BIPED_TYPE {
                continue;
            }
            let named = tr.def_wire.and_then(|w| table.resolve(w)).and_then(|tag| {
                Some((native.tag_class(cache, tag)?, native.tag_name(cache, tag)?))
            });
            match named {
                Some((g, n)) => {
                    let e = inv.entry((tr.entity_type, g, n)).or_default();
                    e.0 += 1;
                    if tr.moves() {
                        e.1 += 1;
                    }
                }
                None => *unresolved.entry(tr.entity_type).or_default() += 1,
            }
        }

        println!("\n=== non-biped tracks by resolved definition (tracks / of which ever moved) ===");
        let mut last_ty = u8::MAX;
        for ((ty, g, n), (count, moved)) in &inv {
            if *ty != last_ty {
                println!("  -- entity type {ty} --");
                last_ty = *ty;
            }
            let leaf = n.rsplit('\\').next().unwrap_or(n);
            println!("     {g}  {count:>4} / {moved:>4}   {leaf}");
        }
        for (ty, n) in &unresolved {
            println!("  entity type {ty}: {n} track(s) whose definition did not resolve");
        }
        println!(
            "  {} distinct definition(s) over {} non-biped track(s)",
            inv.len(),
            inv.values().map(|v| v.0).sum::<usize>()
        );
    }
}

#[cfg(test)]
mod weapon_chud_probe {
    //! #crosshair  Why 34 of 64 weapons report "no chud interface".
    //!
    //! Every hand-held sandbox weapon (magnum, DMR, assault rifle, shotgun, rockets, sword...)
    //! comes back with no reticule, while all 29 that DO decode are vehicle/turret weapons and
    //! carried objectives -- yet `ui\chud\magnum` plainly exists in the same cache and authors a
    //! six-part reticule. This prints the raw `weapon_interface_struct` bytes so the cause is
    //! read rather than guessed.

    #[test]
    #[ignore]
    fn print_weapon_chud_refs() {
        let Some((native, cache)) = super::objective_inventory::open_reach("forge_halo") else {
            eprintln!("no forge_halo cache");
            return;
        };
        let want = [
            "magnum", "dmr", "assault_rifle", "shotgun", "energy_sword", // report none
            "ghost_dual_cannon", "warthog_chaingun", "flag",             // report a reticule
        ];
        println!("\n=== weap +0x3B8 weapon_interface_struct (chud interface @ +0x0C) ===");
        for t in native.tags_by_group(cache, "weap") {
            let Some(name) = native.tag_name(cache, t) else { continue };
            let leaf = name.rsplit('\\').next().unwrap_or(&name).to_string();
            if !want.contains(&leaf.as_str()) {
                continue;
            }
            let meta_len = (0..=5)
                .map(|i| 0x100u32 << i)
                .find(|&n| native.tag_meta(cache, t, 0, n).is_none())
                .map(|n| format!("<{n:#x}"))
                .unwrap_or_else(|| ">=0x2000".to_string());
            let Some(meta) = native.tag_meta(cache, t, 0, 0x400) else {
                println!("  {leaf:<22} tag_meta(0,0x400) = None   (reachable {meta_len})");
                continue;
            };
            let at = |o: usize| -> String {
                meta.get(o..o + 16)
                    .map(|b| b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" "))
                    .unwrap_or_else(|| "(past end)".into())
            };
            println!("  {leaf:<22} meta_len={} reachable={meta_len}", meta.len());
            println!("      +0x3B8 first-person block : {}", at(0x3B8));
            println!("      +0x3C4 chud interface     : {}", at(0x3C4));
            println!("      +0x3D4 alt state chud     : {}", at(0x3D4));
        }
    }
}
