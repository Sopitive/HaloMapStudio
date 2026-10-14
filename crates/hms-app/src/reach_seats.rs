//! #seats  Reach's `unit_seat_block`: the seats a vehicle authors, and the ENTRY rule.
//!
//! This is the tag half of "walk up to a vehicle and get in". Nothing here invents a radius, an
//! angle or a prompt: every number is read out of the `vehi`/`unit` tag, and the one thing the
//! tags do NOT answer — the English the engine prints — is reported as unresolved rather than
//! replaced with words of HMS's own.
//!
//! # The seats block had to be recovered, and it had FOUR candidates
//!
//! `reverseme tagdefs reach_tag_test.exe unit` prints the seats block as
//! `+?3AC block seats -> unit_seat_block [max 64]` with `struct unit_seat_block // 0 bytes
//! (no static field list: a runtime-defined struct; size not in the reflection data)`. It is one
//! of the runtime-installed field sets described in `re-notes/reach_runtime_installed_field_sets.md`:
//! the field-set record IS in the image (`.data` +0x2139430, name `unit_seat_block`, source
//! `omaha\units\unit_definitions.cpp`) but its `fields` pointer and element size read 0, because a
//! C++ dynamic initializer installs them at startup.
//!
//! The field ARRAY is in `.data` all along. Found by content — the 8-byte pointer to the string
//! `"entry radius#how close to the entry marker a unit must be"` sits at field record +0x08 — and
//! there are **four** of them, which is exactly the trap that note warns about. All four begin
//! with `long flags flags -> unit_seat_flags (32)` (they are the only four field records in the
//! image whose definition pointer is that enum):
//!
//! ```text
//!   .data +0x21365F0   22 records   260 bytes   (unit_seat_fields_v1)
//!   .data +0x21368B0   24 records   276 bytes   (unit_seat_fields_v2)
//!   .data +0x2136BB0   39 records   328 bytes   (unit_seat_fields_v3)
//!   .data +0x2137090   46 records   300 bytes   (unit_seat_fields_v4)  <- the current one
//! ```
//!
//! and the strings `unit_seat_fields_v1` … `_v4` sit together at `.rdata +0x1A13F00` in that
//! order, which is how the four arrays are ordered in `.data`. The newest is the SMALLEST (300 <
//! 328) because the upgrade reclaimed four `pad` runs; a size ordering would have picked the wrong
//! one.
//!
//! # The INSTALLER says all of it outright, and it agrees
//!
//! The content scan and the map-side discrimination came first; afterwards the dynamic
//! initializer itself was found and it states every number independently
//! (`reverseme ripxref reach_tag_test.exe 0x2137090` names it, then
//! `reverseme func reach_tag_test.exe unit_seat_fields_v1`):
//!
//! ```text
//!   declare_legacy_field_set(.., &unk_1421365F0, "unit_seat_fields_v1", 0x104, 0, -1, sub_140D877E0)
//!   declare_legacy_field_set(.., &unk_1421368B0, "unit_seat_fields_v2", 0x114, 0, -1, sub_140D877E0)
//!   declare_legacy_field_set(.., &unk_142136BB0, "unit_seat_fields_v3", 0x148, 0, -1, sub_140D877E0)
//!   declare_legacy_field_set(.., &unk_142137090, "unit_seat_fields_v4", 0x12C, 0, -1, 0)
//! ```
//!
//! The third argument is the ELEMENT SIZE: `0x104 = 260`, `0x114 = 276`, `0x148 = 328`,
//! **`0x12C = 300`** — the four sums the content scan computed, to the byte. And the last
//! argument is the UPGRADE function: `v1`..`v3` each carry one and **`v4` passes NULL**, which is
//! the engine's own statement of which set is current. So the layout is now confirmed twice from
//! independent directions — the engine's declaration and `forge_halo`'s own seats — and
//! [`gates::the_seat_layout_reads_as_seats`] keeps the second, because the first does not prove
//! the reader consumes it correctly.
//!
//! # The block's OFFSET inside the tag, and the self-check that proves it
//!
//! `tagdefs` cannot place the fields after `unit +0x348` either, because TWO more
//! runtime-installed structs sit in front of them. Both were recovered the same way:
//!
//! ```text
//!   unit_additional_node_names_struct ('uHnd', record .data +0x21396C0)
//!       array .data +0x2137F30 -> 2 string ids                        =   8 bytes
//!   unit_boarding_melee_struct        ('ubms', record .data +0x2139590)
//!       array .data +0x2137DD0 -> 9 tag references                    = 144 bytes
//! ```
//!
//! and their own installers state the same two numbers —
//! `declare_legacy_field_set(.., &unk_142137F30, "unit_additional_node_names", 8, ..)` and
//! `declare_legacy_field_set(.., &unk_142137DD0, "unit_boarding_melee_fields_v3", 0x90, ..)`,
//! `0x90 = 144`.
//!
//! so every `?` offset `tagdefs` prints from `+?348` onward is short by exactly `8 + 144 = 152`
//! (`0x98`). **That shift is not a preference: it is forced.** `tagdefs` states `unit`'s own
//! segment as 868 bytes (the compiler's own `sizeof(struct _unit_definition)`), the printed layout
//! ends at `+?434`, and `0x434 + 0x98 = 0x4CC = 360 + 868` — the exact offset at which `tagdefs`
//! independently starts `vehi`'s own fields. Two further corroborations, both from work done
//! before this module existed:
//!
//! * `item owner size` lands at `unit +0x3F6`, which is the offset HMS's Reach item-scale work
//!   measured from the engine (`re-notes`/memory: "selector = unit +0x3F6 item owner size");
//! * `seat acceleration` lands at `unit +0x2B4`, inside the statically-placed region, and
//!   `tagdefs` prints it there with no `?`.
//!
//! The corrected tail of `unit`, for the record:
//!
//! ```text
//!   +0x350 melee damage class     +0x354 melee damage (tag ref)   +0x364 boarding melee (144 B)
//!   +0x3F4 motion sensor blip     +0x3F6 item owner size          +0x3F8 equipment variant name
//!   +0x400 NEW HUD INTERFACES     +0x40C grenade angle            +0x418 grenade velocity
//!   +0x420 powered seats          +0x42C weapons                  +0x438 target tracking
//!   +0x444 SEATS                  +0x450 open time                +0x454 closing time
//!   +0x458 emp disabled time      +0x45C emp disabled effect      +0x46C boost (64 B)
//!   +0x4AC exit and detach damage +0x4BC exit and detach weapon    end 0x4CC
//! ```
//!
//! # What the entry rule actually is
//!
//! Three authored gates, and the field help text in the tool build states each one:
//!
//! * `entry radius` — "how close to the entry marker a unit must be", with the authors' own note
//!   on the block: *"note: the entry radius shouldn't exceed 3 world units, as that is as far as
//!   the player will search for a vehicle to enter."* That sentence is the reason
//!   [`gates::the_seat_layout_reads_as_seats`] can discriminate the four candidates at all.
//! * `entry marker cone angle` — "angle from marker forward the unit must be"
//! * `entry marker facing angle` — "angle from unit facing the marker must be"
//!
//! so proximity is not enough: the player must be inside a cone around the entry marker's OWN
//! forward, and the marker must be inside a cone around the player's facing.
//!
//! # What is NOT here
//!
//! * **The prompt's English.** `enter seat string` and `label` are `string_id`s; the engine
//!   resolves them through the localized string tables. [`SeatPrompt`] carries the ids and says
//!   whether anything resolved; a caller that cannot resolve must draw nothing.
//! * **Seat occupancy and the engine's tie-break between two seats in range.** [`best_seat`]
//!   implements the authored gates and a stated preference order; it is not claimed to be the
//!   engine's own loop.

use hms_native::{CacheHandle, NativeDll};

// ---------------------------------------------------------------------------------------------
// THE LAYOUT
// ---------------------------------------------------------------------------------------------

/// `unit`'s `seats` block descriptor: element count at +0x00, meta pointer at +0x04.
///
/// Recovered as described in the module comment; `0x344 + 0x98 + 0x64`… no: it is literally
/// `tagdefs`'s `+?3AC` plus the 152 bytes of the two runtime-installed structs in front of it.
pub const UNIT_SEATS: usize = 0x444;

/// Every `unit_seat_block` element size the image offers, newest first. The gates assert that
/// exactly one of these reads as seats on a real map and that the others do not.
pub const SEAT_STRIDE_CANDIDATES: [usize; 4] = [300, 328, 276, 260];

/// `unit_seat_fields_v4`, the current element size.
pub const SEAT_STRIDE: usize = 300;

/// `MAXIMUM_SEATS_PER_UNIT_DEFINITION`, straight off the block definition's max-count name.
pub const SEAT_MAX: usize = 64;

// Offsets inside one seat element (v4).
pub const S_FLAGS: usize = 0x000;
pub const S_LABEL: usize = 0x004;
pub const S_MARKER_NAME: usize = 0x008;
pub const S_ENTRY_MARKERS: usize = 0x00C;
pub const S_BOARDING_GRENADE_MARKER: usize = 0x010;
pub const S_BOARDING_GRENADE_STRING: usize = 0x014;
pub const S_BOARDING_MELEE_STRING: usize = 0x018;
pub const S_IN_SEAT_STRING: usize = 0x01C;
pub const S_PING_SCALE: usize = 0x020;
pub const S_TURNOVER_TIME: usize = 0x024;
pub const S_SEAT_ACCELERATION: usize = 0x028;
pub const S_AI_SCARINESS: usize = 0x038;
pub const S_AI_SEAT_TYPE: usize = 0x03C;
pub const S_BOARDING_SEAT: usize = 0x03E;
pub const S_ADDITIONAL_BOARDING_SEATS: usize = 0x040;
pub const S_LISTENER_INTERP: usize = 0x04C;
pub const S_YAW_RATE_BOUNDS: usize = 0x050;
pub const S_PITCH_RATE_BOUNDS: usize = 0x058;
pub const S_PITCH_INTERP_TIME: usize = 0x060;
pub const S_MIN_SPEED_REFERENCE: usize = 0x064;
pub const S_MAX_SPEED_REFERENCE: usize = 0x068;
pub const S_SPEED_EXPONENT: usize = 0x06C;
/// The seat's own `unit_camera_struct`, 120 bytes.
pub const S_CAMERA: usize = 0x070;
pub const S_HUD_INTERFACE: usize = 0x0E8;
pub const S_ENTER_SEAT_STRING: usize = 0x0F4;
pub const S_YAW_MINIMUM: usize = 0x0F8;
pub const S_YAW_MAXIMUM: usize = 0x0FC;
pub const S_ENTRY_RADIUS: usize = 0x100;
pub const S_ENTRY_CONE_ANGLE: usize = 0x104;
pub const S_ENTRY_FACING_ANGLE: usize = 0x108;
pub const S_MAX_RELATIVE_VELOCITY: usize = 0x10C;
pub const S_OPEN_TIME: usize = 0x110;
pub const S_CLOSE_TIME: usize = 0x114;
pub const S_OPEN_FUNCTION: usize = 0x118;
pub const S_OPENING_FUNCTION: usize = 0x11C;
pub const S_CLOSING_FUNCTION: usize = 0x120;
pub const S_INVISIBLE_SEAT_REGION: usize = 0x124;
pub const S_RUNTIME_INVISIBLE_REGION: usize = 0x128;

// Offsets inside the seat's `unit_camera_struct`, relative to `S_CAMERA`.
pub const CAM_FLAGS: usize = 0x000;
pub const CAM_MARKER_NAME: usize = 0x004;
pub const CAM_PITCH_AUTO_LEVEL: usize = 0x008;
pub const CAM_PITCH_RANGE: usize = 0x00C;
pub const CAM_OVERRIDE_FOV: usize = 0x038;
pub const CAM_SIZE: usize = 0x078;

/// `unit_seat_flags`, in the tool build's own order (bit index == this index).
pub const SEAT_FLAG_NAMES: [&str; 32] = [
    "invisible",
    "locked",
    "driver",
    "gunner",
    "third person camera",
    "allows weapons",
    "third person on enter",
    "first person camera slaved to gun",
    "not valid without driver",
    "allow AI noncombatants",
    "boarding seat",
    "ai firing disabled by max acceleration",
    "boarding enters seat",
    "boarding need any passenger",
    "invalid for player",
    "invalid for non-player",
    "invalid for hero",
    "gunner (player only)",
    "invisible under major damage",
    "melee instant killable",
    "leader preference",
    "allows exit and detach",
    "blocks headshots",
    "exits to ground",
    "forward from attachment",
    "disallow AI shooting",
    "prevents weapon stowing",
    "takes top level aoe damage",
    "disallow exit",
    "local aiming",
    "pelvis relative attachment",
    "apply velocity on death exit",
];

pub const FLAG_INVISIBLE: u32 = 1 << 0;
pub const FLAG_LOCKED: u32 = 1 << 1;
pub const FLAG_DRIVER: u32 = 1 << 2;
pub const FLAG_GUNNER: u32 = 1 << 3;
pub const FLAG_THIRD_PERSON_CAMERA: u32 = 1 << 4;
pub const FLAG_ALLOWS_WEAPONS: u32 = 1 << 5;
pub const FLAG_NOT_VALID_WITHOUT_DRIVER: u32 = 1 << 8;
pub const FLAG_INVALID_FOR_PLAYER: u32 = 1 << 14;
pub const FLAG_GUNNER_PLAYER_ONLY: u32 = 1 << 17;
pub const FLAG_DISALLOW_EXIT: u32 = 1 << 28;

/// `global_ai_seat_type_enum`.
pub const AI_SEAT_TYPE_NAMES: [&str; 6] = ["none", "passenger", "gunner", "small cargo", "large cargo", "driver"];

// ---------------------------------------------------------------------------------------------
// WHAT ONE SEAT IS
// ---------------------------------------------------------------------------------------------

/// One authored seat, as read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Seat {
    /// Index in the block — this is the seat index the engine and the saved-film attachment
    /// record both carry.
    pub index: usize,
    pub flags: u32,
    /// `label`, a `string_id`: the seat's own name (`"driver"`, `"passenger"`, `"gunner"`).
    pub label: String,
    /// `marker name`: the model marker the occupant is parented to.
    pub marker_name: String,
    /// `entry marker(s) name`: the marker group the entry test runs against. Empty means the
    /// engine falls back to the seat's own `marker name`.
    pub entry_markers: String,
    /// `entry radius`, world units. The block's own note: should not exceed 3.
    pub entry_radius: f32,
    /// `entry marker cone angle`, as stored. **Blam stores an `angle` field in radians**; the
    /// gate prints the raw value so the units are a measurement, not an assumption.
    pub entry_cone_angle: f32,
    /// `entry marker facing angle`, as stored.
    pub entry_facing_angle: f32,
    /// `maximum relative velocity`: above this the engine refuses entry.
    pub max_relative_velocity: f32,
    /// `enter seat string`, a `string_id` naming the localized prompt.
    pub enter_seat_string: String,
    /// `in-seat string`, the prompt shown while riding.
    pub in_seat_string: String,
    pub ai_seat_type: u8,
    /// `boarding seat`, a block index into this same seats block, -1 when none.
    pub boarding_seat: i16,
    /// The seat camera's `camera marker name`, empty when the seat does not name one.
    pub camera_marker: String,
    /// The seat camera's `override fov`, as stored (0 = inherit).
    pub camera_override_fov: f32,
    /// `pitch range`, the seat's own look limits, as stored.
    pub pitch_range: [f32; 2],
    /// `yaw minimum` / `yaw maximum`, as stored.
    pub yaw_range: [f32; 2],
    /// `max speed reference` / `speed exponent`: how the yaw/pitch rate bounds interpolate.
    pub max_speed_reference: f32,
    pub speed_exponent: f32,
    pub yaw_rate_bounds: [f32; 2],
    pub pitch_rate_bounds: [f32; 2],
}

impl Seat {
    pub fn is_driver(&self) -> bool {
        self.flags & FLAG_DRIVER != 0
    }
    pub fn is_gunner(&self) -> bool {
        self.flags & (FLAG_GUNNER | FLAG_GUNNER_PLAYER_ONLY) != 0
    }
    pub fn third_person(&self) -> bool {
        self.flags & FLAG_THIRD_PERSON_CAMERA != 0
    }
    /// Can a PLAYER take this seat at all? `invalid for player` and `locked` are the two the
    /// tag states outright.
    pub fn player_allowed(&self) -> bool {
        self.flags & (FLAG_INVALID_FOR_PLAYER | FLAG_LOCKED) == 0
    }
    /// The flag names set on this seat, for a diagnostic.
    pub fn flag_names(&self) -> Vec<&'static str> {
        (0..32).filter(|b| self.flags & (1 << b) != 0).map(|b| SEAT_FLAG_NAMES[b]).collect()
    }
    /// The marker group the entry test uses: `entry marker(s) name` when authored, else the
    /// seat's own `marker name`.
    pub fn entry_marker_group(&self) -> &str {
        if self.entry_markers.is_empty() { &self.marker_name } else { &self.entry_markers }
    }
    /// The authored prompt ids, and nothing else. A caller with no localized-string reader must
    /// draw NOTHING rather than substitute English.
    pub fn prompt(&self) -> SeatPrompt {
        SeatPrompt {
            enter_seat_string: self.enter_seat_string.clone(),
            label: self.label.clone(),
            seat_index: self.index,
        }
    }
}

/// The ids that name a seat's entry prompt. Carries no English: resolving these is the
/// localization layer's job, and when it fails the honest answer is to draw nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeatPrompt {
    /// `enter seat string`, the `string_id` the engine resolves to the prompt.
    pub enter_seat_string: String,
    /// `label`, the seat's own name — the fallback the engine uses when no prompt string is
    /// authored (and still a `string_id`, not English).
    pub label: String,
    pub seat_index: usize,
}

impl SeatPrompt {
    /// The id this prompt would be drawn from, or `None` when the seat authors neither.
    pub fn id(&self) -> Option<&str> {
        if !self.enter_seat_string.is_empty() {
            Some(&self.enter_seat_string)
        } else if !self.label.is_empty() {
            Some(&self.label)
        } else {
            None
        }
    }
}

/// A unit's seats, as read from its tag.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnitSeats {
    pub unit_tag: u32,
    pub seats: Vec<Seat>,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() { return 0 }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_i16(b: &[u8], o: usize) -> i16 {
    if o + 2 > b.len() { return 0 }
    i16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(b, o))
}
fn sid(native: &NativeDll, cache: CacheHandle, v: u32) -> String {
    if v == 0 || v == u32::MAX { return String::new() }
    native.string_id(cache, v).unwrap_or_default()
}

impl UnitSeats {
    /// Read a `unit`-derived tag's seats (`vehi`, `bipd`, `giant`, …). `None` when the tag does
    /// not read to the seats block or the block is empty.
    pub fn read(native: &NativeDll, cache: CacheHandle, unit_tag: u32) -> Option<UnitSeats> {
        Self::read_with_stride(native, cache, unit_tag, SEAT_STRIDE)
    }

    /// The same read with an explicit element size, so the gate can try every candidate the
    /// image offers against the same map and the same tags.
    pub fn read_with_stride(native: &NativeDll, cache: CacheHandle, unit_tag: u32, stride: usize) -> Option<UnitSeats> {
        let h = native.tag_meta(cache, unit_tag, 0, (UNIT_SEATS + 8) as u32)?;
        let count = rd_u32(&h, UNIT_SEATS) as usize;
        let ptr = rd_u32(&h, UNIT_SEATS + 4);
        if count == 0 || count > SEAT_MAX { return None }
        let b = native.tag_ptr_bytes(cache, ptr, (count * stride) as u32)?;
        let mut seats = Vec::with_capacity(count);
        for i in 0..count {
            let e = &b[i * stride..(i + 1) * stride];
            let cam = S_CAMERA;
            seats.push(Seat {
                index: i,
                flags: rd_u32(e, S_FLAGS),
                label: sid(native, cache, rd_u32(e, S_LABEL)),
                marker_name: sid(native, cache, rd_u32(e, S_MARKER_NAME)),
                entry_markers: sid(native, cache, rd_u32(e, S_ENTRY_MARKERS)),
                entry_radius: rd_f32(e, S_ENTRY_RADIUS),
                entry_cone_angle: rd_f32(e, S_ENTRY_CONE_ANGLE),
                entry_facing_angle: rd_f32(e, S_ENTRY_FACING_ANGLE),
                max_relative_velocity: rd_f32(e, S_MAX_RELATIVE_VELOCITY),
                enter_seat_string: sid(native, cache, rd_u32(e, S_ENTER_SEAT_STRING)),
                in_seat_string: sid(native, cache, rd_u32(e, S_IN_SEAT_STRING)),
                ai_seat_type: rd_u32(e, S_AI_SEAT_TYPE) as u8,
                boarding_seat: rd_i16(e, S_BOARDING_SEAT),
                camera_marker: sid(native, cache, rd_u32(e, cam + CAM_MARKER_NAME)),
                camera_override_fov: rd_f32(e, cam + CAM_OVERRIDE_FOV),
                pitch_range: [rd_f32(e, cam + CAM_PITCH_RANGE), rd_f32(e, cam + CAM_PITCH_RANGE + 4)],
                yaw_range: [rd_f32(e, S_YAW_MINIMUM), rd_f32(e, S_YAW_MAXIMUM)],
                max_speed_reference: rd_f32(e, S_MAX_SPEED_REFERENCE),
                speed_exponent: rd_f32(e, S_SPEED_EXPONENT),
                yaw_rate_bounds: [rd_f32(e, S_YAW_RATE_BOUNDS), rd_f32(e, S_YAW_RATE_BOUNDS + 4)],
                pitch_rate_bounds: [rd_f32(e, S_PITCH_RATE_BOUNDS), rd_f32(e, S_PITCH_RATE_BOUNDS + 4)],
            });
        }
        Some(UnitSeats { unit_tag, seats })
    }

    /// The seat a player would be offered, of those whose authored tests `cands` says passed.
    ///
    /// **The preference order is HMS's own and is stated as such**: driver first, then gunner,
    /// then the lowest seat index. The engine's own tie-break has not been read out of
    /// `haloreach.dll`, so this does not claim to be it; what IS the tag's is every gate that got
    /// a seat into `cands` in the first place.
    pub fn best(&self, cands: &[usize]) -> Option<usize> {
        cands
            .iter()
            .copied()
            .filter(|&i| self.seats.get(i).is_some_and(|s| s.player_allowed()))
            .min_by_key(|&i| {
                let s = &self.seats[i];
                let rank = if s.is_driver() { 0 } else if s.is_gunner() { 1 } else { 2 };
                (rank, i)
            })
    }

    /// One line per seat, for a diagnostic or a report.
    pub fn describe(&self) -> Vec<String> {
        self.seats
            .iter()
            .map(|s| {
                format!(
                    "  [{}] label '{}' marker '{}' entry '{}' r={:.2} cone={:.4} facing={:.4} \
                     maxrelvel={:.1} enter='{}' cam='{}' fov={:.4} flags[{}]",
                    s.index,
                    s.label,
                    s.marker_name,
                    s.entry_markers,
                    s.entry_radius,
                    s.entry_cone_angle,
                    s.entry_facing_angle,
                    s.max_relative_velocity,
                    s.enter_seat_string,
                    s.camera_marker,
                    s.camera_override_fov,
                    s.flag_names().join(", ")
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// #seat-entry  THE ENTRY TEST, and the SEAT VIEW
// ---------------------------------------------------------------------------------------------

use glam::{Mat4, Vec3};

/// A marker's own FORWARD axis, in Blam's object convention: **+X forward, +Y left, +Z up**.
///
/// This is not taken from the convention alone —
/// [`gates::the_entry_markers_point_away_from_the_vehicle`] measures it: an entry marker must
/// point at where the player stands, which is OUTWARD from the vehicle's own origin, and over
/// every entry marker in the cache only one axis is consistently outward.
pub const MARKER_FORWARD: Vec3 = Vec3::X;

/// What the three authored entry gates measured, and whether each passed. Carried as numbers so
/// a diagnostic can say *which* gate refused rather than just "you cannot get in".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntryTest {
    /// Distance from the entry marker, world units, against `entry radius`.
    pub distance: f32,
    pub radius: f32,
    /// Angle between the marker's own forward and the direction marker -> player, against
    /// `entry marker cone angle`. Radians.
    pub cone: f32,
    pub cone_limit: f32,
    /// Angle between the player's facing and the direction player -> marker, against
    /// `entry marker facing angle`. Radians.
    pub facing: f32,
    pub facing_limit: f32,
    /// Speed of the player RELATIVE to the vehicle, against `maximum relative velocity`.
    pub relative_speed: f32,
    pub relative_speed_limit: f32,
}

impl EntryTest {
    pub fn within_radius(&self) -> bool { self.distance <= self.radius }
    pub fn within_cone(&self) -> bool { self.cone <= self.cone_limit }
    pub fn within_facing(&self) -> bool { self.facing <= self.facing_limit }
    pub fn slow_enough(&self) -> bool { self.relative_speed <= self.relative_speed_limit }
    /// All four authored gates.
    pub fn passes(&self) -> bool {
        self.within_radius() && self.within_cone() && self.within_facing() && self.slow_enough()
    }
    /// Which gate refused, for a caption. `None` when it passed.
    pub fn refusal(&self) -> Option<&'static str> {
        if !self.within_radius() {
            Some("too far from the entry marker")
        } else if !self.within_cone() {
            Some("not inside the entry marker's cone")
        } else if !self.within_facing() {
            Some("not looking at the entry marker")
        } else if !self.slow_enough() {
            Some("moving too fast relative to the vehicle")
        } else {
            None
        }
    }
    /// A one-line diagnostic of every number, so a wrong limit is visible rather than felt.
    pub fn describe(&self) -> String {
        format!(
            "d {:.2}/{:.2}{} cone {:.3}/{:.3}{} facing {:.3}/{:.3}{} relvel {:.2}/{:.2}{}",
            self.distance, self.radius, if self.within_radius() { "" } else { " X" },
            self.cone, self.cone_limit, if self.within_cone() { "" } else { " X" },
            self.facing, self.facing_limit, if self.within_facing() { "" } else { " X" },
            self.relative_speed, self.relative_speed_limit, if self.slow_enough() { "" } else { " X" },
        )
    }
}

/// The angle between two directions, in radians, safe for zero-length input (which returns
/// `PI` — "as wrong as possible" — so a degenerate case refuses rather than passes).
fn angle_between(a: Vec3, b: Vec3) -> f32 {
    let (la, lb) = (a.length(), b.length());
    if !(la > 1e-6) || !(lb > 1e-6) {
        return std::f32::consts::PI;
    }
    ((a / la).dot(b / lb)).clamp(-1.0, 1.0).acos()
}

/// Run a seat's four authored entry gates against one entry marker.
///
/// `marker_world` is the entry marker's frame in WORLD space (the vehicle's transform times the
/// posed marker), `player_pos` the player's position, `player_facing` where they are looking, and
/// `relative_velocity` the player's velocity minus the vehicle's.
///
/// Every limit comes from the tag. The only thing this function chooses is [`MARKER_FORWARD`],
/// and that is measured.
pub fn entry_test(seat: &Seat, marker_world: Mat4, player_pos: Vec3, player_facing: Vec3, relative_velocity: Vec3) -> EntryTest {
    let marker_pos = marker_world.w_axis.truncate();
    let marker_fwd = marker_world.transform_vector3(MARKER_FORWARD);
    let to_player = player_pos - marker_pos;
    EntryTest {
        distance: to_player.length(),
        radius: seat.entry_radius,
        cone: angle_between(marker_fwd, to_player),
        cone_limit: seat.entry_cone_angle,
        facing: angle_between(player_facing, -to_player),
        facing_limit: seat.entry_facing_angle,
        relative_speed: relative_velocity.length(),
        relative_speed_limit: seat.max_relative_velocity,
    }
}

/// The best of several entry markers for one seat: the marker group may hold more than one
/// instance (`entry marker(s) name` is plural in the editor), and the engine accepts any of them.
/// Returns the index of the marker that PASSES and is closest, else the closest test so a
/// diagnostic can say why none did.
pub fn best_entry_marker(
    seat: &Seat,
    markers: &[Mat4],
    player_pos: Vec3,
    player_facing: Vec3,
    relative_velocity: Vec3,
) -> Option<(usize, EntryTest)> {
    let mut best: Option<(usize, EntryTest)> = None;
    for (i, &m) in markers.iter().enumerate() {
        let t = entry_test(seat, m, player_pos, player_facing, relative_velocity);
        let better = match &best {
            None => true,
            Some((_, b)) => match (t.passes(), b.passes()) {
                (true, false) => true,
                (false, true) => false,
                _ => t.distance < b.distance,
            },
        };
        if better {
            best = Some((i, t));
        }
    }
    best
}

/// Where the camera goes for a seat, computed from the vehicle's pose and the seat's own
/// `unit_camera_struct`. **A pure function on purpose**: a camera derived from the vehicle
/// transform cannot go stale the way a field written in one place and read in another can.
///
/// `camera_marker_world` is the seat's `camera marker name` marker in world space when the seat
/// authors one; `seat_marker_world` is the seat's own marker, used when it does not.
/// `(yaw, pitch)` is the rider's look, already clamped by the caller to the seat's authored
/// `pitch range` / `yaw minimum` / `yaw maximum`. `fov_default` is the game's FOV, used when the
/// seat authors no `override fov`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeatView {
    pub eye: Vec3,
    pub forward: Vec3,
    /// Radians. The seat's `override fov` when it authors one, else `fov_default`.
    pub fov: f32,
    /// True when the seat's `third person camera` flag is set, so a caller can pull the eye back
    /// along `-forward` by whatever stand-off it uses. The FLAG is the tag's; the distance is not
    /// authored in the seat and is therefore not invented here.
    pub third_person: bool,
}

pub fn seat_view(
    seat: &Seat,
    camera_marker_world: Option<Mat4>,
    seat_marker_world: Mat4,
    yaw: f32,
    pitch: f32,
    fov_default: f32,
) -> SeatView {
    let frame = camera_marker_world.unwrap_or(seat_marker_world);
    let eye = frame.w_axis.truncate();
    // The rider's look is in WORLD yaw/pitch, as the on-foot controller's is: a seat limits how
    // far it may swing (`yaw minimum`/`maximum`, `pitch range`) but does not re-base it.
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let forward = Vec3::new(cy * cp, sy * cp, sp);
    SeatView {
        eye,
        forward,
        fov: if seat.camera_override_fov > 0.0 { seat.camera_override_fov } else { fov_default },
        third_person: seat.third_person(),
    }
}

/// Clamp a rider's look to the seat's authored limits, relative to the vehicle's own yaw.
///
/// `yaw minimum` / `yaw maximum` and `pitch range` are authored per seat; a seat that authors
/// `0, 0` for a pair does not limit that axis (nothing in the tag distinguishes "no limit" from
/// "pinned", and every seat in the cache that pins would be unusable, so zero is no limit).
pub fn clamp_seat_look(seat: &Seat, vehicle_yaw: f32, yaw: f32, pitch: f32) -> (f32, f32) {
    let wrap = |a: f32| {
        let t = std::f32::consts::TAU;
        let mut v = (a + std::f32::consts::PI).rem_euclid(t) - std::f32::consts::PI;
        if v == -std::f32::consts::PI {
            v = std::f32::consts::PI;
        }
        v
    };
    let [ymin, ymax] = seat.yaw_range;
    let yaw = if ymin != 0.0 || ymax != 0.0 {
        vehicle_yaw + wrap(yaw - vehicle_yaw).clamp(ymin.min(ymax), ymin.max(ymax))
    } else {
        yaw
    };
    let [pmin, pmax] = seat.pitch_range;
    let pitch = if pmin != 0.0 || pmax != 0.0 { pitch.clamp(pmin.min(pmax), pmin.max(pmax)) } else { pitch };
    (yaw, pitch)
}

// ---------------------------------------------------------------------------------------------
// THE GATES
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod gates {
    //! #seats  The seats layout, measured against a shipped Reach cache.
    //!
    //! `cargo test -p hms-app --profile quick the_seat -- --ignored --nocapture`

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// TWO predicates, kept apart on purpose.
    ///
    /// `resolves` is "this really is a seat": a `label` and a `marker name` that both resolve
    /// through the cache's own string-id table, and a positive finite `entry radius`. Zeros fail
    /// it and so do arbitrary bytes.
    ///
    /// `in_band` adds the authors' own note on the block — *"the entry radius shouldn't exceed 3
    /// world units"* — and it is the HARDER test, which is why the stride discrimination runs on
    /// it. It is **not** a correctness requirement: `phantom_mp` authors 10.0 on its driver and
    /// its door pseudo-seats, i.e. the authors exceeded their own note on a dropship. A gate that
    /// demanded the band outright would have called the correct layout wrong.
    fn resolves(s: &Seat) -> bool {
        !s.label.is_empty() && !s.marker_name.is_empty() && s.entry_radius.is_finite() && s.entry_radius > 0.0
    }
    fn in_band(s: &Seat) -> bool {
        resolves(s) && s.entry_radius <= 3.0
    }

    /// **THE STRIDE GATE.** Of the four `unit_seat_block` field arrays in `reach_tag_test.exe`'s
    /// `.data`, exactly one reads as seats on every multi-seat vehicle in `forge_halo`, and the
    /// other three do not.
    ///
    /// The bite is **element 1 and beyond**: a stride is only testable on a unit with two or more
    /// seats, because seat 0 starts at the block pointer whatever the stride is. That is the
    /// mistake `weapon_barrels` made and this gate is built not to repeat it — it asserts the
    /// per-seat count on the SECOND and later seats separately.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_seat_layout_reads_as_seats() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_seats gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let vehis = native.tags_by_group(cache, "vehi");
        assert!(vehis.len() > 10, "forge_halo should carry many vehi tags, found {}", vehis.len());

        // Which vehicles even have a seats block, and how many seats, at the chosen stride?
        let mut multi: Vec<(u32, String, usize)> = Vec::new();
        for &v in &vehis {
            if let Some(u) = UnitSeats::read_with_stride(native, cache, v, SEAT_STRIDE) {
                if u.seats.len() >= 2 {
                    multi.push((v, native.tag_name(cache, v).unwrap_or_default(), u.seats.len()));
                }
            }
        }
        assert!(
            multi.len() >= 3,
            "need several multi-seat vehicles to test a STRIDE; found {} -- a gate that only ever \
             reads seat 0 cannot discriminate",
            multi.len()
        );
        println!("{} vehi tags, {} with 2+ seats", vehis.len(), multi.len());

        // Per candidate: how many seats read as seats, counted separately for seat 0 and for
        // seats 1.. (the only ones a stride can get wrong).
        // Per candidate: how many seats read as seats, counted separately for seat 0 and for
        // seats 1.. (the only ones a stride can get wrong).
        struct Row { stride: usize, r0: (usize, usize), rrest: (usize, usize), b0: (usize, usize), brest: (usize, usize) }
        let mut table: Vec<Row> = Vec::new();
        for stride in SEAT_STRIDE_CANDIDATES {
            let mut row = Row { stride, r0: (0, 0), rrest: (0, 0), b0: (0, 0), brest: (0, 0) };
            let mut over_band: Vec<String> = Vec::new();
            let mut broken: Vec<String> = Vec::new();
            for &v in &vehis {
                let Some(u) = UnitSeats::read_with_stride(native, cache, v, stride) else { continue };
                for s in &u.seats {
                    let (r, b) = (resolves(s), in_band(s));
                    let (rt, bt) = if s.index == 0 { (&mut row.r0, &mut row.b0) } else { (&mut row.rrest, &mut row.brest) };
                    rt.1 += 1;
                    rt.0 += r as usize;
                    bt.1 += 1;
                    bt.0 += b as usize;
                    if stride == SEAT_STRIDE {
                        let line = format!(
                            "{} seat {} label '{}' marker '{}' r={:.3}",
                            native.tag_name(cache, v).unwrap_or_default(), s.index, s.label, s.marker_name, s.entry_radius
                        );
                        if !r { broken.push(line) } else if !b { over_band.push(line) }
                    }
                }
            }
            println!(
                "  stride {stride:>3}: resolves as a seat -- seat 0 {}/{}, seats 1.. {}/{}  |  \
                 entry radius also in (0,3] -- seat 0 {}/{}, seats 1.. {}/{}",
                row.r0.0, row.r0.1, row.rrest.0, row.rrest.1, row.b0.0, row.b0.1, row.brest.0, row.brest.1
            );
            for l in &broken { println!("      DOES NOT RESOLVE: {l}") }
            for l in &over_band { println!("      resolves, radius over the authors' own 3 wu note: {l}") }
            table.push(row);
        }

        let chosen = table.iter().find(|t| t.stride == SEAT_STRIDE).expect("the chosen stride is in the table");
        // Every seat, first and later, must RESOLVE. No exceptions: this is the layout claim.
        assert_eq!(chosen.r0.0, chosen.r0.1, "stride {SEAT_STRIDE}: seat 0 does not resolve on {} tags", chosen.r0.1 - chosen.r0.0);
        assert_eq!(
            chosen.rrest.0, chosen.rrest.1,
            "stride {SEAT_STRIDE}: {} of {} later seats do NOT resolve as seats",
            chosen.rrest.1 - chosen.rrest.0, chosen.rrest.1
        );
        assert!(chosen.rrest.1 > 0, "no later seats were examined -- the stride was never tested");
        // And the authors' own band holds on all but a named handful.
        assert!(
            chosen.brest.1 - chosen.brest.0 <= 2 && chosen.b0.1 - chosen.b0.0 <= 1,
            "stride {SEAT_STRIDE}: {} seats sit outside the authors' own (0,3] entry-radius note; \
             two on `phantom_mp` are expected and more than that says the field is not the radius",
            (chosen.b0.1 - chosen.b0.0) + (chosen.brest.1 - chosen.brest.0)
        );

        // And every rejected candidate must FAIL on the later seats, on BOTH predicates. This is
        // the half that makes the answer a measurement: a layout nobody compared against the
        // alternatives is a preference.
        for t in &table {
            if t.stride == SEAT_STRIDE { continue }
            assert!(
                t.rrest.0 < t.rrest.1 && t.brest.0 * 10 < t.brest.1,
                "rejected stride {} reads {}/{} later seats as seats ({}/{} even inside the radius \
                 band) -- this gate does not discriminate and the chosen stride is not measured",
                t.stride, t.rrest.0, t.rrest.1, t.brest.0, t.brest.1
            );
        }

        // What the chosen layout actually says, printed so the numbers can be checked by eye.
        for (v, name, n) in &multi {
            let u = UnitSeats::read(native, cache, *v).unwrap();
            println!("\n{name}  ({n} seats)");
            for l in u.describe() {
                println!("{l}");
            }
        }
    }

    /// **THE AUTHORED-VALUE GATE.** The entry parameters must read as parameters somebody tuned:
    /// a radius in the authors' own band on every seat of every vehicle, a cone angle that is a
    /// real angle, and `driver` set on exactly one seat of a drivable vehicle.
    ///
    /// Zeros satisfy none of these, which is the point: an ordering assertion would have.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_entry_parameters_are_authored() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_seats gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let vehis = native.tags_by_group(cache, "vehi");
        let mut seats_seen = 0usize;
        let mut radius_band = 0usize;
        let mut cone_positive = 0usize;
        let mut with_driver = 0usize;
        let mut vehicles = 0usize;
        let mut rmin = f32::MAX;
        let mut rmax = 0f32;
        let mut cmin = f32::MAX;
        let mut cmax = 0f32;
        for &v in &vehis {
            let Some(u) = UnitSeats::read(native, cache, v) else { continue };
            vehicles += 1;
            let drivers = u.seats.iter().filter(|s| s.is_driver()).count();
            // A `vehi` either offers a player a driver's or a gunner's seat, or it offers a
            // player NOTHING -- `warthog_troop`, the rear bench, authors three passenger seats
            // and marks all three `invalid for player`. All three cases are the tag's.
            let open_to_player = u.seats.iter().any(|s| s.player_allowed());
            if drivers == 1 || u.seats.iter().any(|s| s.is_gunner()) || !open_to_player {
                with_driver += 1;
            } else {
                println!("  no driver, no gunner, yet open to a player: {}", native.tag_name(cache, v).unwrap_or_default());
            }
            assert!(drivers <= 1, "{} authors {drivers} driver seats", native.tag_name(cache, v).unwrap_or_default());
            for s in &u.seats {
                seats_seen += 1;
                assert!(s.entry_radius.is_finite() && s.entry_cone_angle.is_finite(), "non-finite entry params on {}", s.label);
                if s.entry_radius > 0.0 && s.entry_radius <= 3.0 {
                    radius_band += 1;
                    rmin = rmin.min(s.entry_radius);
                    rmax = rmax.max(s.entry_radius);
                }
                if s.entry_cone_angle > 0.0 {
                    cone_positive += 1;
                    cmin = cmin.min(s.entry_cone_angle);
                    cmax = cmax.max(s.entry_cone_angle);
                }
            }
        }
        println!(
            "{vehicles} vehicles, {seats_seen} seats: {radius_band} with entry radius in (0,3] \
             [{rmin:.2}..{rmax:.2}], {cone_positive} with a positive cone angle [{cmin:.4}..{cmax:.4}], \
             {with_driver} offering a player a driver/gunner seat or nothing at all"
        );
        assert!(
            seats_seen - radius_band <= 3,
            "{} seats have an entry radius outside (0,3] -- `phantom_mp` authors 10.0 on three of \
             its own, and anything beyond that says the field is not the radius",
            seats_seen - radius_band
        );
        assert_eq!(cone_positive, seats_seen, "{} seats have a non-positive entry cone angle", seats_seen - cone_positive);
        // Not "every vehicle has a driver": the turrets and the Warthog's three weapon variants
        // are `vehi` tags in their own right with a GUNNER and no driver, which is why the
        // obvious assertion fails on 11 of 35 and the right one is "a seat a player can take".
        assert_eq!(
            with_driver, vehicles,
            "{} of {vehicles} vehicles offer a player a seat that is neither a driver's nor a gunner's",
            vehicles - with_driver
        );
        // 2*pi is the largest cone in the cache (the Mongoose driver and the Banshee boarders
        // author pi), which is what settles RADIANS: a degrees field would read 360.
        assert!(cmax > 6.0 && cmax <= std::f32::consts::TAU + 1e-3, "cone angles top out at {cmax}");
        // Radians or degrees is a MEASUREMENT, not an assumption: an `angle` field stored in
        // radians cannot exceed 2*pi, and a cone of 90 degrees reads 1.5708.
        assert!(cmax <= std::f32::consts::TAU, "cone angles read up to {cmax} -- these are not radians");
    }

    /// What `unic` tags the cache carries, and which one the seat prompt ids live in. A
    /// diagnostic, so the tag path in [`super::HUD_MESSAGES_UNIC`] is read off a map rather than
    /// taken on trust.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_prompt_strings_live_in_a_unic_the_cache_names() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_seats gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let unics = native.tags_by_group(cache, "unic");
        println!("{} unic tags:", unics.len());
        for &u in &unics {
            println!("   {:#010x}  {}", u, native.tag_name(cache, u).unwrap_or_default());
        }
    }

    /// Print-only: the per-unit and per-vehicle SPEED-like fields nothing else reads, so the
    /// question "what is a Warthog's authored top speed" can be answered from data.
    ///
    /// `vehi +0x600 blur speed` and `unit +0x33C object velocity maximum` ("Debug value for
    /// object velocity that corresponds to a blend screen weight of 1, 0 defaults to 5.0"), next
    /// to the `alien_scout` families' explicit `maximum forward speed` for scale, and the driver
    /// seat's own `yaw rate bounds` / `max speed reference` / `speed exponent`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_vehicle_speed_fields() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let rdf = |t: u32, off: u32| {
            native
                .tag_meta(cache, t, off, 4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .filter(|v| v.is_finite())
                .unwrap_or(f32::NAN)
        };
        println!("{:<52} {:>10} {:>10} {:>10}   driver seat yaw rate bounds / max speed ref / exp", "vehicle", "blur", "obj vel max", "authored");
        for v in native.tags_by_group(cache, "vehi") {
            let Some(u) = UnitSeats::read(native, cache, v) else { continue };
            let p = crate::reach_vehicle::VehiclePhysics::read(native, cache, v);
            let authored = p.as_ref().and_then(|p| p.family.velocity_model().map(|m| m.max_forward_speed)).unwrap_or(f32::NAN);
            let d = u.seats.iter().find(|s| s.is_driver()).or_else(|| u.seats.first());
            let name = native.tag_name(cache, v).unwrap_or_default();
            let short: String = name.rsplit('\\').next().unwrap_or(&name).to_string();
            println!(
                "{short:<52} {:>10.3} {:>10.3} {:>10.3}   {:?} / {:.2} / {:.2}  family {}",
                rdf(v, 0x600),
                rdf(v, 0x33C),
                authored,
                d.map(|s| s.yaw_rate_bounds).unwrap_or([f32::NAN; 2]),
                d.map(|s| s.max_speed_reference).unwrap_or(f32::NAN),
                d.map(|s| s.speed_exponent).unwrap_or(f32::NAN),
                p.as_ref().map(|p| p.family.kind()).unwrap_or("?")
            );
        }
    }

    /// Print-only: what an entry marker's frame and authored `direction` actually are, so the
    /// axis question is answered from the data rather than from Blam folklore.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_entry_marker_frames() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        for v in native.tags_by_group(cache, "vehi") {
            let name = native.tag_name(cache, v).unwrap_or_default();
            if !(name.ends_with("warthog") || name.ends_with("mongoose") || name.ends_with("ghost")) {
                continue;
            }
            let Some(u) = UnitSeats::read(native, cache, v) else { continue };
            let mode = native.resolve_mode(cache, v);
            let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) else { continue };
            let pose = crate::reach_actor::Pose::bind(&m.skel);
            println!("\n{name}");
            for s in &u.seats {
                for (label, group) in [("seat", s.marker_name.as_str()), ("entry", s.entry_marker_group())] {
                    if group.is_empty() { continue }
                    let Some(g) = m.marker_group(native, cache, group) else {
                        println!("  [{}] {label} group '{group}' NOT IN THE MODEL", s.index);
                        continue;
                    };
                    for mk in &g.markers {
                        let f = pose.marker_model(mk);
                        let p = f.w_axis.truncate();
                        println!(
                            "  [{}] {label} '{group}' node {} pos ({:.3},{:.3},{:.3}) quat ({:.3},{:.3},{:.3},{:.3}) dir ({:.3},{:.3},{:.3}) frameX ({:.3},{:.3},{:.3})",
                            s.index, mk.node, p.x, p.y, p.z,
                            mk.rotation.x, mk.rotation.y, mk.rotation.z, mk.rotation.w,
                            mk.direction.x, mk.direction.y, mk.direction.z,
                            f.x_axis.x, f.x_axis.y, f.x_axis.z
                        );
                    }
                }
            }
        }
    }

    /// **THE MARKER-AXIS GATE.** Which axis of a marker frame is "forward" decides the entry
    /// cone, and getting it wrong puts the prompt on the wrong side of every vehicle.
    ///
    /// # The reference that works, and the two that do not
    ///
    /// The obvious reference is the marker's POSITION — "forward points away from the vehicle" —
    /// and it fails twice, which is worth recording because both failures look like answers:
    ///
    /// * against the **model origin in 3D**, `+Z` wins at 92%. An artefact: a vehicle's pivot
    ///   sits near the bottom centre, so nearly every marker is above it and any upright frame's
    ///   `+Z` scores.
    /// * against the **horizontal** offset from the pivot, or from the seat marker, NO axis beats
    ///   58%. Also real: entry markers sit close to the vehicle's own centre — the Warthog's
    ///   `driver_enter` is 0.26 wu off the pivot — so their positions carry almost no direction.
    ///
    /// The reference that does work is the seat's own **side**, because Blam's object convention
    /// puts `+Y` on the LEFT. A seat on the left must be entered from the left, so the entry
    /// marker's forward must have a positive `Y`; a seat on the right, negative. The side is
    /// known two independent ways and this gate takes a marker only when at least one gives it:
    ///
    /// * the marker NAME says it — `ghost_b_d_l_enter` versus `ghost_b_d_r_enter`,
    ///   `scorpion_b_d_l` / `_r`, `passenger_lf` / `rf`, `falcon_side_gun_left` / `right`;
    /// * the SEAT marker's own `Y` says it — the Warthog's `driver` sits at `y = +0.166` and its
    ///   `passenger` at `y = -0.159`.
    ///
    /// And it discriminates by construction: a constant axis like `+Y` matches every left seat
    /// and no right seat, so it scores ~50% however many seats there are, while `±Z` has no `Y`
    /// component at all and scores noise. Only an axis that actually TURNS with the seat can beat
    /// them, and `-X` is `+X`'s mirror, so the two cannot both win.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_entry_markers_point_away_from_the_vehicle() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_seats gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let axes: [(&str, glam::Vec3); 6] = [
            ("+X", glam::Vec3::X),
            ("-X", -glam::Vec3::X),
            ("+Y", glam::Vec3::Y),
            ("-Y", -glam::Vec3::Y),
            ("+Z", glam::Vec3::Z),
            ("-Z", -glam::Vec3::Z),
        ];
        /// +1 = left (+Y in Blam's object convention), -1 = right, 0 = the name says neither.
        fn side_from_name(name: &str) -> i32 {
            let n = name.to_ascii_lowercase();
            let has = |t: &str| {
                n.split(|c: char| !c.is_ascii_alphanumeric()).any(|p| p == t)
                    || n.split('_').any(|p| p.len() <= 3 && p.starts_with(t) && p[1..].chars().all(|c| c.is_ascii_digit()))
            };
            let left = has("l") || has("left") || has("lf") || has("lb");
            let right = has("r") || has("right") || has("rf") || has("rb");
            match (left, right) {
                (true, false) => 1,
                (false, true) => -1,
                _ => 0,
            }
        }
        let mut matched = [0usize; 6];
        let mut total = 0usize;
        let mut sideways = 0usize;
        let mut sideways_matched = 0usize;
        let mut by_name = 0usize;
        let mut by_seat = 0usize;
        for v in native.tags_by_group(cache, "vehi") {
            let Some(u) = UnitSeats::read(native, cache, v) else { continue };
            let mode = native.resolve_mode(cache, v);
            let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) else { continue };
            let pose = crate::reach_actor::Pose::bind(&m.skel);
            for s in &u.seats {
                let group = s.entry_marker_group();
                if group.is_empty() {
                    continue;
                }
                let Some(entry) = m.marker_group(native, cache, group) else { continue };
                let name_side = side_from_name(group);
                let seat_side = m
                    .marker_group(native, cache, &s.marker_name)
                    .and_then(|g| g.markers.first().copied())
                    .map(|mk| pose.marker_model(&mk).w_axis.y)
                    .filter(|y| y.abs() > 0.10)
                    .map_or(0, |y| if y > 0.0 { 1 } else { -1 });
                // Both references must agree when both speak; otherwise take whichever does.
                let side = match (name_side, seat_side) {
                    (0, 0) => continue,
                    (a, 0) => {
                        by_name += 1;
                        a
                    }
                    (0, b) => {
                        by_seat += 1;
                        b
                    }
                    (a, b) if a == b => {
                        by_name += 1;
                        a
                    }
                    _ => continue,
                };
                for mk in &entry.markers {
                    let f = pose.marker_model(mk);
                    total += 1;
                    // Many entry markers legitimately point fore or aft -- a Ghost and a
                    // Mongoose are mounted from BEHIND, and their `+X` is the model's `-X` with
                    // no Y at all. Those carry no side information, so the conditional rate
                    // (among the markers that DO point sideways) is counted separately and is
                    // the number that means something.
                    let fx_y = f.transform_vector3(glam::Vec3::X).y;
                    if fx_y.abs() > 0.1 {
                        sideways += 1;
                        if (fx_y > 0.0) == (side > 0) {
                            sideways_matched += 1;
                        }
                    }
                    for (i, (_, a)) in axes.iter().enumerate() {
                        let y = f.transform_vector3(*a).y;
                        // An axis with no Y to speak of says nothing and must not score.
                        if y.abs() > 0.1 && (y > 0.0) == (side > 0) {
                            matched[i] += 1;
                        }
                    }
                }
            }
        }
        assert!(
            total >= 40,
            "only {total} entry markers have a known side ({by_name} from the name, {by_seat} from \
             the seat marker) -- not enough to decide an axis"
        );
        let mut scored: Vec<(usize, &str)> = axes.iter().enumerate().map(|(i, (n, _))| (matched[i], *n)).collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        println!(
            "{total} entry markers with a known side ({by_name} named, {by_seat} from the seat \
             marker's own Y); how often each axis of the entry marker's frame points to THAT side:"
        );
        for (n, name) in &scored {
            println!("   {name}  {n}/{total}  ({:.0}%)", 100.0 * *n as f32 / total as f32);
        }
        println!(
            "   of the {sideways} markers whose +X has a Y component at all, {sideways_matched} \
             point to their own seat's side ({:.0}%)",
            100.0 * sideways_matched as f32 / sideways.max(1) as f32
        );
        let (best_n, best_name) = scored[0];
        let (second_n, second_name) = scored[1];
        let (worst_n, worst_name) = *scored.last().unwrap();
        // THE DECISIVE PAIR. If the convention were `-X`, the mirror axis would score what `+X`
        // scores and `+X` would score nothing -- so the two together cannot both be near zero or
        // both be high, and this is the assertion a wrong convention cannot survive.
        assert!(
            worst_name == "-X" && worst_n * 20 <= total,
            "the mirror axis -X scores {worst_n}/{total} ({worst_name} is last) -- if forward were \
             -X it would score what +X does, so this is the test that separates them"
        );
        assert!(
            best_n * 2 >= total,
            "the best axis {best_name} points to the seat's own side on only {best_n}/{total} \
             markers -- no axis is consistently the marker's forward"
        );
        assert!(
            best_n > second_n + total / 5,
            "{best_name} ({best_n}) barely beats {second_name} ({second_n}) of {total} -- this \
             gate does not discriminate between two axes"
        );
        assert!(
            sideways >= 20 && sideways_matched * 10 >= sideways * 9,
            "of {sideways} entry markers that point sideways at all, only {sideways_matched} point \
             to their own seat's side"
        );
        // And the measured winner must be what [`MARKER_FORWARD`] actually says, read from the
        // constant rather than from a literal -- so changing the constant turns this red.
        let declared = axes
            .iter()
            .find(|(_, a)| (*a - MARKER_FORWARD).length() < 1e-6)
            .map(|(n, _)| *n)
            .unwrap_or("(not an axis)");
        assert_eq!(best_name, declared, "the measured forward axis is {best_name}, but MARKER_FORWARD is {declared}");
    }

    /// The entry test must refuse for each authored reason, one at a time. **Data sabotage**: the
    /// seat is constructed, so perturbing one authored limit proves the behaviour follows that
    /// field and not the others, with no recompile of the thing under test.
    #[test]
    fn the_entry_test_refuses_for_each_authored_reason_separately() {
        use glam::{Mat4, Vec3};
        // A seat whose limits are the Warthog driver's own, read from the tag.
        let seat = Seat {
            index: 0,
            entry_radius: 1.25,
            entry_cone_angle: 1.3090,
            entry_facing_angle: 1.0472,
            max_relative_velocity: 3.5,
            ..Default::default()
        };
        // The marker at the origin facing +X; the player one world unit along +X, looking back.
        let marker = Mat4::IDENTITY;
        let at = |p: Vec3, f: Vec3, v: Vec3| entry_test(&seat, marker, p, f, v);

        let ok = at(Vec3::new(1.0, 0.0, 0.0), -Vec3::X, Vec3::ZERO);
        assert!(ok.passes(), "standing in front of the marker looking at it must pass: {}", ok.describe());
        assert_eq!(ok.refusal(), None);

        // 1. too far -- 2 wu against an authored 1.25
        let far = at(Vec3::new(2.0, 0.0, 0.0), -Vec3::X, Vec3::ZERO);
        assert!(!far.passes());
        assert_eq!(far.refusal(), Some("too far from the entry marker"));

        // 2. inside the radius but outside the marker's cone: directly BEHIND it
        let behind = at(Vec3::new(-1.0, 0.0, 0.0), Vec3::X, Vec3::ZERO);
        assert!(!behind.passes());
        assert_eq!(behind.refusal(), Some("not inside the entry marker's cone"));

        // 3. in the cone but looking away
        let away = at(Vec3::new(1.0, 0.0, 0.0), Vec3::X, Vec3::ZERO);
        assert!(!away.passes());
        assert_eq!(away.refusal(), Some("not looking at the entry marker"));

        // 4. everything right but moving relative to the vehicle
        let fast = at(Vec3::new(1.0, 0.0, 0.0), -Vec3::X, Vec3::new(5.0, 0.0, 0.0));
        assert!(!fast.passes());
        assert_eq!(fast.refusal(), Some("moving too fast relative to the vehicle"));

        // DATA SABOTAGE: the pass above must follow the AUTHORED radius. Shrink it and the same
        // geometry refuses; widen the cone and `behind` still refuses for the cone's own reason.
        let tight = Seat { entry_radius: 0.5, ..seat.clone() };
        assert!(!entry_test(&tight, marker, Vec3::new(1.0, 0.0, 0.0), -Vec3::X, Vec3::ZERO).passes());
        // and widening ONLY the cone makes the same `behind` geometry pass, which is what proves
        // that refusal followed `entry marker cone angle` and not some other limit
        let wide = Seat { entry_cone_angle: std::f32::consts::PI, ..seat.clone() };
        let b2 = entry_test(&wide, marker, Vec3::new(-1.0, 0.0, 0.0), Vec3::X, Vec3::ZERO);
        assert!(b2.passes(), "a pi cone admits the marker's own back side: {}", b2.describe());
        assert_eq!(b2.refusal(), None);
    }

    /// `best_entry_marker` must prefer a marker that PASSES over a nearer one that does not — the
    /// engine accepts any instance of the group, so picking the closest regardless would refuse
    /// entry while standing at a perfectly valid door.
    #[test]
    fn the_best_entry_marker_prefers_one_that_passes_over_a_nearer_one_that_does_not() {
        use glam::{Mat4, Vec3};
        let seat = Seat { entry_radius: 1.5, entry_cone_angle: 0.6, entry_facing_angle: 1.5, max_relative_velocity: 3.0, ..Default::default() };
        // marker 0 is nearer but faces AWAY from the player; marker 1 is further and faces them
        let near_wrong = Mat4::from_rotation_translation(glam::Quat::from_rotation_z(std::f32::consts::PI), Vec3::new(0.0, 0.0, 0.0));
        let far_right = Mat4::from_translation(Vec3::new(0.4, 0.0, 0.0));
        let player = Vec3::new(1.2, 0.0, 0.0);
        let (i, t) = best_entry_marker(&seat, &[near_wrong, far_right], player, -Vec3::X, Vec3::ZERO).unwrap();
        assert_eq!(i, 1, "the passing marker wins: {}", t.describe());
        assert!(t.passes());
        // with only the wrong one, it still reports the closest test so a caption can say why
        let (i, t) = best_entry_marker(&seat, &[near_wrong], player, -Vec3::X, Vec3::ZERO).unwrap();
        assert_eq!(i, 0);
        assert_eq!(t.refusal(), Some("not inside the entry marker's cone"));
        assert!(best_entry_marker(&seat, &[], player, -Vec3::X, Vec3::ZERO).is_none(), "no markers -> no answer");
    }

    /// A seat's look limits must come from the tag, and a seat that authors none must not clamp.
    #[test]
    fn the_seat_look_clamp_follows_the_authored_range_and_zero_means_no_limit() {
        let free = Seat::default();
        assert_eq!(clamp_seat_look(&free, 0.0, 3.0, 1.0), (3.0, 1.0), "an unauthored range must not pin the look");
        let limited = Seat { yaw_range: [-0.5, 0.5], pitch_range: [-0.2, 0.4], ..Default::default() };
        assert_eq!(clamp_seat_look(&limited, 0.0, 3.0, 1.0), (0.5, 0.4));
        assert_eq!(clamp_seat_look(&limited, 0.0, -3.0, -1.0), (-0.5, -0.2));
        // relative to the VEHICLE's yaw, not to world zero
        let (y, _) = clamp_seat_look(&limited, 2.0, 2.2, 0.0);
        assert!((y - 2.2).abs() < 1e-6, "inside the range it is untouched, got {y}");
        let (y, _) = clamp_seat_look(&limited, 2.0, 4.0, 0.0);
        assert!((y - 2.5).abs() < 1e-6, "clamped to vehicle yaw + 0.5, got {y}");
    }

    /// The seat camera must take the authored `override fov` when there is one and the default
    /// when there is not, and must report the authored third-person flag rather than deciding.
    #[test]
    fn the_seat_view_uses_the_authored_fov_and_flag() {
        use glam::{Mat4, Vec3};
        let seat_marker = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let cam_marker = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        let plain = Seat::default();
        let v = seat_view(&plain, None, seat_marker, 0.0, 0.0, 1.3);
        assert_eq!(v.eye, Vec3::new(1.0, 2.0, 3.0), "no camera marker -> the seat's own");
        assert_eq!(v.fov, 1.3, "no override -> the default");
        assert!(!v.third_person);
        let authored = Seat { camera_override_fov: 1.1, flags: FLAG_THIRD_PERSON_CAMERA, ..Default::default() };
        let v = seat_view(&authored, Some(cam_marker), seat_marker, 0.0, 0.0, 1.3);
        assert_eq!(v.eye, Vec3::new(4.0, 5.0, 6.0), "the authored camera marker wins");
        assert_eq!(v.fov, 1.1);
        assert!(v.third_person);
        // forward follows yaw/pitch
        let v = seat_view(&plain, None, seat_marker, std::f32::consts::FRAC_PI_2, 0.0, 1.3);
        assert!((v.forward - Vec3::Y).length() < 1e-5, "yaw pi/2 looks along +Y, got {:?}", v.forward);
    }

    /// `best` must prefer the driver's seat, and must refuse a seat the tag marks invalid for a
    /// player. Pure logic over constructed seats: no map needed, so this one always runs.
    #[test]
    fn best_prefers_the_driver_and_refuses_an_invalid_seat() {
        let mk = |index: usize, flags: u32| Seat { index, flags, label: format!("s{index}"), ..Default::default() };
        let u = UnitSeats {
            unit_tag: 0,
            seats: vec![mk(0, 0), mk(1, FLAG_DRIVER), mk(2, FLAG_GUNNER), mk(3, FLAG_DRIVER | FLAG_INVALID_FOR_PLAYER)],
        };
        assert_eq!(u.best(&[0, 1, 2]), Some(1), "the driver's seat wins");
        assert_eq!(u.best(&[0, 2]), Some(2), "a gunner beats a passenger");
        assert_eq!(u.best(&[0]), Some(0));
        assert_eq!(u.best(&[3]), None, "a seat invalid for a player is refused outright");
        assert_eq!(u.best(&[]), None);
    }

    /// #seat-camera  **WHAT DOES EACH PLAYER SEAT ACTUALLY AUTHOR FOR THE LOOK, AND WHAT DOES THE
    /// CLAMP DO WITH IT?**
    ///
    /// Reported: *"the camera is WRONG for the warthog, it won't let me move it really AT ALL!"*
    /// Print-only, and deliberately so: the numbers go on the record before anything is changed,
    /// because the candidate explanations (radians vs degrees, a half-range, `0` meaning "no
    /// limit" rather than "pinned") are told apart by the values and not by argument.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_what_each_seat_authors_for_the_look_and_what_the_clamp_does() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut pinned = 0usize;
        let mut free = 0usize;
        let mut slivered = Vec::new();
        for t in native.tags_by_group(cache, "vehi") {
            let Some(s) = UnitSeats::read(native, cache, t) else { continue };
            let path = native.tag_name(cache, t).unwrap_or_default();
            let short = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
            for seat in &s.seats {
                if !seat.player_allowed() {
                    continue;
                }
                // The clamp's own answer over a FULL sweep: how much of a turn does the rider get?
                let swept = |f: &dyn Fn(f32) -> f32| -> (f32, f32) {
                    let mut lo = f32::INFINITY;
                    let mut hi = f32::NEG_INFINITY;
                    for k in -180..=180 {
                        let v = f(k as f32 * std::f32::consts::PI / 180.0);
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    (lo, hi)
                };
                let (ylo, yhi) = swept(&|w| clamp_seat_look(seat, 0.0, w, 0.0).0);
                let (plo, phi) = swept(&|w| clamp_seat_look(seat, 0.0, 0.0, w).1);
                let yaw_span = (yhi - ylo).to_degrees();
                let pitch_span = (phi - plo).to_degrees();
                println!(
                    "{short:26} seat {:2} '{:14}' D{} G{}  yaw_range [{:+.4},{:+.4}] pitch_range [{:+.4},{:+.4}] \
                     -> the rider gets {yaw_span:7.1} deg of yaw, {pitch_span:6.1} deg of pitch",
                    seat.index,
                    seat.label,
                    if seat.is_driver() { "1" } else { "0" },
                    if seat.is_gunner() { "1" } else { "0" },
                    seat.yaw_range[0],
                    seat.yaw_range[1],
                    seat.pitch_range[0],
                    seat.pitch_range[1]
                );
                if yaw_span < 1.0 && pitch_span < 1.0 {
                    pinned += 1;
                    slivered.push(format!("{short} seat {}", seat.index));
                } else if yaw_span > 359.0 {
                    free += 1;
                }
            }
        }
        println!("\nplayer seats PINNED to under a degree in both axes: {pinned} {slivered:?}");
        println!("player seats with a FULL 360 of yaw: {free}");
    }

    /// #vehicle-scale  **MEASURE the drawn biped against the drawn vehicle.** Print-only.
    ///
    /// Reported: *"the scale of the biped compared to the vehicle is totally WRONG"*, and
    /// separately *"the biped we are controlling is far too small"*. Those may be one bug or two,
    /// and the question is decided by the model's own `compression info` bounds -- the same
    /// measurement that settled the Elite-torso question -- not by looking at it.
    ///
    /// The yardstick: 1 world unit is 3.048 m, a Spartan is about 2.1 m, so a correct Spartan is
    /// **~0.69 wu** tall. A Warthog is about 5.3 m long, so **~1.74 wu**.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_drawn_scale_of_every_biped_and_vehicle() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        const WU_M: f32 = 3.048;
        for group in ["bipd", "vehi"] {
            for t in native.tags_by_group(cache, group) {
                let path = native.tag_name(cache, t).unwrap_or_default();
                let short = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
                let mode = native.resolve_mode(cache, t);
                let Some((lo, hi)) = crate::reach_actor::model_bounds(native, cache, mode) else {
                    println!("{group} {short:26} NO compression info");
                    continue;
                };
                let d = hi - lo;
                println!(
                    "{group} {short:26} x {:6.3} y {:6.3} z {:6.3} wu  = {:5.2} x {:5.2} x {:5.2} m   (z from {:+.4} to {:+.4})",
                    d.x, d.y, d.z, d.x * WU_M, d.y * WU_M, d.z * WU_M, lo.z, hi.z
                );
            }
        }
    }
}
