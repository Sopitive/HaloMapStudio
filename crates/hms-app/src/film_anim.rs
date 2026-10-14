//! Reach Theater: what a recorded body is DOING — locomotion, vehicles and combat actions
//! (#film-anim).
//!
//! `film_decode` answers *where* an entity is; this module answers *what state it is in*. The two
//! are separated because the honesty bar differs. A position is **proven** (474/474 static Forge
//! objects land on a real object), and a biped's body yaw is proven against the aim oracle at
//! 5.7-7.3 degrees. A locomotion state is not in evidence anywhere, so everything here is either
//! DERIVED from those proven quantities or openly SYNTHESISED, and every state carries a
//! [`Provenance`] that the UI prints.
//!
//! What that buys, concretely. The build before this one picked a walk cycle from the recorded
//! ground SPEED alone, which can only ever play `move_front`: a player walking backwards ran
//! backwards while facing forward, a player strafing slid sideways in a forward run, and a player
//! falling off a tower jogged through the air. The body yaw IS decoded, so the direction of
//! travel *in the body's own frame* is available, and that one quantity picks between
//! `move_front`, `move_back`, `move_left` and `move_right` — the animations the graph has had all
//! along. Sprinting and airborne follow from the same two numbers.
//!
//! What is still not decoded, and is therefore not drawn: crouch, the armour ability a player is
//! holding, and firing. See `probes.rs` for the sweeps that failed to find them and the numbers
//! that say so — a clean negative is recorded here rather than papered over.

/// Where a piece of per-entity state came from. The UI prints this, so a viewer is never told
/// that a synthesised walk cycle is the film's own animation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// Read out of the film's stream and validated against an oracle.
    Decoded,
    /// Computed from decoded quantities — the recorded position and the recorded body yaw. Honest
    /// and reproducible, but it is this reader's inference, not the recording's own statement.
    Derived,
    /// Invented. Nothing in the film supports it.
    #[default]
    Synthesised,
}

impl Provenance {
    pub fn label(self) -> &'static str {
        match self {
            Provenance::Decoded => "decoded",
            Provenance::Derived => "derived",
            Provenance::Synthesised => "synthesised",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the locomotion state
// ---------------------------------------------------------------------------------------------

/// The jmad MODE a body is in — the first colon-separated field of a Reach animation name.
///
/// These are the modes the shipped Spartan graph actually has (1393 animations): `combat:`,
/// `crouch:`, `flight:` (airborne), `sprint:`, and the per-vehicle seat modes `warthog_d:`,
/// `mongoose_p:`, `ghost_d:` and so on. `Crouch` is listed because the graph has it, but nothing
/// here ever produces it: the crouch bit is not decoded, and guessing it from a low speed would
/// make every player who stops to aim drop to one knee.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Stance {
    /// Standing, on the ground, weapon up.
    #[default]
    Combat,
    /// Not drawn — and now known to be **undecodable from a Reach film**, not merely undecoded.
    ///
    /// #jump  There are exactly two places a crouch could be stated, and both are measured empty:
    /// the unit's derived dirty block carries the crouch button in field **0x37** (the 0x90-byte
    /// control-input record) and the jump/land transitions in field **0x36** (the action enum
    /// whose own names in the binary are `_action_jump`, `_action_land`, …) — and bits 0x36 and
    /// 0x37 **never appear, not once in 5,438,363 type-11 messages over 8 films** (the census in
    /// the [`unit`] module docs). The 72-type EVENT channel has no crouch or jump event either
    /// (see `film_action::EVENT_NAMES`: every biped verb it carries is a melee, a throw, a
    /// pickup, a board or an assassination).
    ///
    /// So a crouching player is drawn standing, and that is a property of the FILE. The variant
    /// is kept, and [`anim_candidates`] asks the `crouch:` mode for the leaf names that mode
    /// actually authors, so the day a crouch signal is found the clips are already right —
    /// but nothing here produces it. Guessing it from a low speed would put every player who
    /// stops to aim on one knee.
    Crouch,
    /// Off the ground: a jump, a fall, a man-cannon launch or a jetpack.
    Flight,
    /// Moving faster than the run cap in a straight line — Reach's sprint armour ability.
    Sprint,
    /// Riding a vehicle. The seat decides the mode entirely.
    Seated,
    /// #film-action  Locked into a two-party SYNCHRONISED ACTION — an assassination. Unlike
    /// every other stance here this one is raised from a **decoded** event, not inferred: the
    /// stream's `incident` names the kill, both parties and the tick. See [`Assassination`].
    SyncAction,
    /// #death This body's life ENDED. The stream never says a body died, let alone how — so this
    /// stance is raised by [`death_tick`] from the shape of the track (its last reported position,
    /// on a track the recording later destroyed) and the fall itself is SYNTHESISED from the
    /// biped's own `landing_dead` clips. See [`death_candidates`].
    Dead,
}

/// Which way a body is moving **in its own frame**. This is the whole point of the module: the
/// recorded yaw and the recorded positions together say whether a player is advancing, backing
/// off, or strafing, and the graph has a separate cycle for each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Gait {
    #[default]
    Idle,
    /// Standing still but rotating — the graph's `turn_left` / `turn_right`.
    TurnLeft,
    TurnRight,
    WalkFront,
    RunFront,
    BackSlow,
    Back,
    Left,
    Right,
}

/// Thresholds. Every one of these is measured off the corpus rather than guessed, and the
/// measurement is in `probes::probe_speed_distribution_and_ability_oracles`.
pub mod cuts {
    /// Below this a body is standing still, wu/s. (The pooled distribution's lowest bin is
    /// 0.00-0.50 with 283k of 2.70M samples; a replicated position jitters by a fraction of a
    /// quantisation step even when the player is motionless.)
    pub const STILL: f32 = 0.35;
    // #gait-authored  THE THREE SPEED CUTS BELOW COME FROM THE ENGINE'S OWN TAGS, and they
    // replace values fitted to the film corpus. The corpus measurement was real; its
    // INTERPRETATION was wrong, and that is the whole correction.
    //
    // The old cuts read the pooled histogram's dominant 3.50-4.00 mode (307,984 samples, falling
    // off a cliff to 77,216 above 4.00) as "the ordinary run". It is not. `matg`'s
    // `@player information` row authors, per species:
    //
    //              walk    run fwd   sprint x    sprint top
    //   Spartan    0.512    2.200     x1.623       3.571
    //   Elite      0.650    2.600     x1.400       3.640
    //
    // So that mode is **sprinting players**, which in Reach is nearly everyone, and the cliff
    // above 4.0 is simply nothing on foot going faster than 3.64. The old numbers therefore put
    // the walk/run split (2.4) ABOVE a real full-speed run and the sprint floor (4.6) ABOVE the
    // engine's own sprint speed -- so a genuine run was drawn as a walk, and **a sprint could
    // never be drawn at all**.
    //
    // What makes the authored row trustworthy rather than just another guess: the sprint
    // multiplier is a separate authored field, and `2.200 x 1.623 = 3.571` lands inside the
    // corpus histogram's peak bin. A value read at a recovered offset predicting a number
    // measured independently from 2.7 million film samples is not a coincidence.
    //
    // These are single values rather than per-species because **the two authored rows do not
    // overlap** -- every cut below sits clear of both species' numbers. If a third species ever
    // appears, or a gametype trait scales speed, they have to become per-body.
    /// Walk / run split, wu/s. Above the Elite's authored walk (0.650) and below the Spartan's
    /// authored run (2.200).
    pub const WALK: f32 = 1.4;
    /// The ordinary RUN CAP, wu/s — the faster of the two authored `run forward` speeds.
    pub const RUN_CAP: f32 = 2.6;
    /// Sprint floor, wu/s. Above the Elite's authored run (2.600) and below the Spartan's
    /// authored sprint top (3.571), so both species' sprints are claimed and neither's ordinary
    /// run is.
    pub const SPRINT: f32 = 3.1;
    /// …and only BELOW this, wu/s. Sprint tops out at 3.640 (Elite), so this leaves headroom for
    /// a downhill run while still excluding what the corpus shows above it: the pooled histogram
    /// is flat and thin from 4.5 to 12 wu/s (about 1% of samples per half-unit bin) with a spike
    /// in the top bin — the signature of a man cannon, an explosion or a carry, not a second
    /// locomotion mode. Putting a sprint cycle on a launched body was the defect this bounds.
    pub const SPRINT_MAX: f32 = 5.0;
    /// Vertical speed above which a body is treated as off the ground, wu/s.
    pub const AIRBORNE_VZ: f32 = 1.6;
    /// #jump  How long after touching down the LANDING clip still owns the body, in film ticks.
    ///
    /// `combat:<class>:land_hard` is 13 frames, and a clip's frame is driven at half the tick
    /// rate (60 Hz ticks -> 30 fps frames), so the clip is finished after 26 ticks and the body
    /// must be back on its locomotion cycle. Nothing is guessed by this number: it is the shipped
    /// clip's own length. (The `crouch:` set authors a 50-frame landing, which would want 100 --
    /// it is unreachable while the crouch state cannot be decoded, and the shorter window is the
    /// safe one to share.)
    pub const LAND_TICKS: u32 = 26;
    /// How far off straight ahead still counts as "advancing", radians.
    pub const FRONT_ARC: f32 = 0.87; // 50 degrees
    /// How far round counts as "backing off", radians.
    pub const BACK_ARC: f32 = 2.27; // 130 degrees
    /// Yaw rate above which a standing body is turning, radians/s.
    pub const TURN_RATE: f32 = 1.2;
}

/// A body riding a vehicle: which vehicle, and which seat family the animation comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct Seat {
    /// The vehicle entity the body is riding.
    pub vehicle_net_id: u32,
    /// The jmad seat-mode prefix, e.g. `warthog_d` or `mongoose_p`.
    pub mode: String,
    /// The vehicle's tag leaf name, for the UI (`warthog`, `mongoose`, ...).
    pub vehicle: String,
}

/// #film-action  A body in a two-party action: which side it is playing and how far in.
#[derive(Clone, Debug, PartialEq)]
pub struct SyncState {
    pub side: SyncSide,
    /// Ticks since the action started. Drives the clip's frame, so the whole performance is a
    /// pure function of the playback tick and scrubbing backwards rewinds it.
    pub age: u32,
    /// Which of the species pair's moves is playing.
    pub move_index: usize,
    pub attacker_elite: bool,
    pub victim_elite: bool,
    /// True when the specific move came off the wire (`biped_assassinate`, host-recorded films
    /// only) rather than being chosen from the two entity ids.
    pub move_decoded: bool,
    /// The incident that named this kill, or how it was inferred.
    pub incident: &'static str,
    /// #film-action  True when the RECORDING STATES this was an assassination. False when HMS
    /// worked it out from the damage and melee channels -- see
    /// [`inferred_assassinations`] -- in which case nothing here may claim it was decoded.
    pub stated: bool,
    /// The other body in the action, so the victim can be drawn on the attacker's transform.
    pub partner: Option<u32>,
}

/// Everything this module knows about one body at one tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyState {
    pub stance: Stance,
    pub gait: Gait,
    /// Ground speed, wu/s (the recorded positions' own derivative).
    pub speed: f32,
    /// Horizontal speed only — what the gait is chosen from, so a fall does not read as a sprint.
    pub ground_speed: f32,
    /// Vertical speed, wu/s. Positive is up.
    pub vz: f32,
    /// Direction of travel in the body's own frame, radians. 0 = straight ahead, +pi/2 = strafing
    /// left. `None` when the body is not moving or has no decoded facing.
    pub travel_angle: Option<f32>,
    /// **Not filled in.** It used to hold the distance travelled along the whole track, which
    /// meant every `derive` call walked every position of that track — and `cycle_phase` calls
    /// `derive` once per segment, making the per-body cost quadratic in its own length. On a
    /// 2,243-position body that is five million operations per body per frame, which is what made
    /// Theater playback cost ~31 ms of CPU per Spartan. The phase comes from
    /// [`cycle_phase`] now, which integrates in one pass.
    pub distance: f32,
    pub seat: Option<Seat>,
    /// #death Ticks since this body died, when [`Stance::Dead`]. Drives the death clip's frame,
    /// so the fall is a pure function of the playback tick and scrubbing backwards undoes it.
    pub dead_ticks: u32,
    /// #death Which of the biped's death clips this body falls with, chosen from its entity id so
    /// the same body always dies the same way however many times the clip is re-watched.
    pub death_variant: u8,
    /// #death This body was OFF THE GROUND when it died, so it falls with the `airborne_dead`
    /// flail rather than the ground landing. Derived from the decoded vertical speed of its own
    /// last reported motion — see [`death_candidates`].
    pub death_airborne: bool,
    /// #death This body's graph is the ELITE's, whose death clips are named differently from the
    /// Spartan's. Comes from the biped tag the recording names for this body.
    pub death_elite: bool,
    /// #death Who was standing next to this body when it died, if anyone — see [`death_company`].
    /// Filled only on the caption path, never on the per-frame draw path.
    pub death_company: Option<DeathCompany>,
    /// #film-action  Set when this body is inside a decoded assassination.
    pub sync: Option<SyncState>,
    /// #melee  Ticks since this body's last `biped_melee_initiate`, when that is still inside
    /// the swing. The event is DECODED — 9,567 of them over the corpus, ref 0 naming the biped
    /// that swung — so unlike the death clip this is a stated fact, not an inference.
    pub melee_ticks: Option<u32>,
    /// #melee  Which of the class's two or three authored swings to play. Not recorded, so it is
    /// derived from the event's own tick and therefore replays identically.
    pub melee_variant: u8,
    /// #jump  Ticks since this body TOUCHED DOWN, while the landing clip still owns it.
    ///
    /// `None` for a body that is airborne, or that has been on the ground longer than the clip
    /// lasts. DERIVED, like the airborne state it follows: a Reach film has no land event (field
    /// 0x36, which carries `_action_land`, never appears) so this is read off the recorded
    /// vertical speed by [`land_age`]. Filled by the CALLER rather than by [`derive`], because
    /// `cycle_phase` calls `derive` once per segment and a lookback inside it would put the cost
    /// back on the path that was rescued from being quadratic.
    pub land_ticks: Option<u32>,
    /// #jump-loop  Ticks this body has been OFF THE GROUND, while it is. `None` when grounded.
    ///
    /// Drives the airborne clip as a HELD one-shot instead of a wrapping cycle — see
    /// [`airborne_age`]. Filled by the caller, for the same reason as `land_ticks`.
    pub air_ticks: Option<u32>,
    /// #wclass The ANIMATION CLASS of the weapon this body is carrying (`rifle`, `pistol`,
    /// `missile`, `sword`, `hammer`, `ball`, `turret`, ...) — the middle component of the
    /// `mode:class:action` clip name, read off the weapon tag's own `weapon class` field.
    ///
    /// `None` means the recording never named a weapon for this body, so the clip comes from the
    /// generic [`DEFAULT_WEAPON_CLASS`] set and the caption says so. That is the common case: a
    /// film reports a unit's weapon only when it CHANGES hands.
    pub weapon_class: Option<String>,
    /// The weakest provenance contributing to this state — what the UI must print.
    pub provenance: Provenance,
    /// Why the provenance is what it is, in one clause, for the per-player status line.
    pub because: &'static str,
}

/// #aim  [`ang_delta`], public, for the driver's aim-offset read.
pub fn ang_delta_pub(a: f32, b: f32) -> f32 {
    ang_delta(a, b)
}

/// Signed smallest difference between two angles, radians.
fn ang_delta(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// Velocity around `tick` from the two bracketing recorded positions, wu/s, and the yaw rate.
///
/// The film replicates a position only 8-86% of ticks (the client is expected to integrate
/// velocity), so this deliberately uses the bracketing KEYS rather than a fixed window: that is
/// the only honest derivative of a sparse track.
pub fn velocity_at(t: &crate::film_decode::EntityTrack, tick: u32) -> Option<[f32; 3]> {
    // Smoothed over a short WINDOW rather than taken from the single bracketing pair of keys.
    // The gait is re-derived every tick from this value, so a noisy derivative makes the choice
    // flap between two clips near a threshold — the 4.00 wu/s run cap, or a direction sitting on
    // the boundary between "forward" and "strafe" — and every flip is a hard pop because the clip
    // is swapped instantly. Smoothing the INPUT fixes that at the source and, unlike a stateful
    // hysteresis, keeps the pose a pure function of the tick, so scrubbing to the same moment
    // always gives the same pose.
    const W: u32 = 6; // 0.1 s either side
    if let (Some(a), Some(b)) = (t.sample(tick.saturating_sub(W)), t.sample(tick + W)) {
        let span = (tick + W - tick.saturating_sub(W)) as f32 / 60.0;
        if span > 0.0 {
            let v = [(b[0] - a[0]) / span, (b[1] - a[1]) / span, (b[2] - a[2]) / span];
            if v.iter().all(|c| c.is_finite()) && v[0].hypot(v[1]) <= 40.0 {
                return Some(v);
            }
        }
    }
    let i = t.positions.partition_point(|(t, _)| *t <= tick);
    if i == 0 || i >= t.positions.len() {
        return None;
    }
    let (t0, p0) = t.positions[i - 1];
    let (t1, p1) = t.positions[i];
    if t1 <= t0 {
        return None;
    }
    let dt = (t1 - t0) as f32 / 60.0;
    let v = [(p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt, (p1[2] - p0[2]) / dt];
    // A respawn is a teleport, not a 200 wu/s dash.
    if v.iter().any(|c| !c.is_finite()) || v[0].hypot(v[1]) > 40.0 {
        return None;
    }
    Some(v)
}

/// Yaw rate around `tick`, radians/s, from the decoded facing samples.
fn yaw_rate_at(t: &crate::film_decode::EntityTrack, tick: u32) -> Option<f32> {
    if t.facings.len() < 2 {
        return None;
    }
    let i = t.facings.partition_point(|(t, _)| *t <= tick);
    if i == 0 || i >= t.facings.len() {
        return None;
    }
    let (t0, a0) = t.facings[i - 1];
    let (t1, a1) = t.facings[i];
    if t1 <= t0 || t1 - t0 > 20 {
        return None;
    }
    Some(ang_delta(a1, a0) / ((t1 - t0) as f32 / 60.0))
}

/// #jump-loop  How long this body has been OFF THE GROUND, in ticks. `None` when it is not.
///
/// The airborne clip is a one-shot, not a cycle. `combat:<class>:airborne_arc` is 45 frames and
/// `:airborne` is 40, while a jump lasts as long as the body is in the air — usually longer — so
/// driving it from the locomotion phase (which is `rem_euclid(1.0)`, i.e. it WRAPS) replayed the
/// arc from the start again and again: "the airborne animation is looping, almost like a player
/// is jumping repeatedly". Fed through the same one-shot lane as the death, the sync action and
/// the landing, the clip advances to its end and HOLDS there until touchdown.
///
/// Capped rather than unbounded: a body that has been falling for ten seconds is long past the
/// clip's last frame, so there is nothing to gain by counting further and the scan stays O(1).
///
/// ## KNOWN, MEASURED, NOT FIXED HERE: the apex gap
///
/// Airborne is `|vz| > AIRBORNE_VZ`, and at the APEX of a jump the vertical speed passes through
/// zero — so for a few ticks in the middle of its own jump a body reads as GROUNDED. It drops out
/// of [`Stance::Flight`] onto a ground gait and then back, which both poses it standing in mid-air
/// and restarts the age here. Caught by this module's own gate, whose fixture had to be changed
/// from a jump arc to a fall to stop it testing this instead of the one-shot rule it is for:
/// `airborne_age` went `Some(11)` at tick 50, `None` across the apex, `Some(1)` again at tick 70.
/// `velocity_at`'s ±6-tick smoothing widens the gap rather than closing it. Fixing it needs a
/// state that survives the apex — height above the ground under the body, or a rise-then-fall
/// pairing — and is deliberately left as its own change rather than bolted on here.
pub fn airborne_age(t: &crate::film_decode::EntityTrack, tick: u32) -> Option<u32> {
    const MAX: u32 = 180; // 3 s; every airborne clip is finished well inside it
    if !airborne_at(t, tick) {
        return None;
    }
    for k in 1..=MAX {
        let Some(back) = tick.checked_sub(k) else { return Some(k.saturating_sub(1)) };
        if velocity_at(t, back).is_none() {
            return Some(k);
        }
        // The first grounded tick behind us is where this flight began.
        if !airborne_at(t, back) {
            return Some(k);
        }
    }
    Some(MAX)
}

/// #jump-apex  Is this body OFF THE GROUND at `tick`, including at the top of its own jump?
///
/// Airborne is a vertical-speed test, and **at the apex of a jump the vertical speed passes
/// through zero** — so the plain `|vz| > AIRBORNE_VZ` reads a body as GROUNDED for a stretch in
/// the middle of its own jump. It drops onto a ground gait and back, which both poses it standing
/// in mid-air and restarts the airborne clip. Measured by this module's own gate before the
/// bridge existed: `Some(11)` at tick 50, `None` across the apex, `Some(1)` again at tick 70.
///
/// The bridge is the one condition that is true at an apex and false everywhere else on the
/// ground: the body was clearly moving vertically **a moment ago AND will be again a moment from
/// now**. Rising then falling is a jump; nothing else does both. It deliberately does not dilate
/// the test in one direction only, which would delay the landing (a body just down has a moving
/// sample behind it) or anticipate the takeoff.
///
/// `APEX_SPAN` comes from the recordings' own gravity rather than being tuned: at the 10.46 wu/s²
/// fitted off these films, vertical speed crosses the ±1.6 wu/s cut in 3.2/10.46 s = 18 ticks,
/// and `velocity_at`'s ±6-tick smoothing widens that — so the samples are taken 30 ticks out,
/// comfortably past the widest apex a Reach jump can have while still inside any real flight.
pub fn airborne_at(t: &crate::film_decode::EntityTrack, tick: u32) -> bool {
    const APEX_WINDOW: u32 = 30;
    let moving = |k: u32| velocity_at(t, k).is_some_and(|v| v[2].abs() > cuts::AIRBORNE_VZ);
    if moving(tick) {
        return true;
    }
    // The NEAREST clearly-moving sample on each side, searched rather than sampled at a fixed
    // offset: a fixed offset has to be wider than the apex and narrower than the flight at once,
    // and a 30-tick probe from early in a jump lands before the takeoff. Searching has neither
    // boundary to straddle.
    let near_back = (1..=APEX_WINDOW).any(|d| tick.checked_sub(d).is_some_and(moving));
    near_back && (1..=APEX_WINDOW).any(|d| moving(tick + d))
}

/// #jump  How long ago this body TOUCHED DOWN, in ticks, while the landing clip still owns it.
///
/// A Reach film states neither the jump nor the landing — the action enum that names
/// `_action_jump` and `_action_land` is unit field 0x36, which never appears in any film on this
/// machine — so a landing is read off the recorded motion, the same way the airborne state above
/// it is: the body must be on the ground NOW, and must have been FALLING clearly (past the same
/// `AIRBORNE_VZ` cut that raises [`Stance::Flight`]) within the clip's own length. The age
/// returned is how many ticks ago the most recent such fall sample was, which drives the clip's
/// frame directly and makes the landing a pure function of the playback tick.
///
/// `None` means "do not play a landing": still in the air, or down longer than the clip lasts.
/// Called ONCE per body per frame by the driver, never from inside [`derive`] — see
/// [`BodyState::land_ticks`].
pub fn land_age(t: &crate::film_decode::EntityTrack, tick: u32) -> Option<u32> {
    // On the ground now, or this is a fall in progress and not a landing.
    let now = velocity_at(t, tick)?;
    if now[2].abs() > cuts::AIRBORNE_VZ {
        return None;
    }
    // The most recent tick inside the clip's window at which the body was clearly falling.
    for k in 1..=cuts::LAND_TICKS {
        let Some(back) = tick.checked_sub(k) else { return None };
        if let Some(v) = velocity_at(t, back) {
            if v[2] < -cuts::AIRBORNE_VZ {
                return Some(k);
            }
        }
    }
    None
}

/// Derive a body's state at `tick` from its recorded track. `seat` is supplied by the caller
/// because it comes from the vehicle catalog, not from the track.
pub fn derive(t: &crate::film_decode::EntityTrack, tick: u32, seat: Option<Seat>) -> BodyState {
    let mut st = BodyState::default();
    if let Some(s) = seat {
        // A seated body's pose is fixed by the seat; nothing is derived and nothing is guessed.
        st.stance = Stance::Seated;
        st.gait = Gait::Idle;
        st.seat = Some(s);
        st.provenance = Provenance::Derived;
        st.because = "seated: the body's position tracks a vehicle's";
        return st;
    }
    let v = velocity_at(t, tick);
    let facing = t.facing_at(tick);
    let (Some(v), Some(facing)) = (v, facing) else {
        // No velocity or no decoded facing: there is nothing to derive. Say so rather than
        // inventing a walk.
        st.speed = t.speed_at(tick).unwrap_or(0.0).clamp(0.0, 8.0);
        st.ground_speed = st.speed;
        st.provenance = Provenance::Synthesised;
        st.because = if facing.is_none() {
            "no decoded body facing at this tick: pose is a standing default"
        } else {
            "no recorded motion at this tick: pose is a standing default"
        };
        st.gait = Gait::Idle;
        return st;
    };
    let ground = v[0].hypot(v[1]);
    st.speed = (ground * ground + v[2] * v[2]).sqrt();
    st.ground_speed = ground;
    st.vz = v[2];
    st.provenance = Provenance::Derived;
    st.because = "from the recorded position and body yaw";

    // Off the ground. Checked FIRST, because a man-cannon launch or a fall off a tower is the
    // single most obvious thing the old speed ladder got wrong — it jogged through the air.
    if v[2].abs() > cuts::AIRBORNE_VZ {
        st.stance = Stance::Flight;
        st.gait = if ground > cuts::WALK { Gait::RunFront } else { Gait::Idle };
        st.because = "airborne: vertical speed above the ground-contact cut";
        return st;
    }

    if ground < cuts::STILL {
        st.gait = match yaw_rate_at(t, tick) {
            Some(r) if r > cuts::TURN_RATE => Gait::TurnLeft,
            Some(r) if r < -cuts::TURN_RATE => Gait::TurnRight,
            _ => Gait::Idle,
        };
        return st;
    }

    // The direction of travel in the BODY's frame. This is the quantity the old build did not
    // have, and the whole reason strafing and backpedalling were drawn as a forward run.
    let theta = ang_delta(v[1].atan2(v[0]), facing);
    st.travel_angle = Some(theta);
    let a = theta.abs();
    if a < cuts::FRONT_ARC {
        // Sprint is claimed only for a straight-line run clear of the measured run cap, and only
        // with the body level: Reach cannot strafe or shoot while sprinting, so a fast sideways
        // move is something else and a fast fall is gravity.
        if ground >= cuts::SPRINT_MAX {
            st.gait = Gait::RunFront;
            st.because = "moving far faster than any Spartan runs: launched or carried, NOT a sprint";
        } else if ground > cuts::SPRINT && a < 0.45 {
            st.stance = Stance::Sprint;
            st.gait = Gait::RunFront;
            st.because = "sprint: straight-line ground speed past the measured run cap";
        } else {
            st.gait = if ground > cuts::WALK { Gait::RunFront } else { Gait::WalkFront };
        }
    } else if a > cuts::BACK_ARC {
        st.gait = if ground > cuts::WALK { Gait::Back } else { Gait::BackSlow };
    } else if theta > 0.0 {
        st.gait = Gait::Left;
    } else {
        st.gait = Gait::Right;
    }
    st
}

// ---------------------------------------------------------------------------------------------
// turning a state into an animation
// ---------------------------------------------------------------------------------------------

/// #wclass The animation set a body falls back to when the recording never named its weapon.
///
/// `rifle` is the broadest set in the shipped Spartan graph and the one a Reach player carries
/// by default. The REAL class is read from the weapon tag (`weap` +0x3AC, see
/// [`crate::reach_actor::weapon_class`]) whenever the film names a weapon; this is what the
/// middle slot holds when it does not.
pub const DEFAULT_WEAPON_CLASS: &str = "rifle";

/// Candidate animation names for a state, best first. The driver tries them in order and takes
/// the first the graph has, which is what makes this robust across vehicles whose seat sets are
/// named inconsistently (`warthog_d:any:idle` but `ghost_d:idle`).
/// The jmad MODE one stance plays in — the first colon-separated field of a clip name.
///
/// One writer: [`anim_candidates`] names clips with it and `play_mode`'s `#gait-blend` resolves
/// the authored gait table with it, and the two must not be able to disagree about which mode a
/// body is in.
pub fn jmad_mode(stance: Stance) -> &'static str {
    match stance {
        Stance::Crouch => "crouch",
        Stance::Sprint => "sprint",
        // #jump  `flight:` is the JETPACK mode, not "off the ground" — see [`anim_candidates`].
        Stance::Combat | Stance::Flight | Stance::Seated | Stance::Dead | Stance::SyncAction => "combat",
    }
}

/// #gait-blend  The `animation_gait_directions` value one gait moves in, or `None` for a gait
/// with no direction (idle).
///
/// **0 front, 1 back, 2 left, 3 right, 4 turn left, 5 turn right** — measured, not assumed: the
/// shipped Spartan `gaits` block puts `turn_left`/`turn_right` at `move_state` 4/5 and the four
/// movement gaits at 0..3, and `is_gait_direction_movement` asserts the direction is `<= 3`
/// (`haloreach_debug.dll` `sub_1808A6560`, `animation_definitions.cpp:3042`). See
/// [`crate::reach_actor::GaitDef`].
pub fn gait_direction(gait: Gait) -> Option<i16> {
    match gait {
        Gait::WalkFront | Gait::RunFront => Some(0),
        Gait::BackSlow | Gait::Back => Some(1),
        Gait::Left => Some(2),
        Gait::Right => Some(3),
        Gait::TurnLeft => Some(4),
        Gait::TurnRight => Some(5),
        Gait::Idle => None,
    }
}

/// #gait-blend  The two CARDINAL movement directions a travel angle lies between, and how far it
/// is toward the second.
///
/// `theta` is the direction of travel in the body's own frame: 0 straight ahead, `+pi/2`
/// strafing LEFT, `+-pi` backwards, `-pi/2` strafing right — the convention
/// [`BodyState::travel_angle`] is built with and the one `play_ctl::body_state` picks its gait
/// from. The returned directions are `animation_gait_directions` values (see
/// [`gait_direction`]).
///
/// The weight is **smoothstepped**, for the same reason the cross-fade is: a weight that is
/// linear in the angle has a kink where the bracketing pair changes, which is a step in the
/// pose's angular ACCELERATION every time the body passes straight ahead or straight sideways.
/// `t*t*(3-2t)` has zero derivative at both ends, so the mix leaves and enters each cardinal
/// clip smoothly. It still reads exactly 0.5 at 45 degrees, so a true diagonal is an even mix.
pub fn travel_directions(theta: f32) -> (i16, i16, f32) {
    let u = theta / std::f32::consts::FRAC_PI_2;
    let (a, b, t) = if u >= 1.0 {
        (2, 1, u - 1.0) // left -> back
    } else if u >= 0.0 {
        (0, 2, u) // front -> left
    } else if u >= -1.0 {
        (3, 0, u + 1.0) // right -> front
    } else {
        (1, 3, u + 2.0) // back -> right
    };
    let t = t.clamp(0.0, 1.0);
    (a, b, t * t * (3.0 - 2.0 * t))
}

pub fn anim_candidates(st: &BodyState, weapon: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(s) = &st.seat {
        // Seat modes come in `<vehicle>_<seat>` form with either `any`, a weapon class, or
        // nothing in the middle field.
        for mid in ["any", weapon, ""] {
            if mid.is_empty() {
                out.push(format!("{}:idle", s.mode));
            } else {
                out.push(format!("{}:{mid}:idle", s.mode));
            }
        }
        // Fall back to a plain standing pose rather than a walk cycle if the seat set is missing.
        out.push(format!("combat:{weapon}:idle"));
        return out;
    }
    // #film-action  A synchronised action overrides everything: it is a whole-body two-party
    // clip with its own root motion, and it is the one state here raised from a DECODED event.
    if let Some(sy) = &st.sync {
        let mut out = sync_candidates(sy.attacker_elite, sy.victim_elite, sy.side, sy.move_index);
        // If this biped's graph has none of them, stand rather than freeze — and `is_sync_clip`
        // then reports that nothing of the kind loaded.
        out.push(format!("combat:{weapon}:idle"));
        return out;
    }
    if st.stance == Stance::Dead {
        return death_candidates(st.death_variant, weapon, st.death_elite, st.death_airborne);
    }
    // #melee  A swing is NOT a base clip. Reach's melee clips are animation type 3,
    // REPLACEMENT, and a replacement animates only part of the skeleton -- 34 of a Spartan's 82
    // nodes on `combat:sword:melee_strike_1`, with `b_pelvis`, `b_l_calf` and `b_r_foot` carrying
    // nothing at all. Installed here as the body's sole clip it made every node the swing does
    // not animate fall back to the skeleton's BIND pose, which is why a meleeing player's legs
    // went strange. The locomotion clip stays the base and the swing goes in the REPLACEMENT
    // lane (`melee_candidates` -> `Actor::replacement`), which is what the engine does.
    // #jump  An airborne body is NOT in the `flight:` mode.
    //
    // This is the whole of "when they jump they look like they're in an A pose". `flight:` is the
    // JETPACK mode, not "off the ground": the shipped Spartan graph authors `flight:<class>:idle`
    // as a **70-frame looping HOVER** for every weapon class, with `flight:rifle:takeoff` (18f),
    // `flight:rifle:land` (24f) and a `flight:<class>:throttle_still_up` overlay beside it — the
    // word `throttle` is what settles it. A ballistic jump has its own clips in the combat set:
    // `combat:<class>:airborne_arc` (45f) and `combat:<class>:airborne` (40f), authored for every
    // armed class.
    //
    // The old order asked for `{mode}:{weapon}:{leaf}` first with `mode = "flight"`. A body
    // jumping with little horizontal speed derives `Gait::Idle`, so the first name tried was
    // `flight:<class>:idle` — which EXISTS — and the body was posed in the slack, upright jetpack
    // hover instead of a jump arc. A RUNNING jump asked for `flight:<class>:move_front_fast`,
    // which does not exist, fell through to `combat:<class>:airborne_arc` and looked right. That
    // is exactly the reported split: running jumps fine, standing and crouch jumps "A posed".
    //
    // Which armour ability a player is carrying is NOT in the recording, so a real jetpack flight
    // cannot be told from a jump and `flight:` is not asked for at all. Stated, not guessed.
    let mode = jmad_mode(st.stance);
    // #jump  A body that has just TOUCHED DOWN plays the landing, which is the other half of
    // "jumping/falling needs tuning": `combat:<class>:land_hard` (13 frames) is authored for every
    // armed class and `crouch:<class>:land_hard` (50) for the crouch set, and neither was ever
    // asked for, so a body snapped straight from the jump arc back to its walk cycle. A one-shot,
    // driven by the age of the touchdown rather than by the locomotion phase.
    if let (Some(_), false) = (st.land_ticks, st.stance == Stance::Flight) {
        out.push(format!("{mode}:{weapon}:land_hard"));
        if mode != "combat" {
            out.push(format!("combat:{weapon}:land_hard"));
        }
    }
    // #jump  The airborne arc comes FIRST, before any gait leaf: while a body is off the ground
    // its gait is not what it is doing.
    if st.stance == Stance::Flight {
        out.push(format!("{mode}:{weapon}:airborne_arc"));
        out.push(format!("{mode}:{weapon}:airborne"));
        // `crouch:` authors both of those too, so a crouch-jump keeps its own arc; the combat set
        // is the fallback when the mode does not.
        if mode != "combat" {
            out.push(format!("combat:{weapon}:airborne_arc"));
            out.push(format!("combat:{weapon}:airborne"));
        }
    }
    // #jump  The gait leaves are per MODE: `crouch:` names its cycles `walk_front` / `move_front`,
    // where `combat:` names them `move_front_slow` / `move_front_fast`. Asking the crouch mode for
    // a combat leaf is the same class of miss this function was just fixed for, so the table is
    // chosen by mode rather than shared.
    let leaf: &[&str] = if mode == "crouch" {
        match st.gait {
            Gait::Idle => &["idle"],
            Gait::TurnLeft => &["turn_left", "idle"],
            Gait::TurnRight => &["turn_right", "idle"],
            Gait::WalkFront => &["walk_front", "move_front"],
            Gait::RunFront => &["move_front", "walk_front"],
            Gait::BackSlow => &["walk_back", "move_back"],
            Gait::Back => &["move_back", "walk_back"],
            Gait::Left => &["move_left", "walk_left"],
            Gait::Right => &["move_right", "walk_right"],
        }
    } else {
        match st.gait {
            Gait::Idle => &["idle"],
            Gait::TurnLeft => &["turn_left", "idle"],
            Gait::TurnRight => &["turn_right", "idle"],
            Gait::WalkFront => &["move_front_slow", "move_front"],
            Gait::RunFront => &["move_front_fast", "move_front"],
            Gait::BackSlow => &["move_back_slow", "move_back"],
            Gait::Back => &["move_back", "move_back_slow"],
            Gait::Left => &["move_left"],
            Gait::Right => &["move_right"],
        }
    };
    for l in leaf {
        out.push(format!("{mode}:{weapon}:{l}"));
    }
    if st.stance == Stance::Sprint {
        out.push(format!("combat:{weapon}:move_front_fast"));
    }
    for l in leaf {
        out.push(format!("combat:{weapon}:{l}"));
    }
    out.push(format!("combat:{weapon}:idle"));
    // #wclass  Last resort: the GENERIC set. A weapon class read off the tag is not guaranteed to
    // exist in this biped's graph (`fixed`, the class every vehicle-mounted gun carries, has no
    // clips at all), and a candidate list that the graph cannot satisfy would leave the body
    // frozen on whatever it was playing before. The driver counts how often it lands here.
    if weapon != DEFAULT_WEAPON_CLASS {
        for l in leaf {
            out.push(format!("{mode}:{}:{l}", DEFAULT_WEAPON_CLASS));
        }
        for l in leaf {
            out.push(format!("combat:{}:{l}", DEFAULT_WEAPON_CLASS));
        }
        out.push(format!("combat:{}:idle", DEFAULT_WEAPON_CLASS));
    }
    out
}

/// #aim  The AIM OVERLAY clips to try for a body, best first.
///
/// Reach poses a look direction as an overlay blended onto the locomotion clip, not as a clip of
/// its own: `combat:<class>:aim_<state>_up`, per weapon class and per movement state. What a
/// frame of one of those MEANS is written into the clip (see `reach_actor::AimGrid`), so all this
/// has to do is name the right one.
///
/// The `_down` twin is deliberately NOT preferred: measured on `forge_halo`, the Spartan's
/// `combat:rifle:aim_still_up` and `aim_still_down` label the SAME +90..-90 degree pitch span, so
/// the `_up` clip alone covers a body looking anywhere, and asking for both would blend two
/// authorings of the same pose.
pub fn aim_candidates(st: &BodyState, weapon: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // A seated or dead body, and a body in a two-party action, has no aim to pose: the seat
    // animation and the sync clip own the whole skeleton.
    if st.seat.is_some() || st.sync.is_some() || st.stance == Stance::Dead {
        return out;
    }
    let state: &[&str] = match (st.stance, st.gait) {
        (Stance::Flight, _) => &["airborne_arc"],
        (_, Gait::Idle) => &["still"],
        (_, Gait::TurnLeft) => &["turn_left", "still"],
        (_, Gait::TurnRight) => &["turn_right", "still"],
        (_, Gait::WalkFront) => &["walk_front", "move_front"],
        (_, Gait::RunFront) => &["move_front", "walk_front"],
        (_, Gait::Back | Gait::BackSlow) => &["move_back", "walk_back"],
        (_, Gait::Left) => &["move_left", "move_front"],
        (_, Gait::Right) => &["move_right", "move_front"],
    };
    for cls in [weapon, DEFAULT_WEAPON_CLASS] {
        for st8 in state {
            out.push(format!("combat:{cls}:aim_{st8}_up"));
        }
        out.push(format!("combat:{cls}:aim_still_up"));
        if cls == DEFAULT_WEAPON_CLASS && weapon == DEFAULT_WEAPON_CLASS {
            break;
        }
    }
    // The spine-only aim, which is all some graphs have. Present on the Spartan, ABSENT on the
    // Elite -- which is why this is a candidate list and not a name.
    out.push("any:aim_spine".to_string());
    out
}

/// #fire  The FIRING overlay clips to try, best first. Type 2 (overlay) in every graph checked,
/// so a shot is a recoil added to whatever the body was already doing.
pub fn fire_candidates(weapon: &str) -> Vec<String> {
    let mut out = vec![format!("combat:{weapon}:fire_1")];
    if weapon != DEFAULT_WEAPON_CLASS {
        out.push(format!("combat:{}:fire_1", DEFAULT_WEAPON_CLASS));
    }
    out
}

/// #melee  The MELEE clips to try, best first.
///
/// These are type 3 (REPLACEMENT) in both shipped graphs, not overlays — a Reach melee replaces
/// the whole body pose — so they are driven as the body's base clip for their length, the same
/// one-shot treatment a death clip gets.
pub fn melee_candidates(weapon: &str, airborne: bool, variant: u8) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for cls in [weapon, DEFAULT_WEAPON_CLASS] {
        if airborne {
            out.push(format!("combat:{cls}:melee_airborne"));
        }
        // Two or three authored swings per class; which one is not recorded, so it is chosen
        // deterministically from the event's own tick (see the death clip for the same rule).
        for k in [variant % 3 + 1, 1, 2, 3] {
            out.push(format!("combat:{cls}:melee_strike_{k}"));
        }
        out.push(format!("combat:{cls}:melee_strike_dash"));
        if cls == weapon && weapon == DEFAULT_WEAPON_CLASS {
            break;
        }
    }
    out
}

/// #melee  Is `name` one of the melee clips (so the caption can count the bodies that got one)?
pub fn is_melee_clip(name: &str) -> bool {
    name.rsplit(':').next().is_some_and(|l| l.starts_with("melee_"))
}

/// #wclass The class component of a `mode:class:action` clip name, or `None` for a two-part name.
pub fn class_of_clip(name: &str) -> Option<&str> {
    let p: Vec<&str> = name.split(':').collect();
    (p.len() == 3 && !p[1].is_empty()).then_some(p[1])
}

// ---------------------------------------------------------------------------------------------
// #death  when a body's life ends, and what it falls with
// ---------------------------------------------------------------------------------------------

/// How long a body is left on the ground after it dies, seconds.
///
/// A SYNTHESISED number, and deliberately a short one. The engine keeps the real corpse until it
/// despawns — a median 20 s after the death, measured — but drawing every body that long litters
/// a busy clip (peak 17 corpses at once on `53C14E9D` at only 5 s, measured by
/// `probes::probe_death_census`), and a corpse costs an actor.
pub const CORPSE_SECONDS: f32 = 5.0;
/// [`CORPSE_SECONDS`] in 60 Hz ticks.
pub const CORPSE_TICKS: u32 = (CORPSE_SECONDS * 60.0) as u32;

// ---------------------------------------------------------------------------------------------
// #film-action  SYNCHRONISED ACTIONS — decoded, from the event channel
// ---------------------------------------------------------------------------------------------

/// How long a Reach assassination runs. The clips are authored at **75 frames at 30 fps** for
/// every ground move in the shipped set (`svs_ground_close_backstab`, `svs_ground_skull_stab`,
/// `svs_pounce_neck_snap`, ... all 75; `svs_airborne_smackdown` is 30), so 150 ticks at 60 Hz.
/// Read off the graph, not chosen.
pub const SYNC_TICKS: u32 = 150;

/// Which side of a two-party action a body is playing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncSide {
    Attacker,
    Victim,
}

/// #film-action  One synchronised action the recording states happened: an assassination.
///
/// **Decoded**, out of list B command kind 1 event 14 (`incident`). The engine broadcasts each
/// Reach medal with the medal's own engine string id and up to two object references — the
/// player who earned it and the other party — so the fact, both parties and the tick come
/// straight off the wire. See `crate::film_action` and `re-notes` (#film-action).
///
/// What is NOT decoded on an ordinary film is *which of the eight moves* played: that rides
/// `biped_assassinate` (event 22), which is a client→host request and so appears only on a film
/// recorded by the match host. `move_decoded` says which case this is, and the move is otherwise
/// chosen deterministically from the two entity ids so a clip always replays the same way.
#[derive(Clone, Debug, PartialEq)]
pub struct Assassination {
    /// The tick the incident fired — the moment the kill landed.
    pub tick: u32,
    /// The killer's biped entity. `None` on a kill the stream gives no killer for.
    pub attacker: Option<u32>,
    /// The victim's biped entity.
    pub victim: Option<u32>,
    /// The incident's own name: `finishing_move_kill`, `assassination_kill` or
    /// `terminal_velocity_assassination`.
    pub incident: &'static str,
    /// `biped_assassinate`'s 19-bit action index, when the film carried that event too.
    pub action_index: Option<u32>,
    /// True when `action_index` came off the wire, i.e. the MOVE is decoded and not chosen here.
    pub move_decoded: bool,
    /// #film-action  True when the RECORDING STATES this kill was an assassination -- an
    /// `incident` row on the event channel. False when HMS INFERRED it; see
    /// [`inferred_assassinations`] for exactly what from, and why the two have to be told apart
    /// everywhere they are shown.
    pub stated: bool,
}

impl Assassination {
    /// The window the two bodies are locked together for.
    pub fn covers(&self, tick: u32) -> bool {
        tick >= self.tick && tick < self.tick + SYNC_TICKS
    }
    /// Ticks into the action at `tick`.
    pub fn age(&self, tick: u32) -> u32 {
        tick.saturating_sub(self.tick)
    }
}

/// #film-action  Every assassination the recording states, in tick order.
///
/// `biped_assassinate` (event 22) is folded in where present: it names the same two parties, so
/// an event-22 record within a second of an incident is the same kill seen from the host's side,
/// and it upgrades that row's move from chosen to decoded.
pub fn assassinations(idx: &crate::film_decode::FilmIndex) -> Vec<Assassination> {
    let mut out: Vec<Assassination> = Vec::new();
    for ev in &idx.events {
        let Some(name) = ev.incident_label() else { continue };
        if !crate::film_action::ASSASSINATION_INCIDENTS.contains(&name) {
            continue;
        }
        out.push(Assassination {
            tick: ev.tick,
            attacker: ev.subject,
            victim: ev.other,
            incident: name,
            action_index: None,
            move_decoded: false,
            stated: true,
        });
    }
    // The host's own request record, where the film has one. Matched on the PAIR, inside a
    // second, because the request and the broadcast are the same kill a few ticks apart.
    for ev in idx.events.iter().filter(|e| e.kind == crate::film_action::EV_ASSASSINATE) {
        let Some(a) = out.iter_mut().find(|a| {
            a.attacker == ev.subject && a.victim == ev.other && ev.tick.abs_diff(a.tick) <= 60
        }) else {
            // A host-recorded film can carry the request with no incident at all (that film has
            // no replicated bipeds either), so the request stands on its own.
            out.push(Assassination {
                tick: ev.tick,
                attacker: ev.subject,
                victim: ev.other,
                incident: "biped_assassinate",
                action_index: ev.action_index,
                move_decoded: true,
                stated: true,
            });
            continue;
        };
        a.action_index = ev.action_index;
        a.move_decoded = true;
    }
    out.sort_by_key(|a| a.tick);
    out
}

/// #film-action  How near, and how far round the victim's back, an inferred assassination's
/// killer has to be. See [`inferred_assassinations`].
pub mod sync_infer {
    /// How close the two bodies have to be at the moment of death, world units.
    ///
    /// **From first principles, not fitted.** A Reach biped's collision cylinder is about 0.35 wu
    /// in radius, so two body ORIGINS under 1.0 wu apart are interpenetrating — which the
    /// engine's own collision does not allow unless it has LOCKED the pair into a synchronised
    /// action and is driving both from one root.
    pub const NEAR_WU: f32 = 1.0;
    /// How far either side of the death a `damage_aftermath` counts as the killing blow.
    pub const DAMAGE_BEFORE: u32 = 120;
    pub const DAMAGE_AFTER: u32 = 60;
}

/// #film-action  **REJECTED, MEASURED, AND NOT WIRED IN.** Nothing calls this. It is kept, with
/// its numbers, because the next person to look at "most assassinations are missing" will reach
/// for exactly these two ideas and should be able to see that both were tried and what they cost.
///
/// **THE VERDICT (held-out corpus, 6 assassination-titled clips and 9 others, the development
/// clip excluded):** 3 of the 4 pre-registered criteria FAIL.
///
/// | criterion | bar | measured | |
/// |---|---|---|---|
/// | recall on held-out stated rows | >= 75% | **11/22 = 50%** | FAIL |
/// | contrast, assassination clips vs others | >= 5x | **77.57% / 49.20% = 1.58x** | FAIL |
/// | specificity on non-assassination clips | < 5.00% | **49.20%** | FAIL |
/// | prevalence on assassination clips | 25-90% | **77.57%** | pass |
///
/// And the confound check settles what it really is: **87-90% of the rows it fires on are ones
/// the rejected melee test also fires on**. It is a close-range-melee-kill detector wearing an
/// assassination's name, and 1.58x is worse than the 1.8x geometry test this project already
/// threw out.
///
/// **So HMS stages the 8 assassinations the recording STATES and no others**, and the panel says
/// the recording carries only the recorder's own kill feed. That is the honest end of it: the
/// corpus has been spent — each registered rule consumes its held-out set — so a third
/// hypothesis would need clips nobody here has looked at.
///
/// What follows is the reasoning and the two negatives, kept for the record.
///
/// ---
///
/// The assassinations the recording does NOT state, inferred from the channels it does carry.
///
/// **Why this exists.** A Reach film states only the RECORDER'S OWN kill feed. Measured on the
/// user's `Chill Assassinations.` clip: the recording destroys **163 bodies** and its `incident`
/// channel carries **16 kill rows, 8 of them assassinations** -- and all 16 name the *same*
/// killer, a player with one life and no deaths, i.e. the person recording. Medals are personal
/// and a film is one client's view, so nobody else's assassination is on the wire as an incident
/// at all. That is the whole of "the assassinations do NOT play dude, the victim just flops on
/// the ground": for 155 of those 163 deaths there was nothing to stage.
///
/// **What this uses instead**, one decoded fact and one physical one:
///
/// 1. `damage_aftermath` (698 on that clip) carries two object references, and the generic
///    header reader already decodes both: reference 0 is the object damaged and reference 1 is
///    the responsible unit. The last one naming this body before it died is a STATED KILLER for
///    a death the kill feed never mentions. **Validated 8 of 8**: on every assassination the
///    recording does state, the responsible unit it names is the same killer the kill feed
///    names, arrived at independently.
/// 2. The two bodies are INTERPENETRATING at the moment of death — closer than
///    [`sync_infer::NEAR_WU`], which is inside a biped's own collision radius. Normal collision
///    makes that impossible; the engine has locked the pair onto one root, which is what a
///    synchronised action is.
///
/// **TWO THINGS THAT WERE TRIED AND MEASURED OUT, recorded so they are not tried again.**
///
/// * **`biped_melee_initiate` does NOT mark an assassination.** The first version of this rule
///   required it, on the reasoning that a Reach finisher is what a melee from behind turns into.
///   Measured: the killer of all 8 stated assassinations on the development clip fired **zero**
///   melee-initiate events in the whole recording, so recall was **0 of 8** while the rule fired
///   on 69 of 160 deaths — i.e. it was finding ordinary beatdowns. Rejected.
/// * **"From behind" is NOT supported at the kill tick.** The victim's decoded facing puts the
///   killer 177.7, 73.7, 8.5, 112.9, 105.9, 73.1, 16.9 and 158.2 degrees off their back on those
///   same eight — four of them in FRONT, because the sync action has already turned the victim
///   by the time the incident fires. A 90-degree cone would have failed half of them on its own.
///
/// **THE ACCEPTANCE BAR, REGISTERED BEFORE THE PROBE WAS RUN.** The four constants above and the
/// four windows in [`sync_infer`] are part of the hypothesis, not knobs: this rule ships only if
/// all four of these hold on the numbers as first measured, and a parameter changed after seeing
/// a result makes it a NEW experiment that has to be re-measured on data not already looked at.
///
/// Because the development clip's own distances were inspected when the first rule was rejected,
/// **recall on that clip is in-sample and is NOT a criterion** -- it is reported and flagged. The
/// bar is on clips that had not been looked at when the parameter was frozen:
///
/// 1. **Recall >= 75%** of the stated assassination victims on held-out assassination clips.
/// 2. **Contrast >= 5x** -- the share of deaths fired on, pooled over held-out assassination
///    gametypes, against the share pooled over every non-assassination clip. (The geometry-only
///    test this project rejected managed 1.8x; the melee version above managed 0 recall.)
/// 3. **Specificity: under 5.00% of deaths on non-assassination clips**, reported to two
///    decimals whichever way it lands.
/// 4. **Prevalence between 25% and 90% on held-out assassination clips.** Below 25% it would not
///    fix what the user reported; above 90% it is just calling every death an assassination.
///
/// `probes::probe_inferred_assassination_recall_and_contrast` takes `HMS_PROBE_FILM` so one clip
/// can be developed against with the rest of the corpus untouched. The stated rows are also a
/// BIASED oracle and have to be read that way: they are all the RECORDER's own kills, and a
/// player recording an assassination clip is recording their own highlights, so recall against
/// them may flatter the rule on other players' deaths.
///
/// **This is DERIVED and it is labelled as derived everywhere it is shown** -- `Assassination::
/// stated` is false, the body's provenance is `Derived`, and the panel says the recording did
/// not state this one. The bar is deliberately higher than the geometry-only test this project
/// measured and rejected (28.4% on assassination gametypes against 15.6% everywhere else, which
/// is not a classifier); that test had neither the killer nor the melee, both of which are on
/// the wire and decoded.
pub fn inferred_assassinations(
    idx: &crate::film_decode::FilmIndex,
    stated: &[Assassination],
) -> Vec<Assassination> {
    use sync_infer::*;
    let already: std::collections::HashSet<u32> = stated.iter().filter_map(|a| a.victim).collect();
    let mut out: Vec<Assassination> = Vec::new();
    for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
        let Some(d) = death_tick(t) else { continue };
        if already.contains(&t.net_id) {
            continue;
        }
        // 1. the killing blow's responsible unit
        let Some(k) = idx
            .events
            .iter()
            .filter(|e| {
                e.kind == crate::film_action::EV_DAMAGE_AFTERMATH
                    && e.subject == Some(t.net_id)
                    && e.tick <= d + DAMAGE_AFTER
                    && e.tick + DAMAGE_BEFORE >= d
            })
            .max_by_key(|e| e.tick)
            .and_then(|e| e.other)
            .filter(|&k| k != t.net_id)
        else {
            continue;
        };
        // 2. and the two were INSIDE each other when it died
        let Some(kt) = idx.tracks.iter().find(|q| q.net_id == k && q.entity_type == 11) else { continue };
        let (Some(kp), Some(vp)) = (kt.sample(d), t.sample(d)) else { continue };
        let dist = ((kp[0] - vp[0]).powi(2) + (kp[1] - vp[1]).powi(2) + (kp[2] - vp[2]).powi(2)).sqrt();
        if dist > NEAR_WU {
            continue;
        }
        out.push(Assassination {
            tick: d,
            attacker: Some(k),
            victim: Some(t.net_id),
            incident: "inferred from the damage and melee channels",
            action_index: None,
            move_decoded: false,
            stated: false,
        });
    }
    out
}

/// The shipped Spartan-on-Spartan assassination moves, read off `objects\characters\spartans\
/// spartans`'s own graph (all 1,393 animations enumerated, not a sample).
///
/// Both halves of each pair are in that one graph, as
/// `any:any:any:sync_actions:<move>_attacker` and `..._victim`.
pub const SVS_MOVES: [&str; 8] = [
    "svs_ground_close_backstab",
    "svs_ground_skull_stab",
    "svs_ground_left_backstab_tear",
    "svs_ground_right_gutstab_drop",
    "svs_pounce_neck_snap",
    "svs_phonebooth_ground_hold_to_gutstab",
    "svs_phonebooth_pounce_hold_to_gutstab",
    "svs_airborne_smackdown",
];

/// Elite-on-Elite: both halves live in `objects\characters\elite\elite`.
pub const EVE_MOVES: [&str; 6] = [
    "eve_ground_back_headstab",
    "eve_ground_right_backstab",
    "eve_ground_choke_to_bellystab",
    "eve_phonebooth_ground_pull_head_neckstab",
    "eve_phonebooth_pounce_pull_head_neckstab",
    "eve_airborne_swipe",
];

/// Spartan attacker, Elite victim. **The attacker half is in the SPARTAN's graph and the victim
/// half is in the ELITE's** — a cross-species pair is split across the two skeletons, which is
/// why each side has to be looked up in its own body's graph.
pub const SVE_MOVES: [&str; 8] = [
    "sve_ground_left_necksnapper",
    "sve_ground_left_eyedrag",
    "sve_ground_right_snap_tackle_pop",
    "sve_ground_climb_neckstab",
    "sve_phonebooth_ground_skullstab",
    "sve_phonebooth_pounce_skullstab",
    "sve_pounce_jump_to_headsmash",
    "sve_airborne_neckstab",
];

/// Elite attacker, Spartan victim. Victim half in the Spartan's graph.
pub const EVS_MOVES: [&str; 8] = [
    "evs_ground_push_down_backstab",
    "evs_ground_push_down_headstab",
    "evs_ground_right_swipe_back_kick",
    "evs_ground_left_stab_stab_slice",
    "evs_pounce_pin_headstab",
    "evs_phonebooth_ground_dual_neckstab",
    "evs_phonebooth_pounce_dual_neckstab",
    "evs_airborne_toss_behind",
];

/// Is this biped tag path an Elite? Reach's two playable species are the only ones that carry a
/// `sync_actions` set, so the question is binary.
pub fn is_elite(biped_tag_path: &str) -> bool {
    biped_tag_path.contains("characters\\elite")
}

/// The move set for an (attacker, victim) species pair, and the prefix that names it.
pub fn sync_move_set(attacker_elite: bool, victim_elite: bool) -> &'static [&'static str] {
    match (attacker_elite, victim_elite) {
        (false, false) => &SVS_MOVES,
        (false, true) => &SVE_MOVES,
        (true, false) => &EVS_MOVES,
        (true, true) => &EVE_MOVES,
    }
}

/// The clip candidates for one side of an assassination, best first.
///
/// `move_index` picks the move; `None` means "choose deterministically", which a caller does by
/// passing the pair's own ids. The whole set is appended after the chosen move so that a graph
/// missing one clip still plays a real assassination rather than nothing, and the caller is told
/// (via [`is_sync_clip`]) whether what loaded was one at all.
pub fn sync_candidates(
    attacker_elite: bool,
    victim_elite: bool,
    side: SyncSide,
    move_index: usize,
) -> Vec<String> {
    let set = sync_move_set(attacker_elite, victim_elite);
    let suffix = match side {
        SyncSide::Attacker => "attacker",
        SyncSide::Victim => "victim",
    };
    let n = set.len();
    (0..n).map(|i| format!("any:any:any:sync_actions:{}_{suffix}", set[(move_index + i) % n])).collect()
}

/// Is this clip name one of the two-party `sync_actions` clips? The driver uses it the way it
/// uses [`is_death_clip`] — to say honestly in the panel when the graph had no such clip and the
/// body is standing in a pose instead of being assassinated.
pub fn is_sync_clip(name: &str) -> bool {
    name.contains(":sync_actions:")
}

/// #death The tick a biped DIED, or `None` if this track is not a life that ended.
///
/// **The recording never says a body died.** The biped's 40-field derived dirty block was read
/// off `haloreach.dll` (`sub_1803B4B88`) and swept over 5.4 M messages: there is no death bit, no
/// ragdoll bit and no action enum in any of them (`probes::probe_biped_mask_census` — bits 0x34
/// and 0x36..0x3B never appear, not once). What the stream does carry is the shape of the track,
/// and that pins the moment:
///
/// * the kind-4 **destroy** is the corpse DESPAWN, not the death — it lands a median **974 ticks
///   (16 s)** after the body's last message, and a median 10-38 s *after the same player's next
///   life has already started*, which no death can do;
/// * `last_tick` is the last message of any kind, a median **239 ticks (4 s)** after the body's
///   last reported POSITION — which is why a killed player used to stand frozen on the spot for
///   several seconds and then vanish;
/// * the **last reported position** is the death. Oracle: a player's next life starts a fixed
///   delay after they die, so the tightest spread of `next_life.first_tick - candidate` wins.
///   Over 12 films the last position wins on **10**, `last_tick` on 2 and the destroy on **0**,
///   and its medians are physically right — +3.1 s to +8.9 s before the respawn, where Reach's
///   own respawn delay is 5 s. On the largest well-paired film (510 pairs) the inter-quartile
///   range is **39 ticks (0.65 s)** against 158 for `last_tick` and 1,670 for the destroy: a 4x
///   and a 43x margin. See `probes::probe_death_instant_oracle`.
///
/// A track with no destroy was still alive when the recording stopped, so it has no death.
pub fn death_tick(t: &crate::film_decode::EntityTrack) -> Option<u32> {
    if t.entity_type != 11 {
        return None;
    }
    t.destroyed_tick?;
    let last = t.positions.last()?.0;
    // Guard the pathological track whose destroy arrives BEFORE its last position (a recycled
    // net id, measured down to -18,610 ticks): the track is two lives spliced together and there
    // is no single moment to end. One second of slack for an out-of-order message.
    (t.destroyed_tick? + 60 >= last).then_some(last)
}

/// #film-action  The tick a biped died **as the recording states it**, with the incident that
/// named the kill and the killer's entity where there was one.
///
/// This is strictly better than [`death_tick`], which infers the moment from the shape of the
/// track (its last reported position) because the old reading of the stream had nothing else.
/// The event channel states it: an `incident` naming this body as the other party of a kill is
/// the kill. Prefer this and fall back to the derived moment only where the film's event channel
/// carried nothing for this body.
pub fn death_from_events(
    idx: &crate::film_decode::FilmIndex,
    net_id: u32,
) -> Option<(u32, &'static str, Option<u32>)> {
    idx.events
        .iter()
        .filter(|e| e.other == Some(net_id))
        .filter_map(|e| {
            let n = e.incident_label()?;
            // A kill names the victim second. `death`, `suicide` and `fell_to_death` are the
            // rows for a death with no killer, and they name the victim second too.
            (crate::film_action::KILL_INCIDENTS.contains(&n) || matches!(n, "death" | "suicide" | "fell_to_death"))
                .then_some((e.tick, n, e.subject))
        })
        // The FIRST such incident: a body can appear in later rows as a corpse (`avenged`).
        .min_by_key(|&(t, _, _)| t)
}

/// #death The death clips of **this body's own graph**, in the order it should try them.
///
/// There is no `death_*` or `die_*` set in either playable graph, which is why the name hunt that
/// looked for one came back empty. What they do have, enumerated in full off the shipped caches
/// (`--dump-jmad forge_halo.map spartans 2048`, 1,393 entries, and `elite`, 1,271 — the default
/// 12-entry limit is what once made a whole set look absent):
///
/// | | Spartan (`characters\spartans\spartans`) | Elite (`characters\elite\elite`) |
/// |---|---|---|
/// | hit the ground | `combat:any:landing_dead` (34f), `combat:landing_dead_back` (60f), `combat:landing_dead_back:var1` (76f) | `combat:landing_dead` (25f) |
/// | killed in the air | `combat:any:airborne_dead` (149f), `combat:any:airborne_dead:var1` (150f), `combat:airborne_dead_back` (149f), `combat:airborne_dead_back:var1` (150f) | `combat:airborne_dead` (149f), `:var1` (150f), `:var2` (149f) |
///
/// **The Elite's names are not the Spartan's** — `combat:landing_dead` has no `any:` field — and
/// the old hard-coded list held only the Spartan's three. An Elite therefore matched nothing and
/// fell through to its idle: "elites don't ragdoll at all when they die". So the list is built
/// per species, from the graph that will actually be asked for it.
///
/// `airborne` picks the mid-air flail over the ground landing. That is DERIVED, from the decoded
/// vertical speed of this body's own last reported motion — a player shot off a tower should
/// flail down, not lie flat in the air — and it is the only part of the choice the recording
/// supports at all.
///
/// Within a set the list is ROTATED by `variant` so the same entity always falls the same way
/// however often the clip is re-watched. **Which ragdoll really happened is NOT recorded**: a
/// Reach death is a physics ragdoll, and the one thing on the wire that could narrow it — the
/// damage direction inside `damage_aftermath`'s 122-bit parameter block — is undecoded (no bit
/// offset for it is known). The panel says so rather than implying the fall is the film's own.
///
/// The other species' names are appended after this one's, so a third biped (a campaign Grunt,
/// a Monitor) still gets whatever it has, and the ordinary idle goes last so a graph with none of
/// them stands rather than freezing mid-pose — the caption counts the bodies that got a real
/// death clip and names the clip each one is on.
pub fn death_candidates(variant: u8, weapon: &str, elite: bool, airborne: bool) -> Vec<String> {
    const SPARTAN_GROUND: [&str; 3] =
        ["combat:any:landing_dead", "combat:landing_dead_back", "combat:landing_dead_back:var1"];
    const SPARTAN_AIR: [&str; 4] = [
        "combat:any:airborne_dead",
        "combat:any:airborne_dead:var1",
        "combat:airborne_dead_back",
        "combat:airborne_dead_back:var1",
    ];
    const ELITE_GROUND: [&str; 1] = ["combat:landing_dead"];
    const ELITE_AIR: [&str; 3] = ["combat:airborne_dead", "combat:airborne_dead:var1", "combat:airborne_dead:var2"];
    let rotated = |set: &[&str]| -> Vec<String> {
        let n = set.len();
        (0..n).map(|i| set[(variant as usize + i) % n].to_string()).collect()
    };
    let (own_ground, own_air, other_ground, other_air) = if elite {
        (&ELITE_GROUND[..], &ELITE_AIR[..], &SPARTAN_GROUND[..], &SPARTAN_AIR[..])
    } else {
        (&SPARTAN_GROUND[..], &SPARTAN_AIR[..], &ELITE_GROUND[..], &ELITE_AIR[..])
    };
    let mut out: Vec<String> = Vec::new();
    // This body's own graph first, the wanted orientation first within it.
    if airborne {
        out.extend(rotated(own_air));
        out.extend(rotated(own_ground));
    } else {
        out.extend(rotated(own_ground));
        out.extend(rotated(own_air));
    }
    // Then the other species' names, for a biped that is neither.
    if airborne {
        out.extend(rotated(other_air));
        out.extend(rotated(other_ground));
    } else {
        out.extend(rotated(other_ground));
        out.extend(rotated(other_air));
    }
    out.push(format!("combat:{weapon}:idle"));
    out
}

/// #death The state of a body that is lying dead, `age` ticks after its life ended.
///
/// `elite` is the species of the graph that will be asked for the clip, and `airborne` says the
/// body was off the ground when it died — see [`death_candidates`] for both. `Synthesised`, and
/// the `because` clause says exactly what is and is not in the recording: WHICH ragdoll played
/// is not on the wire in any form this reader can read.
pub fn dead_state(net_id: u32, age: u32, elite: bool, airborne: bool) -> BodyState {
    BodyState {
        stance: Stance::Dead,
        gait: Gait::Idle,
        dead_ticks: age,
        // Deterministic in the entity id, so re-watching a clip shows the same fall every time.
        // 12 is a common multiple of every set's length, so the rotation is uniform whichever
        // species and orientation `death_candidates` ends up using.
        death_variant: (net_id.rotate_right(8) % 12) as u8,
        death_airborne: airborne,
        death_elite: elite,
        provenance: Provenance::Synthesised,
        because: "the recording states the kill and the tick, but not which ragdoll played",
        ..Default::default()
    }
}

/// #death Was this body OFF THE GROUND when its life ended?
///
/// Derived, from the last pair of reported positions at or before `tick`: a body whose vertical
/// speed over that pair is past the airborne cut was falling or rising, so the `airborne_dead`
/// flail is the right clip and the ground landing is not. A teleport (a respawn, or a recycled
/// net id) is excluded by the same horizontal-speed guard [`cycle_phase_with`] uses.
///
/// This is the whole of what the recording supports about HOW a body died. The rest — which of
/// the engine's ragdolls, from which direction — is not decoded; see [`death_candidates`].
pub fn died_airborne(t: &crate::film_decode::EntityTrack, tick: u32) -> bool {
    let i = t.positions.partition_point(|&(k, _)| k <= tick);
    if i < 2 {
        return false;
    }
    let (t0, p0) = t.positions[i - 2];
    let (t1, p1) = t.positions[i - 1];
    if t1 <= t0 {
        return false;
    }
    let dt = (t1 - t0) as f32 / 60.0;
    let ground = (p1[0] - p0[0]).hypot(p1[1] - p0[1]) / dt;
    if ground > 12.0 {
        return false;
    }
    ((p1[2] - p0[2]) / dt).abs() > cuts::AIRBORNE_VZ
}

/// #death How close the nearest other player was when a body died, and from what direction.
///
/// `(distance in wu, bearing from the victim's BACK in degrees)`. This is a geometric observation
/// about one instant, nothing more — see [`death_company`] for exactly how much it is worth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeathCompany {
    pub distance: f32,
    /// Degrees away from directly behind the victim, 0..180. `None` when the victim reported no
    /// facing, so "behind" cannot be defined.
    pub from_behind: Option<f32>,
}

/// How near the victim's back another player was at [`death_tick`] — **weak, and labelled as such**.
///
/// The first attempt at finding assassinations looked for two bodies held at a fixed offset for a
/// couple of seconds; it found MORE candidate pairs on ordinary clips than on a gametype built
/// around assassinations, and was killed (see `probes::probe_sync_pair_contrast`). This asks a
/// different question that needs only ONE instant, now that the death instant is pinned: at the
/// moment the body died, was another player within 2 wu, and within 70 degrees of directly behind?
///
/// **Measured contrast** (`probes::probe_death_from_behind_contrast`, 15 films, 4,212 deaths):
/// assassination gametypes **85/299 = 28.4%**, every other clip **612/3,913 = 15.6%**. So the
/// signal is real and in the RIGHT direction this time (1.8x, and the three assassination clips
/// score 29.6 / 23.2 / 30.5% against a 15.6% baseline) — but it is **nowhere near a classifier**:
/// one ordinary social clip scores 43.7%, higher than all three assassination clips, and 15.6% of
/// ordinary deaths pass the same test. Per clip it is 3 against 12 with heavy overlap.
///
/// So HMS **reports the geometry and names it as geometry**. It does not call a death an
/// assassination, and it does not stage a two-body assassination performance: the stream carries
/// no killer, no action enum and no filled assassination record (see the negatives in
/// `re-notes/reach_saved_film_layout.md`), and a 1.8x contrast cannot stand in for one.
pub fn death_company(
    tracks: &[crate::film_decode::EntityTrack],
    victim: &crate::film_decode::EntityTrack,
    tick: u32,
) -> Option<DeathCompany> {
    const NEAR_WU: f32 = 2.0;
    let vp = victim.sample(tick)?;
    let face = victim.facing_at(tick);
    let mut best: Option<DeathCompany> = None;
    for k in tracks.iter().filter(|t| t.entity_type == 11) {
        if k.net_id == victim.net_id || (k.owner_raw.is_some() && k.owner_raw == victim.owner_raw) {
            continue;
        }
        if !k.alive_at(tick) || death_tick(k).is_some_and(|kd| kd < tick) {
            continue;
        }
        let Some(kp) = k.sample(tick) else { continue };
        let d = ((kp[0] - vp[0]).powi(2) + (kp[1] - vp[1]).powi(2) + (kp[2] - vp[2]).powi(2)).sqrt();
        if d > NEAR_WU || best.is_some_and(|b| b.distance <= d) {
            continue;
        }
        let from_behind = face.map(|y| {
            let to = (kp[1] - vp[1]).atan2(kp[0] - vp[0]);
            ang_delta(to, y + std::f32::consts::PI).abs().to_degrees()
        });
        best = Some(DeathCompany { distance: d, from_behind });
    }
    best
}

/// Is `name` one of the death clips (so the caption can count the bodies that really got one)?
///
/// Both halves of the set count: `landing_dead` is the body hitting the ground and
/// `airborne_dead` is the flail of a body killed in mid-air, which [`death_candidates`] now
/// plays when the recording's own vertical speed says the body was off the ground.
pub fn is_death_clip(name: &str) -> bool {
    name.contains("landing_dead") || name.contains("airborne_dead")
}

// ---------------------------------------------------------------------------------------------
// driving the cycle: how far through its stride a body is
// ---------------------------------------------------------------------------------------------

/// How many full animation CYCLES one world unit of travel is worth, per gait.
///
/// **Why this is not one number.** The build before this had a single global
/// `CYCLE_FRAMES_PER_WU = 7.5` for every clip. But `move_front_slow`, `move_front_fast`,
/// `move_back`, `move_left/right` and `sprint:*` have different stride lengths AND different
/// frame counts (24, 22, 18, 19/18 and 17 frames respectively), so one constant cannot match
/// them: at 7.5 a walking body's cycle advances barely a third as fast as its feet travel, which
/// is exactly the stiff, gliding, over-reaching look the owner described.
///
/// **Why it is measured rather than read off the tag.** A Reach locomotion clip is authored IN
/// PLACE: `probes::probe_clip_root_motion` samples the root node at the first and last frame of
/// all eleven of them and every one translates by exactly (0.000, 0.000). The stride length is in
/// the `movement_data` block, which this reader does not decode — so the number comes from the
/// recordings instead. A clip loops once per stride at 30 fps, so a body performing that gait at
/// its own natural speed `v` covers one cycle every `n / 30 * v` world units, i.e.
/// `cycles_per_wu = 30 / (v * n)`.
///
/// `v` is the MEDIAN ground speed measured for that gait across 8 films
/// (`probes::probe_speed_per_gait`, 1.3 M samples) and `n` is the clip's authored frame count:
///
/// | gait | v (wu/s) | n | cycles/wu |
/// |------|----------|---|-----------|
/// | walk forward | 1.32 | 24 | 0.95 |
/// | run forward | 3.52 | 22 | 0.39 |
/// | back | 2.72 | 18 | 0.61 |
/// | back slow | 1.26 | 24 | 0.99 |
/// | strafe left | 1.53 | 19 | 1.03 |
/// | strafe right | 1.53 | 18 | 1.09 |
/// | sprint | 6.15 | 17 | 0.29 |
///
/// `None` means the gait has no ground travel to drive it (idle, turning on the spot, airborne),
/// so its cycle runs on the CLOCK instead — see [`cycle_phase`].
pub fn cycles_per_wu(stance: Stance, gait: Gait) -> Option<f32> {
    let (v, n) = gait_speed_and_frames(stance, gait)?;
    Some(CLIP_FPS / (v * n))
}

/// The MEASURED natural speed of a gait (wu/s) and the authored frame count of the clip that
/// plays it — the two numbers `cycles_per_wu` is built from, exposed separately so the frame
/// count can come from the clip that is ACTUALLY loaded.
///
/// Keying the SPEED on the gait alone is still an approximation: `v` was measured across films
/// without knowing what anyone was carrying, so it is the population's median for that gait, not
/// the rocket launcher's. The FRAME COUNT is no longer assumed — the driver learns the real one
/// from the clip it loaded, per weapon class, and feeds it to [`cycles_per_wu_for_clip`]. The
/// counts below are the rifle set's, used only until a clip of that class has been loaded once.
pub fn gait_speed_and_frames(stance: Stance, gait: Gait) -> Option<(f32, f32)> {
    if stance == Stance::Sprint {
        return Some((6.15, 17.0));
    }
    if stance == Stance::Flight || stance == Stance::Seated {
        return None;
    }
    match gait {
        Gait::Idle | Gait::TurnLeft | Gait::TurnRight => None,
        Gait::WalkFront => Some((1.32, 24.0)),
        Gait::RunFront => Some((3.52, 22.0)),
        Gait::Back => Some((2.72, 18.0)),
        Gait::BackSlow => Some((1.26, 24.0)),
        Gait::Left => Some((1.53, 19.0)),
        Gait::Right => Some((1.53, 18.0)),
    }
}

/// As [`cycles_per_wu`], but for the clip that is actually loaded: `frames` is its own frame
/// count, so a clip with a different length than the rifle set's is driven at its own rate rather
/// than the rifle's.
pub fn cycles_per_wu_for_clip(stance: Stance, gait: Gait, frames: f32) -> Option<f32> {
    let (v, _) = gait_speed_and_frames(stance, gait)?;
    (frames >= 1.0).then(|| CLIP_FPS / (v * frames))
}

/// #play-jolt  The frame a CLOCK-DRIVEN clip is on at `seconds`, played forward and then
/// BACKWARD — a triangle wave over `[0, frames - 1]` rather than a wrap.
///
/// **Because Reach's idle clips are not loops, and that is measured.** Posing every frame of the
/// shipped graph and taking the largest distance any one bone moves between neighbouring frames
/// (`play_mode::probes::probe_whether_the_clock_driven_clips_loop`, `forge_halo`), the wrap from
/// the last frame back to the first, against the 90th percentile of an ordinary frame step:
///
/// ```text
///   combat:rifle:move_front_fast   22 frames   wrap = 1.4x p90   <- loops
///   combat:rifle:turn_left         18 frames   wrap = 0.6x p90   <- loops
///   combat:rifle:idle             156 frames   wrap = 72.8x p90  <- DOES NOT
///   crouch:rifle:idle              77 frames   wrap = 476.8x p90 <- DOES NOT
/// ```
///
/// The Spartan idle is still to within 0.0006 wu for 155 frames and then its last frame sits
/// 0.0426 wu from its first; the crouch idle's is 0.183 wu out. Wrapping that is a visible pop
/// once a cycle — reported as *"the idle animation seems to play through then skip/shift the
/// biped"*. The clips' `loop frame index` is 0 on every one of them, so the tag does not
/// distinguish them, and the ROOT NODE translates by exactly (0,0,0) over all of them, so this is
/// **not** root motion and nothing is fighting the controller for the body's position.
///
/// **Playing back down the clip is HMS's choice, not the engine's**, and it is the only one that
/// is pop-free by construction: the phase is continuous at both ends, so no frame can ever be a
/// discontinuity whatever the clip's geometry. The price is that a cycle takes twice as long (a
/// Spartan's idle breathes over 10.4 s instead of 5.2) and that half of it plays in reverse,
/// which on an idle is invisible. A clip that DOES loop is unharmed by it.
///
/// `frames <= 1` has no timeline and returns 0.
pub fn ping_pong_frame(seconds: f32, frames: f32) -> f32 {
    let last = frames - 1.0;
    if !(last > 0.0) {
        return 0.0;
    }
    let t = (seconds * CLIP_FPS).rem_euclid(2.0 * last);
    if t <= last {
        t
    } else {
        2.0 * last - t
    }
}

/// #play-rate  How many animation CYCLES one world unit of travel is worth, for a clip of
/// `frames` frames whose body covers ground at `speed` wu/s while it plays once at 30 fps.
///
/// The same arithmetic as [`cycles_per_wu_for_clip`] with BOTH inputs supplied by the caller —
/// `30 / (speed × frames)` — so a host that knows the authored speed for a state does not have
/// to go through a `(stance, gait)` table that does not.
///
/// Test mode is that host: `play_phys::PlayerPhysics::gait_speed` reads the speed out of
/// `matg @player information`, per species AND per stance, which is what the film-corpus medians
/// behind [`gait_speed_and_frames`] cannot be (they are one bucket's median across every species
/// and both stances). Theater keeps the medians, because a recorded body's speed is whatever it
/// was.
///
/// `None` when either input cannot define a stride.
pub fn cycles_per_wu_at_speed(speed: f32, frames: f32) -> Option<f32> {
    (speed > 1e-3 && frames >= 1.0).then(|| CLIP_FPS / (speed * frames))
}

/// Frames per second a clock-driven clip (idle, a turn, an airborne arc) plays at. The authored
/// rate; `reach_jmad::FRAME_RATE` is the same number.
pub const CLIP_FPS: f32 = 30.0;

/// #anim-rate  The authored LENGTH, in frames, of the clip a clock-driven state plays — the
/// seed used until the driver has loaded that clip once and can report its real length.
///
/// Read off the shipped graphs, not chosen (`--dump-jmad <map> spartans 2048`):
/// `combat:<class>:idle` is **156** frames on the Spartan and **50** on the Elite,
/// `turn_left` / `turn_right` are **18** on both, `flight:<class>:idle` is **70** on the Spartan
/// and **30** on the Elite. Where the two species disagree this returns the Spartan's, because
/// that is the default pool; the real number replaces it on the next frame, per graph, from the
/// clip that actually loaded (`ActorDriver::clip_frames`, keyed on the graph as well as the
/// class for exactly this reason).
///
/// `None` only for a state that has no clock-driven clip at all, which the caller treats as one
/// second.
pub fn clock_clip_frames(stance: Stance, gait: Gait) -> Option<f32> {
    match stance {
        // `flight:<class>:idle` on the Spartan; an airborne body whose graph has no flight set
        // falls back to `combat:<class>:airborne_arc`, 45 frames, which is close enough for one
        // frame of seed.
        Stance::Flight => Some(70.0),
        // A seated pose holds frame 0, so its rate never matters.
        Stance::Seated => None,
        _ => match gait {
            Gait::Idle => Some(156.0),
            Gait::TurnLeft | Gait::TurnRight => Some(18.0),
            // Every other gait is stride-driven and never reaches here.
            _ => None,
        },
    }
}

/// The gait over one segment of a track, from that segment alone.
///
/// A cheap, allocation-free equivalent of the stance/gait half of [`derive`] for use inside
/// [`cycle_phase`], which visits every segment: the full `derive` re-walks the whole track and
/// does several binary searches, so calling it per segment made the per-body cost quadratic.
/// The thresholds are the same ones.
fn segment_gait(
    t: &crate::film_decode::EntityTrack,
    t0: u32,
    p0: [f32; 3],
    t1: u32,
    p1: [f32; 3],
    dt: f32,
    step: f32,
) -> (Stance, Gait) {
    let ground = step / dt;
    let vz = (p1[2] - p0[2]) / dt;
    if vz.abs() > cuts::AIRBORNE_VZ {
        return (Stance::Flight, if ground > cuts::WALK { Gait::RunFront } else { Gait::Idle });
    }
    if ground < cuts::STILL {
        return (Stance::Combat, Gait::Idle);
    }
    let Some(facing) = t.facing_at(t0 + (t1 - t0) / 2) else {
        return (Stance::Combat, if ground > cuts::WALK { Gait::RunFront } else { Gait::WalkFront });
    };
    let theta = ang_delta((p1[1] - p0[1]).atan2(p1[0] - p0[0]), facing);
    let a = theta.abs();
    if a < cuts::FRONT_ARC {
        if ground >= cuts::SPRINT_MAX {
            (Stance::Combat, Gait::RunFront)
        } else if ground > cuts::SPRINT && a < 0.45 {
            (Stance::Sprint, Gait::RunFront)
        } else if ground > cuts::WALK {
            (Stance::Combat, Gait::RunFront)
        } else {
            (Stance::Combat, Gait::WalkFront)
        }
    } else if a > cuts::BACK_ARC {
        (Stance::Combat, if ground > cuts::WALK { Gait::Back } else { Gait::BackSlow })
    } else if theta > 0.0 {
        (Stance::Combat, Gait::Left)
    } else {
        (Stance::Combat, Gait::Right)
    }
}

/// How far through its stride a body is at `tick`, as a NORMALISED cycle count.
///
/// Integrated along the track, `distance x cycles_per_wu(gait at that moment)`, so:
///
/// * each gait advances at its own correct rate, which is what stops the feet sliding;
/// * the value is CONTINUOUS across a change of gait — the old code computed
///   `(distance * k) % frame_count` against the *new* clip's frame count, so the instant the gait
///   switched the pose jumped to an unrelated point in the cycle and a foot mid-stride snapped
///   somewhere else;
/// * and it stays a pure function of the tick, so scrubbing to the same moment gives the same
///   pose. (A stateful cross-fade would not; see the note on `ActorDriver::sync`.)
///
/// Gaits with no ground travel fall back to the clock, so an idling body breathes instead of
/// standing frozen.
pub fn cycle_phase(t: &crate::film_decode::EntityTrack, tick: u32) -> f32 {
    cycle_phase_with(t, tick, &|_, _| None)
}

/// As [`cycle_phase`], with the caller supplying each gait's REAL clip frame count where it knows
/// it (the driver does, from the clip it loaded). Falls back to the rifle set's authored counts.
/// #film-perf  ONE segment's contribution to the cycle phase, in cycles.
///
/// Lifted out of [`cycle_phase_with`] so the live walk and the prefix-sum cache
/// ([`cycle_phase_prefix`]) compute it with the SAME expression. Two copies of this arithmetic is
/// exactly how a cache comes to disagree with the thing it caches.
///
/// Returns 0.0 for a segment the walk skips (no elapsed time, or a respawn teleport rather than a
/// stride), so a prefix sum over every segment adds the same numbers in the same order as the
/// walk's `cycles += rate` did.
fn segment_rate(
    t: &crate::film_decode::EntityTrack,
    t0: u32,
    p0: [f32; 3],
    tk: u32,
    p: [f32; 3],
    frames_of: &dyn Fn(Stance, Gait) -> Option<f32>,
) -> f32 {
    let dt = (tk.saturating_sub(t0)) as f32 / 60.0;
    if !(dt > 0.0) {
        return 0.0;
    }
    let step = (p[0] - p0[0]).hypot(p[1] - p0[1]);
    // A respawn is a teleport, not a stride.
    if step / dt > 12.0 {
        return 0.0;
    }
    // The gait over THIS segment, computed from the segment itself. Calling the
    // full `derive` here was the quadratic: it re-walked the whole track for its
    // `distance` field and did three binary searches, once per segment.
    let (stance, gait) = segment_gait(t, t0, p0, tk, p, dt, step);
    match frames_of(stance, gait)
        .and_then(|n| cycles_per_wu_for_clip(stance, gait, n))
        .or_else(|| cycles_per_wu(stance, gait))
    {
        // A stride-driven gait: the cycle follows the ground covered.
        Some(r) => r * step,
        // #anim-rate  A clock-driven stretch (idle, a turn on the spot, an
        // airborne arc) still has to advance the phase, or a body that walks,
        // stands still, then walks again resumes mid-stride from before -- and it
        // advances at the clip's OWN AUTHORED RATE: one cycle every
        // `frames / 30` seconds. This used to be a flat `CLIP_FPS * dt / 20.0`,
        // i.e. 1.5 cycles a second whatever was loaded, which is where "the idle
        // animations play way too fast" came from: the shipped Spartan
        // `combat:*:idle` is 156 frames (5.2 s a cycle), so a standing player
        // breathed 7.8x too fast. See [`clock_clip_frames`] for the seed.
        None => {
            let n = frames_of(stance, gait)
                .or_else(|| clock_clip_frames(stance, gait))
                .unwrap_or(CLIP_FPS);
            dt * CLIP_FPS / n.max(1.0)
        }
    }
}

pub fn cycle_phase_with(
    t: &crate::film_decode::EntityTrack,
    tick: u32,
    frames_of: &dyn Fn(Stance, Gait) -> Option<f32>,
) -> f32 {
    let mut cycles = 0.0f32;
    let mut last: Option<(u32, [f32; 3])> = None;
    for &(tk, p) in &t.positions {
        if let Some((t0, p0)) = last {
            if t0 >= tick {
                break;
            }
            let rate = segment_rate(t, t0, p0, tk, p, frames_of);
            let f = if tk <= tick { 1.0 } else { (tick - t0) as f32 / (tk - t0) as f32 };
            cycles += rate * f;
        }
        last = Some((tk, p));
        if tk >= tick {
            break;
        }
    }
    cycles
}

/// #film-perf  The running cycle count at the START of every segment: `pre[i]` is the phase a
/// body reaches after its first `i` segments.
///
/// [`cycle_phase_with`] walks every position sample the track holds BEFORE `tick`, and the
/// Theater frame loop calls it once per drawn body per frame. That makes one frame's cost
/// proportional to how far into the clip the playhead is, which is measurable as the thing it
/// sounds like: on a 7-minute clip, `ActorDriver::sync` cost 9.0 ms a frame at 1:00 and 36.8 ms
/// at 7:00, and 97% of that was this walk's phase. "It gets laggier the longer you watch" is this
/// function.
///
/// The phase is a PREFIX SUM -- every segment but the last contributes its whole rate, and only
/// the segment the tick lands inside is scaled -- so the whole past can be accumulated once and
/// the per-frame answer becomes a binary search plus one segment.
///
/// It is accumulated in the SAME ORDER, with the same additions, as the walk's own
/// `cycles += rate`, so [`cycle_phase_at`] returns a bit-identical `f32` and not an approximation.
/// That matters here more than the speed does: the phase decides the pose, and a pose that
/// depended on how you arrived at a tick would break the rule that a Theater frame is a pure
/// function of the playhead.
pub fn cycle_phase_prefix(
    t: &crate::film_decode::EntityTrack,
    frames_of: &dyn Fn(Stance, Gait) -> Option<f32>,
) -> Vec<f32> {
    let n = t.positions.len();
    let mut pre = Vec::with_capacity(n);
    pre.push(0.0f32);
    let mut cycles = 0.0f32;
    for i in 0..n.saturating_sub(1) {
        let (t0, p0) = t.positions[i];
        let (tk, p) = t.positions[i + 1];
        cycles += segment_rate(t, t0, p0, tk, p, frames_of);
        pre.push(cycles);
    }
    pre
}

/// #film-perf  The cycle phase at `tick`, from a prefix built by [`cycle_phase_prefix`].
///
/// Bit-identical to [`cycle_phase_with`] for the same track, tick and `frames_of`; the gate
/// `the_prefix_phase_is_bit_identical_to_the_walk` asserts exactly that over a track with
/// teleports, repeated ticks and gait changes in it. `pre` that does not match the track's sample
/// count is refused (returns `None`) rather than trusted, so a stale cache cannot pose a body.
pub fn cycle_phase_at(
    t: &crate::film_decode::EntityTrack,
    tick: u32,
    pre: &[f32],
    frames_of: &dyn Fn(Stance, Gait) -> Option<f32>,
) -> Option<f32> {
    let n = t.positions.len();
    if pre.len() != n || n == 0 {
        return None;
    }
    // The walk processes segment `i` (positions[i] -> positions[i+1]) exactly when
    // `positions[i].0 < tick`, and gives it the whole rate when `positions[i+1].0 <= tick`.
    // `m2` counts the samples at or before the tick, so the segments that get their whole rate
    // are `0..m2-1`, and at most ONE segment straddles the tick.
    let m2 = t.positions.partition_point(|(tk, _)| *tk <= tick);
    let k = m2.saturating_sub(1).min(n.saturating_sub(1));
    let mut cycles = pre[k];
    if k + 1 < n {
        let (t0, p0) = t.positions[k];
        let (t1, p1) = t.positions[k + 1];
        if t0 < tick && t1 > tick {
            let rate = segment_rate(t, t0, p0, t1, p1, frames_of);
            cycles += rate * ((tick - t0) as f32 / (t1 - t0) as f32);
        }
    }
    Some(cycles)
}

/// A short, human label for the state — what the per-player status line shows.
pub fn describe(st: &BodyState) -> String {
    // #film-action  The one state here that is decoded rather than inferred, so it says so, and
    // it is explicit about the part that still is not: WHICH move.
    if let Some(sy) = &st.sync {
        let set = sync_move_set(sy.attacker_elite, sy.victim_elite);
        let name = set[sy.move_index % set.len()];
        return format!(
            "{} an assassination ({name}, {}) \u{2014} {:.1}s in; {}",
            match sy.side {
                SyncSide::Attacker => "performing",
                SyncSide::Victim => "being killed by",
            },
            if sy.move_decoded { "the move is decoded too" } else { "the move is chosen: this film carries the broadcast, not the host's own record" },
            sy.age as f32 / 60.0,
            if sy.stated {
                format!("the recording STATES the kill, both players and the tick ({})", sy.incident)
            } else {
                format!(
                    "DERIVED, not stated: a film carries only the RECORDER's own kill feed, so this kill \
                     has no incident row. It is {} \u{2014} the victim's last `damage_aftermath` names \
                     this killer, and the two bodies were inside each other ({:.1} wu apart or \
                     less) at the moment it died, which only happens when the engine has locked \
                     them into a synchronised action",
                    sy.incident,
                    sync_infer::NEAR_WU
                )
            }
        );
    }
    if st.stance == Stance::Dead {
        let near = match st.death_company {
            Some(c) => match c.from_behind {
                Some(b) if b <= 70.0 => format!(
                    "; another player was {:.1} wu away and {:.0} deg off their back at that instant (geometry only — this body's own death carries no recorded killer)",
                    c.distance, b
                ),
                Some(b) => format!("; another player was {:.1} wu away, {:.0} deg off their back", c.distance, b),
                None => format!("; another player was {:.1} wu away (no facing, so no direction)", c.distance),
            },
            None => String::new(),
        };
        return format!(
            "dead {:.1}s ({} — the recording does not state which ragdoll played){near}",
            st.dead_ticks as f32 / 60.0,
            if st.death_airborne {
                "killed off the ground, so the airborne flail"
            } else {
                "a ground landing"
            }
        );
    }
    if let Some(s) = &st.seat {
        return format!("in a {} ({})", s.vehicle, s.mode);
    }
    let g = match st.gait {
        Gait::Idle => "still",
        Gait::TurnLeft => "turning left",
        Gait::TurnRight => "turning right",
        Gait::WalkFront => "walking",
        Gait::RunFront => "running",
        Gait::BackSlow => "stepping back",
        Gait::Back => "backing off",
        Gait::Left => "strafing left",
        Gait::Right => "strafing right",
    };
    let base = match st.stance {
        Stance::Flight => format!("airborne ({:+.1} wu/s)", st.vz),
        Stance::Sprint => format!("sprinting at {:.1} wu/s", st.ground_speed),
        Stance::Crouch => format!("crouched, {g}"),
        _ => format!("{g} at {:.1} wu/s", st.ground_speed),
    };
    // #wclass  Which animation SET the clip came from, and whether that was read or assumed.
    match &st.weapon_class {
        Some(c) => format!("{base}, {c} set (from the weapon the recording names)"),
        None => format!("{base}, {DEFAULT_WEAPON_CLASS} set (no weapon named for this body)"),
    }
}


// ---------------------------------------------------------------------------------------------
// vehicles: which entity is one, which vehicle it is, and who is riding it
// ---------------------------------------------------------------------------------------------

/// The engine entity type that is a VEHICLE.
///
/// Proven two independent ways over 7-14 real films, neither of which needs the engine:
///
/// 1. **The ride oracle.** A biped's position becomes coincident with another entity's for a
///    sustained run of ticks and then travels with it. 16 such rides were found across 4 films,
///    totalling 243 wu travelled together, including one of 15.7 s over 60 wu — and **every one
///    of them is on a type-15 entity**. The only other type that scored (13, twice) was a player
///    standing on a slowly-moving Forge platform for 32 s while it travelled 12 wu.
/// 2. **The spawn-point oracle.** 97% of type-15 entities first appear within 1.5 wu of an object
///    in the film's own embedded map variant, and all 17 distinct palette objects they land on
///    are `vehi` tags: falcon, mongoose, pickup, scorpion, warthog, oni_van, sabre, shade,
///    banshee, machinegun, wraith, cart_electric, forklift, ghost and three more.
///
/// For the record, the same spawn-point oracle says type 17 is the WEAPON (it lands on
/// `weap rocket_launcher`, `weap sniper_rifle`, `weap gravity_hammer` and on respawn points,
/// where a player's loadout weapon is created) and type 12 the EQUIPMENT (`eqip jet_pack`,
/// `eqip evade`, `eqip armor_lockup`, `eqip active_camouflage`, `eqip drop_shield`, grenades).
/// Neither is drawn yet.
pub const VEHICLE_TYPE: u8 = 15;
/// The engine entity type that is a unit/biped.
pub const BIPED_TYPE: u8 = 11;

/// How near a spawned vehicle must be to a VEHICLE spawn point in the film's own embedded map
/// variant to be called that vehicle, in wu.
///
/// Measured: 97% of type-15 entities land within 1.5 wu of *some* object and the identified ones
/// typically match to 0.00-0.02 wu, so this is far looser than it needs to be for the common
/// case. It is loose on purpose, because the match is constrained to be a `vehi` object: a
/// vehicle that has already rolled a little off its spawn before its first position is replicated
/// still resolves, and `VehicleIdent::spawn_error` carries the real distance so a loose match is
/// visible in the UI rather than hidden.
pub const SPAWN_MATCH_WU: f32 = 4.0;

/// One identified vehicle in a film.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleIdent {
    pub net_id: u32,
    /// The `vehi` object tag, which is what the renderer needs.
    pub obj_tag: u32,
    /// Tag leaf name — `warthog`, `mongoose`, `ghost`, ...
    pub name: String,
    /// The jmad seat-mode family for a rider, or empty when this vehicle has no authored seat
    /// animations in the Spartan graph (the Sabre, for one).
    pub seat_family: String,
    /// How far its first position was from the Forge object it was matched to, wu.
    pub spawn_error: f32,
    /// The Forge spawn point it was matched to, in world units.
    pub spawn_pos: [f32; 3],
}

/// Vehicle tag leaf name -> the jmad seat-mode family, read off the shipped Spartan graph's own
/// 1393 animation names (`warthog_d:any:idle`, `mongoose_p:rifle:idle`, `ghost_d:idle`, ...).
///
/// Longest match first, so `warthog` does not shadow a hypothetical `warthog_rocket`.
pub const SEAT_FAMILIES: &[(&str, &str)] = &[
    ("warthog", "warthog"),
    ("mongoose", "mongoose"),
    ("revenant", "revenant"),
    ("scorpion", "scorpion"),
    ("banshee", "banshee"),
    ("falcon", "falcon"),
    ("wraith", "wraith"),
    ("ghost", "ghost"),
    ("shade", "shade"),
    // The civilian vehicles all share the `truck_*` seat set.
    ("pickup", "truck"),
    ("oni_van", "truck"),
    ("truck", "truck"),
    ("cart_electric", "truck"),
    ("forklift", "truck"),
    ("machinegun", "turret_g"),
    ("turret", "turret_g"),
];

/// The seat family for a vehicle tag path, or `""` when the graph has no seat set for it.
pub fn seat_family_for(tag_path: &str) -> String {
    let p = tag_path.to_lowercase();
    for (needle, fam) in SEAT_FAMILIES {
        if p.contains(needle) {
            return (*fam).to_string();
        }
    }
    String::new()
}

/// One stretch of one biped riding one vehicle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ride {
    pub biped_net_id: u32,
    pub vehicle_net_id: u32,
    pub from_tick: u32,
    pub to_tick: u32,
}

impl Ride {
    pub fn covers(&self, tick: u32) -> bool {
        tick >= self.from_tick && tick <= self.to_tick
    }
    pub fn seconds(&self) -> f32 {
        (self.to_tick - self.from_tick) as f32 / 60.0
    }
}

/// Ride-detector constants, all measured (`probes::probe_ride_distance_calibration`).
pub mod ride {
    /// Window the displacement match is computed over, ticks.
    pub const WINDOW: u32 = 30;
    /// Sampling step, ticks. 10 Hz catches a half-second ride and costs nothing.
    pub const STEP: u32 = 6;
    /// How far the vehicle must travel HORIZONTALLY across a window for the match to mean
    /// anything, wu. Horizontal is the operative word: vertical motion is shared by everything in
    /// free fall, so a window in which the vehicle only moved in z is no evidence at all.
    pub const MIN_MOVE: f32 = 0.5;
    /// How closely the biped's HORIZONTAL displacement must match the vehicle's, wu. Equivalently
    /// (the algebra is the same thing) how much the body's offset from the vehicle is allowed to
    /// change across the window. A rider is bolted to a seat, so its offset is constant; two
    /// players running side by side never hold it.
    pub const MATCH_WU: f32 = 0.18;
    /// Distance bound. A rider's origin sits 0.62-1.55 wu from the vehicle origin (p05-p95 over
    /// 5,152 measured ride samples), because the biped's origin is at its feet and the vehicle's
    /// is at its centre — so this cannot be tight, which is exactly WHY the detector keys on
    /// matched displacement and uses distance only as a sanity bound. Non-riders within 4 wu sit
    /// at p25 1.92 / p50 2.82, so the two distributions overlap and distance alone would be
    /// guesswork.
    pub const MAX_WU: f32 = 2.5;
    /// A confirmed ride is extended through a stationary stretch for up to this long, so parking
    /// the Warthog does not eject the driver.
    pub const HOLD_TICKS: u32 = 240;
}

/// Find every ride in a film: matched displacement between a biped and a vehicle.
///
/// The detector does NOT use proximity as its evidence, because the measured distributions
/// overlap (see `ride::MAX_WU`). It uses the fact that a rider's displacement over a window
/// *equals* the vehicle's, which nothing else in the game does.
pub fn find_rides(idx: &crate::film_decode::FilmIndex) -> Vec<Ride> {
    let veh: Vec<usize> = (0..idx.tracks.len())
        .filter(|&i| idx.tracks[i].entity_type == VEHICLE_TYPE && idx.tracks[i].positions.len() > 4)
        .collect();
    if veh.is_empty() {
        return Vec::new();
    }
    let bip: Vec<usize> = (0..idx.tracks.len())
        .filter(|&i| idx.tracks[i].entity_type == BIPED_TYPE && idx.tracks[i].positions.len() > 4)
        .collect();
    let mut out: Vec<Ride> = Vec::new();
    for &vi in &veh {
        let v = &idx.tracks[vi];
        for &bi in &bip {
            let b = &idx.tracks[bi];
            let lo = b.first_tick.max(v.first_tick);
            let hi = b.last_tick.min(v.destroyed_tick.unwrap_or(u32::MAX));
            if hi <= lo + ride::WINDOW {
                continue;
            }
            let mut run: Option<(u32, u32)> = None; // (start, last confirmed)
            let mut t = lo;
            while t + ride::WINDOW <= hi {
                let ok = (|| {
                    let (b0, b1) = (b.sample(t)?, b.sample(t + ride::WINDOW)?);
                    let (v0, v1) = (v.sample(t)?, v.sample(t + ride::WINDOW)?);
                    let d = ((b0[0] - v0[0]).powi(2) + (b0[1] - v0[1]).powi(2) + (b0[2] - v0[2]).powi(2)).sqrt();
                    if d > ride::MAX_WU {
                        return Some(false);
                    }
                    // HORIZONTAL displacement only. Vertical agreement carries no information,
                    // because GRAVITY gives any two falling objects the same vertical
                    // displacement -- and that is not hypothetical: on one real clip both
                    // reported "rides" were a body and a Scorpion falling past each other, 1.4 wu
                    // apart, the vehicle's x,y frozen for the whole window and only z agreeing,
                    // ending the instant the co-fall ended (hence both lasting exactly 0.5 s).
                    let dv = [v1[0] - v0[0], v1[1] - v0[1]];
                    let mv = (dv[0] * dv[0] + dv[1] * dv[1]).sqrt();
                    if mv < ride::MIN_MOVE {
                        // The vehicle did not travel horizontally, so nothing about this window
                        // distinguishes a rider from a bystander. Hold an existing ride, start
                        // none. (A vehicle in free fall reaches here, which is the fix.)
                        return None;
                    }
                    let db = [b1[0] - b0[0], b1[1] - b0[1]];
                    // This is the change in the body's offset FROM the vehicle: a rider is bolted
                    // to a seat, so its offset is constant and this is ~0.
                    let err = ((db[0] - dv[0]).powi(2) + (db[1] - dv[1]).powi(2)).sqrt();
                    Some(err < ride::MATCH_WU)
                })();
                match ok {
                    Some(true) => {
                        let s = run.map(|r| r.0).unwrap_or(t);
                        run = Some((s, t + ride::WINDOW));
                    }
                    Some(false) => {
                        if let Some((s, e)) = run.take() {
                            out.push(Ride { biped_net_id: b.net_id, vehicle_net_id: v.net_id, from_tick: s, to_tick: e });
                        }
                    }
                    None => {
                        // stationary: hold, but not forever
                        if let Some((s, e)) = run {
                            if t > e + ride::HOLD_TICKS {
                                out.push(Ride { biped_net_id: b.net_id, vehicle_net_id: v.net_id, from_tick: s, to_tick: e });
                                run = None;
                            }
                        }
                    }
                }
                t += ride::STEP;
            }
            if let Some((s, e)) = run {
                out.push(Ride { biped_net_id: b.net_id, vehicle_net_id: v.net_id, from_tick: s, to_tick: e });
            }
        }
    }
    out.sort_by_key(|r| (r.biped_net_id, r.from_tick));
    out
}

/// Join up ride fragments. The displacement test goes quiet whenever the vehicle stops — at a
/// corner, at a wall, waiting for a passenger — so one continuous drive arrives as a run of short
/// stretches (measured: one 8.5-second drive came back as seven fragments of 0.5-1.7 s). Any two
/// stretches of the same body in the same vehicle closer together than `MERGE_GAP` are one ride.
pub const MERGE_GAP: u32 = 180; // 3 seconds

pub fn merge_rides(mut v: Vec<Ride>) -> Vec<Ride> {
    v.sort_by_key(|r| (r.biped_net_id, r.vehicle_net_id, r.from_tick));
    let mut out: Vec<Ride> = Vec::with_capacity(v.len());
    for r in v {
        match out.last_mut() {
            Some(p)
                if p.biped_net_id == r.biped_net_id
                    && p.vehicle_net_id == r.vehicle_net_id
                    && r.from_tick <= p.to_tick + MERGE_GAP =>
            {
                p.to_tick = p.to_tick.max(r.to_tick);
            }
            _ => out.push(r),
        }
    }
    out.sort_by_key(|r| (r.from_tick, r.biped_net_id));
    out
}

/// How far the vehicle must have carried the body HORIZONTALLY, over the whole merged ride, for
/// the ride to be accepted, in wu.
///
/// The window test alone admits half-second matches in which the vehicle moved less than a world
/// unit. Those are not worth drawing and they are not distinguishable from a body that happened
/// to drift alongside for half a second, so they are refused. The cost is honest and worth
/// stating: a genuine hop-in-hop-out that never went anywhere is refused too. Real drives are
/// nowhere near this bound — the longest in the corpus carries its rider 203 wu.
pub const MIN_RIDE_TRAVEL_WU: f32 = 2.0;

/// Every ride in a film, merged and accepted. This is the list the UI shows and the seat lookup
/// uses; [`find_rides`] and [`merge_rides`] are the stages, exposed for measurement.
pub fn accepted_rides(idx: &crate::film_decode::FilmIndex) -> Vec<Ride> {
    let mut out = merge_rides(find_rides(idx));
    out.retain(|r| {
        let Some(v) = idx.tracks.iter().find(|t| t.net_id == r.vehicle_net_id) else { return false };
        let mut flat = 0.0f32;
        let mut t = r.from_tick;
        while t + ride::STEP <= r.to_tick {
            if let (Some(a), Some(b)) = (v.sample(t), v.sample(t + ride::STEP)) {
                flat += (b[0] - a[0]).hypot(b[1] - a[1]);
            }
            t += ride::STEP;
        }
        flat >= MIN_RIDE_TRAVEL_WU
    });
    out
}

// ---------------------------------------------------------------------------------------------
// assassinations, derived from the motion
// ---------------------------------------------------------------------------------------------

/// One body locked to another for long enough to be an assassination, ending in a death.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncPair {
    pub attacker_net_id: u32,
    pub victim_net_id: u32,
    pub from_tick: u32,
    pub to_tick: u32,
}

impl SyncPair {
    pub fn covers(&self, tick: u32) -> bool {
        tick >= self.from_tick && tick <= self.to_tick
    }
    pub fn seconds(&self) -> f32 {
        (self.to_tick - self.from_tick) as f32 / 60.0
    }
}

/// Constants for the paired-melee detector.
///
/// **The event is not on the wire.** Field 0x36, the engine's named action enum that carries
/// `_action_assassinate`, never appears in a film: 0 occurrences in 5,438,363 type-11 messages.
/// Field 0x35, the assassination record itself, IS present 4,706 times across 14 films but is
/// always the CLEARED record — null target, sentinel position, not one live target, including on
/// the clip whose gametype is called "Assassination." So anything drawn here is **derived from
/// the motion**, never a recorded event, and it is labelled that way.
///
/// The signature: an assassination locks two bodies together for roughly 2-3 seconds at a fixed
/// short offset, and the victim's track ends.
pub mod sync_action {
    /// How close the two bodies must be, in wu. Far tighter than a vehicle ride: the animation
    /// puts the attacker against the victim's back.
    pub const MAX_WU: f32 = 1.1;
    /// How steady the relative offset must be. This is the discriminator that separates a locked
    /// pair from two players who merely happen to be standing next to each other: during the
    /// animation the offset between them barely changes, whereas two independent bodies drift.
    pub const OFFSET_DRIFT_WU: f32 = 0.30;
    /// Shortest run that counts, ticks.
    pub const MIN_TICKS: u32 = 72; // 1.2 s
    /// Longest run that counts, ticks — past this it is two people standing together, not an
    /// animation.
    pub const MAX_TICKS: u32 = 300; // 5 s
    /// The victim's track must end within this many ticks of the run ending.
    pub const DEATH_WINDOW: u32 = 60;
    pub const STEP: u32 = 6;
}

/// Find paired melee (assassination) events by their MOTION, since the event itself is not
/// recorded.
///
/// # NOTHING MAY CONSUME THIS. It does not work, and the measurement says why.
///
/// The idea was sound and it is the same shape as the ride detector, which does work: two bodies
/// held at a fixed short offset for a couple of seconds, ending in a death. It was validated the
/// only way that means anything — against clips whose gametype is literally **"Assassination."**,
/// a custom game built around them, versus ordinary social and Castle Wars clips. A working
/// detector would be far richer on the former.
///
/// **It is the other way round.** Over 14 films:
///
/// > assassination-gametype clips: **0.00 pairs per minute** (3 films).
/// > everything else: **0.11 pairs per minute** (11 films).
///
/// And the data is not the limitation: the assassination clip carries 111 bodies at 2,243
/// positions each, bodies approach to 0.00 wu, and there are 471 runs of two bodies held inside
/// 1.1 wu. The gate census (`probes::probe_sync_pair_gates`) shows where they go — 471 runs, 141
/// of a plausible length, **6** surviving the offset-drift test, **0** surviving the death test.
///
/// The drift distribution is the heart of it. Two bodies within 1.1 wu of each other drift apart
/// and back by a median of **0.93 wu** (p90 1.21) over the run — as much as the proximity bound
/// itself — and that is true on the assassination clips (0.93, 1.01, 0.81) exactly as it is on
/// the others (0.86, 0.78, 0.51, 0.89). There is no "locked together" to find: a biped's position
/// is replicated sparsely enough that the interpolated offset between two of them wanders by
/// about a world unit no matter what the players are doing. Nor does the raw rate of close
/// encounters separate the clips — 65 runs/min on the assassination clip against 66 on Castle
/// Wars.
///
/// So **this detector draws no assassinations**. Inventing kills that did not happen would be
/// worse than drawing none, and loosening the drift bound far enough to catch anything on the
/// assassination clips would admit hundreds of ordinary close-quarters encounters on every other
/// clip.
///
/// **SUPERSEDED 2026-10-11 (#film-action).** What would have unblocked it was "a per-tick signal
/// that the bodies are in a synchronised action at all", and the film carries one after all —
/// just not in the unit's dirty block, where this note went looking. List B command **kind 1** is
/// the engine's own 72-type event channel, and its `incident` event broadcasts the kill with the
/// medal's own engine string id and both parties' object references. So an assassination is now
/// **decoded**: see [`assassinations`]. This function is kept only as the apparatus that produced
/// the numbers above, so the next person does not re-run a failed experiment.
pub fn find_sync_pairs(idx: &crate::film_decode::FilmIndex) -> Vec<SyncPair> {
    let bip: Vec<usize> = (0..idx.tracks.len())
        .filter(|&i| idx.tracks[i].entity_type == BIPED_TYPE && idx.tracks[i].positions.len() > 4)
        .collect();
    let mut out: Vec<SyncPair> = Vec::new();
    for (ai, &i) in bip.iter().enumerate() {
        for &j in bip.iter().skip(ai + 1) {
            let (a, b) = (&idx.tracks[i], &idx.tracks[j]);
            // Two bodies of the SAME player are respawns, never a pair.
            if a.owner_raw.is_some() && a.owner_raw == b.owner_raw {
                continue;
            }
            let lo = a.first_tick.max(b.first_tick);
            let hi = a.last_tick.min(b.last_tick);
            if hi <= lo + sync_action::MIN_TICKS {
                continue;
            }
            let mut run: Option<u32> = None;
            let mut t = lo;
            let mut offs: Vec<[f32; 3]> = Vec::new();
            while t <= hi {
                let near = match (a.sample(t), b.sample(t)) {
                    (Some(p), Some(q)) => {
                        let o = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                        let d = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
                        if d <= sync_action::MAX_WU {
                            offs.push(o);
                            true
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if near {
                    run.get_or_insert(t);
                } else if let Some(s) = run.take() {
                    close_pair(&mut out, idx, a, b, s, t.saturating_sub(sync_action::STEP), &offs);
                    offs.clear();
                }
                t += sync_action::STEP;
            }
            if let Some(s) = run {
                close_pair(&mut out, idx, a, b, s, hi, &offs);
            }
        }
    }
    out.sort_by_key(|p| (p.from_tick, p.attacker_net_id));
    out
}

fn close_pair(
    out: &mut Vec<SyncPair>,
    idx: &crate::film_decode::FilmIndex,
    a: &crate::film_decode::EntityTrack,
    b: &crate::film_decode::EntityTrack,
    from: u32,
    to: u32,
    offs: &[[f32; 3]],
) {
    let len = to.saturating_sub(from);
    if len < sync_action::MIN_TICKS || len > sync_action::MAX_TICKS || offs.len() < 4 {
        return;
    }
    // How much did the offset between the two bodies move over the run? A locked pair barely
    // changes; two bodies that merely share a corner of the map drift apart and back.
    let n = offs.len() as f32;
    let mean = [
        offs.iter().map(|o| o[0]).sum::<f32>() / n,
        offs.iter().map(|o| o[1]).sum::<f32>() / n,
        offs.iter().map(|o| o[2]).sum::<f32>() / n,
    ];
    let drift = offs
        .iter()
        .map(|o| ((o[0] - mean[0]).powi(2) + (o[1] - mean[1]).powi(2) + (o[2] - mean[2]).powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    if drift > sync_action::OFFSET_DRIFT_WU {
        return;
    }
    // One of them must DIE as the run ends. That is the half of the signature a ride does not
    // have, and it is what keeps two team-mates standing shoulder to shoulder out of the list.
    let ends = |t: &crate::film_decode::EntityTrack| -> bool {
        let end = t.destroyed_tick.unwrap_or(t.last_tick);
        end >= from && end <= to + sync_action::DEATH_WINDOW && end + sync_action::DEATH_WINDOW < idx.ticks
    };
    let (attacker, victim) = match (ends(a), ends(b)) {
        (false, true) => (a, b),
        (true, false) => (b, a),
        // Both or neither: not a clean kill, so do not claim one.
        _ => return,
    };
    out.push(SyncPair {
        attacker_net_id: attacker.net_id,
        victim_net_id: victim.net_id,
        from_tick: from,
        to_tick: to,
    });
}

/// Every vehicle in a film, identified, plus every ride. Built once when a film is opened.
#[derive(Clone, Debug, Default)]
pub struct VehicleTable {
    pub vehicles: std::collections::HashMap<u32, VehicleIdent>,
    pub rides: Vec<Ride>,
    /// Type-15 entities that never replicated a position.
    ///
    /// Measured, and NOT a decode failure: these live for minutes and send hundreds to thousands
    /// of messages, but the union of every dirty mask they ever send excludes bit 0 entirely
    /// (typical union `{4, 6, 8, 11, 12, 13, 21, 23}`). They are the vehicles nobody ever
    /// touched — a vehicle that does not move has no position to replicate — and they are
    /// **already drawn**, as part of the film's own embedded Forge scene, sitting on the spawn
    /// point they never left. On one 20-minute social clip 73 of 163 vehicles are in this
    /// category; on a busy vehicle film it is 2 of 209.
    pub no_position: usize,
    /// Type-15 entities that HAVE a position but no vehicle spawn point within
    /// [`SPAWN_MATCH_WU`] of it, so their model is genuinely unknown.
    pub unidentified: usize,
    /// Every type-15 entity in the clip. `total == vehicles.len() + never_moved + unidentified`,
    /// exactly — the three are disjoint, which is what lets the UI print figures that add up.
    pub total: usize,
}

impl VehicleTable {
    /// `resolve` maps a `(quota index, variant item)` to `(obj_tag, tag path)` through the
    /// scenario sandbox palette, returning `None` for anything that is not a `vehi` — the caller
    /// supplies it because it needs the loaded cache.
    ///
    /// The search is for the nearest **vehicle** spawn point, not the nearest object. That
    /// matters: a measured case had a ridden Warthog whose nearest object of any kind was 1.96 wu
    /// away (a respawn point it had been parked beside), while its own vehicle spawn was further
    /// off. Requiring the match to BE a vehicle is what makes a looser radius safe, and the
    /// distance is carried into `spawn_error` and printed, so a loose match is visible.
    pub fn build(
        idx: &crate::film_decode::FilmIndex,
        mut resolve: impl FnMut(u16, u8) -> Option<(u32, String)>,
    ) -> VehicleTable {
        let mut t = VehicleTable { rides: accepted_rides(idx), ..Default::default() };
        // Resolve the variant's vehicle spawns ONCE; the palette walk is not free.
        let spawns: Vec<([f32; 3], u32, String)> = idx
            .variant_objects
            .iter()
            .filter_map(|&(q, it, pos)| resolve(q, it).map(|(tag, path)| (pos, tag, path)))
            .collect();
        t.total = idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE).count();
        for tr in idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE) {
            let Some(&(_, first)) = tr.positions.first() else {
                t.no_position += 1;
                continue;
            };
            let mut best: Option<(f32, u32, &String, &[f32; 3])> = None;
            for (pos, tag, path) in &spawns {
                let d = (pos[0] - first[0]).powi(2) + (pos[1] - first[1]).powi(2) + (pos[2] - first[2]).powi(2);
                if best.map_or(true, |b| d < b.0) {
                    best = Some((d, *tag, path, pos));
                }
            }
            match best.filter(|b| b.0.sqrt() <= SPAWN_MATCH_WU) {
                Some((d2, obj_tag, path, pos)) => {
                    let leaf = path.rsplit('\\').next().unwrap_or(path).to_string();
                    t.vehicles.insert(
                        tr.net_id,
                        VehicleIdent {
                            net_id: tr.net_id,
                            obj_tag,
                            seat_family: seat_family_for(path),
                            name: leaf,
                            spawn_error: d2.sqrt(),
                            spawn_pos: *pos,
                        },
                    );
                }
                None => t.unidentified += 1,
            }
        }
        t
    }

    /// The seat a biped occupies at `tick`, ready to hand to [`derive`].
    pub fn seat_of(&self, biped_net_id: u32, tick: u32) -> Option<Seat> {
        let r = self
            .rides
            .iter()
            .find(|r| r.biped_net_id == biped_net_id && r.covers(tick))?;
        let v = self.vehicles.get(&r.vehicle_net_id)?;
        if v.seat_family.is_empty() {
            return None;
        }
        // Which seat? The driver's, unless another body is already riding and started earlier —
        // the stream does not carry a seat index, so this is ordering, not decoding, and the UI
        // says so.
        let earlier = self
            .rides
            .iter()
            .filter(|o| o.vehicle_net_id == r.vehicle_net_id && o.covers(tick) && o.biped_net_id != biped_net_id)
            .filter(|o| (o.from_tick, o.biped_net_id) < (r.from_tick, r.biped_net_id))
            .count();
        let suffix = match earlier {
            0 => "_d",
            1 => "_g",
            _ => "_p",
        };
        Some(Seat {
            vehicle_net_id: r.vehicle_net_id,
            mode: format!("{}{suffix}", v.seat_family),
            vehicle: v.name.clone(),
        })
    }

    /// How close to its own Forge spawn point a vehicle has to be for the film's Forge scene to
    /// be drawing it already, in wu.
    ///
    /// The film's embedded map variant is drawn in full, vehicles included — on one clip the
    /// scene carries 14 of them, and `mongoose` sits at (-1.9, -18.8, 2.7) permanently. A stream
    /// vehicle that has not moved matches its spawn to 0.01 wu, so drawing it as well puts two
    /// identical models in exactly the same place and they z-fight. While it is parked there the
    /// Forge scene's copy is the right thing to show; the moment it moves, the stream's copy is.
    ///
    /// The cost, stated: a vehicle ROTATED on the spot without being driven anywhere keeps the
    /// authored orientation until it moves.
    pub const SPAWN_COINCIDENT_WU: f32 = 0.25;

    /// Is this vehicle still sitting on the spawn point the Forge scene already draws it on?
    pub fn on_its_spawn(v: &VehicleIdent, pos: [f32; 3]) -> bool {
        let d = (pos[0] - v.spawn_pos[0]).powi(2)
            + (pos[1] - v.spawn_pos[1]).powi(2)
            + (pos[2] - v.spawn_pos[2]).powi(2);
        d.sqrt() <= Self::SPAWN_COINCIDENT_WU
    }

    /// Vehicles alive at `tick` that the STREAM has to draw — i.e. everything except the ones
    /// still parked on the spawn point the Forge scene is already drawing them on.
    pub fn to_draw_at<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
    ) -> Vec<(&'a VehicleIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        self.to_draw_at_with(idx, tick, &|_| false)
    }

    /// #vehicle-damage  As [`VehicleTable::to_draw_at`], with the WRECK exception.
    ///
    /// The spawn-coincident skip exists so the stream's copy does not z-fight the Forge scene's,
    /// and while the vehicle is intact the Forge scene's copy is the right thing to show. A
    /// DESTROYED one is different: the Forge scene draws it pristine, so a wreck that came to
    /// rest on (or respawned onto) its own spawn point has to be drawn from the stream anyway.
    /// `wreck(net_id)` is the caller's own destroyed-state test — the table itself cannot know,
    /// because that needs the object's `hlmt` damage responses.
    pub fn to_draw_at_with<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
        wreck: &dyn Fn(u32) -> bool,
    ) -> Vec<(&'a VehicleIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        self.alive_at(idx, tick)
            .into_iter()
            .filter(|(v, pos, _)| !Self::on_its_spawn(v, *pos) || wreck(v.net_id))
            .collect()
    }

    /// How many alive vehicles are parked on their own spawn point at `tick` (so the Forge scene
    /// draws them and the stream does not).
    pub fn parked_on_spawn_at(&self, idx: &crate::film_decode::FilmIndex, tick: u32) -> usize {
        self.alive_at(idx, tick).iter().filter(|(v, pos, _)| Self::on_its_spawn(v, *pos)).count()
    }

    /// Vehicles alive at `tick`, for drawing.
    pub fn alive_at<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
    ) -> Vec<(&'a VehicleIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        let mut out = Vec::new();
        for tr in idx.tracks.iter().filter(|t| t.entity_type == VEHICLE_TYPE) {
            let Some(v) = self.vehicles.get(&tr.net_id) else { continue };
            if !tr.alive_at(tick) {
                continue;
            }
            let Some(p) = tr.sample(tick) else { continue };
            // #vehicle-rot  The INTERPOLATED basis, not the held one: base field 2 is a full
            // orthonormal (forward, up) pair and a vehicle reports it often but not every tick,
            // so slerping between the bracketing keys is what makes a tumble read as a tumble.
            out.push((v, p, tr.orientation_at(tick)));
        }
        out
    }
}


// ---------------------------------------------------------------------------------------------
// the biped's DERIVED dirty block — combat events
// ---------------------------------------------------------------------------------------------

/// Decoding a biped's derived (unit) dirty block, fields 0x14..0x3B.
///
/// **Read off `haloreach.dll`, not searched for.** Entity type 11's dirty reader is
/// `sub_1803B4B88`; it calls the field iterator `sub_18043442C` with `first = 0x14`,
/// `count = 0x28` and the gate list `{0x15, 0x1A, 0x1F, 0x24, 0x29, 0x2E}` (the literal at
/// `0x180A48500`). Field 0x14 is the unit's 12-bit yaw + 11-bit pitch aiming vector; 0x15..0x28
/// are four identical five-field weapon-slot blocks; 0x29 the weapon selection; 0x2A..0x2D the
/// four `'weap'` tag references; 0x2E the two grenade counts; 0x2F the controlling player;
/// **0x35 the assassination / synchronised-melee record**; 0x36 a 7-bit object-action enum whose
/// names are string literals in the binary (`_action_assassinate`, `_action_vehicle_enter`,
/// `_action_jump`…); 0x37 the 0x90-byte control-input record.
///
/// ## What is actually in a film, measured
///
/// Over **5,438,363 type-11 messages across 8 films**, the derived bits that ever appear are:
///
/// | bit | field | messages | share |
/// |-----|-------|----------|-------|
/// | 0x14 | aim yaw + pitch | 2,882,858 | 53.01% |
/// | 0x29 | weapon selection | 10,329 | 0.190% |
/// | 0x2E | grenade counts | 7,412 | 0.136% |
/// | 0x2A | weapon slot 0 `'weap'` tag | 6,945 | 0.128% |
/// | 0x2F | controlling player | 6,215 | 0.114% |
/// | **0x35** | **assassination** | **4,175** | **0.077%** |
/// | 0x33 | flashlight | 3,558 | 0.065% |
/// | 0x32 | equipment object reference | 3,253 | 0.060% |
/// | 0x31 | equipment / ability mode | 140 | 0.003% |
///
/// and, as a **clean negative worth recording: bits 0x36, 0x37, 0x38, 0x39, 0x3A and 0x3B never
/// appear at all — not once in 5.4 million messages.** So the named action enum (and with it
/// `_action_vehicle_enter`, `_action_jump`, `_action_land`, `_action_melee_ai`) and the control
/// input record are NOT in an entity-replication film, and no amount of decoding will produce
/// them. Vehicle entry and exit have to be derived from the geometry, which is what
/// [`find_rides`] does.
pub mod unit {
    use crate::film_decode::{BitReader, DecodeParams, DecodeStop, EntityMessage};

    /// The biped reader's gate list, read off `0x180A48500`.
    pub const GATES: [u32; 6] = [0x15, 0x1A, 0x1F, 0x24, 0x29, 0x2E];
    /// First and last derived field index (`first = 0x14`, `count = 0x28`).
    pub const FIRST: u32 = 0x14;
    pub const COUNT: u32 = 0x28;

    /// The assassination / synchronised-melee record, field 0x35.
    ///
    /// The engine applies it with
    /// `PlayerInteraction_ExecuteAssassinationAction(player, target_object, …, position,
    /// direction, 0)`, and that function stamps `4` — `_action_assassinate` — as the action type.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Assassination {
        /// The object being assassinated, as a 27-bit object reference (`None` = the null
        /// reference, which is how the record is cleared at the end of the action).
        pub target: Option<u32>,
        /// Where it happens, as the three RAW quantised axis values — the same
        /// `compressed_vector3(bits = 0x10)` encoding base field 0 uses, so the same
        /// `PositionFrame` turns it into world units.
        pub position_raw: Option<[u32; 3]>,
        /// A 32-bit field the engine passes through; meaning unknown.
        pub stamp: u32,
    }

    /// Everything the derived block yielded.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct UnitState {
        /// Field 0x14: 12-bit yaw over a full turn. `yaw = (raw + 0.5) * TAU / 4096`.
        pub aim_yaw_q: Option<u16>,
        /// Field 0x14: 11-bit pitch. `pitch = -PI + (raw + 0.5) * TAU / 2048`.
        pub aim_pitch_q: Option<u16>,
        /// Field 0x2A: weapon slot 0's index into the map's `'weap'` group list. `None` when the
        /// reference is null (value 0), which is most of them.
        pub weapon_tag: Option<u32>,
        /// Fields 0x2A..=0x2D: ALL FOUR weapon slots' `'weap'` references, in slot order. `None`
        /// where the field was absent or its reference null. `weapon_tag` is `[0]`.
        pub weapon_slots: [Option<u32>; 4],
        /// Field 0x29's first sub-field: `ReadBits(3) - 1`, the unit's own weapon-selection
        /// state. -1 means "none".
        pub weapon_sel: Option<i8>,
        /// Field 0x29's two 2-bit sub-fields, each `flag ? -1 : ReadBits(2)`. One of them is the
        /// INDEX of the weapon in the unit's hands, which is what a swap changes.
        pub weapon_sel_ab: [Option<i8>; 2],
        /// Field 0x2F sub-field 1: the index of the player controlling this unit. The engine
        /// feeds this one to its player collection directly, so it NAMES the owner outright.
        pub controlling_player: Option<u8>,
        /// Field 0x31: a 3-bit value, -1 = none. The engine's equipment/ability mode.
        pub ability_mode: Option<i8>,
        /// Field 0x32: the unit's equipment (armour-ability) object reference.
        pub equipment_ref: Option<u32>,
        /// Field 0x35.
        pub assassination: Option<Assassination>,
        /// The mask rebuilt from the presence bits actually read. On a correct walk this equals
        /// the message's own 64-bit header mask -- the stream states the same thing twice, which
        /// is a free and very sharp oracle.
        pub rebuilt_mask: u64,
        /// Bits left over at the end of the payload. The writer rounds up to a byte, so a correct
        /// walk ends within 7.
        pub leftover_bits: usize,
        /// Which field the walk gave up on, if it did.
        pub stopped_at: Option<u32>,
        /// Which gated field latched the walk, if one did.
        pub latched_at: Option<u32>,
        /// Bits remaining in the payload at the moment field 0x35 began — its own bit budget,
        /// used to pin the record's internal layout.
        pub bits_at_0x35: Option<usize>,
    }

    impl UnitState {
        /// Did this walk pass BOTH oracles? Only a message that did may be believed.
        pub fn trustworthy(&self, header_mask: u64) -> bool {
            self.stopped_at.is_none() && self.leftover_bits < 8 && self.rebuilt_mask == header_mask
        }
    }

    fn cv3(r: &mut BitReader, p: &DecodeParams) -> Result<Option<[u32; 3]>, DecodeStop> {
        if !r.flag()? {
            return Err(DecodeStop::UnexercisedVectorPath);
        }
        let w = p.cv3_axis_bits;
        Ok(Some([r.bits(w[0])? as u32, r.bits(w[1])? as u32, r.bits(w[2])? as u32]))
    }

    /// `sub_1803CCC04` / `sub_1803D2D84`: an optional 27-bit object reference. A set flag means
    /// the null reference and nothing follows.
    fn object_ref(r: &mut BitReader) -> Result<Option<u32>, DecodeStop> {
        if r.flag()? {
            return Ok(None);
        }
        let hi = r.bits(16)?;
        let lo = r.bits(11)?;
        Ok(Some(((hi << 16) | lo) as u32))
    }

    /// `sub_1803C94B8`: `ReadBits(3) - 1`, then two `flag ? -1 : ReadBits(2)`.
    ///
    /// Returns all three, because one of the two 2-bit values is the index of the weapon the unit
    /// is HOLDING among its four slots -- the field a primary/secondary swap changes, and the one
    /// HMS needs if a body is not to latch onto whichever weapon it picked up last.
    fn weapon_selection(r: &mut BitReader) -> Result<(i8, [Option<i8>; 2]), DecodeStop> {
        let sel = r.bits(3)? as i8 - 1;
        let mut ab = [None, None];
        for s in ab.iter_mut() {
            if !r.flag()? {
                *s = Some(r.bits(2)? as i8);
            }
        }
        Ok((sel, ab))
    }

    /// The base-class walk (`sub_1803D7E58`), fields 0..0x13, gate list `{2, 8}`. Reimplemented
    /// here rather than reused because `film_decode`'s copy stops at the end of the base block
    /// and applies a biped-specific shortcut; this one has to carry straight on into the derived
    /// block. The two agree on every field cost.
    fn base_walk(r: &mut BitReader, p: &DecodeParams, st: &mut UnitState) -> Result<bool, DecodeStop> {
        const GATES: [u32; 2] = [2, 8];
        let mut g = 0usize;
        let mut field = 0u32;
        while field < 0x14 {
            if g < GATES.len() && field == GATES[g] {
                g += 1;
                if !r.flag()? {
                    // A CLEAR gate skips this gate's whole GROUP -- everything up to the next
                    // gate, or to the end of the block when it is the last one. It does not end
                    // the walk: a biped whose base block carries only a position reads a clear
                    // gate at field 2 AND a clear gate at field 8, and the derived block follows
                    // both. That is exactly the two bits HMS's body-yaw field was measured to sit
                    // past "the end of the base block", found against the aim oracle long before
                    // this walk existed, and the mask oracle here agrees: with these semantics the
                    // walk rebuilds the message's own header mask on 98.4% of 5.4M messages.
                    st.latched_at.get_or_insert(field);
                    field = GATES.get(g).copied().unwrap_or(0x14);
                    continue;
                }
            }
            let present = r.flag()?;
            let this = field;
            field += 1;
            if !present {
                continue;
            }
            let field = this;
            st.rebuilt_mask |= 1u64 << field;
            match field {
                0 => {
                    cv3(r, p)?;
                }
                1 => {
                    if !r.flag()? {
                        r.bits(19)?;
                        r.bits(10)?;
                    }
                }
                2 => {
                    let (axis_bits, angle_bits) = if p.host_authority { (20, 14) } else { (19, 8) };
                    if !r.flag()? {
                        r.bits(axis_bits)?;
                    }
                    r.bits(angle_bits)?;
                }
                3 => {
                    if !r.flag()? {
                        r.bits(19)?;
                        r.bits(8)?;
                    }
                }
                4 => {
                    r.bits(8)?;
                    r.bits(1)?;
                }
                5 => {}
                6 | 7 => {
                    r.bits(8)?;
                    r.bits(4)?;
                }
                8 => {
                    r.bits(5)?;
                    r.bits(48)?;
                }
                9 => {
                    r.bits(5)?;
                    for _ in 0..16 {
                        if r.flag()? {
                            r.bits(7)?;
                        }
                    }
                }
                10 => {
                    let n = r.bits(5)? as u32;
                    if n != 0 {
                        r.bits(n)?;
                        r.bits(n)?;
                    }
                }
                _ => {
                    st.stopped_at = Some(field);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Decode a type-11 message's whole dirty payload, base block then derived block.
    ///
    /// Returns `None` only when the message is not a decodable unit update at all. Otherwise the
    /// caller MUST check [`UnitState::trustworthy`] before believing any field: the walk stops
    /// cleanly at the first field it cannot cost, and a stopped walk's later fields are absent,
    /// not wrong.
    /// `weapref_bits` is `bits_needed(weap_tag_count + 1)` over the loaded map — 6 on
    /// forge_halo, which has 63 `'weap'` tags. It is map state, not film state, exactly like the
    /// position widths, and it only matters for the 0.13% of messages that carry a weapon tag;
    /// the assassination path does not touch it.
    pub fn decode(e: &EntityMessage, p: &DecodeParams, weapref_bits: u32) -> Option<UnitState> {
        if e.entity_type != 11 || e.dirty_mask == 0 {
            return None;
        }
        let mut r = BitReader::new(&e.payload);
        let mut st = UnitState::default();
        if e.is_update {
            r.skip(p.update_prefix_bits as usize).ok()?;
        } else {
            // A creation payload's own body comes first and is not costed here.
            return None;
        }
        match base_walk(&mut r, p, &mut st) {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                st.stopped_at = st.stopped_at.or(Some(0));
                st.leftover_bits = r.remaining();
                return Some(st);
            }
        }
        if derived_walk(&mut r, p, &mut st, weapref_bits).is_err() {
            st.stopped_at = st.stopped_at.or(Some(FIRST));
        }
        st.leftover_bits = r.remaining();
        Some(st)
    }

    fn derived_walk(r: &mut BitReader, p: &DecodeParams, st: &mut UnitState, weapref_bits: u32) -> Result<(), DecodeStop> {
        let mut g = 0usize;
        let mut field = FIRST;
        let end = FIRST + COUNT;
        while field < end {
            if g < GATES.len() && field == GATES[g] {
                g += 1;
                let Ok(open) = r.flag() else { return Ok(()) };
                if !open {
                    // skip this gate's group -- the four weapon slots are exactly such groups
                    field = GATES.get(g).copied().unwrap_or(end);
                    continue;
                }
            }
            // Running out of bits AT A PRESENCE BIT is a clean end, not a failure: the writer
            // stops emitting once every remaining field is absent, and rounds to a byte.
            let Ok(present) = r.flag() else { return Ok(()) };
            let this = field;
            field += 1;
            if !present {
                continue;
            }
            let field = this;
            st.rebuilt_mask |= 1u64 << field;
            match field {
                // the unit's aiming vector: 12-bit yaw then 11-bit pitch
                0x14 => {
                    st.aim_yaw_q = Some(r.bits(12)? as u16);
                    st.aim_pitch_q = Some(r.bits(11)? as u16);
                }
                // four weapon slots x five fields
                0x15 | 0x1A | 0x1F | 0x24 => {
                    r.bits(9)?;
                }
                0x16 | 0x1B | 0x20 | 0x25 => {
                    r.bits(11)?;
                }
                0x17 | 0x1C | 0x21 | 0x26 => {
                    r.bits(7)?;
                }
                0x18 | 0x1D | 0x22 | 0x27 => {
                    r.bits(7)?;
                    r.bits(2)?;
                }
                // weapon model/variant + damage state: 15 + 4, then sub_1804EC888
                0x19 | 0x1E | 0x23 | 0x28 => {
                    r.bits(15)?;
                    r.bits(4)?;
                    let kind = r.bits(2)?;
                    let n = match kind {
                        0 => 0,
                        1 => 1,
                        2 => 3,
                        _ => 4,
                    };
                    for _ in 0..n {
                        r.bits(11)?;
                    }
                }
                0x29 => {
                    let (sel, ab) = weapon_selection(r)?;
                    st.weapon_sel = Some(sel);
                    st.weapon_sel_ab = ab;
                }
                0x2A | 0x2B | 0x2C | 0x2D => {
                    let t = r.bits(weapref_bits)? as u32;
                    // 0 is the NULL reference, not the first weapon in the group. See
                    // `film_decode::WEAPREF_BITS` for the measurement that establishes both the
                    // width and this.
                    if field == 0x2A && t != 0 {
                        st.weapon_tag = Some(t);
                    }
                    if t != 0 {
                        st.weapon_slots[(field - 0x2A) as usize] = Some(t);
                    }
                    r.bits(12)?;
                    // The engine writes EVERY one of these into the same `unit + 0x348`
                    // selection struct, so the last one read wins -- mirror that rather than
                    // taking field 0x29's copy and ignoring the rest.
                    let (sel, ab) = weapon_selection(r)?;
                    st.weapon_sel = Some(sel);
                    st.weapon_sel_ab = ab;
                }
                // the two grenade counts, one byte each
                0x2E => {
                    r.bits(16)?;
                }
                // who controls this unit: five optional sub-fields, 4/6/9/5/8 bits
                0x2F => {
                    for (i, w) in [4u32, 6, 9, 5, 8].into_iter().enumerate() {
                        if r.flag()? {
                            let v = r.bits(w)?;
                            if i == 0 {
                                st.controlling_player = Some(v as u8);
                            }
                        }
                    }
                }
                0x30 => {
                    r.bits(if p.host_authority { 4 } else { 2 })?;
                }
                0x31 => {
                    st.ability_mode = Some(r.bits(3)? as i8 - 1);
                }
                0x32 => {
                    st.equipment_ref = object_ref(r)?;
                }
                0x33 => {
                    r.bits(1)?;
                }
                0x34 => {
                    r.bits(8)?;
                }
                // THE ASSASSINATION RECORD
                // The assassination record. Its internal layout is pinned by its own bit
                // BUDGET, measured rather than assumed: across four films the bits remaining when
                // the walk reaches this field are 105, 109, 110 and 112 -- and those films'
                // per-axis position widths sum to 45, 49, 50 and 52. The constant difference is
                // exactly 60 = 28 + 32, i.e. a 27-bit object reference under one present flag,
                // a 32-bit value, and a `compressed_vector3` with NO leading flag of its own.
                0x35 => {
                    st.bits_at_0x35 = Some(r.remaining());
                    let target = object_ref(r)?;
                    let stamp = r.bits(32)? as u32;
                    let w = p.cv3_axis_bits;
                    let position_raw = Some([r.bits(w[0])? as u32, r.bits(w[1])? as u32, r.bits(w[2])? as u32]);
                    st.assassination = Some(Assassination { target, position_raw, stamp });
                }
                _ => {
                    // 0x36..0x3B: the object-action enum, the control-input record and four AI
                    // fields. NONE of them occurs in any film measured (0 of 5,438,363 type-11
                    // messages), so their costs are unexercised and the walk stops rather than
                    // guessing.
                    st.stopped_at = Some(field);
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod probes;

#[cfg(test)]
mod unit_tests {
    use super::unit::*;
    use crate::film_decode::{DecodeParams, EntityMessage};

    /// Build a type-11 update message from a bit string, for the walk tests.
    fn msg(mask: u64, bits: &[(u64, u32)]) -> EntityMessage {
        let mut v: Vec<bool> = Vec::new();
        for &(val, n) in bits {
            for i in (0..n).rev() {
                v.push((val >> i) & 1 != 0);
            }
        }
        let mut bytes = vec![0u8; (v.len() + 7) / 8];
        for (i, b) in v.iter().enumerate() {
            if *b {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        EntityMessage { entity_type: 11, net_id: 0x1234_0001, dirty_mask: mask, payload: bytes, is_update: true }
    }

    fn params() -> DecodeParams {
        DecodeParams { cv3_axis_bits: [15, 16, 14], cv3_total_bits: 45, tagref_bits: 14, update_prefix_bits: 1, ..Default::default() }
    }

    /// The common biped update: a position, a velocity, and the aiming vector — and BOTH base
    /// gates clear, which is what ends the base block two bits before the derived one starts.
    #[test]
    fn the_common_biped_update_decodes_and_rebuilds_its_own_mask() {
        let p = params();
        let m = msg(
            (1 << 0) | (1 << 1) | (1 << 0x14),
            &[
                (0, 1),      // update prefix
                (1, 1),      // presence field 0
                (1, 1),      // compressed_vector3 bounds flag
                (111, 15),   // x
                (222, 16),   // y
                (333, 14),   // z
                (1, 1),      // presence field 1
                (0, 1),      // velocity present
                (0x3FFFF, 19),
                (0x155, 10),
                (0, 1), // GATE at field 2: clear -> skip to field 8
                (0, 1), // GATE at field 8: clear -> skip to the end of the base block
                (1, 1), // presence field 0x14
                (1024, 12), // yaw
                (1024, 11), // pitch
                (0, 1), // GATE 0x15 clear -> skip the weapon slots
                (0, 1), // GATE 0x1A clear
                (0, 1), // GATE 0x1F clear
                (0, 1), // GATE 0x24 clear
                (0, 1), // GATE 0x29 clear
                (0, 1), // GATE 0x2E clear -> nothing further
            ],
        );
        let st = decode(&m, &p, 6).expect("decodes");
        assert_eq!(st.aim_yaw_q, Some(1024));
        assert_eq!(st.aim_pitch_q, Some(1024));
        assert_eq!(st.rebuilt_mask, m.dirty_mask, "the walk must rebuild the header mask exactly");
        assert!(st.trustworthy(m.dirty_mask), "leftover {} stopped {:?}", st.leftover_bits, st.stopped_at);
    }

    /// A CLEAR gate skips that gate's GROUP, not the rest of the walk. Field 0x35 sits past five
    /// cleared gates, and it is only reachable because of that.
    #[test]
    fn a_cleared_gate_skips_its_group_not_the_rest_of_the_walk() {
        let p = params();
        let m = msg(
            (1 << 0x14) | (1u64 << 0x35),
            &[
                (0, 1), // update prefix
                (0, 1), // field 0 absent
                (0, 1), // field 1 absent
                (0, 1), // GATE 2 clear
                (0, 1), // GATE 8 clear
                (1, 1), // presence 0x14
                (4095, 12),
                (0, 11),
                (0, 1), // GATE 0x15 clear  -> 0x1A
                (0, 1), // GATE 0x1A clear  -> 0x1F
                (0, 1), // GATE 0x1F clear  -> 0x24
                (0, 1), // GATE 0x24 clear  -> 0x29
                (0, 1), // GATE 0x29 clear  -> 0x2E
                (1, 1), // GATE 0x2E OPEN
                (0, 1), // 0x2E absent
                (0, 1), // 0x2F
                (0, 1), // 0x30
                (0, 1), // 0x31
                (0, 1), // 0x32
                (0, 1), // 0x33
                (0, 1), // 0x34
                (1, 1), // 0x35 PRESENT
                (0, 1), // target present flag (clear = a real reference follows)
                (0x0ABC, 16),
                (0x123, 11),
                (0xDEAD_BEEF, 32),
                (7, 15),
                (9, 16),
                (11, 14),
            ],
        );
        let st = decode(&m, &p, 6).expect("decodes");
        assert_eq!(st.rebuilt_mask, m.dirty_mask);
        let a = st.assassination.expect("the assassination record was reached");
        assert_eq!(a.target, Some(0x0ABC_0123));
        assert_eq!(a.stamp, 0xDEAD_BEEF);
        assert_eq!(a.position_raw, Some([7, 9, 11]));
        assert!(st.trustworthy(m.dirty_mask), "leftover {}", st.leftover_bits);
    }

    #[test]
    fn a_truncated_unit_payload_stops_rather_than_panicking() {
        let p = params();
        for pattern in [0x00u8, 0xFF, 0xAA, 0x55] {
            for n in 0..200 {
                let m = EntityMessage {
                    entity_type: 11,
                    net_id: 1,
                    dirty_mask: 0xFFFF_FFFF_FFFF_FFFF,
                    payload: vec![pattern; n],
                    is_update: true,
                };
                let _ = decode(&m, &p, 6);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_decode::EntityTrack;

    /// Build a biped track that walks along +X with a fixed facing.
    fn track(points: &[(u32, [f32; 3])], facing: f32) -> EntityTrack {
        EntityTrack {
            net_id: 1,
            entity_type: 11,
            first_tick: points[0].0,
            last_tick: points[points.len() - 1].0,
            positions: points.to_vec(),
            orientations: Vec::new(),
            facings: points.iter().map(|p| (p.0, facing)).collect(),
            owner_raw: None,
            appearance: false,
            look: None,
            attachments: Vec::new(),
            moves_cached: None,
            weapons: Vec::new(),
            spawn_weapon: None,
            def_wire: None,
            spawned_on: None,
            destroyed_tick: None,
        }
    }

    /// #film-action  An assassination is read off the event channel, and the clip names it
    /// produces are the ones the biped's graph really has. The gate: change the species pair or
    /// the side and the clip has to change with it — a test that passed for any name would be
    /// worthless, because a wrong name silently leaves the body in whatever pose it was in.
    #[test]
    fn an_assassination_is_decoded_and_names_a_real_clip_pair() {
        use crate::film_action::ActionEvent;
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.events = vec![
            // the pair the engine always sends together
            ActionEvent { tick: 500, kind: 14, subject: Some(0xAA), other: Some(0xBB), incident: Some(0x0078_0000), ..Default::default() },
            ActionEvent { tick: 500, kind: 14, subject: Some(0xAA), other: Some(0xBB), incident: Some(0x0078_0015), ..Default::default() },
            // an ordinary melee kill, which must NOT become an assassination
            ActionEvent { tick: 900, kind: 14, subject: Some(0xCC), other: Some(0xDD), incident: Some(0x0078_0018), ..Default::default() },
        ];
        let a = assassinations(&idx);
        assert_eq!(a.len(), 1, "only the finishing move is an assassination: {a:?}");
        assert_eq!(a[0].tick, 500);
        assert_eq!(a[0].attacker, Some(0xAA));
        assert_eq!(a[0].victim, Some(0xBB));
        assert_eq!(a[0].incident, "finishing_move_kill");
        assert!(!a[0].move_decoded, "no host record, so the move is chosen");
        assert!(a[0].covers(500) && a[0].covers(500 + SYNC_TICKS - 1));
        assert!(!a[0].covers(499) && !a[0].covers(500 + SYNC_TICKS));

        // The host's own record upgrades the SAME row rather than adding a second one.
        idx.events.push(ActionEvent {
            tick: 503,
            kind: crate::film_action::EV_ASSASSINATE,
            subject: Some(0xAA),
            other: Some(0xBB),
            action_index: Some(3),
            ..Default::default()
        });
        let a = assassinations(&idx);
        assert_eq!(a.len(), 1, "the request and the broadcast are one kill: {a:?}");
        assert!(a[0].move_decoded);
        assert_eq!(a[0].action_index, Some(3));
    }

    #[test]
    fn each_species_pair_gets_its_own_sync_clips_and_the_two_sides_differ() {
        // Spartan on Spartan, both halves in the Spartan's own graph.
        let att = sync_candidates(false, false, SyncSide::Attacker, 0);
        let vic = sync_candidates(false, false, SyncSide::Victim, 0);
        assert_eq!(att[0], "any:any:any:sync_actions:svs_ground_close_backstab_attacker");
        assert_eq!(vic[0], "any:any:any:sync_actions:svs_ground_close_backstab_victim");
        assert_ne!(att[0], vic[0], "the two sides are DIFFERENT clips");
        assert!(is_sync_clip(&att[0]) && is_sync_clip(&vic[0]));
        assert!(!is_sync_clip("combat:rifle:idle"));
        // Shift the move and the clip shifts with it.
        assert_eq!(sync_candidates(false, false, SyncSide::Attacker, 1)[0], "any:any:any:sync_actions:svs_ground_skull_stab_attacker");
        // The species pair picks a different SET — and these prefixes are the ones the shipped
        // graphs carry, which is what makes them loadable at all.
        assert!(sync_candidates(true, true, SyncSide::Attacker, 0)[0].contains(":eve_"));
        assert!(sync_candidates(false, true, SyncSide::Attacker, 0)[0].contains(":sve_"));
        assert!(sync_candidates(true, false, SyncSide::Attacker, 0)[0].contains(":evs_"));
        // Every candidate list ends up covering its whole set, so one missing clip is survivable.
        assert_eq!(att.len(), SVS_MOVES.len());
        assert!(is_elite("objects\\characters\\elite\\elite"));
        assert!(!is_elite("objects\\characters\\spartans\\spartans"));
    }

    /// #film-action  The decoded death moment, and the negative: a body the event channel says
    /// nothing about still falls back to the derived one.
    #[test]
    fn a_death_the_recording_states_beats_the_one_we_infer() {
        use crate::film_action::ActionEvent;
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.events = vec![
            ActionEvent { tick: 1200, kind: 14, subject: Some(7), other: Some(9), incident: Some(0x0078_0018), ..Default::default() },
            // a later row naming the same body must not move the moment
            ActionEvent { tick: 1800, kind: 14, subject: Some(7), other: Some(9), incident: Some(0x0078_0041), ..Default::default() },
        ];
        assert_eq!(death_from_events(&idx, 9), Some((1200, "melee_kill", Some(7))));
        // The killer's own body is named FIRST in those rows, so it is not a death of theirs.
        assert_eq!(death_from_events(&idx, 7), None);
        assert_eq!(death_from_events(&idx, 1234), None);
    }

    /// #death The death instant is the last reported POSITION of a track the film destroyed —
    /// not its last message and not the destroy. Both of those are many seconds later (measured
    /// medians: 4 s and 16 s), which is why a killed player used to stand frozen and then vanish.
    #[test]
    fn a_death_is_the_last_reported_position_of_a_destroyed_track() {
        let mut t = track(&[(100, [0.0, 0.0, 0.0]), (200, [5.0, 0.0, 0.0])], 0.0);
        t.last_tick = 440; // kept reporting other fields for another 4 s
        assert_eq!(death_tick(&t), None, "a track the film never destroyed is not a death");
        t.destroyed_tick = Some(1400); // deleted 16 s after that
        assert_eq!(death_tick(&t), Some(200), "the death is the last POSITION, not 440 or 1400");
        // a recycled net id (destroy long before the last position) is not one life, so no death
        let mut r = t.clone();
        r.destroyed_tick = Some(10);
        assert_eq!(death_tick(&r), None);
        // only bipeds
        let mut v = t.clone();
        v.entity_type = 15;
        assert_eq!(death_tick(&v), None);
    }

    /// #death The "close and from behind" geometry. It is reported, never acted on — the
    /// measured contrast is 28% against 16%, which is real but not an identification.
    #[test]
    fn death_company_measures_distance_and_bearing_from_the_victims_back() {
        // victim faces +X and dies at (0,0,0)
        let mut victim = track(&[(0, [0.0, 0.0, 0.0]), (100, [0.0, 0.0, 0.0])], 0.0);
        victim.destroyed_tick = Some(600);
        let mk = |id: u32, x: f32, y: f32| {
            let mut k = track(&[(0, [x, y, 0.0]), (100, [x, y, 0.0])], 0.0);
            k.net_id = id;
            k.owner_raw = Some(id as u8);
            k
        };
        victim.owner_raw = Some(9);
        // directly behind, 1 wu away
        let tracks = vec![victim.clone(), mk(2, -1.0, 0.0)];
        let c = death_company(&tracks, &tracks[0], 100).expect("someone was there");
        assert!((c.distance - 1.0).abs() < 1e-4);
        assert!(c.from_behind.unwrap() < 1.0, "directly behind is 0 deg, got {:?}", c.from_behind);
        // directly in FRONT, same distance: still reported, but 180 deg off the back
        let tracks = vec![victim.clone(), mk(3, 1.0, 0.0)];
        let c = death_company(&tracks, &tracks[0], 100).unwrap();
        assert!((c.from_behind.unwrap() - 180.0).abs() < 1.0, "{:?}", c.from_behind);
        // out of range, and a body of the SAME player, are both ignored
        let tracks = vec![victim.clone(), mk(4, -8.0, 0.0)];
        assert_eq!(death_company(&tracks, &tracks[0], 100), None, "8 wu is not company");
        let mut same = mk(5, -1.0, 0.0);
        same.owner_raw = Some(9);
        let tracks = vec![victim.clone(), same];
        assert_eq!(death_company(&tracks, &tracks[0], 100), None, "a player's own other life is not a killer");
    }

    /// The defect behind "the players look like Mr. Krabs": ONE stride constant for every clip.
    /// A walking body and a running body cover very different ground per animation cycle, so the
    /// rate has to differ — and by the measured numbers it differs by about 2.4x.
    #[test]
    fn each_gait_advances_its_cycle_at_its_own_rate() {
        let walk = cycles_per_wu(Stance::Combat, Gait::WalkFront).unwrap();
        let run = cycles_per_wu(Stance::Combat, Gait::RunFront).unwrap();
        let sprint = cycles_per_wu(Stance::Sprint, Gait::RunFront).unwrap();
        assert!(walk > run && run > sprint, "faster gaits cover more ground per cycle: {walk} {run} {sprint}");
        assert!((walk / run - 2.45).abs() < 0.2, "walk/run ratio {} is not the measured 2.45", walk / run);
        // the old single constant, 7.5 frames/wu over a 24-frame walk clip, is 0.3125 cycles/wu --
        // three times too slow for a walk, which is why the feet skated
        assert!(walk / (7.5 / 24.0) > 2.5, "the old global constant was far too slow for a walk");
        // idle and turning have no ground travel to drive them
        assert!(cycles_per_wu(Stance::Combat, Gait::Idle).is_none());
        assert!(cycles_per_wu(Stance::Combat, Gait::TurnLeft).is_none());
    }

    /// The phase must be CONTINUOUS when the gait changes. The old code recomputed
    /// `(distance * k) % new_frame_count`, so the pose jumped the instant a body went from
    /// walking to running.
    /// #gait-blend  The ASSERTION BATTERY for [`travel_directions`], taken as a function so the
    /// gate below can run it against the real implementation and
    /// [`the_direction_mix_battery_catches_a_broken_implementation`] can run it against
    /// deliberately broken ones.
    ///
    /// Factored this way for a specific reason: a gate is not trusted until it has been WATCHED
    /// GO RED, and watching it go red by editing the shared tree means a cold rebuild — which,
    /// on the machine this was written on, means `systemd-oomd` with 36 GB of zram already in
    /// use. One battery, applied to the real function and to one saboteur per assertion, is the
    /// same evidence without the mutation, and it stays in the repository instead of being
    /// restored and forgotten.
    ///
    /// Each numbered block is one assertion, and no two saboteurs below trip the same one.
    fn direction_mix_battery(f: &dyn Fn(f32) -> (i16, i16, f32), check: usize) {
        use std::f32::consts::{FRAC_PI_2, PI};
        // The convention, written out rather than read back: 0 = straight ahead, +pi/2 = LEFT,
        // +-pi = back, -pi/2 = right (the frame `BodyState::travel_angle` is built in), and the
        // engine's gait directions are 0 front / 1 back / 2 left / 3 right (measured; see
        // `gait_direction`).
        if check == 1 {
            // 1. straight ahead is wholly direction 0, bracketed toward LEFT.
            let (a, b, w) = f(0.0);
            assert_eq!((a, b), (0, 2), "straight ahead must bracket front -> left");
            assert_eq!(w, 0.0, "straight ahead must be wholly the front clip");
        }
        if check == 2 {
            // 2. straight left is wholly direction 2 — and it is reached as the SECOND of
            //    (front, left) from below and the FIRST of (left, back) at pi/2, so an off-by-one
            //    in the branch boundary shows here.
            let (a, b, w) = f(FRAC_PI_2);
            assert_eq!((a, b), (2, 1), "straight left must bracket left -> back");
            assert_eq!(w, 0.0, "straight left must be wholly the left clip");
        }
        if check == 3 {
            // 3. straight right, and straight back from BOTH sides of the wrap.
            assert_eq!(f(-FRAC_PI_2).0, 3, "straight right is direction 3");
            assert_eq!(f(-FRAC_PI_2).2, 0.0, "...and wholly it");
            assert_eq!(f(PI), (2, 1, 1.0), "+pi is wholly BACK, reached from the left side");
            assert_eq!(f(-PI), (1, 3, 0.0), "-pi is wholly BACK, reached from the right side");
        }
        if check == 4 {
            // 4. a true 45-degree diagonal is an EVEN mix. This is the user's own case.
            let (a, b, w) = f(FRAC_PI_2 * 0.5);
            assert_eq!((a, b), (0, 2));
            assert!((w - 0.5).abs() < 1e-6, "45 degrees must be half and half, got {w}");
        }
        if check == 5 {
            // 5. CONTINUITY of the whole weight vector, swept right round the circle through
            //    all four boundaries. Expressed per DIRECTION so that a bracketing pair which
            //    swaps while the departing weight is still non-zero reads as a jump.
            let vec_at = |t: f32| {
                let (a, b, w) = f(t);
                let mut v = [0.0f32; 4];
                if (0..4).contains(&a) {
                    v[a as usize] += 1.0 - w;
                }
                if (0..4).contains(&b) {
                    v[b as usize] += w;
                }
                v
            };
            const N: usize = 4096;
            let step = std::f32::consts::TAU / N as f32;
            let mut worst = 0.0f32;
            let mut prev = vec_at(-PI + 1e-6);
            for i in 1..=N {
                let mut t = -PI + 1e-6 + step * i as f32;
                if t > PI {
                    t -= std::f32::consts::TAU;
                }
                let now = vec_at(t);
                let d = prev.iter().zip(now.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                worst = worst.max(d);
                prev = now;
            }
            // One step turns the body 0.088 degrees. The smoothstepped weight's steepest slope
            // is 1.5 per unit of `u = theta / (pi/2)`, so the largest honest change per step is
            // `1.5 * step / (pi/2)` = 0.00147. A pair that swapped while a weight was still 1.0
            // would read 1.0 here — three orders of magnitude out, not a tolerance question.
            let bound = 1.5 * step / FRAC_PI_2;
            assert!(
                worst <= bound * 1.05,
                "the direction mix jumps: worst step {worst} over {} degrees, bound {bound}",
                step.to_degrees()
            );
        }
        if check == 6 {
            // 6. ...and the smoothstep is really there: the weight's slope at a cardinal must be
            //    ZERO, which a linear weight cannot satisfy. A finite difference just inside the
            //    sector, against what a LINEAR ramp would read at the same offset.
            let eps = 1e-3;
            let lin = eps / FRAC_PI_2;
            let got = f(eps).2;
            assert!(
                got < lin * 0.05,
                "the directional weight is linear, not smoothstepped: {got} at {eps} rad, a linear ramp gives {lin}"
            );
        }
    }

    /// #gait-blend  **THE DIRECTION-MIX GATE.** The real [`travel_directions`] against every
    /// assertion in [`direction_mix_battery`].
    #[test]
    fn the_travel_direction_mix_is_right_at_the_cardinals_and_continuous_between_them() {
        for check in 1..=6 {
            direction_mix_battery(&travel_directions, check);
        }
    }

    /// #gait-blend  **THE GATE'S OWN GATE.** Every assertion in [`direction_mix_battery`] is
    /// shown to catch a broken implementation, and every assertion is shown to be NECESSARY --
    /// i.e. there is a bug that only it catches.
    ///
    /// Two things are checked, and the distinction matters because the first alone is weak:
    ///
    /// 1. every saboteur is caught by the assertion it targets (`contains`), and
    /// 2. for every assertion there is a saboteur caught by **that assertion alone**, which is
    ///    what proves the assertion is not redundant with its neighbours.
    ///
    /// The battery runs each assertion in its own `catch_unwind`, so one red cannot MASK
    /// another the way it can inside a single `#[test]` -- which is why a saboteur is allowed
    /// to trip several assertions here. The first run of this test found exactly that: the
    /// "front partner is right, not left" saboteur was caught by 1, 4 and 5, which is three
    /// gates doing their job rather than two of them hiding.
    ///
    /// **What this does NOT cover.** A closure battery proves the ASSERTIONS discriminate. It
    /// says nothing about whether the production path invokes [`travel_directions`] at all --
    /// that is the "the capability exists and nothing invokes it" trap, and it is covered
    /// separately by `play_mode::probes::probe_the_gait_blend_at_the_call_site`.
    #[test]
    fn the_direction_mix_battery_catches_a_broken_implementation() {
        use std::f32::consts::{FRAC_PI_2, PI};
        type F = Box<dyn Fn(f32) -> (i16, i16, f32)>;
        // (target assertion, exclusive?, label, saboteur)
        let saboteurs: Vec<(usize, bool, &str, F)> = vec![
            // --- REALISTIC bugs. Several assertions may catch each, and that is fine. -------
            // The front sector's partner is RIGHT instead of LEFT. The two are easy to
            // transpose, and transposing them strafes every diagonal the wrong way.
            (1, false, "front partner is right, not left", Box::new(|t: f32| {
                let (a, b, w) = travel_directions(t);
                if a == 0 && b == 2 { (0, 3, w) } else { (a, b, w) }
            })),
            // The negative wrap loses its `+ 2.0`. Note which assertion catches it: NOT the
            // `-pi` equality (at exactly -pi the clamp happens to give the right answer) but
            // the continuity sweep, which sees the jump this leaves in the middle of the
            // back-to-right sector. That is the sweep earning its keep.
            (5, false, "the -pi wrap loses its offset", Box::new(|t: f32| {
                let u = t / FRAC_PI_2;
                if u < -1.0 {
                    let x = (u + 1.0).clamp(0.0, 1.0);
                    (1, 3, x * x * (3.0 - 2.0 * x))
                } else { travel_directions(t) }
            })),
            // The mix SWITCHES instead of blending -- the shape the code had before this round,
            // and the user's actual complaint.
            (5, false, "the mix switches instead of blending", Box::new(|t: f32| {
                let (a, b, w) = travel_directions(t);
                (a, b, if w >= 0.5 { 1.0 } else { 0.0 })
            })),
            // --- NARROW saboteurs, one per assertion, each caught by that assertion ALONE. ---
            // 1: only the straight-ahead answer is wrong.
            (1, true, "straight ahead brackets the wrong partner", Box::new(|t: f32| {
                if t == 0.0 { (0, 3, 0.0) } else { travel_directions(t) }
            })),
            // 2: the `u >= 1.0` boundary becomes `u > 1.0`, so EXACTLY straight left falls in
            //    the front sector. The weight VECTOR is unchanged (both say "wholly left"), so
            //    only the bracketing-pair equality can see it.
            (2, true, "straight left falls in the front sector", Box::new(|t: f32| {
                if (t - FRAC_PI_2).abs() < 1e-7 { (0, 2, 1.0) } else { travel_directions(t) }
            })),
            // 3: only -pi is wrong, and wrong in a way the weight vector cannot see (it still
            //    says "wholly back").
            (3, true, "-pi is reached from the wrong side", Box::new(|t: f32| {
                if (t + PI).abs() < 1e-7 { (2, 1, 1.0) } else { travel_directions(t) }
            })),
            // 4: a small SMOOTH bump in the middle of the weight. It vanishes at both ends, it
            //    keeps the slope inside the continuity sweep's bound (1.5 x (1 + 0.01*pi) =
            //    1.547 against a bound of 1.575) and it leaves the slope at a cardinal zero --
            //    so ONLY the 45-degree equality can catch it. (A plain `w*w` would ALSO be
            //    caught by the sweep, whose bound is tight enough to see the doubled slope near
            //    w = 1; that is the sweep working, but it makes w*w a poor test of assertion 4.)
            (4, true, "the weight is bumped off centre, so 45 degrees is not half and half", Box::new(|t: f32| {
                let (a, b, w) = travel_directions(t);
                (a, b, w + 0.01 * (std::f32::consts::PI * w).sin())
            })),
            // 5: a jump in the MIDDLE of a sector. Every cardinal is right, 45 degrees is right,
            //    the slope at 0 is right -- only the continuity sweep can catch it.
            (5, true, "the weight jumps inside the sector", Box::new(|t: f32| {
                let (a, b, w) = travel_directions(t);
                (a, b, if (0.6..0.95).contains(&w) { 0.95 } else { w })
            })),
            // 6: the smoothstep is dropped for a linear ramp. Continuous, 0.5 at 45 degrees and
            //    right at every cardinal -- so ONLY the zero-slope assertion can catch it.
            (6, true, "linear weight, no smoothstep", Box::new(|t: f32| {
                let u = t / FRAC_PI_2;
                let (a, b, x) = if u >= 1.0 { (2, 1, u - 1.0) }
                    else if u >= 0.0 { (0, 2, u) }
                    else if u >= -1.0 { (3, 0, u + 1.0) }
                    else { (1, 3, u + 2.0) };
                (a, b, x.clamp(0.0, 1.0))
            })),
        ];
        const N: usize = 6;
        let run = |f: &dyn Fn(f32) -> (i16, i16, f32)| -> Vec<usize> {
            (1..=N)
                .filter(|&c| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| direction_mix_battery(f, c))).is_err()
                })
                .collect()
        };
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // the saboteurs are MEANT to panic; keep it quiet
        let results: Vec<(usize, bool, &str, Vec<usize>)> = saboteurs
            .iter()
            .map(|(t, excl, l, f)| (*t, *excl, *l, run(f.as_ref())))
            .collect();
        let real = run(&travel_directions);
        std::panic::set_hook(prev);
        for (target, exclusive, label, failed) in &results {
            assert!(
                failed.contains(target),
                "saboteur '{label}' was NOT caught by assertion {target} (caught by {failed:?}) -- \
                 that assertion is vacuous for this bug"
            );
            if *exclusive {
                assert_eq!(
                    failed,
                    &vec![*target],
                    "saboteur '{label}' must be caught by assertion {target} ALONE, to prove that \
                     assertion is not redundant; it was caught by {failed:?}"
                );
            }
        }
        // Coverage: every assertion must have at least one saboteur that ONLY it catches.
        for k in 1..=N {
            assert!(
                results.iter().any(|(_, _, _, f)| f == &vec![k]),
                "assertion {k} has no saboteur of its own, so nothing proves it is doing any work"
            );
        }
        // ...and the real implementation must be caught by nothing, or the battery is asserting
        // something no correct implementation can satisfy.
        assert!(real.is_empty(), "the real implementation fails assertions {real:?}");
    }

    /// #film-perf  THE GATE on the prefix-sum phase cache: for the same track, the same tick and
    /// the same `frames_of`, [`cycle_phase_at`] must return the SAME `f32` as the walk
    /// [`cycle_phase_with`] -- bit for bit, not within a tolerance.
    ///
    /// Bit equality is the right standard here and not pedantry. The phase decides the pose, and
    /// the rule a Theater frame lives by is that it is a pure function of the playhead: if the
    /// cached phase differed in the last bit, a body posed after a scrub would differ from the
    /// same body posed by playing up to the same tick, and "the same tick twice is exact" would
    /// stop being true. Equivalence is what buys the speed, so equivalence is what is asserted.
    ///
    /// The fixture contains every case the walk treats specially: a respawn TELEPORT (skipped as
    /// not a stride), REPEATED ticks (zero elapsed time), a change of gait, a stand-still stretch
    /// (clock-driven rather than stride-driven), and ticks that fall strictly INSIDE a segment,
    /// before the first sample and after the last.
    ///
    /// Sabotage check: dropping the partial-segment term from `cycle_phase_at` makes this fail at
    /// every tick that lands inside a segment.
    #[test]
    fn the_prefix_phase_is_bit_identical_to_the_walk() {
        let mut pts: Vec<(u32, [f32; 3])> = Vec::new();
        let mut x = 0.0f32;
        for t in (0..=120).step_by(6) {
            pts.push((t, [x, 0.0, 0.0]));
            x += if t < 60 { 1.3 / 10.0 } else { 3.6 / 10.0 };
        }
        pts.push((126, [x, 0.0, 0.0]));
        pts.push((126, [x, 0.0, 0.0])); // a repeated tick: zero elapsed time
        x += 400.0;
        pts.push((132, [x, 0.0, 0.0])); // a respawn teleport, which is not a stride
        for t in (138..=400).step_by(7) {
            pts.push((t, [x, 0.0, 0.0])); // standing still: the clock drives the phase
        }
        for t in (407..=600).step_by(5) {
            x += 0.2;
            pts.push((t, [x, 0.0, 0.0]));
        }
        let tr = track(&pts, 0.0);
        let knows: &dyn Fn(Stance, Gait) -> Option<f32> = &|_, _| Some(37.0);
        let blind: &dyn Fn(Stance, Gait) -> Option<f32> = &|_, _| None;
        // Two different `frames_of`, because they take different branches inside `segment_rate`.
        for (label, f) in [("knows the clip lengths", knows), ("knows none", blind)] {
            let pre = cycle_phase_prefix(&tr, f);
            assert_eq!(pre.len(), tr.positions.len(), "one prefix entry per sample");
            for tick in 0..=700u32 {
                let walk = cycle_phase_with(&tr, tick, f);
                let cached = cycle_phase_at(&tr, tick, &pre, f).expect("the prefix matches the track");
                assert_eq!(
                    walk.to_bits(),
                    cached.to_bits(),
                    "{label}: tick {tick}: the walk gave {walk} and the prefix gave {cached}"
                );
            }
            // and the phase really does move over this track, so this is not asserting 0 == 0
            assert!(cycle_phase_with(&tr, 700, f) > 0.5, "{label}: the fixture must actually advance");
            // the partial-segment term must really be exercised: at least one tick has to land
            // strictly inside a segment and give a value between its two endpoints
            let inside = (1..700u32).any(|t| {
                let a = cycle_phase_with(&tr, t, f);
                let b = cycle_phase_with(&tr, t + 1, f);
                b > a && !tr.positions.iter().any(|(tk, _)| *tk == t)
            });
            assert!(inside, "{label}: the fixture must have ticks strictly inside a segment");
        }
        // A prefix that does not describe this track must be REFUSED, never used: that is what
        // stops a stale cache from posing a body.
        let pre = cycle_phase_prefix(&tr, blind);
        assert!(cycle_phase_at(&tr, 300, &pre[..pre.len() - 1], blind).is_none());
        assert!(cycle_phase_at(&tr, 300, &[], blind).is_none());
    }

    /// A body that stops still must keep breathing rather than freezing, and must not resume a
    /// stride from where it left off minutes earlier.
    #[test]
    fn a_standing_body_is_driven_by_the_clock_not_by_distance() {
        let pts: Vec<(u32, [f32; 3])> = (0..=600).step_by(6).map(|t| (t, [0.0f32, 0.0, 0.0])).collect();
        let t = track(&pts, 0.0);
        let a = cycle_phase(&t, 60);
        let b = cycle_phase(&t, 300);
        assert!(b > a, "an idle body's cycle must still advance: {a} -> {b}");
    }

    /// #anim-rate  A CLOCK-DRIVEN clip plays at ITS OWN authored rate — the single number behind
    /// "the idle animations are playing way too fast".
    ///
    /// The shipped Spartan `combat:<class>:idle` is **156 frames**, so one cycle is 5.2 s. The
    /// old fallback was a flat `CLIP_FPS * dt / 20.0` = 1.5 cycles a SECOND whatever was loaded,
    /// i.e. 7.8x too fast on that clip; and because it hard-coded 20 frames it was wrong by a
    /// different factor for every other clock-driven clip in the graph.
    #[test]
    fn a_clock_driven_clip_advances_at_the_rate_its_own_length_implies() {
        // ten seconds of a body standing perfectly still
        let pts: Vec<(u32, [f32; 3])> = (0..=600).step_by(6).map(|t| (t, [0.0f32, 0.0, 0.0])).collect();
        let t = track(&pts, 0.0);
        // every clock-driven clip the two shipped graphs actually have at these states
        for n in [156.0f32, 50.0, 18.0, 70.0, 30.0, 45.0] {
            let cycles = cycle_phase_with(&t, 600, &|_, _| Some(n));
            let want = 10.0 * CLIP_FPS / n; // 10 s at the clip's authored rate
            assert!(
                (cycles - want).abs() < 0.05,
                "a {n}-frame clip must run {want:.3} cycles in 10 s, got {cycles:.3}"
            );
        }
        // and with no clip loaded yet it seeds off the shipped graph's own length, not 20 frames
        let seeded = cycle_phase_with(&t, 600, &|_, _| None);
        let want = 10.0 * CLIP_FPS / clock_clip_frames(Stance::Combat, Gait::Idle).unwrap();
        assert!((seeded - want).abs() < 0.05, "seeded rate {seeded:.3}, wanted {want:.3}");
        // the bug, stated as a number: the old law ran 7.8 cycles of the real idle in 10 s
        assert!(seeded < 10.0 * CLIP_FPS / 20.0 / 7.0, "still near the old 1.5 cycles/s law: {seeded}");
    }

    /// Keying the stride on the GAIT assumes the rifle set's clip lengths. When the clip that is
    /// actually loaded is a different length, the rate must follow it — otherwise a rocket
    /// launcher's walk would be driven at the rifle walk's stride once the weapon class lands.
    #[test]
    fn the_stride_rate_follows_the_clip_that_is_actually_loaded() {
        let gait_keyed = cycles_per_wu(Stance::Combat, Gait::WalkFront).unwrap();
        // The rifle walk is 24 frames; asking for 24 must reproduce the gait-keyed answer.
        let same = cycles_per_wu_for_clip(Stance::Combat, Gait::WalkFront, 24.0).unwrap();
        assert!((gait_keyed - same).abs() < 1e-4, "{gait_keyed} vs {same}");
        // A longer clip covers more ground per cycle, so fewer cycles per world unit.
        let longer = cycles_per_wu_for_clip(Stance::Combat, Gait::WalkFront, 48.0).unwrap();
        assert!((longer - gait_keyed / 2.0).abs() < 1e-4, "{longer}");
        // A clip with no frames cannot drive anything.
        assert!(cycles_per_wu_for_clip(Stance::Combat, Gait::WalkFront, 0.0).is_none());
        // And a gait with no ground travel still has none.
        assert!(cycles_per_wu_for_clip(Stance::Combat, Gait::Idle, 24.0).is_none());
    }

    /// The phase integration must use the supplied clip length when one is known, and fall back
    /// to the measured default when it is not.
    #[test]
    fn the_phase_uses_the_supplied_clip_length() {
        let pts: Vec<(u32, [f32; 3])> =
            (0..=120).step_by(6).map(|t| (t, [t as f32 * 1.32 / 60.0, 0.0, 0.0])).collect();
        let t = track(&pts, 0.0);
        let default = cycle_phase(&t, 120);
        // Twice the frames = half the cycles over the same ground.
        let doubled = cycle_phase_with(&t, 120, &|_, _| Some(48.0));
        assert!(default > 0.0, "the body walked, so the phase must advance");
        assert!(
            (doubled - default / 2.0).abs() < default * 0.15,
            "a clip twice as long should cover about half the cycles: {default} -> {doubled}"
        );
    }

    #[test]
    fn a_body_walking_the_way_it_faces_runs_forward() {
        // #gait-authored  The SPARTAN'S OWN authored `run forward`, 2.200 wu/s, facing +X and
        // travelling +X. This fixture used to be 4 wu/s, which the old corpus-fitted cuts called
        // an ordinary run and the authored cuts correctly call a sprint.
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [2.2, 0.0, 0.0])], 0.0);
        let st = derive(&t, 30, None);
        assert_eq!(st.gait, Gait::RunFront);
        assert_eq!(st.stance, Stance::Combat);
        assert_eq!(st.provenance, Provenance::Derived);
        assert!(anim_candidates(&st, "rifle")[0].ends_with("move_front_fast"));
    }

    /// The defect this module exists to fix: the old ladder chose from SPEED alone, so a body
    /// moving backwards played a forward run.
    #[test]
    fn a_body_walking_backwards_plays_the_back_cycle() {
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [-3.0, 0.0, 0.0])], 0.0);
        let st = derive(&t, 30, None);
        assert_eq!(st.gait, Gait::Back);
        assert!(anim_candidates(&st, "rifle")[0].contains("move_back"));
    }

    #[test]
    fn strafing_picks_the_side_the_body_is_actually_moving_to() {
        // facing +X, travelling +Y = to the body's LEFT
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [0.0, 3.0, 0.0])], 0.0);
        assert_eq!(derive(&t, 30, None).gait, Gait::Left);
        // facing +X, travelling -Y = to its RIGHT
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [0.0, -3.0, 0.0])], 0.0);
        assert_eq!(derive(&t, 30, None).gait, Gait::Right);
        // and the same motion with the body turned 90 degrees is now FORWARD, which is the
        // whole point: the gait is in the body's frame, not the world's.
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [0.0, 3.0, 0.0])], std::f32::consts::FRAC_PI_2);
        assert_eq!(derive(&t, 30, None).gait, Gait::RunFront);
    }

    #[test]
    fn a_falling_body_is_airborne_not_sprinting() {
        // 10 wu/s straight down: the old ladder's top tier was "move_front_fast".
        let t = track(&[(0, [0.0, 0.0, 20.0]), (60, [0.0, 0.0, 10.0])], 0.0);
        let st = derive(&t, 30, None);
        assert_eq!(st.stance, Stance::Flight);
        assert!(anim_candidates(&st, "rifle").iter().any(|n| n.contains("airborne")));
    }

    /// #gait-authored  **THE AUTHORED SPEEDS MUST LAND IN THE RIGHT GAIT.** This is the gate the
    /// old corpus-fitted cuts could not pass, and it is written in the engine's own numbers so it
    /// fails if anyone retunes them by feel.
    ///
    /// The three speeds a Reach player can actually hold on foot, per `matg`'s
    /// `@player information` row: the authored **walk**, the authored **run forward**, and the
    /// authored **sprint** (run x the authored `full speed multiplier`). Each must choose the
    /// matching animation, for BOTH species, whose rows differ.
    ///
    /// With the old cuts (`WALK` 2.4, `SPRINT` 4.6) this gate fails twice over: a full-speed run
    /// of 2.200 falls below the walk/run split and is drawn WALKING, and no reachable speed ever
    /// exceeds the sprint floor, so a sprint is never drawn at all.
    #[test]
    fn the_authored_player_speeds_choose_the_matching_gait() {
        // (label, speed wu/s) -- from `--dump-play-phys` on a real forge_halo:
        //   Spartan  walk 0.512  run 2.200  sprint x1.623 -> 3.571
        //   Elite    walk 0.650  run 2.600  sprint x1.400 -> 3.640
        let walks = [("spartan walk", 0.512f32), ("elite walk", 0.650)];
        let runs = [("spartan run", 2.200f32), ("elite run", 2.600)];
        let sprints = [("spartan sprint", 3.571f32), ("elite sprint", 3.640)];

        for (what, v) in walks {
            let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [v, 0.0, 0.0])], 0.0);
            let st = derive(&t, 30, None);
            assert_eq!(st.gait, Gait::WalkFront, "{what} ({v} wu/s) must walk, got {:?}", st.gait);
            assert_eq!(st.stance, Stance::Combat, "{what} must not be a sprint");
        }
        for (what, v) in runs {
            let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [v, 0.0, 0.0])], 0.0);
            let st = derive(&t, 30, None);
            assert_eq!(
                st.gait, Gait::RunFront,
                "{what} ({v} wu/s) is the FASTEST a Spartan runs without sprinting and must play \
                 the run cycle, got {:?} -- the old 2.4 walk/run split drew it walking",
                st.gait
            );
            assert_eq!(st.stance, Stance::Combat, "{what} is not a sprint");
        }
        for (what, v) in sprints {
            let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [v, 0.0, 0.0])], 0.0);
            let st = derive(&t, 30, None);
            assert_eq!(
                st.stance, Stance::Sprint,
                "{what} ({v} wu/s) is the authored sprint and must BE a sprint, got {:?} -- the \
                 old 4.6 floor sat above every speed a player can reach, so sprint was dead code",
                st.stance
            );
        }
        // And the cuts must genuinely separate the three, not merely admit them: the fastest
        // authored walk must be below the split and the fastest authored run below the sprint
        // floor, or two of the bands overlap and the ordering above is luck.
        assert!(cuts::WALK > 0.650 && cuts::WALK < 2.200, "the split must part walk from run");
        assert!(cuts::SPRINT > 2.600 && cuts::SPRINT < 3.571, "the floor must part run from sprint");
        assert!(cuts::SPRINT_MAX > 3.640, "a real sprint must not be mistaken for a launch");
    }

    #[test]
    fn sprint_needs_a_straight_line_past_the_run_cap() {
        // #gait-authored  The SPARTAN'S OWN authored sprint top, 2.200 x 1.623 = 3.571 wu/s,
        // dead ahead and level. The old fixtures here were 6.0 and 4.2 wu/s, numbers no Reach
        // player can reach on foot -- 6.0 is now correctly read as a launch.
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [3.571, 0.0, 0.0])], 0.0);
        assert_eq!(derive(&t, 30, None).stance, Stance::Sprint);
        // the same speed SIDEWAYS is not a sprint (Reach cannot strafe while sprinting)
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [0.0, 3.571, 0.0])], 0.0);
        assert_eq!(derive(&t, 30, None).stance, Stance::Combat);
        // and the authored ORDINARY run is not a sprint -- the cut has to separate 2.200 from
        // 3.571, which the old 4.6 floor could not do because it was above both.
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [2.2, 0.0, 0.0])], 0.0);
        assert_eq!(derive(&t, 30, None).stance, Stance::Combat);
        // 13 wu/s across the ground is a man cannon, not a sprint -- measured on a real clip,
        // where the old unbounded rule put a sprint cycle on a launched body.
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [13.0, 0.0, 0.0])], 0.0);
        let st = derive(&t, 30, None);
        assert_eq!(st.stance, Stance::Combat);
        assert!(st.because.contains("NOT a sprint"));
    }

    #[test]
    fn a_body_with_no_decoded_facing_says_so_instead_of_walking() {
        let mut t = track(&[(0, [0.0, 0.0, 0.0]), (60, [4.0, 0.0, 0.0])], 0.0);
        t.facings.clear();
        let st = derive(&t, 30, None);
        assert_eq!(st.provenance, Provenance::Synthesised);
        assert_eq!(st.gait, Gait::Idle);
        assert!(st.because.contains("no decoded body facing"));
    }

    #[test]
    fn a_standing_body_that_spins_turns_rather_than_walking() {
        let mut t = track(&[(0, [0.0, 0.0, 0.0]), (60, [0.05, 0.0, 0.0])], 0.0);
        // Facing samples arrive far more often than once a second, and a yaw rate measured over
        // a whole second is not a turn — so the pair has to be a close one.
        t.facings = vec![(24, 0.0), (36, 0.4)]; // 0.4 rad in 0.2 s = 2.0 rad/s
        assert_eq!(derive(&t, 30, None).gait, Gait::TurnLeft);
        t.facings = vec![(24, 0.0), (36, -0.4)];
        assert_eq!(derive(&t, 30, None).gait, Gait::TurnRight);
    }

    /// The ride detector keys on MATCHED DISPLACEMENT, not proximity — because the measured
    /// distance distributions overlap (a rider sits 0.62-1.55 wu from the vehicle origin, a
    /// non-rider within 4 wu sits at 1.92-3.48). Two bodies running side by side at the same
    /// speed are NOT a ride; a body whose displacement equals the vehicle's is.
    #[test]
    fn the_ride_detector_needs_matched_displacement_not_proximity() {
        use crate::film_decode::{EntityTrack, FilmIndex};
        let pts = |f: &dyn Fn(u32) -> [f32; 3]| -> Vec<(u32, [f32; 3])> {
            (0..=600).step_by(6).map(|t| (t as u32, f(t as u32))).collect()
        };
        // a vehicle driving along +X at 8 wu/s
        let veh = EntityTrack {
            net_id: 100,
            entity_type: VEHICLE_TYPE,
            first_tick: 0,
            last_tick: 600,
            positions: pts(&|t| [t as f32 / 7.5, 0.0, 0.0]),
            ..Default::default()
        };
        // its rider: the same motion, held at a fixed seat offset
        let rider = EntityTrack {
            net_id: 1,
            entity_type: BIPED_TYPE,
            first_tick: 0,
            last_tick: 600,
            positions: pts(&|t| [t as f32 / 7.5 - 0.3, 0.6, 0.4]),
            ..Default::default()
        };
        // a Spartan running alongside at a Spartan's own top speed, close enough the whole time
        // to be inside the distance bound -- which is exactly the case proximity cannot separate.
        let runner = EntityTrack {
            net_id: 2,
            entity_type: BIPED_TYPE,
            first_tick: 0,
            last_tick: 600,
            positions: pts(&|t| [t as f32 / 15.0, 1.2, 0.0]),
            ..Default::default()
        };
        let idx = FilmIndex { ticks: 601, tracks: vec![veh, rider, runner], ..Default::default() };
        let rides = merge_rides(find_rides(&idx));
        assert_eq!(rides.len(), 1, "exactly one ride, got {rides:?}");
        assert_eq!(rides[0].biped_net_id, 1, "the RIDER, not the runner alongside");
        assert_eq!(rides[0].vehicle_net_id, 100);
        assert!(rides[0].seconds() > 8.0, "the whole drive, not a fragment: {:?}", rides[0]);
    }

    /// **Regression, from real data.** Two objects in FREE FALL have identical vertical
    /// displacement, so a match scored on 3D displacement calls them a ride. This is the exact
    /// case that got through: on `asq_mglo-16_forge_hal_28AC2CC7_680042F8` at 12:18.9 a body and a
    /// Scorpion fell past each other 1.4 wu apart, the vehicle's x,y frozen for the whole window,
    /// and the "ride" lasted exactly as long as the co-fall. The numbers below are that capture.
    #[test]
    fn two_objects_in_free_fall_are_not_a_ride() {
        use crate::film_decode::{EntityTrack, FilmIndex};
        // Sampled off the real clip, extended to a full window so the detector gets a fair look.
        let z = |t: u32| 7.75 - 0.5 * ((t - 738) as f32 / 60.0 * 60.0).powi(2) * 0.012;
        let veh = EntityTrack {
            net_id: 0xE9C80055,
            entity_type: VEHICLE_TYPE,
            first_tick: 738,
            last_tick: 900,
            // x,y FROZEN -- the only thing moving is z
            positions: (738..=900).step_by(3).map(|t| (t, [3.20f32, -11.28, z(t)])).collect(),
            ..Default::default()
        };
        let bip = EntityTrack {
            net_id: 0xF2EC0061,
            entity_type: BIPED_TYPE,
            first_tick: 738,
            last_tick: 900,
            // 1.4 wu away in x,y and drifting on its own, falling at the same rate
            positions: (738..=900)
                .step_by(3)
                .map(|t| (t, [2.73f32 - (t - 738) as f32 * 0.004, -9.99 + (t - 738) as f32 * 0.003, z(t) - 0.48]))
                .collect(),
            ..Default::default()
        };
        let idx = FilmIndex { ticks: 901, tracks: vec![veh, bip], ..Default::default() };
        let rides = merge_rides(find_rides(&idx));
        assert!(rides.is_empty(), "free fall is not a ride, got {rides:?}");
    }

    /// One continuous drive arrives as a run of short fragments because the displacement test
    /// goes quiet whenever the vehicle stops. They are one ride.
    #[test]
    fn ride_fragments_separated_by_a_pause_are_merged() {
        let v = vec![
            Ride { biped_net_id: 1, vehicle_net_id: 9, from_tick: 100, to_tick: 160 },
            Ride { biped_net_id: 1, vehicle_net_id: 9, from_tick: 220, to_tick: 300 },
            // a gap of 10 s: a different ride
            Ride { biped_net_id: 1, vehicle_net_id: 9, from_tick: 900, to_tick: 960 },
            // a different vehicle never merges
            Ride { biped_net_id: 1, vehicle_net_id: 8, from_tick: 170, to_tick: 200 },
        ];
        let m = merge_rides(v);
        assert_eq!(m.len(), 3);
        assert_eq!((m[0].from_tick, m[0].to_tick), (100, 300));
    }

    #[test]
    fn seat_families_come_from_the_tag_path_and_unknown_vehicles_get_none() {
        assert_eq!(seat_family_for("objects\\vehicles\\human\\warthog\\warthog"), "warthog");
        assert_eq!(seat_family_for("objects\\vehicles\\covenant\\ghost\\ghost"), "ghost");
        // the civilian vehicles share the truck seat set
        assert_eq!(seat_family_for("objects\\vehicles\\human\\civilian\\pickup\\pickup"), "truck");
        assert_eq!(seat_family_for("objects\\vehicles\\human\\civilian\\forklift\\forklift"), "truck");
        // the Sabre has no seat animations in the Spartan graph: say so rather than invent one
        assert_eq!(seat_family_for("objects\\vehicles\\human\\sabre\\sabre"), "");
    }

    #[test]
    fn a_seated_body_takes_the_vehicles_seat_animation() {
        let t = track(&[(0, [0.0, 0.0, 0.0]), (60, [8.0, 0.0, 0.0])], 0.0);
        let seat = Seat { vehicle_net_id: 7, mode: "warthog_d".into(), vehicle: "warthog".into() };
        let st = derive(&t, 30, Some(seat));
        assert_eq!(st.stance, Stance::Seated);
        let c = anim_candidates(&st, "rifle");
        assert_eq!(c[0], "warthog_d:any:idle");
        assert!(c.contains(&"warthog_d:idle".to_string()));
        assert!(describe(&st).contains("warthog"));
    }

    // ---------------------------------------------------------------------------------------
    // #wclass the per-weapon animation class
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_weapon_class_goes_in_the_middle_slot_and_a_strange_one_still_falls_back() {
        let mut st = BodyState { gait: Gait::RunFront, ..Default::default() };
        st.stance = Stance::Combat;
        // The class the weapon tag names is what the first candidate asks for.
        let c = anim_candidates(&st, "sword");
        assert_eq!(c[0], "combat:sword:move_front_fast");
        // ... and the GENERIC set is still reachable, so a class this biped's graph has no clips
        // for (every vehicle-mounted gun reads `fixed`) can never leave a body frozen.
        assert!(
            c.iter().any(|n| n == "combat:rifle:move_front_fast"),
            "a non-rifle class must still fall back to the generic set: {c:?}"
        );
        // The default class must not append a duplicate tail of itself.
        let r = anim_candidates(&st, DEFAULT_WEAPON_CLASS);
        assert_eq!(r.iter().filter(|n| *n == "combat:rifle:idle").count(), 1, "{r:?}");
    }

    #[test]
    fn a_clips_class_is_its_middle_component() {
        assert_eq!(class_of_clip("combat:sword:idle"), Some("sword"));
        assert_eq!(class_of_clip("combat:landing_dead_back"), None);
        assert_eq!(class_of_clip("first_person:idle"), None);
    }

    #[test]
    fn the_caption_says_whether_the_class_was_read_or_assumed() {
        let mut st = BodyState { gait: Gait::WalkFront, ..Default::default() };
        assert!(describe(&st).contains("no weapon named"), "{}", describe(&st));
        st.weapon_class = Some("hammer".into());
        let d = describe(&st);
        assert!(d.contains("hammer set"), "{d}");
        assert!(d.contains("the weapon the recording names"), "{d}");
    }

    /// #jump THE GATE for "when they jump they look like they're in an A pose".
    ///
    /// `flight:` is the JETPACK mode of the shipped Spartan graph, not "off the ground": it
    /// authors a 70-frame looping HOVER as `flight:<class>:idle` for every weapon class, beside
    /// `flight:rifle:takeoff`/`land` and a `throttle_still_up` overlay. A jump has its own clips
    /// in the combat set, `combat:<class>:airborne_arc` (45f) and `:airborne` (40f).
    ///
    /// The candidate list used to ask for `{mode}:{weapon}:{leaf}` first with `mode = "flight"`,
    /// so a body jumping with little horizontal speed (gait `Idle`) asked for
    /// `flight:<class>:idle`, got the hover, and stood slack in mid-air — while a RUNNING jump
    /// asked for `flight:<class>:move_front_fast`, missed, and fell through to the arc. That is
    /// the reported split exactly. Nothing in a Reach film says who is carrying a jetpack, so the
    /// `flight:` mode must not be asked for at all.
    #[test]
    fn an_airborne_body_asks_for_the_jump_arc_and_never_for_the_jetpack_hover() {
        // Both halves of the split: a standing/crouch jump (no horizontal speed) and a running
        // one. Before the fix the first of these started with `flight:sword:idle`.
        for gait in [Gait::Idle, Gait::RunFront, Gait::Left, Gait::Back] {
            let st = BodyState { stance: Stance::Flight, gait, ..Default::default() };
            for weapon in ["sword", "rifle", "pistol"] {
                let c = anim_candidates(&st, weapon);
                assert_eq!(c.first().map(String::as_str), Some(format!("combat:{weapon}:airborne_arc").as_str()),
                    "an airborne {weapon} body must ask for its jump arc first, got {c:?}");
                assert!(c.iter().any(|n| n == &format!("combat:{weapon}:airborne")),
                    "the 40-frame `airborne` fallback must still be offered: {c:?}");
                assert!(!c.iter().any(|n| n.starts_with("flight:")),
                    "the jetpack mode must never be asked for -- the recording does not say who has one: {c:?}");
            }
        }
    }

    /// #jump THE GATE for the LANDING, the other half of "jumping/falling needs tuning".
    ///
    /// `combat:<class>:land_hard` (13 frames) is authored for every armed class and was never
    /// asked for, so a body snapped straight from the jump arc back to its walk cycle. It is a
    /// one-shot BASE clip driven by the age of the touchdown, and `land_age` reads that age off
    /// the recorded vertical speed because a Reach film states no landing (the action enum that
    /// names `_action_land` is unit field 0x36, which never appears).
    ///
    /// The two halves that matter are that it FIRES on a touchdown and that it RELEASES again:
    /// a landing that never ended would leave the body frozen on the clip's last frame.
    #[test]
    fn a_body_that_has_just_landed_plays_the_landing_and_then_gives_the_body_back() {
        // Falls for half a second, then stands on the floor. 60 Hz ticks.
        let mut pts: Vec<(u32, [f32; 3])> = Vec::new();
        for k in 0..30u32 {
            // -4 wu/s descent, well past the AIRBORNE_VZ cut of 1.6.
            pts.push((k, [k as f32 / 60.0, 0.0, 20.0 - 4.0 * k as f32 / 60.0]));
        }
        let touchdown = 30u32;
        for k in 0..180u32 {
            pts.push((touchdown + k, [(touchdown + k) as f32 / 60.0, 0.0, 18.0]));
        }
        let t = track(&pts, 0.0);

        // Mid-fall: no landing, and the body is airborne.
        assert_eq!(land_age(&t, 15), None, "a body still in the air has not landed");
        assert_eq!(derive(&t, 15, None).stance, Stance::Flight);

        // The window in which the landing owns the body. `velocity_at` is SMOOTHED over +-6
        // ticks, so the touchdown is not a sharp edge and the exact first tick is a property of
        // that smoothing rather than something to assert; what must hold is that it fires soon
        // after the body is down and that it ENDS.
        let fired: Vec<u32> = (touchdown..touchdown + 150).filter(|&k| land_age(&t, k).is_some()).collect();
        assert!(!fired.is_empty(), "a body that fell 2 wu and stopped must play a landing");
        let (first, last) = (fired[0], fired[fired.len() - 1]);
        assert!(first <= touchdown + 14, "the landing must start promptly after touchdown, got +{}", first - touchdown);
        assert!(last < first + cuts::LAND_TICKS + 14,
            "the landing must END -- it ran from +{} to +{}", first - touchdown, last - touchdown);
        // Contiguous: the body must not flicker in and out of its landing.
        assert_eq!(fired.len() as u32, last - first + 1, "the landing window has a hole in it: {fired:?}");

        // While it owns the body it is asked for FIRST, over the gait cycle.
        let mut st = derive(&t, first, None);
        st.land_ticks = land_age(&t, first);
        assert_ne!(st.stance, Stance::Flight, "the body is on the ground once the landing fires");
        let c = anim_candidates(&st, "sword");
        assert_eq!(c.first().map(String::as_str), Some("combat:sword:land_hard"), "{c:?}");

        // ...and afterwards the body is back on its locomotion cycle. Without this it would hold
        // the landing clip's last frame for ever.
        let mut late = derive(&t, last + 1, None);
        late.land_ticks = land_age(&t, last + 1);
        assert_eq!(late.land_ticks, None);
        let c = anim_candidates(&late, "sword");
        assert!(!c.iter().any(|n| n.ends_with(":land_hard")),
            "a body long since down must be back on its locomotion cycle: {c:?}");

        // A body that never left the ground never lands.
        let flat = track(&(0..120u32).map(|k| (k, [k as f32 / 60.0, 0.0, 18.0])).collect::<Vec<_>>(), 0.0);
        assert!((0..100).all(|k| land_age(&flat, k).is_none()), "walking along the floor is not a landing");
    }

    /// #jump-loop THE GATE for "the airborne animation is looping, almost like a player is
    /// jumping repeatedly".
    ///
    /// `combat:<class>:airborne_arc` is 45 frames; a jump lasts as long as the body is off the
    /// ground, which is usually longer. Driven from the locomotion phase — which is
    /// `rem_euclid(1.0)`, so it WRAPS — the arc replayed from frame 0 again and again. It is a
    /// one-shot: it advances to its end and HOLDS until touchdown, then the landing takes over.
    ///
    /// This asserts the frame sequence over a jump LONGER than the clip: monotonically
    /// non-decreasing for the whole airborne span, which a wrap to 0 cannot satisfy.
    #[test]
    fn the_airborne_clip_advances_once_and_holds_instead_of_looping() {
        // A 2-second hang time: far longer than the 45-frame (1.5 s at 30 fps) arc, so a wrapping
        // lane would restart at least once inside it.
        //
        // A long FALL rather than a jump arc, deliberately: at the apex of a jump the vertical
        // speed passes through zero, so `|vz| > AIRBORNE_VZ` is false there and the body reads as
        // grounded for a few ticks in the middle of its own jump. That is a SEPARATE defect in
        // the airborne DETECTION (see the note on `airborne_age`), and letting it into this
        // fixture would make this gate test that instead of the one-shot rule it is for.
        let mut pts: Vec<(u32, [f32; 3])> = Vec::new();
        for k in 0..30u32 {
            pts.push((k, [k as f32 / 60.0, 0.0, 40.0]));          // standing on a ledge
        }
        let takeoff = 30u32;
        for k in 0..150u32 {
            // stepping off and falling: vz is past the cut for the whole span and never reverses
            let s = k as f32 / 60.0;
            pts.push((takeoff + k, [(takeoff + k) as f32 / 60.0, 0.0, 40.0 - 5.0 * s * s]));
        }
        let t = track(&pts, 0.0);

        // The airborne span, and the frame the one-shot lane would drive each tick with.
        // `film_play` computes `frame = age / 2` (60 Hz ticks -> 30 fps frames), held at the
        // clip's last frame; this mirrors that arithmetic over the clip's real length.
        const ARC_FRAMES: f32 = 45.0;
        let mut span: Vec<(u32, f32)> = Vec::new();
        for k in takeoff..takeoff + 150 {
            if let Some(age) = airborne_age(&t, k) {
                span.push((k, (age as f32 / 2.0).min(ARC_FRAMES - 0.001)));
            }
        }
        assert!(span.len() as f32 > ARC_FRAMES * 2.0 + 10.0,
            "this fixture must stay airborne LONGER than the clip, got {} ticks", span.len());

        // THE assertion: never goes backwards. A wrap to frame 0 is exactly what this catches.
        for w in span.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-4,
                "the airborne clip restarted: tick {} frame {:.2} -> tick {} frame {:.2}",
                w[0].0, w[0].1, w[1].0, w[1].1);
        }
        // It really does reach the end and hold there, rather than creeping.
        assert!(span.last().unwrap().1 >= ARC_FRAMES - 0.01, "the arc never reached its last frame");
        assert!(span.first().unwrap().1 < 2.0, "the arc must start at the beginning, got {:.2}", span.first().unwrap().1);

        // And a body standing still is never airborne.
        for k in 0..25u32 {
            assert_eq!(airborne_age(&t, k), None, "tick {k} is standing on the ledge");
        }
    }

    /// #fp-shake THE GATE for "the FP model arms shake or have some sort of waviness".
    ///
    /// A clock-driven clip's frame is half the playhead in ticks. The playhead is a FLOAT advanced
    /// by real frame time, and rounding it to a whole tick first makes the animation step at the
    /// TICK rate instead of the display rate — so the clip holds for some rendered frames and
    /// jumps on others. It is exact at 60 Hz 1x, which is why it survived; it judders at any
    /// higher refresh rate and at any speed below 1x.
    ///
    /// Asserts the property that distinguishes the two: off the float the per-frame advance is
    /// CONSTANT, and off the rounded tick it is not.
    #[test]
    fn a_clock_driven_clip_advances_evenly_at_any_refresh_rate_and_speed() {
        use crate::film_play::clock_frame;
        const N: f32 = 45.0; // combat:<class>:airborne_arc, and about the length of an FP clip
        // (name, playhead advance per RENDERED frame)
        for (name, step) in [
            ("60Hz 1x", 1.0f32),
            ("144Hz 1x", 60.0 / 144.0),
            ("165Hz 1x", 60.0 / 165.0),
            ("60Hz 0.25x", 0.25),
            ("240Hz 0.5x", 0.5 * 60.0 / 240.0),
        ] {
            let mut ph = 100.0f32;
            let mut frames = Vec::new();
            for _ in 0..24 {
                frames.push(clock_frame(ph, N));
                ph += step;
            }
            let deltas: Vec<f32> = frames.windows(2).map(|w| w[1] - w[0]).collect();
            let want = step * 0.5;
            for (i, d) in deltas.iter().enumerate() {
                assert!((d - want).abs() < 1e-3,
                    "{name}: frame {i} advanced {d:.4}, not {want:.4} -- the clip is not running at the display rate");
            }
            // A judder is a VARYING delta; this is the scalar that separates the two forms.
            let spread = deltas.iter().cloned().fold(f32::MIN, f32::max)
                - deltas.iter().cloned().fold(f32::MAX, f32::min);
            assert!(spread < 1e-3, "{name}: per-frame advance varies by {spread:.4} -- that variation IS the shake");

            // The ROUNDED form, which this replaced, fails that at every rate but 60 Hz 1x.
            let mut ph = 100.0f32;
            let mut rough = Vec::new();
            for _ in 0..24 {
                rough.push((ph.round().max(0.0) * 0.5) % N);
                ph += step;
            }
            let rd: Vec<f32> = rough.windows(2).map(|w| w[1] - w[0]).collect();
            let rspread = rd.iter().cloned().fold(f32::MIN, f32::max) - rd.iter().cloned().fold(f32::MAX, f32::min);
            if (step - 1.0).abs() < 1e-6 {
                assert!(rspread < 1e-3, "at 60Hz 1x the rounded form is exact, which is why this was never seen");
            } else {
                assert!(rspread > 0.2, "{name}: the rounded form should judder, spread {rspread:.4}");
            }
        }
    }

    /// #fp-shake A clip frame is a pure function of the playhead: the same position gives the same
    /// frame however it was reached. Scrubbing must not shimmer.
    #[test]
    fn a_clock_driven_clip_frame_is_a_pure_function_of_the_playhead() {
        use crate::film_play::clock_frame;
        for n in [13.0f32, 40.0, 45.0, 156.0] {
            for ph in [0.0f32, 0.5, 1.0, 37.25, 100.0, 1647.75, 10_000.5] {
                assert_eq!(clock_frame(ph, n), clock_frame(ph, n));
                assert!(clock_frame(ph, n) >= 0.0 && clock_frame(ph, n) < n.max(1.0),
                    "frame {} out of range for a {n}-frame clip", clock_frame(ph, n));
            }
            // Negative playheads cannot produce a negative frame.
            assert_eq!(clock_frame(-5.0, n), 0.0);
        }
    }

    /// #sync-rate THE RATE of an assassination, pinned — reported as "too quick or something so
    /// you basically never see them".
    ///
    /// Reach's ground sync moves are authored at **75 frames at 30 fps** against the film's 60 Hz
    /// ticks, which is where [`SYNC_TICKS`] = 150 comes from, and a real chain on the `21897D8A`
    /// clip was traced entering at 1647 and leaving at 1797 — exactly 150. So the window is right
    /// and the question is the playback rate: 75 frames over 150 ticks needs **one frame per two
    /// ticks**, and advancing one frame per TICK would finish the move in 1.25 s and then hold.
    ///
    /// This asserts the arithmetic the one-shot lane actually uses (`frame = age/2`, clamped to
    /// the clip's own last frame) over the whole window, for a 75-frame ground move AND for
    /// `svs_airborne_smackdown`, which is 30 frames rather than 75 — so the rate must come from
    /// each clip's own length, never from a constant.
    ///
    /// **Measured: the rate is correct.** A 75-frame move fills the window exactly, reaching its
    /// last frame at the final tick. The 30-frame airborne move reaches its end after 60 ticks and
    /// then HOLDS for the remaining 90 — which is the opposite of too quick, and says the
    /// remaining complaint is about never landing on one of the three in a seven-minute clip
    /// rather than about the animation.
    #[test]
    fn an_assassination_plays_at_one_frame_per_two_ticks_across_its_whole_window() {
        // What `film_play`'s one-shot lane computes for a clip of `n` frames at age `age`.
        let frame_at = |age: u32, n: f32| ((age as f32) * 0.5).min(n - 0.001).max(0.0);

        // A 75-frame ground move: starts at 0, advances every other tick, and its LAST tick is
        // its last frame -- the move fills the window rather than finishing early.
        const GROUND: f32 = 75.0;
        assert_eq!(frame_at(0, GROUND), 0.0, "the move must start at its first frame");
        let mut prev = -1.0f32;
        for age in 0..SYNC_TICKS {
            let f = frame_at(age, GROUND);
            assert!(f >= prev, "the sync clip went backwards at age {age}");
            prev = f;
        }
        let last = frame_at(SYNC_TICKS - 1, GROUND);
        assert!(last > GROUND - 1.5, "a 75-frame move must still be running at the end of its 150-tick window, got frame {last:.2}");
        // Half way through the window it must be half way through the clip -- the property a
        // per-TICK advance breaks (it would already be at frame 74 by age 75).
        let mid = frame_at(SYNC_TICKS / 2, GROUND);
        assert!((mid - GROUND / 2.0).abs() < 1.0, "half the window should be half the move, got frame {mid:.2}");
        assert!(frame_at(75, GROUND) < GROUND - 30.0,
            "at age 75 a correctly-paced move is barely half done, not finished");

        // `svs_airborne_smackdown` is 30 frames: it must take its OWN length (60 ticks) and then
        // hold, not be stretched or squashed to fit the window.
        const AIR: f32 = 30.0;
        assert!(frame_at(59, AIR) > AIR - 1.5, "the 30-frame move should be at its end by age 59");
        assert_eq!(frame_at(60, AIR), frame_at(SYNC_TICKS - 1, AIR),
            "once finished it HOLDS its last frame for the rest of the window");
        assert!(frame_at(60, AIR) < AIR, "and holds INSIDE the clip, never past its last frame");
    }

    /// #jump-apex THE GATE for "in air animations are different so that needs to be fixed" —
    /// specifically, for a body reading as GROUNDED in the middle of its own jump.
    ///
    /// Airborne is `|vz| > AIRBORNE_VZ`, and at the APEX of a jump the vertical speed passes
    /// through zero. Before [`airborne_at`] bridged it, a real jump arc measured
    /// `Some(11)` at tick 50, `None` across the apex and `Some(1)` again at tick 70 — the body
    /// dropped onto a ground gait mid-flight and the airborne clip restarted.
    ///
    /// This is the fixture the loop gate had to be moved OFF: a full up-and-down jump arc, where
    /// the airborne span must be CONTINUOUS from takeoff to touchdown with no grounded hole, and
    /// the clip must still advance monotonically across the join.
    #[test]
    fn a_jump_stays_airborne_across_its_own_apex() {
        // Ballistic: up at 5 wu/s, down under the recordings' own 10.46 wu/s². Apex at ~0.48 s.
        let mut pts: Vec<(u32, [f32; 3])> = Vec::new();
        for k in 0..40u32 {
            pts.push((k, [k as f32 / 60.0, 0.0, 18.0]));            // running along the floor
        }
        let takeoff = 40u32;
        let hang = 118u32;                                          // ~1.96 s of flight
        for k in 0..hang {
            let s = k as f32 / 60.0;
            pts.push((takeoff + k, [(takeoff + k) as f32 / 60.0, 0.0, 18.0 + 5.0 * s - 5.23 * s * s]));
        }
        for k in 0..120u32 {
            pts.push((takeoff + hang + k, [(takeoff + hang + k) as f32 / 60.0, 0.0, 18.0]));
        }
        let t = track(&pts, 0.0);

        // The apex is where the plain instantaneous test fails; find it and show it is bridged.
        let apex: Vec<u32> = (takeoff..takeoff + hang)
            .filter(|&k| velocity_at(&t, k).is_some_and(|v| v[2].abs() <= cuts::AIRBORNE_VZ))
            .collect();
        assert!(apex.len() >= 8,
            "this fixture must HAVE an apex the instantaneous test calls grounded, got {} ticks", apex.len());
        for k in &apex {
            assert!(airborne_at(&t, *k), "tick {k} is the apex of a jump and must still be airborne");
        }

        // The airborne span is now ONE continuous run with no hole in the middle.
        let span: Vec<u32> = (takeoff..takeoff + hang + 40).filter(|&k| airborne_age(&t, k).is_some()).collect();
        assert!(!span.is_empty(), "the jump must be airborne at all");
        let (first, last) = (span[0], span[span.len() - 1]);
        assert_eq!(span.len() as u32, last - first + 1,
            "the airborne span has a HOLE in it -- the apex gap is back. ticks {first}..{last}, {} of {} present",
            span.len(), last - first + 1);
        assert!(last - first > 90, "the span must cover the real hang time, got {} ticks", last - first);

        // ...and the age rises monotonically across the apex, so the clip never restarts there.
        let mut prev = 0u32;
        for k in first..=last {
            let age = airborne_age(&t, k).expect("continuous");
            assert!(age >= prev, "the airborne age went backwards at tick {k}: {prev} -> {age}");
            prev = age;
        }
    }

    /// #jump-loop The airborne clip must actually REACH the one-shot lane, and in the right
    /// order. The gate above proves the age advances and holds; this proves the age is routed to
    /// the held lane at all — deleting the airborne arm, or putting it ahead of the landing,
    /// fails here. Both halves are needed: an age nothing consumes changes nothing on screen.
    #[test]
    fn the_one_shot_lane_picks_the_right_clip_and_the_landing_beats_the_jump() {
        use crate::film_play::{oneshot_for, ONESHOT_AIR, ONESHOT_DEATH, ONESHOT_LAND, ONESHOT_SYNC};
        // Nothing happening: the body is on its locomotion cycle, which is the ONLY case that
        // may return None. An airborne body returning None here is the looping bug.
        assert_eq!(oneshot_for(None, None, None, None), None);
        // Airborne alone -> the held airborne lane.
        assert_eq!(oneshot_for(None, None, None, Some(7)), Some((7, ONESHOT_AIR)));
        // Landing alone -> the landing.
        assert_eq!(oneshot_for(None, None, Some(3), None), Some((3, ONESHOT_LAND)));
        // Both, which really happens for a few ticks at touchdown because `velocity_at` is
        // smoothed: the LANDING wins, or the body would re-enter its jump as it lands.
        assert_eq!(oneshot_for(None, None, Some(3), Some(40)), Some((3, ONESHOT_LAND)));
        // Death beats either, and a synchronised action beats everything.
        assert_eq!(oneshot_for(None, Some(9), Some(3), Some(40)), Some((9, ONESHOT_DEATH)));
        assert_eq!(oneshot_for(Some(2), Some(9), Some(3), Some(40)), Some((2, ONESHOT_SYNC)));
        // The four kinds are distinct, or two clips would share a lane's bookkeeping.
        let ks = [ONESHOT_SYNC, ONESHOT_DEATH, ONESHOT_LAND, ONESHOT_AIR];
        assert_eq!(ks.iter().collect::<std::collections::HashSet<_>>().len(), 4);
    }

    /// #jump A body ON THE GROUND is unaffected by the fix above: it still asks for its own gait
    /// cycle in the combat set, first and by name. Without this the previous gate could be
    /// satisfied by making everything airborne.
    #[test]
    fn a_grounded_body_still_asks_for_its_own_gait_cycle_first() {
        for (gait, want) in [
            (Gait::Idle, "combat:sword:idle"),
            (Gait::RunFront, "combat:sword:move_front_fast"),
            (Gait::WalkFront, "combat:sword:move_front_slow"),
            (Gait::Back, "combat:sword:move_back"),
            (Gait::Left, "combat:sword:move_left"),
            (Gait::TurnLeft, "combat:sword:turn_left"),
        ] {
            let st = BodyState { stance: Stance::Combat, gait, ..Default::default() };
            let c = anim_candidates(&st, "sword");
            assert_eq!(c.first().map(String::as_str), Some(want), "gait {gait:?} -> {c:?}");
            assert!(!c.iter().any(|n| n.starts_with("flight:")), "{c:?}");
        }
    }

    /// #jump The `crouch:` mode names its cycles DIFFERENTLY from `combat:` — `walk_front` and
    /// `move_front` where combat has `move_front_slow` and `move_front_fast`, and `walk_back`
    /// where combat has `move_back_slow`. Every name this asserts was read off the shipped
    /// Spartan graph (`--dump-anim`), so when the crouch state is decoded the mode is already
    /// asking for clips that exist rather than missing every one of them the way `flight:` did.
    #[test]
    fn the_crouch_mode_asks_for_the_leaf_names_that_mode_actually_authors() {
        for (gait, want) in [
            (Gait::Idle, "crouch:sword:idle"),
            (Gait::WalkFront, "crouch:sword:walk_front"),
            (Gait::RunFront, "crouch:sword:move_front"),
            (Gait::BackSlow, "crouch:sword:walk_back"),
            (Gait::Back, "crouch:sword:move_back"),
            (Gait::Left, "crouch:sword:move_left"),
            (Gait::Right, "crouch:sword:move_right"),
            (Gait::TurnLeft, "crouch:sword:turn_left"),
        ] {
            let st = BodyState { stance: Stance::Crouch, gait, ..Default::default() };
            let c = anim_candidates(&st, "sword");
            assert_eq!(c.first().map(String::as_str), Some(want), "crouch gait {gait:?} -> {c:?}");
            // None of the combat-only leaf names may be asked of the crouch mode.
            for bad in ["crouch:sword:move_front_fast", "crouch:sword:move_front_slow", "crouch:sword:move_back_slow"] {
                assert!(!c.iter().any(|n| n == bad), "`{bad}` is not authored: {c:?}");
            }
        }
        // A CROUCH JUMP keeps the crouch mode's own arc, which the graph does author.
        let st = BodyState { stance: Stance::Crouch, gait: Gait::Idle, ..Default::default() };
        let mut air = st.clone();
        air.stance = Stance::Flight;
        let c = anim_candidates(&st, "sword");
        assert!(!c.iter().any(|n| n.starts_with("flight:")), "{c:?}");
        assert_eq!(anim_candidates(&air, "sword").first().map(String::as_str), Some("combat:sword:airborne_arc"));
    }
}
