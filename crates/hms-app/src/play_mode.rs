//! #play  TEST MODE: spawn as a Spartan or an Elite and walk around the map you are building.
//!
//! This is the app-side half — entering, leaving, the per-frame step, the biped, the weapon in
//! its hands, the first-person rig and the collision triangles. The two halves it stands on are
//! [`crate::play_phys`] (the movement constants, read from the tags) and [`crate::play_ctl`]
//! (the controller, pure over a triangle world).
//!
//! # Almost nothing here is new
//!
//! The Theater work supplies the rest, and it is used through its own public surface rather than
//! copied:
//!
//! * the biped, GPU-skinned and posed — `reach_actor::ActorScene`;
//! * the clip for a state — `film_anim::anim_candidates`, which does not care that the state came
//!   from a keyboard instead of a recording;
//! * the melee swing — the REPLACEMENT lane (`Actor::replacement`), never a base clip, because a
//!   Reach melee animates 34 of a Spartan's 82 nodes and installing one as the sole clip snaps
//!   every other node to the bind pose (that is what made the legs go strange in Theater);
//! * the first-person arms and the gun in them — `matg @player representation` for the arms and
//!   the WEAPON's own `first person animations` graph, which lists both models' nodes and drives
//!   them from one clip, so the grip needs no marker attach;
//! * the spawn point — `spawn_camera::choose`, already per-game and already stood off the walls;
//! * the collision triangles — the level's own grid plus the `phmo` hulls of the placed objects,
//!   the same ones a ragdoll lands on.
//!
//! # The player spawns HOLDING A WEAPON, and that is not a nicety
//!
//! A Reach first-person graph belongs to the WEAPON, not to the arms. With no weapon there is no
//! authored pose at all and the arms draw their bind pose, which is both arms straight out — the
//! exact symptom this project has already had reported once ("the first person model is still
//! bad, its just their arms sticking straight out"). So test mode arms the player, which also
//! gives the melee a weapon class to pick a swing from.
//!
//! # Per-frame cost is one capsule
//!
//! The triangles are gathered into [`PlayWorld`] for a box around the player and re-gathered only
//! when the player leaves the inner sphere of that box — so the object table is walked roughly
//! once a second while running, not once a frame, and the level grid is queried on the same
//! schedule. Steady-state cost is a few dozen sphere-triangle tests for three spheres.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};

use crate::objscene::ObjectScene;
use crate::physics_settle::StaticWorld;
use crate::play_ctl::{self, MoveInput, PlayerState, StepOut};
use crate::play_damage::{DamageHit, DamageSource, ObjectDamage};
use crate::play_phys::{self, PlayerPhysics};
use crate::reach_actor::ActorScene;

/// Which species the player is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Spartan,
    Elite,
}

impl Species {
    pub fn label(self) -> &'static str {
        match self {
            Species::Spartan => "Spartan",
            Species::Elite => "Elite",
        }
    }
    pub fn is_elite(self) -> bool {
        self == Species::Elite
    }
    /// Substrings that name this species' player biped in a Reach cache, best first.
    fn bipd_hints(self) -> &'static [&'static str] {
        match self {
            Species::Spartan => &["spartan_mp", "spartan", "mp_spartan"],
            Species::Elite => &["elite_mp", "elite", "mp_elite"],
        }
    }
}

/// #ride-park  A vehicle left where the player parked it: its ride state, the variant it was
/// drawn with and the actors that draw it.
struct Parked {
    ride: crate::play_vehicle::Riding,
    variant_sid: u32,
    actors: Vec<u32>,
    /// Where the actors' meshes (and so their baked object lighting) were last built.
    baked: Vec3,
}

/// Which camera the player is looking through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayView {
    /// Through the biped's own eyes, at the authored camera height, with the arms and the gun on
    /// the rig. The body itself is parked out of the shot — the engine does not draw you the
    /// inside of your own helmet, and an unposed torso across the middle of the view is worse
    /// than nothing.
    FirstPerson,
    /// Behind and above the biped, so you can watch the Spartan you are driving. Useful for
    /// checking a jump against a block you just placed.
    ThirdPerson,
}

/// Where a hidden actor waits: far below the map, out of every shot. The same absolute point
/// Theater parks unused actors at (`film_play::ActorDriver::park`), and absolute rather than an
/// offset from the player so that moving does not drag a 10,000 wu outlier around with the camera.
pub const PARK: Vec3 = Vec3::new(0.0, 0.0, -10_000.0);

/// How far behind the eye the third-person camera sits, wu, and how far above it.
///
/// **Unmeasured**: Reach has no third-person player camera to read a distance off (its only
/// third-person cameras are the vehicle ones, authored per seat in `vehi`). These are a framing
/// choice for a view the engine does not have, which is why they are two named constants with
/// this note and not scattered numbers.
pub const THIRD_PERSON_BACK: f32 = 1.6;
pub const THIRD_PERSON_UP: f32 = 0.25;

/// How far in front of the eye a melee swing looks for something to hit when the weapon authors
/// no melee parameters at all, wu.
///
/// **Every armed swing uses the AUTHORED range instead** — `weap` +0x2AC `melee damage
/// parameters` `maximum lunge range`, read by `reach_damage::melee_parameters`, which also
/// carries the engine's own 1.22 wu default for a tag that authors 0. This value is only the
/// unarmed case, where there is no weapon to read a range from.
pub const MELEE_REACH_UNARMED_WU: f32 = 0.8;

/// Half-extent of the triangle box gathered around the player, wu.
pub const WORLD_BOX: f32 = 6.0;
/// #play  How far below the map's own bounds the body must fall before the failsafe returns it.
///
/// **A UX guard, not physics.** Reach authors no terminal velocity for a biped — the physics side
/// measured that `bipd +0x2FC` is an animation weight, not a speed cap — so nothing here may be
/// presented as the engine's. This exists only so that "I fell out of the world" becomes "you
/// fell out of the world, here you are again" instead of an endless drop.
pub const FAILSAFE_BELOW_BOUNDS: f32 = 20.0;

/// #play  The most sub-ticks one frame may be split into. 32 x half the re-gather margin is ~48 wu
/// of travel in a single frame, which nothing in this mode reaches; it bounds the loop, not the
/// physics.
pub const TICK_SPLIT_GUARD: usize = 32;

/// The player may travel this far from the box's centre before it is re-gathered, wu. A run at
/// the authored ~4 wu/s crosses it in about three quarters of a second, so the object table is
/// walked about once a second at a sprint and not at all while standing still.
pub const WORLD_REGATHER: f32 = 3.0;

/// The triangles the player collides with: the level's plus the nearby objects', gathered once
/// per neighbourhood rather than once per frame.
#[derive(Default)]
pub struct PlayWorld {
    tris: Vec<[Vec3; 3]>,
    centre: Vec3,
    /// False until the first gather.
    valid: bool,
    /// Per-datum object hulls, flattened — the same cache shape the ragdoll lane uses.
    obj: HashMap<u32, Arc<Vec<Vec3>>>,
    /// #collision  Per-datum closed shells (the objects whose physics is boxes / polyhedra), and
    /// the ones near the player. A body inside one is pushed out the nearest way.
    obj_shells: HashMap<u32, Vec<Arc<crate::physics_settle::Shell>>>,
    shells: Vec<Arc<crate::physics_settle::Shell>>,
    /// #collision  `obje tag -> does its physics collide with a biped`, memoised. The answer is a
    /// property of the TAG, not of the placement, so it is read once per tag rather than once per
    /// object per gather.
    collide: HashMap<u32, bool>,
    /// How many times the neighbourhood has been re-gathered, and how many triangles it holds.
    pub gathers: usize,
    /// **Counted SEPARATELY and never summed into one "has collision" bool.** "No level geometry
    /// here" and "no object hull passed" are different faults with different fixes, and a single
    /// count cannot tell them apart — the same mistake as a load status that cannot distinguish
    /// "not started" from "in flight".
    pub level_tris: usize,
    pub object_tris: usize,
    /// Objects the COLLISION FILTER rejected — spawn points, respawn zones, named-location
    /// areas, safe and kill volumes. Before the filter existed every one of these was solid, so a
    /// dense Forge arena was full of invisible walls: "we bump into the simplest of things".
    pub objects_filtered: usize,
    /// Placed objects near enough to matter, and how many of them yielded NO physics triangles.
    /// A Forge block whose `phmo` does not resolve is invisible to collision: the player stands
    /// on it in the picture and falls through it in the simulation.
    pub objects_near: usize,
    pub objects_without_hulls: usize,
    /// #collision  Datums whose PLACEMENT is not where the object is (a vehicle being driven or
    /// parked elsewhere, a prop that has been knocked off its saved pose): their placement hull
    /// is an invisible block at the old spot, so they are left out of the gather.
    skip: Vec<u32>,
}

impl StaticWorld for PlayWorld {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        StaticWorld::gather(self.tris.as_slice(), mn, mx, out);
    }
    fn solid_shells(&self, mn: Vec3, mx: Vec3, out: &mut Vec<Arc<crate::physics_settle::Shell>>) {
        out.extend(self.shells.iter().filter(|s| (0..3).all(|k| s.mn[k] <= mx[k] && s.mx[k] >= mn[k])).cloned());
    }
}

impl PlayWorld {
    /// Set the datums to leave out of the gather; a change makes the cached neighbourhood stale.
    pub fn set_skip(&mut self, mut skip: Vec<u32>) {
        skip.sort_unstable();
        skip.dedup();
        if skip != self.skip {
            self.skip = skip;
            self.valid = false;
        }
    }

    /// Is the cached neighbourhood still the right one for a player at `pos`?
    pub fn covers(&self, pos: Vec3) -> bool {
        self.valid && (pos - self.centre).length() <= WORLD_REGATHER
    }

    /// Re-gather around `pos`. The only place in test mode that touches the object table.
    pub fn regather(
        &mut self,
        pos: Vec3,
        scene: &dyn ObjectScene,
        objects: &[ObjectInfo],
        native: &NativeDll,
        cache: CacheHandle,
    ) {
        let pad = Vec3::splat(WORLD_BOX);
        let (mn, mx) = (pos - pad, pos + pad);
        self.tris.clear();
        self.shells.clear();
        scene.static_tris_in_aabb(mn, mx, &mut self.tris);
        self.level_tris = self.tris.len();
        // Objects whose own origin is anywhere near the box. Their hulls are cached per datum, so
        // a re-gather on a map the player has already walked is a filter rather than a decode.
        let near = WORLD_BOX + 12.0;
        self.objects_near = 0;
        self.objects_without_hulls = 0;
        self.objects_filtered = 0;
        for o in objects {
            let p = Vec3::from(o.pos);
            if (p - pos).length() > near || self.skip.binary_search(&o.datum).is_ok() {
                continue;
            }
            self.objects_near += 1;
            // #collision  THE FILTER. `reach_collision::collides_with_biped` answers from the
            // tag: no physics geometry, or an authored multiplayer REGION type, means it does not
            // collide — everything else is solid, including objects with no multiplayer block at
            // all, because 204 of 496 physics-bearing objects have none and they are vehicle
            // parts (`hull_shell`, `seat`, `bumper`, `hubcap`). Defaulting those to pass-through
            // would have ghosted every vehicle. The type -> region mapping is an INFERENCE and
            // this is the single place it is consulted, so if the engine's own collision group
            // refutes it, exactly one call changes.
            let solid = match self.collide.get(&o.primary_tag) {
                Some(v) => *v,
                None => {
                    let v = crate::reach_collision::collides_with_biped(native, cache, o.primary_tag);
                    self.collide.insert(o.primary_tag, v);
                    v
                }
            };
            if !solid {
                self.objects_filtered += 1;
                continue;
            }
            let tris = match self.obj.get(&o.datum) {
                Some(t) => t.clone(),
                None => {
                    let flat: Vec<Vec3> = scene.physics_world_tris(o).into_iter().flatten().collect();
                    let a = Arc::new(flat);
                    self.obj.insert(o.datum, a.clone());
                    a
                }
            };
            if tris.is_empty() {
                self.objects_without_hulls += 1;
            }
            let shells = match self.obj_shells.get(&o.datum) {
                Some(v) => v.clone(),
                None => {
                    let v: Vec<_> = scene.physics_world_shells(o).into_iter().map(|t| Arc::new(crate::physics_settle::Shell::new(t))).collect();
                    self.obj_shells.insert(o.datum, v.clone());
                    v
                }
            };
            self.shells.extend(shells.into_iter().filter(|s| (0..3).all(|k| s.mn[k] <= mx[k] && s.mx[k] >= mn[k])));
            for c in tris.chunks_exact(3) {
                let t = [c[0], c[1], c[2]];
                if crate::physics_settle::tri_overlaps(&t, mn, mx) {
                    self.tris.push(t);
                }
            }
        }
        self.object_tris = self.tris.len() - self.level_tris;
        self.centre = pos;
        self.valid = true;
        self.gathers += 1;
    }

    /// Nothing at all to collide with here. **A body over a void falls for ever and, before this
    /// existed, said nothing while doing it** — which is one of the two ways "I fell straight
    /// through the entire map" happens.
    pub fn is_void(&self) -> bool {
        self.valid && self.tris.is_empty()
    }

    /// One line about what is under the player, for the caption and the live gate.
    pub fn describe(&self) -> String {
        format!(
            "collision: {} level + {} object triangle(s) from {} nearby object(s){}{} | {} gather(s)",
            self.level_tris,
            self.object_tris,
            self.objects_near,
            if self.objects_filtered > 0 {
                format!(
                    ", {} filtered out as non-colliding (spawn points, zones, volumes)",
                    self.objects_filtered
                )
            } else {
                String::new()
            },
            if self.objects_without_hulls > 0 {
                format!(
                    ", {} of which resolved NO physics hull (those are solid in the picture and \
                     empty in the simulation)",
                    self.objects_without_hulls
                )
            } else {
                String::new()
            },
            self.gathers
        )
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }
    /// Forget everything — the objects moved, or a different map opened.
    pub fn invalidate(&mut self) {
        self.valid = false;
        self.obj.clear();
        self.obj_shells.clear();
    }
}

/// #use-hold  What [`PlayMode::use_decide`] tells the host to do with the use button this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UseDecision {
    Idle,
    /// Leave the seat (a press, seated).
    Exit,
    /// The hold completed with a seat or a weapon in range: act on it.
    Hold,
    /// A press with nothing in range: report why.
    Nothing,
}

/// The editor camera at the moment Play was pressed — **an INPUT**, used to place the player.
///
/// Note what is not here: a saved pose to put the camera back to. Leaving test mode deliberately
/// leaves the camera where the player walked to, so retaining the entry pose would be dead state
/// encoding a REJECTED behaviour — and the next reader to find a `restore.pos` beside a
/// `leave_play` that ignores it would reasonably conclude the exit was broken and "fix" it,
/// reinstating the exact teleport the user asked us to remove. The only thing that survives entry
/// is the field of view, which is a property of the rig rather than a place anyone navigated to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntryCamera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    /// The editor's vertical field of view, restored on leaving (the rig changes it).
    pub fov_y: f32,
}

/// A live test-mode session.
pub struct PlayMode {
    pub phys: PlayerPhysics,
    pub st: PlayerState,
    pub species: Species,
    pub view: PlayView,
    pub biped_tag: u32,
    pub biped_path: String,
    /// The weapon in the player's hands, and its ANIMATION CLASS (`rifle`, `pistol`, …) read off
    /// the weapon tag. The class is what picks every locomotion and melee clip.
    pub weapon_tag: u32,
    pub weapon_path: String,
    pub weapon_class: Option<String>,
    /// The body actor, and the first-person actors (arms, then the gun).
    pub body: Option<u32>,
    pub fp: Vec<u32>,
    /// #fp-legs  The `@player representation` FIRST-PERSON BODY actor — the legs and hips the
    /// player sees when they look down — or `None` when this cache authors none.
    ///
    /// It is a separate model from the third-person biped and from the arms, and it is the
    /// engine's own answer to "what do you see of yourself": measured on `forge_halo`, the
    /// Spartan's `fp_body` carries 5 meshes over the regions `decals` and `flair_knees` (no
    /// head, no arms, no torso) on the SAME 82-node skeleton as the full biped, and the biped's
    /// own animation graph reaches 100% of its nodes while the weapon's first-person graph
    /// reaches 0%. So it poses off the body's locomotion clip, at the player's feet, and the
    /// third-person biped is parked while it is drawn.
    pub fp_body: Option<u32>,
    /// `(hands render model, first-person graph)` for the arms rig, when one resolved.
    fp_rig: Option<(u32, Option<u32>)>,
    /// #play-rate  The locomotion cycle's phase, in CYCLES, wrapped to `[0, 1)`.
    ///
    /// **Integrated, not recomputed.** It used to be `(total distance * cycles_per_wu) % 1`,
    /// which is only continuous while `cycles_per_wu` never changes — and it changes at every
    /// change of gait and stance. With 40 wu of travel behind you, a walk-to-run switch moved
    /// the product by tens of whole cycles and the pose SNAPPED to an unrelated point in the
    /// stride. Theater hit the identical defect and fixed it the identical way (see
    /// `film_anim::cycle_phase`, which integrates `distance x rate` segment by segment); test
    /// mode had re-introduced it.
    cycle: f32,
    /// The ground distance the cycle has already been advanced for, so one frame advances it by
    /// the ground that frame covered and no more.
    cycle_distance: f32,
    /// #play-jolt  The TURN-IN-PLACE cycle's phase, in cycles, and the facing it was last
    /// advanced from.
    ///
    /// A turn is driven by **accumulated yaw**, for the same reason a stride is driven by
    /// accumulated distance: the clip is authored to carry the body through a certain angle, so
    /// indexing it on anything else makes the legs step at one rate while the body yaws at
    /// another. It used to run on the WALL CLOCK, which also meant a turn was entered at
    /// `(session seconds x 30) mod 18` — an arbitrary point in the step. Reported: *"when
    /// turning... its doing a sudden jolt of the legs instead of cleanly interpolating the legs as
    /// we turn"*.
    ///
    /// `None` means the body is not in a turn, and entering one resets the phase to 0 so the step
    /// starts at the beginning of the step.
    turn_cycle: f32,
    turn_from: Option<f32>,
    pub world: PlayWorld,
    /// The editor's own field of view, to put back on leaving. **Only** the FOV: see
    /// [`EntryCamera`] for why no entry POSE is kept.
    pub restore_fov: f32,
    /// Seconds into a melee swing, and the clip's own length in seconds.
    melee: Option<(f32, f32)>,
    melee_variant: u8,
    /// The held weapon's AUTHORED `maximum lunge range`, wu (`None` = unarmed / unauthored).
    melee_reach: Option<f32>,
    /// The held weapon's authored melee damage pyramid: (yaw, pitch) half-angles in radians and
    /// its depth, for the caption. Read, reported, and NOT reduced to a ray by this module.
    melee_pyramid: Option<([f32; 2], f32)>,
    /// Seconds since the trigger was pulled, while the shot still owns the arms.
    fired: Option<f32>,
    /// Shots fired this session, for the caption.
    pub shots: u32,
    /// #weapon-fire  The held weapon's firing machine -- the engine's per-tick trigger / barrel /
    /// magazine state machines ([`crate::play_fire`]). `None` for empty hands.
    pub weapon_fire: Option<crate::play_fire::WeaponFire>,
    /// Where the player was first placed. **Used ONLY by the out-of-world failsafe** — this is
    /// not the rejected "put the camera back where you pressed Play" state, which was deleted; it
    /// is a recovery anchor, read every frame the body is below the map.
    spawn_feet: Vec3,
    /// Faults worth printing: how many times the failsafe has fired, and how many frames the body
    /// has spent over a void with nothing at all to collide with.
    pub failsafe_returns: u32,
    pub void_frames: u32,
    /// The third-person camera, solved in `tick` where the scene is available so it can be kept
    /// out of the walls.
    third_person_cam: Vec3,
    /// What the last swing hit, for the caption.
    pub last_hit: Option<String>,
    pub last_step: StepOut,
    /// The clip the body is playing, read back off the actor rather than off the intent.
    /// The base clip last ASKED FOR (`anim_candidates`'s first entry), which is not always the
    /// one that loaded — see [`pose`](Self::pose).
    asked: Option<String>,
    /// #gait-blend  The authored locomotion table for the `(mode, weapon class)` in force, and
    /// the two labels it was resolved for. Resolved once per change, never per frame: it walks
    /// the graph's `sets` block.
    gait_table: Option<(String, String, crate::reach_actor::GaitTable)>,
    /// #crouch-rate  The stance the body was in on the PREVIOUS frame, so a change into or out
    /// of the crouch can be given the biped's own authored transition time.
    last_stance: crate::film_anim::Stance,
    /// #vehicle-ride  The editor's own PER-DATUM appearance maps, as the host holds them:
    /// `(forge colour (team, colour), scale multiplier, damage state)` keyed by object datum.
    ///
    /// Written by [`PlayMode::set_object_appearance`] once a frame and read only by
    /// [`PlayMode::enter_vehicle`], which is the single place that still has the PLACEMENT in
    /// hand. Carried rather than threaded through `mount`'s signature because the smaller shape
    /// is the one that survives: a setter the host already calls cannot drift out of step with
    /// the maps, where a fourth and fifth parameter on a mount path would.
    appearance_of: std::collections::HashMap<u32, (Option<(u8, u8)>, f32, u8)>,
    /// #gait-blend  One line naming this frame's mix, for the status caption. Empty when the
    /// body is not on a blended locomotion cycle (idle, airborne, seated, a turn in place) or
    /// when the graph authors no gaits and the single-clip path is in force.
    pub gait_caption: String,
    /// #gait-blend  The stride (wu per cycle) the blended cycle's phase was advanced by this
    /// frame, or `None` when no blend plan was in force (the single-clip rate applies). Read by
    /// the snap gate, which must compare the phase step with the stride actually in force.
    pub gait_stride: Option<f32>,
    pub body_clip: String,
    pub body_replacement: String,
    /// #aim  True when the biped's graph has no aim overlay this frame, so the body really is
    /// posed straight ahead. Reported rather than hidden, because "stuck aiming straight ahead"
    /// is what a missing overlay looks like and the player should be told which it is.
    pub aim_missing: bool,
    /// #aim  The BEST candidate name the overlay walk was last run for, and which of the list
    /// answered (`None` = none did).
    ///
    /// The same memo the base clip uses, for the same reason: walking the candidate list means a
    /// `graph.find` per name over 979 named entries, and a biped whose graph has NO aim overlay
    /// would do the whole walk every frame for ever. Keyed on what was ASKED FOR rather than on
    /// what loaded, so a steady state costs one string compare.
    aim_asked: Option<String>,
    aim_use: Option<String>,
    pub fp_clip: String,
    /// As [`asked`](Self::asked), for the first-person rig.
    fp_asked: Option<String>,
    /// #fp-legs  As [`asked`](Self::asked), for the first-person LEGS — the clip name last
    /// installed on them, which is the body's own loaded name.
    fp_legs_asked: Option<String>,
    /// Notes raised at entry — what was read, what was not, what is unmeasured.
    pub notes: Vec<String>,
    /// #play  Seconds of test mode elapsed — the clock a clip with nothing else to drive it
    /// runs on (an idle, a turn, the airborne arc, the first-person arms).
    ///
    /// **Accumulated real time, not a frame count.** Reach's clips are authored at 30 fps against
    /// the wall clock, so driving them from the frame counter made their speed follow the display:
    /// an idle would run at half speed on a 30 Hz frame and at double on a 120 Hz one. `dt` is
    /// already clamped by the controller, so a stalled frame cannot jump a clip either.
    pub clock: f32,
    /// #play  The input the WINDOW last read through the binding table. The `play step` verb
    /// replays it, which is what lets a gate hold a real key and then advance deterministically
    /// without ever building an input of its own.
    pub last_input: MoveInput,
    /// #play-sound  The body's locomotion clip and the frame it was posed at this tick: the
    /// engine's foot-down and foley EVENTS are keyed by this clip's frame.
    pub body_clip_state: crate::play_sound::BodyClip,
    /// #seat-entry  The ridden placement's own `variant_name_sid`, captured on entry — it picks
    /// which of a Warthog's four weapon attachments is drawn.
    riding_variant_sid: u32,
    /// #seat-entry  The actors drawing the ridden vehicle — its own render model plus each of
    /// its attachments. Empty on foot.
    ///
    /// **The placement is hidden while it is ridden, so everything the placement drew has to be
    /// drawn here or the vehicle vanishes.** A Warthog is two models (body + turret) and a Falcon
    /// is seven, so this is a list and not an `Option`.
    vehicle_actors: Vec<u32>,
    /// #play-world  Every other physics object, simulated (`play_world`). Public so the app can
    /// hand it the Forge placement bytes when test mode starts.
    pub props: crate::play_world::PropSession,
    /// #ride-park  Vehicles the player has driven and got out of. **They stay where they were
    /// left**: a ridden vehicle is drawn by test mode itself (its placement is hidden), so
    /// removing the actors on exit would put the editor's placement back at its ORIGINAL spot --
    /// the vehicle you just parked would vanish and a fresh one appear where it started.
    parked: Vec<Parked>,
    /// #ride-grip  Each hand's grip on the steering node, in the MARKER's frame: `[left, right]`.
    /// Measured once per ride, from the seat's own authored idle pose, on the first seated frame
    /// (the wheel is at rest then, so the target IS the authored hand).
    grip_ref: [Option<glam::Mat4>; 2],
    /// #ride-camera  The camera the window was last handed, which the next mode change blends FROM.
    cam_prev: Option<crate::reach_unit_camera::CamPose>,
    /// #ride-camera  The engine's camera MODE this frame, and the observer-style blend between two.
    cam_mode: crate::reach_unit_camera::CamMode,
    cam_blend: crate::reach_unit_camera::CameraBlend,
    /// #ride-camera  The pose to render while a transition is running OR a ride is on; `None`
    /// leaves [`PlayMode::camera`] on the plain on-foot formula, byte-identical to before.
    cam_final: Option<crate::reach_unit_camera::CamPose>,
    /// #ride-camera  This frame's seat shot, produced by `drive_and_seat`.
    seat_shot: Option<crate::play_vehicle::SeatShot>,
    /// #seat-entry  The seat's authored `override fov` in radians, when it authors one. `App`
    /// owns the field of view, so this is reported rather than applied here.
    seat_fov: Option<f32>,
    /// #seat-entry  The seat the player is in, or `None` — and `None` leaves every existing
    /// path byte-identical: the controller steps, the camera is the biped's, and nothing in
    /// this module behaves differently from before vehicles existed.
    pub riding: Option<crate::play_vehicle::Riding>,
    /// #vehicle-dup  One line from [`crate::play_vehicle::duplicate_report`], recomputed in
    /// `tick` where the object set is in hand. **The LINE is stored, not the object set**: a
    /// second full copy of every placement per frame is the cost `#gfx-perf` already removed once.
    dup_report: String,
    /// #seat-entry  The seat the player could get into right now, recomputed each frame from
    /// the seats' own authored entry tests. `None` = no prompt is drawn.
    pub offer: Option<crate::play_vehicle::VehicleOffer>,
    /// #pickup  The weapon the player could pick up off the map right now, with its authored
    /// prompt. Recomputed every frame on foot, like the seat offer, and `None` while riding.
    pub pickup: Option<crate::play_pickup::PickupOffer>,
    /// #pickup  The map's own `weap` tags, `(tag, path)` — read ONCE at entry and reused, because
    /// the pickup search runs every frame and the tag table does not change under it.
    pub map_weapons: Vec<(u32, String)>,
    /// #seat-entry  Per-`vehi`-tag cache of seats, markers, physics and the drive model.
    vehicles: crate::play_vehicle::VehicleIndex,
    /// #seat-entry  The label of the key bound to the enter/exit action, substituted for the
    /// engine's own button glyph in the prompt. Supplied by the window, which owns the bindings.
    pub action_key_label: String,
    /// #use-hold  How long the use button has been held, in 60 Hz ticks, and the authored time it
    /// has to reach (`matg` `minimum action hold time`). A seat or a weapon on the ground acts on
    /// the HOLD, never on the press -- see [`crate::reach_action_hold`].
    pub use_hold: crate::reach_action_hold::HoldTimer,
    pub hold_cfg: crate::reach_action_hold::ActionHold,
    /// #use-hold  May the next press LEAVE the seat? False from the hold that got in until the
    /// button has been seen released: the key's auto-repeat while the player is still holding it
    /// down would otherwise throw them straight back out.
    exit_armed: bool,
    /// Frames of test mode, for the status line. **Not used to mean TIME** — see
    /// [`clock`](Self::clock); a clip driven off a frame count plays at the display's rate rather
    /// than the authored one.
    pub frames: u64,
}

impl PlayMode {
    /// Start test mode. `camera` is the editor camera, used as the spawn when the map has no
    /// spawn point of its own.
    ///
    /// `Err` with a reason rather than a half-working mode: a missing `matg` row or an
    /// unreadable `bipd` means the movement constants are unknown, and test mode will not invent
    /// them.
    #[allow(clippy::too_many_arguments)]
    pub fn enter(
        native: &NativeDll,
        cache: CacheHandle,
        scene: &dyn ObjectScene,
        objects: &[ObjectInfo],
        actors: &mut ActorScene,
        species: Species,
        camera: EntryCamera,
    ) -> Result<PlayMode, String> {
        let bipeds = ActorScene::list_bipeds(native, cache);
        if bipeds.is_empty() {
            return Err("this map has no biped ('bipd') tags, so there is nobody to play as".into());
        }
        let (biped_tag, biped_path, row, how) = resolve_species(native, cache, &bipeds, species)?;
        let phys = PlayerPhysics::read(native, cache, biped_tag, row)?;

        let mut notes = Vec::new();
        notes.push(format!("playing as {} ({biped_path}), row {row} — resolved from {how}", species.label()));
        notes.push(phys.describe());
        notes.push(play_ctl::law_summary(&phys));

        // --- where to stand ---------------------------------------------------------------
        //
        // **THE EDITOR CAMERA, not the map's spawn point.** Reported: "we also spawn way outside
        // the actual map variant, we need to create the spartan biped right where the camera
        // was." That is right, and the earlier order was wrong: the user flies the camera to the
        // part of their build they want to test and then presses Play, so the camera's position
        // IS the intent. A scenario spawn point answers a different question — where a match
        // would start — and on a Forge variant that can be nowhere near the structure being
        // built. The spawn point is kept only as the fallback for a caller with no camera pose.
        let (eye, yaw, pitch, src) = (
            camera.pos,
            camera.yaw,
            camera.pitch,
            "the editor camera (where you were looking from when you pressed Play)",
        );
        let mut feet = eye - Vec3::Z * phys.eye_standing;
        // ...and DROPPED TO THE GROUND. A camera flies; a biped does not. The probe starts at the
        // camera itself rather than at the derived feet so that a camera hovering high over the
        // map still lands the player on what is underneath it, and it reaches far enough to cover
        // a camera parked well above a Forge structure.
        const DROP: f32 = 500.0;
        let mut landed = false;
        if let Some(d) = scene.raycast_solid(eye, -Vec3::Z, DROP) {
            feet.z = eye.z - d;
            landed = true;
        }
        notes.push(format!(
            "spawned at {src}: ({:.2}, {:.2}, {:.2}){}",
            feet.x,
            feet.y,
            feet.z,
            if landed {
                format!(" — dropped {:.2} wu onto the surface under the camera", eye.z - feet.z - phys.eye_standing)
            } else {
                format!(" — nothing solid within {DROP:.0} wu below the camera, so the player starts at the camera's own height and will fall")
            }
        ));

        let mut world = PlayWorld::default();
        world.regather(feet, scene, objects, native, cache);
        if world.is_empty() {
            notes.push(
                "WARNING: no collision triangles at the spawn — this scene provides none, so the \
                 player would fall through the map. Test mode needs the level's own triangles."
                    .into(),
            );
        } else {
            notes.push(format!(
                "collision: {} level + {} object triangles in the {:.0} wu box around the player",
                world.level_tris, world.object_tris, WORLD_BOX
            ));
        }

        // --- the weapon, BEFORE the body, because the body's clips are keyed on its class ----
        let weapons = ActorScene::list_weapons(native, cache);
        // PREFER A WEAPON THAT CAN ACTUALLY POSE THE ARMS. The first-person graph belongs to the
        // WEAPON, so a weapon authoring no `player interface` leaves the arms with no pose at all
        // — and the rig then declines to draw them, which reads as "the weapon isn't even in my
        // hands". So the candidates are tried in order and the first one that resolves a
        // first-person graph wins; only if none does do we fall back to the first by name, and the
        // notes say which happened.
        let species_row = usize::from(species.is_elite());
        let mut weapon_tag = 0u32;
        let mut weapon_path = String::new();
        let mut fp_capable = false;
        for hint in ["assault_rifle", "rifle", "pistol", "magnum", "shotgun"] {
            let Some((t, path)) = pick_by_hints(&weapons, &[hint]) else { continue };
            let (_, graph) = crate::reach_actor::first_person_weapon(native, cache, t, species_row);
            if graph.is_some() {
                weapon_tag = t;
                weapon_path = path;
                fp_capable = true;
                break;
            }
            if weapon_tag == 0 {
                weapon_tag = t;
                weapon_path = path;
            }
        }
        let weapon_class = if weapon_tag != 0 {
            crate::reach_actor::weapon_class(native, cache, weapon_tag)
        } else {
            None
        };
        if weapon_tag == 0 {
            notes.push(
                "no weapon in this map's tags, so the first-person arms have no authored pose \
                 (the first-person graph belongs to the WEAPON) and the melee has no class"
                    .into(),
            );
        } else {
            notes.push(format!(
                "holding {weapon_path} (animation class {}){}",
                weapon_class.clone().unwrap_or_else(|| crate::film_anim::DEFAULT_WEAPON_CLASS.into()),
                if fp_capable {
                    " — it authors a first-person interface, so the arms have a real pose"
                } else {
                    " — WARNING: no weapon in this map authored a first-person interface, so the                      arms cannot be posed and are not drawn; the weapon is still in the biped's                      hand and visible in both views"
                }
            ));
        }

        // `PlayerState` has private fields, so it is built by its own constructor and then
        // adjusted — a struct-update expression here stops compiling the moment the controller
        // gains an internal field, which it has.
        // ...and then OUT of anything solid: a camera parked inside a block drops a body that
        // starts inside it, and the ground probe above starts from the camera, inside the hull.
        let st = {
            let mut st = PlayerState::standing_at(feet, yaw);
            st.pitch = pitch;
            let moved = st.settle_out(&phys, &world);
            if moved > 1e-3 {
                notes.push(format!("the spawn point was inside solid geometry: moved {moved:.2} wu to where a body fits"));
            }
            st
        };
        let feet = st.pos;
        let mut pm = PlayMode {
            phys,
            st,
            species,
            view: PlayView::FirstPerson,
            biped_tag,
            biped_path,
            weapon_tag,
            weapon_path,
            weapon_class,
            body: None,
            fp: Vec::new(),
            fp_body: None,
            fp_rig: None,
            cycle: 0.0,
            cycle_distance: 0.0,
            turn_cycle: 0.0,
            turn_from: None,
            world,
            restore_fov: camera.fov_y,
            melee: None,
            melee_variant: 0,
            melee_reach: None,
            melee_pyramid: None,
            fired: None,
            shots: 0,
            spawn_feet: feet,
            failsafe_returns: 0,
            void_frames: 0,
            third_person_cam: feet,
            last_hit: None,
            last_step: StepOut::default(),
            riding_variant_sid: 0,
            vehicle_actors: Vec::new(),
            props: Default::default(),
            parked: Vec::new(),
            grip_ref: [None, None],
            cam_prev: None,
            cam_mode: crate::reach_unit_camera::CamMode::Foot,
            cam_blend: Default::default(),
            cam_final: None,
            seat_shot: None,
            seat_fov: None,
            riding: None,
            dup_report: String::new(),
            offer: None,
            pickup: None,
            // #pickup  The map's weapon tags, read once here. The same list the weapon choice
            // above was made from, so the search cannot disagree with the menu about what a
            // weapon is.
            map_weapons: weapons.clone(),
            vehicles: Default::default(),
            action_key_label: "E".to_string(),
            use_hold: Default::default(),
            hold_cfg: crate::reach_action_hold::ActionHold::read(native, cache),
            exit_armed: false,
            asked: None,
            gait_table: None,
            last_stance: crate::film_anim::Stance::Combat,
            appearance_of: Default::default(),
            gait_caption: String::new(),
            gait_stride: None,
            body_clip: String::new(),
            body_replacement: String::new(),
            aim_missing: false,
            aim_asked: None,
            aim_use: None,
            fp_clip: String::new(),
            fp_asked: None,
            fp_legs_asked: None,
            notes,
            clock: 0.0,
            last_input: MoveInput::default(),
            body_clip_state: Default::default(),
            frames: 0,
            weapon_fire: None,
        };
        // --- the body -----------------------------------------------------------------------
        let id = actors.add_object(native, cache, biped_tag, feet, yaw.to_degrees())?;
        pm.body = Some(id);
        if pm.weapon_tag != 0 {
            if let Err(e) = actors.set_weapon_tag(native, cache, id, Some(pm.weapon_tag)) {
                pm.notes.push(format!("the weapon did not go into the hands: {e}"));
            }
            pm.arm_weapon(native, cache);
            // The held weapon's own melee law, measured by the damage side.
            if let Some(m) = crate::reach_damage::melee_parameters(native, cache, pm.weapon_tag) {
                pm.melee_reach = Some(m.maximum_lunge_range);
                pm.melee_pyramid = Some((m.pyramid_angles, m.pyramid_depth));
                pm.notes.push(format!(
                    "melee: authored lunge range {:.2} wu, damage pyramid {:.1}/{:.1} deg x {:.2} wu (weap +0x2AC)",
                    m.maximum_lunge_range,
                    m.pyramid_angles[0].to_degrees(),
                    m.pyramid_angles[1].to_degrees(),
                    m.pyramid_depth
                ));
            } else {
                pm.notes.push("this weapon authors no melee parameters".into());
            }
        }
        Ok(pm)
    }

    /// Resolve and mount Reach's first-person arms and the gun in them. Call once the scene can
    /// answer `first_person_models`; a no-op after that.
    ///
    /// The arms are bare `mode` tags with no object and no graph of their own, so they are placed
    /// as render models and POSED BY THE WEAPON's first-person graph — which lists the gun's nodes
    /// beside the arms', so one clip drives both and the grip needs no marker attach.
    pub fn mount_first_person(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        rows: &[(String, u32, u32)],
        elite_of: &dyn Fn(u32) -> bool,
        actors: &mut ActorScene,
    ) {
        if self.fp_rig.is_some() || rows.is_empty() {
            return;
        }
        // One row per playable species, and they are not interchangeable. The row is picked ONCE
        // and both of its models come from it — the arms and the first-person body — because
        // mixing a Spartan's legs with an Elite's arms is exactly the kind of cross-species
        // pairing `first_person_weapon`'s species row exists to prevent.
        let row = rows
            .iter()
            .find(|(_, h, _)| *h != 0xFFFF_FFFF && elite_of(*h) == self.species.is_elite())
            .or_else(|| rows.iter().find(|(_, h, _)| *h != 0xFFFF_FFFF));
        let Some(&(_, hands, fp_body_tag)) = row else {
            self.notes.push("this cache has no '@player representation' arms model".into());
            return;
        };
        // #fp-legs  THE FIRST-PERSON LEGS, posed by the BIPED's own graph.
        //
        // Not by the weapon's first-person graph: measured on `forge_halo`, that graph reaches 0
        // of the Spartan `fp_body`'s 82 nodes, while the biped's own graph reaches all 82. They
        // are the same skeleton as the third-person biped, which is why one clip can drive both
        // and why the legs are framed from the body's own frame in `pose`.
        if fp_body_tag != 0xFFFF_FFFF && self.fp_body.is_none() {
            let jmad = crate::reach_actor::resolve_jmad(native, cache, self.biped_tag);
            match actors.add_render_model(native, cache, fp_body_tag, jmad, self.st.pos, self.st.facing.to_degrees()) {
                Ok(id) => {
                    self.fp_body = Some(id);
                    self.notes.push(format!(
                        "first-person legs: '{}' posed by the biped's own graph",
                        native.tag_name(cache, fp_body_tag).unwrap_or_default()
                    ));
                }
                Err(e) => self.notes.push(format!(
                    "this cache's '@player representation' first-person body {fp_body_tag:#x} did not load                      ({e}), so looking down in first person shows the third-person body instead"
                )),
            }
        } else if fp_body_tag == 0xFFFF_FFFF {
            self.notes.push(
                "this cache authors no '@player representation' first person body, so first person                  draws the third-person biped for the legs and you will see your own arms twice"
                    .into(),
            );
        }
        let (fp_model, fp_graph) = if self.weapon_tag != 0 {
            crate::reach_actor::first_person_weapon(native, cache, self.weapon_tag, usize::from(self.species.is_elite()))
        } else {
            (None, None)
        };
        self.fp_rig = Some((hands, fp_graph));
        if fp_graph.is_none() {
            self.notes.push(
                "the weapon authors no first-person interface, so there is no authored pose for \
                 the arms and they are NOT drawn (their bind pose is both arms straight out)"
                    .into(),
            );
            return;
        }
        let eye = self.st.eye(&self.phys);
        for tag in [Some(hands), fp_model].into_iter().flatten() {
            match actors.add_render_model(native, cache, tag, fp_graph, eye, self.st.yaw.to_degrees()) {
                Ok(id) => {
                    if let Some(a) = actors.get_mut(id) {
                        a.set_first_person(true);
                        a.model_yaw = crate::reach_actor::FP_ARMS_MODEL_YAW;
                    }
                    self.fp.push(id);
                }
                Err(e) => self.notes.push(format!("first-person model {tag:#x} did not load: {e}")),
            }
        }
        // #fp-attach  PUT THE GUN IN THE HAND, through the graph's own hierarchy.
        //
        // Reported: "the weapon isn't even in the first person hands, we had this figured out in
        // the theater where we held the weapon properly." They are right, and that is the whole
        // fault: the resolution lived inside Theater's driver, so a rig mounted by test mode had
        // `fp_attach: None` and the gun was composed through the WEAPON model's own two-node
        // skeleton — where its root has no parent — which draws it at the rig's origin, inside
        // the player's head. `ActorScene::fp_grip_of` is now that same resolution, shared, and
        // Theater calls it too. Measured on `forge_halo`'s assault rifle: unparented the gun's
        // root sits 0.088 wu (Spartan) / 0.098 wu (Elite) from the `r_hand` holding it;
        // parented, 0.026 / 0.031 wu, which is the authored grip offset.
        let arms_id = self.fp.first().copied();
        if let Some(arms_id) = arms_id {
            for id in self.fp.clone() {
                let grip = actors.fp_grip_of(arms_id, id);
                if let Some(a) = actors.get_mut(id) {
                    a.fp_attach = grip;
                    if grip.is_none() {
                        a.world_override = None;
                    }
                }
                if id != arms_id {
                    match grip {
                        Some((_, node)) => self.notes.push(format!(
                            "the first-person weapon is parented to the arms' node {node} '{}' — the                              graph's own hierarchy",
                            actors
                                .get(arms_id)
                                .and_then(|a| a.def.model.node_names.get(node).cloned())
                                .unwrap_or_default()
                        )),
                        None => self.notes.push(
                            "this weapon's first-person graph does not parent its model to a hand node,                              so the gun is drawn at the rig's origin rather than in the grip"
                                .into(),
                        ),
                    }
                }
            }
        }
        self.notes.push(format!("first-person rig: {} model(s) on the camera", self.fp.len()));
    }

    /// #play  Put a weapon in the player's hands (or empty them with `None`), at any time.
    ///
    /// Reported: "no way to add/pickup a weapon and use it in any way." This is the handing-over
    /// half — the `Play ▸ Give weapon` menu and the `play weapon <name>` verb both come here.
    ///
    /// Changing weapons changes three things at once and they must move together: the animation
    /// CLASS every locomotion and melee clip is keyed on, the melee REACH (each weapon authors
    /// its own `maximum lunge range`), and the FIRST-PERSON RIG — the arms are posed by the
    /// weapon's own first-person graph, so a new weapon means a new graph, which means the rig has
    /// to be rebuilt rather than retargeted.
    pub fn give_weapon(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        weapon: Option<u32>,
        actors: &mut ActorScene,
        rows: &[(String, u32, u32)],
        elite_of: &dyn Fn(u32) -> bool,
    ) -> String {
        let Some(id) = self.body else { return "test mode is not running".into() };
        self.weapon_tag = weapon.unwrap_or(0);
        self.weapon_path = weapon.and_then(|w| native.tag_name(cache, w)).unwrap_or_default();
        self.weapon_class = weapon.and_then(|w| crate::reach_actor::weapon_class(native, cache, w));
        if let Err(e) = actors.set_weapon_tag(native, cache, id, weapon) {
            return format!("the weapon did not go into the hands: {e}");
        }
        // The melee law of the weapon now held — measured by the damage side, not chosen here.
        let mp = weapon.and_then(|w| crate::reach_damage::melee_parameters(native, cache, w));
        self.melee_reach = mp.map(|m| m.maximum_lunge_range);
        self.melee_pyramid = mp.map(|m| (m.pyramid_angles, m.pyramid_depth));
        // A different weapon is a different set of trigger / barrel / magazine machines.
        self.arm_weapon(native, cache);
        // Rebuild the first-person rig on the new weapon's graph.
        let old: Vec<u32> = self.fp.drain(..).collect();
        actors.actors.retain(|a| !old.contains(&a.id));
        self.fp_rig = None;
        self.fp_asked = None;
        self.fp_clip.clear();
        self.mount_first_person(native, cache, rows, elite_of, actors);
        actors.rebuild = true;
        self.asked = None; // the class changed, so the locomotion clip must be re-chosen
        match weapon {
            Some(_) => format!(
                "holding {} (class {}){}",
                self.weapon_path,
                self.weapon_class.clone().unwrap_or_else(|| crate::film_anim::DEFAULT_WEAPON_CLASS.into()),
                match (self.melee_reach, self.melee_pyramid) {
                    (Some(r), Some((a, d))) => format!(
                        " | melee: authored lunge range {r:.2} wu, damage pyramid {:.1}/{:.1} deg x {d:.2} wu",
                        a[0].to_degrees(),
                        a[1].to_degrees()
                    ),
                    _ => " | this weapon authors no melee parameters".into(),
                }
            ),
            None => "hands empty".into(),
        }
    }

    /// #play  Pull the trigger.
    ///
    /// Test mode has something a recording does not: **it knows exactly where the player is
    /// aiming**, so the direction of a shot is a stated fact here rather than an inference. The
    /// shot plays the weapon's own firing animation and hands the hit to the damage seam; what it
    /// does NOT do is invent a damage number — that is `reach_damage`'s, and with no model
    /// installed the caption says so.
    ///
    /// **One call is one FIRE EVENT**, i.e. one "firing effect" of one barrel: `shots` goes up by
    /// one, and every projectile the event launched (`spreads`, one entry per projectile, from
    /// [`crate::play_fire::FireEvent::Shot`]) is resolved as an aim ray offset by its own error
    /// cone and fan angle. An empty `spreads` is one ray along the crosshair (the `play fire`
    /// verb). WHEN a call may happen -- the trigger behavior, the rate of fire, the recovery, the
    /// magazine -- is [`crate::play_fire`]'s, not decided here: it used to be a 0.25 s timer, which
    /// is why a held button fired once.
    pub fn fire(
        &mut self,
        scene: &dyn ObjectScene,
        damage: Option<&mut (dyn ObjectDamage + 'static)>,
        spreads: &[crate::play_fire::Spread],
    ) {
        if self.weapon_tag == 0 {
            self.last_hit = Some("no weapon in hand".into());
            return;
        }
        // Each event restarts the firing animation, so a string of shots keeps the arms in it.
        self.fired = Some(0.0);
        self.shots += 1;
        let eye = self.st.eye(&self.phys);
        let aim = look_dir(self.st.yaw, self.st.pitch);
        let dirs: Vec<Vec3> = if spreads.is_empty() { vec![aim] } else { spreads.iter().map(|s| crate::play_fire::spread_dir(aim, s)).collect() };
        // The shot's range: far enough to cross any Forge structure. A projectile's real flight
        // (gravity, its own speed, its detonation) is `reach_projectile`'s law and is not
        // reimplemented here — this is the aim ray, which is what makes the hit point honest.
        const SHOT_RANGE: f32 = 200.0;
        let mut damage = damage;
        let mut first = String::new();
        let mut hits = 0usize;
        for (k, &dir) in dirs.iter().enumerate() {
            let hit = scene.pick(eye, dir).and_then(|datum| {
                let d = scene.raycast_solid(eye, dir, SHOT_RANGE)?;
                Some((datum, eye + dir * d, d))
            });
            hits += usize::from(hit.is_some());
            let text = match (hit, damage.as_deref_mut()) {
                (Some((datum, point, d)), Some(model)) => {
                    let r = model.apply(DamageHit {
                        target_datum: datum,
                        point,
                        dir,
                        source: DamageSource::Shot { weapon: self.weapon_tag },
                        // A direct hit: the falloff distance is zero, not the distance flown.
                        distance: 0.0,
                    });
                    match r {
                        Some(r) => format!("shot {}", describe_damage(datum, d, &r)),
                        None => format!("shot {datum:#x} at {d:.2} wu, the damage model applied nothing"),
                    }
                }
                (Some((datum, _, d)), None) => {
                    format!("shot would hit {datum:#x} at {d:.2} wu — {}", crate::play_damage::NO_MODEL)
                }
                (None, _) => "shot hit nothing".to_string(),
            };
            if k == 0 {
                first = text;
            }
        }
        self.last_hit = Some(if dirs.len() > 1 {
            format!("{first} | {} projectile(s), {hits} struck something", dirs.len())
        } else {
            first
        });
    }

    /// #weapon-fire  Build the firing machine for the weapon now in hand (called whenever
    /// `weapon_tag` changes). Every magazine starts full, which is also what the HUD's `Ammo`
    /// does for a freshly drawn weapon.
    pub fn arm_weapon(&mut self, native: &NativeDll, cache: CacheHandle) {
        self.weapon_fire = None;
        if self.weapon_tag == 0 {
            return;
        }
        let Some(def) = crate::reach_weapon::WeaponDef::read(native, cache, self.weapon_tag) else { return };
        if def.barrels.is_empty() {
            return; // melee-only (sword, hammer, knife): there is nothing to fire
        }
        let loaded: Vec<i32> = (0..crate::play_hud::MAGAZINE_MAX as usize)
            .map(|i| {
                crate::play_hud::Magazine::read(native, cache, self.weapon_tag, i)
                    .map_or(0, |m| m.rounds_loaded_maximum.clamp(0, crate::play_hud::ROUNDS_LOADED_MASK))
            })
            .collect();
        self.notes.push(format!(
            "firing: {}",
            crate::play_fire::describe(&def)
        ));
        self.weapon_fire = Some(crate::play_fire::WeaponFire::new(std::sync::Arc::new(def), loaded));
    }

    /// #weapon-fire  Run the held weapon's trigger / barrel / magazine machines for `dt` seconds
    /// with the fire button in state `held`, and return what happened, in tick order. The caller
    /// turns each [`crate::play_fire::FireEvent::Shot`] into a [`PlayMode::fire`] (through the HUD's
    /// ammo gate) -- this does not touch the scene, so a gate can drive it with no map.
    ///
    /// `ammo` is the HUD's view of the first magazine: `(rounds loaded, reload running, rounds in
    /// reserve)`. The HUD owns the count between frames; the machine spends from it as it fires.
    pub fn advance_fire(&mut self, dt: f32, held: bool, ammo: Option<(i32, bool, bool)>) -> Vec<crate::play_fire::FireEvent> {
        let usable = self.weapon_usable();
        let meleeing = self.melee.is_some();
        let crouched = self.st.crouched;
        let Some(wf) = self.weapon_fire.as_mut() else { return Vec::new() };
        let (reloading, has_reserve) = match ammo {
            Some((loaded, reloading, reserve)) => {
                if let Some(m) = wf.def.barrels.first().and_then(|b| b.magazine) {
                    wf.sync_loaded(m, loaded);
                }
                (reloading, reserve)
            }
            None => (false, false),
        };
        wf.advance(
            dt,
            &crate::play_fire::FireInput {
                held: held && usable,
                alt_held: false,
                usable,
                reloading,
                has_reserve,
                crouched,
                melee: meleeing,
            },
        )
    }

    /// Start a melee swing, if one is not already running. Returns the hit it would report.
    ///
    /// The SWING always plays; the DAMAGE is a separate question that belongs to
    /// [`crate::play_damage`]. With no damage model installed this finds the target and says so.
    /// `damage` is typed with an explicit `'static` trait-object bound rather than an elided one
    /// because `&mut` is INVARIANT in what it points at: a `&mut (dyn ObjectDamage + 'static)`
    /// taken out of the app's own `Box<dyn ObjectDamage>` will not coerce to a shorter-lived
    /// trait object, so eliding it here would force every caller's borrow to be `'static`.
    pub fn melee(&mut self, scene: &dyn ObjectScene, damage: Option<&mut (dyn ObjectDamage + 'static)>) {
        if self.melee.is_some() {
            return;
        }
        // Which of the class's authored swings. Derived from the CLOCK in engine ticks, the same
        // rule the Theater lane uses (`(age / 7) % 2`) — and from the clock rather than the frame
        // counter, so which swing you get does not depend on the display rate. (Everything else
        // in this module that used the frame counter to mean time has been moved to the clock for
        // the same reason: a clip driven off a frame count plays at half speed on a 30 Hz frame
        // and double on a 120 Hz one.)
        self.melee_variant = ((self.clock * 60.0) as u64 / 7 % 2) as u8;
        // The length comes from the clip itself once it is installed; until then one second.
        self.melee = Some((0.0, 1.0));
        let eye = self.st.eye(&self.phys);
        let dir = look_dir(self.st.yaw, self.st.pitch);
        // THE REACH IS THE WEAPON'S OWN. `maximum lunge range` off `weap` +0x2AC, with the
        // engine's 1.22 wu default for a tag that authors 0 — measured by the damage side, not
        // chosen here. Unarmed falls back to the one named constant that says it is not authored.
        let reach = self.melee_reach.unwrap_or(MELEE_REACH_UNARMED_WU);
        let hit = scene.pick(eye, dir).and_then(|datum| {
            let d = scene.raycast_solid(eye, dir, reach)?;
            Some((datum, eye + dir * d, d))
        });
        self.last_hit = Some(match (hit, damage) {
            (Some((datum, point, d)), Some(model)) => {
                let report = model.apply(DamageHit {
                    target_datum: datum,
                    point,
                    dir,
                    source: DamageSource::Melee { weapon: self.weapon_tag },
                    // A melee is a CONTACT hit: there is no falloff distance.
                    distance: 0.0,
                });
                match report {
                    Some(r) => describe_damage(datum, d, &r),
                    None => format!("hit {datum:#x} at {:.2} wu, the damage model applied nothing", d),
                }
            }
            (Some((datum, _, d)), None) => format!(
                "would hit {datum:#x} at {:.2} wu — {}",
                d,
                crate::play_damage::NO_MODEL
            ),
            (None, _) => "swing hit nothing".to_string(),
        });
    }

    /// #seat-entry  The seat's authored `override fov`, radians, when it authors one.
    ///
    /// Datums whose PLACEMENT must not be drawn are [`hidden_datums`](Self::hidden_datums), with
    /// **one source for both directions** — showing and hiding read the same `Option`, so a ride
    /// cannot leave a hidden placement behind.
    pub fn seat_fov(&self) -> Option<f32> {
        self.seat_fov
    }

    /// #ride-weapon  May the biped's own weapon be drawn, fired and swung right now? On foot yes;
    /// seated, only in a seat whose `allows weapons` flag is set -- and not mid-boarding or
    /// mid-exit of one that forbids it.
    /// #use-hold  What the use button does THIS frame, from whether it is `down`, whether it was
    /// `pressed` (went down) and the frame's `dt`. On foot a seat or a weapon acts on the HOLD
    /// ([`crate::reach_action_hold`]) -- never on the press, so a tap does nothing, and letting go
    /// early starts the count again. Seated, a press leaves.
    pub fn use_decide(&mut self, down: bool, pressed: bool, dt: f32) -> UseDecision {
        self.use_hold.step(down, dt);
        if self.riding.is_some() {
            if !down {
                self.exit_armed = true;
            }
            if pressed && self.exit_armed {
                self.exit_armed = false;
                return UseDecision::Exit;
            }
            return UseDecision::Idle;
        }
        let target = self.offer.is_some() || self.pickup.is_some();
        if target && self.use_hold.reached(&self.hold_cfg) {
            self.use_hold.consume();
            self.exit_armed = false;
            return UseDecision::Hold;
        }
        // A press with NOTHING in range still reports why, as it always did.
        if pressed && !target {
            return UseDecision::Nothing;
        }
        UseDecision::Idle
    }

    /// #use-hold  The action prompt on offer on foot, as the words the player reads (the seat's or
    /// the weapon's own authored string with the player's key in place of the button glyph).
    /// `None` when nothing is in range, while riding, or when the authored string did not
    /// resolve -- and then nothing is drawn, as the engine draws nothing for a prompt it has no
    /// text for.
    pub fn action_prompt(&self) -> Option<String> {
        if self.riding.is_some() {
            return None;
        }
        match (&self.offer, &self.pickup) {
            (Some(o), _) => o.prompt.clone(),
            (None, Some(p)) => p.prompt.clone(),
            (None, None) => None,
        }
    }

    pub fn weapon_usable(&self) -> bool {
        self.riding.as_ref().map_or(true, |r| r.allows_weapons)
    }

    /// #play-sound  Hand this frame to test mode's audio (see [`crate::play_sound`]): the engine
    /// of the vehicle being driven, the player's own weapon, and where the listener is.
    pub fn drive_sound(
        &mut self,
        ps: &mut crate::play_sound::PlaySound,
        eng: &mut crate::sound::SoundEngine,
        host: &crate::play_sound::Host,
    ) {
        let player = crate::play_sound::PlayerState {
            weapon: self.weapon_tag,
            shots: self.shots,
            melee: self.melee.is_some(),
            riding: self.riding.as_ref(),
            stick_forward: self.last_input.fwd,
            body: self.body_clip_state,
            input: self.last_input,
            pos: self.st.pos,
            vel: self.st.vel,
            grounded: self.st.grounded,
            jumped: self.last_step.jumped,
            landed: self.last_step.landed,
            biped: self.biped_tag,
            offered: self.offer.as_ref().map(|o| o.vehi_tag),
        };
        crate::play_sound::drive(ps, eng, host, &player, &mut self.vehicles);
    }

    /// #seat-entry  Datums whose PLACEMENT must not be drawn this frame.
    pub fn hidden_datums(&self) -> Vec<u32> {
        self.riding
            .iter()
            .map(|r| r.hidden_datum)
            .chain(self.parked.iter().map(|p| p.ride.hidden_datum))
            // #play-world  ...and every prop that has left its saved pose
            .chain(self.props.hidden_datums())
            .collect()
    }

    /// #ride-park  The object set the offer and the entry test run against: the editor's own, with
    /// every PARKED vehicle moved to where it was left. The editor's table is never written; this is
    /// a copy, made only when something is parked.
    fn objects_with_parked<'a>(&self, objects: &'a [ObjectInfo]) -> std::borrow::Cow<'a, [ObjectInfo]> {
        if self.parked.is_empty() {
            return std::borrow::Cow::Borrowed(objects);
        }
        let mut v = objects.to_vec();
        for p in &self.parked {
            if let Some(o) = v.iter_mut().find(|o| o.datum == p.ride.hidden_datum) {
                let b = p.ride.drive.basis();
                o.pos = p.ride.drive.pos.to_array();
                o.fwd = (b * Vec3::X).to_array();
                o.up = (b * Vec3::Z).to_array();
            }
        }
        std::borrow::Cow::Owned(v)
    }

    /// #pickup  Pick up the weapon the current offer names.
    ///
    /// **It is `give_weapon` with a different trigger** — the same one function the menu and the
    /// script verb call, so the animation class, the melee reach and the first-person rig are
    /// moved together exactly as they are everywhere else. The placement is left where it is:
    /// test mode never writes the editor's object table, and a weapon you can pick up twice is a
    /// far smaller surprise in a test mode than a weapon the editor has quietly deleted.
    pub fn pick_up(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        actors: &mut ActorScene,
        rows: &[(String, u32, u32)],
        elite_of: &dyn Fn(u32) -> bool,
    ) -> Result<String, String> {
        let Some(o) = self.pickup.clone() else {
            return Err("no weapon is in reach".into());
        };
        let was = self.weapon_tag;
        let gave = self.give_weapon(native, cache, Some(o.weap_tag), actors, rows, elite_of);
        let what = o.leaf().to_string();
        Ok(format!(
            "{} {what} ({:.2} wu away) — {gave}",
            if was == 0 { "picked up" } else { "swapped to" },
            o.distance
        ))
    }

    /// #vehicle-ride  Hand test mode the editor's own per-datum appearance maps.
    ///
    /// The host calls this once a frame with the three maps it already builds. It exists because
    /// **riding draws the vehicle as an ACTOR built from the tag**, so everything that lives on
    /// the PLACEMENT rather than on the tag is lost the moment the player gets in — which is the
    /// report: *"it gets in it and its basically spawning a new one so it doesn't keep the
    /// colours and everything of the original vehicle at all."* The model variant already
    /// travelled (`riding_variant_sid`); the colour, the scale and the damage state did not.
    ///
    /// A whole-map replace rather than a merge, so a datum the editor has deleted cannot keep a
    /// stale colour alive, and cheap: it is a move of three maps, not a copy of the scene.
    pub fn set_object_appearance(
        &mut self,
        map: std::collections::HashMap<u32, (Option<(u8, u8)>, f32, u8)>,
    ) {
        self.props.appearance = map.clone();
        self.appearance_of = map;
    }

    /// #seat-entry  Get into the seat the current offer names. `Err` with a reason when there is
    /// nothing to get into, or when the placement has gone.
    pub fn enter_vehicle(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        objects: &[ObjectInfo],
    ) -> Result<String, String> {
        let Some(o) = self.offer.clone() else { return Err("no vehicle seat is in range".into()) };
        let want = o.parent_datum.unwrap_or(o.datum);
        let objects = self.objects_with_parked(objects).into_owned();
        let Some(placement) = objects.iter().find(|p| p.datum == want) else {
            return Err(format!("the placement {want:#x} is no longer in the scene"));
        };
        let mut r = crate::play_vehicle::begin(&mut self.vehicles, native, cache, &o, placement);
        self.riding_variant_sid = placement.variant_name_sid;
        // #ride-park  Getting back into a vehicle that was parked resumes THAT vehicle: its
        // attitude and speed, its colour and the actors already drawing it -- not a fresh copy of
        // the placement.
        if let Some(i) = self.parked.iter().position(|p| p.ride.hidden_datum == want) {
            let p = self.parked.remove(i);
            r.drive = p.ride.drive;
            r.appearance = p.ride.appearance;
            self.riding_variant_sid = p.variant_sid;
            self.vehicle_actors = p.actors;
        }
        let msg = format!(
            "entered {} seat {} '{}'{}",
            r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path),
            r.seat,
            r.seat_label,
            match &r.drivable {
                Ok(()) => String::new(),
                Err(why) => format!(" — {why}"),
            }
        );
        self.riding = Some(r);
        self.grip_ref = [None, None];
        // #vehicle-ride  Everything that lives on the PLACEMENT rather than the tag travels with
        // the ride, or getting in visibly respawns the vehicle. `enter_vehicle` is the only
        // place that still has the placement's datum, which is why it happens here and not in
        // `play_vehicle::begin`.
        //
        // A datum the host has not told us about keeps the DEFAULT appearance (the tag's own),
        // which is exactly today's behaviour — so a host that never calls
        // [`PlayMode::set_object_appearance`] is unchanged rather than drawing a black Warthog.
        if let (Some(r), Some(&(forge_color, scale, damage))) =
            (self.riding.as_mut(), self.appearance_of.get(&want))
        {
            r.appearance = crate::play_vehicle::RideAppearance { forge_color, scale, damage };
        }
        self.offer = None;
        Ok(msg)
    }

    /// #seat-entry  Get out: start the seat's authored EXIT animation. The rider only leaves the
    /// seat when it has played for its authored length (see [`finish_exit`](Self::finish_exit)),
    /// and a rider who is still boarding, or already leaving, cannot start it again -- the
    /// engine's enter and exit tasks are not interruptible.
    pub fn exit_vehicle(&mut self, _native: &NativeDll, _cache: CacheHandle) -> String {
        use crate::play_vehicle::{begin_exit, RidePhase};
        let Some(r) = self.riding.as_mut() else { return "not in a vehicle".into() };
        let name = r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path).to_string();
        match r.phase {
            RidePhase::Boarding => format!("still getting into {name}"),
            RidePhase::Exiting => format!("already getting out of {name}"),
            RidePhase::Seated => {
                begin_exit(r);
                let msg = format!("getting out of {name} seat {} ({}:exit)", r.seat, r.seat_label);
                self.notes.push(msg.clone());
                msg
            }
        }
    }

    /// #seat-entry  The exit animation has finished: the player is put where they would have
    /// stood to get IN -- the entry marker's own position pushed out along its own forward by the
    /// seat's own `entry radius` -- and carries the vehicle's velocity, which is what Reach does.
    ///
    /// When the markers cannot be resolved the player is left where the seat was rather than
    /// teleported somewhere invented, and the reason is recorded.
    fn finish_exit(&mut self, native: &NativeDll, cache: CacheHandle, scene: &dyn ObjectScene, objects: &[ObjectInfo]) {
        let Some(r) = self.riding.take() else { return };
        let carry = crate::play_vehicle::exit_velocity(&r);
        let cands = crate::play_vehicle::exit_candidates(&mut self.vehicles, native, cache, &r);
        // #ride-park  The vehicle stays where it was left, drawn by the same actors.
        self.parked.push(Parked {
            ride: r.clone(),
            variant_sid: self.riding_variant_sid,
            actors: std::mem::take(&mut self.vehicle_actors),
            baked: r.drive.pos,
        });
        // #play-world  The empty vehicle becomes a rigid body at the pose and velocity it had.
        self.props.mark_adopted(r.hidden_datum);
        self.props.request_claim(crate::play_world::VehicleClaim {
            datum: r.hidden_datum,
            origin: r.drive.pos,
            rot: r.drive.basis(),
            velocity: r.drive.vel,
            angular: r.drive.ang_vel,
        });
        let name = r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path).to_string();
        let picked = self.choose_exit(&cands, scene, objects, native, cache);
        let msg = match picked {
            Some((p, how)) => {
                // The player keeps LOOKING where they were looking from the seat.
                self.st = play_ctl::PlayerState::standing_at(p, r.yaw);
                self.st.pitch = r.pitch;
                self.st.vel = carry;
                format!("left {name} seat {} {how}", r.seat)
            }
            None => {
                self.st.vel = carry;
                format!("left {name} seat {} — its entry marker could not be resolved, so you are where the seat was", r.seat)
            }
        };
        self.notes.push(msg);
    }

    /// #collision  Where the rider steps out to: the first of `cands` (the vehicle's entry
    /// markers, nearest door first) that a standing capsule fits at -- inside no block and into
    /// no surface -- else the nearest free spot around the first one, else the first one as it
    /// is (the controller's own push-out and stuck fallback take it from there). Each candidate
    /// is tried stepped down onto the GROUND under it (a door sill is not where to stand: that
    /// drops the player through an airborne clip and a hard landing) and then as the marker has
    /// it. `None` only when the vehicle has no resolvable marker at all.
    fn choose_exit(&mut self, cands: &[Vec3], scene: &dyn ObjectScene, objects: &[ObjectInfo], native: &NativeDll, cache: CacheHandle) -> Option<(Vec3, String)> {
        let first = *cands.first()?;
        for (i, c) in cands.iter().enumerate() {
            let mut spots = Vec::new();
            if let Some(d) = scene.raycast_solid(*c + Vec3::Z * 0.75, -Vec3::Z, 4.0) {
                spots.push(Vec3::new(c.x, c.y, c.z + 0.75 - d));
            }
            spots.push(*c);
            for p in spots {
                if !self.world.covers(p) {
                    self.world.regather(p, scene, objects, native, cache);
                }
                if !PlayerState::overlaps_at(p, &self.phys, &self.world, 0.02) {
                    return Some((p, format!("(exit spot {} of {})", i + 1, cands.len())));
                }
            }
        }
        // nothing free at a marker: the nearest free place around the first
        self.world.regather(first, scene, objects, native, cache);
        let probe = PlayerState::standing_at(first, 0.0);
        match probe.find_free(&self.phys, &self.world, 3.0, &[0.0, 0.3, -0.3]).or_else(|| probe.find_free(&self.phys, &self.world, 6.0, &[0.8, 2.0, 4.0])) {
            Some(p) => Some((p, format!("(no marker was free: moved {:.2} wu to the nearest free spot)", p.distance(first)))),
            None => Some((first, "(no free spot within reach: the controller will push it out)".into())),
        }
    }

    /// #play-world  Keep the ride path's parked vehicles and the rigid-body world in step: a
    /// vehicle that started to move is adopted (drawn by the parked-vehicle actors), a simulated
    /// one writes its pose and velocity into its ride state every frame, and one a driver is in
    /// (or got back into) leaves the world with its state intact -- no snap.
    fn sync_vehicle_bodies(&mut self, native: &NativeDll, cache: CacheHandle, objects: &[ObjectInfo]) {
        if let Some(r) = &self.riding {
            self.props.release_vehicle(r.hidden_datum);
        }
        let parked_ids: Vec<u32> = self.parked.iter().map(|p| p.ride.hidden_datum).collect();
        self.props.sync_released(&parked_ids);
        for d in self.props.displaced_vehicles() {
            if parked_ids.contains(&d) || self.riding.as_ref().is_some_and(|r| r.hidden_datum == d) {
                continue;
            }
            let Some(o) = objects.iter().find(|o| o.datum == d) else { continue };
            let Some(ride) = crate::play_world::bare_riding(&mut self.vehicles, native, cache, o) else { continue };
            self.props.mark_adopted(d);
            // what the static draw carries (colour, scale, damage) and the variant travel with it
            let mut ride = ride;
            if let Some(&(forge_color, scale, damage)) = self.appearance_of.get(&d) {
                ride.appearance = crate::play_vehicle::RideAppearance { forge_color, scale, damage };
            }
            let baked = ride.drive.pos;
            self.parked.push(Parked { ride, variant_sid: o.variant_name_sid, actors: Vec::new(), baked });
        }
        for p in self.parked.iter_mut() {
            if let Some(st) = self.props.vehicle_state(p.ride.hidden_datum) {
                crate::play_world::write_drive_state(&mut p.ride.drive, st);
            }
        }
    }

    /// #seat-entry  Keep the actors that DRAW the ridden vehicle in step with the ride.
    ///
    /// **Showing and hiding are the same `Option`**: the placement is filtered out of the render
    /// set by [`hidden_datums`](Self::hidden_datums) and drawn here, both from `self.riding`. So
    /// a ride cannot show two vehicles (the placement is gone the frame the actors appear) and
    /// ending one cannot show none (the actors go the frame the placement comes back).
    fn sync_vehicle_actors(&mut self, native: &NativeDll, cache: CacheHandle, actors: &mut ActorScene) {
        match self.riding.clone() {
            Some(r) => {
                let variant = self.riding_variant_sid;
                Self::sync_ride_actors(&mut self.vehicles, &mut self.notes, native, cache, &r, variant, &mut self.vehicle_actors, actors);
            }
            None => {
                if !self.vehicle_actors.is_empty() {
                    let ids = std::mem::take(&mut self.vehicle_actors);
                    actors.actors.retain(|a| !ids.contains(&a.id));
                    actors.rebuild = true;
                }
            }
        }
        for i in 0..self.parked.len() {
            let p = &mut self.parked[i];
            let r = p.ride.clone();
            Self::sync_ride_actors(&mut self.vehicles, &mut self.notes, native, cache, &r, p.variant_sid, &mut p.actors, actors);
            // The object lighting is baked into the actor's meshes where they were built; a parked
            // vehicle that has come to rest somewhere else is re-lit there (the static path samples
            // the probe at the object's position).
            if r.drive.vel.length() < 0.05 && (r.drive.pos - p.baked).length() > crate::play_world::RELIGHT_DISTANCE {
                for &id in &p.actors {
                    if let Some(a) = actors.get_mut(id) {
                        a.meshes_dirty = true;
                    }
                }
                actors.rebuild = true;
                p.baked = r.drive.pos;
            }
        }
    }

    /// One ride's actors: built once, then only moved, plus the vehicle's own driven animation.
    #[allow(clippy::too_many_arguments)]
    fn sync_ride_actors(
        vehicles: &mut crate::play_vehicle::VehicleIndex,
        notes: &mut Vec<String>,
        native: &NativeDll,
        cache: CacheHandle,
        r: &crate::play_vehicle::Riding,
        variant: u32,
        ids: &mut Vec<u32>,
        actors: &mut ActorScene,
    ) {
        // The placement's own model VARIANT decides which of a Warthog's four weapon
        // attachments is drawn, so it has to be passed in rather than defaulted.
        let pieces = crate::play_vehicle::draw_pieces(vehicles, native, cache, r, variant);
        // Added once, then only moved. `add_render_model` reads a render model, so doing it per
        // frame would decode the Warthog's meshes sixty times a second.
        if ids.len() != pieces.len() {
            let old = std::mem::take(ids);
            actors.actors.retain(|a| !old.contains(&a.id));
            for p in &pieces {
                match actors.add_render_model(native, cache, p.mode_tag, None, Vec3::ZERO, 0.0) {
                    Ok(id) => ids.push(id),
                    Err(why) => notes.push(format!("the ridden vehicle's model {:#x} could not be drawn: {why}", p.mode_tag)),
                }
            }
            actors.rebuild = true;
        }
        // The pose is a full MATRIX, not a position and a yaw: an attachment carries its own
        // authored rotation, and a turret mounted at an angle would be wrong placed by yaw alone.
        for (id, p) in ids.iter().zip(pieces.iter()) {
            if let Some(a) = actors.get_mut(*id) {
                a.world_override = Some(p.world);
                a.playing = false;
            }
        }
        // The placement's model variant belongs to the object's own model (the first piece).
        if let Some(&first) = ids.first() {
            crate::play_vehicle::set_variant(actors, first, variant);
        }
        // #vehicle-anim  ...and then the vehicle's OWN driven animation: the wheels rolling at
        // `distance / radius` and the steering at the authored lock. `play_vehicle` measures all
        // of it off the tags and classifies each overlay as a wrapping CYCLE or a monotone RANGE;
        // this is the one call that makes any of it reach the screen.
        notes.extend(crate::play_vehicle::drive_actors(vehicles, native, cache, r, &pieces, ids, actors));
    }

    /// #seat-entry  One frame of a RIDE: drive the vehicle from the stick, put the body on the
    /// seat marker, and solve the seat camera.
    ///
    /// Returns `true` when the exit animation has just finished and the rider must now leave the
    /// seat (the caller does that, because it needs the whole `PlayMode`).
    fn drive_and_seat(
        &mut self,
        input: &MoveInput,
        dt: f32,
        native: &NativeDll,
        cache: CacheHandle,
        scene: &dyn ObjectScene,
    ) -> bool {
        use crate::play_vehicle as pv;
        let Some(mut r) = self.riding.take() else { return false };
        // The rider's look is the mouse's, clamped by the seat's own authored range inside
        // `seat_shot`; stored unclamped here so that releasing the look does not creep.
        r.yaw = input.yaw;
        r.pitch = input.pitch;
        // #ride-phase  The phase clock. The enter and exit clips run for their AUTHORED length
        // before control changes hands; the lengths are filled by `pose` once the rider's graph is
        // in hand (a `None` holds the phase).
        let event = pv::advance_phase(&mut r, dt);
        let seat = self.vehicles.entry(native, cache, r.vehi_tag).and_then(|e| e.seats.seats.get(r.seat).cloned());
        // Only the DRIVER drives, and only once SEATED. A gunner's or passenger's stick moves
        // nothing, which is what the authored `driver` flag means; a boarding or leaving rider's
        // moves nothing either.
        let driving = r.controls_live(seat.as_ref().is_some_and(|s| s.is_driver()));
        let (class, saturation) = self
            .vehicles
            .entry(native, cache, r.vehi_tag)
            .and_then(|e| {
                e.drive
                    .as_ref()
                    .ok()
                    .map(|m| (pv::control_class(m.law), m.wheel_law.as_ref().map(|w| w.steer.saturation_angle())))
            })
            .unwrap_or((pv::ControlClass::SteeredWheels, None));
        let di = if driving {
            pv::drive_input(class, input, r.drive.yaw, saturation)
        } else {
            crate::reach_vehicle::DriveInput { want_yaw: r.drive.yaw, ..Default::default() }
        };
        let (yaw0, pitch0) = (r.drive.yaw, r.drive.pitch);
        pv::step(&mut self.vehicles, native, cache, &mut r, di, dt, scene);
        // #ride-camera  `local aiming` (seat flag 29): the rider's aim carries the vehicle's own
        // rotation, so as the hull turns the view turns with it. The engine adds the seat frame's
        // yaw and pitch DELTA to the player's look every frame (`sub_1401E2530`); a seat without
        // the flag keeps a world-fixed look.
        if seat.as_ref().is_some_and(|s| s.local_aiming()) {
            let wrap = |a: f32| (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            r.yaw += wrap(r.drive.yaw - yaw0);
            r.pitch += r.drive.pitch - pitch0;
        }
        // The body goes on the seat marker, and the controller is never asked.
        if let Some((feet, yaw)) = pv::rider_pose(&mut self.vehicles, native, cache, &r) {
            self.st.set_pose_from_vehicle(feet, yaw, r.pitch, r.drive.vel);
            // #seat-camera  **AND GIVE THE LOOK BACK.** Reported: *"the camera is WRONG for the
            // warthog, it won't let me move it really AT ALL!"*
            //
            // This file already carries the invariant that broke here, written down a hundred
            // lines below on the body's own draw: **`st.yaw` is where the player is LOOKING and
            // `st.facing` is where the body is POINTING**, and they are two quantities precisely
            // because the biped used to "just rotate with the camera which is wrong".
            //
            // `set_pose_from_vehicle` sets both to the vehicle's heading. For `facing` that is
            // exactly right -- a seated body faces the way the vehicle does. For `yaw` it is
            // fatal, because `yaw` is the LOOK ACCUMULATOR: `App` integrates the mouse as
            // `play_look(p.st.yaw, p.st.pitch, dx, dy)`, so overwriting it with the hull's heading
            // every frame meant the look was re-based to the vehicle's forward sixty times a
            // second and only ever carried ONE frame of mouse delta. The camera did not freeze --
            // it was being dragged back to the bonnet as fast as the player could push it, which
            // is also why it was "in the wrong spot": it always pointed along the hull.
            //
            // So the placement stands and the look is restored. One accumulator, and it is the
            // ride's own: `r.yaw`/`r.pitch` are the unclamped truth and the seat's authored range
            // is applied where it belongs, inside `seat_shot`.
            self.st.yaw = r.yaw;
            self.st.pitch = r.pitch;
        }
        // The seat's own camera: the chase shot for a third-person seat, the marker view for a
        // first-person one. `seat_fov` is reported and `App` applies it; one writer either way.
        //
        // Measured: **every seat in `forge_halo` authors `override fov` 0.0**, so the FOV path is
        // one the shipped tags never take. It is implemented because the field exists and because
        // a modded vehicle could use it, not because anything in the game does.
        self.seat_shot = pv::seat_shot(&mut self.vehicles, native, cache, &mut r, scene, dt);
        self.seat_fov = self.seat_shot.and_then(|s| (s.fov > 0.0).then_some(s.fov));
        self.riding = Some(r);
        event == pv::PhaseEvent::Exited
    }

    /// #ride-camera  The on-foot camera: the biped's eye, or HMS's own optional third-person view
    /// of it. The engine has no third-person player camera to read a framing from (see
    /// [`THIRD_PERSON_BACK`]).
    fn foot_pose(&self) -> crate::reach_unit_camera::CamPose {
        use crate::reach_unit_camera::CamPose;
        let eye = self.st.eye(&self.phys);
        match self.view {
            PlayView::FirstPerson => CamPose { eye, yaw: self.st.yaw, pitch: self.st.pitch },
            // Solved in `tick`, where the scene is available to stand it off the walls.
            PlayView::ThirdPerson => CamPose { eye: self.third_person_cam, yaw: self.st.yaw, pitch: self.st.pitch },
        }
    }

    /// #ride-camera  Is the camera away from the player's eyes -- on a chase shot, or still
    /// blending back from one? The body is drawn and the first-person rig hidden while it is.
    fn camera_detached(&self) -> bool {
        self.cam_mode == crate::reach_unit_camera::CamMode::Chase || self.cam_blend.active()
    }

    /// #ride-camera  Choose this frame's camera MODE, run the engine's transition when it
    /// changed, and publish the pose [`camera`](Self::camera) will hand the window.
    ///
    /// The mode is the engine's own (`c_game_director::update`): a third-person seat is the
    /// `following` camera, anything else is first person. Changing INTO the following camera
    /// blends for the globals' `enter vehicle transition time`, changing OUT of it for `exit
    /// vehicle transition time`; every other change is a cut. The blend leaves the pose the window
    /// was last given, so entering from a first-person eye starts exactly where the player was
    /// looking from.
    fn update_camera(&mut self, dt: f32, native: &NativeDll, cache: CacheHandle) {
        use crate::reach_unit_camera::{transition_seconds, CamMode};
        let (mode, target) = match (self.riding.is_some(), self.seat_shot) {
            (true, Some(shot)) => (if shot.third_person { CamMode::Chase } else { CamMode::SeatFirst }, shot.pose),
            _ => (CamMode::Foot, self.foot_pose()),
        };
        if mode != self.cam_mode {
            let g = self.vehicles.camera_globals(native, cache);
            let secs = transition_seconds(self.cam_mode, mode, &g);
            if let Some(prev) = self.cam_prev {
                self.cam_blend.start(prev, secs);
            }
            self.cam_mode = mode;
        }
        let out = self.cam_blend.step(target, dt);
        self.cam_prev = Some(out);
        // On foot with no transition running the camera is the plain formula, untouched.
        self.cam_final = (mode != CamMode::Foot || self.cam_blend.active()).then_some(out);
    }

    /// One frame: step the controller, pose the body, move the rig.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        input: &MoveInput,
        dt: f32,
        native: &NativeDll,
        cache: CacheHandle,
        scene: &dyn ObjectScene,
        objects: &[ObjectInfo],
        actors: &mut ActorScene,
    ) {
        self.frames += 1;
        self.clock += dt.clamp(0.0, 0.1);
        self.last_input = *input;
        // #vehicle-dup  While riding, count the copies of the ridden vehicle in the set the
        // renderer was actually given. One pass over the objects, and only while riding.
        self.dup_report = match &self.riding {
            Some(r) => crate::play_vehicle::duplicate_report(r, objects, self.vehicle_actors.len()),
            None => String::new(),
        };
        // #seat-entry  THE BYPASS. A seated body's transform is the VEHICLE's transform times
        // the seat marker, so the character controller is taken out of the loop entirely rather
        // than taught about vehicles — one writer on the body's position, which is the rule this
        // project keeps relearning. `StepOut::default()` is what the caption then reports, and it
        // is honest: no capsule swept, no contacts, no ground plane.
        self.seat_shot = None;
        if self.riding.is_some() {
            let before = self.riding.as_ref().map(|r| r.drive.pos);
            let left = self.drive_and_seat(input, dt, native, cache, scene);
            // #vehicle-hull  The driven hull against the static world, after its own law moved it.
            if let (Some(prev), Some(r)) = (before, self.riding.as_mut()) {
                self.props.collide_driven(prev, r, native, cache, scene, objects);
            }
            self.last_step = StepOut::default();
            // #ride-phase  The exit clip has run its authored length: NOW the rider leaves.
            if left {
                self.finish_exit(native, cache, scene, objects);
            }
        } else {
            // SPLIT THE FRAME SO NO STEP OUTRUNS THE GATHERED NEIGHBOURHOOD.
            //
            // `step` collides against the triangles cached in `self.world`, and that cache is
            // only valid near its own centre — so a step travelling further than the margin
            // sweeps the body through space whose triangles were never gathered, and it passes
            // through whatever was there. At a steady 60 Hz the body would need 180 wu/s to
            // manage it, but `dt` is clamped to 0.1 s and **a 100 ms frame needs only 30 wu/s**,
            // about three seconds of falling. This app hitches, so that is reachable, and it is
            // one of the two ways "I fell straight through the entire map" happens. Each
            // sub-tick re-gathers if it has left the cached neighbourhood, so the collision data
            // always covers the motion being resolved.
            let dts = dt.clamp(0.0, 0.1);
            let margin = WORLD_REGATHER * 0.5;
            let splits = (((self.st.vel.length() * dts) / margin).ceil() as usize)
                .max(1)
                .min(TICK_SPLIT_GUARD);
            let h = dts / splits as f32;
            let mut guard = false;
            for k in 0..splits {
                if k == 0 {
                    let hidden = self.hidden_datums();
                    self.world.set_skip(hidden);
                }
                if !self.world.covers(self.st.pos) {
                    self.world.regather(self.st.pos, scene, objects, native, cache);
                }
                let mut one = *input;
                // A jump is an EDGE: it belongs to the first sub-tick only.
                one.jump = input.jump && k == 0;
                self.last_step = self.st.step(&one, &self.phys, &self.world, h);
                guard |= self.last_step.substep_guard_tripped;
            }
            self.last_step.substep_guard_tripped = guard;
            self.last_step.substeps = self.last_step.substeps.max(splits);
            // NOTHING TO STAND ON, said out loud. A body over a void falls for ever and, before
            // this existed, did so in silence — which is why the first report of it was the
            // user's rather than ours.
            if self.world.is_void() {
                self.void_frames = self.void_frames.saturating_add(1);
            }
            // THE FAILSAFE. Below the map's own bounds by a clear margin the body is returned to
            // where it spawned. Clearly a UX guard: it does not pretend the fall was survivable,
            // it says it happened and puts the player back so the mode stays usable. Reach
            // authors no terminal velocity for a biped (the physics side measured that
            // `bipd +0x2FC` is an animation weight), so nothing here is presented as the engine's.
            if let Some((mn, _)) = scene.scene_bounds() {
                if self.st.pos.z < mn[2] - FAILSAFE_BELOW_BOUNDS {
                    let from = self.st.pos.z;
                    self.st.pos = self.spawn_feet;
                    self.st.vel = Vec3::ZERO;
                    self.failsafe_returns = self.failsafe_returns.saturating_add(1);
                    self.world.invalidate();
                    self.world.regather(self.st.pos, scene, objects, native, cache);
                    // (the world around the spawn may have changed since: a block placed on it)
                    self.st.settle_out(&self.phys, &self.world);
                    self.last_hit = Some(format!(
                        "you fell out of the world (z {from:.0}, below the map's own bounds at {:.0}) \
                         and have been returned to your spawn — {}",
                        mn[2],
                        self.world.describe()
                    ));
                }
            }
            // And the offer: which seat, if any, the player could get into from where they
            // stand. Only computed on foot — there is nothing to offer a body already in a seat.
            let facing = look_dir(self.st.yaw, self.st.pitch);
            let key = self.action_key_label.clone();
            let offer_objects = self.objects_with_parked(objects);
            self.offer = crate::play_vehicle::offer(
                &mut self.vehicles,
                native,
                cache,
                &offer_objects,
                self.st.eye(&self.phys),
                facing,
                self.st.vel,
                &key,
            );
            // #pickup  And the weapon offer, on the SAME key. Reported: "I also can't pick up any
            // of the weapons on the map".
            //
            // The vehicle takes precedence when both are in range, which is the engine's own
            // order -- one action key, and a Warthog you are standing in the door of is what you
            // meant. A pickup is still computed either way so the status line can say what was
            // passed over.
            let holding = self.weapon_tag != 0;
            let height = self.st.height(&self.phys);
            let feet = self.st.pos;
            let weapons = std::mem::take(&mut self.map_weapons);
            self.pickup = {
                let strings = self.vehicles.strings(native, cache);
                crate::play_pickup::offer(
                    native, cache, strings, objects, &weapons, feet, height, holding, &key,
                )
            };
            self.map_weapons = weapons;
        }
        // #ride-park  A vehicle left parked holds still: the handbrake is on, so it settles on
        // whatever it stands on instead of freezing in mid-air or rolling away.
        for p in self.parked.iter_mut() {
            // #play-world  A vehicle the rigid-body world simulates (a hover vehicle, a flyer, a
            // never-driven one that was knocked about) is moved by the world, not by the wheel law
            if self.props.owns_vehicle(p.ride.hidden_datum) {
                continue;
            }
            let hold = crate::reach_vehicle::DriveInput { want_yaw: p.ride.drive.yaw, brake: true, ..Default::default() };
            crate::play_vehicle::step(&mut self.vehicles, native, cache, &mut p.ride, hold, dt, scene);
        }
        // #play-world  Every other physics object. The player's capsule (on foot) or the driven
        // hull pushes; the rest of the world is simulated at the engine's 60 Hz tick.
        {
            let on_foot = self.riding.is_none();
            let player = on_foot.then(|| crate::play_world::PlayerPusher {
                feet: self.st.pos,
                velocity: self.st.vel,
                radius: self.phys.radius,
                height: self.phys.capsule_height(self.st.crouched),
                mass: self.phys.mass,
            });
            let mut pushers: Vec<crate::play_world::VehiclePusher> = Vec::new();
            if let Some(r) = &self.riding {
                pushers.push(crate::play_world::VehiclePusher { datum: r.hidden_datum, origin: r.drive.pos, rot: r.drive.basis(), velocity: r.drive.vel });
            }
            for p in &self.parked {
                if !self.props.owns_vehicle(p.ride.hidden_datum) {
                    let d = &p.ride.drive;
                    pushers.push(crate::play_world::VehiclePusher { datum: p.ride.hidden_datum, origin: d.pos, rot: d.basis(), velocity: d.vel });
                }
            }
            self.props.tick(dt, native, cache, scene, objects, player, &pushers, actors);
            self.sync_vehicle_bodies(native, cache, objects);
        }
        self.sync_vehicle_actors(native, cache, actors);
        if let Some((age, len)) = self.melee.as_mut() {
            *age += dt;
            if *age >= *len {
                self.melee = None;
            }
        }
        if let Some(age) = self.fired.as_mut() {
            *age += dt;
            // The shot owns the arms for the length of one firing animation; 0.25 s is longer
            // than any Reach `first_person:fire_1` and shorter than any reload, so the arms are
            // never stuck in it. The clip's own length governs what is DRAWN (the frame is
            // clamped to it), so this only bounds how long the state lasts.
            if *age > 0.25 {
                self.fired = None;
            }
        }
        // THE THIRD-PERSON CAMERA, solved here because it needs the scene: pulled back from the
        // eye and then stood off whatever is behind the player, so looking at your own Spartan
        // does not put the camera inside the block you just placed.
        let back = -look_dir(self.st.yaw, self.st.pitch);
        let eye = self.st.eye(&self.phys);
        let want = THIRD_PERSON_BACK;
        let clear = scene
            .raycast_solid(eye, back, want + 0.2)
            .map(|d| (d - 0.2).max(0.0))
            .unwrap_or(want);
        self.third_person_cam = eye + back * clear.min(want) + Vec3::Z * THIRD_PERSON_UP;
        self.update_camera(dt, native, cache);
        self.pose(input, native, cache, actors);
    }

    /// #ride-grip  Keep a driver's hands on the steering node: see [`crate::reach_grip`].
    ///
    /// The grip is measured, not guessed: on the first SEATED frame the authored idle pose has the
    /// hands where the seat clip put them with the wheel at rest, so each hand's frame relative to
    /// the `left_hand` / `right_hand` marker is recorded then. Every frame after, the hand's target
    /// is that frame carried by the marker as the `vehicle:steering` overlay turns its node. Off
    /// while boarding (the clip is bringing the hands to the wheel) and faded out as the exit clip
    /// starts to take them off it.
    fn update_grips(&mut self, native: &NativeDll, cache: CacheHandle, actors: &mut ActorScene, body_id: u32) {
        use crate::play_vehicle::RidePhase;
        use crate::reach_grip::{ArmChain, Grip};
        let clear = |actors: &mut ActorScene| {
            if let Some(a) = actors.get_mut(body_id) {
                if !a.grips.is_empty() {
                    a.grips.clear();
                }
            }
        };
        let Some(r) = self.riding.as_ref() else {
            self.grip_ref = [None, None];
            clear(actors);
            return;
        };
        // Only a seat on the vehicle's own hull has a steering node to hold; a turret's rider
        // holds a gun.
        if r.vehi_tag != r.placement_tag || r.phase == RidePhase::Boarding {
            clear(actors);
            return;
        }
        let weight = match r.phase {
            RidePhase::Seated => 1.0,
            _ => {
                let u = (1.0 - r.phase_age / 0.25).clamp(0.0, 1.0);
                u * u * (3.0 - 2.0 * u)
            }
        };
        let seated = r.phase == RidePhase::Seated;
        let markers: [Option<crate::reach_skel::Marker>; 2] = {
            let Some(e) = self.vehicles.entry(native, cache, r.vehi_tag) else {
                clear(actors);
                return;
            };
            let species = match self.species {
                Species::Spartan => "spartan",
                Species::Elite => "elite",
            };
            [true, false].map(|left| {
                crate::play_vehicle::hand_marker_candidates(&r.seat_label, left, species)
                    .iter()
                    .find_map(|n| e.first_marker(native, cache, n))
            })
        };
        let Some(&veh_id) = self.vehicle_actors.first() else {
            clear(actors);
            return;
        };
        let Some(veh) = actors.get(veh_id) else {
            clear(actors);
            return;
        };
        let (veh_world, veh_pose) = (veh.world(), veh.pose.clone());
        let Some(body) = actors.get(body_id) else { return };
        let body_world = body.world();
        let mut grips = Vec::new();
        for (k, left) in [(0usize, true), (1usize, false)] {
            let Some(mk) = markers[k] else { continue };
            let Some(chain) = ArmChain::find(&body.def.model.node_names, left) else { continue };
            if chain.hand >= body.pose.model.len() {
                continue;
            }
            let marker_world = veh_world * veh_pose.marker_model(&mk);
            if self.grip_ref[k].is_none() && seated {
                self.grip_ref[k] = Some(marker_world.inverse() * (body_world * body.pose.model[chain.hand]));
            }
            if let Some(g) = self.grip_ref[k] {
                grips.push(Grip { chain, target: body_world.inverse() * (marker_world * g), weight });
            }
        }
        if let Some(a) = actors.get_mut(body_id) {
            a.grips = grips;
        }
    }

    /// #aim  The two numbers an aim overlay is asked for: `(residual yaw, pitch)`, radians.
    ///
    /// The yaw is what the body has NOT turned to cover -- the difference between where the player
    /// is looking and where the biped's feet are pointing. It is zero when they agree, which is
    /// why a body that merely rotated with the camera needed no overlay and looked like a
    /// mannequin on a turntable.
    ///
    /// Same convention as a recorded body's (`film_play::ActorDriver::aim_offset`):
    /// `ang_delta(aim, body)` for the yaw and the pitch straight through.
    pub fn aim_offset(&self) -> (f32, f32) {
        (self.st.aim_residual(), self.st.pitch)
    }

    /// Install the clip, the frame and the placement on the body and the rig. Separated from
    /// [`tick`](Self::tick) so the controller can be stepped in a test with no cache at all.
    fn pose(&mut self, input: &MoveInput, native: &NativeDll, cache: CacheHandle, actors: &mut ActorScene) {
        let class = self
            .weapon_class
            .clone()
            .unwrap_or_else(|| crate::film_anim::DEFAULT_WEAPON_CLASS.to_string());
        let mut state = play_ctl::body_state(&self.st, input, &self.phys, Some(class.clone()));
        // #seat-entry  A SEATED body's pose is fixed by the seat and nothing is derived: exactly
        // what `film_anim::derive` does for a recorded rider. The seat's own `label` IS the jmad
        // mode name (`warthog_d`, `ghost_d`, `mongoose_p`), so test mode picks the exact clip
        // from the tag where Theater has to infer driver-versus-passenger from who started
        // riding first. A vehicle the shipped graph has no seat set for leaves `seat_mode` empty
        // and the body keeps its on-foot pose rather than loading a clip that does not exist.
        if let Some(r) = &self.riding {
            // #seat-entry  A seat with NO authored rider clip is a permanent correct state, not a
            // fault: 36 of `forge_halo`'s 138 player seats have none in either rider graph (every
            // `*_b_*` boarding seat, which is species-specific by design, plus `wraith_d` and
            // `sabre_d`, which no rider graph answers at all). The body is still SEATED and still
            // holds frame 0 — what it must never do is fall through to the standing
            // `combat:rifle:idle`, which draws a rider bolt upright inside the hull and reads as
            // broken. The caption names the seat so a report says which one.
            if r.seat_mode.is_empty() {
                state.stance = crate::film_anim::Stance::Seated;
                state.gait = crate::film_anim::Gait::Idle;
                state.air_ticks = None;
                state.land_ticks = None;
                state.because = "seated with NO authored rider clip: the body holds frame 0 of \
                                 whatever it had rather than standing up";
                self.last_hit = Some(format!(
                    "this seat has no rider animation in the shipped graph, so the body is held at \
                     frame 0 rather than posed: {}",
                    r.seat_caption()
                ));
            }
            if !r.seat_mode.is_empty() {
                // Built the same way `film_anim::derive` builds a recorded rider's: `Seated`,
                // idle, no air or land ticks. That last part matters — `set_pose_from_vehicle`
                // leaves `grounded` false, so `body_state` returns `Stance::Flight` with
                // `air_ticks` set, and leaving those in place would drive the seated clip's
                // frame off the airborne counter instead of holding frame 0.
                state.stance = crate::film_anim::Stance::Seated;
                state.gait = crate::film_anim::Gait::Idle;
                state.air_ticks = None;
                state.land_ticks = None;
                state.because = "seated: the body's position is the vehicle's seat marker";
                state.seat = Some(crate::film_anim::Seat {
                    vehicle_net_id: r.hidden_datum,
                    mode: r.seat_mode.clone(),
                    vehicle: r.vehi_path.rsplit(['\\', '/']).next().unwrap_or("").to_string(),
                });
            }
        }
        let Some(id) = self.body else { return };
        // --- the base clip ---------------------------------------------------------------
        //
        // The candidate list is tried in order and the first one the graph HAS wins, which is
        // what makes this robust to a weapon class the biped's graph does not author.
        //
        // The "did it change" test compares against what was ASKED FOR, not against what
        // loaded. Those differ exactly when the best candidate is missing and a later one won —
        // and comparing against the loaded name would then see a difference every single frame
        // and walk the whole candidate list again, with `set_anim` reading an animation resource
        // each time. A steady state must cost one string compare, not a resource load per frame.
        // (This is the same reason Theater's driver remembers `cands[0]` rather than the clip.)
        // #seat-entry  A seated body with no authored rider clip asks for NOTHING: the standing
        // candidates would load `combat:rifle:idle` and draw a rider standing upright inside the
        // hull. It keeps whatever it has, held at frame 0, and the caption says so.
        let seated_no_clip = state.stance == crate::film_anim::Stance::Seated && state.seat.is_none();
        // #ride-entry  The clip a ride in its boarding / leaving phase plays, and -- the first
        // frame the graph is in hand -- the AUTHORED length of both, which is what hands control
        // over (`play_vehicle::advance_phase`). A seat with no such clip has length 0: the engine
        // has nothing to wait for there, and the phase passes at once rather than hanging.
        let mut phase_cands: Vec<String> = Vec::new();
        let mut ride_frame: Option<f32> = None;
        if let Some(r) = self.riding.as_mut() {
            if r.seat_mode.is_empty() {
                r.enter_len.get_or_insert(0.0);
                r.exit_len.get_or_insert(0.0);
            } else {
                if r.enter_len.is_none() || r.exit_len.is_none() {
                    let graph = actors.get(id).and_then(|a| a.def.graph.as_ref());
                    let len_of = |ph| {
                        graph
                            .and_then(|g| {
                                crate::play_vehicle::first_authored_clip(g, &crate::play_vehicle::phase_clip_candidates(&r.seat_mode, &class, ph))
                            })
                            .map_or(0.0, |(_, secs)| secs)
                    };
                    let (e, x) = (len_of(crate::play_vehicle::RidePhase::Boarding), len_of(crate::play_vehicle::RidePhase::Exiting));
                    r.enter_len.get_or_insert(e);
                    r.exit_len.get_or_insert(x);
                }
                phase_cands = crate::play_vehicle::phase_clip_candidates(&r.seat_mode, &class, r.phase);
                // Only names the graph authors EXACTLY: `set_anim_with` resolves through the
                // substring-falling `AnimGraph::find`, which would play `...:enter_a` for a
                // missing `...:enter` or a one-frame `_pose` overlay for a missing `:exit`.
                if let Some(g) = actors.get(id).and_then(|a| a.def.graph.as_ref()) {
                    phase_cands.retain(|c| crate::play_vehicle::authors_base_clip(g, c));
                }
                if !phase_cands.is_empty() {
                    ride_frame = Some(r.phase_age * crate::film_anim::CLIP_FPS);
                }
            }
        }
        // #crouch-rate  HOW LONG a change of clip takes to cross-fade. The generic 0.18 s for
        // everything except going into or out of the crouch, which takes the biped's OWN
        // authored `crouch transition time` (`bipd +0x534`: 0.30 s Spartan, 0.20 s Elite).
        //
        // That field is confirmed twice over rather than taken from its name: the cache also
        // holds the two values the engine DERIVES from it at post-process, and both come out
        // exactly as the time predicts -- `runtime crouch transition velocity` (+0x580) reads
        // 3.3333 = 1/0.30 for the Spartan and 5.0000 = 1/0.20 for the Elite, and
        // `runtime_camera_height_velocity` (+0x584) reads 0.6000 = (0.600-0.420)/0.30 and
        // 0.9000 = (0.690-0.510)/0.20. Four numbers, four exact matches, so +0x534 really is a
        // time in seconds and this really is the rate the engine moves at.
        //
        // The eye and the capsule were already on that rate (`play_ctl::step`); the BODY was
        // not -- it dropped at the generic 0.18 s, which is 1.7x too fast for a Spartan, and
        // that is "the speed that he goes down ... is a bit fast for crouch right now".
        let fade = self.phys.crouch_fade_seconds(self.last_stance, state.stance);
        self.last_stance = state.stance;
        // --- #gait-blend  THE AUTHORED LOCOMOTION MIX ------------------------------------
        //
        // Reach does not play one clip per locomotion state: its `gaits` block authors, per
        // movement direction, two or three clips with their OWN speeds, and the engine blends
        // the pair that brackets the body's speed. HMS adds the directional half of the mix on
        // top of that -- see `reach_actor::gait_plan`, which states exactly which half is the
        // engine's and which is HMS's. When the graph authors no gaits at all, or the body is
        // not on a ground cycle, this produces nothing and the single-clip path below runs
        // unchanged.
        let mode = crate::film_anim::jmad_mode(state.stance).to_string();
        if self.gait_table.as_ref().is_none_or(|(m, c, _)| *m != mode || *c != class) {
            let t = actors
                .get(id)
                .and_then(|a| a.def.graph.as_ref())
                .map(|g| g.gait_table(&mode, &class))
                .unwrap_or_default();
            self.gait_table = Some((mode, class.clone(), t));
        }
        let plan = {
            use crate::film_anim::Stance;
            let on_cycle = !seated_no_clip
                && self.riding.is_none()
                && state.seat.is_none()
                && state.sync.is_none()
                && state.air_ticks.is_none()
                && state.land_ticks.is_none()
                && !matches!(state.stance, Stance::Flight | Stance::Dead | Stance::Seated | Stance::SyncAction);
            match (
                on_cycle.then(|| crate::film_anim::gait_direction(state.gait)).flatten(),
                state.travel_angle,
                self.gait_table.as_ref(),
            ) {
                // Directions 4 and 5 are the TURNS, which have no ground travel to drive them
                // and keep the turn-cycle path below.
                (Some(d), Some(theta), Some((_, _, t))) if d <= 3 && !t.gaits.is_empty() => {
                    crate::reach_actor::gait_plan(t, theta, state.ground_speed)
                }
                _ => None,
            }
        };
        self.gait_caption.clear();
        self.gait_stride = plan.as_ref().map(|p| p.stride);
        if let Some(p) = &plan {
            let primary = p.clips[0].0;
            let outgoing = actors.get(id).and_then(|a| a.anim);
            if outgoing != Some(primary) {
                // NO cross-fade when the outgoing clip is still in the mix: it is already being
                // posed at its own weight, and fading it in a second time would count it twice
                // -- a visible wobble every time the body passes straight ahead or sideways.
                let still_in = outgoing.is_some_and(|o| p.clips.iter().any(|c| c.0 == o));
                let f = (!still_in).then_some(fade);
                if actors.set_anim_index(native, cache, id, primary, f).is_ok() {
                    self.asked = actors.get(id).map(|a| a.anim_name.clone());
                }
            }
            for &(anim, _, _, _) in p.clips.iter().skip(1) {
                if let Some(a) = actors.get_mut(id) {
                    a.ensure_overlay(native, cache, anim);
                }
            }
            self.gait_caption = p.caption.clone();
        } else {
            // #ride-entry  Boarding and leaving play the seat's own authored clip, not the idle.
            let cands = if !phase_cands.is_empty() {
                phase_cands.clone()
            } else if seated_no_clip {
                Vec::new()
            } else if let (Some(r), Some(g)) = (self.riding.as_ref(), actors.get(id).and_then(|a| a.def.graph.as_ref())) {
                // #ride-pose  A seat is answered by the idle clips the rider's graph really
                // authors for it (`play_vehicle::seated_idle_candidates`) and never by the
                // standing `combat:` idle `anim_candidates` ends on.
                crate::play_vehicle::seated_idle_candidates(g, &r.seat_mode, &class)
            } else {
                crate::film_anim::anim_candidates(&state, &class)
            };
            if !seated_no_clip && self.asked.as_deref() != cands.first().map(|c| c.as_str()) {
                for c in &cands {
                    if actors.set_anim_with(native, cache, id, Some(c), Some(fade)).is_ok() {
                        self.asked = cands.first().cloned();
                        break;
                    }
                }
            }
        }
        // --- its frame -------------------------------------------------------------------
        // A locomotion cycle is driven by the GROUND DISTANCE travelled, so the feet match the
        // ground; a clip with nothing to drive it (idle, a turn, the airborne arc, the landing)
        // runs on the clock or on the age of the event that started it. Exactly the split
        // `film_anim` makes for a recorded body.
        // #fp-legs  Are the authored first-person LEGS the thing drawn at the player's feet this
        // frame? Only in first person, and only when this cache gave us an `fp_body` to draw.
        let fp_body_drawn = self.fp_body.is_some()
            && self.view == PlayView::FirstPerson
            && self.riding.is_none()
            && !self.camera_detached();
        // #play-rate  The ground THIS frame covered — what the cycle is advanced by. Taken as a
        // delta rather than read as a total because the phase is integrated (see `self.cycle`),
        // and clamped at zero so a controller reset cannot wind the stride backwards.
        let travelled = (self.st.distance - self.cycle_distance).max(0.0);
        self.cycle_distance = self.st.distance;
        // #play-jolt  ...and the YAW this frame covered, which is what drives a turn in place.
        // Entering a turn resets the phase, so the step starts at the start of the step instead
        // of wherever the session clock happened to be.
        let turning = matches!(state.gait, crate::film_anim::Gait::TurnLeft | crate::film_anim::Gait::TurnRight);
        let turned = match (turning, self.turn_from) {
            (false, _) => {
                self.turn_from = None;
                self.turn_cycle = 0.0;
                0.0
            }
            (true, None) => {
                self.turn_from = Some(self.st.facing);
                0.0
            }
            (true, Some(was)) => {
                self.turn_from = Some(self.st.facing);
                crate::film_anim::ang_delta_pub(self.st.facing, was).abs()
            }
        };
        // #gait-blend  The extra base clips are rewritten every frame (below) or cleared, so a
        // mix can never outlive the state that asked for it.
        if let Some(a) = actors.get_mut(id) {
            a.base_blend.clear();
        }
        if let Some(a) = actors.get_mut(id) {
            a.playing = false;
            let n = a.frame_count().max(1.0);
            let frame = if seated_no_clip {
                0.0
            } else if let Some(f) = ride_frame {
                // #ride-entry  A one-shot at the clip's own 30 fps, then HELD on its last frame
                // (the enter clip's end IS the seated pose; the exit clip's end is standing).
                f.min(n - 0.001).max(0.0)
            } else if let Some(air) = state.air_ticks {
                // One-shot, then HOLD: an arc driven by a wrapping phase replays over and over,
                // which is "the airborne animation is looping".
                ((air as f32 / 2.0).min(n - 0.001)).max(0.0)
            } else if let Some(land) = state.land_ticks {
                ((land as f32 / 2.0).min(n - 0.001)).max(0.0)
            } else if let Some(p) = &plan {
                // #gait-blend  ONE integrated phase for the WHOLE mix.
                //
                // `p.stride` is the weighted mean of the clips' own authored strides, so the
                // phase advances by `travelled / stride` -- still an accumulator, never
                // `(total distance x rate) mod 1`, which is the defect that made the walk
                // "very slidy" and is fixed once.
                //
                // **Whose rate drives it:** none of them individually. Every clip in the mix is
                // read at the SAME normalised phase and the rate is the weighted mean stride, so
                // the two clips of a speed-tier pair -- authored at, say, 0.8607 and 2.1004 wu/s
                // -- cannot tear against each other: their feet are at the same fraction of
                // their own cycle at every instant, which is the only arrangement where blending
                // two walk cycles produces a walk and not a shuffle. At a single clip and weight
                // 1 the expression reduces to the single-clip rule it replaces.
                self.cycle = (self.cycle + travelled / p.stride).rem_euclid(1.0);
                a.base_blend = p
                    .clips
                    .iter()
                    .skip(1)
                    .map(|&(anim, w, frames, _)| {
                        (anim, (self.cycle * frames).clamp(0.0, (frames - 0.001).max(0.0)), w)
                    })
                    .collect();
                self.cycle * n
            } else if let Some(cpw) = self
                .phys
                .gait_speed(state.stance, state.gait)
                .and_then(|v| crate::film_anim::cycles_per_wu_at_speed(v, n))
            {
                self.cycle = (self.cycle + travelled * cpw).rem_euclid(1.0);
                // #play-rate  THE STRIDE IS THE AUTHORED ONE, not the film corpus's.
                //
                // `cycles_per_wu_at_speed(v, n)` is `30 / (v * n)`, so one cycle of this clip is
                // `v * n / 30` wu of ground -- the distance the body covers while the clip runs
                // once at its authored 30 fps at the authored speed `v` for THIS stance and this
                // direction. Driving the phase from the distance actually travelled then puts the
                // feet on the ground at any speed, and plays the clip at exactly 30 fps at the
                // authored one.
                //
                // It used to be `cycles_per_wu_for_clip`, whose speed comes from a film-corpus
                // median per GAIT -- which carries no stance, so a crouching body was given the
                // standing clip's speed, and whose run-forward median (3.52 wu/s) is the
                // SPRINT's (2.200 x 1.623 = 3.571) because the corpus histogram's dominant mode
                // is sprinting players. `film_anim::cuts` has already had that correction made
                // to it; the rate had not. Measured on `forge_halo` (the probe prints the whole
                // table): a Spartan walking was playing the walk cycle at 11.6 fps instead of
                // 30, strafing at 38.8, and crouch-walking at **7.7** -- which is the report,
                // "its very slidy as well especially crouching and strafing".
                self.cycle * n
            } else if turning && self.phys.turning_speed > 1e-4 {
                // #play-jolt  A TURN IN PLACE is driven by the yaw the body has actually covered,
                // the exact analogue of a stride driven by the ground it has covered. One cycle of
                // the clip is `turning speed x frames / 30` radians — the angle the body sweeps
                // while the clip runs once at its authored 30 fps at the authored
                // `bipd +0x4CC moving turning speed` (486 deg/s on both playable bipeds) — so the
                // legs step in lockstep with the rotation at any turn rate, and the clip starts at
                // frame 0 when the turn starts.
                let cpr = crate::film_anim::cycles_per_wu_at_speed(self.phys.turning_speed, n)
                    .unwrap_or(0.0);
                self.turn_cycle = (self.turn_cycle + turned * cpr).rem_euclid(1.0);
                self.turn_cycle * n
            } else {
                // #play-jolt  Clock-driven, at the authored 30 fps on the WALL CLOCK, WRAPPED --
                // which is what the engine does, because these clips DO loop.
                //
                // #walk-bob  This used to play forward-then-backward on the strength of a
                // measurement that the idle clips "are not loops" (last frame 0.0426 wu from the
                // first, 73x an ordinary step; crouch idle 477x). That measurement was the curve
                // sampler losing the pelvis translation on the last 7 frames of every record that
                // ends its block (`reach_jmad::curve_translation`), so the last frames fell back
                // to the bind-pose pelvis and "popped". With the sampler fixed the same probe
                // (`probe_whether_the_clock_driven_clips_loop`) reads wrap = 0.2x p90 for the
                // Spartan idle and 0.6x for the crouch idle (Elite 0.5x / 1.0x): every one loops,
                // and the reversed breathing at half rate was a workaround for our own bug.
                (self.clock * crate::film_anim::CLIP_FPS) % n
            };
            a.frame = frame.min(n - 0.001).max(0.0);
            a.pitch = 0.0;
            self.body_clip_state = crate::play_sound::BodyClip { anim: a.anim, frame: a.frame, frames: n };
            // THE BODY IS DRAWN AT ITS OWN FACING, NOT AT THE CAMERA'S AIM. Reported: *"the
            // biped doesn't like pivot on its feet at all, the whole biped seems to just rotate
            // with the camera which is wrong"*. It did, because this line was `self.st.yaw` --
            // the mouse. `play_ctl` now carries them as two quantities (see
            // `PlayerState::facing`), the feet follow at the biped's authored turning speed, and
            // the difference goes into the aim overlay below, which is how Reach poses a look
            // direction.
            a.yaw = self.st.facing.to_degrees();
            // In first person the body is parked out of the shot rather than destroyed, so its
            // clip and pose stay truthful (and the caption can report them) while the camera is
            // not looking through its chest.
            // THE BODY IS DRAWN IN FIRST PERSON TOO. Reported: "I see my FP arms, but NOT my FP
            // legs when I look down."
            //
            // It used to be parked out of the shot, because that is what the engine does — Reach
            // has no reason to draw what you cannot see, and its `@player representation` first
            // person model is arms-only. **This is one of the few places where engine fidelity and
            // what the mode is FOR legitimately diverge**: a player walking their own Forge
            // geometry needs to see where their feet are relative to a block they have just
            // placed, and that is the entire reason test mode exists. So the authored
            // third-person body — posed by this controller, nothing invented — is drawn in both
            // views, and it is HMS's choice rather than the engine's.
            //
            // The cost, stated: the camera sits at the eye, so the player's own head and neck
            // intersect the near plane and will clip. Suppressing just the head needs a region →
            // section mapping the actor pipeline does not expose today; whole-body-with-clipping
            // is the deliberate trade, because the ask was legs, not perfection.
            //
            // It also means the weapon in the biped's HAND is visible in first person, which is a
            // second thing the user reported missing.
            //
            // #fp-legs  **AND THAT WENT TOO FAR.** Reported next: "there is also a duplicate
            // biped model in first person now" — because the arms rig and a whole third-person
            // Spartan were both on screen, so the player saw two pairs of arms. The engine
            // authors a model for exactly this job and `@player representation` names it: the
            // FIRST-PERSON BODY. Measured on `forge_halo`, the Spartan's is 5 meshes over the
            // regions `decals` and `flair_knees` — legs and hips, no head, no arms, no torso —
            // on the same 82-node skeleton as the biped, so the body's own locomotion clip poses
            // it. So the LEGS stay (that was the whole ask) and the duplicate goes: in first
            // person the third-person biped is parked and `fp_body` is drawn in its place, posed
            // from the same clip at the same frame. A cache with no `fp_body` keeps the previous
            // behaviour and says so in the notes, because "no legs" is better than silence.
            a.pos = if self.view == PlayView::FirstPerson && fp_body_drawn { PARK } else { self.st.pos };
        }
        // #ride-seat  A SEATED body is placed by the seat marker's whole frame, and it is DRAWN
        // whenever the camera is on a chase shot -- the engine does not park your own unit in
        // third person. A first-person seat keeps it out of the lens.
        {
            let seat_world = match &self.riding {
                Some(r) => crate::play_vehicle::rider_world(&mut self.vehicles, native, cache, r),
                None => None,
            };
            let riding = self.riding.is_some();
            let chase = self.cam_mode == crate::reach_unit_camera::CamMode::Chase;
            if let Some(a) = actors.get_mut(id) {
                match (riding, seat_world) {
                    (true, Some(w)) if chase => a.world_override = Some(w),
                    (true, _) => {
                        a.world_override = None;
                        a.pos = PARK;
                    }
                    (false, _) => a.world_override = None,
                }
            }
        }
        self.update_grips(native, cache, actors, id);
        // #ride-weapon  A weapon the seat does not allow is not drawn, in either view.
        let hide_weapon = !self.weapon_usable();
        if let Some(a) = actors.get_mut(id) {
            a.weapon_hidden = hide_weapon;
        }
        // #fp-legs  The legs, posed from the BODY's own clip and frame — read back off the body
        // actor rather than re-derived, so the two models cannot disagree about which stride the
        // player is mid-way through, and so a candidate the biped's graph answered with a
        // different name reaches the legs as the name that actually loaded.
        if let Some(lid) = self.fp_body {
            // #gait-blend  The MIX comes across too, not just the primary clip and its frame.
            // The legs are posed by the biped's OWN graph (see `mount_first_person`), so a graph
            // animation index means the same clip on both actors and the extra lanes transfer by
            // index. Without this the legs would play the heaviest clip alone while the body
            // played the blend, and a first-person player's feet would disagree with their own
            // third-person shadow.
            let from = self
                .body
                .and_then(|b| actors.get(b))
                .map(|a| (a.anim_name.clone(), a.anim, a.frame, a.yaw, a.base_blend.clone()));
            if let Some((name, anim, frame, yaw, mix)) = from {
                let asked_changed = !name.is_empty() && self.fp_legs_asked.as_deref() != Some(name.as_str());
                if asked_changed {
                    let ok = match anim {
                        Some(i) => actors.set_anim_index(native, cache, lid, i, Some(crate::reach_actor::BLEND_SECONDS)).is_ok(),
                        None => actors.set_anim(native, cache, lid, Some(&name)).is_ok(),
                    };
                    if ok {
                        self.fp_legs_asked = Some(name);
                    }
                }
                for &(a, _, _) in &mix {
                    if let Some(l) = actors.get_mut(lid) {
                        l.ensure_overlay(native, cache, a);
                    }
                }
                if let Some(a) = actors.get_mut(lid) {
                    a.playing = false;
                    let n = a.frame_count().max(1.0);
                    a.frame = frame.min(n - 0.001).max(0.0);
                    a.base_blend = mix;
                    a.yaw = yaw;
                    a.pitch = 0.0;
                    a.pos = if fp_body_drawn { self.st.pos } else { PARK };
                }
            }
        }
        // --- the aim overlay -------------------------------------------------------------
        //
        // #aim  How the body looks where the player is looking. Reported: *"the third person unit
        // also doesn't move their head/torso or anything as I look up or down at all, its stuck
        // aiming straight ahead"* -- it was, because nothing was ever pushed into `a.overlays`
        // for the body (only for the first-person arms).
        //
        // Reach poses a look direction as an OVERLAY added on top of whatever locomotion clip is
        // running, and the clip itself carries what each of its frames means
        // (`reach_actor::AimGrid`), so the two numbers handed in are all that is needed: the yaw
        // the body has NOT turned to cover, and the pitch. Composed after the base clip is chosen
        // and framed, for the reason `film_play` gives: an overlay added to a pose is a look
        // direction, an overlay installed AS the pose cancels the stride.
        //
        // Deliberately the same path Theater drives (`aim_candidates` -> `aim_grid` -> `cells`),
        // not a second one: a body in test mode and a body in a film are the same body.
        //
        // **Not while seated.** A seated body's whole skeleton belongs to the seat animation (see
        // `Stance::Seated` above), so composing a look direction onto it would fight a pose that
        // is already complete. The vehicle side asked for this gate explicitly and it is the same
        // call `film_anim::aim_candidates` makes for a recorded rider -- it returns an empty list
        // for a seat -- so this is belt and braces on a rule that is already stated once.
        self.aim_missing = false;
        // THE CLEAR IS UNCONDITIONAL, the compose is not. An overlay left over from the last
        // on-foot frame would otherwise stay composed onto the seated pose for the whole ride --
        // the body would get into a Warthog still wearing the look direction it had when it
        // walked up to it. "Stop computing it" and "stop applying it" are two different
        // instructions and only the first one is about riding.
        if let Some(a) = actors.get_mut(id) {
            a.overlays.clear();
        }
        // #ride-look  A SEATED rider looks around too: the seat graph authors its own aim /
        // look overlay (`warthog_p:rifle:aim_still_up`, `mongoose_d:look_still_up`, ...), driven
        // by the look relative to the hull and limited by the seat's own range, exactly as the
        // on-foot one is driven by the look relative to the body.
        let aim_job = match &self.riding {
            None => Some((self.aim_offset(), crate::film_anim::aim_candidates(&state, &class))),
            Some(r) if r.phase == crate::play_vehicle::RidePhase::Seated && !r.seat_mode.is_empty() => {
                crate::play_vehicle::seat_aim(&mut self.vehicles, native, cache, r)
                    .map(|a| (a, crate::play_vehicle::seat_aim_candidates(&r.seat_mode, &class)))
            }
            Some(_) => None,
        };
        if let Some(((dy, dp), cands)) = aim_job {
            let best = cands.first().cloned();
            if self.aim_asked != best {
                // The state changed, so re-walk the list ONCE and remember which name answered.
                self.aim_asked = best;
                self.aim_use = None;
                if let Some(a) = actors.get_mut(id) {
                    for c in &cands {
                        if a.aim_grid(native, cache, c).is_some() {
                            self.aim_use = Some(c.clone());
                            break;
                        }
                    }
                }
            }
            if let Some(a) = actors.get_mut(id) {
                if !cands.is_empty() {
                    let mut got = false;
                    if let Some(c) = self.aim_use.clone() {
                        if let Some((anim, grid)) = a.aim_grid(native, cache, &c) {
                            let cells = grid.cells(dy, dp);
                            if !cells.is_empty() {
                                a.overlays.push((anim, cells));
                                got = true;
                            }
                        }
                    }
                    // A graph with no aim overlay at all is reported, not silently straight-ahead:
                    // that is exactly the symptom this block was written for.
                    self.aim_missing = !got;
                }
            }
        }
        // --- the melee swing, in the REPLACEMENT lane ------------------------------------
        let swing = self.melee.map(|(age, _)| age);
        if let Some(a) = actors.get_mut(id) {
            a.replacement = None;
            a.replacement_name.clear();
        }
        if let Some(age) = swing {
            let cands = crate::film_anim::melee_candidates(
                &class,
                state.stance == crate::film_anim::Stance::Flight,
                self.melee_variant,
            );
            let mut installed = None;
            if let Some(a) = actors.get_mut(id) {
                for c in &cands {
                    // The lane only accepts a type-3 REPLACEMENT clip, so a graph whose melee is
                    // authored some other way is reported rather than posed wrongly.
                    if let Some((anim, n)) = a.replacement_clip(native, cache, c) {
                        a.replacement = Some(((anim), (age * crate::film_anim::CLIP_FPS).min(n - 0.001).max(0.0)));
                        a.replacement_name = c.clone();
                        installed = Some(n / crate::film_anim::CLIP_FPS);
                        break;
                    }
                }
            }
            match installed {
                // The swing lasts exactly as long as the clip the graph actually has.
                Some(len) => {
                    if let Some(m) = self.melee.as_mut() {
                        m.1 = len;
                    }
                }
                None => {
                    self.melee = None;
                    self.last_hit = Some(format!(
                        "this graph has no REPLACEMENT melee clip for class '{class}' (tried {})",
                        cands.join(", ")
                    ));
                }
            }
        }
        self.body_clip = actors.get(id).map(|a| a.anim_name.clone()).unwrap_or_default();
        self.body_replacement = actors.get(id).map(|a| a.replacement_name.clone()).unwrap_or_default();
        // --- the first-person rig --------------------------------------------------------
        if !self.fp.is_empty() {
            let eye = self.st.eye(&self.phys);
            let (bases, oneshot, overlays) = crate::film_play::ActorDriver::fp_clips(
                self.melee.map(|(age, _)| (age * 60.0) as u32),
                self.fired.map(|age| (age * 60.0) as u32),
                state.stance == crate::film_anim::Stance::Sprint,
                !self.st.grounded,
                state.gait != crate::film_anim::Gait::Idle,
            );
            // #ride-camera  The arms belong to the first-person eye: gone while seated and while
            // the camera is still on its way back from a chase shot.
            let show = self.view == PlayView::FirstPerson && self.riding.is_none() && !self.camera_detached();
            let ids = self.fp.clone();
            // Same rule as the body's base clip: remember what was ASKED for, so a weapon whose
            // first-person graph lacks the best candidate does not re-walk the list and re-read a
            // resource on every frame.
            let want = bases.first().copied().unwrap_or("");
            let changed = self.fp_asked.as_deref() != Some(want);
            for fid in ids {
                let mut chosen = self.fp_clip.clone();
                if changed {
                    for c in &bases {
                        if actors.set_anim(native, cache, fid, Some(c)).is_ok() {
                            chosen = (*c).to_string();
                            break;
                        }
                    }
                }
                if let Some(a) = actors.get_mut(fid) {
                    a.playing = false;
                    let n = a.frame_count().max(1.0);
                    a.frame = match oneshot {
                        Some(age) => ((age as f32 / 2.0).min(n - 0.001)).max(0.0),
                        // #play-jolt  Forward then backward, for the same measured reason the
                        // body's idle is (see `film_anim::ping_pong_frame`): a clip that does not
                        // loop pops once a cycle, and the first-person rig is the one the camera
                        // is closest to.
                        None => crate::film_anim::ping_pong_frame(self.clock, n),
                    };
                    a.pos = if show { eye } else { PARK };
                    a.yaw = self.st.yaw.to_degrees();
                    a.pitch = self.st.pitch.to_degrees();
                    a.model_yaw = crate::reach_actor::FP_ARMS_MODEL_YAW;
                }
                // The overlays compose onto that base — never installed as a base themselves,
                // which is the bug that drew the arms' bind pose in Theater. `clock_overlay`
                // REFUSES a clip the tag does not mark as an overlay, so a base clip cannot be
                // installed here by mistake.
                if let Some(a) = actors.get_mut(fid) {
                    a.overlays.clear();
                    // `clock_frame` takes a playhead in 60 Hz TICKS, which is the clock in
                    // seconds times the tick rate.
                    let t = self.clock * 60.0;
                    for o in &overlays {
                        if let Some((anim, frames)) = a.clock_overlay(native, cache, o) {
                            let f = crate::film_play::clock_frame(t, frames);
                            a.overlays.push((anim, vec![(f, 1.0)]));
                        }
                    }
                }
                if !chosen.is_empty() {
                    self.fp_clip = chosen;
                }
            }
            if changed {
                self.fp_asked = Some(want.to_string());
            }
        }
    }

    /// Where the camera goes this frame: `(pos, yaw, pitch)`.
    pub fn camera(&self) -> (Vec3, f32, f32) {
        // #ride-camera  On a chase shot, a seat camera, or in the middle of a transition between
        // modes the pose is the one `update_camera` produced (the engine's blend included).
        // Otherwise it is the plain on-foot formula, exactly as before vehicles existed.
        if let Some(p) = self.cam_final {
            return (p.eye, p.yaw, p.pitch);
        }
        let p = self.foot_pose();
        (p.eye, p.yaw, p.pitch)
    }

    /// Take every actor test mode created out of the scene. Leaving must not leave a Spartan
    /// standing in the editor.
    pub fn leave(&mut self, actors: &mut ActorScene) {
        // #play-world  Every prop back at its saved pose, every prop actor gone.
        self.props.leave(actors);
        // #seat-entry  Including the ridden vehicle's own actors: leaving test mode while sitting
        // in a Warthog must not leave a second Warthog in the editor beside the real one.
        let mut ids: Vec<u32> = self.vehicle_actors.drain(..).collect();
        // #ride-park  ...and every vehicle left parked.
        for p in self.parked.drain(..) {
            ids.extend(p.actors);
        }
        ids.extend(self.fp.drain(..));
        // #fp-legs  Including the first-person legs: leaving test mode must not leave a pair of
        // disembodied Spartan legs standing in the editor.
        if let Some(l) = self.fp_body.take() {
            ids.push(l);
        }
        self.riding = None;
        if let Some(b) = self.body.take() {
            ids.push(b);
        }
        actors.actors.retain(|a| !ids.contains(&a.id));
        actors.rebuild = true;
    }

    /// The status line: what the player is doing and what is posing them.
    pub fn status(&self) -> String {
        let mut s = format!(
            "TEST MODE — {} | {} | {}",
            self.species.label(),
            match self.view {
                PlayView::FirstPerson => "first person",
                PlayView::ThirdPerson => "third person",
            },
            self.st.describe(&self.phys)
        );
        if let Some(r) = &self.riding {
            // #vehicle-dup  Print-only, and unconditional: the user double-clicks the app and sets
            // no env vars, so a diagnostic that needs one cannot answer a report.
            let _ = r;
            s += &format!("\n{}", self.dup_report);
            s += &format!(
                "\nRIDING {} seat {} '{}'{} | {}",
                r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path),
                r.seat,
                r.seat_label,
                if r.seat_mode.is_empty() { "  (no seat animation in the shipped graph)" } else { "" },
                match &r.drivable {
                    Ok(()) => format!("{:.2} wu/s", glam::Vec3::new(r.drive.vel.x, r.drive.vel.y, 0.0).length()),
                    Err(why) => format!("NOT DRIVABLE: {why}"),
                }
            );
            // #ride-camera  What the camera is doing and how far into the boarding clip we are, so
            // a report says which phase and which camera mode it was looking at.
            s += &format!(
                "\nride phase {:?} {:.2}s (clips {} in / {} out) | camera {:?}{}",
                r.phase,
                r.phase_age,
                r.enter_len.map_or("?".into(), |l| format!("{l:.2}s")),
                r.exit_len.map_or("?".into(), |l| format!("{l:.2}s")),
                self.cam_mode,
                if self.cam_blend.active() { format!(" (blending, {:.2}s left)", self.cam_blend.remaining()) } else { String::new() }
            );
        } else if let Some(o) = &self.offer {
            s += &format!("\n{}  [{}]", o.caption(), o.test.describe());
        } else if let Some(o) = &self.pickup {
            // #pickup  The reach is reported with the offer, because it is OURS and the player
            // should be able to see the number that decided.
            s += &format!(
                "\n{}  [{} at {:.2} wu of the {:.2} wu reach (HMS's, not authored)]",
                o.caption(),
                o.leaf(),
                o.distance,
                crate::play_pickup::PICKUP_REACH_WU
            );
        }
        s += &format!(
            "\nbody clip: {}{}{} | first person: {} | {} tri(s) cached, {} gather(s), {} contact(s) {} steep this frame",
            if self.body_clip.is_empty() { "<none>" } else { &self.body_clip },
            if self.body_replacement.is_empty() {
                String::new()
            } else {
                format!("  + REPLACEMENT {}", self.body_replacement)
            },
            // #gait-blend  The MIX, so the caption says what is actually being posed rather
            // than naming only the heaviest clip.
            if self.gait_caption.is_empty() {
                String::new()
            } else {
                format!("  [gait mix: {}]", self.gait_caption)
            },
            if self.fp_clip.is_empty() { "<not drawn>" } else { &self.fp_clip },
            self.world.len(),
            self.world.gathers,
            self.last_step.contacts,
            self.last_step.steep_contacts
        );
        s += &format!(" | {} frame(s) over {:.1}s", self.frames, self.clock);
        // #aim  The two angles the body is actually drawn at, and the overlay request derived
        // from them. This is the readout a scripted gate asserts on, so it prints the facing and
        // the residual separately rather than only the look direction.
        let (dy, dp) = self.aim_offset();
        s += &format!(
            "\naim: looking {:.1} deg / pitch {:.1} deg, body facing {:.1} deg (turning {:.1} deg/s) -> overlay asked for yaw {:.1} deg pitch {:.1} deg{}",
            self.st.yaw.to_degrees(),
            self.st.pitch.to_degrees(),
            self.st.facing.to_degrees(),
            self.st.facing_rate.to_degrees(),
            dy.to_degrees(),
            dp.to_degrees(),
            if self.aim_missing { "  <-- THIS GRAPH HAS NO AIM OVERLAY: the body is posed straight ahead" } else { "" }
        );
        s += &format!("\n{}", self.world.describe());
        if self.world.is_void() {
            s += "  <-- NOTHING TO COLLIDE WITH HERE: the body will fall";
        }
        if self.void_frames > 0 {
            s += &format!("\n{} frame(s) spent over a void", self.void_frames);
        }
        if self.failsafe_returns > 0 {
            s += &format!(
                "\nthe out-of-world failsafe has returned you {} time(s) (a UX guard, not physics)",
                self.failsafe_returns
            );
        }
        if self.last_step.substep_guard_tripped {
            s += "\nWARNING: this frame moved further than the collision solver can safely                   subdivide, so the body may have stepped over thin geometry";
        }
        s += &format!(
            "\nholding: {}{}",
            if self.weapon_tag == 0 { "nothing".into() } else { self.weapon_path.clone() },
            match self.melee_reach {
                Some(r) => format!(" | authored melee lunge range {r:.2} wu | {} shot(s) fired", self.shots),
                None => format!(" | no authored melee parameters | {} shot(s) fired", self.shots),
            }
        );
        if let Some(h) = &self.last_hit {
            s += &format!("\nlast action: {h}");
        }
        s
    }

    /// Everything read at entry, for the caption and the live gate.
    pub fn report(&self) -> String {
        self.notes.join("\n")
    }
}

/// #play  Resolve a species to `(biped tag, tag path, which `@player information` row)`.
///
/// **The authored route first.** `matg @player representation` states, per playable species, the
/// `third person unit` — the `bipd` that species' player actually IS — and the row index is also
/// the pairing with `@player information`, so the body and the speeds come from the same row.
/// That is the engine's own answer; matching tag-path substrings is only the fallback for a cache
/// whose representation block cannot be read.
///
/// Which row is which species is read off the biped's own tag path
/// (`film_anim::is_elite`), never assumed from the index.
pub fn resolve_species(
    native: &NativeDll,
    cache: CacheHandle,
    bipeds: &[(u32, String)],
    species: Species,
) -> Result<(u32, String, usize, &'static str), String> {
    let path_of = |t: u32| -> String {
        bipeds
            .iter()
            .find(|(x, _)| *x == t)
            .map(|(_, p)| p.clone())
            .or_else(|| native.tag_name(cache, t))
            .unwrap_or_default()
    };
    for rep in play_phys::PlayerPhysics::player_representations(native, cache) {
        if rep.unit == 0xFFFF_FFFF || rep.unit == 0 {
            continue;
        }
        let path = path_of(rep.unit);
        if path.is_empty() {
            continue;
        }
        if crate::film_anim::is_elite(&path) == species.is_elite() {
            // The row states which `@player information` it uses; that stated index is the
            // pairing, not the row's own position.
            return Ok((rep.unit, path, rep.info_row, "matg @player representation `third person unit`"));
        }
    }
    let (tag, path) = pick_by_hints(bipeds, species.bipd_hints())
        .ok_or_else(|| format!("no {} biped in this map ({} bipd tags)", species.label(), bipeds.len()))?;
    Ok((tag, path, usize::from(species.is_elite()), "the biped tag path (the representation block could not be read)"))
}

/// #play  The movement constants for one species, with the tag path that supplied them — for the
/// `play phys` verb, so the values can be read off a real map and checked rather than taken on
/// trust.
pub fn player_physics_for(
    native: &NativeDll,
    cache: CacheHandle,
    bipeds: &[(u32, String)],
    species: Species,
) -> Result<(String, PlayerPhysics), String> {
    let (tag, path, row, how) = resolve_species(native, cache, bipeds, species)?;
    let p = PlayerPhysics::read(native, cache, tag, row)?;
    Ok((format!("{path}  (row {row} via {how})"), p))
}

/// #play  One line about a hit, for the caption.
///
/// **An undetermined damage is printed as undetermined.** `DamageReport::applied` is an `Option`
/// because the `jpt!` falloff radii are not measured yet, so some hits genuinely have no
/// determinable damage — and printing `0` there would read as "took no damage", i.e. as a bug in
/// the swing rather than as the honest gap it is. Same for the damage SECTION: the engine picks
/// it through the collision model's MATERIAL rather than from the hit point, so on a
/// multi-section model (a Warthog) the section is not resolvable yet and that is said rather
/// than guessed.
pub fn describe_damage(datum: u32, dist: f32, r: &crate::play_damage::DamageReport) -> String {
    let dmg = match r.applied {
        Some(v) => format!("{v:.2} damage"),
        None => "damage not determined (the jpt! falloff is not measured yet)".to_string(),
    };
    let sections = if r.sections_changed.is_empty() {
        "no damage section changed state".to_string()
    } else {
        r.sections_changed
            .iter()
            .map(|(n, st)| format!("{n} -> state {st}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!("hit {datum:#x} at {dist:.2} wu: {dmg} | {sections}{}{}", if r.note.is_empty() { "" } else { " | " }, r.note)
}

/// A unit forward vector from a yaw and a pitch, radians — the engine's convention (0 = +X).
pub fn look_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    Vec3::new(cp * cy, cp * sy, sp)
}

/// Pick a `(tag, path)` whose path matches one of `hints`, best hint first. An exact leaf match
/// beats a substring, so `elite` does not pick `elite_jetpack` when the plain one exists.
fn pick_by_hints(list: &[(u32, String)], hints: &[&str]) -> Option<(u32, String)> {
    for h in hints {
        let mut best: Option<(usize, u32, String)> = None;
        for (t, p) in list {
            let leaf = p.rsplit(['\\', '/']).next().unwrap_or(p).to_lowercase();
            if leaf == *h {
                return Some((*t, p.clone()));
            }
            if leaf.contains(h) && best.as_ref().map(|b| leaf.len() < b.0).unwrap_or(true) {
                best = Some((leaf.len(), *t, p.clone()));
            }
        }
        if let Some((_, t, p)) = best {
            return Some((t, p));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in scene: a floor at z = 0 and nothing else. At module level because three gates
    /// need it, not one, and `pub(super)` because the map-backed gates next door drive the real
    /// `PlayMode::tick` against it.
    pub(super) struct Floor;
    impl crate::objscene::ObjectScene for Floor {
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
        // #ride-gate  A vehicle's wheels and a chase camera cast through the `_excluding` form,
        // whose trait default is built from `raycast_scene` + objects (both empty here).
        fn raycast_solid_excluding(&self, origin: Vec3, dir: Vec3, max: f32, _x: &[u32]) -> Option<f32> {
            self.raycast_solid(origin, dir, max)
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


    fn list() -> Vec<(u32, String)> {
        vec![
            (1, "objects\\characters\\elite\\elite_jetpack.biped".into()),
            (2, "objects\\characters\\elite\\elite_mp.biped".into()),
            (3, "objects\\characters\\spartans\\spartan_mp.biped".into()),
            (4, "objects\\weapons\\rifle\\assault_rifle\\assault_rifle.weapon".into()),
        ]
    }

    /// #play-gate  The species pick must not take a jetpack Elite for the plain one, and must not
    /// take the Elite when asked for a Spartan.
    #[test]
    fn the_biped_pick_prefers_the_plain_player_biped() {
        let l = list();
        assert_eq!(pick_by_hints(&l, Species::Elite.bipd_hints()).unwrap().0, 2);
        assert_eq!(pick_by_hints(&l, Species::Spartan.bipd_hints()).unwrap().0, 3);
        assert_eq!(pick_by_hints(&l, &["assault_rifle"]).unwrap().0, 4);
        assert!(pick_by_hints(&l, &["warthog"]).is_none());
    }

    /// #play-gate  The world cache re-gathers when the player walks out of it, and NOT before —
    /// which is the whole of the performance promise.
    #[test]
    fn the_triangle_gather_is_reused_until_the_player_leaves_it() {
        let mut w = PlayWorld::default();
        assert!(!w.covers(Vec3::ZERO), "an ungathered world covers nothing");
        w.centre = Vec3::ZERO;
        w.valid = true;
        assert!(w.covers(Vec3::new(WORLD_REGATHER - 0.1, 0.0, 0.0)));
        assert!(!w.covers(Vec3::new(WORLD_REGATHER + 0.1, 0.0, 0.0)));
        // And the box is bigger than the trip, so the triangles a step needs are always inside it.
        assert!(WORLD_BOX > WORLD_REGATHER, "the player could leave the gathered box before it is refreshed");
    }

    /// #play-gate  SPAWNING USES THE CAMERA, NOT THE MAP'S SPAWN POINT.
    ///
    /// Reported: "we also spawn way outside the actual map variant, we need to create the spartan
    /// biped right where the camera was." The earlier order had the scenario spawn first, which
    /// on a Forge variant can be nowhere near what the user is building. This pins the three
    /// things that fix had to get right: the feet are under the CAMERA, the yaw is the camera's,
    /// and the body is dropped to the surface rather than left at the camera's altitude.
    #[test]
    fn the_spawn_is_the_camera_dropped_to_the_ground() {
        let scene = Floor;
        let eye_height = 0.600f32;
        let cam = Vec3::new(12.0, -34.0, 20.0);
        // The arithmetic `enter` does, in one place so the gate can assert it without a cache:
        // feet under the camera, dropped onto the surface below it.
        let feet = {
            let mut f = cam - Vec3::Z * eye_height;
            if let Some(d) = crate::objscene::ObjectScene::raycast_solid(&scene, cam, -Vec3::Z, 500.0) {
                f.z = cam.z - d;
            }
            f
        };
        assert_eq!((feet.x, feet.y), (cam.x, cam.y), "the feet must be under the CAMERA, not at a spawn marker");
        assert!((feet.z - 0.0).abs() < 1e-5, "the body must be dropped to the floor, got z {}", feet.z);
        // And the far-away spawn point is reachable from the scene, so the gate is not passing
        // because the stand-in has no spawn point to prefer.
        assert_eq!(
            crate::objscene::ObjectScene::spawn_camera_pose(&scene).map(|p| p.0),
            Some(Vec3::new(500.0, 500.0, 50.0))
        );
    }

    /// #play-gate  The look direction is the engine's convention: yaw 0 is +X, and a positive
    /// pitch looks up.
    #[test]
    fn the_look_direction_matches_the_engines_convention() {
        let d = look_dir(0.0, 0.0);
        assert!((d - Vec3::X).length() < 1e-6, "{d}");
        let up = look_dir(0.0, std::f32::consts::FRAC_PI_4);
        assert!(up.z > 0.7 && up.x > 0.7, "{up}");
        let left = look_dir(std::f32::consts::FRAC_PI_2, 0.0);
        assert!((left - Vec3::Y).length() < 1e-5, "{left}");
    }
}

// =============================================================================================
// #play-rate  MAP-BACKED GATES — the real cache, the real actors, the production entry point
// =============================================================================================

#[cfg(test)]
mod map_gates {
    //! `cargo test -p hms-app --profile quick play_mode::map_gates -- --ignored --nocapture`
    //!
    //! Every gate here drives `PlayMode::enter` and `PlayMode::tick` — the same two calls the
    //! window makes — against a stand-in floor, so what it reads is what the player sees and not
    //! a re-implementation of it.

    use super::tests::Floor;
    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// Enter test mode on a flat floor at the origin, with the first-person rig mounted.
    fn start(
        native: &NativeDll,
        cache: CacheHandle,
        actors: &mut ActorScene,
        species: Species,
    ) -> Option<(PlayMode, Floor)> {
        let scene = Floor;
        let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = match PlayMode::enter(native, cache, &scene, &[], actors, species, cam) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("enter failed: {e}");
                return None;
            }
        };
        let rows: Vec<(String, u32, u32)> = crate::play_phys::PlayerPhysics::player_representations(native, cache)
            .into_iter()
            .map(|r| (String::new(), r.hands, r.body))
            .collect();
        let elite_of = |t: u32| crate::film_anim::is_elite(&native.tag_name(cache, t).unwrap_or_default());
        pm.mount_first_person(native, cache, &rows, &elite_of, actors);
        for n in pm.report().lines() {
            println!("   | {n}");
        }
        Some((pm, Floor))
    }

    fn hold(fwd: f32, walk: bool) -> MoveInput {
        MoveInput { fwd, right: 0.0, yaw: 0.0, pitch: 0.0, jump: false, crouch: false, walk, sprint: false }
    }

    /// #play-rate  **THE SNAP GATE.** The stride's phase must advance by EXACTLY the ground
    /// covered times the rate of the state the body is in — frame by frame, across every change
    /// of gait.
    ///
    /// The phase was `(TOTAL distance travelled x cycles_per_wu) mod 1`, and `cycles_per_wu`
    /// changes at every change of gait or stance — so with tens of world units of travel behind
    /// it, a walk-to-run switch moved the product by many whole cycles and the pose jumped to an
    /// unrelated point in the stride, snapping a foot mid-step. (Theater had exactly this defect
    /// and fixed it by integrating; see `film_anim::cycle_phase`. Test mode had re-introduced it.)
    ///
    /// # This gate was VACUOUS and the sabotage is what found it
    ///
    /// It first asserted only that the phase moved **no more than** `travelled x rate` could
    /// account for. Reverting the production code to the broken law **passed**: across the one
    /// walk-to-run transition the broken phase moved 0.040 cycles against a bound of 0.057,
    /// because `D x (rate_run - rate_walk) mod 1` at that particular `D` happened to land small.
    /// The bound was right and the fixture was lucky — a one-in-nine coin flip decided whether the
    /// gate worked.
    ///
    /// Two changes make it bite. **The assertion is now an equality, not a bound**: the phase step
    /// must EQUAL the ground that frame covered times the authored rate for that state, to 1e-3.
    /// Under the broken law that holds inside a constant-gait stretch and fails at a transition by
    /// `D x (rate_new - rate_old) mod 1`, which 1e-3 catches unless `D` lands on a one-in-a-
    /// thousand value. **And the script makes the transition four times** at four different
    /// accumulated distances, so no single unlucky `D` can hide it.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_change_of_gait_does_not_snap_the_stride() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode map_gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut actors = ActorScene::default();
        let Some((mut pm, scene)) = start(native, cache, &mut actors, Species::Spartan) else { return };
        let body = pm.body.expect("a body actor");
        let dt = 1.0 / 60.0;
        // SETTLE FIRST. The body is spawned above its own feet and dropped, so the first step
        // reports a LANDING -- and a landing owns the body for `cuts::LAND_TICKS` ticks, during
        // which the frame comes from the landing one-shot and not from the stride at all. The
        // first version of this gate measured that and reported 0.0385 cycles of "snap", which is
        // exactly `0.5 frames per tick / the 13-frame land_hard clip`: a number derivable from the
        // fixture, i.e. a fault in the ruler and not in the code.
        for _ in 0..120 {
            pm.tick(&hold(0.0, false), dt, native, cache, &scene, &[], &mut actors);
        }
        // Walk, run, walk, run -- FOUR crossings of the walk/run boundary, at four different
        // accumulated distances.
        // Per sample: (phase in cycles, the ground THIS frame covered, the authored rate for the
        // state it was in, was it stride-driven, the clip).
        let mut seen: Vec<(f32, f32, f32, bool, String)> = Vec::new();
        let mut last_d = pm.st.distance;
        for i in 0..480 {
            let walk = (i / 60) % 2 == 0;
            let input = hold(1.0, walk);
            pm.tick(&input, dt, native, cache, &scene, &[], &mut actors);
            let (frame, n, clip) = actors
                .get(body)
                .map(|a| (a.frame, a.frame_count().max(1.0), a.anim_name.clone()))
                .unwrap();
            // The state the frame was chosen from, re-derived through the SAME production call
            // `pose` uses -- and the rate that state implies for the clip that actually loaded.
            let s = play_ctl::body_state(&pm.st, &input, &pm.phys, pm.weapon_class.clone());
            // The rate IN FORCE: the blend plan's weighted stride when the graph's speed-tier
            // clips are being mixed (mid-acceleration, which is where every walk/run crossing
            // lives), else the state's single authored rate. Comparing the phase step with the
            // single rate during a blend was the gate's own error -- a blend legitimately
            // advances the phase by the mix's stride, not the run clip's.
            let rate = match pm.gait_stride {
                Some(stride) if stride > 1e-4 => Some(1.0 / stride),
                _ => pm
                    .phys
                    .gait_speed(s.stance, s.gait)
                    .and_then(|v| crate::film_anim::cycles_per_wu_at_speed(v, n)),
            };
            let stride_driven = s.air_ticks.is_none() && s.land_ticks.is_none() && rate.is_some();
            let travelled = pm.st.distance - last_d;
            last_d = pm.st.distance;
            seen.push((frame / n, travelled, rate.unwrap_or(0.0), stride_driven, clip));
        }
        // --- LIVENESS, first: this gate must be looking at something that happened ----------
        let clips: Vec<&str> = {
            let mut v: Vec<&str> = seen.iter().filter(|s| s.3).map(|s| s.4.as_str()).collect();
            v.dedup();
            v
        };
        println!("   stride-driven clips over the run: {clips:?}");
        assert!(
            clips.len() >= 4,
            "the gait changed {} time(s) inside the stride-driven samples ({clips:?}) -- this gate \
             is meant to watch four crossings of the walk/run boundary",
            clips.len().saturating_sub(1)
        );
        assert!(
            seen.iter().filter(|s| s.3).count() > 400,
            "only {} of {} samples were stride-driven, so this gate barely looked at a stride",
            seen.iter().filter(|s| s.3).count(),
            seen.len()
        );
        let total: f32 = seen.iter().filter(|s| s.3).map(|s| s.1 * s.2).sum();
        assert!(total > 2.0, "the stride only advanced {total:.2} cycles over 8 s of walking and running");
        // --- THE EQUALITY, frame by frame ---------------------------------------------------
        let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
        for i in 1..seen.len() {
            // Only where BOTH ends of the step are the stride's own phase.
            if !(seen[i].3 && seen[i - 1].3) {
                continue;
            }
            // The circular step the phase actually took...
            let mut got = (seen[i].0 - seen[i - 1].0).rem_euclid(1.0);
            if got > 0.5 {
                got -= 1.0;
            }
            // ...and the step it OWES: the ground this frame covered at this state's own rate.
            let want = seen[i].1 * seen[i].2;
            if (got - want).abs() > worst.0 {
                worst = ((got - want).abs(), i, got, want);
            }
        }
        assert!(worst.1 > 0, "the comparison loop never ran -- this gate asserted nothing");
        println!(
            "   worst disagreement: {:.5} cycles at sample {} (phase moved {:.5}, owed {:.5}), \
             clip '{}' -> '{}'",
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            seen[worst.1 - 1].4,
            seen[worst.1].4
        );
        assert!(
            worst.0 < 1e-3,
            "at sample {} the stride's phase moved {:.5} cycles while the body covered {:.5} wu, \
             which at this state's authored rate is {:.5} cycles -- a disagreement of {:.5}. That is \
             the `(total distance x rate) mod 1` defect: the phase is being recomputed from the \
             whole journey instead of advanced by this frame's ground. Clip '{}' -> '{}'",
            worst.1,
            worst.2,
            seen[worst.1].1,
            worst.3,
            worst.0,
            seen[worst.1 - 1].4,
            seen[worst.1].4
        );
    }

    /// #fp-attach  **THE GRIP GATE.** The first-person weapon must be parented to the HAND that
    /// holds it, by the animation graph's own hierarchy.
    ///
    /// Reported: *"the weapon isn't even in the first person hands, we had this figured out in
    /// the theater where we held the weapon properly."* It was figured out in Theater — and the
    /// function that knew how was inside Theater's driver, so a rig mounted by test mode had no
    /// parent at all and the gun was composed through the weapon model's own skeleton, where its
    /// root node has none. That draws it at the rig's origin, which is inside the player's head.
    ///
    /// **The assertion is geometric and takes no magic number**: of all the arms model's nodes,
    /// the one nearest the gun's root must be a HAND. Unparented it is not (it is whichever node
    /// sits at the rig origin), which is what makes this discriminate rather than describe.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_first_person_weapon_is_held_in_the_hand() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode map_gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let Some((pm, _scene)) = start(native, cache, &mut actors, species) else { return };
            assert!(
                pm.fp.len() >= 2,
                "{}: the first-person rig is {} model(s) -- there is no gun to put in a hand",
                species.label(),
                pm.fp.len()
            );
            let arms = pm.fp[0];
            for &gun in &pm.fp[1..] {
                // The grip the graph states.
                let grip = actors.fp_grip_of(arms, gun).unwrap_or_else(|| {
                    panic!("{}: the graph does not parent the first-person weapon to any node", species.label())
                });
                let hand = actors.get(arms).unwrap().def.model.node_names[grip.1].clone();
                assert!(
                    hand.contains("hand"),
                    "{}: the graph parents the gun to '{hand}', which is not a hand",
                    species.label()
                );
                // Is node `i` the grip node or something hanging off it? The gun's root lands at
                // the GRIP, and the nearest bone to it is whichever finger joint happens to be
                // closest -- measured, `r_pinky_tip` at 0.012 wu. Asserting the nearest node is
                // literally named "hand" therefore failed on a correct implementation: a number
                // (and a name) derivable from the fixture, so the ruler was wrong, not the code.
                // The structural question is the right one and the skeleton answers it.
                let in_hand = |actors: &ActorScene, i: usize| -> bool {
                    let a = actors.get(arms).unwrap();
                    let mut k = i as i32;
                    for _ in 0..64 {
                        if k < 0 {
                            return false;
                        }
                        if k as usize == grip.1 {
                            return true;
                        }
                        k = a.def.model.skel.nodes[k as usize].parent as i32;
                    }
                    false
                };
                // ...and it is INSTALLED, by the mount, not merely resolvable.
                assert_eq!(
                    actors.get(gun).unwrap().fp_attach,
                    Some(grip),
                    "{}: the grip resolves but the mount did not install it -- which is exactly \
                     the fault (Theater installed it, test mode did not)",
                    species.label()
                );
                // Now the geometry, through both routes. `nearest` names the arms node closest to
                // the gun's root once everything is posed.
                let nearest = |actors: &mut ActorScene| -> (String, f32, usize) {
                    for id in [arms, gun] {
                        let _ = actors.set_anim(native, cache, id, Some("first_person:idle"));
                        if let Some(a) = actors.get_mut(id) {
                            a.playing = false;
                            a.frame = 0.0;
                        }
                    }
                    actors.update_poses_headless(0.0);
                    let g = actors.get(gun).unwrap();
                    let root = g.def.model.skel.nodes.iter().position(|n| n.parent < 0).unwrap();
                    let at = (g.world() * g.pose.model[root]).w_axis.truncate();
                    let a = actors.get(arms).unwrap();
                    let mut best = (String::new(), f32::MAX, 0usize);
                    for (i, nm) in a.def.model.node_names.iter().enumerate() {
                        let p = (a.world() * a.pose.model[i]).w_axis.truncate();
                        let d = (p - at).length();
                        if d < best.1 {
                            best = (nm.clone(), d, i);
                        }
                    }
                    best
                };
                let with = nearest(&mut actors);
                // And without, which is what the user is looking at.
                if let Some(a) = actors.get_mut(gun) {
                    a.fp_attach = None;
                    a.world_override = None;
                }
                let without = nearest(&mut actors);
                println!(
                    "   {}: parented -> nearest arms node '{}' at {:.3} wu | NOT parented -> '{}' at {:.3} wu",
                    species.label(),
                    with.0,
                    with.1,
                    without.0,
                    without.1
                );
                assert!(
                    in_hand(&actors, with.2),
                    "{}: with the grip installed the gun's root is nearest '{}' ({:.3} wu), which is \
                     not the grip node '{hand}' nor anything hanging off it",
                    species.label(),
                    with.0,
                    with.1
                );
                assert!(
                    !in_hand(&actors, without.2),
                    "{}: the gun's root is inside the hand ('{}') even WITHOUT the grip installed, so \
                     this gate cannot tell the two apart and proves nothing",
                    species.label(),
                    without.0
                );
                // ...and the grip brings it CLOSER to the hand, by a factor, which is the half a
                // reader will want as a number.
                println!(
                    "   {}: the grip moved the gun's root from {:.3} wu of its nearest bone ('{}') to \
                     {:.3} wu ('{}')",
                    species.label(),
                    without.1,
                    without.0,
                    with.1,
                    with.0
                );
                // Put it back, since the scene outlives the measurement.
                if let Some(a) = actors.get_mut(gun) {
                    a.fp_attach = Some(grip);
                }
            }
        }
    }

    /// #fp-legs  **THE DUPLICATE-BIPED GATE.** In first person the legs come from the authored
    /// `@player representation` FIRST-PERSON BODY, and the third-person biped is parked.
    ///
    /// Reported: *"there is also a duplicate biped model in first person now"* — the arms rig and
    /// a whole third-person Spartan were both on screen, so the player saw two pairs of arms.
    /// The legs were the point and they stay; what goes is the second copy of everything else.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn first_person_draws_the_authored_legs_and_parks_the_body() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode map_gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let Some((mut pm, scene)) = start(native, cache, &mut actors, species) else { return };
            let body = pm.body.expect("a body actor");
            let legs = pm
                .fp_body
                .unwrap_or_else(|| panic!("{}: no first-person body mounted", species.label()));
            // The legs model must be LESS than the body: that is the whole of "not a duplicate".
            let (bm, lm) = (
                actors.get(body).unwrap().def.model.mesh_fmt.len(),
                actors.get(legs).unwrap().def.model.mesh_fmt.len(),
            );
            let lr = actors.get(legs).unwrap().def.model.region_names.clone();
            println!("   {}: body {bm} meshes, fp legs {lm} meshes, legs regions {lr:?}", species.label());
            assert!(
                lm < bm,
                "{}: the first-person body has {lm} meshes against the third-person body's {bm} -- \
                 it is not a reduced model and drawing it is drawing a second biped",
                species.label()
            );
            assert!(
                !lr.iter().any(|r| r == "head"),
                "{}: the first-person body carries a 'head' region {lr:?} -- the camera is inside it",
                species.label()
            );
            // Walk a little so there is a real locomotion clip and a real frame to compare.
            for _ in 0..60 {
                pm.tick(&hold(1.0, false), 1.0 / 60.0, native, cache, &scene, &[], &mut actors);
            }
            for (view, drawn, parked) in [(PlayView::FirstPerson, legs, body), (PlayView::ThirdPerson, body, legs)] {
                pm.view = view;
                pm.tick(&hold(1.0, false), 1.0 / 60.0, native, cache, &scene, &[], &mut actors);
                let d = actors.get(drawn).unwrap();
                let p = actors.get(parked).unwrap();
                println!(
                    "   {:?}: drawn #{} '{}' at ({:.2},{:.2},{:.2}) frame {:.2}/{:.0} | parked #{} at z {:.0}",
                    view, d.id, d.anim_name, d.pos.x, d.pos.y, d.pos.z, d.frame, d.frame_count(), p.id, p.pos.z
                );
                assert!(
                    (d.pos - pm.st.pos).length() < 1e-4,
                    "{:?}: the drawn model is at {:?}, not at the player's feet {:?}",
                    view,
                    d.pos,
                    pm.st.pos
                );
                assert_eq!(p.pos, PARK, "{view:?}: the other body is not parked, so both are on screen");
                assert!(!d.anim_name.is_empty(), "{view:?}: the drawn model has no clip at all");
            }
            // ...and in first person the two models agree about the stride, because they are
            // posed from ONE clip at ONE frame. A disagreement here is two writers.
            pm.view = PlayView::FirstPerson;
            pm.tick(&hold(1.0, false), 1.0 / 60.0, native, cache, &scene, &[], &mut actors);
            let b = actors.get(body).unwrap();
            let l = actors.get(legs).unwrap();
            assert_eq!(
                l.anim_name, b.anim_name,
                "the legs are playing '{}' while the body plays '{}'",
                l.anim_name, b.anim_name
            );
            assert!(
                (l.frame - b.frame).abs() < 1e-3,
                "the legs are on frame {:.3} and the body on {:.3} of the same clip",
                l.frame,
                b.frame
            );
            // #fp-legs  CHANGING WEAPON REBUILDS THE ARMS RIG AND MUST NOT ADD A SECOND PAIR OF
            // LEGS. `give_weapon` drops every `fp` actor and re-mounts on the new weapon's graph,
            // and `mount_first_person` is the same call that created the legs -- so without the
            // `fp_body.is_none()` guard every weapon change would leave another set of legs
            // standing at the player's feet. Asserted on the ACTOR COUNT, which is the thing that
            // would grow.
            let before = actors.actors.len();
            let legs_before = pm.fp_body;
            let rows: Vec<(String, u32, u32)> = crate::play_phys::PlayerPhysics::player_representations(native, cache)
                .into_iter()
                .map(|r| (String::new(), r.hands, r.body))
                .collect();
            let elite_of = |t: u32| crate::film_anim::is_elite(&native.tag_name(cache, t).unwrap_or_default());
            let weap = pm.weapon_tag;
            let said = pm.give_weapon(native, cache, Some(weap), &mut actors, &rows, &elite_of);
            println!("   {}: re-armed -> {said}", species.label());
            assert_eq!(
                actors.actors.len(),
                before,
                "{}: changing weapon changed the actor count from {before} to {} -- the                  first-person legs were mounted a second time",
                species.label(),
                actors.actors.len()
            );
            assert_eq!(pm.fp_body, legs_before, "{}: the legs actor was replaced", species.label());
        }
    }

    /// #fp-legs  **THE "NO TORSO IN FIRST PERSON" GATE.** The authored first-person body's own
    /// geometry must stop BELOW the eye it is seen from — for both species, from the tags.
    ///
    /// # This gate exists because I got the answer wrong from region names
    ///
    /// I reported that the Elite's `fp_body` is "the character minus the head" and that an Elite
    /// might therefore see its own torso, on the evidence that its regions are `armor_pieces` and
    /// `body` (the same names the full biped uses) and that it has 9 meshes against the biped's
    /// 10. **Both of those are facts about naming and packing, not about geometry**, and the user
    /// — who has thousands of hours in the game — said flatly that you never see an Elite's torso
    /// in MCC or in stock Reach. They were right.
    ///
    /// The measurement that settles it is the model's own `compression info`
    /// (`render_model` +0x64 `render geometry` +0x10), whose position bounds are what the
    /// compressed vertices are decompressed against — so they bound every vertex the model can
    /// draw. Read off `forge_halo`:
    ///
    /// ```text
    ///   model                     z extent (wu)     authored eye     headroom
    ///   spartans/fp_body/fp_body  -0.0015 .. 0.4621     0.600          0.138
    ///   spartans/spartans         -0.0027 .. 0.7641       --             --
    ///   elite/fp_body/fp_body     -0.0100 .. 0.4822     0.690          0.208
    ///   elite/elite               -0.0100 .. 0.8424       --             --
    /// ```
    ///
    /// **The Elite's first-person body ends 0.208 wu below the Elite's own eye**, and is 0.36 wu
    /// shorter than the Elite biped. There is no torso to hide: it is legs and hips, exactly like
    /// the Spartan's. `body` in that region list means "the base mesh of this model", not "the
    /// torso".
    ///
    /// So no mechanism is needed and none is invented — the region-exclusion hack this nearly
    /// became would have had nothing to exclude. What is asserted instead is the OBSERVABLE
    /// stated as geometry: **nothing the first-person body can draw reaches the viewpoint it is
    /// drawn from**, plus the two things that make that mean something — the model really is
    /// shorter than the biped, and the biped really is tall enough to reach the eye (so the
    /// comparison is not passing because every model is short).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_first_person_body_never_reaches_the_eye_it_is_seen_from() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode map_gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        /// `render_model` +0x64 `render geometry` +0x10 `compression info`: flags, then the three
        /// (min, max) position pairs, then the texture-coordinate bounds. 56-byte elements.
        fn z_extent(native: &NativeDll, cache: CacheHandle, mode_tag: u32) -> Option<(f32, f32)> {
            let h = native.tag_meta(cache, mode_tag, 0x64 + 0x10, 8)?;
            let c = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
            let ptr = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
            if c == 0 || c >= 64 {
                return None;
            }
            let b = native.tag_ptr_bytes(cache, ptr, c * 56)?;
            let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            // Every element's bounds, unioned: a model with several compression sets draws all of
            // them, so the widest is the one that matters.
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for k in 0..c as usize {
                lo = lo.min(f(k * 56 + 20));
                hi = hi.max(f(k * 56 + 24));
            }
            (hi > lo).then_some((lo, hi))
        }
        let bipeds = ActorScene::list_bipeds(native, cache);
        let mut checked = 0usize;
        for species in [Species::Spartan, Species::Elite] {
            let Ok((_, phys)) = player_physics_for(native, cache, &bipeds, species) else { continue };
            let Ok((unit, _, _, _)) = resolve_species(native, cache, &bipeds, species) else { continue };
            let rep = crate::play_phys::PlayerPhysics::player_representations(native, cache)
                .into_iter()
                .find(|r| r.unit == unit)
                .expect("the species resolved through the representation block, so its row exists");
            assert_ne!(rep.body, 0xFFFF_FFFF, "{}: no authored first person body", species.label());
            let legs = z_extent(native, cache, rep.body).expect("the fp body's compression info reads");
            let body = z_extent(native, cache, native.resolve_mode(cache, unit))
                .expect("the biped's compression info reads");
            // **THE STANDING EYE, and that is the right comparison rather than the lower one.**
            //
            // `compression info` bounds the model's vertices in MODEL space with the skeleton in
            // its BIND pose, which is standing. Comparing a standing bound against the CROUCHING
            // eye compares two different poses: the first version of this gate did, and reported
            // the Spartan's 0.4621 wu of standing geometry against a 0.4200 wu crouched eye -- a
            // 0.042 wu "failure" that is just the hips being 0.15 wu lower when the capsule drops
            // from 0.650 to 0.500. A number derivable from the fixture, so the ruler was wrong.
            //
            // What this therefore does NOT bound is the crouched case: nothing in the tag says
            // where a posed hip ends up, and the honest statement is that the standing pose is
            // what is measured here.
            let eye = phys.eye_standing;
            println!(
                "   {}: fp body z {:+.4}..{:+.4} | biped z {:+.4}..{:+.4} | eye standing {:.3} \
                 crouching {:.3} -> headroom {:.4} wu",
                species.label(),
                legs.0,
                legs.1,
                body.0,
                body.1,
                phys.eye_standing,
                phys.eye_crouching,
                eye - legs.1
            );
            // 1. THE OBSERVABLE: nothing the first-person body can draw reaches the viewpoint.
            assert!(
                legs.1 < eye,
                "{}: the authored first-person body's geometry reaches {:.4} wu while the eye it is \
                 seen from is at {:.4} wu -- the player would be looking at the inside of their own \
                 chest",
                species.label(),
                legs.1,
                eye
            );
            // 2. ...and it is genuinely a REDUCED model, not the biped under another name. (This
            //    is the assertion that would have caught my own wrong conclusion: a wrong tag or a
            //    wrong representation row lands on the biped, which is 0.36 wu taller.)
            assert!(
                body.1 - legs.1 > 0.2,
                "{}: the first-person body reaches {:.4} wu and the full biped {:.4} -- only \
                 {:.4} wu apart, so what was read is probably the biped itself",
                species.label(),
                legs.1,
                body.1,
                body.1 - legs.1
            );
            // 3. ...and the comparison is not passing because every model here is short: the
            //    BIPED must reach past the eye, which is what makes assertion 1 discriminating.
            assert!(
                body.1 > eye,
                "{}: even the full biped only reaches {:.4} wu against a {:.4} wu eye, so assertion \
                 1 cannot distinguish a legs model from a whole body",
                species.label(),
                body.1,
                eye
            );
            checked += 1;
        }
        assert_eq!(checked, 2, "both playable species must be checked, got {checked}");
    }

    /// #jump  **THE JUMP-IS-READ GATE.** The jump impulse must be the one authored on THAT
    /// biped, and the two playable species must get different ones.
    ///
    /// Reported: *"jump height is still not being read or respected it would seem."* This gate
    /// settles the "read" half and nothing else: the apex additionally depends on the fall
    /// constant, which is an open question (see the `reach-engine-physics-gravity` note — the
    /// previously fitted value was a clamp artefact) and **no constant is touched here.** What is
    /// asserted is that the impulse comes out of `bipd +0x4F0` per biped, reaches the controller,
    /// and leaves the ground at exactly that speed.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_jump_impulse_is_read_from_each_biped_and_reaches_the_body() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode map_gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut vels = Vec::new();
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let Some((mut pm, scene)) = start(native, cache, &mut actors, species) else { return };
            let dt = 1.0 / 60.0;
            // Settle on the floor first: a jump only exists from the ground.
            for _ in 0..60 {
                pm.tick(&hold(0.0, false), dt, native, cache, &scene, &[], &mut actors);
            }
            assert!(pm.st.grounded, "{}: never settled on the floor", species.label());
            let z0 = pm.st.pos.z;
            let mut jump = hold(0.0, false);
            jump.jump = true;
            pm.tick(&jump, dt, native, cache, &scene, &[], &mut actors);
            assert!(pm.last_step.jumped, "{}: the jump did not fire", species.label());
            // The vertical speed one frame in is the impulse less one frame of gravity: the
            // controller applies the impulse and THEN integrates, so this is a closed form and
            // not a tolerance band.
            let want = pm.phys.jump_velocity - pm.phys.gravity() * dt;
            println!(
                "   {}: authored jump velocity {:.4} wu/s -> vz {:.4} after one frame (want {:.4}), apex {:.3} wu",
                species.label(),
                pm.phys.jump_velocity,
                pm.st.vel.z,
                want,
                crate::reach_physics::jump_apex(pm.phys.jump_velocity, pm.phys.gravity())
            );
            assert!(
                (pm.st.vel.z - want).abs() < 1e-3,
                "{}: left the ground at {:.4} wu/s, not the authored {:.4}",
                species.label(),
                pm.st.vel.z,
                pm.phys.jump_velocity
            );
            // ...and it actually rose, measured on the body rather than on the velocity.
            let mut apex = pm.st.pos.z;
            for _ in 0..180 {
                pm.tick(&hold(0.0, false), dt, native, cache, &scene, &[], &mut actors);
                apex = apex.max(pm.st.pos.z);
            }
            // THE EXPECTATION IS THE INTEGRATOR'S OWN CLOSED FORM, not the continuous one.
            //
            // The controller is semi-implicit Euler and it applies a tick's gravity BEFORE that
            // tick's integration -- including on the jump frame itself -- so the rise is
            //
            //     sum over k of (v - (k+1) g h) h   for every step with v - (k+1) g h > 0
            //       = K v h - g h^2 K(K+1)/2,   K = floor(v / (g h))
            //
            // which is the continuous `v^2/2g` LESS `v h / 2`: a half-tick of velocity, 0.0221 wu
            // for an Elite at 60 Hz. The first version of this gate compared against the
            // continuous apex with a 0.02 band and the Elite missed it by 0.0221 -- a number
            // derivable from the fixture's own timestep, so the ruler was wrong. With the right
            // closed form the tolerance tightens to 1e-3, fifty times stricter, and it now also
            // proves the integrator loses exactly half a tick and not a whole one.
            let (v, g, h) = (pm.phys.jump_velocity, pm.phys.gravity(), dt);
            let k = (v / (g * h)).floor();
            let euler = k * v * h - g * h * h * k * (k + 1.0) / 2.0;
            let continuous = crate::reach_physics::jump_apex(v, g);
            println!(
                "   {}: measured apex {:.4} wu above the floor | semi-implicit Euler {euler:.4} | \
                 continuous v^2/2g {continuous:.4} (the integrator costs v*h/2 = {:.4}) | airtime {:.3} s",
                species.label(),
                apex - z0,
                v * h / 2.0,
                2.0 * v / g
            );
            assert!(
                (apex - z0 - euler).abs() < 1e-3,
                "{}: rose {:.4} wu, the integrator's own closed form says {euler:.4} -- the impulse \
                 is not being respected",
                species.label(),
                apex - z0
            );
            // And the deficit against the continuous apex is HALF A TICK of velocity and no more,
            // which is the "is gravity applied before or after the impulse" question answered as
            // a number instead of a guess.
            assert!(
                ((continuous - euler) - v * h / 2.0).abs() < 2e-3,
                "{}: the integrator loses {:.4} wu against the continuous apex, not the half tick \
                 ({:.4}) a single pre-integration gravity step costs",
                species.label(),
                continuous - euler,
                v * h / 2.0
            );
            // ...and came back down and STAYED down, which is the other half of "the jump feels
            // wrong": a landing that never settles.
            assert!(pm.st.grounded, "{}: still airborne 3 s after a jump", species.label());
            assert!((pm.st.pos.z - z0).abs() < 1e-3, "{}: landed {:.4} wu off the floor it left", species.label(), pm.st.pos.z - z0);
            vels.push((species, pm.phys.jump_velocity));
        }
        // THE READ IS PER BIPED. Reach authors 2.300 for the Spartan and 2.650 for the Elite, so
        // one shared constant -- or a read at the wrong offset returning the same bytes for both
        // -- cannot pass this.
        assert_eq!(vels.len(), 2);
        assert!(
            (vels[0].1 - vels[1].1).abs() > 0.05,
            "both species jump at {:.4} wu/s: the impulse is not being read per biped",
            vels[0].1
        );
    }
}

// =============================================================================================
// #play-rate  PROBES — print-only measurements off a real cache
// =============================================================================================

#[cfg(test)]
mod probes {
    //! `cargo test -p hms-app --profile quick play_mode::probes -- --ignored --nocapture`
    //!
    //! These MEASURE; they assert almost nothing. Every number the fixes in this round rest on
    //! was read here first.

    use super::*;
    use crate::film_anim::{BodyState, Gait, Stance};

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// Every (stance, gait) test mode can be in, with the clip name candidates and the authored
    /// speed the controller would cap that direction at.
    fn gait_table() -> Vec<(Stance, Gait, &'static str)> {
        vec![
            (Stance::Combat, Gait::WalkFront, "walk forward"),
            (Stance::Combat, Gait::RunFront, "run forward"),
            (Stance::Combat, Gait::BackSlow, "walk backward"),
            (Stance::Combat, Gait::Back, "run backward"),
            (Stance::Combat, Gait::Left, "strafe left"),
            (Stance::Combat, Gait::Right, "strafe right"),
            (Stance::Crouch, Gait::WalkFront, "crouch forward slow"),
            (Stance::Crouch, Gait::RunFront, "crouch forward"),
            (Stance::Crouch, Gait::BackSlow, "crouch backward slow"),
            (Stance::Crouch, Gait::Back, "crouch backward"),
            (Stance::Crouch, Gait::Left, "crouch left"),
            (Stance::Crouch, Gait::Right, "crouch right"),
            (Stance::Sprint, Gait::RunFront, "sprint"),
        ]
    }

    /// #play-rate  **THE SLIDING MEASUREMENT.** For every locomotion state: which clip the graph
    /// actually loads, how long it is, the stride each rate law implies, and -- the number that
    /// matters -- what frame rate the clip would be played at while the body travels at the
    /// AUTHORED speed for that gait. 30 fps is the authored rate; anything else is a foot
    /// sliding, by exactly that factor.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_locomotion_rate_against_the_authored_speed() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = ActorScene::list_bipeds(native, cache);
        for species in [Species::Spartan, Species::Elite] {
            let Ok((path, phys)) = player_physics_for(native, cache, &bipeds, species) else {
                eprintln!("  {}: no player row", species.label());
                continue;
            };
            println!("\n=== {} — {path}", species.label());
            println!("{}", phys.describe());
            let (tag, _, _, _) = resolve_species(native, cache, &bipeds, species).unwrap();
            let mut actors = ActorScene::default();
            let id = actors.add_object(native, cache, tag, Vec3::ZERO, 0.0).expect("biped actor");
            let class = "rifle";
            println!(
                "  {:<22} {:<40} {:>6} {:>8} {:>8} {:>9} {:>9} {:>7}",
                "state", "clip the graph loaded", "frames", "v_auth", "v_film", "stride_a", "stride_f", "fps_film"
            );
            for (stance, gait, label) in gait_table() {
                let mut st = BodyState::default();
                st.stance = stance;
                st.gait = gait;
                st.weapon_class = Some(class.to_string());
                let cands = crate::film_anim::anim_candidates(&st, class);
                let mut loaded = String::new();
                for c in &cands {
                    if actors.set_anim(native, cache, id, Some(c)).is_ok() {
                        loaded = c.clone();
                        break;
                    }
                }
                let n = actors.get(id).map(|a| a.frame_count()).unwrap_or(1.0);
                let v_auth = phys.gait_speed(stance, gait).unwrap_or(0.0);
                let v_film = crate::film_anim::gait_speed_and_frames(stance, gait).map(|(v, _)| v).unwrap_or(0.0);
                // stride = the world units one cycle of this clip covers under each law.
                let stride_a = v_auth * n / crate::film_anim::CLIP_FPS;
                let stride_f = v_film * n / crate::film_anim::CLIP_FPS;
                // ...and the frame rate the clip is played at while the body moves at v_auth:
                // cycles/s = v_auth / stride, frames/s = cycles/s * n.
                let fps_film = if stride_f > 1e-6 { v_auth / stride_f * n } else { 0.0 };
                println!(
                    "  {label:<22} {:<40} {n:>6.0} {v_auth:>8.3} {v_film:>8.3} {stride_a:>9.3} {stride_f:>9.3} {fps_film:>7.1}",
                    if loaded.is_empty() { "<THE GRAPH HAS NONE OF THEM>".to_string() } else { loaded }
                );
            }
        }
    }

    /// #play-jolt  **THE JOLT MEASUREMENT.** Drive a real `PlayMode` through a scripted walk and
    /// print, per frame, the largest distance any one of the body's bones moved since the frame
    /// before — beside the gait, the clip and the frame it was posed at.
    ///
    /// "A sudden jolt of the legs", "doesn't animate cleanly", "plays through then skips" are all
    /// one measurable thing: a frame on which the skeleton moves much further than its
    /// neighbours. This prints that number so the argument is had over it rather than over what a
    /// pose ought to look like.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_pose_continuity_over_a_scripted_walk() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let scene = crate::play_mode::tests::Floor;
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = match PlayMode::enter(native, cache, &scene, &[], &mut actors, Species::Spartan, cam) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("enter failed: {e}");
                return;
            }
        };
        let body = pm.body.expect("a body");
        let dt = 1.0 / 60.0;
        // The script, in (seconds, label, input) — every state the user named.
        let mut plan: Vec<(&str, MoveInput)> = Vec::new();
        let base = MoveInput { fwd: 0.0, right: 0.0, yaw: 0.0, pitch: 0.0, jump: false, crouch: false, walk: false, sprint: false };
        for _ in 0..420 {
            plan.push(("idle", base));
        }
        for _ in 0..90 {
            plan.push(("forward", MoveInput { fwd: 1.0, ..base }));
        }
        for _ in 0..90 {
            plan.push(("diagonal", MoveInput { fwd: 1.0, right: 1.0, ..base }));
        }
        for _ in 0..90 {
            plan.push(("strafe", MoveInput { right: 1.0, ..base }));
        }
        for _ in 0..90 {
            plan.push(("crouch fwd", MoveInput { fwd: 1.0, crouch: true, ..base }));
        }
        // ...and a turn in place: standing still with the aim swinging away.
        for i in 0..120 {
            plan.push(("turn", MoveInput { yaw: (i as f32) * 0.03, ..base }));
        }
        let mut prev: Option<Vec<Vec3>> = None;
        let mut rows: Vec<(usize, &str, String, String, f32, f32, f32)> = Vec::new();
        for (i, (label, input)) in plan.iter().enumerate() {
            pm.tick(input, dt, native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(dt);
            let a = actors.get(body).unwrap();
            let now: Vec<Vec3> = a.pose.model.iter().map(|m| m.w_axis.truncate()).collect();
            let jolt = prev
                .as_ref()
                .map(|p| {
                    p.iter()
                        .zip(now.iter())
                        .map(|(x, y)| (*x - *y).length())
                        .fold(0.0f32, f32::max)
                })
                .unwrap_or(0.0);
            let st = play_ctl::body_state(&pm.st, input, &pm.phys, pm.weapon_class.clone());
            rows.push((
                i,
                label,
                format!("{:?}/{:?}", st.stance, st.gait),
                a.anim_name.clone(),
                a.frame,
                jolt,
                pm.st.facing.to_degrees(),
            ));
            prev = Some(now);
        }
        // Per phase: the median jolt and the worst, so "worse than its neighbours" is a number.
        let mut by_label: std::collections::BTreeMap<&str, Vec<f32>> = Default::default();
        for r in &rows {
            by_label.entry(r.1).or_default().push(r.5);
        }
        println!("\n  phase        frames   median jolt   p90 jolt   WORST jolt (wu)");
        for (label, mut v) in by_label {
            let n = v.len();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  {label:<12} {n:>6}   {:>11.5}   {:>8.5}   {:>8.5}",
                v[n / 2],
                v[n * 9 / 10],
                v[n - 1]
            );
        }
        // The twenty worst frames in the whole run, with what changed on them.
        let mut worst = rows.clone();
        worst.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());
        println!("\n  the 20 largest single-frame bone moves:");
        for w in worst.iter().take(20) {
            let before = rows.get(w.0.saturating_sub(1)).map(|r| (r.2.clone(), r.3.clone(), r.4)).unwrap_or_default();
            println!(
                "   #{:<4} {:<10} {:<22} '{}' frame {:>6.2}  jolt {:.5} wu  facing {:>7.1}  (was {} '{}' frame {:.2})",
                w.0, w.1, w.2, w.3, w.4, w.5, w.6, before.0, before.1, before.2
            );
        }
        // And the GAIT TRACE for the diagonal phase, which is where a flicker would live.
        println!("\n  gait per frame through the diagonal phase:");
        let mut last = String::new();
        for r in rows.iter().filter(|r| r.1 == "forward" || r.1 == "diagonal" || r.1 == "strafe") {
            if r.2 != last {
                println!("   #{:<4} {:<10} -> {:<22} '{}'", r.0, r.1, r.2, r.3);
                last = r.2.clone();
            }
        }
    }

    /// #fp-legs  **THE FIRST-PERSON POSING CHAIN, measured.** Reported: *"sometimes the FP model
    /// is a bit wavy or something ... the FP model needs significant work."*
    ///
    /// "Wavy" is a symptom, and a predicate that measures a symptom instead of naming a cause
    /// keeps coming back. The cause candidates that produce a DEFORMING surface (rather than a
    /// mis-placed one) all live in the node chain, and all three are measured here rather than
    /// argued:
    ///
    /// 1. **The node ORDER.** The first-person body is posed by the BIPED's graph, not by the
    ///    weapon's first-person graph (measured: that graph reaches 0 of the Spartan `fp_body`'s
    ///    82 nodes, the biped's reaches all 82). A matching node COUNT is not a matching ORDER,
    ///    and applying a pose by index across two differently-ordered node lists warps a mesh in
    ///    exactly a wavy way. So this prints, per model, how many of the graph's nodes resolve
    ///    BY NAME, how many sit at the same index, and which route
    ///    [`crate::reach_actor::anim_node_map`] therefore takes -- `None` (positional) or
    ///    `Some` (remapped). The dangerous combination is a LOW name-resolution share with a
    ///    positional read, because that is a silent wrong binding.
    /// 2. **The per-mesh node map, including the identity-omission case.** The engine omits a
    ///    mesh's `per mesh node map` when it would be the identity, so an EMPTY map has two
    ///    meanings (identity, or the mesh is rigid) and the two take different paths. Printed per
    ///    mesh with the format and the rigid node, for both species' rig models and the
    ///    third-person biped, so neither case is assumed.
    /// 3. **The skeleton's own shape.** A node whose parent index is out of range, or a cycle,
    ///    chains a pose against garbage. Counted.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_first_person_posing_chain() {
        use crate::play_phys::{PLAYER_REPRESENTATION, PLAYER_REPRESENTATION_ELEM, REP_BODY, REP_HANDS};
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = ActorScene::list_bipeds(native, cache);
        let matg = native.tags_by_group(cache, "matg").into_iter().next().unwrap_or(0);
        let b = crate::reach_actor::tag_block(
            native, cache, matg, PLAYER_REPRESENTATION, PLAYER_REPRESENTATION_ELEM, 3,
        );
        let tagref = |e: &[u8], o: usize| -> u32 {
            if o + 16 > e.len() { return 0xFFFF_FFFF }
            let d = u32::from_le_bytes([e[o + 0x0C], e[o + 0x0D], e[o + 0x0E], e[o + 0x0F]]);
            if d == 0xFFFF_FFFF { 0xFFFF_FFFF } else { d & 0xFFFF }
        };
        let mut worst_positional_share = 1.0f32;
        for species in [Species::Spartan, Species::Elite] {
            let Ok((bipd, _, _, _)) = resolve_species(native, cache, &bipeds, species) else { continue };
            let Some(jmad) = crate::reach_actor::resolve_jmad(native, cache, bipd) else {
                println!("{}: the biped resolves NO animation graph", species.label());
                continue;
            };
            let Some(graph) = crate::reach_actor::AnimGraph::read(native, cache, jmad) else { continue };
            // The rig row for this species, picked the way `mount_first_person` picks it.
            let mut row: Option<(u32, u32)> = None;
            for i in 0..b.len() / PLAYER_REPRESENTATION_ELEM {
                let e = &b[i * PLAYER_REPRESENTATION_ELEM..(i + 1) * PLAYER_REPRESENTATION_ELEM];
                let (h, fb) = (tagref(e, REP_HANDS), tagref(e, REP_BODY));
                if h == 0xFFFF_FFFF { continue; }
                let elite = native.tag_name(cache, h).unwrap_or_default().to_lowercase().contains("elite");
                if elite == matches!(species, Species::Elite) {
                    row = Some((h, fb));
                    break;
                }
            }
            println!("\n  == {} == graph '{}' with {} skeleton node(s)",
                species.label(), native.tag_name(cache, jmad).unwrap_or_default(), graph.node_names.len());
            // EACH MODEL AGAINST THE GRAPH THAT ACTUALLY POSES IT, which is not the same graph
            // for all three — pairing them all with the biped's graph was the first version of
            // this probe and it went red on the HANDS for a reason that was my own fixture
            // error, not a defect: `mount_first_person` poses the hands from the WEAPON's
            // first-person graph (`weap +0x3B8 player interface -> first person animations`),
            // and only the legs and the third-person body from the biped's own.
            let mut models: Vec<(String, u32, crate::reach_actor::AnimGraph)> = Vec::new();
            models.push(("third-person biped".into(), native.resolve_mode(cache, bipd), graph.clone()));
            if let Some((hands, fp_body)) = row {
                if fp_body != 0xFFFF_FFFF {
                    models.push(("fp body (legs)".into(), fp_body, graph.clone()));
                }
                if hands != 0xFFFF_FFFF {
                    // A weapon that authors a first-person interface for this species. Walked
                    // rather than named, so this is not pinned to one gun.
                    let mut paired = 0usize;
                    for weap in native.tags_by_group(cache, "weap") {
                        let (fpm, fpg) = crate::reach_actor::first_person_weapon(
                            native, cache, weap, usize::from(matches!(species, Species::Elite)),
                        );
                        let Some(g) = fpg.and_then(|t| crate::reach_actor::AnimGraph::read(native, cache, t)) else {
                            continue;
                        };
                        let label = format!(
                            "fp hands + '{}'",
                            native.tag_name(cache, weap).unwrap_or_default().rsplit(['\\', '/']).next().unwrap_or("")
                        );
                        models.push((label, hands, g.clone()));
                        // ...and the GUN's own first-person model, which the same graph poses as
                        // a child of a hand node.
                        if let Some(fpm) = fpm {
                            models.push((
                                format!("fp gun model of '{}'", native.tag_name(cache, weap).unwrap_or_default().rsplit(['\\', '/']).next().unwrap_or("")),
                                fpm,
                                g,
                            ));
                        }
                        paired += 1;
                    }
                    if paired == 0 {
                        println!("   no weapon in this cache authors a first-person graph for this species");
                    }
                }
            }
            for (label, mode, graph) in models {
                let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) else {
                    println!("   {label:<20} mode {mode:#x} -> NOT READABLE as a render model");
                    continue;
                };
                // 1. the node order.
                let resolved = graph
                    .node_names
                    .iter()
                    .filter(|n| m.node_names.iter().any(|x| x == *n))
                    .count();
                let same_index = graph
                    .node_names
                    .iter()
                    .enumerate()
                    .filter(|(i, n)| m.node_names.get(*i) == Some(n))
                    .count();
                let map = crate::reach_actor::anim_node_map(&m, &graph);
                let route = match &map {
                    None => "POSITIONAL (identity)",
                    Some(_) => "REMAPPED by name",
                };
                let share = if graph.node_names.is_empty() { 0.0 } else { resolved as f32 / graph.node_names.len() as f32 };
                let model_share = if m.node_names.is_empty() { 0.0 } else { resolved as f32 / m.node_names.len() as f32 };
                // #fp-wavy  THE ROUTE-FLIP MARGIN. `anim_node_map` keeps the REMAPPED route
                // while EITHER share reaches 90%, and falls back to a POSITIONAL read when
                // neither does. The hands sit at 7 of 47 nodes in the same position, so a
                // positional read of them would deform the mesh -- which makes the distance of
                // the better share from 0.90 the margin this whole lane is safe by.
                let margin = share.max(model_share) - 0.9;
                if margin < 0.05 {
                    println!("     *** ROUTE-FLIP MARGIN {:.3} (graph share {:.3}, model share {:.3}) ***", margin, share, model_share);
                }
                println!(
                    "   {label:<20} '{}'\n     {} model node(s); {resolved} of {} graph node(s) resolve by NAME ({:.1}%), {same_index} at the SAME INDEX -> {route}",
                    native.tag_name(cache, mode).unwrap_or_default(),
                    m.node_names.len(), graph.node_names.len(), share * 100.0
                );
                if map.is_none() {
                    worst_positional_share = worst_positional_share.min(share);
                }
                // The first mismatch, by index, is the thing that would warp the mesh.
                if let Some(k) = (0..graph.node_names.len().min(m.node_names.len()))
                    .find(|&i| graph.node_names[i] != m.node_names[i])
                {
                    println!(
                        "     FIRST ORDER MISMATCH at index {k}: the graph says '{}', the model says '{}'",
                        graph.node_names[k], m.node_names[k]
                    );
                } else if graph.node_names.len() == m.node_names.len() {
                    println!("     the two node lists are IDENTICAL, name for name, in order");
                }
                // 3. the skeleton's own shape.
                let bad_parents = (0..m.skel.len())
                    .filter(|&i| {
                        let pa = m.skel.nodes[i].parent;
                        pa >= 0 && pa as usize >= m.skel.len()
                    })
                    .count();
                let self_parent = (0..m.skel.len()).filter(|&i| m.skel.nodes[i].parent == i as i16).count();
                println!("     skeleton: {} node(s), {bad_parents} parent index out of range, {self_parent} self-parented", m.skel.len());
                assert_eq!(bad_parents, 0, "{label}: {bad_parents} node(s) point at a parent outside the skeleton");
                assert_eq!(self_parent, 0, "{label}: {self_parent} node(s) are their own parent");
                // 2. the per-mesh node maps, both cases named.
                let mut identity = 0usize;
                let mut mapped = 0usize;
                let mut rigid = 0usize;
                let mut over = Vec::new();
                for i in 0..m.mesh_fmt.len() {
                    let skinned = crate::reach_skel::is_skinned_format(m.mesh_fmt[i]);
                    let nm = m.node_maps.get(i).map(|v| v.len()).unwrap_or(0);
                    if !skinned {
                        rigid += 1;
                    } else if nm == 0 {
                        identity += 1;
                    } else {
                        mapped += 1;
                    }
                    // A map that names a node the model does not have is the wrong-binding case.
                    if let Some(v) = m.node_maps.get(i) {
                        if v.iter().any(|&s| s as usize >= m.skel.len() && s != 0xFF) {
                            over.push(i);
                        }
                    }
                }
                println!(
                    "     meshes: {} total -- {mapped} with a per-mesh node map, {identity} skinned with the map OMITTED (identity), {rigid} rigid; {} map(s) naming a node outside the skeleton",
                    m.mesh_fmt.len(), over.len()
                );
                assert!(
                    over.is_empty(),
                    "{label}: mesh(es) {over:?} have a per-mesh node map pointing outside the skeleton -- \
                     that binds vertices to garbage and is exactly what a wavy surface looks like"
                );
            }
        }
        // THE ASSERTION THAT NAMES A CAUSE. A positional read is only safe when the two node
        // lists really are the same list: if a model takes the positional route while fewer than
        // 90% of the graph's nodes even EXIST in it by name, the pose is being applied across two
        // different orderings and the mesh will deform.
        assert!(
            worst_positional_share >= 0.9,
            "a model is posed POSITIONALLY while only {:.1}% of the graph's nodes resolve in it by \
             name -- the pose is being applied across two different node orderings",
            worst_positional_share * 100.0
        );
    }

    /// #gait-blend  **THE CALL-SITE GATE.** Does the PRODUCTION PATH actually blend?
    ///
    /// This exists because the closure batteries in `reach_actor::gait_blend` and
    /// `film_anim::tests` prove the ASSERTIONS discriminate and say nothing at all about whether
    /// anything invokes them. "The capability exists and nothing invokes it" has bitten this
    /// project five times, and the locomotion mix is exactly the shape that lands that way: a
    /// planner, a table and a pose lane, any one of which could be unreachable.
    ///
    /// So this drives a REAL [`PlayMode`] on `forge_halo` with the controller's own inputs,
    /// through forward -> diagonal -> strafe, and asserts on what the ACTOR ends up holding --
    /// `base_blend`, the posed clip names, and the per-frame bone motion, which is the pose that
    /// would be on screen. Nothing here reaches into the planner.
    ///
    /// Both species, because the Elite's graph authors a different gait set (7 gaits with the
    /// intermediate tiers filled, against the Spartan's 8 with two of them empty) and a rule
    /// that works on one is not thereby proven on the other.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_gait_blend_at_the_call_site() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let scene = crate::play_mode::tests::Floor;
        let dt = 1.0 / 60.0;
        let base = MoveInput { fwd: 0.0, right: 0.0, yaw: 0.0, pitch: 0.0, jump: false, crouch: false, walk: false, sprint: false };
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
            let mut pm = match PlayMode::enter(native, cache, &scene, &[], &mut actors, species, cam) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}: enter failed: {e}", species.label());
                    continue;
                }
            };
            // #fp-legs  MOUNT THE FIRST-PERSON RIG, so assertion 4 is not vacuous. Without
            // this the legs actor never exists and "0 of 0 frames matched" would pass while
            // proving nothing -- which is exactly the shape of gate this project keeps finding.
            // The `@player representation` rows are read here rather than taken from a
            // `SceneController` (the probe's scene is a bare floor) using the offsets
            // `play_phys` already carries, and a Reach tag reference is its datum at +0x0C.
            {
                use crate::play_phys::{PLAYER_REPRESENTATION, PLAYER_REPRESENTATION_ELEM, REP_BODY, REP_HANDS};
                let b = crate::reach_actor::tag_block(
                    native,
                    cache,
                    native.tags_by_group(cache, "matg").into_iter().next().unwrap_or(0),
                    PLAYER_REPRESENTATION,
                    PLAYER_REPRESENTATION_ELEM,
                    3,
                );
                let tagref = |e: &[u8], o: usize| -> u32 {
                    if o + 16 > e.len() { return 0xFFFF_FFFF }
                    let d = u32::from_le_bytes([e[o + 0x0C], e[o + 0x0D], e[o + 0x0E], e[o + 0x0F]]);
                    if d == 0xFFFF_FFFF { 0xFFFF_FFFF } else { d & 0xFFFF }
                };
                let rows: Vec<(String, u32, u32)> = (0..b.len() / PLAYER_REPRESENTATION_ELEM)
                    .map(|i| {
                        let e = &b[i * PLAYER_REPRESENTATION_ELEM..(i + 1) * PLAYER_REPRESENTATION_ELEM];
                        (String::new(), tagref(e, REP_HANDS), tagref(e, REP_BODY))
                    })
                    .collect();
                let elite_of = |t: u32| {
                    native.tag_name(cache, t).unwrap_or_default().to_lowercase().contains("elite")
                };
                pm.mount_first_person(native, cache, &rows, &elite_of, &mut actors);
                pm.view = PlayView::FirstPerson;
            }
            let body = pm.body.expect("a body");
            // Run straight, then diagonally, then sideways. The diagonal is the user's case.
            let mut plan: Vec<(&str, MoveInput)> = Vec::new();
            for _ in 0..90 { plan.push(("forward", MoveInput { fwd: 1.0, ..base })); }
            for _ in 0..90 { plan.push(("diagonal", MoveInput { fwd: 1.0, right: 1.0, ..base })); }
            for _ in 0..90 { plan.push(("strafe", MoveInput { right: 1.0, ..base })); }
            #[derive(Default)]
            struct Phase { frames: usize, max_clips: usize, both_dirs: usize, worst_jolt: f32, caption: String }
            let mut phases: std::collections::BTreeMap<&str, Phase> = std::collections::BTreeMap::new();
            let mut prev: Option<Vec<Vec3>> = None;
            let mut legs_matched = 0usize;
            let mut legs_seen = 0usize;
            for (label, input) in &plan {
                pm.tick(input, dt, native, cache, &scene, &[], &mut actors);
                actors.update_poses_headless(dt);
                let a = actors.get(body).expect("the body survives");
                // Every clip the pose is built from, by NAME: the primary plus the extra lanes.
                let g = a.def.graph.clone();
                let mut names: Vec<String> = vec![a.anim_name.clone()];
                for &(anim, _, _) in &a.base_blend {
                    if let Some(gr) = g.as_ref() {
                        if let Some(e) = gr.anims.get(anim) {
                            names.push(e.name.clone());
                        }
                    }
                }
                let front = names.iter().any(|n| n.contains("front"));
                let side = names.iter().any(|n| n.contains("left") || n.contains("right"));
                let now: Vec<Vec3> = a.pose.model.iter().map(|m| m.w_axis.truncate()).collect();
                let jolt = prev
                    .as_ref()
                    .map(|p| p.iter().zip(now.iter()).map(|(x, y)| (*x - *y).length()).fold(0.0f32, f32::max))
                    .unwrap_or(0.0);
                prev = Some(now);
                // The first-person LEGS must be holding the same mix, by graph index.
                if let Some(l) = pm.fp_body.and_then(|l| actors.get(l)) {
                    legs_seen += 1;
                    if l.base_blend.len() == a.base_blend.len()
                        && l.base_blend.iter().zip(a.base_blend.iter()).all(|(x, y)| x.0 == y.0 && (x.2 - y.2).abs() < 1e-6)
                    {
                        legs_matched += 1;
                    }
                }
                let e = phases.entry(label).or_default();
                e.frames += 1;
                e.max_clips = e.max_clips.max(names.len());
                if front && side { e.both_dirs += 1; }
                // The first 30 frames of a phase are the CONTROLLER accelerating into it, not
                // the mix: measured at 0.20 wu on the first frames of the straight run, which
                // is the body starting to move at all. The bound below is derived from the
                // straight-run phase, so letting a transient into it would make the bound loose
                // enough to pass anything.
                if e.frames > 30 { e.worst_jolt = e.worst_jolt.max(jolt); }
                e.caption = pm.gait_caption.clone();
            }
            println!("\n  == {} ==", species.label());
            for (label, p) in &phases {
                println!(
                    "   {label:<9} {:>3} frame(s) | up to {} clip(s) posed | {} frame(s) mixing BOTH directions | worst bone step {:.4} wu | last mix: {}",
                    p.frames, p.max_clips, p.both_dirs, p.worst_jolt, p.caption
                );
            }
            println!("   first-person legs carried the same mix on {legs_matched} of {legs_seen} frame(s)");
            // 1. THE DIAGONAL MIXES BOTH DIRECTIONS. Before this round the diagonal held one
            //    clip for all 90 frames, so this is the assertion that was red.
            let d = phases.get("diagonal").expect("the diagonal phase ran");
            assert!(
                d.both_dirs * 10 >= d.frames * 8,
                "{}: the diagonal mixed both directions on only {} of {} frames",
                species.label(), d.both_dirs, d.frames
            );
            // 2. ...and it is a real MIX, not a swap: more than one clip is posed at once.
            assert!(
                d.max_clips >= 3,
                "{}: the diagonal posed at most {} clip(s); two bracketing directions x a \
                 bracketing tier pair is up to four",
                species.label(), d.max_clips
            );
            // 3. THE POSE IS SMOOTH. The reference is the straight-run phase, where nothing
            //    about the mix is changing: measured there at 0.03-0.04 wu per frame on a
            //    walking Spartan, and the old linear cross-fade's first frame moved the busiest
            //    bone 0.255 wu. Three times the straight-run figure is the bound, and a clip
            //    appearing at full weight reads far above it.
            let f = phases.get("forward").expect("the forward phase ran");
            let bound = (f.worst_jolt * 3.0).max(0.12);
            assert!(
                d.worst_jolt <= bound,
                "{}: the diagonal's worst bone step is {:.4} wu against {:.4} wu while running \
                 straight (bound {bound:.4})",
                species.label(), d.worst_jolt, f.worst_jolt
            );
            // 4. The first-person legs hold the body's own mix. Drawn from the same graph by
            //    index, so a mismatch means a player's feet disagree with their own body.
            assert!(
                legs_seen > 0,
                "{}: the first-person legs actor was never mounted, so assertion 4 would be \
                 VACUOUS -- the probe must mount the rig, not skip it",
                species.label()
            );
            assert!(
                legs_matched * 10 >= legs_seen * 9,
                "{}: the first-person legs carried the body's mix on only {legs_matched} of \
                 {legs_seen} frame(s)",
                species.label()
            );
        }
    }

    /// #crouch-rate  **THE AUTHORED CROUCH RATE, END TO END.**
    ///
    /// Two halves, and the first is what makes the second trustworthy:
    ///
    /// 1. the field at `bipd +0x534` really is a TIME IN SECONDS, proven by the two values the
    ///    engine derives from it at tag post-process and leaves in the cache -- `+0x580
    ///    runtime crouch transition velocity` must equal `1 / time` and `+0x584
    ///    runtime_camera_height_velocity` must equal `(standing eye - crouching eye) / time`;
    /// 2. the production path uses it: after a scripted crouch the ACTOR's `blend_span` -- the
    ///    span its cross-fade was actually armed with -- must be that time and not the generic
    ///    0.18 s, which is what "the speed that he goes down ... is a bit fast" was.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_authored_crouch_rate() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let scene = crate::play_mode::tests::Floor;
        let dt = 1.0 / 60.0;
        let base = MoveInput { fwd: 0.0, right: 0.0, yaw: 0.0, pitch: 0.0, jump: false, crouch: false, walk: false, sprint: false };
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
            let mut pm = match PlayMode::enter(native, cache, &scene, &[], &mut actors, species, cam) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}: enter failed: {e}", species.label());
                    continue;
                }
            };
            let p = pm.phys;
            println!(
                "\n  == {} == crouch transition {:.4} s | +0x580 {:.4} | +0x584 {:.4} | eye {:.3} -> {:.3}",
                species.label(), p.crouch_transition, p.runtime_crouch_velocity,
                p.runtime_camera_height_velocity, p.eye_standing, p.eye_crouching
            );
            // 1a. the runtime crouch velocity IS one over the time.
            let want = 1.0 / p.crouch_transition;
            assert!(
                (p.runtime_crouch_velocity - want).abs() < 1e-3,
                "{}: +0x580 reads {:.6} but 1/{:.4} = {want:.6}; +0x534 is not a time in seconds",
                species.label(), p.runtime_crouch_velocity, p.crouch_transition
            );
            // 1b. ...and the camera height velocity is the eye's own SPAN over that time, which
            //     is a different number again, so one constant cannot satisfy both.
            let want = (p.eye_standing - p.eye_crouching) / p.crouch_transition;
            assert!(
                (p.runtime_camera_height_velocity - want).abs() < 1e-3,
                "{}: +0x584 reads {:.6} but ({:.3} - {:.3})/{:.4} = {want:.6}",
                species.label(), p.runtime_camera_height_velocity,
                p.eye_standing, p.eye_crouching, p.crouch_transition
            );
            // 2. THE CALL SITE. Stand for a moment, then crouch, and read the span the actor's
            //    cross-fade was armed with on the frame the stance changed.
            let body = pm.body.expect("a body");
            for _ in 0..30 {
                pm.tick(&base, dt, native, cache, &scene, &[], &mut actors);
            }
            let before = actors.get(body).map(|a| a.blend_span).unwrap_or(0.0);
            pm.tick(&MoveInput { crouch: true, ..base }, dt, native, cache, &scene, &[], &mut actors);
            let armed = actors.get(body).map(|a| a.blend_span).unwrap_or(0.0);
            println!("   blend span before the crouch {before:.3} s, on the crouch frame {armed:.3} s");
            assert!(
                (armed - p.crouch_transition).abs() < 1e-4,
                "{}: the crouch cross-fade was armed for {armed:.4} s, not the authored {:.4} s \
                 (the generic span is {:.4})",
                species.label(), p.crouch_transition, crate::reach_actor::BLEND_SECONDS
            );
            // ...and the authored time must actually DIFFER from the generic one, or this gate
            // could be satisfied by doing nothing at all.
            assert!(
                (p.crouch_transition - crate::reach_actor::BLEND_SECONDS).abs() > 1e-3,
                "{}: the authored crouch time equals the generic blend, so this gate proves nothing",
                species.label()
            );
            // 3. The EYE and the CAPSULE move together and complete in that same time, which is
            //    the other half of the user's report. Driven from one interpolant, so this is a
            //    check that one writer is in force rather than two.
            let h0 = (pm.st.eye(&p).z - pm.st.pos.z, pm.st.height(&p));
            let steps = (p.crouch_transition / dt).ceil() as usize + 2;
            for _ in 0..steps {
                pm.tick(&MoveInput { crouch: true, ..base }, dt, native, cache, &scene, &[], &mut actors);
            }
            let h1 = (pm.st.eye(&p).z - pm.st.pos.z, pm.st.height(&p));
            println!("   eye {:.4} -> {:.4} wu, capsule {:.4} -> {:.4} wu over {:.2} s", h0.0, h1.0, h0.1, h1.1, p.crouch_transition);
            assert!(
                (h1.0 - p.eye_crouching).abs() < 2e-3,
                "{}: the eye reached {:.4} wu, not the authored crouching {:.4}",
                species.label(), h1.0, p.eye_crouching
            );
            assert!(
                (h1.1 - p.height_crouching).abs() < 2e-3,
                "{}: the capsule reached {:.4} wu, not the authored crouching {:.4}",
                species.label(), h1.1, p.height_crouching
            );
        }
    }

    /// #play-jolt  **THE IDLE LOOP MEASUREMENT.** Does the clip a standing body plays actually
    /// LOOP, and does its root node move?
    ///
    /// "The idle animation seems to play through then skip/shift the biped" is the signature of a
    /// clip whose last frame is not its first. A pure cycle has frame 0 identical to the last
    /// frame; anything else pops once per cycle at the wrap. The root node is printed separately
    /// because a clip-driven root translation and the controller both writing the body's place is
    /// two writers on one field.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_whether_the_clock_driven_clips_loop() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = ActorScene::list_bipeds(native, cache);
        for species in [Species::Spartan, Species::Elite] {
            let Ok((tag, _, _, _)) = resolve_species(native, cache, &bipeds, species) else { continue };
            let mut actors = ActorScene::default();
            let id = actors.add_object(native, cache, tag, Vec3::ZERO, 0.0).expect("biped");
            println!("\n=== {}", species.label());
            for name in [
                "combat:rifle:idle",
                "combat:rifle:turn_left",
                "combat:rifle:turn_right",
                "crouch:rifle:idle",
                "combat:rifle:move_front_fast",
                "combat:rifle:move_front",
            ] {
                if actors.set_anim(native, cache, id, Some(name)).is_err() {
                    println!("  {name:<30} THIS GRAPH HAS NO SUCH CLIP");
                    continue;
                }
                // **SETTLE THE CROSS-FADE, or this measures the previous clip.** `set_anim` arms
                // `prev_anim` + `blend_left`, and the blend decays on WALL TIME inside
                // `Actor::advance` -- so posing at `dt = 0` leaves the weight fully on the
                // OUTGOING clip, whose frame this loop never moves. The first run of this probe
                // reported a per-frame bone move of exactly 0.00000 wu for every clip after the
                // first, which is impossible for a walk cycle and was the tell.
                actors.settle_blends();
                let n = actors.get(id).unwrap().frame_count();
                // Pose at each frame and keep every node's model-space origin.
                let mut frames: Vec<Vec<Vec3>> = Vec::new();
                for f in 0..n as usize {
                    if let Some(a) = actors.get_mut(id) {
                        a.playing = false;
                        a.frame = f as f32;
                    }
                    actors.update_poses_headless(0.0);
                    frames.push(actors.get(id).unwrap().pose.model.iter().map(|m| m.w_axis.truncate()).collect());
                }
                let span = |a: &Vec<Vec3>, b: &Vec<Vec3>| {
                    a.iter().zip(b.iter()).map(|(x, y)| (*x - *y).length()).fold(0.0f32, f32::max)
                };
                // The ordinary step between neighbouring frames, and the WRAP step.
                let mut steps: Vec<f32> = (1..frames.len()).map(|i| span(&frames[i - 1], &frames[i])).collect();
                let wrap = span(&frames[frames.len() - 1], &frames[0]);
                steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = steps[steps.len() / 2];
                let root = actors
                    .get(id)
                    .unwrap()
                    .def
                    .model
                    .skel
                    .nodes
                    .iter()
                    .position(|nd| nd.parent < 0)
                    .unwrap_or(0);
                let root_travel = (frames[frames.len() - 1][root] - frames[0][root]).length();
                let root_max = frames.iter().map(|f| (f[root] - frames[0][root]).length()).fold(0.0f32, f32::max);
                // The ENGINE'S OWN loop point, `c_model_animation` +0x08 `loop frame index`, and
                // the clip's type — because "does this clip loop" is a question the tag may
                // already answer, and reading it beats inferring it from the geometry.
                let (loop_frame, ty) = actors
                    .get(id)
                    .and_then(|a| a.def.graph.as_ref().zip(a.anim))
                    .map(|(g, i)| (g.anims[i].loop_frame, format!("{:?}", g.anims[i].anim_type)))
                    .unwrap_or((-1, String::new()));
                // **THE DISCRIMINATOR is the wrap against the 90th percentile of ordinary steps,
                // not against the median.** A locomotion cycle is in motion on every frame, so
                // its median step is large and a clean wrap looks like 1.9x it; an idle is still
                // for 155 frames out of 156, so its median is ~0 and ANY wrap looks like 180x.
                // The p90 is the size of a real frame of motion in both cases.
                let p90 = steps[steps.len() * 9 / 10];
                println!(
                    "  {name:<30} {n:>4.0} frames  loop_frame {loop_frame:>4}  {ty:<12} | per-frame bone move: \
                     median {median:.5} p90 {p90:.5} max {:.5} wu | WRAP last->first {wrap:.5} wu \
                     = {:>6.1}x p90 | root travel {root_travel:.5} (max excursion {root_max:.5})",
                    steps[steps.len() - 1],
                    if p90 > 1e-6 { wrap / p90 } else { 0.0 }
                );
            }
        }
    }

    /// #fp-attach  **THE FIRST-PERSON GRIP MEASUREMENT.** Mount the rig exactly as test mode
    /// does and print where the gun lands relative to the hand holding it, with and without the
    /// graph's own parenting.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_first_person_grip() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let reps = crate::play_phys::PlayerPhysics::player_representations(native, cache);
        for r in &reps {
            println!(
                "rep row {}: hands {:#x} '{}' | body {:#x} '{}' | unit {:#x} '{}' (info row {})",
                r.row,
                r.hands,
                native.tag_name(cache, r.hands).unwrap_or_default(),
                r.body,
                native.tag_name(cache, r.body).unwrap_or_default(),
                r.unit,
                native.tag_name(cache, r.unit).unwrap_or_default(),
                r.info_row
            );
        }
        let weapons = ActorScene::list_weapons(native, cache);
        let Some((weap, wpath)) = weapons
            .iter()
            .find(|(_, p)| p.to_lowercase().contains("assault_rifle"))
            .cloned()
        else {
            eprintln!("no assault rifle in forge_halo");
            return;
        };
        for (species_row, rep) in reps.iter().enumerate().take(2) {
            let (fp_model, fp_graph) = crate::reach_actor::first_person_weapon(native, cache, weap, species_row);
            println!(
                "\n=== species row {species_row}: {wpath}  fp model {:?} '{}'  fp graph {:?} '{}'",
                fp_model,
                fp_model.and_then(|t| native.tag_name(cache, t)).unwrap_or_default(),
                fp_graph,
                fp_graph.and_then(|t| native.tag_name(cache, t)).unwrap_or_default()
            );
            let (Some(fp_model), Some(_)) = (fp_model, fp_graph) else { continue };
            let mut actors = ActorScene::default();
            let arms = actors
                .add_render_model(native, cache, rep.hands, fp_graph, Vec3::ZERO, 0.0)
                .expect("arms");
            let gun = actors
                .add_render_model(native, cache, fp_model, fp_graph, Vec3::ZERO, 0.0)
                .expect("gun");
            for id in [arms, gun] {
                let _ = actors.set_anim(native, cache, id, Some("first_person:idle"));
                if let Some(a) = actors.get_mut(id) {
                    a.playing = false;
                    a.frame = 0.0;
                    a.model_yaw = crate::reach_actor::FP_ARMS_MODEL_YAW;
                }
            }
            // The graph's own parenting, through the shared resolver.
            let grip = actors.fp_grip_of(arms, gun);
            let hand_name = grip
                .and_then(|(_, n)| actors.get(arms).and_then(|a| a.def.model.node_names.get(n).cloned()))
                .unwrap_or_default();
            println!("  fp_grip_of(arms, gun) = {grip:?}  -> arms node '{hand_name}'");
            for (label, attach) in [("NOT parented (what test mode did)", None), ("parented (Theater)", grip)] {
                if let Some(a) = actors.get_mut(gun) {
                    a.fp_attach = attach;
                    a.world_override = None;
                }
                actors.update_poses_headless(0.0);
                let gun_root = actors
                    .get(gun)
                    .map(|a| {
                        let root = a.def.model.skel.nodes.iter().position(|n| n.parent < 0).unwrap_or(0);
                        (a.world() * a.pose.model[root]).w_axis.truncate()
                    })
                    .unwrap();
                let hand = grip
                    .and_then(|(_, n)| actors.get(arms).map(|a| (a.world() * a.pose.model[n]).w_axis.truncate()))
                    .unwrap_or(Vec3::ZERO);
                println!(
                    "    {label:<34} gun root at ({:+.3},{:+.3},{:+.3})  hand at ({:+.3},{:+.3},{:+.3})  |gun-hand| = {:.3} wu",
                    gun_root.x, gun_root.y, gun_root.z, hand.x, hand.y, hand.z,
                    (gun_root - hand).length()
                );
            }
        }
    }

    /// #fp-body  **THE DUPLICATE-BIPED MEASUREMENT.** What `@player representation` offers for
    /// the legs you see in first person, and whether the weapon's first-person graph can pose it.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_what_can_draw_first_person_legs() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let reps = crate::play_phys::PlayerPhysics::player_representations(native, cache);
        let weapons = ActorScene::list_weapons(native, cache);
        let Some((weap, _)) = weapons.iter().find(|(_, p)| p.to_lowercase().contains("assault_rifle")).cloned() else {
            return;
        };
        for (row, rep) in reps.iter().enumerate().take(2) {
            let (_, fp_graph) = crate::reach_actor::first_person_weapon(native, cache, weap, row);
            println!("\n=== rep row {row}");
            for (what, tag) in [("hands", rep.hands), ("fp body", rep.body), ("third person unit", rep.unit)] {
                let mode = if what == "third person unit" { native.resolve_mode(cache, tag) } else { tag };
                let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) else {
                    println!("  {what:<18} {tag:#x} -> mode {mode:#x}: NOT a readable render model");
                    continue;
                };
                let graph = fp_graph.and_then(|j| crate::reach_actor::AnimGraph::read(native, cache, j));
                let matched = graph
                    .as_ref()
                    .map(|g| g.node_names.iter().filter(|gn| m.node_names.iter().any(|n| n == *gn)).count());
                println!(
                    "  {what:<18} {tag:#x} -> mode {mode:#x} '{}': {} nodes, {} meshes, regions {:?}",
                    native.tag_name(cache, mode).unwrap_or_default(),
                    m.node_names.len(),
                    m.mesh_fmt.len(),
                    m.region_names
                );
                if let (Some(k), Some(g)) = (matched, graph.as_ref()) {
                    println!(
                        "      the WEAPON's first-person graph ({} nodes) reaches {k} of this model's {} nodes ({:.0}%)",
                        g.node_names.len(),
                        m.node_names.len(),
                        100.0 * k as f32 / m.node_names.len().max(1) as f32
                    );
                }
                // ...and the BIPED's OWN graph, which is the other candidate for posing legs.
                if let Some(bg) = crate::reach_actor::resolve_jmad(native, cache, rep.unit)
                    .and_then(|j| crate::reach_actor::AnimGraph::read(native, cache, j))
                {
                    let k = bg.node_names.iter().filter(|gn| m.node_names.iter().any(|n| n == *gn)).count();
                    println!(
                        "      the BIPED's own graph ({} nodes, {} clips) reaches {k} of this model's {} nodes ({:.0}%)",
                        bg.node_names.len(),
                        bg.anims.len(),
                        m.node_names.len(),
                        100.0 * k as f32 / m.node_names.len().max(1) as f32
                    );
                }
            }
        }
    }

    /// #fp-legs  **WHAT IS ACTUALLY IN THE FIRST-PERSON BODY.** Per mesh, the skeleton nodes its
    /// vertices are skinned to — which says what part of the body each mesh IS, without needing
    /// the geometry.
    ///
    /// The user's report is that an Elite never sees its own torso in MCC or in stock Reach, and
    /// HMS draws one. Region NAMES cannot settle that (`armor_pieces` and `body` are the same
    /// names the full biped uses — an artist's convention, not a description of the geometry), and
    /// the mesh COUNT cannot either. What a mesh is skinned to can: a mesh whose vertices only
    /// address `pelvis`, `thigh`, `calf` and `foot` is legs, and one that addresses `spine`,
    /// `clavicle` or `head` is not.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_what_the_first_person_body_is_made_of() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        for rep in crate::play_phys::PlayerPhysics::player_representations(native, cache) {
            if rep.body == 0xFFFF_FFFF {
                continue;
            }
            for (what, tag) in [("FIRST-PERSON BODY", rep.body), ("third-person unit", native.resolve_mode(cache, rep.unit))] {
                let Some(m) = crate::reach_actor::ReachModel::read(native, cache, tag) else { continue };
                println!(
                    "\n=== row {} {what}: '{}'  ({} nodes, {} meshes, regions {:?})",
                    rep.row,
                    native.tag_name(cache, tag).unwrap_or_default(),
                    m.node_names.len(),
                    m.mesh_fmt.len(),
                    m.region_names
                );
                // Which nodes does ANY mesh of this model touch?
                let mut touched: Vec<usize> = Vec::new();
                for i in 0..m.mesh_fmt.len() {
                    let map = m.node_maps.get(i).cloned().unwrap_or_default();
                    let names: Vec<String> = if map.is_empty() {
                        // A rigid mesh rides one node.
                        let r = m.mesh_rigid_node.get(i).copied().unwrap_or(0xFF);
                        if r == 0xFF {
                            Vec::new()
                        } else {
                            touched.push(r as usize);
                            vec![m.node_names.get(r as usize).cloned().unwrap_or_default()]
                        }
                    } else {
                        map.iter()
                            .map(|&n| {
                                touched.push(n as usize);
                                m.node_names.get(n as usize).cloned().unwrap_or_default()
                            })
                            .collect()
                    };
                    println!(
                        "   mesh {i:<3} fmt {:#04x} {} -> {}",
                        m.mesh_fmt[i],
                        if map.is_empty() { "RIGID  " } else { "skinned" },
                        if names.is_empty() { "<no node>".to_string() } else { names.join(" ") }
                    );
                }
                // #fp-legs  **THE GEOMETRY'S OWN BOUNDING BOX**, straight out of the tag:
                // `render_model` +0x64 `render geometry` +0x10 `compression info`, whose first six
                // floats are the position bounds the compressed vertices are decompressed
                // against. This is the measurement that actually answers "is there a torso in
                // here" -- a model whose geometry stops at the waist cannot draw one, whatever
                // its regions are called.
                const COMPRESSION_INFO: u32 = 0x64 + 0x10;
                if let Some(h) = native.tag_meta(cache, tag, COMPRESSION_INFO, 8) {
                    let c = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
                    let ptr = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
                    if c > 0 && c < 64 {
                        if let Some(b) = native.tag_ptr_bytes(cache, ptr, c * 56) {
                            for k in 0..c as usize {
                                let f = |o: usize| {
                                    let o = k * 56 + o;
                                    if o + 4 <= b.len() {
                                        f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
                                    } else {
                                        f32::NAN
                                    }
                                };
                                // The whole 56-byte element as floats, so the layout is read
                                // rather than assumed: the first word is flags, then the three
                                // (min, max) position pairs, then the texture-coordinate bounds.
                                let all: Vec<String> = (0..14).map(|j| format!("{:+.4}", f(j * 4))).collect();
                                println!("   compression info[{k}] raw: {}", all.join(" "));
                                println!(
                                    "   compression info[{k}]: x {:+.4}..{:+.4}  y {:+.4}..{:+.4}  \
                                     z {:+.4}..{:+.4}   (extent {:.4} x {:.4} x {:.4} wu)",
                                    f(4), f(8), f(12), f(16), f(20), f(24),
                                    f(8) - f(4), f(16) - f(12), f(24) - f(20)
                                );
                            }
                        }
                    }
                }
                touched.sort_unstable();
                touched.dedup();
                // The classification, by node NAME on the parts of a Reach biped skeleton that
                // decide the question.
                let has = |pat: &str| touched.iter().any(|&n| m.node_names.get(n).is_some_and(|s| s.contains(pat)));
                println!(
                    "   touches {} of {} nodes | head {} | spine {} | clavicle {} | hand {} | pelvis {} | thigh {} | foot {}",
                    touched.len(),
                    m.node_names.len(),
                    has("head"),
                    has("spine"),
                    has("clavicle"),
                    has("hand"),
                    has("pelvis"),
                    has("thigh"),
                    has("foot"),
                );
                println!(
                    "   node names touched: {}",
                    touched
                        .iter()
                        .filter_map(|&n| m.node_names.get(n).cloned())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }
    }

    /// #jump  **THE JUMP MEASUREMENT.** What the authored jump velocity actually reads on a real
    /// cache, and what apex and airtime it produces under the gravity the mode uses. **No
    /// constant is changed here** — this prints the inputs so the argument can be had over
    /// measured numbers (see the `reach-engine-physics-gravity` note: the fall constant is an
    /// open question and the fitted one was an artefact).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_jump_as_authored() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("play_mode probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = ActorScene::list_bipeds(native, cache);
        for species in [Species::Spartan, Species::Elite] {
            let Ok((path, p)) = player_physics_for(native, cache, &bipeds, species) else { continue };
            let g_used = p.gravity();
            let apex = |g: f32| p.jump_velocity * p.jump_velocity / (2.0 * g);
            let air = |g: f32| 2.0 * p.jump_velocity / g;
            println!(
                "\n{} ({path})\n  bipd +{:#x} jump velocity = {:.4} wu/s | capsule {:.3} wu tall | eye {:.3} wu",
                species.label(),
                crate::play_phys::BIPD_JUMP_VELOCITY,
                p.jump_velocity,
                p.height_standing,
                p.eye_standing
            );
            for (name, g) in [
                ("the one test mode uses", g_used),
                ("engine authored 4.1712594", crate::reach_physics::WORLD_GRAVITY),
            ] {
                println!(
                    "    g = {g:8.4} ({name:<26}) -> apex {:.3} wu = {:.0}% of standing height, airtime {:.3} s",
                    apex(g),
                    100.0 * apex(g) / p.height_standing.max(1e-4),
                    air(g)
                );
            }
        }
    }
}

/// #ride-gate  A headless RIDE: a real `PlayMode` on a flat floor, a real Warthog placed on it,
/// driven through `tick` exactly as the window drives it. Shared by the ride gates and probes.
#[cfg(test)]
pub(crate) mod ride_harness {
    use super::*;

    pub struct Ride {
        pub pm: PlayMode,
        pub actors: ActorScene,
        pub objects: Vec<ObjectInfo>,
        pub native: std::sync::Arc<NativeDll>,
        pub cache: CacheHandle,
        pub look_yaw: f32,
        pub look_pitch: f32,
        pub time: f32,
    }

    pub const DT: f32 = 1.0 / 60.0;

    impl Ride {
        /// A Spartan standing at the entry marker of `vehicle`'s seat `seat`, looking at it, the
        /// vehicle placed at the origin facing +X.
        pub fn at_seat(
            native: std::sync::Arc<NativeDll>,
            cache: CacheHandle,
            vehicle_path: &str,
            seat: usize,
        ) -> Option<Ride> {
            let tag = native
                .tags_by_group(cache, "vehi")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).as_deref() == Some(vehicle_path))?;
            let placement = ObjectInfo {
                datum: 0x2468,
                type_sig: 0,
                sig0: 0,
                sig1: 0,
                pos: [0.0, 0.0, 0.0],
                health: 1.0,
                shield: 1.0,
                mode_tag: 0,
                fwd: [1.0, 0.0, 0.0],
                up: [0.0, 0.0, 1.0],
                attached: [0; 8],
                primary_tag: tag,
                variant_name_sid: 0,
            };
            let objects = vec![placement];
            let scene = tests::Floor;
            let mut actors = ActorScene::default();
            let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
            let mut pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).ok()?;
            // Where a player stands to use the seat: its entry marker pushed out along its own
            // forward by the authored radius, looking back at the marker.
            let (stand, face) = {
                let e = pm.vehicles.entry(&native, cache, tag)?;
                let s = e.seats.seats.get(seat)?.clone();
                let m = e.marker_frames(&native, cache, s.entry_marker_group()).first().copied()?;
                let world = crate::scene::object_matrix(&objects[0]);
                let w = world * m;
                let fwd = w.transform_vector3(crate::reach_seats::MARKER_FORWARD).normalize();
                let p = w.w_axis.truncate() + fwd * (s.entry_radius * 0.8);
                (Vec3::new(p.x, p.y, 0.0), (-fwd).y.atan2((-fwd).x))
            };
            pm.st = play_ctl::PlayerState::standing_at(stand, face);
            let mut r = Ride { pm, actors, objects, native, cache, look_yaw: face, look_pitch: 0.0, time: 0.0 };
            let look = r.input();
            r.tick(&look);
            Some(r)
        }

        pub fn input(&self) -> MoveInput {
            MoveInput { yaw: self.look_yaw, pitch: self.look_pitch, ..Default::default() }
        }

        /// A vehicle placed at the origin (matched by the END of its tag path) and the player
        /// somewhere clear of it, for tests that seat the player directly.
        pub fn with_vehicle(native: std::sync::Arc<NativeDll>, cache: CacheHandle, suffix: &str) -> Option<Ride> {
            Self::with_vehicle_as(native, cache, suffix, Species::Spartan)
        }

        /// [`Ride::with_vehicle`] for a rider of `species` (Spartan or Elite).
        pub fn with_vehicle_as(native: std::sync::Arc<NativeDll>, cache: CacheHandle, suffix: &str, species: Species) -> Option<Ride> {
            let tag = native
                .tags_by_group(cache, "vehi")
                .into_iter()
                .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(suffix)))?;
            let placement = ObjectInfo {
                datum: 0x2468,
                type_sig: 0,
                sig0: 0,
                sig1: 0,
                pos: [0.0, 0.0, 0.0],
                health: 1.0,
                shield: 1.0,
                mode_tag: 0,
                fwd: [1.0, 0.0, 0.0],
                up: [0.0, 0.0, 1.0],
                attached: [0; 8],
                primary_tag: tag,
                variant_name_sid: 0,
            };
            let mut actors = ActorScene::default();
            let cam = EntryCamera { pos: Vec3::new(30.0, 30.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
            let pm = PlayMode::enter(&native, cache, &tests::Floor, &[], &mut actors, species, cam).ok()?;
            let mut r = Ride { pm, actors, objects: vec![placement], native, cache, look_yaw: 0.0, look_pitch: 0.0, time: 0.0 };
            let look = r.input();
            r.tick(&look);
            Some(r)
        }

        /// Seat the player in seat `seat` of the vehicle `tag` (the hull itself, or one of its
        /// attached turrets: pass the turret's tag) with the boarding already over.
        pub fn force_seat(&mut self, tag: u32, seat: usize) -> Result<(), String> {
            let (native, cache) = (self.native.clone(), self.cache);
            let (path, label, is_child) = {
                let e = self.pm.vehicles.entry(&native, cache, tag).ok_or("not a vehicle")?;
                let s = e.seats.seats.get(seat).ok_or("no such seat")?;
                (e.path.clone(), s.label.clone(), tag != self.objects[0].primary_tag)
            };
            let offer = crate::play_vehicle::VehicleOffer {
                datum: 0x2468,
                parent_datum: is_child.then_some(0x2468),
                vehi_tag: tag,
                vehi_path: path,
                seat,
                seat_label: label,
                prompt_id: String::new(),
                prompt: None,
                substituted: Vec::new(),
                test: crate::reach_seats::EntryTest {
                    distance: 0.0, radius: 1.0, cone: 0.0, cone_limit: 1.0, facing: 0.0, facing_limit: 1.0,
                    relative_speed: 0.0, relative_speed_limit: 1.0,
                },
            };
            self.pm.offer = Some(offer);
            let objects = self.objects.clone();
            self.pm.enter_vehicle(&native, cache, &objects)?;
            if let Some(r) = self.pm.riding.as_mut() {
                r.enter_len = Some(0.0);
                r.exit_len = Some(0.0);
                r.phase = crate::play_vehicle::RidePhase::Seated;
            }
            Ok(())
        }

        /// One frame at 60 Hz through the whole production path.
        pub fn tick(&mut self, input: &MoveInput) {
            self.tick_in(input, &tests::Floor);
        }

        /// [`Ride::tick`] against a scene of the caller's (a real map's collision, placed Forge
        /// blocks) instead of the flat floor: the same `PlayMode::tick`, the world is the only
        /// difference.
        pub fn tick_in(&mut self, input: &MoveInput, scene: &dyn ObjectScene) {
            self.look_yaw = input.yaw;
            self.look_pitch = input.pitch;
            self.pm.tick(input, DT, &self.native, self.cache, scene, &self.objects, &mut self.actors);
            self.actors.update_poses_headless(DT);
            self.time += DT;
        }

        /// A vehicle of `suffix` placed at `pos` facing `yaw` (the extra objects ride along in the
        /// object table, so the hull gather and the exclusion lists see them as the app does).
        pub fn with_vehicle_placed(native: std::sync::Arc<NativeDll>, cache: CacheHandle, suffix: &str, pos: Vec3, yaw: f32, extra: &[ObjectInfo]) -> Option<Ride> {
            let mut r = Self::with_vehicle(native, cache, suffix)?;
            r.objects[0].pos = pos.to_array();
            r.objects[0].fwd = [yaw.cos(), yaw.sin(), 0.0];
            r.objects.extend_from_slice(extra);
            Some(r)
        }

        /// One frame of `dt` seconds through the same path as [`Ride::tick`] (the display's frame
        /// time, not the fixed 60 Hz), for gates that must hold at any frame rate.
        pub fn tick_dt(&mut self, input: &MoveInput, dt: f32) {
            let scene = tests::Floor;
            self.look_yaw = input.yaw;
            self.look_pitch = input.pitch;
            self.pm.tick(input, dt, &self.native, self.cache, &scene, &self.objects, &mut self.actors);
            self.actors.update_poses_headless(dt);
            self.time += dt;
        }

        pub fn run(&mut self, seconds: f32, input: &MoveInput) {
            for _ in 0..(seconds / DT).round() as usize {
                self.tick(input);
            }
        }

        /// #sound-audit  One fire event / one melee swing through the production entry points
        /// (`PlayMode::fire` / `PlayMode::melee`), for the sound audit.
        pub fn fire_once(&mut self) {
            self.pm.fire(&tests::Floor, None, &[]);
        }

        pub fn melee_once(&mut self) {
            self.pm.melee(&tests::Floor, None);
        }

        pub fn body_world(&self) -> Option<glam::Mat4> {
            self.pm.body.and_then(|b| self.actors.get(b)).map(|a| a.world())
        }
    }

    /// #ride-smooth  One drive on a seat (flat floor): 1 s idle, 3 s ahead with a weaving look, 2 s ahead
    /// turning hard, 2 s braking. Returns, per frame, the rider's joints in the HULL's frame
    /// (what the eye sees move against the vehicle) and the hull's own world position.
    pub fn drive_series(r: &mut Ride, scenario: u8) -> (Vec<Vec<Vec3>>, Vec<Vec3>, Vec<String>) {
        let mut joints: Vec<Vec<Vec3>> = Vec::new();
        let mut hull: Vec<Vec3> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let want = ["b_pelvis", "b_spine1", "b_spine2", "b_head", "b_l_hand", "b_r_hand"];
        let total = 8.0f32;
        let n = (total / DT).round() as usize;
        for f in 0..n {
            let t = f as f32 * DT;
            // A mouse look is continuous, so the look is too: a weave that starts and ends at zero,
            // a smooth hard turn, a smooth return -- a stepped yaw is a flick no hand makes and
            // would measure the overlay following it, not roughness.
            let ss = |x: f32| {
                let x = x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            };
            let (fwd, yaw) = if t < 1.0 {
                (0.0, 0.0)
            } else if t < 4.0 {
                (1.0, 0.35 * (std::f32::consts::TAU * (t - 1.0) / 3.0).sin())
            } else if t < 6.0 {
                (1.0, 0.7 * ss(t - 4.0))
            } else {
                (-1.0, 0.7 * (1.0 - ss(t - 6.0)))
            };
            // Scenario 0 is the whole drive, 1 holds still, 2 only drives straight, 3 only weaves the look.
            let (fwd, yaw) = match scenario {
                1 => (0.0, 0.0),
                2 => (if t < 1.0 { 0.0 } else { 1.0 }, 0.0),
                3 => (0.0, 0.35 * (std::f32::consts::TAU * (t - 1.0).max(0.0) / 3.0).sin()),
                _ => (fwd, yaw),
            };
            r.tick(&MoveInput { fwd, yaw, ..Default::default() });
            let (Some(rd), Some(body)) = (r.pm.riding.as_ref(), r.pm.body.and_then(|b| r.actors.get(b))) else { continue };
            let hw = rd.world();
            let inv = hw.inverse();
            let bw = body.world();
            if names.is_empty() {
                names = want.iter().map(|s| s.to_string()).collect();
            }
            let row: Vec<Vec3> = want
                .iter()
                .map(|n| {
                    let k = body.def.model.node_names.iter().position(|x| x == n).expect("node");
                    inv.transform_point3(bw.transform_point3(body.pose.model[k].w_axis.truncate()))
                })
                .collect();
            let mut row = row;
            // The seat marker as the vehicle's ANIMATED model carries it, in the hull's frame.
            let (native, cache) = (r.native.clone(), r.cache);
            let animated = r.pm.vehicle_actors.first().and_then(|&v| r.actors.get(v)).and_then(|veh| {
                let e = r.pm.vehicles.entry(&native, cache, rd.vehi_tag)?;
                let mk = e.first_marker(&native, cache, &e.seats.seats.get(rd.seat)?.marker_name)?;
                Some(inv.transform_point3((veh.world() * veh.pose.marker_model(&mk)).w_axis.truncate()))
            });
            if names.len() == want.len() {
                names.push("seat_marker_anim".into());
            }
            row.push(animated.unwrap_or(Vec3::ZERO));
            joints.push(row);
            hull.push(hw.w_axis.truncate());
        }
        (joints, hull, names)
    }

    /// The frames a ride needs to settle after the seat is taken (the first second is the idle
    /// lead-in of [`drive_series`]; the hands blend onto their grips and the pose onto its clip in
    /// the first ~0.2 s and that transient is not roughness).
    pub const SETTLE_FRAMES: usize = 60;

    /// How far a point wanders in the hull's frame after the settle: the largest extent on any axis.
    pub fn travel(series: &[Vec3]) -> f32 {
        let series = series.get(SETTLE_FRAMES..).unwrap_or(&[]);
        let (lo, hi) = series.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |a, p| (a.0.min(*p), a.1.max(*p)));
        if series.is_empty() { 0.0 } else { (hi - lo).max_element() }
    }

    /// RMS of the second difference (wu per frame^2, scaled to wu/s^2) of a series of points,
    /// after the settle.
    pub fn jitter(series: &[Vec3]) -> f32 {
        let series = series.get(SETTLE_FRAMES..).unwrap_or(&[]);
        if series.len() < 3 {
            return 0.0;
        }
        let mut acc = 0.0f32;
        let mut n = 0;
        for w in series.windows(3) {
            let d2 = (w[2] - 2.0 * w[1] + w[0]) / (DT * DT);
            acc += d2.length_squared();
            n += 1;
        }
        (acc / n as f32).sqrt()
    }
}

/// #ride-gate  PRINT-ONLY: one whole ride (board, drive, get out) on the Warthog, frame by frame.
#[cfg(test)]
mod ride_probes {
    use super::ride_harness::*;
    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_a_whole_warthog_ride() {
        let Some((_g, native, cache)) = open() else { return };
        let Some(mut r) = Ride::at_seat(native, cache, "objects\\vehicles\\human\\warthog\\warthog", 0) else {
            eprintln!("no warthog");
            return;
        };
        println!("offer: {:?}", r.pm.offer.as_ref().map(|o| (o.seat, o.seat_label.clone())));
        if r.pm.offer.is_none() {
            let tag = r.objects[0].primary_tag;
            let (native, cache) = (r.native.clone(), r.cache);
            let world = crate::scene::object_matrix(&r.objects[0]);
            let e = r.pm.vehicles.entry(&native, cache, tag).unwrap();
            let seat = e.seats.seats[0].clone();
            let frames: Vec<glam::Mat4> = e.marker_frames(&native, cache, seat.entry_marker_group()).into_iter().map(|m| world * m).collect();
            let eye = r.pm.st.eye(&r.pm.phys);
            let facing = crate::reach_unit_camera::look_dir(r.pm.st.yaw, r.pm.st.pitch);
            for (i, f) in frames.iter().enumerate() {
                let t = crate::reach_seats::entry_test(&seat, *f, eye, facing, r.pm.st.vel);
                println!("   entry marker {i} at {:?}: {}", f.w_axis.truncate(), t.describe());
            }
            println!("   player eye {eye:?} feet {:?} vel {:?}", r.pm.st.pos, r.pm.st.vel);
        }
        let fp_eye = r.pm.camera();
        println!("on foot: camera {:?}", fp_eye);
        let m = r.pm.enter_vehicle(&r.native.clone(), r.cache, &r.objects.clone());
        println!("enter: {m:?}");
        let print = |r: &Ride, tag: &str| {
            let (p, y, pi) = r.pm.camera();
            let rd = r.pm.riding.as_ref();
            let body = r.pm.body.and_then(|b| r.actors.get(b));
            println!(
                "t={:5.2} {tag:<8} phase {:?} age {:.2} enter_len {:?} cam {:?} blend {:.2} pos ({:6.2},{:6.2},{:6.2}) yaw {:+.2} pitch {:+.2} | hull ({:.2},{:.2},{:.2}) yaw {:+.2} | clip {} frame {:.1}",
                r.time,
                rd.map(|x| x.phase).map(|p| format!("{p:?}")).unwrap_or_default(),
                rd.map(|x| x.phase_age).unwrap_or(0.0),
                rd.and_then(|x| x.enter_len),
                r.pm.cam_mode,
                r.pm.cam_blend.remaining(),
                p.x, p.y, p.z, y, pi,
                rd.map(|x| x.drive.pos.x).unwrap_or(0.0), rd.map(|x| x.drive.pos.y).unwrap_or(0.0), rd.map(|x| x.drive.pos.z).unwrap_or(0.0),
                rd.map(|x| x.drive.yaw).unwrap_or(0.0),
                body.map(|b| b.anim_name.clone()).unwrap_or_default(),
                body.map(|b| b.frame).unwrap_or(0.0),
            );
        };
        let hold = MoveInput { fwd: 1.0, yaw: r.look_yaw, ..Default::default() };
        for i in 0..300 {
            r.tick(&hold);
            if i % 6 == 0 {
                print(&r, "board");
            }
        }
        let steer = MoveInput { fwd: 1.0, yaw: 1.0, ..Default::default() };
        for i in 0..240 {
            r.tick(&steer);
            if i % 20 == 0 {
                print(&r, "drive");
            }
        }
        let still = MoveInput { yaw: r.look_yaw, ..Default::default() };
        r.run(2.0, &still);
        let msg = r.pm.exit_vehicle(&r.native.clone(), r.cache);
        println!("exit: {msg}");
        for i in 0..200 {
            r.tick(&still);
            if i % 6 == 0 {
                print(&r, "exit");
            }
        }

        println!("{}", r.pm.status());
    }

    /// #ride-camera  For each vehicle and seat, at rest and driving on flat ground: the camera
    /// exactly as `PlayMode::camera` hands it to the window, against what the seat's authored
    /// `trak` says it should be.
    ///
    /// Columns: `center` = the bounding-sphere centre the engine focuses on (hull z, centre z);
    /// `ideal` = centre + yaw frame * track offset at the look pitch; `cam` = what was fed to the
    /// renderer; `dev` = |cam - ideal| (the obstruction filter's pull); `h` = camera height above
    /// the ground (the floor is z = 0); `hull-frame` = the camera relative to the hull origin in
    /// the hull's own frame.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_camera_focus() {
        let Some((_g, native, cache)) = open() else { return };
        let cases: [(&str, usize, Option<&str>); 7] = [
            ("warthog\\warthog", 0, None),
            ("warthog\\warthog", 1, None),
            ("warthog\\warthog", 0, Some("warthog_chaingun")),
            ("mongoose\\mongoose", 0, None),
            ("ghost\\ghost", 0, None),
            ("revenant\\revenant", 0, None),
            ("revenant\\revenant", 1, None),
        ];
        for (suffix, seat, child) in cases {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else {
                println!("{suffix}: not in this map");
                continue;
            };
            let hull_tag = r.objects[0].primary_tag;
            let tag = match child {
                None => hull_tag,
                Some(c) => native
                    .enumerate_attachments(cache, hull_tag)
                    .iter()
                    .map(|a| a.child_obj)
                    .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(c)))
                    .expect("child"),
            };
            if let Err(e) = r.force_seat(tag, seat) {
                println!("{suffix} seat {seat}: {e}");
                continue;
            }
            let bo = native
                .tag_meta(cache, hull_tag, 0x10, 12)
                .map(|b| {
                    let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                    Vec3::new(f(0), f(4), f(8))
                })
                .unwrap_or_default();
            let (seat_def, track) = {
                let e = r.pm.vehicles.entry(&native, cache, tag).unwrap();
                let sd = e.seats.seats[seat].clone();
                let t = sd.camera.tracks.first().copied().flatten();
                (sd, t)
            };
            let trak = track.and_then(|t| r.pm.vehicles.track(&native, cache, t).cloned());
            let label = format!("{suffix}#{seat}{}", child.map(|c| format!("({c})")).unwrap_or_default());
            for (tag_s, fwd, pitch) in [("rest", 0.0f32, 0.0f32), ("rest up15", 0.0, 0.2618), ("rest dn15", 0.0, -0.2618), ("driving", 1.0, 0.0), ("driving up15", 1.0, 0.2618)] {
                if fwd > 0.0 && !seat_def.is_driver() {
                    continue;
                }
                let input = MoveInput { fwd, yaw: 0.0, pitch, ..Default::default() };
                r.run(2.5, &input);
                let (cam, cyaw, cpitch) = r.pm.camera();
                let rd = r.pm.riding.as_ref().unwrap();
                let world = rd.world();
                let centre = world.transform_point3(bo);
                let ideal = trak.as_ref().map(|t| {
                    let (_, off) = t.solve(rd.pitch.clamp(seat_def.pitch_range[0].min(seat_def.pitch_range[1]), seat_def.pitch_range[0].max(seat_def.pitch_range[1])));
                    crate::reach_unit_camera::track_eye(centre, rd.yaw, off)
                });
                let hull_frame = world.inverse().transform_point3(cam);
                println!(
                    "{label:<34} {tag_s:<13} hull z {:+.2} centre z {:+.2} | cam ({:+.2},{:+.2},{:+.2}) h {:+.2} | ideal {} dev {} | hull-frame ({:+.2},{:+.2},{:+.2}) | look yaw {:+.2} pitch {:+.2} (cam pitch {:+.2}) seat pitch [{:+.2},{:+.2}] | speed {:.1}",
                    rd.drive.pos.z, centre.z, cam.x, cam.y, cam.z, cam.z,
                    ideal.map_or("none".into(), |i| format!("({:+.2},{:+.2},{:+.2})", i.x, i.y, i.z)),
                    ideal.map_or("-".into(), |i| format!("{:.2}", (i - cam).length())),
                    hull_frame.x, hull_frame.y, hull_frame.z, cyaw, rd.pitch, cpitch,
                    seat_def.pitch_range[0], seat_def.pitch_range[1], rd.drive.vel.length(),
                );
            }
        }
    }

    /// #ride-weapon  `allows weapons` (and the rest of what decides who may sit where) for EVERY
    /// seat of EVERY vehicle in the map.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seat_flag_table() {
        let Some((_g, native, cache)) = open() else { return };
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let mut tags = native.tags_by_group(cache, "vehi");
        tags.sort_by_key(|&t| native.tag_name(cache, t).unwrap_or_default());
        println!("{:<26} {:>2} {:<18} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6}", "vehicle", "#", "seat", "allowW", "driver", "gunner", "boardg", "needDrv", "empty?");
        for t in tags {
            let leaf = native.tag_name(cache, t).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string();
            let Some(e) = index.entry(&native, cache, t) else { continue };
            for s in &e.seats.seats {
                let y = |b: bool| if b { "YES" } else { "-" };
                println!(
                    "{leaf:<26} {:>2} {:<18} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6}",
                    s.index, s.label, y(s.allows_weapons()), y(s.is_driver()), y(s.is_gunner()),
                    y(s.flags & crate::reach_seats::FLAG_BOARDING_SEAT != 0),
                    y(s.flags & crate::reach_seats::FLAG_NOT_VALID_WITHOUT_DRIVER != 0),
                    y(s.enterable_in_empty_vehicle()),
                );
            }
        }
    }

    /// #ride-look  Which aim / look overlay each seat's graph answers, and whether a look moves it.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seat_aim_overlays() {
        let Some((_g, native, cache)) = open() else { return };
        for (suffix, seat, child) in [
            ("warthog\\warthog", 0usize, None::<&str>),
            ("warthog\\warthog", 1, None),
            ("warthog\\warthog", 0, Some("warthog_chaingun")),
            ("mongoose\\mongoose", 0, None),
            ("ghost\\ghost", 0, None),
            ("revenant\\revenant", 0, None),
            ("revenant\\revenant", 1, None),
        ] {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { continue };
            let hull = r.objects[0].primary_tag;
            let tag = match child {
                None => hull,
                Some(c) => native.enumerate_attachments(cache, hull).iter().map(|a| a.child_obj).find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(c))).expect("child"),
            };
            if r.force_seat(tag, seat).is_err() {
                continue;
            }
            let mut result = Vec::new();
            for (yaw, pitch) in [(0.0f32, 0.0f32), (0.8, 0.0), (-0.8, 0.0), (0.0, 0.5), (0.0, -0.5)] {
                let input = MoveInput { yaw, pitch, ..Default::default() };
                r.run(0.5, &input);
                let ov = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| (a.overlays.len(), a.overlays.iter().map(|(_, c)| c.len()).sum::<usize>())).unwrap_or((0, 0));
                // The pose of the head, to see the body actually moves with the look.
                let head = r.pm.body.and_then(|b| r.actors.get(b)).and_then(|a| {
                    a.def.model.node_names.iter().position(|n| n == "b_head").map(|i| a.pose.model[i].w_axis.truncate())
                });
                result.push(format!("look({yaw:+.1},{pitch:+.1}) overlays {ov:?} head {:?}", head.map(|h| (h.x * 100.0).round() / 100.0)));
            }
            println!("{suffix}#{seat}{} aim_missing {} | {}", child.unwrap_or(""), r.pm.aim_missing, result.join(" | "));
        }
    }

    /// #ride-pose  PRINT-ONLY: the pose every enterable seat of every vehicle gives a Spartan and
    /// an Elite -- the clip asked for, whether the graph resolved it, and where the pelvis sits.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seat_pose_table() {
        let Some((_g, native, cache)) = open() else { return };
        for tag in native.tags_by_group(cache, "vehi") {
            let Some(name) = native.tag_name(cache, tag) else { continue };
            let suffix = name.clone();
            for species in [Species::Spartan, Species::Elite] {
                let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, &suffix, species) else { continue };
                let hull = r.objects[0].primary_tag;
                let mut tags = vec![hull];
                tags.extend(native.enumerate_attachments(cache, hull).iter().map(|a| a.child_obj));
                for t in tags {
                    let seats: Vec<(usize, String, u32)> = match r.pm.vehicles.entry(&native, cache, t) {
                        Some(e) => e.seats.seats.iter().filter(|s| s.player_allowed()).map(|s| (s.index, s.label.clone(), s.flags)).collect(),
                        None => continue,
                    };
                    for (i, label, flags) in seats {
                        let Ok(()) = r.force_seat(t, i) else { continue };
                        r.run(0.6, &MoveInput::default());
                        let (mode, class) = (r.pm.riding.as_ref().map(|x| x.seat_mode.clone()).unwrap_or_default(), r.pm.weapon_class.clone().unwrap_or_else(|| crate::film_anim::DEFAULT_WEAPON_CLASS.to_string()));
                        let line = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| {
                            let names = &a.def.model.node_names;
                            let pos = |n: &str| names.iter().position(|x| x == n).map(|k| a.pose.model[k].w_axis.truncate());
                            let (pel, head) = (pos("b_pelvis"), pos("b_head"));
                            let tilt = match (pel, head) { (Some(p), Some(h)) => { let d = (h - p).normalize(); d.z.clamp(-1.0, 1.0).acos().to_degrees() } _ => f32::NAN };
                            let cands = crate::film_anim::anim_candidates(&crate::film_anim::BodyState { seat: Some(crate::film_anim::Seat { vehicle_net_id: 0, mode: mode.clone(), vehicle: String::new() }), ..Default::default() }, &class);
                            let have: Vec<bool> = cands.iter().map(|c| a.def.graph.as_ref().is_some_and(|g| g.find(c).is_some())).collect();
                            format!("clip {:?} standing_fallback {} cands_have {:?} pelvis_z {:.3} spine_tilt {:.0}deg", a.anim_name, a.anim_name.starts_with("combat:"), have, pel.map_or(f32::NAN, |p| p.z), tilt)
                        });
                        println!("{} seat {i} {label} mode {mode:?} {species:?} flags {flags:#x} | {}", name.rsplit('\\').next().unwrap_or(""), line.unwrap_or_default());
                    }
                }
            }
        }
    }

    /// #ride-smooth  PRINT-ONLY: the second-difference jitter of each rider joint (hull frame)
    /// against the hull's own, for the seats the owner rides.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_ride_smoothness() {
        let Some((_g, native, cache)) = open() else { return };
        for (suffix, seat, child) in [
            ("warthog\\warthog", 0usize, None::<&str>),
            ("warthog\\warthog", 1, None),
            ("warthog\\warthog", 0, Some("warthog_chaingun")),
            ("mongoose\\mongoose", 0, None),
            ("mongoose\\mongoose", 1, None),
            ("ghost\\ghost", 0, None),
            ("revenant\\revenant", 1, None),
        ] {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { continue };
            let hull_tag = r.objects[0].primary_tag;
            let tag = match child {
                None => hull_tag,
                Some(c) => native.enumerate_attachments(cache, hull_tag).iter().map(|a| a.child_obj).find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(c))).expect("child"),
            };
            if r.force_seat(tag, seat).is_err() {
                continue;
            }
            for scenario in 0..4u8 {
                let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { continue };
                let _ = r.force_seat(tag, seat);
                let (joints, hull, names) = drive_series(&mut r, scenario);
                let mut cols = Vec::new();
                for (k, n) in names.iter().enumerate() {
                    let series: Vec<Vec3> = joints.iter().map(|row| row[k]).collect();
                    cols.push(format!("{n} {:.1}", jitter(&series)));
                }
                if scenario == 0 {
                    let tr: Vec<String> = names.iter().enumerate().map(|(k, n)| format!("{n} {:.3}", travel(&joints.iter().map(|row| row[k]).collect::<Vec<_>>()))).collect();
                    println!("  hull-frame travel wu after settle: {}", tr.join(", "));
                    for (k, n) in [(3usize, "head"), (4, "l_hand")] {
                        let col: Vec<Vec3> = joints.iter().map(|row| row[k]).collect();
                        let d2: Vec<f32> = col.windows(3).map(|w| ((w[2] - 2.0 * w[1] + w[0]) / (DT * DT)).length()).collect();
                        let bins: Vec<String> = d2.chunks(30).map(|c| format!("{:.0}", c.iter().cloned().fold(0.0f32, f32::max))).collect();
                        println!("  {n} max d2 per 0.5 s: {}", bins.join(" "));
                    }
                }
                if scenario == 1 {
                    let lh: Vec<Vec3> = joints.iter().map(|row| row[4]).collect();
                    let d2: Vec<f32> = lh.windows(3).map(|w| ((w[2] - 2.0 * w[1] + w[0]) / (DT * DT)).length()).collect();
                    let peak = d2.iter().cloned().enumerate().fold((0, 0.0f32), |a, (i, v)| if v > a.1 { (i, v) } else { a });
                    let span = lh.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |a, p| (a.0.min(*p), a.1.max(*p)));
                    println!("  l_hand still: peak d2 {:.1} at frame {}, travel span {:?}, first frames {:?}", peak.1, peak.0, (span.1 - span.0), &d2[..12.min(d2.len())]);
                }
                println!("{suffix}#{seat}{} scenario {scenario} frames {} hull {:.1} | hullframe jitter wu/s2: {}", child.unwrap_or(""), joints.len(), jitter(&hull), cols.join(", "));
            }
        }
    }

    /// #ride-pose  PRINT-ONLY: every clip of the rider graphs whose name starts with a vehicle
    /// seat family, to see how a seat the table says falls back is really named.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seat_family_clips() {
        let Some((_g, native, cache)) = open() else { return };
        for species in [Species::Spartan, Species::Elite] {
            let Some(r) = Ride::with_vehicle_as(native.clone(), cache, "warthog\\warthog", species) else { continue };
            let Some(g) = r.pm.body.and_then(|b| r.actors.get(b)).and_then(|a| a.def.graph.as_ref()) else { continue };
            for fam in ["wraith_d", "wraith_g", "sabre_d", "scorpion_d", "warthog_g", "warthog_p", "falcon_p_l1", "pelican_p_l04", "pelican_p_l05", "warthog_b_g", "mongoose_p", "revenant_p"] {
                let pre = format!("{fam}:");
                let names: Vec<&str> = g.anims.iter().map(|a| a.name.as_str()).filter(|n| n.starts_with(&pre)).collect();
                println!("{species:?} family {fam}: {} clips {:?}", names.len(), names);
            }
        }
    }

    /// #ride-scale  Everything that decides how big the player looks against the world.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_player_scale() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = tests::Floor;
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let input = MoveInput::default();
        for _ in 0..180 {
            pm.tick(&input, 1.0 / 60.0, &native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(1.0 / 60.0);
        }
        let body = actors.get(pm.body.unwrap()).unwrap();
        let w = body.world();
        let names = &body.def.model.node_names;
        let z = |n: &str| names.iter().position(|x| x == n).map(|i| w.transform_point3(body.pose.model[i].w_axis.truncate()).z);
        println!("standing idle: feet (st.pos) z {:+.3} | body world origin z {:+.3} | l_foot {:?} r_foot {:?} l_toe? | pelvis {:?} head {:?} | actor scale {}",
            pm.st.pos.z, w.w_axis.z, z("b_l_foot"), z("b_r_foot"), z("b_pelvis"), z("b_head"), w.x_axis.length());
        println!("capsule: standing height {:.3} eye {:.3} crouch eye {:.3}", pm.phys.height_standing, pm.phys.eye_standing, pm.phys.eye_crouching);
        let bounds = crate::reach_actor::posed_bounds;
        let _ = bounds;
        // The vehicles' bounding data next to it.
        for suffix in ["warthog\\warthog", "mongoose\\mongoose", "ghost\\ghost"] {
            let Some(tag) = native.tags_by_group(cache, "vehi").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(suffix))) else { continue };
            let m = native.resolve_mode(cache, tag);
            let b = crate::reach_actor::model_bounds(&native, cache, m);
            let rad = native.tag_meta(cache, tag, 0x0C, 16).map(|b| {
                let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                (f(0), f(4), f(8), f(12))
            });
            println!("{suffix}: bounds {b:?} | bounding radius/offset {rad:?}");
        }
        println!("fov: world vertical {:.4} rad ({:.1} deg)", cam.fov_y, cam.fov_y.to_degrees());
    }

    /// Where the rider sits against the seat, and how far the hands are from the steering
    /// markers with the grip solve on and off.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_rider_against_the_seat() {
        let Some((_g, native, cache)) = open() else { return };
        for (path, label) in [
            ("objects\\vehicles\\human\\warthog\\warthog", "warthog"),
            ("objects\\vehicles\\human\\mongoose\\mongoose", "mongoose"),
            ("objects\\vehicles\\covenant\\ghost\\ghost", "ghost"),
        ] {
            let Some(mut r) = Ride::at_seat(native.clone(), cache, path, 0) else {
                println!("{label}: not in this map");
                continue;
            };
            let (n, o) = (r.native.clone(), r.objects.clone());
            if let Err(e) = r.pm.enter_vehicle(&n, r.cache, &o) {
                println!("{label}: {e}");
                continue;
            }
            r.look_yaw = 0.0;
            let idle = r.input();
            r.run(4.0, &idle);
            let report = |r: &mut Ride, tag: &str| {
                let body = r.actors.get(r.pm.body.unwrap()).unwrap();
                let veh = r.actors.get(r.pm.vehicle_actors[0]).unwrap();
                let rd = r.pm.riding.as_ref().unwrap();
                let names = &body.def.model.node_names;
                let node = |n: &str| names.iter().position(|x| x == n);
                let bw = body.world();
                let at = |n: &str| node(n).map(|i| bw.transform_point3(body.pose.model[i].w_axis.truncate()));
                let species = "spartan";
                let mut line = format!("{label:<9} {tag:<26} rider origin {:?} steer {:+.1} deg", bw.w_axis.truncate(), rd.drive.steer_angle.to_degrees());
                for (left, hn) in [(true, "b_l_hand"), (false, "b_r_hand")] {
                    let cands = crate::play_vehicle::hand_marker_candidates(&rd.seat_label, left, species);
                    let mk = r.pm.vehicles.entry(&n, cache, rd.vehi_tag).and_then(|e| cands.iter().find_map(|c| e.first_marker(&n, cache, c).map(|m| (c.clone(), m))));
                    if let (Some((cn, m)), Some(h)) = (mk, at(hn)) {
                        let mw = (veh.world() * veh.pose.marker_model(&m)).w_axis.truncate();
                        line += &format!(" | {hn}->{cn} {:.3} wu", (h - mw).length());
                    }
                }
                if let (Some(p), Some(f)) = (at("b_pelvis"), at("b_l_foot")) {
                    line += &format!(" | pelvis z {:.2} foot z {:.2} (hull z {:.2})", p.z, f.z, rd.drive.pos.z);
                }
                println!("{line}");
            };
            report(&mut r, "seated, wheel at rest");
            let right = MoveInput { yaw: -1.2, ..Default::default() };
            r.run(1.2, &right);
            report(&mut r, "full right, grip ON");
            let body_id = r.pm.body.unwrap();
            r.actors.get_mut(body_id).unwrap().grips.clear();
            r.actors.update_poses_headless(0.0);
            report(&mut r, "full right, grip OFF");
            let left = MoveInput { yaw: 1.2, ..Default::default() };
            r.run(1.5, &left);
            report(&mut r, "full left, grip ON");
            r.actors.get_mut(body_id).unwrap().grips.clear();
            r.actors.update_poses_headless(0.0);
            report(&mut r, "full left, grip OFF");
        }
    }
}

/// #ride-gate  The ride, end to end, on the real Warthog from `forge_halo`: the camera blend, the
/// boarding clip, the hand-over of control, the exit and the parked vehicle. Each gate has a
/// saboteur recorded in the commit notes.
///
/// `cargo test --release -p hms-app -- ride_gates --ignored --nocapture`
#[cfg(test)]
mod ride_gates {
    use super::ride_harness::*;
    use super::*;
    use crate::play_vehicle::RidePhase;
    use crate::reach_unit_camera::CamMode;

    const HOG: &str = "objects\\vehicles\\human\\warthog\\warthog";

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// A Warthog with the player boarding it: the harness, the on-foot eye the player had.
    fn boarded() -> Option<(std::sync::MutexGuard<'static, ()>, Ride, Vec3)> {
        let (g, native, cache) = open()?;
        let mut r = Ride::at_seat(native, cache, HOG, 0)?;
        assert!(r.pm.offer.is_some(), "standing at the driver's entry marker must offer the seat");
        let eye = r.pm.camera().0;
        let (n, o) = (r.native.clone(), r.objects.clone());
        r.pm.enter_vehicle(&n, r.cache, &o).expect("enter");
        // The player turns to look along the vehicle once seated: the chase offset is carried by
        // the LOOK yaw, so a look across the hull would put the camera over the player's own head.
        r.look_yaw = 0.0;
        Some((g, r, eye))
    }

    /// The authored chase offset for the Warthog's driver, read from the shipped `trak` at level
    /// pitch: `(-3.00, 0.00, 0.85)` behind and above the hull origin (measured by
    /// `play_vehicle::probes::probe_seat_cameras`).
    fn chase_target(r: &Ride) -> Vec3 {
        let rd = r.pm.riding.as_ref().unwrap();
        let (s, c) = r.look_yaw.sin_cos();
        // The focus is the hull's BOUNDING CENTRE (obje +0x10), 0.38 above the Warthog's origin.
        let bo = r.native.tag_meta(r.cache, rd.placement_tag, 0x10, 12).map(|b| {
            let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            Vec3::new(f(0), f(4), f(8))
        }).unwrap_or_default();
        rd.world().transform_point3(bo) + Vec3::new(c * -3.0, s * -3.0, 0.85)
    }

    /// **THE CAMERA-TRANSITION GATE.** Entering does NOT cut to the third-person camera: the first
    /// frame is still the first-person eye, and the camera arrives on the authored chase offset
    /// after exactly the globals' `enter vehicle transition time`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn entering_a_warthog_blends_from_the_first_person_eye_to_the_authored_chase_offset() {
        let Some((_g, mut r, eye)) = boarded() else { return };
        let look = r.input();
        r.tick(&look);
        let first = r.pm.camera().0;
        let target = chase_target(&r);
        let total = (eye - target).length();
        println!("first-person eye {eye:?}  chase target {target:?}  distance {total:.2}");
        assert!(total > 2.0, "the fixture must start well away from the chase camera, {total}");
        assert!(
            (first - eye).length() < 0.35 * total,
            "frame 1 of the ride must still be near the first-person eye: {first:?} (eye {eye:?}, target {target:?})"
        );
        assert_eq!(r.pm.cam_mode, CamMode::Chase, "the engine's mode is the following camera from frame 1");
        // Sample the approach.
        let mut ds = vec![(first - target).length()];
        let mut done_at = None;
        for i in 0..90 {
            r.tick(&look);
            let p = r.pm.camera().0;
            ds.push((p - chase_target(&r)).length());
            if !r.pm.cam_blend.active() && done_at.is_none() {
                done_at = Some((i + 2) as f32 * DT);
            }
        }
        let done = done_at.expect("the blend never finished");
        println!("blend finished at {done:.3} s; distances to target {:?}", &ds[..8]);
        assert!((done - 0.55).abs() <= 2.0 * DT, "the blend took {done:.3} s; the globals' enter time is 0.55 s");
        // Monotone approach to the target (the target creeps as the hull settles, hence the slack).
        for w in ds[..30].windows(2) {
            assert!(w[1] <= w[0] + 0.02, "the camera moved AWAY from the chase offset: {} -> {}", w[0], w[1]);
        }
        // Halfway it is an ease: neither at the start nor at the end.
        let mid = ds[(0.275 / DT) as usize];
        assert!(mid > 0.1 * total && mid < 0.9 * total, "midway distance {mid:.2} of {total:.2}");
        // And it ENDS on the authored offset, with the look yaw.
        let end = r.pm.camera().0;
        let t = chase_target(&r);
        assert!((end - t).length() < 0.06, "settled at {end:?}, the authored chase offset is {t:?}");
    }

    /// The track the Warthog's driver seat names really is the (-3.00, 0, 0.85) the gate above
    /// assumes -- so the gate is anchored to the tag and not to a copy of the implementation.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_warthogs_authored_chase_track_puts_the_camera_three_behind_and_below_a_metre_above() {
        let Some((_g, native, cache)) = open() else { return };
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let tag = native
            .tags_by_group(cache, "vehi")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).as_deref() == Some(HOG))
            .expect("warthog");
        let seat = index.entry(&native, cache, tag).unwrap().seats.seats[0].clone();
        assert!(seat.is_driver() && seat.third_person(), "seat 0 is the third-person driver: {:?}", seat.flag_names());
        let trak = seat.camera.tracks[0].expect("the driver authors a track");
        let t = index.track(&native, cache, trak).expect("the trak reads").clone();
        let (u, off) = t.solve(0.0);
        println!("track {} at level pitch: t {u:.3} offset {off:?}", native.tag_name(cache, trak).unwrap_or_default());
        assert!((off - Vec3::new(-3.0, 0.0, 0.85)).length() < 0.02, "{off:?}");
        let (_, up) = t.solve(30.0f32.to_radians());
        assert!(up.z < off.z - 1.0, "looking up must drop the camera: {up:?} vs {off:?}");
    }

    /// **THE BOARDING-CLIP GATE.** The rider plays the seat's authored `<seat>:enter` clip for its
    /// authored length, the stick is dead the whole time, and control passes when it ends.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_rider_plays_the_authored_enter_clip_and_cannot_drive_until_it_ends() {
        let Some((_g, mut r, _)) = boarded() else { return };
        let hold = MoveInput { fwd: 1.0, yaw: r.look_yaw, ..Default::default() };
        r.tick(&hold);
        let len = r.pm.riding.as_ref().unwrap().enter_len.expect("the clip length is known after one frame");
        let clip = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| a.anim_name.clone()).unwrap_or_default();
        println!("enter clip '{clip}' {len:.3} s");
        assert!(len > 0.5, "the Warthog's enter clip is authored, it is not zero: {len}");
        assert!(clip.ends_with(":enter"), "the body is playing '{clip}', not the enter clip");
        let mut last_frame = -1.0f32;
        let mut seated_at = None;
        for i in 0..((len + 1.0) / DT) as usize {
            r.tick(&hold);
            let rd = r.pm.riding.as_ref().unwrap();
            if rd.phase == RidePhase::Boarding {
                assert!(rd.drive.odometer < 0.05, "frame {i}: the hull moved {} wu while the rider was still boarding", rd.drive.odometer);
                let flat = Vec3::new(rd.drive.vel.x, rd.drive.vel.y, 0.0).length();
                assert!(flat < 0.1, "frame {i}: horizontal speed {flat} while boarding");
                let f = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| a.frame).unwrap_or(0.0);
                assert!(f >= last_frame, "the clip ran backwards: {last_frame} -> {f}");
                last_frame = f;
            } else if seated_at.is_none() {
                seated_at = Some(r.time);
            }
        }
        let at = seated_at.expect("never seated");
        assert!((at - len).abs() < 0.1, "seated at {at:.3} s but the clip is {len:.3} s");
        assert!(last_frame > len * 30.0 * 0.9, "the clip only reached frame {last_frame} of {}", len * 30.0);
        // Now the stick works.
        r.run(1.5, &hold);
        let rd = r.pm.riding.as_ref().unwrap();
        assert!(rd.drive.odometer > 0.5, "the seated driver must be able to drive: {}", rd.drive.odometer);
        let clip = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| a.anim_name.clone()).unwrap_or_default();
        assert!(clip.ends_with(":idle"), "seated, the body holds the seat idle, not '{clip}'");
    }

    /// **THE LOOK GATE** (kept from the camera fix). The hull turning must not move the player's
    /// look: `st.yaw` is where the player LOOKS, not where the vehicle points.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_hull_turning_does_not_move_the_players_look() {
        let Some((_g, mut r, _)) = boarded() else { return };
        let look = 0.9f32;
        // The jeep steers toward the LOOK (proven), so looking 0.9 rad left of the hull turns it.
        let steer = MoveInput { fwd: 1.0, yaw: look, ..Default::default() };
        r.run(2.5, &steer);
        let hull0 = r.pm.riding.as_ref().unwrap().drive.yaw;
        let mut worst = 0.0f32;
        let mut hull_travel = 0.0f32;
        let mut prev = hull0;
        for _ in 0..240 {
            r.tick(&steer);
            worst = worst.max((r.pm.st.yaw - look).abs());
            let h = r.pm.riding.as_ref().unwrap().drive.yaw;
            hull_travel += (h - prev).abs();
            prev = h;
        }
        println!("hull swept {hull_travel:.2} rad while the look stayed within {worst:.6}");
        assert!(hull_travel > 0.3, "the fixture must actually turn the hull: {hull_travel}");
        let miss = (r.pm.riding.as_ref().unwrap().drive.yaw - look + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        println!("hull yaw is {miss:+.3} rad from the look");
        assert!(miss.abs() < 0.7, "the hull did not come round toward the look: {miss:+.3} rad off after 6.5 s");
        assert!(worst < 1e-5, "the player's look moved by {worst} rad when the hull turned");
        assert!(r.pm.camera().1.is_finite());
    }

    /// **THE EXIT GATE.** Getting out plays `<seat>:exit` for its authored length with the stick
    /// dead, then returns the camera to the first-person eye over the globals' `exit vehicle
    /// transition time`, and the vehicle STAYS where it was left.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn getting_out_plays_the_exit_clip_returns_to_first_person_and_leaves_the_vehicle_parked() {
        let Some((_g, mut r, _)) = boarded() else { return };
        let hold = MoveInput { fwd: 1.0, yaw: r.look_yaw, ..Default::default() };
        r.run(4.0, &hold);
        assert_eq!(r.pm.riding.as_ref().unwrap().phase, RidePhase::Seated);
        let still = MoveInput { yaw: r.look_yaw, ..Default::default() };
        r.run(2.0, &still);
        let (n, c) = (r.native.clone(), r.cache);
        let msg = r.pm.exit_vehicle(&n, c);
        println!("{msg}");
        assert_eq!(r.pm.riding.as_ref().unwrap().phase, RidePhase::Exiting);
        r.tick(&still);
        let exit_len = r.pm.riding.as_ref().unwrap().exit_len.expect("known");
        let clip = r.pm.body.and_then(|b| r.actors.get(b)).map(|a| a.anim_name.clone()).unwrap_or_default();
        println!("exit clip '{clip}' {exit_len:.3} s");
        assert!(clip.ends_with(":exit"), "the body is playing '{clip}' instead of the exit clip");
        // Still riding for the whole clip.
        let n_frames = (exit_len / DT) as usize;
        for i in 0..n_frames.saturating_sub(4) {
            r.tick(&hold);
            assert!(r.pm.riding.is_some(), "left the seat at frame {i}, before the {exit_len:.2} s clip had played");
        }
        r.run(0.2, &still);
        assert!(r.pm.riding.is_none(), "still in the seat after the exit clip");
        // Stepped out onto the ground, not dropped from the marker's height.
        assert!(r.pm.st.pos.z.abs() < 0.05, "the player left the seat at z {} instead of on the floor", r.pm.st.pos.z);
        let parked_at = r.pm.parked.first().map(|p| p.ride.drive.pos).expect("the vehicle is parked, not removed");
        let chase_last = r.pm.cam_prev.unwrap();
        // The camera comes back to the eye: not instantly, and not later than the exit time.
        let eye = r.pm.st.eye(&r.pm.phys);
        let d0 = (r.pm.camera().0 - eye).length();
        println!("camera {d0:.2} wu from the eye on the first foot frames (chase pose {chase_last:?})");
        // Leaving a chase shot is a BLEND, not a cut: on the first foot frame the camera is still
        // out at the chase position and a transition is running.
        assert!(d0 > 1.0 && r.pm.cam_blend.active(), "the camera cut to the eye ({d0:.2} wu away, blending {})", r.pm.cam_blend.active());
        r.run(0.7, &still);
        let d1 = (r.pm.camera().0 - r.pm.st.eye(&r.pm.phys)).length();
        assert!(d1 < 1e-3 && !r.pm.cam_blend.active(), "the camera never got back to the first-person eye: {d1}");
        assert_eq!(r.pm.cam_mode, CamMode::Foot);
        // The vehicle was left where it was driven to, and the handbrake then holds it still.
        assert!(parked_at.length() > 3.0, "it was parked at {parked_at:?}, nowhere near the 10+ wu it was driven");
        r.run(3.0, &still);
        let settled = r.pm.parked[0].ride.drive.pos;
        r.run(1.0, &still);
        let after = r.pm.parked[0].ride.drive.pos;
        assert!((after - settled).length() < 0.05, "the parked vehicle kept rolling: {} wu in a second", (after - settled).length());
        assert_eq!(r.pm.hidden_datums(), vec![0x2468], "the original placement stays hidden once, not twice");
        assert_eq!(r.pm.parked[0].actors.len(), 2, "a Warthog is its body and one turret, still drawn");
    }

    /// **THE NO-DUPLICATE GATE.** Whatever phase the ride is in, the render set loses the ridden
    /// placement, there is exactly one set of vehicle actors, and the actors are where the
    /// vehicle is -- and after a full ride and a second boarding it is the SAME vehicle, not a new
    /// one.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn there_is_one_warthog_before_during_and_after_the_ride_and_it_is_the_same_one() {
        let Some((_g, mut r, _)) = boarded() else { return };
        // The Warthog's hull is the only actor with a steering wheel node.
        let count = |r: &Ride| {
            r.actors.actors.iter().filter(|a| a.def.model.node_names.iter().any(|n| n == "b_steering_wheel")).count()
        };
        let all_vehicle_actors = |r: &Ride| -> usize {
            r.pm.vehicle_actors.len() + r.pm.parked.iter().map(|p| p.actors.len()).sum::<usize>()
        };
        let hold = MoveInput { fwd: 1.0, yaw: r.look_yaw, ..Default::default() };
        r.run(0.5, &hold);
        assert_eq!(count(&r), 1, "while boarding: one hull");
        assert_eq!(all_vehicle_actors(&r), 2, "while boarding: the hull and its turret");
        r.run(4.0, &hold);
        let shown = crate::play_vehicle::shown(&r.objects, &r.pm.hidden_datums()).expect("a ride filters");
        assert!(shown.iter().all(|o| o.datum != 0x2468), "the render set still holds the ridden placement");
        assert_eq!(count(&r), 1);
        assert_eq!(all_vehicle_actors(&r), 2);
        let moved = r.pm.riding.as_ref().unwrap().drive.pos;
        assert!(moved.length() > 3.0, "the Warthog should have driven off: {moved:?}");
        let still = MoveInput { yaw: r.look_yaw, ..Default::default() };
        r.run(1.0, &still);
        let (n, c) = (r.native.clone(), r.cache);
        r.pm.exit_vehicle(&n, c);
        r.run(3.0, &still);
        assert!(r.pm.riding.is_none());
        assert_eq!(count(&r), 1, "after getting out the vehicle is still drawn once");
        assert_eq!(all_vehicle_actors(&r), 2);
        let ids_before: Vec<u32> = r.pm.parked[0].actors.clone();
        // It is parked WHERE IT WAS LEFT, not where it began.
        let at = r.pm.parked[0].ride.drive.pos;
        assert!(at.length() > 3.0 && at.x >= moved.x - 0.1, "parked at {at:?}, was driven to {moved:?}");
        // Walk back to it, board again: the same actors, the same place.
        let objs = r.objects.clone();
        let eff = r.pm.objects_with_parked(&objs).into_owned();
        assert!((Vec3::from(eff[0].pos) - at).length() < 1e-3, "the entry test must see the parked vehicle, not the placement");
        let e = {
            let ent = r.pm.vehicles.entry(&n, c, eff[0].primary_tag).unwrap();
            let s = ent.seats.seats[0].clone();
            let m = ent.marker_frames(&n, c, s.entry_marker_group())[0];
            (s, m)
        };
        let world = crate::scene::object_matrix(&eff[0]);
        let w = world * e.1;
        let fwd = w.transform_vector3(crate::reach_seats::MARKER_FORWARD).normalize();
        let p = w.w_axis.truncate() + fwd * (e.0.entry_radius * 0.8);
        r.pm.st = play_ctl::PlayerState::standing_at(Vec3::new(p.x, p.y, 0.0), (-fwd).y.atan2((-fwd).x));
        let look = MoveInput { yaw: r.pm.st.yaw, ..Default::default() };
        r.tick(&look);
        r.tick(&look);
        assert!(r.pm.offer.is_some(), "the parked vehicle must be offered where it stands");
        r.pm.enter_vehicle(&n, c, &objs).expect("re-enter");
        assert_eq!(r.pm.vehicle_actors, ids_before, "re-boarding must resume the SAME actors, not spawn new ones");
        assert!(r.pm.parked.is_empty());
        r.run(0.3, &look);
        assert_eq!(count(&r), 1, "re-boarding must not add a second hull");
        assert_eq!(all_vehicle_actors(&r), 2);
        assert!((r.pm.riding.as_ref().unwrap().drive.pos - at).length() < 0.5, "resumed where it was parked");
    }

    /// **THE GRIP GATE.** A driver's hands stay on the steering wheel / handlebars as it turns.
    /// Measured in the MARKER's own frame: with the grip solve the hand does not move relative to
    /// the marker it holds (the total drift is less than half of what it is without the solve, and
    /// no hand is more than 5 cm off); without it the turning wheel leaves the hands behind -- so
    /// the gate can tell the two apart.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_drivers_hands_stay_on_the_wheel_when_it_turns() {
        let Some((_g, native, cache)) = open() else { return };
        for (path, side) in [
            ("objects\\vehicles\\human\\warthog\\warthog", "warthog"),
            ("objects\\vehicles\\human\\mongoose\\mongoose", "mongoose"),
        ] {
            let Some(mut r) = Ride::at_seat(native.clone(), cache, path, 0) else {
                eprintln!("{side} is not in this map");
                continue;
            };
            let (n, o) = (r.native.clone(), r.objects.clone());
            r.pm.enter_vehicle(&n, r.cache, &o).expect("enter");
            r.look_yaw = 0.0;
            let idle = r.input();
            r.run(4.0, &idle);
            // Each hand's position in the frame of the marker it grips.
            let in_marker_frame = |r: &mut Ride| -> [Option<Vec3>; 2] {
                let rd = r.pm.riding.as_ref().unwrap().clone();
                let body = r.actors.get(r.pm.body.unwrap()).unwrap();
                let veh = r.actors.get(r.pm.vehicle_actors[0]).unwrap();
                let (bw, vw, vpose) = (body.world(), veh.world(), veh.pose.clone());
                let names = body.def.model.node_names.clone();
                let hands: Vec<Vec3> = [true, false]
                    .iter()
                    .map(|&left| {
                        let chain = crate::reach_grip::ArmChain::find(&names, left).expect("the biped has arms");
                        bw.transform_point3(body.pose.model[chain.hand].w_axis.truncate())
                    })
                    .collect();
                let mut out = [None, None];
                for (k, left) in [(0usize, true), (1usize, false)] {
                    let cands = crate::play_vehicle::hand_marker_candidates(&rd.seat_label, left, "spartan");
                    let mk = r.pm.vehicles.entry(&n, cache, rd.vehi_tag).and_then(|e| cands.iter().find_map(|c| e.first_marker(&n, cache, c)));
                    if let Some(m) = mk {
                        let mw = vw * vpose.marker_model(&m);
                        out[k] = Some(mw.inverse().transform_point3(hands[k]));
                    }
                }
                out
            };
            let rest = in_marker_frame(&mut r);
            let right = MoveInput { yaw: -1.2, ..Default::default() };
            r.run(1.2, &right);
            let steer = r.pm.riding.as_ref().unwrap().drive.steer_angle.to_degrees();
            let on = in_marker_frame(&mut r);
            let body_id = r.pm.body.unwrap();
            r.actors.get_mut(body_id).unwrap().grips.clear();
            r.actors.update_poses_headless(0.0);
            let off = in_marker_frame(&mut r);
            println!("{side}: steer {steer:+.0} deg; hand drift in the marker frame ON {:?} OFF {:?}",
                [0, 1].map(|k| on[k].zip(rest[k]).map(|(a, b)| (a - b).length())),
                [0, 1].map(|k| off[k].zip(rest[k]).map(|(a, b)| (a - b).length())));
            assert!(steer.abs() > 20.0, "{side}: the fixture must actually turn the wheel ({steer})");
            let (mut on_sum, mut off_sum, mut seen) = (0.0f32, 0.0f32, 0);
            for k in 0..2 {
                let (Some(a), Some(b), Some(c)) = (rest[k], on[k], off[k]) else { continue };
                seen += 1;
                // An arm that is nearly at full stretch cannot follow the wheel all the way, so a
                // hand may stay a few cm off -- but never far.
                assert!((b - a).length() < 0.05, "{side} hand {k}: {:.3} wu off the wheel with the grip ON", (b - a).length());
                on_sum += (b - a).length();
                off_sum += (c - a).length();
            }
            assert!(seen >= 1, "{side}: no hand marker resolved");
            assert!(off_sum > 0.06, "{side}: without the grip the hands only drift {off_sum:.3} wu in total, so this gate cannot tell the grip from nothing");
            assert!(on_sum < 0.5 * off_sum, "{side}: the grip recovers too little of the drift: ON {on_sum:.3} vs OFF {off_sum:.3} wu");
        }
    }

    /// **THE CAMERA-HEIGHT GATE.** Reported: the Mongoose and Warthog cameras "constantly clip into
    /// the ground". The engine focuses a third-person seat that names no camera marker on the
    /// root object's BOUNDING-SPHERE CENTRE (`sub_140D467A0`), not its ground-level model origin,
    /// and the authored `trak` offsets are relative to that centre. Measured numbers (level pitch,
    /// flat ground): Warthog 0.87 -> 1.25 above the ground, Mongoose 0.51 -> 0.75, Ghost 0.95 ->
    /// 1.10. The gate: the camera sits at `centre + track offset`, and never closer to the ground
    /// than the shortest authored level offset allows.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_chase_camera_is_focused_on_the_bounding_centre_and_clears_the_ground() {
        let Some((_g, native, cache)) = open() else { return };
        for (suffix, min_h) in [("warthog\\warthog", 1.1f32), ("mongoose\\mongoose", 0.7), ("ghost\\ghost", 1.0)] {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { continue };
            let tag = r.objects[0].primary_tag;
            r.force_seat(tag, 0).expect("seat");
            let bo = native.tag_meta(cache, tag, 0x10, 12).map(|b| f32::from_le_bytes([b[8], b[9], b[10], b[11]])).unwrap();
            let level = r.input();
            r.run(3.0, &level);
            let (cam, _, _) = r.pm.camera();
            let hull = r.pm.riding.as_ref().unwrap().drive.pos;
            let trak = {
                let t = r.pm.vehicles.entry(&native, cache, tag).unwrap().seats.seats[0].camera.tracks[0].unwrap();
                r.pm.vehicles.track(&native, cache, t).unwrap().clone()
            };
            let (_, off) = trak.solve(0.0);
            println!("{suffix}: hull z {:.3}, bounding centre +{bo:.3}, track level z {:.3} -> camera z {:.3} above the ground", hull.z, off.z, cam.z);
            assert!(bo > 0.1, "{suffix}: the bounding centre must sit above the origin or this gate proves nothing ({bo})");
            assert!((cam.z - (hull.z + bo + off.z)).abs() < 0.03, "{suffix}: camera z {:.3} but centre + track says {:.3}", cam.z, hull.z + bo + off.z);
            assert!(cam.z > min_h, "{suffix}: the camera is only {:.2} above the ground at level pitch", cam.z);
        }
    }

    /// **THE EMPTY-VEHICLE GATE.** Standing at a hijack seat's entry marker of an EMPTY Mongoose
    /// (or at its passenger seat's) must not put you in it: the offer is the driver's seat or
    /// nothing.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn an_empty_mongoose_cannot_be_boarded_through_its_hijack_or_passenger_seats() {
        let Some((_g, native, cache)) = open() else { return };
        let path = "objects\\vehicles\\human\\mongoose\\mongoose";
        let mut seen = 0;
        for seat in 1..4 {
            let Some(r) = Ride::at_seat(native.clone(), cache, path, seat) else { continue };
            seen += 1;
            let taken = r.pm.offer.as_ref().map(|o| o.seat);
            println!("standing at seat {seat}'s entry marker: offer {taken:?}");
            assert!(matches!(taken, None | Some(0)), "seat {seat}'s marker offered seat {taken:?} of an empty Mongoose");
        }
        assert!(seen >= 3, "the fixture must stand at three marker groups (saw {seen})");
        // The driver's own marker still works.
        let r = Ride::at_seat(native.clone(), cache, path, 0).expect("mongoose");
        assert_eq!(r.pm.offer.as_ref().map(|o| o.seat), Some(0));
    }

    /// #ride-pose  **No player seat of any vehicle draws its rider standing.** Every seat of every
    /// `forge_halo` vehicle, for a Spartan and for an Elite: the clip the rider's graph actually
    /// loads is never a `combat:` one, it belongs to the seat's own mode (or to a declared
    /// stand-in family when the graph authors nothing for it), and the pelvis sits within the
    /// seated band of the seat marker. Measured before the fix: 87 of 217 rows loaded
    /// `combat:rifle:idle` (the Elite Warthog gunner, both Wraith seats, the Elite Scorpion
    /// driver, `sabre_d`, the Pelican, the Elite Falcon passengers ...); after: 0.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn no_player_seat_draws_its_rider_standing() {
        let Some((_g, native, cache)) = open() else { return };
        let mut rows = 0;
        let mut stand_ins = 0;
        let mut worst_pelvis = 0.0f32;
        for tag in native.tags_by_group(cache, "vehi") {
            let Some(name) = native.tag_name(cache, tag) else { continue };
            for species in [Species::Spartan, Species::Elite] {
                let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, &name, species) else { continue };
                let hull = r.objects[0].primary_tag;
                let mut tags = vec![hull];
                tags.extend(native.enumerate_attachments(cache, hull).iter().map(|a| a.child_obj));
                for t in tags {
                    let seats: Vec<(usize, bool)> = match r.pm.vehicles.entry(&native, cache, t) {
                        Some(e) => e.seats.seats.iter().filter(|s| s.player_allowed()).map(|s| (s.index, s.enterable_in_empty_vehicle())).collect(),
                        None => continue,
                    };
                    for (i, enterable) in seats {
                        if r.force_seat(t, i).is_err() {
                            continue;
                        }
                        r.run(0.6, &MoveInput::default());
                        let mode = r.pm.riding.as_ref().unwrap().seat_mode.clone();
                        let a = r.actors.get(r.pm.body.unwrap()).unwrap();
                        let clip = a.anim_name.clone();
                        assert!(!clip.starts_with("combat:"), "{name} seat {i} '{mode}' {species:?} draws the STANDING clip {clip}");
                        let own = clip.split(':').next() == Some(mode.as_str());
                        if !own {
                            stand_ins += 1;
                            let family = ["warthog_d", "warthog_g", "warthog_p", "ghost_d", "mongoose_d", "mongoose_p", "wraith_g"];
                            let sibling = clip.split('_').next() == mode.split('_').next();
                            assert!(sibling || family.iter().any(|f| clip.starts_with(f)), "{name} seat '{mode}' {species:?}: unexpected stand-in {clip}");
                        }
                        let pel = a.def.model.node_names.iter().position(|n| n == "b_pelvis").map(|k| a.pose.model[k].w_axis.z).unwrap();
                        worst_pelvis = worst_pelvis.max(pel.abs());
                        // A hijack seat (`boarding seat`) is a perch on the outside of the hull, not a
                        // sitting pose: its authored idle is held to the clip rule, not the seated band.
                        assert!(!enterable || pel.abs() < 0.25, "{name} seat '{mode}' {species:?} pelvis {pel:.3} from the seat marker is a standing height ({clip})");
                        rows += 1;
                    }
                }
            }
        }
        println!("{rows} seat rows, {stand_ins} on a stand-in clip, worst |pelvis| {worst_pelvis:.3}");
        assert!(rows > 150 && stand_ins > 0, "liveness: {rows} rows, {stand_ins} stand-ins");
    }

    /// #ride-smooth  **A passenger rides smoothly through a drive.** Still, straight, a weaving
    /// look, a hard turn, a brake (continuous inputs, `drive_series`): in the HULL's frame the
    /// passenger's pelvis, spine, head and hands move smoothly -- the second difference after the
    /// settle stays under 3 wu/s^2 (measured 0.0-0.6 on the Warthog and Mongoose passengers; a
    /// look quantised to 3 degrees, the saboteur, reads 6.7 on the head).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_passenger_rides_smoothly_through_a_drive() {
        let Some((_g, native, cache)) = open() else { return };
        for (suffix, seat) in [("warthog\\warthog", 1usize), ("mongoose\\mongoose", 1)] {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { return };
            let tag = r.objects[0].primary_tag;
            r.force_seat(tag, seat).expect("seat");
            let (joints, _hull, names) = drive_series(&mut r, 0);
            assert!(joints.len() > 400, "liveness: {} frames", joints.len());
            let moved = names.iter().enumerate().map(|(k, _)| travel(&joints.iter().map(|row| row[k]).collect::<Vec<_>>())).fold(0.0f32, f32::max);
            assert!(moved > 0.03, "liveness: nothing on the rider moved ({moved}) -- the drive did not happen");
            for (k, n) in names.iter().enumerate() {
                let series: Vec<Vec3> = joints.iter().map(|row| row[k]).collect();
                let j = jitter(&series);
                assert!(n == "seat_marker_anim" || j < 3.0, "{suffix} seat {seat} {n}: jitter {j:.1} wu/s2 >= 3");
            }
        }
    }

    /// #ride-grip  **Only a driver's hands are gripped to the steering markers.** The shared marker
    /// names (`left_hand_d_spartan`, bare `left_hand`) are the DRIVER's; a passenger matched
    /// against them had his hands pinned to the driver's wheel in the vehicle's frame while his
    /// own look moved his arms. The Warthog passenger authors no hand marker (free hands); the
    /// Mongoose and Revenant passengers author `<seat>_left_hand`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn only_a_driver_is_gripped_to_the_steering_markers() {
        let Some((_g, native, cache)) = open() else { return };
        for (suffix, seat, grips) in [("warthog\\warthog", 0usize, 2usize), ("warthog\\warthog", 1, 0), ("mongoose\\mongoose", 0, 2), ("mongoose\\mongoose", 1, 1), ("revenant\\revenant", 1, 2)] {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { return };
            let tag = r.objects[0].primary_tag;
            r.force_seat(tag, seat).expect("seat");
            r.run(0.5, &MoveInput::default());
            let n = r.actors.get(r.pm.body.unwrap()).unwrap().grips.len();
            assert_eq!(n, grips, "{suffix} seat {seat}: {n} hand grips");
        }
    }


    /// #ride-entry  **Enter and exit are the seat's real clips, by exact name.** Every player seat
    /// of every `forge_halo` vehicle, for both species and a spread of weapon classes: the clip
    /// `first_authored_clip` resolves is one of the names asked for, a BASE clip, never a `_pose`
    /// overlay or a neighbouring `enter_a` / `exit_b`, and its length is the authored clip's.
    /// Measured on the substring rule it replaced: `mongoose_d:rifle:exit` resolved to the 1-frame
    /// overlay `mongoose_d:rifle:exit_pose` (`MISMATCH` rows below count every such seat).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn enter_and_exit_resolve_to_the_authored_clip_by_exact_name() {
        let Some((_g, native, cache)) = open() else { return };
        let (mut checked, mut substring_wrong, mut resolved) = (0, 0, 0);
        for tag in native.tags_by_group(cache, "vehi") {
            let Some(name) = native.tag_name(cache, tag) else { continue };
            for species in [Species::Spartan, Species::Elite] {
                let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, &name, species) else { continue };
                let hull = r.objects[0].primary_tag;
                let mut tags = vec![hull];
                tags.extend(native.enumerate_attachments(cache, hull).iter().map(|a| a.child_obj));
                let Some(graph) = r.pm.body.and_then(|b| r.actors.get(b)).and_then(|a| a.def.graph.clone()) else { continue };
                for t in tags {
                    let modes: Vec<String> = match r.pm.vehicles.entry(&native, cache, t) {
                        Some(e) => e.seats.seats.iter().filter(|s| s.player_allowed()).map(|s| s.label.clone()).collect(),
                        None => continue,
                    };
                    for mode in modes {
                        for class in ["rifle", "pistol", "unarmed", "sword", "any"] {
                            for ph in [RidePhase::Boarding, RidePhase::Exiting] {
                                let cands = crate::play_vehicle::phase_clip_candidates(&mode, class, ph);
                                checked += 1;
                                let old = cands.iter().find_map(|c| graph.find(c)).map(|i| (graph.anims[i].name.clone(), graph.anims[i].frame_count));
                                let new = crate::play_vehicle::first_authored_clip(&graph, &cands);
                                if let Some((n, secs)) = &new {
                                    resolved += 1;
                                    assert!(cands.iter().any(|c| c.eq_ignore_ascii_case(n)), "{name} {mode} {class} {ph:?}: '{n}' is not a name that was asked for");
                                    assert!(!n.ends_with("_pose"), "{name} {mode}: '{n}' is a pose overlay");
                                    let a = graph.anims.iter().find(|a| &a.name == n).unwrap();
                                    assert_eq!(a.anim_type, crate::reach_jmad::AnimType::Base, "{n} is not a base clip");
                                    assert!((secs * crate::film_anim::CLIP_FPS - a.frame_count as f32).abs() < 0.5, "{n}: length {secs}");
                                }
                                if let (Some((on, of)), Some((nn, ns))) = (&old, &new) {
                                    if on != nn || (*of as f32 - ns * crate::film_anim::CLIP_FPS).abs() > 0.5 {
                                        substring_wrong += 1;
                                    }
                                } else if old.is_some() && new.is_none() {
                                    substring_wrong += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("{checked} seat/class/phase lookups, {resolved} resolved exactly, {substring_wrong} where the substring rule gave a different clip");
        assert!(checked > 1500 && resolved > 300, "liveness: {checked} lookups, {resolved} resolved");
        assert!(substring_wrong > 0, "the substring bug no longer reproduces; the gate can no longer distinguish the rules");
    }

    /// #ride-entry  End to end: boarding the Mongoose plays an authored `enter` clip of its full
    /// length, not a 1-frame overlay, and the phase waits for it.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn boarding_plays_the_authored_enter_clip_for_its_full_length() {
        let Some((_g, native, cache)) = open() else { return };
        let Some(mut r) = Ride::at_seat(native.clone(), cache, "objects\\vehicles\\human\\mongoose\\mongoose", 0) else { return };
        let (n, o) = (r.native.clone(), r.objects.clone());
        r.pm.enter_vehicle(&n, r.cache, &o).expect("enter");
        let look = r.input();
        r.tick(&look);
        let rd = r.pm.riding.as_ref().unwrap();
        let len = rd.enter_len.expect("enter length read");
        let a = r.actors.get(r.pm.body.unwrap()).unwrap();
        println!("boarding clip '{}' len {len:.2}s", a.anim_name);
        assert!(a.anim_name.contains("enter") && !a.anim_name.ends_with("_pose"), "boarding plays '{}'", a.anim_name);
        let frames = a.frame_count();
        assert!(frames > 5.0 && (len * crate::film_anim::CLIP_FPS - frames).abs() < 1.5, "clip {frames} frames but the phase waits {len}s");
        // ...and so does leaving: the Mongoose authors no `mongoose_d:rifle:exit`, only the
        // 1-frame overlay `mongoose_d:rifle:exit_pose` the substring rule used to hand it.
        r.run(len + 0.2, &look);
        assert_eq!(r.pm.riding.as_ref().unwrap().phase, RidePhase::Seated, "the boarding phase ended");
        let _ = r.pm.exit_vehicle(&n, r.cache);
        r.tick(&look);
        let a = r.actors.get(r.pm.body.unwrap()).unwrap();
        let (frames, xl) = (a.frame_count(), r.pm.riding.as_ref().and_then(|x| x.exit_len).unwrap_or(0.0));
        println!("leaving clip '{}' len {xl:.2}s", a.anim_name);
        assert!(a.anim_name.contains("exit") && !a.anim_name.ends_with("_pose"), "leaving plays '{}'", a.anim_name);
        assert!(frames > 5.0 && (xl * crate::film_anim::CLIP_FPS - frames).abs() < 1.5, "exit clip {frames} frames but the phase waits {xl}s");
    }

    /// #use-hold  **A seat is entered by HOLDING the use button for `matg`'s authored time.** At the
    /// Warthog's driver marker: the prompt reads "Hold E to drive Warthog" and sits on the player;
    /// a tap enters nothing; 14 ticks (0.2333 s) enters and 13 do not; letting go early cancels
    /// and restarts; the hold takes the same time at 30, 60 and 144 fps; the key's auto-repeat
    /// while still holding it does not throw the player back out, and a fresh press does.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_seat_is_entered_by_holding_the_use_button_for_the_authored_time() {
        let Some((_g, native, cache)) = open() else { return };
        let fresh = || Ride::at_seat(native.clone(), cache, HOG, 0);
        let step = |r: &mut Ride, down: bool, pressed: bool, dt: f32| {
            let look = r.input();
            r.tick_dt(&look, dt);
            r.pm.use_decide(down, pressed, dt)
        };
        // The prompt: the seat's own authored words with the player's key.
        let r = fresh().expect("warthog");
        let prompt = r.pm.action_prompt().expect("a prompt is on offer at the driver's door");
        assert_eq!(prompt, "Hold E to drive Warthog", "the engine's own words, the player's key");
        assert_eq!(r.pm.hold_cfg.ticks_to_hold(), 14, "matg 0.2333 s");

        // A tap: down for 3 frames, then up for a second. Nothing.
        let mut r = fresh().unwrap();
        for f in 0..180 {
            let down = f < 3;
            assert_ne!(step(&mut r, down, f == 0, DT), UseDecision::Hold, "a tap entered at frame {f}");
        }
        assert!(r.pm.riding.is_none());

        // Exactly the authored time: frame 14 acts, frame 13 did not.
        let mut r = fresh().unwrap();
        for f in 1..=14 {
            let d = step(&mut r, true, f == 1, DT);
            assert_eq!(d == UseDecision::Hold, f == 14, "frame {f}: {d:?}");
        }

        // Early release cancels and resets.
        let mut r = fresh().unwrap();
        for f in 1..=13 {
            assert_ne!(step(&mut r, true, f == 1, DT), UseDecision::Hold);
        }
        assert_ne!(step(&mut r, false, false, DT), UseDecision::Hold);
        for f in 1..=13 {
            assert_ne!(step(&mut r, true, f == 1, DT), UseDecision::Hold, "the 13 ticks before the release must not count (frame {f})");
        }
        assert_eq!(step(&mut r, true, false, DT), UseDecision::Hold);

        // Frame-rate independence through the whole production path.
        for fps in [30.0f32, 60.0, 144.0] {
            let dt = 1.0 / fps;
            let mut r = fresh().unwrap();
            let mut t = 0.0;
            let mut at = None;
            for f in 0..600 {
                if step(&mut r, true, f == 0, dt) == UseDecision::Hold {
                    at = Some(t + dt);
                    break;
                }
                t += dt;
            }
            let at = at.unwrap_or_else(|| panic!("{fps} fps never held"));
            assert!(at >= 0.2333 - 1e-3 && at <= 0.2333 + dt + 1.0 / 60.0 + 1e-3, "{fps} fps entered after {at} s");
        }

        // Enter by the hold, keep holding (auto-repeat presses): still seated. Release, press: out.
        let mut r = fresh().unwrap();
        let mut entered = false;
        for f in 0..60 {
            if step(&mut r, true, f == 0, DT) == UseDecision::Hold {
                let (n, o) = (r.native.clone(), r.objects.clone());
                r.pm.enter_vehicle(&n, r.cache, &o).expect("enter");
                entered = true;
                break;
            }
        }
        assert!(entered && r.pm.riding.is_some());
        for _ in 0..90 {
            assert_ne!(step(&mut r, true, true, DT), UseDecision::Exit, "auto-repeat while the key is still down must not exit");
        }
        assert_ne!(step(&mut r, false, false, DT), UseDecision::Exit);
        assert_eq!(step(&mut r, true, true, DT), UseDecision::Exit, "a fresh press leaves");
    }

    /// #use-hold  No prompt, and no hold, when no seat is in range.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn there_is_no_prompt_and_no_hold_target_out_of_range() {
        let Some((_g, native, cache)) = open() else { return };
        let Some(mut r) = Ride::with_vehicle(native.clone(), cache, "warthog\\warthog") else { return };
        let look = r.input();
        r.tick(&look);
        assert!(r.pm.offer.is_none() && r.pm.action_prompt().is_none(), "the player starts well away from the vehicle");
        for _ in 0..60 {
            let look = r.input();
            r.tick(&look);
            assert_ne!(r.pm.use_decide(true, false, DT), UseDecision::Hold, "nothing to act on");
        }
    }

    /// **THE WEAPON-PER-SEAT GATE.** The biped's own weapon is usable and drawn in a seat whose
    /// `allows weapons` flag is SET and hidden and unusable where it is CLEAR. The data contradict
    /// the report: driver and gunner seats do NOT set it (every one of them, in the whole map);
    /// passenger seats do.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_weapon_is_usable_and_drawn_exactly_in_seats_that_allow_weapons() {
        let Some((_g, native, cache)) = open() else { return };
        // The map-wide fact: no driver seat and no gunner seat allows the biped's weapon.
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let (mut drivers, mut passengers) = (0, 0);
        for t in native.tags_by_group(cache, "vehi") {
            let Some(e) = index.entry(&native, cache, t) else { continue };
            for s in &e.seats.seats {
                if s.is_driver() {
                    drivers += 1;
                    assert!(!s.allows_weapons(), "{} driver seat '{}' allows weapons", e.path, s.label);
                }
                if ["warthog_p", "mongoose_p", "revenant_p", "scorpion_p_lf", "pelican_p_l01", "warthog_p_l", "truck_p"].contains(&s.label.as_str()) && s.flags & crate::reach_seats::FLAG_GUNNER == 0 {
                    passengers += 1;
                    if s.label != "truck_p" { assert!(s.allows_weapons(), "{} passenger seat '{}' does not", e.path, s.label); }
                }
            }
        }
        assert!(drivers > 10 && passengers > 5, "liveness: {drivers} drivers, {passengers} passengers");
        let Some(mut r) = Ride::with_vehicle(native.clone(), cache, "warthog\\warthog") else { return };
        let tag = r.objects[0].primary_tag;
        let hidden = |r: &Ride| r.actors.get(r.pm.body.unwrap()).unwrap().weapon_hidden;
        assert!(r.pm.weapon_usable() && !hidden(&r), "on foot the weapon is usable and drawn");
        for (seat, allows) in [(0usize, false), (1, true)] {
            if r.pm.riding.is_some() {
                // Leave the previous seat the quick way.
                r.pm.riding = None;
            }
            r.force_seat(tag, seat).expect("seat");
            let look = r.input();
            r.run(0.3, &look);
            assert_eq!(r.pm.weapon_usable(), allows, "seat {seat}: usable");
            assert_eq!(hidden(&r), !allows, "seat {seat}: weapon hidden");
        }
    }

    /// **THE SEATED-LOOK GATE.** In every seat the rider's body answers the look: the seat's aim
    /// overlay is installed, and turning the look from hard left to hard right moves the skeleton.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_seated_rider_looks_around_in_every_seat() {
        let Some((_g, native, cache)) = open() else { return };
        let cases: [(&str, usize, Option<&str>); 6] = [
            ("warthog\\warthog", 0, None),
            ("warthog\\warthog", 1, None),
            ("warthog\\warthog", 0, Some("warthog_chaingun")),
            ("mongoose\\mongoose", 0, None),
            ("revenant\\revenant", 1, None),
            ("ghost\\ghost", 0, None),
        ];
        for (suffix, seat, child) in cases {
            let Some(mut r) = Ride::with_vehicle(native.clone(), cache, suffix) else { continue };
            let hull = r.objects[0].primary_tag;
            let tag = match child {
                None => hull,
                Some(c) => native.enumerate_attachments(cache, hull).iter().map(|a| a.child_obj).find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(c))).expect("child"),
            };
            r.force_seat(tag, seat).expect("seat");
            let pose_at = |r: &mut Ride, yaw: f32, pitch: f32| -> Vec<Vec3> {
                let input = MoveInput { yaw, pitch, ..Default::default() };
                r.run(0.6, &input);
                let a = r.actors.get(r.pm.body.unwrap()).unwrap();
                assert!(!a.overlays.is_empty(), "{suffix}#{seat}: no aim/look overlay is installed (aim_missing = {})", r.pm.aim_missing);
                a.pose.model.iter().map(|m| m.w_axis.truncate()).collect()
            };
            let left = pose_at(&mut r, 0.7, 0.0);
            let right = pose_at(&mut r, -0.7, 0.0);
            let up = pose_at(&mut r, 0.0, 0.4);
            let down = pose_at(&mut r, 0.0, -0.4);
            let spread = |a: &[Vec3], b: &[Vec3]| a.iter().zip(b).map(|(x, y)| (*x - *y).length()).fold(0.0f32, f32::max);
            let (yaw_move, pitch_move) = (spread(&left, &right), spread(&up, &down));
            println!("{suffix}#{seat}: look left<->right moves the skeleton {yaw_move:.3} wu, up<->down {pitch_move:.3} wu");
            assert!(yaw_move + pitch_move > 0.01, "{suffix}#{seat}: the rider does not answer the look at all");
        }
    }

    /// **THE 1:1 GATE.** What the player is made of, against the tags: the first-person eye is the
    /// biped's own `standing camera height`, the body is drawn at scale 1 with its origin on the
    /// ground (the feet neither float nor sink), and the head is where a 0.6 eye says it is.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_standing_player_is_one_to_one_with_the_tags() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = tests::Floor;
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let input = MoveInput::default();
        for _ in 0..180 {
            pm.tick(&input, 1.0 / 60.0, &native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(1.0 / 60.0);
        }
        let (eye, _, _) = pm.camera();
        assert!((eye.z - crate::film_play::EYE_HEIGHT_WU).abs() < 1e-3, "eye {} vs the biped's authored {}", eye.z, crate::film_play::EYE_HEIGHT_WU);
        let body = actors.get(pm.body.unwrap()).unwrap();
        let w = body.world();
        assert!((w.x_axis.length() - 1.0).abs() < 1e-4 && (w.z_axis.length() - 1.0).abs() < 1e-4, "the body is drawn at scale 1");
        assert!(w.w_axis.z.abs() < 1e-3, "the body origin is on the ground");
        let z = |n: &str| {
            let i = body.def.model.node_names.iter().position(|x| x == n).unwrap();
            w.transform_point3(body.pose.model[i].w_axis.truncate()).z
        };
        assert!(z("b_l_foot") > 0.0 && z("b_l_foot") < 0.08, "ankle {}: the feet float or sink", z("b_l_foot"));
        assert!((z("b_head") - eye.z).abs() < 0.12, "head node {} vs eye {}", z("b_head"), eye.z);
    }

    /// **THE RENDER GATE** ("posed" is not "on screen"). Through the camera the engine would use,
    /// the rider's HEAD must project inside the frame, in front of the lens, at a believable
    /// distance -- in the chase view the player SEES their driver.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_driver_is_on_screen_in_the_chase_view() {
        let Some((_g, mut r, _)) = boarded() else { return };
        let look = r.input();
        r.run(4.0, &look);
        let (eye, yaw, pitch) = r.pm.camera();
        let body = r.pm.body.and_then(|b| r.actors.get(b)).expect("body");
        let head = body
            .def
            .model
            .node_names
            .iter()
            .position(|n| n == "b_head")
            .expect("the biped has a head node");
        let head_w = body.world().transform_point3(body.pose.model[head].w_axis.truncate());
        let fwd = crate::reach_unit_camera::look_dir(yaw, pitch);
        let to = head_w - eye;
        let depth = to.dot(fwd);
        assert!(depth > 1.0 && depth < 8.0, "the head is {depth:.2} wu in front of the lens (head {head_w:?}, eye {eye:?})");
        // 78 degrees vertical, 16:9.
        let right = fwd.cross(Vec3::Z).normalize();
        let up = right.cross(fwd).normalize();
        let tan_v = (78.0f32.to_radians() * 0.5).tan();
        let (x, y) = (to.dot(right) / depth / (tan_v * 16.0 / 9.0), to.dot(up) / depth / tan_v);
        println!("head at depth {depth:.2}, NDC ({x:+.2},{y:+.2})");
        assert!(x.abs() < 0.9 && y.abs() < 0.9, "the driver's head is off-screen: NDC ({x:.2},{y:.2})");
        // The body is drawn (not parked 10 km below the map), and sits on the vehicle.
        let hull = r.pm.riding.as_ref().unwrap().drive.pos;
        assert!((body.world().w_axis.truncate() - hull).length() < 3.0, "the rider is not on the vehicle: {:?}", body.world().w_axis);
    }
}

/// #walk-bob  The player's vertical signals while walking, through the production path.
#[cfg(test)]
mod walk_bob {
    use super::*;

    /// A tessellated height field: `z = ground(x)`, flat or with a ramp, cut into 1 wu cells whose
    /// diagonals alternate, so a body crossing it meets seams like a real map's triangle soup.
    pub(super) struct Tess {
        /// Ramp slope (rise per run) between x = 10 and x = 40; 0 for a flat floor.
        pub slope: f32,
    }
    impl Tess {
        pub fn ground(&self, x: f32) -> f32 {
            self.slope * (x.clamp(10.0, 40.0) - 10.0)
        }
        fn vtx(&self, x: f32, y: f32) -> Vec3 {
            Vec3::new(x, y, self.ground(x))
        }
    }
    impl crate::objscene::ObjectScene for Tess {
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
        fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> { Some((Vec3::new(500.0, 500.0, 50.0), 0.0, 0.0)) }
        fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> {
            // Only the downward ray the chase camera and wheels cast.
            (dir.z < -0.5 && origin.z > self.ground(origin.x) && origin.z - self.ground(origin.x) <= max)
                .then_some(origin.z - self.ground(origin.x))
        }
        fn raycast_solid_excluding(&self, origin: Vec3, dir: Vec3, max: f32, _x: &[u32]) -> Option<f32> {
            self.raycast_solid(origin, dir, max)
        }
        fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            let (x0, x1) = (mn.x.floor() as i32 - 1, mx.x.ceil() as i32 + 1);
            let (y0, y1) = (mn.y.floor() as i32 - 1, mx.y.ceil() as i32 + 1);
            for i in x0..x1 {
                for j in y0..y1 {
                    let (a, b) = (i as f32, (i + 1) as f32);
                    let (c, d) = (j as f32, (j + 1) as f32);
                    let (p00, p10, p11, p01) = (self.vtx(a, c), self.vtx(b, c), self.vtx(b, d), self.vtx(a, d));
                    if (i + j) & 1 == 0 {
                        out.push([p00, p10, p11]);
                        out.push([p00, p11, p01]);
                    } else {
                        out.push([p00, p10, p01]);
                        out.push([p10, p11, p01]);
                    }
                }
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub(super) struct Row {
        pub t: f32,
        pub dt: f32,
        pub feet: f32,
        pub root: f32,
        pub pelvis: f32,
        pub head: f32,
        pub foot_l: f32,
        pub foot_r: f32,
        pub eye: f32,
        pub ground: f32,
        pub grounded: bool,
        pub clip: String,
        pub frame: f32,
        pub speed: f32,
        pub x: f32,
        pub y: f32,
    }

    pub(super) fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// Walk `script(t) -> (input, dt)` for `secs` and record every frame.
    pub(super) fn walk(
        native: &std::sync::Arc<NativeDll>,
        cache: CacheHandle,
        species: Species,
        slope: f32,
        start_x: f32,
        secs: f32,
        mut script: impl FnMut(f32, usize) -> (MoveInput, f32),
    ) -> Vec<Row> {
        let scene = Tess { slope };
        let mut actors = ActorScene::default();
        let g = scene.ground(start_x);
        let cam = EntryCamera { pos: Vec3::new(start_x, 0.0, g + 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = PlayMode::enter(native, cache, &scene, &[], &mut actors, species, cam).expect("enter");
        for _ in 0..120 {
            pm.tick(&MoveInput::default(), 1.0 / 60.0, native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(1.0 / 60.0);
        }
        let (mut t, mut n) = (0.0f32, 0usize);
        let mut rows = Vec::new();
        while t < secs {
            let (input, dt) = script(t, n);
            pm.tick(&input, dt, native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(dt);
            let body = actors.get(pm.body.unwrap()).unwrap();
            let w = body.world();
            let nodes = &body.def.model.node_names;
            let z = |name: &str| {
                nodes.iter().position(|x| x == name).map_or(f32::NAN, |i| w.transform_point3(body.pose.model[i].w_axis.truncate()).z)
            };
            rows.push(Row {
                t,
                dt,
                feet: pm.st.pos.z,
                root: w.w_axis.z,
                pelvis: z("b_pelvis"),
                head: z("b_head"),
                foot_l: z("b_l_foot"),
                foot_r: z("b_r_foot"),
                eye: pm.camera().0.z,
                ground: scene.ground(pm.st.pos.x),
                grounded: pm.st.grounded,
                clip: body.anim_name.clone(),
                frame: body.frame,
                speed: pm.st.ground_speed,
                x: pm.st.pos.x,
                y: pm.st.pos.y,
            });
            t += dt;
            n += 1;
        }
        rows
    }

    fn report(label: &str, rows: &[Row]) {
        let mut worst: [(f32, usize); 5] = [(0.0, 1); 5];
        let sig: [(&str, fn(&Row) -> f32); 5] = [
            ("feet-ground", |r| r.feet - r.ground),
            ("root-ground", |r| r.root - r.ground),
            ("pelvis-ground", |r| r.pelvis - r.ground),
            ("head-ground", |r| r.head - r.ground),
            ("eye-ground", |r| r.eye - r.ground),
        ];
        for i in 1..rows.len() {
            for (k, (_, f)) in sig.iter().enumerate() {
                let d = (f(&rows[i]) - f(&rows[i - 1])).abs();
                if d > worst[k].0 {
                    worst[k] = (d, i);
                }
            }
        }
        if std::env::var("WALK_TRACE").is_ok() {
            let i = worst[2].1;
            for r in &rows[i.saturating_sub(6)..(i + 6).min(rows.len())] {
                println!("      t={:.3} clip {} @ {:.2} grounded {} pelvis {:+.4} head {:+.4} footL {:+.4} footR {:+.4} feet {:+.4}",
                    r.t, r.clip, r.frame, r.grounded, r.pelvis - r.ground, r.head - r.ground, r.foot_l - r.ground, r.foot_r - r.ground, r.feet - r.ground);
            }
        }
        let flips = rows.windows(2).filter(|w| w[0].grounded != w[1].grounded).count();
        println!("== {label}: {} frames, grounded flips {flips}", rows.len());
        for (k, (name, f)) in sig.iter().enumerate() {
            let (d, i) = worst[k];
            let r = &rows[i];
            println!(
                "   {name:<14} max frame-to-frame {d:.4} wu at t={:.2} (frame dt {:.4}) clip {} @ {:.2} speed {:.2} | prev {:+.4} -> {:+.4}",
                r.t, r.dt, r.clip, r.frame, r.speed, f(&rows[i - 1]), f(r)
            );
        }
    }

    /// Every base clip of the Spartan graph whose pelvis translation is present at frame 0 and
    /// then drops out: how many frames it loses at the end of the clip.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_translation_dropout_survey() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = Tess { slope: 0.0 };
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(2.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let a = actors.get(pm.body.unwrap()).unwrap();
        let g = a.def.graph.as_ref().unwrap();
        let (mut total, mut dropped, mut lost_hist) = (0usize, 0usize, std::collections::BTreeMap::<usize, usize>::new());
        let mut examples = Vec::new();
        for i in 0..g.anims.len() {
            let Some(data) = g.load(&native, cache, i) else { continue };
            let ty = g.anims[i].anim_type;
            let n = data.frame_count.max(1) as usize;
            let present = |f: usize| crate::reach_jmad::sample(&data, crate::reach_jmad::AnimType::Overlay, f as f32).iter().any(|c| c.trans.is_some());
            if !present(0) {
                continue;
            }
            total += 1;
            let first_bad = (0..n).find(|&f| !present(f));
            if let Some(f) = first_bad {
                dropped += 1;
                *lost_hist.entry(n - f).or_default() += 1;
                if examples.len() < 12 {
                    examples.push(format!("{} ({:?}, {n} frames, lost from {f})", g.anims[i].name, ty));
                }
            }
        }
        println!("clips with a translation channel at frame 0: {total}; of those, drop out before the last frame: {dropped}; frames lost -> clips: {lost_hist:?}");
        for e in examples {
            println!("   {e}");
        }
    }

    /// The raw curve record of the pelvis translation channel: segment count, summed durations,
    /// dense flag, against the clip's frame count.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_pelvis_curve_records() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = Tess { slope: 0.0 };
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(2.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let a = actors.get(pm.body.unwrap()).unwrap();
        let g = a.def.graph.as_ref().unwrap();
        for name in ["combat:rifle:move_front_fast", "combat:rifle:move_right", "sprint:rifle:move_front_fast", "combat:rifle:idle", "combat:rifle:walk_front_slow"] {
            let data = g.load(&native, cache, g.find(name).unwrap()).unwrap();
            let blk = data.region(1).unwrap();
            let u32_at = |o: usize| u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]) as usize;
            println!("{name}: frames {} codec id {} header trans offset {:#x}", data.frame_count, blk[0], u32_at(0x0C));
            let p = u32_at(0x14);
            let tab = p + u32_at(0x0C);
            let at = p + u32_at(tab);
            let rec = &blk[at..];
            let n = i16::from_le_bytes([rec[2], rec[3]]) as usize;
            let dur: Vec<u32> = rec[0x18..0x18 + n].iter().map(|&b| b as u32).collect();
            println!("   rec type {:?} n {n} dense {} durations {:?} sum {}", rec[0] as char, rec[4] & 1, dur, dur.iter().sum::<u32>());
            // Where does the record END? The next record in the block, or the block's end.
            let mut starts: Vec<usize> = Vec::new();
            for kind_tab in [0usize, u32_at(0x0C)] {
                let t0 = p + kind_tab;
                let cnt = if kind_tab == 0 { (blk[1] as usize) } else { blk[2] as usize };
                for c in 0..cnt {
                    starts.push(p + u32_at(t0 + 4 * c));
                }
            }
            starts.push(blk.len());
            starts.sort_unstable();
            let end = starts.iter().copied().find(|&x| x > at).unwrap_or(blk.len());
            println!("   record at {at:#x}, next boundary {end:#x}: {} bytes; layout 0x18 + n + 9*(n+1) = {}, with 9*n = {}; block {} bytes", end - at, 0x18 + n + 9 * (n + 1), 0x18 + n + 9 * n, blk.len());
        }
    }

    /// The pelvis height of one clip at every frame, alone (no blend, no overlay).
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_clip_pelvis() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = Tess { slope: 0.0 };
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(2.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let id = pm.body.unwrap();
        for clip in ["combat:rifle:move_front_fast", "combat:rifle:walk_front_slow", "combat:rifle:move_right", "sprint:rifle:move_front_fast", "combat:rifle:idle"] {
            if actors.set_anim_with(&native, cache, id, Some(clip), None).is_err() {
                println!("{clip}: not in graph");
                continue;
            }
            let a = actors.get_mut(id).unwrap();
            a.base_blend.clear();
            a.overlays.clear();
            a.blend_left = 0.0;
            a.prev_anim = None;
            let n = a.frame_count();
            let pi = a.def.model.node_names.iter().position(|x| x == "b_pelvis").unwrap();
            let mut line = format!("{clip} ({n} frames) pelvis z by frame:");
            let mut f = 0.0f32;
            while f < n {
                a.frame = f;
                a.compute_pose();
                line += &format!(" {:.2}:{:.4}", f, a.pose.model[pi].w_axis.z);
                f += 1.0;
            }
            a.frame = n - 0.001;
            a.compute_pose();
            line += &format!(" end:{:.4}", a.pose.model[pi].w_axis.z);
            if clip != "combat:rifle:idle" && clip != "combat:rifle:walk_front_slow" {
                println!("{line}");
            }
            // WHICH CHANNELS DROP OUT: animated-region samples only (Overlay type skips the static region).
            let g = a.def.graph.as_ref().unwrap();
            let data = g.load(&native, cache, g.find(clip).unwrap()).unwrap();
            let mut f = 0usize;
            while (f as f32) < n {
                let ch = crate::reach_jmad::sample(&data, crate::reach_jmad::AnimType::Overlay, f as f32);
                let r = ch.iter().filter(|c| c.rot.is_some()).count();
                let t = ch.iter().filter(|c| c.trans.is_some()).count();
                let pelvis_t = ch.get(pi).and_then(|c| c.trans).is_some();
                println!("   {clip} frame {f}: {r}/{} nodes rot, {t} trans, pelvis(ch {pi}) trans {pelvis_t}", ch.len());
                f += 1;
            }
        }
    }

    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_walk_bob() {
        let Some((_g, native, cache)) = open() else { return };
        let jitter = |n: usize| [1.0 / 60.0, 1.0 / 144.0, 1.0 / 30.0, 1.0 / 75.0, 1.0 / 60.0][n % 5];
        for species in [Species::Spartan] {
            for slope in [0.0f32, 0.35] {
                let sx = if slope > 0.0 { 2.0 } else { 2.0 };
                let cases: Vec<(&str, Box<dyn FnMut(f32, usize) -> (MoveInput, f32)>)> = vec![
                    ("run fwd 60Hz", Box::new(|_, _| (MoveInput { fwd: 1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("walk fwd 60Hz", Box::new(|_, _| (MoveInput { fwd: 1.0, walk: true, ..Default::default() }, 1.0 / 60.0))),
                    ("sprint 60Hz", Box::new(|_, _| (MoveInput { fwd: 1.0, sprint: true, ..Default::default() }, 1.0 / 60.0))),
                    ("crouch walk 60Hz", Box::new(|_, _| (MoveInput { fwd: 1.0, crouch: true, ..Default::default() }, 1.0 / 60.0))),
                    ("strafe right 60Hz", Box::new(|_, _| (MoveInput { right: 1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("backpedal 60Hz", Box::new(|_, _| (MoveInput { fwd: -1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("run fwd jitter", Box::new(move |_, n| (MoveInput { fwd: 1.0, ..Default::default() }, jitter(n)))),
                    ("start/stop 60Hz", Box::new(|t, _| (MoveInput { fwd: if (t as i32) % 2 == 0 { 1.0 } else { 0.0 }, ..Default::default() }, 1.0 / 60.0))),
                    ("run fwd 144Hz", Box::new(|_, _| (MoveInput { fwd: 1.0, ..Default::default() }, 1.0 / 144.0))),
                ];
                for (name, script) in cases {
                    // Backpedal and strafe must stay on the ramp: the ramp runs along +x only.
                    let start = if name.starts_with("backpedal") { 38.0 } else { sx };
                    let rows = walk(&native, cache, species, slope, start, 6.0, script);
                    report(&format!("{species:?} slope {slope} {name}"), &rows);
                }
            }
        }
    }
    /// **GATE 1 (the cause).** No clip of the Spartan's graph may lose its pelvis translation
    /// part-way through: every clip that has the channel at frame 0 has it at EVERY frame.
    /// Liveness first: the survey must actually have walked the graph (931 clips carry the
    /// channel on `forge_halo`); before the fix 322 of them lost it on their last 1-9 frames.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn no_clip_loses_its_pelvis_translation_before_its_last_frame() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = Tess { slope: 0.0 };
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(2.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, Species::Spartan, cam).expect("enter");
        let g = actors.get(pm.body.unwrap()).unwrap().def.graph.as_ref().expect("graph");
        let (mut with_channel, mut broken) = (0usize, Vec::new());
        for i in 0..g.anims.len() {
            let Some(data) = g.load(&native, cache, i) else { continue };
            let n = data.frame_count.max(1) as usize;
            let present = |f: usize| {
                crate::reach_jmad::sample(&data, crate::reach_jmad::AnimType::Overlay, f as f32).iter().any(|c| c.trans.is_some())
            };
            if !present(0) {
                continue;
            }
            with_channel += 1;
            if let Some(f) = (0..n).find(|&f| !present(f)) {
                broken.push(format!("{} lost it from frame {f} of {n}", g.anims[i].name));
            }
        }
        assert!(with_channel > 800, "the survey walked only {with_channel} clips with a translation channel");
        assert!(broken.is_empty(), "{} clips lose the pelvis translation before their last frame, e.g. {:?}", broken.len(), &broken[..broken.len().min(4)]);
    }

    /// **GATE 2 (the symptom).** Walking, running, sprinting, crouch-walking, strafing and
    /// backpedalling, started and stopped, at 60 Hz, 144 Hz and uneven frame times, on a flat
    /// triangulated floor and up a 0.35 ramp, through the real `PlayMode::tick`: the posed pelvis
    /// never moves more than 0.04 wu between two frames (the authored gait's largest step between
    /// adjacent clip frames is 0.0194 wu, the run cycle's; the defect was 0.077 -- 24 cm -- in one
    /// frame, once a stride), the first-person eye moves only at the authored camera-height
    /// velocity (`bipd +0x584`: 0.6 wu/s Spartan, 0.9 Elite), and the body root follows the ground and nothing else.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn walking_never_pops_the_body_or_the_eye() {
        let Some((_g, native, cache)) = open() else { return };
        const PELVIS_STEP: f32 = 0.04;
        let jitter = |n: usize| [1.0 / 60.0, 1.0 / 144.0, 1.0 / 30.0, 1.0 / 75.0, 1.0 / 60.0][n % 5];
        type Script = Box<dyn FnMut(f32, usize) -> (MoveInput, f32)>;
        let mut ran = 0usize;
        for species in [Species::Spartan, Species::Elite] {
            // `bipd +0x584 runtime_camera_height_velocity`, read off the tags: 0.6000 = (0.600-0.420)/0.30
            // for the Spartan and 0.9000 = (0.690-0.510)/0.20 for the Elite.
            let eye_velocity = if species == Species::Spartan { 0.6f32 } else { 0.9 };
            for slope in [0.0f32, 0.35] {
                let cases: Vec<(&str, f32, Script)> = vec![
                    ("run", 2.0, Box::new(|_, _| (MoveInput { fwd: 1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("walk", 2.0, Box::new(|_, _| (MoveInput { fwd: 1.0, walk: true, ..Default::default() }, 1.0 / 60.0))),
                    ("sprint", 2.0, Box::new(|_, _| (MoveInput { fwd: 1.0, sprint: true, ..Default::default() }, 1.0 / 60.0))),
                    ("crouch-walk", 2.0, Box::new(|_, _| (MoveInput { fwd: 1.0, crouch: true, ..Default::default() }, 1.0 / 60.0))),
                    ("strafe", 2.0, Box::new(|_, _| (MoveInput { right: 1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("backpedal", 38.0, Box::new(|_, _| (MoveInput { fwd: -1.0, ..Default::default() }, 1.0 / 60.0))),
                    ("run, uneven frames", 2.0, Box::new(move |_, n| (MoveInput { fwd: 1.0, ..Default::default() }, jitter(n)))),
                    ("start/stop", 2.0, Box::new(|t, _| (MoveInput { fwd: if (t as i32) % 2 == 0 { 1.0 } else { 0.0 }, ..Default::default() }, 1.0 / 60.0))),
                    ("run, 144 Hz", 2.0, Box::new(|_, _| (MoveInput { fwd: 1.0, ..Default::default() }, 1.0 / 144.0))),
                ];
                for (name, start, script) in cases {
                    let rows = walk(&native, cache, species, slope, start, 6.0, script);
                    assert!(rows.len() >= 340, "{species:?} {name}: only {} frames ran", rows.len());
                    let (mut pelvis_max, mut travelled) = (0.0f32, 0.0f32);
                    for w in rows.windows(2) {
                        let (a, b) = (&w[0], &w[1]);
                        let dp = ((b.pelvis - b.ground) - (a.pelvis - a.ground)).abs();
                        pelvis_max = pelvis_max.max(dp);
                        travelled += ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
                        assert!(
                            dp < PELVIS_STEP,
                            "{species:?} slope {slope} {name}: the pelvis moved {dp:.4} wu in one frame at t={:.2} (clip {} @ {:.2}): {:+.4} -> {:+.4}",
                            b.t, b.clip, b.frame, a.pelvis - a.ground, b.pelvis - b.ground
                        );
                        if slope == 0.0 {
                            let de = ((b.eye - b.ground) - (a.eye - a.ground)).abs();
                            assert!(de <= eye_velocity * b.dt * 1.05 + 1e-4, "{species:?} {name}: the eye moved {de:.4} wu in a {:.4} s frame (t={:.2})", b.dt, b.t);
                            assert!((b.feet - a.feet).abs() < 1e-3, "{species:?} {name}: the feet moved {:.5} wu vertically on a flat floor", (b.feet - a.feet).abs());
                        } else {
                            // The ground itself rises at most (fastest authored speed) x slope.
                            let rise = 3.6 * slope * b.dt;
                            assert!((b.feet - a.feet).abs() <= 1.5 * rise + 1e-3, "{species:?} {name}: the feet jumped {:.4} wu on the ramp (t={:.2})", (b.feet - a.feet).abs(), b.t);
                        }
                    }
                    // Liveness: the body went somewhere and the gait really animated the pelvis.
                    if name != "start/stop" {
                        assert!(travelled > 2.5, "{species:?} {name}: travelled only {travelled:.2} wu");
                    }
                    assert!(pelvis_max > 0.001, "{species:?} {name}: the pelvis never moved at all ({pelvis_max}) -- the gait did not animate");
                    ran += 1;
                }
            }
        }
        assert_eq!(ran, 36, "every scenario must run");
    }
}


/// #scale  What the player is made of, measured in the WORLD: the skinned, posed mesh against
/// the numbers the tags fix. Print-only probes plus the gates the measurement earned.
#[cfg(test)]
mod scale_probes {
    use super::ride_harness::*;
    use super::*;
    use glam::Mat4;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// World-space AABB of an actor's SKINNED, POSED mesh (the sections its default variant
    /// draws): the CPU twin of the skinning shader, through `Actor::world()`.
    pub(super) fn posed_world_aabb(actor: &crate::reach_actor::Actor, native: &NativeDll, cache: CacheHandle) -> Option<(Vec3, Vec3)> {
        let pts = posed_world_points(actor, native, cache)?;
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        Some((lo, hi))
    }

    /// Every vertex of the actor's SKINNED, POSED mesh in world space.
    pub(super) fn posed_world_points(actor: &crate::reach_actor::Actor, native: &NativeDll, cache: CacheHandle) -> Option<Vec<Vec3>> {
        let def = &actor.def;
        let mask = crate::reach_actor::default_section_mask(native, cache, def.mode_tag);
        let h = native.open_model(cache, def.mode_tag).ok()?;
        let world = actor.world();
        let mut out: Vec<Vec3> = Vec::new();
        for s in 0..native.model_section_count(h) {
            if mask.get(s as usize) == Some(&false) {
                continue;
            }
            let (Some(sec), Some(g)) = (native.model_section(h, s), native.model_decode_geometry(h, s)) else { continue };
            let stride = (sec.vertex_stride as usize).max(12);
            let map = def.model.slots_for_mesh(s as usize);
            let blend = native.model_decode_blend(h, s);
            // palette[slot] = pose * inverse_bind of the node that slot addresses
            let pal: Vec<Mat4> = map
                .iter()
                .map(|&n| actor.pose.model.get(n as usize).copied().unwrap_or(Mat4::IDENTITY) * def.inverse_bind.get(n as usize).copied().unwrap_or(Mat4::IDENTITY))
                .collect();
            if std::env::var("SCALE_SECTIONS").is_ok() {
                println!("      section {s}: vertices {} skinned-blend {} map {} node_index {}", g.vertices.len() / stride, blend.is_some(), map.len(), sec.node_index);
            }
            let rigid = (sec.node_index >= 0).then(|| {
                let n = sec.node_index as usize;
                actor.pose.model[n] * def.inverse_bind[n]
            });
            for v in 0..g.vertices.len() / stride {
                let o = v * stride;
                let f = |k: usize| f32::from_le_bytes([g.vertices[o + k], g.vertices[o + k + 1], g.vertices[o + k + 2], g.vertices[o + k + 3]]);
                let p = Vec3::new(f(0), f(4), f(8));
                if !p.is_finite() {
                    continue;
                }
                let q = match (&blend, rigid) {
                    (Some(b), _) if !pal.is_empty() && b.len() >= (v + 1) * 8 => {
                        let c = &b[v * 8..v * 8 + 8];
                        crate::reach_skel::skin_point(&pal, [c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]], p)
                    }
                    (_, Some(m)) => m.transform_point3(p),
                    _ => p,
                };
                out.push(world.transform_point3(q));
            }
        }
        native.close_model(h);
        (!out.is_empty()).then_some(out)
    }

    fn marker_world(actor: &crate::reach_actor::Actor, native: &NativeDll, cache: CacheHandle, name: &str) -> Vec<(usize, Vec3)> {
        let mut out = Vec::new();
        for g in &actor.def.model.markers {
            if native.string_id(cache, g.name).as_deref() != Some(name) {
                continue;
            }
            for (i, m) in g.markers.iter().enumerate() {
                if m.node < 0 {
                    continue;
                }
                let local = actor.pose.model[m.node as usize] * Mat4::from_rotation_translation(m.rotation, m.translation);
                out.push((i, actor.world().transform_point3(local.w_axis.truncate())));
            }
        }
        out
    }

    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seated_body_against_the_vehicle() {
        let Some((_g, native, cache)) = open() else { return };
        for species in [Species::Spartan, Species::Elite] {
            for suffix in ["mongoose\\mongoose", "warthog\\warthog", "ghost\\ghost"] {
                for seat in [0usize, 1] {
                    let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, suffix, species) else { continue };
                    let tag = r.objects[0].primary_tag;
                    if r.force_seat(tag, seat).is_err() {
                        continue;
                    }
                    let look = r.input();
                    r.run(3.0, &look);
                    let body = r.actors.get(r.pm.body.unwrap()).unwrap();
                    let Some((blo, bhi)) = posed_world_aabb(body, &native, cache) else { continue };
                    let mut vlo = Vec3::splat(f32::MAX);
                    let mut vhi = Vec3::splat(f32::MIN);
                    for id in r.pm.vehicle_actors.clone() {
                        if let Some((lo, hi)) = r.actors.get(id).and_then(|a| posed_world_aabb(a, &native, cache)) {
                            vlo = vlo.min(lo);
                            vhi = vhi.max(hi);
                        }
                    }
                    let w = body.world();
                    let pelvis = body.def.model.node_names.iter().position(|x| x == "b_pelvis").map(|i| w.transform_point3(body.pose.model[i].w_axis.truncate()));
                    let head = body.def.model.node_names.iter().position(|x| x == "b_head").map(|i| w.transform_point3(body.pose.model[i].w_axis.truncate()));
                    println!(
                        "{species:?} on {suffix} seat {seat}: body z {:+.3}..{:+.3} (h {:.3}) x {:+.3}..{:+.3} | vehicle z {:+.3}..{:+.3} (h {:.3}) x {:+.3}..{:+.3} y {:+.3}..{:+.3} | rider/vehicle height {:.2} | pelvis {:?} head {:?}",
                        blo.z, bhi.z, bhi.z - blo.z, blo.x, bhi.x, vlo.z, vhi.z, vhi.z - vlo.z, vlo.x, vhi.x, vlo.y, vhi.y,
                        (bhi.z - blo.z) / (vhi.z - vlo.z), pelvis, head
                    );
                }
            }
        }
    }

    /// The authored `bounding radius` (obje +0x0C, +offset) against the measured render-model
    /// box of EVERY biped and vehicle: a unit and a vehicle are authored in the same world units,
    /// so if the biped were drawn at the wrong scale its ratio would stand apart from the vehicles'.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_bounding_radius_against_the_render_box() {
        let Some((_g, native, cache)) = open() else { return };
        for group in ["bipd", "vehi"] {
            let mut ratios = Vec::new();
            for tag in native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, tag).unwrap_or_default();
                let Some(rad) = native.tag_meta(cache, tag, 0x0C, 16) else { continue };
                let f = |o: usize| f32::from_le_bytes([rad[o], rad[o + 1], rad[o + 2], rad[o + 3]]);
                let (r, off) = (f(0), Vec3::new(f(4), f(8), f(12)));
                let mode = native.resolve_mode(cache, tag);
                let Some((lo, hi)) = crate::reach_actor::measured_model_bounds(&native, cache, mode) else { continue };
                if r <= 0.0 {
                    continue;
                }
                // the farthest box corner from the authored sphere centre
                let far = (0..8)
                    .map(|i| Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z }))
                    .map(|c| (c - off).length())
                    .fold(0.0f32, f32::max);
                ratios.push((far / r, name, r, hi.z - lo.z));
            }
            ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            println!("== {group}: {} with a radius and a model; farthest-corner / authored radius, sorted", ratios.len());
            for (q, n, r, h) in &ratios {
                if n.contains("spartans\\spartans") || n.contains("elite\\elite") || n.contains("mongoose") || n.contains("warthog\\warthog") || n.contains("ghost\\ghost") || n.contains("scorpion") {
                    println!("   {q:.3}  radius {r:.3}  box height {h:.3}  {n}");
                }
            }
            let med = ratios[ratios.len() / 2].0;
            println!("   median {med:.3}  (min {:.3} max {:.3})", ratios[0].0, ratios[ratios.len() - 1].0);
        }
    }

    /// The owner's own last variant: how many placed objects carry a Forge SCALE label, and what
    /// "Scaled objects" (default ON) makes of them -- the surroundings the Spartan is compared with.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_the_owners_variant_scales() {
        let Some(dir) = crate::project::settings_dir() else { return };
        let Ok(last) = std::fs::read_to_string(dir.join("last_variant.txt")) else { return };
        let Some(v) = crate::mvar::parse_variant(std::path::Path::new(last.trim())) else { println!("variant did not parse"); return };
        println!("variant '{}': {} objects, labels {:?}", v.title, v.objects.len(), v.labels);
        let mut by: std::collections::BTreeMap<String, Vec<i32>> = Default::default();
        for o in &v.objects {
            let l = v.labels.get(o.label_idx as usize).cloned().unwrap_or_default();
            by.entry(l).or_default().push(o.spawn_seq);
        }
        for (l, seqs) in &by {
            let nz = seqs.iter().filter(|&&q| q != 0).count();
            let (mn, mx) = (seqs.iter().min().unwrap(), seqs.iter().max().unwrap());
            println!("   label {l:?}: {} objects, {nz} with spawn_seq != 0 (range {mn}..{mx})", seqs.len());
        }
        for conv in crate::forge_scale::ScaleConvention::ALL {
            let scales: Vec<f32> = v
                .objects
                .iter()
                .filter(|o| v.labels.get(o.label_idx as usize).is_some_and(|l| l.eq_ignore_ascii_case("scale")))
                .map(|o| crate::forge_scale::spawn_seq_to_scale(o.spawn_seq, *conv, crate::forge_scale::team_from_u8(o.team)))
                .collect();
            if scales.is_empty() {
                continue;
            }
            let mut sorted = scales.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("   {:?}: {} SCALE objects, scale min {:.3} median {:.3} max {:.3}", conv, scales.len(), sorted[0], sorted[sorted.len() / 2], sorted[sorted.len() - 1]);
        }
    }

    /// Every camera-height field of the player bipeds, raw: standing/running/crouching/crouch
    /// walking, the per-weapon-class table (`bipd +0x54C`), the root-offset camera scale/dampening
    /// (`+0x56C/+0x570`, which would make the first-person camera follow the animated root).
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_bipd_camera_fields() {
        let Some((_g, native, cache)) = open() else { return };
        for tag in native.tags_by_group(cache, "bipd") {
            let name = native.tag_name(cache, tag).unwrap_or_default();
            if !(name.ends_with("spartans\\spartans") || name.ends_with("elite\\elite")) {
                continue;
            }
            let Some(b) = native.tag_meta(cache, tag, 0x500, 0xB0) else { continue };
            let f = |o: usize| f32::from_le_bytes([b[o - 0x500], b[o - 0x500 + 1], b[o - 0x500 + 2], b[o - 0x500 + 3]]);
            let u = |o: usize| u32::from_le_bytes([b[o - 0x500], b[o - 0x500 + 1], b[o - 0x500 + 2], b[o - 0x500 + 3]]);
            println!("{name}: standing {:.4} running {:.4} crouching {:.4} crouch-walking {:.4} transition {:.3}s | interp start {:.1} end {:.1} deg | fp offset ({:.3},{:.3},{:.3}) | root offset camera scale {:.4} dampening {:.4} | runtime camera velocity {:.4}",
                f(0x524), f(0x528), f(0x52C), f(0x530), f(0x534), f(0x558), f(0x55C), f(0x560), f(0x564), f(0x568), f(0x56C), f(0x570), f(0x584));
            let (n, ptr) = (u(0x54C) as i32, u(0x550));
            println!("   camera heights block: {n} entries");
            if n > 0 && n < 17 {
                if let Some(rows) = native.tag_ptr_bytes(cache, ptr, n as u32 * 12) {
                    for i in 0..n as usize {
                        let r = &rows[i * 12..i * 12 + 12];
                        let sid = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                        let g = |o: usize| f32::from_le_bytes([r[o], r[o + 1], r[o + 2], r[o + 3]]);
                        println!("      class {:?}: standing {:.4} crouching {:.4}", native.string_id(cache, sid), g(4), g(8));
                    }
                }
            }
        }
    }

    /// A standing body of `species`, settled, with its world points and the data around it.
    fn standing(native: &std::sync::Arc<NativeDll>, cache: CacheHandle, species: Species) -> (Vec<Vec3>, PlayMode, ActorScene) {
        let scene = tests::Floor;
        let mut actors = ActorScene::default();
        let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
        let mut pm = PlayMode::enter(native, cache, &scene, &[], &mut actors, species, cam).expect("enter");
        for _ in 0..180 {
            pm.tick(&MoveInput::default(), 1.0 / 60.0, native, cache, &scene, &[], &mut actors);
            actors.update_poses_headless(1.0 / 60.0);
        }
        let pts = posed_world_points(actors.get(pm.body.unwrap()).unwrap(), native, cache).expect("posed points");
        (pts, pm, actors)
    }

    /// **THE 1:1 GATE (posed, skinned, in the world).** The player's drawn body against what the
    /// TAGS fix, for both playable species:
    ///   * the top of the skinned, posed mesh sits at the biped's authored standing capsule height
    ///     (Spartan 0.6552 against 0.650, Elite 0.7943 against 0.790: within 3%);
    ///   * the `head` marker sits within 4 cm of the authored standing camera height;
    ///   * the model's own farthest corner from the authored `bounding radius` sphere is that
    ///     radius -- 0.998 Spartan, 1.000 Elite, and every vehicle with a radius (43 in
    ///     `forge_halo`: 0.996..1.000; the Monitor editor biped 0.979) -- so a biped and a vehicle are in the same units.
    /// Measured 2026-10-21; before this gate "the Spartan is too small" had been checked only
    /// against the bind pose and the actor matrix.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_posed_player_is_one_to_one_with_the_tags() {
        let Some((_g, native, cache)) = open() else { return };
        for species in [Species::Spartan, Species::Elite] {
            let (pts, pm, actors) = standing(&native, cache, species);
            assert!(pts.len() > 10_000, "{species:?}: only {} skinned vertices -- the body was not built", pts.len());
            let ground = pm.st.pos.z;
            let top = pts.iter().map(|p| p.z).fold(f32::MIN, f32::max) - ground;
            let capsule = pm.phys.height_standing;
            assert!((top / capsule - 1.0).abs() < 0.03, "{species:?}: the posed helmet top is {top:.4} wu against an authored capsule of {capsule:.4}");
            let body = actors.get(pm.body.unwrap()).unwrap();
            let heads = marker_world(body, &native, cache, "head");
            assert!(!heads.is_empty(), "{species:?}: no `head` marker");
            let head = heads[0].1.z - ground;
            assert!((head - pm.phys.eye_standing).abs() < 0.04, "{species:?}: head marker {head:.4} vs the authored standing camera height {:.4}", pm.phys.eye_standing);
        }
        // the unit-consistency half: bipeds and vehicles share one world unit
        let mut bipeds = 0usize;
        let mut vehicles: Vec<(f32, String)> = Vec::new();
        for group in ["bipd", "vehi"] {
            for tag in native.tags_by_group(cache, group) {
                let Some(rad) = native.tag_meta(cache, tag, 0x0C, 16) else { continue };
                let f = |o: usize| f32::from_le_bytes([rad[o], rad[o + 1], rad[o + 2], rad[o + 3]]);
                let (r, off) = (f(0), Vec3::new(f(4), f(8), f(12)));
                let Some((lo, hi)) = crate::reach_actor::measured_model_bounds(&native, cache, native.resolve_mode(cache, tag)) else { continue };
                if r <= 0.0 {
                    continue;
                }
                let far = (0..8)
                    .map(|i| Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z }))
                    .map(|c| (c - off).length())
                    .fold(0.0f32, f32::max);
                let name = native.tag_name(cache, tag).unwrap_or_default();
                if group == "bipd" {
                    assert!((far / r - 1.0).abs() < 0.03, "{name}: model corner {far:.4} vs authored bounding radius {r:.4} ({:.3})", far / r);
                    bipeds += 1;
                } else {
                    vehicles.push((far / r, name));
                }
            }
        }
        assert!(bipeds >= 4, "only {bipeds} bipeds carried a radius and a model");
        assert!(vehicles.len() >= 40, "only {} vehicles carried a radius and a model", vehicles.len());
        // Hand-set radii on a few vehicles (the Wraith's reads 1.475) are authoring slack; the
        // MEDIAN says whether the unit is right, and the stock four must each be exact.
        let mut ratios: Vec<f32> = vehicles.iter().map(|v| v.0).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ratios[ratios.len() / 2] - 1.0).abs() < 0.01, "median vehicle ratio {}", ratios[ratios.len() / 2]);
        let mut stock = 0;
        for want in ["warthog\\warthog", "mongoose\\mongoose", "ghost\\ghost", "scorpion\\scorpion"] {
            let (q, n) = vehicles.iter().find(|v| v.1.ends_with(want)).unwrap_or_else(|| panic!("no vehicle {want}"));
            assert!((q - 1.0).abs() < 0.01, "{n}: {q:.3}");
            stock += 1;
        }
        assert_eq!(stock, 4);
    }

    /// **THE SEAT GATE.** A seated driver's hands against the vehicle's own steering markers:
    /// the hand-to-hand span over the marker-to-marker span is 1.02 on the Ghost and 1.11 on the
    /// Mongoose (hands slightly WIDER than the markers). A body drawn 15% small would read 0.87
    /// and 0.94; the bound is 0.95..1.15. Both hands must resolve (liveness).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_seated_spartan_spans_the_steering_markers_one_to_one() {
        let Some((_g, native, cache)) = open() else { return };
        let mut resolved = 0;
        for suffix in ["ghost\\ghost", "mongoose\\mongoose"] {
            let mut r = Ride::with_vehicle_as(native.clone(), cache, suffix, Species::Spartan).expect("vehicle");
            let tag = r.objects[0].primary_tag;
            r.force_seat(tag, 0).expect("seat");
            let look = r.input();
            r.run(3.0, &look);
            r.actors.get_mut(r.pm.body.unwrap()).unwrap().grips.clear();
            r.actors.update_poses_headless(0.0);
            let (n, c) = (r.native.clone(), r.cache);
            let body = r.actors.get(r.pm.body.unwrap()).unwrap();
            let veh = r.actors.get(r.pm.vehicle_actors[0]).unwrap();
            let rd = r.pm.riding.as_ref().unwrap();
            let bw = body.world();
            let hand = |nm: &str| body.def.model.node_names.iter().position(|x| x == nm).map(|i| bw.transform_point3(body.pose.model[i].w_axis.truncate()));
            let mut hands = Vec::new();
            let mut marks = Vec::new();
            for (left, hn) in [(true, "b_l_hand"), (false, "b_r_hand")] {
                let cands = crate::play_vehicle::hand_marker_candidates(&rd.seat_label, left, "spartan");
                let mk = r.pm.vehicles.entry(&n, c, rd.vehi_tag).and_then(|e| cands.iter().find_map(|cn| e.first_marker(&n, c, cn)));
                if let (Some(m), Some(h)) = (mk, hand(hn)) {
                    hands.push(h);
                    marks.push((veh.world() * veh.pose.marker_model(&m)).w_axis.truncate());
                }
            }
            assert_eq!(hands.len(), 2, "{suffix}: both hands and both markers must resolve");
            let ratio = (hands[0] - hands[1]).length() / (marks[0] - marks[1]).length();
            assert!((0.95..1.15).contains(&ratio), "{suffix}: hand span / marker span = {ratio:.3}");
            resolved += 1;
        }
        assert_eq!(resolved, 2);
    }

    /// **THE SCREEN GATE.** Through the camera the app builds at its default field of view (78
    /// degrees horizontal, `reach_vertical_fov_for`), a standing Spartan 3 wu away (seen from the side, so every point is within 0.2 wu of that depth) fills the
    /// fraction of the frame height the authored numbers predict: capsule 0.650 over
    /// `2 * 3 * tan(vfov/2)` = 0.237 (measured 0.2515 on the skinned mesh: it is 0.660 tall, the arms reach 0.2 wu nearer the lens). Taking 78 as the
    /// VERTICAL angle instead would give 0.134, which is the "swapped FOV interpretation" this
    /// gate exists to catch.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_standing_spartan_fills_the_predicted_fraction_of_the_screen() {
        let Some((_g, native, cache)) = open() else { return };
        let (pts, pm, _a) = standing(&native, cache, Species::Spartan);
        let vfov = crate::film_play::reach_vertical_fov_for(crate::film_play::REACH_FIELD_OF_VIEW_DEG);
        assert!((vfov.to_degrees() - 49.097).abs() < 0.05, "the default vertical FOV is {:.3} degrees", vfov.to_degrees());
        let cam = hms_render::Camera { pos: Vec3::new(pm.st.pos.x, pm.st.pos.y - 3.0, pm.st.pos.z + 0.6), yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, fov_y: vfov, near: 0.01, far: 1000.0 };
        let vp = cam.view_proj(16.0 / 9.0);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in &pts {
            let c = vp * p.extend(1.0);
            let y = c.y / c.w;
            lo = lo.min(y);
            hi = hi.max(y);
        }
        let fraction = (hi - lo) * 0.5;
        let predicted = pm.phys.height_standing / (2.0 * 3.0 * (vfov * 0.5).tan());
        assert!((0.97..1.10).contains(&(fraction / predicted)), "the Spartan fills {fraction:.4} of the frame height, the authored numbers predict {predicted:.4}");
        assert!(fraction > 0.2 && fraction < 0.28, "{fraction}");
    }

    fn put(img: &mut [u8], w: usize, h: usize, x: i32, y: i32, c: [u8; 3], r: i32) {
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                    let o = (py as usize * w + px as usize) * 3;
                    img[o..o + 3].copy_from_slice(&c);
                }
            }
        }
    }

    fn line(img: &mut [u8], w: usize, h: usize, a: (i32, i32), b: (i32, i32), c: [u8; 3]) {
        let n = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
        for i in 0..=n {
            put(img, w, h, a.0 + (b.0 - a.0) * i / n, a.1 + (b.1 - a.1) * i / n, c, 0);
        }
    }

    fn save_png(path: &std::path::Path, img: &[u8], w: usize, h: usize) {
        let f = std::fs::File::create(path).expect("png file");
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().expect("png header").write_image_data(img).expect("png data");
    }

    /// Side-by-side outline of the posed, skinned Spartan and Elite beside the Mongoose, the
    /// Warthog and the Ghost with their riders, to ONE scale, with a 1 wu ruler; and the same
    /// bodies projected through the world camera at three fields of view. PNGs for a person.
    #[test]
    #[ignore = "writes PNGs to the scratchpad; run alone with --ignored"]
    fn render_scale_comparison_pngs() {
        let Some((_g, native, cache)) = open() else { return };
        let out = std::path::PathBuf::from("/tmp/claude-1000/-home-sopitive/05268b5e-9d50-4b4a-9c35-93b66f9dc5fa/scratchpad/scale");
        let _ = std::fs::create_dir_all(&out);
        // --- gather world points (all facing +x) for each exhibit --------------------------------
        let mut exhibits: Vec<(String, Vec<(Vec<Vec3>, [u8; 3])>)> = Vec::new();
        {
            let scene = tests::Floor;
            for (species, label, col) in [(Species::Spartan, "Spartan standing", [230u8, 140, 40]), (Species::Elite, "Elite standing", [170, 90, 220])] {
                let mut actors = ActorScene::default();
                let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
                let mut pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, species, cam).expect("enter");
                for _ in 0..180 {
                    pm.tick(&MoveInput::default(), 1.0 / 60.0, &native, cache, &scene, &[], &mut actors);
                    actors.update_poses_headless(1.0 / 60.0);
                }
                let pts = posed_world_points(actors.get(pm.body.unwrap()).unwrap(), &native, cache).expect("points");
                exhibits.push((label.to_string(), vec![(pts, col)]));
            }
        }
        for (suffix, label) in [("mongoose\\mongoose", "Spartan on Mongoose"), ("warthog\\warthog", "Spartan on Warthog"), ("ghost\\ghost", "Spartan on Ghost")] {
            let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, suffix, Species::Spartan) else { continue };
            let tag = r.objects[0].primary_tag;
            if r.force_seat(tag, 0).is_err() {
                continue;
            }
            let look = r.input();
            r.run(3.0, &look);
            let mut parts = Vec::new();
            for id in r.pm.vehicle_actors.clone() {
                if let Some(p) = r.actors.get(id).and_then(|a| posed_world_points(a, &native, cache)) {
                    parts.push((p, [120u8, 130, 140]));
                }
            }
            if let Some(p) = r.actors.get(r.pm.body.unwrap()).and_then(|a| posed_world_points(a, &native, cache)) {
                parts.push((p, [230, 140, 40]));
            }
            exhibits.push((label.to_string(), parts));
        }
        // --- the side view: x across, z up, 600 px per world unit -------------------------------
        let (scale, margin) = (600.0f32, 40.0f32);
        let (w, h) = (2000usize, 760usize);
        let mut img = vec![250u8; w * h * 3];
        let ground_y = (h as f32 - margin - 20.0) as i32;
        line(&mut img, w, h, (0, ground_y), (w as i32, ground_y), [60, 60, 60]);
        // ruler: 1 wu vertical at the left edge, ticks every 0.1 wu, longer every 0.5
        let rx = 30;
        line(&mut img, w, h, (rx, ground_y), (rx, ground_y - scale as i32), [0, 70, 220]);
        for k in 0..=10 {
            let y = ground_y - (k as f32 * 0.1 * scale) as i32;
            line(&mut img, w, h, (rx, y), (rx + if k % 5 == 0 { 28 } else { 12 }, y), [0, 70, 220]);
        }
        let mut cursor = 110.0f32;
        for (_label, parts) in &exhibits {
            let (mut xlo, mut xhi) = (f32::MAX, f32::MIN);
            for (pts, _) in parts {
                for p in pts {
                    xlo = xlo.min(p.x);
                    xhi = xhi.max(p.x);
                }
            }
            for (pts, col) in parts {
                for p in pts {
                    let px = (cursor + (p.x - xlo) * scale) as i32;
                    let py = ground_y - (p.z * scale) as i32;
                    put(&mut img, w, h, px, py, *col, 1);
                }
            }
            cursor += (xhi - xlo) * scale + 70.0;
        }
        save_png(&out.join("side_by_side_ruler_1wu.png"), &img, w, h);
        println!("wrote side_by_side_ruler_1wu.png ({} exhibits: {})", exhibits.len(), exhibits.iter().map(|e| e.0.clone()).collect::<Vec<_>>().join(" | "));
        // --- through the world camera: eye 0.6 high, objects 3 wu ahead ---------------------------
        let (sw, sh) = (640usize, 360usize);
        for (name, deg) in [("authored_78", 78.0f32), ("owner_hms_97_5", 97.5), ("owner_mcc_120", 120.0)] {
            let vfov = crate::film_play::reach_vertical_fov_for(deg);
            let f = 1.0 / (vfov * 0.5).tan();
            let mut im = vec![235u8; sw * sh * 3];
            let horizon = (sh as f32 * 0.5) as i32;
            line(&mut im, sw, sh, (0, horizon), (sw as i32, horizon), [90, 90, 90]);
            let mut xo = -0.55f32;
            for (idx, (_label, parts)) in exhibits.iter().enumerate() {
                if idx != 0 && idx != 2 {
                    continue; // Spartan standing and Spartan on the Mongoose
                }
                // The first exhibit is 0.55 wu to the left, the second to the right, both 3 wu out.
                let (mut xlo, mut xhi, mut ylo, mut yhi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for (pts, _) in parts {
                    for p in pts {
                        xlo = xlo.min(p.x);
                        xhi = xhi.max(p.x);
                        ylo = ylo.min(p.y);
                        yhi = yhi.max(p.y);
                    }
                }
                let (cx, cy) = ((xlo + xhi) * 0.5, (ylo + yhi) * 0.5);
                for (pts, col) in parts {
                    for p in pts {
                        // camera at the origin looking along +x; the exhibit is rotated to face the camera
                        let depth = 3.0 + (cx - p.x);
                        let lateral = xo + (cy - p.y) * 1.0;
                        let up = p.z - 0.6;
                        if depth <= 0.1 {
                            continue;
                        }
                        let ndc_x = f * lateral / depth / (sw as f32 / sh as f32);
                        let ndc_y = f * up / depth;
                        let px = ((ndc_x * 0.5 + 0.5) * sw as f32) as i32;
                        let py = ((0.5 - ndc_y * 0.5) * sh as f32) as i32;
                        put(&mut im, sw, sh, px, py, *col, 0);
                    }
                }
                xo += 1.1;
            }
            save_png(&out.join(format!("screen_3wu_{name}.png")), &im, sw, sh);
            println!("wrote screen_3wu_{name}.png: vertical FOV {:.1} deg", vfov.to_degrees());
        }
    }

    /// Standard Forge pieces against the player: measured render boxes of the sandbox walls,
    /// doors, blocks and crates whose names say what they are.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_forge_pieces_against_the_player() {
        let Some((_g, native, cache)) = open() else { return };
        let mut rows = Vec::new();
        for group in ["scen", "bloc", "crea", "mach", "ctrl"] {
            for tag in native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, tag).unwrap_or_default();
                let l = name.to_lowercase();
                if !(l.contains("forge") || l.contains("sandbox")) || !(l.contains("wall") || l.contains("door") || l.contains("block") || l.contains("crate") || l.contains("bridge") || l.contains("ramp") || l.contains("stair")) {
                    continue;
                }
                let mode = native.resolve_mode(cache, tag);
                if let Some((lo, hi)) = crate::reach_actor::measured_model_bounds(&native, cache, mode) {
                    rows.push((name, hi - lo));
                }
            }
        }
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        println!("{} forge pieces; Spartan standing height 0.655, eye 0.600, capsule 0.650 radius 0.175", rows.len());
        for (n, d) in rows.iter().take(70) {
            println!("   {:.3} x {:.3} x {:.3}  {n}", d.x, d.y, d.z);
        }
    }

    /// The seated hands against the steering markers, as a VECTOR in the vehicle's own frame
    /// (x forward, y left, z up) and the hand-to-hand span against the marker-to-marker span:
    /// a body drawn too small pulls its hands in toward its own centre line by a fixed fraction
    /// of the span, a body offset in the seat moves both hands the same way.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_seated_hands_against_the_markers_as_vectors() {
        let Some((_g, native, cache)) = open() else { return };
        for species in [Species::Spartan, Species::Elite] {
            for suffix in ["mongoose\\mongoose", "warthog\\warthog", "ghost\\ghost"] {
                let Some(mut r) = Ride::with_vehicle_as(native.clone(), cache, suffix, species) else { continue };
                let tag = r.objects[0].primary_tag;
                if r.force_seat(tag, 0).is_err() {
                    continue;
                }
                let look = r.input();
                r.run(3.0, &look);
                r.actors.get_mut(r.pm.body.unwrap()).unwrap().grips.clear();
                r.actors.update_poses_headless(0.0);
                let (n, c) = (r.native.clone(), r.cache);
                let body = r.actors.get(r.pm.body.unwrap()).unwrap();
                let veh = r.actors.get(r.pm.vehicle_actors[0]).unwrap();
                let rd = r.pm.riding.as_ref().unwrap();
                let vinv = veh.world().inverse();
                let species_s = if species == Species::Spartan { "spartan" } else { "elite" };
                let bw = body.world();
                let node = |nm: &str| body.def.model.node_names.iter().position(|x| x == nm).map(|i| vinv.transform_point3(bw.transform_point3(body.pose.model[i].w_axis.truncate())));
                let mut hands = Vec::new();
                let mut marks = Vec::new();
                for (left, hn) in [(true, "b_l_hand"), (false, "b_r_hand")] {
                    let cands = crate::play_vehicle::hand_marker_candidates(&rd.seat_label, left, species_s);
                    let mk = r.pm.vehicles.entry(&n, c, rd.vehi_tag).and_then(|e| cands.iter().find_map(|cn| e.first_marker(&n, c, cn)));
                    if let (Some(m), Some(h)) = (mk, node(hn)) {
                        let mw = vinv.transform_point3((veh.world() * veh.pose.marker_model(&m)).w_axis.truncate());
                        println!("{species:?} {suffix} {hn}: hand {h:.3?} marker {mw:.3?} hand-marker {:.3?}", h - mw);
                        hands.push(h);
                        marks.push(mw);
                    }
                }
                if hands.len() == 2 {
                    println!("   span hand-hand {:.3} marker-marker {:.3} ratio {:.3}", (hands[0] - hands[1]).length(), (marks[0] - marks[1]).length(), (hands[0] - hands[1]).length() / (marks[0] - marks[1]).length());
                }
            }
        }
    }

    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_posed_body_against_the_tags() {
        let Some((_g, native, cache)) = open() else { return };
        let scene = tests::Floor;
        for species in [Species::Spartan, Species::Elite] {
            let mut actors = ActorScene::default();
            let cam = EntryCamera { pos: Vec3::new(0.0, 0.0, 2.0), yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
            let mut pm = PlayMode::enter(&native, cache, &scene, &[], &mut actors, species, cam).expect("enter");
            let p = pm.phys.clone();
            println!("== {species:?}: capsule standing {:.4} crouching {:.4} radius {:.4} | eye standing {:.4} running {:.4} crouching {:.4} crouch-walking {:.4}",
                p.height_standing, p.height_crouching, p.radius, p.eye_standing, p.eye_running, p.eye_crouching, p.eye_crouch_walking);
            for (label, input, secs) in [
                ("standing idle", MoveInput::default(), 3.0f32),
                ("running", MoveInput { fwd: 1.0, ..Default::default() }, 2.0),
                ("crouching", MoveInput { crouch: true, ..Default::default() }, 2.0),
            ] {
                for _ in 0..(secs * 60.0) as usize {
                    pm.tick(&input, 1.0 / 60.0, &native, cache, &scene, &[], &mut actors);
                    actors.update_poses_headless(1.0 / 60.0);
                }
                let a = actors.get(pm.body.unwrap()).unwrap();
                let (lo, hi) = posed_world_aabb(a, &native, cache).expect("posed bounds");
                let ground = pm.st.pos.z;
                let w = a.world();
                let z = |n: &str| {
                    let i = a.def.model.node_names.iter().position(|x| x == n);
                    i.map(|i| w.transform_point3(a.pose.model[i].w_axis.truncate()).z - ground)
                };
                let (eye, _, _) = pm.camera();
                println!(
                    "   {label:<14} posed skinned AABB: height {:.4} (sole {:+.4} .. top {:+.4}) width(y) {:.4} depth(x) {:.4} | pelvis {:?} head node {:?} | camera eye {:.4}",
                    hi.z - lo.z, lo.z - ground, hi.z - ground, hi.y - lo.y, hi.x - lo.x, z("b_pelvis"), z("b_head"), eye.z - ground
                );
                for name in ["head", "eye", "camera", "primary_weapon", "right_hand", "left_hand", "origin"] {
                    let ms = marker_world(a, &native, cache, name);
                    if !ms.is_empty() {
                        println!("      marker '{name}': {}", ms.iter().map(|(i, v)| format!("[{i}] z {:+.4} (x {:+.3}, y {:+.3})", v.z - ground, v.x - pm.st.pos.x, v.y - pm.st.pos.y)).collect::<Vec<_>>().join(", "));
                    }
                }
            }
        }
    }
}

/// #collision  **NEVER TRAPPED IN A BLOCK.** The exit-from-a-vehicle gates, with REAL Forge blocks
/// parked on the exit spots: the seat's entry markers are tried in order, the first spot a
/// standing capsule fits at is taken, and the player is never left inside a block.
///
/// `cargo test --profile quick -p hms-app -- exit_gates --include-ignored --test-threads=1`
#[cfg(test)]
mod exit_gates {
    const DT_G: f32 = 1.0 / 60.0;
    use super::ride_harness::*;
    use super::*;
    use crate::objscene::ObjectScene;

    /// The harness floor plus real block hulls: what a map's objects look like to the player's
    /// collision, with the blocks' shells and triangles as the game scene would serve them.
    struct BlockScene {
        floor: super::tests::Floor,
        shells: HashMap<u32, Vec<Vec<[Vec3; 3]>>>,
    }
    impl ObjectScene for BlockScene {
        fn has_cache(&self) -> bool { true }
        fn invalidate(&mut self) {}
        fn rebuild_pending(&self) -> bool { false }
        fn maybe_rebuild(
            &mut self,
            _o: &[ObjectInfo],
            _c: &HashMap<u32, (u8, u8)>,
            _m: &hms_render::MeshRenderer,
            _d: &eframe::wgpu::Device,
            _q: &eframe::wgpu::Queue,
        ) -> Option<crate::objscene::RebuiltLanes> { None }
        fn set_object_scales(&mut self, _s: HashMap<u32, f32>) {}
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
        fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> { self.floor.spawn_camera_pose() }
        fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> { self.floor.raycast_solid(origin, dir, max) }
        fn raycast_solid_excluding(&self, origin: Vec3, dir: Vec3, max: f32, x: &[u32]) -> Option<f32> {
            self.floor.raycast_solid_excluding(origin, dir, max, x)
        }
        fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) { self.floor.static_tris_in_aabb(mn, mx, out) }
        fn physics_world_tris(&self, o: &ObjectInfo) -> Vec<[Vec3; 3]> {
            self.shells.get(&o.datum).map(|s| s.iter().flatten().copied().collect()).unwrap_or_default()
        }
        fn physics_world_shells(&self, o: &ObjectInfo) -> Vec<Vec<[Vec3; 3]>> {
            self.shells.get(&o.datum).cloned().unwrap_or_default()
        }
    }

    /// Real Forge blocks that are SOLID closed boxes at least 0.9 thick every way (the kind a
    /// vehicle driven into one leaves its exit spots inside) and thin walls / plates.
    fn blocks(native: &NativeDll, cache: CacheHandle) -> Vec<(String, u32, Vec<Vec<[Vec3; 3]>>)> {
        let mut out = Vec::new();
        for &tag in native.tags_by_group(cache, "bloc").iter() {
            let name = native.tag_name(cache, tag).unwrap_or_default();
            if !name.contains("levels\\forge\\ff_") {
                continue;
            }
            let ph = native.resolve_phmo(cache, tag);
            if ph == 0 || ph == 0xFFFF_FFFF {
                continue;
            }
            let Some(blk) = crate::object_hulls::read_blocks(native, cache, ph) else { continue };
            let Some(sh) = crate::object_hulls::hull_shells(&blk, &crate::object_hulls::mode_node_xforms(native, cache, tag)) else { continue };
            if crate::reach_collision::collides_with_biped(native, cache, tag) && sh.iter().flatten().count() >= 12 {
                out.push((name, tag, sh));
            }
        }
        out
    }

    fn extent(sh: &[Vec<[Vec3; 3]>]) -> (Vec3, Vec3) {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in sh.iter().flatten().flatten() {
            mn = mn.min(*p);
            mx = mx.max(*p);
        }
        (mn, mx)
    }

    /// A seated Warthog at the origin and the exit candidates it offers.
    fn seated() -> Option<(std::sync::MutexGuard<'static, ()>, Ride, Vec<Vec3>)> {
        let (g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let mut r = Ride::with_vehicle(native, cache, "warthog\\warthog")?;
        let tag = r.objects[0].primary_tag;
        r.force_seat(tag, 0).ok()?;
        let (n, c) = (r.native.clone(), r.cache);
        let riding = r.pm.riding.clone()?;
        let cands = crate::play_vehicle::exit_candidates(&mut r.pm.vehicles, &n, c, &riding);
        Some((g, r, cands))
    }

    /// Step out of the seat with `blocks` parked in the world and report the result.
    /// Returns `(chosen position, cands[0] was inside a block, final position after 2 s)`.
    fn exit_with(r: &mut Ride, scene: &BlockScene, extra: &[ObjectInfo]) -> (Vec3, Vec3) {
        let base = r.objects[0].clone();
        r.objects = std::iter::once(base).chain(extra.iter().cloned()).collect();
        r.pm.world.invalidate();
        let still = MoveInput { yaw: 0.0, ..Default::default() };
        let (n, c) = (r.native.clone(), r.cache);
        r.pm.exit_vehicle(&n, c);
        let mut chosen = None;
        for _ in 0..30 {
            r.pm.tick(&still, DT, &n, c, scene, &r.objects, &mut r.actors);
            if r.pm.riding.is_none() {
                chosen = Some(r.pm.st.pos);
                break;
            }
        }
        let chosen = chosen.expect("the exit never finished");
        for _ in 0..120 {
            r.pm.tick(&still, DT, &n, c, scene, &r.objects, &mut r.actors);
        }
        (chosen, r.pm.st.pos)
    }

    fn placement(datum: u32, tag: u32, pos: Vec3) -> ObjectInfo {
        ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.to_array(),
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

    /// **GETTING OUT NEVER LEAVES THE PLAYER IN A BLOCK.** The Warthog's driver seat, with a real
    /// Forge block parked so that the FIRST exit spot (the seat's entry marker pushed out by its
    /// radius) is inside it -- every solid block, at several offsets -- and again with blocks on
    /// the first AND second spot (the near door and the far door both against a wall). Each exit
    /// must land where a standing capsule fits (inside no block, into no surface), must stay
    /// there for two seconds, and must not be sent somewhere absurd.
    ///
    /// BEFORE: the single first spot as the old `exit_position` gave it. SABOTAGE: take the first
    /// candidate without testing it in `choose_exit` -> red.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn getting_out_next_to_real_blocks_never_ends_inside_one() {
        let Some((_g, mut r, cands)) = seated() else { return };
        assert!(cands.len() >= 2, "the Warthog offers {} exit spots", cands.len());
        let (native, cache) = (r.native.clone(), r.cache);
        let blocks = blocks(&native, cache);
        let phys = r.pm.phys.clone();
        let (mut cases, mut old_inside, mut new_inside, mut moved_far, mut tight) = (0, 0, 0, 0, 0);
        let mut report: Vec<String> = Vec::new();
        for (name, tag, sh) in &blocks {
            let (mn, mx) = extent(sh);
            let size = mx - mn;
            if size.min_element() < 0.05 {
                continue;
            }
            let centre = (mn + mx) * 0.5;
            for off in [Vec3::ZERO, Vec3::new(0.4, 0.0, 0.0), Vec3::new(0.0, 0.4, 0.0), Vec3::new(-0.3, -0.3, 0.2)].iter() {
                for both in [false, true] {
                    // restore a seated rider for each case
                    let mut spots = vec![cands[0]];
                    if both {
                        spots.push(cands[1]);
                    }
                    let mut shells = HashMap::new();
                    let mut extra = Vec::new();
                    for (i, s) in spots.iter().enumerate() {
                        // the block's centre on the spot (z so that the spot is inside it)
                        let at = Vec3::new(s.x, s.y, s.z) + *off - Vec3::new(centre.x, centre.y, 0.0) + Vec3::new(0.0, 0.0, -mn.z * 0.0);
                        let datum = 0x7000 + i as u32;
                        shells.insert(datum, sh.iter().map(|shell| shell.iter().map(|t| t.map(|p| p + at)).collect()).collect());
                        extra.push(placement(datum, *tag, at));
                    }
                    let scene = BlockScene { floor: super::tests::Floor, shells };
                    // was the first spot really inside one?
                    let probe = {
                        let w = &mut r.pm.world;
                        w.invalidate();
                        w.regather(cands[0], &scene, &[r.objects[0].clone()].iter().chain(extra.iter()).cloned().collect::<Vec<_>>(), &native, cache);
                        PlayerState::overlaps_at(Vec3::new(cands[0].x, cands[0].y, 0.0), &phys, &r.pm.world, 0.02)
                    };
                    if !probe {
                        continue; // the block missed the capsule at that offset: not a trapped exit
                    }
                    cases += 1;
                    tight += both as usize;
                    old_inside += 1; // the old code took cands[0] snapped to z 0, which `probe` just measured as overlapping
                    // fresh ride for the case
                    let (chosen, end) = exit_with(&mut r, &scene, &extra);
                    let w = &r.pm.world;
                    let inside = PlayerState::overlaps_at(chosen, &phys, w, 0.03) || PlayerState::overlaps_at(end, &phys, w, 0.03);
                    new_inside += inside as usize;
                    // a block the vehicle is buried in can only be left over its top: allow its own size
                    let far = chosen.distance(cands[0]) > 4.0 + size.max_element();
                    moved_far += far as usize;
                    if (inside || far) && report.len() < 10 {
                        report.push(format!("{name} off {off:?} both {both}: chosen {chosen:?} end {end:?} (first spot {:?})", cands[0]));
                    }
                    // back in the seat for the next case
                    let tag0 = r.objects[0].primary_tag;
                    r.pm.parked.clear();
                    r.pm.props = Default::default();
                    r.force_seat(tag0, 0).unwrap();
                }
            }
        }
        println!("{} blocks, {cases} exits with a block on the first spot ({tight} with the second as well): old rule left the player overlapping in {old_inside}, new rule {new_inside}; {moved_far} sent further than 4 wu plus the block's own size", blocks.len());
        for l in &report {
            println!("   {l}");
        }
        assert!(cases >= 100, "liveness: only {cases} trapped exits tested");
        assert_eq!((new_inside, moved_far), (0, 0));
    }

    /// **ENTERING TEST MODE INSIDE A BLOCK.** The editor camera is flown into a real Forge block
    /// (the middle of it, a corner region, near a face) and Play is pressed: the body is dropped
    /// from the camera and must not start inside the block. Also the failsafe respawn after a
    /// block is placed on the spawn point.
    ///
    /// SABOTAGE: drop the `settle_out` call in `PlayMode::enter` -> red.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn entering_test_mode_with_the_camera_inside_a_block_starts_outside_it() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let blocks = blocks(&native, cache);
        let (mut cases, mut inside_before, mut inside_after) = (0, 0, 0);
        let mut report = Vec::new();
        for (name, tag, sh) in blocks.iter().step_by(2) {
            let (mn, mx) = extent(sh);
            if (mx - mn).min_element() < 0.2 {
                continue;
            }
            for f in [Vec3::splat(0.5), Vec3::new(0.15, 0.15, 0.5), Vec3::new(0.85, 0.5, 0.3)] {
                let at = Vec3::new(1000.0, 1000.0, 0.0) - Vec3::new(mn.x, mn.y, 0.0); // away from the origin
                let datum = 0x7100;
                let shells: Vec<Vec<[Vec3; 3]>> = sh.iter().map(|s| s.iter().map(|t| t.map(|p| p + at)).collect()).collect();
                let scene = BlockScene { floor: super::tests::Floor, shells: HashMap::from([(datum, shells)]) };
                let objects = vec![placement(datum, *tag, at)];
                let eye = mn + (mx - mn) * f + at;
                let cam = EntryCamera { pos: eye, yaw: 0.0, pitch: 0.0, fov_y: 1.2 };
                let mut actors = ActorScene::default();
                let Ok(pm) = PlayMode::enter(&native, cache, &scene, &objects, &mut actors, Species::Spartan, cam) else { continue };
                // where the body would have started with no resolution at all
                let raw = Vec3::new(eye.x, eye.y, eye.z - pm.phys.eye_standing);
                let was_inside = PlayerState::overlaps_at(raw, &pm.phys, &pm.world, 0.02);
                if !was_inside {
                    continue;
                }
                cases += 1;
                inside_before += 1;
                // right after `enter`, before any tick could push the body out
                let immediate = PlayerState::overlaps_at(pm.st.pos, &pm.phys, &pm.world, 0.03);
                let mut pm = pm;
                for _ in 0..60 {
                    pm.tick(&MoveInput::default(), DT_G, &native, cache, &scene, &objects, &mut actors);
                }
                let bad = immediate || PlayerState::overlaps_at(pm.st.pos, &pm.phys, &pm.world, 0.03);
                inside_after += bad as usize;
                if bad && report.len() < 8 {
                    report.push(format!("{name} cam {eye:?}: body at {:?}", pm.st.pos));
                }
            }
        }
        println!("{cases} camera-inside-a-block entries: body would start overlapping in {inside_before}, after the fix {inside_after}");
        for l in &report {
            println!("   {l}");
        }
        assert!(cases >= 60, "liveness: only {cases} entries inside a block");
        assert_eq!(inside_after, 0);
    }

    /// **A MOVED VEHICLE LEAVES NO INVISIBLE BLOCK BEHIND.** A placement the ride has hidden (the
    /// vehicle was driven away) must not stay in the player's collision at its old spot.
    ///
    /// SABOTAGE: ignore `skip` in `PlayWorld::regather` -> the second count stays at the first.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_hidden_placement_is_not_collision() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let blocks = blocks(&native, cache);
        let (_, tag, sh) = blocks.iter().find(|(n, _, _)| n.ends_with("ff_plat_2x2_flat")).expect("a Forge platform");
        let datum = 0x7200;
        let scene = BlockScene { floor: super::tests::Floor, shells: HashMap::from([(datum, sh.clone())]) };
        let objects = vec![placement(datum, *tag, Vec3::ZERO)];
        let mut w = PlayWorld::default();
        w.regather(Vec3::ZERO, &scene, &objects, &native, cache);
        let shown = w.object_tris;
        w.set_skip(vec![datum]);
        assert!(!w.covers(Vec3::ZERO), "changing the skip set must make the cached neighbourhood stale");
        w.regather(Vec3::ZERO, &scene, &objects, &native, cache);
        println!("object triangles around the placement: {shown} while shown, {} once hidden", w.object_tris);
        assert!(shown >= 12 && w.object_tris == 0);
        let mut shells = Vec::new();
        w.solid_shells(Vec3::splat(-5.0), Vec3::splat(5.0), &mut shells);
        assert!(shells.is_empty(), "a hidden placement still offers its shells");
    }
}
