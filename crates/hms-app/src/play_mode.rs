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
}

impl StaticWorld for PlayWorld {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        StaticWorld::gather(self.tris.as_slice(), mn, mx, out);
    }
}

impl PlayWorld {
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
            if (p - pos).length() > near {
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
    }
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
    /// #seat-entry  The seat camera this frame, `(eye, yaw, pitch)`. `None` on foot.
    seat_cam: Option<(Vec3, f32, f32)>,
    /// #seat-entry  The seat's authored `override fov` in radians, when it authors one. `App`
    /// owns the field of view, so this is reported rather than applied here.
    seat_fov: Option<f32>,
    /// #seat-entry  The seat the player is in, or `None` — and `None` leaves every existing
    /// path byte-identical: the controller steps, the camera is the biped's, and nothing in
    /// this module behaves differently from before vehicles existed.
    pub riding: Option<crate::play_vehicle::Riding>,
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
        notes.push(format!(
            "step-up {:.4} wu (emergent from the authored radius {:.3} and slope limit {:.1} deg — \
             Reach authors no step height), ground probe {:.4} wu (= the steepest legal descent in one engine tick)",
            play_ctl::max_step_height(&phys),
            phys.radius,
            phys.max_slope.to_degrees(),
            play_ctl::ground_probe_depth(&phys)
        ));

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
        let st = {
            let mut st = PlayerState::standing_at(feet, yaw);
            st.pitch = pitch;
            st
        };
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
            seat_cam: None,
            seat_fov: None,
            riding: None,
            offer: None,
            pickup: None,
            // #pickup  The map's weapon tags, read once here. The same list the weapon choice
            // above was made from, so the search cannot disagree with the menu about what a
            // weapon is.
            map_weapons: weapons.clone(),
            vehicles: Default::default(),
            action_key_label: "E".to_string(),
            asked: None,
            gait_table: None,
            last_stance: crate::film_anim::Stance::Combat,
            appearance_of: Default::default(),
            gait_caption: String::new(),
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
            frames: 0,
        };
        // --- the body -----------------------------------------------------------------------
        let id = actors.add_object(native, cache, biped_tag, feet, yaw.to_degrees())?;
        pm.body = Some(id);
        if pm.weapon_tag != 0 {
            if let Err(e) = actors.set_weapon_tag(native, cache, id, Some(pm.weapon_tag)) {
                pm.notes.push(format!("the weapon did not go into the hands: {e}"));
            }
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
    pub fn fire(&mut self, scene: &dyn ObjectScene, damage: Option<&mut (dyn ObjectDamage + 'static)>) {
        if self.weapon_tag == 0 {
            self.last_hit = Some("no weapon in hand".into());
            return;
        }
        if self.fired.is_some() {
            return;
        }
        self.fired = Some(0.0);
        self.shots += 1;
        let eye = self.st.eye(&self.phys);
        let dir = look_dir(self.st.yaw, self.st.pitch);
        // The shot's range: far enough to cross any Forge structure. A projectile's real flight
        // (gravity, its own speed, its detonation) is `reach_projectile`'s law and is not
        // reimplemented here — this is the aim ray, which is what makes the hit point honest.
        const SHOT_RANGE: f32 = 200.0;
        let hit = scene.pick(eye, dir).and_then(|datum| {
            let d = scene.raycast_solid(eye, dir, SHOT_RANGE)?;
            Some((datum, eye + dir * d, d))
        });
        self.last_hit = Some(match (hit, damage) {
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
        });
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

    /// #seat-entry  Datums whose PLACEMENT must not be drawn this frame.
    pub fn hidden_datums(&self) -> Vec<u32> {
        self.riding.iter().map(|r| r.hidden_datum).collect()
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
        let Some(placement) = objects.iter().find(|p| p.datum == want) else {
            return Err(format!("the placement {want:#x} is no longer in the scene"));
        };
        let r = crate::play_vehicle::begin(&mut self.vehicles, native, cache, &o, placement);
        self.riding_variant_sid = placement.variant_name_sid;
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

    /// #seat-entry  Get out. The player is put where they would have stood to get IN — the entry
    /// marker's own position pushed out along its own forward by the seat's own `entry radius` —
    /// and carries the vehicle's velocity, which is what Reach does.
    ///
    /// When the markers cannot be resolved the player is left where the seat was rather than
    /// teleported somewhere invented, and the reason is returned.
    pub fn exit_vehicle(&mut self, native: &NativeDll, cache: CacheHandle) -> String {
        let Some(r) = self.riding.take() else { return "not in a vehicle".into() };
        self.seat_cam = None;
        let carry = crate::play_vehicle::exit_velocity(&r);
        let out = crate::play_vehicle::exit_position(&mut self.vehicles, native, cache, &r);
        let msg = match out {
            Some(p) => {
                self.st = play_ctl::PlayerState::standing_at(p, r.drive.yaw);
                self.st.vel = carry;
                format!("left {} seat {}", r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path), r.seat)
            }
            None => {
                self.st.vel = carry;
                format!(
                    "left {} seat {} — its entry marker could not be resolved, so you are where the seat was",
                    r.vehi_path.rsplit(['\\', '/']).next().unwrap_or(&r.vehi_path),
                    r.seat
                )
            }
        };
        self.notes.push(msg.clone());
        msg
    }

    /// #seat-entry  Keep the actors that DRAW the ridden vehicle in step with the ride.
    ///
    /// **Showing and hiding are the same `Option`**: the placement is filtered out of the render
    /// set by [`hidden_datums`](Self::hidden_datums) and drawn here, both from `self.riding`. So
    /// a ride cannot show two vehicles (the placement is gone the frame the actors appear) and
    /// ending one cannot show none (the actors go the frame the placement comes back).
    fn sync_vehicle_actors(&mut self, native: &NativeDll, cache: CacheHandle, actors: &mut ActorScene) {
        let Some(r) = self.riding.clone() else {
            if !self.vehicle_actors.is_empty() {
                let ids = std::mem::take(&mut self.vehicle_actors);
                actors.actors.retain(|a| !ids.contains(&a.id));
                actors.rebuild = true;
            }
            return;
        };
        // The placement's own model VARIANT decides which of a Warthog's four weapon
        // attachments is drawn, so it has to be passed in rather than defaulted.
        let variant = self.riding_variant_sid;
        let pieces = crate::play_vehicle::draw_pieces(&mut self.vehicles, native, cache, &r, variant);
        // Added once, then only moved. `add_render_model` reads a render model, so doing it per
        // frame would decode the Warthog's meshes sixty times a second.
        if self.vehicle_actors.len() != pieces.len() {
            let ids = std::mem::take(&mut self.vehicle_actors);
            actors.actors.retain(|a| !ids.contains(&a.id));
            for p in &pieces {
                match actors.add_render_model(native, cache, p.mode_tag, None, Vec3::ZERO, 0.0) {
                    Ok(id) => self.vehicle_actors.push(id),
                    Err(why) => self.notes.push(format!(
                        "the ridden vehicle's model {:#x} could not be drawn: {why}",
                        p.mode_tag
                    )),
                }
            }
            actors.rebuild = true;
        }
        // The pose is a full MATRIX, not a position and a yaw: an attachment carries its own
        // authored rotation, and a turret mounted at an angle would be wrong placed by yaw alone.
        for (id, p) in self.vehicle_actors.iter().zip(pieces.iter()) {
            if let Some(a) = actors.get_mut(*id) {
                a.world_override = Some(p.world);
                a.playing = false;
            }
        }
        // #vehicle-anim  ...and then the vehicle's OWN driven animation: the wheels rolling at
        // `distance / radius` and the steering at the authored lock. `play_vehicle` measures all
        // of it off the tags and classifies each overlay as a wrapping CYCLE or a monotone RANGE;
        // this is the one call that makes any of it reach the screen.
        let notes = crate::play_vehicle::drive_actors(
            &mut self.vehicles,
            native,
            cache,
            &r,
            &pieces,
            &self.vehicle_actors,
            actors,
        );
        self.notes.extend(notes);
    }

    /// #seat-entry  One frame of a RIDE: drive the vehicle from the stick, put the body on the
    /// seat marker, and solve the seat camera.
    fn drive_and_seat(
        &mut self,
        input: &MoveInput,
        dt: f32,
        native: &NativeDll,
        cache: CacheHandle,
        scene: &dyn ObjectScene,
    ) {
        let Some(r) = self.riding.as_mut() else { return };
        // The rider's look is the mouse's, clamped by the seat's own authored range inside
        // `seat_camera`; stored unclamped here so that releasing the look does not creep.
        r.yaw = input.yaw;
        r.pitch = input.pitch;
        // Only the DRIVER drives. A gunner's or passenger's stick moves nothing, which is what
        // the authored `driver` flag means.
        let driving = self
            .vehicles
            .entry(native, cache, r.vehi_tag)
            .and_then(|e| e.seats.seats.get(r.seat).map(|s| s.is_driver()))
            .unwrap_or(false);
        let di = if driving {
            crate::reach_vehicle::DriveInput {
                throttle: input.fwd,
                slide: -input.right,
                want_yaw: input.yaw,
                brake: input.crouch,
            }
        } else {
            crate::reach_vehicle::DriveInput { want_yaw: r.drive.yaw, ..Default::default() }
        };
        let mut r2 = r.clone();
        crate::play_vehicle::step(&mut self.vehicles, native, cache, &mut r2, di, dt, scene);
        *self.riding.as_mut().expect("still riding") = r2;
        let r = self.riding.clone().expect("still riding");
        // The body goes on the seat marker, and the controller is never asked.
        if let Some((feet, yaw)) = crate::play_vehicle::rider_pose(&mut self.vehicles, native, cache, &r) {
            self.st.set_pose_from_vehicle(feet, yaw, r.pitch, r.drive.vel);
        }
        // The seat's own camera.
        // `fov_default` is 0.0 because the FIELD OF VIEW is `App`'s to own (it sets
        // `camera.fov_y` from the user's own setting). A seat that authors an `override fov`
        // reports it through [`seat_fov`] and `App` applies it; one writer either way.
        //
        // Measured: **every seat in `forge_halo` authors `override fov` 0.0**, so this is a path
        // the shipped tags never take. It is implemented because the field exists and because a
        // modded vehicle could use it, not because anything in the game does.
        let view = crate::play_vehicle::seat_camera(&mut self.vehicles, native, cache, &r, 0.0);
        self.seat_fov = view.and_then(|v| (v.fov > 0.0).then_some(v.fov));
        self.seat_cam = view
            .map(|v| {
                let eye = if v.third_person { v.eye - v.forward * THIRD_PERSON_BACK + Vec3::Z * THIRD_PERSON_UP } else { v.eye };
                let yaw = v.forward.y.atan2(v.forward.x);
                let pitch = v.forward.z.asin();
                (eye, yaw, pitch)
            });
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
        // #seat-entry  THE BYPASS. A seated body's transform is the VEHICLE's transform times
        // the seat marker, so the character controller is taken out of the loop entirely rather
        // than taught about vehicles — one writer on the body's position, which is the rule this
        // project keeps relearning. `StepOut::default()` is what the caption then reports, and it
        // is honest: no capsule swept, no contacts, no ground probe.
        if self.riding.is_some() {
            self.drive_and_seat(input, dt, native, cache, scene);
            self.last_step = StepOut::default();
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
            self.offer = crate::play_vehicle::offer(
                &mut self.vehicles,
                native,
                cache,
                objects,
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
        self.pose(input, native, cache, actors);
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
            let cands = if seated_no_clip { Vec::new() } else { crate::film_anim::anim_candidates(&state, &class) };
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
        let fp_body_drawn = self.fp_body.is_some() && self.view == PlayView::FirstPerson;
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
                // #play-jolt  Clock-driven, at the authored 30 fps on the WALL CLOCK — and played
                // FORWARD THEN BACKWARD rather than wrapped, because Reach's idle clips are not
                // loops. Measured on the shipped graph: the Spartan's `combat:rifle:idle` is still
                // to within 0.0006 wu for 155 of its 156 frames and then its last frame sits
                // 0.0426 wu from its first (73x an ordinary frame of motion), and the crouch
                // idle's is 0.183 wu out (477x). Wrapping that is a pop once a cycle, reported as
                // "the idle animation seems to play through then skip/shift the biped". See
                // `film_anim::ping_pong_frame` for the whole measurement and for why this is HMS's
                // choice rather than the engine's.
                crate::film_anim::ping_pong_frame(self.clock, n)
            };
            a.frame = frame.min(n - 0.001).max(0.0);
            a.pitch = 0.0;
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
        if self.riding.is_none() {
            let (dy, dp) = self.aim_offset();
            let cands = crate::film_anim::aim_candidates(&state, &class);
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
            let show = self.view == PlayView::FirstPerson;
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
        // #seat-entry  A rider looks through the SEAT's camera, solved in `drive_and_seat` where
        // the vehicle's pose and the cache are both available. Pure in, pure out: the view is
        // computed from the vehicle transform and the seat's own `unit_camera_struct`, so it
        // cannot go stale.
        if let Some((eye, yaw, pitch)) = self.seat_cam {
            return (eye, yaw, pitch);
        }
        let eye = self.st.eye(&self.phys);
        match self.view {
            PlayView::FirstPerson => (eye, self.st.yaw, self.st.pitch),
            // Solved in `tick`, where the scene is available to stand it off the walls.
            PlayView::ThirdPerson => (self.third_person_cam, self.st.yaw, self.st.pitch),
        }
    }

    /// Take every actor test mode created out of the scene. Leaving must not leave a Spartan
    /// standing in the editor.
    pub fn leave(&mut self, actors: &mut ActorScene) {
        // #seat-entry  Including the ridden vehicle's own actors: leaving test mode while sitting
        // in a Warthog must not leave a second Warthog in the editor beside the real one.
        let mut ids: Vec<u32> = self.vehicle_actors.drain(..).collect();
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
            let rate = pm
                .phys
                .gait_speed(s.stance, s.gait)
                .and_then(|v| crate::film_anim::cycles_per_wu_at_speed(v, n));
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
