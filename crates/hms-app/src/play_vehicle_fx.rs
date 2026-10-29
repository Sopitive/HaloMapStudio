//! #vehicle-fx  A driven vehicle's LIGHTS and EFFECTS: the brake lights, the headlights, the
//! Ghost's anti-gravity glow, the boost jets -- everything a `vehi` hangs on itself as an
//! `obje` ATTACHMENT and switches with an object function.
//!
//! Reported: *"Add the lights when braking, power slide basically; vehicles have headlights /
//! taillights that illuminate on certain conditions. The Ghost has some effects under it, same as
//! the Revenant and Wraith. Some of these effects and lights have different intensities based on
//! what the user is doing."* Before this module a ridden vehicle drew NONE of them: test mode drew
//! the vehicle as an actor built from the tag, and the persistent-attachment filters the world
//! lanes use (`scale_sid == 0`) skip every attachment that names a function, which is all of
//! them.
//!
//! # How the engine does it (PROVEN, `reach_tag_test.exe`; the game's own `haloreach.dll` carries
//! the same code stripped of its asserts)
//!
//! ```text
//! obje +0x104 attachments (32 B): type (ligh / effe / lens / ltvl / lsnd), marker, flags@+0x15,
//!                                 primary scale@+0x18, secondary scale@+0x1C   (object-function NAMES)
//! obje +0xEC  functions   (64 B): import (a built-in INPUT) -> mapping curve -> export, with
//!                                 `turn off with`, `scale by`, `invert`, `always active`
//! ```
//!
//! * `sub_140D37320` (the attachment update) asks `sub_140D47160(object, primary scale)` every
//!   tick. A **light** exists while that function is ACTIVE; one light is made **per marker** of the
//!   named marker group (up to 8), none if the model has no such marker. An **effect** exists
//!   while it is active and is destroyed (or let finish) when it is not.
//! * `sub_140D61E60` (the light update) sets the light's radiance to
//!   `colour_fn(t) * intensity_fn(t) * primary_scale_value`: **the brightness is the function's
//!   VALUE**, not just its on/off. So a taillight is `2.0 * brakelights(jump) * driver_seat_power`.
//! * An effect's particle systems read the function values as their input states 13 and 14
//!   (`c_particle_state_list::update_states_internal`: `sub_140509250(system, 0/1)`), i.e. `effect
//!   A scale` / `effect B scale` = primary / secondary scale. With no function named they are the
//!   creation defaults, 1.0 (`sub_14036CC90`).
//!
//! # The built-in inputs a vehicle supplies
//!
//! | input | engine | here |
//! |---|---|---|
//! | `driver_seat_power` | unit `+0x3A0` (a 0..1 power level), only if the unit authors `powered seats` (`unit +0x420`, driver powerup / powerdown seconds) | ramps to 1 while the driver seat is occupied, back to 0 after (ramp SHAPE inferred: linear) |
//! | `driver_seat_occupied` | the unit's driver slot is set (`sub_140D6E730`) | the player is in the driver seat |
//! | `jump` (jeeps) | `sub_140E69630` case `0x1F6`: reverse-against-travel brake window (`sub_140E60B00`) OR the hand brake (control bit 15, `sub_140E69C30`) | `drive::brake_active(brake_ticks)` OR Ctrl held |
//! | `engine_power`, `engine_rpm` | jeep: flywheel / max * shift ramp, `* power`; others: spring-smoothed (`sub_140E83C10`, unported) | jeep PROVEN, others the sound module's STAND-IN |
//! | `abs_throttle`, `in_reverse` | `sub_140E61950`: stick magnitude * power, signed by forward | exact |
//! | `hover`, `anti_gravity_strength`, `mean_antigrav` | hover families: engaged anti-gravity points / count, capped by power, slewed 0.1 per tick (`sub_140E62060`) | same formula (the `max(0.4, vehicle +0x64)` cap taken as 1) |
//! | `flashlights_allowed` | game state byte set ONLY by the script call `game_can_use_flashlights` (`sub_140D5CE80`); never scripted = 0 after the first second | 0 (so the Wraith/pickup/truck night lights stay off in a map that never scripts them) |
//! | `boost` | the unit's boost level, or the boost control bit | 0: HMS has no boost control (stated, not guessed) |
//! | `one` / `zero` | constants | exact |
//!
//! Everything else a vehicle names (`thrust`, `wingtip_contrail`, `flying`, ...) resolves to
//! nothing and is REPORTED in [`FxState::unresolved`]; the attachment stays off rather than being
//! lit by a guess.
//!
//! # What is NOT done (stated, with why)
//!
//! * `ltvl` light volumes (the visible headlight cone) are not drawn on a driven vehicle: they are
//!   built as a mesh once per object rebuild and have no per-frame lane. The `lens` flares (the
//!   glow AT a lit lamp), the lights and the effects are. A lens flare's brightness is taken as
//!   the primary scale function's value (INFERRED: the engine's lens attachment update is not in
//!   the function decompiled here).
//! * The lamp MESHES' own `self_illum_intensity` (the Warthog's `warthog_brakelights` shader is
//!   driven by `brakelights`, its console by `driver_seat_power`) is baked at rest by the object
//!   pipeline and does not follow the drive state: the glow comes from the lens flare and the
//!   light instead.
//! * The tire/skid/grinding material effects: the engine's route is identified (see the memory
//!   note `vehicle-lights-and-effects`) but not ported -- it needs the physics-model sphere
//!   contacts and the surface material under the tyre, neither of which test mode has.
//! * A frustum light's direction is the marker's `+X` (INFERRED, as for every other marker-aimed
//!   thing in this project); the engine's packer takes it from the light datum's own frame.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

use eframe::wgpu;
use glam::{Mat4, Vec3};
use hms_native::{CacheHandle, NativeDll};

use crate::play_ctl::MoveInput;
use crate::play_sound::{Inputs as EngineInputs, ObjFn, FN_FLAG_ALWAYS_ACTIVE, FN_FLAG_INVERT};
use crate::play_vehicle::{Riding, VehicleIndex};
use crate::reach_actor::tag_block;
use crate::reach_vehicle::drive;
use crate::reach_vehicle::Family;
use crate::scene::{DynLight, LiftLight, ParticleSource, SceneController};

// ---------------------------------------------------------------------------------------------
// tag layout
// ---------------------------------------------------------------------------------------------

/// `obje +0x104` `attachments` (`global_object_attachment_block`, 32 B, max 24).
const OBJE_ATTACHMENTS: usize = 0x104;
const ATT_SIZE: usize = 32;
const ATT_MAX: i32 = 24;
/// The `type` tag reference: class fourcc (big-endian in the cache) at +0, datum at +0xC.
const ATT_CLASS: usize = 0x00;
const ATT_DATUM: usize = 0x0C;
const ATT_MARKER: usize = 0x10;
/// `flags` byte; bit 0 makes the attachment UNCONDITIONAL (`sub_140D37320` skips the function).
const ATT_FLAGS: usize = 0x15;
const ATT_FLAG_ALWAYS_ON: u8 = 1;
const ATT_PRIMARY: usize = 0x18;
const ATT_SECONDARY: usize = 0x1C;
/// `unit +0x420` `powered seats` (8 B: driver powerup seconds, driver powerdown seconds; max 2).
pub const UNIT_POWERED_SEATS: usize = 0x420;

/// The game-state `flashlights_allowed` value of a map whose scripts never call
/// `game_can_use_flashlights`. See the module table.
pub const FLASHLIGHTS_ALLOWED: f32 = 0.0;

/// The engine's hover slew: `sub_140E62060` moves `hover` by at most 0.1 per 60 Hz tick.
const HOVER_SLEW_PER_SECOND: f32 = 0.1 * 60.0;

fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(b, o))
}
fn sid_name(native: &NativeDll, cache: CacheHandle, sid: u32) -> String {
    if sid == 0 || sid == u32::MAX {
        return String::new();
    }
    native.string_id(cache, sid).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------------------------

/// The built-in object-function inputs of ONE vehicle for one tick.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FxInputs {
    /// `driver_seat_power`.
    pub driver_power: f32,
    /// `driver_seat_occupied`.
    pub occupied: bool,
    /// `jump`: the jeeps' brake (reverse-against-travel window or hand brake).
    pub brake: bool,
    /// `engine_power` / `engine_rpm` / `abs_throttle`, as the sound module builds them.
    pub engine: EngineInputs,
    /// `in_reverse`: the stick is pulling back.
    pub reverse: bool,
    /// `hover` / `anti_gravity_strength` / `mean_antigrav`.
    pub hover: f32,
}

impl FxInputs {
    /// A built-in input by name, or `None` for one this module does not supply.
    pub fn builtin(&self, name: &str) -> Option<f32> {
        Some(match name {
            "one" => 1.0,
            "zero" => 0.0,
            "driver_seat_power" => self.driver_power,
            "driver_seat_occupied" => f32::from(self.occupied),
            "jump" => f32::from(self.brake),
            "engine_power" => self.engine.engine_power,
            "engine_rpm" => self.engine.engine_rpm,
            "abs_throttle" => self.engine.abs_throttle,
            "in_reverse" => f32::from(self.reverse),
            "hover" | "anti_gravity_strength" | "mean_antigrav" => self.hover,
            "flashlights_allowed" => FLASHLIGHTS_ALLOWED,
            // No boost control exists in test mode: a stated 0, not a guess.
            "boost" => 0.0,
            // An undamaged, living object (`sub_140D3FE80`: `alive` = 1 until the dead flag).
            "alive" | "body_vitality" => 1.0,
            _ => return None,
        })
    }
}

/// A function's value and whether the engine would call it active.
type Value = (f32, bool);

fn value_of(fns: &[ObjFn], name: &str, inp: &FxInputs, depth: u8, unresolved: &mut BTreeSet<String>) -> Option<Value> {
    if name.is_empty() {
        return None;
    }
    if depth < 8 {
        if let Some(f) = fns.iter().find(|f| f.export == name) {
            return eval_fn(fns, f, inp, depth + 1, unresolved);
        }
    }
    match inp.builtin(name) {
        // A built-in input is active when it is above zero (`*a5 = v > 0`).
        Some(v) => Some((v, v > 0.0)),
        None => {
            unresolved.insert(name.to_string());
            None
        }
    }
}

/// One entry of the object's function block, evaluated the way `sub_140D47160` does: `turn off
/// with` first, then import -> mapping -> `scale by` -> `invert`, active when above zero (or
/// flagged always-active) and above `min value`. The same rules the sound module applies to the
/// same block.
fn eval_fn(fns: &[ObjFn], f: &ObjFn, inp: &FxInputs, depth: u8, unresolved: &mut BTreeSet<String>) -> Option<Value> {
    if !f.off_with.is_empty() {
        let (_, on) = value_of(fns, &f.off_with, inp, depth, unresolved)?;
        if !on {
            return Some((0.0, false));
        }
    }
    let (x, _) = value_of(fns, &f.import, inp, depth, unresolved)?;
    let mut v = match &f.map {
        Some(m) => m.eval(x),
        None => x,
    };
    if !f.scale_by.is_empty() {
        v *= value_of(fns, &f.scale_by, inp, depth, unresolved)?.0;
    }
    if f.flags & FN_FLAG_INVERT != 0 {
        v = 1.0 - v;
    }
    let always = f.flags & FN_FLAG_ALWAYS_ACTIVE != 0;
    let mut active = always || v > 0.0;
    if f.min_value > 0.0 && v <= f.min_value && !always {
        active = false;
    }
    Some((v, active))
}

/// An attachment's scale function: `(value, active)`. A name of "" is unconditional (1.0), the
/// engine's `function_name == NONE`; the always-on flag skips the function (`sub_140D37320`).
pub fn attachment_state(fns: &[ObjFn], flags: u8, name: &str, inp: &FxInputs, unresolved: &mut BTreeSet<String>) -> Option<Value> {
    if flags & ATT_FLAG_ALWAYS_ON != 0 || name.is_empty() {
        return Some((1.0, true));
    }
    value_of(fns, name, inp, 0, unresolved)
}

// ---------------------------------------------------------------------------------------------
// a vehicle's attachments, resolved once per tag
// ---------------------------------------------------------------------------------------------

/// A frustum light's cone (`sub_14082B6B0`): `cosCut = cos(fov/2)`, `ratio = 1 / (cos(hotspot/2) -
/// cosCut)`, `power = angular falloff speed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    pub cos_cut: f32,
    pub ratio: f32,
    pub power: f32,
}

/// One `ligh` attachment.
#[derive(Clone, Debug)]
pub struct LightAtt {
    pub index: usize,
    pub flags: u8,
    pub primary: String,
    /// The decoded light, model space; `radiance_at` evaluates its colour and intensity functions.
    pub base: Arc<LiftLight>,
    pub cone: Option<Cone>,
    /// EVERY marker of the named group, model space (the engine makes one light per marker).
    pub frames: Vec<Mat4>,
}

/// One `effe` attachment.
#[derive(Clone, Debug)]
pub struct EffectAtt {
    pub index: usize,
    pub effe: u32,
    pub marker_sid: u32,
    pub flags: u8,
    pub primary: String,
    pub secondary: String,
}

/// One `lens` attachment.
#[derive(Clone, Debug)]
pub struct LensAtt {
    pub index: usize,
    pub lens: u32,
    pub flags: u8,
    pub primary: String,
    pub frames: Vec<Mat4>,
}

/// What a vehicle tag hangs on itself that this module drives, plus what it does not.
#[derive(Clone, Debug, Default)]
pub struct FxDef {
    pub obj_tag: u32,
    /// The render model whose markers the effects name.
    pub mode: u32,
    pub fns: Vec<ObjFn>,
    pub lights: Vec<LightAtt>,
    pub effects: Vec<EffectAtt>,
    pub lenses: Vec<LensAtt>,
    /// `unit +0x420 powered seats[0]`: driver powerup / powerdown seconds. `None` = the unit
    /// authors none, so `driver_seat_power` is permanently 0.
    pub powered: Option<(f32, f32)>,
}

/// A marker group lists a marker once per (region, permutation) that carries it, so a Warthog's
/// tail-light marker appears once per weapon variant at the SAME place. The engine makes one light
/// per marker the VARIANT shows; this keeps one per distinct frame, which is that for every
/// chassis light (a marker that exists only on one variant would show on all -- a stated limit).
fn distinct_frames(frames: Vec<Mat4>) -> Vec<Mat4> {
    let mut out: Vec<Mat4> = Vec::new();
    for f in frames {
        if !out.iter().any(|g| (g.w_axis - f.w_axis).length() < 1e-3 && (g.x_axis - f.x_axis).length() < 1e-3) {
            out.push(f);
        }
    }
    out
}

fn read_light(native: &NativeDll, cache: CacheHandle, ligh: u32, marker_sid: u32) -> Option<(LiftLight, Option<Cone>)> {
    // `light_struct_definition` (204 B): +0 method, +2 flags, +4 type (0 sphere, 1 frustum), +8
    // attenuation end, +0xC start, +0x1C fov, +0x20 hotspot, +0x24 falloff speed, colour function
    // data size/address +0x48/+0x54, intensity function +0x6C/+0x78. The same reads `scene.rs`
    // makes for a Forge light.
    let m = native.tag_meta(cache, ligh, 0, 0x80)?;
    let (end, start) = (rd_f32(&m, 0x08), rd_f32(&m, 0x0C));
    let blob = |size_at: usize, addr_at: usize| {
        let (size, addr) = (rd_u32(&m, size_at), rd_u32(&m, addr_at));
        if size < 0x0C || size > 0x400 || addr == 0 { Vec::new() } else { native.tag_ptr_bytes(cache, addr, size).unwrap_or_default() }
    };
    let radius = if end.is_finite() && end > 0.0 { end } else { 5.0 };
    let atten_start = if start.is_finite() && start >= 0.0 && start < radius { start } else { 0.0 };
    let kind = u16::from_le_bytes([m[4], m[5]]);
    let cone = (kind != 0).then(|| {
        let cos_cut = (rd_f32(&m, 0x1C) * 0.5).cos();
        let d = (rd_f32(&m, 0x20) * 0.5).cos() - cos_cut;
        Cone { cos_cut, ratio: if d > 1e-4 { 1.0 / d } else { 1.0 }, power: rd_f32(&m, 0x24) }
    });
    Some((
        LiftLight {
            pos: Vec3::ZERO,
            color: [1.0; 3],
            intensity: 1.0,
            radius,
            atten_start,
            method: m[0],
            flags: u16::from_le_bytes([m[2], m[3]]),
            color_blob: blob(0x48, 0x54),
            intensity_blob: blob(0x6C, 0x78),
            light_tag: ligh,
            marker_sid,
            scale_sid: 0,
        },
        cone,
    ))
}

impl FxDef {
    pub fn resolve(vehicles: &mut VehicleIndex, native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> FxDef {
        let mut def = FxDef { obj_tag, mode: native.resolve_mode(cache, obj_tag), fns: ObjFn::read_all(native, cache, obj_tag), ..Default::default() };
        let powered = tag_block(native, cache, obj_tag, UNIT_POWERED_SEATS, 8, 2);
        if powered.len() >= 8 {
            def.powered = Some((rd_f32(&powered, 0), rd_f32(&powered, 4)));
        }
        let atts = tag_block(native, cache, obj_tag, OBJE_ATTACHMENTS, ATT_SIZE, ATT_MAX);
        for (index, e) in atts.chunks_exact(ATT_SIZE).enumerate() {
            let datum = rd_u32(e, ATT_DATUM);
            if datum == 0 || datum == u32::MAX {
                continue;
            }
            let tag = datum & 0xFFFF;
            let class = rd_u32(e, ATT_CLASS).to_be_bytes();
            let marker_sid = rd_u32(e, ATT_MARKER);
            let marker = sid_name(native, cache, marker_sid);
            let flags = e[ATT_FLAGS];
            let primary = sid_name(native, cache, rd_u32(e, ATT_PRIMARY));
            let secondary = sid_name(native, cache, rd_u32(e, ATT_SECONDARY));
            match &class {
                b"ligh" => {
                    let Some((light, cone)) = read_light(native, cache, tag, marker_sid) else { continue };
                    let frames = distinct_frames(vehicles.entry(native, cache, obj_tag).map(|v| v.marker_frames(native, cache, &marker)).unwrap_or_default());
                    def.lights.push(LightAtt { index, flags, primary, base: Arc::new(light), cone, frames });
                }
                b"effe" => def.effects.push(EffectAtt { index, effe: tag, marker_sid, flags, primary, secondary }),
                b"lens" => {
                    let frames = distinct_frames(vehicles.entry(native, cache, obj_tag).map(|v| v.marker_frames(native, cache, &marker)).unwrap_or_default());
                    def.lenses.push(LensAtt { index, lens: tag, flags, primary, frames });
                }
                // sounds are the sound module's; `ltvl` light volumes are not drawn (module docs)
                _ => {}
            }
        }
        def
    }
}

// ---------------------------------------------------------------------------------------------
// one frame
// ---------------------------------------------------------------------------------------------

/// A light to put in the world this frame.
#[derive(Clone, Debug)]
pub struct FxLight {
    pub pos: Vec3,
    /// The direction the light shines, world.
    pub dir: Vec3,
    pub cone: Option<Cone>,
    pub light: Arc<LiftLight>,
    /// The primary scale function's VALUE: the light's radiance is multiplied by it.
    pub scale: f32,
}

/// A lens flare to show this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FxLens {
    pub att: usize,
    pub lens: u32,
    pub pos: Vec3,
    pub fwd: Vec3,
    /// The primary scale function's value: the flare's brightness.
    pub value: f32,
}

/// An effect to play this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FxEffect {
    pub att: usize,
    pub effe: u32,
    pub marker_sid: u32,
    /// The particle states `effect A scale` / `effect B scale`.
    pub a: f32,
    pub b: f32,
}

/// Everything one vehicle puts out this frame.
#[derive(Clone, Debug)]
pub struct VehicleFx {
    pub datum: u32,
    pub mode: u32,
    pub world: Mat4,
    pub inputs: FxInputs,
    pub lights: Vec<FxLight>,
    pub lenses: Vec<FxLens>,
    pub effects: Vec<FxEffect>,
}

#[derive(Clone, Debug, Default)]
pub struct FxFrame {
    pub vehicles: Vec<VehicleFx>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Power {
    value: f32,
    hover: f32,
}

/// Test mode's per-session state for all of this.
#[derive(Default)]
pub struct FxState {
    pub(crate) defs: HashMap<u32, Arc<FxDef>>,
    power: HashMap<u32, Power>,
    pub frame: FxFrame,
    /// `vehicle: input` for every function input a vehicle names that this module cannot supply.
    pub unresolved: BTreeSet<String>,
    trails: HashMap<u32, VecDeque<(f32, Vec3)>>,
    /// Decoded `lens` flares, once each (`None` = the tag did not decode).
    lens_cache: HashMap<u32, Option<LensSprite>>,
}

type LensSprite = (Arc<hms_render::LensFlareDef>, Arc<(wgpu::TextureView, wgpu::Texture)>);

/// Move `v` toward `target` at `rate` per second; a zero or negative authored time is instant.
fn approach(v: f32, target: f32, time: f32, dt: f32) -> f32 {
    if !(time > 1e-4) {
        return target;
    }
    let step = dt / time;
    if target > v { (v + step).min(target) } else { (v - step).max(target) }
}

impl FxState {
    pub fn def(&mut self, vehicles: &mut VehicleIndex, native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Arc<FxDef> {
        self.defs.entry(obj_tag).or_insert_with(|| Arc::new(FxDef::resolve(vehicles, native, cache, obj_tag))).clone()
    }

    /// Advance every vehicle's lights and effects by `dt` and rebuild [`FxState::frame`].
    ///
    /// `riding` is the ride in progress; `parked` the vehicles left behind (they power DOWN for
    /// the authored seconds, which is why they are still here). The driver's stick and hand brake
    /// are `input`, honoured only while the ride is SEATED in a driver seat.
    pub fn step(
        &mut self,
        dt: f32,
        input: &MoveInput,
        riding: Option<&Riding>,
        parked: &[&Riding],
        vehicles: &mut VehicleIndex,
        native: &NativeDll,
        cache: CacheHandle,
    ) {
        let mut frame = FxFrame::default();
        if let Some(r) = riding {
            let driver = vehicles
                .entry(native, cache, r.vehi_tag)
                .and_then(|e| e.seats.seats.get(r.seat))
                .is_some_and(|s| s.is_driver())
                && r.vehi_tag == r.placement_tag;
            let driving = r.controls_live(driver);
            let v = self.vehicle(dt, r, driver, driving.then_some(input), vehicles, native, cache);
            frame.vehicles.push(v);
        }
        for r in parked {
            let v = self.vehicle(dt, r, false, None, vehicles, native, cache);
            frame.vehicles.push(v);
        }
        // A vehicle with no power left and nobody in it has nothing to keep state for.
        let live: Vec<u32> = frame.vehicles.iter().map(|v| v.datum).collect();
        self.power.retain(|d, p| live.contains(d) && (p.value > 0.0 || p.hover > 0.0 || riding.is_some_and(|r| r.hidden_datum == *d)));
        self.trails.retain(|d, _| live.contains(d));
        frame.vehicles.retain(|v| !v.lights.is_empty() || !v.lenses.is_empty() || !v.effects.is_empty() || v.inputs.driver_power > 0.0);
        self.frame = frame;
    }

    #[allow(clippy::too_many_arguments)]
    fn vehicle(
        &mut self,
        dt: f32,
        r: &Riding,
        occupied: bool,
        stick: Option<&MoveInput>,
        vehicles: &mut VehicleIndex,
        native: &NativeDll,
        cache: CacheHandle,
    ) -> VehicleFx {
        let def = self.def(vehicles, native, cache, r.placement_tag);
        let datum = r.hidden_datum;
        // The seat power ramp (unit +0x3A0): up while the driver is in, down after.
        let (up, down) = def.powered.unwrap_or((0.0, 0.0));
        let pw = self.power.entry(datum).or_default();
        pw.value = if def.powered.is_some() { approach(pw.value, f32::from(occupied), if occupied { up } else { down }, dt) } else { 0.0 };
        let power = pw.value;

        // abs_throttle / in_reverse: the stick magnitude, signed by forward, times the power
        // (`sub_140E61950`).
        let (fwd, side) = stick.map_or((0.0, 0.0), |i| (i.fwd, i.right));
        let mag = (fwd * fwd + side * side).sqrt().min(1.0);
        let throttle = if fwd < 0.0 { -mag } else { mag } * power;
        let handbrake = stick.is_some_and(|i| i.crouch);

        let (engine, brake, hover_target) = match vehicles.entry(native, cache, r.vehi_tag).and_then(|e| e.phys.as_ref()).map(|p| &p.family) {
            Some(Family::Jeep { engine, .. }) => {
                let max = engine.max_angular_velocity;
                let omega = if max > 1e-3 { r.drive.engine.angular_velocity / max } else { 0.0 };
                let mut e = EngineInputs::jeep(omega, r.drive.engine.idle_ticks, throttle.abs());
                if power <= 0.0 {
                    // `engine_power` (case 0x21D) is 0 unless `unit +0x3A0 > 0`.
                    e.engine_power = 0.0;
                }
                (e, drive::brake_active(r.drive.brake_ticks, false) || handbrake, 0.0)
            }
            Some(f) => {
                let top = match f {
                    Family::Tank { limits, .. } | Family::Revenant { limits, .. } => limits.as_ref().map_or(0.0, |l| l.top_speed_ceiling),
                    _ => 0.0,
                }
                .max(f.velocity_model().map_or(0.0, |m| m.max_forward_speed));
                let frac = if top > 1e-3 { r.drive.vel.length() / top } else { 0.0 };
                let mut e = EngineInputs::stand_in(frac, throttle.abs());
                if power <= 0.0 {
                    e.engine_power = 0.0;
                }
                // Hover families: engaged anti-gravity points / count, capped by the power.
                let n = usize::from(r.drive.contact_count).min(r.drive.touching.len());
                let hover = if matches!(f, Family::Scout { .. } | Family::Revenant { .. }) && n > 0 {
                    r.drive.touching[..n].iter().filter(|t| **t).count() as f32 / n as f32
                } else {
                    0.0
                };
                (e, false, hover.min(power))
            }
            None => (EngineInputs::default(), false, 0.0),
        };
        let hover = approach(pw.hover, hover_target, 1.0 / HOVER_SLEW_PER_SECOND, dt);
        pw.hover = hover;
        let inputs = FxInputs { driver_power: power, occupied, brake, engine, reverse: throttle < 0.0, hover };
        let who = native.tag_name(cache, r.placement_tag).unwrap_or_default();
        self.evaluate(&def, datum, r.world(), inputs, &who)
    }

    /// Every attachment of `def` against `inputs`: the part test mode and Theater share.
    pub fn evaluate(&mut self, def: &FxDef, datum: u32, world: Mat4, inputs: FxInputs, who: &str) -> VehicleFx {
        let mut unres = BTreeSet::new();
        let mut lights = Vec::new();
        for l in &def.lights {
            let Some((value, active)) = attachment_state(&def.fns, l.flags, &l.primary, &inputs, &mut unres) else { continue };
            if !active {
                continue;
            }
            for f in &l.frames {
                let pos = world.transform_point3(f.w_axis.truncate());
                let dir = world.transform_vector3(f.transform_vector3(Vec3::X)).normalize_or_zero();
                if pos.is_finite() {
                    lights.push(FxLight { pos, dir, cone: l.cone, light: l.base.clone(), scale: value.max(0.0) });
                }
            }
        }
        let mut lenses = Vec::new();
        for l in &def.lenses {
            let Some((value, active)) = attachment_state(&def.fns, l.flags, &l.primary, &inputs, &mut unres) else { continue };
            if !active {
                continue;
            }
            for f in &l.frames {
                let pos = world.transform_point3(f.w_axis.truncate());
                let fwd = world.transform_vector3(f.transform_vector3(Vec3::X)).normalize_or_zero();
                if pos.is_finite() {
                    lenses.push(FxLens { att: l.index, lens: l.lens, pos, fwd, value: value.max(0.0) });
                }
            }
        }
        let mut effects = Vec::new();
        for e in &def.effects {
            let Some((a, active)) = attachment_state(&def.fns, e.flags, &e.primary, &inputs, &mut unres) else { continue };
            if !active {
                continue;
            }
            let b = if e.secondary.is_empty() { Some((1.0, true)) } else { attachment_state(&def.fns, 0, &e.secondary, &inputs, &mut unres) };
            effects.push(FxEffect { att: e.index, effe: e.effe, marker_sid: e.marker_sid, a, b: b.map_or(0.0, |v| v.0) });
        }
        if !unres.is_empty() {
            let who = who.rsplit(['\\', '/']).next().unwrap_or(who).to_string();
            self.unresolved.extend(unres.into_iter().map(|n| format!("{who}: {n}")));
        }
        VehicleFx { datum, mode: def.mode, world, inputs, lights, lenses, effects }
    }

    /// Hand this frame to the scene: the lights to its dynamic-light lane, the effects to its
    /// vehicle particle lane. `t` is the scene clock the particle lane runs on.
    ///
    /// Strict about markers, like every other effect lane here: a system that names a marker the
    /// model lacks creates nothing, it is not moved to the origin (`hms-fx-marker-fallback`).
    pub fn install(&mut self, sc: &mut SceneController, t: f32) {
        let mut lights = Vec::new();
        let mut sources = Vec::new();
        for v in &self.frame.vehicles {
            for l in &v.lights {
                lights.push(DynLight {
                    pos: l.pos,
                    dir: l.dir,
                    cone: l.cone.map(|c| [c.cos_cut, c.ratio, c.power]),
                    light: l.light.clone(),
                    scale: l.scale,
                });
            }
            // The source's own motion: a particle is placed where the vehicle WAS when it was
            // born (`ParticleSource::path`), or the trail would be dragged along behind it.
            let trail = self.trails.entry(v.datum).or_default();
            trail.push_back((t, v.world.w_axis.truncate()));
            while trail.front().is_some_and(|s| t - s.0 > TRAIL_SECONDS) {
                trail.pop_front();
            }
            let path = Arc::new(trail.iter().copied().collect::<Vec<_>>());
            for e in &v.effects {
                let systems = sc.effect_particle_systems(e.effe);
                // An attached effect lives as long as its attachment is active whether or not the
                // `effe` itself loops (the Ghost's anti-gravity glow is a 0.1 s event with no loop
                // start): this lane's sources are pre-rolled looping ones (`birth: None`).
                if systems.is_empty() {
                    continue;
                }
                let (mut kept, mut locs) = (Vec::new(), Vec::new());
                for sy in systems.iter() {
                    if sy.camera_mode == 1 {
                        continue; // first-person-only
                    }
                    let sid = if sy.loc_sid != 0 { sy.loc_sid } else { e.marker_sid };
                    let loc = if sid == 0 || sc.string_of(sid) == "root" {
                        Some((Vec3::ZERO, glam::Quat::IDENTITY))
                    } else {
                        sc.model_marker(v.mode, sid)
                    };
                    if let Some(loc) = loc {
                        kept.push(sy.clone());
                        locs.push(loc);
                    }
                }
                if kept.is_empty() {
                    continue;
                }
                sources.push((ParticleSource {
                    seed: VEHICLE_FX_SEED ^ ((v.datum as u64) << 20) ^ ((e.att as u64) << 8) ^ e.effe as u64,
                    world: v.world,
                    effe: e.effe,
                    label: format!("vehicle {:#x} att {}", v.datum, e.att),
                    systems: Arc::new(kept),
                    locs,
                    birth: None,
                    path: Some(path.clone()),
                }, [e.a, e.b]));
            }
        }
        sc.set_dynamic_lights(lights);
        sc.set_vehicle_particle_sources(sources);
    }

    /// The lens flares of this frame (the glow at a lit lamp). Separate from [`FxState::install`]
    /// because decoding a flare's sprite needs the GPU device; the sprite is decoded once per tag.
    pub fn install_lenses(&mut self, sc: &mut SceneController, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut flares = Vec::new();
        for v in &self.frame.vehicles {
            for l in &v.lenses {
                let sprite = self.lens_cache.entry(l.lens).or_insert_with(|| sc.lens_flare_for(l.lens, device, queue)).clone();
                if let Some((def, tex)) = sprite {
                    flares.push(hms_render::LensFlareInstance { def, tex, pos: l.pos, fwd: l.fwd, scale: 1.0, color: [l.value; 3] });
                }
            }
        }
        sc.set_vehicle_lens_flares(flares);
    }
}

// ---------------------------------------------------------------------------------------------
// THEATER
// ---------------------------------------------------------------------------------------------

/// The seat power at film tick `tick`, as a pure function of the playhead: the driver rides
/// (`(from, to)` in ticks, sorted) ramp it up at the authored powerup time and the gaps ramp it
/// down at the powerdown time, exactly as [`FxState`] does in test mode, so scrubbing back and
/// forth gives the same lights at the same tick. `(power, occupied)`.
pub fn seat_power_at(spans: &[(f32, f32)], up: f32, down: f32, tick: f32) -> (f32, bool) {
    let secs = |ticks: f32| ticks / 60.0;
    let (mut v, mut t) = (0.0f32, 0.0f32);
    for &(from, to) in spans {
        if from >= tick {
            break;
        }
        v = approach(v, 0.0, down, secs(from - t));
        let end = to.min(tick);
        v = approach(v, 1.0, up, secs(end - from));
        t = end;
        if tick <= to {
            return (v, true);
        }
    }
    (approach(v, 0.0, down, secs(tick - t)), false)
}

/// Theater's lights and effects for the recorded vehicles.
///
/// **What a film records of a vehicle**: position, velocity, orientation, angular velocity and
/// the attachment records of whoever rides it (a seat index and a tick span). **What it does
/// not**: the driver's stick, the brake, the hand brake, the boost, the engine speed. So only
/// what the seat states is lit -- `driver_seat_occupied` / `driver_seat_power` (headlights,
/// running lights, the Ghost / Wraith / Revenant underside glow, the dashboard) -- and every
/// input that needs the stick (`jump` = brake, `engine_power`, `boost`) stays at its idle value:
/// the taillights show at their authored running brightness, never at their brake brightness,
/// because the recording cannot say when the driver braked. A brake guessed from the speed
/// dropping would be an invention (coasting down is not braking).
#[derive(Default)]
pub struct TheaterFx {
    pub state: FxState,
    vehicles: VehicleIndex,
}

impl TheaterFx {
    pub fn step(
        &mut self,
        driver: &crate::film_play::ActorDriver,
        pb: &crate::film_play::Playback,
        actors: &crate::reach_actor::ActorScene,
        native: &NativeDll,
        cache: CacheHandle,
    ) {
        let mut frame = FxFrame::default();
        if let Some(vt) = driver.vehicles.as_ref() {
            let mut nets: Vec<(&u32, &u32)> = driver.vehicle_actor.iter().collect();
            nets.sort();
            for (&net, &aid) in nets {
                let Some(a) = actors.get(aid) else { continue };
                // a parked actor is a vehicle that is not alive at this tick
                if (a.pos - Vec3::from(driver.park)).length() < 1.0 {
                    continue;
                }
                let obj_tag = a.def.obj_tag;
                let def = self.state.def(&mut self.vehicles, native, cache, obj_tag);
                let (driver_seats, hover_family) = match self.vehicles.entry(native, cache, obj_tag) {
                    Some(e) => (
                        e.seats.seats.iter().map(|s| s.is_driver()).collect::<Vec<_>>(),
                        matches!(e.phys.as_ref().map(|p| &p.family), Some(Family::Scout { .. } | Family::Revenant { .. })),
                    ),
                    None => (Vec::new(), false),
                };
                let mut spans: Vec<(f32, f32)> = vt
                    .rides
                    .iter()
                    .filter(|r| r.vehicle_net_id == net && r.seat.is_some_and(|s| driver_seats.get(usize::from(s)) == Some(&true)))
                    .map(|r| (r.from_tick as f32, r.to_tick as f32))
                    .collect();
                spans.sort_by(|x, y| x.0.total_cmp(&y.0));
                let (up, down) = def.powered.unwrap_or((0.0, 0.0));
                let (power, occupied) = if def.powered.is_some() { seat_power_at(&spans, up, down, pb.tick) } else { (0.0, false) };
                // a hover vehicle's points are taken as engaged while it is powered (not recorded)
                let inputs = FxInputs { driver_power: power, occupied, hover: if hover_family { power } else { 0.0 }, ..Default::default() };
                let who = native.tag_name(cache, obj_tag).unwrap_or_default();
                let v = self.state.evaluate(&def, net, a.world(), inputs, &who);
                if !v.lights.is_empty() || !v.lenses.is_empty() || !v.effects.is_empty() {
                    frame.vehicles.push(v);
                }
            }
        }
        self.state.frame = frame;
    }
}

/// [`FxState::install`], or -- with no test mode running -- the removal of whatever it installed.
pub fn install(sc: &mut SceneController, fx: Option<&mut FxState>, t: f32, device: &wgpu::Device, queue: &wgpu::Queue) {
    match fx {
        Some(fx) => {
            fx.install(sc, t);
            fx.install_lenses(sc, device, queue);
        }
        None if !sc.dynamic_lights().is_empty() || sc.vehicle_particle_source_count() > 0 || sc.vehicle_lens_flare_count() > 0 => {
            sc.set_dynamic_lights(Vec::new());
            sc.set_vehicle_particle_sources(Vec::new());
            sc.set_vehicle_lens_flares(Vec::new());
        }
        None => {}
    }
}

/// Seconds of a vehicle's path kept for its moving particle sources.
const TRAIL_SECONDS: f32 = 6.0;
/// Seed space of the vehicle lane (the others use `0x51F7..`, `0x9E37..`, `0xA7F0..`, `0xF112..`).
const VEHICLE_FX_SEED: u64 = 0xB3_17_0000_0000_0000;

#[cfg(test)]
mod probes {
    use hms_native::{CacheHandle, NativeDll};

    use crate::reach_actor::{block_at, tag_block};

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }
    fn rd(b: &[u8], o: usize) -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }
    fn rf(b: &[u8], o: usize) -> f32 { f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }

    fn sid(n: &NativeDll, c: CacheHandle, v: u32) -> String {
        if v == 0 || v == 0xFFFF_FFFF { "-".into() } else { n.string_id(c, v).unwrap_or_else(|| format!("{v:#x}")) }
    }
    fn tref(n: &NativeDll, c: CacheHandle, e: &[u8], o: usize) -> String {
        let d = rd(e, o + 0xC);
        if d == 0 || d == 0xFFFF_FFFF { return "-".into(); }
        format!("{}:{}", n.tag_class(c, d & 0xFFFF).unwrap_or_default(), n.tag_name(c, d & 0xFFFF).unwrap_or_default())
    }

    fn dump_effe(n: &NativeDll, c: CacheHandle, t: u32) {
        println!("--- EFFE {t:#x} {}", n.tag_name(c, t).unwrap_or_default());
        let m = n.tag_meta(c, t, 0, 0x60).unwrap();
        println!("    flags={:#x} loop_start_event={}", rd(&m, 0), i16::from_le_bytes([m[0x14], m[0x15]]));
        let locs = tag_block(n, c, t, 0x20, 8, 8);
        for (i, l) in locs.chunks_exact(8).enumerate() { println!("    loc[{i}] marker='{}' flags={:#x}", sid(n, c, rd(l, 0)), u16::from_le_bytes([l[4], l[5]])); }
        let evs = tag_block(n, c, t, 0x2C, 64, 32);
        for (i, ev) in evs.chunks_exact(64).enumerate() {
            println!("    ev[{i}] '{}' flags={:#x} skip={} delay=({},{}) dur=({},{})", sid(n, c, rd(ev, 0)), u16::from_le_bytes([ev[4], ev[5]]), rf(ev, 8), rf(ev, 0xC), rf(ev, 0x10), rf(ev, 0x14), rf(ev, 0x18));
            let parts = block_at(n, c, ev, 0x1C, 100, 32);
            for (j, p) in parts.chunks_exact(100).enumerate() {
                println!("       part[{j}] in={} viol={} loc={} flags={:#x} type={} A={:#x} B={:#x}", u16::from_le_bytes([p[0], p[1]]), u16::from_le_bytes([p[2], p[3]]), i16::from_le_bytes([p[4], p[5]]), u16::from_le_bytes([p[8], p[9]]), tref(n, c, p, 0x14), rd(p, 0x5C), rd(p, 0x60));
            }
            let ps = block_at(n, c, ev, 0x34, 112, 32);
            for (j, p) in ps.chunks_exact(112).enumerate() {
                let ems = block_at(n, c, p, 0x5C, 796, 8);
                println!("       psys[{j}] particle={} loc={} coord={} env={} flags={:#x} emitters={}", tref(n, c, p, 4), rd(p, 0x14), u16::from_le_bytes([p[0x18], p[0x19]]), u16::from_le_bytes([p[0x1A], p[0x1B]]), rd(p, 0x24), ems.len() / 796);
                for (k, em) in ems.chunks_exact(796).enumerate() {
                    let prop = |lbl: &str, o: usize| println!("          em[{k}] {lbl}: in={} range={} mod={} modin={} const={}", em[o], em[o + 1], em[o + 2], em[o + 3], rf(em, o + 0x18));
                    prop("rate", 0x158); prop("count0", 0x118); prop("lifespan", 0x178);
                }
            }
        }
    }

    /// Print-only: effects by tag-path substring (`VFX_EFFE`).
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_effects_named() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_EFFE").unwrap_or_default();
        for t in n.tags_by_group(c, "effe") {
            if n.tag_name(c, t).unwrap_or_default().contains(&filt) { dump_effe(&n, c, t); }
        }
    }

    /// Print-only: every vehicle's object-function block, attachments, and the attached effects.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn probe_vehicle_functions_and_attachments() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_FILTER").unwrap_or_default();
        for v in n.tags_by_group(c, "vehi") {
            let name = n.tag_name(c, v).unwrap_or_default();
            if !filt.is_empty() && !name.contains(&filt) { continue; }
            println!("=== VEHI {v:#x} {name}");
            let fns = tag_block(&n, c, v, 0xEC, 64, 256);
            for (i, e) in fns.chunks_exact(64).enumerate() {
                println!("  fn[{i}] flags={:#x} import='{}' export='{}' off_with='{}' range='{}' min={:.3} scale_by='{}' mapsz={}",
                    rd(e, 0), sid(&n, c, rd(e, 4)), sid(&n, c, rd(e, 8)), sid(&n, c, rd(e, 0xC)), sid(&n, c, rd(e, 0x10)),
                    rf(e, 0x14), sid(&n, c, rd(e, 0x2C)), rd(e, 0x18));
            }
            let atts = tag_block(&n, c, v, 0x104, 32, 24);
            for (i, e) in atts.chunks_exact(32).enumerate() {
                println!("  att[{i}] {} marker='{}' chg={} flags={:#x} primary='{}' secondary='{}'",
                    tref(&n, c, e, 0), sid(&n, c, rd(e, 0x10)), e[0x14], e[0x15], sid(&n, c, rd(e, 0x18)), sid(&n, c, rd(e, 0x1C)));
            }
            let m = n.tag_meta(c, v, 0xC0, 16).unwrap();
            println!("  material_effects: {}", tref(&n, c, &m, 0));
            let ps = tag_block(&n, c, v, 0x420, 8, 2);
            for (i, e) in ps.chunks_exact(8).enumerate() { println!("  powered_seat[{i}] up={} down={}", rf(e, 0), rf(e, 4)); }
        }
    }

    /// Print-only: one ligh / effe tag in full.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_vehicle_attached_tags() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_FILTER").unwrap_or_default();
        let mut seen = std::collections::BTreeSet::new();
        for v in n.tags_by_group(c, "vehi") {
            let name = n.tag_name(c, v).unwrap_or_default();
            if !filt.is_empty() && !name.contains(&filt) { continue; }
            let atts = tag_block(&n, c, v, 0x104, 32, 24);
            for e in atts.chunks_exact(32) {
                let d = rd(e, 0xC);
                if d == 0 || d == 0xFFFF_FFFF { continue; }
                let t = d & 0xFFFF;
                if !seen.insert(t) { continue; }
                match n.tag_class(c, t).as_deref() {
                    Some("ligh") => {
                        let m = n.tag_meta(c, t, 0, 0xCC).unwrap();
                        println!("--- LIGH {t:#x} {} method={} type={} flags={:#x} end={} start={} fov={} hot={} fall={} diff={} nearfade={} shadowcut={} q={}",
                            n.tag_name(c, t).unwrap_or_default(), m[0], rd(&m, 4) & 0xFFFF, rd(&m, 0) >> 16, rf(&m, 8), rf(&m, 0xC), rf(&m, 0x1C), rf(&m, 0x20), rf(&m, 0x24), rf(&m, 0x28), rf(&m, 0x2C), rf(&m, 0x30), m[0x34]);
                        println!("      color in='{}' range='{}' mod={} modin='{}' mapsz={} | intensity in='{}' range='{}' mod={} modin='{}' mapsz={}",
                            sid(&n, c, rd(&m, 0x38)), sid(&n, c, rd(&m, 0x3C)), m[0x40], sid(&n, c, rd(&m, 0x44)), rd(&m, 0x48),
                            sid(&n, c, rd(&m, 0x5C)), sid(&n, c, rd(&m, 0x60)), m[0x64], sid(&n, c, rd(&m, 0x68)), rd(&m, 0x6C));
                        for (lbl, so, ao) in [("color", 0x48usize, 0x54usize), ("intensity", 0x6C, 0x78)] {
                            let (sz, ad) = (rd(&m, so), rd(&m, ao));
                            if sz > 0 && sz < 0x400 && ad != 0 {
                                if let Some(b) = n.tag_ptr_bytes(c, ad, sz) { println!("      {lbl} blob {}", b.iter().map(|x| format!("{x:02x}")).collect::<String>()); }
                            }
                        }
                    }
                    Some("effe") => dump_effe(&n, c, t),
                    other => println!("--- {other:?} {t:#x} {}", n.tag_name(c, t).unwrap_or_default()),
                }
            }
        }
    }

    /// Print-only: every vehicle's `foot` (material effects) table.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_vehicle_material_effects() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_FILTER").unwrap_or_default();
        let mut seen = std::collections::BTreeSet::new();
        for v in n.tags_by_group(c, "vehi") {
            let name = n.tag_name(c, v).unwrap_or_default();
            if !filt.is_empty() && !name.contains(&filt) { continue; }
            let m = n.tag_meta(c, v, 0xC0, 16).unwrap();
            let d = rd(&m, 0xC);
            if d == 0 || d == 0xFFFF_FFFF { continue; }
            let foot = d & 0xFFFF;
            if !seen.insert(foot) { continue; }
            println!("=== FOOT {} (of {name})", n.tag_name(c, foot).unwrap_or_default());
            let outer = tag_block(&n, c, foot, 0, 36, 64);
            for (i, e) in outer.chunks_exact(36).enumerate() {
                for (lbl, off) in [("sounds", 0xCusize), ("effects", 0x18)] {
                    let rows = block_at(&n, c, e, off, 44, 512);
                    let named: Vec<String> = rows.chunks_exact(44).map(|r| {
                        let t = rd(r, 0xC); let t2 = rd(r, 0x1C);
                        let f = |d: u32| if d == 0 || d == 0xFFFF_FFFF { "-".to_string() } else { n.tag_name(c, d & 0xFFFF).unwrap_or_default() };
                        format!("{}={}|{}", sid(&n, c, rd(r, 0x20)), f(t), f(t2))
                    }).collect();
                    if !named.is_empty() { println!("  type[{i}] {lbl} ({}): {}", named.len(), named.join("  ")); }
                }
            }
        }
    }

    /// Print-only: the animated shader parameters of every vehicle's render model.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_vehicle_shader_animation() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_FILTER").unwrap_or_default();
        for v in n.tags_by_group(c, "vehi") {
            let name = n.tag_name(c, v).unwrap_or_default();
            if name.contains("turret") || name.contains("weapons") || name.contains("attachments") { continue; }
            if !filt.is_empty() && !name.contains(&filt) { continue; }
            crate::scene::dump_object_anim_params(&n, c, v);
        }
    }

    /// Print-only: every tag path containing `VFX_TAG` (any group), with its class.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_tag_names() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_TAG").unwrap_or_default();
        for g in ["effe", "ligh", "cntl", "ltvl", "lens", "foot", "snd!", "lsnd", "decs", "prt3", "sefc"] {
            for t in n.tags_by_group(c, g) {
                let nm = n.tag_name(c, t).unwrap_or_default();
                if nm.contains(&filt) { println!("TAG {g} {t:#x} {nm}"); }
            }
        }
    }

    /// Print-only: `foot` tags by path substring (`VFX_FOOT`), every effect type's rows.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_foot_named() {
        let Some((_g, n, c)) = open() else { return };
        let filt = std::env::var("VFX_FOOT").unwrap_or_default();
        for foot in n.tags_by_group(c, "foot") {
            let nm = n.tag_name(c, foot).unwrap_or_default();
            if !nm.contains(&filt) { continue; }
            println!("=== FOOT {nm}");
            let outer = tag_block(&n, c, foot, 0, 36, 64);
            for (i, e) in outer.chunks_exact(36).enumerate() {
                for (lbl, off) in [("sounds", 0xCusize), ("effects", 0x18)] {
                    let rows = block_at(&n, c, e, off, 44, 512);
                    let named: Vec<String> = rows.chunks_exact(44).map(|r| {
                        let f = |d: u32| if d == 0 || d == 0xFFFF_FFFF { "-".to_string() } else { n.tag_name(c, d & 0xFFFF).unwrap_or_default() };
                        format!("{}={}|{}", sid(&n, c, rd(r, 0x20)), f(rd(r, 0xC)), f(rd(r, 0x1C)))
                    }).collect();
                    if !named.is_empty() { println!("  type[{i}] {lbl} ({}): {}", named.len(), named.join("  ")); }
                }
            }
        }
    }

    /// Print-only: drive a vehicle through the production tick and print what lights.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_ride_lights() {
        use crate::play_mode::ride_harness::Ride;
        let Some((_g, n, c)) = open() else { return };
        let suffix = std::env::var("VFX_RIDE").unwrap_or("warthog\\warthog".into());
        let Some(mut r) = Ride::with_vehicle(n.clone(), c, &suffix) else { println!("no vehicle {suffix}"); return };
        let tag = r.objects[0].primary_tag;
        r.force_seat(tag, 0).unwrap();
        let show = |r: &Ride, label: &str| {
            let f = &r.pm.vehicle_fx.frame;
            let Some(v) = f.vehicles.first() else { println!("{label}: no vehicle fx"); return };
            println!("{label}: power={:.2} brake={} eng={:.2} hover={:.2} lights={} effects={}", v.inputs.driver_power, v.inputs.brake, v.inputs.engine.engine_power, v.inputs.hover, v.lights.len(), v.effects.len());
            for l in &v.lights { println!("    light ligh={:#x} scale={:.3} pos=({:.2},{:.2},{:.2}) rad={:?} cone={:?}", l.light.light_tag, l.scale, l.pos.x, l.pos.y, l.pos.z, l.light.radiance_at(0.0), l.cone); }
            for e in &v.effects { println!("    effect att{} effe={:#x} a={:.3} b={:.3}", e.att, e.effe, e.a, e.b); }
        };
        {
            let def = r.pm.vehicle_fx.defs.get(&tag).cloned();
            let mut vi = crate::play_vehicle::VehicleIndex::default();
            let e = vi.entry(&n, c, tag).unwrap();
            for name in ["fx_headlight_light", "fx_brakelight_light", "fx_headlights_light", "fx_antigrav_light", "fx_antigrav", "fx_boost"] {
                for f in e.marker_frames(&n, c, name) {
                    println!("MARKER {name}: x=({:.2},{:.2},{:.2}) z=({:.2},{:.2},{:.2}) pos=({:.2},{:.2},{:.2})", f.x_axis.x, f.x_axis.y, f.x_axis.z, f.z_axis.x, f.z_axis.y, f.z_axis.z, f.w_axis.x, f.w_axis.y, f.w_axis.z);
                }
            }
            println!("powered seats: {:?}", def.map(|d| d.powered));
        }
        let idle = r.input();
        for k in 0..4 { r.run(0.5, &idle); show(&r, &format!("idle {:.1}s", (k + 1) as f32 * 0.5)); }
        let mut go = r.input(); go.fwd = 1.0;
        r.run(3.0, &go); show(&r, "cruise");
        let mut back = r.input(); back.fwd = -1.0;
        r.run(0.2, &back); show(&r, "reverse stick 0.2s");
        r.run(0.6, &back); show(&r, "reverse stick 0.8s");
        let mut hb = r.input(); hb.crouch = true;
        r.run(0.1, &hb); show(&r, "handbrake");
        r.run(2.0, &idle); show(&r, "released");
        println!("unresolved: {:?}", r.pm.vehicle_fx.unresolved);
    }

    /// Print-only: which particle properties of each vehicle effect read `effect A/B scale`
    /// (states 13 / 14), through the scene's own decode.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn probe_effect_scale_users() {
        let Some((_g, n, c)) = open() else { return };
        let mut sc = crate::scene::SceneController::new_shared(n.clone());
        sc.attach_cache(c);
        let filt = std::env::var("VFX_FILTER").unwrap_or_default();
        for v in n.tags_by_group(c, "vehi") {
            let name = n.tag_name(c, v).unwrap_or_default();
            if !filt.is_empty() && !name.contains(&filt) { continue; }
            for (effe, marker, p1) in crate::reach_projectile::object_attachment_effects(&n, c, v) {
                let systems = sc.effect_particle_systems(effe);
                let mut users = Vec::new();
                for (si, sy) in systems.iter().enumerate() {
                    for (ei, e) in sy.emitters.iter().enumerate() {
                        for (pn, pf) in [("offset", &e.offset), ("direction", &e.direction), ("radius", &e.radius), ("angle", &e.angle), ("axis_angle", &e.axis_angle), ("start", &e.starting_count), ("max", &e.max_count), ("rate", &e.rate), ("life", &e.lifespan), ("accel", &e.self_accel), ("velocity", &e.velocity), ("rotation", &e.rotation), ("rot_rate", &e.rotation_rate), ("size", &e.size), ("scale", &e.scale), ("tint", &e.tint), ("alpha", &e.alpha), ("black", &e.black_point)] {
                            let hit = [pf.input, pf.range, pf.mod_input].iter().any(|s| *s == 13 || *s == 14);
                            if hit { users.push(format!("sys{si}.em{ei}.{pn}(in{} rng{} mod{} modin{})", pf.input, pf.range, pf.modifier, pf.mod_input)); }
                        }
                    }
                }
                println!("{} effe={effe:#x} '{}' marker={} primary={} looping={} systems={} A/B users: {}", name.rsplit('\\').next().unwrap_or(""), n.tag_name(c, effe).unwrap_or_default().rsplit('\\').take(2).collect::<Vec<_>>().join("\\"), n.string_id(c, marker).unwrap_or_default(), n.string_id(c, p1).unwrap_or_default(), systems.first().map_or(false, |s| s.looping), systems.len(), if users.is_empty() { "none".into() } else { users.join(", ") });
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GATES
// ---------------------------------------------------------------------------------------------

/// The pure ones run in the normal suite; the ones that read the real `forge_halo` cache are
/// `#[ignore]` like every other native-opening gate:
/// `cargo test --release -p hms-app play_vehicle_fx::gates -- --ignored --test-threads=1`.
///
/// Every native gate drives the REAL `PlayMode::tick` through `ride_harness` (the production
/// brake counter, the production seat ramp, the production marker lookup) and reads
/// [`FxState::frame`]; the expectations are tag literals read by a separate path, never the
/// function under test.
#[cfg(test)]
mod gates {
    use super::*;
    use crate::play_mode::ride_harness::{Ride, DT};
    use crate::scene::MapFn;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    fn objfn(export: &str, import: &str, off_with: &str, scale_by: &str, map: Option<MapFn>) -> ObjFn {
        ObjFn { flags: 0, import: import.into(), export: export.into(), off_with: off_with.into(), scale_by: scale_by.into(), min_value: 0.0, map }
    }

    /// The object-function algebra the engine's `sub_140D47160` applies: `turn off with` gates the
    /// function, `scale by` multiplies it, an unknown input is REPORTED and never guessed, and an
    /// attachment that names nothing is unconditional.
    #[test]
    fn functions_gate_scale_and_report_what_they_cannot_supply() {
        // brakelights: jump through a 0.25 -> 1.0 ramp, scaled by driver_seat_power
        let ramp = MapFn::Identity { lo: 0.25, hi: 1.0 };
        let fns = vec![
            objfn("brakelights", "jump", "", "driver_seat_power", Some(ramp)),
            objfn("gated", "driver_seat_power", "driver_seat_occupied", "", None),
        ];
        let mut unres = BTreeSet::new();
        let mut inp = FxInputs { driver_power: 0.5, occupied: true, brake: false, ..Default::default() };
        let (v, on) = attachment_state(&fns, 0, "brakelights", &inp, &mut unres).unwrap();
        assert!((v - 0.125).abs() < 1e-6 && on, "coasting: 0.25 * power 0.5 = 0.125, got {v}");
        inp.brake = true;
        let (v, _) = attachment_state(&fns, 0, "brakelights", &inp, &mut unres).unwrap();
        assert!((v - 0.5).abs() < 1e-6, "braking: 1.0 * power 0.5 = 0.5, got {v}");
        inp.driver_power = 0.0;
        let (v, on) = attachment_state(&fns, 0, "brakelights", &inp, &mut unres).unwrap();
        assert!(v == 0.0 && !on, "no power: the lamp is off, got {v} {on}");
        // turn off with: an empty seat switches the function off whatever its input says
        inp.driver_power = 1.0;
        inp.occupied = false;
        let (v, on) = attachment_state(&fns, 0, "gated", &inp, &mut unres).unwrap();
        assert!(v == 0.0 && !on, "turn off with driver_seat_occupied");
        // unconditional attachments
        assert_eq!(attachment_state(&fns, 0, "", &inp, &mut unres), Some((1.0, true)));
        assert_eq!(attachment_state(&fns, ATT_FLAG_ALWAYS_ON, "thrust", &inp, &mut unres), Some((1.0, true)));
        // an input this module does not supply is reported, not lit by a guess
        assert!(unres.is_empty(), "nothing unresolved so far: {unres:?}");
        assert_eq!(attachment_state(&fns, 0, "thrust", &inp, &mut unres), None);
        assert!(unres.contains("thrust"), "{unres:?}");
        // boost is a stated 0 (no boost control exists), and flashlights follow the unscripted game state
        assert_eq!(inp.builtin("boost"), Some(0.0));
        assert_eq!(inp.builtin("flashlights_allowed"), Some(FLASHLIGHTS_ALLOWED));
    }

    /// The seat ramp is linear over the AUTHORED time and instant when none is authored.
    #[test]
    fn the_seat_power_ramp_runs_at_the_authored_rate() {
        assert!((approach(0.0, 1.0, 2.0, 0.5) - 0.25).abs() < 1e-6);
        assert!((approach(0.9, 1.0, 2.0, 0.5) - 1.0).abs() < 1e-6, "no overshoot");
        assert!((approach(1.0, 0.0, 4.0, 1.0) - 0.75).abs() < 1e-6, "powerdown has its own time");
        assert_eq!(approach(0.0, 1.0, 0.0, 0.016), 1.0, "a zero authored time is instant");
    }

    /// Theater's seat power is a pure function of the playhead: two rides, a two-second powerup, a
    /// one-second powerdown.
    #[test]
    fn the_theater_seat_power_is_a_function_of_the_playhead() {
        let spans = [(60.0, 180.0), (300.0, 360.0)];
        let at = |t: f32| seat_power_at(&spans, 2.0, 1.0, t);
        assert_eq!(at(30.0), (0.0, false), "before the first ride");
        let (p, occ) = at(120.0);
        assert!((p - 0.5).abs() < 1e-5 && occ, "a second into a two-second powerup: {p}");
        let (p, occ) = at(180.0);
        assert!((p - 1.0).abs() < 1e-5 && occ, "full power at the end of the ride: {p}");
        let (p, occ) = at(210.0);
        assert!((p - 0.5).abs() < 1e-5 && !occ, "half a second into a one-second powerdown: {p}");
        let (p, _) = at(299.0);
        assert!(p == 0.0, "gone before the next ride: {p}");
        let (p, occ) = at(330.0);
        assert!((p - 0.25).abs() < 1e-5 && occ, "the second ride starts from zero: {p}");
        // a pure function: the same tick, asked in any order, answers the same
        let first = at(330.0);
        let _ = at(100.0);
        assert_eq!(at(330.0), first, "scrubbing must not change the answer");
        // a ride that begins with power left over carries it (200 -> 230 ticks of a 1 s powerdown = 0.5 left)
        let (p, _) = seat_power_at(&[(60.0, 180.0), (210.0, 330.0)], 2.0, 1.0, 240.0);
        assert!((p - 0.75).abs() < 1e-5, "0.5 left over + 0.5 s of powerup at 2 s: {p}");
    }

    fn seated(n: &Arc<NativeDll>, c: CacheHandle, suffix: &str) -> Option<Ride> {
        let mut r = Ride::with_vehicle(n.clone(), c, suffix)?;
        let tag = r.objects[0].primary_tag;
        r.force_seat(tag, 0).ok()?;
        Some(r)
    }

    fn fx(r: &Ride) -> Option<&VehicleFx> {
        r.pm.vehicle_fx.frame.vehicles.first()
    }

    fn named<'a>(n: &NativeDll, c: CacheHandle, v: &'a VehicleFx, part: &str) -> Vec<&'a FxLight> {
        v.lights.iter().filter(|l| n.tag_name(c, l.light.light_tag).is_some_and(|t| t.ends_with(part))).collect()
    }

    /// **THE BRAKE LIGHTS.** Through the production tick: the Warthog's two taillights (one `ligh`,
    /// two markers) sit at 0.25 of their authored brightness while it is driven and go to 1.0 for
    /// as long as it is braking -- a reverse command against the travel (the first 0.5 s only, the
    /// engine's brake window) or the hand brake -- and back down the tick the brake ends.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn braking_drives_the_taillights_between_their_authored_values() {
        let Some((_g, n, c)) = open() else { return };
        let mut r = seated(&n, c, "warthog\\warthog").expect("a Warthog to seat the player in");
        let idle = r.input();
        let mut go = r.input();
        go.fwd = 1.0;
        let mut back = r.input();
        back.fwd = -1.0;
        let mut hand = r.input();
        hand.crouch = true;
        // Authored: `brakelights` maps `jump` 0 -> 0.25 and 1 -> 1.0 (read from the tag below too).
        let (coast, brake) = (0.25f32, 1.0f32);
        {
            let fns = ObjFn::read_all(&n, c, r.objects[0].primary_tag);
            let f = fns.iter().find(|f| f.export == "brakelights").expect("the Warthog authors `brakelights`");
            let m = f.map.as_ref().expect("a mapping curve");
            assert!((m.eval(0.0) - coast).abs() < 1e-3 && (m.eval(1.0) - brake).abs() < 1e-3, "tag says {} -> {}", m.eval(0.0), m.eval(1.0));
        }
        let scales = |r: &Ride| -> Vec<f32> {
            let v = fx(r).expect("a lit vehicle");
            named(&n, c, v, "running\\brakelights").iter().map(|l| l.scale).collect()
        };
        r.run(0.5, &idle);
        r.run(3.0, &go);
        let at = scales(&r);
        assert_eq!(at.len(), 2, "two taillight markers (left and right): {at:?}");
        assert!(at.iter().all(|s| (s - coast).abs() < 1e-3), "driven and not braking: {at:?}");
        let tails: Vec<Vec3> = named(&n, c, fx(&r).unwrap(), "running\\brakelights").iter().map(|l| l.pos).collect();
        let hull = r.pm.vehicle_fx.frame.vehicles[0].world.w_axis.truncate();
        assert!(tails.iter().all(|p| (p.y - hull.y).abs() > 0.1 && (p.y - hull.y).abs() < 0.4), "one lamp each side: {tails:?}");
        assert!(tails.iter().all(|p| p.x < hull.x - 0.5), "behind the hull origin: {tails:?} vs {hull:?}");
        // reverse against the travel: the engine's half-second brake window, 30 ticks
        r.run(DT * 5.0, &back);
        assert!(scales(&r).iter().all(|s| (s - brake).abs() < 1e-3), "braking: {:?}", scales(&r));
        r.run(DT * 20.0, &back);
        assert!(scales(&r).iter().all(|s| (s - brake).abs() < 1e-3), "still inside the window: {:?}", scales(&r));
        r.run(1.0, &back);
        assert!(scales(&r).iter().all(|s| (s - coast).abs() < 1e-3), "the window is over, reversing is not braking: {:?}", scales(&r));
        // the hand brake
        r.run(DT * 3.0, &hand);
        assert!(scales(&r).iter().all(|s| (s - brake).abs() < 1e-3), "hand brake: {:?}", scales(&r));
        r.run(DT * 2.0, &idle);
        assert!(scales(&r).iter().all(|s| (s - coast).abs() < 1e-3), "released: {:?}", scales(&r));
    }

    /// **THE HEADLIGHT.** A frustum light with the authored cone, shining along the vehicle and
    /// riding with it.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn the_headlights_are_authored_cones_that_ride_with_the_vehicle() {
        let Some((_g, n, c)) = open() else { return };
        let (mut ran, mut drove) = (0, 0);
        for (suffix, part) in [("warthog\\warthog", "headlight_beam"), ("scorpion\\scorpion", "running\\headlight")] {
            let Some(mut r) = seated(&n, c, suffix) else { continue };
            let idle = r.input();
            r.run(4.0, &idle);
            let before: Vec<Vec3> = named(&n, c, fx(&r).unwrap(), part).iter().map(|l| l.pos).collect();
            let lamps = named(&n, c, fx(&r).unwrap(), part);
            assert!(!lamps.is_empty(), "{suffix}: no `{part}` light");
            let fwd = fx(&r).unwrap().world.transform_vector3(Vec3::X).normalize();
            for l in &lamps {
                let cone = l.cone.expect("a headlight is a frustum light");
                // authored: field of view 45 degrees (0.7853982 rad) -> cos(22.5 deg)
                assert!((cone.cos_cut - 0.923_879_5).abs() < 1e-4, "{suffix}: cosCut {}", cone.cos_cut);
                assert!(l.dir.dot(fwd) > 0.95, "{suffix}: shines {:?}, vehicle faces {fwd:?}", l.dir);
            }
            let hull0 = fx(&r).unwrap().world.w_axis.truncate();
            let mut go = r.input();
            go.fwd = 1.0;
            r.run(3.0, &go);
            let after: Vec<Vec3> = named(&n, c, fx(&r).unwrap(), part).iter().map(|l| l.pos).collect();
            let moved = fx(&r).unwrap().world.w_axis.truncate() - hull0;
            assert_eq!(before.len(), after.len());
            // each lamp travelled with the hull (to a few cm: the hull also pitches as it accelerates)
            assert!(before.iter().zip(&after).all(|(a, b)| ((*b - *a) - moved).length() < 0.3), "{suffix}: the lamps did not follow the hull {moved:?}: {before:?} -> {after:?}");
            if moved.length() > 5.0 {
                drove += 1;
            }
            ran += 1;
        }
        assert!(ran >= 2, "ran on {ran} vehicles");
        assert!(drove >= 1, "no vehicle actually drove, so `rides with the vehicle` was never tested");
    }

    /// **NOBODY, NO LIGHTS; THE SEAT POWER RAMPS.** An empty vehicle is dark. Sitting in the
    /// driver seat ramps `driver_seat_power` up over the authored `driver powerup time`, and
    /// leaving ramps it down over `driver powerdown time` until the lights are gone.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn the_lights_follow_the_driver_seat_power() {
        let Some((_g, n, c)) = open() else { return };
        let mut ran = 0;
        for suffix in ["scorpion\\scorpion", "wraith\\wraith"] {
            let Some(mut r) = Ride::with_vehicle(n.clone(), c, suffix) else { continue };
            let tag = r.objects[0].primary_tag;
            let idle = r.input();
            r.run(0.5, &idle);
            assert!(fx(&r).is_none(), "{suffix}: an empty vehicle is dark");
            // the authored times, read straight from the unit's `powered seats` block
            let ps = tag_block(&n, c, tag, UNIT_POWERED_SEATS, 8, 2);
            let (up, down) = (rd_f32(&ps, 0), rd_f32(&ps, 4));
            assert!(up > 1.0, "{suffix}: this gate needs a slow power-up to measure; authored {up}");
            r.force_seat(tag, 0).unwrap();
            r.run(up * 0.5, &idle);
            let half = fx(&r).expect("lit once seated").inputs.driver_power;
            assert!((half - 0.5).abs() < 0.03, "{suffix}: half the powerup time is half power, got {half} (up {up})");
            r.run(up, &idle);
            assert!((fx(&r).unwrap().inputs.driver_power - 1.0).abs() < 1e-3);
            // leave
            r.pm.exit_vehicle(&n, c);
            let mut left_at = None;
            for k in 0..((up + down + 12.0) / DT) as usize {
                r.tick(&idle);
                if left_at.is_none() && r.pm.vehicle_fx.frame.vehicles.first().is_some_and(|v| v.inputs.driver_power < 1.0) {
                    left_at = Some(k);
                }
                if r.pm.vehicle_fx.frame.vehicles.is_empty() {
                    break;
                }
            }
            assert!(r.pm.vehicle_fx.frame.vehicles.is_empty(), "{suffix}: the lights never went out");
            ran += 1;
        }
        assert!(ran >= 1, "ran on no vehicle");
    }

    /// **THE UNDERSIDE.** The Ghost, the Wraith and the Revenant each hang an `antigrav_lift`
    /// effect on `driver_seat_power`: it exists while somebody is in the driver seat, its `effect
    /// A scale` IS the seat power (so it fades in with it) and it reaches the scene's vehicle
    /// particle lane with every system located.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn the_hover_vehicles_show_their_underside_effect() {
        let Some((_g, n, c)) = open() else { return };
        let mut sc = SceneController::new_shared(n.clone());
        sc.attach_cache(c);
        let mut ran = 0;
        for suffix in ["ghost\\ghost", "wraith\\wraith", "revenant\\revenant"] {
            let Some(mut r) = Ride::with_vehicle(n.clone(), c, suffix) else { continue };
            let tag = r.objects[0].primary_tag;
            let idle = r.input();
            r.run(0.2, &idle);
            assert!(fx(&r).is_none(), "{suffix}: dark and silent with nobody aboard");
            r.force_seat(tag, 0).unwrap();
            r.run(0.25, &idle);
            let v = fx(&r).unwrap_or_else(|| panic!("{suffix}: no effects while driven"));
            let e = v
                .effects
                .iter()
                .find(|e| n.tag_name(c, e.effe).is_some_and(|t| t.ends_with("antigrav_lift")))
                .unwrap_or_else(|| panic!("{suffix}: no antigrav_lift effect: {:?}", v.effects))
                .clone();
            assert!((e.a - v.inputs.driver_power).abs() < 1e-6, "{suffix}: effect A scale {} must be the seat power {}", e.a, v.inputs.driver_power);
            r.run(4.0, &idle);
            let full = fx(&r).unwrap();
            assert!((full.effects.iter().find(|e| n.tag_name(c, e.effe).is_some_and(|t| t.ends_with("antigrav_lift"))).unwrap().a - 1.0).abs() < 1e-3, "{suffix}: full power, full scale");
            // into the scene
            r.pm.vehicle_fx.install(&mut sc, 0.0);
            assert!(sc.vehicle_particle_source_count() >= 1, "{suffix}: the lane got nothing");
            let src = sc.effect_particle_systems(e.effe);
            assert!(!src.is_empty(), "{suffix}: the effect decodes to no particle system");
            ran += 1;
        }
        assert!(ran >= 2, "ran on {ran} hover vehicles");
    }

    /// **ON THE DRAWN LAMP, NOT JUST SOMEWHERE.** The light is where the DRAWN vehicle's lamp is:
    /// the ridden vehicle's actor, posed by the production tick, carries the render model's
    /// `fx_brakelight_light` markers; each taillight must sit on one of them (read here through the
    /// actor's own pose and world, not through the module under test), and the same holds after the
    /// hull has moved and turned.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn the_lights_sit_on_the_drawn_lamps() {
        let Some((_g, n, c)) = open() else { return };
        let mut r = seated(&n, c, "warthog\\warthog").expect("a Warthog");
        let tag = r.objects[0].primary_tag;
        let mode = n.resolve_mode(c, tag);
        let model = crate::reach_actor::ReachModel::read(&n, c, mode).expect("the render model");
        let group = model.marker_group(&n, c, "fx_brakelight_light").expect("the Warthog's model carries its tail-light markers").clone();
        let mut go = r.input();
        go.fwd = 1.0;
        go.yaw = 0.0;
        let mut checked = 0;
        for (secs, yaw) in [(1.0, 0.0f32), (3.0, 0.9)] {
            go.yaw = yaw;
            r.run(secs, &go);
            let actor = r.actors.actors.iter().find(|a| a.def.mode_tag == mode).expect("the ridden vehicle's actor");
            let world = actor.world();
            let drawn: Vec<Vec3> = group.markers.iter().map(|mk| world.transform_point3(actor.pose.marker_model(mk).w_axis.truncate())).collect();
            let v = fx(&r).unwrap();
            let tails = named(&n, c, v, "running\\brakelights");
            assert_eq!(tails.len(), 2);
            for l in &tails {
                let d = drawn.iter().map(|p| (*p - l.pos).length()).fold(f32::MAX, f32::min);
                assert!(d < 0.05, "a taillight is {d:.3} wu from the nearest drawn lamp marker (hull at {:?})", v.world.w_axis);
                checked += 1;
            }
        }
        assert_eq!(checked, 4, "ran on {checked} lamp checks");
        let hull = r.pm.vehicle_fx.frame.vehicles[0].world.w_axis.truncate();
        assert!(hull.length() > 5.0, "the vehicle never left the origin: {hull:?}");
    }

    /// One 20-float record of the packed simple-light uniform.
    fn packed(sl: &[f32]) -> Vec<&[f32]> {
        sl.chunks_exact(20).collect()
    }

    /// WGSL `calc_simple_lights` for one light and a surface facing the light, in Rust.
    fn lit(rec: &[f32], p: Vec3) -> f32 {
        let to_l = Vec3::new(rec[0], rec[1], rec[2]) - p;
        let d2 = to_l.length_squared();
        if d2 >= rec[3] {
            return 0.0;
        }
        let d = d2.sqrt().max(1e-4);
        let dist_att = ((rec[17] - d) * rec[18]).clamp(0.0, 1.0).powi(2);
        let dir = Vec3::new(rec[4], rec[5], rec[6]);
        let cos_ang = (-(to_l / d)).dot(dir).clamp(0.0, 1.0);
        let cone = ((cos_ang - rec[12]) * rec[13].max(1e-4)).clamp(0.0, 1.0).powf(rec[14].max(1.0));
        let ang = cone + (1.0 - cone) * rec[7];
        rec[8].max(rec[9]).max(rec[10]) * ang * dist_att * 0.3183
    }

    /// **THE LIGHT REACHES THE UNIFORM.** Through the scene's own packer: the brake lights are in
    /// the 5x vec4 records the shader reads, at the lamp, at 4x the brightness when braking, and
    /// they light a point just behind the lamp and not one far from it.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn the_brake_light_is_in_the_packed_uniform_at_its_brake_brightness() {
        let Some((_g, n, c)) = open() else { return };
        let mut sc = SceneController::new_shared(n.clone());
        sc.attach_cache(c);
        let mut r = seated(&n, c, "warthog\\warthog").expect("a Warthog");
        let idle = r.input();
        let mut hand = r.input();
        hand.crouch = true;
        let mut pack = |r: &mut Ride, input: &MoveInput| -> (Vec<Vec<f32>>, Vec<Vec3>) {
            r.run(DT * 3.0, input);
            r.pm.vehicle_fx.install(&mut sc, 0.0);
            let tails: Vec<Vec3> = named(&n, c, fx(r).unwrap(), "running\\brakelights").iter().map(|l| l.pos).collect();
            let cam = tails[0] + Vec3::new(-3.0, 0.0, 1.0);
            (packed(&sc.simple_lights_with_forge(cam, 0.0)).into_iter().map(|s| s.to_vec()).collect(), tails)
        };
        r.run(2.0, &idle);
        let (coast, tails) = pack(&mut r, &idle);
        let (brake, _) = pack(&mut r, &hand);
        // authored: red (1.0, 0.286, 0.149)-ish colour at intensity 2.0 -> red channel 2.0 * scale
        let red = |recs: &[Vec<f32>]| -> Vec<f32> { recs.iter().filter(|rec| tails.iter().any(|t| (Vec3::new(rec[0], rec[1], rec[2]) - *t).length() < 0.05)).map(|rec| rec[8]).collect() };
        let (rc, rb) = (red(&coast), red(&brake));
        assert_eq!(rb.len(), 2, "both taillights in the uniform: {rb:?} of {} records", brake.len());
        assert!(rc.iter().all(|v| (v - 0.5).abs() < 0.01), "coasting red channel is 2.0 * 0.25: {rc:?}");
        assert!(rb.iter().all(|v| (v - 2.0).abs() < 0.01), "braking red channel is the authored 2.0: {rb:?}");
        // it lights the road at the lamp and does not light a point a long way off
        let rec = brake.iter().find(|rec| (Vec3::new(rec[0], rec[1], rec[2]) - tails[0]).length() < 0.05).unwrap();
        let near = tails[0] + Vec3::new(-0.15, 0.0, 0.0);
        let far = tails[0] + Vec3::new(-3.0, 0.0, 0.0);
        assert!(lit(rec, near) > 0.01, "a point 15 cm behind the lamp must be lit, got {}", lit(rec, near));
        assert_eq!(lit(rec, far), 0.0, "3 wu away is outside the authored 0.35 wu reach");
        // driving off removes it from the uniform once the lights are gone (nobody aboard)
        r.pm.exit_vehicle(&n, c);
        for _ in 0..(40.0 / DT) as usize {
            r.tick(&idle);
            if r.pm.vehicle_fx.frame.vehicles.is_empty() { break; }
        }
        r.pm.vehicle_fx.install(&mut sc, 0.0);
        assert!(sc.dynamic_lights().is_empty(), "the lights outlived the driver");
        assert!(sc.has_forge_lights(), "the uniform must be re-packed once more without them");
        let _ = sc.simple_lights_with_forge(Vec3::ZERO, 0.0);
        assert!(!sc.has_forge_lights(), "...and then not again");
    }
}
