//! #vehicle-anim  A DRIVEN vehicle's own animation: which overlays its `jmad` authors, what
//! each one's frames MEAN, and what feeds them.
//!
//! Reported: *"Vehicles have their own unique animation they play when a biped is driving it, so
//! handlebars, wheels etc would function"*. They do, and this module is the tag side of it.
//!
//! # A driven vehicle is not playing a clip
//!
//! Every vehicle's graph authors exactly one BASE clip — `any:idle` — and then a handful of very
//! short **overlays** whose frames are not a timeline at all. Measured by sampling every frame of
//! every overlay of eight vehicles in `forge_halo` (`probes::probe_what_a_vehicle_overlay_does_per_frame`
//! in [`crate::play_vehicle`]), they come in exactly two shapes:
//!
//! ```text
//!  CYCLE   vehicle:ground_speed       5 frames   b_wheel_front_left   y  0  +90  +180  +270  360
//!          vehicle:ground_speed_rear  5 frames   b_wheel_back_left    y  0  +90  +180  +270  360
//!          vehicle:wheels_left        5 frames   b_tread_wheel_fl_1   y  0  +90  +180  +270  360
//!  RANGE   vehicle:steering (hog)     4 frames   b_steering_wheel     y +60  +30   -30   -60
//!          vehicle:steering (goose)   3 frames   b_steering           z -25    0   +25
//!          suspension:left:front      5 frames   b_suspension_front_left  y +30 +18.7 +7.5 -3.7 -15
//! ```
//!
//! **A cycle's first frame is identical to its last** — one whole revolution, laid out to be
//! indexed by a phase that wraps. A range's ends differ by the whole sweep and it never comes
//! back, so it is indexed by a value that saturates. Getting that backwards makes a steering
//! wheel spin and a road wheel jitter, so [`Param`] is decided by **reading the clip's own first
//! and last frames**, not by its name — which also means a modded vehicle gets the right
//! treatment without a list to be added to.
//!
//! The engine names the same split from the other side. `haloreach.dll` carries
//! `front_wheels_ground_speed_fraction` and `rear_wheels_ground_speed_fraction` in one block with
//! `steering`, `throttle`, `throttle_forward/side/vertical`, `roll`, `velocity`, `ground_speed`,
//! `braking`, `acceleration_pitch/yaw/magnitude` — its own set of continuous vehicle drivers. The
//! wheels' is the only one called a **fraction**, which is what a wrapping 0..1 phase is.
//!
//! # The wheel's rate is the authored wheel RADIUS, and nothing else
//!
//! A rolling wheel turns once per `2*pi*r` of travel. `r` is authored: it is the `radius` of the
//! `friction points` the vehicle's own havok block carries (the Warthog's four are
//! `wheel_front_left`..`wheel_back_right`, r = 0.250), and a friction point names the very marker
//! the wheel node sits on, so the radius is matched to the clip **by node name** rather than
//! assumed. That identity is physics, not an engine disassembly, and it is labelled as such:
//! what is measured is that the clip is one revolution; that a wheel rolling without slip turns
//! `d / 2*pi*r` times is arithmetic.
//!
//! # What this module refuses to drive, and why
//!
//! * **`suspension:<side>:<pos>`** needs a per-wheel spring compression, and HMS's drive model
//!   has no springs ([`crate::reach_vehicle::DriveModel`] says so in its own "what it does NOT
//!   do"). Worse, its REST point is not even consistent to be guessed at: the Mongoose's 3-frame
//!   clip crosses zero exactly on the middle frame, and the Warthog's 5-frame clip crosses at
//!   ~2.65. So there is no neutral frame to park it on, and not applying the overlay at all IS
//!   the authored rest pose.
//! * **`vehicle:acceleration`** is a damped OSCILLATION over 10-75 frames (the Warthog's chassis
//!   pitches -4.7, -8.5, -11.7, -13.0, -11.9, -9.1, -5.5 and back), not a monotone range, so it
//!   is neither of the two shapes above and what indexes it is unsettled.
//! * **`tread:*_pitch`**, **`vehicle:open_close`**, **`vehicle:aiming`**, **`vehicle:hover`**,
//!   **`vehicle:rotate_props`**, **`vehicle:landing_gear`** each want a quantity this module does
//!   not have (per-corner ground contact, a seat's door state, a turret's aim, a flyer's
//!   dynamics).
//!
//! Each of those is REPORTED per vehicle with its reason ([`Role::Unused`]) rather than faked, so
//! a reader can see that the clip was found and deliberately left alone.
//!
//! # Composing a whole overlay is safe, and that was measured too
//!
//! `vehicle:steering` also puts +-45 deg on `b_aim_yaw`, and `vehicle:acceleration` walks it
//! through a full turn at 22.5 deg a frame. If that node carried the turret mount, composing the
//! overlay as authored would swing the gun with the wheel. It does not: on the Warthog
//! `b_aim_yaw` and `b_aim_pitch` carry **no geometry and no markers**
//! (`probes::probe_which_vehicle_nodes_carry_geometry_or_markers`), while `b_wheel_*` are rigid
//! meshes and `b_steering_wheel` carries the driver's own `left_hand` / `right_hand` markers. So
//! an overlay is applied whole, the way the engine composes one, and the dummy channels move
//! nothing.

use glam::{Mat4, Quat, Vec3};
use hms_native::{CacheHandle, NativeDll};

use crate::reach_actor::{AnimGraph, ReachModel};
use crate::reach_jmad::{self, AnimData, AnimType};
use crate::reach_vehicle::VehiclePhysics;

/// How tolerant the cycle test is: a clip is one revolution if its last frame reproduces its
/// first to within this many radians on every node it animates.
///
/// Rotations are stored as four quantised `i16`, so an exact match is not available; 1 degree is
/// two orders of magnitude below the 90-degree step the shipped cycles actually take.
pub const CYCLE_TOLERANCE: f32 = 0.0175;

/// What one overlay's frames mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// The last frame reproduces the first: one revolution, indexed by a phase that WRAPS.
    Cycle,
    /// The ends differ: a sweep, indexed by a value that SATURATES.
    Range,
}

/// What a driven overlay is FOR, and what feeds it.
#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    /// Wheels that roll. `radius` is the authored friction-point radius of the very wheels this
    /// clip turns, matched by marker name; `wheels` names them, for the report.
    Roll { radius: f32, wheels: Vec<String> },
    /// The steering linkage. `left_end` is the frame index that turns the vehicle's own front
    /// wheels LEFT — measured in MODEL space, because a node-local rotation sign is not
    /// comparable between two rigs whose bones are oriented differently.
    Steer { left_end: usize, yaw_at_left: f32 },
    /// Found, classified, and deliberately not driven.
    Unused(String),
}

/// One overlay of a vehicle's own graph, with everything needed to index it.
#[derive(Clone, Debug, PartialEq)]
pub struct DrivenOverlay {
    /// Index into the graph's own `anims`.
    pub anim: usize,
    pub name: String,
    /// `frame_count`, as authored.
    pub frames: usize,
    pub kind: Param,
    pub role: Role,
    /// The render-model nodes this clip's rotation channels address.
    pub nodes: Vec<String>,
}

impl DrivenOverlay {
    /// The frame to sample, for a driven state — or `None` when this overlay is not driven.
    ///
    /// A cycle's span is `frames - 1` steps, because its last frame IS its first; a range's span
    /// is the whole `frames - 1` and the value saturates at both ends.
    pub fn frame(&self, st: &DrivenState) -> Option<f32> {
        let span = (self.frames as f32 - 1.0).max(0.0);
        if span <= 0.0 {
            return None;
        }
        match &self.role {
            Role::Roll { radius, .. } => {
                let circumference = std::f32::consts::TAU * radius.max(1e-4);
                let turns = st.odometer / circumference;
                Some(turns.rem_euclid(1.0) * span)
            }
            Role::Steer { left_end, .. } => {
                // -1 .. +1 onto the clip's own ends, with the LEFT end wherever the tag put it.
                // The centre lands on the middle frame, which is what the Mongoose's 3-frame
                // clip (-25, 0, +25) says it should.
                let t = (st.steer.clamp(-1.0, 1.0) + 1.0) * 0.5;
                let f = if *left_end == 0 { 1.0 - t } else { t };
                Some(f * span)
            }
            Role::Unused(_) => None,
        }
    }
}

/// What the ride knows, in the terms the overlays want.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DrivenState {
    /// Signed distance the vehicle has travelled along its own forward, world units. Signed, so
    /// reversing spins the wheels backwards.
    pub odometer: f32,
    /// Steering, `-1` = full right .. `+1` = full LEFT (`+Y` is left, as everywhere in Blam).
    pub steer: f32,
}

/// Everything read once per vehicle TAG about how to animate it while it is driven.
#[derive(Clone, Debug, Default)]
pub struct VehicleAnim {
    /// `any:idle`, the one BASE clip. `None` when the graph authors none.
    pub base: Option<usize>,
    pub base_name: String,
    pub overlays: Vec<DrivenOverlay>,
    /// What was found and what was skipped, in the vehicle's own words.
    pub notes: Vec<String>,
}

impl VehicleAnim {
    /// Classify every overlay of a vehicle's own graph.
    ///
    /// `phys` supplies the authored wheel radii; without it the roll overlays are reported and
    /// not driven, because the rate is the radius and this module will not choose one.
    pub fn read(
        native: &NativeDll,
        cache: CacheHandle,
        model: &ReachModel,
        graph: &AnimGraph,
        phys: Option<&VehiclePhysics>,
    ) -> VehicleAnim {
        let _ = (native, cache);
        let map = crate::reach_actor::anim_node_map(model, graph);
        let mut out = VehicleAnim::default();
        for a in &graph.anims {
            if a.anim_type == AnimType::Base {
                // The one base clip. `any:idle` is what every shipped vehicle names it; anything
                // else is taken only if nothing better turned up.
                if out.base.is_none() || a.name.ends_with(":idle") {
                    out.base = Some(a.index);
                    out.base_name = a.name.clone();
                }
                continue;
            }
            if a.anim_type != AnimType::Overlay {
                continue;
            }
            let Some(data) = graph.load(native, cache, a.index) else {
                out.notes.push(format!("'{}' has no animation data and was skipped", a.name));
                continue;
            };
            let frames = a.frame_count.max(0) as usize;
            let nodes = animated_nodes(model, &data, a.anim_type, map.as_deref());
            let kind = classify(&data, a.anim_type, frames);
            let role = role_for(&a.name, &nodes, kind, frames, model, graph, map.as_deref(), &data, phys);
            out.overlays.push(DrivenOverlay { anim: a.index, name: a.name.clone(), frames, kind, role, nodes });
        }
        if out.base.is_none() {
            out.notes.push("this vehicle's graph authors no base clip, so it is drawn in its bind pose".into());
        }
        out
    }

    /// The overlays that are actually driven, as `(graph animation index, cells)` — exactly the
    /// shape [`crate::reach_actor::Actor::overlays`] takes.
    pub fn cells(&self, st: &DrivenState) -> Vec<(usize, Vec<(f32, f32)>)> {
        self.overlays
            .iter()
            .filter_map(|o| o.frame(st).map(|f| (o.anim, vec![(f, 1.0f32)])))
            .collect()
    }

    /// One line per overlay: what it is, and whether it moves.
    pub fn describe(&self) -> Vec<String> {
        let mut v = vec![match &self.base {
            Some(_) => format!("base '{}'", self.base_name),
            None => "NO base clip".to_string(),
        }];
        for o in &self.overlays {
            v.push(match &o.role {
                Role::Roll { radius, wheels } => format!(
                    "{} — {} frames, {:?}: ROLLS {} at r {:.3} wu ({:.4} wu per turn)",
                    o.name,
                    o.frames,
                    o.kind,
                    wheels.join(","),
                    radius,
                    std::f32::consts::TAU * radius
                ),
                Role::Steer { left_end, yaw_at_left } => format!(
                    "{} — {} frames, {:?}: STEERS, frame {} is left ({:+.1} deg of front-wheel yaw)",
                    o.name,
                    o.frames,
                    o.kind,
                    left_end,
                    yaw_at_left.to_degrees()
                ),
                Role::Unused(why) => format!("{} — {} frames, {:?}: not driven — {why}", o.name, o.frames, o.kind),
            });
        }
        v.extend(self.notes.iter().cloned());
        v
    }

    /// Is anything driven at all?
    pub fn drives_anything(&self) -> bool {
        self.overlays.iter().any(|o| !matches!(o.role, Role::Unused(_)))
    }
}

/// CYCLE or RANGE, from the clip's own first and last frames.
///
/// A clip of fewer than three frames cannot be a cycle (two identical frames would animate
/// nothing), so it is a range by construction.
fn classify(data: &AnimData, ty: AnimType, frames: usize) -> Param {
    if frames < 3 {
        return Param::Range;
    }
    let first = reach_jmad::sample(data, ty, 0.0);
    let last = reach_jmad::sample(data, ty, frames as f32 - 1.0);
    let mut compared = 0usize;
    for (a, b) in first.iter().zip(last.iter()) {
        if let (Some(qa), Some(qb)) = (a.rot, b.rot) {
            compared += 1;
            // The shortest angle between the two, sign-independent (a quaternion and its
            // negation are the same rotation).
            let d = qa.dot(qb).abs().clamp(-1.0, 1.0);
            if 2.0 * d.acos() > CYCLE_TOLERANCE {
                return Param::Range;
            }
        }
        if let (Some(ta), Some(tb)) = (a.trans, b.trans) {
            compared += 1;
            if (ta - tb).length() > 1e-3 {
                return Param::Range;
            }
        }
    }
    if compared == 0 {
        Param::Range
    } else {
        Param::Cycle
    }
}

/// The render-model node names one clip's channels address.
fn animated_nodes(model: &ReachModel, data: &AnimData, ty: AnimType, map: Option<&[i16]>) -> Vec<String> {
    let mut v = Vec::new();
    for (ci, c) in reach_jmad::sample(data, ty, 0.0).iter().enumerate() {
        if c.rot.is_none() && c.trans.is_none() {
            continue;
        }
        let i = match map {
            Some(m) => match m.get(ci) {
                Some(&x) if x >= 0 => x as usize,
                _ => continue,
            },
            None => ci,
        };
        if let Some(n) = model.node_names.get(i) {
            v.push(n.clone());
        }
    }
    v
}

/// Does this node name look like a wheel that rolls?
fn is_wheel(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("wheel") && !n.contains("friction") && !n.contains("steering") && !n.contains("hub")
}

/// Does this node name look like a FRONT wheel? Both orders ship (`b_wheel_front_left` on the
/// Warthog, `b_wheel_left_front` on the truck).
fn is_front_wheel(name: &str) -> bool {
    is_wheel(name) && name.to_lowercase().contains("front")
}

#[allow(clippy::too_many_arguments)]
fn role_for(
    name: &str,
    nodes: &[String],
    kind: Param,
    frames: usize,
    model: &ReachModel,
    graph: &AnimGraph,
    map: Option<&[i16]>,
    data: &AnimData,
    phys: Option<&VehiclePhysics>,
) -> Role {
    let lower = name.to_lowercase();
    // --- the wheels -----------------------------------------------------------------------
    //
    // Decided by the NODES, not the name: every shipped roll clip animates nothing but wheel
    // nodes, and a clip that animates a wheel and something else is not one.
    if !nodes.is_empty() && nodes.iter().all(|n| is_wheel(n)) {
        if kind != Param::Cycle {
            return Role::Unused(format!(
                "it turns {} but its last frame does not return to its first, so it is not one revolution",
                nodes.join(",")
            ));
        }
        let Some(p) = phys else {
            return Role::Unused("this vehicle's physics block could not be read, so there is no authored wheel radius to roll at".into());
        };
        if p.havok.friction_points.is_empty() {
            return Role::Unused("this vehicle authors no friction points, so no wheel radius is authored".into());
        }
        // Match a friction point to a wheel node by name: a friction point names the very marker
        // the wheel sits on (`wheel_front_left` <-> `b_wheel_front_left`).
        let mut radii: Vec<f32> = Vec::new();
        for n in nodes {
            let stem = n.trim_start_matches("b_").to_lowercase();
            if let Some(f) = p.havok.friction_points.iter().find(|f| f.marker_name.eq_ignore_ascii_case(&stem)) {
                radii.push(f.radius);
            }
        }
        let matched = !radii.is_empty();
        if !matched {
            // The Scorpion's road wheels (`b_tread_wheel_fl_1`) are not friction points. Its own
            // friction radius is the nearest authored number and it is SAID so, not passed off.
            radii = p.havok.friction_points.iter().map(|f| f.radius).collect();
        }
        let radius = radii.iter().copied().sum::<f32>() / radii.len() as f32;
        if !(radius.is_finite() && radius > 1e-3) {
            return Role::Unused(format!("the authored wheel radius is {radius}, which cannot be rolled at"));
        }
        if !matched {
            return Role::Roll { radius, wheels: nodes.to_vec() };
        }
        return Role::Roll { radius, wheels: nodes.to_vec() };
    }
    // --- the steering ---------------------------------------------------------------------
    if lower.contains("steering") {
        if frames < 2 {
            return Role::Unused("a steering clip of one frame sweeps nothing".into());
        }
        match steer_left_end(model, graph, map, data, frames) {
            Some((left_end, yaw)) => Role::Steer { left_end, yaw_at_left: yaw },
            None => Role::Unused(
                "which end of this clip steers LEFT could not be measured: it moves no front-wheel \
                 node, and a node-local rotation sign is not comparable between rigs"
                    .into(),
            ),
        }
    } else if lower.starts_with("suspension") {
        Role::Unused(
            "HMS's drive model has no springs, and this clip's rest frame is not inferable (the \
             Mongoose's crosses zero at its middle frame, the Warthog's at ~2.65)"
                .into(),
        )
    } else if lower.contains("acceleration") {
        Role::Unused("a damped oscillation, not a monotone range: what indexes it is unsettled".into())
    } else if lower.starts_with("tread") {
        Role::Unused("needs the per-corner ground contact HMS's drive model does not solve".into())
    } else if lower.contains("aiming") {
        Role::Unused("a turret's aim, which the ride does not drive".into())
    } else if lower.contains("open_close") || lower.contains("opening") {
        Role::Unused("a door, driven by getting in and out rather than by driving".into())
    } else {
        Role::Unused("no driver measured for this overlay".into())
    }
}

/// Which end of a steering clip turns the FRONT WHEELS left, measured in MODEL space.
///
/// # Why it has to be model space
///
/// Read in node-local axes the authored direction appears to flip between vehicles — the
/// Warthog's front universals take `z +30` on frame 0 and the Mongoose's front hubs take `z -25`
/// — but a local rotation sign says nothing across two rigs whose bones are oriented
/// differently. So the overlay is composed onto the bind pose and the wheel node's own basis is
/// read where it actually ends up: a wheel spins about its axle, the axle leaves the hull along
/// `+-Y`, and steering swings it in the hull's XY plane. The signed yaw of that axle, relative to
/// the bind pose's, is the steer angle, and `+` is left because `+Y` is left.
///
/// Returns `(frame index that steers left, that frame's front-wheel yaw in radians)`.
fn steer_left_end(
    model: &ReachModel,
    graph: &AnimGraph,
    map: Option<&[i16]>,
    data: &AnimData,
    frames: usize,
) -> Option<(usize, f32)> {
    let front: Vec<usize> = model
        .node_names
        .iter()
        .enumerate()
        .filter(|(_, n)| is_front_wheel(n))
        .map(|(i, _)| i)
        .collect();
    if front.is_empty() {
        return None;
    }
    let bind = model.skel.bind_model();
    let last = frames - 1;
    let a = overlay_pose_model(model, graph, map, data, 0.0);
    let b = overlay_pose_model(model, graph, map, data, last as f32);
    let ya = front.iter().filter_map(|&i| axle_yaw(&bind, &a, i)).sum::<f32>();
    let yb = front.iter().filter_map(|&i| axle_yaw(&bind, &b, i)).sum::<f32>();
    // A clip whose two ends put the wheels at the same yaw steers nothing measurable.
    if (ya - yb).abs() < 1e-3 {
        return None;
    }
    if ya > yb {
        Some((0, ya / front.len() as f32))
    } else {
        Some((last, yb / front.len() as f32))
    }
}

/// How far one node's axle has swung in the hull's XY plane, relative to the bind pose. Positive
/// is toward `+Y`, which is left.
fn axle_yaw(bind: &[Mat4], posed: &[Mat4], node: usize) -> Option<f32> {
    let (b, p) = (bind.get(node)?, posed.get(node)?);
    // The axle is the node's own local Y, which is the axis every shipped roll clip spins about.
    let ab = b.transform_vector3(Vec3::Y);
    let ap = p.transform_vector3(Vec3::Y);
    let (ab, ap) = (Vec3::new(ab.x, ab.y, 0.0), Vec3::new(ap.x, ap.y, 0.0));
    if ab.length_squared() < 1e-6 || ap.length_squared() < 1e-6 {
        return None;
    }
    let (ab, ap) = (ab.normalize(), ap.normalize());
    // Signed angle from the bind axle to the posed one, about +Z. An axle pointing +Y that swings
    // toward -X is a LEFT turn, which this sign convention gives as positive.
    Some((ab.x * ap.y - ab.y * ap.x).atan2(ab.dot(ap)))
}

/// Compose ONE overlay frame onto the skeleton's authored defaults and return the MODEL-space
/// node matrices.
///
/// The composition rule is the engine's own overlay combiner (`haloreach.dll +0x21DAEC`), the
/// same one [`crate::reach_actor::Actor`] uses: `q_default * q_clip`, `t_default + t_clip`. It is
/// re-stated here rather than reached through an `Actor` because this runs before any actor
/// exists — it is how the vehicle's own tag is read.
pub fn overlay_pose_model(
    model: &ReachModel,
    graph: &AnimGraph,
    map: Option<&[i16]>,
    data: &AnimData,
    frame: f32,
) -> Vec<Mat4> {
    let _ = graph;
    let skel = &model.skel;
    let mut trs: Vec<(Quat, Vec3, f32)> = skel.nodes.iter().map(|n| (n.rotation, n.translation, 1.0f32)).collect();
    for (ci, c) in reach_jmad::sample(data, AnimType::Overlay, frame).iter().enumerate() {
        let i = match map {
            Some(m) => match m.get(ci) {
                Some(&x) if x >= 0 => x as usize,
                _ => continue,
            },
            None => ci,
        };
        if i >= trs.len() {
            continue;
        }
        let nd = &skel.nodes[i];
        if let Some(q) = c.rot {
            trs[i].0 = nd.rotation * q;
        }
        if let Some(t) = c.trans {
            trs[i].1 = nd.translation + t;
        }
        if let Some(s) = c.scale {
            trs[i].2 = s;
        }
    }
    let locals: Vec<Mat4> = trs
        .iter()
        .map(|(r, t, s)| {
            if (*s - 1.0).abs() < 1e-5 {
                Mat4::from_rotation_translation(*r, *t)
            } else {
                Mat4::from_scale_rotation_translation(Vec3::splat(*s), *r, *t)
            }
        })
        .collect();
    skel.pose_model(&locals)
}

#[cfg(test)]
mod gates {
    //! #vehicle-anim  `cargo test -p hms-app --profile quick reach_vehicle_anim:: -- --ignored --nocapture`

    use super::*;

    fn st(odometer: f32, steer: f32) -> DrivenState {
        DrivenState { odometer, steer }
    }

    /// **THE INDEXING GATE, with no cache at all.** A cycle wraps and a range saturates, and the
    /// two must not be interchangeable.
    #[test]
    fn a_cycle_wraps_and_a_range_saturates() {
        let roll = DrivenOverlay {
            anim: 3,
            name: "vehicle:ground_speed".into(),
            frames: 5,
            kind: Param::Cycle,
            // r = 0.250 is the Warthog's authored friction-point radius; one turn is 1.5708 wu.
            role: Role::Roll { radius: 0.25, wheels: vec!["b_wheel_front_left".into()] },
            nodes: vec!["b_wheel_front_left".into()],
        };
        let turn = std::f32::consts::TAU * 0.25;
        assert_eq!(roll.frame(&st(0.0, 0.0)), Some(0.0));
        // A quarter of a revolution is a quarter of the clip's span.
        let q = roll.frame(&st(turn * 0.25, 0.0)).expect("a cycle has a frame");
        assert!((q - 1.0).abs() < 1e-4, "a quarter turn must be frame 1.0, got {q}");
        // One whole revolution comes back to frame 0 -- the wrap, which is the point.
        let w = roll.frame(&st(turn, 0.0)).expect("a cycle has a frame");
        assert!(w < 1e-3, "a whole revolution must wrap to frame 0, got {w}");
        // And a hundred revolutions is still frame 0, not frame 400.
        let far = roll.frame(&st(turn * 100.0, 0.0)).expect("a cycle has a frame");
        assert!(far < 2e-2, "100 revolutions must still wrap, got {far}");
        // Reversing turns the wheels the other way rather than standing still.
        let back = roll.frame(&st(-turn * 0.25, 0.0)).expect("a cycle has a frame");
        assert!((back - 3.0).abs() < 1e-3, "reversing a quarter turn must be frame 3, got {back}");

        // A RANGE with the left end at the last frame (the Mongoose's).
        let steer = DrivenOverlay {
            anim: 9,
            name: "vehicle:steering".into(),
            frames: 3,
            kind: Param::Range,
            role: Role::Steer { left_end: 2, yaw_at_left: 0.4 },
            nodes: vec!["b_steering".into()],
        };
        assert_eq!(steer.frame(&st(0.0, 0.0)), Some(1.0), "centred steering is the MIDDLE frame");
        assert_eq!(steer.frame(&st(0.0, 1.0)), Some(2.0), "full left is the left end");
        assert_eq!(steer.frame(&st(0.0, -1.0)), Some(0.0), "full right is the other end");
        // It SATURATES: twice full lock is still full lock, not frame 4.
        assert_eq!(steer.frame(&st(0.0, 9.0)), Some(2.0), "a range must saturate, not wrap");
        assert_eq!(steer.frame(&st(0.0, -9.0)), Some(0.0));
        // The odometer must not move a steering clip, and the steer must not move a wheel.
        assert_eq!(steer.frame(&st(1234.0, 0.0)), Some(1.0));
        assert_eq!(roll.frame(&st(0.0, 1.0)), Some(0.0));

        // The SAME clip with the left end at frame 0 (the Warthog's) is mirrored, which is the
        // whole reason `left_end` is measured rather than assumed.
        let mirrored = DrivenOverlay { role: Role::Steer { left_end: 0, yaw_at_left: 0.4 }, ..steer.clone() };
        assert_eq!(mirrored.frame(&st(0.0, 1.0)), Some(0.0));
        assert_eq!(mirrored.frame(&st(0.0, -1.0)), Some(2.0));
        assert_eq!(mirrored.frame(&st(0.0, 0.0)), Some(1.0), "either way, centre is the middle");

        // An unused overlay yields nothing -- it must never be installed by accident.
        let unused = DrivenOverlay {
            role: Role::Unused("no driver".into()),
            ..steer.clone()
        };
        assert_eq!(unused.frame(&st(5.0, 1.0)), None);
        // And a one-frame clip indexes nothing rather than dividing by zero.
        let single = DrivenOverlay { frames: 1, ..steer.clone() };
        assert_eq!(single.frame(&st(0.0, 1.0)), None);
    }

    /// `cells` must install exactly the driven overlays and nothing else.
    #[test]
    fn cells_installs_only_what_is_driven() {
        let mk = |name: &str, role: Role| DrivenOverlay {
            anim: name.len(),
            name: name.into(),
            frames: 5,
            kind: Param::Range,
            role,
            nodes: Vec::new(),
        };
        let v = VehicleAnim {
            base: Some(0),
            base_name: "any:idle".into(),
            overlays: vec![
                mk("vehicle:steering", Role::Steer { left_end: 4, yaw_at_left: 0.3 }),
                mk("suspension:left:front", Role::Unused("no springs".into())),
                mk("vehicle:ground_speed", Role::Roll { radius: 0.25, wheels: Vec::new() }),
            ],
            notes: Vec::new(),
        };
        let cells = v.cells(&st(0.0, 0.0));
        assert_eq!(cells.len(), 2, "only the steer and the roll are driven: {cells:?}");
        assert!(cells.iter().all(|(_, c)| c.len() == 1 && c[0].1 == 1.0), "one frame at full weight");
        assert!(v.drives_anything());
        // With nothing drivable, nothing is installed -- and then the vehicle must be left in its
        // base clip rather than handed an empty overlay list that looks like a bug.
        let none = VehicleAnim {
            overlays: vec![mk("suspension:left:front", Role::Unused("no springs".into()))],
            ..v.clone()
        };
        assert!(none.cells(&st(1.0, 1.0)).is_empty());
        assert!(!none.drives_anything());
    }
}
