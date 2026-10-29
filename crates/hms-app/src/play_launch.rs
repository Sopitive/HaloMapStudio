//! #man-cannon  Phantom volumes: man cannons, grav lifts, and every other "push whatever is inside
//! this box" object, in test mode.
//!
//! ## What the engine does (all from `reach_tag_test.exe`, which carries the engine WITH symbols)
//!
//! There is no man-cannon tag group and no script. A man cannon is a plain `bloc` (crate) whose
//! `phmo` carries a **phantom type** (`physics_model_phantom_type`, `phmo +0x38`, 0x6C bytes per
//! element, `physics_model_definitions.cpp`) and a **phantom shape** (`phmo +0x190`, 0x58 bytes).
//! The shape is a Havok phantom: every body that overlaps it is put on the owner's
//! `bodies_in_phantom` list (`sub_140434B40`, filtered by `sub_1404415D0` / `sub_140441750`), and
//! every tick the owner's update walks that list (`sub_14044DD50`) and calls `sub_14044C930` for
//! each entry. That function is the whole law:
//!
//! * the marker named by the type (`marker name` @0xC) on the owner's render model gives a
//!   point `M` and the marker's FORWARD axis `F`;
//! * up to three motions, each an acceleration and a maximum speed: **center** (toward `M`),
//!   **axis** (toward the line through `M` along `F`) and **direction** (along `F`);
//! * each motion's per-tick velocity change comes from `sub_14034C5A0` ([`motion_dv`]), and the
//!   sum is ADDED to the object's velocity (`object_apply_acceleration`, which adds to the rigid
//!   body's linear velocity, `sub_14044A650`) -- a velocity change per 60 Hz tick, never a
//!   position move and never a velocity set;
//! * `negates gravity` adds `g * dt` back every tick; a centre motion that is not saturated adds
//!   back `min(g * dt, slack)`.
//!
//! What a biped does with the new velocity is the ordinary character controller's business: it
//! leaves the ground the moment `dot(n, v)` passes the adhesion speed (`play_ctl::ground_plane`),
//! and the airborne branch of `control_velocity` is the air control. Nothing here moves a body.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};

use crate::physics_settle::TICK_HZ;

/// One engine tick, seconds (`game_time_globals +8`; 60 Hz in MCC, see [`TICK_HZ`]).
pub const TICK_SECONDS: f32 = 1.0 / TICK_HZ;

/// `phantom_flags` bit numbers (`physics_model_definitions.cpp` field names, in order).
pub mod flag {
    pub const USE_ACC_AS_FORCE: u32 = 1 << 1;
    pub const NEGATES_GRAVITY: u32 = 1 << 2;
    pub const IGNORES_PLAYERS: u32 = 1 << 3;
    pub const IGNORES_NONPLAYERS: u32 = 1 << 4;
    /// Bit `5 + object type`: biped 5, vehicle 6, weapon 7, equipment 8, garbage 9, projectile 10,
    /// scenery 11, machine 12, control 13, sound scenery 14, crate 15, creature 16, giant 17,
    /// effect scenery 18 (`sub_1404415D0`: `1 << (type + 5)`).
    pub const IGNORES_TYPE_BASE: u32 = 5;
    pub const IGNORES_GROUNDED_BIPEDS: u32 = 1 << 23;
    pub const DISABLE_LINEAR_DAMPING: u32 = 1 << 25;
    pub const IGNORES_DEAD_BIPEDS: u32 = 1 << 27;
    pub const MIRRORED_AXIS: u32 = 1 << 31;
}

/// An `obje` type, in the engine's order (the index `sub_1404415D0` shifts by).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjType {
    Biped = 0,
    Vehicle = 1,
    Weapon = 2,
    Equipment = 3,
    Projectile = 5,
    Scenery = 6,
    Machine = 7,
    Control = 8,
    SoundScenery = 9,
    Crate = 10,
    Creature = 11,
    Giant = 12,
    EffectScenery = 13,
}

impl ObjType {
    pub fn from_group(group: &str) -> Option<ObjType> {
        Some(match group {
            "bipd" => ObjType::Biped,
            "vehi" => ObjType::Vehicle,
            "weap" => ObjType::Weapon,
            "eqip" => ObjType::Equipment,
            "proj" => ObjType::Projectile,
            "scen" => ObjType::Scenery,
            "mach" => ObjType::Machine,
            "ctrl" => ObjType::Control,
            "ssce" => ObjType::SoundScenery,
            "bloc" => ObjType::Crate,
            "crea" => ObjType::Creature,
            "gint" => ObjType::Giant,
            "efsc" => ObjType::EffectScenery,
            _ => return None,
        })
    }
}

/// `physics_model_phantom_type` (0x6C bytes at `phmo +0x38`), the fields the law reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhantomType {
    pub flags: u32,
    pub marker: u32,
    pub hooke: f32,
    pub dead_radius: f32,
    pub center: (f32, f32),
    pub axis: (f32, f32),
    pub direction: (f32, f32),
}

pub const PHANTOM_TYPE_ELEM: usize = 0x6C;

fn f32_at(d: &[u8], o: usize) -> f32 { d.get(o..o + 4).map_or(0.0, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])) }
fn u32_at(d: &[u8], o: usize) -> u32 { d.get(o..o + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])) }
fn i16_at(d: &[u8], o: usize) -> i16 { d.get(o..o + 2).map_or(0, |b| i16::from_le_bytes([b[0], b[1]])) }

impl PhantomType {
    pub fn parse(e: &[u8]) -> PhantomType {
        PhantomType {
            flags: u32_at(e, 0x00),
            marker: u32_at(e, 0x0C),
            hooke: f32_at(e, 0x1C),
            dead_radius: f32_at(e, 0x20),
            center: (f32_at(e, 0x24), f32_at(e, 0x28)),
            axis: (f32_at(e, 0x2C), f32_at(e, 0x30)),
            direction: (f32_at(e, 0x34), f32_at(e, 0x38)),
        }
    }

    /// Whether this type accelerates anything at all.
    pub fn pushes(&self) -> bool {
        [self.center, self.axis, self.direction].iter().any(|&(a, m)| a != 0.0 && m != 0.0)
    }
}

/// `sub_14034C5A0`: one motion's velocity change for ONE tick.
///
/// `dist` is the distance the motion pulls along (to the centre, to the axis; 10000 for the
/// direction motion), `vel` the body's speed along the pull, `vmax` the speed it stops
/// accelerating at. Returns the change to ADD to `vel`.
pub fn motion_dv(use_force: bool, mass: f32, dist: f32, acc: f32, vel: f32, vmax: f32, dead: f32, hooke: f32, no_damp: bool, dt: f32) -> f32 {
    let ad = dist.abs();
    let mut a = if hooke <= 0.001 { acc } else { ad.powf(hooke) * acc };
    if use_force && mass > 0.001 {
        a /= mass;
    }
    let step = dt * a;
    let cap = step.abs();
    let sign = |x: f32| if x == 0.0 { 0.0 } else if x < 0.0 { -1.0 } else { 1.0 };
    let target = if step <= 0.001 {
        // braking or reversed: slide toward -vmax
        (step + vel).max(-vmax)
    } else {
        let t = if no_damp {
            step + vel
        } else if ad <= ((vel.abs() / cap + 1.0) * 0.5 * (dt * vel)).abs() {
            // inside the stopping distance: the speed that arrives exactly
            (2.0 * ad * cap).sqrt() * sign(vel)
        } else {
            sign(dist - sign(dist) * dead) * ad.min(cap) + vel
        };
        if vmax <= t { vmax } else { t }
    };
    (target - vel).max(-cap).min(cap)
}

/// A phantom volume of one object tag, in MODEL space.
#[derive(Clone, Debug)]
pub struct ModelVolume {
    pub pts: Vec<Vec3>,
    pub ty: PhantomType,
    /// The type's marker on the owner's render model: (position, forward), model space. `None` when
    /// the model carries no such group (the engine warns and falls back to the object's up).
    pub marker: Option<(Vec3, Vec3)>,
}

/// The marker group `sid` of render model `mode`, first instance, as (position, forward) in MODEL
/// space: the node's model matrix times the marker's own frame, forward = local +X.
fn marker_frame(native: &NativeDll, cache: CacheHandle, mode: u32, sid: u32, nodes: &[Mat4]) -> Option<(Vec3, Vec3)> {
    if mode == 0 || mode == 0xFFFF_FFFF || sid == 0 || sid == 0xFFFF_FFFF {
        return None;
    }
    let h = native.tag_meta(cache, mode, 0x3C, 8)?;
    let (gcnt, gptr) = (u32_at(&h, 0), u32_at(&h, 4));
    if gcnt == 0 || gcnt > 4096 {
        return None;
    }
    let g = native.tag_ptr_bytes(cache, gptr, gcnt * 0x10)?;
    let gi = (0..gcnt as usize).find(|&i| u32_at(&g, i * 0x10) == sid)?;
    let (mcnt, mptr) = (u32_at(&g, gi * 0x10 + 4), u32_at(&g, gi * 0x10 + 8));
    if mcnt == 0 || mcnt > 4096 {
        return None;
    }
    let mk = native.tag_ptr_bytes(cache, mptr, 0x30)?;
    let pos = Vec3::new(f32_at(&mk, 4), f32_at(&mk, 8), f32_at(&mk, 12));
    let rot = glam::Quat::from_xyzw(f32_at(&mk, 0x10), f32_at(&mk, 0x14), f32_at(&mk, 0x18), f32_at(&mk, 0x1C)).normalize();
    let node = mk[2];
    let nm = if node == 0xFF { Mat4::IDENTITY } else { nodes.get(node as usize).copied().unwrap_or(Mat4::IDENTITY) };
    let frame = nm * Mat4::from_rotation_translation(rot, pos);
    let (p, f) = (frame.w_axis.truncate(), frame.x_axis.truncate().normalize_or_zero());
    (p.is_finite() && f.is_finite() && f != Vec3::ZERO).then_some((p, f))
}

/// Every phantom volume of an object tag whose type actually pushes (empty for nearly every tag).
pub fn decode_tag(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Vec<ModelVolume> {
    let phmo = native.resolve_phmo(cache, obj_tag);
    if phmo == 0 || phmo == 0xFFFF_FFFF {
        return Vec::new();
    }
    let Some(meta) = native.tag_meta(cache, phmo, 0, crate::phmo_shapes::REACH_PHMO_META_SIZE as u32) else { return Vec::new() };
    let block = |off: usize, elem: usize| -> Vec<u8> {
        let (n, ptr) = (u32_at(&meta, off) as i32, u32_at(&meta, off + 4));
        if n <= 0 || n > 0x10000 || ptr == 0 || ptr == 0xFFFF_FFFF {
            return Vec::new();
        }
        native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
    };
    let types: Vec<PhantomType> = block(0x38, PHANTOM_TYPE_ELEM).chunks_exact(PHANTOM_TYPE_ELEM).map(PhantomType::parse).collect();
    if !types.iter().any(PhantomType::pushes) {
        return Vec::new();
    }
    let materials = block(0x68, 0x10);
    let phantoms = block(crate::phmo_shapes::OFF_PHANTOMS, crate::phmo_shapes::PHANTOM_ELEM);
    let Some(blocks) = crate::object_hulls::read_blocks(native, cache, phmo) else { return Vec::new() };
    let nodes = crate::object_hulls::mode_node_xforms(native, cache, obj_tag);
    let mode = native.resolve_mode(cache, obj_tag);
    let mut out = Vec::new();
    for v in crate::phmo_shapes::phantom_volumes(&blocks, &crate::phmo_shapes::REACH, &nodes, &phantoms) {
        // phantom TYPE = materials[primitive material].phantom type (i16 @0xC), -1 = none
        let ti = materials.get(v.material as usize * 0x10..).map_or(-1, |m| i16_at(m, 0xC));
        let Some(ty) = usize::try_from(ti).ok().and_then(|i| types.get(i)).copied().filter(PhantomType::pushes) else { continue };
        let crate::physics_settle::ShapeGeom::Hull { pts, .. } = &v.shape.geom else { continue };
        out.push(ModelVolume { pts: pts.clone(), ty, marker: marker_frame(native, cache, mode, ty.marker, &nodes) });
    }
    out
}

/// A phantom volume placed in the world.
#[derive(Clone, Debug)]
pub struct WorldVolume {
    pub ty: PhantomType,
    pts: Vec<Vec3>,
    /// SAT axes: the hull's face normals.
    normals: Vec<Vec3>,
    pub aabb: (Vec3, Vec3),
    pub marker_pos: Vec3,
    pub marker_fwd: Vec3,
}

impl WorldVolume {
    pub fn place(m: &ModelVolume, world: &Mat4, up: Vec3, origin: Vec3) -> Option<WorldVolume> {
        let pts: Vec<Vec3> = m.pts.iter().map(|p| world.transform_point3(*p)).collect();
        let hull = crate::physics_settle::ConvexShape::hull(&pts, 0.0, crate::physics_settle::ShapeMaterial { friction: 0.0, restitution: 0.0, mass: 0.0 })?;
        let crate::physics_settle::ShapeGeom::Hull { pts, planes, .. } = hull.geom else { return None };
        let aabb = pts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let (marker_pos, marker_fwd) = match m.marker {
            Some((p, f)) => (world.transform_point3(p), world.transform_vector3(f).normalize_or_zero()),
            None => (origin, up),
        };
        Some(WorldVolume { ty: m.ty, pts, normals: planes.iter().map(|(n, _)| *n).collect(), aabb, marker_pos, marker_fwd })
    }

    /// Whether the box `(mn, mx)` overlaps this convex volume (separating axis test: the box's
    /// three axes, the hull's face normals and their crosses with the box axes).
    pub fn overlaps(&self, mn: Vec3, mx: Vec3) -> bool {
        if (0..3).any(|k| self.aabb.1[k] < mn[k] || self.aabb.0[k] > mx[k]) {
            return false;
        }
        let (c, h) = ((mn + mx) * 0.5, (mx - mn) * 0.5);
        let separated = |axis: Vec3| -> bool {
            let l = axis.length();
            if l < 1e-6 {
                return false;
            }
            let a = axis / l;
            let r = h.x * a.x.abs() + h.y * a.y.abs() + h.z * a.z.abs();
            let (lo, hi) = self.pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.dot(a)), hi.max(p.dot(a))));
            hi < c.dot(a) - r || lo > c.dot(a) + r
        };
        for n in &self.normals {
            if separated(*n) {
                return false;
            }
            for e in [Vec3::X, Vec3::Y, Vec3::Z] {
                if separated(n.cross(e)) {
                    return false;
                }
            }
        }
        true
    }
}

/// What the phantoms act on.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub kind: ObjType,
    /// A player-controlled biped (`ignores players` / `ignores nonplayers`).
    pub player: bool,
    pub dead: bool,
    /// A biped standing on something (`ignores grounded bipeds`).
    pub grounded: bool,
    pub aabb: (Vec3, Vec3),
    /// The point the centre / axis motions pull (`sub_140D467A0`: the body's centre).
    pub centre: Vec3,
    pub velocity: Vec3,
}

impl PhantomType {
    /// `sub_1404415D0` (object type) and `sub_140441750` (player / dead / grounded): does this
    /// type act on `b` at all.
    pub fn affects(&self, b: &Body) -> bool {
        let f = self.flags;
        if f & (1 << (flag::IGNORES_TYPE_BASE + b.kind as u32)) != 0 {
            return false;
        }
        if f & flag::IGNORES_PLAYERS != 0 && b.player {
            return false;
        }
        if f & flag::IGNORES_NONPLAYERS != 0 && !b.player {
            return false;
        }
        let biped = b.kind == ObjType::Biped;
        if f & flag::IGNORES_DEAD_BIPEDS != 0 && biped && b.dead {
            return false;
        }
        !(f & flag::IGNORES_GROUNDED_BIPEDS != 0 && biped && b.grounded)
    }

    /// A type with `latching disabled` (bit 30) cannot catch a body, only keep one a latching
    /// type caught this tick or last (`sub_14044C930`, flag `0x20000`/`0x10000`).
    pub fn keeps_only(&self) -> bool {
        self.flags & 0x4000_0000 != 0
    }
}

/// `sub_14044C930` for a whole-object body: the velocity change ONE phantom volume applies in one
/// tick. `vel` is the body's velocity at the moment (earlier volumes of the same tick have
/// already been added), `g` the world gravity (`g * dt` is what `negates gravity` gives back).
pub fn volume_dv(ty: &PhantomType, marker_pos: Vec3, marker_fwd: Vec3, centre: Vec3, vel: Vec3, g: f32, dt: f32) -> Vec3 {
    let use_force = ty.flags & flag::USE_ACC_AS_FORCE != 0;
    let no_damp = ty.flags & flag::DISABLE_LINEAR_DAMPING != 0;
    let negates = ty.flags & flag::NEGATES_GRAVITY != 0;
    // a whole object has no rigid body index, so the mass the force flag divides by is 1
    let law = |dist, acc, along, vmax| motion_dv(use_force, 1.0, dist, acc, along, vmax, ty.dead_radius, ty.hooke, no_damp, dt);
    let mut out = Vec3::ZERO;
    let to_marker = marker_pos - centre;

    // CENTER: toward the marker point
    let (acc, vmax) = ty.center;
    if acc != 0.0 && vmax != 0.0 {
        let dist = to_marker.length();
        let dir = if dist > 1e-4 { to_marker / dist } else { Vec3::Z };
        let dist = if dist > 1e-4 { dist } else { 0.0 };
        let along = dir.dot(vel);
        let d1 = law(dist, acc, along, vmax);
        let mut push = dir * d1;
        let slack = dt * acc;
        if !negates && slack > d1 {
            push.z += (g * dt).min(slack - d1);
        }
        if dist - ty.dead_radius >= 0.0 {
            out += push - damp_residual(vel - along * dir, slack);
        } else {
            out += push;
        }
    }

    // AXIS: toward the line through the marker along its forward
    let (acc, vmax) = ty.axis;
    if acc != 0.0 && vmax != 0.0 {
        let perp = to_marker - to_marker.dot(marker_fwd) * marker_fwd;
        let dist = perp.length();
        if dist > 1e-4 {
            let dir = perp / dist;
            let along = dir.dot(vel);
            let d1 = law(dist, acc, along, vmax);
            let mut push = dir * d1;
            if dist - ty.dead_radius >= 0.0 {
                let resid = vel - marker_fwd.dot(vel) * marker_fwd - along * dir;
                push -= damp_residual(resid, dt * acc);
            }
            out += push;
        }
    }

    // DIRECTION: along the marker's forward
    let (mut acc, vmax) = ty.direction;
    if acc != 0.0 && vmax != 0.0 {
        if ty.flags & flag::MIRRORED_AXIS != 0 && to_marker.dot(marker_fwd) > 0.0 {
            acc = -acc;
        }
        out += marker_fwd * law(10000.0, acc, marker_fwd.dot(vel), vmax);
    }

    if negates {
        out.z += g * dt;
    }
    out
}

/// The sideways velocity a motion also removes, at most `limit` (= `dt * acc`) per tick: all of it
/// when it is smaller, else the fraction `limit / |resid|` of it.
fn damp_residual(resid: Vec3, limit: f32) -> Vec3 {
    let len = resid.length();
    if len <= 1e-3 {
        Vec3::ZERO
    } else if len <= limit {
        resid
    } else {
        resid * (1.0 - (len - limit.abs()) / len)
    }
}

/// The world box of a model-space box posed at `origin` / `rot`.
pub fn box_in_world(local: (Vec3, Vec3), origin: Vec3, rot: glam::Quat) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for i in 0..8u32 {
        let c = Vec3::new(
            if i & 1 == 0 { local.0.x } else { local.1.x },
            if i & 2 == 0 { local.0.y } else { local.1.y },
            if i & 4 == 0 { local.0.z } else { local.1.z },
        );
        let w = origin + rot * c;
        lo = lo.min(w);
        hi = hi.max(w);
    }
    (lo, hi)
}

/// Every phantom volume in the map, and which bodies are latched.
#[derive(Default)]
pub struct Launchers {
    by_tag: HashMap<u32, std::sync::Arc<Vec<ModelVolume>>>,
    /// A placed vehicle's hull box in its own frame, by datum (decoding a physics body is slow).
    hulls: HashMap<u32, Option<(Vec3, Vec3)>>,
    pub volumes: Vec<WorldVolume>,
    /// What [`Launchers::refresh`] last built the volumes from.
    sig: u64,
    /// Engine ticks run so far (the latch is "this tick or the one before").
    tick: u64,
    latched_at: HashMap<u64, u64>,
}

/// The world volumes of one placed object, from its tag's model-space volumes and its pose.
fn place_object(o: &ObjectInfo, vols: &[ModelVolume]) -> Vec<WorldVolume> {
    let world = crate::scene::object_matrix(o);
    let up = world.z_axis.truncate().normalize_or_zero();
    vols.iter().filter_map(|m| WorldVolume::place(m, &world, up, Vec3::from(o.pos))).collect()
}

/// The key of the on-foot player in [`Launchers::tick_body`].
pub const PLAYER_KEY: u64 = u64::MAX;

impl Launchers {
    /// Collect the placed objects whose tag carries a pushing phantom. The volumes are rebuilt only
    /// when one of those objects (or the set of them) changed, so a map without launchers costs one
    /// table lookup per object per tick.
    pub fn refresh(&mut self, native: &NativeDll, cache: CacheHandle, objects: &[ObjectInfo]) {
        let mut sig = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |bits: u32| sig = (sig ^ bits as u64).wrapping_mul(0x0100_0000_01b3);
        let mut found: Vec<(&ObjectInfo, std::sync::Arc<Vec<ModelVolume>>)> = Vec::new();
        for o in objects {
            let tag = if o.primary_tag != 0 && o.primary_tag != 0xFFFF_FFFF { o.primary_tag } else { o.mode_tag };
            if tag == 0 || tag == 0xFFFF_FFFF {
                continue;
            }
            let vols = self.by_tag.entry(tag).or_insert_with(|| std::sync::Arc::new(decode_tag(native, cache, tag))).clone();
            if vols.is_empty() {
                continue;
            }
            for b in std::iter::once(o.datum).chain(o.pos.iter().chain(&o.fwd).chain(&o.up).map(|f| f.to_bits())) {
                mix(b);
            }
            found.push((o, vols));
        }
        if sig == self.sig {
            return;
        }
        self.sig = sig;
        self.volumes.clear();
        for (o, vols) in found {
            self.volumes.extend(place_object(o, &vols));
        }
    }

    /// The world volumes ONE placed object carries (empty for an object that launches nothing),
    /// placed exactly as [`Launchers::refresh`] places them. The editor's trajectory preview
    /// asks this to find the catch volume of the cannon you selected.
    pub fn volumes_of(&mut self, native: &NativeDll, cache: CacheHandle, o: &ObjectInfo) -> Vec<WorldVolume> {
        let tag = if o.primary_tag != 0 && o.primary_tag != 0xFFFF_FFFF { o.primary_tag } else { o.mode_tag };
        if tag == 0 || tag == 0xFFFF_FFFF {
            return Vec::new();
        }
        let vols = self.by_tag.entry(tag).or_insert_with(|| std::sync::Arc::new(decode_tag(native, cache, tag))).clone();
        place_object(o, &vols)
    }

    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Does the object tag `tag` carry a phantom that pushes anything (decoded once per tag).
    pub fn is_launcher(&mut self, native: &NativeDll, cache: CacheHandle, tag: u32) -> bool {
        tag != 0 && tag != 0xFFFF_FFFF && !self.by_tag.entry(tag).or_insert_with(|| std::sync::Arc::new(decode_tag(native, cache, tag))).is_empty()
    }

    /// The model-space box of a placed vehicle's physics hull, decoded by `f` once per datum.
    pub fn hull_aabb(&mut self, datum: u32, f: impl FnOnce() -> Option<(Vec3, Vec3)>) -> Option<(Vec3, Vec3)> {
        *self.hulls.entry(datum).or_insert_with(f)
    }

    /// Start one engine tick.
    pub fn begin_tick(&mut self) {
        self.tick += 1;
        if self.tick % 600 == 0 {
            let now = self.tick;
            self.latched_at.retain(|_, t| *t + 1 >= now);
        }
    }

    /// One engine tick of every phantom on the on-foot player: the velocity change goes onto
    /// `st.vel` and `st.in_phantom` is set (the ground update skips its adhesion while a phantom
    /// acts). The one place the player's phantom `Body` is built, shared by test mode's
    /// `run_phantoms` and the editor's man-cannon trajectory preview so the two cannot drift.
    pub fn tick_player(&mut self, st: &mut crate::play_ctl::PlayerState, phys: &crate::play_phys::PlayerPhysics, g: f32) {
        let (p, r, h) = (st.pos, phys.radius, st.height(phys));
        let body = Body {
            kind: ObjType::Biped,
            player: true,
            dead: false,
            grounded: st.grounded,
            aabb: (p - Vec3::new(r, r, 0.0), p + Vec3::new(r, r, h)),
            centre: p + Vec3::Z * (h * 0.5),
            velocity: st.vel,
        };
        let (dv, acted) = self.tick_body(PLAYER_KEY, &body, g);
        st.vel += dv;
        st.in_phantom = acted;
    }

    /// One engine tick of every phantom on one body: the velocity change to ADD to it
    /// (`object_apply_acceleration`) and whether any volume acted. Volumes act in the order they were placed; a type that
    /// cannot latch only acts on a body already latched this tick or the one before.
    pub fn tick_body(&mut self, key: u64, b: &Body, g: f32) -> (Vec3, bool) {
        let prev = self.latched_at.get(&key).is_some_and(|t| t + 1 >= self.tick);
        let mut latched = self.latched_at.get(&key).is_some_and(|t| *t == self.tick);
        let mut vel = b.velocity;
        let mut acted = false;
        for v in &self.volumes {
            if !v.ty.affects(b) || !v.overlaps(b.aabb.0, b.aabb.1) {
                continue;
            }
            if v.ty.keeps_only() && !(latched || prev) {
                continue;
            }
            latched = true;
            acted = true;
            vel += volume_dv(&v.ty, v.marker_pos, v.marker_fwd, b.centre, vel, g, TICK_SECONDS);
        }
        if latched {
            self.latched_at.insert(key, self.tick);
        }
        (vel - b.velocity, acted)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::play_ctl::{MoveInput, PlayerState};
    use crate::play_ledge_gates::{ground_at, level_seeds, spread, MapScene};
    use crate::play_mode::ride_harness::Ride;
    use crate::scene::SceneController;

    /// The engine's world gravity as a LITERAL (physics-constants block, `sub_18014FC94`), not the
    /// constant the code under test reads.
    const G: f32 = 4.171_259_4;

    fn close(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (+-{tol})");
    }

    /// A placed object of `tag` at `pos` facing `yaw`.
    pub(crate) fn placed(tag: u32, datum: u32, pos: Vec3, yaw: f32) -> ObjectInfo {
        ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.to_array(),
            health: 1.0,
            shield: 1.0,
            mode_tag: 0,
            fwd: [yaw.cos(), yaw.sin(), 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: tag,
            variant_name_sid: 0,
        }
    }

    pub(crate) fn tag_ending(native: &NativeDll, cache: CacheHandle, group: &str, suffix: &str) -> Option<u32> {
        native.tags_by_group(cache, group).into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(suffix)))
    }

    const DT: f32 = 1.0 / 60.0;

    fn box_pts(mn: Vec3, mx: Vec3) -> Vec<Vec3> {
        (0..8u32).map(|k| Vec3::new(if k & 4 != 0 { mx.x } else { mn.x }, if k & 2 != 0 { mx.y } else { mn.y }, if k & 1 != 0 { mx.z } else { mn.z })).collect()
    }

    // ---- the law, hand-worked from the decompile of sub_14034C5A0 (dt = 1/60) -------------------

    #[test]
    fn the_motion_law_is_the_engines_hand_worked() {
        // DIRECTION motion of the normal cannon's long volume: acc 11, max 7.5, a pull of 10000
        let dir = |vel: f32| motion_dv(false, 1.0, 10000.0, 11.0, vel, 7.5, 0.0, 0.0, false, DT);
        close(dir(0.0), 11.0 / 60.0, 1e-6, "from rest it gains acc * dt");
        close(dir(7.4), 0.1, 1e-5, "0.1 below the cap it gains only what is left");
        close(dir(7.5), 0.0, 1e-6, "at the cap it gains nothing");
        close(dir(8.5), -11.0 / 60.0, 1e-6, "above the cap it brakes by acc * dt, it never snaps");
        close(dir(-3.0), 11.0 / 60.0, 1e-6, "going the other way it still gains acc * dt");
        // AXIS motion (acc 25, max 20): a body 0.2 off the axis and at rest arrives in ONE tick...
        let axis = |dist: f32, vel: f32| motion_dv(false, 1.0, dist, 25.0, vel, 20.0, 0.0, 0.0, false, DT);
        close(axis(0.2, 0.0), 0.2, 1e-6, "a short pull is the distance itself");
        // ...a long one is capped at acc * dt per tick...
        close(axis(1.0, 0.0), 25.0 / 60.0, 1e-6, "a long pull is acc * dt");
        // ...and a body already closing fast inside its stopping distance is braked, by at most acc * dt
        close(axis(0.1, 5.0), -25.0 / 60.0, 1e-6, "inside the stopping distance it brakes");
        // a NEGATIVE acceleration slides toward -max instead
        close(motion_dv(false, 1.0, 10000.0, -11.0, 0.0, 7.5, 0.0, 0.0, false, DT), -11.0 / 60.0, 1e-6, "reversed");
        close(motion_dv(false, 1.0, 10000.0, -11.0, -7.45, 7.5, 0.0, 0.0, false, DT), -0.05, 1e-5, "reversed, at the cap");
        // `disable linear damping` skips the stopping-distance rule
        close(motion_dv(false, 1.0, 0.1, 25.0, 5.0, 20.0, 0.0, 0.0, true, DT), 25.0 / 60.0, 1e-6, "no damping: always acc * dt");
        // `use acc as force` divides by the mass
        close(motion_dv(true, 2.0, 10000.0, 11.0, 0.0, 7.5, 0.0, 0.0, false, DT), 11.0 / 120.0, 1e-6, "force flag");
    }

    // ---- who a phantom acts on (sub_1404415D0 / sub_140441750) ---------------------------------

    fn body(kind: ObjType, player: bool) -> Body {
        Body { kind, player, dead: false, grounded: false, aabb: (Vec3::ZERO, Vec3::ONE), centre: Vec3::splat(0.5), velocity: Vec3::ZERO }
    }

    #[test]
    fn a_phantom_acts_only_on_what_its_flags_do_not_ignore() {
        // the shipped normal man cannon's two types: ignores equipment, garbage, scenery, machines
        let cannon = PhantomType { flags: 0x1b00, ..Default::default() };
        for (kind, acts) in [
            (ObjType::Biped, true), (ObjType::Vehicle, true), (ObjType::Weapon, true), (ObjType::Projectile, true), (ObjType::Crate, true),
            (ObjType::Equipment, false), (ObjType::Scenery, false), (ObjType::Machine, false),
        ] {
            assert_eq!(cannon.affects(&body(kind, kind == ObjType::Biped)), acts, "man cannon vs {kind:?}");
        }
        // the grav lift's third type (flags 0x08030468) is the PROPS' lift: it ignores players,
        // bipeds, vehicles, projectiles, creatures and giants, and so lifts weapons, crates, scenery
        let props = PhantomType { flags: 0x0803_0468, ..Default::default() };
        assert!(!props.affects(&body(ObjType::Biped, true)) && !props.affects(&body(ObjType::Vehicle, false)));
        assert!(props.affects(&body(ObjType::Weapon, false)) && props.affects(&body(ObjType::Crate, false)) && props.affects(&body(ObjType::Scenery, false)));
        // the vehicle cannon (0x0800fb80) ignores dead bipeds only when they are dead
        let veh = PhantomType { flags: 0x0800_fb80, ..Default::default() };
        let mut b = body(ObjType::Biped, false);
        assert!(veh.affects(&b));
        b.dead = true;
        assert!(!veh.affects(&b), "ignores dead bipeds");
        // player / nonplayer / grounded-biped flags
        let no_players = PhantomType { flags: flag::IGNORES_PLAYERS, ..Default::default() };
        assert!(!no_players.affects(&body(ObjType::Biped, true)) && no_players.affects(&body(ObjType::Biped, false)));
        let no_np = PhantomType { flags: flag::IGNORES_NONPLAYERS, ..Default::default() };
        assert!(no_np.affects(&body(ObjType::Biped, true)) && !no_np.affects(&body(ObjType::Biped, false)));
        let no_grounded = PhantomType { flags: flag::IGNORES_GROUNDED_BIPEDS, ..Default::default() };
        let mut g = body(ObjType::Biped, true);
        g.grounded = true;
        assert!(!no_grounded.affects(&g));
        g.grounded = false;
        assert!(no_grounded.affects(&g));
    }

    // ---- the latch (sub_14044C930, flag 0x20000 / 0x10000) -------------------------------------

    fn synthetic_cannon() -> Launchers {
        let world = Mat4::IDENTITY;
        let small = ModelVolume {
            pts: box_pts(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            ty: PhantomType { flags: 0, direction: (8.0, 5.0), ..Default::default() },
            marker: Some((Vec3::ZERO, Vec3::Z)),
        };
        let long = ModelVolume {
            pts: box_pts(Vec3::new(-1.0, -1.0, 0.5), Vec3::new(1.0, 1.0, 4.0)),
            ty: PhantomType { flags: 0x4000_0000, direction: (11.0, 7.5), ..Default::default() },
            marker: Some((Vec3::ZERO, Vec3::Z)),
        };
        let mut l = Launchers::default();
        for m in [&small, &long] {
            l.volumes.push(WorldVolume::place(m, &world, Vec3::Z, Vec3::ZERO).unwrap());
        }
        l
    }

    fn at(z0: f32, z1: f32) -> Body {
        Body { kind: ObjType::Biped, player: true, dead: false, grounded: false, aabb: (Vec3::new(-0.1, -0.1, z0), Vec3::new(0.1, 0.1, z1)), centre: Vec3::new(0.0, 0.0, (z0 + z1) * 0.5), velocity: Vec3::ZERO }
    }

    #[test]
    fn a_volume_that_cannot_latch_only_keeps_what_a_latching_one_caught() {
        let mut l = synthetic_cannon();
        let tick = |l: &mut Launchers, b: &Body| {
            l.begin_tick();
            l.tick_body(7, b, G)
        };
        // 1. in the long volume ALONE: it never caught this body, so it does nothing
        let (dv, acted) = tick(&mut l, &at(2.0, 2.6));
        assert!(!acted && dv == Vec3::ZERO, "an uncaught body in the long volume was pushed: {dv:?}");
        // 2. in both: the small one catches it and BOTH push (up the marker's +Z)
        let (dv, acted) = tick(&mut l, &at(0.6, 1.2));
        assert!(acted, "the catch volume did not act");
        close(dv.z, 8.0 / 60.0 + 11.0 / 60.0, 1e-4, "both volumes add their acc * dt");
        // 3. moved on into the long volume alone: still caught from last tick, so it keeps pushing
        let (dv, acted) = tick(&mut l, &at(2.0, 2.6));
        assert!(acted, "the latch did not carry to the next tick");
        close(dv.z, 11.0 / 60.0, 1e-4, "the long volume alone, once caught");
        // 4. gone for a tick, back in the long volume alone: the catch has lapsed
        tick(&mut l, &at(8.0, 8.6));
        let (dv, acted) = tick(&mut l, &at(2.0, 2.6));
        assert!(!acted && dv == Vec3::ZERO, "a lapsed latch still pushed: {dv:?}");
    }

    #[test]
    fn a_rotated_volume_is_an_oriented_box_not_its_bounding_box() {
        // a thin slab 2 wu long turned 45 degrees about Z: its world bounding box is +-0.78 in x and y,
        // but the slab itself runs along the diagonal x = y
        let m = ModelVolume {
            pts: box_pts(Vec3::new(-1.0, -0.1, 0.0), Vec3::new(1.0, 0.1, 1.0)),
            ty: PhantomType { direction: (8.0, 5.0), ..Default::default() },
            marker: Some((Vec3::ZERO, Vec3::Z)),
        };
        let v = WorldVolume::place(&m, &Mat4::from_rotation_z(std::f32::consts::FRAC_PI_4), Vec3::Z, Vec3::ZERO).unwrap();
        let probe = |c: Vec3| v.overlaps(c - Vec3::splat(0.05), c + Vec3::splat(0.05));
        assert!(probe(Vec3::new(0.5, 0.5, 0.5)), "a box on the slab must overlap it");
        assert!(probe(Vec3::new(-0.6, -0.6, 0.5)), "...at the other end too");
        assert!(!probe(Vec3::new(0.5, -0.5, 0.5)), "a box inside the slab's bounding box but 0.7 wu off the slab must not");
        assert!(!probe(Vec3::new(0.5, 0.5, 1.5)), "above it must not");
    }

    // ---- the shipped tags (needs the map) ---------------------------------------------------

    /// One expected volume of a shipped launcher: (phantom flags, marker sid, axis (acc, max),
    /// direction (acc, max), marker forward, hull box min..max in MODEL space). Read off the
    /// shipped `forge_halo` tags with a one-off dump of the raw `phmo` blocks (the layout is
    /// `physics_model_phantom_type`, 0x6C bytes, `phmo +0x38`), then written here as literals.
    type Want = (u32, u32, (f32, f32), (f32, f32), [f32; 3], [f32; 3], [f32; 3]);

    const NORMAL: [Want; 2] = [
        (0x1b00, 0xed, (10.0, 6.0), (8.0, 5.0), [-0.5, 0.0, 0.866_025_4], [-0.6588, -0.3588, 0.0149], [0.6860, 0.3584, 0.7836]),
        (0x4000_1b00, 0x2531, (25.0, 20.0), (11.0, 7.5), [-0.819_152_1, 0.0, 0.573_576_4], [-3.5261, -0.3524, 0.4831], [0.1881, 0.3519, 3.1183]),
    ];
    const HEAVY: [Want; 2] = [
        (0x1b00, 0xed, (10.0, 6.0), (8.0, 5.0), [-0.5, 0.0, 0.866_025_4], [-0.6588, -0.3588, 0.0149], [0.6860, 0.3584, 0.7836]),
        (0x4000_1b00, 0x2531, (35.0, 25.0), (16.0, 11.0), [-0.819_152_1, 0.0, 0.573_576_4], [-3.9374, -0.3524, 0.4831], [0.1881, 0.3519, 3.0574]),
    ];
    const LIGHT: [Want; 2] = [
        (0x1b00, 0xed, (10.0, 6.0), (8.0, 5.0), [-0.5, 0.0, 0.866_025_4], [-0.6588, -0.3588, 0.0149], [0.6860, 0.3584, 0.7836]),
        (0x4000_1b00, 0x2531, (25.0, 20.0), (10.0, 8.0), [-0.819_152_1, 0.0, 0.573_576_4], [-2.1799, -0.3524, 0.4831], [0.1881, 0.3519, 2.1692]),
    ];
    const VEHICLE: [Want; 2] = [
        (0x0800_fb80, 0x2d5a, (8.0, 4.0), (30.0, 25.0), [0.0, 0.0, 1.0], [-0.0847, -0.0336, 0.1664], [0.0825, 0.0336, 2.1336]),
        (0x0803_0068, 0x2d5b, (5.0, 2.0), (5.0, 2.0), [0.258_819_3, 0.0, 0.965_925_7], [-0.0847, -0.0336, 0.1664], [0.0825, 0.0336, 2.1336]),
    ];
    const LIFT: [Want; 3] = [
        (0x0800_8b80, 0xed, (8.0, 5.0), (10.0, 7.0), [0.0, 0.0, 1.0], [-0.1847, -0.1836, 0.2194], [0.1825, 0.1836, 2.1866]),
        (0x0800_8b80, 0x2d62, (8.0, 5.0), (10.0, 7.0), [0.0, 0.0, 1.0], [-0.1847, -0.1836, 0.0435], [0.1825, 0.1836, 0.2107]),
        (0x0803_0468, 0x2d5b, (5.0, 2.0), (5.0, 2.0), [0.342_02, 0.0, 0.939_693], [-0.1847, -0.1836, 0.0427], [0.1825, 0.1836, 2.2133]),
    ];

    /// The shipped launcher tags by the end of their path.
    const LAUNCHERS: [(&str, &[Want]); 5] = [
        ("ff_man_cannon_forge\\ff_man_cannon_forge", &NORMAL),
        ("ff_man_cannon_forge_heavy\\ff_man_cannon_forge_heavy", &HEAVY),
        ("ff_man_cannon_forge_light\\ff_man_cannon_forge_light", &LIGHT),
        ("ff_veh_man_cannon\\ff_veh_man_cannon", &VEHICLE),
        ("ff_grav_lift\\ff_grav_lift", &LIFT),
    ];

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn the_shipped_man_cannons_and_grav_lift_decode_to_the_authored_numbers() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let mut volumes = 0;
        for (suffix, want) in LAUNCHERS {
            let tag = tag_ending(&native, cache, "bloc", suffix).unwrap_or_else(|| panic!("{suffix} is not in forge_halo"));
            let got = decode_tag(&native, cache, tag);
            assert_eq!(got.len(), want.len(), "{suffix}: volume count");
            for (g, w) in got.iter().zip(want.iter()) {
                let (flags, sid, axis, dir, fwd, mn, mx) = *w;
                assert_eq!((g.ty.flags, g.ty.marker), (flags, sid), "{suffix}: flags / marker");
                assert_eq!((g.ty.axis, g.ty.direction, g.ty.center), (axis, dir, (0.0, 0.0)), "{suffix}: authored accelerations");
                let (_, f) = g.marker.unwrap_or_else(|| panic!("{suffix}: marker {sid:#x} is not on the render model"));
                for k in 0..3 {
                    close(f[k], fwd[k], 2e-3, &format!("{suffix}: marker forward [{k}]"));
                }
                let (lo, hi) = g.pts.iter().fold((Vec3::splat(1e9), Vec3::splat(-1e9)), |(a, b), p| (a.min(*p), b.max(*p)));
                for k in 0..3 {
                    close(lo[k], mn[k], 2e-3, &format!("{suffix}: hull min [{k}]"));
                    close(hi[k], mx[k], 2e-3, &format!("{suffix}: hull max [{k}]"));
                }
                volumes += 1;
            }
        }
        assert_eq!(volumes, 2 + 2 + 2 + 2 + 3, "liveness: every shipped launcher volume was checked");
    }

    // ---- through the real PlayMode::tick ----------------------------------------------------------

    /// A level spot with 18 wu of flat ground in front of a cannon placed at yaw 0 (the barrel
    /// points toward -X) and 6 wu behind it.
    pub(crate) fn with_site<R>(f: impl FnOnce(&std::sync::Arc<NativeDll>, CacheHandle, &SceneController, Vec3) -> R) -> Option<R> {
        let (_g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let mut sc = SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        sc.load_collision_for_test();
        let site = flat_site(&native, cache, &sc)?;
        Some(f(&native, cache, &sc, site))
    }

    /// A level spot with 18 wu of flat ground all round it (feet position).
    pub(crate) fn flat_site(native: &NativeDll, cache: CacheHandle, sc: &SceneController) -> Option<Vec3> {
        let seeds = spread(level_seeds(native, cache), 4000);
        let flat = MapScene::new(sc, Vec::new());
        let site = seeds.iter().copied().find(|c| {
            let Some(z0) = ground_at(&flat, c.x, c.y, c.z) else { return false };
            (-18..=18).step_by(2).all(|dx| (-18..=18).step_by(2).all(|dy| ground_at(&flat, c.x + dx as f32, c.y + dy as f32, z0).is_some_and(|z| (z - z0).abs() < 0.1)))
        })?;
        let z0 = ground_at(&flat, site.x, site.y, site.z)?;
        Some(Vec3::new(site.x, site.y, z0))
    }

    /// One flight of the player: dropped into the cannon, run for `seconds` at 60 Hz.
    struct Flight {
        /// First tick the phantom stopped acting after acting for at least 20 ticks in a row.
        exit: usize,
        exit_pos: Vec3,
        exit_vel: Vec3,
        apex_z: f32,
        landed: Option<(usize, Vec3)>,
        acted_ticks: usize,
    }

    fn fly_biped(native: &std::sync::Arc<NativeDll>, cache: CacheHandle, sc: &SceneController, site: Vec3, tag: u32, yaw: f32, standing: bool) -> Flight {
        let cannon = placed(tag, 0x7100, site, yaw);
        let scene = MapScene::new(sc, vec![cannon]);
        let mut r = Ride::with_vehicle(native.clone(), cache, "warthog").expect("a Warthog to build the play mode on");
        r.objects = vec![cannon];
        // in the cannon's own frame
        let (c, s) = (yaw.cos(), yaw.sin());
        let to_world = |l: Vec3| site + Vec3::new(l.x * c - l.y * s, l.x * s + l.y * c, l.z);
        // either dropped in from above the pad, or already standing on it (0.5 wu up, the pad's top)
        r.pm.st = PlayerState::standing_at(to_world(Vec3::new(0.0, 0.0, if standing { 0.5 } else { 1.2 })), yaw + std::f32::consts::PI);
        r.pm.st.grounded = standing;
        let idle = MoveInput { yaw: yaw + std::f32::consts::PI, ..Default::default() };
        let (mut run, mut acted, mut exit) = (0usize, 0usize, None);
        let mut apex = f32::MIN;
        let mut landed = None;
        let mut exit_state = (Vec3::ZERO, Vec3::ZERO);
        for t in 0..600 {
            r.tick_in(&idle, &scene);
            let st = &r.pm.st;
            if st.in_phantom {
                run += 1;
                acted += 1;
            } else {
                if run >= 20 && exit.is_none() {
                    exit = Some(t);
                    exit_state = (st.pos, st.vel);
                }
                run = 0;
            }
            if exit.is_some() {
                apex = apex.max(st.pos.z);
                if st.grounded && landed.is_none() {
                    landed = Some((t, st.pos));
                }
            }
        }
        Flight { exit: exit.unwrap_or(usize::MAX), exit_pos: exit_state.0, exit_vel: exit_state.1, apex_z: apex, landed, acted_ticks: acted }
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn a_biped_dropped_into_a_man_cannon_flies_the_authored_arc_in_every_direction() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let tag = tag_ending(native, cache, "bloc", "ff_man_cannon_forge\\ff_man_cannon_forge").expect("the man cannon");
            let mut launches = 0;
            for k in 0..6 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_3 + 0.2;
                let f = fly_biped(native, cache, sc, site, tag, yaw, false);
                eprintln!("yaw {yaw:.2}: exit tick {} at {:?} v {:?} |v| {:.3} apex {:.2} landed {:?} acted {}", f.exit, f.exit_pos - site, f.exit_vel, f.exit_vel.length(), f.apex_z - site.z, f.landed.map(|(t, p)| (t, p - site)), f.acted_ticks);
                assert!(f.exit != usize::MAX, "yaw {yaw}: the player was never launched");
                // the long volume's direction cap (7.5 wu/s, tag) along a barrel 35 degrees above
                // the horizon (marker forward, tag)
                let speed = f.exit_vel.length();
                assert!((7.2..=7.55).contains(&speed), "yaw {yaw}: exit speed {speed}");
                let horiz = Vec3::new(f.exit_vel.x, f.exit_vel.y, 0.0).length();
                let elevation = f.exit_vel.z.atan2(horiz).to_degrees();
                assert!((elevation - 35.0).abs() < 3.0, "yaw {yaw}: elevation {elevation} deg");
                // toward the cannon's local -X, rotated by its yaw
                let want = Vec3::new(-yaw.cos(), -yaw.sin(), 0.0);
                let along = Vec3::new(f.exit_vel.x, f.exit_vel.y, 0.0).normalize().dot(want);
                assert!(along > 0.995, "yaw {yaw}: heading off the barrel by {:.1} deg", along.acos().to_degrees());
                // ballistics after the phantom lets go: apex and flight time from world gravity
                let vz = f.exit_vel.z;
                let rise = f.apex_z - f.exit_pos.z;
                close(rise, vz * vz / (2.0 * G), 0.1, "apex above the exit");
                let (t_land, p_land) = f.landed.expect("it came down");
                let fall = f.exit_pos.z - p_land.z;
                let flight = (vz + (vz * vz + 2.0 * G * fall).sqrt()) / G;
                close((t_land - f.exit) as f32 / 60.0, flight, 0.08, "flight time");
                assert!(Vec3::new(p_land.x - site.x, p_land.y - site.y, 0.0).length() > 10.0, "yaw {yaw}: landed only {:.1} wu away", p_land.distance(site));
                launches += 1;
            }
            assert_eq!(launches, 6, "liveness: every direction launched");
            // standing on the pad when it starts (the ground update, not the air, is what runs first)
            let f = fly_biped(native, cache, sc, site, tag, 0.0, true);
            eprintln!("standing start: exit tick {} |v| {:.3} v {:?}", f.exit, f.exit_vel.length(), f.exit_vel);
            assert!(f.exit != usize::MAX, "a body standing on the pad was never launched");
            assert!((7.2..=7.55).contains(&f.exit_vel.length()), "standing start: exit speed {}", f.exit_vel.length());
            assert!(f.exit_vel.x < -4.0 && f.exit_vel.z > 3.0, "standing start: exit velocity {:?}", f.exit_vel);
        }) else { return };
    }

    /// A Warthog placed at `start` (cannon-local, rotated by `yaw`) and seated in, run for `ticks`
    /// 60 Hz frames through the real `PlayMode::tick`.
    struct VFlight {
        /// (hull origin, velocity) per tick.
        track: Vec<(Vec3, Vec3)>,
        /// Ticks at which the hull's box left every phantom volume after being in one for 10+ ticks.
        exits: Vec<usize>,
        /// The lowest point of the hull above its origin.
        hull_min_z: f32,
    }

    fn fly_vehicle(native: &std::sync::Arc<NativeDll>, cache: CacheHandle, sc: &SceneController, site: Vec3, cannon_tag: u32, yaw: f32, start: Vec3, ticks: usize) -> VFlight {
        use crate::objscene::ObjectScene;
        let cannon = placed(cannon_tag, 0x7100, site, yaw);
        let scene = MapScene::new(sc, vec![cannon]);
        let (c, s) = (yaw.cos(), yaw.sin());
        let at = site + Vec3::new(start.x * c - start.y * s, start.x * s + start.y * c, start.z);
        let mut r = Ride::with_vehicle_placed(native.clone(), cache, "warthog", at, yaw, &[cannon]).expect("a Warthog");
        let tag = r.objects[0].primary_tag;
        let hull = scene.physics_body(&r.objects[0]).expect("the Warthog's physics body").aabb();
        let mut vols = Launchers::default();
        vols.refresh(native, cache, &[cannon]);
        r.force_seat(tag, 0).expect("seat the player");
        let idle = MoveInput { yaw, ..Default::default() };
        let (mut track, mut exits, mut inside_for) = (Vec::with_capacity(ticks), Vec::new(), 0usize);
        for t in 0..ticks {
            r.tick_in(&idle, &scene);
            let d = &r.pm.riding.as_ref().expect("still riding").drive;
            let bx = box_in_world(hull, d.pos, d.basis());
            if vols.volumes.iter().any(|v| v.overlaps(bx.0, bx.1)) {
                inside_for += 1;
            } else {
                if inside_for >= 10 {
                    exits.push(t);
                }
                inside_for = 0;
            }
            track.push((d.pos - site, d.vel));
        }
        VFlight { track, exits, hull_min_z: hull.0.z }
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn a_warthog_dropped_into_a_man_cannon_leaves_along_the_barrel_at_the_authored_speed() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let tag = tag_ending(native, cache, "bloc", "ff_man_cannon_forge\\ff_man_cannon_forge").expect("the man cannon");
            let f = fly_vehicle(native, cache, sc, site, tag, 0.0, Vec3::new(0.0, 0.0, 1.5), 420);
            let t = *f.exits.first().expect("the Warthog never left the volumes: it was never launched");
            let (p, v) = f.track[t];
            // marker forward of the long volume (tag): 35 degrees above the horizon toward -X
            let barrel = Vec3::new(-0.819_152_1, 0.0, 0.573_576_4);
            let along = v.dot(barrel);
            eprintln!("Warthog exit at tick {t}: pos {p:?} v {v:?}, along the barrel {along:.3}");
            // the long volume's direction cap (7.5 wu/s) is reached and not exceeded
            assert!((7.2..=7.6).contains(&along), "speed along the barrel {along}");
            assert!((v - barrel * along).length() < 0.6, "left the barrel line by {}", (v - barrel * along).length());
            // after release it is a free projectile: apex from the release height and velocity
            let apex = f.track[t..].iter().take(200).map(|x| x.0.z).fold(f32::MIN, f32::max);
            close(apex - p.z, v.z * v.z / (2.0 * G), 0.2, "apex above the release");
            let range = f.track.iter().map(|x| -x.0.x).fold(f32::MIN, f32::max);
            assert!(range > 14.0, "the Warthog only travelled {range} wu along the barrel");
        }) else { return };
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn a_warthog_on_the_vehicle_cannon_is_thrown_straight_up_again_and_again() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let tag = tag_ending(native, cache, "bloc", "ff_veh_man_cannon\\ff_veh_man_cannon").expect("the vehicle cannon");
            let f = fly_vehicle(native, cache, sc, site, tag, 0.0, Vec3::new(0.0, 0.0, 1.0), 600);
            eprintln!("vehicle cannon: exits at {:?}, hull min z {}", f.exits, f.hull_min_z);
            assert!(f.exits.len() >= 3, "liveness: expected the Warthog to be thrown, land and be thrown again twice, saw {} launch(es)", f.exits.len());
            for (n, &t) in f.exits.iter().enumerate() {
                let (p, v) = f.track[t];
                eprintln!("  launch {n}: exit tick {t}, z {:.2}, vz {:.2}", p.z, v.z);
                assert!(v.x.abs() < 0.3 && v.y.abs() < 0.3, "drifted off the shaft: {v:?}");
                // top of the rise (only for a launch with its whole arc inside the run)
                if t + 200 <= f.track.len() {
                    let apex = f.track[t..t + 200].iter().map(|x| x.0.z).fold(f32::MIN, f32::max);
                    close(apex - p.z, v.z * v.z / (2.0 * G), 0.25, "apex above the release");
                }
            }
            // The shaft is 1.97 wu tall (tag: z 0.1664 .. 2.1336 above the cannon); the volume pushes
            // straight up (marker forward = +Z) at 30 wu/s^2 (direction acc, tag) against a gravity of
            // 4.17 for as long as the hull's box overlaps it, i.e. from the box's bottom to the shaft's
            // top: v^2 = 2 (30 - g) L. Worked from the tag numbers, not from the code under test.
            let t0 = f.exits[0];
            let l = 2.1336 - (f.track[0].0.z + f.hull_min_z);
            let want = (2.0 * (30.0 - G) * l).sqrt();
            close(f.track[t0].1.z, want, 0.08 * want, "release speed from v^2 = 2 (30 - g) L");
        }) else { return };
    }

    /// Print-only: every object tag of the map whose phmo carries a phantom type that pushes.
    #[test]
    #[ignore = "opens the native library; print-only census"]
    fn census_of_pushing_phantoms() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let mut n = 0;
        for group in ["bloc", "scen", "mach", "ctrl", "vehi", "weap", "eqip", "crea", "ssce", "efsc", "bipd", "proj"] {
            for tag in native.tags_by_group(cache, group) {
                let vols = decode_tag(&native, cache, tag);
                if vols.is_empty() {
                    continue;
                }
                n += 1;
                let desc: Vec<String> = vols
                    .iter()
                    .map(|v| {
                        let (lo, hi) = v.pts.iter().fold((Vec3::splat(1e9), Vec3::splat(-1e9)), |(a, b), p| (a.min(*p), b.max(*p)));
                        format!("[{:#x} axis {:?} dir {:?} marker {:?} box {:.2?}..{:.2?}]", v.ty.flags, v.ty.axis, v.ty.direction, v.marker, lo, hi)
                    })
                    .collect();
                eprintln!("PUSHER {group} {} {}", native.tag_name(cache, tag).unwrap_or_default(), desc.join(" "));
            }
        }
        eprintln!("PUSHER census: {n} tag(s)");
    }


    /// The Forge shield doors are NOT obstacles to a biped. Their one solid shape is in the `Environment
    /// Only` group, whose `hcfd` mask (0x3) meets only the environment groups, so by the engine's
    /// own group rule it touches no biped (11), vehicle (15), prop or projectile (7) body; what
    /// stops anything at a door is the phantom push (`ff_one_way_door`: players, along its marker;
    /// `ff_two_way_door`: vehicles). Through the production gather `PlayWorld::regather`: the doors
    /// are filtered out of the player's solid set and a Forge platform beside them is kept.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn shield_doors_are_not_in_the_players_solid_set_but_blocks_are() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let doors = ["ff_one_way_door_small", "ff_one_way_door_xsmall", "ff_one_way_door_medium", "ff_one_way_door_large", "ff_one_way_door_garage", "ff_two_way_door_small", "ff_two_way_door_medium", "ff_two_way_door_large", "ff_shield_door_small", "ff_shield_door_medium", "ff_shield_door_large"];
            let mut objects: Vec<ObjectInfo> = doors
                .iter()
                .enumerate()
                .map(|(i, d)| placed(tag_ending(native, cache, "bloc", &format!("{d}\\{d}")).unwrap_or_else(|| panic!("{d}")), 0x7200 + i as u32, site + Vec3::new(i as f32 * 0.5, 2.0, 0.0), 0.0))
                .collect();
            let plat = placed(tag_ending(native, cache, "bloc", "ff_plat_4x4\\ff_plat_4x4").expect("a platform"), 0x7300, site + Vec3::new(0.0, -3.0, 0.0), 0.0);
            objects.push(plat);
            let scene = MapScene::new(sc, objects.clone());
            let mut w = crate::play_mode::PlayWorld::default();
            w.regather(site, &scene, &objects, native, cache);
            assert_eq!(w.objects_near, doors.len() + 1, "liveness: every placed object was considered");
            assert_eq!(w.objects_filtered, doors.len(), "the doors must be filtered out of the biped's solid set, the platform kept");
            assert!(w.object_tris > 0, "liveness: the platform contributed geometry");
        }) else { return };
    }

}
