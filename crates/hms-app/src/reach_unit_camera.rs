//! #ride-camera  Reach's UNIT camera: the authored chase camera a seat looks through, and the
//! engine's blend into and out of it.
//!
//! Every number and every rule here was read out of `reach_tag_test.exe` (the MCC TOOL build, which
//! carries the engine WITH symbols); the function each came from is named where it is used. HMS
//! chooses nothing: where the engine's rule could not be recovered the module says UNRESOLVED.
//!
//! # What the engine does (PROVEN unless a line says INFERRED)
//!
//! **The mode.** `unit_should_be_third_person` (`sub_1401DA050`) reads the seat the player's unit
//! sits in: `seat flags` bit 4 `third person camera` is the answer, bit 6 `third person on enter`
//! keeps third person through the boarding animation. The game director
//! (`c_game_director::update`, `sub_1404E66B0`) then calls `director_set_camera_mode(0 = following,
//! transition)` when the answer turns yes and `(3 = first person, transition)` when it turns no.
//! The two transition times are the globals camera block's `enter vehicle transition time`
//! (`matg +0x3FC +0xB8`, **0.55 s**) and `exit vehicle transition time` (`+0xBC`, **0.5 s**) --
//! see `film_camera::ENTER_VEHICLE_TRANSITION_SECONDS`.
//!
//! **The shot** (`c_following_camera::update`, `sub_140517220` -> `sub_140516620`). The camera
//! definition is the SEAT's own `unit_camera_struct` (`seat +0x70`; `sub_140D76980`, which picks the
//! occupied seat's block for a seated unit and the unit's own `unit camera` at `+0x198` otherwise).
//! The FOCUS point is the camera marker (`camera marker name`) on the VEHICLE for a third-person
//! seat, the vehicle's origin when the seat names none. Then:
//!
//! * **no `camera tracks`** (count 0): the camera sits AT the marker, looks along the marker's
//!   own forward -- or, with `unit_camera_flags` bit 3 `use aiming vector instead of marker
//!   forward`, along the player's aim.
//! * **with `camera tracks`**: the shot is the `trak`'s control points, a cubic through four
//!   neighbouring points (`sub_140515BE0`, Newton form `sub_140516EC0`). The engine binary-searches
//!   the track parameter `t in [0,1]` (loop until the bracket is under 1e-4) for the point whose
//!   orientation pitch equals the PLAYER'S LOOK PITCH, and the camera stands at
//!   `focus + yaw_frame * (x, y, z)` of the track position there, where `yaw_frame` is
//!   `(forward-horizontal, left, up)` of the look yaw. Looking up therefore swings the camera DOWN
//!   and back along the authored curve, which is the whole point of a chase track.
//!   Which of the (max 2) tracks: index 0, or 1 under `unit_set_prefer_tight_camera_track`.
//!   A seat with a track count of 0 on a camera that wants one takes the globals camera block's
//!   `default unit camera track` (`sub_140515BE0`'s fallback).
//!
//! The observer then places the eye at `focus_position + hfwd*off.x + right*off.y + up*off.z -
//! forward*distance` (`sub_140130B10`, the `up == NULL` branch, plus `observer_compute_result`);
//! with the offset `(cos(p)*d + x, -y, sin(p)*d + z)` the engine builds, the `forward*distance`
//! terms cancel exactly and the eye is `focus + (hfwd*x + left*y + up*z)`.
//!
//! **The blend.** The command's `transition time` (`observer command +0x94`, filled from the
//! director's remaining time) is the duration of a QUINTIC: `sub_140136710` builds, per camera
//! parameter, the coefficients that carry the CURRENT value, velocity and acceleration to the
//! commanded target arriving at rest in the remaining time (`sub_140136BA0` evaluates it: a
//! degree-5 polynomial in the elapsed time). That is the minimum-jerk blend, and it is what
//! [`QuinticFollower`] implements. The following camera also forces `max(0.5 s, ...)` of blend
//! when the UNIT it follows changes (`sub_140517220`: `a4[37] = fmaxf(0.5, a4[37])`).

use glam::Vec3;
use hms_native::{CacheHandle, NativeDll};

// ---------------------------------------------------------------------------------------------
// THE TRAK
// ---------------------------------------------------------------------------------------------

/// One `camera_track_control_point_block` element: 28 bytes, `position` (3 floats) then an
/// `orientation` quaternion (x, y, z, w).
const POINT_STRIDE: usize = 28;

/// A `trak` (`camera_track_struct_definition`, 16 bytes): `flags` at +0, `control points`
/// (`[max 16]`, 28 B each) at +4.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraTrack {
    pub positions: Vec<Vec3>,
    /// `orientation` quaternions as `[x, y, z, w]`.
    pub orientations: Vec<[f32; 4]>,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(b, o))
}

impl CameraTrack {
    /// Read a `trak` tag. `None` when it does not read or has fewer than the four control points
    /// the engine's cubic needs (`sub_140515BE0` refuses such a track: "camera track %s has less
    /// than 4 control points").
    pub fn read(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<CameraTrack> {
        if native.tag_class(cache, tag).as_deref() != Some("trak") {
            return None;
        }
        let h = native.tag_meta(cache, tag, 0, 16)?;
        let count = rd_u32(&h, 4) as usize;
        let ptr = rd_u32(&h, 8);
        if !(4..=16).contains(&count) {
            return None;
        }
        let b = native.tag_ptr_bytes(cache, ptr, (count * POINT_STRIDE) as u32)?;
        let mut t = CameraTrack::default();
        for i in 0..count {
            let o = i * POINT_STRIDE;
            t.positions.push(Vec3::new(rd_f32(&b, o), rd_f32(&b, o + 4), rd_f32(&b, o + 8)));
            t.orientations.push([rd_f32(&b, o + 12), rd_f32(&b, o + 16), rd_f32(&b, o + 20), rd_f32(&b, o + 24)]);
        }
        Some(t)
    }

    /// `sub_140516EC0`: the cubic through four samples at `t0, t0+h, t0+2h, t0+3h`, in Newton's
    /// divided-difference form, evaluated at `t`.
    fn cubic(p: [f32; 4], t0: f32, h: f32, t: f32) -> f32 {
        let d1 = p[1] - p[0];
        let d2 = p[2] - p[1];
        let d3 = p[3] - p[2];
        let s2 = d2 - d1;
        let s3 = d3 - d2;
        let third = s3 - s2;
        // ((third * (t-(t0+2h))/(3h) + s2) * (t-(t0+h))/(2h) + d1) * (t-t0)/h + p0
        (((third * (t - (t0 + 2.0 * h)) / (3.0 * h) + s2) * (t - (t0 + h)) / (2.0 * h) + d1) * (t - t0) / h) + p[0]
    }

    /// The index of the first control point of the four the engine interpolates at `t`
    /// (`sub_140515BE0`): `floor((n-1)*t) - 1`, clamped to `0 ..= n-4`.
    pub fn window_start(&self, t: f32) -> usize {
        let n = self.positions.len();
        let seg = ((n - 1) as f32 * t) as i32;
        (seg - 1).clamp(0, n as i32 - 4) as usize
    }

    fn lerp4<F: Fn(usize) -> f32>(&self, t: f32, f: F) -> f32 {
        let n = self.positions.len();
        let s = self.window_start(t);
        let h = 1.0 / (n - 1) as f32;
        Self::cubic([f(s), f(s + 1), f(s + 2), f(s + 3)], s as f32 * h, h, t.clamp(0.0, 1.0))
    }

    /// The track's position at `t` (`sub_140515BE0`'s `a4` output).
    pub fn position(&self, t: f32) -> Vec3 {
        Vec3::new(
            self.lerp4(t, |i| self.positions[i].x),
            self.lerp4(t, |i| self.positions[i].y),
            self.lerp4(t, |i| self.positions[i].z),
        )
    }

    /// Where the camera looks at `t`: the `(1,0,0)` axis rotated by the interpolated orientation
    /// (`sub_1400A1830` rotates by each of the four quaternions, `sub_140517010` interpolates the
    /// four resulting vectors componentwise -- NOT the quaternions).
    pub fn forward(&self, t: f32) -> Vec3 {
        let rot = |q: [f32; 4]| {
            let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
            // Same expansion as sub_1400A1830 with v = (1,0,0).
            Vec3::new(2.0 * w * w - 1.0 + 2.0 * x * x, 2.0 * w * z + 2.0 * x * y, -2.0 * w * y + 2.0 * x * z)
        };
        let vs: Vec<Vec3> = self.orientations.iter().map(|&q| rot(q)).collect();
        Vec3::new(
            self.lerp4(t, |i| vs[i].x),
            self.lerp4(t, |i| vs[i].y),
            self.lerp4(t, |i| vs[i].z),
        )
    }

    /// Pitch of the track's look direction at `t` (`observer_try_and_get_camera`: `atan2(z,
    /// hypot(x, y))`).
    pub fn pitch(&self, t: f32) -> f32 {
        let f = self.forward(t);
        f.z.atan2((f.x * f.x + f.y * f.y).sqrt())
    }

    /// `sub_140516620`'s search: the parameter `t` whose track pitch equals `look_pitch`, by
    /// bisection on `[0, 1]` until the bracket is under `1e-4`, and the camera offset there.
    ///
    /// Returns `(t, offset)` with `offset = (x, y, z)` in the yaw frame `(forward-horizontal,
    /// left, up)`; the camera stands at `focus + yaw_frame * offset`.
    pub fn solve(&self, look_pitch: f32) -> (f32, Vec3) {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        let mut mid;
        loop {
            mid = (hi + lo) * 0.5;
            // `if (look_pitch <= track_pitch(mid)) hi = mid else lo = mid`.
            if look_pitch <= self.pitch(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo <= 1e-4 {
                break;
            }
        }
        (mid, self.position(mid))
    }
}

/// The eye of a track shot: `focus` plus the track offset carried into the world by the look
/// yaw (`sub_140130B10`: `pos.x += fx*off.x + fy*off.y'`, with the engine's own `off.y' = -y`
/// against `(fy, -fx)` = right, i.e. `+y` is LEFT in the track's authoring convention).
pub fn track_eye(focus: Vec3, look_yaw: f32, offset: Vec3) -> Vec3 {
    let (s, c) = look_yaw.sin_cos();
    let fwd = Vec3::new(c, s, 0.0);
    let left = Vec3::new(-s, c, 0.0);
    focus + fwd * offset.x + left * offset.y + Vec3::Z * offset.z
}

// ---------------------------------------------------------------------------------------------
// THE BLEND
// ---------------------------------------------------------------------------------------------

/// One scalar of the observer's quintic blend (`sub_140136710` / `sub_140136BA0`).
///
/// State is `(value, velocity, acceleration)`. Each step the coefficients are rebuilt to carry
/// that state to the CURRENT target arriving with zero velocity and acceleration after the
/// remaining time -- the observer does exactly this every frame, which is what keeps the blend
/// continuous when the target moves (the vehicle is driving while the camera settles).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quintic {
    pub x: f32,
    pub v: f32,
    pub a: f32,
}

impl Quintic {
    pub fn at(x: f32) -> Quintic {
        Quintic { x, v: 0.0, a: 0.0 }
    }

    /// Advance by `dt` toward `target`, which must be reached in `remaining` seconds.
    /// `remaining <= dt` lands on the target at rest.
    pub fn step(&mut self, target: f32, remaining: f32, dt: f32) {
        if remaining <= dt || remaining <= 1e-4 {
            *self = Quintic::at(target);
            return;
        }
        let t = remaining;
        // Quintic in the displacement from the target, d(s) = x - target, with d(T) = d'(T) =
        // d''(T) = 0:  d(s) = d0 + v0 s + a0 s^2/2 + c3 s^3 + c4 s^4 + c5 s^5.
        let d0 = self.x - target;
        let (v0, a0) = (self.v, self.a);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        // Solve the three terminal conditions for (c3, c4, c5).
        let c3 = -(20.0 * d0 + 12.0 * v0 * t + 3.0 * a0 * t2) / (2.0 * t3);
        let c4 = (30.0 * d0 + 16.0 * v0 * t + 3.0 * a0 * t2) / (2.0 * t4);
        let c5 = -(12.0 * d0 + 6.0 * v0 * t + a0 * t2) / (2.0 * t5);
        let s = dt;
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s3 * s;
        let s5 = s4 * s;
        self.x = target + d0 + v0 * s + a0 * s2 * 0.5 + c3 * s3 + c4 * s4 + c5 * s5;
        let nv = v0 + a0 * s + 3.0 * c3 * s2 + 4.0 * c4 * s3 + 5.0 * c5 * s4;
        let na = a0 + 6.0 * c3 * s + 12.0 * c4 * s2 + 20.0 * c5 * s3;
        self.v = nv;
        self.a = na;
    }
}

// ---------------------------------------------------------------------------------------------
// THE GLOBALS
// ---------------------------------------------------------------------------------------------

/// `matg +0x3FC -> camera_block`, the fields a ride reads. Element 0 of a `[max 1]` block, 216
/// (0xD8) bytes: see `film_camera` for the whole table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraGlobals {
    /// `+0xB8 enter vehicle transition time` -- the first-to-third-person blend, seconds.
    pub enter_seconds: f32,
    /// `+0xBC exit vehicle transition time` -- the third-to-first-person blend, seconds.
    pub exit_seconds: f32,
    /// `+0xC0 obstruction` (`camera_obstruction_struct`, 24 B): cylinder fraction, test angle
    /// (radians), max inward accel, max outward accel, max velocity, return delay. A seat that
    /// authors a ZERO for one of these takes the globals' value (`sub_1401DF570` fills each
    /// zero command field from the globals).
    pub obstruction: [f32; 6],
    /// `+0x00 default unit camera track`, a `trak`.
    pub default_track: Option<u32>,
}

impl CameraGlobals {
    /// The static defaults the engine falls back on when the block cannot be read: the values
    /// `.data +0x1EFA3F0..0x404` holds (5 deg test angle, accel 2 in / 1 out, velocity 10, delay
    /// 0.1 s), and the 0.55 / 0.5 transition times measured in `forge_halo`.
    pub const FALLBACK: CameraGlobals = CameraGlobals {
        enter_seconds: 0.55,
        exit_seconds: 0.5,
        obstruction: [0.0, 0.087_266_46, 2.0, 1.0, 10.0, 0.1],
        default_track: None,
    };

    pub fn read(native: &NativeDll, cache: CacheHandle) -> CameraGlobals {
        let Some(matg) = native.tags_by_group(cache, "matg").into_iter().next() else { return Self::FALLBACK };
        let Some(h) = native.tag_meta(cache, matg, 0x3FC, 8) else { return Self::FALLBACK };
        let (count, ptr) = (rd_u32(&h, 0), rd_u32(&h, 4));
        if count == 0 {
            return Self::FALLBACK;
        }
        let Some(b) = native.tag_ptr_bytes(cache, ptr, 0xD8) else { return Self::FALLBACK };
        let d = rd_u32(&b, 0x0C);
        let mut g = CameraGlobals {
            enter_seconds: rd_f32(&b, 0xB8),
            exit_seconds: rd_f32(&b, 0xBC),
            obstruction: std::array::from_fn(|k| rd_f32(&b, 0xC0 + 4 * k)),
            default_track: (d != u32::MAX && d != 0).then_some(d & 0xFFFF),
        };
        // A block that reads as nonsense (negative, NaN) is the fallback, not a negative blend.
        if !(g.enter_seconds >= 0.0 && g.enter_seconds.is_finite()) {
            g.enter_seconds = Self::FALLBACK.enter_seconds;
        }
        if !(g.exit_seconds >= 0.0 && g.exit_seconds.is_finite()) {
            g.exit_seconds = Self::FALLBACK.exit_seconds;
        }
        g
    }
}

// ---------------------------------------------------------------------------------------------
// THE SHOT
// ---------------------------------------------------------------------------------------------

/// The world direction of a `(yaw, pitch)` look, `+X` forward at yaw 0, pitch up positive --
/// `hms_render::Camera::forward()`'s convention.
pub fn look_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(cy * cp, sy * cp, sp)
}

/// What one frame of a third-person seat camera is made from.
#[derive(Clone, Copy, Debug)]
pub struct ShotInput<'a> {
    pub cam: &'a crate::reach_seats::SeatCamera,
    /// The track `cam` selects, already read; `None` when the seat authors none.
    pub track: Option<&'a CameraTrack>,
    /// World focus point: the camera marker on the vehicle, or the vehicle's origin when the seat
    /// names none (`sub_140D76980`).
    pub focus: Vec3,
    /// World forward of the camera marker, when the seat has one.
    pub marker_forward: Option<Vec3>,
    pub look_yaw: f32,
    pub look_pitch: f32,
}

/// The seat's shot before the wall filter: `(eye, forward, track parameter)`.
///
/// With a track the eye is [`track_eye`] at the parameter the look pitch selects; the camera
/// looks along the player's aim. Without one the eye IS the marker and the camera looks along the
/// marker's forward, unless `use aiming vector instead of marker forward` is set
/// (`sub_140516620`'s `*v16 & 8` branch).
pub fn chase_shot(i: &ShotInput) -> (Vec3, Vec3, Option<f32>) {
    let aim = look_dir(i.look_yaw, i.look_pitch);
    match i.track {
        Some(track) => {
            let (t, off) = track.solve(i.look_pitch);
            (track_eye(i.focus, i.look_yaw, off), aim, Some(t))
        }
        None => {
            let fwd = if i.cam.flags & crate::reach_seats::SeatCamera::USE_AIMING_VECTOR != 0 {
                aim
            } else {
                i.marker_forward.unwrap_or(aim)
            };
            (i.focus, fwd, None)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// THE POSE BLEND
// ---------------------------------------------------------------------------------------------

/// A camera pose: eye, yaw and pitch (radians), the form `PlayMode::camera` hands the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamPose {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

fn wrap_pi(a: f32) -> f32 {
    let t = std::f32::consts::TAU;
    (a + std::f32::consts::PI).rem_euclid(t) - std::f32::consts::PI
}

/// The observer's transition between two cameras (`sub_140136710`): every component carried by
/// its own [`Quintic`] over a shared `remaining` time. Yaw is blended as an unwrapped angle, so
/// the camera goes the SHORT way round.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraBlend {
    eye: [Quintic; 3],
    yaw: Quintic,
    pitch: Quintic,
    remaining: f32,
    active: bool,
}

impl CameraBlend {
    /// Begin a blend that leaves `from` and arrives at whatever the target is in `seconds`.
    /// `seconds <= 0` is a CUT (the death chain does exactly that).
    pub fn start(&mut self, from: CamPose, seconds: f32) {
        self.eye = [Quintic::at(from.eye.x), Quintic::at(from.eye.y), Quintic::at(from.eye.z)];
        self.yaw = Quintic::at(from.yaw);
        self.pitch = Quintic::at(from.pitch);
        self.remaining = seconds.max(0.0);
        self.active = seconds > 0.0;
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// Seconds still to run.
    pub fn remaining(&self) -> f32 {
        if self.active { self.remaining } else { 0.0 }
    }

    /// Advance `dt` toward `target` and return the pose to render. Inactive: the target itself.
    pub fn step(&mut self, target: CamPose, dt: f32) -> CamPose {
        if !self.active {
            return target;
        }
        // Take the target's yaw as the equivalent angle nearest to where the follower is, or a
        // camera crossing +-pi would spin the long way.
        let ty = self.yaw.x + wrap_pi(target.yaw - self.yaw.x);
        let r = self.remaining;
        self.eye[0].step(target.eye.x, r, dt);
        self.eye[1].step(target.eye.y, r, dt);
        self.eye[2].step(target.eye.z, r, dt);
        self.yaw.step(ty, r, dt);
        self.pitch.step(target.pitch, r, dt);
        self.remaining = (r - dt).max(0.0);
        if self.remaining <= 0.0 {
            self.active = false;
        }
        CamPose {
            eye: Vec3::new(self.eye[0].x, self.eye[1].x, self.eye[2].x),
            yaw: wrap_pi(self.yaw.x),
            pitch: self.pitch.x,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// THE OBSTRUCTION FILTER
// ---------------------------------------------------------------------------------------------

/// `accelerate_to_position` (`real_math.cpp`), one axis: move `pos` toward `target` under an
/// acceleration limit `accel` and a speed limit `vmax`, braking so it ARRIVES rather than
/// overshooting. Returns the new `(pos, vel)`.
pub fn accelerate_to_position(pos: f32, vel: f32, target: f32, dt: f32, accel: f32, vmax: f32) -> (f32, f32) {
    let d = target - pos;
    // Within one step of arriving: land on it.
    if (d - vel * dt).abs() <= (dt * vmax).min(dt * dt * accel) {
        return (target, 0.0);
    }
    let mut speed = vmax;
    let stop = d.abs() * 2.0 * accel;
    if stop < vmax * vmax {
        speed = stop.sqrt();
    }
    if d < 0.0 {
        speed = -speed;
    }
    let a = ((speed - vel) / dt).clamp(-accel, accel);
    let v = vel + a * dt;
    (pos + (a * 0.5 * dt + v) * dt, v)
}

/// The follow camera's anti-wall filter (`camera_obstruction_update_*`, `sub_140414D40`).
///
/// The engine casts six rays from the focus point toward the wanted eye, finds the largest
/// fraction `t` of the way out that is clear, and keeps a 1-D follower on that fraction: it comes
/// IN at the inward acceleration, goes back OUT at the outward acceleration after
/// `return delay`, and is never allowed further out than the clear fraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstruction {
    /// Fraction of the focus-to-eye line the camera currently sits at.
    pub t: f32,
    vel: f32,
    delay: f32,
}

impl Default for Obstruction {
    fn default() -> Self {
        Obstruction { t: 1.0, vel: 0.0, delay: 0.0 }
    }
}

impl Obstruction {
    /// `params` is `[cylinder fraction, test angle, inward accel, outward accel, max velocity,
    /// return delay]`, with zero meaning "use the globals" -- resolve with [`resolve_params`].
    pub fn step(&mut self, clear_t: f32, params: &[f32; 6], dt: f32) -> f32 {
        let clear_t = clear_t.clamp(0.0, 1.0);
        let (inward, outward, vmax, delay) = (params[2], params[3], params[4], params[5]);
        if clear_t <= self.t {
            // Walls closing in: use the inward acceleration (the larger of the two while moving
            // out) and re-arm the return delay.
            let accel = if self.vel > 0.0 { inward.max(outward) } else { inward };
            self.delay = delay;
            let (t, v) = accelerate_to_position(self.t, self.vel, clear_t, dt, accel, vmax);
            self.t = t;
            self.vel = v;
        } else if clear_t - self.t >= 0.05 && self.delay > 0.0 {
            // Wants to go back out, but not yet.
            self.vel = 0.0;
            self.delay = (self.delay - dt).max(0.0);
        } else {
            self.delay = 0.0;
            // Moving IN while the way has opened: brake with the larger of the two.
            let accel = if self.vel < 0.0 { inward.max(outward) } else { outward };
            let (t, v) = accelerate_to_position(self.t, self.vel, clear_t, dt, accel, vmax);
            self.t = t;
            self.vel = v;
        }
        // `camera_obstruction_update_postprocess`: never further out than what is clear.
        if self.t > clear_t {
            self.t = clear_t;
            self.vel = 0.0;
            self.delay = delay;
        }
        self.t
    }
}

/// A seat that authors zero for an obstruction field takes the globals' (`sub_1401DF570`).
pub fn resolve_params(seat: [f32; 6], globals: [f32; 6]) -> [f32; 6] {
    std::array::from_fn(|k| if seat[k] == 0.0 { globals[k] } else { seat[k] })
}

// ---------------------------------------------------------------------------------------------
// THE MODE
// ---------------------------------------------------------------------------------------------

/// Which camera the player is looking through, in the engine's own terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamMode {
    /// On foot: the biped's own eye (or HMS's optional third-person view of it).
    Foot,
    /// A first-person SEAT camera (a turret's, a passenger's).
    SeatFirst,
    /// The `following` camera: the chase shot of a third-person seat.
    Chase,
}

/// How long the engine blends when the mode changes (`c_game_director::update`).
///
/// The only non-zero transitions it ever passes are INTO the following camera (the globals'
/// `enter vehicle transition time`) and OUT of it (`exit vehicle transition time`). First person to
/// first person is a cut: `director_set_camera_mode` only arms a transition when the MODE
/// differs.
pub fn transition_seconds(from: CamMode, to: CamMode, g: &CameraGlobals) -> f32 {
    match (from, to) {
        (a, b) if a == b => 0.0,
        (_, CamMode::Chase) => g.enter_seconds,
        (CamMode::Chase, _) => g.exit_seconds,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------------------------
// THE GATES
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod gates {
    //! #ride-camera  Pure gates: no cache, they run everywhere. The map-backed ones (the real
    //! Warthog track, the whole ride) live in `play_mode::ride_gates`.

    use super::*;

    /// A track whose pitch rises linearly from -45 to +45 degrees over `t in [0, 1]` and whose
    /// position is a CUBIC in `t` -- so the engine's four-point cubic must reproduce it exactly.
    fn ramp_track(n: usize) -> CameraTrack {
        let mut t = CameraTrack::default();
        for i in 0..n {
            let u = i as f32 / (n - 1) as f32;
            let pitch = (-45.0f32 + 90.0 * u).to_radians();
            // forward (cos p, 0, sin p): the +X axis rotated about Y by -pitch.
            let th = -pitch;
            t.orientations.push([0.0, (th * 0.5).sin(), 0.0, (th * 0.5).cos()]);
            t.positions.push(Vec3::new(-3.0 + 2.0 * u * u * u, 0.5 * u * u, 2.0 - 4.0 * u + 1.5 * u * u * u));
        }
        t
    }

    #[test]
    fn the_cubic_reproduces_a_cubic_exactly_across_every_window() {
        let t = ramp_track(15);
        for k in 0..=200 {
            let u = k as f32 / 200.0;
            let want = Vec3::new(-3.0 + 2.0 * u * u * u, 0.5 * u * u, 2.0 - 4.0 * u + 1.5 * u * u * u);
            let got = t.position(u);
            assert!((got - want).length() < 2e-4, "u={u}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn the_window_is_centred_on_the_segment_and_clamped_to_the_track() {
        // n = 15: floor(14 t) - 1, clamped to 0..=11 (`sub_140515BE0`).
        let t = ramp_track(15);
        assert_eq!(t.window_start(0.0), 0);
        assert_eq!(t.window_start(0.05), 0, "segment 0 is clamped up to window 0");
        assert_eq!(t.window_start(0.5), 6);
        assert_eq!(t.window_start(0.72), 9);
        assert_eq!(t.window_start(1.0), 11, "the last window is n-4");
    }

    #[test]
    fn the_orientation_port_agrees_with_a_quaternion_library() {
        // `sub_1400A1830` is the (1,0,0) axis rotated by a quaternion; glam is the reference.
        for k in 0..40 {
            let a = k as f32 * 0.37;
            let q = glam::Quat::from_euler(glam::EulerRot::ZYX, a, a * 0.6 - 0.4, a * 0.3).normalize();
            let mut t = CameraTrack::default();
            for _ in 0..4 {
                t.positions.push(Vec3::ZERO);
                t.orientations.push([q.x, q.y, q.z, q.w]);
            }
            let got = t.forward(0.5);
            let want = q * Vec3::X;
            assert!((got - want).length() < 1e-4, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn the_search_lands_on_the_parameter_whose_pitch_is_the_look_pitch() {
        let t = ramp_track(15);
        for deg in [-40.0f32, -20.0, 0.0, 17.0, 33.0] {
            let (u, off) = t.solve(deg.to_radians());
            let want_u = (deg + 45.0) / 90.0;
            assert!((u - want_u).abs() < 5e-4, "pitch {deg}: t {u} vs {want_u}");
            assert!((t.pitch(u).to_degrees() - deg).abs() < 0.1, "pitch {deg}: landed on {}", t.pitch(u).to_degrees());
            assert!((off - t.position(u)).length() < 1e-6, "the offset is the position at that parameter");
        }
        // Past the ends it saturates rather than extrapolating.
        let (lo, _) = t.solve(-80.0f32.to_radians());
        let (hi, _) = t.solve(80.0f32.to_radians());
        assert!(lo < 2e-4 && hi > 1.0 - 2e-4, "{lo} {hi}");
    }

    #[test]
    fn looking_up_swings_the_camera_down_the_authored_curve() {
        // The chase camera's whole point: pitch up -> the eye drops (the ramp's z falls with t).
        let t = ramp_track(15);
        let focus = Vec3::new(10.0, 20.0, 1.0);
        let down = track_eye(focus, 0.0, t.solve((-30.0f32).to_radians()).1);
        let level = track_eye(focus, 0.0, t.solve(0.0).1);
        let up = track_eye(focus, 0.0, t.solve(30.0f32.to_radians()).1);
        assert!(down.z > level.z && level.z > up.z, "{down:?} {level:?} {up:?}");
    }

    #[test]
    fn the_track_offset_is_carried_by_the_look_yaw_with_y_to_the_left() {
        let focus = Vec3::new(1.0, 2.0, 3.0);
        let off = Vec3::new(-3.0, 0.5, 0.85);
        // Yaw 0: behind is -X, left is +Y.
        let e = track_eye(focus, 0.0, off);
        assert!((e - Vec3::new(-2.0, 2.5, 3.85)).length() < 1e-5, "{e:?}");
        // Yaw 90 deg (looking +Y): behind is -Y, left is -X.
        let e = track_eye(focus, std::f32::consts::FRAC_PI_2, off);
        assert!((e - Vec3::new(0.5, -1.0, 3.85)).length() < 1e-5, "{e:?}");
    }

    #[test]
    fn a_quintic_blend_arrives_on_time_at_rest_without_overshoot() {
        let dt: f32 = 1.0 / 60.0;
        let total: f32 = 0.55;
        let mut q = Quintic::at(10.0);
        let mut prev_v = 0.0f32;
        let mut x_prev = 10.0f32;
        let steps = (total / dt).round() as usize;
        for i in 0..steps {
            let remaining = total - i as f32 * dt;
            q.step(0.0, remaining, dt);
            assert!(q.x <= x_prev + 1e-5, "step {i}: moved away from the target ({} -> {})", x_prev, q.x);
            assert!(q.x >= -1e-4, "step {i}: overshot to {}", q.x);
            // C1: velocity never jumps by more than the minimum-jerk profile allows.
            assert!((q.v - prev_v).abs() < 12.0, "step {i}: velocity jumped {prev_v} -> {}", q.v);
            x_prev = q.x;
            prev_v = q.v;
            if i + 1 < steps {
                assert!(q.x > 0.0, "step {i}: arrived early");
            }
        }
        assert!(q.x.abs() < 1e-3 && q.v.abs() < 0.5, "at the end: x {} v {}", q.x, q.v);
        // The whole path is the minimum-jerk curve -- checked against its CLOSED FORM, not against
        // the coefficients that produced it: x(t) = x0 * (1 - (10 u^3 - 15 u^4 + 6 u^5)), u = t/T.
        let mut p = Quintic::at(10.0);
        for i in 0..steps - 1 {
            p.step(0.0, total - i as f32 * dt, dt);
            let u = (i + 1) as f32 * dt / total;
            let want = 10.0 * (1.0 - (10.0 * u.powi(3) - 15.0 * u.powi(4) + 6.0 * u.powi(5)));
            assert!((p.x - want).abs() < 0.02, "step {i}: x {} but the minimum-jerk curve says {want}", p.x);
        }
        // Halfway it is nowhere near either end (an ease, not a cut and not a snap).
        let mut h = Quintic::at(10.0);
        for i in 0..(steps / 2) {
            h.step(0.0, total - i as f32 * dt, dt);
        }
        assert!(h.x > 2.0 && h.x < 8.0, "halfway is {}", h.x);
    }

    #[test]
    fn a_camera_blend_takes_exactly_its_transition_time() {
        let dt = 1.0 / 60.0;
        let a = CamPose { eye: Vec3::new(0.0, 0.0, 1.6), yaw: 0.2, pitch: 0.0 };
        let b = CamPose { eye: Vec3::new(-3.0, 0.0, 0.85), yaw: 0.2, pitch: -0.1 };
        let mut c = CameraBlend::default();
        c.start(a, 0.55);
        let mut t = 0.0;
        let mut last = a;
        while c.active() {
            last = c.step(b, dt);
            t += dt;
            assert!(t < 0.7, "the blend never ended");
        }
        assert!((t - 0.55).abs() <= dt * 1.01, "blend took {t} s, the transition is 0.55 s");
        assert!((last.eye - b.eye).length() < 1e-3, "it must END on the target: {last:?}");
        // And a transition of zero is a CUT: the first frame is already the target.
        let mut cut = CameraBlend::default();
        cut.start(a, 0.0);
        assert!(!cut.active());
        assert_eq!(cut.step(b, dt), b);
    }

    #[test]
    fn a_blend_follows_a_moving_target_and_goes_the_short_way_round_in_yaw() {
        let dt = 1.0 / 60.0;
        let a = CamPose { eye: Vec3::ZERO, yaw: 3.0, pitch: 0.0 };
        let mut c = CameraBlend::default();
        c.start(a, 0.5);
        // The target is across the +-pi seam: 3.0 -> -3.0 is a 0.28 rad turn, not 6.0.
        let mut max_step = 0.0f32;
        let mut prev = 3.0f32;
        for i in 0..40 {
            let p = c.step(CamPose { eye: Vec3::new(i as f32 * 0.01, 0.0, 0.0), yaw: -3.0, pitch: 0.0 }, dt);
            let d = (p.yaw - prev + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            max_step = max_step.max(d.abs());
            prev = p.yaw;
        }
        assert!(max_step < 0.1, "yaw spun the long way round: {max_step} rad in one frame");
    }

    #[test]
    fn the_obstruction_comes_in_fast_and_waits_before_going_back_out() {
        let params = [0.0, 0.0873, 2.0, 1.0, 10.0, 0.1];
        let dt = 1.0 / 60.0;
        let mut o = Obstruction::default();
        // A wall at 30% of the way out.
        for _ in 0..120 {
            o.step(0.3, &params, dt);
        }
        assert!((o.t - 0.3).abs() < 1e-3, "settles on the wall: {}", o.t);
        // The wall goes away: for the return delay it does not move.
        let mut waited = 0;
        loop {
            let before = o.t;
            let t = o.step(1.0, &params, dt);
            if t > before + 1e-5 {
                break;
            }
            waited += 1;
            assert!(waited < 60, "never started going back out");
        }
        let waited_s = waited as f32 * dt;
        assert!(waited_s >= 0.08 && waited_s <= 0.2, "waited {waited_s} s, the authored delay is 0.1 s");
        // And it never exceeds what is clear.
        for _ in 0..30 {
            let t = o.step(0.5, &params, dt);
            assert!(t <= 0.5 + 1e-5, "{t} is past the new wall");
        }
    }

    #[test]
    fn a_seat_that_authors_zero_takes_the_globals() {
        let g = [0.0, 0.0873, 2.0, 1.0, 10.0, 0.1];
        assert_eq!(resolve_params([0.0; 6], g), g);
        let mut s = [0.0; 6];
        s[2] = 5.0;
        assert_eq!(resolve_params(s, g)[2], 5.0);
        assert_eq!(resolve_params(s, g)[3], 1.0);
    }

    #[test]
    fn only_a_change_into_or_out_of_the_following_camera_blends() {
        let g = CameraGlobals::FALLBACK;
        assert_eq!(transition_seconds(CamMode::Foot, CamMode::Chase, &g), 0.55);
        assert_eq!(transition_seconds(CamMode::Chase, CamMode::Foot, &g), 0.5);
        assert_eq!(transition_seconds(CamMode::SeatFirst, CamMode::Chase, &g), 0.55);
        assert_eq!(transition_seconds(CamMode::Chase, CamMode::SeatFirst, &g), 0.5);
        assert_eq!(transition_seconds(CamMode::Foot, CamMode::SeatFirst, &g), 0.0, "first person to first person is a cut");
        assert_eq!(transition_seconds(CamMode::Chase, CamMode::Chase, &g), 0.0);
    }

    #[test]
    fn a_chase_shot_with_a_track_stands_where_the_track_says_and_looks_along_the_aim() {
        let t = ramp_track(15);
        let cam = crate::reach_seats::SeatCamera::default();
        let focus = Vec3::new(10.0, 20.0, 1.0);
        for (yaw, pitch) in [(0.0f32, 0.0f32), (1.2, 0.3), (-2.0, -0.4)] {
            let s = ShotInput { cam: &cam, track: Some(&t), focus, marker_forward: Some(Vec3::Y), look_yaw: yaw, look_pitch: pitch };
            let (eye, fwd, param) = chase_shot(&s);
            let (u, off) = t.solve(pitch);
            assert_eq!(param, Some(u));
            assert!((eye - track_eye(focus, yaw, off)).length() < 1e-5, "yaw {yaw} pitch {pitch}: eye {eye:?}");
            assert!((fwd - look_dir(yaw, pitch)).length() < 1e-6, "the camera looks along the AIM, not the marker: {fwd:?}");
            // And it is BEHIND the aim: the track's x is negative.
            assert!((eye - focus).dot(look_dir(yaw, 0.0)) < -1.0, "yaw {yaw}: the camera is not behind the focus");
        }
    }

    #[test]
    fn a_chase_shot_without_a_track_sits_on_the_marker_and_looks_along_it() {
        let cam = crate::reach_seats::SeatCamera::default();
        let s = ShotInput {
            cam: &cam,
            track: None,
            focus: Vec3::new(5.0, 6.0, 7.0),
            marker_forward: Some(Vec3::Y),
            look_yaw: 0.0,
            look_pitch: 0.0,
        };
        let (eye, fwd, t) = chase_shot(&s);
        assert_eq!(eye, s.focus);
        assert_eq!(fwd, Vec3::Y, "marker forward, not the look");
        assert!(t.is_none());
        // `use aiming vector instead of marker forward` swaps it for the look.
        let aimed = crate::reach_seats::SeatCamera { flags: crate::reach_seats::SeatCamera::USE_AIMING_VECTOR, ..cam.clone() };
        let (_, fwd, _) = chase_shot(&ShotInput { cam: &aimed, ..s });
        assert!((fwd - Vec3::X).length() < 1e-6, "{fwd:?}");
    }
}
