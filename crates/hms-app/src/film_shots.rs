//! #projectile-sim  THEATER's simulated shots: the rounds a Reach film never replicates.
//!
//! # What the recording does and does not say
//!
//! Replicated projectiles (grenades, rockets, missiles, flak, needles on some builds) are film
//! entities and are drawn from their recorded tracks (`ActorDriver::film_effects`). A rifle,
//! pistol, plasma or needler round is not: it is predicted on each client, so the film carries only
//! the `action_weapon_fire` EVENT -- who fired, and when. This module turns each such event into a
//! real flight of the weapon's own `proj` through [`crate::play_projectile`], the same integrator,
//! the same collision and the same drawing path test mode uses.
//!
//! # Where the shot goes -- the recording's limit, stated per shot
//!
//! A film records ONE player's look direction exactly (the recorder's) and zero-fills every other
//! slot's. So the direction of a shot has three honest sources, and each shot says which it used:
//!
//! * [`AimSource::Recorded`] -- the recorder's own aim, exact.
//! * [`AimSource::Target`] -- INFERRED: the shooter's recorded body facing, tilted at the other body
//!   that is nearest the line of fire (within [`TARGET_CONE`] and [`TARGET_RANGE`]). Reach bodies
//!   turn to their aim while shooting, which is why facing is a usable proxy; the pitch to a body
//!   in front is the only part that is not on the wire.
//! * [`AimSource::Level`] -- INFERRED: the facing, level, when nobody is in front.
//!
//! The muzzle is not on the wire either: it sits [`MUZZLE_FRAC`] of the eye height above the feet
//! and [`MUZZLE_FORWARD`] in front (INFERRED from the held-weapon pose; the real pose of a past
//! tick is not kept).

use glam::Vec3;

/// Muzzle height as a fraction of the shooter's eye height.
pub const MUZZLE_FRAC: f32 = 0.85;
/// How far in front of the body the muzzle is, wu.
pub const MUZZLE_FORWARD: f32 = 0.45;
/// How far either side of the facing a body may be and still be what the shooter is firing at.
pub const TARGET_CONE: f32 = 25.0 * std::f32::consts::PI / 180.0;
/// How far away a target may be, wu.
pub const TARGET_RANGE: f32 = 150.0;
/// The aim point on a target, as a fraction of the eye height above its feet (chest).
pub const TARGET_CHEST: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AimSource {
    Recorded,
    Target,
    Level,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub origin: Vec3,
    pub dir: Vec3,
    pub source: AimSource,
}

/// Where a shot leaves and where it goes.
///
/// `recorded` is the shooter's own `(yaw, pitch)` in radians when the recording states one;
/// `facing` is the recorded body yaw; `others` are the feet positions of every other live body at
/// the firing tick.
pub fn aim(feet: Vec3, facing: f32, recorded: Option<(f32, f32)>, eye: f32, others: &[Vec3]) -> Aim {
    let base = feet + Vec3::Z * (eye * MUZZLE_FRAC);
    if let Some((yaw, pitch)) = recorded {
        let dir = Vec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin());
        let flat = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
        return Aim { origin: base + flat * MUZZLE_FORWARD, dir, source: AimSource::Recorded };
    }
    let flat = Vec3::new(facing.cos(), facing.sin(), 0.0);
    let origin = base + flat * MUZZLE_FORWARD;
    let mut best: Option<(f32, Vec3)> = None;
    for &o in others {
        let to = o + Vec3::Z * (eye * TARGET_CHEST) - origin;
        let horiz = Vec3::new(to.x, to.y, 0.0);
        let d = horiz.length();
        if !(0.5..=TARGET_RANGE).contains(&d) {
            continue;
        }
        let ang = (horiz / d).dot(flat).clamp(-1.0, 1.0).acos();
        if ang <= TARGET_CONE && best.map_or(true, |(a, _)| ang < a) {
            best = Some((ang, to.normalize()));
        }
    }
    match best {
        Some((_, dir)) => Aim { origin, dir, source: AimSource::Target },
        None => Aim { origin, dir: flat, source: AimSource::Level },
    }
}

/// How long to keep simulating a shot of this projectile, ticks: its fuse if it has one, else the
/// time to fly its range at its slowest authored speed, else a default. Only an upper bound so a
/// flight that never ends (a bullet fired at the sky) is cut off.
pub fn life_bound_ticks(def: &crate::reach_projectile::ProjectileDef) -> u32 {
    let timer = def.timer[0].max(def.timer[1]);
    let secs = if timer > 0.0 {
        timer
    } else if def.maximum_range > 0.0 {
        let v = [def.initial_velocity, def.final_velocity].into_iter().filter(|v| *v > 0.0).fold(f32::MAX, f32::min);
        if v < f32::MAX { def.maximum_range / v * 1.2 } else { 2.0 }
    } else {
        2.0
    };
    ((secs * 60.0).ceil() as u32).clamp(30, 900)
}

/// A stable key for one shot, so its trail / burst clocks are the same on every frame.
pub fn shot_key(net: u32, tick: u32) -> u32 {
    net.wrapping_mul(0x9E37_79B1) ^ tick.wrapping_mul(0x85EB_CA6B)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EYE: f32 = 0.60;

    /// The recorder's aim is used as stated, whatever else is around.
    #[test]
    fn the_recorders_aim_is_used_exactly() {
        let a = aim(Vec3::ZERO, 0.0, Some((1.0, 0.3)), EYE, &[Vec3::new(10.0, 0.0, 0.0)]);
        assert_eq!(a.source, AimSource::Recorded);
        assert!((a.dir - Vec3::new(1.0f32.cos() * 0.3f32.cos(), 1.0f32.sin() * 0.3f32.cos(), 0.3f32.sin())).length() < 1.0e-5);
    }

    /// Without a recorded aim the shot goes at the body in front, tilted up/down to it.
    #[test]
    fn an_unrecorded_shot_goes_at_the_body_in_front_of_the_shooter() {
        // a target 20 wu ahead and 3 wu up on a ledge
        let t = Vec3::new(20.0, 0.0, 3.0);
        let a = aim(Vec3::ZERO, 0.0, None, EYE, &[t]);
        assert_eq!(a.source, AimSource::Target);
        assert!(a.dir.z > 0.1, "should aim up at a body on a ledge, got {:?}", a.dir);
        // the line from the muzzle along the aim passes through the target's chest
        // chest = 0.75 of a 0.60 eye height = 0.45 wu above the feet, by hand (not from TARGET_CHEST)
        let chest = t + Vec3::Z * 0.45;
        let reach = a.origin + a.dir * (chest - a.origin).length();
        assert!((reach - chest).length() < 1.0e-3, "the aim misses the chest by {}", (reach - chest).length());
    }

    /// A body behind the shooter, or outside the cone, or out of range, is not a target; the shot
    /// is level along the facing.
    #[test]
    fn a_body_outside_the_cone_or_range_is_not_a_target() {
        let behind = Vec3::new(-10.0, 0.0, 0.0);
        let aside = Vec3::new(1.0, 20.0, 0.0); // ~87 degrees off
        let far = Vec3::new(TARGET_RANGE + 50.0, 0.0, 0.0);
        let a = aim(Vec3::ZERO, 0.0, None, EYE, &[behind, aside, far]);
        assert_eq!(a.source, AimSource::Level);
        assert!(a.dir.z.abs() < 1.0e-6 && (a.dir.x - 1.0).abs() < 1.0e-6, "level along +x expected, got {:?}", a.dir);
    }

    /// Of two bodies in the cone the one nearer the line of fire is chosen.
    #[test]
    fn the_body_nearest_the_line_of_fire_is_chosen() {
        let near_line = Vec3::new(30.0, 1.0, 0.0);
        let off_line = Vec3::new(10.0, 3.5, 0.0);
        let a = aim(Vec3::ZERO, 0.0, None, EYE, &[off_line, near_line]);
        assert_eq!(a.source, AimSource::Target);
        assert!(a.dir.y.abs() < 0.1, "should favour the body 2 degrees off, not 19: {:?}", a.dir);
    }

    #[test]
    fn the_muzzle_is_in_front_and_at_chest_height() {
        let a = aim(Vec3::new(5.0, 5.0, 2.0), std::f32::consts::FRAC_PI_2, None, EYE, &[]);
        assert!((a.origin.x - 5.0).abs() < 1.0e-5 && (a.origin.y - (5.0 + MUZZLE_FORWARD)).abs() < 1.0e-5);
        // 0.85 of a 0.60 eye height = 0.51 wu above the feet, by hand (not from MUZZLE_FRAC)
        assert!((a.origin.z - 2.51).abs() < 1.0e-5, "muzzle height {}", a.origin.z - 2.0);
    }

    #[test]
    fn a_fuse_bounds_the_flight_and_a_range_does() {
        use crate::reach_projectile::ProjectileDef;
        let fuse = ProjectileDef { timer: [2.0, 2.5], initial_velocity: 10.0, ..Default::default() };
        assert_eq!(life_bound_ticks(&fuse), 150);
        let range = ProjectileDef { maximum_range: 600.0, initial_velocity: 3000.0, final_velocity: 3000.0, ..Default::default() };
        assert_eq!(life_bound_ticks(&range), 30, "a 600 wu bullet at 3000 wu/s needs 0.2 s: bound clamps to the 30 tick floor");
        let none = ProjectileDef { initial_velocity: 8.0, ..Default::default() };
        assert_eq!(life_bound_ticks(&none), 120);
    }

    #[test]
    fn shot_keys_differ_by_body_and_tick() {
        let k = |n, t| shot_key(n, t);
        assert_ne!(k(1, 100), k(2, 100));
        assert_ne!(k(1, 100), k(1, 101));
        assert_eq!(k(7, 9), k(7, 9));
    }
}
