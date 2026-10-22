//! The solver's constants: the numbers the engine's `hkpWorld` constructor derives from the cinfo
//! (`sub_180D4DE10` in `haloreach_debug.dll`, line "v14 = a1 + 0x308") and the per-step copies the
//! integrate pass makes (`sub_180D72850`).
//!
//! # Provenance
//!
//! **PROVEN** (decompiled `haloreach_debug.dll`, same code in `haloreach.dll` `sub_1808E4120`):
//!
//! ```text
//!   hkpSolverInfo (world+0x2A0)              cinfo             value in Reach
//!     +0x04 tau            = cinfo +0x88      solverTau         0.3
//!     +0x08 damping        = cinfo +0x8C      solverDamp        0.9
//!     +0x30 avgGain        = tau / damp       (broadcast)       1/3
//!     +0x40 avgScale0      = damp / tau       (broadcast)       3
//!     +0x114 numSteps      = cinfo +0x90      solverIterations  8
//!     +0x118 numMicro      = cinfo +0x94      solverMicrosteps  1
//!     +0x120 invNumSteps   = 1 / numSteps
//!   step copy (`v33` in sub_180D72850), dt = tick length
//!     f00 subDt   = dt / numSteps          f0C dt              f10 1/dt
//!     f18 = 1 / (numMicro * numSteps)      f20 damping
//!     f1C = (tau/damp) * numSteps / dt     (contact rhs scale)
//!     f24 = ((tau/2)/damp) * numSteps / dt (friction anchor scale)
//! ```
//!
//! The six deactivation-class rows (world ctor, `v16` loop) are built from `|g|`, the class
//! thresholds `1.19e-7/0, 1.19e-7/0, 0.01/0.08, 0.017/0.2, 0.02/0.3, 0.025/0.4` and the 60 Hz
//! reference `0.016`. See [`DeactClass`].

use glam::Vec3;

/// `hkMotionState.deactivationClass` values: `INVALID, OFF, LOW, MEDIUM, HIGH, AGGRESSIVE`
/// (world ctor builds one [`DeactClass`] row for each of the six).
pub const CLASS_COUNT: usize = 6;

/// `(linear threshold factor, slow-object factor)` per class, multiplied by `|g|` (world ctor
/// switch statements; **PROVEN** -- the three blocks of the `v16` loop give the same six pairs).
const CLASS_THRESHOLDS: [(f32, f32); CLASS_COUNT] = [
    (1.192_092_9e-7, 0.0),
    (1.192_092_9e-7, 0.0),
    (0.01, 0.08),
    (0.017, 0.2),
    (0.02, 0.3),
    (0.025, 0.4),
];

/// The 60 Hz reference step the world ctor scales the slow-object damping with (literal
/// `0.0160000008` in `sub_180D4DE10`).
const REFERENCE_STEP: f32 = 0.016;

/// `FLT_MAX` as the ctor writes it for a class with no slow-object factor (`2.12676251e+37`).
const NO_ZERO_THRESHOLD: f32 = 2.126_762_5e37;

/// One row of the solver's per-deactivation-class table (`hkpSolverInfo::m_deactivationInfo`,
/// seven floats at `solverInfo + 0x64 + class * 0x1C`). **PROVEN** layout and formulas:
///
/// ```text
///   v20 = threshold * |g|
///   +0x00 lin_inv      = 1 / v20
///   +0x04 ang_inv      = 1 / (v20 * 0.1|g|)
///   +0x08 slow_mult    = 1 - (slow * |g| / v20) * (0.016 / numSteps)
///   +0x0C zero_below   = slow <= 0 ? FLT_MAX : (0.016 / numSteps) / slow
///   +0x10 ref_sq       = deactivationReferenceDistance^2
///   +0x14 ref4_sq      = (4 * deactivationReferenceDistance)^2
///   +0x18 ang_bits     = 0x3D6B3B6B  (two int16 halves, each read as the TOP half of an f32)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DeactClass {
    pub lin_inv: f32,
    pub ang_inv: f32,
    pub slow_mult: f32,
    pub zero_below: f32,
    pub ref_sq: f32,
    pub ref4_sq: f32,
    /// Squared-distance threshold of the orientation check in the 4-step window
    /// (`(int16)0x3B6B << 16` as an f32).
    pub ang_short: f32,
    /// The same for the 16-step window (`(int16)0x3D6B << 16`).
    pub ang_long: f32,
}

/// `hkpWorldCinfo::deactivationReferenceDistance` (`sub_1807F66B0`): `0.1 * (1 / (tickRate / 15))` with
/// an INTEGER divide (`sub_180644BD0`, magic number `0x88888889`) = 0.025 wu at 60 Hz.
/// **PROVEN** in `04-deactivation-islands.md` 2.3; `reach_havok::deactivation_reference_distance`
/// (0.05) divides by 30 and is off by a factor two.
pub fn deactivation_reference_distance(tick_hz: f32) -> f32 {
    0.1 * (1.0 / ((tick_hz as i32 / 15).max(1) as f32))
}

/// The solver's derived constants for one world.
#[derive(Clone, Debug)]
pub struct SolverInfo {
    /// Tick length, s (`physicsDeltaTime`, step info +8).
    pub dt: f32,
    pub inv_dt: f32,
    /// `solverIterations` (Havok's `numSteps`).
    pub iterations: u32,
    pub micro_steps: u32,
    pub tau: f32,
    pub damp: f32,
    pub gravity: Vec3,
    /// `dt / iterations`.
    pub sub_dt: f32,
    /// `f1C`: scale of the contact rhs.
    pub rhs_scale: f32,
    /// `f24`: scale of the friction anchor rhs.
    pub anchor_scale: f32,
    /// `f18`: scale of the per-pass friction cap.
    pub friction_pass: f32,
    /// `tau / damp`: the gain of the running average (`solverInfo +0x30`).
    pub avg_gain: f32,
    /// `(damp / tau) / iterations`: what the average is scaled by after the last step.
    pub avg_scale: f32,
    /// `gravity * sub_dt`  (world +0x2B0).
    pub gravity_sub: Vec3,
    /// `gravity * dt` (world +0x2C0).
    pub gravity_tick: Vec3,
    pub classes: [DeactClass; CLASS_COUNT],
    /// `contactRestingVelocity` (world +0x300): `FLT_MAX` in Reach.
    pub contact_resting_velocity: f32,
}

impl SolverInfo {
    /// The engine's values for tick rate `tick_hz` and world gravity `gravity`
    /// (`hkpWorldCinfo` as filled by `sub_1807F66B0` + the world ctor).
    pub fn engine(tick_hz: f32, gravity: Vec3) -> Self {
        Self::build(
            1.0 / tick_hz,
            gravity,
            crate::reach_havok::SOLVER_ITERATIONS,
            crate::reach_havok::SOLVER_MICROSTEPS,
            crate::reach_havok::SOLVER_TAU,
            crate::reach_havok::SOLVER_DAMP,
            deactivation_reference_distance(tick_hz),
        )
    }

    /// Build from explicit cinfo values (the gates sabotage these).
    pub fn build(dt: f32, gravity: Vec3, iterations: u32, micro_steps: u32, tau: f32, damp: f32, ref_distance: f32) -> Self {
        let inv_dt = 1.0 / dt;
        let n = iterations as f32;
        let sub_dt = dt / n;
        let mut g = gravity.length();
        if g == 0.0 {
            g = 9.81;
        }
        let inv_iters = 1.0 / n;
        let mut classes = [DeactClass { lin_inv: 0.0, ang_inv: 0.0, slow_mult: 0.0, zero_below: 0.0, ref_sq: 0.0, ref4_sq: 0.0, ang_short: 0.0, ang_long: 0.0 }; CLASS_COUNT];
        for (c, (thr, slow)) in classes.iter_mut().zip(CLASS_THRESHOLDS) {
            let v20 = thr * g;
            c.lin_inv = 1.0 / v20;
            c.ang_inv = 1.0 / (v20 * (g * 0.1));
            c.slow_mult = 1.0 - slow * g / v20 * (inv_iters * REFERENCE_STEP);
            c.zero_below = if slow <= 0.0 { NO_ZERO_THRESHOLD } else { inv_iters * REFERENCE_STEP / slow };
            c.ref_sq = ref_distance * ref_distance;
            c.ref4_sq = (ref_distance * 4.0) * (ref_distance * 4.0);
            // 0x3D6B3B6B: low half 0x3B6B for the 4-step window, high half 0x3D6B for the 16-step
            c.ang_short = f32::from_bits(0x3B6B_0000);
            c.ang_long = f32::from_bits(0x3D6B_0000);
        }
        SolverInfo {
            dt,
            inv_dt,
            iterations,
            micro_steps,
            tau,
            damp,
            gravity,
            sub_dt,
            rhs_scale: (tau / damp) * n * inv_dt,
            anchor_scale: ((tau * 0.5) / damp) * n * inv_dt,
            friction_pass: (1.0 / micro_steps as f32) * inv_iters,
            avg_gain: tau / damp,
            avg_scale: (damp / tau) * inv_iters,
            gravity_sub: gravity * sub_dt,
            gravity_tick: gravity * dt,
            classes,
            contact_resting_velocity: f32::MAX,
        }
    }
}
