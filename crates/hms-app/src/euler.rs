//! Forward/up vectors <-> X/Y/Z rotation angles.
//!
//! A forge object's orientation is stored as a FORWARD and an UP vector, which is exact but
//! useless to type: nobody knows what "forward = (0.7071, 0.7071, 0)" looks like. The properties
//! panel shows the same rotation as three angles in DEGREES instead, which is what the user
//! actually thinks in ("turn it 90 degrees").
//!
//! Convention, matching the engine's object basis (forward = +X, up = +Z, right-handed):
//!   * **Z** = yaw, turning on the spot (the one people use constantly)
//!   * **Y** = pitch, nose up/down
//!   * **X** = roll, banking around the forward axis
//!
//! The rotation is `Rz(z) * Ry(y) * Rx(x)` (intrinsic Z-Y-X, the usual games convention), so the
//! object's basis vectors are the COLUMNS of that matrix: column 0 is forward, column 2 is up.

use glam::Vec3;

/// An orthonormal basis (forward, left, up) derived from a possibly non-unit, possibly
/// non-perpendicular forward/up pair.
///
/// The engine does NOT re-orthonormalise: inflating a component stretches the model and a
/// non-perpendicular pair shears it. So the angles we report are those of the nearest
/// clean rotation — we orthogonalise `up` against `forward` (Gram-Schmidt) and keep `forward` as
/// the primary axis, which is the choice that leaves yaw exactly where the user expects it.
fn basis(fwd: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let x = fwd.normalize_or_zero();
    let x = if x.length_squared() < 0.5 { Vec3::X } else { x };
    let mut z = up - x * up.dot(x);
    if z.length_squared() < 1e-12 {
        // up is parallel to forward (degenerate) — pick any perpendicular axis
        let r = if x.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        z = r - x * r.dot(x);
    }
    let z = z.normalize_or_zero();
    let y = z.cross(x); // left-hand column of the basis; completes a right-handed set
    (x, y, z)
}

/// Rotation as (x_roll, y_pitch, z_yaw) in DEGREES.
///
/// Gimbal lock (pointing straight up or down) collapses roll and yaw into one degree of freedom;
/// there we report roll = 0 and fold everything into yaw, so the numbers stay stable instead of
/// flickering between equivalent representations.
pub fn to_euler_deg(fwd: [f32; 3], up: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = basis(Vec3::from(fwd), Vec3::from(up));
    // R = [x y z] as columns; for R = Rz(a)Ry(b)Rx(c):
    //   r20 = -sin b, r00 = cos a cos b, r10 = sin a cos b, r21 = cos b sin c, r22 = cos b cos c
    let pitch = (-x.z).clamp(-1.0, 1.0).asin();
    let cb = pitch.cos();
    let (yaw, roll) = if cb.abs() < 1e-4 {
        // looking straight up/down: roll and yaw are the same axis — put it all in yaw
        (y.x.atan2(y.y), 0.0)
    } else {
        (x.y.atan2(x.x), y.z.atan2(z.z))
    };
    [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
}

/// Rebuild forward/up from (x_roll, y_pitch, z_yaw) in DEGREES.
///
/// `fwd_len`/`up_len` scale the results, so an object that was STRETCHED (non-unit forward/up)
/// keeps its stretch when only its angles are edited.
pub fn from_euler_deg(deg: [f32; 3], fwd_len: f32, up_len: f32) -> ([f32; 3], [f32; 3]) {
    let (c, b, a) = (deg[0].to_radians(), deg[1].to_radians(), deg[2].to_radians());
    let (sa, ca) = (a.sin(), a.cos());
    let (sb, cb) = (b.sin(), b.cos());
    let (sc, cc) = (c.sin(), c.cos());
    // columns of Rz(a)*Ry(b)*Rx(c)
    let col0 = Vec3::new(ca * cb, sa * cb, -sb);
    let col2 = Vec3::new(ca * sb * cc + sa * sc, sa * sb * cc - ca * sc, cb * cc);
    let fl = if fwd_len.is_finite() && fwd_len > 1e-6 { fwd_len } else { 1.0 };
    let ul = if up_len.is_finite() && up_len > 1e-6 { up_len } else { 1.0 };
    ((col0 * fl).into(), (col2 * ul).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    #[test]
    fn identity_is_all_zero() {
        let e = to_euler_deg([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(close(e, [0.0, 0.0, 0.0], 1e-3), "identity gave {e:?}");
    }

    #[test]
    fn yaw_is_the_z_angle() {
        // forward swung 90 deg from +X to +Y is a +90 yaw, and nothing else
        let e = to_euler_deg([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((e[2] - 90.0).abs() < 1e-3, "expected yaw 90, got {e:?}");
        assert!(e[0].abs() < 1e-3 && e[1].abs() < 1e-3, "yaw leaked into pitch/roll: {e:?}");
    }

    /// ★ The displayed angles must mean the same thing as the app's own `rotate <axis> <deg>`
    /// command, which uses `Quat::from_axis_angle` (right-hand rule). If these disagreed, typing
    /// 90 into the Y box and pressing rotate-90-about-Y would do different things.
    #[test]
    fn angles_match_the_apps_rotate_command_per_axis() {
        for (axis_i, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            for deg in [15.0f32, 45.0, -30.0, 90.0] {
                let q = glam::Quat::from_axis_angle(axis, deg.to_radians());
                let f: [f32; 3] = (q * Vec3::X).into();
                let u: [f32; 3] = (q * Vec3::Z).into();
                let e = to_euler_deg(f, u);
                assert!((e[axis_i] - deg).abs() < 1e-2,
                    "rotating {deg} deg about axis {axis_i} reported {e:?} — the panel would disagree with `rotate`");
                for other in 0..3 {
                    if other != axis_i {
                        assert!(e[other].abs() < 1e-2,
                            "rotation about axis {axis_i} leaked into axis {other}: {e:?}");
                    }
                }
            }
        }
    }

    /// Positive Y follows the right-hand rule, so it pitches the nose DOWN (toward -Z). Asserted
    /// explicitly because it is the one sign people expect the other way round.
    #[test]
    fn positive_y_pitches_the_nose_down() {
        let (f, _u) = from_euler_deg([0.0, 30.0, 0.0], 1.0, 1.0);
        assert!(f[2] < -0.1, "positive Y should tilt forward toward -Z, got {f:?}");
    }

    #[test]
    fn roll_is_the_x_angle() {
        // forward stays +X, up rolled 90 deg toward -Y
        let e = to_euler_deg([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        assert!((e[0].abs() - 90.0).abs() < 1e-2, "expected |roll| 90, got {e:?}");
        assert!(e[1].abs() < 1e-3 && e[2].abs() < 1e-3, "roll leaked: {e:?}");
    }

    #[test]
    fn round_trips_over_a_sweep_of_angles() {
        for zi in (-180..180).step_by(17) {
            for yi in (-80..80).step_by(13) {
                for xi in (-180..180).step_by(23) {
                    let want = [xi as f32, yi as f32, zi as f32];
                    let (f, u) = from_euler_deg(want, 1.0, 1.0);
                    let got = to_euler_deg(f, u);
                    // compare the RESULTING orientation, since different angle triples can encode
                    // the same rotation
                    let (f2, u2) = from_euler_deg(got, 1.0, 1.0);
                    assert!(close(f, f2, 1e-3) && close(u, u2, 1e-3),
                        "angles {want:?} -> {got:?} changed the orientation\n  fwd {f:?} vs {f2:?}\n  up {u:?} vs {u2:?}");
                }
            }
        }
    }

    #[test]
    fn stretch_is_preserved_when_only_angles_change() {
        let (f, u) = from_euler_deg([0.0, 0.0, 30.0], 2.5, 0.5);
        assert!((Vec3::from(f).length() - 2.5).abs() < 1e-4, "forward length lost");
        assert!((Vec3::from(u).length() - 0.5).abs() < 1e-4, "up length lost");
        // and the angles still read back correctly despite the non-unit vectors
        let e = to_euler_deg(f, u);
        assert!((e[2] - 30.0).abs() < 1e-2, "stretched object reported yaw {}", e[2]);
    }

    #[test]
    fn a_non_perpendicular_pair_still_reports_sane_angles() {
        // The engine does not re-orthonormalise, so the panel must not panic or produce
        // NaN on a sheared object — it reports the nearest clean rotation.
        let e = to_euler_deg([1.0, 0.0, 0.0], [0.4, 0.0, 1.0]);
        assert!(e.iter().all(|v| v.is_finite()), "sheared object gave {e:?}");
        assert!(e[2].abs() < 1e-2, "shear should not invent yaw: {e:?}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        for (f, u) in [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), // up parallel to forward
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            let e = to_euler_deg(f, u);
            assert!(e.iter().all(|v| v.is_finite()), "f={f:?} u={u:?} gave {e:?}");
        }
    }

    #[test]
    fn straight_up_is_stable_not_flickering() {
        // gimbal lock: pointing straight up must give a finite, repeatable answer
        let e = to_euler_deg([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]);
        assert!(e.iter().all(|v| v.is_finite()), "straight up gave {e:?}");
        assert!((e[1] + 90.0).abs() < 1e-2 || (e[1] - 90.0).abs() < 1e-2, "expected +-90 pitch, got {e:?}");
        assert!(e[0].abs() < 1e-3, "roll should be folded into yaw at gimbal lock: {e:?}");
    }
}
