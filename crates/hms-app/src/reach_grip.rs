//! #ride-grip  A rider's hands on a vehicle's steering wheel or handlebars.
//!
//! The driver's grip markers (`left_hand`, `right_hand`) are carried by the vehicle's OWN steering
//! node (`b_steering_wheel` on the Warthog; see `reach_vehicle_anim`), and that node turns with the
//! `vehicle:steering` overlay. A rider posed from the seat's idle clip holds the wheel at its
//! centre and nowhere else, so as the wheel turns the hands stay behind. In the engine the rider's
//! hands follow it through animation-driven IK (`jmad` `ik chain events` / `ik chain proxies`,
//! whose `target marker` names a marker on the vehicle).
//!
//! # What this reproduces, and what it does not
//!
//! The per-frame IK data in the animation resource (regions 7..16 of the blob) is NOT decoded, so
//! the engine's chain solve is not reproduced. What IS reproduced is its effect, from first
//! principles that need no undecoded data: the hand is carried RIGIDLY with the marker it grips
//! (the grip offset is measured once, from the seat's own authored idle pose, at the instant the
//! wheel is at rest), and a two-bone solve (upper arm, forearm) puts the hand there. When the wheel
//! is at rest the target IS the authored hand, so the authored pose is not altered at all.
//!
//! The elbow keeps the bend direction the clip authored (the pole is the clip's own elbow offset),
//! so an arm never flips.

use glam::{Mat4, Quat, Vec3};

/// The three nodes of one arm, as indices into the rider's skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmChain {
    pub upper: usize,
    pub fore: usize,
    pub hand: usize,
}

impl ArmChain {
    /// Find the left or right arm by node name. Reach's biped names them `b_l_upperarm`,
    /// `b_l_forearm`, `b_l_hand` (and `b_r_*`).
    pub fn find(node_names: &[String], left: bool) -> Option<ArmChain> {
        let side = if left { "l" } else { "r" };
        let at = |leaf: &str| node_names.iter().position(|n| n == &format!("b_{side}_{leaf}"));
        Some(ArmChain { upper: at("upperarm")?, fore: at("forearm")?, hand: at("hand")? })
    }
}

/// One hand to place: the chain, where the hand's full frame must be (BODY model space), and how
/// much of the correction to apply (`0` = the authored pose, `1` = fully on the target).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grip {
    pub chain: ArmChain,
    pub target: Mat4,
    pub weight: f32,
}

/// Every node at or below `root`, given the parent index of each node (`-1` = root).
fn subtree(parents: &[i16], root: usize) -> Vec<usize> {
    let n = parents.len();
    let mut in_tree = vec![false; n];
    in_tree[root] = true;
    // Parents can come after their children in the array, so sweep until nothing changes.
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            let p = parents[i];
            if !in_tree[i] && p >= 0 && (p as usize) < n && in_tree[p as usize] {
                in_tree[i] = true;
                changed = true;
            }
        }
    }
    (0..n).filter(|&i| in_tree[i]).collect()
}

/// Rotate every matrix in `nodes` by `rot` about `pivot` (all in model space).
fn rotate_about(model: &mut [Mat4], nodes: &[usize], pivot: Vec3, rot: Quat) {
    let m = Mat4::from_translation(pivot) * Mat4::from_quat(rot) * Mat4::from_translation(-pivot);
    for &i in nodes {
        model[i] = m * model[i];
    }
}

/// Place one hand: a two-bone solve of the upper arm and forearm so the hand node reaches
/// `grip.target`'s position, then the hand (and everything below it) turned to its orientation.
///
/// `model` is the rider's per-node MODEL-space pose (parents before children are NOT assumed);
/// `parents` is the skeleton's parent array. A target out of reach is clamped to the arm's full
/// extension rather than extrapolated.
pub fn apply_grip(model: &mut [Mat4], parents: &[i16], grip: &Grip) {
    let w = grip.weight.clamp(0.0, 1.0);
    let ArmChain { upper, fore, hand } = grip.chain;
    if w <= 0.0 || upper >= model.len() || fore >= model.len() || hand >= model.len() {
        return;
    }
    let s = model[upper].w_axis.truncate();
    let e = model[fore].w_axis.truncate();
    let h = model[hand].w_axis.truncate();
    let want = h.lerp(grip.target.w_axis.truncate(), w);
    let (l1, l2) = ((e - s).length(), (h - e).length());
    if l1 < 1e-4 || l2 < 1e-4 {
        return;
    }
    let to_target = want - s;
    let d = to_target.length().clamp((l1 - l2).abs() + 1e-3, l1 + l2 - 1e-3);
    let axis = if to_target.length() > 1e-6 { to_target / to_target.length() } else { (h - s).normalize_or_zero() };
    // The bend stays where the clip put it: the pole is the authored elbow's offset from the
    // shoulder-to-target line.
    let rel = e - s;
    let mut pole = rel - axis * rel.dot(axis);
    if pole.length() < 1e-4 {
        pole = Vec3::Z - axis * Vec3::Z.dot(axis);
    }
    let pole = pole.normalize_or_zero();
    let a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d);
    let hgt = (l1 * l1 - a * a).max(0.0).sqrt();
    let new_e = s + axis * a + pole * hgt;
    // Upper arm: swing the shoulder-to-elbow vector onto the new one, carrying everything below.
    let up_nodes = subtree(parents, upper);
    let r1 = Quat::from_rotation_arc((e - s).normalize(), (new_e - s).normalize());
    let h1 = s + r1 * (h - s);
    rotate_about(model, &up_nodes, s, r1);
    // Forearm: swing elbow-to-hand onto elbow-to-target.
    let fore_nodes = subtree(parents, fore);
    let goal = s + axis * d;
    let r2 = Quat::from_rotation_arc((h1 - new_e).normalize(), (goal - new_e).normalize());
    rotate_about(model, &fore_nodes, new_e, r2);
    // Hand orientation: turn the hand to the target's, by the same weight.
    let hand_nodes = subtree(parents, hand);
    let cur = Quat::from_mat4(&model[hand]).normalize();
    let tgt = Quat::from_mat4(&grip.target).normalize();
    let turn = (tgt * cur.inverse()).normalize();
    let turn = Quat::IDENTITY.slerp(turn, w);
    let pivot = model[hand].w_axis.truncate();
    rotate_about(model, &hand_nodes, pivot, turn);
}

#[cfg(test)]
mod gates {
    //! #ride-grip  Pure gates on a synthetic arm: shoulder at the origin, 0.30 upper arm, 0.28
    //! forearm, hanging along +X with a slight bend toward +Z.

    use super::*;

    fn arm() -> (Vec<Mat4>, Vec<i16>) {
        // 0 chest (root), 1 upper, 2 fore, 3 hand, 4 finger (child of hand)
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.29, 0.0, 0.05),
            Vec3::new(0.55, 0.0, 0.0),
            Vec3::new(0.62, 0.0, 0.0),
        ];
        let model = p.iter().map(|&t| Mat4::from_translation(t)).collect();
        (model, vec![-1, 0, 1, 2, 3])
    }

    fn chain() -> ArmChain {
        ArmChain { upper: 1, fore: 2, hand: 3 }
    }

    #[test]
    fn a_target_at_the_authored_hand_changes_nothing() {
        let (mut m, par) = arm();
        let before = m.clone();
        let g = Grip { chain: chain(), target: before[3], weight: 1.0 };
        apply_grip(&mut m, &par, &g);
        for (i, (a, b)) in m.iter().zip(before.iter()).enumerate() {
            assert!((a.w_axis - b.w_axis).length() < 2e-3, "node {i} moved: {:?} -> {:?}", b.w_axis, a.w_axis);
        }
    }

    #[test]
    fn the_hand_reaches_a_reachable_target_and_the_bones_keep_their_length() {
        let (mut m, par) = arm();
        let (l1, l2) = ((m[2].w_axis - m[1].w_axis).truncate().length(), (m[3].w_axis - m[2].w_axis).truncate().length());
        let target = Vec3::new(0.25, 0.20, 0.15);
        let g = Grip { chain: chain(), target: Mat4::from_translation(target), weight: 1.0 };
        apply_grip(&mut m, &par, &g);
        let hand = m[3].w_axis.truncate();
        assert!((hand - target).length() < 3e-3, "hand at {hand:?}, wanted {target:?}");
        let (n1, n2) = ((m[2].w_axis - m[1].w_axis).truncate().length(), (m[3].w_axis - m[2].w_axis).truncate().length());
        assert!((n1 - l1).abs() < 1e-3 && (n2 - l2).abs() < 1e-3, "bone lengths changed: {l1} {l2} -> {n1} {n2}");
        // The finger (a child of the hand) travelled with it.
        let fingers = m[4].w_axis.truncate();
        assert!(((fingers - hand).length() - 0.07).abs() < 2e-3, "the finger was left behind");
        // The chest (root) never moves.
        assert!(m[0].w_axis.truncate().length() < 1e-6);
    }

    #[test]
    fn an_unreachable_target_extends_the_arm_fully_toward_it_instead_of_stretching() {
        let (mut m, par) = arm();
        let target = Vec3::new(3.0, 0.0, 0.0);
        apply_grip(&mut m, &par, &Grip { chain: chain(), target: Mat4::from_translation(target), weight: 1.0 });
        let reach = (m[3].w_axis - m[1].w_axis).truncate().length();
        assert!(reach > 0.55 && reach < 0.60, "fully extended is l1+l2 = 0.58, got {reach}");
        assert!(m[3].w_axis.x > 0.5, "and it points at the target");
    }

    #[test]
    fn the_elbow_keeps_the_bend_direction_the_clip_authored() {
        let (mut m, par) = arm();
        // The authored elbow is bent toward +Z (0.05 above the shoulder-hand line).
        apply_grip(&mut m, &par, &Grip { chain: chain(), target: Mat4::from_translation(Vec3::new(0.35, 0.2, 0.0)), weight: 1.0 });
        let line = (m[3].w_axis - m[1].w_axis).truncate().normalize();
        let off = (m[2].w_axis - m[1].w_axis).truncate();
        let bend = off - line * off.dot(line);
        assert!(bend.z > 0.0, "the elbow flipped to the other side: {bend:?}");
    }

    #[test]
    fn weight_zero_is_the_authored_pose_and_half_is_halfway() {
        let (m0, par) = arm();
        let target = Vec3::new(0.25, 0.20, 0.15);
        let mut none = m0.clone();
        apply_grip(&mut none, &par, &Grip { chain: chain(), target: Mat4::from_translation(target), weight: 0.0 });
        assert_eq!(none[3].w_axis, m0[3].w_axis);
        let mut half = m0.clone();
        apply_grip(&mut half, &par, &Grip { chain: chain(), target: Mat4::from_translation(target), weight: 0.5 });
        let want = m0[3].w_axis.truncate().lerp(target, 0.5);
        assert!((half[3].w_axis.truncate() - want).length() < 3e-3, "{:?} vs {want:?}", half[3].w_axis);
    }

    #[test]
    fn arms_are_found_by_the_bipeds_own_node_names() {
        let names: Vec<String> =
            ["b_pelvis", "b_l_upperarm", "b_l_forearm", "b_l_hand", "b_r_upperarm", "b_r_forearm", "b_r_hand"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(ArmChain::find(&names, true), Some(ArmChain { upper: 1, fore: 2, hand: 3 }));
        assert_eq!(ArmChain::find(&names, false), Some(ArmChain { upper: 4, fore: 5, hand: 6 }));
        assert_eq!(ArmChain::find(&names[..3], true), None);
    }
}
