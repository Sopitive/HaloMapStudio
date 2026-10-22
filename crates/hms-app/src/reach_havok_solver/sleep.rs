//! Deactivation: the per-body inactivity windows of the integrator tail (`sub_180D883A0` /
//! `sub_180D86820`, the block after `LABEL_2`) and the island rule that consumes them
//! (`sub_180D72850`: `if (minWindows > 5 && island flags & 0xC) deactivate`).
//!
//! # PROVEN (decompiled, `haloreach_debug.dll`)
//!
//! * every integrate of an awake body increments a byte counter (`motion+0x11`, called `counter`
//!   here); when `counter % 4 == 0` a window check runs: the SHORT window when `counter % 16 != 0`
//!   (so at 4, 8, 12), the LONG window at 16 (the counter is then reset to 0);
//! * a window checks two things against a reference pose stored when the window was last reset:
//!   `|referencePosition - centreOfMass|^2 > threshold` and an orientation test. The position
//!   threshold is the class row's `ref_sq` (short) or `ref4_sq` (long), i.e.
//!   `deactivationReferenceDistance^2 = 0.0025` and `(4 * 0.05)^2 = 0.04`; the orientation
//!   thresholds are `(f32)0x3B6B0000 = 0.00355` (short) and `(f32)0x3D6B0000 = 0.0566` (long);
//! * a failed check RESETS the window: count = 0, the previous count is shifted up, the reference
//!   pose is replaced by the current one (quaternion packed to four bytes, `sub_18100CC60`:
//!   `byte = (bits(196736 + 116.364 q) >> 6) & 0xFF`); a passed check INCREMENTS the 7-bit count
//!   (`count - bit6 + 1`: it saturates at 64);
//! * the body's figure is `max(shortCount, longCount)`; the island's figure is the minimum over its
//!   bodies; the island is deactivated when that figure is **> 5** (and the island allows it).
//!
//! So a body sleeps after **6 quiet 4-tick windows (the fastest possible is the 36th integrate,
//! 0.6 s) without moving 0.025 wu from where the window started**; the request is executed at the
//! START of the next integrate (dirty list) after an energy veto, which also zeroes the velocities
//! (`04-deactivation-islands.md` 5.2). The slow-object damping and zeroing of the solver
//! ([`super::info::DeactClass`]) is what brings a settling box inside that radius.

use super::info::SolverInfo;
use super::solve::BodyAux;
use crate::reach_havok::{HavokBody, MotionType};
use glam::{Quat, Vec3};

/// The deactivation memory of one body.
#[derive(Clone, Debug, Default)]
pub struct SleepState {
    counter: u8,
    /// `[short, long]` window words (`motion+0x12`, `+0x14`).
    win: [u16; 2],
    ref_pos: [Vec3; 2],
    ref_rot: [u32; 2],
    primed: [bool; 2],
    /// `refPosition.w`: the largest `r^2 |w|^2 + |v|^2` sampled in the window (the sleep veto reads it).
    max_energy: [f32; 2],
}

impl SleepState {
    /// Called when the body is (re)activated (`sub_180D59E90`): the counts restart at zero, the
    /// reference poses stay where the last window left them.
    pub fn reset(&mut self) {
        self.win = [0; 2];
    }
    /// The engine's sleep veto (`sub_180D88D20`): a body that is currently faster than the energy
    /// its last window saw does not go to sleep.
    pub fn vetoes_sleep(&self, radius: f32, v: Vec3, w: Vec3) -> bool {
        let r = radius.min(1.0);
        let e = (r * r * w.length_squared() + v.length_squared()) * 0.25 - 0.01;
        let win = if (self.win[0] & 0x7F) <= (self.win[1] & 0x7F) { self.max_energy[1] } else { self.max_energy[0] };
        e > win
    }
    /// `max(short, long)` window count (what the island rule compares with 5).
    pub fn count(&self) -> u32 {
        (self.win[0] & 0x7F).max(self.win[1] & 0x7F) as u32
    }
}

/// `sub_18100CC60`: pack a quaternion into four bytes, `w` in the top byte.
pub fn pack_quat(q: Quat) -> u32 {
    let b = |x: f32| -> u32 { ((196_736.0f32 + x * 116.364).to_bits() >> 6) & 0xFF };
    (b(q.w) << 24) | (b(q.z) << 16) | (b(q.y) << 8) | b(q.x)
}

/// The inverse of [`pack_quat`] (`(byte - 128) * 0.00859375`).
pub fn unpack_quat(p: u32) -> Quat {
    let d = |s: u32| (((p >> s) & 0xFF) as i32 - 128) as f32 * 0.008_593_75;
    Quat::from_xyzw(d(0), d(8), d(16), d(24))
}

fn quat_dist_sq(a: Quat, b: Quat) -> f32 {
    (Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)).length_squared() + (a.w - b.w) * (a.w - b.w)
}

/// One integrate tick of the deactivation tail for an awake body; returns its window figure.
pub fn tick(info: &SolverInfo, aux: &mut BodyAux, b: &HavokBody) -> u32 {
    if matches!(b.motion, MotionType::Fixed | MotionType::Keyframed) {
        return u32::MAX;
    }
    let class = &info.classes[aux.class.min(5) as usize];
    let s = &mut aux.sleep;
    s.counter = s.counter.wrapping_add(1);
    if s.counter & 3 == 0 {
        let long = s.counter & 0xF == 0;
        if long {
            s.counter = 0;
        }
        let w = long as usize;
        let r = aux.radius.min(1.0);
        s.max_energy[w] = s.max_energy[w].max(r * r * b.w.length_squared() + b.v.length_squared());
        let (pos_thr, ang_thr) = if long { (class.ref4_sq, class.ang_long) } else { (class.ref_sq, class.ang_short) };
        let moved = !s.primed[w] || (s.ref_pos[w] - b.com).length_squared() > pos_thr || quat_dist_sq(unpack_quat(s.ref_rot[w]), b.q) > ang_thr;
        let c = s.win[w];
        let count = c & 0x7F;
        if moved {
            s.win[w] = (count) << 7;
            s.ref_pos[w] = b.com;
            s.ref_rot[w] = pack_quat(b.q);
            s.max_energy[w] = 0.0;
            s.primed[w] = true;
        } else {
            s.win[w] = (count - ((c >> 6) & 1) + 1) | (count << 7);
        }
    }
    s.count()
}

/// The island rule: connected components of awake dynamic bodies (joined by broad-phase pairs,
/// `links`: every pair whose tolerance-inflated bounds overlap has an agent, with or without contact
/// points -- `sub_180D66C50` merges the islands when the pair is created) go to sleep
/// together once every member's figure exceeds 5. A body with a zero `class` row index of 1 ("off")
/// or a non-dynamic partner keeps its island awake only through its own figure.
pub fn islands_to_sleep(counts: &[u32], awake: &[bool], links: &[(usize, usize)], aux: &[BodyAux]) -> Vec<Vec<usize>> {
    let n = counts.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let live = |i: usize| awake[i] && counts[i] != u32::MAX;
    for &(a, b) in links {
        if live(a) && live(b) {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            parent[ra] = rb;
        }
    }
    let mut min_count = vec![u32::MAX; n];
    for i in 0..n {
        if live(i) {
            let r = find(&mut parent, i);
            // class OFF (1) / INVALID (0) never deactivates
            let c = if aux[i].class <= 1 { 0 } else { counts[i] };
            min_count[r] = min_count[r].min(c);
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..n {
        if live(i) {
            let r = find(&mut parent, i);
            if min_count[r] > 5 {
                groups.entry(r).or_default().push(i);
            }
        }
    }
    groups.into_values().collect()
}
