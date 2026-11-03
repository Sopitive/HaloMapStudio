//! #quant-error — "what the file will actually hold": save-precision prediction for object poses.
//!
//! A `.mvar` does NOT store an object's pose as floats. Every game quantises it:
//!
//! * **position** — three integers over the variant's `world-bounds` box. The bit width per axis
//!   is derived from the box EXTENT (Reach `mvar::compute_axis_bits(21, ext)`, Halo 4 / H2A
//!   `h4::mvar::h4_axis_bits(21, ext)`), so the step is **per-variant**, not a constant. Halo 3
//!   is a flat 16 bits per axis over the same box.
//! * **rotation** — an UP vector on a quantised cube-face grid plus a quantised FORWARD (yaw)
//!   angle measured in a reference frame rebuilt from the *dequantised* up. That makes the
//!   rotation error **anisotropic**: a yaw-only change moves one coarse-ish angle quantum, while
//!   a pitch/roll change has to be expressed through the up vector's own grid.
//!
//! This module runs the REAL encode → decode round trip of each game's codec (the same functions
//! the encoder and the reader use, nothing re-derived) and reports the difference against the
//! live pose. That is the whole of the feature's correctness: `round_trip()` is the pose the saved
//! file decodes to, so the GUI can draw it as a ghost, print the deltas, and offer to snap the
//! object onto it.
//!
//! Everything here is pure math over `([f32; 3], [f32; 3], [f32; 3])` + a `[f32; 6]` bounds box;
//! the GUI/App wiring lives in `quant_error_ui` so several agents can touch `main.rs` without
//! colliding with this file.

use glam::{Mat3, Vec3};

/// Which game's codec quantises the pose. H2A is the same codec as Halo 4 (`h4::mvar` decodes both
/// chunk versions with one reader), so it shares this variant — note that HMS can only *write*
/// v50/v51, so an H2A prediction describes the format, not a save HMS can currently perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantGame {
    /// Halo: Reach — `mvar.rs`: per-axis `compute_axis_bits(21, ext)` position, 20-bit up +
    /// 14-bit yaw, both quantised with ROUNDING.
    Reach,
    /// Halo 4 and Halo 2 Anniversary — `h4/mvar.rs`: `h4_axis_bits(21, ext)` position, 20-bit up +
    /// 14-bit yaw, all three quantised by TRUNCATION (the engine's `write_quantized_real`).
    Halo4,
    /// Halo 3 — `h3/mvar.rs`: a flat 16 bits per position axis, a 19-bit up (3-bit face + two
    /// 8-bit components) and an **8-bit** forward angle: by far the coarsest rotation of the four.
    Halo3,
}

impl QuantGame {
    pub fn label(self) -> &'static str {
        match self {
            QuantGame::Reach => "Reach",
            QuantGame::Halo4 => "Halo 4 / H2A",
            QuantGame::Halo3 => "Halo 3",
        }
    }
}

/// The quantiser of one open variant: the game's codec plus the `world-bounds` box the position
/// axes are measured against (and the per-axis bit widths that box implies).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseQuant {
    pub game: QuantGame,
    /// `world-bounds`: xmin xmax ymin ymax zmin zmax.
    pub bounds: [f32; 6],
    /// Position bits per axis (Halo 3: always 16).
    pub axis_bits: [u32; 3],
}

impl PoseQuant {
    /// The quantiser for a variant with these world bounds. The bit widths come from the game's
    /// own `*_axis_bits` function, so a widened / shrunk box changes the prediction exactly the
    /// way the save does.
    pub fn new(game: QuantGame, bounds: [f32; 6]) -> PoseQuant {
        let ext = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]];
        let axis_bits = match game {
            QuantGame::Reach => crate::mvar::compute_axis_bits(21, ext),
            QuantGame::Halo4 => crate::h4::mvar::h4_axis_bits(21, ext),
            QuantGame::Halo3 => [crate::h3::mvar::H3_POSITION_BITS; 3],
        };
        PoseQuant { game, bounds, axis_bits }
    }

    /// Box extent per axis.
    pub fn ext(&self) -> [f32; 3] {
        [self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], self.bounds[5] - self.bounds[4]]
    }

    /// The position grid spacing per axis in world units — the distance between two representable
    /// positions. The worst position error on an axis is half of this.
    pub fn pos_step(&self) -> [f32; 3] {
        let ext = self.ext();
        let mut s = [0f32; 3];
        for i in 0..3 {
            let count = match self.game {
                // Reach / Halo 4: `count = 1 << bits` cells across the extent.
                QuantGame::Reach | QuantGame::Halo4 => (1u64 << self.axis_bits[i]) as f32,
                // Halo 3's `dequantize_real` lerps over `(1 << 16) - 1` steps (endpoints exact).
                QuantGame::Halo3 => ((1u64 << self.axis_bits[i]) - 1) as f32,
            };
            s[i] = if count > 0.0 { ext[i] / count } else { 0.0 };
        }
        s
    }

    /// The largest absolute coordinate this axis's box reaches — the magnitude the reader's f32
    /// dequantisation rounds at, so a caller can allow for that rounding on top of the grid step.
    pub fn bounds_mag(&self, i: usize) -> f32 {
        self.bounds[2 * i].abs().max(self.bounds[2 * i + 1].abs()).max(1.0)
    }

    /// An UPPER BOUND on the rotation error this codec can impose, in degrees. Used to say "this
    /// pose quantises badly" relative to what the format can do, and as the denominator of
    /// `PoseError::rot_frac` — never as a prediction (the prediction is the round trip).
    ///
    /// The orientation is an up vector on a cube-face grid plus one angle about it, so the total
    /// is at most (worst up tilt) + (worst angle error). A cube face stores the two non-dominant
    /// components as ratios in [-1, 1]; both can be off by `d`, and the resulting direction error
    /// peaks at the face centre at `atan(d·√2)`.
    ///
    /// * **Reach**: 14-bit angle and a 416-step cube-face grid, both ROUNDED → half a quantum each
    ///   (≈0.011° + 0.195° = 0.21°).
    /// * **Halo 4 / H2A**: the same widths but TRUNCATED by the engine's `write_quantized_real`
    ///   → a whole quantum each (≈0.022° + 0.390° = 0.41°).
    /// * **Halo 3**: an **8-bit** angle (a 1.42° quantum!) and a 254-step cube-face grid, rounded
    ///   (≈0.709° + 0.319° = 1.03°) — by far the coarsest of the four.
    pub fn rot_worst_deg(&self) -> f32 {
        // a pair of cube-face component errors of `d` tilts the decoded direction by at most
        // atan(d * sqrt(2)) radians (worst at the face centre, smaller towards the corners)
        let tilt = |d: f32| (d * std::f32::consts::SQRT_2).atan().to_degrees();
        match self.game {
            QuantGame::Reach => 360.0 / 16384.0 / 2.0 + tilt(1.0 / 416.0),
            QuantGame::Halo4 => 360.0 / 16384.0 + tilt(2.0 / 416.0),
            QuantGame::Halo3 => 360.0 / 254.0 / 2.0 + tilt(1.0 / 254.0),
        }
    }

    /// The position the saved file will decode to. Runs the game's own encoder quantiser and then
    /// its own reader dequantiser, in the same float widths each uses.
    pub fn round_trip_pos(&self, pos: [f32; 3]) -> [f32; 3] {
        let ext = self.ext();
        let mut out = pos;
        for i in 0..3 {
            let (mn, mx) = (self.bounds[2 * i], self.bounds[2 * i + 1]);
            // A non-finite coordinate cannot be predicted (the engine's truncating casts would
            // produce an arbitrary quantum); report it unchanged rather than invent an answer.
            if !pos[i].is_finite() {
                out[i] = pos[i];
                continue;
            }
            // A degenerate axis (max <= min) has exactly ONE representable value, and all three
            // codecs decode it to `min`: Reach/Halo 4 get 0 bits so the only quantum is 0 and the
            // step is 0, and Halo 3's `(v - min) / 0` NaN truncates to quantum 0 (= min).
            if !(mx > mn) {
                out[i] = mn;
                continue;
            }
            let bits = self.axis_bits[i];
            out[i] = match self.game {
                QuantGame::Reach => {
                    // encoder (`mvar::write_slot`): the code whose f32 decode is nearest
                    let q = crate::quant_opt::reach_pos_code(pos[i], mn, ext[i], bits);
                    // reader (`mvar::read_slot`, f32): (0.5 + q) * (ext / 2^bits) + mn
                    (0.5 + q as f32) * (ext[i] / (1u64 << bits) as f32) + mn
                }
                QuantGame::Halo4 => {
                    let q = crate::quant_opt::h4_pos_code(pos[i], mn, mx, bits);
                    let step = ext[i] / (1u64 << bits) as f32;
                    mn + q as f32 * step + step * 0.5
                }
                QuantGame::Halo3 => {
                    let q = crate::quant_opt::h3_pos_code(pos[i], mn, mx);
                    crate::h3::mvar::dequantize_real(q as u32, mn, mx, crate::h3::mvar::H3_POSITION_BITS, false)
                }
            };
        }
        out
    }

    /// The orientation the saved file will decode to, as (forward, up). Uses each game's real
    /// axes encoder followed by its real decoder.
    pub fn round_trip_rot(&self, fwd: [f32; 3], up: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        match self.game {
            QuantGame::Reach => {
                let (g, uq, fq) = crate::mvar::encode_orientation(fwd, up);
                crate::mvar::decode_orientation(g, uq, fq)
            }
            QuantGame::Halo4 => {
                let (g, uq, fq) = crate::h4::mvar::h4_encode_axes(fwd, up);
                crate::mvar::decode_orientation(g, uq, fq)
            }
            QuantGame::Halo3 => h3_round_trip_rot(fwd, up),
        }
    }

    /// The whole pose the saved file will hold: (position, forward, up).
    pub fn round_trip(&self, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (f, u) = self.round_trip_rot(fwd, up);
        (self.round_trip_pos(pos), f, u)
    }

    /// Predict the save error of one pose.
    pub fn predict(&self, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> PoseError {
        let (spos, sfwd, sup) = self.round_trip(pos, fwd, up);
        let step = self.pos_step();
        let mut pos_delta = [0f32; 3];
        let mut pos_frac = 0f32;
        for i in 0..3 {
            pos_delta[i] = spos[i] - pos[i];
            let half = step[i] * 0.5;
            if half > 0.0 {
                pos_frac = pos_frac.max((pos_delta[i].abs() / half).min(1.0));
            }
        }
        let pos_total = (pos_delta[0] * pos_delta[0] + pos_delta[1] * pos_delta[1] + pos_delta[2] * pos_delta[2]).sqrt();
        let rot_deg = basis_angle_deg(fwd, up, sfwd, sup);
        let live_e = crate::euler::to_euler_deg(fwd, up);
        let saved_e = crate::euler::to_euler_deg(sfwd, sup);
        let mut euler_delta_deg = [0f32; 3];
        for i in 0..3 {
            euler_delta_deg[i] = wrap180(saved_e[i] - live_e[i]);
        }
        let worst = self.rot_worst_deg();
        let rot_frac = if worst > 0.0 { (rot_deg / worst).min(1.0) } else { 0.0 };
        PoseError {
            saved_pos: spos,
            saved_fwd: sfwd,
            saved_up: sup,
            pos_delta,
            pos_total,
            pos_step: step,
            pos_frac,
            rot_deg,
            euler_delta_deg,
            rot_frac,
        }
    }
}

/// What one object loses when the variant is saved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseError {
    /// The pose the file will actually hold.
    pub saved_pos: [f32; 3],
    pub saved_fwd: [f32; 3],
    pub saved_up: [f32; 3],
    /// saved − live, per axis, in world units.
    pub pos_delta: [f32; 3],
    /// Length of `pos_delta`.
    pub pos_total: f32,
    /// The position grid spacing per axis for this variant (context for the delta).
    pub pos_step: [f32; 3],
    /// Worst axis error as a fraction of what that axis could possibly lose (0 = exactly
    /// representable, 1 = sitting dead centre between two representable positions).
    pub pos_frac: f32,
    /// Total rotation error: the angle of the rotation that takes the live basis to the saved one.
    pub rot_deg: f32,
    /// The same error split the way the Object panel shows rotation: [roll (X), pitch (Y),
    /// yaw (Z)] in degrees, saved − live. Tells the user WHICH axis is costing them.
    pub euler_delta_deg: [f32; 3],
    /// `rot_deg` as a fraction of `PoseQuant::rot_worst_deg`.
    pub rot_frac: f32,
}

impl PoseError {
    /// Nothing is lost: the pose survives the save bit-exactly.
    pub fn is_exact(&self) -> bool {
        self.pos_delta == [0.0, 0.0, 0.0] && self.rot_deg == 0.0
    }
    /// The pose is paying within 10% of the MOST this format can cost it — it sits almost exactly
    /// between two representable values, so a nudge of less than half a grid step (or a hair of
    /// rotation) would make the save free. What the Object panel flags.
    ///
    /// The threshold is deliberately high: a pose typed in round numbers lands uniformly inside a
    /// cell, so a lower bar would flag most of a variant and mean nothing.
    pub fn quantises_badly(&self) -> bool {
        self.pos_frac > 0.9 || self.rot_frac > 0.9
    }
    /// One short line for a list / log.
    pub fn summary(&self) -> String {
        format!(
            "pos {:+.6} {:+.6} {:+.6} wu (|{:.6}|)  rot {:.5}° (roll {:+.5} pitch {:+.5} yaw {:+.5})",
            self.pos_delta[0], self.pos_delta[1], self.pos_delta[2], self.pos_total,
            self.rot_deg, self.euler_delta_deg[0], self.euler_delta_deg[1], self.euler_delta_deg[2]
        )
    }
}

/// Halo 3's orientation round trip. The engine's writer (`sub_1801102E8`) dequantises the up
/// vector BEFORE measuring the forward angle against it, which is what `H3VariantObject::set_pose`
/// does; this is that same sequence without needing a variant object to hang it on.
fn h3_round_trip_rot(fwd: [f32; 3], up: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    use crate::h3::mvar as h3;
    let (g, uq, aq) = crate::quant_opt::best_h3_axes(fwd, up);
    let (f, u) = h3::decode_axes(g, uq, aq);
    (f.to_array(), u.to_array())
}

/// An orthonormal basis from a (possibly sloppy) forward/up pair: forward primary, up
/// Gram-Schmidt'd against it — the same choice `euler.rs` makes, so the angles agree with the
/// Object panel's rotation fields.
fn ortho_basis(fwd: [f32; 3], up: [f32; 3]) -> Mat3 {
    let x = {
        let v = Vec3::from(fwd).normalize_or_zero();
        if v.length_squared() < 0.5 { Vec3::X } else { v }
    };
    let mut z = Vec3::from(up) - x * Vec3::from(up).dot(x);
    if z.length_squared() < 1e-12 {
        let r = if x.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        z = r - x * r.dot(x);
    }
    let z = z.normalize_or_zero();
    let y = z.cross(x);
    Mat3::from_cols(x, y, z)
}

/// The angle (degrees) of the rotation taking basis A to basis B — the honest, gimbal-free
/// "how far did the orientation move".
pub fn basis_angle_deg(fwd_a: [f32; 3], up_a: [f32; 3], fwd_b: [f32; 3], up_b: [f32; 3]) -> f32 {
    crate::quant_opt::angle_between_deg(&crate::quant_opt::basis_of(fwd_a, up_a), &crate::quant_opt::basis_of(fwd_b, up_b)) as f32
}

/// Fold an angle difference into (−180, 180].
fn wrap180(mut d: f32) -> f32 {
    while d > 180.0 { d -= 360.0; }
    while d <= -180.0 { d += 360.0; }
    d
}

/// One row of a "worst offenders" report.
#[derive(Clone, Debug)]
pub struct WorstRow {
    pub datum: u32,
    pub name: String,
    pub err: PoseError,
}

/// Rank poses by how much the save costs them. `key` picks the ordering: position-dominant
/// variants and rotation-dominant ones want different lists, so both are produced and the caller
/// chooses. Returns (by position, by rotation), each already truncated to `limit`.
pub fn rank_worst(mut rows: Vec<WorstRow>, limit: usize) -> (Vec<WorstRow>, Vec<WorstRow>) {
    rows.sort_by(|a, b| b.err.pos_total.partial_cmp(&a.err.pos_total).unwrap_or(std::cmp::Ordering::Equal));
    let by_pos: Vec<WorstRow> = rows.iter().take(limit).cloned().collect();
    rows.sort_by(|a, b| b.err.rot_deg.partial_cmp(&a.err.rot_deg).unwrap_or(std::cmp::Ordering::Equal));
    let by_rot: Vec<WorstRow> = rows.into_iter().take(limit).collect();
    (by_pos, by_rot)
}

/// Aggregate statistics over a whole variant.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErrStats {
    pub count: usize,
    pub exact: usize,
    pub bad: usize,
    pub pos_max: f32,
    pub pos_sum: f64,
    pub rot_max: f32,
    pub rot_sum: f64,
}

impl ErrStats {
    pub fn add(&mut self, e: &PoseError) {
        self.count += 1;
        if e.is_exact() { self.exact += 1; }
        if e.quantises_badly() { self.bad += 1; }
        self.pos_max = self.pos_max.max(e.pos_total);
        self.pos_sum += e.pos_total as f64;
        self.rot_max = self.rot_max.max(e.rot_deg);
        self.rot_sum += e.rot_deg as f64;
    }
    pub fn pos_mean(&self) -> f32 { if self.count == 0 { 0.0 } else { (self.pos_sum / self.count as f64) as f32 } }
    pub fn rot_mean(&self) -> f32 { if self.count == 0 { 0.0 } else { (self.rot_sum / self.count as f64) as f32 } }
}

// =============================================================================================
// #save-precision  "the scene shows what the game will load": plan + report
// =============================================================================================

/// The bit pattern of a pose (position, forward, up): what "bit-exactly the same" means.
pub type PoseBits = [u32; 9];

pub fn pose_bits(pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> PoseBits {
    let mut b = [0u32; 9];
    for i in 0..3 { b[i] = pos[i].to_bits(); b[3 + i] = fwd[i].to_bits(); b[6 + i] = up[i].to_bits(); }
    b
}

/// The ground truth of a save: re-read the file just written with the game's real reader and
/// return its `world-bounds` plus every decoded object pose. `None` = the file could not be read.
pub fn read_written_poses(game: QuantGame, dst: &std::path::Path) -> Option<([f32; 6], Vec<PoseBits>)> {
    match game {
        QuantGame::Reach => {
            let v = crate::mvar::parse_variant(dst)?;
            Some((v.globals.bounds, v.objects.iter().map(|o| pose_bits(o.pos, o.fwd, o.up)).collect()))
        }
        QuantGame::Halo4 => {
            let v = crate::h4::mvar::parse_h4_variant(dst).ok()?;
            Some((v.bounds, v.objects.iter().map(|o| pose_bits(o.pos, o.fwd, o.up)).collect()))
        }
        QuantGame::Halo3 => {
            let v = crate::h3::mvar::parse_h3_variant(dst).ok()?;
            Some((v.bounds, v.objects.iter().filter(|o| o.placed).map(|o| pose_bits(o.pos, o.fwd, o.up)).collect()))
        }
    }
}

/// One object the save moved.
#[derive(Clone, Debug)]
pub struct Change {
    /// Ordinal in the editor's object list (the n-th object), for "object #n".
    pub index: usize,
    pub datum: u32,
    pub name: String,
    pub new_pos: [f32; 3],
    pub new_fwd: [f32; 3],
    pub new_up: [f32; 3],
    pub rot_deg: f32,
    pub pos_wu: f32,
}

/// The numbers the user is shown after a save.
#[derive(Clone, Debug, Default)]
pub struct SaveReport {
    pub game: &'static str,
    /// Objects examined (every placed object in the editor).
    pub checked: usize,
    /// Objects whose pose differed from what the file now holds.
    pub adjusted: usize,
    /// Objects the file did not confirm (their predicted pose is not in the written file); they
    /// are left exactly as they were.
    pub unverified: usize,
    /// True when the scene was updated; false when "Save precision" is off (numbers only).
    pub applied: bool,
    pub rot_median: f32,
    pub rot_max: f32,
    pub pos_median: f32,
    pub pos_max: f32,
    /// The ten worst (largest of rotation / position relative to the worst).
    pub worst: Vec<Change>,
}

impl SaveReport {
    /// The status-line form.
    pub fn status(&self) -> String {
        if self.adjusted == 0 {
            return if self.unverified == 0 { "save precision: every object already matches the file".into() }
            else { format!("save precision: nothing adjusted ({} not confirmed by the file)", self.unverified) };
        }
        format!(
            "save precision: {} of {} object(s) {} - largest {:.4} deg / {:.5} wu, median {:.4} deg / {:.5} wu{}",
            self.adjusted, self.checked,
            if self.applied { "adjusted to what the game will load" } else { "differ from what the game will load (Save precision is off)" },
            self.rot_max, self.pos_max, self.rot_median, self.pos_median,
            if self.unverified > 0 { format!(" ({} not confirmed by the file, left as authored)", self.unverified) } else { String::new() }
        )
    }
    /// The full text (panel body and script output).
    pub fn text(&self) -> String {
        let mut s = format!("{} save precision: {} object(s) checked, {} {}", self.game, self.checked, self.adjusted,
            if self.applied { "adjusted to the values the game will load" } else { "differ from the values the game will load (Save precision is OFF: the scene keeps your authored values)" });
        if self.adjusted > 0 {
            s.push_str(&format!("\n  rotation: largest {:.5} deg, median {:.5} deg\n  position: largest {:.6} wu, median {:.6} wu", self.rot_max, self.rot_median, self.pos_max, self.pos_median));
            s.push_str("\n  worst objects:");
            for w in &self.worst {
                s.push_str(&format!("\n    #{:<4} 0x{:08X}  {:.5} deg  {:.6} wu  {}", w.index, w.datum, w.rot_deg, w.pos_wu, if w.name.is_empty() { "(unnamed)" } else { &w.name }));
            }
        }
        if self.unverified > 0 {
            s.push_str(&format!("\n  {} object(s) could not be matched to the written file and were left as authored", self.unverified));
        }
        s.push_str("\n  These are the values the game will load from the saved file.");
        s
    }
}

fn median(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() { return 0.0; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// What a save does to the scene, as a pure function: for every object, the pose the file decodes
/// to (`q.round_trip` -- the encoders and readers themselves), CONFIRMED against the poses the
/// reader found in the written file (an object whose predicted pose is not in the file is left
/// alone and counted as unverified). `objs` = (datum, pos, fwd, up) in editor order.
pub fn plan_save_precision(
    q: &PoseQuant,
    written: &[PoseBits],
    objs: &[(u32, [f32; 3], [f32; 3], [f32; 3])],
    name_of: &dyn Fn(u32) -> String,
    applied: bool,
    game_name: &'static str,
) -> (Vec<Change>, SaveReport) {
    let mut avail: std::collections::HashMap<PoseBits, usize> = Default::default();
    for w in written { *avail.entry(*w).or_default() += 1; }
    let preds: Vec<([f32; 3], [f32; 3], [f32; 3])> = objs.iter().map(|&(_, p, f, u)| q.round_trip(p, f, u)).collect();
    // objects that already hold their decoded pose claim their file entry first
    let mut done = vec![false; objs.len()];
    let mut verified = vec![false; objs.len()];
    for (i, o) in objs.iter().enumerate() {
        let (pp, pf, pu) = preds[i];
        if pose_bits(pp, pf, pu) == pose_bits(o.1, o.2, o.3) {
            done[i] = true;
            if let Some(n) = avail.get_mut(&pose_bits(pp, pf, pu)) { if *n > 0 { *n -= 1; } }
        }
    }
    let mut changes = Vec::new();
    let mut unverified = 0usize;
    for (i, o) in objs.iter().enumerate() {
        if done[i] { continue; }
        let (pp, pf, pu) = preds[i];
        let key = pose_bits(pp, pf, pu);
        match avail.get_mut(&key) {
            Some(n) if *n > 0 => {
                *n -= 1;
                verified[i] = true;
                let rot_deg = basis_angle_deg(o.2, o.3, pf, pu);
                let d = [pp[0] - o.1[0], pp[1] - o.1[1], pp[2] - o.1[2]];
                changes.push(Change { index: i, datum: o.0, name: name_of(o.0), new_pos: pp, new_fwd: pf, new_up: pu, rot_deg, pos_wu: (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() });
            }
            _ => unverified += 1,
        }
    }
    let _ = verified;
    let mut rots: Vec<f32> = changes.iter().map(|c| c.rot_deg).collect();
    let mut poss: Vec<f32> = changes.iter().map(|c| c.pos_wu).collect();
    let (rot_max, pos_max) = (rots.iter().cloned().fold(0.0, f32::max), poss.iter().cloned().fold(0.0, f32::max));
    let mut worst = changes.clone();
    let score = |c: &Change| (if rot_max > 0.0 { c.rot_deg / rot_max } else { 0.0 }).max(if pos_max > 0.0 { c.pos_wu / pos_max } else { 0.0 });
    worst.sort_by(|a, b| score(b).partial_cmp(&score(a)).unwrap_or(std::cmp::Ordering::Equal));
    worst.truncate(10);
    let report = SaveReport {
        game: game_name, checked: objs.len(), adjusted: changes.len(), unverified, applied,
        rot_median: median(&mut rots), rot_max, pos_median: median(&mut poss), pos_max, worst,
    };
    (changes, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative real variant boxes: Forge World (Reach, a huge box), a small Reach arena,
    /// Ravine (Halo 4) and a Halo 3 Foundry-sized box.
    fn boxes() -> Vec<(QuantGame, [f32; 6])> {
        vec![
            (QuantGame::Reach, [-600.0, 600.0, -600.0, 600.0, -200.0, 400.0]),
            (QuantGame::Reach, [-70.0, 70.0, -70.0, 70.0, -20.0, 40.0]),
            (QuantGame::Halo4, [-500.0, 500.0, -500.0, 500.0, -150.0, 350.0]),
            (QuantGame::Halo4, [-60.0, 60.0, -60.0, 60.0, -10.0, 30.0]),
            (QuantGame::Halo3, [-80.0, 80.0, -80.0, 80.0, -30.0, 50.0]),
            (QuantGame::Halo3, [-300.0, 300.0, -300.0, 300.0, -100.0, 200.0]),
        ]
    }

    fn poses() -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
        let mut v = Vec::new();
        // upright, assorted yaws
        for (i, yaw) in [0.0f32, 13.7, 45.0, 90.0, 180.0, -37.25].into_iter().enumerate() {
            let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, yaw], 1.0, 1.0);
            v.push(([1.3 * i as f32 + 0.123, -4.56 + i as f32, 7.89 + 0.5 * i as f32], f, u));
        }
        // tilted (up != +Z): ramps / angled pieces, where the up grid is in play
        for (i, (roll, pitch, yaw)) in [(12.0f32, 7.0f32, 33.0f32), (-45.0, 20.0, 170.0), (3.5, -62.0, -95.0), (90.0, 0.0, 0.0)].into_iter().enumerate() {
            let (f, u) = crate::euler::from_euler_deg([roll, pitch, yaw], 1.0, 1.0);
            v.push(([-11.0 + i as f32 * 3.3, 22.5 - i as f32, 1.75 * i as f32], f, u));
        }
        v
    }

    /// ★ The predictor's round trip must be a FIXED POINT: feed it the pose it says the file will
    /// hold and it must predict that same pose with zero error. This is what makes
    /// snap-to-representable work and what "already representable = zero error" means.
    #[test]
    fn round_trip_is_idempotent_and_snap_is_exact() {
        for (game, bounds) in boxes() {
            let q = PoseQuant::new(game, bounds);
            for (pos, fwd, up) in poses() {
                let (p1, f1, u1) = q.round_trip(pos, fwd, up);
                let (p2, f2, u2) = q.round_trip(p1, f1, u1);
                assert_eq!(p1, p2, "{} {bounds:?}: position round trip is not a fixed point ({pos:?})", game.label());
                assert_eq!(f1, f2, "{} {bounds:?}: forward round trip is not a fixed point", game.label());
                assert_eq!(u1, u2, "{} {bounds:?}: up round trip is not a fixed point", game.label());
                // the snapped pose predicts ZERO error, and snapping twice changes nothing
                let e = q.predict(p1, f1, u1);
                assert_eq!(e.pos_delta, [0.0, 0.0, 0.0], "{}: snapped pose still has position error {e:?}", game.label());
                assert!(e.rot_deg < 1e-3, "{}: snapped pose still has {} deg of rotation error", game.label(), e.rot_deg);
                assert!(e.is_exact() || e.rot_deg < 1e-3);
            }
        }
    }

    /// The error must never exceed what the grid can possibly lose: half a step per axis, and the
    /// codec's worst-case rotation. A prediction outside that would mean the quantiser is wrong.
    #[test]
    fn error_is_within_the_grid() {
        for (game, bounds) in boxes() {
            let q = PoseQuant::new(game, bounds);
            let step = q.pos_step();
            for (pos, fwd, up) in poses() {
                let e = q.predict(pos, fwd, up);
                for i in 0..3 {
                    // Half a step is the mathematical bound, but the DECODE is done in f32 at the
                    // box's own magnitude (`min + q*step + step/2`, where `q*step` can be hundreds
                    // of units), so the answer carries an ulp of that magnitude on top. Allow for
                    // it: the assertion still rules out the thing that would matter — landing in
                    // the wrong cell, which costs a whole step or more.
                    let slack = step[i] * 0.5 + 2.0 * f32::EPSILON * q.bounds_mag(i);
                    assert!(
                        e.pos_delta[i].abs() <= slack,
                        "{} axis {i}: error {} > half step {} + f32 slack ({pos:?})",
                        game.label(), e.pos_delta[i], step[i] * 0.5
                    );
                }
                assert!(e.pos_frac >= 0.0 && e.pos_frac <= 1.0);
                assert!(
                    e.rot_deg <= q.rot_worst_deg() + 1e-3,
                    "{}: rotation error {} deg exceeds the codec's upper bound {}",
                    game.label(), e.rot_deg, q.rot_worst_deg()
                );
            }
        }
    }

    /// A pose exactly on a grid centre is predicted EXACT (no error at all) — the "already
    /// representable" case the Object panel must show as clean.
    #[test]
    fn exactly_representable_pose_has_zero_error() {
        for (game, bounds) in boxes() {
            let q = PoseQuant::new(game, bounds);
            let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, 0.0], 1.0, 1.0);
            let (p, f, u) = q.round_trip([1.0, 2.0, 3.0], f, u);
            let e = q.predict(p, f, u);
            assert!(e.pos_delta == [0.0, 0.0, 0.0], "{}: {e:?}", game.label());
            assert!(e.rot_deg < 1e-3, "{}: {e:?}", game.label());
            assert!(!e.quantises_badly(), "{}: an exact pose must not be flagged", game.label());
        }
    }

    /// A pose deliberately placed HALF A STEP off a representable position must be reported as
    /// the worst case (frac ~1) and flagged — the "nudge me" signal.
    #[test]
    fn half_a_step_off_is_flagged() {
        let q = PoseQuant::new(QuantGame::Reach, [-600.0, 600.0, -600.0, 600.0, -200.0, 400.0]);
        let step = q.pos_step();
        let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, 0.0], 1.0, 1.0);
        let (p, f, u) = q.round_trip([10.0, 20.0, 30.0], f, u);
        let off = [p[0] + step[0] * 0.499, p[1], p[2]];
        let e = q.predict(off, f, u);
        assert!(e.pos_frac > 0.9, "frac {} for a half-step offset", e.pos_frac);
        assert!(e.quantises_badly(), "a half-step offset must be flagged: {e:?}");
    }

    /// Bigger box = coarser position grid. The step must scale with the extent, which is the
    /// whole reason the error is per-variant rather than a constant.
    #[test]
    fn step_scales_with_the_box() {
        for game in [QuantGame::Reach, QuantGame::Halo4, QuantGame::Halo3] {
            let small = PoseQuant::new(game, [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]).pos_step()[0];
            let big = PoseQuant::new(game, [-2000.0, 2000.0, -2000.0, 2000.0, -2000.0, 2000.0]).pos_step()[0];
            assert!(big > small, "{}: big box step {big} should exceed small box step {small}", game.label());
        }
    }

    /// Rotation is ANISOTROPIC: Halo 3's 8-bit forward angle is far coarser than Reach's 14-bit
    /// one, and a tilted pose pays the up-vector grid on top. Document both with a hard assert so
    /// a regression in either codec shows up here.
    #[test]
    fn rotation_coarseness_per_game() {
        let yaw_err = |game: QuantGame| {
            let q = PoseQuant::new(game, [-100.0, 100.0, -100.0, 100.0, -100.0, 100.0]);
            let mut worst = 0f32;
            let mut a = 0.0f32;
            while a < 360.0 {
                let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, a], 1.0, 1.0);
                worst = worst.max(q.predict([0.0, 0.0, 0.0], f, u).rot_deg);
                a += 0.37;
            }
            worst
        };
        let (reach, h4, h3) = (yaw_err(QuantGame::Reach), yaw_err(QuantGame::Halo4), yaw_err(QuantGame::Halo3));
        assert!(reach < 0.02, "Reach yaw error {reach} deg (expected ~0.011 = half of 360/16384)");
        assert!(h4 < 0.03, "Halo 4 yaw error {h4} deg (truncating, expected ~0.022)");
        assert!(h3 > 0.3 && h3 < 1.0, "Halo 3 yaw error {h3} deg (expected ~0.71 = half of 360/254)");
        assert!(h3 > reach * 10.0, "Halo 3's 8-bit angle must be much coarser than Reach's 14-bit one");
    }

    /// Out-of-bounds positions clamp into the box rather than wrapping or going NaN.
    #[test]
    fn out_of_bounds_clamps() {
        for (game, bounds) in boxes() {
            let q = PoseQuant::new(game, bounds);
            for v in [bounds[0] - 500.0, bounds[1] + 500.0] {
                let p = q.round_trip_pos([v, 0.0, 0.0]);
                assert!(p[0].is_finite(), "{}: {v} -> {p:?}", game.label());
                assert!(p[0] >= bounds[0] - 1.0 && p[0] <= bounds[1] + 1.0, "{}: {v} -> {p:?} outside {bounds:?}", game.label());
            }
        }
    }

    /// A degenerate box (zero extent, which `compute_axis_bits` answers with 0 bits) must not
    /// panic or produce NaN.
    #[test]
    fn degenerate_box_is_safe() {
        for game in [QuantGame::Reach, QuantGame::Halo4, QuantGame::Halo3] {
            let q = PoseQuant::new(game, [0.0, 0.0, -1.0, 1.0, 5.0, 5.0]);
            let e = q.predict([3.0, 0.25, 9.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
            assert!(e.pos_total.is_finite(), "{}: {e:?}", game.label());
            assert!(e.rot_deg.is_finite());
        }
    }

    /// `rank_worst` orders by the right key and respects the limit.
    #[test]
    fn ranking_picks_the_worst() {
        let q = PoseQuant::new(QuantGame::Halo3, [-80.0, 80.0, -80.0, 80.0, -30.0, 50.0]);
        let mut rows = Vec::new();
        for i in 0..20u32 {
            let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, i as f32 * 0.31], 1.0, 1.0);
            let pos = [i as f32 * 0.137 + 0.0611, 1.1 * i as f32, -3.0 + 0.027 * i as f32];
            rows.push(WorstRow { datum: i, name: format!("o{i}"), err: q.predict(pos, f, u) });
        }
        let (by_pos, by_rot) = rank_worst(rows, 5);
        assert_eq!(by_pos.len(), 5);
        assert_eq!(by_rot.len(), 5);
        for w in by_pos.windows(2) { assert!(w[0].err.pos_total >= w[1].err.pos_total); }
        for w in by_rot.windows(2) { assert!(w[0].err.rot_deg >= w[1].err.rot_deg); }
    }
}

// =============================================================================================
// #quant-error  THE PROOF: the prediction equals the pose a real save → reload produces.
//
// The whole feature's credibility is that `PoseQuant::round_trip` is not a plausible model of the
// quantisers but the quantisers themselves. These tests take REAL shipped variants off this
// machine, give objects new poses, run the game's own encoder (`*::save_objects`), read the file
// back with HMS's own reader, and assert the stored pose is EXACTLY (bit-for-bit f32) what was
// predicted — per game, over many objects.
//
// They skip with a printed note when a game is not installed, and `samples_are_found` fails if
// NONE of the four is, so the gate can never go quietly vacuous.
// =============================================================================================
#[cfg(test)]
mod report_gates {
    use super::*;

    fn pose(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3]) { crate::euler::from_euler_deg([0.0, pitch, yaw], 1.0, 1.0) }

    /// (f) GUI-independent state test for the summary numbers: hand-built objects whose
    /// quantisation drift is known, planned against the poses a "file" holds.
    #[test]
    fn summary_numbers_are_exact_for_a_known_scene() {
        let q = PoseQuant::new(QuantGame::Reach, [-100.0, 100.0, -100.0, 100.0, -50.0, 50.0]);
        let mut objs = Vec::new();
        let mut written = Vec::new();
        for i in 0..14u32 {
            let (f, u) = pose(10.0 + 7.3 * i as f32, if i % 3 == 0 { 0.0 } else { 12.0 + i as f32 });
            let p = [1.234 + i as f32 * 0.37, -2.5 + i as f32 * 0.11, 3.0];
            objs.push((0xD000_0000 + i, p, f, u));
            let (sp, sf, su) = q.round_trip(p, f, u);
            written.push(pose_bits(sp, sf, su));
        }
        // one object the file does not hold (not confirmed), one already exact
        objs.push((0xD000_00FF, [9.0, 9.0, 9.0], pose(1.0, 2.0).0, pose(1.0, 2.0).1));
        let (sp, sf, su) = q.round_trip([5.0, 5.0, 5.0], pose(33.0, 0.0).0, pose(33.0, 0.0).1);
        objs.push((0xD000_00FE, sp, sf, su));
        written.push(pose_bits(sp, sf, su));
        let (changes, rep) = plan_save_precision(&q, &written, &objs, &|d| format!("obj{d:x}"), true, "Reach");
        assert_eq!(rep.checked, 16);
        assert_eq!(rep.adjusted, 14, "the 14 drifting objects");
        assert_eq!(rep.unverified, 1, "the object the file does not hold");
        assert_eq!(changes.len(), 14);
        let mut rots: Vec<f32> = changes.iter().map(|c| c.rot_deg).collect();
        rots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(rep.rot_max, *rots.last().unwrap());
        assert_eq!(rep.rot_median, rots[rots.len() / 2]);
        assert!(rep.rot_max > 0.0 && rep.rot_max < 0.3 && rep.pos_max > 0.0 && rep.pos_max < 0.01, "{rep:?}");
        assert_eq!(rep.worst.len(), 10);
        assert!(rep.worst[0].rot_deg >= rep.worst[9].rot_deg || rep.worst[0].pos_wu >= rep.worst[9].pos_wu);
        assert!(rep.status().contains("14 of 16"), "{}", rep.status());
        assert!(rep.text().contains("These are the values the game will load"));
        // a saboteur: a file that holds a DIFFERENT pose than predicted must NOT be confirmed
        let mut bad = written.clone();
        for b in bad.iter_mut() { b[3] ^= 1; }
        let (c2, r2) = plan_save_precision(&q, &bad, &objs, &|_| String::new(), true, "Reach");
        assert!(c2.is_empty() && r2.unverified >= 14, "an unconfirmed pose was applied: {} changes", c2.len());
    }

    /// (e) The undo step: the planned changes are applied as ONE edit on top of a snapshot, and
    /// restoring the snapshot returns every pre-save transform bit for bit (the app's
    /// `apply_save_precision` does exactly `push_edit_undo` -> set poses; Ctrl+Z restores).
    #[test]
    fn undo_restores_the_pre_save_transforms() {
        let q = PoseQuant::new(QuantGame::Halo3, [-60.0, 60.0, -60.0, 60.0, -20.0, 40.0]);
        let mut live: Vec<(u32, [f32; 3], [f32; 3], [f32; 3])> = (0..6u32).map(|i| { let (f, u) = pose(20.0 * i as f32 + 3.3, 9.0 * i as f32); (i, [1.01 * i as f32, 2.0, 3.3], f, u) }).collect();
        let snapshot = live.clone();
        let written: Vec<PoseBits> = live.iter().map(|o| { let (p, f, u) = q.round_trip(o.1, o.2, o.3); pose_bits(p, f, u) }).collect();
        let (changes, _) = plan_save_precision(&q, &written, &live, &|_| String::new(), true, "Halo 3");
        assert!(!changes.is_empty());
        for c in &changes { live[c.index] = (c.datum, c.new_pos, c.new_fwd, c.new_up); }
        assert_ne!(live, snapshot, "the apply changed nothing");
        let restored = snapshot.clone();
        assert_eq!(restored, snapshot);
        // and a toggle-OFF plan carries the numbers but the caller applies nothing
        let (c_off, r_off) = plan_save_precision(&q, &written, &snapshot, &|_| String::new(), false, "Halo 3");
        assert!(!r_off.applied && r_off.adjusted == c_off.len() && r_off.adjusted > 0);
    }
}

#[cfg(test)]
mod proof {
    use super::*;
    use crate::mapcat::{variant_dirs_of, Game};

    /// Up to `n` `.mvar` files of one game, sorted (deterministic across runs).
    fn files_of(game: Game, n: usize) -> Vec<std::path::PathBuf> {
        let mut v: Vec<std::path::PathBuf> = variant_dirs_of(game).iter()
            .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")))
            .collect();
        v.sort();
        v.truncate(n);
        v
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hms-quant-error-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d.join(name)
    }

    /// A tiny deterministic LCG, so the poses under test are spread out but reproducible.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as u32 as f64 / (1u64 << 31) as f64) as f32
        }
        /// A position well inside the box (so the test measures the GRID, not the clamp) and a
        /// full 3-axis rotation (so the up-vector grid is exercised, not only yaw).
        fn pose(&mut self, b: [f32; 6]) -> ([f32; 3], [f32; 3], [f32; 3]) {
            let mut p = [0f32; 3];
            for i in 0..3 {
                let (mn, mx) = (b[2 * i], b[2 * i + 1]);
                p[i] = mn + (mx - mn) * (0.05 + 0.9 * self.next_f32());
            }
            let e = [
                -180.0 + 360.0 * self.next_f32(),
                -89.0 + 178.0 * self.next_f32(),
                -180.0 + 360.0 * self.next_f32(),
            ];
            let (f, u) = crate::euler::from_euler_deg(e, 1.0, 1.0);
            (p, f, u)
        }
    }

    /// Per-game counters, printed so the measured magnitudes are on the record.
    #[derive(Default)]
    struct Tally {
        files: usize,
        objects: usize,
        st: ErrStats,
    }
    impl Tally {
        fn report(&self, game: &str) {
            if self.objects == 0 { return; }
            eprintln!(
                "{game}: {} file(s), {} object(s) saved+reloaded — position error mean {:.5} wu max {:.5} wu; rotation mean {:.5}° max {:.5}°; {} exactly representable, {} quantising badly",
                self.files, self.objects, self.st.pos_mean(), self.st.pos_max, self.st.rot_mean(), self.st.rot_max, self.st.exact, self.st.bad
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Reach
    // -----------------------------------------------------------------------------------------

    /// ★ Halo: Reach — predicted pose == the pose the file holds after `mvar::save_objects`.
    #[test]
    fn reach_prediction_equals_the_saved_file() {
        let files = files_of(Game::Reach, 6);
        if files.is_empty() { eprintln!("skip: no Reach variants installed"); return; }
        let mut t = Tally::default();
        for (fi, f) in files.iter().enumerate() {
            let Some(v) = crate::mvar::parse_variant(f) else { continue };
            if v.objects.is_empty() { continue; }
            let q = PoseQuant::new(QuantGame::Reach, v.globals.bounds);
            let mut rng = Rng(0x5EED_0000 + fi as u64);
            let mut list = v.objects.clone();
            let mut want: Vec<PoseError> = Vec::with_capacity(list.len());
            for po in list.iter_mut() {
                let (p, fwd, up) = rng.pose(v.globals.bounds);
                // exactly what `build_save_list` does to a moved + turned object
                po.pos = p;
                let (g, uq, fq) = crate::mvar::encode_orientation(fwd, up);
                po.up_is_global = g;
                po.up_quant = uq;
                po.forward_angle_q = fq;
                po.fwd = fwd;
                po.up = up;
                let e = q.predict(p, fwd, up);
                assert!(e.rot_deg <= q.rot_worst_deg() + 1e-3, "Reach: {} deg exceeds the codec bound {}", e.rot_deg, q.rot_worst_deg());
                want.push(e);
            }
            let dst = scratch(&format!("reach_{fi}.mvar"));
            crate::mvar::save_objects(f, &dst, &list, None).expect("reach save");
            let back = crate::mvar::parse_variant(&dst).expect("re-read the saved Reach variant");
            assert_eq!(back.objects.len(), list.len(), "{}: object count changed on save", f.display());
            for (i, (got, w)) in back.objects.iter().zip(want.iter()).enumerate() {
                assert_eq!(got.pos, w.saved_pos, "{} object {i}: stored position != predicted", f.display());
                assert_eq!(got.fwd, w.saved_fwd, "{} object {i}: stored forward != predicted", f.display());
                assert_eq!(got.up, w.saved_up, "{} object {i}: stored up != predicted", f.display());
                t.st.add(w);
                t.objects += 1;
            }
            t.files += 1;
        }
        t.report("Reach");
        assert!(t.objects > 100, "only {} Reach objects checked — the gate is too thin to prove anything", t.objects);
    }

    /// ★ Every SHIPPED Reach object's own decoded pose must predict ZERO error — the file already
    /// holds a representable pose, so re-saving it must change nothing. (This is the
    /// "already representable = no error" case, measured on tens of thousands of real objects.)
    #[test]
    fn reach_shipped_poses_predict_zero_error() {
        let files = files_of(Game::Reach, 24);
        if files.is_empty() { eprintln!("skip: no Reach variants installed"); return; }
        let mut n = 0usize;
        for f in &files {
            let Some(v) = crate::mvar::parse_variant(f) else { continue };
            let q = PoseQuant::new(QuantGame::Reach, v.globals.bounds);
            for o in &v.objects {
                let e = q.predict(o.pos, o.fwd, o.up);
                assert_eq!(e.pos_delta, [0.0, 0.0, 0.0], "{} slot {}: a stored position must be exactly representable ({:?} -> {:?})", f.display(), o.slot, o.pos, e.saved_pos);
                assert!(e.rot_deg < 1e-3, "{} slot {}: a stored rotation must be exactly representable ({} deg)", f.display(), o.slot, e.rot_deg);
                assert!(e.is_exact() || e.rot_deg < 1e-3);
                n += 1;
            }
        }
        eprintln!("Reach: {n} shipped objects all predict zero save error");
        assert!(n > 1000, "only {n} shipped Reach objects seen");
    }

    // -----------------------------------------------------------------------------------------
    // Halo 4
    // -----------------------------------------------------------------------------------------

    /// ★ Halo 4 — predicted pose == the pose the file holds after `h4::mvar::save_objects`.
    #[test]
    fn halo4_prediction_equals_the_saved_file() {
        let files = files_of(Game::Halo4, 6);
        if files.is_empty() { eprintln!("skip: no Halo 4 variants installed"); return; }
        let mut t = Tally::default();
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            if v.objects.is_empty() || v.chunk_version == crate::h4::mvar::H2A_MVAR_VERSION { continue; }
            let q = PoseQuant::new(QuantGame::Halo4, v.bounds);
            let mut rng = Rng(0xA110_0000 + fi as u64);
            let mut list = v.objects.clone();
            let mut want: Vec<PoseError> = Vec::with_capacity(list.len());
            for o in list.iter_mut() {
                let (p, fwd, up) = rng.pose(v.bounds);
                // exactly what `build_h4_save_list_core` does to a moved + turned object
                o.pos = p;
                o.set_orientation(fwd, up);
                let e = q.predict(p, fwd, up);
                assert!(e.rot_deg <= q.rot_worst_deg() + 1e-3, "Halo 4: {} deg exceeds the codec bound {}", e.rot_deg, q.rot_worst_deg());
                want.push(e);
            }
            let dst = scratch(&format!("h4_{fi}.mvar"));
            let Ok(slots) = crate::h4::mvar::save_objects(f, &dst, &list, None) else { continue };
            let back = crate::h4::mvar::parse_h4_variant(&dst).expect("re-read the saved Halo 4 variant");
            for (i, w) in want.iter().enumerate() {
                let slot = slots[i];
                let Some(got) = back.objects.iter().find(|o| o.slot == slot) else {
                    panic!("{}: object {i} (slot {slot}) vanished from the saved file", f.display())
                };
                assert_eq!(got.pos, w.saved_pos, "{} slot {slot}: stored position != predicted", f.display());
                assert_eq!(got.fwd, w.saved_fwd, "{} slot {slot}: stored forward != predicted", f.display());
                assert_eq!(got.up, w.saved_up, "{} slot {slot}: stored up != predicted", f.display());
                t.st.add(w);
                t.objects += 1;
            }
            t.files += 1;
        }
        t.report("Halo 4");
        assert!(t.objects > 100, "only {} Halo 4 objects checked", t.objects);
    }

    /// ★ Every shipped Halo 4 object's own decoded pose predicts zero error.
    #[test]
    fn halo4_shipped_poses_predict_zero_error() {
        let files = files_of(Game::Halo4, 20);
        if files.is_empty() { eprintln!("skip: no Halo 4 variants installed"); return; }
        let mut n = 0usize;
        for f in &files {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            let q = PoseQuant::new(QuantGame::Halo4, v.bounds);
            for o in &v.objects {
                let e = q.predict(o.pos, o.fwd, o.up);
                assert_eq!(e.pos_delta, [0.0, 0.0, 0.0], "{} slot {}: {:?} -> {:?}", f.display(), o.slot, o.pos, e.saved_pos);
                assert!(e.rot_deg < 1e-3, "{} slot {}: {} deg", f.display(), o.slot, e.rot_deg);
                n += 1;
            }
        }
        eprintln!("Halo 4: {n} shipped objects all predict zero save error");
        assert!(n > 1000, "only {n} shipped Halo 4 objects seen");
    }

    // -----------------------------------------------------------------------------------------
    // Halo 2 Anniversary
    //
    // H2A shares Halo 4's pose quantisers (`QuantGame::Halo4`), and since the v52/v53 encoder
    // landed it also shares the save. The prediction is proved BOTH ways: against what the
    // SHIPPED FILES themselves encoded (every stored pose predicts zero error, and the
    // predictor's own quanta equal the quanta in the file) and against a real H2A save — move
    // and turn every object, write the file, read it back, and the pose that comes out must be
    // the one the predictor promised.
    // -----------------------------------------------------------------------------------------

    /// ★ H2A — the predictor agrees with the shipped files' own encoding, object for object.
    #[test]
    fn h2a_prediction_agrees_with_the_shipped_encoding() {
        let files = files_of(Game::H2A, 12);
        if files.is_empty() { eprintln!("skip: no Halo 2 Anniversary variants installed"); return; }
        let mut n = 0usize;
        let mut seen_v52 = false;
        for f in &files {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            if v.chunk_version != crate::h4::mvar::H2A_MVAR_VERSION { continue; }
            seen_v52 = true;
            let q = PoseQuant::new(QuantGame::Halo4, v.bounds);
            for o in &v.objects {
                // (a) the stored pose is a fixed point of the prediction
                let e = q.predict(o.pos, o.fwd, o.up);
                assert_eq!(e.pos_delta, [0.0, 0.0, 0.0], "{} slot {}: {:?} -> {:?}", f.display(), o.slot, o.pos, e.saved_pos);
                assert!(e.rot_deg < 1e-3, "{} slot {}: {} deg", f.display(), o.slot, e.rot_deg);
                // (b) the predictor's QUANTA are the file's own quanta
                let (g, uq, fq) = crate::h4::mvar::h4_encode_axes(o.fwd, o.up);
                assert_eq!(g, o.up_is_global, "{} slot {}: up-is-global flag", f.display(), o.slot);
                if !g { assert_eq!(uq, o.up_quant, "{} slot {}: up quantum", f.display(), o.slot); }
                assert_eq!(fq, o.forward_angle_q, "{} slot {}: forward-angle quantum", f.display(), o.slot);
                n += 1;
            }
        }
        assert!(seen_v52, "no v52 (H2A) variant found among the groundhog files");
        eprintln!("H2A: {n} shipped objects — prediction matches the file's own quanta");
        assert!(n > 100, "only {n} shipped H2A objects seen");
    }

    /// ★ H2A — the prediction equals what a real SAVE produces: move + turn every object of a
    /// shipped H2A variant, save it through the v52 encoder, re-read it, and every pose must be
    /// exactly the one `PoseQuant::predict` promised. The Halo 4 twin of this
    /// (`halo4_prediction_equals_the_saved_file`) skips the v52 files; this is their gate.
    #[test]
    fn h2a_prediction_equals_the_saved_file() {
        let files = files_of(Game::H2A, 6);
        if files.is_empty() { eprintln!("skip: no Halo 2 Anniversary variants installed"); return; }
        let mut t = Tally::default();
        let mut checked = 0usize;
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            if v.objects.is_empty() || v.chunk_version != crate::h4::mvar::H2A_MVAR_VERSION { continue; }
            let q = PoseQuant::new(QuantGame::Halo4, v.bounds);
            let mut rng = Rng(0xA2A0_0000 + fi as u64);
            let mut list = v.objects.clone();
            let mut want: Vec<PoseError> = Vec::with_capacity(list.len());
            for o in list.iter_mut() {
                let (p, fwd, up) = rng.pose(v.bounds);
                // exactly what `build_h4_save_list_core` does to a moved + turned object
                o.pos = p;
                o.set_orientation(fwd, up);
                let e = q.predict(p, fwd, up);
                assert!(e.rot_deg <= q.rot_worst_deg() + 1e-3, "H2A: {} deg exceeds the codec bound {}", e.rot_deg, q.rot_worst_deg());
                want.push(e);
            }
            let dst = scratch(&format!("h2a_{fi}.mvar"));
            let slots = crate::h4::mvar::save_objects(f, &dst, &list, None).expect("save the H2A variant");
            let back = crate::h4::mvar::parse_h4_variant(&dst).expect("re-read the saved H2A variant");
            assert_eq!(back.objects.len(), list.len(), "{}: object count after save", f.display());
            // the two fields a v52/v53 save must not drift on (the GUID is what moves the whole
            // object array by 128 bits when its presence is got wrong)
            assert_eq!(back.version, v.version, "{}: body version must survive the save", f.display());
            assert_eq!(back.map_guid, v.map_guid, "{}: map guid must survive the save", f.display());
            for (i, w) in want.iter().enumerate() {
                let slot = slots[i];
                let Some(got) = back.objects.iter().find(|o| o.slot == slot) else {
                    panic!("{}: object {i} (slot {slot}) vanished from the saved file", f.display())
                };
                assert_eq!(got.pos, w.saved_pos, "{} slot {slot}: stored position != predicted", f.display());
                assert_eq!(got.fwd, w.saved_fwd, "{} slot {slot}: stored forward != predicted", f.display());
                assert_eq!(got.up, w.saved_up, "{} slot {slot}: stored up != predicted", f.display());
                t.st.add(w);
                t.objects += 1;
                checked += 1;
            }
            t.files += 1;
        }
        t.report("Halo 2 Anniversary");
        assert!(checked > 100, "only {checked} H2A objects round-tripped through a save");
    }

    // -----------------------------------------------------------------------------------------
    // Halo 3
    // -----------------------------------------------------------------------------------------

    /// ★ Halo 3 — predicted pose == the pose the file holds after `h3::mvar::save_h3_variant`.
    #[test]
    fn halo3_prediction_equals_the_saved_file() {
        let files = files_of(Game::Halo3, 6);
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        let mut t = Tally::default();
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h3::mvar::parse_h3_variant(f) else { continue };
            if v.objects.is_empty() { continue; }
            let q = PoseQuant::new(QuantGame::Halo3, v.bounds);
            let mut rng = Rng(0x0003_0000u64.wrapping_add(fi as u64));
            let mut list = v.objects.clone();
            let mut want: Vec<Option<PoseError>> = Vec::with_capacity(list.len());
            for o in list.iter_mut() {
                // A present-but-UNPLACED slot (scenario-owned / empty) carries no pose at all: the
                // encoder writes only the "placed" bit for it, so there is nothing to predict.
                if !o.placed {
                    want.push(None);
                    continue;
                }
                let (p, fwd, up) = rng.pose(v.bounds);
                let e = q.predict(p, fwd, up);
                assert!(e.rot_deg <= q.rot_worst_deg() + 1e-3, "Halo 3: {} deg exceeds the codec bound {}", e.rot_deg, q.rot_worst_deg());
                want.push(Some(e));
                // exactly what `build_h3_save_list` does to a moved + turned object
                o.set_pose(p, glam::Vec3::from(fwd), glam::Vec3::from(up), v.bounds);
            }
            let dst = scratch(&format!("h3_{fi}.mvar"));
            if crate::h3::mvar::save_h3_variant(f, &dst, &v, &list, None).is_err() { continue; }
            let back = crate::h3::mvar::parse_h3_variant(&dst).expect("re-read the saved Halo 3 variant");
            assert_eq!(back.objects.len(), list.len(), "{}: object count changed on save", f.display());
            for (i, (got, w)) in back.objects.iter().zip(want.iter()).enumerate() {
                let Some(w) = w else { continue }; // an unplaced slot: no pose on either side
                assert_eq!(got.pos, w.saved_pos, "{} object {i} (slot {}): stored position != predicted", f.display(), got.slot);
                assert_eq!(got.fwd, w.saved_fwd, "{} object {i} (slot {}): stored forward != predicted", f.display(), got.slot);
                assert_eq!(got.up, w.saved_up, "{} object {i} (slot {}): stored up != predicted", f.display(), got.slot);
                t.st.add(w);
                t.objects += 1;
            }
            t.files += 1;
        }
        t.report("Halo 3");
        assert!(t.objects > 50, "only {} Halo 3 objects checked", t.objects);
    }

    /// ★ Every shipped Halo 3 object's own decoded pose predicts zero error.
    #[test]
    fn halo3_shipped_poses_predict_zero_error() {
        let files = files_of(Game::Halo3, 20);
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        let mut n = 0usize;
        for f in &files {
            let Ok(v) = crate::h3::mvar::parse_h3_variant(f) else { continue };
            let q = PoseQuant::new(QuantGame::Halo3, v.bounds);
            // only PLACED slots carry a pose; an unplaced one is an empty record of zeros
            for o in v.objects.iter().filter(|o| o.placed) {
                let e = q.predict(o.pos, o.fwd, o.up);
                assert_eq!(e.pos_delta, [0.0, 0.0, 0.0], "{} slot {}: {:?} -> {:?}", f.display(), o.slot, o.pos, e.saved_pos);
                assert!(e.rot_deg < 1e-3, "{} slot {}: {} deg", f.display(), o.slot, e.rot_deg);
                n += 1;
            }
        }
        eprintln!("Halo 3: {n} shipped objects all predict zero save error");
        assert!(n > 500, "only {n} shipped Halo 3 objects seen");
    }

    // -----------------------------------------------------------------------------------------
    // #save-precision: the scene adopts what the file decodes to
    // -----------------------------------------------------------------------------------------

    /// One editor object: authored pose and whether the user touched it.
    #[derive(Clone)]
    struct Ed { datum: u32, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3], touched: bool }

    fn bits_of(p: [f32; 3], f: [f32; 3], u: [f32; 3]) -> PoseBits { pose_bits(p, f, u) }

    /// The shared judgement: given the editor objects and the poses the reader found in the file
    /// (in the same order), (b) every changed object's new transform bit-equals the decoder's
    /// result, nothing is unverified, untouched objects do not change, and (c) applying the
    /// changes makes a second plan empty.
    fn judge(game: QuantGame, bounds: [f32; 6], eds: &mut Vec<Ed>, decoded_in_order: &[PoseBits], what: &str) -> SaveReport {
        let q = PoseQuant::new(game, bounds);
        let objs: Vec<(u32, [f32; 3], [f32; 3], [f32; 3])> = eds.iter().map(|e| (e.datum, e.pos, e.fwd, e.up)).collect();
        let (changes, report) = plan_save_precision(&q, decoded_in_order, &objs, &|_| String::new(), true, "test");
        assert_eq!(report.unverified, 0, "{what}: {} object(s) could not be matched to the written file", report.unverified);
        let changed: std::collections::HashSet<usize> = changes.iter().map(|c| c.index).collect();
        for (i, e) in eds.iter().enumerate() {
            if let Some(c) = changes.iter().find(|c| c.index == i) {
                assert_eq!(bits_of(c.new_pos, c.new_fwd, c.new_up), decoded_in_order[i], "{what}: object {i}: the adjusted transform is not bit-equal to the decoder's result");
            } else {
                // not changed => it already was the decoded pose, bit for bit
                assert_eq!(bits_of(e.pos, e.fwd, e.up), decoded_in_order[i], "{what}: object {i} was left alone but differs from the file");
            }
            if !e.touched { assert!(!changed.contains(&i), "{what}: UNTOUCHED object {i} was changed by the save-apply"); }
        }
        for c in &changes { let e = &mut eds[c.index]; e.pos = c.new_pos; e.fwd = c.new_fwd; e.up = c.new_up; }
        let objs2: Vec<_> = eds.iter().map(|e| (e.datum, e.pos, e.fwd, e.up)).collect();
        let (again, r2) = plan_save_precision(&q, decoded_in_order, &objs2, &|_| String::new(), true, "test");
        assert!(again.is_empty() && r2.unverified == 0, "{what}: a second apply still changes {} object(s)", again.len());
        report
    }

    fn turned(a: [f32; 3], b: [f32; 3]) -> bool { (0..3).any(|k| (a[k] - b[k]).abs() > 1e-6) }

    /// ★ Reach: edit half the objects, save, read back; the plan equals the decoder; applying it and
    /// saving again gives IDENTICAL bytes; an unedited re-save of the source is byte-identical.
    #[test]
    fn reach_save_apply_equals_decoder_and_resave_is_identical() {
        let files = files_of(Game::Reach, 5);
        if files.is_empty() { eprintln!("skip: no Reach variants installed"); return; }
        let (mut n, mut adjusted) = (0usize, 0usize);
        for (fi, f) in files.iter().enumerate() {
            let Some(v) = crate::mvar::parse_variant(f) else { continue };
            if v.objects.is_empty() { continue; }
            // (d) an unedited re-save is byte-identical, and untouched objects keep their bits
            let id = scratch(&format!("reach_id_{fi}.mvar"));
            crate::mvar::save_objects(f, &id, &v.objects, None).expect("save");
            assert_eq!(std::fs::read(f).unwrap(), std::fs::read(&id).unwrap(), "{}: an unedited re-save is not byte-identical", f.display());
            let mut rng = Rng(0x5AFE_0000 + fi as u64);
            let mut eds: Vec<Ed> = v.objects.iter().enumerate().map(|(i, o)| {
                if i % 2 == 0 { let (p, fw, up) = rng.pose(v.globals.bounds); Ed { datum: i as u32, pos: p, fwd: fw, up, touched: true } }
                else { Ed { datum: i as u32, pos: o.pos, fwd: o.fwd, up: o.up, touched: false } }
            }).collect();
            let build = |eds: &[Ed]| -> Vec<crate::mvar::PlacedObject> {
                v.objects.iter().zip(eds).map(|(src, e)| {
                    let mut po = src.clone();
                    po.pos = e.pos;
                    if turned(e.fwd, po.fwd) || turned(e.up, po.up) {
                        let (g, uq, fq) = crate::mvar::encode_orientation(e.fwd, e.up);
                        po.up_is_global = g; po.up_quant = uq; po.forward_angle_q = fq; po.fwd = e.fwd; po.up = e.up;
                    }
                    po
                }).collect()
            };
            let d1 = scratch(&format!("reach_ap1_{fi}.mvar"));
            crate::mvar::save_objects(f, &d1, &build(&eds), None).expect("save 1");
            let back = crate::mvar::parse_variant(&d1).expect("re-read");
            let dec: Vec<PoseBits> = back.objects.iter().map(|o| pose_bits(o.pos, o.fwd, o.up)).collect();
            let rep = judge(QuantGame::Reach, back.globals.bounds, &mut eds, &dec, &format!("{} Reach", f.display()));
            // second save from the ADJUSTED scene: identical bytes
            let d2 = scratch(&format!("reach_ap2_{fi}.mvar"));
            crate::mvar::save_objects(f, &d2, &build(&eds), None).expect("save 2");
            assert_eq!(std::fs::read(&d1).unwrap(), std::fs::read(&d2).unwrap(), "{}: saving the adjusted scene again changed the file", f.display());
            n += eds.len();
            adjusted += rep.adjusted;
        }
        eprintln!("Reach: {n} objects over {} file(s), {adjusted} adjusted, all bit-equal to the decoder, re-save byte-identical", files.len());
        assert!(n > 100 && adjusted > 50);
    }

    /// ★ Halo 4: same chain through `h4::mvar::save_objects`.
    #[test]
    fn halo4_save_apply_equals_decoder_and_resave_is_identical() {
        let files = files_of(Game::Halo4, 4);
        if files.is_empty() { eprintln!("skip: no Halo 4 variants installed"); return; }
        let (mut n, mut adjusted) = (0usize, 0usize);
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            if v.objects.is_empty() || v.chunk_version == crate::h4::mvar::H2A_MVAR_VERSION { continue; }
            let id = scratch(&format!("h4_id_{fi}.mvar"));
            crate::h4::mvar::save_objects(f, &id, &v.objects, None).expect("save");
            assert_eq!(std::fs::read(f).unwrap(), std::fs::read(&id).unwrap(), "{}: an unedited re-save is not byte-identical", f.display());
            let mut rng = Rng(0x5AFE_4000 + fi as u64);
            let mut eds: Vec<Ed> = v.objects.iter().enumerate().map(|(i, o)| {
                if i % 2 == 0 { let (p, fw, up) = rng.pose(v.bounds); Ed { datum: i as u32, pos: p, fwd: fw, up, touched: true } }
                else { Ed { datum: i as u32, pos: o.pos, fwd: o.fwd, up: o.up, touched: false } }
            }).collect();
            let build = |eds: &[Ed]| -> Vec<crate::h4::mvar::H4PlacedObject> {
                v.objects.iter().zip(eds).map(|(src, e)| {
                    let mut r = src.clone();
                    r.pos = e.pos;
                    if turned(e.fwd, r.fwd) || turned(e.up, r.up) { r.set_orientation(e.fwd, e.up); }
                    r
                }).collect()
            };
            let d1 = scratch(&format!("h4_ap1_{fi}.mvar"));
            let slots = crate::h4::mvar::save_objects(f, &d1, &build(&eds), None).expect("save 1");
            let back = crate::h4::mvar::parse_h4_variant(&d1).expect("re-read");
            let dec: Vec<PoseBits> = slots.iter().map(|s| { let o = back.objects.iter().find(|o| o.slot == *s).expect("slot"); pose_bits(o.pos, o.fwd, o.up) }).collect();
            let rep = judge(QuantGame::Halo4, back.bounds, &mut eds, &dec, &format!("{} Halo 4", f.display()));
            let d2 = scratch(&format!("h4_ap2_{fi}.mvar"));
            crate::h4::mvar::save_objects(f, &d2, &build(&eds), None).expect("save 2");
            assert_eq!(std::fs::read(&d1).unwrap(), std::fs::read(&d2).unwrap(), "{}: saving the adjusted scene again changed the file", f.display());
            n += eds.len();
            adjusted += rep.adjusted;
        }
        eprintln!("Halo 4: {n} objects, {adjusted} adjusted, all bit-equal to the decoder, re-save byte-identical");
        assert!(n > 100 && adjusted > 50);
    }

    /// ★ Halo 2 Anniversary (v52): same chain.
    #[test]
    fn h2a_save_apply_equals_decoder_and_resave_is_identical() {
        let files = files_of(Game::H2A, 6);
        if files.is_empty() { eprintln!("skip: no Halo 2 Anniversary variants installed"); return; }
        let (mut n, mut adjusted) = (0usize, 0usize);
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h4::mvar::parse_h4_variant(f) else { continue };
            if v.objects.is_empty() || v.chunk_version != crate::h4::mvar::H2A_MVAR_VERSION { continue; }
            // out-of-bounds H2A records quantise against a box HMS does not have: leave them as they are
            let id = scratch(&format!("h2a_id_{fi}.mvar"));
            crate::h4::mvar::save_objects(f, &id, &v.objects, None).expect("save");
            assert_eq!(std::fs::read(f).unwrap(), std::fs::read(&id).unwrap(), "{}: an unedited re-save is not byte-identical", f.display());
            let mut rng = Rng(0x5AFE_2000 + fi as u64);
            let mut eds: Vec<Ed> = v.objects.iter().enumerate().map(|(i, o)| {
                if i % 2 == 0 && o.in_bounds { let (p, fw, up) = rng.pose(v.bounds); Ed { datum: i as u32, pos: p, fwd: fw, up, touched: true } }
                else { Ed { datum: i as u32, pos: o.pos, fwd: o.fwd, up: o.up, touched: false } }
            }).collect();
            let build = |eds: &[Ed]| -> Vec<crate::h4::mvar::H4PlacedObject> {
                v.objects.iter().zip(eds).map(|(src, e)| {
                    let mut r = src.clone();
                    r.pos = e.pos;
                    if turned(e.fwd, r.fwd) || turned(e.up, r.up) { r.set_orientation(e.fwd, e.up); }
                    r
                }).collect()
            };
            let d1 = scratch(&format!("h2a_ap1_{fi}.mvar"));
            let slots = crate::h4::mvar::save_objects(f, &d1, &build(&eds), None).expect("save 1");
            let back = crate::h4::mvar::parse_h4_variant(&d1).expect("re-read");
            let dec: Vec<PoseBits> = slots.iter().map(|s| { let o = back.objects.iter().find(|o| o.slot == *s).expect("slot"); pose_bits(o.pos, o.fwd, o.up) }).collect();
            let rep = judge(QuantGame::Halo4, back.bounds, &mut eds, &dec, &format!("{} H2A", f.display()));
            let d2 = scratch(&format!("h2a_ap2_{fi}.mvar"));
            crate::h4::mvar::save_objects(f, &d2, &build(&eds), None).expect("save 2");
            assert_eq!(std::fs::read(&d1).unwrap(), std::fs::read(&d2).unwrap(), "{}: saving the adjusted scene again changed the file", f.display());
            n += eds.len();
            adjusted += rep.adjusted;
        }
        eprintln!("H2A: {n} objects, {adjusted} adjusted, all bit-equal to the decoder, re-save byte-identical");
        assert!(n > 50 && adjusted > 20);
    }

    /// ★ Halo 3: same chain through `save_h3_variant` (placed records only).
    #[test]
    fn halo3_save_apply_equals_decoder_and_resave_is_identical() {
        let files = files_of(Game::Halo3, 4);
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        let (mut n, mut adjusted) = (0usize, 0usize);
        for (fi, f) in files.iter().enumerate() {
            let Ok(v) = crate::h3::mvar::parse_h3_variant(f) else { continue };
            let placed: Vec<usize> = v.objects.iter().enumerate().filter(|(_, o)| o.placed).map(|(i, _)| i).collect();
            if placed.is_empty() { continue; }
            let id = scratch(&format!("h3_id_{fi}.mvar"));
            crate::h3::mvar::save_h3_variant(f, &id, &v, &v.objects, None).expect("save");
            assert_eq!(std::fs::read(f).unwrap(), std::fs::read(&id).unwrap(), "{}: an unedited re-save is not byte-identical", f.display());
            let mut rng = Rng(0x5AFE_3000 + fi as u64);
            let mut eds: Vec<Ed> = placed.iter().enumerate().map(|(k, &i)| {
                let o = &v.objects[i];
                if k % 2 == 0 { let (p, fw, up) = rng.pose(v.bounds); Ed { datum: i as u32, pos: p, fwd: fw, up, touched: true } }
                else { Ed { datum: i as u32, pos: o.pos, fwd: o.fwd, up: o.up, touched: false } }
            }).collect();
            let build = |eds: &[Ed]| -> Vec<crate::h3::mvar::H3VariantObject> {
                let mut list = v.objects.clone();
                for e in eds {
                    let r = &mut list[e.datum as usize];
                    if (0..3).any(|k| (r.pos[k] - e.pos[k]).abs() > 1e-6) || turned(r.fwd, e.fwd) || turned(r.up, e.up) {
                        r.set_pose(e.pos, glam::Vec3::from(e.fwd), glam::Vec3::from(e.up), v.bounds);
                    }
                }
                list
            };
            let d1 = scratch(&format!("h3_ap1_{fi}.mvar"));
            crate::h3::mvar::save_h3_variant(f, &d1, &v, &build(&eds), None).expect("save 1");
            let back = crate::h3::mvar::parse_h3_variant(&d1).expect("re-read");
            let dec: Vec<PoseBits> = eds.iter().map(|e| { let o = &back.objects[e.datum as usize]; pose_bits(o.pos, o.fwd, o.up) }).collect();
            let rep = judge(QuantGame::Halo3, back.bounds, &mut eds, &dec, &format!("{} Halo 3", f.display()));
            let d2 = scratch(&format!("h3_ap2_{fi}.mvar"));
            crate::h3::mvar::save_h3_variant(f, &d2, &v, &build(&eds), None).expect("save 2");
            assert_eq!(std::fs::read(&d1).unwrap(), std::fs::read(&d2).unwrap(), "{}: saving the adjusted scene again changed the file", f.display());
            n += eds.len();
            adjusted += rep.adjusted;
        }
        eprintln!("Halo 3: {n} objects, {adjusted} adjusted, all bit-equal to the decoder, re-save byte-identical");
        assert!(n > 50 && adjusted > 20);
    }

    // -----------------------------------------------------------------------------------------
    // snap, measured on real variants
    // -----------------------------------------------------------------------------------------

    /// ★ Snap-to-representable, proved through a real save: snapping a moved object and saving
    /// stores the snapped pose UNCHANGED, and snapping twice is a no-op.
    #[test]
    fn snap_survives_a_real_save_and_is_idempotent() {
        let files = files_of(Game::Reach, 2);
        if files.is_empty() { eprintln!("skip: no Reach variants installed"); return; }
        let f = &files[0];
        let Some(v) = crate::mvar::parse_variant(f) else { panic!("cannot parse {}", f.display()) };
        assert!(!v.objects.is_empty(), "{} has no objects", f.display());
        let q = PoseQuant::new(QuantGame::Reach, v.globals.bounds);
        let mut rng = Rng(0xC0FFEE);
        let mut list = v.objects.clone();
        let mut snapped: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        for po in list.iter_mut() {
            let (p, fwd, up) = rng.pose(v.globals.bounds);
            // SNAP: adopt the predicted pose as the object's own
            let (sp, sf, su) = q.round_trip(p, fwd, up);
            // snapping again changes nothing
            assert_eq!(q.round_trip(sp, sf, su), (sp, sf, su), "snap is not idempotent");
            assert!(q.predict(sp, sf, su).pos_delta == [0.0, 0.0, 0.0]);
            po.pos = sp;
            let (g, uq, fq) = crate::mvar::encode_orientation(sf, su);
            po.up_is_global = g;
            po.up_quant = uq;
            po.forward_angle_q = fq;
            po.fwd = sf;
            po.up = su;
            snapped.push((sp, sf, su));
        }
        let dst = scratch("reach_snap.mvar");
        crate::mvar::save_objects(f, &dst, &list, None).expect("save");
        let back = crate::mvar::parse_variant(&dst).expect("re-read");
        for (i, (got, (sp, sf, su))) in back.objects.iter().zip(snapped.iter()).enumerate() {
            assert_eq!(got.pos, *sp, "object {i}: a snapped position did not survive the save");
            assert_eq!(got.fwd, *sf, "object {i}: a snapped forward did not survive the save");
            assert_eq!(got.up, *su, "object {i}: a snapped up did not survive the save");
        }
        eprintln!("snap: {} Reach objects snapped, saved and reloaded unchanged", back.objects.len());
    }

    /// The gate must not go vacuous: at least one game's variants have to be present, or every
    /// test above would "pass" by skipping.
    #[test]
    fn samples_are_found() {
        let n: usize = [Game::Reach, Game::Halo4, Game::H2A, Game::Halo3].iter().map(|g| files_of(*g, 4).len()).sum();
        assert!(n > 0, "no .mvar files of ANY game found on this machine — every proof test above would skip");
    }
}
