//! Gates over the whole pipeline. Expectations come from the decoded constants (cited in each
//! test) or from independent physics (Coulomb friction, energy), never from the production
//! formulas themselves; `~/hk/sab.py` runs a saboteur per gate (list in `02-contact-solver.md`).

use super::*;
use crate::reach_havok::{HavokBody, MotionType};
use glam::{Mat3, Quat, Vec3};

pub const WORLD_G: f32 = 4.1712594;

pub fn info() -> SolverInfo {
    SolverInfo::engine(60.0, Vec3::new(0.0, 0.0, -WORLD_G))
}

pub fn box_body(half: Vec3, mass: f32, pos: Vec3, friction: f32, restitution: f32) -> SimBody {
    let i = Vec3::new(
        mass / 12.0 * (half.y * half.y + half.z * half.z) * 4.0,
        mass / 12.0 * (half.x * half.x + half.z * half.z) * 4.0,
        mass / 12.0 * (half.x * half.x + half.y * half.y) * 4.0,
    );
    let hav = HavokBody::new(MotionType::BoxInertia, mass, Mat3::from_diagonal(Vec3::new(1.0 / i.x, 1.0 / i.y, 1.0 / i.z)), pos, Quat::IDENTITY);
    SimBody::new(hav, vec![LocalShape::cuboid(half, 0.0164)], friction, restitution)
}

pub fn sphere_body(r: f32, mass: f32, pos: Vec3, friction: f32, restitution: f32) -> SimBody {
    let i = 0.4 * mass * r * r;
    let hav = HavokBody::new(MotionType::SphereInertia, mass, Mat3::from_diagonal(Vec3::splat(1.0 / i)), pos, Quat::IDENTITY);
    SimBody::new(hav, vec![LocalShape::Sphere { center: Vec3::ZERO, radius: r }], friction, restitution)
}

/// a 1 x 1 x 1 wu box (core 0.4836 + radius 0.0164, the Forge block)
pub fn block(pos: Vec3, mass: f32, mu: f32, e: f32) -> SimBody {
    box_body(Vec3::splat(0.5 - 0.0164), mass, pos, mu, e)
}

/// A big flat ground at z = 0 made of two polygons with the engine's radius.
pub fn ground(friction: f32, restitution: f32) -> PolyList {
    let s = 300.0;
    let (a, b, c, d) = (Vec3::new(-s, -s, 0.0), Vec3::new(s, -s, 0.0), Vec3::new(s, s, 0.0), Vec3::new(-s, s, 0.0));
    PolyList::new(vec![StaticPoly::triangle(a, b, c, ENV_RADIUS), StaticPoly::triangle(a, c, d, ENV_RADIUS)], (friction, restitution))
}

/// run until `ticks`, returning the z history of body `i`
fn run(sim: &mut Sim, g: &PolyList, ticks: usize, i: usize) -> Vec<f32> {
    (0..ticks)
        .map(|_| {
            sim.tick(g);
            sim.bodies[i].hav.com.z
        })
        .collect()
}

/// A body with nothing around it falls asleep after exactly the decoded number of quiet windows:
/// counter 4/8/12 are 4-tick windows, 16 is the long window (counter resets), so the short
/// window count reaches 6 on the 36th integrate (anchor at 4 counts 0; 8,12 -> 2; 20,24,28 -> 5;
/// 36 -> 6 > 5) and the island goes to sleep at the start of the 37th (`04` 2.4 / 5.2).
#[test]
fn deactivation_after_six_quiet_windows() {
    let mut sim = Sim::new(SolverInfo::engine(60.0, Vec3::ZERO));
    let i = sim.add(block(Vec3::new(0.0, 0.0, 5.0), 10.0, 0.9, 0.0));
    let g = PolyList::engine(vec![]);
    let mut slept = None;
    for t in 1..=100 {
        sim.tick(&g);
        if !sim.bodies[i].awake && slept.is_none() {
            slept = Some(t);
        }
    }
    assert_eq!(slept, Some(37), "the 6th quiet window is the 36th integrate; the request runs at the start of the next one");
}

/// A body drifting 0.04 wu/s stays inside the 0.025 wu window radius for the 32 ticks between the
/// anchor and the sixth sample (0.021 wu) and sleeps; one moving 0.5 wu/s leaves it and never does.
#[test]
fn deactivation_needs_less_than_the_reference_distance() {
    for (v, sleeps) in [(0.04f32, true), (0.5, false)] {
        let mut sim = Sim::new(SolverInfo::engine(60.0, Vec3::ZERO));
        let i = sim.add(block(Vec3::new(0.0, 0.0, 5.0), 10.0, 0.9, 0.0));
        sim.bodies[i].hav.v = Vec3::new(v, 0.0, 0.0);
        let g = PolyList::engine(vec![]);
        for _ in 0..120 {
            sim.tick(&g);
        }
        assert_eq!(!sim.bodies[i].awake, sleeps, "drift {v} wu/s");
    }
}

/// The solver-info rows the world ctor derives, by hand: |g| = 4.1712594, class 3 (MEDIUM, what
/// every non-biped body gets): thr 0.017 -> 0.0709114; slow factor 0.2.
#[test]
fn deactivation_class_rows() {
    let c = info().classes[3];
    assert!((c.lin_inv - 14.1024).abs() < 1e-3, "{}", c.lin_inv);
    assert!((c.slow_mult - 0.976_470_6).abs() < 1e-5, "{}", c.slow_mult);
    assert!((c.zero_below - 0.01).abs() < 1e-6, "{}", c.zero_below);
    assert!((c.ref_sq - 6.25e-4).abs() < 1e-8 && (c.ref4_sq - 0.01).abs() < 1e-7, "0.025^2 and (4 * 0.025)^2: {} {}", c.ref_sq, c.ref4_sq);
    let off = info().classes[1];
    assert!(off.zero_below > 1.0e30, "class OFF has no slow factor");
}

/// A dropped block rests on the ground at core 0.4836 + its radius 0.0164 + the environment
/// polygon's radius 0.0164 = 0.5164 (the engine's props float 1.6 cm over the BSP), a few mm
/// into it at most (contacts are created by penetration), and sleeps.
#[test]
fn block_lands_rests_sleeps() {
    let mut sim = Sim::new(info());
    let i = sim.add(block(Vec3::new(0.0, 0.0, 1.0), 10.0, 0.9, 0.2));
    let g = ground(0.9, 0.2);
    let zs = run(&mut sim, &g, 300, i);
    let z = *zs.last().unwrap();
    assert!((0.5120..=0.5170).contains(&z), "rest height {z}");
    assert!(!sim.bodies[i].awake, "asleep");
}

/// Coulomb: a block sliding at 3 wu/s stops after v^2 / (2 mu g) with mu the QUANTISED
/// sqrt(fA fB). The engine's friction cap uses the previous tick's impulse, so allow +10 %.
#[test]
fn sliding_stops_at_the_coulomb_distance() {
    // the last case is a flat slab: a tall block with mu > 1 would tip over instead of sliding
    for (fa, fb, slab) in [(0.9f32, 0.9f32, false), (0.8, 0.5, false), (0.2, 0.2, false), (1.0, 1.0, true)] {
        let mut sim = Sim::new(info());
        let body = if slab { box_body(Vec3::new(0.5, 0.5, 0.1), 10.0, Vec3::new(0.0, 0.0, 0.1335), fa, 0.0) } else { block(Vec3::new(0.0, 0.0, 0.52), 10.0, fa, 0.0) };
        let i = sim.add(body);
        sim.bodies[i].hav.v = Vec3::new(3.0, 0.0, 0.0);
        let g = ground(fb, 0.0);
        let mut stop = None;
        for _ in 0..1200 {
            sim.tick(&g);
            if stop.is_none() && sim.bodies[i].hav.v.x < 0.01 {
                stop = Some(sim.bodies[i].hav.com.x);
            }
        }
        let mu = ContactMaterial::mix(fa, 0.0, fb, 0.0).friction();
        // independent expectation: the table entry at or above sqrt(fa*fb)
        let table_mu = (fa * fb).sqrt();
        assert!(mu >= table_mu && mu < table_mu * 1.08 + 0.011, "quantised up: {mu} vs {table_mu}");
        let want = 9.0 / (2.0 * mu * WORLD_G);
        let got = stop.expect("it stops");
        assert!(got > want * 0.97 && got < want * 1.12, "mu {fa}*{fb}: stopped at {got}, Coulomb {want}");
    }
}

/// Restitution has two regimes in the engine. A FAST impact that the TOI search catches (the pair
/// is predicted to end the step more than `0.4 d` deep, `d` = 0.2 for this ball) is resolved at
/// the event: the normal velocity becomes `-(1 + e) vn + 0.1` (Newtonian, `03` sec 6), so the ball
/// bounces to `(e vn + 0.1)^2 / 2g`. A SLOW impact goes through the PSI solver, where restitution
/// is only a `1.3 e |vn| dt/8` displacement seed: millimetres.
#[test]
fn restitution_toi_is_newtonian_psi_is_weak() {
    // a 20 wu/s ball (0.33 wu per tick): the phase decides whether the TOI or the PSI contact
    // catches it, so scan the phase and require that the TOI fires for some of them
    let run_one = |e: f32, phase: f32| -> (f32, usize, f32) {
        let mut sim = Sim::new(info());
        let i = sim.add(sphere_body(0.5, 5.0, Vec3::new(0.0, 0.0, 0.5164 + 3.0 + phase), 0.9, e));
        sim.bodies[i].hav.v = Vec3::new(0.0, 0.0, -20.0);
        let g = ground(0.9, e);
        let zs = run(&mut sim, &g, 120, i);
        let land = zs.iter().position(|z| *z < 0.5164 + 0.4).unwrap_or(0);
        let peak = zs[land + 1..].iter().cloned().fold(0.0f32, f32::max) - 0.5164;
        (peak, sim.toi_count, 20.0)
    };
    let mut fired = 0;
    for k in 0..8 {
        let phase = k as f32 * 0.04;
        let (peak0, _, _) = run_one(0.0, phase);
        assert!(peak0 < 0.05, "e = 0 never bounces: phase {phase} -> {peak0}");
        let (peak, tois, vn) = run_one(0.6, phase);
        if tois > 0 {
            fired += 1;
            let want = (0.6 * vn + 0.1).powi(2) / (2.0 * WORLD_G);
            assert!(peak > want * 0.6 && peak < want * 1.2, "phase {phase}: bounced {peak}, Newtonian {want}");
        }
    }
    assert!(fired >= 2, "the TOI fired for {fired} of 8 phases");
    // slow: a 0.25 wu drop (2.0 wu/s) never meets a TOI and stays in the PSI solver
    let mut sim = Sim::new(info());
    let i = sim.add(sphere_body(0.5, 5.0, Vec3::new(0.0, 0.0, 0.5164 + 0.25), 0.9, 0.9));
    let g = ground(0.9, 0.9);
    let zs = run(&mut sim, &g, 120, i);
    let land = zs.iter().position(|z| *z < 0.5164 + 0.05).unwrap();
    let slow = zs[land + 1..].iter().cloned().fold(0.0f32, f32::max) - 0.5164;
    assert!(sim.toi_count == 0 && slow < 0.02, "slow impact stays in the PSI solver: {slow}");
}

/// Five blocks in a stack (0.01 gaps) settle within 1 cm of nominal, barely drift, and sleep.
#[test]
fn stack_of_five_is_stable_and_asleep() {
    let mut sim = Sim::new(info());
    for k in 0..5 {
        sim.add(block(Vec3::new(0.0, 0.0, 0.52 + k as f32 * 1.01), 10.0, 0.9, 0.2));
    }
    let g = ground(0.9, 0.2);
    for _ in 0..400 {
        sim.tick(&g);
    }
    for (k, b) in sim.bodies.iter().enumerate() {
        assert!((b.hav.com.z - (0.5164 + k as f32)).abs() < 0.01, "block {k} at {}", b.hav.com.z);
        assert!(b.hav.com.x.abs() + b.hav.com.y.abs() < 0.02, "block {k} drifted {:?}", b.hav.com);
        assert!(!b.awake, "block {k} asleep");
    }
}

/// The same stack with 24 iterations is as stable as with the engine's 8; with ONE iteration it
/// is not (the row relaxation alone cannot carry the weight).
#[test]
fn solver_iterations_eight_vs_twenty_four_vs_one() {
    let height = |iters: u32| -> f32 {
        let mut i = SolverInfo::build(1.0 / 60.0, Vec3::new(0.0, 0.0, -WORLD_G), iters, 1, 0.3, 0.9, 0.05);
        i.contact_resting_velocity = f32::MAX;
        let mut sim = Sim::new(i);
        for k in 0..5 {
            sim.add(block(Vec3::new(0.0, 0.0, 0.52 + k as f32 * 1.01), 10.0, 0.9, 0.2));
        }
        let g = ground(0.9, 0.2);
        for _ in 0..300 {
            sim.tick(&g);
        }
        sim.bodies[4].hav.com.z
    };
    let (h8, h24, h1) = (height(8), height(24), height(1));
    assert!((h8 - 4.5164).abs() < 0.01 && (h24 - 4.5164).abs() < 0.01, "8: {h8} 24: {h24}");
    assert!((h8 - h24).abs() < 0.005, "8 and 24 agree: {h8} {h24}");
    assert!((h1 - 4.5164).abs() > (h8 - 4.5164).abs(), "one iteration is worse: {h1} vs {h8}");
}

/// Total mechanical energy never grows in a pile of 14 random boxes (rest 0.2), nothing is NaN,
/// everything ends asleep above the ground.
#[test]
fn pile_energy_never_grows() {
    let mut seed = 12345u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    };
    let mut sim = Sim::new(info());
    for k in 0..14 {
        let pos = Vec3::new(lcg() * 2.0 - 1.0, lcg() * 2.0 - 1.0, 1.0 + k as f32 * 0.9);
        let half = Vec3::new(0.25 + 0.3 * lcg(), 0.25 + 0.3 * lcg(), 0.25 + 0.3 * lcg());
        let mut b = box_body(half, 5.0 + 20.0 * lcg(), pos, 0.9, 0.2);
        b.hav.q = Quat::from_euler(glam::EulerRot::XYZ, lcg() * 3.0, lcg() * 3.0, lcg() * 3.0);
        sim.add(b);
    }
    let energy = |s: &Sim| -> f32 { s.bodies.iter().map(|b| (1.0 / b.hav.inv_mass) * (WORLD_G * b.hav.com.z + 0.5 * b.hav.v.length_squared())).sum() };
    let e0 = energy(&sim);
    let g = ground(0.9, 0.2);
    let mut emax = f32::MIN;
    for _ in 0..1500 {
        sim.tick(&g);
        emax = emax.max(energy(&sim));
        assert!(sim.bodies.iter().all(|b| b.hav.com.is_finite()));
    }
    assert!(emax <= e0 * 1.001, "energy grew: {e0} -> {emax}");
    assert!(sim.bodies.iter().all(|b| !b.awake && b.hav.com.z > 0.25), "settled above the ground");
}

/// 1000:1 heavy-on-light, with the 16:1 clamp: the light block under a 2000-mass block is not
/// crushed through (without it the heavy block ends inside the light one).
#[test]
fn heavy_on_light_obeys_the_clamp() {
    let held = |clamp: bool| -> (f32, f32) {
        let mut sim = Sim::new(info());
        sim.mass_clamp = clamp;
        let l = sim.add(block(Vec3::new(0.0, 0.0, 0.5), 2.0, 0.9, 0.0));
        let h = sim.add(block(Vec3::new(0.0, 0.0, 1.51), 2000.0, 0.9, 0.0));
        let g = ground(0.9, 0.0);
        for _ in 0..300 {
            sim.tick(&g);
        }
        (sim.bodies[l].hav.com.z, sim.bodies[h].hav.com.z)
    };
    let (lz, hz) = held(true);
    assert!(lz > 0.50 && hz > 1.50, "clamped: light {lz} heavy {hz}");
    let (lz2, hz2) = held(false);
    assert!(hz2 < 1.2 || lz2 < 0.45, "unclamped the 1000:1 pair collapses: light {lz2} heavy {hz2}");
}

/// A block balanced on a corner-edge of a 45-degree block does not fall through (box edges on
/// box edges are contacts), over 3 s.
#[test]
fn edge_on_edge_holds() {
    let mut sim = Sim::new(info());
    let mut under = block(Vec3::ZERO, 10.0, 0.9, 0.2);
    under.hav.q = Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_4);
    under.hav.inv_mass = 0.0;
    under.hav.inv_inertia_local = Mat3::ZERO;
    under.hav.motion = MotionType::Fixed;
    sim.add(under);
    let edge = 0.5 * std::f32::consts::SQRT_2;
    let mut top = block(Vec3::new(0.0, 0.0, 2.0 * edge + 0.02), 10.0, 0.9, 0.2);
    top.hav.q = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
    let ti = sim.add(top);
    let g = PolyList::engine(vec![]);
    let mut min_z = f32::MAX;
    for _ in 0..180 {
        sim.tick(&g);
        min_z = min_z.min(sim.bodies[ti].hav.com.z);
    }
    assert!(min_z > 2.0 * edge - 0.05, "the top block stayed on the edge, lowest {min_z}");
}


/// Contacts push but never pull: a block shot upward off the ground rises like a projectile
/// (v^2 / 2g above its start, 0.479 wu for 2 wu/s) instead of being held back by the contacts
/// that were generated while it left (they exist out to the 0.0328 tolerance).
#[test]
fn launched_block_is_not_pulled_back() {
    // a block (four points: coupled pair rows) and a sphere (one point: a single row)
    for body in [block(Vec3::new(0.0, 0.0, 0.5), 10.0, 0.9, 0.2), sphere_body(0.5, 5.0, Vec3::new(0.0, 0.0, 0.5), 0.9, 0.2)] {
        let mut sim = Sim::new(info());
        let i = sim.add(body);
        sim.bodies[i].hav.v = Vec3::new(0.0, 0.0, 2.0);
        let g = ground(0.9, 0.2);
        let zs = run(&mut sim, &g, 120, i);
        let peak = zs.iter().cloned().fold(0.0f32, f32::max) - 0.5164;
        let want = 4.0 / (2.0 * WORLD_G);
        assert!(peak > want * 0.93 && peak < want * 1.05, "peak {peak} want {want}");
    }
}





/// A thin vertical wall (one polygon at x = 5, 10 x 10 wu, the engine's radius) and a ball /
/// block fired at it at 60 wu/s (1 wu per tick): the time-of-impact event stops it on the near side.
/// Without the continuous search (`Sim::continuous = false`, or DEBRIS quality, which the engine
/// never gives a TOI) the same shot goes through.
fn wall() -> PolyList {
    let (a, b, c, d) = (Vec3::new(5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, -5.0), Vec3::new(5.0, 5.0, 5.0), Vec3::new(5.0, -5.0, 5.0));
    PolyList::new(vec![StaticPoly { verts: vec![a, b, c, d], radius: ENV_RADIUS }], (0.5, 0.0))
}

fn shoot(mut body: SimBody, continuous: bool, quality: crate::reach_havok::QualityType) -> (f32, usize) {
    body.hav.v = Vec3::new(60.0, 0.0, 0.0);
    body.quality = quality;
    let mut sim = Sim::new(SolverInfo::engine(60.0, Vec3::ZERO));
    sim.continuous = continuous;
    let i = sim.add(body);
    let w = wall();
    let mut max_x = f32::MIN;
    for _ in 0..90 {
        sim.tick(&w);
        max_x = max_x.max(sim.bodies[i].hav.com.x);
    }
    (max_x, sim.toi_count)
}

#[test]
fn fast_bodies_do_not_tunnel_a_thin_wall() {
    use crate::reach_havok::QualityType;
    let ball = || sphere_body(0.1, 1.0, Vec3::new(0.3, 0.0, 0.0), 0.5, 0.0);
    let cube = || box_body(Vec3::splat(0.2 - 0.0164), 2.0, Vec3::new(0.3, 0.0, 0.0), 0.5, 0.0);
    for (name, mk) in [("ball", &ball as &dyn Fn() -> SimBody), ("cube", &cube)] {
        let (x, tois) = shoot(mk(), true, QualityType::Critical);
        assert!(x < 5.0, "{name}: with the TOI it stays on the near side, got x {x}");
        assert!(tois >= 1, "{name}: the TOI search fired");
        let (x2, _) = shoot(mk(), false, QualityType::Critical);
        assert!(x2 > 5.5, "{name}: without it the same shot tunnels (liveness of the gate), got x {x2}");
        let (x3, tois3) = shoot(mk(), true, QualityType::Debris);
        assert!(x3 > 5.5 && tois3 == 0, "{name}: DEBRIS quality never gets a TOI, got x {x3}");
    }
}

/// Contact points persist across ticks (the engine's only "warm start": the points live while
/// they stay inside the tolerance and keep `rec0`, `rec4`, `mem`) and a resting block stays quiet;
/// when the manifold is thrown away each tick the new points are admitted only by penetration and
/// the block chatters.
#[test]
fn contact_points_persist_across_ticks() {
    use super::poly::collide;
    let info = info();
    let ground_poly = StaticPoly::triangle(Vec3::new(-50.0, -50.0, 0.0), Vec3::new(50.0, -50.0, 0.0), Vec3::new(50.0, 50.0, 0.0), ENV_RADIUS);
    let run_it = |persist: bool| -> f32 {
        // well inside the one triangle (the origin lies on its hypotenuse)
        let mut b = block(Vec3::new(10.0, -10.0, 0.5164), 10.0, 0.9, 0.0);
        let mut mf = Manifold::new(0, None, ContactMaterial::mix(0.9, 0.0, 0.9, 0.0));
        let adm = Admission::engine(0.0328, WORLD_G, info.dt);
        let mut sq = 0.0f32;
        let mut hav = vec![b.hav.clone()];
        let mut aux = vec![BodyAux::new(3)];
        // start slightly sunk so there are contact points from the first tick
        hav[0].com.z = 0.5164 - 0.002;
        for t in 0..240 {
            let sp = b.shapes[0].world(hav[0].com, hav[0].q);
            let mut raw = Vec::new();
            collide(&{ let WShape::Poly(p) = sp else { unreachable!() }; p }, &{ let WShape::Poly(p) = ground_poly.shape_for_test() else { unreachable!() }; p }, 0.0328, 8, &mut raw);
            let cands: Vec<Candidate> = raw.iter().map(|c| Candidate { point: ContactPoint::new(c.id, c.pos, c.normal, c.dist), edge: c.edge, approach: 0.0 }).collect();
            if !persist {
                mf = Manifold::new(0, None, mf.material);
            }
            mf.update(&cands, &adm);
            let mut ms = vec![mf.clone()];
            step(&info, &mut hav, &mut aux, &[true], &mut ms, &[]);
            mf = ms.remove(0);
            if t >= 60 {
                sq += hav[0].v.z * hav[0].v.z;
            }
        }
        b.hav = hav[0].clone();
        (sq / 180.0).sqrt()
    };
    let with = run_it(true);
    let without = run_it(false);
    assert!(with < 1.0e-3, "persistent contact memory: rms vz {with}");
    assert!(without > with * 10.0, "re-creating the points every tick is far noisier: {without} vs {with}");
}


/// Sliding over a seam between coplanar polygons does not disturb the block: the stop distance
/// matches the one on a single polygon. Contacts on a polygon's EDGE or VERTEX (where the block's
/// radius touches the seam) are the ghost-edge bump source; the engine drops them when a deeper
/// face contact covers them (`sub_180D45890`).
#[test]
fn seam_between_coplanar_polygons_is_invisible() {
    let quad = |x0: f32, x1: f32| StaticPoly { verts: vec![Vec3::new(x0, -20.0, 0.0), Vec3::new(x1, -20.0, 0.0), Vec3::new(x1, 20.0, 0.0), Vec3::new(x0, 20.0, 0.0)], radius: ENV_RADIUS };
    let slide = |world: &PolyList| -> (f32, f32) {
        let mut sim = Sim::new(info());
        // a flat slab: a tall block would slowly tip over its leading edge while sliding
        let i = sim.add(box_body(Vec3::new(0.5, 0.5, 0.1), 10.0, Vec3::new(1.0, 0.0, 0.1335), 0.9, 0.0));
        sim.bodies[i].hav.v = Vec3::new(8.0, 0.0, 0.0);
        let mut max_vz = 0.0f32;
        let mut max_w = 0.0f32;
        for _ in 0..300 {
            sim.tick(world);
            max_vz = max_vz.max(sim.bodies[i].hav.v.z.abs());
            max_w = max_w.max(sim.bodies[i].hav.w.length());
        }
        (sim.bodies[i].hav.com.x, max_vz.max(max_w))
    };
    let m = (0.9, 0.0);
    let one = PolyList::new(vec![quad(-20.0, 40.0)], m);
    let two = PolyList::new(vec![quad(-20.0, 5.0), quad(5.0, 40.0)], m);
    let three = PolyList::new(vec![quad(-20.0, 3.0), quad(3.0, 5.0), quad(5.0, 40.0)], m);
    let (x1, d1) = slide(&one);
    let (x2, d2) = slide(&two);
    let (x3, d3) = slide(&three);
    assert!((x2 - x1).abs() < 0.03 * x1 && (x3 - x1).abs() < 0.03 * x1, "stop x: one {x1}, two {x2}, three {x3}");
    assert!(d2 < d1 + 0.05 && d3 < d1 + 0.05, "no kick: {d1} {d2} {d3}");
}



/// A capsule lying on the ground comes to rest on its side (the line contact of the three
/// end/middle spheres), at radius + the polygon's radius, and sleeps; a sphere dropped on a box
/// rests on it. (Smoke gates for the GJK paths; the numbers are geometry, not decode.)
#[test]
fn capsule_and_sphere_rest() {
    let mut sim = Sim::new(info());
    let len = 0.5f32;
    let r = 0.2f32;
    let mass = 4.0;
    let i_axis = 0.5 * mass * r * r;
    let i_perp = mass * (3.0 * r * r + 4.0 * len * len) / 12.0;
    let hav = HavokBody::new(MotionType::BoxInertia, mass, Mat3::from_diagonal(Vec3::new(1.0 / i_axis, 1.0 / i_perp, 1.0 / i_perp)), Vec3::new(0.0, 0.0, 1.0), Quat::from_rotation_y(0.2));
    let cap = sim.add(SimBody::new(hav, vec![LocalShape::Capsule { a: Vec3::new(-len, 0.0, 0.0), b: Vec3::new(len, 0.0, 0.0), radius: r }], 0.9, 0.0));
    let slab = sim.add(box_body(Vec3::new(1.0, 1.0, 0.1), 20.0, Vec3::new(5.0, 0.0, 0.2), 0.9, 0.0));
    let ball = sim.add(sphere_body(0.25, 2.0, Vec3::new(5.0, 0.0, 1.2), 0.9, 0.0));
    let g = ground(0.9, 0.0);
    for _ in 0..400 {
        sim.tick(&g);
    }
    let c = &sim.bodies[cap].hav;
    assert!((c.com.z - (r + ENV_RADIUS)).abs() < 0.006, "capsule rests on its side: z {}", c.com.z);
    assert!(c.q.mul_vec3(Vec3::X).z.abs() < 0.05, "axis is horizontal");
    let s = &sim.bodies[slab].hav;
    let b = &sim.bodies[ball].hav;
    assert!((b.com.z - (s.com.z + 0.1 + 0.0164 + 0.25)).abs() < 0.01, "ball on the slab: ball {} slab {}", b.com.z, s.com.z);
    assert!(sim.bodies.iter().all(|x| !x.awake), "everything asleep");
}

/// Robustness: three seeds of a pile of 20 mixed boxes / balls / capsules dropped from random poses
/// neither explode nor fall through the ground, and every body ends asleep above it.
#[test]
fn mixed_piles_settle() {
    for seed0 in [1u64, 7, 99] {
        let mut seed = seed0;
        let mut lcg = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
        };
        let mut sim = Sim::new(info());
        for k in 0..20 {
            let pos = Vec3::new(lcg() * 3.0 - 1.5, lcg() * 3.0 - 1.5, 1.0 + k as f32 * 0.7);
            let q = Quat::from_euler(glam::EulerRot::XYZ, lcg() * 3.0, lcg() * 3.0, lcg() * 3.0);
            let mut b = match k % 3 {
                0 => box_body(Vec3::new(0.2 + 0.3 * lcg(), 0.2 + 0.3 * lcg(), 0.2 + 0.3 * lcg()), 4.0 + 20.0 * lcg(), pos, 0.9, 0.2),
                1 => sphere_body(0.2 + 0.2 * lcg(), 3.0 + 5.0 * lcg(), pos, 0.9, 0.2),
                _ => {
                    let (len, r, mass) = (0.3 + 0.2 * lcg(), 0.15 + 0.1 * lcg(), 3.0);
                    let ia = 0.5 * mass * r * r;
                    let ip = mass * (3.0 * r * r + 4.0 * len * len) / 12.0;
                    SimBody::new(HavokBody::new(MotionType::BoxInertia, mass, Mat3::from_diagonal(Vec3::new(1.0 / ia, 1.0 / ip, 1.0 / ip)), pos, Quat::IDENTITY), vec![LocalShape::Capsule { a: Vec3::new(-len, 0.0, 0.0), b: Vec3::new(len, 0.0, 0.0), radius: r }], 0.9, 0.2)
                }
            };
            b.hav.q = q;
            // the phmo rigid body's damping fields (a rolling ball otherwise rolls to the horizon)
            b.hav = b.hav.clone().with_damping(0.1, 0.3);
            sim.add(b);
        }
        let g = ground(0.9, 0.2);
        for t in 0..2500 {
            sim.tick(&g);
            for (k, b) in sim.bodies.iter().enumerate() {
                assert!(b.hav.com.is_finite() && b.hav.com.z > -0.2 && b.hav.v.length() < 80.0, "seed {seed0} tick {t} body {k} ({:?}): pos {:?} v {:?} w {:?} awake {}", b.shapes[0], b.hav.com, b.hav.v, b.hav.w, b.awake);
            }
        }
        let awake = sim.bodies.iter().filter(|b| b.awake).count();
        assert_eq!(awake, 0, "seed {seed0}: {awake} bodies still awake");
        assert!(sim.bodies.iter().all(|b| b.hav.com.z > 0.1), "seed {seed0}: nothing sinks into the ground");
    }
}


/// The decoded cinfo value (`sub_1807F66B0`: 0.0328, not scaled by the tick rate: the flag
/// `byte_1852D3C8C` has no writer).
#[test]
fn collision_tolerance_pin() {
    assert_eq!(COLLISION_TOLERANCE, 0.0328);
}

/// A keyframed slab (a lift / a sweeping door) moving at 1 wu/s carries a block standing on it and
/// pushes one beside it; neither is left behind or launched.
#[test]
fn keyframed_bodies_carry_and_push() {
    let mut sim = Sim::new(info());
    let mut slab = box_body(Vec3::new(2.0, 1.0, 0.1), 100.0, Vec3::new(0.0, 0.0, 0.1328), 0.9, 0.0);
    slab.hav.motion = MotionType::Keyframed;
    slab.hav.inv_mass = 0.0;
    slab.hav.inv_inertia_local = Mat3::ZERO;
    slab.quality = crate::reach_havok::QualityType::Keyframed;
    let s = sim.add(slab);
    let passenger = sim.add(block(Vec3::new(0.0, 0.0, 0.1328 + 0.1 + 0.0164 + 0.5), 10.0, 0.9, 0.0));
    let g = PolyList::engine(vec![]);
    for _ in 0..180 {
        sim.bodies[s].hav.v = Vec3::new(1.0, 0.0, 0.0);
        sim.tick(&g);
    }
    let slab_x = sim.bodies[s].hav.com.x;
    let p = &sim.bodies[passenger].hav;
    assert!((slab_x - 3.0).abs() < 0.05, "slab moved {slab_x}");
    assert!((p.com.x - slab_x).abs() < 0.3, "the passenger rides along: {} vs {}", p.com.x, slab_x);
    assert!(p.com.z > 0.7 && p.com.z < 0.8, "and stays on it: {}", p.com.z);
}

/// Fixed objects are bodies too (a Forge block): a block dropped on a fixed block rests on it, and a
/// convex hull (a pyramid, `c_convex_vertices_shape`) dropped on the ground settles on a face.
#[test]
fn rests_on_fixed_blocks_and_hulls_settle() {
    let mut sim = Sim::new(info());
    let mut fixed = block(Vec3::new(0.0, 0.0, 0.5), 10.0, 1.0, 0.0);
    fixed.hav.motion = MotionType::Fixed;
    fixed.hav.inv_mass = 0.0;
    fixed.hav.inv_inertia_local = Mat3::ZERO;
    fixed.quality = crate::reach_havok::QualityType::Fixed;
    sim.add(fixed);
    let top = sim.add(block(Vec3::new(0.2, 0.1, 2.0), 10.0, 1.0, 0.0));
    // a square pyramid, apex up, base 1 x 1, height 1
    let verts = vec![Vec3::new(-0.5, -0.5, -0.3), Vec3::new(0.5, -0.5, -0.3), Vec3::new(0.5, 0.5, -0.3), Vec3::new(-0.5, 0.5, -0.3), Vec3::new(0.0, 0.0, 0.7)];
    let loops = vec![vec![0, 3, 2, 1], vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]];
    let mass = 5.0;
    let hav = HavokBody::new(MotionType::BoxInertia, mass, Mat3::from_diagonal(Vec3::splat(1.0 / 0.5)), Vec3::new(8.0, 0.0, 1.5), Quat::from_rotation_x(2.4));
    let pyr = sim.add(SimBody::new(hav, vec![LocalShape::Hull { verts, loops, radius: ENV_RADIUS }], 0.9, 0.0));
    let g = ground(0.9, 0.0);
    for _ in 0..500 {
        sim.tick(&g);
    }
    let t = &sim.bodies[top].hav;
    assert!((t.com.z - 1.5).abs() < 0.02, "the block rests on the fixed one: z {}", t.com.z);
    assert!(!sim.bodies[top].awake);
    let p = &sim.bodies[pyr].hav;
    // resting on a face: either the base (centre 0.3 + radii above the ground) or a slanted side
    assert!(p.com.z > 0.2 && p.com.z < 0.4, "the pyramid lies on the ground: z {}", p.com.z);
    assert!(!sim.bodies[pyr].awake, "and sleeps");
}

/// A ball rolling over a seam of two coplanar polygons keeps its contact, losing it for at most one
/// tick (its centre leaves the first polygon's face, the second's is admitted a tick later): it
/// sinks by less than 4 mm and its vertical speed stays under one tick of gravity (0.0695 wu/s).
/// (The potential-contact filter with the ball's own radius is what keeps the polygon-edge contact of
/// the seam from kicking it.)
#[test]
fn ball_rolls_over_a_seam_keeping_contact() {
    let quad = |x0: f32, x1: f32| StaticPoly { verts: vec![Vec3::new(x0, -20.0, 0.0), Vec3::new(x1, -20.0, 0.0), Vec3::new(x1, 20.0, 0.0), Vec3::new(x0, 20.0, 0.0)], radius: ENV_RADIUS };
    let world = PolyList::new(vec![quad(-20.0, 5.0), quad(5.0, 40.0)], (0.9, 0.0));
    let mut sim = Sim::new(info());
    let i = sim.add(sphere_body(0.5, 10.0, Vec3::new(1.0, 0.0, 0.5164), 0.9, 0.0));
    sim.bodies[i].hav.v = Vec3::new(8.0, 0.0, 0.0);
    let (mut min_z, mut max_vz) = (f32::MAX, 0.0f32);
    for _ in 0..60 {
        sim.tick(&world);
        let b = &sim.bodies[i].hav;
        if b.com.x > 3.0 {
            min_z = min_z.min(b.com.z);
            max_vz = max_vz.max(b.v.z.abs());
        }
    }
    assert!(min_z > 0.5164 - 0.004, "the ball sank to {min_z} crossing the seam");
    assert!(max_vz < 0.1, "and hopped {max_vz}");
}


/// A fast-tumbling capsule (12 rad/s) hitting the ground: the TOI event's closing speed comes from the
/// secant slope of the gap, not from the conservative rotation bound, so the impact does not launch
/// it (with the bound as closing speed the same state left at 8 wu/s).
#[test]
fn tumbling_capsule_impact_is_not_a_launch() {
    let (len, r, mass) = (0.4357481f32, 0.15425743f32, 3.0f32);
    let ia = 0.5 * mass * r * r;
    let ip = mass * (3.0 * r * r + 4.0 * len * len) / 12.0;
    let hav = HavokBody::new(MotionType::BoxInertia, mass, Mat3::from_diagonal(Vec3::new(1.0 / ia, 1.0 / ip, 1.0 / ip)), Vec3::new(-2.7916057, -0.6666639, 0.47204897), Quat::from_xyzw(-0.5336929, 0.31714883, 0.19505233, 0.75930446));
    let mut sim = Sim::new(info());
    let i = sim.add(SimBody::new(hav, vec![LocalShape::Capsule { a: Vec3::new(-len, 0.0, 0.0), b: Vec3::new(len, 0.0, 0.0), radius: r }], 0.9, 0.2));
    sim.bodies[i].hav.v = Vec3::new(-0.9954741, 0.23042554, -1.4827863);
    sim.bodies[i].hav.w = Vec3::new(-10.097548, 1.7169005, 6.7314606);
    let g = ground(0.9, 0.2);
    for _ in 0..4 {
        sim.tick(&g);
        assert!(sim.bodies[i].hav.v.length() < 3.0, "launched: {:?}", sim.bodies[i].hav.v);
    }
}

/// `Sim::retire` removes a body without renumbering: the block that lay on it falls to the ground.
#[test]
fn retired_bodies_stop_colliding() {
    let mut sim = Sim::new(info());
    let low = sim.add(block(Vec3::new(0.0, 0.0, 0.5164), 10.0, 0.9, 0.0));
    let high = sim.add(block(Vec3::new(0.0, 0.0, 1.5164), 10.0, 0.9, 0.0));
    let g = ground(0.9, 0.0);
    for _ in 0..200 {
        sim.tick(&g);
    }
    assert!((sim.bodies[high].hav.com.z - 1.5164).abs() < 0.02 && !sim.bodies[high].awake);
    sim.retire(low);
    sim.wake(high);
    for _ in 0..200 {
        sim.tick(&g);
    }
    assert!((sim.bodies[high].hav.com.z - 0.5164).abs() < 0.02, "the upper block came down: {}", sim.bodies[high].hav.com.z);
}

/// Static friction (the engine's slip-anchor memory): a flat slab rests on slopes up to the angle of
/// repose `atan(mu)` of the QUANTISED mixed friction (0.9 * 0.9 -> sqrt = 0.9 -> table 0.934 ->
/// 43.0 degrees) and slides beyond it with the Coulomb acceleration `g (sin - mu cos)`.
#[test]
fn static_friction_holds_up_to_the_angle_of_repose() {
    let run = |deg: f32, ticks: usize| -> (f32, f32) {
        let th = deg.to_radians();
        let rot = Quat::from_rotation_y(-th);
        let n = rot * Vec3::Z;
        let downhill = -(rot * Vec3::X);
        let (c, s) = (Vec3::ZERO, 200.0);
        let (u, v) = (rot * Vec3::X, Vec3::Y);
        let quad = StaticPoly { verts: vec![c - u * s - v * s, c + u * s - v * s, c + u * s + v * s, c - u * s + v * s], radius: ENV_RADIUS };
        let world = PolyList::new(vec![quad], (0.9, 0.0));
        let mut sim = Sim::new(info());
        // slab half z 0.1 + 2 radii above the plane
        let hover = 0.1 + 2.0 * ENV_RADIUS;
        let mut body = box_body(Vec3::new(0.5, 0.5, 0.1), 10.0, c + n * (hover + 0.0005), 0.9, 0.0);
        body.hav.q = rot;
        let i = sim.add(body);
        let start = sim.bodies[i].hav.com;
        for _ in 0..ticks {
            sim.tick(&world);
        }
        let b = &sim.bodies[i].hav;
        ((b.com - start).dot(downhill), b.v.dot(downhill))
    };
    let mu = ContactMaterial::mix(0.9, 0.0, 0.9, 0.0).friction();
    let repose = mu.atan().to_degrees();
    assert!((repose - 43.0).abs() < 0.5, "repose {repose}");
    for deg in [20.0f32, 35.0, 41.0] {
        let (moved, _) = run(deg, 300);
        assert!(moved.abs() < 0.02, "{deg} deg: crept {moved}");
    }
    // 50 degrees: slides with a = g (sin 50 - mu cos 50) = 0.7 wu/s^2: v after 3 s ~ 2.1 (+-20 %)
    let th = 50.0f32.to_radians();
    let a = WORLD_G * (th.sin() - mu * th.cos());
    let (_, v) = run(50.0, 180);
    assert!(v > 3.0 * a * 0.8 && v < 3.0 * a * 1.2, "50 deg: v {v} vs Coulomb {}", 3.0 * a);
}
