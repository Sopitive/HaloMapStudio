//! #havok-solver  The engine's real Havok rigid-body pipeline, ported from the decompiles.
//!
//! The previous layer (`reach_havok::solve_contacts` / `solve_pair_contacts`, `physics_settle`) was a
//! sequential-impulse stand-in. What Reach runs is stock Havok 2011 (`hkpContinuousSimulation`)
//! with the engine's world settings, and every piece below is a port of a decompiled function of
//! `haloreach_debug.dll` (retail twins confirmed by structure). The decode documents are in
//! `research/havok/`: `01-collision-detection.md` (agents, contact generation, welding, materials),
//! `02-contact-solver.md` (this module's solver, line by line), `03-continuous-toi.md`,
//! `04-deactivation-islands.md`, `05-shapes-mass-inertia.md`, `06-port-and-wiring.md` (this port:
//! gates, saboteurs, deviations, wiring). Every claim there is labelled PROVEN / INFERRED /
//! UNRESOLVED; the doc comments below repeat the label of each rule.
//!
//! # Map
//!
//! | file | engine function(s) | what |
//! |---|---|---|
//! | [`info`] | `sub_180D4DE10` (world ctor), `sub_180D72850` | tau/damp/iterations -> row scales, the six deactivation-class rows |
//! | [`manifold`] | `sub_180D77270`, `sub_180D78040`, `sub_180D776E0`, `sub_180E2A180` | contact points, material mix, persistence, admission rules, the 4-point reduction |
//! | [`solve`] | `sub_180D8AC10`, `sub_180D99B00`, `sub_180D9AEE0`, `sub_180DA26B0`, `sub_180D883A0`, `sub_180D86820` | accumulators, row builder, `hkSolve`, post-solve, write-back |
//! | [`sleep`] | tails of `sub_180D883A0`/`sub_180D86820`, `sub_180D72850`, `sub_180D5A0C0` | inactivity windows, island rule, sleep request |
//! | [`poly`], [`shape`], [`gjk`] | `sub_180D0AB60` agents, `sub_180E2A180`, `sub_180D3F730` | contact generation: SAT + clipping, GJK/EPA, spheres/capsules |
//! | [`toi`] | `sub_180D0C1D0`, `sub_180D2C670`, `sub_180E2AAD0`, `sub_180D7CB80`, `sub_180D7E340` | quality-pair table, TOI search, one-shot impulse |
//! | [`sim`] | `hkpContinuousSimulation::stepDeltaTime` | the whole step: sleep requests, integrate, collide, advance time |
//!
//! # API (what the prop world wires in)
//!
//! * [`Sim::new`]`(`[`SolverInfo::engine`]`(60.0, gravity))`, [`Sim::add`] a [`SimBody`] per live
//!   prop (`SimBody::new(hav, shapes, PROP_BODY_MATERIAL...)`), then one [`Sim::tick`]`(&world)` per
//!   game tick (dt = 1/60; Havok steps once per tick). `world` is any [`StaticSource`]: the polygons
//!   of the level collision BSP ([`StaticPoly`], radius [`ENV_RADIUS`]) plus the Fixed object hulls.
//! * A body sleeps by itself ([`SimBody::awake`]) and wakes when an awake body's bounds touch it;
//!   set `hav.v/w` and clear nothing else to push it. A KEYFRAMED body (lift, player capsule) is a
//!   `MotionType::Keyframed`, quality `Keyframed` body whose `hav.v/w` the caller sets each tick.
//! * Engine materials: a prop is friction 1.0 / restitution 0.0 ([`PROP_BODY_MATERIAL`]) and the
//!   level 0.5 / 0.4 ([`ENV_BODY_MATERIAL`]); the solver mixes `sqrt(fA fB)` and quantises (see
//!   [`ContactMaterial`]). Per-shape `phmo` values are not used by the solver (`01` 5.3).
//! * Lower-level pieces ([`step`], [`Manifold`], [`poly::collide`], [`toi::find_toi`]) are public
//!   for a caller that owns its own broad phase.
//!
//! # Deviations from the engine (each INFERRED or a named omission)
//!
//! * the narrow phase is SAT + clipping (boxes/hulls) and GJK/EPA (spheres, capsules as 3 spheres),
//!   not the engine's `PRED_GSK_AGENT3` feature caches; the engine's ADMISSION rules are applied on
//!   top (`Manifold::update`);
//! * the "TOI solver" over a growing island after an event (tau 0.5, damping 1.4) is not ported;
//! * the 16:1 mass clamp (`havok_utilities.cpp:365`) is applied ([`Sim::mass_clamp`]) although the
//!   debug build has no caller of it (UNRESOLVED whether Reach attaches it);
//! * gyroscopic angular-momentum conservation of thin-box motions, `maxImpulse` rows, constraints.

#![allow(dead_code, unused_imports)] // the prop world wires this in (coordinator); until then nothing calls it

mod gjk;
mod info;
mod manifold;
mod poly;
mod shape;
mod sim;
mod sleep;
mod toi;
mod solve;

pub use info::{DeactClass, SolverInfo};
pub use manifold::{Admission, AtomState, Candidate, ContactMaterial, ContactPoint, Manifold, PointState};
pub use poly::{collide as collide_polys, Poly, RawContact};
pub use shape::{LocalShape, WShape};
pub use sim::{PolyList, Sim, SimBody, StaticPoly, StaticSource, COLLISION_TOLERANCE, ENV_BODY_MATERIAL, ENV_RADIUS, PROP_BODY_MATERIAL};
pub use sleep::SleepState;
pub use solve::{step, BodyAux, StepReport};

#[cfg(test)]
mod gates;
