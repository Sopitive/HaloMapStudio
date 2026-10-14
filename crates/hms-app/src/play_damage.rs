//! #play  THE DAMAGE SEAM — declared here, implemented elsewhere, deliberately.
//!
//! Test mode can melee a vehicle, and the user wants that vehicle to take damage. **The damage
//! law is not this module's and must never be guessed here**: which `jpt!` a swing applies, how
//! it falls off with distance, which damage SECTION of the target model the hit lands on and
//! which `model_state_enum` that section moves to are all being recovered from the engine and
//! its tags by the agent doing that work. A plausible-but-wrong damage number would be worse
//! than none, because it would compete with the measured one.
//!
//! So this file is a seam and nothing else. Test mode holds an `Option<Box<dyn ObjectDamage>>`,
//! and with it absent the caption says so rather than pretending.
//!
//! # What each side owns
//!
//! **This side** finds a target and describes the hit: a datum, a world point, a direction and
//! what caused it. Nothing here resolves a damage section, reads a tag, or tracks vitality.
//!
//! **The other side** owns everything after that: the `jpt!` for the source, the falloff, the
//! damage sections the point lands on, the vitality that makes the second hit different from the
//! first (which is why [`ObjectDamage::apply`] takes `&mut self`) and the response that fires.
//!
//! The payoff is already built: HMS renders `model_state_enum` damage states today — the DAMAGE
//! dropdown drives `hlmt` variant-permutation states, and Theater decodes a recorded vehicle's
//! states — so "shoot a vehicle and watch it visibly take damage" is response → state → render
//! already working. Only hit → damage → response was missing, and that is theirs.

// ---------------------------------------------------------------------------------------------
// ONE declaration, and it is the damage side's. Test mode only CONSUMES it.
// ---------------------------------------------------------------------------------------------

/// Re-exported from `reach_damage`, which owns the damage law and therefore owns these types.
/// Test mode used to declare its own copies while that module was being written; keeping two
/// declarations of one contract is how the two halves drift apart, so there is now exactly one.
///
/// **`DamageReport::applied` is `Option<f32>` and the `None` must be PRINTED, not defaulted.**
/// The `jpt!` falloff radii are not measured yet, so there are hits whose damage genuinely cannot
/// be determined — and a `0.0` there would read as "took no damage", which looks like a bug in
/// the swing rather than an honest absence. Test mode's caption says "damage not determined".
pub use crate::reach_damage::{DamageHit, DamageReport, DamageSource, ObjectDamage, NO_MODEL};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// A stand-in that only records what it was handed — the seam's own gate. It proves test mode
    /// fills every field of a hit, which is what the damage side needs, without asserting
    /// anything about damage itself.
    struct Recorder(Vec<DamageHit>);
    impl ObjectDamage for Recorder {
        fn apply(&mut self, hit: DamageHit) -> Option<DamageReport> {
            self.0.push(hit);
            Some(DamageReport { applied: None, note: "recorded".into(), ..Default::default() })
        }
    }

    #[test]
    fn the_seam_carries_a_whole_hit_and_is_stateful() {
        let mut r = Recorder(Vec::new());
        let h = DamageHit {
            target_datum: 0x1234,
            point: Vec3::new(1.0, 2.0, 3.0),
            dir: Vec3::X,
            source: DamageSource::Melee { weapon: 7 },
            distance: 0.0,
        };
        assert!(r.apply(h).is_some());
        assert!(r.apply(h).is_some());
        // Two calls, two records: the model is allowed to remember, which a function pointer
        // could not.
        assert_eq!(r.0.len(), 2);
        assert_eq!(r.0[0].distance, 0.0, "a melee is a CONTACT hit");
        assert_eq!(r.0[0].source.label(), "melee");
    }

    /// #play-gate  AN UNDETERMINED DAMAGE IS PRINTED AS UNDETERMINED. The whole point of
    /// `applied` being an `Option` is that `0.0` would look like "took no damage" when the truth
    /// is "not known yet", so the formatting test is the gate.
    #[test]
    fn an_undetermined_damage_is_not_printed_as_zero() {
        let r = DamageReport { applied: None, note: "contact hit".into(), ..Default::default() };
        let line = crate::play_mode::describe_damage(0x1234, 0.5, &r);
        assert!(line.contains("not determined"), "{line}");
        assert!(!line.contains("0 damage") && !line.contains("0.0 damage"), "{line}");
        let r2 = DamageReport { applied: Some(5.834), note: "assault rifle melee".into(), ..Default::default() };
        let line2 = crate::play_mode::describe_damage(0x1234, 0.5, &r2);
        assert!(line2.contains("5.83"), "{line2}");
    }
}
