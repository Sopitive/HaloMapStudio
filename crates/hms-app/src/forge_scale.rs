//! Forge object scale conventions — a faithful port of the C#
//! `MccMapStudio/Scene/ForgeScaleConvention.cs` (itself a bit-exact port of
//! Mjolnir-Forge-Editor's `forge.py::spawnSeqToScale`).
//!
//! Some custom Reach gametypes use the `SCALE` gtLabel to overload an object's
//! `spawnSequence` (signed byte, −100..+100) into a uniform visual scale
//! multiplier. There are FOUR community conventions:
//!
//! | convention | author / tool                      | 1% sentinel seq |
//! |------------|------------------------------------|-----------------|
//! | `X33`      | Trusty's Old                       | −10 |
//! | `X47`      | Anvil Editor / Trusty's New (default in Anvil) | −10 |
//! | `X71`      | Rabid MidgetMan's                  | −10 |
//! | `X330`     | Tx Titan Scale (broadest range)    | −20 |
//!
//! TEAM only matters for **X330**: RED team seeds the recursion at 32732
//! ("cosmic scaling") instead of 100, amplifying the *same* recursive algorithm
//! to a much larger range. It is NOT a separate convention — every other team
//! (and every other convention) ignores it.

/// A named Forge scale-encoding convention. The converter's "Convert from"
/// picker selects one; the object renderer always uses [`ScaleConvention::X330`]
/// (the label the engine's `SCALE` script encodes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleConvention {
    /// 33X — Trusty's Old.
    X33,
    /// 71X — Rabid MidgetMan's.
    X71,
    /// 47X — Anvil Editor / Trusty's New.
    X47,
    /// 330X — Tx Titan Scale (broadest range; the object-render default).
    X330,
}

/// Team branch. Only `Red` changes anything, and only under [`ScaleConvention::X330`],
/// where it seeds the recursion at 32732 ("cosmic scaling").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleTeam {
    None,
    Red,
}

/// Team index for RED in the raw (`−1`-normalised) mvar team field.
const TEAM_RED: u8 = 0;

/// Map the object's raw team byte to the scale branch (only red is cosmic).
#[inline]
pub fn team_from_u8(team: u8) -> ScaleTeam {
    if team == TEAM_RED { ScaleTeam::Red } else { ScaleTeam::None }
}

impl ScaleConvention {
    /// The conventions offered in the "Convert from" picker.
    /// X330 first (object-render default), then the flat conventions.
    pub const ALL: &'static [ScaleConvention] =
        &[Self::X330, Self::X47, Self::X33, Self::X71];

    pub fn label(self) -> &'static str {
        match self {
            Self::X33 => "33X — Trusty's Old",
            Self::X71 => "71X — Rabid MidgetMan's",
            Self::X47 => "47X — Anvil / Trusty's New",
            Self::X330 => "330X — Tx Titan Scale",
        }
    }

    pub fn hover(self) -> &'static str {
        match self {
            Self::X33 => "Trusty's Old convention. Flat (non-recursive) mapping; 1% sentinel at spawn seq −10.",
            Self::X71 => "Rabid MidgetMan's convention. Flat mapping; 1% sentinel at spawn seq −10.",
            Self::X47 => "Anvil Editor / Trusty's New convention (Anvil's default). Integer-domain mapping; 1% sentinel at spawn seq −10.",
            Self::X330 => "Tx Titan Scale — the broadest range, recursive. 1% sentinel at spawn seq −20. RED team uses the cosmic branch (recursion seeded at 32732 → up to ×151485 vs ×327).",
        }
    }

    /// Does the RED-team "cosmic" branch change this convention's result? (Only X330.)
    #[cfg(test)]
    pub fn uses_cosmic(self) -> bool {
        matches!(self, Self::X330)
    }

    /// spawnSequence → scale multiplier under this convention + team.
    #[cfg(test)]
    pub fn seq_to_scale(self, seq: i32, team: ScaleTeam) -> f32 {
        spawn_seq_to_scale(seq, self, team)
    }

    /// scale multiplier → nearest (spawnSequence, actual-scale) under this
    /// convention (team-neutral — the inverse never changes an object's team).
    #[cfg(test)]
    pub fn scale_to_seq(self, target: f32) -> (i8, f32) {
        scale_to_spawn_seq(target, self)
    }

    /// Largest default-team (non-cosmic) scale this convention can encode.
    pub fn max_scale(self) -> f32 {
        max_default_scale(self)
    }
}

/// Port of `forge.py::recursive_330x` (flattened to a loop). Callers guarantee
/// `i >= 0` and `scale > 0`, so C# `/` (truncation) matches Python `//` (floor).
fn recursive_330x(mut i: i64, mut scale: i64) -> i64 {
    while i > 0 {
        let step = scale / 33 + scale / 228;
        scale += step;
        i -= 1;
    }
    scale
}

/// spawnSequence (−100..+100) → uniform scale multiplier. Bit-exact port of
/// `ForgeScaleConvention.SpawnSeqToScale` / `forge.py::spawnSeqToScale`.
pub fn spawn_seq_to_scale(seq: i32, convention: ScaleConvention, team: ScaleTeam) -> f32 {
    // The 1% sentinel is convention-dependent: X330 = −20, everything else −10.
    let one_percent_seq = if convention == ScaleConvention::X330 { -20 } else { -10 };
    if seq == one_percent_seq {
        return 0.01;
    }
    match convention {
        ScaleConvention::X33 => {
            let mut scale = 0.1 * seq as f64;
            if seq < -10 {
                scale *= -2.0;
                if seq > -81 {
                    scale += 8.0;
                } else if seq < -80 {
                    scale = 2.0 * scale - 8.0;
                }
            }
            scale += 1.0;
            scale as f32
        }
        ScaleConvention::X71 => {
            let mut scale = 0.1 * seq as f64;
            if seq < -10 {
                scale *= -4.0;
                if seq > -81 {
                    scale += 6.0;
                } else if seq < -80 {
                    scale = 4.0 * scale - 90.0;
                }
            }
            scale += 1.0;
            scale as f32
        }
        ScaleConvention::X47 => {
            // Integer-domain until the final ×0.01 (forge.py L125-137).
            let mut scale = seq as i64;
            let lthn10 = seq < -10;
            let gthn71 = seq > -71;
            let gthn41 = seq > -41;
            if lthn10 {
                scale = 2 * (scale + 101);
                if gthn71 {
                    scale *= if gthn41 { 3 } else { 2 };
                }
            }
            scale = 10 * scale + 100;
            if lthn10 {
                scale += 1000;
                if gthn71 {
                    scale -= if gthn41 { 1800 } else { 600 };
                }
            }
            (scale as f64 * 0.01) as f32
        }
        ScaleConvention::X330 => {
            let mut scale: i64 = 100;
            let mut i: i64 = seq as i64;
            if seq < 0 {
                i *= 5;
                scale += i;
                if seq <= -20 {
                    i = seq as i64 + 201;
                    if seq == -20 {
                        scale = 1;
                    }
                }
            }
            if seq < -20 || seq > 0 {
                // RED = cosmic seed 32732; otherwise 100. Same recursion.
                let seed = if team == ScaleTeam::Red { 32732 } else { 100 };
                scale = recursive_330x(i, seed);
            }
            (scale as f64 * 0.01) as f32
        }
    }
}

/// Inverse: nearest (spawnSequence, actual-scale) whose forward lookup is closest
/// to `target` in LOG space, scanning the whole −100..=100 domain at team=None.
/// Team-neutral by design — the converter must change only the spawn sequence,
/// never the object's team (matching the C# `TEAM_NEUTRAL_CONVERT`).
pub fn scale_to_spawn_seq(target: f32, convention: ScaleConvention) -> (i8, f32) {
    let target_log = target.max(1e-6).ln();
    let mut best = 0i8;
    let mut best_scale = 1.0f32;
    let mut best_err = f32::MAX;
    for sq in -100i32..=100 {
        let s = spawn_seq_to_scale(sq, convention, ScaleTeam::None);
        if s <= 0.0 {
            continue;
        }
        let err = (s.ln() - target_log).abs();
        if err < best_err {
            best_err = err;
            best = sq as i8;
            best_scale = s;
        }
    }
    (best, best_scale)
}

/// Largest default-team (non-cosmic) scale a convention can encode.
fn max_default_scale(convention: ScaleConvention) -> f32 {
    let mut best = 0.0f32;
    for sq in -100i32..=100 {
        let s = spawn_seq_to_scale(sq, convention, ScaleTeam::None);
        if s > best {
            best = s;
        }
    }
    best
}

// ---- Object-render convenience (the forge "scale" label always encodes X330) ----

/// Object visual scale from its spawnSequence + team byte (X330; red = cosmic).
pub fn object_scale(spawn_seq: i32, team: u8) -> f32 {
    spawn_seq_to_scale(spawn_seq, ScaleConvention::X330, team_from_u8(team))
}

/// Render clamp: the largest scale the object's X330 branch can reach.
pub fn object_max_scale(team: u8) -> f32 {
    let t = team_from_u8(team);
    let mut best = 0.0f32;
    for sq in -100i32..=100 {
        let s = spawn_seq_to_scale(sq, ScaleConvention::X330, t);
        if s > best {
            best = s;
        }
    }
    best.max(1.0)
}

/// Encode a desired object scale back into a spawnSequence (X330, team-neutral).
pub fn object_scale_to_seq(target: f32) -> (i8, f32) {
    scale_to_spawn_seq(target, ScaleConvention::X330)
}

// ---- Per-object PSEUDO-flags (never written to the .mvar) ----
//
// Two per-object switches the editor keeps beside each placed object:
//   * SCALED — "behave as a scaled object": the spawn sequence is read as an X330 size.
//     Default ON exactly when the object carries the forge `scale` label.
//   * SHADOW — draw the object into the sun shadow map so it shadows the ground / other
//     objects. Default ON for the community gametype rule "GREEN team + `scale` label casts a
//     shadow" (Megalo scripts people set up in game), so a shadow-casting map renders as it
//     would look with that gametype loaded.
// Each flag is an `Option<bool>` OVERRIDE over a DERIVED default: `None` follows the rule live
// (change the team/label and the default re-derives), `Some(x)` is the user's explicit choice and
// survives until they change it. Engine-default casters (vehicles, weapons — obje
// lightmap_shadow_mode) cast regardless; the flag only ADDS casters.

/// Forge team index for GREEN in the raw mvar team byte (0 red, 1 blue, 2 green, …).
pub const TEAM_GREEN: u8 = 2;

/// Is this the forge `scale` label (case-insensitive)?
#[inline]
pub fn is_scale_label(label: &str) -> bool {
    label.trim().eq_ignore_ascii_case("scale")
}

/// Derived SCALED default: the object has the `scale` label.
#[inline]
fn scaled_default(label: &str) -> bool {
    is_scale_label(label)
}

/// Derived SHADOW default: the gametype rule — GREEN team AND the `scale` label.
#[inline]
fn shadow_default(team: u8, label: &str) -> bool {
    team == TEAM_GREEN && is_scale_label(label)
}

/// The two per-object pseudo-flags as user overrides (`None` = follow the derived default).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct ObjFlags {
    pub scaled: Option<bool>,
    pub shadow: Option<bool>,
}

impl ObjFlags {
    /// Effective SCALED state for an object with this label.
    #[inline]
    pub fn scaled_on(&self, label: &str) -> bool {
        self.scaled.unwrap_or_else(|| scaled_default(label))
    }
    /// Effective SHADOW state for an object with this team + label.
    #[inline]
    pub fn shadow_on(&self, team: u8, label: &str) -> bool {
        self.shadow.unwrap_or_else(|| shadow_default(team, label))
    }
    /// True when neither flag is overridden (nothing worth persisting).
    #[inline]
    pub fn is_default(&self) -> bool {
        self.scaled.is_none() && self.shadow.is_none()
    }
}

/// Parse a script/CLI boolean: on/off, true/false, 1/0, yes/no; `default`/`auto`/`none` clears
/// an override (returns `Ok(None)`).
pub fn parse_flag_value(v: &str) -> Result<Option<bool>, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(Some(true)),
        "off" | "false" | "0" | "no" => Ok(Some(false)),
        "default" | "auto" | "none" | "derive" | "reset" => Ok(None),
        other => Err(format!("expected on|off|default, got '{other}'")),
    }
}

/// The two GLOBAL switches (Settings ▸ Lighting & rendering): both default ON.
///   * `scaled` OFF → no object scales, whatever its flag says.
///   * `shadowcasters` OFF → only the engine-default casters cast (vehicles etc.); no Forge
///     piece is added to the shadow map by the SHADOW flag / green+scale rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GlobalFlags {
    pub scaled: bool,
    pub shadowcasters: bool,
}

impl Default for GlobalFlags {
    fn default() -> Self {
        Self { scaled: true, shadowcasters: true }
    }
}

const SCALED_SETTING_FILE: &str = "scaled_objects.txt";
const CASTERS_SETTING_FILE: &str = "shadow_casters.txt";

fn read_bool_file(dir: &std::path::Path, file: &str) -> Option<bool> {
    let s = std::fs::read_to_string(dir.join(file)).ok()?;
    parse_flag_value(&s).ok().flatten()
}

impl GlobalFlags {
    /// The persisted toggles (settings dir; default ON when absent/unreadable), then the
    /// headless/batch environment overrides `HMS_SCALED=0|1` / `HMS_SHADOWCASTERS=0|1` on top.
    pub fn load() -> Self {
        let g = match crate::project::settings_dir() {
            Some(dir) => Self::load_from_dir(&dir),
            None => Self::default(),
        };
        g.apply_env()
    }

    /// Read the toggles from an explicit settings directory (no env override).
    pub fn load_from_dir(dir: &std::path::Path) -> Self {
        let mut g = Self::default();
        if let Some(v) = read_bool_file(dir, SCALED_SETTING_FILE) { g.scaled = v; }
        if let Some(v) = read_bool_file(dir, CASTERS_SETTING_FILE) { g.shadowcasters = v; }
        g
    }

    /// Apply the `HMS_SCALED` / `HMS_SHADOWCASTERS` environment overrides (batch renders).
    fn apply_env(mut self) -> Self {
        if let Some(v) = std::env::var("HMS_SCALED").ok().and_then(|s| parse_flag_value(&s).ok().flatten()) {
            self.scaled = v;
        }
        if let Some(v) = std::env::var("HMS_SHADOWCASTERS").ok().and_then(|s| parse_flag_value(&s).ok().flatten()) {
            self.shadowcasters = v;
        }
        self
    }

    /// Persist both toggles (real settings — the GUI reloads them next start).
    pub fn save(&self) {
        if let Some(dir) = crate::project::settings_dir() {
            self.save_to_dir(&dir);
        }
    }

    /// Write the toggles into an explicit settings directory.
    pub fn save_to_dir(&self, dir: &std::path::Path) {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(SCALED_SETTING_FILE), if self.scaled { "1" } else { "0" });
        let _ = std::fs::write(dir.join(CASTERS_SETTING_FILE), if self.shadowcasters { "1" } else { "0" });
    }

    pub fn status_line(&self) -> String {
        format!(
            "scale {} | shadowcasters {}",
            if self.scaled { "on" } else { "off" },
            if self.shadowcasters { "on" } else { "off" }
        )
    }
}

/// Effective per-object result under the globals: (scaled, forge-caster). `forge_caster` is only
/// the ADDED (flag/gametype) casting — engine-default casters are decided by the scene from the
/// obje tag and are unaffected by either switch.
#[inline]
pub fn effective_flags(g: &GlobalFlags, f: &ObjFlags, team: u8, label: &str) -> (bool, bool) {
    (g.scaled && f.scaled_on(label), g.shadowcasters && f.shadow_on(team, label))
}

#[cfg(test)]
mod obj_flag_tests {
    use super::*;

    /// The derivation rule. SCALED follows the `scale` label; SHADOW follows the
    /// gametype rule GREEN + `scale`; an explicit override wins over both; clearing it re-derives.
    #[test]
    fn defaults_follow_label_and_green_team() {
        let f = ObjFlags::default();
        assert!(f.scaled_on("scale"));
        assert!(f.scaled_on("SCALE"), "label match must be case-insensitive");
        assert!(!f.scaled_on(""));
        assert!(!f.scaled_on("koth_hill"));
        // shadow: green + scale only
        assert!(f.shadow_on(TEAM_GREEN, "scale"));
        assert!(!f.shadow_on(TEAM_RED, "scale"), "red + scale is the cosmic-scale branch, not a caster");
        assert!(!f.shadow_on(TEAM_GREEN, ""), "green without the label does not cast");
        assert!(!f.shadow_on(0xFF, "scale"), "no team does not cast");
        assert!(!f.shadow_on(8, "scale"), "neutral does not cast");
    }

    #[test]
    fn override_wins_and_clears() {
        let mut f = ObjFlags { scaled: Some(false), shadow: Some(true) };
        assert!(!f.scaled_on("scale"), "explicit off beats the label default");
        assert!(f.shadow_on(0xFF, ""), "explicit on beats the rule default");
        assert!(!f.is_default());
        f.scaled = None;
        f.shadow = None;
        assert!(f.is_default());
        // moving the object onto green with the scale label starts casting again
        assert!(f.shadow_on(TEAM_GREEN, "scale"));
        assert!(!f.shadow_on(1, "scale"));
    }

    #[test]
    fn globals_gate_everything() {
        let f = ObjFlags { scaled: Some(true), shadow: Some(true) };
        let on = GlobalFlags::default();
        assert_eq!(effective_flags(&on, &f, 0xFF, ""), (true, true));
        let off = GlobalFlags { scaled: false, shadowcasters: false };
        assert_eq!(effective_flags(&off, &f, TEAM_GREEN, "scale"), (false, false));
        let half = GlobalFlags { scaled: false, shadowcasters: true };
        assert_eq!(effective_flags(&half, &ObjFlags::default(), TEAM_GREEN, "scale"), (false, true));
    }

    #[test]
    fn flag_value_grammar() {
        assert_eq!(parse_flag_value("on"), Ok(Some(true)));
        assert_eq!(parse_flag_value("FALSE"), Ok(Some(false)));
        assert_eq!(parse_flag_value("1"), Ok(Some(true)));
        assert_eq!(parse_flag_value("default"), Ok(None));
        assert!(parse_flag_value("maybe").is_err());
    }

    /// The globals are REAL persisted settings (no env needed): save → load round-trips, and an
    /// empty directory yields the defaults (both on).
    #[test]
    fn globals_round_trip_through_settings_dir() {
        let tmp = std::env::temp_dir().join(format!("hms-objflags-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(GlobalFlags::load_from_dir(&tmp), GlobalFlags::default(), "missing files = defaults");
        GlobalFlags { scaled: false, shadowcasters: true }.save_to_dir(&tmp);
        assert_eq!(GlobalFlags::load_from_dir(&tmp), GlobalFlags { scaled: false, shadowcasters: true });
        GlobalFlags { scaled: true, shadowcasters: false }.save_to_dir(&tmp);
        assert_eq!(GlobalFlags::load_from_dir(&tmp), GlobalFlags { scaled: true, shadowcasters: false });
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Anchor values cross-checked against ForgeScaleConvention.cs / forge.py.
    #[test]
    fn identity_at_zero_and_sentinels() {
        for c in [ScaleConvention::X33, ScaleConvention::X47, ScaleConvention::X71, ScaleConvention::X330] {
            assert!((c.seq_to_scale(0, ScaleTeam::None) - 1.0).abs() < 0.02, "{c:?} @0");
        }
        // Convention-dependent 1% sentinel.
        assert!((ScaleConvention::X330.seq_to_scale(-20, ScaleTeam::None) - 0.01).abs() < 1e-4);
        assert!((ScaleConvention::X47.seq_to_scale(-10, ScaleTeam::None) - 0.01).abs() < 1e-4);
        assert!((ScaleConvention::X33.seq_to_scale(-10, ScaleTeam::None) - 0.01).abs() < 1e-4);
    }

    #[test]
    fn x33_x47_x71_flat_mappings() {
        // 0.1*seq + 1 on the positive side.
        assert!((ScaleConvention::X33.seq_to_scale(100, ScaleTeam::None) - 11.0).abs() < 0.01);
        assert!((ScaleConvention::X71.seq_to_scale(100, ScaleTeam::None) - 11.0).abs() < 0.01);
        // X47 positive: 10*seq+100 → ×0.01.
        assert!((ScaleConvention::X47.seq_to_scale(100, ScaleTeam::None) - 11.0).abs() < 0.01);
    }

    #[test]
    fn x330_cosmic_only_and_amplifies() {
        // Cosmic changes X330...
        assert!(
            ScaleConvention::X330.seq_to_scale(100, ScaleTeam::Red)
                > ScaleConvention::X330.seq_to_scale(100, ScaleTeam::None) * 10.0
        );
        // ...but NOT the flat conventions (team ignored there).
        assert_eq!(
            ScaleConvention::X47.seq_to_scale(100, ScaleTeam::Red),
            ScaleConvention::X47.seq_to_scale(100, ScaleTeam::None)
        );
        assert!(ScaleConvention::X330.uses_cosmic());
        assert!(!ScaleConvention::X47.uses_cosmic());
    }

    #[test]
    fn round_trip_stable_each_convention() {
        for c in ScaleConvention::ALL {
            for seq in -100i32..=100 {
                let s = c.seq_to_scale(seq, ScaleTeam::None);
                if s <= 0.0 {
                    continue;
                }
                let (back, _) = c.scale_to_seq(s);
                let re = c.seq_to_scale(back as i32, ScaleTeam::None);
                assert!((re - s).abs() <= s.abs() * 0.05 + 0.02, "{c:?} seq {seq}: {s} -> {back} -> {re}");
            }
        }
    }
}
