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

    /// Short name for compact UI ("47X").
    pub fn short(self) -> &'static str {
        match self {
            Self::X33 => "33X",
            Self::X71 => "71X",
            Self::X47 => "47X",
            Self::X330 => "330X",
        }
    }

    /// Stable key for the settings file.
    pub fn key(self) -> &'static str {
        match self {
            Self::X33 => "x33",
            Self::X71 => "x71",
            Self::X47 => "x47",
            Self::X330 => "x330",
        }
    }

    pub fn from_key(s: &str) -> Option<Self> {
        let t = s.trim();
        Self::ALL.iter().copied().find(|c| c.key().eq_ignore_ascii_case(t) || c.short().eq_ignore_ascii_case(t))
    }

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
    scale_to_spawn_seq_for_team(target, convention, ScaleTeam::None)
}

/// #scale-sel  scale multiplier -> nearest (spawnSequence, actual scale) **for a given team**.
///
/// The team matters and getting it wrong mis-encodes silently: under [`ScaleConvention::X330`] a
/// RED-team object reads the "cosmic" branch (the recursion is seeded at 32732 instead of 100), so
/// the same sequence byte means a far larger multiplier. [`ObjMeta::scale`] reads with the
/// object's own team, so the inverse has to invert with it too or a red-team object silently lands
/// on a wildly different size from the one asked for.
///
/// Searches in LOG space, because scale is multiplicative: the nearest representable value to
/// 2.0x should be judged by ratio, not by absolute difference, or a large target drags the choice
/// toward whichever huge step happens to be numerically closest.
pub fn scale_to_spawn_seq_for_team(
    target: f32,
    convention: ScaleConvention,
    team: ScaleTeam,
) -> (i8, f32) {
    let target_log = target.max(1e-6).ln();
    let mut best = 0i8;
    let mut best_scale = 1.0f32;
    let mut best_err = f32::MAX;
    for sq in -100i32..=100 {
        let s = spawn_seq_to_scale(sq, convention, team);
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

// ---- #scale-use  Typing a size: the convention is a saved setting, not a converter option ----

/// The smallest and largest scale a convention can encode for a team (over the whole
/// spawn-sequence domain). RED under X330 reads the cosmic branch, so its range is far larger.
pub fn representable_range(convention: ScaleConvention, team: ScaleTeam) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::MAX, 0.0f32);
    for sq in -100i32..=100 {
        let s = spawn_seq_to_scale(sq, convention, team);
        if s > 0.0 {
            lo = lo.min(s);
            hi = hi.max(s);
        }
    }
    (lo, hi)
}

/// What setting a size resolves to: the spawn-sequence byte to write, the size that byte really
/// gives, how far it is from the request (as a ratio, `got / want - 1`), and whether the request
/// was outside what the convention can reach.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleSet {
    pub seq: i8,
    pub got: f32,
    pub ratio_err: f32,
    pub clamped: bool,
}

/// Resolve a typed size `want` (a multiplier: 1.0 = normal) to the nearest representable one under
/// `convention` for an object of this `team`. Nearest is judged by RATIO (see
/// [`scale_to_spawn_seq_for_team`]); a request outside the reachable range lands on the end of it
/// and reports `clamped`. A non-positive or non-finite request is refused (None).
pub fn set_scale(want: f32, convention: ScaleConvention, team: ScaleTeam) -> Option<ScaleSet> {
    if !(want.is_finite() && want > 0.0) {
        return None;
    }
    let (lo, hi) = representable_range(convention, team);
    let (seq, got) = scale_to_spawn_seq_for_team(want, convention, team);
    Some(ScaleSet { seq, got, ratio_err: got / want - 1.0, clamped: want < lo * 0.9999 || want > hi * 1.0001 })
}

/// Reach's variant label table holds 511 names (`mvar_labels` cap in the label picker).
pub const MAX_LABELS: usize = 511;

/// Index of the forge `scale` label in a variant's label table, appending it when the table lacks
/// one. `None` when the table is full: a size needs this label to mean anything, so the caller
/// must refuse rather than write a spawn sequence that the gametype will read as a spawn ORDER.
pub fn ensure_scale_label(labels: &mut Vec<String>) -> Option<u16> {
    if let Some(i) = labels.iter().position(|l| is_scale_label(l)) {
        return Some(i as u16);
    }
    if labels.len() >= MAX_LABELS {
        return None;
    }
    labels.push("scale".to_string());
    Some((labels.len() - 1) as u16)
}

const CONVENTION_SETTING_FILE: &str = "scale_convention.txt";

/// The convention the owner builds for (the settings dir; X330 when absent or unreadable).
pub fn load_convention_from_dir(dir: &std::path::Path) -> ScaleConvention {
    std::fs::read_to_string(dir.join(CONVENTION_SETTING_FILE)).ok()
        .and_then(|s| ScaleConvention::from_key(&s))
        .unwrap_or(ScaleConvention::X330)
}

pub fn save_convention_to_dir(dir: &std::path::Path, c: ScaleConvention) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(CONVENTION_SETTING_FILE), c.key());
}

pub fn load_convention() -> ScaleConvention {
    crate::project::settings_dir().map_or(ScaleConvention::X330, |d| load_convention_from_dir(&d))
}

pub fn save_convention(c: ScaleConvention) {
    if let Some(d) = crate::project::settings_dir() {
        save_convention_to_dir(&d, c);
    }
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

/// The per-object pseudo-flags as user overrides (`None` = follow the derived default).
/// `damage` is the DISPLAYED model damage state (hlmt `model_state_enum`: 1 minor, 2 medium,
/// 3 major, 4 destroyed; None = the regular undamaged state 0). A viewer/editor option like the
/// other two: never written to the .mvar, kept in the project file, and it only changes which
/// render_model permutations the object draws (physics / collision hulls are untouched).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct ObjFlags {
    pub scaled: Option<bool>,
    pub shadow: Option<bool>,
    pub damage: Option<u8>,
}

/// The hlmt `model_state_enum_definition` names, indexed by state (0..=4).
pub const DAMAGE_STATE_NAMES: [&str; 5] = ["default", "minor", "medium", "major", "destroyed"];

/// Human label for a damage state (0 = the undamaged default).
pub fn damage_state_name(state: u8) -> &'static str {
    DAMAGE_STATE_NAMES.get(state as usize).copied().unwrap_or("?")
}

/// Parse a script/CLI damage state: `default|minor|medium|major|destroyed` (also `minor damage`,
/// `medium_damage` ...) or the enum index 0..4. `default`/`0`/`auto`/`none` clears the override
/// (returns `Ok(None)`).
pub fn parse_damage_value(v: &str) -> Result<Option<u8>, String> {
    let t = v.trim().to_ascii_lowercase();
    let t = t.trim_end_matches("damage").trim_end_matches(['_', '-', ' ']);
    match t {
        "default" | "auto" | "none" | "undamaged" | "reset" | "0" | "normal" => Ok(None),
        "minor" | "1" => Ok(Some(1)),
        "medium" | "2" => Ok(Some(2)),
        "major" | "3" => Ok(Some(3)),
        "destroyed" | "destroy" | "dead" | "4" => Ok(Some(4)),
        other => Err(format!("expected default|minor|medium|major|destroyed (or 0..4), got '{other}'")),
    }
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
    /// The displayed damage state (0 = the regular undamaged model).
    #[inline]
    pub fn damage_state(&self) -> u8 {
        self.damage.unwrap_or(0)
    }
    /// True when nothing is overridden (nothing worth persisting).
    #[inline]
    pub fn is_default(&self) -> bool {
        self.scaled.is_none() && self.shadow.is_none() && self.damage.is_none()
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
        let mut f = ObjFlags { scaled: Some(false), shadow: Some(true), damage: None };
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
        let f = ObjFlags { scaled: Some(true), shadow: Some(true), damage: None };
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

    /// The damage-state override: names and enum indices, `default` clears, and a fresh
    /// `ObjFlags` with only a damage state set is no longer "default" (so it persists).
    #[test]
    fn parse_damage_value_names_and_indices() {
        assert_eq!(parse_damage_value("default"), Ok(None));
        assert_eq!(parse_damage_value("0"), Ok(None));
        assert_eq!(parse_damage_value("minor"), Ok(Some(1)));
        assert_eq!(parse_damage_value("Medium Damage"), Ok(Some(2)));
        assert_eq!(parse_damage_value("major_damage"), Ok(Some(3)));
        assert_eq!(parse_damage_value("DESTROYED"), Ok(Some(4)));
        assert_eq!(parse_damage_value("4"), Ok(Some(4)));
        assert!(parse_damage_value("5").is_err());
        assert!(parse_damage_value("broken").is_err());
        let f = ObjFlags { damage: Some(4), ..Default::default() };
        assert!(!f.is_default());
        assert_eq!(f.damage_state(), 4);
        assert_eq!(ObjFlags::default().damage_state(), 0);
        assert_eq!(damage_state_name(4), "destroyed");
        assert_eq!(damage_state_name(0), "default");
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

// ===================== #scale-sel  THE SCALE INVERSE =====================
//
// `commit_scale_sizes` turns "make this build 2x bigger" into a sequence byte per object. These
// gate the inverse, because a wrong answer here is SILENT: the object simply ends up a different
// size from the one asked for, which looks like a design choice rather than a bug.
#[cfg(test)]
mod scale_inverse_gates {
    use super::*;

    /// Assertion battery over the function under test, so the companion can drive broken
    /// variants of it without touching the shared tree. Each assertion is numbered so a
    /// saboteur can be pinned to the one it must trip.
    fn battery(
        inv: &dyn Fn(f32, ScaleConvention, ScaleTeam) -> (i8, f32),
        caught: &mut Vec<usize>,
    ) {
        let conv = ScaleConvention::X330;

        // 1. A factor of 1.0 must round-trip to the identity sequence, or every unscaled object
        //    in a selection would be nudged off its own size by being scaled by 1.
        let (seq, got) = inv(1.0, conv, ScaleTeam::None);
        if !(seq == 0 && (got - 1.0).abs() < 1e-6) {
            caught.push(1);
        }

        // 2. The result must be MONOTONE: asking for more must never give back less. A search
        //    that compares in the wrong space, or with an absolute error, can invert order.
        let mut prev = 0.0f32;
        let mut mono = true;
        for t in [0.25f32, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0] {
            let (_, g) = inv(t, conv, ScaleTeam::None);
            if g + 1e-6 < prev {
                mono = false;
            }
            prev = g;
        }
        if !mono {
            caught.push(2);
        }

        // 3. The chosen value must be the NEAREST representable one in LOG space -- and this
        //    assertion SWEEPS the ladder instead of testing a few hand-picked targets.
        //
        //    That is deliberate and it was earned: the first version used four guessed targets
        //    (0.3, 1.37, 2.0, 5.5) and an absolute-error saboteur passed it, because at those
        //    particular targets absolute-nearest and log-nearest happen to pick the SAME step.
        //    A correct assertion that the fixture never exercises proves nothing -- so instead
        //    of guessing again, the targets are DERIVED: between each adjacent pair of
        //    representable sizes sits a band where the two metrics disagree (between the
        //    geometric and the arithmetic mean), and every such band is tested.
        let ladder: Vec<f32> = {
            let mut v: Vec<f32> = (-100i32..=100)
                .map(|sq| spawn_seq_to_scale(sq, conv, ScaleTeam::None))
                .filter(|s| *s > 0.0)
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            v
        };
        let mut discriminating = 0usize;
        let mut wrong = 0usize;
        for w in ladder.windows(2) {
            let (a, b) = (w[0], w[1]);
            let geo = (a * b).sqrt();
            let ari = 0.5 * (a + b);
            if !(ari - geo > 1e-9) {
                continue;
            }
            let t = 0.5 * (geo + ari);
            let want = if (a.ln() - t.ln()).abs() <= (b.ln() - t.ln()).abs() { a } else { b };
            let other = if want == a { b } else { a };
            // Only a band where the two metrics actually disagree can discriminate.
            if (a - t).abs() < (b - t).abs() {
                if other != a {
                    continue;
                }
            } else if other != b {
                continue;
            }
            discriminating += 1;
            let (_, got) = inv(t, conv, ScaleTeam::None);
            if (got - want).abs() > 1e-4 {
                wrong += 1;
            }
        }
        // Positive liveness: the sweep must have found bands to test, or it passed vacuously.
        if discriminating < 20 || wrong > 0 {
            caught.push(3);
        }

        // 4. THE TEAM TRAP. Under X330 a RED-team object reads the cosmic branch, so the same
        //    sequence means a different multiplier. Inverting team-blind lands a red-team object
        //    on a wildly wrong size -- so the same target must give DIFFERENT sequences for the
        //    two teams, and each must decode back under ITS OWN team.
        let target = 4.0f32;
        let (sn, gn) = inv(target, conv, ScaleTeam::None);
        let (sr, gr) = inv(target, conv, ScaleTeam::Red);
        let back_n = spawn_seq_to_scale(sn as i32, conv, ScaleTeam::None);
        let back_r = spawn_seq_to_scale(sr as i32, conv, ScaleTeam::Red);
        if (back_n - gn).abs() > 1e-4 || (back_r - gr).abs() > 1e-4 {
            caught.push(4);
        }

        // 5. Positive content: the inverse must actually reach a range worth having, or the
        //    feature is pointless. 4x must be representable to within 10% on the broad
        //    convention -- an assertion that FAILS if someone narrows the search range.
        let (_, g4) = inv(4.0, conv, ScaleTeam::None);
        if (g4 / 4.0 - 1.0).abs() > 0.10 {
            caught.push(5);
        }
    }

    #[test]
    fn the_scale_inverse_is_nearest_monotone_and_team_aware() {
        let mut caught = Vec::new();
        battery(&scale_to_spawn_seq_for_team, &mut caught);
        assert!(caught.is_empty(), "the real inverse failed assertions {caught:?}");
    }

    /// Each saboteur must be caught by the assertion it targets and no other, so no assertion is
    /// carried by a neighbour. A green saboteur here means an assertion proves nothing.
    #[test]
    fn the_inverse_battery_catches_broken_variants() {
        // Compares with ABSOLUTE error instead of log error -> picks a far-off huge step.
        let absolute = |t: f32, c: ScaleConvention, m: ScaleTeam| {
            let mut best = (0i8, 1.0f32, f32::MAX);
            for sq in -100i32..=100 {
                let s = spawn_seq_to_scale(sq, c, m);
                if s <= 0.0 {
                    continue;
                }
                let e = (s - t).abs();
                if e < best.2 {
                    best = (sq as i8, s, e);
                }
            }
            (best.0, best.1)
        };
        // Ignores the team -> the exact bug this function exists to prevent.
        let team_blind =
            |t: f32, c: ScaleConvention, _m: ScaleTeam| scale_to_spawn_seq_for_team(t, c, ScaleTeam::None);
        // Searches only a narrow window -> cannot reach 4x.
        let narrow = |t: f32, c: ScaleConvention, m: ScaleTeam| {
            let tl = t.max(1e-6).ln();
            let mut best = (0i8, 1.0f32, f32::MAX);
            for sq in -5i32..=5 {
                let s = spawn_seq_to_scale(sq, c, m);
                if s <= 0.0 {
                    continue;
                }
                let e = (s.ln() - tl).abs();
                if e < best.2 {
                    best = (sq as i8, s, e);
                }
            }
            (best.0, best.1)
        };

        for (name, f, must) in [
            (
                "absolute error instead of log",
                &absolute as &dyn Fn(f32, ScaleConvention, ScaleTeam) -> (i8, f32),
                3usize,
            ),
            ("team-blind", &team_blind, 4),
            ("narrow search window", &narrow, 5),
        ] {
            let mut caught = Vec::new();
            battery(f, &mut caught);
            assert!(
                caught.contains(&must),
                "saboteur '{name}' must be caught by assertion {must}; it was caught by {caught:?}"
            );
        }
    }
}

#[cfg(test)]
mod scale_use_gates {
    use super::*;

    const TEAMS: [ScaleTeam; 2] = [ScaleTeam::None, ScaleTeam::Red];

    /// Every distinct size a convention can give a team, ascending, with a spawn sequence that gives it.
    fn ladder(c: ScaleConvention, t: ScaleTeam) -> Vec<(f32, i8)> {
        let mut v: Vec<(f32, i8)> = (-100i32..=100).map(|q| (spawn_seq_to_scale(q, c, t), q as i8)).filter(|(s, _)| *s > 0.0).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        v.dedup_by(|a, b| a.0 == b.0);
        v
    }

    /// A typed size lands on the nearest representable one BY RATIO, for every convention and team.
    /// The sweep is the midpoint of EVERY adjacent pair of representable sizes (geometric, since
    /// the quantity is multiplicative), a hair either side: below picks the smaller, above the
    /// larger. An absolute-error chooser agrees with that at tiny gaps and disagrees where the gap
    /// is wide, so the sweep also counts how many pairs separate the two metrics and refuses to
    /// pass if too few did (a fixture that cannot tell them apart proves nothing).
    #[test]
    fn a_typed_size_lands_on_the_nearest_size_by_ratio_for_every_convention_and_team() {
        let (mut pairs, mut discriminating) = (0usize, 0usize);
        for &c in ScaleConvention::ALL {
            for &t in &TEAMS {
                let l = ladder(c, t);
                assert!(l.len() > 50, "{c:?} {t:?}: only {} distinct sizes", l.len());
                for w in l.windows(2) {
                    let (a, b) = (w[0].0, w[1].0);
                    let geo = (a as f64 * b as f64).sqrt();
                    let lo = set_scale((geo * 0.999) as f32, c, t).unwrap();
                    let hi = set_scale((geo * 1.001) as f32, c, t).unwrap();
                    assert_eq!(lo.got, a, "{c:?} {t:?}: just below the geometric midpoint of {a}..{b} must pick {a}, got {}", lo.got);
                    assert_eq!(hi.got, b, "{c:?} {t:?}: just above the geometric midpoint of {a}..{b} must pick {b}, got {}", hi.got);
                    assert_eq!(spawn_seq_to_scale(lo.seq as i32, c, t), lo.got, "the returned byte must decode to the returned size");
                    pairs += 1;
                    if (a as f64 + b as f64) / 2.0 > geo * 1.001 { discriminating += 1; }
                }
            }
        }
        eprintln!("SCALE-USE: {pairs} adjacent pairs swept, {discriminating} where a ratio and an absolute metric disagree");
        assert!(pairs > 1000, "swept only {pairs} pairs");
        assert!(discriminating >= 20, "only {discriminating} pairs tell a ratio chooser from an absolute one");
    }

    /// The convention is not decoration: the same typed size is a different byte under a different
    /// convention, and a RED object differs from a BLUE one under X330 only.
    #[test]
    fn the_convention_and_the_team_change_the_byte() {
        let targets = [0.25f32, 0.5, 2.0, 3.0, 5.0, 10.0, 25.0];
        let mut differ = 0;
        for &x in &targets {
            let a = set_scale(x, ScaleConvention::X47, ScaleTeam::None).unwrap();
            let b = set_scale(x, ScaleConvention::X330, ScaleTeam::None).unwrap();
            if a.seq != b.seq { differ += 1; }
        }
        assert!(differ >= 4, "X47 and X330 agreed on {} of {} sizes", targets.len() - differ, targets.len());
        let mut red_differs = 0;
        for &x in &[2.0f32, 10.0, 100.0, 1000.0] {
            let n = set_scale(x, ScaleConvention::X330, ScaleTeam::None).unwrap();
            let r = set_scale(x, ScaleConvention::X330, ScaleTeam::Red).unwrap();
            if n.seq != r.seq { red_differs += 1; }
            // the OTHER conventions ignore the team entirely
            assert_eq!(set_scale(x, ScaleConvention::X47, ScaleTeam::None), set_scale(x, ScaleConvention::X47, ScaleTeam::Red));
        }
        assert!(red_differs >= 2, "a RED object got the same X330 byte as a BLUE one for {} of 4 sizes", 4 - red_differs);
    }

    /// Out of reach lands on the end of the range and says so; in reach does not; nonsense refuses.
    #[test]
    fn a_size_outside_the_range_clamps_and_says_so_and_nonsense_is_refused() {
        for &c in ScaleConvention::ALL {
            let (lo, hi) = representable_range(c, ScaleTeam::None);
            assert!(lo < 1.0 && hi > 1.0, "{c:?} range {lo}..{hi} must straddle 1");
            let big = set_scale(hi * 10.0, c, ScaleTeam::None).unwrap();
            assert!(big.clamped && big.got == hi, "{c:?}: {big:?}");
            let small = set_scale(lo / 10.0, c, ScaleTeam::None).unwrap();
            assert!(small.clamped && small.got == lo, "{c:?}: {small:?}");
            let mid = set_scale(1.0, c, ScaleTeam::None).unwrap();
            assert!(!mid.clamped && (mid.got - 1.0).abs() < 0.2, "{c:?}: x1 -> {mid:?}");
        }
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(set_scale(bad, ScaleConvention::X330, ScaleTeam::None), None, "{bad}");
        }
    }

    /// The `scale` label is reused when present (any case), appended when absent, and refused when
    /// the table is full, which must not silently write a size the gametype would read as an order.
    #[test]
    fn the_scale_label_is_reused_appended_or_refused() {
        let mut t = vec!["red_spawn".to_string(), "SCALE".to_string()];
        assert_eq!(ensure_scale_label(&mut t), Some(1));
        assert_eq!(t.len(), 2, "an existing label must not be duplicated");
        let mut t = vec!["a".to_string(), "b".to_string()];
        assert_eq!(ensure_scale_label(&mut t), Some(2));
        assert_eq!(t[2], "scale");
        let mut full: Vec<String> = (0..MAX_LABELS).map(|i| format!("l{i}")).collect();
        assert_eq!(ensure_scale_label(&mut full), None);
        assert_eq!(full.len(), MAX_LABELS, "a refused request must not grow the table");
        let mut full_with: Vec<String> = (0..MAX_LABELS).map(|i| if i == 300 { "scale".to_string() } else { format!("l{i}") }).collect();
        assert_eq!(ensure_scale_label(&mut full_with), Some(300), "a full table that already HAS the label still works");
    }

    /// The convention you build for survives a restart; an absent or garbled file is X330.
    #[test]
    fn the_convention_is_remembered() {
        let dir = std::env::temp_dir().join(format!("hms-scale-conv-{}", std::process::id()));
        assert_eq!(load_convention_from_dir(&dir), ScaleConvention::X330, "no file: X330");
        for &c in ScaleConvention::ALL {
            save_convention_to_dir(&dir, c);
            assert_eq!(load_convention_from_dir(&dir), c, "{c:?} did not round trip");
        }
        std::fs::write(dir.join("scale_convention.txt"), "nonsense").unwrap();
        assert_eq!(load_convention_from_dir(&dir), ScaleConvention::X330, "garbled: X330");
        std::fs::write(dir.join("scale_convention.txt"), " 47X \n").unwrap();
        assert_eq!(load_convention_from_dir(&dir), ScaleConvention::X47, "the short name and stray whitespace are accepted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
