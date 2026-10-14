//! Settings ▸ Graphics — the user-facing render-quality settings: the preset ladder, the
//! individual knobs behind it, and their persistence in `project::settings_dir()`.
//!
//! Three rules this file exists to enforce:
//!
//! 1. **The default is exactly today's quality.** `GfxSettings::default()` is the `High` preset,
//!    which projects to `hms_render::quality::RenderQuality::default()` — i.e. the renderer the
//!    app shipped with. A user who never opens the page sees no change at all.
//! 2. **Opening the page changes nothing.** Every widget reads this struct and only a `.changed()`
//!    push applies. The Lighting panel once pushed its own *seed* values into the renderer on
//!    first draw and silently replaced the engine-accurate Halo 4 exposure, filmic curve and
//!    grading LUT (`// #h4-expo-3` in `main.rs`); a preset ladder is exactly the kind of thing
//!    that re-commits that mistake, so presets are applied on CLICK and never on draw.
//! 3. **Picking a preset does not lock the knobs.** A preset writes a group of values; the user
//!    can then move any one of them and the preset label becomes `Custom`.
//!
//! Scope note: `vsync` and `anisotropy` are read at startup (the swapchain present mode and the
//! shared material sampler are both fixed when they are created), so the UI says so instead of
//! pretending otherwise. `texture_max_dim` applies to the next map load for the same reason —
//! the texture is already in a bind group by the time the page is open.

use hms_render::quality::{RenderQuality, ShadowQuality};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Preset {
    /// Flat / white mesh: the cheapest usable view.
    Flat,
    Low,
    Medium,
    /// Today's shipped quality — the default.
    #[default]
    High,
    /// Today's picture, nothing held back.
    Ultra,
    /// The user has moved an individual knob.
    Custom,
}

impl Preset {
    pub const ALL: [Preset; 5] = [Preset::Flat, Preset::Low, Preset::Medium, Preset::High, Preset::Ultra];
    pub fn label(self) -> &'static str {
        match self {
            Preset::Flat => "Flat / white mesh",
            Preset::Low => "Low",
            Preset::Medium => "Medium",
            Preset::High => "High (default)",
            Preset::Ultra => "Ultra",
            Preset::Custom => "Custom",
        }
    }
    pub fn hint(self) -> &'static str {
        match self {
            Preset::Flat => "Unlit white geometry with outlines. No textures, shadows, post or foliage — the cheapest view you can still build in.",
            Preset::Low => "75% render scale, no shadows, no bloom or grading, no foliage, decals or particles, 1024-pixel textures. For weak or integrated GPUs.",
            Preset::Medium => "100% render scale, 1024-pixel shadows, bloom and grading kept, thinned foliage, 2048-pixel textures.",
            Preset::High => "The full picture: 100% render scale, 2048-pixel shadows, the whole post chain, uncapped textures. Raise Render scale to 200% for a supersampled screenshot.",
            Preset::Ultra => "The same picture as High with nothing held back: no frame-rate cap and no idle throttle.",
            Preset::Custom => "Your own mix of the knobs below.",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadowLevel {
    Off,
    Low,
    Medium,
    High,
}

impl ShadowLevel {
    pub const ALL: [ShadowLevel; 4] = [ShadowLevel::Off, ShadowLevel::Low, ShadowLevel::Medium, ShadowLevel::High];
    pub fn label(self) -> &'static str {
        match self {
            ShadowLevel::Off => "Off",
            ShadowLevel::Low => "Low (512)",
            ShadowLevel::Medium => "Medium (1024)",
            ShadowLevel::High => "High (2048)",
        }
    }
    fn to_render(self) -> ShadowQuality {
        match self {
            ShadowLevel::Off => ShadowQuality::Off,
            ShadowLevel::Low => ShadowQuality::Low,
            ShadowLevel::Medium => ShadowQuality::Medium,
            ShadowLevel::High => ShadowQuality::High,
        }
    }
}

/// Render-scale choices offered in the UI, as a percentage of the viewport. 100% is the shipped
/// default; 200% is true 2× supersampling (four times the pixels) and is kept for screenshots and
/// calibration work rather than for everyday use -- see `GfxSettings::of`.
pub const RENDER_SCALES: [(f32, &str); 6] = [
    (0.5, "50%"),
    (0.75, "75%"),
    (1.0, "100% (default)"),
    (1.25, "125%"),
    (1.5, "150%"),
    (2.0, "200% (2× supersampling)"),
];

/// Texture-size caps offered in the UI. 0 = no cap.
pub const TEXTURE_CAPS: [(u32, &str); 4] = [(0, "Full (default)"), (2048, "2048"), (1024, "1024"), (512, "512")];

pub const ANISO_LEVELS: [(u8, &str); 4] = [(16, "16× (default)"), (8, "8×"), (4, "4×"), (1, "Off")];

/// Foliage density choices: a decorator INSTANCE budget, drawn nearest-first, which is
/// simultaneously the density and the draw distance (the furthest instances are the dropped
/// ones). 0 = every instance, the shipped default.
pub const FOLIAGE_BUDGETS: [(u32, &str); 5] =
    [(0, "Full (default)"), (60_000, "High"), (25_000, "Medium"), (8_000, "Low"), (1, "Minimal")];

/// Particle simulation rate: 0 = off, else "simulate every Nth frame". The CPU particle
/// simulation is rebuilt from scratch every frame for the whole map, so dividing its rate is the
/// one knob here that saves CPU rather than GPU.
pub const PARTICLE_RATES: [(u8, &str); 4] = [(1, "Full (default)"), (2, "Half rate"), (4, "Quarter rate"), (0, "Off")];

pub const FPS_CAPS: [(u32, &str); 6] =
    [(0, "Unlimited"), (30, "30"), (60, "60 (default)"), (75, "75"), (120, "120"), (144, "144")];

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct GfxSettings {
    pub preset: Preset,

    // ---- geometry / resolution ----
    pub render_scale: f32,
    pub flat_mode: bool,

    // ---- lighting ----
    pub shadows: ShadowLevel,
    /// Real-time probe GI (already existed as a Lighting-Lab toggle; folded into the ladder).
    pub rtgi: bool,

    // ---- post ----
    pub bloom: bool,
    pub screen_fx: bool,
    pub color_grading: bool,

    // ---- textures ----
    /// 0 = no cap. Applies to the next map load.
    pub texture_max_dim: u32,
    /// Applies after a restart (the shared material sampler is created once).
    pub anisotropy: u8,

    // ---- scene content ----
    pub foliage: bool,
    /// Decorator instance budget; 0 = all.
    pub foliage_budget: u32,
    pub decals: bool,
    /// 0 = off, else simulate the particle systems every Nth frame.
    pub particle_rate: u8,
    pub lens_flares: bool,

    // ---- frame pacing ----
    /// 0 = unlimited.
    pub fps_cap: u32,
    /// Applies after a restart (the swapchain present mode is fixed at creation).
    pub vsync: bool,
    /// Stop redrawing a scene that is not changing.
    pub idle_throttle: bool,
    /// #bg-throttle  Draw far less often while the window is not the one being used.
    ///
    /// "We essentially want to make it not a pain to run it alongside MCC." Without this HMS
    /// holds its full frame rate while its window is unfocused, occluded or minimised -- the
    /// whole `update()`, the whole offscreen 3D render and a command submission of several
    /// thousand draws, sixty times a second, for a window nobody is looking at. Alt-tab to the
    /// game and the editor is still doing all of it on the core the game wants.
    ///
    /// Playback is NOT paused: a film's clock advances on real elapsed time, so a clip keeps
    /// playing and keeps its timing, it is simply drawn less often.
    pub background_throttle: bool,
}

/// #bg-throttle  How often the viewport may be redrawn while the window is NOT in focus.
///
/// 10 Hz: still visibly alive on a second monitor, a twelfth of the work at 120 Hz. Chosen to be
/// a rate rather than a stop so nothing ever looks frozen.
pub const BG_UNFOCUSED_HZ: u32 = 10;
/// #bg-throttle  ...and while it is minimised or fully occluded, where nothing can be seen at all.
pub const BG_HIDDEN_HZ: u32 = 2;

impl Default for GfxSettings {
    fn default() -> Self {
        Self::of(Preset::High)
    }
}

impl GfxSettings {
    /// The knob group behind one preset. `High` is today's renderer exactly.
    pub fn of(preset: Preset) -> Self {
        // Start from today's values, then take away.
        let base = Self {
            preset,
            // #render-scale  100%, not the 200% this shipped with. HMS rendered at TRUE 2x
            // SUPERSAMPLING -- four times the pixels -- by default, which measured 10.7 ms of GPU
            // per frame (9.9 ms of it the scene pass), about two thirds of a high-end card at 60
            // fps. Nobody chose that; it was simply the default, and it is a lot to ask while a
            // game is running. 200% is still offered for screenshots and calibration work.
            //
            // Resolution ONLY: render scale does not touch the calibrated lighting, exposure or
            // tonemap laws, so the sole difference is edge aliasing.
            render_scale: 1.0,
            flat_mode: false,
            shadows: ShadowLevel::High,
            rtgi: false,
            bloom: true,
            screen_fx: true,
            color_grading: true,
            texture_max_dim: 0,
            anisotropy: 16,
            foliage: true,
            foliage_budget: 0,
            decals: true,
            particle_rate: 1,
            lens_flares: true,
            fps_cap: 60,
            vsync: true,
            idle_throttle: true,
            // #bg-throttle  ON in every preset INCLUDING Ultra. The other pacing knobs trade
            // smoothness for heat and Ultra turns them off; this one only ever affects a window
            // the user is not looking at, so there is nothing for Ultra to win by spending it.
            background_throttle: true,
        };
        match preset {
            // High == what the app shipped with, plus the two frame-pacing additions (a 60 fps
            // cadence the app already ran at via `request_repaint_after(16ms)`, and an idle
            // throttle that only ever REMOVES redundant redraws of an unchanged picture).
            Preset::High | Preset::Custom => base,
            // Ultra: the identical picture, nothing held back on pacing.
            Preset::Ultra => Self { fps_cap: 0, idle_throttle: false, ..base },
            Preset::Medium => Self {
                render_scale: 1.0,
                shadows: ShadowLevel::Medium,
                texture_max_dim: 2048,
                anisotropy: 8,
                foliage_budget: 25_000,
                particle_rate: 2,
                ..base
            },
            Preset::Low => Self {
                render_scale: 0.75,
                shadows: ShadowLevel::Off,
                bloom: false,
                screen_fx: false,
                color_grading: false,
                texture_max_dim: 1024,
                anisotropy: 4,
                foliage: false,
                foliage_budget: 1,
                decals: false,
                particle_rate: 0,
                lens_flares: false,
                ..base
            },
            Preset::Flat => Self {
                flat_mode: true,
                render_scale: 1.0,
                shadows: ShadowLevel::Off,
                rtgi: false,
                bloom: false,
                screen_fx: false,
                color_grading: false,
                texture_max_dim: 512,
                anisotropy: 1,
                foliage: false,
                foliage_budget: 1,
                decals: false,
                particle_rate: 0,
                lens_flares: false,
                ..base
            },
        }
    }

    /// The renderer's view of these settings.
    pub fn render_quality(&self) -> RenderQuality {
        RenderQuality {
            scene_scale: self.render_scale.clamp(0.25, 4.0),
            shadows: if self.flat_mode { ShadowQuality::Off } else { self.shadows.to_render() },
            bloom: self.bloom && !self.flat_mode,
            color_grading: self.color_grading && !self.flat_mode,
            flat: self.flat_mode,
            foliage: self.foliage && !self.flat_mode,
            deco_budget: if self.flat_mode { 1 } else { self.foliage_budget },
            decals: self.decals && !self.flat_mode,
            particles: self.particle_rate > 0 && !self.flat_mode,
            lens_flares: self.lens_flares && !self.flat_mode,
        }
    }

    /// Should the CPU particle simulation run on this frame counter? Also false in flat mode,
    /// where nothing draws the result.
    pub fn simulate_particles(&self, frame: u64) -> bool {
        if self.flat_mode || self.particle_rate == 0 {
            return false;
        }
        frame % self.particle_rate.max(1) as u64 == 0
    }

    /// Minimum seconds between frames for the frame-rate cap (`None` = uncapped).
    pub fn frame_interval(&self) -> Option<f32> {
        (self.fps_cap > 0).then(|| 1.0 / self.fps_cap as f32)
    }

    /// #bg-throttle  The longest a frame may be deferred for, given the window's state.
    ///
    /// `None` means "do not throttle" and is the answer whenever the window is FOCUSED or the
    /// setting is off -- so the foreground path cannot be affected by this feature at all, which
    /// is the property that makes it safe to leave on by default.
    ///
    /// `hidden` covers minimised AND fully occluded: nothing can be seen either way, so there is
    /// nothing to draw for.
    pub fn background_budget(&self, focused: bool, hidden: bool) -> Option<std::time::Duration> {
        if !self.background_throttle || focused {
            return None;
        }
        let hz = if hidden { BG_HIDDEN_HZ } else { BG_UNFOCUSED_HZ };
        Some(std::time::Duration::from_secs_f32(1.0 / hz as f32))
    }

    /// Which preset (if any) this exact knob set corresponds to. Used to re-label the dropdown
    /// after a manual edit, and to show the right selection on load.
    pub fn matched_preset(&self) -> Preset {
        for p in Preset::ALL {
            let mut c = Self::of(p);
            c.preset = self.preset;
            if c == *self {
                return p;
            }
        }
        Preset::Custom
    }

    fn path() -> Option<std::path::PathBuf> {
        crate::project::settings_dir().map(|d| d.join("graphics.json"))
    }

    pub fn load() -> Self {
        let Some(p) = Self::path() else { return Self::default() };
        match std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str::<Self>(&t).ok()) {
            Some(s) => s.sanitised(),
            // No file (or an unreadable one) must mean "the shipped quality", never a guess.
            None => Self::default(),
        }
    }

    pub fn save(&self) -> anyhow::Result<std::path::PathBuf> {
        let p = Self::path().ok_or_else(|| {
            anyhow::anyhow!("cannot locate a settings directory (no LOCALAPPDATA, XDG_DATA_HOME or HOME)")
        })?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(self)?)?;
        Ok(p)
    }

    /// Clamp anything a hand-edited (or future/older) file could hold into a legal range, so a
    /// bad value degrades to a usable setting instead of a broken or enormous render target.
    pub fn sanitised(mut self) -> Self {
        if !self.render_scale.is_finite() {
            self.render_scale = 1.0; // #render-scale  the shipped default, see `GfxSettings::of`
        }
        self.render_scale = self.render_scale.clamp(0.25, 4.0);
        self.anisotropy = self.anisotropy.clamp(1, 16);
        self.fps_cap = if self.fps_cap == 0 { 0 } else { self.fps_cap.clamp(10, 1000) };
        self.particle_rate = self.particle_rate.min(16);
        if self.texture_max_dim != 0 {
            self.texture_max_dim = self.texture_max_dim.clamp(64, 16384);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An untouched install renders exactly what it did before the Graphics page existed, with
    /// **one deliberate exception: the render scale**.
    ///
    /// #render-scale  The app used to ship at 200% -- true 2x supersampling, four times the
    /// pixels, measured at 10.7 ms of GPU a frame -- and now ships at 100%. Everything else about
    /// the default picture is unchanged, and this gate is written to prove exactly that: the
    /// quality struct must differ from the renderer's own bare default **in `scene_scale` and in
    /// nothing else**.
    ///
    /// The renderer's `RenderQuality::default()` (`SS_DEFAULT`) is deliberately LEFT at 2.0,
    /// because a headless capture never goes through `GfxSettings` -- `SceneRenderer::new` takes
    /// that bare default -- and every calibration capture and byte-identical screenshot gate in
    /// this project was taken supersampled. Changing it would silently move every reference
    /// image. If someone later "unifies" the two numbers, this test says why they are not one.
    #[test]
    fn default_projects_to_the_shipped_renderer() {
        let s = GfxSettings::default();
        assert_eq!(s.preset, Preset::High);
        let want = RenderQuality { scene_scale: 1.0, ..RenderQuality::default() };
        assert_eq!(s.render_quality(), want);
        // ...and so does an absent/garbage settings file.
        assert_eq!(serde_json::from_str::<GfxSettings>("{}").unwrap().render_quality(), want);
        // The ONLY difference from the renderer's bare default is the scale: every other knob of
        // the shipped picture is still what it always was.
        let bare = RenderQuality::default();
        assert_ne!(want.scene_scale, bare.scene_scale, "this gate is vacuous if they agree");
        assert_eq!(RenderQuality { scene_scale: bare.scene_scale, ..want }, bare,
            "nothing but the render scale may differ from the shipped renderer");
    }

    #[test]
    fn ultra_is_the_same_picture_as_high() {
        let hi = GfxSettings::of(Preset::High);
        let ul = GfxSettings::of(Preset::Ultra);
        assert_eq!(hi.render_quality(), ul.render_quality(), "Ultra must not change the image");
        assert_eq!(hi.screen_fx, ul.screen_fx);
        assert_eq!(hi.texture_max_dim, ul.texture_max_dim);
        assert_eq!(hi.anisotropy, ul.anisotropy);
        // the difference is pacing only
        assert_eq!(ul.fps_cap, 0);
        assert!(!ul.idle_throttle);
    }

    /// Each rung must be cheaper than the one above on the knobs that actually cost GPU time.
    #[test]
    fn the_ladder_descends() {
        let q: Vec<RenderQuality> =
            [Preset::Ultra, Preset::High, Preset::Medium, Preset::Low, Preset::Flat].iter().map(|p| GfxSettings::of(*p).render_quality()).collect();
        // Render scale never increases going down the GRADED rungs. Flat is excluded on purpose:
        // it is a different VIEW, not a lower rung — with no textures, lighting or post left to
        // resolve it is cheaper at 100% than Low is at 75%, and dropping below native would only
        // make the outlines ragged for no measurable saving (GPU: Low 1.51 ms, Flat 0.29 ms).
        for w in q[..4].windows(2) {
            assert!(w[1].scene_scale <= w[0].scene_scale, "render scale went UP: {:?}", w);
        }
        assert_eq!(q[4].scene_scale, 1.0, "the flat view renders at native resolution");
        // shadows: High, High, Medium, Off, Off
        assert_eq!(q[0].shadows.map_size(), Some(2048));
        assert_eq!(q[2].shadows.map_size(), Some(1024));
        assert_eq!(q[3].shadows, ShadowQuality::Off);
        assert_eq!(q[4].shadows, ShadowQuality::Off);
        // Low and Flat must switch off the expensive content lists
        for k in [3, 4] {
            assert!(!q[k].bloom && !q[k].decals && !q[k].particles && !q[k].lens_flares && !q[k].foliage, "preset {k} still draws something expensive");
        }
        // only Flat is flat, and it forces everything off regardless of the other fields
        assert!(q[4].flat);
        assert!(!q[0].flat && !q[1].flat && !q[2].flat && !q[3].flat);
    }

    /// Flat mode must override the image knobs even if the user had them on (it is a *view*, not
    /// a preset, so it can be ticked on top of High).
    #[test]
    fn flat_mode_overrides_everything_expensive() {
        let mut s = GfxSettings::of(Preset::High);
        s.flat_mode = true;
        let q = s.render_quality();
        assert!(q.flat);
        assert!(!q.bloom && !q.color_grading && !q.decals && !q.particles && !q.lens_flares && !q.foliage);
        assert_eq!(q.shadows, ShadowQuality::Off);
        assert!(!s.simulate_particles(0), "flat mode must not pay for the CPU particle simulation");
    }

    #[test]
    fn particle_rate_divides_the_simulation() {
        let mut s = GfxSettings::default();
        s.particle_rate = 1;
        assert!((0..8).all(|f| s.simulate_particles(f)));
        s.particle_rate = 2;
        assert_eq!((0..8).filter(|f| s.simulate_particles(*f)).count(), 4);
        s.particle_rate = 4;
        assert_eq!((0..8).filter(|f| s.simulate_particles(*f)).count(), 2);
        s.particle_rate = 0;
        assert!(!(0..8).any(|f| s.simulate_particles(f)));
    }

    /// #bg-throttle  THE GATE on the background budget table. It is the whole of the feature's
    /// decision, so it is asserted directly rather than inferred from a frame count:
    ///
    /// * focused -> `None`, meaning "do not throttle at all". A foreground frame must be
    ///   untouched by this feature; anything else is a regression the user would feel instantly.
    /// * unfocused but visible -> the 10 Hz budget.
    /// * minimised or occluded -> the slower 2 Hz budget, which must be SLOWER than unfocused.
    /// * the setting off -> `None` in every state, because a switched-off feature has to be
    ///   switched off and not merely gentler.
    ///
    /// Sabotage check: returning `None` for the unfocused case, or making the hidden budget
    /// faster than the unfocused one, each makes this fail.
    #[test]
    fn the_background_budget_depends_only_on_window_state_and_the_setting() {
        let on = GfxSettings { background_throttle: true, ..GfxSettings::default() };
        let off = GfxSettings { background_throttle: false, ..GfxSettings::default() };
        assert_eq!(on.background_budget(true, false), None, "a focused window is never throttled");
        assert_eq!(on.background_budget(true, true), None, "focused wins over minimised");
        let unfocused = on.background_budget(false, false).expect("unfocused is throttled");
        let hidden = on.background_budget(false, true).expect("minimised is throttled");
        assert!(hidden > unfocused, "minimised must be SLOWER than merely unfocused: {hidden:?} vs {unfocused:?}");
        assert!(
            (unfocused.as_secs_f32() - 1.0 / BG_UNFOCUSED_HZ as f32).abs() < 1e-6,
            "the unfocused budget must be 1/{BG_UNFOCUSED_HZ} s, got {unfocused:?}"
        );
        assert!(
            (hidden.as_secs_f32() - 1.0 / BG_HIDDEN_HZ as f32).abs() < 1e-6,
            "the hidden budget must be 1/{BG_HIDDEN_HZ} s, got {hidden:?}"
        );
        for (f, h) in [(true, true), (true, false), (false, false), (false, true)] {
            assert_eq!(off.background_budget(f, h), None, "switched off must mean off (focused={f} hidden={h})");
        }
        // ...and it must survive a round trip through the settings file, or it would silently
        // revert for every user who already has one.
        let json = serde_json::to_string(&off).unwrap();
        let back: GfxSettings = serde_json::from_str(&json).unwrap();
        assert!(!back.background_throttle);
        // An OLD settings file has no such key at all; `#[serde(default)]` must give it the
        // shipped value rather than `false`.
        let old = r#"{"preset":"High","fps_cap":60}"#;
        let s: GfxSettings = serde_json::from_str::<GfxSettings>(old).unwrap().sanitised();
        assert!(s.background_throttle, "an existing settings file must inherit the throttle");
    }

    /// #render-scale  THE GATE on the shipped render scale. HMS rendered at 200% -- true 2x
    /// supersampling, four times the pixels -- for its whole life, which measured 10.7 ms of GPU
    /// a frame and is most of why it was painful to run next to a game. The default is now 100%.
    ///
    /// A default is exactly the kind of value a later edit "helpfully" restores, so it is pinned
    /// here, along with the two things that are easy to get wrong about changing one:
    ///
    /// * a settings file with NO `render_scale` key must inherit 100%, not the old 200%;
    /// * an EXISTING file that says 2.0 must be left alone -- a saved 200% is indistinguishable
    ///   from a user who chose it deliberately, and silently downgrading someone's picture is
    ///   worse than leaving it. Only a fresh install changes.
    ///
    /// Sabotage check: setting the default back to 2.0 makes this fail on the first assertion.
    #[test]
    fn the_shipped_render_scale_is_native_and_an_existing_choice_is_left_alone() {
        assert_eq!(GfxSettings::default().render_scale, 1.0, "a fresh install renders at 100%");
        assert_eq!(GfxSettings::of(Preset::High).render_scale, 1.0);
        // 200% must still be OFFERED -- it is wanted for screenshots and calibration.
        assert!(RENDER_SCALES.iter().any(|(v, _)| *v == 2.0), "200% must remain available");
        assert!(RENDER_SCALES.iter().any(|(v, _)| *v == 1.0));
        // A file with no key at all inherits the shipped value through `#[serde(default)]`.
        let bare: GfxSettings = serde_json::from_str::<GfxSettings>(r#"{"preset":"High"}"#).unwrap().sanitised();
        assert_eq!(bare.render_scale, 1.0, "a settings file with no render_scale must inherit 100%");
        // An existing explicit 200% survives a load untouched: no migration.
        let chosen: GfxSettings =
            serde_json::from_str::<GfxSettings>(r#"{"preset":"High","render_scale":2.0}"#).unwrap().sanitised();
        assert_eq!(chosen.render_scale, 2.0, "an existing 200% must NOT be migrated away");
        // ...and a hostile value falls back to the shipped default rather than the old one.
        //
        // NOT written as `{"render_scale":null}`: serde treats an explicit null as a TYPE ERROR
        // and not as an absent key, so that parse FAILS and an `unwrap` on it panics before any
        // assertion runs -- a fixture that cannot be constructed proves nothing. JSON cannot
        // express NaN at all, so the struct is built directly, which is also the only way to
        // reach the `is_finite` branch this is here to test.
        let mut hostile = GfxSettings::default();
        hostile.render_scale = f32::NAN;
        assert_eq!(hostile.sanitised().render_scale, 1.0, "NaN must fall back to the shipped 100%");
        hostile.render_scale = f32::INFINITY;
        assert_eq!(hostile.sanitised().render_scale, 1.0);
        // A finite but absurd value is CLAMPED rather than reset, which is the existing contract.
        let huge: GfxSettings = serde_json::from_str::<GfxSettings>(r#"{"render_scale":1e30}"#).unwrap().sanitised();
        assert_eq!(huge.render_scale, 4.0);
    }

    #[test]
    fn frame_interval_follows_the_cap() {
        let mut s = GfxSettings::default();
        s.fps_cap = 60;
        assert!((s.frame_interval().unwrap() - 1.0 / 60.0).abs() < 1e-6);
        s.fps_cap = 0;
        assert!(s.frame_interval().is_none());
    }

    /// A hand-edited or corrupt file must not be able to produce an illegal render target, a
    /// zero frame interval or an out-of-range anisotropy.
    #[test]
    fn sanitise_clamps_hostile_values() {
        let bad = r#"{"render_scale": 1e30, "anisotropy": 250, "fps_cap": 3, "texture_max_dim": 7, "particle_rate": 99}"#;
        let s: GfxSettings = serde_json::from_str::<GfxSettings>(bad).unwrap().sanitised();
        assert_eq!(s.render_scale, 4.0);
        assert_eq!(s.anisotropy, 16);
        assert_eq!(s.fps_cap, 10);
        assert_eq!(s.texture_max_dim, 64);
        assert_eq!(s.particle_rate, 16);
        // NaN must not reach the render target maths
        let nan = r#"{"render_scale": null}"#;
        let _ = serde_json::from_str::<GfxSettings>(nan); // null is not an f32; the default is used
        let mut s = GfxSettings::default();
        s.render_scale = f32::NAN;
        // #render-scale  the SHIPPED default, which is now 100% and not the old 200%
        assert_eq!(s.sanitised().render_scale, 1.0);
    }

    /// A settings file written by an older build (fewer keys) must load, and the missing keys
    /// must come back as today's quality rather than zeros.
    #[test]
    fn partial_files_fill_in_todays_quality() {
        let s: GfxSettings = serde_json::from_str(r#"{"render_scale": 1.0}"#).unwrap();
        assert_eq!(s.render_scale, 1.0);
        assert_eq!(s.shadows, ShadowLevel::High);
        assert!(s.bloom && s.color_grading && s.decals);
        assert_eq!(s.texture_max_dim, 0);
    }

    #[test]
    fn preset_round_trips_and_an_edit_becomes_custom() {
        for p in Preset::ALL {
            assert_eq!(GfxSettings::of(p).matched_preset(), p, "{p:?} did not match itself");
        }
        let mut s = GfxSettings::of(Preset::High);
        s.bloom = false;
        assert_eq!(s.matched_preset(), Preset::Custom);
    }

    #[test]
    fn settings_round_trip_through_json() {
        let mut s = GfxSettings::of(Preset::Medium);
        s.flat_mode = true;
        s.fps_cap = 144;
        let back: GfxSettings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }

    /// Every option list the UI offers must contain the default the app ships with, otherwise
    /// a user who touches the control can never get back to it.
    #[test]
    fn ui_option_lists_contain_the_defaults() {
        let d = GfxSettings::default();
        assert!(RENDER_SCALES.iter().any(|(v, _)| (*v - d.render_scale).abs() < 1e-6));
        assert!(TEXTURE_CAPS.iter().any(|(v, _)| *v == d.texture_max_dim));
        assert!(ANISO_LEVELS.iter().any(|(v, _)| *v == d.anisotropy));
        assert!(FOLIAGE_BUDGETS.iter().any(|(v, _)| *v == d.foliage_budget));
        assert!(PARTICLE_RATES.iter().any(|(v, _)| *v == d.particle_rate));
        assert!(FPS_CAPS.iter().any(|(v, _)| *v == d.fps_cap));
        assert!(ShadowLevel::ALL.contains(&d.shadows));
    }
}

// ---------------------------------------------------------------------------------------------
// The panel. Returns true when the user moved something, which is the ONLY thing that makes the
// caller push to the renderer and write the file (see rule 2 at the top of this file).
// ---------------------------------------------------------------------------------------------

/// One labelled dropdown over a `(value, label)` table. Returns true when the value changed.
fn combo<T: Copy + PartialEq>(
    ui: &mut eframe::egui::Ui,
    id: &str,
    label: &str,
    cur: &mut T,
    opts: &[(T, &str)],
    hint: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        let shown = opts.iter().find(|(v, _)| *v == *cur).map(|(_, l)| *l).unwrap_or("Custom");
        eframe::egui::ComboBox::from_id_salt(id).selected_text(shown).show_ui(ui, |ui| {
            for (v, l) in opts {
                if ui.selectable_label(*cur == *v, *l).clicked() {
                    *cur = *v;
                    changed = true;
                }
            }
        });
    });
    changed
}

/// Draw Settings ▸ Graphics & performance. `restart_note` is shown for the two knobs that can
/// only take effect on the next start.
pub fn ui(s: &mut GfxSettings, ui: &mut eframe::egui::Ui) -> bool {
    use eframe::egui;
    let mut changed = false;

    ui.label(
        egui::RichText::new("Trade picture quality for speed. High is the default and is the full picture; only Render scale changes how many pixels the scene is drawn at.")
            .small(),
    );
    ui.add_space(4.0);

    // ---- presets ----
    ui.label(egui::RichText::new("PRESET").small().strong());
    let shown = s.matched_preset();
    ui.horizontal_wrapped(|ui| {
        for p in Preset::ALL {
            if ui
                .selectable_label(shown == p, p.label())
                .on_hover_text(p.hint())
                .clicked()
            {
                // A preset is applied on CLICK only — never while the panel merely draws.
                *s = GfxSettings::of(p);
                changed = true;
            }
        }
    });
    if shown == Preset::Custom {
        ui.label(egui::RichText::new("Custom — your own mix of the knobs below.").small().weak());
    } else {
        ui.label(egui::RichText::new(shown.hint()).small().weak());
    }

    ui.add_space(8.0);
    ui.separator();

    // ---- the headline knob ----
    ui.label(egui::RichText::new("VIEW").small().strong());
    changed |= ui
        .checkbox(&mut s.flat_mode, "Flat / white mesh (no textures, no lighting, no post)")
        .on_hover_text(
            "Draws the map as plain white clay with outlined edges. Nothing is textured, lit, \
             shadowed or post-processed, and foliage, decals, particles and water surfaces are \
             dropped. The grid, your selection and the transform gizmo all still work, so you can \
             keep building. This is the cheapest view the renderer has.",
        )
        .changed();
    if s.flat_mode {
        ui.label(
            egui::RichText::new("Flat mode is on — the lighting, post and content knobs below are ignored while it is.")
                .small()
                .weak(),
        );
    }
    changed |= combo(
        ui,
        "gfx-scale",
        "Render scale",
        &mut s.render_scale,
        &RENDER_SCALES,
        "How many pixels the 3D scene is rendered at, as a percentage of the viewport. The user \
         interface always stays at native resolution. 100% is the default. 200% is true \
         supersampling -- FOUR TIMES the pixels, and the single most expensive thing the renderer \
         does (measured: 10.7 ms of graphics-card time a frame, which is a lot to ask while a game \
         is running) -- so it is best kept for a screenshot or for calibration work. Below 100% \
         the scene is rendered smaller and scaled up. It is RESOLUTION ONLY: the lighting, \
         exposure and tonemap are identical at every setting, so the only difference is how \
         jagged an edge looks.",
    );

    ui.add_space(6.0);
    ui.label(egui::RichText::new("LIGHTING").small().strong());
    changed |= combo(
        ui,
        "gfx-shadows",
        "Shadows",
        &mut s.shadows,
        &ShadowLevel::ALL.map(|l| (l, l.label())),
        "Real-time sun shadows cast by objects. Off skips the shadow depth pass entirely; the \
         lower sizes render a smaller shadow map (blockier edges, much less fill).",
    );
    changed |= ui
        .checkbox(&mut s.rtgi, "Real-time probe GI")
        .on_hover_text(
            "Dynamic bounce light from a probe grid updated on the GPU every frame. Off by \
             default; it is the most expensive optional lighting feature.",
        )
        .changed();

    ui.add_space(6.0);
    ui.label(egui::RichText::new("POST-PROCESSING").small().strong());
    changed |= ui
        .checkbox(&mut s.bloom, "Bloom")
        .on_hover_text("The engine bloom chain — eleven render passes. Off replaces all of them with one clear.")
        .changed();
    changed |= ui
        .checkbox(&mut s.screen_fx, "Screen effects (Forge FX colour)")
        .on_hover_text("The Forge special-FX colour matrix and screen brightness in the final pass.")
        .changed();
    changed |= ui
        .checkbox(&mut s.color_grading, "Colour grading LUT")
        .on_hover_text("The map's authored colour-grading volume. Off binds a neutral table instead.")
        .changed();

    ui.add_space(6.0);
    ui.label(egui::RichText::new("TEXTURES").small().strong());
    changed |= combo(
        ui,
        "gfx-texcap",
        "Texture quality",
        &mut s.texture_max_dim,
        &TEXTURE_CAPS,
        "Largest texture size uploaded to the GPU. Lower caps drop the top mip level(s), which \
         cuts video memory and sampling bandwidth. Applies the next time a map is loaded.",
    );
    changed |= combo(
        ui,
        "gfx-aniso",
        "Anisotropic filtering",
        &mut s.anisotropy,
        &ANISO_LEVELS,
        "Sharpness of tiled surfaces viewed at a grazing angle. Applies after restarting \
         (one sampler is shared by every material).",
    );

    ui.add_space(6.0);
    ui.label(egui::RichText::new("SCENE CONTENT").small().strong());
    changed |= ui
        .checkbox(&mut s.foliage, "Foliage / decorators")
        .on_hover_text("Grass, flowers and other decorator instances.")
        .changed();
    ui.add_enabled_ui(s.foliage, |ui| {
        changed |= combo(
            ui,
            "gfx-foliage",
            "Foliage density",
            &mut s.foliage_budget,
            &FOLIAGE_BUDGETS,
            "A cap on how many decorator instances are drawn, nearest first — so it is both the \
             density and the draw distance: the instances that disappear are the far ones.",
        );
    });
    changed |= ui.checkbox(&mut s.decals, "Decals").on_hover_text("Scorches, blood and other projected surface decals.").changed();
    changed |= combo(
        ui,
        "gfx-particles",
        "Particles",
        &mut s.particle_rate,
        &PARTICLE_RATES,
        "Effect particle systems (grav-lift plasma, waterfalls, sparks). These are simulated on \
         the CPU for the whole map every frame, so a lower rate saves processor time as well as \
         drawing time.",
    );
    changed |= ui.checkbox(&mut s.lens_flares, "Lens flares").on_hover_text("The sun glare and placed light flares (each one also costs a depth read).").changed();

    ui.add_space(6.0);
    ui.label(egui::RichText::new("FRAME PACING").small().strong());
    changed |= combo(
        ui,
        "gfx-fps",
        "Frame-rate cap",
        &mut s.fps_cap,
        &FPS_CAPS,
        "Upper limit on how often the viewport is redrawn. A cap is the simplest way to stop the \
         editor heating up a laptop while you think.",
    );
    changed |= ui
        .checkbox(&mut s.vsync, "Vertical sync")
        .on_hover_text("Synchronise presentation to the display. Applies after restarting.")
        .changed();
    changed |= ui
        .checkbox(&mut s.background_throttle, "Throttle when not in focus")
        .on_hover_text(
            "While Halo Map Studio's window is not the one you are using, redraw it a few times \
             a second instead of continuously -- so it stays out of the way of a game running at \
             the same time. A film keeps PLAYING and keeps its timing; it is only drawn less \
             often. Minimised or fully covered, it drops further still. (A script driving the \
             editor over its command port is answered at the same slower rate while the window \
             is in the background -- that is the throttle working, not a hang.)",
        )
        .changed();
    changed |= ui
        .checkbox(&mut s.idle_throttle, "Idle throttle (stop redrawing a still scene)")
        .on_hover_text(
            "When nothing is moving — no camera input, no animation, no load in flight — the \
             viewport stops being redrawn until something changes. It cannot change the picture, \
             only how often an identical picture is produced.",
        )
        .changed();

    ui.add_space(8.0);
    if ui.button("Reset to the default (High)").on_hover_text("Back to exactly what Halo Map Studio ships with.").clicked() {
        *s = GfxSettings::default();
        changed = true;
    }
    if changed {
        *s = s.sanitised();
        if s.matched_preset() == Preset::Custom {
            s.preset = Preset::Custom;
        }
    }
    changed
}
