//! Renderer-side render-quality knobs — the part of Settings ▸ Graphics the renderer itself
//! has to honour (pass gating, target resolution, caster counts, upload caps).
//!
//! **Every default here is exactly what the renderer did before this module existed**, so an
//! untouched install renders byte-identically. That is deliberate: the Settings panel history in
//! this codebase includes a bug (`// #h4-expo-3`) where merely *opening* a panel pushed its own
//! idea of the lighting terms into the renderer and changed the picture. Quality knobs must only
//! ever change anything when the user moves one.
//!
//! The owning (serialisable, user-facing) type is `hms-app`'s `gfxsettings::GfxSettings`; this is
//! the small subset it projects down to the renderer.

use std::sync::atomic::{AtomicU32, Ordering};

/// Sun-shadow quality. The Reach/base sun map is a single camera-centred box; `Off` skips the
/// whole depth pass (and, on Halo 4, the cascade pass too).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadowQuality {
    Off,
    /// 512² sun map, 1-tap cascade.
    Low,
    /// 1024² sun map.
    Medium,
    /// 2048² sun map — the shipped default.
    High,
}

impl ShadowQuality {
    /// Side of the sun shadow map. `None` = the pass is skipped entirely.
    pub fn map_size(self) -> Option<u32> {
        match self {
            ShadowQuality::Off => None,
            ShadowQuality::Low => Some(512),
            ShadowQuality::Medium => Some(1024),
            ShadowQuality::High => Some(2048),
        }
    }
    /// Halo 4 floating-cascade PCF taps (the map's authored value is used at `High`).
    pub fn cascade_taps(self) -> Option<u32> {
        match self {
            ShadowQuality::Off => None,
            ShadowQuality::Low => Some(1),
            ShadowQuality::Medium => Some(4),
            ShadowQuality::High => None, // keep the map's authored tap count
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RenderQuality {
    /// Scene (HDR + depth) resolution as a multiple of the viewport. The shipped renderer is a
    /// hard-coded 2× supersample (true SSAA; see the `SS_DEFAULT` comment this replaced), so 2.0
    /// is the default and anything below it is the "render scale" knob. Values ≥ 1.75 keep the
    /// 4-tap box resolve; below that the resolve is a single bilinear tap.
    pub scene_scale: f32,
    pub shadows: ShadowQuality,
    /// Bloom pyramid (11 passes, unconditional before this flag existed). Off replaces all
    /// eleven with one quarter-resolution clear, so the post pass composites no bloom.
    pub bloom: bool,
    /// The map's 3D colour-grading LUT in the post pass. Off binds the identity LUT.
    pub color_grading: bool,
    /// FLAT / white-mesh view: unlit clay geometry + a depth outline, no textures, no bloom, no
    /// tonemap, no shadows, no foliage, no particles. The cheapest view that still shows layout.
    pub flat: bool,
    /// Decorators / foliage.
    pub foliage: bool,
    /// Decorator instance budget — the engine-style nearest-first cap, which is simultaneously
    /// the foliage density AND its draw distance (the furthest instances are the ones dropped).
    /// 0 = draw everything, the shipped default.
    pub deco_budget: u32,
    /// Scenario + projected decals.
    pub decals: bool,
    /// Effect particle systems (the GPU half; the app skips the CPU simulation too).
    pub particles: bool,
    /// Lens flares — the sun glare and the placed efsc flares, which also costs an occlusion
    /// depth read per flare.
    pub lens_flares: bool,
}

impl Default for RenderQuality {
    fn default() -> Self {
        Self {
            scene_scale: crate::SS_DEFAULT,
            shadows: ShadowQuality::High,
            bloom: true,
            color_grading: true,
            flat: false,
            foliage: true,
            deco_budget: 0,
            decals: true,
            particles: true,
            lens_flares: true,
        }
    }
}

impl RenderQuality {
    /// True when the post resolve must average the 4 supersample centres — the shipped
    /// behaviour. Below 1.75× the box would straddle fewer than 4 real sub-texels, so one
    /// bilinear tap is both cheaper and no blurrier.
    pub fn resolve_taps(&self) -> u32 {
        if self.scene_scale >= 1.75 {
            4
        } else {
            1
        }
    }
    /// Scene target size for a given output size, clamped to something a GPU will accept.
    pub fn scene_target(&self, size: (u32, u32)) -> (u32, u32) {
        let s = self.scene_scale.clamp(0.25, 4.0);
        (
            ((size.0 as f32 * s).round() as u32).max(1),
            ((size.1 as f32 * s).round() as u32).max(1),
        )
    }
}

/// Texture upload cap, in texels per side. 0 = no cap (the shipped default). Read by the DDS/BC
/// and mipped RGBA upload paths in `mesh.rs`: a cap of 512 drops every mip larger than 512², so
/// both the VRAM footprint and the sampling bandwidth fall without touching any bind group.
///
/// A global rather than a parameter because the upload helpers are free functions called from
/// ~40 sites across `scene.rs`, `h3/`, `h4/` and `h2a/`; threading a parameter through all of
/// them to express one user setting would be all risk and no benefit.
static TEX_MAX_DIM: AtomicU32 = AtomicU32::new(0);

pub fn set_texture_max_dim(d: u32) {
    TEX_MAX_DIM.store(d, Ordering::Relaxed);
}

pub fn texture_max_dim() -> u32 {
    TEX_MAX_DIM.load(Ordering::Relaxed)
}

/// Anisotropic-filter clamp for the shared material sampler (1 = off, 16 = the shipped
/// default). Read once, when `MeshRenderer` creates that sampler — every material bind group
/// bakes the sampler in, so this is a startup setting and the UI says so.
static ANISO: AtomicU32 = AtomicU32::new(16);

pub fn set_anisotropy(n: u16) {
    ANISO.store(n.clamp(1, 16) as u32, Ordering::Relaxed);
}

pub fn anisotropy() -> u16 {
    ANISO.load(Ordering::Relaxed).clamp(1, 16) as u16
}

/// How many top mip levels to drop from a `w`×`h` chain of `mips` levels to honour the cap.
/// Never drops the whole chain, and never drops anything when there is no cap or no mip chain
/// to fall back on.
pub fn mip_skip(w: u32, h: u32, mips: u32) -> u32 {
    let cap = texture_max_dim();
    if cap == 0 || mips <= 1 {
        return 0;
    }
    let mut k = 0u32;
    while k + 1 < mips && ((w >> k).max(1) > cap || (h >> k).max(1) > cap) {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default must be the shipped renderer: 2× supersample, 4-tap resolve, 2048² shadows,
    /// every post stage on, nothing distance-culled.
    #[test]
    fn default_is_todays_quality() {
        let q = RenderQuality::default();
        assert_eq!(q.scene_scale, 2.0);
        assert_eq!(q.resolve_taps(), 4);
        assert_eq!(q.shadows.map_size(), Some(2048));
        assert_eq!(q.shadows.cascade_taps(), None, "High must keep the map's authored tap count");
        assert!(q.bloom && q.color_grading && q.foliage && q.decals && q.particles && q.lens_flares);
        assert!(!q.flat);
        assert_eq!(q.deco_budget, 0, "0 = every decorator instance, the shipped behaviour");
        // and the target size is exactly the 2× the const SS used to produce
        assert_eq!(q.scene_target((1600, 900)), (3200, 1800));
    }

    #[test]
    fn scene_target_scales_and_never_degenerates() {
        let mut q = RenderQuality::default();
        q.scene_scale = 1.0;
        assert_eq!(q.scene_target((1600, 900)), (1600, 900));
        assert_eq!(q.resolve_taps(), 1);
        q.scene_scale = 0.5;
        assert_eq!(q.scene_target((1600, 900)), (800, 450));
        // a 1-pixel viewport (a collapsed panel) must still produce a legal texture
        q.scene_scale = 0.25;
        assert_eq!(q.scene_target((1, 1)), (1, 1));
        // out-of-range values are clamped, not trusted
        q.scene_scale = 99.0;
        assert_eq!(q.scene_target((100, 100)), (400, 400));
        q.scene_scale = 0.0;
        assert_eq!(q.scene_target((100, 100)), (25, 25));
    }

    /// 1.75 is the 4-tap boundary: High (2.0) and a hypothetical 1.75 keep true SSAA, 1.5 and
    /// below take the single-tap resolve.
    #[test]
    fn resolve_tap_boundary() {
        let mut q = RenderQuality::default();
        for (s, taps) in [(2.0, 4), (1.75, 4), (1.74, 1), (1.5, 1), (1.0, 1), (0.5, 1)] {
            q.scene_scale = s;
            assert_eq!(q.resolve_taps(), taps, "scale {s}");
        }
    }

    #[test]
    fn mip_skip_honours_the_cap_and_keeps_a_level() {
        set_texture_max_dim(0);
        assert_eq!(mip_skip(2048, 2048, 12), 0, "no cap must never drop a mip");
        set_texture_max_dim(512);
        assert_eq!(mip_skip(2048, 2048, 12), 2, "2048 -> 1024 -> 512");
        assert_eq!(mip_skip(512, 512, 10), 0, "already within the cap");
        assert_eq!(mip_skip(1024, 256, 11), 1, "the LONG side decides");
        // a texture with no mip chain cannot be dropped, cap or not
        assert_eq!(mip_skip(4096, 4096, 1), 0);
        // and a short chain stops before it runs out of levels
        assert_eq!(mip_skip(4096, 4096, 2), 1);
        set_texture_max_dim(0);
    }
}
