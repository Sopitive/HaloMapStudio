//! hms-render — wgpu renderer for the 3D viewport.
//!
//! Renders the scene to an offscreen texture which the app shows inside an egui image,
//! decoupling the 3D renderer from egui's own pass. This module owns the frame: render
//! targets, the camera/light/fog bind group, the shadow, sky, scene, transparent and
//! post-processing (exposure meter, bloom, tonemap) passes. Mesh pipelines and the WGSL
//! material shaders live in `mesh`.

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::{Mat4, Vec3};

mod mesh;
mod bake;
pub mod lightbake_gpu;
pub mod rtgi;
pub mod lensfx;
pub use lensfx::{LensFlareDef, LensFlareInstance, LensFlareReflection};
pub use bake::{BakeLayer, DetailBaker};
pub use mesh::memprof;
pub use mesh::{
    GpuMesh, LightmapInputs, Material, MeshRenderer, MeshVertex, TerrainMaterialParams, WaterParams,
    make_geom_buffers,
    upload_texture_bgra, upload_texture_bgra_gamma_enc, gamma_encode_lut,
    upload_texture_bgra_nomip,
    sky_legacy,
    upload_texture_rgba16f_nomip,
    upload_texture_dds,
    upload_texture_rgba,
    build_authored_env_cube, build_hdr_env_cube,
};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// sRGB view format of COLOR_FORMAT — the view handed to egui so it doesn't
/// double-gamma-encode our already-sqrt-encoded color (see make_targets).
pub const COLOR_FORMAT_SRGB: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Light uniform size: 11 Reach vec4s + 9 Halo 4 shadow vec4s (see `render`'s light block).
pub(crate) const LIGHT_UNIFORM_BYTES: u64 = 320;
/// Sun shadow depth map side (Reach camera-centred box AND the Halo 4 object burn map).
pub const SHADOW_MAP_SIZE: u32 = 2048;
/// Halo 4 cascade depth texture side (the engine's largest resolution option).
pub(crate) const H4_CASCADE_TEX: u32 = 800;
/// The atmosphere fog world-unit bridge baked into `FOG_WGSL` (`const FOG_WU`), used when no
/// `set_fog_wu` override is in force. Mirrored here so a UI can SEED a fog slider from the value
/// that is actually rendering instead of pushing a guess into the renderer (`// #h4-expo-3`);
/// `tests::fog_wu_default_matches_shader` pins the two together.
pub const FOG_WU_DEFAULT: f32 = 0.01;

/// One Halo 4 floating-shadow cascade (scnr structure_bsps +0xBC .. 24-B records, halo4.dll
/// sub_18035F368 / sub_180361EFC): a square orthographic box around the viewer that REPLACES the baked
/// sun visibility with a rotated-poisson PCF of a dynamic depth map (objects + BSP instances inside it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H4CascadeCfg {
    /// Cascade half-width across the light (wu; the box is `2 * half_width` square).
    pub half_width: f32,
    /// Box length along the light (wu), centred on the box centre.
    pub length: f32,
    /// Box centre offset ahead of the viewer along the horizontal forward direction (wu).
    pub offset: f32,
    /// Receiver depth bias in normalized shadow depth (`p.z - bias`).
    pub bias: f32,
    /// Poisson filter width: taps are scaled by `filter / 800` in shadow-map UV.
    pub filter: f32,
    /// Box centre slide toward the sun (wu; "Sun Direction Offset").
    pub sun_offset: f32,
    /// Depth map side: 512 or 800 (scnr Resolution enum).
    pub resolution: u32,
    /// Poisson tap count: 8 (quality 0), 12 (1), 6 (2).
    pub taps: u32,
}

/// HDR intermediate the scene renders into; the post pass tonemaps it down to
/// COLOR_FORMAT (the texture egui displays).
pub const SCENE_HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// A water volume's surface plane + volume XY footprint, for detecting when the camera is
/// submerged. `z` = water surface height (Halo +Z up), `min`/`max` = the volume's XY AABB,
/// `murk` = underwater murkiness, `color` = the underwater fog tint. Set once per map via
/// `set_water_planes`.
///
/// The engine's underwater tint uses the dedicated WaterSharedPS globals
/// `k_ps_water_underwater_fog_color` + `k_ps_water_underwater_murkiness`, distinct from the
/// surface `water_murkiness`. These are NOT in the per-material water rmt2 (the
/// `ZH_BSP_UnderwaterProbe` diagnostic confirmed the rmt2 Arguments carry only surface params);
/// the engine fills them at runtime from the `atmosphere_globals` (`atgf`) tag's
/// `underwater_setting` block (element 0x14 bytes: Name(sid)@+0, Murkiness(real_fraction)@+4,
/// Fog Color(real_rgb float3)@+8 — from reach_tag_test.exe tag defs + the .rdata field-name
/// strings). `NativeDll::bsp_underwater_fog` walks the atgf tag (see `ZH_BSP_GetUnderwaterFog`)
/// and scene.rs feeds the authored `(rgb, murkiness)` here, falling back to the deep-colour /
/// surface `water_murkiness` stand-in only when a map has no atgf underwater setting (e.g.
/// Forge canvases). Fog Color is a linear, pre-exposure colour.
#[derive(Clone, Debug)]
pub struct WaterPlane {
    pub z: f32,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub murk: f32,
    pub color: [f32; 3],
    /// The water surface's triangles (x0,y0,z0, x1,y1,z1, x2,y2,z2). The camera is submerged only
    /// when its XY lies inside one of these AND it is below that triangle's INTERPOLATED surface
    /// height — an AABB test alone false-positives on land that sits below the water level within
    /// the volume's bounding box, and a single `z` = mesh max Z flags any camera below the shore
    /// geometry's top as underwater (wrong fresnel path when flying low).
    pub tris: Vec<[f32; 9]>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnderwaterUniform {
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    fog: [f32; 4], // rgb = fog colour, w = murkiness
}

/// One authored planar fog volume — its plane (normal + `-dot(n,ctr)`), colour, density, and
/// world-space AABB. Set per map via `set_planar_fog_volumes`; the screen-space pass fogs scene
/// pixels below the plane and inside the AABB. Up to 8 are composited per frame.
#[derive(Clone, Copy, Debug)]
pub struct PlanarFogVolume {
    pub plane: [f32; 4],
    pub color: [f32; 3],
    pub density: f32,
    pub bmin: [f32; 3],
    pub bmax: [f32; 3],
    /// Engine base_depth: the authored `full fog depth` (sddt.PlanarFog +0x2C, "fog thickness
    /// below the plane") over which extinction ramps quadratically before going linear.
    /// Surfaced through the native FFI (`ZhPlanarFogVolume.depth`). <= 0 disables the ramp.
    pub base_depth: f32,
}

const MAX_PFOG: usize = 8;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PFogVolGpu {
    plane: [f32; 4],
    color_density: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PlanarFogUniform {
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    count: [f32; 4],
    vols: [PFogVolGpu; MAX_PFOG],
}

/// Fly camera (Halo space: +Z up).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,   // radians, around +Z
    pub pitch: f32, // radians, clamped ±~85°
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Vec3::new(-8.0, -8.0, 4.0),
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: -0.3,
            fov_y: 70f32.to_radians(),
            near: 0.05,
            far: 50_000.0,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cp * cy, cp * sy, sp).normalize()
    }
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Z).normalize()
    }
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.pos, self.pos + self.forward(), Vec3::Z);
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(0.01), self.near, self.far);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

/// Vertex for the translucent "zone" volume pass (boundary shapes): world position + RGBA
/// (alpha carries the fill opacity). A separate format from `Vertex` because the holographic
/// fill needs per-vertex alpha.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ZoneVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    time: [f32; 4],
    // Inverse view-proj — water reconstructs the opaque bed's world position from the
    // sampled scene depth to compute depth-based murkiness. Shaders that don't need it
    // declare the shorter prefix struct.
    inv_view_proj: [[f32; 4]; 4],
}

/// Sky pass uniform: inverse view-proj to reconstruct a per-pixel world ray so
/// the horizon gradient stays anchored to the world as the camera pitches.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkyUniform {
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4], // xyz = per-map key dir (to sun), w unused — sun disc/glow
    // rgb = the map's OWN atmosphere_fog sky inscatter color (not an invented blue);
    // w = has_atmosphere (0 on space maps → the gradient isn't drawn anyway).
    horizon: [f32; 4],
}

pub struct SceneRenderer {
    size: (u32, u32),
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    /// sRGB reinterpretation of color_tex, handed to egui (see make_targets).
    color_view_srgb: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    // Opaque-depth copy (sampleable) for water depth-murkiness. depth_tex is the copy
    // SOURCE; depth_copy_tex/view is bound into the camera group (binding 5).
    depth_tex: wgpu::Texture,
    depth_copy_tex: wgpu::Texture,
    depth_copy_view: wgpu::TextureView,
    // HDR scene target (tonemapped into color_tex by the post pass).
    hdr_view: wgpu::TextureView,
    // Kept so a headless run can read back the LINEAR pre-tonemap scene (HMS_HDR_DUMP; the
    // only reliable probe — post-pass readbacks go through tonemap+auto-exposure).
    hdr_tex: wgpu::Texture,
    post_pipeline: wgpu::RenderPipeline,
    post_bgl: wgpu::BindGroupLayout,
    post_sampler: wgpu::Sampler,
    post_bg: wgpu::BindGroup,
    post_buf: wgpu::Buffer,
    // Auto-exposure luminance meter: 1×1 log-avg-luminance, fed to the post pass.
    lum_view: wgpu::TextureView,
    /// Kept for the HMS_EXPODIAG readback of the metered mean-log-luminance.
    lum_tex: wgpu::Texture,
    /// Non-blocking meter readback ring (see `read_mean_log_async`). Three staging
    /// buffers so a copy can be in flight while an older one is being read; state per slot:
    /// 0 = free, 1 = copy submitted + map pending, 2 = mapped (value ready).
    lum_rb: Vec<(wgpu::Buffer, std::sync::Arc<std::sync::atomic::AtomicU8>)>,
    lum_rb_next: std::cell::Cell<usize>,
    lum_rb_last: std::cell::Cell<Option<f32>>,
    lum_pipeline: wgpu::RenderPipeline,
    /// The Halo 4 meter: `lum_bgl` binds the post uniform + the 48x32 pre-pass
    /// (`h4pre_*`: box-filtered HDR rgb + self-illum coverage) next to the HDR scene.
    lum_bgl: wgpu::BindGroupLayout,
    h4pre_pipeline: wgpu::RenderPipeline,
    h4pre_view: wgpu::TextureView,
    h4pre_bg: wgpu::BindGroup,
    h4_meter_on: std::cell::Cell<bool>,
    /// The post pass's 3D colour-grading LUT (group 1): identity until a map sets one.
    post_lut_bgl: wgpu::BindGroupLayout,
    post_lut_bg: wgpu::BindGroup,
    post_lut_sampler: wgpu::Sampler,
    lum_bg: wgpu::BindGroup,
    // Analytical-sun shadow map (real-time cast shadows).
    shadow_view: wgpu::TextureView,
    /// The Halo 4 floating-shadow cascade depth map view (group 0 binding 14).
    h4_cascade_view: wgpu::TextureView,
    /// The cascade's own light bind group (its view_proj lives in the same light uniform
    /// at the cascade offset; the caster VS reads `light.light_view_proj`, so the cascade pass binds a
    /// second uniform buffer holding the cascade matrix in slot 0).
    h4_cascade_light_buf: wgpu::Buffer,
    h4_cascade_pass_bg: wgpu::BindGroup,
    light_buf: wgpu::Buffer,
    simple_lights_buf: wgpu::Buffer,
    shadow_pass_bg: wgpu::BindGroup,
    // Engine-shaped bloom (HREK postprocess chain): 1/4 curve level -> 8x8-box to 1/16 ->
    // 8x8-box to 1/64 -> blur11 (x intensity) -> add into 1/16 -> blur11 -> add into 1/4 ->
    // blur11 -> composite. Every pass clamps at 2.0 (engine fp16 store).
    tex_bgl: wgpu::BindGroupLayout,
    bright_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    blur_v_i_pipeline: wgpu::RenderPipeline, // vertical blur x bloom_intensity (level 1/64)
    downsample_pipeline: wgpu::RenderPipeline, // 8x8 box (16 bilinear taps at +-1,+-3)
    env_mip_pipeline: wgpu::RenderPipeline,    // tex-only 2x2 box for the env-cube mip chain
    bloom_add_pipeline: wgpu::RenderPipeline, // 1/16: intensity*medium*original + up(add)
    bloom_add_s_pipeline: wgpu::RenderPipeline, // 1/4: intensity*small*original + up(add)
    // The Halo 4 chain (BLOOM_WGSL `*_h4`, selected while `h4_bloom_on`): same pyramid /
    // bind groups, own curve level, kernels, alpha-weighted combine; group 1 = the grading LUT.
    h4_bloom_on: std::cell::Cell<bool>,
    bright_h4_pipeline: wgpu::RenderPipeline,
    box8_h4_pipeline: wgpu::RenderPipeline,
    blur_h_h4_pipeline: wgpu::RenderPipeline,
    blur_v_h4_pipeline: wgpu::RenderPipeline,
    blur_v_i_h4_pipeline: wgpu::RenderPipeline,
    add_m_h4_pipeline: wgpu::RenderPipeline,
    add_s_h4_pipeline: wgpu::RenderPipeline,
    bloom: BloomViews,
    bright_in_bg: wgpu::BindGroup,   // reads HDR scene → q_a
    bg_q_a: wgpu::BindGroup,         // src q_a, add e_b
    bg_q_b: wgpu::BindGroup,         // src q_b
    bg_e_a: wgpu::BindGroup,         // src e_a, add t_a
    bg_e_a_solo: wgpu::BindGroup,    // src e_a only (the 1/16 -> 1/64 box writes t_a)
    bg_e_b: wgpu::BindGroup,         // src e_b
    bg_t_a: wgpu::BindGroup,         // src t_a
    bg_t_b: wgpu::BindGroup,         // src t_b

    grid_pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    fog_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    /// Real-time probe GI state (compute update + the probe buffers bound at camera 10-12).
    rtgi: rtgi::RtGi,
    // vMF diffuse LUT (group0 binding 7/8) — kept so the resize rebuild can re-bind it.
    vmf_lut_view: wgpu::TextureView,
    vmf_lut_samp: wgpu::Sampler,
    /// The engine's 3D diffuse_power_specular LUT (camera binding 13), kept for the resize rebuild.
    dps_lut_view: wgpu::TextureView,
    // Kept so the camera bind group can be rebuilt on resize (it binds the resized
    // scene-depth copy at binding 5).
    camera_bgl: wgpu::BindGroupLayout,
    shadow_samp: wgpu::Sampler,

    sky_pipeline: wgpu::RenderPipeline,
    sky_buf: wgpu::Buffer,
    sky_bg: wgpu::BindGroup,

    grid_vbuf: wgpu::Buffer,
    grid_vcount: u32,

    mesh_renderer: std::sync::Arc<MeshRenderer>,
    static_meshes: Vec<GpuMesh>,  // BSP (rebuilt on map load)
    imported_meshes: Vec<GpuMesh>, // external imported geometry (persists across map loads)
    /// Static marker meshes (a Halo 4 map's own spawn-family placements) drawn like
    /// `static_meshes` only while `show_markers` - the live "Show map spawn points" toggle
    /// for geometry that is uploaded once per load rather than rebuilt per frame.
    marker_meshes: Vec<GpuMesh>,
    /// The markers' additive / alpha-blend parts (a spawn point's light beam), sorted into
    /// the transparent pass with the dynamic holos while `show_markers`.
    marker_holo_meshes: Vec<GpuMesh>,
    marker_blend_meshes: Vec<GpuMesh>,
    show_markers: bool,
    dynamic_meshes: Vec<GpuMesh>, // forge/objects (rebuilt when they change)
    dynamic_cutout_meshes: Vec<GpuMesh>, // cutout objects (tree canopies) → alpha-test pass
    dynamic_holo_meshes: Vec<GpuMesh>, // holo/objective objects (hill markers, globe) → additive pass
    dynamic_holo_solid_meshes: Vec<GpuMesh>, // object markers (spawns/hill/kill-safe shells) → alpha-blended holo (always-readable shape)
    dynamic_blend_meshes: Vec<GpuMesh>, // transparent OBJECT parts (forge glass windows) → alpha-blend pass
    particle_meshes: Vec<GpuMesh>, // per-frame effect particle batches (per-mesh blend routing)
    water_meshes: Vec<GpuMesh>,   // shader_water surfaces (transparent pass)
    // Per-map water volume planes for camera-underwater detection, and the fullscreen
    // underwater-fog pass state. The bind group is rebuilt on resize (depth_copy_view changes).
    water_planes: Vec<WaterPlane>,
    underwater_pipeline: wgpu::RenderPipeline,
    underwater_bgl: wgpu::BindGroupLayout,
    underwater_buf: wgpu::Buffer,
    underwater_bg: wgpu::BindGroup,
    underwater_enabled: bool,
    planar_fog_volumes: Vec<PlanarFogVolume>,
    planar_fog_pipeline: wgpu::RenderPipeline,
    planar_fog_bgl: wgpu::BindGroupLayout,
    planar_fog_buf: wgpu::Buffer,
    planar_fog_bg: wgpu::BindGroup,
    terrain_meshes: Vec<GpuMesh>, // shader_terrain multi-layer blend
    alphatest_meshes: Vec<GpuMesh>, // foliage cutout (discard pass)
    foliage_meshes: Vec<GpuMesh>,   // decorators (grass/flowers) — multiplicative pipeline
    blend_meshes: Vec<GpuMesh>,     // glass/translucent (alpha-blend pass)
    additive_meshes: Vec<GpuMesh>,  // holograms/force-fields/energy (additive pass)
    lens_pipeline: wgpu::RenderPipeline, // sun lens flare (screen-space additive)
    lens_elements: Vec<f32>,        // count×8: [axis_offset, radius, brightness, r, g, b, tint_power, modulation]
    lensfx: lensfx::LensFx,         // world lens flares (street lights etc.)
    decal_meshes: Vec<(GpuMesh, u8)>, // decals (mesh, blend bucket 0=alpha/1=add/2=multiply)
    sky_segments: Vec<(GpuMesh, u8, f32)>,  // sky render_model segments (mesh, blend mode, draw-order key: authored section index + radius tiebreak, sorted DESC = back-to-front)
    /// Six per-face camera bind groups (origin eye, 90° FOV, cube-face orientation) for the
    /// env-cube sky capture, and a one-shot flag set on each map's sky load.
    env_face_cams: Vec<wgpu::BindGroup>,
    env_capture_pending: std::cell::Cell<bool>,
    /// Pre-recorded render bundles for the STATIC mesh lists (BSP opaque, imported, terrain,
    /// alpha-test, decorators, water, decals, additive). Re-recorded lazily in `render`
    /// whenever `static_gen` differs from the cached generation -- every list mutator and every
    /// camera bind-group rebuild bumps it. Replaying a bundle costs a fraction of re-encoding
    /// ~9000 validated draws per frame (Panopticon: 15.8 ms -> see HMS_FRAMES).
    static_gen: u64,
    static_bundles: std::cell::RefCell<Option<StaticBundles>>,
    /// HMS_RENDER_PROF accumulator (section sums, frame count) for THIS renderer.
    prof_acc: std::cell::Cell<([f32; 16], u32)>,

    highlight_vbuf: Option<wgpu::Buffer>,
    highlight_vcount: u32,
    /// Skip drawing the selection highlight this frame WITHOUT dropping its buffer (a
    /// team/colour dropdown hover previews the colour on the bare object; the wireframe comes
    /// straight back when the hover ends).
    highlight_hidden: bool,
    /// #wire-visible  Draw the selection wireframe THROUGH other objects (depth test ALWAYS).
    /// View > "Selection wireframe through objects", off by default.
    highlight_xray: bool,
    /// #wire-visible  The selection-wireframe lane: black halo + exposure-compensated bright core
    /// (WIRE_WGSL), depth-tested and depth-always variants.
    wire_pipeline: wgpu::RenderPipeline,
    wire_xray_pipeline: wgpu::RenderPipeline,
    wire_bg: wgpu::BindGroup,
    wire_buf: wgpu::Buffer,
    /// Transform gizmo (always-on-top colored axis lines + solid triangles).
    gizmo_pipeline: wgpu::RenderPipeline,
    gizmo_vbuf: Option<wgpu::Buffer>,
    gizmo_vcount: u32,
    gizmo_tri_pipeline: wgpu::RenderPipeline,
    gizmo_tri_vbuf: Option<wgpu::Buffer>,
    gizmo_tri_vcount: u32,
    // Translucent volumetric zone fill (boundary shapes of the SELECTED object): depth-tested
    // holographic fill (fresnel rim + pulse) drawn under the bright wireframe outline.
    zone_pipeline: wgpu::RenderPipeline,
    zone_vbuf: Option<wgpu::Buffer>,
    zone_vcount: u32,
    overlay_vbuf: Option<wgpu::Buffer>,
    overlay_vcount: u32,
    /// X-ray line lane (depth test ALWAYS, the gizmo line pipeline) -- the soft ceiling
    /// edges, so the kill floor under the water / terrain reads from above.
    xray_vbuf: Option<wgpu::Buffer>,
    xray_vcount: u32,

    // Whether the loaded map authored atmosphere (fogg). Gates the procedural horizon
    // gradient: the engine paints no gradient — outdoor maps get their horizon color from
    // atmosphere fog, space maps (no fogg) get nothing behind the sky model. Default true
    // so pre-load behaviour is unchanged; set per-map from SceneController::has_atmosphere.
    sky_has_atmosphere: bool,
    // Space maps (condemned/zealot): clear to true black (void) and skip the horizon
    // gradient. Default false.
    space_sky: bool,
    sky_tint: [f32; 3], // map's atmosphere_fog sky inscatter color (outdoor horizon)

    // View toggles (draw gating).
    pub show_sky: bool,
    pub show_grid: bool,
    pub show_bsp: bool,
    pub show_terrain: bool,
    pub show_water: bool,
    pub show_objects: bool,

    // Per-map key/sun direction (TO the sun, world space, Z-up). Drives the shadow
    // frustum + all shading. A placeholder until the scene loader supplies the per-map value.
    pub sun_dir: Vec3,
    /// Per-scenario sceg colour-grade tints (default white = no change). The sun term is
    /// multiplied by `sun_tint`, the baked/ambient term by `ambient_tint`, so each map gets
    /// its authored hue (e.g. Zealot's purple).
    pub sun_tint: Vec3,
    pub ambient_tint: Vec3,
    /// Lighting Lab live multipliers (UI sliders). 1.0 = no-op. sun_mult scales the analytical
    /// sun term, ambient_mult the airprobe/sky ambient, lm_mult the baked lightmap/PVL term.
    /// fog_wu overrides the atmosphere density bridge (0 → use the const/env default).
    pub sun_mult: f32,
    /// Map-level baked sun visibility (0..1). See `set_sun_reach`.
    pub sun_reach: f32,
    /// Halo 4 object "forge lightmap burn" shadow: Some(world AABB of every object caster)
    /// fits the sun shadow map to the casters at the engine's texel density instead of the Reach
    /// camera-centred box; None = Reach behaviour.
    pub h4_shadow_bounds: Option<(Vec3, Vec3)>,
    /// The map's Halo 4 floating-shadow cascade (None = no dynamic cascade).
    pub h4_cascade: Option<H4CascadeCfg>,
    /// Print-only diag of the last Halo 4 shadow fit (texel wu, depth range wu, casters).
    pub h4_shadow_diag: std::cell::Cell<[f32; 4]>,
    pub ambient_mult: f32,
    pub lm_mult: f32,
    pub fog_wu: f32,
    /// Per-map dynamic-OBJECT directional soft-shade strength (the map's airprobe directional
    /// fraction). Carried to the shader in sun_dir.w; 0 → shader uses its modest default. Set
    /// from `SceneController::obj_dir_strength()` on load.
    pub obj_dir_strength: f32,
    /// Fixed engine g_exposure written into the per-frame grade.w. >0 → the post pass uses this
    /// exact gain instead of the auto-exposure meter (engine-faithful, view-independent). 0 → meter.
    pub fixed_exposure: f32,
    /// [stops_gain, auto_key, lo(2^min_ev), hi(2^max_ev), exp_cal, illum_preferred P, illum_scale s].
    pub illum_params: std::cell::Cell<[f32; 7]>,
    /// Light-uniform lane kept for layout stability; no shader reads it.
    pub sky_ambient: [f32; 4],
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32)) -> Self {
        let size = (size.0.max(1), size.1.max(1));

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Atmospheric fog uniform (4 vec4 = the fogg two-band params). Shared on
        // the camera bind group so mesh + terrain fragment shaders can apply fog.
        let fog_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-uniform"),
            size: 112, // 7 × vec4 (sky+ground bands, fog-light disc, sun dir)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bufent = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Analytical-sun shadow map: a single depth texture rendered from the sun's
        // POV. group0 gains binding 2 = light uniform (light_view_proj + sun_dir),
        // 3 = shadow depth texture, 4 = comparison sampler. Baked PVL already gives
        // terrain/instances self-shadow; this adds real-time CAST shadows (objects
        // onto the world). 2048² fixed (doesn't resize with the window).
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-uniform"),
            // mat4 lvp (64) + sun_dir + sun_tint + ambient_tint + sky_ambient + dbg mults + expo + expo2 (11 vec4)
            // + Halo 4 shadows: h4s0, h4s1, cascade view_proj (mat4), casc0, casc1 (9 vec4) = 320 B. The Reach
            // shaders declare an 11-vec4 `Light` struct (a larger buffer is legal); only the mesh module's
            // struct carries the H4 tail (zeros on Reach maps = inactive).
            size: LIGHT_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // SimpleLights uniform: u32 count padded to a vec4 (16B) + 8 lights ×
        // 5×vec4 (80B) = 656B. Zero-initialised → count 0 → no contribution.
        let simple_lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simple-lights-uniform"),
            size: 16 + 8 * 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d { width: 2048, height: 2048, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        // The Halo 4 floating-shadow cascade depth map (scnr resolution 512 or 800,
        // rendered into the top-left res x res of this 800² texture).
        let h4_cascade_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("h4-cascade-shadow"),
            size: wgpu::Extent3d { width: H4_CASCADE_TEX, height: H4_CASCADE_TEX, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let h4_cascade_view = h4_cascade_tex.create_view(&Default::default());
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-cmp-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[
                bufent(0),
                bufent(1),
                bufent(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // binding 5 = opaque scene-depth copy (water depth-murkiness, particle / halogram
                // depth fade); read via textureLoad (no sampler).
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 6 = scenario SimpleLights uniform (point/spot).
                bufent(6),
                // binding 7 = the byte-exact vMF diffuse LUT (128×128 R8, the engine's
                // rasterizer\diffusetable bitmap). The baked-lightmap atlas decode samples this
                // at (dot(domDir,N)*0.5+0.5, clamp(bandwidth,0,1)) — the engine's
                // dual_vmf_diffuse (docs/hrek_re/00 §3).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 8 = filtering sampler for the vMF LUT (clamp-to-edge, linear).
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 9 = the 1x1 auto-exposure luminance meter (shaders derive the exposure E from it).
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // bindings 10-12 = real-time GI: probe SH records (storage), GI params (uniform), probe visibility (storage).
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // binding 13 = the engine's 3D diffuse_power_specular LUT (64x64x4 R8), sampled
                // with the vMF LUT sampler (binding 8) by blend_shade's glass area term.
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 14 = the Halo 4 floating-shadow cascade depth map (800², point
                // depth compares in the mesh shader's h4_cascade_shadow; textureLoad, no sampler).
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // Shadow (depth-only) pipeline: transforms casters by light_view_proj. Uses
        // a minimal bind group (just the light uniform at group0 binding 2 via a
        // dedicated 1-entry layout) so it doesn't need the shadow texture bound.
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-bgl"),
            entries: &[bufent(2)],
        });
        // The cascade caster pass: the same depth shader reads `light.light_view_proj`,
        // so it gets its own light-sized uniform whose matrix slot holds the cascade view_proj.
        let h4_cascade_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("h4-cascade-light-uniform"),
            size: LIGHT_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let h4_cascade_pass_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("h4-cascade-pass-bg"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 2, resource: h4_cascade_light_buf.as_entire_binding() }],
        });
        let shadow_pass_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-pass-bg"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("basic-color"),
            source: wgpu::ShaderSource::Wgsl(BASIC_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid-pl"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });
        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Transform gizmo: same LineList shader as the grid, but depth_compare=Always
        // (+ no depth write) so the gizmo draws OVER everything and is grabbable through
        // other objects.
        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // SOLID gizmo: same shader, TriangleList, always-on-top — for solid arrow shafts,
        // cones, and rotate tori (the line pipeline handles only the thin constraint guide).
        let gizmo_tri_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo-tri-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No back-face cull so the gizmo reads solid from any angle.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Translucent zone volume: holographic boundary-shape fill. Depth-tested (LessEqual, no
        // write) so it reads as a real volume occluded by nearer geometry; alpha-blended; no cull
        // so both faces show. Reuses the camera bind group (needs cam_pos + time for fresnel/pulse).
        let zone_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zone-volume"),
            source: wgpu::ShaderSource::Wgsl(ZONE_WGSL.into()),
        });
        let zone_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ZoneVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 12, shader_location: 1 },
            ],
        };
        let zone_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("zone-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &zone_shader,
                entry_point: Some("vs"),
                buffers: &[zone_vbl],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &zone_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // ---- sun lens flare: screen-space additive sprites, NDC vertex buffer built
        // per-frame, no bind groups. Vertex = pos(ndc xy) + uv + color + tint_power. ----
        let lens_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens-flare"),
            source: wgpu::ShaderSource::Wgsl(LENS_WGSL.into()),
        });
        let lens_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lens-pl"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let lens_vbl = wgpu::VertexBufferLayout {
            array_stride: 36, // vec2 pos + vec2 uv + vec4 color + f32 tint_power
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 32, shader_location: 3 },
            ],
        };
        let lens_additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let lens_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lens-pipeline"),
            layout: Some(&lens_layout),
            vertex: wgpu::VertexState { module: &lens_shader, entry_point: Some("vs"), buffers: &[lens_vbl], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always, // flare draws over everything
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &lens_shader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format: SCENE_HDR_FORMAT, blend: Some(lens_additive), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            multiview: None,
            cache: None,
        });

        // ---- procedural sky (fullscreen triangle, own uniform) ----
        let sky_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky-uniform"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let sky_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky-bg"),
            layout: &sky_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: sky_buf.as_entire_binding() }],
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky"),
            source: wgpu::ShaderSource::Wgsl(SKY_WGSL.into()),
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-pl"),
            bind_group_layouts: &[&sky_bgl],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky-pipeline"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // Sky fills the far plane and never occludes geometry.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let grid_verts = build_grid(64, 1.0);
        let grid_vcount = grid_verts.len() as u32;
        let grid_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid-vbuf"),
            size: (grid_verts.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        grid_vbuf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&grid_verts));
        grid_vbuf.unmap();

        let Targets {
            color_tex,
            color_view,
            depth_tex,
            depth_view,
            depth_copy_tex,
            depth_copy_view,
            hdr_view,
            hdr_tex,
        } = make_targets(device, size);
        let color_view_srgb = color_tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(COLOR_FORMAT_SRGB),
            ..Default::default()
        });
        // The engine's own `rasterizer\diffusetable` bitmap (128x128, format 11 A8R8G8B8, curve
        // linear, all four channels identical), decoded through the normal bitmap decoder
        // (ZH decode_bitmap_keep_alpha on the tag) and stored verbatim — byte-identical on
        // 52_ivory_tower / forge_halo / 45_aftship / condemned. x = dot(dom_dir, n)·0.5+0.5,
        // y = bandwidth (convertBandwidth2TextureCoord is the identity; spherical_harmonics.hlsl:50-66),
        // sampled with the shader's own bilinear+clamp. Sanity: at dot=1 the table reads
        // 0.52/0.78/0.97 at bandwidth .5/.8/1, and the bandwidth=0 row sits at ~0.25 = the
        // isotropic answer, the engine's hard-coded fill coefficient in dual_vmf_diffuse.
        const VMF_DIFFUSE_LUT: &[u8] = include_bytes!("vmf_diffuse_lut.bin");
        const VMF_LUT_W: u32 = 128;
        const VMF_LUT_H: u32 = 128;
        let vmf_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vmf-diffuse-lut"),
            size: wgpu::Extent3d { width: VMF_LUT_W, height: VMF_LUT_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &vmf_lut_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            VMF_DIFFUSE_LUT,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(VMF_LUT_W), rows_per_image: Some(VMF_LUT_H) },
            wgpu::Extent3d { width: VMF_LUT_W, height: VMF_LUT_H, depth_or_array_layers: 1 },
        );
        let vmf_lut_view = vmf_lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let vmf_lut_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vmf-lut-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // The engine's `rasterizer\diffuse_power_specular\diffuse_power` LUT (Zealot tag 0x1e:
        // 64x64x4, bitmap format 36 = BC4 mono, decoded through the native decoder's format-36 case and
        // stored verbatim, x fastest). cook_torrance_core:227-236 / materials/diffuse_specular.hlsl:41-62:
        // sh_glossy = final_tint * (lut(dot(dom_dir, R)*0.5+0.5, bandwidth, roughness) * 3 * dom_rgb + 0.25 * fill_rgb)
        // -- the DIRECTIONAL area term glass reflections are scaled by. Bound as a real texture_3d at camera
        // binding 13 (sampled with the vMF LUT's clamp+linear sampler, so z lerps between roughness slices
        // exactly like the engine's convert_3d_texture_coord_to_array_texture lerp).
        const DPS_LUT: &[u8] = include_bytes!("diffuse_power_lut.bin");
        const DPS_W: u32 = 64;
        const DPS_H: u32 = 64;
        const DPS_D: u32 = 4;
        let dps_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("diffuse-power-specular-lut"),
            size: wgpu::Extent3d { width: DPS_W, height: DPS_H, depth_or_array_layers: DPS_D },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &dps_lut_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            DPS_LUT,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(DPS_W), rows_per_image: Some(DPS_H) },
            wgpu::Extent3d { width: DPS_W, height: DPS_H, depth_or_array_layers: DPS_D },
        );
        let dps_lut_view = dps_lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Auto-exposure luminance meter (1×1 R32F) + its pipeline (samples the HDR).
        let lum_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lum-1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let lum_view = lum_tex.create_view(&Default::default());
        let rtgi = rtgi::RtGi::new(device);
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fog_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&depth_copy_view) },
                wgpu::BindGroupEntry { binding: 6, resource: simple_lights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&vmf_lut_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&vmf_lut_samp) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&lum_view) },
                wgpu::BindGroupEntry { binding: 10, resource: rtgi.probes_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: rtgi.params_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: rtgi.vis_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&dps_lut_view) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&h4_cascade_view) },
            ],
        });

        // Six cube-face cameras (origin eye, 90° FOV, standard cube-face fwd/up) for the
        // env-cube sky capture. Each reuses the shared fog/light/shadow/depth/slights bindings + a
        // per-face view_proj buffer, so draw_sky renders the dome into the matching cube face.
        let env_face_cams: Vec<wgpu::BindGroup> = {
            use wgpu::util::DeviceExt;
            let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 30000.0);
            let faces: [(Vec3, Vec3); 6] = [
                (Vec3::X, Vec3::NEG_Y), (Vec3::NEG_X, Vec3::NEG_Y),
                (Vec3::Y, Vec3::Z), (Vec3::NEG_Y, Vec3::NEG_Z),
                (Vec3::Z, Vec3::NEG_Y), (Vec3::NEG_Z, Vec3::NEG_Y),
            ];
            faces.iter().map(|&(fwd, up)| {
                let vp = proj * Mat4::look_at_rh(Vec3::ZERO, fwd, up);
                let u = CameraUniform {
                    view_proj: vp.to_cols_array_2d(),
                    cam_pos: [0.0, 0.0, 0.0, 0.0],
                    time: [0.0, 0.0, 0.0, 0.0],
                    inv_view_proj: vp.inverse().to_cols_array_2d(),
                };
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("env-face-cam"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("env-face-cam-bg"),
                    layout: &camera_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: fog_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&depth_copy_view) },
                        wgpu::BindGroupEntry { binding: 6, resource: simple_lights_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&vmf_lut_view) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&vmf_lut_samp) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&lum_view) },
                        wgpu::BindGroupEntry { binding: 10, resource: rtgi.probes_buf().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 11, resource: rtgi.params_buf().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 12, resource: rtgi.vis_buf().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&dps_lut_view) },
                        wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&h4_cascade_view) },
                    ],
                })
            }).collect()
        };

        // ---- post-processing (tonemap + fog + vignette): fullscreen composite ----
        let post_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        // Generic single-texture+sampler layout (bright pass + blur inputs).
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex-bgl"),
            entries: &[tex_entry(0), samp_entry(1)],
        });
        // Non-filtering float texture entry (the R32F luminance meter isn't filterable).
        let tex_entry_nf = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // Post layout: HDR scene (0) + sampler (1) + bloom (2) + uniform (3) + lum (4).
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-bgl"),
            entries: &[tex_entry(0), samp_entry(1), tex_entry(2), bufent(3), tex_entry_nf(4)],
        });
        // Post uniform (14 vec4; see `Post` in POST_WGSL). Defaults until a map sets them:
        //   p     @0   = [exposure stops gain, auto_key, exp_min_log2, exp_max_log2]
        //   grade @16  = [rgb grade (identity), fixed g_exposure (0 = metered)]
        //   bloom @32  = [user bloom scale, meter cal, cfxs bloom point, cfxs inherent]
        //   bloom2@48  = [cfxs intensity, h4 filmic on, P0, P1]
        //   sfx   @64  = [screen-effect gain (2^stops), gamma exponent (0.5 = sqrt), 1/w, 1/h]
        //   sfx2  @80  = [0, 0, 0, h4 P2]
        //   bcl/bcm/bcs @96/112/128 = engine default cfxs bloom colours (globals\defaults\default):
        //                large BLACK, medium 0.2406, small WHITE; .w pads = h4 P3, P4, resolution scale
        //   cm0..cm2 @144/160/176 = the engine's k_ps_color_matrix as three float4 COLUMNS
        //                (out[j] = dot(vec4(rgb,1), cm[j])); identity until `set_screen_fx`
        //   h4a   @192 = (bloom highlight, inherent, self-illum, sensitivity)
        //   h4b   @208 = (screen brightness log2, H4 meter on, LUT on, H4 bloom on)
        let post_buf = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post-uniform"),
                contents: bytemuck::cast_slice(&[1.0f32, 0.25, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.2406, 0.2406, 0.2406, 0.0, 1.0, 1.0, 1.0, 0.0,
                                                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        // #wire-visible  Selection WIREFRAME lane: its own shader + bind group so the line stays
        // legible on ANY background (a white Forge piece under a blown-out Halo 4 sky read as a
        // pale line on pale pixels). One line list is drawn as several INSTANCES: instances 0-7 are
        // a near-black halo offset around the line in screen space, instances 8-11 the bright core
        // on top -- a dark outline under a bright core, so one of the two edges always carries the
        // contrast whatever is behind it. The core is exposure-compensated in the fragment stage
        // (it lands at a fixed EXPOSED value instead of following the scene's auto-exposure down).
        // Two pipelines: depth-tested (normal) and depth-ALWAYS (View > wireframe through objects).
        let wire_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wire-lines"),
            source: wgpu::ShaderSource::Wgsl(WIRE_WGSL.into()),
        });
        let wire_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wire-bgl"),
            entries: &[
                // 0 = the POST uniform itself (p / grade / bloom prefix) so the exposure the
                // resolve pass will apply is read from the same source, never a stale CPU mirror.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 1 = wire params: xy = 1 / HDR target size (px -> clip), z = halo radius (HDR px),
                // w = the exposed value the core aims for.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let wire_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wire-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wire_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wire-bg"),
            layout: &wire_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: post_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wire_buf.as_entire_binding() },
            ],
        });
        let wire_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wire-pl"),
            bind_group_layouts: &[&camera_bgl, &wire_bgl],
            push_constant_ranges: &[],
        });
        let make_wire = |label: &str, always: bool| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&wire_pl),
            vertex: wgpu::VertexState {
                module: &wire_shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Never writes depth: the halo ring must not punch holes for the core, and the
                // wireframe must not occlude anything drawn after it.
                depth_write_enabled: false,
                depth_compare: if always { wgpu::CompareFunction::Always } else { wgpu::CompareFunction::LessEqual },
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &wire_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let wire_pipeline = make_wire("wire-pipeline", false);
        let wire_xray_pipeline = make_wire("wire-xray-pipeline", true);

        // Meter blend SCALE_X = geomean(0.0)↔arith-mean(1.0), default 0.5 (the engine's
        // exposure_downsample register; pure geomean over-brightens interiors — Ivory Tower mean
        // 134→153 — the arith term lifts dark enclosed scenes so they don't over-gain).
        // HMS_METER_SCALE tunes it.
        let ms: f32 = std::env::var("HMS_METER_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
        // `// #h4-expo-2` __SS__ = the supersample factor: the Halo 4 meter's `exposure_downsample`
        // footprint is defined in QUARTER-resolution texels of the engine's render target, which
        // is HMS's OUTPUT size (SS is HMS's own anti-aliasing, not an engine resolution).
        let lum_src = LUM_WGSL.replace("__METER_SCALE__", &format!("{:.4}", ms)).replace("__SS__", &format!("{:.1}", SS as f32));
        let lum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lum"),
            source: wgpu::ShaderSource::Wgsl(lum_src.into()),
        });
        // Meter layout: HDR (0) + sampler (1) + post uniform (2) + the 48x32 H4 pre-pass (3)
        let lum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lum-bgl"),
            entries: &[tex_entry(0), samp_entry(1), bufent(2), tex_entry_nf(3)],
        });
        let lum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lum-pl"),
            bind_group_layouts: &[&lum_bgl],
            push_constant_ranges: &[],
        });
        let h4pre_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("h4pre"),
            source: wgpu::ShaderSource::Wgsl(H4PRE_WGSL.into()),
        });
        let h4pre_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h4pre-pl"),
            bind_group_layouts: &[&tex_bgl],
            push_constant_ranges: &[],
        });
        let h4pre_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("h4pre-pipeline"),
            layout: Some(&h4pre_layout),
            vertex: wgpu::VertexState { module: &h4pre_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &h4pre_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let h4pre_view = make_h4pre_tex(device);
        // Colour-grading LUT group: texture_3d (0) + sampler (1)
        let post_lut_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-lut-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                samp_entry(1),
            ],
        });
        let post_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let post_lut_bg = make_lut_bg(device, queue, &post_lut_bgl, &post_lut_sampler, None);
        let lum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lum-pipeline"),
            layout: Some(&lum_layout),
            vertex: wgpu::VertexState { module: &lum_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &lum_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R32Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let bloom = make_bloom_pyramid(device, size);
        let lum_bg = make_lum_bg(device, &lum_bgl, &hdr_view, &post_sampler, &post_buf, &h4pre_view);
        let h4pre_bg = make_tex_bg(device, &tex_bgl, &hdr_view, &post_sampler);
        let post_bg =
            make_post_bg(device, &post_bgl, &hdr_view, &bloom.out, &post_sampler, &post_buf, &lum_view);
        // Bright pass binds the post layout: hdr(0)+sampler(1)+[bloom slot unused→hdr](2)+post(3)+lum(4).
        let bright_in_bg = make_post_bg(device, &post_bgl, &hdr_view, &hdr_view, &post_sampler, &post_buf, &lum_view);
        let bg_q_a = make_post_bg(device, &post_bgl, &bloom.q_a, &bloom.e_b, &post_sampler, &post_buf, &lum_view);
        let bg_q_b = make_post_bg(device, &post_bgl, &bloom.q_b, &bloom.q_b, &post_sampler, &post_buf, &lum_view);
        let bg_e_a = make_post_bg(device, &post_bgl, &bloom.e_a, &bloom.t_a, &post_sampler, &post_buf, &lum_view);
        let bg_e_a_solo = make_post_bg(device, &post_bgl, &bloom.e_a, &bloom.e_a, &post_sampler, &post_buf, &lum_view);
        let bg_e_b = make_post_bg(device, &post_bgl, &bloom.e_b, &bloom.e_b, &post_sampler, &post_buf, &lum_view);
        let bg_t_a = make_post_bg(device, &post_bgl, &bloom.t_a, &bloom.t_a, &post_sampler, &post_buf, &lum_view);
        let bg_t_b = make_post_bg(device, &post_bgl, &bloom.t_b, &bloom.t_b, &post_sampler, &post_buf, &lum_view);

        // Underwater-fog pass: uniform (0) + pre-transparent scene depth copy (1). Alpha-blends
        // the fog colour into the HDR scene by reconstructed-depth distance (see UNDERWATER_WGSL).
        let underwater_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("underwater-bgl"),
            entries: &[
                bufent(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let underwater_buf = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("underwater-uniform"),
                contents: bytemuck::bytes_of(&UnderwaterUniform {
                    inv_view_proj: [[0.0; 4]; 4],
                    cam_pos: [0.0; 4],
                    fog: [0.0; 4],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let underwater_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("underwater"),
            source: wgpu::ShaderSource::Wgsl(UNDERWATER_WGSL.into()),
        });
        let underwater_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("underwater-pl"),
            bind_group_layouts: &[&underwater_bgl],
            push_constant_ranges: &[],
        });
        let underwater_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("underwater-pipeline"),
            layout: Some(&underwater_layout),
            vertex: wgpu::VertexState { module: &underwater_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &underwater_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let underwater_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("underwater-bg"),
            layout: &underwater_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: underwater_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_copy_view) },
            ],
        });

        // Planar-fog pass: uniform (0) + scene depth copy (1). Same shape as underwater but
        // PREMULTIPLIED alpha blend (front-to-back inscatter accumulation over volumes).
        let planar_fog_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planar-fog-bgl"),
            entries: &[
                bufent(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let planar_fog_buf = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("planar-fog-uniform"),
                contents: bytemuck::bytes_of(&PlanarFogUniform {
                    inv_view_proj: [[0.0; 4]; 4],
                    cam_pos: [0.0; 4],
                    count: [0.0; 4],
                    vols: [PFogVolGpu { plane: [0.0; 4], color_density: [0.0; 4], bmin: [0.0; 4], bmax: [0.0; 4] }; MAX_PFOG],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let planar_fog_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planar-fog"),
            source: wgpu::ShaderSource::Wgsl(PLANAR_FOG_WGSL.into()),
        });
        let planar_fog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planar-fog-pl"),
            bind_group_layouts: &[&planar_fog_bgl],
            push_constant_ranges: &[],
        });
        let planar_fog_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar-fog-pipeline"),
            layout: Some(&planar_fog_layout),
            vertex: wgpu::VertexState { module: &planar_fog_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &planar_fog_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        // premultiplied: result = src.rgb + dst·(1-src.a)
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let planar_fog_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar-fog-bg"),
            layout: &planar_fog_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: planar_fog_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_copy_view) },
            ],
        });

        // Bloom pipelines (bright-pass + horizontal/vertical Gaussian blur).
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_WGSL.into()),
        });
        // EVERY bloom pass binds the post layout (src(0) + sampler(1) + second
        // input(2) + post uniform(3) + lum meter(4)) so the per-map cfxs intensity and the
        // per-frame exposure gain are visible in each pass (the engine scales each pyramid level
        // by bloom_intensity at a specific point of the chain, with a 2.0 clamp per pass).
        let bright_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bright-pl"),
            bind_group_layouts: &[&post_bgl, &post_lut_bgl], // group 1 = the grading LUT (Halo 4 bloom)
            push_constant_ranges: &[],
        });
        let make_fs_pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom-pipeline"),
                layout: Some(&bright_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };
        let bright_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bright-pipeline"),
            layout: Some(&bright_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_bright"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        // env-cube mip generation keeps a tex-only (src + sampler) 2x2 box pipeline.
        let env_mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("env-mip-pl"),
            bind_group_layouts: &[&tex_bgl],
            push_constant_ranges: &[],
        });
        let env_mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("env-mip-pipeline"),
            layout: Some(&env_mip_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_downsample_env"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let blur_h_pipeline = make_fs_pipeline("fs_blur_h");
        let blur_v_pipeline = make_fs_pipeline("fs_blur_v");
        let blur_v_i_pipeline = make_fs_pipeline("fs_blur_v_i");
        let downsample_pipeline = make_fs_pipeline("fs_box8");
        let bloom_add_pipeline = make_fs_pipeline("fs_add_m");
        let bloom_add_s_pipeline = make_fs_pipeline("fs_add_s");
        // Halo 4 chain
        let bright_h4_pipeline = make_fs_pipeline("fs_bright_h4");
        let box8_h4_pipeline = make_fs_pipeline("fs_box8_h4");
        let blur_h_h4_pipeline = make_fs_pipeline("fs_blur_h_h4");
        let blur_v_h4_pipeline = make_fs_pipeline("fs_blur_v_h4");
        let blur_v_i_h4_pipeline = make_fs_pipeline("fs_blur_v_i_h4");
        let add_m_h4_pipeline = make_fs_pipeline("fs_add_m_h4");
        let add_s_h4_pipeline = make_fs_pipeline("fs_add_s_h4");
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(POST_WGSL.into()),
        });
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-pl"),
            bind_group_layouts: &[&post_bgl, &post_lut_bgl], // group 1 = the grading LUT
            push_constant_ranges: &[],
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post-pipeline"),
            layout: Some(&post_layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let mesh_renderer = MeshRenderer::new(device, &camera_bgl, &shadow_bgl, SCENE_HDR_FORMAT, DEPTH_FORMAT);
        // World lens-flare pass (samples the opaque depth copy + the luminance meter).
        let lensfx = lensfx::LensFx::new(device, &camera_buf, &light_buf, &depth_copy_view, &lum_view, SCENE_HDR_FORMAT, DEPTH_FORMAT);

        Self {
            size,
            color_tex,
            color_view,
            color_view_srgb,
            depth_view,
            depth_tex,
            depth_copy_tex,
            depth_copy_view,
            hdr_view,
            hdr_tex,
            post_pipeline,
            post_bgl,
            post_sampler,
            post_bg,
            post_buf,
            lum_view,
            lum_tex,
            lum_rb: (0..3).map(|i| (device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["lum-rb-0", "lum-rb-1", "lum-rb-2"][i]),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }), std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)))).collect(),
            lum_rb_next: std::cell::Cell::new(0),
            lum_rb_last: std::cell::Cell::new(None),
            lum_pipeline,
            lum_bgl,
            h4pre_pipeline,
            h4pre_view,
            h4pre_bg,
            h4_meter_on: std::cell::Cell::new(false),
            post_lut_bgl,
            post_lut_bg,
            post_lut_sampler,
            lum_bg,
            shadow_view,
            h4_cascade_view,
            h4_cascade_light_buf,
            h4_cascade_pass_bg,
            light_buf,
            simple_lights_buf,
            shadow_pass_bg,
            tex_bgl,
            bright_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            blur_v_i_pipeline,
            downsample_pipeline,
            env_mip_pipeline,
            bloom_add_pipeline,
            bloom_add_s_pipeline,
            h4_bloom_on: std::cell::Cell::new(false),
            bright_h4_pipeline,
            box8_h4_pipeline,
            blur_h_h4_pipeline,
            blur_v_h4_pipeline,
            blur_v_i_h4_pipeline,
            add_m_h4_pipeline,
            add_s_h4_pipeline,
            bloom,
            bright_in_bg,
            bg_q_a,
            bg_q_b,
            bg_e_a,
            bg_e_a_solo,
            bg_e_b,
            bg_t_a,
            bg_t_b,
            grid_pipeline,
            camera_buf,
            fog_buf,
            camera_bg,
            rtgi,
            vmf_lut_view,
            vmf_lut_samp,
            dps_lut_view,
            camera_bgl,
            shadow_samp,
            sky_pipeline,
            sky_buf,
            sky_bg,
            grid_vbuf,
            grid_vcount,
            mesh_renderer: std::sync::Arc::new(mesh_renderer),
            static_meshes: Vec::new(),
            imported_meshes: Vec::new(),
            marker_meshes: Vec::new(),
            marker_holo_meshes: Vec::new(),
            marker_blend_meshes: Vec::new(),
            show_markers: false,
            dynamic_meshes: Vec::new(),
            dynamic_cutout_meshes: Vec::new(),
            dynamic_holo_meshes: Vec::new(),
            dynamic_holo_solid_meshes: Vec::new(),
            dynamic_blend_meshes: Vec::new(),
            particle_meshes: Vec::new(),
            water_meshes: Vec::new(),
            water_planes: Vec::new(),
            underwater_pipeline,
            underwater_bgl,
            underwater_buf,
            underwater_bg,
            // OFF by default. Forge-style water is a flat quad the terrain pokes through, so a
            // camera on dry land sits below the water surface within its XY footprint —
            // indistinguishable from being submerged without water-bed/terrain data. Auto-activating
            // would fog the normal above-water preview, so the pass is gated on an explicit enable
            // (HMS_UNDERWATER, or set_underwater_enabled). TODO: a water-column (bed depth) test.
            // When enabled, it activates on the footprint+below-surface test.
            underwater_enabled: std::env::var("HMS_UNDERWATER").is_ok(),
            planar_fog_volumes: Vec::new(),
            planar_fog_pipeline,
            planar_fog_bgl,
            planar_fog_buf,
            planar_fog_bg,
            terrain_meshes: Vec::new(),
            alphatest_meshes: Vec::new(),
            foliage_meshes: Vec::new(),
            blend_meshes: Vec::new(),
            additive_meshes: Vec::new(),
            lens_pipeline,
            lens_elements: Vec::new(),
            lensfx,
            decal_meshes: Vec::new(),
            sky_segments: Vec::new(),
            env_face_cams,
            env_capture_pending: std::cell::Cell::new(false),
            static_gen: 1,
            static_bundles: std::cell::RefCell::new(None),
            prof_acc: std::cell::Cell::new(([0.0; 16], 0)),
            highlight_vbuf: None,
            highlight_hidden: false,
            highlight_xray: false,
            wire_pipeline,
            wire_xray_pipeline,
            wire_bg,
            wire_buf,
            gizmo_pipeline,
            gizmo_vbuf: None,
            gizmo_vcount: 0,
            xray_vbuf: None,
            xray_vcount: 0,
            gizmo_tri_pipeline,
            gizmo_tri_vbuf: None,
            gizmo_tri_vcount: 0,
            zone_pipeline,
            zone_vbuf: None,
            zone_vcount: 0,
            highlight_vcount: 0,
            overlay_vbuf: None,
            overlay_vcount: 0,
            sky_has_atmosphere: true,
            space_sky: false,
            sky_tint: [0.21, 0.51, 0.91],
            show_sky: true,
            show_grid: true,
            // Pass-isolation debug toggles (HMS_NO_WATER / HMS_NO_BSP / HMS_NO_TERRAIN).
            show_bsp: std::env::var("HMS_NO_BSP").is_err(),
            show_terrain: std::env::var("HMS_NO_TERRAIN").is_err(),
            show_water: std::env::var("HMS_NO_WATER").is_err(),
            show_objects: true,
            sun_dir: Vec3::new(0.4, 0.7, 0.6).normalize(),
            sun_tint: Vec3::ONE,
            ambient_tint: Vec3::ONE,
            sun_mult: 1.0,
            sun_reach: 1.0,
            h4_shadow_bounds: None,
            h4_cascade: None,
            h4_shadow_diag: std::cell::Cell::new([0.0; 4]),
            ambient_mult: 1.0,
            lm_mult: 1.0,
            fog_wu: 0.0,
            obj_dir_strength: 0.0, // 0 → shader default (OBJ_DIR_SHADE); set per-map on load
            fixed_exposure: 0.0, // 0 → metered auto-exposure; >0 → fixed engine g_exposure
            illum_params: std::cell::Cell::new([1.0, 0.25, 0.25, 2.0, 1.0, 0.0, 0.3]),
            sky_ambient: [0.0, 0.0, 0.0, 0.0], // off by default
        }
    }

    /// Set the per-map key/sun direction (TO the sun, world Z-up). Renormalized;
    /// a zero/non-finite vector is ignored so the loader can pass through safely.
    pub fn set_sun_dir(&mut self, dir: Vec3) {
        let l = dir.length();
        if l.is_finite() && l > 1e-4 {
            self.sun_dir = dir / l;
        }
    }

    /// Set the per-scenario sceg colour-grade tints (sun + ambient). Non-finite or
    /// all-zero components fall back to white so a mis-read can't black out the scene.
    pub fn set_scene_tints(&mut self, sun_tint: [f32; 3], ambient_tint: [f32; 3]) {
        let san = |t: [f32; 3]| {
            let v = Vec3::from(t);
            if v.is_finite() && v.max_element() > 1e-4 { v } else { Vec3::ONE }
        };
        self.sun_tint = san(sun_tint);
        self.ambient_tint = san(ambient_tint);
    }

    /// Per-map dynamic-object directional soft-shade strength (the map's airprobe
    /// directional fraction). 0 → shader uses its modest default. Clamped to a sane band.
    pub fn set_obj_dir_strength(&mut self, s: f32) {
        self.obj_dir_strength = if s.is_finite() { s.clamp(0.0, 0.7) } else { 0.0 };
    }

    /// Hide / show the selection highlight without touching its buffer.
    pub fn set_highlight_hidden(&mut self, hidden: bool) {
        self.highlight_hidden = hidden;
    }

    /// #wire-visible  Draw the selection wireframe THROUGH other geometry (depth test ALWAYS)
    /// instead of only where it is unoccluded. View > "Selection wireframe through objects".
    pub fn set_highlight_xray(&mut self, on: bool) {
        self.highlight_xray = on;
    }

    /// #wire-visible  Whether the selection wireframe currently draws through objects.
    pub fn highlight_xray(&self) -> bool {
        self.highlight_xray
    }

    /// Set the selection highlight to a WIREFRAME line-list (world-space vertex pairs)
    /// — the actual object mesh edges, not a bounding box. Empty/None clears it. Reuses the
    /// same highlight line pipeline as the AABB box.
    pub fn set_highlight_lines(&mut self, device: &wgpu::Device, lines: Option<&[[f32; 3]]>) {
        match lines {
            None => {
                self.highlight_vbuf = None;
                self.highlight_vcount = 0;
            }
            Some(pts) if pts.is_empty() => {
                self.highlight_vbuf = None;
                self.highlight_vcount = 0;
            }
            Some(pts) => {
                let c = [0.98, 0.78, 0.25];
                let verts: Vec<Vertex> = pts.iter().map(|p| Vertex { pos: *p, color: c }).collect();
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("highlight-wireframe-vbuf"),
                    size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&verts));
                buf.unmap();
                self.highlight_vcount = verts.len() as u32;
                self.highlight_vbuf = Some(buf);
            }
        }
    }

    /// Set (or clear) the transform-gizmo line list — world-space (pos, color) vertex
    /// pairs drawn ALWAYS-ON-TOP (grabbable through other objects). None/empty clears it.
    pub fn set_gizmo_lines(&mut self, device: &wgpu::Device, lines: Option<&[([f32; 3], [f32; 3])]>) {
        match lines {
            Some(pts) if !pts.is_empty() => {
                let verts: Vec<Vertex> = pts.iter().map(|(p, c)| Vertex { pos: *p, color: *c }).collect();
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gizmo-vbuf"),
                    size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&verts));
                buf.unmap();
                self.gizmo_vcount = verts.len() as u32;
                self.gizmo_vbuf = Some(buf);
            }
            _ => {
                self.gizmo_vbuf = None;
                self.gizmo_vcount = 0;
            }
        }
    }

    /// Set (or clear) the SOLID gizmo triangle list — world-space (pos, color) triples
    /// (3 verts per triangle) drawn always-on-top. None/empty clears it.
    pub fn set_gizmo_tris(&mut self, device: &wgpu::Device, tris: Option<&[([f32; 3], [f32; 3])]>) {
        match tris {
            Some(pts) if !pts.is_empty() => {
                let verts: Vec<Vertex> = pts.iter().map(|(p, c)| Vertex { pos: *p, color: *c }).collect();
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gizmo-tri-vbuf"),
                    size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&verts));
                buf.unmap();
                self.gizmo_tri_vcount = verts.len() as u32;
                self.gizmo_tri_vbuf = Some(buf);
            }
            _ => {
                self.gizmo_tri_vbuf = None;
                self.gizmo_tri_vcount = 0;
            }
        }
    }

    /// Set (or clear) the translucent zone volume — a flat triangle list of (world pos, rgba)
    /// (3 verts per triangle). Drawn depth-tested + alpha-blended with a fresnel rim + pulse so a
    /// boundary shape reads as a holographic volume. None/empty clears it.
    pub fn set_zone_tris(&mut self, device: &wgpu::Device, tris: Option<&[([f32; 3], [f32; 4])]>) {
        match tris {
            Some(pts) if !pts.is_empty() => {
                let verts: Vec<ZoneVertex> = pts.iter().map(|(p, c)| ZoneVertex { pos: *p, color: *c }).collect();
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zone-vbuf"),
                    size: (verts.len() * std::mem::size_of::<ZoneVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&verts));
                buf.unmap();
                self.zone_vcount = verts.len() as u32;
                self.zone_vbuf = Some(buf);
            }
            _ => {
                self.zone_vbuf = None;
                self.zone_vcount = 0;
            }
        }
    }

    /// Set (or clear) the amber selection box drawn around the picked object's AABB.
    pub fn set_highlight(&mut self, device: &wgpu::Device, aabb: Option<([f32; 3], [f32; 3])>) {
        match aabb {
            None => {
                self.highlight_vbuf = None;
                self.highlight_vcount = 0;
            }
            Some((mn, mx)) => {
                let c = [0.98, 0.78, 0.25];
                let corner = |i: usize| Vertex {
                    pos: [
                        if i & 1 == 0 { mn[0] } else { mx[0] },
                        if i & 2 == 0 { mn[1] } else { mx[1] },
                        if i & 4 == 0 { mn[2] } else { mx[2] },
                    ],
                    color: c,
                };
                let edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0), // bottom
                    (4, 5), (5, 7), (7, 6), (6, 4), // top
                    (0, 4), (1, 5), (2, 6), (3, 7), // verticals
                ];
                let mut verts = Vec::with_capacity(24);
                for (a, b) in edges {
                    verts.push(corner(a));
                    verts.push(corner(b));
                }
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("highlight-vbuf"),
                    size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&verts));
                buf.unmap();
                self.highlight_vcount = verts.len() as u32;
                self.highlight_vbuf = Some(buf);
            }
        }
    }

    /// Access to the mesh uploader (scene controller builds meshes with this).
    pub fn mesh_renderer(&self) -> &MeshRenderer {
        &self.mesh_renderer
    }
    /// Clone the shared mesh uploader for use on a background load thread (it is
    /// Send+Sync — wgpu pipelines/layouts/samplers only).
    pub fn mesh_renderer_arc(&self) -> std::sync::Arc<MeshRenderer> {
        self.mesh_renderer.clone()
    }

    /// Set overlay line segments (e.g. trigger-volume wireframes). Each segment
    /// is (start, end, color). Pass an empty slice to clear.
    pub fn set_overlay_lines(&mut self, device: &wgpu::Device, segments: &[([f32; 3], [f32; 3], [f32; 3])]) {
        if segments.is_empty() {
            self.overlay_vbuf = None;
            self.overlay_vcount = 0;
            return;
        }
        let mut verts = Vec::with_capacity(segments.len() * 2);
        for (a, b, c) in segments {
            verts.push(Vertex { pos: *a, color: *c });
            verts.push(Vertex { pos: *b, color: *c });
        }
        use wgpu::util::DeviceExt;
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay-vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.overlay_vcount = verts.len() as u32;
        self.overlay_vbuf = Some(buf);
    }

    /// Set (or clear) the X-RAY line list -- world-space (a, b, rgb) segments
    /// drawn with the depth test OFF (the gizmo line pipeline), so an overlay hidden under the
    /// terrain or the water (the map's kill floor) still reads. Empty clears it.
    pub fn set_xray_lines(&mut self, device: &wgpu::Device, segments: &[([f32; 3], [f32; 3], [f32; 3])]) {
        if segments.is_empty() {
            self.xray_vbuf = None;
            self.xray_vcount = 0;
            return;
        }
        let mut verts = Vec::with_capacity(segments.len() * 2);
        for (a, b, c) in segments {
            verts.push(Vertex { pos: *a, color: *c });
            verts.push(Vertex { pos: *b, color: *c });
        }
        use wgpu::util::DeviceExt;
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xray-vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.xray_vcount = verts.len() as u32;
        self.xray_vbuf = Some(buf);
    }

    /// Rebuild the camera bind group (bindings 5 and 10-12 point at the current depth copy and
    /// probe buffers).
    fn rebuild_camera_bg(&mut self, device: &wgpu::Device) {
        self.camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &self.camera_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fog_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.depth_copy_view) },
                wgpu::BindGroupEntry { binding: 6, resource: self.simple_lights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.vmf_lut_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.vmf_lut_samp) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.lum_view) },
                wgpu::BindGroupEntry { binding: 10, resource: self.rtgi.probes_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.rtgi.params_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.rtgi.vis_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&self.dps_lut_view) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&self.h4_cascade_view) },
            ],
        });
        self.bump_static(); // bundles bake in the camera bind group
    }
    /// Upload the map's real-time GI tracer scene (BVH, materials, lights) and size the probe grid.
    pub fn set_rtgi_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &rtgi::RtGiScene) {
        let env = self.mesh_renderer.env_cube_view();
        let env_tex = self.mesh_renderer.env_cube_texture().map(|t| (t, mesh::EV_CUBE_MIPS));
        self.rtgi.set_scene(device, queue, scene, env, env_tex);
        self.rebuild_camera_bg(device);
    }
    pub fn set_rtgi_enabled(&mut self, queue: &wgpu::Queue, on: bool) { self.rtgi.set_enabled(queue, on); }
    pub fn rtgi_has_scene(&self) -> bool { self.rtgi.has_scene() }
    /// (probe count, spacing, dims, updates run)
    pub fn rtgi_info(&self) -> (u32, f32, [u32; 3], u32) { (self.rtgi.nprobes, self.rtgi.spacing, self.rtgi.dims, self.rtgi.updates.get()) }
    pub fn rtgi_warm_up(&self, device: &wgpu::Device, queue: &wgpu::Queue, n: u32) { self.rtgi.warm_up(device, queue, n); }
    pub fn rtgi_set_sun(&mut self, queue: &wgpu::Queue, dir: [f32; 3], color: [f32; 3]) { self.rtgi.set_sun(queue, dir, color); }
    pub fn rtgi_set_gain(&mut self, queue: &wgpu::Queue, gain: f32) { self.rtgi.set_gain(queue, gain); }
    /// The tracer's current sun (direction TO the sun, HDR irradiance) — the map's own values
    /// until the user moves it.
    pub fn rtgi_sun(&self) -> ([f32; 3], [f32; 3]) {
        let p = &self.rtgi.params;
        ([p.sun_dir[0], p.sun_dir[1], p.sun_dir[2]], [p.sun_color[0], p.sun_color[1], p.sun_color[2]])
    }
    pub fn rtgi_kick(&self) { self.rtgi.kick(); }
    /// Forget the tracer scene (new map) — the mode turns off until re-enabled.
    pub fn rtgi_clear_scene(&mut self, queue: &wgpu::Queue) { self.rtgi.clear_scene(queue); }
    pub fn rtgi_dump_at(&self, device: &wgpu::Device, queue: &wgpu::Queue, p: [f32; 3]) -> Vec<String> { self.rtgi.dump_at(device, queue, p) }
    /// Scenario exposure stops gain (post.p.x), applied in the post/tonemap pass. 1.0 = neutral.
    pub fn set_exposure(&self, queue: &wgpu::Queue, exposure: f32) {
        { let mut p = self.illum_params.get(); p[0] = exposure; self.illum_params.set(p); }
        // Write only p.x (the scenario stops-gain); p.yzw (auto-exposure key/min/max)
        // keep their defaults. Auto-exposure (the 1×1 lum meter) supplies the rest.
        queue.write_buffer(&self.post_buf, 0, bytemuck::cast_slice(&[exposure]));
    }

    /// User bloom scale (post.bloom.x, offset 32): multiplies the whole bloom contribution
    /// in resolve(). 1.0 = engine default; <1 dials bloom down (fixing the washed-shadow /
    /// over-bloom look), 0 = no bloom. Exposed as a UI slider + the HMS_BLOOM headless knob.
    pub fn set_bloom(&self, queue: &wgpu::Queue, scale: f32) {
        queue.write_buffer(&self.post_buf, 32, bytemuck::cast_slice(&[scale.max(0.0)]));
    }

    /// The COMPOSED screen effect (scenario default + placed Forge FX objects, composed on the
    /// app side by `screenfx::compose`): `gain` = 2^(exposure boost - deboost), a
    /// linear gain before the gamma-2 pow; `gamma_exp` = the engine's gamma.z (0.5 = plain sqrt,
    /// 0.5 * clamp(1 + enhance - reduce)); `cols` = the engine's k_ps_color_matrix as three float4
    /// columns (hue rotation * saturation * sqrt(filter)/contrast/floor), applied AFTER the pow
    /// (final_composite `saturate(mul(float4(color,1), color_matrix))`). Identity: (1, 0.5, I).
    pub fn set_screen_fx(&self, queue: &wgpu::Queue, gain: f32, gamma_exp: f32, cols: [[f32; 4]; 3]) {
        let gain = if gain.is_finite() && gain > 0.0 { gain.clamp(1.0 / 256.0, 256.0) } else { 1.0 };
        let ge = if gamma_exp.is_finite() && gamma_exp > 0.0 { gamma_exp.clamp(0.005, 5.0) } else { 0.5 };
        queue.write_buffer(&self.post_buf, 64, bytemuck::cast_slice(&[gain, ge])); // sfx.zw = the Halo 4 bloom output size
        queue.write_buffer(&self.post_buf, 80, bytemuck::cast_slice(&[0.0f32, 0.0, 0.0]));
        let mut m = [[0.0f32; 4]; 3];
        for j in 0..3 {
            for i in 0..4 {
                let v = cols[j][i];
                m[j][i] = if v.is_finite() { v.clamp(-64.0, 64.0) } else if i == j { 1.0 } else { 0.0 };
            }
        }
        queue.write_buffer(&self.post_buf, 144, bytemuck::cast_slice(&m));
    }

    /// Per-map bloom curve from the cfxs tag — `point` → post.bloom.z (offset 40), `inherent` →
    /// post.bloom.w (offset 44), `intensity` → post.bloom2.x (offset 48). Pass point = 0 to keep
    /// the built-in defaults. `intensity` is an ENGINE parameter with its own lane; post.bloom.x
    /// (`set_bloom`) is the USER's bloom slider.
    pub fn set_bloom_curve(&self, queue: &wgpu::Queue, point: f32, inherent: f32, intensity: f32) {
        queue.write_buffer(&self.post_buf, 40, bytemuck::cast_slice(&[point.max(0.0), inherent.max(0.0)]));
        queue.write_buffer(&self.post_buf, 48, bytemuck::cast_slice(&[intensity.max(0.0)]));
    }
    /// cfxs BLOOM LARGE / MEDIUM / SMALL COLOR -> the 1/64, 1/16 and 1/4 level
    /// multipliers (engine kernel_5 / bloom_add_alpha1 / add `scale` = intensity * colour).
    pub fn set_bloom_colors(&self, queue: &wgpu::Queue, c: [[f32; 3]; 3]) {
        for (i, col) in c.iter().enumerate() {
            // rgb only: the .w pads of bcl / bcm carry the Halo 4 filmic constants (set_h4_filmic)
            let v = [col[0].max(0.0), col[1].max(0.0), col[2].max(0.0)];
            queue.write_buffer(&self.post_buf, 96 + 16 * i as u64, bytemuck::cast_slice(&v));
        }
    }

    /// Halo 4 `cfxs` filmic tone curve (final_composite): `p` = the five composite
    /// constants (P0..P4, lighting.rs `H4CameraFx::filmic_params`), None = off (Reach path).
    /// Lanes: bloom2.y flag (52), bloom2.z/w = P0/P1 (56/60), sfx2.w = P2 (92), bcl.w = P3 (108),
    /// bcm.w = P4 (124) - all otherwise-unused pads of the post uniform.
    pub fn set_h4_filmic(&self, queue: &wgpu::Queue, p: Option<[f32; 5]>) {
        let (flag, v) = match p { Some(v) => (1.0f32, v), None => (0.0f32, [0.0; 5]) };
        queue.write_buffer(&self.post_buf, 52, bytemuck::cast_slice(&[flag, v[0], v[1]]));
        queue.write_buffer(&self.post_buf, 92, bytemuck::cast_slice(&[v[2]]));
        queue.write_buffer(&self.post_buf, 108, bytemuck::cast_slice(&[v[3]]));
        queue.write_buffer(&self.post_buf, 124, bytemuck::cast_slice(&[v[4]]));
    }

    /// The Halo 4 auto-exposure meter (halo4.dll `sub_180359768` + the explicit
    /// `downsample_block_bloom_2x2` / `exposure_downsample` shaders, see LUM_WGSL): `p` = (bloom
    /// highlight, inherent, self-illum, sensitivity, screen brightness log2) of the map's cfxs. The
    /// meter then SOLVES the stops s inside the band set by `set_exposure_band_abs` for which the
    /// engine's measurement of the exposed scene equals the screen brightness (the engine's
    /// adaptation converges to that point). None = the Reach meter.
    pub fn set_h4_meter(&self, queue: &wgpu::Queue, p: Option<[f32; 5]>) {
        self.h4_meter_on.set(p.is_some());
        let v = p.unwrap_or([0.0; 5]);
        queue.write_buffer(&self.post_buf, 192, bytemuck::cast_slice(&[v[0], v[1], v[2], v[3], v[4], if p.is_some() { 1.0f32 } else { 0.0 }]));
    }

    /// True while the Halo 4 engine meter is selected (`set_h4_meter(Some(..))`): the 1x1 lum
    /// texture then carries `log2(key) - solved_stops` instead of a mean-log luminance, and the
    /// band set by `set_exposure_band_abs` is in ABSOLUTE gains. A caller that reproduces the
    /// resolve pass's gain on the CPU (the interactive adaptation in hms-app) needs this to pick
    /// the Halo 4 adaptation law instead of Reach's.
    pub fn h4_meter_on(&self) -> bool { self.h4_meter_on.get() }

    /// The exposure terms the resolve pass reads, exactly as it reads them:
    /// `(base, key, lo_gain, hi_gain, cal)` with `gain = base * clamp(key / metered * cal, lo, hi)`
    /// (post.p.xyzw + post.bloom.y; see `resolve`). The one source of truth for any CPU replica of
    /// the gain -- `// #h4-expo-3`: the interactive adaptation used to re-derive these from the
    /// Reach `SceneController` alone, which silently replaced a Halo 4 map's solved exposure.
    pub fn exposure_params(&self) -> (f32, f32, f32, f32, f32) {
        let p = self.illum_params.get();
        (p[0], p[1], p[2], p[3], if p[4] > 0.0 { p[4] } else { 1.0 })
    }

    /// The atmosphere fog density actually in force (world-unit bridge): the `set_fog_wu`
    /// override when one is set, else the shader constant `FOG_WU_DEFAULT`. A UI that seeds a
    /// slider from the live state must start here, not from a hard-coded guess.
    pub fn fog_wu_effective(&self) -> f32 { if self.fog_wu > 0.0 { self.fog_wu } else { FOG_WU_DEFAULT } }

    /// Select the Halo 4 bloom chain (BLOOM_WGSL `*_h4`; h4b.w @220). The curve /
    /// colour lanes (`set_bloom_curve`, `set_bloom_colors`) and the self-illum weight (h4a.z,
    /// `set_h4_meter`) are shared with the meter; `render` refreshes the output-size lanes
    /// (sfx.zw = 1 / size, bcs.w = min(1, 921600 / (w h))) every frame.
    pub fn set_h4_bloom(&self, queue: &wgpu::Queue, on: bool) {
        self.h4_bloom_on.set(on);
        queue.write_buffer(&self.post_buf, 220, bytemuck::cast_slice(&[if on { 1.0f32 } else { 0.0 }]));
    }

    /// The Halo 4 final_composite colour-grading volume (cfxs +0xF0, 16^3 BGRA8,
    /// sampled at `sqrt(filmic) * (n-1)/n + 0.5/n` AFTER the sqrt). None = identity (Reach).
    pub fn set_h4_color_grading(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lut: Option<(&[u8], u32)>) {
        self.post_lut_bg = make_lut_bg(device, queue, &self.post_lut_bgl, &self.post_lut_sampler, lut);
        queue.write_buffer(&self.post_buf, 216, bytemuck::cast_slice(&[if lut.is_some() { 1.0f32 } else { 0.0 }]));
    }

    /// Absolute exposure band: gain = clamp(key / metered_luminance, lo, hi) with lo / hi
    /// as LINEAR gains (Halo 4's `cfxs` exposure range is in absolute stops, E = 2^stops, no
    /// Reach-style key scaling). Writes p.y/p.z/p.w of the post uniform directly.
    pub fn set_exposure_band_abs(&self, queue: &wgpu::Queue, key: f32, lo: f32, hi: f32) {
        let (lo, hi) = (lo.max(1e-4), hi.max(1e-4));
        { let mut p = self.illum_params.get(); p[1] = key; p[2] = lo; p[3] = hi; self.illum_params.set(p); }
        queue.write_buffer(&self.post_buf, 4, bytemuck::cast_slice(&[key, lo.log2(), hi.log2()]));
    }

    /// Meter-unit exposure calibration (post.bloom.y, offset 36). 0 → shader default (1.0).
    pub fn set_exp_cal(&self, queue: &wgpu::Queue, cal: f32) {
        { let mut p = self.illum_params.get(); p[4] = if cal > 0.0 { cal } else { 1.0 }; self.illum_params.set(p); }
        queue.write_buffer(&self.post_buf, 36, bytemuck::cast_slice(&[cal.max(0.0)]));
    }

    /// The map's baked sun-visibility population mean (see `SceneController::baked_sun_reach`).
    /// 1.0 = unknown (no per-vertex bake) -> the glass sun lanes stay ungated.
    pub fn set_sun_reach(&mut self, v: f32) { self.sun_reach = v.clamp(0.0, 1.0); }
    /// Halo 4 sun shadows: `bounds` = the world AABB of every object caster (None turns the
    /// Halo 4 fit off = Reach behaviour), `cascade` = the BSP's floating-shadow cascade (None = none).
    pub fn set_h4_shadow(&mut self, bounds: Option<(Vec3, Vec3)>, cascade: Option<H4CascadeCfg>) {
        self.h4_shadow_bounds = bounds.filter(|(a, b)| a.is_finite() && b.is_finite() && b.cmpge(*a).all());
        self.h4_cascade = cascade;
    }

    /// Lighting Lab live multipliers (written into the light buffer each frame). 1.0 = no-op.
    pub fn set_lighting_mults(&mut self, sun: f32, ambient: f32, lightmap: f32) {
        self.sun_mult = sun.max(0.0);
        self.ambient_mult = ambient.max(0.0);
        self.lm_mult = lightmap.max(0.0);
    }
    /// Live atmosphere fog density bridge (cam.time.z). 0 → const/env default.
    pub fn set_fog_wu(&mut self, fog_wu: f32) { self.fog_wu = fog_wu.max(0.0); }

    /// Fixed per-frame g_exposure scalar that REPLACES the live auto-exposure meter (written to
    /// grade.w, post uniform offset 28). >0 → the post pass uses this exact gain (no view-dependent
    /// adaptation, matching the engine's fixed sceg exposure); 0 → the metered auto-exposure.
    /// Pairs with absolute-unit lm_k (HMS_ENGINE_DECODE).
    pub fn set_fixed_exposure(&mut self, queue: &wgpu::Queue, g: f32) {
        self.fixed_exposure = g.max(0.0); // the per-frame grade write re-uses this field
        queue.write_buffer(&self.post_buf, 28, bytemuck::cast_slice(&[self.fixed_exposure]));
    }

    /// HMS_EXPODIAG: read back the 1×1 luminance meter — returns the metered
    /// mean-log2-luminance so the exposure calibration can see whether each map is
    /// CLAMPING at its cfxs band edge or ADAPTING inside it. Call after a rendered frame.
    pub fn read_mean_log(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<f32> {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lum-readback"),
            size: 256, // min copy alignment
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lum_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        let data = buf.slice(..).get_mapped_range();
        let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        buf.unmap();
        Some(v)
    }

    /// NON-BLOCKING meter readback for the interactive auto-exposure. The blocking
    /// `read_mean_log` (poll Wait) stalls the UI thread for the whole GPU frame (5-8 ms on a 4090,
    /// 30-60 ms on the first frame) -- a periodic hitch on top of the vsync budget, and it
    /// serialises CPU and GPU. This queues a 1x1 copy into a free staging
    /// slot, maps it asynchronously, polls WITHOUT waiting, and returns the most recent value
    /// that has completed (2-3 frames old, irrelevant for the 0.10/frame adaptation EMA). Call
    /// once per frame after `render`; None until the first value lands.
    pub fn read_mean_log_async(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<f32> {
        use std::sync::atomic::Ordering;
        // 1. harvest any slot whose map completed (callback set state 2).
        for (buf, st) in &self.lum_rb {
            if st.load(Ordering::Acquire) == 2 {
                let v = {
                    let data = buf.slice(..).get_mapped_range();
                    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
                };
                buf.unmap();
                st.store(0, Ordering::Release);
                if v.is_finite() { self.lum_rb_last.set(Some(v)); }
            }
        }
        // 2. queue this frame's copy into the next free slot (skip if all are in flight).
        let n = self.lum_rb.len();
        let start = self.lum_rb_next.get();
        for k in 0..n {
            let i = (start + k) % n;
            let (buf, st) = &self.lum_rb[i];
            if st.load(Ordering::Acquire) != 0 { continue; }
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lum-readback") });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &self.lum_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            queue.submit(Some(enc.finish()));
            st.store(1, Ordering::Release);
            let flag = st.clone();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                // a failed map (device lost) frees the slot instead of wedging it
                flag.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release);
            });
            self.lum_rb_next.set((i + 1) % n);
            break;
        }
        // 3. no explicit poll: every queue.submit() runs a non-blocking maintain that fires the
        // completed map callbacks (wgpu-core queue.rs), so the value is harvested on a later call.
        // An extra device.poll here only added lock traffic against the loader thread's uploads.
        self.lum_rb_last.get()
    }

    /// Read back the LINEAR pre-tonemap HDR scene (Rgba16Float, SS×size) as f32 RGBA (HMS_HDR_DUMP).
    /// This is the ONLY reliable per-pixel probe — the PNG capture and any debug-return go through the
    /// post pass (exposure + sqrt tonemap + bloom), which saturates and (with auto-exposure) confounds
    /// every ratio. Returns (rgba_f32, width, height). Call after `render`.
    pub fn dump_hdr(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<f32>, u32, u32) {
        let (w, h) = (self.size.0 * SS, self.size.1 * SS);
        self.dump_rgba16f(device, queue, &self.hdr_tex, w, h)
    }
    /// Read back the final bloom level (1/4 output res, RGBA f32: rgb = the bloom the
    /// composite adds to `scene * gain`, a = the Halo 4 alpha lane). Diag only.
    pub fn dump_bloom(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<f32>, u32, u32) {
        let (w, h) = ((self.size.0 / 4).max(1), (self.size.1 / 4).max(1));
        self.dump_rgba16f(device, queue, &self.bloom.out_tex, w, h)
    }
    fn dump_rgba16f(&self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32) -> (Vec<f32>, u32, u32) {
        // Rgba16Float = 8 bytes/texel; align bytes_per_row to 256.
        let unpadded = w * 8;
        let padded = (unpadded + 255) & !255;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr-readback"),
            size: (padded as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        let data = buf.slice(..).get_mapped_range();
        let mut out = vec![0.0f32; (w as usize) * (h as usize) * 4];
        for y in 0..h as usize {
            let row = &data[y * padded as usize..];
            for x in 0..w as usize {
                for c in 0..4 {
                    let o = (x * 4 + c) * 2;
                    let bits = u16::from_le_bytes([row[o], row[o + 1]]);
                    out[(y * w as usize + x) * 4 + c] = half::f16::from_bits(bits).to_f32();
                }
            }
        }
        drop(data);
        buf.unmap();
        (out, w, h)
    }

    /// cfxs self-illum exposure (P = preferred exposure stops, s = exposure change scale) for
    /// ILLUM_SCALE = g_alt_exposure.r = 2^((1-s)(P-E)) (mesh/sky `illum_scale_now`).
    /// None restores the engine defaults (0, 0.3). Takes effect at the next light-uniform upload.
    pub fn set_illum_exposure(&self, ps: Option<(f32, f32)>) {
        let (p, s) = ps.unwrap_or((0.0, 0.3));
        let mut a = self.illum_params.get(); a[5] = p; a[6] = s; self.illum_params.set(a);
    }
    /// Tag-driven auto-exposure band from the map's cfxs camera_fx_settings: writes p.y = key
    /// (target), p.z/p.w = the gain band in stops, so bright scenes (Ivory/Tempest) can dim as far
    /// as the engine allows. Values are clamped to sane bounds. Call after `set_exposure` (which
    /// writes p.x). Also clears the Halo 4 filmic curve / meter / grading-LUT flags (a Reach map
    /// loaded after a Halo 4 one must not keep them; the LUT bind group itself is left, the flag gates it).
    pub fn set_auto_exposure(&self, queue: &wgpu::Queue, key: f32, min_ev: f32, max_ev: f32) {
        self.set_h4_filmic(queue, None);
        self.set_h4_meter(queue, None);
        queue.write_buffer(&self.post_buf, 216, bytemuck::cast_slice(&[0.0f32]));
        let key = key.clamp(0.02, 4.0);
        let mut lo = min_ev.clamp(-8.0, 8.0);
        let mut hi = max_ev.clamp(-8.0, 8.0);
        if hi <= lo { lo = -2.0; hi = 1.0; } // degenerate → fall back to the prior band
        // Engine band (Sapien Ivory Tower frame_1/bb_1 — same camera, same FOV, the engine's OWN
        // exposure EV read from its 240B global CB): the cfxs min/max EV band does NOT clamp the
        // linear gain to [2^min_ev, 2^max_ev]. The engine adapts in stops around the authored
        // key: g = key · 2^clamp(−log2(m_e), min_ev, max_ev), where m_e is the meter in ENGINE
        // radiance units (= ENGINE_METER_UNITS × our meter; the same constant converts the engine
        // gain back, so the UNCLAMPED gain is still exactly key/m). Only the band moves:
        // [key·C·2^min_ev, key·C·2^max_ev]. Ivory (key 0.09, min_ev −1): engine EV −4.5046 =
        // key_stops(−3.474) + min_ev(−1) − 0.03, i.e. the engine sat AT its band floor, 0.249 in
        // our units. C is anchored on the Sapien FORGE capture (bb8_terrain: engine rock exposed
        // 0.351 vs our rock radiance 0.309 → the engine's gain at Forge's band floor ≈ 1.13 in our
        // units → key(0.1)·C ≈ 1.13, C ≈ 11). Ivory's white walls imply 5.66; the 2× between them
        // is a MID-TONE RADIANCE gap on our side (inside the same Ivory frame the wood/floor
        // blocks read 2–3× darker than the walls relative to the engine), not exposure.
        // TODO: revisit C when the mid-tone radiance is fixed.
        const ENGINE_METER_UNITS: f32 = 10.0;
        let kc = (key * ENGINE_METER_UNITS).max(1e-4);
        let lo_g = kc * lo.exp2();
        let hi_g = kc * hi.exp2();
        // p.y/z/w live at byte offsets 4/8/12 in the post uniform (z/w are STOPS; the post
        // shader exp2's them). illum_params carries the linear band for the mesh shaders.
        { let mut p = self.illum_params.get(); p[1] = key; p[2] = lo_g; p[3] = hi_g; self.illum_params.set(p); }
        queue.write_buffer(&self.post_buf, 4, bytemuck::cast_slice(&[key, lo_g.log2(), hi_g.log2()]));
    }

    /// Upload the atmospheric fog uniform (7 vec4 packed by the caller from the map's fogg
    /// params; see FOG_WGSL). atm3.w == 0 disables fog in the shader.
    pub fn set_fog(&mut self, queue: &wgpu::Queue, fog: &[f32; 28]) {
        queue.write_buffer(&self.fog_buf, 0, bytemuck::cast_slice(&fog[..]));
        // Stash the atmosphere sky inscatter color (fog[0..3]) so the outdoor sky gradient
        // is colored from the MAP's own atmosphere tag, not a hardcoded blue.
        self.sky_tint = [fog[0], fog[1], fog[2]];
        // Highlands yellow-sky guard: some maps render a PROCEDURAL sky dome and their fogg
        // "sky" colour is an HDR WARM value (Highlands = 5.0,3.35,0.0) that is NOT an atmosphere
        // inscatter tint. Painted into the fullscreen gradient it makes the whole sky solid yellow.
        // A real atmosphere sky inscatter is blue-dominant; when the authored colour is warm (blue
        // is not the max channel) or HDR, it is the wrong role — substitute a neutral daytime sky so
        // the fallback reads as sky. TODO: un-cull the procedural dome instead of this stopgap.
        let t = self.sky_tint;
        let blue_dominant = t[2] >= t[0] * 0.98 && t[2] >= t[1] * 0.98;
        if !blue_dominant {
            self.sky_tint = [0.36, 0.52, 0.72];
        }
    }

    /// The placed world lens flares (effect `lens` parts at their marker positions).
    pub fn set_lens_flare_instances(&mut self, instances: Vec<LensFlareInstance>) {
        self.lensfx.instances = instances;
    }
    /// Set the sun lens-flare elements: `count`×8 f32 [axis_offset, radius, brightness, r, g, b,
    /// tint_power, modulation]. Empty = no flare (the map's sun has no lens tag).
    pub fn set_lens_flare(&mut self, elements: Vec<f32>) {
        self.lens_elements = elements;
    }

    /// Upload the scenario SimpleLights. `light_floats` is `count`×20 f32 (each
    /// light = the 80B ZhSimpleLight = 5×vec4). Clamped to 8. count 0 → no lights.
    pub fn set_simple_lights(&self, queue: &wgpu::Queue, count: u32, light_floats: &[f32]) {
        let n = (count as usize).min(8);
        let mut buf = [0f32; 4 + 8 * 20]; // 16B meta (count in [0]) + 8×80B lights = 656B
        buf[0] = n as f32;
        let copy = (n * 20).min(light_floats.len());
        buf[4..4 + copy].copy_from_slice(&light_floats[..copy]);
        queue.write_buffer(&self.simple_lights_buf, 0, bytemuck::cast_slice(&buf));
    }

    /// Static (BSP) meshes — set once per map load.
    pub fn set_static_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.static_meshes = meshes;
        self.bump_static();
    }
    /// Append a chunk of static meshes (incremental/chunked map load). No bump: the bundle
    /// chain records only the new tail.
    pub fn append_static_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.static_meshes.append(&mut meshes);
    }
    /// Imported external geometry — a base to forge over. Persists across map loads
    /// (NOT cleared like static/BSP meshes), so import + load-map / import-only both work.
    pub fn set_imported_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.imported_meshes = meshes;
        self.bump_static();
    }
    pub fn append_imported_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.imported_meshes.append(&mut meshes);
    }
    pub fn clear_imported_meshes(&mut self) {
        self.imported_meshes.clear();
        self.bump_static();
    }
    /// The static marker lane (replaced per map load; drawn while `show_markers`):
    /// opaque / alpha-test parts, additive parts, alpha-blend parts.
    pub fn set_marker_meshes(&mut self, opaque: Vec<GpuMesh>, additive: Vec<GpuMesh>, blend: Vec<GpuMesh>) {
        self.marker_meshes = opaque;
        self.marker_holo_meshes = additive;
        self.marker_blend_meshes = blend;
        self.bump_static();
    }
    /// Show / hide the marker lane (no re-upload; the bundle is already recorded).
    pub fn set_show_markers(&mut self, on: bool) { self.show_markers = on; }
    /// Append a chunk of water meshes (incremental/chunked map load).
    pub fn append_water_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.water_meshes.append(&mut meshes);
    }
    /// Append a chunk of terrain meshes (incremental/chunked map load).
    pub fn append_terrain_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.terrain_meshes.append(&mut meshes);
    }
    /// Append a chunk of alpha-test (foliage cutout) meshes.
    pub fn append_alphatest_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.alphatest_meshes.append(&mut meshes);
    }
    /// Dynamic (forge/object) meshes — set whenever the object set changes.
    pub fn set_dynamic_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.dynamic_meshes = meshes;
    }

    /// Cutout objects (tree canopies/plants) drawn through the alpha-test pass.
    pub fn set_dynamic_cutout_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.dynamic_cutout_meshes = meshes;
    }

    /// Holo/objective objects (hill markers, globe) drawn through the additive holo pass.
    pub fn set_dynamic_holo_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.dynamic_holo_meshes = meshes;
    }
    /// Object markers (spawn Spartans, hill globes, kill/safe boundary shells) drawn through the
    /// alpha-blended holo pass so their translucent SHAPE stays readable at every angle.
    pub fn set_dynamic_holo_solid_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.dynamic_holo_solid_meshes = meshes;
    }
    /// This frame's effect particle batches (object lifts / effect scenery); drawn after the
    /// additive BSP pass with the per-mesh particle blend routing (see MeshRenderer::draw_particles).
    pub fn set_particle_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.particle_meshes = meshes;
    }
    /// Transparent OBJECT parts (forge glass windows) drawn through the alpha-blend pass.
    pub fn set_dynamic_blend_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.dynamic_blend_meshes = meshes;
    }
    /// Water (shader_water) meshes — set once per map load; drawn transparent.
    pub fn set_water_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.water_meshes = meshes;
        self.bump_static();
    }
    /// The map's water volume planes (for camera-submerged detection). Set once per map load;
    /// the fullscreen fog pass activates automatically when the camera drops below a plane
    /// within its XY footprint.
    pub fn set_water_planes(&mut self, planes: Vec<WaterPlane>) {
        self.water_planes = planes;
    }
    /// Accumulate water planes across chunked load steps (mirrors append_water_meshes).
    pub fn append_water_planes(&mut self, mut planes: Vec<WaterPlane>) {
        self.water_planes.append(&mut planes);
    }
    /// The map's authored planar fog volumes (screen-space depth composite). Set once per map.
    pub fn set_planar_fog_volumes(&mut self, vols: Vec<PlanarFogVolume>) {
        self.planar_fog_volumes = vols;
    }
    /// Enable/disable the underwater fog pass at runtime (a Forge water-column trigger or a UI
    /// toggle would call this). See the `underwater_enabled` field for why it is gated.
    pub fn set_underwater_enabled(&mut self, on: bool) {
        self.underwater_enabled = on;
    }
    /// The (fog_color, murk) of the water plane the camera is submerged under, when the
    /// underwater fog pass is enabled.
    fn underwater_fog(&self, cam_pos: glam::Vec3) -> Option<([f32; 3], f32)> {
        if !self.underwater_enabled {
            return None;
        }
        self.submerged_plane(cam_pos).map(|p| (p.color, p.murk))
    }
    /// The highest water plane the camera is physically beneath (within its XY footprint),
    /// regardless of `underwater_enabled` (the water shader's fresnel flip is a physical fact,
    /// independent of the optional full-screen underwater-fog pass toggle).
    fn submerged_plane(&self, cam_pos: glam::Vec3) -> Option<&WaterPlane> {
        // Surface height of triangle `t` at (px,py) when the point is inside its XY projection
        // (barycentric sign test).
        let surface_z = |px: f32, py: f32, t: &[f32; 9]| -> Option<f32> {
            let (x0, y0, z0, x1, y1, z1, x2, y2, z2) = (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
            let det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            if det.abs() < 1e-9 { return None; }
            let l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det;
            let l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det;
            let l2 = 1.0 - l0 - l1;
            if l0 < -1e-4 || l1 < -1e-4 || l2 < -1e-4 { return None; }
            Some(l0 * z0 + l1 * z1 + l2 * z2)
        };
        let mut best: Option<&WaterPlane> = None;
        for p in &self.water_planes {
            // Cheap rejects first: below the surface, inside the XY AABB, and a better candidate.
            if cam_pos.z >= p.z
                || cam_pos.x < p.min[0] || cam_pos.x > p.max[0]
                || cam_pos.y < p.min[1] || cam_pos.y > p.max[1]
                || !best.map_or(true, |b| p.z > b.z)
            {
                continue;
            }
            // Precise: the camera's XY must fall inside the actual water surface (not just its AABB),
            // so land sitting below the water level within the bounding box is NOT flagged underwater,
            // and only below the triangle's interpolated surface height (+ a small margin).
            if p.tris.iter().any(|t| surface_z(cam_pos.x, cam_pos.y, t).map_or(false, |z| cam_pos.z < z - 0.02)) {
                best = Some(p);
            }
        }
        best
    }
    /// Terrain (shader_terrain) meshes — set once per map load.
    pub fn set_terrain_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.terrain_meshes = meshes;
        self.bump_static();
    }
    /// Per-map: does the scenario author atmosphere (fogg)? Gates the procedural horizon
    /// gradient so space maps (no fogg, e.g. Zealot) show only the sky model over a
    /// near-black clear instead of an invented blue gradient.
    pub fn set_sky_atmosphere(&mut self, has_atmosphere: bool) {
        self.sky_has_atmosphere = has_atmosphere;
    }

    /// Mark this map as a SPACE map (starfield/planet dome over a void). Drives the black
    /// clear and gradient skip in `render`. Set per-map from `SceneController::is_space_sky`.
    pub fn set_space_sky(&mut self, space: bool) {
        self.space_sky = space;
    }

    /// Sky render_model segments (mesh, blend mode, draw-order key) — set once per map load.
    /// Empty → only the procedural gradient sky is drawn. Triggers a recapture of the env cube.
    pub fn set_sky_meshes(&mut self, segments: Vec<(GpuMesh, u8, f32)>) {
        self.sky_segments = segments;
        self.env_capture_pending.set(true);
    }
    /// Alpha-test (foliage cutout) meshes — set/clear per map load.
    pub fn set_alphatest_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.alphatest_meshes = meshes;
        self.bump_static();
    }
    /// Append a chunk of decorator (grass/flower) meshes — drawn through the
    /// multiplicative foliage pipeline (engine decorator model, not mesh_shade).
    pub fn append_foliage_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.foliage_meshes.append(&mut meshes);
    }
    /// Decorator (grass/flower) meshes — set/clear per map load.
    pub fn set_foliage_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.foliage_meshes = meshes;
        self.bump_static();
    }
    /// Glass/translucent (alpha-blend) meshes — set/clear per map load.
    pub fn set_blend_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.blend_meshes = meshes;
    }
    /// Per-buffer draw state (vertex counts of the overlay lanes, mesh counts per transparent
    /// list) for the headless diagnostics.
    pub fn debug_draw_counts(&self) -> String {
        format!(
            "zone_vtx={} overlay_vtx={} xray_vtx={} highlight_vtx={} gizmo_ln={} gizmo_tri={} | water={} blend={} additive={} decal={} | dyn_op={} dyn_cut={} dyn_holo={} dyn_holo_solid={} dyn_blend={}",
            self.zone_vcount, self.overlay_vcount, self.xray_vcount, self.highlight_vcount, self.gizmo_vcount, self.gizmo_tri_vcount,
            self.water_meshes.len(), self.blend_meshes.len(), self.additive_meshes.len(), self.decal_meshes.len(),
            self.dynamic_meshes.len(), self.dynamic_cutout_meshes.len(), self.dynamic_holo_meshes.len(),
            self.dynamic_holo_solid_meshes.len(), self.dynamic_blend_meshes.len(),
        )
    }
    /// How much of the object set goes into the sun shadow pass — (caster meshes,
    /// caster instances, total object meshes, total object instances) over the opaque + cutout
    /// batches `draw_shadow` walks. Lets a script/headless run prove which objects cast.
    pub fn shadow_caster_counts(&self) -> (usize, u32, usize, u32) {
        let mut out = (0usize, 0u32, 0usize, 0u32);
        for m in self.dynamic_meshes.iter().chain(self.dynamic_cutout_meshes.iter()) {
            out.2 += 1;
            out.3 += m.instance_count;
            if m.casts_shadow {
                out.0 += 1;
                out.1 += m.instance_count;
            }
        }
        out
    }
    /// Decals (mesh, blend bucket) — set/clear per map load.
    pub fn set_decal_meshes(&mut self, meshes: Vec<(GpuMesh, u8)>) {
        self.decal_meshes = meshes;
        self.bump_static();
    }
    pub fn append_decal_meshes(&mut self, mut meshes: Vec<(GpuMesh, u8)>) {
        self.decal_meshes.append(&mut meshes);
    }
    /// Append a chunk of glass/translucent meshes.
    pub fn append_blend_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.blend_meshes.append(&mut meshes);
    }
    /// Additive/hologram (blend==1) meshes — set/clear + append per map load.
    pub fn set_additive_meshes(&mut self, meshes: Vec<GpuMesh>) {
        self.additive_meshes = meshes;
        self.bump_static();
    }
    pub fn append_additive_meshes(&mut self, mut meshes: Vec<GpuMesh>) {
        self.additive_meshes.append(&mut meshes);
    }

    /// Smoke-test scene: a couple of lit cubes, so the mesh pipeline is
    /// exercised before the object table is wired.
    pub fn set_demo_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (verts, indices) = mesh::test_cube();
        let instances = [
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            Mat4::from_translation(Vec3::new(4.0, 2.0, 1.0)) * Mat4::from_scale(Vec3::splat(1.5)),
            Mat4::from_translation(Vec3::new(-3.0, 3.0, 1.0)),
        ];
        let m = self
            .mesh_renderer
            .upload_mesh(device, queue, &verts, &indices, &instances, None, None, None, None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.0, 1.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None);
        self.dynamic_meshes = vec![m];
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The offscreen color texture's raw (Unorm) view.
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }
    /// sRGB view of the color texture — register THIS with egui (not color_view) so
    /// egui's mandatory gamma_from_linear() doesn't double-encode our sqrt-encoded
    /// output (which made the viewport brighter than the PNG capture).
    pub fn color_view_srgb(&self) -> &wgpu::TextureView {
        &self.color_view_srgb
    }

    /// Read the current scene color texture back to CPU RGBA8. Returns
    /// (rgba, width, height). Handles the 256-byte bytes-per-row alignment.
    pub fn capture_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.size;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = ((unpadded + align - 1) / align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((unpadded * h) as usize);
        for row in 0..h {
            let start = (row * padded) as usize;
            rgba.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        buf.unmap();
        (rgba, w, h)
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let size = (size.0.max(1), size.1.max(1));
        if size == self.size {
            return;
        }
        let Targets {
            color_tex: c,
            color_view: cv,
            depth_tex: dt,
            depth_view: dv,
            depth_copy_tex: dct,
            depth_copy_view: dcv,
            hdr_view: hv,
            hdr_tex: ht,
        } = make_targets(device, size);
        let bloom = make_bloom_pyramid(device, size);
        self.color_view_srgb = c.create_view(&wgpu::TextureViewDescriptor {
            format: Some(COLOR_FORMAT_SRGB),
            ..Default::default()
        });
        self.color_tex = c;
        self.color_view = cv;
        self.depth_view = dv;
        self.depth_tex = dt;
        self.depth_copy_tex = dct;
        self.depth_copy_view = dcv;
        // Rebuild the camera bind group so binding 5 points at the RESIZED depth copy.
        self.rebuild_camera_bg(device);
        self.lensfx.rebind(device, &self.camera_buf, &self.light_buf, &self.depth_copy_view, &self.lum_view);
        // Rebuild the fog pass bind groups so binding 1 points at the resized depth copy.
        self.underwater_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("underwater-bg"),
            layout: &self.underwater_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.underwater_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_copy_view) },
            ],
        });
        self.planar_fog_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar-fog-bg"),
            layout: &self.planar_fog_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.planar_fog_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_copy_view) },
            ],
        });
        // lum meter re-binds the new HDR view (1×1 target itself is size-independent).
        self.lum_bg = make_lum_bg(device, &self.lum_bgl, &hv, &self.post_sampler, &self.post_buf, &self.h4pre_view);
        self.h4pre_bg = make_tex_bg(device, &self.tex_bgl, &hv, &self.post_sampler);
        self.post_bg = make_post_bg(device, &self.post_bgl, &hv, &bloom.out, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bright_in_bg = make_post_bg(device, &self.post_bgl, &hv, &hv, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_q_a = make_post_bg(device, &self.post_bgl, &bloom.q_a, &bloom.e_b, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_q_b = make_post_bg(device, &self.post_bgl, &bloom.q_b, &bloom.q_b, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_e_a = make_post_bg(device, &self.post_bgl, &bloom.e_a, &bloom.t_a, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_e_a_solo = make_post_bg(device, &self.post_bgl, &bloom.e_a, &bloom.e_a, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_e_b = make_post_bg(device, &self.post_bgl, &bloom.e_b, &bloom.e_b, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_t_a = make_post_bg(device, &self.post_bgl, &bloom.t_a, &bloom.t_a, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.bg_t_b = make_post_bg(device, &self.post_bgl, &bloom.t_b, &bloom.t_b, &self.post_sampler, &self.post_buf, &self.lum_view);
        self.hdr_view = hv;
        self.hdr_tex = ht;
        self.bloom = bloom;
        self.size = size;
    }

    /// Render one frame to the offscreen texture. `time` (seconds) drives the
    /// decorator wind sway.
    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera, time: f32) {
        let t_render_entry = std::time::Instant::now(); // HMS_RENDER_PROF prologue lap
        let aspect = self.size.0 as f32 / self.size.1 as f32;
        let vp = camera.view_proj(aspect);
        // Camera-underwater flag → cam.time.y (water_shading:1112). The water shader flips the
        // fresnel normal (−n) when the eye is below a water surface. Physical test (submerged_plane),
        // independent of the optional underwater-fog pass. HMS_FORCE_UW forces it on.
        let underwater_flag = if self.submerged_plane(camera.pos).is_some() || std::env::var("HMS_FORCE_UW").is_ok() { 1.0 } else { 0.0 };
        // cam.time = [seconds, underwater flag, FOG_WU override (0 = shader const; HMS_FOG_WU), unused]
        let fog_wu = if self.fog_wu > 0.0 { self.fog_wu } else { std::env::var("HMS_FOG_WU").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0) };
        let cam = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            cam_pos: [camera.pos.x, camera.pos.y, camera.pos.z, 1.0],
            time: [time, underwater_flag, fog_wu, 0.0],
            inv_view_proj: vp.inverse().to_cols_array_2d(),
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        // #wire-visible  Selection-wireframe params: xy = 1 / HDR target size (the halo offset is
        // in HDR pixels, and the HDR target is SS x the output), z = halo radius, w = the exposed
        // value the core aims for. Cheap enough to write every frame.
        {
            const WIRE_HALO_R: f32 = 2.2;   // HDR px (SS = 2 -> ~1.1 output px of dark outline)
            // Exposed units the core's BRIGHTEST channel aims for. Deliberately BELOW the
            // tonemap's saturation point: aiming at 1.0+ clips red and green together and the
            // gold core comes out cream-white -- exactly the colour of the Halo 4 Forge pieces it
            // has to stand out against. 0.55 lands the core as a solid, saturated gold whatever
            // the scene exposure, and the black halo (not the core's brightness) carries the
            // contrast against a blown-out background.
            const WIRE_CORE_TARGET: f32 = 1.00;
            let (tw, th) = ((self.size.0 * SS).max(1) as f32, (self.size.1 * SS).max(1) as f32);
            queue.write_buffer(&self.wire_buf, 0, bytemuck::cast_slice(&[1.0 / tw, 1.0 / th, WIRE_HALO_R, WIRE_CORE_TARGET]));
        }
        // Sun lens flare: project the (per-map) sun direction to screen; if it's on
        // screen and in front, build a transient NDC vertex buffer of additive sprites
        // strung along the sun→centre axis (pos = sunNdc·(1-axisOffset)). Edge-faded.
        let mut lens_vbuf: Option<wgpu::Buffer> = None;
        let mut lens_vcount: u32 = 0;
        if !self.lens_elements.is_empty() {
            use wgpu::util::DeviceExt;
            let far = camera.pos + self.sun_dir * 100000.0;
            let clip = vp * far.extend(1.0);
            if clip.w > 0.0 {
                let sx = clip.x / clip.w;
                let sy = clip.y / clip.w;
                if sx.abs() < 1.3 && sy.abs() < 1.3 {
                    let edge = (1.0 - sx.abs().max(sy.abs())).clamp(0.0, 1.0);
                    let mut verts: Vec<f32> = Vec::new();
                    let n = self.lens_elements.len() / 8;
                    for i in 0..n {
                        let e = &self.lens_elements[i * 8..i * 8 + 8];
                        let t = e[0];
                        // modulation (e[7]) scales brightness; tint_power (e[6]) shapes the
                        // sprite falloff per-vertex in the shader (tight cores vs soft halos).
                        let bright = e[2] * e[7] * edge * 0.5;
                        let tint_power = e[6];
                        let cx = sx * (1.0 - t);
                        let cy = sy * (1.0 - t);
                        let hy = (e[1] * 0.06).clamp(0.005, 0.2);
                        let hx = hy / aspect;
                        let col = [e[3] * bright, e[4] * bright, e[5] * bright, 1.0f32];
                        let corner = |dx: f32, dy: f32, u: f32, v: f32| {
                            [cx + dx * hx, cy + dy * hy, u, v, col[0], col[1], col[2], col[3], tint_power]
                        };
                        let c00 = corner(-1.0, -1.0, 0.0, 0.0);
                        let c10 = corner(1.0, -1.0, 1.0, 0.0);
                        let c11 = corner(1.0, 1.0, 1.0, 1.0);
                        let c01 = corner(-1.0, 1.0, 0.0, 1.0);
                        for cv in [c00, c10, c11, c00, c11, c01] {
                            verts.extend_from_slice(&cv);
                        }
                    }
                    if !verts.is_empty() {
                        lens_vbuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("lens-vb"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        }));
                        lens_vcount = (verts.len() / 9) as u32;
                    }
                    if std::env::var("HMS_DIAG").is_ok() {
                        eprintln!("HMS_DIAG LENS n={} sx={:.2} sy={:.2} vcount={}", n, sx, sy, lens_vcount);
                    }
                }
            }
        }
        // World lens flares — CPU projection/fades now, GPU occlusion query at draw time.
        let lensfx_batches = {
            let view = Mat4::look_at_rh(camera.pos, camera.pos + camera.forward(), Vec3::Z);
            let proj = Mat4::perspective_rh(camera.fov_y, aspect.max(0.01), camera.near, camera.far);
            self.lensfx.build(device, camera.pos, camera.forward(), &view, &proj, (self.size.0 * SS, self.size.1 * SS), std::env::var("HMS_LENSFX_DIAG").is_ok())
        };
        // Camera-centred sun shadow frustum, updated per-frame. A tight radius keeps the
        // 2048² shadow map crisp near the viewer (fitting the whole map AABB — r~5000 —
        // makes shadows too coarse to see). Same to-sun dir as fog/lighting.
        {
            let sun = self.sun_dir;
            // Camera-CENTERED (not offset ahead), wide + deep enough to cover the viewed
            // terrain (hw=450, depth 1500 each way). A too-small/mis-centered box makes
            // shadow_strength read "lit" for all visible terrain from overviews.
            let hw = 450.0f32;
            let back = 1500.0f32;
            let ctr = camera.pos;
            let eye = ctr + sun * back;
            let view = Mat4::look_at_rh(eye, ctr, Vec3::Z);
            let proj = Mat4::orthographic_rh(-hw, hw, -hw, hw, 0.1, back * 2.0);
            let mut lvp = (proj * view).to_cols_array();
            // Light uniform (see LIGHT_UNIFORM_BYTES): mat4 light_view_proj, sun_dir, sun_tint,
            // ambient_tint, sky_ambient, dbg (Lighting Lab multipliers), expo, expo2, then the
            // Halo 4 shadow tail (h4s0, h4s1, cascade view_proj, casc0, casc1).
            let mut lb = [0.0f32; 80];
            // Halo 4: the sun shadow map is the engine's "forge lightmap burn" map - an
            // orthographic depth map of the OBJECT casters only (halo4.dll sub_180344BE8: the tile
            // bounds are the union of the registered objects' bounding spheres in sun space, 512
            // texels per 12.8 wu = 0.025 wu/texel, depth normalized over the bounds), whose 4x4
            // bilinear-weighted PCF is MULTIPLIED into the BSP lightmap's analytic sun visibility
            // (`forge_lightmap_render_sun_structure` PS, blend DEST_COLOR x ZERO) - see
            // docs/halo4_lighting_model.md §10. HMS renders it every frame (the editor moves objects)
            // at 2048² over the caster AABB: the engine density until the casters span more than
            // 51.2 wu, coarser beyond (tiling is not replicated). The box is snapped to texel
            // multiples like the engine's cascades (sub_18035F368 fmodf) so it does not shimmer.
            if let Some((bmin, bmax)) = self.h4_shadow_bounds {
                let l = sun; // to-sun; the map's shadow direction is -l
                let up = if l.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
                let x = up.cross(l).normalize();
                let y = l.cross(x).normalize();
                let corners = [
                    Vec3::new(bmin.x, bmin.y, bmin.z), Vec3::new(bmax.x, bmin.y, bmin.z), Vec3::new(bmin.x, bmax.y, bmin.z), Vec3::new(bmax.x, bmax.y, bmin.z),
                    Vec3::new(bmin.x, bmin.y, bmax.z), Vec3::new(bmax.x, bmin.y, bmax.z), Vec3::new(bmin.x, bmax.y, bmax.z), Vec3::new(bmax.x, bmax.y, bmax.z),
                ];
                let (mut x0, mut x1, mut y0, mut y1, mut d0, mut d1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for c in corners {
                    let (px, py, pd) = (c.dot(x), c.dot(y), c.dot(l));
                    x0 = x0.min(px); x1 = x1.max(px); y0 = y0.min(py); y1 = y1.max(py); d0 = d0.min(pd); d1 = d1.max(pd);
                }
                // square footprint at the engine texel (0.025 wu) or coarser when the casters need it
                let side = ((x1 - x0).max(y1 - y0) + 1.0).max(0.025 * 64.0);
                let texel = (side / SHADOW_MAP_SIZE as f32).max(0.025);
                let side = texel * SHADOW_MAP_SIZE as f32;
                let cx = ((x0 + x1) * 0.5 / texel).round() * texel;
                let cy = ((y0 + y1) * 0.5 / texel).round() * texel;
                // depth range = the casters' extent along the light (+0.5 wu each side); receivers
                // beyond the far plane (the ground below every caster) are handled by the shader's
                // plain compare, exactly like the engine's burn PS (no far-plane clip).
                let near_d = d1 + 0.5;
                let far_d = d0 - 0.5;
                let eye = x * cx + y * cy + l * near_d;
                let view = Mat4::look_at_rh(eye, eye - l, y);
                let proj = Mat4::orthographic_rh(-side * 0.5, side * 0.5, -side * 0.5, side * 0.5, 0.0, near_d - far_d);
                lvp = (proj * view).to_cols_array();
                // h4s0 = (texel uv, bias inner, bias edge, bias corner): the burn PS's per-tap receiver
                // biases in normalized depth (0.002 / 0.002·sqrt(1.25) / 0.004).
                lb[44] = 1.0 / SHADOW_MAP_SIZE as f32; lb[45] = 0.002; lb[46] = 0.0022360680; lb[47] = 0.004;
                // h4s1 = (burn active, depth range wu, texel wu, 0)
                lb[48] = 1.0; lb[49] = near_d - far_d; lb[50] = texel; lb[51] = 0.0;
                self.h4_shadow_diag.set([texel, near_d - far_d, side, 1.0]);
            }
            // The floating-shadow cascade box (halo4.dll sub_18035F368): centre = viewer +
            // horizontal forward * offset + to-sun * sun_offset, snapped to texel multiples across the
            // light, `2 * half_width` square, `length` deep along the light centred on it (the engine
            // takes the player's unit; the editor has none, so the camera is the viewer).
            if let Some(cfg) = self.h4_cascade {
                let l = sun;
                let up = if l.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
                let x = up.cross(l).normalize();
                let y = l.cross(x).normalize();
                let f = camera.forward();
                let fwd = Vec3::new(f.x, f.y, 0.0).normalize_or_zero();
                let res = cfg.resolution.clamp(64, H4_CASCADE_TEX) as f32;
                let texel = 2.0 * cfg.half_width / res;
                let mut c = camera.pos + fwd * cfg.offset + l * cfg.sun_offset;
                c -= x * ((c.dot(x)) % texel) + y * ((c.dot(y)) % texel);
                let eye = c + l * (cfg.length * 0.5);
                let view = Mat4::look_at_rh(eye, eye - l, y);
                let proj = Mat4::orthographic_rh(-cfg.half_width, cfg.half_width, -cfg.half_width, cfg.half_width, 0.0, cfg.length);
                let cvp = (proj * view).to_cols_array();
                lb[52..68].copy_from_slice(&cvp);
                lb[68] = cfg.half_width; lb[69] = cfg.length; lb[70] = cfg.bias; lb[71] = cfg.filter / 800.0;
                lb[72] = cfg.taps as f32; lb[73] = res; lb[74] = 1.0; lb[75] = 0.0;
                // the cascade caster pass reads its matrix from the light_view_proj slot of its own buffer
                let mut cb = lb;
                cb[..16].copy_from_slice(&cvp);
                queue.write_buffer(&self.h4_cascade_light_buf, 0, bytemuck::cast_slice(&cb));
            }
            lb[..16].copy_from_slice(&lvp);
            lb[16] = sun.x; lb[17] = sun.y; lb[18] = sun.z;
            lb[19] = self.obj_dir_strength; // per-map object directional soft-shade strength
            // sun_tint.w (lb[23]) is the self-illum ILLUM_SCALE hook (engine g_alt_exposure.r,
            // entry_points.hlsl:473); 1.0 = no-op (mesh_shade multiplies it into illum_scale).
            lb[20] = self.sun_tint.x * self.sun_mult; lb[21] = self.sun_tint.y * self.sun_mult; lb[22] = self.sun_tint.z * self.sun_mult; lb[23] = 1.0;
            // ambient_tint.w (lb[27]) is not read by any shader.
            lb[24] = self.ambient_tint.x * self.ambient_mult; lb[25] = self.ambient_tint.y * self.ambient_mult; lb[26] = self.ambient_tint.z * self.ambient_mult;
            lb[27] = 1.0;
            // sky_ambient: real sky_light SH ambient rgb + enable(w). Off (w=0) → no-op.
            lb[28] = self.sky_ambient[0] * self.ambient_mult; lb[29] = self.sky_ambient[1] * self.ambient_mult;
            lb[30] = self.sky_ambient[2] * self.ambient_mult; lb[31] = self.sky_ambient[3];
            // dbg: Lighting Lab live multipliers sun_mult, ambient_mult, lightmap_mult (1.0 = no-op;
            // wired to the UI sliders so the user can isolate a lighting term) + dbg.w = how much of
            // this map the lightmapper says the sun reaches (mean of the baked per-vertex
            // sun-visibility population). Gates the GLASS sun lanes, which carry no per-surface
            // visibility of their own. 1.0 = unknown/atlas-only map.
            lb[32] = self.sun_mult; lb[33] = self.ambient_mult; lb[34] = self.lm_mult;
            lb[35] = self.sun_reach;
            // expo/expo2: exposure meter parameters mirrored from the post uniform so mesh/sky shaders
            // can reproduce the resolve gain (E) and derive g_alt_exposure.r = 2^((1-s)(P-E)).
            let ap = self.illum_params.get();
            lb[36] = ap[0]; lb[37] = ap[1]; lb[38] = ap[2]; lb[39] = ap[3];
            lb[40] = ap[4]; lb[41] = self.fixed_exposure; lb[42] = ap[5]; lb[43] = ap[6];
            queue.write_buffer(&self.light_buf, 0, bytemuck::cast_slice(&lb));
            // Post grade vec4 (offset 16): the engine has NO whole-frame post grade (final
            // color_matrix ≈ identity for every Reach cfxs); per-scenario warmth is per-domain
            // lighting applied to lit surfaces only, through `scene_grade(rgb)` in the mesh /
            // terrain / foliage shaders (sky and water keep their own colour). So the grade
            // is neutral; grade.w carries the fixed g_exposure (0 → metered auto-exposure). This
            // write runs per-frame, so it must re-use the stored field rather than a literal 0.
            let grade = [1.0f32, 1.0, 1.0, self.fixed_exposure];
            queue.write_buffer(&self.post_buf, 16, bytemuck::cast_slice(&grade));
        }
        let sky = SkyUniform {
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            cam_pos: [camera.pos.x, camera.pos.y, camera.pos.z, 1.0],
            sun_dir: [self.sun_dir.x, self.sun_dir.y, self.sun_dir.z, 0.0],
            horizon: [self.sky_tint[0], self.sky_tint[1], self.sky_tint[2],
                      if self.sky_has_atmosphere { 1.0 } else { 0.0 }],
        };
        queue.write_buffer(&self.sky_buf, 0, bytemuck::bytes_of(&sky));

        // Static-list draws come from cached render bundles (re-recorded only when a list or the
        // camera bind group changed). HMS_RENDER_PROF=1 prints a per-section CPU breakdown of this
        // function every 120 frames (print-only diagnostic; the env var is read once).
        static RENDER_PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let prof = *RENDER_PROF.get_or_init(|| std::env::var("HMS_RENDER_PROF").is_ok());
        // slots: 0 prologue (uniform writes + lens), 1 bundle (re)record, 2 shadow+sky passes, 3 scene
        // pass, 4 transparent pass, 5 post (fog/lum/bloom/tonemap), 6 submit, 9-12 the scene pass's
        // bundle executes (recorded lazily: ~0), 13 scene pass END (where wgpu-core replays every
        // draw into hal and merges the resource trackers -- this is the real per-draw CPU cost).
        let mut prof_t = [0f32; 14];
        let mut prof_clock = t_render_entry;
        let mut lap = |slot: usize, t: &mut std::time::Instant| { if prof { prof_t[slot] = t.elapsed().as_secs_f32() * 1000.0; *t = std::time::Instant::now(); } };
        lap(0, &mut prof_clock);
        let bundles = self.static_bundles(device);
        lap(1, &mut prof_clock);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scene") });
        // Real-time GI: advance the probe field one step before anything samples it this frame.
        if self.rtgi.enabled && self.rtgi.has_scene() && std::env::var("HMS_RTGI_FREEZE").is_err() {
            let parts = self.rtgi.parts_per_frame();
            let part = self.rtgi.updates.get() % parts;
            self.rtgi.dispatch_part(&mut encoder, queue, None, parts, part);
        }
        // One-shot env-cube SKY CAPTURE. Render the dome into the 6 cube faces so glossy
        // surfaces reflect the MAP's actual sky (clouds/nebula/planet), not a gradient. Runs once
        // per map (flag set in set_sky_meshes); the cube-writes precede the opaque pass in this
        // encoder, so the reflection sampled by mesh_shade is up to date the same frame.
        if self.env_capture_pending.get() && !self.sky_segments.is_empty() {
            if let Some(cube) = self.mesh_renderer.env_cube_texture() {
                // The sky pipelines were built with a depth-stencil attachment (compare Always, no
                // write) — the capture pass must supply a matching depth target even though it's
                // unused. One small depth texture reused across the 6 faces.
                let cap_depth = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("env-cube-depth"),
                    size: wgpu::Extent3d { width: mesh::EV_CUBE_SZ, height: mesh::EV_CUBE_SZ, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let cap_depth_view = cap_depth.create_view(&wgpu::TextureViewDescriptor::default());
                // Render into a TEMP cube (the bound env cube is sampled by the sky materials, so it
                // can't be a colour target in the same pass), then copy temp → env cube.
                let tmp = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("env-cube-tmp"),
                    size: wgpu::Extent3d { width: mesh::EV_CUBE_SZ, height: mesh::EV_CUBE_SZ, depth_or_array_layers: 6 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: SCENE_HDR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                for f in 0..6u32 {
                    let fv = tmp.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("env-cube-face"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: f,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
                    let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("env-cube-capture"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &fv,
                            resolve_target: None,
                            // Clear to BLACK, not a flat blue: a blue clear would make every
                            // "uncaptured" cube direction read as lit, so water/glossy surfaces mirror
                            // one constant colour. Black lets the shader detect uncaptured directions
                            // and fall back to the procedural sky gradient, while any real dome/cloud
                            // panels drawn below overwrite their texels.
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &cap_depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.mesh_renderer.draw_sky(&mut sp, &self.env_face_cams[f as usize], &self.sky_segments);
                }
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo { texture: &tmp, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyTextureInfo { texture: cube, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::Extent3d { width: mesh::EV_CUBE_SZ, height: mesh::EV_CUBE_SZ, depth_or_array_layers: 6 },
                );
                // Generate the cube mip chain (box-downsample per face, previous mip → current) so the
                // roughness-LOD reflection sampling actually blurs. Reuses the bloom fs_downsample
                // pipeline (SCENE_HDR_FORMAT, tex_bgl). Per-face (minor edge seams, fine for reflections).
                for m in 1..mesh::EV_CUBE_MIPS {
                    for f in 0..6u32 {
                        let src = cube.create_view(&wgpu::TextureViewDescriptor {
                            label: Some("env-mip-src"), dimension: Some(wgpu::TextureViewDimension::D2),
                            base_mip_level: m - 1, mip_level_count: Some(1),
                            base_array_layer: f, array_layer_count: Some(1), ..Default::default()
                        });
                        let dst = cube.create_view(&wgpu::TextureViewDescriptor {
                            label: Some("env-mip-dst"), dimension: Some(wgpu::TextureViewDimension::D2),
                            base_mip_level: m, mip_level_count: Some(1),
                            base_array_layer: f, array_layer_count: Some(1), ..Default::default()
                        });
                        let bg = make_tex_bg(device, &self.tex_bgl, &src, &self.post_sampler);
                        let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("env-cube-mipgen"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &dst, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                        });
                        mp.set_pipeline(&self.env_mip_pipeline);
                        mp.set_bind_group(0, &bg, &[]);
                        mp.draw(0..3, 0..1);
                    }
                }
                self.env_capture_pending.set(false);
            }
        }
        // Shadow pass: render casters from the sun POV into the shadow depth map. ONLY OBJECTS
        // cast (engine-exact): the engine's sun shadow is the screen-space `shadow_mask` of
        // DYNAMIC casters (shadows\shadow_mask.hlsl_include: `.a` gates the analytical light,
        // `.r` darkens the dominant lobe); the world's own sun occlusion is BAKED — DM.g
        // visibility (vmf[0].w) at lightmap resolution — and the BSP is never rendered into the
        // sun shadow map. Drawing the BSP/terrain/foliage as casters would (a) darken every
        // sun-facing interior surface's BAKED dominant lobe by `darken` (0.55 floor) on top of the
        // bake's own occlusion, and (b) draw crisp slat/ceiling shadow streaks through openings
        // that the engine shows only as a soft lightmap-resolution visibility patch (Zealot's
        // "sun through a window"). Objects still drop grounding shadows onto the world.
        if self.show_bsp {
            let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if self.h4_shadow_bounds.is_some() {
                // Halo 4: back-face caster pass over the editor objects (dynamic lanes) AND the
                // scenario objects flagged `h4_caster` in the static lanes (build_meshes). Reach
                // never sets the flag, so its static BSP lanes stay out.
                self.mesh_renderer.draw_shadow_h4(
                    &mut sp,
                    &self.shadow_pass_bg,
                    &[&self.dynamic_meshes, &self.dynamic_cutout_meshes],
                    &[&self.static_meshes, &self.alphatest_meshes],
                    false,
                );
            } else {
                self.mesh_renderer.draw_shadow(
                    &mut sp,
                    &self.shadow_pass_bg,
                    &[
                        &self.dynamic_meshes,
                        &self.dynamic_cutout_meshes,
                    ],
                );
            }
        }
        // The Halo 4 floating-shadow cascade depth pass (halo4.dll sub_18035ECB4): the objects
        // AND the BSP (the engine takes the sbsp instances whose sphere meets the box; HMS draws every
        // static mesh - the clusters too, a superset) from the sun into the res x res cascade map.
        if self.show_bsp && self.h4_cascade.is_some() {
            let res = self.h4_cascade.map_or(H4_CASCADE_TEX, |c| c.resolution.clamp(64, H4_CASCADE_TEX)) as f32;
            let mut cp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("h4-cascade-shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.h4_cascade_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            cp.set_viewport(0.0, 0.0, res, res, 0.0, 1.0);
            cp.set_scissor_rect(0, 0, res as u32, res as u32);
            self.mesh_renderer.draw_shadow_h4(
                &mut cp,
                &self.h4_cascade_pass_bg,
                &[&self.dynamic_meshes, &self.dynamic_cutout_meshes],
                &[&self.static_meshes, &self.alphatest_meshes],
                true,
            );
        }
        // The sky render_model draws in its OWN pass with its own depth (opaque sky units
        // depth-test + write, translucent units depth-test without write — mesh.rs draw_sky),
        // and the depth buffer is then CLEARED again for the world pass, so world geometry
        // always draws over every sky panel (the sky model's radius is not a world distance).
        // Colour is cleared here and LOADED by the scene pass.
        {
            let mut sky_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Space maps: clear to TRUE black so the void reads as space and
                        // auto-exposure has nothing to lift where there's no sky geometry (0×gain=0).
                        // Atmospheric maps keep the dim blue clear behind their horizon gradient.
                        load: wgpu::LoadOp::Clear(if self.space_sky {
                            wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
                        } else {
                            wgpu::Color { r: 0.05, g: 0.06, b: 0.09, a: 1.0 }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Prefer the real Reach sky render_model (dome + cloud panels); fall back to the
            // procedural gradient when the map has no resolvable sky.
            if self.show_sky {
                // Outdoor atmosphere: on maps that authored atmosphere_fog, paint a horizon
                // gradient colored from the MAP's OWN fog sky tint (no invented blue, NO sun
                // disc — see SKY_WGSL). This is the map's real atmosphere, so Forge World gets
                // its daytime sky instead of the near-black clear. SPACE maps (no fogg →
                // sky_has_atmosphere=false) skip it entirely and keep the black clear behind
                // their starfield sky model. The real sky render_model then composites on top.
                if self.sky_has_atmosphere && !self.space_sky && std::env::var("HMS_NO_SKYGRAD").is_err() {
                    sky_pass.set_pipeline(&self.sky_pipeline);
                    sky_pass.set_bind_group(0, &self.sky_bg, &[]);
                    sky_pass.draw(0..3, 0..1);
                }
                if !self.sky_segments.is_empty() {
                    self.mesh_renderer.draw_sky(&mut sky_pass, &self.camera_bg, &self.sky_segments);
                }
            }
        }
        lap(2, &mut prof_clock);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // sky-pass cleared + drew the sky
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // fresh depth for the world (sky depth discarded)
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if self.show_grid {
                pass.set_pipeline(&self.grid_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, self.grid_vbuf.slice(..));
                pass.draw(0..self.grid_vcount, 0..1);
            }

            // BSP + objects on top of the grid. Static lists replay their bundles
            // (execute_bundles resets pass state; every later draw sets its own pipeline + groups).
            if self.show_bsp {
                bundles.opaque.execute(&mut pass);
                lap(9, &mut prof_clock);
            }
            // Imported external geometry (persistent base to forge over) — always drawn.
            bundles.imported.execute(&mut pass);
            // A Halo 4 map's own spawn markers: only while the View toggle is on.
            if self.show_markers && !self.marker_meshes.is_empty() {
                bundles.markers.execute(&mut pass);
            }
            if self.show_terrain {
                bundles.terrain.execute(&mut pass);
                lap(10, &mut prof_clock);
            }
            if self.show_objects {
                self.mesh_renderer
                    .draw(&mut pass, &self.camera_bg, &self.dynamic_meshes);
                // Terrain-shaded objects (Forge rocks) share the dynamic list but need the
                // terrain pipeline; draw() skipped them above.
                if self.dynamic_meshes.iter().any(|m| m.is_terrain) {
                    self.mesh_renderer.draw_terrain(&mut pass, &self.camera_bg, &self.dynamic_meshes);
                }
                // Cutout OBJECTS (tree canopies/plants) through the alpha-test pass so
                // their transparent leaf texels are discarded.
                self.mesh_renderer
                    .draw_alphatest(&mut pass, &self.camera_bg, &self.dynamic_cutout_meshes);
            }
            // Alpha-test foliage (cutout) — opaque pass, discards transparent texels. Draws
            // BEFORE the object markers (transparent pass): the markers are alpha-blended with NO
            // depth-write, so foliage must have written depth first for them to depth-test against
            // it (foliage in FRONT of a marker hides it, foliage BEHIND is hidden).
            if self.show_bsp {
                bundles.alphatest.execute(&mut pass);
                lap(11, &mut prof_clock);
            }
            // Decorators (grass/flowers) — multiplicative foliage pipeline (engine
            // decorator model; no specular/white-sun/sky-fill).
            if self.show_bsp {
                // HMS_DECO_BUDGET=<n>: draw decorator groups nearest-first up to a total instance
                // budget (the engine's decimation is a runtime quality setting, not on disk; a
                // global distance budget over-culls legitimate foliage, so the default is 0 =
                // draw all). The default replays the cached bundle; the opt-in budget needs the
                // per-frame nearest-first walk, so it keeps the live path.
                let budget: u32 = std::env::var("HMS_DECO_BUDGET").ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                if budget == 0 {
                    bundles.foliage.execute(&mut pass);
                    lap(12, &mut prof_clock);
                } else {
                    let eye = [camera.pos.x, camera.pos.y, camera.pos.z];
                    self.mesh_renderer
                        .draw_foliage(&mut pass, &self.camera_bg, &self.foliage_meshes, eye, budget);
                }
            }
            // ---- WATER DEPTH ----
            // End the opaque pass, copy its depth to a sampleable texture, then reopen a
            // transparent pass that LOADS color+depth. Water then samples the opaque bed
            // depth (camera bind group binding 5) to compute depth-based murkiness. The
            // live depth_view stays the attachment (writable), so no read-only conflict.
            let mut t_end = std::time::Instant::now();
            drop(pass);
            lap(13, &mut t_end);
            lap(3, &mut prof_clock);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.depth_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.depth_copy_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::Extent3d { width: self.size.0 * SS, height: self.size.1 * SS, depth_or_array_layers: 1 },
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-transparent-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Transparent water pass (after opaque, no depth write).
            if self.show_water {
                bundles.water.execute(&mut pass);
            }
            // Decals: unlit, per-decal blend (multiply scorch darkens the ground), onto opaque
            // surfaces. Drawn BEFORE the sorted transparents — they sit on opaque geometry
            // (depth-tested at the surface), so a marker/pane in front of a decal must paint over it.
            if self.show_bsp {
                bundles.decals.execute(&mut pass);
            }
            // Object markers (fs_holo_solid), additive holos (fs_holo: hill globes, armor icons,
            // force fields, the option-11 spawn/respawn-zone halograms), BSP glass and transparent
            // OBJECT parts (forge glass panes) draw as ONE back-to-front sequence — per instance,
            // by distance from the eye — each with its own pipeline (additive stays One/One, glass
            // keeps its blend_mode routing), so a pane in front of a marker paints over it and vice
            // versa (none of these write depth). Drawn AFTER water so they render over a water
            // backdrop; depth is LOADED, so opaque geometry / foliage in front still hides them.
            // Water stays first, BSP additive + particles last.
            {
                use crate::mesh::TransparentKind as K;
                let eye = [camera.pos.x, camera.pos.y, camera.pos.z];
                let mut lists: Vec<(&[GpuMesh], K)> = Vec::with_capacity(4);
                if self.show_objects {
                    lists.push((&self.dynamic_holo_solid_meshes, K::HoloSolid));
                    lists.push((&self.dynamic_holo_meshes, K::Additive));
                }
                if self.show_bsp {
                    lists.push((&self.blend_meshes, K::Blend));
                }
                if self.show_objects {
                    lists.push((&self.dynamic_blend_meshes, K::Blend));
                }
                if self.show_markers {
                    lists.push((&self.marker_holo_meshes, K::Additive));
                    lists.push((&self.marker_blend_meshes, K::Blend));
                }
                self.mesh_renderer.draw_transparent_sorted(&mut pass, &self.camera_bg, &lists, eye);
            }
            // Additive holograms/force-fields/energy (blend==1) — additive glow, last.
            if self.show_bsp {
                bundles.additive.execute(&mut pass);
            }
            // Effect particle sprites (grav-lift plasma/sparks, waterfall sheets/mist) —
            // depth-tested, no depth write, per-prt3 blend routing, sorted back-to-front.
            if !self.particle_meshes.is_empty() {
                self.mesh_renderer
                    .draw_particles(&mut pass, &self.camera_bg, &self.particle_meshes, [camera.pos.x, camera.pos.y, camera.pos.z]);
            }
            // World lens flares (after the particles, before the sun flare).
            self.lensfx.draw(&mut pass, &lensfx_batches);
            // Sun lens flare: screen-space additive sprites over the scene.
            if self.show_sky {
                if let Some(vb) = &lens_vbuf {
                    pass.set_pipeline(&self.lens_pipeline);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.draw(0..lens_vcount, 0..1);
                }
            }

            // Translucent zone volume fill (selected object's boundary shape) — before the
            // overlay lines so the bright wireframe outline draws on top of the holographic fill.
            if let Some(zv) = &self.zone_vbuf {
                pass.set_pipeline(&self.zone_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, zv.slice(..));
                pass.draw(0..self.zone_vcount, 0..1);
            }
            // Overlay lines (trigger volumes etc.) + selection highlight box.
            if let Some(ov) = &self.overlay_vbuf {
                pass.set_pipeline(&self.grid_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, ov.slice(..));
                pass.draw(0..self.overlay_vcount, 0..1);
            }
            // #wire-visible  Selection wireframe: instances 0-7 = the black halo ring, 8-11 =
            // the exposure-compensated bright core (WIRE_WGSL). A very large selection (Ctrl+A on a
            // full variant) skips the halo and draws the core alone so the instance count can't run
            // away. `highlight_xray` swaps in the depth-ALWAYS pipeline (View menu).
            if let (Some(hl), false) = (&self.highlight_vbuf, self.highlight_hidden) {
                const WIRE_HALO_MAX_VERTS: u32 = 4_000_000;
                let insts = if self.highlight_vcount > WIRE_HALO_MAX_VERTS { 8..12 } else { 0..12 };
                pass.set_pipeline(if self.highlight_xray { &self.wire_xray_pipeline } else { &self.wire_pipeline });
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_bind_group(1, &self.wire_bg, &[]);
                pass.set_vertex_buffer(0, hl.slice(..));
                pass.draw(0..self.highlight_vcount, insts);
            }
            // X-ray lines (no depth test) -- the soft-ceiling edges through the map.
            if let Some(xv) = &self.xray_vbuf {
                pass.set_pipeline(&self.gizmo_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, xv.slice(..));
                pass.draw(0..self.xray_vcount, 0..1);
            }
            // Transform gizmo LAST, with the always-on-top pipeline so it's grabbable
            // through other geometry. Solid triangles first, then any thin guide lines.
            if let Some(gz) = &self.gizmo_tri_vbuf {
                pass.set_pipeline(&self.gizmo_tri_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, gz.slice(..));
                pass.draw(0..self.gizmo_tri_vcount, 0..1);
            }
            if let Some(gz) = &self.gizmo_vbuf {
                pass.set_pipeline(&self.gizmo_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, gz.slice(..));
                pass.draw(0..self.gizmo_vcount, 0..1);
            }
        }
        lap(4, &mut prof_clock);
        // Underwater fog: when the camera is submerged in a water volume, tint the HDR scene
        // toward the fog colour by reconstructed-depth distance (engine water_ripple.hlsl). Runs
        // after all scene geometry, before bloom, so bloom + resolve exposure apply to the fogged
        // result. Alpha-blended (src.a = 1-transparence) → lerp(scene, fog, 1-transparence).
        if let Some((fog_color, murk)) = self.underwater_fog(camera.pos) {
            let uw = UnderwaterUniform {
                inv_view_proj: vp.inverse().to_cols_array_2d(),
                cam_pos: [camera.pos.x, camera.pos.y, camera.pos.z, 1.0],
                fog: [fog_color[0], fog_color[1], fog_color[2], murk],
            };
            queue.write_buffer(&self.underwater_buf, 0, bytemuck::bytes_of(&uw));
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("underwater-fog-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.underwater_pipeline);
            p.set_bind_group(0, &self.underwater_bg, &[]);
            p.draw(0..3, 0..1);
        }
        // Planar fog: screen-space depth composite over the map's fog volumes (after scene, before
        // bloom). Premultiplied front-to-back inscatter; skipped when no volumes are authored.
        if !self.planar_fog_volumes.is_empty() {
            let mut u = PlanarFogUniform {
                inv_view_proj: vp.inverse().to_cols_array_2d(),
                cam_pos: [camera.pos.x, camera.pos.y, camera.pos.z, 1.0],
                count: [self.planar_fog_volumes.len().min(MAX_PFOG) as f32, 0.0, 0.0, 0.0],
                vols: [PFogVolGpu { plane: [0.0; 4], color_density: [0.0; 4], bmin: [0.0; 4], bmax: [0.0; 4] }; MAX_PFOG],
            };
            for (i, v) in self.planar_fog_volumes.iter().take(MAX_PFOG).enumerate() {
                u.vols[i] = PFogVolGpu {
                    plane: v.plane,
                    color_density: [v.color[0], v.color[1], v.color[2], v.density],
                    // the authored base_depth (full fog depth) rides the free bmin.w slot
                    bmin: [v.bmin[0], v.bmin[1], v.bmin[2], v.base_depth],
                    bmax: [v.bmax[0], v.bmax[1], v.bmax[2], 0.0],
                };
            }
            queue.write_buffer(&self.planar_fog_buf, 0, bytemuck::bytes_of(&u));
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planar-fog-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.planar_fog_pipeline);
            p.set_bind_group(0, &self.planar_fog_bg, &[]);
            p.draw(0..3, 0..1);
        }
        // Auto-exposure meter FIRST: reduce the HDR scene to a 1×1 mean-log-luminance. Runs before
        // the bloom pass so fs_bright can read THIS frame's exposure (bright_gain()) and evaluate the
        // bloom knee in exposed space — the post pass reads the same lum after.
        // The Halo 4 meter's pre-pass: 48x32 box-filtered (rgb, self-illum coverage)
        // of the HDR scene = the engine's block-bloom / gaussian chain before exposure_downsample.
        if self.h4_meter_on.get() {
            let mut pp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("h4pre-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.h4pre_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pp.set_pipeline(&self.h4pre_pipeline);
            pp.set_bind_group(0, &self.h4pre_bg, &[]);
            pp.draw(0..3, 0..1);
        }
        {
            let mut lp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lum-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.lum_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            lp.set_pipeline(&self.lum_pipeline);
            lp.set_bind_group(0, &self.lum_bg, &[]);
            lp.draw(0..3, 0..1);
        }
        // Halo 4 bloom output-size lanes: sfx.zw = 1 / size (blur_11_horizontal's stale
        // `vs_texture_size`), bcs.w = the intensity resolution scale min(1, 921600 / (w h)).
        if self.h4_bloom_on.get() {
            let (w, h) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
            let rf = if w > 1280.0 || h > 720.0 { (921_600.0 / (w * h)).clamp(0.0, 1.0) } else { 1.0 };
            queue.write_buffer(&self.post_buf, 72, bytemuck::cast_slice(&[1.0 / w, 1.0 / h]));
            queue.write_buffer(&self.post_buf, 140, bytemuck::cast_slice(&[rf]));
        }
        // The engine's PC bloom chain (reach_tag_test.exe postprocess_player_view /
        // postprocess_bloom_buffer, HREK postprocess shaders):
        //   P1  curve level at 1/4:  S = 2*E4*(point*lum(E4) + inherent), E4 = 4x4 box of min(E,8)
        //   P2  8x8 box -> 1/16 (M)      P3  8x8 box -> 1/64 (G)
        //   P4  blur11 at 1/64 x intensity (G')
        //   P5  1/16: intensity*M + up(G')   P6  blur11 at 1/16
        //   P7  1/4:  intensity*S + up(T')   P8  blur11 at 1/4  -> composite adds it (bilinear up)
        // every pass output is clamped at 2.0 (the engine stores min(4x,8)/32 in fp16).
        {
            let mut fs = |pipeline: &wgpu::RenderPipeline,
                          bg: &wgpu::BindGroup,
                          target: &wgpu::TextureView,
                          load: wgpu::LoadOp<wgpu::Color>| {
                let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                p.set_pipeline(pipeline);
                p.set_bind_group(0, bg, &[]);
                p.set_bind_group(1, &self.post_lut_bg, &[]); // grading LUT (Halo 4 bright pass)
                p.draw(0..3, 0..1);
            };
            let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
            // The Halo 4 chain runs the same eight steps with its own shaders
            let h4 = self.h4_bloom_on.get();
            let (bright, box8, blur_h, blur_v, blur_v_i, add_m, add_s) = if h4 {
                (&self.bright_h4_pipeline, &self.box8_h4_pipeline, &self.blur_h_h4_pipeline, &self.blur_v_h4_pipeline, &self.blur_v_i_h4_pipeline, &self.add_m_h4_pipeline, &self.add_s_h4_pipeline)
            } else {
                (&self.bright_pipeline, &self.downsample_pipeline, &self.blur_h_pipeline, &self.blur_v_pipeline, &self.blur_v_i_pipeline, &self.bloom_add_pipeline, &self.bloom_add_s_pipeline)
            };
            // P1: curve level S at 1/4 (q_a)
            fs(bright, &self.bright_in_bg, &self.bloom.q_a, clear);
            // P2/P3: 8x8 box downsamples -> M (e_a, 1/16) -> G (t_a, 1/64)
            fs(box8, &self.bg_q_a, &self.bloom.e_a, clear);
            fs(box8, &self.bg_e_a_solo, &self.bloom.t_a, clear);
            // P4: blur11 at 1/64, vertical pass x intensity -> G' in t_a
            fs(blur_h, &self.bg_t_a, &self.bloom.t_b, clear);
            fs(blur_v_i, &self.bg_t_b, &self.bloom.t_a, clear);
            // P5: T = intensity*M + up(G') -> e_b ; P6: blur11 -> T' in e_b
            fs(add_m, &self.bg_e_a, &self.bloom.e_b, clear);
            fs(blur_h, &self.bg_e_b, &self.bloom.e_a, clear);
            fs(blur_v, &self.bg_e_a, &self.bloom.e_b, clear);
            // P7: U = intensity*S + up(T') -> q_b ; P8: blur11 -> U' in out
            fs(add_s, &self.bg_q_a, &self.bloom.q_b, clear);
            fs(blur_h, &self.bg_q_b, &self.bloom.q_a, clear);
            fs(blur_v, &self.bg_q_a, &self.bloom.out, clear);
        }
        // Post pass: tonemap + fog + vignette from HDR into the egui-visible color.
        {
            let mut post = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            post.set_pipeline(&self.post_pipeline);
            post.set_bind_group(0, &self.post_bg, &[]);
            post.set_bind_group(1, &self.post_lut_bg, &[]); // grading LUT
            post.draw(0..3, 0..1);
        }
        lap(5, &mut prof_clock);
        queue.submit(std::iter::once(encoder.finish()));
        lap(6, &mut prof_clock);
        // explicit drops so the profiler can see what releasing the per-frame resources costs
        let t_drop = std::time::Instant::now();
        drop(lens_vbuf);
        drop(lensfx_batches);
        drop(bundles);
        let drop_ms = t_drop.elapsed().as_secs_f32() * 1000.0;
        if prof {
            // per-renderer accumulator (the app has a main and a preview SceneRenderer)
            let (mut sum, mut n) = self.prof_acc.get();
            for k in 0..14 { sum[k] += prof_t[k]; }
            sum[14] += drop_ms;
            sum[15] += t_render_entry.elapsed().as_secs_f32() * 1000.0; // whole render() wall
            n += 1;
            if n >= 120 {
                let f = n as f32;
                eprintln!("HMS_RENDER_PROF [{}x{}] avg over {n} frames: prologue={:.2} bundles={:.2} shadow+sky={:.2} scene={:.2} (opaque={:.2} terrain={:.2} alphatest={:.2} foliage={:.2} pass-end={:.2}) transparent={:.2} post={:.2} submit={:.2} drops={:.2} | render() total={:.2} ms",
                    self.size.0, self.size.1, sum[0] / f, sum[1] / f, sum[2] / f, sum[3] / f, sum[9] / f, sum[10] / f, sum[11] / f, sum[12] / f, sum[13] / f, sum[4] / f, sum[5] / f, sum[6] / f, sum[14] / f, sum[15] / f);
                sum = [0.0; 16]; n = 0;
            }
            self.prof_acc.set((sum, n));
        }
    }
}

/// The cached static-list render bundles (see `SceneRenderer::static_bundles`).
/// Each list gets its own bundle chain so the show_* toggles just pick which to execute. Bundles
/// are recorded against the scene/transparent pass attachments (HDR colour + writable depth),
/// which both passes share.
struct StaticBundles {
    gen: u64,
    opaque: BundleChain,
    imported: BundleChain,
    markers: BundleChain,
    terrain: BundleChain,
    alphatest: BundleChain,
    foliage: BundleChain,
    water: BundleChain,
    decals: BundleChain,
    additive: BundleChain,
}

/// Bundles covering `[0..covered)` of one mesh list, in order. A streaming load
/// APPENDS chunks to the lists many times per second; recording only the new tail keeps that
/// O(chunk) instead of O(whole list) per chunk (re-recording 8000 draws took 60-80 ms).
#[derive(Default)]
struct BundleChain {
    bundles: Vec<wgpu::RenderBundle>,
    covered: usize,
}

impl BundleChain {
    /// Extend the chain to cover `len` items; `rec(enc, from)` records items `from..len`.
    fn extend<'s>(&mut self, device: &wgpu::Device, label: &str, len: usize, rec: impl FnOnce(&mut wgpu::RenderBundleEncoder<'s>, usize)) {
        if len > self.covered {
            let from = self.covered;
            if let Some(b) = record_bundle(device, label, false, |e| rec(e, from)) {
                self.bundles.push(b);
            }
            self.covered = len;
        }
    }
    fn execute(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.bundles.is_empty() {
            pass.execute_bundles(self.bundles.iter());
        }
    }
}

impl SceneRenderer {
    /// Mark the static bundles stale (a list was REPLACED/cleared or the camera bind
    /// group changed). Appends do not need this: `static_bundles` sees the longer list and records
    /// just the new tail.
    fn bump_static(&mut self) {
        self.static_gen = self.static_gen.wrapping_add(1);
    }

    /// Return the static bundles, (re)recording whatever is stale: everything after a
    /// generation bump, only the appended tail of a list that grew. Steady state records nothing.
    fn static_bundles(&self, device: &wgpu::Device) -> std::cell::Ref<'_, StaticBundles> {
        let mut cur = self.static_bundles.borrow_mut();
        if cur.as_ref().map_or(true, |b| b.gen != self.static_gen) {
            *cur = Some(StaticBundles {
                gen: self.static_gen,
                opaque: BundleChain::default(), imported: BundleChain::default(), markers: BundleChain::default(), terrain: BundleChain::default(),
                alphatest: BundleChain::default(), foliage: BundleChain::default(), water: BundleChain::default(),
                decals: BundleChain::default(), additive: BundleChain::default(),
            });
        }
        let b = cur.as_mut().unwrap();
        let mr = &*self.mesh_renderer;
        let cb = &self.camera_bg;
        b.opaque.extend(device, "bundle-opaque", self.static_meshes.len(), |e, from| mr.draw(e, cb, &self.static_meshes[from..]));
        b.imported.extend(device, "bundle-imported", self.imported_meshes.len(), |e, from| mr.draw(e, cb, &self.imported_meshes[from..]));
        b.markers.extend(device, "bundle-markers", self.marker_meshes.len(), |e, from| mr.draw(e, cb, &self.marker_meshes[from..]));
        b.terrain.extend(device, "bundle-terrain", self.terrain_meshes.len(), |e, from| mr.draw_terrain(e, cb, &self.terrain_meshes[from..]));
        b.alphatest.extend(device, "bundle-alphatest", self.alphatest_meshes.len(), |e, from| mr.draw_alphatest(e, cb, &self.alphatest_meshes[from..]));
        // budget 0 = draw every decorator group (the default); order is irrelevant for a
        // depth-writing alpha-test pass, so the per-frame nearest-first sort is not needed.
        b.foliage.extend(device, "bundle-foliage", self.foliage_meshes.len(), |e, from| mr.draw_foliage(e, cb, &self.foliage_meshes[from..], [0.0; 3], 0));
        b.water.extend(device, "bundle-water", self.water_meshes.len(), |e, from| mr.draw_water(e, cb, &self.water_meshes[from..]));
        // decals are sorted by blend bucket across the WHOLE list (draw order affects the
        // multiply/additive composite), so a grown decal list is re-recorded as one bundle.
        if self.decal_meshes.len() > b.decals.covered {
            b.decals = BundleChain::default();
            b.decals.extend(device, "bundle-decals", self.decal_meshes.len(), |e, _| mr.draw_decals(e, cb, &self.decal_meshes));
        }
        b.additive.extend(device, "bundle-additive", self.additive_meshes.len(), |e, from| mr.draw_additive(e, cb, &self.additive_meshes[from..]));
        drop(cur);
        std::cell::Ref::map(self.static_bundles.borrow(), |b| b.as_ref().unwrap())
    }
}

/// Record one static-list bundle against the scene-pass attachments (HDR colour +
/// writable depth). `empty` short-circuits to None so an empty list costs nothing per frame.
fn record_bundle<'s>(
    device: &wgpu::Device,
    label: &str,
    empty: bool,
    f: impl FnOnce(&mut wgpu::RenderBundleEncoder<'s>),
) -> Option<wgpu::RenderBundle> {
    if empty {
        return None;
    }
    let mut enc = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
        label: Some(label),
        color_formats: &[Some(SCENE_HDR_FORMAT)],
        depth_stencil: Some(wgpu::RenderBundleDepthStencil {
            format: DEPTH_FORMAT,
            depth_read_only: false,
            stencil_read_only: false,
        }),
        sample_count: 1,
        multiview: None,
    });
    f(&mut enc);
    Some(enc.finish(&wgpu::RenderBundleDescriptor { label: Some(label) }))
}

struct Targets {
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_copy_tex: wgpu::Texture,
    depth_copy_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    hdr_tex: wgpu::Texture,
}

// SUPERSAMPLE factor. The scene (hdr/depth) renders at SS× the output resolution and the post
// pass BOX-DOWNSAMPLES to the 1× color target — true SSAA. This is what kills the terrain
// rock/foliage GRAIN, which is high-frequency SHADING aliasing (bump/detail per-pixel lighting):
// mip-LOD floors and MSAA can't fix it (shading runs per fragment), but supersampling runs the
// fragment per sub-sample and matches the engine (a 2× render downscaled is smooth like Sapien's
// bb_8). Color stays 1× so egui/headless capture are unchanged.
const SS: u32 = 2;

fn make_targets(device: &wgpu::Device, size: (u32, u32)) -> Targets {
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    // Scene render targets (hdr + depth) are supersampled; color output stays at `size`.
    let ss_extent = wgpu::Extent3d {
        width: size.0 * SS,
        height: size.1 * SS,
        depth_or_array_layers: 1,
    };
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        // Allow an sRGB VIEW of this Unorm texture. egui's shader always does
        // gamma_from_linear(sample) — it assumes registered textures are LINEAR — so
        // if we hand it the raw Unorm view of our already-sqrt-encoded color, it
        // DOUBLE-encodes and the viewport reads brighter than the raw-bytes PNG
        // capture (the "screenshot ≠ viewport" bug). Registering the sRGB view makes
        // egui's sample auto-DECODE first, so its re-encode nets 1:1 → viewport == PNG.
        view_formats: &[COLOR_FORMAT_SRGB],
    });
    let color_view = color_tex.create_view(&Default::default());
    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-hdr"),
        size: ss_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let hdr_view = hdr_tex.create_view(&Default::default());
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
        size: ss_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&Default::default());
    // Sampleable copy of the opaque depth: after opaque geometry is drawn we copy
    // depth_tex → depth_copy_tex, then the water pass samples it (the live depth_view
    // stays the attachment, so no read-only-attachment conflict).
    let depth_copy_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth-copy"),
        size: ss_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let depth_copy_view = depth_copy_tex.create_view(&Default::default());
    Targets {
        color_tex,
        color_view,
        depth_tex,
        depth_view,
        depth_copy_tex,
        depth_copy_view,
        hdr_view,
        hdr_tex,
    }
}

/// Build the post pass's bind group (HDR scene + sampler + bloom). Rebuilt on
/// resize because the views are recreated.
fn make_post_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    hdr: &wgpu::TextureView,
    bloom: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    post_buf: &wgpu::Buffer,
    lum: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(bloom) },
            wgpu::BindGroupEntry { binding: 3, resource: post_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(lum) },
        ],
    })
}

/// The meter's bind group: HDR (0) + sampler (1) + post uniform (2) + H4 pre-pass (3).
fn make_lum_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    hdr: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    post_buf: &wgpu::Buffer,
    h4pre: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lum-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: post_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(h4pre) },
        ],
    })
}

/// The 48x32 RGBA32F pre-pass target of the Halo 4 exposure meter (H4PRE_WGSL).
const H4PRE_W: u32 = 48;
const H4PRE_H: u32 = 32;
fn make_h4pre_tex(device: &wgpu::Device) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("h4pre-48x32"),
        size: wgpu::Extent3d { width: H4PRE_W, height: H4PRE_H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&Default::default())
}

/// The post pass's colour-grading LUT bind group. `lut` = (BGRA8 texels, n) of an
/// n^3 volume (x fastest, then y, then z); None = a 2^3 identity volume.
fn make_lut_bg(device: &wgpu::Device, queue: &wgpu::Queue, bgl: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, lut: Option<(&[u8], u32)>) -> wgpu::BindGroup {
    let ident: Vec<u8> = (0..8).flat_map(|i: u32| { let (x, y, z) = (i & 1, (i >> 1) & 1, i >> 2); [(z * 255) as u8, (y * 255) as u8, (x * 255) as u8, 255u8] }).collect();
    let (bytes, n) = match lut { Some((b, n)) if b.len() >= (n * n * n * 4) as usize && n >= 2 => (b, n), _ => (ident.as_slice(), 2) };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("post-lut-3d"),
        size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &bytes[..(n * n * n * 4) as usize],
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
        wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
    );
    let view = tex.create_view(&Default::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post-lut-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// Single texture + sampler bind group (bright-pass / blur inputs).
fn make_tex_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    tex: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tex-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tex) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// Engine bloom mip pyramid: curve level at ¼-res, box-downsample to 1/16 and 1/64, blur each
/// level (ping-pong a/b), additively combine back up into the ¼-res `out` level.
struct BloomViews {
    q_a: wgpu::TextureView, q_b: wgpu::TextureView, // 1/4  (engine aux_small / aux_small2)
    e_a: wgpu::TextureView, e_b: wgpu::TextureView, // 1/16 (aux_tiny / aux_tiny2)
    t_a: wgpu::TextureView, t_b: wgpu::TextureView, // 1/64 (aux_mini / aux_mini2)
    out: wgpu::TextureView,                          // 1/4  final blurred level (resolve reads this)
    out_tex: wgpu::Texture,                          // `dump_bloom` readback source
}
fn make_bloom_pyramid(device: &wgpu::Device, size: (u32, u32)) -> BloomViews {
    let mk_tex = |div: u32| {
        let extent = wgpu::Extent3d {
            width: (size.0 / div).max(1),
            height: (size.1 / div).max(1),
            depth_or_array_layers: 1,
        };
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom-mip"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let mk = |div: u32| mk_tex(div).create_view(&Default::default());
    let out_tex = mk_tex(4);
    BloomViews {
        q_a: mk(4), q_b: mk(4),
        e_a: mk(16), e_b: mk(16),
        t_a: mk(64), t_b: mk(64),
        out: out_tex.create_view(&Default::default()),
        out_tex,
    }
}

/// Ground grid (Z-up, on the XY plane) + colored XYZ axes.
fn build_grid(half: i32, step: f32) -> Vec<Vertex> {
    let mut v = Vec::new();
    let grey = [0.22, 0.24, 0.28];
    let ext = half as f32 * step;
    for i in -half..=half {
        let t = i as f32 * step;
        v.push(Vertex { pos: [t, -ext, 0.0], color: grey });
        v.push(Vertex { pos: [t, ext, 0.0], color: grey });
        v.push(Vertex { pos: [-ext, t, 0.0], color: grey });
        v.push(Vertex { pos: [ext, t, 0.0], color: grey });
    }
    // axes
    v.push(Vertex { pos: [0.0, 0.0, 0.0], color: [0.9, 0.2, 0.2] });
    v.push(Vertex { pos: [ext, 0.0, 0.0], color: [0.9, 0.2, 0.2] });
    v.push(Vertex { pos: [0.0, 0.0, 0.0], color: [0.2, 0.9, 0.2] });
    v.push(Vertex { pos: [0.0, ext, 0.0], color: [0.2, 0.9, 0.2] });
    v.push(Vertex { pos: [0.0, 0.0, 0.0], color: [0.3, 0.5, 1.0] });
    v.push(Vertex { pos: [0.0, 0.0, ext], color: [0.3, 0.5, 1.0] });
    v
}

const BASIC_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VOut {
    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(pos, 1.0);
    o.color = color;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.color, 1.0);
}
"#;

// #wire-visible  The SELECTION WIREFRAME shader (halo + core).
//
// The problem it solves: the wireframe is drawn into the LINEAR HDR scene, so the auto-exposure
// that follows scales the line down with everything else. On a bright, low-contrast Halo 4 scene
// (white Forge pieces under a blown-out sky) a pale gold line over pale pixels is nearly
// invisible, and simply raising its brightness cannot help -- both sides saturate together.
//
// Two mechanisms, neither of which depends on what is behind the line:
//   1. HALO. The same line list is drawn as several instances: 0-7 are the line offset around a
//      small circle in SCREEN space and painted BLACK, 8-11 the bright core (a slightly thicker
//      dark outline underneath, a bright core on top). Black survives any exposure, so against a
//      blown-out background the halo carries the contrast; against a dark background the core
//      does. One of the two edges always reads.
//   2. EXPOSURE-COMPENSATED CORE. The fragment stage reproduces the resolve pass's per-frame gain
//      from the same post uniform + 1x1 luminance meter and divides it out, so the core lands at a
//      fixed EXPOSED value (WIRE_CORE_TARGET) instead of following the scene down. Clamped to
//      [1, WIRE_MAX_BOOST] so it can only ever brighten the old colour, never dim it.
//
// Both pipelines also pull the line 0.4% of its distance TOWARD the camera. That is a shift along
// the view ray, so the line lands on exactly the same pixel but a hair nearer in depth: the
// wireframe of the selected object cannot z-fight with its own surface, at any distance.
const WIRE_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(9) var lum_tex: texture_2d<f32>;
// Prefix of the post uniform: p = (stops gain, key, lo log2, hi log2), grade.w = fixed
// g_exposure (0 = metered), bloom.y = the meter-unit calibration.
struct WPost { p: vec4<f32>, grade: vec4<f32>, bloom: vec4<f32> };
@group(1) @binding(0) var<uniform> post: WPost;
// xy = 1 / HDR target size (pixels -> NDC), z = halo radius in HDR pixels, w = core target.
struct Wire { v: vec4<f32> };
@group(1) @binding(1) var<uniform> wire: Wire;

const TAU: f32 = 6.2831853;
// Depth pull toward the camera, as a fraction of the distance (scale free: no z-fighting near
// or far, and far too small to let the line show through unrelated geometry).
const WIRE_DEPTH_PULL: f32 = 0.004;
// Core half-width in HDR pixels (SS=2, so 0.6 HDR px ~ 0.3 output px each side of centre).
const WIRE_CORE_R: f32 = 0.6;
// The core aims for this EXPOSED value (>1 = into the top of the tonemap curve).
const WIRE_MAX_BOOST: f32 = 24.0;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) @interpolate(flat) core: u32,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>,
      @builtin(instance_index) inst: u32) -> VOut {
    // Along-the-view-ray depth bias (same pixel, a hair nearer).
    let p = pos + (cam.cam_pos.xyz - pos) * WIRE_DEPTH_PULL;
    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(p, 1.0);
    // Instances 0-7: the black halo ring. 8-11: the bright core, offset by a fraction of a pixel
    // in four directions so it reads as a solid line after the supersample downsample.
    var r = wire.v.z;
    var n = 8u;
    var k = inst;
    o.core = 0u;
    if (inst >= 8u) {
        r = WIRE_CORE_R;
        n = 4u;
        k = inst - 8u;
        o.core = 1u;
    }
    let a = f32(k) * (TAU / f32(n));
    let off = vec2<f32>(cos(a), sin(a)) * r * wire.v.xy * 2.0;
    o.clip = vec4<f32>(o.clip.xy + off * o.clip.w, o.clip.z, o.clip.w);
    o.color = color;
    return o;
}

// The resolve pass's per-frame gain, from the same inputs (lib.rs resolve()).
fn wire_gain() -> f32 {
    let mean_log = textureLoad(lum_tex, vec2<i32>(0, 0), 0).r;
    let hi = exp2(post.p.w);
    let lo = min(exp2(post.p.z), hi);
    let cal = select(1.0, post.bloom.y, post.bloom.y > 0.0);
    var g = post.p.x * clamp(post.p.y / max(exp2(mean_log), 1e-4) * cal, lo, hi);
    if (post.grade.w > 1e-6) { g = post.grade.w; }
    return g;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    if (i.core == 0u) {
        // The halo: black. Alpha-blended at a = 1, so it REPLACES the pixel -- after exposure and
        // tonemap it is still black whatever the scene did, which is the whole point.
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Normalise on the BRIGHTEST channel before boosting: scaling the raw colour would push
    // every channel past the tonemap's saturation point and the gold core would turn white.
    let peak = max(max(i.color.r, max(i.color.g, i.color.b)), 1e-4);
    let boost = clamp(wire.v.w / (peak * max(wire_gain(), 1e-6)), 1.0, WIRE_MAX_BOOST);
    return vec4<f32>(i.color * boost, 1.0);
}
"#;

// Translucent zone volume (boundary shapes). Holographic look: a low-opacity fill whose rim
// brightens at grazing view angles (fresnel from the face normal reconstructed via screen-space
// derivatives, so no per-vertex normals are needed) plus a slow vertical scan pulse. Reuses the
// scene camera uniform for cam_pos (fresnel) and time (pulse). Alpha-blended, depth-tested.
const ZONE_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    time: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) wp: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(pos, 1.0);
    o.color = color;
    o.wp = pos;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let n = normalize(cross(dpdx(i.wp), dpdy(i.wp)));
    let v = normalize(cam.cam_pos.xyz - i.wp);
    let ndv = abs(dot(n, v));
    let fres = pow(1.0 - ndv, 2.5);                    // grazing-angle rim glow
    let pulse = 0.85 + 0.15 * sin(i.wp.z * 0.6 - cam.time.x * 2.5);
    // Keep the fill faint (so it barely occludes the scene); most of the read comes from the
    // brighter grazing-angle rim, which stays legible without a heavy flat wash.
    let a = clamp(i.color.a + fres * 0.30, 0.0, 0.6) * pulse;
    let rgb = i.color.rgb * (0.5 + fres * 1.2) * pulse;
    return vec4<f32>(rgb, a);
}
"#;

// Sun lens flare: screen-space additive procedural sprites. Vertices carry NDC position
// directly (no camera transform); the fragment draws a soft radial disc tinted by the
// per-vertex colour. Additive One/One into the HDR scene.
const LENS_WGSL: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) col: vec4<f32>, @location(2) tp: f32 };
@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) col: vec4<f32>, @location(3) tp: f32) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.uv = uv;
    o.col = col;
    o.tp = tp;
    return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let d = length(i.uv - vec2<f32>(0.5, 0.5)) * 2.0;   // 0 centre .. 1 edge
    if (d >= 1.0) { discard; }
    // Engine lens_flare.hlsl pow(color.g, tint_power): tint_power shapes the sprite falloff —
    // high power = a tight bright core, low power = a soft wide halo. Per-element authored.
    let a = pow(clamp(1.0 - d, 0.0, 1.0), max(i.tp, 0.5));
    return vec4<f32>(i.col.rgb * (a * i.col.a), 1.0);
}
"#;

/// Procedural atmosphere sky: a fullscreen triangle whose fragment reconstructs a world-space
/// view ray from the inverse view-proj, then blends a three-band gradient (zenith → horizon →
/// ground) coloured from the map's atmosphere fog, plus a pinpoint HDR sun that bloom spreads.
/// Drawn behind the sky render_model on maps that author atmosphere.
const SKY_WGSL: &str = r#"
struct Sky { inv_view_proj: mat4x4<f32>, cam_pos: vec4<f32>, sun_dir: vec4<f32>, horizon: vec4<f32> };
@group(0) @binding(0) var<uniform> sky: Sky;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    // Oversized triangle covering the whole screen.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var o: VOut;
    o.clip = vec4<f32>(p[vi], 0.0, 1.0);
    o.ndc = p[vi];
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    // World-space ray through this pixel: unproject near & far, subtract.
    let near = sky.inv_view_proj * vec4<f32>(i.ndc, 0.0, 1.0);
    let far  = sky.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let up = clamp(dir.z, -1.0, 1.0);           // +Z is up in Halo space
    // Horizon = the MAP's OWN atmosphere_fog sky inscatter color (sky.horizon.rgb), so the
    // sky matches the map's authored atmosphere and fogged BSP dissolves into it. Zenith = a
    // darker shade of the same (a bright flat sky washes out translucent sky-model layers,
    // most visibly the Halo ring's semi-transparent arc); ground = dim. This pass only runs on
    // maps WITH atmosphere (space maps stay on the near-black clear).
    let horizon = sky.horizon.rgb;
    let zenith  = horizon * 0.35;
    let ground  = horizon * 0.20;

    var col: vec3<f32>;
    if (up >= 0.0) {
        // Steep ramp (pow 0.80) so the sky reaches the deeper zenith blue soon overhead,
        // keeping only the low sky near the bright horizon colour.
        col = mix(horizon, zenith, pow(up, 0.80));
    } else {
        col = mix(horizon, ground, pow(-up, 0.5));
    }
    // The sun: ONE engine-form term (sky_dome_simple `pow(dot,exp)·sunColor`) painted as a TINY
    // very-bright HDR point that the post bloom pass spreads into a soft round glare — the
    // in-game look is a lens-flare glare, not a hard disc (the sky-model dome has no opaque
    // backplate over the sun on Forge, so the open sky is this procedural pass). Gated to
    // atmosphere maps (horizon.w>0.5) with a valid sun dir, so space maps stay clean.
    let sd = sky.sun_dir.xyz;
    if (sky.horizon.w > 0.5 && dot(sd, sd) > 1e-4) {
        let s = normalize(sd);
        let cd = max(dot(dir, s), 0.0);
        let sun_col = vec3<f32>(1.0, 0.96, 0.88);  // warm-white daylight sun (SKY_WGSL has no sun_tint)
        col = col + sun_col * (pow(cd, 4000.0) * 7.0);
    }
    return vec4<f32>(col, 1.0);
}
"#;

/// Underwater fog (water_ripple.hlsl_include:576-614): a fullscreen pass that tints the scene
/// toward the underwater fog colour by reconstructed-depth distance when the camera is inside a
/// water volume. Reach form: fog_factor = 1 - exp2(-murk·dist); transparence = 0.5·saturate(1 -
/// fog_factor) = 0.5·saturate(exp2(-murk·dist)); output = lerp(fog_color, pixel, transparence).
/// We composite into the HDR buffer with ALPHA BLEND (src.a = 1-transparence), so resolve()'s
/// exposure applies to the fog with the scene (engine's ×g_exposure is our resolve stage — not
/// re-applied here). Depth is the pre-transparent scene depth copy (depth_copy_view).
const UNDERWATER_WGSL: &str = r#"
struct UW { inv_view_proj: mat4x4<f32>, cam_pos: vec4<f32>, fog: vec4<f32> }; // fog.rgb=color, fog.w=murkiness
@group(0) @binding(0) var<uniform> uw: UW;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    var o: VOut;
    o.clip = vec4<f32>(p[vi], 0.0, 1.0);
    o.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i.clip.xy);
    let d = textureLoad(depth_tex, px, 0);
    // Reconstruct world pos from depth (far plane when no geometry → full fog into the void).
    let ndc = vec3<f32>(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0, d);
    let wh = uw.inv_view_proj * vec4<f32>(ndc, 1.0);
    let world = wh.xyz / wh.w;
    let dist = length(uw.cam_pos.xyz - world);
    let transparence = 0.5 * clamp(exp2(-uw.fog.w * dist), 0.0, 1.0);
    return vec4<f32>(uw.fog.rgb, 1.0 - transparence);
}
"#;

/// Planar fog (explicit/planar_fog.hlsl_include): a screen-space depth composite. For each scene
/// pixel inside a fog volume's AABB and below its plane, extinction = exp(-thickness·depth_below),
/// inscatter = (1-extinction)·color, front-to-back accumulated over volumes. Composited into HDR
/// with PREMULTIPLIED alpha (src.rgb = accumulated inscatter, src.a = 1-transmittance), so resolve()
/// exposes it with the scene. thickness = authored density × PFOG_SCALE.
const PLANAR_FOG_WGSL: &str = r#"
struct PFogVol { plane: vec4<f32>, color_density: vec4<f32>, bmin: vec4<f32>, bmax: vec4<f32> };
struct PFog { inv_view_proj: mat4x4<f32>, cam_pos: vec4<f32>, count: vec4<f32>, vols: array<PFogVol, 8> };
@group(0) @binding(0) var<uniform> pf: PFog;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    var o: VOut;
    o.clip = vec4<f32>(p[vi], 0.0, 1.0);
    o.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return o;
}
const PFOG_SCALE: f32 = 0.05; // density·depth → extinction rate (tunable; density units unbridged)
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i.clip.xy);
    let d = textureLoad(depth_tex, px, 0);
    let ndc = vec3<f32>(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0, d);
    let wh = pf.inv_view_proj * vec4<f32>(ndc, 1.0);
    let world = wh.xyz / wh.w;
    var trans = 1.0;
    var insc = vec3<f32>(0.0);
    let n = u32(pf.count.x);
    for (var k = 0u; k < n; k = k + 1u) {
        let v = pf.vols[k];
        if (world.x < v.bmin.x || world.x > v.bmax.x
         || world.y < v.bmin.y || world.y > v.bmax.y
         || world.z < v.bmin.z || world.z > v.bmax.z) { continue; }
        let sdist = dot(v.plane.xyz, world) + v.plane.w; // >0 above the plane
        var depth_below = max(-sdist, 0.0);
        // Engine planar_fog.hlsl_include:143: depth_in_fog = max(camera_depth_in_fog, depth_in_fog).
        // When the camera is submerged in the fog volume the whole (in-AABB) screen must fog at
        // least to the camera's own depth below the plane — otherwise near geometry / sky reads
        // depth≈0 and gets no fog.
        let cam_sdist = dot(v.plane.xyz, pf.cam_pos.xyz) + v.plane.w;
        let cam_depth_below = max(-cam_sdist, 0.0);
        depth_below = max(depth_below, cam_depth_below);
        if (depth_below <= 0.0) { continue; } // both pixel AND camera above the plane → no fog
        // Engine planar_fog.hlsl_include:111,146: extinction = saturate(exp(-thickness ·
        // saturate(depth/base_depth) · depth)) — a QUADRATIC ramp for depth < base_depth then linear.
        // base_depth (v.bmin.w) is the authored sddt "full fog depth" surfaced via the FFI; unset
        // (<=0) → ramp = 1 (pure linear). TODO: the `_planar_fog_thickness` bridge (fogg density →
        // engine thickness) is packed at runtime by planar_fog.cpp (not RE'd); PFOG_SCALE is the
        // calibrated stand-in.
        let base_depth = v.bmin.w;
        let ramp = select(1.0, clamp(depth_below / base_depth, 0.0, 1.0), base_depth > 1e-3);
        let e = clamp(exp(-v.color_density.w * PFOG_SCALE * ramp * depth_below), 0.0, 1.0);
        insc = insc + v.color_density.rgb * (1.0 - e) * trans;
        trans = trans * e;
    }
    return vec4<f32>(insc, 1.0 - trans); // premultiplied inscatter + coverage
}
"#;

/// Post-processing composite (the engine's final_composite on the PC/DX11 path): samples the
/// supersampled HDR scene, applies the auto-exposure gain, adds bloom, the screen-effect gain,
/// the Halo 4 filmic curve when active, the gamma-2 (sqrt) encode, the colour-grading LUT
/// (Halo 4) and the screen-effect colour matrix, then a hard saturate. There is NO filmic /
/// Reinhard tone curve on Reach (`apply_tone_curve` is commented out in
/// final_composite_base.hlsl_include:274), no vignette (the engine applies film grain instead,
/// not reproduced) and an identity colour matrix for every shipped Reach cfxs.
const POST_WGSL: &str = r#"
// Reach SDR color-control saturation (fn_18026BA9C). Engine default 1.0 (the Sapien capture of
// ivory_tower shows muted, natural colors; 1.2 oversaturates wood to neon orange).
const POST_SATURATION: f32 = 1.0;
// Meter-unit exposure calibration default (post.bloom.y overrides it when > 0). The engine has no
// such term: the meter already normalizes scene brightness to the cfxs key
// (g_exposure = clamp(key/metered, band)), so 1.0 is engine-correct.
const POST_EXPOSURE_CAL: f32 = 1.0;
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
// p = (stops_gain = pow(2,camera_exposure_stops), auto_key, exp_min_log2, exp_max_log2).
struct Post { p: vec4<f32>, grade: vec4<f32>, bloom: vec4<f32>, bloom2: vec4<f32>, sfx: vec4<f32>, sfx2: vec4<f32>, bcl: vec4<f32>, bcm: vec4<f32>, bcs: vec4<f32>, cm0: vec4<f32>, cm1: vec4<f32>, cm2: vec4<f32>, h4a: vec4<f32>, h4b: vec4<f32> };
@group(0) @binding(3) var<uniform> post: Post;
// 1×1 luminance meter: r = mean log2(scene luminance) (engine exposure_downsample).
@group(0) @binding(4) var lum_tex: texture_2d<f32>;
// Halo 4 colour-grading volume (cfxs +0xF0; h4b.z > 0.5 enables it).
@group(1) @binding(0) var lut_tex: texture_3d<f32>;
@group(1) @binding(1) var lut_samp: sampler;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let pos = p[vi];
    var o: VOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    // NDC → UV (flip Y for texture space).
    o.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return o;
}

// Full composite for one HDR texel. Factored out so fs() can evaluate it at the four
// supersample centres and box-average.
fn resolve(uv: vec2<f32>) -> vec3<f32> {
    let hdr = textureSample(hdr_tex, hdr_samp, uv).rgb;
    let bloom_s = textureSample(bloom_tex, hdr_samp, uv).rgb;
    let mean_log = textureLoad(lum_tex, vec2<i32>(0, 0), 0).r;
    // Auto-exposure: gain = key / metered luminance, clamped to the per-map cfxs band
    // (p.z/p.w in stops, see set_auto_exposure). The meter-unit correction (post.bloom.y,
    // HMS_EXPCAL; bridges the RGB-luminance meter to the engine's metering) applies to the raw
    // gain BEFORE the band clamp, since the band bounds the FINAL engine g_exposure.
    let hi = exp2(post.p.w);
    let lo = min(exp2(post.p.z), hi);
    let raw_gain = post.p.y / max(exp2(mean_log), 1e-4);
    let cal = select(POST_EXPOSURE_CAL, post.bloom.y, post.bloom.y > 0.0);
    let auto_gain = clamp(raw_gain * cal, lo, hi);
    var gain = post.p.x * auto_gain;
    // Fixed exposure: when grade.w > 0 the map drives a FIXED per-frame g_exposure scalar (the
    // engine's sceg-derived exposure, no live meter) — so bright panels don't blow out when the
    // view turns toward a dark interior.
    if (post.grade.w > 1e-6) { gain = post.grade.w; }
    let exposed = max(hdr, vec3<f32>(0.0)) * gain;
    // Engine CALC_BLOOM (final_composite_base.hlsl_include): `blend = combined + bloom·8` where
    // combined = scene·exposure and the ×8 only undoes the bloom buffer's /DARK_COLOR_MULTIPLIER
    // storage (apply_bloom_curve.hlsl_include returns color/8) — net gain 1. The engine extracts
    // bloom AFTER shade-time exposure, so its bloom is in EXPOSED units; fs_bright lifts the HDR
    // scene by the same per-frame gain (bright_gain()) before the curve, so bloom_s is already
    // exposed and is added straight in. post.bloom.x = the user bloom slider (1.0 = engine).
    const BLOOM_BIAS: f32 = 1.0;
    var col = max(exposed + bloom_s * BLOOM_BIAS * post.bloom.x, vec3<f32>(0.0));
    // Saturation hook (POST_SATURATION = 1.0 → exact no-op), in linear.
    let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = max(vec3<f32>(luma) + POST_SATURATION * (col - vec3<f32>(luma)), vec3<f32>(0.0));
    // Screen effect (scnr -> sefc area_screen_effect, engine screen_effect.cpp): the "exposure
    // boost:stops" raises the exposure linearly before the gamma-2 sqrt (sfx.x = 2^stops); its
    // colour terms (hue rotation, saturation, contrast, sqrt(filter), floor) of the scenario
    // default AND the placed Forge special-FX objects are composed on the CPU by the engine's
    // per-term-max rule (screenfx.rs) into the k_ps_color_matrix cm0..cm2, applied AFTER the
    // sqrt (convert_to_gamma2_and_apply_color_adjustments: pow(color, gamma.z) then
    // saturate(mul(float4(color,1), color_matrix))); sfx.y (0.5 = sqrt) carries gamma
    // enhance/reduce. Zealot authors filter (0.867,0.893,1.0); Forge World boost 0.1 stop.
    col = col * post.sfx.x;
    // Halo 4 final_composite (explicit shader `final_composite`, verified asm): the exposed scene
    // + bloom goes through the cfxs FILMIC curve t = c (P0 c + P1) / (c (P2 c + P3) + P4)
    // (= Hable(c) / Hable(W), P from sub_1803818E0) BEFORE the sqrt. bloom2.y > 0.5 selects it;
    // Reach never sets it.
    if (post.bloom2.y > 0.5) {
        let c0 = max(col, vec3<f32>(0.0));
        col = c0 * (post.bloom2.z * c0 + post.bloom2.w) / (c0 * (post.sfx2.w * c0 + post.bcl.w) + post.bcm.w);
    }
    var outc = pow(max(col, vec3<f32>(0.0)), vec3<f32>(post.sfx.y));
    // Halo 4 final_composite: `sample_l(lut, sqrt_t * scale + offset, 0)` with scale = (n-1)/n,
    // offset = 0.5/n (sub_180381578 ps_color_grading_scale_offset) - the LUT is the LAST op.
    if (post.h4b.z > 0.5) {
        let n = vec3<f32>(textureDimensions(lut_tex));
        let luv = clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0)) * (n - 1.0) / n + 0.5 / n;
        outc = textureSampleLevel(lut_tex, lut_samp, luv, 0.0).rgb;
    }
    let v4 = vec4<f32>(outc, 1.0);
    outc = clamp(vec3<f32>(dot(v4, post.cm0), dot(v4, post.cm1), dot(v4, post.cm2)), vec3<f32>(0.0), vec3<f32>(1.0));
    // Luminance-preserving post grade (post.grade.rgb; identity → exact no-op).
    let g_luma = max(dot(post.grade.rgb, vec3<f32>(0.299, 0.587, 0.114)), 1e-3);
    outc = outc * (post.grade.rgb / g_luma);
    return saturate(outc);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    // SSAA box downsample. The hdr target is SS(=2)× the output, so each output pixel covers a
    // 2×2 hdr sub-block. Run the full composite at the 4 sub-texel centres (uv ± half an hdr
    // texel) and average → true supersampling (per-sub-sample shading kills the terrain
    // rock/foliage bump/detail shading aliasing; the box average also anti-aliases edges).
    let o = 0.5 / vec2<f32>(textureDimensions(hdr_tex));
    let a = resolve(i.uv + vec2<f32>(-o.x, -o.y));
    let b = resolve(i.uv + vec2<f32>( o.x, -o.y));
    let c = resolve(i.uv + vec2<f32>(-o.x,  o.y));
    let d = resolve(i.uv + vec2<f32>( o.x,  o.y));
    return vec4<f32>((a + b + c + d) * 0.25, 1.0);
}
"#;

/// Luminance meter (engine `exposure_downsample`): one fullscreen pass into a 1×1 R32F target
/// holding the log2 of the metered scene luminance the post pass auto-exposes to (a blend of
/// the geometric and arithmetic means over a centre-weighted grid of HDR samples). With the
/// Halo 4 meter on, the pass instead SOLVES the engine's adaptation fixed point.
const LUM_WGSL: &str = r#"
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
// The post uniform (h4a = bloom highlight / inherent / self-illum / sensitivity, h4b.x =
// screen brightness, h4b.y = Halo 4 meter on; p.z / p.w = the exposure band in stops) + the 48x32
// pre-pass (rgb = box-filtered HDR, a = self-illum coverage).
struct Post { p: vec4<f32>, grade: vec4<f32>, bloom: vec4<f32>, bloom2: vec4<f32>, sfx: vec4<f32>, sfx2: vec4<f32>, bcl: vec4<f32>, bcm: vec4<f32>, bcs: vec4<f32>, cm0: vec4<f32>, cm1: vec4<f32>, cm2: vec4<f32>, h4a: vec4<f32>, h4b: vec4<f32> };
@group(0) @binding(2) var<uniform> post: Post;
@group(0) @binding(3) var h4pre_tex: texture_2d<f32>;

// The engine's `shaders\default_bitmaps\bitmaps\auto_exposure_weight` (18x10 A8R8G8B8,
// rasg default bitmaps [9]; read from the shipped cache): the centre-weighted window
// `exposure_downsample` multiplies its 12x8 taps by (sampled bilinearly at the tap uv).
const H4_WT: array<f32, 180> = array<f32, 180>(
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0078, 0.0314, 0.0863, 0.1490, 0.2275, 0.2941, 0.3294, 0.3294, 0.2941, 0.2275, 0.1490, 0.0863, 0.0314, 0.0078, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0196, 0.0627, 0.1373, 0.2392, 0.3451, 0.4353, 0.4941, 0.4941, 0.4353, 0.3451, 0.2392, 0.1373, 0.0627, 0.0157, 0.0000, 0.0000,
    0.0000, 0.0078, 0.0392, 0.0980, 0.2000, 0.3333, 0.4902, 0.6157, 0.6902, 0.6902, 0.6157, 0.4902, 0.3333, 0.2000, 0.0980, 0.0314, 0.0078, 0.0000,
    0.0000, 0.0118, 0.0510, 0.1373, 0.2627, 0.4353, 0.6196, 0.7804, 0.8667, 0.8667, 0.7804, 0.6196, 0.4353, 0.2627, 0.1373, 0.0510, 0.0078, 0.0000,
    0.0039, 0.0157, 0.0667, 0.1608, 0.3137, 0.5098, 0.7294, 0.8980, 1.0000, 1.0000, 0.8980, 0.7294, 0.5098, 0.3137, 0.1608, 0.0667, 0.0118, 0.0000,
    0.0039, 0.0196, 0.0667, 0.1686, 0.3255, 0.5294, 0.7529, 0.9373, 1.0000, 1.0000, 0.9373, 0.7529, 0.5294, 0.3255, 0.1686, 0.0667, 0.0157, 0.0000,
    0.0039, 0.0118, 0.0588, 0.1412, 0.2784, 0.4588, 0.6431, 0.8078, 0.8980, 0.8980, 0.8078, 0.6431, 0.4588, 0.2784, 0.1412, 0.0588, 0.0118, 0.0000,
    0.0000, 0.0078, 0.0314, 0.0980, 0.1961, 0.3294, 0.4706, 0.5961, 0.6627, 0.6627, 0.5961, 0.4706, 0.3294, 0.1961, 0.0980, 0.0314, 0.0078, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
);
fn h4_weight(uv: vec2<f32>) -> f32 {
    let p = uv * vec2<f32>(18.0, 10.0) - 0.5;
    let x0 = clamp(floor(p.x), 0.0, 17.0);
    let y0 = clamp(floor(p.y), 0.0, 9.0);
    let x1 = min(x0 + 1.0, 17.0);
    let y1 = min(y0 + 1.0, 9.0);
    let fx = clamp(p.x - x0, 0.0, 1.0);
    let fy = clamp(p.y - y0, 0.0, 1.0);
    let w00 = H4_WT[i32(y0) * 18 + i32(x0)];
    let w10 = H4_WT[i32(y0) * 18 + i32(x1)];
    let w01 = H4_WT[i32(y1) * 18 + i32(x0)];
    let w11 = H4_WT[i32(y1) * 18 + i32(x1)];
    return mix(mix(w00, w10, fx), mix(w01, w11, fx), fy);
}
// Halo 4 ENGINE METER (halo4.dll sub_180380FD0 -> explicit shaders 159 downsample_block_bloom_2x2,
// 49 downsample_4x4_gaussian x2, 35 exposure_downsample; sub_180359768 adaptation): the exposed scene
// c = min(L * 2^s, 8) per channel is weighted w = highlight * luma(c) + inherent + self_illum *
// min(alpha, 1/32) (alpha = the surface's self-illum luminance), box-filtered, then 96 centre-
// weighted taps measure M = (1 - sens) * mean(w log2(luma(w c) + 1e-5)) + sens * log2(mean(w luma(w c))
// + 1e-5) with luma = (0.3086, 0.6094, 0.082); the adaptation moves the stops by
// screen_brightness - M each frame, i.e. converges to M(s) = screen_brightness inside the band.
// `// #h4-expo-2` The log is taken of a LOCAL MEAN, not of a single cell. The engine writes
// `w c` at QUARTER resolution, runs TWO `downsample_4x4_gaussian` passes (each a uniform 8x8 box
// of the previous level with a stride of 4) and then `exposure_downsample` takes 12x8 BILINEAR
// taps of the 1/64 level - so one tap of `log2` sees a ~160-output-pixel-wide average. Taking
// log2 per 48x32 cell instead (33x28 px at 1600x900) let the dark pixels drag the geometric mean
// down (Jensen), M read 0.65 (Ravine) .. 1.38 (Abandon) too LOW, and the solve came out
// 0.1 .. 0.65 stops TOO BRIGHT on every map. Here each tap averages the pre-pass cells under the
// engine's footprint first: the box chain 8 (x) 32 (x) 16 quarter-res texels convolves to a
// separable trapezoid, half-support 28 and flat top 8 quarter-res texels, centred on the tap.
// Residual vs a full CPU replica of the shipped chain: 0.14 (Ravine) / 0.02 (Abandon) in M
// = <0.1 stop, from the pre-pass still applying the clamp / weight at CELL resolution.
const H4_TAP_HALF: f32 = 28.0;   // quarter-res texels
const H4_TAP_FLAT: f32 = 8.0;
fn h4_tap_w(d: f32, half: f32, flat: f32) -> f32 {
    return clamp((half - abs(d)) / max(half - flat, 1e-4), 0.0, 1.0);
}
fn h4_cell_yw(x: i32, y: i32, e: f32) -> f32 {
    let t = textureLoad(h4pre_tex, vec2<i32>(x, y), 0);
    let c = min(max(t.rgb, vec3<f32>(0.0)) * e, vec3<f32>(8.0));
    let yc = dot(c, vec3<f32>(0.3086, 0.6094, 0.082));
    // With a colour-grading LUT bound (every MP cfxs) the highlight weight is
    // the LUT's ALPHA channel sampled at sat(c) (downsample_block_bloom_2x2 variant 1:
    // `sample_l t8.wxyz` at c * 15/16 + 1/32 -> clamps at c = 1) = 0.996 * luma601 of
    // the graded colour; the identity-grade luma601(sat(c)) stands in (the shipped LUTs
    // deviate by < 5 %).
    // `// #h4-expo-2` WITHOUT a LUT the engine's variant 0 is `dp2 r1.x, cb1[1].xzxx, r0.w`:
    // the highlight term multiplies `min(alpha, 1/32)` too, NOT luma(c) -
    // `w = (highlight + self_illum) * min(alpha, 1/32) + inherent`. (No shipped MP cfxs takes
    // this branch; the old `hy = luma(c)` fallback was several stops out on one that would.)
    let si = min(t.a, 0.03125);
    if (post.h4b.z <= 0.5) { return ((post.h4a.x + post.h4a.z) * si + post.h4a.y) * yc; }
    let hy = 0.996 * dot(min(c, vec3<f32>(1.0)), vec3<f32>(0.299, 0.587, 0.114));
    return (post.h4a.x * hy + post.h4a.y + post.h4a.z * si) * yc;
}
fn h4_measure(s: f32) -> f32 {
    let e = exp2(s);
    // Output (engine render-target) size: SS is HMS's own supersampling, not an engine resolution.
    let outp = vec2<f32>(textureDimensions(hdr_tex)) / __SS__;
    // quarter-res texels per pre-pass cell, per axis
    let r = max(outp * 0.25 / vec2<f32>(48.0, 32.0), vec2<f32>(1e-3));
    // Cost guard: below ~700x500 output the engine footprint would exceed the 48x32 pre-pass grid
    // and the tap sum degenerates into a global mean anyway, so cap it at half the grid (a
    // no-op at every real render size - 3.4 x 4.0 cells at 1600x900).
    let half = min(vec2<f32>(H4_TAP_HALF) / r, vec2<f32>(24.0, 16.0));
    let flat = min(vec2<f32>(H4_TAP_FLAT) / r, half);
    // One gather per tap. (A scatter over the 48x32 cells into 96 accumulators reads the pre-pass
    // 4x less but was measurably SLOWER - the dynamically indexed `array<f32, 96>` spills.)
    var slog = 0.0;
    var slin = 0.0;
    var wsum = 0.0;
    for (var ty = 0; ty < 8; ty = ty + 1) {
        let cyf = (f32(ty) + 0.5) / 8.0 * 32.0;
        let y0 = max(i32(floor(cyf - half.y)), 0);
        let y1 = min(i32(ceil(cyf + half.y)), 31);
        for (var tx = 0; tx < 12; tx = tx + 1) {
            let cxf = (f32(tx) + 0.5) / 12.0 * 48.0;
            let x0 = max(i32(floor(cxf - half.x)), 0);
            let x1 = min(i32(ceil(cxf + half.x)), 47);
            var acc = 0.0;
            var aw = 0.0;
            for (var y = y0; y <= y1; y = y + 1) {
                let wy = h4_tap_w(f32(y) + 0.5 - cyf, half.y, flat.y);
                for (var x = x0; x <= x1; x = x + 1) {
                    let ww = wy * h4_tap_w(f32(x) + 0.5 - cxf, half.x, flat.x);
                    acc = acc + ww * h4_cell_yw(x, y, e);
                    aw = aw + ww;
                }
            }
            let yw = acc / max(aw, 1e-6);
            // the `auto_exposure_weight` window sampled at the TAP uv (engine exposure_downsample)
            let w = h4_weight(vec2<f32>((f32(tx) + 0.5) / 12.0, (f32(ty) + 0.5) / 8.0));
            slog = slog + w * log2(yw + 1e-5);
            slin = slin + w * yw;
            wsum = wsum + w;
        }
    }
    let ws = max(wsum, 1e-6);
    return mix(slog / ws, log2(slin / ws + 1e-5), post.h4a.w);
}
// Solve M(s) = screen_brightness for s in the band [p.z, p.w] (M is monotonic in s); returns the
// value the post pass turns back into the gain 2^s through raw_gain = p.y / 2^mean_log.
fn h4_solve() -> f32 {
    let lo = min(post.p.z, post.p.w);
    let hi = post.p.w;
    let sb = post.h4b.x;
    var a = lo;
    var b = hi;
    var s = lo;
    if (h4_measure(lo) >= sb) {
        s = lo;
    } else if (h4_measure(hi) <= sb) {
        s = hi;
    } else {
        // `// #h4-expo-2` 10 halvings: the widest shipped cfxs band is 3.25 stops (Wreckage), so
        // the residual is 3.25 / 2^10 = 0.003 stops - far below the meter's own accuracy, and
        // each evaluation now sums the 12x8 engine taps (4 steps saved = ~0.1 ms/frame).
        for (var k = 0; k < 10; k = k + 1) {
            let m = 0.5 * (a + b);
            if (h4_measure(m) < sb) { a = m; } else { b = m; }
        }
        s = 0.5 * (a + b);
    }
    return log2(max(post.p.y, 1e-6)) - s;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-3.0), vec2<f32>(-1.0,1.0), vec2<f32>(3.0,1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    if (post.h4b.y > 0.5) { return vec4<f32>(h4_solve(), 0.0, 0.0, 1.0); }
    // 16×16 = 256 samples across the frame. Engine meter (exposure_downsample.hlsl_include):
    // a BLEND of the log2 arithmetic-mean and the log2 geometric-mean luminance, weighted by
    // a fixed center/subject-biased weight texture — not a pure geomean, with no
    // brightness-inverse term and no per-sample clamp (only a +1e-5 epsilon):
    //   average.x = Σ w·log2(I)     → log2(geometric mean)
    //   average.y = log2(Σ w·I / Σw)→ log2(arithmetic mean)
    //   return    = average.y·scale.x + average.x·(1−scale.x)
    // scale.x is the tag exposure_downsample register (TODO: tag-drive it); SCALE_X = 0.5 is the
    // interim (HMS_METER_SCALE). The centre weight `cw` stands in for the engine's weight
    // texture (sky/edges downweighted).
    const SCALE_X: f32 = __METER_SCALE__;
    var slog = 0.0; // Σ w·log2(I)   (geomean accumulator)
    var slin = 0.0; // Σ w·I         (arith-mean accumulator)
    var wsum = 0.0;
    for (var y = 0; y < 16; y = y + 1) {
        for (var x = 0; x < 16; x = x + 1) {
            let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / 16.0;
            let c = textureSampleLevel(hdr_tex, hdr_samp, uv, 0.0).rgb;
            let lum = max(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-5);
            // Meter the SUBJECT (gameplay ground, lower-center) and heavily down-weight the SKY
            // + distant peaks (upper frame) so a bright backdrop doesn't drag the meter into
            // darkening the terrain. uv.y: 0=top, 1=bottom. Upper ~35% → ~0.03 weight (sky);
            // lower-center → full. Horizontal edges eased off.
            let vert = mix(0.03, 1.0, smoothstep(0.30, 0.62, uv.y));
            let horiz = mix(1.0, 0.45, smoothstep(0.35, 1.0, abs(uv.x - 0.5) * 2.0));
            let cw = vert * horiz;
            slog = slog + cw * log2(lum);
            slin = slin + cw * lum;
            wsum = wsum + cw;
        }
    }
    let geomean_log = slog / max(wsum, 1e-4);
    let arith_log = log2(max(slin / max(wsum, 1e-4), 1e-5));
    return vec4<f32>(mix(geomean_log, arith_log, SCALE_X), 0.0, 0.0, 1.0);
}
"#;

/// The Halo 4 meter's pre-pass: 48x32 cells, each the mean of 4x4 bilinear taps of the
/// HDR scene (rgb) and of the self-illum coverage 1 - alpha (the Halo 4 opaque lane writes
/// alpha = 1 - min(luma(self_illum), 1); every other lane's 1.0 = no self-illum). Stands in for
/// the engine's 2x2 block + two 4x4 gaussian downsamples ahead of exposure_downsample.
const H4PRE_WGSL: &str = r#"
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-3.0), vec2<f32>(-1.0,1.0), vec2<f32>(3.0,1.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell = vec2<f32>(1.0 / 48.0, 1.0 / 32.0);
    let base = pos.xy * cell; // pos.xy = pixel centre
    var acc = vec4<f32>(0.0);
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let uv = base + ((vec2<f32>(f32(i), f32(j)) + 0.5) / 4.0 - 0.5) * cell;
            let t = textureSampleLevel(hdr_tex, hdr_samp, uv, 0.0);
            acc = acc + vec4<f32>(max(t.rgb, vec3<f32>(0.0)), 1.0 - clamp(t.a, 0.0, 1.0));
        }
    }
    return acc / 16.0;
}
"#;

/// Bloom: the engine's pyramid — a curve level extracted at 1/4 res from the exposed scene,
/// 8x8 box downsamples to 1/16 and 1/64, the separable 11-tap binomial blur at each level and
/// the intensity/colour-scaled additive combine back up (Reach `fs_*` entries), plus the Halo 4
/// chain (`fs_*_h4`). The result is added in the post pass. Texel size comes from
/// `textureDimensions`, so no size uniform is needed.
const BLOOM_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
// second input of the two-input `fs_add` passes (the coarser, blurred level)
@group(0) @binding(2) var add_tex: texture_2d<f32>;
// All bloom passes bind the post uniform + luminance meter: the curve level is evaluated in
// EXPOSED space (the engine blooms its post-exposure LDR target) and the per-map cfxs
// bloom point / inherent / intensity (bloom.zw, bloom2.x) scale specific levels of the chain.
// Halo 4: h4a = (highlight, inherent, self-illum weight, -), h4b = (-, -, LUT bound, H4 chain on);
// sfx.zw = 1 / output size (the engine's stale `vs_texture_size`), bcs.w = intensity resolution scale.
struct BPost { p: vec4<f32>, grade: vec4<f32>, bloom: vec4<f32>, bloom2: vec4<f32>, sfx: vec4<f32>, sfx2: vec4<f32>, bcl: vec4<f32>, bcm: vec4<f32>, bcs: vec4<f32>, cm0: vec4<f32>, cm1: vec4<f32>, cm2: vec4<f32>, h4a: vec4<f32>, h4b: vec4<f32> };
@group(0) @binding(3) var<uniform> bpost: BPost;
@group(0) @binding(4) var blum: texture_2d<f32>;
// The Halo 4 colour-grading volume (post group 1): `downsample_block_bloom_2x2` variant 1 weights
// the highlight term by the LUT's ALPHA sampled at the exposed colour.
@group(1) @binding(0) var blut: texture_3d<f32>;
@group(1) @binding(1) var blut_samp: sampler;
const BRIGHT_EXPOSURE_CAL: f32 = 1.0;
// Replicates resolve()'s per-frame gain EXACTLY so the bright pass extracts in the same exposed
// units the post composite adds bloom into (engine: bloom is extracted AFTER g_exposure, then
// `combined + bloom·8`). In absolute-HDR units bright interior surfaces (Ivory panels, m≫1)
// would bloom catastrophically while the sky (m≈1) looks fine.
fn bright_gain() -> f32 {
    let hi = exp2(bpost.p.w);
    let lo = min(exp2(bpost.p.z), hi);
    let mean_log = textureLoad(blum, vec2<i32>(0, 0), 0).r;
    let raw_gain = bpost.p.y / max(exp2(mean_log), 1e-4);
    let auto_gain = clamp(raw_gain * BRIGHT_EXPOSURE_CAL, lo, hi);
    var g = bpost.p.x * auto_gain;
    if (bpost.grade.w > 1e-6) { g = bpost.grade.w; }
    return g;
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let pos = p[vi];
    var o: VOut;
    o.clip = vec4<f32>(pos, 0.0, 1.0);
    o.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return o;
}

// Engine bloom (reach_tag_test.exe postprocess + HREK postprocess HLSL):
// the PC bloom source is the post-exposure LDR target (E = 8*saturate(LDR) = exposed scene
// clamped at 8), box-averaged 4x4 to 1/4 res, then shaped by the cfxs curve
//   S = 2 * E4 * (bloom_point * lum(E4) + bloom_inherent)        (lum = NTSC (.299,.587,.114))
// (downsample_2x2_block_bloom: no knee, no threshold; the extra x2 is the /4 store read back x8).
// No cfxs on the map (bloom.z == 0): fall back to the engine default cfxs values.
// (engine default cfxs globals\defaults\default: 0.1 / 0.0 / 0.3)
const FALLBACK_POINT: f32 = 0.1;
const FALLBACK_INHERENT: f32 = 0.0;
const FALLBACK_INTENSITY: f32 = 0.3;
fn bloom_point() -> f32 { return select(FALLBACK_POINT, bpost.bloom.z, bpost.bloom.z > 0.0); }
fn bloom_inherent() -> f32 { return select(FALLBACK_INHERENT, bpost.bloom.w, bpost.bloom.z > 0.0 && bpost.bloom.w >= 0.0); }
fn bloom_intensity() -> f32 { return select(FALLBACK_INTENSITY, bpost.bloom2.x, bpost.bloom2.x > 0.0); }
// Engine per-pass store: min(4x, 8)/32 in fp16, read back x8 -> identity with a clamp at 2.0.
const PASS_CAP: f32 = 2.0;
@fragment
fn fs_bright(i: VOut) -> @location(0) vec4<f32> {
    // 4x4 box of the exposed scene: four bilinear taps at +-1 full-res texel around the 1/4-res
    // texel centre (each tap averages 2x2). Each tap is clamped at 8 (the LDR saturate).
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let g = bright_gain();
    var e = min(max(textureSample(src_tex, src_samp, i.uv + vec2<f32>(-t.x, -t.y)).rgb, vec3<f32>(0.0)) * g, vec3<f32>(8.0));
    e += min(max(textureSample(src_tex, src_samp, i.uv + vec2<f32>( t.x, -t.y)).rgb, vec3<f32>(0.0)) * g, vec3<f32>(8.0));
    e += min(max(textureSample(src_tex, src_samp, i.uv + vec2<f32>(-t.x,  t.y)).rgb, vec3<f32>(0.0)) * g, vec3<f32>(8.0));
    e += min(max(textureSample(src_tex, src_samp, i.uv + vec2<f32>( t.x,  t.y)).rgb, vec3<f32>(0.0)) * g, vec3<f32>(8.0));
    let e4 = e * 0.25;
    let lum = dot(e4, vec3<f32>(0.299, 0.587, 0.114));
    let bs = bloom_point() * lum + bloom_inherent();
    // stored = min(out/4, 8) read back x8 -> 2*out, cap 64
    return vec4<f32>(min(2.0 * e4 * bs, vec3<f32>(64.0)), 1.0);
}

// Engine blur_11_horizontal + kernel_5 (vertical): the net separable kernel is the 11-tap
// binomial [1 10 45 120 210 252 210 120 45 10 1]/1024 at unit texel spacing of the level.
fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let d = dir * texel;
    let w0 = 252.0 / 1024.0;
    let w1 = 210.0 / 1024.0;
    let w2 = 120.0 / 1024.0;
    let w3 =  45.0 / 1024.0;
    let w4 =  10.0 / 1024.0;
    let w5 =   1.0 / 1024.0;
    var sum = textureSample(src_tex, src_samp, uv).rgb * w0;
    sum += (textureSample(src_tex, src_samp, uv + d).rgb
          + textureSample(src_tex, src_samp, uv - d).rgb) * w1;
    sum += (textureSample(src_tex, src_samp, uv + d * 2.0).rgb
          + textureSample(src_tex, src_samp, uv - d * 2.0).rgb) * w2;
    sum += (textureSample(src_tex, src_samp, uv + d * 3.0).rgb
          + textureSample(src_tex, src_samp, uv - d * 3.0).rgb) * w3;
    sum += (textureSample(src_tex, src_samp, uv + d * 4.0).rgb
          + textureSample(src_tex, src_samp, uv - d * 4.0).rgb) * w4;
    sum += (textureSample(src_tex, src_samp, uv + d * 5.0).rgb
          + textureSample(src_tex, src_samp, uv - d * 5.0).rgb) * w5;
    return sum;
}

@fragment
fn fs_blur_h(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(min(blur(i.uv, vec2<f32>(1.0, 0.0)), vec3<f32>(PASS_CAP)), 1.0);
}

@fragment
fn fs_blur_v(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(min(blur(i.uv, vec2<f32>(0.0, 1.0)), vec3<f32>(PASS_CAP)), 1.0);
}

// kernel_5 with scale = (intensity * large_colour, 1): the coarsest level is scaled here.
@fragment
fn fs_blur_v_i(i: VOut) -> @location(0) vec4<f32> {
    // scale = (intensity * BLOOM LARGE COLOR, 1): the engine default large colour is BLACK, so
    // the 1/64 level is normally off (Ivory/Forge author ~0.04) -- dark doorways stay dark.
    return vec4<f32>(min(blur(i.uv, vec2<f32>(0.0, 1.0)) * bloom_intensity() * bpost.bcl.rgb, vec3<f32>(PASS_CAP)), 1.0);
}

// downsample_4x4_gaussian at a 4:1 size ratio: 16 bilinear taps at (+-1,+-3)x(+-1,+-3) SOURCE
// texels around the destination centre = a uniform 8x8 box of source texels.
@fragment
fn fs_box8(i: VOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var s = vec3<f32>(0.0);
    for (var y: i32 = 0; y < 4; y = y + 1) {
        let oy = f32(y * 2 - 3); // -3, -1, 1, 3
        for (var x: i32 = 0; x < 4; x = x + 1) {
            let ox = f32(x * 2 - 3);
            s += textureSample(src_tex, src_samp, i.uv + vec2<f32>(ox * t.x, oy * t.y)).rgb;
        }
    }
    return vec4<f32>(min(s / 16.0, vec3<f32>(PASS_CAP)), 1.0);
}

// env-cube mip chain only (tex-only layout): 4 bilinear taps at +-1 source texel = 4x4 box.
@fragment
fn fs_downsample_env(i: VOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var s = textureSample(src_tex, src_samp, i.uv + vec2<f32>(-t.x, -t.y)).rgb;
    s += textureSample(src_tex, src_samp, i.uv + vec2<f32>( t.x, -t.y)).rgb;
    s += textureSample(src_tex, src_samp, i.uv + vec2<f32>(-t.x,  t.y)).rgb;
    s += textureSample(src_tex, src_samp, i.uv + vec2<f32>( t.x,  t.y)).rgb;
    return vec4<f32>(s * 0.25, 1.0);
}

// bloom_add_alpha1 / add: colour = scale.rgb * original + bilinear-upsampled(add), scale =
// (intensity * level colour (white), 1).
// 1/16 level: scale = intensity * BLOOM MEDIUM COLOR (bloom_add_alpha1)
@fragment
fn fs_add_m(i: VOut) -> @location(0) vec4<f32> {
    let orig = textureSample(src_tex, src_samp, i.uv).rgb;
    let add = textureSample(add_tex, src_samp, i.uv).rgb;
    return vec4<f32>(min(orig * bloom_intensity() * bpost.bcm.rgb + add, vec3<f32>(PASS_CAP)), 1.0);
}
// 1/4 level: scale = intensity * BLOOM SMALL COLOR (add)
@fragment
fn fs_add_s(i: VOut) -> @location(0) vec4<f32> {
    let orig = textureSample(src_tex, src_samp, i.uv).rgb;
    let add = textureSample(add_tex, src_samp, i.uv).rgb;
    return vec4<f32>(min(orig * bloom_intensity() * bpost.bcs.rgb + add, vec3<f32>(PASS_CAP)), 1.0);
}

// ---- Halo 4 bloom (MCC halo4.dll sub_1803812B4 / sub_180380FD0 + the explicit shaders 159
// downsample_block_bloom_2x2, 49 downsample_4x4_gaussian, 11 blur_11_horizontal, 34 kernel_5,
// 66 bloom_add_alpha1, 10 add; final_composite `c = surface * 8 + bloom`) ------------------------
// Same pyramid as Reach (aux_small 1/4 -> aux_tiny 1/16 -> aux_mini 1/64 -> back up), fp16 targets
// (no per-pass cap), every level carries an ALPHA lane = the LUT-alpha "highlight luma" of the
// curve level that the `add` pass multiplies the finest level by.
//   curve:   c = 4x4 box of min(E L, 8) (the LDR store, two 2x2 downsamples), A = LUT(c).a (no
//            LUT: no highlight term at all), w = hl A + inh + si min(a_si, 1/32); out = (w c, A)
//   gauss:   16 bilinear taps at +-1,+-3 source texels = a uniform 8x8 box (rgba)
//   kernel5: taps (0.5, -0.5 + {-3.6, -1.8, 0, 1.8, 3.6}) level texels, weights 5 60 126 60 5 / 256
//            (cb Kernel5PS @0x180D5E9B0), output x scale (large colour x intensity, alpha x 1)
//   blur_h:  the same weights, centre at (-0.5, 0.5) level texels, outer taps at {-2.3, 1.3,
//            -4.1, 3.1} x `vs_texture_size.zw` - which the DX11 port never sets for the blur passes
//            (only the composite: 1 / output size), so the horizontal spread is in OUTPUT texels
//   add_a1:  1/16 = medium I orig + up(add); a = luma(medium I) orig.a + add.a
//   add:     1/4  = small I orig.rgb add.a + up(add).rgb; a = add.a
//   I = cfxs intensity x min(1, 921600 / (W H)) (render resolution above 720p)
fn h4_intensity() -> f32 { return bpost.bloom2.x * select(1.0, bpost.bcs.w, bpost.bcs.w > 0.0); }
const H4_W0: f32 = 126.0 / 256.0;
const H4_W1: f32 = 60.0 / 256.0;
const H4_W2: f32 = 5.0 / 256.0;

@fragment
fn fs_bright_h4(i: VOut) -> @location(0) vec4<f32> {
    // the 1/4-res pixel covers 4x4 OUTPUT pixels = 8x8 HDR texels (SS = 2): 16 bilinear taps at
    // the 2x2-block corners = one output pixel each, clamped at 8 like the engine's LDR store
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let g = bright_gain();
    var c = vec3<f32>(0.0);
    var a = 0.0;
    for (var y = 0; y < 4; y = y + 1) {
        for (var x = 0; x < 4; x = x + 1) {
            let uv = i.uv + (vec2<f32>(f32(x), f32(y)) - 1.5) * 2.0 * t;
            let s = textureSample(src_tex, src_samp, uv);
            c += min(max(s.rgb, vec3<f32>(0.0)) * g, vec3<f32>(8.0));
            a += 1.0 - clamp(s.a, 0.0, 1.0);
        }
    }
    c *= 0.0625;
    a = min(a * 0.0625, 0.03125);
    var hw = 0.0;
    if (bpost.h4b.z > 0.5) {
        let n = vec3<f32>(textureDimensions(blut));
        hw = textureSampleLevel(blut, blut_samp, c * (n - 1.0) / n + 0.5 / n, 0.0).a;
    } else {
        hw = a; // variant 0: dp2((hl, si), a_si) + inherent
    }
    let w = bpost.h4a.x * hw + bpost.h4a.y + bpost.h4a.z * a;
    return vec4<f32>(w * c, hw);
}

@fragment
fn fs_box8_h4(i: VOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var s = vec4<f32>(0.0);
    for (var y: i32 = 0; y < 4; y = y + 1) {
        let oy = f32(y * 2 - 3);
        for (var x: i32 = 0; x < 4; x = x + 1) {
            let ox = f32(x * 2 - 3);
            s += textureSample(src_tex, src_samp, i.uv + vec2<f32>(ox * t.x, oy * t.y));
        }
    }
    return s / 16.0;
}

// blur_11_horizontal: centre (-0.5, 0.5) level texels; outer taps in OUTPUT texels (sfx.zw)
@fragment
fn fs_blur_h_h4(i: VOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let tf = select(t, bpost.sfx.zw, bpost.sfx.z > 0.0);
    let c = i.uv + vec2<f32>(-0.5, 0.5) * t;
    var s = textureSample(src_tex, src_samp, c) * H4_W0;
    s += (textureSample(src_tex, src_samp, c + vec2<f32>(-2.3 * tf.x, 0.0)) + textureSample(src_tex, src_samp, c + vec2<f32>(1.3 * tf.x, 0.0))) * H4_W1;
    s += (textureSample(src_tex, src_samp, c + vec2<f32>(-4.1 * tf.x, 0.0)) + textureSample(src_tex, src_samp, c + vec2<f32>(3.1 * tf.x, 0.0))) * H4_W2;
    return s;
}

// kernel_5: (0.5, y) level texels, y = -4.1 -2.3 -0.5 1.3 3.1
fn h4_kernel5(uv: vec2<f32>) -> vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var s = textureSample(src_tex, src_samp, uv + vec2<f32>(0.5, -0.5) * t) * H4_W0;
    s += (textureSample(src_tex, src_samp, uv + vec2<f32>(0.5, -2.3) * t) + textureSample(src_tex, src_samp, uv + vec2<f32>(0.5, 1.3) * t)) * H4_W1;
    s += (textureSample(src_tex, src_samp, uv + vec2<f32>(0.5, -4.1) * t) + textureSample(src_tex, src_samp, uv + vec2<f32>(0.5, 3.1) * t)) * H4_W2;
    return s;
}
@fragment
fn fs_blur_v_h4(i: VOut) -> @location(0) vec4<f32> { return h4_kernel5(i.uv); }
// 1/64 level: scale = (large colour x I, 1)
@fragment
fn fs_blur_v_i_h4(i: VOut) -> @location(0) vec4<f32> {
    return h4_kernel5(i.uv) * vec4<f32>(bpost.bcl.rgb * h4_intensity(), 1.0);
}
// bloom_add_alpha1 (1/16): scale = medium x I
@fragment
fn fs_add_m_h4(i: VOut) -> @location(0) vec4<f32> {
    let orig = textureSample(src_tex, src_samp, i.uv);
    let add = textureSample(add_tex, src_samp, i.uv);
    let sc = bpost.bcm.rgb * h4_intensity();
    return vec4<f32>(sc, dot(sc, vec3<f32>(0.299, 0.587, 0.114))) * orig + add;
}
// add (1/4): rgb = small I orig.rgb x add.a + add.rgb; a = add.a
@fragment
fn fs_add_s_h4(i: VOut) -> @location(0) vec4<f32> {
    let orig = textureSample(src_tex, src_samp, i.uv);
    let add = textureSample(add_tex, src_samp, i.uv);
    return vec4<f32>(orig.rgb * bpost.bcs.rgb * h4_intensity() * add.a + add.rgb, add.a);
}
"#;

#[cfg(test)]
mod wgsl_tests {
    //! Offline WGSL validation. `cargo build` compiles only the Rust host code;
    //! the shader strings are parsed by naga at pipeline-creation time on a live
    //! device. These tests run naga's front-end + validator so a shader typo is
    //! caught in CI without a GPU.
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    fn check(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: WGSL validation error: {e:?}"));
    }

    /// The mesh/water/terrain shaders are TEMPLATES: pipeline creation substitutes `__NAME__`
    /// placeholders (debug lanes, tuning knobs) before handing the source to naga. Substitute
    /// the same DEFAULTS the pipelines use so the validated source is what a default run compiles.
    fn subst(src: &str) -> String {
        let mut s = src.to_string();
        for (k, v) in [
            ("__TDDBG__", "0.0"), ("__MDBG__", "0.0"), ("__TDBG__", "0"), ("__WDBG__", "0"),
            ("__GI_DEBUG__", "0"), ("__DECO_CX__", "0.5000"), ("__DECO_CY__", "0.5000"),
        ] {
            s = s.replace(k, v);
        }
        s
    }

    /// `// #h4-expo-3` `FOG_WU_DEFAULT` is the CPU mirror of the fog shader's own constant, so a
    /// UI can seed a slider from the density that is really rendering. Pin them together: a drift
    /// would make opening Settings > Lighting change the fog again (the bug this replaced).
    #[test]
    fn fog_wu_default_matches_shader() {
        let want = format!("const FOG_WU: f32 = {};", super::FOG_WU_DEFAULT);
        assert!(crate::mesh::FOG_WGSL.contains(&want), "FOG_WGSL no longer declares `{want}`");
    }

    #[test]
    fn all_shaders_valid() {
        check("BASIC_WGSL", super::BASIC_WGSL);
        check("WIRE_WGSL", super::WIRE_WGSL);
        check("ZONE_WGSL", super::ZONE_WGSL);
        check("SKY_WGSL", super::SKY_WGSL);
        check("POST_WGSL", super::POST_WGSL);
        check("BLOOM_WGSL", super::BLOOM_WGSL);
        check("LUM_WGSL", &super::LUM_WGSL.replace("__METER_SCALE__", "0.5").replace("__SS__", "2.0"));
        check("H4PRE_WGSL", super::H4PRE_WGSL);
        // Fog / shadow / GI are shared modules prepended at pipeline creation; validate the same
        // concatenated + substituted form the GPU sees.
        let fog = crate::mesh::FOG_WGSL;
        let shad = crate::mesh::SHADOW_SAMPLE_WGSL;
        let gi = format!("{}{}", crate::rtgi::GI_FRAGMENT_WGSL, crate::rtgi::GI_SHARED_WGSL);
        // the Halo 4 material lane is appended to the mesh module at pipeline creation
        check("MESH_WGSL", &subst(&format!("{fog}{shad}{gi}{}{}", crate::mesh::MESH_WGSL, crate::mesh::H4_MESH_WGSL)).replace("__H4DBG__", "0.0"));
        check("WATER_WGSL", &subst(&format!("{fog}{gi}{}", crate::mesh::WATER_WGSL)));
        check("TERRAIN_WGSL", &subst(&format!("{fog}{shad}{gi}{}", crate::mesh::TERRAIN_WGSL)));
        check("SKY_MESH_WGSL", crate::mesh::SKY_MESH_WGSL);
        check("SHADOW_WGSL", crate::mesh::SHADOW_WGSL);
    }
}
