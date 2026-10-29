//! Real-time probe global illumination (docs/hrek_re/18_realtime_gi_plan.md).
//!
//! A regular grid of light probes over the level. Every frame a compute pass shoots rays from each
//! probe through the same BVH the offline path-traced lightmapper uses (`lightbake_gpu`), shades the
//! hits with direct sun / scenario lights / emissive fixtures + the PREVIOUS frame's probe irradiance
//! (which is how multi-bounce accumulates for free), and blends the result into two per-probe records:
//!   * irradiance as 2nd-order SH (9 × RGB), and
//!   * a DDGI-style octahedral VISIBILITY map (8×8 texels of mean / mean² ray distance) that the lookup
//!     uses for a Chebyshev test so probes behind walls do not leak into rooms.
//! The shading side (`GI_WGSL`, prepended to the mesh/terrain/water shaders) replaces the baked
//! lightmap term with `gi_irradiance(p, n, v)` when `gi.dims.w > 0` — the result is in the SAME units
//! as the baked `bake.rgb` (irradiance / π), so exposure, bloom and tonemap are untouched.
//!
//! All probe data lives in storage buffers bound in the CAMERA bind group (bindings 10–12), so no
//! pipeline layout or draw call had to change.
//!
//! ENGINE MODE (the scene carries the map's sky render_model light list, `engine_gi`): a probe is lit the
//! way the Tool's lightmapper lights one. Its record holds (slot map above) the SKY-DIRECT SH (every record
//! of the list seen through the scene, the sun excluded — it is analytic), the GATHER SH (surfaces' exit
//! radiance: albedo x (sun x `sun 1st bounce` + sky x `sky 1st bounce` + lights)/pi, higher bounces through
//! the previous gather field with the photon albedo clamp 0.8) and their sum. Rays that escape carry nothing.
//! Placed Forge pieces live in a second BVH region (`set_dynamic`) and an edit re-estimates only the probes
//! it reaches (`invalidate_region`). Without a sky list the legacy model (sky cube / flat ambient) runs.

use eframe::wgpu;
use wgpu::util::DeviceExt;

pub use crate::lightbake_gpu::{GpuEmitter, GpuLight};
use crate::engine_gi::SkyLights;

/// Octahedral visibility map resolution per probe (texels per side).
pub const VIS_RES: u32 = 8;
pub const VIS_TEXELS: u32 = VIS_RES * VIS_RES;
/// vec4 slots per probe: 9 SH coefficients (L2, rgb) + slot 9 = relocation offset xyz, w = VALID flag (0 = inside geometry).
pub const SH_N: u32 = 28;
/// Slot map of one probe record (vec4 each): `0..9` the TOTAL SH the shading side reads (sky direct
/// + gather), `9` the relocation offset xyz / VALID flag, `10..19` the GATHER-only SH (bounce light
/// the probe received from surfaces; the recursion source for the next bounce), `19..28` the
/// engine-mode SKY-DIRECT SH (the sky-lights list seen through the scene; deterministic).
pub const SLOT_TOTAL: u32 = 0;
pub const SLOT_RELOC: u32 = 9;
pub const SLOT_GATHER: u32 = 10;
pub const SLOT_SKY: u32 = 19;
/// RAY-TRACED SUN SHADOW MAPS (see `update_sun`): cascades of an orthographic first-hit depth map seen
/// from the sun, traced through the SAME BVH as the probes (static map + the placed Forge pieces).
pub const SUN_CASCADES: usize = 4;
pub const SUN_RES: u32 = 1024;
/// Half extent (wu) of each cascade's square window: texels of 0.031 / 0.094 / 0.28 / 0.84 wu (a x3 step each: the
/// handoff to the next cascade is at most 3x coarser, <= 3.4 px per texel at 1080p / 90 deg FOV).
pub const SUN_HALF: [f32; SUN_CASCADES] = [16.0, 48.0, 144.0, 432.0];
/// Rays start this far along +sun from the world origin (the map is a few thousand wu across).
pub const SUN_ORIGIN_DIST: f32 = 5000.0;
/// bytes of the buffer header: (cascade windows + 1) x vec4
const SUN_HDR_BYTES: u64 = 16 * (SUN_CASCADES as u64 + 1);
/// The placeholder buffers bound in the camera group before a scene exists. wgpu checks every draw against the
/// SHADER's minimum binding size (a struct ending in a runtime array needs its header plus ONE element), so each
/// must be at least that big: the sun placeholder was 8 bytes short when its texels grew from vec2 to vec4 and the
/// first object draw panicked (`Buffer is bound with size 72 where the shader expects 80`).
/// `the_camera_group_buffers_are_at_least_the_size_the_shaders_declare` pins them against the WGSL.
pub(crate) const SUN_DUMMY_BYTES: u64 = SUN_HDR_BYTES + 16;
pub(crate) const PROBES_DUMMY_BYTES: u64 = (SH_N as u64) * 16;
pub(crate) const VIS_DUMMY_BYTES: u64 = (VIS_TEXELS as u64) * 8;
/// Hard cap on the probe count (the update dispatch is one workgroup per probe).
pub const MAX_PROBES: u32 = 32_768;
/// Capacity (triangles / BVH nodes / materials) of the DYNAMIC occluder region appended behind the
/// static scene in the tracer buffers: the Forge pieces the user places, moves and deletes. Fixed so
/// an edit rewrites bytes in place (`queue.write_buffer`) instead of rebuilding bind groups.
pub const DYN_TRI_CAP: u32 = 131_072;
pub const DYN_NODE_CAP: u32 = 131_072;
pub const DYN_MAT_CAP: u32 = 1_024;
/// Most sky-light records the engine-mode tracer carries (the workgroup holds one RGB per record).
pub const MAX_SKY_RECORDS: usize = 256;

/// The camera-centred INNER probe grid: a fine lattice (`spacing` wu, `dims` cells) that follows the camera as a toroidal
/// window, blended into the coarse map-wide grid over its outer 3.5 cells. It exists because the coarse grid (about 9 wu on
/// Forge World) lets sky light leak into shadowed nooks and pits (measured: 29 % of sky-clear shadowed points more than 1.5x
/// the shipped atlas, 18 % with a 2.5 wu grid). Its probes COUNT against `MAX_PROBES`, so the per-frame tracing budget
/// (`parts_per_frame`: 4096 probes of 64 x rays-per-thread rays) does not change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnerGrid { pub spacing: f32, pub dims: [u32; 3] }
impl InnerGrid {
    pub fn count(&self) -> u32 { self.dims[0] * self.dims[1] * self.dims[2] }
}
/// Default inner grid and the coarse spacing above which it is worth having.
/// (3.0 wu x 32 x 32 x 16 = 96 x 96 x 48 wu around the camera, 16384 probes = half the cap; measured on Forge World against the shipped atlas,
/// together with the baked-light term: mean |ln(ratio)| 0.39 -> 0.32; see `rtgi_shadow_fill_vs_the_shipped_atlas`)
pub const INNER_DEFAULT: InnerGrid = InnerGrid { spacing: 3.0, dims: [32, 32, 16] };
pub const INNER_MIN_OUTER_SPACING: f32 = 6.0;
/// How the inner grid is chosen: `Auto` = the default one on big maps (coarse spacing >= `INNER_MIN_OUTER_SPACING`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InnerPolicy { Auto, Off, Fixed(InnerGrid) }
#[derive(Default)]
struct InnerState { init: bool, base: [i32; 3] }

/// Per-dispatch / per-frame GI parameters (std140; matches `GiParams` in the WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiParams {
    pub origin: [f32; 4],    // grid origin xyz, w = probe spacing (wu)
    pub dims: [u32; 4],      // nx, ny, nz, mode (0 = off, 1 = on)
    pub sun_dir: [f32; 4],   // to-sun, normalized
    pub sun_color: [f32; 4], // HDR sun radiance (irradiance on a facing surface)
    pub sky_color: [f32; 4], // sky radiance for escaped rays; w = factor for DOWNWARD misses
    pub misc: [u32; 4],      // frame, rays per thread (1..4), node_count, tri_count
    pub misc2: [u32; 4],     // light_count, emitter_count, probe_count, 0
    pub fog: [f32; 4],       // sigma, fog_top_z, total_emitter_weight, temporal blend alpha (0..1)
    pub bias: [f32; 4],      // normal bias (× spacing), view bias (× spacing), vis max-distance (× spacing), energy scale
    pub nee: [f32; 4],       // emitter NEE samples per hit, per-sample irradiance cap, 0, 0
    /// x = root node of the dynamic occluder BVH (0xFFFFFFFF = none), y = vec4 index of the sky header
    /// inside the `mats` buffer, z = index of the probe list inside `tri_mat`, w = flags (bit0 the
    /// dispatch reads its probes from the list, bit1 refresh the sky-direct SH of every probe).
    pub dynp: [u32; 4],
    /// INNER (camera-centred) probe grid, a toroidal window of cells of the lattice anchored at `inner_org.xyz` with
    /// spacing `inner_org.w` (0 = no inner grid); `inner_dims` = cells per axis + the index of its first probe;
    /// `inner_base` = the integer cell coordinate of the window's minimum corner.
    pub inner_org: [f32; 4],
    pub inner_dims: [u32; 4],
    pub inner_base: [i32; 4],
}

impl Default for GiParams {
    fn default() -> Self {
        GiParams {
            origin: [0.0, 0.0, 0.0, 1.0],
            dims: [1, 1, 1, 0],
            sun_dir: [0.0, 0.0, 1.0, 0.0],
            sun_color: [0.0; 4],
            sky_color: [0.0, 0.0, 0.0, 0.25],
            misc: [0, 1, 0, 0],
            misc2: [0, 0, 1, 0],
            fog: [0.0, 0.0, 0.0, 1.0],
            bias: [0.2, 0.3, 1.5, 1.0],
            nee: [2.0, 12.0, 0.0, 0.0],
            dynp: [0xFFFF_FFFF, 0, 0, 0],
            inner_org: [0.0; 4],
            inner_dims: [1, 1, 1, 0],
            inner_base: [0; 4],
        }
    }
}

/// Everything the probe tracer needs about a map. Built by the app (scene.rs `rtgi_scene_data`).
pub struct RtGiScene {
    /// Flat BVH bytes (`lightbake::BvhNode[]`) + reordered triangles (`BvhTri[]`).
    pub nodes: Vec<u8>,
    pub tris: Vec<u8>,
    /// Per (reordered) triangle: index into `mats`.
    pub tri_mat: Vec<u32>,
    /// Per material THREE vec4: [linear albedo rgb, flags (0 opaque, 1 foliage, 2 sky shell = never
    /// hit, 3 translucent)], [self-illum radiance rgb, 0], [transmission tint rgb, 0].
    pub mats: Vec<[f32; 4]>,
    pub lights: Vec<GpuLight>,
    pub emitters: Vec<GpuEmitter>,
    pub total_emitter_weight: f32,
    /// World bounds of the NON-sky occluders (the probe grid covers this box).
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
    pub sky_color: [f32; 3],
    pub fog_sigma: f32,
    pub fog_top: f32,
    /// The map's sky render_model light list + first-bounce scalers. `Some` = ENGINE mode: the probes
    /// are lit the way the Tool's lightmapper lights a probe (see `engine_gi`); `None` (a map with no
    /// usable sky list) = the legacy sky-cube / flat-ambient escape model.
    pub sky: Option<SkyLights>,
}

/// One probe as the tracer stores it: SH of incident radiance in the graphics basis (see
/// `engine_gi::from_graphics_sh`), RGB per coefficient.
#[derive(Clone, Debug)]
pub struct ProbeRecord {
    /// what the shading side reads: sky-direct + gather
    pub total: [[f32; 3]; 9],
    /// bounce light only (surfaces' exit radiance seen from the probe)
    pub gather: [[f32; 3]; 9],
    /// the sky-lights list seen through the scene (engine mode; zero in legacy mode)
    pub sky: [[f32; 3]; 9],
    /// relocation offset from the nominal grid position
    pub offset: [f32; 3],
    pub valid: bool,
}

/// The dynamic occluders (placed Forge pieces): a BVH over world triangles built WITHOUT the static
/// scene, `local` node/leaf indices. `nodes` is `lightbake::BvhNode[]` bytes, `tris` `BvhTri[]`.
pub struct RtGiDynamic {
    pub nodes: Vec<u8>,
    pub tris: Vec<u8>,
    /// Per (reordered) triangle: index into `mats` (3 vec4 each, like the static table).
    pub tri_mat: Vec<u32>,
    pub mats: Vec<[f32; 4]>,
}

struct SceneBufs {
    nodes: wgpu::Buffer,
    tris: wgpu::Buffer,
    tri_mat: wgpu::Buffer,
    mats: wgpu::Buffer,
    _lights: wgpu::Buffer,
    _emitters: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

pub struct RtGi {
    bgl: wgpu::BindGroupLayout,
    _dummy_cube: wgpu::Texture,
    dummy_cube_view: wgpu::TextureView,
    dummy_samp: wgpu::Sampler,
    pipeline: wgpu::ComputePipeline,
    pub params: GiParams,
    params_buf: wgpu::Buffer,
    probes_buf: wgpu::Buffer,
    vis_buf: wgpu::Buffer,
    scene: Option<SceneBufs>,
    pub enabled: bool,
    frame: std::cell::Cell<u32>,
    /// rays per thread per frame (64 threads per probe → 64·rpt rays per probe per frame)
    pub rays_per_thread: u32,
    /// temporal hysteresis (0.9 = keep 90% of the previous frame)
    pub hysteresis: f32,
    /// number of update passes already run since `set_scene` (warm-up bookkeeping)
    pub updates: std::cell::Cell<u32>,
    pub nprobes: u32,
    pub spacing: f32,
    pub dims: [u32; 3],
    /// per-frame GPU time is not measured; this is the last dispatch's probe count for diagnostics
    pub last_dispatch_probes: std::cell::Cell<u32>,
    /// slice updates still owed a full-weight (alpha = 1) pass after a lighting change
    kick: std::cell::Cell<u32>,
    /// ENGINE mode (the scene carried a sky list): sky-direct SH + gather SH, no escape radiance.
    pub engine: bool,
    /// static counts, so the dynamic region can be appended behind them
    n_nodes: u32,
    n_tris: u32,
    n_mats: u32,
    /// vec4 index of the sky header inside `mats`, index of the probe list inside `tri_mat`
    sky_base: u32,
    list_base: u32,
    /// nodes / tris of the dynamic region currently live (0 = none)
    dyn_nodes: std::cell::Cell<u32>,
    dyn_tris: std::cell::Cell<u32>,
    /// probe indices owed a full-quality re-estimate after an edit, and how many passes already ran
    region: std::cell::RefCell<Option<(Vec<u32>, u32)>>,
    /// the inner grid in use (None = single coarse grid), how it is chosen, its window, and the probes whose cell was
    /// just handed over by a window move (traced from scratch before anything else)
    pub inner: Option<InnerGrid>,
    pub inner_policy: InnerPolicy,
    inner_state: std::cell::RefCell<InnerState>,
    pending_new: std::cell::RefCell<Vec<u32>>,
    pub n_outer: u32,
    /// ray-traced sun shadow cascades: pipeline, layout, the buffer the shading side reads (binding 15 of
    /// the camera group) and its update state
    sun_pipeline: wgpu::ComputePipeline,
    sun_bgl: wgpu::BindGroupLayout,
    sun_buf: wgpu::Buffer,
    sun_bg: Option<wgpu::BindGroup>,
    sun_state: std::cell::RefCell<SunState>,
}

/// Where each cascade's window currently is (light-plane coordinates) and which need a refresh.
#[derive(Default)]
struct SunState {
    centers: [[f32; 2]; SUN_CASCADES],
    valid: [bool; SUN_CASCADES],
    dirty: [bool; SUN_CASCADES],
    tan_half: f32,
    /// ray-traced sun maps off (e.g. no scene): the header says inactive, fragments see a lit sun
    active: bool,
}

/// Orthonormal light-plane basis for a unit direction TOWARD the sun; the WGSL (`gi_sun_basis_x`) is the
/// same formula so compute and shading agree on texel coordinates.
pub fn sun_basis(l: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if l[2].abs() > 0.95 { [1.0f32, 0.0, 0.0] } else { [0.0f32, 0.0, 1.0] };
    let c = [up[1] * l[2] - up[2] * l[1], up[2] * l[0] - up[0] * l[2], up[0] * l[1] - up[1] * l[0]];
    let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let x = [c[0] / n, c[1] / n, c[2] / n];
    let y = [l[1] * x[2] - l[2] * x[1], l[2] * x[0] - l[0] * x[2], l[0] * x[1] - l[1] * x[0]];
    (x, y)
}

/// tan of the sun disc's half angle: the Tool shoots sun rays in a cone of half angle `acos(1 - Omega/2pi)`
/// (`faux_light_sky.cpp`, `sub_1401904E0`: `v5 = acos(1 - a1[40]*0.5/pi)`), Omega = the sun record's solid angle.
pub fn sun_tan_half(omega: f32) -> f32 {
    let c = (1.0 - omega * 0.5 / std::f32::consts::PI).clamp(-1.0, 1.0);
    let a = c.acos();
    a.tan()
}

/// Passes of a region update (blend 1, 2/3, 1/2, 2/5: a running mean of four independent ray sets).
pub const REGION_PASSES: u32 = 4;

impl RtGi {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtgi-probe-update"),
            source: wgpu::ShaderSource::Wgsl(compute_source(std::env::var("HMS_RTGI_CDEBUG").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0)).into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtgi-bgl"),
            entries: &[
                storage(0, true),  // nodes
                storage(1, true),  // tris
                storage(2, true),  // tri_mat
                storage(3, true),  // mats
                storage(4, true),  // lights
                storage(5, true),  // emitters
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage(7, false), // probes (read_write)
                storage(8, false), // vis (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::Cube, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        // 1x1 black cube + sampler for when no sky capture is available
        let dummy_cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rtgi-dummy-cube"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_cube_view = dummy_cube.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let dummy_samp = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rtgi-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rtgi-probe-update"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // sun shadow pipeline: a different layout (5 storage + 2 uniform, under the default per-stage limit)
        let sun_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        let sun_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rtgi-sun-bgl"),
            entries: &[
                sun_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                sun_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                sun_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                sun_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                sun_entry(6, wgpu::BufferBindingType::Uniform),
                sun_entry(11, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let sun_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("rtgi-sun-pl"), bind_group_layouts: &[&sun_bgl], push_constant_ranges: &[] });
        let sun_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rtgi-sun-shadow"), layout: Some(&sun_pl), module: &shader, entry_point: Some("sun_main"), compilation_options: Default::default(), cache: None,
        });
        let sun_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-sun-dummy"), size: SUN_DUMMY_BYTES, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let params = GiParams::default();
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtgi-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Dummy (one-probe) records so the camera bind group is valid before a scene is set.
        let probes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-probes"),
            size: PROBES_DUMMY_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-vis"),
            size: VIS_DUMMY_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rpt: u32 = std::env::var("HMS_RTGI_RAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(2u32).clamp(1, 4);
        let hyst: f32 = std::env::var("HMS_RTGI_HYST").ok().and_then(|v| v.parse().ok()).unwrap_or(0.92f32).clamp(0.0, 0.99);
        RtGi {
            bgl, _dummy_cube: dummy_cube, dummy_cube_view, dummy_samp,
            pipeline, params, params_buf, probes_buf, vis_buf, scene: None, enabled: false,
            frame: std::cell::Cell::new(0), rays_per_thread: rpt, hysteresis: hyst,
            updates: std::cell::Cell::new(0), nprobes: 1, spacing: 1.0, dims: [1, 1, 1],
            last_dispatch_probes: std::cell::Cell::new(0),
            kick: std::cell::Cell::new(0),
            engine: false, n_nodes: 0, n_tris: 0, n_mats: 0, sky_base: 0, list_base: 0,
            dyn_nodes: std::cell::Cell::new(0), dyn_tris: std::cell::Cell::new(0),
            region: std::cell::RefCell::new(None),
            inner: None, inner_policy: InnerPolicy::Auto, inner_state: std::cell::RefCell::new(InnerState::default()), pending_new: std::cell::RefCell::new(Vec::new()), n_outer: 1,
            sun_pipeline, sun_bgl, sun_buf, sun_bg: None, sun_state: std::cell::RefCell::new(SunState::default()),
        }
    }

    pub fn probes_buf(&self) -> &wgpu::Buffer { &self.probes_buf }
    /// The ray-traced sun cascades the shading side reads (camera group binding 15).
    pub fn sun_buf(&self) -> &wgpu::Buffer { &self.sun_buf }
    pub fn vis_buf(&self) -> &wgpu::Buffer { &self.vis_buf }
    pub fn params_buf(&self) -> &wgpu::Buffer { &self.params_buf }
    pub fn has_scene(&self) -> bool { self.scene.is_some() }

    /// Choose the probe grid for a bounds box: uniform spacing so the count stays under `max_probes`
    /// (never below `min_spacing`). Returns (origin, spacing, dims).
    pub fn plan_grid(bmin: [f32; 3], bmax: [f32; 3], max_probes: u32, min_spacing: f32) -> ([f32; 3], f32, [u32; 3]) {
        let ext = [(bmax[0] - bmin[0]).max(1.0), (bmax[1] - bmin[1]).max(1.0), (bmax[2] - bmin[2]).max(1.0)];
        let vol = ext[0] as f64 * ext[1] as f64 * ext[2] as f64;
        let mut spacing = (vol / max_probes as f64).cbrt() as f32;
        spacing = spacing.max(min_spacing);
        // iterate: the +1 fence probes per axis can push the count over the cap on thin boxes
        for _ in 0..8 {
            let d = [(ext[0] / spacing).ceil() as u32 + 1, (ext[1] / spacing).ceil() as u32 + 1, (ext[2] / spacing).ceil() as u32 + 1];
            let n = d[0] as u64 * d[1] as u64 * d[2] as u64;
            if n <= max_probes as u64 { return (bmin, spacing, d); }
            spacing *= 1.08;
        }
        let d = [(ext[0] / spacing).ceil() as u32 + 1, (ext[1] / spacing).ceil() as u32 + 1, (ext[2] / spacing).ceil() as u32 + 1];
        (bmin, spacing, d)
    }

    /// Upload a scene, size the probe grid, reset the probe records. The caller must rebuild the
    /// camera bind group afterwards (the probe/vis buffers are new objects).
    pub fn set_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, s: &RtGiScene, env: Option<(&wgpu::TextureView, &wgpu::Sampler)>, env_tex: Option<(&wgpu::Texture, u32)>) {
        // Sky-cube magnitude: the capture went through the exposure-dependent sky pass, so its absolute
        // level is arbitrary. Read back the cube's 1x1 mip (6 face averages) and scale the cube so its
        // upper-hemisphere mean luminance equals the map's authored sky ambient (`sky_color`, the
        // lightmapper's sky irradiance DC) — the cube then only supplies the angular distribution.
        let engine = s.sky.as_ref().map_or(false, |k| !k.sky.is_empty() && k.sky.len() <= MAX_SKY_RECORDS);
        let mut cube_norm = 1.0f32;
        // (engine mode never reads the cube: the sky is the map's light list, so skip the GPU readback)
        if let Some((tex, mips)) = env_tex.filter(|_| !engine) {
            let mip = mips.saturating_sub(1);
            let rb = device.create_buffer(&wgpu::BufferDescriptor { label: Some("rtgi-cube-read"), size: 256 * 6, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &rb, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            );
            queue.submit(Some(enc.finish()));
            let slice = rb.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let mut faces = [[0f32; 3]; 6];
            for f in 0..6 {
                let b = &data[f * 256..f * 256 + 8];
                for c in 0..3 { faces[f][c] = f16_to_f32(u16::from_le_bytes([b[c * 2], b[c * 2 + 1]])); }
            }
            drop(data); rb.unmap();
            // faces: +x, -x, +y, -y, +z, -z; upper hemisphere ≈ (+z + ½·sides) / 3
            let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            let up = lum(faces[4]);
            let sides = (lum(faces[0]) + lum(faces[1]) + lum(faces[2]) + lum(faces[3])) * 0.25;
            let l_avg = (up + 0.5 * sides) / 1.5;
            let target = lum(s.sky_color);
            // The raw cube luminance already reproduces the shipped bake's sky term (Forge islet beach
            // 0.23 vs 0.27); only the HUE is off (the rendered sky is far bluer than the lightmapper's
            // sky colour), which the shader fixes by recolouring cube luminance with the authored hue.
            // HMS_RTGI_CUBE_NORM=1 switches to luminance normalisation for experiments.
            if std::env::var("HMS_RTGI_CUBE_NORM").is_ok() && l_avg > 1e-5 && target > 1e-5 { cube_norm = target / l_avg; }
            if std::env::var("HMS_DIAG").is_ok() || std::env::var("HMS_RTGI_DIAG").is_ok() {
                eprintln!("RTGI sky cube faces (lum): +x {:.3} -x {:.3} +y {:.3} -y {:.3} +z {:.3} -z {:.3} → upper mean {:.3}, authored sky {:.3}, cube gain {:.3}",
                    lum(faces[0]), lum(faces[1]), lum(faces[2]), lum(faces[3]), lum(faces[4]), lum(faces[5]), l_avg, target, cube_norm);
            }
        }
        let max_probes: u32 = std::env::var("HMS_RTGI_MAX_PROBES").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_PROBES).clamp(64, MAX_PROBES);
        // (the cap was 16384; measured on a Forge World patch, the sky-light error of the grid falls with its spacing:
        // p90 error 0.29 / 0.13 / 0.046 of the open-sky irradiance at 16.5 / 10 / 5 wu — see `forge_world_probe_spacing_vs_sky_occlusion_truth`)
        let min_spacing: f32 = std::env::var("HMS_RTGI_MIN_SPACING").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32).max(0.25);
        let (mut origin, mut spacing, mut dims) = Self::plan_grid(s.bounds_min, s.bounds_max, max_probes, min_spacing);
        // camera-centred inner grid on maps whose coarse grid is wide; its probes come out of the same cap
        let want_inner = match self.inner_policy {
            InnerPolicy::Off => None,
            InnerPolicy::Fixed(g) => Some(g),
            InnerPolicy::Auto => if spacing >= INNER_MIN_OUTER_SPACING { Some(INNER_DEFAULT) } else { None },
        };
        let mut inner: Option<InnerGrid> = None;
        if let Some(g) = want_inner {
            if g.count() + 512 < max_probes {
                let (o2, s2, d2) = Self::plan_grid(s.bounds_min, s.bounds_max, max_probes - g.count(), min_spacing);
                if s2 > 1.5 * g.spacing || matches!(self.inner_policy, InnerPolicy::Fixed(_)) { origin = o2; spacing = s2; dims = d2; inner = Some(g); }
            }
        }
        let n_outer = dims[0] * dims[1] * dims[2];
        let nprobes = n_outer + inner.map_or(0, |g| g.count());
        self.nprobes = nprobes.max(1);
        self.n_outer = n_outer;
        self.inner = inner;
        *self.inner_state.borrow_mut() = InnerState::default();
        self.pending_new.borrow_mut().clear();
        self.spacing = spacing;
        self.dims = dims;
        if let Some(k) = s.sky.as_ref() {
            if !engine && (std::env::var("HMS_DIAG").is_ok() || std::env::var("HMS_RTGI_DIAG").is_ok()) {
                eprintln!("RTGI: sky list unusable for the engine model ({} records, cap {MAX_SKY_RECORDS}); legacy sky", k.sky.len());
            }
        }
        // One storage buffer per kind with the DYNAMIC region reserved behind the static scene, so a
        // Forge edit rewrites bytes in place (no bind-group rebuild) and no binding is added (the
        // compute stage is already at the default limit of 8 storage buffers).
        let mk_cap = |label: &str, bytes: &[u8], extra: u64| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label), size: (bytes.len() as u64 + extra).max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: true,
            });
            { let mut v = b.slice(..).get_mapped_range_mut(); v[..bytes.len()].copy_from_slice(bytes); }
            b.unmap();
            b
        };
        let mk = |label: &str, bytes: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytes, usage: wgpu::BufferUsages::STORAGE,
        });
        let nodes = mk_cap("rtgi-nodes", &s.nodes, DYN_NODE_CAP as u64 * 32);
        let tris = mk_cap("rtgi-tris", &s.tris, DYN_TRI_CAP as u64 * 48);
        let mut tm = s.tri_mat.clone(); if tm.is_empty() { tm.push(0); }
        let n_tris_mat = tm.len() as u32;
        let tri_mat = mk_cap("rtgi-tri-mat", bytemuck::cast_slice(&tm), (DYN_TRI_CAP as u64 + MAX_PROBES as u64) * 4);
        let mut mats = s.mats.clone(); if mats.len() < 3 { mats = vec![[0.5, 0.5, 0.5, 0.0], [0.0; 4], [0.0; 4]]; }
        let n_mats = (mats.len() / 3) as u32;
        let sky_base = mats.len() as u32 + DYN_MAT_CAP * 3;
        let mut sky_region: Vec<[f32; 4]> = vec![[0.0; 4]; 2 + 2 * MAX_SKY_RECORDS];
        if engine {
            let k = s.sky.as_ref().unwrap();
            let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            let mut cum = 0.0f32;
            for (i, r) in k.sky.iter().enumerate() {
                cum += (lum(r.rgb) * r.omega).max(1e-9);
                sky_region[2 + 2 * i] = [r.dir[0], r.dir[1], r.dir[2], r.omega];
                sky_region[3 + 2 * i] = [r.rgb[0], r.rgb[1], r.rgb[2], cum];
            }
            sky_region[0] = [k.sky.len() as f32, k.sun_1st, k.sky_1st, cum];
        }
        let mats_b = mk_cap("rtgi-mats", bytemuck::cast_slice(&mats), (DYN_MAT_CAP as u64 * 3 + sky_region.len() as u64) * 16);
        queue.write_buffer(&mats_b, sky_base as u64 * 16, bytemuck::cast_slice(&sky_region));
        let mut lights = s.lights.clone(); if lights.is_empty() { lights.push(GpuLight::default()); }
        let lights_b = mk("rtgi-lights", bytemuck::cast_slice(&lights));
        let mut emitters = s.emitters.clone(); if emitters.is_empty() { emitters.push(GpuEmitter::default()); }
        let emit_b = mk("rtgi-emitters", bytemuck::cast_slice(&emitters));
        self.engine = engine;
        self.n_nodes = (s.nodes.len() / 32) as u32;
        self.n_tris = (s.tris.len() / 48) as u32;
        self.n_mats = n_mats;
        self.sky_base = sky_base;
        self.list_base = n_tris_mat + DYN_TRI_CAP;
        self.dyn_nodes.set(0);
        self.dyn_tris.set(0);
        *self.region.borrow_mut() = None;
        self.probes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-probes"),
            size: self.nprobes as u64 * SH_N as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-vis"),
            size: self.nprobes as u64 * VIS_TEXELS as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rtgi-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_mat.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mats_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: lights_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: emit_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.probes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.vis_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(env.map(|e| e.0).unwrap_or(&self.dummy_cube_view)) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(env.map(|e| e.1).unwrap_or(&self.dummy_samp)) },
            ],
        });
        let sun_scale: f32 = std::env::var("HMS_RTGI_SUN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let sky_scale: f32 = std::env::var("HMS_RTGI_SKY").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        // The 1.3 energy gain was fitted against the old sky model (captured sky cube, ~30 % light
        // missing); with the engine's own sky list there is nothing left to compensate, so engine mode
        // shades at 1.0 and the legacy fallback keeps its fitted 1.3.
        let energy: f32 = std::env::var("HMS_RTGI_GAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if engine { 1.0 } else { 1.3 });
        let ground: f32 = std::env::var("HMS_RTGI_GROUND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
        // sky radiance source: the captured sky env cube (the map's own sky model, HDR) unless disabled
        let use_cube = !engine && env.is_some() && std::env::var("HMS_RTGI_NO_CUBE").is_err();
        let cube_gain: f32 = std::env::var("HMS_RTGI_CUBE_GAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let nn = s.nodes.len() / 32;
        let nt = s.tris.len() / 48;
        self.params = GiParams {
            origin: [origin[0], origin[1], origin[2], spacing],
            dims: [dims[0], dims[1], dims[2], if self.enabled { 1 } else { 0 }],
            sun_dir: [s.sun_dir[0], s.sun_dir[1], s.sun_dir[2], if use_cube { 1.0 } else { 0.0 }],
            sun_color: [s.sun_color[0] * sun_scale, s.sun_color[1] * sun_scale, s.sun_color[2] * sun_scale, cube_gain * sky_scale * cube_norm],
            sky_color: [s.sky_color[0] * sky_scale, s.sky_color[1] * sky_scale, s.sky_color[2] * sky_scale, ground],
            misc: [0, self.rays_per_thread, nn as u32, nt as u32],
            misc2: [s.lights.len() as u32, s.emitters.len() as u32, self.nprobes, 0],
            // the Tool's lightmapper knows no fog: engine mode lights the probes fog-free (the fog is a
            // render-time effect applied to what the player sees)
            fog: [if engine { 0.0 } else { s.fog_sigma }, s.fog_top, s.total_emitter_weight, 1.0],
            bias: [0.2, 0.3, 1.5, energy],
            nee: [std::env::var("HMS_RTGI_NEE").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f32).clamp(0.0, 64.0),
                  std::env::var("HMS_RTGI_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0f32).max(0.1),
                  std::env::var("HMS_RTGI_NEE_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05f32).max(0.0), 0.0],
            dynp: [0xFFFF_FFFF, sky_base, self.list_base, 0],
            inner_org: match &inner { Some(i) => [origin[0], origin[1], origin[2], i.spacing], None => [0.0; 4] },
            inner_dims: match &inner { Some(i) => [i.dims[0], i.dims[1], i.dims[2], n_outer], None => [1, 1, 1, 0] },
            inner_base: [0; 4],
        };
        self.push_params(queue);
        // ray-traced sun cascades: a header (windows + global record) then 3 square maps of (first-hit distance,
        // translucent transmittance); inactive (header zero) until `update_sun` has traced a cascade
        let sun_texels = SUN_CASCADES as u64 * SUN_RES as u64 * SUN_RES as u64;
        self.sun_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-sun"), size: SUN_HDR_BYTES + sun_texels * 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        self.sun_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rtgi-sun-bg"),
            layout: &self.sun_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_mat.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mats_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.sun_buf.as_entire_binding() },
            ],
        }));
        *self.sun_state.borrow_mut() = SunState { tan_half: s.sky.as_ref().and_then(|k| k.sun).map_or(0.0, |sn| sun_tan_half(sn.omega)), dirty: [true; SUN_CASCADES], ..Default::default() };
        self.scene = Some(SceneBufs { nodes, tris, tri_mat, mats: mats_b, _lights: lights_b, _emitters: emit_b, bg });
        self.frame.set(0);
        self.updates.set(0);
        if std::env::var("HMS_DIAG").is_ok() || std::env::var("HMS_RTGI_DIAG").is_ok() {
            eprintln!("RTGI scene: {} nodes, {} tris, {} mats, {} lights, {} emitters; grid {}x{}x{} = {} probes, spacing {:.2} wu, origin ({:.1},{:.1},{:.1}); sun=({:.2},{:.2},{:.2}) sky=({:.2},{:.2},{:.2})",
                nn, nt, s.mats.len() / 3, s.lights.len(), s.emitters.len(), dims[0], dims[1], dims[2], self.nprobes, spacing, origin[0], origin[1], origin[2],
                self.params.sun_color[0], self.params.sun_color[1], self.params.sun_color[2], self.params.sky_color[0], self.params.sky_color[1], self.params.sky_color[2]);
        }
    }

    /// Switch the shading-side mode flag (the probe update keeps running only while enabled).
    pub fn set_enabled(&mut self, queue: &wgpu::Queue, on: bool) {
        self.enabled = on;
        self.params.dims[3] = if on && self.scene.is_some() { 1 } else { 0 };
        self.push_params(queue);
    }

    pub fn set_gain(&mut self, queue: &wgpu::Queue, gain: f32) {
        self.params.bias[3] = gain.clamp(0.01, 16.0);
        self.push_params(queue);
    }
    pub fn clear_scene(&mut self, queue: &wgpu::Queue) {
        self.scene = None;
        self.enabled = false;
        self.params.dims[3] = 0;
        self.push_params(queue);
    }
    /// Override the sun (direction to sun + HDR colour) for the tracer — the Lighting Lab sun controls.
    pub fn set_sun(&mut self, queue: &wgpu::Queue, dir: [f32; 3], color: [f32; 3]) {
        self.params.sun_dir = [dir[0], dir[1], dir[2], self.params.sun_dir[3]];
        self.params.sun_color = [color[0], color[1], color[2], self.params.sun_color[3]];
        self.push_params(queue);
        // The probe field still holds the previous sun and the temporal blend would take ~30 frames
        // to wash it out; kick every slice once at full weight so a dragged sun slider follows immediately.
        self.kick.set(self.parts_per_frame() + 1);
        self.mark_sun_dirty();
    }
    /// Force the next slices to REPLACE rather than blend (after any lighting change).
    pub fn kick(&self) { self.kick.set(self.parts_per_frame() + 1); }

    /// Re-trace the sun cascades the next frames (the sun moved, or an occluder did).
    pub fn mark_sun_dirty(&self) { self.sun_state.borrow_mut().dirty = [true; SUN_CASCADES]; }

    /// Keep the ray-traced sun shadow cascades around the camera: ONE cascade per call is re-traced when it is
    /// stale (never traced, an occluder / the sun changed) or when the camera has left the middle of its window.
    /// Each cascade is an orthographic view from the sun (`sun_basis`) of the first-hit distance through the
    /// probe tracer's BVH, so the static map AND the placed Forge pieces cast. Recorded into `encoder` ahead of
    /// the passes that read it (`gi_sun_vis` in the shading shaders).
    pub fn update_sun(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, cam: [f32; 3]) {
        let Some(bg) = self.sun_bg.as_ref() else { return };
        if self.scene.is_none() || !self.enabled { return; }
        let sd = self.params.sun_dir;
        let n = (sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]).sqrt();
        if n < 0.5 { return; }
        let l = [sd[0] / n, sd[1] / n, sd[2] / n];
        let (bx, by) = sun_basis(l);
        let (cu, cv) = (cam[0] * bx[0] + cam[1] * bx[1] + cam[2] * bx[2], cam[0] * by[0] + cam[1] * by[1] + cam[2] * by[2]);
        let mut st = self.sun_state.borrow_mut();
        let mut pick: Option<(usize, [f32; 2])> = None;
        for c in 0..SUN_CASCADES {
            let texel = 2.0 * SUN_HALF[c] / SUN_RES as f32;
            let want = [(cu / texel).round() * texel, (cv / texel).round() * texel];
            let moved = (want[0] - st.centers[c][0]).abs().max((want[1] - st.centers[c][1]).abs()) > 0.25 * SUN_HALF[c];
            if !st.valid[c] || st.dirty[c] || moved { pick = Some((c, want)); break; }
        }
        let Some((c, want)) = pick else { return };
        st.centers[c] = want;
        let mut hdr = [[0f32; 4]; SUN_CASCADES + 1];
        // z: 0 = never traced (the shading side skips it), 1 = valid, 2 = being re-traced by this dispatch
        for k in 0..SUN_CASCADES { hdr[k] = [st.centers[k][0], st.centers[k][1], if k == c { 2.0 } else if st.valid[k] { 1.0 } else { 0.0 }, SUN_HALF[k]]; }
        hdr[SUN_CASCADES] = [1.0, st.tan_half, SUN_RES as f32, SUN_ORIGIN_DIST];
        queue.write_buffer(&self.sun_buf, 0, bytemuck::cast_slice(&hdr));
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rtgi-sun-shadow"), timestamp_writes: None });
            cp.set_pipeline(&self.sun_pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(SUN_RES / 8, SUN_RES / 8, SUN_CASCADES as u32);
        }
        st.valid[c] = true;
        st.dirty[c] = false;
        st.active = true;
    }

    /// Run the shading shaders' own `gi_sun_vis(p, n)` (the very WGSL text the mesh / terrain shaders compile) for a
    /// list of (position, normal) against the cascades now held; returns sqrt(visible fraction) per point. A blocking
    /// measurement (gates, `HMS_RTGI_COMPARE`), never called per frame.
    pub fn sample_sun_vis(&self, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3])], objects: bool) -> Vec<f32> {
        let src = format!(
            "{}{}{}",
            GI_FRAGMENT_WGSL,
            GI_SHARED_WGSL.replace("__GI_DEBUG__", "0"),
            r#"
    @group(0) @binding(20) var<storage, read> t_in: array<vec4<f32>>;
    @group(0) @binding(21) var<storage, read_write> t_out: array<f32>;
    @compute @workgroup_size(1)
    fn t_main(@builtin(global_invocation_id) g: vec3<u32>) {
        t_out[g.x] = gi_sun_vis(t_in[2u * g.x].xyz, t_in[2u * g.x + 1u].xyz, __OBJ__);
    }
    "#
        )
        .replace("__OBJ__", if objects { "true" } else { "false" });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sunvis-test"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let ent = |b: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                ent(11, wgpu::BufferBindingType::Uniform),
                ent(15, wgpu::BufferBindingType::Storage { read_only: true }),
                ent(20, wgpu::BufferBindingType::Storage { read_only: true }),
                ent(21, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: Some("t_main"), compilation_options: Default::default(), cache: None });
        let mut input: Vec<[f32; 4]> = Vec::new();
        for (p, n) in pts { input.push([p[0], p[1], p[2], 0.0]); input.push([n[0], n[1], n[2], 0.0]); }
        let inb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
        let outb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: pts.len() as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: pts.len() as u64 * 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 11, resource: self.params_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.sun_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: inb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 21, resource: outb.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(pts.len() as u32, 1, 1); }
        enc.copy_buffer_to_buffer(&outb, 0, &rb, 0, pts.len() as u64 * 4);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let v: Vec<f32> = bytemuck::cast_slice(&sl.get_mapped_range()).to_vec();
        v
    }

    /// Run the shading shaders' own `gi_irradiance(p, n, v)` (the WGSL text the mesh / terrain shaders compile) for a list of
    /// (position, normal) against the probes now held; the viewer vector is the normal. Returns irradiance/pi RGB per
    /// point. A blocking measurement for gates, never called per frame.
    pub fn sample_irradiance(&self, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3])]) -> Vec<[f32; 3]> {
        self.sample_irradiance_variant(device, queue, pts, 0, &[])
    }

    /// `sample_irradiance` for probe SH slot `slot` (0 = total, 10 = gather only, 19 = sky-direct only) with the
    /// shader text edited by `edits` (exact substring replacements: the test seam that turns one term of
    /// `gi_irradiance_set` off to measure how much of the result it carries). Panics when an edit does not apply.
    pub fn sample_irradiance_variant(&self, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3])], slot: u32, edits: &[(&str, &str)]) -> Vec<[f32; 3]> {
        let mut shared = GI_SHARED_WGSL.replace("__GI_DEBUG__", "0");
        for (a, b) in edits { assert!(shared.contains(a), "shader edit does not apply: {a}"); shared = shared.replace(a, b); }
        let src = format!(
            "{}{}{}",
            GI_FRAGMENT_WGSL,
            shared,
            r#"
    @group(0) @binding(20) var<storage, read> t_in: array<vec4<f32>>;
    @group(0) @binding(21) var<storage, read_write> t_out: array<vec4<f32>>;
    @compute @workgroup_size(1)
    fn t_main(@builtin(global_invocation_id) g: vec3<u32>) {
        let n = t_in[2u * g.x + 1u].xyz;
        t_out[g.x] = vec4<f32>(gi_irradiance_set(t_in[2u * g.x].xyz, n, n, __SLOT__), 0.0);
    }
    "#
        ).replace("__SLOT__", &format!("{slot}u"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("irr-test"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let ent = |b: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ent(10, ro), ent(11, wgpu::BufferBindingType::Uniform), ent(12, ro), ent(20, ro), ent(21, wgpu::BufferBindingType::Storage { read_only: false })],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: Some("t_main"), compilation_options: Default::default(), cache: None });
        let mut input: Vec<[f32; 4]> = Vec::new();
        for (p, n) in pts { input.push([p[0], p[1], p[2], 0.0]); input.push([n[0], n[1], n[2], 0.0]); }
        let inb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
        let bytes = pts.len() as u64 * 16;
        let outb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 10, resource: self.probes_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.params_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.vis_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: inb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 21, resource: outb.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(pts.len() as u32, 1, 1); }
        enc.copy_buffer_to_buffer(&outb, 0, &rb, 0, bytes);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let v: Vec<[f32; 4]> = bytemuck::cast_slice(&sl.get_mapped_range()).to_vec();
        v.iter().map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// Run `gi_light_vis` (the WGSL the pixel shaders use to hide a baked light behind walls) for (point, normal, light position)
    /// triples against the probes now held. A blocking measurement for gates.
    pub fn sample_light_vis(&self, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3], [f32; 3])]) -> Vec<f32> {
        let src = format!(
            "{}{}{}",
            GI_FRAGMENT_WGSL,
            GI_SHARED_WGSL.replace("__GI_DEBUG__", "0"),
            r#"
    @group(0) @binding(20) var<storage, read> t_in: array<vec4<f32>>;
    @group(0) @binding(21) var<storage, read_write> t_out: array<f32>;
    @compute @workgroup_size(1)
    fn t_main(@builtin(global_invocation_id) g: vec3<u32>) {
        t_out[g.x] = gi_light_vis(t_in[3u * g.x].xyz, t_in[3u * g.x + 1u].xyz, t_in[3u * g.x + 2u].xyz);
    }
    "#
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("lightvis-test"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let ent = |b: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[ent(10, ro), ent(11, wgpu::BufferBindingType::Uniform), ent(12, ro), ent(20, ro), ent(21, wgpu::BufferBindingType::Storage { read_only: false })] });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: Some("t_main"), compilation_options: Default::default(), cache: None });
        let mut input: Vec<[f32; 4]> = Vec::new();
        for (p, n, l) in pts { input.push([p[0], p[1], p[2], 0.0]); input.push([n[0], n[1], n[2], 0.0]); input.push([l[0], l[1], l[2], 0.0]); }
        let inb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
        let bytes = pts.len() as u64 * 4;
        let outb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 10, resource: self.probes_buf().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: self.params_buf().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: self.vis_buf().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 20, resource: inb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 21, resource: outb.as_entire_binding() },
        ] });
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(pts.len() as u32, 1, 1); }
        enc.copy_buffer_to_buffer(&outb, 0, &rb, 0, bytes);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let v: Vec<f32> = bytemuck::cast_slice(&sl.get_mapped_range()).to_vec();
        v
    }

    /// GROUND TRUTH for the shadow maps: the visible fraction of the sun disc from each (position, normal) by 32 real
    /// rays through the SAME BVH (static + placed pieces), no cascades, no filter. Returns the fraction (not its
    /// square root). A blocking measurement for gates and diagnostics.
    pub fn trace_sun_fraction(&self, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3])]) -> Vec<f32> {
        let Some(sc) = self.scene.as_ref() else { return Vec::new() };
        let tan_h = self.sun_state.borrow().tan_half;
        let src = format!("{}{}", compute_source(0), r#"
@group(0) @binding(20) var<storage, read> t_in: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read_write> t_out: array<f32>;
@compute @workgroup_size(1)
fn tv_main(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = t_in[2u * g.x].xyz;
    let th = t_in[2u * g.x].w;
    let n = t_in[2u * g.x + 1u].xyz;
    let l = normalize(gi.sun_dir.xyz);
    let bx = gi_sun_basis_x(l);
    let by = cross(l, bx);
    var acc = 0.0;
    for (var k = 0u; k < 32u; k = k + 1u) {
        let r = sqrt((f32(k) + 0.5) / 32.0) * th;
        let a = f32(k) * 2.39996323;
        let d = normalize(l + (bx * cos(a) + by * sin(a)) * r);
        let v = visibility(p + n * 0.05, d, T_MAX);
        acc = acc + (v.x + v.y + v.z) / 3.0;
    }
    t_out[g.x] = acc / 32.0;
}
"#);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sun-truth"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let ent = |b: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
            ent(0, ro), ent(1, ro), ent(2, ro), ent(3, ro), ent(6, wgpu::BufferBindingType::Uniform), ent(20, ro), ent(21, wgpu::BufferBindingType::Storage { read_only: false }),
        ] });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &module, entry_point: Some("tv_main"), compilation_options: Default::default(), cache: None });
        let mut input: Vec<[f32; 4]> = Vec::new();
        for (p, n) in pts { input.push([p[0], p[1], p[2], tan_h]); input.push([n[0], n[1], n[2], 0.0]); }
        let inb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
        let outb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: pts.len() as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: pts.len() as u64 * 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sc.nodes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: sc.tris.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: sc.tri_mat.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sc.mats.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: self.params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 20, resource: inb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 21, resource: outb.as_entire_binding() },
        ] });
        let mut enc = device.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default()); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(pts.len() as u32, 1, 1); }
        enc.copy_buffer_to_buffer(&outb, 0, &rb, 0, pts.len() as u64 * 4);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let v: Vec<f32> = bytemuck::cast_slice(&sl.get_mapped_range()).to_vec();
        v
    }

    /// Trace every stale cascade now (headless runs and tests): loops `update_sun` until nothing is pending.
    pub fn update_sun_now(&self, device: &wgpu::Device, queue: &wgpu::Queue, cam: [f32; 3]) {
        for _ in 0..(2 * SUN_CASCADES) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rtgi-sun-now") });
            self.update_sun(&mut enc, queue, cam);
            queue.submit(Some(enc.finish()));
        }
        let _ = device.poll(wgpu::Maintain::Wait);
    }

    /// Record one probe-update pass into `encoder`. `blend` overrides the temporal alpha (None =
    /// 1 − hysteresis, or 1.0 on the very first update so the records start from a full estimate).
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, blend: Option<f32>) {
        self.dispatch_part(encoder, queue, blend, 1, 0);
    }
    /// Update slice `part` of `parts` (round-robin partial updates keep the per-frame cost bounded).
    /// A pending edit region (see `invalidate_region`) takes the frame's dispatch first.
    pub fn dispatch_part(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, blend: Option<f32>, parts: u32, part: u32) {
        let Some(sc) = self.scene.as_ref() else { return };
        // probes whose cell the inner window just handed over: traced from scratch, ahead of everything (4096 a frame)
        {
            let mut pend = self.pending_new.borrow_mut();
            if !pend.is_empty() {
                let take = pend.len().min(4096);
                let list: Vec<u32> = pend.drain(..take).collect();
                drop(pend);
                self.dispatch_list(encoder, queue, sc, &list, 1.0, 1 | 2 | 4);
                return;
            }
        }
        if blend.is_none() {
            if let Some((list, done)) = self.next_region_pass() {
                self.dispatch_list(encoder, queue, sc, &list, 2.0 / (done as f32 + 2.0), 3);
                return;
            }
        }
        let f = self.frame.get();
        let kicking = self.kick.get() > 0;
        let alpha = match blend { Some(a) => a, None => if kicking { 1.0 } else { 1.0 - self.hysteresis } };
        if kicking { self.kick.set(self.kick.get() - 1); }
        let parts = parts.max(1);
        let slice = (self.nprobes + parts - 1) / parts;
        let base = (part % parts) * slice;
        if base >= self.nprobes { return; }
        let count = slice.min(self.nprobes - base);
        let mut p = self.params;
        p.misc[0] = f;
        p.misc[1] = self.rays_per_thread;
        p.misc2[3] = base;
        p.fog[3] = alpha;
        p.dynp[3] = 0;
        p.inner_base = self.inner_base_now();
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rtgi-probe-update"), timestamp_writes: None });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &sc.bg, &[]);
            cp.dispatch_workgroups(count, 1, 1);
        }
        self.frame.set(f.wrapping_add(1));
        self.updates.set(self.updates.get() + 1);
        self.last_dispatch_probes.set(count);
    }

    /// One dispatch over an explicit probe list (an edit region): the list rides in the `tri_mat` buffer
    /// behind the dynamic region, `dynp.w` bit0 makes the shader read it, bit1 re-traces the sky-direct
    /// SH of those probes (an occluder moved: their sky visibility changed).
    fn dispatch_list(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, sc: &SceneBufs, list: &[u32], alpha: f32, flags: u32) {
        if list.is_empty() { return; }
        let f = self.frame.get();
        queue.write_buffer(&sc.tri_mat, self.list_base as u64 * 4, bytemuck::cast_slice(list));
        let mut p = self.params;
        p.misc[0] = f;
        p.misc[1] = self.rays_per_thread;
        p.misc2[3] = 0;
        p.fog[3] = alpha;
        p.dynp[3] = flags;
        p.inner_base = self.inner_base_now();
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rtgi-region-update"), timestamp_writes: None });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &sc.bg, &[]);
            cp.dispatch_workgroups(list.len() as u32, 1, 1);
        }
        self.frame.set(f.wrapping_add(1));
        self.updates.set(self.updates.get() + 1);
        self.last_dispatch_probes.set(list.len() as u32);
    }

    fn push_params(&self, queue: &wgpu::Queue) {
        let mut p = self.params;
        p.inner_base = self.inner_base_now();
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }
    fn inner_base_now(&self) -> [i32; 4] { let b = self.inner_state.borrow().base; [b[0], b[1], b[2], 0] }

    /// The integer cell an inner slot (0-based within the inner grid) holds for a window whose minimum corner is `base`.
    pub fn inner_cell_of(g: &InnerGrid, base: [i32; 3], slot: u32) -> [i32; 3] {
        let d = [g.dims[0] as i32, g.dims[1] as i32, g.dims[2] as i32];
        let r = slot as i32;
        let sc = [r % d[0], (r / d[0]) % d[1], r / (d[0] * d[1])];
        let mut c = [0i32; 3];
        for a in 0..3 { c[a] = base[a] + (sc[a] - base[a]).rem_euclid(d[a]); }
        c
    }
    /// The inner slot (0-based) that holds integer cell `c`.
    pub fn inner_slot_of(g: &InnerGrid, c: [i32; 3]) -> u32 {
        let d = [g.dims[0] as i32, g.dims[1] as i32, g.dims[2] as i32];
        let m = [c[0].rem_euclid(d[0]), c[1].rem_euclid(d[1]), c[2].rem_euclid(d[2])];
        (m[0] + m[1] * d[0] + m[2] * d[0] * d[1]) as u32
    }
    /// Where the inner window's minimum cell sits now (diagnostics / gates).
    #[cfg(test)]
    pub(crate) fn default_probe_base_for_test() -> [i32; 3] { [0, 0, 0] }
    pub fn inner_window(&self) -> Option<(InnerGrid, [i32; 3], [f32; 3])> {
        let g = self.inner?;
        Some((g, self.inner_state.borrow().base, [self.params.origin[0], self.params.origin[1], self.params.origin[2]]))
    }

    /// Centre the inner window on `cam` (world position). The window moves in whole cells and only once the camera is two
    /// cells off its centre (hysteresis, so a camera hovering on a cell border does not shuffle probes); the cells that
    /// change owner are queued for a from-scratch trace. Returns how many probes were queued. Cheap when nothing moves.
    pub fn focus(&self, cam: [f32; 3]) -> usize {
        let Some(g) = self.inner else { return 0 };
        let o = self.params.origin;
        let mut st = self.inner_state.borrow_mut();
        // vertical: two thirds of the window above the camera's floor-ish position is wasted sky; centre it slightly low
        let want: [i32; 3] = std::array::from_fn(|a| {
            let c = ((cam[a] - o[a]) / g.spacing).floor() as i32;
            c - (g.dims[a] as i32) / 2
        });
        let moved = if !st.init { true } else { (0..3).any(|a| (want[a] - st.base[a]).abs() >= 2) };
        if !moved { return 0; }
        let old = st.base;
        let first = !st.init;
        st.base = want;
        st.init = true;
        let mut list: Vec<u32> = Vec::new();
        for slot in 0..g.count() {
            if first || Self::inner_cell_of(&g, old, slot) != Self::inner_cell_of(&g, want, slot) { list.push(self.n_outer + slot); }
        }
        let n = list.len();
        self.pending_new.borrow_mut().extend(list);
        n
    }

    /// The pending edit region's probe list and the number of passes already done; advances the count.
    fn next_region_pass(&self) -> Option<(Vec<u32>, u32)> {
        let mut r = self.region.borrow_mut();
        let (list, done) = r.as_mut()?;
        let out = (list.clone(), *done);
        *done += 1;
        if *done >= REGION_PASSES { *r = None; }
        Some(out)
    }

    /// Probe indices whose NOMINAL grid position lies inside `[min, max]` grown by `pad` (a probe may
    /// relocate by up to 0.6 spacing, which `pad` should cover). Pure; the unit tests pin it.
    pub fn probes_in_box(origin: [f32; 3], spacing: f32, dims: [u32; 3], min: [f32; 3], max: [f32; 3], pad: f32) -> Vec<u32> {
        let mut rng = [(0u32, 0u32); 3];
        for a in 0..3 {
            let lo = ((min[a] - pad - origin[a]) / spacing).ceil().max(0.0);
            let hi = ((max[a] + pad - origin[a]) / spacing).floor().min(dims[a] as f32 - 1.0);
            if hi < lo { return Vec::new(); }
            rng[a] = (lo as u32, hi as u32);
        }
        let mut out = Vec::new();
        for z in rng[2].0..=rng[2].1 {
            for y in rng[1].0..=rng[1].1 {
                for x in rng[0].0..=rng[0].1 {
                    out.push(x + y * dims[0] + z * dims[0] * dims[1]);
                }
            }
        }
        out
    }

    /// An occluder changed inside `[min, max]` (world units): queue every probe whose lighting it can
    /// change for a four-pass full-quality re-estimate (sky visibility, bounce, shadowed bounce). The
    /// reach is the box grown by two probe spacings: beyond that a one-piece edit changes a probe's
    /// radiance by less than the tracer's own noise. Returns the number of probes queued.
    pub fn invalidate_region(&self, min: [f32; 3], max: [f32; 3]) -> usize {
        if self.scene.is_none() { return 0; }
        let pad = 2.0 * self.spacing;
        let mut add = Self::probes_in_box(
            [self.params.origin[0], self.params.origin[1], self.params.origin[2]], self.spacing, self.dims, min, max, pad);
        if let Some(g) = self.inner {
            let (base, o) = (self.inner_state.borrow().base, self.params.origin);
            let ipad = 2.0 * g.spacing;
            let mut lo = [0i32; 3]; let mut hi = [0i32; 3];
            for a in 0..3 {
                lo[a] = (((min[a] - ipad - o[a]) / g.spacing).ceil() as i32).max(base[a]);
                hi[a] = (((max[a] + ipad - o[a]) / g.spacing).floor() as i32).min(base[a] + g.dims[a] as i32 - 1);
            }
            for z in lo[2]..=hi[2] { for y in lo[1]..=hi[1] { for x in lo[0]..=hi[0] { add.push(self.n_outer + Self::inner_slot_of(&g, [x, y, z])); } } }
        }
        if add.is_empty() { return 0; }
        let mut r = self.region.borrow_mut();
        let mut list = r.take().map(|(l, _)| l).unwrap_or_default();
        list.extend(add);
        list.sort_unstable();
        list.dedup();
        let n = list.len();
        *r = Some((list, 0));
        n
    }

    /// Replace the dynamic occluders (placed Forge pieces). `None` clears them. Returns false when the
    /// set exceeds the reserved region (the previous set stays live). The caller then calls
    /// `invalidate_region` over the old and new bounds of whatever moved.
    pub fn set_dynamic(&mut self, queue: &wgpu::Queue, d: Option<&RtGiDynamic>) -> bool {
        let Some(sc) = self.scene.as_ref() else { return false };
        let (nn, nt, nm) = d.map_or((0, 0, 0), |d| (d.nodes.len() / 32, d.tris.len() / 48, d.mats.len() / 3));
        if nn as u32 > DYN_NODE_CAP || nt as u32 > DYN_TRI_CAP || nm as u32 > DYN_MAT_CAP { return false; }
        if let Some(d) = d {
            if nn > 0 && nt > 0 {
                // BVH words: 8 per node = aabb_min[3], left_first, aabb_max[3], count. Rebase the local
                // indices behind the static scene: inner nodes point at nodes, leaves at triangles.
                let mut w: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&d.nodes).to_vec();
                for n in 0..nn {
                    let b = n * 8;
                    w[b + 3] += if w[b + 7] > 0 { self.n_tris } else { self.n_nodes };
                }
                queue.write_buffer(&sc.nodes, self.n_nodes as u64 * 32, bytemuck::cast_slice(&w));
                queue.write_buffer(&sc.tris, self.n_tris as u64 * 48, &d.tris);
                let tm: Vec<u32> = d.tri_mat.iter().map(|m| m + self.n_mats).collect();
                queue.write_buffer(&sc.tri_mat, self.n_tris as u64 * 4, bytemuck::cast_slice(&tm));
                queue.write_buffer(&sc.mats, self.n_mats as u64 * 3 * 16, bytemuck::cast_slice(&d.mats));
            }
        }
        self.mark_sun_dirty();
        let live = nn > 0 && nt > 0;
        self.dyn_nodes.set(if live { nn as u32 } else { 0 });
        self.dyn_tris.set(if live { nt as u32 } else { 0 });
        self.params.dynp[0] = if live { self.n_nodes } else { 0xFFFF_FFFF };
        self.push_params(queue);
        true
    }
    pub fn dynamic_tris(&self) -> u32 { self.dyn_tris.get() }

    /// Read every probe record back (tests and the engine-record export of the lightmap baker; a
    /// blocking readback, never called per frame).
    pub fn read_probes(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ProbeRecord> {
        if self.scene.is_none() { return Vec::new(); }
        let pb = self.nprobes as u64 * SH_N as u64 * 16;
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: Some("rtgi-read-probes"), size: pb, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.probes_buf, 0, &rb, 0, pb);
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let v: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let grab = |base: usize, off: u32| -> [[f32; 3]; 9] {
            let mut o = [[0.0f32; 3]; 9];
            for k in 0..9 { let r = v[base + off as usize + k]; o[k] = [r[0], r[1], r[2]]; }
            o
        };
        let out = (0..self.nprobes as usize).map(|i| {
            let b = i * SH_N as usize;
            let rl = v[b + SLOT_RELOC as usize];
            ProbeRecord { total: grab(b, SLOT_TOTAL), gather: grab(b, SLOT_GATHER), sky: grab(b, SLOT_SKY), offset: [rl[0], rl[1], rl[2]], valid: rl[3] > 0.5 }
        }).collect();
        drop(data);
        rb.unmap();
        out
    }
    /// Probe slices per frame for the interactive update (1 = whole field every frame).
    pub fn parts_per_frame(&self) -> u32 {
        let target: u32 = std::env::var("HMS_RTGI_PROBES_PER_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        ((self.nprobes + target - 1) / target.max(64)).max(1)
    }

    /// Diagnostic: read back the 8 probe records (SH, offset/valid, vis mean/max) around world point `p`.
    pub fn dump_at(&self, device: &wgpu::Device, queue: &wgpu::Queue, p: [f32; 3]) -> Vec<String> {
        let mut out = Vec::new();
        if self.scene.is_none() { out.push("RTGI dump: no scene".into()); return out; }
        let sp = self.params.origin[3];
        let o = self.params.origin;
        let d = self.dims;
        let g = [(p[0] - o[0]) / sp, (p[1] - o[1]) / sp, (p[2] - o[2]) / sp];
        let base = [g[0].floor().clamp(0.0, (d[0] as f32 - 2.0).max(0.0)) as u32, g[1].floor().clamp(0.0, (d[1] as f32 - 2.0).max(0.0)) as u32, g[2].floor().clamp(0.0, (d[2] as f32 - 2.0).max(0.0)) as u32];
        out.push(format!("RTGI dump at ({:.1},{:.1},{:.1}): grid coord ({:.2},{:.2},{:.2}) base ({},{},{}) spacing {:.2} dims {:?}", p[0], p[1], p[2], g[0], g[1], g[2], base[0], base[1], base[2], sp, d));
        // read back the whole probe + vis buffers (diagnostic only)
        let pb = self.nprobes as u64 * SH_N as u64 * 16;
        let vb = self.nprobes as u64 * VIS_TEXELS as u64 * 8;
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: Some("rtgi-read"), size: pb + vb, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.probes_buf, 0, &rb, 0, pb);
        enc.copy_buffer_to_buffer(&self.vis_buf, 0, &rb, pb, vb);
        queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let probes: &[[f32; 4]] = bytemuck::cast_slice(&data[..pb as usize]);
        let vis: &[[f32; 2]] = bytemuck::cast_slice(&data[pb as usize..]);
        for k in 0..8u32 {
            let c = [base[0] + (k & 1), base[1] + ((k >> 1) & 1), base[2] + ((k >> 2) & 1)];
            if c[0] >= d[0] || c[1] >= d[1] || c[2] >= d[2] { continue; }
            let pi = (c[0] + c[1] * d[0] + c[2] * d[0] * d[1]) as usize;
            let r = &probes[pi * SH_N as usize..(pi + 1) * SH_N as usize];
            let v = &vis[pi * VIS_TEXELS as usize..(pi + 1) * VIS_TEXELS as usize];
            let vmean: f32 = v.iter().map(|t| t[0]).sum::<f32>() / VIS_TEXELS as f32;
            let vmax = v.iter().map(|t| t[0]).fold(0.0f32, f32::max);
            let e_up = 3.14159 * 0.282095 * r[0][1] + 2.0944 * 0.488603 * r[2][1];
            out.push(format!("  probe {} at ({:.1},{:.1},{:.1}) off=({:.2},{:.2},{:.2}) valid={} c0=({:.4},{:.4},{:.4}) c2(z)=({:.4},{:.4},{:.4}) E_up(g)={:.4} vis mean={:.2} max={:.2}",
                pi, o[0] + c[0] as f32 * sp + r[9][0], o[1] + c[1] as f32 * sp + r[9][1], o[2] + c[2] as f32 * sp + r[9][2], r[9][0], r[9][1], r[9][2], r[9][3],
                r[0][0], r[0][1], r[0][2], r[2][0], r[2][1], r[2][2], e_up, vmean, vmax));
        }
        drop(data);
        rb.unmap();
        out
    }

    /// Run `n` update passes back to back (headless warm-up / after a scene change) and wait.
    pub fn warm_up(&self, device: &wgpu::Device, queue: &wgpu::Queue, n: u32) {
        for i in 0..n {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rtgi-warmup") });
            // fresh probes take the full estimate anyway; then a fast blend that converges in ~n passes
            let a = (2.0 / (i as f32 + 2.0)).max(1.0 - self.hysteresis);
            self.dispatch(&mut enc, queue, Some(a));
            queue.submit(Some(enc.finish()));
        }
        let _ = device.poll(wgpu::Maintain::Wait);
    }
}

/// The probe-update compute module exactly as the pipeline compiles it (`cdebug` = the HMS_RTGI_CDEBUG lane).
pub(crate) fn compute_source(cdebug: i32) -> String {
    format!("{COMPUTE_DECLS_WGSL}{GI_SHARED_WGSL}{COMPUTE_WGSL}").replace("__GI_DEBUG__", "0").replace("__GI_CDEBUG__", &cdebug.to_string())
}

/// IEEE half → f32 (for the tiny env-cube readback; no half crate dependency).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 { sign << 31 } else {
            // subnormal
            let mut e = -1i32; let mut f = frac;
            while f & 0x400 == 0 { f <<= 1; e -= 1; }
            let f = f & 0x3ff;
            (sign << 31) | (((e + 1 + 127 - 15) as u32) << 23) | (f << 13)
        }
    } else if exp == 31 { (sign << 31) | 0x7f80_0000 | (frac << 13) }
    else { (sign << 31) | (((exp + 127 - 15) as u32) << 23) | (frac << 13) };
    f32::from_bits(bits)
}

/// Camera-group declarations + the shared lookup, prepended to every shading shader. Bindings 10–12
/// of the camera bind group.
pub const GI_FRAGMENT_WGSL: &str = r#"
struct GiParams {
    origin: vec4<f32>, dims: vec4<u32>, sun_dir: vec4<f32>, sun_color: vec4<f32>, sky_color: vec4<f32>,
    misc: vec4<u32>, misc2: vec4<u32>, fog: vec4<f32>, bias: vec4<f32>, nee: vec4<f32>, dynp: vec4<u32>,
    inner_org: vec4<f32>, inner_dims: vec4<u32>, inner_base: vec4<i32>,
};
@group(0) @binding(10) var<storage, read> gi_probes: array<vec4<f32>>;
@group(0) @binding(11) var<uniform> gi: GiParams;
@group(0) @binding(12) var<storage, read> gi_vis: array<vec2<f32>>;
// Ray-traced sun shadow cascades (rtgi.rs `update_sun`): hdr[k] = (window centre u, v in the light plane, _, half
// extent; z = 0 never traced / 1 valid / 2 being re-traced) for k < 3, hdr[3] = (active, tan(sun half angle), resolution, ray-origin distance); tex = (distance
// from the ray origin to the first opaque hit in the STATIC map, the same in the placed pieces, transmittance of
// translucent surfaces in front of the nearer one, 0).
struct GiSunBuf { hdr: array<vec4<f32>, 5>, tex: array<vec4<f32>> };
@group(0) @binding(15) var<storage, read> gi_sun: GiSunBuf;

fn gi_sun_vogel(k: u32, n: u32, rot: f32) -> vec2<f32> {
    let r = sqrt((f32(k) + 0.5) / f32(n));
    let a = f32(k) * 2.39996323 + rot;
    return vec2<f32>(cos(a), sin(a)) * r;
}
fn gi_sun_tex(c: u32, x: i32, y: i32, res: u32) -> vec4<f32> {
    let xi = u32(clamp(x, 0, i32(res) - 1));
    let yi = u32(clamp(y, 0, i32(res) - 1));
    return gi_sun.tex[c * res * res + yi * res + xi];
}
// The sun's visibility at a surface point, as the engine stores it: the engine multiplies the analytic sun by
// DM.g TWICE, so this returns sqrt(visible fraction) and the existing `vis * vis` gates stay as they are.
// The fraction is a percentage-closer filter over the sun DISC: a blocker search over the occluder-to-receiver
// gap, then a filter of radius tan(half angle) x gap (the Tool's sun cone, `acos(1 - Omega/2pi)`), so contact
// shadows are sharp and far ones soft. Every filter tap is a BILINEAR comparison (the four texels' lit/shadowed
// results weighted by the sub-texel position) and the radius never drops below 2 texels (a binary staircase of edge samples needs ~2 texels of blur to read as a straight line), so a shadow edge is
// smooth at any distance instead of stepping texel by texel. Near the edge of a cascade's window the next, coarser
// cascade is blended in so the handoff does not pop. 1.0 when the maps are not active or the point is outside them.
// `objects` = the receiver is a placed piece / dynamic object: it sees the STATIC map only (its own coarse physics
// hull is in the placed-piece layer and would shadow its own surface; pieces shadow each other through the raster
// shadow pass), level receivers see static + placed pieces.
fn gi_sun_t(c: u32, x: i32, y: i32, res: u32, objects: bool) -> f32 {
    let v = gi_sun_tex(c, x, y, res);
    return select(min(v.x, v.y), v.x, objects);
}
// bilinear lit fraction at continuous texel coordinates (x, y) against receiver distance `lim`
fn gi_sun_lit_bilinear(c: u32, x: f32, y: f32, res: u32, objects: bool, lim: f32) -> f32 {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let ax = fx - floor(fx);
    let ay = fy - floor(fy);
    let l00 = select(0.0, 1.0, gi_sun_t(c, x0, y0, res, objects) >= lim);
    let l10 = select(0.0, 1.0, gi_sun_t(c, x0 + 1, y0, res, objects) >= lim);
    let l01 = select(0.0, 1.0, gi_sun_t(c, x0, y0 + 1, res, objects) >= lim);
    let l11 = select(0.0, 1.0, gi_sun_t(c, x0 + 1, y0 + 1, res, objects) >= lim);
    return mix(mix(l00, l10, ax), mix(l01, l11, ax), ay);
}
// visible fraction through cascade `c` (window `hc`) at point p
fn gi_sun_frac(c: u32, hc: vec4<f32>, p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, bx: vec3<f32>, by: vec3<f32>, h3: vec4<f32>, objects: bool) -> f32 {
    let res = u32(h3.z);
    let resf = h3.z;
    let texel = 2.0 * hc.w / resf;
    // receiver, pulled off the surface along its normal and toward the sun by a texel or two
    // slope-scaled bias: the depth of a sloped receiver varies by texel x tan(angle) across one texel
    let ndl_b = max(dot(n, l), 0.05);
    let tan_t = min(sqrt(1.0 - ndl_b * ndl_b) / ndl_b, 6.0);
    let pb = p + n * (texel * 0.5) + l * (texel * 0.25);
    let u0 = ((dot(pb, bx) - hc.x) / hc.w * 0.5 + 0.5) * resf;
    let v0 = ((dot(pb, by) - hc.y) / hc.w * 0.5 + 0.5) * resf;
    let tr = h3.w - dot(pb, l);
    let eps = texel * (0.5 + 0.5 * tan_t) + 0.02;
    // a FIXED tap pattern (no per-pixel rotation): the bilinear taps make the result a continuous function of the
    // position, where a per-pixel random rotation turned every edge into grain of +-0.15 (16 taps)
    let rot = 0.0;
    let tan_h = h3.y;
    let rs = max(clamp(tan_h * 24.0, texel * 1.5, 1.5), texel * 2.0); // never narrower than the minimum filter radius
    var nb = 0.0;
    var sum_t = 0.0;
    for (var k = 0u; k < 12u; k = k + 1u) {
        let o = gi_sun_vogel(k, 12u, rot) * (rs / texel);
        let t = gi_sun_t(c, i32(floor(u0 + o.x)), i32(floor(v0 + o.y)), res, objects);
        if (t < tr - eps) { nb = nb + 1.0; sum_t = sum_t + t; }
    }
    let trans = gi_sun_tex(c, i32(floor(u0)), i32(floor(v0)), res).z;
    if (nb < 0.5) { return clamp(trans, 0.0, 1.0); }
    let gap = tr - sum_t / nb;
    let w = clamp(tan_h * gap, texel * 2.0, 6.0);
    var lit = 0.0;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let o = gi_sun_vogel(k, 16u, rot + 1.7) * (w / texel);
        lit = lit + gi_sun_lit_bilinear(c, u0 + o.x, v0 + o.y, res, objects, tr - eps);
    }
    return clamp(lit / 16.0 * trans, 0.0, 1.0);
}
fn gi_sun_vis(p: vec3<f32>, n: vec3<f32>, objects: bool) -> f32 {
    let h3 = gi_sun.hdr[4];
    if (h3.x < 0.5 || gi.dims.w == 0u) { return 1.0; }
    let l = normalize(gi.sun_dir.xyz);
    let ndl = dot(n, l);
    if (ndl <= 0.0) { return 1.0; }
    let bx = gi_sun_basis_x(l);
    let by = cross(l, bx);
    let pu = dot(p, bx);
    let pv = dot(p, by);
    var c = 4u;
    var edge = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let h = gi_sun.hdr[k];
        let m = max(abs((pu - h.x) / h.w), abs((pv - h.y) / h.w));
        if (h.z > 0.5 && m < 0.94) { c = k; edge = m; break; }
    }
    if (c == 4u) { return 1.0; }
    var f = gi_sun_frac(c, gi_sun.hdr[c], p, n, l, bx, by, h3, objects);
    // blend toward the next (coarser, valid) cascade over the outer 14 % of this window
    if (c < 3u && edge > 0.80) {
        let hn = gi_sun.hdr[c + 1u];
        if (hn.z > 0.5 && max(abs((pu - hn.x) / hn.w), abs((pv - hn.y) / hn.w)) < 0.94) {
            let fn_ = gi_sun_frac(c + 1u, hn, p, n, l, bx, by, h3, objects);
            f = mix(f, fn_, smoothstep(0.80, 0.94, edge));
        }
    }
    return sqrt(f);
}
"#;

const COMPUTE_DECLS_WGSL: &str = r#"
struct BvhNode { aabb_min: vec3<f32>, left_first: u32, aabb_max: vec3<f32>, count: u32 };
struct Tri { a: vec3<f32>, _pa: f32, b: vec3<f32>, _pb: f32, c: vec3<f32>, _pc: f32 };
struct Light {
    pos: vec3<f32>, range: f32,
    color: vec3<f32>, sphere_pct: f32,
    dir: vec3<f32>, cos_cutoff: f32,
    angle_ratio: f32, angle_power: f32, far_end: f32, far_ratio: f32,
};
struct Emitter { v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, rad: vec4<f32> };
struct GiParams {
    origin: vec4<f32>, dims: vec4<u32>, sun_dir: vec4<f32>, sun_color: vec4<f32>, sky_color: vec4<f32>,
    misc: vec4<u32>, misc2: vec4<u32>, fog: vec4<f32>, bias: vec4<f32>, nee: vec4<f32>, dynp: vec4<u32>,
    inner_org: vec4<f32>, inner_dims: vec4<u32>, inner_base: vec4<i32>,
};
@group(0) @binding(0) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> tri_mat: array<u32>;
@group(0) @binding(3) var<storage, read> mats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> lights: array<Light>;
@group(0) @binding(5) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(6) var<uniform> gi: GiParams;
@group(0) @binding(7) var<storage, read_write> gi_probes: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> gi_vis: array<vec2<f32>>;
@group(0) @binding(9) var env_cube: texture_cube<f32>;
@group(0) @binding(10) var env_samp: sampler;
struct SunBuf { hdr: array<vec4<f32>, 5>, tex: array<vec4<f32>> };
@group(0) @binding(11) var<storage, read_write> sun: SunBuf;
"#;

/// The probe LOOKUP shared by the compute (previous-frame irradiance at ray hits) and fragment
/// (surface shading) sides. Both declare `gi`, `gi_probes`, `gi_vis` before this text.
pub const GI_SHARED_WGSL: &str = r#"
const GI_PI: f32 = 3.14159265;
const GI_VIS_RES: u32 = 8u;
const GI_SH_N: u32 = 28u;

fn gi_sign_not_zero(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}
// unit direction → octahedral [0,1]²
fn gi_oct_encode(d: vec3<f32>) -> vec2<f32> {
    let l1 = abs(d.x) + abs(d.y) + abs(d.z);
    var p = d.xy / max(l1, 1e-6);
    if (d.z < 0.0) { p = (1.0 - abs(p.yx)) * gi_sign_not_zero(p); }
    return p * 0.5 + 0.5;
}
// octahedral [0,1]² → unit direction
fn gi_oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - 1.0;
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) { let t = (1.0 - abs(v.yx)) * gi_sign_not_zero(v.xy); v = vec3<f32>(t.x, t.y, v.z); }
    return normalize(v);
}
fn gi_n_outer() -> u32 { return gi.dims.x * gi.dims.y * gi.dims.z; }
fn gi_probe_count() -> u32 { return gi.misc2.z; }
// INNER grid: a toroidal window of lattice cells (see GiParams.inner_*). Slot of an integer cell, and the cell a slot holds.
fn gi_inner_slot(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(gi.inner_dims.xyz);
    let m = ((c % d) + d) % d;
    return gi.inner_dims.w + u32(m.x + m.y * d.x + m.z * d.x * d.y);
}
fn gi_inner_cell_of(rel: u32) -> vec3<i32> {
    let d = vec3<i32>(gi.inner_dims.xyz);
    let r = i32(rel);
    let sc = vec3<i32>(r % d.x, (r / d.x) % d.y, r / (d.x * d.y));
    let b = gi.inner_base.xyz;
    return b + (((sc - b) % d) + d) % d;
}
fn gi_probe_index(c: vec3<u32>) -> u32 { return c.x + c.y * gi.dims.x + c.z * gi.dims.x * gi.dims.y; }
fn gi_probe_pos(c: vec3<u32>) -> vec3<f32> { return gi.origin.xyz + vec3<f32>(c) * gi.origin.w; }
fn gi_vis_texel(pi: u32, t: vec2<i32>) -> vec2<f32> {
    let tc = clamp(t, vec2<i32>(0), vec2<i32>(i32(GI_VIS_RES) - 1));
    return gi_vis[pi * GI_VIS_RES * GI_VIS_RES + u32(tc.y) * GI_VIS_RES + u32(tc.x)];
}
// bilinear octahedral visibility sample (mean, mean²) of probe `pi` in direction `d`
fn gi_vis_sample(pi: u32, d: vec3<f32>) -> vec2<f32> {
    let uv = gi_oct_encode(d) * f32(GI_VIS_RES) - 0.5;
    let f = floor(uv);
    let w = uv - f;
    let b = vec2<i32>(f);
    let s00 = gi_vis_texel(pi, b);
    let s10 = gi_vis_texel(pi, b + vec2<i32>(1, 0));
    let s01 = gi_vis_texel(pi, b + vec2<i32>(0, 1));
    let s11 = gi_vis_texel(pi, b + vec2<i32>(1, 1));
    return mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);
}
// irradiance E(n) from the L2 SH stored at slot `off` of the probe (0 = the total SH the shading
// side reads, 10 = the gather-only SH the tracer recurses on); Ramamoorthi-Hanrahan cosine convolution
fn gi_sh_irradiance_at(pi: u32, n: vec3<f32>, slot: u32) -> vec3<f32> {
    let b = pi * GI_SH_N + slot;
    let c0 = gi_probes[b + 0u].rgb; let c1 = gi_probes[b + 1u].rgb; let c2 = gi_probes[b + 2u].rgb;
    let c3 = gi_probes[b + 3u].rgb; let c4 = gi_probes[b + 4u].rgb; let c5 = gi_probes[b + 5u].rgb;
    let c6 = gi_probes[b + 6u].rgb; let c7 = gi_probes[b + 7u].rgb; let c8 = gi_probes[b + 8u].rgb;
    let x = n.x; let y = n.y; let z = n.z;
    let a0 = 3.14159265; let a1 = 2.0943951; let a2 = 0.78539816;
    var e = a0 * 0.282095 * c0;
    e = e + a1 * 0.488603 * (y * c1 + z * c2 + x * c3);
    e = e + a2 * (1.092548 * (x * y * c4 + y * z * c5 + x * z * c7) + 0.315392 * (3.0 * z * z - 1.0) * c6 + 0.546274 * (x * x - y * y) * c8);
    return max(e, vec3<f32>(0.0));
}
// light-plane basis for a unit direction toward the sun (rtgi.rs `sun_basis` is the same formula)
fn gi_sun_basis_x(l: vec3<f32>) -> vec3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.z) > 0.95);
    return normalize(cross(up, l));
}
fn gi_on() -> bool { return gi.dims.w > 0u; }
fn gdbg_ignore_valid() -> bool { return __GI_DEBUG__ == 3; }
// Irradiance / π at surface point p with normal n (v = unit vector from p toward the viewer/probe
// side, used for the DDGI bias), from the 8 surrounding probes with backface + Chebyshev
// visibility weights. Same units as the baked `bake.rgb`.
fn gi_outer_set(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, slot: u32) -> vec3<f32> {
    let sp = gi.origin.w;
    let o = gi.origin.xyz;
    let bp = p + (n * gi.bias.x + v * gi.bias.y) * sp;
    let g = (bp - o) / sp;
    let maxc = vec3<f32>(gi.dims.xyz) - vec3<f32>(1.0);
    let gc = clamp(g, vec3<f32>(0.0), maxc);
    let base = floor(min(gc, maxc - vec3<f32>(1.0)));
    let f = clamp(gc - base, vec3<f32>(0.0), vec3<f32>(1.0));
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    var nvalid = 0u; var nin = 0u;
    let vmax = sp * gi.bias.z;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let off = vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
        let cc = base + off;
        if (any(cc < vec3<f32>(0.0)) || any(cc > maxc)) { continue; }
        nin = nin + 1u;
        let ci = vec3<u32>(cc);
        let pi = gi_probe_index(ci);
        let rec = gi_probes[pi * GI_SH_N + 9u];
        if (rec.w < 0.5 && gdbg_ignore_valid() == false) { continue; }
        nvalid = nvalid + 1u;
        let ppos = o + cc * sp + rec.xyz;
        let tw = mix(vec3<f32>(1.0) - f, f, off);
        var w = tw.x * tw.y * tw.z;
        let to_probe = ppos - p;
        let dir = normalize(to_probe);
        // backface: probes behind the surface contribute (almost) nothing
        let wb = (dot(dir, n) + 1.0) * 0.5;
        w = w * (wb * wb + 0.2);
        // Chebyshev visibility from the probe's octahedral distance map
        let dist = min(length(ppos - bp), vmax);
        let vs = gi_vis_sample(pi, -dir);
        let mean = vs.x;
        if (dist > mean) {
            let variance = abs(vs.y - mean * mean) + 1e-4;
            var cheb = variance / (variance + (dist - mean) * (dist - mean));
            cheb = max(cheb * cheb * cheb, 0.0);
            w = w * max(cheb, 0.02);
        }
        sum = sum + w * gi_sh_irradiance_at(pi, n, slot);
        wsum = wsum + w;
    }
    let gdbg = __GI_DEBUG__;
    if (gdbg == 1) { return vec3<f32>(f32(nvalid) / 8.0, f32(nin) / 8.0, select(0.0, 1.0, wsum > 1e-5)); }
    if (gdbg == 2) { let gg = (p - o) / sp; return fract(gg * 0.25); }
    if (wsum <= 1e-5) {
        // No corner of the cell survived (every probe of the cell sits inside geometry): the old code returned BLACK, a hole
        // in the lighting. Fall back to the valid probes of the surrounding 4x4x4 block, weighted by distance, by how
        // squarely they sit in FRONT of the surface (a probe behind it lights the other side of the wall) and by the same
        // Chebyshev visibility the corners get, so a pocket takes the light of the air it faces and not the next room's.
        var fsum = vec3<f32>(0.0); var fw = 0.0; var fden = 0.0;
        var asum = vec3<f32>(0.0); var aw = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            let cc = base + vec3<f32>(f32(k & 3u) - 1.0, f32((k >> 2u) & 3u) - 1.0, f32((k >> 4u) & 3u) - 1.0);
            if (any(cc < vec3<f32>(0.0)) || any(cc > maxc)) { continue; }
            let pi = gi_probe_index(vec3<u32>(cc));
            let rec = gi_probes[pi * GI_SH_N + 9u];
            if (rec.w < 0.5) { continue; }
            let d = cc - gc;
            let wd = 1.0 / (0.25 + dot(d, d));
            let e = gi_sh_irradiance_at(pi, n, slot);
            asum = asum + wd * e; aw = aw + wd;
            let ppos = o + cc * sp + rec.xyz;
            let to_probe = ppos - p;
            let dist0 = length(to_probe);
            let dir = to_probe / max(dist0, 1e-4);
            let fr = max(dot(dir, n), 0.0);
            var w = wd * fr * fr;
            fden = fden + w; // the same weight WITHOUT the visibility test: an occluded pocket stays dark instead of renormalising to the open air
            let dist = min(length(ppos - bp), vmax);
            let vs = gi_vis_sample(pi, -dir);
            if (dist > vs.x) {
                let variance = abs(vs.y - vs.x * vs.x) + 1e-4;
                var cheb = variance / (variance + (dist - vs.x) * (dist - vs.x));
                cheb = max(cheb * cheb * cheb, 0.0);
                w = w * max(cheb, 0.25); // a pocket narrower than the probe spacing is not resolved: some light always seeps in
            }
            fsum = fsum + w * e; fw = fw + w;
        }
        // nothing in front and visible: the plain distance-weighted valid probes (still better than black)
        if (fden <= 1e-4) { fsum = asum; fden = aw; }
        if (fden <= 0.0) { return vec3<f32>(0.0); }
        return (fsum / fden) * (gi.bias.w / GI_PI);
    }
    return (sum / wsum) * (gi.bias.w / GI_PI);
}

// Irradiance / pi from the INNER grid (w = 1 when at least one corner contributed, else 0 and the caller uses the outer grid).
// Same weights as the outer grid (trilinear x backface x Chebyshev), without the hole fallback.
fn gi_inner_eval(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, slot: u32) -> vec4<f32> {
    let sp = gi.inner_org.w;
    let o = gi.inner_org.xyz;
    let bp = p + (n * gi.bias.x + v * gi.bias.y) * sp;
    let g = (bp - o) / sp;
    let bf = floor(g);
    let f = g - bf;
    let b0 = gi.inner_base.xyz;
    let d = vec3<i32>(gi.inner_dims.xyz);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    var cover = 0.0;
    let vmax = sp * gi.bias.z;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let offi = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let ci = vec3<i32>(bf) + offi;
        if (any(ci < b0) || any(ci >= b0 + d)) { continue; }
        let pi = gi_inner_slot(ci);
        let rec = gi_probes[pi * GI_SH_N + 9u];
        if (rec.w < 0.5) { continue; }
        let off = vec3<f32>(offi);
        let ppos = o + vec3<f32>(ci) * sp + rec.xyz;
        let tw = mix(vec3<f32>(1.0) - f, f, off);
        var w = tw.x * tw.y * tw.z;
        cover = cover + w;
        let to_probe = ppos - p;
        let dir = normalize(to_probe);
        let wb = (dot(dir, n) + 1.0) * 0.5;
        w = w * (wb * wb + 0.2);
        let dist = min(length(ppos - bp), vmax);
        let vs = gi_vis_sample(pi, -dir);
        let mean = vs.x;
        if (dist > mean) {
            let variance = abs(vs.y - mean * mean) + 1e-4;
            var cheb = variance / (variance + (dist - mean) * (dist - mean));
            cheb = max(cheb * cheb * cheb, 0.0);
            w = w * max(cheb, 0.02);
        }
        sum = sum + w * gi_sh_irradiance_at(pi, n, slot);
        wsum = wsum + w;
    }
    if (wsum <= 1e-5) { return vec4<f32>(0.0); }
    // w = how much of the cell's trilinear weight sits on valid probes: the caller fades to the coarse grid as it goes to 0, so
    // a cell whose probes sit in geometry hands over smoothly instead of jumping
    return vec4<f32>((sum / wsum) * (gi.bias.w / GI_PI), clamp(cover * 2.0, 0.0, 1.0));
}
// Weight of the inner grid at p: 0 outside / near the window's edge, 1 deep inside (a 2-cell smooth fade, so the seam to
// the coarse grid is continuous and a freshly exposed slab of probes is never shown before it has been traced).
fn gi_inner_weight(p: vec3<f32>) -> f32 {
    if (gi.inner_org.w <= 0.0) { return 0.0; }
    let gp = (p - gi.inner_org.xyz) / gi.inner_org.w;
    let lo = vec3<f32>(gi.inner_base.xyz);
    let hi = lo + vec3<f32>(gi.inner_dims.xyz) - vec3<f32>(1.0);
    let e3 = min(gp - lo, hi - gp);
    let e = min(e3.x, min(e3.y, e3.z));
    let t = clamp((e - 1.5) / 2.0, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
fn gi_irradiance_set(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, slot: u32) -> vec3<f32> {
    let t = gi_inner_weight(p);
    if (t > 0.0) {
        let ei = gi_inner_eval(p, n, v, slot);
        let tc = t * ei.w;
        if (tc > 0.0) {
            if (tc >= 1.0) { return ei.rgb; }
            return mix(gi_outer_set(p, n, v, slot), ei.rgb, tc);
        }
    }
    return gi_outer_set(p, n, v, slot);
}
fn gi_irradiance(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> { return gi_irradiance_set(p, n, v, 0u); }
// How visible a point light at `lp` is from the surface point p, read from the probe field instead of tracing: the nearest valid
// probe of the cell (inner grid when p is inside it) knows how far it sees in every direction (its distance map), so the light is
// hidden when the probe sees a wall before reaching the light's distance (the Chebyshev test the probes' own weights use). Resolution is the
// probe spacing, so a light behind a thin wall next to the probe can leak or a doorway can clip; it is a shadow at room scale.
fn gi_light_vis(p: vec3<f32>, n: vec3<f32>, lp: vec3<f32>) -> f32 {
    var best_d = 1.0e9;
    var best_pi = 0u;
    var best_pos = vec3<f32>(0.0);
    var have = false;
    let tin = gi_inner_weight(p);
    if (tin > 0.5) {
        let sp = gi.inner_org.w;
        let g = (p + n * 0.2 * sp - gi.inner_org.xyz) / sp;
        let bf = vec3<i32>(floor(g));
        let b0 = gi.inner_base.xyz;
        let d3 = vec3<i32>(gi.inner_dims.xyz);
        for (var k = 0u; k < 8u; k = k + 1u) {
            let ci = bf + vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
            if (any(ci < b0) || any(ci >= b0 + d3)) { continue; }
            let pi = gi_inner_slot(ci);
            let rec = gi_probes[pi * GI_SH_N + 9u];
            if (rec.w < 0.5) { continue; }
            let pp = gi.inner_org.xyz + vec3<f32>(ci) * sp + rec.xyz;
            let dd = length(pp - p);
            if (dd < best_d) { best_d = dd; best_pi = pi; best_pos = pp; have = true; }
        }
    }
    if (!have) {
        let sp = gi.origin.w;
        let maxc = vec3<f32>(gi.dims.xyz) - vec3<f32>(1.0);
        let gc = clamp((p + n * 0.2 * sp - gi.origin.xyz) / sp, vec3<f32>(0.0), maxc);
        let base = floor(min(gc, maxc - vec3<f32>(1.0)));
        for (var k = 0u; k < 8u; k = k + 1u) {
            let cc = base + vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
            if (any(cc < vec3<f32>(0.0)) || any(cc > maxc)) { continue; }
            let pi = gi_probe_index(vec3<u32>(cc));
            let rec = gi_probes[pi * GI_SH_N + 9u];
            if (rec.w < 0.5) { continue; }
            let pp = gi.origin.xyz + cc * sp + rec.xyz;
            let dd = length(pp - p);
            if (dd < best_d) { best_d = dd; best_pi = pi; best_pos = pp; have = true; }
        }
    }
    if (!have) { return 1.0; }
    let to_l = lp - best_pos;
    let dl = length(to_l);
    if (dl < 1e-3) { return 1.0; }
    let vs = gi_vis_sample(best_pi, to_l / dl);
    // the light is nearer than the first wall the probe sees in that direction: visible; else a soft Chebyshev fall-off
    if (dl <= vs.x) { return 1.0; }
    let variance = abs(vs.y - vs.x * vs.x) + 1e-3;
    let c = variance / (variance + (dl - vs.x) * (dl - vs.x));
    return clamp(c * c * c, 0.0, 1.0);
}
"#;

const COMPUTE_WGSL: &str = r#"
// ---------------- BVH tracer (mirrors lightbake_gpu; sky-flagged materials are never hit) ----------------
const EPS: f32 = 0.02;
const T_MAX: f32 = 1.0e9;
const FIREFLY_CAP: f32 = 12.0;

fn pcg(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x * 747796405u + 2891336453u;
    let word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    *state = x;
    return (word >> 22u) ^ word;
}
fn randf(state: ptr<function, u32>) -> f32 { return f32(pcg(state)) * (1.0 / 4294967296.0); }

fn hit_aabb(o: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tsm = min(t0, t1);
    let tbg = max(t0, t1);
    let tn = max(max(tsm.x, tsm.y), tsm.z);
    let tf = min(min(tbg.x, tbg.y), tbg.z);
    if (tf >= max(tn, 0.0) && tn < tmax) { return max(tn, 0.0); }
    return -1.0;
}
fn hit_tri(o: vec3<f32>, d: vec3<f32>, i: u32, tmax: f32) -> f32 {
    let tr = tris[i];
    let e1 = tr.b - tr.a;
    let e2 = tr.c - tr.a;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-8) { return -1.0; }
    let invd = 1.0 / det;
    let tv = o - tr.a;
    let u = dot(tv, p) * invd;
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(tv, e1);
    let v = dot(d, q) * invd;
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let t = dot(e2, q) * invd;
    if (t > EPS && t < tmax) { return t; }
    return -1.0;
}
// material flag: 0 opaque, 1 foliage, 2 sky shell (never hit), 3 translucent (passes light × transmittance)
fn tri_flag(i: u32) -> f32 { return mats[tri_mat[i] * 3u].w; }
fn tri_is_sky(i: u32) -> bool { let f = tri_flag(i); return f > 1.5; }
// Light transmitted along o+d·[0,tmax] through the BVH rooted at `root`: 0 behind an opaque surface, the
// product of the transmittances of any translucent surfaces crossed (glass roofs / panes), 1 when clear.
fn visibility_root(root: u32, o: vec3<f32>, d: vec3<f32>, tmax: f32) -> vec3<f32> {
    let inv = 1.0 / d;
    var stack: array<u32, 48>;
    var sp = 0;
    stack[0] = root; sp = 1;
    var tr = vec3<f32>(1.0);
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = nodes[ni];
        if (hit_aabb(o, inv, node.aabb_min, node.aabb_max, tmax) < 0.0) { continue; }
        if (node.count > 0u) {
            for (var k = 0u; k < node.count; k = k + 1u) {
                let ti = node.left_first + k;
                let f = tri_flag(ti);
                if (f > 1.5 && f < 2.5) { continue; }
                if (hit_tri(o, d, ti, tmax) > 0.0) {
                    if (f > 2.5) { tr = tr * mats[tri_mat[ti] * 3u + 2u].rgb; if (max(tr.r, max(tr.g, tr.b)) < 0.01) { return vec3<f32>(0.0); } }
                    else { return vec3<f32>(0.0); }
                }
            }
        } else {
            if (sp < 46) {
                stack[sp] = node.left_first; sp = sp + 1;
                stack[sp] = node.left_first + 1u; sp = sp + 1;
            }
        }
    }
    return tr;
}
// The static map first, then the dynamic occluders (placed Forge pieces) when there are any.
fn visibility(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> vec3<f32> {
    var tr = visibility_root(0u, o, d, tmax);
    if (gi.dynp.x != 0xffffffffu && max(tr.r, max(tr.g, tr.b)) > 0.0) { tr = tr * visibility_root(gi.dynp.x, o, d, tmax); }
    return tr;
}
fn occluded(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> bool { let v = visibility(o, d, tmax); return max(v.r, max(v.g, v.b)) <= 0.0; }
// nearest hit under `root` closer than `best_in`; returns the distance (or -1) and the triangle
fn trace_root(root: u32, o: vec3<f32>, d: vec3<f32>, best_in: f32, out_tri: ptr<function, u32>) -> f32 {
    let inv = 1.0 / d;
    var stack: array<u32, 48>;
    var sp = 0;
    stack[0] = root; sp = 1;
    var best = best_in;
    var found = 0xffffffffu;
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = nodes[ni];
        if (hit_aabb(o, inv, node.aabb_min, node.aabb_max, best) < 0.0) { continue; }
        if (node.count > 0u) {
            for (var k = 0u; k < node.count; k = k + 1u) {
                let ti = node.left_first + k;
                if (tri_flag(ti) > 1.5) { continue; }
                let t = hit_tri(o, d, ti, best);
                if (t > 0.0) { best = t; found = ti; }
            }
        } else {
            if (sp < 46) {
                stack[sp] = node.left_first; sp = sp + 1;
                stack[sp] = node.left_first + 1u; sp = sp + 1;
            }
        }
    }
    *out_tri = found;
    if (found == 0xffffffffu) { return -1.0; }
    return best;
}
fn trace(o: vec3<f32>, d: vec3<f32>, out_tri: ptr<function, u32>) -> f32 {
    var tri_s: u32 = 0xffffffffu;
    var best = trace_root(0u, o, d, T_MAX, &tri_s);
    *out_tri = tri_s;
    if (gi.dynp.x != 0xffffffffu) {
        var tri_d: u32 = 0xffffffffu;
        let limit = select(T_MAX, best, best >= 0.0);
        let td = trace_root(gi.dynp.x, o, d, limit, &tri_d);
        if (td >= 0.0) { best = td; *out_tri = tri_d; }
    }
    return best;
}
fn tri_normal(i: u32) -> vec3<f32> {
    let tr = tris[i];
    return normalize(cross(tr.b - tr.a, tr.c - tr.a));
}
fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn fog_transmittance(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> f32 {
    let sigma = gi.fog.x;
    if (sigma <= 0.0) { return 1.0; }
    let top = gi.fog.y;
    var t_below = 0.0;
    let dz = d.z;
    if (abs(dz) < 1e-6) {
        if (o.z < top) { t_below = tmax; }
    } else {
        let tc = (top - o.z) / dz;
        if (dz < 0.0) { t_below = max(0.0, tmax - max(tc, 0.0)); } else { t_below = clamp(tc, 0.0, tmax); }
    }
    return exp(-sigma * min(t_below, 1.0e5));
}
// one next-event-estimation sample of the emissive area lights (area×luminance CDF)
fn sample_emitters(x: vec3<f32>, n: vec3<f32>, s: ptr<function, u32>) -> vec3<f32> {
    let ne = gi.misc2.y;
    if (ne == 0u) { return vec3<f32>(0.0); }
    let total = gi.fog.z;
    if (total <= 0.0) { return vec3<f32>(0.0); }
    let u = randf(s) * total;
    var lo = 0u;
    var hi = ne - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (emitters[mid].v0.w < u) { lo = mid + 1u; } else { hi = mid; }
    }
    let E = emitters[lo];
    let a = E.v0.xyz; let b = E.v1.xyz; let c = E.v2.xyz;
    var b0 = randf(s); var b1 = randf(s);
    if (b0 + b1 > 1.0) { b0 = 1.0 - b0; b1 = 1.0 - b1; }
    let y = a + b0 * (b - a) + b1 * (c - a);
    let nn = cross(b - a, c - a);
    let nlen = length(nn);
    if (nlen < 1e-8) { return vec3<f32>(0.0); }
    let n_emit = nn / nlen;
    let w = y - x;
    let dist = length(w);
    if (dist < 1e-3) { return vec3<f32>(0.0); }
    let wi = w / dist;
    let cos_recv = dot(n, wi);
    if (cos_recv <= 0.0) { return vec3<f32>(0.0); }
    let cos_emit = abs(dot(n_emit, wi));
    if (cos_emit <= 0.0) { return vec3<f32>(0.0); }
    // shadow ray stops `gi.nee.z` short of the emitter: fixtures sit recessed behind their housings
    var ve = vec3<f32>(1.0);
    if (__GI_CDEBUG__ != 7) { ve = visibility(x, wi, max(dist - max(gi.nee.z, 2.0 * EPS), EPS)); }
    if (max(ve.r, max(ve.g, ve.b)) <= 0.0) { return vec3<f32>(0.0); }
    let lum_i = max(E.v1.w, 1e-6);
    let g = min(cos_recv * cos_emit / (dist * dist), 1.0);
    var dE = E.rad.xyz * ve * (g * (total / lum_i));
    let m = lum(dE);
    let cap = max(gi.nee.y, 0.1);
    if (m > cap) { dE = dE * (cap / m); }
    return dE * fog_transmittance(x, wi, dist);
}
// ---- the engine sky (sky render_model "sky lights" list) lives behind the materials in `mats` ----
// header = (record count, sun 1st bounce scaler, sky 1st bounce scaler, total weight); record i =
// (dir toward the light, solid angle), (radiance rgb, cumulative sampling weight). Count 0 = legacy sky.
fn sky_n() -> u32 { return u32(mats[gi.dynp.y].x); }
fn sky_sun_1st() -> f32 { return mats[gi.dynp.y].y; }
fn sky_sky_1st() -> f32 { return mats[gi.dynp.y].z; }
fn sky_total_w() -> f32 { return mats[gi.dynp.y].w; }
fn sky_dir_om(i: u32) -> vec4<f32> { return mats[gi.dynp.y + 2u + 2u * i]; }
fn sky_rgb_cum(i: u32) -> vec4<f32> { return mats[gi.dynp.y + 3u + 2u * i]; }
const SKY_NEE: u32 = 2u;
// Unbiased estimate of the sky irradiance sum(rgb * Omega * max(n.l, 0) * V) at a surface point from
// SKY_NEE records picked by emission (the Tool shoots every record; probes can not afford 100 rays per hit).
fn sky_irradiance(x: vec3<f32>, n: vec3<f32>, s: ptr<function, u32>) -> vec3<f32> {
    let ns = sky_n();
    if (ns == 0u) { return vec3<f32>(0.0); }
    let total = sky_total_w();
    var acc = vec3<f32>(0.0);
    for (var k = 0u; k < SKY_NEE; k = k + 1u) {
        let u = randf(s) * total;
        var lo = 0u;
        var hi = ns - 1u;
        loop {
            if (lo >= hi) { break; }
            let mid = (lo + hi) / 2u;
            if (sky_rgb_cum(mid).w < u) { lo = mid + 1u; } else { hi = mid; }
        }
        let dr = sky_dir_om(lo);
        let c = dot(n, dr.xyz);
        if (c <= 0.0) { continue; }
        let cr = sky_rgb_cum(lo);
        var prev = 0.0;
        if (lo > 0u) { prev = sky_rgb_cum(lo - 1u).w; }
        let p = (cr.w - prev) / total;
        let v = visibility(x, dr.xyz, T_MAX);
        if (max(v.r, max(v.g, v.b)) <= 0.0) { continue; }
        acc = acc + cr.xyz * (dr.w * c) * v / p;
    }
    return acc / f32(SKY_NEE);
}
// direct irradiance at a surface point: sun + scenario point/spot lights + one emitter NEE sample.
// Engine mode: the sun is the BOUNCE source here, scaled by the sky's `sun 1st bounce scaler`.
fn direct_irradiance(x: vec3<f32>, n: vec3<f32>, s: ptr<function, u32>) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    let sun_dir = normalize(gi.sun_dir.xyz);
    let ndl = dot(n, sun_dir);
    let sun_scale = select(1.0, sky_sun_1st(), sky_n() > 0u);
    if (ndl > 0.0 && lum(gi.sun_color.xyz) > 0.0) {
        let vs = visibility(x, sun_dir, T_MAX);
        if (max(vs.r, max(vs.g, vs.b)) > 0.0) { e = e + gi.sun_color.xyz * vs * (ndl * sun_scale) * fog_transmittance(x, sun_dir, T_MAX); }
    }
    let nl = gi.misc2.x;
    for (var li = 0u; li < nl; li = li + 1u) {
        let L = lights[li];
        let toL = L.pos - x;
        let dist = length(toL);
        let range = select(L.far_end, L.range, L.range > 1.0);
        if (dist > range || dist < 1e-4) { continue; }
        let ldir = toL / dist;
        let ndl2 = dot(n, ldir);
        if (ndl2 <= 0.0) { continue; }
        var att = clamp(1.0 - dist / max(range, 0.01), 0.0, 1.0);
        att = att * att;
        if (L.sphere_pct < 0.5) {
            let cd = dot(-ldir, normalize(L.dir));
            if (cd < L.cos_cutoff) { continue; }
            let cone = clamp((cd - L.cos_cutoff) / max(1.0 - L.cos_cutoff, 1e-3), 0.0, 1.0);
            att = att * pow(cone, max(L.angle_power, 1.0));
        }
        if (att <= 0.0) { continue; }
        let vl = visibility(x, ldir, dist - 0.05);
        if (max(vl.r, max(vl.g, vl.b)) <= 0.0) { continue; }
        e = e + L.color * vl * (ndl2 * att) * fog_transmittance(x, ldir, dist);
    }
    let ns = u32(clamp(gi.nee.x, 0.0, 64.0));
    if (ns > 0u) {
        var acc = vec3<f32>(0.0);
        for (var k = 0u; k < ns; k = k + 1u) { acc = acc + sample_emitters(x, n, s); }
        e = e + acc / f32(ns);
    }
    return e;
}
fn fib_dir(i: u32, n: u32) -> vec3<f32> {
    let phi = f32(i) * 2.39996323;
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn sh_basis(d: vec3<f32>, k: u32) -> f32 {
    let x = d.x; let y = d.y; let z = d.z;
    switch (k) {
        case 0u: { return 0.282095; }
        case 1u: { return 0.488603 * y; }
        case 2u: { return 0.488603 * z; }
        case 3u: { return 0.488603 * x; }
        case 4u: { return 1.092548 * x * y; }
        case 5u: { return 1.092548 * y * z; }
        case 6u: { return 0.315392 * (3.0 * z * z - 1.0); }
        case 7u: { return 1.092548 * x * z; }
        default: { return 0.546274 * (x * x - y * y); }
    }
}

const WG: u32 = 64u;
const SLOT_GATHER: u32 = 10u;
const SLOT_SKY: u32 = 19u;
var<workgroup> r_dir: array<vec3<f32>, 256>;
var<workgroup> r_rad: array<vec3<f32>, 256>;
var<workgroup> r_dist: array<f32, 256>;
var<workgroup> s_rad: array<vec3<f32>, 256>;
var<workgroup> backfaces: atomic<u32>;
var<workgroup> cb_dist: array<f32, 64>;
var<workgroup> cb_dir: array<vec3<f32>, 64>;
var<workgroup> cf_dist: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // a region update dispatches an explicit probe list (parked behind the dynamic region of tri_mat)
    let listed = (gi.dynp.w & 1u) != 0u;
    let probe = select(wid.x + gi.misc2.w, tri_mat[gi.dynp.z + wid.x], listed);
    let nprobes = gi_probe_count();
    if (probe >= nprobes) { return; }
    let t = lid.x;
    if (t == 0u) { atomicStore(&backfaces, 0u); }
    workgroupBarrier();
    let rpt = clamp(gi.misc.y, 1u, 4u);
    let nrays = WG * rpt;
    let nx = gi.dims.x; let ny = gi.dims.y;
    let n_out = gi_n_outer();
    let is_inner = probe >= n_out;
    // a probe whose cell was just handed to it by the moving inner window starts from scratch
    let new_cell = (gi.dynp.w & 4u) != 0u;
    var rec_old = gi_probes[probe * GI_SH_N + 9u];
    if (new_cell) { rec_old = vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    var ppos: vec3<f32>;
    var sp_p = gi.origin.w;
    if (is_inner) {
        sp_p = gi.inner_org.w;
        ppos = gi.inner_org.xyz + vec3<f32>(gi_inner_cell_of(probe - n_out)) * sp_p + rec_old.xyz;
    } else {
        let c = vec3<u32>(probe % nx, (probe / nx) % ny, probe / (nx * ny));
        ppos = gi_probe_pos(c) + rec_old.xyz;
    }
    // ENGINE mode = the scene carried the sky render_model's light list (see engine_gi.rs)
    let nsky = sky_n();
    let engine = nsky > 0u;
    // uninitialised probe (vis record still zero) → take the full estimate this pass
    let fresh = gi_vis[probe * GI_VIS_RES * GI_VIS_RES].y <= 0.0 || new_cell;
    // per-frame random rotation of the fibonacci set (Shoemake uniform quaternion)
    var rs: u32 = gi.misc.x * 7919u + 17u;
    let u1 = randf(&rs); let u2 = randf(&rs); let u3 = randf(&rs);
    let q = vec4<f32>(sqrt(1.0 - u1) * sin(6.2831853 * u2), sqrt(1.0 - u1) * cos(6.2831853 * u2), sqrt(u1) * sin(6.2831853 * u3), sqrt(u1) * cos(6.2831853 * u3));
    var rng: u32 = probe * 9781u + gi.misc.x * 6271u + t * 131u + 1u;
    let vmax = sp_p * gi.bias.z;
    var my_back = 0u;
    var my_cb = 1.0e9; var my_cbd = vec3<f32>(0.0); var my_cf = 1.0e9;
    for (var k = 0u; k < rpt; k = k + 1u) {
        let ri = t * rpt + k;
        let d = normalize(quat_rotate(q, fib_dir(ri, nrays)));
        var tri: u32 = 0u;
        let th = trace(ppos, d, &tri);
        var rad = vec3<f32>(0.0);
        var dist = vmax;
        if (th < 0.0) {
            if (!engine) {
                // LEGACY sky: the captured sky env cube (the map's own sky model, sun glare included)
                // when available, else the flat sky ambient; a dimmer ground/ocean stand-in below the horizon.
                var sky = gi.sky_color.xyz * select(gi.sky_color.w, 1.0, d.z >= 0.0);
                if (gi.sun_dir.w > 0.5) {
                    // cube LUMINANCE (angular distribution, sun glare) × the authored sky-light HUE
                    let cl = lum(textureSampleLevel(env_cube, env_samp, d, 1.0).rgb) * gi.sun_color.w;
                    let hue = gi.sky_color.xyz / max(lum(gi.sky_color.xyz), 1e-4);
                    let cs = cl * hue;
                    sky = select(max(cs, gi.sky_color.xyz * gi.sky_color.w * 0.1), cs, d.z >= 0.0);
                }
                rad = sky * visibility(ppos, d, 1.0e4) * fog_transmittance(ppos, d, 1.0e4);
            }
            // ENGINE: a ray that escapes carries nothing — the sky is the deterministic term below
            // (the Tool's final gather integrates L_exit of the HIT surfaces; escape is 0).
        } else {
            let ng = tri_normal(tri);
            if (dot(ng, d) > 0.0) {
                // back face: inside geometry from this probe's view — no light, short distance
                my_back = my_back + 1u;
                if (th < my_cb) { my_cb = th; my_cbd = d; }
                dist = min(th, vmax) * 0.2;
            } else {
                let m = mats[tri_mat[tri] * 3u];
                let em = mats[tri_mat[tri] * 3u + 1u].rgb;
                let hp = ppos + d * th;
                let ho = hp + ng * EPS;
                let e_direct = direct_irradiance(ho, ng, &rng);
                if (engine) {
                    // Tool C7/C11: exit radiance of the first bounce = albedo × D with D = (sun×sun_1st +
                    // sky×sky_1st + lights)/π; higher bounces = the photon chain (albedo clamped so its
                    // largest channel is <= 0.8) fed by the previous gather field (slots 10..18). The
                    // lookup's energy gain is undone: the recursion must not amplify itself.
                    let e_sky = sky_irradiance(ho, ng, &rng) * sky_sky_1st();
                    let amax = max(m.r, max(m.g, m.b));
                    let rho_c = select(m.rgb, m.rgb * (0.8 / max(amax, 1e-4)), amax > 0.8);
                    let e_g = gi_irradiance_set(ho, ng, -d, SLOT_GATHER) * GI_PI / max(gi.bias.w, 0.01);
                    rad = (m.rgb * (e_direct + e_sky) / GI_PI + rho_c * e_g / GI_PI + em) * fog_transmittance(ppos, d, th);
                } else {
                    let e_prev = gi_irradiance(ho, ng, -d) * GI_PI; // previous frame, back to irradiance units
                    // outgoing radiance of the hit = Lambert bounce + the surface's own self-illum
                    rad = (m.rgb * (e_direct + e_prev) / GI_PI + em) * fog_transmittance(ppos, d, th);
                }
                dist = min(th, vmax);
                my_cf = min(my_cf, th);
            }
        }
        let cdbg = __GI_CDEBUG__;
        if (cdbg == 1) { rad = select(select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), dist > 0.0 && th >= 0.0 && dot(tri_normal(tri), d) <= 0.0), vec3<f32>(1.0, 0.0, 0.0), th < 0.0); }
        if (cdbg == 3) { rad = select(vec3<f32>(0.0), vec3<f32>(1.0), !occluded(ppos, vec3<f32>(0.0, 0.0, 1.0), T_MAX)); }
        if (cdbg == 4) { var tt: u32 = 0u; let tu = trace(ppos, vec3<f32>(0.0, 0.0, 1.0), &tt); rad = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, min(tu, 200.0) / 200.0, 0.0), tu >= 0.0); }
        if (cdbg == 5 && th >= 0.0) { let ng5 = tri_normal(tri); if (dot(ng5, d) <= 0.0) { rad = sample_emitters(ppos + d * th + ng5 * EPS, ng5, &rng) + sample_emitters(ppos + d * th + ng5 * EPS, ng5, &rng); } else { rad = vec3<f32>(0.0); } }
        if (cdbg == 6) { rad = vec3<f32>(f32(gi.misc2.y) / 10000.0, gi.fog.z / 1000.0, gi.nee.x / 10.0); }
        if (cdbg == 2) { rad = select(vec3<f32>(0.0), vec3<f32>(1.0), th >= 0.0 && dot(tri_normal(tri), d) <= 0.0 && !occluded(ppos + d * th + tri_normal(tri) * EPS, normalize(gi.sun_dir.xyz), T_MAX)); }
        r_dir[ri] = d;
        r_rad[ri] = rad;
        r_dist[ri] = dist;
    }
    // ENGINE sky-direct term: every sky record once, through the scene (the Tool's `faux_light_sky`;
    // a probe has no normal so no hemisphere clip). Deterministic — it is refreshed only for a fresh
    // probe or when an occluder changed (region dispatch), then kept in slots 19..27.
    let do_sky = engine && (fresh || (gi.dynp.w & 2u) != 0u);
    if (do_sky) {
        for (var i = t; i < nsky; i = i + WG) {
            let dr = sky_dir_om(i);
            let cr = sky_rgb_cum(i);
            s_rad[i] = cr.xyz * dr.w * visibility(ppos, dr.xyz, T_MAX);
        }
    }
    if (my_back > 0u) { atomicAdd(&backfaces, my_back); }
    cb_dist[t] = my_cb; cb_dir[t] = my_cbd; cf_dist[t] = my_cf;
    workgroupBarrier();
    let alpha = select(gi.fog.w, 1.0, fresh);
    let nb = atomicLoad(&backfaces);
    let back_frac = f32(nb) / f32(nrays);
    let valid = select(1.0, 0.0, back_frac > 0.25);
    // DDGI probe relocation (thread 0): push a probe that sees too many back faces away from the
    // closest one; keep a small clearance from the closest front face; never leave its own cell.
    if (t == 0u) {
        var best = 1.0e9; var bd = vec3<f32>(0.0); var cf = 1.0e9;
        for (var i = 0u; i < WG; i = i + 1u) {
            if (cb_dist[i] < best) { best = cb_dist[i]; bd = cb_dir[i]; }
            cf = min(cf, cf_dist[i]);
        }
        var off = rec_old.xyz;
        let sp = sp_p;
        if (back_frac > 0.25 && best < 1.0e8) {
            off = off - bd * (best + 0.1 * sp);
        } else if (cf < 0.08 * sp) {
            // too close to a wall in front: nudge back along the grid-centre direction a little
            off = off * 0.9;
        }
        let lim = 0.6 * sp;
        off = clamp(off, vec3<f32>(-lim), vec3<f32>(lim));
        gi_probes[probe * GI_SH_N + 9u] = vec4<f32>(off, valid);
    }
    // SH projection: threads 0..8 each own one coefficient (all rays, all channels). Slot 9 is the
    // relocation record and is NOT an SH coefficient (an earlier version let a tenth thread blend SH
    // data into it).
    if (t < 9u) {
        var acc = vec3<f32>(0.0);
        for (var i = 0u; i < nrays; i = i + 1u) {
            acc = acc + r_rad[i] * sh_basis(r_dir[i], t);
        }
        acc = acc * (4.0 * GI_PI / f32(nrays));
        let ig = probe * GI_SH_N;
        if (engine) {
            let og = gi_probes[ig + SLOT_GATHER + t];
            let gat = mix(og.rgb, acc, alpha);
            gi_probes[ig + SLOT_GATHER + t] = vec4<f32>(gat, 0.0);
            var sk = gi_probes[ig + SLOT_SKY + t].rgb;
            if (do_sky) {
                sk = vec3<f32>(0.0);
                for (var i = 0u; i < nsky; i = i + 1u) { sk = sk + s_rad[i] * sh_basis(sky_dir_om(i).xyz, t); }
                gi_probes[ig + SLOT_SKY + t] = vec4<f32>(sk, 0.0);
            }
            gi_probes[ig + t] = vec4<f32>(sk + gat, 0.0);
        } else {
            let old = gi_probes[ig + t];
            gi_probes[ig + t] = vec4<f32>(mix(old.rgb, acc, alpha), old.w);
        }
    }
    // visibility: thread t owns octahedral texel t (8×8)
    {
        let tx = t % GI_VIS_RES; let ty = t / GI_VIS_RES;
        let td = gi_oct_decode((vec2<f32>(f32(tx), f32(ty)) + 0.5) / f32(GI_VIS_RES));
        var m1 = 0.0; var m2 = 0.0; var ws = 0.0;
        for (var i = 0u; i < nrays; i = i + 1u) {
            let w = pow(max(dot(td, r_dir[i]), 0.0), 12.0);
            let dd = r_dist[i];
            m1 = m1 + w * dd; m2 = m2 + w * dd * dd; ws = ws + w;
        }
        var nv = vec2<f32>(vmax, vmax * vmax);
        if (ws > 1e-5) { nv = vec2<f32>(m1 / ws, m2 / ws); }
        let vi = probe * GI_VIS_RES * GI_VIS_RES + t;
        gi_vis[vi] = mix(gi_vis[vi], nv, alpha);
    }
}

// Ray-traced sun shadow cascade (rtgi.rs `update_sun`): one ray per texel from the sun through the probe tracer's
// BVH (static map + placed pieces). Stores the distance to the first OPAQUE hit and the transmittance of the
// translucent surfaces (glass, foliage cards) in front of it. Sky shells are never hit.
@compute @workgroup_size(8, 8, 1)
fn sun_main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.z;
    let h3 = sun.hdr[4];
    if (h3.x < 0.5 || c >= 4u) { return; }
    let hc = sun.hdr[c];
    if (hc.z < 1.5) { return; }
    let res = u32(h3.z);
    if (g.x >= res || g.y >= res) { return; }
    let l = normalize(gi.sun_dir.xyz);
    let bx = gi_sun_basis_x(l);
    let by = cross(l, bx);
    let d = -l;
    // Coarse cascades (texel >= 0.25 wu) trace 2x2 rays per texel and keep the NEAREST hit: an occluder thinner than a
    // texel (rail, trunk, wall edge) anywhere in the texel marks it, which is what the Tool's per-fragment rays see.
    let nsub = select(1u, 2u, c >= 2u);
    var tsv = 1.0e9;
    var tdv = 1.0e9;
    var trn = 1.0;
    for (var sy = 0u; sy < nsub; sy = sy + 1u) {
        for (var sx = 0u; sx < nsub; sx = sx + 1u) {
            let jx = select(0.0, (f32(sx) + 0.5) / f32(nsub) - 0.5, nsub > 1u);
            let jy = select(0.0, (f32(sy) + 0.5) / f32(nsub) - 0.5, nsub > 1u);
            let fx = (f32(g.x) + 0.5 + jx) / h3.z * 2.0 - 1.0;
            let fy = (f32(g.y) + 0.5 + jy) / h3.z * 2.0 - 1.0;
            let o = bx * (hc.x + fx * hc.w) + by * (hc.y + fy * hc.w) + l * h3.w;
            var tri: u32 = 0u;
            let ts = trace_root(0u, o, d, T_MAX, &tri);
            var td = -1.0;
            if (gi.dynp.x != 0xffffffffu) { td = trace_root(gi.dynp.x, o, d, T_MAX, &tri); }
            let a = select(1.0e9, ts, ts >= 0.0);
            let b = select(1.0e9, td, td >= 0.0);
            tsv = min(tsv, a);
            tdv = min(tdv, b);
            let tm = min(a, b);
            let vis = visibility(o, d, select(T_MAX, max(tm - 0.02, 0.0), tm < 1.0e8));
            trn = min(trn, (vis.x + vis.y + vis.z) / 3.0);
        }
    }
    sun.tex[c * res * res + g.y * res + g.x] = vec4<f32>(tsv, tdv, trn, 0.0);
}
"#;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    /// wgpu's minimum binding size of every uniform / storage global of a WGSL module, by (group, binding): the
    /// struct's size, or for a struct ending in a runtime array / a bare runtime array, header + ONE element.
    pub(crate) fn min_binding_sizes(src: &str) -> std::collections::HashMap<(u32, u32), u64> {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("WGSL parse error:\n{}", e.emit_to_string(src)));
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let mut out = std::collections::HashMap::new();
        for (_, g) in module.global_variables.iter() {
            let (Some(b), true) = (&g.binding, matches!(g.space, naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. })) else { continue };
            let size = match &module.types[g.ty].inner {
                naga::TypeInner::Struct { members, span } => match members.last().map(|m| (&module.types[m.ty].inner, m.offset)) {
                    Some((naga::TypeInner::Array { size: naga::ArraySize::Dynamic, stride, .. }, off)) => (off + *stride) as u64,
                    _ => *span as u64,
                },
                naga::TypeInner::Array { size: naga::ArraySize::Dynamic, stride, .. } => *stride as u64,
                _ => layouter[g.ty].size as u64,
            };
            out.insert((b.group, b.binding), size);
        }
        out
    }

    /// **THE SIZE GATE.** Every buffer the camera bind group carries for the GI must be at least as large as the
    /// shader's declaration needs, placeholders included — wgpu validates this only at the DRAW, so a short buffer
    /// ships as a crash on the first frame that draws an object (it did: sun placeholder 72 B, shader 80 B).
    #[test]
    fn the_camera_group_buffers_are_at_least_the_size_the_shaders_declare() {
        let frag = min_binding_sizes(&format!("{}{}", GI_FRAGMENT_WGSL, GI_SHARED_WGSL.replace("__GI_DEBUG__", "0")));
        let have = [(10u32, PROBES_DUMMY_BYTES, "probes"), (11, std::mem::size_of::<GiParams>() as u64, "params"), (12, VIS_DUMMY_BYTES, "vis"), (15, SUN_DUMMY_BYTES, "sun")];
        for (binding, bytes, name) in have {
            let need = *frag.get(&(0, binding)).unwrap_or_else(|| panic!("binding {binding} ({name}) is not declared"));
            assert!(bytes >= need, "camera-group binding {binding} ({name}): buffer is {bytes} B, the shader declares {need} B");
        }
        // the real scene buffers are always larger than their placeholders (set_scene sizes them from the counts)
        assert!(SUN_HDR_BYTES + SUN_CASCADES as u64 * SUN_RES as u64 * SUN_RES as u64 * 16 >= SUN_DUMMY_BYTES);
    }

    fn check(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)));
        Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{name}: WGSL validation error: {e:?}"));
    }

    /// The probe-update compute module parses and validates (default lane and the debug lanes).
    #[test]
    fn the_probe_update_shader_is_valid() {
        for lane in [0, 1, 2, 3, 4, 5, 6, 7] {
            check(&format!("rtgi compute (cdebug {lane})"), &compute_source(lane));
        }
    }

    /// The probe-update workgroup keeps its ray results and the sky-direct terms in workgroup memory; the
    /// WebGPU default limit is 16384 bytes (`max_compute_workgroup_storage_size`) and a real adapter created
    /// with default limits refuses a module that exceeds it (the software adapter has 32 KB and would not
    /// notice). Counted with naga's own layouter.
    #[test]
    fn the_probe_update_workgroup_fits_the_default_limit() {
        let module = naga::front::wgsl::parse_str(&compute_source(0)).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let bytes: u32 = module.global_variables.iter().filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup).map(|(_, g)| layouter[g.ty].size).sum();
        eprintln!("rtgi workgroup memory: {bytes} bytes");
        assert!(bytes > 8000, "the count looks wrong ({bytes} B): is the layouter reading the workgroup arrays?");
        assert!(bytes <= 16384, "{bytes} B of workgroup memory exceeds the 16384 B default limit");
    }

    /// Rust and WGSL describe the same record: the stride, the slot map and the uniform block.
    #[test]
    fn rust_and_wgsl_agree_on_the_record_layout() {
        assert!(GI_SHARED_WGSL.contains(&format!("const GI_SH_N: u32 = {}u;", SH_N)), "GI_SH_N drifted from SH_N");
        assert!(COMPUTE_WGSL.contains(&format!("const SLOT_GATHER: u32 = {}u;", SLOT_GATHER)));
        assert!(COMPUTE_WGSL.contains(&format!("const SLOT_SKY: u32 = {}u;", SLOT_SKY)));
        assert_eq!(SLOT_RELOC, 9);
        assert!(SLOT_GATHER == 10 && SLOT_SKY == SLOT_GATHER + 9 && SH_N == SLOT_SKY + 9);
        // 14 vec4 in the uniform block (WGSL struct has the same 14 members: 11 + the inner grid's three)
        assert_eq!(std::mem::size_of::<GiParams>(), 14 * 16);
        assert_eq!(GI_FRAGMENT_WGSL.matches("inner_base: vec4<i32>").count(), 1);
        assert_eq!(COMPUTE_DECLS_WGSL.matches("inner_base: vec4<i32>").count(), 1);
        assert_eq!(GI_FRAGMENT_WGSL.matches("dynp: vec4<u32>").count(), 1);
        assert_eq!(COMPUTE_DECLS_WGSL.matches("dynp: vec4<u32>").count(), 1);
    }

    /// The edit region: exactly the probes inside the grown box, none outside, and it never leaves the grid.
    #[test]
    fn probes_in_box_selects_the_grown_box_and_nothing_else() {
        let (o, sp, d) = ([0.0, 0.0, 0.0], 2.0f32, [10u32, 10, 5]);
        let at = |i: u32| [(i % d[0]) as f32 * sp, ((i / d[0]) % d[1]) as f32 * sp, (i / (d[0] * d[1])) as f32 * sp];
        let (lo, hi, pad) = ([7.0, 7.0, 3.0], [9.0, 9.0, 5.0], 1.0);
        let got = RtGi::probes_in_box(o, sp, d, lo, hi, pad);
        assert!(!got.is_empty());
        let mut want = Vec::new();
        for i in 0..d[0] * d[1] * d[2] {
            let p = at(i);
            if (0..3).all(|a| p[a] >= lo[a] - pad - 1e-4 && p[a] <= hi[a] + pad + 1e-4) { want.push(i); }
        }
        assert_eq!(got, want);
        // a box wholly outside the grid selects nothing; one that spills over is clipped
        assert!(RtGi::probes_in_box(o, sp, d, [100.0; 3], [101.0; 3], 0.5).is_empty());
        let big = RtGi::probes_in_box(o, sp, d, [-50.0; 3], [50.0; 3], 0.0);
        assert_eq!(big.len() as u32, d[0] * d[1] * d[2]);
    }
}
