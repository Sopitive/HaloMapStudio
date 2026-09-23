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

use eframe::wgpu;
use wgpu::util::DeviceExt;

pub use crate::lightbake_gpu::{GpuEmitter, GpuLight};

/// Octahedral visibility map resolution per probe (texels per side).
pub const VIS_RES: u32 = 8;
pub const VIS_TEXELS: u32 = VIS_RES * VIS_RES;
/// vec4 slots per probe: 9 SH coefficients (L2, rgb) + slot 9 = relocation offset xyz, w = VALID flag (0 = inside geometry).
pub const SH_N: u32 = 10;
/// Hard cap on the probe count (the update dispatch is one workgroup per probe).
pub const MAX_PROBES: u32 = 32_768;

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
}

struct SceneBufs {
    _nodes: wgpu::Buffer,
    _tris: wgpu::Buffer,
    _tri_mat: wgpu::Buffer,
    _mats: wgpu::Buffer,
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
}

impl RtGi {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtgi-probe-update"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMPUTE_DECLS_WGSL}{GI_SHARED_WGSL}{COMPUTE_WGSL}").replace("__GI_DEBUG__", "0").replace("__GI_CDEBUG__", &std::env::var("HMS_RTGI_CDEBUG").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0).to_string()).into()),
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
        let params = GiParams::default();
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rtgi-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Dummy (one-probe) records so the camera bind group is valid before a scene is set.
        let probes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-probes"),
            size: (SH_N as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rtgi-vis"),
            size: (VIS_TEXELS as u64) * 8,
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
        }
    }

    pub fn probes_buf(&self) -> &wgpu::Buffer { &self.probes_buf }
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
        let mut cube_norm = 1.0f32;
        if let Some((tex, mips)) = env_tex {
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
        let max_probes: u32 = std::env::var("HMS_RTGI_MAX_PROBES").ok().and_then(|v| v.parse().ok()).unwrap_or(16_384u32).clamp(64, MAX_PROBES);
        let min_spacing: f32 = std::env::var("HMS_RTGI_MIN_SPACING").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32).max(0.25);
        let (origin, spacing, dims) = Self::plan_grid(s.bounds_min, s.bounds_max, max_probes, min_spacing);
        let nprobes = dims[0] * dims[1] * dims[2];
        self.nprobes = nprobes.max(1);
        self.spacing = spacing;
        self.dims = dims;
        let mk = |label: &str, bytes: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytes, usage: wgpu::BufferUsages::STORAGE,
        });
        let nodes = mk("rtgi-nodes", &s.nodes);
        let tris = mk("rtgi-tris", &s.tris);
        let mut tm = s.tri_mat.clone(); if tm.is_empty() { tm.push(0); }
        let tri_mat = mk("rtgi-tri-mat", bytemuck::cast_slice(&tm));
        let mut mats = s.mats.clone(); if mats.len() < 2 { mats = vec![[0.5, 0.5, 0.5, 0.0], [0.0; 4]]; }
        let mats_b = mk("rtgi-mats", bytemuck::cast_slice(&mats));
        let mut lights = s.lights.clone(); if lights.is_empty() { lights.push(GpuLight::default()); }
        let lights_b = mk("rtgi-lights", bytemuck::cast_slice(&lights));
        let mut emitters = s.emitters.clone(); if emitters.is_empty() { emitters.push(GpuEmitter::default()); }
        let emit_b = mk("rtgi-emitters", bytemuck::cast_slice(&emitters));
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
        let energy: f32 = std::env::var("HMS_RTGI_GAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.3);
        let ground: f32 = std::env::var("HMS_RTGI_GROUND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
        // sky radiance source: the captured sky env cube (the map's own sky model, HDR) unless disabled
        let use_cube = env.is_some() && std::env::var("HMS_RTGI_NO_CUBE").is_err();
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
            fog: [s.fog_sigma, s.fog_top, s.total_emitter_weight, 1.0],
            bias: [0.2, 0.3, 1.5, energy],
            nee: [std::env::var("HMS_RTGI_NEE").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f32).clamp(0.0, 64.0),
                  std::env::var("HMS_RTGI_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0f32).max(0.1),
                  std::env::var("HMS_RTGI_NEE_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05f32).max(0.0), 0.0],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        self.scene = Some(SceneBufs { _nodes: nodes, _tris: tris, _tri_mat: tri_mat, _mats: mats_b, _lights: lights_b, _emitters: emit_b, bg });
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
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn set_gain(&mut self, queue: &wgpu::Queue, gain: f32) {
        self.params.bias[3] = gain.clamp(0.01, 16.0);
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }
    pub fn clear_scene(&mut self, queue: &wgpu::Queue) {
        self.scene = None;
        self.enabled = false;
        self.params.dims[3] = 0;
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }
    /// Override the sun (direction to sun + HDR colour) for the tracer — the Lighting Lab sun controls.
    pub fn set_sun(&mut self, queue: &wgpu::Queue, dir: [f32; 3], color: [f32; 3]) {
        self.params.sun_dir = [dir[0], dir[1], dir[2], self.params.sun_dir[3]];
        self.params.sun_color = [color[0], color[1], color[2], self.params.sun_color[3]];
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        // The probe field still holds the previous sun and the temporal blend would take ~30 frames
        // to wash it out; kick every slice once at full weight so a dragged sun slider follows immediately.
        self.kick.set(self.parts_per_frame() + 1);
    }
    /// Force the next slices to REPLACE rather than blend (after any lighting change).
    pub fn kick(&self) { self.kick.set(self.parts_per_frame() + 1); }

    /// Record one probe-update pass into `encoder`. `blend` overrides the temporal alpha (None =
    /// 1 − hysteresis, or 1.0 on the very first update so the records start from a full estimate).
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, blend: Option<f32>) {
        self.dispatch_part(encoder, queue, blend, 1, 0);
    }
    /// Update slice `part` of `parts` (round-robin partial updates keep the per-frame cost bounded).
    pub fn dispatch_part(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, blend: Option<f32>, parts: u32, part: u32) {
        let Some(sc) = self.scene.as_ref() else { return };
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
    misc: vec4<u32>, misc2: vec4<u32>, fog: vec4<f32>, bias: vec4<f32>, nee: vec4<f32>,
};
@group(0) @binding(10) var<storage, read> gi_probes: array<vec4<f32>>;
@group(0) @binding(11) var<uniform> gi: GiParams;
@group(0) @binding(12) var<storage, read> gi_vis: array<vec2<f32>>;
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
    misc: vec4<u32>, misc2: vec4<u32>, fog: vec4<f32>, bias: vec4<f32>, nee: vec4<f32>,
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
"#;

/// The probe LOOKUP shared by the compute (previous-frame irradiance at ray hits) and fragment
/// (surface shading) sides. Both declare `gi`, `gi_probes`, `gi_vis` before this text.
pub const GI_SHARED_WGSL: &str = r#"
const GI_PI: f32 = 3.14159265;
const GI_VIS_RES: u32 = 8u;
const GI_SH_N: u32 = 10u;

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
fn gi_probe_count() -> u32 { return gi.dims.x * gi.dims.y * gi.dims.z; }
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
// irradiance E(n) from the probe's L2 SH (Ramamoorthi-Hanrahan cosine convolution)
fn gi_sh_irradiance(pi: u32, n: vec3<f32>) -> vec3<f32> {
    let b = pi * GI_SH_N;
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
fn gi_on() -> bool { return gi.dims.w > 0u; }
fn gdbg_ignore_valid() -> bool { return __GI_DEBUG__ == 3; }
// Irradiance / π at surface point p with normal n (v = unit vector from p toward the viewer/probe
// side, used for the DDGI bias), from the 8 surrounding probes with backface + Chebyshev
// visibility weights. Same units as the baked `bake.rgb`.
fn gi_irradiance(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
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
        sum = sum + w * gi_sh_irradiance(pi, n);
        wsum = wsum + w;
    }
    let gdbg = __GI_DEBUG__;
    if (gdbg == 1) { return vec3<f32>(f32(nvalid) / 8.0, f32(nin) / 8.0, select(0.0, 1.0, wsum > 1e-5)); }
    if (gdbg == 2) { let gg = (p - o) / sp; return fract(gg * 0.25); }
    if (wsum <= 1e-5) { return vec3<f32>(0.0); }
    return (sum / wsum) * (gi.bias.w / GI_PI);
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
// Light transmitted along o+d·[0,tmax]: 0 behind an opaque surface, the product of the transmittances
// of any translucent surfaces crossed (glass roofs / panes), 1 when clear.
fn visibility(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> vec3<f32> {
    let inv = 1.0 / d;
    var stack: array<u32, 48>;
    var sp = 0;
    stack[0] = 0u; sp = 1;
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
fn occluded(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> bool { let v = visibility(o, d, tmax); return max(v.r, max(v.g, v.b)) <= 0.0; }
fn trace(o: vec3<f32>, d: vec3<f32>, out_tri: ptr<function, u32>) -> f32 {
    let inv = 1.0 / d;
    var stack: array<u32, 48>;
    var sp = 0;
    stack[0] = 0u; sp = 1;
    var best = T_MAX;
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
// direct irradiance at a surface point: sun + scenario point/spot lights + one emitter NEE sample
fn direct_irradiance(x: vec3<f32>, n: vec3<f32>, s: ptr<function, u32>) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    let sun_dir = normalize(gi.sun_dir.xyz);
    let ndl = dot(n, sun_dir);
    if (ndl > 0.0 && lum(gi.sun_color.xyz) > 0.0) {
        let vs = visibility(x, sun_dir, T_MAX);
        if (max(vs.r, max(vs.g, vs.b)) > 0.0) { e = e + gi.sun_color.xyz * vs * ndl * fog_transmittance(x, sun_dir, T_MAX); }
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
var<workgroup> r_dir: array<vec3<f32>, 256>;
var<workgroup> r_rad: array<vec3<f32>, 256>;
var<workgroup> r_dist: array<f32, 256>;
var<workgroup> backfaces: atomic<u32>;
var<workgroup> cb_dist: array<f32, 64>;
var<workgroup> cb_dir: array<vec3<f32>, 64>;
var<workgroup> cf_dist: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let probe = wid.x + gi.misc2.w;
    let nprobes = gi_probe_count();
    if (probe >= nprobes) { return; }
    let t = lid.x;
    if (t == 0u) { atomicStore(&backfaces, 0u); }
    workgroupBarrier();
    let rpt = clamp(gi.misc.y, 1u, 4u);
    let nrays = WG * rpt;
    let nx = gi.dims.x; let ny = gi.dims.y;
    let c = vec3<u32>(probe % nx, (probe / nx) % ny, probe / (nx * ny));
    let rec_old = gi_probes[probe * GI_SH_N + 9u];
    let ppos = gi_probe_pos(c) + rec_old.xyz;
    // uninitialised probe (vis record still zero) → take the full estimate this pass
    let fresh = gi_vis[probe * GI_VIS_RES * GI_VIS_RES].y <= 0.0;
    // per-frame random rotation of the fibonacci set (Shoemake uniform quaternion)
    var rs: u32 = gi.misc.x * 7919u + 17u;
    let u1 = randf(&rs); let u2 = randf(&rs); let u3 = randf(&rs);
    let q = vec4<f32>(sqrt(1.0 - u1) * sin(6.2831853 * u2), sqrt(1.0 - u1) * cos(6.2831853 * u2), sqrt(u1) * sin(6.2831853 * u3), sqrt(u1) * cos(6.2831853 * u3));
    var rng: u32 = probe * 9781u + gi.misc.x * 6271u + t * 131u + 1u;
    let vmax = gi.origin.w * gi.bias.z;
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
            // escaped: the captured sky env cube (the map's own sky model, sun glare included) when
            // available, else the flat sky ambient; a dimmer ground/ocean stand-in below the horizon.
            var sky = gi.sky_color.xyz * select(gi.sky_color.w, 1.0, d.z >= 0.0);
            if (gi.sun_dir.w > 0.5) {
                // cube LUMINANCE (angular distribution, sun glare) × the authored sky-light HUE
                let cl = lum(textureSampleLevel(env_cube, env_samp, d, 1.0).rgb) * gi.sun_color.w;
                let hue = gi.sky_color.xyz / max(lum(gi.sky_color.xyz), 1e-4);
                let cs = cl * hue;
                sky = select(max(cs, gi.sky_color.xyz * gi.sky_color.w * 0.1), cs, d.z >= 0.0);
            }
            rad = sky * visibility(ppos, d, 1.0e4) * fog_transmittance(ppos, d, 1.0e4);
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
                let e_prev = gi_irradiance(ho, ng, -d) * GI_PI; // previous frame, back to irradiance units
                // outgoing radiance of the hit = Lambert bounce + the surface's own self-illum
                rad = (m.rgb * (e_direct + e_prev) / GI_PI + em) * fog_transmittance(ppos, d, th);
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
        let sp = gi.origin.w;
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
    // SH projection: threads 0..8 each own one coefficient (all rays, all channels)
    if (t < GI_SH_N) {
        var acc = vec3<f32>(0.0);
        for (var i = 0u; i < nrays; i = i + 1u) {
            acc = acc + r_rad[i] * sh_basis(r_dir[i], t);
        }
        acc = acc * (4.0 * GI_PI / f32(nrays));
        let idx = probe * GI_SH_N + t;
        let old = gi_probes[idx];
        let blended = mix(old.rgb, acc, alpha);
        gi_probes[idx] = vec4<f32>(blended, old.w);
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
"#;
