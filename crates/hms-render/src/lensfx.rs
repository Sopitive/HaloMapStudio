//! World-positioned lens flares (Reach `lens` tags attached to effects — Boneyard's
//! `street_light_glare` lamp posts, effect-scenery / object attachment flares in general).
//!
//! Engine port (reach_tag_test.exe `render_lens_flares.cpp`: sub_14085BA80 render loop,
//! sub_14085B000 reflection render data, sub_14085B7E0 projection, sub_1401BC730 transition LUTs;
//! HREK `shaders\effects\lens_flare.hlsl`):
//!   * per flare: depth = dot(pos - cam, cam_fwd); brightness factor f = 1
//!       - far fade:  if near_fade != far_fade  f  = saturate((depth - far) / (near - far))
//!       - near fade: if begin != end && end > depth  f *= saturate((depth - begin) / (end - begin))
//!       - angle:     ang = acos(dot(marker_fwd, dir_to_camera)); if ang > falloff
//!                    f *= 1 - saturate((ang - falloff) / (cutoff - falloff))
//!       - occlusion: fraction of the corona's occlusion box (radius x lerp(1, inner_scale, px/600),
//!                    pushed `occlusion offset` wu toward the viewer) that passes the depth test,
//!                    run through the `falloff function` transition LUT (0 linear, 1 sqrt, 2 x^1/4,
//!                    3 x^2, 4 x^4, 5 cosine, 6 one, 7 zero) = the "external input".
//!   * per reflection: world pos = corona + axis_offset * 2 * (cam_fwd * depth - (corona - cam))
//!     (0 = on the corona, 0.5 = screen centre, 1 = mirrored); radius = curve value (WORLD units,
//!     x depth when flag bit1 "radius scaled by distance", x saturate(occ*0.5+0.5) when bit2);
//!     screen half-extents = radius * (P00, P11) / clip.w * bitmap aspect (h/w on x for a tall
//!     bitmap, w/h on y for a wide one); rotation = rotation offset (+ screen angle when bit0).
//!   * pixel: out = modulation * pow(tex.g, tint_power) + tex * tint; brightness = curve value x
//!     f x ILLUM_EXPOSURE; additive One/One into the HDR scene.
use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::{Mat4, Vec3};
use std::sync::Arc;

/// One reflection sprite of a `lens` tag (see scene.rs `decode_lens_flare` for the tag layout).
#[derive(Clone, Debug)]
pub struct LensFlareReflection {
    /// byte flags @0: b0 rotate from centre of screen, b1 radius scaled by distance,
    /// b2 radius scaled by occlusion factor, b4 ignore external colour.
    pub flags: u8,
    pub bitmap_index: i16,
    /// radians (tag: degrees)
    pub rotation_offset: f32,
    pub axis_offset: f32,
    /// world units (the radius CURVE's value; the tag's radius bounds are never read by the engine)
    pub radius: f32,
    pub brightness: f32,
    pub modulation: f32,
    pub tint: [f32; 3],
    pub tint_power: f32,
}

/// A decoded `lens` tag + its uploaded sprite bitmap.
#[derive(Debug)]
pub struct LensFlareDef {
    pub tag: u32,
    pub falloff_angle: f32,
    pub cutoff_angle: f32,
    /// occlusion offset distance (wu) along `occl_dir` (0 toward viewer, 1 marker forward, 2 none)
    pub occl_offset: f32,
    pub occl_dir: u16,
    /// occlusion inner radius scale (fraction of the corona to test against when it is big on screen)
    pub occl_inner: f32,
    pub near_fade_begin: f32,
    pub near_fade_end: f32,
    pub near_fade: f32,
    pub far_fade: f32,
    /// lens flags @0x34: b0 rotate occlusion box with flare, b1 no occlusion test, b4 simple occlusion box
    pub flags: u16,
    /// transition function applied to the occlusion fraction (0..7, see module doc)
    pub falloff_fn: u16,
    pub bitmap: u32,
    pub tex_w: u32,
    pub tex_h: u32,
    /// true when the bitmap's colour curve is gamma-encoded (needs pow 2.2); the shipped flare
    /// sprites are Linear-curve (curve 3) and are sampled raw.
    pub gamma: bool,
    pub reflections: Vec<LensFlareReflection>,
}

/// A placed lens flare: a decoded def + its sprite texture at a world position/orientation.
#[derive(Clone)]
pub struct LensFlareInstance {
    pub def: Arc<LensFlareDef>,
    pub tex: Arc<(wgpu::TextureView, wgpu::Texture)>,
    pub pos: Vec3,
    /// marker forward (the angle falloff / occlusion offset direction)
    pub fwd: Vec3,
    /// effect / placement scale (multiplies every reflection radius)
    pub scale: f32,
    /// instance colour (the effect's tint; white for scenario placements)
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensVertex {
    centre: [f32; 2],   // reflection centre, NDC
    offset: [f32; 2],   // rotated + scaled corner offset, NDC
    uv: [f32; 2],
    color: [f32; 3],    // tint x instance colour
    brightness: f32,    // curve brightness x distance/angle fades
    modulation: f32,
    tint_power: f32,
    box_centre: [f32; 2], // corona occlusion box centre, NDC
    box_half: [f32; 2],   // corona occlusion box half-extents, NDC
    box_z: f32,           // corona (offset) NDC depth for the visibility test
    falloff_fn: f32,      // transition LUT index (0..7); 8 = no occlusion test
    occl_scale: f32,      // 1 = radius scaled by the occlusion factor (reflection flag b2)
    gamma: f32,           // 1 = bitmap needs pow 2.2
}
const LENS_VERTEX_FLOATS: usize = 20;

const WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32>, inv_view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
// same layout as mesh.rs SHADOW_SAMPLE_WGSL `Light` (binding 2) + the 1x1 luminance meter (binding 9)
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;
@group(0) @binding(5) var scene_depth: texture_depth_2d;
@group(0) @binding(9) var lum_meter: texture_2d<f32>;
@group(1) @binding(0) var flare_tex: texture_2d<f32>;
@group(1) @binding(1) var flare_samp: sampler;

// Engine ILLUM_EXPOSURE (g_alt_exposure.r) — verbatim copy of mesh.rs illum_scale_now().
fn illum_scale_now() -> f32 {
    let mean_log = textureLoad(lum_meter, vec2<i32>(0, 0), 0).r;
    let hi = light.expo.w;
    let lo = min(light.expo.z, hi);
    let raw_gain = light.expo.y / max(exp2(mean_log), 1e-4);
    var gain = light.expo.x * clamp(raw_gain * light.expo2.x, lo, hi);
    if (light.expo2.y > 1e-6) { gain = light.expo2.y; }
    // E = stops around the key (see mesh.rs illum_scale_now).
    let e = log2(max(gain, 1e-6) / max(light.expo.y * 10.0, 1e-4));
    return exp2((1.0 - light.expo2.w) * (light.expo2.z - e));
}

// Engine transition functions (reach_tag_test 1024-entry byte LUTs at 0x141E3CB30, verified at x=0.5/0.75).
fn transition(fn_idx: f32, x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    let k = i32(fn_idx + 0.5);
    if (k == 1) { return sqrt(t); }
    if (k == 2) { return pow(t, 0.25); }
    if (k == 3) { return t * t; }
    if (k == 4) { return t * t * t * t; }
    if (k == 5) { return 0.5 - 0.5 * cos(t * 3.14159265); }
    if (k == 6) { return 1.0; }
    if (k == 7) { return 0.0; }
    return t;
}

// Occlusion query stand-in: fraction of an 8x8 grid over the corona's screen box whose opaque
// scene depth is not in front of the (viewer-offset) corona. Off-screen taps are not counted
// (the engine clamps the box to the viewport and counts only rendered pixels).
fn occlusion(box_centre: vec2<f32>, box_half: vec2<f32>, z: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    var vis = 0.0;
    var tot = 0.0;
    for (var j = 0; j < 8; j = j + 1) {
        for (var i = 0; i < 8; i = i + 1) {
            let f = (vec2<f32>(f32(i), f32(j)) + 0.5) / 8.0 * 2.0 - 1.0;
            let ndc = box_centre + f * box_half;
            if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) { continue; }
            let px = vec2<i32>((vec2<f32>(ndc.x, -ndc.y) * 0.5 + 0.5) * dims);
            let d = textureLoad(scene_depth, clamp(px, vec2<i32>(0), vec2<i32>(dims) - 1), 0);
            tot = tot + 1.0;
            if (d >= z * (1.0 - 2e-5)) { vis = vis + 1.0; }
        }
    }
    if (tot < 0.5) { return 0.0; }
    return vis / tot;
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) col: vec3<f32>,
    @location(2) @interpolate(flat) params: vec4<f32>, // brightness*occ, modulation, tint_power, gamma
};

@vertex
fn vs(
    @location(0) centre: vec2<f32>, @location(1) offset: vec2<f32>, @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>, @location(4) brightness: f32, @location(5) modulation: f32,
    @location(6) tint_power: f32, @location(7) box_centre: vec2<f32>, @location(8) box_half: vec2<f32>,
    @location(9) box_z: f32, @location(10) falloff_fn: f32, @location(11) occl_scale: f32, @location(12) gamma: f32,
) -> VOut {
    var o: VOut;
    var occ = 1.0;
    if (falloff_fn < 7.5) { occ = transition(falloff_fn, occlusion(box_centre, box_half, box_z)); }
    var s = 1.0;
    if (occl_scale > 0.5) { s = clamp(occ * 0.5 + 0.5, 0.0, 1.0); }
    var p = centre + offset * s;
    if (occ <= 1e-4) { p = vec2<f32>(4.0, 4.0); } // engine: falloff(occlusion) <= 1e-4 -> not drawn
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.uv = uv;
    o.col = color;
    o.params = vec4<f32>(brightness * occ, modulation, tint_power, gamma);
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    var tex = textureSample(flare_tex, flare_samp, i.uv);
    if (i.params.w > 0.5) { tex = vec4<f32>(pow(tex.rgb, vec3<f32>(2.2)), tex.a); }
    // HREK lens_flare.hlsl default_ps
    let nth = pow(max(tex.g, 0.00001), i.params.z);
    let out_color = i.params.y * vec3<f32>(nth) + tex.rgb * i.col;
    let brightness = i.params.x * illum_scale_now();
    return vec4<f32>(out_color * brightness, 1.0);
}
"#;

/// GPU state for the world lens-flare pass.
pub struct LensFx {
    pipeline: wgpu::RenderPipeline,
    bgl0: wgpu::BindGroupLayout,
    bg0: wgpu::BindGroup,
    tex_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub instances: Vec<LensFlareInstance>,
}

/// One frame's draw list: (vertex buffer, texture bind group, vertex count).
pub struct LensFxBatch {
    vbuf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    count: u32,
}

impl LensFx {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        light_buf: &wgpu::Buffer,
        depth_copy_view: &wgpu::TextureView,
        lum_view: &wgpu::TextureView,
        hdr_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lensfx-bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // the opaque scene-depth copy, read in the VERTEX stage (the occlusion box query)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lensfx-tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lensfx-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lensfx"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lensfx-pl"),
            bind_group_layouts: &[&bgl0, &tex_bgl],
            push_constant_ranges: &[],
        });
        let attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 36, shader_location: 4 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 40, shader_location: 5 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 44, shader_location: 6 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 48, shader_location: 7 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 56, shader_location: 8 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 64, shader_location: 9 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 68, shader_location: 10 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 72, shader_location: 11 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 76, shader_location: 12 },
        ];
        let vbl = wgpu::VertexBufferLayout {
            array_stride: (LENS_VERTEX_FLOATS * 4) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attrs,
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lensfx-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[vbl], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            // no depth test: visibility is the occlusion-box query (the engine draws flares over everything)
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(additive), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            multiview: None,
            cache: None,
        });
        let bg0 = Self::make_bg0(device, &bgl0, camera_buf, light_buf, depth_copy_view, lum_view);
        Self { pipeline, bgl0, bg0, tex_bgl, sampler, instances: Vec::new() }
    }

    fn make_bg0(device: &wgpu::Device, bgl0: &wgpu::BindGroupLayout, camera_buf: &wgpu::Buffer, light_buf: &wgpu::Buffer, depth_copy_view: &wgpu::TextureView, lum_view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lensfx-bg0"),
            layout: bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth_copy_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(lum_view) },
            ],
        })
    }

    /// Resize: the depth copy view changed.
    pub fn rebind(&mut self, device: &wgpu::Device, camera_buf: &wgpu::Buffer, light_buf: &wgpu::Buffer, depth_copy_view: &wgpu::TextureView, lum_view: &wgpu::TextureView) {
        self.bg0 = Self::make_bg0(device, &self.bgl0, camera_buf, light_buf, depth_copy_view, lum_view);
    }

    /// Per-frame CPU side of the engine's render loop (projection, fades, reflection placement);
    /// the occlusion query runs in the vertex shader. `view`/`proj` are the camera matrices,
    /// `size` the viewport in pixels. Returns one batch per distinct sprite texture.
    pub fn build(&self, device: &wgpu::Device, cam_pos: Vec3, cam_fwd: Vec3, view: &Mat4, proj: &Mat4, size: (u32, u32), diag: bool) -> Vec<LensFxBatch> {
        use wgpu::util::DeviceExt;
        if self.instances.is_empty() { return Vec::new(); }
        let vp = *proj * *view;
        let p00 = proj.x_axis.x;
        let p11 = proj.y_axis.y;
        let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
        // batches keyed by texture identity (Arc pointer)
        let mut batches: Vec<(Arc<(wgpu::TextureView, wgpu::Texture)>, Vec<f32>)> = Vec::new();
        let sat = |x: f32| x.clamp(0.0, 1.0);
        let no_occ = std::env::var("HMS_LENSFX_NO_OCC").is_ok();
        // HMS_LENSFX_GAIN=<f>: diag brightness multiplier (default 1) to inspect faint sprites' shape/placement.
        let gain: f32 = std::env::var("HMS_LENSFX_GAIN").ok().and_then(|v| v.parse().ok()).filter(|g: &f32| g.is_finite() && *g > 0.0).unwrap_or(1.0);
        for inst in &self.instances {
            let def = &*inst.def;
            let Some(corona) = def.reflections.first() else { continue };
            let delta = inst.pos - cam_pos;
            let depth = delta.dot(cam_fwd);
            if !(depth > 0.001) { continue; }
            let to_cam = -delta.normalize_or_zero();
            let mut f = 1.0f32;
            if def.near_fade != def.far_fade { f = sat((depth - def.far_fade) / (def.near_fade - def.far_fade)); }
            if def.near_fade_begin != def.near_fade_end && def.near_fade_end > depth {
                f *= sat((depth - def.near_fade_begin) / (def.near_fade_end - def.near_fade_begin));
            }
            if def.falloff_angle != def.cutoff_angle {
                let ang = inst.fwd.dot(to_cam).clamp(-1.0, 1.0).acos();
                if ang > def.falloff_angle { f *= 1.0 - sat((ang - def.falloff_angle) / (def.cutoff_angle - def.falloff_angle)); }
            }
            if f <= 0.0 { continue; }
            // bitmap aspect (engine: integer h/w on x for a tall sprite, w/h on y for a wide one)
            let (mut ax, mut ay) = (1.0f32, 1.0f32);
            if def.tex_w > 0 && def.tex_h > 0 {
                if def.tex_h > def.tex_w { ax = (def.tex_h / def.tex_w) as f32; } else if def.tex_w > def.tex_h { ay = (def.tex_w / def.tex_h) as f32; }
            }
            // corona projection + occlusion box (offset toward the viewer / along the marker forward)
            let occ_dir = match def.occl_dir { 0 => to_cam, 1 => inst.fwd, _ => Vec3::ZERO };
            let occ_pos = inst.pos + occ_dir * def.occl_offset;
            let cc = vp * inst.pos.extend(1.0);
            let co = vp * occ_pos.extend(1.0);
            if cc.w <= 0.0 || co.w <= 0.0 { continue; }
            let corona_ndc = [cc.x / cc.w, cc.y / cc.w];
            let box_centre = [co.x / co.w, co.y / co.w];
            let box_z = co.z / co.w;
            let mut corona_r = corona.radius * inst.scale;
            if corona.flags & 2 != 0 { corona_r *= depth; }
            // per-unit NDC scale at the corona depth
            let (sx, sy) = (p00 / cc.w * ax, p11 / cc.w * ay);
            let px_w = sx * 0.5 * w;
            let px_h = sy * 0.5 * h;
            let t = sat((2.0 * px_w + 2.0 * px_h) / 600.0);
            let inner = if def.flags & 0x10 != 0 { def.occl_inner } else { (1.0 - t) * (1.0 - def.occl_inner) + def.occl_inner };
            let box_r = corona_r * inner;
            let box_half = [(sx * box_r).min(1.0), (sy * box_r).min(1.0)];
            // HMS_LENSFX_NO_OCC=1 bypasses the occlusion query (tells a self-occluded corona from a
            // placement/fade bug).
            let falloff_fn = if def.flags & 0x2 != 0 || no_occ { 8.0 } else { def.falloff_fn.min(7) as f32 };
            let screen_angle = corona_ndc[1].atan2(corona_ndc[0]);
            let mirror = cam_fwd * depth - delta; // reflection axis: rpos = pos + axis_offset * 2 * mirror
            let bi = match batches.iter().position(|(t, _)| Arc::ptr_eq(t, &inst.tex)) {
                Some(i) => i,
                None => { batches.push((inst.tex.clone(), Vec::new())); batches.len() - 1 }
            };
            for r in &def.reflections {
                let rpos = inst.pos + mirror * (2.0 * r.axis_offset);
                let cr = vp * rpos.extend(1.0);
                if cr.w <= 0.0 { continue; }
                let centre = [cr.x / cr.w, cr.y / cr.w];
                let mut radius = r.radius * inst.scale;
                if r.flags & 2 != 0 { radius *= depth; }
                let bright = r.brightness * f * gain;
                if radius <= 0.0 || bright <= 0.0 { continue; }
                let mut rot = r.rotation_offset;
                if r.flags & 1 != 0 { rot += screen_angle; }
                let (hx, hy) = (p00 / cr.w * ax * radius, p11 / cr.w * ay * radius);
                let col = if r.flags & 0x10 != 0 { r.tint } else { [r.tint[0] * inst.color[0], r.tint[1] * inst.color[1], r.tint[2] * inst.color[2]] };
                let (s, c) = rot.sin_cos();
                let corner = |x: f32, y: f32| -> [f32; LENS_VERTEX_FLOATS] {
                    // HREK lens_flare.hlsl: rotate the unit corner, then the anisotropic screen scale
                    let rx = x * c - y * s;
                    let ry = x * s + y * c;
                    [
                        centre[0], centre[1], rx * hx, ry * hy, x * 0.5 + 0.5, y * 0.5 + 0.5,
                        col[0], col[1], col[2], bright, r.modulation, r.tint_power,
                        box_centre[0], box_centre[1], box_half[0], box_half[1], box_z, falloff_fn,
                        if r.flags & 4 != 0 { 1.0 } else { 0.0 }, if def.gamma { 1.0 } else { 0.0 },
                    ]
                };
                let v = &mut batches[bi].1;
                for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    v.extend_from_slice(&corner(x, y));
                }
                if diag {
                    eprintln!("LENSFX lens {:#x} at ({:.2},{:.2},{:.2}) depth={depth:.2} fade={f:.3} -> ndc=({:.3},{:.3}) half_ndc=({:.4},{:.4}) radius_wu={radius:.3} bright={bright:.3} box=({:.3},{:.3})+-({:.4},{:.4}) z={box_z:.6} fn={falloff_fn}",
                        def.tag, inst.pos.x, inst.pos.y, inst.pos.z, centre[0], centre[1], hx, hy, box_centre[0], box_centre[1], box_half[0], box_half[1]);
                }
            }
        }
        batches.into_iter().filter(|(_, v)| !v.is_empty()).map(|(tex, verts)| {
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lensfx-vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lensfx-tex-bg"),
                layout: &self.tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex.0) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            LensFxBatch { vbuf, bg, count: (verts.len() / LENS_VERTEX_FLOATS) as u32 }
        }).collect()
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, batches: &'a [LensFxBatch]) {
        if batches.is_empty() { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg0, &[]);
        for b in batches {
            pass.set_bind_group(1, &b.bg, &[]);
            pass.set_vertex_buffer(0, b.vbuf.slice(..));
            pass.draw(0..b.count, 0..1);
        }
    }
}

// keep the vertex struct honest about its float count
const _: () = assert!(std::mem::size_of::<LensVertex>() == LENS_VERTEX_FLOATS * 4);
