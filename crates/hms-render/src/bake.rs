//! GPU base×detail texture bake: composites a material's base colour map with its
//! detail layers on the GPU (sampling the already-uploaded views) instead of
//! re-decoding and multiplying on the CPU, which was the single largest load-time cost.
//!
//! The composite is the engine's albedo law (calc_albedo_default_ps / two_detail / ...):
//! base × Π(detail_layer × 4.59479) in linear space. Textures are *Unorm (raw sRGB bytes;
//! the mesh shader linearises with pow 2.2), so in this encoded byte space the detail
//! multiplier is 2.03 = 4.59479^(1/2.2):
//!   composite = base * Π_layers( detail_layer * 2.03 )
//!   composite = max(composite, base * 30/256)         // black-spot floor
//!   if (tint_active) composite *= tint^(1/2.2)         // albedo_color, pre-desaturated on the CPU
//!   out.a = base.a                                     // base alpha preserved
//! The mean-preserving variant (`detail / layer_mean`, default-albedo materials only,
//! HMS_NO_LIT) divides each channel by the layer's 1×1 mip so the detail's colour bias is
//! removed. Absent or "flatten" (self-detail) layers contribute 1.0. Output is Bgra8Unorm
//! with a full mip chain, the same format the CPU upload produced, so the downstream draw
//! path (bake_cache bind → mesh_shade → pow 2.2) is unchanged.

use eframe::wgpu;

pub struct BakeLayer<'a> {
    pub view: &'a wgpu::TextureView,
    pub ratio: [f32; 2],
    pub flatten: bool,
}

// Uniform matching the WGSL `BakeU` (5×vec4, std140-friendly 16B rows).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakeU {
    r0: [f32; 4], // [ratio.x, ratio.y, present, flatten] layer 0
    r1: [f32; 4], // layer 1
    r2: [f32; 4], // layer 2
    tint: [f32; 4], // [r, g, b, active]
    _pad: [f32; 4], // [base_lod, literal, 0, 0]
}

pub struct DetailBaker {
    composite_pl: wgpu::RenderPipeline,
    mip_pl: wgpu::RenderPipeline,
    comp_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
    samp: wgpu::Sampler,
    grey: wgpu::TextureView, // 1×1 neutral fallback for absent detail layers
}

const BAKE_WGSL: &str = r#"
struct BakeU {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    tint: vec4<f32>,
    pad: vec4<f32>,
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var base_t: texture_2d<f32>;
@group(0) @binding(2) var d0: texture_2d<f32>;
@group(0) @binding(3) var d1: texture_2d<f32>;
@group(0) @binding(4) var d2: texture_2d<f32>;
@group(0) @binding(5) var<uniform> u: BakeU;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    // Fullscreen triangle.
    var o: VOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.uv = vec2<f32>(x, y);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

fn layer(t: texture_2d<f32>, uv: vec2<f32>, r: vec4<f32>, literal: f32) -> vec3<f32> {
    // present (r.z) off OR flatten (r.w) on → contributes 1.0 (a flattened layer's texel ==
    // its mean → detail/mean == 1). Self-detail (flatten) stays 1.0 in literal mode too:
    // baking base×base literally over-brightens (wood).
    if (r.z < 0.5 || r.w > 0.5) { return vec3<f32>(1.0, 1.0, 1.0); }
    let d = textureSampleLevel(t, samp, fract(uv * r.xy), 0.0).rgb;       // tiled detail texel
    // Literal mode (literal>0.5): the engine's base×detail×DETAIL_MULT — 2.03 = 4.59^(1/2.2) in
    // sRGB byte space so the shader's pow(2.2) reconstructs ×4.59 linear. Two-detail nets to
    // base×d1×d2×4.59² (linear), the darkening/contrast the mean-preserving path removes.
    if (literal > 0.5) { return d * 2.03; }
    let m = textureSampleLevel(t, samp, vec2<f32>(0.5, 0.5), 30.0).rgb;   // 1×1 mip = full-tex mean
    return d / max(m, vec3<f32>(1.0 / 255.0));
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let base = textureSampleLevel(base_t, samp, i.uv, u.pad.x);   // raw sRGB + alpha (pad.x = base_lod)
    let lit = u.pad.y; // >0.5 → literal multi-detail composite (multi-detail materials)
    var prod = vec3<f32>(1.0, 1.0, 1.0);
    prod = prod * layer(d0, i.uv, u.r0, lit);
    prod = prod * layer(d1, i.uv, u.r1, lit);
    prod = prod * layer(d2, i.uv, u.r2, lit);
    var comp = base.rgb * prod;
    comp = max(comp, base.rgb * (30.0 / 256.0));             // black-spot floor
    // albedo_color (engine: albedo = base × detail × albedo_color, always). `comp` is in
    // sRGB-ENCODED space (the base is sampled raw; the literal detail multiplier is 2.03 =
    // 4.59^(1/2.2) for that reason) while albedo_color is a LINEAR multiplier, so the tint is
    // encoded (tint^(1/2.2)) and the mesh shader's pow(2.2) decode yields exactly one linear
    // multiply. Multiplying the linear tint into encoded texels would decode to tint^2.2 —
    // measured on Zealot cov_metal_strip_a (albedo ON/OFF = tint^2.21 on all three channels),
    // which rendered the purple Covenant walls (tint .54/.29/.79) 4x too dark and grey.
    if (u.tint.w > 0.5) {
        comp = comp * pow(max(u.tint.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(comp, base.a);
}
"#;

const MIP_WGSL: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.uv = vec2<f32>(x, y);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, i.uv, 0.0);   // bilinear box downsample
}
"#;

impl DetailBaker {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bake-composite"),
            source: wgpu::ShaderSource::Wgsl(BAKE_WGSL.into()),
        });
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bake-mip"),
            source: wgpu::ShaderSource::Wgsl(MIP_WGSL.into()),
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
        let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake-comp-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake-mip-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(1),
            ],
        });
        let target = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };
        let make_pl = |label: &str, shader: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs"),
                    targets: &[Some(target.clone())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let composite_pl = make_pl("bake-composite-pl", &comp_shader, &comp_bgl);
        let mip_pl = make_pl("bake-mip-pl", &mip_shader, &mip_bgl);
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bake-samp"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            ..Default::default()
        });
        // 1×1 neutral fallback for absent detail layers (never sampled meaningfully — the
        // present flag gates it — but a valid view must be bound).
        let grey_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bake-grey"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &grey_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[128u8, 128, 128, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let grey = grey_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { composite_pl, mip_pl, comp_bgl, mip_bgl, samp, grey }
    }

    /// Render the composite (+ black-spot floor + optional tint) into a fresh Bgra8Unorm
    /// texture at base resolution with a full mip chain. `layers` holds up to 3 detail
    /// layers (extra ignored); `tint` is the (CPU-pre-desaturated) albedo_color and
    /// `tint_nonidentity` gates it. Returns the same (view, texture) shape
    /// `upload_texture_bgra` returns, for bake_cache.
    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base: &wgpu::TextureView,
        w: u32,
        h: u32,
        layers: &[BakeLayer],
        tint: [f32; 3],
        tint_nonidentity: bool,
        literal: bool, // literal base×detail×DM (multi-detail) vs mean-preserving
    ) -> (wgpu::TextureView, wgpu::Texture) {
        use wgpu::util::DeviceExt;
        // Composite base mip 1 when max(w,h) > 1024, else mip 0: sample the base at `base_lod`
        // and render the target at the reduced size so large bakes stay within the memory budget.
        let base_lod: u32 = if w.max(h) > 1024 { 1 } else { 0 };
        let tw = (w >> base_lod).max(1);
        let th = (h >> base_lod).max(1);
        let mip_level_count = 32 - tw.max(th).max(1).leading_zeros(); // floor(log2(max))+1
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bake-target"),
            size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Memory accounting: base level + ~1/3 for the mip chain (matches the upload path).
        crate::mesh::memprof::add_tex_uncompressed((w as u64) * (h as u64) * 4 * 4 / 3);

        let lp = |i: usize| -> ([f32; 4], &wgpu::TextureView) {
            match layers.get(i) {
                Some(l) => (
                    [l.ratio[0], l.ratio[1], 1.0, if l.flatten { 1.0 } else { 0.0 }],
                    l.view,
                ),
                None => ([0.0, 0.0, 0.0, 0.0], &self.grey),
            }
        };
        let (r0, v0) = lp(0);
        let (r1, v1) = lp(1);
        let (r2, v2) = lp(2);
        let u = BakeU {
            r0,
            r1,
            r2,
            tint: [tint[0], tint[1], tint[2], if tint_nonidentity { 1.0 } else { 0.0 }],
            _pad: [base_lod as f32, if literal { 1.0 } else { 0.0 }, 0.0, 0.0],
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake-u"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let comp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake-comp-bg"),
            layout: &self.comp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&self.samp) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(v0) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(v1) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(v2) },
                wgpu::BindGroupEntry { binding: 5, resource: ubuf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake-enc") });
        // Mip 0 = the composite.
        let mip0 = tex.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        {
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bake-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mip0,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.composite_pl);
            p.set_bind_group(0, &comp_bg, &[]);
            p.draw(0..3, 0..1);
        }
        // Remaining mips: box-downsample the previous level.
        for level in 1..mip_level_count {
            let src_view = tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst_view = tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let mip_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bake-mip-bg"),
                layout: &self.mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&self.samp) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_view) },
                ],
            });
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bake-mip-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.mip_pl);
            p.set_bind_group(0, &mip_bg, &[]);
            p.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let full = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (full, tex)
    }
}
