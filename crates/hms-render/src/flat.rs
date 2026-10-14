//! FLAT / white-mesh view — the cheapest view that still shows the map's layout well enough to
//! build in. Settings ▸ Graphics ▸ "Flat / white mesh".
//!
//! Everything expensive is gone: no textures, no material bind groups, no lightmap decode, no
//! shadows, no bloom pyramid, no tonemap, no fog, no foliage, no particles. Geometry is drawn
//! once through a single pipeline that reads only `pos` + `normal` from the mesh vertex stream
//! and the model matrix from the instance stream (locations 0/1 and 4..7 — a stable subset of
//! `mesh.rs`'s layout, which is why this file does not have to touch it).
//!
//! It is made to look *deliberate* rather than broken, which is the whole point of offering it:
//!   * clay shading — a fixed three-quarter key light plus a sky/ground hemisphere term, so
//!     surfaces separate by orientation instead of flattening into one silhouette;
//!   * a dark depth-discontinuity outline in the resolve pass, which is what makes a white model
//!     read as a technical drawing instead of a white blob;
//!   * a faint distance shade so big maps still have depth cues.
//!
//! The resolve pass writes the same `COLOR_FORMAT` target the post pass writes, so egui, the
//! screenshot path and the headless capture are all unchanged.

use eframe::wgpu;

use super::{COLOR_FORMAT, DEPTH_FORMAT, SCENE_HDR_FORMAT};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeomUniform {
    view_proj: [[f32; 4]; 4],
    /// xyz = direction TO the key light, w = unused.
    key: [f32; 4],
    /// x = ambient floor, y = key weight, z = hemisphere weight, w = selected tint boost.
    shade: [f32; 4],
    /// xyz = camera position, w = distance-shade range in world units (0 = off).
    eye: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveUniform {
    /// x = outline strength (0 = no outline), y = relative depth threshold,
    /// z/w = 1/scene-target size.
    p: [f32; 4],
}

pub struct FlatRenderer {
    geom_pipeline: wgpu::RenderPipeline,
    geom_bg: wgpu::BindGroup,
    geom_buf: wgpu::Buffer,
    /// 4-tap (supersampled) and 1-tap resolves — two pipelines rather than a shader branch so
    /// the tap count is a compile-time constant.
    resolve4: wgpu::RenderPipeline,
    resolve1: wgpu::RenderPipeline,
    resolve_bgl: wgpu::BindGroupLayout,
    resolve_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Rebuilt whenever the scene targets are reallocated (resize / render-scale change).
    resolve_bg: Option<wgpu::BindGroup>,
}

impl FlatRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let geom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-geom"),
            source: wgpu::ShaderSource::Wgsl(GEOM_WGSL.into()),
        });
        let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flat-geom-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let geom_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-geom-uniform"),
            size: std::mem::size_of::<GeomUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let geom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-geom-bg"),
            layout: &geom_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: geom_buf.as_entire_binding() }],
        });
        let geom_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flat-geom-pl"),
            bind_group_layouts: &[&geom_bgl],
            push_constant_ranges: &[],
        });
        // Only the attributes this pass actually reads are declared; the vertex/instance strides
        // match `mesh.rs` so the very same buffers bind without a copy.
        let vbl = wgpu::VertexBufferLayout {
            array_stride: crate::mesh::MESH_VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            ],
        };
        let inst = wgpu::VertexBufferLayout {
            array_stride: crate::mesh::MESH_INSTANCE_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 7 },
                // instance tint (location 8): only its ALPHA is read, as a "hide me" flag so a
                // mesh the editor dims stays dim here too.
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 8 },
            ],
        };
        let geom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat-geom-pipeline"),
            layout: Some(&geom_layout),
            vertex: wgpu::VertexState {
                module: &geom_shader,
                entry_point: Some("vs"),
                buffers: &[vbl, inst],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Halo winding varies (same reason as the main mesh pipeline)
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
                module: &geom_shader,
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

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flat-resolve-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flat-resolve-pl"),
            bind_group_layouts: &[&resolve_bgl],
            push_constant_ranges: &[],
        });
        let mk_resolve = |taps: u32| {
            let src = RESOLVE_WGSL.replace("__TAPS__", &taps.to_string());
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flat-resolve"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("flat-resolve-pipeline"),
                layout: Some(&resolve_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
            })
        };
        let resolve4 = mk_resolve(4);
        let resolve1 = mk_resolve(1);
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-resolve-uniform"),
            size: std::mem::size_of::<ResolveUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("flat-resolve-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            geom_pipeline,
            geom_bg,
            geom_buf,
            resolve4,
            resolve1,
            resolve_bgl,
            resolve_buf,
            sampler,
            resolve_bg: None,
        }
    }

    /// Point the resolve at the current scene targets. Called on construction-of-targets
    /// (resize / render-scale change), never per frame.
    pub fn retarget(&mut self, device: &wgpu::Device, hdr: &wgpu::TextureView, depth: &wgpu::TextureView) {
        self.resolve_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-resolve-bg"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 3, resource: self.resolve_buf.as_entire_binding() },
            ],
        }));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        key: glam::Vec3,
        eye: glam::Vec3,
        distance_shade: f32,
        outline: f32,
        scene_size: (u32, u32),
    ) {
        let k = if key.length_squared() > 1e-6 { key.normalize() } else { glam::Vec3::Z };
        queue.write_buffer(
            &self.geom_buf,
            0,
            bytemuck::bytes_of(&GeomUniform {
                view_proj,
                key: [k.x, k.y, k.z, 0.0],
                // Clay weights, tuned by looking at the result on Forge World: the first pass
                // (0.34 floor / 0.44 key / 0.22 hemisphere) came out almost uniformly white, with
                // only the outlines carrying the shape. A lower floor and a stronger key give the
                // terrain and the Forge pieces real form while keeping every face readable.
                shade: [0.20, 0.58, 0.26, 0.0],
                eye: [eye.x, eye.y, eye.z, distance_shade.max(0.0)],
            }),
        );
        queue.write_buffer(
            &self.resolve_buf,
            0,
            bytemuck::bytes_of(&ResolveUniform {
                p: [
                    outline.max(0.0),
                    0.012,
                    1.0 / scene_size.0.max(1) as f32,
                    1.0 / scene_size.1.max(1) as f32,
                ],
            }),
        );
    }

    pub fn geom_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.geom_pipeline
    }
    pub fn geom_bind_group(&self) -> &wgpu::BindGroup {
        &self.geom_bg
    }

    /// Draw one mesh list through the flat pipeline. `pass` must already have the pipeline and
    /// bind group set (see `SceneRenderer::render`'s flat branch) — set once for the whole list.
    pub fn draw_list(pass: &mut wgpu::RenderPass<'_>, meshes: &[super::GpuMesh]) {
        for m in meshes {
            let (vb, ib, n, inst, ninst) = m.flat_parts();
            if n == 0 || ninst == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_vertex_buffer(1, inst.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..n, 0, 0..ninst);
        }
    }

    pub fn resolve(&self, pass: &mut wgpu::RenderPass<'_>, taps: u32) {
        let Some(bg) = &self.resolve_bg else { return };
        pass.set_pipeline(if taps >= 4 { &self.resolve4 } else { &self.resolve1 });
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

const GEOM_WGSL: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    key: vec4<f32>,
    shade: vec4<f32>,
    eye: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) wpos: vec3<f32>,
    @location(2) fade: f32,
};

@vertex
fn vs(v: VIn) -> VOut {
    let m = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let wp = m * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    o.clip = u.view_proj * wp;
    o.n = (m * vec4<f32>(v.normal, 0.0)).xyz;
    o.wpos = wp.xyz;
    // A mesh the editor has faded out (instance tint alpha) stays faded here.
    o.fade = clamp(v.tint.a, 0.0, 1.0);
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    // Facet normal when the stored one is degenerate (particle/ribbon lanes store zeros), so
    // nothing ever renders as an unlit black patch.
    var n = i.n;
    if (dot(n, n) < 1e-8) {
        n = normalize(cross(dpdx(i.wpos), dpdy(i.wpos)) + vec3<f32>(0.0, 0.0, 1e-6));
    } else {
        n = normalize(n);
    }
    // Two-sided: clay reads the same from either side of a single-sided Halo surface.
    let kd = abs(dot(n, u.key.xyz));
    let hemi = 0.5 + 0.5 * n.z;
    var g = u.shade.x + u.shade.y * kd + u.shade.z * hemi;
    // Faint distance shade so a 2 km Forge World still has depth.
    if (u.eye.w > 0.0) {
        let d = clamp(length(i.wpos - u.eye.xyz) / u.eye.w, 0.0, 1.0);
        g = mix(g, 0.66, 0.22 * d * d);
    }
    g = g * mix(0.35, 1.0, i.fade);
    return vec4<f32>(vec3<f32>(g), 1.0);
}
"#;

const RESOLVE_WGSL: &str = r#"
struct RU { p: vec4<f32> };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var dep: texture_depth_2d;
@group(0) @binding(3) var<uniform> r: RU;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VOut;
    o.clip = vec4<f32>(p[vi], 0.0, 1.0);
    o.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return o;
}

/// Depth-discontinuity outline: the RELATIVE step between a texel and its 4 neighbours, so one
/// threshold works at every distance (absolute depth deltas do not — a far wall's neighbours
/// differ by a thousandth of a near one's).
fn outline(uv: vec2<f32>) -> f32 {
    if (r.p.x <= 0.0) { return 0.0; }
    let dim = vec2<i32>(textureDimensions(dep));
    let c = vec2<i32>(uv * vec2<f32>(dim));
    let d = textureLoad(dep, clamp(c, vec2<i32>(0), dim - 1), 0);
    if (d >= 1.0) { return 0.0; } // sky / nothing drawn
    var m = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        var off = vec2<i32>(1, 0);
        if (k == 1) { off = vec2<i32>(-1, 0); }
        if (k == 2) { off = vec2<i32>(0, 1); }
        if (k == 3) { off = vec2<i32>(0, -1); }
        let dn = textureLoad(dep, clamp(c + off, vec2<i32>(0), dim - 1), 0);
        // only an edge where the NEIGHBOUR is further away: keeps the line on the silhouette
        // instead of doubling it on both sides of every step.
        m = max(m, (dn - d) / max(1.0 - d, 1e-6));
    }
    return clamp((m - r.p.y) * 24.0, 0.0, 1.0) * r.p.x;
}

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var c: vec3<f32>;
    if (__TAPS__ == 4) {
        let o = 0.5 * vec2<f32>(r.p.z, r.p.w);
        c = (textureSample(src, samp, uv + vec2<f32>(-o.x, -o.y)).rgb
           + textureSample(src, samp, uv + vec2<f32>( o.x, -o.y)).rgb
           + textureSample(src, samp, uv + vec2<f32>(-o.x,  o.y)).rgb
           + textureSample(src, samp, uv + vec2<f32>( o.x,  o.y)).rgb) * 0.25;
    } else {
        c = textureSample(src, samp, uv).rgb;
    }
    let e = outline(uv);
    c = mix(c, vec3<f32>(0.10, 0.11, 0.13), e);
    // The colour target is read back through an sRGB VIEW (see make_targets), so write encoded
    // bytes: the same convention the post pass uses.
    return vec4<f32>(pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
"#;

// Hooks for the crate's offline WGSL validation test (`wgsl_tests`), which naga-validates every
// shader source at `cargo test` time so a typo cannot ship as a runtime pipeline panic.
pub(crate) const GEOM_WGSL_FOR_TEST: &str = GEOM_WGSL;
pub(crate) fn resolve_wgsl_for_test(taps: u32) -> String {
    RESOLVE_WGSL.replace("__TAPS__", &taps.to_string())
}
