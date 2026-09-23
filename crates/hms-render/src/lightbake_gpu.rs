//! GPU compute path tracer for the lightmap bake. The CPU tracer (hms-app/lightbake.rs) is
//! the ground-truth oracle; this runs the shipping bake on the GPU.
//!
//! Inputs (all storage buffers): a flat BVH (`BvhNode[]`) + reordered triangles (`BvhTri[]`) over
//! the MAP BSP soup, and a per-lightmap-texel G-buffer (world pos + normal; normal.w=0 → uncovered).
//! The compute shader traces, per covered texel: a sun shadow ray + `samples` cosine-hemisphere rays
//! each bouncing `bounces` times against the BVH, accumulating incoming radiance + a direction moment,
//! then writes the dual-VMF fields (ambient, dominant dir, dominant colour, bandwidth) to `out[]`.
//! The host reads those back and (later stage) encodes them into DM/SDM atlas textures.
//!
//! Buffer layouts mirror hms-app's `lightbake::{BvhNode,BvhTri}` byte-for-byte (std430).

use eframe::wgpu;
use wgpu::util::DeviceExt;

/// Per-dispatch params (std140 uniform). Matches `Params` in the WGSL below.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBakeParams {
    pub sun_dir: [f32; 4],       // xyz = to-sun (normalized), w unused
    pub sun_color: [f32; 4],     // rgb radiance, w unused
    pub sky_color: [f32; 4],     // rgb sky/ambient radiance
    pub albedo: [f32; 4],        // rgb bounce albedo fallback
    pub dims: [u32; 4],          // [texel_count, samples, bounces, node_count]
    pub misc: [u32; 4],          // [tri_count, seed, light_count, texel_base]
    pub fog: [f32; 4],           // [sigma (extinction/unit), fog_top_z, total_emitter_weight, 0]
    pub emisc: [u32; 4],         // [emitter_count, 0, 0, 0]
}

/// One scenario point/spot light for the bake (64B, std430). From `ZhSimpleLight`. `range` = the
/// far cutoff distance; `sphere_pct >= 0.5` = omni/point (no cone), else spot (cone via `cos_cutoff`).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub pos: [f32; 3],
    pub range: f32,
    pub color: [f32; 3],
    pub sphere_pct: f32,
    pub dir: [f32; 3],
    pub cos_cutoff: f32,
    pub angle_ratio: f32,
    pub angle_power: f32,
    pub far_end: f32,
    pub far_ratio: f32,
}

/// One emissive AREA-LIGHT triangle for the bake (64B, std430). World-space verts + the material's
/// unclamped HDR self-illum radiance. `v0.w` = running cumulative CDF weight (Σ area×luminance up to
/// and including this emitter); `v1.w` = this emitter's luminance. The tracer importance-samples
/// emitters by that CDF and does next-event estimation (shadow ray) so fixtures light their rooms.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEmitter {
    pub v0: [f32; 4], // xyz + w = cumulative CDF weight
    pub v1: [f32; 4], // xyz + w = radiance luminance
    pub v2: [f32; 4], // xyz + pad
    pub rad: [f32; 4], // emitter radiance rgb + pad
}

/// Per-texel dual-VMF output (48B, 3×vec4): matches the WGSL `Out` struct.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTexelOut {
    pub ambient: [f32; 4],   // rgb irradiance mean + w=coverage(1/0)
    pub dom: [f32; 4],       // dominant dir xyz + w=bandwidth
    pub dom_color: [f32; 4], // dominant lobe colour rgb + pad
}

/// One G-buffer texel (32B): world pos + world normal. normal.w = 0 means uncovered (skipped).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTexel {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}

pub struct GpuLightBaker {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuLightBaker {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lightbake-compute"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lightbake-bgl"),
            entries: &[
                storage_entry(0, true),  // nodes (read)
                storage_entry(1, true),  // tris (read)
                storage_entry(2, true),  // gbuf (read)
                storage_entry(3, false), // out (read-write)
                storage_entry(5, true),  // lights (read)
                storage_entry(6, true),  // emitters (read)
                storage_entry(9, true),  // per-triangle material id (BVH order)
                storage_entry(10, true), // materials (3 vec4 each: albedo+flag, emission, transmittance)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lightbake-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lightbake-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        GpuLightBaker { pipeline, bgl }
    }

    /// Run the bake. `nodes`/`tris` = raw BVH bytes; `gbuf` = `GpuTexel[texel_count]`; `lights` =
    /// scenario point/spot lights. Dispatched in CHUNKS so the caller can show smooth progress and so
    /// each GPU submission stays short (the bake runs on a worker thread → the window stays live).
    /// `progress` is advanced from `plo`→`phi` (fixed-point ×1000) as chunks complete. Returns the
    /// per-texel dual-VMF fields read back once at the end.
    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nodes: &[u8],
        tris: &[u8],
        gbuf: &[GpuTexel],
        lights: &[GpuLight],
        emitters: &[GpuEmitter],
        // per (BVH-reordered) triangle material id; empty = flat `params.albedo`, fully opaque
        tri_mat: &[u32],
        // 3 vec4 per material — [albedo rgb, flag (0 opaque, 1 foliage, 2 sky, 3 translucent)],
        // [self-illum radiance rgb, 0], [transmission tint rgb, 0]
        mats: &[[f32; 4]],
        mut params: GpuBakeParams,
        progress: &std::sync::atomic::AtomicU32,
        plo: u32,
        phi: u32,
    ) -> Vec<GpuTexelOut> {
        use std::sync::atomic::Ordering;
        let texel_count = gbuf.len();
        if texel_count == 0 { return Vec::new(); }
        let nodes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh-nodes"), contents: nodes, usage: wgpu::BufferUsages::STORAGE,
        });
        let tris_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh-tris"), contents: tris, usage: wgpu::BufferUsages::STORAGE,
        });
        let gbuf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gbuf"), contents: bytemuck::cast_slice(gbuf), usage: wgpu::BufferUsages::STORAGE,
        });
        // lights buffer — at least one element so the binding is never zero-sized.
        let mut lb = lights.to_vec();
        if lb.is_empty() { lb.push(GpuLight::default()); }
        let lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake-lights"), contents: bytemuck::cast_slice(&lb), usage: wgpu::BufferUsages::STORAGE,
        });
        // emitter buffer — at least one element so the binding is never zero-sized.
        let mut eb = emitters.to_vec();
        if eb.is_empty() { eb.push(GpuEmitter::default()); }
        let emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake-emitters"), contents: bytemuck::cast_slice(&eb), usage: wgpu::BufferUsages::STORAGE,
        });
        let mut tm = tri_mat.to_vec(); if tm.is_empty() { tm.push(0); }
        let tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake-tri-mat"), contents: bytemuck::cast_slice(&tm), usage: wgpu::BufferUsages::STORAGE,
        });
        let mut mv = mats.to_vec(); if mv.len() < 2 { mv = vec![[0.5, 0.5, 0.5, 0.0], [0.0; 4]]; }
        let mats_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake-mats"), contents: bytemuck::cast_slice(&mv), usage: wgpu::BufferUsages::STORAGE,
        });
        let use_mats = !tri_mat.is_empty() && mats.len() >= 2;
        let out_size = (texel_count * std::mem::size_of::<GpuTexelOut>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake-out"), size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        params.dims[0] = texel_count as u32;
        params.misc[2] = lights.len() as u32; // light_count (0 when only the dummy)
        params.emisc[0] = emitters.len() as u32; // emitter_count (0 when only the dummy)
        params.emisc[1] = if use_mats { 1 } else { 0 }; // per-triangle materials bound
        let param_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake-params"), size: std::mem::size_of::<GpuBakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake-bg"), layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gbuf_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: lights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: emit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: tri_mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: mats_buf.as_entire_binding() },
            ],
        });
        // Chunked dispatch: ~a few hundred k texels per submission → smooth progress + short GPU calls.
        const CHUNK: u32 = 262_144;
        let total = texel_count as u32;
        let mut base = 0u32;
        while base < total {
            let this = CHUNK.min(total - base);
            params.misc[3] = base; // texel base offset for this chunk (WGSL idx = base + gid.x)
            queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.pipeline);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups((this + 63) / 64, 1, 1);
            }
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::Maintain::Wait);
            base += this;
            let frac = base as f32 / total as f32;
            progress.store(plo + ((phi - plo) as f32 * frac) as u32, Ordering::Relaxed);
        }
        // single readback of the finished output
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake-read"), size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
        queue.submit(Some(enc.finish()));
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let out: Vec<GpuTexelOut> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();
        out
    }
}

/// Pack ONE baked texel into (DM, SDM0, SDM1, SDM2) RGBA8, the exact inverse of
/// `mesh_shade::gpu_lightmap_tint`, so binding these as the atlas makes the shipped decode reproduce
/// the path-traced lighting (visual == a future Reach export). Decode reference:
///   f_int = exp(-9*DM.r);  dom_dir = normalize((s0.a,s1.a,s2.a)*2-1);  bandwidth = |that|
///   dom_col = (s0.rgb + s1.rgb*2-1)*f_int;  fill_col = s2.rgb*f_int;  vis = DM.g
///   irr = (vmf_lut(N·dom_dir, bw)*dom_col + 0.25*fill_col)/PI * hdr * k
/// We bind the baked meshes with hdr=1 and this `k` (calibrate `k` so exposure matches the shipped
/// atlas). The dominant-lobe vMF peak is folded into `k` (approximated as 1 here; live calibration).
pub fn encode_dual_vmf(o: &GpuTexelOut, k: f32) -> [[u8; 4]; 4] {
    if o.ambient[3] < 0.5 {
        return [[0, 0, 0, 255], [0, 0, 128, 128], [0, 0, 128, 128], [0, 0, 128, 128]];
    }
    const PI: f32 = std::f32::consts::PI;
    let kk = k.max(1e-4);
    let dir = [o.dom[0], o.dom[1], o.dom[2]];
    let bw = o.dom[3].clamp(0.0, 1.0);
    // per-channel "needed" slice values BEFORE the per-texel f_int exposure divide.
    let fill_need = [o.ambient[0] * 4.0 * PI / kk, o.ambient[1] * 4.0 * PI / kk, o.ambient[2] * 4.0 * PI / kk];
    let dom_need = [o.dom_color[0] * PI / kk, o.dom_color[1] * PI / kk, o.dom_color[2] * PI / kk];
    // choose f_int so fill_col (≤1) and dom_col (≤2) both fit; clamp to the encodable HDR range.
    let need = fill_need.iter().cloned().fold(0.0f32, f32::max)
        .max(dom_need.iter().cloned().fold(0.0f32, f32::max) * 0.5);
    let f_int = need.clamp((-9.0f32).exp(), 1.0);
    let dm_r = (-(f_int.ln()) / 9.0).clamp(0.0, 1.0);
    let inv = 1.0 / f_int;
    let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    // direction → the three slice alphas: (dir*bw)*0.5+0.5
    let a = |i: usize| u8c(dir[i] * bw * 0.5 + 0.5);
    // dominant colour split across s0.rgb (0..1) + s1.rgb (carries the >1 part via *2-1)
    let mut s0 = [0u8; 4]; let mut s1 = [0u8; 4]; let mut s2 = [0u8; 4];
    for c in 0..3 {
        let dom_col = (dom_need[c] * inv).clamp(0.0, 2.0);
        let lo = dom_col.min(1.0);
        let hi = ((dom_col - lo) + 1.0) * 0.5; // s1.rgb such that s0+s1*2-1 = dom_col
        s0[c] = u8c(lo);
        s1[c] = u8c(hi);
        s2[c] = u8c(fill_need[c] * inv); // fill/isotropic lobe
    }
    s0[3] = a(0); s1[3] = a(1); s2[3] = a(2);
    let dm = [u8c(dm_r), 255, 0, 255]; // r=intensity code, g=visibility(1), b/a unused
    [dm, s0, s1, s2]
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const WGSL: &str = r#"
struct BvhNode { aabb_min: vec3<f32>, left_first: u32, aabb_max: vec3<f32>, count: u32 };
struct Tri { a: vec3<f32>, _pa: f32, b: vec3<f32>, _pb: f32, c: vec3<f32>, _pc: f32 };
struct Texel { pos: vec4<f32>, normal: vec4<f32> };
struct Out { ambient: vec4<f32>, dom: vec4<f32>, dom_color: vec4<f32> };
struct Params {
    sun_dir: vec4<f32>, sun_color: vec4<f32>, sky_color: vec4<f32>, albedo: vec4<f32>,
    dims: vec4<u32>,  // texel_count, samples, bounces, node_count
    misc: vec4<u32>,  // tri_count, seed, light_count, texel_base
    fog: vec4<f32>,   // sigma (extinction/unit), fog_top_z, total_emitter_weight, _
    emisc: vec4<u32>, // emitter_count, _, _, _
};
struct Light {
    pos: vec3<f32>, range: f32,
    color: vec3<f32>, sphere_pct: f32,   // sphere_pct>=0.5 = omni/point, else spot
    dir: vec3<f32>, cos_cutoff: f32,
    angle_ratio: f32, angle_power: f32, far_end: f32, far_ratio: f32,
};
struct Emitter {
    v0: vec4<f32>, // xyz + w = cumulative CDF weight
    v1: vec4<f32>, // xyz + w = radiance luminance
    v2: vec4<f32>, // xyz + pad
    rad: vec4<f32>,// emitter radiance rgb + pad
};

@group(0) @binding(0) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> gbuf: array<Texel>;
@group(0) @binding(3) var<storage, read_write> outb: array<Out>;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(4) var<uniform> P: Params;
@group(0) @binding(9) var<storage, read> tri_mat: array<u32>;
@group(0) @binding(10) var<storage, read> mats: array<vec4<f32>>;

// --- PRNG (PCG-ish) ---
fn pcg(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x * 747796405u + 2891336453u;
    let word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    *state = x;
    return (word >> 22u) ^ word;
}
fn randf(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) * (1.0 / 4294967296.0);
}

const EPS: f32 = 0.02;
const T_MAX: f32 = 1.0e9;
const FIREFLY_CAP: f32 = 12.0; // max per-sample emitter irradiance (firefly suppression)

// ray-aabb slab test → tnear (or -1 miss)
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

// Möller–Trumbore. Returns t (or -1); the caller recomputes the geometric normal from the
// hit triangle (tri_normal).
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

// Per-triangle material lookups (flat P.albedo / fully opaque when no materials are bound).
fn have_mats() -> bool { return P.emisc.y > 0u; }
fn tri_flag(i: u32) -> f32 { if (!have_mats()) { return 0.0; } return mats[tri_mat[i] * 3u].w; }
fn tri_albedo(i: u32) -> vec3<f32> { if (!have_mats()) { return P.albedo.xyz; } return mats[tri_mat[i] * 3u].rgb; }
fn tri_emission(i: u32) -> vec3<f32> { if (!have_mats()) { return vec3<f32>(0.0); } return mats[tri_mat[i] * 3u + 1u].rgb; }
fn tri_transmit(i: u32) -> vec3<f32> { if (!have_mats()) { return vec3<f32>(0.0); } return mats[tri_mat[i] * 3u + 2u].rgb; }

// Light transmitted from `o` along `d` for up to `tmax`: 0 behind an opaque surface, the product of the
// transmittances of any TRANSLUCENT surfaces crossed (glass roofs / panes / grates), 1 when clear. Sky-flagged
// triangles never occlude. This is what lets daylight into a glass-roofed atrium — treating those as opaque
// left Sword Base's interior ~100x darker than the shipped lightmap (measured with HMS_BAKE_COMPARE).
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
                    if (f > 2.5) { tr = tr * tri_transmit(ti); if (max(tr.r, max(tr.g, tr.b)) < 0.01) { return vec3<f32>(0.0); } }
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
// closest hit → returns t (or -1) and writes the hit triangle index into `*out_tri`.
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

// cosine-weighted hemisphere sample around n
fn cosine_hemi(n: vec3<f32>, s: ptr<function, u32>) -> vec3<f32> {
    let sgn = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sgn + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    let bt = vec3<f32>(b, sgn + n.y * n.y * a, -n.y);
    let u1 = randf(s);
    let u2 = randf(s);
    let r = sqrt(u1);
    let phi = 6.2831853 * u2;
    let z = sqrt(max(1.0 - u1, 0.0));
    return normalize(t * (r * cos(phi)) + bt * (r * sin(phi)) + n * z);
}

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

// Beer-Lambert transmittance of light travelling from `o` along `d` for up to `tmax`
// units, counting only the portion BELOW the fog top plane (z < fog_top). sigma is the per-unit
// extinction derived from the map's atmosphere fog. Returns 1.0 (no attenuation) when there's no fog.
// We do NOT add in-scatter here — the fog VEIL is applied at render time, so baking it would double it.
fn fog_transmittance(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> f32 {
    let sigma = P.fog.x;
    if (sigma <= 0.0) { return 1.0; }
    let top = P.fog.y;
    var t_below = 0.0;
    let dz = d.z;
    if (abs(dz) < 1e-6) {
        if (o.z < top) { t_below = tmax; }
    } else {
        let tc = (top - o.z) / dz; // ray param where z == fog_top
        if (dz < 0.0) {
            t_below = max(0.0, tmax - max(tc, 0.0)); // going down → below for t > tc
        } else {
            t_below = clamp(tc, 0.0, tmax);          // going up → below for t < tc
        }
    }
    return exp(-sigma * t_below);
}

// Next-event estimation: importance-sample ONE emissive area-light triangle (by the area×luminance CDF),
// pick a point on it, and return the incoming irradiance it contributes at surface (x, n) via a
// shadow ray — the low-noise "next event estimation" that makes light fixtures illuminate rooms.
// Writes the light direction into *out_wi (for the dominant-direction moment). Fog-attenuated.
fn sample_emitters(x: vec3<f32>, n: vec3<f32>, s: ptr<function, u32>, out_wi: ptr<function, vec3<f32>>) -> vec3<f32> {
    let ne = P.emisc.x;
    if (ne == 0u) { return vec3<f32>(0.0); }
    let total = P.fog.z; // total emitter weight (Σ area×lum)
    if (total <= 0.0) { return vec3<f32>(0.0); }
    let u = randf(s) * total;
    // binary search the monotonic cumulative-weight CDF (v0.w) — O(log N) over the emitter set
    // (there can be tens of thousands of self-illum triangles; a linear scan would dominate the bake).
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
    if (b0 + b1 > 1.0) { b0 = 1.0 - b0; b1 = 1.0 - b1; } // uniform point in the triangle
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
    let cos_emit = abs(dot(n_emit, wi)); // two-sided emitter (panel orientation is unreliable)
    if (cos_emit <= 0.0) { return vec3<f32>(0.0); }
    let ve = visibility(x, wi, dist - 2.0 * EPS);
    if (max(ve.r, max(ve.g, ve.b)) <= 0.0) { return vec3<f32>(0.0); }
    // pdf_area = lum_i / total  →  dE = L_e * cos_recv * cos_emit / dist^2 * (total / lum_i).
    // The geometry term G = cos_recv*cos_emit/dist^2 is CLAMPED to kill the near-field 1/dist^2
    // fireflies that dominate variance when a receiver sits right next to a small bright fixture.
    let lum_i = max(E.v1.w, 1e-6);
    let g = min(cos_recv * cos_emit / (dist * dist), 1.0);
    var dE = E.rad.xyz * ve * (g * (total / lum_i));
    // Firefly clamp: with tens of thousands of heterogeneous emitters the W/lum_i importance weight
    // spikes for dim/rarely-sampled fixtures, and those spikes don't average out at practical sample
    // counts (salt-and-pepper on the dense backdrop). Cap the per-sample contribution — slight bias,
    // large variance win. FIREFLY_CAP is the max plausible single-fixture irradiance in this bake's units.
    let m = lum(dE);
    if (m > FIREFLY_CAP) { dE = dE * (FIREFLY_CAP / m); }
    *out_wi = wi;
    return dE * fog_transmittance(x, wi, dist);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = P.misc.w + gid.x; // texel base offset (chunked dispatch) + local index
    if (idx >= P.dims.x) { return; }
    let tx = gbuf[idx];
    if (tx.normal.w < 0.5) {
        outb[idx] = Out(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let n = normalize(tx.normal.xyz);
    let origin = tx.pos.xyz + n * EPS;
    let sun_dir = normalize(P.sun_dir.xyz);
    let samples = P.dims.y;
    let bounces = P.dims.z;
    var rng: u32 = idx * 9781u + P.misc.y * 6271u + 1u;

    // DIRECT lighting at this texel — computed ONCE (not averaged over the hemisphere samples);
    // direct and indirect are separate accumulators so the sun is not divided by the sample count.
    var direct = vec3<f32>(0.0);
    var direct_moment = vec3<f32>(0.0);
    let ne = P.emisc.x;

    // direct sun (dedicated shadow ray), fog-attenuated along the ray to the sky
    let ndl = dot(n, sun_dir);
    if (ndl > 0.0 && lum(P.sun_color.xyz) > 0.0) {
        let vs = visibility(origin, sun_dir, T_MAX);
        if (max(vs.r, max(vs.g, vs.b)) > 0.0) {
            let c = P.sun_color.xyz * vs * ndl * fog_transmittance(origin, sun_dir, T_MAX);
            direct = direct + c;
            direct_moment = direct_moment + sun_dir * lum(c);
        }
    }

    // direct scenario point/spot lights (ZhSimpleLight) — own shadow ray + fog attenuation.
    let nlights = P.misc.z;
    for (var li = 0u; li < nlights; li = li + 1u) {
        let L = lights[li];
        let toL = L.pos - origin;
        let dist = length(toL);
        let range = select(L.far_end, L.range, L.range > 1.0);
        if (dist > range || dist < 1e-4) { continue; }
        let ldir = toL / dist;
        let ndl2 = dot(n, ldir);
        if (ndl2 <= 0.0) { continue; }
        var att = clamp(1.0 - dist / max(range, 0.01), 0.0, 1.0);
        att = att * att; // smooth quadratic falloff to the range cutoff
        if (L.sphere_pct < 0.5) { // spot cone
            let cd = dot(-ldir, normalize(L.dir));
            if (cd < L.cos_cutoff) { continue; }
            let cone = clamp((cd - L.cos_cutoff) / max(1.0 - L.cos_cutoff, 1e-3), 0.0, 1.0);
            att = att * pow(cone, max(L.angle_power, 1.0));
        }
        if (att <= 0.0) { continue; }
        let vl = visibility(origin, ldir, dist - 0.05);
        if (max(vl.r, max(vl.g, vl.b)) <= 0.0) { continue; }
        let c = L.color * vl * (ndl2 * att) * fog_transmittance(origin, ldir, dist);
        direct = direct + c;
        direct_moment = direct_moment + ldir * lum(c);
    }

    // direct EMISSIVE area lights (self-illum fixtures) via next-event estimation. A few samples
    // keeps the interior noise low; each returns fog-attenuated incoming irradiance.
    if (ne > 0u) {
        let NEE = 16u; // many emitters (tens of thousands of fixtures) → more direct samples = less speckle
        for (var ei = 0u; ei < NEE; ei = ei + 1u) {
            var ewi = vec3<f32>(0.0);
            let dE = sample_emitters(origin, n, &rng, &ewi);
            direct = direct + dE / f32(NEE);
            direct_moment = direct_moment + ewi * (lum(dE) / f32(NEE));
        }
    }

    // INDIRECT bounce GI + sky + emitter light bounced off other surfaces — averaged over samples.
    let inv_s = 1.0 / f32(max(samples, 1u));
    var indirect = vec3<f32>(0.0);
    var indirect_moment = vec3<f32>(0.0);
    for (var si = 0u; si < samples; si = si + 1u) {
        let first_dir = cosine_hemi(n, &rng);
        var d = first_dir;
        var throughput = vec3<f32>(1.0);
        var o = origin;
        var radiance = vec3<f32>(0.0);
        for (var bnc = 0u; bnc <= bounces; bnc = bnc + 1u) {
            var tri: u32 = 0u;
            let t = trace(o, d, &tri);
            if (t < 0.0) {
                // escaped to sky — attenuated by the fog it passed through on the way out
                radiance = radiance + throughput * P.sky_color.xyz * visibility(o, d, T_MAX) * fog_transmittance(o, d, T_MAX);
                break;
            }
            throughput = throughput * fog_transmittance(o, d, t); // fog along the traveled segment
            var hn = tri_normal(tri);
            if (dot(hn, d) > 0.0) { hn = -hn; }
            let hit = o + d * t;
            let ho = hit + hn * EPS;
            let alb = tri_albedo(tri);
            // the hit surface's own self-illum (light fixtures seen directly by a bounce ray)
            radiance = radiance + throughput * tri_emission(tri);
            // direct sun at the bounce surface
            let bndl = dot(hn, sun_dir);
            if (bndl > 0.0) {
                let vb = visibility(ho, sun_dir, T_MAX);
                if (max(vb.r, max(vb.g, vb.b)) > 0.0) { radiance = radiance + throughput * alb * P.sun_color.xyz * vb * bndl * fog_transmittance(ho, sun_dir, T_MAX); }
            }
            // emitter NEE at the bounce surface — propagates fixture light indirectly
            if (ne > 0u) {
                var bwi = vec3<f32>(0.0);
                let dEb = sample_emitters(ho, hn, &rng, &bwi);
                radiance = radiance + throughput * alb * dEb;
            }
            if (bnc == bounces) { break; }
            throughput = throughput * alb;
            o = ho;
            d = cosine_hemi(hn, &rng);
        }
        indirect = indirect + radiance;
        indirect_moment = indirect_moment + first_dir * lum(radiance);
    }

    // combine: direct is exact (1×), indirect is the Monte-Carlo average
    let sum = direct + indirect * inv_s;
    let mom = direct_moment + indirect_moment * inv_s;
    let total_lum = max(lum(sum), 1e-6);
    let r_vec = mom / total_lum;
    let r_len = clamp(length(r_vec), 0.0, 1.0);
    var dom_dir = n;
    if (r_len > 1e-3) { dom_dir = r_vec / r_len; }
    let dom_color = sum * r_len;
    outb[idx] = Out(
        vec4<f32>(max(sum, vec3<f32>(0.0)), 1.0),
        vec4<f32>(dom_dir, r_len),
        vec4<f32>(max(dom_color, vec3<f32>(0.0)), 0.0),
    );
}
"#;
