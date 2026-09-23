//! Offline PATH-TRACED lightmapper (visual-preview stage).
//!
//! Goal: a toggleable option that IGNORES the map's baked lightmaps/shadows and
//! replaces them with our own path-traced global illumination — and crucially, what we SEE must
//! equal what a future Reach-format lightmap EXPORT would look like. We guarantee that by baking
//! into the SAME representation the engine/HMS lightmap path already consumes: per-lightmap-texel
//! **dual-VMF** irradiance (an ambient/mean colour + a dominant incoming direction + a concentration
//! "bandwidth"), evaluated through the exact same `mesh_shade` dual-VMF decode + vMF diffuse LUT the
//! real lightmaps use. Bake rich GI here, compress to dual-VMF, and the shader can't tell our atlas
//! from Bungie's — so the preview is export-accurate by construction. No exporting yet.
//!
//! SCOPE: the MAP ONLY. We bake the BSP's lightmaps and use the BSP triangle soup (`bsp_soup`) as
//! the sole occluder/bounce geometry — FORGE OBJECTS ARE EXCLUDED (they're separate meshes, never
//! added to the soup). That's intentional: re-lightmapping the map is the point; forge props neither
//! cast into nor receive our bake.
//!
//! This module is the ENGINE-INDEPENDENT core: given the world triangle soup (for occlusion/bounce),
//! a set of surface sample points (texel centres, filled by the raster stage), and the scene's light
//! parameters, it path-traces directional irradiance per point and fits it to dual-VMF. The texel
//! G-buffer (raster into lightmap-UV atlas space) and the DM/SDM atlas encode/bind live in scene.rs.

use crate::decal_projector::TriSoup;
use glam::Vec3;

/// Scene lighting inputs for the bake, gathered from the SceneController (sun from the airprobe-
/// derived direction, sky/ambient from the scenario tints). Absolute-ish linear radiances; the
/// atlas encode later normalises to the map's per-BSP K so exposure matches the shipped lightmaps.
#[derive(Clone, Copy)]
pub struct BakeLights {
    /// Direction TO the sun (normalized). Rays are traced toward this for the direct term.
    pub sun_dir: Vec3,
    /// Sun radiance (linear rgb). Direct contribution = sun_color * NdotL * visibility.
    pub sun_color: Vec3,
    /// Sky/ambient radiance seen by a ray that escapes to the sky (linear rgb). Also the
    /// hemisphere fill for open areas. A crude uniform dome for now (a real sky-cube sample is a
    /// later refinement); still physically a proper hemisphere integral, not a flat add.
    pub sky_color: Vec3,
    /// Ground/bounce albedo fallback when a bounce ray hits geometry whose material albedo we
    /// can't resolve (kept mid-grey so colour bleed is present but never invents saturated hue).
    pub fallback_albedo: Vec3,
}

/// Path-traced result for one surface point: `ambient` = the rotation-invariant mean irradiance
/// (SH L0) and `bandwidth` = how CONCENTRATED the incoming light is (the vMF mean-resultant
/// length: 0 = fully diffuse sky, 1 = a sharp single direction like direct sun).
#[derive(Clone, Copy, Default)]
pub struct TexelIrradiance {
    pub ambient: Vec3,
    pub bandwidth: f32,
}

/// Deterministic per-texel PRNG (xorshift64*) — reproducible bakes (same map → same result), and
/// no `rand` dependency. Seeded from the texel index so neighbouring texels decorrelate.
#[inline]
fn seed_rng(mut x: u64) -> u64 {
    // avalanche the seed so low texel indices don't start correlated
    x ^= x >> 33; x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33; x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x | 1
}
#[inline]
fn next_u64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}
#[inline]
fn next_f32(s: &mut u64) -> f32 {
    // top 24 bits → [0,1)
    ((next_u64(s) >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Build an orthonormal basis around `n` (Duff et al. 2017 — branchless, stable).
#[inline]
fn onb(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0f32 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// Cosine-weighted hemisphere sample around `n` (concentrates samples where the cosine term is
/// large → low-variance diffuse integration; the pdf cancels the NdotL so we accumulate radiance).
#[inline]
fn cosine_hemisphere(n: Vec3, rng: &mut u64) -> Vec3 {
    let (t, bt) = onb(n);
    let u1 = next_f32(rng);
    let u2 = next_f32(rng);
    let r = u1.sqrt();
    let phi = std::f32::consts::TAU * u2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt();
    (t * x + bt * y + n * z).normalize_or_zero()
}

const EPS: f32 = 0.02; // ray origin offset along the normal to avoid self-intersection

/// Path-trace directional irradiance at one surface point. `albedo_at` resolves a bounce hit's
/// diffuse albedo (from the material/texture) for colour bleed; the raster stage supplies it.
///
/// The integrand is the incoming radiance over the hemisphere. Cosine-weighted sampling means each
/// sample already carries the NdotL·(1/pi) diffuse weight in its pdf, so we accumulate raw incoming
/// radiance and the mean IS the diffuse irradiance (× albedo happens at shade time, matching how the
/// engine lightmap stores incident light, not exitance). We track the direction-weighted first
/// moment to recover the dominant direction + concentration for the dual-VMF fit.
pub fn bake_point<F>(
    soup: &TriSoup,
    pos: Vec3,
    normal: Vec3,
    lights: &BakeLights,
    samples: u32,
    bounces: u32,
    texel_index: u64,
    albedo_at: &F,
) -> TexelIrradiance
where
    F: Fn(u32, Vec3) -> Vec3,
{
    let n = normal.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return TexelIrradiance::default();
    }
    let mut rng = seed_rng(texel_index);
    let origin = pos + n * EPS;

    let mut sum = Vec3::ZERO; // mean incoming radiance (→ ambient / SH L0)
    let mut dir_moment = Vec3::ZERO; // luminance-weighted sum of incoming directions (→ dominant dir)

    // --- direct sun (its own dedicated shadow ray — a sharp delta light, sampled once/texel not
    // via the hemisphere, so its hard shadow is crisp regardless of `samples`). ---
    let ndl = n.dot(lights.sun_dir);
    if ndl > 0.0 && lights.sun_color.length_squared() > 0.0 {
        let occluded = soup.raycast(origin, lights.sun_dir, 1.0e5).is_some();
        if !occluded {
            let contrib = lights.sun_color * ndl;
            sum += contrib;
            dir_moment += lights.sun_dir * luminance(contrib);
        }
    }

    // --- indirect + sky (cosine-weighted hemisphere). ---
    let inv_s = 1.0 / samples.max(1) as f32;
    for _ in 0..samples {
        let d = cosine_hemisphere(n, &mut rng);
        if d.length_squared() < 0.5 { continue; }
        let radiance = trace_indirect(soup, origin, d, lights, bounces, &mut rng, albedo_at);
        sum += radiance;
        dir_moment += d * luminance(radiance);
    }
    // cosine-weighted mean of the hemisphere samples (sun added separately above, already weighted)
    let ambient = sum * inv_s.max(0.0) + Vec3::ZERO;
    // Recover the concentration. |dir_moment| relative to total luminance is the vMF
    // mean-resultant length R∈[0,1]: R→0 uniform (bandwidth 0), R→1 a single direction.
    let total_lum = luminance(sum).max(1e-6);
    let r_vec = dir_moment / total_lum;
    let r_len = r_vec.length().clamp(0.0, 1.0);
    TexelIrradiance {
        ambient: (ambient).max(Vec3::ZERO),
        bandwidth: r_len,
    }
}

/// Trace one indirect path from `origin` along `dir`: if it hits geometry, add that surface's
/// direct-sun response × its albedo (1 bounce), then continue for further bounces; if it escapes,
/// return the sky radiance. This is a simple unidirectional path tracer with Russian-roulette-free
/// fixed depth (fine for a few bounces of diffuse GI).
fn trace_indirect<F>(
    soup: &TriSoup,
    origin: Vec3,
    dir: Vec3,
    lights: &BakeLights,
    bounces: u32,
    rng: &mut u64,
    albedo_at: &F,
) -> Vec3
where
    F: Fn(u32, Vec3) -> Vec3,
{
    match soup.raycast_hit(origin, dir, 1.0e5) {
        None => lights.sky_color, // escaped to sky
        Some((t, nrm, tri)) => {
            if bounces == 0 {
                return Vec3::ZERO; // depth exhausted; treat as black (ambient already covers fill)
            }
            let hit = origin + dir * t;
            let mut hn = nrm.normalize_or_zero();
            if hn.dot(dir) > 0.0 { hn = -hn; } // face the incoming ray
            let albedo = albedo_at(tri, hit);
            let horigin = hit + hn * EPS;
            // direct sun at the bounce surface
            let mut lit = Vec3::ZERO;
            let ndl = hn.dot(lights.sun_dir);
            if ndl > 0.0 && soup.raycast(horigin, lights.sun_dir, 1.0e5).is_none() {
                lit += lights.sun_color * ndl;
            }
            // one further diffuse bounce toward the sky/geometry
            let d2 = cosine_hemisphere(hn, rng);
            let further = trace_indirect(soup, horigin, d2, lights, bounces - 1, rng, albedo_at);
            albedo * (lit + further)
        }
    }
}

#[inline]
pub fn luminance(c: Vec3) -> f32 {
    c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722
}

// =============================== GPU BVH (stage 1) ===============================
// The CPU tracer above is the ground-truth oracle; the SHIPPING bake runs on the GPU (655s CPU →
// target seconds). The GPU compute shader needs the map geometry as flat storage buffers: a linear
// BVH (median-split, GPU-traversable with a fixed stack) + a reordered triangle array. Built on the
// CPU once per bake from the BSP triangle soup, then uploaded.

/// One GPU BVH node (32 bytes, std430-friendly). Inner node: `count == 0`, `left_first` = index of
/// the LEFT child (right = left+1). Leaf: `count > 0`, `left_first` = first triangle index into the
/// reordered `GpuBvh::tris`. AABB in world space.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    pub left_first: u32,
    pub aabb_max: [f32; 3],
    pub count: u32,
}

/// One GPU triangle (48 bytes: 3 × vec3 + pad). World space.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhTri {
    pub a: [f32; 3],
    pub _pa: f32,
    pub b: [f32; 3],
    pub _pb: f32,
    pub c: [f32; 3],
    pub _pc: f32,
}

/// Flat BVH ready for GPU upload: `nodes[0]` is the root.
pub struct GpuBvh {
    pub nodes: Vec<BvhNode>,
    pub tris: Vec<BvhTri>,
    /// `tris[i]` came from input triangle `order[i]` — lets callers reorder per-triangle payloads
    /// (material ids) to match the leaf ranges.
    pub order: Vec<u32>,
}

impl GpuBvh {
    /// Build a median-split BVH over world-space triangles. Leaves hold ≤ `LEAF` triangles. Simple,
    /// robust, and produces a compact linear array the compute shader can traverse with a small stack.
    pub fn build(triangles: &[[Vec3; 3]]) -> GpuBvh {
        const LEAF: usize = 4;
        let n = triangles.len();
        let mut tris: Vec<BvhTri> = triangles
            .iter()
            .map(|t| BvhTri {
                a: t[0].into(), _pa: 0.0,
                b: t[1].into(), _pb: 0.0,
                c: t[2].into(), _pc: 0.0,
            })
            .collect();
        // centroids + per-tri aabb for partitioning
        let centroid = |t: &BvhTri| {
            (Vec3::from(t.a) + Vec3::from(t.b) + Vec3::from(t.c)) / 3.0
        };
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(n.max(1) * 2);
        // reserve root
        nodes.push(BvhNode::default());
        // recursive build over the `order` index range [start,end); writes node `ni`.
        // Iterative stack to avoid deep recursion on big maps.
        struct Task { ni: usize, start: usize, end: usize }
        let mut stack = vec![Task { ni: 0, start: 0, end: n }];
        while let Some(Task { ni, start, end }) = stack.pop() {
            // compute node AABB over [start,end)
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &oi in &order[start..end] {
                let t = &tris[oi as usize];
                for p in [Vec3::from(t.a), Vec3::from(t.b), Vec3::from(t.c)] {
                    mn = mn.min(p); mx = mx.max(p);
                }
            }
            let cnt = end - start;
            if cnt <= LEAF {
                nodes[ni] = BvhNode { aabb_min: mn.into(), left_first: start as u32, aabb_max: mx.into(), count: cnt as u32 };
                continue;
            }
            // split on the largest centroid-extent axis at the median
            let ext = mx - mn;
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let mid = start + cnt / 2;
            order[start..end].select_nth_unstable_by(cnt / 2, |&x, &y| {
                let cx = centroid(&tris[x as usize])[axis];
                let cy = centroid(&tris[y as usize])[axis];
                cx.partial_cmp(&cy).unwrap_or(std::cmp::Ordering::Equal)
            });
            let l = nodes.len();
            nodes.push(BvhNode::default());
            nodes.push(BvhNode::default());
            nodes[ni] = BvhNode { aabb_min: mn.into(), left_first: l as u32, aabb_max: mx.into(), count: 0 };
            stack.push(Task { ni: l, start, end: mid });
            stack.push(Task { ni: l + 1, start: mid, end });
        }
        // reorder triangles to match leaf ranges (leaves index `order`, so materialize that order)
        let reordered: Vec<BvhTri> = order.iter().map(|&oi| tris[oi as usize]).collect();
        tris = reordered;
        GpuBvh { nodes, tris, order }
    }
}
