//! Mesh rendering: the material pipelines (opaque, alpha-test, decorator, glass, additive,
//! particle, water, terrain, sky, decal, shadow), GPU mesh upload, texture upload, and the
//! WGSL material shaders as string constants (ports of the Reach and Halo 4 shipped shaders).
//!
//! Fed from `hms-native` decode output (BSP/model geometry + decoded bitmaps). Each mesh
//! carries a material bind group and an instance buffer of model matrices + per-material
//! lanes so repeated objects batch in one draw.

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::Mat4;

/// GPU memory accounting (HMS_MEMPROF) — sums the bytes committed to GPU textures
/// (split uncompressed RGBA/BGRA vs block-compressed BC) and mesh vertex/index/instance
/// buffers, so the load path can report where the multi-GB footprint actually goes.
pub mod memprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static TEX_UNCOMPRESSED: AtomicU64 = AtomicU64::new(0);
    pub static TEX_BC: AtomicU64 = AtomicU64::new(0);
    pub static BUF: AtomicU64 = AtomicU64::new(0);
    pub static TEX_COUNT: AtomicU64 = AtomicU64::new(0);
    pub static MESH_COUNT: AtomicU64 = AtomicU64::new(0);
    pub fn add_tex_uncompressed(n: u64) { TEX_UNCOMPRESSED.fetch_add(n, Ordering::Relaxed); TEX_COUNT.fetch_add(1, Ordering::Relaxed); }
    pub fn add_tex_bc(n: u64) { TEX_BC.fetch_add(n, Ordering::Relaxed); TEX_COUNT.fetch_add(1, Ordering::Relaxed); }
    pub fn add_buf(n: u64) { BUF.fetch_add(n, Ordering::Relaxed); }
    pub fn add_mesh() { MESH_COUNT.fetch_add(1, Ordering::Relaxed); }
    /// (uncompressed_tex_bytes, bc_tex_bytes, buffer_bytes, tex_count, mesh_count)
    pub fn report() -> (u64, u64, u64, u64, u64) {
        (TEX_UNCOMPRESSED.load(Ordering::Relaxed), TEX_BC.load(Ordering::Relaxed),
         BUF.load(Ordering::Relaxed), TEX_COUNT.load(Ordering::Relaxed), MESH_COUNT.load(Ordering::Relaxed))
    }
    // Per-label-category totals (count, bytes) of every GPU texture/buffer the load path
    // creates (HMS_MEMDIAG). Cumulative over the process (creations, not live set); wgpu frees
    // GPU memory when the last handle drops, so a cleared cache still shows here as "ever created".
    static CATS: std::sync::Mutex<std::collections::BTreeMap<&'static str, (u64, u64)>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    pub fn note(category: &'static str, bytes: u64) {
        if let Ok(mut m) = CATS.lock() {
            let e = m.entry(category).or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes;
        }
    }
    /// Snapshot of the per-category totals: (category, count, bytes), sorted by category.
    pub fn categories() -> Vec<(&'static str, u64, u64)> {
        CATS.lock().map(|m| m.iter().map(|(k, v)| (*k, v.0, v.1)).collect()).unwrap_or_default()
    }
}

/// Interleaved render vertex assembled from the DLL's separate position /
/// normal / uv decode streams.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Per-vertex colour (baked ambient for decorators; white for everything
    /// else). Multiplies the shaded output.
    pub color: [f32; 4],
    /// Per-vertex world-space sway basis (decorators only; zero elsewhere). The
    /// VS adds `sway.xy * sin(phase)` so foliage bends with a wind wave.
    pub sway: [f32; 3],
    /// Secondary (lightmap-atlas) UV. Submap-LOCAL [0,1] — maps directly onto the
    /// per-submap DM/SDM texture uploaded via `raw_dds`. Only populated for atlas-lit
    /// meshes; [0,0] elsewhere. The fragment shader samples DM/SDM at this UV and
    /// dual-VMF-decodes the baked irradiance on the GPU.
    pub uv2: [f32; 2],
    /// Stored per-vertex world-space tangent frame — `[T.x, T.y, T.z, handedness]`.
    /// T is the IA-decoded (SNORM16) tangent (native ZH_BSP_DecodeMeshTangents, already
    /// instance-rotated); handedness = `sign(dot(cross(N,T), B))` from the native binormal
    /// export (recovers the tangent's discarded 4th component). The shader builds
    /// `binormal = normalize(cross(N,T)·handedness)` from this — the engine's stored frame.
    /// `[0,0,0,0]` (default, all non-BSP callers: objects/decorators/sky/decals) → the shader
    /// falls back to a screen-space-derivative Gram-Schmidt frame. Some lanes re-use the slot
    /// (light-volume ribbons, BSP foliage back-face lighting, particle billboard UV).
    pub tangent: [f32; 4],
}

impl Default for MeshVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0; 2],
            color: [1.0; 4],
            sway: [0.0; 3],
            uv2: [0.0; 2],
            tangent: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    /// (sx, sy, _, _) UV scroll rate/sec for animated surfaces (waterfalls);
    /// [0;4] for everything else (auv == uv → no change). Location 10.
    scroll: [f32; 4],
    /// (tile.x, tile.y, off.x, off.y) opaque detail-map UV xform from the authored
    /// `detail_map` rmt2 constant. [0;4] → shader falls back to MESH_DETAIL_SCALE=6.0.
    /// Location 11.
    detail_xform: [f32; 4],
    /// [uv_density, spec_from_alpha, analytical spec strength, roughness]. Location 12.
    aux: [f32; 4],
    /// GPU lightmap: [flag, hdr_scale, k_scale, mode]. flag>0.5 → mesh_shade samples the
    /// DM/SDM atlas at uv2 and dual-VMF-decodes the baked irradiance on the GPU. 0 → the
    /// shader uses the per-vertex baked i.tint; mode 2 (`is_object`) flags a dynamic object.
    /// Location 14.
    lm: [f32; 4],
    /// Artist-Fresnel specular material constants from the rmt2 REAL constants:
    /// [albedo_blend(metalness), fresnel_curve_steepness, specular_tint luma (dielectric F0
    /// base), fresnel_color luma (edge reflectance)]. [0;4] → a fixed dielectric F0=0.04 /
    /// Schlick pow-5 / white edge. metalness>0 lerps F0 toward the ALBEDO so metals reflect
    /// their own colour. Location 15.
    spec2: [f32; 4],
    /// Authored `bump_map` UV xform [tile.x, tile.y, off.x, off.y]. The engine samples the
    /// normal at THIS scale (near unit), not the base tile. [0;4] → the base-auv sample.
    /// Location 16.
    bump_xform: [f32; 4],
    /// FINE (high-frequency) self-detail layer applied LIVE, [tile.x, tile.y, flag, _]. For
    /// tile floors (concrete pavers) the engine keeps the fine grain as a runtime layer sampled
    /// at its own high tiling: when flag(z)>0.5 the shader samples the BASE texture (SELF-detail:
    /// detail_tag==base_tag) at uv*tile and multiplies it in (mean-preserving). [0;4] → no
    /// second layer. Location 17.
    fine_xform: [f32; 4],
    /// bump_detail_map UV xform [tile.x, tile.y, flag, _]. flag(z)>0.5 → the shader samples the
    /// SECOND normal (bump_detail_tex) at uv*tile and adds it (`bump.xy += detail.xy`, HREK
    /// calc_bumpmap_detail_ps). [0;4] → single bump. Location 18.
    bump_detail_xform: [f32; 4],
    /// Env control [env_tint_color.r, .g, .b, specular_coefficient]. The engine multiplies the
    /// opaque env reflection by the authored `env_tint_color` (environment_mapping.hlsl_include:74)
    /// and by `specular_coefficient` (two_lobe_phong.hlsl_include:164). rgb==0 → the shader
    /// hue-normalises by sun_tint instead; w==0 → factor 1.0. Only the opaque BSP path passes a
    /// real value. Location 19.
    env_ctl: [f32; 4],
    /// Per-shader material_model dispatch, ENCODED as (material_model + 1) so that 0.0 means
    /// "unset → the unconditional cook_torrance path" (every non-BSP caller passes 0.0). The
    /// opaque BSP path passes material_model+1: 1=diffuse_only(0), 2=cook_torrance(1),
    /// 4=foliage(3), 5=none(4), etc. mesh_shade decodes mm=round(w)-1 and, for mm in {0,3,4},
    /// zeroes the specular contribution the engine never computes. Halo 4 materials pass
    /// -(1 + specular model) (h4_shade). Location 20.
    matmodel: f32,
    /// Self-illumination MODE routing [mode, cmul.r, cmul.g, cmul.b]. mode = the engine
    /// self_illumination enum (0 off / 1 simple / 4 from_albedo / …). Only mode==4 (from_albedo)
    /// is routed: mesh_shade emits `albedo · cmul` (= albedo · self_illum_color · intensity) as the
    /// self-illum and ZEROS the lit albedo term (self_illumination.hlsl_include:115-116).
    /// FOLIAGE_SI_MODE (20) flags an rmfl foliage material. Every non-BSP caller passes [0;4].
    /// Location 21.
    si_ctl: [f32; 4],
    /// Per-texel `material_texture` roughness/spec override control
    /// [material_texture_black_roughness, material_texture_black_specular_multiplier,
    /// has_material_texture flag, mode]. When flag(z)>0.5 the shader samples the bound
    /// material_texture (binding 14) at `mattex_xform` and applies the engine override
    /// (cook_torrance_core.hlsl_include:112-118): `roughness = lerp(black_roughness, roughness,
    /// material_texture.a)` and `specular_coefficient *= lerp(black_specmult, 1,
    /// material_texture.a)`. [0;4] → no override (the default 1×1 grey material_texture has a=1
    /// → both lerps are identities). .w selects the glass / change-colour / two-detail lanes.
    /// Location 23.
    mattex: [f32; 4],
    /// Authored `material_texture` UV xform [tile.x, tile.y, off.x, off.y]. [0;4] → the shader
    /// samples the material_texture at the base auv (the common Reach authoring; the map is the
    /// base's roughness/spec companion). Location 24.
    mattex_xform: [f32; 4],
    /// Extended bump `detail_blend` / `three_detail_blend` UV tiles for the 2nd/3rd detail-bump
    /// maps — [bump_detail_map2.tile.x, .y, bump_detail_map3.tile.x, .y]. The engine blends
    /// detail1/2/3 by the base bump map's alpha (extended_bump_mapping.hlsl_include:4,26).
    /// Location 25.
    xbump: [f32; 4],
    /// Control [bump_detail2_present, bump_detail3_present, env_authored, snorm bits]. .x/.y gate
    /// the extended detail_blend / three_detail_blend combine (bindings 15/16). .z (env_authored)
    /// = 1.0 when a REAL per-material `environment_map` cube is bound at binding 10 →
    /// env_reflection does the engine per_pixel decode (`reflect_dir.y=-y`, `rgb·a`) instead of
    /// sampling the captured sky cube. Location 26.
    xctl: [f32; 4],
    /// Per-object dual-vMF probe [dom_dir.xyz, bandwidth] (w=0 -> lane unused)
    obj_probe0: [f32; 4],
    /// [dom_rgb, sun-visibility mask]
    obj_probe1: [f32; 4],
    /// Authored specular_tint.rgb (w=1 when present) - the artist-Fresnel F0 COLOUR.
    spec_rgb: [f32; 4],
    /// Authored fresnel_color.rgb (w=1 when present) - the grazing-angle Fresnel COLOUR.
    fres_rgb: [f32; 4],
    /// ENGINE object-lighting lane (docs/hrek_re/16_object_lighting.md), packed f16 pairs because
    /// the vertex-attribute budget is exhausted (32/32 on NVIDIA): .x = oct(fill-lobe dir),
    /// .y = oct(bounce dir = -N_hit), .z = bounce.rg, .w = (bounce.b, flag). flag 1 → mesh_shade
    /// does the engine lobes+sun merge (`sub_140828CD0`) for the object's key light and adds the
    /// bounce light (2× into the fill lobe + a third light). All zero → key = dominant lobe.
    /// Location 31.
    obj_light: [u32; 4],
}

/// Per-material `shader_terrain` parameters (read from the widened
/// `ZH_TerrainLayers`). Carries the engine's per-layer base + detail UV transforms
/// (`transform_texcoord(uv) = uv*scale + offset`), the blend mask's own UV
/// transform, the global albedo tint, the active-layer set, and per-layer detail
/// presence. Threaded into `upload_terrain_mesh` and packed into the terrain
/// uniform (see `TERRAIN_WGSL`).
#[derive(Clone, Copy)]
pub struct TerrainMaterialParams {
    pub base_tile: [[f32; 2]; 4],
    pub base_offset: [[f32; 2]; 4],
    pub detail_tile: [[f32; 2]; 4],
    pub detail_offset: [[f32; 2]; 4],
    pub blend_xform: [f32; 4],
    pub global_tint: [f32; 4],
    pub active: [f32; 4],
    pub detail_present: [f32; 4],
    pub bump_tile: [[f32; 2]; 4],
    pub bump_offset: [[f32; 2]; 4],
    pub bump_present: [f32; 4],
    /// Per-layer detail_bump (2nd normal map) xform + presence. Engine adds
    /// detail_bump.xy to bump.xy UNWEIGHTED when ACTIVE_MATERIAL_COUNT<4
    /// (terrain_new.hlsl_include:84,214-216). present gates the per-layer add.
    pub detail_bump_tile: [[f32; 2]; 4],
    pub detail_bump_offset: [[f32; 2]; 4],
    pub detail_bump_present: [f32; 4],
    /// Per-mesh UV density (uv-diagonal / world-diagonal). Packed into global_tint.w; the
    /// terrain shader samples with plain hardware derivatives, so this is carried but unused.
    pub uv_density: f32,
    /// distance_blend_base far base->target colour lerp (terrain_new.hlsl:102-129).
    /// mode: 1.0 = distance_blend_base (apply), 0.0 = morph (no-op).
    pub dbb_mode: f32,
    pub dbb_slope: f32,
    pub dbb_offset: f32,
    /// per-layer target colour (rgb) toward which the base is lerped at distance.
    pub blend_target: [[f32; 4]; 4],
    /// per-layer max blend amount (clamps base_blend before the lerp).
    pub blend_max: [f32; 4],
    /// Per-layer texture decode EXPONENT from the bitmap's authored gamma curve
    /// (ZH_BitmapInfo.Curve): 1.0 = linear/offset_log (sampled RAW), 2.2 = gamma2/sRGB/unknown
    /// (gamma decode). The engine samples through the format the curve selects; a blanket pow(2.2)
    /// over-darkens the linear-authored overlay bases.
    pub base_curve: [f32; 4],
    pub detail_curve: [f32; 4],
    /// Per-layer 1.0 when the bump / detail_bump texture is a raw BC5_SNORM (DXN) upload.
    pub bump_snorm: [f32; 4],
    pub detail_bump_snorm: [f32; 4],
}

/// Per-material water shader parameters (read from the rmt2 RealConstants via
/// `ZH_BSP_GetMaterialShaderConstants`). Uploaded as a uniform for the water pass.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WaterParams {
    /// watercolor_coefficient, water_murkiness, fresnel_coefficient, fresnel_dark_spot
    pub p0: [f32; 4],
    /// reflection_coefficient, slope_scaler, time_warp (layer A rate), time_warp_aux (layer B rate)
    pub p1: [f32; 4],
    /// deep water tint (water_color_pure.rgb) + has_slope_tex flag (.w: 1=real wave_slope_array bound, 0=procedural noise)
    pub deep: [f32; 4],
    /// wave UV xform: layerA tile (.xy) from wave_slope_array/wave_displacement_array, layerB tile (.zw)
    pub wave: [f32; 4],
    /// puddle/edge params: (refraction_extinct_distance, foam_cut, bankalpha_mode, normal_variation_tweak).
    /// bankalpha_mode: 1 = paint (watercolor.a is the shore-coverage mask → puddle edge fade); 0 = none (ocean).
    pub edge: [f32; 4],
    /// Engine diffuse: water_diffuse.rgb (the authored blue-green self-cast, water_shading
    /// .hlsl_include:1021 `color_diffuse = water_diffuse * saturate(dot(up,N))`), + sunspot_cut (.w).
    pub diff: [f32; 4],
    /// Authored environment_map cube control:
    ///   .x = authored-cube flag (1 = the REAL authored cube is bound at binding 10 → the engine
    ///        `env.rgb*256`, alpha-split-at-sunspot_cut decode; 0 = the captured sky dome
    ///        stand-in), .y = HDR scale (unused; the shader applies the engine ×256),
    ///        .z = chop debug view, .w = 1 when a real global_shape_texture is bound.
    pub env: [f32; 4],
    /// (foam_coefficient, foam_pow, shadow_intensity_mark, detail_slope_steepness).
    ///   .x = foam_coefficient (final linear gain on auto-foam; engine PARAM, default 1.0)
    ///   .y = foam_pow (auto-foam contrast, clamped >=1; default 1.0)
    ///   .z = shadow_intensity_mark (reflection sun_scale lightmap floor; engine default 0.5)
    ///   .w = detail_slope_steepness (3rd wave-slope layer gain; 0 = detail layer off)
    pub foam: [f32; 4],
    /// The authored UV OFFSET (xform.zw) for the two wave layers. Engine
    /// `uv = mesh_uv·xform.xy + xform.zw` [texture_xform:26].
    ///   .xy = layerA offset (wave_displacement_array_xform.zw)
    ///   .zw = layerB offset (wave_slope_array_xform.zw)
    /// Forge ocean authors 0,0 → no-op; a stream authoring non-zero shifts the wave pattern.
    pub xform2: [f32; 4],
    /// from_shape bank_alpha body-UV: the water body's world-space XY AABB, used to map
    /// the NON-tiling global_shape_texture across the body ONCE (like the engine's authored body
    /// texcoords) instead of the world-tiled wave UV. `.xy` = aabb_min.xy, `.zw` = 1/aabb_size.xy
    /// (inverse extent). `body_uv = (world_pos.xy - .xy) * .zw`, clamped to [0,1]. `.zw == 0`
    /// (degenerate/zero-area body) → shader falls back to the mesh texcoord i.uv.
    pub body: [f32; 4],
    /// Per-material slice-cursor animation. The engine's `time_warp` /
    /// `time_warp_aux` are PARAMETERS animated by a Value overlay (frac(t/period) ramp, 0..1) —
    /// `.x` = time_warp period (s), `.y` = time_warp_aux period (s); 0 = not animated → the shader
    /// uses the constant authored arg in `.z` / `.w` instead.
    pub anim: [f32; 4],
    /// Per-parameter TranslationX/Y overlay scroll (uv/s, sign already matched to
    /// the engine's visual direction): `.xy` = wave_displacement_array_xform, `.zw` = wave_slope_array_xform.
    pub scroll: [f32; 4],
    /// (detail_slope_scale_x, detail_slope_scale_y, detail_slope_scale_z, mode).
    /// mode 1 = engine-exact path (mesh texcoords × xform + scroll, frac(t/period)·N cursors,
    /// compute_detail_slope); 0 = the world/30 fallback path (unusable mesh UVs, or HMS_WATER_LEGACY=1).
    pub detail: [f32; 4],
    /// watercolor_texture_xform (uv·xy + zw). The engine samples the watercolor
    /// texture — whose alpha is the paint-mode puddle coverage — at the STATIC mesh texcoord × this
    /// xform (water_shading.hlsl_include:857); the exact path uses it for paint-mode (puddle) coverage.
    pub wcxf: [f32; 4],
    /// The rmt2 template CATEGORY options, read from the water template tag name
    /// (`shaders\water_templates\_<waveshape>_<watercolor>_<reflection>_<refraction>_<bankalpha>_
    /// <appearance>_<global_shape>_<foam>_<detail>`, the category order of water.render_method_definition).
    ///   .x = watercolor option (0 = pure -> water_color_pure, 1 = texture -> watercolor_texture * coef)
    ///   .y = reflection option (0 = none -> no cube term, 1 = static, 2 = dynamic)
    ///   .z = refraction option (0 = none -> opaque water_color, 1 = dynamic -> Beer-Lambert see-through)
    ///   .w = wave source: 0 = wave_slope_array slices (waveshape default), 2 = waveshape bump with a
    ///        raw BC5_SNORM bump_map bound in the slope slot (sample signed xy), 3 = bump with an 8-bit
    ///        decoded bump_map (xy*2-1). Bump mode: wave/xform2/scroll carry bump_map / bump_detail_map xforms.
    /// The shader branches on these exactly like the HLSL TEST_CATEGORY_OPTION switches; bankalpha
    /// rides in edge.z. (Inferring the puddle/ocean split from which bitmaps are bound would route a
    /// pure-watercolor stream that also authors foam through the texture path.)
    pub cat: [f32; 4],
}

impl Default for WaterParams {
    fn default() -> Self {
        // The fallback when the rmt2 constants don't resolve: engine defaults, neutral.
        Self {
            // watercolor_coefficient 1.0 (engine default), murkiness 8, fresnel 4, dark spot 0.8
            p0: [1.0, 8.0, 4.0, 0.80],
            p1: [1.0, 1.0, 0.1, 0.07],
            deep: [0.015, 0.05, 0.08, 0.0],
            wave: [1.0, 1.0, 1.0, 1.0],
            edge: [30.0, 1.0, 0.0, 0.7],
            // water_diffuse: a faint blue-green; sunspot_cut 0.2.
            diff: [0.0, 0.02, 0.04, 0.2],
            // no authored cube (captured sky dome stand-in)
            env: [0.0, 256.0, 0.0, 0.0],
            // foam_coefficient 1.0, foam_pow 1.0, shadow_intensity_mark 0.5, detail_slope_steepness 0.
            foam: [1.0, 1.0, 0.5, 0.0],
            // no authored UV offset (Forge ocean = 0,0)
            xform2: [0.0, 0.0, 0.0, 0.0],
            // no body AABB (.zw=0 → shader uses i.uv for global_shape)
            body: [0.0, 0.0, 0.0, 0.0],
            // no overlay periods / scroll; detail scale 1; mode 0 until the material's constants
            // resolve (scene.rs flips .w to 1 on the exact path)
            anim: [0.0, 0.0, 0.0, 0.0],
            scroll: [0.0, 0.0, 0.0, 0.0],
            detail: [1.0, 1.0, 1.0, 0.0],
            wcxf: [1.0, 1.0, 0.0, 0.0],
            // texture watercolor, static reflection, dynamic refraction, no foam = the Forge
            // ocean's categories, the safest default when the template name does not resolve.
            cat: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

/// A material's base-colour texture as a ready-to-bind group.
pub struct Material {
    pub bind_group: wgpu::BindGroup,
}

/// GPU-lightmap inputs for a mesh: the per-submap DM/SDM atlas texture views (uploaded raw-BC
/// via `raw_dds`, sampled at `uv2` in the fragment shader) plus the per-mesh HDR/compression
/// scalars. `None` on `upload_mesh` → the mesh keeps the CPU-baked `i.tint` (dummy DM/SDM
/// bound, lm.flag=0).
pub struct LightmapInputs {
    /// Owned texture views (a wgpu TextureView keeps its parent texture alive, so the
    /// caller can drop the returned Texture handle). Bound at material_bgl slots 5/6.
    pub dm: wgpu::TextureView,
    pub sdm: wgpu::TextureView,
    /// SDM z-slices 1 and 2 (the dual-VMF decode reads direction from all three slice alphas +
    /// dominant colour from slice0.rgb + slice1.rgb·2−1). None → slice 0 is bound in their place.
    pub sdm1: Option<wgpu::TextureView>,
    pub sdm2: Option<wgpu::TextureView>,
    /// entry.hdr_scale (per-cluster HDR multiplier; 1.0 for instances). Negative = a Halo 4 atlas
    /// (-K_direct).
    pub hdr: f32,
    /// The atlas compression constant K (Halo 4: K_indirect), folded into the decode.
    pub k: f32,
    /// Not read by the shader for atlas meshes (lm.w > 1.5 marks dynamic objects, which never
    /// carry an atlas).
    pub mode: f32,
}

impl GpuMesh {
    /// Write per-instance probe lanes ([dom_dir.xyz, bandwidth], [dom_rgb, mask]) + the packed
    /// object-lighting lane (see `InstanceRaw::obj_light`; all-zero = key = dominant lobe) into
    /// the instance stream.
    pub fn set_instance_probes(&mut self, queue: &wgpu::Queue, probes: &[([f32; 8], [u32; 4])]) {
        if self.inst_raw.is_empty() { return; }
        for (i, (p, l)) in probes.iter().enumerate().take(self.inst_raw.len()) {
            self.inst_raw[i].obj_probe0 = [p[0], p[1], p[2], p[3]];
            self.inst_raw[i].obj_probe1 = [p[4], p[5], p[6], p[7]];
            self.inst_raw[i].obj_light = *l;
        }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.inst_raw));
    }
    /// Diag: the instance-0 lightmap flag lane (1 = GPU atlas bound at upload, 0 = per-vertex).
    pub fn lm_flag(&self) -> f32 { self.inst_raw.first().map_or(-1.0, |i| i.lm[0]) }
    /// The Halo 4 material lanes that upload_mesh cannot take as parameters
    /// (spec colour / fresnel / the two probe float4s are otherwise object-only lanes).
    pub fn set_h4_lanes(&mut self, queue: &wgpu::Queue, spec_rgb: [f32; 4], fres_rgb: [f32; 4], probe0: [f32; 4], probe1: [f32; 4]) {
        if self.inst_raw.is_empty() { return; }
        for r in self.inst_raw.iter_mut() { r.spec_rgb = spec_rgb; r.fres_rgb = fres_rgb; r.obj_probe0 = probe0; r.obj_probe1 = probe1; }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.inst_raw));
    }
    /// Set the material's specular_tint / fresnel_color COLOURS on every instance.
    pub fn set_spec_tints(&mut self, queue: &wgpu::Queue, spec: [f32; 4], fres: [f32; 4]) {
        if self.inst_raw.is_empty() { return; }
        for r in self.inst_raw.iter_mut() { r.spec_rgb = spec; r.fres_rgb = fres; }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.inst_raw));
    }
}
/// One uploaded mesh: geometry + material + per-instance model matrices and lanes.
pub struct GpuMesh {
    /// Built by upload_terrain_mesh (terrain pipeline + bind group); draw() skips it,
    /// draw_terrain() draws only these - lets terrain-shaded OBJECTS ride in dynamic_meshes.
    pub is_terrain: bool,
    /// false → skipped by the shadow (caster) pass. Objects honour their obje
    /// lightmap_shadow_mode (most Forge pieces are 'default/never' and cast nothing in the engine).
    pub casts_shadow: bool,
    /// Halo 4 OBJECT mesh (scenario placement / Forge piece) in a STATIC lane: drawn into
    /// the sun shadow map by `draw_shadow_h4` (the engine's forge-lightmap casters are exactly
    /// the objects). Default false; Reach meshes never set it.
    pub h4_caster: bool,
    /// CPU copy of the instance stream so per-object probe lanes can be patched after upload.
    inst_raw: Vec<InstanceRaw>,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    instance_buf: wgpu::Buffer,
    pub(crate) instance_count: u32, // read by SceneRenderer::shadow_caster_counts
    material: Material,
    /// World centroid: mean instance translation (the transparent back-to-front sort key), or
    /// the decorator group's geometry centroid (the decorator budget's distance key).
    centroid: [f32; 3],
    /// Decorator instances in this group-mesh (the budget weight). Zero for non-decorator
    /// meshes (they are never budget-culled).
    deco_weight: u32,
    /// Engine-exact transparent blend routing: the fixed-function blend the transparent pass
    /// selects for this mesh. 0 = straight alpha (SrcAlpha/InvSrcAlpha, default — glass),
    /// 1 = pre_multiplied_alpha (One/InvSrcAlpha, rgb premultiplied in-shader), 2 = multiply
    /// (Zero/SrcColor → darken), 3 = double_multiply (DstColor/SrcColor → 2·src·dst), 4 =
    /// additive (particles). Mirrors the render_method BLEND_TYPE enum; set by the BSP
    /// transparent routing in scene.rs.
    blend_mode: u8,
}

/// Which transparent pipeline family a list handed to `draw_transparent_sorted` uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransparentKind {
    /// alpha-blend family, routed per mesh by `GpuMesh::blend_mode` (glass, multiply overlays).
    Blend,
    /// One/One additive (fs_holo: holograms, force fields, glow overlays).
    Additive,
    /// alpha-blended solid marker holo (fs_holo_solid: spawn/hill/kill-safe shapes).
    HoloSolid,
}

impl GpuMesh {
    /// Tag a decorator group-mesh with its centroid + instance count so `draw_foliage` can
    /// distance-sort and budget-cull it like the engine.
    pub fn set_deco_budget(&mut self, centroid: [f32; 3], deco_weight: u32) {
        self.centroid = centroid;
        self.deco_weight = deco_weight;
    }
    /// Tag a transparent mesh with its engine blend mode (see `blend_mode`). Only the
    /// transparent / particle passes read it; opaque/additive/decal passes are unaffected.
    pub fn set_blend_mode(&mut self, mode: u8) {
        self.blend_mode = mode;
    }
    /// World centroid (mean instance translation) — the transparent back-to-front sort key.
    pub fn centroid(&self) -> [f32; 3] { self.centroid }
    /// The engine blend mode set by `set_blend_mode` (0 when untagged).
    pub fn blend_mode(&self) -> u8 { self.blend_mode }
    /// Per-instance colour lane rewrite (Halo 4 objects: probe irradiance + sun vis).
    pub fn set_instance_colors(&mut self, queue: &wgpu::Queue, colors: &[[f32; 4]]) {
        if self.inst_raw.is_empty() { return; }
        for (r, c) in self.inst_raw.iter_mut().zip(colors) { r.color = *c; }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.inst_raw));
    }
    /// Halo 4 object lobe lane: the packed per-instance `obj_light` words (h4_shade's probe
    /// lobes; see hms-app h4/scene.rs `ObjectLightField::object_lanes`). Zeros = no lobes.
    pub fn set_instance_obj_light(&mut self, queue: &wgpu::Queue, lanes: &[[u32; 4]]) {
        if self.inst_raw.is_empty() { return; }
        for (r, l) in self.inst_raw.iter_mut().zip(lanes) { r.obj_light = *l; }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.inst_raw));
    }
    /// Override the sort centroid (per-frame particle batches are built in world space).
    pub fn set_centroid(&mut self, c: [f32; 3]) { self.centroid = c; }
}

pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    alphatest_pipeline: wgpu::RenderPipeline,
    foliage_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,
    /// Non-decal pre_multiplied_alpha transparents (render_method BLEND_TYPE 6). Engine
    /// premul = ONE/INV_SRC_ALPHA with rgb premultiplied in the shader (fs_blend_premul).
    /// Shares the decal-premul fixed-function blend state; selected by GpuMesh.blend_mode==1.
    blend_premul_pipeline: wgpu::RenderPipeline,
    /// Non-decal multiply transparents (BLEND_TYPE 2). ZERO/SRC_COLOR → dst·src (darken).
    /// Shares the sky/decal multiply blend state; selected by GpuMesh.blend_mode==2.
    blend_multiply_pipeline: wgpu::RenderPipeline,
    /// Non-decal double_multiply transparents (BLEND_TYPE 3). DST_COLOR/SRC_COLOR →
    /// 2·src·dst. Shares the decal double_multiply blend state; blend_mode==3.
    blend_double_multiply_pipeline: wgpu::RenderPipeline,
    additive_pipeline: wgpu::RenderPipeline,
    /// fs_particle pipelines by engine blend mode — additive (One/One, rgb premultiplied by
    /// alpha in-shader = additive / add_src_times_srcalpha), alpha (SrcAlpha/InvSrcAlpha), premultiplied
    /// (One/InvSrcAlpha), multiply (Zero/SrcColor). Depth-tested, no depth write, cull none.
    particle_add_pipeline: wgpu::RenderPipeline,
    particle_alpha_pipeline: wgpu::RenderPipeline,
    particle_premul_pipeline: wgpu::RenderPipeline,
    particle_multiply_pipeline: wgpu::RenderPipeline,
    /// Alpha-blended "solid" hologram (shader_halogram look) for OBJECT markers — spawn
    /// points, hill/objective globes, kill/safe boundary shells. Unlike the additive glow,
    /// this keeps a translucent tinted silhouette + fresnel rim so the SHAPE is always
    /// readable (additive vanished against bright sky/snow). SrcAlpha/OneMinusSrcAlpha.
    holo_solid_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    terrain_pipeline: wgpu::RenderPipeline,
    sky_opaque_pipeline: wgpu::RenderPipeline,
    sky_blend_pipeline: wgpu::RenderPipeline,
    sky_multiply_pipeline: wgpu::RenderPipeline,
    sky_alpha_pipeline: wgpu::RenderPipeline,
    /// pre_multiplied_alpha (HMS blend 6/8) sky panels — One/OneMinusSrcAlpha.
    sky_premul_pipeline: wgpu::RenderPipeline,
    decal_alpha_pipeline: wgpu::RenderPipeline,
    decal_additive_pipeline: wgpu::RenderPipeline,
    decal_multiply_pipeline: wgpu::RenderPipeline,
    decal_double_multiply_pipeline: wgpu::RenderPipeline,
    decal_premul_pipeline: wgpu::RenderPipeline,
    decal_invalpha_pipeline: wgpu::RenderPipeline,
    decal_vector_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_pipeline_h4: wgpu::RenderPipeline,
    material_bgl: wgpu::BindGroupLayout,
    terrain_bgl: wgpu::BindGroupLayout,
    water_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// GPU lightmap sampler (bilinear, top mip only; the engine samples the lightmap atlas
    /// bilinearly). HMS_LM_NEAREST selects a nearest fetch.
    lm_sampler: wgpu::Sampler,
    default_white: wgpu::TextureView,
    /// The 1×1 emissive/detail/bump fallbacks + the env cube, built ONCE, lazily (upload_mesh
    /// has the queue; new() does not). Re-creating them per upload_mesh (3 texture creates +
    /// writes per mesh) cost ~30ms/mesh. OnceLock is thread-safe for the parallel prewarm/upload paths.
    fallbacks: std::sync::OnceLock<FallbackTex>,
    /// GPU base×detail baker (composite on the GPU instead of a CPU loop). Lazily built
    /// (needs the queue) and thread-safe for the parallel prewarm/bake paths.
    detail_baker: std::sync::OnceLock<crate::bake::DetailBaker>,
}

/// Constant 1×1 material fallbacks (emissive off / detail neutral / bump flat), built once.
struct FallbackTex {
    _black_t: wgpu::Texture,
    black: wgpu::TextureView,
    _grey_t: wgpu::Texture,
    grey: wgpu::TextureView,
    _flat_t: wgpu::Texture,
    flat: wgpu::TextureView,
    /// Environment cubemap (Rgba16Float render-target cube) + its sampler, built once here
    /// (the only place with both device+queue) and bound at material slots 10/11 for every mesh.
    /// `env_cube_tex` is retained so `SceneRenderer::render` can build per-face render views
    /// and draw the real sky into it once per map.
    env_cube_tex: wgpu::Texture,
    env_cube: wgpu::TextureView,
    env_samp: wgpu::Sampler,
}

impl FallbackTex {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (black, _black_t) = upload_texture_rgba(device, queue, &[0, 0, 0, 255], 1, 1);
        let (grey, _grey_t) = upload_texture_rgba(device, queue, &[128, 128, 128, 255], 1, 1);
        let (flat, _flat_t) = upload_texture_rgba(device, queue, &[128, 128, 255, 255], 1, 1);
        let (env_cube_tex, env_cube, env_samp) = build_env_cube(device, queue);
        Self { _black_t, black, _grey_t, grey, _flat_t, flat, env_cube_tex, env_cube, env_samp }
    }
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        shadow_bgl: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material-bgl"),
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
                // binding 2: self-illum / emissive texture (default black = off)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: detail map (default mid-gray = neutral under ×4.59)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: bump/normal map (default flat-normal (128,128,255) = off)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: GPU-lightmap DM atlas submap (dominant dir + intensity/vis).
                // Default 1×1 grey (dummy) for non-atlas meshes (lm.flag=0 → never sampled).
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 6: GPU-lightmap SDM atlas submap (dominant colour + vis).
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 7: lightmap sampler (top-mip; declared Filtering since the atlas
                // textures are filterable floats).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 8/9: GPU-lightmap SDM atlas z-slices 1 and 2 (the dual-VMF decode
                // reads the DIRECTION from the ALPHAs of all three SDM slices and the dominant
                // colour from slice0.rgb + slice1.rgb·2−1, per HREK DecompressVMF). Slot 6 = slice 0.
                // Dummy 1×1 black for non-atlas meshes (lm.flag=0 → never sampled).
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 10/11: environment cubemap + its sampler (roughness-blurred sky
                // reflection). Same scene cube bound to every mesh.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 12: bump_detail_map (second high-frequency tangent-space normal,
                // HREK calc_bumpmap_detail_ps `bump.xy += detail.xy`). Default flat-normal when absent.
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 13: dedicated alpha_test_map (engine calc_alpha_test_ps clips on
                // alpha_test_map.a at 0.5, a map DISTINCT from the base). Bound to the base view when
                // the material authors no distinct alpha_test_map → fs_alphatest reads base.a.
                // Several lanes re-use the slot (change-colour mask, opacity map, halogram alpha mask).
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 14: material_texture (per-texel roughness/spec override,
                // cook_torrance_core.hlsl_include). Bound to a 1×1 grey view (a=1 → power_modifier=1
                // → no-op) when the material authors none.
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 15/16: bump_detail_map2 / bump_detail_map3 (the 2nd/3rd tangent-space
                // detail normals for the engine extended detail_blend / three_detail_blend variants,
                // extended_bump_mapping.hlsl_include:4,26). Bound to the flat-normal view (xy≈0 → adds
                // nothing) when the material authors none. 15 sampled textures on this bgl (the app
                // requests the full adapter limits, not the conservative 16-texture default).
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            // 16× anisotropy (requires linear min/mag/mip + a mip chain): tiled floors/walls/
            // detail maps viewed at grazing angles minification-alias into shimmer, which edge-AA
            // can't touch; 16× (the hardware max, auto-clamped to the device limit) samples along
            // the true texel footprint. Shared by ALL scene passes (mesh/decal/foliage bgl
            // binding 1, terrain bgl binding 5, water bgl binding 1).
            anisotropy_clamp: 16,
            ..Default::default()
        });

        // GPU lightmap sampler: the engine samples the lightmap atlas BILINEARLY
        // (g_sample_vmf_diffuse / DM textures through a linear sampler), top mip only, Repeat.
        // HMS_LM_NEAREST selects a nearest fetch (the CPU bake's decode_bitmap_keep_alpha at
        // mip 0 + nearest fetch; baked shadow edges then read as stair-stepped texels).
        let lm_nearest = std::env::var("HMS_LM_NEAREST").is_ok();
        let lm_filter = if lm_nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        let lm_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lm-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: lm_filter,
            min_filter: lm_filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            ..Default::default()
        });

        // 1x1 default texture bound in the water bind group's optional slots (watercolor /
        // slope / foam / global_shape) when the material authors none. new() has no queue, so
        // the texel is never written: wgpu zero-initialises it (the water shader's fallbacks key
        // off the WaterParams flags, not the texel). The view keeps the texture alive.
        let white = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white-1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_white = white.create_view(&Default::default());

        // HMS_DECO_CONTRAST="x,y": the engine's k_ps_decorators_contrast global (not recovered
        // from a capture; default 0.5/0.5).
        let (deco_cx, deco_cy) = { let v = std::env::var("HMS_DECO_CONTRAST").unwrap_or_else(|_| "0.5,0.5".into()); let mut it = v.split(',').map(|x| x.trim().parse::<f32>().unwrap_or(0.5)); let a = it.next().unwrap_or(0.5); let b = it.next().unwrap_or(0.5); (format!("{:.4}", a), format!("{:.4}", b)) };
        // Probe-GI declarations + lookup shared by the lit shaders (camera bindings 10-12).
        let gi_wgsl = format!("{}{}", crate::rtgi::GI_FRAGMENT_WGSL, crate::rtgi::GI_SHARED_WGSL).replace("__GI_DEBUG__", &std::env::var("HMS_RTGI_DEBUG").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0).to_string());
        // Print-only probe lanes (read back through HMS_HDR_DUMP): HMS_MDBG (mesh_shade / glass /
        // halogram terms), HMS_TDDBG (two-detail lane), HMS_H4_SHADEDBG (Halo 4 lane).
        let dbg_f = |var: &str| format!("{:.1}", std::env::var(var).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh-lit"),
            source: wgpu::ShaderSource::Wgsl(format!("{FOG_WGSL}{SHADOW_SAMPLE_WGSL}{gi_wgsl}{MESH_WGSL}{H4_MESH_WGSL}")
                .replace("__H4DBG__", &dbg_f("HMS_H4_SHADEDBG"))
                .replace("__TDDBG__", &dbg_f("HMS_TDDBG"))
                .replace("__MDBG__", &dbg_f("HMS_MDBG"))
                .replace("__DECO_CX__", &deco_cx).replace("__DECO_CY__", &deco_cy).into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-pl"),
            bind_group_layouts: &[camera_bgl, &material_bgl],
            push_constant_ranges: &[],
        });

        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 48, shader_location: 9 },
                // uv2 (submap-local lightmap UV), after sway (offset 60).
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 60, shader_location: 13 },
                // stored per-vertex tangent frame [T.xyz, handedness], after uv2 (offset 68).
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 68, shader_location: 22 },
            ],
        };
        // Per-instance model matrix (4 x vec4) + tint color, step_mode Instance.
        let inst = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 7 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 8 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 10 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 96, shader_location: 11 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 112, shader_location: 12 },
                // lm=[flag,hdr,k,mode] after aux (offset 128).
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 128, shader_location: 14 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 144, shader_location: 15 }, // spec2
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 160, shader_location: 16 }, // bump_xform
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 176, shader_location: 17 }, // fine_xform
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 192, shader_location: 18 }, // bump_detail_xform
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 208, shader_location: 19 }, // env_ctl [env_tint.rgb, specular_coefficient]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 224, shader_location: 20 }, // matmodel (material_model+1; 0=unset)
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 228, shader_location: 21 }, // si_ctl [mode, cmul.rgb]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 244, shader_location: 23 }, // mattex [black_rough, black_specmult, has_flag, mode]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 260, shader_location: 24 }, // mattex_xform [tile.xy, off.xy]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 276, shader_location: 25 }, // xbump [bd2.tile.xy, bd3.tile.xy]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 292, shader_location: 26 }, // xctl [bd2_present, bd3_present, env_authored, snorm bits]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 308, shader_location: 27 }, // obj_probe0 [dom_dir.xyz, bandwidth]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 324, shader_location: 28 }, // obj_probe1 [dom_rgb, mask]
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 340, shader_location: 29 }, // spec_rgb specular_tint
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 356, shader_location: 30 }, // fres_rgb fresnel_color
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x4, offset: 372, shader_location: 31 }, // obj_light packed [oct(fill_dir), oct(bounce_dir), bounce.rg, (bounce.b, flag)] — LAST free attribute (32/32)
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Halo winding varies
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
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
                    format: color_format,
                    // Opaque: BSP/object geometry is solid. Halo diffuse maps store
                    // specular/gloss in the alpha channel (NOT opacity), so blending
                    // on tex.a would make solid surfaces see-through. Genuinely
                    // transparent surfaces go through the dedicated water pass.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Alpha-test pipeline: identical to the opaque mesh pipeline but the
        // fragment discards fully-transparent texels (foliage cutout). Opaque
        // blend + depth-write so leaves occlude correctly.
        let alphatest_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-alphatest-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_alphatest"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Foliage (decorator grass/flowers) pipeline: same config as alphatest
        // (cutout, opaque, depth-write) but the MULTIPLICATIVE `fs_foliage` entry —
        // the engine decorator shader has no specular / white-sun / sky fill, so
        // decorators must NOT go through mesh_shade (see fs_foliage in MESH_WGSL).
        let foliage_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-foliage-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_foliage"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Blend pipeline: glass/translucent (alpha_blend / pre_multiplied_alpha
        // materials). Alpha-blended, depth-tested but NO depth-write so it
        // composites over the opaque scene behind it.
        let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-blend-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blend"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Non-decal transparent geometry that isn't straight-alpha needs the SAME
        // fixed-function blend states the engine's render_method BLEND_TYPE selects (premul /
        // multiply / double_multiply), not the default SrcAlpha/InvSrcAlpha. Mirror the
        // blend_pipeline config (fs entry over the lit-glass shader, depth-test LessEqual, NO
        // depth-write, cull none) but swap in the mode's blend state. These share the exact
        // blend-state VALUES used by the decal/sky pipelines below — routed per-mesh in
        // draw_transparent_sorted by GpuMesh.blend_mode. (The blend states themselves are
        // defined with the decal set further down; the pipelines are built there.)
        let make_blend = |label: &str, blend: wgpu::BlendState, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[vbl.clone(), inst.clone()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };

        // Additive/hologram pipeline: world/object surfaces with the additive
        // blend enum (blend==1) — holograms, force-fields, energy/plasma. One/One,
        // depth-tested but NO depth-write, cull none, fs_holo (emissive glow, alpha
        // ignored by the blend op). Reuses the mesh shader module + MeshVertex layout.
        let additive_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let additive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-additive-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_holo"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(additive_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // fs_particle pipelines (see draw_particles). Same vertex layout / depth state as
        // the additive pipeline; the fragment entry routes albedo option + blend per material.
        let make_particle = |label: &str, blend: wgpu::BlendState| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[vbl.clone(), inst.clone()], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: depth_format, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_particle"), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            multiview: None,
            cache: None,
        });
        let particle_add_pipeline = make_particle("particle-add-pipeline", wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::REPLACE,
        });
        let particle_alpha_pipeline = make_particle("particle-alpha-pipeline", wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::OVER,
        });
        let particle_premul_pipeline = make_particle("particle-premul-pipeline", wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::OVER,
        });
        let particle_multiply_pipeline = make_particle("particle-multiply-pipeline", wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::REPLACE,
        });

        // Solid-hologram pipeline: alpha-blended (SrcAlpha/OneMinusSrcAlpha) so OBJECT
        // markers (spawn Spartans, hill globes, kill/safe boundary shells) keep a readable
        // translucent silhouette at ALL angles — the additive glow washed out against bright
        // backgrounds (sky/snow) so the shape vanished. Same depth-state as additive (tested,
        // no write, cull none) but fs_holo_solid outputs alpha with a coverage floor + rim.
        let holo_solid_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let holo_solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-holo-solid-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_holo_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(holo_solid_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Sky pipelines: the Reach sky is a render_model (dome + cloud panels) that rides the
        // camera and is UNLIT (vert_color·albedo + self-illum). The engine draws it as a normal
        // render_model — opaque parts depth-test+WRITE, translucent parts depth-test without
        // write — so nearer panels win regardless of authored index. The sky is drawn in its own
        // pass whose depth is cleared before the world pass (lib.rs), so world geometry still
        // draws over every sky panel. HMS_SKY_LEGACY=1 selects the pre-audit path
        // (SKY_MESH_WGSL_LEGACY, depth Always/no-write, index-descending order; scene.rs builds
        // its meshes with build_sky_meshes_legacy).
        let legacy_sky = sky_legacy();
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky-mesh"),
            source: wgpu::ShaderSource::Wgsl((if legacy_sky { SKY_MESH_WGSL_LEGACY } else { SKY_MESH_WGSL }).into()),
        });
        let make_sky = |label: &str, blend: Option<wgpu::BlendState>, depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &sky_shader,
                    entry_point: Some("vs"),
                    buffers: &[vbl.clone(), inst.clone()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: depth_write && !legacy_sky,
                    depth_compare: if legacy_sky { wgpu::CompareFunction::Always } else { wgpu::CompareFunction::LessEqual },
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &sky_shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };
        let sky_opaque_pipeline = make_sky("sky-opaque-pipeline", None, true);
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let sky_blend_pipeline = make_sky("sky-blend-pipeline", Some(additive), false);
        // Multiply (mode 2/3): src·dst — DARKENS the sky behind (haze/overcast). double_multiply
        // (3) = the shader doubles rgb (engine BLEND_MULTIPLICATIVE 2.0).
        let multiply = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let sky_multiply_pipeline = make_sky("sky-multiply-pipeline", Some(multiply), false);
        // AlphaBlend (mode 4): src·a + dst·(1-a), a = albedo.w (engine ALPHA_CHANNEL_OUTPUT).
        let sky_alpha_pipeline = make_sky("sky-alpha-pipeline", Some(wgpu::BlendState::ALPHA_BLENDING), false);
        // pre_multiplied_alpha (mode 6; 8 routed here too): src + dst·(1-a).
        let sky_premul_pipeline = make_sky("sky-premul-pipeline", Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false);

        // Decal pipelines: UNLIT (fs_decal, no spec/fog), depth-test LessEqual + NO depth
        // write, cull NONE (decal quads are double-sided), per-decal blend. Three buckets
        // matching the decs enum: alpha, additive, multiply (multiply DARKENS the ground —
        // scorch/AO decals). Uses MESH_WGSL's shader module + MeshVertex/instance layout.
        let make_decal = |label: &str, blend: wgpu::BlendState, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_decal"), // camera-ward nudge (see vs_decal)
                    buffers: &[vbl.clone(), inst.clone()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    // No hardware depth bias: a camera-ward bias lets decals win the depth test
                    // against geometry genuinely IN FRONT of them. Z-fighting on a decal's own
                    // coplanar surface is handled in WORLD space (vs_decal's nudge + the `fit.n *
                    // bias` lift in scene.rs).
                    bias: wgpu::DepthBiasState { constant: 0, slope_scale: 0.0, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };
        // Decal-specific additive: Src=SrcAlpha, Dst=One (masked-out texels add nothing;
        // the shared `additive` above is Src=One which would glow on alpha-edge fringe).
        let decal_additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        // Pre-multiplied alpha (decs enum 10 + rerouted diffuse_plus_alpha): One/(1-SrcAlpha).
        let decal_premul = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        // Inverse alpha blend (decs enum 9): src·(1-a) + dst·a.
        let decal_invalpha = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                dst_factor: wgpu::BlendFactor::SrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let decal_alpha_pipeline = make_decal("decal-alpha-pipeline", wgpu::BlendState::ALPHA_BLENDING, "fs_decal");
        let decal_additive_pipeline = make_decal("decal-additive-pipeline", decal_additive, "fs_decal");
        let decal_multiply_pipeline = make_decal("decal-multiply-pipeline", multiply, "fs_decal_multiply");
        // double_multiply (decs enum 4): 2·dst·src (shared/blend.hlsl_include DEST_COLOR/
        // SRC_COLOR). result = src·Dst + dst·Src = 2·(src·dst). Uses fs_decal_multiply.
        let double_multiply = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let decal_double_multiply_pipeline = make_decal("decal-double-multiply-pipeline", double_multiply, "fs_decal_multiply");
        let decal_premul_pipeline = make_decal("decal-premul-pipeline", decal_premul, "fs_decal_premul");
        let decal_invalpha_pipeline = make_decal("decal-invalpha-pipeline", decal_invalpha, "fs_decal");
        // SDF "vector" decals (numbers/text/glyphs). base_map is a flat colour swatch;
        // the glyph SILHOUETTE is a signed-distance field in the vector_map (bound via the
        // emissive slot), sampled from .g with an fwidth-based antialiased threshold. Without
        // this the flat swatch fills the whole quad → a plain coloured rectangle.
        let decal_vector_pipeline = make_decal("decal-vector-pipeline", wgpu::BlendState::ALPHA_BLENDING, "fs_decal_vector");

        // Non-decal transparent geometry, sharing the decal/sky fixed-function blend states
        // above (same `decal_premul` One/INV_SRC_ALPHA, `multiply` ZERO/SRC_COLOR,
        // `double_multiply` DST_COLOR/SRC_COLOR). Premul uses fs_blend_premul (rgb premultiplied
        // in-shader, engine convert_to_render_target_premultiplied_alpha). Selected per-mesh in
        // draw_transparent_sorted by GpuMesh.blend_mode. There is no non-decal inv_alpha route:
        // the BSP/model render_method BLEND_TYPE enum (0 opaque,1 add,2 mul,3 dbl_mul,4 alpha,
        // 5 add_src×srca,6 premul,7 max,8 add_src×dsta) has no inverse_alpha_blend ordinal.
        let blend_premul_pipeline = make_blend("mesh-blend-premul-pipeline", decal_premul, "fs_blend_premul");
        // multiplicative transparents are UNLIT in the engine (fs_blend_multiplicative).
        let blend_multiply_pipeline = make_blend("mesh-blend-multiply-pipeline", multiply, "fs_blend_multiplicative");
        let blend_double_multiply_pipeline = make_blend("mesh-blend-double-multiply-pipeline", double_multiply, "fs_blend_multiplicative");

        // Shadow (depth-only) pipeline: renders casters from the sun POV into the
        // shadow depth map. No colour target; front-face cull reduces acne. Uses
        // the shadow bind group (light_view_proj at group0 binding 2).
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-depth"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_WGSL.into()),
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-pl"),
            bind_group_layouts: &[shadow_bgl],
            push_constant_ranges: &[],
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                // Constant + slope-scaled depth bias in hardware to fight shadow acne.
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview: None,
            cache: None,
        });
        // The Halo 4 caster pipeline: the engine's forge-lightmap / floating-shadow
        // depth passes draw the casters' BACK faces (halo4.dll sub_1803443DC cull mode 0xC) with no
        // hardware bias - the `shadow >= z + bias` receiver test of the apply shaders (a lit surface
        // sees its own back face further along the light than itself) only works that way round.
        let shadow_pipeline_h4 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-pipeline-h4"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs"),
                buffers: &[vbl.clone(), inst.clone()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 0, slope_scale: 0.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview: None,
            cache: None,
        });

        // Water pipeline: same layouts, WATER_WGSL shader (animated wave normals +
        // watercolor tex + fresnel alpha + sun spec), premultiplied-alpha blended, no
        // depth-write so it composites over the opaque scene. HMS_WDBG = print-only probe lane.
        let water_src = format!("{FOG_WGSL}{gi_wgsl}{WATER_WGSL}")
            .replace("__WDBG__", &std::env::var("HMS_WDBG").unwrap_or_else(|_| "0".into()));
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water"),
            source: wgpu::ShaderSource::Wgsl(water_src.into()),
        });
        let wvbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 48, shader_location: 9 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 60, shader_location: 13 }, // uv2
            ],
        };
        let winst = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 7 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 8 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 10 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 96, shader_location: 11 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 112, shader_location: 12 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 128, shader_location: 14 }, // lm
            ],
        };
        let wtex_e = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // Water bind group: watercolor tex (0) + sampler (1) + per-material water
        // params uniform (2) so waves/colors/fresnel/murk match the authored tag.
        let water_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-bgl"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: wave_slope_array slice-0 (RG slope map) driving the
                // animated wave normal. Falls back to the 1×1 white default when
                // the material authors no slope array (shader uses noise then).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: foam_texture (RGB × A) sampled at the wave xform. Default
                // 1×1 white → foam_color = white, factor gated by wave_choppiness (harmless when
                // foam_cut>=1 disables auto-foam). Bound in place of default_white when authored.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 10/11: the environment CUBE for the water REFLECTION (the authored
                // environment_map, else the captured per-map sky dome) + its sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 12: global_shape_texture. Its .a channel is the from_shape
                // bank_alpha (shore/coverage mask); default 1×1 white (a=1 → full coverage, no
                // change) when the material authors none. Sampled at the wave UV.
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 13-17: per-pixel lightmap atlas — DM, SDM slice0, lm sampler, SDM slice1, SDM slice2
                // (same set as terrain 19-23 / material 5-6). The water FS dual-VMF-decodes them at
                // uv2 (w_lightmap_tint) so the baked lighting comes from the SAME per-pixel atlas the
                // opaque BSP uses, not the airprobe grid (whose nearest probes are outdoors for a cave
                // puddle → blown-out white). Grey/black 1×1 fallbacks + lm.x=0 when no atlas is bound.
                wtex_e(13), wtex_e(14),
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wtex_e(16), wtex_e(17),
            ],
        });
        let water_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water-pl"),
            bind_group_layouts: &[camera_bgl, &water_bgl],
            push_constant_ranges: &[],
        });
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water-pipeline"),
            layout: Some(&water_layout),
            vertex: wgpu::VertexState {
                module: &water_shader,
                entry_point: Some("vs"),
                buffers: &[wvbl, winst],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // transparent — test but don't occlude
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &water_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // engine compose adds water_color on top of the refraction lerp → src is premultiplied, dst weight = (1-fresnel)·visibility
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Terrain pipeline: 4 base maps (0-3) + blend mask (4) + sampler (5) +
        // per-layer tile uniform (6), blended per-pixel by the blend RGBA weights.
        let tex_e = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-bgl"),
            entries: &[
                tex_e(0), tex_e(1), tex_e(2), tex_e(3), tex_e(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7-10: per-layer detail maps. 11-14: per-layer bump (normal) maps.
                tex_e(7), tex_e(8), tex_e(9), tex_e(10),
                tex_e(11), tex_e(12), tex_e(13), tex_e(14),
                // 15-18: per-layer detail_bump (2nd normal) maps. 21 sampled textures total —
                // within adapter.limits() (this app requests the full adapter limits, not the
                // conservative 16-tex default).
                tex_e(15), tex_e(16), tex_e(17), tex_e(18),
                // 19-23: lightmap atlas DM, SDM slice0, lm sampler, SDM slice1, SDM slice2.
                tex_e(19), tex_e(20),
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_e(22), tex_e(23),
            ],
        });
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(format!("{FOG_WGSL}{SHADOW_SAMPLE_WGSL}{gi_wgsl}{TERRAIN_WGSL}").replace("__TDBG__", &std::env::var("HMS_TDBG").unwrap_or_else(|_| "0".into())).into()),
        });
        let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pl"),
            bind_group_layouts: &[camera_bgl, &terrain_bgl],
            push_constant_ranges: &[],
        });
        let tvbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 48, shader_location: 9 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 60, shader_location: 13 }, // uv2
            ],
        };
        let tinst = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 7 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 8 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 10 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 96, shader_location: 11 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 112, shader_location: 12 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 128, shader_location: 14 }, // lm
            ],
        };
        let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipeline"),
            layout: Some(&terrain_layout),
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs"),
                buffers: &[tvbl, tinst],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // Terrain is opaque — the shader_terrain diffuse alpha is a spec
                    // mask, not coverage (HREK terrain_new.hlsl_include).
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self { pipeline, alphatest_pipeline, foliage_pipeline, blend_pipeline, blend_premul_pipeline, blend_multiply_pipeline, blend_double_multiply_pipeline, additive_pipeline, particle_add_pipeline, particle_alpha_pipeline, particle_premul_pipeline, particle_multiply_pipeline, holo_solid_pipeline, water_pipeline, terrain_pipeline, sky_opaque_pipeline, sky_blend_pipeline, sky_multiply_pipeline, sky_alpha_pipeline, sky_premul_pipeline, decal_alpha_pipeline, decal_additive_pipeline, decal_multiply_pipeline, decal_double_multiply_pipeline, decal_premul_pipeline, decal_invalpha_pipeline, decal_vector_pipeline, shadow_pipeline, shadow_pipeline_h4, material_bgl, terrain_bgl, water_bgl, sampler, lm_sampler, default_white, fallbacks: std::sync::OnceLock::new(), detail_baker: std::sync::OnceLock::new() }
    }

    /// Build a material bind group from a base texture view + an emissive view.
    /// Pass a 1x1 black view as `emissive` to disable self-illum.
    pub fn material_from_view(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        emissive: &wgpu::TextureView,
        detail: &wgpu::TextureView,
        bump: &wgpu::TextureView,
        // GPU-lightmap DM/SDM atlas views (dummy 1×1 for non-atlas meshes). sdm = slice 0;
        // sdm1/sdm2 = SDM z-slices 1/2 for the full dual-VMF decode (dummy black off-atlas).
        dm: &wgpu::TextureView,
        sdm: &wgpu::TextureView,
        sdm1: &wgpu::TextureView,
        sdm2: &wgpu::TextureView,
        // environment cubemap + sampler (the captured scene cube, or an authored cube).
        env_cube: &wgpu::TextureView,
        env_samp: &wgpu::Sampler,
        // second normal (bump_detail_map); flat-normal view when absent.
        bump_detail: &wgpu::TextureView,
        // dedicated alpha_test_map; pass the base `view` when the material authors none.
        alpha_test: &wgpu::TextureView,
        // material_texture (per-texel roughness/spec); pass a 1×1 grey view (a=1 → no-op)
        // when the material authors none.
        material_texture: &wgpu::TextureView,
        // bump_detail_map2 / bump_detail_map3 (extended detail_blend / three_detail_blend);
        // pass the flat-normal view when the material authors none (xy≈0 → the blend adds nothing).
        bump_detail2: &wgpu::TextureView,
        bump_detail3: &wgpu::TextureView,
    ) -> Material {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material-bg"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(emissive) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(detail) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(bump) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(dm) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(sdm) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.lm_sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(sdm1) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(sdm2) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(env_cube) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(env_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(bump_detail) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(alpha_test) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(material_texture) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(bump_detail2) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(bump_detail3) },
            ],
        });
        Material { bind_group }
    }

    /// The lazily-built GPU base×detail baker (see bake.rs). Call from the load
    /// worker's bake pass with the same device/queue used for uploads.
    pub fn detail_baker(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> &crate::bake::DetailBaker {
        self.detail_baker.get_or_init(|| crate::bake::DetailBaker::new(device, queue))
    }

    pub fn default_material(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Material {
        // Neutral fallback for a mesh whose diffuse failed to resolve: 1x1 white
        // base + black emissive + mid-gray detail + flat-normal bump.
        let (view, _tex) = upload_texture_rgba(device, queue, &[255, 255, 255, 255], 1, 1);
        let (emis, _etex) = upload_texture_rgba(device, queue, &[0, 0, 0, 255], 1, 1);
        let (det, _dtex) = upload_texture_rgba(device, queue, &[128, 128, 128, 255], 1, 1);
        let (bmp, _btex) = upload_texture_rgba(device, queue, &[128, 128, 255, 255], 1, 1);
        // dummy lightmap DM/SDM (lm.flag=0 → never sampled).
        let fb = self.fallbacks.get_or_init(|| FallbackTex::new(device, queue));
        self.material_from_view(device, &view, &emis, &det, &bmp, &fb.grey, &fb.black, &fb.black, &fb.black, &fb.env_cube, &fb.env_samp, &fb.flat, &view, &fb.grey, &fb.flat, &fb.flat)
    }

    /// Upload a mesh with an optional texture view (None → default white).
    pub fn upload_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[MeshVertex],
        indices: &[u32],
        instances: &[Mat4],
        colors: Option<&[[f32; 4]]>,
        texture_view: Option<&wgpu::TextureView>,
        emissive_view: Option<&wgpu::TextureView>,
        detail_view: Option<&wgpu::TextureView>,
        bump_view: Option<&wgpu::TextureView>,
        bump_detail_view: Option<&wgpu::TextureView>,
        scroll: [f32; 2],
        // Per-material base-map UV tile (engine `base_map` xform scale, from
        // ZH_BSP_GetMaterialDiffuseTiling). (1,1) = no tiling. Packed into scroll.zw
        // and multiplied into the base UV in MESH_WGSL — fixes stretched opaque
        // BSP surfaces (boardwalk floor etc.) that authored a tile > 1.
        tile: [f32; 2],
        // Per-material opaque detail-map UV xform [tile.x, tile.y, off.x, off.y] from the
        // authored `detail_map` rmt2 constant. [0;4] → the MESH_DETAIL_SCALE=6.0 fallback.
        detail_xform: [f32; 4],
        // Per-mesh UV density (uv-units per world-unit); carried in aux.x.
        uv_density: f32,
        // 1.0 when the diffuse alpha is a REAL specular/gloss mask (Bc3 with varying alpha);
        // 0.0 when it has no usable alpha (Bc1/opaque-stamped → sampled a=1.0). Gates
        // spec_mask so matte surfaces (sandy ground) don't reflect the blue sky as sparkle.
        spec_from_alpha: f32,
        // Per-material specular from rmt2 REAL constants — [analytical_spec_strength, roughness].
        // spec>0 → the shader drives an analytical + env lobe from these constants (also on
        // alpha-less BC1 diffuse). [0.0, -1.0] = material authored no spec constant → the shader
        // keeps the diffuse-alpha gate and an inverse-luminance roughness stand-in.
        mat_pbr: [f32; 2],
        // Artist-Fresnel constants [metalness, fresnel_steepness, specular_tint_luma,
        // fresnel_color_luma] from the rmt2 REALs. [0;4] → the fixed-dielectric Schlick path.
        mat_spec2: [f32; 4],
        // Authored bump_map UV xform [tile.x, tile.y, off.x, off.y]. [0;4] → the bump is sampled
        // at the base auv. Only opaque BSP materials pass a real value.
        bump_xform: [f32; 4],
        // FINE self-detail xform [tile.x, tile.y, flag, _]. flag>0.5 → sample base_tex at
        // uv*tile and multiply as crisp live grain (concrete tile floors). [0;4] → no second layer.
        fine_xform: [f32; 4],
        // Second normal (bump_detail_map) xform [tile.xy, flag, _]. flag>0.5 → shader adds it
        // to the base bump (HREK calc_bumpmap_detail_ps). [0;4] → single bump. View passed above.
        bump_detail_xform: [f32; 4],
        // Env control [env_tint_color.rgb, specular_coefficient]. rgb==0 → the shader
        // hue-normalises by sun_tint; w==0 → factor 1.0. Only the opaque BSP path passes a real value.
        env_ctl: [f32; 4],
        // Some → sample the DM/SDM atlas at uv2 + dual-VMF-decode in the fragment shader.
        // None → dummy DM/SDM, lm.flag=0, the shader uses the per-vertex baked i.tint.
        lightmap: Option<LightmapInputs>,
        // true for dynamic forge/scenario objects (see upload_mesh_prebuilt).
        is_object: bool,
        // Per-shader material_model dispatch, ENCODED as (material_model + 1). 0.0 = unset →
        // the cook_torrance path (every non-BSP caller passes 0.0). See InstanceRaw.matmodel.
        matmodel: f32,
        // Self-illum routing [mode, cmul.rgb]; [0;4] = no routing. See InstanceRaw.si_ctl.
        si_ctl: [f32; 4],
        // Dedicated alpha_test_map view (None → base.a fallback). See upload_mesh_prebuilt.
        alpha_test_view: Option<&wgpu::TextureView>,
        // material_texture view + control [black_roughness, black_specmult, has_flag, mode] + UV
        // xform. None/[0;4]/[0;4] (every non-BSP caller) → the per-texel override is inert. See
        // upload_mesh_prebuilt.
        mattex_view: Option<&wgpu::TextureView>,
        mattex_ctl: [f32; 4],
        mattex_xform: [f32; 4],
        // bump_detail_map2 / bump_detail_map3 views + [bd2.tile.xy, bd3.tile.xy] + control
        // [bd2_present, bd3_present, env_authored, snorm bits]. None/[0;4] → single bump.
        bump_detail2_view: Option<&wgpu::TextureView>,
        bump_detail3_view: Option<&wgpu::TextureView>,
        xbump: [f32; 4],
        xctl: [f32; 4],
        // Authored per-material environment_map cube (opaque per_pixel). None → binding 10
        // falls back to the captured sky cube (env_authored stays 0 → no decode change).
        env_cube_override: Option<&wgpu::TextureView>,
    ) -> GpuMesh {
        use wgpu::util::DeviceExt;
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-vb"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-ib"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        // Account the geometry bytes here; the instance bytes + mesh count are tallied
        // inside upload_mesh_prebuilt (both entry points converge there).
        memprof::add_buf((std::mem::size_of_val(verts) + std::mem::size_of_val(indices)) as u64);
        memprof::note("buf:mesh-vb/ib", (std::mem::size_of_val(verts) + std::mem::size_of_val(indices)) as u64);
        self.upload_mesh_prebuilt(
            device, queue, vbuf, ibuf, indices.len() as u32, instances, colors, texture_view,
            emissive_view, detail_view, bump_view, bump_detail_view, scroll, tile, detail_xform, uv_density,
            spec_from_alpha, mat_pbr, mat_spec2, bump_xform, fine_xform, bump_detail_xform, env_ctl, lightmap,
            is_object, matmodel, si_ctl, alpha_test_view, mattex_view, mattex_ctl, mattex_xform,
            bump_detail2_view, bump_detail3_view, xbump, xctl, env_cube_override,
        )
    }

    /// Build a GpuMesh from ALREADY-CREATED vertex/index buffers (created in parallel inside
    /// `decode_bsp_mesh`), so the serial finalize pays only for the small instance buffer +
    /// material bind group — the two big `create_buffer_init` calls (the dominant ~2/3 of the
    /// serial per-mesh upload cost) move off the critical path onto the rayon decode. The
    /// resulting GpuMesh is identical to `upload_mesh`'s. All non-BSP callers call
    /// `upload_mesh` (which builds the buffers, then delegates here).
    pub fn upload_mesh_prebuilt(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vbuf: wgpu::Buffer,
        ibuf: wgpu::Buffer,
        index_count: u32,
        instances: &[Mat4],
        colors: Option<&[[f32; 4]]>,
        texture_view: Option<&wgpu::TextureView>,
        emissive_view: Option<&wgpu::TextureView>,
        detail_view: Option<&wgpu::TextureView>,
        bump_view: Option<&wgpu::TextureView>,
        bump_detail_view: Option<&wgpu::TextureView>,
        scroll: [f32; 2],
        tile: [f32; 2],
        detail_xform: [f32; 4],
        uv_density: f32,
        spec_from_alpha: f32,
        mat_pbr: [f32; 2],
        mat_spec2: [f32; 4],
        bump_xform: [f32; 4],
        fine_xform: [f32; 4],
        bump_detail_xform: [f32; 4],
        // env control [env_tint_color.rgb, specular_coefficient]; [0;4] = no-op.
        env_ctl: [f32; 4],
        lightmap: Option<LightmapInputs>,
        // true for DYNAMIC forge/scenario OBJECTS (not BSP). Objects are lit by the
        // per-position airprobe ambient (baked into i.tint) but carry NO baked directional
        // term, so the shader adds a dominant-lobe geometric-normal soft-shade for them —
        // gated by the lm.w==2 sentinel below so per-vertex-baked BSP (which HAS baked
        // directional) is never double-lit.
        is_object: bool,
        // per-shader material_model dispatch, ENCODED as (material_model + 1). 0.0 = unset →
        // the cook_torrance path (all non-BSP callers pass 0.0). See InstanceRaw.matmodel.
        matmodel: f32,
        // self-illum routing [mode, cmul.rgb]; [0;4] = no routing. See InstanceRaw.si_ctl.
        si_ctl: [f32; 4],
        // dedicated alpha_test_map view. None → binding 13 falls back to the base view
        // so fs_alphatest clips on base.a (every non-BSP caller passes None).
        alpha_test_view: Option<&wgpu::TextureView>,
        // material_texture view + control. mattex_view None → binding 14 falls back to a
        // 1×1 grey view (a=1 → power_modifier=1 → no-op). mattex_ctl = [black_roughness,
        // black_specular_multiplier, has_material_texture flag, mode]; mattex_xform = UV xform.
        // [0;4]/[0;4]/None (every non-BSP caller) → the per-texel override is fully inert.
        mattex_view: Option<&wgpu::TextureView>,
        mattex_ctl: [f32; 4],
        mattex_xform: [f32; 4],
        // bump_detail_map2 / bump_detail_map3 views + tiles + control (see upload_mesh).
        bump_detail2_view: Option<&wgpu::TextureView>,
        bump_detail3_view: Option<&wgpu::TextureView>,
        xbump: [f32; 4],
        xctl: [f32; 4],
        // authored environment_map cube (opaque per_pixel); None → the captured sky cube.
        env_cube_override: Option<&wgpu::TextureView>,
    ) -> GpuMesh {
        use wgpu::util::DeviceExt;
        // per-mesh lightmap scalars — [flag, hdr, k, mode]. flag=1 activates the GPU atlas
        // decode in mesh_shade; 0 keeps the baked i.tint. lm.w=2 flags a dynamic object (no
        // atlas, no baked directional → the shader adds the sun soft-shade).
        let lm = match &lightmap {
            Some(l) => [1.0, l.hdr, l.k, l.mode],
            None => [0.0, 0.0, 0.0, if is_object { 2.0 } else { 0.0 }],
        };
        let inst_data: Vec<InstanceRaw> = instances
            .iter()
            .enumerate()
            .map(|(i, m)| InstanceRaw {
                model: m.to_cols_array_2d(),
                color: colors.and_then(|c| c.get(i)).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                scroll: [scroll[0], scroll[1], tile[0], tile[1]],
                detail_xform,
                aux: [uv_density, spec_from_alpha, mat_pbr[0], mat_pbr[1]],
                lm,
                spec2: mat_spec2,
                bump_xform,
                fine_xform,
                bump_detail_xform,
                env_ctl,
                matmodel,
                si_ctl,
                mattex: mattex_ctl,
                mattex_xform,
                xbump,
                xctl,
                obj_probe0: [0.0; 4],
                obj_probe1: [0.0; 4],
                spec_rgb: [0.0; 4],
                fres_rgb: [0.0; 4],
                obj_light: [0; 4],
            })
            .collect();
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-inst"),
            contents: bytemuck::cast_slice(&inst_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        memprof::add_buf((inst_data.len() * std::mem::size_of::<InstanceRaw>()) as u64);
        memprof::note("buf:mesh-inst", (inst_data.len() * std::mem::size_of::<InstanceRaw>()) as u64);
        let inst_raw_keep = inst_data.clone();
        memprof::add_mesh();
        let material = match texture_view {
            Some(v) => {
                // constant 1×1 fallbacks (emissive off / detail neutral ×4.59 / bump flat)
                let fb = self.fallbacks.get_or_init(|| FallbackTex::new(device, queue));
                let e = emissive_view.unwrap_or(&fb.black);
                let d = detail_view.unwrap_or(&fb.grey);
                let b = bump_view.unwrap_or(&fb.flat);
                // real DM/SDM when this mesh is GPU-lit, else dummy (lm.flag=0).
                let dm = lightmap.as_ref().map(|l| &l.dm).unwrap_or(&fb.grey);
                let sdm = lightmap.as_ref().map(|l| &l.sdm).unwrap_or(&fb.black);
                // SDM slices 1/2 for the dual-VMF decode; slice 0 when a mesh only supplied one
                // slice, black off-atlas.
                let sdm1 = lightmap.as_ref().and_then(|l| l.sdm1.as_ref()).unwrap_or(sdm);
                let sdm2 = lightmap.as_ref().and_then(|l| l.sdm2.as_ref()).unwrap_or(sdm);
                let bd = bump_detail_view.unwrap_or(&fb.flat);
                // 2nd/3rd detail-bump maps, or the flat-normal no-op default (xy≈0).
                let bd2 = bump_detail2_view.unwrap_or(&fb.flat);
                let bd3 = bump_detail3_view.unwrap_or(&fb.flat);
                // the authored env cube when present, else the captured sky cube.
                let ec = env_cube_override.unwrap_or(&fb.env_cube);
                // dedicated alpha_test_map when authored & distinct; else the base view
                // (fs_alphatest then reads base.a).
                let at = alpha_test_view.unwrap_or(v);
                // bound material_texture, or the 1×1 grey no-op default (a=1).
                let mt = mattex_view.unwrap_or(&fb.grey);
                self.material_from_view(device, v, e, d, b, dm, sdm, sdm1, sdm2, ec, &fb.env_samp, bd, at, mt, bd2, bd3)
            }
            None => self.default_material(device, queue),
        };
        // Centroid = mean instance translation (back-to-front sort of transparent meshes).
        let centroid = {
            let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
            for m in instances { let t = m.w_axis; cx += t.x; cy += t.y; cz += t.z; }
            let n = instances.len().max(1) as f32;
            [cx / n, cy / n, cz / n]
        };
        GpuMesh {
            casts_shadow: true,
            h4_caster: false,
            is_terrain: false,
            inst_raw: inst_raw_keep,
            vbuf,
            ibuf,
            index_count,
            instance_buf,
            instance_count: instances.len().max(1) as u32,
            material,
            centroid,
            deco_weight: 0,
            blend_mode: 0,
        }
    }

    // The static-list draw functions (draw / draw_alphatest / draw_foliage / draw_additive /
    // draw_water / draw_terrain / draw_decals) are generic over `wgpu::util::RenderEncoder`, so
    // the SAME code records into a live RenderPass (per-frame lists) or into a
    // RenderBundleEncoder (SceneRenderer's cached bundles for the static BSP lists). Encoding
    // ~9000 draws through wgpu-core's validated pass path costs 15.8 ms of CPU per frame on
    // Panopticon; a bundle replays them as raw hal calls.
    pub fn draw<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        for m in meshes {
            if m.is_terrain { continue; } // drawn by draw_terrain
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Draw alpha-test (foliage cutout) meshes through the discard pipeline.
    pub fn draw_alphatest<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
    ) {
        pass.set_pipeline(&self.alphatest_pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        for m in meshes {
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Draw decorator (grass/flower) group-meshes through the multiplicative foliage pipeline
    /// (engine decorator model — no specular/sky-fill; see fs_foliage), with the engine's
    /// DECIMATION: at runtime the engine sorts decorator groups by camera distance and draws
    /// nearest-first up to a total instance BUDGET, dropping the rest (haloreach.dll
    /// sub_1806F9FAC). Each `GpuMesh` is ONE per-cluster group with a centroid + `deco_weight`
    /// (instance count): sort by distance², walk nearest-first, stop once the budget is spent.
    ///
    /// `budget` == 0 disables culling (draw all; the default, see HMS_DECO_BUDGET).
    pub fn draw_foliage<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
        eye: [f32; 3],
        budget: u32,
    ) {
        pass.set_pipeline(&self.foliage_pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        let d2 = |m: &GpuMesh| -> f32 {
            let dx = m.centroid[0] - eye[0];
            let dy = m.centroid[1] - eye[1];
            let dz = m.centroid[2] - eye[2];
            dx * dx + dy * dy + dz * dz
        };
        // Nearest-first order over the group-meshes (index sort — cheap, no GPU churn).
        let mut order: Vec<usize> = (0..meshes.len()).collect();
        order.sort_by(|&a, &b| d2(&meshes[a]).partial_cmp(&d2(&meshes[b])).unwrap_or(std::cmp::Ordering::Equal));
        let mut spent: u32 = 0;
        for &i in &order {
            let m = &meshes[i];
            // budget==0 → unlimited. Otherwise stop once the nearest groups have used it.
            if budget != 0 && spent >= budget {
                break;
            }
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
            spent = spent.saturating_add(m.deco_weight.max(1));
        }
    }

    /// Draw additive/hologram meshes (blend==1: holograms/force-fields/energy) through
    /// the additive pipeline (One/One, fs_holo emissive glow). Transparent pass.
    pub fn draw_additive<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
    ) {
        pass.set_pipeline(&self.additive_pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        for m in meshes {
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Draw the per-frame particle batches back-to-front with the pipeline their
    /// GpuMesh.blend_mode routes to (4 additive, 1 premultiplied, 2 multiply, else straight alpha).
    pub fn draw_particles<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
        cam_pos: [f32; 3],
    ) {
        pass.set_bind_group(0, camera_bg, &[]);
        let d2 = |c: [f32; 3]| { let (dx, dy, dz) = (c[0] - cam_pos[0], c[1] - cam_pos[1], c[2] - cam_pos[2]); dx * dx + dy * dy + dz * dz };
        let mut order: Vec<usize> = (0..meshes.len()).collect();
        order.sort_by(|&a, &b| d2(meshes[b].centroid).partial_cmp(&d2(meshes[a].centroid)).unwrap_or(std::cmp::Ordering::Equal));
        for &idx in &order {
            let m = &meshes[idx];
            let pipe = match m.blend_mode {
                4 => &self.particle_add_pipeline,
                1 => &self.particle_premul_pipeline,
                2 | 3 => &self.particle_multiply_pipeline,
                _ => &self.particle_alpha_pipeline,
            };
            pass.set_pipeline(pipe);
            pass.set_bind_group(1, &m.material.bind_group, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Draw several transparent lists as ONE back-to-front sequence. Each list carries the
    /// pipeline family it belongs to (`TransparentKind`); within it a Blend mesh still routes by its
    /// `blend_mode`. Multi-instance meshes (one GpuMesh per object tag) are split into per-instance
    /// draws sorted by their own translation, so a marker in FRONT of a glass pane draws after the
    /// pane even though every pane of that tag shares one mesh. Single-instance meshes sort by
    /// their centroid. Depth is tested (loaded) but never written by any of these pipelines, so the
    /// draw order alone decides the composite.
    pub fn draw_transparent_sorted<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_bg: &'a wgpu::BindGroup,
        lists: &[(&'a [GpuMesh], TransparentKind)],
        cam_pos: [f32; 3],
    ) {
        let d2 = |c: [f32; 3]| {
            let (dx, dy, dz) = (c[0] - cam_pos[0], c[1] - cam_pos[1], c[2] - cam_pos[2]);
            dx * dx + dy * dy + dz * dz
        };
        // (list, mesh, instance or u32::MAX = all instances, distance²)
        let mut items: Vec<(usize, usize, u32, f32)> = Vec::new();
        for (li, (meshes, _)) in lists.iter().enumerate() {
            for (mi, m) in meshes.iter().enumerate() {
                if m.instance_count > 1 && m.inst_raw.len() as u32 == m.instance_count {
                    for (ii, r) in m.inst_raw.iter().enumerate() {
                        items.push((li, mi, ii as u32, d2([r.model[3][0], r.model[3][1], r.model[3][2]])));
                    }
                } else {
                    items.push((li, mi, u32::MAX, d2(m.centroid)));
                }
            }
        }
        if items.is_empty() {
            return;
        }
        items.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        pass.set_bind_group(0, camera_bg, &[]);
        let mut cur: *const wgpu::RenderPipeline = std::ptr::null();
        for (li, mi, ii, _) in items {
            let (meshes, kind) = &lists[li];
            let m = &meshes[mi];
            let pipe: &wgpu::RenderPipeline = match kind {
                TransparentKind::HoloSolid => &self.holo_solid_pipeline,
                TransparentKind::Additive => &self.additive_pipeline,
                TransparentKind::Blend => match m.blend_mode {
                    1 => &self.blend_premul_pipeline,
                    2 => &self.blend_multiply_pipeline,
                    3 => &self.blend_double_multiply_pipeline,
                    _ => &self.blend_pipeline,
                },
            };
            if !std::ptr::eq(pipe, cur) {
                pass.set_pipeline(pipe);
                cur = pipe;
            }
            pass.set_bind_group(1, &m.material.bind_group, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            if ii == u32::MAX {
                pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
            } else {
                pass.draw_indexed(0..m.index_count, 0, ii..ii + 1);
            }
        }
    }

    /// Draw water meshes (shader_water surfaces) through the water pipeline.
    pub fn draw_water<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
    ) {
        pass.set_pipeline(&self.water_pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        for m in meshes {
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Upload a terrain-blend mesh: 4 base colour maps + an RGBA blend mask +
    /// per-layer UV tiling. Rendered (identity instance) through the terrain
    /// pipeline which blends the 4 layers by the mask weights.
    pub fn upload_terrain_mesh(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        verts: &[MeshVertex],
        indices: &[u32],
        base_views: [&wgpu::TextureView; 4],
        detail_views: [&wgpu::TextureView; 4],
        blend_view: &wgpu::TextureView,
        bump_views: [&wgpu::TextureView; 4],
        // per-layer detail_bump (2nd normal) views (grey 1×1 when absent).
        detail_bump_views: [&wgpu::TextureView; 4],
        params: &TerrainMaterialParams,
        // Some → per-pixel DM/SDM atlas lighting (lm.flag=1); None → per-vertex v.color.
        lightmap: Option<LightmapInputs>,
        model: Mat4, // placement matrix (BSP terrain passes IDENTITY; terrain-shaded objects their transform)
    ) -> GpuMesh {
        use wgpu::util::DeviceExt;
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-vb"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-ib"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        memprof::note("buf:terrain-vb/ib", (std::mem::size_of_val(verts) + std::mem::size_of_val(indices)) as u64);
        let tlm = match &lightmap { Some(l) => [1.0, l.hdr, l.k, l.mode], None => [0.0; 4] };
        let inst = InstanceRaw { model: model.to_cols_array_2d(), color: [1.0; 4], scroll: [0.0; 4], detail_xform: [0.0; 4], aux: [0.0; 4], lm: tlm, spec2: [0.0; 4], bump_xform: [0.0; 4], fine_xform: [0.0; 4], bump_detail_xform: [0.0; 4], env_ctl: [0.0; 4], matmodel: 0.0, si_ctl: [0.0; 4], mattex: [0.0; 4], mattex_xform: [0.0; 4], xbump: [0.0; 4], xctl: [0.0; 4], obj_probe0: [0.0; 4], obj_probe1: [0.0; 4], spec_rgb: [0.0; 4], fres_rgb: [0.0; 4], obj_light: [0; 4] };
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-inst"),
            contents: bytemuck::bytes_of(&inst),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // Terrain uniform (std140, 12×vec4 = 192B). Per-layer base/detail xforms as
        // full float4 (xy=scale, zw=offset) matching the engine transform_texcoord.
        let bx = |l: usize| [
            params.base_tile[l][0], params.base_tile[l][1],
            params.base_offset[l][0], params.base_offset[l][1],
        ];
        // Detail xform: authored detail tile. If a present detail layer left tile at
        // identity (unresolved scale), fall back to base_tile × DETAIL_FALLBACK — the
        // engine tiles detail MUCH finer than base (≈6–8×, 02_shader_opaque_detail.md:82);
        // a base-tile fallback reads as a low-frequency blur.
        const DETAIL_FALLBACK: f32 = 8.0;
        let dx = |l: usize| {
            let mut sx = params.detail_tile[l][0];
            let mut sy = params.detail_tile[l][1];
            if (sx - 1.0).abs() < 1e-4 && (sy - 1.0).abs() < 1e-4 {
                sx = params.base_tile[l][0] * DETAIL_FALLBACK;
                sy = params.base_tile[l][1] * DETAIL_FALLBACK;
            }
            [sx, sy, params.detail_offset[l][0], params.detail_offset[l][1]]
        };
        // Bump xform: authored bump tile (finely tiled like detail), + offset.
        let bpx = |l: usize| [
            params.bump_tile[l][0], params.bump_tile[l][1],
            params.bump_offset[l][0], params.bump_offset[l][1],
        ];
        // detail_bump xform: fine-tiled like detail. If a present detail_bump layer
        // left tile at identity (unresolved scale), fall back to base_tile × DETAIL_FALLBACK
        // (same rule as the detail map dx() above) so the relief tiles at the intended fine
        // frequency instead of 1:1 (which would read as a low-freq smear).
        let dbpx = |l: usize| {
            let mut sx = params.detail_bump_tile[l][0];
            let mut sy = params.detail_bump_tile[l][1];
            if (sx - 1.0).abs() < 1e-4 && (sy - 1.0).abs() < 1e-4 {
                sx = params.base_tile[l][0] * DETAIL_FALLBACK;
                sy = params.base_tile[l][1] * DETAIL_FALLBACK;
            }
            [sx, sy, params.detail_bump_offset[l][0], params.detail_bump_offset[l][1]]
        };
        // Terrain uniform, 32×vec4 = 512B: [0..4] base_xform, [4..8] detail_xform, [8] blend_xform,
        // [9] global_tint (+ uv_density in .w), [10] active, [11] detail_present, [12..16] bump_xform,
        // [16] bump_present, [17] = (dbb_mode, dbb_slope, dbb_offset, _), [18..22] blend_target[0..4],
        // [22] blend_max[0..4], [23..27] detail_bump_xform, [27] detail_bump_present,
        // [28] base_curve exponents, [29] detail_curve exponents, [30] bump_snorm, [31] detail_bump_snorm.
        let mut tile_data = [0.0f32; 128];
        for l in 0..4 {
            let b = bx(l); let d = dx(l); let bp = bpx(l); let dbp = dbpx(l);
            tile_data[l * 4..l * 4 + 4].copy_from_slice(&b);          // base_xform[0..4]
            tile_data[16 + l * 4..16 + l * 4 + 4].copy_from_slice(&d); // detail_xform[4..8]
            tile_data[48 + l * 4..48 + l * 4 + 4].copy_from_slice(&bp); // bump_xform[12..16]
            tile_data[92 + l * 4..92 + l * 4 + 4].copy_from_slice(&dbp); // detail_bump_xform[23..27]
        }
        tile_data[32..36].copy_from_slice(&params.blend_xform);   // blend_xform
        tile_data[36..40].copy_from_slice(&params.global_tint);   // global_tint (rgb)
        tile_data[39] = params.uv_density;                        // uv_density in .w slot
        tile_data[40..44].copy_from_slice(&params.active);        // active
        tile_data[44..48].copy_from_slice(&params.detail_present); // detail_present
        tile_data[64..68].copy_from_slice(&params.bump_present);  // bump_present [16]
        tile_data[108..112].copy_from_slice(&params.detail_bump_present); // detail_bump_present [27]
        // distance_blend_base params.
        tile_data[68] = params.dbb_mode;
        tile_data[69] = params.dbb_slope;
        tile_data[70] = params.dbb_offset;
        for l in 0..4 {
            tile_data[72 + l * 4..72 + l * 4 + 4].copy_from_slice(&params.blend_target[l]); // [18..22]
            tile_data[88 + l] = params.blend_max[l];                                        // [22]
        }
        // per-layer decode exponents.
        tile_data[112..116].copy_from_slice(&params.base_curve);   // t[28]
        tile_data[116..120].copy_from_slice(&params.detail_curve); // t[29]
        tile_data[120..124].copy_from_slice(&params.bump_snorm);        // t[30]
        tile_data[124..128].copy_from_slice(&params.detail_bump_snorm); // t[31]
        let tile_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-tiles"),
            contents: bytemuck::cast_slice(&tile_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // same dummy policy as upload_mesh_prebuilt (DM→grey, SDM→black, slices→sdm).
        let fb = self.fallbacks.get_or_init(|| FallbackTex::new(device, _queue));
        let t_dm = lightmap.as_ref().map(|l| &l.dm).unwrap_or(&fb.grey);
        let t_sdm = lightmap.as_ref().map(|l| &l.sdm).unwrap_or(&fb.black);
        let t_sdm1 = lightmap.as_ref().and_then(|l| l.sdm1.as_ref()).unwrap_or(t_sdm);
        let t_sdm2 = lightmap.as_ref().and_then(|l| l.sdm2.as_ref()).unwrap_or(t_sdm);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-bg"),
            layout: &self.terrain_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(base_views[0]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base_views[1]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(base_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(base_views[3]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(blend_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: tile_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(detail_views[0]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(detail_views[1]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(detail_views[2]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(detail_views[3]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(bump_views[0]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(bump_views[1]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(bump_views[2]) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(bump_views[3]) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(detail_bump_views[0]) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(detail_bump_views[1]) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(detail_bump_views[2]) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(detail_bump_views[3]) },
                wgpu::BindGroupEntry { binding: 19, resource: wgpu::BindingResource::TextureView(t_dm) },
                wgpu::BindGroupEntry { binding: 20, resource: wgpu::BindingResource::TextureView(t_sdm) },
                wgpu::BindGroupEntry { binding: 21, resource: wgpu::BindingResource::Sampler(&self.lm_sampler) },
                wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::TextureView(t_sdm1) },
                wgpu::BindGroupEntry { binding: 23, resource: wgpu::BindingResource::TextureView(t_sdm2) },
            ],
        });
        GpuMesh {
            casts_shadow: true,
            h4_caster: false,
            is_terrain: true,
            inst_raw: Vec::new(),
            vbuf,
            ibuf,
            index_count: indices.len() as u32,
            instance_buf,
            instance_count: 1,
            material: Material { bind_group },
            centroid: [0.0; 3],
            deco_weight: 0,
            blend_mode: 0,
        }
    }

    /// Upload a water mesh: watercolor texture + per-material WaterParams uniform.
    pub fn upload_water_mesh(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        verts: &[MeshVertex],
        indices: &[u32],
        watercolor: Option<&wgpu::TextureView>,
        slope: Option<&wgpu::TextureView>,
        params: &WaterParams,
        // The REAL authored environment_map cube for this water material (built by
        // `build_authored_env_cube`). Some → bound at binding 10 in place of the captured sky
        // dome (and params.env.x must be 1 so the shader uses the engine decode). None → the
        // captured dome stand-in stays bound (params.env.x = 0).
        env_cube_override: Option<&wgpu::TextureView>,
        // Authored foam_texture (RGBA). None → the 1×1 default (foam still gated by foam_cut).
        foam: Option<&wgpu::TextureView>,
        // Authored global_shape_texture (RGBA; .a = from_shape bank_alpha). None → the 1×1
        // default. params.env.w flags whether it is real+used.
        global_shape: Option<&wgpu::TextureView>,
        // Some → the water FS samples the per-pixel DM/SDM lightmap atlas at the vertex uv2
        // (lm = [1, hdr, k, mode]) for its `lightmap_intensity`; None → per-vertex v.color
        // (airprobe) path (lm = [0;4], dummy DM/SDM bound).
        lightmap: Option<LightmapInputs>,
    ) -> GpuMesh {
        use wgpu::util::DeviceExt;
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water-vb"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water-ib"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        memprof::note("buf:water-vb/ib", (std::mem::size_of_val(verts) + std::mem::size_of_val(indices)) as u64);
        let wlm = match &lightmap { Some(l) => [1.0, l.hdr, l.k, l.mode], None => [0.0; 4] };
        let inst = InstanceRaw { model: Mat4::IDENTITY.to_cols_array_2d(), color: [1.0; 4], scroll: [0.0; 4], detail_xform: [0.0; 4], aux: [0.0; 4], lm: wlm, spec2: [0.0; 4], bump_xform: [0.0; 4], fine_xform: [0.0; 4], bump_detail_xform: [0.0; 4], env_ctl: [0.0; 4], matmodel: 0.0, si_ctl: [0.0; 4], mattex: [0.0; 4], mattex_xform: [0.0; 4], xbump: [0.0; 4], xctl: [0.0; 4], obj_probe0: [0.0; 4], obj_probe1: [0.0; 4], spec_rgb: [0.0; 4], fres_rgb: [0.0; 4], obj_light: [0; 4] };
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water-inst"),
            contents: bytemuck::bytes_of(&inst),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water-params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // The captured per-map environment cube (real sky dome) is the FALLBACK; when the
        // material authored an environment_map cube, `env_cube_override` supplies the REAL cube.
        let fb = self.fallbacks.get_or_init(|| FallbackTex::new(device, _queue));
        let env_cube_view = env_cube_override.unwrap_or(&fb.env_cube);
        // same dummy policy as upload_terrain_mesh (DM→grey, SDM→black, slices→sdm).
        let w_dm = lightmap.as_ref().map(|l| &l.dm).unwrap_or(&fb.grey);
        let w_sdm = lightmap.as_ref().map(|l| &l.sdm).unwrap_or(&fb.black);
        let w_sdm1 = lightmap.as_ref().and_then(|l| l.sdm1.as_ref()).unwrap_or(w_sdm);
        let w_sdm2 = lightmap.as_ref().and_then(|l| l.sdm2.as_ref()).unwrap_or(w_sdm);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-bg"),
            layout: &self.water_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(watercolor.unwrap_or(&self.default_white)),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(slope.unwrap_or(&self.default_white)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(foam.unwrap_or(&self.default_white)),
                },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(env_cube_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&fb.env_samp) },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(global_shape.unwrap_or(&self.default_white)),
                },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(w_dm) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(w_sdm) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::Sampler(&self.lm_sampler) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(w_sdm1) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(w_sdm2) },
            ],
        });
        GpuMesh {
            casts_shadow: true,
            h4_caster: false,
            is_terrain: false,
            inst_raw: Vec::new(),
            vbuf,
            ibuf,
            index_count: indices.len() as u32,
            instance_buf,
            instance_count: 1,
            material: Material { bind_group },
            centroid: [0.0; 3],
            deco_weight: 0,
            blend_mode: 0,
        }
    }

    /// Draw terrain-blend meshes through the terrain pipeline.
    pub fn draw_terrain<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        meshes: &'a [GpuMesh],
    ) {
        pass.set_pipeline(&self.terrain_pipeline);
        pass.set_bind_group(0, Some(camera_bg), &[]);
        for m in meshes {
            if !m.is_terrain { continue; }
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Halo 4 sun shadow map: the object casters' BACK faces (`shadow_pipeline_h4`) -
    /// `dynamic` lists by `casts_shadow` (editor objects), `static_flagged` lists by `h4_caster`
    /// (scenario objects in the static lanes; the BSP meshes there are skipped).
    pub fn draw_shadow_h4<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        shadow_bg: &'a wgpu::BindGroup,
        dynamic: &[&'a [GpuMesh]],
        static_flagged: &[&'a [GpuMesh]],
        static_all: bool,
    ) {
        pass.set_pipeline(&self.shadow_pipeline_h4);
        pass.set_bind_group(0, shadow_bg, &[]);
        for meshes in dynamic {
            for m in *meshes {
                if !m.casts_shadow { continue; }
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_vertex_buffer(1, m.instance_buf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
            }
        }
        for meshes in static_flagged {
            for m in *meshes {
                if (!m.h4_caster && !static_all) || !m.casts_shadow { continue; }
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_vertex_buffer(1, m.instance_buf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
            }
        }
    }

    /// Depth-only shadow pass: render caster meshes from the sun POV into the
    /// shadow depth map. `shadow_bg` provides the light_view_proj (group0 binding 2).
    pub fn draw_shadow<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        shadow_bg: &'a wgpu::BindGroup,
        batches: &[&'a [GpuMesh]],
    ) {
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, shadow_bg, &[]);
        for meshes in batches {
            for m in *meshes {
                if !m.casts_shadow { continue; }
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_vertex_buffer(1, m.instance_buf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
            }
        }
    }

    /// The captured sky env cube view + its sampler (the probe tracer's sky radiance), if the
    /// fallbacks (which own it) have been initialised. None before the first mesh upload.
    pub fn env_cube_view(&self) -> Option<(&wgpu::TextureView, &wgpu::Sampler)> {
        self.fallbacks.get().map(|f| (&f.env_cube, &f.env_samp))
    }
    /// The environment cube texture (render target for the per-map sky capture).
    pub fn env_cube_texture(&self) -> Option<&wgpu::Texture> {
        self.fallbacks.get().map(|f| &f.env_cube_tex)
    }

    /// Draw the sky render_model: opaque units first (depth-tested + written), then translucent
    /// units far-to-near, switching the blend pipeline per segment by its rmsh blend mode
    /// (0=opaque/Replace, 1/5/7=additive, 2/3=multiply→DARKENS, 4=alpha_blend, 6/8=premultiplied).
    /// Camera-followed (the VS offsets by cam_pos).
    pub fn draw_sky<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_bg: &'a wgpu::BindGroup,
        segments: &'a [(GpuMesh, u8, f32)],
    ) {
        if segments.is_empty() {
            return;
        }
        pass.set_bind_group(0, camera_bg, &[]);
        let legacy = sky_legacy();
        let mut order: Vec<usize> = (0..segments.len()).collect();
        if legacy {
            // HMS_SKY_LEGACY: depth Always/no-write, so occlusion is submission order.
            // Opaque first, then translucent sorted DESCENDING by the authored-index key.
            order.sort_by(|&a, &b| {
                let ga = if segments[a].1 == 0 { 0u8 } else { 1u8 };
                let gb = if segments[b].1 == 0 { 0u8 } else { 1u8 };
                ga.cmp(&gb)
                    .then(segments[b].2.partial_cmp(&segments[a].2).unwrap_or(std::cmp::Ordering::Equal))
            });
        } else {
            // Opaque units first (depth-tested + written, so their mutual order is irrelevant),
            // then translucent units depth-tested against them, sorted FAR-TO-NEAR by section
            // radius (segments[i].2), authored unit order within equal radius (the engine's own
            // translucent sort key inside the sky model is unknown).
            order.sort_by(|&a, &b| {
                let ga = if segments[a].1 == 0 { 0u8 } else { 1u8 };
                let gb = if segments[b].1 == 0 { 0u8 } else { 1u8 };
                ga.cmp(&gb)
                    .then(segments[b].2.partial_cmp(&segments[a].2).unwrap_or(std::cmp::Ordering::Equal))
                    .then(a.cmp(&b))
            });
        }
        for &idx in &order {
            let (m, blend, _radius) = &segments[idx];
            // Blend enum: 0 opaque, 1 additive, 2 multiply, 3 double_multiply, 4 alpha_blend,
            // 5 add_src×srcalpha, 6 pre_multiplied_alpha, 7 maximum, 8 add_src×dstalpha.
            // 7 (max)→additive, 6/8→premultiplied (legacy path: alpha).
            let pipe = match blend {
                1 | 5 | 7 | 9 => &self.sky_blend_pipeline,   // 9 = add_src_times_srcalpha (Halo 4 only; rgb pre-multiplied in fs)
                2 | 3 => &self.sky_multiply_pipeline,
                4 => &self.sky_alpha_pipeline,
                6 | 8 => if legacy { &self.sky_alpha_pipeline } else { &self.sky_premul_pipeline },
                _ => &self.sky_opaque_pipeline,
            };
            pass.set_pipeline(pipe);
            pass.set_bind_group(1, &m.material.bind_group, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }

    /// Draw decals (mesh, bucket): UNLIT, per-decal blend. bucket 0=alpha, 1=additive,
    /// 2=multiply. Drawn after opaque/water so they composite onto the surfaces; alpha
    /// first, then additive/multiply, so multiplicative scorch darkens the final colour.
    pub fn draw_decals<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        pass: &mut E,
        camera_bg: &'a wgpu::BindGroup,
        decals: &'a [(GpuMesh, u8)],
    ) {
        if decals.is_empty() {
            return;
        }
        pass.set_bind_group(0, Some(camera_bg), &[]);
        let mut order: Vec<usize> = (0..decals.len()).collect();
        // alpha(0) → additive(1) → multiply(2) → premul(3) → inv-alpha(4)
        order.sort_by_key(|&i| decals[i].1);
        for &idx in &order {
            let (m, bucket) = &decals[idx];
            let pipe = match bucket {
                1 => &self.decal_additive_pipeline,
                2 => &self.decal_multiply_pipeline,
                3 => &self.decal_premul_pipeline,
                4 => &self.decal_invalpha_pipeline,
                5 => &self.decal_vector_pipeline,
                6 => &self.decal_double_multiply_pipeline,
                _ => &self.decal_alpha_pipeline,
            };
            pass.set_pipeline(pipe);
            pass.set_bind_group(1, Some(&m.material.bind_group), &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_vertex_buffer(1, m.instance_buf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..m.instance_count);
        }
    }
}

/// Build the GPU vertex + index buffers for a mesh. Called from the app's
/// PARALLEL BSP decode (`decode_bsp_mesh`, rayon) so the two big `create_buffer_init` calls — the
/// dominant serial per-mesh upload cost — happen in parallel instead of one-at-a-time in the
/// finalize. The buffers are then handed to `upload_mesh_prebuilt`. (Lives here because `bytemuck`
/// + `MeshVertex` are hms-render's; the app crate has neither.)
pub fn make_geom_buffers(
    device: &wgpu::Device,
    verts: &[MeshVertex],
    indices: &[u32],
) -> (wgpu::Buffer, wgpu::Buffer) {
    use wgpu::util::DeviceExt;
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-vb"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-ib"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf)
}

/// Upload an RGBA8 image to a wgpu texture; returns (view, texture).
pub fn upload_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::Texture) {
    upload_texture_fmt(device, queue, rgba, w, h, wgpu::TextureFormat::Rgba8Unorm)
}

/// Upload a BGRA8 image (the byte order the C++ HaloMapStudioDLL emits — it was
/// written for C#/WPF's Bgra32). Uploading as Bgra8Unorm lets the GPU present the
/// channels to the shader as true RGBA with zero CPU swizzle. Use this for ALL
/// DLL-decoded bitmaps; keep `upload_texture_rgba` for Rust-authored constants.
pub fn upload_texture_bgra(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgra: &[u8],
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::Texture) {
    upload_texture_fmt_mips(device, queue, bgra, w, h, wgpu::TextureFormat::Bgra8Unorm, true, None)
}

/// Apply a per-channel byte LUT to the RGB of 4-byte texels (alpha untouched).
fn remap_rgb<'a>(data: &'a [u8], lut: Option<&[u8; 256]>) -> std::borrow::Cow<'a, [u8]> {
    match lut {
        None => std::borrow::Cow::Borrowed(data),
        Some(l) => { let mut v = data.to_vec(); for c in v.chunks_exact_mut(4) { c[0] = l[c[0] as usize]; c[1] = l[c[1] as usize]; c[2] = l[c[2] as usize]; } std::borrow::Cow::Owned(v) }
    }
}
/// Gamma-encode LUT v^(1/2.2) so a shader `pow(x, 2.2)` returns the raw (Linear-curve) texel.
pub fn gamma_encode_lut() -> &'static [u8; 256] {
    static LUT: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| { let mut l = [0u8; 256]; for (i, v) in l.iter_mut().enumerate() { *v = ((i as f32 / 255.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8; } l })
}
/// Upload a LINEAR-curve colour map: mips are box-filtered in linear space, then every level is gamma-encoded.
pub fn upload_texture_bgra_gamma_enc(device: &wgpu::Device, queue: &wgpu::Queue, linear_bgra: &[u8], w: u32, h: u32) -> (wgpu::TextureView, wgpu::Texture) {
    upload_texture_fmt_mips(device, queue, linear_bgra, w, h, wgpu::TextureFormat::Bgra8Unorm, true, Some(gamma_encode_lut()))
}

pub(crate) const EV_CUBE_SZ: u32 = 128;
// Full mip chain for the env cube: 128 → 1 = 8 levels (floor(log2(128))+1). Reflections sample it
// by roughness LOD (env_reflection up to 6.0, water 2.0); without the chain every sample clamped to
// mip 0 → a sharp mirror regardless of roughness. The chain is generated after each sky capture.
pub(crate) const EV_CUBE_MIPS: u32 = 8;
/// The environment cubemap: an `Rgba16Float` RENDER-TARGET cube (6 faces) that
/// `SceneRenderer::render` fills once per map by rendering the actual sky dome into the 6
/// faces (so reflections show the MAP's real sky — clouds/nebula/planet — not a gradient). The
/// format matches the sky pipeline's `SCENE_HDR_FORMAT`. Returns the texture too (the capture
/// pass needs per-face array-layer views). `EV_CUBE_SZ` is the per-face resolution.
fn build_env_cube(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("env-cube"),
        size: wgpu::Extent3d { width: EV_CUBE_SZ, height: EV_CUBE_SZ, depth_or_array_layers: 6 },
        mip_level_count: EV_CUBE_MIPS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::SCENE_HDR_FORMAT,
        // COPY_DST: the per-map sky capture renders into a TEMP cube then copies here (the sky
        // meshes' material bind groups still bind this cube as a sampled resource, so it can't
        // also be a colour target in the same pass).
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // Initial clear to BLACK: a mesh sampling the cube before/without a sky capture then
    // reads ~0 → the shader's uncaptured-detection falls back to the procedural sky gradient,
    // instead of a flat blue that reads as "captured" and mirrors one constant colour.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("env-cube-init") });
    for f in 0..6u32 {
        let fv = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("env-cube-face-init"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: f,
            array_layer_count: Some(1),
            ..Default::default()
        });
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("env-cube-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &fv,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    queue.submit(std::iter::once(enc.finish()));
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("env-cube-view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let samp = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("env-cube-samp"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear, // trilinear across mips → smooth roughness transitions
        ..Default::default()
    });
    (tex, view, samp)
}

/// Build the REAL authored `environment_map` cubemap from its 6 decoded faces.
/// Each face arrives as native BGRA8 mip-0 bytes (from `decode_bitmap_face`, stride_mode=1 /
/// mip-major). Uploads them into a 6-layer D2 `Rgba8Unorm` texture with a Cube view, generating
/// the full mip chain on the CPU (a plain 2×2 box filter — faces are 128², so this is cheap and
/// keeps the authored cube independent of the SCENE_HDR sky-capture pipeline). ALPHA IS PRESERVED
/// (BC3 cube env maps encode a per-texel HDR/sky-hemisphere mask in alpha that the water shader
/// splits at `sunspot_cut`). Faces are stored in cube-face order 0..5 as the map authored them.
/// Returns None on a malformed face set (wrong count / zero or mismatched dims) so the caller
/// falls back to the captured sky dome.
pub fn build_authored_env_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    faces_bgra: &[(Vec<u8>, u32, u32)],
) -> Option<(wgpu::Texture, wgpu::TextureView)> {
    if faces_bgra.len() != 6 {
        return None;
    }
    let (w0, h0) = (faces_bgra[0].1, faces_bgra[0].2);
    if w0 == 0 || h0 == 0 {
        return None;
    }
    for f in faces_bgra {
        if f.1 != w0 || f.2 != h0 || f.0.len() < (w0 as usize * h0 as usize * 4) {
            return None;
        }
    }
    // Full mip chain down to 1×1: mips = floor(log2(max(w,h))) + 1.
    let mips = (32 - w0.max(h0).leading_zeros()).max(1);
    memprof::note("tex:authored-env-cube", (w0 as u64) * (h0 as u64) * 4 * 6 * 4 / 3);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("authored-env-cube"),
        size: wgpu::Extent3d { width: w0, height: h0, depth_or_array_layers: 6 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Box-downsample one RGBA level to the next.
    fn downsample_rgba(src: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut out = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            let y0 = (y * 2).min(h - 1);
            let y1 = (y * 2 + 1).min(h - 1);
            for x in 0..nw {
                let x0 = (x * 2).min(w - 1);
                let x1 = (x * 2 + 1).min(w - 1);
                for c in 0..4 {
                    let s = src[(y0 * w + x0) * 4 + c] as u32
                        + src[(y0 * w + x1) * 4 + c] as u32
                        + src[(y1 * w + x0) * 4 + c] as u32
                        + src[(y1 * w + x1) * 4 + c] as u32;
                    out[(y * nw + x) * 4 + c] = (s / 4) as u8;
                }
            }
        }
        (out, nw, nh)
    }
    for (layer, face) in faces_bgra.iter().enumerate() {
        // Native decode is BGRA; Rgba8Unorm wants RGBA → swap R/B. Alpha kept.
        let mut data = face.0.clone();
        for px in data.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let (mut cw, mut ch) = (w0 as usize, h0 as usize);
        for m in 0..mips {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: m,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((cw * 4) as u32),
                    rows_per_image: Some(ch as u32),
                },
                wgpu::Extent3d { width: cw as u32, height: ch as u32, depth_or_array_layers: 1 },
            );
            if m + 1 < mips {
                let (nd, nw, nh) = downsample_rgba(&data, cw, ch);
                data = nd;
                cw = nw;
                ch = nh;
            }
        }
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("authored-env-cube-view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    Some((tex, view))
}

/// An HDR (Rgba16Float) 6-face cube from already-DECODED linear float faces (RGBA f32,
/// row-major; alpha should be 1 so the authored-cube sampler's rgb*a is the value). Full box-filtered
/// mip chain like build_authored_env_cube. Used for the engine's per-cluster "dynamic" cubemaps.
pub fn build_hdr_env_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    faces: &[(Vec<f32>, u32, u32)],
) -> Option<(wgpu::Texture, wgpu::TextureView)> {
    if faces.len() != 6 { return None; }
    let (w0, h0) = (faces[0].1, faces[0].2);
    if w0 == 0 || h0 == 0 { return None; }
    for f in faces { if f.1 != w0 || f.2 != h0 || f.0.len() < (w0 as usize * h0 as usize * 4) { return None; } }
    let mips = (32 - w0.max(h0).leading_zeros()).max(1);
    memprof::note("tex:cluster-env-cube", (w0 as u64) * (h0 as u64) * 8 * 6 * 4 / 3);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cluster-env-cube"),
        size: wgpu::Extent3d { width: w0, height: h0, depth_or_array_layers: 6 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    fn down_f32(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
        let nw = (w / 2).max(1); let nh = (h / 2).max(1);
        let mut out = vec![0f32; nw * nh * 4];
        for y in 0..nh { let y0 = (y * 2).min(h - 1); let y1 = (y * 2 + 1).min(h - 1);
            for x in 0..nw { let x0 = (x * 2).min(w - 1); let x1 = (x * 2 + 1).min(w - 1);
                for c in 0..4 {
                    out[(y * nw + x) * 4 + c] = 0.25 * (src[(y0 * w + x0) * 4 + c] + src[(y0 * w + x1) * 4 + c] + src[(y1 * w + x0) * 4 + c] + src[(y1 * w + x1) * 4 + c]);
                }
            }
        }
        (out, nw, nh)
    }
    for (layer, face) in faces.iter().enumerate() {
        let mut data = face.0.clone();
        let (mut cw, mut ch) = (w0 as usize, h0 as usize);
        for m in 0..mips {
            let mut bytes = Vec::with_capacity(cw * ch * 8);
            for v in &data { bytes.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes()); }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: m, origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 }, aspect: wgpu::TextureAspect::All },
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((cw * 8) as u32), rows_per_image: Some(ch as u32) },
                wgpu::Extent3d { width: cw as u32, height: ch as u32, depth_or_array_layers: 1 },
            );
            if m + 1 < mips { let (nd, nw, nh) = down_f32(&data, cw, ch); data = nd; cw = nw; ch = nh; }
        }
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor { label: Some("cluster-env-cube-view"), dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
    Some((tex, view))
}

/// Upload a bitmap's RAW DDS (DXT10 header + BC blocks + the map's own mip chain)
/// straight to a block-compressed GPU texture — no CPU BCn-decode and no CPU mip-regen (the
/// two costs that dominate load time). Returns None when: the device lacks BC support, the
/// DDS is malformed, or the dxgi format isn't a BC format we map — the caller then falls
/// back to the CPU decode path, so this is safe to attempt for every bitmap.
pub fn upload_texture_dds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dds: &[u8],
) -> Option<(wgpu::TextureView, wgpu::Texture)> {
    if !device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        return None;
    }
    // 148-byte header: magic(4)+DDS_HEADER(124)+DXT10(20). Fields per BuildDdsHeaderDXT10:
    // height@12, width@16, mipCount@28, dxgiFormat@128. Block data begins at 148.
    if dds.len() < 148 || &dds[0..4] != b"DDS " {
        return None;
    }
    let rd = |o: usize| u32::from_le_bytes([dds[o], dds[o + 1], dds[o + 2], dds[o + 3]]);
    let h = rd(12);
    let w = rd(16);
    let mut mips = rd(28).max(1);
    let dxgi = rd(128);
    if w == 0 || h == 0 {
        return None;
    }
    // dxgi → wgpu BC format + bytes per 4×4 block. NON-sRGB (Unorm) BC variants, NOT the
    // *Srgb ones: the CPU-bgra upload path uses Bgra8Unorm (raw sRGB bytes delivered as-is)
    // and EVERY base sampler (mesh_shade/fs_holo/fs_foliage/terrain) does pow(rgb, 2.2) to
    // linearize. An sRGB BC variant would ALSO auto-linearize on sample (double-linearized,
    // ~1.5× too dark). Unorm delivers raw sRGB bytes exactly like the CPU path.
    let (fmt, block_bytes) = match dxgi {
        71 | 72 => (wgpu::TextureFormat::Bc1RgbaUnorm, 8usize),
        74 | 75 => (wgpu::TextureFormat::Bc2RgbaUnorm, 16),
        77 | 78 => (wgpu::TextureFormat::Bc3RgbaUnorm, 16),
        80 => (wgpu::TextureFormat::Bc4RUnorm, 8),
        83 => (wgpu::TextureFormat::Bc5RgUnorm, 16),
        84 => (wgpu::TextureFormat::Bc5RgSnorm, 16), // signed DXN normal maps
        _ => return None, // uncompressed / BC7 / unknown → CPU path
    };
    // Clamp mips to what the data actually contains (the DDS may promise more than present).
    let avail = dds.len() - 148;
    let mut total = 0usize;
    let mut real_mips = 0u32;
    for m in 0..mips {
        let mw = (w >> m).max(1);
        let mh = (h >> m).max(1);
        // BC copies must be block-aligned (4×4): wgpu 24 rejects a sub-4 BC write_texture
        // (it kills the load worker), so cap the chain at the last whole-block mip. The 2×2/1×1
        // tail would need a compute-shader downsample or CPU path.
        if mw % 4 != 0 || mh % 4 != 0 {
            break;
        }
        let bx = (mw / 4) as usize;
        let by = (mh / 4) as usize;
        let sz = bx * by * block_bytes;
        if total + sz > avail {
            break;
        }
        total += sz;
        real_mips += 1;
    }
    if real_mips == 0 {
        return None;
    }
    mips = real_mips;
    memprof::add_tex_bc(total as u64); // exact BC bytes committed
    memprof::note("tex:dds-bc", total as u64);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dds-bc"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut off = 148usize;
    for m in 0..mips {
        let mw = (w >> m).max(1);
        let mh = (h >> m).max(1);
        let bx = ((mw + 3) / 4).max(1) as usize;
        let by = ((mh + 3) / 4).max(1) as usize;
        let sz = bx * by * block_bytes;
        if off + sz > dds.len() {
            break;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: m,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &dds[off..off + sz],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((bx * block_bytes) as u32),
                rows_per_image: Some(by as u32),
            },
            wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
        );
        off += sz;
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Some((view, tex))
}

/// Like `upload_texture_bgra` but with NO mip chain — skips the CPU box-filter
/// cost. Use for high-count, transient, or close-viewed textures (foliage cards,
/// decals, sky panels) where minification aliasing is minor and the mip-gen cost
/// dominates load time. The deduped terrain/opaque surface cache keeps mips.
pub fn upload_texture_bgra_nomip(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgra: &[u8],
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::Texture) {
    upload_texture_fmt_mips(device, queue, bgra, w, h, wgpu::TextureFormat::Bgra8Unorm, false, None)
}

/// Upload an Rgba16Float texture from already-packed f16 bytes (8 bytes/texel, RGBA,
/// little-endian half floats), no mip chain. Used for the HDR opaque self-illum emissive so
/// authored intensity>1 survives into the exposed-space bloom pass (engine keeps emissive in float).
/// Rgba16Float is linearly filterable by default (unlike Rgba32Float), matching the emis sampler.
pub fn upload_texture_rgba16f_nomip(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    f16_bytes: &[u8],
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::Texture) {
    memprof::add_tex_uncompressed((w as u64) * (h as u64) * 8);
    memprof::note("tex:emis-hdr", (w as u64) * (h as u64) * 8);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("emis-hdr-tex"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        f16_bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * w), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let view = tex.create_view(&Default::default());
    (view, tex)
}

/// Box-filter one 4-byte-per-texel level down to half resolution (min 1). The
/// average is done in the stored channel order (works identically for RGBA or
/// BGRA — order is irrelevant to a per-channel box filter).
fn downsample_2x(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let dw = (w / 2).max(1);
    let dh = (h / 2).max(1);
    let mut dst = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh {
        for x in 0..dw {
            // Source 2x2 block (clamped for odd dimensions).
            let sx0 = (x * 2).min(w - 1);
            let sy0 = (y * 2).min(h - 1);
            let sx1 = (x * 2 + 1).min(w - 1);
            let sy1 = (y * 2 + 1).min(h - 1);
            let px = |sx: u32, sy: u32, c: usize| src[((sy * w + sx) * 4) as usize + c] as u32;
            let di = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                let sum = px(sx0, sy0, c) + px(sx1, sy0, c) + px(sx0, sy1, c) + px(sx1, sy1, c);
                dst[di + c] = ((sum + 2) / 4) as u8;
            }
        }
    }
    (dst, dw, dh)
}

fn upload_texture_fmt(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::Texture) {
    upload_texture_fmt_mips(device, queue, rgba, w, h, format, true, None)
}

fn upload_texture_fmt_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    mips: bool,
    lut: Option<&[u8; 256]>, // per-level RGB remap applied AFTER downsampling (mips built in linear space)
) -> (wgpu::TextureView, wgpu::Texture) {
    // Full mip chain: without mipmaps, minified textures (esp. the finely tiled terrain detail
    // maps and any surface viewed at a grazing angle) alias into salt-and-pepper "graininess".
    // Generate levels on the CPU (box filter) — the bitmaps are already CPU-decoded to 4-byte
    // texels, so no GPU blit pass is needed.
    let mip_level_count = if mips { 32 - w.max(h).max(1).leading_zeros() } else { 1 }; // floor(log2(max))+1
    // account the committed bytes (base level + ~1/3 for the mip chain).
    memprof::add_tex_uncompressed((w as u64) * (h as u64) * 4 * (if mips { 4 } else { 3 }) / 3);
    memprof::note(if mips { "tex:rgba-mips" } else { "tex:rgba-nomip" }, (w as u64) * (h as u64) * 4 * (if mips { 4 } else { 3 }) / 3);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh-tex"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let write = |level: u32, data: &[u8], lw: u32, lh: u32| {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * lw),
                rows_per_image: Some(lh),
            },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
    };
    write(0, &remap_rgb(rgba, lut), w, h);
    let (mut cur, mut cw, mut ch) = (rgba.to_vec(), w, h);
    for level in 1..mip_level_count {
        let (next, nw, nh) = downsample_2x(&cur, cw, ch);
        write(level, &remap_rgb(&next, lut), nw, nh);
        cur = next;
        cw = nw;
        ch = nh;
    }
    // HMS_VERIFY: copy back the base level's first rows and compare to the source (print-only).
    // Distinguishes GPU STORAGE corruption (readback ≠ source) from a sampling issue
    // (readback == source). The readback is expensive.
    if std::env::var("HMS_VERIFY").is_ok() && w >= 4 && h >= 4 {
        let src_bpr = (4 * w) as usize;
        let padded_bpr = ((src_bpr + 255) / 256) * 256; // 256-align for copy_texture_to_buffer
        let rows = 4u32;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verify-readback"),
            size: (padded_bpr * rows as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("verify-enc") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_bpr as u32), rows_per_image: Some(rows) } },
            wgpu::Extent3d { width: w, height: rows, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        {
            let mapped = buf.slice(..).get_mapped_range();
            let mut mismatch = 0usize;
            let mut total = 0usize;
            for r in 0..rows as usize {
                let gpu_row = &mapped[r * padded_bpr..r * padded_bpr + src_bpr];
                let src_row = &rgba[r * src_bpr..r * src_bpr + src_bpr];
                for (a, b) in gpu_row.iter().zip(src_row.iter()) {
                    total += 1;
                    if a != b { mismatch += 1; }
                }
            }
            eprintln!("HMS_DIAG VERIFY {}x{} readback mismatch {}/{} bytes ({}%)",
                w, h, mismatch, total, if total > 0 { mismatch * 100 / total } else { 0 });
        }
        buf.unmap();
    }
    let view = tex.create_view(&Default::default());
    (view, tex)
}

/// A unit cube (per-face normals) for smoke-testing the mesh pipeline.
pub(crate) fn test_cube() -> (Vec<MeshVertex>, Vec<u32>) {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for (n, u, v) in faces {
        let base = verts.len() as u32;
        let c = [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5];
        let corners = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        for (i, (a, b)) in corners.iter().enumerate() {
            let p = [
                c[0] + u[0] * a + v[0] * b,
                c[1] + u[1] * a + v[1] * b,
                c[2] + u[2] * a + v[2] * b,
            ];
            verts.push(MeshVertex { pos: p, normal: n, uv: uvs[i], ..Default::default() });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

// Shared fog module — prepended to every shader that fogs (mesh/terrain/water) so
// the two-band engine algorithm lives in ONE place. Port of Reach
// atmosphere_core.hlsl_include: radiance × extinction + additive inscatter, a
// ground band with an extinction-path-weighted mixed colour, and the fog-light
// sun disc (crepuscular glow through haze).
//   atm0=(sky_rgb, sky_thickness);  atm1=(sky_h, sky_b, sky_max_dist, dist_bias);
//   atm2=(ground_rgb, ground_thickness); atm3=(ground_h, ground_b, ground_max_dist, enabled);
//   atm4=(fog_light_rgb, fog_light_dist_falloff);
//   atm5=(fog_light_ang_falloff, fog_light_near_cut, has_fog_light, _);
//   atm6=(sun_dir_TO_sun_Zup, sun_valid).
/// Shared lighting module — prepended to the mesh + terrain shaders. Declares the light uniform
/// (group 0 binding 2), the sun shadow depth map (3) + comparison sampler (4) and the 1x1
/// luminance meter (9), and carries: `shadow_strength` (a 3×3-PCF lit factor [0,1] that GATES the
/// analytical sun term; the baked lightmap keeps its own shadows), the Halo 4 shadow lanes, the
/// exposure-derived ILLUM_SCALE, the scenario SimpleLights and the per-domain scene grade.
pub(crate) const SHADOW_SAMPLE_WGSL: &str = r#"
// Halo 4 tail (zeros on Reach maps): h4s0 = (shadow texel uv, bias inner, edge, corner), h4s1 =
// (burn active, depth range wu, texel wu, _), cvp = the floating-shadow cascade view_proj, casc0 =
// (half_width, length, receiver bias, poisson radius uv), casc1 = (taps, resolution, active, _).
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32>, h4s0: vec4<f32>, h4s1: vec4<f32>, cvp: mat4x4<f32>, casc0: vec4<f32>, casc1: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;
@group(0) @binding(3) var shadow_tex: texture_depth_2d;
@group(0) @binding(4) var shadow_cmp: sampler_comparison;

// Halo 4 "forge lightmap burn" sun visibility of a world point: the engine's
// forge_lightmap_render_sun_structure pixel shader (docs/halo4_lighting_model.md §10) - 16 point
// depth compares on the 4x4 texel block around the projected position, bilinear-weighted
// ((1-f), 1, 1, f per axis) and divided by 9, receiver bias per tap ring (0.002 inner, 0.002236
// edge, 0.004 corner, normalized depth), `shadow_depth >= z + bias` = lit. Outside the map = 1
// (the engine discards those texels = the baked value stays). No far clip: a receiver beyond the
// casters' depth range compares like any other (a cleared texel = 1.0 = no caster = lit).
fn h4_burn_shadow(world_pos: vec3<f32>) -> f32 {
    if (light.h4s1.x < 0.5) { return 1.0; }
    let clip = light.light_view_proj * vec4<f32>(world_pos, 1.0);
    if (abs(clip.w) < 1e-6) { return 1.0; }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) { return 1.0; }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let z = ndc.z;
    let dim = vec2<f32>(textureDimensions(shadow_tex, 0));
    let p = uv * dim + vec2<f32>(0.5);
    let f = fract(p);
    let base = vec2<i32>(floor(p)) - vec2<i32>(2, 2);   // texel (-1.5, -1.5) of the block
    let wx = vec4<f32>(1.0 - f.x, 1.0, 1.0, f.x);
    let wy = vec4<f32>(1.0 - f.y, 1.0, 1.0, f.y);
    let maxi = vec2<i32>(dim) - vec2<i32>(1, 1);
    var sum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let ring = i32(i == 0 || i == 3) + i32(j == 0 || j == 3);   // 0 inner, 1 edge, 2 corner
            var bias = light.h4s0.y;
            if (ring == 1) { bias = light.h4s0.z; }
            if (ring == 2) { bias = light.h4s0.w; }
            let tc = clamp(base + vec2<i32>(i, j), vec2<i32>(0, 0), maxi);
            let d = textureLoad(shadow_tex, tc, 0);
            let lit = select(0.0, 1.0, d >= z + bias || d >= 0.999999);
            sum = sum + wx[i] * wy[j] * lit;
        }
    }
    return sum / 9.0;
}

// The Halo 4 floating-shadow cascade apply (halo4.dll sub_180361EFC + the shadow_apply_poisson_6/8/12
// explicit pixel shaders): the box is the scnr cascade around the viewer; inside it the shadow mask's
// red channel = the mean of N rotated-poisson point compares `depth >= p.z - bias` at radius
// filter/800 (uv), rotation angle = frac(dot(pixel.xy, (123.456, 321.321))) * pi. The lighting entry
// then uses lerp(mask.r, static_vis, fade) with fade = sat(max(|v| - extent) + 1) over the box (v =
// position in cascade space, 1 wu ramp at its edge). Returns (cascade visibility, fade).
var<private> H4_POISSON: array<vec2<f32>, 26> = array<vec2<f32>, 26>(
    // 6 taps (quality 2)
    vec2<f32>(0.4052, -0.4212), vec2<f32>(0.4451, 0.7681), vec2<f32>(-0.5668, -0.8087), vec2<f32>(-0.0607, 0.2464), vec2<f32>(-0.7401, -0.1157), vec2<f32>(-0.2276, 0.9700),
    // 8 taps (quality 0)
    vec2<f32>(0.1081, -0.7125), vec2<f32>(-0.4382, -0.1275), vec2<f32>(0.7456, -0.5406), vec2<f32>(0.2046, 0.0989), vec2<f32>(-0.6491, -0.7163), vec2<f32>(0.1087, 0.7960), vec2<f32>(0.8769, 0.2976), vec2<f32>(-0.7880, 0.4524),
    // 12 taps (quality 1)
    vec2<f32>(-0.7916, 0.5977), vec2<f32>(0.5195, -0.7670), vec2<f32>(-0.6959, -0.4571), vec2<f32>(0.4734, 0.4800), vec2<f32>(-0.3262, 0.4058), vec2<f32>(-0.2033, -0.6207), vec2<f32>(-0.3219, 0.9326), vec2<f32>(0.9623, 0.1950), vec2<f32>(0.8964, -0.4125), vec2<f32>(0.1855, 0.8931), vec2<f32>(-0.8401, 0.0736), vec2<f32>(0.5074, -0.0644)
);
@group(0) @binding(14) var h4_cascade_tex: texture_depth_2d;
fn h4_cascade_shadow(world_pos: vec3<f32>, pixel: vec2<f32>) -> vec2<f32> {
    let clip = light.cvp * vec4<f32>(world_pos, 1.0);
    let ndc = clip.xyz;   // orthographic: w == 1; z in [0, 1] over the box length
    let hw = light.casc0.x;
    let len = light.casc0.y;
    // cascade-space position (wu): across = ndc.xy * hw, along = (z - 0.5) * len
    let v = vec3<f32>((ndc.z - 0.5) * len, ndc.x * hw, ndc.y * hw);
    let fade = clamp(max(max(abs(v.y) - hw, abs(v.z) - hw), abs(v.x) - len) + 1.0, 0.0, 1.0);
    let inside = abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;
    if (!inside) { return vec2<f32>(1.0, 1.0); }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let receiver = ndc.z - light.casc0.z;
    let ang = fract(dot(pixel, vec2<f32>(123.456, 321.321))) * 3.14159;
    let cs = vec2<f32>(cos(ang), sin(ang));
    let n = i32(light.casc1.x + 0.5);
    var first = 0;
    if (n == 8) { first = 6; }
    if (n == 12) { first = 14; }
    let res = light.casc1.y;
    let maxi = i32(res) - 1;
    var lit = 0.0;
    for (var k = 0; k < n; k = k + 1) {
        let t = H4_POISSON[first + k] * light.casc0.w;
        let off = vec2<f32>(cs.x * t.x - cs.y * t.y, cs.y * t.x + cs.x * t.y);
        let tc = clamp(vec2<i32>((uv + off) * res), vec2<i32>(0, 0), vec2<i32>(maxi, maxi));
        let d = textureLoad(h4_cascade_tex, tc, 0);
        lit = lit + select(0.0, 1.0, d >= receiver);
    }
    return vec2<f32>(lit / f32(max(n, 1)), fade);
}

// Engine ILLUM_SCALE (reach_tag_test sub_140838800 / sub_1407E89C0) = g_alt_exposure.r =
// 2^((1 - s)·(P - E)) with E = current exposure in stops (log2 of the resolve gain, auto-adapted & clamped to the
// cfxs band), P = cfxs self_illum_preferred exposure (default 0), s = self_illum_scale "exposure change" (default
// 0.3). Self-illum follows only 30% of the scene exposure. The gain replicates the post pass `resolve()` from the
// same 1x1 luminance meter: light.expo = [stops_gain, key, lo, hi], light.expo2 = [cal, fixed_gain, P, s].
@group(0) @binding(9) var lum_meter: texture_2d<f32>;
fn illum_scale_now() -> f32 {
    let mean_log = textureLoad(lum_meter, vec2<i32>(0, 0), 0).r;
    let hi = light.expo.w;
    let lo = min(light.expo.z, hi);
    let raw_gain = light.expo.y / max(exp2(mean_log), 1e-4);
    var gain = light.expo.x * clamp(raw_gain * light.expo2.x, lo, hi);
    if (light.expo2.y > 1e-6) { gain = light.expo2.y; }
    // E is the engine's exposure in STOPS AROUND THE KEY (the cfxs-band-clamped adaptation; the
    // resolve gain is key·ENGINE_METER_UNITS(10)·2^stops, see lib.rs set_auto_exposure). Ivory
    // capture check: ev = -1.03 → 2^(0.7·1.03) = 1.65 vs captured g_alt_exposure.r 1.619.
    let e = log2(max(gain, 1e-6) / max(light.expo.y * 10.0, 1e-4));
    return exp2((1.0 - light.expo2.w) * (light.expo2.z - e));
}
// The engine's 2^(exposure stops) — the adapted, cfxs-band-clamped factor AROUND the key (Ivory
// capture: 2^-0.99 = 0.503). The resolve gain is key·ENGINE_METER_UNITS(10)·2^stops (lib.rs
// set_auto_exposure), so stops-gain = gain / (key·10). Used by the chmt min-luminance lift of
// dynamic objects (sub_1407E9430: s = max(min_lum / (2^stops · lum), 1)).
fn exposure_stops_gain() -> f32 {
    let mean_log = textureLoad(lum_meter, vec2<i32>(0, 0), 0).r;
    let hi = light.expo.w;
    let lo = min(light.expo.z, hi);
    let raw_gain = light.expo.y / max(exp2(mean_log), 1e-4);
    var gain = clamp(raw_gain * light.expo2.x, lo, hi);
    if (light.expo2.y > 1e-6) { gain = light.expo2.y; }
    return gain / max(light.expo.y * 10.0, 1e-4);
}

fn shadow_strength(world_pos: vec3<f32>) -> f32 {
    let clip = light.light_view_proj * vec4<f32>(world_pos, 1.0);
    // No shadow matrix set (zero mat → w==0) → fully lit (avoids /0 NaN).
    if (abs(clip.w) < 1e-6) { return 1.0; }
    let ndc = clip.xyz / clip.w;
    // Outside the sun frustum → fully lit.
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let cur = ndc.z - 0.0012; // small constant bias (HW slope bias also applied)
    let texel = 1.0 / 2048.0;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            sum = sum + textureSampleCompareLevel(
                shadow_tex, shadow_cmp, uv + vec2<f32>(f32(x), f32(y)) * texel, cur);
        }
    }
    return sum / 9.0; // 1 = lit, 0 = fully shadowed
}

// PER-DOMAIN scene grade — the sceg ambient_tint applied to LIT surfaces ONLY (the engine has NO
// whole-frame post grade; the sky dome and water use their own colour and never call this).
// Luma-preserving warm tint (the hue shift survives the post exposure) with a blue/green floor so
// daylight R/G warmth reads without over-yellowing. Strength 0.68; a neutral [1,1,1] → exact no-op.
fn scene_grade(rgb: vec3<f32>) -> vec3<f32> {
    let a = light.ambient_tint.rgb;
    var g = vec3<f32>(
        1.0 + (a.x - 1.0) * 0.68,
        max(1.0 + (a.y - 1.0) * 0.68, 0.80),
        max(1.0 + (a.z - 1.0) * 0.68, 0.62),
    );
    g = g * g;
    let gl = max(dot(g, vec3<f32>(0.299, 0.587, 0.114)), 1e-3);
    return rgb * (g / gl);
}

// Scenario SimpleLights (point/spot). Each light = 5×vec4 mirroring the native
// ZhSimpleLight: pos+cutoff², dir+sphere%, color+smooth, cone(cosCut,ratio,power,_),
// atten(cutoff,farEnd,farRatio,_). hdr.x = active count (0..8).
struct SimpleLight {
    pos_cutoffsq: vec4<f32>,
    dir_sphere: vec4<f32>,
    color_smooth: vec4<f32>,
    cone: vec4<f32>,
    atten: vec4<f32>,
};
struct SimpleLights { hdr: vec4<f32>, lights: array<SimpleLight, 8> };
@group(0) @binding(6) var<uniform> slights: SimpleLights;

// Forward point/spot SimpleLights accumulation (engine simple_lights.hlsl_include /
// calculate_simple_light; the engine runs inline-method `ligh` lights through it on BSP,
// objects and terrain alike). Flat unrolled loop over up to 8 lights; Lambert /π, distance
// falloff, spot cone. Returns the summed incident radiance to multiply by albedo.
fn calc_simple_lights(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    let count = i32(slights.hdr.x + 0.5);
    for (var i = 0; i < 8; i = i + 1) {
        if (i >= count) { break; }
        let L = slights.lights[i];
        let to_l = L.pos_cutoffsq.xyz - world_pos;
        let d2 = dot(to_l, to_l);
        if (d2 >= L.pos_cutoffsq.w) { continue; }      // bounding-radius² cull
        let d = sqrt(max(d2, 1e-8));
        let lhat = to_l / d;
        // distance falloff: saturate((farEnd - d) * farRatio)², squared linear ramp.
        let dist_att = pow(saturate((L.atten.y - d) * L.atten.z), 2.0);
        // spot cone: saturate((cos_ang - cosCut) * ratio)^power; sphere%=1 → omni (=1).
        let cos_ang = saturate(dot(-lhat, L.dir_sphere.xyz));
        let cone = pow(saturate((cos_ang - L.cone.x) * max(L.cone.y, 1e-4)), max(L.cone.z, 1.0));
        let ang_atten = mix(cone, 1.0, L.dir_sphere.w);
        let ndl = saturate(dot(n, lhat) + 0.06);        // engine cosine raise
        sum += L.color_smooth.xyz * (ndl * ang_atten * dist_att * 0.3183);
    }
    return min(sum, vec3<f32>(8.0, 8.0, 8.0));
}
"#;

pub(crate) const FOG_WGSL: &str = r#"
struct Fog {
    atm0: vec4<f32>, atm1: vec4<f32>, atm2: vec4<f32>, atm3: vec4<f32>,
    atm4: vec4<f32>, atm5: vec4<f32>, atm6: vec4<f32>,
};
@group(0) @binding(1) var<uniform> fog: Fog;

// FOG_WU is the world-unit → fog-extinction-depth bridge: `ext = exp(-thickness · FOG_WU · dist)`
// with `dist` = raw world-unit `length(world − cam)`, so FOG_WU folds the engine's own thickness
// conversion. The runtime atmosphere cbuffer prep (reach_tag_test.exe sub_1407FF2D0 @ 0x1407ff2d0)
// converts the authored fogg `thickness` → `_sky_fog_thickness` by × 0.0099999998 (= ×0.01) for
// thickness ≥ 0.0002 (a 2·(t−1e-4) ramp below), then evaluates
// `exp(-_sky_fog_thickness · weight² · fog_view_distance)`. Captured for forge_halo: authored
// thickness 0.2 → 0.002/wu, confirmed against the engine atmosphere CB (frame_1.json 240B, fog0_color
// (0.210,0.510,0.907) — matching our sky_tint exactly). The authored thickness / heights / bias /
// max_distance are used verbatim; this constant is only the unit bridge, and FOG_WU scales dist AND
// max/bias together so it is algebraically that ×0.01. Live-tunable via HMS_FOG_WU (cam.time.z).
const FOG_WU: f32 = 0.01;
// atmosphere_lut.hlsl_include:40: LUT_get_x_coord ends in `saturate(x)` where
// `x = sqrt(view_distance·ONE_OVER_MAX_VIEW_DISTANCE·…)`, so the sampled scattering FREEZES past the
// far edge of the table (view_distance ≥ MAX_VIEW_DISTANCE = lut_data[0].x) instead of thickening
// further. The scattering core is evaluated analytically here with no such clamp, so the raw
// world-unit view distance is clamped instead. MAX_VIEW_DISTANCE is a per-map LUT constant not
// surfaced through the fog params (which carry only the per-BAND sky/ground max_distance, a
// different clamp inside compute_scattering_core), so this is the documented engine default.
// TODO: plumb the authored per-map MAX_VIEW_DISTANCE (lut_data[0].x) through the FogParser/FFI.
const MAX_VIEW_DISTANCE: f32 = 10000.0;

struct FogBand { ext: f32, approx: f32, fvd: f32, raw_fvd: f32 };
fn fog_band(dist: f32, vtop: f32, vbot: f32, vdiff: f32, fh: f32, fbase: f32, fthick: f32, fmaxd: f32, dbias: f32) -> FogBand {
    let top = min(vtop - fbase, fh);
    let bot = vbot - fbase;
    let dratio = clamp((top - bot) / max(vdiff, 1e-5), 0.0, 1.0);
    // get_fog_thickness_at_relative_height evaluates the quadratic
    // density at the segment BOTTOM (atmosphere_core.hlsl_include:64/102:
    // approx_relative_height = lerp(view_bottom, view_top, 0.0) == view_bottom), NOT the midpoint.
    // Density is highest at the band floor; the midpoint systematically under-estimated thickness.
    let rel = bot;
    let w = clamp((fh - rel) / max(fh, 1e-5), 0.0, 1.0);
    let approx = fthick * w * w;
    // engine: fog_view_distance = max(view_distance*dist_ratio + bias, 0), THEN min(., max_distance).
    // raw_fvd keeps the pre-max value so the MIXED band can clamp the base-sky term to the SKY max
    // (sky_max - sky_travel), not this band's own (tiny) ground max.
    let raw_fvd = max(dist * dratio + dbias, 0.0);
    let fvd = min(raw_fvd, max(fmaxd, 1e-4));
    var r: FogBand;
    r.ext = clamp(exp(-approx * fvd), 0.0, 1.0);
    r.approx = approx;
    r.fvd = fvd;
    r.raw_fvd = raw_fvd;
    return r;
}
fn apply_fog(shaded: vec3<f32>, wp: vec3<f32>, cp: vec3<f32>) -> vec3<f32> {
    if (fog.atm3.w < 0.5) { return shaded; }
    // FOG_WU is the effective density/distance scale. cam.time.z (HMS_FOG_WU env) overrides
    // the const for live calibration; 0 → use the const.
    let FW = select(FOG_WU, cam.time.z, cam.time.z > 0.0);
    let sky_c = fog.atm0.xyz; let sky_t = fog.atm0.w;
    let sky_h = max(fog.atm1.x, 1e-4); let sky_b = fog.atm1.y;
    let sky_md = fog.atm1.z * FW; let dbias = fog.atm1.w * FW;
    let gnd_c = fog.atm2.xyz; let gnd_t = fog.atm2.w;
    let gnd_h = max(fog.atm3.x, 1e-4); let gnd_b = fog.atm3.y; let gnd_md = fog.atm3.z * FW;

    let to_scene = wp - cp;
    let vd_true = length(to_scene);
    let view_dir = select(vec3<f32>(0.0, 0.0, 1.0), to_scene / vd_true, vd_true > 1e-5);
    // Freeze scattering past MAX_VIEW_DISTANCE (engine LUT saturate(x); atmosphere_lut:40).
    // Clamp the raw world-unit view distance BEFORE the FOG_WU scale — the view direction keeps the
    // true length so the fog-light disc is unaffected. Near/mid distances (< MAX_VIEW_DISTANCE) are a
    // no-op, so only far vistas beyond the LUT far edge stop over-thickening.
    let vd = min(vd_true, MAX_VIEW_DISTANCE);
    let dist = vd * FW;
    let top = max(cp.z, wp.z) + 0.001;
    let bot = min(cp.z, wp.z);
    let diff = max(top - bot, 1e-4);

    // Sky band (no ground floor yet).
    var sky = fog_band(dist, top, bot, diff, sky_h, sky_b, sky_t, sky_md, dbias);
    var sky_i = sky_c * (1.0 - sky.ext);
    var ext = sky.ext;
    var insc = sky_i;

    if (gnd_t > 0.0) {
        let gah = gnd_h + gnd_b;
        // The sky solo band only fogs the ray segment ABOVE the
        // ground ceiling (calc_solo_fog_extinction floors view_bottom to ground_fog_absolute
        // _height, atmosphere_core.hlsl_include:140). The below-ceiling segment is filled by the
        // ground band + base_sky_ext below. Recompute sky floored.
        let sky_bot_f = max(bot, gah);
        sky = fog_band(dist, top, sky_bot_f, diff, sky_h, sky_b, sky_t, sky_md, dbias);
        sky_i = sky_c * (1.0 - sky.ext);
        // Ground band, view-top clamped to the ceiling (fogs the below-ceiling segment).
        let g = fog_band(dist, min(top, gah), bot, diff, gnd_h, gnd_b, gnd_t, gnd_md, dbias);
        // Engine calc_mixed_fog_extinction (atmosphere_core:106-119):
        // base_fog_view_distance = min(RAW ground fvd, sky_max - sky_travel). Use the RAW ground fvd
        // (g.raw_fvd), NOT the ground-max-clamped g.fvd — the ground max_distance (~10wu) is tiny, so
        // clamping the base-sky term to it killed the low-terrain distance fog and left an abrupt
        // "wall" at the ground ceiling. The base-sky term must reach the full SKY max so the low
        // terrain fogs gradually with sky thickness over distance, exactly like the engine.
        let base_fvd = min(g.raw_fvd, max(sky_md - sky.fvd, 0.0));
        let base_sky_ext = clamp(exp(-sky_t * base_fvd), 0.0, 1.0);
        let mixed_ext = g.ext * base_sky_ext;   // == engine mixed_fog_extinction
        // Path-weighted mixed colour: ground by its own path, sky by RAW thickness × base dist.
        let gw = g.approx * g.fvd;
        let base_w = sky_t * base_fvd;
        let mixed_c = (gnd_c * gw + sky_c * base_w) / max(gw + base_w, 1e-5);
        // Total extinction = sky-solo (above ceiling) × mixed (near). sky.ext·base_sky_ext applies
        // sky thickness over the whole ray; g.ext adds ground thickness over the near segment.
        ext = sky.ext * mixed_ext;
        // The engine forms TWO inscatter estimates blended by altitude (compute_scattering
        // _core, atmosphere_core.hlsl_include:168-179) — NOT one collapsed mixed colour.
        //   inscatter_a = ground-dominant (low, inside the ground fog)
        //   inscatter_b = sky-dominant    (high, above the ground ceiling)
        let sky_insc = sky_i;
        let mixed_insc = mixed_c * (1.0 - mixed_ext);      // uses mixed_fog_extinction
        let inscatter_a = sky_insc * mixed_ext + mixed_insc;   // ground-dominant
        let inscatter_b = sky_insc + mixed_insc * sky.ext;     // sky-dominant
        let low_rate = clamp((gah - cp.z) / gnd_h * gnd_t / max(sky_t, 1e-5), 0.0, 1.0);
        insc = mix(inscatter_b, inscatter_a, low_rate);
    }

    // Fog-light sun disc (engine fog_light_color): a directional glow toward the
    // sun that builds with distance — the sun-through-haze that reads as authored
    // atmosphere. Off unless the map authors it AND a sun direction is valid.
    let sun_dir = fog.atm6.xyz;
    // Gate on the authored has_fog_light flag (atm5.z) + a non-degenerate direction. atm6.w is the
    // engine radius_offset (−cosθ/(1−cosθ)), often negative, so it is not a validity flag.
    if (fog.atm5.z > 0.5 && dot(sun_dir, sun_dir) > 0.001) {
        let sd = normalize(sun_dir);
        // Angular disc: with a real radius_scale (atm5.w = 1/(1−cosθ) ≥ 0.5 for any authored cone)
        // use the engine fog_light_color affine ratio = saturate(cosine·radius_scale + radius_offset);
        // the FogParser fallback (scale=1, offset=0) reduces this to clamp(cosine,0,1) = full hemisphere.
        let cosine = dot(view_dir, sd);
        let ratio = select(clamp(cosine, 0.0, 1.0), clamp(cosine * fog.atm5.w + fog.atm6.w, 0.0, 1.0), fog.atm5.w > 0.001);
        let tint = fog.atm4.xyz * pow(max(ratio, 1e-6), fog.atm5.x);
        let near_cut = fog.atm5.y;
        // engine: fog_light_scale = saturate(((1-extinction) - nearby_cutoff)/(1-nearby_cutoff)),
        // then pow(., distance_falloff). extinction here is the monochrome `ext`.
        var scale = clamp((1.0 - ext - near_cut) / max(1.0 - near_cut, 1e-5), 0.0, 1.0);
        scale = pow(max(scale, 1e-6), fog.atm4.w);
        insc += scale * tint;
    }
    // NO transmittance floor: the engine lets extinction reach 0 at the horizon, which is WHY the
    // far band is pure sky-fog colour (shaded·0 + insc == the authored sky-fog colour, an
    // atmospheric horizon). The inscatter term supplies the correct horizon colour as ext→0.
    return shaded * ext + insc;
}
"#;

pub(crate) const MESH_WGSL: &str = r#"
const TDDBG: f32 = __TDDBG__;
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32>, inv_view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
// the opaque scene depth copy (camera group binding 5, same view the water shader reads): halogram
// and particle depth fades
@group(0) @binding(5) var scene_depth: texture_depth_2d;
// byte-exact engine vMF diffuse LUT (rasterizer\diffusetable, 128×128 R8): the dual_vmf_diffuse
// dominant-lobe coefficient indexed by (dot(domDir,N)*0.5+0.5, bandwidth).
@group(0) @binding(7) var vmf_lut: texture_2d<f32>;
@group(0) @binding(8) var vmf_samp: sampler;
@group(0) @binding(13) var dps_lut: texture_3d<f32>;  // rasterizer\diffuse_power_specular\diffuse_power (x = dot(dom,R)*0.5+0.5, y = bandwidth, z = roughness), value*3
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_samp: sampler;
@group(1) @binding(2) var emis_tex: texture_2d<f32>;
@group(1) @binding(3) var detail_tex: texture_2d<f32>;
@group(1) @binding(4) var bump_tex: texture_2d<f32>;
// GPU lightmap: per-submap DM/SDM atlas (raw-BC, sampled at uv2). Dummy 1×1 for
// non-atlas meshes (lm.flag=0 → never sampled).
@group(1) @binding(5) var lm_dm_tex: texture_2d<f32>;
@group(1) @binding(6) var lm_sdm_tex: texture_2d<f32>;
@group(1) @binding(7) var lm_samp: sampler;
@group(1) @binding(8) var lm_sdm1_tex: texture_2d<f32>;  // SDM z-slice 1
@group(1) @binding(9) var lm_sdm2_tex: texture_2d<f32>;  // SDM z-slice 2
@group(1) @binding(10) var env_cube: texture_cube<f32>;  // environment reflection cube
@group(1) @binding(11) var env_cube_samp: sampler;
@group(1) @binding(12) var bump_detail_tex: texture_2d<f32>;  // 2nd normal (bump_detail_map)
@group(1) @binding(13) var at_tex: texture_2d<f32>;  // dedicated alpha_test_map (=base view when absent)
@group(1) @binding(14) var mat_tex: texture_2d<f32>;  // material_texture (per-texel roughness/spec); 1×1 grey (a=1) when absent → no-op
@group(1) @binding(15) var bump_detail2_tex: texture_2d<f32>;  // 3rd normal (bump_detail_map2); flat-normal when absent
@group(1) @binding(16) var bump_detail3_tex: texture_2d<f32>;  // 4th normal (bump_detail_map3); flat-normal when absent

// DETAIL_MULTIPLIER 4.59479 = 2^2.2: rmsh rock/cliff surfaces use a bland base + a fine detail
// map for their texture; a mid-grey detail texel is neutral.
const MESH_DETAIL_MULT: f32 = 4.59479;
// Ground-bounce ambient floor (mesh_shade): a small warm floor from the map's own ground-fog
// colour so PVL-less instanced geometry isn't pure black; the baked probe carries the real
// shadow fill (the engine has NO flat hemisphere ambient — the baked dual-VMF probe IS the ambient).
const GROUND_AMB: f32 = 0.05;
// Per-vertex-lit bump relief strength. Instanced/PVL surfaces (lm.x==0) carry one flat baked
// colour per vertex, so the bump normal can't modulate the dominant light. This scales a
// mean-zero, n-vs-gn deviation term along the sun; identity where n==gn.
const BUMP_PVL_RELIEF: f32 = 0.8;
// Dynamic-object dominant-lobe soft-shade strength (geometric normal · scene sun), used when
// the map's airprobe directional fraction (light.sun_dir.w) is unset. Modest so it adds daylight
// shape (top/bottom, sun/shade side) without over-darkening the shadowed side.
const OBJ_DIR_SHADE: f32 = 0.35;
// si_ctl.x sentinel flagging an rmfl (foliage shader) part (scene.rs FOLIAGE_SI_MODE); outside
// the self_illumination enum range 0..12 the BSP path uploads verbatim.
const FOLIAGE_SI_MODE: i32 = 20;
// MeshVertex.tangent.w sentinel — an rmfl BSP vertex whose tangent.xyz carries the engine
// BACK-face per-vertex lighting dual_vmf_diffuse(-N) (scene.rs BSP_FOLIAGE_TAN_SENTINEL; ±1 = a real
// stored tangent frame, 0 = derivative fallback).
const BSP_FOLIAGE_TAN_SENTINEL: f32 = -4.0;
// Engine global directional BOUNCE fill for probe/SH-lit OBJECTS (entry_points.hlsl_include:438):
// `saturate(dot(bounce_dir,N))·bounce_intensity/π`. The HREK tag default
// render_bounce_light_intensity=10.0, but that raw value × the (unknown) bounce colour × exposure
// is what lands in the live k_ps_bounce_light_intensity cbuffer; without that capture 10/π≈3.18
// (× the ~3 HDR sun tint) would blow objects out. BOUNCE_FILL is a scaled-down stand-in so the
// fill is subtle (per-map colour = the sun tint, so max add ≈ sun_tint·0.02). TODO: capture the
// live k_ps_bounce_light_intensity; the STRUCTURE (dir·N Lambert fill, objects-only) is engine-faithful.
const BOUNCE_FILL: f32 = 0.05;
// Detail UV scale for materials with no authored detail_map xform.
const MESH_DETAIL_SCALE: f32 = 6.0;
// Surface-contrast gain for macro-tile floors (mean-preserving; 1.0 = no boost).
const TILE_SURFACE_CONTRAST: f32 = 1.0;
// Engine analytical specular (cook_torrance_core.hlsl_include):
//   D        = exp(-tan²α/m²) / (m²·NdH⁴ + 1e-5)        [NO π in the Beckmann denominator]
//   radiance = D·G·F / (N·V·π)                          [single π, NO 1/4, NO N·L]
//   strength = specular_coefficient·analytical_specular_contribution = `mat_spec` (folded)
// SPEC_CAL = 1.0: the specular term lives in the same pre-exposure absolute-HDR space as the
// diffuse (sun diffuse = sun_tint/π·ndl, entry_points:429; sun specular = D·G·F/(N·V·π)·
// specular_coefficient·analytical_specular_contribution·specular_mask·light_intensity at FULL
// magnitude, no /π, entry_points:455). SPEC_MAX_SCALAR clamps the raw D·G/(N·V·π) microfacet
// weight (the engine relies on the sqrt tonemap saturate() to clip metal glints to white).
const SPEC_PI: f32 = 3.141592658;
const SPEC_CAL: f32 = 1.0;
const SPEC_MAX_SCALAR: f32 = 4.0;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(9) sway: vec3<f32>,
    @location(13) uv2: vec2<f32>,
    @location(22) tangent: vec4<f32>,  // G1 stored per-vertex tangent frame [T.xyz, handedness]; [0;4]=derivative fallback
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
    @location(10) scroll: vec4<f32>,
    @location(11) detail_xform: vec4<f32>,
    @location(12) aux: vec4<f32>,
    @location(14) lm: vec4<f32>,
    @location(15) spec2: vec4<f32>,  // [metalness, fresnel_steepness, spec_tint_luma, fres_color_luma]
    @location(16) bump_xform: vec4<f32>,  // authored bump_map UV xform [tile.xy, off.xy]; [0;4]=fallback
    @location(17) fine_xform: vec4<f32>,  // fine self-detail xform [tile.xy, flag, _]
    @location(18) bump_detail_xform: vec4<f32>,  // bump_detail_map xform [tile.xy, flag, _]
    @location(19) env_ctl: vec4<f32>,  // [env_tint_color.rgb, specular_coefficient]
    @location(20) matmodel: f32,  // material_model+1 (0=unset); negative = a Halo 4 material
    @location(21) si_ctl: vec4<f32>,  // [self_illum mode, cmul.rgb]
    @location(23) mattex: vec4<f32>,  // [black_rough, black_specmult, has_flag, mode]
    @location(24) mattex_xform: vec4<f32>,  // material_texture UV xform [tile.xy, off.xy]
    @location(25) xbump: vec4<f32>,  // extended detail-bump tiles [bd2.tile.xy, bd3.tile.xy]
    @location(26) xctl: vec4<f32>,  // [bd2_present, bd3_present, env_authored, snorm bits]
    @location(27) obj_probe0: vec4<f32>,  // OBJ-PROBE [dom_dir.xyz, bandwidth]
    @location(28) obj_probe1: vec4<f32>,  // OBJ-PROBE [dom_rgb, mask]
    @location(29) spec_rgb: vec4<f32>,  // specular_tint colour (w=1)
    @location(30) fres_rgb: vec4<f32>,  // fresnel_color colour (w=1)
    @location(31) obj_light: vec4<u32>,  // packed [oct(fill_dir), oct(bounce_dir), bounce.rg, (bounce.b, flag)]
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
    @location(3) vdepth: f32,
    @location(4) world_pos: vec3<f32>,
    @location(5) scroll: vec2<f32>,
    @location(6) tile: vec2<f32>,
    @location(7) detail_xform: vec4<f32>,
    @location(8) uv_density: f32,
    @location(9) spec_from_alpha: f32,
    @location(10) mat_spec: f32,
    @location(11) mat_rough: f32,
    @location(12) uv2: vec2<f32>,
    @location(13) lm: vec4<f32>,
    @location(14) mat_spec2: vec4<f32>,  // artist-Fresnel material constants
    @location(15) bump_xform: vec4<f32>,  // authored bump_map UV xform
    @location(16) fine_xform: vec4<f32>,  // fine self-detail xform
    @location(17) bump_detail_xform: vec4<f32>,  // bump_detail_map xform
    @location(19) itint: vec4<f32>,  // RAW per-instance colour (NO per-vertex baked ambient folded in)
    @location(20) env_ctl: vec4<f32>,  // [env_tint_color.rgb, specular_coefficient]
    @location(18) matmodel: f32,  // material_model+1 (0=unset); negative = a Halo 4 material
    @location(21) si_ctl: vec4<f32>,  // [self_illum mode, cmul.rgb]
    @location(22) tangent: vec4<f32>,  // G1 world-space stored tangent [T.xyz, handedness]; [0;4]=derivative fallback
    @location(23) mattex: vec4<f32>,  // [black_rough, black_specmult, has_flag, mode]
    @location(24) mattex_xform: vec4<f32>,  // material_texture UV xform [tile.xy, off.xy]
    @location(25) xbump: vec4<f32>,  // extended detail-bump tiles [bd2.tile.xy, bd3.tile.xy]
    @location(26) xctl: vec4<f32>,  // [bd2_present, bd3_present, env_authored, snorm bits]
    @location(27) obj_probe0: vec4<f32>,  // OBJ-PROBE [dom_dir.xyz, bandwidth]
    @location(28) obj_probe1: vec4<f32>,  // OBJ-PROBE [dom_rgb, mask]
    @location(29) spec_rgb: vec4<f32>,  // specular_tint colour (w=1)
    @location(30) fres_rgb: vec4<f32>,  // fresnel_color colour (w=1)
    @location(31) @interpolate(flat) obj_light: vec4<u32>,  // packed object-lighting lane (per instance)
};

// Octahedral direction decode (matches the CPU `oct_encode` in scene.rs).
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-v.z, 0.0);
    v.x = v.x + select(t, -t, v.x >= 0.0);
    v.y = v.y + select(t, -t, v.y >= 0.0);
    let l = length(v);
    return select(vec3<f32>(0.0, 0.0, 1.0), v / l, l > 1e-6);
}
// The packed per-instance object-lighting lane (scene.rs `probe_lanes`):
//   .x = 4x8 snorm [oct(fill_dir).xy, oct(bounce_dir).xy]
//   .y = 4x8 unorm [bounce_to_ambient/4, bounce_scale, max_contrast/4, max_contrast_distant/4]  (chmt per object type)
//   .z = f16x2 [bounce.r, bounce.g]  (RAW 10·A·E bounce from the ray-hit surface)
//   .w = f16x2 [bounce.b, flag], flag = base(1 = fill dir valid, 2 = fill dir ZERO) + 4·round(min_lum·500)
// valid == false (flag 0) = no engine sample (key = dominant lobe × mask).
struct ObjLightLane { valid: bool, fill_dir_zero: bool, fill_dir: vec3<f32>, bounce_dir: vec3<f32>, bounce: vec3<f32>, min_lum: f32, bta: f32, bs: f32, mc: f32, mcd: f32 }
fn obj_light_unpack(ol: vec4<u32>) -> ObjLightLane {
    var o: ObjLightLane;
    let w = unpack2x16float(ol.w);
    let flag = i32(round(w.y));
    o.valid = flag > 0;
    let base = flag & 3;
    o.fill_dir_zero = base == 2;
    o.min_lum = f32(flag >> 2) / 500.0;
    let d = unpack4x8snorm(ol.x);
    o.fill_dir = select(oct_decode(d.xy), vec3<f32>(0.0), o.fill_dir_zero);
    o.bounce_dir = oct_decode(d.zw);
    let brg = unpack2x16float(ol.z);
    o.bounce = vec3<f32>(brg.x, brg.y, w.x);
    let c = unpack4x8unorm(ol.y);
    o.bta = c.x * 4.0;
    o.bs = c.y;
    o.mc = c.z * 4.0;
    o.mcd = c.w * 4.0;
    return o;
}
// The engine's per-frame chmt post-process of an object's lighting constants (sub_1407E9430,
// docs/hrek_re/16_object_lighting.md §2b), given the RAW sample (col0/bw0 dominant lobe, col1 fill, the raw
// bounce, the merge gate b and the scene sun colour). Returns [col0', col1', bounce'] (c[1].rgb, c[3].rgb,
// k_ps_bounce_light_intensity):
//   1. col1 += bounce_to_ambient · bounce;  bounce *= bounce_scale
//   2. max contrast (bipeds..equipment): mcx = lerp(mc, mc_distant, sat((dist − 3)/22)); if
//      (bw0·lum(col0) + b·lum(sun)) / (lum(col1) + (1−bw0)·lum(col0)) > mcx:  k = (1 − mcx/ratio)/mcx,
//      col1 += k·col0 + k·b·sun            (lum = .299/.587/.114)
//   3. min luminance: s = max(min_lum / (2^stops · (.3,.59,.11)·(col0 + col1 + b·sun)/(2π)), 1);
//      col0, col1, bounce *= s.   The merged analytical light is NOT scaled (it was built from the raw sample).
struct ObjChmtOut { col0: vec3<f32>, col1: vec3<f32>, bounce: vec3<f32> }
fn obj_chmt_apply(col0: vec3<f32>, bw0: f32, col1_raw: vec3<f32>, bounce_raw: vec3<f32>, b: f32, sunc: vec3<f32>, cam_dist: f32, ol: ObjLightLane) -> ObjChmtOut {
    var o: ObjChmtOut;
    o.col0 = col0;
    o.col1 = col1_raw + ol.bta * bounce_raw;
    o.bounce = ol.bs * bounce_raw;
    let t = clamp((cam_dist - 3.0) / 22.0, 0.0, 1.0);
    let mcx = ol.mc + t * (ol.mcd - ol.mc);
    if (mcx > 1e-4) {
        let lw = vec3<f32>(0.299, 0.587, 0.114);
        let lum0 = dot(col0, lw);
        let lum1 = dot(o.col1, lw);
        let lums = dot(sunc, lw);
        let direct = bw0 * lum0 + b * lums;
        let ambient = max(lum1 + (1.0 - bw0) * lum0, 1e-4);
        let ratio = direct / ambient;
        if (ratio > mcx) {
            let k = (1.0 - mcx / ratio) / mcx;
            o.col1 = o.col1 + k * col0 + (k * b) * sunc;
        }
    }
    if (ol.min_lum > 1e-4) {
        let lum = max(dot(o.col0 + o.col1 + b * sunc, vec3<f32>(0.3, 0.59, 0.11)) * 0.15915494, 1e-4);
        let s = max(ol.min_lum / (exposure_stops_gain() * lum), 1.0);
        o.col0 = o.col0 * s;
        o.col1 = o.col1 * s;
        o.bounce = o.bounce * s;
    }
    return o;
}

// Body shared by `vs` (every scene pass) and `vs_decal` (decal pipelines).
fn vs_impl(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var world = model * vec4<f32>(v.pos, 1.0);
    // ENGINE decorator wind: a shared WORLD-SPACE wind flow-field — coherent gusts
    // over the whole field, not a per-instance local sway. The phase is already world-XY-coherent (so
    // neighbouring blades move together); add a slow large-scale GUST that modulates the sway amplitude
    // so the field breathes in waves (the engine's DECORATOR_WIND flow-field). Height foreshortening is
    // carried by the per-vertex sway basis (v.sway magnitude ∝ blade height). Zero sway → untouched.
    let gust = 0.55 + 0.45 * sin(cam.time.x * 0.4 + world.x * 0.04 + world.y * 0.04);
    let phase = cam.time.x * 1.7 + world.x * 0.35 + world.y * 0.35;
    let amp = sin(phase) * gust;
    world.x += v.sway.x * amp;
    world.y += v.sway.y * amp;
    world.z += v.sway.z * amp;
    var nrm = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    // LIGHT VOLUME RIBBON (bump_xform.w == 4): the engine's light_volume vertex type builds
    // a camera-facing strip along the volume axis (camera_to_world in light_volume_shared_vertex_shaders).
    // Each vertex carries the CENTERLINE point in v.pos, the world axis in v.tangent.xyz and its signed
    // across offset (± half thickness) in v.tangent.w; the across direction is re-derived per frame as
    // cross(axis, to_camera) so the ribbon always faces the viewer without a CPU rebuild.
    if (v.bump_xform.w > 3.5 && v.bump_xform.w < 4.5) {
        let axis = normalize(v.tangent.xyz);
        let to_cam = cam.cam_pos.xyz - world.xyz;
        var across = cross(axis, to_cam);
        let al = length(across);
        let fallback = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.x) > 0.9);
        across = select(normalize(cross(axis, fallback)), across / al, al > 1e-5);
        world = vec4<f32>(world.xyz + across * v.tangent.w, 1.0);
        nrm = normalize(to_cam);
    }
    var o: VOut;
    o.clip = cam.view_proj * world;
    o.normal = nrm;
    // World-space stored tangent. Rotate T by the model matrix (no-op for BSP's identity
    // instance) and carry the handedness in .w. Not normalized here (zero tangent stays zero →
    // the fragment shader detects it and falls back to the derivative frame).
    o.tangent = vec4<f32>((model * vec4<f32>(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    o.uv = v.uv;
    o.tint = v.tint * v.vcolor;   // fold per-vertex baked colour into the tint (lit surfaces)
    o.itint = v.tint;             // RAW instance colour — holo markers must NOT be modulated by the
                                  // per-vertex airprobe ambient (it corrupted team-tint hues).
    o.vdepth = o.clip.w;   // linear view-space distance for aerial fog
    o.world_pos = world.xyz;
    o.scroll = v.scroll.xy;   // UV scroll rate/sec (waterfalls); 0 elsewhere
    o.tile = v.scroll.zw;     // per-material base-map UV tile (1,1 = untiled)
    o.detail_xform = v.detail_xform;  // opaque detail-map xform; [0;4] = fallback
    o.uv_density = v.aux.x;            // per-mesh UV density (unused by the shading)
    o.spec_from_alpha = v.aux.y;       // 1 = diffuse alpha is a real spec mask
    o.mat_spec = v.aux.z;              // per-material analytical spec strength (0 = unauthored)
    o.mat_rough = v.aux.w;             // per-material roughness (<0 = unauthored)
    o.uv2 = v.uv2;                     // submap-local lightmap UV
    o.lm = v.lm;                       // [flag,hdr,k,mode] GPU-lightmap params
    o.mat_spec2 = v.spec2;             // artist-Fresnel material constants
    o.bump_xform = v.bump_xform;       // authored bump_map UV xform ([0;4]=fallback)
    o.fine_xform = v.fine_xform;       // fine self-detail xform ([0;4]=none)
    o.bump_detail_xform = v.bump_detail_xform;  // bump_detail_map xform ([0;4]=none)
    o.env_ctl = v.env_ctl;             // [env_tint_color.rgb, specular_coefficient] ([0;4]=no-op)
    o.matmodel = v.matmodel;           // material_model+1 (0=unset)
    o.si_ctl = v.si_ctl;               // [self_illum mode, cmul.rgb]
    o.mattex = v.mattex;               // [black_rough, black_specmult, has_flag, mode]
    o.mattex_xform = v.mattex_xform;   // material_texture UV xform
    o.xbump = v.xbump;                 // extended detail-bump tiles
    o.xctl = v.xctl;                   // [bd2_present, bd3_present, env_authored, snorm bits]
    o.obj_probe0 = v.obj_probe0;
    o.obj_probe1 = v.obj_probe1;
    o.spec_rgb = v.spec_rgb;
    o.fres_rgb = v.fres_rgb;
    o.obj_light = v.obj_light;
    return o;
}

@vertex
fn vs(v: VIn) -> VOut { return vs_impl(v); }

// DECAL vertex shader: the geometry is built exactly COPLANAR with its receiver (engine
// rule, decal_projector.rs), so the depth test alone would z-fight. The engine resolves that at
// draw time with a per-decal VS depth bias that scales as 1/d^2 (decal_manager.cpp: bias =
// decal_bias_coeff_d / (d^2 + k), coeff_d = -4.166e-5 -- i.e. a CONSTANT world-space pull toward
// the camera, since NDC depth slope is ~near/d^2).
// HMS does the same in world space: nudge the vertex toward the camera by a fraction of a mm,
// growing with d^2 only as far as Depth32Float precision needs (n/w^2 depth slope, near = 0.05):
// 4 mm up to ~15 wu, 1 cm at 30 wu, 2.6 cm at 60 wu. Never a fixed NDC bias -- that pokes decals
// through nearer geometry at distance.
@vertex
fn vs_decal(v: VIn) -> VOut {
    var o = vs_impl(v);
    let to_cam = cam.cam_pos.xyz - o.world_pos;
    let d = length(to_cam);
    // Measured on Powerhouse (cam 17 wu from the wall): a 0.5 mm + 3-ulp nudge leaves a third
    // of the decal pixels BEHIND their own receiver (clipped sub-triangles interpolate depth a few
    // ulps off the big receiver triangle), so the constant is 4 mm. Overlays a few mm above the
    // receiver that this nudge would poke through are removed at build time instead
    // (decal_projector.rs emit_unoccluded).
    let nudge = clamp(0.004 + 6e-6 * d * d, 0.004, 0.15);
    let w = o.world_pos + to_cam * (min(nudge, d * 0.5) / max(d, 1e-5));
    o.clip = cam.view_proj * vec4<f32>(w, 1.0);
    o.vdepth = o.clip.w;
    return o;
}

// The GPU lightmap decode's result: the DOMINANT+FILL total (rgb) + baked sun visibility (a),
// plus the FILL-lobe contribution separately so the cast-shadow `darken` can be applied to the
// dominant lobe only and SPARE the fill lobe — the engine darkens only vmf[1] (dominant, by
// shadow_mask.r) and leaves 0.25·vmf[3] (fill) un-darkened (shadow_mask.hlsl_include:38 is
// commented out).
struct BakeLobes { total: vec4<f32>, fill: vec3<f32> };
// Halo 4 lightmap atlas, ENGINE decode (crates/hms-app/src/h4/lightmaps.rs module doc;
// VERIFIED against the shipped srf_blinn pixel shaders + halo4.dll sub_18038C1E0). The four bound
// textures are CPU-composed (compose_atlas_textures): dm = lobe A (direct) rgb + alpha, sdm = lobe B
// (indirect) rgb + alpha, sdm1 = direction A unorm + analytic sun visibility, sdm2 = direction B
// unorm + the engine's 64x64 sharpen_falloff LUT in the top-left corner of .w.
//   I_k   = K_k * (512 * exp2(-9 a_k) - 1) / 511          (k_a = Lbsp +0x14, k_b = +0x18)
//   u_k   = 0.2821 * sqrt(1 - |d_k|^2) + 0.3257 * dot(d_k, n)   (SH L1 irradiance / pi)
//   v_k   = the same with the VERTEX normal nv; f_k = LUT(u_k, v_k) (identity on the diagonal)
//   total = rgb_A * I_A * f_A + rgb_B * I_B * f_B ; fill = the lobe-B half (spared by the cast shadow)
//   .a    = sqrt(sat(2 vis - 0.5)): the engine's sharpened sun visibility (s = 1 on the MP maps);
//           the caller squares it (Reach law), so the sun gets exactly sat(2 vis - 0.5). k_b < 0 =
//           the BSP's floating sun is disabled -> .a = 0 (its sun is baked into the lobes).
// The engine's dynamic-shadow gate on lobe A (sat(mask.g + 0.3), only where vis = 0) is 1 here.
// Reached only through the hdr < 0 sentinel below.
fn h4_sharpen_lut(u: f32, v: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(lm_sdm2_tex, 0));
    let tx = clamp(u * 64.0, 0.5, 63.5);
    let ty = clamp(v * 64.0, 0.5, 63.5);
    return textureSampleLevel(lm_sdm2_tex, lm_samp, vec2<f32>(tx, ty) / dims, 0.0).w;
}
fn h4_lightmap_tint(uv2: vec2<f32>, n: vec3<f32>, nv: vec3<f32>, k_a: f32, k_b: f32) -> BakeLobes {
    let t0 = textureSampleLevel(lm_dm_tex,   lm_samp, uv2, 0.0);
    let t1 = textureSampleLevel(lm_sdm_tex,  lm_samp, uv2, 0.0);
    let t2 = textureSampleLevel(lm_sdm1_tex, lm_samp, uv2, 0.0);
    let t3 = textureSampleLevel(lm_sdm2_tex, lm_samp, uv2, 0.0);
    let da = t2.xyz * 2.0 - 1.0;
    let db = t3.xyz * 2.0 - 1.0;
    let wa = sqrt(max(1.0 - dot(da, da), 0.0));
    let wb = sqrt(max(1.0 - dot(db, db), 0.0));
    let ia = k_a * (512.0 * exp2(-9.0 * t0.w) - 1.0) / 511.0;
    let ib = abs(k_b) * (512.0 * exp2(-9.0 * t1.w) - 1.0) / 511.0;
    let nvn = normalize(nv);
    let ua = 0.2820948 * wa + 0.325735 * dot(da, n);
    let ub = 0.2820948 * wb + 0.325735 * dot(db, n);
    let va = 0.2820948 * wa + 0.325735 * dot(da, nvn);
    let vb = 0.2820948 * wb + 0.325735 * dot(db, nvn);
    let fa = h4_sharpen_lut(ua, va);
    let fb = h4_sharpen_lut(ub, vb);
    let fill = max(t1.rgb * ib * fb, vec3<f32>(0.0));
    let tint = max(t0.rgb * ia * fa, vec3<f32>(0.0)) + fill;
    // k_b < 0 = the BSP has no floating-sun record (Lbsp flag 0x200 clear): no analytic sun at all;
    // analytic.y (sdm2.w outside the LUT corner) multiplies the sharpened visibility
    let dims = vec2<f32>(textureDimensions(lm_sdm2_tex, 0));
    let in_corner = uv2.x * dims.x < 64.0 && uv2.y * dims.y < 64.0;
    let ana_y = select(t3.w, 1.0, in_corner);
    let vis = select(0.0, sqrt(clamp(2.0 * t2.w - 0.5, 0.0, 1.0) * ana_y), k_b > 0.0);
    return BakeLobes(vec4<f32>(tint, vis), fill);
}
fn gpu_lightmap_tint(uv2: vec2<f32>, n: vec3<f32>, nv: vec3<f32>, hdr: f32, k: f32) -> BakeLobes {
    // Halo 4 atlas sentinel (hdr = -K_direct); nv = the vertex normal for the sharpen LUT.
    // Reach meshes pass entry.hdr_scale clamped > 0 (scene.rs lm_hdr), so this branch never runs
    // for a Reach mesh and nv is unused there.
    if (hdr < 0.0) { return h4_lightmap_tint(uv2, n, nv, -hdr, k); }
    // HREK DecompressVMF (lightmap_sampling.hlsl_include, DX11 path):
    //   dir       = normalize( float3(sdm0.a, sdm1.a, sdm2.a)*2 - 1 )   // 3 SLICE ALPHAS
    //   sharpness = length( that vector )                                // vmf[1].a
    //   fIntensity= exp(-6.238325 * DM.r)                                // DM.r, not DM.a
    //   dom_color = (sdm0.rgb + sdm1.rgb*2 - 1) * COMPRESS * fIntensity  // dominant lobe
    //   fill_color= sdm2.rgb * COMPRESS * fIntensity                     // second lobe
    //   analytical/visibility mask = DM.a
    // COMPRESS (p_lightmap_compress_constant_0.x) is a global SCALE folded into the per-map `k`
    // (the hue fix is scale-independent — it comes from the 3-slice reconstruction, not magnitude).
    let d  = textureSampleLevel(lm_dm_tex,   lm_samp, uv2, 0.0);
    let s0 = textureSampleLevel(lm_sdm_tex,  lm_samp, uv2, 0.0);
    let s1 = textureSampleLevel(lm_sdm1_tex, lm_samp, uv2, 0.0);
    let s2 = textureSampleLevel(lm_sdm2_tex, lm_samp, uv2, 0.0);
    var dom_dir = vec3<f32>(s0.w * 2.0 - 1.0, s1.w * 2.0 - 1.0, s2.w * 2.0 - 1.0);
    let dl = length(dom_dir);
    if (dl > 1e-4) { dom_dir = dom_dir / dl; } else { dom_dir = vec3<f32>(0.0, 0.0, 1.0); }
    // lobe sharpness (bandwidth) = |unnormalized dir| = dl; used directly as the vMF LUT y below.
    // Intensity from DM.r (d.x). HREK lightmap_sampling.hlsl DecompressVMF writes
    // fIntensity = exp(-6.238325*mask.r), and exp(-6.238325*x) ≡ exp2(-9*x) (−6.238325 = −9·ln2);
    // the shipped shaders' captured −9.000001 is the exp2 coefficient.
    let f_int = exp2(-9.000001 * d.x);
    // The reconstructed lobe colours stay SIGNED in the engine: DecompressVMF
    // (lightmap_sampling.hlsl_include:66) keeps `Colors[0] = tex0.rgb + tex1.rgb*2 - 1` and
    // `Colors[1] = tex2.rgb` with NO max(…,0) — the `*2-1` residual is legitimately negative.
    // The final irradiance is still ≥0 downstream.
    let dom_col  = (s0.rgb + s1.rgb * 2.0 - 1.0) * f_int;  // dominant lobe colour (signed)
    let fill_col = s2.rgb * f_int;                          // fill lobe colour (signed)
    let vis = d.y;                                    // visibility = DM.GREEN (d.y), NOT alpha:
    // HLSL DecompressVMF: mask=DM.xyxy; visibility=mask.a=DM.y. DM is a 2-channel texture (no real
    // alpha → d.w≈1 would un-gate the analytical sun → fullbright enclosed floors).
    let ndotd = dot(dom_dir, n);
    let cc = clamp(ndotd * 0.5 + 0.5, 0.0, 1.0);
    // Engine dual_vmf_diffuse — dominant coeff = the byte-exact vMF LUT sampled at
    // (cc, clamp(bandwidth,0,1)); bandwidth = |unnormalized dominant dir| = dl. Replaces the
    // analytic 0.5-floor stand-in (which bled light onto back-faces and lost the correct
    // directional falloff). Matches lightprobe.rs::vmf_diffuse_coeff / docs/hrek_re/00 §3.
    let dom_co = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(cc, clamp(dl, 0.0, 1.0)), 0.0).r;
    let inv_pi = 0.31830989;
    var tint = vec3<f32>(0.0, 0.0, 0.0);
    var fill = vec3<f32>(0.0, 0.0, 0.0);
    for (var c = 0; c < 3; c = c + 1) {
        let irr = (dom_co * dom_col[c] + 0.25 * fill_col[c]) * inv_pi;
        // The baked term is NOT visibility-scaled: the engine never shadows the baked dual-VMF
        // (entry_points.hlsl); vis gates ONLY the analytical sun (downstream, via the returned .a),
        // consistently with the PVL path. No upper clamp either (dual_vmf_diffuse just returns
        // vmf_lighting/π), so absolute-HDR bakes reach the post exposure+saturate uncapped; the ≥0
        // floor stays (irradiance is non-negative).
        tint[c] = max(irr * hdr * k, 0.0);
        // The FILL-lobe contribution (0.25·fill_col/π), in the same linear units as `tint` and
        // signed (fill_col is signed, engine-faithful), returned so the caller can spare it from
        // the cast-shadow darken. The split is algebraic: (total-fill)·1 + fill == total.
        fill[c] = 0.25 * fill_col[c] * inv_pi * hdr * k;
    }
    return BakeLobes(vec4<f32>(tint, vis), fill);
}

// Environment cubemap reflection. Samples the mipped captured-sky cube at the mirror direction
// with LOD = roughness·maxLod, so a ROUGH surface reads a BLURRED reflection (the engine's
// `dynamic_environment_map` roughness→mip behaviour), re-applying the map's sun tint + a soft
// sun disc; or, for a material with an authored cube, the engine per_pixel decode.
fn env_reflection(refl: vec3<f32>, rough: f32, lod_bias: f32, env_tint: vec3<f32>, authored: f32) -> vec3<f32> {
    // environment_mapping.hlsl_include:69-70: a REAL per-material `environment_map` cube is bound
    // at binding 10. Decode the engine per_pixel way — flip reflect_dir.y (cube-face handedness),
    // then `reflection.rgb · reflection.a` (the HDR range is packed in the per-texel ALPHA; this
    // is NOT ×256, which is the water-static path). per_pixel LOD is roughness-INDEPENDENT (the
    // engine passes base_roughness=0), so sample near mip 0 with only the distance lod_bias
    // (far-sparkle guard). Multiply by the authored env_tint_color (white when unauthored).
    if (authored > 0.5) {
        let dir = vec3<f32>(refl.x, -refl.y, refl.z);
        let s = textureSampleLevel(env_cube, env_cube_samp, dir, min(lod_bias, 6.0));
        let tint = select(vec3<f32>(1.0), env_tint, dot(env_tint, vec3<f32>(1.0)) > 1e-4);
        return s.rgb * s.a * tint;
    }
    // Distance LOD bias: textureSampleLevel picks a FIXED mip from roughness alone, ignoring
    // minification. On distant glossy surfaces (snow) the mirror direction varies fast across the
    // partially-bumped normal and a sharp full-res cubemap sample aliases into blue/purple sky
    // speckle; the distance LOD forces far reflections to a blurred mip.
    var base = textureSampleLevel(env_cube, env_cube_samp, refl, min(clamp(rough, 0.0, 1.0) * 6.0 + lod_bias, 6.0)).rgb;
    // The env cube is BLACK-initialised so an UNCAPTURED face is detectable: where the cube
    // reads ~black (no sky captured for this direction), fall back to a procedural atmosphere
    // gradient (inlined — the WATER_WGSL sky_radiance isn't in this module's scope) so
    // reflections vary with the sky instead of one constant colour.
    if (dot(base, vec3<f32>(0.333)) <= 0.002) {
        let up = clamp(refl.z, 0.0, 1.0);
        let sky = mix(vec3<f32>(0.55, 0.68, 0.85), vec3<f32>(0.24, 0.45, 0.78), up);
        base = select(vec3<f32>(0.14, 0.15, 0.14), sky, refl.z >= 0.0);
    }
    // environment_mapping.hlsl_include:74: the engine multiplies the cube reflection by the
    // authored `env_tint_color` (float3). When a material authors env_tint_color (env_tint != 0)
    // apply it as the engine does; otherwise hue-normalise by the map's sun tint so reflections
    // agree with the map's actual sky (brightness-preserved).
    let st = light.sun_tint.rgb;
    let stn = st / max((st.r + st.g + st.b) / 3.0, 1e-4);
    let has_tint = dot(env_tint, vec3<f32>(1.0)) > 1e-4;
    let tint = select(stn, env_tint, has_tint);
    let sd = normalize(light.sun_dir.xyz);
    let d = max(dot(refl, sd), 0.0);
    return base * tint + vec3<f32>(1.0, 0.9, 0.75) * pow(d, 8.0) * 0.15 * (1.0 - clamp(rough, 0.0, 1.0));
}

// The Reach change-colour palette (CHANGE_COLORS_DEFAULT in scene.rs, kept in sync). Exact
// authored values from game_globals ('matg' meta+0x4D4, real_argb_color[8], RGB@+4).
// 0 red,1 blue,2 green,3 orange,4 purple,5 gold,6 brown,7 pink,8 white,9 black,10 zombie.
// This static table is the GPU-side copy for opaque object change-colour tinting; the
// runtime matg-populated table only feeds the CPU (holo) path.
fn change_color_palette(idx: i32) -> vec3<f32> {
    switch (idx) {
        case 0:  { return vec3<f32>(0.5029, 0.0437, 0.0437); }
        case 1:  { return vec3<f32>(0.0902, 0.2157, 0.5255); }
        case 2:  { return vec3<f32>(0.2039, 0.3176, 0.0627); }
        case 3:  { return vec3<f32>(0.8431, 0.3490, 0.0431); }
        case 4:  { return vec3<f32>(0.2078, 0.1216, 0.5490); }
        case 5:  { return vec3<f32>(0.6902, 0.8706, 0.1059); }
        case 6:  { return vec3<f32>(0.4706, 0.4000, 0.1059); }
        case 7:  { return vec3<f32>(0.8995, 0.6173, 0.6746); }
        case 8:  { return vec3<f32>(1.0000, 1.0000, 1.0000); }
        case 9:  { return vec3<f32>(0.0500, 0.0500, 0.0500); }
        case 10: { return vec3<f32>(0.4000, 0.5000, 0.2000); }
        default: { return vec3<f32>(1.0000, 1.0000, 1.0000); }
    }
}

// Shared shading body — used by the opaque `fs` and the alpha-test `fs_alphatest` (foliage
// cutout) entry points. The composite is the engine's (entry_points.hlsl_include):
// albedo·(baked dual-VMF probe + analytical sun/π·visibility) + specular + self-illum + env.
fn mesh_shade(i: VOut) -> vec4<f32> {
    // Halo 4 materials (matmodel = -(1 + spec model), never < 0 for Reach) take the
    // shipped-shader lane in H4_MESH_WGSL and skip every Reach term below.
    if (i.matmodel < -0.5) { return h4_shade(i); }
    // Per-material base-map UV tile (engine base_map_xform.xy), then the animated
    // scroll: waterfalls carry a per-instance rate; everything else scroll=0. Tile
    // defaults to (1,1) so untiled surfaces are unchanged.
    let auv = i.uv * i.tile + i.scroll * cam.time.x;
    let gn = normalize(i.normal);
    // The baked lighting: the per-vertex i.tint, or (atlas meshes, lm.x>0.5) the dual-VMF
    // irradiance decoded from DM/SDM at uv2 BELOW, after the bump normal `n` is derived, so the
    // baked DIRECTIONAL light shades the normal map and tiles/concrete read 3D.
    var bake = i.tint;
    // a change-colour index rides the instance alpha as -(idx+2); it is NOT visibility.
    if (i.tint.a < -1.5) { bake.a = 1.0; }
    // the spared FILL-lobe contribution (atlas path only). Stays 0 for the PVL path (a
    // per-vertex single tint has no separable fill lobe → the cast-shadow darken applies fully).
    var bake_fill = vec3<f32>(0.0);
    // Bump/normal mapping: perturb the geometric normal by the tangent-space bump map. Texture
    // sample + derivatives stay in uniform control flow (WGSL rule); only the arithmetic is
    // branched. A flat-normal default map (128,128,255 → ≈0) leaves n unchanged.
    // The engine samples the normal at bump_map_xform — a register DISTINCT from base_map_xform
    // (Sapien capture: base=(1,4), detail=(2,8), bump has its own reg; panopticon floor mat 49
    // bump_map=[1,1] = 10× the base 0.10). buv = uv·bump_xform.xy + bump_xform.zw when authored,
    // else auv. tile is a positive per-axis scale, so the auv-derived tangent frame below is still
    // the correct world basis (only the sample scale changes, not the u/v world directions). The
    // authored layers scroll with the base (waterfalls); i.scroll is [0,0] elsewhere.
    let has_bxf = (abs(i.bump_xform.x) + abs(i.bump_xform.y)) > 1e-6;
    let buv = select(auv, i.uv * i.bump_xform.xy + i.bump_xform.zw + i.scroll * cam.time.x, has_bxf);
    let base_bump = textureSample(bump_tex, base_samp, buv);
    // raw BC5_SNORM bumps sample signed (engine: no *2-1); CPU-decoded bumps are 0.5-centred.
    let bsn_bits = u32(i.xctl.w + 0.5);
    var bump_s = select(base_bump.xy * 2.0 - 1.0, base_bump.xy, (bsn_bits & 1u) != 0u);
    // bump_mapping.hlsl:96-97: the engine keeps the BASE map's reconstructed Z through the detail
    // combine — `normalize(x+dx, y+dy, sqrt(1-x²-y²))` — the detail sample's Z is discarded.
    let base_bz = sqrt(max(1.0 - dot(bump_s, bump_s), 0.0));
    // SECOND normal — bump_detail_map. HREK calc_bumpmap_detail_ps: `bump.xy += detail.xy`
    // (the panopticon floor authors bump_detail_map at xform [10,10] = 100× base — a very fine
    // micro-normal). bump_detail_xform.z>0.5 gates it. extended_bump_mapping.hlsl_include:4,26:
    // the detail_blend / three_detail_blend variants blend detail1/2/3 by the BASE bump map's
    // alpha (base.w) then `bump.xy += detail.xy`; xctl.x/.y flag bump_detail_map2/3.
    if (i.bump_detail_xform.z > 0.5) {
        let d1r = textureSample(bump_detail_tex, base_samp, i.uv * i.bump_detail_xform.xy).xy; let d1 = select(d1r * 2.0 - 1.0, d1r, (bsn_bits & 2u) != 0u);
        var detail = d1;
        let has2 = i.xctl.x > 0.5;
        let has3 = i.xctl.y > 0.5;
        if (has3) {
            // three_detail_blend: blend1=sat(2·bw); blend2=sat(2·bw-1); lerp(lerp(d1,d2,blend1),d3,blend2)
            let bw = base_bump.w;
            let d2r = textureSample(bump_detail2_tex, base_samp, i.uv * i.xbump.xy).xy; let d2 = select(d2r * 2.0 - 1.0, d2r, (bsn_bits & 4u) != 0u);
            let d3r = textureSample(bump_detail3_tex, base_samp, i.uv * i.xbump.zw).xy; let d3 = select(d3r * 2.0 - 1.0, d3r, (bsn_bits & 8u) != 0u);
            let b1 = saturate(2.0 * bw);
            let b2 = saturate(2.0 * bw - 1.0);
            detail = mix(mix(d1, d2, b1), d3, b2);
        } else if (has2) {
            // detail_blend: detail = lerp(detail1, detail2, base.w)
            let d2r = textureSample(bump_detail2_tex, base_samp, i.uv * i.xbump.xy).xy; let d2 = select(d2r * 2.0 - 1.0, d2r, (bsn_bits & 4u) != 0u);
            detail = mix(d1, d2, base_bump.w);
        }
        bump_s = bump_s + detail;
    }
    let dpx = dpdx(i.world_pos);
    let dpy = dpdy(i.world_pos);
    let dux = dpdx(auv);
    let duy = dpdy(auv);
    let bdet = dux.x * duy.y - duy.x * dux.y;
    var n = gn;
    // Build the TBN from the STORED per-vertex tangent when present (engine uses the IA-decoded
    // world tangent, `binormal = safe_normalize(cross(N,T)·handedness)`), falling back to the
    // screen-space-derivative Gram-Schmidt frame for meshes with no stored tangent (objects/sky/
    // decals → i.tangent==0). Derivatives (dpx/dux/bdet) stay computed in uniform control flow above.
    let t_len = length(i.tangent.xyz);
    let have_tan = t_len > 1e-4;
    if (dot(bump_s, bump_s) > 4e-3 && (have_tan || abs(bdet) > 1e-8)) {
        var tng: vec3<f32>;
        var bit: vec3<f32>;
        if (have_tan) {
            // Engine stored frame: orthonormalize T vs the (interpolated) geometric normal, then
            // binormal = cross(N,T)·handedness. More stable than the view-dependent derivative frame.
            tng = normalize(i.tangent.xyz - gn * dot(gn, i.tangent.xyz));
            bit = normalize(cross(gn, tng) * i.tangent.w);
        } else {
            let r = 1.0 / bdet;
            var t = (dpx * duy.y - dpy * dux.y) * r;
            tng = normalize(t - gn * dot(gn, t));   // Gram-Schmidt vs geometric normal
            bit = normalize(cross(gn, tng));
        }
        // reconstruct from the BASE map's Z (captured above), not the summed xy. The engine
        // applies the bump normal at FULL strength at all distances (it relies on the mip chain,
        // not a per-pixel fade-to-geometric; bump-detail-normal.md 7.2) and derives relief ONLY
        // from authored bump / detail_bump maps (7.3).
        let bumped = normalize(tng * bump_s.x + bit * bump_s.y + gn * base_bz);
        n = bumped;
    }
    // Decode the baked directional lightmap with the bump-perturbed normal `n` so the dual-VMF
    // dominant-direction term (dot(dom_dir, n)) varies per normal-map texel → visible 3D relief
    // on tiled concrete/floors under baked lighting. Atlas-lit meshes only (lm.x>0.5);
    // per-vertex-baked meshes keep i.tint (they carry no per-pixel directional term).
    if (i.lm.x > 0.5) { let lobes = gpu_lightmap_tint(i.uv2, n, i.normal, i.lm.y, i.lm.z); bake = lobes.total; bake_fill = lobes.fill; }
    // Analytical sun = the shadow-map sun direction, so the real-time cast shadow gates the same
    // term that brightens sunlit faces. A DYNAMIC object's analytical light is NOT the scene sun
    // (52_ivory_tower frame_1 LightingPS/MeshPS capture): its direction equals the object probe's
    // dominant direction (c0.xyz) and its intensity is the dominant colour (c1.rgb) scaled by a
    // per-object visibility scalar; the scene sun only reaches lightmapped geometry. So for
    // probe-lit objects the key light becomes the probe's dominant lobe and its colour
    // dom_rgb × mask (vmf[0].w — the only per-position visibility term).
    let obj_probe_lit = i.lm.w > 1.5 && i.obj_probe0.w > 0.0;
    var key = select(normalize(light.sun_dir.xyz), normalize(i.obj_probe0.xyz), obj_probe_lit);
    var sun_col = select(light.sun_tint.rgb, i.obj_probe1.rgb * clamp(i.obj_probe1.w, 0.0, 1.0), obj_probe_lit);
    // Engine object lighting (docs/hrek_re/16_object_lighting.md §2, sub_140828CD0 + sub_1407E9430 +
    // sub_14087E4E0): when the packed object-lighting lane is valid, the object's ANALYTICAL light
    // is the lobes+sun MERGE —
    //   f0 = Σcol0/(Σcol0+Σcol1), Ls = Σsun·mask0, b = min(mask0, Ls/(Ls+Σcol0+Σcol1)),
    //   dir = normalize(b·sun_dir + (1−b)·normalize(f0·dir0 + f1·dir1)), colour = b·sun + (1−b)·(f0·col0 + f1·col1)
    // — applied through the existing sun_term (c[0].w = 1 → no mask factor; a fully shadowed sample
    // makes the key light the dominant lobe itself). The BOUNCE light from the hit surface is added
    // ×2 to the fill lobe (bounce-to-ambient) and as a third light sat(N·bounce_dir)·I/π below.
    let oll = obj_light_unpack(i.obj_light);
    let ol_flag = select(0.0, select(1.0, 2.0, oll.fill_dir_zero), oll.valid);
    let obj_engine = obj_probe_lit && oll.valid;
    var obj_fill = i.tint.rgb;
    // the dominant-lobe colour the LUT term uses (c[1].rgb) — scaled by the chmt min-luminance lift
    var obj_dom = i.obj_probe1.rgb;
    // the engine's analytical-light seed gate for objects (lighting_coefficients[2].w = b).
    var obj_b = clamp(i.obj_probe1.w, 0.0, 1.0);
    var obj_bounce_dir = vec3<f32>(0.0, 0.0, 1.0);
    var obj_bounce_col = vec3<f32>(0.0);
    if (obj_engine) {
        // flag 2 = the sample's fill-lobe direction is ZERO (lightmap texel / PVL) → lobe dir = dir0.
        let dir1 = oll.fill_dir;
        obj_bounce_dir = oll.bounce_dir;
        let col0 = i.obj_probe1.rgb;
        let col1 = i.tint.rgb;
        let l0 = col0.r + col0.g + col0.b;
        let l1 = col1.r + col1.g + col1.b;
        let tot = max(1e-5, l0 + l1);
        let f0 = l0 / tot;
        let f1 = 1.0 - f0;
        let mask0 = clamp(i.obj_probe1.w, 0.0, 1.0);
        let sunc = light.sun_tint.rgb;
        let ls = (sunc.r + sunc.g + sunc.b) * mask0;
        var b = mask0;
        if (tot + ls > 1e-4) { b = min(b, ls / (tot + ls)); }
        let dir0 = normalize(i.obj_probe0.xyz);
        let ld = f0 * dir0 + f1 * dir1;
        let lobe_dir = select(dir0, ld / max(length(ld), 1e-6), length(ld) > 1e-5);
        let ad = b * normalize(light.sun_dir.xyz) + (1.0 - b) * lobe_dir;
        key = select(dir0, ad / max(length(ad), 1e-6), length(ad) > 1e-5);
        sun_col = b * sunc + (1.0 - b) * (f0 * col0 + f1 * col1);
        // chmt per-object-type post-process (bounce-to-ambient / bounce scale / max contrast /
        // min luminance) on the CONSTANTS; the merged analytical light above stays raw (engine order).
        let ch = obj_chmt_apply(col0, clamp(i.obj_probe0.w, 0.0, 1.0), col1, oll.bounce, b, sunc, length(cam.cam_pos.xyz - i.world_pos), oll);
        obj_dom = ch.col0;
        obj_fill = ch.col1;
        obj_bounce_col = ch.bounce;
        obj_b = b;
    }
    // PER-VERTEX-LIT RELIEF: instanced/PVL meshes (lm.x==0) carry ONE flat baked colour per
    // vertex (bake=i.tint), so the bump normal `n` never reaches the dominant baked light and
    // tiled concrete/floors would render FLAT. Modulate the flat bake by the normal's deviation
    // from the geometric normal along the sun: tile faces tilted toward the sun brighten, those
    // tilted away darken. The delta averages ~0 over a bumpy tile (mean-preserving) and is
    // IDENTITY where n==gn. Atlas-lit meshes (lm.x>0.5) already get per-pixel relief inside
    // gpu_lightmap_tint, so they are skipped.
    if (i.lm.x <= 0.5) {
        var relief = BUMP_PVL_RELIEF * (dot(n, key) - dot(gn, key));
        // OBJECT dominant-lobe soft-shade: dynamic objects (lm.w==2) carry the per-position
        // airprobe ambient in i.tint but NO baked directional term. The engine shades them by the
        // light-probe's dominant lobe → add a geometric-normal term facing the key light, using
        // the FULL normal so the whole object gets top/bottom + sun-side/shade-side shape.
        // Mean-preserving over the object and gated to objects, so per-vertex-baked BSP (which
        // already has baked directional) is NOT double-lit. Strength = the map's own airprobe
        // directional fraction (light.sun_dir.w), so a strong-sun map gets crisp object shading
        // and an overcast/enclosed one stays soft; OBJ_DIR_SHADE when unset.
        if (i.lm.w > 1.5) {
            let dstr = select(OBJ_DIR_SHADE, light.sun_dir.w, light.sun_dir.w > 0.001);
            relief = relief + dstr * dot(n, key);
        }
        bake = vec4<f32>(i.tint.rgb * clamp(1.0 + relief, 0.4, 1.8), select(i.tint.a, 1.0, i.tint.a < -1.5));
        // Engine per-object lighting (docs/hrek_re/00 §4): irradiance(N) = [LUT(N.domDir, k).domRgb +
        // 0.25*fill]/pi evaluated per PIXEL from the object's blended probe (i.tint.rgb carries the
        // fill lobe), replacing the flat up-facing tint + colourless OBJ_DIR_SHADE for dynamic
        // objects when the lane is present. fill = col1 + bounce_to_ambient·bounce (chmt); + the
        // bounce as a third light sat(N·(−N_hit))·I_bounce/π (entry_points.hlsl static_sh path,
        // k_ps_bounce_light).
        if (i.lm.w > 1.5 && i.obj_probe0.w > 0.0) {
            let pd = normalize(i.obj_probe0.xyz);
            let pcc = clamp(dot(pd, n) * 0.5 + 0.5, 0.0, 1.0);
            let pco = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(pcc, clamp(i.obj_probe0.w, 0.0, 1.0)), 0.0).r;
            let third = max(dot(n, obj_bounce_dir), 0.0) * obj_bounce_col;
            bake = vec4<f32>((pco * obj_dom + 0.25 * obj_fill + third) * 0.31830989, select(i.tint.a, 1.0, i.tint.a < -1.5));
        }
    }
    // Real-time probe GI replaces the baked term (same irradiance/π units); the analytical
    // light becomes the scene sun again (probe-lit objects otherwise key off their baked lobe).
    if (gi_on()) {
        let gi_v = normalize(cam.cam_pos.xyz - i.world_pos);
        let gi_e = gi_irradiance(i.world_pos, n, gi_v);
        bake = vec4<f32>(gi_e, 1.0);
        bake_fill = gi_e;
        key = normalize(light.sun_dir.xyz);
        sun_col = light.sun_tint.rgb;
    }
    var ndl = max(dot(n, key), 0.0);
    // rmfl (foliage shader) OBJECT part, flagged si_ctl = [FOLIAGE_SI_MODE, foliage_translucency,
    // material type]. Engine foliage.hlsl_include single_pass_common_ps / static_common_ps: the vertex
    // shader evaluates the probe dual-vMF at +N (front) AND -N (back); the pixel shader picks the side
    // facing the camera (`lighting_blend` = vertex normal vs the derivative normal) and adds the other
    // side × translucency — MATERIAL_TYPE_flat (0): front + back outright; specular (1): t =
    // foliage_translucency; translucent (2): t = saturate((1 − alpha) · 2 · foliage_translucency). The
    // analytical (sun/key) dot is the same two-sided sum — sat(L·N) + t·sat(−L·N) — and the foliage
    // shader adds no Cook-Torrance lobe (killed below via mm_kill_spec; its shadow_mask-free gate is NOT
    // ported, see `shadow`). Only the probe-lit object lane (the tint carries the fill lobe, obj_probe =
    // dominant lobe) is ported; the BSP foliage path (its own si_ctl) and every non-rmfl mesh are
    // untouched (`rmfl` false).
    let rmfl = i32(round(i.si_ctl.x)) == FOLIAGE_SI_MODE;
    if (rmfl && i.lm.w > 1.5 && i.obj_probe0.w > 0.0 && !gi_on()) {
        let to_cam = normalize(cam.cam_pos.xyz - i.world_pos);
        let facing = select(-gn, gn, dot(gn, to_cam) >= 0.0);
        let ftype = i32(round(i.si_ctl.z));
        var t = clamp(i.si_ctl.y, 0.0, 1.0);
        if (ftype == 2) { t = clamp((1.0 - textureSampleGrad(base_tex, base_samp, auv, dux, duy).a) * 2.0 * i.si_ctl.y, 0.0, 1.0); }
        if (ftype == 0) { t = 1.0; }
        let pd = normalize(i.obj_probe0.xyz);
        let kk = clamp(i.obj_probe0.w, 0.0, 1.0);
        let pcf = clamp(dot(pd, facing) * 0.5 + 0.5, 0.0, 1.0);
        let lf = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(pcf, kk), 0.0).r;
        let lb = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(1.0 - pcf, kk), 0.0).r;
        let vmf_front = lf * obj_dom + 0.25 * obj_fill;
        let vmf_back = lb * obj_dom + 0.25 * obj_fill;
        bake = vec4<f32>((vmf_front + t * vmf_back) * 0.31830989, bake.a);
        ndl = max(dot(facing, key), 0.0) + t * max(-dot(facing, key), 0.0);
    }
    // rmfl BSP / instanced-geometry foliage (si_ctl.w >= 1, lm.w < 1.5, per-vertex or
    // probe tier — the engine has NO per-pixel/atlas foliage entry point, static_per_pixel_vs outputs
    // red as a warning). Engine foliage.hlsl_include single_pass_per_vertex_vs + single_pass_common_ps:
    //   VS: front = dual_vmf_diffuse(+N)  (i.tint.rgb, CPU decode at the STORED vertex normal)
    //       back  = dual_vmf_diffuse(-N)  (tangent.xyz when tangent.w is the sentinel; flat tiers = front)
    //       front.w = mask·sat(L·N), back.w = mask·sat(-L·N)   (mask = vmf[0].w = baked vis, applied ONCE)
    //   PS: flat (0):        lighting = front + back, dot = front.w + back.w  (= mask·|L·N|)
    //       specular (1) / translucent (2): blend = (dot(N, geometric_normal) + 1) / 2 picks the side
    //         facing the camera; lighting = lerp(back, front, blend) + t·lerp(front, back, blend), same
    //         for the analytical dot; t = foliage_translucency (specular) or
    //         saturate((1 - alpha)·2·foliage_translucency) (translucent), alpha = the alpha-test alpha
    //         (albedo.a = base.a × detail.a for albedo_default).
    //   out = (lighting + mask·dot·I/π)·albedo, no shadow_mask, no specular lobe, no env, no wetness.
    // The geometric normal is the screen-derivative normal oriented toward the camera (the engine's
    // calc_normal_from_position(fragment_to_camera_world)); for a flat card it is ±N exactly.
    let rmfl_bsp = rmfl && i.lm.w < 1.5 && i.si_ctl.w > 0.5 && !gi_on();
    if (rmfl_bsp) {
        let to_cam = normalize(cam.cam_pos.xyz - i.world_pos);
        var geo_n = cross(dpx, dpy);
        geo_n = select(vec3<f32>(0.0, 0.0, 1.0), normalize(geo_n), dot(geo_n, geo_n) > 1e-12);
        geo_n = select(-geo_n, geo_n, dot(geo_n, to_cam) >= 0.0);
        let blend = clamp(dot(gn, geo_n) * 0.5 + 0.5, 0.0, 1.0);
        let ftype = i32(round(i.si_ctl.z));
        let at_digit = i32(round(i.si_ctl.w)) - 1;
        let a_base = textureSampleGrad(base_tex, base_samp, auv, dux, duy).a;
        let a_det = textureSampleGrad(detail_tex, base_samp, i.uv * i.detail_xform.xy + i.detail_xform.zw + i.scroll * cam.time.x, dux, duy).a;
        let a_at = textureSampleGrad(at_tex, base_samp, auv, dux, duy).a;
        // from_albedo_alpha (1): albedo.a = base.a × detail.a (× albedo_color.a, authored 1); from_texture (2): alpha_test_map.a
        let a_cov = select(a_base * select(1.0, a_det, at_digit == 1), a_at, at_digit == 2);
        var t = max(i.si_ctl.y, 0.0);
        if (ftype == 2) { t = clamp((1.0 - a_cov) * 2.0 * i.si_ctl.y, 0.0, 1.0); }
        // Sentinel present (per-vertex tier): front/back = the CPU dual-vMF pair. Flat probe/airprobe
        // tiers (and an atlas-claimed cluster mesh) carry no back-face term: back = front, so the
        // flat type doubles the isotropic ambient (≈ dual_vmf(N) + dual_vmf(-N)) and the blended
        // types reduce to the current one-sided bake.
        let has_back = i.tangent.w < BSP_FOLIAGE_TAN_SENTINEL + 0.5;
        let front = select(bake.rgb, i.tint.rgb, has_back);
        let back = select(front, i.tangent.xyz, has_back);
        let ndl_f = max(dot(gn, key), 0.0);
        let ndl_b = max(-dot(gn, key), 0.0);
        if (ftype == 0) {
            bake = vec4<f32>(front + back, bake.a);
            ndl = ndl_f + ndl_b;
        } else {
            bake = vec4<f32>(mix(back, front, blend) + t * mix(front, back, blend), bake.a);
            ndl = mix(ndl_b, ndl_f, blend) + t * mix(ndl_f, ndl_b, blend);
        }
    }
    let hemi = 0.5 + 0.5 * n.z;
    // Real-time cast shadow (objects onto world). The baked term stays UNGATED; the analytical
    // sun and glint are gated so dynamic casters darken what they cover. The engine foliage
    // shader samples no shadow_mask; an rmfl OBJECT part still takes this gate (any rewrite of
    // `shadow` perturbs the compiled shader; the only visible effect is another dynamic caster's
    // shadow on a leaf card, and foliage objects author no-shadow themselves).
    let shadow = shadow_strength(i.world_pos);
    // Cast shadow darkens the whole composite; the cf envelope kills shadowing on
    // back/underside faces.
    let cf = clamp(0.65 + dot(n, key) * 5.0, 0.0, 1.0);
    // Engine squared shadow curve (shadow_apply.hlsl:237): the darken is SQUARED (crisper
    // falloff). The 0.55 floor is the fill-spare stand-in (the engine spares the fill lobe).
    // rmfl BSP foliage reads no shadow_mask — neither its baked term nor its analytical sun is
    // darkened by a dynamic caster (foliage.hlsl_include has no shadow_apply pass).
    let darken = select(0.55 + 0.45 * pow(1.0 - cf * (1.0 - shadow), 2.0), 1.0, rmfl_bsp);
    // ENGINE SUN (entry_points.hlsl:429): the analytical sun is the authored HDR colour
    // `light_intensity` ÷ π, applied at FULL magnitude and gated by the BAKED sun VISIBILITY
    // (vmf[0].w, entry_points.hlsl_include:431). Enclosed interior verts decode baked vis→0, so
    // no sun is added on top of their dark baked lightmap; pvl=false/airprobe, water and glass
    // verts carry alpha=1.0. The engine's foliage shader adds NO back-face sun through leaves —
    // Reach's `translucency` "ONLY AFFECTS DYNAMIC LIGHTS" (reach_tag_test.exe).
    let baked_vis = clamp(bake.a, 0.0, 1.0);
    // The engine's DIFFUSE analytical sun carries the post-shadow sun visibility TWICE:
    // `diffuse += analytical_mask · saturate(L·N) · intensity/π · vmf[0].w`, where
    // `analytical_mask` itself = `vmf[0].a·(1−vmf[2].a+cloud·vmf[2].a)` and `vmf[0].a == vmf[0].w`
    // (same float4 lane) ⇒ w is squared in DIFFUSE, applied ONCE in specular
    // [entry_points.hlsl_include:428-431; analytical_mask.hlsl_include:56-64]. The cloud/gel
    // factor is a no-op when vmf[2].a=0 (needs a projected gel), so it is omitted. rmfl BSP
    // foliage applies the baked visibility ONCE (front.w = vmf[0].w·sat(L·N) in the VS, then
    // get_analytical_mask_from_projected_texture_coordinate = that × cloud(1)) and no shadow_mask.
    let sun_gate = select(shadow * baked_vis * baked_vis, baked_vis, rmfl_bsp);
    let sun_term = sun_col * (0.3183 * ndl * sun_gate);
    let lit = 0.45 + 0.25 * hemi;
    // Base map, sampled with the plain screen-space derivatives like the engine (textureSampleGrad
    // so the sample is valid in any control flow). Halo bitmaps are sRGB-encoded; the render
    // target is linear HDR, so the albedo is linearised (pow 2.2) below.
    let tex = textureSampleGrad(base_tex, base_samp, auv, dux, duy);
    // Detail map (fine-tiled) × 4.59479 supplies rock/cliff surface texture over the
    // deliberately-bland base; the mid-gray default = neutral (no detail bound). Detail-map UV:
    // the authored `detail_map` xform (tile.xy + offset.zw) when the material supplied one; else
    // the 6.0 fallback. i.uv (not auv) so the detail tile is independent of the base-map tile,
    // matching the engine's separate xform; the authored layers scroll with the base (waterfalls).
    // A COARSE (low-tile) detail is a MACRO pattern — floor TILES/pavers (is_macro_detail).
    // The engine has no detail distance fade (docs/hrek_re/02_shader_opaque_detail.md s2b: "relies
    // on mip/LOD of the detail bitmap"); the colour shift with distance IS the engine look (mips
    // average to the detail mean).
    let has_dxf = i.detail_xform.x > 0.01 && i.detail_xform.y > 0.01;
    let detail_tile_mag = max(i.detail_xform.x, i.detail_xform.y);
    let is_macro_detail = has_dxf && detail_tile_mag < 2.0;
    let detail_uv = select(auv * MESH_DETAIL_SCALE, i.uv * i.detail_xform.xy + i.detail_xform.zw + i.scroll * cam.time.x, has_dxf);
    let ddux = dpdx(detail_uv);
    let dduy = dpdy(detail_uv);
    // the sky_halo backdrop flag (fine_xform.w>0.5): its baked term is lifted below
    let no_dist_fade = i.fine_xform.w > 0.5;
    let detail_s4 = textureSampleGrad(detail_tex, base_samp, detail_uv, ddux, dduy);
    let detail = pow(detail_s4.rgb, vec3<f32>(2.2)) * MESH_DETAIL_MULT;
    // The engine spec mask for specular_mask_from_diffuse is albedo.w = base.a x detail.a (x
    // albedo_color.a, authored 1 on every vehicle paint; not carried). Object material lane only
    // (flagged below by obj_mat_lane); the fallback 1x1 grey detail has a = 1 so unauthored
    // details are a no-op.
    let detail_a = select(1.0, detail_s4.a, has_dxf);
    // LIVE fine self-detail (concrete grain): the engine keeps the high-frequency grain as a
    // runtime layer. Sample the BASE texture (self-detail: detail_tag==base_tag) at its high
    // tiling and multiply in, mean-preserving (×MESH_DETAIL_MULT). Gradients computed in
    // uniform control flow; `select` gates the result (flag = fine_xform.z).
    let fuv = i.uv * i.fine_xform.xy + i.scroll * cam.time.x;
    let fgx = dpdx(fuv);
    let fgy = dpdy(fuv);
    let fine_s = pow(textureSampleGrad(base_tex, base_samp, fuv, fgx, fgy).rgb, vec3<f32>(2.2)) * MESH_DETAIL_MULT;
    let fine = select(vec3<f32>(1.0), fine_s, i.fine_xform.z > 0.5);
    // Surface-contrast gain for macro-tile floors: the combined detail deviation around its mean
    // (1.0), mean-preserving. Only the low-frequency tile/grout layer (detail_tex) — it survives
    // minification; the fine grain minifies to a sub-neutral flat at range, so boosting it would
    // only darken.
    var detail_c = detail;
    if (is_macro_detail) {
        detail_c = vec3<f32>(1.0) + (detail - vec3<f32>(1.0)) * TILE_SURFACE_CONTRAST;
    }
    var albedo = pow(tex.rgb, vec3<f32>(2.2)) * detail_c * fine;
    // two/four_change_color albedo (Forge pallets): mattex.w == 6 → at_tex holds the change_color_map
    // (r = primary mask, g = secondary), the instance colour alpha carries the change-colour index sentinel.
    // Engine: albedo *= (1 − m.r + m.r·primary)·(1 − m.g + m.g·secondary); Forge sets both from the object colour.
    // With NO forge team/object colour (tint.a >= 0) the engine still applies the object's
    // AUTHORED default change colours (obje change-colors block) through the same masks: primary rides
    // xbump.rgb, secondary si_ctl.yzw. Vehicles (Mongoose, Shade) author these; Forge pieces author
    // none (white) so their albedo is unchanged.
    if (i.mattex.w > 5.5 && i.mattex.w < 6.5) {
        let m = textureSample(at_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw);
        let forge = i.tint.a < -1.5;
        let cc_i = i32(round(select(0.0, -i.tint.a - 2.0, forge)));
        // Index 8 = a Forge placement on the NEUTRAL team (or no team). The engine overwrites
        // every variant-placed object's primary change colour: team colour for teams 0..7, the
        // constant (0.5, 0.5, 0.5) for neutral/none (reach_tag_test.exe 0x14070C890) — never the
        // authored default permutation. Secondary stays authored (only the primary is overwritten).
        let neutral = cc_i == 8;
        let cc = select(change_color_palette(cc_i), vec3<f32>(0.5, 0.5, 0.5), neutral);
        let cp = select(i.xbump.rgb, cc, forge);
        let cs = select(i.si_ctl.yzw, cc, forge && !neutral);
        albedo = albedo * (vec3<f32>(1.0 - m.r) + m.r * cp) * (vec3<f32>(1.0 - m.g) + m.g * cs);
    }
    // The engine SATURATES the composed albedo — HREK entry_points.hlsl_include:100
    // `albedo = saturate(albedo)` after calc_albedo_ps, and the shipped DX11 pixel shaders compile
    // it as `mul_sat r, r, l(4.594790…)` (1733 of the 2099 captured Sapien PS that carry
    // DETAIL_MULTIPLIER; the rest write the albedo G-buffer, an 8-bit target = the same clamp). So
    // base × detail × 4.59479 × albedo_color can never exceed 1.0 (Ivory's zen walls bind their OWN
    // white diffuse as detail_map: base² × 4.59 = 4.3 unclamped would light them 2.7-4x brighter
    // than the engine frame).
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));

    // Analytical specular (engine Cook-Torrance, cook_torrance_core.hlsl_include).
    let view = normalize(cam.cam_pos.xyz - i.world_pos);
    let h = normalize(view + key);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, view), 0.0);
    let lum = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
    // Roughness: the material's authored constant when it has one (Reach opaque spec is
    // constant-driven, not a gloss bitmap); else an inverse-luminance stand-in — spec-masked
    // surfaces (Bc3 metal/tile/glass) get a glossier curve so their highlight reads, matte
    // surfaces (Bc1, spec_mask=0 below) a high, broad roughness (their spec is zeroed by the
    // mask anyway).
    let rough_legacy = select(
        clamp(0.88 - lum * 0.3, 0.55, 0.95),          // matte: broad, effectively flat
        clamp(0.70 - lum * 0.55, 0.10, 0.90),         // spec-masked: glossier, visible highlight
        i.spec_from_alpha > 0.5,
    );
    let rough_base = select(rough_legacy, clamp(i.mat_rough, 0.04, 1.0), i.mat_rough >= 0.0);
    // Per-texel material_texture roughness/spec override (cook_torrance_core.hlsl_include
    // :112-118). power_modifier = material_texture.a, sampled at the authored material_texture_xform
    // (base auv when unauthored). Then engine-exact:
    //   roughness             = lerp(material_texture_black_roughness, roughness, power_modifier)
    //   specular_coefficient *= lerp(material_texture_black_specular_multiplier, 1, power_modifier)
    // Sampled UNCONDITIONALLY (WGSL uniformity: textureSample stays in uniform control flow); the
    // has_material_texture flag (mattex.z) then selects between the sample and 1.0 (no-op). Materials
    // without a material_texture bind the 1×1 grey default (a=1) AND pass flag=0 → power_modifier=1
    // → both lerps are identities.
    let has_mattex = i.mattex.z > 0.5;
    let mt_auth = (abs(i.mattex_xform.x) + abs(i.mattex_xform.y)) > 1e-6;
    let mtuv = select(auv, i.uv * i.mattex_xform.xy + i.mattex_xform.zw, mt_auth);
    let mt_a = textureSample(mat_tex, base_samp, mtuv).a;
    let power_modifier = select(1.0, mt_a, has_mattex);
    let rough = mix(i.mattex.x, rough_base, power_modifier);          // lerp(black_roughness, rough, pm)
    let eff_spec_mult = mix(i.mattex.y, 1.0, power_modifier);         // lerp(black_specmult, 1, pm)
    // Engine Beckmann distribution (cook_torrance_core.hlsl_include:
    // `D = exp(-sqr_tan_alpha/SQR(m)) / (SQR(m)·SQR(SQR(fNDotH)) + 0.00001)`). NO π in the
    // denominator — the single π lives in the radiance normalization below. m = roughness.
    let m = clamp(rough, 0.04, 1.0);
    let m2 = m * m;
    let ndh2 = max(ndh * ndh, 1e-4);
    let tan2 = (1.0 - ndh2) / ndh2;
    let ndf = exp(-tan2 / m2) / (m2 * ndh2 * ndh2 + 1e-5);   // engine-exact Beckmann (no π)
    // ARTIST FRESNEL (the engine's opaque Fresnel):
    //   F0 = lerp(specular_tint, albedo, albedo_blend)      ; metalness lerps F0 toward the albedo
    //   F  = lerp(F0, fresnel_color, (1-NdV)^steepness)      ; artist edge tint + curve
    // spec2 = [metalness, steepness, specular_tint_luma, fresnel_color_luma]; all-zero → a
    // dielectric Schlick (metal=0 → F0=0.04, edge=white, steep=5).
    let metal = clamp(i.mat_spec2.x, 0.0, 1.0);
    let steep = select(5.0, i.mat_spec2.y, i.mat_spec2.y > 0.0);
    let f0_base = select(0.04, i.mat_spec2.z, i.mat_spec2.z > 0.0);
    let fres_edge = select(1.0, i.mat_spec2.w, i.mat_spec2.w > 0.0);
    // authored specular_tint / fresnel_color COLOURS when present (engine lerps normal_specular_tint ->
    // glancing tint by the Fresnel blend); the scalar luminance lanes otherwise.
    let f0_col = select(vec3<f32>(f0_base), i.spec_rgb.rgb, i.spec_rgb.w > 0.5);
    let edge_col = select(vec3<f32>(fres_edge), i.fres_rgb.rgb, i.fres_rgb.w > 0.5);
    let F0 = mix(f0_col, albedo, metal);
    // The engine's artist Fresnel curve is on N·V (a rim/edge tint), NOT the microfacet V·H:
    // `F = lerp(F0, fresnel_color, pow(1-N·V, steepness))`.
    let Fv = mix(F0, edge_col, pow(1.0 - max(ndv, 0.0), steep));  // artist Fresnel (vec3, on N·V)
    // Engine Cook-Torrance geometry: G = min(1, min(2·NdH·NdV/VdH, 2·NdH·NdL/VdH)).
    let vdh = max(dot(view, h), 1e-4);
    let geom = min(1.0, min(2.0 * ndh * ndv / vdh, 2.0 * ndh * ndl / vdh));
    // Specular mask. Material-constant spec (Reach opaque = specular_coefficient ×
    // analytical_specular_contribution = mat_spec) applies even on BC1 (alpha-less) diffuse; the
    // base_map alpha is a per-texel multiplier ONLY when it's a real gloss mask (BC3,
    // spec_from_alpha). Materials that authored no spec constant (mat_spec==0) take the diffuse
    // alpha alone: Bc1/opaque diffuse have no alpha → sampled a=1.0 would make matte surfaces
    // (sandy ground) reflect the full blue sky as per-pixel sparkle, so no-alpha diffuse reads as
    // matte (spec_mask=0). The ENGINE OBJECT MATERIAL lane (unit/item objects: fres_rgb.w == 2
    // from scene.rs) uses the engine's albedo.w = base.a × detail.a mask.
    let obj_mat_lane = i.lm.w > 1.5 && obj_probe_lit && i.fres_rgb.w > 1.5;
    let mask_det = select(1.0, detail_a, obj_mat_lane);
    let spec_mask = select(
        tex.a * i.spec_from_alpha,
        i.mat_spec * select(1.0, tex.a * mask_det, i.spec_from_alpha > 0.5),
        i.mat_spec > 0.0,
    );
    let baked_lum = dot(lit * bake.rgb, vec3<f32>(0.299, 0.587, 0.114));
    // Environment-reflection gate: the baked luminance alone (no floor). A dark interior texel
    // (Zealot ~0.006) reflects ~nothing, a sunlit outdoor one saturates to 1. (On a SPACE map the
    // sky cube is a starfield + planet; a floor would paint a grey-blue reflection over Zealot's
    // saturated purple walls — measured 17% of its radiance.)
    let sky_vis = clamp(1.3 * baked_lum, 0.0, 1.0);
    // Engine specular radiance normalization (cook_torrance_core.hlsl_include:
    // `analytic_specular_radiance = max(0, D·G·F / fVDotN / 3.141592658 · light_irradiance)`).
    // Scalar microfacet weight = D·G / (N·V·π) — NO 1/4 factor, NO N·L divide, NO extra N·L
    // multiply; the vec3 Fresnel Fv carries F. Clamped to SPEC_MAX_SCALAR (blowout guard).
    let spec_weight = min((ndf * geom) / max(ndv * SPEC_PI, 1e-5), SPEC_MAX_SCALAR);
    // strength = specular_coefficient·analytical_specular_contribution (= mat_spec, folded into
    // spec_mask) × the per-texel material_texture multiplier × the baked sun visibility (the
    // SAME gate as the diffuse sun_term: a heuristic sky_vis gate with a floor lit the sun's
    // SPECULAR lane in sealed interiors — measured on Zealot, real baked sun visibility 0.011,
    // ~36% of frame radiance) × the cast shadow. Specular takes the light's full HDR
    // magnitude/hue (NO ÷π, unlike diffuse; entry_points.hlsl_include:455). The analytical
    // light of a probe-lit object is its merged key light (sun_col = b*sun + (1-b)*lobes,
    // entry_points.hlsl_include `light_intensity*analytical_mask` -> calc_material), NOT the scene sun.
    let spec_light = select(light.sun_tint.rgb, sun_col, obj_mat_lane);
    let spec = Fv * (spec_weight * SPEC_CAL * spec_mask * eff_spec_mult * baked_vis * shadow) * spec_light;

    // Self-illumination — SIMPLE / opaque mode: a SINGLE flat `self_illum_map · color ·
    // intensity` sample [self_illumination.hlsl_include:27] (the view-parallax walk belongs
    // ONLY to the halogram blend_box/multilayer path, fs_holo). The authored self_illum_color ×
    // intensity is already baked into the (linear, HDR) texel. ADDED after the lit term,
    // INDEPENDENT of it (entry_points.hlsl_include: out = max(0, diffuse*albedo + spec +
    // self_illum)); NOT re-modulated by i.tint (a glowing panel in a shadowed lightmap region
    // must not go dark). Non-emissive surfaces sample the 1×1 black emis_tex → 0.
    var emis = textureSample(emis_tex, base_samp, auv).rgb;
    // change_color: an opaque/blend forge object part flagged as change_color carries its
    // colour index in the instance-colour ALPHA as a negative sentinel (-(idx+2)). Tint the
    // emissive (self-illum accent strips) by the team/object colour — NOT the body albedo.
    // Normal geometry has alpha >= 0 → this never triggers.
    if (i.tint.a < -1.5) {
        let cc_idx = i32(round(-i.tint.a - 2.0));
        emis = emis * change_color_palette(cc_idx);
    }
    // Scenario point/spot SimpleLights: forward-accumulated, added to the lit
    // term (× albedo, like the engine `lit += diffuse.rgb * simple_lights`).
    let simple = albedo * calc_simple_lights(i.world_pos, n);
    // Environment reflection: the engine's `reflection × env_specular_contribution × spec_mask
    // × Fresnel-tint` (the AUTHORED per-material contribution rides bump_detail_xform.w; the
    // artist Fresnel `Fv` carries F0=mix(f0,albedo,metal), so a metal reflects its own colour),
    // gated by the baked sky-visibility and the cast shadow. Added, not lerped.
    let refl = reflect(-view, n);
    // Distance LOD bias blurs the cubemap sample so glossy snow/rock stops sparkling (the
    // reflected sky varies fast across a bumpy normal and textureSampleLevel ignores
    // minification). Ramp from ~15wu to a flat-sky sample by ~115wu.
    let env_lod = clamp((i.vdepth - 15.0) / 20.0, 0.0, 5.0);
    // DYNAMIC OBJECTS (forge blocks etc., lm.w==2) do NOT mirror the sky — forerunner forge
    // material is matte; any genuine reflection on a forge object comes from its GLASS
    // submeshes (fs_blend) or the object material lane below. Sun GLINT (the `spec` term) is
    // unaffected, so object metal still highlights.
    let obj_matte = select(1.0, 0.0, i.lm.w > 1.5);
    let menv = max(i.bump_detail_xform.w, 0.0);
    // two_lobe_phong.hlsl_include:164: the engine's env sv_refl includes ×specular_coefficient
    // (env_ctl.w, the RAW authored value — NOT mat_spec, which folds in
    // analytical_specular_contribution). Default 0 → factor 1.0 (no-op).
    let scoef = select(1.0, i.env_ctl.w, i.env_ctl.w > 0.0);
    // The engine has NO (1-rough) linear darkening on env radiance — roughness only selects the
    // reflection mip (energy is preserved). env_ctl.rgb carries the authored env_tint_color
    // (applied inside env_reflection).
    var env = env_reflection(refl, rough, env_lod, i.env_ctl.rgb, i.xctl.z) * Fv * (spec_mask * sky_vis * shadow * menv * obj_matte * scoef);
    // ENGINE env reflection + AREA specular for the object material lane (cook_torrance_core
    // .hlsl_include:227-246 + entry_points:374/393-394 + environment_mapping dynamic). Per part:
    //   sh_glossy   = final_tint * (diffuse_power_specular(dot(dom_dir,R), bandwidth, roughness)*3*dom + 0.25*fill)
    //   seed        = raised(N.L) * key_light * b * 0.25 * final_tint          (the prebaked analytical tint)
    //   area_env    = max(sh_glossy + seed, 0.001) * shadow_mask
    //   env         = cube(R, lod) * env_tint * (contribution * mask * specular_coefficient) * area_env
    //   spec_area   = mask * specular_coefficient * max(sh_glossy, 0) * area_specular_contribution
    // mask = the diffuse alpha (specular_mask_from_diffuse parts keep it at decode), lanes: bump_detail_xform.w =
    // contribution, env_ctl = [env_tint, specular_coefficient], fine_xform.x = area contribution, spec_rgb.w - 1 =
    // `roughness` (LUT z), xctl.z = cube bound (cluster cube for env DYNAMIC, authored cube for per_pixel),
    // xctl.w >> 4 = cube lod x 4. The Covenant purple / UNSC satin paint is this glossy env + area lobe
    // (`obj_matte` zeroes the plain env term for every dynamic object).
    var spec_area = vec3<f32>(0.0);
    if (obj_mat_lane) {
        let dom_dir_o = normalize(i.obj_probe0.xyz);
        let bw_o = clamp(i.obj_probe0.w, 0.0, 1.0);
        let dom_rgb_o = max(i.obj_probe1.rgb, vec3<f32>(0.0));
        let rough_lut_o = clamp(i.spec_rgb.w - 1.0, 0.0, 1.0);
        let rdotd_o = dot(dom_dir_o, refl);
        let lut3_o = textureSampleLevel(dps_lut, vmf_samp, vec3<f32>(rdotd_o * 0.5 + 0.5, bw_o, rough_lut_o), 0.0).r * 3.0;
        let sh_glossy = Fv * (lut3_o * dom_rgb_o + 0.25 * max(obj_fill, vec3<f32>(0.0)));
        let raised_o = saturate(dot(n, key) * 0.45 + 0.55);
        let seed_o = raised_o * sun_col * obj_b * 0.25 * Fv;
        let area_env = max(sh_glossy + seed_o, vec3<f32>(0.001)) * shadow;
        let omask = select(1.0, tex.a * detail_a, i.spec_from_alpha > 0.5) * eff_spec_mult;
        let lod_o = f32(u32(i.xctl.w + 0.5) >> 4u) * 0.25;
        let cube_o = env_reflection(refl, rough, lod_o, i.env_ctl.rgb, i.xctl.z);
        env = select(vec3<f32>(0.0), cube_o * (menv * omask * scoef) * area_env, i.xctl.z > 0.5 && menv > 0.0);
        spec_area = omask * scoef * max(sh_glossy, vec3<f32>(0.0)) * max(i.fine_xform.x, 0.0);
    }
    // Baked term. shadow_mask.hlsl_include:38: the engine's sun shadow darkens ONLY the dominant vMF
    // lobe (vmf[1] *= mask.r) and SPARES the 0.25·fill lobe (vmf[3], its multiply is commented
    // out). Spare the fill by interpolating the effective darken toward 1.0 by the fill's linear
    // fraction of the baked total (frac_fill): fully-fill channels see darken→1 (spared), zero-fill
    // channels see the full `darken`. The PVL path (bake_fill==0 → frac_fill==0) gets the full
    // darken (the engine PVL is a single per-vertex tint with no separable fill lobe).
    let frac_fill = clamp(bake_fill / max(bake.rgb, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0));
    let eff_darken = darken + (1.0 - darken) * frac_fill;
    var baked_lit = bake.rgb * eff_darken * light.dbg.z; // dbg.z = Lighting Lab lightmap multiplier
    // The sky_halo backdrop (the sky flag fine_xform.w) is PVL-lit and its baked tint is
    // Reinhard-compressed to ~unit, so shadowed faces read near-black. Lift its baked term
    // modestly toward the engine's brighter distant-terrain look (×1.5, +0.06 floor, soft so snow
    // peaks don't blow out).
    if (no_dist_fade) {
        baked_lit = baked_lit * 1.5 + vec3<f32>(0.06);
    }
    // The engine composite is albedo·(baked_probe + analytical_sun) + spec + self_illum with NO
    // flat/hemisphere/ground/fog ambient pedestal (entry_points.hlsl_include:401-431 — the baked
    // dual-VMF probe IS the ambient), on mesh_shade (foliage/trees/props/objects) and terrain alike.
    // The one runtime fill is the engine's global directional BOUNCE light for probe/airprobe/SH-lit
    // OBJECTS (entry_points.hlsl_include:436-438; the add is gated `#ifndef must_be_environment`,
    // so it is compiled OUT for lightmapped BSP/terrain, whose bounce is baked into the lightmap):
    // `saturate(dot(bounce_dir,N)) · bounce_intensity/π`, bounce_dir = the key light, colour = the
    // map's HDR sun tint, magnitude = BOUNCE_FILL. Added INSIDE the `·albedo` group, exactly as the
    // engine adds it to diffuse_radiance before the ×albedo multiply.
    var bounce = vec3<f32>(0.0);
    if (i.lm.w > 1.5) {
        bounce = light.sun_tint.rgb * (BOUNCE_FILL * max(dot(key, n), 0.0));
    }
    // entry_points.hlsl_include:473: the engine scales self_illum_radiance by ILLUM_SCALE =
    // g_alt_exposure.r at add-time (illum_scale_now, per frame from the exposure meter);
    // light.sun_tint.w is a 1.0 hook on top.
    let illum_scale = light.sun_tint.w * illum_scale_now();
    // Per-shader material_model dispatch: the engine links exactly one calc_material_<m>_ps per
    // shader variant. Models 0 (diffuse_only), 3 (foliage) and 4 (none) compute NO analytical/area
    // specular lobe, so the Cook-Torrance `spec` term is zeroed there; diffuse_only (0)
    // additionally carries no environment reflection. Every other model (1 cook_torrance,
    // 2 two_lobe_phong, 5-9) and the unset sentinel (matmodel==0 → mm<0) keep the Cook-Torrance
    // path. TODO: two_lobe_phong keeps the Cook-Torrance lobe rather than a Phong lobe.
    let mm = i32(round(i.matmodel)) - 1;
    // An rmfl object part (si_ctl sentinel) is the foliage material model too — no Cook-Torrance
    // lobe; its base alpha is alpha-test COVERAGE, which spec_from_alpha=1 would read as a full
    // gloss mask (a white glint film over every leaf card). (The foliage `specular` type's own
    // pow(2a−1,power)·colour·intensity term is ≤0.004 on the shipped materials; not ported.)
    let mm_kill_spec = (mm == 0) || (mm == 3) || (mm == 4) || rmfl;
    let mm_kill_env = (mm == 0) || rmfl_bsp; // the foliage shader has no environment term
    let spec_d = select(spec, vec3<f32>(0.0), mm_kill_spec);
    let env_d = select(env, vec3<f32>(0.0), mm_kill_env);
    let spec_area_d = select(spec_area, vec3<f32>(0.0), mm_kill_spec);
    // self_illumination MODE routing: the engine links exactly one calc_self_illumination_<m>_ps
    // per shader. Only from_albedo (mode 4) diverges from the single-composite emissive path: it
    // emits `albedo · self_illum_color · intensity` as the self-illum AND ZEROS albedo so the lit
    // `diffuse·albedo` term contributes nothing (self_illumination.hlsl_include:115-116). No
    // self_illum_map is baked for from_albedo, so `emis` is 0 here; synthesize the emission from
    // the sampled albedo × cmul (=color×intensity, carried in si_ctl.yzw) and gate the lit-albedo
    // terms off. Every other mode (0/1/2/3/5..) keeps the composite path.
    let si_from_albedo = i32(round(i.si_ctl.x)) == 4;
    if (si_from_albedo) {
        emis = albedo * i.si_ctl.yzw;
    }
    let lit_albedo = select(albedo, vec3<f32>(0.0), si_from_albedo);
    let simple_d = select(simple, vec3<f32>(0.0), si_from_albedo);
    var shaded = lit_albedo * (baked_lit + sun_term + bounce) + simple_d + emis * illum_scale + env_d + spec_d + spec_area_d;
    // HMS_MDBG per-term HDR probes (mesh path; read via HMS_HDR_DUMP, pre-fog): 1 bake, 2 baked_lit, 3 sun_term,
    // 4 albedo, 5 tint(fill lane), 6 obj_probe1 (dom rgb), 7 [probe0.w, probe1.w, ol_flag], 8 [shadow, darken, baked_vis]
    let mdbg = __MDBG__;
    if (mdbg > 0.5) {
        if (mdbg < 1.5) { return vec4<f32>(bake.rgb, 1.0); }
        if (mdbg < 2.5) { return vec4<f32>(baked_lit, 1.0); }
        if (mdbg < 3.5) { return vec4<f32>(sun_term, 1.0); }
        if (mdbg < 4.5) { return vec4<f32>(albedo, 1.0); }
        if (mdbg < 5.5) { return vec4<f32>(i.tint.rgb, 1.0); }
        if (mdbg < 6.5) { return vec4<f32>(i.obj_probe1.rgb, 1.0); }
        if (mdbg < 7.5) { return vec4<f32>(i.obj_probe0.w, i.obj_probe1.w, ol_flag, 1.0); }
        if (mdbg < 8.5) { return vec4<f32>(shadow, darken, baked_vis, 1.0); }
        // 12: the SELF-ILLUM term as composited (emis * illum_scale); 13: raw emis before the scale.
        // Zealot's interior "bright spots" are emissive light fixtures, and nothing else probed the
        // emissive lane -- HMS_TDDBG=4 only covers the mesh two_detail path, not the BSP.
        if (mdbg > 11.5 && mdbg < 12.5) { return vec4<f32>(emis * illum_scale, 1.0); }
        if (mdbg > 12.5 && mdbg < 13.5) { return vec4<f32>(emis, 1.0); }
        // 16: analytical-sun SPECULAR; 17: environment reflection; 18: the sky_vis gate both use.
        if (mdbg > 15.5 && mdbg < 16.5) { return vec4<f32>(spec_d, 1.0); }
        if (mdbg > 16.5 && mdbg < 17.5) { return vec4<f32>(env_d, 1.0); }
        if (mdbg > 17.5 && mdbg < 18.5) { return vec4<f32>(vec3<f32>(sky_vis), 1.0); }
        // 22: object AREA specular (probe glossy lobe x area_specular_contribution)
        if (mdbg > 21.5 && mdbg < 22.5) { return vec4<f32>(spec_area_d, 1.0); }
        // 23: [spec mask (tex.a x mattex mult), raw tex.a, lum(Fv)]; 24: Fv rgb; 25: raw cube sample rgb
        if (mdbg > 22.5 && mdbg < 23.5) { return vec4<f32>(select(1.0, tex.a * mask_det, i.spec_from_alpha > 0.5) * eff_spec_mult, tex.a, dot(Fv, vec3<f32>(0.2126, 0.7152, 0.0722)), 1.0); }
        if (mdbg > 23.5 && mdbg < 24.5) { return vec4<f32>(Fv, 1.0); }
        if (mdbg > 24.5 && mdbg < 25.5) { return vec4<f32>(env_reflection(refl, rough, 0.0, vec3<f32>(1.0), i.xctl.z), 1.0); }
        // 19: [baked_vis, lm.x (>0.5 = lightmap ATLAS, else per-vertex), 1] — both together, so a
        // sweep can compare the ATLAS visibility population against the per-vertex one in ONE pass.
        if (mdbg > 18.5 && mdbg < 19.5) { return vec4<f32>(baked_vis, i.lm.x, 1.0, 1.0); }
        // 20: [baked_vis, lum(bake.rgb) scaled, lm.x] — the DARK-BAKE-BUT-VISIBLE contradiction test.
        // A texel the lightmapper baked DARK but flagged as sun-VISIBLE is the fake-sun signature.
        // 21: raw instance colour (HMS_MATTINT material hash)
        if (mdbg > 20.5 && mdbg < 21.5) { return vec4<f32>(i.itint.rgb, 1.0); }
        if (mdbg > 19.5 && mdbg < 20.5) {
            let bl = dot(bake.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            return vec4<f32>(baked_vis, clamp(bl * select(10.0, 1.0, light.dbg.x > 1.5), 0.0, 1.0), i.lm.x, 1.0);
        }
        // 14: [illum_scale as composited, illum_scale_now(), sun_tint.w]
        if (mdbg > 13.5 && mdbg < 14.5) { return vec4<f32>(illum_scale, illum_scale_now(), light.sun_tint.w, 1.0); }
        // 15: [resolve gain as the mesh shader sees it, mean_log from lum_meter, expo.y key]
        if (mdbg > 14.5 && mdbg < 15.5) {
            let ml = textureLoad(lum_meter, vec2<i32>(0, 0), 0).r;
            let hi2 = light.expo.w; let lo2 = min(light.expo.z, hi2);
            let rg = light.expo.y / max(exp2(ml), 1e-4);
            return vec4<f32>(light.expo.x * clamp(rg * light.expo2.x, lo2, hi2), ml, light.expo.y, 1.0);
        }
        // 9: the shipped atlas' dominant light DIRECTION (xyz*0.5+0.5) for this texel; 10: DM.g sun visibility
        let a0 = textureSampleLevel(lm_sdm_tex, lm_samp, i.uv2, 0.0).a;
        let a1 = textureSampleLevel(lm_sdm1_tex, lm_samp, i.uv2, 0.0).a;
        let a2 = textureSampleLevel(lm_sdm2_tex, lm_samp, i.uv2, 0.0).a;
        let dm = textureSampleLevel(lm_dm_tex, lm_samp, i.uv2, 0.0);
        let draw = vec3<f32>(a0, a1, a2) * 2.0 - 1.0;
        if (mdbg < 9.5) { return vec4<f32>(normalize(draw) * 0.5 + 0.5, length(draw)); }
        // 11: WHICH lighting path fed this pixel — [lm.x (>0.5 = atlas, else per-vertex/PVL),
        // lm.w (2 = dynamic object), raw i.tint.a (vertex-colour alpha before the sentinel)].
        // Needed because "baked_vis is wrong" means nothing until you know which path supplied it.
        if (mdbg > 10.5) { return vec4<f32>(i.lm.x, i.lm.w, i.tint.a, 1.0); }
        return vec4<f32>(dm.g, dm.r, length(draw), 1.0);
    }
    // No highlight compression in scene HDR space (neither the engine nor protomorph do);
    // highlights are handled downstream by the exposure + saturate in resolve().
    // Per-domain sceg grade (lit surfaces only), then the Reach atmospheric fog
    // (extinction × radiance + inscatter).
    shaded = apply_fog(scene_grade(shaded), i.world_pos, cam.cam_pos.xyz);
    // Opaque: diffuse alpha is a spec/gloss mask in Halo, not coverage.
    return vec4<f32>(shaded, 1.0);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return mesh_shade(i);
}

// Alpha-test (foliage cutout): the base map's alpha IS coverage here (decoded
// keep-alpha), so discard fully-transparent texels — leaves/branches read as
// see-through instead of solid cards.
@fragment
fn fs_alphatest(i: VOut) -> @location(0) vec4<f32> {
    // Sample the coverage mask at the SAME tiled UV mesh_shade uses (auv = uv*tile+scroll);
    // sampling untiled i.uv misaligned the discard from the visible texels (regression A2).
    let auv = i.uv * i.tile + i.scroll * cam.time.x;
    // Clip on the dedicated alpha_test_map.a (engine calc_alpha_test_ps samples a map DISTINCT
    // from the base, before albedo compose). at_tex is bound to the base view when the material
    // authors no distinct alpha_test_map. Sampled at the base tile (the alpha_test_map is the
    // base's coverage companion and shares its xform in Reach).
    let a_at = textureSample(at_tex, base_samp, auv).a;
    // An rmfl BSP material with alpha_test = from_albedo_alpha (si_ctl = [FOLIAGE_SI_MODE,
    // _, _, 2]) clips the engine's albedo.a = base.a × detail.a (calc_albedo_default_ps; Forge World's
    // pine binds its own branch map as detail_map, so the coverage is base.a² — clip at base.a ≥ 0.707).
    // Every other material keeps the plain at_tex.a clip (the detail sample is a no-op 1.0 there).
    let a_det = textureSample(detail_tex, base_samp, i.uv * i.detail_xform.xy + i.detail_xform.zw + i.scroll * cam.time.x).a;
    let rmfl_albedo_alpha = i32(round(i.si_ctl.x)) == FOLIAGE_SI_MODE && i.lm.w < 1.5 && i32(round(i.si_ctl.w)) == 2;
    let a0 = select(a_at, a_at * a_det, rmfl_albedo_alpha);
    // DISTANT-FOLIAGE FIX: a minified leaf texture's alpha mips toward the mid-range, so the
    // TRANSPARENT background between branches rises above a fixed 0.5 cutoff → distant trees
    // render as SOLID cards. Sharpen the alpha by its screen-space gradient (Golus) so the 0.5
    // crossing stays a ~1px edge at ANY distance — preserving the see-through silhouette without
    // touching near trees (their edges are already ~1px, so the rescale is a no-op there).
    let a = clamp((a0 - 0.5) / max(fwidth(a0), 1e-4) + 0.5, 0.0, 1.0);
    if (a < 0.5) { discard; }
    return mesh_shade(i);
}

// Alpha-blend (glass/translucent): the base map's alpha is opacity, lifted by the
// engine's opacity-FRESNEL so glass is see-through head-on but opaque/shiny at grazing
// angles. Engine calc_alpha_blend_opacity (shared/blend.hlsl_include):
//   fres_op = clamp(coef·pow(1-NoV, steepness) + bias, 0, 1)  (0 if NoV<=0)
//   out.a   = 1 - (1-albedo.a)·(1-fres_op)
// Uses the documented shader_glass default triplet (coef=1, steepness=3, bias=0); the
// composite only ever RAISES alpha above the base (fres_op>=0) so the center stays as
// transparent as before — a strict improvement. specular_scalar = 1+fres_op lifts the
// GGX highlight for the rim sheen. (Per-material opacity_fresnel_* not yet plumbed.)
//
// This is the shared shading body for the transparent-geometry pass. `fs_blend` returns it
// straight (non-premultiplied rgb) for the SrcAlpha/InvSrcAlpha and multiply/double_multiply
// pipelines; `fs_blend_premul` premultiplies rgb by alpha for the engine-exact
// pre_multiplied_alpha (One/INV_SRC_ALPHA) pipeline. For a coverage-scaled body the two
// composite identically (rgb·a + dst·(1−a)); the premul path only diverges when the shader
// emits un-coverage-scaled glow.
// A dynamic OBJECT's analytical ("key") light for the glass lanes = the engine lobes+sun MERGE
// (doc 16 s2, sub_140828CD0; the same math mesh_shade applies inline for opaque objects):
//   f0 = sum(col0)/(sum(col0)+sum(col1)), Ls = sum(sun)*mask0, b = min(mask0, Ls/(Ls+sum(col0)+sum(col1))),
//   dir = normalize(b*sun_dir + (1-b)*normalize(f0*dir0 + f1*dir1)), colour = b*sun + (1-b)*(f0*col0 + f1*col1)
// and c[3] = fill + 2*bounce (bounce-to-ambient). The shader-side constants are then c[0].w = 1 and
// c[2].w = b (the seed gate). With no packed obj_light lane: key = dominant lobe x mask, b = mask.
// Returns [dir.xyz, b] and writes colour / fill / bounce through the out params.
struct ObjKey { dir: vec3<f32>, col: vec3<f32>, b: f32, fill: vec3<f32>, bounce_dir: vec3<f32>, bounce_col: vec3<f32> }
fn obj_key_light(probe0: vec4<f32>, probe1: vec4<f32>, fill_lane: vec3<f32>, ol: vec4<u32>) -> ObjKey {
    var k: ObjKey;
    let dir0 = normalize(probe0.xyz);
    let mask0 = clamp(probe1.w, 0.0, 1.0);
    k.dir = dir0;
    k.col = probe1.rgb * mask0;
    k.b = mask0;
    k.fill = fill_lane;
    k.bounce_dir = vec3<f32>(0.0, 0.0, 1.0);
    k.bounce_col = vec3<f32>(0.0);
    let oll = obj_light_unpack(ol);
    if (oll.valid) {
        let dir1 = oll.fill_dir;
        k.bounce_dir = oll.bounce_dir;
        let col0 = probe1.rgb;
        let col1 = fill_lane;
        let l0 = col0.r + col0.g + col0.b;
        let l1 = col1.r + col1.g + col1.b;
        let tot = max(1e-5, l0 + l1);
        let f0 = l0 / tot;
        let f1 = 1.0 - f0;
        let sunc = light.sun_tint.rgb;
        let ls = (sunc.r + sunc.g + sunc.b) * mask0;
        var b = mask0;
        if (tot + ls > 1e-4) { b = min(b, ls / (tot + ls)); }
        let ld = f0 * dir0 + f1 * dir1;
        let lobe_dir = select(dir0, ld / max(length(ld), 1e-6), length(ld) > 1e-5);
        let ad = b * normalize(light.sun_dir.xyz) + (1.0 - b) * lobe_dir;
        k.dir = select(dir0, ad / max(length(ad), 1e-6), length(ad) > 1e-5);
        k.col = b * sunc + (1.0 - b) * (f0 * col0 + f1 * col1);
        k.b = b;
        // chmt bounce-to-ambient / bounce scale (glass lanes carry no world position for the
        // max-contrast distance; the min-luminance lift is applied with the merge gate only).
        k.fill = col1 + oll.bta * oll.bounce;
        k.bounce_col = oll.bs * oll.bounce;
        if (oll.min_lum > 1e-4) {
            let lum = max(dot(col0 + k.fill + b * sunc, vec3<f32>(0.3, 0.59, 0.11)) * 0.15915494, 1e-4);
            let s = max(oll.min_lum / (exposure_stops_gain() * lum), 1.0);
            k.fill = k.fill * s;
            k.bounce_col = k.bounce_col * s;
        }
    }
    return k;
}

fn blend_shade(i: VOut) -> vec4<f32> {
    // OBJECT GLASS (forge forerunner glass) — RE docs/hrek_re/09_re_glass.md. The real
    // rmgl is authored with albedo_color≈[0,0,0]: a BLACK diffuse body whose visible colour is
    // ENTIRELY reflection, tinted by env_tint_color (dominant) + a warm glancing_specular_tint
    // rim. (My earlier synthetic light-blue was wrong in the opposite direction — real forge
    // glass is DARK teal-reflective.) Material tints are carried per-part in the object-unused
    // instance slots: mat_spec2.rgb=albedo_color body, bump_xform.rgb=env_tint_color (w=1 flag),
    // fine_xform.rgb=glancing rim, detail_xform.y=fresnel steepness. Env reflection uses the sky
    // gradient as a stand-in for the authored environment_map cubemap (Tier 4 will add the cube).
    // Engine albedo_default opacity = base.a(placeholder 1) x detail.a x albedo_color.a; mattex.w==3
    // flags the lane (detail map in the detail slot, mattex_xform = detail xform, mattex.z = albedo_color.a).
    let gd_s = textureSample(detail_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw);
    // The engine draws transparents with back-face culling ON (reach_tag_test render_transparents
    // sub_14085FD10: cull mode 2 unless render_debug_transparent_cull is cleared; part flags carry no two-sided
    // bit), and Forge glass panes are authored as TWO face sets (front + flipped copy: 84%/16% of the coliseum
    // pane's last-drawn fragments face away / toward the camera). HMS's blend pipelines are cull-none, so both
    // copies composited (the pane blocked ~2x what one sheet does). Cull by the vertex NORMAL for OBJECT glass
    // (lm.w == 2) -- winding-agnostic and identical to the engine's cull for a properly authored copy. BSP glass
    // is left as it is (non-regression gate).
    if (i.lm.w > 1.5 && dot(normalize(i.normal), cam.cam_pos.xyz - i.world_pos) < 0.0) { discard; }
    if (i.lm.w > 1.5 && i.bump_xform.w > 0.5) {
        // EXACT engine compose (RE_glass_exact.md, glass.hlsl_include + environment_mapping.hlsl):
        //   out.rgb = specular + envmap ; body is BLACK (albedo_color=0)
        //   envmap  = cube(0.24 gray) · env_map_spec_contrib(0.75) · low_freq_spec · env_tint_color
        //   low_freq_spec = surrounding_light · spec_tint ; spec_tint = mix(normal_spec, glancing, fb)
        // The env cubemap (d_cubemap_vertical_noise) is a DIM ~uniform gray (avg 0.24, alpha 1) — NOT a
        // bright sky. Using the bright sky gradient before is what made the glass over-reflective/white.
        // Result: a subtle DIM dark-teal sheen on a mostly-see-through pane, warming at grazing.
        let gn = normalize(i.normal);
        let gv = normalize(cam.cam_pos.xyz - i.world_pos);
        let gnov = max(dot(gn, gv), 0.0);
        let gsteep = select(5.0, i.detail_xform.y, i.detail_xform.y > 0.01);
        let fb = pow(saturate(1.0 - gnov), gsteep);
        let spec_tint = mix(i.mat_spec2.rgb, i.fine_xform.rgb, fb); // normal_spec → glancing (warm rim)
        // Surrounding-light energy the reflection is scaled by. Offline stand-in: a modest scene
        // ambient (NOT the full-bright sky). Slight per-map warmth from the sun colour, kept modest.
        let surround = 1.4 + 0.15 * dot(light.sun_tint.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let low_freq = surround * spec_tint;
        let cube_avg = 0.24;
        let envmap0 = cube_avg * 0.75 * low_freq * i.bump_xform.rgb; // × env_tint_color (dark teal)
        // environment_mapping.hlsl_include per_pixel + glass.hlsl_include:125: with the material's
        // authored cube bound (xctl.z), envmap = cube.rgb·cube.a × env_tint × contribution × 0.25·(dom_rgb + fill)
        // — the object probe's lobe colours ride obj_probe1.rgb / tint.rgb, so the pane brightens with the scene.
        // The object's analytical light is the engine lobes+sun MERGE (obj_key_light), c[3] = fill +
        // 2*bounce, and the seed gate is c[2].w = b (glass.hlsl_include:228). Transparents get NO shadow mask
        // (shadows/shadow_mask.hlsl_include:33-34: shadow_mask = 1 outside SCOPE_LIGHTING_OPAQUE) and NO cloud
        // mask (analytical_mask = c[0].w = 1 for objects), so shadow_strength() is not applied on this lane.
        let ok = obj_key_light(i.obj_probe0, i.obj_probe1, i.tint.rgb, i.obj_light);
        let area_g = select(low_freq * 0.5, 0.25 * (i.obj_probe1.rgb + ok.fill), i.obj_probe0.w > 0.0);
        let refl_g = reflect(-gv, gn);
        let envmap = select(envmap0, env_reflection(refl_g, 0.1, 0.0, i.env_ctl.rgb, 1.0) * i.env_ctl.w * area_g, i.xctl.z > 0.5);
        // tiny analytic specular glint (roughness 0.1) — small, warms the rim
        // glass.hlsl_include + two_lobe_phong/phong_specular.hlsl_include: the analytic specular is
        // sat(R.L)^analytical_power * (1+analytical_power) * final_specular_tint * light * analytical_specular_contribution,
        // final_specular_tint = lerp(normal_specular_tint, glancing_specular_tint, (1-NoV)^steepness). The light is the
        // object's merged key light (OBJ-PROBE lanes) and there is NO area/vmf specular (area_specular_contribution=0).
        // fine_xform.w = analytical_power, mat_spec2.w = contribution (both 0 on the legacy synthetic path).
        let key_g = select(normalize(light.sun_dir.xyz), ok.dir, i.obj_probe0.w > 0.0);
        let lint_g = select(light.sun_tint.rgb, ok.col, i.obj_probe0.w > 0.0);
        let seed_b = select(1.0, ok.b, i.obj_probe0.w > 0.0);
        let engine_g = i.fine_xform.w > 0.5;
        let apow_g = select(6.8, i.fine_xform.w, engine_g);
        let acon_g = select(0.4, i.mat_spec2.w, engine_g);
        let rdl_g = saturate(dot(reflect(-gv, gn), key_g));
        let spec_engine = pow(rdl_g, apow_g) * (1.0 + apow_g) * spec_tint * lint_g * acon_g;
        let ndl_g = max(dot(gn, normalize(light.sun_dir.xyz)), 0.0);
        let spec_legacy = pow(ndl_g, 40.0) * spec_tint * 0.25;
        let spec_g = select(spec_legacy, spec_engine, engine_g);
        // engine: envmap from the authored cube when the template binds one (xctl.z); otherwise the material's
        // environment mapping is DYNAMIC (the engine's runtime-captured cubemap) → reflect the scene env cube
        // (sky capture) through the same compose: cube × contribution(0.75) × surrounding_light(0.25·(dom+fill))
        // × spec_tint × env_tint_color. #glass-dyn: Forge platform/bridge TOPS are rmgl `_0_1_0_2_0_0` with
        // albedo_color (0,0,0,2) — a black body whose whole look is this reflection; with the stand-in at 0
        // every platform floor rendered PURE BLACK (Pinnacle capture).
        // glass.hlsl_include:129, single-probe object entry :1375: envmap_area = raised_dot *
        // key_light * unshadowed * 0.25 * final_tint + 0.25*(dom+fill). Only the KEY-LIGHT SEED is
        // tinted by the spec tint; the 0.25*(dom+fill) ambient is UNTINTED. raised_dot =
        // sat(N.L*(max-min)/2 + (max+min)/2) with the raised_analytical_light_maximum/minimum
        // defines 1.0 / 0.1 (shared/utilities.hlsl_include:46-47).
        let raised_o = saturate(dot(gn, key_g) * 0.45 + 0.55);
        let seed_o = raised_o * lint_g * seed_b * 0.25 * spec_tint;
        let area_o = select(low_freq * 0.5, area_g + seed_o, i.obj_probe0.w > 0.0);
        // xctl.z == 1 also covers the engine's DYNAMIC cube = the per-cluster baked cubemap (bound
        // pre-decoded, alpha 1, lod = xctl.w): env = cube * contribution(env_ctl.w) *
        // env_tint(env_ctl.rgb) * area (environment_mapping.hlsl_include:152). The sky-capture
        // stand-in is used only when no cluster cubemap resolves (xctl.z == 0).
        let envmap_seeded = select(envmap0, env_reflection(refl_g, 0.1, i.xctl.w, i.env_ctl.rgb, 1.0) * i.env_ctl.w * area_o, i.xctl.z > 0.5);
        let env_dyn = env_reflection(refl_g, 0.1, 0.0, vec3<f32>(1.0), 0.0) * 0.75 * area_o * i.bump_xform.rgb;
        let env_final = select(envmap, select(env_dyn, envmap_seeded, i.xctl.z > 0.5), engine_g);
        let grgb = env_final + spec_g; // body black
        // Opacity: albedo.w = detail.w·albedo_color.w(2); no base/detail texture on the pane here, so
        // a low head-on alpha (mostly see-through) with a modest grazing lift via the spec fresnel.
        let ga = select(clamp(0.14 + 0.30 * fb, 0.0, 0.6), clamp(gd_s.a * i.mattex.z, 0.0, 1.0), i.mattex.w > 2.5);
        // HMS_MDBG probes for the rmgl object lane: 50 [ga, gd_s.a, fb], 51 env_final, 52 spec_g,
        // 53 area_o, 54 spec_tint, 55 lint_g, 56 raw dynamic cube, 57 flags [engine, xctl.z, probe0.w], 58 mask.
        let rdbg = __MDBG__;
        if (rdbg > 49.5 && rdbg < 60.5) {
            if (rdbg < 50.5) { return vec4<f32>(ga, gd_s.a, fb, 1.0); }
            if (rdbg < 51.5) { return vec4<f32>(env_final, 1.0); }
            if (rdbg < 52.5) { return vec4<f32>(spec_g, 1.0); }
            if (rdbg < 53.5) { return vec4<f32>(area_o, 1.0); }
            if (rdbg < 54.5) { return vec4<f32>(spec_tint, 1.0); }
            if (rdbg < 55.5) { return vec4<f32>(lint_g, 1.0); }
            if (rdbg < 56.5) { return vec4<f32>(env_reflection(refl_g, 0.1, 0.0, vec3<f32>(1.0), 0.0), 1.0); }
            if (rdbg < 57.5) { return vec4<f32>(select(0.0, 1.0, engine_g), i.xctl.z, i.obj_probe0.w, 1.0); }
            if (rdbg < 58.5) { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
            if (rdbg < 59.5) { return vec4<f32>(gd_s.rgb, 1.0); }
            return vec4<f32>(i.mattex.zw, i.mattex_xform.x, 1.0);
        }
        return vec4<f32>(apply_fog(grgb, i.world_pos, cam.cam_pos.xyz), ga); // engine fogs transparents (T6)
    }
    // docs/hrek_re/09_re_glass.md §2-3 (rmsh alpha_blend / shader_glass): albedo = base ×
    // (detail·4.59479) × albedo_color; opacity = base.a × detail.a × albedo_color.a, lifted by the
    // authored opacity fresnel (1-(1-a)(1-f)) ONLY when the triplet exists (fine_xform, w=2). The
    // pattern/frit lives in the base/detail maps' rgb+alpha.
    // mattex.w == 4: engine albedo two_detail (albedo.hlsl_include) for alpha-blend layered
    // materials — waterfalls: albedo = base × detail×4.59479 × detail2×4.59479 × albedo_color, alpha = product of
    // alphas, + self_illum simple (self_illum_map × colour × intensity × ILLUM_SCALE); every layer scrolls at its
    // own authored rate (xbump / xctl.xy), material diffuse_only.
    if (i.mattex.w > 3.5) {
        let tt = cam.time.x;
        let b2 = textureSample(base_tex, base_samp, i.uv * i.tile + i.scroll * tt);
        let d1 = textureSample(detail_tex, base_samp, i.uv * i.detail_xform.xy + i.detail_xform.zw + i.xbump.xy * tt);
        let d2 = textureSample(at_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw + i.xbump.zw * tt);
        let e2 = textureSample(emis_tex, base_samp, i.uv * i.fine_xform.xy + i.fine_xform.zw + i.xctl.xy * tt);
        let alb = pow(b2.rgb, vec3<f32>(2.2)) * pow(d1.rgb, vec3<f32>(2.2)) * 4.59479 * pow(d2.rgb, vec3<f32>(2.2)) * 4.59479 * i.tint.rgb;
        let a2 = clamp(b2.a * d1.a * d2.a * i.tint.a, 0.0, 1.0);
        let n2 = normalize(i.normal);
        let ndl2 = max(dot(n2, normalize(light.sun_dir.xyz)), 0.0);
        // The engine lights alpha-blend BSP surfaces with the SAME static lighting as opaques — the
        // per-pixel dual-VMF lightmap (i.lm.x) plus the analytical sun gated by baked visibility² and
        // the shadow map (mesh_shade's sun_term). PVL meshes carry their bake in i.tint (already in alb) → 1.
        var diff2 = vec3<f32>(1.0);
        if (i.lm.x > 0.5) {
            let lb2 = gpu_lightmap_tint(i.uv2, n2, i.normal, i.lm.y, i.lm.z);
            let vis2 = clamp(lb2.total.a, 0.0, 1.0);
            diff2 = lb2.total.rgb + light.sun_tint.rgb * (0.3183 * ndl2 * shadow_strength(i.world_pos) * vis2 * vis2);
        }
        // emis_tex is the HDR emissive (linear radiance, colour×intensity already folded in — see
        // decode_bsp_mesh) exactly as mesh_shade consumes it: no pow(2.2) decode here.
        let si2 = e2.rgb * i.mattex.rgb * illum_scale_now();
        let rgb2 = alb * diff2 + si2;
        // HMS_TDDBG per-term probes (read via HMS_HDR_DUMP): 1 albedo, 2 alpha, 3 diffuse light, 4 self-illum,
        // 5 base rgba raw, 6 detail alpha, 7 detail2 alpha.
        let tdd = TDDBG;
        if (tdd > 0.5 && tdd < 1.5) { return vec4<f32>(alb, 1.0); }
        if (tdd > 1.5 && tdd < 2.5) { return vec4<f32>(vec3<f32>(a2), 1.0); }
        if (tdd > 2.5 && tdd < 3.5) { return vec4<f32>(diff2, 1.0); }
        if (tdd > 3.5 && tdd < 4.5) { return vec4<f32>(si2, 1.0); }
        if (tdd > 4.5 && tdd < 5.5) { return vec4<f32>(b2.rgb, 1.0); }
        if (tdd > 5.5 && tdd < 6.5) { return vec4<f32>(vec3<f32>(d1.a), 1.0); }
        if (tdd > 6.5 && tdd < 7.5) { return vec4<f32>(vec3<f32>(d2.a), 1.0); }
        if (tdd > 7.5 && tdd < 8.5) { return vec4<f32>(d1.rgb, 1.0); }
        if (tdd > 8.5 && tdd < 9.5) { return vec4<f32>(d2.rgb, 1.0); }
        if (tdd > 9.5 && tdd < 10.5) { return vec4<f32>(i.tint.rgb, 1.0); }
        if (tdd > 10.5 && tdd < 11.5) { return vec4<f32>(vec3<f32>(i.tint.a, i.lm.x, i.lm.w), 1.0); }
        if (tdd > 11.5 && tdd < 12.5) { return vec4<f32>(e2.rgb, 1.0); }
        if (tdd > 12.5 && tdd < 13.5) { return vec4<f32>(i.mattex.rgb, 1.0); }
        if (tdd > 13.5 && tdd < 14.5) { return vec4<f32>(vec3<f32>(illum_scale_now()), 1.0); }
        return vec4<f32>(apply_fog(rgb2, i.world_pos, cam.cam_pos.xyz), a2);
    }
    let base_col = textureSample(base_tex, base_samp, i.uv * i.tile + i.scroll * cam.time.x);
    let base_rgb = pow(base_col.rgb, vec3<f32>(2.2));
    let has_det = i.detail_xform.x > 1e-4 || i.detail_xform.y > 1e-4;
    let dt = textureSample(detail_tex, base_samp, i.uv * i.detail_xform.xy + i.detail_xform.zw);
    let det_rgb = select(vec3<f32>(1.0), pow(dt.rgb, vec3<f32>(2.2)) * MESH_DETAIL_MULT, has_det);
    let det_a = select(1.0, dt.a, has_det);
    // mat_spec2.w > 1.5: BSP glass carries albedo_color in mat_spec2 ([rgb, 2+a]) because the vertex
    // colour holds the per-vertex probe area term (scene.rs decode_bsp_mesh); otherwise the tint.
    // calc_albedo_default_ps: albedo.rgb = base×detail×albedo_color.rgb, albedo.w = base.a×detail.a×albedo_color.w.
    let gl = i.mat_spec2.w > 1.5;
    let alb_rgb = select(i.tint.rgb, i.mat_spec2.rgb, gl);
    // Object glass (lm.w == 2) carries albedo_color in the same lane; its alpha multiplier is
    // applied UNCLAMPED (albedo_color.a = 2 on for_forge_glass-style materials doubles the map alpha before
    // the engine's saturate). The BSP lane keeps its clamp.
    let is_obj_g = i.lm.w > 1.5;
    let alb_a = select(select(i.tint.a, clamp(i.mat_spec2.w - 2.0, 0.0, 1.0), gl), max(i.mat_spec2.w - 2.0, 0.0), gl && is_obj_g);
    let body = base_rgb * det_rgb * alb_rgb;
    let albedo_a0 = clamp(base_col.a * det_a * alb_a, 0.0, 1.0);
    // blend.hlsl_include calc_alpha_blend_opacity (ALPHA_BLEND_SOURCE opacity_map_*): when the
    // material binds an opacity_texture (mattex.w = 1 alpha / 2 rgb-luma, at_tex slot, mattex_xform UV xform)
    // the albedo alpha is REPLACED by the map — final_opacity = 1-(1-opacity)(1-fresnel).
    let op_s = textureSample(at_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw);
    // mattex.y = 1 → the opacity map is gamma-curved (decode); 0 → linear (raw bytes = value).
    let op_rgb = select(op_s.rgb, pow(op_s.rgb, vec3<f32>(2.2)), i.mattex.y > 0.5);
    let op_luma = dot(op_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // mode 1 = map alpha, 2 = map rgb luma, 3 = opacity_map_alpha_and_albedo_alpha (blend.hlsl_include:150:
    // 1-(1-map.a)(1-fresnel)(1-albedo.a) — folded as an effective albedo alpha; the fresnel lift below applies once).
    let op_val = select(select(op_luma, op_s.a, i.mattex.w < 1.5), 1.0 - (1.0 - op_s.a) * (1.0 - albedo_a0), i.mattex.w > 2.5 && i.mattex.w < 3.5);
    let albedo_a = select(albedo_a0, clamp(op_val, 0.0, 1.0), i.mattex.w > 0.5 && i.mattex.w < 3.5);
    let n = normalize(i.normal);
    let v = normalize(cam.cam_pos.xyz - i.world_pos);
    let n_dot_v = dot(n, v);
    let has_fres = i.fine_xform.w > 1.5;
    var fres_op = 0.0;
    if (has_fres && n_dot_v > 0.0) {
        fres_op = clamp(i.fine_xform.x * pow(saturate(1.0 - n_dot_v), i.fine_xform.y) + i.fine_xform.z, 0.0, 1.0);
    }
    let out_a = select(albedo_a, 1.0 - (1.0 - albedo_a) * (1.0 - fres_op), has_fres);
    // 09_re_glass.md §2 + hangar mat 1 constants (cook_torrance, environment_map 0x214e, env_tint
    // (0.70,0.81,1.0) × environment_map_specular_contribution 0.3, specular_coefficient 1, roughness
    // 0.04, analytical_specular_contribution 0.2, base map near-BLACK): the pane is a dark,
    // blue-tinted REFLECTIVE surface — envmap_radiance × specular_scalar(1+fresnel) + analytical
    // specular, over an almost-black lit albedo.
    let refl = reflect(-v, n);
    let sun_dir_g = normalize(light.sun_dir.xyz);
    let ndl_glass = max(dot(n, sun_dir_g), 0.0);
    // Glass is the one lane with NO per-surface sun visibility to gate with — its vertex alpha is
    // repurposed as an OPACITY multiplier, so the baked mask is gone by shade time. Both glass sun
    // lanes are therefore gated by dbg.w = the map's OWN baked sun-visibility population mean
    // (Zealot 0.009, Forge World 0.254), which would otherwise take the scene sun at FULL strength
    // and light every pane in a sealed interior. An atlas-only map reports 1.0.
    let sun_reach = light.dbg.w;
    let glass_diff = 0.3183 * ndl_glass * dot(light.sun_tint.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) * sun_reach;
    // xctl.w = the engine's per_pixel base LOD ((env_roughness_offset-0.5)*8, scene.rs). An AUTHORED
    // cube (xctl.z == 1) takes the engine LOD only (environment_mapping.hlsl_include:44-45,74-75 has
    // no distance term); the far distance bias stays on the procedural/dynamic stand-ins.
    let env_lod_g = select(max(clamp((i.vdepth - 15.0) / 20.0, 0.0, 5.0), i.xctl.w), i.xctl.w, i.xctl.z > 0.5 && i.xctl.z < 1.5);
    let menv_g = max(i.bump_detail_xform.w, 0.0);
    let scoef_g = select(1.0, i.env_ctl.w, i.env_ctl.w > 0.0);
    // xctl.z == 2: the engine samples a runtime cubemap of the surroundings; stand-in = the sky cube's
    // LUMINANCE (neutral grey), so interior glass does not take on the sky's blue.
    let env_raw = env_reflection(refl, i.mat_rough, env_lod_g, i.env_ctl.rgb, select(i.xctl.z, 0.0, i.xctl.z > 1.5));
    let env_lum = vec3<f32>(dot(env_raw, vec3<f32>(0.2126, 0.7152, 0.0722)));
    // A dynamic object (lm.w = 2) carries its light probe in the object-probe lanes
    // (obj_probe0 = [dom_dir, bandwidth], obj_probe1.rgb = dominant lobe colour, .w = sun mask, itint.rgb = fill).
    let obj_area = i.lm.w > 1.5 && i.obj_probe0.w > 0.0 && i.xbump.w < 0.5;
    let has_area = i.xbump.w > 0.5 || obj_area;
    let area_pv = i.xbump.w > 1.5;
    // Glass control lanes (scene.rs): xbump = [albedo_blend, diffuse_coefficient,
    // 1 + specular_mask digit (+4 when the specular_mask_texture rides the material_texture slot), 2 (BSP
    // per-vertex) / 0 (object)]. Present on BSP glass_pv meshes and on object rmsh glass with a GlassOp.
    let eng = (area_pv || obj_area) && i.xbump.z > 0.5;
    let albedo_blend = select(0.0, clamp(i.xbump.x, 0.0, 1.0), eng);
    let dcoef = select(1.0, max(i.xbump.y, 0.0), eng);
    let smask_code = select(0.0, i.xbump.z - 1.0, eng);
    let smask_tex = smask_code > 3.5;
    let smask_digit = select(smask_code, smask_code - 4.0, smask_tex);
    // specular_mask.hlsl_include:29-52: 0 -> 1, 1 -> albedo.w (Zealot cov_glass: the diffuse alpha), 2 -> albedo.w
    // x mask_tex.a, 3 -> mask_tex.a. BSP: the mask texture's own UV xform is not carried, so it
    // samples at the base uv (TODO: carry the authored tiling). Objects: specular_mask_texture_xform.xy
    // rides bump_detail_xform.xy (z = 1 flag) -- the coliseum window tiles its frit mask 8x
    // (specular_mask.hlsl_include:45,51).
    let obj_lanes_g = obj_area && i.bump_detail_xform.z > 0.5;
    let muv = i.uv * select(vec2<f32>(1.0), i.bump_detail_xform.xy, obj_lanes_g);
    let mtex_a = select(1.0, textureSample(mat_tex, base_samp, muv).a, smask_tex);
    var mask_g = 1.0;
    if (smask_digit > 0.5 && smask_digit < 1.5) { mask_g = albedo_a0; }
    if (smask_digit > 1.5 && smask_digit < 2.5) { mask_g = albedo_a0 * mtex_a; }
    if (smask_digit > 2.5) { mask_g = mtex_a; }
    // cook_torrance_core:244-246 env_reflectance = contribution x specular_mask x specular_coefficient (rmgl:
    // contribution only, mask digit 0 -> 1); x (1 + opacity fresnel) = the engine specular_scalar.
    let env_g = select(env_raw, env_lum, i.xctl.z > 1.5) * menv_g * scoef_g * mask_g * (1.0 + fres_op);
    // Probe lobe per pixel. BSP glass_pv (decode_bsp_mesh #glass-engine): tangent = [dom_rgb, bandwidth],
    // uv2 = dom_dir.xy, tint.a = dom_dir.z, tint.rgb = 0.25*(dom+fill) -> fill = 4*area - dom. Objects: OBJ-PROBE.
    let dom_rgb_g = select(select(vec3<f32>(0.0), max(i.tangent.xyz, vec3<f32>(0.0)), area_pv), max(i.obj_probe1.rgb, vec3<f32>(0.0)), obj_area);
    let fill_rgb_g = select(select(vec3<f32>(0.0), max(4.0 * i.tint.rgb - i.tangent.xyz, vec3<f32>(0.0)), area_pv), max(i.itint.rgb, vec3<f32>(0.0)), obj_area);
    let dir_pv = vec3<f32>(i.uv2.x, i.uv2.y, i.tint.a);
    let dom_dir_g = select(select(vec3<f32>(0.0, 0.0, 1.0), dir_pv / max(length(dir_pv), 1e-4), area_pv), normalize(i.obj_probe0.xyz), obj_area);
    let bw_g = select(select(1.0, clamp(i.tangent.w, 0.0, 1.0), area_pv), clamp(i.obj_probe0.w, 0.0, 1.0), obj_area);
    let area_uniform = 0.25 * (dom_rgb_g + fill_rgb_g);
    let area_legacy = select(i.xbump.rgb, area_uniform, area_pv || obj_area);
    // BSP lanes: si_ctl = [specular_tint, steepness], obj_probe0 = [fresnel_color, ct (1 rmsh cook_torrance, 0 rmgl)].
    // Object lanes (#glass-engine): si_ctl = [specular_tint, steepness], fres_rgb = fresnel_color.
    let ct_bsp = area_pv && i.obj_probe0.w > 0.5;
    let rmgl_bsp = area_pv && i.obj_probe0.w < 0.5;
    let tinted = ct_bsp || (obj_area && eng);
    let fb_g = pow(saturate(1.0 - max(n_dot_v, 0.0)), max(i.si_ctl.w, 1e-3));
    let fres_col_g = select(i.obj_probe0.rgb, i.fres_rgb.rgb, obj_area);
    // cook_torrance_core:80-83: F0 = lerp(specular_tint, albedo, albedo_blend); final_tint = lerp(F0, fresnel_color, (1-NoV)^steep).
    let f0_g = mix(i.si_ctl.rgb, body, albedo_blend);
    let spec_tint_g = mix(f0_g, fres_col_g, fb_g);
    // DIRECTIONAL area (cook_torrance_core:227-236 + materials/diffuse_specular.hlsl:41-62):
    // sh_glossy = final_tint * (g_diffuse_power_specular(dot(dom_dir, R)*0.5+0.5, bandwidth, roughness) * 3 * dom_rgb
    // + 0.25 * fill_rgb). The engine lobe peaks where the mirror direction meets the dominant light
    // and falls to the fill floor away from it (a uniform 0.25*(dom+fill) instead puts the same flat
    // sheen on every pane and washes out Zealot's hex pattern, 46-80% of the glass luminance).
    // rmgl (glass.hlsl:129) keeps the UNTINTED, non-directional 0.25*(dom+fill).
    let rdotd = dot(dom_dir_g, refl);
    // objects: this PART's `roughness` (LUT z) rides spec_rgb.w (mat_rough is the model's shader-0 value).
    let rough_lut_g = select(i.mat_rough, i.spec_rgb.w, obj_lanes_g);
    let lut_g = textureSampleLevel(dps_lut, vmf_samp, vec3<f32>(rdotd * 0.5 + 0.5, bw_g, clamp(rough_lut_g, 0.0, 1.0)), 0.0).r * 3.0;
    // objects: c[3] = fill + 2*bounce (bounce-to-ambient, doc 16 s2.3) -- the merge helper's fill.
    let okg = obj_key_light(i.obj_probe0, i.obj_probe1, i.itint.rgb, i.obj_light);
    let fill_area_g = select(fill_rgb_g, max(okg.fill, vec3<f32>(0.0)), obj_lanes_g);
    let area_lobe = lut_g * dom_rgb_g + 0.25 * fill_area_g;
    // Analytical-light seed (entry_points:374 / glass.hlsl:227 / object entry :1375): raised_dot(N.L) * light *
    // visibility * 0.25 * final_tint; raised_dot = sat(N.L*0.45 + 0.55) (raised_analytical_light_maximum 1.0 /
    // minimum 0.1, shared/utilities.hlsl_include:46-47). BSP: the scene sun gated by the map's baked sun
    // visibility mean (#sun-reach) + shadow map; objects: the probe key light (dominant lobe x sun mask).
    // objects: the key light is the engine lobes+sun MERGE (obj_key_light; c[0].w = 1, seed gate
    // c[2].w = b) and transparents take NO shadow mask / cloud mask (shadow_mask.hlsl_include:33-34,
    // analytical_mask.hlsl_include:72 -> 1), so shadow_strength() applies only to the BSP lane.
    let key_dir_g = select(sun_dir_g, select(normalize(i.obj_probe0.xyz), okg.dir, obj_lanes_g), obj_area);
    let key_rgb_g = select(light.sun_tint.rgb * sun_reach, select(i.obj_probe1.rgb * clamp(i.obj_probe1.w, 0.0, 1.0), okg.col, obj_lanes_g), obj_area);
    let seed_gate_g = select(1.0, okg.b, obj_lanes_g);
    let shadow_g = select(shadow_strength(i.world_pos), 1.0, obj_lanes_g);
    let raised_g = saturate(dot(n, key_dir_g) * 0.45 + 0.55);
    let seed_g = raised_g * key_rgb_g * shadow_g * seed_gate_g * 0.25 * spec_tint_g;
    let area_engine = select(area_lobe * spec_tint_g, area_uniform, rmgl_bsp) + seed_g;
    let area_env = select(area_legacy, area_engine, tinted || rmgl_bsp);
    let env_ga = select(env_g, env_g * area_env, has_area);
    let h_g = normalize(sun_dir_g + v);
    let shin_g = 2.0 / max(i.mat_rough * i.mat_rough, 0.002);
    let spec_bsp_g = pow(max(dot(n, h_g), 0.0), min(shin_g, 512.0)) * light.sun_tint.rgb * i.env_ctl.rgb * 0.2 * (1.0 + fres_op) * sun_reach * mask_g;
    // objects: the engine cook-torrance analytic specular (cook_torrance_core:120-153,238-242) with the
    // merged key light: Beckmann D (m = analytical_roughness, fres_rgb.w), G = sat(min(2NH.NV/HV, 2NH.NL/HV)),
    // F = final_tint; radiance = D*G*F/(N.V)/pi * light * mask * specular_coefficient * analytical_specular_contribution
    // * (1 + opacity fresnel). TODO: analytical_specular_contribution is not carried per part, so
    // the BSP lane's 0.2 (the Forge glass value) stands in for every material.
    let m_ct = clamp(select(0.25, i.fres_rgb.w, obj_lanes_g), 0.02, 1.0);
    let h_o = normalize(key_dir_g + v);
    let nh_o = max(dot(n, h_o), 1e-4);
    let nl_o = dot(n, key_dir_g);
    let hv_o = max(dot(h_o, v), 1e-4);
    let tan2_o = (1.0 - nh_o * nh_o) / (nh_o * nh_o);
    let d_o = exp(-tan2_o / (m_ct * m_ct)) / (m_ct * m_ct * nh_o * nh_o * nh_o * nh_o + 0.00001);
    let g_o = saturate(min(2.0 * nh_o * n_dot_v / hv_o, 2.0 * nh_o * nl_o / hv_o));
    let spec_obj_g = select(vec3<f32>(0.0), d_o * g_o * spec_tint_g / max(n_dot_v, 1e-3) * 0.3183 * key_rgb_g * mask_g * scoef_g * 0.2 * (1.0 + fres_op), n_dot_v > 0.0 && nl_o > 0.0);
    let spec_g = select(spec_bsp_g, spec_obj_g, obj_lanes_g);
    // Body = diffuse_radiance x albedo x diffuse_coefficient (entry_points:346-365 + cook_torrance_core:248):
    // rmsh: dual_vmf_diffuse(N) = (lut(N.dom, bw) * dom + 0.25 * fill) / pi (spherical_harmonics.hlsl:68) + the
    // analytical light sat(N.L) * light / pi * vis; rmgl (glass.hlsl:219-226,131): the analytical light ONLY —
    // an rmgl pane in a sunless interior has a black body (Ivory glass_zen_cheap: diffuse_coefficient 10 x 0).
    // Lanes without the engine control constants keep the area x ratio (per-vertex) /
    // x pi (centroid) stand-ins.
    let cdom_g = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(dot(n, dom_dir_g) * 0.5 + 0.5, bw_g), 0.0).r;
    let dvmf_g = (cdom_g * dom_rgb_g + 0.25 * fill_area_g) * 0.3183;
    let key_body_g = 0.3183 * max(dot(n, key_dir_g), 0.0) * key_rgb_g * shadow_g;
    // objects: + the bounce light sat(N.bounce_dir) * bounce / pi (entry_points:370, k_ps_bounce_light).
    let bounce_body_g = select(vec3<f32>(0.0), 0.3183 * max(dot(n, okg.bounce_dir), 0.0) * okg.bounce_col, obj_lanes_g);
    let body_engine = select(dvmf_g + key_body_g + bounce_body_g, key_body_g, rmgl_bsp) * dcoef;
    let body_legacy = select(area_legacy * 3.14159, area_legacy * i.tint.a, area_pv);
    let body_e = select(body_legacy, body_engine, tinted || rmgl_bsp);
    let body_lit = select(body * glass_diff, body * body_e, has_area);
    let rgb = body_lit + env_ga + spec_g;
    // HMS_MDBG per-term probes for the rmsh glass lane (read via HMS_HDR_DUMP; alpha 1 so the
    // probe REPLACES the background): 30 [out_a, albedo_a, fres_op], 31 body (albedo), 32 env_g (cube x
    // reflectance x scalar), 33 area_env, 34 env_ga, 35 spec_g, 36 body_lit, 37 [mask_g, mtex_a, n_dot_v],
    // 38 dom_rgb, 39 fill_rgb, 40 key_rgb, 41 [lut_g, bw, rdotd], 42 spec_tint_g, 43 seed_g, 44 raw cube,
    // 45 [op_luma, base_col.a, det_a], 46 lane flags [eng, has_area, obj_area], 47 pane mask (1,0,0).
    let gdbg = __MDBG__;
    if (gdbg > 29.5 && gdbg < 49.5) {
        if (gdbg < 30.5) { return vec4<f32>(out_a, albedo_a, fres_op, 1.0); }
        if (gdbg < 31.5) { return vec4<f32>(body, 1.0); }
        if (gdbg < 32.5) { return vec4<f32>(env_g, 1.0); }
        if (gdbg < 33.5) { return vec4<f32>(area_env, 1.0); }
        if (gdbg < 34.5) { return vec4<f32>(env_ga, 1.0); }
        if (gdbg < 35.5) { return vec4<f32>(spec_g, 1.0); }
        if (gdbg < 36.5) { return vec4<f32>(body_lit, 1.0); }
        if (gdbg < 37.5) { return vec4<f32>(mask_g, mtex_a, n_dot_v, 1.0); }
        if (gdbg < 38.5) { return vec4<f32>(dom_rgb_g, 1.0); }
        if (gdbg < 39.5) { return vec4<f32>(fill_rgb_g, 1.0); }
        if (gdbg < 40.5) { return vec4<f32>(key_rgb_g, 1.0); }
        if (gdbg < 41.5) { return vec4<f32>(lut_g, bw_g, rdotd, 1.0); }
        if (gdbg < 42.5) { return vec4<f32>(spec_tint_g, 1.0); }
        if (gdbg < 43.5) { return vec4<f32>(seed_g, 1.0); }
        if (gdbg < 44.5) { return vec4<f32>(env_raw, 1.0); }
        if (gdbg < 45.5) { return vec4<f32>(op_luma, base_col.a, det_a, 1.0); }
        if (gdbg < 46.5) { return vec4<f32>(select(0.0, 1.0, eng), select(0.0, 1.0, has_area), select(0.0, 1.0, obj_area), 1.0); }
        if (gdbg < 47.5) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
        if (gdbg < 48.5) { return vec4<f32>(i.tint.rgb, 1.0); }
        return vec4<f32>(base_rgb, 1.0);
    }
    return vec4<f32>(apply_fog(rgb, i.world_pos, cam.cam_pos.xyz), out_a); // engine fogs transparents (T6)
}

// Straight-alpha / multiply / double_multiply transparent entry (blend_mode 0/2/3).
// Non-premultiplied rgb: the SrcAlpha/InvSrcAlpha, ZERO/SrcColor and DstColor/SrcColor
// blend states consume the colour directly.
@fragment
fn fs_blend(i: VOut) -> @location(0) vec4<f32> {
    return blend_shade(i);
}

// Multiply / double_multiply transparents (blend_mode 2/3). Engine entry_points.hlsl_include
// :422-426 (BLEND_MULTIPLICATIVE): out.rgb = max(0, albedo + self_illum) * BLEND_MULTIPLICATIVE -- NO lighting,
// NO fog, NO exposure; the x1 / x2 comes from the fixed-function state (ZERO/SRC_COLOR vs DST_COLOR/SRC_COLOR
// = 2.src.dst), so this returns the plain albedo. albedo = base x detail x 4.59479 x albedo_color (mat_spec2 =
// [rgb, 2+a] lane; itint = the raw instance tint when absent). Routing these through the LIT
// blend_shade instead would give a probe-lit body of ~0 in a dark interior, so 2·dst·0 paints a
// multiply skylight (Sword Base's m20_glass_panel_no_break_large, material_model none) BLACK.
@fragment
fn fs_blend_multiplicative(i: VOut) -> @location(0) vec4<f32> {
    let base_col = textureSample(base_tex, base_samp, i.uv * i.tile + i.scroll * cam.time.x);
    let has_det = i.detail_xform.x > 1e-4 || i.detail_xform.y > 1e-4;
    let dt = textureSample(detail_tex, base_samp, i.uv * i.detail_xform.xy + i.detail_xform.zw);
    let det_rgb = select(vec3<f32>(1.0), pow(dt.rgb, vec3<f32>(2.2)) * MESH_DETAIL_MULT, has_det);
    let gl = i.mat_spec2.w > 1.5;
    let alb_rgb = select(i.itint.rgb, i.mat_spec2.rgb, gl);
    let albedo = pow(base_col.rgb, vec3<f32>(2.2)) * det_rgb * alb_rgb;
    // Only materials with a self-illum option carry an emissive map in emis_tex; for every
    // other multiplicative material that slot holds an unrelated placeholder (Countdown's
    // unsc_danger_ramp floor overlay rendered that texture as black/white noise). si_ctl.x is the
    // self-illum mode lane (0 = none).
    let has_si = i.si_ctl.x > 0.5;
    let emis = select(vec3<f32>(0.0), textureSample(emis_tex, base_samp, i.uv * i.tile).rgb * illum_scale_now(), has_si);
    // HMS_MDBG probes (print-only diag): 61 base, 62 detail term, 63 albedo_color lane, 64 [tile, detail_xform.xy].
    let mdbg = __MDBG__;
    if (mdbg > 60.5 && mdbg < 64.5) {
        if (mdbg < 61.5) { return vec4<f32>(base_col.rgb, 1.0); }
        if (mdbg < 62.5) { return vec4<f32>(det_rgb, 1.0); }
        if (mdbg < 63.5) { return vec4<f32>(alb_rgb, 1.0); }
        return vec4<f32>(i.tile.x, i.detail_xform.x, i.detail_xform.y, 1.0);
    }
    if (mdbg > 64.5 && mdbg < 65.5) { return vec4<f32>(0.5, 0.5, 0.5, 1.0); } // neutral: result == destination
    return vec4<f32>(max(albedo + emis, vec3<f32>(0.0)), 1.0);
}

// T1 pre_multiplied_alpha entry (blend_mode 1). Engine
// convert_to_render_target_premultiplied_alpha premultiplies rgb by the coverage in-shader,
// then the fixed-function blend is ONE/INV_SRC_ALPHA. Emit rgb·a so src + dst·(1−a) composites
// exactly. (For the coverage-scaled glass body this equals the straight-alpha result.)
@fragment
fn fs_blend_premul(i: VOut) -> @location(0) vec4<f32> {
    let c = blend_shade(i);
    return vec4<f32>(c.rgb * c.a, c.a);
}

// Additive hologram / force-field / energy surface. blend==1 surfaces composite
// ONE/ONE over the background — a pure additive glow (the blend op ignores alpha). The
// emissive tex (baked self_illum_color×intensity) + the albedo form the glow; a fresnel
// rim lift brightens grazing edges (holograms glow at the silhouette); the base alpha
// (coverage) fades masked/hollow texels so the shape reads. Animated scroll respected.
// function.hlsl evaluate_periodic_internal for the animated-parameter kinds fs_holo
// evaluates (x = the wrapped input in 0..1): 2/3 cosine, 4/5 diagonal wave, 6/7 slide; anything else
// (kind 0 = no animation) returns 0 so the f = 0 lanes are used unchanged.
fn halo_periodic(kind: f32, x: f32) -> f32 {
    if (kind > 1.5 && kind < 3.5) { return 0.5 * (cos(6.2831853 * x) + 1.0); }
    if (kind > 3.5 && kind < 5.5) { return select(2.0 - 2.0 * x, 2.0 * x, x < 0.5); }
    if (kind > 5.5 && kind < 7.5) { return x; }
    return 0.0;
}

@fragment
fn fs_holo(i: VOut) -> @location(0) vec4<f32> {
    let t = cam.time.x;
    // bump_xform.w flags: >2.5 = STATIC icon (armor-ability icon — no animation at all,
    // crisp alpha-cut, no rim/pulse/vignette); 1.5..2.5 = forcefield energy field. 0 = normal.
    let is_static = i.bump_xform.w > 2.5;
    // effect-scenery billboards were static. cam.time is already bound to this pass,
    // so animate for free — a slow UV drift + a per-sprite breathing pulse (phase offset by
    // world position so effects don't pulse in lockstep) makes mist/spray/glow read as alive.
    let auv = select(i.uv * i.tile + i.scroll * t + vec2<f32>(0.03, 0.06) * t, i.uv * i.tile, is_static);
    let tex = textureSample(base_tex, base_samp, auv);
    let albedo = pow(tex.rgb, vec3<f32>(2.2)) * i.tint.rgb;
    // Tint the self-illum by the team colour (engine: self_illum × primary_change_color).
    // Holo/objective markers carry their image in the emissive map; without this they glowed
    // GRAY. i.tint carries the forge team colour for objects (blue etc.).
    let em = pow(textureSample(emis_tex, base_samp, auv).rgb, vec3<f32>(2.2)) * i.tint.rgb;
    let n = normalize(i.normal);
    let v = normalize(cam.cam_pos.xyz - i.world_pos);
    // screen-space-derivative tangent frame (uniform control flow) for the multilayer
    // halogram's tangent-space view offset (mode 7 below). Object meshes carry no stored tangent.
    let ml_dpx = dpdx(i.world_pos);
    let ml_dpy = dpdy(i.world_pos);
    let ml_dux = dpdx(i.uv);
    let ml_duy = dpdy(i.uv);
    let edge = pow(clamp(1.0 - abs(dot(n, v)), 0.0, 1.0), 2.0);   // rim glow
    let pulse = 0.7 + 0.3 * sin(t * 2.0 + i.world_pos.x * 0.5 + i.world_pos.y * 0.5);
    // Particle sprites are authored RGB-on-BLACK with OPAQUE alpha (a=255), so gating
    // on tex.a does nothing → the whole quad glows = solid square. Under One/One additive,
    // black already adds nothing, so LUMINANCE is the natural coverage. Gate on max(alpha,
    // luminance) so the sprite shape reads and the black border adds ~0. A soft radial
    // falloff on the quad further guarantees it never reads as a hard square.
    let lum = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
    let cov = max(clamp(tex.a, 0.0, 1.0), clamp(lum * 2.0, 0.0, 1.0));
    // Forcefield (shield wall/door): bump_xform.w==2 flags an energy FIELD that covers the whole
    // surface — drop the billboard radial vignette (it faded the wall's edges to nothing) and hold a
    // gentler pulse so the plasma churn reads as a flat translucent field, not a glowing sprite.
    let is_ff = i.bump_xform.w > 1.5 && i.bump_xform.w < 2.5;
    // Exact shader_halogram palettized_plasma (self_illumination_halogram.hlsl_include) +
    // additive_detail overlay (overlays.hlsl_include). bump_xform.z==1 → the engine lanes are present:
    // tile/mattex.xy/scroll = noise_a xform+scroll, detail_xform/bump_xform.xy = noise_b, mat_spec2 =
    // self_illum colour+intensity, fine_xform = [v_coordinate, alpha_mask, alpha_modulation, has_overlay],
    // bump_detail_xform/env_ctl = overlay xforms, si_ctl = overlay tint+intensity, mattex.zw = overlay scroll.
    // Samples are taken here (uniform control flow) so the branch below only selects.
    let ff_uv_a = i.uv * i.tile + i.mattex.xy + i.scroll * t;
    let ff_uv_b = i.uv * i.detail_xform.xy + i.detail_xform.zw + i.bump_xform.xy * t;
    let ff_na_r = textureSample(base_tex, base_samp, ff_uv_a).r;
    let ff_nb_r = textureSample(emis_tex, base_samp, ff_uv_b).r;
    // xctl = [palette, overlay, noise_a, noise_b] gamma flags (1 = bitmap curve is gamma → decode; 0 = linear raw)
    let ff_na = select(ff_na_r, pow(ff_na_r, 2.2), i.xctl.z > 0.5);
    let ff_nb = select(ff_nb_r, pow(ff_nb_r, 2.2), i.xctl.w > 0.5);
    // (self_illumination_halogram.hlsl_include compute_depth_fade): depth_fade =
    // saturate((scene_depth - fragment_depth) * view_dot_normal / depth_fade_range), depths along the view;
    // where the field meets geometry the fade → 0 and the palette index climbs by alpha_modulation_factor
    // (the bright contact glow). xbump.x = depth_fade_range (0 → fade 1).
    // Lane MODE (bump_xform.z): 1 palettized_plasma, 2 palettized change-colour, 3 simple,
    // 4 plasma. Modes 3/4 keep the edge-fade centre tint in xbump (no depth fade / palette there).
    let ff_mode = i.bump_xform.z;
    var ff_dfa = 1.0;
    if (ff_mode < 2.5 && i.xbump.x > 1e-4) {
        let px = vec2<i32>(i.clip.xy);
        let bed_ndc = textureLoad(scene_depth, px, 0);
        if (bed_ndc < 0.9999) {
            let dim = vec2<f32>(textureDimensions(scene_depth));
            let uv2 = i.clip.xy / dim;
            let ndc = vec3<f32>(uv2.x * 2.0 - 1.0, 1.0 - uv2.y * 2.0, bed_ndc);
            let bedH = cam.inv_view_proj * vec4<f32>(ndc, 1.0);
            let bed = bedH.xyz / bedH.w;
            // #grid-render Both depths are VIEW-FORWARD distances in the engine, not radial: `scene_depth`
            // is 1/(global_depth_constants.z - depth * .y) and `particle_depth` is
            // |dot(fragment_to_camera_world, global_camera_forward)|
            // (self_illumination_halogram.hlsl_include compute_depth_fade; confirmed in the
            // compiled halogram_templates\_2_10_1_0_0_2_0 pixel shader, `dp3 r0.z, v6, cb0[1]`).
            // The clip w of a perspective projection IS that distance, so the fragment's is
            // i.vdepth and the bed's is the same row-3 dot. Radial `length()` overstated the
            // delta by 1/cos(off-axis angle) - up to 1.4x at the corners of a 90-degree view -
            // so the contact fade was too narrow away from the screen centre.
            let frag_d = i.vdepth;
            let bed_d = cam.view_proj[0][3] * bed.x + cam.view_proj[1][3] * bed.y + cam.view_proj[2][3] * bed.z + cam.view_proj[3][3];
            // view_dot_normal is SIGNED in the engine; back-facing halogram fragments are
            // discarded below (the engine culls them), so the two forms agree on what draws.
            let vdn = abs(dot(normalize(i.normal), normalize(cam.cam_pos.xyz - i.world_pos)));
            ff_dfa = clamp((bed_d - frag_d) * vdn / i.xbump.x, 0.0, 1.0);
        }
    }
    // Engine samples alpha_mask_map.a per texel (self_illumination_halogram.hlsl_include
    // calc_self_illumination_palettized_plasma_ps). fine_xform.y < 0 flags the authored mask bound in the
    // at_tex slot with mattex_xform as its UV xform (BSP forcefields: Spire's dome honeycomb;
    // Object halograms too -- the Forge grid's mask IS its grid-line texture); otherwise .y is
    // the mean-alpha constant (1x1 default masks, e.g. the shield doors' alpha_grey50). Sampled
    // unconditionally (uniform control flow), selected below.
    let ff_am = textureSample(at_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw).a;
    let ff_alpha = select(i.fine_xform.y, ff_am, i.fine_xform.y < 0.0);
    let ff_index = clamp(abs(ff_na - ff_nb) + (1.0 - ff_alpha * ff_dfa) * i.fine_xform.z, 0.0, 1.0);
    // The palette is a LUT: the engine binds it CLAMPED (rmt2 texture entry sampler byte 0x11,
    // shared with rasterizer\direction_lut / diffusetable), so index 1.0 reads the LAST texel. base_samp
    // REPEATS, which blended the black end with the white start (0.5 -> x5 = 2.5 HDR wherever the grid's
    // cell index saturated: 66% of its surface). Clamp to the texel centres = clamp-to-edge addressing.
    // xbump.y >= 4 flags a palette authored WITHOUT mips (the grid's palette_simple_gradient_b1): the
    // engine has only its base level, so sample LOD 0 instead of the index-derivative mip chain.
    let ff_pal_dim = vec2<f32>(textureDimensions(detail_tex));
    let ff_pal_uv = clamp(vec2<f32>(ff_index, i.fine_xform.x), 0.5 / ff_pal_dim, 1.0 - 0.5 / ff_pal_dim);
    let ff_pal_r = select(textureSample(detail_tex, base_samp, ff_pal_uv).rgb, textureSampleLevel(detail_tex, base_samp, ff_pal_uv, 0.0).rgb, ff_mode < 2.5 && i.xbump.y >= 3.5);
    let ff_pal = select(ff_pal_r, pow(ff_pal_r, vec3<f32>(2.2)), i.xctl.x > 0.5);
    let ff_uv_o = i.uv * i.bump_detail_xform.xy + i.bump_detail_xform.zw + i.mattex.zw * t;
    // overlay_detail_map = its OWN bitmap in mat_tex when xbump.y > 0.5 (1 = linear, 2 = gamma
    // curve), scrolled by its own TranslationX/Y animation (xbump.zw); else the overlay map doubles as
    // the detail factor (the shield doors bind the same bitmap for both usages).
    let ff_uv_d = i.uv * i.env_ctl.xy + i.env_ctl.zw + i.xbump.zw * t;
    let ff_ov_r = textureSample(bump_tex, base_samp, ff_uv_o).rgb;
    let ff_od_mode = i.xbump.y - select(0.0, 4.0, i.xbump.y >= 3.5); // low part: overlay_detail mode (0/1/2)
    let ff_od_r = select(textureSample(bump_tex, base_samp, ff_uv_d).rgb, textureSample(mat_tex, base_samp, ff_uv_d).rgb, ff_od_mode > 0.5);
    let ff_od_gamma = select(i.xctl.y > 0.5, ff_od_mode > 1.5, ff_od_mode > 0.5);
    let ff_ov = select(ff_ov_r, pow(ff_ov_r, vec3<f32>(2.2)), i.xctl.y > 0.5);
    // Od mode 3 = the shader binds NO overlay_detail_map (null tag ref; Condemned's projector
    // lamp / cpu_readouts): the engine samples the option's default_detail (sRGB 50% grey) whose
    // x DETAIL_MULTIPLIER is 1.0 -- NOT the overlay map re-sampled at the detail xform.
    let ff_od = select(select(ff_od_r, pow(ff_od_r, vec3<f32>(2.2)), ff_od_gamma), vec3<f32>(1.0 / 4.59479), ff_od_mode > 2.5);
    // ---- HALO 4 `srf_ca_boundary*` (the Forge GRID piece, the boundary walls) -- #grid-render --
    // bump_xform.w == 8 selects this lane. The law is the family's shipped forward transparent
    // pixel shader (entries 21/22/23/31/32/33, identical bodies; hms-app h4/shading.rs `Boundary`
    // carries the transcription + the cbuffer map). Slots: base = noise A, emis = noise B,
    // detail = palette, bump = overlay, mat = overlay detail, at = self-illum alpha mask (.a).
    // Lanes: tile/mattex.xy/scroll = noise A xform + scroll, detail_xform/bump_xform.xy = noise B,
    // mattex_xform = alpha mask, bump_detail_xform/mattex.zw = overlay, env_ctl/obj_probe0.zw =
    // overlay detail, fine_xform = palette xform, si_ctl = up160 (tint, overlay intensity),
    // mat_spec2 = up161, spec_rgb = up162, xbump = up163, xctl/fres_rgb.x = the gamma flags.
    let h4b_uv_d = i.uv * i.env_ctl.xy + i.env_ctl.zw + i.obj_probe0.zw * t;
    let h4b_od_raw = textureSample(mat_tex, base_samp, h4b_uv_d).rgb;
    if (i.bump_xform.w > 7.5 && i.bump_xform.w < 8.5) {
        let ndv = clamp(dot(v, n), 0.0, 1.0);
        // palette-index fresnel: sat(scale * pow(sat(lerp(N.V, 1 - N.V, invert)), power))
        let fres = clamp(i.mat_spec2.y * pow(clamp(mix(ndv, 1.0 - ndv, i.mat_spec2.w), 0.0, 1.0), i.mat_spec2.z), 0.0, 1.0);
        // uv.y fade (up163.z = 0 on every shipped material -> the division saturates to 0)
        let hfade = select(0.0, clamp(1.0 - i.uv.y / i.xbump.z, 0.0, 1.0), i.xbump.z > 1e-6);
        let edge = mix(fres, 1.0, hfade);
        // soft intersection: (1 - sat((scene depth - fragment depth) / range))^4 -> the contact glow
        var soft = 0.0;
        if (i.xbump.x > 1e-4) {
            let px = vec2<i32>(i.clip.xy);
            let bed_ndc = textureLoad(scene_depth, px, 0);
            if (bed_ndc < 0.9999) {
                let dim = vec2<f32>(textureDimensions(scene_depth));
                let uv2 = i.clip.xy / dim;
                let ndc = vec3<f32>(uv2.x * 2.0 - 1.0, 1.0 - uv2.y * 2.0, bed_ndc);
                let bedH = cam.inv_view_proj * vec4<f32>(ndc, 1.0);
                let bed = bedH.xyz / bedH.w;
                let bed_d = cam.view_proj[0][3] * bed.x + cam.view_proj[1][3] * bed.y + cam.view_proj[2][3] * bed.z + cam.view_proj[3][3];
                let q = 1.0 - clamp((bed_d - i.vdepth) / i.xbump.x, 0.0, 1.0);
                soft = q * q * q * q;
            }
        }
        let na = select(ff_na_r, pow(ff_na_r, 2.2), i.xctl.x > 0.5);
        let nb = select(ff_nb_r, pow(ff_nb_r, 2.2), i.xctl.y > 0.5);
        let idx = clamp(sqrt(edge * edge + soft) * (1.0 - i.xbump.w * abs(na - nb)), 0.0, 1.0);
        // the palette is a LUT: the engine binds it clamped, so clamp to the texel centres
        let pdim = vec2<f32>(textureDimensions(detail_tex));
        let puv = clamp(vec2<f32>(idx * i.fine_xform.x + i.fine_xform.z, i.spec_rgb.w * i.fine_xform.y + i.fine_xform.w),
                        0.5 / pdim, 1.0 - 0.5 / pdim);
        let pal_r = textureSampleLevel(detail_tex, base_samp, puv, 0.0).r;
        let pal = select(pal_r, pow(pal_r, 2.2), i.xctl.z > 0.5);
        // self-illum: pow(|palette.r|, up163.y) * up160.rgb * change colour * up161.x * mask.a
        let si = pow(abs(pal), i.xbump.y) * i.si_ctl.rgb * i.itint.rgb * i.mat_spec2.x * ff_am;
        // overlay: overlay * overlay_detail * up160.rgb * up160.w * DETAIL_MULTIPLIER
        let ov_r = select(ff_ov_r, pow(ff_ov_r, vec3<f32>(2.2)), i.xctl.w > 0.5);
        let od_r = select(h4b_od_raw, pow(h4b_od_raw, vec3<f32>(2.2)), i.fres_rgb.x > 0.5);
        let ovl = ov_r * od_r * i.si_ctl.rgb * i.si_ctl.w * 4.59479;
        // output fade fresnel (up162; all-zero on the grid -> 1)
        let fade = 1.0 - clamp(i.spec_rgb.x * pow(clamp(mix(ndv, 1.0 - ndv, i.spec_rgb.z), 0.0, 1.0), i.spec_rgb.y), 0.0, 1.0);
        var col = (si + ovl) * fade * illum_scale_now();
        // HMS_MDBG probes: 80 [fres, soft, index], 81 [palette.r, mask.a, N.V], 82 self-illum rgb,
        // 83 overlay rgb.
        let hdbg4 = __MDBG__;
        if (hdbg4 > 79.5 && hdbg4 < 80.5) { return vec4<f32>(fres, soft, idx, 1.0); }
        if (hdbg4 > 80.5 && hdbg4 < 81.5) { return vec4<f32>(pal, ff_am, ndv, 1.0); }
        if (hdbg4 > 81.5 && hdbg4 < 82.5) { return vec4<f32>(si, 1.0); }
        if (hdbg4 > 82.5 && hdbg4 < 83.5) { return vec4<f32>(ovl, 1.0); }
        let insc4 = apply_fog(vec3<f32>(0.0), i.world_pos, cam.cam_pos.xyz);
        let ext4 = apply_fog(vec3<f32>(1.0), i.world_pos, cam.cam_pos.xyz) - insc4;
        return vec4<f32>(max(col, vec3<f32>(0.0)) * ext4, 1.0);
    }
    let ff_engine = is_ff && i.bump_xform.z > 0.5;
    // shield-wall force fields (one-way AND two-way doors) are TWO back-to-back
    // single-sided planes — one halogram plane facing +N, the other facing -N (for a one-way
    // door the +N plane is BLUE and the -N plane is RED; a two-way door has both planes blue).
    // The engine renders these single-sided (culls back faces), so from each side you only see
    // the NEAR plane. We reproduce that by discarding force-field fragments whose surface normal
    // faces AWAY from the camera (dot(n,v) < 0): blue side shows only blue, red side only red,
    // with neither colour bleeding through; two-way still reads blue from both sides. This uses
    // the decoded normal, so it is independent of triangle winding / front-face convention.
    if (is_ff && dot(n, v) < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (ff_engine) {
        // HMS_MDBG per-term probes for the halogram lane (read via HMS_HDR_DUMP, additive so
        // probe over a black/dark bed): 70 [noise_a, noise_b, |a-b|], 71 [mask alpha, depth fade, index],
        // 72 palette rgb, 73 overlay term rgb, 74 [ILLUM_SCALE, fog extinction, 0].
        let hdbg = __MDBG__;
        if (hdbg > 69.5 && hdbg < 74.5) {
            // (75 / 76 are the #forge-lights mode 3/4 probes further down)
            if (hdbg < 70.5) { return vec4<f32>(ff_na, ff_nb, abs(ff_na - ff_nb), 1.0); }
            if (hdbg < 71.5) { return vec4<f32>(ff_alpha, ff_dfa, ff_index, 1.0); }
            if (hdbg < 72.5) { return vec4<f32>(ff_pal, 1.0); }
            if (hdbg < 73.5) { return vec4<f32>(ff_ov * ff_od * 4.59479 * i.si_ctl.rgb * i.si_ctl.w, 1.0); }
        }
        // Engine entry_points.hlsl_include:417/444-462 for a material_model NONE additive
        // halogram: out = max(0, self_illum * ILLUM_SCALE) + overlay, then * fog extinction (additive
        // BLEND_FOG_INSCATTER_SCALE = 0, blend.hlsl_include:78) and * g_exposure (the tonemap here).
        // ILLUM_SCALE = g_alt_exposure.r applies to EVERY self-illum option (and is 1.0 wherever
        // the meter sits at gain 1, e.g. Forge World).
        var ff_col = ff_pal * i.mat_spec2.rgb * i.mat_spec2.w * illum_scale_now();
        // Halogram self_illum option 11 (palettized_plasma_change_color): self_illum_color is
        // the object's PRIMARY change colour (raw instance tint = forge team/colour, white when neutral).
        // Mode 2's mat_spec2 = [lo, hi, period, kind] = the self_illum_intensity periodic
        // record (period 0 = static lo): the spawn markers pulse 0.25 -> 1.0 on a 2 s wave, and the
        // snapshot at the trough let the fixed blue overlay out-glow the team colour.
        if (ff_mode > 1.5 && ff_mode < 2.5) {
            let cc_int = select(i.mat_spec2.x, mix(i.mat_spec2.x, i.mat_spec2.y, halo_periodic(i.mat_spec2.w, fract(t / max(i.mat_spec2.z, 1e-3)))), i.mat_spec2.z > 1e-3);
            ff_col = ff_pal * cc_int * illum_scale_now() * i.itint.rgb;
        }
        if (ff_mode > 3.5) {
            // Halogram PLASMA (self_illumination.hlsl_include calc_self_illumination_plasma_ps,
            // verified against the compiled _0_3_1_0_0_0_1 pixel shader): the two scrolling noise maps'
            // red channels, diff = max(0, 1 - |a - b|), three power bands (thinness in .w) with the medium
            // band minus the sharp one and the wide band minus the medium one, each premultiplied colour
            // (x alpha x self_illum_intensity, folded at decode) summed, x alpha_mask.a x ILLUM_SCALE.
            let diff = max(0.0, 1.0 - abs(ff_na - ff_nb));
            let sharp = pow(diff, max(i.bump_detail_xform.w, 0.0));
            var medium = pow(diff, max(i.mat_spec2.w, 0.0));
            var wide = pow(diff, max(i.env_ctl.w, 0.0));
            wide = wide - medium;
            medium = medium - sharp;
            ff_col = (i.mat_spec2.rgb * medium + i.bump_detail_xform.rgb * sharp + i.env_ctl.rgb * wide) * ff_alpha * illum_scale_now();
        } else if (ff_mode > 2.5) {
            // Halogram SIMPLE (calc_self_illumination_simple_ps): self_illum_map(uv * xform +
            // scroll) x self_illum_color x self_illum_intensity x ILLUM_SCALE. The colour / intensity may be
            // PERIODIC animated parameters (bump_detail_xform / env_ctl = [period, freq, phase, kind], kind 0 =
            // static): value(t) = mix(f=0 lanes (mat_spec2), f=1 lanes (detail_xform), g(frac(frac(t /
            // period) * freq + phase))) -- the flashing Forge lights blink red -> black, 8 -> 2 on a 4 s cosine.
            let fc = halo_periodic(i.bump_detail_xform.w, fract(fract(t / max(i.bump_detail_xform.x, 1e-3)) * i.bump_detail_xform.y + i.bump_detail_xform.z));
            let fi = halo_periodic(i.env_ctl.w, fract(fract(t / max(i.env_ctl.x, 1e-3)) * i.env_ctl.y + i.env_ctl.z));
            let si_col = mix(i.mat_spec2.rgb, i.detail_xform.xyz, fc);
            let si_int = mix(i.mat_spec2.w, i.detail_xform.w, fi);
            let si_map_r = textureSample(base_tex, base_samp, ff_uv_a).rgb;
            let si_map = select(si_map_r, pow(si_map_r, vec3<f32>(2.2)), i.xctl.z > 0.5);
            ff_col = si_map * si_col * si_int * illum_scale_now();
        }
        if (ff_mode > 4.5) {
            // Halogram FROM_DIFFUSE (self_illumination.hlsl_include calc_self_illumination_from_albedo_ps,
            // verified against the compiled _3_4_1_0_2_1_1 pixel shader = the armor-ability pickup's drop_holo
            // field): self_illum = saturate(base(uv) x detail(uv) x DETAIL_MULTIPLIER) x self_illum_color x
            // self_illum_intensity x ILLUM_SCALE; base_map rides base_tex (xctl.z gamma flag), detail_map emis_tex
            // (xctl.w). The intensity may be periodic (env_ctl record; f=0 -> mat_spec2.w, f=1 -> fine_xform.x):
            // the drop_holo pulses 0.25..1 x 1.47 on a 2 s cosine. (The two_change_color factor is the object's
            // primary/secondary change colour = white on a pickup, not carried.)
            let fi = halo_periodic(i.env_ctl.w, fract(fract(t / max(i.env_ctl.x, 1e-3)) * i.env_ctl.y + i.env_ctl.z));
            let si_int = mix(i.mat_spec2.w, i.fine_xform.x, fi);
            let base_r = textureSample(base_tex, base_samp, ff_uv_a).rgb;
            let base_l = select(base_r, pow(base_r, vec3<f32>(2.2)), i.xctl.z > 0.5);
            let det_r = textureSample(emis_tex, base_samp, ff_uv_b).rgb;
            let det_l = select(det_r, pow(det_r, vec3<f32>(2.2)), i.xctl.w > 0.5);
            let alb = clamp(base_l * det_l * 4.59479, vec3<f32>(0.0), vec3<f32>(1.0));
            ff_col = alb * i.mat_spec2.rgb * si_int * illum_scale_now();
        }
        if (ff_mode > 6.5) {
            // Halogram MULTILAYER_ADDITIVE (self_illumination_halogram.hlsl_include
            // calc_self_illumination_multilayer_ps): layers_of_4 x 4 samples of the self_illum_map, each
            // stepped by -view_dir_ts.xy * xform.xy * (texcoord_aspect_ratio, 1) * layer_depth / N and
            // weighted by depth_darken^k, averaged; out = pow(mean, layer_contrast) * self_illum_color *
            // self_illum_intensity * ILLUM_SCALE. Lanes: tile/mattex.xy/scroll = self_illum_map xform +
            // scroll, detail_xform = [layer_depth, layer_contrast, aspect, depth_darken], mat_spec2 =
            // [colour, intensity at f=0], xbump.x = intensity at f=1, bump_xform.xy = [period, freq] and
            // fine_xform.yz = [phase, kind] of the intensity's periodic record, fine_xform.x = layers_of_4.
            let fi = halo_periodic(i.fine_xform.z, fract(fract(t / max(i.bump_xform.x, 1e-3)) * i.bump_xform.y + i.fine_xform.y));
            let si_int = mix(i.mat_spec2.w, i.xbump.x, fi);
            let ml_lane = i32(i.fine_xform.x + 0.5);
            let n_layers = clamp(ml_lane % 8, 1, 4) * 4;
            let ml_clamp_u = (ml_lane & 8) != 0;   // authored sampler address mode = clamp
            let ml_clamp_v = (ml_lane & 16) != 0;
            let ml_dim = vec2<f32>(textureDimensions(base_tex));
            let ml_lo = 0.5 / ml_dim;
            let ml_hi = 1.0 - 0.5 / ml_dim;
            let bdet = ml_dux.x * ml_duy.y - ml_duy.x * ml_dux.y;
            var view_ts = vec2<f32>(0.0);
            if (abs(bdet) > 1e-12) {
                let r = 1.0 / bdet;
                let tng = normalize((ml_dpx * ml_duy.y - ml_dpy * ml_dux.y) * r);
                let bit = normalize((ml_dpy * ml_dux.x - ml_dpx * ml_duy.x) * r);
                view_ts = vec2<f32>(dot(v, tng), dot(v, bit));
            }
            let ml_off = view_ts * i.tile * vec2<f32>(i.detail_xform.z, 1.0) * i.detail_xform.x / f32(n_layers);
            var ml_uv = ff_uv_a;
            var ml_acc = vec3<f32>(0.0);
            var ml_w = 1.0;
            for (var k = 0; k < 16; k = k + 1) {
                if (k >= n_layers) { break; }
                let ml_uv_c = vec2<f32>(select(ml_uv.x, clamp(ml_uv.x, ml_lo.x, ml_hi.x), ml_clamp_u), select(ml_uv.y, clamp(ml_uv.y, ml_lo.y, ml_hi.y), ml_clamp_v));
                let s_r = textureSample(base_tex, base_samp, ml_uv_c).rgb;
                ml_acc = ml_acc + ml_w * select(s_r, pow(s_r, vec3<f32>(2.2)), i.xctl.z > 0.5);
                ml_uv = ml_uv - ml_off;
                ml_w = ml_w * i.detail_xform.w;
            }
            ml_acc = ml_acc / f32(n_layers);
            ff_col = pow(max(ml_acc, vec3<f32>(0.0)), vec3<f32>(max(i.detail_xform.y, 1e-3))) * i.mat_spec2.rgb * si_int * illum_scale_now();
        } else if (ff_mode > 5.5) {
            // self_illumination OFF (calc_self_illumination_none_ps): the material is its
            // overlay + edge fade only (material_model none -> no lit term; the constant albedo is unused).
            ff_col = vec3<f32>(0.0);
        }
        if (ff_mode > 5.5) {
            // Modes 6 / 7: APPLY_OVERLAYS order = overlay FIRST, then edge_fade_simple
            // (overlays.hlsl_include): out = (self_illum + overlay) * lerp(edge_tint, center_tint, |N.V|^power).
            // The overlay lanes are the palettized layout (fine_xform.w mode, si_ctl tint + intensity, bump_tex
            // overlay, mat_tex / od mode detail); the edge record rides spec_rgb / fres_rgb (fres_rgb.w = 1).
            var ovl6 = vec3<f32>(0.0);
            if (i.fine_xform.w > 1.5) {
                ovl6 = ff_ov * i.si_ctl.rgb * i.si_ctl.w;
            } else if (i.fine_xform.w > 0.5) {
                ovl6 = ff_ov * ff_od * 4.59479 * i.si_ctl.rgb * i.si_ctl.w;
            }
            // Overlay option 4 (calc_overlay_multiply_and_additive_detail_ps) first MULTIPLIES the
            // self-illum colour by overlay_multiply_map.rgb (bump_detail_tex at mattex_xform; xctl.w = 1 + gamma,
            // 0 = no map bound = white). Sword Base's coastal_subsurface is masked by `ice_floating` this way.
            if (i.xctl.w > 0.5) {
                let mul_r = textureSample(bump_detail_tex, base_samp, i.uv * i.mattex_xform.xy + i.mattex_xform.zw).rgb;
                ff_col = ff_col * select(mul_r, pow(mul_r, vec3<f32>(2.2)), i.xctl.w > 1.5);
            }
            var edge6 = vec3<f32>(1.0);
            if (i.fres_rgb.w > 0.5) {
                let e6 = pow(abs(dot(n, v)), max(i.spec_rgb.w, 0.0));
                edge6 = mix(i.spec_rgb.rgb, i.fres_rgb.rgb, e6);
            }
            if (hdbg > 74.5 && hdbg < 75.5) { return vec4<f32>(ff_col, 1.0); }
            let col6 = max(ff_col + ovl6, vec3<f32>(0.0)) * edge6;
            let insc6 = apply_fog(vec3<f32>(0.0), i.world_pos, cam.cam_pos.xyz);
            let ext6 = apply_fog(vec3<f32>(1.0), i.world_pos, cam.cam_pos.xyz) - insc6;
            return vec4<f32>(col6 * ext6, 1.0);
        }
        ff_col = max(ff_col, vec3<f32>(0.0));
        // edge_fade_simple (halogram category 6; RE'd from the compiled pixel shaders): the
        // additive output x lerp(edge_tint, center_tint, pow(|dot(N, V)|, power)). xbump.w flags it.
        // HMS_MDBG 75 [self-illum term before edge fade, rgb] (ILLUM_SCALE included), 76 [edge e, |N.V|, mode].
        var ff_edge = vec3<f32>(1.0);
        var ff_e = 1.0;
        if (ff_mode > 2.5 && i.xbump.w > 0.5) {
            ff_e = pow(abs(dot(n, v)), max(i.si_ctl.w, 0.0));
            ff_edge = mix(i.si_ctl.rgb, i.xbump.rgb, ff_e);
        }
        if (hdbg > 74.5 && hdbg < 75.5) { return vec4<f32>(ff_col, 1.0); }
        if (hdbg > 75.5 && hdbg < 76.5) { return vec4<f32>(ff_e, abs(dot(n, v)), ff_mode, 1.0); }
        // In the compiled _3_4_1_0_2_1_1 shader the overlay is ADDED before the edge-fade
        // multiply (out = (self_illum + overlay x tint x intensity) x edge); mode 5 keeps that order, its
        // overlay_tint x overlay_intensity riding the raw instance colour (si_ctl is the edge record there).
        if (ff_mode > 4.5 && i.fine_xform.w > 1.5) {
            ff_col = ff_col + ff_ov * i.itint.rgb;
        }
        ff_col = ff_col * ff_edge;
        if (ff_mode < 4.5 && i.fine_xform.w > 1.5) {
            ff_col = ff_col + ff_ov * i.si_ctl.rgb * i.si_ctl.w;                    // overlay additive
        } else if (ff_mode < 4.5 && i.fine_xform.w > 0.5) {
            ff_col = ff_col + ff_ov * ff_od * 4.59479 * i.si_ctl.rgb * i.si_ctl.w;   // DETAIL_MULTIPLIER
        }
        let ff_insc = apply_fog(vec3<f32>(0.0), i.world_pos, cam.cam_pos.xyz);
        let ff_ext = apply_fog(vec3<f32>(1.0), i.world_pos, cam.cam_pos.xyz) - ff_insc;
        if (hdbg > 73.5 && hdbg < 74.5) { return vec4<f32>(illum_scale_now(), ff_ext.g, 0.0, 1.0); }
        return vec4<f32>(ff_col * ff_ext, 1.0);
    }
    let rad = select(clamp(1.0 - length(i.uv - vec2<f32>(0.5, 0.5)) * 2.0, 0.0, 1.0), 1.0, is_ff || is_static);
    let puls = select(pulse, 0.85 + 0.15 * sin(t * 2.0 + i.world_pos.x * 0.5 + i.world_pos.y * 0.5), is_ff);
    var glow = (albedo + em) * (0.55 + 0.9 * edge) * cov * rad * puls;
    // STATIC icon: flat additive texture cut by its own alpha — no rim, no pulse, no drift.
    if (is_static) {
        glow = (albedo + em) * clamp(tex.a, 0.0, 1.0);
    }
    // LIGHT VOLUME (bump_xform.w == 4): the lvtl `albedo_circular` option — an untextured
    // circular cross-section. The per-profile colour × alpha × intensity is baked in the vertex colour
    // (i.tint); across the ribbon (uv.x 0..1) the chord of a unit circle sqrt(1-t²) is scaled by the
    // template's `center_offset` real constant (fine_xform.x) and raised to `falloff` (fine_xform.y).
    // (light_volume.fx albedo_circular, spec §3.4): per profile sprite delta = 2·uv − 1 on
    // BOTH axes, radius = saturate(center_offset·(1 − dx² − dy²)), alpha = radius^falloff (NO sqrt — the
    // old chord read 3-10× too bright at the rim = a hard tube). The ribbon spans the profile stack
    // along the axis, so the axial dy term is folded in as its mean over dy ∈ [−1,1] (8 samples), then
    // the 0.25 wu depth fade against the opaque scene depth (fine_xform.z = depth_fade_range).
    if (i.bump_xform.w > 3.5 && i.bump_xform.w < 4.5) {
        let tt = i.uv.x * 2.0 - 1.0;
        let co = i.fine_xform.x;
        let fo = max(i.fine_xform.y, 0.01);
        var acc = 0.0;
        for (var k = 0; k < 8; k = k + 1) {
            let dy = (f32(k) + 0.5) / 8.0;      // symmetric: sample the positive half
            acc = acc + pow(clamp(co * (1.0 - tt * tt - dy * dy), 0.0, 1.0), fo);
        }
        let rad = acc / 8.0;
        var dfa = 1.0;
        if (i.fine_xform.z > 1e-4) {
            let px = vec2<i32>(i.clip.xy);
            let bed_ndc = textureLoad(scene_depth, px, 0);
            if (bed_ndc < 0.9999) {
                let dim = vec2<f32>(textureDimensions(scene_depth));
                let uv2 = i.clip.xy / dim;
                let ndc = vec3<f32>(uv2.x * 2.0 - 1.0, 1.0 - uv2.y * 2.0, bed_ndc);
                let bedH = cam.inv_view_proj * vec4<f32>(ndc, 1.0);
                let bed = bedH.xyz / bedH.w;
                let bed_d = cam.view_proj[0][3] * bed.x + cam.view_proj[1][3] * bed.y + cam.view_proj[2][3] * bed.z + cam.view_proj[3][3];
                dfa = clamp((bed_d - i.vdepth) / i.fine_xform.z, 0.0, 1.0);
            }
        }
        return vec4<f32>(i.tint.rgb * rad * dfa * illum_scale_now(), 1.0);
    }
    return vec4<f32>(glow, 1.0);
}

// Engine particle_render.fx default_ps for the CPU-simulated effect particles (scene.rs
// build_particle_meshes). Per vertex: uv = sprite frame uv0, uv2 = second sprite uv (plasma / frame
// blend), tangent.xy = billboard uv, tangent.z = black point, tangent.w = palette v, tint = colour
// (rgb) × alpha (a). Per material: xctl = [albedo option, gamma(base), gamma(palette), gamma(base2)],
// bump_xform = [depth_fade_range, black_point on, palette_shift | alpha_modulation, 5], fine_xform =
// base_map_xform, bump_detail_xform = base_map2_xform, mat_spec2 = per-second overlay scroll of the two
// xform offsets, env_ctl = [blend enum, fog on, self-illum blend, depth_fade mode], si_ctl = [gain,
// has alpha_map, frame_blend, _]. Textures: base_tex = base_map, detail_tex = base_map2, emis_tex =
// palette, bump_tex = alpha_map.
fn particle_remap_alpha(black_point: f32, alpha: f32) -> f32 {
    let mid = (black_point + 1.0) * 0.5;
    return mid * clamp((alpha - black_point) / max(mid - black_point, 1e-4), 0.0, 1.0) + clamp(alpha - mid, 0.0, 1.0);
}
@fragment
fn fs_particle(i: VOut) -> @location(0) vec4<f32> {
    let t = cam.time.x;
    let albedo = i32(i.xctl.x + 0.5);
    let uv_bb = i.tangent.xy;
    let uv_s0 = i.uv;
    let uv_s1 = i.uv2;
    // depth fade (compute_depth_fade): saturate((scene_depth − particle_depth) / range), view depths
    var depth_fade = 1.0;
    if (i.env_ctl.w > 0.5 && i.bump_xform.x > 1e-5) {
        let px = vec2<i32>(i.clip.xy);
        let bed_ndc = textureLoad(scene_depth, px, 0);
        if (bed_ndc < 0.9999) {
            let dim = vec2<f32>(textureDimensions(scene_depth));
            let uv2 = i.clip.xy / dim;
            let ndc = vec3<f32>(uv2.x * 2.0 - 1.0, 1.0 - uv2.y * 2.0, bed_ndc);
            let bedH = cam.inv_view_proj * vec4<f32>(ndc, 1.0);
            let bed = bedH.xyz / bedH.w;
            let bed_d = cam.view_proj[0][3] * bed.x + cam.view_proj[1][3] * bed.y + cam.view_proj[2][3] * bed.z + cam.view_proj[3][3];
            depth_fade = clamp((bed_d - i.vdepth) / i.bump_xform.x, 0.0, 1.0);
        }
    }
    // samples (uniform control flow), gamma-decoded per the bitmap curve flags
    let base_r = textureSample(base_tex, base_samp, uv_s0);
    let base = vec4<f32>(select(base_r.rgb, pow(base_r.rgb, vec3<f32>(2.2)), i.xctl.y > 0.5), base_r.a);
    let xa = i.fine_xform;
    let xb = i.bump_detail_xform;
    let na = textureSample(base_tex, base_samp, uv_s0 * xa.xy + xa.zw + i.mat_spec2.xy * t).r;
    let nb = textureSample(detail_tex, base_samp, uv_s1 * xb.xy + xb.zw + i.mat_spec2.zw * t).r;
    let na_l = select(na, pow(na, 2.2), i.xctl.y > 0.5);
    let nb_l = select(nb, pow(nb, 2.2), i.xctl.w > 0.5);
    let amap_bb = textureSample(bump_tex, base_samp, uv_bb).a;
    let amap_sp = textureSample(bump_tex, base_samp, uv_s0).a;
    let palette_v = i.tangent.w;
    let particle_alpha = i.tint.a;
    var index = 0.0;
    var pal_v = palette_v;
    if (albedo == 8 || albedo == 9) {
        index = abs(na_l - nb_l);
        if (albedo == 8) { index = clamp(index + (1.0 - amap_bb * particle_alpha * depth_fade) * i.bump_xform.z, 0.0, 1.0); }
        else { pal_v = depth_fade; if (i.env_ctl.w > 2.5) { index = clamp(index + (1.0 - amap_bb * particle_alpha) * i.bump_xform.z, 0.0, 1.0); } }
    } else {
        index = base.r;
        if (i.env_ctl.w > 2.5) { index = clamp(index + (1.0 - depth_fade * particle_alpha) * i.bump_xform.z, 0.0, 1.0); }
    }
    let pal_r = textureSample(emis_tex, base_samp, vec2<f32>(index, pal_v)).rgb;
    let pal = select(pal_r, pow(pal_r, vec3<f32>(2.2)), i.xctl.z > 0.5);
    var tex = base;
    if (albedo == 1) { tex = vec4<f32>(base.rgb, amap_bb); }
    else if (albedo == 4) { tex = vec4<f32>(base.rgb, amap_sp); }
    else if (albedo == 2) { tex = vec4<f32>(pal, base.a); }
    else if (albedo == 3) { tex = vec4<f32>(pal, amap_bb); }
    else if (albedo == 5) { tex = vec4<f32>(pal, amap_sp); }
    else if (albedo == 7) { let glow = palette_v * base.g; tex = vec4<f32>(clamp(i.tint.rgb * glow * exp2(glow * 6.0), vec3<f32>(0.0), vec3<f32>(1.0)), base.g); }
    else if (albedo == 8 || albedo == 9) { tex = vec4<f32>(pal, amap_bb); }
    var a = tex.a * depth_fade;
    if (i.bump_xform.y > 0.5) { a = particle_remap_alpha(i.tangent.z, a); }
    a = clamp(a * particle_alpha, 0.0, 1.0);
    var rgb = tex.rgb * i.tint.rgb * i.si_ctl.x;
    // exposure: self-illum blends (additive / add_src_times_srcalpha) carry V_ILLUM_EXPOSURE; the rest
    // ride the scene exposure applied at tonemap like every lit surface.
    let illum = illum_scale_now();
    if (i.env_ctl.z > 0.5) { rgb = rgb * illum; }
    // fog (template digit): extinction on the colour; inscatter (colour_add) only for non-additive blends
    var color_add = vec3<f32>(0.0);
    if (i.env_ctl.y > 0.5) {
        let insc = apply_fog(vec3<f32>(0.0), i.world_pos, cam.cam_pos.xyz);
        let ext = apply_fog(vec3<f32>(1.0), i.world_pos, cam.cam_pos.xyz) - insc;
        rgb = rgb * ext;
        if (i.env_ctl.z < 0.5 && i.env_ctl.x > 1.5) { color_add = color_add + insc; }
    }
    // self_illumination constant_color: added × V_ILLUM_EXPOSURE (xbump = colour, w = present)
    if (i.xbump.w > 0.5) { color_add = color_add + i.xbump.rgb * illum * i.si_ctl.x; }
    rgb = rgb + color_add;
    let blend = i32(i.env_ctl.x + 0.5);
    // HMS_PARTICLE_DEBUG=1: solid magenta quads (coverage check); =2: raw texture rgba; =3: alpha as grey
    if (i.si_ctl.w > 2.5) { return vec4<f32>(a, a, a, 1.0); }
    if (i.si_ctl.w > 1.5) { return vec4<f32>(tex.rgb, tex.a); }
    if (i.si_ctl.w > 0.5) { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }
    if (blend == 2 || blend == 4) {
        // multiply: lerp(1, colour, alpha) under Zero/SrcColor
        return vec4<f32>(mix(vec3<f32>(1.0), rgb, a), 1.0);
    }
    if (blend == 5) {
        return vec4<f32>(rgb * a, a);           // pre_multiplied_alpha (One/InvSrcAlpha)
    }
    if (blend == 3 || blend == 10 || blend == 0) {
        return vec4<f32>(rgb, a);               // alpha_blend (SrcAlpha/InvSrcAlpha)
    }
    return vec4<f32>(rgb * a, 1.0);             // additive family: dst += rgb·alpha (One/One)
}

// SOLID hologram (shader_halogram look) for OBJECT markers — spawn Spartans, hill/objective
// globes, kill/safe boundary shells. Alpha-blended (SrcAlpha/OneMinusSrcAlpha) so the shape
// is ALWAYS readable: a translucent tinted body with a bright fresnel rim, plus a scanline
// shimmer for the holo feel. The additive fs_holo washed out against bright sky/snow and the
// markers vanished — this keeps a guaranteed silhouette (alpha floor on covered texels).
@fragment
fn fs_holo_solid(i: VOut) -> @location(0) vec4<f32> {
    let t = cam.time.x;
    let auv = i.uv * i.tile + i.scroll * t;
    let tex = textureSample(base_tex, base_samp, auv);
    // Use the RAW instance tint (itint), NOT i.tint: i.tint folds in the per-vertex airprobe
    // ambient, which corrupts the team/forced marker hue (green read as purple, etc.).
    let mt = i.itint.rgb;
    let albedo = pow(tex.rgb, vec3<f32>(2.2)) * mt;
    // Emissive carries the marker's image (globe glyph, spawn glow); tint by team/forced colour.
    let em = pow(textureSample(emis_tex, base_samp, auv).rgb, vec3<f32>(2.2)) * mt;
    let n = normalize(i.normal);
    let v = normalize(cam.cam_pos.xyz - i.world_pos);
    let edge = pow(clamp(1.0 - abs(dot(n, v)), 0.0, 1.0), 1.5);       // fresnel rim
    let pulse = 0.85 + 0.15 * sin(t * 2.0 + i.world_pos.x * 0.5 + i.world_pos.y * 0.5);
    let scan = 0.9 + 0.1 * sin(i.world_pos.z * 14.0 - t * 4.0);       // holo scanline shimmer
    // Coverage: hollow/masked texels (a≈0, black) stay transparent; solid body keeps a floor.
    let lum = dot(albedo + em, vec3<f32>(0.333));
    let cov = clamp(max(tex.a, lum * 1.5), 0.0, 1.0);
    // Body colour is TINT-DOMINANT so a saturated marker colour (kill=red, safe=green, hill=blue,
    // team spawn) reads strongly even over a dark/subtle base texture — the fill is mostly the
    // tint, with the texture/emissive adding detail and the fresnel rim glowing brightly.
    let body = mt * (0.60 + 0.95 * edge) + albedo * 0.35 + em;
    let col = body * pulse * scan;
    // Alpha: always-visible silhouette — floor 0.40 on covered texels, up to ~0.95 at the rim.
    let a = cov * clamp(0.40 + 0.55 * edge, 0.0, 0.95);
    return vec4<f32>(col, a);
}

// Atmospheric fog for decals. The engine's decal pixel shader (decal.hlsl_include)
// has NO fog code of its own: pre-lighting decals are written into the ALBEDO G-buffer, so the
// surface's own lighting pass fogs the decalled pixel exactly like the bare surface
// (`shaded * extinction + inscatter`). HMS draws decals AFTER the fogged surface, so the decal's
// own output must carry the same fog: alpha-blended colour -> rgb*ext + ins (blending then yields
// lerp(surface, decal, a) fogged as one), additive -> rgb*ext (adds nothing at full fog),
// multiply / double_multiply -> the modulator is pulled toward its neutral (1 / 0.5) by the
// extinction (a fully fogged pixel is inscatter only, which the decal cannot darken).
fn decal_fog_ext(wp: vec3<f32>) -> vec3<f32> {
    let ins = apply_fog(vec3<f32>(0.0), wp, cam.cam_pos.xyz);
    return apply_fog(vec3<f32>(1.0), wp, cam.cam_pos.xyz) - ins;
}
fn decal_fog_ins(wp: vec3<f32>) -> vec3<f32> {
    return apply_fog(vec3<f32>(0.0), wp, cam.cam_pos.xyz);
}

// UNLIT decal (engine decal.hlsl_include): albedo × tint, mask = sampled alpha.
// NO lighting, NO specular. Fogged like the receiver (#decal-fog, see decal_fog_ext).
// Blend mode (alpha/additive/multiply) is selected by the pipeline, not the shader.
@fragment
fn fs_decal(i: VOut) -> @location(0) vec4<f32> {
    // Engine clips UVs outside [0,1] (sprite-atlas sub-rects handled CPU-side).
    if (i.uv.x < 0.0 || i.uv.x > 1.0 || i.uv.y < 0.0 || i.uv.y > 1.0) { discard; }
    let tex = textureSample(base_tex, base_samp, i.uv);
    // COVERAGE. RE (effects\decal.hlsl_include): the engine hard-CLIPS UVs with NO per-pixel
    // fade. The soft ORGANIC footprint of a `diffuse_plus_alpha` region decal (snow/scorch) comes
    // from a SEPARATE alpha_map (Textures[1]) — e.g. snow = base ca_snow_01 (a flat opaque WHITE
    // tile) + alpha mask dirt_pile_swept (the irregular shape). `float4(base.rgb, alpha_map.a)`.
    // When tile.x==1 the mask is bound to emis_tex → take coverage from ITS alpha, which is why the
    // snow is an organic patch and NOT a square (the base tile's own alpha is uniform ~0.8).
    var cov = tex.a;
    if (i.tile.x > 0.5) {
        cov = textureSample(emis_tex, base_samp, i.uv).a;
    }
    if (cov < 0.004) { discard; }   // ~1/255 alpha-test on the coverage
    let rgb = pow(tex.rgb, vec3<f32>(2.2)) * i.tint.rgb;   // linearize + per-decal tint
    // Additive pipelines share this entry (bucket 1): they must not add inscatter.
    // tile.y carries 1 for additive decals (set by build_decals), else 0.
    let ins = decal_fog_ins(i.world_pos) * select(1.0, 0.0, i.tile.y > 0.5);
    return vec4<f32>(rgb * decal_fog_ext(i.world_pos) + ins, clamp(cov, 0.0, 1.0) * i.tint.a);
}

// SDF "vector" decal (numbers/text/glyphs — albedo_vector_alpha template). The
// base_map (base_tex) is a flat colour swatch; the glyph SILHOUETTE is a signed-distance
// field carried in the vector_map (bound to the emissive slot) .g channel. We threshold it
// at 0.5 with an fwidth-based antialias width so edges stay crisp at any zoom. Colour comes
// from the flat swatch × the per-decal tint. Without this the swatch fills the whole quad
// → the "plain rectangle where a number should be" symptom.
@fragment
fn fs_decal_vector(i: VOut) -> @location(0) vec4<f32> {
    if (i.uv.x < 0.0 || i.uv.x > 1.0 || i.uv.y < 0.0 || i.uv.y > 1.0) { discard; }
    let d = textureSample(emis_tex, base_samp, i.uv).g;   // SDF distance in vector_map.g
    let w = max(fwidth(d), 1e-4);
    let mask = smoothstep(0.5 - w, 0.5 + w, d);
    if (mask < 0.01) { discard; }
    let base = textureSample(base_tex, base_samp, i.uv);
    let rgb = pow(base.rgb, vec3<f32>(2.2)) * i.tint.rgb;
    return vec4<f32>(rgb * decal_fog_ext(i.world_pos) + decal_fog_ins(i.world_pos), mask * i.tint.a);
}

// PRE_MULTIPLIED_ALPHA decal (decs blend enum 10, and the diffuse_plus_alpha family
// the engine reroutes there). render_target premultiplied: rgb is pre-multiplied by
// the mask in the PS, blend state is One/OneMinusSrcAlpha. Fed the premul pipeline.
// This is the fix for the "washed out / not blending" 1B1B-family decals (regression
// A1): plain alpha-blend under-darkened by a factor of alpha everywhere alpha<1.
@fragment
fn fs_decal_premul(i: VOut) -> @location(0) vec4<f32> {
    if (i.uv.x < 0.0 || i.uv.x > 1.0 || i.uv.y < 0.0 || i.uv.y > 1.0) { discard; }
    let tex = textureSample(base_tex, base_samp, i.uv);
    let a = clamp(tex.a, 0.0, 1.0) * i.tint.a;
    if (a < 0.004) { discard; }
    let rgb = pow(tex.rgb, vec3<f32>(2.2)) * i.tint.rgb;
    let fogged = rgb * decal_fog_ext(i.world_pos) + decal_fog_ins(i.world_pos);
    return vec4<f32>(fogged * a, a);   // premultiply — blend is One/(1-SrcAlpha)
}

// MULTIPLY decal (decs blend enum 2/4). The engine emits (base.rgb, base.a) and the
// Multiply blend op (Zero/SrcColor) modulates dst by src.rgb — src.a is IGNORED. So
// the standard fs_decal alpha-test (a<0.004 discard) WRONGLY kills BC4-only scorch/AO
// bitmaps whose alpha decodes to ~0 everywhere (the "renders transparent when it
// shouldn't be" bug). Skip the alpha-test, output the sRGB rgb as a straight
// multiplier (NOT linearized — the modulator is applied in the target's own space),
// alpha=1 so the blend op has a defined src.a.
@fragment
fn fs_decal_multiply(i: VOut) -> @location(0) vec4<f32> {
    if (i.uv.x < 0.0 || i.uv.x > 1.0 || i.uv.y < 0.0 || i.uv.y > 1.0) { discard; }
    let tex = textureSample(base_tex, base_samp, i.uv);
    // Decal blend: HMS composites into a LINEAR Rgba16Float target, but the engine's
    // multiply decals darken a gamma-2.0 LDR framebuffer, so their effective result is
    // linear_dst · tex_srgb². Emitting the raw sRGB texel here (linear_dst · tex_srgb) leaves
    // grunge/scorch/scuff decals ~2× too light — they read as faint overlays (Prisoner). SQUARE
    // the multiplier to reproduce the gamma-2 darkening.
    let m = tex.rgb * tex.rgb;
    let rgb = clamp(m * i.tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    // Pull the modulator toward its neutral by the extinction. tile.y = 1 marks a
    // double_multiply decal (2*src*dst: neutral 0.5), else plain multiply (neutral 1).
    let neutral = select(1.0, 0.5, i.tile.y > 0.5);
    return vec4<f32>(mix(vec3<f32>(neutral), rgb, decal_fog_ext(i.world_pos)), 1.0);
}

// Decorators (grass / flowers). RE (docs: decorators.hlsl_include:97-120,347-372 +
// C# DecoratorFoliageTechnique.cs): the engine decorator shader is PURELY
// MULTIPLICATIVE — albedo × (baked per-instance ambient + sun) — with NO specular,
// NO flat-white sun term, and NO sky-dome hemisphere fill. Routing decorators
// through mesh_shade (which adds all three) is what turned the flowers into vivid
// confetti. i.tint.rgb already carries the toned ABSOLUTE airprobe ambient
// (build_one_decorator vcolor × identity tint). Sun is front-gated, wrap-softened,
// and scaled by the engine /pi Lambert coefficient (0.270) — not a flat 0.85.
@fragment
fn fs_foliage(i: VOut) -> @location(0) vec4<f32> {
    let s = textureSample(base_tex, base_samp, i.uv);
    if (s.a < 0.33) { discard; }   // engine k_decorator_alpha_test_threshold ~0.33
    let albedo = pow(s.rgb, vec3<f32>(2.2));
    let n = normalize(i.normal);
    // Decorator cards are single-sided quads — flip the normal to face the camera.
    let to_cam = cam.cam_pos.xyz - i.world_pos;
    let vn = select(-n, n, dot(n, to_cam) >= 0.0);
    // The engine decorator/foliage PS is PURE albedo × baked per-vertex light (no PS sun, no
    // real-time darken, no occlusion multiply — protomorph entry_decorator/entry_foliage;
    // Sapien DXBC ps_13465015724195547627 + decorators.hlsl_include).
    var fol_amb = i.tint.rgb;
    if (gi_on()) { fol_amb = gi_irradiance(i.world_pos, vn, normalize(to_cam)); }
    var rgb = albedo * fol_amb;
    // Engine per-pixel cosine lobe: `saturate(N·sun)·contrast.y + contrast.x` (k_ps_decorators_contrast
    // — an engine global whose value is not recovered from the capture; default 0.5/0.5 via
    // HMS_DECO_CONTRAST="x,y"). tint.a = 2+cos flags decorator vertices.
    if (i.tint.a > 1.5) {
        let cosl = saturate(i.tint.a - 2.0);
        rgb = rgb * (cosl * __DECO_CY__ + __DECO_CX__);
    }
    rgb = apply_fog(scene_grade(rgb), i.world_pos, cam.cam_pos.xyz); // per-domain grade (lit only)
    return vec4<f32>(rgb, 1.0);
}
"#;

/// shader_water surfaces. Port of the engine water shading intent (not a 1:1 of
/// water_shading.hlsl, which needs foam/refraction buffers): animated dual-scroll
/// wave normal perturbation, the watercolor bitmap as base colour, Fresnel-driven
/// transparency (more see-through looking straight down, more reflective at
/// grazing angles), and a sun specular glint. `cam.time.x` drives the animation.
/// Halo 4 material lane appended to the mesh shader module (see the header comment).
pub(crate) const H4_MESH_WGSL: &str = r#"
// ===================================================================================================
// HALO 4 MATERIAL LANE. Reached from mesh_shade through the `matmodel < -0.5` sentinel
// (Reach passes material_model + 1 >= 0). Everything below is the shipped srf_* pixel-shader law
// (crates/hms-app/src/h4/shading.rs module doc, docs/halo4_lighting_model.md): the deferred
// G-buffer pass (entry 01: albedo, normal, spec mask) fused with the static-lighting pass
// (entries 04/06: two SH-L1 lightmap lobes + the floating sun, Blinn or Phong specular from the
// same three lights, cube reflection with an artist fresnel, self-illum).
// Instance lanes (h4/scene.rs `h4_lanes`):
//   matmodel   -(1 + model): 0 none, 1 Blinn (N.H)^(1/r), 2 Phong (R.L)^p, 3 Phong (R.L)^(1/r)
//   xctl       [spec_src, srgb_bits, flag_bits, lay]   lay 0 std, 1 snow (2-layer), 2 layered3,
//              3 srf_char_cov* (the Covenant body ramp, #h4-veh)
//   spec2      [rough_min | power, rough_max, spec_tint_by_albedo, spec_intensity]
//   spec_rgb   [spec colour, spec_mask_alpha_weight]      env_ctl  [refl tint, refl intensity]
//   fres_rgb   [fresnel scale, power, weight, invert]     mattex   [env_lit_by_diffuse, refl_normal_blend, diffuse_intensity, nd_strength]
//   aux        [0, si_mode, nd_fade_end, nd_fade_start]   si_ctl   [si colour, si intensity]
//   xbump      diffspec [desat, pow, scale, bias] | snow [cov scale, cov power, bias, alpha weight]
//   obj_probe0 std [albedo tint, detail_spec_weight] | snow [color1 tint, normal blend] | layered [layer1 tint, height weight 1]
//   obj_probe1 std [the object's PRIMARY change colour rgb, mask weight = ps_material_object_parameters[0]]
//              | snow [spec1 colour, spec1 intensity] | layered [layer2 tint, height weight 2]
//   lay 3 (srf_char_cov*): obj_probe1 = up161 (face-on ramp colour + exponent), fres_rgb = up162
//              (mid-angle), xbump = up163 (grazing + the reflection fresnel power), env_ctl =
//              [up166.rgb, 1], uv_density = up166.w (reflection saturation), at = spec_detail_map
//   env_ctl    snow / layered: [albedo tint, 0];  si_ctl snow: [layer dir, spec1 power];  mattex.z snow: detail_spec_weight
//   scroll.zw  base tile;  bump_xform normal;  bump_detail_xform normal detail (| layer1 nm);
//   detail_xform color detail (| all-layers detail);  fine_xform spec/control (| snow normal1 | layer2 co);
//   mattex_xform self-illum map (| snow color1 | layer1 co);  si_ctl layered: layer2 nm xform
// Textures: base = color (| layer0 co), bump = normal (| layer0 nm), bump_detail = normal detail
//   (| layer1 nm), detail = color detail (| all-layers detail), mat = specular/control (| blend map),
//   emis = self-illum map (| snow color1 | layer1 co), at = snow normal1 | layer2 co,
//   bump_detail2 = specular_map when a control map ALSO exists (spec_src 5), bump_detail3 = layer2 nm,
//   env_cube = the material's reflection_map cube (linear floats).
const H4_DETAIL_MULT: f32 = 4.59479;   // 2^2.2: the linear-format detail maps are stored mid-grey = 0.5^2.2
const H4_INV_PI: f32 = 0.31830989;
const H4DBG: f32 = __H4DBG__;

struct H4Lobes { col_a: vec3<f32>, col_b: vec3<f32>, dir_a: vec3<f32>, dir_b: vec3<f32>, w_a: f32, w_b: f32, vis: f32, has_dir: f32, vis_raw: f32, ana_y: f32 };

// The two lightmap lobes of a texel (lightmaps.rs `compose_atlas_textures` lay), colours WITHOUT
// the sharpen LUT (the spec lanes use them raw); vis = the sharpened floating-sun visibility.
fn h4_lobes(uv2: vec2<f32>, k_a: f32, k_b: f32) -> H4Lobes {
    let t0 = textureSampleLevel(lm_dm_tex,   lm_samp, uv2, 0.0);
    let t1 = textureSampleLevel(lm_sdm_tex,  lm_samp, uv2, 0.0);
    let t2 = textureSampleLevel(lm_sdm1_tex, lm_samp, uv2, 0.0);
    let t3 = textureSampleLevel(lm_sdm2_tex, lm_samp, uv2, 0.0);
    var o: H4Lobes;
    o.dir_a = t2.xyz * 2.0 - 1.0;
    o.dir_b = t3.xyz * 2.0 - 1.0;
    o.w_a = sqrt(max(1.0 - dot(o.dir_a, o.dir_a), 0.0));
    o.w_b = sqrt(max(1.0 - dot(o.dir_b, o.dir_b), 0.0));
    let ia = k_a * (512.0 * exp2(-9.0 * t0.w) - 1.0) / 511.0;
    let ib = abs(k_b) * (512.0 * exp2(-9.0 * t1.w) - 1.0) / 511.0;
    o.col_a = max(t0.rgb * ia, vec3<f32>(0.0));
    o.col_b = max(t1.rgb * ib, vec3<f32>(0.0));
    o.vis = select(0.0, clamp(2.0 * t2.w - 0.5, 0.0, 1.0), k_b > 0.0);
    o.vis_raw = select(0.0, t2.w, k_b > 0.0);   // the un-sharpened analytic texel
    // analytic.y = the sun's extra multiplier (entry 06 `mul r0.w, r9.y, r0.w`, applied
    // after the sharpening and the cascade lerp); it rides sdm2.w outside the 64x64 LUT corner
    let dims = vec2<f32>(textureDimensions(lm_sdm2_tex, 0));
    let in_corner = uv2.x * dims.x < 64.0 && uv2.y * dims.y < 64.0;
    o.ana_y = select(t3.w, 1.0, in_corner);
    o.has_dir = 1.0;
    return o;
}

// SH-L1 irradiance of one lobe at n, through the engine sharpen LUT (u = pixel normal, v = vertex normal).
fn h4_lobe_diffuse(d: vec3<f32>, w: f32, n: vec3<f32>, nv: vec3<f32>) -> f32 {
    let u = 0.2820948 * w + 0.325735 * dot(d, n);
    let v = 0.2820948 * w + 0.325735 * dot(d, nv);
    return h4_sharpen_lut(u, v);
}

// One specular lobe: Blinn (N.H)^p / pi or Phong (R.L)^p / pi; `p` is the final exponent.
fn h4_spec_lobe(model: i32, n: vec3<f32>, e: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    if (model == 1) {
        let h = normalize(l + e);
        return pow(max(dot(n, h), 1e-6), p) * H4_INV_PI;
    }
    if (model >= 2) {
        let r = reflect(-e, n);
        return pow(max(dot(r, l), 1e-6), p) * H4_INV_PI;
    }
    return 0.0;
}

// `// #h4-expo-2` The engine's `*_SRGB` DXGI formats are decoded by the SAMPLER with the exact
// sRGB EOTF (the piecewise 1/12.92 toe + ((v + 0.055) / 1.055)^2.4), not with pow(v, 2.2): a
// plain 2.2 is up to 1.6x too DARK below v = 0.15 and 2 % too bright around v = 0.6.
fn h4_srgb(c: vec3<f32>, on: bool) -> vec3<f32> {
    let v = max(c, vec3<f32>(0.0));
    let lin = select(pow((v + 0.055) / 1.055, vec3<f32>(2.4)), v / 12.92, v <= vec3<f32>(0.04045));
    return select(v, lin, on);
}

// shared-exponent RGB9E5 decode (hms-app h4/scene.rs `pack_rgb9e5`): m * 2^(e - 24).
fn h4_unpack_rgb9e5(v: u32) -> vec3<f32> {
    let e = i32(v >> 27u) - 24;
    let s = exp2(f32(e));
    return vec3<f32>(f32(v & 511u), f32((v >> 9u) & 511u), f32((v >> 18u) & 511u)) * s;
}

fn h4_tbn_normal(gn: vec3<f32>, tangent: vec4<f32>, ts: vec2<f32>, dpx: vec3<f32>, dpy: vec3<f32>, dux: vec2<f32>, duy: vec2<f32>) -> vec3<f32> {
    let tz = sqrt(max(1.0 - dot(ts, ts), 0.0));
    let t_len = length(tangent.xyz);
    var tng: vec3<f32>;
    var bit: vec3<f32>;
    if (t_len > 1e-4) {
        tng = normalize(tangent.xyz - gn * dot(gn, tangent.xyz));
        bit = normalize(cross(gn, tng) * tangent.w);
    } else {
        let bdet = dux.x * duy.y - duy.x * dux.y;
        if (abs(bdet) < 1e-10) { return gn; }
        let r = 1.0 / bdet;
        let t = (dpx * duy.y - dpy * dux.y) * r;
        tng = normalize(t - gn * dot(gn, t));
        bit = normalize(cross(gn, tng));
    }
    return normalize(tng * ts.x + bit * ts.y + gn * tz);
}

fn h4_shade(i: VOut) -> vec4<f32> {
    let model = i32(-i.matmodel - 0.5);       // 0 none, 1 blinn, 2 phong power, 3 phong rough
    let spec_src = i32(i.xctl.x + 0.5);
    let srgb = u32(i.xctl.y + 0.5);
    let flags = u32(i.xctl.z + 0.5);
    let lay = i32(i.xctl.w + 0.5);
    let has_normal = (flags & 1u) != 0u;
    let has_nd = (flags & 2u) != 0u;
    let has_cdetail = (flags & 4u) != 0u;
    let has_cube = (flags & 8u) != 0u;
    let has_spectex = (flags & 16u) != 0u;
    let has_simap = (flags & 32u) != 0u;
    let has_layer = (flags & 64u) != 0u;
    // `at` slot bound: pcc_amount_map (lay 0) or spec_detail_map (lay 3, srf_char_cov_specdetail)
    let has_pcc = (flags & 128u) != 0u;
    let is_cov = lay == 3;                // #h4-veh srf_char_cov*: the Covenant body ramp
    let gn = normalize(i.normal);
    let cam_pos = cam.cam_pos.xyz;
    let to_cam = cam_pos - i.world_pos;
    let dist = length(to_cam);
    let e = to_cam / max(dist, 1e-4);
    let uv = i.uv;
    let uv_base = uv * i.tile;
    // derivatives for the fallback tangent frame (uniform control flow)
    let dpx = dpdx(i.world_pos);
    let dpy = dpdy(i.world_pos);
    let dux = dpdx(uv_base);
    let duy = dpdy(uv_base);
    let uv_n = uv * i.bump_xform.xy + i.bump_xform.zw;
    let uv_nd = uv * i.bump_detail_xform.xy + i.bump_detail_xform.zw;
    let uv_cd = uv * i.detail_xform.xy + i.detail_xform.zw;
    let uv_sp = uv * i.fine_xform.xy + i.fine_xform.zw;
    let uv_si = uv * i.mattex_xform.xy + i.mattex_xform.zw;

    // ---- G-buffer pass: albedo, tangent normal, spec mask ------------------------------------
    let base = textureSample(base_tex, base_samp, uv_base);
    let albedo_tint = select(i.obj_probe0.xyz, i.env_ctl.xyz, lay == 1 || lay == 2);
    var base_lin = h4_srgb(base.rgb, (srgb & 1u) != 0u);
    // *_colorchangemap (entry 01 asm): albedo = lerp(c, luma709(c) * cc.rgb, pcc.r * cc.w) * tint
    // with the engine's neutral primary change colour cc = (0.5, 0.5, 0.5, 1) (no team on the Forge
    // pieces; PROVISIONAL: the per-object team colour is not carried)
    let pcc_s = textureSample(at_tex, base_samp, uv_si).x;
    if (lay == 0 && has_pcc) {
        // cc = ps_material_object_parameters[0], the object's PRIMARY change colour (the engine's
        // neutral is (0.5, 0.5, 0.5, 1)); .w is the per-object MASK WEIGHT, not an opacity. #h4-veh
        let cc = i.obj_probe1;
        let l709 = dot(base_lin, vec3<f32>(0.2125, 0.7154, 0.0721));
        base_lin = mix(base_lin, vec3<f32>(l709) * cc.rgb, clamp(pcc_s * cc.w, 0.0, 1.0));
    }
    var albedo = base_lin * albedo_tint;
    var color_a = base.a;
    var spec_mask = mix(1.0, color_a, i.spec_rgb.w);
    let nsamp = textureSample(bump_tex, base_samp, uv_n).xy;            // BC5_SNORM: xy raw
    var ts = select(vec2<f32>(0.0), nsamp, has_normal);
    let ndsamp = textureSample(bump_detail_tex, base_samp, uv_nd).xy;
    if (has_nd) {
        // normal detail faded by distance: full below fade_start, gone past fade_end
        let fade = 1.0 - clamp((dist - i.mat_rough) / max(i.mat_spec - i.mat_rough, 1e-4), 0.0, 1.0);
        let combined = ts + fade * ndsamp;
        ts = mix(ts, combined, i.mattex.w);
    }
    let cdet = textureSample(detail_tex, base_samp, uv_cd);
    let detail_spec_weight = select(i.obj_probe0.w, i.mattex.z, lay == 1 || lay == 2);
    if (has_cdetail && lay != 2) {
        albedo = albedo * h4_srgb(cdet.rgb, (srgb & 2u) != 0u) * H4_DETAIL_MULT;
        spec_mask = spec_mask * mix(1.0, cdet.a, detail_spec_weight);
    }
    // specular / control map
    let sp = textureSample(mat_tex, base_samp, uv_sp);
    let sp_lin = h4_srgb(sp.rgb, (srgb & 4u) != 0u);
    var spec_col = vec3<f32>(1.0);
    var gloss = 1.0;
    var refl_mask = 1.0;
    var si_mask = 0.0;
    if (spec_src == 1) { spec_col = sp_lin; gloss = sp.a; }
    else if (spec_src == 2) { spec_col = vec3<f32>(sp.r); gloss = sp.g; refl_mask = sp.b; }
    else if (spec_src == 3) { spec_col = vec3<f32>(sp.r); gloss = sp.g; si_mask = sp.b; }
    else if (spec_src == 4) {
        // diffspec: the spec colour is the (linear) colour map, desaturated, powered, scaled
        let c = h4_srgb(base.rgb, (srgb & 1u) != 0u);
        let lum = dot(c, vec3<f32>(0.299, 0.587, 0.114));
        spec_col = clamp(pow(max(mix(c, vec3<f32>(lum), i.xbump.x), vec3<f32>(1e-5)), vec3<f32>(i.xbump.y)) * i.xbump.z + i.xbump.w, vec3<f32>(0.0), vec3<f32>(1.0));
        gloss = sp.g; refl_mask = sp.b;
        if (spec_src == 4 && !has_spectex) { gloss = 1.0; refl_mask = 1.0; }
        if (has_spectex) { spec_col = spec_col * sp.r; }
    }
    else if (spec_src == 5) {
        // specular_map rgb (bump_detail2 slot) x control SpGlRf
        let sm = textureSample(bump_detail2_tex, base_samp, uv_sp);
        spec_col = h4_srgb(sm.rgb, (srgb & 4u) != 0u) * sp.r; gloss = sp.g; refl_mask = sp.b;
    }
    else if (spec_src == 6) { spec_col = vec3<f32>(sp.r); gloss = sp.g; refl_mask = sp.b; si_mask = sp.a; }
    if (!has_spectex && spec_src != 4) { spec_col = vec3<f32>(1.0); gloss = 1.0; refl_mask = 1.0; si_mask = 0.0; }

    // ---- snow (srf_ca_snow_detail): top layer by world-direction coverage ---------------------
    var cov = 0.0;
    var n_pix = gn;
    if (lay == 1 && has_layer) {
        let n_base = h4_tbn_normal(gn, i.tangent, ts, dpx, dpy, dux, duy);
        let sdir = normalize(i.si_ctl.xyz);   // world-space accumulation direction (up8.zxy)
        let nb = mix(gn, n_base, i.obj_probe0.w);
        var d = dot(sdir, nb) + i.xbump.z;
        d = d * mix(1.0, 1.0 - color_a, i.xbump.w);
        cov = pow(clamp(d * i.xbump.x, 0.0, 1.0), i.xbump.y);
        let n1 = textureSample(at_tex, base_samp, uv_sp).xy;   // normal1_map (fine_xform)
        let ts2 = mix(ts, n1, cov);
        n_pix = h4_tbn_normal(gn, i.tangent, ts2, dpx, dpy, dux, duy);
        let c1 = textureSample(emis_tex, base_samp, uv_si);    // color1_map (mattex_xform)
        let c1l = h4_srgb(c1.rgb, (srgb & 16u) != 0u) * i.obj_probe0.xyz;
        albedo = mix(albedo, c1l, cov);
        spec_mask = mix(spec_mask, 1.0, cov);
    } else if (lay == 2 && has_layer) {
        // ---- layered3 (srf_ca_layered_three_height_detailnormal): height blend of 3 layers ---
        let bm = textureSample(mat_tex, base_samp, uv);        // blend map, untiled
        let c1 = textureSample(emis_tex, base_samp, uv_si);    // layer1 co (mattex_xform)
        let c2 = textureSample(at_tex, base_samp, uv_sp);      // layer2 co (fine_xform)
        let va = clamp(i.tint.a, 0.0, 1.0);                    // vertex blend scalar (v2.w)
        let soft0 = 1.0 - min(0.99, mix(1.0, bm.b, i.mattex.z) * va);
        let soft1 = 1.0 - min(0.99, mix(1.0, bm.b, i.mattex.w) * va);
        let h1 = 1.0 - c1.a;
        let w1 = mix(bm.r, clamp((bm.r - h1) / max(soft0, 1e-4), 0.0, 1.0), i.obj_probe0.w);
        let h2 = 1.0 - c2.a;
        var w2 = mix(bm.g, clamp((bm.g - h2) / max(soft1, 1e-4), 0.0, 1.0), i.obj_probe1.w);
        let c1l = h4_srgb(c1.rgb, (srgb & 16u) != 0u) * i.obj_probe0.xyz;
        let c2l = h4_srgb(c2.rgb, (srgb & 32u) != 0u) * i.obj_probe1.xyz;
        let n1 = textureSample(bump_detail_tex, base_samp, uv_nd).xy;                          // layer1 nm
        let n2 = textureSample(bump_detail3_tex, base_samp, uv * i.si_ctl.xy + i.si_ctl.zw).xy; // layer2 nm
        var tsl = mix(mix(ts, n1, w1), n2, w2);
        let mask_mode = i32(i.mattex.x + 0.5);
        if (mask_mode == 0) {
            albedo = mix(mix(albedo, c1l, w1), c2l, w2);
        } else {
            // srf_layered_three / srf_layered_two_detail*: plain mask sum
            // l0 * B.r + l1 * B.g + l2 * B.b (the two_detail variants normalise B by |B.rgba|)
            var bw = bm.rgb;
            if (mask_mode == 2) { bw = bm.rgb / max(length(bm), 1e-4); }
            albedo = albedo * bw.r + c1l * bw.g + c2l * bw.b;
            tsl = ts * bw.r + n1 * bw.g + n2 * bw.b;
            w2 = bw.b;
        }
        if (has_cdetail) { albedo = albedo * h4_srgb(cdet.rgb, (srgb & 2u) != 0u) * H4_DETAIL_MULT; }
        n_pix = h4_tbn_normal(gn, i.tangent, tsl, dpx, dpy, dux, duy);
        cov = w2;
        spec_mask = w1;
    } else {
        n_pix = h4_tbn_normal(gn, i.tangent, ts, dpx, dpy, dux, duy);
    }
    let n = n_pix;

    // ---- srf_char_cov* (#h4-veh): the Covenant three-colour view-angle SCREEN ramp -------------
    // entry 01 asm (h4/shading.rs `CharCov`): the colour map is a sheen sheet, not the albedo -
    // it is crushed to a few per cent and re-tinted between a face-on (up161), mid (up162) and
    // grazing (up163) colour, masked by the control map's ALPHA. Needs the mapped normal, so it
    // runs here rather than in the G-buffer block above.
    if (is_cov) {
        let c = clamp(dot(n, e), 0.0, 1.0);
        let f = 1.0 - c;
        let w_a = pow(max(f, 1e-6), i.xbump.w);              // up163.w
        let w_c = pow(max(f, 1e-6), i.fres_rgb.w);           // up162.w
        let w_b = pow(max(c, 1e-6), i.obj_probe1.w);         // up161.w
        let t_c = (1.0 - w_a) * w_c;
        let t_b = w_b * (1.0 - t_c);
        let ramp = vec3<f32>(1.0) - sp.a * (vec3<f32>(1.0) - t_b * i.obj_probe1.rgb)
                                        * (vec3<f32>(1.0) - w_a * i.xbump.rgb)
                                        * (vec3<f32>(1.0) - t_c * i.fres_rgb.rgb);
        albedo = clamp(base_lin * i.obj_probe0.xyz * ramp, vec3<f32>(0.0), vec3<f32>(1.0));
        // `_specdetail`: the G-buffer spec mask is spec_detail_map.r (the `at` slot at the
        // self-illum xform), a plain multiplier on the whole specular term; `srf_char_cov`
        // writes 0 there and binds no map at all.
        spec_mask = select(1.0, textureSample(at_tex, base_samp, uv_si).x, has_pcc);
    }

    // ---- lighting pass -------------------------------------------------------------------------
    let sun_dir = normalize(light.sun_dir.xyz);
    var lobes: H4Lobes;
    var diffuse = vec3<f32>(0.0);
    var vis = 1.0;
    // The object shadow: the engine's forge-lightmap burn multiplies its 4x4 PCF into
    // the analytic sun-visibility texel BEFORE the lighting shader's sharpening,
    // and the per-vertex / object lanes multiply the raw visibility (no sharpening there).
    let burn = h4_burn_shadow(i.world_pos);
    var vis_static_nonzero = true;   // the cascade never lights a texel whose baked sun is exactly 0
    if (i.lm.x > 0.5) {
        lobes = h4_lobes(i.uv2, -i.lm.y, i.lm.z);
        let fa = h4_lobe_diffuse(lobes.dir_a, lobes.w_a, n, gn);
        let fb = h4_lobe_diffuse(lobes.dir_b, lobes.w_b, n, gn);
        diffuse = lobes.col_a * fa + lobes.col_b * fb;
        // `// #h4-expo-2` entry 06: `mad_sat vis = analytic.x * cb2[14].x - cb2[14].y` with
        // `cb2[14] = (s + 1, (s + 1) / 2 - 0.5)` (`sub_18035E814`), s = the BSP's static
        // floating-shadow sharpening (scnr structure_bsps +184) carried in lm.w. s = 1 is the
        // familiar sat(2 a - 0.5); s = 0 (20 of 37 shipped MP BSPs) is sat(a) - no sharpening.
        let sh1 = i.lm.w + 1.0;
        vis = select(0.0, clamp(lobes.vis_raw * burn * sh1 - (sh1 * 0.5 - 0.5), 0.0, 1.0), i.lm.z > 0.0);
        vis_static_nonzero = lobes.vis_raw > 0.0 && i.lm.z > 0.0;
    } else {
        // per-vertex lit (i.tint = the CPU-evaluated lobes, a = sqrt(vis)) or flat (white)
        lobes.col_a = vec3<f32>(0.0); lobes.col_b = vec3<f32>(0.0); lobes.dir_a = vec3<f32>(0.0); lobes.dir_b = vec3<f32>(0.0);
        lobes.w_a = 0.0; lobes.w_b = 0.0; lobes.has_dir = 0.0; lobes.ana_y = 1.0;
        diffuse = i.tint.rgb;
        vis = clamp(i.tint.a, 0.0, 1.0) * clamp(i.tint.a, 0.0, 1.0) * burn;
        if (i.tint.a < -1.5) { vis = burn; }
        vis_static_nonzero = i.tint.a > 0.0 || i.tint.a < -1.5;
        // OBJECT lane, AIRPROBE mode (tint.a in [-0.75, -0.25] = -(0.25 + 0.5 sqrt(vis));
        // h4/scene.rs `lanes_pack_sh`): the object is lit by the Lbsp airprobe SH (engine
        // sub_1802E3900, lobes off). tint.rgb = the packer's per-channel DC term, obj_light = the
        // 8 remaining quadratic-form coefficients of the LUMINANCE SH as f16 pairs (a.xyz, b.xyzw,
        // c); the colour follows the DC chroma (PROVISIONAL: the per-channel L1 / L2 chroma is
        // not carried - the instance layout has no free attribute).
        let airprobe_sh = i.tint.a < -0.1 && i.tint.a > -1.4;
        if (airprobe_sh) {
            let sv = (-i.tint.a - 0.25) * 2.0;
            vis = sv * sv * burn;
            vis_static_nonzero = sv > 0.0;
            let p0 = unpack2x16float(i.obj_light.x);
            let p1 = unpack2x16float(i.obj_light.y);
            let p2 = unpack2x16float(i.obj_light.z);
            let p3 = unpack2x16float(i.obj_light.w);
            let dc = dot(i.tint.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            let e_lum = p0.x * n.x + p0.y * n.y + p1.x * n.z + dc + p1.y * n.x * n.y + p2.x * n.y * n.z + p2.y * n.z * n.z + p3.x * n.z * n.x + p3.y * (n.x * n.x - n.y * n.y);
            diffuse = max(i.tint.rgb * (max(e_lum, 0.0) / max(dc, 1e-4)), vec3<f32>(0.0));
        }
        // OBJECT lane (halo4.dll model lighting entry 11): i.tint.rgb = the probe's
        // quadratic SH (hemisphere mean, PROVISIONAL) and the packed obj_light lane carries the
        // probe's two SH-L1 lobes, evaluated per pixel through the sharpen LUT exactly like the
        // BSP lobes (they also act as specular lights below).
        if (i.obj_light.y != 0u && !airprobe_sh) {
            let d = unpack4x8snorm(i.obj_light.x);
            let w = unpack2x16float(i.obj_light.y);
            lobes.dir_a = oct_decode(d.xy); lobes.dir_b = oct_decode(d.zw);
            lobes.w_a = w.x; lobes.w_b = w.y;
            lobes.col_a = h4_unpack_rgb9e5(i.obj_light.z); lobes.col_b = h4_unpack_rgb9e5(i.obj_light.w);
            lobes.has_dir = 1.0;
            // The SURFACE PROBE (halo4.dll sub_1802E4388 / sub_1802E5A28, h4/scene.rs
            // `object_lanes_aabb`): the object's lane carries the two lobes of the lightmap texel
            // under it and NO SH (i.tint.rgb = 0); the model lighting entries (09 / 11 / 36 / 37)
            // add col_k * sat(0.2821 w_k + 0.3257 dot(d_k, N)) per lobe with the RAW lobe vector
            // (|d| = sqrt(1 - w^2)), no sharpen LUT, lobe scale cb3[7].w = 1 - and the same lobes
            // are the two specular lights below. (The airprobe / instance-probe fallbacks carry an
            // SH mean in i.tint.rgb and no lobes: cb3[7].w = 0 on those engine paths.)
            let la_raw = sqrt(max(1.0 - lobes.w_a * lobes.w_a, 0.0));
            let lb_raw = sqrt(max(1.0 - lobes.w_b * lobes.w_b, 0.0));
            let ua = clamp(0.2820948 * lobes.w_a + 0.325735 * la_raw * dot(lobes.dir_a, n), 0.0, 1.0);
            let ub = clamp(0.2820948 * lobes.w_b + 0.325735 * lb_raw * dot(lobes.dir_b, n), 0.0, 1.0);
            diffuse = diffuse + lobes.col_a * ua + lobes.col_b * ub;
        }
    }
    // The floating-shadow cascade (dynamic box around the viewer) replaces the static
    // visibility inside its box: vis = lerp(cascade, static, edge fade), 0 stays 0 (entry 06/11 law)
    if (light.casc1.z > 0.5 && vis_static_nonzero) {
        let c = h4_cascade_shadow(i.world_pos, i.clip.xy);
        vis = mix(c.x, vis, c.y);
    }
    vis = vis * lobes.ana_y;   // analytic.y (entry 06: after the cascade lerp)
    let sun_col = light.sun_tint.rgb * vis;
    diffuse = diffuse + sun_col * max(dot(n, sun_dir), 0.0) * H4_INV_PI;

    // specular: the same three lights through the family's lobe
    var spec = vec3<f32>(0.0);
    if (model > 0) {
        var p = 1.0 / max(mix(i.mat_spec2.x, i.mat_spec2.y, 1.0 - gloss), 1e-5);
        if (model == 2) { p = mix(i.mat_spec2.x, select(i.mat_spec2.x, i.si_ctl.w, lay == 1), cov); }
        var s = sun_col * h4_spec_lobe(model, n, e, sun_dir, p);
        if (lobes.has_dir > 0.5) {
            let la = length(lobes.dir_a);
            let lb = length(lobes.dir_b);
            if (la > 1e-4) { s = s + lobes.col_a * h4_spec_lobe(model, n, e, lobes.dir_a / la, p); }
            if (lb > 1e-4) { s = s + lobes.col_b * h4_spec_lobe(model, n, e, lobes.dir_b / lb, p); }
        }
        var tint = mix(i.spec_rgb.rgb, albedo, i.mat_spec2.z);
        if (model == 2) { tint = i.spec_rgb.rgb * mix(vec3<f32>(1.0), albedo, i.mat_spec2.z); }
        var sc = spec_col * tint * i.mat_spec2.w;
        if (lay == 1) {
            let t1 = i.obj_probe1.rgb * i.obj_probe1.w;
            sc = mix(sc, t1, cov);
        }
        spec = s * sc * spec_mask;
    }

    // environment reflection (srf_*_reflection): cube * tint * intensity * Rf * fresnel
    var env = vec3<f32>(0.0);
    if (has_cube && i.env_ctl.w > 0.0 && !is_cov) {
        let nr = normalize(mix(gn, n, i.mattex.y));
        let r = reflect(-e, nr);
        let cube = textureSample(env_cube, env_cube_samp, r).rgb;
        let ndv = clamp(dot(e, nr), 0.0, 1.0);
        let fb = mix(ndv, 1.0 - ndv, i.fres_rgb.w);
        let fres = mix(1.0, i.fres_rgb.x * pow(max(fb, 1e-5), i.fres_rgb.y), i.fres_rgb.z);
        env = cube * i.env_ctl.rgb * i.env_ctl.w * refl_mask * fres;
        env = mix(env, env * diffuse, i.mattex.x);
    } else if (has_cube && is_cov) {
        // #h4-veh srf_char_cov*: cube x up166.rgb, saturated by lerp(luma709, tinted, up166.w),
        // x (1 - NdotV)^up163.w (the same exponent as the ramp's wA - no fresnel scale/weight
        // triple here), x control.b, x diffuse.RED (the family is hard-wired env-lit-by-diffuse
        // and reads only the red channel of the accumulated lit term).
        let cube = textureSample(env_cube, env_cube_samp, reflect(-e, n)).rgb;
        let tinted = cube * i.env_ctl.rgb;
        let l709 = dot(tinted, vec3<f32>(0.2125, 0.7154, 0.0721));
        let sat_c = mix(vec3<f32>(l709), tinted, i.uv_density);   // up166.w
        let ndv = clamp(dot(e, n), 0.0, 1.0);
        let fres = pow(max(1.0 - ndv, 1e-6), i.xbump.w);
        env = sat_c * fres * refl_mask * max(diffuse.r, 0.0);
    }

    // self-illum
    var si = vec3<f32>(0.0);
    let si_mode = i32(i.spec_from_alpha + 0.5);
    let sim = textureSample(emis_tex, base_samp, uv_si);
    if (si_mode == 1) { si = si_mask * i.si_ctl.rgb * i.si_ctl.w * i.si_ctl.w; }
    else if (si_mode == 2) { si = albedo * i.si_ctl.rgb * i.si_ctl.w * si_mask; }
    else if (si_mode == 3 && has_simap) { si = h4_srgb(sim.rgb, (srgb & 8u) != 0u) * i.si_ctl.rgb * i.si_ctl.w; }
    // #h4-veh 4 = srf_char_cov_selfillum (up167.rgb * up167.w * color_map.a);
    // 5 = srf_char_constant (a pure emissive constant - no textures, no lit term at all)
    else if (si_mode == 4) { si = i.si_ctl.rgb * i.si_ctl.w * color_a; }
    else if (si_mode == 5) { si = i.si_ctl.rgb * i.si_ctl.w; }

    let diffuse_intensity = select(i.mattex.z, 1.0, lay == 1 || lay == 2);
    var out = albedo * diffuse_intensity * diffuse + spec + env + si;
    if (H4DBG > 0.5) {
        if (H4DBG < 1.5) { return vec4<f32>(albedo, 1.0); }
        if (H4DBG < 2.5) { return vec4<f32>(n * 0.5 + 0.5, 1.0); }
        if (H4DBG < 3.5) { return vec4<f32>(spec, 1.0); }
        if (H4DBG < 4.5) { return vec4<f32>(env, 1.0); }
        if (H4DBG < 5.5) { return vec4<f32>(diffuse, 1.0); }
        if (H4DBG < 6.5) { return vec4<f32>(spec_col, 1.0); }
        if (H4DBG < 7.5) { return vec4<f32>(vec3<f32>(gloss, spec_mask, cov), 1.0); }
        if (H4DBG < 8.5) { return vec4<f32>(si, 1.0); }
        if (H4DBG < 9.5) { return vec4<f32>(vec3<f32>(burn), 1.0); }             // burn PCF
        if (H4DBG < 10.5) { let c = h4_cascade_shadow(i.world_pos, i.clip.xy); return vec4<f32>(c.x, c.y, 0.0, 1.0); } // cascade vis / fade
        if (H4DBG < 11.5) { // sun share of the diffuse: r = sun term / total, g = vis, b = raw analytic texel
            let sun_l = dot(sun_col * max(dot(n, sun_dir), 0.0) * H4_INV_PI, vec3<f32>(0.299, 0.587, 0.114));
            let tot_l = max(dot(diffuse, vec3<f32>(0.299, 0.587, 0.114)), 1e-5);
            return vec4<f32>(sun_l / tot_l, vis, select(i.tint.a * i.tint.a, lobes.vis_raw, i.lm.x > 0.5), 1.0);
        }
    }
    // self-illum exposure (srf_*_selfillum entries: `out * lerp(ps_view_exposure.x,
    // ps_view_self_illum_exposure.y, sat(luma(si)))`, luma = (0.3086, 0.6094, 0.082) of the raw
    // self-illum term): the post pass multiplies by the scene E, so scale here by E_si / E =
    // illum_scale_now() (h4/lighting.rs `apply_post`). The alpha lane carries 1 - sat(luma(si)) =
    // the engine's o0.w (self-illum luminance; 0 on every non-si material) for the meter's
    // self-illum bloom weight (LUM_WGSL h4_measure).
    let si_luma = clamp(dot(si, vec3<f32>(0.3086, 0.6094, 0.082)), 0.0, 1.0);
    out = out * mix(1.0, illum_scale_now(), si_luma);
    out = h4_fog(max(out, vec3<f32>(0.0)), i.world_pos, cam_pos);
    return vec4<f32>(out, 1.0 - si_luma);
}

// Halo 4 atmospheric fog (h4/lighting.rs module doc: the deferred screen_atmospheric_fog
// pass, ground layer only - the shipped MP maps author no sky-layer thickness). Uniform packing:
// atm1.rgb ground colour, atm2.rgb fog-light colour, atm3.x 1/(nearby_cutoff - 1), atm4 = (density,
// base, height, falloff_end), atm5 = (dist bias, light on, cos radius, angular falloff), atm6 =
// (light dir xyz, 4.0 marker). final = scene * T + colour * (1 - T) + fog light.
fn h4_fog(shaded: vec3<f32>, wp: vec3<f32>, cp: vec3<f32>) -> vec3<f32> {
    if (fog.atm6.w < 3.5 || fog.atm4.x <= 0.0) { return shaded; }
    let dens = fog.atm4.x; let base = fog.atm4.y; let hgt = max(fog.atm4.z, 1e-3); let fend = fog.atm4.w;
    let rel = wp - cp;
    let dist = length(rel);
    let lo = min(wp.z, cp.z);
    let hi = max(wp.z, cp.z) + 1e-3;
    let dh = min(hi - base, hgt) - (lo - base);
    let frac = clamp(dh / (hi - lo), 0.0, 1.0);
    let d = min(max(0.0, frac * (dist + fog.atm5.x)), fend);
    let k = pow(clamp((hgt - (lo - base)) / hgt, 0.0, 1.0), 2.0);
    let t = min(1.0, exp2(-d * dens * k));
    var out = shaded * t + fog.atm1.rgb * (1.0 - t);
    if (fog.atm5.y > 0.5 && dist > 1e-3) {
        let vd = rel / dist;
        let cr = fog.atm5.z;
        // The engine clamps both bases to 1e-7 before its log/exp pow
        // (screen_atmospheric_fog entry 4: `max r, 1e-7; log; mul; exp`), so a zero distance
        // falloff (Longbow: 0) or angular falloff yields 1, not the NaN of pow(0, 0) that painted
        // Longbow's whole ground black.
        let ang = pow(max(clamp((dot(vd, fog.atm6.xyz) - cr) / (1.001 - cr), 0.0, 1.0), 1e-7), fog.atm5.w);
        let near = pow(max(clamp(1.0 + t * fog.atm3.x, 0.0, 1.0), 1e-7), fog.atm2.w);
        out = out + fog.atm2.rgb * (ang * near);
    }
    return out;
}
"#;

pub(crate) const WATER_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32>, inv_view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
// Opaque scene-depth copy: the bed depth behind the water, for depth murkiness.
@group(0) @binding(5) var scene_depth: texture_depth_2d;
@group(1) @binding(0) var water_tex: texture_2d<f32>;
@group(1) @binding(1) var water_samp: sampler;
// Per-material water params (rmt2 RealConstants via ZH_BSP_GetMaterialShaderConstants):
//   p0 = (watercolor_coefficient, water_murkiness, fresnel_coefficient, fresnel_dark_spot)
//   p1 = (reflection_coefficient, slope_scaler, time_warp_rate, _)
//   deep = deep-water tint (.rgb)
//   anim = (time_warp period, time_warp_aux period, const time_warp, const time_warp_aux)   [#water-anim-exact]
//   scroll = (disp_xform scroll.xy, slope_xform scroll.xy) uv/s;  detail = (detail_slope_scale_x/y/z, exact-mode flag)
//   cat = (watercolor option 0 pure/1 texture, reflection 0 none/1 static/2 dynamic, refraction 0 none/1 dynamic, wave source 0 slope array/2 bump snorm/3 bump unorm)  [#water-2]
struct WaterParams { p0: vec4<f32>, p1: vec4<f32>, deep: vec4<f32>, wave: vec4<f32>, edge: vec4<f32>, diff: vec4<f32>, env: vec4<f32>, foam: vec4<f32>, xform2: vec4<f32>, body: vec4<f32>, anim: vec4<f32>, scroll: vec4<f32>, detail: vec4<f32>, wcxf: vec4<f32>, cat: vec4<f32> };
@group(1) @binding(2) var<uniform> wp: WaterParams;
// wave_slope_array slice-0: RG = (∂h/∂x, ∂h/∂y) slope. White 1×1 fallback → noise.
@group(1) @binding(3) var slope_tex: texture_2d<f32>;
// foam_texture (RGB × A coverage). 1×1 white fallback (foam gated by foam_cut).
@group(1) @binding(4) var foam_tex: texture_2d<f32>;
// Captured per-map environment cube (real sky dome) for the reflection ray.
@group(1) @binding(10) var env_cube: texture_cube<f32>;
@group(1) @binding(11) var env_cube_samp: sampler;
// W18/W20: global_shape_texture (.a = from_shape bank_alpha). 1×1 white (a=1) fallback.
@group(1) @binding(12) var gshape_tex: texture_2d<f32>;
// Shared sun light (group0 binding2) — real to-sun dir for the glint/reflection.
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;
// per-pixel DM/SDM lightmap atlas (raw-BC DM + decoded SDM slices) sampled at the vertex
// uv2 exactly like mesh_shade's gpu_lightmap_tint / terrain's t_lightmap_tint. Engine truth
// (water_shading.hlsl_include:841-862): water reads the SAME dual-VMF lightmap as the opaque BSP and
// `water_color *= lightmap_intensity`. Previously water lit per-vertex from the airprobe grid, whose
// nearest probes for an enclosed cave puddle are OUTDOOR probes → the puddle blew out white.
@group(1) @binding(13) var wlm_dm: texture_2d<f32>;
@group(1) @binding(14) var wlm_sdm: texture_2d<f32>;
@group(1) @binding(15) var wlm_samp: sampler;
@group(1) @binding(16) var wlm_sdm1: texture_2d<f32>;
@group(1) @binding(17) var wlm_sdm2: texture_2d<f32>;
// byte-exact vMF diffuse LUT (camera group, shared with mesh_shade / terrain)
@group(0) @binding(7) var vmf_lut: texture_2d<f32>;
@group(0) @binding(8) var vmf_samp: sampler;
// total = (LUT(N·dom,κ)·dom_col + 0.25·fill_col)/π · hdr · k  (the mesh_shade `bake` units), .a = sun vis (DM.g)
// fill  = 0.25·fill_col/π · hdr · k;  dom = raw dominant lobe colour · hdr · k (engine vmf[1].rgb)
struct WLobes { total: vec4<f32>, fill: vec3<f32>, dom: vec3<f32> };
// The water copy of the dual-VMF atlas decode: the same engine law as the mesh shader's
// `gpu_lightmap_tint`, against this module's own DM/SDM bindings, and additionally returning the
// raw dominant lobe the engine's water `lightmap_intensity` uses.
fn w_lightmap_tint(uv2: vec2<f32>, n: vec3<f32>, hdr: f32, k: f32) -> WLobes {
    let d  = textureSampleLevel(wlm_dm,   wlm_samp, uv2, 0.0);
    let s0 = textureSampleLevel(wlm_sdm,  wlm_samp, uv2, 0.0);
    let s1 = textureSampleLevel(wlm_sdm1, wlm_samp, uv2, 0.0);
    let s2 = textureSampleLevel(wlm_sdm2, wlm_samp, uv2, 0.0);
    var dom_dir = vec3<f32>(s0.w * 2.0 - 1.0, s1.w * 2.0 - 1.0, s2.w * 2.0 - 1.0);
    let dl = length(dom_dir);
    if (dl > 1e-4) { dom_dir = dom_dir / dl; } else { dom_dir = vec3<f32>(0.0, 0.0, 1.0); }
    let f_int = exp2(-9.000001 * d.x);
    let dom_col  = (s0.rgb + s1.rgb * 2.0 - 1.0) * f_int;
    let fill_col = s2.rgb * f_int;
    let vis = d.y;
    let ndotd = dot(dom_dir, n);
    let cc = clamp(ndotd * 0.5 + 0.5, 0.0, 1.0);
    let dom_co = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(cc, clamp(dl, 0.0, 1.0)), 0.0).r;
    let inv_pi = 0.31830989;
    var tint = vec3<f32>(0.0, 0.0, 0.0);
    var fill = vec3<f32>(0.0, 0.0, 0.0);
    var dom  = vec3<f32>(0.0, 0.0, 0.0);
    for (var c = 0; c < 3; c = c + 1) {
        let irr = (dom_co * dom_col[c] + 0.25 * fill_col[c]) * inv_pi;
        tint[c] = max(irr * hdr * k, 0.0);
        fill[c] = 0.25 * fill_col[c] * inv_pi * hdr * k;
        dom[c]  = dom_col[c] * hdr * k;
    }
    return WLobes(vec4<f32>(tint, vis), fill, dom);
}

// Direction-correct sky radiance for the reflected ray (stand-in for the per-
// material environment cubemap — the engine's no-cubemap fallback). A simple
// zenith/horizon/ground gradient so the reflection SHIFTS with the wave normal.
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    // brighter, bluer sky reflection (toward the C# water sky ~0.55,0.70,0.92) so
    // grazing fresnel reflections read as sky instead of a dull grey band.
    let up = clamp(dir.z, 0.0, 1.0);
    let horizon = vec3<f32>(0.55, 0.68, 0.85);
    let zenith = vec3<f32>(0.24, 0.45, 0.78);
    let ground = vec3<f32>(0.14, 0.15, 0.14);
    let sky = mix(horizon, zenith, up);
    return select(ground, sky, dir.z >= 0.0);
}

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(9) sway: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
    @location(13) uv2: vec2<f32>,   // lightmap atlas UV (submap-local)
    @location(14) lm: vec4<f32>,    // [atlas flag, hdr, k, mode]
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(4) uv2: vec2<f32>,
    @location(5) @interpolate(flat) lm: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    o.clip = cam.view_proj * world;
    o.normal = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    o.uv = v.uv;
    o.world_pos = world.xyz;
    o.vcolor = v.vcolor;   // baked lighting (airprobe ambient) → water_color *= lightmap_intensity
    o.uv2 = v.uv2;
    o.lm = v.lm;
    return o;
}

// Cheap value noise for wave normal perturbation.
fn h21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}
fn vnoise(uv: vec2<f32>) -> f32 {
    let i = floor(uv); let f = fract(uv);
    let a = h21(i); let b = h21(i + vec2<f32>(1.0, 0.0));
    let c = h21(i + vec2<f32>(0.0, 1.0)); let d = h21(i + vec2<f32>(1.0, 1.0));
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Sample ONE animation frame of the wave_slope_array atlas (N slices stacked
// vertically), 2-slice lerped by a fractional cursor — the engine's time_warp slice index.
fn sample_wave_slice(uv: vec2<f32>, cursor: f32, n: f32) -> vec2<f32> {
    let ci = floor(cursor);
    let s0 = ci - floor(ci / n) * n;              // frame index (mod n)
    let s1 = (s0 + 1.0) - floor((s0 + 1.0) / n) * n; // next frame (wrap)
    let fr = cursor - ci;
    let u = fract(uv.x);
    let vv = fract(uv.y);
    let a = textureSampleLevel(slope_tex, water_samp, vec2<f32>(u, (s0 + vv) / n), 0.0).rg;
    let b = textureSampleLevel(slope_tex, water_samp, vec2<f32>(u, (s1 + vv) / n), 0.0).rg;
    return mix(a, b, fr) * 2.0 - 1.0;            // remap [0,1]→[-1,1]
}
// Sample the wave slope map → world-space XY slope. The engine advances an array-slice cursor by
// time_warp; we cycle the N decoded frames (main + aux layers at different tiles/rates/phase).
// n_slices == wp.deep.w (0 = no authored array → procedural noise fallback).
// Returns (slope.x, slope.y, wave_choppiness_ratio). choppiness_ratio = max over the two animated
// wave layers of (1 - |slope.x| - |slope.y|) — engine compose_slope_original L694-696. High (≈1)
// on FLAT water (slope≈0), low on steep wave faces. Feeds the auto-foam factor.
fn wave_slope(wuv: vec2<f32>, t: f32, warpA: f32, warpB: f32, tileA: vec2<f32>, tileB: vec2<f32>, offA: vec2<f32>, offB: vec2<f32>, detail_steep: f32, n_slices: f32) -> vec3<f32> {
    // The engine animates by a frame cursor over the slope ARRAY plus TranslationX/Y overlays on the
    // xform offsets — in the engine's MESH-UV space. HMS samples in world/30 space (the mesh-uv mapping read as a
    // flat texture), where the authored 1/period rates run several times too fast, so the earlier calibrated
    // speeds are kept: slow drift + ~0.7 frames/s cursor (user A/B: closer than the engine-rate version).
    let uvA = wuv * tileA + offA + vec2<f32>(0.9, 0.5) * (t * warpA);
    let uvB = wuv * tileB + offB + vec2<f32>(-0.6, 0.8) * (t * warpB);
    if (n_slices > 0.5) {
        let cursorA = t * (0.6 + warpA * 3.0);
        let cursorB = t * (0.4 + warpB * 3.0) + 0.37 * n_slices;
        let sA = sample_wave_slice(uvA, cursorA, n_slices);
        let sB = sample_wave_slice(uvB, cursorB, n_slices);
        var sD = vec2<f32>(0.0, 0.0);
        if (detail_steep > 0.0001) {
            let cursorD = t * (0.5 + (warpA + warpB) * 1.5) + 0.13 * n_slices;
            sD = sample_wave_slice(wuv * tileA * 2.5 + offA, cursorD, n_slices) * detail_steep;
        }
        let r1 = 1.0 - abs(sA.x) - abs(sA.y);
        let r2 = 1.0 - abs(sB.x) - abs(sB.y);
        let chop = max(r1, r2);
        return vec3<f32>(sA + sB + sD, chop);
    }
    let e = 0.75;
    let nx = (vnoise(uvA + vec2<f32>(e, 0.0)) - vnoise(uvA - vec2<f32>(e, 0.0)))
           + (vnoise(uvB + vec2<f32>(e, 0.0)) - vnoise(uvB - vec2<f32>(e, 0.0)));
    let ny = (vnoise(uvA + vec2<f32>(0.0, e)) - vnoise(uvA - vec2<f32>(0.0, e)))
           + (vnoise(uvB + vec2<f32>(0.0, e)) - vnoise(uvB - vec2<f32>(0.0, e)));
    return vec3<f32>(nx, ny, 1.0 - abs(nx) - abs(ny));
}

// — ENGINE-EXACT wave animation, per material (HMS_WATER_LEGACY=1 → old path above):
//   water_shading.hlsl_include:653-702 compose_slope_original
//     texcoord     = (transform_texcoord(mesh_uv, wave_displacement_array_xform), time_warp)
//     texcoord_aux = (transform_texcoord(mesh_uv, wave_slope_array_xform),        time_warp_aux)
//     slope = lerp(slice[floor(z)], slice[next], frac(z)) with z = frac(time_warp)·N   (global_texture
//             .hlsl_include convert_3d_texture_coord_to_array_texture; next wraps N-1 → 0)
//     wave_choppiness_ratio = max(1-|slope|, 1-|slope_aux|);  slope_shading = slope + slope_aux + detail
//   water_shading.hlsl_include:586-614 compute_detail_slope
//     xform_d = disp_xform · (detail_slope_scale_x, detail_slope_scale_y, 1, 1)
//     z_d = time_warp · detail_slope_scale_z;  slope_detail = sample · detail_slope_steepness
//   transform_texcoord (shared/texture_xform.hlsl_include) = uv·xform.xy + xform.zw. The TranslationX/Y
//   overlays animate xform.zw (offset + rate·t); a Value overlay on time_warp/time_warp_aux is a 0..1 ramp
//   over its period (frac(t/period)); a material with no overlay keeps the constant authored arg.
// The atlas has a mip chain, so the samples take explicit UV gradients (the engine's PS `Sample`
// auto-mips the array; sampling level 0 only would alias at distance).
fn sample_wave_slice_grad(uv: vec2<f32>, cursor: f32, n: f32, ddx: vec2<f32>, ddy: vec2<f32>) -> vec2<f32> {
    let ci = floor(cursor);
    let s0 = ci - floor(ci / n) * n;                 // slice index (mod n)
    let s1 = (s0 + 1.0) - floor((s0 + 1.0) / n) * n; // next slice (wrap)
    let fr = cursor - ci;
    let u = fract(uv.x);
    let vv = fract(uv.y);
    let gx = vec2<f32>(ddx.x, ddx.y / n);
    let gy = vec2<f32>(ddy.x, ddy.y / n);
    let a = textureSampleGrad(slope_tex, water_samp, vec2<f32>(u, (s0 + vv) / n), gx, gy).rg;
    let b = textureSampleGrad(slope_tex, water_samp, vec2<f32>(u, (s1 + vv) / n), gx, gy).rg;
    return mix(a, b, fr) * 2.0 - 1.0;               // remap [0,1]→[-1,1]
}
// Animated time_warp parameter: Value overlay → frac(t/period); no overlay → the constant arg.
fn time_warp_at(t: f32, period: f32, constant: f32) -> f32 {
    return select(constant, fract(t / max(period, 1e-4)), period > 1e-4);
}
// Returns (slope.x, slope.y, wave_choppiness_ratio) in the mesh-UV tangent frame, before slope_scaler.
fn wave_slope_exact(uv: vec2<f32>, t: f32, n: f32, dux: vec2<f32>, duy: vec2<f32>) -> vec3<f32> {
    let off_a = wp.xform2.xy + wp.scroll.xy * t;      // animated wave_displacement_array_xform.zw
    let off_b = wp.xform2.zw + wp.scroll.zw * t;      // animated wave_slope_array_xform.zw
    // Waveshape == bump (water_shading.hlsl_include:799-808): two tangent-space normal maps
    // (bump_map at bump_map_xform, bump_detail_map at bump_detail_map_xform; the SAME texture is bound in
    // the slope slot for both -- shipped materials author one bitmap for both usages), then
    //   bump.xy += detail.xy;  slope = bump.xy / max(bump.z, 0.01)
    // sample_bumpmap for a DXN map: xy signed, z = sqrt(saturate(1 - dot(xy, xy))). No choppiness ratio
    // (auto foam is only computed by compose_slope_original) -> return 0 for it.
    if (wp.cat.w > 1.5) {
        let uv_a = uv * wp.wave.xy + off_a;
        let uv_b = uv * wp.wave.zw + off_b;
        var b = textureSampleGrad(slope_tex, water_samp, uv_a, dux * wp.wave.xy, duy * wp.wave.xy).xy;
        var d = textureSampleGrad(slope_tex, water_samp, uv_b, dux * wp.wave.zw, duy * wp.wave.zw).xy;
        if (wp.cat.w > 2.5) { b = b * 2.0 - 1.0; d = d * 2.0 - 1.0; }
        let bz = sqrt(saturate(1.0 - dot(b, b)));
        let bxy = b + d;
        return vec3<f32>(bxy / max(bz, 0.01), 0.0);
    }
    let tw  = time_warp_at(t, wp.anim.x, wp.anim.z);
    let twa = time_warp_at(t, wp.anim.y, wp.anim.w);
    let uv_a = uv * wp.wave.xy + off_a;
    let uv_b = uv * wp.wave.zw + off_b;
    let s_a = sample_wave_slice_grad(uv_a, fract(tw) * n, n, dux * wp.wave.xy, duy * wp.wave.xy);
    let s_b = sample_wave_slice_grad(uv_b, fract(twa) * n, n, dux * wp.wave.zw, duy * wp.wave.zw);
    var s_d = vec2<f32>(0.0, 0.0);
    if (wp.foam.w > 0.0001) {                         // detail_slope_steepness > 0 (detail = repeat)
        let tile_d = wp.wave.xy * wp.detail.xy;
        let uv_d = uv * tile_d + off_a;
        s_d = sample_wave_slice_grad(uv_d, fract(tw * wp.detail.z) * n, n, dux * tile_d, duy * tile_d) * wp.foam.w;
    }
    let r1 = 1.0 - abs(s_a.x) - abs(s_a.y);
    let r2 = 1.0 - abs(s_b.x) - abs(s_b.y);
    return vec3<f32>(s_a + s_b + s_d, max(r1, r2));
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let t = cam.time.x;
    // Per-material params (rmt2 RealConstants, scalars in .x).
    let wc_coef  = wp.p0.x;
    let murk     = wp.p0.y;
    let fres_coef = wp.p0.z;
    let dark_spot = wp.p0.w;
    let refl_coef = wp.p1.x;
    let slope_scaler = wp.p1.y;
    // time_warp / time_warp_aux drive the two wave layers' scroll rate: p1.z/p1.w carry the
    // per-map wave rate = Σ 1/TimePeriod over the material's TranslationX/Y overlays, in tiles/sec
    // (the same units the waterfall path uses, bsp_material_yscroll). Per-map speeds vary ~12×
    // (calm mat54 1/12 vs fast mat52 1.03) so there is no global multiplier, and rate 0 (no
    // overlay) = STATIC water. TODO: the sum over-counts a material with several overlays
    // (settlement mat205 sums to ~1.36) because per-SID layer routing does not yet isolate the
    // dominant one, so the summed rate is scaled down (≈dominant-of-3) and capped.
    let warpA = min(abs(wp.p1.z) * 0.30, 0.12);
    let warpB = min(abs(wp.p1.w) * 0.30, 0.12);
    let has_slope = wp.deep.w > 0.5;
    let tileA = select(vec2<f32>(0.6, 0.6), wp.wave.xy, wp.wave.x > 0.001);
    let tileB = select(vec2<f32>(1.1, 1.1), wp.wave.zw, wp.wave.z > 0.001);
    // W5: authored UV offsets (xform.zw). wp.xform2.xy = layerA, .zw = layerB.
    let offA = wp.xform2.xy;
    let offB = wp.xform2.zw;
    // The engine does NOT clamp refraction_extinct_distance to ≥1
    // (water_shading.hlsl_include:943 uses it raw in saturate(extinct·one_over_camera_distance)).
    // Guard only against a zero/negative authored value; materials authoring <1 now honour it.
    let extinct = max(wp.edge.x, 0.0);
    let bank_mode = wp.edge.z;            // 1.0 = paint (puddle/stream)
    let nvt = wp.edge.w;                  // normal_variation_tweak

    // Sample in WORLD space (water UVs are often degenerate). Engine-equivalent viewer
    // scale is 1/30 (BspWaterTechnique) — 0.12 was ~3.6× too dense (too many wave tiles).
    // world-space mapping kept — the mesh-texcoord mapping (engine asm) made the colour/wave tiling read as
    // an obvious flat texture (user A/B'd it: worse). The ocean's texcoords are not what the engine's xforms assume here.
    let wuv = i.world_pos.xy * (1.0 / 30.0);
    // The WAVE layers sample in the engine's MESH-texcoord space (uv·xform + scroll·t)
    // with per-material frac(t/period)·N slice cursors — see wave_slope_exact. wuv (world/30) stays only
    // for the legacy path (HMS_WATER_LEGACY=1 / no usable mesh UVs) and the untouched colour/foam lookups.
    // Screen derivatives must be taken in uniform control flow: mesh uv (explicit sample gradients) and
    // world_pos (the UV tangent frame the engine gets from the vertex tangent/binormal, :809-813).
    let dux = dpdx(i.uv);
    let duy = dpdy(i.uv);
    let dpx = dpdx(i.world_pos);
    let dpy = dpdy(i.world_pos);
    let exact = wp.detail.w > 0.5 && has_slope;
    var ws: vec3<f32>;
    if (exact) {
        ws = wave_slope_exact(i.uv, t, wp.deep.w, dux, duy);
    } else {
        ws = wave_slope(wuv, t, warpA, warpB, tileA, tileB, offA, offB, wp.foam.w, wp.deep.w);
    }
    let slope = ws.xy;
    let chop = ws.z;   // wave_choppiness_ratio (1-|slope|), for auto-foam

    let base_n = normalize(i.normal);
    // Wave normal: slope perturbs the flat normal, amplitude × slope_scaler. Honor
    // a small authored slope_scaler (puddles are near-flat) instead of an ocean
    // floor — a tiny epsilon only, so calm puddles read calm.
    // Engine: nTangent = (slope·slope_scaler, 1) — NO extra amplitude (the 0.9/2.5
    // multipliers were viewer inventions). Floor 0.001 so calm puddles read calm.
    let amp = max(slope_scaler, 0.001);
    var n: vec3<f32>;
    if (exact) {
        // Engine :809-813: normal = normalize(mul(float3(slope_shading, 1), {tangent, binormal, normal}))
        // — the slope lives in the MESH-UV tangent frame. The water vertex stream carries no tangent, so
        // rebuild dP/du, dP/dv from the screen derivatives of (world_pos, uv) — exact for the planar
        // water mapping. Degenerate quads (|det|≈0) fall back to the world X/Y axes.
        let det = dux.x * duy.y - duy.x * dux.y;
        var tng = vec3<f32>(1.0, 0.0, 0.0);
        var bin = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(det) > 1e-12) {
            tng = (dpx * duy.y - dpy * dux.y) / det;
            bin = (dpy * dux.x - dpx * duy.x) / det;
        }
        let tl = length(tng);
        let bl = length(bin);
        tng = select(vec3<f32>(1.0, 0.0, 0.0), tng / tl, tl > 1e-6);
        bin = select(vec3<f32>(0.0, 1.0, 0.0), bin / bl, bl > 1e-6);
        n = normalize(tng * (slope.x * amp) + bin * (slope.y * amp) + base_n);
    } else {
        n = normalize(base_n + vec3<f32>(slope.x, slope.y, 0.0) * amp);
    }

    let view = normalize(cam.cam_pos.xyz - i.world_pos);
    let nov = clamp(dot(n, view), 0.0, 1.0);

    // Base water colour = watercolor bitmap × watercolor_coefficient (engine
    // water_shading base term), sRGB→linear. Keep the alpha: for bankalpha=paint
    // (puddles/streams) watercolor.a is the shore-coverage mask (0 at the water↔land
    // edge → 1 at the pool centre); oceans (mode 0) get bank_alpha=1 (no fade).
    // Engine samples watercolor at a STATIC uv*xform. The viewer lacks the authored
    // watercolor_texture_xform (usually low-frequency), so the fixed 1/30 tiling made the
    // overlay far too DEFINED/sharp. Sample it at a COARSER (×0.4) scale so it reads as a
    // soft low-frequency tint rather than a crisp repeating texture sitting on the water.
    // WATER ANIMATION: the watercolor bitmap was sampled at a STATIC uv, so only the wave
    // NORMAL animated → the visible surface tint sat still while the water "moved" underneath. The
    // engine ripples the surface via screen-space refraction (scene UV offset by the animated wave
    // slope). We reproduce that bump on the watercolor sample. #273b: `slope` is the SUMMED two-layer
    // wave slope (range ~±2, oscillating at the wave rate) — the first shift (0.12) was ~±0.24 uv of
    // FAST jitter → "super agitated" water, nothing like the game's calm surface. The engine's real
    // refraction_texcoord_shift is TINY; use a small shift so it reads as a subtle surface shimmer,
    // not violent motion. Materials with no wave rates (slope≈0) stay static.
    // The engine perturbs its scene/colour lookup by slope × refraction_texcoord_shift so the body
    // colour moves WITH the waves; a static uv leaves the tint sitting on top of the motion. (The
    // shift must stay small: `slope` is the summed two-layer wave slope, range ~±2, so a large
    // coefficient reads as violent jitter rather than a surface shimmer.)
    var wc = textureSample(water_tex, water_samp, wuv * 0.4 + slope * 0.15);   // refraction_texcoord_shift 0.05 (Forge ocean) × 3 in this uv space
    // bank_mode: 2 = ocean (watercolor=texture → tint from the bitmap; from_shape
    // coverage ≈ opaque), 1 = puddle (watercolor=PURE → tint = water_color_pure
    // (wp.deep.rgb); paint coverage = wc.a shore mask), 0 = none. Puddles ignore the watercolor
    // bitmap RGB and tint by water_color_pure.
    // The rmt2 CATEGORY options drive the branches (HLSL TEST_CATEGORY_OPTION), not the
    // bound-bitmap heuristic. bank_paint = bankalpha==paint (coverage from watercolor.a);
    // wc_pure = watercolor==pure (tint = water_color_pure, whatever bitmaps the material also binds).
    // Ridgeline's spawn stream (template _0_0_1_1_2_0_1_3_1: pure + from_shape + foam both) was routed
    // as an "ocean" because it binds a foam texture, so its near-black water_color_pure (0.004) was
    // replaced by a fallback bitmap (~0.26) x lightmap -> a blown-white river.
    let bank_paint = bank_mode > 0.5 && bank_mode < 1.5;
    let wc_pure = wp.cat.x < 0.5;
    let refl_none = wp.cat.y < 0.5;
    let refr_none = wp.cat.z < 0.5;
    let is_puddle = bank_paint;
    // (water_shading.hlsl_include:857 + :868-869): for bankalpha=paint the puddle
    // COVERAGE is watercolor_texture.a sampled at the STATIC mesh uv × watercolor_texture_xform — the
    // authored puddle shapes span the puddle mesh's texcoords once. The world/75 + slope-jittered
    // lookup above tiled that mask at the wrong scale AND re-sampled it every frame with the wave
    // slope, so puddles rendered as scattered blotches that flickered at the ocean's wave rate
    // (user: "puddles animate generically with the ocean"). Ocean/texture-mode tint keeps the
    // world-space lookup (user-A/B'd earlier; only the paint coverage is switched).
    let wc_mesh = textureSample(water_tex, water_samp, i.uv * wp.wcxf.xy + wp.wcxf.zw);
    if (exact && is_puddle) { wc = wc_mesh; }
    let tex_color = pow(wc.rgb, vec3<f32>(2.2)) * wc_coef;
    // water_shading.hlsl:862 `water_color *= lightmap_intensity`: the water tint is LIT by the
    // baked lighting (sky/sun on the surface). lightmap_intensity is a float3 RGB applied
    // PER-CHANNEL (HREK:841-843), not a scalar luminance. The engine applies NO warm hue boost —
    // the ocean tint is the watercolor bitmap × lightmap_intensity.
    // lm.x>0.5 → this mesh carries the per-pixel DM/SDM atlas: decode the dual-VMF at uv2 (flat
    // +Z water-plane normal) and use the engine's literal water_shading.hlsl_include:841-843
    // `lightmap_intensity = vmf[1].rgb + vmf[0].a` (the raw dominant lobe colour·compress·
    // fIntensity plus the sun-visibility mask — no vMF LUT cosine, no 1/π). lm.x==0 → the
    // per-vertex airprobe stand-in in i.vcolor (no atlas).
    var lm_intensity = max(i.vcolor.rgb, vec3<f32>(0.0));
    if (i.lm.x > 0.5) {
        let wl = w_lightmap_tint(i.uv2, vec3<f32>(0.0, 0.0, 1.0), i.lm.y, i.lm.z);
        lm_intensity = max(wl.dom, vec3<f32>(0.0)) + vec3<f32>(clamp(wl.total.a, 0.0, 1.0));
    }
    // water_shading.hlsl_include:854-862 -- pure -> water_color_pure, texture -> bitmap * coef.
    var water_color = select(tex_color, wp.deep.rgb, wc_pure) * lm_intensity;
    // SAFETY INVARIANT: real water is blue/teal — its red channel is never the brightest, so a
    // red-DOMINANT water tint is always a bug (a red airprobe ambient over teal creek water once
    // produced a flat red slab). Cap R to just above max(G,B). Only for the bitmap (texture) path:
    // an authored water_color_pure is used verbatim, since Forge World's canyon puddles author a
    // muddy BROWN (0.047, 0.031, 0.012) that the cap would re-tint toward grey-green.
    if (!wc_pure) {
        water_color.r = min(water_color.r, max(water_color.g, water_color.b) * 1.10 + 0.02);
    }
    // Coverage: puddle → wc.a shore fade; ocean/none → opaque (from_shape ≈ 1).
    // water_shading:871: bankalpha has a 3rd mode `from_shape_texture_alpha` — the shore
    // coverage comes from the authored global_shape_texture .a (R=height, G=choppy, B=foam_paint,
    // A=bank_alpha per W20). env.w>0.5 flags a real global_shape is bound; from_shape is the ocean
    // path (bank_mode 2). The global_shape is a per-BODY coverage mask authored to span the water
    // body ONCE (~1 in the deep centre → 0 at the water↔land edge) — it does NOT tile. HMS's water
    // samples a WORLD-TILED wave UV (wuv), and the mesh texcoord i.uv is not a reliable body-span in
    // HMS's decode, so both would repeat/misalign the mask. Instead derive a per-body UV from the
    // water mesh's world-space XY AABB (wp.body.xy = aabb_min, wp.body.zw = 1/aabb_size), which maps
    // global_shape 0..1 across the body once — like the engine's authored body texcoords. Clamp to
    // [0,1] (edge coverage). Degenerate AABB (wp.body.zw==0) → fall back to i.uv. Gated by HMS_NO_W18
    // (env.w forced 0 upstream) for before/after isolation. The WAVE textures keep their world-tiled
    // wuv (they DO tile) — only global_shape uses the body UV.
    let is_from_shape = bank_mode > 1.5;
    let has_gshape = wp.env.w > 0.5;
    let has_body = wp.body.z > 0.0 && wp.body.w > 0.0;
    let body_uv = clamp((i.world_pos.xy - wp.body.xy) * wp.body.zw, vec2<f32>(0.0), vec2<f32>(1.0));
    let gshape_uv = select(i.uv, body_uv, has_body);
    let gshape = textureSample(gshape_tex, water_samp, gshape_uv);
    var bank_alpha = select(1.0, clamp(wc.a, 0.0, 1.0), is_puddle);
    if (is_from_shape && has_gshape) {
        bank_alpha = clamp(gshape.a, 0.0, 1.0);
    }

    // TRANSPARENCY — engine Beer-Lambert (water_shading.hlsl_include:717-943):
    //   transparency = saturate(exp2(murkiness · (bed.z − surface.z))) · saturate(extinct/camDist)
    // Sample the OPAQUE scene depth behind this water pixel, unproject it to the
    // bed's world position, and use the REAL geometric thickness. Shallow water (puddles,
    // shorelines) → tiny thickness → see-through to the ground; deep water → murky.
    let murkN = clamp(murk / 8.0, 0.0, 1.0);
    let cam_dist = length(cam.cam_pos.xyz - i.world_pos);
    let px = vec2<i32>(i.clip.xy);
    let bed_ndc = textureLoad(scene_depth, px, 0);
    var neg_depth: f32;
    if (bed_ndc >= 0.9999) {
        // No opaque geometry behind (open water over the void / far plane) — fall back to
        // the angle proxy so it isn't uniformly opaque.
        let ocean_depth = mix(mix(0.15, 0.6, murkN), mix(1.5, 4.0, murkN), nov);
        let puddle_depth = mix(0.04, 0.30, nov);
        neg_depth = -select(ocean_depth, puddle_depth, is_puddle);
    } else {
        let dim = vec2<f32>(textureDimensions(scene_depth));
        let uv2 = i.clip.xy / dim;
        let ndc = vec3<f32>(uv2.x * 2.0 - 1.0, 1.0 - uv2.y * 2.0, bed_ndc);
        let bedH = cam.inv_view_proj * vec4<f32>(ndc, 1.0);
        let bed = bedH.xyz / bedH.w;
        // thickness = water surface Z − bed Z (Halo world is +Z up); engine negative_depth.
        let thickness = max(i.world_pos.z - bed.z, 0.0);
        neg_depth = -thickness;
    }
    // bank_alpha does NOT enter compute_fog_transparency in the engine — it scales the refraction
    // bump + lerps the bed colour, and is applied separately below as the engine's coverage /
    // bed-colour lerp (out_rgb *= bank_alpha, coverage *= bank_alpha).
    // Murk base exp2, the Reach source (water_shading.hlsl:717 compute_fog_transparency
    // `saturate(exp2(murkiness * negative_depth))`; H3's shader uses base e, which decays faster
    // and makes every shallow body ~1.5-2x less see-through than this engine).
    var transparency = clamp(exp2(murk * neg_depth), 0.0, 1.0);
    transparency = transparency * clamp(extinct / max(cam_dist, 0.001), 0.0, 1.0);
    // (water_shading.hlsl_include:877-882): refraction == none -> color_refraction_blend =
    // (water_color, a = 0): the body is opaque water_color, no Beer-Lambert see-through.
    if (refr_none) { transparency = 0.0; }

    // FRESNEL — engine (fresnel_dark_spot − NoV)² · fresnel_coefficient, NO baseline
    // (the "always-on reflection" is really the sun glint below).
    // water_shading:1112: fresnel_normal = k_is_camera_underwater ? -normal : normal. When the
    // eye is below the surface, the fresnel normal flips so the term reads the underside grazing
    // angle (looking UP through the water gives a strong internal-reflection fresnel). cam.time.y
    // is the per-frame camera-underwater flag set in render() (0 = above).
    let underwater = cam.time.y > 0.5;
    let fresnel_normal = select(n, -n, underwater);
    let nov_f = clamp(dot(fresnel_normal, view), 0.0, 1.0);
    // water_ps_full_17132.asm 205-208: `sat(dot(N,V)) → add_sat(dark_spot - NoV) → x² →
    // mul_sat × fresnel_coefficient`. With the Forge ocean's dark_spot 0.9 / coefficient 0.04 that
    // is ≤ 0.032 at ANY angle — the engine's water is almost non-reflective; its blue is the
    // refracted scene plus the additive water colour, not a sky mirror.
    let e = clamp(dark_spot - nov_f, 0.0, 1.0);
    let fresnel = clamp(fres_coef * e * e, 0.0, 1.0);

    // REFLECTION — reflected-ray sky (flatten the wave normal toward vertical by
    // normal_variation_tweak=0.7 so reflections aren't over-choppy), gated by
    // reflection_coefficient. Plus the analytical SUN glint (engine power 20) on
    // the wave-perturbed normal, driven by the real scene sun.
    let n_reflect = normalize(mix(vec3<f32>(0.0, 0.0, 1.0), n, nvt));
    let refl_dir = reflect(-view, n_reflect);
    // water_shading.hlsl_include:1049: the engine flips the reflection vector's
    // Y before the cube lookup, unconditionally (cube-face handedness). `build_authored_env_cube`
    // stores faces in raw engine order, so the authored-cube path needs this flip to match the sky
    // orientation. Applied only to the authored-cube sample (the captured sky dome stand-in uses
    // its own capture orientation).
    let refl_dir_cube = vec3<f32>(refl_dir.x, -refl_dir.y, refl_dir.z);
    // docs/hrek_re/12_re_water.md §3: the engine gates the cubemap reflection by
    // reflAlpha = min(env.a, sunspot_cut) (~0.2) AWAY from the sun-disk — across the bulk of the
    // surface it reflects only a MUTED ambient sky, not a full mirror — so the diffuse blue water
    // colour dominates and the sun GLINT stays full-strength and localized (the engine's only
    // bright reflection off-disk).
    let refl_ambient = wp.diff.w;   // authored sunspot_cut ambient-reflection floor
    // Reflect the REAL per-map environment cube (captured sky dome),
    // not a hand-tuned gradient — so the reflection matches the actual sky above each map (blue
    // ocean sky vs a stormy/space sky). Slightly blurred (mip 2) so it reads as a soft water
    // reflection, not a mirror. Falls back to sky_radiance if the cube reads ~black (uncaptured).
    var reflection: vec3<f32>;
    if (wp.env.x > 0.5) {
        // (re7_agent_06 §17, water_shading.hlsl H:1055-1084): the REAL AUTHORED
        // environment_map cube is bound at binding 10. It is a STATIC Forge cube → no exponential
        // bias, so lift its LDR rgb to HDR radiance by ×wp.env.y (engine `env.rgb *= 256`). Its
        // per-texel ALPHA is a sky-hemisphere HDR mask, split at sunspot_cut into an always-on
        // ambient-sky floor (parts.y) plus a bright-sky/sun-disk excess (parts.x) gated by the
        // baked sun visibility. There is NO analytic sun Blinn-Phong on this path — the sun disc
        // lives in the cube's alpha, so the legacy glint is intentionally dropped here.
        // water_shading.hlsl_include:1055: the engine uses sampleCUBE with hardware auto-mip
        // (≈mip0 sharp mirror), not a forced blur. Sample mip 0 (Y-flipped dir, see above).
        let env = textureSampleLevel(env_cube, env_cube_samp, refl_dir_cube, 0.0);
        // (protomorph entry_water_shading `color_reflection = env.rgb·env.a·refl_coef`):
        // reflect the cube at DISPLAY brightness × its alpha mask × reflection_coefficient — NO env×256
        // HDR lift. Paired with the Schlick fresnel, grazing water becomes a natural sky mirror instead
        // 03_shader_water.md §3a asm 143-159: reflection = env.rgb*256 * reflAlpha *
        // reflection_coefficient, with reflAlpha = saturate(env.a - sunspot_cut)*slopeFactor +
        // min(env.a, sunspot_cut) and slopeFactor = saturate(slope - shadow_intensity_mark)².
        // The authored reflection_coefficient (0.03 on the Forge ocean) is written AGAINST that
        // ×256, so the lift is part of the law, not a tuning knob.
        let slope_mag = length(n.xy) / max(abs(n.z), 1e-3);
        let slope_above = saturate(slope_mag - wp.foam.z);
        // (water_shading.hlsl_include:1069-1077): the bright-sky/sun excess of the
        // cube alpha (parts.x) is gated by the LIGHTMAP: sun_light_rate = saturate(lightmap_intensity
        // - shadow_intensity_mark); sun_scale = dot(sun_light_rate, sun_light_rate). The asm reading
        // above took the gate to be the wave slope — that lit the sun-sky excess on every steep facet
        // even in an unlit cave (cyan puddle blotches). Legacy path keeps the slope gate.
        let sun_rate = saturate(lm_intensity - vec3<f32>(wp.foam.z));
        let sun_scale = select(slope_above * slope_above, dot(sun_rate, sun_rate), exact);
        let refl_alpha = saturate(env.a - wp.diff.w) * sun_scale + min(env.a, wp.diff.w);
        reflection = env.rgb * 256.0 * refl_alpha * refl_coef;
    } else {
        // Legacy FALLBACK: the captured sky-dome cube (a stand-in when no authored cube exists),
        // gated by the sunspot_cut ambient floor + a hand-tuned analytic sun glint.
        let cube = textureSampleLevel(env_cube, env_cube_samp, refl_dir, 2.0).rgb;
        let cube_lit = dot(cube, vec3<f32>(0.333)) > 0.002;
        let refl_src = select(sky_radiance(refl_dir), cube, cube_lit);
        reflection = refl_src * refl_coef * refl_ambient;
        let sun = normalize(light.sun_dir.xyz);
        let h = normalize(sun + view);
        let glint = pow(max(dot(n, h), 0.0), 24.0);
        reflection += vec3<f32>(1.0, 0.97, 0.9) * glint * refl_coef;   // sun-disk stays full-strength + localized
    }
    // (water_shading.hlsl_include:1029-1032): reflection == none -> color_reflection = 0.
    if (refl_none) { reflection = vec3<f32>(0.0, 0.0, 0.0); }

    // water_shading.hlsl_include:1021+1088: color_diffuse =
    // water_diffuse × saturate(dot(up,N)) × lightmap_intensity — the authored blue-green self-cast
    // modulated by the baked ambient (was a hardcoded vec3(0,0.02,0.04) ignoring both).
    // ColorLit = lerp(refracted, reflection, fresnel) + water_color_lit — the lit water colour is ADDED on top of the lerp (old compose kept it only inside → body ~half as bright and grey).

    // COMPOSITE — engine ALPHA-BLEND path (water_shading.hlsl_include:1120-1145).
    // rgb = water×(1-transparency) lerped to reflection by fresnel; coverage alpha
    // = 1 - transparency·(1-fresnel). With wgpu ALPHA_BLENDING the framebuffer (the
    // lake bed drawn in the opaque pass) shows through where coverage is low, so
    // shallow/clear water is see-through and deep/grazing water is opaque — no
    // scene-copy needed.
    // — engine composite (03_shader_water.md §3a asm 200-227), expressed for ONE/INV_SRC_ALPHA blending:
    //   refracted    = lerp(water_color_lit, scene, vis)
    //   colorFresnel = lerp(refracted, reflection, fresnel)
    //   colorLit     = colorFresnel + water_color_lit            (+ dynamic_light·water_diffuse = 0)
    //   colorCover   = lerp(scene, colorLit, bank_alpha)        (watercolor.a)
    //   final        = lerp(colorCover, foam, foamMask)
    // ⇒ src_premul = bank·[(1-fres)(1-vis)·wc + fres·refl + wc], dst_weight = bank·(1-fres)·vis + (1-bank).
    let vis = transparency;
    // (engine-exact, water_shading.hlsl_include:800/850/878): the layer ADDED on top of the
    // refraction/reflection lerp is `color_diffuse = water_diffuse * saturate(dot(sun,N)) * lightmap_intensity`
    // — the material's own authored diffuse colour, NOT a second copy of the refraction water_color. Forge
    // World's ocean authors water_diffuse (0.008, 0.031, 0.047) while its watercolor texture x 0.1 is much
    // darker and bluer, so reusing water_color would read ~20-25% too dark and too blue against
    // the engine capture.
    let sun_ndl = clamp(dot(normalize(light.sun_dir.xyz), n), 0.0, 1.0);
    let color_diffuse = wp.diff.rgb * sun_ndl * lm_intensity;
    var src = bank_alpha * (water_color * (1.0 - fresnel) * (1.0 - vis) + reflection * fresnel + color_diffuse);
    var dst_w = bank_alpha * (1.0 - fresnel) * vis + (1.0 - bank_alpha);
    var out_rgb = src; // (probe compatibility)
    let wdbg = __WDBG__; // HMS_WDBG per-term HDR probes: 1 water_color, 2 lm_intensity, 3 reflection, 4 (transparency, fresnel, nov_f)
    if (wdbg == 1) { return vec4<f32>(water_color, 1.0); }
    if (wdbg == 2) { return vec4<f32>(lm_intensity, 1.0); }
    if (wdbg == 3) { return vec4<f32>(reflection, 1.0); }
    if (wdbg == 4) { return vec4<f32>(transparency, fresnel, nov_f, 1.0); }
    if (wdbg == 5) { return vec4<f32>(n * 0.5 + 0.5, 1.0); }
    if (wdbg == 6) { return vec4<f32>(abs(slope.x), abs(slope.y), select(0.0, 1.0, underwater), 1.0); }
    if (wdbg == 8) { return vec4<f32>(color_diffuse, 1.0); }
    if (wdbg == 7) { return vec4<f32>(-neg_depth, select(0.0, 1.0, bed_ndc >= 0.9999), cam_dist, 1.0); }
    let foam_cut = wp.edge.y;
    if (foam_cut < 1.0) {
        let foam_pow = max(wp.foam.y, 1.0);
        var ff = saturate(chop - foam_cut) / max(saturate(1.0 - foam_cut), 1e-4);
        ff = pow(ff, foam_pow);
        ff = ff * wp.foam.x;                                   // foam_coefficient
        ff = ff * clamp(20.0 / max(cam_dist, 0.001), 0.0, 1.0); // distance fade
        if (ff > 0.002) {
            let foam_s = textureSample(foam_tex, water_samp, wuv * 2.5);
            let foam_rgb = pow(foam_s.rgb, vec3<f32>(2.2)) * lm_intensity;
            ff = clamp(foam_s.a * ff, 0.0, 1.0);
            src = src * (1.0 - ff) + foam_rgb * ff;
            dst_w = dst_w * (1.0 - ff);
        }
    }
    if (wp.env.z > 0.5) {
        let over = select(0.0, 1.0, chop > foam_cut);
        return vec4<f32>(clamp(chop, 0.0, 1.0), over, 0.0, 1.0);
    }
    let alpha_out = clamp(1.0 - dst_w, 0.0, 1.0);
    // Fog the premultiplied colour: src·extinction + inscatter·alpha (the destination is already fogged).
    let insc = apply_fog(vec3<f32>(0.0), i.world_pos, cam.cam_pos.xyz);
    let ext = apply_fog(vec3<f32>(1.0), i.world_pos, cam.cam_pos.xyz) - insc;
    let fogged = src * ext + insc * alpha_out;
    return vec4<f32>(fogged, alpha_out);
}
"#;

/// HMS_SKY_LEGACY=1 → the pre-audit sky path (SKY_MESH_WGSL_LEGACY, depth Always/no-write,
/// index-descending draw order, scene.rs build_sky_meshes_legacy). Cached once per process.
pub fn sky_legacy() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("HMS_SKY_LEGACY").map_or(false, |v| v != "0"))
}

/// SKY-EXACT: engine-exact Reach sky material shading (sky audit 2026-09, items D1-D7).
/// Every sky material on shipped maps is a plain `rmsh` on the generic `shaders\shader` rmdf and
/// the sky meshes carry a vertex-colour stream, so the engine pixel is `static_per_vertex_color_ps`
/// (templated__entry_points.hlsl_include:1073-1160):
///   albedo = per the template's albedo option (templated__albedo.hlsl_include; DETAIL_MULTIPLIER
///            4.59479): default base·detail·DM·albedo_color | constant_color | detail_blend |
///            three_detail_blend | two_detail base·DM²·detail·detail2 (alpha = product of alphas)
///   si     = per the self_illumination option (templated__self_illumination.hlsl_include):
///            simple si_map·color·intensity | from_albedo albedo·color·intensity (and albedo ← 0) |
///            detail si_map·si_detail·DM·color·intensity — × ILLUM_SCALE (g_alt_exposure.r; folded
///            into `gain` by scene.rs)
///   rgb    = max(0, vert_color·albedo.rgb + si)        (:1153; the sky has no simple lights)
///   alpha  = ALPHA_CHANNEL_OUTPUT per blend (shared__blend.hlsl_include:38-84): additive 0,
///            multiply 1, alpha_blend/premul albedo.w
/// Each texture samples at `uv·xform.xy + xform.zw` of ITS parameter (shared__texture_xform:27);
/// TranslationX/Y overlays animate that parameter's xform.zw only (folded in the VS). Bitmaps with
/// a Linear curve are pre-encoded at upload so the blanket pow(2.2) here round-trips to raw.
/// No fog (additive panels are never fogged; whether opaque/alpha sky panels see live atmosphere
/// constants is undetermined — audit §5), no brightness knees/caps; exposure is the global resolve.
/// The sky rides the camera (VS offsets by cam_pos). Instance lanes (scene.rs build_sky_meshes):
///   @8 gain = self_illum_color.rgb·self_illum_intensity·ILLUM_SCALE
///   @10 [base scroll.xy, detail scroll.xy]  @21 [self_illum scroll.xy, self_illum_detail scroll.xy]
///   @23 [detail2 scroll.xy, detail3 scroll.xy]   (uv units / s)
///   @11 base_xf  @15 detail_xf  @16 detail2_xf  @17 si_xf  @18 sid_xf  @24 detail3_xf
///   @19 albedo_color   @26 ctl = [albedo option, self_illum option, HMS blend enum, 0]
/// Textures: base@0, self_illum@2, detail@3, self_illum_detail@4, detail2@12, detail3@15.
pub(crate) const SKY_MESH_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;

// Engine ILLUM_SCALE (reach_tag_test sub_140838800 / sub_1407E89C0) = g_alt_exposure.r =
// 2^((1 - s)·(P - E)) with E = current exposure in stops (log2 of the resolve gain, auto-adapted & clamped to the
// cfxs band), P = cfxs self_illum_preferred exposure (default 0), s = self_illum_scale "exposure change" (default
// 0.3). Self-illum follows only 30% of the scene exposure. The gain replicates the post pass `resolve()` from the
// same 1x1 luminance meter: light.expo = [stops_gain, key, lo, hi], light.expo2 = [cal, fixed_gain, P, s].
@group(0) @binding(9) var lum_meter: texture_2d<f32>;
fn illum_scale_now() -> f32 {
    let mean_log = textureLoad(lum_meter, vec2<i32>(0, 0), 0).r;
    let hi = light.expo.w;
    let lo = min(light.expo.z, hi);
    let raw_gain = light.expo.y / max(exp2(mean_log), 1e-4);
    var gain = light.expo.x * clamp(raw_gain * light.expo2.x, lo, hi);
    if (light.expo2.y > 1e-6) { gain = light.expo2.y; }
    // E is the engine's exposure in STOPS AROUND THE KEY (the cfxs-band-clamped adaptation; the
    // resolve gain is key·ENGINE_METER_UNITS(10)·2^stops, see lib.rs set_auto_exposure). Ivory
    // capture check: ev = -1.03 → 2^(0.7·1.03) = 1.65 vs captured g_alt_exposure.r 1.619.
    let e = log2(max(gain, 1e-6) / max(light.expo.y * 10.0, 1e-4));
    return exp2((1.0 - light.expo2.w) * (light.expo2.z - e));
}
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var si_tex: texture_2d<f32>;
@group(1) @binding(3) var detail_tex: texture_2d<f32>;
@group(1) @binding(4) var si_detail_tex: texture_2d<f32>;
@group(1) @binding(12) var detail2_tex: texture_2d<f32>;
@group(1) @binding(15) var detail3_tex: texture_2d<f32>;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) gain: vec4<f32>,
    @location(10) scroll_a: vec4<f32>,
    @location(11) base_xf: vec4<f32>,
    @location(15) detail_xf: vec4<f32>,
    @location(16) detail2_xf: vec4<f32>,
    @location(17) si_xf: vec4<f32>,
    @location(18) sid_xf: vec4<f32>,
    @location(19) albedo_color: vec4<f32>,
    @location(21) scroll_b: vec4<f32>,
    @location(23) scroll_c: vec4<f32>,
    @location(24) detail3_xf: vec4<f32>,
    @location(26) ctl: vec4<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) vcolor: vec4<f32>,
    @location(2) @interpolate(flat) gain: vec4<f32>,
    @location(3) @interpolate(flat) base_xf: vec4<f32>,
    @location(4) @interpolate(flat) detail_xf: vec4<f32>,
    @location(5) @interpolate(flat) detail2_xf: vec4<f32>,
    @location(6) @interpolate(flat) detail3_xf: vec4<f32>,
    @location(7) @interpolate(flat) si_xf: vec4<f32>,
    @location(8) @interpolate(flat) sid_xf: vec4<f32>,
    @location(9) @interpolate(flat) albedo_color: vec4<f32>,
    @location(10) @interpolate(flat) ctl: vec4<f32>,
};

// TranslationX/Y overlay: xform.zw advances at rate (uv/s) — the engine animates only .zw.
fn anim(xf: vec4<f32>, rate: vec2<f32>, t: f32) -> vec4<f32> {
    return vec4<f32>(xf.xy, xf.zw + rate * t);
}

@vertex
fn vs(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let local = model * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    // Camera-follow: the dome is authored origin-centred; translate by the eye so it stays
    // fixed relative to the viewer.
    o.clip = cam.view_proj * vec4<f32>(local.xyz + cam.cam_pos.xyz, 1.0);
    o.uv = v.uv;
    o.vcolor = v.vcolor;
    let t = cam.time.x;
    o.gain = v.gain;
    o.base_xf = anim(v.base_xf, v.scroll_a.xy, t);
    o.detail_xf = anim(v.detail_xf, v.scroll_a.zw, t);
    o.detail2_xf = anim(v.detail2_xf, v.scroll_c.xy, t);
    o.detail3_xf = anim(v.detail3_xf, v.scroll_c.zw, t);
    o.si_xf = anim(v.si_xf, v.scroll_b.xy, t);
    o.sid_xf = anim(v.sid_xf, v.scroll_b.zw, t);
    o.albedo_color = v.albedo_color;
    o.ctl = v.ctl;
    return o;
}

// transform_texcoord + sample; rgb to linear (Linear-curve maps were pre-encoded at upload).
fn tex(t: texture_2d<f32>, uv: vec2<f32>, xf: vec4<f32>) -> vec4<f32> {
    let s = textureSample(t, samp, uv * xf.xy + xf.zw);
    return vec4<f32>(pow(s.rgb, vec3<f32>(2.2)), s.a);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let DM = 4.59479;   // DETAIL_MULTIPLIER (templated__albedo.hlsl_include:7)
    // Sample everything up front (uniform control flow); unused slots hold 1x1 neutrals.
    let base = tex(base_tex, i.uv, i.base_xf);
    let d1 = tex(detail_tex, i.uv, i.detail_xf);
    let d2 = tex(detail2_tex, i.uv, i.detail2_xf);
    let d3 = tex(detail3_tex, i.uv, i.detail3_xf);
    let sim = tex(si_tex, i.uv, i.si_xf);
    let sid = tex(si_detail_tex, i.uv, i.sid_xf);
    let opt = i32(round(i.ctl.x));
    let si_mode = i32(round(i.ctl.y));
    let blend = i32(round(i.ctl.z));
    let ac = i.albedo_color;
    // --- albedo (templated__albedo.hlsl_include) ---
    var albedo: vec4<f32>;
    switch opt {
        case 2: {   // constant_color (:64-72)
            albedo = ac;
        }
        case 7: {   // two_detail (:282-293)
            albedo = vec4<f32>(base.rgb * (DM * DM) * d1.rgb * d2.rgb, base.a * d1.a * d2.a);
        }
        case 1: {   // detail_blend (:147-163; blend_alpha = 1)
            let m = mix(d1, d2, base.a);
            albedo = vec4<f32>(DM * base.rgb * m.rgb, m.a);
        }
        case 5: {   // three_detail_blend (:172-194)
            let b1 = saturate(2.0 * base.a);
            let b2 = saturate(2.0 * base.a - 1.0);
            let f = mix(d1, d2, b1);
            let sb = mix(f, d3, b2);
            albedo = vec4<f32>(DM * base.rgb * sb.rgb, sb.a);
        }
        default: {  // 0 default (:112-138); other options (change_color etc.) never authored on skies
            albedo = vec4<f32>(base.rgb * (d1.rgb * DM) * ac.rgb, base.a * d1.a * ac.a);
        }
    }
    // --- self illumination (templated__self_illumination.hlsl_include); gain carries ILLUM_SCALE ---
    var si = vec3<f32>(0.0);
    switch si_mode {
        case 0: { }                                        // off
        case 4: {                                          // from_albedo (:107-119): albedo ← 0
            si = albedo.rgb * i.gain.rgb;
            albedo = vec4<f32>(0.0, 0.0, 0.0, albedo.a);
        }
        case 5: {                                          // illum_detail (:127-142)
            si = sim.rgb * (sid.rgb * DM) * i.gain.rgb;
        }
        case 8: {                                          // simple_with_alpha_mask (:33-45)
            si = sim.rgb * sim.a * i.gain.rgb;
        }
        default: {                                         // 1 simple (:19-31); others treated as simple
            si = sim.rgb * i.gain.rgb;
        }
    }
    // --- compose (entry_points:1150-1155) ---
    // Si × ILLUM_SCALE (g_alt_exposure.r, per frame from the exposure meter)
    var rgb = max(vec3<f32>(0.0), i.vcolor.rgb * albedo.rgb + si * illum_scale_now());
    if (blend == 3) { rgb = rgb * 2.0; }   // double_multiply: BLEND_MULTIPLICATIVE 2.0
    var alpha = 1.0;                       // opaque (velocity alpha in-engine) / multiply 1.0
    if (blend == 1 || blend == 5 || blend == 7) { alpha = 0.0; }          // additive
    if (blend == 4 || blend == 6 || blend == 8) { alpha = albedo.a; }     // alpha_blend / premul
    // Halo 4 `srf_constant*` alpha lane (ctl.w = 1; Reach passes 0): the shipped
    // srf_constant_vertalpha PS = lerp(vertex.a, vertex.a * color.a, p2.z), plain srf_constant =
    // color.a (gain.w = 1, vcolor.a = 1). Blend 9 = add_src_times_srcalpha (Forge Island's
    // `mp_hillside_fake_haze`): rgb * alpha added, One/One.
    if (i.ctl.w > 0.5) {
        let a4 = i.vcolor.a * mix(1.0, albedo.a, i.gain.w);
        if (blend == 4) { alpha = a4; }
        if (blend == 9) { rgb = rgb * a4; }
    }
    if (blend == 9) { alpha = 0.0; }
    return vec4<f32>(rgb, alpha);
}
"#;

/// LEGACY (HMS_SKY_LEGACY=1) Reach sky render_model shading. UNLIT: albedo (× authored albedo
/// tint) + self-illum (baked into emissive). The sky rides the camera (VS offsets every
/// vertex by cam_pos) so it never gets nearer, and is NOT fogged. Reuses the
/// standard MeshVertex + instance layout and the material bind group
/// (base@0, samp@1, emissive@2, detail@3).
pub(crate) const SKY_MESH_WGSL_LEGACY: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
// Fog uniform (shared camera bind group, binding 1) — used ONLY for its sky
// inscatter colour (atm0.rgb) to grade the sky model's lower edge into the horizon.
struct Fog { atm0: vec4<f32>, atm1: vec4<f32>, atm2: vec4<f32>, atm3: vec4<f32>, atm4: vec4<f32>, atm5: vec4<f32>, atm6: vec4<f32> };
@group(0) @binding(1) var<uniform> fog: Fog;
// (Tier 5): the light uniform is bound to the sky pass too (camera_bg, binding 2) —
// gives the sky shader the REAL sun direction (sun_dir.xyz, to-sun) + HDR colour (sun_tint.rgb)
// for the analytic sun disc (sky_dome_simple.hlsl).
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_samp: sampler;
@group(1) @binding(2) var emis_tex: texture_2d<f32>;
@group(1) @binding(3) var detail_tex: texture_2d<f32>;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(9) sway: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
    @location(10) scroll: vec4<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) dir_z: f32,   // normalized view-to-vertex Z (down = negative)
    @location(3) scroll: vec2<f32>,   // animated UV scroll rate/sec (clouds); 0 = static
    @location(4) tile: vec2<f32>,     // base_map_xform.xy UV tile (sky texture scale)
    @location(5) wdir: vec3<f32>,     // normalized world dome direction
};

@vertex
fn vs(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let local = model * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    // Camera-follow: the dome is authored origin-centred; translate by the eye so
    // it stays fixed relative to the viewer (engine UpdateSkyAnchor).
    o.clip = cam.view_proj * vec4<f32>(local.xyz + cam.cam_pos.xyz, 1.0);
    o.uv = v.uv;
    o.tint = v.tint * v.vcolor;
    let d = local.xyz;
    o.dir_z = d.z / max(length(d), 1e-4);
    o.wdir = d / max(length(d), 1e-4);   // world dome direction
    o.scroll = v.scroll.xy;
    o.tile = v.scroll.zw;   // base_map tile packed into scroll.zw (1,1 = untiled)
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    // Apply the authored base_map UV tile (sky texture SCALE — was never applied, so
    // tiled cloud/atmosphere panels rendered stretched), then the animated scroll.
    let auv = i.uv * i.tile + i.scroll * cam.time.x;
    let s = textureSample(base_tex, base_samp, auv);
    let base = pow(s.rgb, vec3<f32>(2.2)) * i.tint.rgb;
    let emis_s = textureSample(emis_tex, base_samp, auv);
    let emis = pow(emis_s.rgb, vec3<f32>(2.2));
    // When a real self-illum map is present it IS the color (planet/nebula/stars); the
    // diffuse is often just a silhouette MASK (e.g. reach_planet_mask) whose white RGB must
    // NOT be added — adding base+emis blew the planet to a solid white sphere. Use emissive
    // alone when present; otherwise the diffuse albedo (backplate/constant-tint panels).
    let has_emis = dot(emis, vec3<f32>(1.0)) > 0.003;
    // SKY EDGE FADE: additive self-illum cloud/wisp cards (cumulonimbus, mountains_wisps, stars)
    // add their emissive across the WHOLE card mesh, so a texture with content to its edge shows a
    // hard rectangle floating in the sky (the "non-blended rectangle"). The engine composites these
    // as PRE-MULTIPLIED-alpha additive: emissive × texture coverage, so the card dissolves at its
    // soft (low-alpha) edges. Multiply the emissive by its own alpha. NOT applied to the opaque
    // dome backplate (tint.a>0.5), whose alpha channel is an unused/opaque mask.
    // smoothstep so the very-low-alpha card EDGES drop fully to transparent (killing the faint
    // residual band a plain ×alpha leaves), while cloud cores (alpha ≳0.4) stay full strength.
    let alpha_cover = smoothstep(0.06, 0.42, emis_s.a);
    // Some cloud cards (cumulonimbus_backlit) are FULL-COVERAGE sheets — alpha≈1 to the card's
    // geometry edge — so the alpha fade above can't soften them and they read as a hard rectangle
    // floating in the sky. Feather the card's OWN UV border (i.uv spans 0..1 across the card) so
    // the additive contribution ramps to zero at the card silhouette. NOT applied to the opaque
    // dome (tint.a>0.5), whose UV wraps a full sphere (a border fade there would seam).
    let b = 0.10;
    let border = smoothstep(0.0, b, i.uv.x) * smoothstep(0.0, b, 1.0 - i.uv.x)
               * smoothstep(0.0, b, i.uv.y) * smoothstep(0.0, b, 1.0 - i.uv.y);
    let cover = select(alpha_cover * border, 1.0, i.tint.a > 0.5);
    // Emissive panels carry their HDR self_illum gain (self_illum_color × intensity, e.g.
    // blue_sky_dome 2.0) in tint.rgb — multiply it in AFTER linearization so the dome/clouds/stars
    // reach their authored HDR brightness for the post tonemap to range (was clamped to the raw
    // u8 bitmap → flat/dim sky). Non-emissive panels already fold tint.rgb into `base` above.
    var col = select(base, emis * i.tint.rgb * cover, has_emis);
    // Guard against UNNATURAL MAGENTA sky. Countdown's additive `facility_fancy_cloud`
    // twilight cloud has texels where green collapses far below red AND blue (~144,42,190) —
    // magenta, not a natural pink/violet twilight — which then reflects off the metallic
    // silo (env=2.0) as "magenta pipes". Lift green to a floor of the smaller of R/B so the
    // hue reads as violet/pink instead of magenta. Fires ONLY on genuinely magenta texels
    // (g far below min(r,b)); ordinary blue/white/orange skies are untouched.
    let mn_rb = min(col.r, col.b);
    if (col.g < mn_rb * 0.55) { col.g = mn_rb * 0.55; }
    // Aerial-perspective grade: the sky-model distant mountains are drawn UNFOGGED,
    // so their bases meet the fogged BSP at a hard seam (the "fog wall"). Blend the
    // sky toward the fog SKY inscatter colour as the view drops toward/below the
    // horizon (dir_z: ~0 at horizon, negative below), so the mountain bases dissolve
    // into the same haze the terrain fogs to. Off above the horizon (open sky/clouds).
    if (fog.atm3.w > 0.5) {
        let h = clamp((0.12 - i.dir_z) / 0.30, 0.0, 1.0); // 0 high, 1 at/below horizon
        // Light (0.15): the engine does NOT fog the sky dome (RE 13 §5.5) — the authored horizon
        // panels ARE the haze — so this is only a thin seam-hider where the sky-model mountains
        // meet the fogged BSP.
        col = mix(col, fog.atm0.rgb, h * 0.15);
    }
    // No analytic sun disc: the `pow(saturate(dot(sunDir,dir)),exp)·sunColor` term of
    // `sky_dome_simple.hlsl` is DEAD on PC/MCC — it reads cbuffer constants
    // `p_lighting_constant_0/1` which are written ONLY by setup_sun_constants (@0x1806912E0, from
    // render_decorators, which runs AFTER the sky pass), so for the dome they are left ZERO, the
    // exponent is 0 and the term collapses; the dome PS reduces to `vert_color·g_exposure`. The
    // visible in-game sun is the authored `cumulonimbus_backlit` cloud panel (an additive
    // self-illum card drawn here) plus the screen-space lens-flare tag.
    return vec4<f32>(col, s.a);
}
"#;

/// Depth-only shadow VS: transform each caster vertex by the sun's light_view_proj
/// (× per-instance model). No fragment stage — the pass writes depth only.
pub(crate) const SHADOW_WGSL: &str = r#"
struct Light { light_view_proj: mat4x4<f32>, sun_dir: vec4<f32>, sun_tint: vec4<f32>, ambient_tint: vec4<f32>, sky_ambient: vec4<f32>, dbg: vec4<f32>, expo: vec4<f32>, expo2: vec4<f32> };
@group(0) @binding(2) var<uniform> light: Light;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(9) sway: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.pos, 1.0);
    return light.light_view_proj * world;
}
"#;

/// shader_terrain multi-layer blend. Port of the engine terrain intent: 4 base
/// colour maps, each UV-tiled per-layer, blended by the RGBA weights of a blend
/// mask (r→layer0 … a→layer3), then lit + fogged like the standard mesh path.
pub(crate) const TERRAIN_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32>, cam_pos: vec4<f32>, time: vec4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var base0: texture_2d<f32>;
@group(1) @binding(1) var base1: texture_2d<f32>;
@group(1) @binding(2) var base2: texture_2d<f32>;
@group(1) @binding(3) var base3: texture_2d<f32>;
@group(1) @binding(4) var blend_tex: texture_2d<f32>;
@group(1) @binding(5) var samp: sampler;
// Terrain uniform (23×vec4). base_xform[0..4], detail_xform[4..8] (each xy=scale,
// zw=offset, engine transform_texcoord), blend_xform[8], global_tint[9],
// active[10] (per-layer 0/1), detail_present[11] (per-layer 0/1), bump_xform[12..16],
// bump_present[16]. [17]=(dbb_mode,dbb_slope,dbb_offset,_),
// [18..22]=blend_target[0..4] (per-layer target colour), [22]=blend_max[0..4].
// [23..27]=detail_bump_xform (per layer, xy=scale zw=offset), [27]=detail_bump_present.
struct Tiles { t: array<vec4<f32>, 32> };
@group(1) @binding(6) var<uniform> tiles: Tiles;
// per-instance DM/SDM lightmap atlas (raw-BC DM + decoded SDM slices) sampled in the
// FS exactly like mesh_shade's gpu_lightmap_tint. Previously the terrain pipeline had NO atlas bindings,
// so a terrain mesh routed to the GPU-lightmap path (has_pvl=true, v.color=[1,1,1,1]) was lit by a FLAT
// WHITE baked term — the root cause of the dark/flat/grainy forge rock terrain. lm.x>0.5 activates.
@group(1) @binding(19) var tlm_dm: texture_2d<f32>;
@group(1) @binding(20) var tlm_sdm: texture_2d<f32>;
@group(1) @binding(21) var tlm_samp: sampler;
@group(1) @binding(22) var tlm_sdm1: texture_2d<f32>;
@group(1) @binding(23) var tlm_sdm2: texture_2d<f32>;
// byte-exact vMF diffuse LUT (camera group, shared with mesh_shade)
@group(0) @binding(7) var vmf_lut: texture_2d<f32>;
@group(0) @binding(8) var vmf_samp: sampler;
struct BakeLobes { total: vec4<f32>, fill: vec3<f32> };
// The terrain copy of the dual-VMF atlas decode: the same engine law as the mesh shader's
// `gpu_lightmap_tint` (see there for the DecompressVMF derivation), against this module's own
// DM/SDM bindings.
fn t_lightmap_tint(uv2: vec2<f32>, n: vec3<f32>, hdr: f32, k: f32) -> BakeLobes {
    let d  = textureSampleLevel(tlm_dm,   tlm_samp, uv2, 0.0);
    let s0 = textureSampleLevel(tlm_sdm,  tlm_samp, uv2, 0.0);
    let s1 = textureSampleLevel(tlm_sdm1, tlm_samp, uv2, 0.0);
    let s2 = textureSampleLevel(tlm_sdm2, tlm_samp, uv2, 0.0);
    var dom_dir = vec3<f32>(s0.w * 2.0 - 1.0, s1.w * 2.0 - 1.0, s2.w * 2.0 - 1.0);
    let dl = length(dom_dir);   // lobe sharpness (bandwidth) = |unnormalized dir|, the vMF LUT y
    if (dl > 1e-4) { dom_dir = dom_dir / dl; } else { dom_dir = vec3<f32>(0.0, 0.0, 1.0); }
    let f_int = exp2(-9.000001 * d.x);
    let dom_col  = (s0.rgb + s1.rgb * 2.0 - 1.0) * f_int;  // dominant lobe colour (signed)
    let fill_col = s2.rgb * f_int;                          // fill lobe colour (signed)
    let vis = d.y;                                          // visibility = DM.GREEN
    let cc = clamp(dot(dom_dir, n) * 0.5 + 0.5, 0.0, 1.0);
    let dom_co = textureSampleLevel(vmf_lut, vmf_samp, vec2<f32>(cc, clamp(dl, 0.0, 1.0)), 0.0).r;
    let inv_pi = 0.31830989;
    var tint = vec3<f32>(0.0, 0.0, 0.0);
    var fill = vec3<f32>(0.0, 0.0, 0.0);
    for (var c = 0; c < 3; c = c + 1) {
        let irr = (dom_co * dom_col[c] + 0.25 * fill_col[c]) * inv_pi;
        tint[c] = max(irr * hdr * k, 0.0);
        fill[c] = 0.25 * fill_col[c] * inv_pi * hdr * k;
    }
    return BakeLobes(vec4<f32>(tint, vis), fill);
}

@group(1) @binding(7) var detail0: texture_2d<f32>;
@group(1) @binding(8) var detail1: texture_2d<f32>;
@group(1) @binding(9) var detail2: texture_2d<f32>;
@group(1) @binding(10) var detail3: texture_2d<f32>;
@group(1) @binding(11) var bump0: texture_2d<f32>;
@group(1) @binding(12) var bump1: texture_2d<f32>;
@group(1) @binding(13) var bump2: texture_2d<f32>;
@group(1) @binding(14) var bump3: texture_2d<f32>;
// Per-layer detail_bump (2nd normal) maps.
@group(1) @binding(15) var dbump0: texture_2d<f32>;
@group(1) @binding(16) var dbump1: texture_2d<f32>;
@group(1) @binding(17) var dbump2: texture_2d<f32>;
@group(1) @binding(18) var dbump3: texture_2d<f32>;

// DETAIL_MULTIPLIER (HREK terrain_new.hlsl:320): base maps are deliberately near-gray and
// the fine-tiled detail map supplies the real texture, re-brightened by this constant so an
// average (linear 0.2176) detail texel is neutral.
const DETAIL_MULT: f32 = 4.59479;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vcolor: vec4<f32>,
    @location(9) sway: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
    @location(13) uv2: vec2<f32>,   // submap-local lightmap UV
    @location(14) lm: vec4<f32>,    // [flag,hdr,k,mode]
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) vcolor: vec4<f32>,
    @location(3) vdepth: f32,
    @location(4) world_pos: vec3<f32>,
    @location(5) uv2: vec2<f32>,
    @location(6) lm: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    o.clip = cam.view_proj * world;
    o.normal = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    o.uv = v.uv;
    o.vcolor = v.vcolor;
    o.vdepth = o.clip.w;
    o.world_pos = world.xyz;
    o.uv2 = v.uv2;
    o.lm = v.lm;
    return o;
}

// One terrain layer: base×detail×DETAIL_MULT when a detail map is present,
// base-only otherwise (engine treats absent detail as neutral — applying
// DETAIL_MULT there would make the layer ~4.6× too bright). Base & detail sampled
// through their OWN engine UV transform (uv*scale + offset), not a shared guess.
fn terrain_layer(base_c: vec3<f32>, det_c: vec3<f32>, present: f32, base_exp: f32, det_exp: f32) -> vec3<f32> {
    // The engine samples each texture through the format its authored CURVE selects
    // (ZH_BitmapInfo.Curve, protomorph TexelEncoding::from_curve): Gamma2/Srgb/Unknown →
    // gamma/sRGB decode (≈pow 2.2), Linear/OffsetLog → RAW (no decode). base_exp/det_exp carry
    // the per-layer decode exponent (1.0 = linear/raw, 2.2 = gamma/sRGB). A blanket pow(2.2)
    // would crush the linear-authored overlay bases (forge grass_overlay/dirt_overlay/
    // overlay_beach are curve=1 linear) ~3× too dark on the grass/dirt paths.
    let b = pow(base_c, vec3<f32>(base_exp));
    // DETAIL_MULTIPLIER 4.59479 = 1/pow(0.5,2.2) is mean-preserving only with a LINEARIZED detail
    // (mid-gray sRGB byte 128 → linear 0.2176 → ×4.59479 = 1.0); a raw-sRGB detail (neutral ~0.5)
    // runs every layer ~2.3× hot (green grass washes to sandy khaki).
    let d = pow(det_c, vec3<f32>(det_exp));
    // terrain_new.hlsl: the engine ALWAYS does base·detail·MULT — a no-detail layer samples the
    // shader's DEFAULT detail bitmap (default_detail=0.21961 linear), so base-only ≈
    // base·0.21961·4.59479 = base·1.009 (NOT bare base). No distance fade (the mip chain
    // handles minification; a fade-to-neutral would discard the detail's colour, which is where
    // the brown/green of the ground lives — the base "overlay" maps are desaturated tan).
    return select(b * 1.009, b * d * DETAIL_MULT, present > 0.5);
}

// Plain hardware auto-mip (textureSampleGrad on the tiled uv's own derivatives) for ALL layers,
// like the engine (protomorph entry_albedo_terrain.wgsl); the detail's correct mip carries its
// authored colour. Sampled through explicit gradients so the call is valid in any control flow.
fn aa_sample(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleGrad(t, samp, uv, dpdx(uv), dpdy(uv));
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let bx0 = tiles.t[0]; let bx1 = tiles.t[1]; let bx2 = tiles.t[2]; let bx3 = tiles.t[3];
    let dx0 = tiles.t[4]; let dx1 = tiles.t[5]; let dx2 = tiles.t[6]; let dx3 = tiles.t[7];
    let blend_x = tiles.t[8];
    // global_albedo_tint is a SCALAR float in the engine (PARAM(float,…), terrain_new.hlsl_include:67)
    // — only .x is meaningful; the engine splats it to rgb (`albedo.xyz *= global_albedo_tint`).
    // t[9].w carries uv_density; splat .x to all channels to match the engine contract.
    let global_tint = vec3<f32>(tiles.t[9].x);
    let layer_active = tiles.t[10];
    let present = tiles.t[11];
    // Per layer: base × detail × DETAIL_MULT (each decoded per its curve), sampled at
    // uv*xform.xy + xform.zw (engine transform_texcoord).
    // distance_blend_base (terrain_new.hlsl:102-129, blend_type == distance_blend_base):
    // base_blend = saturate(dist*blend_slope + blend_offset), then per layer BEFORE base*detail:
    // amount = min(base_blend, blend_max_N); base = lerp(base, blend_target_N, amount). In the
    // default morph mode base_blend is forced 0 so the lerp is a no-op. dbb_mode (t[17].x)
    // gates it (1 = apply, 0 = morph).
    let dbb_mode = tiles.t[17].x;
    let dbb_dist = length(cam.cam_pos.xyz - i.world_pos);
    let base_blend = select(0.0, clamp(dbb_dist * tiles.t[17].y + tiles.t[17].z, 0.0, 1.0), dbb_mode > 0.5);
    let b0 = mix(aa_sample(base0, i.uv * bx0.xy + bx0.zw).rgb, tiles.t[18].rgb, min(base_blend, tiles.t[22].x));
    let b1 = mix(aa_sample(base1, i.uv * bx1.xy + bx1.zw).rgb, tiles.t[19].rgb, min(base_blend, tiles.t[22].y));
    let b2 = mix(aa_sample(base2, i.uv * bx2.xy + bx2.zw).rgb, tiles.t[20].rgb, min(base_blend, tiles.t[22].z));
    let b3 = mix(aa_sample(base3, i.uv * bx3.xy + bx3.zw).rgb, tiles.t[21].rgb, min(base_blend, tiles.t[22].w));
    // per-layer decode exponents (t[28]=base, t[29]=detail; 1.0 linear/raw, 2.2 gamma).
    let base_exp = tiles.t[28];
    let det_exp = tiles.t[29];
    let c0 = terrain_layer(b0, aa_sample(detail0, i.uv * dx0.xy + dx0.zw).rgb, present.x, base_exp.x, det_exp.x);
    let c1 = terrain_layer(b1, aa_sample(detail1, i.uv * dx1.xy + dx1.zw).rgb, present.y, base_exp.y, det_exp.y);
    let c2 = terrain_layer(b2, aa_sample(detail2, i.uv * dx2.xy + dx2.zw).rgb, present.z, base_exp.z, det_exp.z);
    let c3 = terrain_layer(b3, aa_sample(detail3, i.uv * dx3.xy + dx3.zw).rgb, present.w, base_exp.w, det_exp.w);
    // Blend weights: sample the mask at its OWN xform (engine samples blend at
    // uv*blend_map_xform, not bare uv), then normalise over ACTIVE layers only
    // (engine does blend /= sum(active channels) — no 1-r-g-b spillover).
    var w = textureSample(blend_tex, samp, i.uv * blend_x.xy + blend_x.zw);
    w += vec4<f32>(1e-8);   // DX11 kills pure-black pixels (terrain_new:163)
    // ACTIVE-LAYER MASK (bake-authored ∧ base-resolved): an UNUSED layer's base
    // falls back to white; if the blend mask still carries weight in that channel
    // (esp. .a stamped 1.0 while layer 3 is unbound) it blends WHITE into every
    // pixel — the "white film". Zero unused layers in BOTH numerator and
    // denominator so they never contribute and never dilute the real layers.
    // Engine normalize: weights = (mask+1e-8)·active / Σ(active). NO hard "snap to
    // layer 0 when the sum is tiny" — that snapped whole shadowed/blend-dark regions
    // to grass, blotching the dirt path (terrain RE D2). The +1e-8 epsilon already
    // yields an even split of the active layers when the mask is ~black.
    var weights = vec4<f32>(w.r, w.g, w.b, w.a) * layer_active;
    let wsum = max(weights.x + weights.y + weights.z + weights.w, 1e-8);
    weights = weights / wsum;
    // ENGINE near-zero layer cutoff (terrain_new.hlsl:336 `if (w_N > 0.04)`): the engine
    // simply SKIPS a layer whose normalized weight ≤ 0.04 — a perf skip of a near-invisible
    // layer — and does NOT re-normalize afterward (the dropped ≤4% just goes missing). An
    // earlier HMS version re-normalized the survivors back to sum=1.0 (terrain RE): that was
    // a FABRICATED edge-sharpener the engine has no equivalent of — it compressed the dirt→
    // grass transition band (too much contrast) AND lifted every near-edge pixel toward full
    // brightness (path/grass boundary too bright). Drop the skipped layer, keep the denominator.
    weights = select(vec4<f32>(0.0), weights, weights > vec4<f32>(0.04));
    var albedo = c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w;
    // global_albedo_tint (engine multiplies the composited albedo; default 1,1,1).
    albedo *= global_tint;
    // The terrain albedo is written to the same 8-bit albedo G-buffer → clamped to 1
    // (see mesh_shade). Bright layers (snow, pale rock × detail × 4.59) cannot exceed white.
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    // Engine compose: shaded = albedo × (baked irradiance + analytical sun). The
    // per-vertex baked lightmap (i.vcolor — absolute K-Reinhard irradiance incl.
    // baked GI/AO/cliff shadows) is the DOMINANT term (no white ambient floor). The
    // sun is a colored linear term gated by the real-time cast shadow; baked PVL
    // stays ungated (it carries its own sun/shadow).
    // BUMP/NORMAL MAPPING — perturb the flat geometric normal with the per-layer bump
    // (normal) maps, accumulated by blend weight, so cliffs/rock get real surface RELIEF
    // instead of reading flat/painted (terrain_new.hlsl §3c). Bump maps store the
    // tangent-space normal xy in RG; z is reconstructed. Tangent frame is derived from
    // screen-space world-pos/uv derivatives (the terrain mesh carries no tangents).
    let bpx0 = tiles.t[12]; let bpx1 = tiles.t[13]; let bpx2 = tiles.t[14]; let bpx3 = tiles.t[15];
    let bpres = tiles.t[16];
    // terrain_new.hlsl_include:84,214-216: DETAIL_BUMP_ENABLED = ACTIVE_MATERIAL_COUNT<4.
    // When enabled the engine adds a SECOND per-layer normal map (detail_bump_m_N) to the layer's
    // bump.xy UNWEIGHTED (bump.xy += detail_bump.xy) INSIDE calc_bumpmap — i.e. before the per-layer
    // blend weight is applied — so the combined (bump+detail_bump) is what gets ×blend_amount in the
    // accumulate. layer_active (t[10]) is the runtime active mask; its count proxies the compiled
    // ACTIVE_MATERIAL_COUNT (detail_bump bitmaps only resolve for <4-layer terrain templates anyway,
    // so the per-layer present flag t[27] is the real gate). dbump maps are uploaded UNSIGNED
    // (flat=0.5), so *2-1 to signed like the primary bump.
    let active_count = layer_active.x + layer_active.y + layer_active.z + layer_active.w;
    let dbump_en = select(0.0, 1.0, active_count < 3.5);
    let dbx0 = tiles.t[23]; let dbx1 = tiles.t[24]; let dbx2 = tiles.t[25]; let dbx3 = tiles.t[26];
    let dbpres = tiles.t[27] * dbump_en;
    // Bump maps tile finely too; their aliasing normals become per-pixel lighting
    // noise at distance. The mip chain (aa_sample world-footprint LOD floor) handles minification;
    // engine does NOT fade terrain bump relief with distance (terrain-blend.md D5).
    var bxy = vec2<f32>(0.0);
    // DXN (BC5) bump maps are uploaded as Bc5RgSnorm and sampled SIGNED (engine: no *2-1);
    // non-BC bumps come CPU-decoded as 0.5-centred UNORM and still need *2-1. t[30]/t[31] = per-layer
    // snorm flags for bump / detail_bump.
    let bsn = tiles.t[30]; let dsn = tiles.t[31];
    let s0r = aa_sample(bump0, i.uv * bpx0.xy + bpx0.zw).xy;
    let s1r = aa_sample(bump1, i.uv * bpx1.xy + bpx1.zw).xy;
    let s2r = aa_sample(bump2, i.uv * bpx2.xy + bpx2.zw).xy;
    let s3r = aa_sample(bump3, i.uv * bpx3.xy + bpx3.zw).xy;
    var lb0 = select(s0r * 2.0 - 1.0, s0r, bsn.x > 0.5) * bpres.x;
    var lb1 = select(s1r * 2.0 - 1.0, s1r, bsn.y > 0.5) * bpres.y;
    var lb2 = select(s2r * 2.0 - 1.0, s2r, bsn.z > 0.5) * bpres.z;
    var lb3 = select(s3r * 2.0 - 1.0, s3r, bsn.w > 0.5) * bpres.w;
    let d0r = aa_sample(dbump0, i.uv * dbx0.xy + dbx0.zw).xy;
    let d1r = aa_sample(dbump1, i.uv * dbx1.xy + dbx1.zw).xy;
    let d2r = aa_sample(dbump2, i.uv * dbx2.xy + dbx2.zw).xy;
    let d3r = aa_sample(dbump3, i.uv * dbx3.xy + dbx3.zw).xy;
    lb0 += select(d0r * 2.0 - 1.0, d0r, dsn.x > 0.5) * dbpres.x;
    lb1 += select(d1r * 2.0 - 1.0, d1r, dsn.y > 0.5) * dbpres.y;
    lb2 += select(d2r * 2.0 - 1.0, d2r, dsn.z > 0.5) * dbpres.z;
    lb3 += select(d3r * 2.0 - 1.0, d3r, dsn.w > 0.5) * dbpres.w;
    bxy += lb0 * weights.x;
    bxy += lb1 * weights.y;
    bxy += lb2 * weights.z;
    bxy += lb3 * weights.w;
    // The engine does NOT fade terrain bump relief with distance (bump-detail-normal.md 7.2 /
    // terrain-blend.md D5; the mip chain handles minification).
    let geo_n = normalize(i.normal);
    var n = geo_n;
    let dpx = dpdx(i.world_pos); let dpy = dpdy(i.world_pos);
    let dux = dpdx(i.uv); let duy = dpdy(i.uv);
    let tdenom = dux.x * duy.y - duy.x * dux.y;
    if (abs(tdenom) > 1e-9 && dot(bxy, bxy) > 1e-6) {
        let tan = normalize((dpx * duy.y - dpy * dux.y) / tdenom);
        let bit = normalize(cross(geo_n, tan));
        // bump-detail-normal.md 7.7 / terrain_new.hlsl:390: engine z = sqrt(1-saturate(x²+y²))
        // reaches 0; the √0.05 floor capped max tilt. Drop the floor (clamp only against negatives).
        let bz = sqrt(max(1.0 - dot(bxy, bxy), 0.0));
        n = normalize(tan * bxy.x + bit * bxy.y + geo_n * bz);
    }
    // Baked lightmap = per-pixel dual-VMF atlas decode (with the bump-perturbed
    // normal, like mesh_shade) when this terrain mesh is atlas-lit; else the per-vertex i.vcolor.
    var bake = i.vcolor;
    if (i.lm.x > 0.5) { bake = t_lightmap_tint(i.uv2, n, i.lm.y, i.lm.z).total; }
    if (gi_on()) { bake = vec4<f32>(gi_irradiance(i.world_pos, n, normalize(cam.cam_pos.xyz - i.world_pos)), 1.0); }
    let sun_dir = normalize(light.sun_dir.xyz);
    let ndl = max(dot(n, sun_dir), 0.0);
    let shadow = shadow_strength(i.world_pos);
    // ENGINE SUN (terrain_new.hlsl_include:822): the analytical sun is the authored HDR colour /π,
    // gated by the real-time cast shadow and by the baked sun-visibility vmf[0].w (entry_points.hlsl:431)
    // applied TWICE (vis²; the shipped forward-lit PS ps_13963404832330100447 lines 30-37 + HREK
    // entry_points get_analytical_mask: ndl · gel(shadow) · vis · intensity · vis / π), exactly like mesh_shade.
    let bvis = clamp(bake.a, 0.0, 1.0);
    let sun_term = light.sun_tint.rgb * (0.3183 * ndl * shadow * bvis * bvis);
    // ENGINE AMBIENT (terrain_new:820, spherical_harmonics:73): the engine has NO flat hemisphere
    // ambient and NO runtime sky-ambient fill — the ambient IS the baked dual-VMF (incl. sky bounce
    // + GI + AO + baked shadows), DIRECTIONAL and carried per-vertex in i.vcolor (decoded from the
    // cluster PVL) or per-pixel from the atlas. Composite = albedo × (baked irradiance + analytical sun).
    // The engine NEVER shadows the baked dual-VMF term with the real-time shadow (the gel/shadow_mask
    // gates only the analytical sun; entry_points.hlsl `diffuse_radiance += analytical_mask·…` on top
    // of an un-shadowed dual_vmf_diffuse). Baked shadows stay baked.
    var shaded = albedo * (bake.rgb * light.dbg.z + sun_term);
    shaded = shaded + albedo * calc_simple_lights(i.world_pos, n); // dynamic point/spot lights (terrain_new default_dynamic_light_ps)
    let tdbg = __TDBG__; // HMS_TDBG per-term HDR probes (1 albedo, 2 bake, 3 sun_term, 4 ndl/shadow/bvis) — read via HMS_HDR_DUMP, pre-fog
    if (tdbg == 1) { return vec4<f32>(albedo, 1.0); }
    if (tdbg == 2) { return vec4<f32>(bake.rgb, 1.0); }
    if (tdbg == 3) { return vec4<f32>(sun_term, 1.0); }
    if (tdbg == 4) { return vec4<f32>(ndl, shadow, bvis, 1.0); }
    // No terrain specular: the engine's terrain per-layer specular needs per-layer material
    // params that are not carried (TODO), and a synthetic sheen reads as reflective grass.
    // Per-domain sceg grade (lit only), then the Reach atmospheric fog (extinction × radiance + inscatter).
    shaded = apply_fog(scene_grade(shaded), i.world_pos, cam.cam_pos.xyz);
    return vec4<f32>(shaded, 1.0);
}
"#;
