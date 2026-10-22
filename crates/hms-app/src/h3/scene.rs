//! Halo 3 (MCC) scene build - the ONE mesh builder behind the headless screenshot (render.rs)
//! and the GUI load worker (gui.rs): opens the cache, decodes every BSP's clusters + instanced
//! geometry, the scenario sky and the scenario objects, uploads them textured and lit through the
//! shared renderer and streams the batches out per lane through `emit`. #h3
//!
//! Cluster meshes (identity transform - the BSP shell) AND instanced geometry (every sbsp
//! +0x1C0 placement, resolved through its definition to an Lbsp mesh, drawn with its
//! scale/rotation/position matrix - the walls, floors and props that make a cluster-only view a
//! sparse shell) are drawn PER PART with each part's albedo (rmsh/rmtr/... `base_map` bitmap)
//! bound, one draw per (mesh, part) carrying every placement matrix of that mesh; a part whose
//! material or bitmap does not resolve binds a mid-grey placeholder albedo (not white). A malformed
//! instance (unresolved definition / mesh, non-finite transform, not-for-render flag) is skipped
//! and counted, never fatal. Each part draws in the lane its render method's template names
//! (materials.rs `blend`): additive / alpha-blend surfaces (glass, holograms, sun cards) leave
//! the opaque lane, an alpha test (foliage / grates) takes the cutout lane.
//!
//! Lighting = the map's own bake (h3::lightmaps, the engine decode): every cluster vertex and
//! per-pixel-lit instance vertex samples the Lbsp lightprobe array at its lightmap UV, every
//! per-vertex-lit instance reads its kind-6 stream, and the engine diffuse law is evaluated at
//! the (world) vertex normal into the renderer's baked vertex colour with alpha 0 (the bake holds
//! the sun; no analytic sun on top). A lit instance is its own draw (its stream is per instance);
//! an instance with no usable stream gets the BSP's mean bake. Without a lightmap the placeholder
//! sun stays. The bake is in the engine's HDR units (its exposure comes from Halo 3's own
//! auto-exposure, not ported), so each BSP's bake is normalised by a gain that lands the mean
//! exposed diffuse at BAKE_TARGET - a stand-in for that meter (the placeholder sky keeps the fixed
//! exposure `FIXED_EXPOSURE`).
//!
//! `rmtr` terrain parts (riverworld's ground) go through the renderer's terrain pass: the
//! template's `blend_map` weights its `base_map_m_N` x `detail_map_m_N` layers (sampler names
//! + texcoord transforms from materials.rs), normalised over the bound layers - the classic
//! terrain blend the Reach terrain pass implements; without it only layer 0 (a near-white
//! overlay base) showed.
//!
//! Scenario objects (objects.rs: scnr placements -> obje -> hlmt -> mode) are drawn through the
//! same per-part path with the placement matrix, lit by the map's mean bake (the objects have no
//! lightmap of their own; the engine lights them from the airprobes, not ported). The scenario
//! sky (scnr +0x34 skies -> scen -> mode) goes to the renderer's sky lane as a camera-follow
//! dome: a part with no real diffuse is SKIPPED (never a white dome); its blend mode, albedo
//! option (detail layers) and self-illum map come from the rmt2 template (materials.rs).
//!
//! Env (print-only diagnostics, the fixes are unconditional): HMS_H3_NOTEX (skip textures,
//! flat-lit), HMS_H3_NOLM (placeholder sun instead of the bake), HMS_H3_BAKE_GAIN (override the
//! gain), HMS_H3_MATDIAG (print per-material diffuse resolution), HMS_H3_INSTDIAG (print every
//! instance: name, mesh, transform, lighting, skip reason), HMS_H3_NOINST (clusters only),
//! HMS_H3_NOOBJ (no scenario objects), HMS_H3_NOSKY (no sky model), HMS_H3_NOAT (alpha-tested
//! parts stay opaque), HMS_H3_NOTERRAIN (rmtr parts through the plain first-layer path),
//! HMS_H3_OBJDIAG (print every placement + its model resolution), HMS_H3_ONLYMAT=<substring>
//! (draw only the BSP parts whose material name / class matches), HMS_EXPO_FIXED (the fixed
//! exposure).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Result};
use eframe::wgpu;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use hms_render::{GpuMesh, MeshRenderer, MeshVertex};

/// The renderer lane a batch is built for (the Halo 4 lanes + the terrain pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    Opaque,
    AlphaTest,
    Blend,
    Additive,
    Water,
    /// The scenario sky object's parts (renderer sky lane; `GpuMesh::blend_mode` carries
    /// the Reach sky blend enum, `centroid().x` the draw order).
    Sky,
    /// `rmtr` multi-layer terrain through the renderer's terrain pass (blend map weighted
    /// base x detail layers, the classic `terrain.hlsl` blend the Reach pass implements).
    Terrain,
}

use super::cache::H3Cache;
use super::geometry::{decode_mesh_with_bounds, load_bsp, Bounds, H3Bsp};
use super::lightmaps::{cluster_lightmap_uvs, load_lightmap, pervertex_probe, pervertex_stream, shade, uv_stream, H3Lightmap, LightingMode, PERVERTEX_STRIDE};
use super::materials::{load_material, H3Blend, H3Material};
use hms_render::TerrainMaterialParams;
use super::objects::{decode_model_mesh, load_model, load_placements, model_of, model_vertex_colors, placement_matrix, OFF_SCNR_SKIES, SCNR_SKY_ELEM};

/// Mean exposed bake diffuse the per-BSP gain aims for (the auto-exposure stand-in).
pub const BAKE_TARGET: f32 = 0.8;
/// The fixed scene exposure every Halo 3 map renders at (no cfxs meter port yet); the bake
/// gain is solved against it, so headless and GUI must use the same value.
pub const FIXED_EXPOSURE: f32 = 0.25;

/// The opened cache + the open-time log lines (+ the map variant riding on this load).
pub struct H3Loaded {
    pub cache: Arc<H3Cache>,
    pub path: String,
    pub log: Vec<String>,
    /// A Halo 3 `.mvar` whose Forge objects `build_meshes` places (mvar.rs). #h3
    pub variant: Option<super::mvar::H3Variant>,
    /// With no variant: the shipped `.mvar` of this map the editor starts a blank canvas from
    /// (scenario.rs `find_template`). #h3-scenario
    pub canvas: Option<(std::path::PathBuf, super::mvar::H3Variant)>,
    /// The variant's (or canvas's) scenario slots are editor objects: the GUI build leaves the
    /// base map's own vehicles / weapons / equipment / scenery / crates out of the static scenery.
    pub scenario_owned: bool,
}

/// Open a Halo 3 cache (no GPU work).
pub fn load_map(map_path: &str) -> Result<H3Loaded> {
    let cache = Arc::new(H3Cache::open(std::path::Path::new(map_path))?);
    let log = vec![format!("h3: {} ({}) map='{}' scenario='{}' tags={} classes={}",
        map_path, cache.engine.label(), cache.map_name, cache.scenario, cache.tag_count(), cache.class_count())];
    Ok(H3Loaded { cache, path: map_path.to_string(), log, variant: None, canvas: None, scenario_owned: false })
}

/// Open a Halo 3 cache with a map variant to place (a variant whose base map is another map is
/// refused: its quotas index THIS map's tag table).
pub fn load_map_with_variant(map_path: &str, variant: Option<&std::path::Path>) -> Result<H3Loaded> {
    let mut loaded = load_map(map_path)?;
    if let Some(vp) = variant {
        let v = super::mvar::parse_h3_variant(vp)?;
        let map_id = crate::mapcat::read_map_id(std::path::Path::new(map_path));
        if map_id.is_some() && map_id != Some(v.map_id) {
            anyhow::bail!("{}: base map id {} is not this map's ({:?})", vp.display(), v.map_id, map_id);
        }
        loaded.log.push(format!("h3: variant '{}' by {} ({}): map {} v{}, {} scenario / {} variant slots, {} quotas, {} placed objects, budget {} / {}",
            v.header.name, v.header.author, vp.display(), v.map_id, v.version, v.num_scenario_objects, v.num_variant_objects, v.num_quotas,
            v.placed().count(), v.spent_budget, v.maximum_budget));
        loaded.scenario_owned = super::scenario::is_editable(&v, &super::scenario::scenario_layout(&loaded.cache));
        loaded.variant = Some(v);
    } else if let Some(id) = crate::mapcat::read_map_id(std::path::Path::new(map_path)) {
        // no variant: a blank canvas from one of the map's own, so its objects can be edited
        if let Some((tpl, v)) = super::scenario::find_template(id, None).and_then(|p| super::mvar::parse_h3_variant(&p).ok().map(|v| (p, v))) {
            loaded.scenario_owned = super::scenario::is_editable(&v, &super::scenario::scenario_layout(&loaded.cache));
            loaded.log.push(format!("h3: blank canvas from {} ({} scenario objects{})", tpl.file_name().unwrap_or_default().to_string_lossy(),
                v.num_scenario_objects, if loaded.scenario_owned { ", editable" } else { ", static: this map's variants register none" }));
            loaded.canvas = Some((tpl, v));
        }
    }
    Ok(loaded)
}

/// The fixed exposure in force (HMS_EXPO_FIXED overrides, diagnostics only).
pub fn fixed_exposure() -> f32 {
    std::env::var("HMS_EXPO_FIXED").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(FIXED_EXPOSURE)
}

/// The placeholder sun for a BSP without a bake (and the sky lane's light direction).
pub fn placeholder_sun_dir() -> Vec3 { Vec3::new(0.3, 0.4, 0.85).normalize() }

/// The fog uniform a Halo 3 map renders with: no atmosphere fog (the H3 `fogg`/sky fog is not
/// ported) and the renderer's own default horizon tint in fog[0..3] (the outdoor gradient the
/// placeholder sky paints). The GUI must push this explicitly - the previous map's fog would
/// leak otherwise - and headless pushes the same so the two agree.
pub fn default_fog() -> [f32; 28] {
    let mut f = [0.0f32; 28];
    f[0] = 0.21; f[1] = 0.51; f[2] = 0.91;
    f
}

/// What the GUI editor scene takes from a build (edit_scene.rs): the BSP triangle soup every
/// drawn cluster / instance triangle went into (picking, `raycast_scene`, the camera stand-off),
/// the objects' flat light and the texture context (so the editor's uploads share the build's
/// texture cache). #h3-edit
pub struct H3EditorCollect {
    pub soup: crate::decal_projector::TriSoup,
    pub obj_color: [f32; 4],
    pub(super) tex: Option<TexCtx>,
}

impl H3EditorCollect {
    pub fn new() -> Self { H3EditorCollect { soup: crate::decal_projector::TriSoup::new(4.0), obj_color: [1.0; 4], tex: None } }
}

/// One decoded render model (its lod0 / default-permutation meshes in model space) for the
/// editor's model cache: the static object path's per-model decode, kept.
pub struct H3ModelGeom {
    pub name: String,
    /// Render-method tag per material slot.
    pub materials: Vec<Option<usize>>,
    pub meshes: Vec<H3GeomMesh>,
    pub min: Vec3,
    pub max: Vec3,
}

pub struct H3GeomMesh {
    pub verts: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<super::geometry::Part>,
}

impl H3ModelGeom {
    pub fn tri_count(&self) -> usize { self.meshes.iter().map(|m| m.indices.len() / 3).sum() }
}

/// `load_model` + `decode_model_mesh` over the meshes an object shows in hlmt variant
/// `variant` (attach.rs `variant_meshes`; None = its default) - obje -> hlmt -> mode.
pub fn decode_object_geom(c: &H3Cache, obje: usize, variant: Option<usize>) -> Result<H3ModelGeom> {
    decode_object_state_geom(c, obje, variant, 0)
}

/// #damage `decode_object_geom` in a DAMAGE display state: the permutations hlmt variant
/// `variant` authors for `state` (attach.rs `variant_state_meshes`; 0 = the regular model).
/// A viewer/editor display option - nothing here is ever written back to a map variant.
pub fn decode_object_state_geom(c: &H3Cache, obje: usize, variant: Option<usize>, state: u8) -> Result<H3ModelGeom> {
    let mode = model_of(c, obje).ok_or_else(|| anyhow::anyhow!("{}: no hlmt / mode", c.tag_name(obje)))?;
    let model = load_model(c, mode)?;
    let meshes = super::attach::variant_state_meshes(c, obje, &model, variant, state);
    decode_model_geom_meshes(c, &model, &meshes)
}

/// `load_model` + `decode_model_mesh` over a `mode` tag's lod0 meshes.
pub fn decode_model_geom(c: &H3Cache, mode: usize) -> Result<H3ModelGeom> {
    let model = load_model(c, mode)?;
    let meshes = model.lod0_meshes.clone();
    decode_model_geom_meshes(c, &model, &meshes)
}

fn decode_model_geom_meshes(c: &H3Cache, model: &super::objects::H3Model, meshes: &[usize]) -> Result<H3ModelGeom> {
    let mut g = H3ModelGeom { name: model.name.clone(), materials: model.materials.clone(), meshes: Vec::new(), min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN) };
    for &mi in meshes {
        let dm = match decode_model_mesh(c, model, mi) { Ok(Some(d)) => d, Ok(None) => continue, Err(_) => continue };
        if dm.verts.is_empty() || dm.indices.len() < 3 { continue; }
        let verts: Vec<MeshVertex> = dm.verts.iter().map(|v| MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: [1.0; 4], ..Default::default() }).collect();
        for v in &verts { g.min = g.min.min(Vec3::from(v.pos)); g.max = g.max.max(Vec3::from(v.pos)); }
        g.meshes.push(H3GeomMesh { verts, indices: dm.indices, parts: dm.parts });
    }
    if g.meshes.is_empty() { g.min = Vec3::ZERO; g.max = Vec3::ZERO; }
    Ok(g)
}

/// Draw a decoded model at `mats` (one draw per part, every instance), lit by the flat
/// `color` (the static object path's lighting), into `out(lane, mesh)`.
pub(super) fn upload_model_parts(tc: &mut TexCtx, mr: &MeshRenderer, geom: &H3ModelGeom, mats: &[Mat4], color: [f32; 4], out: &mut dyn FnMut(Lane, GpuMesh)) -> usize {
    let mut drawn = 0usize;
    let (device, queue) = (tc.device.clone(), tc.queue.clone());
    for m in &geom.meshes {
        let verts: Vec<MeshVertex> = m.verts.iter().map(|v| MeshVertex { color, ..*v }).collect();
        for part in &m.parts {
            let s = part.index_start as usize;
            let e = (s + part.index_count as usize).min(m.indices.len());
            if e < s + 3 { continue; }
            let render_tag = usize::try_from(part.material).ok().and_then(|k| geom.materials.get(k).copied().flatten());
            let mut tex = None;
            let mut cut = false;
            if !tc.notex {
                if let Some(rt) = render_tag {
                    match tc.constant_color(rt) {
                        ConstColor::Cut => cut = true,
                        ConstColor::Color(v) => tex = Some(v),
                        ConstColor::NotConstant => tex = tc.albedo(rt),
                    }
                }
            }
            if cut { continue }
            let lane = tc.lane_for(render_tag, tex.is_some());
            let tex = tex.or_else(|| (!tc.notex).then(|| tc.grey.clone()));
            out(lane, upload_part(mr, &device, &queue, &verts, &m.indices[s..e], mats, tex.as_ref()));
            drawn += 1;
        }
    }
    drawn
}

/// What the build produced (the status line + the camera framing).
#[derive(Default, Debug)]
pub struct H3Stats {
    pub bsps: usize,
    pub meshes: usize,
    pub draws: usize,
    pub tris: usize,
    pub instances_drawn: usize,
    pub instances_skipped: usize,
    pub tex_ok: usize,
    pub tex_fail: usize,
    pub flat: usize,
    /// Uses lit per pixel / per vertex / mean bake / placeholder sun.
    pub lm_pp: usize,
    pub lm_pv: usize,
    pub lm_mean: usize,
    pub lm_none: usize,
    pub has_lightmap: bool,
    pub objects_placed: usize,
    pub objects_skipped: usize,
    /// Map-variant (Forge) objects placed / unresolved (no quota tag / model). #h3
    pub variant_placed: usize,
    pub variant_skipped: usize,
    /// hlmt variant child objects drawn at their markers (attach.rs). #h3-attach
    pub attachments: usize,
    /// The map's atmosphere fog uniform (lighting.rs; None = no atmosphere). #h3-fog
    pub fog: Option<[f32; 28]>,
    /// BSP parts drawn through the renderer's water pass (`rmw`). #h3-water
    pub water_parts: usize,
    pub sky_parts: usize,
    /// BSP parts drawn through the terrain pass (rmtr layers).
    pub terrain_parts: usize,
    pub lanes: Vec<(Lane, usize)>,
    /// Frame bounds of the playable clusters (`frame_bounds`), for the camera.
    pub bmin: Vec3,
    pub bmax: Vec3,
    pub expo: f32,
    /// The map's `camera_fx_settings` (`cfxs`): the authored metering key, exposure band and
    /// bloom curve. None when the scenario names none. #h3-expo
    pub camera_fx: Option<super::lighting::H3CameraFx>,
    pub secs: f32,
}

impl H3Stats {
    pub fn frame_bounds(&self) -> (Vec3, Vec3) {
        if self.meshes == 0 || self.bmin.x > self.bmax.x { (Vec3::splat(-10.0), Vec3::splat(10.0)) } else { (self.bmin, self.bmax) }
    }
    fn lane_add(&mut self, lane: Lane, n: usize) {
        if let Some(e) = self.lanes.iter_mut().find(|(l, _)| *l == lane) { e.1 += n; } else { self.lanes.push((lane, n)); }
    }
}

/// How one mesh use gets its baked vertex colour.
#[derive(Clone, Copy, Debug)]
enum UseLight {
    /// A cluster mesh: its own `vb[1]` lightmap-UV stream.
    Cluster,
    /// An instance with a per-instance lightmap-UV stream (vb index) or a per-vertex stream.
    Instance { mode: LightingMode, vb: usize },
    /// No bake available: the placeholder (sun) shading / the mean bake.
    Flat,
}

/// Bake the engine diffuse into `verts[i].color` for one use; Ok(mode) on success.
fn bake_use(c: &H3Cache, bsp: &H3Bsp, lm: &H3Lightmap, mesh: usize, light: UseLight, mats: &[Mat4],
    verts: &mut [MeshVertex], lm_uv: &mut Vec<[f32; 2]>) -> Result<LightingMode> {
    let m0 = mats.first().copied().unwrap_or(Mat4::IDENTITY);
    let wn = |n: [f32; 3]| m0.transform_vector3(Vec3::from(n)).normalize_or_zero();
    match light {
        UseLight::Cluster => {
            let uvs = cluster_lightmap_uvs(c, bsp, mesh, verts.len())?;
            for (v, uv) in verts.iter_mut().zip(&uvs) {
                let (sh, i) = lm.probe_at(uv[0], uv[1]);
                let d = shade(&sh, i, Vec3::from(v.normal));
                v.color = [d[0], d[1], d[2], 0.0];
            }
            *lm_uv = uvs;
            Ok(LightingMode::PerPixel)
        }
        UseLight::Instance { mode: LightingMode::PerPixel, vb } => {
            let uvs = uv_stream(c, bsp, vb, verts.len())?;
            for (v, uv) in verts.iter_mut().zip(&uvs) {
                let (sh, i) = lm.probe_at(uv[0], uv[1]);
                let d = shade(&sh, i, wn(v.normal));
                v.color = [d[0], d[1], d[2], 0.0];
            }
            *lm_uv = uvs;
            Ok(LightingMode::PerPixel)
        }
        UseLight::Instance { mode: LightingMode::PerVertex, vb } => {
            let rec = pervertex_stream(c, bsp, vb, verts.len())?;
            for (k, v) in verts.iter_mut().enumerate() {
                let (sh, i) = pervertex_probe(&rec[k * PERVERTEX_STRIDE..(k + 1) * PERVERTEX_STRIDE]);
                let d = shade(&sh, i, wn(v.normal));
                v.color = [d[0], d[1], d[2], 0.0];
            }
            Ok(LightingMode::PerVertex)
        }
        _ => Ok(LightingMode::Flat),
    }
}

/// H3 classic bitmap format -> DXGI enum for the DDS/BC upload path (matches `upload_texture_dds`).
fn h3_dxgi(format: u16) -> Option<u32> {
    Some(match format {
        0x0E => 71,                            // DXT1 -> BC1
        0x0F => 74,                            // DXT3 -> BC2
        0x10 => 77,                            // DXT5 -> BC3
        0x1E | 0x1F | 0x23 | 0x24 | 0x25 | 0x26 => 80, // DXT5a -> BC4
        0x21 | 0x27 => 83,                     // DXN  -> BC5_UNORM
        _ => return None,
    })
}

/// Upload a `bitm` tag's mip chain as a GPU texture view. BCn formats go through a DXT10 DDS
/// header (block data as-is, sRGB linearised by the mesh shader's `pow(2.2)`); A8R8G8B8 / X8R8G8B8
/// upload mip 0 as BGRA8. Err for a format this path does not model (the caller falls back to flat).
fn upload_bitmap(c: &H3Cache, device: &wgpu::Device, queue: &wgpu::Queue, tag: usize) -> Result<wgpu::TextureView> {
    let (info, bytes) = super::bitmaps::bitmap_pixels_avail(c, tag)?;
    if info.kind != 0 { return Err(anyhow!("kind {} (not 2D)", info.kind)); }
    if info.w == 0 || info.h == 0 || bytes.is_empty() { return Err(anyhow!("empty {}x{} ({} B)", info.w, info.h, bytes.len())); }
    let (bpb, _) = super::bitmaps::format_desc(info.format).ok_or_else(|| anyhow!("unmodelled format {:#x}", info.format))?;
    if bpb > 0 {
        let dxgi = h3_dxgi(info.format).ok_or_else(|| anyhow!("no DXGI for format {:#x}", info.format))?;
        let mut dds = crate::h4::bitmaps::dds_header_dxt10(info.w as u32, info.h as u32, info.levels, dxgi);
        dds.extend_from_slice(&bytes);
        hms_render::upload_texture_dds(device, queue, &dds).map(|(v, _)| v)
            .ok_or_else(|| anyhow!("BC upload refused ({}x{} fmt {:#x})", info.w, info.h, info.format))
    } else if info.format == 0x0B || info.format == 0x0A {
        let n = info.w as usize * info.h as usize * 4;
        if bytes.len() < n { return Err(anyhow!("BGRA short: {} < {n}", bytes.len())); }
        Ok(hms_render::upload_texture_bgra(device, queue, &bytes[..n], info.w as u32, info.h as u32).0)
    } else {
        Err(anyhow!("uncompressed format {:#x} not modelled", info.format))
    }
}

/// Union of the cluster AABBs, excluding clusters whose CENTER sits far outside the pack (a sky /
/// vista mesh placed hundreds of units away), so the camera frames the playable space.
pub fn frame_bounds(boxes: &[(Vec3, Vec3)]) -> (Vec3, Vec3) {
    let full = |sub: &[usize]| {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for &i in sub { mn = mn.min(boxes[i].0); mx = mx.max(boxes[i].1); }
        (mn, mx)
    };
    if boxes.is_empty() { return (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); }
    let centers: Vec<Vec3> = boxes.iter().map(|(a, b)| (*a + *b) * 0.5).collect();
    // median center (per-axis median = robust centroid)
    let med_axis = |k: usize| { let mut v: Vec<f32> = centers.iter().map(|c| c[k]).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v[v.len() / 2] };
    let mc = Vec3::new(med_axis(0), med_axis(1), med_axis(2));
    let dists: Vec<f32> = centers.iter().map(|c| (*c - mc).length()).collect();
    let mut sorted = dists.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_dist = sorted[sorted.len() / 2].max(1.0);
    let keep: Vec<usize> = (0..boxes.len()).filter(|&i| dists[i] <= med_dist * 8.0 + 30.0).collect();
    if keep.len() >= 2 { full(&keep) } else { full(&(0..boxes.len()).collect::<Vec<_>>()) }
}

/// Material + texture resolution shared by every part of the build (both caches are keyed by
/// tag index, so a bitmap used by the BSP, an object and the sky uploads once). Owns its cache
/// / device handles so the editor scene (edit_scene.rs) can keep one across rebuilds.
pub(super) struct TexCtx {
    pub(super) c: Arc<H3Cache>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    mat_cache: HashMap<usize, H3Material>,
    tex_cache: HashMap<usize, Option<wgpu::TextureView>>,
    /// Composited BSP albedo, keyed by render-method tag (`albedo.rs`): the engine
    /// `calc_albedo_*_ps` product of base x detail layers, NOT the bare `base_map`. #h3-albedo
    albedo_cache: HashMap<usize, Option<wgpu::TextureView>>,
    /// Decoded mip-0 images the composite reads, keyed by `bitm` tag (detail maps are shared
    /// between dozens of materials, so decoding each one once matters).
    img_cache: HashMap<usize, Option<Arc<super::albedo::Img>>>,
    /// HMS_H3_NOALBEDO: bind the bare `base_map` like the pre-composite build (diagnostic).
    noalbedo: bool,
    /// composited / fell back, for the build log.
    pub(super) alb_made: usize,
    pub(super) alb_plain: usize,
    pub(super) notex: bool,
    matdiag: bool,
    /// HMS_H3_NOAT: keep alpha-tested parts on the opaque lane (diagnostic).
    noat: bool,
    /// HMS_H3_NOTERRAIN: draw rmtr parts through the plain (first-layer) path (diagnostic).
    noterrain: bool,
    /// 1x1 mid-grey: the unbound terrain layer / detail slot (the terrain pass gates absent
    /// layers by `active` / `detail_present`, so the bound value never reaches the blend).
    pub(super) grey: wgpu::TextureView,
    /// 1x1 white: the base of a constant-colour (vertex-painted) sky part.
    white: wgpu::TextureView,
    /// #h3-albedo 1x1 views of a `constant_color` material's authored `albedo_color`, and the
    /// mean alpha of an `alpha_test_map` (a fully transparent one cuts its whole part away).
    const_color: HashMap<usize, wgpu::TextureView>,
    at_alpha: HashMap<usize, f32>,
    /// constant-colour parts bound / cut away, for the build log.
    pub(super) const_bound: usize,
    pub(super) const_cut: usize,
    /// HMS_H3_ONLYMAT=<substring>: draw only the BSP parts whose material name / class
    /// contains it (print-only isolation diagnostic).
    onlymat: Option<String>,
}

impl TexCtx {
    /// A fresh context over `cache` (the diagnostics env knobs read once here).
    pub(super) fn new(c: Arc<H3Cache>, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let grey = hms_render::upload_texture_rgba(device, queue, &[128, 128, 128, 255], 1, 1).0;
        let white = hms_render::upload_texture_rgba(device, queue, &[255, 255, 255, 255], 1, 1).0;
        TexCtx { c, device: device.clone(), queue: queue.clone(), mat_cache: HashMap::new(), tex_cache: HashMap::new(),
            albedo_cache: HashMap::new(), img_cache: HashMap::new(), noalbedo: std::env::var("HMS_H3_NOALBEDO").is_ok(), alb_made: 0, alb_plain: 0,
            notex: std::env::var("HMS_H3_NOTEX").is_ok(), matdiag: std::env::var("HMS_H3_MATDIAG").is_ok(), noat: std::env::var("HMS_H3_NOAT").is_ok(),
            noterrain: std::env::var("HMS_H3_NOTERRAIN").is_ok(), grey, white,
            const_color: HashMap::new(), at_alpha: HashMap::new(), const_bound: 0, const_cut: 0,
            onlymat: std::env::var("HMS_H3_ONLYMAT").ok().filter(|s| !s.is_empty()) }
    }
    pub(super) fn material(&mut self, render_tag: usize) -> &H3Material {
        let c = &self.c;
        self.mat_cache.entry(render_tag).or_insert_with(|| load_material(c, render_tag))
    }
    /// The uploaded view of a `bitm` tag (None after a failed upload, remembered).
    pub(super) fn texture(&mut self, bitm: usize) -> Option<wgpu::TextureView> {
        let (c, device, queue, matdiag) = (&self.c, &self.device, &self.queue, self.matdiag);
        self.tex_cache.entry(bitm).or_insert_with(|| match upload_bitmap(c, device, queue, bitm) {
            Ok(v) => Some(v),
            Err(e) => { if matdiag { eprintln!("  H3TEX fail {}: {e}", c.tag_name(bitm).rsplit('\\').next().unwrap_or("")); } None }
        }).clone()
    }
    /// The BSP surface colour of a render method: the engine's composed albedo
    /// (`albedo.rs::composite` = base x detail layers x DETAIL_MULTIPLIER, the option's law)
    /// when the material names a detail layer, else the plain `base_map` upload. Cached per
    /// render tag. HMS_H3_NOALBEDO forces the old base-only bind. #h3-albedo
    pub(super) fn albedo(&mut self, render_tag: usize) -> Option<wgpu::TextureView> {
        if let Some(v) = self.albedo_cache.get(&render_tag) { return v.clone(); }
        let m = self.material(render_tag).clone();
        let plain = |me: &mut Self| m.base.and_then(|b| me.texture(b));
        if self.noalbedo || !super::albedo::needs_composite(&m) {
            let v = plain(self);
            if v.is_some() { self.alb_plain += 1; }
            self.albedo_cache.insert(render_tag, v.clone());
            return v;
        }
        let built = {
            let c = self.c.clone();
            super::albedo::composite(&c, &m, &mut self.img_cache)
        };
        let v = match built {
            Some((bgra, w, h)) => { self.alb_made += 1; Some(hms_render::upload_texture_bgra(&self.device, &self.queue, &bgra, w, h).0) }
            // a layer would not decode: the bare base is still better than nothing
            None => { let v = plain(self); if v.is_some() { self.alb_plain += 1; } v }
        };
        self.albedo_cache.insert(render_tag, v.clone());
        v
    }

    /// #h3-albedo A BSP material whose albedo option is `constant_color` (2) binds NO base map:
    /// the engine's surface colour is the authored `albedo_color` constant (times a vertex
    /// colour stream where the geometry has one). HMS used to draw these through the mid-grey
    /// placeholder, which painted High Ground's `z_black` light boxes and Heretic's blockers as
    /// bright grey slabs - 304 and 299 parts respectively.
    ///
    /// Returns `Cut` when the material also alpha-tests on a map that is ENTIRELY transparent
    /// (`z_invisible_quads` binds `alpha_black`): every texel fails the engine's alpha test, so
    /// the part is not drawn at all - that is what makes those quads invisible in game.
    pub(super) fn constant_color(&mut self, render_tag: usize) -> ConstColor {
        let m = self.material(render_tag).clone();
        if m.albedo_opt() != 2 || m.base.is_some() { return ConstColor::NotConstant }
        if let Some(at) = m.param("alpha_test_map") {
            let c = self.c.clone();
            let a = *self.at_alpha.entry(at).or_insert_with(|| {
                match super::albedo::decode_rgba(&c, at) {
                    Some(img) => {
                        let (mut s, mut n) = (0u64, 0u64);
                        for p in img.px.chunks_exact(4).step_by(7) { s += p[3] as u64; n += 1; }
                        s as f32 / n.max(1) as f32 / 255.0
                    }
                    None => 1.0,
                }
            });
            // the engine's alpha_test threshold is well above a map that is black everywhere
            if a < 0.02 { self.const_cut += 1; return ConstColor::Cut }
        }
        let col = m.float("albedo_color").unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let (device, queue) = (self.device.clone(), self.queue.clone());
        let v = self.const_color.entry(render_tag).or_insert_with(|| {
            // the mesh shader pow(2.2)s its base sample, so encode the LINEAR constant
            let e = |x: f32| (x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            hms_render::upload_texture_rgba(&device, &queue, &[e(col[0]), e(col[1]), e(col[2]), 255], 1, 1).0
        }).clone();
        self.const_bound += 1;
        ConstColor::Color(v)
    }

    /// Does this render method cut out on an alpha-test map (foliage / grates)?
    fn alpha_tested(&mut self, render_tag: usize) -> bool {
        if self.noat { return false; }
        self.material(render_tag).alpha_test()
    }
    /// The renderer lane a part of this render method draws in: the template's blend_mode
    /// (materials.rs) picks additive / alpha-blend (multiply and premultiplied alpha ride the
    /// alpha lane - the closest the renderer has), an alpha test the cutout lane, else opaque.
    /// HMS_H3_NOAT (diagnostic) keeps everything opaque.
    pub(super) fn lane_for(&mut self, render_tag: Option<usize>, textured: bool) -> Lane {
        let Some(rt) = render_tag else { return Lane::Opaque };
        if self.noat || !textured { return Lane::Opaque; }
        match self.material(rt).blend() {
            H3Blend::Additive => Lane::Additive,
            H3Blend::AlphaBlend | H3Blend::PremulAlpha | H3Blend::Multiply | H3Blend::DoubleMultiply => Lane::Blend,
            H3Blend::Opaque => if self.alpha_tested(rt) { Lane::AlphaTest } else { Lane::Opaque },
        }
    }
    fn uploaded(&self) -> usize { self.tex_cache.values().filter(|v| v.is_some()).count() }
}

/// #h3-albedo What a `constant_color` BSP material resolves to.
pub(super) enum ConstColor {
    /// Not a constant-colour material (it has a base map).
    NotConstant,
    /// The authored `albedo_color` as a 1x1 surface colour.
    Color(wgpu::TextureView),
    /// Alpha-tested entirely away: do not draw the part.
    Cut,
}

/// One textured, lit, instanced draw through the shared mesh renderer (the ONE upload call of
/// the Halo 3 build: BSP parts, object parts).
pub(super) fn upload_part(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[MeshVertex], idx: &[u32], mats: &[Mat4], tex: Option<&wgpu::TextureView>) -> GpuMesh {
    mr.upload_mesh(device, queue, verts, idx, mats,
        None, tex, None, None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
        0.0, 1.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None,
        false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None)
}

/// One Halo 3 dynamic-OBJECT part through the engine's `static_sh` entry point (mesh.rs
/// `h3_object_shade`, reached by the `matmodel = 100 + material_model` sentinel). #h3-objlight
///
/// THE BUG THIS FIXES. Halo 3 weapons / vehicles were drawn through the same flat `upload_part`
/// the BSP uses: albedo x one baked colour. Their diffuse maps are genuinely dark (a battle
/// rifle's is ~0.003 linear) because the engine's visible metal is almost entirely the
/// cook-torrance SPECULAR and the probe's area/environment lobe - so every weapon, vehicle and
/// piece of scenery read black. This binds the real law:
///   diffuse  = ravi_order_2_with_dominant_light(N, probe SH) x albedo x diffuse_coefficient
///   specular = spec_mask x specular_coefficient x specular_tint x
///              (cook_torrance analytic x analytical_specular_contribution
///               + sh_glossy(probe, R) x area_specular_contribution)
///   envmap   = cube(R) x environment_map_specular_contribution x env_tint x area
///   + self_illum, all from the material's own rmt2 constants (`H3Shading`).
/// The probe is the structure lightmap under the placement (`H3LightField::sample`).
#[allow(clippy::too_many_arguments)]
fn upload_object_part(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, tc: &mut TexCtx,
    render_tag: Option<usize>, verts: &[MeshVertex], idx: &[u32], mats: &[Mat4],
    tex: Option<&wgpu::TextureView>, probes: &[super::objlight::H3ObjProbe], fallback: [f32; 4]) -> GpuMesh {
    let Some(rt) = render_tag else { return upload_part(mr, device, queue, verts, idx, mats, tex) };
    let m = tc.material(rt).clone();
    let sh = m.shading();
    // the engine samplers this lane reads: detail_map (albedo), self_illum_map, the
    // specular_mask texture (specular_mask option 3) and the authored environment cube
    let detail = m.param("detail_map").and_then(|b| tc.texture(b));
    let detail_xf = if detail.is_some() { m.xform("detail_map") } else { [0.0; 4] };
    let si_tex = m.param("self_illum_map").and_then(|b| tc.texture(b));
    let spec_tex = m.spec_mask_texture().and_then(|b| tc.texture(b));
    let base_xf = m.xform("base_map");
    let tile = [if base_xf[0].abs() > 1e-6 { base_xf[0] } else { 1.0 }, if base_xf[1].abs() > 1e-6 { base_xf[1] } else { 1.0 }];
    // self_illumination `simple`: map x self_illum_color x intensity
    let si_col = m.float("self_illum_color").unwrap_or([1.0, 1.0, 1.0, 1.0]);
    let si_int = m.float("self_illum_intensity").map(|v| v[0]).unwrap_or(1.0);
    let si_ctl = if si_tex.is_some() { [1.0, si_col[0] * si_int, si_col[1] * si_int, si_col[2] * si_int] } else { [0.0; 4] };
    // fine_xform = [diffuse_coefficient, spec_mask mode, albedo_color.a, env mode]
    let smode = if sh.spec_mask == 3 && spec_tex.is_none() { 0 } else { sh.spec_mask } as f32;
    let fine = [sh.diffuse_coefficient.max(0.0), smode, sh.albedo_color[3], sh.env_mode as f32];
    let mat_spec2 = [sh.specular_coefficient, sh.analytical_specular, sh.area_specular, sh.env_specular];
    let env_ctl = [sh.env_tint[0], sh.env_tint[1], sh.env_tint[2], sh.env_roughness_scale];
    let xbump = [sh.albedo_color[0], sh.albedo_color[1], sh.albedo_color[2], 0.0];
    let matmodel = 100.0 + sh.model as f32;
    // per-instance probe lanes (objlight.rs `H3ObjProbe::lanes`)
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(mats.len());
    let mut pr: Vec<([f32; 8], [u32; 4])> = Vec::with_capacity(mats.len());
    for k in 0..mats.len() {
        match probes.get(k) {
            Some(p) => { let (c, a, l) = p.lanes(); colors.push(c); pr.push((a, l)); }
            None => { let (c, a, l) = super::objlight::H3ObjProbe::flat([fallback[0], fallback[1], fallback[2]]).lanes(); colors.push(c); pr.push((a, l)); }
        }
    }
    let mut gm = mr.upload_mesh(device, queue, verts, idx, mats, Some(&colors), tex, si_tex.as_ref(),
        detail.as_ref(), None, None, [0.0, 0.0], tile, detail_xf, 0.0, 0.0, [0.0, -1.0], mat_spec2,
        [0.0; 4], fine, [0.0; 4], env_ctl, None, true, matmodel, si_ctl, None,
        spec_tex.as_ref(), [0.0; 4], [1.0, 1.0, 0.0, 0.0], None, None, xbump, [0.0; 4], None);
    gm.set_spec_tints(queue, [sh.specular_tint[0], sh.specular_tint[1], sh.specular_tint[2], sh.roughness],
        [sh.fresnel_color[0], sh.fresnel_color[1], sh.fresnel_color[2], sh.albedo_blend]);
    gm.set_instance_probes(queue, &pr);
    gm
}

/// A decoded `rmtr` terrain material: the blend map + up to four base / detail layers with
/// their texcoord transforms, bound to the renderer's terrain pass. None when the material is
/// not a blendable terrain (no blend map / no layer resolves) - the caller draws it flat.
struct TerrainLayers {
    blend: wgpu::TextureView,
    blend_xf: [f32; 4],
    base: [Option<wgpu::TextureView>; 4],
    base_xf: [[f32; 4]; 4],
    detail: [Option<wgpu::TextureView>; 4],
    detail_xf: [[f32; 4]; 4],
}

impl TexCtx {
    /// Resolve an rmtr render method's layers (`blend_map`, `base_map_m_N`, `detail_map_m_N`:
    /// the classic terrain template's sampler names, verified on riverworld / every rmtr).
    fn terrain_layers(&mut self, render_tag: usize) -> Option<TerrainLayers> {
        if self.notex || self.noterrain { return None; }
        let m = self.material(render_tag).clone();
        if &m.class != b"rmtr" { return None; }
        let blend = m.param("blend_map").and_then(|b| self.texture(b))?;
        let mut t = TerrainLayers { blend, blend_xf: m.xform("blend_map"), base: [None, None, None, None], base_xf: [[1.0, 1.0, 0.0, 0.0]; 4], detail: [None, None, None, None], detail_xf: [[1.0, 1.0, 0.0, 0.0]; 4] };
        let mut any = false;
        for l in 0..4 {
            let bn = format!("base_map_m_{l}");
            let dn = format!("detail_map_m_{l}");
            t.base[l] = m.param(&bn).and_then(|b| self.texture(b));
            t.base_xf[l] = m.xform(&bn);
            t.detail[l] = m.param(&dn).and_then(|b| self.texture(b));
            t.detail_xf[l] = m.xform(&dn);
            any |= t.base[l].is_some();
        }
        any.then_some(t)
    }
}

/// A water (`rmw`) part through the renderer's water pass. Body colour / murkiness / fresnel /
/// diffuse / extinction from the material's rmt2-named constants (`water_color_pure`,
/// `watercolor_coefficient`, `water_murkiness`, `fresnel_coefficient`, `water_diffuse`,
/// `refraction_extinct_distance`, `sunspot_cut`, `shadow_intensity_mark`), the body colour of a
/// texture-coloured water (riverworld: `watercolor_texture` x coefficient) from the colour map's
/// mean texel - the water is drawn PURE-COLOUR (`cat.x = 0`): this renderer's texture-colour +
/// refraction path renders magenta (the H2A lesson), so the map is folded into the constant.
/// Reflection / refraction follow the template's category options; the authored cube, foam and
/// wave arrays are not bound (the renderer's captured sky dome / procedural waves stand in).
/// Instanced water is baked to world space (the water pass takes no instance matrix). #h3-water
fn upload_water_part(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, tc: &mut TexCtx, render_tag: usize, verts: &[MeshVertex], idx: &[u32], mats: &[Mat4], log: &mut dyn FnMut(String)) -> Option<GpuMesh> {
    let m = tc.material(render_tag).clone();
    let f = |name: &str| m.floats.iter().find(|(n, _)| n == name).map(|(_, v)| *v);
    let mut wp = hms_render::WaterParams::default();
    let coef = f("watercolor_coefficient").map(|v| v[0]).unwrap_or(1.0);
    // the body colour: the pure constant, else the colour map's mean texel x coefficient
    let mut body = f("water_color_pure").map(|v| [v[0], v[1], v[2]]);
    if body.is_none() {
        if let Some(bt) = m.param("watercolor_texture") {
            body = mean_texel_linear(&tc.c, bt).map(|c| [c[0] * coef, c[1] * coef, c[2] * coef]);
        }
    }
    let body = body.unwrap_or([0.015, 0.05, 0.08]);
    wp.deep = [body[0], body[1], body[2], 0.0];
    wp.p0 = [coef, f("water_murkiness").map(|v| v[0]).unwrap_or(wp.p0[1]), f("fresnel_coefficient").map(|v| v[0]).unwrap_or(wp.p0[2]), wp.p0[3]];
    if let Some(d) = f("water_diffuse") { wp.diff = [d[0], d[1], d[2], f("sunspot_cut").map(|v| v[0]).unwrap_or(wp.diff[3])]; }
    if let Some(e) = f("refraction_extinct_distance") { wp.edge[0] = e[0]; }
    if let Some(sm) = f("shadow_intensity_mark") { wp.foam[2] = sm[0]; }
    // template categories (materials.rs: waveshape, watercolor, reflection, refraction, ...)
    let opt = |i: usize| m.template_opts.get(i).copied().unwrap_or(0) as f32;
    wp.cat = [0.0, opt(2).min(1.0), opt(3).min(1.0), 0.0];
    let wverts: Vec<MeshVertex> = if mats.len() == 1 && mats[0] != Mat4::IDENTITY {
        let mm = mats[0]; let n = glam::Mat3::from_mat4(mm);
        verts.iter().map(|v| MeshVertex { pos: mm.transform_point3(Vec3::from(v.pos)).into(), normal: (n * Vec3::from(v.normal)).normalize_or_zero().into(), ..*v }).collect()
    } else { verts.to_vec() };
    log(format!("h3 water: '{}' body [{:.3} {:.3} {:.3}] murk {:.2} fresnel {:.2} diffuse [{:.3} {:.3} {:.3}] extinct {:.0} cat {:?} ({} verts)",
        m.name.rsplit('\\').next().unwrap_or(""), body[0], body[1], body[2], wp.p0[1], wp.p0[2], wp.diff[0], wp.diff[1], wp.diff[2], wp.edge[0], wp.cat, wverts.len()));
    Some(mr.upload_water_mesh(device, queue, &wverts, idx, None, None, &wp, None, None, None, None))
}

/// Mean texel of a BC1 / BC3 colour map, linear (the block colour endpoints' average, sRGB
/// decoded); None for another format.
fn mean_texel_linear(c: &H3Cache, bitm: usize) -> Option<[f32; 3]> {
    let (info, px) = super::bitmaps::bitmap_pixels_avail(c, bitm).ok()?;
    let (bs, coff) = match info.format { 0x0E => (8usize, 0usize), 0x10 => (16, 8), _ => return None };
    let w = (info.w as usize + 3) / 4; let h = (info.h as usize + 3) / 4;
    let n = (w * h).min(px.len() / bs);
    if n == 0 { return None; }
    let mut acc = [0f64; 3];
    for i in 0..n {
        for k in 0..2 {
            let o = i * bs + coff + k * 2;
            let v = u16::from_le_bytes([px[o], px[o + 1]]);
            let r = ((v >> 11) & 31) as f64 / 31.0; let g = ((v >> 5) & 63) as f64 / 63.0; let b = (v & 31) as f64 / 31.0;
            acc[0] += r.powf(2.2); acc[1] += g.powf(2.2); acc[2] += b.powf(2.2);
        }
    }
    let d = (2 * n) as f64;
    Some([(acc[0] / d) as f32, (acc[1] / d) as f32, (acc[2] / d) as f32])
}

/// One terrain part through the renderer's terrain pass, one upload per placement matrix.
/// The layer weights come from the blend map at its own transform, normalised over the ACTIVE
/// (bound) layers; every base decodes as sRGB (2.2) like the mesh path's albedo. No bump layers
/// (the Halo 3 terrain bump / detail-bump / specular terms are not carried).
fn upload_terrain(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, grey: &wgpu::TextureView, verts: &[MeshVertex], idx: &[u32], mats: &[Mat4], t: &TerrainLayers) -> Vec<GpuMesh> {
    fn view<'v>(v: &'v Option<wgpu::TextureView>, grey: &'v wgpu::TextureView) -> &'v wgpu::TextureView { v.as_ref().unwrap_or(grey) }
    let bv = [view(&t.base[0], grey), view(&t.base[1], grey), view(&t.base[2], grey), view(&t.base[3], grey)];
    let dv = [view(&t.detail[0], grey), view(&t.detail[1], grey), view(&t.detail[2], grey), view(&t.detail[3], grey)];
    let flat = [grey; 4];
    let on = |v: &Option<wgpu::TextureView>| if v.is_some() { 1.0 } else { 0.0 };
    let tile = |x: &[f32; 4]| [x[0], x[1]];
    let off = |x: &[f32; 4]| [x[2], x[3]];
    let params = TerrainMaterialParams {
        base_tile: [tile(&t.base_xf[0]), tile(&t.base_xf[1]), tile(&t.base_xf[2]), tile(&t.base_xf[3])],
        base_offset: [off(&t.base_xf[0]), off(&t.base_xf[1]), off(&t.base_xf[2]), off(&t.base_xf[3])],
        detail_tile: [tile(&t.detail_xf[0]), tile(&t.detail_xf[1]), tile(&t.detail_xf[2]), tile(&t.detail_xf[3])],
        detail_offset: [off(&t.detail_xf[0]), off(&t.detail_xf[1]), off(&t.detail_xf[2]), off(&t.detail_xf[3])],
        blend_xform: t.blend_xf,
        global_tint: [1.0, 1.0, 1.0, 0.0],
        active: [on(&t.base[0]), on(&t.base[1]), on(&t.base[2]), on(&t.base[3])],
        detail_present: [on(&t.detail[0]), on(&t.detail[1]), on(&t.detail[2]), on(&t.detail[3])],
        bump_tile: [[1.0, 1.0]; 4], bump_offset: [[0.0, 0.0]; 4], bump_present: [0.0; 4],
        detail_bump_tile: [[1.0, 1.0]; 4], detail_bump_offset: [[0.0, 0.0]; 4], detail_bump_present: [0.0; 4],
        uv_density: 0.0, dbb_mode: 0.0, dbb_slope: 0.0, dbb_offset: 0.0,
        blend_target: [[0.0; 4]; 4], blend_max: [0.0; 4],
        base_curve: [2.2; 4], detail_curve: [2.2; 4],
        bump_snorm: [0.0; 4], detail_bump_snorm: [0.0; 4],
    };
    mats.iter().map(|m| mr.upload_terrain_mesh(device, queue, verts, idx, bv, dv, &t.blend, flat, flat, &params, None, *m)).collect()
}

/// Per-lane upload batches of one build unit, flushed to `emit` in one go per lane.
#[derive(Default)]
struct LaneBatches { v: Vec<(Lane, Vec<GpuMesh>)> }

impl LaneBatches {
    fn push(&mut self, lane: Lane, gm: GpuMesh) {
        match self.v.iter_mut().find(|(l, _)| *l == lane) { Some((_, b)) => b.push(gm), None => self.v.push((lane, vec![gm])) }
    }
    fn flush(self, st: &mut H3Stats, emit: &mut dyn FnMut(Lane, Vec<GpuMesh>)) {
        for (lane, b) in self.v { if !b.is_empty() { st.lane_add(lane, b.len()); emit(lane, b); } }
    }
}

/// Decode + upload every BSP, the scenario sky and the scenario objects into the renderer's
/// lanes through `emit`; `progress(done, total)` ticks per BSP / object model; `cancel` aborts
/// between units (Err("cancelled")); `log` receives the scene statistics.
pub fn build_meshes(
    loaded: &H3Loaded,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mr: &MeshRenderer,
    emit: &mut dyn FnMut(Lane, Vec<GpuMesh>),
    progress: &mut dyn FnMut(u32, u32),
    cancel: Option<&AtomicBool>,
    log: &mut dyn FnMut(String),
    // the GUI's editor collect (edit_scene.rs): the BSP triangle soup for picking / raycasts, the
    // objects' flat light; when given, the map variant's objects are NOT drawn statically (the
    // editor scene draws them as dynamic, pickable lanes). #h3-edit
    mut editor: Option<&mut H3EditorCollect>,
) -> Result<H3Stats> {
    let t0 = std::time::Instant::now();
    let c: &H3Cache = &loaded.cache;
    let mut st = H3Stats { expo: fixed_exposure(), ..Default::default() };
    let cancelled = || cancel.map_or(false, |f| f.load(Ordering::Relaxed));

    let sbsps = c.find_tags(b"sbsp");
    log(format!("h3: {} sbsp tag(s)", sbsps.len()));
    let noinst = std::env::var("HMS_H3_NOINST").is_ok();
    let instdiag = std::env::var("HMS_H3_INSTDIAG").is_ok();
    let nolm = std::env::var("HMS_H3_NOLM").is_ok();
    let noobj = std::env::var("HMS_H3_NOOBJ").is_ok();
    let nosky = std::env::var("HMS_H3_NOSKY").is_ok();
    let gain_override = std::env::var("HMS_H3_BAKE_GAIN").ok().and_then(|s| s.parse::<f32>().ok());
    let mut tc = TexCtx::new(loaded.cache.clone(), device, queue);
    // per-mesh AABBs, so the camera can frame the PLAYABLE clusters (skip vista-sized sky meshes)
    let mut mesh_boxes: Vec<(Vec3, Vec3)> = Vec::new();
    // the exposed mean bake over every BSP (the scenario objects' flat light), summed over uses
    let mut world_bake = [0f64; 3];
    let mut world_bake_n = 0usize;
    // bake-weighted mean of the per-BSP gains: the sky lane is emissive at the SCENE exposure
    // in-engine, so it must carry the same auto-exposure stand-in as the lit geometry or the
    // dome sits ~14x darker than the map it surrounds (guardian's skydome_dark read near-black).
    let mut world_gain = 0f64;
    // the object light field (objlight.rs): filled per BSP below, sampled per placement when the
    // scenario / variant objects are drawn. #h3-objlight
    let mut field = super::objlight::H3LightField::new();
    let total_units = sbsps.len() as u32 + 2; // + sky + objects
    let mut done_units = 0u32;
    progress(0, total_units);

    for &sbsp in &sbsps {
        if cancelled() { anyhow::bail!("cancelled"); }
        let bsp = match load_bsp(c, sbsp) { Ok(b) => b, Err(e) => { log(format!("h3: sbsp {sbsp} load failed: {e}")); continue } };
        st.bsps += 1;
        log(format!("h3: bsp '{}' meshes={} clusters={} instances={} (defs {}, identity {}) materials={} geometry={} rot_check row/col/tested {:?} -> {}",
            bsp.name, bsp.meshes.len(), bsp.clusters.len(), bsp.instances.len(), bsp.instance_defs.len(),
            bsp.instances.iter().filter(|i| i.is_identity()).count(), bsp.materials.len(), bsp.geometry.is_some(),
            bsp.rot_check, if bsp.rows { "rows" } else { "cols" }));
        if tc.matdiag {
            for (i, bm) in bsp.materials.iter().enumerate() {
                let Some(rt) = bm.render_tag else { eprintln!("  H3MAT [{i:3}] <null>"); continue };
                let m = tc.material(rt).clone();
                let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
                let base = m.base.map(|b| format!("{} (fmt {})", short(c.tag_name(b)),
                    super::bitmaps::bitmap_info(c, b).map(|i| i.format as i32).unwrap_or(-1)));
                let plist: Vec<String> = m.params.iter().enumerate()
                    .map(|(k, (n, b))| format!("{}{}={}", if Some(*b) == m.base { "*" } else { "" }, if n.is_empty() { format!("tex_{k}") } else { n.clone() }, short(c.tag_name(*b))))
                    .collect();
                eprintln!("  H3MAT [{i:3}] {} ({}) tmpl[{}] -> base {}{} [{} bitmap params: {}]",
                    short(&m.name), String::from_utf8_lossy(&bm.class),
                    m.template_opts.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
                    base.as_deref().unwrap_or("<none>"),
                    if m.from_postprocess { " (postprocess)" } else { "" }, m.params.len(), plist.join(" "));
            }
        }
        // mean baked diffuse over this BSP's lit vertices (log + the flat fallback colour),
        // split per-pixel / per-vertex: the two encodings are independent, so agreeing
        // magnitudes cross-check the per-pixel scale decode.
        let mut bake_sum = [0f64; 3];
        let mut bake_n = 0usize;
        let (mut pp_sum, mut pp_n, mut pv_sum, mut pv_n) = (0f64, 0usize, 0f64, 0usize);
        // the bake (None = placeholder sun for this BSP)
        let lm: Option<Arc<H3Lightmap>> = if nolm { None } else {
            match load_lightmap(c, &bsp) {
                Ok(l) => { log(format!("h3: lightmap {}x{} x{} slices, consts {:?}, {} per-vertex blocks, {} instance uv blocks", l.w, l.h, l.lp.len(), l.consts, l.pervertex_vb.len(), l.inst_texcoord_vb.len())); Some(Arc::new(l)) }
                Err(e) => { log(format!("h3: no lightmap for '{}': {e}", bsp.name)); None }
            }
        };
        if lm.is_some() { st.has_lightmap = true; }
        // Mesh uses: (mesh index, compression-bounds override, lighting) -> placement
        // matrices. Cluster meshes are drawn once at identity; an instance adds its matrix to
        // its mesh's list (an instance whose definition names a different compression entry
        // is its own key, since the dequantised vertices differ), and a LIT instance is its
        // own use (its lighting stream is per instance).
        struct Use { mesh: usize, comp: Option<usize>, light: UseLight, mats: Vec<Mat4> }
        let mut uses: Vec<Use> = Vec::new();
        let mut use_index: HashMap<(usize, Option<usize>), usize> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for &mi in &bsp.clusters {
            if !seen.insert(mi) { continue; }
            let light = if lm.is_some() { UseLight::Cluster } else { UseLight::Flat };
            if std::env::var("HMS_H3_ONLYINST").is_err() || std::env::var("HMS_H3_ONLYINST_KEEPBSP").is_ok() { uses.push(Use { mesh: mi as usize, comp: None, light, mats: vec![Mat4::IDENTITY] }); }
        }
        if !noinst {
            let mut skips: HashMap<&'static str, usize> = HashMap::new();
            // HMS_H3_ONLYINST=<record index>: draw that one instance alone (diagnostic)
            let only_inst: Option<usize> = std::env::var("HMS_H3_ONLYINST").ok().and_then(|v| v.parse().ok());
            // HMS_H3_ONLYFLAGS=<u16>: draw only the instances whose record flags equal this value
            let only_flags: Option<u16> = std::env::var("HMS_H3_ONLYFLAGS").ok().and_then(|v| v.parse().ok());
            for inst in &bsp.instances {
                if let Some(o) = only_inst { if inst.index != o { continue; } }
                if let Some(f) = only_flags { if inst.flags != f { continue; } }
                let reason = if inst.def < 0 || inst.mesh < 0 { Some("unresolved definition") }
                    else if !bsp.meshes.get(inst.mesh as usize).map_or(false, |m| m.has_geometry()) { Some("mesh without geometry") }
                    else if !inst.is_finite() { Some("non-finite transform") }
                    else if inst.not_for_render() { Some("not-for-render flag") }
                    else { None };
                // lighting: the Lbsp instance record picks per-pixel (own UV stream) or per-vertex
                let light = match lm.as_ref().and_then(|l| l.instances.get(inst.index).copied()) {
                    Some((lmi, _, _)) if lmi >= 0 => match usize::try_from(inst.lm_texcoord_block).ok().and_then(|b| lm.as_ref()?.inst_texcoord_vb.get(b)).copied() {
                        Some(vb) if vb >= 0 => UseLight::Instance { mode: LightingMode::PerPixel, vb: vb as usize },
                        _ => UseLight::Flat,
                    },
                    Some((_, pv, _)) if pv >= 0 => match lm.as_ref().and_then(|l| l.pervertex_vb.get(pv as usize)).copied() {
                        Some(vb) if vb >= 0 => UseLight::Instance { mode: LightingMode::PerVertex, vb: vb as usize },
                        _ => UseLight::Flat,
                    },
                    _ => UseLight::Flat,
                };
                if instdiag {
                    eprintln!("  H3INST [{:4}] '{}' def {} mesh {} comp {} scale {:.3} pos [{:.2} {:.2} {:.2}] flags {:#x} sphere r {:.2} light {:?}{}",
                        inst.index, inst.name, inst.def, inst.mesh, inst.compression, inst.scale,
                        inst.translation[0], inst.translation[1], inst.translation[2], inst.flags, inst.world_sphere[3], light,
                        reason.map(|r| format!(" SKIP: {r}")).unwrap_or_default());
                }
                if let Some(r) = reason { *skips.entry(r).or_default() += 1; st.instances_skipped += 1; continue; }
                let comp = bsp.instance_bounds(inst).map(|_| inst.compression as usize);
                let m = bsp.instance_matrix(inst);
                match light {
                    UseLight::Instance { .. } => uses.push(Use { mesh: inst.mesh as usize, comp, light, mats: vec![m] }),
                    _ => {
                        let key = (inst.mesh as usize, comp);
                        let k = *use_index.entry(key).or_insert_with(|| { uses.push(Use { mesh: key.0, comp: key.1, light: UseLight::Flat, mats: Vec::new() }); uses.len() - 1 });
                        uses[k].mats.push(m);
                    }
                }
                st.instances_drawn += 1;
            }
            if !skips.is_empty() { log(format!("h3: instances skipped: {:?}", skips)); }
        }
        // Decode every use (lit uses bake their vertex colours), then upload; the flat uses
        // take the BSP's mean bake so they sit in the same exposure as the lit geometry.
        struct Decoded { verts: Vec<MeshVertex>, dm: super::geometry::DecodedMesh, mats: Vec<Mat4>, lit: bool,
            /// per-vertex LIGHTMAP uv of a per-pixel-lit use (empty otherwise): the object light
            /// field samples the structure's probe through these. #h3-objlight
            lm_uv: Vec<[f32; 2]> }
        let mut decoded: Vec<Decoded> = Vec::new();
        let mut lm_fail = 0usize;
        for u in &uses {
            let bounds: Option<&Bounds> = u.comp.and_then(|k| bsp.bounds.get(k));
            let dm = match decode_mesh_with_bounds(c, &bsp, u.mesh, bounds) { Ok(Some(d)) => d, Ok(None) => continue, Err(e) => { log(format!("h3: mesh {} decode failed: {e}", u.mesh)); continue } };
            if dm.verts.is_empty() || dm.indices.len() < 3 { continue; }
            let mut verts: Vec<MeshVertex> = dm.verts.iter().map(|v| MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: [1.0; 4], ..Default::default() }).collect();
            let mut lit = false;
            let mut mode = LightingMode::Flat;
            let mut lm_uv: Vec<[f32; 2]> = Vec::new();
            if let Some(l) = lm.as_ref() {
                match bake_use(c, &bsp, l, u.mesh, u.light, &u.mats, &mut verts, &mut lm_uv) {
                    Ok(LightingMode::PerPixel) => { st.lm_pp += 1; lit = true; mode = LightingMode::PerPixel; }
                    Ok(LightingMode::PerVertex) => { st.lm_pv += 1; lit = true; mode = LightingMode::PerVertex; }
                    Ok(LightingMode::Flat) => {}
                    Err(e) => { lm_fail += 1; if lm_fail <= 5 { log(format!("h3: mesh {} bake failed: {e} - flat", u.mesh)); } }
                }
            }
            if lit {
                for v in &verts {
                    for k in 0..3 { bake_sum[k] += v.color[k] as f64; }
                    bake_n += 1;
                    let l = (0.2126 * v.color[0] + 0.7152 * v.color[1] + 0.0722 * v.color[2]) as f64;
                    if mode == LightingMode::PerPixel { pp_sum += l; pp_n += 1; } else { pv_sum += l; pv_n += 1; }
                }
            }
            // HMS_H3_CPUXFORM=1 (diagnostic): bake the placement into the vertices on the CPU and
            // upload identity matrices, to isolate the GPU per-instance matrix path.
            let (verts, mats) = if std::env::var("HMS_H3_CPUXFORM").is_ok() && u.mats.len() == 1 && u.mats[0] != Mat4::IDENTITY {
                let m = u.mats[0]; let n = glam::Mat3::from_mat4(m);
                (verts.into_iter().map(|v| MeshVertex { pos: m.transform_point3(glam::Vec3::from(v.pos)).into(), normal: (n * glam::Vec3::from(v.normal)).normalize_or_zero().into(), ..v }).collect(), vec![Mat4::IDENTITY])
            } else { (verts, u.mats.clone()) };
            decoded.push(Decoded { verts, dm, mats, lit, lm_uv });
        }
        if lm_fail > 5 { log(format!("h3: {lm_fail} uses failed to bake (flat)")); }
        let mean_bake = if bake_n > 0 { [(bake_sum[0] / bake_n as f64) as f32, (bake_sum[1] / bake_n as f64) as f32, (bake_sum[2] / bake_n as f64) as f32] } else { [1.0; 3] };
        // the bake gain: mean exposed diffuse -> BAKE_TARGET (see the module doc)
        let mean_lum = 0.2126 * mean_bake[0] + 0.7152 * mean_bake[1] + 0.0722 * mean_bake[2];
        let gain = gain_override.unwrap_or(if bake_n > 0 && mean_lum > 1e-4 { (BAKE_TARGET / (mean_lum * st.expo)).clamp(0.05, 1000.0) } else { 1.0 });
        if lm.is_some() {
            log(format!("h3: bsp '{}' bake mean {:?} (lum {mean_lum:.4}) over {bake_n} verts -> gain {gain:.2}; per-pixel lum {:.4} ({pp_n} verts) vs per-vertex lum {:.4} ({pv_n} verts)",
                bsp.name, mean_bake, if pp_n > 0 { pp_sum / pp_n as f64 } else { 0.0 }, if pv_n > 0 { pv_sum / pv_n as f64 } else { 0.0 }));
            if bake_n > 0 { for k in 0..3 { world_bake[k] += (mean_bake[k] * gain) as f64 * bake_n as f64; } world_bake_n += bake_n; world_gain += gain as f64 * bake_n as f64; }
        }
        // the object light field: this BSP's lightmap at its bake gain, then every per-pixel-lit
        // triangle (cluster meshes + per-pixel instances) in world space with its lightmap uv.
        // An object's probe is the structure's lightmap read at the lit surface beneath it -
        // the engine's `static_sh` lighting point (objlight.rs). #h3-objlight
        let field_lm = lm.as_ref().map(|l| field.add_lightmap(l.clone(), gain));
        let mut lanes = LaneBatches::default();
        for dec in decoded {
            let Decoded { mut verts, dm, mats, lit, lm_uv } = dec;
            if let Some(fl) = field_lm {
                if lit && lm_uv.len() == verts.len() && !lm_uv.is_empty() {
                    let pos: Vec<[f32; 3]> = verts.iter().map(|v| v.pos).collect();
                    for m in &mats { field.add_tris(fl, &pos, &lm_uv, &dm.indices, m); }
                }
            }
            if lit {
                for v in verts.iter_mut() { for k in 0..3 { v.color[k] *= gain; } }
            } else if lm.is_some() {
                st.lm_mean += 1;
                for v in verts.iter_mut() { v.color = [mean_bake[0] * gain, mean_bake[1] * gain, mean_bake[2] * gain, 0.0]; }
            } else {
                st.lm_none += 1;
            }
            let mut mn = Vec3::splat(f32::MAX);
            let mut mx = Vec3::splat(f32::MIN);
            for v in &verts { for k in 0..3 { mn[k] = mn[k].min(v.pos[k]); mx[k] = mx[k].max(v.pos[k]); } }
            // the editor's BSP soup: every drawn triangle of every placement, world space
            if let Some(ed) = editor.as_deref_mut() {
                for m in &mats {
                    for tri in dm.indices.chunks_exact(3) {
                        let (Some(a), Some(b), Some(cc)) = (verts.get(tri[0] as usize), verts.get(tri[1] as usize), verts.get(tri[2] as usize)) else { continue };
                        ed.soup.add_tri_owned(m.transform_point3(Vec3::from(a.pos)), m.transform_point3(Vec3::from(b.pos)), m.transform_point3(Vec3::from(cc.pos)), (0, 0, -1), u32::MAX);
                    }
                }
            }
            // frame bounds: every placement's transformed local box
            for m in &mats {
                let (mut wmn, mut wmx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for k in 0..8 {
                    let p = Vec3::new(if k & 1 == 0 { mn.x } else { mx.x }, if k & 2 == 0 { mn.y } else { mx.y }, if k & 4 == 0 { mn.z } else { mx.z });
                    let w = m.transform_point3(p);
                    wmn = wmn.min(w); wmx = wmx.max(w);
                }
                if wmn.is_finite() && wmx.is_finite() { mesh_boxes.push((wmn, wmx)); }
            }
            st.meshes += 1;
            // draw each part with its own diffuse texture (one part = one material = one draw).
            for part in &dm.parts {
                let s = part.index_start as usize;
                let e = (s + part.index_count as usize).min(dm.indices.len());
                if e < s + 3 { continue; }
                let idx = &dm.indices[s..e];
                // resolve this part's diffuse bitmap -> texture view (both cached).
                let render_tag = usize::try_from(part.material).ok().and_then(|mi| bsp.materials.get(mi)).and_then(|bm| bm.render_tag);
                if let Some(only) = tc.onlymat.as_deref() {
                    let bm = usize::try_from(part.material).ok().and_then(|mi| bsp.materials.get(mi));
                    let hit = bm.map_or(false, |bm| bm.name.contains(only) || String::from_utf8_lossy(&bm.class).contains(only));
                    if !hit { continue; }
                }
                // rmw water: the water pass (transparent, sky reflection, refraction) #h3-water
                if let Some(rt) = render_tag.filter(|_| usize::try_from(part.material).ok().and_then(|mi| bsp.materials.get(mi)).map_or(false, |bm| &bm.class == b"rmw ")) {
                    if let Some(gm) = upload_water_part(mr, device, queue, &mut tc, rt, &verts, idx, &mats, log) {
                        st.draws += 1;
                        st.tris += idx.len() / 3 * mats.len();
                        st.water_parts += 1;
                        lanes.push(Lane::Water, gm);
                        continue;
                    }
                }
                // rmtr terrain: the blend-map weighted layers through the terrain pass
                if let Some(t) = render_tag.and_then(|rt| tc.terrain_layers(rt)) {
                    st.draws += 1;
                    st.tris += idx.len() / 3 * mats.len();
                    st.tex_ok += 1;
                    st.terrain_parts += 1;
                    for gm in upload_terrain(mr, device, queue, &tc.grey, &verts, idx, &mats, &t) { lanes.push(Lane::Terrain, gm); }
                    continue;
                }
                let mut tex: Option<wgpu::TextureView> = None;
                if !tc.notex {
                    let base = render_tag.and_then(|rt| tc.material(rt).base);
                    if let Some(rt) = render_tag {
                        // #h3-albedo a constant_color material has no base map: its colour is the
                        // authored constant, and a fully transparent alpha_test_map cuts it away
                        match tc.constant_color(rt) {
                            ConstColor::Cut => continue,
                            ConstColor::Color(v) => tex = Some(v),
                            ConstColor::NotConstant => tex = tc.albedo(rt),
                        }
                    }
                    match tex { Some(_) => st.tex_ok += 1, None => { st.flat += 1; if base.is_some() { st.tex_fail += 1; } } }
                } else { st.flat += 1; }
                let lane = tc.lane_for(render_tag, tex.is_some());
                st.draws += 1;
                st.tris += idx.len() / 3 * mats.len();
                // no diffuse: the mid-grey placeholder (an untextured part rendered WHITE, which
                // under the bake gain blew out riverworld's lake bed)
                let tex = tex.or_else(|| (!tc.notex).then(|| tc.grey.clone()));
                lanes.push(lane, upload_part(mr, device, queue, &verts, idx, &mats, tex.as_ref()));
            }
        }
        lanes.flush(&mut st, emit);
        done_units += 1;
        progress(done_units, total_units);
    }
    // the objects' flat light: the exposed mean bake over the whole map (alpha 0 = no analytic
    // sun, like the BSP), else the placeholder sun (alpha 1)
    let obj_color: [f32; 4] = if world_bake_n > 0 {
        [(world_bake[0] / world_bake_n as f64) as f32, (world_bake[1] / world_bake_n as f64) as f32, (world_bake[2] / world_bake_n as f64) as f32, 0.0]
    } else { [1.0; 4] };

    // --- the map's camera FX (cfxs -> the exposure key / band + the bloom curve) #h3-expo ---
    st.camera_fx = super::lighting::load_camera_fx(c);
    match st.camera_fx.as_ref() {
        Some(x) => {
            let (lo, hi) = x.stops_band();
            let d = |b: u8| if x.defaulted >> b & 1 != 0 { "*" } else { "" };
            log(format!("h3 camera fx: '{}' key {:.4} (screen brightness log2 {:.4}){} band [{:.2}, {:.2}] stops, {} | bloom point {}{} inherent {}{} intensity {}{} (* = cfxs 'use default' -> globals\\default)",
                x.name.rsplit('\\').next().unwrap_or(""), x.key(), x.screen_brightness, d(0), lo, hi,
                if x.auto() { format!("auto-adjust, sensitivity {}{}", x.sensitivity, d(1)) } else { format!("MANUAL {} stops", x.manual) },
                x.bloom_point, d(2), x.bloom_inherent, d(3), x.bloom_intensity, d(4)));
        }
        None => log("h3 camera fx: the scenario names no cfxs (built-in exposure / bloom defaults)".to_string()),
    }

    // --- the map's atmosphere (skya -> the renderer's fog uniform, lighting.rs) #h3-fog ---
    let scene_gain = if world_bake_n > 0 { (world_gain / world_bake_n as f64) as f32 } else { 1.0 };
    if std::env::var("HMS_H3_NOFOG").is_err() {
        if let Some(a) = super::lighting::scene_atmosphere(c) {
            let f = a.fog_uniform(scene_gain, 1.0 / st.expo.max(1e-3));
            log(format!("h3 fog: atmosphere '{}' (engine two-species) -> inscatter [{:.3} {:.3} {:.3}] bias {:.1} sun/beta [{:.3} {:.3} {:.3}] max {:.0} | beta_m {:?} beta_p {:?} ref {:.1} mie_h {:.1} ray_h {:.1} g {:.2} sun_dir [{:.2} {:.2} {:.2}] (scene gain {scene_gain:.2})",
                a.name, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], &f[8..11], &f[12..15], f[11], f[19], f[23], f[27] * 0.5, f[24], f[25], f[26]));
            st.fog = Some(f);
        } else {
            log("h3 fog: no enabled atmosphere setting selected (clear)".to_string());
        }
    }

    // --- the scenario sky (renderer sky lane) ---
    if !nosky {
        if cancelled() { anyhow::bail!("cancelled"); }
        let sky_gain = scene_gain;
        let sky = build_sky(c, device, queue, mr, &mut tc, &mut st, sky_gain, log);
        if !sky.is_empty() { st.lane_add(Lane::Sky, sky.len()); emit(Lane::Sky, sky); }
    }
    done_units += 1;
    progress(done_units, total_units);

    // --- the scenario objects ---
    field.fallback = [obj_color[0], obj_color[1], obj_color[2]];
    log(format!("h3 object light: {} lightmapped triangles over {} BSP lightmap(s), flat fallback {:?}",
        field.tri_count(), field.lightmap_count(), field.fallback));
    if !noobj {
        if cancelled() { anyhow::bail!("cancelled"); }
        build_objects(c, device, queue, mr, &mut tc, &mut st, obj_color, &field, loaded.scenario_owned && editor.is_some(), emit, log);
        // the map variant's Forge objects, through the same model path #h3 (the GUI editor scene
        // draws them itself: dynamic, pickable lanes)
        if let (Some(v), None) = (loaded.variant.as_ref(), editor.as_ref()) {
            if cancelled() { anyhow::bail!("cancelled"); }
            build_variant_objects(c, v, device, queue, mr, &mut tc, &mut st, obj_color, &field, emit, log);
        }
    }
    done_units += 1;
    progress(done_units, total_units);

    log(format!("h3: uploaded {} meshes / {} part draws ({} placed triangles); instances: {} drawn, {} skipped; \
        textures: {} bound, {} flat ({} resolve->upload failures), {} bitmaps cached; objects: {} placed, {} skipped, {} attachments; variant objects: {} placed, {} skipped; sky: {} parts; terrain: {} parts; water: {} parts",
        st.meshes, st.draws, st.tris, st.instances_drawn, st.instances_skipped, st.tex_ok, st.flat, st.tex_fail, tc.uploaded(),
        st.objects_placed, st.objects_skipped, st.attachments, st.variant_placed, st.variant_skipped, st.sky_parts, st.terrain_parts, st.water_parts));
    log(format!("h3 albedo: {} materials composited (engine calc_albedo_*: base x detail x DETAIL_MULTIPLIER), {} bound as a plain base_map (HMS_H3_NOALBEDO reverts); constant_color: {} parts bound to their albedo_color, {} alpha-tested away", tc.alb_made, tc.alb_plain, tc.const_bound, tc.const_cut));
    log(format!("h3: lighting: {} uses per-pixel bake, {} per-vertex bake, {} mean-bake flat, {} placeholder sun", st.lm_pp, st.lm_pv, st.lm_mean, st.lm_none));
    if let Some(ed) = editor.as_deref_mut() {
        ed.soup.rebalance(8);
        ed.soup.shrink();
        ed.obj_color = obj_color;
        ed.tex = Some(tc);
        log(format!("h3 editor collect: {} BSP triangles in the pick soup", ed.soup.len()));
    }
    // frame the playable clusters: drop meshes whose diagonal is a large outlier (sky / vista)
    let (bmin, bmax) = frame_bounds(&mesh_boxes);
    st.bmin = bmin;
    st.bmax = bmax;
    let (fmin, fmax) = st.frame_bounds();
    log(format!("h3: frame bounds min {:?} max {:?}", fmin.to_array(), fmax.to_array()));
    st.secs = t0.elapsed().as_secs_f32();
    Ok(st)
}

/// The scenario sky: scnr +0x34 skies -> `scen` -> hlmt -> mode, every default-permutation part
/// with a REAL diffuse as a sky-lane segment (camera-follow, unfogged, drawn back-to-front by
/// the segment key). A part without a resolvable base map is skipped - drawn, it could only be
/// a flat white dome hiding every real layer (the H2A Ascension lesson). `alpha_test_map`
/// parts (tree canopies, branch cards) alpha-blend on their base alpha.
fn build_sky(c: &H3Cache, device: &wgpu::Device, queue: &wgpu::Queue, mr: &MeshRenderer, tc: &mut TexCtx, st: &mut H3Stats, sky_gain: f32, log: &mut dyn FnMut(String)) -> Vec<GpuMesh> {
    let mut out = Vec::new();
    let Some(scnr) = c.find_tags(b"scnr").first().copied() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let Some((n, so)) = c.block(sm + OFF_SCNR_SKIES) else { return out };
    for k in 0..n.min(4) {
        let Some(scen) = c.tag_ref_of(so + k * SCNR_SKY_ELEM, b"scen") else { continue };
        let Some(mode) = model_of(c, scen) else { log(format!("h3 sky '{}': no render model", c.tag_name(scen))); continue };
        let model = match load_model(c, mode) { Ok(m) => m, Err(e) => { log(format!("h3 sky '{}': {e}", c.tag_name(mode))); continue } };
        let (mut parts, mut skipped) = (0usize, 0usize);
        for &mi in &model.lod0_meshes {
            let Ok(Some(dm)) = decode_model_mesh(c, &model, mi) else { continue };
            for part in &dm.parts {
                let s0 = part.index_start as usize;
                let e0 = (s0 + part.index_count as usize).min(dm.indices.len());
                if e0 < s0 + 3 { continue; }
                let Some(rt) = usize::try_from(part.material).ok().and_then(|k| model.materials.get(k).copied().flatten()) else { skipped += 1; continue };
                let m = tc.material(rt).clone();
                // the per-vertex colour stream (kind 14 f32x3): the classic vertex-painted dome
                let vcols = model_vertex_colors(c, &model, mi);
                // a constant_color part (albedo option 2, no base map) is REAL only through its
                // vertex colours x `albedo_color`; anything else without a diffuse is skipped
                // (never a flat white dome)
                let constant = m.albedo_opt() == 2 && m.base.is_none() && vcols.is_some();
                let tex = match (m.base.and_then(|b| tc.texture(b)), constant) {
                    (Some(t), _) => t,
                    (None, true) => tc.white.clone(),
                    (None, false) => { skipped += 1; continue }
                };
                let albedo_color = if constant { m.float("albedo_color").unwrap_or([1.0; 4]) } else { [1.0; 4] };
                // the template's blend_mode -> the sky lane's Reach blend enum (0 opaque, 1
                // additive, 2 multiply, 3 double multiply, 4 alpha); an alpha test = alpha
                let blend: u8 = match m.blend() {
                    H3Blend::Additive => 1, H3Blend::Multiply => 2, H3Blend::DoubleMultiply => 3,
                    H3Blend::AlphaBlend | H3Blend::PremulAlpha => 4,
                    H3Blend::Opaque => if m.alpha_test() { 4 } else { 0 },
                };
                // the template's albedo option (the sky shader's switch: 0 default = base x
                // detail x DM, 1 detail_blend, 2 constant_color, 5 three_detail_blend, 7
                // two_detail); the detail layers are the same-named samplers
                let opt = match m.albedo_opt() { o @ (1 | 2 | 5 | 7) => o as f32, _ => 0.0 };
                let detail = m.param("detail_map").and_then(|b| tc.texture(b));
                let detail2 = m.param("detail_map2").and_then(|b| tc.texture(b));
                let detail3 = m.param("detail_map3").and_then(|b| tc.texture(b));
                // self illumination: the template's `self_illum_map` sampler through the lane's
                // simple mode (gain 1: the authored colour/intensity constants are stripped)
                let si = m.param("self_illum_map").and_then(|b| tc.texture(b));
                let si_mode = if si.is_some() { 1.0 } else { 0.0 };
                let si_detail = m.param("self_illum_detail_map").and_then(|b| tc.texture(b));
                let gain = if si.is_some() { [1.0, 1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0, 1.0] };
                // THE AUTHORED LAYER UV XFORMS. These were all pinned to (1,1,0,0), so a cloud
                // sheet authored at 8x8 drew one stretched tile; the sky pass takes one xform per
                // sampler through the parameters below (its VS names them base/detail/detail2/
                // detail3/si/sid - see SKY_MESH_WGSL).
                let xf = |n: &str| m.xform(n);
                // THE SCROLL RATES (uv/s) from the render method's animated parameters
                // (materials.rs `layer_scroll_rates`). The sky pipeline reads its six rate pairs
                // out of the `scroll`+`tile`, `si_ctl` and `mattex_ctl` instance lanes
                // (mesh.rs InstanceRaw::scroll = [scroll.xy, tile.xy]) - passing the usual
                // tile (1,1) there made EVERY Halo 3 sky detail layer translate a full uv per
                // SECOND ("the sky animates way too fast"), while the base layer sat frozen at
                // rate 0 ("completely static"): both reports, one defect. #h3-sky-anim
                let r = super::materials::layer_scroll_rates(&m.anim);
                let verts: Vec<MeshVertex> = dm.verts.iter().enumerate().map(|(k, v)| {
                    let c = vcols.as_ref().and_then(|cs| cs.get(k)).map(|c| [c[0], c[1], c[2]]).unwrap_or([1.0; 3]);
                    // x the scene's bake gain (see build_meshes): same exposure as the lit map
                    let vc = [c[0] * sky_gain, c[1] * sky_gain, c[2] * sky_gain, 1.0];
                    MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: vc, ..Default::default() }
                }).collect();
                let mut gm = mr.upload_mesh(device, queue, &verts, &dm.indices[s0..e0], &[Mat4::IDENTITY], Some(&[gain]),
                    // view -> binding: 0 base, 2 si, 3 detail, 4 si_detail, 12 detail2, 15 detail3
                    Some(&tex), si.as_ref(), detail.as_ref(), si_detail.as_ref(), detail2.as_ref(),
                    r[0] /* scroll_a.xy: base rate */, r[1] /* scroll_a.zw: detail rate */,
                    xf("base_map") /* -> base_xf */, 0.0, 0.0, [0.0, 0.0],
                    xf("detail_map") /* -> detail_xf */, xf("detail_map2") /* -> detail2_xf */,
                    xf("self_illum_map") /* -> si_xf */, xf("self_illum_detail_map") /* -> sid_xf */, albedo_color,
                    None, false, 0.0,
                    [0.0; 4] /* scroll_b: si / si_detail rates (the sky materials animate no self-illum layer) */,
                    None, None,
                    [r[2][0], r[2][1], r[3][0], r[3][1]] /* scroll_c: detail2 / detail3 rates */,
                    xf("detail_map3") /* -> detail3_xf */, detail3.as_ref(), None, [0.0; 4],
                    [opt, si_mode, blend as f32, 0.0], None);
                gm.set_blend_mode(blend);
                gm.set_centroid([-(out.len() as f32), 0.0, 0.0]);
                out.push(gm);
                parts += 1;
            }
        }
        st.sky_parts += parts;
        log(format!("h3 sky '{}': {} parts as sky segments, {} skipped (no diffuse)", model.name.rsplit('\\').next().unwrap_or(&model.name), parts, skipped));
    }
    out
}

/// The scenario objects: every placement of every renderable class, grouped by palette tag
/// (one model decode + one draw per part, carrying every placement matrix), flat-lit by the
/// map's exposed mean bake. Unresolved / undecodable objects are counted with their reason.
fn build_objects(c: &H3Cache, device: &wgpu::Device, queue: &wgpu::Queue, mr: &MeshRenderer, tc: &mut TexCtx, st: &mut H3Stats, color: [f32; 4],
    field: &super::objlight::H3LightField, editor_owns_scenario: bool, emit: &mut dyn FnMut(Lane, Vec<GpuMesh>), log: &mut dyn FnMut(String)) {
    let objdiag = std::env::var("HMS_H3_OBJDIAG").is_ok();
    let placements = load_placements(c);
    // palette tag -> placement matrices (table order kept for the diag)
    let mut groups: Vec<(usize, Vec<Mat4>)> = Vec::new();
    let mut skips: HashMap<&'static str, usize> = HashMap::new();
    let mut editor_owned = 0usize;
    for p in &placements {
        // #h3-scenario the base map's own variant-registered objects are editor objects (selectable,
        // deletable, movable: scenario.rs), so the static scenery must not draw them as well
        if editor_owns_scenario && super::scenario::editor_owns(c, p.class, p.palette_tag) {
            editor_owned += 1;
            continue;
        }
        let reason = if p.never_placed() { Some("never-placed flag") }
            else if !p.pos.iter().chain(p.rot.iter()).all(|v| v.is_finite()) { Some("non-finite placement") }
            else { None };
        if objdiag {
            eprintln!("  H3OBJ {} '{}' {} pos [{:.2} {:.2} {:.2}] rot [{:.2} {:.2} {:.2}] scale {:.2} flags {:#x}{}",
                p.class_str(), p.name, c.tag_name(p.palette_tag).rsplit('\\').next().unwrap_or(""), p.pos[0], p.pos[1], p.pos[2],
                p.rot[0], p.rot[1], p.rot[2], p.scale, p.flags, reason.map(|r| format!(" SKIP: {r}")).unwrap_or_default());
        }
        if let Some(r) = reason { *skips.entry(r).or_default() += 1; st.objects_skipped += 1; continue; }
        let m = placement_matrix(p);
        match groups.iter_mut().find(|(t, _)| *t == p.palette_tag) {
            Some((_, v)) => v.push(m),
            None => groups.push((p.palette_tag, vec![m])),
        }
    }
    let (placed, skipped, models) = draw_object_groups(c, &groups, device, queue, mr, tc, st, color, field, emit, &mut skips, objdiag);
    st.objects_placed += placed;
    st.objects_skipped += skipped;
    log(format!("h3 objects: {} placements -> {} placed over {models} models ({} distinct palette tags), {} skipped {:?}, {editor_owned} left to the editor (variant slots)",
        placements.len(), st.objects_placed, groups.len(), st.objects_skipped, skips));
}

/// The map variant's Forge objects: every placed slot resolved through its quota to the object
/// definition tag (mvar.rs `resolve_placed`), grouped by tag and drawn like the scenario
/// objects. Read-only: nothing here feeds an editor scene. #h3
fn build_variant_objects(c: &H3Cache, v: &super::mvar::H3Variant, device: &wgpu::Device, queue: &wgpu::Queue, mr: &MeshRenderer, tc: &mut TexCtx,
    st: &mut H3Stats, color: [f32; 4], field: &super::objlight::H3LightField, emit: &mut dyn FnMut(Lane, Vec<GpuMesh>), log: &mut dyn FnMut(String)) {
    let objdiag = std::env::var("HMS_H3_OBJDIAG").is_ok();
    let (resolved, unresolved) = super::mvar::resolve_placed(c, v);
    let mut groups: Vec<(usize, Vec<Mat4>)> = Vec::new();
    for (tag, m, slot) in &resolved {
        if objdiag {
            let p = m.w_axis;
            eprintln!("  H3VAR slot {slot} {} pos [{:.2} {:.2} {:.2}]", c.tag_name(*tag).rsplit('\\').next().unwrap_or(""), p.x, p.y, p.z);
        }
        match groups.iter_mut().find(|(t, _)| t == tag) {
            Some((_, v)) => v.push(*m),
            None => groups.push((*tag, vec![*m])),
        }
    }
    let mut skips: HashMap<&'static str, usize> = HashMap::new();
    if unresolved > 0 { skips.insert("no quota tag", unresolved); }
    let (placed, skipped, models) = draw_object_groups(c, &groups, device, queue, mr, tc, st, color, field, emit, &mut skips, objdiag);
    st.variant_placed += placed;
    st.variant_skipped += skipped + unresolved;
    log(format!("h3 variant '{}': {} placed slots -> {} placed over {models} models ({} distinct tags), {} skipped {:?}",
        v.header.name, v.placed().count(), st.variant_placed, groups.len(), st.variant_skipped, skips));
}

/// Draw (object definition tag, placement matrices) groups: one model decode + one draw per
/// part carrying every matrix. Returns (objects placed, objects skipped, models drawn); the
/// skip reasons accumulate in `skips`.
fn draw_object_groups(c: &H3Cache, groups: &[(usize, Vec<Mat4>)], device: &wgpu::Device, queue: &wgpu::Queue, mr: &MeshRenderer, tc: &mut TexCtx,
    st: &mut H3Stats, color: [f32; 4], field: &super::objlight::H3LightField, emit: &mut dyn FnMut(Lane, Vec<GpuMesh>), skips: &mut HashMap<&'static str, usize>, objdiag: bool) -> (usize, usize, usize) {
    let mut lanes = LaneBatches::default();
    let (mut placed, mut skipped, mut models) = (0usize, 0usize, 0usize);
    // the objects in their default hlmt variant, plus every variant CHILD (a Warthog's turret)
    // at its marker pose (attach.rs) - the children are drawn as groups of their own
    let mut work: Vec<(usize, Option<usize>, Vec<Mat4>, bool)> = Vec::new();
    let mut children: Vec<((usize, Option<usize>), Vec<Mat4>)> = Vec::new();
    for (tag, mats) in groups {
        let variant = super::attach::object_variant_index(c, *tag, 0);
        for m in mats {
            for a in super::attach::expand_attachments(c, *tag, variant, *m) {
                match children.iter_mut().find(|(k, _)| *k == (a.obje, a.variant)) { Some((_, v)) => v.push(a.world), None => children.push(((a.obje, a.variant), vec![a.world])) }
            }
        }
        work.push((*tag, variant, mats.clone(), true));
    }
    let n_children: usize = children.iter().map(|(_, v)| v.len()).sum();
    if n_children > 0 { st.attachments += n_children; }
    for ((tag, variant), mats) in children { work.push((tag, variant, mats, false)); }
    for (tag, variant, mats, counts) in &work {
        let (tag, mats) = (*tag, mats);
        let short = c.tag_name(tag).rsplit('\\').next().unwrap_or("").to_string();
        let Some(mode) = model_of(c, tag) else { *skips.entry("no render model").or_default() += 1; if *counts { skipped += mats.len(); } if objdiag { eprintln!("  H3OBJ {short}: no hlmt/mode"); } continue };
        let model = match load_model(c, mode) {
            Ok(m) => m,
            Err(e) => { *skips.entry("model load failed").or_default() += 1; if *counts { skipped += mats.len(); } if objdiag { eprintln!("  H3OBJ {short}: {e}"); } continue }
        };
        // HMS_H3_DAMAGE=<0..4> (diagnostic): draw every scenario object in that DAMAGE display
        // state, so the headless capture can A/B the state swap the editor's DAMAGE dropdown
        // drives through `H3ObjectScene` (which the headless path does not go through). #damage
        let dmg: u8 = std::env::var("HMS_H3_DAMAGE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let meshes = if dmg > 0 && super::attach::variant_damage_states(c, tag, *variant).contains(&dmg) {
            if objdiag { eprintln!("  H3OBJ {short}: damage state {dmg}"); }
            super::attach::variant_state_meshes(c, tag, &model, *variant, dmg)
        } else {
            super::attach::variant_meshes(c, tag, &model, *variant)
        };
        // one structure-lightmap probe per placement (the engine's object lighting point)
        let probes = field.sample_mats(mats);
        if objdiag {
            let srcs = probes.iter().filter(|p| p.source != super::objlight::ProbeSource::Fallback).count();
            eprintln!("  H3OBJ {short}: {} placements, {srcs} lit by a structure probe, sh0[0] {:?}", mats.len(), probes.first().map(|p| p.sh0()));
        }
        let mut drawn = 0usize;
        for &mi in &meshes {
            let dm = match decode_model_mesh(c, &model, mi) { Ok(Some(d)) => d, Ok(None) => continue, Err(e) => { if objdiag { eprintln!("  H3OBJ {short} mesh {mi}: {e}"); } continue } };
            if dm.verts.is_empty() || dm.indices.len() < 3 { continue; }
            let verts: Vec<MeshVertex> = dm.verts.iter().map(|v| MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color, ..Default::default() }).collect();
            for part in &dm.parts {
                let s = part.index_start as usize;
                let e = (s + part.index_count as usize).min(dm.indices.len());
                if e < s + 3 { continue; }
                let render_tag = usize::try_from(part.material).ok().and_then(|k| model.materials.get(k).copied().flatten());
                let mut tex = None;
                if !tc.notex {
                    let base = render_tag.and_then(|rt| tc.material(rt).base);
                    if let Some(bt) = base { tex = tc.texture(bt); }
                    match tex { Some(_) => st.tex_ok += 1, None => { st.flat += 1; if base.is_some() { st.tex_fail += 1; } } }
                } else { st.flat += 1; }
                let lane = tc.lane_for(render_tag, tex.is_some());
                st.draws += 1;
                st.tris += (e - s) / 3 * mats.len();
                let tex = tex.or_else(|| (!tc.notex).then(|| tc.grey.clone()));
                // the engine object lane (probe SH + cook-torrance + env), not the flat BSP one
                lanes.push(lane, upload_object_part(mr, device, queue, tc, render_tag, &verts, &dm.indices[s..e], mats, tex.as_ref(), &probes, color));
                drawn += 1;
            }
        }
        if drawn == 0 { *skips.entry("no decodable part").or_default() += 1; if *counts { skipped += mats.len(); } continue; }
        models += 1;
        if *counts { placed += mats.len(); }
    }
    lanes.flush(st, emit);
    (placed, skipped, models)
}
