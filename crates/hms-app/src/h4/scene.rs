//! Halo 4 scene build: cache -> BSPs -> GPU meshes, shared by the headless screenshot
//! (render.rs) and the GUI load worker (gui.rs).
//!
//! `load_map` parses the cache + every sbsp (cheap, CPU only). `build_meshes` decodes each mesh
//! once, uploads one GpuMesh per part with all its instance matrices, and hands finished batches
//! to the caller's sink per render lane (opaque / alpha-test / blend / additive) so the GUI can
//! stream them in while the load runs on a worker thread. Lane routing is by material kind
//! (materials.rs) - everything is opaque until a kind says otherwise.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use eframe::wgpu;
use glam::{Mat4, Vec3};
use hms_render::{GpuMesh, LightmapInputs, MeshRenderer, MeshVertex};

use super::bitmaps::{bitmap_dxgi, bitmap_info, dxgi_is_srgb, load_base_texture, load_cube_faces, Texel};
use super::cache::H4Cache;
use super::shading::{Boundary, H4Shading};
use super::geometry::{decode_mesh, load_bsp, H4Bsp};
use super::lightmaps::{airprobe_sample, atlas_texel_lobes, compose_atlas_textures, engine_stats, instance_pervertex, instance_probe, instance_uv_vb, load_airprobes, load_atlas, load_pervertex, load_probes, load_vertex_ao, mesh_lightmap_uvs, pervertex_colors, probe_vertex_colors, sharpen_lut, H4AirProbe, H4LightmapAtlas, H4PerVertexLighting, H4Probe, LBSP_FLAG_FLOATING_SUN, OFF_LBSP_SUN_DIR, OFF_LBSP_SUN_RGB};
use super::materials::{load_bsp_materials, load_material, print_material_diag, H4MatKind, H4Material};
use super::objects::{decode_model_mesh, load_model, load_placements, model_of, placement_matrix, H4Placement};
use super::geometry::Part;
use super::cache::ByteRead;

/// Scenario object classes placed as static render models (their `mode` at the placement pose).
/// Sound scenery / spawners / effect scenery carry no model worth drawing.
pub const OBJECT_CLASSES: [&[u8; 4]; 8] = [b"scen", b"bloc", b"vehi", b"weap", b"eqip", b"crea", b"mach", b"ctrl"];

// #h4-veh
/// The change colour a part takes when the caller resolves none (BSP surfaces, the `*_colorchangemap`
/// Forge pieces before any team is placed): the engine's neutral primary
/// `ps_material_object_parameters[0]` = (0.5, 0.5, 0.5, 1) - see `h4::tint`.
pub const NEUTRAL_CC: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

/// Renderer lane a part is routed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    Opaque,
    AlphaTest,
    Blend,
    Additive,
    /// The scenario sky object's parts (renderer sky lane; `GpuMesh::blend_mode` carries
    /// the Reach sky blend enum, `centroid().x` the draw order).
    Sky,
}

pub struct H4Loaded {
    /// Shared with the editor scene (`H4EditorAssets`) once the load worker is done.
    pub cache: Arc<H4Cache>,
    pub bsps: Vec<H4Bsp>,
    /// Per BSP (same order as `bsps`): the classified materials, parallel to `H4Bsp::materials`.
    pub materials: Vec<Vec<H4Material>>,
    /// Every scnr placement (all classes; `build_meshes` draws the OBJECT_CLASSES ones).
    pub placements: Vec<H4Placement>,
    /// Forge objects of the loaded map variant (mvar.rs `variant_placements`), drawn
    /// through the same object path as the scenario placements; empty without a variant.
    pub variant_placements: Vec<H4Placement>,
    /// (variant file name, title, resolution counters) when a variant was loaded.
    pub variant: Option<(String, String, super::mvar::H4VariantStats)>,
    /// The parsed variant (its path + every decoded record) kept for the editor.
    pub variant_full: Option<(std::path::PathBuf, super::mvar::H4Variant)>,
    /// Forge objects HMS placed from the palette (`HMS_H4_PLACE` diag; the editor
    /// creates its own through `palette::instantiate`), drawn statically like the scenario ones
    /// with their hlmt model variant's permutations.
    pub forge: Vec<super::palette::H4ForgeInstance>,
    /// Baked sun of the first Lbsp that carries one: (direction the light TRAVELS, rgb intensity)
    /// from Lbsp +0x24 / +0x30 (lightmaps.rs; agrees with the analytic map on both MP maps).
    pub sun: Option<([f32; 3], [f32; 3])>,
    /// Human-readable load facts (one line per BSP), also printed by the callers.
    pub log: Vec<String>,
}

/// Lbsp +0x24 sun direction (unit, light travel direction) + +0x30 rgb, when present.
fn lbsp_sun(c: &H4Cache, lbsp_tag: usize) -> Option<([f32; 3], [f32; 3])> {
    let m = c.tag_meta(lbsp_tag)?;
    let d = c.data();
    let dir = [d.f32_at(m + OFF_LBSP_SUN_DIR), d.f32_at(m + OFF_LBSP_SUN_DIR + 4), d.f32_at(m + OFF_LBSP_SUN_DIR + 8)];
    let rgb = [d.f32_at(m + OFF_LBSP_SUN_RGB), d.f32_at(m + OFF_LBSP_SUN_RGB + 4), d.f32_at(m + OFF_LBSP_SUN_RGB + 8)];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    // The engine builds the floating-sun record when Lbsp flags bit 0x200 is set
    // (halo4.dll sub_18035F24C byte 0); bit 0x2000 only disables the shadow cascade
    let flags = d.u16_at(m);
    let enabled = flags & LBSP_FLAG_FLOATING_SUN != 0;
    let ok = enabled && (len - 1.0).abs() < 0.05 && rgb.iter().all(|v| v.is_finite() && *v >= 0.0 && *v < 1000.0) && rgb.iter().any(|v| *v > 0.0);
    ok.then_some((dir, rgb))
}

/// Which renderer lane a material kind draws in (None = not drawn).
fn lane_for(kind: H4MatKind) -> Option<Lane> {
    match kind {
        H4MatKind::Opaque => Some(Lane::Opaque),
        H4MatKind::AlphaTest => Some(Lane::AlphaTest),
        H4MatKind::AlphaBlend | H4MatKind::Multiply => Some(Lane::Blend),
        H4MatKind::Additive => Some(Lane::Additive),
        H4MatKind::Invisible => None,
    }
}

/// Upload one part into its lane. Blend kinds carry the material's constant alpha in the
/// per-instance tint (the blend shader's opacity = base.a x tint.a); Multiply selects the
/// renderer's Zero/SrcColor fixed-function blend (GpuMesh::blend_mode 2). `lm` binds the
/// Halo 4 atlas decode (lightmaps.rs; hdr = -K_direct is the shader's Halo 4 sentinel).
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_part(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[MeshVertex], indices: &[u32], mats: &[Mat4],
    tex: Option<&wgpu::TextureView>, kind: H4MatKind, base_alpha: f32, lm: Option<LightmapInputs>) -> GpuMesh {
    upload_part_tinted(mr, device, queue, verts, indices, mats, tex, kind, base_alpha, lm, None)
}

/// The emissive tint of an UNLIT transparent family (`srf_ca_color_warp*`,
/// `srf_constant*`, `srf_ca_skybox*`, `srf_plasma*`: `diffuse * p0.rgb * p0.w + selfillum * p1.rgb *
/// p1.w`, no lighting at all - sky shader asm, h4expo/sky); the shipped parts bind the same bitmap
/// for both maps, so the two tints sum. None for lit families (the flat placeholder stays).
pub(crate) fn emissive_tint(m: &H4Material) -> Option<[f32; 3]> {
    let fam = m.mats_name.rsplit('\\').next().unwrap_or("").to_ascii_lowercase();
    if !(fam.starts_with("srf_ca_color_warp") || fam.starts_with("srf_constant") || fam.starts_with("srf_ca_skybox") || fam.starts_with("srf_plasma")) { return None; }
    let f = |i: usize| m.floats.get(i).map(|f| f.1).unwrap_or([0.0; 4]);
    let (p0, p1) = (f(0), f(1));
    let t = [p0[0] * p0[3] + p1[0] * p1[3], p0[1] * p0[3] + p1[1] * p1[3], p0[2] * p0[3] + p1[2] * p1[3]];
    if t.iter().all(|v| v.is_finite() && *v >= 0.0) { Some(t) } else { None }
}

/// `upload_part` with an optional emissive tint: an UNLIT transparent part in the additive lane
/// is drawn as a static glow `(base^2.2 * tint) * base.a` (fs_holo's static path: no rim / pulse
/// / radial vignette, no sun) - the engine's srf_ca_color_warp / srf_constant law minus the
/// self-illum exposure lerp (PROVISIONAL). Without the tint the foam / wave / waterfall sheets of
/// Tower and Forge Island glow at full texture brightness with the billboard gimmicks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_part_tinted(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[MeshVertex], indices: &[u32], mats: &[Mat4],
    tex: Option<&wgpu::TextureView>, kind: H4MatKind, base_alpha: f32, lm: Option<LightmapInputs>, emissive: Option<[f32; 3]>) -> GpuMesh {
    let t = emissive.unwrap_or([1.0; 3]);
    let tints: Vec<[f32; 4]> = mats.iter().map(|_| [t[0], t[1], t[2], base_alpha.clamp(0.0, 1.0)]).collect();
    let colors = if matches!(kind, H4MatKind::AlphaBlend | H4MatKind::Multiply | H4MatKind::Additive) { Some(tints.as_slice()) } else { None };
    // Transparent lanes: blend_shade's default BSP body is only the /pi analytical sun (the Reach
    // glass lane), which renders the Ravine foam sheets as a dark cloud. The "area" lane
    // (xbump.w = 1, body = xbump.rgb * pi) with xbump.rgb = 1/pi lights them flat like the opaque
    // placeholder (body = albedo); no baked term in either path (a stand-in, not the engine law).
    let xbump = if matches!(kind, H4MatKind::AlphaBlend) { [0.318_309_9, 0.318_309_9, 0.318_309_9, 1.0] } else { [0.0; 4] };
    // additive + emissive family: fs_holo's static lane (bump_xform.w = 3)
    let bump_xform = if kind == H4MatKind::Additive && emissive.is_some() { [1.0, 1.0, 0.0, 3.0] } else { [0.0; 4] };
    let mut gm = mr.upload_mesh(device, queue, verts, indices, mats, colors, tex, None, None, None, None,
        [0.0, 0.0], [1.0, 1.0], [0.0; 4], 0.0, 0.0, [0.0, -1.0], [0.0; 4], bump_xform, [0.0; 4], [0.0; 4], [0.0; 4],
        lm, false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, xbump, [0.0; 4], None);
    if kind == H4MatKind::Multiply { gm.set_blend_mode(2); }
    gm
}

// #grid-render
/// `srf_ca_boundary*` (the Halo 4 Forge GRID piece, the boundary walls) through the renderer's
/// additive halogram lane (`fs_holo`, Halo 4 boundary mode `bump_xform.w == 8`). The law is the
/// family's forward transparent pixel shader, transcribed in the `Boundary` doc; the lane packing
/// below is the one the WGSL branch reads. Texture slots follow the Reach halogram convention:
/// base = noise A, emis = noise B, detail = palette, bump = overlay, mat = overlay detail,
/// at = self-illum alpha mask (its ALPHA channel).
pub(crate) fn upload_part_boundary(tc: &mut TexCtx, mr: &MeshRenderer, verts: &[MeshVertex], indices: &[u32],
    mats: &[Mat4], b: &Boundary, stats: &mut H4Stats) -> GpuMesh {
    let (device, queue) = (tc.device.clone(), tc.queue.clone());
    let (noise_a, ga) = tc.get(b.noise_a, stats);
    let (noise_b, gb) = tc.get(b.noise_b, stats);
    let (palette, gp) = tc.get(b.palette, stats);
    let (overlay, go) = tc.get(b.overlay, stats);
    let (odetail, gd) = tc.get(b.overlay_detail, stats);
    let (amask, _) = tc.get(b.alpha_mask, stats);
    let f = |v: bool| if v { 1.0 } else { 0.0 };
    let xa = b.noise_a_xform;
    let xf5 = b.palette_xform;
    // scroll / tile / mattex.xy = noise A xform (the renderer's halogram convention);
    // bump_xform.w = 8 selects the Halo 4 branch, .xy = noise B scroll, .z is unused (mode 0).
    let mut gm = mr.upload_mesh(&device, &queue, verts, indices, mats, None,
        noise_a.as_ref(), noise_b.as_ref(), palette.as_ref(), overlay.as_ref(), None,
        [0.0, 0.0],                                            // noise A scroll (per-second)
        [xa[0], xa[1]],                                        // noise A tile
        b.noise_b_xform,                                       // detail_xform = noise B xform
        0.0, 0.0, [0.0, 0.0],
        [b.si_intensity, b.fres_scale, b.fres_power, b.fres_invert],   // mat_spec2 = up161
        [0.0, 0.0, 0.0, 8.0],                                  // bump_xform = [noise B scroll, mode, H4 flag]
        xf5,                                                   // fine_xform = palette xform
        b.overlay_xform,                                       // bump_detail_xform = overlay xform
        b.overlay_detail_xform,                                // env_ctl = overlay detail xform
        None, false, 0.0,
        [b.tint[0], b.tint[1], b.tint[2], b.overlay_intensity], // si_ctl = up160
        amask.as_ref(), odetail.as_ref(),
        [xa[2], xa[3], 0.0, 0.0],                              // mattex = [noise A offset, overlay scroll]
        b.alpha_mask_xform,                                    // mattex_xform = alpha mask xform
        None, None,
        [b.depth_fade_range, b.palette_power, b.height_fade, b.noise_diff_scale],  // xbump = up163
        [f(ga), f(gb), f(gp), f(go)],                          // xctl = gamma flags of the four
        None);
    // spec_rgb = up162 (the output fade fresnel + the palette v); fres_rgb / obj_probe0 carry the
    // remaining gamma flags and the overlay-detail scroll.
    gm.set_h4_lanes(&queue, [b.fade_scale, b.fade_power, b.fade_invert, b.palette_v],
        [f(gd), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0; 4]);
    gm
}

/// GPU textures of one Halo 4 material, resolved per shading role (scene texture cache).
#[derive(Default)]
pub(crate) struct H4PartTex {
    base: Option<wgpu::TextureView>,
    base_srgb: bool,
    normal: Option<wgpu::TextureView>,
    normal_detail: Option<wgpu::TextureView>,
    color_detail: Option<wgpu::TextureView>,
    color_detail_srgb: bool,
    /// specular_map / control map (the shading's `spec_src` says which), or the layered blend map.
    spec: Option<wgpu::TextureView>,
    spec_srgb: bool,
    /// specular_map when a control map is ALSO bound (spec_src 5).
    spec2: Option<wgpu::TextureView>,
    /// self-illum map | snow color1 | layer1 co.
    emis: Option<wgpu::TextureView>,
    emis_srgb: bool,
    /// snow normal1 | layer2 co.
    at: Option<wgpu::TextureView>,
    at_srgb: bool,
    /// layer2 nm.
    bd3: Option<wgpu::TextureView>,
    cube: Option<wgpu::TextureView>,
}

/// Texture / shading resolution state of one `build_meshes` run (GPU uploads are cached per
/// bitmap). Owns its cache / device / queue handles so the editor scene (h4/edit_scene.rs) can
/// keep one alive across rebuilds (textures are uploaded once per bitmap, like the Reach scene).
pub(crate) struct TexCtx {
    pub(crate) cache: Arc<H4Cache>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    notex: bool,
    tex_cache: HashMap<usize, (Option<wgpu::TextureView>, bool)>,
    cube_cache: HashMap<usize, Option<wgpu::TextureView>>,
    shading_cache: HashMap<usize, H4Shading>,
    /// Linear-format sky bitmaps uploaded gamma-encoded (`fetch_tex_sky`).
    sky_cache: HashMap<usize, Option<wgpu::TextureView>>,
    shade_diag: bool,
    white_tex: wgpu::TextureView,
}

impl TexCtx {
    /// An empty texture context (HMS_H4_NOTEX = no textures, HMS_H4_MATDIAG = print every material's shading).
    pub(crate) fn new(cache: Arc<H4Cache>, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        TexCtx {
            cache, device: device.clone(), queue: queue.clone(),
            notex: std::env::var("HMS_H4_NOTEX").is_ok(),
            tex_cache: HashMap::new(), cube_cache: HashMap::new(), shading_cache: HashMap::new(), sky_cache: HashMap::new(),
            shade_diag: std::env::var("HMS_H4_MATDIAG").is_ok(),
            white_tex: hms_render::upload_texture_bgra(device, queue, &[255, 255, 255, 255], 1, 1).0,
        }
    }
    /// A 2D bitmap as a GPU view + its sRGB flag (the DXGI format at def +0x4C).
    pub(crate) fn fetch_tex(&mut self, tag: usize, stats: &mut H4Stats) -> (Option<wgpu::TextureView>, bool) {
        if self.notex { return (None, false); }
        if let Some(v) = self.tex_cache.get(&tag) { return v.clone(); }
        let (cache, device, queue) = (&*self.cache, &self.device, &self.queue);
        let srgb = bitmap_dxgi(cache, tag).map_or(true, dxgi_is_srgb);
        let view = match load_base_texture(cache, tag) {
            Ok(Texel::Dds { bytes, w, h, mips }) => match hms_render::upload_texture_dds(device, queue, &bytes) {
                Some((view, _tex)) => { stats.tex_ok += 1; Some(view) }
                None => { stats.tex_fail.push(format!("{} ({w}x{h} mips {mips}): GPU BC upload refused", cache.tag_name(tag))); None }
            },
            Ok(Texel::Bgra { bytes, w, h }) => { stats.tex_ok += 1; Some(hms_render::upload_texture_bgra(device, queue, &bytes, w, h).0) }
            Err(e) => { stats.tex_fail.push(format!("{}: {e}", cache.tag_name(tag))); None }
        };
        self.tex_cache.insert(tag, (view.clone(), srgb));
        (view, srgb)
    }
    /// A bitmap for the renderer's SKY lane, which decodes every texel with `pow(2.2)` (Reach's
    /// colour maps are all sRGB). Halo 4 binds its sky maps with the per-bitmap DXGI format: the
    /// *_SRGB ones (Redoubt `newbase_diff` 72, Ravine's dome 78) are right as they are, the
    /// LINEAR ones (Haven's cloud sheets 77, Ravine's clouds 87, Forge Island's horizon 77) must
    /// NOT be decoded - they are decompressed on the CPU and uploaded gamma-ENCODED so the lane's
    /// pow(2.2) hands back the raw texel (decoding them makes Haven's sky 1.2-2 stops too dark,
    /// 0.5 -> 0.22).
    pub(crate) fn fetch_tex_sky(&mut self, tag: usize, stats: &mut H4Stats) -> Option<wgpu::TextureView> {
        if self.notex { return None; }
        let srgb = bitmap_dxgi(&self.cache, tag).map_or(true, dxgi_is_srgb);
        if srgb { return self.fetch_tex(tag, stats).0; }
        if let Some(v) = self.sky_cache.get(&tag) { return v.clone(); }
        let (cache, device, queue) = (&*self.cache, &self.device, &self.queue);
        let info = bitmap_info(cache, tag);
        let view = match (load_base_texture(cache, tag), info) {
            (Ok(Texel::Dds { bytes, w, h, .. }), Some(i)) => {
                let lvl = &bytes[148..];
                let bgra = match i.format {
                    14 => Some(super::bitmaps::decode_bc1_bgra(lvl, w, h)),
                    16 => Some(super::lightmaps::decode_bc3_bgra(lvl, w, h)),
                    _ => None,
                };
                match bgra {
                    Some(b) => { stats.tex_ok += 1; Some(hms_render::upload_texture_bgra_gamma_enc(device, queue, &b, w, h).0) }
                    None => self.fetch_tex(tag, stats).0,
                }
            }
            (Ok(Texel::Bgra { bytes, w, h }), _) => { stats.tex_ok += 1; Some(hms_render::upload_texture_bgra_gamma_enc(device, queue, &bytes, w, h).0) }
            _ => self.fetch_tex(tag, stats).0,
        };
        self.sky_cache.insert(tag, view.clone());
        view
    }
    fn get(&mut self, tag: Option<usize>, stats: &mut H4Stats) -> (Option<wgpu::TextureView>, bool) {
        match tag { Some(b) => self.fetch_tex(b, stats), None => (None, false) }
    }
    /// Reflection cube (mip 0 of the six faces, linear floats).
    fn fetch_cube(&mut self, tag: usize, stats: &mut H4Stats) -> Option<wgpu::TextureView> {
        if self.notex { return None; }
        if let Some(v) = self.cube_cache.get(&tag) { return v.clone(); }
        let (cache, device, queue) = (&*self.cache, &self.device, &self.queue);
        let view = match load_cube_faces(cache, tag) {
            Ok(faces) => match hms_render::build_hdr_env_cube(device, queue, &faces) {
                Some((_tex, view)) => { stats.tex_ok += 1; Some(view) }
                None => { stats.tex_fail.push(format!("{}: cube upload refused", cache.tag_name(tag))); None }
            },
            Err(e) => { stats.tex_fail.push(format!("{}: {e}", cache.tag_name(tag))); None }
        };
        self.cube_cache.insert(tag, view.clone());
        view
    }
    /// The material's decoded shading constants alone (cached; no texture uploads) - the
    /// transparent lanes need only the family to route on.
    pub(crate) fn shading_of(&mut self, m: &H4Material) -> H4Shading {
        if !self.shading_cache.contains_key(&m.mat_tag) {
            let s = H4Shading::from_material(m);
            if self.shade_diag { eprintln!("  HMS_H4_MATDIAG {} -> {}", m.name.rsplit('\\').next().unwrap_or(&m.name), super::shading::describe(&s)); }
            self.shading_cache.insert(m.mat_tag, s);
        }
        self.shading_cache[&m.mat_tag].clone()
    }

    /// The material's shading constants + every texture its family samples.
    pub(crate) fn resolve_h4(&mut self, m: &H4Material, stats: &mut H4Stats) -> (H4Shading, H4PartTex) {
        if !self.shading_cache.contains_key(&m.mat_tag) {
            let s = H4Shading::from_material(m);
            if self.shade_diag { eprintln!("  HMS_H4_MATDIAG {} -> {}", m.name.rsplit('\\').next().unwrap_or(&m.name), super::shading::describe(&s)); }
            self.shading_cache.insert(m.mat_tag, s);
        }
        let sh = self.shading_cache[&m.mat_tag].clone();
        let mut t = H4PartTex::default();
        let (b, bs) = self.get(sh.tex.color, stats);
        t.base = b.or_else(|| Some(self.white_tex.clone()));
        t.base_srgb = bs;
        t.normal = self.get(sh.tex.normal, stats).0;
        t.normal_detail = self.get(sh.tex.normal_detail, stats).0;
        let (cd, cds) = self.get(sh.tex.color_detail, stats);
        t.color_detail = cd; t.color_detail_srgb = cds;
        if let Some(l) = sh.layered.clone() {
            t.spec = self.get(l.blend, stats).0;
            let (e, es) = self.get(l.layer_co[1], stats); t.emis = e; t.emis_srgb = es;
            let (a, as_) = self.get(l.layer_co[2], stats); t.at = a; t.at_srgb = as_;
            t.normal_detail = self.get(l.layer_nm[1], stats).0;
            t.bd3 = self.get(l.layer_nm[2], stats).0;
            let (cd, cds) = self.get(l.all_detail, stats); t.color_detail = cd; t.color_detail_srgb = cds;
        } else {
            use super::shading::SpecSource::*;
            let (sp, sps) = match sh.spec_src {
                SpecularMap => self.get(sh.tex.specular, stats),
                ControlSpGlRf | ControlSpGlSi | ControlSpGlRfSiAlpha | SpecMapAndControl | DiffSpec => self.get(sh.tex.control, stats),
                None => (Option::None, false),
            };
            t.spec = sp; t.spec_srgb = sps;
            if sh.spec_src == SpecMapAndControl { let (s2, s2s) = self.get(sh.tex.specular, stats); t.spec2 = s2; t.spec_srgb = s2s; }
            if sh.snow.is_some() {
                let (e, es) = self.get(sh.tex.color1, stats); t.emis = e; t.emis_srgb = es;
                t.at = self.get(sh.tex.normal1, stats).0;
            } else if let Some(cc) = sh.char_cov.clone() {
                // #h4-veh srf_char_cov*: the `at` slot carries spec_detail_map (a plain
                // multiplier on the whole specular term); the family has no self-illum map.
                t.at = self.get(cc.spec_detail, stats).0;
            } else {
                let (e, es) = self.get(sh.tex.self_illum, stats); t.emis = e; t.emis_srgb = es;
                t.at = self.get(sh.tex.pcc, stats).0;   // colour-change amount map (std layout)
            }
            if sh.refl_intensity > 0.0 { t.cube = sh.tex.reflection_cube.and_then(|c| self.fetch_cube(c, stats)); }
        }
        (sh, t)
    }
}

/// True unless HMS_H4_SHADING=0: route Halo 4 materials through the shipped-shader lane.
pub fn shading_requested() -> bool {
    std::env::var("HMS_H4_SHADING").map(|v| v != "0").unwrap_or(true)
}

/// Upload one part through the Halo 4 material lane (mesh.rs `h4_shade`): every lane
/// packing below mirrors the header comment of H4_MESH_WGSL. `lm` binds the lightmap atlas as in
/// `upload_part`; `vcolor_is_blend` = the vertex colour alpha carries the per-vertex blend scalar.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_part_h4(mr: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[MeshVertex], indices: &[u32], mats: &[Mat4],
    tex: &H4PartTex, sh: &H4Shading, kind: H4MatKind, base_alpha: f32, cc: [f32; 4], lm: Option<LightmapInputs>) -> GpuMesh {
    let tints: Vec<[f32; 4]> = mats.iter().map(|_| [1.0, 1.0, 1.0, base_alpha.clamp(0.0, 1.0)]).collect();
    let colors = if matches!(kind, H4MatKind::AlphaBlend | H4MatKind::Multiply) { Some(tints.as_slice()) } else { None };
    // #h4-veh layout 3 = srf_char_cov* (the Covenant body ramp), which repurposes the free
    // std-layout lanes (probe1 / fres_rgb / xbump / aux.x) for its three ramp colours.
    let layout: f32 = if sh.snow.is_some() { 1.0 } else if sh.layered.is_some() { 2.0 } else if sh.char_cov.is_some() { 3.0 } else { 0.0 };
    let mut srgb = 0u32;
    if tex.base_srgb { srgb |= 1; }
    if tex.color_detail_srgb { srgb |= 2; }
    if tex.spec_srgb { srgb |= 4; }
    if tex.emis_srgb && layout == 0.0 { srgb |= 8; }
    if tex.emis_srgb && layout != 0.0 { srgb |= 16; }
    if tex.at_srgb { srgb |= 32; }
    let mut flags = 0u32;
    if tex.normal.is_some() { flags |= 1; }
    if tex.normal_detail.is_some() { flags |= 2; }
    if tex.color_detail.is_some() { flags |= 4; }
    if tex.cube.is_some() { flags |= 8; }
    if tex.spec.is_some() { flags |= 16; }
    if tex.emis.is_some() && layout == 0.0 { flags |= 32; }
    if layout == 1.0 && tex.emis.is_some() && tex.at.is_some() { flags |= 64; }
    if layout == 2.0 && tex.spec.is_some() { flags |= 64; }
    // `at` slot bound: pcc_amount_map (layout 0) or spec_detail_map (layout 3, #h4-veh)
    if (layout == 0.0 || layout == 3.0) && tex.at.is_some() { flags |= 128; }
    let xf = |x: [f32; 4]| x;
    let t = &sh.tex;
    // per-layout lane packing
    let (spec2, spec_rgb, env_ctl, fres_rgb, mattex, aux, si_ctl, xbump, probe0, probe1, fine_xform, mattex_xform, bump_detail_xform, si_mode);
    spec2 = [sh.rough_min, sh.rough_max, sh.spec_tint_by_albedo, sh.spec_intensity];
    spec_rgb = [sh.spec_color[0], sh.spec_color[1], sh.spec_color[2], sh.spec_mask_alpha_weight];
    fres_rgb = [sh.fresnel_scale, sh.fresnel_power, sh.fresnel_weight, sh.fresnel_invert];
    if let Some(sn) = &sh.snow {
        env_ctl = [sh.albedo_tint[0], sh.albedo_tint[1], sh.albedo_tint[2], 0.0];
        mattex = [0.0, 0.0, sh.detail_spec_weight, sh.normal_detail_strength];
        si_ctl = [sn.dir[0], sn.dir[1], sn.dir[2], sn.spec1_power];
        xbump = [sn.coverage_scale, sn.coverage_power, sn.bias, sn.alpha_mask_weight];
        probe0 = [sn.color1_tint[0], sn.color1_tint[1], sn.color1_tint[2], sn.normal_blend];
        probe1 = [sn.spec1_color[0], sn.spec1_color[1], sn.spec1_color[2], sn.spec1_intensity];
        fine_xform = xf(t.normal1_xform);
        mattex_xform = xf(t.color1_xform);
        bump_detail_xform = xf(t.normal_detail_xform);
        si_mode = 0.0;
    } else if let Some(l) = &sh.layered {
        env_ctl = [sh.albedo_tint[0], sh.albedo_tint[1], sh.albedo_tint[2], 0.0];
        mattex = [l.mask_mode as f32, 0.0, l.blend1[1], l.blend2[1]]; // .x = mask mode
        si_ctl = xf(l.layer_nm_xform[2]);
        xbump = sh.diffspec;
        probe0 = [l.layer_tint[1][0], l.layer_tint[1][1], l.layer_tint[1][2], l.blend1[0]];
        probe1 = [l.layer_tint[2][0], l.layer_tint[2][1], l.layer_tint[2][2], l.blend2[0]];
        fine_xform = xf(l.layer_co_xform[2]);
        mattex_xform = xf(l.layer_co_xform[1]);
        bump_detail_xform = xf(l.layer_nm_xform[1]);
        si_mode = 0.0;
    } else {
        env_ctl = [sh.refl_tint[0], sh.refl_tint[1], sh.refl_tint[2], sh.refl_intensity];
        mattex = [sh.env_lit_by_diffuse, sh.refl_normal_blend, sh.diffuse_intensity, sh.normal_detail_strength];
        si_ctl = [sh.self_illum_color[0], sh.self_illum_color[1], sh.self_illum_color[2], sh.self_illum_intensity];
        xbump = sh.diffspec;
        probe0 = [sh.albedo_tint[0], sh.albedo_tint[1], sh.albedo_tint[2], sh.detail_spec_weight];
        probe1 = [0.0; 4];
        fine_xform = xf(if t.control.is_some() { t.control_xform } else { t.specular_xform });
        mattex_xform = xf(if t.pcc.is_some() && t.self_illum.is_none() { t.pcc_xform } else { t.self_illum_xform });
        bump_detail_xform = xf(t.normal_detail_xform);
        si_mode = if sh.self_illum_mode == 0 && tex.emis.is_some() && sh.self_illum_intensity > 0.0 { 3.0 } else { sh.self_illum_mode as f32 };
    }
    aux = [0.0, si_mode, sh.normal_detail_fade_end, sh.normal_detail_fade_start];
    // #h4-veh srf_char_cov*: probe1 = up161 (face-on colour + exponent), fres_rgb = up162
    // (mid-angle), xbump = up163 (grazing + the reflection fresnel power), env_ctl.w = 1 (the
    // reflection intensity is folded into up166.rgb), aux.x = up166.w (reflection saturation),
    // mattex_xform = the spec_detail_map xform. The PRIMARY change colour rides probe1 in the
    // STD layout instead (`ps_material_object_parameters[0]`); it is the neutral (0.5, 0.5, 0.5, 1)
    // for a BSP part and for every caller that does not resolve one.
    let (fres_rgb, xbump_v, probe1, env_ctl, mattex_xform, uv_density) = match &sh.char_cov {
        Some(cv) => (cv.mid, cv.graze, cv.face, [sh.refl_tint[0], sh.refl_tint[1], sh.refl_tint[2], 1.0], xf(cv.spec_detail_xform), cv.refl_saturation),
        None => (fres_rgb, xbump, if layout == 0.0 { cc } else { probe1 }, env_ctl, mattex_xform, 0.0),
    };
    let detail_xform = if let Some(l) = &sh.layered { xf(l.all_detail_xform) } else { xf(t.color_detail_xform) };
    let xctl = [sh.spec_src_code(), srgb as f32, flags as f32, layout];
    let base_tile = [t.color_xform[0].abs().max(1e-4), t.color_xform[1].abs().max(1e-4)];
    let mut gm = mr.upload_mesh(device, queue, verts, indices, mats, colors, tex.base.as_ref(), tex.emis.as_ref(), tex.color_detail.as_ref(), tex.normal.as_ref(), tex.normal_detail.as_ref(),
        [0.0, 0.0], base_tile, detail_xform, uv_density, aux[1], [aux[2], aux[3]], spec2, xf(t.normal_xform), fine_xform, bump_detail_xform, env_ctl,
        lm, false, sh.matmodel_code(), si_ctl, tex.at.as_ref(), tex.spec.as_ref(), mattex, mattex_xform, tex.spec2.as_ref(), tex.bd3.as_ref(), xbump_v, xctl, tex.cube.as_ref());
    gm.set_h4_lanes(queue, spec_rgb, fres_rgb, probe0, probe1);
    if kind == H4MatKind::Multiply { gm.set_blend_mode(2); }
    gm
}

/// True unless HMS_H4_LIGHTMAP=0: the Halo 4 lightmap atlas path (the decode is the engine's own
/// law - lightmaps.rs module doc, halo4.dll + shipped-shader disassembly). The flat placeholder
/// lighting is the fallback for BSPs without a per-pixel atlas and for instances without a UV
/// stream.
pub fn lightmap_requested() -> bool {
    std::env::var("HMS_H4_LIGHTMAP").map(|v| v != "0").unwrap_or(true)
}

/// The composed + uploaded atlas of one BSP (lightmaps.rs `compose_atlas_textures`).
struct BspAtlasGpu {
    atlas: H4LightmapAtlas,
    views: [wgpu::TextureView; 4],
    /// The Lbsp's 5-slice per-vertex lighting array (lightmaps.rs `load_pervertex`).
    pervertex: Option<H4PerVertexLighting>,
    /// The Lbsp's instance light probes (+0xD4) and the per-vertex AO array (+0xAC).
    probes: Vec<H4Probe>,
    vertex_ao: Option<H4PerVertexLighting>,
    /// The composed atlas bytes (the CPU surface-probe sample of placed objects
    /// reads the SAME texels the shader does) and the Lbsp +0x3E8 airprobes.
    cpu: Arc<[(Vec<u8>, u32, u32); 4]>,
    airprobes: Vec<H4AirProbe>,
}

impl BspAtlasGpu {
    fn inputs(&self) -> LightmapInputs {
        LightmapInputs {
            dm: self.views[0].clone(),
            sdm: self.views[1].clone(),
            sdm1: Some(self.views[2].clone()),
            sdm2: Some(self.views[3].clone()),
            // hdr < 0 = the mesh shader's Halo 4 sentinel; |hdr| = K_DIRECT (+0x14, lobe A),
            // k = K_INDIRECT (+0x18, lobe B); halo4.dll sub_18038C1E0
            hdr: -self.atlas.k_dom.max(1e-6),
            // k < 0 = this BSP's floating sun is disabled (Ravine vista): the shader returns sun
            // visibility 0 so no analytic sun is added over its (sun-baked) lobes
            k: if self.atlas.floating_sun { self.atlas.k.max(1e-6) } else { -self.atlas.k.max(1e-6) },
            // `// #h4-expo-2` mode = this BSP's static floating-shadow SHARPENING s (scnr
            // structure_bsps +184, `sub_18035E814`: `cb2[14] = (s + 1, (s + 1) / 2 - 0.5)` and
            // entry 06 `mad_sat vis = analytic.x * (s + 1) - ((s + 1) / 2 - 0.5)`). s = 1 gives the
            // familiar `sat(2 a - 0.5)`; s = 0 (20 of the 37 shipped MP BSPs, incl. Redoubt,
            // Valhalla, Forge Island bsp01, Skyline, Monolith, Shatter, Wreckage) passes the
            // analytic texel through UNSHARPENED - the shader used to hard-code s = 1 and lit
            // partially-shadowed texels up to 33 % too brightly. The Halo 4 lane is the only
            // consumer of `lm.w` (the Reach `lm.w > 1.5` object flag is unreachable here:
            // `mesh_shade` dispatches to `h4_shade` on `matmodel < -0.5` before reading it).
            mode: self.atlas.shadow_sharpen.clamp(0.0, 8.0),
        }
    }
}

/// Open a Halo 4 cache and parse every sbsp (no GPU work).
pub fn load_map(map_path: &str) -> Result<H4Loaded> { load_map_with_variant(map_path, None) }

/// `load_map` plus a Halo 4 map variant: its objects are resolved through the base
/// map's forge palette (mvar.rs) into `variant_placements`. A variant whose base map id is
/// not this cache's id (`maps/info/<map>.mapinfo`) is REFUSED (the quota indices would point
/// into the wrong palette); an unreadable .mapinfo only logs a warning.
pub fn load_map_with_variant(map_path: &str, variant: Option<&std::path::Path>) -> Result<H4Loaded> {
    let mut loaded = load_map_inner(map_path)?;
    if let Some(vp) = variant {
        let v = super::mvar::parse_h4_variant(vp)?;
        let name = vp.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match super::mvar::map_id_of_cache(std::path::Path::new(map_path)) {
            Some(id) if id != v.map_id => bail!("variant '{name}' targets map id {} but '{}' is map id {id}", v.map_id, loaded.cache.map_name),
            Some(_) => {}
            None => loaded.log.push(format!("h4 mvar: no .mapinfo for '{}' - base map id {} of '{name}' UNCHECKED", loaded.cache.map_name, v.map_id)),
        }
        let palette = super::mvar::load_forge_palette(&loaded.cache);
        let (placements, st) = super::mvar::variant_placements(&loaded.cache, &v, &palette);
        loaded.log.push(format!("h4 mvar '{name}': '{}' by {} - {} objects, {} placed ({} no quota, {} quota out of the {}-entry palette, {} variant out of range, {} null tag, {} no render model); budget {}/{} labels {:?}",
            v.title, v.author, st.objects, st.placed, st.skipped_no_quota, st.skipped_quota_range, palette.len(), st.skipped_variant_range, st.skipped_null_tag, st.skipped_no_model, v.budget_spent, v.budget_max, v.labels));
        if std::env::var("HMS_MVAR_LIST").is_ok() { eprint!("{}", super::mvar::describe(&v, Some(&palette))); }
        loaded.variant_placements = placements;
        loaded.variant = Some((name, v.title.clone(), st));
        loaded.variant_full = Some((vp.to_path_buf(), v));
    }
    // HMS_H4_PALETTE_DUMP=1 prints the Forge palette; HMS_H4_PLACE=<spec> places palette objects
    // for a headless look (both print-only / headless diagnostics; the GUI sets neither)
    let dump = std::env::var("HMS_H4_PALETTE_DUMP").map_or(false, |v| v != "0");
    let place = std::env::var("HMS_H4_PLACE").ok().filter(|s| !s.trim().is_empty());
    if dump || place.is_some() {
        let pal = super::palette::palette(&loaded.cache);
        if dump { eprint!("{}", pal.describe()); }
        if let Some(spec) = place {
            // origin: the variant's bounds centre when a variant is loaded, else the scenario's
            // sandbox origin point (scnr +0x80)
            let origin = match &loaded.variant_full {
                Some((_, v)) => [(v.bounds[0] + v.bounds[1]) * 0.5, (v.bounds[2] + v.bounds[3]) * 0.5, (v.bounds[4] + v.bounds[5]) * 0.5],
                None => sandbox_origin(&loaded.cache).unwrap_or([0.0; 3]),
            };
            let mut lines = Vec::new();
            loaded.forge = super::palette::place_spec(&loaded.cache, &pal, &spec, origin, &mut |l| lines.push(l));
            loaded.log.extend(lines);
        }
    }
    Ok(loaded)
}

/// scnr +0x80 "sandbox origin point" (Assembly Halo4MCC scnr.xml: "forge coordinates
/// are relative to this point"); None when the scenario is missing.
pub fn sandbox_origin(c: &H4Cache) -> Option<[f32; 3]> {
    let &scnr = c.find_tags(b"scnr").first()?;
    let sm = c.tag_meta(scnr)?;
    let d = c.data();
    let p = [d.f32_at(sm + 0x80), d.f32_at(sm + 0x84), d.f32_at(sm + 0x88)];
    p.iter().all(|v| v.is_finite()).then_some(p)
}

fn load_map_inner(map_path: &str) -> Result<H4Loaded> {
    let t0 = std::time::Instant::now();
    let cache = H4Cache::open(std::path::Path::new(map_path))?;
    let mut log = Vec::new();
    log.push(format!("h4: '{}' type {} scenario '{}' tags {} classes {} pages {} segments {} resources {} ({:.2}s)",
        cache.map_name, cache.map_type, cache.scenario, cache.tag_count(), cache.class_count(), cache.pages.len(), cache.segments.len(), cache.resources.len(), t0.elapsed().as_secs_f32()));
    let mut bsps: Vec<H4Bsp> = Vec::new();
    for t in cache.find_tags(b"sbsp") {
        match load_bsp(&cache, t) {
            Ok(b) => {
                let n_geo = b.meshes.iter().filter(|m| m.has_geometry()).count();
                log.push(format!("h4 bsp '{}': meshes {} (with geometry {}), vbs/ibs {}, clusters {:?}, instances {} (identity {}), materials {}, rot_check row/col/tested {:?} -> {}",
                    b.name, b.meshes.len(), n_geo,
                    b.geometry.as_ref().map(|g| format!("{}/{}", g.vbs.len(), g.ibs.len())).unwrap_or_else(|| "none".into()),
                    b.clusters, b.instances.len(), b.instances.iter().filter(|i| i.is_identity()).count(), b.materials.len(), b.rot_check, if b.rows { "rows" } else { "cols" }));
                bsps.push(b);
            }
            Err(e) => log.push(format!("h4 bsp {} failed: {e:#}", cache.tag_name(t))),
        }
    }
    if bsps.is_empty() { return Err(anyhow!("no BSP loaded")); }
    // materials: mats name + blend byte -> lane kind (materials.rs); HMS_H4_MATDIAG=1 prints them
    let matdiag = std::env::var("HMS_H4_MATDIAG").is_ok();
    let mut materials = Vec::with_capacity(bsps.len());
    for b in &bsps {
        let mats = load_bsp_materials(&cache, b.sbsp_tag);
        let mut hist: std::collections::BTreeMap<String, usize> = Default::default();
        for m in &mats { *hist.entry(format!("{:?}", m.kind)).or_default() += 1; }
        log.push(format!("h4 bsp '{}' material kinds {:?}", b.name, hist));
        if matdiag { eprintln!("HMS_H4_MATDIAG bsp '{}'", b.name); print_material_diag(&cache, &mats); }
        materials.push(mats);
    }
    let sun = bsps.iter().filter_map(|b| b.lbsp_tag).find_map(|lt| lbsp_sun(&cache, lt));
    match sun {
        Some((d, rgb)) => log.push(format!("h4 baked sun (Lbsp +0x24/+0x30): travel dir {:?} rgb {:?}", d, rgb)),
        None => log.push("h4 baked sun: none in any Lbsp - no analytic sun (the engine builds no sun record)".into()),
    }
    // `// #h4-expo-2` diag: the renderer carries ONE sun for the whole scene, but the sun is a
    // per-Lbsp field. `ca_forge_erosion` is the only shipped MP map whose two BSPs disagree
    // (bsp01 (7.97, 6.13, 3.40) vs bsp02 (1.06, 1.25, 1.25) = 7.5x), so bsp02's geometry is lit
    // with bsp01's sun. Print it rather than silently mis-light (a per-BSP sun needs a second
    // light-uniform slot indexed per mesh - not ported).
    {
        let suns: Vec<[f32; 3]> = bsps.iter().filter_map(|b| b.lbsp_tag).filter_map(|lt| lbsp_sun(&cache, lt)).map(|(_, rgb)| rgb).collect();
        if suns.windows(2).any(|w| w[0].iter().zip(w[1].iter()).any(|(a, b)| (a - b).abs() > 1e-3 * a.abs().max(1.0))) {
            log.push(format!("h4 baked sun: WARNING the map's BSPs carry DIFFERENT suns {suns:?} - the renderer uses the first for all of them"));
        }
    }
    let placements = load_placements(&cache);
    {
        let mut hist: std::collections::BTreeMap<String, usize> = Default::default();
        for p in &placements { *hist.entry(p.class_str()).or_default() += 1; }
        log.push(format!("h4 scnr placements {} {:?}", placements.len(), hist));
    }
    Ok(H4Loaded { cache: Arc::new(cache), bsps, materials, placements, variant_placements: Vec::new(), variant: None, variant_full: None, forge: Vec::new(), sun, log })
}

/// What `build_meshes` produced (counts + bounds for the camera framing).
#[derive(Clone, Debug, Default)]
pub struct H4Stats {
    pub draws: usize,
    pub tris: usize,
    pub verts: usize,
    pub decode_fail: usize,
    pub tex_ok: usize,
    pub tex_fail: Vec<String>,
    /// Bounds of every decoded, instanced vertex.
    pub bmin: [f32; 3],
    pub bmax: [f32; 3],
    /// Bounds of the PLAYABLE clusters (vista-sized clusters skipped); equal to bmin/bmax when none.
    pub pmin: [f32; 3],
    pub pmax: [f32; 3],
    pub lanes: HashMap<Lane, usize>,
    pub seconds: f32,
    /// Scenario objects: placements drawn, distinct models loaded, models that failed to load /
    /// decode (name + reason, first few), object draws.
    pub objects_placed: usize,
    /// Of `objects_placed`, the map-variant objects (drawn through the same path).
    pub variant_objects_placed: usize,
    /// Of `objects_placed`, the HMS-placed Forge objects (`H4Loaded::forge`).
    pub forge_objects_placed: usize,
    /// #h4-veh hlmt model-variant child objects drawn (turrets / guns).
    pub attachments_drawn: usize,
    pub models_ok: usize,
    pub models_fail: Vec<String>,
    pub object_draws: usize,
    /// Baked sun (light travel direction, rgb) copied from `H4Loaded::sun` for the callers.
    pub sun: Option<([f32; 3], [f32; 3])>,
    /// Parts skipped because their material is Invisible.
    pub skipped_invisible: usize,
    /// Lightmap path: BSP atlases uploaded, instances drawn with their atlas UVs, instances that
    /// had no UV stream (flat).
    pub lm_atlases: usize,
    pub lm_instances: usize,
    pub lm_flat_instances: usize,
    /// Instances lit through the per-vertex 5-slice array (CPU-evaluated vertex colours).
    pub lm_pervertex_instances: usize,
    /// Instances lit by an Lbsp probe (+ per-vertex AO) - CPU-evaluated vertex colours.
    pub lm_probe_instances: usize,
    /// The scenario's atmosphere fog packed for the renderer (lighting.rs), None when
    /// the map has no active fog layer (Haven).
    pub fog: Option<[f32; 28]>,
    /// The scenario's default camera fx (exposure band / bloom / filmic), lighting.rs.
    pub camera_fx: Option<super::lighting::H4CameraFx>,
    /// The cfxs colour-grading volume (BGRA8 texels, n) decoded for the post pass.
    pub color_grading_lut: Option<(Vec<u8>, u32)>,
    /// Scenario placements NOT drawn statically: objects whose tag is in the Forge palette (the
    /// map variant owns those, the same rule as Reach) and the spawn-family
    /// markers (`map_spawns::is_map_spawn_name`, diverted to the renderer's marker lane / hidden).
    pub scnr_forge_owned: usize,
    pub scnr_spawn_markers: usize,
    /// World AABB of every statically drawn object (scenario + HMS-placed Forge pieces),
    /// the renderer's shadow-map fit; None when nothing was placed.
    pub caster_bounds: Option<(Vec3, Vec3)>,
    /// The first playable BSP's floating-shadow cascade (scnr structure_bsps).
    pub cascade: Option<hms_render::H4CascadeCfg>,
}

impl H4Stats {
    /// Direction TO the sun for the renderer (`set_sun_dir`), from the baked travel direction.
    pub fn sun_dir(&self) -> Vec3 {
        match self.sun {
            Some((d, _)) => -Vec3::from(d).normalize_or_zero(),
            None => placeholder_sun_dir(),
        }
    }
    /// Sun colour for `set_scene_tints`: the Lbsp +0x30 rgb, ABSOLUTE when the atlas path is on
    /// (the engine adds sun_rgb . n / pi next to the baked lobes in the same units), else
    /// normalised to max 1 so the flat placeholder lighting keeps its white-sun brightness and
    /// only takes the hue.
    ///
    /// `// #h4-expo-2` No Lbsp carries a floating sun (flags bit 0x200 clear: Abandon, Perdition,
    /// Daybreak, Pitfall, Vertigo - `sub_18035F24C` builds no sun record at all) -> the engine
    /// adds NO analytic sun anywhere, so the engine lane must get BLACK, not the flat lane's white
    /// placeholder. The atlas lane already gated itself (`k < 0`) and so did the per-vertex /
    /// probe bakes, but the OBJECT lane (`object_lanes_aabb`, `object_sun_vis`) read this tint
    /// straight and lit every Forge piece with a fabricated `1.0 * sat(N.L) / pi`.
    pub fn sun_tint(&self) -> [f32; 3] {
        match self.sun {
            Some((_, rgb)) if self.lm_instances > 0 => rgb,
            Some((_, rgb)) => { let m = rgb[0].max(rgb[1]).max(rgb[2]).max(1e-6); [rgb[0] / m, rgb[1] / m, rgb[2] / m] }
            None if self.lm_instances > 0 => [0.0; 3],
            None => [1.0; 3],
        }
    }
}

impl H4Stats {
    /// The bounds a default camera should frame (playable clusters when known).
    pub fn frame_bounds(&self) -> (Vec3, Vec3) {
        let (p0, p1) = (Vec3::from(self.pmin), Vec3::from(self.pmax));
        if p0.x <= p1.x && std::env::var("HMS_H4_FRAME_ALL").is_err() { (p0, p1) } else { (Vec3::from(self.bmin), Vec3::from(self.bmax)) }
    }
}

/// Decode + upload every cluster / instance mesh of every BSP. `emit` receives finished batches
/// per lane (called on the build thread); `progress(done, total)` counts mesh uses; `cancel`
/// aborts between meshes (the partial result is still returned).
pub fn build_meshes(
    loaded: &H4Loaded,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mr: &MeshRenderer,
    emit: &mut dyn FnMut(Lane, Vec<GpuMesh>),
    progress: &mut dyn FnMut(u32, u32),
    cancel: Option<&AtomicBool>,
    log: &mut dyn FnMut(String),
    mut editor: Option<&mut H4EditorCollect>,
) -> Result<H4Stats> {
    let t0 = std::time::Instant::now();
    let cache: &H4Cache = &loaded.cache;
    let max_mesh: usize = std::env::var("HMS_H4_MAXMESH").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let mut stats = H4Stats { bmin: [f32::MAX; 3], bmax: [f32::MIN; 3], pmin: [f32::MAX; 3], pmax: [f32::MIN; 3], sun: loaded.sun, ..Default::default() };
    // The floating-shadow cascade of the first BSP that carries the baked sun (the
    // playable BSP; vistas author a disabled sun) - see lightmaps::scnr_cascade.
    stats.cascade = loaded.bsps.iter().filter_map(|b| b.lbsp_tag).find(|&lt| lbsp_sun(&loaded.cache, lt).is_some()).and_then(|lt| super::lightmaps::scnr_cascade(&loaded.cache, lt));
    if let Some(c) = &stats.cascade {
        log(format!("h4 shadows: floating-shadow cascade half_width {} length {} offset {} bias {} filter {} sun_off {} res {} taps {}", c.half_width, c.length, c.offset, c.bias, c.filter, c.sun_offset, c.resolution, c.taps));
    }

    // --- textures (one per bitmap, shared across BSPs) + the material resolver; the context
    //     outlives this run when an editor scene is requested (`H4EditorCollect`) ---
    let mut tc = TexCtx::new(loaded.cache.clone(), device, queue);
    // the shipped-shader material lane (HMS_H4_SHADING=0 -> the flat diffuse lane)
    let shading_on = shading_requested();

    // --- lightmap atlases (default on, HMS_H4_LIGHTMAP=0 off): one composed 4-texture set per BSP ---
    let lm_on = lightmap_requested();
    let lm_diag = std::env::var("HMS_H4_LMDIAG").is_ok();
    let mut atlases: Vec<Option<BspAtlasGpu>> = Vec::with_capacity(loaded.bsps.len());
    for bsp in &loaded.bsps {
        let mut got = None;
        if let (true, Some(lt)) = (lm_on || lm_diag, bsp.lbsp_tag) {
            match load_atlas(cache, lt).and_then(|a| compose_atlas_textures(cache, &a).map(|t| (a, t))) {
                Ok((atlas, tex)) => {
                    if lm_diag { engine_stats(cache, bsp, &atlas, &tex); }
                    if lm_on {
                        let up = |t: &(Vec<u8>, u32, u32)| hms_render::upload_texture_bgra_nomip(device, queue, &t.0, t.1, t.2).0;
                        let views = [up(&tex[0]), up(&tex[1]), up(&tex[2]), up(&tex[3])];
                        let pervertex = match load_pervertex(cache, lt) {
                            Ok(p) => p,
                            Err(e) => { log(format!("h4 lightmap '{}': per-vertex array: {e}", bsp.name)); None }
                        };
                        log(format!("h4 lightmap '{}': atlas {}x{} bound (K_direct {:.3} K_indirect {:.3}, sun {:?}, shadow sharpen {:.2}, sharpen LUT {})", bsp.name, atlas.w, atlas.h, atlas.k_dom, atlas.k, atlas.sun_rgb, atlas.shadow_sharpen, if sharpen_lut(cache).is_some() { "engine" } else { "identity fallback" }));
                        stats.lm_atlases += 1;
                        let probes = load_probes(cache, lt);
                        let vertex_ao = match load_vertex_ao(cache, lt) { Ok(p) => p, Err(e) => { log(format!("h4 lightmap '{}': vertex ao array: {e}", bsp.name)); None } };
                        if !probes.is_empty() { log(format!("h4 lightmap '{}': {} instance probes, vertex ao array {}", bsp.name, probes.len(), if vertex_ao.is_some() { "bound" } else { "none" })); }
                        let airprobes = load_airprobes(cache, lt);
                        if !airprobes.is_empty() { log(format!("h4 lightmap '{}': {} airprobes (Lbsp +0x3E8; object lighting fallback)", bsp.name, airprobes.len())); }
                        got = Some(BspAtlasGpu { atlas, views, pervertex, probes, vertex_ao, cpu: Arc::new(tex), airprobes });
                    }
                }
                Err(e) => log(format!("h4 lightmap '{}': no per-pixel atlas ({e})", bsp.name)),
            }
        }
        atlases.push(got);
    }

    // mesh uses per BSP: mesh index -> instance matrices (clusters drawn once at identity);
    // with the atlas path on, every lightmapped instance is its own entry (its UV stream is
    // per instance, so it cannot share a vertex buffer with the other uses of the mesh).
    let mut plans: Vec<(usize, Vec<(usize, Vec<Mat4>, Option<usize>)>)> = Vec::new();
    let mut total = 0u32;
    for (bi, bsp) in loaded.bsps.iter().enumerate() {
        let mut uses: HashMap<usize, Vec<Mat4>> = HashMap::new();
        let mut singles: Vec<(usize, Vec<Mat4>, Option<usize>)> = Vec::new();
        for &ci in &bsp.clusters { uses.entry(ci as usize).or_default().push(Mat4::IDENTITY); }
        for (ii, inst) in bsp.instances.iter().enumerate() {
            if inst.mesh < 0 { continue; }
            // An instance is its own draw when it has an atlas UV stream OR a per-vertex
            // lighting record in the 5-slice array (bitmap select 0; the vertex_ao path stays flat)
            let lit = atlases[bi].is_some() && bsp.lbsp_tag.map_or(false, |lt| {
                instance_uv_vb(cache, lt, ii).is_some()
                    || (atlases[bi].as_ref().map_or(false, |a| a.pervertex.is_some()) && instance_pervertex(cache, lt, ii).map_or(false, |(b, _, _)| b == 0))
                    // probe-lit instances (mode 0x100, or 0x200 / 0x300 with the AO array)
                    || (atlases[bi].as_ref().map_or(false, |a| !a.probes.is_empty()) && instance_probe(cache, lt, ii).is_some())
            });
            if lit {
                singles.push((inst.mesh as usize, vec![bsp.instance_matrix(inst)], Some(ii)));
            } else {
                if atlases[bi].is_some() { stats.lm_flat_instances += 1; }
                uses.entry(inst.mesh as usize).or_default().push(bsp.instance_matrix(inst));
            }
        }
        let mut order: Vec<usize> = uses.keys().copied().collect();
        order.sort();
        let mut plan: Vec<(usize, Vec<Mat4>, Option<usize>)> = order.into_iter().take(max_mesh).map(|mi| (mi, uses.remove(&mi).unwrap(), None)).collect();
        plan.extend(singles.into_iter().take(max_mesh));
        total += plan.len() as u32;
        plans.push((bi, plan));
        for (mn, mx) in &bsp.cluster_bounds {
            let (mn, mx) = (Vec3::from(*mn), Vec3::from(*mx));
            if (mx - mn).length() < 150.0 {
                for k in 0..3 { stats.pmin[k] = stats.pmin[k].min(mn[k]); stats.pmax[k] = stats.pmax[k].max(mx[k]); }
            }
        }
    }

    // Object lighting stand-ins for maps without airprobes (`ObjectLightField::object_lanes_aabb`
    // steps 3): the Lbsp instance probes with their instance positions (record @+24) and the
    // mean per-vertex baked colour
    let mut probe_field: Vec<([f32; 3], usize, usize)> = Vec::new(); // (pos, bsp index, probe index)
    for (bi, bsp) in loaded.bsps.iter().enumerate() {
        if let (Some(ag), Some(lt)) = (atlases[bi].as_ref(), bsp.lbsp_tag) {
            if ag.probes.is_empty() { continue; }
            if let Some((n, o)) = cache.tag_meta(lt).and_then(|m| cache.block(m + super::lightmaps::OFF_LBSP_INSTANCE_INFO)) {
                let d = cache.data();
                for ii in 0..n {
                    let e = o + ii * super::lightmaps::INSTANCE_INFO_ELEM;
                    let pi = d.i16_at(e + 4);
                    if pi < 0 || pi as usize >= ag.probes.len() { continue; }
                    probe_field.push(([d.f32_at(e + 24), d.f32_at(e + 28), d.f32_at(e + 32)], bi, pi as usize));
                }
            }
        }
    }
    let mut bake_sum = [0.0f64; 4];
    let mut bake_n = 0usize;
    let mut tri_uv2: Vec<[[f32; 2]; 3]> = Vec::new(); // soup triangle -> atlas uv2 corners
    let mut done = 0u32;
    let mut pending: HashMap<Lane, Vec<GpuMesh>> = HashMap::new();
    let flush = |pending: &mut HashMap<Lane, Vec<GpuMesh>>, emit: &mut dyn FnMut(Lane, Vec<GpuMesh>)| {
        for (lane, v) in pending.drain() { if !v.is_empty() { emit(lane, v); } }
    };
    'outer: for (bi, plan) in &plans {
        let bsp = &loaded.bsps[*bi];
        for (mi, mats, lm_inst) in plan {
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) { break 'outer; }
            done += 1;
            progress(done, total);
            let dm = match decode_mesh(cache, bsp, *mi) {
                Ok(Some(d)) => d,
                Ok(None) => continue,
                Err(e) => { stats.decode_fail += 1; if stats.decode_fail <= 5 { log(format!("h4 mesh {mi} decode failed: {e}")); } continue; }
            };
            // The renderer's stored tangent frame is WORLD space (Reach pre-rotates it per instance);
            // ours is mesh-local, so it is only handed over for identity placements - rotated
            // instances keep the shader's derivative fallback ([0;4]).
            let local_is_world = mats.iter().all(|m| *m == Mat4::IDENTITY);
            let mut verts: Vec<MeshVertex> = dm.verts.iter().map(|v| MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: [1.0; 4], tangent: if local_is_world { v.tangent } else { [0.0; 4] }, ..Default::default() }).collect();
            // atlas: this instance's own u16x2 atlas-space UV stream -> uv2
            let mut lm: Option<&BspAtlasGpu> = None;
            if let (Some(ii), Some(ag)) = (lm_inst, atlases[*bi].as_ref()) {
                match mesh_lightmap_uvs(cache, bsp, *mi, *ii) {
                    Ok(Some(uvs)) if uvs.len() == verts.len() => {
                        for (v, uv) in verts.iter_mut().zip(uvs) { v.uv2 = uv; }
                        lm = Some(ag);
                        stats.lm_instances += 1;
                    }
                    Ok(Some(uvs)) => { stats.lm_flat_instances += 1; if stats.lm_flat_instances <= 5 { log(format!("h4 lightmap: instance {ii} mesh {mi}: {} uvs for {} verts - flat", uvs.len(), verts.len())); } }
                    Ok(None) => {
                        // per-vertex lit: engine-law colour per vertex (lightmaps.rs
                        // `pervertex_colors`), world-space normals, alpha = sun visibility
                        let mut done = false;
                        if let (Some(pv), Some((0, off, _))) = (ag.pervertex.as_ref(), bsp.lbsp_tag.and_then(|lt| instance_pervertex(cache, lt, *ii))) {
                            let m0 = mats.first().copied().unwrap_or(Mat4::IDENTITY);
                            let normals: Vec<[f32; 3]> = dm.verts.iter().map(|v| m0.transform_vector3(Vec3::from(v.normal)).normalize_or_zero().to_array()).collect();
                            if let Some(cols) = pervertex_colors(pv, &ag.atlas, off, &normals) {
                                for c in &cols { for k in 0..4 { bake_sum[k] += c[k] as f64; } bake_n += 1; }
                                for (v, c) in verts.iter_mut().zip(cols) { v.color = c; }
                                stats.lm_pervertex_instances += 1;
                                done = true;
                            }
                        }
                        // probe-lit (Lbsp probe + optional per-vertex AO): probe irradiance at the vertex normal
                        if !done {
                            if let Some(pi) = bsp.lbsp_tag.and_then(|lt| instance_probe(cache, lt, *ii)) {
                                if let Some(probe) = ag.probes.get(pi) {
                                    let m0 = mats.first().copied().unwrap_or(Mat4::IDENTITY);
                                    let normals: Vec<[f32; 3]> = dm.verts.iter().map(|v| m0.transform_vector3(Vec3::from(v.normal)).normalize_or_zero().to_array()).collect();
                                    let ao = match (ag.vertex_ao.as_ref(), bsp.lbsp_tag.and_then(|lt| instance_pervertex(cache, lt, *ii))) {
                                        (Some(pv), Some((1, off, _))) => Some((pv, off)),
                                        _ => None,
                                    };
                                    if let Some(cols) = probe_vertex_colors(probe, ao, ag.atlas.floating_sun, &normals) {
                                        for c in &cols { for k in 0..4 { bake_sum[k] += c[k] as f64; } bake_n += 1; }
                                        for (v, c) in verts.iter_mut().zip(cols) { v.color = c; }
                                        stats.lm_probe_instances += 1;
                                        done = true;
                                    }
                                }
                            }
                        }
                        if !done { stats.lm_flat_instances += 1; }
                    }
                    Err(e) => { stats.lm_flat_instances += 1; if stats.lm_flat_instances <= 5 { log(format!("h4 lightmap: instance {ii} mesh {mi}: {e} - flat")); } }
                }
            }
            for m in mats {
                for v in &dm.verts {
                    let p = m.transform_point3(Vec3::from(v.pos));
                    for k in 0..3 { stats.bmin[k] = stats.bmin[k].min(p[k]); stats.bmax[k] = stats.bmax[k].max(p[k]); }
                }
            }
            // raycast soup: every DRAWN triangle of every instance in world space (the
            // editor's floor / wall / placement surface; the Reach `bsp_soup` analogue)
            if let Some(ed) = editor.as_deref_mut() {
                for part in &dm.parts {
                    let mi_mat = part.material.max(0) as usize;
                    let kind = loaded.materials.get(*bi).and_then(|v| v.get(mi_mat)).map(|m| m.kind).unwrap_or(H4MatKind::Opaque);
                    if lane_for(kind).is_none() { continue; }
                    let s = part.index_start as usize;
                    let e = (s + part.index_count as usize).min(dm.indices.len());
                    for m in mats {
                        for tri in dm.indices[s.min(e)..e].chunks_exact(3) {
                            let (Some(a), Some(b), Some(c)) = (dm.verts.get(tri[0] as usize), dm.verts.get(tri[1] as usize), dm.verts.get(tri[2] as usize)) else { continue };
                            // lightmapped (atlas uv2) triangles carry (bsp + 1, uv2 table
                            // index) in the soup's material word: the object SURFACE PROBE (engine
                            // `sub_1802E4388`) raycasts down onto them and reads the lightmap texel
                            let mat = if lm.is_some() { (*bi as u32 + 1, tri_uv2.len() as u32, -1) } else { (0, 0, -1) };
                            let before = ed.soup.len();
                            ed.soup.add_tri_owned(m.transform_point3(Vec3::from(a.pos)), m.transform_point3(Vec3::from(b.pos)), m.transform_point3(Vec3::from(c.pos)), mat, u32::MAX);
                            if lm.is_some() && ed.soup.len() > before {
                                let uv = |k: u32| verts.get(k as usize).map(|v| v.uv2).unwrap_or([0.0, 0.0]);
                                tri_uv2.push([uv(tri[0]), uv(tri[1]), uv(tri[2])]);
                            }
                        }
                    }
                }
            }
            stats.verts += verts.len() * mats.len();
            for part in &dm.parts {
                let s = part.index_start as usize;
                let e = (s + part.index_count as usize).min(dm.indices.len());
                if e <= s { continue; }
                let indices = &dm.indices[s..e];
                let mi_mat = part.material.max(0) as usize;
                let material = bsp.materials.get(mi_mat);
                let classified = loaded.materials.get(*bi).and_then(|v| v.get(mi_mat));
                let kind = classified.map(|m| m.kind).unwrap_or(H4MatKind::Opaque);
                let Some(lane) = lane_for(kind) else { stats.skipped_invisible += 1; continue };
                let base_alpha = classified.map(|m| m.base_alpha).unwrap_or(1.0);
                let tex = classified.and_then(|m| m.diffuse()).or_else(|| material.and_then(|m| m.diffuse_bitm)).and_then(|t| tc.fetch_tex(t, &mut stats).0);
                // vertmask shaders: the per-vertex blend scalar is the sheet's opacity mask
                // (vertex colour alpha x instance alpha = the blend lane's albedo alpha)
                let masked: Vec<MeshVertex>;
                // opaque / cutout parts take the shipped-shader lane; the blend / additive lanes
                // keep the flat placeholder (their transparent shaders are not decoded).
                let h4_lane = shading_on && matches!(kind, H4MatKind::Opaque | H4MatKind::AlphaTest) && classified.is_some();
                let part_verts: &[MeshVertex] = if classified.map_or(false, |m| m.uses_vertex_mask()) {
                    masked = verts.iter().zip(&dm.verts).map(|(v, s)| MeshVertex { color: [1.0, 1.0, 1.0, s.blend.clamp(0.0, 1.0)], ..*v }).collect();
                    &masked
                } else if h4_lane && lm.is_some() {
                    // atlas-lit: the vertex colour is free, its alpha carries the per-vertex blend
                    // scalar (the layered terrain shader's v2.w)
                    masked = verts.iter().zip(&dm.verts).map(|(v, s)| MeshVertex { color: [1.0, 1.0, 1.0, s.blend.clamp(0.0, 1.0)], ..*v }).collect();
                    &masked
                } else { &verts };
                // `srf_ca_boundary*` (the boundary walls): its own ADDITIVE self-illum shader,
                // drawn through the renderer's halogram lane (`upload_part_boundary`).
                let bnd = match (classified, shading_on) { (Some(m), true) => tc.shading_of(m).boundary, _ => None };
                let gm = if let Some(b) = bnd {
                    upload_part_boundary(&mut tc, mr, part_verts, indices, mats, &b, &mut stats)
                } else if h4_lane {
                    let (sh, ht) = tc.resolve_h4(classified.unwrap(), &mut stats);
                    upload_part_h4(mr, device, queue, part_verts, indices, mats, &ht, &sh, kind, base_alpha, NEUTRAL_CC, lm.map(|ag| ag.inputs()))
                } else {
                    upload_part_tinted(mr, device, queue, part_verts, indices, mats, tex.as_ref(), kind, base_alpha, lm.map(|ag| ag.inputs()), classified.and_then(emissive_tint))
                };
                pending.entry(lane).or_default().push(gm);
                *stats.lanes.entry(lane).or_default() += 1;
                stats.tris += indices.len() / 3 * mats.len();
                stats.draws += 1;
            }
            if pending.values().map(|v| v.len()).sum::<usize>() >= 48 { flush(&mut pending, emit); }
        }
    }
    flush(&mut pending, emit);
    if stats.bmin[0] > stats.bmax[0] { return Err(anyhow!("no geometry decoded")); }
    if lm_on {
        log(format!("h4 lightmap: {} atlases, {} instances drawn with atlas UVs, {} per-vertex lit, {} probe lit, {} instances flat", stats.lm_atlases, stats.lm_instances, stats.lm_pervertex_instances, stats.lm_probe_instances, stats.lm_flat_instances));
    }

    // --- the sky object (drawn camera-relative by the renderer's sky lane; HMS_H4_NOSKY=1 skips it) ---
    if std::env::var("HMS_H4_NOSKY").is_err() {
        let sky = build_sky(loaded, device, queue, mr, &mut tc, &mut stats, log);
        if !sky.is_empty() { emit(Lane::Sky, sky); }
    }

    // --- scenario objects: one draw per (model, part) with every placement's matrix ---
    // mean baked colour of the per-vertex / probe-lit BSP vertices: the fallback object ambient
    let mean_bake: [f32; 4] = if bake_n > 0 { [(bake_sum[0] / bake_n as f64) as f32, (bake_sum[1] / bake_n as f64) as f32, (bake_sum[2] / bake_n as f64) as f32, (bake_sum[3] / bake_n as f64) as f32] } else { [1.0, 1.0, 1.0, 0.0] };
    let light = ObjectLightField {
        probe_field,
        probes: atlases.iter().map(|a| a.as_ref().map(|a| a.probes.clone()).unwrap_or_default()).collect(),
        mean_bake,
        // the engine's object lighting sources: the lightmap texel under the object (surface
        // probe) and the Lbsp airprobes (fallback), see `ObjectLightField::object_lanes_aabb`
        atlas_cpu: atlases.iter().map(|a| a.as_ref().map(|a| (a.cpu.clone(), a.atlas.k_dom, a.atlas.k))).collect(),
        airprobes: atlases.iter().flat_map(|a| a.as_ref().map(|a| a.airprobes.clone()).unwrap_or_default()).collect(),
        tri_uv2: std::mem::take(&mut tri_uv2),
        stats: Default::default(),
    };
    // The BSP soup is complete here (every drawn BSP triangle): balance it now so the
    // object sun-visibility raycasts (and the editor later) run on the final grid
    if let Some(ed) = editor.as_deref_mut() { ed.soup.rebalance(8); ed.soup.shrink(); }
    let sun_to = stats.sun_dir();
    let soup: Option<&crate::decal_projector::TriSoup> = editor.as_deref().map(|e| &e.soup);
    let mut marker_meshes_out: Vec<(Lane, GpuMesh)> = Vec::new();
    if std::env::var("HMS_H4_NOOBJ").is_err() && !cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
        // #h4-veh keyed by (render model, quantised change colour): the change colour is a
        // per-MESH instance lane, so placements may only share a draw when it matches.
        let mut by_mode: HashMap<(usize, [i32; 4]), (Vec<Mat4>, [f32; 4])> = HashMap::new();
        let mut order: Vec<(usize, [i32; 4])> = Vec::new();
        let mut obj_mats: HashMap<usize, H4Material> = HashMap::new();
        // #h4-veh every drawn parent as (obje tag, hlmt variant, world matrix, mp properties),
        // expanded into its hlmt-variant CHILD objects (turrets / guns) after both loops.
        let mut parents: Vec<(usize, Option<usize>, Mat4, Option<(i8, Option<u8>)>)> = Vec::new();
        // The variant's Forge objects follow the scenario placements through the same palette
        // tag -> hlmt -> mode path (their basis comes from the variant, see objects.rs) - unless
        // an editor scene is requested: then the variant objects are the editor's (dynamic lanes
        // + picks, h4/edit_scene.rs) and only the scenario stays static.
        let n_scnr = loaded.placements.len();
        let variant_static: &[H4Placement] = if editor.is_some() { &[] } else { &loaded.variant_placements };
        // The Reach scenario rules: (1) a scenario placement whose obje tag is in the map's Forge
        // palette is canvas / default-variant content the map VARIANT owns (the engine's
        // sub_1800B5C90 matches exactly these into a new variant), so it is never drawn from the
        // scnr (HMS_SCNR_FORGE=1 restores it, a diagnostic); (2) the remaining spawn-family
        // markers (initial / respawn points, respawn zones, the cinematic camera) go to the
        // marker lane (editor scene) or are dropped unless HMS_MAP_SPAWNS=1 (headless).
        // Placement flags are 0 on every Forge map, so the tag path is the only signal. The
        // spawn camera (spawncam.rs) still reads the hidden placements.
        let forge_owned: std::collections::HashSet<usize> = if std::env::var("HMS_SCNR_FORGE").is_ok() { Default::default() } else {
            super::mvar::load_forge_palette(cache).iter().flat_map(|e| e.variants.iter().filter_map(|v| v.tag)).collect()
        };
        let show_env_spawns = crate::map_spawns::env_show_map_spawns();
        let mut marker_by_mode: HashMap<usize, (Vec<Mat4>, Vec<[f32; 4]>)> = HashMap::new();
        let mut marker_order: Vec<usize> = Vec::new();
        for (i, p) in loaded.placements.iter().chain(variant_static.iter()).enumerate() {
            if !OBJECT_CLASSES.iter().any(|c| **c == p.class) { continue; }
            if i < n_scnr && forge_owned.contains(&p.palette_tag) { stats.scnr_forge_owned += 1; continue; }
            let Some(mode) = model_of(cache, p.palette_tag) else { continue };
            if i < n_scnr && crate::map_spawns::is_map_spawn_name(cache.tag_name(p.palette_tag)) {
                stats.scnr_spawn_markers += 1;
                if editor.is_some() || show_env_spawns {
                    let e = marker_by_mode.entry(mode).or_insert_with(|| { marker_order.push(mode); Default::default() });
                    e.0.push(placement_matrix(p));
                    e.1.push(light.light(p.pos));
                }
                continue;
            }
            let vsid = super::objects::default_variant_sid(cache, p.palette_tag);
            let cc = super::tint::primary_change_color(cache, p.palette_tag, vsid, p.mp);
            let key = (mode, super::tint::color_key(cc));
            let v = by_mode.entry(key).or_insert_with(|| { order.push(key); (Vec::new(), cc) });
            let m = placement_matrix(p);
            v.0.push(m);
            let pv = p.model_variant.or_else(|| super::objects::object_variant_index(cache, p.palette_tag, 0));
            parents.push((p.palette_tag, pv, m, p.mp));
            stats.objects_placed += 1;
            if i >= n_scnr { stats.variant_objects_placed += 1; }
        }
        let total_obj = order.len() as u32;
        for (k, key) in order.iter().enumerate() {
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) { break; }
            progress(total + k as u32 + 1, total + total_obj);
            let mode = &key.0;
            let (mats, cc) = &by_mode[key];
            let geom = match decode_model_geom(cache, *mode) {
                Ok(g) => g,
                Err(e) => { stats.models_fail.push(format!("{}: {e}", cache.tag_name(*mode))); continue; }
            };
            for f in &geom.mesh_fail { stats.models_fail.push(f.clone()); }
            // per-placement lighting lanes (probe SH ambient + lobes, raycast sun visibility)
            let (cols, lanes) = object_light_lanes(&light, soup, sun_to, geom.min, geom.max, mats);
            let any = upload_model_parts(&mut tc, mr, &geom, mats, &cols, &lanes, *cc, &mut obj_mats, shading_on, &mut stats, &mut |lane, gm| pending.entry(lane).or_default().push(gm));
            if any { stats.models_ok += 1; } else { stats.models_fail.push(format!("{}: no drawable lod0 mesh", geom.name)); }
            if any { for m in mats { grow_caster_bounds(&mut stats.caster_bounds, geom.min, geom.max, m); } }
            if pending.values().map(|v| v.len()).sum::<usize>() >= 48 { flush(&mut pending, emit); }
        }
        // The spawn-family markers: same upload; into the editor's marker list (GUI,
        // renderer marker lane behind the View toggle) or straight into the lanes (headless with
        // HMS_MAP_SPAWNS=1). The lane kind is kept for the headless emit only.
        if !marker_order.is_empty() {
            let mut marker_meshes: Vec<(Lane, GpuMesh)> = Vec::new();
            for mode in &marker_order {
                let (mats, cols) = &marker_by_mode[mode];
                let geom = match decode_model_geom(cache, *mode) {
                    Ok(g) => g,
                    Err(e) => { stats.models_fail.push(format!("{}: {e}", cache.tag_name(*mode))); continue; }
                };
                let is_editor = editor.is_some();
                let _ = upload_model_parts(&mut tc, mr, &geom, mats, cols, &[], NEUTRAL_CC, &mut obj_mats, shading_on, &mut stats, &mut |lane, gm| {
                    if is_editor { marker_meshes.push((lane, gm)); } else { pending.entry(lane).or_default().push(gm); }
                });
            }
            marker_meshes_out = marker_meshes; // handed to the editor collect below (the soup borrow ends there)
        }
        log(format!("h4 scnr placements: {} forge-palette placements skipped (the map variant owns those), {} spawn-family markers {}", stats.scnr_forge_owned, stats.scnr_spawn_markers,
            if editor.is_some() { "uploaded to the marker lane (View > Show map spawn points)" } else if show_env_spawns { "drawn (HMS_MAP_SPAWNS=1)" } else { "hidden (HMS_MAP_SPAWNS=1 shows them)" }));
        // HMS-placed Forge objects: same upload path, keyed by (mode, hlmt variant) so
        // each shows its own permutations (objects.rs `variant_meshes`)
        if !loaded.forge.is_empty() {
            let mut by_key: HashMap<(usize, Option<usize>, [i32; 4]), (Vec<Mat4>, [f32; 4], usize)> = HashMap::new();
            let mut korder: Vec<(usize, Option<usize>, [i32; 4])> = Vec::new();
            for f in &loaded.forge {
                let Some(mode) = f.mode else { continue };
                let p = f.placement();
                let cc = super::tint::primary_change_color(cache, f.tag, f.variant_sid, p.mp);
                let key = (mode, f.model_variant, super::tint::color_key(cc));
                let e = by_key.entry(key).or_insert_with(|| { korder.push(key); (Vec::new(), cc, f.tag) });
                let m = placement_matrix(&p);
                e.0.push(m);
                parents.push((f.tag, f.model_variant, m, p.mp));
                stats.objects_placed += 1;
                stats.forge_objects_placed += 1;
            }
            for key in &korder {
                let (mats, cc, obje) = &by_key[key];
                let geom = match decode_object_geom(cache, *obje, key.0, key.1) {
                    Ok(g) => g,
                    Err(e) => { stats.models_fail.push(format!("{}: {e}", cache.tag_name(key.0))); continue; }
                };
                for f in &geom.mesh_fail { stats.models_fail.push(f.clone()); }
                let (cols, lanes) = object_light_lanes(&light, soup, sun_to, geom.min, geom.max, mats);
                let any = upload_model_parts(&mut tc, mr, &geom, mats, &cols, &lanes, *cc, &mut obj_mats, shading_on, &mut stats, &mut |lane, gm| pending.entry(lane).or_default().push(gm));
                if any { stats.models_ok += 1; } else { stats.models_fail.push(format!("{}: no drawable mesh in variant {:?}", geom.name, key.1)); }
                if any { for m in mats { grow_caster_bounds(&mut stats.caster_bounds, geom.min, geom.max, m); } }
            }
            log(format!("h4 forge: {} HMS-placed objects drawn as {} (model, variant, colour) sets", stats.forge_objects_placed, korder.len()));
        }
        // #h4-veh hlmt model-variant CHILD objects (the engine's `object_create_children` ->
        // `object_attach_to_marker`): a rocket Warthog's turret, the Wraith's mortar, the Scorpion's
        // cannon. Each child is decoded with its OWN object tag so it keeps its materials, its own
        // hlmt variant (the variant's `child variant name`) and its own change colours, and is posed
        // parent_marker_frame x child_marker_frame^-1 in the parent's world frame (h4/attach.rs).
        {
            let mut att: HashMap<(usize, usize, Option<usize>, [i32; 4]), (Vec<Mat4>, [f32; 4])> = HashMap::new();
            let mut aorder: Vec<(usize, usize, Option<usize>, [i32; 4])> = Vec::new();
            for (obje, vi, world, mp) in &parents {
                for a in super::attach::expand_attachments(cache, *obje, *vi, *world) {
                    let vsid = super::objects::default_variant_sid(cache, a.obje);
                    let cc = super::tint::primary_change_color(cache, a.obje, vsid, *mp);
                    let key = (a.obje, a.mode, a.variant, super::tint::color_key(cc));
                    att.entry(key).or_insert_with(|| { aorder.push(key); (Vec::new(), cc) }).0.push(a.world);
                }
            }
            for key in &aorder {
                if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) { break; }
                let (mats, cc) = &att[key];
                let geom = match decode_object_geom(cache, key.0, key.1, key.2) {
                    Ok(g) => g,
                    Err(e) => { stats.models_fail.push(format!("{}: {e}", cache.tag_name(key.1))); continue; }
                };
                for f in &geom.mesh_fail { stats.models_fail.push(f.clone()); }
                let (cols, lanes) = object_light_lanes(&light, soup, sun_to, geom.min, geom.max, mats);
                let any = upload_model_parts(&mut tc, mr, &geom, mats, &cols, &lanes, *cc, &mut obj_mats, shading_on, &mut stats, &mut |lane, gm| pending.entry(lane).or_default().push(gm));
                if any { stats.attachments_drawn += mats.len(); for m in mats { grow_caster_bounds(&mut stats.caster_bounds, geom.min, geom.max, m); } }
                if pending.values().map(|v| v.len()).sum::<usize>() >= 48 { flush(&mut pending, emit); }
            }
            if !aorder.is_empty() { log(format!("h4 attachments: {} hlmt variant child objects drawn as {} (object, model, variant, colour) sets", stats.attachments_drawn, aorder.len())); }
        }
        flush(&mut pending, emit);
        log(format!("h4 objects: {} placements drawn as {} models ({} failed), {} draws; object light = surface probe ({} lightmapped soup tris) / {} airprobes / {} instance probes / mean bake {:?}: {:?}", stats.objects_placed, stats.models_ok, stats.models_fail.len(), stats.object_draws, light.tri_uv2.len(), light.airprobes.len(), light.probe_field.len(), mean_bake, light.stats.lock().map(|s| *s).unwrap_or_default()));
        if let Some((name, title, st)) = &loaded.variant {
            if editor.is_some() {
                log(format!("h4 mvar '{name}' ('{title}'): {} variant objects handed to the editor scene ({} resolved to a render model)", st.objects, st.placed));
            } else {
                log(format!("h4 mvar '{name}' ('{title}'): {} of {} variant objects drawn ({} resolved to a render model, {} skipped)", stats.variant_objects_placed, st.objects, st.placed, st.objects - st.placed));
            }
        }
        for f in stats.models_fail.iter().take(6) { log(format!("  model: {f}")); }
    }
    // hand the editor its raycast soup + object lighting + texture context
    if let Some(ed) = editor.as_deref_mut() {
        ed.map_spawn_meshes = marker_meshes_out;
        ed.light = light;
        ed.tex = Some(tc);
        log(format!("h4 editor: raycast soup {} tris ({:.1} MB), {} instance probes", ed.soup.len(), ed.soup.mem_bytes() as f64 / 1e6, ed.light.probe_field.len()));
    }
    // atmosphere fog (fogg) -> the fog uniform of the Halo 4 lane
    if let Some(f) = super::lighting::load_fog(cache) {
        let sun_dir = stats.sun_dir().to_array();
        let sun_rgb = stats.sun.map(|(_, rgb)| rgb).unwrap_or([0.0; 3]);
        log(format!("h4 fog '{}': flags {:#x} ground base {} height {} thickness {} falloff {} colour {:?}{} -> {}", f.name.rsplit('\\').next().unwrap_or(&f.name), f.flags, f.ground.base, f.ground.height, f.ground.thickness, f.ground.falloff_end, f.ground.color,
            if f.flags & super::lighting::FOGG_FLAG_LIGHT != 0 { format!(", fog light {}x{} (tint {:?}, pitch {} yaw {} radius {} ang falloff {} dist falloff {} nearby cutoff {}), dist bias {}", f.light_intensity, if f.flags & super::lighting::FOGG_FLAG_LIGHT_IS_SUN != 0 { "sun" } else { "tint" }, f.light_tint, f.light_pitch, f.light_yaw, f.light_radius_deg, f.light_ang_falloff, f.light_dist_falloff, f.light_nearby_cutoff, f.dist_bias) } else { String::new() },
            if f.active() { "active" } else { "inactive (thickness 0)" }));
        if f.active() { stats.fog = Some(f.uniform(sun_dir, sun_rgb)); }
    }
    if let Some(x) = super::lighting::load_camera_fx(cache) {
        log(format!("h4 camera fx '{}': exposure range [{}, {}] stops, screen brightness {:.3} sensitivity {} (meter key {:.3}), self-illum stops pref {} change {}, bloom highlight {} inherent {} self-illum {} intensity {} colours {:?}, filmic {:?}",
            x.name.rsplit('\\').next().unwrap_or(&x.name), x.exposure_range[0], x.exposure_range[1], x.screen_brightness, x.sensitivity, x.meter_key(), x.si_preferred_stops, x.si_change, x.bloom_highlight, x.bloom_inherent, x.bloom_self_illum, x.bloom_intensity, x.bloom_colors, x.filmic_params()));
        // the colour-grading volume (every shipped MP cfxs carries one)
        match x.color_grading.map(|t| (t, super::bitmaps::load_volume_lut(cache, t))) {
            Some((t, Ok((b, n, dxgi)))) => { log(format!("h4 colour grading '{}': {n}^3 LUT (dxgi {dxgi})", cache.tag_name(t).rsplit('\\').next().unwrap_or(""))); stats.color_grading_lut = Some((b, n)); }
            Some((t, Err(e))) => log(format!("h4 colour grading '{}': not loaded ({e})", cache.tag_name(t))),
            None => log("h4 colour grading: none (cfxs +0xF0 null)".into()),
        }
        stats.camera_fx = Some(x);
    }
    stats.seconds = t0.elapsed().as_secs_f32();
    log(format!("h4 uploaded {} draws, {} tris, {} verts (instanced), decode failures {}, textures ok {} failed {} ({:.2}s)",
        stats.draws, stats.tris, stats.verts, stats.decode_fail, stats.tex_ok, stats.tex_fail.len(), stats.seconds));
    for f in stats.tex_fail.iter().take(8) { log(format!("  tex: {f}")); }
    log(format!("h4 world bounds min {:?} max {:?}", stats.bmin, stats.bmax));
    Ok(stats)
}

/// Grow the object caster AABB by one placed model instance (`transform_aabb`).
pub fn grow_caster_bounds(b: &mut Option<(Vec3, Vec3)>, min: Vec3, max: Vec3, m: &Mat4) {
    let (mn, mx) = crate::scene::transform_aabb(min, max, m);
    if !(mn.is_finite() && mx.is_finite()) { return; }
    *b = Some(match *b { Some((a, c)) => (a.min(mn), c.max(mx)), None => (mn, mx) });
}

/// The object lighting sources `build_meshes` collected (the composed atlases for the surface
/// probe, the Lbsp airprobes, the Lbsp instance probes at their instance positions and the mean
/// per-vertex baked colour), owned so the editor scene can light NEW objects exactly like the
/// static path does (`object_lanes_aabb`).
#[derive(Default)]
pub struct ObjectLightField {
    /// (instance position, bsp index, probe index)
    pub probe_field: Vec<([f32; 3], usize, usize)>,
    /// Per BSP (same order as `H4Loaded::bsps`): its Lbsp probes.
    pub probes: Vec<Vec<H4Probe>>,
    pub mean_bake: [f32; 4],
    /// Per BSP: the composed atlas bytes + (K_direct, K_indirect) for the CPU
    /// surface-probe texel sample (None = no per-pixel atlas).
    pub atlas_cpu: Vec<Option<(Arc<[(Vec<u8>, u32, u32); 4]>, f32, f32)>>,
    /// Every Lbsp airprobe of the map (volume ids are string ids, unique per map).
    pub airprobes: Vec<H4AirProbe>,
    /// Atlas uv2 corners of the lightmapped soup triangles, indexed by the soup's
    /// material word `.1` (`.0` = bsp + 1, 0 = not lightmapped).
    pub tri_uv2: Vec<[[f32; 2]; 3]>,
    /// Print-only tally: objects lit by (surface probe, airprobe, instance probe, mean bake).
    pub stats: std::sync::Mutex<[u32; 4]>,
}

/// The object lighting lanes (halo4.dll model lighting entry 11, docs/halo4_lighting_model.md
/// section 10): instance colour = ambient rgb + sqrt(sun visibility) in .a (the shader squares
/// `i.tint.a`; the engine's OBJECT sun visibility gate is `sub_1802E5A28`: `+28 = analytic.x *
/// scalar`, sharpened by entry 09 / 11 as the BSP does), and the packed per-instance lobe lane =
/// two SH-L1 lobes (8-bit octahedral dirs, f16 widths, rgb9e5 colours) that h4_shade evaluates
/// per pixel through the sharpen LUT like the BSP lobes.
fn lanes_pack(col: [f32; 3], vis: f32, lobes: Option<([f32; 3], f32, [f32; 3], [f32; 3], f32, [f32; 3])>) -> ([f32; 4], [u32; 4]) {
    let a = vis.clamp(0.0, 1.0).sqrt();
    let l = match lobes {
        Some((da, wa, ca, db, wb, cb)) => {
            let oa = crate::scene::oct_encode(Vec3::from(da).normalize_or_zero());
            let ob = crate::scene::oct_encode(Vec3::from(db).normalize_or_zero());
            let sn = |v: f32| ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8 as u32;
            let x = sn(oa[0]) | (sn(oa[1]) << 8) | (sn(ob[0]) << 16) | (sn(ob[1]) << 24);
            // the lobe LENGTH |d| rides the width lane: w = sqrt(1 - |d|^2) is what the shader needs,
            // and a lobe of zero length (w = 1) must still count -> keep w >= 1e-4 as the "present" flag
            let y = crate::scene::pack_f16x2(wa.max(1e-4), wb.max(1e-4));
            [x, y, pack_rgb9e5(ca), pack_rgb9e5(cb)]
        }
        None => [0; 4],
    };
    ([col[0], col[1], col[2], a], l)
}

impl ObjectLightField {
    /// Nearest instance probe within 60 wu.
    pub fn probe_at(&self, pos: [f32; 3]) -> Option<&H4Probe> {
        let mut best: Option<(f32, usize, usize)> = None;
        for (pp, bi, pi) in &self.probe_field {
            let d2 = (pp[0] - pos[0]).powi(2) + (pp[1] - pos[1]).powi(2) + (pp[2] - pos[2]).powi(2);
            if best.map_or(true, |b| d2 < b.0) { best = Some((d2, *bi, *pi)); }
        }
        match best {
            Some((d2, bi, pi)) if d2 < 60.0 * 60.0 => self.probes.get(bi).and_then(|v| v.get(pi)),
            _ => None,
        }
    }
    /// Nearest instance probe within 60 wu: hemisphere-averaged irradiance, else the map mean.
    pub fn light(&self, pos: [f32; 3]) -> [f32; 4] {
        match self.probe_at(pos) {
            Some(p) => {
                let mut acc = [0.0f32; 3];
                for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]] {
                    let e = super::lightmaps::probe_irradiance(p, n);
                    for k in 0..3 { acc[k] += e[k] / 6.0; }
                }
                [acc[0], acc[1], acc[2], self.mean_bake[3]]
            }
            None => self.mean_bake,
        }
    }
    /// The ENGINE's object lighting (halo4.dll `sub_1802822B4` -> `sub_1802E5208`;
    /// lightmaps.rs "AIRPROBES" comment for the RE): in order,
    ///   1. SURFACE PROBE: a ray from the object's bounding centre + 0.2 wu straight down (the
    ///      engine's lighting point without a model marker; ray reach = the half height + 1 wu,
    ///      the engine's 1 wu from its point) onto a LIGHTMAPPED BSP triangle -> the texel's two
    ///      lobes become the object's diffuse lights (no LUT, `cb3[7].w` = 1), the analytic
    ///      texel its sun visibility; the SH registers stay zero.
    ///   2. AIRPROBE volume (Lbsp +0x3E8, `airprobe_sample`): SH only (`cb3[7].w` = 0), the
    ///      blended probe sun visibility; per-pixel SH is PROVISIONAL (hemisphere mean).
    ///   3. HMS stand-ins when the map has neither: the nearest Lbsp instance probe (SH only),
    ///      then the mean per-vertex bake.
    /// `vis` = the raycast sun visibility against the BSP (the object's own forge lightmap
    /// stand-in); the texel / probe visibility multiplies it.
    pub fn object_lanes_aabb(&self, soup: Option<&crate::decal_projector::TriSoup>, mn: Vec3, mx: Vec3, vis: f32) -> ([f32; 4], [u32; 4]) {
        let c = (mn + mx) * 0.5;
        let pos = c.to_array();
        let tally = |k: usize| { if let Ok(mut s) = self.stats.lock() { s[k] += 1; } };
        let diag = std::env::var("HMS_H4_LMDIAG").is_ok(); // print-only
        // 1. surface probe
        if let Some(soup) = soup {
            let origin = Vec3::new(c.x, c.y, c.z + 0.2);
            let reach = (mx.z - mn.z) * 0.5 + 1.2;
            let hit = soup.raycast_hit(origin, Vec3::NEG_Z, reach);
            if diag { eprintln!("HMS_H4_LMDIAG object at {:?} h {:.2} reach {:.2}: hit {:?} mat {:?}", origin, mx.z - mn.z, reach, hit.map(|h| (h.0, h.2)), hit.and_then(|h| soup.tri_material(h.2))); }
            if let Some((t, _n, tri)) = hit {
                if let Some((bsp1, ti, _)) = soup.tri_material(tri) {
                    if bsp1 > 0 {
                        if let (Some(Some((tex, k_dom, k))), Some(uv)) = (self.atlas_cpu.get(bsp1 as usize - 1), self.tri_uv2.get(ti as usize)) {
                            if let Some([a, b, cc]) = soup.tris().get(tri as usize) {
                                let hit = origin + Vec3::NEG_Z * t;
                                let (u, v, w) = barycentric(hit, *a, *b, *cc);
                                let uv2 = [u * uv[0][0] + v * uv[1][0] + w * uv[2][0], u * uv[0][1] + v * uv[1][1] + w * uv[2][1]];
                                let (da, wa, ca, db, wb, cb, tvis) = atlas_texel_lobes(tex, *k_dom, *k, uv2);
                                tally(0);
                                return lanes_pack([0.0; 3], vis * tvis, Some((da, wa, ca, db, wb, cb)));
                            }
                        }
                    }
                }
            }
        }
        // 2. airprobe volume
        if let Some((sh, pvis)) = airprobe_sample(&self.airprobes, pos) {
            tally(1);
            return lanes_pack_sh(&sh, vis * pvis);
        }
        // 3. stand-ins (the engine leaves such an object BLACK - `sub_18036FB70` reset record):
        // the nearest airprobe of the map beyond the 60 wu radius, the nearest Lbsp instance
        // probe, then the mean per-vertex bake
        if let Some(p) = self.airprobes.iter().min_by(|a, b| {
            let d = |p: &H4AirProbe| (p.pos[0] - pos[0]).powi(2) + (p.pos[1] - pos[1]).powi(2) + (p.pos[2] - pos[2]).powi(2);
            d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            tally(1);
            return lanes_pack_sh(&p.sh, vis * p.vis);
        }
        match self.probe_at(pos) {
            Some(p) => { tally(2); lanes_pack(sh_hemisphere_mean(p), vis, None) }
            None => { tally(3); lanes_pack([self.mean_bake[0], self.mean_bake[1], self.mean_bake[2]], vis, None) }
        }
    }
}

/// The AIRPROBE lane (mesh.rs h4_shade `airprobe_sh`): tint.rgb = the engine
/// packer's per-channel DC term (`probe_sh` constants: 0.2821 c0 - 0.0788 c6), tint.a =
/// -(0.25 + 0.5 sqrt(vis)) (the mode flag + sun visibility), obj_light = the other 8 quadratic-form
/// coefficients of the LUMINANCE-weighted SH as f16 pairs (a.xyz, b.xyzw, c). PROVISIONAL: the
/// per-pixel colour = DC chroma x the luminance shape (no free instance attribute for 27 values).
fn lanes_pack_sh(sh: &[[f32; 9]; 3], vis: f32) -> ([f32; 4], [u32; 4]) {
    let sp = std::f32::consts::PI.sqrt();
    let (k0, k1, k2, k3) = (1.0 / (2.0 * sp), 3f32.sqrt() / (3.0 * sp), 15f32.sqrt() / (8.0 * sp), 5f32.sqrt() / (16.0 * sp));
    let wl = [0.2126f32, 0.7152, 0.0722];
    let mut dc = [0.0f32; 3];
    let mut lum = [0.0f32; 8]; // a.x a.y a.z b.x b.y b.z b.w c
    for ch in 0..3 {
        let c = &sh[ch];
        dc[ch] = k0 * c[0] - k3 * c[6];
        let v = [-(k1 * c[3]), -(k1 * c[1]), k1 * c[2], k2 * c[4], -(k2 * c[5]), 3.0 * k3 * c[6], -(k2 * c[7]), 0.5 * k2 * c[8]];
        for k in 0..8 { lum[k] += wl[ch] * v[k]; }
    }
    let a = -(0.25 + 0.5 * vis.clamp(0.0, 1.0).sqrt());
    let pk = |x: f32, y: f32| crate::scene::pack_f16x2(x, y);
    ([dc[0].max(0.0), dc[1].max(0.0), dc[2].max(0.0), a], [pk(lum[0], lum[1]), pk(lum[2], lum[3]), pk(lum[4], lum[5]), pk(lum[6], lum[7])])
}

/// The probe's quadratic SH averaged over the 6 axis directions (PROVISIONAL stand-in for the
/// per-pixel SH(N) of the model lighting entries).
fn sh_hemisphere_mean(p: &H4Probe) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]] {
        let e = super::lightmaps::probe_sh(p, n);
        for k in 0..3 { acc[k] += e[k] / 6.0; }
    }
    acc
}

/// Barycentric weights of `p` (projected onto the triangle's plane) w.r.t. (a, b, c).
fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, f32, f32) {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-12 { return (1.0, 0.0, 0.0); }
    let v = ((d11 * d20 - d01 * d21) / den).clamp(0.0, 1.0);
    let w = ((d00 * d21 - d01 * d20) / den).clamp(0.0, 1.0);
    let u = (1.0 - v - w).clamp(0.0, 1.0);
    (u, v, w)
}

/// Shared-exponent RGB9E5 (the WGSL `h4_unpack_rgb9e5` inverse): 9-bit mantissas, 5-bit
/// exponent biased by 15, value = m * 2^(e - 24). Negative / NaN -> 0; max 65408.
pub fn pack_rgb9e5(c: [f32; 3]) -> u32 {
    let cl = |v: f32| if v.is_finite() { v.clamp(0.0, 65408.0) } else { 0.0 };
    let (r, g, b) = (cl(c[0]), cl(c[1]), cl(c[2]));
    let m = r.max(g).max(b);
    if m <= 0.0 { return 0; }
    let mut e = (m.log2().floor() as i32 + 1 + 15).clamp(0, 31);
    let mut scale = ((e - 24) as f32).exp2();
    if (m / scale + 0.5).floor() >= 512.0 { e = (e + 1).min(31); scale = ((e - 24) as f32).exp2(); }
    let q = |v: f32| ((v / scale + 0.5).floor() as u32).min(511);
    q(r) | (q(g) << 9) | (q(b) << 18) | ((e as u32) << 27)
}

/// An object's static sun visibility: the fraction of 5 rays (its world AABB centre and the 4
/// top-face corners pulled 10% inward) that reach the sun without hitting the BSP soup - the
/// stand-in for the engine's airprobe sun term at the object position. No soup (plain headless
/// load) -> 1.0 (right for Forge objects in the open; indoor scenario objects get lit as if
/// outdoors on that path).
pub fn object_sun_vis(soup: Option<&crate::decal_projector::TriSoup>, sun_dir: Vec3, min: Vec3, max: Vec3) -> f32 {
    let Some(soup) = soup else { return 1.0 };
    let c = (min + max) * 0.5;
    let e = (max - min) * 0.4;
    let pts = [c, Vec3::new(c.x - e.x, c.y - e.y, max.z), Vec3::new(c.x + e.x, c.y - e.y, max.z), Vec3::new(c.x - e.x, c.y + e.y, max.z), Vec3::new(c.x + e.x, c.y + e.y, max.z)];
    let mut lit = 0;
    for p in pts {
        if soup.raycast(p + sun_dir * 0.05, sun_dir, 2000.0).is_none() { lit += 1; }
    }
    lit as f32 / pts.len() as f32
}

/// The (instance colour, lobe lane) pairs of one model's placements.
pub fn object_light_lanes(light: &ObjectLightField, soup: Option<&crate::decal_projector::TriSoup>, sun_dir: Vec3, gmin: Vec3, gmax: Vec3, mats: &[Mat4]) -> (Vec<[f32; 4]>, Vec<[u32; 4]>) {
    let mut cols = Vec::with_capacity(mats.len());
    let mut lanes = Vec::with_capacity(mats.len());
    for m in mats {
        let (mn, mx) = crate::scene::transform_aabb(gmin, gmax, m);
        let vis = object_sun_vis(soup, sun_dir, mn, mx);
        let (c, l) = light.object_lanes_aabb(soup, mn, mx, vis);
        cols.push(c);
        lanes.push(l);
    }
    (cols, lanes)
}

/// What `build_meshes` collects for an editor scene when one is requested (GUI always,
/// headless with HMS_H4_SOUP=1): the world-space raycast soup of every drawn BSP triangle, the
/// object-light field and the texture context (so the editor re-uses the already uploaded bitmaps).
pub struct H4EditorCollect {
    pub soup: crate::decal_projector::TriSoup,
    pub light: ObjectLightField,
    pub(crate) tex: Option<TexCtx>,
    /// The scenario's spawn-family markers (with their lane), uploaded but NOT
    /// emitted: the GUI hands them to the renderer's marker lane (`split_marker_meshes`), shown
    /// only while "Show map spawn points" is on.
    pub map_spawn_meshes: Vec<(Lane, GpuMesh)>,
}

impl H4EditorCollect {
    pub fn new() -> Self { H4EditorCollect { soup: crate::decal_projector::TriSoup::new(4.0), light: ObjectLightField::default(), tex: None, map_spawn_meshes: Vec::new() } }
}

impl Default for H4EditorCollect {
    fn default() -> Self { Self::new() }
}

/// Marker meshes by lane -> the renderer's `set_marker_meshes` argument order
/// (opaque + alpha-test, additive, blend; the sky lane never carries a marker).
pub fn split_marker_meshes(v: Vec<(Lane, GpuMesh)>) -> (Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>) {
    let (mut o, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for (lane, m) in v {
        match lane {
            Lane::Opaque | Lane::AlphaTest | Lane::Sky => o.push(m),
            Lane::Additive => a.push(m),
            Lane::Blend => b.push(m),
        }
    }
    (o, a, b)
}

/// Everything the editor scene needs once the load worker is done (`H4Msg::Done`):
/// the cache (shared), the base map's forge palette, the parsed variant (if any) and what
/// `build_meshes` collected (`H4EditorCollect`). `H4Cache` is an mmap + mutex caches, so this is
/// `Send` and crosses the worker channel.
pub struct H4EditorAssets {
    pub cache: Arc<H4Cache>,
    pub palette: Vec<super::mvar::H4PaletteEntry>,
    pub variant: Option<(std::path::PathBuf, super::mvar::H4Variant)>,
    pub collect: H4EditorCollect,
    /// Every scnr placement (read-only picks).
    pub placements: Vec<H4Placement>,
    /// World bounds of the decoded BSP geometry (`H4Stats::bmin/bmax`) for `scene_bounds`.
    pub world_bounds: ([f32; 3], [f32; 3]),
    /// The statically drawn scenario objects' caster AABB + the BSP's cascade config
    /// (`H4Stats::caster_bounds` / `cascade`), so the editor scene can keep the renderer's shadow
    /// fit truthful as its own objects move.
    pub static_caster_bounds: Option<(Vec3, Vec3)>,
    pub cascade: Option<hms_render::H4CascadeCfg>,
    /// Direction TO the sun (`H4Stats::sun_dir`).
    pub sun_to: Vec3,
}

impl H4EditorAssets {
    /// Move the editor-relevant parts out of a finished load.
    pub fn from_loaded(loaded: H4Loaded, collect: H4EditorCollect, stats: &H4Stats) -> Self {
        let palette = super::mvar::load_forge_palette(&loaded.cache);
        H4EditorAssets { cache: loaded.cache, palette, variant: loaded.variant_full, collect, placements: loaded.placements, world_bounds: (stats.bmin, stats.bmax), static_caster_bounds: stats.caster_bounds, cascade: stats.cascade, sun_to: stats.sun_dir() }
    }
}

/// One lod0 render model decoded for drawing / picking: every lod0 mesh's vertices
/// (renderer layout, flat white colour) + indices + parts, the material slots and the local AABB.
pub struct H4ModelGeom {
    pub name: String,
    pub materials: Vec<Option<usize>>,
    pub meshes: Vec<H4GeomMesh>,
    pub min: Vec3,
    pub max: Vec3,
    /// Per-mesh decode failures (reported by the caller).
    pub mesh_fail: Vec<String>,
}

pub struct H4GeomMesh {
    pub verts: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<Part>,
}

impl H4ModelGeom {
    /// Triangles over every mesh (for the picks / wireframe).
    pub fn tri_count(&self) -> usize { self.meshes.iter().map(|m| m.indices.len() / 3).sum() }
}

/// `load_model` + `decode_model_mesh` over the model's lod0 meshes (the static
/// object path's per-model decode, shared with the editor's model cache).
pub fn decode_model_geom(cache: &H4Cache, mode: usize) -> Result<H4ModelGeom> {
    let model = load_model(cache, mode)?;
    let meshes = model.lod0_meshes.clone();
    decode_model_geom_meshes(cache, &model, &meshes)
}

/// `decode_model_geom` for the meshes an OBJECT shows in hlmt model variant
/// `variant` (objects.rs `variant_meshes`: the variant's permutation per region, hidden regions
/// dropped; lod0 when the model has no variants). `mode` must be the object's render model.
pub fn decode_object_geom(cache: &H4Cache, obje: usize, mode: usize, variant: Option<usize>) -> Result<H4ModelGeom> {
    let model = load_model(cache, mode)?;
    let meshes = super::objects::variant_meshes(cache, obje, &model, variant);
    decode_model_geom_meshes(cache, &model, &meshes)
}

fn decode_model_geom_meshes(cache: &H4Cache, model: &super::objects::H4Model, meshes: &[usize]) -> Result<H4ModelGeom> {
    let mut g = H4ModelGeom { name: model.name.clone(), materials: model.materials.clone(), meshes: Vec::new(), min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN), mesh_fail: Vec::new() };
    for &mi in meshes {
        let dm = match decode_model_mesh(cache, model, mi) {
            Ok(Some(d)) => d,
            Ok(None) => continue,
            Err(e) => { g.mesh_fail.push(format!("{} mesh {mi}: {e}", model.name)); continue; }
        };
        let verts: Vec<MeshVertex> = dm.verts.iter().map(|v| MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: [1.0; 4], ..Default::default() }).collect();
        for v in &verts { g.min = g.min.min(Vec3::from(v.pos)); g.max = g.max.max(Vec3::from(v.pos)); }
        g.meshes.push(H4GeomMesh { verts, indices: dm.indices, parts: dm.parts });
    }
    if g.min.x > g.max.x { g.min = Vec3::ZERO; g.max = Vec3::ZERO; }
    Ok(g)
}

/// Upload every drawable part of one decoded model with ALL its instance matrices
/// (`mats`) and per-instance object-light colours (`cols`, parallel to `mats`), routed per material
/// lane through `out(lane, mesh)`. The body of the static object section of `build_meshes`, shared
/// with the editor scene's rebuild so both draw byte-identically. Returns true when anything drew.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_model_parts(
    tc: &mut TexCtx,
    mr: &MeshRenderer,
    geom: &H4ModelGeom,
    mats: &[Mat4],
    cols: &[[f32; 4]],
    lanes: &[[u32; 4]],
    // The object's PRIMARY change colour (`h4::tint::primary_change_color`) - the shipped
    // `*_colorchangemap` materials' `ps_material_object_parameters[0]`. `NEUTRAL_CC` when the
    // caller has none. #h4-veh
    cc: [f32; 4],
    obj_mats: &mut HashMap<usize, H4Material>,
    shading_on: bool,
    stats: &mut H4Stats,
    out: &mut dyn FnMut(Lane, GpuMesh),
) -> bool {
    let cache = tc.cache.clone();
    let (device, queue) = (tc.device.clone(), tc.queue.clone());
    let mut any = false;
    for mesh in &geom.meshes {
        let verts = &mesh.verts;
        for part in &mesh.parts {
            let s = part.index_start as usize;
            let e = (s + part.index_count as usize).min(mesh.indices.len());
            if e <= s { continue; }
            let indices = &mesh.indices[s..e];
            let mat_tag = geom.materials.get(part.material.max(0) as usize).copied().flatten();
            let m = mat_tag.map(|t| obj_mats.entry(t).or_insert_with(|| load_material(&cache, t)).clone());
            let kind = m.as_ref().map(|m| m.kind).unwrap_or(H4MatKind::Opaque);
            if tc.shade_diag { // HMS_H4_MATDIAG print-only: every OBJECT part's material + lane
                if let Some(mm) = &m {
                    let texs: Vec<String> = mm.textures.iter().map(|t| format!("{}={}{:?}", t.name, t.bitm.map(|b| cache.tag_name(b).rsplit('\\').next().unwrap_or("").to_string()).unwrap_or_default(), t.xform)).collect();
                    eprintln!("  HMS_H4_MATDIAG obj '{}' part idx {} mat '{}' mats '{}' kind {:?} blend {} ({:?}) flags {:#x} bools {:#x} pp_flags {:#x} alpha {} sort {}/{} | floats {:?} | tex {}",
                        geom.name.rsplit('\\').next().unwrap_or(&geom.name), e - s, mm.name.rsplit('\\').next().unwrap_or(&mm.name),
                        mm.mats_name.rsplit('\\').next().unwrap_or(&mm.mats_name), mm.kind, mm.blend_mode_raw, mm.blend_mode, mm.mats_flags, mm.bools, mm.pp_flags,
                        mm.base_alpha, mm.sort_layer, mm.sort_offset, mm.floats, texs.join(" "));
                } else {
                    eprintln!("  HMS_H4_MATDIAG obj '{}' part idx {} NO MATERIAL", geom.name.rsplit('\\').next().unwrap_or(&geom.name), e - s);
                }
            }
            let Some(lane) = lane_for(kind) else { stats.skipped_invisible += 1; continue };
            let base_alpha = m.as_ref().map(|m| m.base_alpha).unwrap_or(1.0);
            let tex = m.as_ref().and_then(|m| m.diffuse()).and_then(|t| tc.fetch_tex(t, stats).0);
            // objects take the same material lane; their lighting rides the per-instance colour
            // and lobe lanes (`object_light_lanes`) instead of a lightmap binding
            // `srf_ca_boundary*` (the Forge grid / boundary walls) is a shipped ADDITIVE
            // self-illum family with its own pixel shader; it draws in the halogram lane.
            let bnd = match (&m, shading_on) { (Some(mm), true) => tc.shading_of(mm).boundary, _ => None };
            let gm = match (&m, bnd, shading_on && matches!(kind, H4MatKind::Opaque | H4MatKind::AlphaTest)) {
                (_, Some(b), _) => upload_part_boundary(tc, mr, verts, indices, mats, &b, stats),
                (Some(mm), None, true) => {
                    let (sh, ht) = tc.resolve_h4(mm, stats);
                    let mut gm = upload_part_h4(mr, &device, &queue, verts, indices, mats, &ht, &sh, kind, base_alpha, cc, None);
                    // per-placement probe / mean-bake irradiance rides the instance colour
                    // (h4_shade: diffuse = i.tint.rgb, sun vis = i.tint.a^2 when no lightmap)
                    gm.set_instance_colors(&queue, cols);
                    if !lanes.is_empty() { gm.set_instance_obj_light(&queue, lanes); } // probe lobes
                    gm
                }
                _ => upload_part_tinted(mr, &device, &queue, verts, indices, mats, tex.as_ref(), kind, base_alpha, None, m.as_ref().and_then(emissive_tint)),
            };
            // every Halo 4 object is a forge-lightmap shadow caster (halo4.dll
            // sub_1803443DC draws the registered objects, nothing else, into the burn map)
            let mut gm = gm;
            gm.h4_caster = true;
            gm.casts_shadow = true;
            out(lane, gm);
            *stats.lanes.entry(lane).or_default() += 1;
            stats.tris += indices.len() / 3 * mats.len();
            stats.draws += 1;
            stats.object_draws += 1;
            any = true;
        }
    }
    any
}

/// The scenario's SKY object (scnr Skies block +0xDC, 0x34-B elements, `scen` ref @0 ->
/// hlmt -> mode) as renderer sky segments. The Halo 4 sky is plain unlit emissive geometry
/// (from the sky shaders' disassembly): `srf_ca_skybox_gradient` = lerp(bottom, top, sat(uv.y ^ p0.x + p0.y))
/// * p2.w; `srf_ca_color_warp*` / `srf_constant` = diffuse * p0.rgb * p0.w + selfillum * p1.rgb *
/// p1.w (the same bitmap on every shipped part, so the two tints are summed into the vertex colour);
/// alpha-blended parts use the diffuse alpha (x the per-vertex mask on `*_vertmask`;
/// `srf_constant_vertalpha` = lerp(vertex.a, vertex.a * color.a, p2.z), blend 9 = add x src alpha,
/// `srf_constant` uv scroll p2.xy * time). The uv warp / fresnel terms are not ported
/// (PROVISIONAL). Reach sky blend enum: 0 opaque, 1 additive, 4 alpha, 9 add x src alpha (H4 only).
pub const OFF_SCNR_SKIES: usize = 0xDC;
pub const SCNR_SKY_ELEM: usize = 0x34;

fn build_sky(loaded: &H4Loaded, device: &wgpu::Device, queue: &wgpu::Queue, mr: &MeshRenderer, tc: &mut TexCtx, stats: &mut H4Stats, log: &mut dyn FnMut(String)) -> Vec<GpuMesh> {
    let cache: &H4Cache = &loaded.cache;
    let mut out = Vec::new();
    let Some(scnr) = cache.find_tags(b"scnr").first().copied() else { return out };
    let Some(sm) = cache.tag_meta(scnr) else { return out };
    let Some((n, so)) = cache.block(sm + OFF_SCNR_SKIES) else { return out };
    let white = hms_render::upload_texture_bgra(device, queue, &[255, 255, 255, 255], 1, 1).0;
    for k in 0..n.min(4) {
        let Some(scen) = cache.tag_ref_of(so + k * SCNR_SKY_ELEM, b"scen") else { continue };
        let Some(mode) = model_of(cache, scen) else { log(format!("h4 sky '{}': no render model", cache.tag_name(scen))); continue };
        let model = match load_model(cache, mode) { Ok(m) => m, Err(e) => { log(format!("h4 sky '{}': {e}", cache.tag_name(mode))); continue } };
        let mut parts = 0usize;
        for &mi in &model.lod0_meshes {
            let Ok(Some(dm)) = decode_model_mesh(cache, &model, mi) else { continue };
            for part in &dm.parts {
                let s0 = part.index_start as usize;
                let e0 = (s0 + part.index_count as usize).min(dm.indices.len());
                if e0 <= s0 { continue; }
                let mat_tag = model.materials.get(part.material.max(0) as usize).copied().flatten();
                let Some(mt) = mat_tag else { continue };
                let m = load_material(cache, mt);
                let fam = m.mats_name.rsplit('\\').next().unwrap_or("").to_ascii_lowercase();
                if m.kind == H4MatKind::Invisible { continue; }
                let f = |i: usize| m.floats.get(i).map(|f| f.1).unwrap_or([0.0; 4]);
                let (p0, p1, p2) = (f(0), f(1), f(2));
                if tc.shade_diag { // HMS_H4_MATDIAG print-only: the sky part's material + vertex streams
                    let md = &model.meshes[mi];
                    let vbs: Vec<String> = md.vb.iter().filter(|&&v| v >= 0).filter_map(|&v| model.geometry.vertex_buffer(v)).map(|v| format!("k{} s{} n{}", v.kind, v.stride, v.count)).collect();
                    let (mut bmin, mut bmax, mut bsum) = (f32::MAX, f32::MIN, 0.0f32);
                    let (mut rmin, mut rmax, mut zmin, mut zmax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                    for &ix in &dm.indices[s0..e0] { let v = &dm.verts[ix as usize]; let r = Vec3::from(v.pos).length(); rmin = rmin.min(r); rmax = rmax.max(r); zmin = zmin.min(v.pos[2]); zmax = zmax.max(v.pos[2]); bmin = bmin.min(v.blend); bmax = bmax.max(v.blend); bsum += v.blend; }
                    eprintln!("  HMS_H4_MATDIAG sky part radius {rmin:.1}..{rmax:.1} z {zmin:.1}..{zmax:.1} idx {}", e0 - s0);
                    let texs: Vec<String> = m.textures.iter().map(|t| format!("{}={}", t.name, t.bitm.map(|b| cache.tag_name(b).rsplit('\\').next().unwrap_or("").to_string()).unwrap_or_default())).collect();
                    eprintln!("  HMS_H4_MATDIAG sky part {} mesh {mi} fam {fam} blend {} kind {:?} p0 {p0:?} p1 {p1:?} p2 {p2:?} vbs [{}] blend@12 {:.3}..{:.3} mean {:.3} verts {} tex {}",
                        m.name.rsplit('\\').next().unwrap_or(""), m.blend_mode_raw, m.kind, vbs.join(", "), bmin, bmax, bsum / dm.verts.len().max(1) as f32, dm.verts.len(), texs.join(" "));
                }
                // `srf_constant_vertalpha` / `*_vertmask`: the PS reads the vertex alpha (v2.w = the
                // f32 @12 of the vertex; Forge Island's sky mesh: 0..1 per part) - alpha =
                // lerp(vertex.a, vertex.a * color.a, p2.z) (vertalpha) / color.a * vertex.a (vertmask).
                // Forge Island's fake-haze cone (blend 9, radius 470..2160 wu) depends on it: drawn
                // opaque it hides the dome and clouds behind a flat (1.48, 2.05, 2.27) shell.
                let vertmask = fam.contains("vertmask") || fam.contains("vertalpha");
                let tex_alpha_w = if fam.contains("vertalpha") { p2[2].clamp(0.0, 1.0) } else { 1.0 };
                let gradient = fam.contains("skybox_gradient");
                // self-illum exposure of the sky parts (sky shader asm, h4expo/sky):
                // `srf_ca_skybox_gradient` is exposed with the SCENE exposure (ps_view_exposure.x
                // only); `srf_constant` / `srf_ca_color_warp*` with lerp(E, E_si, sat(luma(selfillum
                // * p1 * p1.w))). The per-part factor f = sat(luma(p1 * p1.w)) (the texel luma <= 1
                // bound) splits the tint: (1 - f) rides the vertex colour (scene E), f the sky
                // shader's self-illum lane (x illum_scale_now() = E_si / E), same bitmap as the base.
                let si_f = if gradient { 0.0 } else { (0.3086 * p1[0] * p1[3] + 0.6094 * p1[1] * p1[3] + 0.082 * p1[2] * p1[3]).clamp(0.0, 1.0) };
                let tint = [p0[0] * p0[3] + p1[0] * p1[3], p0[1] * p0[3] + p1[1] * p1[3], p0[2] * p0[3] + p1[2] * p1[3]];
                // vertex colour = the summed emissive tints (or the gradient by uv.y)
                let verts: Vec<MeshVertex> = dm.verts.iter().map(|v| {
                    let mut c = [tint[0] * (1.0 - si_f), tint[1] * (1.0 - si_f), tint[2] * (1.0 - si_f), if vertmask { v.blend.clamp(0.0, 1.0) } else { 1.0 }];
                    if gradient {
                        let t = (v.uv[1].max(0.0).powf(p0[0].max(1e-3)) + p0[1]).clamp(0.0, 1.0);
                        for ch in 0..3 { c[ch] = (p1[ch] + (p2[ch] - p1[ch]) * t) * p2[3]; }
                    }
                    MeshVertex { pos: v.pos, normal: v.normal, uv: v.uv, color: c, ..Default::default() }
                }).collect();
                let (diff, diff_xf) = m.textures.iter().find(|t| matches!(t.name.as_str(), "diffuseMap" | "color_map")).map(|t| (t.bitm, t.xform)).unwrap_or((None, [1.0, 1.0, 0.0, 0.0]));
                let base = diff.and_then(|t| tc.fetch_tex_sky(t, stats)).unwrap_or_else(|| white.clone());
                // Reach sky blend enum: 0 opaque, 1 additive, 2 multiply, 3 double multiply, 4 alpha,
                // 9 = add_src_times_srcalpha (H4 blend 9; the sky lane premultiplies).
                let blend: u8 = match m.blend_mode_raw { 1 | 6 | 7 | 24 => 1, 2 => 2, 4 => 3, 3 | 5 | 10 | 22 | 23 => 4, 8 | 9 => 9, _ => 0 };
                let opt = if gradient { 2.0 } else { 0.0 };
                // instance colour = the sky shader's self-illum gain (x base texel x illum_scale_now);
                // .w = the colour-alpha weight of the H4 alpha lane (ctl.w = 1).
                let si_gain = [tint[0] * si_f, tint[1] * si_f, tint[2] * si_f, tex_alpha_w];
                let si_mode = if si_f > 0.0 { 1.0 } else { 0.0 };
                // srf_constant* uv scroll: uv += p2.xy * time before the xform (PS asm
                // `mad r0.xy, cb13[2].xyxx, cb0[0].z, v1.xyxx`); the lane animates xf.zw, so scale.
                let scroll = if gradient { [0.0, 0.0] } else { [p2[0] * diff_xf[0], p2[1] * diff_xf[1]] };
                let mut gm = mr.upload_mesh(device, queue, &verts, &dm.indices[s0..e0], &[Mat4::IDENTITY], Some(&[si_gain]),
                    Some(&base), if si_f > 0.0 { Some(&base) } else { None }, None, None, None, scroll, [1.0, 1.0], diff_xf, 0.0, 0.0, [0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], diff_xf /* sky si_xf */, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
                    None, false, 0.0, [0.0; 4], None, None, [0.0; 4], [1.0, 1.0, 0.0, 0.0], None, None, [0.0; 4], [opt, si_mode, blend as f32, 1.0], None);
                gm.set_blend_mode(blend);
                gm.set_centroid([-(out.len() as f32), 0.0, 0.0]);
                out.push(gm);
                parts += 1;
            }
        }
        log(format!("h4 sky '{}': {} parts as sky segments", model.name.rsplit('\\').next().unwrap_or(&model.name), parts));
    }
    out
}

/// The placeholder sun direction for a map whose Lbsps carry no baked sun.
pub fn placeholder_sun_dir() -> Vec3 { Vec3::new(0.4, 0.3, 0.85).normalize() }
