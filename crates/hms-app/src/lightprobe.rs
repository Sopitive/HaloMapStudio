//! Per-vertex VMF lightprobe (PVL) decode — the Reach BSP baked-lighting path.
//!
//! Port of the C# `NativeBspMeshAdapter` Phase-B decode + `LightprobeDirectionLut`.
//! Each BSP cluster carries a stride-4 ByteAddressBuffer of 8-byte VMF lobes
//! (fetched via `NativeDll::cluster_pvl`). For a given vertex we unpack the
//! dominant-lobe direction (a 7-bit index into the engine `g_direction_lut`),
//! the primary bounce RGB, the DC ambient, and a 2-bit sun-visibility term, then
//! evaluate `ambient*0.25 + primary*max(0,dot(N,dir))` and Reinhard-compress it
//! into a per-vertex modulation tint that rides in `MeshVertex.color`.
//!
//! This is a verbatim algorithm port (the offsets/bit-layout are RE-confirmed in
//! the C# reference); it reads the on-disk `.map` cache, so no running game is
//! needed to produce it.

use glam::Vec3;

/// Engine `g_direction_lut` — verbatim decode of HREK
/// `tags/rasterizer/direction_lut_1002.bitmap` (92 × A16B16G16R16-SNORM, engine
/// `.wzyx` swizzle applied). Each triple is one unit-length bounce direction;
/// the array index is the 7-bit direction byte.
#[rustfmt::skip]
const DIR_LUT: [[f32; 3]; 92] = [
    [0.0,0.0,1.0], [0.3607004,0.0,0.93268174], [0.11148279,0.343085,0.9326651], [0.6729128,0.0,0.73972183],
    [0.479495,0.34841287,0.8054148], [0.20791394,0.63994884,0.73975486], [0.8944272,0.0,0.4472136], [0.7843774,0.343085,0.51676375],
    [0.5687107,0.639914,0.5167961], [0.27637497,0.8506709,0.44718665], [-0.29179317,0.21198569,0.9326944], [-0.18317886,0.5636977,0.8054132],
    [-0.54441816,0.3955104,0.7397165], [-0.08389671,0.8519986,0.5167781], [-0.43288976,0.7386044,0.5167881], [-0.7236279,0.5257061,0.4472088],
    [-0.29179317,-0.21198569,0.9326944], [-0.592719,0.0,0.8054093], [-0.54441816,-0.3955104,0.7397165], [-0.83621573,0.18348587,0.5167941],
    [-0.83621573,-0.18348587,0.5167941], [-0.7236279,-0.5257061,0.4472088], [0.11148279,-0.343085,0.9326651], [-0.18317886,-0.5636977,0.8054132],
    [0.20791394,-0.63994884,0.73975486], [-0.43288976,-0.7386044,0.5167881], [-0.08389671,-0.8519986,0.5167781], [0.27637497,-0.8506709,0.44718665],
    [0.479495,-0.34841287,0.8054148], [0.5687107,-0.639914,0.5167961], [0.7843774,-0.343085,0.51676375], [0.96472794,-0.21198988,0.15607779],
    [0.96472794,0.21198988,0.15607779], [0.9051085,-0.39549857,-0.1560753], [0.98546225,0.0,-0.16989465], [0.9051085,0.39549857,-0.1560753],
    [0.7236279,-0.5257061,-0.4472088], [0.83621573,-0.18348587,-0.5167941], [0.83621573,0.18348587,-0.5167941], [0.7236279,0.5257061,-0.4472088],
    [0.49976903,0.85198164,0.156071], [0.09650426,0.98301893,0.15607932], [0.6558125,0.73861307,-0.15607898], [0.30449757,0.93723726,-0.16990457],
    [-0.09650426,0.98301893,-0.15607932], [0.43288976,0.7386044,-0.5167881], [0.08389671,0.8519986,-0.5167781], [-0.27637497,0.8506709,-0.44718665],
    [-0.6558125,0.73861307,0.15607898], [-0.9051085,0.39549857,0.1560753], [-0.49976903,0.85198164,-0.156071], [-0.7972978,0.57918155,-0.16989692],
    [-0.96472794,0.21198988,-0.15607779], [-0.5687107,0.639914,-0.5167961], [-0.7843774,0.343085,-0.51676375], [-0.8944272,0.0,-0.4472136],
    [-0.9051085,-0.39549857,0.1560753], [-0.6558125,-0.73861307,0.15607898], [-0.96472794,-0.21198988,-0.15607779], [-0.7972978,-0.57918155,-0.16989692],
    [-0.49976903,-0.85198164,-0.156071], [-0.7843774,-0.343085,-0.51676375], [-0.5687107,-0.639914,-0.5167961], [-0.27637497,-0.8506709,-0.44718665],
    [0.09650426,-0.98301893,0.15607932], [0.49976903,-0.85198164,0.156071], [-0.09650426,-0.98301893,-0.15607932], [0.30449757,-0.93723726,-0.16990457],
    [0.6558125,-0.73861307,-0.15607898], [0.08389671,-0.8519986,-0.5167781], [0.43288976,-0.7386044,-0.5167881], [0.7972978,0.57918155,0.16989692],
    [-0.30449757,0.93723726,0.16990457], [-0.98546225,0.0,0.16989465], [-0.30449757,-0.93723726,0.16990457], [0.7972978,-0.57918155,0.16989692],
    [0.0,0.0,-1.0], [-0.11148279,0.343085,-0.9326651], [0.29179317,0.21198569,-0.9326944], [-0.20791394,0.63994884,-0.73975486],
    [0.18317886,0.5636977,-0.8054132], [0.54441816,0.3955104,-0.7397165], [-0.3607004,0.0,-0.93268174], [-0.6729128,0.0,-0.73972183],
    [-0.479495,0.34841287,-0.8054148], [-0.11148279,-0.343085,-0.9326651], [-0.20791394,-0.63994884,-0.73975486], [-0.479495,-0.34841287,-0.8054148],
    [0.29179317,-0.21198569,-0.9326944], [0.54441816,-0.3955104,-0.7397165], [0.18317886,-0.5636977,-0.8054132], [0.592719,0.0,-0.8054093],
];

fn dir_lut(index: usize) -> Vec3 {
    // Engine texture-clamps an OOB index to the last texel (the 92-wide direction_lut_1002 bitmap),
    // NOT a +Z pole (RE). Real data rarely exceeds 91; clamp matches the engine.
    Vec3::from(DIR_LUT[index.min(DIR_LUT.len() - 1)])
}

/// Decode one vertex's baked-lighting modulation tint from the cluster's PVL
/// buffer. `vid` is the mesh-local vertex index; `pvb_offset` the cluster's
/// `pervertex_block_offset`; `hdr_scale` the per-block scale; `normal` the
/// vertex normal. Returns an RGB multiplier centred near 1.0 (so it modulates
/// the analytical lighting, brighter in open bounce, darker in shadow pockets).
/// Returns white (neutral) for out-of-buffer vertices — matching the engine's
/// D3D11 raw-buffer OOB-reads-return-0 behaviour + our neutral fallback.
///
/// Returns `[r, g, b, vis]`: rgb is the baked-lighting modulation tint; `vis` is the
/// 2-bit baked SUN VISIBILITY (0..1) the caller stores in vertex-colour alpha to gate
/// the shader's analytical `sun_term` (engine `entry_points.hlsl_include:431`:
/// analytical sun × `vmf[0].w`). Without this, enclosed interior surfaces — whose baked
/// lightmap is dark — still receive full analytical sun and render fullbright.
/// Out-of-buffer verts return vis=1.0 (neutral: no baked occlusion known → full sun).
static VIS_HIST: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
];

/// Report and reset the raw 2-bit PVL sun-visibility population.
/// NON-draining read of the baked per-vertex sun-visibility population. `vis_hist_take`
/// swaps the counters to zero, so anything that needs the population AFTER the diagnostic ran (or
/// alongside it) must peek instead.
pub fn vis_hist_peek() -> [u64; 4] {
    let mut o = [0u64; 4];
    for i in 0..4 { o[i] = VIS_HIST[i].load(std::sync::atomic::Ordering::Relaxed); }
    o
}

pub fn vis_hist_take() -> [u64; 4] {
    let mut o = [0u64; 4];
    for i in 0..4 { o[i] = VIS_HIST[i].swap(0, std::sync::atomic::Ordering::Relaxed); }
    o
}

pub fn decode_pvl_tint(bytes: &[u8], vid: usize, pvb_offset: i32, hdr_scale: f32, normal: Vec3, hdr_out: bool, dom_sep: f32) -> [f32; 4] {
    let vid_in_cluster = vid as i64 - pvb_offset as i64;
    if vid_in_cluster < 0 {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let vid_in_cluster = vid_in_cluster as usize;
    // Sliding 8-byte window at 6-byte logical stride over the stride-4 buffer:
    // byteOff = floor(vid * 1.5) * 4 = ((vid*3)>>1)*4. Bytes past the end read 0.
    let o = ((vid_in_cluster * 3) >> 1) * 4;
    let at = |k: usize| -> u8 { bytes.get(o + k).copied().unwrap_or(0) };
    let (b0, b1, b2, b3, b4, b5, b6, b7) =
        (at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7));

    // Parity-based byte selection (adjacent pairs share 4 bytes; engine swaps
    // the mapping per fmod(vid,2)).
    let odd = (vid_in_cluster & 1) != 0;
    let (dir_byte, pri_r, pri_g, pri_b, amb_a, amb_b) = if odd {
        (b2, b5, b6, b7, b3, b4)
    } else {
        (b0, b3, b4, b5, b1, b2)
    };

    let dir_idx = (dir_byte & 0x7F) as usize;
    let dir_hi = (dir_byte >> 7) & 1;
    let bounce_dir = dir_lut(dir_idx);

    let eff = if hdr_scale.is_finite() && hdr_scale > 0.0 { hdr_scale } else { 1.0 };

    // Primary bounce RGB: byte/255 then squared (gamma 2.0), scaled.
    let sq = |b: u8| { let v = b as f32 / 255.0; v * v };
    let (pr, pg, pb) = (sq(pri_r) * eff, sq(pri_g) * eff, sq(pri_b) * eff);

    // Engine-correct DC ambient: 5-bit channels cross-packed across amb_a/amb_b.
    let dc_r = (amb_a >> 1) & 0x1F;
    let dc_g = ((amb_a >> 6) & 0x3) | ((amb_b & 0x7) << 2);
    let dc_b = (amb_b >> 3) & 0x1F;
    let sqn = |v: u8| { let f = v as f32 / 31.0; f * f };
    let (sr, sg, sb) = (sqn(dc_r) * eff, sqn(dc_g) * eff, sqn(dc_b) * eff);

    // 2-bit baked sun-visibility (bit0 = dir hi, bit1 = amb_a lsb) → {0..3}/3, γ².
    let vis_combined = (((amb_a & 1) as u32) << 1) | dir_hi as u32;
    // HMS_VISHIST counts the raw 2-bit population per map. An enclosed interior that
    // decodes mostly "3" (fully sun-visible) is proof the field or its bit layout is wrong.
    VIS_HIST[(vis_combined & 3) as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut vis = vis_combined as f32 / 3.0;
    vis *= vis;

    // Bounce term: indirect VMF lobe response = max(0, dot(N, bounceDir)).
    let ndotd = normal.dot(bounce_dir);
    let ndl = ndotd.clamp(0.0, 1.0);
    // Fold baked visibility into the bounce so shadowed vertices read darker. Ground
    // truth (RE, entry_points.hlsl_include): the engine's sun/bounce term is × baked
    // visibility with NO floor (vis already squared above); a floor would keep
    // fully-occluded vertices at a fraction of the primary lobe so shadows never go dark.
    // The engine does NOT vis-gate the baked dominant lobe — baked sun-visibility (vmf[0].w)
    // gates only the ANALYTICAL sun (entry_points.hlsl:431). The default per-vertex path folds
    // vis into the bounce (a pragmatic occlusion darken); in engine HDR mode use the un-gated lobe
    // response so enclosed floors (vis→0) keep their baked primary lobe instead of going near-black.
    // In engine HDR mode the dominant lobe response comes from the BYTE-EXACT vMF LUT at
    // bandwidth=1 (engine per-vertex vmf[1].w=1, decompress_per_vertex:149) — matching the atlas
    // path's dual_vmf_diffuse — NOT the linear ndl. Default path keeps ndl·vis.
    let bounce = if hdr_out { vmf_diffuse_coeff(ndotd, 1.0) } else { ndl * vis };

    // Compose irradiance: ambient*0.25 (VMF secondary-lobe integral) + primary*bounce.
    // Engine mode divides by π (dual_vmf_diffuse); the default toned path is calibrated without it.
    const AMB_SCALE: f32 = 0.25;
    let pdiv = if hdr_out { 1.0 / std::f32::consts::PI } else { 1.0 };
    // DOMINANT-LIGHT SEPARATION (protomorph spherical_harmonics_fx): the engine SUBTRACTS the
    // dominant (sun) lobe out of the baked term so the lightmap carries INDIRECT/ambient only,
    // then re-adds the sun as a SHARP analytical directional (the terrain/mesh shader's sun_term).
    // The primary bounce `pr*bounce` IS the baked dominant; scale it by (1-dom_sep). dom_sep=0 →
    // full baked (sun double-counted → flat/washed); dom_sep=1 → indirect-only baked, the
    // shader's analytical sun provides all direct light (sharp shadows + contrast, no khaki wash).
    let dk = (1.0 - dom_sep).clamp(0.0, 1.0);
    let ir = [
        (sr * AMB_SCALE + pr * bounce * dk) * pdiv,
        (sg * AMB_SCALE + pg * bounce * dk) * pdiv,
        (sb * AMB_SCALE + pb * bounce * dk) * pdiv,
    ];

    // Tone the irradiance into a lighting multiplier with REAL dynamic range so
    // cliff shadows baked into the PVL actually darken the terrain (a low 0.05 floor
    // lets shadow pockets go dark; a higher floor flattens the baked light into a
    // near-neutral tint with no visible shadows).
    // ENGINE LIGHTING: return the RAW absolute-HDR irradiance (no per-vertex Reinhard)
    // so the PVL path lands in the SAME domain as the atlas path (dual_vmf_diffuse) — the
    // shader/post then does the single tonemap. Default path keeps the toned ~unit tint.
    if hdr_out {
        return [ir[0], ir[1], ir[2], vis];
    }
    let toned = tone_irradiance_rgb(ir);
    // Also surface the baked sun visibility so the shader can gate its
    // analytical sun_term by it (engine multiplies the analytical sun by vmf[0].w).
    // `vis` here is the same 2-bit {0..3}/3 squared value folded into the bounce above.
    [toned[0], toned[1], toned[2], vis]
}

/// One vertex's RAW baked PVL dual-vMF lobe (engine `sub_1406B6940` decode), kept per
/// vertex for the per-object lighting raycast (doc 16 §1.4: a ray that hits a per-vertex-lit
/// instance samples the NEAREST vertex's lobes, bandwidth 1). `pri` = dominant lobe colour (LUT
/// `dir`), `amb` = fill lobe (DC ambient), `vis` = baked sun visibility. Already × hdr_scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct PvlLobe { pub dir: [f32; 3], pub pri: [f32; 3], pub amb: [f32; 3], pub vis: f32 }

/// The raw lobes behind `decode_pvl_tint` (same bit layout / scaling, no irradiance
/// evaluation). Out-of-buffer vertices decode to black with vis=1 (neutral).
pub fn decode_pvl_lobes(bytes: &[u8], vid: usize, pvb_offset: i32, hdr_scale: f32) -> PvlLobe {
    let vid_in_cluster = vid as i64 - pvb_offset as i64;
    if vid_in_cluster < 0 {
        return PvlLobe { dir: [0.0, 0.0, 1.0], pri: [0.0; 3], amb: [0.0; 3], vis: 1.0 };
    }
    let vid_in_cluster = vid_in_cluster as usize;
    let o = ((vid_in_cluster * 3) >> 1) * 4;
    let at = |k: usize| -> u8 { bytes.get(o + k).copied().unwrap_or(0) };
    let (b0, b1, b2, b3, b4, b5, b6, b7) = (at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7));
    let odd = (vid_in_cluster & 1) != 0;
    let (dir_byte, pri_r, pri_g, pri_b, amb_a, amb_b) = if odd { (b2, b5, b6, b7, b3, b4) } else { (b0, b3, b4, b5, b1, b2) };
    let dir = dir_lut((dir_byte & 0x7F) as usize);
    let dir_hi = (dir_byte >> 7) & 1;
    let eff = if hdr_scale.is_finite() && hdr_scale > 0.0 { hdr_scale } else { 1.0 };
    let sq = |b: u8| { let v = b as f32 / 255.0; v * v };
    let dc_r = (amb_a >> 1) & 0x1F;
    let dc_g = ((amb_a >> 6) & 0x3) | ((amb_b & 0x7) << 2);
    let dc_b = (amb_b >> 3) & 0x1F;
    let sqn = |v: u8| { let f = v as f32 / 31.0; f * f };
    let vis_combined = (((amb_a & 1) as u32) << 1) | dir_hi as u32;
    let vis = { let v = vis_combined as f32 / 3.0; v * v };
    PvlLobe {
        dir: dir.into(),
        pri: [sq(pri_r) * eff, sq(pri_g) * eff, sq(pri_b) * eff],
        amb: [sqn(dc_r) * eff, sqn(dc_g) * eff, sqn(dc_b) * eff],
        vis,
    }
}

/// ENGINE LIGHTING (spec doc 07): analytic stand-in for the engine's precomputed
/// `g_sample_vmf_diffuse` LUT — the cosine-weighted irradiance response of a normalized vMF
/// lobe, indexed by (N·dir, bandwidth). bandwidth→0 = a spread/isotropic lobe (flat ~0.5
/// response); bandwidth→~sqrt(3) = a sharp lobe approaching clamped-cosine `saturate(N·dir)`.
/// The engine coord is `(dot(dir,N)*0.5+0.5, bandwidth)` (identity bandwidth map). Calibrated
/// shape; replace with the extracted LUT bitmap for byte-exact match.
/// The BYTE-EXACT engine `g_sample_vmf_diffuse` LUT, extracted from HREK
/// `rasterizer/diffusetable.bitmap` (256×256 A8, box-downsampled to 64×64). Indexed exactly
/// as the engine samples it: x = dot(dir,N)·0.5+0.5, y = bandwidth (convertBandwidth2TextureCoord
/// is identity). Rows are the vMF diffuse convolution: ~isotropic at low bandwidth (spread lobe),
/// clamped-cosine at high bandwidth (sharp lobe).
static VMF_DIFFUSE_LUT: &[u8] = include_bytes!("vmf_diffuse_lut.bin"); // 128x128 A8 (rasterizer\diffusetable)

pub fn vmf_diffuse_coeff(ndotd: f32, bandwidth: f32) -> f32 {
    // The engine's `rasterizer\diffusetable` (128x128 A8, see hms-render lib.rs).
    // Must stay in lockstep with VMF_LUT_W/H there and with vmf_diffuse_lut.bin's size.
    const W: usize = 128;
    const H: usize = 128;
    let x = (ndotd * 0.5 + 0.5).clamp(0.0, 1.0) * (W - 1) as f32;
    let y = bandwidth.clamp(0.0, 1.0) * (H - 1) as f32; // texcoord clamps bandwidth>1 to sharpest row
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(W - 1), (y0 + 1).min(H - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let s = |xi: usize, yi: usize| VMF_DIFFUSE_LUT[yi * W + xi] as f32 / 255.0;
    let top = s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx;
    let bot = s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// ENGINE LIGHTING: CPU port of `dual_vmf_diffuse` (spherical_harmonics.hlsl_include:68):
/// `(coeff_dom·domColor + 0.25·fillColor) / pi`, returning ABSOLUTE linear-HDR irradiance for
/// the given vertex normal. `dir` = dominant direction (normalized), `bandwidth` = pre-normalize
/// length, `dom_color`/`fill_color` = the two lobe colours (already × K·fIntensity by the caller).
pub fn dual_vmf_diffuse(normal: Vec3, dir: Vec3, dom_color: [f32; 3], fill_color: [f32; 3], bandwidth: f32) -> [f32; 3] {
    let coeff_dom = vmf_diffuse_coeff(normal.dot(dir), bandwidth);
    const COEFF_FIL: f32 = 0.25; // engine constant (NOT hemisphere-weighted)
    let inv_pi = 1.0 / std::f32::consts::PI;
    [
        (coeff_dom * dom_color[0] + COEFF_FIL * fill_color[0]) * inv_pi,
        (coeff_dom * dom_color[1] + COEFF_FIL * fill_color[1]) * inv_pi,
        (coeff_dom * dom_color[2] + COEFF_FIL * fill_color[2]) * inv_pi,
    ]
}

/// Engine-model tone of one baked linear-HDR irradiance channel into an ABSOLUTE
/// diffuse-lighting value: `irLinear = raw·K` (exposure lift toward the engine
/// level; per-BSP Brightness cancels), then Reinhard `x/(1+x)` into [0,1). Shared
/// by the terrain/cluster PVL decode AND the decorator (foliage) path so both land
/// in the SAME range. K=33 = the engine's kExposure. Shadow CONTRAST comes from the
/// correct ambient bit-decode + no ambient floor + the real-time cast-shadow darken
/// + the engine toe curve — NOT from a low K (K=8 just made the whole map too dark).
pub const LIGHTMAP_K: f32 = 33.0;
pub fn tone_irradiance(v: f32) -> f32 {
    let x = v * LIGHTMAP_K;
    x / (1.0 + x)
}

/// Luminance-preserving tone: tone the LUMINANCE with the engine Reinhard, then scale
/// all three channels by the SAME factor so chroma/saturation is preserved exactly.
/// The old per-channel `tone_irradiance` pushed each channel independently toward 1.0 at
/// K=33, which collapsed saturated baked colors (Zealot's purple) toward gray — the
/// "muted / no purple / looks gray" bug. Brightness (luminance) maps identically to the
/// per-channel path, so this changes SATURATION only, not overall exposure.
pub fn tone_irradiance_rgb(ir: [f32; 3]) -> [f32; 3] {
    let lum = 0.2126 * ir[0] + 0.7152 * ir[1] + 0.0722 * ir[2];
    if lum <= 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    let x = lum * LIGHTMAP_K;
    let toned = x / (1.0 + x);
    let s = toned / lum;
    [ir[0] * s, ir[1] * s, ir[2] * s]
}
