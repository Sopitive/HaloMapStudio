//! External model import with REAL materials and textures — Wavefront OBJ (+ .mtl) and
//! glTF 2.0 / GLB.
//!
//! The first import feature only ever read POSITION + indices, hard-coded every vertex to
//! `uv: [0,0], color: [1,1,1,1]` and bound no texture, so every imported model drew as a plain
//! white shell (and, for glTF, only the FIRST mesh of the file, with every node transform
//! ignored). This module is the parse half of the fix: one reader per format, both producing the
//! same [`ModelData`] — a list of per-material [`ImportPart`]s carrying positions, normals,
//! texture coordinates and vertex colours, the material list, and the DECODED base-colour images.
//!
//! Everything here treats the file as UNTRUSTED data from the internet:
//!
//!   * no panics and no unbounded allocation — every count is checked against the buffer that
//!     backs it, every index is bounds-checked, and the vertex / index / image budgets below are
//!     hard caps;
//!   * an image that fails to decode, a primitive with no POSITION, a material pointing at a
//!     missing texture, a cyclic node graph and a 32 768² texture all degrade to something
//!     drawable (flat colour / clamped size / skipped part) and are recorded in
//!     [`ModelData::notes`] instead of aborting the import;
//!   * nothing is executed, no path from the file is written to, and a texture `uri` that escapes
//!     the model's own directory is refused.
//!
//! What is deliberately NOT read: normal / metallic-roughness / occlusion / emissive maps,
//! animation, morph targets and glTF skinning (a skinned mesh draws at its bind pose), Draco and
//! KTX2/Basis compressed payloads, and sparse accessors. Each of those leaves a note.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

/// Largest texture edge kept on the GPU. Bigger images are box-downsampled by powers of two
/// until they fit — a 8192² base colour is 256 MiB of RGBA, which no imported placeholder needs
/// and which would stall (or OOM) the upload.
pub const MAX_TEX_DIM: u32 = 2048;
/// Refuse an image whose HEADER claims more than this many pixels before decoding it (a
/// malicious/corrupt PNG header is the cheapest denial of service in the format).
pub const MAX_TEX_PIXELS: u64 = 64 << 20; // 64 Mpx (e.g. 8192×8192)
/// Total decoded RGBA bytes one model may hold.
pub const MAX_TEX_BUDGET: usize = 192 << 20;
/// Hard caps on geometry. A `MeshVertex` is 92 bytes on the GPU, so 4 M vertices is already
/// ~370 MB of vertex buffer — far past anything a placeholder model needs, and the point at
/// which a hostile file stops being a model and starts being an allocation attack. The two real
/// test models are ~690 k verts / 817 k tris (a 26 MB GLB) and ~16 k tris.
pub const MAX_VERTS: usize = 4_000_000;
pub const MAX_INDICES: usize = 16_000_000;
/// Depth limit for the glTF node graph (also the cycle guard).
const MAX_NODE_DEPTH: usize = 64;

/// How a material's alpha is meant to be used.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AlphaMode {
    /// Alpha ignored, the surface is solid.
    #[default]
    Opaque,
    /// Alpha is a coverage mask clipped at `alpha_cutoff`.
    Mask,
    /// Alpha is real transparency.
    Blend,
}

/// Texture address mode (glTF `sampler.wrapS` / `wrapT`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

impl Wrap {
    fn from_gl(v: u64) -> Wrap {
        match v {
            33071 => Wrap::Clamp,
            33648 => Wrap::Mirror,
            _ => Wrap::Repeat, // 10497 GL_REPEAT (and anything unknown)
        }
    }
}

/// A decoded image, straight RGBA8 in the file's own (sRGB-encoded) colour space.
#[derive(Clone)]
pub struct ImportImage {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
    pub name: String,
}

impl std::fmt::Debug for ImportImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImportImage({} {}x{})", self.name, self.w, self.h)
    }
}

/// One material: the base colour (factor × optional texture) plus how its alpha behaves.
/// Only the base-colour half of `pbrMetallicRoughness` / the MTL diffuse is read — see the
/// module note on what is unsupported.
#[derive(Clone, Debug)]
pub struct ImportMaterial {
    pub name: String,
    /// glTF `baseColorFactor` / MTL `Kd` + `d`. LINEAR, so a factor of 0.5 is half the radiance.
    pub base_color: [f32; 4],
    /// Index into [`ModelData::images`], when the material has a base-colour map.
    pub image: Option<usize>,
    /// `KHR_texture_transform` / MTL `map_Kd -s`/`-o`: uv' = offset + scale · rot(uv).
    pub uv_scale: [f32; 2],
    pub uv_offset: [f32; 2],
    pub uv_rotation: f32,
    pub wrap: [Wrap; 2],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for ImportMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            image: None,
            uv_scale: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
            uv_rotation: 0.0,
            wrap: [Wrap::Repeat; 2],
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

/// One vertex, in the FILE's coordinate system (glTF/OBJ Y-up). The Halo Z-up conversion
/// happens at upload, together with the renderer's vertex packing.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// One drawable piece: the triangles that share ONE material. A multi-material model yields one
/// part per (material, source primitive) so it is never flattened to a single look.
#[derive(Clone, Debug)]
pub struct ImportPart {
    pub name: String,
    pub verts: Vec<ImportVertex>,
    pub indices: Vec<u32>,
    pub material: Option<usize>,
}

/// A parsed model file.
#[derive(Clone, Debug, Default)]
pub struct ModelData {
    pub parts: Vec<ImportPart>,
    pub materials: Vec<ImportMaterial>,
    pub images: Vec<ImportImage>,
    /// Everything that was clamped, approximated or skipped — surfaced in the status line and
    /// the log so a model that looks wrong says why.
    pub notes: Vec<String>,
    pub format: &'static str,
}

impl ModelData {
    pub fn vert_count(&self) -> usize {
        self.parts.iter().map(|p| p.verts.len()).sum()
    }
    pub fn tri_count(&self) -> usize {
        self.parts.iter().map(|p| p.indices.len() / 3).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| p.indices.len() < 3)
    }
    /// How many materials actually bind a base-colour image.
    pub fn textured_material_count(&self) -> usize {
        self.materials.iter().filter(|m| m.image.is_some()).count()
    }
    /// Do any parts carry real texture coordinates? (A model with materials but no UVs can only
    /// show its flat base colour.)
    pub fn has_uvs(&self) -> bool {
        self.parts.iter().any(|p| p.verts.iter().any(|v| v.uv != [0.0, 0.0]))
    }
    /// The import browser's probe line tail: "3 materials, 2 textures (1024x1024)".
    pub fn material_summary(&self) -> String {
        let mut s = format!(
            "{} material{}, {} texture{}",
            self.materials.len(),
            if self.materials.len() == 1 { "" } else { "s" },
            self.images.len(),
            if self.images.len() == 1 { "" } else { "s" },
        );
        if let Some(big) = self.images.iter().max_by_key(|i| (i.w as u64) * (i.h as u64)) {
            s += &format!(" (up to {}x{})", big.w, big.h);
        }
        if self.images.is_empty() && !self.materials.is_empty() {
            s += if self.has_uvs() { " — flat colours only" } else { " — flat colours, no UVs" };
        } else if !self.images.is_empty() && !self.has_uvs() {
            s += " — NO UVs, textures cannot show";
        }
        s
    }
    fn note(&mut self, s: impl Into<String>) {
        let s = s.into();
        if self.notes.len() < 32 && !self.notes.contains(&s) {
            self.notes.push(s);
        }
    }
}

/// Read and parse one model file. `Err` is a message fit for the status line.
pub fn load(path: &Path) -> Result<ModelData, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let base = path.parent().map(|p| p.to_path_buf());
    match ext.as_str() {
        "glb" | "gltf" | "vrm" => {
            let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
            parse_gltf(&bytes, base.as_deref())
        }
        _ => {
            let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
            // An OBJ is text; a mis-named GLB still starts with the magic, so sniff first.
            if bytes.len() >= 4 && &bytes[0..4] == b"glTF" {
                return parse_gltf(&bytes, base.as_deref());
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            Ok(parse_obj(&text, base.as_deref()))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// images
// ---------------------------------------------------------------------------------------------

/// Decode a PNG or JPEG byte blob to RGBA8, clamped to [`MAX_TEX_DIM`].
/// `None` for anything else (KTX2/Basis, WebP, a truncated blob) — the caller notes it and falls
/// back to the material's flat colour.
pub fn decode_image(bytes: &[u8], name: &str) -> Option<ImportImage> {
    let img = if bytes.len() > 8 && bytes[0..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        decode_png(bytes)
    } else if bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        decode_jpeg(bytes)
    } else {
        None
    }?;
    let (w, h, mut rgba) = img;
    if w == 0 || h == 0 || rgba.len() < (w as usize) * (h as usize) * 4 {
        return None;
    }
    let (mut w, mut h) = (w, h);
    while w > MAX_TEX_DIM || h > MAX_TEX_DIM {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        rgba = box_halve(&rgba, w, h, nw, nh);
        w = nw;
        h = nh;
    }
    Some(ImportImage { w, h, rgba, name: name.to_string() })
}

fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut limits = png::Limits::default();
    limits.bytes = (MAX_TEX_PIXELS as usize).saturating_mul(4);
    let mut dec = png::Decoder::new_with_limits(std::io::Cursor::new(bytes), limits);
    // EXPAND turns palette/low-bit grayscale into 8-bit channels and materialises tRNS as a real
    // alpha channel; STRIP_16 drops 16-bit samples to 8. Everything then lands in one of the four
    // shapes handled below.
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let info = reader.info();
    let (w, h) = (info.width, info.height);
    if (w as u64) * (h as u64) > MAX_TEX_PIXELS {
        return None;
    }
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let out = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (out.width, out.height);
    let n = (w as usize).checked_mul(h as usize)?;
    let src = &buf[..out.buffer_size().min(buf.len())];
    let mut rgba = vec![255u8; n.checked_mul(4)?];
    match out.color_type {
        png::ColorType::Rgba => {
            let take = src.len().min(rgba.len());
            rgba[..take].copy_from_slice(&src[..take]);
        }
        png::ColorType::Rgb => {
            for (i, c) in src.chunks_exact(3).take(n).enumerate() {
                rgba[i * 4] = c[0];
                rgba[i * 4 + 1] = c[1];
                rgba[i * 4 + 2] = c[2];
            }
        }
        png::ColorType::Grayscale => {
            for (i, g) in src.iter().take(n).enumerate() {
                rgba[i * 4] = *g;
                rgba[i * 4 + 1] = *g;
                rgba[i * 4 + 2] = *g;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for (i, c) in src.chunks_exact(2).take(n).enumerate() {
                rgba[i * 4] = c[0];
                rgba[i * 4 + 1] = c[0];
                rgba[i * 4 + 2] = c[0];
                rgba[i * 4 + 3] = c[1];
            }
        }
        // EXPAND removes Indexed; anything else is not a thing `png` produces.
        png::ColorType::Indexed => return None,
    }
    Some((w, h, rgba))
}

fn decode_jpeg(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let opts = DecoderOptions::default()
        .set_max_width(1 << 15)
        .set_max_height(1 << 15)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    dec.decode_headers().ok()?;
    let info = dec.info()?;
    let (w, h) = (info.width as u32, info.height as u32);
    if w == 0 || h == 0 || (w as u64) * (h as u64) > MAX_TEX_PIXELS {
        return None;
    }
    let rgb = dec.decode().ok()?;
    let n = (w as usize).checked_mul(h as usize)?;
    let mut rgba = vec![255u8; n.checked_mul(4)?];
    for (i, c) in rgb.chunks_exact(3).take(n).enumerate() {
        rgba[i * 4] = c[0];
        rgba[i * 4 + 1] = c[1];
        rgba[i * 4 + 2] = c[2];
    }
    Some((w, h, rgba))
}

/// 2×2 box downsample (straight on the stored bytes — mip-chain quality is not the point here,
/// staying inside the texture budget is).
fn box_halve(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (nw as usize) * (nh as usize) * 4];
    for y in 0..nh {
        for x in 0..nw {
            let (x0, y0) = ((x * 2).min(w - 1), (y * 2).min(h - 1));
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            for c in 0..4 {
                let p = |px: u32, py: u32| src[((py as usize) * (w as usize) + px as usize) * 4 + c] as u32;
                let v = (p(x0, y0) + p(x1, y0) + p(x0, y1) + p(x1, y1) + 2) / 4;
                out[((y as usize) * (nw as usize) + x as usize) * 4 + c] = v as u8;
            }
        }
    }
    out
}

/// Resolve a texture / buffer `uri` against the model's own directory.
///
/// Only plain, DOWNWARD path components are followed. An absolute path, a Windows drive letter
/// and any `..` component are refused outright; the lookup then falls back to the bare file name
/// beside the model (what a stale exporter path usually means). The result is therefore always
/// inside the model's own folder, so a crafted `uri` in a downloaded model can never read
/// anything else on the machine.
fn resolve_sibling(base: Option<&Path>, uri: &str) -> Option<PathBuf> {
    let base = base?;
    // glTF uris are percent-encoded; MTL paths are frequently Windows-style.
    let decoded = percent_decode(uri).replace('\\', "/");
    let rel = Path::new(&decoded);
    let downward = |rel: &Path| -> Option<PathBuf> {
        if rel.is_absolute() || decoded.contains(':') {
            return None;
        }
        let mut out = base.to_path_buf();
        for c in rel.components() {
            match c {
                Component::Normal(s) => out.push(s),
                Component::CurDir => {}
                _ => return None, // `..`, a root, a prefix
            }
        }
        out.starts_with(base).then_some(out)
    };
    if let Some(p) = downward(rel) {
        if p.is_file() {
            return Some(p);
        }
    }
    let p = base.join(rel.file_name()?);
    (p.starts_with(base) && p.is_file()).then_some(p)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hi = (b[i + 1] as char).to_digit(16);
            let lo = (b[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decode a `data:...;base64,...` URI payload. Also accepts a non-base64 data URI (percent-encoded).
fn data_uri_bytes(uri: &str) -> Option<Vec<u8>> {
    let rest = uri.strip_prefix("data:")?;
    let comma = rest.find(',')?;
    let (meta, payload) = (&rest[..comma], &rest[comma + 1..]);
    if meta.ends_with(";base64") || meta.ends_with("base64") {
        base64_decode(payload)
    } else {
        Some(percent_decode(payload).into_bytes())
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        })
    };
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c)?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// glTF 2.0 / GLB
// ---------------------------------------------------------------------------------------------

/// A resolved accessor: the bytes that back it plus the layout needed to walk them.
struct Accessor<'a> {
    data: &'a [u8],
    /// byte offset of element 0 inside `data`
    offset: usize,
    stride: usize,
    count: usize,
    comp: u64,
    comps: usize,
    normalized: bool,
}

impl<'a> Accessor<'a> {
    fn comp_size(comp: u64) -> usize {
        match comp {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4, // 5125 u32 / 5126 f32
        }
    }
    /// Element `i`, channel `c`, as f32 (normalized integers mapped to [0,1] / [-1,1]).
    fn get(&self, i: usize, c: usize) -> f32 {
        if i >= self.count || c >= self.comps {
            return 0.0;
        }
        let cs = Self::comp_size(self.comp);
        let o = self.offset + i * self.stride + c * cs;
        let d = self.data;
        if o + cs > d.len() {
            return 0.0;
        }
        match self.comp {
            5120 => {
                let v = d[o] as i8 as f32;
                if self.normalized { (v / 127.0).max(-1.0) } else { v }
            }
            5121 => {
                let v = d[o] as f32;
                if self.normalized { v / 255.0 } else { v }
            }
            5122 => {
                let v = i16::from_le_bytes([d[o], d[o + 1]]) as f32;
                if self.normalized { (v / 32767.0).max(-1.0) } else { v }
            }
            5123 => {
                let v = u16::from_le_bytes([d[o], d[o + 1]]) as f32;
                if self.normalized { v / 65535.0 } else { v }
            }
            5125 => u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as f32,
            _ => f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]),
        }
    }
    /// Element `i` as an index (integer componentTypes only).
    fn get_index(&self, i: usize) -> usize {
        if i >= self.count {
            return 0;
        }
        let cs = Self::comp_size(self.comp);
        let o = self.offset + i * self.stride;
        let d = self.data;
        if o + cs > d.len() {
            return 0;
        }
        match self.comp {
            5121 | 5120 => d[o] as usize,
            5123 | 5122 => u16::from_le_bytes([d[o], d[o + 1]]) as usize,
            _ => u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize,
        }
    }
}

fn comps_of(ty: &str) -> usize {
    match ty {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => 0,
    }
}

/// Resolve accessor `idx` to the bytes that back it plus its layout.
///
/// `count` is CLAMPED to what the bufferView actually holds, so a file that lies about its
/// element count yields fewer vertices instead of reading past the buffer. Sparse accessors are
/// refused (`None`) rather than silently read as dense.
fn resolve_accessor<'a>(
    accs: &[serde_json::Value],
    views: &[serde_json::Value],
    bufs: &'a [Vec<u8>],
    idx: usize,
) -> Option<Accessor<'a>> {
    let a = accs.get(idx)?;
    if a.get("sparse").is_some() {
        return None;
    }
    let count = a.get("count").and_then(|v| v.as_u64())? as usize;
    let comp = a.get("componentType").and_then(|v| v.as_u64())?;
    let comps = comps_of(a.get("type").and_then(|v| v.as_str()).unwrap_or(""));
    if comps == 0 || count == 0 {
        return None;
    }
    let normalized = a.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false);
    let acc_off = a.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let bv_idx = a.get("bufferView").and_then(|v| v.as_u64())? as usize;
    let bv = views.get(bv_idx)?;
    let buf_idx = bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let data: &'a [u8] = bufs.get(buf_idx)?.as_slice();
    let bv_off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let bv_len = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let tight = Accessor::comp_size(comp) * comps;
    let stride = bv.get("byteStride").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let stride = if stride == 0 { tight } else { stride };
    if bv_off > data.len() || stride == 0 {
        return None;
    }
    let end = bv_off.saturating_add(bv_len).min(data.len());
    let view = &data[bv_off..end];
    let avail = view.len().saturating_sub(acc_off);
    let fit = if avail >= tight { (avail - tight) / stride + 1 } else { 0 };
    Some(Accessor { data: view, offset: acc_off, stride, count: count.min(fit), comp, comps, normalized })
}

/// Parse a .glb (binary container) or a .gltf (JSON, with `data:` or sibling-file buffers).
pub fn parse_gltf(bytes: &[u8], base: Option<&Path>) -> Result<ModelData, String> {
    let mut out = ModelData { format: "GLB", ..Default::default() };
    let (json, glb_bin): (serde_json::Value, Option<&[u8]>) = if bytes.len() >= 12 && &bytes[0..4] == b"glTF" {
        let u32at = |o: usize| -> u32 {
            if o + 4 > bytes.len() { 0 } else { u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) }
        };
        let mut off = 12usize;
        let mut js: Option<serde_json::Value> = None;
        let mut bin: Option<&[u8]> = None;
        // Chunk walk, bounded by the real file length (a lying chunk length must not loop).
        while off + 8 <= bytes.len() {
            let len = u32at(off) as usize;
            let ctype = u32at(off + 4);
            let start = off + 8;
            let end = start.saturating_add(len).min(bytes.len());
            if start > bytes.len() {
                break;
            }
            match ctype {
                0x4E4F_534A => js = serde_json::from_slice(&bytes[start..end]).ok(), // 'JSON'
                0x004E_4942 => bin = Some(&bytes[start..end]),                       // 'BIN\0'
                _ => {}
            }
            let next = start.saturating_add(len);
            if next <= off {
                break;
            }
            off = next;
        }
        (js.ok_or_else(|| "GLB: the JSON chunk is missing or malformed".to_string())?, bin)
    } else {
        out.format = "glTF";
        (serde_json::from_slice(bytes).map_err(|e| format!("glTF: not a JSON document ({e})"))?, None)
    };

    let arr = |k: &str| -> Vec<serde_json::Value> {
        json.get(k).and_then(|v| v.as_array()).cloned().unwrap_or_default()
    };
    let gl_buffers = arr("buffers");
    let gl_views = arr("bufferViews");
    let gl_accessors = arr("accessors");
    let gl_meshes = arr("meshes");
    let gl_nodes = arr("nodes");
    let gl_materials = arr("materials");
    let gl_textures = arr("textures");
    let gl_images = arr("images");
    let gl_samplers = arr("samplers");

    for ext in json.get("extensionsRequired").and_then(|v| v.as_array()).map(|a| a.as_slice()).unwrap_or(&[]) {
        if let Some(name) = ext.as_str() {
            match name {
                "KHR_texture_transform" | "KHR_materials_unlit" | "KHR_mesh_quantization" => {}
                other => out.note(format!("required glTF extension '{other}' is not supported — the model may look wrong")),
            }
        }
    }

    // --- buffers -----------------------------------------------------------------------------
    // Buffer 0 of a GLB is the BIN chunk; a `uri` buffer is a data: blob or a sibling file.
    let mut buf_data: Vec<Vec<u8>> = Vec::with_capacity(gl_buffers.len());
    for (i, b) in gl_buffers.iter().enumerate() {
        match b.get("uri").and_then(|v| v.as_str()) {
            None => {
                if i == 0 {
                    buf_data.push(glb_bin.map(|s| s.to_vec()).unwrap_or_default());
                } else {
                    out.note(format!("buffer {i} has no uri and is not the GLB chunk — skipped"));
                    buf_data.push(Vec::new());
                }
            }
            Some(u) if u.starts_with("data:") => buf_data.push(data_uri_bytes(u).unwrap_or_default()),
            Some(u) => match resolve_sibling(base, u).and_then(|p| std::fs::read(p).ok()) {
                Some(d) => buf_data.push(d),
                None => {
                    out.note(format!("buffer file '{u}' could not be read next to the model — its geometry is missing"));
                    buf_data.push(Vec::new());
                }
            },
        }
    }
    if buf_data.is_empty() {
        buf_data.push(glb_bin.map(|s| s.to_vec()).unwrap_or_default());
    }

    // --- accessors ---------------------------------------------------------------------------
    let acc = |idx: usize| resolve_accessor(&gl_accessors, &gl_views, &buf_data, idx);

    // --- images (lazily, only the ones a baseColorTexture actually points at) -----------------
    // texture index -> (image index, wrapS, wrapT)
    let tex_info = |ti: usize| -> Option<(usize, Wrap, Wrap)> {
        let t = gl_textures.get(ti)?;
        let src = t
            .get("source")
            .and_then(|v| v.as_u64())
            // KHR_texture_basisu puts the KTX2 image here; we cannot decode it, but reading the
            // index lets the caller report the material as "textured but undecodable".
            .or_else(|| {
                t.get("extensions")
                    .and_then(|e| e.get("KHR_texture_basisu"))
                    .and_then(|e| e.get("source"))
                    .and_then(|v| v.as_u64())
            })? as usize;
        let (mut ws, mut wt) = (Wrap::Repeat, Wrap::Repeat);
        if let Some(s) = t.get("sampler").and_then(|v| v.as_u64()).and_then(|i| gl_samplers.get(i as usize)) {
            ws = Wrap::from_gl(s.get("wrapS").and_then(|v| v.as_u64()).unwrap_or(10497));
            wt = Wrap::from_gl(s.get("wrapT").and_then(|v| v.as_u64()).unwrap_or(10497));
        }
        Some((src, ws, wt))
    };
    let mut image_cache: HashMap<usize, Option<usize>> = HashMap::new();
    let mut tex_budget = 0usize;
    let mut load_image = |gi: usize, out: &mut ModelData, tex_budget: &mut usize| -> Option<usize> {
        if let Some(c) = image_cache.get(&gi) {
            return *c;
        }
        let res = (|| -> Option<usize> {
            let im = gl_images.get(gi)?;
            let name = im
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("image{gi}"));
            let mime = im.get("mimeType").and_then(|v| v.as_str()).unwrap_or("");
            let bytes: Vec<u8> = if let Some(bv) = im.get("bufferView").and_then(|v| v.as_u64()) {
                let bv = gl_views.get(bv as usize)?;
                let bi = bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let d = buf_data.get(bi)?;
                let o = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let l = bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if o >= d.len() {
                    return None;
                }
                d[o..o.saturating_add(l).min(d.len())].to_vec()
            } else {
                let uri = im.get("uri").and_then(|v| v.as_str())?;
                if uri.starts_with("data:") {
                    data_uri_bytes(uri)?
                } else {
                    let p = resolve_sibling(base, uri);
                    match p.and_then(|p| std::fs::read(p).ok()) {
                        Some(d) => d,
                        None => {
                            out.note(format!("texture file '{uri}' was not found next to the model"));
                            return None;
                        }
                    }
                }
            };
            let Some(img) = decode_image(&bytes, &name) else {
                out.note(format!(
                    "could not decode texture '{name}'{} — that material falls back to its flat colour",
                    if mime.is_empty() { String::new() } else { format!(" ({mime})") }
                ));
                return None;
            };
            let cost = img.rgba.len();
            if *tex_budget + cost > MAX_TEX_BUDGET {
                out.note(format!("texture budget reached — '{name}' and any later image were dropped"));
                return None;
            }
            *tex_budget += cost;
            if img.w == MAX_TEX_DIM || img.h == MAX_TEX_DIM {
                out.note(format!("texture '{name}' was downsampled to {}x{}", img.w, img.h));
            }
            out.images.push(img);
            Some(out.images.len() - 1)
        })();
        image_cache.insert(gi, res);
        res
    };

    // --- materials ---------------------------------------------------------------------------
    // base-colour texCoord set per material (the shader carries ONE UV channel, so a material
    // whose base colour reads TEXCOORD_1 makes the primitive read TEXCOORD_1 into that channel).
    let mut mat_uvset: Vec<usize> = Vec::with_capacity(gl_materials.len());
    for (mi, m) in gl_materials.iter().enumerate() {
        let mut mat = ImportMaterial {
            name: m.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            ..Default::default()
        };
        if mat.name.is_empty() {
            mat.name = format!("material{mi}");
        }
        mat.double_sided = m.get("doubleSided").and_then(|v| v.as_bool()).unwrap_or(false);
        mat.alpha_mode = match m.get("alphaMode").and_then(|v| v.as_str()).unwrap_or("OPAQUE") {
            "MASK" => AlphaMode::Mask,
            "BLEND" => AlphaMode::Blend,
            _ => AlphaMode::Opaque,
        };
        mat.alpha_cutoff = m
            .get("alphaCutoff")
            .and_then(|v| v.as_f64())
            .map(|v| (v as f32).clamp(0.001, 0.999))
            .unwrap_or(0.5);
        let mut uvset = 0usize;
        if let Some(pbr) = m.get("pbrMetallicRoughness") {
            if let Some(f) = pbr.get("baseColorFactor").and_then(|v| v.as_array()) {
                for (k, v) in f.iter().take(4).enumerate() {
                    mat.base_color[k] = v.as_f64().unwrap_or(1.0).clamp(0.0, 1.0) as f32;
                }
            }
            if let Some(bct) = pbr.get("baseColorTexture") {
                uvset = bct.get("texCoord").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(ti) = bct.get("index").and_then(|v| v.as_u64()) {
                    match tex_info(ti as usize) {
                        Some((gi, ws, wt)) => {
                            mat.wrap = [ws, wt];
                            mat.image = load_image(gi, &mut out, &mut tex_budget);
                        }
                        None => out.note(format!("material '{}' points at a missing texture", mat.name)),
                    }
                }
                if let Some(tt) = bct.get("extensions").and_then(|e| e.get("KHR_texture_transform")) {
                    if let Some(s) = tt.get("scale").and_then(|v| v.as_array()) {
                        for (k, v) in s.iter().take(2).enumerate() {
                            mat.uv_scale[k] = v.as_f64().unwrap_or(1.0) as f32;
                        }
                    }
                    if let Some(o) = tt.get("offset").and_then(|v| v.as_array()) {
                        for (k, v) in o.iter().take(2).enumerate() {
                            mat.uv_offset[k] = v.as_f64().unwrap_or(0.0) as f32;
                        }
                    }
                    mat.uv_rotation = tt.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    if let Some(tc) = tt.get("texCoord").and_then(|v| v.as_u64()) {
                        uvset = tc as usize;
                    }
                }
            }
        }
        if m.get("normalTexture").is_some() || m.get("occlusionTexture").is_some() || m.get("emissiveTexture").is_some() {
            out.note("normal / occlusion / emissive maps are not imported (base colour only)");
        }
        mat_uvset.push(uvset);
        out.materials.push(mat);
    }

    // --- node graph --------------------------------------------------------------------------
    // Flatten scenes[scene].nodes (default scene 0, else every node) into (node, world matrix).
    let roots: Vec<usize> = json
        .get("scenes")
        .and_then(|s| s.as_array())
        .and_then(|s| {
            let si = json.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            s.get(si).or_else(|| s.first())
        })
        .and_then(|sc| sc.get("nodes"))
        .and_then(|n| n.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
        .unwrap_or_else(|| (0..gl_nodes.len()).collect());

    let node_local = |n: &serde_json::Value| -> glam::Mat4 {
        if let Some(m) = n.get("matrix").and_then(|v| v.as_array()) {
            if m.len() >= 16 {
                let mut c = [0.0f32; 16];
                for (k, v) in m.iter().take(16).enumerate() {
                    c[k] = v.as_f64().unwrap_or(0.0) as f32;
                }
                let m = glam::Mat4::from_cols_array(&c);
                return if m.determinant().is_finite() { m } else { glam::Mat4::IDENTITY };
            }
        }
        let t = n
            .get("translation")
            .and_then(|v| v.as_array())
            .map(|a| {
                glam::Vec3::new(
                    a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                )
            })
            .unwrap_or(glam::Vec3::ZERO);
        let r = n
            .get("rotation")
            .and_then(|v| v.as_array())
            .map(|a| {
                let q = glam::Quat::from_xyzw(
                    a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    a.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                );
                if q.length_squared() > 1e-8 { q.normalize() } else { glam::Quat::IDENTITY }
            })
            .unwrap_or(glam::Quat::IDENTITY);
        let s = n
            .get("scale")
            .and_then(|v| v.as_array())
            .map(|a| {
                glam::Vec3::new(
                    a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    a.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    a.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                )
            })
            .unwrap_or(glam::Vec3::ONE);
        glam::Mat4::from_scale_rotation_translation(s, r, t)
    };

    // Iterative DFS with a depth cap; `seen` stops a cyclic/diamond graph from exploding.
    let mut flat: Vec<(usize, glam::Mat4)> = Vec::new();
    let mut stack: Vec<(usize, glam::Mat4, usize)> = roots.iter().rev().map(|&r| (r, glam::Mat4::IDENTITY, 0usize)).collect();
    let mut visits = 0usize;
    let mut skinned = false;
    while let Some((ni, parent, depth)) = stack.pop() {
        visits += 1;
        if visits > 200_000 {
            out.note("node graph is enormous or cyclic — the walk was cut short");
            break;
        }
        let Some(n) = gl_nodes.get(ni) else { continue };
        let world = parent * node_local(n);
        if n.get("skin").is_some() {
            skinned = true;
        }
        if n.get("mesh").is_some() {
            flat.push((ni, world));
        }
        if depth + 1 < MAX_NODE_DEPTH {
            if let Some(ch) = n.get("children").and_then(|v| v.as_array()) {
                for c in ch.iter().rev().filter_map(|v| v.as_u64()) {
                    stack.push((c as usize, world, depth + 1));
                }
            }
        } else {
            out.note("node graph deeper than 64 levels — the deepest branches were dropped");
        }
    }
    if skinned {
        out.note("glTF skinning is not imported — the mesh draws at its bind pose");
    }
    if json.get("animations").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty()) {
        out.note("glTF animation is not imported — the model is static");
    }

    // --- primitives -------------------------------------------------------------------------
    let mut total_v = 0usize;
    let mut total_i = 0usize;
    let mut capped = false;
    for (ni, world) in flat {
        let Some(node) = gl_nodes.get(ni) else { continue };
        let mi = node.get("mesh").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let Some(mesh) = gl_meshes.get(mi) else { continue };
        let mesh_name = mesh.get("name").and_then(|v| v.as_str()).unwrap_or("mesh").to_string();
        let Some(prims) = mesh.get("primitives").and_then(|v| v.as_array()) else { continue };
        // Normals need the inverse-transpose of the linear part (a non-uniform node scale
        // otherwise shears them); for a rigid node this is the rotation itself.
        let lin = glam::Mat3::from_mat4(world);
        let nmat = if lin.determinant().abs() > 1e-20 { lin.inverse().transpose() } else { glam::Mat3::IDENTITY };
        let flip = world.determinant() < 0.0; // a mirroring node reverses triangle winding
        for (pi, prim) in prims.iter().enumerate() {
            if capped {
                break;
            }
            let mode = prim.get("mode").and_then(|v| v.as_u64()).unwrap_or(4);
            if !matches!(mode, 4 | 5 | 6) {
                out.note(format!("primitive mode {mode} (points/lines) is not drawable — skipped"));
                continue;
            }
            let attrs = prim.get("attributes");
            if prim.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")).is_some() {
                out.note("Draco-compressed geometry is not supported — that primitive was skipped");
                continue;
            }
            let Some(pos_i) = attrs.and_then(|a| a.get("POSITION")).and_then(|v| v.as_u64()) else { continue };
            let Some(pos) = acc(pos_i as usize) else {
                out.note("a primitive's POSITION accessor is sparse or unreadable — skipped");
                continue;
            };
            let nrm = attrs.and_then(|a| a.get("NORMAL")).and_then(|v| v.as_u64()).and_then(|i| acc(i as usize));
            let material = prim.get("material").and_then(|v| v.as_u64()).map(|v| v as usize);
            let uvset = material.and_then(|m| mat_uvset.get(m)).copied().unwrap_or(0);
            let uv = attrs
                .and_then(|a| a.get(&format!("TEXCOORD_{uvset}")))
                .or_else(|| attrs.and_then(|a| a.get("TEXCOORD_0")))
                .and_then(|v| v.as_u64())
                .and_then(|i| acc(i as usize));
            let col = attrs.and_then(|a| a.get("COLOR_0")).and_then(|v| v.as_u64()).and_then(|i| acc(i as usize));

            let n_v = pos.count;
            if n_v == 0 {
                continue;
            }
            if total_v + n_v > MAX_VERTS {
                out.note(format!("vertex cap ({MAX_VERTS}) reached — the rest of the model was dropped"));
                capped = true;
                break;
            }
            let mat = material.and_then(|m| out.materials.get(m)).cloned().unwrap_or_default();
            let (us, uo, ur) = (mat.uv_scale, mat.uv_offset, mat.uv_rotation);
            let (sr, cr) = (ur.sin(), ur.cos());
            let mut verts: Vec<ImportVertex> = Vec::with_capacity(n_v);
            for i in 0..n_v {
                let p = world.transform_point3(glam::Vec3::new(pos.get(i, 0), pos.get(i, 1), pos.get(i, 2)));
                let nv = match &nrm {
                    Some(a) if a.count > i => {
                        let v = nmat * glam::Vec3::new(a.get(i, 0), a.get(i, 1), a.get(i, 2));
                        if v.length_squared() > 1e-12 { v.normalize() } else { glam::Vec3::Y }
                    }
                    _ => glam::Vec3::ZERO, // filled in from the faces below
                };
                // KHR_texture_transform, baked into the UV (equivalent to the sampler-side
                // transform, and it keeps the renderer's single shared sampler).
                let mut t = match &uv {
                    Some(a) => [a.get(i, 0), a.get(i, 1)],
                    None => [0.0, 0.0],
                };
                if ur != 0.0 {
                    t = [t[0] * cr + t[1] * sr, -t[0] * sr + t[1] * cr];
                }
                t = [t[0] * us[0] + uo[0], t[1] * us[1] + uo[1]];
                let c = match &col {
                    Some(a) => [
                        a.get(i, 0),
                        a.get(i, 1),
                        a.get(i, 2),
                        if a.comps >= 4 { a.get(i, 3) } else { 1.0 },
                    ],
                    None => [1.0; 4],
                };
                let p = if p.is_finite() { p } else { glam::Vec3::ZERO };
                verts.push(ImportVertex {
                    pos: p.to_array(),
                    normal: nv.to_array(),
                    uv: [if t[0].is_finite() { t[0] } else { 0.0 }, if t[1].is_finite() { t[1] } else { 0.0 }],
                    color: c,
                });
            }

            // index list (or sequential when the primitive is non-indexed)
            let idx: Vec<usize> = match prim.get("indices").and_then(|v| v.as_u64()).and_then(|i| acc(i as usize)) {
                Some(a) => (0..a.count).map(|i| a.get_index(i)).collect(),
                None => (0..n_v).collect(),
            };
            let mut indices: Vec<u32> = Vec::with_capacity(idx.len());
            let push = |indices: &mut Vec<u32>, a: usize, b: usize, c: usize| {
                if a < n_v && b < n_v && c < n_v && a != b && b != c && a != c {
                    if flip {
                        indices.extend_from_slice(&[a as u32, c as u32, b as u32]);
                    } else {
                        indices.extend_from_slice(&[a as u32, b as u32, c as u32]);
                    }
                }
            };
            match mode {
                5 => {
                    for i in 0..idx.len().saturating_sub(2) {
                        if i % 2 == 0 {
                            push(&mut indices, idx[i], idx[i + 1], idx[i + 2]);
                        } else {
                            push(&mut indices, idx[i + 1], idx[i], idx[i + 2]);
                        }
                    }
                }
                6 => {
                    for i in 1..idx.len().saturating_sub(1) {
                        push(&mut indices, idx[0], idx[i], idx[i + 1]);
                    }
                }
                _ => {
                    for c in idx.chunks_exact(3) {
                        push(&mut indices, c[0], c[1], c[2]);
                    }
                }
            }
            if indices.len() < 3 {
                continue;
            }
            if total_i + indices.len() > MAX_INDICES {
                out.note("triangle cap reached — the rest of the model was dropped");
                capped = true;
                break;
            }
            if nrm.is_none() {
                fill_face_normals(&mut verts, &indices);
            }
            total_v += verts.len();
            total_i += indices.len();
            out.parts.push(ImportPart {
                name: format!("{mesh_name}[{pi}]"),
                verts,
                indices,
                material,
            });
        }
    }

    if out.parts.is_empty() {
        return Err("glTF: no drawable triangles (no mesh node, or every primitive was unreadable)".into());
    }
    Ok(out)
}

/// Area-weighted vertex normals from the faces, for a primitive that shipped no NORMAL.
fn fill_face_normals(verts: &mut [ImportVertex], indices: &[u32]) {
    let mut acc = vec![glam::Vec3::ZERO; verts.len()];
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (pa, pb, pc) = (
            glam::Vec3::from(verts[a].pos),
            glam::Vec3::from(verts[b].pos),
            glam::Vec3::from(verts[c].pos),
        );
        let fnorm = (pb - pa).cross(pc - pa);
        if fnorm.is_finite() {
            acc[a] += fnorm;
            acc[b] += fnorm;
            acc[c] += fnorm;
        }
    }
    for (v, n) in verts.iter_mut().zip(acc) {
        v.normal = if n.length_squared() > 1e-16 { n.normalize().to_array() } else { [0.0, 1.0, 0.0] };
    }
}

// ---------------------------------------------------------------------------------------------
// Wavefront OBJ + MTL
// ---------------------------------------------------------------------------------------------

/// Parse an OBJ, loading every `mtllib` next to it. Faces are grouped by `usemtl`, so a
/// multi-material OBJ yields one part per material. An OBJ with no `usemtl` / no .mtl yields
/// one untextured part with the default white material (which still shades correctly).
pub fn parse_obj(text: &str, base: Option<&Path>) -> ModelData {
    let mut out = ModelData { format: "OBJ", ..Default::default() };
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut nrm: Vec<[f32; 3]> = Vec::new();
    // material name -> index into out.materials
    let mut mat_index: HashMap<String, usize> = HashMap::new();
    // Deferred base-colour image paths, so an image is decoded once even when several
    // materials share it.
    let mut img_cache: HashMap<PathBuf, Option<usize>> = HashMap::new();
    let mut tex_budget = 0usize;

    // One builder per material (plus the no-material bucket at key None).
    struct Build {
        verts: Vec<ImportVertex>,
        indices: Vec<u32>,
        map: HashMap<(i64, i64, i64), u32>,
        had_normals: bool,
    }
    let mut builds: HashMap<Option<usize>, Build> = HashMap::new();
    let mut cur: Option<usize> = None;
    let mut mtl_seen = false;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        match key {
            "v" => {
                let f: Vec<f32> = it.filter_map(|t| t.parse().ok()).collect();
                if f.len() >= 3 {
                    pos.push([f[0], f[1], f[2]]);
                }
            }
            "vt" => {
                let f: Vec<f32> = it.filter_map(|t| t.parse().ok()).collect();
                // OBJ v runs bottom-up; glTF/our sampler run top-down.
                uvs.push([*f.first().unwrap_or(&0.0), 1.0 - *f.get(1).unwrap_or(&0.0)]);
            }
            "vn" => {
                let f: Vec<f32> = it.filter_map(|t| t.parse().ok()).collect();
                if f.len() >= 3 {
                    nrm.push([f[0], f[1], f[2]]);
                }
            }
            "mtllib" => {
                mtl_seen = true;
                // The rest of the line is the name (it may contain spaces); also accept several
                // whitespace-separated names, which is what most exporters write.
                let rest = line[key.len()..].trim();
                let names: Vec<&str> = if rest.ends_with(".mtl") && resolve_sibling(base, rest).is_some() {
                    vec![rest]
                } else {
                    rest.split_whitespace().collect()
                };
                for n in names {
                    let Some(p) = resolve_sibling(base, n) else {
                        out.note(format!("material library '{n}' was not found next to the model"));
                        continue;
                    };
                    let Ok(mtl) = std::fs::read(&p) else {
                        out.note(format!("material library '{n}' could not be read"));
                        continue;
                    };
                    let mtl_dir = p.parent().map(|d| d.to_path_buf());
                    for (name, mut mat, map_kd) in parse_mtl(&String::from_utf8_lossy(&mtl)) {
                        if let Some(rel) = map_kd {
                            let path = resolve_sibling(mtl_dir.as_deref(), &rel).or_else(|| resolve_sibling(base, &rel));
                            match path {
                                Some(path) => {
                                    let entry = match img_cache.get(&path) {
                                        Some(v) => *v,
                                        None => {
                                            let loaded = std::fs::read(&path).ok().and_then(|b| {
                                                decode_image(&b, &path.file_name().unwrap_or_default().to_string_lossy())
                                            });
                                            let r = match loaded {
                                                Some(img) if tex_budget + img.rgba.len() <= MAX_TEX_BUDGET => {
                                                    tex_budget += img.rgba.len();
                                                    out.images.push(img);
                                                    Some(out.images.len() - 1)
                                                }
                                                Some(_) => {
                                                    out.note("texture budget reached — a map_Kd image was dropped");
                                                    None
                                                }
                                                None => {
                                                    out.note(format!("could not decode '{rel}' — '{name}' falls back to its flat colour"));
                                                    None
                                                }
                                            };
                                            img_cache.insert(path.clone(), r);
                                            r
                                        }
                                    };
                                    mat.image = entry;
                                }
                                None => out.note(format!("map_Kd '{rel}' was not found next to the model")),
                            }
                        }
                        let name_key = name.clone();
                        mat.name = name;
                        let idx = *mat_index.entry(name_key).or_insert_with(|| {
                            out.materials.push(mat);
                            out.materials.len() - 1
                        });
                        let _ = idx;
                    }
                }
            }
            "usemtl" => {
                let name = line[key.len()..].trim().to_string();
                cur = match mat_index.get(&name) {
                    Some(i) => Some(*i),
                    None => {
                        // `usemtl` for a material no .mtl declared: keep it as a named white
                        // material so the part is still separable.
                        if mtl_seen && !name.is_empty() {
                            out.note(format!("material '{name}' is used but not declared in any mtllib — drawn white"));
                        }
                        if name.is_empty() {
                            None
                        } else {
                            let i = out.materials.len();
                            out.materials.push(ImportMaterial { name: name.clone(), ..Default::default() });
                            mat_index.insert(name, i);
                            Some(i)
                        }
                    }
                };
            }
            "f" => {
                let toks: Vec<&str> = it.collect();
                if toks.len() < 3 {
                    continue;
                }
                let b = builds.entry(cur).or_insert_with(|| Build {
                    verts: Vec::new(),
                    indices: Vec::new(),
                    map: HashMap::new(),
                    had_normals: false,
                });
                let mut corner: Vec<u32> = Vec::with_capacity(toks.len());
                for t in &toks {
                    let mut p = t.split('/');
                    let vi = p.next().and_then(|s| s.parse::<i64>().ok());
                    let ti = p.next().filter(|s| !s.is_empty()).and_then(|s| s.parse::<i64>().ok());
                    let ni = p.next().filter(|s| !s.is_empty()).and_then(|s| s.parse::<i64>().ok());
                    let Some(vi) = vi else { continue };
                    let resolve = |i: i64, n: usize| -> Option<usize> {
                        let r = if i < 0 { n as i64 + i } else { i - 1 };
                        (r >= 0 && (r as usize) < n).then_some(r as usize)
                    };
                    let Some(vp) = resolve(vi, pos.len()) else { continue };
                    let vt = ti.and_then(|i| resolve(i, uvs.len()));
                    let vn = ni.and_then(|i| resolve(i, nrm.len()));
                    let key = (vp as i64, vt.map_or(-1, |v| v as i64), vn.map_or(-1, |v| v as i64));
                    let slot = match b.map.get(&key) {
                        Some(s) => *s,
                        None => {
                            if b.verts.len() >= MAX_VERTS {
                                continue;
                            }
                            let s = b.verts.len() as u32;
                            b.verts.push(ImportVertex {
                                pos: pos[vp],
                                normal: vn.map(|i| nrm[i]).unwrap_or([0.0; 3]),
                                uv: vt.map(|i| uvs[i]).unwrap_or([0.0, 0.0]),
                                color: [1.0; 4],
                            });
                            if vn.is_some() {
                                b.had_normals = true;
                            }
                            b.map.insert(key, s);
                            s
                        }
                    };
                    corner.push(slot);
                }
                // fan-triangulate the (possibly n-gon) face
                for k in 1..corner.len().saturating_sub(1) {
                    let (a, b2, c) = (corner[0], corner[k], corner[k + 1]);
                    if a != b2 && b2 != c && a != c {
                        b.indices.extend_from_slice(&[a, b2, c]);
                    }
                }
            }
            _ => {}
        }
    }

    // Stable part order: the no-material bucket first, then by material index.
    let mut keys: Vec<Option<usize>> = builds.keys().copied().collect();
    keys.sort_by_key(|k| k.map(|v| v as i64 + 1).unwrap_or(0));
    for k in keys {
        let Some(mut b) = builds.remove(&k) else { continue };
        if b.indices.len() < 3 {
            continue;
        }
        if !b.had_normals {
            fill_face_normals(&mut b.verts, &b.indices);
        } else {
            for v in b.verts.iter_mut() {
                let n = glam::Vec3::from(v.normal);
                if n.length_squared() < 1e-12 {
                    v.normal = [0.0, 1.0, 0.0];
                }
            }
        }
        let name = k
            .and_then(|i| out.materials.get(i))
            .map(|m| m.name.clone())
            .unwrap_or_else(|| "default".into());
        out.parts.push(ImportPart { name, verts: b.verts, indices: b.indices, material: k });
    }
    out
}

/// Parse a .mtl: returns (name, material, `map_Kd` path) per `newmtl`.
/// `Kd` is the diffuse colour, `d` / `Tr` the opacity, `map_Kd` the diffuse map (with its
/// `-s` scale / `-o` offset options).
pub fn parse_mtl(text: &str) -> Vec<(String, ImportMaterial, Option<String>)> {
    let mut out: Vec<(String, ImportMaterial, Option<String>)> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = match line.split_once(char::is_whitespace) {
            Some((k, r)) => (k.to_ascii_lowercase(), r.trim()),
            None => (line.to_ascii_lowercase(), ""),
        };
        if key == "newmtl" {
            out.push((rest.to_string(), ImportMaterial { name: rest.to_string(), ..Default::default() }, None));
            continue;
        }
        let Some((_, mat, map)) = out.last_mut() else { continue };
        let nums: Vec<f32> = rest.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        match key.as_str() {
            "kd" => {
                if nums.len() >= 3 {
                    for k in 0..3 {
                        mat.base_color[k] = nums[k].clamp(0.0, 1.0);
                    }
                } else if nums.len() == 1 {
                    let g = nums[0].clamp(0.0, 1.0);
                    mat.base_color = [g, g, g, mat.base_color[3]];
                }
            }
            // `d` is opacity, `Tr` is transparency (1 - d). Either one below 1 means blend.
            "d" => {
                if let Some(v) = nums.first() {
                    mat.base_color[3] = v.clamp(0.0, 1.0);
                    if mat.base_color[3] < 0.999 {
                        mat.alpha_mode = AlphaMode::Blend;
                    }
                }
            }
            "tr" => {
                if let Some(v) = nums.first() {
                    mat.base_color[3] = (1.0 - v).clamp(0.0, 1.0);
                    if mat.base_color[3] < 0.999 {
                        mat.alpha_mode = AlphaMode::Blend;
                    }
                }
            }
            "map_kd" => {
                let (path, scale, offset) = parse_map_options(rest);
                if !path.is_empty() {
                    *map = Some(path);
                    mat.uv_scale = scale;
                    mat.uv_offset = offset;
                }
            }
            // A separate alpha map means the material is a cutout.
            "map_d" => mat.alpha_mode = AlphaMode::Mask,
            _ => {}
        }
    }
    out
}

/// Split a `map_*` statement's leading options from its file name.
/// Handles `-s u v w`, `-o u v w`, `-blendu/-blendv/-clamp/-cc on|off`, `-bm f`, `-texres n`,
/// `-imfchan c`; everything after the options is the (possibly space-bearing) path.
fn parse_map_options(rest: &str) -> (String, [f32; 2], [f32; 2]) {
    let mut scale = [1.0f32, 1.0];
    let mut offset = [0.0f32, 0.0];
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut i = 0usize;
    while i < toks.len() {
        let t = toks[i];
        if !t.starts_with('-') {
            break;
        }
        let nums_after = |i: usize| -> usize {
            let mut n = 0;
            while i + 1 + n < toks.len() && toks[i + 1 + n].parse::<f32>().is_ok() && n < 3 {
                n += 1;
            }
            n
        };
        match t.to_ascii_lowercase().as_str() {
            "-s" | "-o" => {
                let n = nums_after(i);
                let vals: Vec<f32> = (0..n).map(|k| toks[i + 1 + k].parse().unwrap_or(0.0)).collect();
                if t.eq_ignore_ascii_case("-s") {
                    if let Some(u) = vals.first() {
                        scale[0] = *u;
                    }
                    scale[1] = *vals.get(1).unwrap_or(&scale[0]);
                } else {
                    offset[0] = *vals.first().unwrap_or(&0.0);
                    offset[1] = *vals.get(1).unwrap_or(&0.0);
                }
                i += 1 + n;
            }
            "-blendu" | "-blendv" | "-clamp" | "-cc" | "-imfchan" | "-type" => i += 2,
            "-bm" | "-mm" | "-texres" => i += 1 + nums_after(i).max(1),
            _ => i += 1,
        }
    }
    let path = toks[i.min(toks.len())..].join(" ");
    (path, scale, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a synthetic GLB fixture, built here so the test needs no asset on disk -------------

    /// Build a minimal valid GLB: one node holding one textured quad (POSITION, NORMAL,
    /// TEXCOORD_0, COLOR_0, u16 indices) with a 2×2 PNG base colour embedded in the BIN chunk,
    /// a `baseColorFactor`, `KHR_texture_transform`, MASK alpha and `doubleSided`.
    fn synth_glb() -> Vec<u8> {
        // --- geometry (interleaved-free: four tightly-packed accessors, one bufferView each)
        let pos: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, -2.0], [0.0, 0.0, -2.0]];
        let nor: [[f32; 3]; 4] = [[0.0, 1.0, 0.0]; 4];
        let uv: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let col: [[f32; 4]; 4] = [[1.0, 0.5, 0.25, 1.0]; 4];
        let idx: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let png = synth_png_2x2();

        let mut bin: Vec<u8> = Vec::new();
        let mut views: Vec<serde_json::Value> = Vec::new();
        let push = |bin: &mut Vec<u8>, views: &mut Vec<serde_json::Value>, bytes: &[u8]| -> usize {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            let off = bin.len();
            bin.extend_from_slice(bytes);
            views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len() }));
            views.len() - 1
        };
        let bv_pos = push(&mut bin, &mut views, bytemuck::cast_slice(&pos));
        let bv_nor = push(&mut bin, &mut views, bytemuck::cast_slice(&nor));
        let bv_uv = push(&mut bin, &mut views, bytemuck::cast_slice(&uv));
        let bv_col = push(&mut bin, &mut views, bytemuck::cast_slice(&col));
        let bv_idx = push(&mut bin, &mut views, bytemuck::cast_slice(&idx));
        let bv_png = push(&mut bin, &mut views, &png);

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "children": [1], "scale": [2.0, 2.0, 2.0] }, { "mesh": 0, "translation": [1.0, 0.0, 0.0] }],
            "meshes": [{ "name": "quad", "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "COLOR_0": 3 },
                "indices": 4, "material": 0, "mode": 4
            }]}],
            "materials": [{
                "name": "painted",
                "doubleSided": true,
                "alphaMode": "MASK",
                "alphaCutoff": 0.25,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.25, 1.0, 1.0],
                    "baseColorTexture": { "index": 0, "extensions": { "KHR_texture_transform": {
                        "scale": [2.0, 3.0], "offset": [0.1, 0.2]
                    }}}
                }
            }],
            "textures": [{ "sampler": 0, "source": 0 }],
            "samplers": [{ "wrapS": 33071, "wrapT": 33648 }],
            "images": [{ "bufferView": bv_png, "mimeType": "image/png" }],
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": views,
            "accessors": [
                { "bufferView": bv_pos, "componentType": 5126, "count": 4, "type": "VEC3" },
                { "bufferView": bv_nor, "componentType": 5126, "count": 4, "type": "VEC3" },
                { "bufferView": bv_uv,  "componentType": 5126, "count": 4, "type": "VEC2" },
                { "bufferView": bv_col, "componentType": 5126, "count": 4, "type": "VEC4" },
                { "bufferView": bv_idx, "componentType": 5123, "count": 6, "type": "SCALAR" }
            ]
        });
        pack_glb(&json, &bin)
    }

    fn pack_glb(json: &serde_json::Value, bin: &[u8]) -> Vec<u8> {
        let mut js = serde_json::to_vec(json).unwrap();
        while js.len() % 4 != 0 {
            js.push(b' ');
        }
        let mut bin = bin.to_vec();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + js.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(js.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        out.extend_from_slice(&js);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    /// A 2×2 RGBA PNG: red, green / blue, transparent-white.
    fn synth_png_2x2() -> Vec<u8> {
        let px: [u8; 16] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 0];
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&px).unwrap();
        }
        out
    }

    #[test]
    fn glb_yields_uvs_materials_and_a_decoded_texture() {
        let glb = synth_glb();
        let m = parse_gltf(&glb, None).expect("the synthetic GLB must parse");
        assert_eq!(m.format, "GLB");
        assert_eq!(m.parts.len(), 1, "one primitive -> one part");
        assert_eq!(m.materials.len(), 1);
        assert_eq!(m.images.len(), 1, "the embedded PNG must be decoded");
        let img = &m.images[0];
        assert_eq!((img.w, img.h), (2, 2));
        assert_eq!(&img.rgba[0..4], &[255, 0, 0, 255], "texel 0 must be the red pixel");
        assert_eq!(img.rgba[15], 0, "the PNG's alpha channel must survive");

        let mat = &m.materials[0];
        assert_eq!(mat.name, "painted");
        assert_eq!(mat.image, Some(0));
        assert_eq!(mat.base_color, [0.5, 0.25, 1.0, 1.0], "baseColorFactor");
        assert_eq!(mat.alpha_mode, AlphaMode::Mask);
        assert!((mat.alpha_cutoff - 0.25).abs() < 1e-6);
        assert!(mat.double_sided);
        assert_eq!(mat.wrap, [Wrap::Clamp, Wrap::Mirror], "the sampler's wrap modes");

        let p = &m.parts[0];
        assert_eq!(p.verts.len(), 4);
        assert_eq!(p.indices, vec![0, 1, 2, 0, 2, 3]);
        // UVs must be real AND carry KHR_texture_transform (uv*scale + offset).
        assert!((p.verts[0].uv[0] - 0.1).abs() < 1e-6 && (p.verts[0].uv[1] - 0.2).abs() < 1e-6, "uv0 {:?}", p.verts[0].uv);
        assert!((p.verts[2].uv[0] - 2.1).abs() < 1e-6 && (p.verts[2].uv[1] - 3.2).abs() < 1e-6, "uv2 {:?}", p.verts[2].uv);
        // COLOR_0 must be read.
        assert_eq!(p.verts[0].color, [1.0, 0.5, 0.25, 1.0]);
        // The node chain (parent scale 2, child translate +1 in X) must be applied: the quad's
        // first vertex is at (0,0,0) locally -> (1,0,0) after the child -> (2,0,0) after the parent.
        assert!((p.verts[0].pos[0] - 2.0).abs() < 1e-5, "node transform not applied: {:?}", p.verts[0].pos);
        assert!((p.verts[1].pos[0] - 6.0).abs() < 1e-5, "node scale not applied: {:?}", p.verts[1].pos);
        assert_eq!(p.verts[0].normal, [0.0, 1.0, 0.0]);
        assert_eq!(m.textured_material_count(), 1);
        assert!(m.has_uvs());
        assert!(m.material_summary().contains("1 material"), "{}", m.material_summary());
        assert!(m.material_summary().contains("1 texture"), "{}", m.material_summary());
    }

    #[test]
    fn glb_with_a_broken_image_falls_back_to_flat_colour() {
        // Same fixture, but the PNG bytes are replaced by garbage.
        // A fixture whose "PNG" payload is deliberately garbage.
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 }]}],
            "materials": [{ "name": "bad", "pbrMetallicRoughness": {
                "baseColorFactor": [0.25, 0.5, 0.75, 1.0],
                "baseColorTexture": { "index": 0 } } }],
            "textures": [{ "source": 0 }],
            "images": [{ "bufferView": 2, "mimeType": "image/png" }],
            "buffers": [{ "byteLength": 0 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 8 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ]
        });
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&v));
        }
        bin.extend_from_slice(bytemuck::cast_slice(&[0u16, 1, 2]));
        bin.extend_from_slice(&[0, 0]); // pad to 48
        bin.extend_from_slice(b"notapng!"); // the "image"
        let glb = pack_glb(&json, &bin);
        let m = parse_gltf(&glb, None).expect("a broken image must not fail the import");
        assert_eq!(m.parts.len(), 1, "the triangle must still be there");
        assert_eq!(m.images.len(), 0, "nothing decodable");
        assert_eq!(m.materials[0].image, None, "the material must fall back to flat colour");
        assert_eq!(m.materials[0].base_color, [0.25, 0.5, 0.75, 1.0]);
        assert!(m.notes.iter().any(|n| n.contains("could not decode")), "notes: {:?}", m.notes);
    }

    #[test]
    fn glb_with_no_uvs_and_no_material_still_loads() {
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]}],
            "buffers": [{ "byteLength": 0 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ]
        });
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&v));
        }
        bin.extend_from_slice(bytemuck::cast_slice(&[0u16, 1, 2]));
        let m = parse_gltf(&pack_glb(&json, &bin), None).expect("no material is not an error");
        assert_eq!(m.parts.len(), 1);
        assert_eq!(m.parts[0].material, None);
        assert!(m.materials.is_empty());
        assert!(!m.has_uvs());
        // Normals must be synthesised from the face (the primitive shipped none).
        let n = m.parts[0].verts[0].normal;
        assert!(n.iter().map(|v| v * v).sum::<f32>() > 0.9, "normal {n:?}");
    }

    #[test]
    fn truncated_and_lying_glb_headers_do_not_panic() {
        assert!(parse_gltf(b"glTF", None).is_err());
        assert!(parse_gltf(b"", None).is_err());
        // A JSON chunk whose length overruns the file.
        let mut bad = Vec::new();
        bad.extend_from_slice(b"glTF");
        bad.extend_from_slice(&2u32.to_le_bytes());
        bad.extend_from_slice(&999_999u32.to_le_bytes());
        bad.extend_from_slice(&999_999u32.to_le_bytes());
        bad.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        bad.extend_from_slice(b"{\"asset\":{\"version\":\"2.0\"}}");
        assert!(parse_gltf(&bad, None).is_err());
        // An accessor claiming far more elements than its bufferView holds.
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }]}],
            "buffers": [{ "byteLength": 12 }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 12 }],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 1000000, "type": "VEC3" }]
        });
        let r = parse_gltf(&pack_glb(&json, &[0u8; 12]), None);
        // One vertex fits; a triangle does not, so the import is empty rather than a crash.
        assert!(r.is_err() || r.map(|m| m.tri_count()).unwrap_or(0) == 0);
    }

    #[test]
    fn texture_size_is_clamped() {
        // A PNG bigger than MAX_TEX_DIM must come back downsampled, not refused.
        let (w, h) = (MAX_TEX_DIM * 2, 4);
        let mut px = vec![0u8; (w as usize) * (h as usize) * 4];
        for (i, c) in px.chunks_exact_mut(4).enumerate() {
            c[0] = (i % 251) as u8;
            c[3] = 255;
        }
        let mut png_bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut png_bytes, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut wr = enc.write_header().unwrap();
            wr.write_image_data(&px).unwrap();
        }
        let img = decode_image(&png_bytes, "big").expect("an oversized PNG must still decode");
        assert!(img.w <= MAX_TEX_DIM && img.h <= MAX_TEX_DIM, "{}x{} not clamped", img.w, img.h);
        assert_eq!(img.w, MAX_TEX_DIM, "halved exactly once");
        assert_eq!(img.rgba.len(), (img.w as usize) * (img.h as usize) * 4);
        // garbage must simply be None
        assert!(decode_image(b"", "x").is_none());
        assert!(decode_image(&[0x89, b'P', b'N', b'G', 13, 10, 26, 10, 1, 2, 3], "x").is_none());
        assert!(decode_image(&[0xFF, 0xD8, 0xFF, 0x01, 0x02], "x").is_none());
    }

    // ---- OBJ / MTL --------------------------------------------------------------------------

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hms-import-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    #[test]
    fn obj_reads_vt_usemtl_and_the_sibling_mtl() {
        let dir = tmpdir("obj-mtl");
        let png = dir.join("brick.png");
        std::fs::write(&png, synth_png_2x2()).unwrap();
        std::fs::write(
            dir.join("two.mtl"),
            "newmtl red\nKd 0.9 0.1 0.1\n\nnewmtl brick\nKd 1 1 1\nmap_Kd -s 2 4 -o 0.5 0.25 brick.png\n",
        )
        .unwrap();
        let obj = "mtllib two.mtl\n\
                   v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n\
                   vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
                   vn 0 1 0\n\
                   usemtl red\nf 1/1/1 2/2/1 3/3/1\n\
                   usemtl brick\nf 1/1/1 3/3/1 4/4/1\n";
        std::fs::write(dir.join("two.obj"), obj).unwrap();
        let m = load(&dir.join("two.obj")).expect("the OBJ must load");
        assert_eq!(m.format, "OBJ");
        assert_eq!(m.materials.len(), 2, "both newmtl entries");
        assert_eq!(m.parts.len(), 2, "one part per usemtl group");
        assert_eq!(m.images.len(), 1, "map_Kd must be decoded");
        let brick = m.materials.iter().find(|x| x.name == "brick").expect("brick material");
        assert_eq!(brick.image, Some(0));
        assert_eq!(brick.uv_scale, [2.0, 4.0], "map_Kd -s");
        assert_eq!(brick.uv_offset, [0.5, 0.25], "map_Kd -o");
        let red = m.materials.iter().find(|x| x.name == "red").expect("red material");
        assert_eq!(red.image, None);
        assert!((red.base_color[0] - 0.9).abs() < 1e-6 && (red.base_color[1] - 0.1).abs() < 1e-6);
        // real texture coordinates (OBJ v is flipped)
        assert!(m.has_uvs());
        let part = m.parts.iter().find(|p| p.material.map(|i| m.materials[i].name.clone()) == Some("brick".into())).unwrap();
        assert!(part.verts.iter().any(|v| (v.uv[1] - 1.0).abs() < 1e-6 || (v.uv[1] - 0.0).abs() < 1e-6));
        assert_eq!(m.tri_count(), 2);
        assert_eq!(m.textured_material_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn obj_with_no_material_still_looks_sane() {
        let m = parse_obj("v 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n", None);
        assert_eq!(m.parts.len(), 1);
        assert_eq!(m.parts[0].material, None, "no usemtl -> the default white material");
        assert!(m.materials.is_empty());
        assert_eq!(m.tri_count(), 1);
        // a synthesised normal, not zero (a zero normal would render black)
        let n = glam::Vec3::from(m.parts[0].verts[0].normal);
        assert!((n.length() - 1.0).abs() < 1e-4, "normal {n:?}");
        assert_eq!(m.material_summary(), "0 materials, 0 textures");
        // negative / relative indices and n-gons
        let m = parse_obj("v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\nf -4 -3 -2 -1\n", None);
        assert_eq!(m.tri_count(), 2, "a quad fan-triangulates");
    }

    #[test]
    fn obj_mtllib_that_is_missing_is_a_note_not_a_failure() {
        let dir = tmpdir("obj-nomtl");
        std::fs::write(dir.join("a.obj"), "mtllib gone.mtl\nusemtl ghost\nv 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n").unwrap();
        let m = load(&dir.join("a.obj")).unwrap();
        assert_eq!(m.tri_count(), 1);
        assert!(m.notes.iter().any(|n| n.contains("gone.mtl")), "notes {:?}", m.notes);
        // `usemtl ghost` still makes a (white) named material so the part is separable
        assert_eq!(m.materials.len(), 1);
        assert_eq!(m.materials[0].base_color, [1.0; 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mtl_options_and_opacity() {
        let v = parse_mtl("newmtl glass\nKd 0.2 0.4 0.6\nd 0.3\nmap_Kd -clamp on -s 1 1 1 tex with space.png\n");
        assert_eq!(v.len(), 1);
        let (name, mat, map) = &v[0];
        assert_eq!(name, "glass");
        assert_eq!(mat.alpha_mode, AlphaMode::Blend);
        assert!((mat.base_color[3] - 0.3).abs() < 1e-6);
        assert_eq!(map.as_deref(), Some("tex with space.png"));
        let v = parse_mtl("newmtl t\nTr 0.25\n");
        assert!((v[0].1.base_color[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn a_texture_path_cannot_escape_the_model_directory() {
        let dir = tmpdir("escape");
        let sub = dir.join("model");
        let _ = std::fs::create_dir_all(&sub);
        std::fs::write(dir.join("secret.png"), synth_png_2x2()).unwrap();
        assert!(resolve_sibling(Some(&sub), "../secret.png").is_none(), "`..` must never reach the parent's file");
        assert!(resolve_sibling(Some(&sub), "/etc/passwd").is_none());
        assert!(resolve_sibling(Some(&sub), "C:\\windows\\win.ini").is_none());
        std::fs::write(sub.join("ok.png"), synth_png_2x2()).unwrap();
        assert!(resolve_sibling(Some(&sub), "ok.png").is_some());
        // `..` is refused, then the bare name beside the model is tried — which stays inside.
        assert_eq!(resolve_sibling(Some(&sub), "textures/../ok.png"), Some(sub.join("ok.png")));
        // A stale absolute path still resolves to the bare name beside the model.
        assert!(resolve_sibling(Some(&sub), "/home/someone/else/ok.png").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_uris_decode() {
        // "Hi!" base64
        assert_eq!(data_uri_bytes("data:application/octet-stream;base64,SGkh").as_deref(), Some(&b"Hi!"[..]));
        assert_eq!(data_uri_bytes("data:text/plain,a%20b").as_deref(), Some(&b"a b"[..]));
        assert!(data_uri_bytes("notadata").is_none());
    }

    #[test]
    fn gltf_json_with_a_data_uri_buffer_loads() {
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&v));
        }
        bin.extend_from_slice(bytemuck::cast_slice(&[0u16, 1, 2]));
        let b64 = {
            const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut s = String::new();
            for c in bin.chunks(3) {
                let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
                let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
                s.push(T[(n >> 18) as usize & 63] as char);
                s.push(T[(n >> 12) as usize & 63] as char);
                s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
                s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
            }
            s
        };
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]}],
            "buffers": [{ "byteLength": bin.len(), "uri": format!("data:application/octet-stream;base64,{b64}") }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ]
        });
        let m = parse_gltf(serde_json::to_string(&json).unwrap().as_bytes(), None).expect("a .gltf with a data: buffer");
        assert_eq!(m.format, "glTF");
        assert_eq!(m.tri_count(), 1);
    }

    #[test]
    fn byte_stride_and_normalized_attributes_are_honoured() {
        // Interleaved POSITION (f32x3) + TEXCOORD_0 (u16 normalized) in ONE 20-byte-stride view.
        let mut bin: Vec<u8> = Vec::new();
        for (p, uv) in [
            ([0.0f32, 0.0, 0.0], [0u16, 0u16]),
            ([1.0, 0.0, 0.0], [65535, 0]),
            ([0.0, 0.0, 1.0], [0, 65535]),
        ] {
            bin.extend_from_slice(bytemuck::cast_slice(&p));
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
            bin.extend_from_slice(&[0, 0, 0, 0]); // pad to stride 20
        }
        let ioff = bin.len();
        bin.extend_from_slice(bytemuck::cast_slice(&[0u16, 1, 2]));
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "TEXCOORD_0": 1 }, "indices": 2, "material": 0 }]}],
            "materials": [{ "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] } }],
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": ioff, "byteStride": 20 },
                { "buffer": 0, "byteOffset": ioff, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "byteOffset": 0,  "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 0, "byteOffset": 12, "componentType": 5123, "count": 3, "type": "VEC2", "normalized": true },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ]
        });
        let m = parse_gltf(&pack_glb(&json, &bin), None).expect("interleaved attributes");
        let p = &m.parts[0];
        assert_eq!(p.verts.len(), 3);
        assert!((p.verts[1].pos[0] - 1.0).abs() < 1e-6, "stride broke POSITION: {:?}", p.verts[1].pos);
        assert!((p.verts[2].pos[2] - 1.0).abs() < 1e-6, "stride broke POSITION: {:?}", p.verts[2].pos);
        assert!((p.verts[1].uv[0] - 1.0).abs() < 1e-4, "normalized u16 UV: {:?}", p.verts[1].uv);
        assert!((p.verts[2].uv[1] - 1.0).abs() < 1e-4, "normalized u16 UV: {:?}", p.verts[2].uv);
    }

    #[test]
    fn a_cyclic_node_graph_terminates() {
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "children": [1] }, { "children": [0], "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]}],
            "buffers": [{ "byteLength": 0 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ]
        });
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            bin.extend_from_slice(bytemuck::cast_slice(&v));
        }
        bin.extend_from_slice(bytemuck::cast_slice(&[0u16, 1, 2]));
        // The depth cap stops the recursion; the model still has triangles.
        let m = parse_gltf(&pack_glb(&json, &bin), None).expect("a cycle must not hang");
        assert!(m.tri_count() >= 1);
        assert!(m.parts.len() <= MAX_NODE_DEPTH);
    }
}
