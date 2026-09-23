//! Model Painter — import a Wavefront .obj and surface-voxelize it into a set of
//! block positions the caller spawns as forge objects. Port of the C# Model
//! Painter's surface-shell mode (the OBJ subset; .glb is a follow-up). Pure text
//! parsing + triangle sampling, no external deps.

use std::path::{Path, PathBuf};

/// Parsed OBJ triangle mesh (positions + triangle indices).
pub struct ObjMesh {
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[usize; 3]>,
}

/// Parse the subset of .obj we need: `v x y z` and `f a b c …` (triangulated as
/// a fan; negative/relative indices and `v/vt/vn` triples are handled).
pub fn parse_obj(text: &str) -> ObjMesh {
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_whitespace().filter_map(|t| t.parse::<f32>().ok());
            if let (Some(x), Some(y), Some(z)) = (it.next(), it.next(), it.next()) {
                verts.push([x, y, z]);
            }
        } else if let Some(rest) = line.strip_prefix("f ") {
            let idx: Vec<usize> = rest
                .split_whitespace()
                .filter_map(|tok| {
                    let first = tok.split('/').next()?;
                    let i: i64 = first.parse().ok()?;
                    let n = verts.len() as i64;
                    let resolved = if i < 0 { n + i } else { i - 1 };
                    if resolved >= 0 && resolved < n { Some(resolved as usize) } else { None }
                })
                .collect();
            // Fan-triangulate the (possibly n-gon) face.
            for k in 1..idx.len().saturating_sub(1) {
                tris.push([idx[0], idx[k], idx[k + 1]]);
            }
        }
    }
    ObjMesh { verts, tris }
}

/// Surface-voxelize into unique voxel centres, returned in [0,1]^3 (the mesh is
/// normalised to its bounding box). `res` is the grid resolution per axis.
fn voxelize_surface(mesh: &ObjMesh, res: usize) -> Vec<[f32; 3]> {
    if mesh.verts.is_empty() || mesh.tris.is_empty() || res == 0 {
        return Vec::new();
    }
    let res = res.min(64);
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for v in &mesh.verts {
        for a in 0..3 {
            mn[a] = mn[a].min(v[a]);
            mx[a] = mx[a].max(v[a]);
        }
    }
    let ext = [
        (mx[0] - mn[0]).max(1e-4),
        (mx[1] - mn[1]).max(1e-4),
        (mx[2] - mn[2]).max(1e-4),
    ];
    let to_cell = |p: [f32; 3]| -> [usize; 3] {
        [
            (((p[0] - mn[0]) / ext[0]) * res as f32).clamp(0.0, (res - 1) as f32) as usize,
            (((p[1] - mn[1]) / ext[1]) * res as f32).clamp(0.0, (res - 1) as f32) as usize,
            (((p[2] - mn[2]) / ext[2]) * res as f32).clamp(0.0, (res - 1) as f32) as usize,
        ]
    };
    let mut set = std::collections::HashSet::new();
    for t in &mesh.tris {
        let (a, b, c) = (mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]);
        // Sample density from the triangle's size in cells (so we don't miss any).
        let el = |u: [f32; 3], v: [f32; 3]| {
            let d = [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let world_span = el(a, b).max(el(b, c)).max(el(c, a));
        let cell_size = ext.iter().cloned().fold(0.0, f32::max) / res as f32;
        let steps = ((world_span / cell_size).ceil() as usize + 1).clamp(2, 96);
        for i in 0..=steps {
            for j in 0..=(steps - i) {
                let (u, v) = (i as f32 / steps as f32, j as f32 / steps as f32);
                let w = 1.0 - u - v;
                let p = [
                    a[0] * w + b[0] * u + c[0] * v,
                    a[1] * w + b[1] * u + c[1] * v,
                    a[2] * w + b[2] * u + c[2] * v,
                ];
                set.insert(to_cell(p));
            }
        }
    }
    set.into_iter()
        .map(|c| {
            [
                (c[0] as f32 + 0.5) / res as f32,
                (c[1] as f32 + 0.5) / res as f32,
                (c[2] as f32 + 0.5) / res as f32,
            ]
        })
        .collect()
}

/// Parse a binary glTF (.glb): JSON chunk + BIN chunk → first mesh primitive's
/// POSITION + indices accessors → triangle mesh.
pub fn parse_glb(bytes: &[u8]) -> Option<ObjMesh> {
    if bytes.len() < 12 || &bytes[0..4] != b"glTF" {
        return None;
    }
    let u32at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
    // Chunks after the 12-byte header.
    let mut off = 12;
    let mut json: Option<serde_json::Value> = None;
    let mut bin: Option<&[u8]> = None;
    while off + 8 <= bytes.len() {
        let len = u32at(off);
        let ctype = u32at(off + 4);
        let start = off + 8;
        let end = (start + len).min(bytes.len());
        if ctype == 0x4E4F_534A {
            // 'JSON'
            json = serde_json::from_slice(&bytes[start..end]).ok();
        } else if ctype == 0x004E_4942 {
            // 'BIN\0'
            bin = Some(&bytes[start..end]);
        }
        off = start + len;
    }
    let json = json?;
    let bin = bin?;
    let accessors = json.get("accessors")?.as_array()?;
    let views = json.get("bufferViews")?.as_array()?;
    let prims = json.get("meshes")?.as_array()?.first()?.get("primitives")?.as_array()?;

    let read_view = |acc_idx: usize| -> Option<(usize, usize, usize)> {
        // (byte_offset, count, component_type)
        let acc = accessors.get(acc_idx)?;
        let count = acc.get("count")?.as_u64()? as usize;
        let comp = acc.get("componentType")?.as_u64()? as usize;
        let acc_off = acc.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let bv_idx = acc.get("bufferView")?.as_u64()? as usize;
        let bv = views.get(bv_idx)?;
        let bv_off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        Some((acc_off + bv_off, count, comp))
    };

    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();

    // Iterate every primitive in the first mesh; each carries its own POSITION
    // accessor, so offset its indices by the running vertex base.
    for prim in prims {
        let Some(pos_acc) = prim.get("attributes").and_then(|a| a.get("POSITION")).and_then(|v| v.as_u64()) else { continue };
        let Some((poff, pcount, _)) = read_view(pos_acc as usize) else { continue };
        let base = verts.len();
        for i in 0..pcount {
            let o = poff + i * 12;
            if o + 12 > bin.len() {
                break;
            }
            let f = |k: usize| f32::from_le_bytes([bin[o + k], bin[o + k + 1], bin[o + k + 2], bin[o + k + 3]]);
            verts.push([f(0), f(4), f(8)]);
        }

        // Build the primitive-local index list (or sequential if none).
        let mode = prim.get("mode").and_then(|v| v.as_u64()).unwrap_or(4);
        let mut idx: Vec<usize> = Vec::new();
        if let Some(idx_acc) = prim.get("indices").and_then(|v| v.as_u64()) {
            if let Some((ioff, icount, comp)) = read_view(idx_acc as usize) {
                for i in 0..icount {
                    let v = match comp {
                        5121 => *bin.get(ioff + i).unwrap_or(&0) as usize,
                        5123 => {
                            let o = ioff + i * 2;
                            u16::from_le_bytes([*bin.get(o).unwrap_or(&0), *bin.get(o + 1).unwrap_or(&0)]) as usize
                        }
                        _ => {
                            let o = ioff + i * 4;
                            u32::from_le_bytes([
                                *bin.get(o).unwrap_or(&0),
                                *bin.get(o + 1).unwrap_or(&0),
                                *bin.get(o + 2).unwrap_or(&0),
                                *bin.get(o + 3).unwrap_or(&0),
                            ]) as usize
                        }
                    };
                    idx.push(v);
                }
            }
        } else {
            idx = (0..pcount).collect();
        }

        // Assemble triangles per primitive mode (4=list, 5=strip, 6=fan).
        let push = |tris: &mut Vec<[usize; 3]>, a: usize, b: usize, c: usize| {
            tris.push([base + a, base + b, base + c]);
        };
        match mode {
            5 => {
                for i in 0..idx.len().saturating_sub(2) {
                    if i % 2 == 0 {
                        push(&mut tris, idx[i], idx[i + 1], idx[i + 2]);
                    } else {
                        push(&mut tris, idx[i + 1], idx[i], idx[i + 2]);
                    }
                }
            }
            6 => {
                for i in 1..idx.len().saturating_sub(1) {
                    push(&mut tris, idx[0], idx[i], idx[i + 1]);
                }
            }
            _ => {
                for c in idx.chunks_exact(3) {
                    push(&mut tris, c[0], c[1], c[2]);
                }
            }
        }
    }
    if verts.is_empty() {
        return None;
    }
    Some(ObjMesh { verts, tris })
}

/// Scan `<exe>/models` for .obj/.glb files (no path typing — like the map catalog).
pub fn model_catalog() -> Vec<PathBuf> {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("models")))
        .unwrap_or_else(|| PathBuf::from("models"));
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            let ok = p
                .extension()
                .map(|x| x.eq_ignore_ascii_case("obj") || x.eq_ignore_ascii_case("glb"))
                .unwrap_or(false);
            if ok {
                out.push(p);
            }
        }
    }
    out
}

pub fn load_and_voxelize(path: &Path, res: usize) -> Option<Vec<[f32; 3]>> {
    let is_glb = path.extension().map(|x| x.eq_ignore_ascii_case("glb")).unwrap_or(false);
    let mesh = if is_glb {
        parse_glb(&std::fs::read(path).ok()?)?
    } else {
        parse_obj(&std::fs::read_to_string(path).ok()?)
    };
    let v = voxelize_surface(&mesh, res);
    if v.is_empty() { None } else { Some(v) }
}
