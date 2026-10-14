//! Model Painter — import a model file and surface-voxelize it into a set of block positions
//! the caller spawns as forge objects. Port of the C# Model Painter's surface-shell mode.
//!
//! The FILE READING lives in [`crate::modelimport`] (OBJ + .mtl, glTF / GLB, materials and
//! textures); the voxelizer only needs the triangle soup, so it flattens a parsed
//! [`crate::modelimport::ModelData`] down to positions + indices. One parser for both paths, so
//! "what the painter voxelizes" and "what the viewport draws" can never disagree about geometry.

use std::path::{Path, PathBuf};

/// A flattened triangle soup (positions + triangle indices) — all the voxelizer needs.
pub struct ObjMesh {
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[usize; 3]>,
}

/// Flatten every part of a parsed model into one soup, in the FILE's own coordinates.
pub fn flatten(data: &crate::modelimport::ModelData) -> ObjMesh {
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(data.vert_count());
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(data.tri_count());
    for part in &data.parts {
        let base = verts.len();
        verts.extend(part.verts.iter().map(|v| v.pos));
        for t in part.indices.chunks_exact(3) {
            tris.push([base + t[0] as usize, base + t[1] as usize, base + t[2] as usize]);
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
    let data = crate::modelimport::load(path).ok()?;
    let v = voxelize_surface(&flatten(&data), res);
    if v.is_empty() { None } else { Some(v) }
}
