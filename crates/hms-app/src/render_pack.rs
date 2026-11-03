//! The RENDER PACK: a tiny, abstract summary of one base map that the browser viewer
//! (`docs/js/hms-render.js`) uses to draw a variant roughly right. #mvar-js
//!
//! It is generated LOCALLY from the user's own installed game (`variant get render-pack <out.json>`
//! in a headless script) and loaded into the viewer from disk. It is never shipped with HMS or
//! the website, and must not be redistributed: it is derived from game data. It deliberately holds
//! nothing detailed:
//!
//!   palette[i]  per palette (quota) index: name, a category word, and one local axis-aligned box
//!               (the object's physics hull, else its collision hull) -- no vertices;
//!   grid        a 128 x 128 top-down occupancy grid over the variant's world bounds: per cell the
//!               lowest and highest Z of the level's collision triangles that touch it, 8 bits each
//!               over [zmin, zmax] (255 = nothing). It is NOT the triangle mesh.
//!
//! Pack v2 adds an optional SIMPLIFIED MESH per object type (<= ~400 triangles, vertex-clustered,
//! positions as int16 inside the mesh's own box, indices as u8/u16, no normals / UVs / materials):
//! the object's physics hull where the render model is a stub or disagrees with it (blocks, walls,
//! ramps: convex hulls are essentially exact), else the decimated render model. A v1 pack (boxes
//! only) still loads.
//!
//! The generator is one function over `&dyn ObjectScene`, so Reach and Halo 4 share it.

use glam::Vec3;
use std::collections::HashMap;
use hms_ipc::ObjectInfo;
use serde_json::{json, Value};

use crate::objscene::ObjectScene;

pub const GRID: usize = 128;
/// Triangle budget per mesh, and the whole-pack byte budget the builder squeezes the meshes under.
pub const MESH_TRIS: usize = 400;
pub const PACK_BUDGET: usize = 2_500_000;

/// One palette (quota) entry to measure.
pub struct PackItem {
    pub index: usize,
    pub name: String,
    /// The palette's own category text (may be empty).
    pub category: String,
    /// The object tag class ("weap", "vehi", ...), empty when unknown.
    pub class: String,
    /// The editor tag id the scene resolves (Reach: tag index; Halo 4: `h4_tag(obje)`).
    pub obj_tag: u32,
    /// Every variant of the entry (index as stored in a variant's records, name, tag). A palette
    /// entry like "structure_blocks" holds many differently sized objects, so each is measured.
    pub variants: Vec<(u32, String, u32)>,
}

/// A coarse category word from the tag class and the palette category / name (INFERRED from names;
/// the viewer only uses it to pick a colour).
pub fn category_word(class: &str, category: &str, name: &str) -> &'static str {
    let c = category.to_lowercase();
    let n = name.to_lowercase();
    let has = |s: &str| c.contains(s) || n.contains(s);
    if has("spawn") || has("respawn") || has("initial") { return "spawn"; }
    if has("kill") || has("safe") || has("zone") || has("boundary") || has("gravity") || has("trigger") { return "zone"; }
    match class {
        "vehi" => "vehicle",
        "weap" => "weapon",
        "eqip" => "equipment",
        "bloc" => "block",
        "scen" => "scenery",
        "mach" | "ctrl" | "dspn" => "device",
        _ => {
            if has("vehicle") { "vehicle" } else if has("weapon") { "weapon" } else if has("structure") || has("block") || has("wall") { "block" } else { "other" }
        }
    }
}

fn r3(v: f32) -> f64 { ((v as f64) * 1000.0).round() / 1000.0 }

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len() * 4 / 3 + 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

/// The local box of one object type: its physics body AABB, else the AABB of its (posed at the
/// origin) physics / collision triangles. None when the object has neither.
fn local_box(scene: &dyn ObjectScene, tag: u32) -> Option<([f32; 3], [f32; 3])> {
    let o = ObjectInfo {
        datum: 0, type_sig: 0, sig0: 0, sig1: 0, pos: [0.0; 3], health: 1.0, shield: 1.0, mode_tag: 0,
        fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0,
    };
    let from = |mn: Vec3, mx: Vec3| (mn.is_finite() && mx.is_finite() && mx.x >= mn.x).then(|| (mn.to_array(), mx.to_array()));
    if let Some(b) = scene.physics_body(&o) {
        let (mn, mx) = b.aabb();
        if let Some(r) = from(mn, mx) { return Some(r); }
    }
    let tris = scene.physics_world_tris(&o);
    if tris.is_empty() { return None; }
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in tris.iter().flatten() { lo = lo.min(*p); hi = hi.max(*p); }
    from(lo, hi)
}

/// The top-down min/max Z grid of the level over `bounds` (xmin xmax ymin ymax zmin zmax).
fn height_grid(scene: &dyn ObjectScene, bounds: [f32; 6]) -> Value {
    let (zmin, zmax) = (bounds[4], bounds[5]);
    let span = (zmax - zmin).max(1e-3);
    let (cw, ch) = ((bounds[1] - bounds[0]) / GRID as f32, (bounds[3] - bounds[2]) / GRID as f32);
    let mut lo = vec![255u8; GRID * GRID];
    let mut hi = vec![255u8; GRID * GRID];
    let q = |z: f32| ((z.clamp(zmin, zmax) - zmin) / span * 254.0).round() as u8;
    let mut tris: Vec<[Vec3; 3]> = Vec::new();
    for gy in 0..GRID {
        for gx in 0..GRID {
            let (x0, y0) = (bounds[0] + gx as f32 * cw, bounds[2] + gy as f32 * ch);
            tris.clear();
            scene.static_tris_in_aabb(Vec3::new(x0, y0, zmin - 1.0), Vec3::new(x0 + cw, y0 + ch, zmax + 1.0), &mut tris);
            let (mut a, mut b) = (f32::MAX, f32::MIN);
            for t in &tris {
                // only triangles whose own footprint really touches the cell
                let (tx0, tx1) = (t[0].x.min(t[1].x).min(t[2].x), t[0].x.max(t[1].x).max(t[2].x));
                let (ty0, ty1) = (t[0].y.min(t[1].y).min(t[2].y), t[0].y.max(t[1].y).max(t[2].y));
                if tx1 < x0 || tx0 > x0 + cw || ty1 < y0 || ty0 > y0 + ch { continue; }
                for p in t { a = a.min(p.z); b = b.max(p.z); }
            }
            if a <= b && b >= zmin - 1.0 && a <= zmax + 1.0 {
                lo[gy * GRID + gx] = q(a);
                hi[gy * GRID + gx] = q(b);
            }
        }
    }
    json!({ "w": GRID, "h": GRID, "zmin": r3(zmin), "zmax": r3(zmax), "min": b64(&lo), "max": b64(&hi) })
}


// ---------------------------------------------------------------------------------------------
// simplified meshes with a minimal material per part
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tri { pub p: [Vec3; 3], pub uv: [[f32; 2]; 3] }

/// One material part of an object's render model: what the browser needs to draw it see-through or
/// not and in roughly the right colour. `lane` is the scene's own part lane (opaque / cutout /
/// blend / additive), from the shader tag's authored blend type, alpha test, glass / halogram /
/// forcefield class; never a name.
#[derive(Clone)]
pub struct Part {
    pub tris: Vec<Tri>,
    pub lane: &'static str,
    pub color: Option<[f32; 3]>,
    pub opacity: f32,
    pub tex: Option<std::sync::Arc<(Vec<u8>, u32, u32)>>,
    pub halogram: bool,
}

/// What the scene knows about an object's render model.
#[derive(Clone, Default)]
pub struct RenderInfo { pub parts: Vec<Part>, pub model_tris: usize, pub effects: bool }

/// The generated model of one object type, before encoding.
struct Model { parts: Vec<Part>, cat: Option<&'static str> }

fn tris_box(t: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in t.iter().flatten() { lo = lo.min(*p); hi = hi.max(*p); }
    (lo, hi)
}
fn part_box(parts: &[Part]) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in parts { for t in &p.tris { for v in &t.p { lo = lo.min(*v); hi = hi.max(*v); } } }
    (lo, hi)
}

/// Vertex-cluster decimation of one part (cell = the average of the vertices, UVs averaged too),
/// refined by bisection to the finest grid that stays within `max_tris`.
pub fn decimate(tris: &[Tri], lo: Vec3, ext: Vec3, max_tris: usize) -> Vec<Tri> {
    if tris.len() <= max_tris { return tris.to_vec(); }
    let ext = ext.max(Vec3::splat(1e-4));
    let run = |g: u32| -> Vec<Tri> {
        let cell = |p: Vec3| -> (u32, u32, u32) {
            let q = ((p - lo) / ext * g as f32).min(Vec3::splat(g as f32 - 1.0)).max(Vec3::ZERO);
            (q.x as u32, q.y as u32, q.z as u32)
        };
        let mut acc: HashMap<(u32, u32, u32), (Vec3, [f32; 2], u32)> = HashMap::new();
        for t in tris { for k in 0..3 { let e = acc.entry(cell(t.p[k])).or_insert((Vec3::ZERO, [0.0; 2], 0)); e.0 += t.p[k]; e.1[0] += t.uv[k][0]; e.1[1] += t.uv[k][1]; e.2 += 1; } }
        let rep = |p: Vec3| { let (s, u, n) = acc[&cell(p)]; (s / n as f32, [u[0] / n as f32, u[1] / n as f32]) };
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for t in tris {
            let (a, b, c) = (cell(t.p[0]), cell(t.p[1]), cell(t.p[2]));
            if a == b || b == c || a == c { continue; }
            let mut key = [a, b, c]; key.sort();
            if !seen.insert(key) { continue; }
            let (r0, r1, r2) = (rep(t.p[0]), rep(t.p[1]), rep(t.p[2]));
            out.push(Tri { p: [r0.0, r1.0, r2.0], uv: [r0.1, r1.1, r2.1] });
        }
        out
    };
    let (mut lo_g, mut hi_g) = (2u32, 128u32);
    let mut best = run(lo_g);
    while lo_g + 1 < hi_g {
        let mid = (lo_g + hi_g) / 2;
        let r = run(mid);
        if r.len() <= max_tris { best = r; lo_g = mid; } else { hi_g = mid; }
    }
    best
}

/// Merge parts that share lane + texture (so a model of many sub-meshes keeps a handful of draws).
fn merge_parts(parts: Vec<Part>) -> Vec<Part> {
    let mut out: Vec<Part> = Vec::new();
    for p in parts {
        let key = |q: &Part| (q.lane, q.tex.as_ref().map(|t| std::sync::Arc::as_ptr(t) as usize));
        if let Some(o) = out.iter_mut().find(|o| key(o) == key(&p)) {
            let (a, b) = (o.tris.len() as f32, p.tris.len() as f32);
            if let (Some(c0), Some(c1)) = (o.color, p.color) { o.color = Some([(c0[0] * a + c1[0] * b) / (a + b), (c0[1] * a + c1[1] * b) / (a + b), (c0[2] * a + c1[2] * b) / (a + b)]); }
            o.opacity = (o.opacity * a + p.opacity * b) / (a + b);
            o.halogram |= p.halogram;
            o.tris.extend(p.tris);
        } else { out.push(p); }
    }
    out
}

fn simplify(parts: Vec<Part>, max_tris: usize) -> Vec<Part> {
    let mut parts = merge_parts(parts);
    parts.sort_by_key(|p| std::cmp::Reverse(p.tris.len()));
    parts.truncate((max_tris / 8).max(1));
    let (lo, hi) = part_box(&parts);
    let ext = hi - lo;
    let total: usize = parts.iter().map(|p| p.tris.len()).sum();
    for p in parts.iter_mut() {
        let share = ((max_tris as f32 * p.tris.len() as f32 / total.max(1) as f32).floor() as usize).max(8).min(max_tris);
        p.tris = decimate(&p.tris, lo, ext, share);
    }
    // the shares were floored and given a minimum: trim the largest parts if the sum overshoots
    while parts.iter().map(|p| p.tris.len()).sum::<usize>() > max_tris {
        let big = parts.iter_mut().max_by_key(|p| p.tris.len()).unwrap();
        let cut = (big.tris.len() / 10).max(1);
        let tgt = big.tris.len() - cut;
        big.tris = decimate(&big.tris, lo, ext, tgt);
        if big.tris.len() > tgt { big.tris.truncate(tgt); }
    }
    parts.retain(|p| !p.tris.is_empty());
    parts
}

fn tex_small(t: &(Vec<u8>, u32, u32), n: usize) -> Vec<u8> {
    #[allow(non_snake_case)]
    let N = n;
    let (w, h) = (t.1 as usize, t.2 as usize);
    let mut out = vec![0u8; N * N * 4];
    if w == 0 || h == 0 || t.0.len() < w * h * 4 { return out; }
    for y in 0..N { for x in 0..N {
        let (x0, x1) = (x * w / N, ((x + 1) * w / N).max(x * w / N + 1).min(w));
        let (y0, y1) = (y * h / N, ((y + 1) * h / N).max(y * h / N + 1).min(h));
        let mut acc = [0u32; 4]; let mut n = 0u32;
        for yy in y0..y1 { for xx in x0..x1 { let o = (yy * w + xx) * 4; for k in 0..4 { acc[k] += t.0[o + k] as u32; } n += 1; } }
        for k in 0..4 { out[(y * N + x) * 4 + k] = (acc[k] / n.max(1)) as u8; }
    } }
    out
}

/// `{box, n, ib, tris, v, i, [uv, uvb], parts:[{lane, c?, o, t?, s, n}]}`: unique vertices quantised
/// to int16 inside the model's own box (little endian x,y,z) with optional u16 UV pairs, triangle
/// indices as u8 (<= 256 vertices) or u16, ordered part by part; a part is the triangle range
/// [s, s+n).
fn encode_model(parts: &[Part], tex_index: &HashMap<usize, usize>) -> Option<Value> {
    if parts.is_empty() { return None; }
    let (lo, hi) = part_box(parts);
    let ext = (hi - lo).max(Vec3::splat(1e-5));
    let has_uv = parts.iter().any(|p| p.tex.is_some() && tex_index.contains_key(&(std::sync::Arc::as_ptr(p.tex.as_ref().unwrap()) as usize)));
    let (mut u0, mut v0, mut u1, mut v1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    if has_uv { for p in parts { for t in &p.tris { for uv in &t.uv { if uv[0].is_finite() && uv[1].is_finite() { u0 = u0.min(uv[0]); u1 = u1.max(uv[0]); v0 = v0.min(uv[1]); v1 = v1.max(uv[1]); } } } } }
    if !(u1 > u0) { u0 = 0.0; u1 = u0 + 1.0; }
    if !(v1 > v0) { v0 = 0.0; v1 = v0 + 1.0; }
    let mut idx: HashMap<([i16; 3], [u16; 2]), u32> = HashMap::new();
    let mut verts: Vec<([i16; 3], [u16; 2])> = Vec::new();
    let mut ind: Vec<u32> = Vec::new();
    let mut pj: Vec<Value> = Vec::new();
    for p in parts {
        let start = ind.len() / 3;
        for t in &p.tris { for k in 0..3 {
            let q = ((t.p[k] - lo) / ext * 65535.0 - Vec3::splat(32768.0)).round();
            let uv = if has_uv { [(((t.uv[k][0] - u0) / (u1 - u0)).clamp(0.0, 1.0) * 65535.0).round() as u16, (((t.uv[k][1] - v0) / (v1 - v0)).clamp(0.0, 1.0) * 65535.0).round() as u16] } else { [0, 0] };
            let key = ([q.x as i16, q.y as i16, q.z as i16], uv);
            let id = *idx.entry(key).or_insert_with(|| { verts.push(key); verts.len() as u32 - 1 });
            ind.push(id);
        } }
        let mut e = json!({ "lane": p.lane, "o": (p.opacity * 100.0).round() / 100.0, "s": start, "n": ind.len() / 3 - start });
        if let Some(c) = p.color { e["c"] = json!([(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8]); }
        if let Some(t) = &p.tex { if let Some(&ti) = tex_index.get(&(std::sync::Arc::as_ptr(t) as usize)) { e["t"] = json!(ti); } }
        pj.push(e);
    }
    let mut vb = Vec::with_capacity(verts.len() * 6);
    for v in &verts { for c in v.0 { vb.extend_from_slice(&c.to_le_bytes()); } }
    let wide = verts.len() > 256;
    let mut ib = Vec::with_capacity(ind.len() * if wide { 2 } else { 1 });
    for i in &ind { if wide { ib.extend_from_slice(&(*i as u16).to_le_bytes()); } else { ib.push(*i as u8); } }
    let mut m = json!({ "box": [r3(lo.x), r3(lo.y), r3(lo.z), r3(hi.x), r3(hi.y), r3(hi.z)], "n": verts.len(), "tris": ind.len() / 3, "ib": if wide { 16 } else { 8 }, "v": b64(&vb), "i": b64(&ib), "parts": pj });
    if has_uv {
        let mut ub = Vec::with_capacity(verts.len() * 4);
        for v in &verts { for c in v.1 { ub.extend_from_slice(&c.to_le_bytes()); } }
        m["uv"] = json!(b64(&ub)); m["uvb"] = json!([u0, v0, u1, v1]);
    }
    Some(m)
}

/// A translucent cool-tinted hull panel: how the engine's effect-drawn volumes (shield doors,
/// forcefields) are shown, since their render model draws nothing.
const SHIELD_TINT: [f32; 3] = [0.35, 0.75, 0.95];
const SHIELD_OPACITY: f32 = 0.30;

/// Choose and simplify the model of one object tag:
///  - an EFFECT-DRAWN volume (render model of <= 4 triangles, the engine's invisible-wall rule, with
///    a physics hull and an effect attachment) becomes a translucent shield panel from its hull;
///  - a render model whose box agrees with the entry's box is used part by part (material lanes kept);
///  - otherwise the physics hull as one opaque part.
fn choose_model(scene: &dyn ObjectScene, tag: u32, info: &RenderInfo, max_tris: usize) -> Option<Model> {
    let o = ObjectInfo {
        datum: 0, type_sig: 0, sig0: 0, sig1: 0, pos: [0.0; 3], health: 1.0, shield: 1.0, mode_tag: 0,
        fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0,
    };
    let hull = scene.physics_world_tris(&o);
    let hull_parts = |lane: &'static str, color: Option<[f32; 3]>, opacity: f32| -> Vec<Part> {
        let tris: Vec<Tri> = hull.iter().map(|t| Tri { p: *t, uv: [[0.0; 2]; 3] }).collect();
        simplify(vec![Part { tris, lane, color, opacity, tex: None, halogram: lane == "blend" && color.is_some() }], max_tris)
    };
    if info.effects && info.model_tris <= 4 && !hull.is_empty() {
        return Some(Model { parts: hull_parts("blend", Some(SHIELD_TINT), SHIELD_OPACITY), cat: Some("shield") });
    }
    let ref_box = local_box(scene, tag);
    // a forcefield quad (the shield doors are a 4-triangle plane with a forcefield material) is a real
    // visual even though it is tiny; an opaque model of <= 4 triangles is the invisible-wall rule
    let ff = !info.parts.is_empty() && info.parts.iter().all(|p| p.halogram);
    let use_render = (info.model_tris > 4 || ff) && ff || info.model_tris > 4 && match ref_box {
        None => true,
        Some((bl, bh)) => {
            let (rl, rh) = part_box(&info.parts);
            let (re, he) = (rh - rl, Vec3::from(bh) - Vec3::from(bl));
            (0..3).all(|k| (re[k] - he[k]).abs() <= 0.3 * he[k].max(0.25) + 0.08)
        }
    };
    if use_render {
        let parts = simplify(info.parts.clone(), max_tris);
        // a model counts as glass / shield only when most of it is translucent: a rifle with one
        // additive scope lens is still a weapon
        let total: usize = parts.iter().map(|p| p.tris.len()).sum();
        let translucent: usize = parts.iter().filter(|p| p.lane == "blend" || p.lane == "additive").map(|p| p.tris.len()).sum();
        let halo: usize = parts.iter().filter(|p| p.halogram && (p.lane == "blend" || p.lane == "additive")).map(|p| p.tris.len()).sum();
        let additive: usize = parts.iter().filter(|p| p.lane == "additive").map(|p| p.tris.len()).sum();
        let cat = if total > 0 && translucent * 10 >= total * 6 { if halo * 2 >= translucent { Some("shield") } else if additive * 2 >= translucent { Some("glow") } else { Some("glass") } } else { None };
        return Some(Model { parts, cat });
    }
    if hull.is_empty() { return None; }
    Some(Model { parts: hull_parts("opaque", None, 1.0), cat: None })
}

/// `render_info(obj_tag)` supplies an object's render-model parts and whether it attaches an effect
/// (an empty `RenderInfo` on Halo 4: hull / box only). Models are generated at `MESH_TRIS` triangles
/// and re-generated smaller until the whole pack fits `PACK_BUDGET`; the tiny textures go first.
pub fn build_pack(game: &str, map_id: u32, bounds: [f32; 6], scene: &dyn ObjectScene, items: &[PackItem], render_info: &dyn Fn(u32) -> RenderInfo) -> Value {
    let mut infos: HashMap<u32, RenderInfo> = HashMap::new();
    for it in items { for t in std::iter::once(it.obj_tag).chain(it.variants.iter().map(|v| v.2)) { infos.entry(t).or_insert_with(|| render_info(t)); } }
    let mut budget = MESH_TRIS;
    loop {
        for tex_n in [32usize, 16, 0] {
            let pack = build_pack_at(game, map_id, bounds, scene, items, &infos, budget, tex_n);
            if pack.to_string().len() <= PACK_BUDGET { return pack; }
        }
        if budget <= 24 { return build_pack_at(game, map_id, bounds, scene, items, &infos, budget, 0); }
        budget = (budget * 2 / 3).max(24);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pack_at(game: &str, map_id: u32, bounds: [f32; 6], scene: &dyn ObjectScene, items: &[PackItem], infos: &HashMap<u32, RenderInfo>, max_tris: usize, tex_n: usize) -> Value {
    let mut models: HashMap<u32, Option<Model>> = HashMap::new();
    let empty = RenderInfo::default();
    let mut texs: Vec<Vec<u8>> = Vec::new();
    let mut by_content: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut tex_index: HashMap<usize, usize> = HashMap::new();
    let mut get = |t: u32, models: &mut HashMap<u32, Option<Model>>| -> Option<(Value, Option<&'static str>)> {
        if !models.contains_key(&t) { models.insert(t, choose_model(scene, t, infos.get(&t).unwrap_or(&empty), max_tris)); }
        let m = models.get(&t).unwrap().as_ref()?;
        if tex_n > 0 { for p in &m.parts { if let Some(tx) = &p.tex { let k = std::sync::Arc::as_ptr(tx) as usize; if !tex_index.contains_key(&k) { let small = tex_small(tx, tex_n); let id = *by_content.entry(small.clone()).or_insert_with(|| { texs.push(small); texs.len() - 1 }); tex_index.insert(k, id); } } } }
        Some((encode_model(&m.parts, &tex_index)?, m.cat))
    };
    let palette: Vec<Value> = items.iter().map(|it| {
        let mut v = json!({ "i": it.index, "name": it.name, "cat": category_word(&it.class, &it.category, &it.name) });
        let bx = |t: u32| local_box(scene, t).map(|(mn, mx)| json!([r3(mn[0]), r3(mn[1]), r3(mn[2]), r3(mx[0]), r3(mx[1]), r3(mx[2])]));
        if let Some(b) = bx(it.obj_tag) { v["box"] = b; }
        if let Some((m, cat)) = get(it.obj_tag, &mut models) { v["mesh"] = m; if let Some(c) = cat { if !matches!(v["cat"].as_str(), Some("zone") | Some("spawn")) { v["cat"] = json!(c); } } }
        if it.variants.len() > 1 {
            v["variants"] = Value::Array(it.variants.iter().map(|(vi, vn, t)| {
                let mut e = json!({ "v": vi, "name": vn });
                if let Some(b) = bx(*t) { e["box"] = b; }
                if let Some((m, cat)) = get(*t, &mut models) { e["mesh"] = m; if let Some(c) = cat { e["cat"] = json!(c); } }
                e
            }).collect());
        }
        v
    }).collect();
    let tex_json: Vec<Value> = texs.iter().map(|t| json!({ "w": tex_n, "h": tex_n, "rgba": b64(t) })).collect();
    let mut pack = json!({
        "format": "hms-render-pack", "version": 2, "meshTris": max_tris,
        "generator": format!("Halo Map Studio {}", env!("CARGO_PKG_VERSION")),
        "game": game, "mapId": map_id,
        "note": "Derived from a locally installed game. For your own use: do not redistribute.",
        "bounds": bounds.iter().map(|b| r3(*b)).collect::<Vec<_>>(),
        "palette": palette,
        "grid": height_grid(scene, bounds),
    });
    if !tex_json.is_empty() { pack["textures"] = Value::Array(tex_json); }
    pack
}

/// Sanity facts about a pack, shared by the verb's reply and the gate.
pub fn summary(pack: &Value) -> String {
    let pal = pack["palette"].as_array().map_or(0, |a| a.len());
    let boxed = pack["palette"].as_array().map_or(0, |a| a.iter().filter(|e| e.get("box").is_some()).count());
    let meshed = pack["palette"].as_array().map_or(0, |a| a.iter().filter(|e| e.get("mesh").is_some()).count());
    let mut lanes: std::collections::BTreeMap<String, usize> = Default::default();
    let mut shields: Vec<String> = Vec::new();
    for e in pack["palette"].as_array().into_iter().flatten() {
        let ents = std::iter::once(e).chain(e.get("variants").and_then(|v| v.as_array()).into_iter().flatten());
        for x in ents {
            for p in x["mesh"]["parts"].as_array().into_iter().flatten() { *lanes.entry(p["lane"].as_str().unwrap_or("?").to_string()).or_default() += 1; }
            if x["cat"] == "shield" || x["cat"] == "glass" { shields.push(format!("{}/{}:{}", e["name"].as_str().unwrap_or(""), x["name"].as_str().unwrap_or(""), x["cat"].as_str().unwrap_or(""))); }
        }
    }
    let tex = pack["textures"].as_array().map_or(0, |a| a.len());
    format!("render pack v2: {pal} palette entries ({boxed} with a box, {meshed} with a mesh, <= {} triangles each), parts by lane {lanes:?}, {tex} textures, grid {GRID}x{GRID}, {} bytes\ntranslucent entries ({}): {}", pack["meshTris"], pack.to_string().len(), shields.len(), shields.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::objscene::ObjectScene as _;

    #[test]
    fn base64_matches_the_standard_vectors() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foob"), "Zm9vYg==");
        assert_eq!(b64(&[0xFF, 0xFE, 0xFD]), "//79");
    }

    #[test]
    fn category_words() {
        assert_eq!(category_word("weap", "WEAPONS", "Sniper"), "weapon");
        assert_eq!(category_word("vehi", "", "Warthog"), "vehicle");
        assert_eq!(category_word("bloc", "STRUCTURES", "Wall"), "block");
        assert_eq!(category_word("scen", "", "Initial Spawn"), "spawn");
        assert_eq!(category_word("", "", "Kill Boundary"), "zone");
    }

    /// Map-backed gate (run alone: `cargo test --profile quick -p hms-app render_pack_gate -- --ignored
    /// --nocapture --test-threads=1`). Builds the pack for Forge World with the SAME function the
    /// `variant get render-pack` verb calls, writes it to `HMS_RENDER_PACK_OUT` when set (never into
    /// the repo), and checks size, coverage, a known block's proportions and that the height grid
    /// agrees with where a real variant's objects stand.
    #[test]
    #[ignore = "opens the native library and a Forge World cache; run alone with --ignored --nocapture"]
    fn render_pack_gate_forge_world() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("render_pack gate SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        sc.load_collision_for_test();
        let palette = sc.forge_palette_full();
        let types = sc.forge_type_order(&palette);
        let items: Vec<PackItem> = types.iter().enumerate().map(|(i, &(pi, ew))| {
            let e = palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).expect("palette entry");
            let variants = palette.iter().filter(|p| p.palette_index == pi && p.entry_within == ew).map(|p| (p.variant_within, if p.variant_name.is_empty() { sc.tag_name_of(p.tag_short).rsplit('\\').next().unwrap_or("").to_string() } else { p.variant_name.clone() }, p.tag_short)).collect();
            PackItem { index: i, name: e.name.clone(), category: e.category_name.clone(), class: sc.raw_tag_class(e.tag_short).unwrap_or_default(), obj_tag: e.tag_short, variants }
        }).collect();
        // a real Forge World variant supplies the world bounds and objects to check against
        let v = crate::film_export::known_map_variants().into_iter().filter(|(_, id)| *id == 3006)
            .filter_map(|(p, _)| crate::mvar::parse_variant(&p)).max_by_key(|v| v.objects.len()).expect("a Forge World .mvar on this machine");
        let pack = build_pack("reach", v.map_id, v.globals.bounds, &sc, &items, &|t| sc.render_info_for_pack(t));
        let text = pack.to_string();
        eprintln!("{}", summary(&pack));
        if let Ok(out) = std::env::var("HMS_RENDER_PACK_OUT") { std::fs::write(&out, &text).expect("write pack"); }

        let mut slim = pack.clone();
        slim.as_object_mut().unwrap().remove("textures");
        for e in slim["palette"].as_array_mut().unwrap() {
            e.as_object_mut().unwrap().remove("mesh");
            if let Some(vs) = e.get_mut("variants").and_then(|v| v.as_array_mut()) { for v in vs { v.as_object_mut().unwrap().remove("mesh"); } }
        }
        assert!(slim.to_string().len() <= 200_000, "boxes + grid alone are {} bytes, target <= 200 KB", slim.to_string().len());
        assert!(text.len() <= PACK_BUDGET, "pack is {} bytes, budget {PACK_BUDGET}", text.len());
        let pal = pack["palette"].as_array().unwrap();
        assert_eq!(pal.len(), types.len(), "one entry per palette type");
        assert!(pal.len() >= 100, "Forge World has hundreds of types, got {}", pal.len());
        let boxed: Vec<&Value> = pal.iter().filter(|e| e.get("box").is_some()).collect();
        assert!(boxed.len() * 10 >= pal.len() * 8, "only {} of {} entries have a box", boxed.len(), pal.len());
        // every box is a proper, finite, non-degenerate-or-flat AABB
        for e in &boxed {
            let b: Vec<f64> = e["box"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            assert!(b[3] >= b[0] && b[4] >= b[1] && b[5] >= b[2], "inverted box {e}");
        }
        // known proportions: the 5x5 platform is 5/4 the width of the 4x4 one, both square
        let dims = |suffix: &str| -> Option<(f64, f64, f64)> {
            let tag = native.tags_by_group(cache, "bloc").into_iter().find(|t| native.tag_name(cache, *t).is_some_and(|n| n.ends_with(suffix)))?;
            let it = items.iter().find(|it| it.obj_tag == tag)?;
            let e = pal.iter().find(|e| e["i"] == it.index)?;
            let b: Vec<f64> = e["box"].as_array()?.iter().map(|x| x.as_f64().unwrap()).collect();
            Some((b[3] - b[0], b[4] - b[1], b[5] - b[2]))
        };
        // known proportions through the VARIANTS of the structure_blocks entry: ff_plat_1x1 is a
        // 1.016 m cube, and the NxN platforms scale with N (square, same height class)
        let blocks = pal.iter().find(|e| e["name"] == "structure_blocks").expect("structure_blocks entry");
        let vs = blocks["variants"].as_array().expect("structure_blocks lists its variants");
        let size = |name: &str| -> Option<[f64; 3]> {
            let e = vs.iter().find(|e| e["name"].as_str().is_some_and(|n| n == name))?;
            let b: Vec<f64> = e["box"].as_array()?.iter().map(|x| x.as_f64().unwrap()).collect();
            Some([b[3] - b[0], b[4] - b[1], b[5] - b[2]])
        };
        eprintln!("structure_blocks variants: {:?}", vs.iter().map(|e| e["name"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>());
        let one = size("block_1x1").expect("block_1x1 box");
        assert!((one[0] - one[1]).abs() < 0.02 && (one[1] - one[2]).abs() < 0.02 && (one[0] - 1.0).abs() < 0.05, "block_1x1 should be a ~1 unit cube: {one:?}");
        let four = size("block_4x4").expect("block_4x4 box");
        eprintln!("block sizes: 1x1 {one:?} 4x4 {four:?}");
        let mut f = four; f.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((f[2] / one[0] - 4.0).abs() < 0.3 && (f[1] / one[0] - 4.0).abs() < 0.3, "block_4x4 is not ~4x the 1x1 in two dimensions: {four:?}");
        let grid = pal.iter().find(|e| e["name"] == "ff_grid").expect("grid entry")["box"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
        assert!((grid[3] - grid[0] - 30.0).abs() < 0.5, "the Forge grid object is 30 wide: {grid:?}");
        // ---- v2 meshes: counts, budget, decode round trip, hull == known box, render mesh == AABB
        let b64d = |s: &str| -> Vec<u8> {
            let t = |c: u8| -> u32 { match c { b'A'..=b'Z' => (c - b'A') as u32, b'a'..=b'z' => (c - b'a') as u32 + 26, b'0'..=b'9' => (c - b'0') as u32 + 52, b'+' => 62, _ => 63 } };
            let mut out = Vec::new();
            for ch in s.as_bytes().chunks(4) {
                let n = (t(ch[0]) << 18) | (t(ch[1]) << 12) | (if ch[2] == b'=' { 0 } else { t(ch[2]) << 6 }) | (if ch[3] == b'=' { 0 } else { t(ch[3]) });
                out.push((n >> 16) as u8);
                if ch[2] != b'=' { out.push((n >> 8) as u8); }
                if ch[3] != b'=' { out.push(n as u8); }
            }
            out
        };
        let decode_mesh = |m: &Value| -> (Vec<[f32; 3]>, Vec<[usize; 3]>, [f32; 6]) {
            let bx: Vec<f32> = m["box"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let vb = b64d(m["v"].as_str().unwrap());
            let n = m["n"].as_u64().unwrap() as usize;
            assert_eq!(vb.len(), n * 6, "vertex bytes");
            let verts: Vec<[f32; 3]> = (0..n).map(|i| {
                let c = |k: usize| i16::from_le_bytes([vb[i * 6 + k * 2], vb[i * 6 + k * 2 + 1]]) as f32;
                let f = |k: usize| bx[k] + (c(k) + 32768.0) / 65535.0 * (bx[k + 3] - bx[k]);
                [f(0), f(1), f(2)]
            }).collect();
            let ib = b64d(m["i"].as_str().unwrap());
            let idx: Vec<usize> = if m["ib"] == 16 { ib.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as usize).collect() } else { ib.iter().map(|&b| b as usize).collect() };
            assert_eq!(idx.len() % 3, 0);
            assert!(idx.iter().all(|&i| i < n), "index out of range");
            (verts, idx.chunks(3).map(|t| [t[0], t[1], t[2]]).collect(), [bx[0], bx[1], bx[2], bx[3], bx[4], bx[5]])
        };
        let with_mesh = pal.iter().filter(|e| e.get("mesh").is_some()).count();
        eprintln!("meshes: {with_mesh}/{} entries, pack {} bytes, budget {} tris", pal.len(), text.len(), pack["meshTris"]);
        assert!(text.len() <= PACK_BUDGET, "pack over the 2.5 MB budget: {}", text.len());
        assert!(with_mesh * 10 >= pal.len() * 9, "only {with_mesh} of {} entries have a mesh", pal.len());
        let (mut render_picked, mut total_tris) = (0, 0usize);
        for e in pal.iter() {
            let ms: Vec<&Value> = std::iter::once(e.get("mesh")).chain(e.get("variants").and_then(|v| v.as_array()).into_iter().flatten().map(|v| v.get("mesh"))).flatten().collect();
            for m in ms {
                let (v, t, _) = decode_mesh(m);
                assert!(t.len() <= MESH_TRIS && !t.is_empty(), "triangle budget: {}", t.len());
                assert!(v.iter().flatten().all(|c| c.is_finite()));
                total_tris += t.len();
            }
        }
        // a block's mesh IS its hull: the decoded mesh box equals the entry's known box (+/- quantisation)
        let b1 = vs.iter().find(|e| e["name"] == "block_1x1").unwrap();
        let (_, t1, mb) = decode_mesh(&b1["mesh"]);
        let bb: Vec<f32> = b1["box"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        for k in 0..6 { assert!((mb[k] - bb[k]).abs() < 0.03, "block_1x1 mesh box {mb:?} != box {bb:?}"); }
        assert!(t1.len() >= 12, "a cube hull has 12 triangles, got {}", t1.len());
        // vehicles / weapons: the mesh's bounding box agrees with the entry's AABB (extents within 35%),
        // and some of them came from the render model (their mesh box differs from the hull box)
        let mut vw = 0;
        for it in items.iter().filter(|it| it.class == "vehi" || it.class == "weap") {
            let e = pal.iter().find(|e| e["i"] == it.index).unwrap();
            let (Some(m), Some(bx)) = (e.get("mesh"), e.get("box")) else { continue };
            let (_, _, mb) = decode_mesh(m);
            let bb: Vec<f32> = bx.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            for k in 0..3 {
                let (me, be) = (mb[k + 3] - mb[k], bb[k + 3] - bb[k]);
                assert!((me - be).abs() <= 0.4 * be.max(0.25) + 0.1, "{}: mesh extent {me} vs AABB {be} on axis {k}", it.name);
            }
            vw += 1;
            if (0..6).any(|k| (mb[k] - bb[k]).abs() > 0.02) { render_picked += 1; }
        }
        eprintln!("vehicle/weapon entries with a mesh: {vw}, of which render-model meshes: {render_picked}; total triangles {total_tris}");
        assert!(vw >= 8 && render_picked >= 3, "expected several vehicle/weapon meshes from the render model ({render_picked}/{vw})");
        // ---- translucency: classified from tag data, reported for eyeballing
        let lookup = |needle: &str| -> Option<&Value> {
            for it in items.iter() {
                let e = pal.iter().find(|e| e["i"] == it.index)?;
                if it.variants.len() > 1 {
                    for (vi, _, t) in &it.variants { if sc.tag_name_of(*t).ends_with(needle) { return e["variants"].as_array()?.iter().find(|x| x["v"] == *vi); } }
                } else if sc.tag_name_of(it.obj_tag).ends_with(needle) { return Some(e); }
            }
            None
        };
        let lanes_of = |x: &Value| -> Vec<String> { x["mesh"]["parts"].as_array().map(|a| a.iter().map(|p| p["lane"].as_str().unwrap_or("?").to_string()).collect()).unwrap_or_default() };
        for it in items.iter() {
            let e = pal.iter().find(|e| e["i"] == it.index).unwrap();
            let vs: Vec<(&Value, u32)> = if it.variants.len() > 1 { it.variants.iter().filter_map(|(vi, _, t)| e["variants"].as_array().and_then(|a| a.iter().find(|x| x["v"] == *vi)).map(|x| (x, *t))).collect() } else { vec![(e, it.obj_tag)] };
            for (x, t) in vs { if x["cat"] == "shield" || x["cat"] == "glass" { eprintln!("TRANSLUCENT {} | {} | {} | lanes {:?}", x["cat"].as_str().unwrap(), sc.tag_name_of(t), it.name, lanes_of(x)); } }
        }
        for n in ["ff_two_way_door_medium\\ff_two_way_door_medium", "ff_one_way_door_medium\\ff_one_way_door_medium", "ff_shield_door_medium\\ff_shield_door_medium"] {
            for it in items.iter() { for (_, _, t) in &it.variants { if sc.tag_name_of(*t).ends_with(n) { let i = sc.render_info_for_pack(*t); eprintln!("DOORINFO {n}: model_tris {} effects {} parts {:?}", i.model_tris, i.effects, i.parts.iter().map(|p| (p.lane, p.tris.len(), p.halogram)).collect::<Vec<_>>()); } } }
        }
        let opaque_share = |x: &Value| -> f64 { let ps = x["mesh"]["parts"].as_array().cloned().unwrap_or_default(); let tot: f64 = ps.iter().map(|p| p["n"].as_f64().unwrap_or(0.0)).sum(); let op: f64 = ps.iter().filter(|p| p["lane"] == "opaque" || p["lane"] == "cutout").map(|p| p["n"].as_f64().unwrap_or(0.0)).sum(); if tot > 0.0 { op / tot } else { 0.0 } };
        let two = lookup("ff_two_way_door_medium\\ff_two_way_door_medium").expect("two-way shield door in the palette");
        let one = lookup("ff_one_way_door_medium\\ff_one_way_door_medium").expect("one-way shield door in the palette");
        assert_eq!(two["cat"], "shield", "two-way shield door is classified shield"); assert_eq!(one["cat"], "shield", "one-way shield door is classified shield");
        assert!(lanes_of(two).iter().all(|l| l == "blend" || l == "additive") && !lanes_of(two).is_empty(), "two-way door parts are translucent: {:?}", lanes_of(two));
        assert!(lanes_of(one).iter().all(|l| l == "blend" || l == "additive") && !lanes_of(one).is_empty(), "one-way door parts are translucent: {:?}", lanes_of(one));
        let blk = lookup("ff_plat_1x1\\ff_plat_1x1").or_else(|| vs.iter().find(|e| e["name"] == "block_1x1")).expect("a solid block");
        assert!(blk["cat"] != "shield" && blk["cat"] != "glass" && opaque_share(blk) >= 0.6, "a solid block stays opaque: {:?} share {}", lanes_of(blk), opaque_share(blk));
        // the heavy barricade (its tag path contains "shield" but it is a solid wall): found by tag, judged by its parts
        let mut hv: Vec<&Value> = Vec::new();
        for it in items.iter() { for (vi, _, t) in &it.variants { if sc.tag_name_of(*t).contains("heavy_shield") { if let Some(x) = pal.iter().find(|e| e["i"] == it.index).and_then(|e| e["variants"].as_array()).and_then(|a| a.iter().find(|x| x["v"] == *vi)) { eprintln!("HEAVY {} -> cat {} lanes {:?}", sc.tag_name_of(*t), x["cat"], lanes_of(x)); hv.push(x); } } } }
        assert!(!hv.is_empty(), "the heavy barricade variant exists");
        for x in &hv { assert!(x["cat"] != "shield" && x["cat"] != "glass" && opaque_share(x) >= 0.6, "the heavy barricade stays opaque (it is NOT a shield): {:?}", lanes_of(x)); }
        let mut all_lanes: std::collections::BTreeMap<String, usize> = Default::default();
        for e in pal.iter() { for x in std::iter::once(e).chain(e.get("variants").and_then(|v| v.as_array()).into_iter().flatten()) { for l in lanes_of(x) { *all_lanes.entry(l).or_default() += 1; } } }
        eprintln!("parts by lane: {all_lanes:?}; textures {}", pack["textures"].as_array().map_or(0, |a| a.len()));
        assert!(all_lanes.get("blend").copied().unwrap_or(0) + all_lanes.get("additive").copied().unwrap_or(0) >= 4, "Forge World has translucent parts");
        // the height grid is not empty, and the variant's own objects stand on / above it
        let g = &pack["grid"];
        let dec = |k: &str| -> Vec<u8> {
            let s = g[k].as_str().unwrap();
            let t = |c: u8| -> u32 { match c { b'A'..=b'Z' => (c - b'A') as u32, b'a'..=b'z' => (c - b'a') as u32 + 26, b'0'..=b'9' => (c - b'0') as u32 + 52, b'+' => 62, _ => 63 } };
            let mut out = Vec::new();
            for ch in s.as_bytes().chunks(4) {
                let n = (t(ch[0]) << 18) | (t(ch[1]) << 12) | (if ch[2] == b'=' { 0 } else { t(ch[2]) << 6 }) | (if ch[3] == b'=' { 0 } else { t(ch[3]) });
                out.push((n >> 16) as u8);
                if ch[2] != b'=' { out.push((n >> 8) as u8); }
                if ch[3] != b'=' { out.push(n as u8); }
            }
            out
        };
        let (lo, hi) = (dec("min"), dec("max"));
        assert_eq!((lo.len(), hi.len()), (GRID * GRID, GRID * GRID));
        let filled = lo.iter().filter(|&&c| c != 255).count();
        assert!(filled * 100 >= GRID * GRID * 5, "grid nearly empty: {filled} of {} cells", GRID * GRID);
        let bd = v.globals.bounds;
        let (zmin, zmax) = (bd[4] as f64, bd[5] as f64);
        let (mut checked, mut ok) = (0, 0);
        for o in &v.objects {
            let gx = ((o.pos[0] - bd[0]) / (bd[1] - bd[0]) * GRID as f32) as isize;
            let gy = ((o.pos[1] - bd[2]) / (bd[3] - bd[2]) * GRID as f32) as isize;
            if gx < 0 || gy < 0 || gx >= GRID as isize || gy >= GRID as isize { continue; }
            let c = lo[gy as usize * GRID + gx as usize];
            if c == 255 { continue; }
            checked += 1;
            let floor = zmin + c as f64 / 254.0 * (zmax - zmin);
            if floor - 3.0 <= o.pos[2] as f64 { ok += 1; }
        }
        eprintln!("grid vs {} variant objects: {ok}/{checked} stand at or above their cell's lowest collision Z", v.objects.len());
        // orientation / indexing: the cell under each of 150 objects holds exactly what a direct
        // query of the same cell finds (a transposed or flipped grid cannot pass this)
        let (cw, ch) = ((bd[1] - bd[0]) / GRID as f32, (bd[3] - bd[2]) / GRID as f32);
        let (mut same, mut tested) = (0, 0);
        for o in v.objects.iter().step_by((v.objects.len() / 150).max(1)) {
            let gx = (((o.pos[0] - bd[0]) / cw) as isize).clamp(0, GRID as isize - 1) as usize;
            let gy = (((o.pos[1] - bd[2]) / ch) as isize).clamp(0, GRID as isize - 1) as usize;
            let (x0, y0) = (bd[0] + gx as f32 * cw, bd[2] + gy as f32 * ch);
            let mut tris = Vec::new();
            sc.static_tris_in_aabb(Vec3::new(x0, y0, bd[4] - 1.0), Vec3::new(x0 + cw, y0 + ch, bd[5] + 1.0), &mut tris);
            let mut a = f32::MAX;
            for t in &tris {
                if t.iter().map(|p| p.x).fold(f32::MIN, f32::max) < x0 || t.iter().map(|p| p.x).fold(f32::MAX, f32::min) > x0 + cw { continue; }
                if t.iter().map(|p| p.y).fold(f32::MIN, f32::max) < y0 || t.iter().map(|p| p.y).fold(f32::MAX, f32::min) > y0 + ch { continue; }
                for p in t { a = a.min(p.z); }
            }
            let want = if a == f32::MAX { 255 } else { (((a.clamp(bd[4], bd[5]) - bd[4]) / (bd[5] - bd[4]) * 254.0).round()) as i32 };
            tested += 1;
            if (lo[gy * GRID + gx] as i32 - want).abs() <= 1 { same += 1; }
        }
        eprintln!("grid cells equal to a direct query: {same}/{tested}");
        assert!(same * 100 >= tested * 98, "grid cell != direct query of the same cell ({same}/{tested}): wrong orientation or indexing");
        assert!(checked >= 50 && ok * 10 >= checked * 8, "the height grid disagrees with the variant's objects ({ok}/{checked}): wrong orientation?");
    }
}
