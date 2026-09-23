//! Decal projection: turns scenario decal boxes into conformed receiver geometry over the
//! BSP triangle soup, the same way the engine builds them at map load.
//!
//! Reach authors a scenario decal as an oriented box (position + quat -> outward normal, in-plane
//! U/V, half extents). The engine turns it into geometry at map load by casting ONE short
//! collision ray into the receiver and flood-filling edge-connected surfaces from the hit (cull
//! angle, clamp-angle folds) -- see `project_engine` below and docs/hrek_re/21_re_decals.md.
//!
//! This module holds the world-space triangle soup with a uniform spatial grid (`TriSoup`:
//! receivers for decals, and the terrain/BSP collision every placement raycast uses), the
//! per-load edge adjacency (`EdgeIndex`) and the engine-rule projector (`project_engine`).

use glam::Vec3;

/// World-space BSP triangle soup with a uniform spatial grid for near-queries.
pub struct TriSoup {
    tris: Vec<[Vec3; 3]>,
    /// per-triangle (diffuse tag, blend) so a ray-pick can report the ACTUAL material of
    /// the surface under the cursor. Parallel to `tris`. Without this, material picking used
    /// mesh AABBs — and a big enclosing mesh (skybox/backdrop) always won the nearest-AABB test,
    /// so every click returned the same material regardless of where you clicked.
    tri_mat: Vec<(u32, u32, i32)>,
    /// per-triangle owner word, parallel to `tris`: bits 0..30 = the owning mesh's
    /// ordinal, `MESH_INSTANCE` = the mesh is instanced geometry (vs structure cluster geometry),
    /// `MESH_NO_DECAL` = not a decal receiver (transparent blend / glass). `u32::MAX` = unknown.
    tri_mesh: Vec<u32>,
    cell: f32,
    grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
    /// The grid in COMPACT form once the soup is complete (`shrink`): cells sorted by key
    /// with (start, len) into one flat `refs` list, instead of one heap Vec per cell (Forge World:
    /// 2.27M cells -> 2.27M small allocations + 36 B of HashMap entry each). Lookups binary-search
    /// `cells`; results are identical to the HashMap (same per-cell index lists, same order).
    cells: Vec<((i32, i32, i32), u32, u32)>,
    refs: Vec<u32>,
    /// (x, y) column -> its contiguous run in `cells` (sorted by (x, y, z)), so a lookup is
    /// one hash probe + a binary search over the handful of z cells of that column instead of a
    /// binary search over every cell (a ray march does one lookup per cell it crosses).
    cols: std::collections::HashMap<(i32, i32), (u32, u32)>,
    /// Triangles whose AABB spans more grid cells than `BIG_TRI_CELLS` (map-spanning
    /// backdrop / skybox / huge terrain quads). Filling their whole cell-AABB was the load
    /// bottleneck — one such tri touches tens of thousands of cells, and there are enough of
    /// them that `add_tri` spent ~70s of a 92s Ivory Tower load just doing HashMap inserts.
    /// Instead of exploding the grid, they live here and EVERY query (sphere + ray) tests them
    /// unconditionally. They stay few (dozens), so the always-test cost is negligible and
    /// coverage is EXACT — a query can never miss a big tri the way a strided/skipped grid would.
    big: Vec<u32>,
}

/// Max grid-cell AABB volume a triangle may fill before it's treated as a "big" tri and
/// diverted out of the grid. At cell=16 this is ~a 160-unit cube of cells; normal wall/floor
/// tris fall well under it, only genuine backdrops exceed it.
const BIG_TRI_CELLS: i64 = 1024;

impl TriSoup {
    pub fn new(cell: f32) -> Self {
        Self { tris: Vec::new(), tri_mat: Vec::new(), tri_mesh: Vec::new(), cell: cell.max(1.0), grid: std::collections::HashMap::new(), cells: Vec::new(), refs: Vec::new(), cols: std::collections::HashMap::new(), big: Vec::new() }
    }

    /// The triangle indices of one grid cell -- from the compact sorted form when built,
    /// else from the build-time HashMap. Empty slice when the cell holds nothing.
    #[inline]
    fn cell_refs(&self, key: (i32, i32, i32)) -> &[u32] {
        if !self.cells.is_empty() {
            let Some(&(cs, cn)) = self.cols.get(&(key.0, key.1)) else { return &[] };
            let col = &self.cells[cs as usize..(cs + cn) as usize];
            return match col.binary_search_by(|c| c.0 .2.cmp(&key.2)) {
                Ok(i) => { let (_, s, n) = col[i]; &self.refs[s as usize..(s + n) as usize] }
                Err(_) => &[],
            };
        }
        self.grid.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn len(&self) -> usize { self.tris.len() }
    /// Diag: (grid cell size, number of out-of-grid "big" triangles).
    pub fn cell_and_big(&self) -> (f32, usize) { (self.cell, self.big.len()) }
    pub fn is_empty(&self) -> bool { self.tris.is_empty() }
    /// (grid cells, total tri refs, total ref capacity) -- how much of the soup is the grid.
    pub fn grid_stats(&self) -> (usize, usize, usize) {
        if !self.cells.is_empty() {
            return (self.cells.len(), self.refs.len(), self.refs.capacity());
        }
        (self.grid.len(), self.grid.values().map(|v| v.len()).sum(), self.grid.values().map(|v| v.capacity()).sum())
    }
    /// Undo `shrink`'s compaction (only if a triangle is added after the soup was sealed --
    /// never on the load path, kept so a late `add_tri` can never be silently dropped).
    fn decompact(&mut self) {
        if self.cells.is_empty() { return; }
        for &(key, s, n) in &self.cells {
            self.grid.insert(key, self.refs[s as usize..(s + n) as usize].to_vec());
        }
        self.cells = Vec::new();
        self.refs = Vec::new();
        self.cols = std::collections::HashMap::new();
    }
    /// Coarsen the grid when the map's triangles are large relative to the cell. The cell is
    /// fixed at build time (4.0 wu, tuned for conform speed on typical BSPs), but a map of big flat
    /// panels (cato_nemodia: 271k tris -> 10.9M cell refs, 232 MB) puts every tri in dozens of
    /// cells. Doubling the cell while the average refs/tri is above `max_refs_per_tri` bounds the
    /// grid at a small multiple of the tri count. Query RESULTS are unchanged -- candidates are
    /// always exact-tested (ray/box) after gathering -- only the candidate set per query grows.
    pub fn rebalance(&mut self, max_refs_per_tri: usize) {
        const MAX_CELL: f32 = 64.0;
        if self.tris.len() < 1024 {
            return;
        }
        self.decompact(); // Rebalance works on the build-time map
        // How many cell refs a given cell size would produce (pure key arithmetic, no inserts).
        let refs_at = |cell: f32| -> usize {
            let key = |p: Vec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
            self.tris.iter().map(|[a, b, c]| {
                let (k0, k1) = (key(a.min(*b).min(*c)), key(a.max(*b).max(*c)));
                let span = (k1.0 - k0.0 + 1) * (k1.1 - k0.1 + 1) * (k1.2 - k0.2 + 1);
                if span > BIG_TRI_CELLS { 0 } else { span as usize }
            }).sum()
        };
        let limit = max_refs_per_tri * self.tris.len();
        let current: usize = self.grid.values().map(|v| v.len()).sum();
        if current <= limit {
            return;
        }
        let mut cell = self.cell;
        while cell < MAX_CELL && refs_at(cell) > limit {
            cell *= 2.0;
        }
        if cell == self.cell {
            return;
        }
        // One rebuild at the chosen cell (the iterative doubling rebuilt the grid per step).
        self.cell = cell;
        self.grid = std::collections::HashMap::new();
        self.big.clear();
        for idx in 0..self.tris.len() {
            let [a, b, c] = self.tris[idx];
            let (k0, k1) = (self.key(a.min(b).min(c)), self.key(a.max(b).max(c)));
            let span = (k1.0 as i64 - k0.0 as i64 + 1)
                * (k1.1 as i64 - k0.1 as i64 + 1)
                * (k1.2 as i64 - k0.2 as i64 + 1);
            if span > BIG_TRI_CELLS {
                self.big.push(idx as u32);
                continue;
            }
            for x in k0.0..=k1.0 {
                for y in k0.1..=k1.1 {
                    for z in k0.2..=k1.2 {
                        self.grid.entry((x, y, z)).or_default().push(idx as u32);
                    }
                }
            }
        }
    }
    /// Release the growth slack of every Vec once the soup is complete (the grid's per-cell
    /// index lists are built by push, so each carried up to 2x its final size).
    pub fn shrink(&mut self) {
        self.tris.shrink_to_fit();
        self.tri_mat.shrink_to_fit();
        self.tri_mesh.shrink_to_fit();
        self.big.shrink_to_fit();
        // Seal the grid into the compact sorted form (one flat refs list, 20 B per cell)
        // and drop the build-time per-cell Vecs. Same index lists, same order -> same query results.
        if self.cells.is_empty() && !self.grid.is_empty() {
            let mut keys: Vec<(i32, i32, i32)> = self.grid.keys().copied().collect();
            keys.sort_unstable();
            let total: usize = self.grid.values().map(|v| v.len()).sum();
            let mut refs: Vec<u32> = Vec::with_capacity(total);
            let mut cells: Vec<((i32, i32, i32), u32, u32)> = Vec::with_capacity(keys.len());
            for k in keys {
                if let Some(v) = self.grid.get(&k) {
                    cells.push((k, refs.len() as u32, v.len() as u32));
                    refs.extend_from_slice(v);
                }
            }
            self.grid = std::collections::HashMap::new();
            // (x, y) column index over the sorted cells.
            let mut cols: std::collections::HashMap<(i32, i32), (u32, u32)> = std::collections::HashMap::new();
            let mut i = 0usize;
            while i < cells.len() {
                let xy = (cells[i].0 .0, cells[i].0 .1);
                let mut j = i + 1;
                while j < cells.len() && (cells[j].0 .0, cells[j].0 .1) == xy { j += 1; }
                cols.insert(xy, (i as u32, (j - i) as u32));
                i = j;
            }
            self.cols = cols;
            self.cells = cells;
            self.refs = refs;
        }
        for v in self.grid.values_mut() { v.shrink_to_fit(); }
        self.grid.shrink_to_fit();
    }
    /// Approximate heap bytes (tris + materials + the spatial grid), for HMS_MEMDIAG.
    pub fn mem_bytes(&self) -> u64 {
        let grid: usize = self.grid.values().map(|v| v.capacity() * 4 + 48).sum::<usize>()
            + self.cells.capacity() * std::mem::size_of::<((i32, i32, i32), u32, u32)>()
            + self.refs.capacity() * 4
            + self.cols.capacity() * 24; // Compact form + column index
        (self.tris.capacity() * std::mem::size_of::<[Vec3; 3]>()
            + self.tri_mat.capacity() * std::mem::size_of::<(u32, u32, i32)>()
            + self.tri_mesh.capacity() * 4
            + self.big.capacity() * 4
            + grid) as u64
    }
    /// The raw world-space triangles, for building the GPU BVH.
    pub fn tris(&self) -> &[[Vec3; 3]] { &self.tris }

    /// (diffuse tag, blend, material_index) of triangle `i` — for material picking.
    pub fn tri_material(&self, i: u32) -> Option<(u32, u32, i32)> {
        self.tri_mat.get(i as usize).copied()
    }
    /// Owner word of triangle `i` (see `tri_mesh`); `u32::MAX` when never recorded.
    #[inline]
    pub fn tri_owner(&self, i: u32) -> u32 {
        self.tri_mesh.get(i as usize).copied().unwrap_or(u32::MAX)
    }

    fn key(&self, p: Vec3) -> (i32, i32, i32) {
        ((p.x / self.cell).floor() as i32, (p.y / self.cell).floor() as i32, (p.z / self.cell).floor() as i32)
    }

    /// Add a world-space triangle with its owner word (mesh ordinal | `MESH_INSTANCE` |
    /// `MESH_NO_DECAL`; `u32::MAX` = unknown).
    pub fn add_tri_owned(&mut self, a: Vec3, b: Vec3, c: Vec3, mat: (u32, u32, i32), owner: u32) {
        // Skip degenerate tris (zero area) — they can't receive a decal and pollute the grid.
        if (b - a).cross(c - a).length_squared() < 1e-8 {
            return;
        }
        self.decompact(); // no-op on the load path (the soup is sealed only after the last add)
        let idx = self.tris.len() as u32;
        self.tris.push([a, b, c]);
        self.tri_mat.push(mat);
        self.tri_mesh.push(owner);
        let mn = a.min(b).min(c);
        let mx = a.max(b).max(c);
        let (k0, k1) = (self.key(mn), self.key(mx));
        // A map-spanning tri fills a huge cell-AABB; diverting it to `big` (always tested
        // by every query) bounds the grid build without ever losing coverage.
        let span = (k1.0 as i64 - k0.0 as i64 + 1)
            * (k1.1 as i64 - k0.1 as i64 + 1)
            * (k1.2 as i64 - k0.2 as i64 + 1);
        if span > BIG_TRI_CELLS {
            self.big.push(idx);
            return;
        }
        for x in k0.0..=k1.0 {
            for y in k0.1..=k1.1 {
                for z in k0.2..=k1.2 {
                    self.grid.entry((x, y, z)).or_default().push(idx);
                }
            }
        }
    }

    /// Nearest ray-triangle hit (Möller–Trumbore, double-sided). Returns (t, geo-normal).
    ///
    /// Uses a 3D DDA (Amanatides–Woo) voxel march along the ray, testing only the grid
    /// cells the ray actually passes through, in near-to-far order, and early-outs at the
    /// first cell that starts beyond the best hit. This is O(ray_length / cell) cells, not
    /// the O(span³) grid-cube an AABB fill over the ray's extent would produce — for a scene
    /// pick ray (max_t = 1e5) that cube is ~10¹¹ empty-cell HashMap lookups, an effective
    /// hang on every click.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_t: f32) -> Option<(f32, Vec3)> {
        self.raycast_hit(origin, dir, max_t).map(|(t, n, _)| (t, n))
    }

    /// Like `raycast` but also returns the hit triangle INDEX (for material lookup).
    pub fn raycast_hit(&self, origin: Vec3, dir: Vec3, max_t: f32) -> Option<(f32, Vec3, u32)> {
        self.raycast_hit_if(origin, dir, max_t, |_| true)
    }

    /// `raycast_hit` restricted to triangles accepted by `keep(tri_index)` (the decal
    /// projector skips non-receivers / instance-vs-structure). Same march, same tie-break.
    fn raycast_hit_if(&self, origin: Vec3, dir: Vec3, max_t: f32, keep: impl Fn(u32) -> bool) -> Option<(f32, Vec3, u32)> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO || (self.grid.is_empty() && self.cells.is_empty() && self.big.is_empty()) {
            return None;
        }
        let cell = self.cell;
        // Current voxel (matches key(): floor(p / cell)). i64 so a long march can't wrap.
        let mut cx = (origin.x / cell).floor() as i64;
        let mut cy = (origin.y / cell).floor() as i64;
        let mut cz = (origin.z / cell).floor() as i64;
        let step = |d: f32| if d > 0.0 { 1i64 } else if d < 0.0 { -1i64 } else { 0i64 };
        let (sx, sy, sz) = (step(dir.x), step(dir.y), step(dir.z));
        if sx == 0 && sy == 0 && sz == 0 {
            return None;
        }
        // t to the first voxel boundary on each axis, and t between boundaries.
        let boundary = |o: f32, d: f32, c: i64| -> f32 {
            if d == 0.0 {
                return f32::INFINITY;
            }
            let b = if d > 0.0 { (c + 1) as f32 * cell } else { c as f32 * cell };
            ((b - o) / d).max(0.0)
        };
        let mut t_max_x = boundary(origin.x, dir.x, cx);
        let mut t_max_y = boundary(origin.y, dir.y, cy);
        let mut t_max_z = boundary(origin.z, dir.z, cz);
        let t_delta_x = if dir.x != 0.0 { (cell / dir.x).abs() } else { f32::INFINITY };
        let t_delta_y = if dir.y != 0.0 { (cell / dir.y).abs() } else { f32::INFINITY };
        let t_delta_z = if dir.z != 0.0 { (cell / dir.z).abs() } else { f32::INFINITY };

        let mut best: Option<(f32, Vec3, u32)> = None;
        // Big tris live outside the grid — test them once up front (they're few). Seeding
        // `best` here only makes the DDA early-out fire sooner; it's still correct because the
        // march always tests the current voxel before comparing `best` against the NEXT voxel's
        // entry t, so no closer grid hit can be skipped.
        for &i in &self.big {
            if !keep(i) { continue; }
            let [a, b, c] = self.tris[i as usize];
            let e1 = b - a;
            let e2 = c - a;
            let p = dir.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-7 { continue; }
            let inv = 1.0 / det;
            let tvec = origin - a;
            let u = tvec.dot(p) * inv;
            if u < 0.0 || u > 1.0 { continue; }
            let q = tvec.cross(e1);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 { continue; }
            let t = e2.dot(q) * inv;
            if t > 1e-4 && t < max_t && best.map_or(true, |(bt, _, bi)| t < bt || (t == bt && i < bi)) {
                best = Some((t, e1.cross(e2).normalize_or_zero(), i));
            }
        }
        // Backstop: even a pathological ray can't march more than this many voxels.
        let mut guard = 0u32;
        let max_cells = 2_000_000u32;
        loop {
            {
                let v = self.cell_refs((cx as i32, cy as i32, cz as i32));
                for &i in v {
                    if !keep(i) { continue; }
                    let [a, b, c] = self.tris[i as usize];
                    let e1 = b - a;
                    let e2 = c - a;
                    let p = dir.cross(e2);
                    let det = e1.dot(p);
                    if det.abs() < 1e-7 {
                        continue;
                    }
                    let inv = 1.0 / det;
                    let tvec = origin - a;
                    let u = tvec.dot(p) * inv;
                    if u < 0.0 || u > 1.0 {
                        continue;
                    }
                    let q = tvec.cross(e1);
                    let v = dir.dot(q) * inv;
                    if v < 0.0 || u + v > 1.0 {
                        continue;
                    }
                    let t = e2.dot(q) * inv;
                    // Deterministic tie-break on EXACT ties (coplanar duplicate surfaces): the
                    // lowest triangle index wins, so the hit does not depend on the grid cell size /
                    // traversal order (the grid is rebalanced per map -- see `rebalance`).
                    if t > 1e-4 && t < max_t && best.map_or(true, |(bt, _, bi)| t < bt || (t == bt && i < bi)) {
                        best = Some((t, e1.cross(e2).normalize_or_zero(), i));
                    }
                }
            }
            // t at which the ray enters the NEXT voxel. Any closer hit than `best` must lie
            // in a voxel we've already tested (its hit point's voxel is entered at t < that
            // point's t), so once the next voxel starts past `best` we're done.
            let t_next = t_max_x.min(t_max_y).min(t_max_z);
            if let Some((bt, _, _)) = best {
                // Strict: a tie exactly on the voxel boundary is still tested (deterministic tie-break).
                if bt < t_next {
                    break;
                }
            }
            if t_next > max_t {
                break;
            }
            guard += 1;
            if guard > max_cells {
                break;
            }
            if t_max_x <= t_max_y && t_max_x <= t_max_z {
                cx += sx;
                t_max_x += t_delta_x;
            } else if t_max_y <= t_max_z {
                cy += sy;
                t_max_y += t_delta_y;
            } else {
                cz += sz;
                t_max_z += t_delta_z;
            }
        }
        best
    }
}

fn orthonormalize(n: Vec3, u_hint: Vec3) -> (Vec3, Vec3) {
    // Rebuild an in-plane U/V basis orthogonal to n, keeping U as close to the hint as
    // possible (so the decal's authored orientation/UV direction is preserved).
    let mut u = u_hint - n * n.dot(u_hint);
    if u.length_squared() < 1e-6 {
        // Hint was parallel to n — pick any perpendicular.
        let alt = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        u = alt - n * n.dot(alt);
    }
    let u = u.normalize();
    let v = n.cross(u).normalize();
    (u, v)
}

/// Clip a convex polygon (CCW) against one half-space {p : dot(p,axis) <= offset}.
fn clip_halfspace(poly: &[Vec3], axis: Vec3, offset: f32) -> Vec<Vec3> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 2);
    let dist = |p: Vec3| axis.dot(p) - offset;
    for i in 0..poly.len() {
        let cur = poly[i];
        let prev = poly[(i + poly.len() - 1) % poly.len()];
        let dc = dist(cur);
        let dp = dist(prev);
        let cur_in = dc <= 0.0;
        let prev_in = dp <= 0.0;
        if cur_in != prev_in {
            // Crossing — add the intersection.
            let tt = dp / (dp - dc);
            out.push(prev + (cur - prev) * tt);
        }
        if cur_in {
            out.push(cur);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// ENGINE-RULE projection (RE of reach_tag_test.exe c_decal::create / build, see
// docs/hrek_re/21_re_decals.md).
//
// The engine builds a scenario decal's geometry at map load like this:
//   1. ONE collision ray: from (pos + n*0.01) along -n, 1.0 wu long. No hit -> no decal.
//      (Sapien stores `pos` 0.1 wu proud of the receiver, so the ray finds it 0.11 wu in.)
//   2. The hit surface is the seed of a FLOOD FILL across shared EDGES: a neighbour is added when
//      the shared edge crosses the decal's UV box and the neighbour passes the CULL test
//      (angle(-forward, surface normal) <= cull angle). Every accepted surface is emitted whole;
//      the pixel shader clips UVs outside [0,1] (HMS clips the polygon instead: same pixels).
//   3. CLAMP angle: a surface steeper than `clamp` gets the projector FOLDED about the shared edge
//      (the primary about the axis cross(N, forward) through the hit point) until the projection
//      angle equals `clamp` -- with clamp 0 the decal wraps around the corner like a sticker.
//   4. No depth clipping at all, no normal offset for projected geometry (a 0.001 wu offset only on
//      the planar-quad path); z-fighting is handled at draw time (VS depth bias ~ 1/d^2).
//   5. If the primary hit is INSTANCED geometry, a second ray (structure only, from hit + n*r/2,
//      length r) seeds a second flood on the structure surface under it.
// Excluded receivers: two-sided / invisible / breakable / invalid / conveyor collision surfaces.
// HMS's soup is RENDER geometry (clusters + instances), so `MESH_NO_DECAL` (transparent / glass
// meshes) stands in for the collision-flag exclusions.
// ---------------------------------------------------------------------------------------------

/// Owner-word flag: the triangle belongs to instanced geometry (not a structure cluster mesh).
pub const MESH_INSTANCE: u32 = 1 << 31;
/// Owner-word flag: the triangle never receives decals (transparent blend / glass).
pub const MESH_NO_DECAL: u32 = 1 << 30;

/// Engine ray: start offset ABOVE the authored position along the outward normal, and length.
const ENGINE_RAY_START: f32 = 0.01;
const ENGINE_RAY_LEN: f32 = 1.0;
/// Second (structure-under-instance) ray: from hit + n*r/2, length r; the engine's r is the decs
/// "runtime max radius" (0.8 .. 4 wu on the shipped tags), i.e. the structure must lie within r/2
/// below the instance surface. 1.2 covers the floor under any crate/plate a decal sits on.
const UNDER_INSTANCE_RAY: f32 = 1.2;
/// Safety cap on emitted triangles per decal (the engine caps a fragment at 128 collision
/// polygons / 1024 vertices; render triangles are much finer).
const MAX_TRIS_PER_DECAL: usize = 6000;

/// One projector frame: outward normal `n` (= -forward), in-plane `u`/`v`, centre.
#[derive(Clone, Copy)]
struct Frame { c: Vec3, n: Vec3, u: Vec3, v: Vec3 }

impl Frame {
    /// Rotate the whole frame rigidly about the line (pivot, axis) by `ang` radians (Rodrigues).
    fn rotated(&self, pivot: Vec3, axis: Vec3, ang: f32) -> Frame {
        let (s, co) = ang.sin_cos();
        let rot = |p: Vec3| p * co + axis.cross(p) * s + axis * (axis.dot(p) * (1.0 - co));
        Frame { c: pivot + rot(self.c - pivot), n: rot(self.n), u: rot(self.u), v: rot(self.v) }
    }
}

/// Edge adjacency over the whole soup, built ONCE per load for the decal phase and
/// dropped afterwards (12 B per triangle edge; ~50 MB transient on the largest maps). Per-edge
/// grid lookups were pathological on small dense maps whose rebalanced grid cell (16 wu) holds
/// most of the map (Prisoner: 12 s for 255 decals).
pub struct EdgeIndex {
    /// (edge hash, triangle) sorted by hash. Two 2 mm-welded vertices -> one hash.
    entries: Vec<(u64, u32)>,
}

impl EdgeIndex {
    pub fn build(soup: &TriSoup) -> Self {
        use rayon::prelude::*;
        let mut entries: Vec<(u64, u32)> = (0..soup.tris.len()).into_par_iter().flat_map_iter(|i| {
            let t = soup.tris[i];
            vec![(ehash(t[0], t[1]), i as u32), (ehash(t[1], t[2]), i as u32), (ehash(t[2], t[0]), i as u32)].into_iter()
        }).collect();
        entries.par_sort_unstable();
        Self { entries }
    }
    /// Triangles (other than `me`) across the welded DIRECTED edge a->b: a manifold neighbour
    /// carries the same edge REVERSED (b->a). A collision BSP edge has exactly one surface on
    /// each side, and that is what the engine flood crosses; in the render soup a same-direction
    /// twin is a back face welded to the same edge (the far side of a thin parapet / plate) and
    /// must not be entered -- folding onto it mirrored a floor decal up the OUTSIDE of a low
    /// wall it never reached.
    fn neighbours(&self, soup: &TriSoup, a: Vec3, b: Vec3, me: u32, out: &mut Vec<u32>) {
        out.clear();
        let h = ehash(a, b);
        let start = self.entries.partition_point(|e| e.0 < h);
        let (ka, kb) = (qkey(a), qkey(b));
        for &(eh, i) in &self.entries[start..] {
            if eh != h { break; }
            if i == me { continue; }
            let t = soup.tris[i as usize];
            // hash collision guard + orientation: the neighbour must hold b->a
            let reversed = (0..3).any(|k| qkey(t[k]) == kb && qkey(t[(k + 1) % 3]) == ka);
            if reversed && !out.contains(&i) { out.push(i); }
        }
    }
}

#[inline]
fn ehash(a: Vec3, b: Vec3) -> u64 {
    let k = ekey(a, b);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in k {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Result of the engine-rule projection.
pub struct Projection {
    /// Conformed triangles (world positions, decal UVs in 0..1 before any sprite sub-rect).
    pub tris: Vec<([Vec3; 3], [[f32; 2]; 3])>,
    /// The primary ray found a receiver (a decal with `hit == false` is not drawn by the engine).
    pub hit: bool,
    /// Number of soup triangles accepted (before clipping).
    pub surfaces: usize,
    /// Number of surfaces that needed a clamp-angle fold.
    pub folds: usize,
}

/// Fold the projector frame about the line (pivot, axis) so the angle between -forward (= n) and
/// the receiver normal `tn` becomes exactly `cos_clamp` (as close as the geometry allows). Mirrors
/// sub_1404BFDA0's clamp branch: the frame is rotated rigidly, so points on the fold line (the
/// shared edge) keep their UVs and the decal continues around the corner without a seam.
fn fold_frame(f: &Frame, pivot: Vec3, axis: Vec3, tn: Vec3, cos_clamp: f32) -> Frame {
    let axis = axis.normalize_or_zero();
    if axis == Vec3::ZERO { return *f; }
    // Components of n perpendicular to the axis: rotation only moves this part.
    let n_par = axis * axis.dot(f.n);
    let n_perp = f.n - n_par;
    let t_perp = tn - axis * axis.dot(tn); // receiver normal in the rotation plane (edge lies in it)
    let (lp, lt) = (n_perp.length(), t_perp.length());
    if lp < 1e-5 || lt < 1e-5 { return *f; }
    let np = n_perp / lp;
    let tp = t_perp / lt;
    // Current angle between np and tp in the rotation plane (signed about `axis`).
    let cur = np.cross(tp).dot(axis).atan2(np.dot(tp)); // rotating n by `cur` aligns np with tp
    // Target: dot(n', tn) = cos_clamp  ->  lp * cos(phi) + n_par.tn = cos_clamp
    let want = ((cos_clamp - n_par.dot(tn)) / lp).clamp(-1.0, 1.0).acos();
    // Keep n' on the same side as it is now (rotate the shortest way toward tn).
    let target = if cur >= 0.0 { want } else { -want };
    let delta = cur - target;
    if delta.abs() < 1e-4 { return *f; }
    f.rotated(pivot, axis, delta)
}

/// Clip triangle `t` to the in-plane box of `f` (±hx along u, ±hy along v). No depth planes.
fn clip_to_frame(t: &[Vec3; 3], f: &Frame, hx: f32, hy: f32) -> Vec<Vec3> {
    let cu = f.u.dot(f.c);
    let cv = f.v.dot(f.c);
    let mut poly = vec![t[0], t[1], t[2]];
    poly = clip_halfspace(&poly, f.u, cu + hx);
    if poly.len() < 3 { return Vec::new(); }
    poly = clip_halfspace(&poly, -f.u, -(cu - hx));
    if poly.len() < 3 { return Vec::new(); }
    poly = clip_halfspace(&poly, f.v, cv + hy);
    if poly.len() < 3 { return Vec::new(); }
    poly = clip_halfspace(&poly, -f.v, -(cv - hy));
    if poly.len() < 3 { return Vec::new(); }
    poly
}

/// Fine uniform grid over a decal's AABB for the occlusion pass. Testing every emitted piece
/// (x4 samples x up to 85 subdivisions) against ONE flat candidate list gathered from the soup's
/// coarse (4..64 wu) cells over the whole decal's AABB is ~10^12 triangle tests for a decal folded
/// around dense canyon rock (Battle Canyon: 17k pieces, 217k candidates) — a "stuck at 99%" load.
/// With the same candidate set on this grid each 10 cm probe segment tests only the handful of
/// triangles in the cells it crosses; results are identical (every candidate that can intersect
/// a probe is in every cell the probe touches).
struct OcclGrid {
    mn: Vec3,
    inv: f32,
    dims: [i32; 3],
    cells: Vec<Vec<u32>>,
    /// Triangles spanning more than `BIG_CELLS` cells (a backdrop / long wall through the box):
    /// tested by every probe instead of being replicated into thousands of cells.
    big: Vec<u32>,
}

impl OcclGrid {
    const CELL: f32 = 0.5;
    const MAX_DIM: i32 = 48;
    const BIG_CELLS: i64 = 512;

    fn build(soup: &TriSoup, cands: &[u32], mn: Vec3, mx: Vec3) -> Self {
        let ext = (mx - mn).max(Vec3::splat(1e-3));
        // Coarsen the cell until the box fits in MAX_DIM^3 cells (bounded memory for a wrapped decal).
        let mut cell = Self::CELL;
        while (ext.max_element() / cell).ceil() as i32 > Self::MAX_DIM { cell *= 2.0; }
        let inv = 1.0 / cell;
        let dims = [
            ((ext.x * inv).ceil() as i32).max(1),
            ((ext.y * inv).ceil() as i32).max(1),
            ((ext.z * inv).ceil() as i32).max(1),
        ];
        let mut g = OcclGrid { mn, inv, dims, cells: vec![Vec::new(); (dims[0] * dims[1] * dims[2]) as usize], big: Vec::new() };
        for &i in cands {
            let t = soup.tris[i as usize];
            let (a, b) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
            let (k0, k1) = (g.key(a), g.key(b));
            let span = (k1[0] - k0[0] + 1) as i64 * (k1[1] - k0[1] + 1) as i64 * (k1[2] - k0[2] + 1) as i64;
            if span > Self::BIG_CELLS { g.big.push(i); continue; }
            for x in k0[0]..=k1[0] { for y in k0[1]..=k1[1] { for z in k0[2]..=k1[2] {
                let k = g.idx(x, y, z);
                g.cells[k].push(i);
            } } }
        }
        g
    }

    #[inline]
    fn key(&self, p: Vec3) -> [i32; 3] {
        let q = (p - self.mn) * self.inv;
        [
            (q.x.floor() as i32).clamp(0, self.dims[0] - 1),
            (q.y.floor() as i32).clamp(0, self.dims[1] - 1),
            (q.z.floor() as i32).clamp(0, self.dims[2] - 1),
        ]
    }

    #[inline]
    fn idx(&self, x: i32, y: i32, z: i32) -> usize { ((x * self.dims[1] + y) * self.dims[2] + z) as usize }

    /// Every candidate whose cell-AABB overlaps the segment box `[a, b]`, plus the big ones. A
    /// triangle in several of the (at most 8) touched cells is visited more than once; the tests
    /// are idempotent (`hit` breaks, `lift` is a max) so that only costs time, never correctness.
    fn probe(&self, a: Vec3, b: Vec3, out: &mut Vec<u32>) {
        out.clear();
        let (k0, k1) = (self.key(a.min(b)), self.key(a.max(b)));
        for x in k0[0]..=k1[0] { for y in k0[1]..=k1[1] { for z in k0[2]..=k1[2] {
            out.extend_from_slice(&self.cells[self.idx(x, y, z)]);
        } } }
        out.extend_from_slice(&self.big);
    }
}

/// How many of 4 samples (centroid + vertex-centroid midpoints) of the triangle sit
/// under another OPAQUE surface within 10 cm along the triangle normal. Transparent/glass meshes
/// (`MESH_NO_DECAL`) and the triangle's own surface do not count; only surfaces facing the same
/// way (an overlay's top) do.
/// Also returns the largest NEAR-coplanar twin offset (0 .. 1.5 mm) seen: the decal is then lifted
/// onto that twin (see `emit_unoccluded`) so it sits on the topmost of the coincident surfaces.
fn occluded_samples(soup: &TriSoup, grid: &OcclGrid, t: &[Vec3; 3], tn: Vec3, me: u32) -> (u32, f32) {
    const REACH: f32 = 0.10; // must exceed the draw-time nudge (4 mm + 6e-6 d^2: 6.4 cm at 100 wu)
    const TWIN: f32 = 0.0015;
    let c = (t[0] + t[1] + t[2]) / 3.0;
    let samples = [c, (t[0] + c) * 0.5, (t[1] + c) * 0.5, (t[2] + c) * 0.5];
    let mut hits = 0;
    let mut lift = 0.0f32;
    let mut cands: Vec<u32> = Vec::new();
    for p in samples {
        let o = p - tn * 0.0005;
        // Only the triangles around the probe segment [o, o + tn*REACH].
        grid.probe(o, o + tn * REACH, &mut cands);
        let mut hit = false;
        for &i in &cands {
            if i == me { continue; }
            let [a, b, cc] = soup.tris[i as usize];
            let e1 = b - a;
            let e2 = cc - a;
            if e1.cross(e2).dot(tn) <= 0.0 { continue; } // only overlays facing the same way
            let pv = tn.cross(e2);
            let det = e1.dot(pv);
            if det.abs() < 1e-9 { continue; }
            let inv = 1.0 / det;
            let tv = o - a;
            let u = tv.dot(pv) * inv;
            if !(0.0..=1.0).contains(&u) { continue; }
            let qv = tv.cross(e1);
            let v = tn.dot(qv) * inv;
            if v < 0.0 || u + v > 1.0 { continue; }
            let tt = e2.dot(qv) * inv;
            let dz = tt - 0.0005; // height above the sample point
            if dz > TWIN && dz < REACH { hit = true; break; }
            if dz > 0.0 && dz <= TWIN { lift = lift.max(dz); }
        }
        if hit { hits += 1; }
    }
    (hits, lift)
}

/// Emit `t` unless it is covered by an overlay; a partially covered triangle is split
/// in four (midpoints, UVs interpolated) up to 3 levels so only the covered part is dropped.
fn emit_unoccluded(soup: &TriSoup, cands: &OcclGrid, t: [Vec3; 3], uv: [[f32; 2]; 3], tn: Vec3, me: u32, depth: u32, out: &mut Vec<([Vec3; 3], [[f32; 2]; 3])>) {
    let (hits, lift) = occluded_samples(soup, cands, &t, tn, me);
    // A NEAR-coplanar twin (< 1.5 mm above: a material/lightmap split of the same wall,
    // structure vs instance copy) is not an overlay -- the decal is lifted onto it so the
    // camera-ward nudge clears BOTH copies instead of z-fighting the upper one (Powerhouse
    // p_macro_fill_yellow under the emblem: twin 0.7 mm above).
    let lifted = |t: [Vec3; 3]| if lift > 0.0 { [t[0] + tn * lift, t[1] + tn * lift, t[2] + tn * lift] } else { t };
    if hits == 0 { out.push((lifted(t), uv)); return; }
    if hits == 4 { return; }
    let long2 = (t[1] - t[0]).length_squared().max((t[2] - t[1]).length_squared()).max((t[0] - t[2]).length_squared());
    if depth >= 3 || long2 < 0.01 * 0.01 {
        if hits < 2 { out.push((lifted(t), uv)); }
        return;
    }
    let m = |a: usize, b: usize| ((t[a] + t[b]) * 0.5, [(uv[a][0] + uv[b][0]) * 0.5, (uv[a][1] + uv[b][1]) * 0.5]);
    let (m01, u01) = m(0, 1);
    let (m12, u12) = m(1, 2);
    let (m20, u20) = m(2, 0);
    emit_unoccluded(soup, cands, [t[0], m01, m20], [uv[0], u01, u20], tn, me, depth + 1, out);
    emit_unoccluded(soup, cands, [m01, t[1], m12], [u01, uv[1], u12], tn, me, depth + 1, out);
    emit_unoccluded(soup, cands, [m20, m12, t[2]], [u20, u12, uv[2]], tn, me, depth + 1, out);
    emit_unoccluded(soup, cands, [m01, m12, m20], [u01, u12, u20], tn, me, depth + 1, out);
}

#[inline]
fn qkey(p: Vec3) -> [i32; 3] {
    // 2 mm grid: cluster/instance seams are welded by position (BSP vertices are dequantized
    // per mesh, so seam vertices agree to ~1e-4 wu).
    [(p.x * 500.0).round() as i32, (p.y * 500.0).round() as i32, (p.z * 500.0).round() as i32]
}

#[inline]
fn ekey(a: Vec3, b: Vec3) -> [i32; 6] {
    let (ka, kb) = (qkey(a), qkey(b));
    if ka <= kb { [ka[0], ka[1], ka[2], kb[0], kb[1], kb[2]] } else { [kb[0], kb[1], kb[2], ka[0], ka[1], ka[2]] }
}

/// Project one scenario decal with the engine rule (see the module note above).
///
/// `pos` = authored position, `facing` = outward receiver normal (+col2 of the authored quat),
/// `u_hint` = authored in-plane U axis, `hx`/`hy` = half extents (already scaled), `cull_deg` /
/// `clamp_deg` from the decal definition (degrees; NaN/0 -> engine defaults: cull 90, clamp 0).
pub fn project_engine(
    soup: &TriSoup,
    edges: &EdgeIndex,
    pos: Vec3,
    facing: Vec3,
    u_hint: Vec3,
    hx: f32,
    hy: f32,
    cull_deg: f32,
    clamp_deg: f32,
) -> Projection {
    project_engine_traced(soup, edges, pos, facing, u_hint, hx, hy, cull_deg, clamp_deg, false)
}

/// `project_engine` with an optional stderr trace of every flood step (print-only diag).
#[allow(clippy::too_many_arguments)]
pub fn project_engine_traced(
    soup: &TriSoup,
    edges: &EdgeIndex,
    pos: Vec3,
    facing: Vec3,
    u_hint: Vec3,
    hx: f32,
    hy: f32,
    cull_deg: f32,
    clamp_deg: f32,
    trace: bool,
) -> Projection {
    let mut out = Projection { tris: Vec::new(), hit: false, surfaces: 0, folds: 0 };
    let n = facing.normalize_or_zero();
    if n == Vec3::ZERO || hx <= 0.0 || hy <= 0.0 || soup.is_empty() {
        return out;
    }
    let (u, v) = orthonormalize(n, u_hint);
    let cull_deg = if cull_deg.is_finite() && cull_deg > 0.0 { cull_deg.min(180.0) } else { 90.0 };
    let clamp_deg = if clamp_deg.is_finite() && clamp_deg > 0.0 { clamp_deg.min(180.0) } else { 0.0 };
    let cos_cull = cull_deg.to_radians().cos();
    let cos_clamp = clamp_deg.to_radians().cos();

    let receiver = |i: u32| soup.tri_owner(i) & MESH_NO_DECAL == 0;
    let is_instance = |i: u32| { let o = soup.tri_owner(i); o != u32::MAX && o & MESH_INSTANCE != 0 };
    // Collision rays are one-sided: a surface facing away from the ray is not a hit (a sign quad
    // instance sitting exactly on the decal's spot, seen from behind, must not eat the ray).
    let front = |i: u32| { let t = soup.tris[i as usize]; (t[1] - t[0]).cross(t[2] - t[0]).dot(n) > 0.0 };

    // 1. Primary engine ray. A transparent/glass mesh (`MESH_NO_DECAL`) is skipped -- unless it is
    // the ONLY thing within the ray: the engine tests collision, not render blend, and an ice
    // sheet / frosted pane instance does carry collision, so the decal the artist put on it must
    // not vanish. In that case the flood is confined to that mesh.
    let origin = pos + n * ENGINE_RAY_START;
    let mut only_owner: Option<u32> = None;
    let (t0, tri0) = match soup.raycast_hit_if(origin, -n, ENGINE_RAY_LEN, |i| receiver(i) && front(i)) {
        Some((t, _, i)) => (t, i),
        None => match soup.raycast_hit_if(origin, -n, ENGINE_RAY_LEN, front) {
            Some((t, _, i)) => { only_owner = Some(soup.tri_owner(i)); (t, i) }
            None => {
                if trace { eprintln!("DECALTRACE no receiver within {ENGINE_RAY_LEN} wu of {pos:?} along {:?}", -n); }
                return out;
            }
        },
    };
    let receiver = |i: u32| match only_owner { Some(o) => soup.tri_owner(i) == o, None => receiver(i) };
    let hit0 = origin - n * t0;
    out.hit = true;
    if trace { eprintln!("DECALTRACE hit tri {tri0} at {hit0:?} (t={t0:.3}) owner={:#x} u={u:?} v={v:?} hx={hx} hy={hy} cull={cull_deg} clamp={clamp_deg}", soup.tri_owner(tri0)); }
    // The engine re-centres the decal system on the HIT point (decal_manager: position = collision
    // point) -- the frame's centre only matters for folds, which rotate about edge lines.
    let frame0 = Frame { c: hit0, n, u, v };

    // Seeds: (triangle, hit point). 5. structure under an instance hit.
    let mut seeds: Vec<(u32, Vec3)> = vec![(tri0, hit0)];
    if is_instance(tri0) {
        let o2 = hit0 + n * (UNDER_INSTANCE_RAY * 0.5);
        if let Some((t2, _, tri2)) = soup.raycast_hit_if(o2, -n, UNDER_INSTANCE_RAY, |i| receiver(i) && !is_instance(i) && front(i)) {
            let p2 = o2 - n * t2;
            if (p2 - hit0).length() <= UNDER_INSTANCE_RAY && tri2 != tri0 {
                seeds.push((tri2, p2));
            }
        }
    }

    // Emitted pieces before the overlay-occlusion pass (with their source soup triangle).
    let mut raw: Vec<([Vec3; 3], [[f32; 2]; 3], u32)> = Vec::new();
    let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // The render soup can carry the SAME surface twice (instance + structure, or a
    // lightmap-split twin); collision has it once. Skip a triangle whose welded vertex set was
    // already emitted, so a decal never paints the same polygon twice.
    let mut emitted: std::collections::HashSet<[[i32; 3]; 3]> = std::collections::HashSet::new();
    let mut nb: Vec<u32> = Vec::new();
    // Queue entries: (tri, parent frame, fold pivot, fold axis (zero = primary rule)).
    let mut queue: std::collections::VecDeque<(u32, Frame, Vec3, Vec3)> = std::collections::VecDeque::new();
    for &(tri, hp) in &seeds {
        queue.push_back((tri, frame0, hp, Vec3::ZERO));
    }
    let uv_of = |f: &Frame, p: Vec3| -> [f32; 2] {
        let lu = (f.u.dot(p - f.c)) / hx * 0.5 + 0.5;
        let lv = 1.0 - ((f.v.dot(p - f.c)) / hy * 0.5 + 0.5);
        [lu, lv]
    };
    while let Some((tri, pf, pivot, axis)) = queue.pop_front() {
        if !visited.insert(tri) { continue; }
        // The cap counts the flood's pieces (`raw`), not `out.tris` (empty until after the
        // flood, since the occlusion pass fills it) -- else no decal is ever capped.
        if raw.len() >= MAX_TRIS_PER_DECAL { break; }
        let t = soup.tris[tri as usize];
        let tn = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        if tn == Vec3::ZERO { continue; }
        {
            let mut k = [qkey(t[0]), qkey(t[1]), qkey(t[2])];
            k.sort();
            if !emitted.insert(k) { continue; }
        }
        // CULL: angle between the projection direction and the receiver normal.
        let cosang = tn.dot(pf.n);
        if trace { eprintln!("DECALTRACE pop tri {tri} tn={tn:?} cosang={cosang:.3} v={t:?}"); }
        if cosang < cos_cull { if trace { eprintln!("DECALTRACE   culled"); } continue; }
        // CLAMP: fold the frame about the shared edge (or, for a seed, about cross(tn, forward)
        // through the hit point) until the projection angle equals the clamp angle.
        let mut f = pf;
        if cosang < cos_clamp - 1e-4 {
            let ax = if axis == Vec3::ZERO {
                let a = tn.cross(-pf.n);
                if a.length_squared() > 1e-8 { a } else { pf.u }
            } else { axis };
            f = fold_frame(&pf, pivot, ax, tn, cos_clamp);
            out.folds += 1;
        } else if cosang.abs() < 0.02 {
            // Exactly edge-on and inside the clamp: the projection would smear along the surface
            // without bound (UV constant along it). The engine only reaches this with clamp >= 89
            // deg; guard by folding to 89 deg.
            let ax = if axis == Vec3::ZERO { pf.u } else { axis };
            f = fold_frame(&pf, pivot, ax, tn, 89f32.to_radians().cos());
        }
        // Box test / clip in the (possibly folded) frame.
        let poly = clip_to_frame(&t, &f, hx, hy);
        if poly.len() < 3 { if trace { eprintln!("DECALTRACE   outside box"); } continue; }
        out.surfaces += 1;
        if trace { eprintln!("DECALTRACE   accepted: {} verts, frame c={:?} n={:?}", poly.len(), f.c, f.n); }
        for k in 1..poly.len() - 1 {
            let tri3 = [poly[0], poly[k], poly[k + 1]];
            // A piece of the receiver that lies under ANOTHER opaque surface within 10 cm
            // (an instance slab / plate / lightmap-split twin a few mm above the floor the flood
            // painted) is invisible in the engine (its depth bias is sub-mm). HMS's camera-ward
            // nudge grows with d^2 (depth precision), so such a piece pokes through the overlay at
            // 20-40 wu and flickers with camera motion. Drop it at build time instead (partially
            // covered triangles are subdivided so only the covered part goes).
            raw.push((tri3, [uv_of(&f, poly[0]), uv_of(&f, poly[k]), uv_of(&f, poly[k + 1])], tri));
        }
        // Expand across the three edges; the neighbour inherits THIS surface's frame and folds
        // about the shared edge if it needs to.
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            edges.neighbours(soup, a, b, tri, &mut nb);
            if trace { eprintln!("DECALTRACE   edge {e} {a:?}-{b:?} -> neighbours {nb:?}"); }
            for &j in &nb {
                if receiver(j) && !visited.contains(&j) {
                    queue.push_back((j, f, a, b - a));
                }
            }
        }
    }
    // Overlay occlusion pass against a LOCAL candidate list (the soup grid can be as
    // coarse as 16 wu on small maps, so per-ray grid marches cost seconds per map).
    if !raw.is_empty() {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for (t, _, _) in &raw { for q in t { mn = mn.min(*q); mx = mx.max(*q); } }
        let (mn, mx) = (mn - Vec3::splat(0.11), mx + Vec3::splat(0.11));
        let (k0, k1) = (soup.key(mn), soup.key(mx));
        let mut cands: Vec<u32> = Vec::new();
        let mut consider = |i: u32| {
            if soup.tri_owner(i) & MESH_NO_DECAL != 0 { return; }
            let t = soup.tris[i as usize];
            let (a, b) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
            if a.x <= mx.x && b.x >= mn.x && a.y <= mx.y && b.y >= mn.y && a.z <= mx.z && b.z >= mn.z { cands.push(i); }
        };
        for x in k0.0..=k1.0 { for y in k0.1..=k1.1 { for z in k0.2..=k1.2 {
            for &i in soup.cell_refs((x, y, z)) { consider(i); }
        } } }
        for &i in &soup.big { consider(i); }
        cands.sort_unstable();
        cands.dedup();
        let grid = OcclGrid::build(soup, &cands, mn, mx);
        if trace { eprintln!("DECALTRACE occl pass: {} pieces, {} candidates on a {:?} grid ({} big)", raw.len(), cands.len(), grid.dims, grid.big.len()); }
        for (t, uv, src) in raw {
            let tn = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            if trace {
                let (h, l) = occluded_samples(soup, &grid, &t, tn, src);
                eprintln!("DECALTRACE occl piece src={src} c={:?} tn={tn:?} hits={h} lift={l:.4}", (t[0] + t[1] + t[2]) / 3.0);
            }
            emit_unoccluded(soup, &grid, t, uv, tn, src, 0, &mut out.tris);
        }
    }
    out
}

#[cfg(test)]
mod engine_rule_tests {
    use super::*;

    fn quad(soup: &mut TriSoup, z: f32, x0: f32, x1: f32, y0: f32, y1: f32, owner: u32) {
        // CCW seen from +Z (outward normal +Z)
        let (a, b, c, d) = (Vec3::new(x0, y0, z), Vec3::new(x1, y0, z), Vec3::new(x1, y1, z), Vec3::new(x0, y1, z));
        soup.add_tri_owned(a, b, c, (0, 0, 0), owner);
        soup.add_tri_owned(a, c, d, (0, 0, 0), owner);
    }

    #[test]
    fn engine_ray_ignores_surfaces_above_the_authored_pose() {
        // Floor at z=0 and another floor 1.2 wu above it; the decal is authored 0.1 above the
        // lower floor (Sapien convention). The old reach-above raycast landed on the UPPER floor.
        let mut soup = TriSoup::new(4.0);
        quad(&mut soup, 0.0, -5.0, 5.0, -5.0, 5.0, 0);
        quad(&mut soup, 1.2, -5.0, 5.0, -5.0, 5.0, 1);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, 0.0, 0.1), Vec3::Z, Vec3::X, 0.5, 0.5, 90.0, 0.0);
        assert!(p.hit);
        assert!(!p.tris.is_empty());
        for (ps, _) in &p.tris { for q in ps { assert!((q.z - 0.0).abs() < 1e-5, "geometry must lie ON the lower floor, got z={}", q.z); } }
        // exactly the 1x1 box: 4 corners spanned
        let (mut mn, mut mx) = (Vec3::splat(9.0), Vec3::splat(-9.0));
        for (ps, _) in &p.tris { for q in ps { mn = mn.min(*q); mx = mx.max(*q); } }
        assert!((mn.x + 0.5).abs() < 1e-4 && (mx.x - 0.5).abs() < 1e-4 && (mn.y + 0.5).abs() < 1e-4 && (mx.y - 0.5).abs() < 1e-4);
    }

    #[test]
    fn no_receiver_within_one_wu_means_no_decal() {
        let mut soup = TriSoup::new(4.0);
        quad(&mut soup, -1.5, -5.0, 5.0, -5.0, 5.0, 0);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, 0.0, 0.1), Vec3::Z, Vec3::X, 0.5, 0.5, 90.0, 0.0);
        assert!(!p.hit && p.tris.is_empty());
    }

    #[test]
    fn stacked_plate_is_not_reached_unless_edge_connected() {
        // A floor and a thin plate (separate, unwelded instance mesh) resting 0.05 above it under
        // the decal's box: the flood from the floor must NOT paint the plate (z-fight source).
        let mut soup = TriSoup::new(4.0);
        quad(&mut soup, 0.0, -5.0, 5.0, -5.0, 5.0, 0);
        quad(&mut soup, 0.05, 0.2, 0.4, 0.2, 0.4, 1 | MESH_INSTANCE);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, 0.0, 0.1), Vec3::Z, Vec3::X, 1.0, 1.0, 90.0, 0.0);
        assert!(p.hit);
        assert!(p.tris.iter().all(|(ps, _)| ps.iter().all(|q| q.z.abs() < 1e-5)));
    }

    #[test]
    fn floor_under_a_slab_is_not_painted() {
        // A slab 8 mm above the floor covers half of the decal's box: the floor piece under it is
        // invisible in the engine and must not be emitted (it would poke through at distance).
        let mut soup = TriSoup::new(4.0);
        quad(&mut soup, 0.0, -5.0, 5.0, -5.0, 5.0, 0);
        quad(&mut soup, 0.008, 0.0, 3.0, -3.0, 3.0, 1 | MESH_INSTANCE);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, 0.0, 0.1), Vec3::Z, Vec3::X, 1.0, 1.0, 90.0, 0.0);
        assert!(p.hit);
        // the slab top itself receives (it is a front face hit by the flood? no: not edge-connected;
        // the primary ray hits the SLAB at x>0? pos.x = 0 is on the slab edge -> either way the floor
        // piece under the slab (x > 0.05, z = 0) must be absent)
        let mut covered_area = 0.0f32; let mut open_area = 0.0f32;
        for (ps, _) in &p.tris {
            if ps.iter().any(|q| q.z.abs() > 1e-5) { continue; }
            let c = (ps[0] + ps[1] + ps[2]) / 3.0;
            let a = (ps[1] - ps[0]).cross(ps[2] - ps[0]).length() * 0.5;
            if c.x > 0.0 { covered_area += a } else { open_area += a }
        }
        assert!(covered_area < 0.15, "floor under the slab still painted: {covered_area} wu^2 (open {open_area})");
        assert!(open_area > 1.7, "the uncovered floor half must still be painted: {open_area} wu^2");
    }

    #[test]
    fn instance_hit_also_seeds_the_structure_below() {
        // Decal authored on a crate top (instance) 0.5 above the structure floor: engine rule 5
        // adds the floor under it (occluded by the crate where the crate is, visible around it).
        let mut soup = TriSoup::new(4.0);
        quad(&mut soup, 0.0, -5.0, 5.0, -5.0, 5.0, 0);
        quad(&mut soup, 0.5, -0.3, 0.3, -0.3, 0.3, 1 | MESH_INSTANCE);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, 0.0, 0.6), Vec3::Z, Vec3::X, 1.0, 1.0, 90.0, 0.0);
        assert!(p.hit);
        let on_top = p.tris.iter().filter(|(ps, _)| ps.iter().all(|q| (q.z - 0.5).abs() < 1e-5)).count();
        let on_floor = p.tris.iter().filter(|(ps, _)| ps.iter().all(|q| q.z.abs() < 1e-5)).count();
        assert!(on_top > 0 && on_floor > 0, "top={on_top} floor={on_floor}");
    }

    #[test]
    fn clamp_zero_folds_the_decal_up_a_connected_wall() {
        // Floor y in [-5, 0] meeting a wall at y=0 rising along +z (welded edge). A decal centred
        // 0.3 before the wall with half-size 0.5 must continue 0.2 up the wall (sticker wrap),
        // with the wall UVs continuing seamlessly from the edge.
        let mut soup = TriSoup::new(4.0);
        // floor (normal +Z)
        let (a, b, c, d) = (Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, -5.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 0.0));
        soup.add_tri_owned(a, b, c, (0, 0, 0), 0);
        soup.add_tri_owned(a, c, d, (0, 0, 0), 0);
        // wall (normal -Y, facing the decal): vertices share the floor's y=0 edge exactly
        let (w0, w1, w2, w3) = (Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 3.0), Vec3::new(-5.0, 0.0, 3.0));
        soup.add_tri_owned(w0, w1, w2, (0, 0, 0), 0); // wound so (b-a)x(c-a) = -Y
        soup.add_tri_owned(w0, w2, w3, (0, 0, 0), 0);
        soup.shrink();
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, -0.3, 0.1), Vec3::Z, Vec3::X, 0.5, 0.5, 95.0, 0.0);
        assert!(p.hit);
        assert!(p.folds >= 1, "the wall must be folded");
        let mut zmax = 0.0f32;
        for (ps, uvs) in &p.tris {
            for (q, uv) in ps.iter().zip(uvs) {
                zmax = zmax.max(q.z);
                if q.y.abs() < 1e-4 && q.z > 1e-4 {
                    // on the wall: UV.v continues past the edge value (edge at v = 1-0.8 = 0.2)
                    assert!(uv[1] <= 0.2 + 1e-3 && uv[1] >= -1e-3, "wall uv {:?}", uv);
                }
            }
        }
        assert!((zmax - 0.2).abs() < 1e-3, "wrap height {zmax} (expected 0.2)");
        // no geometry beyond the wrap height, none on the far side
        assert!(p.tris.iter().all(|(ps, _)| ps.iter().all(|q| q.y <= 1e-4)));
    }

    #[test]
    fn back_face_welded_to_the_floor_edge_is_not_entered() {
        // Countdown decal 42: the floor's edge is welded to the OUTER face of a low wall (normal
        // pointing away from the floor, same edge direction as the floor). The engine's collision
        // flood only ever crosses to the surface on the other side of the edge (the inner face),
        // which the wrap never reaches; a same-direction twin must be ignored.
        let mut soup = TriSoup::new(4.0);
        let (a, b, c, d) = (Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, -5.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 0.0));
        soup.add_tri_owned(a, b, c, (0, 0, 0), 0);
        soup.add_tri_owned(a, c, d, (0, 0, 0), 0); // floor edge c->d runs -X
        // outer face (normal +Y, away from the floor) welded to the same edge with the SAME direction
        let (w0, w1, w2, w3) = (Vec3::new(5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 3.0), Vec3::new(5.0, 0.0, 3.0));
        // winding (w0,w1,w2): (w1-w0)x(w2-w0) = (-10,0,0)x(-10,0,3) = (0*3-0*0, 0*-10-(-10)*3, 0) = (0,30,0) = +Y
        soup.add_tri_owned(w0, w1, w2, (0, 0, 0), 0);
        soup.add_tri_owned(w0, w2, w3, (0, 0, 0), 0);
        soup.shrink();
        // decal 0.89 before the edge with half 0.64 (does not reach the edge), cull 95, clamp 0
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, -0.89, 0.1), Vec3::Z, Vec3::X, 0.64, 0.64, 95.0, 0.0);
        assert!(p.hit && !p.tris.is_empty());
        assert!(p.tris.iter().all(|(ps, _)| ps.iter().all(|q| q.z.abs() < 1e-5)), "nothing may land on the wall");
    }

    #[test]
    fn cull_angle_rejects_a_wall() {
        let mut soup = TriSoup::new(4.0);
        let (a, b, c, d) = (Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, -5.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 0.0));
        soup.add_tri_owned(a, b, c, (0, 0, 0), 0);
        soup.add_tri_owned(a, c, d, (0, 0, 0), 0);
        let (w0, w1, w2, w3) = (Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 3.0), Vec3::new(-5.0, 0.0, 3.0));
        soup.add_tri_owned(w0, w1, w2, (0, 0, 0), 0);
        soup.add_tri_owned(w0, w2, w3, (0, 0, 0), 0);
        soup.shrink();
        // dirt piles: cull 75 / clamp 75 -> the 90-degree wall is culled
        let p = project_engine(&soup, &EdgeIndex::build(&soup), Vec3::new(0.0, -0.3, 0.1), Vec3::Z, Vec3::X, 0.5, 0.5, 75.0, 75.0);
        assert!(p.hit && !p.tris.is_empty());
        assert!(p.tris.iter().all(|(ps, _)| ps.iter().all(|q| q.z.abs() < 1e-5)));
    }

    /// Dense floor: `n` x `n` cells of two triangles each over [-h, h]^2 at height z.
    fn dense_floor(soup: &mut TriSoup, z: f32, h: f32, n: usize, owner: u32) {
        let step = 2.0 * h / n as f32;
        for iy in 0..n { for ix in 0..n {
            let (x0, y0) = (-h + ix as f32 * step, -h + iy as f32 * step);
            quad(soup, z, x0, x0 + step, y0, y0 + step, owner);
        } }
    }

    #[test]
    fn canyon_stall_occlusion_grid_matches_brute_force() {
        // The per-probe local grid must return exactly what a scan of every candidate
        // returned. Dense floor + overlays of every size class: a tiny plate (one grid cell), a
        // medium slab (a few cells), a room-spanning sheet, a tilted triangle spanning the box in
        // all three axes (the grid's `big` list), and
        // near-coplanar twins (the lift path).
        let mut soup = TriSoup::new(4.0);
        dense_floor(&mut soup, 0.0, 3.0, 24, 0);
        quad(&mut soup, 0.02, 0.1, 0.3, 0.1, 0.3, 1 | MESH_INSTANCE); // tiny plate
        quad(&mut soup, 0.05, -1.5, 0.0, -1.5, 1.5, 2 | MESH_INSTANCE); // medium slab
        quad(&mut soup, 0.3, -40.0, 40.0, -40.0, 40.0, 3 | MESH_INSTANCE); // room-spanning sheet (many cells, out of reach)
        // a tilted room-spanning triangle: spans the box in all three axes -> the grid's big list
        soup.add_tri_owned(Vec3::new(-40.0, -40.0, 0.06), Vec3::new(40.0, -40.0, 0.06), Vec3::new(0.0, 40.0, 3.5), (0, 0, 0), 6 | MESH_INSTANCE);
        quad(&mut soup, 0.0007, 0.5, 2.0, -2.0, -0.5, 4); // near-coplanar twin -> lift
        quad(&mut soup, 0.5, -3.0, 3.0, -3.0, 3.0, 5 | MESH_NO_DECAL); // glass: never counts
        soup.shrink();
        let all: Vec<u32> = (0..soup.len() as u32).collect();
        let (mn, mx) = (Vec3::new(-3.2, -3.2, -0.2), Vec3::new(3.2, 3.2, 3.0));
        let grid = OcclGrid::build(&soup, &all, mn, mx);
        assert!(grid.dims.iter().all(|&d| d > 1), "the box must be split into cells: {:?}", grid.dims);
        assert!(!grid.big.is_empty(), "the tilted room-spanning triangle must land in the big list");
        // Brute force = the same code on a ONE-cell grid (every candidate tested by every probe).
        let brute = OcclGrid::build(&soup, &all, mn, mn + Vec3::splat(1e-4));
        assert_eq!(brute.dims, [1, 1, 1]);
        // Probe every floor triangle plus its 4 quarter pieces (what emit_unoccluded subdivides into).
        let mut n_hit = 0; let mut n_lift = 0; let mut n = 0;
        for i in 0..soup.len() {
            if soup.tri_owner(i as u32) != 0 { continue; }
            let t = soup.tris[i];
            let tn = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
            let c = (t[0] + t[1] + t[2]) / 3.0;
            let m = |a: Vec3, b: Vec3| (a + b) * 0.5;
            let pieces = [t, [t[0], m(t[0], t[1]), m(t[2], t[0])], [m(t[0], t[1]), t[1], m(t[1], t[2])], [m(t[2], t[0]), m(t[1], t[2]), t[2]], [c, m(t[0], t[1]), m(t[1], t[2])]];
            for pc in pieces {
                let a = occluded_samples(&soup, &grid, &pc, tn, i as u32);
                let b = occluded_samples(&soup, &brute, &pc, tn, i as u32);
                assert_eq!(a, b, "tri {i} piece {pc:?}: grid {a:?} vs brute {b:?}");
                n += 1; if a.0 > 0 { n_hit += 1; } if a.1 > 0.0 { n_lift += 1; }
            }
        }
        assert!(n_hit > 0 && n_lift > 0 && n_hit < n, "the fixture must exercise hits ({n_hit}), lifts ({n_lift}) and clear pieces (of {n})");
    }

    #[test]
    fn canyon_stall_flood_is_capped_and_fast_on_a_dense_receiver() {
        // Battle Canyon's decal wrapped 17k pieces of dense canyon rock and tested
        // each against 217k candidates (a load that never finished). A 2.5 wu decal on a 1 cm mesh
        // must (a) stop at the documented per-decal cap and (b) run the occlusion pass in
        // well under a second.
        let mut soup = TriSoup::new(4.0);
        dense_floor(&mut soup, 0.0, 4.0, 160, 0); // 51k triangles, 5 cm steps
        quad(&mut soup, 0.03, -1.0, 1.0, -1.0, 1.0, 1 | MESH_INSTANCE); // an overlay in the middle
        soup.shrink();
        let edges = EdgeIndex::build(&soup);
        let t0 = std::time::Instant::now();
        let p = project_engine(&soup, &edges, Vec3::new(0.0, 0.0, 0.1), Vec3::Z, Vec3::X, 2.5, 2.5, 90.0, 0.0);
        let dt = t0.elapsed();
        assert!(p.hit && !p.tris.is_empty());
        assert!(p.surfaces <= MAX_TRIS_PER_DECAL, "flood not capped: {} surfaces", p.surfaces);
        assert!(p.surfaces >= 1000, "the flood must still cover the dense receiver: {} surfaces", p.surfaces);
        // the FLOOR pieces under the slab were dropped (the slab's own top is painted: the primary
        // ray lands on it and the structure below is seeded through it)
        assert!(p.tris.iter().all(|(ps, _)| { let c = (ps[0] + ps[1] + ps[2]) / 3.0; !(c.z < 0.01 && c.x.abs() < 0.9 && c.y.abs() < 0.9) }), "floor under the slab still painted");
        assert!(p.tris.iter().any(|(ps, _)| { let c = (ps[0] + ps[1] + ps[2]) / 3.0; c.z < 0.01 && c.x.abs() > 1.2 }), "the open floor around the slab must be painted");
        assert!(dt.as_secs_f32() < 5.0, "projection took {dt:?} (release: ~50 ms)");
    }
}
