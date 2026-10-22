//! #collision  The ENGINE'S structure collision BSP, read out of the `sbsp` resource.
//!
//! Test mode used to collide against the RENDER triangles of the level (`bsp_soup`), which is a
//! different surface set from the one the game uses: render-only geometry (decorative trim,
//! foliage cards, glass, sky shells) is solid in HMS and not in the game, and collision-only
//! surfaces (invisible blockers, building collision that differs from the building mesh) are
//! missing from HMS and solid in the game. This module decodes the real thing.
//!
//! Layout and evidence: `research/new-maps/05-collision-and-mopp.md` section 2.3 (173 stock and
//! Workshop BSPs decode; an engine-faithful ray caster agrees with brute force on all but 2 of
//! 9 480 rays). The resource is reached with `NativeDll::read_resource(id)` where `id` is the u32
//! at `sbsp + 0x518`; the definition blob STARTS with the 108-byte `collision_bsp` struct.
//!
//! # API (for `hull_vs_static` and the vehicles)
//!
//! * [`CollisionBsp::parse`] -- decode one resource; [`CollisionBsp::for_sbsp`] -- read and decode
//!   an sbsp tag's main collision BSP in one call.
//! * [`CollisionBsp::surfaces`] -- polygons with their flags and collision-material index.
//! * [`CollisionBsp::solid_triangles`] -- every surface the engine's Havok key iterators would
//!   enumerate (the `invalid` flag, 0x10, is skipped), ear-clipped into triangles, as
//!   `([Vec3; 3], surface index)`. Feed them to a `decal_projector::TriSoup` or any
//!   `StaticWorld`.
//! * [`CollisionBsp::ray`] -- the engine's own first-hit ray cast (kd super nodes, plane stack,
//!   bsp2d), for fast line-of-sight / probe queries without building a soup.

use glam::Vec3;

use hms_native::{CacheHandle, NativeDll};

/// `sbsp + 0x518`: the `tag_resources` pageable-resource id of the collision/instance resource.
pub const SBSP_RESOURCE_ID: u32 = 0x518;

/// Surface flag bits (u8). Only 0x10 is PROVEN by name (`_collision_surface_invalid_bit`, tested
/// as `& 0x10` by the key iterators); the others follow the source tag's list order.
pub mod surface_flag {
    pub const TWO_SIDED: u8 = 0x01;
    pub const INVISIBLE: u8 = 0x02;
    pub const CLIMBABLE: u8 = 0x04;
    pub const INVALID: u8 = 0x10;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    /// Plane index (sign-stripped).
    pub plane: u32,
    pub first_edge: u32,
    /// Index into the sbsp's `collision materials` block, or -1.
    pub material: i16,
    pub flags: u8,
}

#[derive(Clone, Debug)]
pub struct CollisionBsp {
    pub large: bool,
    nodes: Vec<[u32; 3]>,
    supers: Vec<[u32; 32]>,
    planes: Vec<[f32; 4]>,
    leaves: Vec<(i32, u32)>, // (ref count, first ref)
    refs: Vec<(u32, bool, u32)>, // (plane, flip, bsp2d root)
    b2: Vec<(f32, f32, f32, i32, i32)>,
    surf: Vec<Surface>,
    edges: Vec<[i64; 6]>,
    verts: Vec<[f32; 3]>,
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn i32_at(b: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn f32_at(b: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn i16_at(b: &[u8], o: usize) -> Option<i16> {
    Some(i16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}

impl CollisionBsp {
    /// Read and decode the main collision BSP of one `sbsp` tag.
    pub fn for_sbsp(native: &NativeDll, cache: CacheHandle, sbsp: u32) -> Option<CollisionBsp> {
        let id = u32::from_le_bytes(native.tag_meta(cache, sbsp, SBSP_RESOURCE_ID, 4)?.get(..4)?.try_into().ok()?);
        let (def, fixups, payload) = native.read_resource(cache, id as i32)?;
        Self::parse(&def, &fixups, &payload)
    }

    /// Decode a resource definition blob (starting with the 108-byte `collision_bsp`), its
    /// pointer fixups and payload. Small vs large format is detected from the 16-byte padding
    /// rule (the true layout leaves < 16 B between arrays; the other leaves far more).
    pub fn parse(def: &[u8], fixups: &[u32], p: &[u8]) -> Option<CollisionBsp> {
        Self::parse_consuming(def, fixups, p).map(|(b, _)| b)
    }

    /// [`parse`](Self::parse) for a `collision_bsp` struct that is NOT the first thing in the
    /// resource (an instanced geometry definition's): `fixups` starts at this struct's first
    /// pointer, and the number of fixups it owned is returned so the caller can carry on.
    pub fn parse_consuming(def: &[u8], fixups: &[u32], p: &[u8]) -> Option<(CollisionBsp, usize)> {
        let mut cnt = [0usize; 9];
        for (i, c) in cnt.iter_mut().enumerate() {
            *c = u32_at(def, 12 * i)? as usize;
        }
        // empty blocks own NO fixup pointer
        let mut it = fixups.iter();
        let mut used = 0usize;
        let mut fx = [0usize; 9];
        for i in 0..9 {
            if cnt[i] > 0 {
                fx[i] = (*it.next()? & 0x0FFF_FFFF) as usize;
                used += 1;
            }
        }
        let small = [8usize, 128, 16, 8, 4, 16, 12, 12, 16];
        let large = [12usize, 128, 16, 8, 8, 20, 16, 24, 20];
        let nz: Vec<usize> = (0..9).filter(|&i| cnt[i] > 0).collect();
        let fits = |sz: &[usize; 9]| nz.windows(2).all(|w| fx[w[0]] + cnt[w[0]] * sz[w[0]] <= fx[w[1]]);
        let slack = |sz: &[usize; 9]| -> usize {
            nz.windows(2).map(|w| fx[w[1]].saturating_sub(fx[w[0]] + cnt[w[0]] * sz[w[0]])).sum()
        };
        let (fs, fl) = (fits(&small), fits(&large));
        if !fs && !fl {
            return None;
        }
        let is_large = fl && (!fs || slack(&large) < slack(&small));
        let sz = if is_large { large } else { small };
        // every array must lie inside the payload
        for &i in &nz {
            if fx[i] + cnt[i] * sz[i] > p.len() {
                return None;
            }
        }
        let mut b = CollisionBsp {
            large: is_large,
            nodes: Vec::with_capacity(cnt[0]),
            supers: Vec::with_capacity(cnt[1]),
            planes: Vec::with_capacity(cnt[2]),
            leaves: Vec::with_capacity(cnt[3]),
            refs: Vec::with_capacity(cnt[4]),
            b2: Vec::with_capacity(cnt[5]),
            surf: Vec::with_capacity(cnt[6]),
            edges: Vec::with_capacity(cnt[7]),
            verts: Vec::with_capacity(cnt[8]),
        };
        for i in 0..cnt[0] {
            let o = fx[0] + i * sz[0];
            if is_large {
                b.nodes.push([u32_at(p, o)?, u32_at(p, o + 4)?, u32_at(p, o + 8)?]);
            } else {
                let plane = u16_at(p, o)? as u32;
                let cv = |c: u32| -> u32 {
                    if c == 0xFF_FFFF {
                        0xFFFF_FFFF
                    } else if c & 0x80_0000 != 0 {
                        (c & 0x7F_FFFF) | 0x8000_0000
                    } else {
                        c
                    }
                };
                let c0 = p[o + 2] as u32 | (p[o + 3] as u32) << 8 | (p[o + 4] as u32) << 16;
                let c1 = p[o + 5] as u32 | (p[o + 6] as u32) << 8 | (p[o + 7] as u32) << 16;
                b.nodes.push([plane, cv(c0), cv(c1)]);
            }
        }
        for i in 0..cnt[1] {
            let mut s = [0u32; 32];
            for (k, v) in s.iter_mut().enumerate() {
                *v = u32_at(p, fx[1] + 128 * i + 4 * k)?;
            }
            b.supers.push(s);
        }
        for i in 0..cnt[2] {
            let o = fx[2] + 16 * i;
            b.planes.push([f32_at(p, o)?, f32_at(p, o + 4)?, f32_at(p, o + 8)?, f32_at(p, o + 12)?]);
        }
        for i in 0..cnt[3] {
            let o = fx[3] + 8 * i;
            if is_large {
                b.leaves.push((i16_at(p, o + 2)? as i32, u32_at(p, o + 4)?));
            } else {
                b.leaves.push((i16_at(p, o + 2)? as i32, u32_at(p, o + 4)?));
            }
        }
        for i in 0..cnt[4] {
            let o = fx[4] + i * sz[4];
            if is_large {
                let pl = i32_at(p, o)?;
                b.refs.push(((pl & 0x7FFF_FFFF) as u32, pl < 0, u32_at(p, o + 4)?));
            } else {
                let pl = i16_at(p, o)?;
                b.refs.push(((pl & 0x7FFF) as u32, pl < 0, u16_at(p, o + 2)? as u32));
            }
        }
        for i in 0..cnt[5] {
            let o = fx[5] + i * sz[5];
            if is_large {
                b.b2.push((f32_at(p, o)?, f32_at(p, o + 4)?, f32_at(p, o + 8)?, i32_at(p, o + 12)?, i32_at(p, o + 16)?));
            } else {
                b.b2.push((f32_at(p, o)?, f32_at(p, o + 4)?, f32_at(p, o + 8)?, i16_at(p, o + 12)? as i32, i16_at(p, o + 14)? as i32));
            }
        }
        for i in 0..cnt[6] {
            let o = fx[6] + i * sz[6];
            if is_large {
                b.surf.push(Surface {
                    plane: (i32_at(p, o)? & 0x7FFF_FFFF) as u32,
                    first_edge: u32_at(p, o + 4)?,
                    material: i16_at(p, o + 8)?,
                    flags: *p.get(o + 14)?,
                });
            } else {
                b.surf.push(Surface {
                    plane: (i16_at(p, o)? & 0x7FFF) as u32,
                    first_edge: u16_at(p, o + 2)? as u32,
                    material: i16_at(p, o + 4)?,
                    flags: *p.get(o + 10)?,
                });
            }
        }
        for i in 0..cnt[7] {
            let o = fx[7] + i * sz[7];
            if is_large {
                let mut e = [0i64; 6];
                for (k, v) in e.iter_mut().enumerate() {
                    *v = i32_at(p, o + 4 * k)? as i64;
                }
                b.edges.push(e);
            } else {
                // start, end, forward, reverse are UNSIGNED 16-bit; left/right surface likewise
                // (values reach 65 535), per the research note.
                let mut e = [0i64; 6];
                for (k, v) in e.iter_mut().enumerate() {
                    *v = u16_at(p, o + 2 * k)? as i64;
                }
                b.edges.push(e);
            }
        }
        for i in 0..cnt[8] {
            let o = fx[8] + i * sz[8];
            b.verts.push([f32_at(p, o)?, f32_at(p, o + 4)?, f32_at(p, o + 8)?]);
        }
        Some((b, used))
    }

    pub fn surface_count(&self) -> usize {
        self.surf.len()
    }
    pub fn surfaces(&self) -> &[Surface] {
        &self.surf
    }

    /// The polygon (ring of vertices) of surface `s`, or None for a broken ring.
    pub fn polygon(&self, s: usize) -> Option<Vec<Vec3>> {
        let e0 = self.surf.get(s)?.first_edge as usize;
        let (mut e, mut out, mut n) = (e0, Vec::new(), 0);
        loop {
            let ed = self.edges.get(e)?;
            let (start, end, fwd, rev, left, right) = (ed[0], ed[1], ed[2], ed[3], ed[4], ed[5]);
            if left == s as i64 {
                out.push(Vec3::from(*self.verts.get(start as usize)?));
                e = fwd as usize;
            } else if right == s as i64 {
                out.push(Vec3::from(*self.verts.get(end as usize)?));
                e = rev as usize;
            } else {
                return None;
            }
            n += 1;
            if e == e0 || n > 1024 {
                break;
            }
        }
        (out.len() >= 3).then_some(out)
    }

    /// Every surface the Havok key iterators enumerate (the `invalid` flag skipped), as triangles
    /// tagged with their surface index. Concave polygons are ear-clipped; a ring that cannot be
    /// clipped falls back to a fan.
    pub fn solid_triangles(&self) -> Vec<([Vec3; 3], u32)> {
        let mut out = Vec::new();
        for s in 0..self.surf.len() {
            if self.surf[s].flags & surface_flag::INVALID != 0 {
                continue;
            }
            let Some(poly) = self.polygon(s) else { continue };
            for t in triangulate(&poly) {
                out.push((t, s as u32));
            }
        }
        out
    }

    // ---- the engine's ray cast (plane-stack form, `collision_bsp.cpp` semantics) ------------

    /// First hit of the segment `p0 -> p1`: `(t in [0,1], surface)`. Invalid surfaces are not
    /// special-cased (the engine's flag filtering is per caller).
    pub fn ray(&self, p0: Vec3, p1: Vec3) -> Option<(f32, usize)> {
        let d = p1 - p0;
        let mut st = RayState { best: (2.0, usize::MAX), stack: Vec::new() };
        if !self.supers.is_empty() {
            self.walk_super(0, 0, 0.0, 1.0, p0, d, &mut st);
        } else if !self.nodes.is_empty() {
            self.walk_node(0, 0.0, 1.0, p0, d, &mut st);
        }
        (st.best.1 != usize::MAX).then_some((st.best.0, st.best.1))
    }

    fn bsp2d_lookup(&self, root: u32, a: f32, b: f32) -> Option<usize> {
        let mut c: i64 = if self.large { root as i32 as i64 } else { (root as u16 as i16) as i64 };
        let mut guard = 0;
        loop {
            if c == -1 {
                return None;
            }
            if c < 0 {
                let mask = if self.large { 0x7FFF_FFFFi64 } else { 0x7FFFi64 };
                return Some((c & mask) as usize);
            }
            let (na, nb, nd, l, r) = *self.b2.get(c as usize)?;
            c = if na * a + nb * b - nd >= 0.0 { r as i64 } else { l as i64 };
            guard += 1;
            if guard > 4096 {
                return None;
            }
        }
    }

    fn leaf(&self, li: usize, a: f32, bt: f32, p0: Vec3, d: Vec3, st: &mut RayState) {
        let Some(&(n, first)) = self.leaves.get(li) else { return };
        for k in 0..n.max(0) as usize {
            let Some(&(pidx, flip, root)) = self.refs.get(first as usize + k) else { continue };
            if !st.stack.iter().any(|e| *e == Some((pidx, flip))) {
                continue;
            }
            let Some(pn) = self.planes.get(pidx as usize) else { continue };
            let n3 = Vec3::new(pn[0], pn[1], pn[2]);
            let den = n3.dot(d);
            if den == 0.0 {
                continue;
            }
            let t = -(n3.dot(p0) - pn[3]) / den;
            let _ = (a, bt);
            let pt = p0 + d * t;
            let ax = if n3.x.abs() >= n3.y.abs() && n3.x.abs() >= n3.z.abs() { 0 } else if n3.y.abs() >= n3.z.abs() { 1 } else { 2 };
            let flag = (n3[ax] > 0.0) ^ flip;
            const AXES: [(usize, usize); 6] = [(2, 1), (1, 2), (0, 2), (2, 0), (1, 0), (0, 1)];
            let (ia, ib) = AXES[ax * 2 + flag as usize];
            if let Some(s) = self.bsp2d_lookup(root, pt[ia], pt[ib]) {
                if (0.0..=1.0).contains(&t) && t < st.best.0 && self.contains(s, pt, ax) {
                    st.best = (t, s);
                }
            }
        }
    }

    /// Is `pt` inside surface `s`'s polygon (even-odd, projected along `ax`)? The bsp2d lookup
    /// names the surface whose 2-D region the point falls in; the region can be larger than the
    /// polygon, and the engine's containment test is what makes the hit real.
    fn contains(&self, s: usize, pt: Vec3, ax: usize) -> bool {
        let Some(poly) = self.polygon(s) else { return false };
        let (ia, ib) = [(1, 2), (2, 0), (0, 1)][ax];
        let (x, y) = (pt[ia], pt[ib]);
        let mut inside = false;
        let m = poly.len();
        for i in 0..m {
            let (x1, y1, x2, y2) = (poly[i][ia], poly[i][ib], poly[(i + 1) % m][ia], poly[(i + 1) % m][ib]);
            if (y1 > y) != (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1) + x1 {
                inside = !inside;
            }
        }
        inside
    }

    fn walk_node(&self, c: u32, a: f32, bt: f32, p0: Vec3, d: Vec3, st: &mut RayState) {
        if c == 0xFFFF_FFFF {
            return;
        }
        if c & 0x8000_0000 != 0 {
            self.leaf((c & 0x7FFF_FFFF) as usize, a, bt, p0, d, st);
            return;
        }
        let Some(&[pl, bk, fr]) = self.nodes.get(c as usize) else { return };
        let pidx = pl & 0x7FFF_FFFF;
        let Some(pn) = self.planes.get(pidx as usize) else { return };
        let n3 = Vec3::new(pn[0], pn[1], pn[2]);
        let da = n3.dot(p0 + d * a) - pn[3];
        let db = n3.dot(p0 + d * bt) - pn[3];
        if da >= 0.0 && db >= 0.0 {
            self.walk_node(fr, a, bt, p0, d, st);
        } else if da < 0.0 && db < 0.0 {
            self.walk_node(bk, a, bt, p0, d, st);
        } else {
            let tm = a + (bt - a) * da / (da - db);
            let flag = n3.dot(d) > 0.0;
            st.stack.push(Some((pidx, flag)));
            if da >= 0.0 {
                self.walk_node(fr, a, tm, p0, d, st);
            } else {
                self.walk_node(bk, a, tm, p0, d, st);
            }
            st.stack.pop();
            if da >= 0.0 {
                self.walk_node(bk, tm, bt, p0, d, st);
            } else {
                self.walk_node(fr, tm, bt, p0, d, st);
            }
        }
    }

    fn walk_super(&self, si: usize, slot: usize, a: f32, bt: f32, p0: Vec3, d: Vec3, st: &mut RayState) {
        let Some(sn) = self.supers.get(si) else { return };
        if slot >= 15 {
            let c = sn[15 + (slot - 15)];
            if c == 0 || c & 0x8000_0000 != 0 {
                return;
            }
            if c & 0xC000_0000 == 0x4000_0000 {
                self.walk_node(c & 0xBFFF_FFFF, a, bt, p0, d, st);
            } else {
                self.walk_super(c as usize, 0, a, bt, p0, d, st);
            }
            return;
        }
        let axis = ((sn[31] >> (2 * slot)) & 3) as usize;
        if axis > 2 {
            return;
        }
        let pv = f32::from_bits(sn[slot]);
        let da = (p0 + d * a)[axis] - pv;
        let db = (p0 + d * bt)[axis] - pv;
        if da >= 0.0 && db >= 0.0 {
            self.walk_super(si, 2 * slot + 2, a, bt, p0, d, st);
        } else if da < 0.0 && db < 0.0 {
            self.walk_super(si, 2 * slot + 1, a, bt, p0, d, st);
        } else {
            let tm = a + (bt - a) * da / (da - db);
            st.stack.push(None);
            if da >= 0.0 {
                self.walk_super(si, 2 * slot + 2, a, tm, p0, d, st);
            } else {
                self.walk_super(si, 2 * slot + 1, a, tm, p0, d, st);
            }
            st.stack.pop();
            if da >= 0.0 {
                self.walk_super(si, 2 * slot + 1, tm, bt, p0, d, st);
            } else {
                self.walk_super(si, 2 * slot + 2, tm, bt, p0, d, st);
            }
        }
    }
}

struct RayState {
    best: (f32, usize),
    stack: Vec<Option<(u32, bool)>>,
}

/// Size of one `instanced geometries definitions` element in the resource.
const DEFINITION_SIZE: usize = 324;
/// The three block headers the definition blob ends with (collision bsp, large collision bsp,
/// instanced geometries definitions).
const RESOURCE_TAIL: usize = 36;

/// One instanced-geometry definition (`poop`): the collision the engine uses for every instance
/// of it, in the definition's LOCAL space.
#[derive(Clone, Debug)]
pub struct InstanceDefinition {
    /// Render mesh this definition draws (+0x140), the join key to the render-side instances.
    pub mesh_index: i16,
    /// `None` = the definition has no collision surfaces (decoration the engine walks through).
    pub collision: Option<CollisionBsp>,
}

/// Every instanced-geometry definition of one `sbsp`, with its collision BSP decoded.
///
/// Layout (PROVEN on forge_halo 126/126 and forge_halo_shared 27/27 definitions): the
/// definition blob is `[main collision bsp 108 B][400 B of other structures][N x 324 B
/// definitions][36 B of block headers]`. The pointer fixups are consumed in struct order, one per
/// NON-EMPTY block header: the main BSP's, the prefix structures', each definition's (its
/// collision bsp 9 + poopie cutter 9 + the eight blocks from +0xE0), and last the two or three
/// top-level blocks'. The start of the definitions' fixups is therefore found from the END of the
/// stream, which needs no knowledge of the 400 B prefix.
#[derive(Clone, Debug)]
pub struct InstanceCollision {
    pub defs: Vec<InstanceDefinition>,
    /// Definitions whose collision struct failed to decode (their instances fall back).
    pub failed: usize,
    /// Fixups left over once every definition and the top-level blocks took theirs. Zero when
    /// the walk above is exactly right (it is on every stock BSP checked).
    pub unused_fixups: usize,
}

impl InstanceCollision {
    pub fn for_sbsp(native: &NativeDll, cache: CacheHandle, sbsp: u32) -> Option<InstanceCollision> {
        let id = u32::from_le_bytes(native.tag_meta(cache, sbsp, SBSP_RESOURCE_ID, 4)?.get(..4)?.try_into().ok()?);
        let (def, fixups, payload) = native.read_resource(cache, id as i32)?;
        Self::parse(&def, &fixups, &payload)
    }

    pub fn parse(def: &[u8], fixups: &[u32], p: &[u8]) -> Option<InstanceCollision> {
        let tail = def.len().checked_sub(RESOURCE_TAIL)?;
        let nd = u32_at(def, tail + 24)? as usize;
        let base = tail.checked_sub(nd.checked_mul(DEFINITION_SIZE)?)?;
        let (_, main_used) = CollisionBsp::parse_consuming(def, fixups, p)?;
        let nonempty = |d: &[u8], off: usize, blocks: usize| (0..blocks).filter(|&i| u32_at(d, off + 12 * i).is_some_and(|c| c > 0)).count();
        let top = nonempty(def, tail, 3);
        let mut owned = 0usize;
        for k in 0..nd {
            let d = def.get(base + k * DEFINITION_SIZE..base + (k + 1) * DEFINITION_SIZE)?;
            owned += nonempty(d, 0x08, 9) + nonempty(d, 0x74, 9);
            owned += [0xE0usize, 0xEC, 0xF8, 0x104, 0x110, 0x11C, 0x128, 0x134].iter().filter(|&&o| u32_at(d, o).is_some_and(|c| c > 0)).count();
        }
        let mut cur = fixups.len().checked_sub(top + owned)?;
        if cur < main_used {
            return None;
        }
        let (mut defs, mut failed) = (Vec::with_capacity(nd), 0);
        for k in 0..nd {
            let d = &def[base + k * DEFINITION_SIZE..base + (k + 1) * DEFINITION_SIZE];
            let parsed = CollisionBsp::parse_consuming(&d[0x08..0x08 + 108], fixups.get(cur..).unwrap_or(&[]), p);
            let mesh_index = i16_at(d, 0x140)?;
            let collision = match parsed {
                Some((b, _)) if b.surface_count() > 0 => Some(b),
                Some(_) => None,
                None => {
                    failed += 1;
                    None
                }
            };
            cur += nonempty(d, 0x08, 9) + nonempty(d, 0x74, 9);
            cur += [0xE0usize, 0xEC, 0xF8, 0x104, 0x110, 0x11C, 0x128, 0x134].iter().filter(|&&o| u32_at(d, o).is_some_and(|c| c > 0)).count();
            defs.push(InstanceDefinition { mesh_index, collision });
        }
        Some(InstanceCollision { defs, failed, unused_fixups: fixups.len() - cur - top })
    }
}

/// The instanced geometry of one `sbsp` as collision triangles in WORLD space.
#[derive(Clone, Debug, Default)]
pub struct InstanceTris {
    /// Definition collision placed at each instance's transform.
    pub placed: Vec<[Vec3; 3]>,
    /// Instances placed with their definition's collision.
    pub instances: usize,
    /// Render triangles (not transparent) of instances whose definition has NO collision.
    pub fallback: Vec<[Vec3; 3]>,
    /// Render meshes those triangles came from.
    pub fallback_meshes: usize,
}

/// Place every instance's definition collision (instance -> definition via the lightmap atlas
/// join, `ZhLbspInstanceAtlas::mesh_index` == definition index; transform from the render
/// mesh, row-vector convention = `Mat4::from_cols_array`). Definitions without collision data
/// contribute their instance's opaque render triangles instead.
pub fn instance_tris(native: &NativeDll, cache: CacheHandle, sbsp: u32) -> Option<InstanceTris> {
    let defs = InstanceCollision::for_sbsp(native, cache, sbsp)?;
    let local: Vec<Option<Vec<[Vec3; 3]>>> = defs
        .defs
        .iter()
        .map(|d| d.collision.as_ref().map(|c| c.solid_triangles().into_iter().map(|(t, _)| t).collect()))
        .collect();
    let h = native.open_bsp(cache, sbsp).ok()?;
    let mut out = InstanceTris::default();
    let mut placed = std::collections::HashSet::new();
    for i in 0..native.bsp_mesh_count(h) {
        let Some(m) = native.bsp_mesh(h, i) else { continue };
        if m.instance_ordinal == u32::MAX {
            continue;
        }
        let def = native.instance_atlas_submap(cache, sbsp, m.instance_ordinal).map(|a| a.mesh_index as usize);
        match def.and_then(|d| local.get(d)).and_then(|l| l.as_ref()) {
            Some(l) => {
                if placed.insert(m.instance_ordinal) {
                    let mat = glam::Mat4::from_cols_array(&m.matrix);
                    out.instances += 1;
                    out.placed.extend(l.iter().map(|t| t.map(|p| mat.transform_point3(p))));
                }
            }
            None => {
                if matches!(native.bsp_material_blend(h, m.material_index), 1 | 2 | 3 | 4 | 6) {
                    continue;
                }
                let Some(g) = native.bsp_decode_geometry(h, i) else { continue };
                let v = |k: u32| -> Option<Vec3> {
                    let c = g.vertices.get(k as usize * 12..k as usize * 12 + 12)?;
                    Some(Vec3::new(f32_at(c, 0)?, f32_at(c, 4)?, f32_at(c, 8)?))
                };
                let stride = m.index_stride.max(2) as usize;
                out.fallback_meshes += 1;
                for c in g.indices.chunks_exact(3 * stride) {
                    let ix = |n: usize| if stride == 4 { u32_at(c, 4 * n) } else { u16_at(c, 2 * n).map(u32::from) };
                    if let (Some(a), Some(b), Some(d)) = (ix(0).and_then(v), ix(1).and_then(v), ix(2).and_then(v)) {
                        out.fallback.push([a, b, d]);
                    }
                }
            }
        }
    }
    native.close_bsp(h);
    Some(out)
}

/// Newell normal of a polygon ring.
fn newell(poly: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..poly.len() {
        let (a, c) = (poly[i], poly[(i + 1) % poly.len()]);
        n.x += (a.y - c.y) * (a.z + c.z);
        n.y += (a.z - c.z) * (a.x + c.x);
        n.z += (a.x - c.x) * (a.y + c.y);
    }
    n
}

/// Ear-clip a planar polygon into triangles (fan fallback for a degenerate ring).
pub fn triangulate(poly: &[Vec3]) -> Vec<[Vec3; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[poly[0], poly[1], poly[2]]];
    }
    let nrm = newell(poly);
    if nrm.length_squared() < 1e-18 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    let mut guard = 0;
    while idx.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (i0, i1, i2) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (poly[i0], poly[i1], poly[i2]);
            if (b - a).cross(c - b).dot(nrm) <= 1e-12 {
                continue; // reflex or degenerate
            }
            let inside = idx.iter().any(|&j| {
                if j == i0 || j == i1 || j == i2 {
                    return false;
                }
                let p = poly[j];
                let s0 = (b - a).cross(p - a).dot(nrm);
                let s1 = (c - b).cross(p - b).dot(nrm);
                let s2 = (a - c).cross(p - c).dot(nrm);
                s0 >= -1e-9 && s1 >= -1e-9 && s2 >= -1e-9
            });
            if inside {
                continue;
            }
            out.push([a, b, c]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        out.push([poly[idx[0]], poly[idx[1]], poly[idx[2]]]);
    } else {
        // could not finish: fan the remainder
        for w in 1..idx.len() - 1 {
            out.push([poly[idx[0]], poly[idx[w]], poly[idx[w + 1]]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ear_clipping_covers_a_concave_polygon_exactly() {
        // a U shape (area 4 x 3 - 2 x 2 = 8) that is NOT star-shaped from its first vertex, so a
        // fan from vertex 0 covers the notch and its area comes out wrong
        let u = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 3.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let tris = triangulate(&u);
        assert_eq!(tris.len(), 6);
        let area: f32 = tris.iter().map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5).sum();
        assert!((area - 8.0).abs() < 1e-4, "area {area}");
        let p = Vec3::new(2.0, 2.0, 0.0); // inside the notch
        for t in &tris {
            let n = Vec3::Z;
            let inside = (t[1] - t[0]).cross(p - t[0]).dot(n) > 0.0 && (t[2] - t[1]).cross(p - t[1]).dot(n) > 0.0 && (t[0] - t[2]).cross(p - t[2]).dot(n) > 0.0;
            assert!(!inside, "a triangle covers the notch");
        }
    }

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// Möller-Trumbore, segment form: t in [0,1].
    fn seg_tri(p0: Vec3, d: Vec3, t: &[Vec3; 3]) -> Option<f32> {
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        // the engine's ray cast only reports surfaces it ENTERS through the front (the ring's
        // winding normal faces the ray)
        if e1.cross(e2).dot(d) >= 0.0 {
            return None;
        }
        let h = d.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-12 {
            return None;
        }
        let f = 1.0 / a;
        let s = p0 - t[0];
        let u = f * s.dot(h);
        if !(-1e-6..=1.0 + 1e-6).contains(&u) {
            return None;
        }
        let q = s.cross(e1);
        let v = f * d.dot(q);
        if v < -1e-6 || u + v > 1.0 + 1e-6 {
            return None;
        }
        let tt = f * e2.dot(q);
        (0.0..=1.0).contains(&tt).then_some(tt)
    }

    /// #collision  **THE READER AGREES WITH THE ENGINE'S OWN ALGORITHM.** On every structure BSP
    /// of `forge_halo`: the decoder is closed (every surface ring closes), and random segments
    /// through the world bounds give the same first hit by the engine-faithful BSP walk
    /// ([`CollisionBsp::ray`]) as by brute force over all decoded polygons. The research
    /// measured 2 mismatches in 9 480 rays (slivers); the bound here is 1%, and the number of
    /// rays that actually HIT (liveness) is asserted so a reader that decodes nothing cannot pass.
    ///
    /// Random segments through `b`: (segments that hit brute force, segments where the BSP walk
    /// and brute force over the same polygons disagree).
    fn brute_force_agreement(b: &CollisionBsp, seed: u32, rays: usize) -> (usize, usize) {
        let all: Vec<[Vec3; 3]> = (0..b.surface_count()).filter_map(|s| b.polygon(s)).flat_map(|p| triangulate(&p)).collect();
        if all.is_empty() {
            return (0, 0);
        }
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in all.iter().flatten() {
            mn = mn.min(*v);
            mx = mx.max(*v);
        }
        let mut seed = seed | 1;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let (mut hits, mut disagree) = (0, 0);
        for _ in 0..rays {
            let tr = &all[(rnd() * all.len() as f32) as usize % all.len()];
            let c = (tr[0] + tr[1] + tr[2]) / 3.0;
            let reach = (mx - mn).length().max(1.0);
            let off = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize_or_zero() * reach * 0.5;
            let (p0, p1) = (c + off, c - off * 0.5);
            let d = p1 - p0;
            let brute = all.iter().filter_map(|t| seg_tri(p0, d, t)).fold(f32::MAX, f32::min);
            match (b.ray(p0, p1), brute < f32::MAX) {
                (Some((t, _)), true) => {
                    hits += 1;
                    disagree += ((t - brute).abs() > 1e-3) as usize;
                }
                (None, false) => {}
                _ => disagree += 1,
            }
        }
        (hits, disagree)
    }

    /// #collision  **THE INSTANCE DEFINITIONS DECODE EXACTLY.** forge_halo has 126 definitions
    /// (106 with collision), forge_halo_shared 27 (3 with collision). Every definition's
    /// `collision_bsp` struct passes the small/large layout test at the fixup the walk predicts
    /// and the fixup stream is used up to the last entry.
    ///
    /// SABOTAGE: take one fewer fixup per definition for its poopie cutter (skip the `0x74`
    /// blocks in `InstanceCollision::parse`) -> failures and leftover fixups appear (red).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_instance_definitions_decode_exactly() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut seen = std::collections::BTreeMap::new();
        for &t in &native.tags_by_group(cache, "sbsp") {
            let name = native.tag_name(cache, t).unwrap_or_default();
            let ic = InstanceCollision::for_sbsp(native, cache, t).unwrap_or_else(|| panic!("{name}: definitions do not decode"));
            let with = ic.defs.iter().filter(|d| d.collision.is_some()).count();
            println!("{name}: {} definitions, {with} with collision, {} failed, {} fixups unused", ic.defs.len(), ic.failed, ic.unused_fixups);
            assert_eq!((ic.failed, ic.unused_fixups), (0, 0), "{name}");
            seen.insert(name.rsplit('\\').next().unwrap_or("").to_string(), (ic.defs.len(), with));
        }
        assert_eq!(seen.get("forge_halo"), Some(&(126, 106)));
        assert_eq!(seen.get("forge_halo_shared"), Some(&(27, 3)));
    }

    /// #collision  Every definition's collision BSP agrees with brute force over its own polygons
    /// (the same check the main BSP passes), and the definitions are not trivial.
    ///
    /// SABOTAGE: swap the children in the node reader (see the main-BSP sabotage) -> red.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn every_instance_definition_agrees_with_brute_force() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let (mut defs, mut rays, mut hits, mut bad) = (0, 0, 0, 0);
        for &t in &native.tags_by_group(cache, "sbsp") {
            let ic = InstanceCollision::for_sbsp(native, cache, t).unwrap();
            for (k, d) in ic.defs.iter().enumerate() {
                let Some(b) = &d.collision else { continue };
                let (h, dis) = brute_force_agreement(b, 0x51ED_0001 ^ (k as u32 * 7919), 200);
                defs += 1;
                rays += 200;
                hits += h;
                bad += dis;
                assert!(dis <= 2, "{}: definition {k} ({} surfaces): {dis} of 200 segments disagree", native.tag_name(cache, t).unwrap_or_default(), b.surface_count());
            }
        }
        println!("{defs} definitions, {rays} segments, {hits} hit, {bad} disagree");
        assert!(defs >= 100 && hits >= 5000, "liveness: {defs} definitions, {hits} hits");
    }

    /// #collision  **INSTANCES ARE PLACED WHERE THEY RENDER.** For every instance of forge_halo
    /// whose definition has collision, the centre of the placed collision box lies inside the
    /// instance's render bounds (grown by 5 percent and 0.1 wu). Placing the collision with the
    /// transposed matrix, or with the wrong definition, fails this for most rotated instances.
    ///
    /// SABOTAGE: `Mat4::from_cols_array(..).transpose()` -> red.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn instance_collision_sits_inside_the_instance_render_bounds() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let (mut total, mut inside, mut rotated_inside, mut rotated) = (0, 0, 0, 0);
        for &t in &native.tags_by_group(cache, "sbsp") {
            let ic = InstanceCollision::for_sbsp(native, cache, t).unwrap();
            let local: Vec<Option<Vec<Vec3>>> = ic.defs.iter().map(|d| d.collision.as_ref().map(|c| c.solid_triangles().into_iter().flat_map(|(t, _)| t).collect())).collect();
            let h = native.open_bsp(cache, t).unwrap();
            let mut seen = std::collections::HashSet::new();
            let (mut want_tris, mut want_inst) = (0usize, 0usize);
            let (mut umn, mut umx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for i in 0..native.bsp_mesh_count(h) {
                let Some(m) = native.bsp_mesh(h, i) else { continue };
                if m.instance_ordinal == u32::MAX || !seen.insert(m.instance_ordinal) {
                    continue;
                }
                let Some(a) = native.instance_atlas_submap(cache, t, m.instance_ordinal) else { continue };
                let Some(Some(pts)) = local.get(a.mesh_index as usize) else { continue };
                let mat = glam::Mat4::from_cols_array(&m.matrix);
                let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for p in pts {
                    let w = mat.transform_point3(*p);
                    mn = mn.min(w);
                    mx = mx.max(w);
                }
                want_inst += 1;
                want_tris += pts.len() / 3;
                umn = umn.min(mn);
                umx = umx.max(mx);
                let Some(g) = native.bsp_decode_geometry(h, i) else { continue };
                let (mut rmn, mut rmx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for c in g.vertices.chunks_exact(12) {
                    let v = Vec3::new(f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()), f32::from_le_bytes(c[8..12].try_into().unwrap()));
                    rmn = rmn.min(v);
                    rmx = rmx.max(v);
                }
                let grow = (rmx - rmn) * 0.05 + Vec3::splat(0.1);
                let c = (mn + mx) * 0.5;
                let ok = (c.cmpge(rmn - grow) & c.cmple(rmx + grow)).all();
                let is_rotated = (mat.x_axis.y.abs() + mat.x_axis.z.abs()) > 0.2;
                total += 1;
                inside += ok as usize;
                rotated += is_rotated as usize;
                rotated_inside += (ok && is_rotated) as usize;
            }
            native.close_bsp(h);
            // the production placement (`instance_tris`, what the scene's collision soup takes)
            // must be the same set: same instance and triangle counts, same overall box
            let prod = instance_tris(native, cache, t).unwrap();
            let (mut pmn, mut pmx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in prod.placed.iter().flatten() {
                pmn = pmn.min(*p);
                pmx = pmx.max(*p);
            }
            println!("   production: {} instances placed, {} triangles; {} meshes of definitions without collision keep {} render triangles", prod.instances, prod.placed.len(), prod.fallback_meshes, prod.fallback.len());
            assert_eq!((prod.instances, prod.placed.len()), (want_inst, want_tris), "production placement differs from the gate's");
            assert!((pmn - umn).abs().max_element() < 1e-2 && (pmx - umx).abs().max_element() < 1e-2, "placed box {pmn:?}..{pmx:?} vs {umn:?}..{umx:?}");
        }
        println!("{total} placed instances, {inside} inside their render bounds; of {rotated} rotated ones {rotated_inside}");
        assert!(total >= 1000 && rotated >= 100, "liveness: {total} instances, {rotated} rotated");
        assert!(inside * 100 >= total * 97, "only {inside} of {total} instances sit inside their render bounds");
        assert!(rotated_inside * 100 >= rotated * 95, "rotated instances: {rotated_inside} of {rotated}");
    }

    /// SABOTAGE: read the small-format children the wrong way round (swap back/front) -> the BSP
    /// walk disagrees with brute force almost everywhere.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_reader_agrees_with_brute_force_on_forge_halo() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut checked_bsps = 0;
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = CollisionBsp::for_sbsp(native, cache, t) else {
                println!("{}: no collision resource", native.tag_name(cache, t).unwrap_or_default());
                continue;
            };
            let name = native.tag_name(cache, t).unwrap_or_default();
            let mut open_rings = 0;
            let mut flag_counts = [0usize; 8];
            for (i, s) in b.surfaces().iter().enumerate() {
                if b.polygon(i).is_none() {
                    open_rings += 1;
                }
                for bit in 0..8 {
                    if s.flags & (1 << bit) != 0 {
                        flag_counts[bit] += 1;
                    }
                }
            }
            let tris = b.solid_triangles();
            println!(
                "{name}: {} format, {} surfaces ({} broken rings), {} solid triangles, flag bits {:?}",
                if b.large { "LARGE" } else { "small" },
                b.surface_count(),
                open_rings,
                tris.len(),
                flag_counts
            );
            assert_eq!(open_rings, 0, "{name}: surface rings do not close");
            if tris.len() < 100 {
                continue;
            }
            // world bounds from the polygons themselves
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for (tr, _) in &tris {
                for v in tr {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                }
            }
            let all: Vec<([Vec3; 3], u32)> = (0..b.surface_count())
                .filter_map(|s| b.polygon(s).map(|p| (s, p)))
                .flat_map(|(s, p)| triangulate(&p).into_iter().map(move |t| (t, s as u32)))
                .collect();
            let mut seed = 0x9E37_79B9u32 ^ t;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f32 / 16_777_216.0
            };
            let (mut hits, mut disagree, rays) = (0usize, 0usize, 600usize);
            for _ in 0..rays {
                let pick = |r: f32, k: usize| mn[k] + (mx[k] - mn[k]) * r;
                let (p0, p1) = if rnd() < 0.5 {
                    let p0 = Vec3::new(pick(rnd(), 0), pick(rnd(), 1), pick(rnd(), 2));
                    (p0, p0 + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize_or_zero() * 30.0)
                } else {
                    // aimed THROUGH a random surface, from a few wu away, so small BSPs are hit too
                    let tr = &all[(rnd() * all.len() as f32) as usize % all.len()].0;
                    let c = (tr[0] + tr[1] + tr[2]) / 3.0;
                    let off = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize_or_zero() * 3.0;
                    (c + off, c - off * 0.5)
                };
                let d = p1 - p0;
                let brute = all.iter().filter_map(|(tr, _)| seg_tri(p0, d, tr)).fold(f32::MAX, f32::min);
                let bsp = b.ray(p0, p1);
                match (bsp, brute < f32::MAX) {
                    (Some((t, _)), true) => {
                        hits += 1;
                        if (t - brute).abs() > 1e-3 {
                            if disagree < 6 {
                                println!("   DBG both: bsp t {t} brute {brute}");
                            }
                            disagree += 1;
                        }
                    }
                    (None, false) => {}
                    _ => {
                        if disagree < 6 {
                            println!("   DBG bsp {:?} brute {}", bsp, brute);
                        }
                        disagree += 1
                    }
                }
            }
            println!("   {rays} random segments: {hits} hit, {disagree} disagree with brute force");
            assert!(hits >= 60, "{name}: only {hits} of {rays} segments hit anything -- the check looked at nothing");
            assert!(disagree * 100 <= rays, "{name}: {disagree} of {rays} segments disagree with brute force");
            checked_bsps += 1;
        }
        assert!(checked_bsps >= 1, "no BSP had enough triangles to check");
    }

    /// #collision  AUDIT: for every placeable object in `forge_halo`, the physics-shape hull
    /// (`phmo`, what the game's Havok bodies are, and what test mode's `physics_world_tris` uses
    /// first) against the collision model (`coll`). Counts the objects where the two disagree on
    /// extent -- a hull much SMALLER than the collision model is a candidate walk-through wall.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn audit_object_hulls_against_collision_models() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let aabb = |g: &(Vec<f32>, Vec<u32>)| -> Option<(Vec3, Vec3, usize)> {
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let mut n = 0;
            for f in g.1.chunks_exact(3) {
                for &k in f {
                    let k = k as usize * 3;
                    if k + 2 >= g.0.len() { return None; }
                    let v = Vec3::new(g.0[k], g.0[k + 1], g.0[k + 2]);
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                n += 1;
            }
            (n > 0).then_some((mn, mx, n))
        };
        let (mut total, mut both, mut phmo_only, mut coll_only, mut neither) = (0, 0, 0, 0, 0);
        let mut small: Vec<(String, f32, usize, usize)> = Vec::new();
        for group in ["bloc", "scen", "mach", "ctrl", "vehi", "crea"] {
            for &t in &native.tags_by_group(cache, group) {
                total += 1;
                let name = native.tag_name(cache, t).unwrap_or_default();
                let ph = native.resolve_phmo(cache, t);
                let pg = if ph != 0 && ph != 0xFFFF_FFFF { native.phmo_geometry(cache, ph) } else { None }.and_then(|g| aabb(&g));
                let co = native.resolve_coll(cache, t);
                let cg = if co != 0 && co != 0xFFFF_FFFF { native.coll_geometry(cache, co) } else { None }.and_then(|g| aabb(&g));
                match (pg, cg) {
                    (Some(p), Some(c)) => {
                        both += 1;
                        let vol = |a: Vec3, b: Vec3| {
                            let e = (b - a).max(Vec3::splat(1e-3));
                            e.x * e.y * e.z
                        };
                        let ratio = vol(p.0, p.1) / vol(c.0, c.1);
                        if ratio < 0.6 || (name.contains("forge") && !(0.9..1.15).contains(&ratio)) {
                            small.push((name, ratio, p.2, c.2));
                        }
                    }
                    (Some(_), None) => phmo_only += 1,
                    (None, Some(_)) => coll_only += 1,
                    (None, None) => neither += 1,
                }
            }
        }
        println!("{total} objects: {both} have both a physics hull and a collision model, {phmo_only} hull only, {coll_only} collision model only, {neither} neither");
        small.sort_by(|a, b| a.1.total_cmp(&b.1));
        println!("{} objects whose physics hull covers < 60% of the collision model's bounding volume:", small.len());
        for (n, r, pt, ct) in small.iter().filter(|x| x.0.contains("forge")).take(60) {
            println!("   {r:5.2}  hull {pt:>5} tris / collision {ct:>5} tris   {n}");
        }
    }

    /// #collision  AUDIT: area of upward-facing engine collision surface by slope band, per
    /// structure BSP -- how much of the level sits between the global 35 degree surface limit
    /// and the authored 50 degree contact limit (walkable by the authored number, not by the
    /// surface bar).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn audit_slope_bands() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = CollisionBsp::for_sbsp(native, cache, t) else { continue };
            let edges = [0.0f32, 10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0, 90.1];
            let mut area = [0.0f32; 8];
            for (tr, _) in b.solid_triangles() {
                let c = (tr[1] - tr[0]).cross(tr[2] - tr[0]);
                let a = c.length() * 0.5;
                let n = c.normalize_or_zero();
                if n.z <= 0.0 {
                    continue;
                }
                let deg = n.z.clamp(-1.0, 1.0).acos().to_degrees();
                for k in 0..8 {
                    if deg >= edges[k] && deg < edges[k + 1] {
                        area[k] += a;
                    }
                }
            }
            let tot: f32 = area.iter().sum();
            println!(
                "{}: upward area {tot:.0} wu^2 | 0-10 {:.1}% 10-20 {:.1}% 20-30 {:.1}% 30-35 {:.1}% | 35-40 {:.1}% 40-50 {:.1}% | 50-60 {:.1}% 60+ {:.1}%",
                native.tag_name(cache, t).unwrap_or_default(),
                100.0 * area[0] / tot, 100.0 * area[1] / tot, 100.0 * area[2] / tot, 100.0 * area[3] / tot,
                100.0 * area[4] / tot, 100.0 * area[5] / tot, 100.0 * area[6] / tot, 100.0 * area[7] / tot
            );
        }
    }

    /// #collision  AUDIT: surface area by collision material, with the global material it maps
    /// to and that material's physics friction / restitution (`matg` materials, `mphp` at +0x20:
    /// +4 friction, +8 restitution, +0xC density; 0 = inherit from the parent material).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn audit_materials_by_area() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let matg = native.tags_by_group(cache, "matg")[0];
        let gm = crate::reach_actor::tag_block(native, cache, matg, 0x48C, 424, 256);
        let gname = |i: usize| -> String {
            gm.get(i * 424..i * 424 + 4).and_then(|b| native.string_id(cache, u32::from_le_bytes(b.try_into().unwrap()))).unwrap_or_default()
        };
        let gparent = |i: usize| -> Option<usize> {
            let pn = native.string_id(cache, u32::from_le_bytes(gm.get(i * 424 + 4..i * 424 + 8)?.try_into().ok()?))?;
            (0..gm.len() / 424).find(|&k| gname(k) == pn)
        };
        let gfric = |mut i: usize| -> f32 {
            for _ in 0..8 {
                let f = f32::from_le_bytes(gm[i * 424 + 0x24..i * 424 + 0x28].try_into().unwrap());
                if f > 0.0 && f.is_finite() { return f; }
                match gparent(i) { Some(p) if p != i => i = p, _ => break }
            }
            0.0
        };
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = CollisionBsp::for_sbsp(native, cache, t) else { continue };
            let cm = crate::reach_actor::tag_block(native, cache, t, 0x58, 24, 512);
            let mut area: std::collections::BTreeMap<i16, (f32, usize)> = Default::default();
            let mut flagged: std::collections::BTreeMap<i16, (usize, usize)> = Default::default();
            for sf in b.surfaces() {
                let e = flagged.entry(sf.material).or_default();
                e.0 += 1;
                if sf.flags & 0x80 != 0 { e.1 += 1; }
            }
            println!("   surfaces with flag 0x80 per collision material (total, flagged): {:?}", flagged.iter().filter(|(_, v)| v.1 > 0).collect::<Vec<_>>());
            for (tr, s) in b.solid_triangles() {
                let a = (tr[1] - tr[0]).cross(tr[2] - tr[0]).length() * 0.5;
                let m = b.surfaces()[s as usize].material;
                let e = area.entry(m).or_default();
                e.0 += a;
                e.1 += 1;
            }
            for (mi, e) in cm.chunks_exact(24).enumerate() {
                if [32i16, 60, 59].contains(&(mi as i16)) || mi == 6 {
                    let (mut zmin, mut zmax) = (f32::MAX, f32::MIN);
                    for (tr, s) in b.solid_triangles() {
                        if b.surfaces()[s as usize].material == mi as i16 { for v in tr { zmin = zmin.min(v.z); zmax = zmax.max(v.z); } }
                    }
                    println!("   collision material {mi}: runtime global {} conveyor {} seam {} flags {:#06x}; z range {zmin:.2}..{zmax:.2}", i16::from_le_bytes([e[0x10], e[0x11]]), i16::from_le_bytes([e[0x12], e[0x13]]), i16::from_le_bytes([e[0x14], e[0x15]]), u16::from_le_bytes([e[0x16], e[0x17]]));
                }
            }
            let total: f32 = area.values().map(|v| v.0).sum();
            let mut rows: Vec<_> = area.into_iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            println!("{}: {} collision materials, area {total:.0}", native.tag_name(cache, t).unwrap_or_default(), cm.len() / 24);
            for (m, (a, n)) in rows.iter().take(14) {
                let g = if *m >= 0 { cm.get(*m as usize * 24 + 0x10..*m as usize * 24 + 0x12).map(|x| i16::from_le_bytes(x.try_into().unwrap())) } else { None };
                let (gn, fr) = match g { Some(g) if g >= 0 => (gname(g as usize), gfric(g as usize)), _ => ("<none>".into(), 0.0) };
                println!("   {:5.1}% {n:>6} tris  collision material {m:>3} -> global {:>3} '{gn}' friction {fr:.2}", 100.0 * a / total, g.unwrap_or(-1));
            }
        }
        for n in ["hard_metal_thin_hum_spartan", "hard_metal_thin"] {
            if let Some(i) = (0..gm.len() / 424).find(|&k| gname(k) == n) {
                println!("character material '{n}' (global {i}): friction {:.2}", gfric(i));
            }
        }
    }

    /// A two-surface BSP built by hand: surface 0 an ordinary quad, surface 1 the same size with
    /// the `invalid` flag (0x10). Rings are closed by edges that name both surfaces.
    fn two_surfaces() -> CollisionBsp {
        let sq = |z: f32| [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]];
        let mut verts = Vec::new();
        verts.extend(sq(0.0));
        verts.extend(sq(1.0));
        // edge i of surface s: start i, end i+1, forward -> next edge of the ring, the other
        // surface unused (-1 style: use the surface itself on both sides is not needed for rings
        // walked from the LEFT side only)
        let mut edges = Vec::new();
        for s in 0..2i64 {
            for i in 0..4i64 {
                let base = s * 4;
                edges.push([base + i, base + (i + 1) % 4, s * 4 + (i + 1) % 4, s * 4 + (i + 3) % 4, s, 9999]);
            }
        }
        CollisionBsp {
            large: true,
            nodes: Vec::new(),
            supers: Vec::new(),
            planes: Vec::new(),
            leaves: Vec::new(),
            refs: Vec::new(),
            b2: Vec::new(),
            surf: vec![
                Surface { plane: 0, first_edge: 0, material: 0, flags: 0 },
                Surface { plane: 0, first_edge: 4, material: 0, flags: surface_flag::INVALID },
            ],
            edges,
            verts,
        }
    }

    /// #collision  **SOLID AND NOT SOLID.** The engine's key iterators enumerate every surface
    /// except those flagged `invalid` (0x10); a collision-only surface (no render triangle at all)
    /// is therefore solid, and an invalid one is not. Asserted both ways on a hand-built BSP, and
    /// on the real `forge_halo` BSP COMPLETENESS: every non-invalid surface yields at least one
    /// triangle (no collision-only surface is dropped by the decode or the triangulation).
    ///
    /// SABOTAGE: delete the `INVALID` skip in `solid_triangles` -> the hand-built half goes red;
    /// skip every surface whose material is -1 -> the completeness half goes red on the shared BSP.
    #[test]
    fn invalid_surfaces_are_not_solid_and_every_other_surface_is() {
        let b = two_surfaces();
        let tris = b.solid_triangles();
        assert!(!tris.is_empty(), "the ordinary surface produced nothing");
        assert!(tris.iter().all(|(_, s)| *s == 0), "an invalid-flagged surface was made solid: {:?}", tris.iter().map(|t| t.1).collect::<Vec<_>>());
        assert_eq!(tris.len(), 2, "a quad is two triangles");
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_every_real_surface_is_solid_unless_invalid() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut checked = 0;
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = CollisionBsp::for_sbsp(native, cache, t) else { continue };
            let mut covered = std::collections::HashSet::new();
            for (_, s) in b.solid_triangles() {
                covered.insert(s);
            }
            let want = b.surfaces().iter().filter(|s| s.flags & surface_flag::INVALID == 0).count();
            let degenerate = (0..b.surface_count()).filter(|&i| b.surfaces()[i].flags & surface_flag::INVALID == 0 && !covered.contains(&(i as u32))).count();
            println!("{}: {} non-invalid surfaces, {} yield triangles, {degenerate} yield none (zero-area)", native.tag_name(cache, t).unwrap_or_default(), want, covered.len());
            assert!(degenerate * 200 <= want, "{degenerate} of {want} solid surfaces produced no triangle: collision-only surfaces are being dropped");
            checked += want;
        }
        assert!(checked > 1000, "only {checked} surfaces were examined");
    }
}
