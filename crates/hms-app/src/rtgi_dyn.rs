//! Placed Forge pieces as occluders of the real-time probe GI: the part of the lighting that has to
//! follow the user's edits.
//!
//! The probe tracer's BVH is the map's BSP, built once. Everything the user places, moves, rotates or
//! deletes lives in a SECOND small BVH (`RtGi::set_dynamic`, a reserved region behind the static
//! buffers), rebuilt from the objects' physics hulls when they change. Only the probes an edit can
//! reach are re-estimated (`RtGi::invalidate_region`): the old and the new bounds of every object that
//! moved, grown by two probe spacings. Probes farther away are left bit-identical.
//!
//! An object occludes light iff it has a physics shape (`ObjectScene::physics_world_tris`: the phmo
//! hull, else the collision hull) — the same rule that makes it solid for the player. Spawn markers,
//! zones and effects have none, so they cast no light shadow.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_render::rtgi::{RtGiDynamic, DYN_NODE_CAP, DYN_TRI_CAP};

use crate::lightbake::GpuBvh;

/// Flat albedo of a placed piece as a bounce surface. The map's own fallback for a surface whose albedo
/// cannot be resolved (`bake_material_table`) is the same mid grey; per-part albedo of Forge models is a
/// refinement the bounce does not need (a piece's own radiance is small next to the sky it occludes).
pub const PIECE_ALBEDO: f32 = 0.5;

/// A trailing debounce: a drag changes the object list every frame, and a BVH build over every hull per
/// frame would hitch. The set is rebuilt once the edit has been quiet this long (or at most this often).
const QUIET: Duration = Duration::from_millis(120);
const AT_MOST_EVERY: Duration = Duration::from_millis(600);

struct Entry {
    sig: u64,
    tris: Vec<[Vec3; 3]>,
    lo: Vec3,
    hi: Vec3,
}

#[derive(Default)]
pub struct DynCache {
    objs: HashMap<u32, Entry>,
    /// signature of the object set the GPU currently holds
    pub sig: u64,
    pending: u64,
    changed_at: Option<Instant>,
    synced_at: Option<Instant>,
}

/// What a refresh found.
#[derive(Debug, Default)]
pub struct Delta {
    /// world boxes the lighting of which may have changed: old and new bounds of every object that
    /// moved, appeared or vanished
    pub regions: Vec<([f32; 3], [f32; 3])>,
    pub objects: usize,
    pub tris: usize,
}

fn fnv(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x0100_0000_01b3);
}

/// Pose + bounds fingerprint of one object (scale and variant changes move the bounds even when the
/// placement did not change).
pub fn object_sig(o: &ObjectInfo, aabb: Option<([f32; 3], [f32; 3])>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut h, o.mode_tag as u64);
    fnv(&mut h, o.primary_tag as u64);
    fnv(&mut h, o.variant_name_sid as u64);
    for c in o.pos.iter().chain(o.fwd.iter()).chain(o.up.iter()) {
        fnv(&mut h, c.to_bits() as u64);
    }
    if let Some((lo, hi)) = aabb {
        for c in lo.iter().chain(hi.iter()) {
            fnv(&mut h, c.to_bits() as u64);
        }
    }
    h
}

/// Fingerprint of the whole set, order independent.
pub fn set_sig(objects: &[ObjectInfo], aabb_of: &dyn Fn(u32) -> Option<([f32; 3], [f32; 3])>) -> u64 {
    let mut acc = objects.len() as u64;
    for o in objects {
        acc = acc.wrapping_add(object_sig(o, aabb_of(o.datum)).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
    acc
}

fn bounds(tris: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for t in tris {
        for p in t {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    (lo, hi)
}

impl DynCache {
    /// Note the current set signature; true when the GPU copy should be refreshed now (the edit has been
    /// quiet for `QUIET`, or it has been `AT_MOST_EVERY` since the last refresh).
    pub fn due(&mut self, sig: u64, now: Instant) -> bool {
        if sig == self.sig {
            self.pending = sig;
            return false;
        }
        if sig != self.pending {
            self.pending = sig;
            self.changed_at = Some(now);
        }
        let quiet = self.changed_at.map_or(true, |t| now.duration_since(t) >= QUIET);
        let stale = self.synced_at.map_or(true, |t| now.duration_since(t) >= AT_MOST_EVERY);
        quiet || stale
    }

    /// Re-read every object whose fingerprint changed (`tris_of` = its world hull triangles), drop the
    /// vanished ones, and report the regions the changes touch. `sig_of` fingerprints one object.
    pub fn refresh(
        &mut self,
        objects: &[ObjectInfo],
        sig_of: &dyn Fn(&ObjectInfo) -> u64,
        tris_of: &mut dyn FnMut(&ObjectInfo) -> Vec<[Vec3; 3]>,
        set_signature: u64,
        now: Instant,
    ) -> Delta {
        let mut regions: Vec<([f32; 3], [f32; 3])> = Vec::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(objects.len());
        let push = |lo: Vec3, hi: Vec3, regions: &mut Vec<([f32; 3], [f32; 3])>| {
            if lo.is_finite() && hi.is_finite() && lo.x <= hi.x { regions.push((lo.into(), hi.into())); }
        };
        for o in objects {
            seen.insert(o.datum);
            let s = sig_of(o);
            if self.objs.get(&o.datum).map_or(false, |e| e.sig == s) { continue; }
            let tris = tris_of(o);
            let (lo, hi) = if tris.is_empty() { (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)) } else { bounds(&tris) };
            if let Some(old) = self.objs.get(&o.datum) {
                if !old.tris.is_empty() { push(old.lo, old.hi, &mut regions); }
            }
            if !tris.is_empty() { push(lo, hi, &mut regions); }
            self.objs.insert(o.datum, Entry { sig: s, tris, lo, hi });
        }
        let gone: Vec<u32> = self.objs.keys().copied().filter(|d| !seen.contains(d)).collect();
        for d in gone {
            if let Some(e) = self.objs.remove(&d) {
                if !e.tris.is_empty() { push(e.lo, e.hi, &mut regions); }
            }
        }
        self.sig = set_signature;
        self.pending = set_signature;
        self.synced_at = Some(now);
        Delta { regions, objects: self.objs.values().filter(|e| !e.tris.is_empty()).count(), tris: self.objs.values().map(|e| e.tris.len()).sum() }
    }

    /// All current occluder triangles, trimmed to the tracer's reserved capacity (the largest pieces
    /// are kept: they shadow the most). Returns (triangles, objects dropped).
    pub fn triangles(&self) -> (Vec<[Vec3; 3]>, usize) {
        let total: usize = self.objs.values().map(|e| e.tris.len()).sum();
        let cap = DYN_TRI_CAP as usize;
        if total <= cap {
            return (self.objs.values().flat_map(|e| e.tris.iter().copied()).collect(), 0);
        }
        let mut by_size: Vec<&Entry> = self.objs.values().filter(|e| !e.tris.is_empty()).collect();
        by_size.sort_by(|a, b| {
            let v = |e: &Entry| { let d = e.hi - e.lo; d.x * d.y * d.z };
            v(b).partial_cmp(&v(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (mut out, mut used, mut dropped) = (Vec::new(), 0usize, 0usize);
        for e in by_size {
            if used + e.tris.len() > cap { dropped += 1; continue; }
            used += e.tris.len();
            out.extend(e.tris.iter().copied());
        }
        (out, dropped)
    }
}

/// The tracer-side set for these triangles: one BVH, one flat material. None when empty or when the
/// BVH would not fit the reserved node region.
pub fn build_dynamic(tris: &[[Vec3; 3]]) -> Option<RtGiDynamic> {
    if tris.is_empty() { return None; }
    let bvh = GpuBvh::build(tris);
    if bvh.nodes.len() > DYN_NODE_CAP as usize || bvh.tris.len() > DYN_TRI_CAP as usize { return None; }
    let a = PIECE_ALBEDO;
    Some(RtGiDynamic {
        nodes: bytemuck::cast_slice(&bvh.nodes).to_vec(),
        tris: bytemuck::cast_slice(&bvh.tris).to_vec(),
        tri_mat: vec![0; bvh.tris.len()],
        mats: vec![[a, a, a, 0.0], [0.0; 4], [0.0; 4]],
    })
}

impl crate::App {
    /// Per-frame hook: keep the probe tracer's placed-piece occluders (and the probes they reach) in step
    /// with the editor. Free unless real-time lighting is on and the object set changed.
    pub(crate) fn rtgi_dyn_tick(&mut self) {
        if !self.show_rtgi || !self.renderer.rtgi_has_scene() { return; }
        let now = Instant::now();
        // the cache leaves `self` while the object scene (which borrows all of `self`) is read
        let mut cache = std::mem::take(&mut self.rtgi_dyn);
        let out = match self.objscene() {
            None => None,
            Some(os) => {
                let aabb = |d: u32| os.aabb_of(d);
                let sig = set_sig(&self.last_objects, &aabb);
                if !cache.due(sig, now) {
                    None
                } else {
                    let delta = cache.refresh(&self.last_objects, &|o| object_sig(o, aabb(o.datum)), &mut |o| os.physics_world_tris(o), sig, now);
                    let (tris, dropped) = cache.triangles();
                    Some((delta, tris, dropped))
                }
            }
        };
        self.rtgi_dyn = cache;
        let Some((delta, tris, dropped)) = out else { return };
        let dynamic = build_dynamic(&tris);
        if !self.renderer.rtgi_set_dynamic(&self.render_state.queue, dynamic.as_ref()) {
            log::warn!("real-time lighting: {} occluder triangles do not fit the reserved region; keeping the previous set", tris.len());
            return;
        }
        let mut probes = 0usize;
        for (lo, hi) in &delta.regions {
            probes += self.renderer.rtgi_invalidate_region(*lo, *hi);
        }
        if std::env::var("HMS_RTGI_DIAG").is_ok() {
            eprintln!("RTGI dynamic: {} pieces / {} tris ({} dropped), {} edited regions, {} probes queued", delta.objects, tris.len(), dropped, delta.regions.len(), probes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(datum: u32, x: f32) -> ObjectInfo {
        ObjectInfo { datum, type_sig: 0, sig0: 0, sig1: 0, pos: [x, 0.0, 0.0], health: 1.0, shield: 1.0, mode_tag: 7, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: 7, variant_name_sid: 0 }
    }

    /// A unit cube hull centred on the object's position (what `physics_world_tris` would return).
    fn hull(o: &ObjectInfo) -> Vec<[Vec3; 3]> {
        let c = Vec3::from(o.pos);
        let p = |x: f32, y: f32, z: f32| c + Vec3::new(x, y, z) * 0.5;
        let q = |a: Vec3, b: Vec3, cc: Vec3, d: Vec3| [[a, b, cc], [a, cc, d]];
        let f = [
            q(p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.)), q(p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.)),
            q(p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.)), q(p(-1., 1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(-1., 1., 1.)),
            q(p(-1., -1., -1.), p(-1., 1., -1.), p(-1., 1., 1.), p(-1., -1., 1.)), q(p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.)),
        ];
        f.iter().flatten().copied().collect()
    }

    fn refresh(c: &mut DynCache, objs: &[ObjectInfo]) -> Delta {
        let sig = set_sig(objs, &|_| None);
        c.refresh(objs, &|o| object_sig(o, None), &mut |o| hull(o), sig, Instant::now())
    }

    /// Moving ONE piece reports exactly its old and its new bounds; untouched pieces report nothing; a
    /// vanished piece reports its last bounds. (Saboteur: report every piece, or only the new bounds.)
    #[test]
    fn an_edit_reports_the_old_and_new_bounds_of_what_moved_only() {
        let mut c = DynCache::default();
        let mut objs: Vec<ObjectInfo> = (0..5).map(|i| obj(i, i as f32 * 10.0)).collect();
        let first = refresh(&mut c, &objs);
        assert_eq!((first.objects, first.tris, first.regions.len()), (5, 60, 5), "first sync: every piece is new");
        let again = refresh(&mut c, &objs);
        assert!(again.regions.is_empty(), "nothing changed, nothing reported: {:?}", again.regions);

        objs[2].pos[0] = 25.0;
        let d = refresh(&mut c, &objs);
        assert_eq!(d.regions.len(), 2, "one moved piece = its old box + its new box");
        let near = |r: &([f32; 3], [f32; 3]), x: f32| (r.0[0] - (x - 0.5)).abs() < 1e-4 && (r.1[0] - (x + 0.5)).abs() < 1e-4;
        assert!(d.regions.iter().any(|r| near(r, 20.0)) && d.regions.iter().any(|r| near(r, 25.0)), "{:?}", d.regions);

        objs.remove(0);
        let gone = refresh(&mut c, &objs);
        assert_eq!(gone.regions.len(), 1);
        assert!(near(&gone.regions[0], 0.0));
        assert_eq!(gone.objects, 4);
    }

    /// A piece with no hull (a spawn marker, a zone) casts no light shadow and is never reported.
    #[test]
    fn objects_without_a_physics_shape_occlude_nothing() {
        let mut c = DynCache::default();
        let objs = vec![obj(1, 0.0), obj(2, 5.0)];
        let sig = set_sig(&objs, &|_| None);
        let d = c.refresh(&objs, &|o| object_sig(o, None), &mut |o| if o.datum == 1 { hull(o) } else { Vec::new() }, sig, Instant::now());
        assert_eq!((d.objects, d.tris, d.regions.len()), (1, 12, 1));
    }

    /// The debounce: a change waits for the edit to go quiet, but never longer than `AT_MOST_EVERY`.
    #[test]
    fn the_rebuild_waits_for_a_quiet_edit_but_not_forever() {
        let mut c = DynCache::default();
        let t0 = Instant::now();
        assert!(c.due(1, t0), "the very first population is not delayed");
        c.synced_at = Some(t0);
        c.sig = 1;
        assert!(!c.due(1, t0), "no change");
        assert!(!c.due(10, t0), "a fresh edit waits");
        assert!(!c.due(11, t0 + Duration::from_millis(60)), "still dragging");
        assert!(c.due(11, t0 + Duration::from_millis(60) + QUIET), "quiet long enough");
        c.synced_at = Some(t0);
        c.sig = 11;
        // a drag that never goes quiet is still flushed
        assert!(!c.due(12, t0 + Duration::from_millis(100)));
        assert!(!c.due(13, t0 + Duration::from_millis(200)));
        assert!(c.due(14, t0 + AT_MOST_EVERY + Duration::from_millis(1)), "the staleness cap");
    }

    /// The tracer set carries every triangle once, its BVH root spans them, and an over-capacity set is
    /// trimmed largest-first.
    #[test]
    fn the_tracer_set_is_the_union_of_the_hulls_and_respects_the_capacity() {
        let mut c = DynCache::default();
        let objs: Vec<ObjectInfo> = (0..40).map(|i| obj(i, i as f32 * 3.0)).collect();
        refresh(&mut c, &objs);
        let (tris, dropped) = c.triangles();
        assert_eq!((tris.len(), dropped), (480, 0));
        let d = build_dynamic(&tris).expect("a dynamic set");
        assert_eq!(d.tris.len() / 48, 480);
        assert_eq!(d.tri_mat.len(), 480);
        assert_eq!(d.mats.len(), 3);
        let words: &[u32] = bytemuck::cast_slice(&d.nodes);
        let (rlo, rhi) = ([f32::from_bits(words[0]), f32::from_bits(words[1]), f32::from_bits(words[2])], [f32::from_bits(words[4]), f32::from_bits(words[5]), f32::from_bits(words[6])]);
        assert!((rlo[0] + 0.5).abs() < 1e-4 && (rhi[0] - (39.0 * 3.0 + 0.5)).abs() < 1e-3, "root box {rlo:?}..{rhi:?}");
        assert!(build_dynamic(&[]).is_none());

        // over capacity: DYN_TRI_CAP + a few; the big piece must survive, small ones go first
        let mut big = DynCache::default();
        let many = DYN_TRI_CAP as usize / 12 + 5;
        let mut os: Vec<ObjectInfo> = (0..many as u32).map(|i| obj(i, i as f32)).collect();
        os.push(obj(100_000, -50.0));
        let sig = set_sig(&os, &|_| None);
        big.refresh(&os, &|o| object_sig(o, None), &mut |o| {
            let mut h = hull(o);
            if o.datum == 100_000 { for t in &mut h { for p in t { *p = (*p - Vec3::new(-50.0, 0.0, 0.0)) * 40.0 + Vec3::new(-50.0, 0.0, 0.0); } } }
            h
        }, sig, Instant::now());
        let (kept, dropped) = big.triangles();
        assert!(kept.len() <= DYN_TRI_CAP as usize && dropped >= 5, "kept {} dropped {dropped}", kept.len());
        // every ordinary piece sits at x >= -0.5; the 40x piece spans x in [-70, -30]
        assert!(kept.iter().any(|t| t[0].x < -29.0), "the big piece was dropped before the small ones");
    }
}
