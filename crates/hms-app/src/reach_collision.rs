//! #collision  Reach's COLLISION GROUPS: which physics collides with which.
//!
//! The question this answers: "I walk over a spawn point and it freaks out". Test mode's collision
//! gather and Theater's ragdoll gather both feed **every** nearby object's physics geometry into
//! the triangle set, so a spawn point, a trait zone or a kill volume is as solid as a Forge block.
//!
//! **The fix must be the engine's own rule and not a skip-list of tag names**, because a list is
//! wrong the first time somebody places a thing nobody thought of. The engine's rule is in two
//! places, and both are authored:
//!
//! * **`matg` +0x5C4 `havok collision filter`** -> an `hcfd` (`havok_collision_filter`) whose
//!   `groups` block (4-byte elements, **max 27**) is one `long flags` per group named `filter`,
//!   under the tag's own explanation *"specify what each group collides with"*. So **group `i`
//!   collides with group `j` exactly when `groups[i].filter` has bit `j` set.** The rest of the tag
//!   (+0x10 `group filter`) is the runtime Havok `hkpGroupFilter` with its 27-entry
//!   `m_collisionLookupTable00..26`, which is what the engine builds FROM those masks — so the
//!   authored block and the runtime object are the same fact twice.
//! * **Each `phmo` collision shape carries its own group.** Every shape element begins with a
//!   32-byte `base` struct holding `material` (+0x04), a **`phantom` block index (+0x1E)** and
//!   **`runtime collision group` (+0x1F)**.
//!
//! The `phantom` index is the other half of the answer and arguably the more important one: in
//! Havok a **phantom is a trigger volume, not a solid body**. A shape that names a phantom detects
//! overlap and never pushes anything.

use hms_native::{CacheHandle, NativeDll};

/// `matg` +0x5C4 `havok collision filter` -> `hcfd`.
pub const MATG_HAVOK_COLLISION_FILTER: usize = 0x5C4;
/// `hcfd` +0x00 `groups` block: one `long flags` `filter` per group.
pub const HCFD_GROUPS: usize = 0x00;
/// One `havok_collision_filter_group_block` element: a single `long flags`.
pub const HCFD_GROUP_SIZE: usize = 4;
/// `k` in the tag's own `max 27`, and the width of Havok's `m_collisionLookupTable00..26`.
pub const HCFD_GROUP_MAX: i32 = 27;

/// A `phmo` collision shape's `base` struct, which every shape element starts with.
pub const SHAPE_BASE_SIZE: usize = 32;
/// ...+0x04 `material` (a char block index into the `phmo`'s own materials).
pub const SHAPE_MATERIAL: usize = 0x04;
/// ...+0x1E `phantom` (a char block index). **A Havok phantom is a trigger volume, not solid.**
pub const SHAPE_PHANTOM: usize = 0x1E;
/// ...+0x1F `runtime collision group`.
pub const SHAPE_COLLISION_GROUP: usize = 0x1F;

/// The 27 authored group masks: `groups[i]` is the set of groups `i` collides with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollisionFilter {
    pub hcfd: u32,
    pub groups: Vec<u32>,
}

impl CollisionFilter {
    /// Read the filter out of `matg`. `None` when the globals name none.
    pub fn read(native: &NativeDll, cache: CacheHandle, matg: u32) -> Option<CollisionFilter> {
        let m = native.tag_meta(cache, matg, MATG_HAVOK_COLLISION_FILTER as u32, 16)?;
        let d = u32::from_le_bytes([m[0x0C], m[0x0D], m[0x0E], m[0x0F]]);
        if d == u32::MAX || d == 0 {
            return None;
        }
        let hcfd = d & 0xFFFF;
        let b = crate::reach_actor::tag_block(native, cache, hcfd, HCFD_GROUPS, HCFD_GROUP_SIZE, HCFD_GROUP_MAX);
        let groups = (0..b.len() / HCFD_GROUP_SIZE)
            .map(|i| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]))
            .collect();
        Some(CollisionFilter { hcfd, groups })
    }

    /// Does group `a` collide with group `b`? The engine's own rule: bit `b` of `groups[a]`.
    ///
    /// Reflexive by construction in a well-formed table, which [`Self::is_symmetric`] checks.
    pub fn collides(&self, a: usize, b: usize) -> Option<bool> {
        let mask = *self.groups.get(a)?;
        if b >= 32 {
            return None;
        }
        Some(mask & (1u32 << b) != 0)
    }

    /// Is the table symmetric — does `a` collide with `b` exactly when `b` collides with `a`?
    ///
    /// A collision filter that is NOT symmetric would mean the engine's answer depends on which
    /// body is asked first, which no Havok group filter does. Worth asserting rather than
    /// assuming, because it is a cheap check on the whole decode at once: a wrong element size or
    /// offset would almost certainly break it.
    pub fn is_symmetric(&self) -> bool {
        let n = self.groups.len().min(32);
        (0..n).all(|a| (0..n).all(|b| self.collides(a, b) == self.collides(b, a)))
    }
}

/// One `phmo` collision shape's filtering fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShapeFilter {
    /// Which shape block it came from, for the diagnostics.
    pub kind: &'static str,
    pub index: usize,
    /// `runtime collision group`.
    pub group: u8,
    /// `phantom` block index, or `None` when the shape names none. **`Some` means this shape is a
    /// trigger volume and never pushes anything.**
    pub phantom: Option<u8>,
    /// `material` block index.
    pub material: u8,
}

/// Every collision shape of an object's `phmo`, with its group and phantom index.
///
/// Reads the shape blocks `phmo_shapes` already knows the offsets and strides of, so there is no
/// second layout here to drift from the ragdoll's.
pub fn shape_filters(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Vec<ShapeFilter> {
    let Some(b) = crate::ragdoll::phmo_blocks_for(native, cache, obj_tag) else { return Vec::new() };
    use crate::phmo_shapes as ps;
    let blocks: [(&'static str, &Vec<u8>, usize); 6] = [
        ("sphere", &b.spheres, ps::SPHERE_ELEM),
        ("multi_sphere", &b.multi_spheres, ps::MULTI_SPHERE_ELEM),
        ("pill", &b.pills, ps::PILL_ELEM),
        ("box", &b.boxes, ps::BOX_ELEM),
        ("triangle", &b.triangles, ps::TRIANGLE_ELEM),
        ("polyhedron", &b.polyhedra, ps::REACH.poly_elem),
    ];
    let mut out = Vec::new();
    for (kind, bytes, elem) in blocks {
        if elem < SHAPE_BASE_SIZE {
            continue;
        }
        for i in 0..bytes.len() / elem {
            let e = &bytes[i * elem..(i + 1) * elem];
            let ph = e[SHAPE_PHANTOM];
            out.push(ShapeFilter {
                kind,
                index: i,
                group: e[SHAPE_COLLISION_GROUP],
                phantom: (ph != 0xFF).then_some(ph),
                material: e[SHAPE_MATERIAL],
            });
        }
    }
    out
}

/// `obje` +0x140 `multiplayer object` block (188-byte elements, max 1).
pub const OBJE_MULTIPLAYER_OBJECT: usize = 0x140;
pub const MULTIPLAYER_OBJECT_SIZE: usize = 188;
/// ...+0x01 `type`, a char enum of 28 values.
pub const MP_TYPE: usize = 0x01;
/// ...+0x14 `boundary shape`: `0 unused`, `1 sphere`, `2 cylinder`, `3 box`. Under the block's own
/// explanation **"GOAL & RESPAWN ZONE OBJECT BOUNDARY"**, with the dimensions at +0x04..+0x13.
pub const MP_BOUNDARY_SHAPE: usize = 0x14;

/// The 28 authored `multiplayer object` types, in the enum's own order.
pub const MP_TYPE_NAMES: [&str; 28] = [
    "ordinary", "weapon", "grenade", "projectile", "powerup", "equipment", "ammo pack",
    "light land vehicle", "heavy land vehicle", "flying vehicle", "turret", "device",
    "teleporter 2way", "teleporter sender", "teleporter receiver", "player spawn location",
    "player respawn zone", "secondary objective", "primary objective", "named location area",
    "danger zone", "fireteam 1 respawn zone", "fireteam 2 respawn zone", "fireteam 3 respawn zone",
    "fireteam 4 respawn zone", "safe volume", "kill volume", "cinematic camera position",
];

/// What an object's `multiplayer object` block says about it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultiplayerObject {
    /// `type`, 0..27.
    pub kind: u8,
    /// `boundary shape`: 0 unused, 1 sphere, 2 cylinder, 3 box.
    pub boundary_shape: u8,
}

impl MultiplayerObject {
    pub fn type_name(&self) -> &'static str {
        MP_TYPE_NAMES.get(self.kind as usize).copied().unwrap_or("?")
    }
    /// **Is this object's extent a BOUNDARY VOLUME rather than a solid body?**
    ///
    /// The engine's own statement, from the field's own explanation: a non-zero `boundary shape`
    /// means the block's `boundary width/radius`, `box length` and `+/-height` define this
    /// object's region, under the heading "GOAL & RESPAWN ZONE OBJECT BOUNDARY". A thing whose
    /// extent is a region to be tested against is not a thing to walk into.
    pub fn is_boundary_volume(&self) -> bool {
        self.boundary_shape != 0
    }
}

/// Read an object's `multiplayer object` block. `None` when it authors none (most scenery).
pub fn multiplayer_object(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<MultiplayerObject> {
    let b = crate::reach_actor::tag_block(native, cache, obj_tag, OBJE_MULTIPLAYER_OBJECT, MULTIPLAYER_OBJECT_SIZE, 1);
    if b.len() < MULTIPLAYER_OBJECT_SIZE {
        return None;
    }
    Some(MultiplayerObject { kind: b[MP_TYPE], boundary_shape: b[MP_BOUNDARY_SHAPE] })
}

/// The total mass of an object's `phmo` rigid bodies, in kg, as the tag authors it.
///
/// Summed over the rigid-body block (`phmo_shapes::REACH.rb_elem` 208-byte elements, `mass` at
/// `rb_mass`), which is the object's mass: a Warthog's two bodies and a Spartan's eleven each add
/// up to the whole. `None` when the object has no `phmo`.
pub fn rigid_body_mass(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<f32> {
    let b = crate::ragdoll::phmo_blocks_for(native, cache, obj_tag)?.rigid_bodies;
    let l = &crate::phmo_shapes::REACH;
    if b.len() < l.rb_elem {
        return None;
    }
    let mut total = 0.0f32;
    for i in 0..b.len() / l.rb_elem {
        let e = &b[i * l.rb_elem..(i + 1) * l.rb_elem];
        let m = f32::from_le_bytes([e[l.rb_mass], e[l.rb_mass + 1], e[l.rb_mass + 2], e[l.rb_mass + 3]]);
        if m.is_finite() && m > 0.0 {
            total += m;
        }
    }
    (total > 0.0).then_some(total)
}

// ---------------------------------------------------------------------------------------------
// THE PREDICATE
// ---------------------------------------------------------------------------------------------

/// The `multiplayer object` types whose extent is a REGION rather than a body.
///
/// **This mapping is an INFERENCE, and a weaker one than the rest of this module.** What is
/// measured is the field, its 28 authored values and which objects carry which. That these
/// particular values are non-solid is read off the engine's own type NAMES — `player spawn
/// location`, `kill volume`, `safe volume`, `danger zone`, `named location area`, `cinematic
/// camera position` — plus the block's own explanation heading, which calls the boundary fields
/// *"GOAL & RESPAWN ZONE OBJECT BOUNDARY"* and so treats goals and respawn zones as regions. It
/// is not the standard of `reach_physics::WORLD_GRAVITY` (two independent confirmations) or the
/// `jpt!` damage bounds (corroborated by values anyone who has played Reach would predict.)
///
/// **One independent corroboration, which is why it is trusted this far.** At *identical* authored
/// mass the type field separates objects the physics cannot: `kill_volume` and
/// `dlc_hangar_forcefield_small_solid` are both **0.1 kg**, and the type calls the first a
/// `kill volume` and the second `ordinary` — correctly. So the field carries information that
/// exists nowhere else in the tags.
pub const REGION_MP_TYPES: [u8; 11] = [
    15, // player spawn location
    16, // player respawn zone
    17, // secondary objective  -- a GOAL area; see the boundary explanation
    18, // primary objective
    19, // named location area
    20, // danger zone
    21, 22, 23, 24, // fireteam 1-4 respawn zones
    25, // safe volume
];

/// Also a region, kept out of the array above only because the array is sized.
pub const REGION_MP_TYPE_KILL_VOLUME: u8 = 26;
/// A camera position, not a body.
pub const REGION_MP_TYPE_CINEMATIC_CAMERA: u8 = 27;

/// Is this `multiplayer object` type a REGION (walk through) rather than a body (walk into)?
pub fn mp_type_is_region(kind: u8) -> bool {
    REGION_MP_TYPES.contains(&kind) || kind == REGION_MP_TYPE_KILL_VOLUME || kind == REGION_MP_TYPE_CINEMATIC_CAMERA
}

/// **Does this object's physics collide with a BIPED?**
///
/// The one question both collision gathers should ask, so test mode's player and Theater's
/// ragdolls cannot disagree about what is solid.
///
/// The rule, in order:
/// 1. **No physics geometry at all -> nothing to collide with.** Not a judgement, an absence.
/// 2. **An authored multiplayer REGION type -> not solid.** See [`REGION_MP_TYPES`].
/// 3. **Everything else is SOLID, including an object with no `multiplayer object` block at all.**
///
/// **Rule 3's direction was measured, and it is the opposite of this module's first draft.** The
/// asymmetry: a false *solid* is bumping into an invisible marker — annoying and local; a false
/// *pass-through* is walking through a rock or out of the world — catastrophic and global. So the
/// default must be solid unless the data says the no-block set is only editor objects. **It is
/// not.** Of `forge_halo`'s 496 objects that carry physics geometry, **204 author no multiplayer
/// block**, and they are overwhelmingly real geometry: `hull_shell`, `seat`, `wing_left`,
/// `rudder`, `fender_front_left`, `bumper`, `hubcap`, `cab`, `warthog_gauss_barrel` — the
/// detachable and destructible pieces of vehicles — plus props like `can`, `floodlight` and
/// `pylon`. Defaulting those to pass-through would have made every vehicle part and a pile of
/// scenery walk-through. Only `editor_gizmo` in that set is genuinely editor-only.
///
/// **What this does NOT do.** It does not make a light object pushable. A 0.1 kg marker and a
/// 4,084 kg crate are both handed to the solver as immovable terrain, which is wrong for
/// everything under the player's own 125 kg — see this module's note on mass. A filter cannot fix
/// that; pushable bodies can.
pub fn collides_with_biped(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> bool {
    if shape_filters(native, cache, obj_tag).is_empty() {
        return false;
    }
    match multiplayer_object(native, cache, obj_tag) {
        Some(mp) => !mp_type_is_region(mp.kind),
        // No multiplayer block: SOLID. See the note above -- 204 of 496 physics-bearing objects
        // are in this case and they are vehicle parts and props.
        None => true,
    }
}

#[cfg(test)]
mod gates {
    //! #collision  **THE PREDICATE GATE.** Both directions, with the counts separated.

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// A Forge block must be IN scope and a spawn point must be OUT, and the counts on each side
    /// must be non-zero.
    ///
    /// **The non-zero counts are the point.** `ObjectScene::static_tris_in_aabb` has a default
    /// EMPTY implementation, so a filter that rejected everything would look exactly like a map
    /// with no collision — and asserting only "the spawn point is excluded" is satisfied by a
    /// filter that excludes the whole world. Both sides are counted separately for the same
    /// reason: "nothing passed the filter" must never be able to read as "the filter works".
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_block_is_solid_and_a_spawn_point_is_not() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_collision gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut solid = 0usize;
        let mut passthrough = 0usize;
        let mut no_physics = 0usize;
        let mut spawn_family_solid: Vec<String> = Vec::new();
        let mut markers_excluded = 0usize;
        for group in ["scen", "bloc", "ctrl", "mach", "weap", "vehi", "eqip", "crea", "ssce", "gint", "term"] {
            for &t in &native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, t).unwrap_or_default();
                let has_geom = !shape_filters(native, cache, t).is_empty();
                let solid_now = collides_with_biped(native, cache, t);
                if !has_geom {
                    no_physics += 1;
                    assert!(!solid_now, "'{name}' has no physics geometry but is reported solid");
                    continue;
                }
                if solid_now { solid += 1 } else { passthrough += 1 }
                // Every spawn-family tag (by the INDEPENDENT name test the spawn overlay uses)
                // must come out non-solid. The name test is the oracle here, not the rule.
                if crate::map_spawns::is_map_spawn_name(&name) {
                    if solid_now {
                        spawn_family_solid.push(name.rsplit('\\').next().unwrap_or("").to_string());
                    } else {
                        markers_excluded += 1;
                    }
                }
            }
        }
        println!("solid {solid}, pass-through {passthrough}, no physics {no_physics}; {markers_excluded} spawn-family tags excluded");

        // THE POSITIVE SIDE, asserted first and loudly: the filter must keep the world.
        assert!(solid > 300, "only {solid} objects are solid -- the filter has eaten the map");
        // THE NEGATIVE SIDE: the markers really are out.
        assert!(passthrough > 5, "only {passthrough} objects are pass-through -- the filter is doing nothing");
        assert!(
            spawn_family_solid.is_empty(),
            "these spawn-family objects are still solid and would be invisible walls: {spawn_family_solid:?}"
        );
        assert!(markers_excluded >= 5, "only {markers_excluded} spawn-family tags were excluded");
        // ...and solid must DOMINATE. A filter that excluded more than it kept would be wrong in
        // the catastrophic direction (walking through the map), not the annoying one.
        assert!(solid > passthrough * 10, "pass-through ({passthrough}) is too large against solid ({solid})");
    }

    /// The named objects, so the verdicts can be eyeballed rather than trusted.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_verdict_on_the_objects_that_matter() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let want: [(&str, bool); 8] = [
            ("respawn_point", false),
            ("initial_spawn_point", false),
            ("respawn_zone", false),
            ("kill_volume", false),
            ("safe_volume", false),
            ("mp_cinematic_camera", false),
            // ...and the things that MUST stay solid, including the 0.1 kg forcefield that mass
            // could not tell apart from a kill volume.
            ("dlc_hangar_forcefield_small_solid", true),
            ("crate_multi", true),
        ];
        for (leaf, expect) in want {
            let mut found = false;
            for group in ["scen", "bloc", "ctrl", "mach"] {
                let Some(t) = native.tags_by_group(cache, group).into_iter().find(|&t| {
                    native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))
                }) else { continue };
                found = true;
                let got = collides_with_biped(native, cache, t);
                let mp = multiplayer_object(native, cache, t);
                println!(
                    "  {leaf:<38} solid = {got:<5} (expected {expect}) [{}]",
                    mp.map(|m| m.type_name()).unwrap_or("no mp block")
                );
                assert_eq!(got, expect, "'{leaf}' solid = {got}, expected {expect}");
                break;
            }
            assert!(found, "'{leaf}' is not in this map -- the gate cannot check it");
        }
    }
}

#[cfg(test)]
mod probe {
    //! #collision  What the shipped tags actually say, before any predicate is written.

    use super::*;

    fn matg_of(native: &NativeDll, cache: CacheHandle) -> Option<u32> {
        native.tags_by_group(cache, "matg").into_iter().next()
    }

    /// #collision  **The candidate discriminator, measured.** For every object that authors a
    /// `multiplayer object` block: its type, its boundary shape, and whether the collision gather
    /// would currently feed its physics in as solid geometry.
    ///
    /// The question is whether `boundary shape != unused` separates the markers a player walks
    /// through from the blocks a player walks into, with no tag-name matching anywhere.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_multiplayer_object_types() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        // kind -> (count, how many author a boundary, how many have physics geometry, examples)
        let mut by_kind: std::collections::BTreeMap<u8, (usize, usize, usize, Vec<String>)> = Default::default();
        let mut spawn_named_with_mp = 0usize;
        let mut spawn_named_total = 0usize;
        for group in ["scen", "bloc", "ctrl", "mach", "weap", "vehi", "eqip", "bipd", "crea", "efsc", "ssce", "gint", "term"] {
            for &t in &native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, t).unwrap_or_default();
                let is_spawn_by_name = crate::map_spawns::is_map_spawn_name(&name);
                if is_spawn_by_name {
                    spawn_named_total += 1;
                }
                let Some(mp) = multiplayer_object(native, cache, t) else { continue };
                if is_spawn_by_name {
                    spawn_named_with_mp += 1;
                }
                let has_phys = !shape_filters(native, cache, t).is_empty();
                let e = by_kind.entry(mp.kind).or_default();
                e.0 += 1;
                if mp.is_boundary_volume() {
                    e.1 += 1;
                }
                if has_phys {
                    e.2 += 1;
                }
                if e.3.len() < 3 {
                    e.3.push(name.rsplit('\\').next().unwrap_or("").to_string());
                }
            }
        }
        println!("{:<28} {:>5} {:>9} {:>9}  examples", "multiplayer type", "count", "boundary", "w/ physics");
        for (k, (n, bnd, phys, ex)) in by_kind.iter() {
            println!(
                "{:>2} {:<25} {n:>5} {bnd:>9} {phys:>9}  {}",
                k,
                MP_TYPE_NAMES.get(*k as usize).copied().unwrap_or("?"),
                ex.join(", ")
            );
        }
        println!("\n{spawn_named_with_mp} of {spawn_named_total} spawn-family tags (by the existing NAME test) author a multiplayer object block");
    }

    /// #collision  Does a spawn point's RIGID BODY differ from a crate's? A shape can exist for
    /// query purposes while the body is keyframed or massless, and that would be the engine's own
    /// "this does not push things around" with no classification needed.
    ///
    /// Prints the rigid-body element bytes of both and the offsets at which they differ, so the
    /// answer is a diff rather than a guess about which field is the motion type.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_rigid_body_difference() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let find = |leaf: &str| -> Option<u32> {
            for group in ["scen", "bloc", "ctrl"] {
                if let Some(t) = native.tags_by_group(cache, group).into_iter().find(|&t| {
                    native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))
                }) {
                    return Some(t);
                }
            }
            None
        };
        let rb = |t: u32| -> Vec<u8> {
            crate::ragdoll::phmo_blocks_for(native, cache, t).map(|b| b.rigid_bodies).unwrap_or_default()
        };
        let elem = crate::phmo_shapes::REACH.rb_elem;
        let mut rows: Vec<(String, Vec<u8>)> = Vec::new();
        for leaf in ["respawn_point", "initial_spawn_point", "kill_volume", "crate_multi", "space_crate"] {
            let Some(t) = find(leaf) else { continue };
            let b = rb(t);
            println!("{leaf:<22} rigid bodies: {} byte(s) = {} element(s) of {elem}", b.len(), b.len() / elem.max(1));
            if b.len() >= elem {
                let e = b[..elem].to_vec();
                let mass = f32::from_le_bytes([
                    e[crate::phmo_shapes::REACH.rb_mass],
                    e[crate::phmo_shapes::REACH.rb_mass + 1],
                    e[crate::phmo_shapes::REACH.rb_mass + 2],
                    e[crate::phmo_shapes::REACH.rb_mass + 3],
                ]);
                let shape_ty = i16::from_le_bytes([e[crate::phmo_shapes::REACH.rb_shape_type], e[crate::phmo_shapes::REACH.rb_shape_type + 1]]);
                println!("    mass {mass}  shape type {shape_ty}");
                rows.push((leaf.to_string(), e));
            }
        }
        // The diff: which offsets separate a marker from a crate?
        if let (Some(spawn), Some(crate_)) = (
            rows.iter().find(|(n, _)| n == "respawn_point"),
            rows.iter().find(|(n, _)| n == "crate_multi"),
        ) {
            let diffs: Vec<String> = (0..elem)
                .filter(|&i| spawn.1[i] != crate_.1[i])
                .map(|i| format!("+{i:#04x}: spawn {:#04x} vs crate {:#04x}", spawn.1[i], crate_.1[i]))
                .collect();
            println!("\nrigid-body bytes that differ between respawn_point and crate_multi ({}):", diffs.len());
            for d in diffs.iter().take(40) {
                println!("   {d}");
            }
        }
    }

    /// #collision  The MASS DISTRIBUTION of every placeable object, sorted — so a threshold can be
    /// put in a measured gap rather than chosen.
    ///
    /// The reported bug is that HMS hands every object's hull to the solver as STATIC terrain, so
    /// a 0.1 kg marker is infinitely heavy and a 125 kg Spartan is the one that gets thrown. The
    /// question this answers: is there a real gap between the markers and the lightest thing a
    /// player should actually collide with?
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_object_mass_distribution() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let player = native
            .tags_by_group(cache, "bipd")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).map_or(false, |n| n.ends_with("spartans")))
            .and_then(|t| crate::reach_physics::biped_physics(native, cache, t))
            .map(|(_, _, m)| m)
            .unwrap_or(0.0);
        println!("the player's own authored mass: {player} kg\n");
        let mut all: Vec<(f32, String, Option<u8>)> = Vec::new();
        for group in ["scen", "bloc", "ctrl", "mach", "weap", "vehi", "eqip", "crea", "ssce", "gint", "term"] {
            for &t in &native.tags_by_group(cache, group) {
                let Some(m) = rigid_body_mass(native, cache, t) else { continue };
                let name = native.tag_name(cache, t).unwrap_or_default();
                let mp = multiplayer_object(native, cache, t).map(|m| m.kind);
                all.push((m, name.rsplit('\\').next().unwrap_or("").to_string(), mp));
            }
        }
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        println!("{} objects with a phmo mass. The 40 LIGHTEST:", all.len());
        for (m, n, mp) in all.iter().take(40) {
            let ty = mp.map(|k| MP_TYPE_NAMES.get(k as usize).copied().unwrap_or("?")).unwrap_or("-");
            println!("  {m:>12.3} kg  {n:<44} [{ty}]");
        }
        // Where are the gaps? Print every consecutive ratio above 2x in the light end.
        println!("\nconsecutive ratios > 2x below the player's mass:");
        for w in all.windows(2) {
            if w[1].0 >= player || w[0].0 <= 0.0 {
                continue;
            }
            let r = w[1].0 / w[0].0;
            if r > 2.0 {
                println!("  {:.3} kg ({}) -> {:.3} kg ({})  = {r:.1}x", w[0].0, w[0].1, w[1].0, w[1].1);
            }
        }
        let below: Vec<&(f32, String, Option<u8>)> = all.iter().filter(|x| x.0 < player).collect();
        println!("\n{} of {} objects are lighter than the player", below.len(), all.len());
    }

    /// #collision  **What the DEFAULT must be.** How many objects that could reach the collision
    /// gather author no `multiplayer object` block at all, and what are they?
    ///
    /// The asymmetry that decides it: a false SOLID is bumping into an invisible marker, which is
    /// annoying and local. A false PASS-THROUGH is walking through a rock or out of the world,
    /// which is catastrophic and global. So "no block" must default to SOLID unless the data says
    /// the set is only editor-only objects.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_objects_without_a_multiplayer_block() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let mut with = 0usize;
        let mut without: Vec<String> = Vec::new();
        let mut without_with_physics = 0usize;
        for group in ["scen", "bloc", "ctrl", "mach", "crea", "ssce", "gint", "term"] {
            for &t in &native.tags_by_group(cache, group) {
                // Only objects that would contribute geometry at all.
                if shape_filters(native, cache, t).is_empty() {
                    continue;
                }
                let name = native.tag_name(cache, t).unwrap_or_default();
                if multiplayer_object(native, cache, t).is_some() {
                    with += 1;
                } else {
                    without_with_physics += 1;
                    if without.len() < 40 {
                        without.push(name.rsplit('\\').next().unwrap_or("").to_string());
                    }
                }
            }
        }
        println!("objects WITH physics geometry: {with} author a multiplayer block, {without_with_physics} do NOT");
        println!("\nthe first {} without one:", without.len());
        for n in &without {
            println!("   {n}");
        }
    }

    /// The authored group table, and the per-shape groups of the objects that matter — a spawn
    /// point (which must not be solid) and a Forge block (which must).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_collision_groups() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        match matg_of(native, cache).and_then(|m| CollisionFilter::read(native, cache, m)) {
            None => println!("matg names NO havok collision filter"),
            Some(f) => {
                println!("hcfd {:#x}: {} group(s), symmetric = {}", f.hcfd, f.groups.len(), f.is_symmetric());
                for (i, m) in f.groups.iter().enumerate() {
                    let with: Vec<String> =
                        (0..f.groups.len()).filter(|&j| m & (1u32 << j) != 0).map(|j| j.to_string()).collect();
                    println!("  group {i:>2}: mask {m:#010x} collides with [{}]", with.join(","));
                }
            }
        }

        println!("\n== per-object physics, by what the gather would use ==");
        println!("{:<46} {:>5} {:>5}  shapes", "tag", "phmo", "coll");
        let mut rows = 0usize;
        for group in ["scen", "bloc", "ctrl", "mach", "bipd", "vehi", "weap"] {
            for &t in &native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, t).unwrap_or_default();
                let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
                let interesting = crate::map_spawns::is_map_spawn_name(&name)
                    || leaf.contains("kill_volume")
                    || leaf.contains("trait_zone")
                    || leaf.contains("soft_ceiling")
                    || leaf.starts_with("block_")
                    || leaf.starts_with("bridge")
                    || leaf == "spartans"
                    || leaf == "warthog"
                    || leaf.contains("crate");
                if !interesting || rows >= 30 {
                    continue;
                }
                rows += 1;
                let phmo = native.resolve_phmo(cache, t);
                let coll = native.resolve_coll(cache, t);
                let has = |v: u32| if v != 0 && v != 0xFFFF_FFFF { "yes" } else { "-" };
                let sf = shape_filters(native, cache, t);
                let desc: Vec<String> = sf
                    .iter()
                    .take(4)
                    .map(|s| format!("{}#{} g{} ph{:?} m{}", s.kind, s.index, s.group, s.phantom, s.material))
                    .collect();
                println!(
                    "{:<46} {:>5} {:>5}  {} shape(s): {}",
                    &leaf[..leaf.len().min(45)],
                    has(phmo),
                    has(coll),
                    sf.len(),
                    desc.join(", ")
                );
            }
        }
    }
}
