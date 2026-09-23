//! Halo 2 Anniversary (MCC `groundhog`) support. #h2a
//!
//! **H2A's cache format IS Halo 4's.** Everything verified so far - the header layout (build
//! string @152, map name @0xB8, scenario @0xD8, virtual base @728, tag index @736, section
//! offsets @1220, data table @1224, sections @1236), the tag index and class table, the tag-name
//! file table, the 17-bit string ids, the `play` page (88 B @+0x18) and segment (24 B @+0x30)
//! tables, raw-DEFLATE pages, the `zone` resource gestalt (types @+0x04, 68-B entries @+0x58,
//! blob @+0x154/+0x160), the sbsp / Lbsp block offsets and the `mat `/`mats` material tags - is
//! byte-for-byte the same shape as Halo 4's. The ONE difference is the pointer-expansion
//! constant: **0x7AC00000** instead of Halo 4's 0x4FFF0000.
//!
//! So this module does NOT clone the Halo 4 reader: `h4::cache::Engine` carries the per-engine
//! constants, `H4Cache` reads both engines, and every `h4::` module above it (geometry, bitmaps,
//! materials, lightmaps, scene, render) works unchanged. The files here are thin: they open a
//! groundhog cache with the right engine, locate the groundhog maps folder, and carry the tests
//! that assert the invariants on ALL shipped H2A caches (which is what pins the layout).
//!
//!   cache.rs    - `open` / `maps_dir` / `shipped_maps` + the cache invariant tests
//!   geometry.rs - BSP enumeration over `h4::geometry` + the geometry invariant tests
//!   bitmaps.rs  - texture decode over `h4::bitmaps` + its invariant tests
//!   render.rs   - the headless screenshot entry (delegates to `h4::render::headless_run`)
//!
//! Layout facts + the shared-vs-forked table: docs/h2a_support_plan.md.

// Phase 1 is read-only viewing, so a few documented entry points (`SHIPPED_MAPS`, `dxgi`) have no
// caller outside the tests yet; the same allow the h4 module carries. Drop it when phase 2 wires
// the scenario/variant paths.
#![allow(dead_code)]

pub mod bitmaps;
pub mod cache;
pub mod geometry;
pub mod render;

#[allow(unused_imports)]
pub use cache::{is_h2a_cache, maps_dir, open};
