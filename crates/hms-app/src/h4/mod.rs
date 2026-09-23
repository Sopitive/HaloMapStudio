//! Halo 4 (MCC) support: a pure-Rust cache reader and renderer path, separate from the Reach
//! pipeline (scene.rs + the native DLL), plus the Forge variant codec and editor scene.
//!
//!   cache.rs        - cache reader (header, tags, string ids, DEFLATE paging, zone gestalt)
//!   geometry.rs     - sbsp/Lbsp render geometry (meshes, instances, clusters, materials)
//!   bitmaps.rs      - bitm elements + resource streams -> DDS / BGRA for the renderer
//!   materials.rs    - mat / mats / mtsb: parameter names (DXBC reflection), blend byte -> lane kind
//!   shading.rs      - per-family BRDF constants read from the shipped srf_* pixel shaders
//!   lightmaps.rs    - Lbsp atlas bitmaps, per-vertex / probe / airprobe lighting, the baked sun
//!   lighting.rs     - scene lighting tags: fogg atmosphere fog -> fog uniform, cfxs post-process
//!   objects.rs      - scnr placements, hlmt -> mode render models, the shared mesh decoder
//!   attach.rs       - hlmt model-variant CHILD objects (turrets / guns) + marker frames
//!   collision.rs    - coll collision BSPs (in a collision_model_resource) + phmo rigid-body shapes
//!                     -> hull triangles for the editor's collision / physics / hidden-block overlays
//!   tint.rs         - change colours: mulg team palette + the scenario / Forge placement rule
//!   palette.rs      - scnr Forge palette (categories / entries / variants, localized names, MP
//!                     defaults) + `instantiate` = a new placement with its .mvar record
//!   unic.rs         - in-cache UI strings: matg locale tables + `unic` windows
//!   localization.rs - MCC `data/ui/Localization/*.bin` tables: `$key` titles -> English
//!   mvar.rs         - .mvar (chunk v50) codec + forge palette -> placements
//!   scene.rs        - cache -> BSPs / objects -> GpuMesh lanes (shared by headless + GUI)
//!   gui.rs          - App::h4_load_map / h4_drive_load: the picker's Halo 4 load path (worker thread)
//!   edit.rs         - editor records: H4Fields (the Halo 4-only ObjMeta block), tag ids,
//!                     H4ObjMeta, variant_to_editor, the save list builder + its gates
//!   edit_scene.rs   - H4ObjectScene: the editor's ObjectScene (dynamic lanes + picks + raycast
//!                     soup) over the Halo 4 reader;
//!   spawncam.rs     - the post-load camera: variant loadout camera / initial spawn -> scnr player
//!                     starting locations / spawn placements, stood off the geometry
//!   render.rs       - the headless screenshot path (HMS_SHOT + HMS_MAP=<halo4 map>)
//! Layout facts: docs/halo4_support_plan.md, docs/halo4_geometry_layout.md,
//! docs/halo4_lighting_model.md, docs/halo4_forge_palette.md, docs/halo4_mvar_layout.md.

// This module mirrors on-disk Halo 4 / H2A tag layouts, so it deliberately keeps items the
// current build does not all consume: struct fields decoded to document a record's shape even
// when nothing reads them yet, tag-offset constants that name where each field lives, engine
// tables/formulae ported verbatim from halo4.dll (e.g. `h4_map_index` / `h4_map_guid`, kept for
// the not-yet-enabled version-51 variants), and symmetric helper pairs (`shape_raw_to_wu` beside
// `shape_wu_to_raw`, `H4Pose::yawed` beside `at`). These are reference surface for contributors
// extending the reader, not cruft to delete, so `dead_code` is allowed module-wide rather than
// scattered per item. The heavy paths - cache / geometry / bitmaps / materials / lightmaps /
// lighting / shading / scene / render - carry no dead code regardless.
#![allow(dead_code)]

pub mod bitmaps;
pub mod cache;
pub mod geometry;
pub mod lightmaps;
pub mod materials;
pub mod localization;
pub mod mvar;
pub mod attach;
pub mod collision;
pub mod objects;
pub mod palette;
pub mod unic;
pub mod gui;
pub mod render;
pub mod scene;
pub mod shading;
pub mod lighting;
pub mod edit;
pub mod edit_scene;
pub mod spawncam;
pub mod tint;

pub use cache::is_halo4_cache;
