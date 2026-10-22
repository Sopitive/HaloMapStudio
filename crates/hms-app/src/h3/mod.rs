//! Halo 3 (MCC) support. #h3
//!
//! Halo 3 MCC caches are a **Reach-family** container (same `daeh`/v13 header, fields shifted +8
//! after 0x58 vs Halo 4, pointer-expander added-magic 0), NOT the Halo 4 layout - so this is its
//! own reader rather than a thin arm over the Halo 4 one (which the H2A module is). The genuinely
//! new subsystem is the classic Gen3 resource gestalt (`zone`) + layout table (`play`) with
//! DEFLATE-compressed pageable blocks; see `cache.rs`.
//!
//!   cache.rs    - open a halo3 cache: header / tag index / string ids / scnr + the resource
//!                 gestalt/layout paging (DEFLATE page decode + control-data buffer)
//!   geometry.rs - Lbsp BSP render-mesh decode (VB/IB descriptors from the control data, raw bytes
//!                 from the pageable page, SNORM16 dequant via the compression bounds)
//!   albedo.rs   - the engine `calc_albedo_*_ps` composite (base x detail layers x
//!                 DETAIL_MULTIPLIER) baked into the uploaded BSP surface texture
//!   bitmaps.rs  - `bitm` element + pixel-stream extraction (classic bitmap format enum)
//!   materials.rs- render-method (rmsh/rmtr/...) base-texture resolve (classic Gen3 parameters)
//!   lightmaps.rs- Lbsp baked lighting (lightprobe DXT5 array + per-vertex streams, engine decode)
//!   objects.rs  - scnr placements / skies -> obje -> hlmt -> mode render models (same decoder)
//!   scene.rs    - THE mesh builder (BSP clusters + instances, sky, objects; textured, baked
//!                 lighting) streaming lanes to a caller: headless AND the GUI worker
//!   render.rs   - the headless screenshot entry over scene.rs
//!   gui.rs      - the GUI load path (worker thread + per-frame drain), App::h3_load_map
//!   mvar.rs     - the Halo 3 `.mvar` map-variant codec (BLF `mvar` v12 bitstream, halo3.dll
//!                 layout; decoder + byte-exact encoder + save) and the quota -> palette resolve
//!   palette.rs  - the scnr sandbox palette rows + the obje multiplayer defaults a placement takes
//!   edit.rs     - variant records <-> editor objects, the save list
//!   scenario.rs - the base map's own objects as variant slots (layout, editor objects, blank canvas)
//!   edit_scene.rs - `ObjectScene` over the Halo 3 reader (dynamic lanes, picks, BSP soup)
//!   app.rs      - the App side: editor install, palette panel, place, save
//!   attach.rs   - hlmt model variants + child objects (vehicle turrets) at their markers
//!
//! Several documented entry points have no caller yet (headless paths, probes).
#![allow(dead_code)]

pub mod albedo;
pub mod bitmaps;
pub mod cache;
pub mod geometry;
pub mod lightmaps;
pub mod materials;
pub mod gui;
pub mod mvar;
pub mod palette;
pub mod edit;
pub mod edit_scene;
pub mod app;
pub mod attach;
pub mod lighting;
pub mod objects;
pub mod objlight;
pub mod render;
pub mod scene;
pub mod scenario;

#[allow(unused_imports)]
pub use cache::{is_halo3_cache, maps_dir, H3Cache};
