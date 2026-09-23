//! Halo 2 Anniversary headless screenshot. #h2a
//!
//! `HMS_SHOT=<png> HMS_MAP=<groundhog map>` lands here from `headless::run`. The whole render
//! path - cache -> BSP / objects -> GpuMesh lanes -> post chain - is `h4::render::headless_run`
//! unchanged, because `h4::scene::load_map` opens the cache through `H4Cache::open`, which reads
//! groundhog caches with the H2A expander. Every `HMS_*` knob documented in `h4/render.rs`
//! applies here too, except the `.mvar` ones (H2A variants are BLF chunk v52, not yet decoded).

use anyhow::Result;

/// Render one groundhog cache to a PNG. The engine dispatch happens in `headless::run`.
pub fn headless_run(map_path: &str, out: &str, w: u32, h: u32) -> Result<()> {
    crate::h4::render::headless_run(map_path, out, w, h)
}
