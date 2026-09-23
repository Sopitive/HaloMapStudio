// Diagnostic: every structure BSP's world bounds (sbsp+0xF0) vs the physics
// `mopp bounds min/max` (sbsp+0x41C / +0x428) the engine's havok_world_new unions (+-64 wu pad)
// into the Havok broadphase AABB. Usage: moppbounds_probe <dll_path> <map_path>
use hms_native::NativeDll;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let native = NativeDll::load(Path::new(&args[1])).expect("load");
    let c = native.open_cache(&args[2]).expect("open");
    let f = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    for t in native.tags_by_group(c, "sbsp") {
        let wb = native.tag_meta(c, t, 0xF0, 0x18).unwrap();
        let mb = native.tag_meta(c, t, 0x40C, 0x34).unwrap();
        let n = native.tag_name(c, t).unwrap_or_default();
        println!("sbsp {t:#x} {n}");
        println!("  world bounds  x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}", f(&wb, 0), f(&wb, 4), f(&wb, 8), f(&wb, 12), f(&wb, 16), f(&wb, 20));
        println!("  mopp count={} bounds min ({:.1},{:.1},{:.1}) max ({:.1},{:.1},{:.1})",
            u32::from_le_bytes([mb[0], mb[1], mb[2], mb[3]]), f(&mb, 0x10), f(&mb, 0x14), f(&mb, 0x18), f(&mb, 0x1c), f(&mb, 0x20), f(&mb, 0x24));
    }
}
