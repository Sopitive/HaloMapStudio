// Diagnostic: cross-check the SoftCeilingWalker against the raw scnr metadata block
// (scnr+0x25C: flags/name/type per soft ceiling) and every sddt's soft-ceilings block.
// Usage: softceil_probe <dll_path> <map_path>
use hms_native::NativeDll;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let native = NativeDll::load(Path::new(&args[1])).expect("load");
    let c = native.open_cache(&args[2]).expect("open");
    let scnr = native.find_scenario(c).expect("scnr");
    let u32_at = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let u16_at = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    // scnr metadata block
    let blk = native.tag_meta(c, scnr, 0x25C, 12).unwrap();
    let (n, ptr) = (u32_at(&blk, 0), u32_at(&blk, 4));
    println!("scnr+0x25C soft ceilings metadata: count={n} ptr={ptr:#x}");
    if n > 0 && n < 1024 {
        let arr = native.tag_ptr_bytes(c, ptr, n * 12).unwrap();
        for i in 0..n as usize {
            let e = &arr[i * 12..i * 12 + 12];
            println!("  [{i}] flags={:#x} rtflags={:#x} name_sid={:#x} type={}", u16_at(e, 0), u16_at(e, 2), u32_at(e, 4), u16_at(e, 8));
        }
    }
    // every sddt
    for t in native.tags_by_group(c, "sddt") {
        let blk = native.tag_meta(c, t, 0x40, 12).unwrap();
        let (n, ptr) = (u32_at(&blk, 0), u32_at(&blk, 4));
        println!("sddt {t:#x} {:?}: soft ceilings count={n}", native.tag_name(c, t));
        if n > 0 && n < 128 {
            let arr = native.tag_ptr_bytes(c, ptr, n * 0x14).unwrap();
            for i in 0..n as usize {
                let e = &arr[i * 0x14..i * 0x14 + 0x14];
                println!("  [{i}] name_sid={:#x} type={} pad={:#x} tris={}", u32_at(e, 0), u16_at(e, 4), u16_at(e, 6), u32_at(e, 8));
            }
        }
    }
    for sc in native.soft_ceilings(c, scnr) {
        println!("walker: {:?} kind={} flags={} tris={}", sc.name, sc.kind_name(), sc.flags, sc.tris.len());
        if let Some(t) = sc.tris.first() { println!("   first tri {:?}", t); }
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for t in &sc.tris { for v in t { for k in 0..3 { lo[k] = lo[k].min(v[k]); hi[k] = hi[k].max(v[k]); } } }
        println!("   bounds {lo:?} .. {hi:?}");
        // coarse 50-wu XY histogram of triangle centroids + total area
        let mut cells = std::collections::BTreeMap::new();
        let mut area = 0.0f32;
        for t in &sc.tris {
            let c = [(t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0];
            *cells.entry(((c[0] / 50.0).floor() as i32, (c[1] / 50.0).floor() as i32)).or_insert(0) += 1;
            let e1 = [t[1][0]-t[0][0], t[1][1]-t[0][1], t[1][2]-t[0][2]];
            let e2 = [t[2][0]-t[0][0], t[2][1]-t[0][1], t[2][2]-t[0][2]];
            let cx = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
            area += 0.5 * (cx[0]*cx[0]+cx[1]*cx[1]+cx[2]*cx[2]).sqrt();
        }
        println!("   area {area:.0} wu^2; cells (50wu): {cells:?}");
    }
}
