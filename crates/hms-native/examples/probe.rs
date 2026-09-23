// Diagnostic: load the parser DLL and try to open a .map, reporting each step.
// Usage: probe <dll_path> <map_path>
use hms_native::NativeDll;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: probe <dll_path> <map_path>");
        std::process::exit(2);
    }
    let dll = Path::new(&args[1]);
    let map = &args[2];

    println!("DLL : {}", dll.display());
    println!("MAP : {map}");
    println!("dll exists: {}", dll.exists());

    let native = match NativeDll::load(dll) {
        Ok(n) => {
            println!("NativeDll::load OK");
            n
        }
        Err(e) => {
            println!("NativeDll::load FAILED: {e:#}");
            std::process::exit(1);
        }
    };

    match native.open_cache(map) {
        Ok(c) => {
            println!("open_cache OK, handle set");
            let scnr = native.find_scenario(c);
            println!("find_scenario -> {scnr:?}");
            if let Some(s) = scnr {
                let sbsps = native.enumerate_sbsps(c, s, 64);
                println!("enumerate_sbsps -> {} bsp(s): {:?}", sbsps.len(), sbsps);
                let t0 = std::time::Instant::now();
                let mut total_meshes = 0u32;
                let mut total_tris = 0usize;
                for &sb in &sbsps {
                    if let Ok(b) = native.open_bsp(c, sb) {
                        let n = native.bsp_mesh_count(b);
                        total_meshes += n;
                        for i in 0..n {
                            if let Some(g) = native.bsp_decode_geometry(b, i) {
                                total_tris += g.indices.len() / 3;
                            }
                        }
                        native.close_bsp(b);
                    }
                }
                println!(
                    "DECODED all {} meshes ({} tris) in {:.2}s",
                    total_meshes,
                    total_tris,
                    t0.elapsed().as_secs_f32()
                );
            }
            let all_scnr = native.tags_by_group(c, "scnr");
            println!("tags_by_group(scnr) -> {all_scnr:?}");
            native.close_cache(c);
        }
        Err(e) => {
            println!("open_cache FAILED: {e:#}");
        }
    }
}
