// Diagnostic (#objmask): the full authored material record of every shader of a render_model —
// the `environment_map` cube (tag, curve, format, mean of its six decoded faces, so the engine's
// per_pixel decode `reflection.rgb * reflection.a` can be checked against what HMS uploads), the
// mean of each bound base / detail / bump_detail / material texture (the detail map's ALPHA is
// half the engine specular mask, `albedo.hlsl_include:121`), and every named shader constant.
// Usage: envcube_probe <dll_path> <map_path> <mode_tag_hex>...
//   e.g. envcube_probe target/release/libhalomapstudio.so .../forge_halo.map 0x2140   (coliseum wall)
use hms_native::NativeDll;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let native = NativeDll::load(Path::new(&args[1])).expect("load");
    let c = native.open_cache(&args[2]).expect("open");
    for a in &args[3..] {
        let mode = u32::from_str_radix(a.trim_start_matches("0x"), 16).expect("hex");
        let Ok(m) = native.open_model(c, mode) else { println!("{mode:#x}: open_model failed"); continue };
        println!("== mode {mode:#x} {}", native.tag_name(c, mode).unwrap_or_default());
        let n = native.model_submesh_count(m);
        let mut seen: Vec<i32> = Vec::new();
        for i in 0..n {
            let Some(sm) = native.model_submesh(m, i) else { continue };
            let sh = sm.shader_index;
            if sh < 0 || seen.contains(&sh) { continue }
            seen.push(sh);
            let st = native.model_shader_tag(m, sh);
            let name = if st >= 0 { native.tag_name(c, st as u32).unwrap_or_default() } else { String::new() };
            let et = native.model_shader_bitmap_by_usage(m, sh, "environment_map");
            print!("  shader {sh} {name} env_map={et:#x}");
            if et != 0 && et != 0xFFFF_FFFF {
                match native.find_bitmap(c, et, 0) {
                    Some(b) => print!(" [{}x{} fmt={} type={} curve={} mips={}] {}", b.width, b.height, b.format, b.bitmap_type, b.curve, b.mip_count, native.tag_name(c, et).unwrap_or_default()),
                    None => print!(" [no info]"),
                }
                let mut tot = [0f64; 4];
                let mut cnt = 0f64;
                for f in 0..6u32 {
                    if let Some((px, w, h)) = native.decode_bitmap_face(c, et, 0, 0, f, 1) {
                        for p in px.chunks_exact(4) { for k in 0..4 { tot[k] += p[k] as f64; } }
                        cnt += (w * h) as f64;
                    }
                }
                if cnt > 0.0 {
                    let m255 = |k: usize| tot[k] / cnt / 255.0;
                    // bgra order from the native decode
                    println!("  faces mean (b,g,r,a)=({:.3},{:.3},{:.3},{:.3})", m255(0), m255(1), m255(2), m255(3));
                } else { println!("  (no faces)"); }
            } else { println!(); }
            for usage in ["base_map","detail_map","bump_detail_map","material_texture"] {
                let t = native.model_shader_bitmap_by_usage(m, sh, usage);
                if t == 0 || t == 0xFFFF_FFFF { continue }
                let info = native.find_bitmap(c, t, 0);
                let mut tot = [0f64; 4];
                let mut cnt = 0f64;
                if let Some((px, w, h)) = native.decode_bitmap_keep_alpha(c, t, 0, 0) {
                    for p in px.chunks_exact(4) { for k in 0..4 { tot[k] += p[k] as f64; } }
                    cnt += (w * h) as f64;
                }
                let g = |k: usize| if cnt > 0.0 { tot[k] / cnt / 255.0 } else { -1.0 };
                println!("      MAP {usage:<18} {t:#x} fmt={} curve={} mean(b,g,r,a)=({:.3},{:.3},{:.3},{:.3}) {}",
                    info.as_ref().map_or(0, |i| i.format), info.as_ref().map_or(0, |i| i.curve),
                    g(0), g(1), g(2), g(3), native.tag_name(c, t).unwrap_or_default());
            }
            if let Some(k) = native.model_shader_constants(m, sh) {
                let n = k.real_count.max(k.scalar_count).min(64) as usize;
                for i in 0..n {
                    let base = i * 48;
                    let raw = &k.arg_names[base..base + 48];
                    let end = raw.iter().position(|&b| b == 0).unwrap_or(48);
                    let nm = String::from_utf8_lossy(&raw[..end]).to_string();
                    if nm.is_empty() { continue }
                    let o = i * 4;
                    println!("      {nm:<44} real=[{:.4},{:.4},{:.4},{:.4}] scalar={:.4}",
                        k.real_constants[o], k.real_constants[o+1], k.real_constants[o+2], k.real_constants[o+3], k.scalar_constants[i]);
                }
            }
        }
        native.close_model(m);
    }
}
