//! hms-app — Halo Map Studio. eframe/egui shell with a wgpu 3D viewport.
//!
//! This file holds the `App` state and its `update()` loop: window + render-to-texture
//! viewport + fly camera, the menus and panels, the editor verbs (select / transform / place /
//! undo), map + variant loading and saving, the script host hooks and the command-line
//! diagnostics. The scene itself lives in `scene.rs`; Halo 4 support in `h4/` + `h4_app.rs`.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;
use hms_ipc::{
    ForgeObjectTableClient, ForgePaletteClient, ObjectInfo, ObjectTableClient, PaletteEntry,
    TransformQueueClient, WorldSpawnClient,
};
use hms_native::NativeDll;
use hms_render::{Camera, SceneRenderer};

mod bake_compare;
mod cmd_server;
mod construct;
mod euler;
mod decal_projector;
mod forge_scale;
mod h2a; // Halo 2 Anniversary (MCC "groundhog") support -- the Halo 4 reader with one constant changed
mod h4; // Halo 4 (MCC) support
mod h4_app; // Halo 4 Forge editing in the GUI (palette / objects / save / Object window rows)
mod headless;
mod lightbake;
mod lightprobe;
mod map_spawns; // View > Show map spawn points
mod mapcat;
mod physics_outlines; // View > Show physics outlines (hidden-blocker hulls)
mod soft_ceilings; // View > Soft ceilings (the map floor / kill planes)
mod hard_floor; // View > Hard floor (playable BSP world bounds)
mod wire_xray; // View > Selection wireframe through objects (#wire-visible)
mod color_hover; // team/colour dropdown hover preview (pure predicate)
mod mvar;
mod numfield; // signed numeric text entry (a '-' anywhere = sign flip)
mod project;
mod scene;
mod screenfx; // Forge special-FX screen effects
mod obj_identity; // what a clicked object IS (tag path / class / tag id)
mod objscene; // the game-agnostic object-scene trait (Reach SceneController / Halo 4 H4ObjectScene)
mod script;
mod script_host;
// #snap-array: pure geometry for the oriented face snap, edge alignment and line arrays.
mod snap;
mod voxel;
use scene::SceneController;
use objscene::objscene_parts_mut;

/// Status text when the live player pose is unavailable (the pose MMF only exists with the
/// injected DLL, i.e. the `injection` build attached to a running game).
const NO_PLAYER_POSE: &str = if cfg!(feature = "injection") {
    "No player pose (inject + be in a game)."
} else {
    "No player pose (live-game feature; injection build only)."
};

/// Payload DLL filename. Defined here (not in hms-inject) so the offline `dll_path`
/// plumbing compiles with the `injection` feature OFF (hms-inject not linked).
#[cfg(windows)]
pub(crate) const PAYLOAD_DLL: &str = "HaloMapStudioDLL.dll";
/// Same sources, ELF instead of PE: built by `native/build-linux.sh`. Only the
/// offline parsers are present -- the injection TUs are Windows-only.
#[cfg(not(windows))]
pub(crate) const PAYLOAD_DLL: &str = "libhalomapstudio.so";

/// CLI diagnostics: the native parser (next to the exe) wrapped in a `SceneController`.
/// Reports *why* the library is unavailable instead of silently doing nothing -- a
/// missing/mismatched library otherwise looks like "map loading is broken" with no explanation.
fn cli_scene_controller() -> Option<SceneController> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
    let dll_path = exe_dir.join(crate::PAYLOAD_DLL);
    match hms_native::NativeDll::load(&dll_path) {
        Ok(dll) => Some(SceneController::new(dll)),
        Err(e) => {
            eprintln!("native library {} failed to load: {e:#}", dll_path.display());
            None
        }
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    // Diagnostic: `hms-app --dump-mvar <file.mvar> [more.mvar ...]` prints each variant's
    // base map id + object count, then exits. Used to catalog Reach map ids offline.
    {
        let args: Vec<String> = std::env::args().collect();
        // `hms-app --dump-h4-palette <map or map name> [more ...]`: print-only listing
        // of a Halo 4 map's Forge palette (categories, entries, variants, localized names, MP
        // defaults). A bare name (`ca_forge_ravine`) is looked up among the detected Halo 4 maps.
        if let Some(pos) = args.iter().position(|a| a == "--dump-h4-palette") {
            for p in &args[pos + 1..] {
                let path = std::path::Path::new(p);
                let path = if path.is_file() { path.to_path_buf() } else {
                    match mapcat::enumerate().into_iter().find(|c| c.game == mapcat::Game::Halo4 && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(p))) {
                        Some(c) => c.path,
                        None => { println!("{p}: not a file and not a detected Halo 4 map"); continue; }
                    }
                };
                match h4::cache::H4Cache::open(&path) {
                    Ok(c) => print!("{}", h4::palette::palette(&c).describe()),
                    Err(e) => println!("{}: {e}", path.display()),
                }
            }
            std::process::exit(0);
        }
        // `hms-app --dump-h4-mvar <file.mvar> [more ...]`: print-only listing of a Halo 4
        // variant (header + every object; palette names resolved when the base map is found).
        if let Some(pos) = args.iter().position(|a| a == "--dump-h4-mvar") {
            for p in &args[pos + 1..] {
                let path = std::path::Path::new(p);
                match h4::mvar::parse_h4_variant(path) {
                    Ok(v) => {
                        let cache = mapcat::enumerate().into_iter()
                            .find(|c| c.game == mapcat::Game::Halo4 && c.map_id == Some(v.map_id))
                            .and_then(|c| h4::cache::H4Cache::open(&c.path).ok());
                        let palette = cache.as_ref().map(h4::mvar::load_forge_palette);
                        println!("{p}");
                        print!("{}", h4::mvar::describe(&v, palette.as_deref()));
                        if let Some(c) = &cache {
                            let (_, st) = h4::mvar::variant_placements(c, &v, palette.as_deref().unwrap_or(&[]));
                            println!("  base map '{}': {st:?}", c.map_name);
                        } else {
                            println!("  base map id {} not among the detected Halo 4 maps (names unresolved)", v.map_id);
                        }
                    }
                    Err(e) => println!("{p}: {e}"),
                }
            }
            std::process::exit(0);
        }
        // `hms-app --h4-mvar-roundtrip <file.mvar> [out.mvar]`: decode, re-encode
        // with no edits, report whether the result is byte-identical (optionally writing it).
        if let Some(pos) = args.iter().position(|a| a == "--h4-mvar-roundtrip") {
            let file = args.get(pos + 1).map(std::path::PathBuf::from);
            let out = args.get(pos + 2).map(std::path::PathBuf::from);
            match file {
                Some(f) => match h4::mvar::roundtrip_report(&f, out.as_deref()) {
                    Ok(r) => { println!("{r}"); std::process::exit(if r.contains("BYTE-IDENTICAL") { 0 } else { 2 }); }
                    Err(e) => { println!("{}: {e}", f.display()); std::process::exit(1); }
                },
                None => { println!("usage: hms-app --h4-mvar-roundtrip <file.mvar> [out.mvar]"); std::process::exit(1); }
            }
        }
        if let Some(pos) = args.iter().position(|a| a == "--dump-mvar") {
            for p in &args[pos + 1..] {
                match mvar::parse_variant(std::path::Path::new(p)) {
                    Some(v) => {
                        let folders: std::collections::BTreeSet<u16> =
                            v.objects.iter().map(|o| o.folder).collect();
                        let labels: Vec<String> = v.labels.iter().filter(|l| !l.is_empty()).cloned().collect();
                        println!("  labels[{}]: {}", v.labels.len(),
                            if labels.is_empty() { "<none decoded>".to_string() } else { labels.join(", ") });
                        println!(
                            "{p}\n  map_id={} (0x{:08X})  quotas={}  objects={}  folders={:?}",
                            v.map_id, v.map_id, v.num_quotas, v.objects.len(), folders
                        );
                        // Histogram: folder -> count (desc), then (folder,item) pairs, so we can
                        // see which quota indices dominate vs the rare high outliers.
                        let mut hist: std::collections::BTreeMap<u16, usize> =
                            std::collections::BTreeMap::new();
                        for o in &v.objects { *hist.entry(o.folder).or_default() += 1; }
                        let mut by_count: Vec<(u16, usize)> = hist.into_iter().collect();
                        by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        print!("  folder histogram (folder×count):");
                        for (f, c) in &by_count { print!(" {f}×{c}"); }
                        println!();
                        // How many objects the SCALED / SHADOW defaults would touch
                        // (scale label; green + scale label), with the label's team split.
                        let mut by_team: std::collections::BTreeMap<i32, usize> = Default::default();
                        for o in &v.objects {
                            let label = v.labels.get(o.label_idx as usize).map(|s| s.as_str()).unwrap_or("");
                            if o.label_idx != 0xFFFF && forge_scale::is_scale_label(label) {
                                *by_team.entry(if o.team == 0xFF { -1 } else { o.team as i32 }).or_default() += 1;
                            }
                        }
                        let n_scale: usize = by_team.values().sum();
                        let n_green = by_team.get(&(forge_scale::TEAM_GREEN as i32)).copied().unwrap_or(0);
                        println!("  scale-label objects={n_scale} (green+scale shadow casters={n_green}) by team: {by_team:?}");
                        // Every object's team byte (decoded: 0..7 colours, 8 neutral, -1 none)
                        // and explicit colour, so a variant's team usage can be audited offline.
                        let mut tc: std::collections::BTreeMap<(i32, i32), usize> = Default::default();
                        for o in &v.objects {
                            *tc.entry((if o.team == 0xFF { -1 } else { o.team as i32 }, o.color)).or_default() += 1;
                        }
                        let tcs: Vec<String> = tc.iter().map(|((t, c), n)| format!("team {t}/color {c}: {n}")).collect();
                        println!("  team/colour histogram: {}", tcs.join(", "));
                    }
                    None => println!("{p}\n  <parse failed>"),
                }
            }
            std::process::exit(0);
        }
        // `--mvar-delete`: delete objects from a variant by SLOT number, writing a new file
        // (e.g. to strip a canvas's default weapons/spawns out of a build). Never writes over
        // the input.
        if let Some(pos) = args.iter().position(|a| a == "--mvar-delete") {
            let (src, dst, list) = (
                args.get(pos + 1).cloned().unwrap_or_default(),
                args.get(pos + 2).cloned().unwrap_or_default(),
                args.get(pos + 3).cloned().unwrap_or_default(),
            );
            if src.is_empty() || dst.is_empty() || list.is_empty() {
                eprintln!("usage: --mvar-delete <in.mvar> <out.mvar> <slot,slot,...>");
                std::process::exit(2);
            }
            let want: std::collections::HashSet<u16> =
                list.split(',').filter_map(|t| t.trim().parse::<u16>().ok()).collect();
            let srcp = std::path::PathBuf::from(&src);
            let Some(variant) = mvar::parse_variant(&srcp) else {
                eprintln!("cannot parse {src}");
                std::process::exit(1);
            };
            let mut edits: std::collections::HashMap<usize, mvar::ObjEdit> = Default::default();
            for (i, o) in variant.objects.iter().enumerate() {
                if want.contains(&o.slot) {
                    edits.insert(i, mvar::ObjEdit { deleted: true, ..Default::default() });
                }
            }
            println!("{src}: {} objects, deleting {} of the {} slots requested",
                variant.objects.len(), edits.len(), want.len());
            match mvar::save_with_edits(&srcp, std::path::Path::new(&dst), &edits, &[], None) {
                Ok(_) => {
                    let left = mvar::parse_variant(std::path::Path::new(&dst))
                        .map(|v| v.objects.len()).unwrap_or(0);
                    println!("wrote {dst} — {left} objects remain");
                }
                Err(e) => { eprintln!("save failed: {e}"); std::process::exit(1); }
            }
            std::process::exit(0);
        }
        if args.iter().any(|a| a == "--dump-maps") {
            let all = mapcat::enumerate();
            for c in &all {
                println!("{:>10}  modded={}  game={}  {}  [{}]  {}",
                    c.map_id.map(|i| i.to_string()).unwrap_or_else(|| "-".into()),
                    c.modded as u8, c.game.short_name(), c.stem, c.file_name, c.path.display());
            }
            // per-title summary
            let reach: Vec<&mapcat::MapCandidate> = all.iter().filter(|c| c.game == mapcat::Game::Reach).collect();
            let h4: Vec<&mapcat::MapCandidate> = all.iter().filter(|c| c.game == mapcat::Game::Halo4).collect();
            println!("reach: {} total, {} stock", reach.len(), reach.iter().filter(|c| !c.modded).count());
            println!("halo4: {} total (mp={} campaign={})", h4.len(),
                h4.iter().filter(|c| c.h4_type == Some(1)).count(),
                h4.iter().filter(|c| c.h4_type == Some(0)).count());
            // #h2a: groundhog caches share the Halo 4 header, so `h4_type` reads the same field
            let h2a: Vec<&mapcat::MapCandidate> = all.iter().filter(|c| c.game == mapcat::Game::H2A).collect();
            println!("h2a: {} total (mp={})", h2a.len(), h2a.iter().filter(|c| c.h4_type == Some(1)).count());
            for d in mapcat::variant_dirs() {
                println!("  variants dir: {}", d.display());
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--bench-mvar") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            let mvar = args.get(pos + 2).cloned().unwrap_or_default();
            if let Some(mut scene) = cli_scene_controller() {
                let t_open = std::time::Instant::now();
                let opened = scene.open_cache(&map);
                if let Err(e) = &opened {
                    eprintln!("open_cache({map}) failed: {e:#}");
                }
                if opened.is_ok() {
                    println!("open_cache: {} ms", t_open.elapsed().as_millis());
                    let Some(variant) = mvar::parse_variant(std::path::Path::new(&mvar)) else { println!("parse failed"); std::process::exit(1); };
                    let t_pal = std::time::Instant::now();
                    let palette = scene.forge_palette_full();
                    let types = scene.forge_type_order(&palette);
                    println!("palette+types: {} ms  ({} types)", t_pal.elapsed().as_millis(), types.len());
                    let t_res = std::time::Instant::now();
                    let mut tags = Vec::new();
                    let mut unresolved = 0usize;
                    for o in &variant.objects {
                        let (m, _) = scene.resolve_forge_model(&palette, &types, o.folder, o.item);
                        if m == 0 || m == 0xFFFF_FFFF { unresolved += 1; } else { tags.push(m); }
                    }
                    println!("resolve {} objects: {} ms  ({} unresolved)", variant.objects.len(), t_res.elapsed().as_millis(), unresolved);
                    let mut uniq = tags.clone(); uniq.sort_unstable(); uniq.dedup();
                    println!("distinct models: {}", uniq.len());
                    let _ = scene.page_stats(); // reset counters
                    let t_dec = std::time::Instant::now();
                    let ndec = scene.predecode_tags(&tags);
                    let (g, m, p) = scene::obj_decode_timers_ms();
                    let (sh, subm) = scene::obj_shcall_ms();
                    let (bc, bm, bd) = scene::bmp_cache_stats();
                    let (ph, pi, pmb, pms, ps) = scene.page_stats();
                    println!("predecode {} distinct models: {} ms  (CPU-sum geom={:.0} mat={:.0} parts={:.0} [submesh-loop={:.0} shader-calls={:.0}] ms)", ndec, t_dec.elapsed().as_millis(), g, m, p, subm, sh);
                    println!("  bitmap cache: {bc} calls, {bm} misses (decodes), {bd:.0} ms decoding");
                    println!("  PAGE cache: {ph} hits, {pi} inflates ({pmb:.0} MB, {pms:.0} ms CPU-sum), {ps} stores");
                    let failed = scene.count_failed_models(&uniq);
                    println!("decode FAILED (null) models: {}", failed);
                    // AMORTIZATION PROOF: decode the SAME variant's models again. With the cache
                    // warm from the first pass, this is the true per-subsequent-variant cost in a
                    // batch that loads the base map + palette once. Then time resolve+build too.
                    let t_warm = std::time::Instant::now();
                    let ndec2 = scene.predecode_tags(&tags);
                    println!("WARM re-decode (batch per-variant cost): {} ms  ({} newly decoded — expect 0)", t_warm.elapsed().as_millis(), ndec2);
                    // Full per-variant work minus GPU upload: parse a fresh variant + resolve + warm predecode.
                    let t_v2 = std::time::Instant::now();
                    if let Some(v2) = mvar::parse_variant(std::path::Path::new(&mvar)) {
                        let mut tg = Vec::new();
                        for o in &v2.objects {
                            let (m, _) = scene.resolve_forge_model(&palette, &types, o.folder, o.item);
                            if m != 0 && m != 0xFFFF_FFFF { tg.push(m); }
                        }
                        let nd = scene.predecode_tags(&tg);
                        println!("WARM full per-variant (parse+resolve+predecode, no GPU): {} ms  ({} decoded)", t_v2.elapsed().as_millis(), nd);
                    }
                    // ONE-TIME batch setup: pre-warm the ENTIRE palette (all object types) so ANY
                    // variant is warm. This runs once per base map at batch start.
                    let t_pw = std::time::Instant::now();
                    let pal_tags = scene.forge_palette_model_tags();
                    let npw = scene.predecode_tags(&pal_tags);
                    println!("FULL-PALETTE pre-warm (one-time batch setup): {} ms  ({} more models decoded, {} palette types total)", t_pw.elapsed().as_millis(), npw, pal_tags.len());
                }
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--resolve-mvar") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            let mvar = args.get(pos + 2).cloned().unwrap_or_default();
            if let Some(mut scene) = cli_scene_controller() {
                if scene.open_cache(&map).is_ok() {
                    let Some(variant) = mvar::parse_variant(std::path::Path::new(&mvar)) else { println!("parse failed"); std::process::exit(1); };
                    let palette = scene.forge_palette_full();
                    let types = scene.forge_type_order(&palette);
                    println!("{} types, {} objects", types.len(), variant.objects.len());
                    // Only print the objects whose folder is OUT OF RANGE (unresolved by entry-index)
                    // plus a count summary — these are the ones that don't render.
                    let print_all = args.iter().any(|a| a == "--all");
                    let mut oor = 0usize;
                    // Position spread over the resolved (in-range) objects, to see the play-area cluster.
                    let mut cx = (f32::MAX, f32::MIN); let mut cy = (f32::MAX, f32::MIN); let mut cz = (f32::MAX, f32::MIN);
                    for (i, o) in variant.objects.iter().enumerate() {
                        let in_range = (o.folder as usize) < types.len();
                        let (m, _) = scene.resolve_forge_model(&palette, &types, o.folder, o.item);
                        let name = types.get(o.folder as usize).and_then(|&(pi, ew)| {
                            palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.as_str())
                        }).unwrap_or("<OOR>");
                        if in_range && m != 0 && m != 0xFFFF_FFFF {
                            cx.0 = cx.0.min(o.pos[0]); cx.1 = cx.1.max(o.pos[0]);
                            cy.0 = cy.0.min(o.pos[1]); cy.1 = cy.1.max(o.pos[1]);
                            cz.0 = cz.0.min(o.pos[2]); cz.1 = cz.1.max(o.pos[2]);
                        }
                        let bad = !in_range || m == 0 || m == 0xFFFF_FFFF;
                        if bad { oor += 1; }
                        if bad || print_all {
                            println!("  [{i}] folder={} item={} pos=[{:.1},{:.1},{:.1}] name='{name}' mode=0x{m:x} {}",
                                o.folder, o.item, o.pos[0], o.pos[1], o.pos[2],
                                if !in_range { "OUT-OF-RANGE" } else if m == 0 { "decode-fail" } else { "" });
                        }
                    }
                    println!("total not-rendered: {oor}");
                    println!("in-range object bbox: x[{:.0},{:.0}] y[{:.0},{:.0}] z[{:.0},{:.0}]", cx.0, cx.1, cy.0, cy.1, cz.0, cz.1);
                }
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--dump-model") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            let mode = args.get(pos + 2).and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
            if let Some(mut scene) = cli_scene_controller() {
                if scene.open_cache(&map).is_ok() {
                    scene.dump_model_materials(mode);
                }
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--dump-mask") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            let obj = args.get(pos + 2).and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
            if let Some(mut scene) = cli_scene_controller() {
                if scene.open_cache(&map).is_ok() {
                    if obj != 0 {
                        scene.dump_variant_mask(obj);
                    } else {
                        // No obj tag given: treat the 3rd arg as a NAME filter so any palette
                        // entry can be found (not just the hardcoded "ghost").
                        let needle = args.get(pos + 2).cloned().unwrap_or_else(|| "ghost".into()).to_lowercase();
                        for e in scene.forge_palette_full() {
                            let n = format!("{} {}", e.name, e.variant_name).to_lowercase();
                            if n.contains(&needle) {
                                println!("candidate: obj=0x{:08X} name='{}' variant='{}'", e.tag_short, e.name, e.variant_name);
                            }
                        }
                    }
                }
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--dump-parts") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            let key = args.get(pos + 2).cloned().unwrap_or_default();
            if let Some(mut scene) = cli_scene_controller() {
                if scene.open_cache(&map).is_ok() {
                    if let Ok(obj) = u32::from_str_radix(key.trim_start_matches("0x"), 16) {
                        scene.dump_object_parts(obj);
                    } else {
                        // name substring → find matching palette obj(s)
                        for e in scene.forge_palette_full() {
                            let n = format!("{} {}", e.name, e.variant_name).to_lowercase();
                            if !key.is_empty() && n.contains(&key.to_lowercase()) {
                                println!("candidate: obj=0x{:08X} name='{}' variant='{}'", e.tag_short, e.name, e.variant_name);
                                scene.dump_object_parts(e.tag_short);
                            }
                        }
                    }
                }
            }
            std::process::exit(0);
        }
        if let Some(pos) = args.iter().position(|a| a == "--dump-palette") {
            let map = args.get(pos + 1).cloned().unwrap_or_default();
            match cli_scene_controller() {
                Some(mut scene) => {
                    match scene.open_cache(&map) {
                        Ok(()) => {
                            let palette = scene.forge_palette_full();
                            let types = scene.forge_type_order(&palette);
                            println!("map={map}\npalette entries (p,e,v rows): {}\ndistinct object types (quota order): {}", palette.len(), types.len());
                            let show_variants = args.iter().any(|a| a == "--variants");
                            for (q, &(pi, ew)) in types.iter().enumerate() {
                                let nm = palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.as_str()).unwrap_or("");
                                let vars: Vec<_> = palette.iter().filter(|e| e.palette_index == pi && e.entry_within == ew).collect();
                                println!("  [{q:3}] cat={pi} entry={ew} variants={}  {nm}", vars.len());
                                if show_variants && vars.len() > 1 {
                                    for e in &vars {
                                        let mode = scene.resolve_object_mode(e.tag_short);
                                        println!("        v={} obj=0x{:08X} mode=0x{:08X} '{}'", e.variant_within, e.tag_short, mode, e.variant_name);
                                    }
                                }
                            }
                        }
                        Err(e) => println!("open_cache failed: {e:#}"),
                    }
                }
                None => {}
            }
            std::process::exit(0);
        }
    }
    // Cap the rayon pool so the parallel load (geometry/texture/decorator decode) leaves
    // cores for the eframe render/main thread — saturating every core freezes the
    // interactive window during the decode/decorator phases. Reserve 2 cores for the UI.
    let rayon_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(2)
        .max(1);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(rayon_threads).build_global();
    // Headless scripting mode (no window): `--script <file>` runs a script file; `--exec "<cmds>"`
    // runs an inline script — load maps/variants, query data, and screenshot from a script (with
    // `foreach` for batches). Never touches eframe/winit, so it works in non-interactive contexts.
    {
        let args: Vec<String> = std::env::args().collect();
        let script_src = if let Some(p) = args.iter().position(|a| a == "--script") {
            match args.get(p + 1) {
                Some(path) => Some(std::fs::read_to_string(path).unwrap_or_else(|e| {
                    eprintln!("--script: cannot read {path}: {e}");
                    std::process::exit(1);
                })),
                None => {
                    eprintln!("--script needs a file path");
                    std::process::exit(1);
                }
            }
        } else if let Some(p) = args.iter().position(|a| a == "--exec") {
            Some(args[p + 1..].join(" "))
        } else {
            None
        };
        if let Some(src) = script_src {
            match script_host::run(src) {
                Ok(()) => std::process::exit(0),
                Err(e) => {
                    eprintln!("script failed: {e:#}");
                    std::process::exit(1);
                }
            }
        }
    }
    // Headless offscreen single-shot screenshot mode (the render-calibration harness): render a
    // map to a PNG with a scripted camera + scene diagnostics.
    if std::env::var("HMS_SHOT").is_ok() {
        match headless::run() {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("HMS_SHOT failed: {e:#}");
                std::process::exit(1);
            }
        }
    }
    // Crash logger: capture any Rust panic (assert/unwrap/index/alloc) to a file next to
    // the exe so a crash on a user machine is diagnosable without a console. Native access
    // violations (e.g. decoding a foreign tag) can't be caught here — those are prevented
    // upstream by the map-match / cache-validity guards.
    {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown>".into());
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".into());
            let bt = std::backtrace::Backtrace::force_capture();
            let log = format!("PANIC at {loc}\n  {msg}\n\nbacktrace:\n{bt}\n");
            let path = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("hms_crash.log")))
                .unwrap_or_else(|| std::path::PathBuf::from("hms_crash.log"));
            let _ = std::fs::write(&path, &log);
            eprintln!("{log}\n(written to {})", path.display());
            default(info);
        }));
    }
    // Show the EXE's own build (last-modified) time in the title so a STALE running instance is
    // obvious at a glance — a map reload does NOT reload the exe, so "I relaunched" can still be
    // an old process.
    let build_stamp: String = std::env::current_exe()
        .and_then(|p| std::fs::metadata(&p))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| {
            // Local-ish HH:MM from the unix secs (no chrono dep): show secs-of-day in UTC + raw.
            let secs = d.as_secs();
            let hm = (secs % 86400) / 60;
            format!("built@{:02}:{:02}Z (epoch {})", hm / 60, hm % 60, secs)
        })
        .unwrap_or_else(|| "built@?".into());
    let title = format!(
        "Halo Map Studio (Rust) — {build_stamp} [{}]",
        if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" }
    );
    // Also write the running build's stamp + exe path to the settings dir on every launch, so a
    // stale running process can be diagnosed without reading the title bar.
    if let Some(dir) = project::settings_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let exe = std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let _ = std::fs::write(dir.join("hms_build.txt"), format!("{title}\nexe={exe}\n"));
    }
    // Window/taskbar icon. Decoded from the PNG in assets/ (the same art is compiled into
    // the .exe as a resource by build.rs on Windows, which is what Explorer shows).
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")).ok();
    let native_options = eframe::NativeOptions {
        viewport: {
            let vb = egui::ViewportBuilder::default();
            let vb = match icon {
                Some(i) => vb.with_icon(std::sync::Arc::new(i)),
                None => vb,
            };
            vb
        }
            .with_title(&title)
            .with_inner_size([1400.0, 900.0])
            // Force the window on-screen, visible, and focused. eframe's default persists
            // the window rect across runs; a stale off-screen/other-monitor position makes
            // the window open where it can't be seen. Pin a known top-left position each
            // launch + explicit visible/active.
            .with_position([80.0, 80.0])
            .with_visible(true)
            .with_active(true),
        // Do NOT restore a persisted (possibly off-screen) window rect.
        persist_window: false,
        renderer: eframe::Renderer::Wgpu,
        // Force standard vsync (Fifo). The default AutoVsync can pick an unrecognized
        // present mode on some NVIDIA drivers (seen on 610.88), after which the UI
        // thread blocks on present.
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode: eframe::wgpu::PresentMode::Fifo,
            // Request TEXTURE_COMPRESSION_BC so the map's BC blocks upload directly (no CPU
            // decode/mip). Masked by adapter.features() so only supported features are
            // requested — can't break device creation. Everything else is the default.
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: std::sync::Arc::new(|adapter: &eframe::wgpu::Adapter| {
                    eframe::wgpu::DeviceDescriptor {
                        label: Some("hms-device"),
                        required_features: adapter.features() & eframe::wgpu::Features::TEXTURE_COMPRESSION_BC,
                        required_limits: adapter.limits(),
                        // Smaller allocator blocks (16-128 MB instead of 128-512 MB) so one
                        // live allocation cannot pin a half-gigabyte host-visible chunk after the load.
                        memory_hints: eframe::wgpu::MemoryHints::MemoryUsage,
                    }
                }),
                // Pick the adapter here so a machine WITHOUT a usable GPU still runs.
                // Order: discrete > integrated > virtual > other > CPU (software rasterizer: lavapipe
                // on Vulkan, WARP on DX12). HMS_SOFTWARE_GPU=1 forces the CPU adapter (debugging /
                // broken drivers). Only surface-compatible adapters are considered.
                native_adapter_selector: Some(std::sync::Arc::new(|adapters: &[eframe::wgpu::Adapter], surface: Option<&eframe::wgpu::Surface<'_>>| {
                    use eframe::wgpu::DeviceType;
                    let want_soft = std::env::var("HMS_SOFTWARE_GPU").is_ok();
                    let rank = |a: &eframe::wgpu::Adapter| -> i32 {
                        let t = a.get_info().device_type;
                        let base = match t {
                            DeviceType::DiscreteGpu => 0,
                            DeviceType::IntegratedGpu => 1,
                            DeviceType::VirtualGpu => 2,
                            DeviceType::Other => 3,
                            DeviceType::Cpu => 4,
                        };
                        if want_soft { if t == DeviceType::Cpu { -1 } else { base + 10 } } else { base }
                    };
                    let mut best: Option<(i32, usize)> = None;
                    for (i, a) in adapters.iter().enumerate() {
                        if let Some(s) = surface { if !a.is_surface_supported(s) { continue; } }
                        let r = rank(a);
                        if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                    }
                    match best {
                        Some((_, i)) => {
                            let info = adapters[i].get_info();
                            eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
                            Ok(adapters[i].clone())
                        }
                        None => Err("no wgpu adapter: no GPU driver and no software rasterizer (Linux: install lavapipe / vulkan-swrast; Windows: WARP ships with DX12)".to_string()),
                    }
                })),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Halo Map Studio (Rust)",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

/// 2D point-in-polygon (ray casting) for the Fill tool.
fn point_in_poly(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if ((a[1] > p[1]) != (b[1] > p[1]))
            && (p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1] + 1e-9) + a[0])
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Enumerate .mvar variant files the scripting/query layer can see. Scans, recursively:
///   - `<exe>/mvars` (the drop-in folder mvar::catalog uses)
///   - every directory listed in `HMS_MVAR_DIRS` (`;`-separated) — point this at your MCC
///     UserContent / saved-variants folder to surface the real library.
/// De-duplicated by path, sorted by file stem. Kept dependency-free so it works with no game.
pub(crate) fn variant_catalog() -> Vec<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("mvars"));
        }
    }
    if let Ok(dirs) = std::env::var("HMS_MVAR_DIRS") {
        for d in dirs.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            roots.push(std::path::PathBuf::from(d));
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().map(|x| x.eq_ignore_ascii_case("mvar")).unwrap_or(false)
                    && seen.insert(p.clone())
                {
                    out.push(p);
                }
            }
        }
    }
    out.sort_by(|a, b| a.file_stem().unwrap_or_default().cmp(b.file_stem().unwrap_or_default()));
    out
}

/// Tessellate a forge boundary shape into (bright wireframe segments, translucent fill triangles).
/// Shared by the interactive overlay pass and the headless script host so the holographic zone
/// looks identical in both. `boundary` are the raw 11-bit .mvar values; dequantised to world units
/// here (v/2047 × 200). Teleporters (cached 12-14) render cyan, other zones amber.
#[allow(clippy::type_complexity)]
pub(crate) fn boundary_geometry(
    shape: u8,
    boundary: [u16; 4],
    cached_type: u8,
    pos: glam::Vec3,
    fwd_in: glam::Vec3,
    up_in: glam::Vec3,
) -> (Vec<([f32; 3], [f32; 3], [f32; 3])>, Vec<([f32; 3], [f32; 4])>) {
    let mut segs: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
    let mut ztris: Vec<([f32; 3], [f32; 4])> = Vec::new();
    if shape == 0 {
        return (segs, ztris);
    }
    let dq = |v: u16| (v as f32 / 2047.0) * 200.0;
    let mut fwd = fwd_in;
    let mut up = up_in;
    if fwd.length_squared() < 1e-6 { fwd = glam::Vec3::X; }
    if up.length_squared() < 1e-6 { up = glam::Vec3::Z; }
    fwd = fwd.normalize();
    up = up.normalize();
    let right = fwd.cross(up).normalize_or_zero();
    let col = if (12..=14).contains(&cached_type) { [0.3, 0.9, 1.0] } else { [1.0, 0.8, 0.2] };
    let fill = [col[0], col[1], col[2], 0.06f32]; // low base opacity so the volume doesn't occlude surroundings
    let mut seg = |a: glam::Vec3, b: glam::Vec3| segs.push((a.into(), b.into(), col));
    let tri = |a: glam::Vec3, b: glam::Vec3, c: glam::Vec3, out: &mut Vec<([f32; 3], [f32; 4])>| {
        out.push((a.into(), fill)); out.push((b.into(), fill)); out.push((c.into(), fill));
    };
    match shape {
        1 => {
            let r = dq(boundary[0]).max(0.1);
            const N: usize = 28;
            for &(u, v) in &[(right, fwd), (right, up), (fwd, up)] {
                let mut prev = pos + u * r;
                for k in 1..=N {
                    let a = std::f32::consts::TAU * (k as f32 / N as f32);
                    let cur = pos + (u * a.cos() + v * a.sin()) * r;
                    seg(prev, cur);
                    prev = cur;
                }
            }
            const LAT: usize = 12;
            const LON: usize = 20;
            let pt = |ilat: usize, ilon: usize| {
                let th = std::f32::consts::PI * (ilat as f32 / LAT as f32);
                let ph = std::f32::consts::TAU * (ilon as f32 / LON as f32);
                pos + (right * (th.sin() * ph.cos()) + fwd * (th.sin() * ph.sin()) + up * th.cos()) * r
            };
            for a in 0..LAT {
                for b in 0..LON {
                    let (p00, p01, p10, p11) = (pt(a, b), pt(a, b + 1), pt(a + 1, b), pt(a + 1, b + 1));
                    tri(p00, p10, p11, &mut ztris);
                    tri(p00, p11, p01, &mut ztris);
                }
            }
        }
        2 => {
            let r = dq(boundary[0]).max(0.1);
            let top = pos + up * dq(boundary[1]);
            let bot = pos - up * dq(boundary[2]);
            const N: usize = 28;
            let ring_pt = |c: glam::Vec3, k: usize| {
                let a = std::f32::consts::TAU * (k as f32 / N as f32);
                c + (right * a.cos() + fwd * a.sin()) * r
            };
            for k in 0..N {
                seg(ring_pt(top, k), ring_pt(top, k + 1));
                seg(ring_pt(bot, k), ring_pt(bot, k + 1));
            }
            for k in 0..4 {
                let a = std::f32::consts::TAU * (k as f32 / 4.0);
                let d = (right * a.cos() + fwd * a.sin()) * r;
                seg(bot + d, top + d);
            }
            for k in 0..N {
                let (bt0, bt1, tp0, tp1) = (ring_pt(bot, k), ring_pt(bot, k + 1), ring_pt(top, k), ring_pt(top, k + 1));
                tri(bt0, bt1, tp1, &mut ztris);
                tri(bt0, tp1, tp0, &mut ztris);
                tri(top, tp0, tp1, &mut ztris);
                tri(bot, bt1, bt0, &mut ztris);
            }
        }
        _ => {
            let w = dq(boundary[0]).max(0.05);
            let l = dq(boundary[1]).max(0.05);
            let tp = dq(boundary[2]);
            let bt = dq(boundary[3]);
            let corner = |sx: f32, sy: f32, upper: bool| {
                pos + right * (sx * w) + fwd * (sy * l) + up * (if upper { tp } else { -bt })
            };
            let cs = [
                corner(-1.0, -1.0, false), corner(1.0, -1.0, false),
                corner(1.0, 1.0, false), corner(-1.0, 1.0, false),
                corner(-1.0, -1.0, true), corner(1.0, -1.0, true),
                corner(1.0, 1.0, true), corner(-1.0, 1.0, true),
            ];
            for (a, b) in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
                seg(cs[a], cs[b]);
            }
            for &(a, b, c, d) in &[(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)] {
                tri(cs[a], cs[b], cs[c], &mut ztris);
                tri(cs[a], cs[c], cs[d], &mut ztris);
            }
        }
    }
    (segs, ztris)
}

/// Friendly label for a forge category string (scnr palette[p].Name, e.g.
/// "ff_weapons_covenant"). Strips the `ff_`/`ff_thorage_` prefix, title-cases the rest, and applies
/// a few known reorderings ("weapons covenant" → "Covenant Weapons"). Modded categories keep their
/// own name; anything unrecognised is title-cased generically.
fn forge_category_pretty(raw: &str) -> String {
    if raw.is_empty() {
        return "Objects".to_string();
    }
    // Known Reach categories → clean labels.
    let known = match raw {
        "ff_structure" => Some("Structure"),
        "structure_blocks" => Some("Blocks"),
        "ff_weapons_human" => Some("Human Weapons"),
        "ff_weapons_covenant" => Some("Covenant Weapons"),
        "ff_vehicles" => Some("Vehicles"),
        "ff_armor_abilities" => Some("Armor Abilities"),
        "ff_gadgets" => Some("Gadgets"),
        "ff_objectives" => Some("Objectives"),
        "ff_spawning" => Some("Spawning"),
        "ff_scenery" => Some("Scenery"),
        _ => None,
    };
    if let Some(k) = known {
        return k.to_string();
    }
    // Generic: drop a leading ff_ / ff_thorage_ prefix, title-case the words. Modded "thorage"
    // categories are tagged so the user can tell them apart from the stock set.
    let modded = raw.starts_with("ff_thorage_");
    let body = raw.strip_prefix("ff_thorage_").or_else(|| raw.strip_prefix("ff_")).unwrap_or(raw);
    let mut words: Vec<String> = body
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect();
    if words.is_empty() {
        words.push("Objects".to_string());
    }
    let label = words.join(" ");
    if modded { format!("{label} (mod)") } else { label }
}

/// Which GUI save the "objects outside playable space" window is holding.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BspWarnAction {
    Save,
    SaveAs,
}

/// Clip a display string to `max` chars (char-safe), marking the cut with "...".
fn clip_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(3)).collect();
    out.push_str("...");
    out
}

/// Prettify a Halo Reach stringID token ("assault_rifle") into a human display name
/// ("Assault Rifle") for the UI ONLY. The raw stringID stays the source of truth for filtering,
/// categorisation, and rendering — call this at the final display string, never on the stored name.
/// (The engine's real authored menu strings live in 'unic' tags; this Title-Case transform is a
/// close approximation.)
fn prettify_stringid(raw: &str) -> String {
    // Composite labels ("entry · variant" / "entry / variant") → prettify each side.
    for sep in [" · ", " / ", ": "] {
        if let Some((a, b)) = raw.split_once(sep) {
            return format!("{}{}{}", prettify_stringid(a.trim()), sep, prettify_stringid(b.trim()));
        }
    }
    // Uppercase acronyms/initialisms that Title-Case would mangle ("dmr" → "DMR", not "Dmr").
    let acronym = |w: &str| -> Option<&'static str> {
        match w {
            "dmr" => Some("DMR"), "smg" => Some("SMG"), "ar" => Some("AR"), "br" => Some("BR"),
            "ctf" => Some("CTF"), "koth" => Some("KOTH"), "vip" => Some("VIP"), "ff" => Some("FF"),
            "mp" => Some("MP"), "ui" => Some("UI"), "fx" => Some("FX"), "hud" => Some("HUD"),
            "unsc" => Some("UNSC"), "aa" => Some("AA"), "2x2" => Some("2x2"), "4x4" => Some("4x4"),
            "5x5" => Some("5x5"), "1x1" => Some("1x1"), "3x3" => Some("3x3"),
            _ => None,
        }
    };
    raw.split(|c| c == '_' || c == ' ' || c == '-')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let lw = w.to_ascii_lowercase();
            if let Some(a) = acronym(lw.as_str()) { return a.to_string(); }
            let mut cs = w.chars();
            match cs.next() {
                Some(f) => f.to_ascii_uppercase().to_string() + &cs.as_str().to_ascii_lowercase(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// World-units → 11-bit boundary value (inverse of the v/2047×200 dequant).
pub(crate) fn wu_to_bval(v: f32) -> u16 {
    (v / 200.0 * 2047.0).round().clamp(0.0, 2047.0) as u16
}

/// Physics sub-enum of the forge placement-flags byte (bits 6-7; Mjolnir ForgeObject.Flags).
fn placement_physics_name(flags: u8) -> &'static str {
    match flags & 0b1100_0000 {
        0b0000_0000 => "normal",
        0b0100_0000 => "fixed",
        0b1100_0000 => "phased",
        _ => "?",
    }
}

/// One-line human summary of a forge placement-flags byte, for the properties panel + `get`.
/// Bit names are engine-authoritative (reach_tag_test `scenario_map_variant.cpp`); see mvar::PLACE_*.
fn placement_summary(flags: u8) -> String {
    let mut parts = vec![format!("physics:{}", placement_physics_name(flags))];
    if flags & mvar::PLACE_NOT_AT_START == 0 { parts.push("at-start".into()); } else { parts.push("not-at-start".into()); }
    match flags & mvar::PLACE_SYMMETRY_MASK {
        0x0C => parts.push("both".into()),
        mvar::PLACE_SYMMETRIC => parts.push("symmetric".into()),
        mvar::PLACE_ASYMMETRIC => parts.push("asymmetric".into()),
        _ => parts.push("neither-sym".into()),
    }
    if flags & mvar::PLACE_HIDE_UNLESS_REQUIRED != 0 { parts.push("hide-unless-required".into()); }
    if flags & mvar::PLACE_UNIQUE_SPAWN != 0 { parts.push("unique-spawn".into()); }
    if flags & mvar::PLACE_IS_SHORTCUT != 0 { parts.push("shortcut".into()); }
    parts.join(", ")
}

/// `preview where` readout (see App::hover_debug_rects).
#[derive(Clone, Copy, Debug, Default)]
struct HoverDebugRects {
    team_button: Option<egui::Rect>,
    team_popup: Option<egui::Rect>,
    color_button: Option<egui::Rect>,
    color_popup: Option<egui::Rect>,
}

/// Called at the END of a ComboBox `show_ui` closure (so it only runs while the popup
/// is open): is the pointer inside this popup? `ui.min_rect()` is the union of the entry rows (gaps
/// included); the expansion covers the popup frame's padding. `Ui::rect_contains_pointer` also
/// requires the popup's own layer to be top-most at the pointer, so a window drawn over it does not
/// count. No area ids involved.
fn popup_ui_contains_pointer(ui: &egui::Ui) -> (bool, egui::Rect) {
    let rect = ui.min_rect().expand(8.0);
    (ui.rect_contains_pointer(rect), rect)
}

/// egui swatch colour for a forge team/colour index (-1 none → gray, 8 neutral → white, 0..7 palette).
fn forge_swatch32(idx: i32) -> egui::Color32 {
    match idx {
        n if (0..8).contains(&n) => {
            let c = scene::forge_color_srgb(n as usize);
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }
        8 => egui::Color32::WHITE,
        _ => egui::Color32::from_gray(120),
    }
}

/// "0xD0000012 name: scaled ON (auto, x2.5) shadow OFF (set)" for `flags` / `get`.
fn obj_flags_line(datum: u32, m: &ObjMeta, g: &forge_scale::GlobalFlags, conv: forge_scale::ScaleConvention) -> String {
    let how = |o: Option<bool>| if o.is_some() { "set" } else { "auto" };
    let (fs, fc) = forge_scale::effective_flags(g, &m.flags, m.team, &m.label);
    let team_i = if m.team == 0xFF { -1 } else { m.team as i32 };
    format!(
        "0x{datum:08X} {}: team={} label='{}' seq={} | scaled {} ({}, x{:.3}{}) | shadow {} ({}{})",
        m.name, forge_team_name(team_i), m.label, m.spawn_seq,
        if m.scaled_on() { "ON" } else { "off" }, how(m.flags.scaled), m.scale(conv),
        if !g.scaled && fs != m.scaled_on() { ", global OFF" } else { "" },
        if m.shadow_on() { "ON" } else { "off" }, how(m.flags.shadow),
        if !g.shadowcasters && fc != m.shadow_on() { ", global OFF" } else { "" },
    )
}

/// Human name for a forge team index (-1 none, 8 neutral, 0..7 the change-colour names).
fn forge_team_name(idx: i32) -> String {
    match idx {
        -1 => "none".into(),
        8 => "neutral".into(),
        n if (0..8).contains(&n) => scene::FORGE_COLOR_NAMES[n as usize].into(),
        n => format!("{n}"),
    }
}

/// Import external geometry (OBJ/GLB — converted from another game's BSP by the user)
/// as a renderable BASE mesh to forge over. Computes per-vertex normals and converts
/// the source Y-up convention to the viewer's Halo Z-up. Returns a lit static mesh
/// (untextured → default white × lighting). This is the core of the "import geometry
/// and forge on top" feature; the UI/file-picker + a scan folder wrap this.
fn import_geometry_mesh(
    mesh: &hms_render::MeshRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
) -> Option<(hms_render::GpuMesh, Vec<[glam::Vec3; 3]>)> {
    let om = match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("glb") | Some("gltf") => voxel::parse_glb(&std::fs::read(path).ok()?)?,
        _ => voxel::parse_obj(&std::fs::read_to_string(path).ok()?),
    };
    if om.verts.is_empty() || om.tris.is_empty() {
        return None;
    }
    // Y-up (OBJ/GLB) → Halo Z-up: (x, y, z)_src → (x, -z, y).
    let cvt = |v: [f32; 3]| glam::Vec3::new(v[0], -v[2], v[1]);
    let pos: Vec<glam::Vec3> = om.verts.iter().map(|&v| cvt(v)).collect();
    // Per-vertex normals = area-weighted average of adjacent face normals.
    let mut nrm = vec![glam::Vec3::ZERO; pos.len()];
    for t in &om.tris {
        let (a, b, c) = (pos[t[0]], pos[t[1]], pos[t[2]]);
        let fn_ = (b - a).cross(c - a);
        nrm[t[0]] += fn_; nrm[t[1]] += fn_; nrm[t[2]] += fn_;
    }
    let verts: Vec<hms_render::MeshVertex> = pos.iter().enumerate().map(|(i, p)| {
        let n = if nrm[i].length_squared() > 1e-12 { nrm[i].normalize() } else { glam::Vec3::Z };
        hms_render::MeshVertex { pos: [p.x, p.y, p.z], normal: [n.x, n.y, n.z], uv: [0.0, 0.0], color: [1.0; 4], sway: [0.0; 3], uv2: [0.0, 0.0], tangent: [0.0; 4] }
    }).collect();
    let indices: Vec<u32> = om.tris.iter().flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();
    // World-space triangle soup for snap-to-surface raycasting.
    let tris: Vec<[glam::Vec3; 3]> = om.tris.iter().map(|t| [pos[t[0]], pos[t[1]], pos[t[2]]]).collect();
    let gm = mesh.upload_mesh(device, queue, &verts, &indices, &[glam::Mat4::IDENTITY], None, None, None, None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.0, 1.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None);
    Some((gm, tris))
}

/// Möller-Trumbore ray-vs-triangle-soup, returns the nearest hit distance along `dir`.
fn raycast_tris(origin: glam::Vec3, dir: glam::Vec3, tris: &[[glam::Vec3; 3]]) -> Option<f32> {
    let mut best = f32::MAX;
    for t in tris {
        let e1 = t[1] - t[0];
        let e2 = t[2] - t[0];
        let p = dir.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-7 { continue; }
        let inv = 1.0 / det;
        let s = origin - t[0];
        let u = s.dot(p) * inv;
        if u < 0.0 || u > 1.0 { continue; }
        let q = s.cross(e1);
        let v = dir.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 { continue; }
        let dist = e2.dot(q) * inv;
        if dist > 1e-4 && dist < best { best = dist; }
    }
    (best < f32::MAX).then_some(best)
}

/// A full object pose (position + forward/up basis) for the undo history.
#[derive(Clone, Copy)]
struct Pose {
    pos: [f32; 3],
    fwd: [f32; 3],
    up: [f32; 3],
}

/// One reversible pose edit (move or rotate).
#[derive(Clone, Copy)]
struct PoseEdit {
    datum: u32,
    before: Pose,
    after: Pose,
}

/// Blender-style modal transform. Grab (move) or Rotate, with optional axis lock,
/// global/local space, and Ctrl "magnet" snapping. Started by G/R/Shift+D, driven by the
/// mouse each frame relative to `start_pointer`, confirmed by click/Enter, cancelled by Esc.
#[derive(Clone, Copy, PartialEq)]
enum XformKind {
    Grab,
    Rotate,
}

/// What a cursor ray grabs on the gizmo — a move axis handle or a rotate ring.
#[derive(Clone, Copy, PartialEq)]
enum GizmoHit {
    Move(usize),   // 0=X, 1=Y, 2=Z
    Rotate(usize), // rotate ring about axis 0/1/2
}

/// Blender-style tool mode selected from the toolbar — drives which gizmo shows and what a
/// gizmo drag does. Keyboard G/R still start a modal op regardless of mode.
#[derive(Clone, Copy, PartialEq)]
enum ToolMode {
    Select,
    Move,
    Rotate,
    /// Construction-geometry tool — viewport clicks pick anchor points and draw guides.
    Construct,
}

/// The Construct tool's click behaviour. One click-tool per construction action,
/// so every one of them is usable: pick the op here, then click in the VIEWPORT.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ConstructOp {
    /// Two clicks -> a dotted guide between the two anchors.
    Guide,
    /// One click -> a construction circle centred on the clicked point.
    Circle,
    /// One click -> a construction square centred on the clicked point.
    Square,
    /// Two clicks on FACE anchors -> move the selection so face A is coincident with face B.
    Coincident,
    /// One click -> move the selection's centre onto the clicked point.
    AnchorTo,
    /// One click -> mirror the selection through the clicked point.
    Mirror,
    /// #snap-array: two clicks (or one click + a length) -> fill that line with the current
    /// selection, by count or by step, overlap allowed.
    Line,
}

impl ConstructOp {
    const ALL: [ConstructOp; 7] = [
        ConstructOp::Guide, ConstructOp::Circle, ConstructOp::Square,
        ConstructOp::Coincident, ConstructOp::AnchorTo, ConstructOp::Mirror,
        ConstructOp::Line,
    ];
    fn label(self) -> &'static str {
        match self {
            ConstructOp::Guide => "Guide",
            ConstructOp::Circle => "Circle",
            ConstructOp::Square => "Square",
            ConstructOp::Coincident => "Coincident",
            ConstructOp::AnchorTo => "Anchor",
            ConstructOp::Mirror => "Mirror",
            ConstructOp::Line => "Line array",
        }
    }
    /// One line telling the user exactly what a click will do.
    fn hint(self) -> &'static str {
        match self {
            ConstructOp::Guide => "Click two anchor points to draw a dotted guide between them.",
            ConstructOp::Circle => "Click a point to drop a construction circle centred there.",
            ConstructOp::Square => "Click a point to drop a construction square centred there.",
            ConstructOp::Coincident => "Click FACE A (on the object you are moving), then FACE B. The selection moves so the faces meet.",
            ConstructOp::AnchorTo => "Click a point to move the selection's centre exactly onto it.",
            ConstructOp::Mirror => "Click a point to mirror the selection through it, across the chosen axis.",
            ConstructOp::Line => "Click the START then the END of a line (or one point, in length mode). The selection fills it: by count, or by step with overlap allowed.",
        }
    }
}


/// Undo: a snapshot of the entire OFFLINE editing state (mvar + local objects, their
/// meta/colours, and the selection). Ctrl+Z restores the previous snapshot — this covers
/// moves, rotations, duplication, and every properties-panel data edit in one mechanism.
/// PartialEq so a script run that changed nothing can drop the step it pushed.
#[derive(PartialEq)]
struct EditSnapshot {
    mvar_objects: Vec<hms_ipc::ObjectInfo>,
    local_objects: Vec<hms_ipc::ObjectInfo>,
    mvar_meta: std::collections::HashMap<u32, ObjMeta>,
    mvar_colors: std::collections::HashMap<u32, (u8, u8)>,
    selected_set: Vec<u32>,
    selected_datum: Option<u32>,
    /// Construction shapes ride the SAME undo stack as object edits, so Ctrl+Z
    /// backs out "I drew a circle in the wrong place" exactly like any other mistake.
    shapes: Vec<construct::Shape>,
    sel_shape: Option<usize>,
}

struct XformOp {
    kind: XformKind,
    /// Locked axis: 0=X, 1=Y, 2=Z; None = free (screen-plane move / view-axis rotate).
    axis: Option<usize>,
    /// Transform space: true = global world axes, false = the lead object's local basis.
    global: bool,
    /// Selection centroid at op start (the transform pivot).
    pivot: glam::Vec3,
    /// Per-object start state: (datum, pos, fwd, up). Restored on cancel.
    start: Vec<(u32, glam::Vec3, glam::Vec3, glam::Vec3)>,
    /// Screen-space pointer position when the op (or the last mode/axis change) began.
    start_pointer: (f32, f32),
    /// The lead object's local basis at start (for local-space transforms).
    lead_fwd: glam::Vec3,
    lead_up: glam::Vec3,
    /// True when this op created duplicate objects (Shift+D) — cancel deletes them.
    duplicated: bool,
    /// True when this op pushed an undo snapshot at start (Shift+D). Confirm keeps it;
    /// cancel pops it.
    undo_pushed: bool,
    /// Cell size (template extent along the locked axis) once array/line-duplicate mode
    /// engages (Shift+Ctrl + axis lock during a duplicate). 0 = not in array mode.
    array_cell: f32,
    /// #snap-array: wheel adjustment ADDED to `array_cell` to get the real array step, so a line
    /// of copies can be spread out (gap) or driven together (overlap) without leaving the drag.
    /// 0 = exact face-to-face, which is what an untouched array does.
    array_adj: f32,
    /// Accumulated wheel delta, converted to whole notches of `array_adj` (raw scroll units are
    /// device-dependent, so they are integrated rather than used directly).
    wheel_accum: f32,
    /// Datums of each EXTRA flush copy-set spawned along the axis (beyond op.start,
    /// which is copy #1). Index j → axis offset cell·(j+2). Removed on cancel.
    array_copies: Vec<Vec<u32>>,
    /// True when started by dragging a gizmo handle/ring — confirms on mouse RELEASE
    /// (press-drag-release) instead of the modal click-to-place.
    drag_mode: bool,
    /// Typed numeric entry (degrees for Rotate, world units for Grab). When non-empty it OVERRIDES
    /// the mouse: e.g. Rotate + "90" ⇒ exactly +90° about the axis, "-90" ⇒ −90°. Enter confirms.
    num_buf: String,
    /// Precision: integrated "effective" pointer. Each frame the raw cursor displacement is
    /// added scaled by PRECISION_SCALE while Shift is held (fine-tune), 1.0 otherwise.
    /// Grab/Rotate read THIS instead of the raw pointer, so holding Shift lets you nudge an
    /// object a hair even when zoomed right in. At scale 1.0 it telescopes to the raw cursor
    /// exactly, so normal (non-precision) movement is unchanged.
    eff_pointer: (f32, f32),
    /// Raw pointer from the previous frame (for the displacement integral).
    last_raw_pointer: (f32, f32),
    /// #snap-array: the moving set's ORIENTED snap faces, captured once at the op's start pose
    /// (its box faces plus, for a single piece, its model's real large planar faces -- a ramp's
    /// sloped deck and its tall end). A grab is a pure translation, so the drag just offsets
    /// these by the current delta instead of re-deriving them (and re-walking the mesh) per frame.
    snap_src: Vec<snap::Face>,
    /// Candidate TARGET faces: the oriented box faces of every other placed object. Those objects
    /// do not move during the op, so this is gathered once too and only distance-filtered per frame.
    snap_tgt: Vec<snap::Face>,
}

/// Full per-object data parsed from a .mvar, keyed by render datum, so the
/// properties panel can show everything the map variant stored for a forge object.
#[derive(Clone, PartialEq, Default)]
struct ObjMeta {
    name: String,
    folder: u16,
    item: u8,
    pos: [f32; 3],
    team: u8,
    color: i32,
    cached_type: u8,
    spawn_seq: i32,
    respawn: u8,
    label_idx: u16,
    /// Resolved forge label string (from the variant's label table), "" when none. When this
    /// is "scale" the object's spawn_seq encodes an X330 visual scale multiplier.
    label: String,
    placement: u8,
    boundary_shape: u8,
    boundary: [u16; 4],
    weapon_clips: u8,
    tele_channel: u8,
    tele_passability: u8,
    location_name: u16,
    /// Spawn-relative-to PARENT slot index (−1 = none).
    spawn_rel: i32,
    /// This object's own .mvar slot (0..650), or 0xFFFF for a not-yet-saved add.
    /// Used so the parent picker can reference a chosen parent by its slot.
    slot: u16,
    /// The SCALED / SHADOW pseudo-flags as user OVERRIDES (`None` = follow the
    /// derived default: SCALED = has the `scale` label; SHADOW = GREEN team + `scale` label).
    /// Never written to the .mvar; persisted in the project file.
    flags: forge_scale::ObjFlags,
    /// The Halo 4-only record fields (see docs/halo4_mvar_layout.md); None for a Reach object.
    h4: Option<Box<h4::edit::H4Fields>>,
}

impl ObjMeta {
    /// Copy across every field the user actually CHANGED in the properties panel.
    ///
    /// A mass edit must be a per-FIELD diff, never a whole-struct copy: the selected objects
    /// share the panel but not their identity. Blanket-assigning `m` would give every object
    /// the primary's `name`/`slot`/`spawn_rel` and teleport them all onto its position.
    /// `pos`, `name`, `slot` and `spawn_rel` are deliberately excluded here -- position is
    /// applied by the caller as a DELTA so a group keeps its layout, and the other three are
    /// per-object identity that a batch edit must never clone.
    fn apply_changed_fields(&mut self, before: &ObjMeta, after: &ObjMeta) {
        if after.folder != before.folder { self.folder = after.folder; }
        if after.item != before.item { self.item = after.item; }
        if after.team != before.team { self.team = after.team; }
        if after.color != before.color { self.color = after.color; }
        if after.cached_type != before.cached_type { self.cached_type = after.cached_type; }
        if after.spawn_seq != before.spawn_seq { self.spawn_seq = after.spawn_seq; }
        if after.respawn != before.respawn { self.respawn = after.respawn; }
        if after.label_idx != before.label_idx { self.label_idx = after.label_idx; }
        if after.label != before.label { self.label = after.label.clone(); }
        if after.placement != before.placement { self.placement = after.placement; }
        if after.boundary_shape != before.boundary_shape { self.boundary_shape = after.boundary_shape; }
        if after.boundary != before.boundary { self.boundary = after.boundary; }
        if after.weapon_clips != before.weapon_clips { self.weapon_clips = after.weapon_clips; }
        if after.tele_channel != before.tele_channel { self.tele_channel = after.tele_channel; }
        if after.tele_passability != before.tele_passability { self.tele_passability = after.tele_passability; }
        if after.location_name != before.location_name { self.location_name = after.location_name; }
        // A flag toggled in the panel goes to the whole selection too.
        if after.flags.scaled != before.flags.scaled { self.flags.scaled = after.flags.scaled; }
        if after.flags.shadow != before.flags.shadow { self.flags.shadow = after.flags.shadow; }
        // the Halo 4 block diffs per field too (h4/edit.rs); a Reach object stays None
        if after.h4 != before.h4 { self.h4 = h4::edit::merge_h4_changed(self.h4.take(), before.h4.as_deref(), after.h4.as_deref()); }
    }

    /// Effective SCALED state (override, else "has the scale label").
    fn scaled_on(&self) -> bool {
        self.flags.scaled_on(&self.label)
    }
    /// Effective SHADOW state (override, else the green+scale gametype rule).
    fn shadow_on(&self) -> bool {
        self.flags.shadow_on(self.team, &self.label)
    }

    /// Visual scale multiplier under the given (active) scale convention. Only objects tagged
    /// with the forge "scale" label carry a convention-encoded scale in their spawn_seq;
    /// everything else renders at 1.0 (retail spawn_seq is spawn ORDERING, not scale). The team
    /// byte auto-selects the cosmic branch for X330 (RED-team SCALE objects), matching the engine.
    fn scale(&self, conv: forge_scale::ScaleConvention) -> f32 {
        // A Halo 4 object renders at 1.0 unless the gametype rule (SCALED flag / `scale` label
        // + spawn sequence) is on - the same convention picker as Reach. The record's own scale
        // field is stored but NOT drawn by MCC (docs/halo4_mvar_layout.md §11c).
        if self.h4.is_some() {
            return h4::edit::h4_effective_scale(self.scaled_on(), self.spawn_seq, self.team, conv);
        }
        // The SCALED pseudo-flag decides (default = has the label; the user can turn
        // a labelled object's scaling off, or read a label-less object's spawn seq as a size).
        if self.scaled_on() {
            let team = forge_scale::team_from_u8(self.team);
            let max = forge_scale::object_max_scale(self.team).max(conv.max_scale());
            forge_scale::spawn_seq_to_scale(self.spawn_seq, conv, team).clamp(0.01, max)
        } else {
            1.0
        }
    }
}

/// One row in the custom variant file browser (folder or .mvar file) + its column metadata.
struct FileEntry {
    path: std::path::PathBuf,
    name: String,
    is_dir: bool,
    size: u64,
    modified: Option<std::time::SystemTime>,
    /// Halo 4 variants only: "Title - base map, N objects, author" shown after the file
    /// name and matched by the search box. Empty for folders and Reach files.
    hint: String,
}

struct App {
    render_state: egui_wgpu::RenderState,
    renderer: SceneRenderer,
    tex_id: egui::TextureId,
    camera: Camera,
    move_speed: f32,
    /// User bloom scale (1.0 = engine default). Slider-driven; pushed to the renderer's
    /// post.bloom.x. Lets the user dial bloom down when it washes shadows/fog.
    bloom_scale: f32,
    /// Lighting Lab: live sliders for every lighting term so the user can isolate the weak one.
    ll_manual_exp: bool,   // true -> fixed exposure (auto-exposure OFF)
    ll_exp: f32,           // manual/fixed exposure value (when ll_manual_exp)
    ll_base: f32,          // scenario base exposure gain (post.p.x)
    ll_key: f32,           // auto-exposure target key
    ll_min_ev: f32,        // auto-exposure min EV (gain floor = 2^min_ev)
    ll_max_ev: f32,        // auto-exposure max EV (gain ceil = 2^max_ev)
    ll_meter_cal: f32,     // meter-unit correction
    ll_sun: f32,           // analytical sun multiplier
    ll_ambient: f32,       // ambient/sky multiplier
    ll_lightmap: f32,      // baked lightmap/PVL multiplier
    ll_fog: f32,           // atmosphere fog density (FOG_WU); 0 = default
    ll_seeded: bool,       // seeded slider defaults from the loaded scene's cfxs band
    ae_smoothed: f32,      // temporally-smoothed auto-exposure gain (0 = uninit); avoids flicker
    ae_target: f32,        // last read-back target gain (held between throttled meter reads)
    ae_frame: u32,         // frame counter (the meter readback is non-blocking)
    /// `// #h4-expo-3` The loaded Halo 4 map's `cfxs` adaptation block: the authored exposure
    /// range the Lighting panel seeds its band sliders from, and the delay / blend / max-change
    /// dynamics the interactive adaptation runs (`halo4.dll sub_180359768`). None on Reach.
    h4_cfxs: Option<h4::lighting::H4CameraFx>,
    /// Halo 4 adaptation state (stops, the engine's units): the current adapted stops and the
    /// 120-entry target history the delay window is evaluated over (`sub_180359768`'s ring).
    h4_ae_stops: Option<f32>,
    h4_ae_hist: std::collections::VecDeque<f32>,

    // Render-model preview: a small offscreen SceneRenderer showing the selected
    // palette object's model, drawn as an egui image in the side panel.
    preview_renderer: SceneRenderer,
    preview_tex: egui::TextureId,
    preview_for: Option<u32>,  // mode-tag currently uploaded (avoid re-decoding each frame)
    preview_ok: bool,          // false when the selected object has no render_model
    preview_center: glam::Vec3, // model AABB center (orbit target)
    preview_radius: f32,       // model bounding radius (near/far + fallback)
    preview_min: glam::Vec3,   // model AABB (for a tight per-orientation frame fit)
    preview_max: glam::Vec3,
    /// User-driven orbit around the previewed model. yaw/pitch in radians (drag to
    /// rotate), zoom is a multiplier on the fit distance (scroll to zoom in/out).
    preview_yaw: f32,
    preview_pitch: f32,
    preview_zoom: f32,
    /// (preview key, yaw, pitch, zoom) of the last preview render + a frame counter, so the
    /// preview only re-renders (and requests a repaint) when something about it changed.
    preview_last: Option<(u32, f32, f32, f32)>,
    preview_frame: u32,

    // Live parsing/scene (viewer loads the DLL for parsing; scene reads the
    // object-table snapshot each tick).
    scene_ctl: Option<SceneController>,
    /// While a background load runs, the SceneController is owned by the worker;
    /// `scene_ctl` is None and these carry the mesh stream + thread handle.
    load_rx: Option<std::sync::mpsc::Receiver<scene::LoadMsg>>,
    load_handle: Option<std::thread::JoinHandle<()>>,
    /// After a load settles, trim the working set once. The parallel decode leaves
    /// ~3GB of freed-but-retained native heap resident; the renderer never touches it again,
    /// so releasing it drops steady memory from ~4.5GB to the ~1GB render set. Set to a future
    /// instant when a load completes; fired (and cleared) when elapsed.
    trim_at: Option<std::time::Instant>,
    /// Throttle for the DURING-load native-heap trim — returns freed transient decode buffers
    /// to the OS periodically instead of only once post-load, so the working-set/commit peak during
    /// a heavy load (Forge World) stays lower on lower-RAM machines.
    next_load_trim: Option<std::time::Instant>,
    /// Interactive load-freeze instrumentation: timestamp of the previous frame while a
    /// load is in flight. update() logs any frame that took longer than a threshold (a
    /// UI "hitch"/freeze) to `hms_load.log` next to the exe, tagged with the current load
    /// status — so the ACTUAL interactive freeze (invisible to headless) can be located.
    last_frame_at: Option<std::time::Instant>,
    /// HMS_FRAMEPROF=1 print-only frame profiler -- [frame start, stage laps] for the
    /// current update() plus a rolling 300-frame summary (see the end of `update`).
    frame_prof: Option<(std::time::Instant, [f32; 6], Vec<[f32; 7]>)>,
    /// Visible on-screen Forge-object diagnostic (shown in the top status bar) so the
    /// in-memory forge-load pipeline can be diagnosed WITHOUT a console — reports which link
    /// broke (table connected? palette loaded? placements read? map match? placed?).
    forge_diag: String,
    /// Set to true to tell the in-flight load worker to bail out early so a newly
    /// selected map can start loading immediately instead of waiting for the old one.
    load_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The global "Scaled objects" / "Shadow casters (Forge)" switches (persisted).
    obj_globals: forge_scale::GlobalFlags,
    /// Per-object flag overrides read from a project, waiting for its variant to
    /// render (the meta table is rebuilt from the .mvar, then these are re-applied by slot/datum).
    pending_obj_flags: Option<Vec<project::ProjObjFlags>>,
    objtable: Option<ObjectTableClient>,
    forge_table: Option<ForgeObjectTableClient>,
    transform_queue: Option<TransformQueueClient>,
    last_objects: Vec<ObjectInfo>,
    /// The user's "Forge special FX" toggle (View menu > Forge extras; script
    /// `screenfx on|off`). Persisted in the settings dir, default ON. Off = the map's own default
    /// screen effect only, so screenshots can be taken without the placed FX orbs' colour grade.
    forge_fx_enabled: bool,
    /// Per-map decode cache (obje -> sefc, sefc -> elements) + the current active set.
    screenfx_cache: screenfx::ScreenFxCache,
    active_screenfx: screenfx::ActiveScreenFx,
    /// The (enabled, default tag, forge tags) last uploaded, so the post uniform is only
    /// rewritten when the set actually changes (add/delete/duplicate/toggle), not every frame.
    screenfx_pushed: Option<(bool, u32, Vec<u32>)>,
    map_path: String,
    map_status: String,
    /// Wall-clock start of the current chunked load (for the "loaded in Xs" msg).
    load_t0: std::time::Instant,
    /// True from the `Done` handler until the POST-load object-model rebuild has fully settled.
    /// Keeps the frame loop pumping (request_repaint) through the object tail so the window stays
    /// responsive the whole time, and defers the "Ready in Xs" stamp until the map is ACTUALLY
    /// finished (the "Loaded" number at Done does not include the tail).
    settle_pending: bool,
    /// Last time the object rebuild produced meshes; after 30 s without one (nobody is
    /// moving/placing objects) the burst memory the last edit pulled in (native page cache, map
    /// pages, allocator slack) is released once (`idle_released`) until the next rebuild.
    last_rebuild_at: std::time::Instant,
    idle_released: bool,
    /// Headless auto-screenshot: when HMS_AUTOSHOT=<png> is set, frame the map
    /// after load, render a few frames, capture to this path, and exit. Dev aid
    /// for iterating on rendering without a human in the loop.
    autoshot: Option<std::path::PathBuf>,
    autoshot_frames: i32,
    /// Auto-detected maps (no file-path entry). Populated at startup + Refresh.
    map_candidates: Vec<mapcat::MapCandidate>,
    selected_map: Option<usize>,
    /// Map-selector category: false = built-in (the game's own folder; the default, common
    /// case), true = modded (Steam Workshop). Keeps the two groups on SEPARATE pick-lists so
    /// modded maps don't clutter the built-in view.
    show_modded_maps: bool,
    /// #map-picker: which GAME the map pick-list is showing. The picker is two levels -- pick the
    /// title, then Built-in / Modded within it -- so every title gets the same split (the old
    /// flat "Halo 4" tab lumped stock and Workshop Halo 4 caches together and left no room for a
    /// third title).
    picker_game: mapcat::Game,
    /// In-flight Halo 4 load (worker thread + mesh channel), see h4/gui.rs.
    h4_load: Option<h4::gui::H4Load>,
    /// A Halo 4 map is the loaded map (the Reach cache is closed, so `has_cache()` gates the
    /// Reach-only paths off and `objscene()` dispatches to `h4_scene`).
    h4_active: bool,
    /// A Halo 4 map variant (.mvar chunk v50) queued for the next Halo 4 map load
    /// (h4/gui.rs `h4_import_mvar` -> `h4_load_map`).
    h4_pending_mvar: Option<std::path::PathBuf>,
    /// The Halo 4 editor scene (h4/edit_scene.rs) once a Halo 4 map finished loading;
    /// `objscene()` dispatches the editor verbs here while `h4_active`.
    h4_scene: Option<Box<h4::edit_scene::H4ObjectScene>>,
    /// The Halo 4 editor's own state (palette rows, source records, save gate).
    h4_edit: h4_app::H4EditState,
    /// The loaded Halo 4 map's spawn-family scenario placements (marker lane count).
    h4_map_spawn_markers: usize,

    selected_datum: Option<u32>,
    /// Multi-select set (shift+click). `selected_datum` is the primary/last-clicked
    /// member (drives the properties panel + arrow-key edit); this holds all selected.
    selected_set: Vec<u32>,
    /// Active modal transform (grab/rotate), or None when idle.
    xform: Option<XformOp>,
    /// Gizmo element under the cursor this frame (for hover highlight), if any.
    gizmo_hover: Option<GizmoHit>,
    /// The gizmo triangle list uploaded last frame; `refresh_gizmo` runs every frame and only
    /// re-uploads when the list changed.
    gizmo_last_tris: Vec<([f32; 3], [f32; 3])>,
    /// Toolbar tool mode (select / move / rotate).
    tool_mode: ToolMode,
    /// Box-select drag: the screen-space anchor where a rubber-band select started.
    box_select_start: Option<egui::Pos2>,
    /// Shift+A context menu: screen position to open it at (cursor), consumed when shown.
    ctx_menu_pos: Option<egui::Pos2>,
    /// Swallow the next viewport click after a transform confirm/cancel — the confirming
    /// press's trailing release fires a `clicked` the following frame, which would otherwise
    /// pick a background object.
    suppress_next_click: bool,
    /// True while RMB mouse-look is active (started over the viewport); the OS cursor
    /// is hidden and captured for the whole fly, even if the pointer would leave the viewport.
    flying: bool,
    /// The CursorGrab mode currently requested from the OS (None/Confined/Locked).
    cursor_grab: u8,
    /// Datum counter for Shift+D duplicates of .mvar objects (0xD8xxxxxx, distinct from the
    /// 0xD0xxxxxx originals and 0xF0xxxxxx locals).
    next_dup_datum: u32,
    /// Snapshots of the offline editing state. Covers moves, rotations,
    /// duplication, and every properties-panel data edit. Ctrl+Z pops undo → redo; Ctrl+Y
    /// (or Ctrl+Shift+Z) the reverse.
    edit_undo: Vec<EditSnapshot>,
    edit_redo: Vec<EditSnapshot>,
    /// Coalesces a run of properties-panel edits to one object into a single undo entry:
    /// the datum a snapshot was already taken for. Reset on selection change / other actions.
    prop_snapshotted_for: Option<u32>,
    /// Overlays (boundary zones etc.) depend on the current selection, so a selection change
    /// must refresh them. apply_selection_highlight sets this; update() rebuilds once/frame
    /// (a flag avoids recursion — rebuild_overlays itself must never set it).
    overlays_dirty: bool,
    /// The selection highlight (per-object mesh wireframe) only needs REBUILDING when a selected
    /// object actually MOVES — not every frame (rebuilding every selected object's wireframe +
    /// the GPU buffer each frame makes Ctrl+A drop the framerate hard). set_movable_pose sets
    /// this; update()'s per-frame follow rebuilds once then clears it. Selection-change sites call
    /// apply_selection_highlight directly (immediate), so an idle large selection costs nothing.
    highlight_dirty: bool,
    /// The team/colour dropdown entry currently PREVIEWED on the selection, with the
    /// transient `datum -> (team, colour byte)` overrides merged over `mvar_colors` in tick_scene
    /// (never into ObjMeta / undo). Cleared when the pointer leaves the popup / it closes (the
    /// panel reconciles every frame it runs), on deselect / map load / undo-redo.
    hover_preview: Option<(color_hover::HoverEntry, std::collections::HashMap<u32, (u8, u8)>)>,
    /// Which open popup the pointer is inside + the last hovered entry (the pure
    /// state machine; the entry survives the gaps between rows so nothing flickers).
    hover_state: color_hover::PopupState,
    /// The wireframe is hidden -- the pointer is inside a popup in which at least one
    /// entry would change the selection's effective colour. Re-evaluated when `hover_state.inside`
    /// changes (not per frame) and mirrored into SceneRenderer::set_highlight_hidden.
    hover_hidden: bool,
    /// `preview where` debug readout -- this frame's combo button / open popup rects
    /// (egui points, window top-left origin) so a scripted pointer can be aimed at them.
    hover_debug_rects: HoverDebugRects,
    /// Synthetic pointer injected into egui's raw input at the start of update() for
    /// the next N frames (`preview move x y [frames]`) -- drives the REAL hit-testing / hover path
    /// without a physical mouse. `sim_click` = a press on one frame and a release on the next.
    sim_pointer: Option<(egui::Pos2, u32)>,
    sim_click: Option<(egui::Pos2, u8)>,
    /// #construct-h4: scripted keyboard input (`preview key` / `preview type`): one batch of
    /// egui events per frame, so a press and its release land on different frames exactly as a
    /// real keyboard delivers them.
    sim_events: Vec<Vec<egui::Event>>,
    /// #construct-h4: a scripted press-drag (`preview drag`): start, end and the frame counter.
    /// Frame 0 presses at the start, 1..=4 move toward the end, 5 releases there -- enough
    /// motion for egui to call it a drag, which is what the press-drag tools react to.
    sim_drag: Option<(egui::Pos2, egui::Pos2, u8)>,
    /// The pointer position egui saw this frame (for `preview where`).
    dbg_pointer: Option<egui::Pos2>,
    /// #construct-h4: the viewport's own rect this frame, so `preview where` can report the
    /// window coordinates a scripted click must use to land in the 3D view.
    dbg_viewport: Option<egui::Rect>,
    /// A script `preview team|color ...` PINS the preview (the panel's per-frame
    /// reconciliation leaves it alone) until `preview off`; the print-only test hook for the hover.
    hover_preview_pinned: bool,
    /// Script panel: multi-line command buffer + last run's output log.
    script_text: String,
    script_output: String,
    show_script: bool,
    /// Command server: external clients (the hms-mcp bridge, scripting clients) push script text
    /// over TCP; the UI thread drains this each frame and replies with the executor output.
    cmd_rx: Option<std::sync::mpsc::Receiver<cmd_server::CmdRequest>>,

    dll_path: PathBuf,
    status: String,

    // Forge-palette spawn IPC (opened lazily; the injected DLL connects to the
    // same named MMF). Places objects through the game's forge system.
    world_spawn: Option<WorldSpawnClient>,
    spawn_status: String,
    /// Themed variant open dialog. Because .mvar files are named by random UUID-like ids, this
    /// browser lists each file with its parsed TITLE + DESCRIPTION so users can find variants by name.
    variant_browser_open: bool,
    /// Deferred .mvar open. Clicking Open closes the browser *first* and lets a
    /// frame paint without it; only then does the (blocking) import run (loading
    /// inline would leave the dialog on screen for the whole load).
    pending_open_variant: Option<(std::path::PathBuf, u8)>,
    variant_browser_dir: Option<std::path::PathBuf>,
    /// Entries in the current directory — subfolders + .mvar files, with metadata for the columns.
    variant_browser_listing: Vec<FileEntry>,
    variant_browser_selected: Option<std::path::PathBuf>,
    /// Parsed (title, description, author, editor, extra) of the selected file — previewed before
    /// opening. `extra` = the Halo 4 base map + object count line, empty for Reach.
    variant_browser_selected_meta: Option<(String, String, String, String, String)>,
    variant_browser_filter: String,
    /// Editable path field (type/paste a path + Enter to go there).
    variant_browser_path_edit: String,
    /// Back/forward navigation history + cursor into it.
    variant_browser_history: Vec<std::path::PathBuf>,
    variant_browser_hist_pos: usize,
    /// User bookmarks (persisted) + recently-visited folders (session).
    variant_browser_bookmarks: Vec<std::path::PathBuf>,
    variant_browser_recent: Vec<std::path::PathBuf>,
    /// Drive roots (C:\, E:\ …), computed once on open (per-frame drive stat can hang on empty removable drives).
    variant_browser_drives: Vec<std::path::PathBuf>,
    /// Labelled folders that hold .mvar files (game + per-account MCC saves).
    variant_browser_quick: Vec<(String, std::path::PathBuf)>,
    /// Sort column (0=Name, 1=Date modified, 2=Size) + ascending. Folders always sort before files.
    variant_browser_sort_col: u8,
    variant_browser_sort_asc: bool,
    /// Bottom filename field (the selected file's name; in save mode, the target file name).
    variant_browser_filename: String,
    /// #dialogs: the browser doubles as the Save-As picker. false = OPEN mode (pick a file to
    /// load), true = SAVE mode (type/pick a name, press Save). Same window, sidebar and file list.
    variant_browser_save: bool,
    /// #dialogs: a pending overwrite confirmation in save mode — Some(target) shows the inline
    /// "Overwrite <name>? [Overwrite] [Cancel]" strip before an existing file is clobbered.
    variant_browser_overwrite: Option<std::path::PathBuf>,
    /// #dialogs: test-only forced window size. None = normal resizable window; Some([w,h]) pins
    /// the browser to that exact size so a scripted regression run can screenshot the reflow at
    /// min / default / large without a mouse. Never set in normal use.
    variant_browser_force_size: Option<[f32; 2]>,
    /// #dialogs: pending egui-surface screenshot (the whole frame, dialogs included). Set by the
    /// `dialog shot <path>` test hook; serviced in `update` via `ViewportCommand::Screenshot`.
    egui_shot_request: Option<std::path::PathBuf>,
    egui_shot_pending: Option<std::path::PathBuf>,
    /// Spawn-WITH-POSE + live-networked spawn. METHOD_DEFAULT places via
    /// object_placement_data_new/object_new/setpose; METHOD_LIVE adds the networking
    /// authority handover for a live session.
    forge_spawn: Option<hms_ipc::ForgeSpawnClient>,
    /// Full forge-object property editor (team/color/pose/delete) via MMF.
    forge_edit: Option<hms_ipc::ForgeObjectEditClient>,
    /// Cached forge rows (forge_idx, datum, team, color) for the props panel.
    forge_rows: Vec<(u32, u32, u8, u8)>,
    /// Properties-panel edit buffers, synced to `props_for` when selection changes.
    props_for: Option<u32>,
    /// The selected object's material-inspector triple, resolved ONCE per selection change
    /// (native open_model + shader walk on the UI thread — too slow to redo every frame).
    props_mat: Option<(u32, u32, u32)>,
    edit_team: u8,
    edit_color: u8,
    /// Live (injected) scale slider value.
    #[cfg(feature = "injection")]
    edit_scale: f32,
    /// The map's ACTIVE scale convention — drives how SCALE objects render (their spawn_seq is
    /// decoded under this) and is the SOURCE for "Convert to X330". Default X330 (what most
    /// modern gametypes author); switch it when a map was made in 33X/47X/71X. Cosmic (Red team)
    /// is automatic per object, never a UI toggle. See [`forge_scale::ScaleConvention`].
    sc_convention: forge_scale::ScaleConvention,
    /// Tools ▸ Scale converter modal visibility.
    show_scale_converter: bool,
    /// The save-time "objects outside playable space" window — the save action it is
    /// holding (Save / Save As) plus the offending rows (datum, bsp index). None = closed.
    bsp_warn_pending: Option<(BspWarnAction, Vec<(u32, usize)>)>,
    /// A failed variant save shows a modal window (the status line alone is easy to miss:
    /// a save into a read-only mount would otherwise look like it succeeded).
    save_error: Option<String>,
    /// Set by "Save anyway" so the re-issued save skips the gate exactly once.
    bsp_warn_skip_once: bool,
    show_triggers: bool,
    /// #wire-visible  Draw the SELECTION wireframe through other objects (depth test ALWAYS).
    /// Off by default, persisted (`wire_xray.txt`), script `wirexray`.
    wire_xray: bool,
    /// Draw the structure-design soft ceilings (kill floor / acceleration /
    /// slip planes). Off by default, persisted (`soft_ceilings.txt`), script `softceilings`.
    show_soft_ceilings: bool,
    /// Draw the playable structure BSPs' world bounds + floor plane. Off by
    /// default, persisted (`hard_floor.txt`), script `hardfloor`.
    show_hard_floor: bool,
    /// The smaller playable-BSP world-bounds boxes (violet). Off, persisted.
    show_playable_bounds: bool,
    /// Draw forge object boundary shapes (sphere/cylinder/box) — teleporter/objective
    /// zones etc. On by default so shapes are visible while editing them.
    show_boundaries: bool,
    /// Draw the hull overlay (orange volume + outline) of forge-placed
    /// hidden blocks. View > "Show hidden-block physics hulls", persisted `physics_outlines.txt`,
    /// DEFAULT ON (that is how the invisible blocks a variant placed are seen and edited). Only ever
    /// built for variant/user-placed objects -- never scenario objects or BSP.
    show_blockers: bool,
    /// "Path-traced lighting" toggle state + a deferred-apply flag (handled outside the
    /// egui closure to avoid borrow conflicts — bake + BSP reload run in `apply_pathtrace_toggle`).
    show_pathtraced: bool,
    /// Real-time probe GI mode (Lighting Lab); `rtgi_pending` defers the scene build out of the egui closure
    show_rtgi: bool,
    rtgi_pending: bool,
    rtgi_gain: f32,
    /// Draw the scene's lights in the viewport (off by default, like trigger volumes)
    show_lights: bool,
    /// Sun yaw/pitch in degrees and an intensity scale — real-time mode only. `sun_edited`
    /// stays false until the user touches them, so the map's own sun is used verbatim.
    sun_yaw: f32,
    sun_pitch: f32,
    sun_scale: f32,
    sun_edited: bool,
    /// The map's own sun irradiance, captured when the tracer scene is built
    sun_base: Option<[f32; 3]>,
    /// Index into SceneController::editable_lights of the light being moved
    sel_light: Option<u32>,
    /// Index into SceneController::BAKE_QUALITIES for the path-traced bake
    bake_quality: usize,
    pathtrace_dirty: bool,
    /// While a path-trace bake runs on a worker thread the SceneController lives on
    /// that thread; these carry it back + surface progress (0..=10000) so the window stays responsive.
    bake_rx: Option<std::sync::mpsc::Receiver<SceneController>>,
    bake_handle: Option<std::thread::JoinHandle<()>>,
    bake_progress: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
    /// Tools ▸ Light bake map — a top-down scan of what a Forge piece resting on the
    /// ground would be shaded with (scene::light_bake_scan) + a colour finder that moves the selected
    /// object to a matching spot. The scan runs on a worker thread that borrows the SceneController
    /// (same hand-off as the path-trace bake).
    show_lightbake: bool,
    lb_rx: Option<std::sync::mpsc::Receiver<(SceneController, scene::LightBakeColors)>>,
    lb_handle: Option<std::thread::JoinHandle<()>>,
    lb_progress: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
    /// The enumerated colour set + palette (step 1/2) and its rendered map.
    lb_colors: Option<scene::LightBakeColors>,
    lb_img: Option<scene::LightBakeImage>,
    lb_tex: Option<egui::TextureHandle>,
    lb_dirty: bool,
    lb_radius: f32,
    lb_expo: f32,
    /// true → tonemap with the viewport's live exposure (scene stops × auto gain) so the colours match
    /// what the 3D view shows; false → the enumeration's auto key / the manual slider.
    lb_expo_viewport: bool,
    lb_palette_n: usize,
    /// Light-bake panel: fly the camera to the spot after "Move here".
    lb_follow_cam: bool,
    /// Selected swatch (the only way to pick a shade), its probed placements, probes run.
    lb_sel: Option<usize>,
    lb_places: Vec<scene::LightBakePlacement>,
    lb_probes: usize,
    lb_n: usize,
    lb_status: String,
    show_collision: bool,
    show_physics: bool,
    /// Line tool: two clicks spawn `line_count` copies along the segment.
    line_mode: bool,
    line_start: Option<glam::Vec3>,
    line_count: u32,
    /// Fill tool: click a polygon outline, then grid-fill it with the item.
    fill_mode: bool,
    fill_points: Vec<glam::Vec3>,
    fill_spacing: f32,
    /// Keep the camera at least `cam_standoff` world units clear of any solid surface
    /// (terrain, BSP, forge objects) while flying. OFF by default — it moves the camera, which the
    /// user must opt into.
    cam_standoff_on: bool,
    cam_standoff: f32,
    /// Tools/Settings windows moved out of the left panel so it stays about the map itself.
    show_model_painter: bool,
    show_import_geom: bool,
    show_settings: bool,
    /// Why the last save could not write some objects — no palette entry vs no free
    /// slot. Cells because `save_variant_to` takes `&self`.
    last_save_no_palette: std::cell::Cell<usize>,
    last_save_no_slot: std::cell::Cell<usize>,
    /// Enumeration indices of source-variant objects HMS could NOT resolve to a
    /// model at load, so they are absent from `mvar_objects`. They STILL OCCUPY SLOTS in the file.
    /// Save must leave those slots alone (absence means "could not display", not "user deleted"),
    /// and the object counter must include them or it under-reports how full the variant is.
    mvar_unresolved: std::collections::HashSet<usize>,
    /// The .mmsproj currently open by PATH (None = only the quick slot has been
    /// used). Seeds the Save As / Open dialogs so they start where the user last was.
    current_project_path: Option<std::path::PathBuf>,
    /// The SOURCE record each loaded forge object came from, keyed by its DATUM.
    ///
    /// Finding an object's source record by arithmetic ("datum 0xD0000000+i is slot i") is
    /// only valid until the file changes: after one save the slots have shifted, and the next
    /// save would match every object against a DIFFERENT record and write it out with the
    /// wrong type (walls saved as initial spawns). Keyed by datum it stays correct for the
    /// whole session, however many times the user saves.
    mvar_src: std::collections::HashMap<u32, mvar::PlacedObject>,
    /// Records for objects HMS cannot display. They are not in the scene, so they are carried
    /// through a save verbatim (they still occupy a slot in game).
    mvar_unresolved_objs: Vec<mvar::PlacedObject>,
    /// Construction geometry: dotted guide lines drawn between object anchors, and the
    /// snap targets (endpoints / midpoints / intersections) they generate. Editor-only -- guides
    /// are never written to a .mvar.
    guides: Vec<construct::Guide>,
    /// Draw the guides + their snap markers. On once the user adds a guide.
    show_guides: bool,
    /// Construct tool: viewport clicks pick ANCHORS and draw guides instead of selecting.
    construct_mode: bool,
    /// Which anchor family the construct tool offers (all / corners / edges / faces / centres).
    anchor_mode: construct::AnchorMode,
    /// First anchor of the guide being drawn (rubber-bands to the cursor until the second click).
    construct_pending: Option<construct::Anchor>,
    /// Anchor currently under the cursor (highlighted; what a click would take).
    construct_hover: Option<construct::Anchor>,
    /// Snap a move/gizmo drag so the selection centre lands on the nearest guide point.
    snap_guides: bool,
    /// World-unit radius within which a move snaps to a guide point.
    snap_range: f32,
    /// #snap-array: the Ctrl magnet's reach -- the largest face-to-face gap that still snaps (wu).
    magnet_range: f32,
    /// #snap-array: per-render-model cache of the pieces' real large planar faces (a ramp's
    /// sloped deck / tall end), so a drag never re-walks a decoded mesh.
    snap_cache: snap::PlaneCache,
    /// #snap-array: what the last magnet snap mated, for the status line. Written from
    /// `grab_delta` (which is `&self`, hence the cell) and read back when the drag updates status.
    snap_note: std::cell::RefCell<String>,
    /// #snap-array: the Construct LINE tool's settings (count/step, overlap, align, one-click
    /// length) and the first picked point of the line being drawn.
    line_tool: snap::LineTool,
    line_pending: Option<glam::Vec3>,
    /// Mirror plane normal axis (0=X, 1=Y, 2=Z) for the symmetry action.
    mirror_axis: usize,
    /// What a CLICK in the viewport does while the Construct tool is active.
    /// Everything that needs a point in the world is a click-tool, because a control that
    /// says "hover a point then press this button" is impossible to use -- moving the mouse
    /// to the button drops the hover (construct_hover is cleared the moment the cursor
    /// leaves the viewport).
    construct_op: ConstructOp,
    /// Construction shapes the user DREW. Editable after the fact: select one,
    /// drag it to move, resize it, or array objects around its edge.
    shapes: Vec<construct::Shape>,
    sel_shape: Option<usize>,
    /// An in-progress shape drag: which shape, and whether we are sizing it (just drawn) or
    /// moving it (grabbed an existing one). `grab` is the press point minus the centre, so a
    /// move keeps the shape under the cursor instead of snapping its centre to the pointer.
    shape_drag: Option<(usize, bool, glam::Vec3)>,
    /// A MODAL shape move, driven exactly like an object's G: press G, the shape
    /// follows the cursor, left-click/Enter places it, Esc puts it back. Holds the shape index
    /// and its centre when the move started (so Esc can restore it).
    shape_grab: Option<(usize, glam::Vec3)>,
    /// How many copies to space around a shape's edge, and whether to turn each
    /// copy to follow the outline.
    array_count: u32,
    array_rotate: bool,
    /// The face picked as "A" for a coincident constraint — (centre, outward
    /// normal, label). Captured from the hovered face anchor so the pick survives while the
    /// user moves the mouse to the target face.
    cad_face_a: Option<(glam::Vec3, glam::Vec3, String)>,
    /// Coincident mode: false = make the faces coplanar only (slide along the target normal,
    /// keeping position within the plane); true = also centre face A on face B.
    cad_mate_centered: bool,
    /// Turn the part so its face looks INTO the target face. On by default —
    /// a mate that only slides fails on anything not already square to the target.
    cad_mate_rotate: bool,
    /// Construction circle/square parameters.
    cad_shape_seg: u32,
    /// Plane the shape is drawn in: 0/1/2 = normal along X/Y/Z, 3 = the hovered face's normal.
    cad_shape_plane: usize,
    cad_shape_rot: f32,
    /// Mirror leaves the original in place and adds a mirrored copy.
    mirror_copy: bool,
    /// Distance used by "place selection along guide".
    guide_dist: f32,
    /// Guide selected in the list (target of the distance/mirror actions).
    sel_guide: Option<usize>,
    /// Model Painter: voxelize an .obj and spawn a block per surface voxel.
    model_candidates: Vec<std::path::PathBuf>,
    selected_model: Option<usize>,
    paint_res: u32,
    paint_scale: f32,
    /// External geometry import (OBJ/GLB) selection — base to forge over.
    selected_import: Option<usize>,
    /// World-space triangles of imported geometry, for snap-to-surface raycasts.
    import_tris: Vec<[glam::Vec3; 3]>,
    /// Locally-placed forge objects (standalone editor) — rendered WITHOUT a live game,
    /// merged with live objects in tick_scene. Next local datum id.
    local_objects: Vec<ObjectInfo>,
    next_local_datum: u32,
    /// A map's BAKED designer objects (scenery/vehicles/…) enumerated from the scnr
    /// on load, rendered with no live game. Merged like local_objects.
    scenario_objects: Vec<ObjectInfo>,
    /// Per scenario datum (0xE…) the placement coordinates `ObjectInfo` cannot
    /// carry: (category, palette_index, name_index). Built with `scenario_objects`.
    scenario_ident: std::collections::HashMap<u32, (u16, i16, i16)>,
    /// Memoised (tag path, class) per obj tag so the Object panel's per-frame
    /// identity rows do not hit the native tag table every frame.
    tag_ident_cache: std::cell::RefCell<std::collections::HashMap<u32, (String, String)>>,
    /// The obj tags among `scenario_objects` that are the scenario's OWN
    /// spawn-family markers (respawn_point_invisible, initial/respawn points, respawn zones), and
    /// View > "Show map spawn points" (persisted `map_spawns.txt`, DEFAULT OFF). Off = those
    /// placements are left out of the rendered/pickable set; the bookkeeping is untouched and
    /// the variant's own spawns are never affected (they live in mvar_objects).
    scenario_spawn_tags: std::collections::HashSet<u32>,
    show_map_spawns: bool,
    /// Forge objects loaded OFFLINE from a selected .mvar variant (resolved through the
    /// sandbox palette to render_models + placed at each record's pos/orientation).
    /// Merged into the render like scenario_objects, so a variant renders with no live game.
    mvar_objects: Vec<ObjectInfo>,
    /// Full path of the currently-loaded .mvar variant, retained so File→Save can round-trip
    /// the edited objects back into it (and File→Save As can seed the dialog). None until one loads.
    current_variant_path: Option<std::path::PathBuf>,
    /// The four editable header strings of the open variant (Title / Description / Author=CreatedBy /
    /// Editor=ModifiedBy). Seeded from the parsed variant on load; bound to the "Variant Info" UI.
    variant_title: String,
    variant_description: String,
    variant_author: String,
    variant_editor: String,
    /// Set when the user edits any header string → File→Save rewrites the header (else it's copied
    /// verbatim, keeping a byte-exact round-trip).
    variant_header_dirty: bool,
    /// Every global field of the open variant (both games; read-only display) and
    /// the editable subset (category / maximum budget / world bounds / quota min-max), seeded on
    /// load and diffed against `variant_globals` on save. `variant_quota_names` labels the quota
    /// table rows with the base map's palette entry names.
    variant_globals: Option<mvar::VariantGlobals>,
    global_edits: mvar::GlobalEdits,
    variant_quota_names: Vec<String>,
    /// Bounds-edit consent: the panel only unlocks the world-bounds fields after the warning.
    bounds_edit_unlocked: bool,
    /// Help→Hotkeys window visibility.
    show_hotkeys: bool,
    /// Map BSP load fraction 0..1 (from the load worker's Progress messages) for the status bar.
    load_frac: f32,
    /// Per-object (team, color) parsed from the .mvar for the OFFLINE render path, keyed by
    /// the same synthetic datum as `mvar_objects` (0xD000_0000 + index). Merged into `forge_colors`
    /// in update_objects so forge/team change-colour applies to variant objects.
    mvar_colors: std::collections::HashMap<u32, (u8, u8)>,
    /// Full parsed .mvar object data, keyed by render datum (properties panel).
    mvar_meta: std::collections::HashMap<u32, ObjMeta>,
    /// The current variant's forge LABEL string table (indexed by an object's `label_idx`). Kept
    /// so the properties panel can offer a label dropdown by NAME and show the string id.
    mvar_labels: Vec<String>,
    /// Text buffer for the "Add label" field in the Forge-labels editor (the string typed before
    /// it's appended to `mvar_labels`). Not persisted.
    mvar_new_label: String,
    /// Base-map id (c_map_variant m_map_id) of the currently loaded map, so a .mvar whose
    /// target map differs can trigger a base-map load before its objects are rendered.
    loaded_map_id: Option<u32>,
    /// Template .mvar (same map id, fewest objects) that File ▸ New writes into until the
    /// user picks a file with Save / Save As — lets a brand-new variant be created from scratch.
    new_variant_template: Option<std::path::PathBuf>,
    /// A .mvar queued to render once its (different) base map finishes loading. Set when
    /// the user picks a variant for a map that isn't loaded; consumed in the load handler.
    pending_mvar: Option<std::path::PathBuf>,
    /// A .mvar whose first render placed 0 objects because the scene/forge-palette wasn't
    /// ready yet — retried for a few ticks (path, remaining attempts) so the user doesn't
    /// have to open the same variant twice. Cleared on the first successful render.
    variant_retry: Option<(std::path::PathBuf, u8)>,
    /// The last-opened .mvar (persisted), queued to auto-load once the startup map finishes.
    /// Consumed once in the map-load handler.
    autoload_variant: Option<std::path::PathBuf>,
    /// Static forge palette (obj_tag, name) enumerated from the scnr — lets you browse
    /// + place objects with no live game. Selection index into it.
    static_palette: Vec<(u32, String)>,
    /// Per-entry forge CATEGORY index (palette_index) from the native walk, parallel to
    /// `static_palette`, so the object list groups by real forge category.
    static_palette_cat: Vec<u32>,
    /// The forge category NAME per `static_palette` entry (e.g. "ff_weapons_covenant"),
    /// read from the scnr palette[p].Name string. Drives the object-list group headers.
    static_palette_catname: Vec<String>,
    /// hlmt model-variant Name sid per `static_palette` entry (0 = default). Threaded onto a
    /// spawned object so placing "Warthog, Rocket" renders the rocket turret.
    static_palette_variant: Vec<u32>,
    /// A palette index the user started dragging out of the list / preview. When the cursor next
    /// enters the viewport we SPAWN that object under it and enter `placing`. None when not queued.
    pending_place: Option<usize>,
    /// Datum of an object currently being drag-placed: each frame it snaps to the surface (ground or
    /// forge block) directly under the cursor, so it lands EXACTLY where dropped. Cleared on release.
    placing: Option<u32>,
    selected_static_pal: Option<usize>,
    /// Search filter for the sandbox palette list.
    pal_filter: String,
    /// Objects list (left panel): search text + a cache of (datum, label, search key)
    /// rows rebuilt only when the object fingerprint changes, and the filtered row indices
    /// rebuilt only when the fingerprint OR the search text changes.
    obj_list_filter: String,
    obj_list_rows: Vec<(u32, String, String)>,
    obj_list_fp: u64,
    obj_list_filtered: Vec<usize>,
    obj_list_filter_cached: String,
    /// Row last clicked in the list (Shift+click selects the range from here).
    obj_list_anchor: Option<u32>,
    /// Primary selection the list last scrolled to (a change made in the viewport scrolls the
    /// list to the new row once).
    obj_list_last_sel: Option<u32>,

    // MapInfo MMF — the injected DLL publishes the running scenario; used to
    // auto-select which detected map is loaded in the game.
    map_info: Option<hms_ipc::MapInfoClient>,
    pose_client: Option<hms_ipc::PoseSnapshotClient>,
    follow_player: bool,
    auto_selected: bool,
    booted: bool,
    /// Throttles auto-attach retries until the DLL is injected + publishing.
    #[cfg(feature = "injection")]
    attach_timer: Option<std::time::Instant>,
    /// Cached "MCC is running" (refreshed by the throttled auto-attach, so the
    /// status ladder doesn't enumerate processes every frame).
    #[cfg(feature = "injection")]
    mcc_running: bool,
    undo_stack: Vec<PoseEdit>,
    redo_stack: Vec<PoseEdit>,

    // Master palette browser.
    palette_client: Option<ForgePaletteClient>,
    palette: Vec<PaletteEntry>,
    /// Master-palette row picked for a live spawn (injection build only).
    #[cfg(feature = "injection")]
    selected_palette: Option<usize>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Install a clean system UI font (Segoe UI on Windows) as the primary PROPORTIONAL
        // face. egui's defaults stay as fallbacks (glyphs Segoe lacks). Silently keeps the
        // default font if the file isn't present.
        {
            let mut fonts = egui::FontDefinitions::default();
            for path in ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\SegoeUI.ttf"] {
                if let Ok(bytes) = std::fs::read(path) {
                    fonts.font_data.insert("ui_font".to_owned(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
                    fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "ui_font".to_owned());
                    break;
                }
            }
            cc.egui_ctx.set_fonts(fonts);
        }
        // Readability: the default egui text (esp. every `ui.small(...)` label) is too small
        // to read. Bump the global text-style sizes once so ALL panels/labels scale up.
        cc.egui_ctx.style_mut(|s| {
            use egui::{FontFamily, FontId, TextStyle};
            s.text_styles.insert(TextStyle::Small, FontId::new(12.5, FontFamily::Proportional));
            s.text_styles.insert(TextStyle::Body, FontId::new(15.0, FontFamily::Proportional));
            s.text_styles.insert(TextStyle::Button, FontId::new(15.0, FontFamily::Proportional));
            s.text_styles.insert(TextStyle::Monospace, FontId::new(13.5, FontFamily::Monospace));
            s.text_styles.insert(TextStyle::Heading, FontId::new(20.0, FontFamily::Proportional));
        });
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("eframe must run with the wgpu backend");
        let size = (1024u32, 768u32);
        let mut renderer = SceneRenderer::new(&render_state.device, &render_state.queue, size);
        renderer.show_grid = false; // grid off by default (toggle in the UI)
        // Smoke-test content until a map is loaded.
        renderer.set_demo_scene(&render_state.device, &render_state.queue);
        // HMS_IMPORT=<path.obj|glb>: import external (converted) geometry as a base mesh
        // to forge over (recreate-a-mission workflow). Replaces the demo scene; rendered
        // as a lit static mesh. NOTE: a map load clears static meshes, so this is for the
        // import-only workflow (no HMS_MAP).
        let mut env_import_tris: Vec<[glam::Vec3; 3]> = Vec::new();
        if let Ok(p) = std::env::var("HMS_IMPORT") {
            match import_geometry_mesh(renderer.mesh_renderer(), &render_state.device, &render_state.queue, std::path::Path::new(&p)) {
                Some((m, tris)) => {
                    // Persistent layer (survives map loads) so import + forge-over works.
                    renderer.set_imported_meshes(vec![m]);
                    env_import_tris = tris;
                    log::info!("imported base geometry from {p}");
                }
                None => log::warn!("HMS_IMPORT: failed to load geometry from {p}"),
            }
        }
        let tex_id = render_state.renderer.write().register_native_texture(
            &render_state.device,
            // sRGB view: egui always gamma-encodes the sampled texture (it assumes
            // linear input); handing it the sRGB view makes its sample auto-decode
            // first so it doesn't double-encode our sqrt-encoded color (viewport now
            // matches the PNG capture).
            renderer.color_view_srgb(),
            wgpu::FilterMode::Linear,
        );
        // A small second renderer for the palette object preview. Dark background
        // (no sky/grid) so the model reads clearly — the HDR clear is a near-black slate.
        let mut preview_renderer = SceneRenderer::new(&render_state.device, &render_state.queue, (280, 280));
        preview_renderer.show_sky = false;
        preview_renderer.show_grid = false;
        let preview_tex = render_state.renderer.write().register_native_texture(
            &render_state.device,
            preview_renderer.color_view_srgb(),
            wgpu::FilterMode::Linear,
        );

        // The parsing/hooking DLL is expected next to the exe (or in runtime/).
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_default();
        let dll_path = exe_dir.join(crate::PAYLOAD_DLL);

        // Load the DLL into the viewer for parsing (same role as C# NativeBridge).
        let scene_ctl = match NativeDll::load(&dll_path) {
            Ok(dll) => Some(SceneController::new(dll)),
            Err(e) => {
                log::warn!("native DLL not loaded for parsing: {e}");
                None
            }
        };

        // Auto-detected maps + pre-select the last-loaded one (persisted).
        let map_candidates = mapcat::enumerate();
        // HMS_MAP=<stem|path> dev override: force a specific map (e.g. forge_halo) for
        // headless autoshot testing, independent of whatever map the live game is
        // running (which auto_select would otherwise pick). Matches by stem/name/path
        // substring. When set, we mark auto_selected so the MapInfo live-map override
        // is skipped.
        let forced_map = std::env::var("HMS_MAP").ok().filter(|s| !s.is_empty());
        let selected_map = if let Some(want) = &forced_map {
            let w = want.to_lowercase();
            map_candidates.iter().position(|c| {
                c.stem.to_lowercase() == w
                    || c.scenario_name.to_lowercase() == w
                    || c.path.to_string_lossy().to_lowercase().contains(&w)
            })
        } else {
            mapcat::load_last_map().and_then(|last| {
                map_candidates
                    .iter()
                    .position(|c| c.path.to_string_lossy().eq_ignore_ascii_case(&last))
            })
        };

        Self {
            render_state,
            renderer,
            tex_id,
            preview_renderer,
            preview_tex,
            preview_for: None,
            preview_ok: false,
            preview_center: glam::Vec3::ZERO,
            preview_radius: 1.0,
            preview_min: glam::Vec3::splat(-1.0),
            preview_max: glam::Vec3::splat(1.0),
            preview_yaw: 0.7,     // default 3/4 view; user drags to orbit
            preview_pitch: 0.5,
            preview_zoom: 1.0,
            preview_last: None,
            preview_frame: 0,
            camera: Camera::default(),
            move_speed: 12.0,
            // 1.0 = engine-neutral, and it MUST match the post uniform's own default
            // (post.bloom.x is created at 1.0). This field is only pushed to the GPU when the
            // Lighting Lab panel is first opened, so a 0.15 default meant the slider read 0.15 while
            // the renderer was actually running at 1.0 -- and merely OPENING the panel dropped bloom
            // 6.7x with no user action. The per-map cfxs intensity now rides its own lane
            // (set_bloom_curve), so this stays a pure user control.
            bloom_scale: 1.0,
            ll_manual_exp: false,
            ll_exp: 0.66943294,
            ll_base: 0.66943294,
            ll_key: 0.1,
            ll_min_ev: 0.0,
            ll_max_ev: 2.0,
            ll_meter_cal: 1.0,
            ll_sun: 1.0,
            ll_ambient: 1.0,
            ll_lightmap: 1.0,
            // `// #h4-expo-3` = hms_render::FOG_WU_DEFAULT (the fog shader's own bridge). The
            // panel SEEDS this from the renderer per map; the old 0.0025 default was pushed on
            // the panel's first draw and re-bridged every map's fog density 4x.
            ll_fog: hms_render::FOG_WU_DEFAULT,
            ll_seeded: false,
            ae_smoothed: 0.0,
            ae_target: 0.0,
            ae_frame: 0,
            h4_cfxs: None,
            h4_ae_stops: None,
            h4_ae_hist: std::collections::VecDeque::new(),
            selected_datum: None,
            selected_set: Vec::new(),
            xform: None,
            gizmo_hover: None,
            gizmo_last_tris: Vec::new(),
            tool_mode: ToolMode::Select,
            box_select_start: None,
            ctx_menu_pos: None,
            suppress_next_click: false,
            flying: false,
            cursor_grab: 0,
            next_dup_datum: 0xD800_0000,
            edit_undo: Vec::new(),
            edit_redo: Vec::new(),
            prop_snapshotted_for: None,
            overlays_dirty: false,
            highlight_dirty: false,
            hover_preview: None,
            hover_state: Default::default(),
            hover_hidden: false,
            hover_debug_rects: Default::default(),
            sim_pointer: None,
            sim_click: None,
            sim_drag: None,
            sim_events: Vec::new(),
            dbg_pointer: None,
            dbg_viewport: None,
            hover_preview_pinned: false,
            script_text: script::DEFAULT_SCRIPT.to_string(),
            script_output: String::new(),
            show_script: false,
            cmd_rx: cmd_server::spawn(),
            scene_ctl,
            load_rx: None,
            trim_at: None,
            next_load_trim: None,
            last_frame_at: None,
            frame_prof: std::env::var("HMS_FRAMEPROF").is_ok().then(|| (std::time::Instant::now(), [0.0; 6], Vec::new())),
            forge_diag: String::new(),
            load_handle: None,
            load_cancel: None,
            obj_globals: forge_scale::GlobalFlags::load(),
            pending_obj_flags: None,
            objtable: ObjectTableClient::open().ok(),
            forge_table: ForgeObjectTableClient::open().ok(),
            transform_queue: TransformQueueClient::open().ok(),
            last_objects: Vec::new(),
            forge_fx_enabled: screenfx::load_enabled_setting(),
            screenfx_cache: screenfx::ScreenFxCache::default(),
            active_screenfx: screenfx::ActiveScreenFx::default(),
            screenfx_pushed: None,
            map_path: String::new(),
            map_status: String::new(),
            load_t0: std::time::Instant::now(),
            settle_pending: false,
            last_rebuild_at: std::time::Instant::now(),
            idle_released: false,
            autoshot: std::env::var("HMS_AUTOSHOT").ok().map(std::path::PathBuf::from),
            autoshot_frames: 0,
            map_candidates,
            selected_map,
            show_modded_maps: false,
            picker_game: mapcat::Game::Reach,
            h4_load: None,
            h4_active: false,
            h4_pending_mvar: None,
            h4_scene: None,
            h4_edit: Default::default(),
            h4_map_spawn_markers: 0,
            dll_path,
            // Offline (default) build talks about editing variant files, not injecting MCC.
            status: if cfg!(feature = "injection") { "Ready. Attach MCC, then Inject." } else { "Ready. Open a .mvar variant or map to edit." }.to_string(),
            world_spawn: WorldSpawnClient::open().ok(),
            spawn_status: String::new(),
            variant_browser_open: false,
            pending_open_variant: None,
            variant_browser_dir: None,
            variant_browser_listing: Vec::new(),
            variant_browser_selected: None,
            variant_browser_selected_meta: None,
            variant_browser_filter: String::new(),
            variant_browser_path_edit: String::new(),
            variant_browser_history: Vec::new(),
            variant_browser_hist_pos: 0,
            variant_browser_bookmarks: Vec::new(),
            variant_browser_recent: Vec::new(),
            variant_browser_drives: Vec::new(),
            variant_browser_quick: Vec::new(),
            variant_browser_sort_col: 0,
            variant_browser_sort_asc: true,
            variant_browser_filename: String::new(),
            variant_browser_save: false,
            variant_browser_overwrite: None,
            variant_browser_force_size: None,
            egui_shot_request: None,
            egui_shot_pending: None,
            forge_spawn: hms_ipc::ForgeSpawnClient::open().ok(),
            forge_edit: hms_ipc::ForgeObjectEditClient::open().ok(),
            forge_rows: Vec::new(),
            props_for: None,
            props_mat: None,
            edit_team: 0,
            edit_color: 0,
            #[cfg(feature = "injection")]
            edit_scale: 1.0,
            sc_convention: forge_scale::ScaleConvention::X330,
            show_scale_converter: false,
            bsp_warn_pending: None,
            save_error: None,
            bsp_warn_skip_once: false,
            show_triggers: false,
            wire_xray: wire_xray::load_setting(),
            show_soft_ceilings: soft_ceilings::load_setting(),
            show_hard_floor: hard_floor::load_setting(),
            show_playable_bounds: hard_floor::load_playable_setting(),
            show_boundaries: true,
            show_blockers: physics_outlines::load_setting(), // default ON
            show_pathtraced: false,
            show_rtgi: false,
            rtgi_pending: false,
            rtgi_gain: 1.3,
            show_lights: false,
            sun_yaw: 0.0,
            sun_pitch: 45.0,
            sun_scale: 1.0,
            sun_edited: false,
            sun_base: None,
            sel_light: None,
            bake_quality: 3, // medium
            pathtrace_dirty: false,
            bake_rx: None,
            bake_handle: None,
            bake_progress: None,
            show_lightbake: false,
            lb_rx: None,
            lb_handle: None,
            lb_progress: None,
            lb_colors: None,
            lb_img: None,
            lb_tex: None,
            lb_dirty: false,
            lb_radius: 0.4,
            lb_expo: 1.0,
            lb_expo_viewport: true,
            lb_palette_n: 64,
            lb_follow_cam: true,
            lb_sel: None,
            lb_places: Vec::new(),
            lb_probes: 0,
            lb_n: 10,
            lb_status: String::new(),
            show_collision: false,
            show_physics: false,
            line_mode: false,
            line_start: None,
            line_count: 8,
            fill_mode: false,
            fill_points: Vec::new(),
            fill_spacing: 3.0,
            cam_standoff_on: false,
            cam_standoff: 2.0,
            show_model_painter: false,
            show_import_geom: false,
            show_settings: false,
            last_save_no_palette: Default::default(),
            last_save_no_slot: Default::default(),
            mvar_unresolved: Default::default(),
            current_project_path: None,
            mvar_src: Default::default(),
            mvar_unresolved_objs: Vec::new(),
            guides: Vec::new(),
            show_guides: true,
            construct_mode: false,
            anchor_mode: construct::AnchorMode::All,
            construct_pending: None,
            construct_hover: None,
            snap_guides: true,
            snap_range: 2.0,
            magnet_range: snap::DEFAULT_RANGE,
            snap_cache: snap::PlaneCache::default(),
            snap_note: std::cell::RefCell::new(String::new()),
            line_tool: snap::LineTool::default(),
            line_pending: None,
            mirror_axis: 0,
            construct_op: ConstructOp::Guide,
            shapes: Vec::new(),
            sel_shape: None,
            shape_drag: None,
            shape_grab: None,
            array_count: 8,
            array_rotate: true,
            cad_face_a: None,
            cad_mate_centered: false,
            cad_mate_rotate: true,
            cad_shape_seg: 24,
            cad_shape_plane: 2,
            cad_shape_rot: 0.0,
            mirror_copy: false,
            guide_dist: 4.0,
            sel_guide: None,
            model_candidates: voxel::model_catalog(),
            selected_model: None,
            paint_res: 16,
            paint_scale: 30.0,
            selected_import: None,
            import_tris: env_import_tris,
            local_objects: Vec::new(),
            next_local_datum: 0xF000_0000,
            scenario_objects: Vec::new(),
            scenario_ident: std::collections::HashMap::new(),
            tag_ident_cache: Default::default(),
            scenario_spawn_tags: Default::default(),
            show_map_spawns: map_spawns::load_setting(),
            current_variant_path: None,
            variant_title: String::new(),
            variant_globals: None,
            global_edits: mvar::GlobalEdits::default(),
            variant_quota_names: Vec::new(),
            bounds_edit_unlocked: false,
            variant_description: String::new(),
            variant_author: String::new(),
            variant_editor: String::new(),
            variant_header_dirty: false,
            show_hotkeys: false,
            load_frac: 0.0,
            mvar_objects: Vec::new(),
            mvar_colors: std::collections::HashMap::new(),
            mvar_meta: std::collections::HashMap::new(),
            mvar_labels: Vec::new(),
            mvar_new_label: String::new(),
            loaded_map_id: None,
            new_variant_template: None,
            variant_retry: None,
            autoload_variant: mapcat::load_last_variant().map(std::path::PathBuf::from),
            pending_mvar: None,
            static_palette: Vec::new(),
            static_palette_cat: Vec::new(),
            static_palette_catname: Vec::new(),
            static_palette_variant: Vec::new(),
            pending_place: None,
            placing: None,
            selected_static_pal: None,
            pal_filter: String::new(),
            obj_list_filter: String::new(),
            obj_list_rows: Vec::new(),
            obj_list_fp: 0,
            obj_list_filtered: Vec::new(),
            obj_list_filter_cached: String::new(),
            obj_list_anchor: None,
            obj_list_last_sel: None,
            map_info: hms_ipc::MapInfoClient::open().ok(),
            pose_client: hms_ipc::PoseSnapshotClient::open().ok(),
            follow_player: false,
            // When HMS_MAP forces a map, suppress the live-map auto-select entirely.
            auto_selected: forced_map.is_some(),
            booted: false,
            #[cfg(feature = "injection")]
            attach_timer: None,
            #[cfg(feature = "injection")]
            mcc_running: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            palette_client: ForgePaletteClient::open().ok(),
            palette: Vec::new(),
            #[cfg(feature = "injection")]
            selected_palette: None,
        }
    }


    /// Wipe EVERY renderer pass (including the dynamic cutout/holo/blend sub-passes and the
    /// appended effect-scenery billboards) + object list + selection so nothing from the
    /// previous map/variant lingers when a new one loads. Object LISTS are repopulated by the
    /// map-complete handler + pending variant.
    fn clear_all_scene_objects(&mut self) {
        // Renderer: every pass emptied (Vec::new with the correct element type per setter).
        self.renderer.set_static_meshes(Vec::new());
        self.renderer.set_terrain_meshes(Vec::new());
        self.renderer.set_water_meshes(Vec::new());
        self.renderer.set_water_planes(Vec::new());
        self.renderer.set_alphatest_meshes(Vec::new());
        self.renderer.set_foliage_meshes(Vec::new());
        self.renderer.set_blend_meshes(Vec::new());
        self.renderer.set_additive_meshes(Vec::new());
        self.renderer.set_decal_meshes(Vec::new());
        self.renderer.set_sky_meshes(Vec::new());
        self.renderer.set_dynamic_meshes(Vec::new());
        self.renderer.set_dynamic_cutout_meshes(Vec::new());
        self.renderer.set_dynamic_holo_meshes(Vec::new());
        self.renderer.set_dynamic_holo_solid_meshes(Vec::new());
        self.renderer.set_dynamic_blend_meshes(Vec::new());
        // Object lists + per-object side tables (repopulated on load complete / variant render).
        self.mvar_objects.clear();
        self.local_objects.clear();
        self.scenario_objects.clear();
        self.scenario_ident.clear();
        self.tag_ident_cache.borrow_mut().clear();
        // #snap-array: the model plane cache is keyed by render-model TAG, and a new map reuses
        // tag ids, so it must not survive a load.
        self.snap_cache.clear();
        self.scenario_spawn_tags.clear();
        self.mvar_colors.clear();
        self.mvar_meta.clear();
        self.last_objects.clear();
        // The Halo 4 source records / palette rows belong to the old map (the scene
        // itself is dropped by `h4_reset` on the next load).
        self.h4_edit.src.clear();
        self.h4_edit.unresolved.clear();
        self.h4_edit.save_pending = None;
        // Lighting Lab: re-seed the sliders from the new map's cfxs band on the next panel draw.
        self.ll_seeded = false;
        // `// #h4-expo-3` the Halo 4 adaptation belongs to the map that is going away.
        self.h4_ae_stops = None;
        self.h4_ae_hist.clear();
        // RTGI: the tracer scene belongs to the old map — drop it; the checkbox rebuilds on demand.
        self.show_rtgi = false;
        self.rtgi_pending = false;
        self.renderer.rtgi_clear_scene(&self.render_state.queue);
        self.ae_smoothed = 0.0; // re-init auto-exposure adaptation on map load
        // Selection / active transform — a stale datum from the old map must not carry over.
        // Also wipes the selection wireframe GPU buffer.
        self.clear_selection_state();
    }

    /// Drop the selection entirely -- set, primary datum, modal transform -- AND the
    /// renderer's highlight buffer. `selected_set.clear()` alone would leave the last selection
    /// wireframe on screen: update()'s follow path only rebuilds the highlight for a NON-empty
    /// selection, so the old lines would stay uploaded after a map load / variant swap.
    fn clear_selection_state(&mut self) {
        self.selected_set.clear();
        self.selected_datum = None;
        self.xform = None;
        self.highlight_dirty = false;
        self.prop_snapshotted_for = None;
        self.clear_hover_preview(); // a preview cannot outlive its selection
        self.renderer.set_highlight(&self.render_state.device, None);
        self.overlays_dirty = true;
    }

    /// After an undo/redo restored a snapshot, the selection may name objects that no
    /// longer exist (undo of a placement / duplicate) and the scene's pick list -- which the
    /// selection wireframe is derived from -- still holds the PREVIOUS poses until the next object
    /// rebuild. Prune the missing datums, wipe the highlight now, and flag it dirty so it is
    /// re-derived from the CURRENT transforms once tick_scene has rebuilt the picks.
    fn revalidate_selection_after_restore(&mut self) {
        let (set, primary) = physics_outlines::prune_selection(&self.selected_set, self.selected_datum, |d| self.object_datum_present(d));
        self.selected_set = set;
        self.selected_datum = primary;
        self.clear_hover_preview(); // undo/redo restored the REAL colours; drop the preview
        self.renderer.set_highlight(&self.render_state.device, None);
        self.highlight_dirty = !self.selected_set.is_empty();
        self.overlays_dirty = true;
    }

    /// Is `d` an object the editor currently knows? Offline lists (variant / placed /
    /// scenario) are checked by datum; a LIVE game datum (not one of HMS's synthetic 0xD/0xE/0xF
    /// ranges) cannot be validated offline and is kept.
    fn object_datum_present(&self, d: u32) -> bool {
        if !matches!(d >> 28, 0xD | 0xE | 0xF) {
            return true;
        }
        self.mvar_objects.iter().any(|o| o.datum == d)
            || self.local_objects.iter().any(|o| o.datum == d)
            || self.scenario_objects.iter().any(|o| o.datum == d)
    }

    /// Re-decode/re-upload ONLY the BSP with the current lightmap mode (baked vs shipped),
    /// WITHOUT reopening the map (so `baked_atlas` + pathtraced flag survive). Forge objects/dynamic
    /// meshes are left intact (only BSP-derived categories are cleared + rebuilt). Mirrors load_map's
    /// reload core minus open_cache. No-op while another load is in flight.
    fn reload_bsp_lightmaps(&mut self) {
        if self.scene_ctl.is_none() || self.load_rx.is_some() { return; }
        let (fog, exposure, has_atmosphere, space_sky);
        {
            let scene = self.scene_ctl.as_mut().unwrap();
            scene.begin_bsp_load();
            fog = scene.fog_uniform();
            exposure = scene.exposure();
            has_atmosphere = scene.has_atmosphere();
            space_sky = scene.is_space_sky();
        }
        self.renderer.set_fog(&self.render_state.queue, &fog);
        self.renderer.set_sky_atmosphere(has_atmosphere);
        self.renderer.set_space_sky(space_sky);
        self.renderer.set_exposure(&self.render_state.queue, exposure);
        self.renderer.set_static_meshes(Vec::new());
        self.renderer.set_water_meshes(Vec::new());
        self.renderer.set_terrain_meshes(Vec::new());
        self.renderer.set_alphatest_meshes(Vec::new());
        self.renderer.set_foliage_meshes(Vec::new());
        self.renderer.set_blend_meshes(Vec::new());
        self.renderer.set_additive_meshes(Vec::new());
        self.renderer.set_decal_meshes(Vec::new());
        self.renderer.set_sky_meshes(Vec::new());
        self.load_t0 = std::time::Instant::now();
        self.map_status = "Re-lighting BSP…".into();
        let scene = self.scene_ctl.take().unwrap();
        let mesh = self.renderer.mesh_renderer_arc();
        let device = self.render_state.device.clone();
        let queue = self.render_state.queue.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let handle = std::thread::spawn(move || {
            scene::run_load_worker(scene, mesh, device, queue, tx, worker_cancel);
        });
        self.load_rx = Some(rx);
        self.load_handle = Some(handle);
        self.load_cancel = Some(cancel);
    }

    /// Push the Lighting Lab's sun angle/intensity into the probe tracer AND the analytical
    /// sun (shadow map + the shaders' sun term) so the direct light and the GI agree. Real-time only.
    fn apply_sun_edit(&mut self) {
        let dir = {
            let (y, p) = (self.sun_yaw.to_radians(), self.sun_pitch.to_radians());
            glam::Vec3::new(y.cos() * p.cos(), y.sin() * p.cos(), p.sin()).normalize_or_zero()
        };
        // Scale the MAP's sun irradiance (captured when the tracer scene was built), never the tracer's
        // current value — scaling that would compound on every slider tick.
        let base = *self.sun_base.get_or_insert_with(|| self.renderer.rtgi_sun().1);
        let col = [base[0] * self.sun_scale, base[1] * self.sun_scale, base[2] * self.sun_scale];
        self.renderer.rtgi_set_sun(&self.render_state.queue, dir.into(), col);
        self.renderer.set_sun_dir(dir);
        self.renderer.rtgi_kick();
    }

    /// Apply the "Real-time lighting" checkbox (deferred out of the egui closure). Enabling builds
    /// the tracer scene from the loaded map on first use (a few hundred ms), then the per-frame probe
    /// update runs in the renderer; disabling just flips the shading flag (stock lightmaps return).
    fn apply_rtgi_toggle(&mut self) {
        let q = self.render_state.queue.clone();
        let dev = self.render_state.device.clone();
        if self.show_rtgi {
            let Some(scene) = self.scene_ctl.as_ref() else { self.show_rtgi = false; return; };
            if !self.renderer.rtgi_has_scene() {
                match scene.rtgi_scene_data() {
                    Some(sd) => {
                        self.renderer.set_rtgi_scene(&dev, &q, &sd);
                        self.sun_base = Some(self.renderer.rtgi_sun().1);
                        self.renderer.rtgi_set_gain(&q, self.rtgi_gain);
                        self.renderer.rtgi_warm_up(&dev, &q, 8);
                    }
                    None => { self.map_status = "Real-time lighting: no level geometry to trace.".into(); self.show_rtgi = false; return; }
                }
            }
            self.renderer.set_rtgi_enabled(&q, true);
            if self.sun_edited { self.apply_sun_edit(); } // keep a user-moved sun across scene rebuilds
        } else {
            self.renderer.set_rtgi_enabled(&q, false);
        }
    }

    /// Handle a "Path-traced lighting" checkbox change (deferred out of the egui closure
    /// to avoid borrow conflicts). Enable → GPU bake (~seconds, blocks briefly) + reload with our
    /// atlas; disable → reload with the shipped atlas. Requires the scene loaded + no load in flight.
    fn apply_pathtrace_toggle(&mut self) {
        if self.scene_ctl.is_none() || self.load_rx.is_some() || self.bake_rx.is_some() { return; }
        if self.show_pathtraced {
            // Move the SceneController onto a worker thread for the ~10s GPU bake so
            // the window stays live + a progress bar can tick. It's handed back via `bake_rx`, then
            // `drive_bake` restores it + reloads the BSP with the baked atlas.
            let (qs, qb, _, _) = crate::scene::SceneController::bake_quality(crate::scene::SceneController::BAKE_QUALITIES[self.bake_quality.min(5)]);
            let samples: u32 = std::env::var("HMS_BAKE_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(qs);
            let bounces: u32 = std::env::var("HMS_BAKE_BOUNCES").ok().and_then(|s| s.parse().ok()).unwrap_or(qb);
            let k: f32 = std::env::var("HMS_PATHTRACE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            self.map_status = "Path-tracing lighting…".into();
            let mut scene = self.scene_ctl.take().unwrap();
            let dev = self.render_state.device.clone();
            let q = self.render_state.queue.clone();
            let progress = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
            let prog_worker = progress.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                let baker = hms_render::lightbake_gpu::GpuLightBaker::new(&dev);
                scene.set_pathtraced_bake(&baker, &dev, &q, samples, bounces, k, &prog_worker);
                let _ = tx.send(scene);
            });
            self.bake_rx = Some(rx);
            self.bake_handle = Some(handle);
            self.bake_progress = Some(progress);
        } else {
            self.scene_ctl.as_mut().unwrap().set_pathtraced_enabled(false);
            self.reload_bsp_lightmaps();
        }
    }

    /// Poll the bake worker. When it finishes, take the SceneController back,
    /// join the thread, drop the progress handle, and reload the BSP so meshes pick up the baked
    /// atlas. Runs every frame; no-op when no bake is in flight.
    fn drive_bake(&mut self, ctx: &eframe::egui::Context) {
        if self.bake_rx.is_none() { return; }
        // keep repainting so the progress bar animates while the worker runs
        ctx.request_repaint();
        let recv = self.bake_rx.as_ref().unwrap().try_recv();
        match recv {
            Ok(scene) => {
                self.scene_ctl = Some(scene);
                if let Some(h) = self.bake_handle.take() { let _ = h.join(); }
                self.bake_rx = None;
                self.bake_progress = None;
                self.reload_bsp_lightmaps();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // worker died without sending — abort the bake cleanly
                if let Some(h) = self.bake_handle.take() { let _ = h.join(); }
                self.bake_rx = None;
                self.bake_progress = None;
                self.map_status = "Path-trace bake failed (worker exited).".into();
            }
        }
    }

    // ===================== Tools ▸ Light bake map =====================

    /// Start the full colour enumeration on a worker thread (hands the SceneController over, like the
    /// path-trace bake); the palette is built on the worker too.
    fn start_lightbake_enum(&mut self) {
        if self.scene_ctl.is_none() || self.load_rx.is_some() || self.bake_rx.is_some() || self.lb_rx.is_some() { return; }
        let expo = if self.lb_expo_viewport && self.ae_smoothed > 0.0 {
            self.scene_ctl.as_ref().map(|s| (s.exposure() * self.ae_smoothed).max(1e-4))
        } else if self.lb_colors.is_some() { Some(self.lb_expo.max(1e-4)) } else { None };
        let mut scene = self.scene_ctl.take().unwrap();
        let n = self.lb_palette_n.clamp(8, 256);
        let progress = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let pw = progress.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let colors = scene.light_bake_enumerate(expo, n, Some(&pw));
            let _ = tx.send((scene, colors));
        });
        self.lb_rx = Some(rx);
        self.lb_handle = Some(handle);
        self.lb_progress = Some(progress);
        self.lb_status = "Enumerating baked colours…".into();
    }

    /// Poll the enumeration worker; take the SceneController back and (re)build the map image when done.
    fn drive_lightbake_scan(&mut self, ctx: &egui::Context) {
        if self.lb_rx.is_none() { return; }
        ctx.request_repaint();
        match self.lb_rx.as_ref().unwrap().try_recv() {
            Ok((scene, colors)) => {
                self.scene_ctl = Some(scene);
                if let Some(h) = self.lb_handle.take() { let _ = h.join(); }
                self.lb_rx = None;
                self.lb_progress = None;
                let c = &colors.counts;
                self.lb_status = format!("{} samples from {} tris ({} lightmap texels, {} pvl, {} probe, {} airprobe) -> {} colour buckets -> {} swatches, {:.0} ms",
                    colors.total, colors.tris, c[1], c[2] + c[3], c[4], c[5], colors.bucket_count, colors.swatches.len(), colors.elapsed_ms);
                self.lb_expo = colors.exposure;
                self.lb_colors = Some(colors);
                self.lb_sel = None;
                self.lb_places.clear();
                self.lb_dirty = true;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                if let Some(h) = self.lb_handle.take() { let _ = h.join(); }
                self.lb_rx = None;
                self.lb_progress = None;
                self.lb_status = "Enumeration failed (worker exited) — reload the map.".into();
            }
        }
    }

    /// Re-render the enumerated map (selected swatch highlighted) into the egui texture.
    fn lightbake_rebuild_image(&mut self, ctx: &egui::Context) {
        let Some(colors) = self.lb_colors.as_ref() else { return };
        let img = colors.render_map(0, 25.0, self.lb_sel);
        let ci = egui::ColorImage::from_rgba_unmultiplied([img.w, img.h], &img.rgba);
        self.lb_tex = Some(ctx.load_texture("lightbake-map", ci, egui::TextureOptions::NEAREST));
        self.lb_img = Some(img);
        self.lb_dirty = false;
    }

    /// Move the SELECTED object onto a probed placement (x, y, ground + its lighting radius; rotation
    /// kept) through the same offline pose path the gizmo uses (`set_movable_pose`, undo snapshot,
    /// mvar_meta position → the live render + File ▸ Save). Objects that are not offline-movable
    /// (live-game / scenario placements) get the coordinates copied to the clipboard instead.
    fn lightbake_move_selected_to(&mut self, ctx: &egui::Context, p: scene::LightBakePlacement) {
        let Some(datum) = self.selected_datum else {
            self.lb_status = "No object selected — click one in the viewport first.".into();
            ctx.copy_text(format!("{:.2}, {:.2}, {:.2}", p.x, p.y, p.place_z));
            return;
        };
        let movable = self.movable_datums().contains(&datum);
        let obj = self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == datum).cloned();
        let radius = match (self.scene_ctl.as_ref(), obj.as_ref()) {
            (Some(s), Some(o)) => s.object_light_radius(o),
            _ => self.lb_radius,
        };
        let pos = glam::Vec3::new(p.x, p.y, p.z + radius);
        if movable {
            if let Some((_, fwd, up)) = self.movable_pose(datum) {
                self.push_edit_undo();
                self.set_movable_pose(datum, pos, fwd, up);
                self.rebuild_overlays();
                self.lb_status = format!("Moved 0x{datum:08X} to ({:.2}, {:.2}, {:.2}) — ground {:.2} + radius {:.2}. File > Save to persist.", pos.x, pos.y, pos.z, p.z, radius);
                return;
            }
        }
        ctx.copy_text(format!("{:.2}, {:.2}, {:.2}", pos.x, pos.y, pos.z));
        self.lb_status = format!("0x{datum:08X} is not an offline-editable object; copied ({:.2}, {:.2}, {:.2}) to the clipboard.", pos.x, pos.y, pos.z);
    }

    fn lightbake_map_ui(&mut self, ctx: &egui::Context) {
        if !self.show_lightbake { return; }
        if self.lb_dirty && self.lb_rx.is_none() { self.lightbake_rebuild_image(ctx); }
        let mut win_open = true;
        let mut do_enum = false;
        let mut do_find = false;
        let mut new_sel: Option<Option<usize>> = None;
        let mut move_to: Option<scene::LightBakePlacement> = None;
        let mut goto_place: Option<scene::LightBakePlacement> = None;
        let mut copy_place: Option<scene::LightBakePlacement> = None;
        let scanning = self.lb_rx.is_some();
        let have_scene = self.scene_ctl.is_some();
        let panel_h = (ctx.available_rect().height() - 60.0).max(300.0);
        egui::Window::new("Light bake map")
            .open(&mut win_open)
            .default_width(660.0)
            .default_height(panel_h)
            .max_height(panel_h)
            .resizable(true)
            .collapsible(true)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.label("Every baked light colour a Forge piece can receive on this map (read straight from the lightmaps, PVL and airprobes), quantized into a palette of achievable shades. Pick a swatch -> find where it occurs -> probe those spots -> move the selected object there.");
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("swatches");
                    ui.add(egui::DragValue::new(&mut self.lb_palette_n).range(8..=256));
                    ui.label("probe radius");
                    ui.add(egui::DragValue::new(&mut self.lb_radius).speed(0.05).range(0.05..=5.0).max_decimals(2)).on_hover_text("obje bounding radius of the probe object used when probing placements (0.4 = crate-like); the object sits at ground + radius");
                    ui.label("exposure");
                    ui.checkbox(&mut self.lb_expo_viewport, "match viewport").on_hover_text("Tonemap with the 3D view's live exposure so the shades match what you see (applied on the next enumeration)");
                    let vp = self.lb_expo_viewport && self.ae_smoothed > 0.0 && have_scene;
                    ui.add_enabled_ui(!vp, |ui| { ui.add(egui::Slider::new(&mut self.lb_expo, 0.005..=20.0).logarithmic(true).max_decimals(3)); });
                });
                ui.horizontal(|ui| {
                    if ui.add_enabled(have_scene && !scanning && self.load_rx.is_none() && self.bake_rx.is_none(), egui::Button::new("Enumerate map colours")).clicked() { do_enum = true; }
                    if let Some(p) = &self.lb_progress {
                        let frac = p.load(std::sync::atomic::Ordering::Relaxed) as f32 / 10000.0;
                        ui.add(egui::ProgressBar::new(frac).show_percentage().desired_width(180.0));
                    }
                    if !have_scene && !scanning { ui.small("(load a map first)"); }
                    if let Some(c) = &self.lb_colors { ui.small(format!("exposure {:.3}", c.exposure)); }
                });
                if !self.lb_status.is_empty() { ui.small(&self.lb_status); }
                ui.separator();
                // ---- the map (painted from the enumerated samples) ----
                let mut hover_txt: Option<String> = None;
                if let (Some(tex), Some(img), Some(colors)) = (&self.lb_tex, &self.lb_img, &self.lb_colors) {
                    let avail = ui.available_width().max(200.0);
                    let zoom = (avail / img.w as f32).min(2.0);
                    let size = egui::vec2(img.w as f32 * zoom, img.h as f32 * zoom);
                    let resp = ui.add(egui::Image::new(tex).fit_to_exact_size(size).sense(egui::Sense::click()));
                    let rect = resp.rect;
                    let to_screen = |px: f32, py: f32| egui::pos2(rect.min.x + px * zoom, rect.min.y + py * zoom);
                    let painter = ui.painter_at(rect);
                    let r = &colors.raster;
                    for (rank, p) in self.lb_places.iter().enumerate() {
                        let i = ((p.x - r.x0) / r.step).round();
                        let j = ((p.y - r.y0) / r.step).round();
                        if i < 0.0 || j < 0.0 || i as usize >= r.nx || j as usize >= r.ny { continue; }
                        let (px, py) = img.pixel_of(i as usize, j as usize);
                        let sp = to_screen(px, py);
                        painter.circle_stroke(sp, 5.0, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 230, 60)));
                        painter.text(sp + egui::vec2(6.0, -6.0), egui::Align2::LEFT_BOTTOM, format!("{}", rank + 1), egui::FontId::proportional(11.0), egui::Color32::from_rgb(255, 230, 60));
                    }
                    if let Some(d) = self.selected_datum {
                        if let Some(o) = self.last_objects.iter().find(|o| o.datum == d) {
                            let i = ((o.pos[0] - r.x0) / r.step).round();
                            let j = ((o.pos[1] - r.y0) / r.step).round();
                            if i >= 0.0 && j >= 0.0 && (i as usize) < r.nx && (j as usize) < r.ny {
                                let (px, py) = img.pixel_of(i as usize, j as usize);
                                let sp = to_screen(px, py);
                                painter.line_segment([sp + egui::vec2(-7.0, 0.0), sp + egui::vec2(7.0, 0.0)], egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 90, 90)));
                                painter.line_segment([sp + egui::vec2(0.0, -7.0), sp + egui::vec2(0.0, 7.0)], egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 90, 90)));
                            }
                        }
                    }
                    if let Some(hp) = resp.hover_pos() {
                        let (px, py) = ((hp.x - rect.min.x) / zoom, (hp.y - rect.min.y) / zoom);
                        if let Some((i, j)) = img.cell_at(px, py) {
                            let (wx, wy) = (r.x0 + i as f32 * r.step, r.y0 + j as f32 * r.step);
                            match colors.swatch_at_cell(i, j) {
                                Some((sw, z)) => {
                                    let s = &colors.swatches[sw];
                                    hover_txt = Some(format!("x {:.0}  y {:.0}  z {:.1}   swatch {sw}  rgb ({}, {}, {})", wx, wy, z, s.disp[0], s.disp[1], s.disp[2]));
                                    if resp.clicked() { new_sel = Some(Some(sw)); }
                                }
                                None => hover_txt = Some(format!("x {:.0}  y {:.0}   no baked sample", wx, wy)),
                            }
                        }
                    }
                    ui.small(hover_txt.unwrap_or_else(|| "hover: cell -> swatch · click: select that swatch".into()));
                } else if self.lb_colors.is_none() && !scanning {
                    ui.small("No enumeration yet.");
                }
                ui.separator();
                // ---- palette of achievable colours (the ONLY way to pick a shade) ----
                if let Some(colors) = &self.lb_colors {
                    ui.horizontal(|ui| {
                        ui.strong(format!("Achievable shades ({})", colors.swatches.len()));
                        if self.lb_sel.is_some() && ui.small_button("clear selection").clicked() { new_sel = Some(None); }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                        for (i, sw) in colors.swatches.iter().enumerate() {
                            let (rect, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
                            let col = egui::Color32::from_rgb(sw.disp[0], sw.disp[1], sw.disp[2]);
                            ui.painter().rect_filled(rect, 3.0, col);
                            if self.lb_sel == Some(i) {
                                ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(2.0_f32, egui::Color32::WHITE), egui::StrokeKind::Inside);
                            } else if resp.hovered() {
                                ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(200)), egui::StrokeKind::Inside);
                            }
                            let resp = resp.on_hover_text(format!(
                                "swatch {i}\nrgb ({}, {}, {})  hue {:.0}°\nlinear ({:.4}, {:.4}, {:.4})\n{} samples: {} lightmap texels, {} pvl verts, {} probe tris, {} airprobes\n{} occurrence cells (4 wu)",
                                sw.disp[0], sw.disp[1], sw.disp[2], sw.hue, sw.lin[0], sw.lin[1], sw.lin[2], sw.population, sw.kinds[1], sw.kinds[2] + sw.kinds[3], sw.kinds[4], sw.kinds[5], sw.cells.len()));
                            if resp.clicked() { new_sel = Some(Some(i)); }
                        }
                    });
                    if let Some(si) = self.lb_sel {
                        if let Some(sw) = colors.swatches.get(si) {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                                ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(sw.disp[0], sw.disp[1], sw.disp[2]));
                                ui.label(format!("swatch {si}: rgb ({}, {}, {})  linear ({:.4}, {:.4}, {:.4})  {} samples in {} cells", sw.disp[0], sw.disp[1], sw.disp[2], sw.lin[0], sw.lin[1], sw.lin[2], sw.population, sw.cells.len()));
                            });
                            ui.horizontal(|ui| {
                                ui.label("placements");
                                ui.add(egui::DragValue::new(&mut self.lb_n).range(1..=50));
                                ui.checkbox(&mut self.lb_follow_cam, "Camera follows \"Move here\"").on_hover_text("After moving the object, fly the camera to it so you can see where it went");
                                if ui.add_enabled(have_scene && !scanning, egui::Button::new("Find placements")).on_hover_text("Probe the real object placement (down-ray -> ground + radius -> engine object-lighting sample) only around this swatch's occurrence cells").clicked() { do_find = true; }
                                if self.lb_probes > 0 { ui.small(format!("{} probes run", self.lb_probes)); }
                            });
                        }
                    }
                }
                if !self.lb_places.is_empty() {
                    let sel_txt = match self.selected_datum { Some(d) if self.movable_datums().contains(&d) => format!("selected 0x{d:08X}"), Some(d) => format!("selected 0x{d:08X} (not offline-editable -> copies coords)"), None => "no object selected".into() };
                    ui.small(sel_txt);
                    egui::Grid::new("lb-places").num_columns(6).striped(true).spacing([8.0, 2.0]).show(ui, |ui| {
                        ui.strong("#"); ui.strong("x, y"); ui.strong("ground z"); ui.strong("probed colour"); ui.strong("dist"); ui.strong("");
                        ui.end_row();
                        for (rank, p) in self.lb_places.iter().enumerate() {
                            ui.label(format!("{}", rank + 1));
                            ui.label(format!("{:.1}, {:.1}", p.x, p.y));
                            ui.label(format!("{:.2}", p.z));
                            ui.horizontal(|ui| {
                                let (r, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                                ui.painter().rect_filled(r, 2.0, egui::Color32::from_rgb(p.disp[0], p.disp[1], p.disp[2]));
                                ui.small(format!("({}, {}, {}) mask {:.2} {}", p.disp[0], p.disp[1], p.disp[2], p.mask, scene::lb_kind_name(p.kind)));
                            });
                            ui.label(format!("{:.1}", p.dist));
                            ui.horizontal(|ui| {
                                if ui.small_button("Move here").on_hover_text("Move the selected object to this spot (x, y, ground + its lighting radius; rotation kept)").clicked() { move_to = Some(*p); }
                                if ui.small_button("Copy").clicked() { copy_place = Some(*p); }
                                if ui.small_button("Camera").on_hover_text("Fly the camera to this spot (no object move)").clicked() { goto_place = Some(*p); }
                            });
                            ui.end_row();
                        }
                    });
                }
            });
        self.show_lightbake = win_open;
        if do_enum { self.start_lightbake_enum(); }
        if let Some(sel) = new_sel {
            self.lb_sel = sel;
            self.lb_places.clear();
            self.lb_probes = 0;
            self.lb_dirty = true;
        }
        if do_find {
            // probe with the SELECTED object's own lighting radius when one is selected (the engine ray
            // starts at origin + radius, so a taller piece can hit an overhang/branch a crate passes under)
            let sel_obj = self.selected_datum.and_then(|d| self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == d).cloned());
            if let (Some(scene), Some(obj)) = (self.scene_ctl.as_ref(), sel_obj.as_ref()) { self.lb_radius = scene.object_light_radius(obj); }
            if let (Some(scene), Some(colors), Some(si)) = (self.scene_ctl.as_ref(), self.lb_colors.as_ref(), self.lb_sel) {
                let (places, probes) = scene.light_bake_find_placements(colors, si, self.lb_n.max(1), self.lb_radius);
                self.lb_probes = probes;
                self.lb_status = if places.is_empty() { format!("Swatch {si}: no ground placements found around its {} occurrence cells.", colors.swatches[si].cells.len()) } else { format!("Swatch {si}: {} placements from {} local probes at radius {:.2} (best dist {:.1}).", places.len(), probes, self.lb_radius, places[0].dist) };
                self.lb_places = places;
            }
        }
        if let Some(p) = move_to {
            self.lightbake_move_selected_to(ctx, p);
            if self.lb_follow_cam { self.frame_on_point(glam::Vec3::new(p.x, p.y, p.place_z)); }
        }
        if let Some(p) = goto_place { self.frame_on_point(glam::Vec3::new(p.x, p.y, p.place_z)); }
        if let Some(p) = copy_place {
            ctx.copy_text(format!("{:.2}, {:.2}, {:.2}", p.x, p.y, p.place_z));
            self.lb_status = format!("Copied ({:.2}, {:.2}, {:.2}) (ground {:.2} + probe radius) to the clipboard.", p.x, p.y, p.place_z, p.z);
        }
    }

    fn load_map(&mut self) {
        // Resolve the chosen map from the detected catalog — no path typing.
        let Some(path) = self
            .selected_map
            .and_then(|i| self.map_candidates.get(i))
            .map(|c| c.path.to_string_lossy().into_owned())
        else {
            self.map_status = "Select a map from the list first.".into();
            return;
        };
        self.map_path = path;
        let map_path = self.map_path.clone();
        // Purge the previous map's meshes/objects/selection up front so nothing
        // lingers while the new map streams in.
        self.clear_all_scene_objects();
        // A load already in flight owns the SceneController on its worker thread.
        // Instead of refusing, signal the worker to bail, drain its channel until it hands
        // the SceneController back (Done), join, and fall through to load the NEW map now.
        if self.load_rx.is_some() {
            if let Some(flag) = &self.load_cancel {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(rx) = self.load_rx.take() {
                // Blocking drain: worker sends its buffered batches then Done{scene} on bail.
                for msg in rx.iter() {
                    if let scene::LoadMsg::Done { scene, .. } = msg {
                        self.scene_ctl = Some(*scene);
                    }
                }
            }
            if let Some(h) = self.load_handle.take() {
                let _ = h.join();
            }
            self.load_cancel = None;
            // Drop the aborted map's partial geometry so it doesn't linger behind the new one.
            self.renderer.set_static_meshes(Vec::new());
            self.renderer.set_water_meshes(Vec::new());
            self.renderer.set_terrain_meshes(Vec::new());
            self.renderer.set_dynamic_meshes(Vec::new());
            self.renderer.set_alphatest_meshes(Vec::new());
            self.renderer.set_foliage_meshes(Vec::new());
            self.renderer.set_blend_meshes(Vec::new());
            self.renderer.set_additive_meshes(Vec::new());
            self.renderer.set_decal_meshes(Vec::new());
            self.renderer.set_sky_meshes(Vec::new());
        }
        // A Halo 4 or Halo 2 Anniversary cache never touches the native DLL - both load through
        // the pure-Rust h4 reader (h4/gui.rs) on its own worker; the reader detects the engine
        // from the cache and uses the right pointer constant (H2A's format IS Halo 4's apart
        // from that - docs/h2a_support_plan.md). Also cancels a load in flight. #h2a
        self.h4_reset();
        if h4::cache::cache_engine(std::path::Path::new(&map_path)).is_some() {
            self.h4_load_map(map_path);
            return;
        }
        if self.scene_ctl.is_none() {
            self.map_status = "Native DLL not loaded — can't parse maps.".into();
            return;
        }
        let open_result = self.scene_ctl.as_mut().unwrap().open_cache(&map_path);
        match open_result {
            Ok(()) => {
                // Prime the load state + per-scenario fog/exposure, then hand the
                // SceneController to a BACKGROUND worker so the ~30s native decode +
                // GPU upload never blocks the render thread. The main thread only
                // drains ready mesh batches (drive_load) → smooth window during load.
                let (fog, exposure, has_atmosphere, space_sky);
                {
                    let scene = self.scene_ctl.as_mut().unwrap();
                    scene.begin_bsp_load();
                    fog = scene.fog_uniform();
                    exposure = scene.exposure();
                    has_atmosphere = scene.has_atmosphere();
                    space_sky = scene.is_space_sky();
                }
                self.renderer.set_fog(&self.render_state.queue, &fog);
                // Space maps (no fogg) must not get the procedural blue horizon gradient.
                self.renderer.set_sky_atmosphere(has_atmosphere);
                // Space maps (condemned): black void clear + no gradient + bright starfield.
                self.renderer.set_space_sky(space_sky);
                self.renderer.set_exposure(&self.render_state.queue, exposure);
                self.renderer.set_static_meshes(Vec::new());
                self.renderer.set_water_meshes(Vec::new());
                self.renderer.set_terrain_meshes(Vec::new());
                self.renderer.set_dynamic_meshes(Vec::new());
                self.renderer.set_alphatest_meshes(Vec::new());
                self.renderer.set_foliage_meshes(Vec::new());
                self.renderer.set_blend_meshes(Vec::new());
                self.renderer.set_additive_meshes(Vec::new());
                self.renderer.set_decal_meshes(Vec::new());
                self.renderer.set_sky_meshes(Vec::new());
                self.renderer.set_overlay_lines(&self.render_state.device, &[]);
                self.load_t0 = std::time::Instant::now();
                self.map_status = format!("Loading {}…", self.map_path);

                // Spawn the worker: move the SceneController + shared GPU handles.
                let scene = self.scene_ctl.take().unwrap();
                let mesh = self.renderer.mesh_renderer_arc();
                let device = self.render_state.device.clone();
                let queue = self.render_state.queue.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let worker_cancel = cancel.clone();
                let handle = std::thread::spawn(move || {
                    scene::run_load_worker(scene, mesh, device, queue, tx, worker_cancel);
                });
                self.load_rx = Some(rx);
                self.load_handle = Some(handle);
                self.load_cancel = Some(cancel);
            }
            Err(e) => self.map_status = format!("Load failed: {e}"),
        }
    }

    /// Drain the background load worker's mesh stream into the renderer. Called
    /// every frame; a no-op when no load is in flight. The worker does ALL the
    /// heavy work (native decode + GPU upload) off the render thread, so this only
    /// appends ready batches — the window stays smooth during the whole load.
    fn drive_load(&mut self, ctx: &eframe::egui::Context) {
        self.h4_drive_load(); // no-op unless a Halo 4 load is in flight
        if self.load_rx.is_none() {
            return;
        }
        let mut done_scene: Option<Box<SceneController>> = None;
        let mut disconnected = false;
        // Drain everything ready this frame (non-blocking).
        if let Some(rx) = self.load_rx.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(msg) => match msg {
                        scene::LoadMsg::Opaque(m) => self.renderer.append_static_meshes(m),
                        scene::LoadMsg::Water(m) => self.renderer.append_water_meshes(m),
                        scene::LoadMsg::WaterPlanes(p) => self.renderer.append_water_planes(p),
                        scene::LoadMsg::Terrain(m) => self.renderer.append_terrain_meshes(m),
                        scene::LoadMsg::Alphatest(m) => self.renderer.append_alphatest_meshes(m),
                        scene::LoadMsg::Foliage(m) => self.renderer.append_foliage_meshes(m),
                        scene::LoadMsg::Blend(m) => self.renderer.append_blend_meshes(m),
                        scene::LoadMsg::Additive(m) => self.renderer.append_additive_meshes(m),
                        // Decals: dedicated UNLIT decal pipelines, per-decal blend bucket
                        // (alpha/additive/multiply) — no PBR spec/fog, scorch darkens.
                        scene::LoadMsg::StaticExtra(m) => self.renderer.append_decal_meshes(m),
                        scene::LoadMsg::Sky(segs) => {
                            self.renderer.set_sky_meshes(segs);
                        }
                        scene::LoadMsg::Progress { done, total } => {
                            let pct = if total > 0 { (done * 100 / total).min(99) } else { 0 };
                            self.load_frac = if total > 0 { (done as f32 / total as f32).min(0.99) } else { 0.0 };
                            self.map_status = format!("Loading {}… {pct}% ({done}/{total})", self.map_path);
                        }
                        scene::LoadMsg::Done { scene, exposure } => {
                            self.renderer.set_exposure(&self.render_state.queue, exposure);
                            self.preview_renderer.set_exposure(&self.render_state.queue, exposure);
                            done_scene = Some(scene);
                        }
                    },
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if let Some(scene) = done_scene {
            // Load finished — hand the SceneController back to the main thread.
            self.scene_ctl = Some(*scene);
            self.load_rx = None;
            // Schedule a working-set trim a few seconds out (once deferred object/foliage
            // decode has settled) to release the parallel decode's freed-but-retained heap.
            self.trim_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(3));
            // Land the camera IN the map at a spawn point (unless the user pinned an
            // explicit HMS_CAM A/B view), so testing doesn't need flying to the map first.
            if std::env::var("HMS_CAM").is_err() {
                if let Some((pos, yaw, pitch)) =
                    self.objscene().and_then(|s| s.spawn_camera_pose())
                {
                    self.camera.pos = pos;
                    self.camera.yaw = yaw;
                    self.camera.pitch = pitch;
                    log::info!("spawn camera: pos=[{:.1},{:.1},{:.1}] yaw={:.1} pitch={:.1}",
                        pos.x, pos.y, pos.z, yaw.to_degrees(), pitch.to_degrees());
                }
            }
            if let Some(h) = self.load_handle.take() {
                let _ = h.join();
            }
            // The palette preview is lit per map -> rebuild it for the new one.
            self.preview_for = None;
            // Apply the per-scenario sceg colour-grade tints (Zealot purple etc.).
            if let Some((sun_tint, amb_tint)) = self.scene_ctl.as_ref().map(|s| s.scene_tints()) {
                self.renderer.set_scene_tints(sun_tint, amb_tint);
                self.preview_renderer.set_scene_tints(sun_tint, amb_tint); // preview lit like the viewport
                log::info!("scene tints: sun=[{:.2},{:.2},{:.2}] ambient=[{:.2},{:.2},{:.2}]",
                    sun_tint[0], sun_tint[1], sun_tint[2], amb_tint[0], amb_tint[1], amb_tint[2]);
            }
            // Per-map dynamic-object directional soft-shade strength (airprobe dir_weight).
            if let Some(s) = self.scene_ctl.as_ref().map(|s| s.obj_dir_strength()) {
                self.renderer.set_obj_dir_strength(s);
                self.preview_renderer.set_obj_dir_strength(s);
            }
            // Apply the map's tag-driven auto-exposure band (cfxs) to the post pass,
            // replacing the hardcoded 0.25/-2/+1. None → keep the default band.
            if let Some((key, lo, hi)) = self.scene_ctl.as_ref().and_then(|s| s.autoexposure_band()) {
                self.renderer.set_auto_exposure(&self.render_state.queue, key, lo, hi);
                self.preview_renderer.set_auto_exposure(&self.render_state.queue, key, lo, hi);
                log::info!("auto-exposure (cfxs tag): key={key:.3} min_ev={lo:.2} max_ev={hi:.2}");
            }
            // cfxs self-illum exposure (P, s) → ILLUM_SCALE; None = engine defaults.
            {
                let ie = self.scene_ctl.as_ref().and_then(|s| s.illum_exposure());
                self.renderer.set_illum_exposure(ie);
                self.preview_renderer.set_illum_exposure(ie);
            }
            // Gate the glass sun lanes by how much of THIS map the lightmapper says the
            // sun actually reaches (see SceneController::baked_sun_reach).
            if let Some(sc) = self.scene_ctl.as_ref() {
                let r = sc.baked_sun_reach();
                self.renderer.set_sun_reach(r);
                self.preview_renderer.set_sun_reach(r);
                log::info!("sun reach (baked visibility population): {r:.4}");
            }
            // The map's OWN bloom knee + intensity (cfxs). Intensity goes in its OWN lane, never
            // into `bloom_scale` (the USER's slider) — that would double-apply it against the
            // 0.669 default baked into the combine pass. Unconditional: the authored per-map
            // curve IS the correct out-of-the-box lighting.
            if let Some((pt, inh, inten)) = self.scene_ctl.as_ref().and_then(|s| s.bloom_curve()) {
                self.renderer.set_bloom_curve(&self.render_state.queue, pt, inh, inten);
                self.preview_renderer.set_bloom_curve(&self.render_state.queue, pt, inh, inten);
                log::info!("bloom (cfxs tag): point={pt:.3} inherent={inh:.3} intensity={inten:.3}");
            }
            // Per-level bloom colours (large/medium/small) from the same cfxs.
            if let Some(c) = self.scene_ctl.as_ref().and_then(|s| s.bloom_colors()) {
                self.renderer.set_bloom_colors(&self.render_state.queue, c);
                self.preview_renderer.set_bloom_colors(&self.render_state.queue, c);
            }
            // The scenario's default screen effect (every colour term through
            // the engine matrix), unconditional like the bloom curve; identity when the map authors
            // none. The placed Forge special-FX orbs are folded in by tick_scene as objects appear.
            {
                self.screenfx_cache.clear();
                self.screenfx_pushed = None;
                self.refresh_screen_fx(&[]);
            }
            // Per-map sun direction from the baked airprobe grid. No tag stores a sun
            // DIRECTION — the engine recovers it from the lightprobe dual-VMF dominant lobe.
            // None → keep the built-in fallback.
            if let Some(sun) = self.scene_ctl.as_ref().and_then(|s| s.scene_sun_dir()) {
                self.renderer.set_sun_dir(glam::Vec3::from(sun));
                self.preview_renderer.set_sun_dir(glam::Vec3::from(sun));
                log::info!("scene sun dir (airprobe-derived): [{:.3},{:.3},{:.3}]", sun[0], sun[1], sun[2]);
            } else {
                log::info!("scene sun dir: no usable probe data — using fallback");
            }
            // Scenario SimpleLights (point/spot) → renderer uniform.
            let sl = self.scene_ctl.as_ref().map(|s| s.simple_lights()).unwrap_or_default();
            let sl_count = (sl.len() / 20) as u32;
            self.renderer.set_simple_lights(&self.render_state.queue, sl_count, &sl);
            log::info!("scene simple lights: {sl_count}");
            // The authored fog volumes as engine planes for the screen-space depth-composite
            // pass (depth-graded fog, no stacked-alpha slabs).
            let fog_vols = self.scene_ctl.as_ref().map(|s| s.planar_fog_volumes()).unwrap_or_default();
            let n = fog_vols.len();
            self.renderer.set_planar_fog_volumes(fog_vols);
            if n > 0 { log::info!("planar fog volumes: {n}"); }
            // Volumetric light beams: additive cross-billboard tubes from each
            // non-omni SimpleLight, drawn through the additive (hologram) pass.
            let beams = self.scene_ctl.as_ref().map(|s| s.light_beam_meshes()).unwrap_or_default();
            if !beams.is_empty() {
                let mut beam_meshes = Vec::with_capacity(beams.len());
                {
                    let mr = self.renderer.mesh_renderer();
                    for (verts, indices) in &beams {
                        beam_meshes.push(mr.upload_mesh(
                            &self.render_state.device, &self.render_state.queue,
                            verts, indices, &[glam::Mat4::IDENTITY],
                            None, None, None, None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None,
                        None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None));
                    }
                }
                let n = beam_meshes.len();
                self.renderer.append_additive_meshes(beam_meshes);
                log::info!("light beams: {n}");
            }
            // Sun lens flare: elements from the sun's lens tag (empty if none).
            let flare = self.scene_ctl.as_ref().map(|s| s.lens_flare()).unwrap_or_default();
            let flare_n = flare.len() / 6;
            self.renderer.set_lens_flare(flare);
            log::info!("lens flare elements: {flare_n}");
            // Effect scenery: crossed additive billboards at each efsc placement,
            // textured with the resolved particle bitmap (efsc→effe→prt3→bitm) or a soft
            // generic sprite when unresolved. Drawn through the additive (hologram) pass.
            let fx = self.scene_ctl.as_ref().map(|s| s.effect_scenery_billboards()).unwrap_or_default();
            if !fx.is_empty() {
                let mut fx_meshes = Vec::with_capacity(fx.len());
                {
                    let mr = self.renderer.mesh_renderer();
                    for (verts, indices, bitm) in &fx {
                        // Decode the particle bitmap; if it DOESN'T decode (unsupported
                        // format / bad resolve), SKIP this billboard — uploading with tv=None
                        // would bind a 1×1 WHITE default texture, which fs_holo glows across
                        // the whole quad as a solid additive square.
                        let tv = if *bitm != 0 && *bitm != 0xFFFF_FFFF {
                            self.scene_ctl.as_ref()
                                .and_then(|s| s.decode_bitmap_public(*bitm))
                                .map(|t| hms_render::upload_texture_bgra_nomip(&self.render_state.device, &self.render_state.queue, &t.0, t.1, t.2))
                        } else { None };
                        let Some((view, _tex)) = tv else { continue };
                        let m = mr.upload_mesh(
                            &self.render_state.device, &self.render_state.queue,
                            verts, indices, &[glam::Mat4::IDENTITY],
                            None, Some(&view), None, None, None, None,
                            [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None,
                        None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None);
                        fx_meshes.push(m);
                    }
                }
                let n = fx_meshes.len();
                self.renderer.append_additive_meshes(fx_meshes);
                log::info!("effect scenery billboards: {n}");
            }
            // BSP-placed effect_scenery light volumes (Sword Base oni gravlifts etc.) drawn
            // as camera-facing additive ribbons (bump_xform.w = 4 lane). HMS_NO_LIFT_FX=1 disables.
            let lv = self.scene_ctl.as_ref().map(|s| s.effect_scenery_light_volumes()).unwrap_or_default();
            if !lv.is_empty() {
                let mut lv_meshes = Vec::with_capacity(lv.len());
                {
                    let mr = self.renderer.mesh_renderer();
                    let (white, _wt) = hms_render::upload_texture_bgra_nomip(&self.render_state.device, &self.render_state.queue, &[255, 255, 255, 255], 1, 1);
                    for (verts, indices, fxc) in &lv {
                        lv_meshes.push(mr.upload_mesh(
                            &self.render_state.device, &self.render_state.queue,
                            verts, indices, &[glam::Mat4::IDENTITY],
                            None, Some(&white), Some(&white), None, None, None,
                            [0.0, 0.0], [1.0, 1.0], [0.0; 4], 0.0, 0.0, [0.0, -1.0], [0.0; 4], [0.0, 0.0, 0.0, 4.0], *fxc, [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None,
                            None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None));
                    }
                }
                let n = lv_meshes.len();
                self.renderer.append_additive_meshes(lv_meshes);
                log::info!("effect scenery light volumes: {n}");
            }
            // Enumerate the sandbox forge palette so objects can be browsed + placed
            // without a live game. Fresh per map; drop any prior local objects.
            let full_pal = self
                .scene_ctl
                .as_ref()
                .map(|s| s.forge_palette_static())
                .unwrap_or_default();
            self.static_palette = full_pal.iter().map(|(t, n, _, _, _)| (*t, n.clone())).collect();
            self.static_palette_cat = full_pal.iter().map(|(_, _, c, _, _)| *c).collect();
            // The REAL forge category per entry (e.g. "ff_weapons_covenant"), for grouping.
            self.static_palette_catname = full_pal.iter().map(|(_, _, _, cn, _)| cn.clone()).collect();
            // Variant Name sid per entry — selects the hlmt model variant on spawn (rocket hog).
            self.static_palette_variant = full_pal.iter().map(|(_, _, _, _, vs)| *vs).collect();
            self.selected_static_pal = if self.static_palette.is_empty() { None } else { Some(0) };
            self.local_objects.clear();
            // A variant is tied to its base map; drop any previously-imported .mvar objects
            // so they don't render on a different map (re-import after loading the canvas).
            self.mvar_objects.clear();
            self.mvar_colors.clear();
            self.mvar_meta.clear();
            self.next_local_datum = 0xF000_0000;
            log::info!("forge palette: {} placeable objects", self.static_palette.len());
            // Auto-load the map's BAKED designer objects (scenery/vehicles/etc.) so they
            // render immediately with no live game. Converted to ObjectInfo and
            // merged with live + locally-placed objects in tick_scene.
            // Object tags that belong to this map's FORGE SANDBOX PALETTE: a scenario
            // placement of one of those is canvas / default-variant content, which the map
            // VARIANT owns -- so it must never be placed or tracked from the scenario (it would
            // draw the canvas copy ON TOP of the variant's own objects and, because tracked
            // objects are saveable, let base-map scenery be written back into the .mvar). The
            // `placement_flags & 0x41` test below does not catch these -- canvas forge objects
            // ARE auto-placed. HMS_SCNR_FORGE=1 keeps them (diagnostic).
            let forge_owned: std::collections::HashSet<u32> = if std::env::var("HMS_SCNR_FORGE").is_ok() {
                Default::default()
            } else {
                self.scene_ctl.as_ref().map(|s| s.forge_owned_tags()).unwrap_or_default()
            };
            let mut forge_skipped = 0usize;
            let mut scenario_ident = std::collections::HashMap::new();
            self.scenario_objects = self
                .scene_ctl
                .as_ref()
                .map(|s| s.scenario_objects())
                .unwrap_or_default()
                .into_iter()
                .filter(|o| {
                    let owned = forge_owned.contains(&o.obj_tag);
                    if owned {
                        forge_skipped += 1;
                    }
                    !owned
                })
                // Effect scenery (category 8) is rendered as billboards, not object
                // meshes — its mode_tag is a bitmap, not a render_model.
                .filter(|o| o.category != 8)
                // The engine does NOT auto-place objects flagged "not automatically"
                // (bit0) — those are spawned by scripts / the gametype (spawn points,
                // objective markers, mode-specific scenery), and on MP maps the map
                // VARIANT drives them, not the scenario. "never placed" (bit6) is never
                // spawned at all. Skip both so a fresh load matches what the game shows.
                .filter(|o| (o.placement_flags & 0x41) == 0 || std::env::var("HMS_SHOW_ALL_SCNR").is_ok())
                .enumerate()
                .map(|(i, o)| {
                    let datum = 0xE000_0000u32.wrapping_add(i as u32);
                    // Keep the placement coordinates ObjectInfo drops.
                    scenario_ident.insert(datum, (o.category, o.palette_index, o.name_index));
                    ObjectInfo {
                    datum,
                    type_sig: 0,
                    sig0: 0,
                    sig1: 0,
                    pos: o.pos,
                    health: 1.0,
                    shield: 1.0,
                    mode_tag: o.mode_tag,
                    fwd: o.fwd,
                    up: o.up,
                    attached: [0; 8],
                    primary_tag: o.obj_tag,
                    variant_name_sid: 0,
                }})
                .collect();
            self.scenario_ident = scenario_ident;
            log::info!(
                "scenario objects: {} baked placements ({forge_skipped} forge-palette placements skipped — the map variant owns those)",
                self.scenario_objects.len()
            );
            // Classify the scenario's own spawn markers by obj tag path once per load.
            self.scenario_spawn_tags = match self.scene_ctl.as_ref() {
                Some(s) => map_spawns::spawn_obj_tags(self.scenario_objects.iter().map(|o| o.primary_tag), |t| s.tag_name_of(t)),
                None => Default::default(),
            };
            log::info!("{}", self.map_spawns_status());
            // Record the loaded map's base id so a .mvar targeting a DIFFERENT map knows to
            // reload — and, crucially, so a variant for the SAME map (e.g. another Forge World
            // variant) is detected as same-map and swaps in place with NO reload. Read it
            // straight from the loaded map's own .mapinfo (robust; independent of the picker
            // selection/candidate indices, which could be stale or lack a .mapinfo).
            self.loaded_map_id = mapcat::read_map_id(std::path::Path::new(&self.map_path)).or_else(|| {
                self.selected_map
                    .and_then(|i| self.map_candidates.get(i))
                    .and_then(|c| c.map_id)
            });
            log::info!("loaded map id = {:?} ({})", self.loaded_map_id, self.map_path);
            // Then, if a variant was queued pending this load (the user picked a variant for a
            // not-yet-loaded map), render its objects now.
            if let Some(pending) = self.pending_mvar.take() {
                if let Some(variant) = mvar::parse_variant(&pending) {
                    let nm = pending.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    let placed = self.render_variant_objects(&variant, &nm);
                    self.after_variant_render(placed, pending);
                }
            } else if let Some(p) = self.autoload_variant.take() {
                // Startup: auto-load the last-opened .mvar now that its base map is up.
                self.import_mvar(p);
            }
            let load_secs = self.load_t0.elapsed().as_secs_f32();
            // NOTE: this is only the BSP+foliage streaming time. The object-model rebuild runs
            // AFTER this over the next frames (the "lag after loaded"); settle_pending keeps the
            // window pumping until it's done, then stamps the TRUE end-to-end time below.
            self.settle_pending = true;
            self.map_status = format!("Loaded {} geometry in {:.1}s — finalizing objects…", self.map_path, load_secs);
            log::info!("map geometry loaded (BSP + foliage) in {load_secs:.2}s; finalizing objects");
            mapcat::save_last_map(&self.map_path);
            // Headless auto-screenshot: frame the whole map (or use HMS_CAM).
            if self.autoshot.is_some() {
                if let Some((mn, mx)) = self.objscene().and_then(|s| s.scene_bounds()) {
                    let mn = glam::Vec3::from(mn);
                    let mx = glam::Vec3::from(mx);
                    let center = (mn + mx) * 0.5;
                    let radius = ((mx - mn).length() * 0.5).max(1.0);
                    eprintln!("HMS_DIAG BOUNDS min=({:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1}) center=({:.1},{:.1},{:.1}) radius={:.1}",
                        mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, center.x, center.y, center.z, radius);
                    let cam_override = std::env::var("HMS_CAM").ok().and_then(|s| {
                        let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                        if v.len() >= 5 { Some(v) } else { None }
                    });
                    if let Some(v) = cam_override {
                        self.camera.pos = glam::Vec3::new(v[0], v[1], v[2]);
                        self.camera.yaw = v[3];
                        self.camera.pitch = v[4];
                    } else {
                        let dist = radius / (self.camera.fov_y * 0.5).tan() * 1.15 + 5.0;
                        let dir = glam::Vec3::new(0.55, 0.55, -0.45).normalize();
                        self.camera.pos = center - dir * dist;
                        self.camera.yaw = dir.y.atan2(dir.x);
                        self.camera.pitch = dir.z.clamp(-1.0, 1.0).asin();
                    }
                }
                self.autoshot_frames = 6;
            }
            // Place the interactive camera at an exact view to A/B against a Sapien
            // ground-truth capture. HMS_CAM="x,y,z,yawDeg,pitchDeg" (+ optional HMS_FOV degrees).
            if self.autoshot.is_none() {
                if let Ok(s) = std::env::var("HMS_CAM") {
                    let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                    if v.len() >= 5 {
                        self.camera.pos = glam::Vec3::new(v[0], v[1], v[2]);
                        self.camera.yaw = v[3].to_radians();
                        self.camera.pitch = v[4].to_radians();
                        if let Some(f) = std::env::var("HMS_FOV").ok().and_then(|x| x.trim().parse::<f32>().ok()) {
                            if f > 1.0 && f < 179.0 { self.camera.fov_y = f.to_radians(); }
                        }
                        log::info!("HMS_CAM applied: pos=({:.1},{:.1},{:.1}) yaw={:.1} pitch={:.1} fov={:.1}",
                            v[0], v[1], v[2], v[3], v[4], self.camera.fov_y.to_degrees());
                    }
                }
            }
        } else if disconnected {
            // Worker died without sending Done — the SceneController it owned (and its
            // NativeDll) is gone. Self-heal: reload the parsing DLL so the session isn't
            // bricked ("Native DLL not loaded") by one bad load. The old DLL module is
            // leaked (never FreeLibrary'd), so re-loading just bumps the refcount.
            self.load_rx = None;
            if let Some(h) = self.load_handle.take() {
                let _ = h.join();
            }
            if self.scene_ctl.is_none() {
                match NativeDll::load(&self.dll_path) {
                    Ok(dll) => {
                        self.scene_ctl = Some(SceneController::new(dll));
                        self.map_status = "Load failed (worker terminated) — parser reloaded; try again.".into();
                    }
                    Err(e) => {
                        self.map_status = format!("Load failed (worker terminated); parser reload failed: {e}");
                    }
                }
            } else {
                self.map_status = "Load failed (worker terminated).".into();
            }
        }
        // Keep the frame loop pumping while a load is in flight.
        if self.load_rx.is_some() {
            ctx.request_repaint();
        }
    }

    /// If the injected DLL has published the running scenario, auto-select the
    /// matching detected map (once). Retail MCC usually strips the name/path, so
    /// this is best-effort — the dropdown is always the fallback.
    fn auto_select_loaded_map(&mut self) {
        if self.auto_selected {
            return;
        }
        let Some(mi) = &self.map_info else { return };
        let info = mi.read();
        if !info.valid {
            return;
        }
        let want_path = info.map_path.to_lowercase();
        let want_name = info.scenario_name.to_lowercase();
        let hit = self.map_candidates.iter().position(|c| {
            (!want_path.is_empty() && c.path.to_string_lossy().to_lowercase() == want_path)
                || (!want_name.is_empty()
                    && (c.scenario_name.to_lowercase() == want_name
                        || want_name.ends_with(&c.stem.to_lowercase())))
        });
        if let Some(i) = hit {
            self.selected_map = Some(i);
            self.auto_selected = true;
            // Auto-load it — no click needed ("resolve the map and everything").
            if self.scene_ctl.is_some() {
                self.load_map();
            } else {
                let hint = if cfg!(feature = "injection") { " (inject to parse)" } else { "" };
                self.map_status = format!("Loaded map detected: {}{hint}.", self.map_candidates[i].label());
            }
        } else if info.scenario_datum != 0 {
            self.auto_selected = true;
            self.map_status = format!(
                "Game map detected (datum {:#010X}) — pick it from the list.",
                info.scenario_datum
            );
        }
    }

    /// True only when MCC's live-published map is the SAME map the viewer has loaded,
    /// so the live engine object pool's tag ids are valid in the viewer's own tag cache.
    /// Biased toward FALSE (skip live objects) whenever the maps can't be positively
    /// matched — a false negative just hides live objects; a false positive crashes the
    /// viewer by decoding foreign tags.
    fn live_map_matches(&self) -> bool {
        let Some(mi) = self.map_info.as_ref().map(|m| m.read()) else {
            return false;
        };
        if !mi.valid {
            return false;
        }
        let stem = |s: &str| {
            std::path::Path::new(s)
                .file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        };
        let mine = stem(&self.map_path);
        if mine.is_empty() {
            return false;
        }
        let theirs_path = stem(&mi.map_path);
        let theirs_name = mi.scenario_name.to_lowercase();
        (!theirs_path.is_empty() && (mine == theirs_path || theirs_path.ends_with(&mine) || mine.ends_with(&theirs_path)))
            || (!theirs_name.is_empty() && (theirs_name == mine || theirs_name.ends_with(&mine) || theirs_name.ends_with(&format!("\\{mine}")) || theirs_name.ends_with(&format!("/{mine}"))))
    }

    /// `anim_time` = the frame's animation clock (egui time, seconds) — drives the flash
    /// lights' periodic colour/intensity when the SimpleLights uniform is re-packed.
    fn tick_scene(&mut self, anim_time: f32) {
        // On first tick: if the game isn't publishing a map (not attached) but we
        // remembered a map from last time, reopen it automatically.
        if !self.booted {
            self.booted = true;
            let game_map = self.map_info.as_ref().map(|m| m.read().valid).unwrap_or(false);
            // HMS_MAP forces its map regardless of the running game; otherwise only
            // auto-reopen the persisted map when the game isn't publishing one.
            let forced = std::env::var("HMS_MAP").is_ok();
            if (forced || !game_map) && self.selected_map.is_some() && self.scene_ctl.is_some() {
                self.load_map();
            }
        }
        self.auto_select_loaded_map();
        // A variant whose first render placed 0 objects (scene/forge-palette
        // not warm yet) is retried here for a few ticks, so opening it once is enough.
        if self.variant_retry.is_some() && self.load_rx.is_none() && self.scene_ctl.is_some() {
            let (path, attempts) = self.variant_retry.take().unwrap();
            if let Some(variant) = mvar::parse_variant(&path) {
                let nm = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                let placed = self.render_variant_objects(&variant, &nm);
                if placed > 0 {
                    mapcat::save_last_variant(&path.to_string_lossy());
                    self.current_variant_path = Some(path.clone());
                } else if attempts > 1 {
                    self.variant_retry = Some((path, attempts - 1));
                }
            }
        }
        // HMS_DIAG object-path trace (throttled ~1/sec) so a live run reveals where the
        // forge-object pipeline breaks: MMF connected? object count? mode_tags resolved?
        let diag = {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            std::env::var("HMS_DIAG").is_ok() && N.fetch_add(1, Ordering::Relaxed) % 60 == 0
        };
        // The LIVE engine object pool (objtable) carries MCC's tag ids for MCC's
        // CURRENTLY-loaded map. Feeding those ids into the VIEWER's own tag cache (a
        // possibly-DIFFERENT map) would make the native open_model/decode dereference an
        // out-of-range tag entry. Only trust live tags when MCC's live map matches the
        // viewer's loaded cache; otherwise render ONLY the viewer's own scenario +
        // locally-placed objects (always valid).
        let live_map_matches = self.live_map_matches();
        // Live objects from the game MMF (empty when no game / map mismatch) MERGED with
        // locally-placed objects (standalone forge editor) so placed objects render with OR
        // without a live game. No early-return: local objects must render even when objtable=None.
        let live = if live_map_matches {
            self.objtable.as_ref().map(|ot| ot.read()).unwrap_or_default()
        } else {
            Vec::new()
        };
        // LIVE object-table count in isolation. If objtable is connected but this is
        // 0 while a game is running, the DLL isn't publishing the pool (client-side
        // zero-active-pool / AOB-scan fail) — a RUNTIME issue, not a viewer read gap.
        let live_count = live.len();
        let mut objects = live;
        // FORGE-TABLE placements. The engine object pool reads EMPTY client-side (the
        // documented CLIENT_OBJ_DIAG condition) — but placed forge objects still live in the
        // AOB-scanned forge table, which carries pos/fwd/up. Render them from there, MERGED
        // with the live pool (dedup by datum so an object present in both isn't doubled),
        // resolving the render model through the master palette:
        // type_key=(ItemCategory<<8)|ItemVariant matches PaletteEntry.variant_index
        // → mode_tag_id (fall back to the category high byte if no exact variant match).
        // Not gated on live_map_matches: the whole native decode path is SEH-wrapped
        // (SehParseModeTag / DecodeGeometry / DecodeUVs / SehResolveDiffuse / GetParsedBitmap
        // all __try/__except → return failure), so a foreign tag just fails and is skipped.
        let mut forge_rendered = 0usize;
        let mut forge_placements_read = 0usize;
        let mut forge_skipped_nomodel = 0usize;
        {
            if let Some(ft) = self.forge_table.as_ref() {
                let existing: std::collections::HashSet<u32> = objects.iter().map(|o| o.datum).collect();
                for p in ft.read_placements() {
                    forge_placements_read += 1;
                    let want_datum = if p.datum != 0 { p.datum } else { 0xF000_0000 | p.forge_idx };
                    if existing.contains(&want_datum) { continue; }
                    let mode = self
                        .palette
                        .iter()
                        .find(|e| e.variant_index == p.type_key)
                        .or_else(|| self.palette.iter().find(|e| (e.variant_index >> 8) == (p.type_key >> 8)))
                        .map(|e| e.mode_tag_id)
                        .unwrap_or(0);
                    if mode == 0 || mode == 0xFFFF_FFFF {
                        forge_skipped_nomodel += 1;
                        continue;
                    }
                    let fwd = if p.fwd.iter().all(|v| v.is_finite()) && p.fwd.iter().any(|&v| v.abs() > 1e-4) {
                        p.fwd
                    } else {
                        [1.0, 0.0, 0.0]
                    };
                    let up = if p.up.iter().all(|v| v.is_finite()) && p.up.iter().any(|&v| v.abs() > 1e-4) {
                        p.up
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    objects.push(hms_ipc::ObjectInfo {
                        datum: want_datum,
                        type_sig: 0,
                        sig0: 0,
                        sig1: 0,
                        pos: p.pos,
                        health: 1.0,
                        shield: 1.0,
                        mode_tag: mode,
                        fwd,
                        up,
                        attached: [0; 8],
                        primary_tag: 0,
                        variant_name_sid: 0,
                    });
                    forge_rendered += 1;
                }
            }
        }
        // Merge the base scenario's PERMANENT scenery (already filtered by the
        // placement-flag test `& 0x41 == 0`, so spawn/forge content the variant owns is excluded).
        // This set is background/built-in scenery the engine always places on map load — Highlands'
        // covenant cruisers, distant rocks, buildings — and it stays merged while a .mvar variant
        // is loaded (the headless + script-host paths do the same). Spawn points are netgame-flag
        // blocks (not in scenario_objects) and the forge canvas comes from the variant, so no
        // doubling occurs (verified across Highlands/Forge/Countdown/Reflection).
        // The scenario's own spawn markers are dropped from the RENDERED set
        // (and so from the scene's pick list) unless View > "Show map spawn points" is on.
        objects.extend(
            self.scenario_objects
                .iter()
                .filter(|o| self.show_map_spawns || !self.scenario_spawn_tags.contains(&o.primary_tag))
                .cloned(),
        );
        objects.extend(self.mvar_objects.iter().cloned());
        objects.extend(self.local_objects.iter().cloned());
        // Always-on, on-screen forge diagnostic — no console needed. Shows exactly which
        // link of the in-memory forge pipeline is broken.
        self.forge_diag = format!(
            "forge: table={} palette={} map_match={} live_pool={} placements_read={} placed={} skipped_nomodel={}",
            self.forge_table.is_some(), self.palette.len(), live_map_matches,
            live_count, forge_placements_read, forge_rendered, forge_skipped_nomodel,
        );
        if diag {
            let valid = objects.iter().filter(|o| o.mode_tag != 0 && o.mode_tag != 0xFFFF_FFFF).count();
            let sample: Vec<u32> = objects.iter().take(5).map(|o| o.mode_tag).collect();
            eprintln!(
                "HMS_DIAG OBJ live={} forge_table={} scenario={} local={} total={} valid_mode_tag={} sample_mode_tags={:x?} has_cache={} objtable_connected={} forge_table_connected={} palette={}",
                live_count, forge_rendered, self.scenario_objects.len(), self.local_objects.len(),
                objects.len(), valid, sample,
                self.objscene().map(|s| s.has_cache()).unwrap_or(false),
                self.objtable.is_some(), self.forge_table.is_some(), self.palette.len()
            );
        }
        // The placed Forge special-FX orbs' screen effects follow the CURRENT object set
        // (add / delete / duplicate / variant load all flow through here).
        self.refresh_screen_fx(&objects);
        // Nothing to draw and nothing drawn: idle. When the set just BECAME empty (e.g. "Clear
        // placed" with no live game, or hiding a map's only scenario spawn markers) fall
        // through: the rebuild below sees the changed signature and returns empty lists for EVERY
        // pass (opaque/cutout/holo/holo-solid/blend) and clears the scene's pick list.
        if objects.is_empty() && self.last_objects.is_empty() {
            return;
        }
        let mut forge_colors = self
            .forge_table
            .as_ref()
            .map(|c| c.read())
            .unwrap_or_default();
        // Merge the OFFLINE .mvar objects' parsed (team, color) so their forge/team
        // change-colour renders (the live forge_table only has live-injected objects).
        forge_colors.extend(self.mvar_colors.iter().map(|(k, v)| (*k, *v)));
        // The dropdown hover preview overrides the selection's colours LAST (transient;
        // ObjMeta / mvar_colors are untouched). maybe_rebuild hashes forge_colors into its signature,
        // so this costs one object rebuild when the hovered entry changes and nothing per frame.
        if let Some((_, pv)) = self.hover_preview.as_ref() {
            forge_colors.extend(pv.iter().map(|(k, v)| (*k, *v)));
        }
        // Full rows (forge_idx, datum, team, color) for the properties panel.
        self.forge_rows = self
            .forge_table
            .as_ref()
            .map(|c| c.read_full())
            .unwrap_or_default();
        self.last_objects = objects.clone();

        // Per-datum visual scale for offline objects tagged with the forge "scale"
        // label (X330 spawn_seq). Only non-unity entries are stored.
        // The per-object SCALED flag is inside ObjMeta::scale; the global "Scaled
        // objects" switch empties the map outright (no object scales).
        let sconv = self.sc_convention;
        let scales: std::collections::HashMap<u32, f32> = if self.obj_globals.scaled {
            self.mvar_meta
                .iter()
                .filter_map(|(&d, m)| {
                    // ObjMeta::scale handles both games (a Halo 4 object = its
                    // record scale, or the gametype rule when SCALED is on).
                    let s = m.scale(sconv);
                    ((s - 1.0).abs() > 1e-4).then_some((d, s))
                })
                .collect()
        } else {
            Default::default()
        };
        // Objects FORCED into the sun shadow map (SHADOW flag / green+scale rule),
        // unless the global "Shadow casters (Forge)" switch is off (engine-default casters such as
        // vehicles are decided in the scene from the obje tag and are not affected).
        let casters: std::collections::HashSet<u32> = if self.obj_globals.shadowcasters {
            self.mvar_meta.iter().filter(|(_, m)| m.shadow_on()).map(|(&d, _)| d).collect()
        } else {
            Default::default()
        };
        // Through the ObjectScene trait (Reach SceneController / Halo 4 H4ObjectScene);
        // the field-wise accessor keeps `renderer` / `render_state` borrowable alongside.
        if let Some(scene) = objscene_parts_mut(self.h4_active, &mut self.h4_scene, &mut self.scene_ctl) {
            scene.set_object_scales(scales);
            scene.set_object_casters(casters);
        }

        let rebuilt = match objscene_parts_mut(self.h4_active, &mut self.h4_scene, &mut self.scene_ctl) {
            Some(scene) if scene.has_cache() => scene.maybe_rebuild(
                &objects,
                &forge_colors,
                self.renderer.mesh_renderer(),
                &self.render_state.device,
                &self.render_state.queue,
            ),
            _ => None,
        };
        if rebuilt.is_some() {
            self.last_rebuild_at = std::time::Instant::now();
            self.idle_released = false;
            // Keep the Halo 4 sun shadow fit on the moved / added / deleted casters
            if let Some(Some((b, c))) = objscene_parts_mut(self.h4_active, &mut self.h4_scene, &mut self.scene_ctl).map(|s| s.h4_shadow_setup()) {
                self.renderer.set_h4_shadow(b, c);
            }
        }
        if let Some(objscene::RebuiltLanes { opaque, cutout, holo, holo_solid, blend }) = rebuilt {
            self.renderer.set_dynamic_meshes(opaque);
            // Cutout objects (tree canopies) → alpha-test pass so leaves are see-through.
            self.renderer.set_dynamic_cutout_meshes(cutout);
            // Holo/objective objects + armor-ability icons → additive team-tinted glow pass.
            self.renderer.set_dynamic_holo_meshes(holo);
            // Object markers (spawns/hill globes/kill-safe shells) → alpha-blended holo pass so
            // their translucent SHAPE stays readable at every angle (additive vanished on bright bg).
            self.renderer.set_dynamic_holo_solid_meshes(holo_solid);
            // Forge glass panes (window/cover_glass) → alpha-blend pass so they're see-through.
            self.renderer.set_dynamic_blend_meshes(blend);
            // The object rebuild also refreshed the always-on phmo overlay for
            // hidden blocks — flag the overlay buffers to re-merge it next frame.
            self.overlays_dirty = true;
            // The rebuild also refreshed the scene's pick list (object poses), which
            // the selection wireframe is derived from -- re-derive it below from the CURRENT
            // transforms (undo/redo, scripted moves, variant swaps all land here).
            if !self.selected_set.is_empty() {
                self.highlight_dirty = true;
            }
        }
        // Forge lights ILLUMINATE — re-pack the SimpleLights uniform each frame so the 8
        // nearest the camera cast real light on nearby surfaces (not just self-glow). Gated so
        // maps with no forge lights keep their scenario-only lights uploaded at load.
        if let Some(scene) = self.scene_ctl.as_ref() {
            if scene.has_forge_lights() {
                let sl = scene.simple_lights_with_forge(self.camera.pos, anim_time);
                let n = (sl.len() / 20) as u32;
                self.renderer.set_simple_lights(&self.render_state.queue, n, &sl);
            }
        }
        // Keep the highlight following selected objects as they move — but NOT while a modal
        // transform is active (its live wireframe preview owns the highlight buffer then).
        // Only rebuild when an object actually MOVED (highlight_dirty): rebuilding every
        // selected object's wireframe + GPU buffer EVERY frame tanks the framerate with a
        // large (Ctrl+A) selection. Selection-change sites call apply_selection_highlight directly,
        // so an idle selection — however large — costs nothing here.
        if self.highlight_dirty && !self.selected_set.is_empty() && self.xform.is_none() {
            self.apply_selection_highlight();
            self.highlight_dirty = false;
        }
    }

    /// The properties panel reports, EVERY frame it runs, the open team/colour popup
    /// whose rect contains the pointer (None = no popup open under the pointer / popup closed) and
    /// the entry under the pointer (None in the gaps). Drives the state machine; a script-pinned
    /// preview (`preview team ...`) is left alone until `preview off`.
    fn panel_hover_preview(&mut self, inside: Option<color_hover::Popup>, hovered: Option<color_hover::HoverEntry>) {
        // A preview never outlives its selection: deselect-all / click-away / delete / undo of a
        // placement drop the datums from selected_set (many sites) -> end it here, pinned or not.
        let stale = self.hover_preview.as_ref().map_or(false, |(_, pv)| pv.keys().any(|d| !self.selected_set.contains(d)))
            || (self.hover_hidden && self.selected_set.is_empty());
        if stale {
            self.clear_hover_preview();
            return;
        }
        if self.hover_preview_pinned {
            return;
        }
        self.apply_hover(inside, hovered);
    }

    /// One step of the hover rule. Hide rule: the pointer is inside an open popup in
    /// which at least one entry WOULD change the effective colour of a selected, colour-responsive
    /// object (evaluated once on entering the popup). Colour rule: the last hovered entry of that
    /// popup is previewed (objects whose colour it would change get the transient override) and is
    /// KEPT while the pointer crosses the padding between rows; leaving the popup rect or closing it
    /// restores both. Idempotent per (popup, entry): a held pointer costs nothing per frame.
    fn apply_hover(&mut self, inside: Option<color_hover::Popup>, hovered: Option<color_hover::HoverEntry>) {
        // A modal transform owns the highlight buffer (live wireframe preview) -- no hover preview then.
        let inside = if self.xform.is_some() { None } else { inside };
        let entered = self.hover_state.step(inside, hovered);
        if entered {
            self.hover_hidden = match self.hover_state.inside {
                Some(p) => {
                    let sel = self.hover_selection();
                    color_hover::popup_would_change(&sel, p, |d| self.object_responds_to_color(d))
                }
                None => false,
            };
        }
        let entry = self.hover_state.entry;
        if self.hover_preview.as_ref().map(|(e, _)| *e) != entry {
            self.hover_preview = entry.map(|e| {
                let sel = self.hover_selection();
                (e, color_hover::preview_map(sel, e, |d| self.object_responds_to_color(d)))
            });
        }
        self.renderer.set_highlight_hidden(self.hover_hidden);
        // #wire-visible  Keep the selection lane's depth mode in step with the View switch (the
        // renderer starts depth-tested; the switch is persisted, so it can be on from a past run).
        self.renderer.set_highlight_xray(self.wire_xray);
    }

    /// The selection as `(datum, team, colour)` rows (offline forge objects only).
    fn hover_selection(&self) -> Vec<(u32, u8, i32)> {
        self.selected_set.iter()
            .filter_map(|d| self.mvar_meta.get(d).map(|m| (*d, m.team, m.color)))
            .collect()
    }

    /// Does this placed object's render model visibly use the change colour?
    /// (None = not decoded yet -> treated as "no".)
    fn object_responds_to_color(&self, d: u32) -> Option<bool> {
        let mode = self.mvar_objects.iter().find(|o| o.datum == d).map(|o| o.mode_tag)?;
        self.scene_ctl.as_ref()?.object_uses_change_color(mode)
    }

    /// The script hook -- the same step as a pointer that entered `entry`'s popup and
    /// is hovering `entry`. Pinned afterwards so the panel's per-frame reconciliation keeps it.
    fn set_hover_preview(&mut self, entry: color_hover::HoverEntry) {
        self.clear_hover_preview();
        self.apply_hover(Some(entry.popup()), Some(entry));
        self.hover_preview_pinned = true;
    }

    /// End the preview -- objects back to their real colours (next tick_scene sees the
    /// un-overridden forge_colors), wireframe visible again on THIS frame (render runs after the UI).
    fn clear_hover_preview(&mut self) {
        self.hover_preview_pinned = false;
        self.hover_preview = None;
        self.hover_state = Default::default();
        if std::mem::replace(&mut self.hover_hidden, false) {
            self.renderer.set_highlight_hidden(false);
        }
    }

    /// Inject this frame's scripted pointer events (see `sim_pointer` / `sim_click`).
    fn drive_sim_pointer(&mut self, raw: &mut egui::RawInput) {
        let ev = &mut raw.events;
        if let Some((pos, left)) = self.sim_pointer {
            ev.push(egui::Event::PointerMoved(pos));
            self.sim_pointer = (left > 1).then_some((pos, left - 1));
        }
        if let Some((pos, phase)) = self.sim_click {
            ev.push(egui::Event::PointerMoved(pos));
            ev.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: phase == 0, modifiers: Default::default() });
            self.sim_click = (phase == 0).then_some((pos, 1));
        }
        // #construct-h4: one queued keyboard batch per frame (`preview key` / `preview type`).
        if !self.sim_events.is_empty() {
            ev.extend(self.sim_events.remove(0));
        }
        // #construct-h4: the press-drag. One frame per step so egui sees real motion between the
        // press and the release (a press+release at one spot is a click, and the press-drag tools
        // -- Construct Circle / Square -- would never start).
        if let Some((a, b, phase)) = self.sim_drag {
            const STEPS: u8 = 5;
            let t = (phase as f32 / STEPS as f32).clamp(0.0, 1.0);
            let pos = a + (b - a) * t;
            ev.push(egui::Event::PointerMoved(pos));
            if phase == 0 {
                ev.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
            } else if phase >= STEPS {
                ev.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
            }
            self.sim_drag = (phase < STEPS).then_some((a, b, phase + 1));
        }
    }

    /// `preview where` -- the combo/popup rects + the pointer egui currently sees.
    fn hover_where_status(&self) -> String {
        let r = |o: Option<egui::Rect>| o.map(|r| format!("[{:.0},{:.0} .. {:.0},{:.0}]", r.min.x, r.min.y, r.max.x, r.max.y)).unwrap_or_else(|| "-".into());
        let d = self.hover_debug_rects;
        let ptr = self.dbg_pointer.map(|p| format!("{:.0},{:.0}", p.x, p.y)).unwrap_or_else(|| "none".into());
        format!("pointer {ptr} | viewport {} | team button {} popup {} | color button {} popup {} | state inside={:?} entry={:?} hidden={}",
            r(self.dbg_viewport),
            r(d.team_button), r(d.team_popup), r(d.color_button), r(d.color_popup), self.hover_state.inside, self.hover_state.entry, self.hover_hidden)
    }

    /// One-line report for the `preview` script verb.
    fn hover_preview_status(&self) -> String {
        let wire = if self.hover_hidden { "wireframe hidden" } else { "wireframe shown" };
        let pinned = if self.hover_preview_pinned { " (pinned by script; `preview off` ends it)" } else { "" };
        match (self.hover_state.inside, self.hover_preview.as_ref()) {
            (None, _) => "preview: off (wireframe shown, no colour override)".to_string(),
            (Some(p), Some((e, pv))) => {
                let mut ds: Vec<u32> = pv.keys().copied().collect();
                ds.sort();
                let list: Vec<String> = ds.iter().map(|d| format!("0x{d:08X}")).collect();
                format!("preview: in the {} popup, {} on {} object(s) [{}] -- {wire}{pinned}",
                    if p == color_hover::Popup::Team { "team" } else { "color" }, e.describe(), pv.len(), list.join(", "))
            }
            (Some(p), None) => format!("preview: in the {} popup, no entry hovered -- {wire}{pinned}",
                if p == color_hover::Popup::Team { "team" } else { "color" }),
        }
    }

    /// Draw the selection highlight as each selected object's MESH WIREFRAME
    /// (hugs the geometry), falling back to the AABB box when a mesh isn't available.
    /// Renders the full multi-select set (shift+click); empty set clears the highlight.
    fn apply_selection_highlight(&mut self) {
        // Selection-dependent overlays (boundary zones) need a refresh; do it next frame via a
        // flag so this stays cheap and can't recurse into rebuild_overlays.
        self.overlays_dirty = true;
        // NOTE: set_highlight_lines and set_highlight share ONE gpu buffer
        // (highlight_vbuf) — calling both would have the second wipe the first, so
        // exactly one call must run per invocation.
        if self.selected_set.is_empty() {
            self.renderer.set_highlight(&self.render_state.device, None);
            return;
        }
        let mut lines: Vec<[f32; 3]> = Vec::new();
        let mut fallback_box = None;
        if let Some(s) = self.objscene() {
            for &d in &self.selected_set {
                if let Some(w) = s.selection_wireframe(d) {
                    lines.extend(w);
                } else if fallback_box.is_none() {
                    fallback_box = s.aabb_of(d);
                }
            }
        }
        if !lines.is_empty() {
            self.renderer.set_highlight_lines(&self.render_state.device, Some(&lines));
        } else {
            self.renderer.set_highlight(&self.render_state.device, fallback_box);
        }
    }

    // ===================== undo/redo (offline edits) =====================

    /// Capture the full offline editing state for undo/redo.
    fn edit_snapshot(&self) -> EditSnapshot {
        EditSnapshot {
            mvar_objects: self.mvar_objects.clone(),
            local_objects: self.local_objects.clone(),
            mvar_meta: self.mvar_meta.clone(),
            mvar_colors: self.mvar_colors.clone(),
            selected_set: self.selected_set.clone(),
            selected_datum: self.selected_datum,
            shapes: self.shapes.clone(),
            sel_shape: self.sel_shape,
        }
    }

    /// Push the current state onto the undo stack (call BEFORE a mutating edit). Clears redo.
    fn push_edit_undo(&mut self) {
        self.edit_undo.push(self.edit_snapshot());
        if self.edit_undo.len() > 64 {
            self.edit_undo.remove(0);
        }
        self.edit_redo.clear();
    }

    /// Replace the offline editing state with a snapshot and force a render rebuild.
    fn restore_edit_snapshot(&mut self, s: EditSnapshot) {
        self.mvar_objects = s.mvar_objects;
        self.local_objects = s.local_objects;
        self.mvar_meta = s.mvar_meta;
        self.mvar_colors = s.mvar_colors;
        self.selected_set = s.selected_set;
        self.selected_datum = s.selected_datum;
        self.shapes = s.shapes;
        self.sel_shape = s.sel_shape;
        self.overlays_dirty = true;
        if let Some(sc) = self.objscene_mut() {
            sc.invalidate(); // force maybe_rebuild next tick (positions/colours changed)
        }
        self.prop_snapshotted_for = None;
        // NOT apply_selection_highlight() here -- the scene's picks still carry the
        // pre-undo poses until that rebuild runs, so the wireframe would be drawn where the
        // object used to be. Prune + defer instead.
        self.revalidate_selection_after_restore();
    }

    // ===================== modal transform gizmo =====================

    /// Selected datums that live in an OFFLINE, mutable object list (mvar or locally-placed).
    /// These are the only objects the modal grab/rotate/duplicate operates on.
    fn movable_datums(&self) -> Vec<u32> {
        self.selected_set
            .iter()
            .copied()
            .filter(|&d| {
                self.mvar_objects.iter().any(|o| o.datum == d)
                    || self.local_objects.iter().any(|o| o.datum == d)
            })
            .collect()
    }

    /// Current (pos, fwd, up) of an offline object by datum.
    fn movable_pose(&self, datum: u32) -> Option<(glam::Vec3, glam::Vec3, glam::Vec3)> {
        self.mvar_objects
            .iter()
            .chain(self.local_objects.iter())
            .find(|o| o.datum == datum)
            .map(|o| (glam::Vec3::from(o.pos), glam::Vec3::from(o.fwd), glam::Vec3::from(o.up)))
    }

    /// (tag path, object class) of an obj tag, memoised per tag. Empty strings when
    /// the cache cannot name it (no scene / bogus tag).
    fn tag_ident(&self, obj_tag: u32) -> (String, String) {
        if let Some(v) = self.tag_ident_cache.borrow().get(&obj_tag) {
            return v.clone();
        }
        let v = match self.objscene() {
            Some(s) if obj_tag != 0 && obj_tag != 0xFFFF_FFFF => (
                s.tag_name_of(obj_tag),
                obj_identity::class_from_code(&s.raw_tag_class(obj_tag).unwrap_or_default()),
            ),
            _ => (String::new(), String::new()),
        };
        self.tag_ident_cache.borrow_mut().insert(obj_tag, v.clone());
        v
    }

    /// What an object IS — tag path / class / tag id / placement — for ANY
    /// rendered datum: scenario (0xE…), variant (.mvar) or locally placed. None = unknown datum.
    fn object_identity(&self, datum: u32) -> Option<obj_identity::ObjIdentity> {
        use obj_identity::{ObjIdentity, ObjSource};
        if let Some(o) = self.scenario_objects.iter().find(|o| o.datum == datum) {
            let (tag_path, mut class) = self.tag_ident(o.primary_tag);
            let (cat, palette_index, name_index) = self.scenario_ident.get(&datum).copied().unwrap_or((u16::MAX, -1, -1));
            if class.is_empty() {
                class = obj_identity::category_name(cat).to_string();
            }
            return Some(ObjIdentity {
                datum,
                obj_tag: o.primary_tag,
                mode_tag: o.mode_tag,
                tag_path,
                class,
                source: ObjSource::Scenario { index: datum.wrapping_sub(0xE000_0000), palette_index, name_index },
            });
        }
        if let Some(o) = self.mvar_objects.iter().find(|o| o.datum == datum) {
            let (tag_path, class) = self.tag_ident(o.primary_tag);
            let (palette_name, slot) = match self.mvar_meta.get(&datum) {
                Some(m) => (m.name.clone(), if m.slot == 0xFFFF { None } else { Some(m.slot) }),
                None => (String::new(), None),
            };
            return Some(ObjIdentity { datum, obj_tag: o.primary_tag, mode_tag: o.mode_tag, tag_path, class, source: ObjSource::Variant { palette_name, slot } });
        }
        if let Some(o) = self.local_objects.iter().find(|o| o.datum == datum) {
            let (tag_path, class) = self.tag_ident(o.primary_tag);
            let palette_name = self
                .static_palette
                .iter()
                .find(|(t, _)| *t == o.primary_tag)
                .map(|(_, n)| n.split(" · ").last().unwrap_or(n).to_string())
                .unwrap_or_default();
            return Some(ObjIdentity { datum, obj_tag: o.primary_tag, mode_tag: o.mode_tag, tag_path, class, source: ObjSource::Placed { palette_name } });
        }
        None
    }

    /// Write a new pose into whichever offline list holds this datum (and mvar_meta).
    fn set_movable_pose(&mut self, datum: u32, pos: glam::Vec3, fwd: glam::Vec3, up: glam::Vec3) {
        for o in self.mvar_objects.iter_mut().chain(self.local_objects.iter_mut()) {
            if o.datum == datum {
                o.pos = pos.into();
                o.fwd = fwd.into();
                o.up = up.into();
                break;
            }
        }
        if let Some(m) = self.mvar_meta.get_mut(&datum) {
            m.pos = pos.into();
        }
        // An object moved → the selection highlight must follow it. Flag it so
        // update()'s per-frame path rebuilds ONCE next frame instead of rebuilding every frame.
        self.highlight_dirty = true;
    }

    // ===================== construction geometry =====================

    /// Anchors offered for one object, in the current anchor mode.
    /// #construct-h4: through `objscene()`, so the anchors exist on a Halo 4 map too -- this went
    /// to `scene_ctl` and returned NOTHING on Halo 4, which left the Construct tool with no
    /// corner / face / centre markers to click at all (and so no face for a Coincident mate).
    fn object_anchors(&self, datum: u32) -> Vec<construct::Anchor> {
        self.objscene()
            .and_then(|s| s.object_obb(datum))
            .map(|o| o.anchors(Some(datum), self.anchor_mode))
            .unwrap_or_default()
    }

    /// The box the construction tools treat as "the selection". A single object keeps its
    /// ORIENTED box (so a rotated block's corners are its own corners); a multi-object
    /// selection uses the axis-aligned box enclosing them all, whose centre is the group
    /// centre the user means by "the middle of what I picked".
    fn selection_obb(&self) -> Option<construct::Obb> {
        let scene = self.objscene()?;
        let sel: Vec<u32> = if self.selected_set.is_empty() {
            self.selected_datum.into_iter().collect()
        } else {
            self.selected_set.clone()
        };
        match sel.len() {
            0 => None,
            1 => scene.object_obb(sel[0]),
            _ => {
                let mut mn = glam::Vec3::splat(f32::INFINITY);
                let mut mx = glam::Vec3::splat(f32::NEG_INFINITY);
                for d in sel {
                    if let Some((a, b)) = scene.aabb_of(d) {
                        mn = mn.min(glam::Vec3::from(a));
                        mx = mx.max(glam::Vec3::from(b));
                    }
                }
                mn.is_finite().then(|| construct::Obb::from_aabb(mn, mx))
            }
        }
    }

    /// Every point a move can snap to, derived from the current guides.
    fn guide_targets(&self) -> Vec<construct::SnapTarget> {
        construct::snap_targets(&self.guides, 0.05)
    }

    /// #construct-h4: one line of Construct-tool state -- what a viewport click would do.
    fn construct_status(&self) -> String {
        let n_int = self.guide_targets().iter().filter(|t| t.kind == construct::SnapKind::Intersection).count();
        let face_a = match &self.cad_face_a {
            Some((c, n, l)) => format!("{l} at ({:.2},{:.2},{:.2}) n=({:.2},{:.2},{:.2})", c.x, c.y, c.z, n.x, n.y, n.z),
            None => "none".to_string(),
        };
        let pending = match (self.construct_pending, self.line_pending) {
            (Some(a), _) => format!(" | half-drawn guide from {} ({:.2},{:.2},{:.2})", a.kind.label(), a.pos.x, a.pos.y, a.pos.z),
            (_, Some(p)) => format!(" | line array started at ({:.2},{:.2},{:.2})", p.x, p.y, p.z),
            _ => String::new(),
        };
        let mut out = format!(
            "construct: tool {} | op {} | snap {} | {} guide(s), {n_int} intersection(s) | {} shape(s) | face A = {face_a}{pending}",
            if self.construct_mode { "on" } else { "off" },
            self.construct_op.label(),
            self.anchor_mode.label(),
            self.guides.len(),
            self.shapes.len(),
        );
        // The guide / shape lists with their real coordinates -- the panel's lists, in numbers.
        for (i, g) in self.guides.iter().enumerate() {
            out.push_str(&format!(
                "\n  guide {i}: {} ({:.3},{:.3},{:.3}) -> {} ({:.3},{:.3},{:.3})  {:.3} wu",
                g.a.kind.label(), g.a.pos.x, g.a.pos.y, g.a.pos.z,
                g.b.kind.label(), g.b.pos.x, g.b.pos.y, g.b.pos.z,
                g.length()
            ));
        }
        for (i, s) in self.shapes.iter().enumerate() {
            out.push_str(&format!(
                "\n  shape {i}: {} centre ({:.3},{:.3},{:.3}) radius {:.3} wu normal ({:.2},{:.2},{:.2})",
                s.kind.label(), s.center.x, s.center.y, s.center.z, s.radius, s.normal.x, s.normal.y, s.normal.z
            ));
        }
        out
    }

    /// #construct-h4: the `construct` script verb. The Construct tool is otherwise reachable only
    /// by the toolbar / `C` / the CAD panel, so nothing could drive (or test) the CAD workflow
    /// from a script; this is that state, game-agnostic like the ops themselves.
    fn construct_command(&mut self, c: script::ConstructCmd) -> Result<String, String> {
        use script::ConstructCmd;
        match c {
            ConstructCmd::Get => Ok(self.construct_status()),
            ConstructCmd::Enable(on) => {
                self.tool_mode = if on { ToolMode::Construct } else { ToolMode::Select };
                // The per-frame tool sync would arm this next frame; do it now so a script can
                // arm the tool and click in the same batch.
                self.construct_mode = on;
                if !on {
                    self.construct_pending = None;
                    self.construct_hover = None;
                }
                self.overlays_dirty = true;
                Ok(self.construct_status())
            }
            ConstructCmd::Op(name) => {
                let op = match name.as_str() {
                    "guide" | "guides" => ConstructOp::Guide,
                    "circle" => ConstructOp::Circle,
                    "square" => ConstructOp::Square,
                    "coincident" | "mate" | "face" => ConstructOp::Coincident,
                    "anchor" | "anchorto" | "centre" | "center" => ConstructOp::AnchorTo,
                    "mirror" => ConstructOp::Mirror,
                    "line" | "linearray" | "array" => ConstructOp::Line,
                    x => return Err(format!("construct op: unknown '{x}' (guide, circle, square, coincident, anchor, mirror, line)")),
                };
                self.construct_op = op;
                self.construct_pending = None;
                self.cad_face_a = None;
                self.line_pending = None;
                Ok(self.construct_status())
            }
            ConstructCmd::Mode(name) => {
                let m = match name.as_str() {
                    "all" | "any" => construct::AnchorMode::All,
                    "corners" | "corner" => construct::AnchorMode::Corners,
                    "edges" | "edge" => construct::AnchorMode::Edges,
                    "faces" | "face" => construct::AnchorMode::Faces,
                    "centres" | "centers" | "centre" | "center" => construct::AnchorMode::Centers,
                    x => return Err(format!("construct snap: unknown '{x}' (all, corners, edges, faces, centres)")),
                };
                self.anchor_mode = m;
                Ok(self.construct_status())
            }
            ConstructCmd::Anchors(d) => {
                let anchors = self.object_anchors(d);
                if anchors.is_empty() {
                    return Err(format!("construct anchors: 0x{d:08X} offers no anchors (no bounds for it in the scene)"));
                }
                let mut out = format!("0x{d:08X}: {} anchor(s) in mode {}\n", anchors.len(), self.anchor_mode.label());
                for a in &anchors {
                    out.push_str(&format!("  {:<14} ({:.3}, {:.3}, {:.3})\n", a.kind.label(), a.pos.x, a.pos.y, a.pos.z));
                }
                Ok(out)
            }
            ConstructCmd::Clear => {
                let (g, s) = (self.guides.len(), self.shapes.len());
                if g + s > 0 {
                    self.push_edit_undo();
                }
                self.guides.clear();
                self.shapes.clear();
                self.sel_guide = None;
                self.sel_shape = None;
                self.construct_pending = None;
                self.overlays_dirty = true;
                Ok(format!("construct: cleared {g} guide(s) and {s} shape(s)"))
            }
        }
    }

    /// Re-derive each guide endpoint from the object it was anchored to, so guides stay
    /// attached when their object is moved or rotated (a construction line that drifts off
    /// its corner is worse than no line at all). Endpoints not tied to an object -- group
    /// centres, intersections, free points -- keep the position they were placed at.
    fn refresh_guides(&mut self) {
        let Some(scene) = self.objscene() else { return };
        let mut fixes: Vec<(usize, bool, glam::Vec3)> = Vec::new();
        for (i, g) in self.guides.iter().enumerate() {
            for (is_b, a) in [(false, &g.a), (true, &g.b)] {
                let Some(d) = a.datum else { continue };
                let Some(o) = scene.object_obb(d) else { continue };
                let p = match a.kind {
                    construct::AnchorKind::Center => o.c,
                    construct::AnchorKind::Face(i) => o.face_center(i),
                    construct::AnchorKind::Corner(i) => o.corner(i),
                    construct::AnchorKind::Edge(i) => o.edge_mid(i),
                    construct::AnchorKind::Point => continue,
                };
                if (p - a.pos).length() > 1e-4 {
                    fixes.push((i, is_b, p));
                }
            }
        }
        for (i, is_b, p) in fixes {
            let g = &mut self.guides[i];
            if is_b { g.b.pos = p } else { g.a.pos = p }
        }
    }

    /// Nearest anchor to the cursor in SCREEN space: the anchors of the object under the
    /// cursor, the selection centre, and every existing guide point. Screen space (not world
    /// distance) is what makes picking feel right -- a far corner and a near one compete by
    /// how close they look, exactly like the gizmo handles.
    fn pick_anchor(&self, rect: egui::Rect, ptr: (f32, f32)) -> Option<construct::Anchor> {
        const GRAB_PX: f32 = 26.0;
        let (near, dir) = self.cursor_ray(rect, ptr.0, ptr.1);
        let mut cands: Vec<construct::Anchor> = Vec::new();
        if let Some(d) = self.objscene().and_then(|s| s.pick(near, dir)) {
            cands.extend(self.object_anchors(d));
        }
        // The group centre is only meaningful for a real group.
        if self.selected_set.len() > 1 {
            if let Some(o) = self.selection_obb() {
                cands.push(construct::Anchor { datum: None, kind: construct::AnchorKind::Center, pos: o.c });
            }
        }
        // Existing guide points, so lines can be chained off an intersection.
        for t in self.guide_targets() {
            cands.push(construct::Anchor::point(t.pos));
        }
        let mut best: Option<(f32, construct::Anchor)> = None;
        for a in cands {
            let Some((sx, sy)) = self.world_to_screen(a.pos, rect) else { continue };
            let d2 = (sx - ptr.0).powi(2) + (sy - ptr.1).powi(2);
            if d2 > GRAB_PX * GRAB_PX {
                continue;
            }
            if best.map_or(true, |(b, _)| d2 < b) {
                best = Some((d2, a));
            }
        }
        best.map(|(_, a)| a)
    }

    /// Nearest guide point to a world position, within `snap_range`. Used to pull a move
    /// onto a construction point.
    fn nearest_guide_point(&self, p: glam::Vec3) -> Option<glam::Vec3> {
        let mut best: Option<(f32, glam::Vec3)> = None;
        for t in self.guide_targets() {
            let d = (t.pos - p).length();
            if d > self.snap_range {
                continue;
            }
            if best.map_or(true, |(b, _)| d < b) {
                best = Some((d, t.pos));
            }
        }
        best.map(|(_, p)| p)
    }

    /// Translate the whole selection so the centre of its box lands EXACTLY on `target`.
    /// This is the "put it in the middle of that" operation.
    fn anchor_selection_to(&mut self, target: glam::Vec3) {
        let datums = self.movable_datums();
        if datums.is_empty() {
            self.status = "Anchor: select a movable object first".into();
            return;
        }
        let Some(obb) = self.selection_obb() else {
            self.status = "Anchor: selection has no bounds".into();
            return;
        };
        let delta = target - obb.c;
        self.push_edit_undo();
        for d in datums {
            if let Some((p, f, u)) = self.movable_pose(d) {
                self.set_movable_pose(d, p + delta, f, u);
            }
        }
        self.refresh_guides();
        self.overlays_dirty = true;
        self.status = format!("Anchored selection to ({:.2}, {:.2}, {:.2})", target.x, target.y, target.z);
    }

    /// Mirror the selection across the axis-aligned plane through `origin`. With `mirror_copy`
    /// the original stays and a mirrored duplicate is made -- the symmetric-map workflow
    /// (build one half, mirror it) a Griffball court needs.
    fn mirror_selection(&mut self, origin: glam::Vec3) {
        let n = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][self.mirror_axis.min(2)];
        let mut datums = self.movable_datums();
        if datums.is_empty() {
            self.status = "Mirror: select a movable object first".into();
            return;
        }
        self.push_edit_undo();
        if self.mirror_copy {
            let dups = self.duplicate_selection();
            if dups.is_empty() {
                self.status = "Mirror: nothing could be duplicated".into();
                return;
            }
            self.selected_set = dups.clone();
            self.selected_datum = dups.last().copied();
            datums = dups;
        }
        for d in datums {
            if let Some((p, f, u)) = self.movable_pose(d) {
                let np = construct::mirror_point(p, origin, n);
                // Reflecting both basis vectors keeps the object's facing consistent with the
                // mirrored world. It flips handedness, which is exactly what a mirror does.
                let nf = construct::mirror_dir(f, n).normalize_or_zero();
                let nu = construct::mirror_dir(u, n).normalize_or_zero();
                let (nf, nu) = if nf.length_squared() < 1e-6 || nu.length_squared() < 1e-6 { (f, u) } else { (nf, nu) };
                self.set_movable_pose(d, np, nf, nu);
            }
        }
        self.apply_selection_highlight();
        self.refresh_guides();
        self.overlays_dirty = true;
        let ax = ["X", "Y", "Z"][self.mirror_axis.min(2)];
        self.status = format!("Mirrored selection across {ax} = {:.2}", origin[self.mirror_axis.min(2)]);
    }

    /// World point under the cursor, for starting a shape on open ground.
    /// #construct-h4: `objscene()`, so a Circle / Square lands on the Halo 4 map surface instead
    /// of floating 10 wu in front of the camera.
    fn ground_point(&self, rect: egui::Rect, p: (f32, f32)) -> glam::Vec3 {
        let (near, dir) = self.cursor_ray(rect, p.0, p.1);
        self.objscene()
            .and_then(|s| s.raycast_scene(near, dir))
            .unwrap_or(near + dir * 10.0)
    }

    /// How close (world units) a click must be to a shape's centre to GRAB it. Scaled off the
    /// snap range so it tracks the user's own tolerance setting.
    fn shape_grab_radius(&self) -> f32 {
        self.snap_range.max(0.5)
    }

    /// The plane a new shape is drawn in: the configured axis, or -- when the click landed on a
    /// FACE anchor and the user asked for "clicked face" -- that face's own plane, which is how
    /// a ring gets laid out flat ON a surface.
    fn shape_plane_normal(&self, a: &construct::Anchor) -> glam::Vec3 {
        match self.cad_shape_plane {
            0 => glam::Vec3::X,
            1 => glam::Vec3::Y,
            2 => glam::Vec3::Z,
            _ => self.face_of_anchor(a).map(|(_, n, _)| n).unwrap_or(glam::Vec3::Z),
        }
    }

    /// An anchor as a FACE -- (centre, outward normal, label). `None` unless it is a
    /// face anchor of an object we can take an oriented box from, because a coincident
    /// constraint needs a real plane, not just a point.
    fn face_of_anchor(&self, a: &construct::Anchor) -> Option<(glam::Vec3, glam::Vec3, String)> {
        let construct::AnchorKind::Face(i) = a.kind else { return None };
        let d = a.datum?;
        let obb = self.objscene()?.object_obb(d)?;
        let n = obb.face_normal(i);
        (n.length_squared() > 0.5).then(|| (obb.face_center(i), n, format!("{} face of {d:#x}", ["-X", "+X", "-Y", "+Y", "-Z", "+Z"][(i as usize).min(5)])))
    }

    /// Fill a shape's edge with the current selection -- `count` copies spaced
    /// evenly around the outline. The ORIGINAL is moved to the first point and copies fill the
    /// rest, so "draw a circle, pick a block, fill" gives exactly `count` evenly spaced blocks.
    /// With `rotate`, each copy is turned about the shape's normal by its angle around the
    /// outline, so a ring of cover faces outward instead of all pointing the same way.
    fn array_along_shape(&mut self, si: usize, count: u32, rotate: bool) {
        let Some(sh) = self.shapes.get(si).copied() else {
            self.status = "Array: select a shape first".into();
            return;
        };
        if sh.radius < 1e-3 {
            self.status = "Array: that shape has no size".into();
            return;
        }
        if self.movable_datums().is_empty() {
            self.status = "Array: select the object to duplicate first".into();
            return;
        }
        let pts = sh.perimeter_points(count.max(1));
        let Some(obb) = self.selection_obb() else {
            self.status = "Array: selection has no bounds".into();
            return;
        };
        self.push_edit_undo();
        let n = sh.normal.normalize_or_zero();
        let base_out = pts[0].1;
        // 1) move the original onto the first point
        let d0 = pts[0].0 - obb.c;
        for d in self.movable_datums() {
            if let Some((p, f, u)) = self.movable_pose(d) {
                self.set_movable_pose(d, p + d0, f, u);
                if let Some(sc) = self.objscene_mut() { sc.translate_pick(d, d0); }
            }
        }
        // 2) duplicate for the remaining points
        let mut made = 0usize;
        for (pos, out) in pts.iter().skip(1) {
            let dups = self.duplicate_selection();
            if dups.is_empty() { break; }
            // Placement comes from the TESTED helper so the app and the unit
            // tests can never drift apart on this maths.
            let (q, delta) = construct::array_copy_transform(sh.center, n, pts[0].0, base_out, *pos, *out, rotate);
            for d in &dups {
                if let Some((p, f, u)) = self.movable_pose(*d) {
                    let np = sh.center + q * (p - sh.center) + delta;
                    self.set_movable_pose(*d, np, q * f, q * u);
                }
            }
            made += dups.len();
        }
        self.apply_selection_highlight();
        self.refresh_guides();
        self.overlays_dirty = true;
        self.status = format!("Arrayed {} around the {}: {made} copies + the original", if made == 0 { "nothing" } else { "selection" }, sh.kind.label());
    }


    /// Translate the selection so the stored face A becomes coincident with face B.
    /// Only the selection moves — face A should be one of its faces; picking a face of some
    /// other object would translate the selection by that plane's offset, which is why the
    /// pick button records the face the user actually pointed at.
    fn constrain_coincident(&mut self, b_center: glam::Vec3, b_normal: glam::Vec3) {
        let Some((a_center, _a_n, ref label)) = self.cad_face_a.clone() else {
            self.status = "Coincident: click face A first (Snap to = Faces), then the target face".into();
            return;
        };
        let datums = self.movable_datums();
        if datums.is_empty() {
            self.status = "Coincident: select a movable object first".into();
            return;
        }
        let mate = if self.cad_mate_centered { construct::FaceMate::Centered } else { construct::FaceMate::Plane };
        let (q, delta) = construct::face_mate_transform(a_center, _a_n, b_center, b_normal, mate, self.cad_mate_rotate);
        let turned = q.to_array()[3].abs() < 0.999_999;
        if delta.length() < 1e-6 && !turned {
            self.status = format!("Coincident: {label} is already mated");
            return;
        }
        self.push_edit_undo();
        for d in datums {
            if let Some((p, f, u)) = self.movable_pose(d) {
                // rotate about face A's CENTRE so that face stays put while the part swings
                let np = a_center + q * (p - a_center) + delta;
                self.set_movable_pose(d, np, q * f, q * u);
                if let Some(sc) = self.objscene_mut() { sc.translate_pick(d, np - p); }
            }
        }
        // Face A moved (and possibly turned) with the part; drop the stale pick rather than
        // letting a follow-up mate use an out-of-date plane.
        self.cad_face_a = None;
        self.refresh_guides();
        self.overlays_dirty = true;
        let how = if self.cad_mate_centered { "centred on" } else { "flush with" };
        let rot = if turned { ", turned to face it" } else { "" };
        self.status = format!("Coincident: {label} {how} target ({:.2} wu{rot})", delta.length());
    }

    /// Add construction lines to the selection's box. `faces` = the two diagonals of each of
    /// the six faces (each pair crosses at that face's centre); otherwise the four space
    /// diagonals (all four cross at the box centre). Either way the crossing points become
    /// snap targets, which is how a block gets centred exactly.
    fn add_box_diagonals(&mut self, faces: bool) {
        let Some(obb) = self.selection_obb() else {
            self.status = "Diagonals: select an object first".into();
            return;
        };
        // Anchor to the object only for a single-object selection; a group box is not a
        // feature of any one object, so those guides stay where they were drawn.
        let owner = (self.selected_set.len() <= 1).then(|| self.selected_datum).flatten();
        let mut added = 0;
        if faces {
            // Face axis a, sign bit s: the 4 corners with bit(a) == s. A corner's diagonal
            // partner across that face is the one differing in BOTH other bits.
            for a in 0..3u8 {
                let abit = 1u8 << a;
                let other = 7u8 & !abit;
                for s in [0u8, abit] {
                    let mut seen: Vec<u8> = Vec::new();
                    for c in 0..8u8 {
                        if c & abit != s {
                            continue;
                        }
                        let partner = c ^ other;
                        if seen.contains(&partner) {
                            continue;
                        }
                        seen.push(c);
                        self.guides.push(construct::Guide {
                            a: construct::Anchor { datum: owner, kind: construct::AnchorKind::Corner(c), pos: obb.corner(c) },
                            b: construct::Anchor { datum: owner, kind: construct::AnchorKind::Corner(partner), pos: obb.corner(partner) },
                            color: [0.35, 0.85, 1.0],
                        });
                        added += 1;
                    }
                }
            }
        } else {
            for c in 0..4u8 {
                let partner = c ^ 7;
                self.guides.push(construct::Guide {
                    a: construct::Anchor { datum: owner, kind: construct::AnchorKind::Corner(c), pos: obb.corner(c) },
                    b: construct::Anchor { datum: owner, kind: construct::AnchorKind::Corner(partner), pos: obb.corner(partner) },
                    color: [1.0, 0.8, 0.3],
                });
                added += 1;
            }
        }
        self.show_guides = true;
        self.overlays_dirty = true;
        self.status = format!(
            "Added {added} {} diagonals — snap to a crossing to centre",
            if faces { "face" } else { "box" }
        );
    }

    /// Centroid of the movable selection = the transform pivot.
    fn selection_pivot(&self, datums: &[u32]) -> Option<glam::Vec3> {
        if datums.is_empty() {
            return None;
        }
        let mut sum = glam::Vec3::ZERO;
        let mut n = 0.0f32;
        for &d in datums {
            if let Some((p, _, _)) = self.movable_pose(d) {
                sum += p;
                n += 1.0;
            }
        }
        (n > 0.0).then(|| sum / n)
    }

    /// Screen-constant-ish gizmo length (world units) for a pivot at some camera distance.
    fn gizmo_len(&self, pivot: glam::Vec3) -> f32 {
        ((self.camera.pos - pivot).length() * 0.11).max(0.8)
    }

    /// A stable perpendicular unit vector to `a`.
    fn any_perp(a: glam::Vec3) -> glam::Vec3 {
        let p = if a.z.abs() < 0.9 { a.cross(glam::Vec3::Z) } else { a.cross(glam::Vec3::X) };
        p.normalize_or_zero()
    }

    /// Solid gizmo: append an N-sided prism (cylinder) from a→b to the triangle list.
    fn push_cylinder(tris: &mut Vec<([f32; 3], [f32; 3])>, a: glam::Vec3, b: glam::Vec3, radius: f32, col: [f32; 3]) {
        let axis = (b - a).normalize_or_zero();
        let u = Self::any_perp(axis);
        let v = axis.cross(u).normalize_or_zero();
        const N: usize = 10;
        for i in 0..N {
            let a0 = std::f32::consts::TAU * (i as f32 / N as f32);
            let a1 = std::f32::consts::TAU * ((i + 1) as f32 / N as f32);
            let d0 = u * a0.cos() + v * a0.sin();
            let d1 = u * a1.cos() + v * a1.sin();
            let (p0a, p1a) = (a + d0 * radius, a + d1 * radius);
            let (p0b, p1b) = (b + d0 * radius, b + d1 * radius);
            tris.push((p0a.into(), col)); tris.push((p0b.into(), col)); tris.push((p1b.into(), col));
            tris.push((p0a.into(), col)); tris.push((p1b.into(), col)); tris.push((p1a.into(), col));
        }
    }

    /// Solid gizmo: append a cone (base→tip) to the triangle list.
    fn push_cone(tris: &mut Vec<([f32; 3], [f32; 3])>, base: glam::Vec3, tip: glam::Vec3, radius: f32, col: [f32; 3]) {
        let axis = (tip - base).normalize_or_zero();
        let u = Self::any_perp(axis);
        let v = axis.cross(u).normalize_or_zero();
        const N: usize = 12;
        for i in 0..N {
            let a0 = std::f32::consts::TAU * (i as f32 / N as f32);
            let a1 = std::f32::consts::TAU * ((i + 1) as f32 / N as f32);
            let p0 = base + (u * a0.cos() + v * a0.sin()) * radius;
            let p1 = base + (u * a1.cos() + v * a1.sin()) * radius;
            tris.push((p0.into(), col)); tris.push((p1.into(), col)); tris.push((tip.into(), col)); // side
            tris.push((base.into(), col)); tris.push((p1.into(), col)); tris.push((p0.into(), col)); // cap
        }
    }

    /// Solid gizmo: append a torus (tube) around `axis` at `ring_r`, tube radius `tube_r`.
    fn push_torus(tris: &mut Vec<([f32; 3], [f32; 3])>, center: glam::Vec3, axis: glam::Vec3, ring_r: f32, tube_r: f32, col: [f32; 3]) {
        let axis = axis.normalize_or_zero();
        let u = Self::any_perp(axis);
        let v = axis.cross(u).normalize_or_zero();
        const SEG: usize = 40;
        const SIDES: usize = 8;
        let ringpt = |c: glam::Vec3, rad: glam::Vec3, ang: f32| c + (rad * ang.cos() + axis * ang.sin()) * tube_r;
        for s in 0..SEG {
            let a0 = std::f32::consts::TAU * (s as f32 / SEG as f32);
            let a1 = std::f32::consts::TAU * ((s + 1) as f32 / SEG as f32);
            let r0 = u * a0.cos() + v * a0.sin();
            let r1 = u * a1.cos() + v * a1.sin();
            let c0 = center + r0 * ring_r;
            let c1 = center + r1 * ring_r;
            for k in 0..SIDES {
                let b0 = std::f32::consts::TAU * (k as f32 / SIDES as f32);
                let b1 = std::f32::consts::TAU * ((k + 1) as f32 / SIDES as f32);
                let q00 = ringpt(c0, r0, b0);
                let q01 = ringpt(c0, r0, b1);
                let q10 = ringpt(c1, r1, b0);
                let q11 = ringpt(c1, r1, b1);
                tris.push((q00.into(), col)); tris.push((q10.into(), col)); tris.push((q11.into(), col));
                tris.push((q00.into(), col)); tris.push((q11.into(), col)); tris.push((q01.into(), col));
            }
        }
    }


    /// Rebuild the transform gizmo (3 world axes at the selection's centroid), or clear
    /// it when nothing movable is selected. Called every frame; the centroid follows objects
    /// as they move during a live drag.
    fn refresh_gizmo(&mut self) {
        let datums = self.movable_datums();
        let Some(pivot) = self.selection_pivot(&datums) else {
            // No (movable) selection → no gizmo at all. Clear BOTH the lines AND the solid
            // triangles, or a ghost gizmo persists after a delete/deselect.
            if !self.gizmo_last_tris.is_empty() {
                self.gizmo_last_tris.clear();
                self.renderer.set_gizmo_lines(&self.render_state.device, None);
                self.renderer.set_gizmo_tris(&self.render_state.device, None);
            }
            return;
        };
        let len = self.gizmo_len(pivot);
        let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
        // Bright HDR colours so the gizmo reads clearly over any scene (and catches bloom).
        let bright = [
            [3.0f32, 0.10, 0.10], // X red
            [0.10, 3.0, 0.10],    // Y green
            [0.35, 0.70, 3.0],    // Z blue
        ];
        let hover_col = [3.5f32, 3.5, 0.4];
        let shaft_r = len * 0.03; // solid arrow-shaft radius (world units)
        let mut tris: Vec<([f32; 3], [f32; 3])> = Vec::new();

        // ---- Active op: mode-specific overlay (hide the idle gizmo) ----
        if let Some(op) = self.xform.as_ref() {
            match op.kind {
                XformKind::Grab => {
                    // Moving along an axis → ONE solid constraint line (a very thin, very long
                    // cylinder) through the object; free move → nothing.
                    if let Some(ax) = op.axis {
                        let a = Self::xform_axis(op, ax);
                        let col = bright[ax];
                        let big = 1.0e5;
                        Self::push_cylinder(&mut tris, pivot - a * big, pivot + a * big, shaft_r * 0.5, col);
                    }
                }
                XformKind::Rotate => {
                    // Rotating → solid rotate torus(es): the locked axis only when locked, else all.
                    let ring_r = len * 0.9;
                    for i in 0..3 {
                        if let Some(ax) = op.axis {
                            if ax != i {
                                continue;
                            }
                        }
                        Self::push_torus(&mut tris, pivot, axes[i], ring_r, shaft_r * 1.4, bright[i]);
                    }
                }
            }
            self.upload_gizmo_tris(tris);
            return;
        }

        // ---- Idle: gizmo chosen by the toolbar mode. Move → arrow handles; Rotate → rings;
        // Select → nothing (selection only).
        let hov = self.gizmo_hover;
        match self.tool_mode {
            ToolMode::Move => {
                for i in 0..3 {
                    let a = axes[i];
                    let col = if hov == Some(GizmoHit::Move(i)) { hover_col } else { bright[i] };
                    let cone_base = pivot + a * (len * 0.78);
                    let tip = pivot + a * len;
                    Self::push_cylinder(&mut tris, pivot, cone_base, shaft_r, col);
                    Self::push_cone(&mut tris, cone_base, tip, shaft_r * 3.0, col);
                }
            }
            ToolMode::Rotate => {
                let ring_r = len * 0.9;
                for i in 0..3 {
                    let col = if hov == Some(GizmoHit::Rotate(i)) { hover_col } else { bright[i] };
                    Self::push_torus(&mut tris, pivot, axes[i], ring_r, shaft_r * 1.4, col);
                }
            }
            // Select and Construct draw no gizmo — both are click-driven.
            ToolMode::Select | ToolMode::Construct => {}
        }
        self.upload_gizmo_tris(tris);
    }

    /// Upload the gizmo triangles only when they differ from last frame's (an idle
    /// selection re-generates the identical list every frame; the camera moving changes `len`).
    fn upload_gizmo_tris(&mut self, tris: Vec<([f32; 3], [f32; 3])>) {
        if tris == self.gizmo_last_tris {
            return;
        }
        self.renderer.set_gizmo_lines(&self.render_state.device, None);
        self.renderer.set_gizmo_tris(&self.render_state.device, (!tris.is_empty()).then_some(tris.as_slice()));
        self.gizmo_last_tris = tris;
    }

    /// Box-select: replace the selection with every offline object whose CENTRE projects
    /// inside the screen rubber-band rect. A tiny rect is treated as a no-op (that's a click).
    fn box_select(&mut self, a: egui::Pos2, b: egui::Pos2, rect: egui::Rect, additive: bool) {
        let r = egui::Rect::from_two_pos(a, b);
        if r.width() < 4.0 && r.height() < 4.0 {
            return;
        }
        // Shift-box ADDS to the current selection rather than replacing it, so several
        // rubber-band drags can accumulate a selection. Start from the existing set when additive.
        let mut hits = if additive { self.selected_set.clone() } else { Vec::new() };
        // Select an object if ANY part of it overlaps the marquee, not just its center.
        // Project the object's world AABB (all 8 corners) to screen, take their bounding rect, and
        // select when that rect INTERSECTS the marquee. Falls back to the center point when the
        // object has no decoded AABB yet.
        for (datum, pos, aabb) in self.offline_object_bounds() {
            if hits.contains(&datum) {
                continue;
            }
            let overlaps = if let Some((mn, mx)) = aabb {
                // Screen-space bounding rect of the 8 world-AABB corners.
                let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                let mut any = false;
                for i in 0..8 {
                    let c = glam::Vec3::new(
                        if i & 1 == 0 { mn[0] } else { mx[0] },
                        if i & 2 == 0 { mn[1] } else { mx[1] },
                        if i & 4 == 0 { mn[2] } else { mx[2] },
                    );
                    if let Some((sx, sy)) = self.world_to_screen(c, rect) {
                        any = true;
                        minx = minx.min(sx); miny = miny.min(sy); maxx = maxx.max(sx); maxy = maxy.max(sy);
                    }
                }
                any && r.intersects(egui::Rect::from_min_max(egui::pos2(minx, miny), egui::pos2(maxx, maxy)))
            } else {
                // Fallback: the projected center point.
                self.world_to_screen(glam::Vec3::from(pos), rect)
                    .map_or(false, |(sx, sy)| r.contains(egui::pos2(sx, sy)))
            };
            if overlaps {
                hits.push(datum);
            }
        }
        self.selected_set = hits;
        self.selected_datum = self.selected_set.last().copied();
        self.apply_selection_highlight();
        self.status = format!("Box-selected {} object(s)", self.selected_set.len());
    }

    /// Every offline (variant / placed) object with its centre and world AABB --
    /// the scene's pick bounds, which for a hidden block are its PHYSICS HULL (not the 0.1 wu
    /// render nub), so a rubber band over a row of invisible blockers takes all of them. Shared by
    /// the viewport marquee and the scripted `select box`.
    fn offline_object_bounds(&self) -> Vec<(u32, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        self.mvar_objects
            .iter()
            .chain(self.local_objects.iter())
            .map(|o| (o.datum, o.pos, self.objscene().and_then(|s| s.aabb_of(o.datum))))
            .collect()
    }

    /// World-space box select (script `select box`): an object is taken when its
    /// AABB intersects the box (centre fallback when it has no decoded bounds yet).
    fn select_world_box(&mut self, bmin: [f32; 3], bmax: [f32; 3], additive: bool) -> usize {
        let mut hits = if additive { self.selected_set.clone() } else { Vec::new() };
        for (datum, pos, aabb) in self.offline_object_bounds() {
            if hits.contains(&datum) {
                continue;
            }
            if physics_outlines::box_overlaps(bmin, bmax, pos, aabb) {
                hits.push(datum);
            }
        }
        self.selected_set = hits;
        self.selected_datum = self.selected_set.last().copied();
        self.apply_selection_highlight();
        self.selected_set.len()
    }

    /// Minimum distance from a world point to the cursor ray (ahead of the camera only).
    fn point_ray_dist(pt: glam::Vec3, near: glam::Vec3, dir: glam::Vec3) -> Option<f32> {
        let w = pt - near;
        let t = w.dot(dir);
        (t > 0.0).then(|| (w - dir * t).length())
    }

    /// What the cursor ray grabs on the gizmo — a move axis handle or a rotate ring —
    /// whichever is nearest within its grab radius. Rings use `len` as radius (drawn at 0.9·len).
    fn gizmo_pick(&self, pivot: glam::Vec3, len: f32, near: glam::Vec3, dir: glam::Vec3) -> Option<GizmoHit> {
        let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
        let mut best: Option<(f32, GizmoHit)> = None;
        let consider = |d: f32, hit: GizmoHit, thresh: f32, best: &mut Option<(f32, GizmoHit)>| {
            if d < thresh && best.map(|(bd, _)| d < bd).unwrap_or(true) {
                *best = Some((d, hit));
            }
        };
        // Only the ELEMENTS the current toolbar mode draws are grabbable.
        if self.tool_mode == ToolMode::Move {
            let move_thresh = len * 0.16;
            for (i, a) in axes.iter().enumerate() {
                let mut mind = f32::INFINITY;
                for k in 1..=10 {
                    let pt = pivot + *a * (len * (k as f32 / 10.0));
                    if let Some(d) = Self::point_ray_dist(pt, near, dir) {
                        mind = mind.min(d);
                    }
                }
                consider(mind, GizmoHit::Move(i), move_thresh, &mut best);
            }
        }
        if self.tool_mode == ToolMode::Rotate {
            let ring_r = len * 0.9;
            let ring_thresh = len * 0.12;
            for (i, a) in axes.iter().enumerate() {
                let perp = if a.z.abs() < 0.9 { glam::Vec3::Z } else { glam::Vec3::X };
                let u = a.cross(perp).normalize_or_zero();
                let v = a.cross(u).normalize_or_zero();
                let mut mind = f32::INFINITY;
                for k in 0..48 {
                    let ang = std::f32::consts::TAU * (k as f32 / 48.0);
                    let pt = pivot + (u * ang.cos() + v * ang.sin()) * ring_r;
                    if let Some(d) = Self::point_ray_dist(pt, near, dir) {
                        mind = mind.min(d);
                    }
                }
                consider(mind, GizmoHit::Rotate(i), ring_thresh, &mut best);
            }
        }
        best.map(|(_, h)| h)
    }

    /// A world ray (origin, unit dir) through the given viewport pixel.
    fn cursor_ray(&self, rect: egui::Rect, px: f32, py: f32) -> (glam::Vec3, glam::Vec3) {
        let ndc_x = ((px - rect.min.x) / rect.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((py - rect.min.y) / rect.height()) * 2.0;
        let aspect = rect.width() / rect.height();
        let inv = self.camera.view_proj(aspect).inverse();
        let near = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
        (near, (far - near).normalize_or_zero())
    }

    /// Project a world point to a viewport pixel (None if behind the camera).
    fn world_to_screen(&self, p: glam::Vec3, rect: egui::Rect) -> Option<(f32, f32)> {
        let aspect = rect.width() / rect.height();
        let clip = self.camera.view_proj(aspect) * p.extend(1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        let sx = rect.min.x + (ndc.x * 0.5 + 0.5) * rect.width();
        let sy = rect.min.y + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height();
        Some((sx, sy))
    }

    /// The transform axis unit vector for the current op (global world or lead-local).
    fn xform_axis(op: &XformOp, ax: usize) -> glam::Vec3 {
        if op.global {
            [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][ax]
        } else {
            // Local basis: X=forward, Z=up, Y=up×forward (right-handed-ish).
            let f = op.lead_fwd.normalize_or_zero();
            let u = op.lead_up.normalize_or_zero();
            let r = u.cross(f).normalize_or_zero();
            [f, r, u][ax]
        }
    }

    /// Shift+D: clone the selected offline objects into new datums (mvar dups → 0xD8xxxxxx,
    /// local dups → next local). Returns the new datums. Copies pose, meta, and colour.
    fn duplicate_selection(&mut self) -> Vec<u32> {
        let mut out = Vec::new();
        for d in self.selected_set.clone() {
            if let Some(o) = self.mvar_objects.iter().find(|o| o.datum == d).cloned() {
                let nd = self.next_dup_datum;
                self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
                let mut no = o;
                no.datum = nd;
                self.mvar_objects.push(no);
                if let Some(m) = self.mvar_meta.get(&d).cloned() {
                    self.mvar_meta.insert(nd, m);
                }
                if let Some(&c) = self.mvar_colors.get(&d) {
                    self.mvar_colors.insert(nd, c);
                }
                // #construct-h4: give the copy its OWN Halo 4 .mvar record (fresh slot, no
                // parent) exactly as `spawn_copy_of` does -- without this a Mirror-copy or a
                // Fill-edge copy on a Halo 4 map inherited the source's slot. No-op for Reach.
                self.h4_copy_source_record(d, nd);
                out.push(nd);
            } else if let Some(o) = self.local_objects.iter().find(|o| o.datum == d).cloned() {
                let nd = self.next_local_datum;
                self.next_local_datum = self.next_local_datum.wrapping_add(1);
                let mut no = o;
                no.datum = nd;
                self.local_objects.push(no);
                out.push(nd);
            }
        }
        out
    }

    /// Snapshot the current selection into a fresh op (or re-baseline an existing op when the
    /// mode/axis changes so the transform doesn't jump).
    fn begin_xform(&mut self, kind: XformKind, ptr: (f32, f32), duplicated: bool) {
        let datums = self.movable_datums();
        let Some(pivot) = self.selection_pivot(&datums) else { return };
        let lead = self.selected_datum.and_then(|d| self.movable_pose(d));
        let (lead_fwd, lead_up) = lead.map(|(_, f, u)| (f, u)).unwrap_or((glam::Vec3::X, glam::Vec3::Z));
        let start = datums
            .iter()
            .filter_map(|&d| self.movable_pose(d).map(|(p, f, u)| (d, p, f, u)))
            .collect();
        let prev_axis = self.xform.as_ref().and_then(|o| o.axis);
        let prev_global = self.xform.as_ref().map(|o| o.global).unwrap_or(true);
        let prev_undo = self.xform.as_ref().map(|o| o.undo_pushed).unwrap_or(false);
        let prev_drag = self.xform.as_ref().map(|o| o.drag_mode).unwrap_or(false);
        let prev_num = self.xform.as_ref().map(|o| o.num_buf.clone()).unwrap_or_default();
        // A step the user already dialled in survives a mode/axis re-baseline.
        let prev_adj = self.xform.as_ref().map(|o| o.array_adj).unwrap_or(0.0);
        // #snap-array: gather the face sets ONCE, here, at the start pose -- see `XformOp::snap_src`.
        let (snap_src, snap_tgt) = if kind == XformKind::Grab {
            self.gather_snap_faces(&datums, None)
        } else {
            (Vec::new(), Vec::new())
        };
        self.xform = Some(XformOp {
            kind,
            axis: prev_axis,
            global: prev_global,
            pivot,
            start,
            start_pointer: ptr,
            lead_fwd,
            lead_up,
            duplicated,
            undo_pushed: prev_undo,
            drag_mode: prev_drag,
            array_cell: 0.0,
            array_adj: prev_adj,
            wheel_accum: 0.0,
            array_copies: Vec::new(),
            num_buf: prev_num,
            eff_pointer: ptr,
            last_raw_pointer: ptr,
            snap_src,
            snap_tgt,
        });
    }

    /// Duplicate a template set of datums into new objects, each offset by `off`.
    /// Returns the new datums (parallel to `template` order). Copies pose, meta, and colour.
    fn spawn_copy_of(&mut self, template: &[(u32, glam::Vec3, glam::Vec3, glam::Vec3)], off: glam::Vec3) -> Vec<u32> {
        let mut out = Vec::new();
        for &(d, p, f, u) in template {
            let np = p + off;
            if let Some(o) = self.mvar_objects.iter().find(|o| o.datum == d).cloned() {
                let nd = self.next_dup_datum;
                self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
                let mut no = o;
                no.datum = nd;
                no.pos = np.into();
                no.fwd = f.into();
                no.up = u.into();
                self.mvar_objects.push(no);
                if let Some(mut m) = self.mvar_meta.get(&d).cloned() {
                    m.pos = np.into();
                    self.mvar_meta.insert(nd, m);
                }
                if let Some(&c) = self.mvar_colors.get(&d) {
                    self.mvar_colors.insert(nd, c);
                }
                self.h4_copy_source_record(d, nd); // the copy's own .mvar record (no-op for Reach)
                out.push(nd);
            } else if let Some(o) = self.local_objects.iter().find(|o| o.datum == d).cloned() {
                let nd = self.next_local_datum;
                self.next_local_datum = self.next_local_datum.wrapping_add(1);
                let mut no = o;
                no.datum = nd;
                no.pos = np.into();
                no.fwd = f.into();
                no.up = u.into();
                self.local_objects.push(no);
                out.push(nd);
            }
        }
        out
    }

    /// Remove a spawned copy-set's datums from all offline lists.
    fn remove_copy(&mut self, datums: &[u32]) {
        self.mvar_objects.retain(|o| !datums.contains(&o.datum));
        self.local_objects.retain(|o| !datums.contains(&o.datum));
        for d in datums {
            self.mvar_meta.remove(d);
            self.mvar_colors.remove(d);
        }
    }

    /// Closest scalar `s` along the axis line (pivot + s·a) to the cursor ray. None when the
    /// ray is ~parallel to the axis (looking straight down it).
    fn closest_on_axis(&self, pivot: glam::Vec3, a: glam::Vec3, rect: egui::Rect, px: f32, py: f32) -> Option<f32> {
        let (o, d) = self.cursor_ray(rect, px, py);
        let b = a.dot(d);
        let denom = 1.0 - b * b;
        if denom.abs() < 1e-4 {
            return None;
        }
        let w0 = pivot - o;
        Some((b * d.dot(w0) - a.dot(w0)) / denom)
    }

    // ===================== #snap-array: oriented face snap =====================

    /// Gather the snap face sets for a grab: `(source faces, candidate target faces)`.
    ///
    /// Done ONCE per transform op (see `XformOp::snap_src`). Sources come from the moving set at
    /// its current pose; targets are the oriented box faces of every other placed object --
    /// variant pieces, locally placed pieces and the map's own scenario objects -- none of which
    /// move while the op runs. Level geometry is not in here: it is probed by ray each frame
    /// (`snap::bsp_targets`), because a BSP has no object box to take faces from.
    fn gather_snap_faces(&mut self, moving: &[u32], only: Option<u32>) -> (Vec<snap::Face>, Vec<snap::Face>) {
        if moving.is_empty() || self.objscene().is_none() {
            return (Vec::new(), Vec::new());
        }
        // Render-model tag per datum, for the per-model plane cache.
        let tags: std::collections::HashMap<u32, u32> = self
            .mvar_objects
            .iter()
            .chain(self.local_objects.iter())
            .map(|o| (o.datum, o.mode_tag))
            .collect();
        let candidates: Vec<u32> = match only {
            // `snapto <a> to <b>`: only B's faces are on offer.
            Some(d) => vec![d],
            None => self
                .mvar_objects
                .iter()
                .chain(self.local_objects.iter())
                .chain(self.scenario_objects.iter())
                .map(|o| o.datum)
                .filter(|d| !moving.contains(d))
                .collect(),
        };
        // The plane cache is a field, so it cannot be borrowed alongside `objscene()`; take it out
        // for the call and put it back.
        let mut cache = std::mem::take(&mut self.snap_cache);
        let out = {
            let scene = self.objscene().expect("checked above");
            let src = snap::source_faces(scene, moving, &|d| tags.get(&d).copied().unwrap_or(0), &mut cache);
            let tgt = snap::target_faces(scene, &candidates, moving);
            (src, tgt)
        };
        self.snap_cache = cache;
        out
    }

    /// The same face+edge snap as the Ctrl magnet, solved ONCE against the scene as it stands
    /// (no drag). `only` restricts the targets to one object; otherwise every other object and
    /// the level geometry are candidates. `axis` locks the travel to one world axis, exactly as
    /// an axis-locked grab does. Backs the scripted `snapto` verb.
    fn snap_once(&mut self, moving: &[u32], only: Option<u32>, axis: Option<glam::Vec3>) -> Option<snap::SnapResult> {
        let (srcs, tgt_all) = self.gather_snap_faces(moving, only);
        if srcs.is_empty() {
            return None;
        }
        let cfg = snap::SnapCfg { range: self.magnet_range.max(0.05), ..Default::default() };
        let center = srcs.iter().fold(glam::Vec3::ZERO, |a, f| a + f.c) / srcs.len() as f32;
        let reach = srcs.iter().map(|f| (f.c - center).length()).fold(0.0f32, f32::max) + cfg.range;
        let dirs: Vec<glam::Vec3> = match axis {
            Some(a) => vec![a, -a],
            None => Vec::new(),
        };
        let mut tgts = snap::near_targets(&tgt_all, center, reach);
        // Level geometry only when no explicit target object was named.
        if only.is_none() {
            if let Some(scene) = self.objscene() {
                tgts.extend(snap::bsp_targets(&|o, d| scene.raycast_scene_n(o, d), &srcs, &dirs, &cfg));
            }
        }
        snap::solve_faces(&srcs, &tgts, &dirs, &cfg)
    }

    /// The magnet's extra offset for a tentative `delta`, or `None` when nothing is in range.
    fn snap_result(&self, op: &XformOp, delta: glam::Vec3) -> Option<snap::SnapResult> {
        if op.snap_src.is_empty() {
            return None;
        }
        let cfg = snap::SnapCfg { range: self.magnet_range.max(0.05), ..Default::default() };
        // A grab is a pure translation, so the start-pose faces just move with the drag.
        let srcs: Vec<snap::Face> = op.snap_src.iter().map(|f| f.translated(delta)).collect();
        let center = srcs.iter().fold(glam::Vec3::ZERO, |a, f| a + f.c) / srcs.len() as f32;
        let reach = srcs.iter().map(|f| (f.c - center).length()).fold(0.0f32, f32::max) + cfg.range;
        // An axis lock means the snap may only travel on that axis, in either direction.
        let dirs: Vec<glam::Vec3> = match op.axis {
            Some(ax) => {
                let a = Self::xform_axis(op, ax);
                vec![a, -a]
            }
            None => Vec::new(),
        };
        let mut tgts = snap::near_targets(&op.snap_tgt, center, reach);
        if let Some(scene) = self.objscene() {
            tgts.extend(snap::bsp_targets(&|o, d| scene.raycast_scene_n(o, d), &srcs, &dirs, &cfg));
        }
        snap::solve_faces(&srcs, &tgts, &dirs, &cfg)
    }

    /// Face-to-face cell size for an array: the moving set's combined extent along the axis.
    /// This is the array's DEFAULT step, so an untouched array still tiles edge to edge.
    fn array_cell_for(&self, op: &XformOp, a: glam::Vec3) -> f32 {
        let mut gmin = glam::Vec3::splat(f32::INFINITY);
        let mut gmax = glam::Vec3::splat(f32::NEG_INFINITY);
        if let Some(scene) = self.objscene() {
            for &(d, _, _, _) in &op.start {
                if let Some((mn, mx)) = scene.aabb_of(d) {
                    gmin = gmin.min(glam::Vec3::from(mn));
                    gmax = gmax.max(glam::Vec3::from(mx));
                }
            }
        }
        if !gmin.is_finite() {
            return 1.0;
        }
        let ext = gmax - gmin;
        (ext.x * a.x.abs() + ext.y * a.y.abs() + ext.z * a.z.abs()).max(0.1)
    }

    /// The selection's own extent along `dir` (its oriented box's support width), i.e. how long
    /// one copy is along a line -- what a step has to beat to overlap.
    fn selection_extent_along(&self, dir: glam::Vec3) -> f32 {
        let d = dir.normalize_or_zero();
        match self.selection_obb() {
            Some(o) if d.length_squared() > 0.5 => {
                2.0 * (o.hx.dot(d).abs() + o.hy.dot(d).abs() + o.hz.dot(d).abs())
            }
            _ => 0.0,
        }
    }

    /// Fill a LINE with the current selection: the original moves to the first point and copies
    /// fill the rest, at a chosen COUNT over the distance or a chosen STEP (a step smaller than
    /// the piece's own extent makes the copies overlap, which is the point).
    ///
    /// One undo step covers the whole fill, and a multi-object selection is stamped as a unit
    /// (every copy keeps the group's internal layout).
    fn array_along_line(&mut self, from: glam::Vec3, to: glam::Vec3, spec: snap::LineSpec, align: bool) {
        let seg = to - from;
        if seg.length() < 1e-3 && matches!(spec, snap::LineSpec::Step(_)) {
            self.status = "Line array: the two points are the same".into();
            return;
        }
        let pts = snap::line_positions(from, to, spec);
        let dir = seg.normalize_or_zero();
        let made = match self.array_at_points(&pts, dir, align) {
            Ok(n) => n,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        let step = match spec {
            snap::LineSpec::Step(v) => v,
            snap::LineSpec::Count(n) => snap::count_step(from, to, n),
        };
        let ext = self.selection_extent_along(dir);
        self.status = format!(
            "Line array — {} copies + the original over {:.2} wu, step {:.2} wu ({})",
            made,
            seg.length(),
            step,
            snap::spacing_label(step, ext)
        );
    }

    /// The core of every line fill: put the original on `pts[0]` and one copy-set on each
    /// remaining point, optionally yawing the whole line to follow `dir`. Returns how many copies
    /// were made. ONE undo step covers the lot; a multi-object selection stamps as a unit.
    fn array_at_points(&mut self, pts: &[glam::Vec3], dir: glam::Vec3, align: bool) -> Result<usize, String> {
        let datums = self.movable_datums();
        if datums.is_empty() {
            return Err("Line array: select the object(s) to copy first".into());
        }
        let obb = self.selection_obb().ok_or("Line array: selection has no bounds")?;
        if pts.is_empty() {
            return Err("Line array: no points to fill".into());
        }
        self.push_edit_undo();
        self.prop_snapshotted_for = None;
        // Aligning to the line turns the HEADING only (yaw), so a wall stays vertical and a ramp
        // keeps its rise. The same turn goes on every copy, the original included.
        let q = if align {
            self.selected_datum
                .or_else(|| datums.first().copied())
                .and_then(|d| self.movable_pose(d))
                .map(|(_, f, _)| snap::yaw_to(f, dir))
                .unwrap_or(glam::Quat::IDENTITY)
        } else {
            glam::Quat::IDENTITY
        };
        // 1) the ORIGINAL goes to the first point (turned about the group centre).
        for &d in &datums {
            if let Some((p, f, u)) = self.movable_pose(d) {
                let np = pts[0] + q * (p - obb.c);
                self.set_movable_pose(d, np, q * f, q * u);
            }
        }
        // 2) one copy-set per remaining point, offset from the first.
        let template: Vec<(u32, glam::Vec3, glam::Vec3, glam::Vec3)> = datums
            .iter()
            .filter_map(|&d| self.movable_pose(d).map(|(p, f, u)| (d, p, f, u)))
            .collect();
        let mut made = 0usize;
        for p in pts.iter().skip(1) {
            let new = self.spawn_copy_of(&template, *p - pts[0]);
            if new.is_empty() {
                break; // out of variant slots / palette entries
            }
            made += new.len();
        }
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
        self.apply_selection_highlight();
        self.refresh_guides();
        self.overlays_dirty = true;
        Ok(made)
    }

    /// Compute the grab translation for the current pointer (pure — does not move objects),
    /// with axis lock (closest-point-on-axis, so it tracks 1:1 at any view angle) and Ctrl
    /// magnet snapping.
    fn grab_delta(&self, op: &XformOp, rect: egui::Rect, ptr: (f32, f32), magnet: bool) -> glam::Vec3 {
        let mut delta = match op.axis {
            None => {
                // Free move: plane through the pivot facing the camera.
                let n = self.camera.forward();
                let plane_hit = |px: f32, py: f32| -> Option<glam::Vec3> {
                    let (o, d) = self.cursor_ray(rect, px, py);
                    let denom = d.dot(n);
                    if denom.abs() < 1e-5 {
                        return None;
                    }
                    let t = (op.pivot - o).dot(n) / denom;
                    (t > 0.0).then(|| o + d * t)
                };
                match (plane_hit(op.start_pointer.0, op.start_pointer.1), plane_hit(ptr.0, ptr.1)) {
                    (Some(h0), Some(h1)) => h1 - h0,
                    _ => glam::Vec3::ZERO,
                }
            }
            Some(ax) => {
                let a = Self::xform_axis(op, ax);
                match (
                    self.closest_on_axis(op.pivot, a, rect, op.start_pointer.0, op.start_pointer.1),
                    self.closest_on_axis(op.pivot, a, rect, ptr.0, ptr.1),
                ) {
                    (Some(s0), Some(s1)) => a * (s1 - s0),
                    _ => glam::Vec3::ZERO,
                }
            }
        };

        // #snap-array: Ctrl magnet -- an ORIENTED face-pair snap. The moving set's real faces
        // (its oriented box faces plus, for a single piece, its model's large planar faces, so a
        // ramp's sloped deck and tall end count) are pushed along their own outward normals, or
        // along the locked axis, against the nearest other object face / level surface. The best
        // pair by gap + how squarely the faces oppose + how much they overlap wins, and then the
        // piece is slid WITHIN that plane so the nearest edges line up ("it clicks into place").
        // Translation only -- a move never turns the piece.
        if magnet {
            match self.snap_result(op, delta) {
                Some(res) => {
                    delta += res.delta;
                    *self.snap_note.borrow_mut() = res.describe();
                }
                None => self.snap_note.borrow_mut().clear(),
            }
        }

        // With construction guides up, pull the move onto the nearest guide point
        // (endpoint / midpoint / intersection) once it is within `snap_range`. The Ctrl magnet
        // (face-flush snapping) wins when held, so the two never fight.
        if !magnet && self.snap_guides && !self.guides.is_empty() {
            if let Some(p) = self.nearest_guide_point(op.pivot + delta) {
                delta = p - op.pivot;
            }
        }
        delta
    }

    /// Compute the rotation quaternion for the current pointer (pure — does not move objects).
    fn rotate_quat(&self, op: &XformOp, rect: egui::Rect, ptr: (f32, f32)) -> glam::Quat {
        // Numeric entry (e.g. "90", "-90") OVERRIDES the mouse: rotate exactly that many degrees
        // about the chosen axis. With no axis locked, numeric defaults to Z (yaw) — the natural
        // forge rotation — rather than the camera-view axis the mouse path uses.
        // A '-' ANYWHERE in the typed buffer is the sign, so "90-" turns the same way as "-90"
        // (numfield::parse_signed -- the same rule as the panel's angle fields).
        if let Some(deg) = numfield::parse_signed(&op.num_buf).map(|v| v as f32) {
            if deg.is_finite() {
                let axis = op
                    .axis
                    .map(|ax| Self::xform_axis(op, ax))
                    .unwrap_or(glam::Vec3::Z)
                    .normalize_or_zero();
                if axis.length_squared() < 0.5 {
                    return glam::Quat::IDENTITY;
                }
                return glam::Quat::from_axis_angle(axis, deg.to_radians());
            }
        }
        let Some(center) = self.world_to_screen(op.pivot, rect) else { return glam::Quat::IDENTITY };
        let ang = |px: f32, py: f32| (py - center.1).atan2(px - center.0);
        let theta = ang(ptr.0, ptr.1) - ang(op.start_pointer.0, op.start_pointer.1);
        let axis = match op.axis {
            Some(ax) => Self::xform_axis(op, ax),
            None => self.camera.forward(),
        }
        .normalize_or_zero();
        if axis.length_squared() < 0.5 {
            return glam::Quat::IDENTITY;
        }
        // Screen-Y grows downward, so negate for a natural CCW-with-cursor feel.
        glam::Quat::from_axis_angle(axis, -theta)
    }

    /// Apply the op's current transform to the REAL object poses (solid meshes move live).
    /// The texture cache makes the resulting rebuild cheap enough to do every frame.
    fn apply_xform_live(&mut self, op: &XformOp, rect: egui::Rect, ptr: (f32, f32), magnet: bool) {
        match op.kind {
            XformKind::Grab => {
                let delta = self.grab_delta(op, rect, ptr, magnet);
                for &(d, p, f, u) in &op.start {
                    self.set_movable_pose(d, p + delta, f, u);
                }
            }
            XformKind::Rotate => {
                let q = self.rotate_quat(op, rect, ptr);
                for &(d, p, f, u) in &op.start {
                    let np = op.pivot + q * (p - op.pivot);
                    self.set_movable_pose(d, np, q * f, q * u);
                }
            }
        }
    }

    /// Modal transform driver — call before update_camera/handle_pick. Returns true when
    /// an op is active (so the caller suppresses camera fly + object picking).
    fn update_object_transform(&mut self, ctx: &egui::Context, response: &egui::Response) -> bool {
        let rect = response.rect;
        // While a move/rotate/duplicate op is live, the OS cursor is confined to the window
        // (by `apply_cursor_capture`, once per frame, shared with the camera fly) so it cannot
        // leave the app mid-drag, which would lose the release click and freeze the op at the
        // window edge. Confinement is fire-and-forget: on a platform that refuses it the
        // raw-motion integration below still keeps the op alive.
        // Raw mouse motion this frame (device events, logical points). Unlike the pointer
        // position it keeps arriving while the cursor is pinned at the window edge or outside it.
        let raw_motion = Self::raw_mouse_motion(ctx);
        // Pointer position; while an op is live and the pointer has left the window, stay where
        // the op last saw it instead of snapping to the viewport centre.
        let ptr = ctx
            .pointer_latest_pos()
            .map(|p| (p.x, p.y))
            .unwrap_or_else(|| self.xform.as_ref().map(|op| op.last_raw_pointer).unwrap_or((rect.center().x, rect.center().y)));
        let (shift_d, g, r, e, kx, ky, kz, esc, enter, magnet) = ctx.input(|i| {
            (
                // While flying (RMB held), Shift is the camera speed-boost — don't let Shift+D
                // trigger duplicate (nor Shift+A the context menu, gated below).
                i.modifiers.shift && i.key_pressed(egui::Key::D) && !i.pointer.secondary_down(),
                i.key_pressed(egui::Key::G),
                // While RIGHT-CLICK (fly mode) is held, R flies the camera UP — suppress it
                // starting object-rotation here (see update_camera, which reads R as up-thrust).
                i.key_pressed(egui::Key::R) && !i.pointer.secondary_down(),
                i.key_pressed(egui::Key::E),
                i.key_pressed(egui::Key::X),
                i.key_pressed(egui::Key::Y),
                i.key_pressed(egui::Key::Z),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Enter),
                i.modifiers.ctrl || i.modifiers.command,
            )
        });
        // Shift held (array/line-duplicate mode when combined with Ctrl + axis lock).
        let shift_held = ctx.input(|i| i.modifiers.shift);
        let lclick = response.clicked_by(egui::PointerButton::Primary);
        let drag_started = response.drag_started_by(egui::PointerButton::Primary);
        let drag_stopped = response.drag_stopped_by(egui::PointerButton::Primary);
        // Global press/release so a MODAL op can be confirmed even when the cursor has drifted
        // over a side panel (the viewport response wouldn't see that click). Drag ops are owned
        // by the viewport response, so their release fires over UI too.
        let (global_press, global_release) = ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_released()));
        // RIGHT mouse CANCELS an active modal grab/rotate (Blender convention). Without
        // this, a modal op (self.xform) follows the pointer regardless of button — so right-click-
        // dragging to look around would drag the object instead. Only for MODAL ops (not gizmo
        // drag_mode, which is owned by a held primary button); the cancel restores start poses and
        // the same right-drag then flows to the camera next frame.
        let rmb_pressed = ctx.input(|i| i.pointer.secondary_pressed());
        let had_op = self.xform.is_some();
        let hov = response.hovered();

        // Hide the OS cursor during a move/rotate/duplicate so it doesn't obscure placement.
        if had_op {
            ctx.set_cursor_icon(egui::CursorIcon::None);
        }

        // While the construct tool is on, track the anchor under the cursor so the
        // overlay can highlight what a click would grab. Rebuild overlays whenever it changes
        // (only then -- hovering must not repaint the line buffer every frame).
        if self.construct_mode {
            let before = self.construct_hover.map(|a| a.pos);
            self.construct_hover = if hov { self.pick_anchor(rect, ptr) } else { None };
            let after = self.construct_hover.map(|a| a.pos);
            if before.map(|p| p.to_array()) != after.map(|p| p.to_array()) {
                self.overlays_dirty = true;
            }
        } else if self.construct_hover.is_some() {
            self.construct_hover = None;
            self.overlays_dirty = true;
        }

        // Hover highlight: when idle, note which gizmo element is under the cursor so
        // refresh_gizmo can brighten it.
        if self.xform.is_none() {
            self.gizmo_hover = None;
            if hov {
                let datums = self.movable_datums();
                if let Some(pivot) = self.selection_pivot(&datums) {
                    let len = self.gizmo_len(pivot);
                    let (near, dir) = self.cursor_ray(rect, ptr.0, ptr.1);
                    self.gizmo_hover = self.gizmo_pick(pivot, len, near, dir);
                }
            }
        }

        // Esc drops a half-drawn guide (checked before the modal-op handling below, which
        // would otherwise swallow the key).
        if esc && self.cad_face_a.is_some() {
            self.cad_face_a = None;
            self.status = "Coincident: cancelled".into();
        }
        // #snap-array: Esc drops a half-drawn line array (its first point).
        if esc && self.line_pending.is_some() {
            self.line_pending = None;
            self.status = "Line array cancelled".into();
        }
        if esc && self.construct_pending.is_some() {
            self.construct_pending = None;
            self.overlays_dirty = true;
            self.status = "Guide cancelled".into();
            return true;
        }

        // The Circle/Square ops are DRAW tools — press to set the centre,
        // drag to size, release to commit. Pressing on an existing shape's centre grabs and
        // MOVES it instead, so a shape stays editable after it is drawn. Handled before the
        // gizmo/selection code so a construct drag never doubles as an object drag.
        if self.construct_mode && matches!(self.construct_op, ConstructOp::Circle | ConstructOp::Square) {
            if drag_started && hov {
                let press = response.interact_pointer_pos().map(|p| (p.x, p.y)).unwrap_or(ptr);
                let anchor = self.pick_anchor(rect, press)
                    .unwrap_or_else(|| construct::Anchor::point(self.ground_point(rect, press)));
                // grab an existing shape by its centre?
                let grab = self.shapes.iter().enumerate()
                    .map(|(i, sh)| (i, (sh.center - anchor.pos).length()))
                    .filter(|(_, d)| *d < self.shape_grab_radius())
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i);
                self.push_edit_undo(); // one Ctrl+Z backs out the draw or the move
                match grab {
                    Some(i) => {
                        self.sel_shape = Some(i);
                        self.shape_drag = Some((i, false, anchor.pos - self.shapes[i].center));
                        self.status = format!("Moving {} — release to drop", self.shapes[i].kind.label());
                    }
                    None => {
                        let kind = if self.construct_op == ConstructOp::Circle { construct::ShapeKind::Circle } else { construct::ShapeKind::Square };
                        let normal = self.shape_plane_normal(&anchor);
                        let mut sh = construct::Shape::new(kind, anchor.pos, normal);
                        sh.segments = self.cad_shape_seg;
                        sh.rot = self.cad_shape_rot.to_radians();
                        self.shapes.push(sh);
                        let i = self.shapes.len() - 1;
                        self.sel_shape = Some(i);
                        self.shape_drag = Some((i, true, glam::Vec3::ZERO));
                        self.status = format!("Drawing {} — drag out the size", kind.label());
                    }
                }
                self.show_guides = true;
                self.overlays_dirty = true;
            }
            // live update while the button is held
            if let Some((i, sizing, grab)) = self.shape_drag {
                if let Some(sh) = self.shapes.get(i).copied() {
                    let (near, dir) = self.cursor_ray(rect, ptr.0, ptr.1);
                    if let Some(hit) = construct::ray_plane(near, dir, sh.center, sh.normal) {
                        if sizing {
                            self.shapes[i].radius = (hit - sh.center).length();
                        } else {
                            // snap the moved centre to a nearby anchor so shapes land exactly
                            let want = hit - grab;
                            let snapped = self.pick_anchor(rect, ptr).map(|a| a.pos)
                                .filter(|p| (*p - want).length() < self.shape_grab_radius());
                            self.shapes[i].center = snapped.unwrap_or(want);
                        }
                        self.overlays_dirty = true;
                    }
                }
                if drag_stopped || global_release {
                    let (i, sizing, _) = self.shape_drag.take().unwrap();
                    // a click with no drag would leave a zero-size shape behind
                    if sizing && self.shapes.get(i).map(|s| s.radius).unwrap_or(0.0) < 0.05 {
                        self.shapes.remove(i);
                        self.sel_shape = None;
                        self.edit_undo.pop(); // nothing was created -> no undo step
                        self.status = "Shape cancelled (drag to give it a size)".into();
                    } else if let Some(sh) = self.shapes.get(i) {
                        self.status = format!("{} — size {:.2} wu (drag its centre to move, panel to array)", sh.kind.label(), sh.radius);
                    }
                    self.overlays_dirty = true;
                }
            }
            if drag_started || self.shape_drag.is_some() {
                return true; // input consumed by the shape tool
            }
        }

        // A selected SHAPE moves with G, exactly like an object — press G, it
        // follows the cursor, left-click or Enter places it, Esc puts it back where it was.
        // Checked before the object grab so "G" is unambiguous while a shape is selected.
        if let Some((si, start)) = self.shape_grab {
            if si < self.shapes.len() {
                let (near, dir) = self.cursor_ray(rect, ptr.0, ptr.1);
                let n = self.shapes[si].normal;
                // follow the cursor in the shape's own plane, snapping to nearby anchors
                if let Some(hit) = construct::ray_plane(near, dir, self.shapes[si].center, n) {
                    let snapped = self.pick_anchor(rect, ptr).map(|a| a.pos)
                        .filter(|p| (*p - hit).length() < self.shape_grab_radius());
                    self.shapes[si].center = snapped.unwrap_or(hit);
                    self.overlays_dirty = true;
                }
                if esc {
                    self.shapes[si].center = start;
                    self.shape_grab = None;
                    self.overlays_dirty = true;
                    self.status = "Move cancelled".into();
                    return true;
                }
                if lclick || enter || global_press {
                    self.shape_grab = None;
                    self.status = format!("Shape placed at ({:.2}, {:.2}, {:.2})", self.shapes[si].center.x, self.shapes[si].center.y, self.shapes[si].center.z);
                    return true;
                }
                self.status = "Moving shape — click or Enter to place · Esc cancels".into();
                return true;
            }
            self.shape_grab = None;
        }
        if g {
            if let Some(si) = self.sel_shape.filter(|i| *i < self.shapes.len()) {
                // one snapshot covers the whole move, so a single Ctrl+Z undoes it
                self.push_edit_undo();
                self.shape_grab = Some((si, self.shapes[si].center));
                self.status = "Moving shape — click or Enter to place · Esc cancels".into();
                return true;
            }
        }

        // Start an op (only over the viewport, only with a movable selection).
        if self.xform.is_none() && hov {
            // PRESS-DRAG a gizmo handle (move axis) or ring (rotate) → axis-locked op on
            // the whole selection, confirmed on RELEASE. Checked before object picking so the
            // on-top gizmo is grabbable through geometry.
            if drag_started {
                let datums = self.movable_datums();
                if let Some(pivot) = self.selection_pivot(&datums) {
                    let len = self.gizmo_len(pivot);
                    let start_pos = response.interact_pointer_pos().map(|p| (p.x, p.y)).unwrap_or(ptr);
                    let (near, dir) = self.cursor_ray(rect, start_pos.0, start_pos.1);
                    if let Some(hit) = self.gizmo_pick(pivot, len, near, dir) {
                        let (kind, ax, label) = match hit {
                            GizmoHit::Move(a) => (XformKind::Grab, a, "move"),
                            GizmoHit::Rotate(a) => (XformKind::Rotate, a, "rotate"),
                        };
                        self.begin_xform(kind, start_pos, false);
                        if let Some(op) = self.xform.as_mut() {
                            op.axis = Some(ax);
                            op.drag_mode = true;
                        }
                        self.status = format!("Gizmo {label} — {} axis · release to place · Esc", ["X", "Y", "Z"][ax]);
                        return true;
                    }
                }
            }
            if shift_d {
                if !self.movable_datums().is_empty() {
                    // Snapshot BEFORE duplicating so one Ctrl+Z reverts the whole dup+move.
                    self.push_edit_undo();
                    self.prop_snapshotted_for = None;
                    let dups = self.duplicate_selection();
                    if !dups.is_empty() {
                        self.selected_set = dups.clone();
                        self.selected_datum = dups.last().copied();
                        self.begin_xform(XformKind::Grab, ptr, true);
                        if let Some(op) = self.xform.as_mut() {
                            op.undo_pushed = true;
                        }
                        self.status = format!("Duplicated {} object(s) — move, click to place", dups.len());
                        return true;
                    } else {
                        self.edit_undo.pop(); // nothing duplicated -> discard the snapshot
                    }
                }
            }
            if g && !self.movable_datums().is_empty() {
                self.begin_xform(XformKind::Grab, ptr, false);
                self.status = "Grab — X/Y/Z axis · E space · Ctrl magnet · Shift fine · click/Esc".into();
                return true;
            }
            if r && !self.movable_datums().is_empty() {
                self.begin_xform(XformKind::Rotate, ptr, false);
                self.status = "Rotate — X/Y/Z axis · E space · Shift fine · click/Esc".into();
                return true;
            }
        }

        if self.xform.is_none() {
            return false;
        }

        // Mode switch mid-op re-baselines from the current poses so nothing jumps.
        if g {
            let dup = self.xform.as_ref().unwrap().duplicated;
            self.begin_xform(XformKind::Grab, ptr, dup);
        } else if r {
            let dup = self.xform.as_ref().unwrap().duplicated;
            self.begin_xform(XformKind::Rotate, ptr, dup);
        }

        let mut op = self.xform.take().unwrap();
        // Axis lock: pressing the same axis again clears to free.
        if kx {
            op.axis = if op.axis == Some(0) { None } else { Some(0) };
        }
        if ky {
            op.axis = if op.axis == Some(1) { None } else { Some(1) };
        }
        if kz {
            op.axis = if op.axis == Some(2) { None } else { Some(2) };
        }
        if e {
            op.global = !op.global;
        }
        // Numeric entry: type an exact amount (degrees for Rotate, world units for an axis-locked
        // Grab). Digits / '.' / '-' accumulate; Backspace deletes; it overrides the mouse while
        // non-empty. Enter (handled below as `confirm`) commits. A '-' is accepted at ANY point
        // and flips the sign (so "90-" == "-90"): deciding the direction after typing the number
        // is the normal way round, and nobody should have to clear the field to do it.
        ctx.input(|i| {
            for ev in &i.events {
                match ev {
                    egui::Event::Text(t) => {
                        for ch in t.chars() {
                            if ch.is_ascii_digit() || ch == '.' || ch == '-' {
                                op.num_buf.push(ch);
                            }
                        }
                    }
                    egui::Event::Key { key: egui::Key::Backspace, pressed: true, .. } => {
                        op.num_buf.pop();
                    }
                    _ => {}
                }
            }
        });
        if !op.num_buf.is_empty() {
            let (kname, unit) = if op.kind == XformKind::Rotate { ("Rotate", "°") } else { ("Move", "u") };
            let axname = op.axis.map(|a| ["X", "Y", "Z"][a]).unwrap_or(if op.kind == XformKind::Rotate { "Z" } else { "—" });
            // Echo what the buffer MEANS (a trailing '-' has already flipped the sign) next to
            // what was typed, so the flip is visible before Enter.
            let val = match numfield::parse_signed(&op.num_buf) {
                Some(v) if format!("{v}") != op.num_buf => format!("{} = {v}", op.num_buf),
                _ => op.num_buf.clone(),
            };
            self.status = format!("{kname} {val}{unit} on {axname} — Enter to apply · Esc");
        }

        // Right-click cancels a MODAL op (frees the camera); gizmo drag ops are held by
        // the primary button and end on its release, so RMB does not abort those.
        let cancel = esc || (rmb_pressed && !op.drag_mode);
        if cancel {
            // Cancel: restore start poses; delete duplicates if this op created them.
            for &(d, p, f, u) in &op.start {
                self.set_movable_pose(d, p, f, u);
            }
            if op.duplicated {
                let dd: Vec<u32> = op.start.iter().map(|&(d, _, _, _)| d).collect();
                self.mvar_objects.retain(|o| !dd.contains(&o.datum));
                self.local_objects.retain(|o| !dd.contains(&o.datum));
                for d in &dd {
                    self.mvar_meta.remove(d);
                    self.mvar_colors.remove(d);
                }
                self.selected_set.clear();
                self.selected_datum = None;
            }
            // Also delete any line-duplicate copies spawned this op.
            let copies = std::mem::take(&mut op.array_copies);
            for c in &copies {
                self.remove_copy(c);
            }
            // Discard the undo snapshot pushed at Shift+D — the cancel already reverted.
            if op.undo_pushed {
                self.edit_undo.pop();
            }
            self.apply_selection_highlight();
            self.status = "Transform cancelled".into();
            return true;
        }
        // Confirm: modal ops on click/Enter; gizmo drag ops on mouse RELEASE.
        // drag ops confirm on release (over UI too, since the drag is viewport-owned); modal
        // ops confirm on a click/Enter OR any global press (so a click over a panel still places).
        let confirm = if op.drag_mode {
            drag_stopped || global_release
        } else {
            lclick || enter || global_press
        };
        if confirm {
            // Commit: objects already sit at the transformed pose from the live drag below;
            // just record the undo (unless a dup already snapshotted) and finish.
            if !op.undo_pushed {
                // Snapshot the PRE-op state: reconstruct it by momentarily restoring start
                // poses, snapshotting, then re-applying the final transform.
                let final_poses: Vec<(u32, glam::Vec3, glam::Vec3, glam::Vec3)> = op
                    .start
                    .iter()
                    .filter_map(|&(d, _, _, _)| self.movable_pose(d).map(|(p, f, u)| (d, p, f, u)))
                    .collect();
                for &(d, p, f, u) in &op.start {
                    self.set_movable_pose(d, p, f, u);
                }
                self.push_edit_undo();
                for &(d, p, f, u) in &final_poses {
                    self.set_movable_pose(d, p, f, u);
                }
                self.prop_snapshotted_for = None;
            }
            self.status = format!("Transform applied to {} object(s)", op.start.len());
            self.apply_selection_highlight();
            // Swallow the confirming press's trailing release-click so it doesn't pick a
            // background object next frame — but only when confirmed by a mouse press over the
            // viewport (a release still coming). Keyboard/over-UI confirms leave no trailing
            // viewport click, so must NOT arm the flag (it would eat the next real click).
            self.suppress_next_click = global_press && hov;
            return true; // op dropped (confirmed)
        }

        // Precision integrator: Shift alone (not the Shift+Ctrl array combo) scales the
        // per-frame cursor displacement down so you can fine-tune position/angle when zoomed in
        // right on an object. Integrates raw motion → eff_pointer; at scale 1.0 eff == raw.
        const PRECISION_SCALE: f32 = 0.16;
        let precision = shift_held && !magnet && op.num_buf.is_empty();
        let pscale = if precision { PRECISION_SCALE } else { 1.0 };
        // The pointer-position delta keeps the 1:1 cursor feel (OS acceleration,
        // DPI) whenever the cursor actually moved on screen; when it is pinned at the window edge
        // or outside the window the position stops changing, so fall back to the raw device
        // motion, which keeps arriving and lets the op continue past the edge.
        let pos_delta = (ptr.0 - op.last_raw_pointer.0, ptr.1 - op.last_raw_pointer.1);
        let pointer_inside = ctx.pointer_latest_pos().is_some();
        if pointer_inside && (pos_delta.0 != 0.0 || pos_delta.1 != 0.0) {
            op.eff_pointer.0 += pos_delta.0 * pscale;
            op.eff_pointer.1 += pos_delta.1 * pscale;
        } else if let Some((dx, dy)) = raw_motion {
            op.eff_pointer.0 += dx * pscale;
            op.eff_pointer.1 += dy * pscale;
        }
        op.last_raw_pointer = ptr;
        let ptr = op.eff_pointer;

        // Array / line-duplicate: Shift+Ctrl held, on a duplicate grab with an axis locked →
        // stamp flush copies along that axis, one per cell, following the cursor (like drawing
        // a line of blocks). Otherwise, normal live move.
        let array_mode = op.kind == XformKind::Grab
            && op.duplicated
            && op.axis.is_some()
            && shift_held
            && magnet;
        if array_mode {
            // #snap-array: the WHEEL retunes the array step without leaving the drag. Scroll UP
            // grows it (the copies spread out), DOWN shrinks it (they overlap); hold Alt for
            // centimetre steps. Alt rather than Shift because Shift+Ctrl is what array mode IS,
            // so Shift is already held; and the camera's Ctrl+wheel fly-speed handler is not
            // running (a live transform op owns the viewport's input).
            if op.array_cell <= 0.0 {
                let a = Self::xform_axis(&op, op.axis.unwrap_or(0));
                op.array_cell = self.array_cell_for(&op, a);
            }
            let (wheel, alt) = ctx.input(|i| (i.raw_scroll_delta.y, i.modifiers.alt));
            op.wheel_accum += wheel;
            const NOTCH: f32 = 20.0; // raw scroll units per detent (device-dependent -> integrate)
            let mut notches = 0i32;
            while op.wheel_accum >= NOTCH {
                op.wheel_accum -= NOTCH;
                notches += 1;
            }
            while op.wheel_accum <= -NOTCH {
                op.wheel_accum += NOTCH;
                notches -= 1;
            }
            if notches != 0 {
                op.array_adj = snap::adjust_step(op.array_adj, op.array_cell, notches, alt);
            }
            self.array_stamp(&mut op, rect, ptr);
        } else {
            // Live drag: move the REAL objects (solid meshes) every frame.
            self.apply_xform_live(&op, rect, ptr, magnet);
            // #snap-array: say what the magnet actually mated (which face, onto what, the gap it
            // closed and whether the edges clicked into line) -- measured, not guessed.
            if magnet && op.kind == XformKind::Grab && op.num_buf.is_empty() {
                let note = self.snap_note.borrow().clone();
                self.status = if note.is_empty() {
                    "Magnet — nothing in range · X/Y/Z axis · Shift+Ctrl arrays · click/Esc".into()
                } else {
                    format!("Snapped: {note}")
                };
            }
        }
        self.apply_selection_highlight();
        self.xform = Some(op);
        true
    }

    /// Maintain a flush line of duplicate copies along the locked axis, matching how
    /// far the cursor has travelled (round(dist / cell) copies). Copy #1 is op.start; extras
    /// are spawned/removed to track the cursor and repositioned at cell·k offsets.
    fn array_stamp(&mut self, op: &mut XformOp, rect: egui::Rect, ptr: (f32, f32)) {
        let Some(ax) = op.axis else { return };
        let a = Self::xform_axis(op, ax);
        // Cell = template extent along the axis (compute once), so an untouched array sits flush.
        if op.array_cell <= 0.0 {
            op.array_cell = self.array_cell_for(op, a);
        }
        // #snap-array: the real step is the flush cell plus whatever the wheel dialled in, so the
        // copies can be spread apart (gap) or driven into each other (overlap) mid-drag.
        let cell = (op.array_cell + op.array_adj).max(0.05);
        // Signed distance along the axis from where the drag began.
        let dist = match (
            self.closest_on_axis(op.pivot, a, rect, op.start_pointer.0, op.start_pointer.1),
            self.closest_on_axis(op.pivot, a, rect, ptr.0, ptr.1),
        ) {
            (Some(s0), Some(s1)) => s1 - s0,
            _ => 0.0,
        };
        let sign = if dist < 0.0 { -1.0 } else { 1.0 };
        let n_total = ((dist.abs() / cell).round() as i32).max(1); // copies incl. copy #1
        let n_extra = (n_total - 1).max(0) as usize;
        // Copy #1 (op.start) at cell·1.
        let off1 = a * (sign * cell);
        let start = op.start.clone();
        for &(d, p, f, u) in &start {
            self.set_movable_pose(d, p + off1, f, u);
        }
        // Spawn/remove extra copies to reach n_extra.
        while op.array_copies.len() < n_extra {
            let idx = op.array_copies.len();
            let off = a * (sign * cell * (idx as f32 + 2.0));
            let new = self.spawn_copy_of(&start, off);
            op.array_copies.push(new);
        }
        while op.array_copies.len() > n_extra {
            if let Some(c) = op.array_copies.pop() {
                self.remove_copy(&c);
            }
        }
        // Reposition all extras (direction may have flipped).
        let copies = op.array_copies.clone();
        for (j, copy) in copies.iter().enumerate() {
            let off = a * (sign * cell * (j as f32 + 2.0));
            for (k, &cd) in copy.iter().enumerate() {
                if let Some(&(_, p, f, u)) = start.get(k) {
                    self.set_movable_pose(cd, p + off, f, u);
                }
            }
        }
        self.status = format!(
            "Line duplicate — {n_total} copies, step {cell:.2} wu ({}) · wheel = step, +Alt fine · piece {:.2} wu",
            snap::spacing_label(cell, op.array_cell),
            op.array_cell
        );
    }

    /// Delete/Backspace removes the whole selection from the scene (offline mvar + local
    /// objects, and live objects via the transform queue). Undoable. Returns true if it acted
    /// (so the caller skips the arrow-key edit handler). Ignored while typing or transforming.
    fn handle_delete_key(&mut self, ctx: &egui::Context) -> bool {
        let del = ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
        if !del || self.selected_set.is_empty() || self.xform.is_some() {
            return false;
        }
        // Don't delete while a text field (search box, rename, etc.) has focus.
        if ctx.memory(|m| m.focused().is_some()) {
            return false;
        }
        self.delete_selection();
        true
    }

    /// Remove the whole selection from the scene (offline mvar + local objects, and
    /// live objects via the transform queue). Undoable. Shared by Delete/Backspace, the Edit
    /// menu and the Object panel's Delete button.
    fn delete_selection(&mut self) {
        if self.selected_set.is_empty() {
            return;
        }
        let dd = self.selected_set.clone();
        // Snapshot for undo only if it touches offline objects (has render state to restore).
        let touches_offline = dd.iter().any(|d| {
            self.mvar_objects.iter().any(|o| o.datum == *d)
                || self.local_objects.iter().any(|o| o.datum == *d)
        });
        if touches_offline {
            self.push_edit_undo();
            self.prop_snapshotted_for = None;
        }
        self.mvar_objects.retain(|o| !dd.contains(&o.datum));
        self.local_objects.retain(|o| !dd.contains(&o.datum));
        for d in &dd {
            self.mvar_meta.remove(d);
            self.mvar_colors.remove(d);
        }
        // Live objects (from the game) → forge-remove through the transform queue.
        if let Some(tq) = &self.transform_queue {
            for &d in &dd {
                tq.enqueue_delete(d);
            }
        }
        self.selected_set.clear();
        self.selected_datum = None;
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
        self.apply_selection_highlight();
        self.status = format!("Deleted {} object(s)", dd.len());
    }

    /// Gravity settle: drop `datum` onto the surface below it AND tilt it to rest flush on the
    /// slope, the way the engine would let it come to rest — a warthog dropped on a hillside ends up
    /// with all four tyres on the ground, not floating with the body level. We sample the surface
    /// under a few footprint points (BSP terrain, other objects, imports), average the hit normals
    /// to get the local slope, align the object's up to that normal (preserving its heading), and
    /// lower it so its lowest point rests on the ground. Returns true if a surface was found.
    fn settle_object(&mut self, datum: u32) -> bool {
        let Some((pos, fwd, _up)) = self.movable_pose(datum) else { return false };
        // Footprint from the object's world AABB (fall back to a small box around the origin).
        let (mn, mx) = self
            .objscene()
            .and_then(|s| s.aabb_of(datum))
            .unwrap_or(([pos.x - 1.0, pos.y - 1.0, pos.z - 1.0], [pos.x + 1.0, pos.y + 1.0, pos.z + 1.0]));
        let (mn, mx) = (glam::Vec3::from(mn), glam::Vec3::from(mx));
        // How far the origin sits above the object's lowest point — we rest that point on the ground.
        let bottom_offset = (pos.z - mn.z).max(0.0);
        let cx = (mn.x + mx.x) * 0.5;
        let cy = (mn.y + mx.y) * 0.5;
        // Inset the corner samples inside the footprint so we hit real ground under the tyres, not a
        // cliff edge just past the bumper.
        let hx = (mx.x - mn.x) * 0.35;
        let hy = (mx.y - mn.y) * 0.35;
        let samples = [
            (cx, cy),
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx - hx, cy + hy),
            (cx + hx, cy + hy),
        ];
        let top_z = mx.z + 2.0; // start each ray above the object
        let dir = glam::Vec3::NEG_Z;
        // For each footprint sample, find the nearest surface below and its normal.
        let surface_at = |app: &Self, sx: f32, sy: f32| -> Option<(glam::Vec3, glam::Vec3)> {
            let o = glam::Vec3::new(sx, sy, top_z);
            let mut best: Option<(glam::Vec3, glam::Vec3)> = None;
            let consider = |p: glam::Vec3, n: glam::Vec3, best: &mut Option<(glam::Vec3, glam::Vec3)>| {
                if best.map(|(bp, _)| p.z > bp.z).unwrap_or(true) {
                    *best = Some((p, n));
                }
            };
            if let Some((p, n)) = app.objscene().and_then(|s| s.raycast_scene_n(o, dir)) {
                consider(p, n, &mut best);
            }
            if let Some(p) = app.objscene().and_then(|s| s.raycast_objects_excluding(o, dir, &[datum])) {
                consider(p, glam::Vec3::Z, &mut best); // object tops: treat as flat
            }
            if let Some(t) = raycast_tris(o, dir, &app.import_tris) {
                consider(o + dir * t, glam::Vec3::Z, &mut best);
            }
            best
        };
        let mut center_z = None;
        let mut nsum = glam::Vec3::ZERO;
        let mut nhits = 0;
        for (i, (sx, sy)) in samples.iter().enumerate() {
            if let Some((p, n)) = surface_at(self, *sx, *sy) {
                // Make every normal point up before averaging.
                let n = if n.z < 0.0 { -n } else { n };
                nsum += n;
                nhits += 1;
                if i == 0 {
                    center_z = Some(p.z);
                }
            }
        }
        if nhits == 0 {
            return false;
        }
        // If the centre ray missed (e.g. a hole), fall back to the highest corner hit.
        let center_z = center_z.unwrap_or_else(|| {
            samples[1..]
                .iter()
                .filter_map(|(sx, sy)| surface_at(self, *sx, *sy).map(|(p, _)| p.z))
                .fold(f32::NEG_INFINITY, f32::max)
        });
        // Average slope normal. Guard against garbage / near-vertical fits (keep upright then).
        let mut up = (nsum / nhits as f32).normalize_or_zero();
        if !up.is_finite() || up.z < 0.2 {
            up = glam::Vec3::Z;
        }
        // Preserve heading: project the current forward onto the slope plane.
        let mut f = fwd - up * fwd.dot(up);
        if f.length_squared() < 1e-6 {
            f = glam::Vec3::X - up * glam::Vec3::X.dot(up);
            if f.length_squared() < 1e-6 {
                f = glam::Vec3::Y - up * glam::Vec3::Y.dot(up);
            }
        }
        let f = f.normalize_or_zero();
        let new_pos = glam::Vec3::new(pos.x, pos.y, center_z + bottom_offset);
        self.set_movable_pose(datum, new_pos, f, up);
        true
    }

    /// Editor shortcuts (fire only when no text field has focus):
    /// - Ctrl/Cmd+A → select every offline forge object in the scene.
    /// - Shift+A → open the cursor context menu (only meaningful with a selection).
    fn handle_editor_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (ctrl_a, shift_a, frame_f) = ctx.input(|i| {
            (
                (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(egui::Key::A),
                i.modifiers.shift
                    && !i.modifiers.ctrl
                    && !i.modifiers.command
                    && i.key_pressed(egui::Key::A)
                    // Shift is the fly speed-boost while RMB is held — don't open the menu.
                    && !i.pointer.secondary_down(),
                // F frames the selection. While RMB is held F is the fly-DOWN key
                // (update_camera), so it only frames when not flying.
                i.key_pressed(egui::Key::F)
                    && !i.modifiers.any()
                    && !i.pointer.secondary_down(),
            )
        });
        if ctrl_a {
            self.select_all_objects();
        }
        if shift_a && !self.selected_set.is_empty() {
            self.ctx_menu_pos = ctx.pointer_latest_pos();
        }
        if frame_f && self.xform.is_none() {
            self.frame_selected();
        }
    }

    /// Cursor context menu (opened by Shift+A). One selected object → per-type/per-object
    /// actions; several selected → a selection-wide settle. Extensible: add rows here later.
    fn draw_context_menu(&mut self, ctx: &egui::Context) {
        let Some(pos) = self.ctx_menu_pos else { return };
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.ctx_menu_pos = None;
            return;
        }
        let single = (self.selected_set.len() == 1).then(|| self.selected_set[0]);
        let mut close = false;
        let area = egui::Area::new(egui::Id::new("hms_ctx_menu"))
            .fixed_pos(pos)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(190.0);
                    if let Some(d) = single {
                        let tag = self
                            .mvar_objects
                            .iter()
                            .chain(self.local_objects.iter())
                            .find(|o| o.datum == d)
                            .map(|o| o.primary_tag)
                            .unwrap_or(0);
                        if ui.button("Select all of same object type").clicked() {
                            let sel: Vec<u32> = self
                                .mvar_objects
                                .iter()
                                .chain(self.local_objects.iter())
                                .filter(|o| o.primary_tag == tag)
                                .map(|o| o.datum)
                                .collect();
                            let n = sel.len();
                            self.selected_set = sel;
                            self.selected_datum = self.selected_set.last().copied();
                            self.apply_selection_highlight();
                            self.status = format!("Selected {n} of the same type");
                            close = true;
                        }
                        if ui.button("Simulate this object's physics").clicked() {
                            self.push_edit_undo();
                            self.prop_snapshotted_for = None;
                            let moved = self.settle_object(d);
                            if let Some(s) = self.objscene_mut() {
                                s.invalidate();
                            }
                            self.apply_selection_highlight();
                            self.status = if moved {
                                "Settled object onto the surface below".into()
                            } else {
                                "No surface found below the object".into()
                            };
                            close = true;
                        }
                    } else {
                        ui.label(format!("{} objects selected", self.selected_set.len()));
                        ui.separator();
                        if ui.button("Simulate physics for selection").clicked() {
                            self.push_edit_undo();
                            self.prop_snapshotted_for = None;
                            let sel = self.selected_set.clone();
                            let mut moved = 0;
                            for d in sel {
                                if self.settle_object(d) {
                                    moved += 1;
                                }
                            }
                            if let Some(s) = self.objscene_mut() {
                                s.invalidate();
                            }
                            self.apply_selection_highlight();
                            self.status = format!("Settled {moved} object(s)");
                            close = true;
                        }
                    }
                });
            });
        // Click outside the menu closes it.
        if ctx.input(|i| i.pointer.any_pressed()) {
            if let Some(pp) = ctx.pointer_latest_pos() {
                if !area.response.rect.contains(pp) {
                    close = true;
                }
            }
        }
        if close {
            self.ctx_menu_pos = None;
        }
    }

    /// Arrow keys / PageUp-PageDown nudge the selection (0.25 wu, Shift = 1 wu), [ and ] yaw it
    /// (15 deg, Shift = 45 deg). Every OFFLINE selected object (the same set G/R operate on) is
    /// nudged directly, as one undo step per key press; a live-engine object goes through the
    /// transform queue. Ignored while a text field has focus or a modal transform is running.
    fn handle_selected_edit(&mut self, ctx: &egui::Context) {
        let Some(datum) = self.selected_datum else { return };
        if self.xform.is_some() || ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (mut dx, mut dy, mut dz, mut del) = (0.0f32, 0.0f32, 0.0f32, false);
        let mut ryaw = 0.0f32; // radians, yaw about +Z
        ctx.input(|i| {
            let s = if i.modifiers.shift { 1.0 } else { 0.25 };
            if i.key_pressed(egui::Key::ArrowRight) { dx += s; }
            if i.key_pressed(egui::Key::ArrowLeft) { dx -= s; }
            if i.key_pressed(egui::Key::ArrowUp) { dy += s; }
            if i.key_pressed(egui::Key::ArrowDown) { dy -= s; }
            if i.key_pressed(egui::Key::PageUp) { dz += s; }
            if i.key_pressed(egui::Key::PageDown) { dz -= s; }
            // Rotate (yaw): [ and ] — Shift = 45°, else 15°.
            let step = if i.modifiers.shift { std::f32::consts::FRAC_PI_4 } else { 15.0f32.to_radians() };
            if i.key_pressed(egui::Key::CloseBracket) { ryaw += step; }
            if i.key_pressed(egui::Key::OpenBracket) { ryaw -= step; }
            if i.key_pressed(egui::Key::Delete) { del = true; }
        });

        if del {
            if let Some(tq) = &self.transform_queue {
                tq.enqueue_delete(datum);
            }
            self.selected_set.retain(|&x| x != datum);
            self.selected_datum = self.selected_set.last().copied();
            self.apply_selection_highlight();
            self.status = format!("Deleted object 0x{datum:08X}");
            return;
        }
        // Offline objects first (variant / placed): nudge the whole movable selection.
        let movable = self.movable_datums();
        if !movable.is_empty() && (dx != 0.0 || dy != 0.0 || dz != 0.0 || ryaw != 0.0) {
            self.push_edit_undo();
            self.prop_snapshotted_for = None;
            let (s, c) = ryaw.sin_cos();
            let n = movable.len();
            for d in movable {
                let Some((p, f, u)) = self.movable_pose(d) else { continue };
                let rot = |v: glam::Vec3| glam::Vec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
                let (f, u) = if ryaw != 0.0 { (rot(f), rot(u)) } else { (f, u) };
                self.set_movable_pose(d, p + glam::Vec3::new(dx, dy, dz), f, u);
            }
            // (no scene invalidate needed: the per-tick object signature covers pos + fwd/up)
            self.overlays_dirty = true;
            self.status = if ryaw != 0.0 {
                format!("Rotated {n} object(s) by {:.0} deg", ryaw.to_degrees())
            } else {
                format!("Nudged {n} object(s) by ({dx:.2}, {dy:.2}, {dz:.2})")
            };
            return;
        }
        if dx != 0.0 || dy != 0.0 || dz != 0.0 {
            if let Some(o) = self.last_objects.iter().find(|o| o.datum == datum).cloned() {
                let np = [o.pos[0] + dx, o.pos[1] + dy, o.pos[2] + dz];
                if let Some(tq) = &self.transform_queue {
                    tq.enqueue_translate(datum, np);
                }
                self.record_edit(
                    datum,
                    Pose { pos: o.pos, fwd: o.fwd, up: o.up },
                    Pose { pos: np, fwd: o.fwd, up: o.up },
                );
                self.status = format!("Moved 0x{datum:08X} -> ({:.1},{:.1},{:.1})", np[0], np[1], np[2]);
            }
        }
        if ryaw != 0.0 {
            if let Some(o) = self.last_objects.iter().find(|o| o.datum == datum).cloned() {
                // Rotate the forward/up basis about +Z (yaw).
                let (s, c) = ryaw.sin_cos();
                let rot = |v: [f32; 3]| [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
                let fwd = rot(o.fwd);
                let up = rot(o.up);
                if let Some(tq) = &self.transform_queue {
                    tq.enqueue_pose_full(datum, o.pos, fwd, up);
                }
                self.record_edit(
                    datum,
                    Pose { pos: o.pos, fwd: o.fwd, up: o.up },
                    Pose { pos: o.pos, fwd, up },
                );
                self.status = format!("Rotated 0x{datum:08X} by {:.0}°", ryaw.to_degrees());
            }
        }
    }

    /// Record a reversible pose edit (cap 256), clearing the redo stack.
    fn record_edit(&mut self, datum: u32, before: Pose, after: Pose) {
        self.undo_stack.push(PoseEdit { datum, before, after });
        if self.undo_stack.len() > 256 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Ctrl+Z / Ctrl+Y (also Ctrl+Shift+Z) - undo / redo. Ignored while a text field
    /// has focus so the field's own Ctrl+Z does not also rewind the scene; the verbs themselves
    /// live in do_undo / do_redo so the Edit menu shares them.
    fn handle_undo_redo(&mut self, ctx: &egui::Context) {
        if ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (undo, redo) = ctx.input(|i| {
            (
                i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                i.modifiers.command && (i.key_pressed(egui::Key::Y)
                    || (i.modifiers.shift && i.key_pressed(egui::Key::Z))),
            )
        });
        if undo {
            self.do_undo();
        } else if redo {
            self.do_redo();
        }
    }

    /// Undo one step. OFFLINE edits (move/rotate/duplicate/data-edit) undo via full
    /// snapshots, preferred over the live-engine pose stack (transform-queue objects).
    fn do_undo(&mut self) {
        if !self.edit_undo.is_empty() {
            let cur = self.edit_snapshot();
            let prev = self.edit_undo.pop().unwrap();
            self.edit_redo.push(cur);
            self.restore_edit_snapshot(prev);
            self.status = "Undo".into();
            return;
        }
        let Some(tq) = &self.transform_queue else { return };
        if let Some(e) = self.undo_stack.pop() {
            tq.enqueue_pose_full(e.datum, e.before.pos, e.before.fwd, e.before.up);
            self.redo_stack.push(e);
            self.status = format!("Undo -> 0x{:08X}", e.datum);
        }
    }

    /// Redo one step (see do_undo).
    fn do_redo(&mut self) {
        if !self.edit_redo.is_empty() {
            let cur = self.edit_snapshot();
            let next = self.edit_redo.pop().unwrap();
            self.edit_undo.push(cur);
            self.restore_edit_snapshot(next);
            self.status = "Redo".into();
            return;
        }
        let Some(tq) = &self.transform_queue else { return };
        if let Some(e) = self.redo_stack.pop() {
            tq.enqueue_pose_full(e.datum, e.after.pos, e.after.fwd, e.after.up);
            self.undo_stack.push(e);
            self.status = format!("Redo -> 0x{:08X}", e.datum);
        }
    }

    /// Select every offline forge object (variant + placed). Shared by Ctrl+A and
    /// Edit > Select all.
    fn select_all_objects(&mut self) {
        let all: Vec<u32> = self
            .mvar_objects
            .iter()
            .chain(self.local_objects.iter())
            .map(|o| o.datum)
            .collect();
        let n = all.len();
        self.selected_set = all;
        self.selected_datum = self.selected_set.last().copied();
        self.apply_selection_highlight();
        self.status = format!("Selected all {n} object(s)");
    }

    /// Fly the camera to the primary selected object (View > Frame selected, F, and a
    /// double-click in the Objects list). No-op without a selection or when the object has no
    /// known pose.
    /// #wire-attach Frame the selected object on its SELECTION BOX, not its origin: the box is the
    /// union of the object and its attachments (a Wraith's mortar, a rocket Warthog's turret), so a
    /// big vehicle is centred and fully in shot instead of sitting half out of frame above a
    /// fixed-distance camera aimed at its base. Objects with no decoded bounds keep the old
    /// origin-and-fixed-offset behaviour.
    fn frame_selected(&mut self) {
        let Some(d) = self.selected_datum else { return };
        if let Some((mn, mx)) = self.objscene().and_then(|s| s.aabb_of(d)) {
            let (mn, mx) = (glam::Vec3::from(mn), glam::Vec3::from(mx));
            if mn.is_finite() && mx.is_finite() && mx.cmpge(mn).all() {
                let centre = (mn + mx) * 0.5;
                // stand off by the box's own size (the old 6/6/4 offset is the floor)
                let r = ((mx - mn).length() * 0.5).max(1.0);
                let dir = glam::Vec3::new(-1.0, -1.0, 0.667).normalize();
                self.camera.pos = centre + dir * (r * 2.6).max(8.77);
                let look = (centre - self.camera.pos).normalize_or_zero();
                if look.length_squared() > 1e-6 {
                    self.camera.yaw = look.y.atan2(look.x);
                    self.camera.pitch = look.z.clamp(-1.0, 1.0).asin();
                }
                self.status = format!("Framed selection at ({:.1},{:.1},{:.1}), box {:.1} x {:.1} x {:.1} wu",
                    centre.x, centre.y, centre.z, mx.x - mn.x, mx.y - mn.y, mx.z - mn.z);
                return;
            }
        }
        let pos = self
            .movable_pose(d)
            .map(|(p, _, _)| p)
            .or_else(|| self.last_objects.iter().find(|o| o.datum == d).map(|o| glam::Vec3::from(o.pos)));
        if let Some(p) = pos {
            self.frame_on_point(p);
        }
    }


    /// Raw device mouse motion this frame in logical points (keeps arriving while the
    /// cursor is locked in place, pinned at a window edge, or outside the window).
    fn raw_mouse_motion(ctx: &egui::Context) -> Option<(f32, f32)> {
        ctx.input(|i| {
            let pp = i.pixels_per_point.max(1e-3);
            let mut any = false;
            let mut d = (0.0f32, 0.0f32);
            for e in &i.raw.events {
                if let egui::Event::MouseMoved(v) = e { any = true; d.0 += v.x / pp; d.1 += v.y / pp; }
            }
            if any { Some(d) } else { None }
        })
    }

    /// Capture + hide the OS cursor while the camera is flying (RMB look) or a
    /// transform op is live; release it otherwise. Locked (cursor frozen in place, raw deltas only)
    /// where the platform supports it (Windows, Wayland); Confined (clamped to the window) on X11,
    /// which refuses Locked. Either way the cursor can never wander over the side panels mid-fly,
    /// and it is hidden so it does not obscure the view. Called once per frame from the viewport
    /// update after both drivers ran.
    fn apply_cursor_capture(&mut self, ctx: &egui::Context) {
        let want = self.flying || self.xform.is_some();
        let mode: u8 = if !want { 0 } else if cfg!(target_os = "windows") { 2 } else {
            // Wayland supports pointer locking; X11 does not (winit returns an error) -> confine.
            let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
                && std::env::var("XDG_SESSION_TYPE").map(|v| v != "x11").unwrap_or(true);
            if wayland { 2 } else { 1 }
        };
        if mode != self.cursor_grab {
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(match mode {
                2 => egui::viewport::CursorGrab::Locked,
                1 => egui::viewport::CursorGrab::Confined,
                _ => egui::viewport::CursorGrab::None,
            }));
            self.cursor_grab = mode;
        }
        if want {
            ctx.set_cursor_icon(egui::CursorIcon::None);
        }
    }

    /// WASD/QE fly + right-drag mouse-look, when the viewport is interacted with.
    fn update_camera(&mut self, ctx: &egui::Context, viewport_response: &egui::Response) {
        let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let cam_before = self.camera.pos;

        // A fly starts when RMB is PRESSED over the viewport and lasts until RMB is
        // released, wherever the (hidden, captured) cursor is by then. While flying, mouse-look
        // integrates the RAW device motion: with the cursor locked in place the pointer position
        // never changes, so egui's drag_delta would be zero.
        let (rmb_pressed, rmb_released) = ctx.input(|i| (i.pointer.secondary_pressed(), i.pointer.secondary_released()));
        if rmb_pressed && viewport_response.hovered() { self.flying = true; }
        if rmb_released || !ctx.input(|i| i.pointer.secondary_down()) { self.flying = false; }
        if self.flying {
            const SENS: f32 = 0.005;
            let d = Self::raw_mouse_motion(ctx).map(|(x, y)| egui::vec2(x, y))
                .unwrap_or_else(|| if viewport_response.dragged_by(egui::PointerButton::Secondary) { viewport_response.drag_delta() } else { egui::Vec2::ZERO });
            self.camera.yaw -= d.x * SENS;
            self.camera.pitch = (self.camera.pitch - d.y * SENS)
                .clamp(-1.5, 1.5);
            ctx.request_repaint();
        }

        // Movement only when the viewport has focus/hover (so typing in panels
        // doesn't fly the camera) AND the right mouse button is held. Gating WASD on
        // RMB-held means the keys are free for editor shortcuts (Ctrl+A select-all,
        // Shift+A menu) when you're not actively flying the camera.
        let rmb_held = ctx.input(|i| i.pointer.secondary_down());
        if (viewport_response.hovered() || viewport_response.dragged() || self.flying) && rmb_held {
            let (fwd, right) = (self.camera.forward(), self.camera.right());
            let up = glam::Vec3::Z;
            let mut v = glam::Vec3::ZERO;
            ctx.input(|i| {
                let boost = if i.modifiers.shift { 4.0 } else { 1.0 };
                let s = self.move_speed * boost * dt;
                if i.key_down(egui::Key::W) { v += fwd * s; }
                if i.key_down(egui::Key::S) { v -= fwd * s; }
                if i.key_down(egui::Key::D) { v += right * s; }
                if i.key_down(egui::Key::A) { v -= right * s; }
                // Up/down fly: Q/E and R/F (R up, F down) — both bindings supported.
                if i.key_down(egui::Key::E) || i.key_down(egui::Key::R) { v += up * s; }
                if i.key_down(egui::Key::Q) || i.key_down(egui::Key::F) { v -= up * s; }
            });
            self.camera.pos += v;
        }
        // Enforce clearance AFTER movement, so flying into a wall slides along it
        // instead of burying the view. Only when the camera actually moved this frame — a parked
        // camera must never creep on its own.
        if self.cam_standoff_on && (self.camera.pos - cam_before).length_squared() > 1e-12 {
            self.camera_standoff(self.cam_standoff);
        }

        // CTRL + scroll over the viewport adjusts fly speed (plain scroll would fire on every
        // stray wheel nudge while framing a shot; requiring Ctrl makes it deliberate).
        if viewport_response.hovered() {
            ctx.input(|i| {
                let scroll = i.raw_scroll_delta.y;
                if scroll != 0.0 && (i.modifiers.ctrl || i.modifiers.command) {
                    self.move_speed = (self.move_speed * (1.0 + scroll * 0.0025)).clamp(0.5, 500.0);
                    self.status = format!("Fly speed {:.1}", self.move_speed);
                }
            });
        }
    }

    // ===================== camera clearance =====================

    /// Distance from the camera to the nearest solid surface in ANY direction, that direction, and
    /// how many probe rays were blocked. Thin wrapper over the scene solver so the interactive app
    /// and the headless diagnostic can never drift apart.
    fn camera_clearance(&self, max: f32) -> (Option<f32>, glam::Vec3, usize) {
        // Through the ObjectScene trait so the Halo 4 scene (triangle soup) answers too
        match self.objscene() {
            Some(s) => s.nearest_solid(self.camera.pos, max),
            None => (None, glam::Vec3::ZERO, 0),
        }
    }

    /// Move the camera until nothing solid is within `clearance` world units.
    /// Returns how far it actually moved.
    fn camera_standoff(&mut self, clearance: f32) -> f32 {
        let Some(scene) = self.objscene() else { return 0.0 }; // Reach or Halo 4 scene
        let before = self.camera.pos;
        self.camera.pos = scene.standoff(before, clearance);
        (self.camera.pos - before).length()
    }

    /// Spawn the selected palette item at a world point via the live forge palette (injected DLL).
    #[cfg(feature = "injection")]
    fn spawn_at(&mut self, p: glam::Vec3) {
        let Some(i) = self.selected_palette else {
            self.spawn_status = "Select a palette entry first.".into();
            return;
        };
        let Some(e) = self.palette.get(i).cloned() else { return };
        // Prefer the spawn-WITH-POSE path (object_placement_data_new/object_new/
        // setpose) so spawns can carry orientation (identity here; a gizmo/preview can
        // supply fwd/up later). Falls back to the original WorldSpawn ring (position-only,
        // VariantCommand) when the spawn-pose MMF isn't connected.
        if let Some(fs) = &self.forge_spawn {
            let t = fs.queue_spawn_and_pose(
                e.palette_index, e.entry_within(), e.variant_within(),
                [p.x, p.y, p.z], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                hms_ipc::METHOD_DEFAULT,
            );
            self.spawn_status = format!("Spawn+pose {} at ({:.1},{:.1},{:.1}) trig {t}", e.display(), p.x, p.y, p.z);
        } else if let Some(ws) = &self.world_spawn {
            let t = ws.queue_spawn_at(e.palette_index, e.entry_within(), e.variant_within(), p.x, p.y, p.z);
            self.spawn_status = format!("Placed {} at ({:.1},{:.1},{:.1}) trig {t}", e.display(), p.x, p.y, p.z);
        } else {
            self.spawn_status = "Forge spawn MMF unavailable (inject into MCC).".into();
        }
    }

    /// Spawn a palette object as a fully-editable offline object (mvar_objects +
    /// mvar_meta) so subsequent set/rotate/scale/shape commands work on it. Returns its datum.
    fn spawn_palette_object(&mut self, i: usize, pos: glam::Vec3) -> Result<(u32, String), String> {
        // A Halo 4 palette row instantiates its full .mvar record at once (h4_app.rs)
        if self.h4_active {
            return self.h4_place_palette_item(i, pos);
        }
        let (obj_tag, name) = self
            .static_palette
            .get(i)
            .cloned()
            .ok_or_else(|| format!("palette index {i} out of range (0..{})", self.static_palette.len()))?;
        let mode = self.objscene().map(|s| s.resolve_object_mode(obj_tag)).unwrap_or(0);
        if mode == 0 || mode == 0xFFFF_FFFF {
            return Err(format!("'{name}' has no render_model (tag {obj_tag:08x})"));
        }
        // Capture the palette (folder,item) so this spawned object can be SAVED into a .mvar.
        let (folder, item) = self
            .scene_ctl
            .as_ref()
            .and_then(|s| s.folder_item_for_obj(obj_tag))
            .unwrap_or((0xFFFF, 0xFF));
        // Variant of the picked palette row (rocket/gauss/etc.) → renders the right turret.
        let variant_sid = self.static_palette_variant.get(i).copied().unwrap_or(0);
        let datum = self.next_dup_datum;
        self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
        self.mvar_objects.push(hms_ipc::ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.into(),
            health: 1.0,
            shield: 1.0,
            mode_tag: mode,
            fwd: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: obj_tag,
            variant_name_sid: variant_sid,
        });
        self.mvar_meta.insert(datum, ObjMeta {
            name: name.clone(),
            folder,
            item,
            pos: pos.into(),
            team: mvar::TEAM_NEUTRAL, // new placements start NEUTRAL, like the game's Forge
            color: -1,
            cached_type: 0,
            spawn_seq: 0,
            respawn: 0,
            label_idx: 0xFFFF,
            label: String::new(),
            // DEFAULT_SPAWNABLE (not 0) — the save-add path copies m.placement over
            // new_placed_object's default, and a 0 here would save an object the game reads
            // but never spawns.
            placement: mvar::PLACEMENT_DEFAULT_SPAWNABLE,
            boundary_shape: 0,
            boundary: [0; 4],
            weapon_clips: 0,
            tele_channel: 0,
            tele_passability: 0,
            location_name: 0xFFFF,
            spawn_rel: -1,
            slot: 0xFFFF,
            flags: Default::default(),
            h4: None,
        });
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
        Ok((datum, name))
    }

    /// Resolve an ObjRef to a palette index.
    fn resolve_objref(&self, obj: &script::ObjRef) -> Result<usize, String> {
        match obj {
            script::ObjRef::Index(i) => Ok(*i),
            script::ObjRef::Name(n) => {
                let nl = n.to_lowercase();
                self.static_palette
                    .iter()
                    .position(|(_, nm)| nm.to_lowercase().contains(&nl))
                    // Halo 4 rows are localized; the raw string ids match too
                    .or_else(|| self.h4_palette_index_by_raw(&nl))
                    .ok_or_else(|| format!("no palette object matching '{n}'"))
            }
        }
    }

    /// Drain queued external commands (from the TCP command server) and run them on the UI
    /// thread. Each request is one script run (one undo step); the output is sent back to the client.
    fn drain_commands(&mut self) {
        let Some(rx) = self.cmd_rx.take() else { return };
        let mut reqs = Vec::new();
        while let Ok(req) = rx.try_recv() {
            reqs.push(req);
            if reqs.len() >= 32 {
                break; // bound work per frame so a flood can't stall the window
            }
        }
        for req in reqs {
            let out = self.run_script(&req.text);
            let _ = req.reply.send(out);
        }
        self.cmd_rx = Some(rx);
    }

    /// Run a multi-line script through the shared program runner (comments, foreach,
    /// substitution). Whole run is ONE undo step.
    fn run_script(&mut self, text: &str) -> String {
        // The step is pushed AFTER the run, and only if the run changed something, so a
        // read-only run (`camera`, `list`, `screenshot`, `outlines get` ...) leaves no no-op
        // undo step behind and does not clear the redo stack. A script that itself steps
        // history (`undo` / `redo`) takes no snapshot at all: its undo IS the step.
        let before = (!script::steps_history(text)).then(|| self.edit_snapshot());
        self.prop_snapshotted_for = None;
        let out = script::run_program(self, text);
        if let Some(before) = before {
            if before != self.edit_snapshot() {
                self.edit_undo.push(before);
                if self.edit_undo.len() > 64 {
                    self.edit_undo.remove(0);
                }
                self.edit_redo.clear();
            }
        }
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
        self.rebuild_overlays();
        self.apply_selection_highlight();
        out
    }

    /// Execute one editor command (shared by the script panel and the command server).
    fn execute_command(&mut self, cmd: script::EditorCommand) -> Result<String, String> {
        use script::{CameraCmd, EditorCommand, Placement};
        match cmd {
            EditorCommand::Place { obj, pos } => {
                let i = self.resolve_objref(&obj)?;
                let world = match pos {
                    Placement::At(p) => glam::Vec3::from(p),
                    Placement::Camera => {
                        let fwd = self.camera.forward();
                        // #construct-h4: `objscene()` -- `place ... at camera` landed 12 wu in the
                        // air on a Halo 4 map because this raycast was Reach-only.
                        self.objscene()
                            .and_then(|s| s.raycast_scene(self.camera.pos, fwd))
                            .unwrap_or(self.camera.pos + fwd * 12.0)
                    }
                    Placement::Relative { datum, off } => {
                        let (p, _, _) = self.movable_pose(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                        p + glam::Vec3::from(off)
                    }
                    Placement::OnFace { datum, dir } => {
                        let (p, _, _) = self.movable_pose(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                        let d = glam::Vec3::from(dir).normalize_or_zero();
                        // Place at the target's face along `dir` (half its extent out).
                        let half = self
                            .objscene()
                            .and_then(|s| s.aabb_of(datum))
                            .map(|(mn, mx)| {
                                let e = glam::Vec3::from(mx) - glam::Vec3::from(mn);
                                0.5 * (e.x * d.x.abs() + e.y * d.y.abs() + e.z * d.z.abs())
                            })
                            .unwrap_or(1.0);
                        p + d * half
                    }
                };
                let (datum, name) = self.spawn_palette_object(i, world)?;
                self.selected_set = vec![datum];
                self.selected_datum = Some(datum);
                Ok(format!("placed '{name}' -> 0x{datum:08X} at ({:.1},{:.1},{:.1})", world.x, world.y, world.z))
            }
            EditorCommand::Select(d) => {
                self.selected_set = vec![d];
                self.selected_datum = Some(d);
                Ok(format!("selected 0x{d:08X}"))
            }
            EditorCommand::SelectAll => {
                self.selected_set = self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| o.datum).collect();
                self.selected_datum = self.selected_set.last().copied();
                Ok(format!("selected {} objects", self.selected_set.len()))
            }
            EditorCommand::SelectBox { min, max, additive } => {
                let n = self.select_world_box(min, max, additive);
                Ok(format!("selected {n} objects (box)"))
            }
            EditorCommand::Deselect => {
                self.selected_set.clear();
                self.selected_datum = None;
                Ok("deselected".into())
            }
            EditorCommand::Delete(which) => {
                let dd: Vec<u32> = which.map(|d| vec![d]).unwrap_or_else(|| self.selected_set.clone());
                if dd.is_empty() {
                    return Err("nothing to delete".into());
                }
                self.mvar_objects.retain(|o| !dd.contains(&o.datum));
                self.local_objects.retain(|o| !dd.contains(&o.datum));
                for d in &dd {
                    self.mvar_meta.remove(d);
                    self.mvar_colors.remove(d);
                }
                self.selected_set.retain(|d| !dd.contains(d));
                self.selected_datum = self.selected_set.last().copied();
                Ok(format!("deleted {} object(s)", dd.len()))
            }
            EditorCommand::Move { datum, delta } => {
                let (p, f, u) = self.movable_pose(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                self.set_movable_pose(datum, p + glam::Vec3::from(delta), f, u);
                Ok(format!("moved 0x{datum:08X}"))
            }
            EditorCommand::MoveTo { datum, pos } => {
                let (_, f, u) = self.movable_pose(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                self.set_movable_pose(datum, glam::Vec3::from(pos), f, u);
                Ok(format!("moved 0x{datum:08X} to ({:.1},{:.1},{:.1})", pos[0], pos[1], pos[2]))
            }
            // Scripted face-to-face coincident — the same constraint the Construct
            // panel applies, moving object `a` only.
            EditorCommand::Coincident { a, af, b, bf, centered, turn } => {
                let scene = self.objscene().ok_or("coincident: no map loaded")?;
                let oa = scene.object_obb(a).ok_or_else(|| format!("coincident: no bounds for 0x{a:08X}"))?;
                let ob = scene.object_obb(b).ok_or_else(|| format!("coincident: no bounds for 0x{b:08X}"))?;
                let bn = ob.face_normal(bf);
                if bn.length_squared() < 0.5 {
                    return Err(format!("coincident: 0x{b:08X} face {bf} has no normal (zero-extent axis)"));
                }
                let mate = if centered { construct::FaceMate::Centered } else { construct::FaceMate::Plane };
                let ac = oa.face_center(af);
                let (q, delta) = construct::face_mate_transform(ac, oa.face_normal(af), ob.face_center(bf), bn, mate, turn);
                let (p, f, u) = self.movable_pose(a).ok_or_else(|| format!("no object 0x{a:08X}"))?;
                let np = ac + q * (p - ac) + delta;
                self.set_movable_pose(a, np, q * f, q * u);
                // Refresh the cached bounds now, so repeating the constraint within one
                // frame is idempotent. The mate both TURNS and moves the part -- update the
                // cached box the same way, or a repeat sees a stale orientation and computes
                // a small phantom correction instead of reporting "already mated".
                if let Some(sc) = self.objscene_mut() {
                    sc.rotate_pick(a, q, ac);
                    sc.translate_pick(a, delta);
                }
                self.refresh_guides();
                self.overlays_dirty = true;
                let how = if centered { "centred" } else { "flush" };
                Ok(format!("coincident: 0x{a:08X} {how} onto 0x{b:08X} (delta {:.3},{:.3},{:.3})", delta.x, delta.y, delta.z))
            }
            // #snap-array: the scripted Ctrl magnet -- the same oriented face snap + edge
            // alignment, applied as a pure translation to the whole moving set.
            EditorCommand::SnapTo { target, to, axis } => {
                let moving: Vec<u32> = match target {
                    Some(d) => vec![d],
                    None => self.movable_datums(),
                };
                if moving.is_empty() {
                    return Err("snapto needs a datum or a selection".into());
                }
                let res = self
                    .snap_once(&moving, to, axis.map(glam::Vec3::from))
                    .ok_or_else(|| format!("snapto: no face within {:.2} wu of the selection", self.magnet_range))?;
                self.push_edit_undo();
                self.prop_snapshotted_for = None;
                for &d in &moving {
                    if let Some((p, f, u)) = self.movable_pose(d) {
                        self.set_movable_pose(d, p + res.delta, f, u);
                    }
                }
                // Keep the cached pick boxes truthful so a following command sees the new pose.
                for &d in &moving {
                    if let Some(sc) = self.objscene_mut() {
                        sc.translate_pick(d, res.delta);
                    }
                }
                if let Some(s) = self.objscene_mut() {
                    s.invalidate();
                }
                self.apply_selection_highlight();
                Ok(format!(
                    "snapped {} object(s) by ({:.3}, {:.3}, {:.3}) — {}",
                    moving.len(),
                    res.delta.x,
                    res.delta.y,
                    res.delta.z,
                    res.describe()
                ))
            }
            // #snap-array: fill a line (or a world axis) with the selection.
            EditorCommand::ArrayLine { target, from, to, count, step, align } => {
                if let Some(d) = target {
                    self.selected_set = vec![d];
                    self.selected_datum = Some(d);
                }
                let (a, b) = (glam::Vec3::from(from), glam::Vec3::from(to));
                let dir = (b - a).normalize_or_zero();
                if dir.length_squared() < 0.5 {
                    return Err("array line: `from` and `to` are the same point".into());
                }
                // Both count AND step given: exactly N copies at that spacing from `from`
                // (the end point then only sets the direction).
                let pts = match (count, step) {
                    (Some(n), Some(w)) => snap::axis_positions(a, dir, n, w),
                    (Some(n), None) => snap::line_positions(a, b, snap::LineSpec::Count(n)),
                    (None, Some(w)) => snap::line_positions(a, b, snap::LineSpec::Step(w)),
                    (None, None) => return Err("array line needs `count N` or `step WU`".into()),
                };
                let made = self.array_at_points(&pts, dir, align)?;
                let s = step.unwrap_or_else(|| snap::count_step(a, b, count.unwrap_or(1)));
                let ext = self.selection_extent_along(dir);
                Ok(format!(
                    "array line: {} copies + the original, step {s:.3} wu ({}) along ({:.2}, {:.2}, {:.2})",
                    made,
                    snap::spacing_label(s, ext),
                    dir.x, dir.y, dir.z
                ))
            }
            EditorCommand::ArrayAxis { target, dir, count, step, align } => {
                if let Some(d) = target {
                    self.selected_set = vec![d];
                    self.selected_datum = Some(d);
                }
                let d = glam::Vec3::from(dir).normalize_or_zero();
                if d.length_squared() < 0.5 {
                    return Err("array axis: bad direction".into());
                }
                let origin = self.selection_obb().map(|o| o.c).ok_or("array axis: nothing selected")?;
                let ext = self.selection_extent_along(d);
                // Default step = the piece's own extent along the axis: exact face-to-face.
                let s = step.unwrap_or(ext).max(0.05);
                let pts = snap::axis_positions(origin, d, count, s);
                let made = self.array_at_points(&pts, d, align)?;
                Ok(format!(
                    "array axis: {} copies + the original, step {s:.3} wu ({})",
                    made,
                    snap::spacing_label(s, ext)
                ))
            }
            EditorCommand::Rotate { datum, axis, deg } => {
                let (p, f, u) = self.movable_pose(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                let a = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][axis];
                let q = glam::Quat::from_axis_angle(a, deg.to_radians());
                self.set_movable_pose(datum, p, q * f, q * u);
                Ok(format!("rotated 0x{datum:08X} {deg}° about {}", ["X", "Y", "Z"][axis]))
            }
            EditorCommand::Settle(which) => {
                let targets: Vec<u32> = match which {
                    Some(d) => vec![d],
                    None => self.selected_set.clone(),
                };
                if targets.is_empty() {
                    return Err("settle needs a datum or a selection".into());
                }
                let mut n = 0;
                for d in &targets {
                    if self.settle_object(*d) {
                        n += 1;
                    }
                }
                if let Some(s) = self.objscene_mut() {
                    s.invalidate();
                }
                self.apply_selection_highlight();
                Ok(format!("settled {n}/{} object(s)", targets.len()))
            }
            EditorCommand::Set { datum, field, value } if datum == script::SET_SELECTION => {
                // `set selection <field> <value>` — same fan-out the properties
                // panel does, so script and UI agree.
                let targets: Vec<u32> = self.selected_set.iter().copied()
                    .filter(|d| self.mvar_meta.contains_key(d)).collect();
                if targets.is_empty() {
                    return Err("set selection: nothing editable selected".into());
                }
                self.push_edit_undo();
                self.prop_snapshotted_for = None;
                let mut n = 0usize;
                let mut last_err: Option<String> = None;
                for d in targets {
                    match self.execute_command(EditorCommand::Set { datum: d, field: field.clone(), value: value.clone() }) {
                        Ok(_) => n += 1,
                        Err(e) => last_err = Some(e),
                    }
                }
                if n == 0 {
                    self.edit_undo.pop();
                    return Err(last_err.unwrap_or_else(|| "set selection: no editable objects".into()));
                }
                Ok(format!("set {field} on {n} object(s)"))
            }
            EditorCommand::Set { datum, field, value } => {
                let mut labels = self.mvar_labels.clone(); // label names for the Halo 4 fields
                let m = self.mvar_meta.get_mut(&datum).ok_or_else(|| format!("no editable object 0x{datum:08X}"))?;
                // A Halo 4 object takes its own field names first (h4_app.rs); the
                // shared names (team / color / label ...) fall through to the match below.
                // An unknown label NAME is appended to the variant's table (like the
                // panel's "+ name"), so `set <d> label scale` works on a variant without that label.
                let h4_handled = Self::h4_set_field(m, field.as_str(), &value, &mut labels)?;
                if labels.len() != self.mvar_labels.len() {
                    self.mvar_labels = labels;
                    self.variant_header_dirty = true;
                }
                if h4_handled {
                    let (team, color) = (m.team, m.color);
                    let cu8 = if color < 0 { 0xFFu8 } else { color as u8 };
                    self.mvar_colors.insert(datum, (team, cu8));
                    if matches!(field.as_str(), "shape" | "boundary" | "radius" | "width" | "length" | "top" | "bottom" | "b0" | "b1" | "b2" | "b3") {
                        self.rebuild_overlays();
                    }
                    return Ok(format!("set 0x{datum:08X} {field}={value}"));
                }
                match field.as_str() {
                    "team" => m.team = script::parse_team(&value)?,
                    "color" => m.color = value.parse::<i32>().map_err(|_| "color must be -1..7")?,
                    "label" => { m.label = value.clone(); }
                    // Per-object pseudo-flag overrides (default = re-derive).
                    "scaled" => m.flags.scaled = forge_scale::parse_flag_value(&value)?,
                    "shadow" | "castshadow" | "shadowcaster" => m.flags.shadow = forge_scale::parse_flag_value(&value)?,
                    "spawnseq" | "spawn_seq" => m.spawn_seq = value.parse().map_err(|_| "spawnseq must be an int")?,
                    "scale" => {
                        // via X330: needs the "scale" label, then encode into spawn_seq.
                        m.label = "scale".into();
                        let s: f32 = value.parse().map_err(|_| "scale must be a number")?;
                        m.spawn_seq = forge_scale::object_scale_to_seq(s).0 as i32;
                    }
                    "cachedtype" | "cached_type" => m.cached_type = value.parse().map_err(|_| "cached_type must be int")?,
                    "respawn" => m.respawn = value.parse().map_err(|_| "respawn must be int")?,
                    "shape" | "boundary" => {
                        m.boundary_shape = match value.to_lowercase().as_str() {
                            "none" => 0, "sphere" => 1, "cylinder" => 2, "box" => 3,
                            _ => return Err("shape must be none/sphere/cylinder/box".into()),
                        };
                    }
                    // Boundary size: `radius`/`width` world units → b0; b1..b3 by slot (raw 0..2047 or,
                    // for width/length/top/bottom aliases, world units). Slots: sphere=[r]; cylinder=
                    // [r,top,bottom]; box=[width,length,top,bottom].
                    "radius" | "width" => m.boundary[0] = wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "length" => m.boundary[1] = wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "top" => m.boundary[2] = wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "bottom" => m.boundary[3] = wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "b0" => m.boundary[0] = value.parse().map_err(|_| "b0 must be 0..2047")?,
                    "b1" => m.boundary[1] = value.parse().map_err(|_| "b1 must be 0..2047")?,
                    "b2" => m.boundary[2] = value.parse().map_err(|_| "b2 must be 0..2047")?,
                    "b3" => m.boundary[3] = value.parse().map_err(|_| "b3 must be 0..2047")?,
                    other => return Err(format!("unknown field '{other}'")),
                }
                // Push colour to the live render map when team/color changed.
                let (team, color) = (m.team, m.color);
                let cu8 = if color < 0 { 0xFFu8 } else { color as u8 };
                self.mvar_colors.insert(datum, (team, cu8));
                Ok(format!("set 0x{datum:08X} {field}={value}"))
            }
            EditorCommand::Camera(c) => {
                match c {
                    CameraCmd::To(p) => self.camera.pos = glam::Vec3::from(p),
                    CameraCmd::Nudge(d) => self.camera.pos += glam::Vec3::from(d),
                    CameraCmd::LookAt(p) => {
                        let to = (glam::Vec3::from(p) - self.camera.pos).normalize_or_zero();
                        self.camera.yaw = to.y.atan2(to.x);
                        self.camera.pitch = to.z.clamp(-1.0, 1.0).asin();
                    }
                    CameraCmd::Spawn => {
                        if let Some((pos, yaw, pitch)) = self.variant_spawn_camera().or_else(|| self.objscene().and_then(|s| s.spawn_camera_pose())) {
                            self.camera.pos = pos;
                            self.camera.yaw = yaw;
                            self.camera.pitch = pitch;
                        }
                    }
                    CameraCmd::Frame => self.frame_on_player(),
                    CameraCmd::Get => {
                        return Ok(format!(
                            "camera pos {:.2} {:.2} {:.2}  yaw {:.1}°  pitch {:.1}°  fly {:.1}",
                            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z,
                            self.camera.yaw.to_degrees(), self.camera.pitch.to_degrees(), self.move_speed
                        ));
                    }
                    CameraCmd::Clearance => {
                        // Probe generously so "how much room do I have?" gets a real answer.
                        let probe = (self.cam_standoff * 8.0).max(24.0);
                        let (d, dir, blocked) = self.camera_clearance(probe);
                        return Ok(match d {
                            Some(t) => format!(
                                "nearest surface {t:.2} wu toward ({:.2},{:.2},{:.2}) — {blocked}/64 probe rays blocked within {probe:.0} wu",
                                dir.x, dir.y, dir.z
                            ),
                            None => format!("clear — nothing solid within {probe:.0} wu"),
                        });
                    }
                    CameraCmd::Standoff(n) => {
                        let want = n.unwrap_or(self.cam_standoff);
                        let moved = self.camera_standoff(want);
                        let (after, _, _) = self.camera_clearance(want);
                        return Ok(format!(
                            "stand-off {want:.2} wu — camera moved {moved:.2} wu, nearest surface now {}",
                            after.map(|t| format!("{t:.2} wu")).unwrap_or_else(|| format!(">{want:.2} wu"))
                        ));
                    }
                    CameraCmd::Orbit { target, dist, yaw, pitch } => {
                        let focus = match target {
                            script::OrbitTarget::Point(p) => glam::Vec3::from(p),
                            script::OrbitTarget::Datum(d) => self
                                .objscene()
                                .and_then(|s| s.object_obb(d))
                                .map(|o| o.c)
                                .or_else(|| self.movable_pose(d).map(|(p, _, _)| p))
                                .ok_or_else(|| format!("no object 0x{d:08X}"))?,
                            script::OrbitTarget::Selection => self
                                .selection_obb()
                                .map(|o| o.c)
                                .ok_or("nothing selected")?,
                        };
                        let (y, p) = (yaw.to_radians(), pitch.to_radians().clamp(-1.5, 1.5));
                        let off = glam::Vec3::new(y.cos() * p.cos(), y.sin() * p.cos(), p.sin()) * dist;
                        self.camera.pos = focus - off;
                        self.camera.yaw = y;
                        self.camera.pitch = p;
                        return Ok(format!(
                            "orbiting ({:.2},{:.2},{:.2}) at {dist:.1} wu, yaw {yaw:.0}° pitch {pitch:.0}°",
                            focus.x, focus.y, focus.z
                        ));
                    }
                }
                Ok("camera set".into())
            }
            EditorCommand::ListPalette(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                let mut s = String::new();
                let mut n = 0;
                for (i, (_, name)) in self.static_palette.iter().enumerate() {
                    if !f.is_empty() && !name.to_lowercase().contains(&f) {
                        continue;
                    }
                    s.push_str(&format!("  #{i}  {name}\n"));
                    n += 1;
                    if n >= 200 {
                        s.push_str("  … (truncated)\n");
                        break;
                    }
                }
                Ok(format!("{n} palette objects:\n{s}"))
            }
            EditorCommand::Save { path } => {
                // Scripted counterpart of File ▸ Save. `save` overwrites the open variant;
                // `save as <path>` writes a copy, leaving the original alone.
                // A script is non-interactive, so the save is NOT held; the warning is
                // appended to the output instead (`bspwarn list` names the objects).
                let outside = self.bsp_warn_objects().len();
                let bsp_note = if outside > 0 { format!(" — WARNING: {outside} object(s) outside playable space (see `bspwarn list`)") } else { String::new() };
                match path {
                    Some(p) => {
                        let dst = std::path::PathBuf::from(p.trim());
                        let (n, failed) = self.save_variant_to(&dst)?;
                        let note = self.save_failure_note();
                        mapcat::save_last_variant(&dst.to_string_lossy()); // #dialogs: last-saved folder
                        Ok(format!("saved {n} objects -> {}{note}{}{bsp_note}", dst.display(),
                            if failed > 0 && note.is_empty() { format!(" ({failed} not saved)") } else { String::new() }))
                    }
                    None => {
                        // Halo 4 saves in place like Reach (gates + hopper-dir refusal in h4_app.rs)
                        let dst = self.current_variant_path.clone().ok_or("no variant is open — use `save as <path>`")?;
                        let (n, failed) = self.save_variant_to(&dst)?;
                        let note = self.save_failure_note();
                        self.resync_variant_globals(&dst);
                        mapcat::save_last_variant(&dst.to_string_lossy()); // #dialogs: last-saved folder
                        Ok(format!("saved {n} objects -> {}{note}{}{bsp_note}", dst.display(),
                            if failed > 0 && note.is_empty() { format!(" ({failed} not saved)") } else { String::new() }))
                    }
                }
            }
            EditorCommand::Dup { datum, off } => {
                // Duplicate honours the same selection rules the UI uses, so a script and a
                // Shift+D produce the same result.
                if let Some(d) = datum {
                    self.selected_set = vec![d];
                    self.selected_datum = Some(d);
                }
                if self.movable_datums().is_empty() {
                    return Err("nothing to duplicate (select an object, or pass a datum)".into());
                }
                self.push_edit_undo();
                let dups = self.duplicate_selection();
                if dups.is_empty() {
                    return Err("nothing could be duplicated (objects must be offline-editable)".into());
                }
                let delta = glam::Vec3::from(off);
                if delta.length_squared() > 0.0 {
                    for &nd in &dups {
                        if let Some((p, f, u)) = self.movable_pose(nd) {
                            self.set_movable_pose(nd, p + delta, f, u);
                        }
                    }
                }
                self.selected_set = dups.clone();
                self.selected_datum = dups.last().copied();
                self.apply_selection_highlight();
                Ok(format!("duplicated {} object(s) -> {}", dups.len(),
                    dups.iter().map(|d| format!("0x{d:08X}")).collect::<Vec<_>>().join(" ")))
            }
            EditorCommand::Count { type_filter, name_filter, label_filter } => {
                let tf = type_filter.map(|s| s.to_lowercase());
                let nf = name_filter.map(|s| s.to_lowercase());
                let lf = label_filter.map(|s| s.to_lowercase());
                let mut n = 0usize;
                for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                    let meta = self.mvar_meta.get(&o.datum);
                    let name = meta.map(|m| m.name.clone()).unwrap_or_default();
                    let label = meta.map(|m| m.label.clone()).unwrap_or_default();
                    let ctype = meta.map(|m| m.cached_type).unwrap_or(0);
                    if let Some(tf) = &tf {
                        let num_ok = tf.parse::<u8>().map(|v| v == ctype).unwrap_or(false);
                        if !num_ok && !name.to_lowercase().contains(tf) { continue; }
                    }
                    if nf.as_ref().map_or(false, |nf| !name.to_lowercase().contains(nf)) { continue; }
                    if lf.as_ref().map_or(false, |lf| !label.to_lowercase().contains(lf)) { continue; }
                    n += 1;
                }
                // 651 is the variant's hard slot ceiling — report headroom, since that is the
                // number people actually need when deciding what else they can place.
                let total = self.mvar_objects.len() + self.local_objects.len() + self.mvar_unresolved.len();
                Ok(format!("{n} object(s) match — variant holds {total}/651 ({} free)", 651usize.saturating_sub(total)))
            }
            EditorCommand::ListObjects { type_filter, name_filter, label_filter } => {
                let tf = type_filter.map(|s| s.to_lowercase());
                let nf = name_filter.map(|s| s.to_lowercase());
                let lf = label_filter.map(|s| s.to_lowercase());
                let mut s = String::new();
                let mut n = 0;
                for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                    let meta = self.mvar_meta.get(&o.datum);
                    let name = meta.map(|m| m.name.clone()).unwrap_or_default();
                    let label = meta.map(|m| m.label.clone()).unwrap_or_default();
                    let ctype = meta.map(|m| m.cached_type).unwrap_or(0);
                    if let Some(tf) = &tf {
                        // match cached_type as a number OR the object name/label containing the text
                        let num_ok = tf.parse::<u8>().map(|v| v == ctype).unwrap_or(false);
                        if !num_ok && !name.to_lowercase().contains(tf) {
                            continue;
                        }
                    }
                    if let Some(nf) = &nf {
                        if !name.to_lowercase().contains(nf) {
                            continue;
                        }
                    }
                    if let Some(lf) = &lf {
                        if !label.to_lowercase().contains(lf) {
                            continue;
                        }
                    }
                    s.push_str(&format!("  0x{:08X}  ({:.1},{:.1},{:.1})  type={ctype}  {name}{}\n",
                        o.datum, o.pos[0], o.pos[1], o.pos[2],
                        if label.is_empty() { String::new() } else { format!("  [{label}]") }));
                    n += 1;
                    if n >= 500 {
                        s.push_str("  … (truncated)\n");
                        break;
                    }
                }
                Ok(format!("{n} objects:\n{s}"))
            }
            EditorCommand::ListTypes => {
                // Distinct object types present in the scene, by name → count.
                let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                    let name = self.mvar_meta.get(&o.datum).map(|m| m.name.clone()).unwrap_or_else(|| format!("tag_{:08X}", o.primary_tag));
                    *hist.entry(name).or_default() += 1;
                }
                let mut v: Vec<(String, usize)> = hist.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let mut s = String::new();
                for (name, c) in &v {
                    s.push_str(&format!("  {c:>4}  {name}\n"));
                }
                Ok(format!("{} distinct types:\n{s}", v.len()))
            }
            EditorCommand::ListMaps(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                let mut s = String::new();
                let mut n = 0;
                for c in &self.map_candidates {
                    let label = c.label();
                    if !f.is_empty() && !label.to_lowercase().contains(&f) {
                        continue;
                    }
                    let id = c.map_id.map(|v| format!("0x{v:04X}")).unwrap_or_else(|| "?".into());
                    let here = c.game == self.picker_game && c.modded == self.show_modded_maps;
                    s.push_str(&format!("  [{}]{} id={id}  {label}\n", if c.modded { "mod" } else { "stk" }, if here { "*" } else { " " }));
                    n += 1;
                }
                // #map-picker: say which GAME + kind the picker is showing (the rows marked *),
                // so a script can see the same two-level grouping the panel does.
                let count = |g: mapcat::Game, m: bool| self.map_candidates.iter().filter(|c| c.game == g && c.modded == m).count();
                let games: Vec<String> = [mapcat::Game::Reach, mapcat::Game::Halo4, mapcat::Game::H2A]
                    .into_iter()
                    .filter(|g| count(*g, false) + count(*g, true) > 0)
                    .map(|g| format!("{} ({})", g.display_name(), count(g, false) + count(g, true)))
                    .collect();
                let head = format!(
                    "picker: {} / {} — built-in {}, modded {} | games: {}\n",
                    self.picker_game.display_name(),
                    if self.show_modded_maps { "Modded" } else { "Built-in" },
                    count(self.picker_game, false),
                    count(self.picker_game, true),
                    games.join(", ")
                );
                Ok(format!("{head}{n} maps:\n{s}"))
            }
            EditorCommand::ListVariants { sel, filter } => {
                let (items, s) = self.variant_listing(&sel, filter.as_deref());
                Ok(format!("{} variants:\n{s}", items.len()))
            }
            EditorCommand::MapId(r) => {
                let id = self.resolve_map_id(&r)?;
                Ok(format!("0x{id:08X} ({id})"))
            }
            EditorCommand::Get(datum) => Ok(self.dump_object(datum)),
            EditorCommand::LoadMap(name) => {
                let idx = self.resolve_map_index(&name).ok_or_else(|| format!("no map matching '{name}'"))?;
                let label = self.map_candidates[idx].label();
                self.show_modded_maps = self.map_candidates[idx].modded;
                self.picker_game = self.map_candidates[idx].game; // #map-picker
                self.selected_map = Some(idx);
                self.load_map();
                Ok(format!("loading map {label} (streams over frames — use 'wait' headless)"))
            }
            EditorCommand::LoadVariant(path) => {
                let p = std::path::PathBuf::from(&path);
                if !p.exists() {
                    return Err(format!("variant not found: {path}"));
                }
                self.import_mvar(p);
                Ok(format!("loading variant {path}"))
            }
            EditorCommand::Screenshot { path, size } => {
                let _ = size; // interactive screenshot uses the live viewport size
                self.screenshot_to(std::path::Path::new(&path));
                Ok(format!("screenshot -> {path}"))
            }
            EditorCommand::Wait => Ok("wait: interactive load streams over frames; let it settle".into()),
            EditorCommand::Pick { delete, radius } => {
                // Raycast from the LIVE camera forward — "what am I looking at".
                let dir = self.camera.forward();
                let hit = self
                    .objscene()
                    .and_then(|s| s.pick(self.camera.pos, dir))
                    .ok_or("nothing under the camera crosshair")?;
                let center = self.movable_pose(hit).map(|(p, _, _)| p).unwrap_or(glam::Vec3::ZERO);
                let mut set = vec![hit];
                if let Some(r) = radius {
                    let r2 = r * r;
                    for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                        if o.datum != hit && (glam::Vec3::from(o.pos) - center).length_squared() <= r2 {
                            set.push(o.datum);
                        }
                    }
                }
                let name = self.mvar_meta.get(&hit).map(|m| m.name.clone()).unwrap_or_default();
                if delete {
                    self.mvar_objects.retain(|o| !set.contains(&o.datum));
                    self.local_objects.retain(|o| !set.contains(&o.datum));
                    for d in &set {
                        self.mvar_meta.remove(d);
                        self.mvar_colors.remove(d);
                    }
                    self.selected_set.clear();
                    self.selected_datum = None;
                    Ok(format!("deleted {} object(s) under crosshair (nearest: 0x{hit:08X} {name})", set.len()))
                } else {
                    self.selected_set = set.clone();
                    self.selected_datum = Some(hit);
                    // Say WHAT was hit (tag path / class / tag id), not just the datum.
                    let what = match self.object_identity(hit) {
                        Some(id) => id.status_line(),
                        None => format!("0x{hit:08X} {name}"),
                    };
                    Ok(format!("under crosshair: {what} — {} object(s) selected", set.len()))
                }
            }
            EditorCommand::Echo(t) => Ok(t),
            EditorCommand::ScreenFx(set) => {
                if let Some(on) = set {
                    self.forge_fx_enabled = on;
                    screenfx::save_enabled_setting(on);
                    let objs = self.last_objects.clone();
                    self.refresh_screen_fx(&objs);
                }
                Ok(self.screenfx_status())
            }
            // The global switches are real persisted settings; the rebuild tick picks
            // them up next frame (scales / casters are re-derived every tick).
            EditorCommand::ScaledGlobal(set) => {
                if let Some(on) = set {
                    self.obj_globals.scaled = on;
                    self.obj_globals.save();
                }
                Ok(self.obj_flags_status())
            }
            EditorCommand::ShadowCastersGlobal(set) => {
                if let Some(on) = set {
                    self.obj_globals.shadowcasters = on;
                    self.obj_globals.save();
                }
                Ok(self.obj_flags_status())
            }
            // Persisted like the View checkbox; tick_scene re-merges next frame.
            EditorCommand::MapSpawns(set) => {
                if let Some(on) = set {
                    self.show_map_spawns = on;
                    map_spawns::save_setting(on);
                    self.renderer.set_show_markers(on); // the Halo 4 marker lane
                }
                Ok(self.map_spawns_status())
            }
            // Persisted like the View checkbox; rebuild_overlays runs after the script.
            EditorCommand::PhysicsOutlines(set) => {
                if let Some(on) = set {
                    self.show_blockers = on;
                    physics_outlines::save_setting(on);
                    self.overlays_dirty = true;
                }
                Ok(self.physics_outlines_status())
            }
            // #h4-phys The two selected-object hull overlays, exactly as the View rows set them
            // (both games: Reach reads its native walkers, Halo 4 / H2A `h4::collision`).
            EditorCommand::CollisionOverlay(set) => {
                if let Some(on) = set { self.show_collision = on; self.overlays_dirty = true; }
                Ok(self.hull_overlay_status())
            }
            EditorCommand::PhysicsOverlay(set) => {
                if let Some(on) = set { self.show_physics = on; self.overlays_dirty = true; }
                Ok(self.hull_overlay_status())
            }
            // `// #h4-expo-3` Opening the window must not change a single pixel; a script can now
            // capture a frame, open it, and capture again.
            EditorCommand::SettingsPanel(set) => {
                if let Some(on) = set { self.show_settings = on; }
                Ok(format!("settings panel {}", if self.show_settings { "open" } else { "closed" }))
            }
            // #dialogs test hook (print-only): drive the variant browser without a mouse.
            EditorCommand::DialogDbg { sub, a1, a2 } => {
                match sub.as_str() {
                    "open" => self.open_variant_browser(),
                    "saveas" => self.save_variant_as(),
                    "savebrowser" | "save" => {
                        self.open_variant_save_browser();
                        if let Some(n) = &a1 { self.variant_browser_filename = n.clone(); }
                    }
                    "size" => match a1.as_deref() {
                        Some("reset") | None => self.variant_browser_force_size = None,
                        Some(w) => {
                            let ww: f32 = w.parse().map_err(|_| "dialog size <W> <H>|reset")?;
                            let hh: f32 = a2.as_deref().ok_or("dialog size <W> <H>")?.parse().map_err(|_| "dialog size <W> <H>")?;
                            self.variant_browser_force_size = Some([ww, hh]);
                        }
                    },
                    "shot" => {
                        let p = a1.as_deref().ok_or("dialog shot <path>")?;
                        self.egui_shot_request = Some(std::path::PathBuf::from(p));
                    }
                    // Headless equivalents of navigating / pressing Save / confirming overwrite.
                    "nav" => {
                        let d = std::path::PathBuf::from(a1.as_deref().ok_or("dialog nav <dir>")?);
                        let keep = self.variant_browser_save.then(|| self.variant_browser_filename.clone());
                        self.vb_navigate(&d, true);
                        if let Some(n) = keep { self.variant_browser_filename = n; }
                        self.variant_browser_overwrite = None;
                    }
                    "setname" => { self.variant_browser_filename = a1.clone().unwrap_or_default(); }
                    "dosave" => { if self.browser_try_save() { self.variant_browser_open = false; } }
                    "confirmsave" => { if self.browser_confirm_overwrite() { self.variant_browser_open = false; } }
                    "close" => self.variant_browser_open = false,
                    "status" | "resolve" => {}
                    other => return Err(format!("dialog: unknown '{other}' (open|saveas|savebrowser|size|shot|close|status)")),
                }
                let dir = self.variant_browser_dir.clone().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
                let file = self.variant_browser_filename.clone();
                let resolved = self.variant_browser_dir.clone()
                    .and_then(|d| Self::resolve_save_target(&d, &file))
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let overwrite = self.variant_browser_overwrite.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
                Ok(format!("dialog open={} save_mode={} dir={} file={} resolved={} overwrite={} status={}",
                    self.variant_browser_open, self.variant_browser_save, dir, file, resolved, overwrite, self.spawn_status))
            }
            // #wire-visible  Persisted like the View checkbox; applied straight to the lane.
            EditorCommand::WireXray(set) => {
                if let Some(on) = set {
                    self.wire_xray = on;
                    wire_xray::save_setting(on);
                    self.renderer.set_highlight_xray(on);
                }
                Ok(self.wire_xray_status())
            }
            // Persisted like the View checkbox; `get` also lists every ceiling.
            EditorCommand::SoftCeilings(set) => {
                if let Some(on) = set {
                    self.show_soft_ceilings = on;
                    soft_ceilings::save_setting(on);
                    self.overlays_dirty = true;
                }
                let mut out = self.soft_ceilings_status();
                if set.is_none() {
                    let cs = self.scene_ctl.as_ref().map(|s| s.soft_ceilings()).unwrap_or_default();
                    for l in soft_ceilings::listing(&cs) {
                        out.push('\n');
                        out.push_str(&l);
                    }
                }
                Ok(out)
            }
            // Persisted like the View checkbox; `get` lists every structure BSP.
            EditorCommand::HardFloor(set) => {
                if let Some(on) = set {
                    self.show_hard_floor = on;
                    hard_floor::save_setting(on);
                    self.overlays_dirty = true;
                }
                let mut out = self.hard_floor_status();
                if set.is_none() {
                    if let Some(sc) = self.scene_ctl.as_ref() {
                        for l in hard_floor::listing(sc.structure_bsp_flags(), &sc.structure_bsp_mopp_bounds()) {
                            out.push('\n');
                            out.push_str(&l);
                        }
                    }
                }
                Ok(out)
            }
            // The playable-BSP boxes (persisted).
            EditorCommand::PlayableBounds(set) => {
                if let Some(on) = set {
                    self.show_playable_bounds = on;
                    hard_floor::save_playable_setting(on);
                    self.overlays_dirty = true;
                }
                Ok(self.playable_bounds_status())
            }
            // The View > Trigger volumes switch (not persisted).
            EditorCommand::TriggerVolumes(set) => {
                if let Some(on) = set {
                    self.show_triggers = on;
                    self.overlays_dirty = true;
                }
                let n = self.scene_ctl.as_ref().map(|s| s.trigger_volumes().len()).unwrap_or(0);
                Ok(format!("trigger volumes: {} ({n} volume(s))", if self.show_triggers { "shown" } else { "hidden" }))
            }
            EditorCommand::NewVariant => {
                self.new_variant();
                Ok(self.spawn_status.clone())
            }
            // Step the offline edit history. run_script skips its own "whole run is
            // one undo step" snapshot for a script that contains undo/redo (script::steps_history),
            // so these act on the REAL previous edit, exactly like Ctrl+Z / Ctrl+Y.
            EditorCommand::Undo => {
                let had = !self.edit_undo.is_empty();
                self.do_undo();
                Ok(if had { "undo".into() } else { "undo: nothing to undo".into() })
            }
            EditorCommand::Redo => {
                let had = !self.edit_redo.is_empty();
                self.do_redo();
                Ok(if had { "redo".into() } else { "redo: nothing to redo".into() })
            }
            EditorCommand::FlagsGet => {
                let mut out = self.obj_flags_status();
                let mut sel: Vec<u32> = self.selected_set.clone();
                sel.sort();
                for d in sel {
                    if let Some(m) = self.mvar_meta.get(&d) {
                        out.push('\n');
                        out.push_str(&obj_flags_line(d, m, &self.obj_globals, self.sc_convention));
                    }
                }
                Ok(out)
            }
            // The print-only test hook for the dropdown hover -- SAME code path as the
            // panel (set_hover_preview), pinned until `preview off` so a script can screenshot it.
            EditorCommand::PreviewHover { team, color, off } => {
                if off {
                    self.hover_preview_pinned = false;
                    self.clear_hover_preview();
                } else if let Some(entry) = team.map(color_hover::HoverEntry::Team).or(color.map(color_hover::HoverEntry::Color)) {
                    if self.selected_set.is_empty() {
                        return Err("preview: nothing selected".into());
                    }
                    self.set_hover_preview(entry);
                }
                Ok(self.hover_preview_status())
            }
            // Scripted pointer through egui's own input pipeline (see drive_sim_pointer).
            EditorCommand::PointerSim(sim) => {
                match sim {
                    script::PointerSim::Move { x, y, frames } => { self.sim_pointer = Some((egui::pos2(x, y), frames.max(1))); }
                    script::PointerSim::Click { x, y } => { self.sim_click = Some((egui::pos2(x, y), 0)); }
                    script::PointerSim::Drag { x1, y1, x2, y2 } => { self.sim_drag = Some((egui::pos2(x1, y1), egui::pos2(x2, y2), 0)); }
                    script::PointerSim::Key { name } => {
                        // egui's names are capitalised ("Enter", "Escape", "R"), so accept any
                        // casing the caller types.
                        let mut cap = name.clone();
                        if let Some(c) = cap.get_mut(0..1) { c.make_ascii_uppercase(); }
                        if cap.len() > 1 { cap[1..].make_ascii_lowercase(); }
                        let key = egui::Key::from_name(&name)
                            .or_else(|| egui::Key::from_name(&name.to_uppercase()))
                            .or_else(|| egui::Key::from_name(&cap))
                            .ok_or_else(|| format!("preview key: '{name}' is not an egui key name"))?;
                        let mk = |pressed| egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers: Default::default() };
                        self.sim_events.push(vec![mk(true)]);
                        self.sim_events.push(vec![mk(false)]);
                    }
                    script::PointerSim::Type { text } => {
                        // One CHARACTER per frame: the app's numeric entry reads Text events as
                        // they arrive, and a real keyboard never delivers a whole word at once.
                        for ch in text.chars() {
                            self.sim_events.push(vec![egui::Event::Text(ch.to_string())]);
                        }
                    }
                    script::PointerSim::Where => {}
                }
                Ok(self.hover_where_status())
            }
            // #construct-h4: the Construct tool's own state, scriptable, so the CAD workflow can
            // be driven (and regression-tested) the same way on Reach and on Halo 4.
            EditorCommand::Construct(c) => self.construct_command(c),
            EditorCommand::VariantGet(field) => self.variant_globals_report(field.as_deref()),
            EditorCommand::VariantSet { field, value } => self.variant_set(&field, &value),
            // `bspwarn list` — the same list the save-time window shows.
            EditorCommand::BspWarnList => {
                let Some(scene) = self.objscene() else { return Err("no map loaded".into()) };
                let mut out = scene.bsp_flags_report();
                let rows = self.bsp_warn_objects();
                out.push_str(&format!("{} object(s) outside playable space", rows.len()));
                for (d, i) in rows {
                    out.push_str(&format!("\n  0x{d:08X}  {}", self.bsp_warn_row(d, i)));
                }
                Ok(out)
            }
        }
    }

    /// "hidden-block hulls: shown (252 hull triangle(s)) | zone_vtx=...".
    /// Appends the renderer's live overlay/highlight vertex counts so a scripted run can prove what
    /// is actually uploaded (e.g. `highlight_vtx=0` after a map load with nothing selected).
    /// #wire-visible  "selection wireframe: draws THROUGH objects (N edge(s) selected)".
    fn wire_xray_status(&self) -> String {
        let n: usize = self.objscene()
            .map(|sc| self.selected_set.iter().filter_map(|d| sc.selection_wireframe(*d)).map(|w| w.len()).sum())
            .unwrap_or(0);
        wire_xray::status_line(self.wire_xray, n)
    }

    /// #h4-phys One line for the two selected-object hull overlays: their state plus how many
    /// edges the SELECTED object contributes right now (0 = the object resolves no such model).
    fn hull_overlay_status(&self) -> String {
        let (mut ce, mut pe) = (0usize, 0usize);
        if let (Some(d), Some(s)) = (self.selected_datum, self.objscene()) {
            if let Some(o) = self.last_objects.iter().find(|o| o.datum == d) {
                ce = s.collision_world_edges(o).len();
                pe = s.physics_world_edges(o).len();
            }
        }
        let (n, natt) = match (self.selected_datum, self.objscene()) {
            (Some(d), Some(s)) => s.pick_entry_counts(d),
            _ => (0, 0),
        };
        format!(
            "collision hull: {} ({ce} edge(s)) | physics hull: {} ({pe} edge(s)) | selection geometry: {n} pick entr(ies), {natt} attachment(s)",
            if self.show_collision { "shown" } else { "hidden" },
            if self.show_physics { "shown" } else { "hidden" },
        )
    }

    fn physics_outlines_status(&self) -> String {
        let tris = self.objscene().map(|s| s.blocker_overlay().0.len() / 3).unwrap_or(0);
        format!("{} | {}", physics_outlines::status_line(self.show_blockers, tris), self.renderer.debug_draw_counts())
    }

    /// "hard floor: hidden (world box x ... z -75.2..1265.1; floor z -75.2; 2 BSP(s), mopp bounds +-64)".
    fn hard_floor_status(&self) -> String {
        let wb = self.scene_ctl.as_ref().and_then(|s| hard_floor::world_box(s.structure_bsp_flags(), &s.structure_bsp_mopp_bounds()));
        hard_floor::status_line(self.show_hard_floor, wb.as_ref())
    }

    /// "playable bounds: hidden (1 playable BSP(s) of 2; floor z -25.0)".
    fn playable_bounds_status(&self) -> String {
        let bsps = self.scene_ctl.as_ref().map(|s| s.structure_bsp_flags()).unwrap_or(&[]);
        hard_floor::playable_status_line(self.show_playable_bounds, bsps)
    }

    /// "soft ceilings: hidden (5 ceiling(s): 2 soft kill, ...; 812 triangle(s))".
    fn soft_ceilings_status(&self) -> String {
        let cs = self.scene_ctl.as_ref().map(|s| s.soft_ceilings()).unwrap_or_default();
        soft_ceilings::status_line(self.show_soft_ceilings, &cs)
    }

    /// "map spawns: hidden (16 scenario spawn marker(s) not drawn)".
    fn map_spawns_status(&self) -> String {
        let n = if self.h4_active { self.h4_map_spawn_markers } else { self.scenario_objects.iter().filter(|o| self.scenario_spawn_tags.contains(&o.primary_tag)).count() };
        map_spawns::status_line(self.show_map_spawns, n)
    }

    /// One-line summary of the globals + how many objects they currently affect.
    fn obj_flags_status(&self) -> String {
        let (mut n_scaled, mut n_cast) = (0usize, 0usize);
        for m in self.mvar_meta.values() {
            let (fs, fc) = forge_scale::effective_flags(&self.obj_globals, &m.flags, m.team, &m.label);
            n_scaled += fs as usize;
            n_cast += fc as usize;
        }
        let (cm, ci, tm, ti) = self.renderer.shadow_caster_counts();
        format!("{} | {n_scaled} scaled, {n_cast} casting | shadow pass: {ci}/{ti} object instances ({cm}/{tm} meshes)", self.obj_globals.status_line())
    }

    /// Resolve a map name/substring/path to a catalog index.
    fn resolve_map_index(&self, want: &str) -> Option<usize> {
        // Exact path first, then exact stem, then substring on label/stem/path.
        let wl = want.to_lowercase();
        if let Some(i) = self.map_candidates.iter().position(|c| c.path.to_string_lossy().eq_ignore_ascii_case(want)) {
            return Some(i);
        }
        self.map_candidates
            .iter()
            .position(|c| c.stem.to_lowercase() == wl)
            .or_else(|| self.map_candidates.iter().position(|c| {
                c.label().to_lowercase().contains(&wl)
                    || c.stem.to_lowercase().contains(&wl)
                    || c.path.to_string_lossy().to_lowercase().contains(&wl)
            }))
    }

    /// Resolve a MapRef → numeric map id.
    fn resolve_map_id(&self, r: &script::MapRef) -> Result<u32, String> {
        match r {
            script::MapRef::Current => self
                .loaded_map_id
                .or_else(|| mapcat::read_map_id(std::path::Path::new(&self.map_path)))
                .ok_or_else(|| "no map loaded / id unknown".into()),
            script::MapRef::Named(n) => {
                let i = self.resolve_map_index(n).ok_or_else(|| format!("no map matching '{n}'"))?;
                self.map_candidates[i].map_id.ok_or_else(|| format!("map '{n}' has no readable id"))
            }
            script::MapRef::Variant(p) => mvar::parse_variant(std::path::Path::new(p))
                .map(|v| v.map_id)
                .ok_or_else(|| format!("cannot parse variant '{p}'")),
        }
    }

    /// Build a variant listing (path + base map id) for a selector. Returns (paths, printable).
    fn variant_listing(&self, sel: &script::VariantSel, filter: Option<&str>) -> (Vec<std::path::PathBuf>, String) {
        let want_id: Option<u32> = match sel {
            script::VariantSel::All => None,
            script::VariantSel::Id(id) => Some(*id),
            script::VariantSel::Current => self.loaded_map_id,
            script::VariantSel::MapName(n) => self
                .resolve_map_index(n)
                .and_then(|i| self.map_candidates[i].map_id),
        };
        let f = filter.map(|s| s.to_lowercase());
        let mut items = Vec::new();
        let mut s = String::new();
        for p in variant_catalog() {
            let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
            if let Some(f) = &f {
                if !stem.contains(f.as_str()) {
                    continue;
                }
            }
            let vid = mvar::parse_variant(&p).map(|v| v.map_id);
            if let Some(want) = want_id {
                if vid != Some(want) {
                    continue;
                }
            }
            let ids = vid.map(|v| format!("0x{v:04X}")).unwrap_or_else(|| "?".into());
            s.push_str(&format!("  id={ids}  {}\n", p.display()));
            items.push(p);
        }
        (items, s)
    }

    /// Dump every parsed field of one object (properties-panel data as text).
    fn dump_object(&self, datum: u32) -> String {
        // Scenario (map-owned, 0xE…) objects dump too — read-only, but the user
        // can at least learn WHAT they are.
        let Some(o) = self
            .mvar_objects
            .iter()
            .chain(self.local_objects.iter())
            .chain(self.scenario_objects.iter())
            .find(|o| o.datum == datum)
        else {
            return format!("no object 0x{datum:08X}");
        };
        let mut s = format!(
            "0x{datum:08X}\n  pos=({:.2},{:.2},{:.2})\n  fwd=({:.2},{:.2},{:.2}) up=({:.2},{:.2},{:.2})\n  primary_tag=0x{:08X} mode_tag=0x{:08X}\n",
            o.pos[0], o.pos[1], o.pos[2], o.fwd[0], o.fwd[1], o.fwd[2], o.up[0], o.up[1], o.up[2], o.primary_tag, o.mode_tag
        );
        if let Some(id) = self.object_identity(datum) {
            s.push_str(&id.dump_lines());
        }
        // #snap-array: the placed WORLD bounds, so a script can MEASURE a snap / an array's
        // spacing instead of eyeballing it (two pieces are flush when one's max equals the
        // other's min on the shared axis).
        if let Some((mn, mx)) = self.objscene().and_then(|s| s.aabb_of(datum)) {
            s.push_str(&format!(
                "  aabb=({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3}) size=({:.3},{:.3},{:.3})\n",
                mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
            ));
        }
        if let Some(m) = self.mvar_meta.get(&datum) {
            let team_i = if m.team == 0xFF { -1 } else { m.team as i32 };
            let shapes = ["none", "sphere", "cylinder", "box"];
            s.push_str(&format!(
                "  name={}\n  team={} color={} cached_type={} spawn_seq={} respawn={}\n  label='{}' (#{}) placement=0x{:02X} [{}]\n  boundary={} {:?} scale={:.3}\n  weapon_clips={} tele_channel={} tele_passability={} location_name={}\n",
                m.name,
                forge_team_name(team_i),
                if m.color < 0 { "inherit".to_string() } else { forge_team_name(m.color) },
                m.cached_type, m.spawn_seq, m.respawn,
                m.label, m.label_idx as i32, m.placement, placement_summary(m.placement),
                shapes[(m.boundary_shape as usize).min(3)], m.boundary, m.scale(self.sc_convention),
                m.weapon_clips, m.tele_channel, m.tele_passability, m.location_name
            ));
            s.push_str(&format!("  {}\n", obj_flags_line(datum, m, &self.obj_globals, self.sc_convention)));
            s.push_str(&self.h4_dump_lines(m)); // (empty for a Reach object)
        }
        s
    }

    /// Unified placement for the editor tools: when a static-palette object is
    /// selected, place it LOCALLY so it renders with or without a live game;
    /// otherwise (injection build) fall back to the live forge-MMF palette path.
    fn place_dispatch(&mut self, p: glam::Vec3) {
        if self.selected_static_pal.is_some() {
            self.place_local_at(p);
            return;
        }
        #[cfg(feature = "injection")]
        if self.selected_palette.is_some() {
            self.spawn_at(p);
            return;
        }
        // Offline there is no live palette to fall back to -- say what to do.
        self.spawn_status = "Pick an item in the Palette list first (then click in the viewport).".into();
    }

    /// Place the selected static-palette object LOCALLY (standalone editor): resolve its
    /// render_model via the game's tag graph and push it into `mvar_objects` + `mvar_meta`
    /// so it renders with or without a live game and is fully editable + saveable.
    fn place_local_at(&mut self, p: glam::Vec3) {
        let Some(i) = self.selected_static_pal else {
            self.spawn_status = "Select a forge-palette object first.".into();
            return;
        };
        // Halo 4 rows go through `instantiate` so the placed object saves (h4_app.rs)
        if self.h4_active {
            self.spawn_status = match self.h4_place_palette_item(i, p) {
                Ok((_, name)) => format!("Placed '{name}' at ({:.1},{:.1},{:.1})", p.x, p.y, p.z),
                Err(e) => format!("Place failed: {e}"),
            };
            return;
        }
        let Some((obj_tag, name)) = self.static_palette.get(i).cloned() else { return };
        let mode_tag = self
            .objscene()
            .map(|s| s.resolve_object_mode(obj_tag))
            .unwrap_or(0);
        if mode_tag == 0 || mode_tag == 0xFFFF_FFFF {
            self.spawn_status = format!("'{name}' has no render_model (tag {obj_tag:08x}).");
            return;
        }
        // Variant of the picked palette row (rocket/gauss/etc.) so the PLACED object matches
        // what the preview shows.
        let variant_sid = self.static_palette_variant.get(i).copied().unwrap_or(0);
        // Place drag-dropped objects on the SAVEABLE + EDITABLE path (`mvar_objects` + a full
        // `ObjMeta`) — the same path the script `place` cmd uses — so the object panel shows and
        // edits the FULL forge record (pos/rotation/scale/label/placement/cached-type/boundary).
        // No `invalidate()` here (it would freeze the drag): the per-tick `mvar_objects` push
        // feeds the render. placement = DEFAULT_SPAWNABLE so the saved object actually spawns in
        // game (the save-add path copies m.placement over new_placed_object's default; 0 would
        // give an object the game reads but never spawns).
        let (folder, item) = self
            .scene_ctl
            .as_ref()
            .and_then(|s| s.folder_item_for_obj(obj_tag))
            .unwrap_or((0xFFFF, 0xFF));
        let datum = self.next_dup_datum;
        self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
        self.mvar_objects.push(ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: [p.x, p.y, p.z],
            health: 1.0,
            shield: 1.0,
            mode_tag,
            fwd: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: obj_tag,
            variant_name_sid: variant_sid,
        });
        self.mvar_meta.insert(datum, ObjMeta {
            name: name.clone(),
            folder,
            item,
            pos: [p.x, p.y, p.z],
            team: mvar::TEAM_NEUTRAL, // new placements start NEUTRAL, like the game's Forge
            color: -1,
            cached_type: 0,
            spawn_seq: 0,
            respawn: 0,
            label_idx: 0xFFFF,
            label: String::new(),
            placement: mvar::PLACEMENT_DEFAULT_SPAWNABLE,
            boundary_shape: 0,
            boundary: [0; 4],
            weapon_clips: 0,
            tele_channel: 0,
            tele_passability: 0,
            location_name: 0xFFFF,
            spawn_rel: -1,
            slot: 0xFFFF,
            flags: Default::default(),
            h4: None,
        });
        // Also queue it on the live forge MMF if a game is attached (injection build only).
        if let Some(ws) = &self.world_spawn {
            let _ = ws.queue_spawn_at(0, 0, 0, p.x, p.y, p.z);
        }
        self.spawn_status = format!("Placed '{name}' at ({:.1},{:.1},{:.1})", p.x, p.y, p.z);
    }

    /// Index of the candidate map whose base-map id matches `map_id` (the id a .mvar
    /// targets). None if no installed map has that id (or ids aren't known yet).
    fn find_map_by_id(&self, map_id: u32) -> Option<usize> {
        self.map_candidates.iter().position(|c| c.map_id == Some(map_id))
    }

    /// Load a .mvar by path. If the variant targets a DIFFERENT base map than the one
    /// loaded, switch to (and load) that map first, then render the objects once it
    /// finishes (deferred via `pending_mvar`). If it's the SAME map, just re-render the
    /// forge objects in place — no geometry reload. Renders entirely offline.
    fn import_mvar(&mut self, path: std::path::PathBuf) {
        // A Halo 4 variant (mvar chunk v50) never reaches the Reach reader, which would
        // mis-decode it (its content header is one bit shorter) and try to load a random map.
        if h4::mvar::is_h4_variant(&path) {
            self.h4_import_mvar(path);
            return;
        }
        let Some(variant) = mvar::parse_variant(&path) else {
            self.spawn_status = "Failed to parse .mvar.".into();
            return;
        };
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        // If the variant's base map isn't the one loaded, load the correct map FIRST and
        // defer the object render until it's ready. Only when we can actually identify the
        // target map among the installed candidates — otherwise fall through to best-effort.
        if self.loaded_map_id != Some(variant.map_id) {
            if let Some(idx) = self.find_map_by_id(variant.map_id) {
                // If the variant's base map is ALREADY the loaded one (matched by PATH,
                // not the cached loaded_map_id — which can be None after a MANUAL map load whose
                // .mapinfo read_map_id failed), do NOT reload the geometry; just set
                // loaded_map_id so it is known going forward.
                let target_path = self.map_candidates[idx].path.to_string_lossy().into_owned();
                let already_loaded = !self.map_path.is_empty()
                    && std::path::Path::new(&self.map_path) == std::path::Path::new(&target_path);
                if already_loaded {
                    self.loaded_map_id = Some(variant.map_id);
                    let placed = self.render_variant_objects(&variant, &name);
                    self.after_variant_render(placed, path);
                    return;
                }
                let map_label = self.map_candidates[idx].label();
                self.show_modded_maps = self.map_candidates[idx].modded;
                self.picker_game = self.map_candidates[idx].game; // #map-picker
                self.selected_map = Some(idx);
                self.pending_mvar = Some(path);
                self.spawn_status =
                    format!("{name}: loading base map {map_label} (map id {}) before rendering…", variant.map_id);
                self.load_map();
                return;
            }
            // Target map id unknown/uninstalled → render on whatever is loaded (best effort).
        }
        let placed = self.render_variant_objects(&variant, &name);
        self.after_variant_render(placed, path);
    }

    /// Post-render bookkeeping for a variant loaded by PATH: on success, persist it as the
    /// last-opened variant and clear any retry; on 0 placed (scene/palette not ready yet),
    /// queue a few-tick retry so the user doesn't have to open the same variant twice.
    fn after_variant_render(&mut self, placed: usize, path: std::path::PathBuf) {
        if placed > 0 {
            mapcat::save_last_variant(&path.to_string_lossy());
            self.current_variant_path = Some(path.clone()); // remember for File->Save
            self.variant_retry = None;
        } else {
            self.variant_retry = Some((path, 8));
        }
    }

    /// File ▸ New — pick a template .mvar for the loaded map (same map id, fewest objects,
    /// searched in the MCC map_variants / hopper_map_variants folders and the last opened variant's
    /// folder), clear the placed objects and reset the header so Save / Save As write a fresh file.
    fn new_variant(&mut self) {
        if self.h4_active { self.h4_new_variant(); return; }
        let Some(map_id) = self.loaded_map_id else {
            self.spawn_status = "New variant: load a base map first.".into();
            return;
        };
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        let mp = std::path::Path::new(&self.map_path);
        if let Some(hr) = mp.parent().and_then(|p| p.parent()) {
            dirs.push(hr.join("map_variants"));
            dirs.push(hr.join("hopper_map_variants"));
        }
        if let Some(d) = self.current_variant_path.as_ref().and_then(|p| p.parent()) { dirs.push(d.to_path_buf()); }
        if let Some(d) = self.new_variant_template.as_ref().and_then(|p| p.parent()) { dirs.push(d.to_path_buf()); }
        let mut best: Option<(usize, std::path::PathBuf, mvar::Variant)> = None;
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(true, |x| !x.eq_ignore_ascii_case("mvar")) { continue; }
                let Some(v) = mvar::parse_variant(&p) else { continue };
                if v.map_id != map_id { continue; }
                let n = v.objects.len();
                if best.as_ref().map_or(true, |(bn, _, _)| n < *bn) { best = Some((n, p, v)); }
            }
        }
        let Some((_, tpl, v)) = best else {
            self.spawn_status = format!("New variant: no .mvar for map id {map_id} found to use as a template (open any variant of this map once, or place one in map_variants).");
            return;
        };
        self.mvar_objects.clear();
        self.local_objects.clear();
        self.selected_set.clear();
        self.selected_datum = None;
        self.mvar_colors.clear();
        self.mvar_meta.clear();
        self.mvar_labels = v.labels.clone();
        self.seed_variant_globals(v.globals.clone()); // the template's globals
        self.variant_title = "New variant".into();
        self.variant_description = String::new();
        if self.variant_author.trim().is_empty() { self.variant_author = "HMS".into(); }
        self.variant_editor = self.variant_author.clone();
        self.variant_header_dirty = true;
        self.current_variant_path = None;
        self.new_variant_template = Some(tpl.clone());
        self.rebuild_overlays();
        self.spawn_status = format!("New variant (template {}): name it in the Object panel, place objects, then Save.", tpl.file_name().unwrap_or_default().to_string_lossy());
    }

    /// The "Variant properties" section of the Object panel (both games).
    fn variant_properties_ui(&mut self, ui: &mut egui::Ui) {
        let Some(g) = self.variant_globals.clone() else { return };
        let dirty = self.global_edits.is_dirty(&g);
        let title = if dirty { "Variant properties *" } else { "Variant properties" };
        egui::CollapsingHeader::new(title).default_open(false).show(ui, |ui| {
            let mut edited = false;
            egui::Grid::new("variant-globals").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                ui.label("Game / map id");
                ui.label(format!("{}  {}", g.game, g.map_id)).on_hover_text(format!("map-id (content header {}); map-variant-version {}; type {} (map variant)", g.header_map_id, g.version, g.content_type));
                ui.end_row();
                ui.label("Created");
                ui.label(format!("{}  xuid {:016x}", mvar::unix_date(g.created.0), g.created.1)).on_hover_text(format!("CreatedBy: timestamp + author-xuid (stamped by Save; read-only); online flag {}", g.created_online));
                ui.end_row();
                ui.label("Modified");
                ui.label(format!("{}  xuid {:016x}", mvar::unix_date(g.modified.0), g.modified.1)).on_hover_text(format!("ModifiedBy: timestamp + author-xuid (stamped by Save; read-only); online flag {}", g.modified_online));
                ui.end_row();
                ui.label("Activity / mode / engine");
                ui.label(format!("{} {} / {} {} / {} {}", g.activity, mvar::activity_name(g.game, g.activity), g.game_mode, mvar::game_mode_name(g.game, g.game_mode), g.engine, mvar::engine_name(g.game, g.engine)))
                    .on_hover_text("activity / game-mode / game-engine-type: read-only (they change the header layout: activity 2 adds a hopper id, campaign / firefight add their own blocks)");
                ui.end_row();
                ui.label("Category");
                let mut c = self.global_edits.category as i32;
                if ui.add(egui::DragValue::new(&mut c).range(-128..=127)).on_hover_text("megalo-category-index (signed byte; -1 = none on every shipped file). Mirrored into the chdr chunk.").changed() {
                    self.global_edits.category = c as i8;
                    edited = true;
                }
                ui.end_row();
                ui.label("Budget max");
                let mut b = self.global_edits.budget_max;
                if ui.add(egui::DragValue::new(&mut b).range(0..=1_000_000).speed(100.0)).on_hover_text(format!("maximum_budget (spent_budget {} is recomputed on save). NOTE: when the game loads the variant on its own map it restores the map's sandbox budget, so this only affects the file / other tools.", g.budget_spent)).changed() {
                    self.global_edits.budget_max = b;
                    edited = true;
                }
                ui.end_row();
                ui.label("Ids");
                ui.label(format!("{:016x}", g.uid)).on_hover_text(format!("uid {:016x}\nparent-uid {:016x}\nroot-uid {:016x}\ngame-id {:016x}\nThe game keeps the template's ids on every Forge save, so HMS copies them verbatim (Save As too).", g.uid, g.parent_uid, g.root_uid, g.game_id));
                ui.end_row();
                ui.label("Flags");
                ui.label(format!("built_in {}  built_from_xml {}", g.built_in as u8, g.built_from_xml as u8)).on_hover_text(format!("map-variant-checksum {:08x}; m_scenario_palette_crc {:08x} (the loader rebuilds the variant from the scenario when this differs from the map's palette CRC); mcc-map-id {}", g.checksum, g.palette_crc, g.mcc_map_id.map_or("-".to_string(), |x| x.iter().map(|b| format!("{b:02x}")).collect())));
                ui.end_row();
                ui.label("Chunk");
                ui.label(format!("length {}  sha1 {}  _fsm {}", g.chunk_length, if g.hash_ok { "ok" } else { "stale" }, if g.has_fsm { "yes" } else { "no" })).on_hover_text("mvar chunk: bytes written + SHA-1 (recomputed on every save; the engine never enforces it); _fsm = MCC file-share signature block (copied verbatim, never regenerated)");
                ui.end_row();
            });
            // world bounds, behind a warning
            let ob = self.objects_outside(self.global_edits.bounds);
            egui::CollapsingHeader::new("World bounds").default_open(false).show(ui, |ui| {
                ui.small("The box object positions are quantised against. The game REPLACES it with the map's own world bounds when it loads the variant on its map (and clamps positions into that box on its own save), so widening it only lets this file hold objects outside the map's box - such objects are outside the game's supported space.");
                ui.checkbox(&mut self.bounds_edit_unlocked, "I understand - unlock editing");
                let mut b = self.global_edits.bounds;
                let names = ["x min", "x max", "y min", "y max", "z min", "z max"];
                let mut changed = false;
                egui::Grid::new("variant-bounds").num_columns(4).spacing([8.0, 3.0]).show(ui, |ui| {
                    for i in 0..3 {
                        for k in 0..2 {
                            ui.label(names[2 * i + k]);
                            changed |= ui.add_enabled(self.bounds_edit_unlocked, egui::DragValue::new(&mut b[2 * i + k]).speed(1.0).max_decimals(3)).changed();
                        }
                        ui.end_row();
                    }
                });
                if changed && (0..3).all(|i| b[2 * i + 1] > b[2 * i]) {
                    self.global_edits.bounds = b;
                    edited = true;
                }
                if self.global_edits.bounds != g.bounds {
                    ui.small(format!("file: x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}", g.bounds[0], g.bounds[1], g.bounds[2], g.bounds[3], g.bounds[4], g.bounds[5]));
                    if ui.small_button("Reset to the file's box").clicked() { self.global_edits.bounds = g.bounds; edited = true; }
                }
                if ob > 0 {
                    ui.colored_label(egui::Color32::from_rgb(230, 150, 80), format!("{ob} object(s) outside the box - they are clamped to its edge on save"));
                }
            });
            // quota table
            egui::CollapsingHeader::new(format!("Object quotas ({})", g.quotas.len())).default_open(false).show(ui, |ui| {
                ui.small("Per palette entry: minimum_count / maximum_count (both what the Forge object menu edits) and placed_on_map (recomputed on save; a maximum below it is raised to it).");
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    egui::Grid::new("variant-quotas").num_columns(5).spacing([6.0, 2.0]).striped(true).show(ui, |ui| {
                        ui.small("#"); ui.small("entry"); ui.small("min"); ui.small("max"); ui.small("placed");
                        ui.end_row();
                        for i in 0..g.quotas.len() {
                            if self.global_edits.quota_minmax.len() <= i { break; }
                            let name = self.variant_quota_names.get(i).map(|s| s.as_str()).unwrap_or("");
                            ui.small(format!("{i}"));
                            ui.small(name);
                            let (mut mn, mut mx) = self.global_edits.quota_minmax[i];
                            let a = ui.add(egui::DragValue::new(&mut mn).range(0..=255)).changed();
                            let bch = ui.add(egui::DragValue::new(&mut mx).range(0..=255)).changed();
                            ui.small(format!("{}", g.quotas[i].2));
                            if a || bch { self.global_edits.quota_minmax[i] = (mn, mx); edited = true; }
                            ui.end_row();
                        }
                    });
                });
            });
            if edited { self.variant_header_dirty = true; }
            if dirty { ui.small("Edited - File > Save will write these."); }
        });
    }

    /// Seed the Variant-properties panel + the editable global set from a freshly
    /// parsed variant (either game). Quota rows get the base map's palette entry names.
    fn seed_variant_globals(&mut self, g: mvar::VariantGlobals) {
        self.global_edits = mvar::GlobalEdits::from_globals(&g);
        self.variant_quota_names = self.quota_entry_names(g.game == "Halo 4", g.quotas.len());
        self.variant_globals = Some(g);
        self.bounds_edit_unlocked = false;
    }

    /// Palette entry name per quota index (entry i = quota index i): Reach from the
    /// sandbox palette's type order (`Scene::forge_type_order`), Halo 4 from the flattened
    /// forge palette. Empty strings where the map has no entry for that index.
    fn quota_entry_names(&self, h4: bool, n: usize) -> Vec<String> {
        let mut out = vec![String::new(); n];
        if h4 {
            if let Some(pal) = self.h4_edit.pal.as_ref() {
                for (i, e) in pal.entries.iter().enumerate().take(n) {
                    let cat = pal.categories.get(e.category).map(|c| c.display.as_str()).unwrap_or("");
                    out[i] = if cat.is_empty() { e.display.clone() } else { format!("{cat} / {}", e.display) };
                }
            }
        } else if let Some(scene) = self.scene_ctl.as_ref() {
            let pal = scene.forge_palette_full();
            let types = scene.forge_type_order(&pal);
            for (i, &(pi, ew)) in types.iter().enumerate().take(n) {
                if let Some(e) = pal.iter().find(|e| e.palette_index == pi && e.entry_within == ew) {
                    out[i] = if e.category_name.is_empty() { e.name.clone() } else { format!("{} / {}", e.category_name, e.name) };
                }
            }
        }
        out
    }

    /// The `variant get` report: every global field with its engine name, one per
    /// line (both games), plus the quota table.
    fn variant_globals_report(&self, field: Option<&str>) -> Result<String, String> {
        let g = self.variant_globals.as_ref().ok_or("no variant is open")?;
        let e = &self.global_edits;
        let names = &self.variant_quota_names;
        mvar::globals_report(g, e, names, field, &self.variant_title, &self.variant_description, &self.variant_author, &self.variant_editor, &self.mvar_labels)
    }

    /// `variant set <field> <value>` on the open variant (either game); the change
    /// is written by the next Save.
    fn variant_set(&mut self, field: &str, value: &str) -> Result<String, String> {
        let g = self.variant_globals.clone().ok_or("no variant is open")?;
        let (r, header) = mvar::apply_global_set(&g, &mut self.global_edits, field, value, &mut self.variant_title, &mut self.variant_description, &mut self.variant_author, &mut self.variant_editor)?;
        if header { self.variant_header_dirty = true; }
        // a bounds edit must hold every object, like the panel's check
        if field.eq_ignore_ascii_case("bounds") {
            let n = self.objects_outside(self.global_edits.bounds);
            if n > 0 { return Ok(format!("{r} WARNING: {n} object(s) lie outside the new box and would be clamped to its edge on save")); }
        }
        Ok(r)
    }

    /// How many placed objects lie outside `b` (xmin xmax ymin ymax zmin zmax).
    fn objects_outside(&self, b: [f32; 6]) -> usize {
        self.mvar_objects.iter().chain(self.local_objects.iter()).filter(|o| {
            let p = o.pos;
            p[0] < b[0] || p[0] > b[1] || p[1] < b[2] || p[1] > b[3] || p[2] < b[4] || p[2] > b[5]
        }).count()
    }

}

/// Build the EXACT object list a save writes, as a pure function.
///
/// This is literally "loop the placed forge objects and write them": walk the editor's objects IN
/// ORDER, and for each one emit its record. Order in == order out, so slots can never drift.
///
/// An object's SOURCE record is found by its DATUM (`src`), never by arithmetic on a slot
/// index: "datum 0xD0000000+i is source slot i" is only true until the file changes; after one
/// save the slots have shifted, and a second save would match every object against a DIFFERENT
/// record and write it out as the wrong thing (a wall saved as an initial spawn) or drop it
/// entirely (`save_stress::second_save_without_reloading_loses_nothing`).
///
/// Objects HMS cannot display are not in the scene at all, so their records are passed in
/// separately and carried through verbatim. Returns the list plus the count of objects that had no
/// resolvable palette entry.
fn build_save_list(
    src: &std::collections::HashMap<u32, mvar::PlacedObject>,
    live: &[hms_ipc::ObjectInfo],
    extra: &[hms_ipc::ObjectInfo],
    unresolved_objs: &[mvar::PlacedObject],
    meta: &std::collections::HashMap<u32, ObjMeta>,
    colors: &std::collections::HashMap<u32, (u8, u8)>,
    resolve_fi: impl Fn(u32) -> Option<(u16, u8)>,
) -> (Vec<mvar::PlacedObject>, usize) {
    let mut list: Vec<mvar::PlacedObject> = Vec::with_capacity(live.len() + extra.len() + unresolved_objs.len());
    let mut unsaveable = 0usize;
    // where each object ends up, so parent links (stored as SLOT INDICES) can be fixed up
    let mut new_index_of: std::collections::HashMap<u32, usize> = Default::default();

    for o in live.iter().chain(extra.iter()) {
        // Start from this object's OWN source record when it has one, so every field HMS does not
        // model (out-of-bounds escapes, boundary values, unknown bits) is preserved exactly.
        let mut po = match src.get(&o.datum) {
            Some(t) => t.clone(),
            None => {
                // A newly placed / duplicated object: it needs a palette entry to exist at all.
                let fi = match meta.get(&o.datum) {
                    Some(m) if m.folder != 0xFFFF && m.item != 0xFF => Some((m.folder, m.item)),
                    _ => resolve_fi(o.primary_tag),
                };
                let Some((folder, item)) = fi else {
                    unsaveable += 1;
                    continue;
                };
                // A placement the user never touched has no `colors` row; its meta
                // carries the NEUTRAL default (the game's own new-object team), so save that rather
                // than the old 0xFF "none".
                let (team, color) = colors.get(&o.datum).copied().unwrap_or_else(|| {
                    meta.get(&o.datum)
                        .map(|m| (m.team, if m.color < 0 { 0xFF } else { m.color as u8 }))
                        .unwrap_or((mvar::TEAM_NEUTRAL, 0xFF))
                });
                mvar::new_placed_object(
                    folder, item, o.pos, o.fwd, o.up, team,
                    if color == 0xFF { -1 } else { color as i32 },
                )
            }
        };

        po.pos = o.pos;
        // Only re-quantise the orientation when the object was actually turned, so an
        // untouched object keeps its original rotation bits verbatim.
        let turned = {
            let d = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs());
            d(o.fwd, po.fwd) > 1e-6 || d(o.up, po.up) > 1e-6
        };
        if turned {
            let (g, uq, fq) = mvar::encode_orientation(o.fwd, o.up);
            po.up_is_global = g;
            po.up_quant = uq;
            po.forward_angle_q = fq;
            po.fwd = o.fwd;
            po.up = o.up;
        }
        if let Some((team, color)) = colors.get(&o.datum).copied() {
            po.team = team;
            po.color = if color == 0xFF { -1 } else { color as i32 };
        }
        if let Some(m) = meta.get(&o.datum) {
            // the properties panel owns these
            if m.folder != 0xFFFF && m.item != 0xFF {
                po.folder = m.folder;
                po.item = m.item;
            }
            po.placement = m.placement;
            po.spawn_seq = m.spawn_seq;
            po.respawn = m.respawn;
            po.label_idx = m.label_idx;
            po.spawn_rel = m.spawn_rel;
            po.cached_type = m.cached_type;
        }
        new_index_of.insert(o.datum, list.len());
        list.push(po);
    }

    // Objects HMS cannot display: carried through untouched.
    for u in unresolved_objs {
        list.push(u.clone());
    }

    // A parent is stored as a SLOT INDEX, so it must be remapped to where that
    // parent actually ended up. A child whose parent is gone is ORPHANED rather than left pointing
    // at whatever now occupies that index.
    let old_index_to_new: std::collections::HashMap<i32, i32> = new_index_of
        .iter()
        .filter_map(|(d, ni)| {
            (*d >= 0xD000_0000 && *d < 0xD100_0000).then(|| ((*d - 0xD000_0000) as i32, *ni as i32))
        })
        .collect();
    // How many slots the SOURCE had, so "the parent was deleted" can be told apart from "this
    // value was never a slot index". A couple of shipped variants store parents like 754, past the
    // 651 slots; a save must not invent meaning for those, so they are left exactly as found.
    let src_len = src
        .keys()
        .filter(|d| **d >= 0xD000_0000 && **d < 0xD100_0000)
        .map(|d| (d - 0xD000_0000) as i32 + 1)
        .max()
        .unwrap_or(0);
    for po in list.iter_mut() {
        let sr = po.spawn_rel;
        if sr < 0 || sr >= src_len {
            continue; // no parent, or not a slot index we can reason about
        }
        po.spawn_rel = old_index_to_new.get(&sr).copied().unwrap_or(-1); // parent deleted -> orphan
    }
    (list, unsaveable)
}

impl App {
    /// Write EVERY forge object currently in the editor to the file, in order.
    /// The list itself is built by [`build_save_list`] (a pure function, so it can be stress-tested
    /// without a GUI); this method only supplies the editor's state and the header edits.
    fn save_variant_to(&self, dst: &std::path::Path) -> Result<(usize, usize), String> {
        // A Halo 4 map saves through h4_app.rs (build_h4_save_list -> gates ->
        // h4::mvar::save_objects); the Reach codec below must never see one of those files.
        if self.h4_active {
            return self.h4_save_variant_to(dst);
        }
        let src = self.current_variant_path.as_ref().or(self.new_variant_template.as_ref()).ok_or("no variant is open")?;
        let variant = mvar::parse_variant(src).ok_or("cannot re-read the source variant")?;
        let (mut list, unsaveable) = build_save_list(
            &self.mvar_src,
            &self.mvar_objects,
            &self.local_objects,
            &self.mvar_unresolved_objs,
            &self.mvar_meta,
            &self.mvar_colors,
            |tag| self.scene_ctl.as_ref().and_then(|s| s.folder_item_for_obj(tag)),
        );

        // The variant holds 651 slots. Anything past that cannot be written.
        let no_slot = list.len().saturating_sub(651);
        list.truncate(651);

        // Header strings + editable globals: re-encode ONLY the fields the user actually changed,
        // diffed against the SOURCE file's values, so an untouched field keeps its exact original
        // bits (a raw gamertag is never mangled by a round-trip; a bounds edit re-quantises every
        // object).
        let (category, budget_max, bounds, quotas) = self.global_edits.diff(&variant.globals);
        let mut header = mvar::HeaderEdits {
            title: (self.variant_title != variant.title).then(|| self.variant_title.clone()),
            description: (self.variant_description != variant.description).then(|| self.variant_description.clone()),
            author: (self.variant_author != variant.author).then(|| self.variant_author.clone()),
            editor: (self.variant_editor != variant.editor).then(|| self.variant_editor.clone()),
            labels: (self.mvar_labels != variant.labels).then(|| self.mvar_labels.clone()),
            category,
            budget_max,
            bounds,
            quotas,
            ..Default::default()
        };
        // Stamp CreatedBy / ModifiedBy the way the game does: a NEW variant
        // (no file of its own yet) gets created = now, modified = cleared; a re-save into a
        // different file, or with header edits, gets modified = now. An unedited re-save of an
        // opened file over itself stays byte-identical.
        if self.current_variant_path.is_none() {
            header.stamp_new(0);
        } else if !header.is_empty() || Some(dst) != self.current_variant_path.as_deref() {
            header.stamp_modified(0);
        }
        let header = if header.is_empty() { None } else { Some(header) };
        let written = mvar::save_objects(src, dst, &list, header.as_ref()).map_err(|e| e.to_string())?;
        self.last_save_no_palette.set(unsaveable);
        self.last_save_no_slot.set(no_slot);
        Ok((written, unsaveable + no_slot))
    }

    /// A precise note about anything the last save could not write. The two causes
    /// are completely different problems and must never be reported as one number: a variant that
    /// is FULL (Reach gives every map variant 651 object slots, and objects HMS cannot display
    /// still occupy theirs) versus an object with no resolvable palette entry.
    fn save_failure_note(&self) -> String {
        if self.h4_active { return self.h4_edit.last_save_note.borrow().clone(); }
        let (np, ns) = (self.last_save_no_palette.get(), self.last_save_no_slot.get());
        let mut parts: Vec<String> = Vec::new();
        if ns > 0 {
            let hidden = self.mvar_unresolved.len();
            let hid = if hidden > 0 {
                format!(
                    ", {hidden} of which HMS cannot display but still take slots"
                )
            } else {
                String::new()
            };
            parts.push(format!(
                "{ns} had no free slot — the variant holds 651 objects{hid}"
            ));
        }
        if np > 0 {
            parts.push(format!("{np} had no palette entry"));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!(" ({})", parts.join("; "))
        }
    }

    /// CAD-style construction geometry — dotted guides between object anchors, and exact
    /// snapping to where they cross. Surfaced as a toolbar tool and under Tools.
    fn cad_tool_ui(&mut self, ui: &mut egui::Ui) {
            ui.horizontal(|ui| {
                // The Construct TOOL owns "am I drawing?" — this panel only reflects it (and can
                // hand the tool back), so toolbar and panel can never disagree.
                ui.label(egui::RichText::new("✔ tool active").color(egui::Color32::from_rgb(120, 200, 130)));
                if ui.small_button("done").on_hover_text("Back to Select (C)").clicked() {
                    self.tool_mode = ToolMode::Select;
                }
                if ui.checkbox(&mut self.show_guides, "Show").changed() {
                    self.overlays_dirty = true;
                }
            });
            // The TOOL picker goes FIRST and the hint for the chosen tool sits right
            // under it, so the panel always answers "what does a click do right now?".
            ui.horizontal_wrapped(|ui| {
                for op in ConstructOp::ALL {
                    if ui.selectable_label(self.construct_op == op, op.label()).on_hover_text(op.hint()).clicked() {
                        self.construct_op = op;
                        self.construct_pending = None;
                        self.cad_face_a = None;
                        self.line_pending = None;
                        self.status = format!("{}: {}", op.label(), op.hint());
                    }
                }
            });
            ui.small(egui::RichText::new(self.construct_op.hint()).color(egui::Color32::from_rgb(150, 200, 255)));
            ui.horizontal(|ui| {
                ui.label("Snap to:");
                egui::ComboBox::from_id_salt("cad-anchor-mode")
                    .selected_text(self.anchor_mode.label())
                    .show_ui(ui, |ui| {
                        for m in construct::AnchorMode::ALL {
                            ui.selectable_value(&mut self.anchor_mode, m, m.label());
                        }
                    });
            });
            ui.horizontal(|ui| {
                if ui.button("Box diagonals")
                    .on_hover_text("Four corner-to-corner diagonals of the selection's box. All four cross at its exact centre.")
                    .clicked()
                {
                    self.add_box_diagonals(false);
                }
                if ui.button("Face diagonals")
                    .on_hover_text("The two diagonals of each of the six faces. Each pair crosses at that face's exact centre.")
                    .clicked()
                {
                    self.add_box_diagonals(true);
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.snap_guides, "Snap moves");
                ui.add(egui::DragValue::new(&mut self.snap_range).range(0.1..=20.0).speed(0.1).prefix("range "))
                    .on_hover_text("Snap tolerance, and how close a click must be to a shape's centre to grab it.");
                ui.add(egui::DragValue::new(&mut self.magnet_range).range(0.1..=40.0).speed(0.1).prefix("magnet "))
                    .on_hover_text("Ctrl magnet reach: the largest face-to-face gap that still snaps, in world units.");
            });
            ui.separator();
            match self.construct_op {
                ConstructOp::Coincident => {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.cad_mate_rotate, "turn to face")
                            .on_hover_text("Rotate the part so its face looks straight INTO the target face. Leave on — without it a mate only slides, which fails on anything not already square to the target.");
                        ui.checkbox(&mut self.cad_mate_centered, "centre on face")
                            .on_hover_text("Off: make the faces FLUSH (coplanar), keeping position within the plane.\nOn: also put face A's centre on face B's.");
                        match &self.cad_face_a {
                            Some((_, _, l)) => ui.small(format!("A = {l} — click face B")),
                            None => ui.small("click face A"),
                        };
                    });
                }
                // #snap-array: the LINE array tool. Two viewport clicks (or one plus a length)
                // define the line; the current selection fills it.
                ConstructOp::Line => {
                    let n_obj = self.movable_datums().len();
                    ui.small(format!("1. select the object(s) to copy — {n_obj} selected"));
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.line_tool.use_step, false, "by count")
                            .on_hover_text("Spread N copies evenly: the first sits on the start point, the last on the end point.");
                        ui.selectable_value(&mut self.line_tool.use_step, true, "by step")
                            .on_hover_text("One copy every STEP world units. A step smaller than the piece makes the copies OVERLAP.");
                        if self.line_tool.use_step {
                            ui.add(egui::DragValue::new(&mut self.line_tool.step).range(0.05..=500.0).speed(0.05).prefix("step ").suffix(" wu"));
                        } else {
                            ui.add(egui::DragValue::new(&mut self.line_tool.count).range(1..=512).prefix("count "));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.line_tool.align, "turn to follow the line")
                            .on_hover_text("Yaw each copy so it faces along the line. Yaw only — nothing tips over.");
                        ui.checkbox(&mut self.line_tool.one_click, "one click + length")
                            .on_hover_text("On: ONE click sets the start and the line runs the given length along the chosen axis.\nOff: click the start, then the end.");
                    });
                    if self.line_tool.one_click {
                        ui.horizontal(|ui| {
                            ui.label("along");
                            egui::ComboBox::from_id_salt("line-axis")
                                .selected_text(["X", "Y", "Z"][self.line_tool.axis.min(2)])
                                .show_ui(ui, |ui| {
                                    for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                                        ui.selectable_value(&mut self.line_tool.axis, i, *n);
                                    }
                                });
                            ui.checkbox(&mut self.line_tool.negative, "negative");
                            ui.add(egui::DragValue::new(&mut self.line_tool.length).range(0.05..=4000.0).speed(0.5).prefix("length ").suffix(" wu"));
                        });
                    }
                    // Live spacing readout, so "will these overlap?" is answered before the click.
                    let ext = self.selection_extent_along(self.line_tool.dir());
                    let (n, step) = match self.line_tool.spec() {
                        snap::LineSpec::Count(c) => (c, if self.line_tool.one_click && c > 1 { self.line_tool.length / (c - 1) as f32 } else { 0.0 }),
                        snap::LineSpec::Step(st) => (((self.line_tool.length / st).floor() as u32 + 1).min(512), st),
                    };
                    let spacing = if step > 0.0 && ext > 0.0 { snap::spacing_label(step, ext) } else { "—".to_string() };
                    ui.small(egui::RichText::new(match self.line_pending {
                        Some(p) => format!("start ({:.2}, {:.2}, {:.2}) — click the END point (Esc cancels)", p.x, p.y, p.z),
                        None => if self.line_tool.one_click {
                            format!("click a point: {n} copies, step {step:.2} wu ({spacing}), piece {ext:.2} wu")
                        } else {
                            format!("click the START of the line — piece is {ext:.2} wu along {}", ["X", "Y", "Z"][self.line_tool.axis.min(2)])
                        },
                    }).color(egui::Color32::from_rgb(150, 200, 255)));
                }
                ConstructOp::Circle | ConstructOp::Square => {
                    ui.horizontal(|ui| {
                        ui.label("Plane");
                        egui::ComboBox::from_id_salt("cad-shape-plane")
                            .selected_text(["X", "Y", "Z", "clicked face"][self.cad_shape_plane.min(3)])
                            .show_ui(ui, |ui| {
                                for (i, n) in ["X", "Y", "Z", "clicked face"].iter().enumerate() {
                                    ui.selectable_value(&mut self.cad_shape_plane, i, *n);
                                }
                            });
                        if self.construct_op == ConstructOp::Circle {
                            ui.add(egui::DragValue::new(&mut self.cad_shape_seg).range(3..=256).prefix("seg "))
                                .on_hover_text("Circle vertices: 4 = diamond, 8 = octagon, more = rounder.");
                        } else {
                            // Signed: the square turns either way, and a '-' typed anywhere is the
                            // sign ("45 -" == "-45"), same as every other angle field.
                            ui.add(egui::DragValue::new(&mut self.cad_shape_rot).range(-360.0..=360.0).speed(1.0).suffix("°").custom_parser(numfield::parse_signed))
                                .on_hover_text("Turn the square in its own plane (45° = diamond). Negative turns the other way; a '-' typed anywhere flips the sign.");
                        }
                    });
                    ui.small("Press and DRAG to draw. Press a shape's centre to move it.");
                }
                _ => {}
            }
            ui.separator();
            // The drawn shapes, editable after the fact.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("Shapes ({})", self.shapes.len())).small().strong());
                if !self.shapes.is_empty() && ui.small_button("clear").clicked() {
                    self.push_edit_undo();
                    self.shapes.clear();
                    self.sel_shape = None;
                    self.overlays_dirty = true;
                }
            });
            let mut del_shape: Option<usize> = None;
            egui::ScrollArea::vertical().max_height(90.0).id_salt("cad-shapes").show(ui, |ui| {
                for i in 0..self.shapes.len() {
                    let (kind, r) = { let sh = &self.shapes[i]; (sh.kind.label(), sh.radius) };
                    ui.horizontal(|ui| {
                        let sel = self.sel_shape == Some(i);
                        if ui.selectable_label(sel, format!("{i}: {kind} {r:.2} wu")).clicked() {
                            self.sel_shape = if sel { None } else { Some(i) };
                            self.overlays_dirty = true;
                        }
                        if ui.small_button("x").clicked() { del_shape = Some(i); }
                    });
                }
            });
            if let Some(i) = del_shape {
                self.push_edit_undo();
                self.shapes.remove(i);
                self.sel_shape = None;
                self.overlays_dirty = true;
            }
            if let Some(si) = self.sel_shape.filter(|i| *i < self.shapes.len()) {
                ui.horizontal(|ui| {
                    let mut r = self.shapes[si].radius;
                    if ui.add(egui::DragValue::new(&mut r).range(0.05..=1000.0).speed(0.1).prefix("size ")).changed() {
                        self.push_edit_undo();
                        self.shapes[si].radius = r;
                        self.overlays_dirty = true;
                    }
                    if ui.button("Centre on selection")
                        .on_hover_text("Move this shape's centre to the centre of the current object selection.")
                        .clicked()
                    {
                        if let Some(o) = self.selection_obb() {
                            self.push_edit_undo();
                            self.shapes[si].center = o.c;
                            self.overlays_dirty = true;
                        }
                    }
                });
                ui.small("Move it: click the shape, press G, click to place (Esc cancels). Ctrl+Z undoes.");
                // Spelled out as a recipe — the feature is useless if nobody can
                // work out that it needs BOTH a shape and an object selection.
                ui.separator();
                ui.label(egui::RichText::new("FILL EDGE WITH AN OBJECT").small().strong());
                let n_obj = self.movable_datums().len();
                ui.small(format!("1. select the object(s) to copy — {} selected", n_obj));
                ui.small("2. set how many, then press Fill edge");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.array_count).range(1..=256).prefix("count "));
                    ui.checkbox(&mut self.array_rotate, "turn to follow")
                        .on_hover_text("Rotate each copy so a ring faces outward, instead of every copy keeping the original's facing.");
                    if ui.add_enabled(n_obj > 0, egui::Button::new("Fill edge"))
                        .on_hover_text("Space the selected object(s) evenly around this shape: the original moves to the first point and copies fill the rest.")
                        .on_disabled_hover_text("Select an object in the viewport first")
                        .clicked()
                    {
                        self.array_along_shape(si, self.array_count, self.array_rotate);
                    }
                });
            } else if !self.shapes.is_empty() {
                ui.small("Click a shape (or a row above) to select it — then you can move it with G or fill its edge.");
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Mirror across");
                egui::ComboBox::from_id_salt("cad-mirror-axis")
                    .selected_text(["X", "Y", "Z"][self.mirror_axis.min(2)])
                    .show_ui(ui, |ui| {
                        for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut self.mirror_axis, i, *n);
                        }
                    });
                ui.checkbox(&mut self.mirror_copy, "copy")
                    .on_hover_text("Leave the original and mirror a duplicate — build one half of a symmetric map, then mirror it.");
                if ui.button("Mirror about centre")
                    .on_hover_text("Mirror the selection about its OWN centre. To mirror about a point you pick, use the Mirror TOOL above and click that point.")
                    .clicked()
                {
                    match self.selection_obb().map(|o| o.c) {
                        Some(p) => self.mirror_selection(p),
                        None => self.status = "Mirror: select an object first".into(),
                    }
                }
            });
            ui.separator();
            let n_int = self.guide_targets().iter().filter(|t| t.kind == construct::SnapKind::Intersection).count();
            ui.small(format!("{} guide(s), {n_int} intersection(s)", self.guides.len()));
            let mut del: Option<usize> = None;
            egui::ScrollArea::vertical().max_height(120.0).id_salt("cad-guides").show(ui, |ui| {
                for i in 0..self.guides.len() {
                    let (len, a, b) = {
                        let g = &self.guides[i];
                        (g.length(), g.a.kind.label(), g.b.kind.label())
                    };
                    ui.horizontal(|ui| {
                        let sel = self.sel_guide == Some(i);
                        if ui.selectable_label(sel, format!("{i}: {a}->{b}  {len:.2} wu")).clicked() {
                            self.sel_guide = if sel { None } else { Some(i) };
                        }
                        if ui.small_button("x").clicked() {
                            del = Some(i);
                        }
                    });
                }
            });
            if let Some(i) = del {
                self.guides.remove(i);
                self.sel_guide = None;
                self.overlays_dirty = true;
            }
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.guide_dist).range(0.0..=500.0).speed(0.1).prefix("dist "));
                if ui.button("Place along guide")
                    .on_hover_text("Move the selection to exactly this distance from the selected guide's first point, along the guide.")
                    .clicked()
                {
                    match self.sel_guide.and_then(|i| self.guides.get(i)).map(|g| (g.a.pos, g.dir())) {
                        Some((a, d)) if d.length_squared() > 0.0 => {
                            let t = a + d * self.guide_dist;
                            self.anchor_selection_to(t);
                        }
                        _ => self.status = "Place along guide: pick a guide in the list first".into(),
                    }
                }
                if ui.button("Clear guides").clicked() {
                    self.guides.clear();
                    self.construct_pending = None;
                    self.sel_guide = None;
                    self.overlays_dirty = true;
                }
            });
    }

    /// Model Painter: voxelise an .obj and spawn a forge block per surface voxel.
    fn model_painter_ui(&mut self, ui: &mut egui::Ui) {
            if self.model_candidates.is_empty() {
                ui.small("Put .obj files in the exe's models\\ folder.");
            }
            let sel = self
                .selected_model
                .and_then(|i| self.model_candidates.get(i))
                .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                .unwrap_or_else(|| "Select .obj…".into());
            egui::ComboBox::from_id_salt("model-picker")
                .selected_text(sel)
                .show_ui(ui, |ui| {
                    for (i, p) in self.model_candidates.iter().enumerate() {
                        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        ui.selectable_value(&mut self.selected_model, Some(i), name);
                    }
                });
            if ui.button("↻").on_hover_text("Rescan models\\").clicked() {
                self.model_candidates = voxel::model_catalog();
            }
            ui.add(egui::DragValue::new(&mut self.paint_res).range(4..=48).prefix("res "));
            ui.add(egui::DragValue::new(&mut self.paint_scale).range(2.0..=200.0).prefix("size "));
            if ui.button("Voxelize & paint (uses selected palette item)").clicked() {
                self.paint_model();
            }
    }

    /// Import external geometry (OBJ/GLB — converted from another game's BSP) as a persistent
    /// base to forge over.
    fn import_geometry_ui(&mut self, ui: &mut egui::Ui) {
            ui.small("OBJ/GLB in the exe's models\\ folder. Convert other-game BSPs to OBJ/GLB, then import + forge on top.");
            let sel = self
                .selected_import
                .and_then(|i| self.model_candidates.get(i))
                .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                .unwrap_or_else(|| "Select .obj/.glb…".into());
            egui::ComboBox::from_id_salt("import-geo-picker")
                .selected_text(sel)
                .show_ui(ui, |ui| {
                    for (i, p) in self.model_candidates.iter().enumerate() {
                        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        ui.selectable_value(&mut self.selected_import, Some(i), name);
                    }
                });
            ui.horizontal(|ui| {
                if ui.button("↻").on_hover_text("Rescan models\\").clicked() {
                    self.model_candidates = voxel::model_catalog();
                }
                if ui.button("Import as base").clicked() {
                    self.do_import_geometry();
                }
                if ui.button("Clear imported").clicked() {
                    self.renderer.clear_imported_meshes();
                    self.import_tris.clear();
                    self.spawn_status = "Cleared imported geometry.".into();
                }
            });
    }

    /// Lighting Lab: live sliders for every lighting term, plus real-time probe GI and the
    /// path-traced bake. Lives in Settings ▸ Lighting & rendering.
    fn lighting_lab_ui(&mut self, ui: &mut egui::Ui) {
            let q = self.render_state.queue.clone();
            let q = &q;
            let h4 = self.renderer.h4_meter_on();
            // `// #h4-expo-3` SEED ONLY -- read what is already rendering, never write.
            // This block used to PUSH its own values on the panel's first draw (base gain, key +
            // band, fog, bloom, lighting mults) with the band seeded from `scene_ctl` alone. On a
            // Halo 4 map `scene_ctl` is closed, so opening Settings > Lighting replaced the cfxs
            // absolute band with Reach's [2^0, 2^2], the base gain with 0.669, and -- because
            // `set_auto_exposure` also clears them -- the engine meter, the filmic curve and the
            // colour-grading LUT that `h4::lighting::apply_post` had installed. On EVERY map it
            // also re-bridged the fog density from the shader's 0.01 to 0.0025. Opening a panel
            // must not change the picture: the values below come from the renderer + the loaded
            // map's own tags, and only a widget's `.changed()` pushes.
            if !self.ll_seeded {
                let (base, key, lo_g, hi_g, cal) = self.renderer.exposure_params();
                self.ll_base = base;
                self.ll_meter_cal = cal;
                self.ll_key = key;
                if let Some(x) = &self.h4_cfxs {
                    // Halo 4: the band is ABSOLUTE stops (cfxs exposure range) and the meter's
                    // key is 1.0 (`apply_post`), so show the authored stops themselves.
                    self.ll_min_ev = x.exposure_range[0];
                    self.ll_max_ev = x.exposure_range[1];
                } else {
                    // Reach: `set_auto_exposure` stores the band as key * 10 * 2^EV gains.
                    let c = (key * 10.0).max(1e-4);
                    self.ll_min_ev = (lo_g / c).max(1e-6).log2();
                    self.ll_max_ev = (hi_g / c).max(1e-6).log2();
                }
                self.ll_fog = self.renderer.fog_wu_effective();
                self.ll_seeded = true;
            }
            ui.label(egui::RichText::new("EXPOSURE").small().strong());
            if ui.checkbox(&mut self.ll_manual_exp, "manual (auto-exposure OFF)").changed() {
                self.renderer.set_fixed_exposure(q, if self.ll_manual_exp { self.ll_exp } else { 0.0 });
                if !self.ll_manual_exp { self.h4_ae_stops = None; self.h4_ae_hist.clear(); }
            }
            if self.ll_manual_exp {
                if ui.add(egui::Slider::new(&mut self.ll_exp, 0.02..=4.0).logarithmic(true).text("exposure")).changed() {
                    self.renderer.set_fixed_exposure(q, self.ll_exp);
                }
            } else {
                if ui.add(egui::Slider::new(&mut self.ll_base, 0.05..=3.0).logarithmic(true).text("base gain")).changed() {
                    self.renderer.set_exposure(q, self.ll_base);
                }
                let mut ae = false;
                // The Halo 4 meter has no key (it solves the stops the engine converges to and
                // the post pass reads key = 1.0), so the slider would be meaningless there.
                if !h4 {
                    ae |= ui.add(egui::Slider::new(&mut self.ll_key, 0.01..=0.5).text("auto key")).changed();
                }
                let (lbl_lo, lbl_hi) = if h4 { ("band min stops", "band max stops") } else { ("auto min EV", "auto max EV") };
                ae |= ui.add(egui::Slider::new(&mut self.ll_min_ev, -4.0..=2.0).text(lbl_lo)).changed();
                ae |= ui.add(egui::Slider::new(&mut self.ll_max_ev, 0.0..=5.0).text(lbl_hi)).changed();
                if ae { self.ll_push_band(q); }
                if ui.add(egui::Slider::new(&mut self.ll_meter_cal, 0.1..=4.0).logarithmic(true).text("meter cal")).changed() {
                    self.renderer.set_exp_cal(q, self.ll_meter_cal);
                }
            }
            ui.separator();
            ui.label(egui::RichText::new("LIGHTING").small().strong());
            let mut lm = false;
            lm |= ui.add(egui::Slider::new(&mut self.ll_sun, 0.0..=4.0).text("sun")).changed();
            lm |= ui.add(egui::Slider::new(&mut self.ll_ambient, 0.0..=4.0).text("ambient")).changed();
            lm |= ui.add(egui::Slider::new(&mut self.ll_lightmap, 0.0..=4.0).text("lightmap")).changed();
            if lm { self.renderer.set_lighting_mults(self.ll_sun, self.ll_ambient, self.ll_lightmap); }
            ui.separator();
            ui.label(egui::RichText::new("POST").small().strong());
            if ui.add(egui::Slider::new(&mut self.bloom_scale, 0.0..=2.0).text("bloom")).changed() {
                self.renderer.set_bloom(q, self.bloom_scale);
            }
            if ui.add(egui::Slider::new(&mut self.ll_fog, 0.0..=0.03).text("fog density")).changed() {
                self.renderer.set_fog_wu(self.ll_fog);
            }
            if ui.button("Reset all").clicked() {
                // Back to what the LOADED MAP authored (not to Reach's hardcoded numbers):
                // `// #h4-expo-3` a Halo 4 map's base gain is 1.0 and its band is the cfxs
                // absolute stops, and its meter / filmic / LUT must survive the reset.
                self.bloom_scale = 1.0; self.ll_manual_exp = false; self.ll_exp = 0.66943294;
                self.ll_meter_cal = 1.0;
                self.ll_sun = 1.0; self.ll_ambient = 1.0; self.ll_lightmap = 1.0;
                self.ll_fog = hms_render::FOG_WU_DEFAULT;
                if let Some(x) = &self.h4_cfxs {
                    self.ll_base = 1.0;
                    self.ll_key = 1.0;
                    self.ll_min_ev = x.exposure_range[0];
                    self.ll_max_ev = x.exposure_range[1];
                } else {
                    let (k, lo, hi) = self.scene_ctl.as_ref().and_then(|s| s.autoexposure_band()).unwrap_or((0.1, 0.0, 2.0));
                    self.ll_key = k; self.ll_min_ev = lo; self.ll_max_ev = hi;
                    self.ll_base = self.scene_ctl.as_ref().map_or(0.66943294, |sc| sc.exposure());
                }
                self.renderer.set_bloom(q, self.bloom_scale);
                self.renderer.set_fixed_exposure(q, 0.0);
                self.h4_ae_stops = None; self.h4_ae_hist.clear();
                self.renderer.set_exposure(q, self.ll_base);
                self.ll_push_band(q);
                self.renderer.set_exp_cal(q, self.ll_meter_cal);
                self.renderer.set_lighting_mults(self.ll_sun, self.ll_ambient, self.ll_lightmap);
                self.renderer.set_fog_wu(self.ll_fog);
            }
    }

    /// Push the exposure band the Lighting panel's sliders describe. `// #h4-expo-3` Halo 4 keeps
    /// its own lane: its band is ABSOLUTE gains (`E = 2^stops / 1.4938016`, halo4.dll
    /// `sub_180375EAC`) with the meter's key 1.0, and Reach's `set_auto_exposure` would clear the
    /// engine meter, the filmic curve and the colour-grading LUT that `apply_post` installed.
    fn ll_push_band(&mut self, queue: &eframe::wgpu::Queue) {
        let hi_ev = self.ll_max_ev.max(self.ll_min_ev + 0.01);
        if self.renderer.h4_meter_on() {
            let g = h4::lighting::gain_of_stops;
            self.renderer.set_exposure_band_abs(queue, 1.0, g(self.ll_min_ev), g(hi_ev));
        } else {
            self.renderer.set_auto_exposure(queue, self.ll_key, self.ll_min_ev, hi_ev);
        }
    }

    /// Settings > Advanced (diagnostics): camera pose, the selected object's raw datum / forge
    /// index / render-model material, the variant load diagnostic, load timing, and "rebuild
    /// forge objects".
    fn advanced_diag_ui(&mut self, ui: &mut egui::Ui) {
        ui.small(egui::RichText::new("Read-only readouts for troubleshooting; nothing here is needed to forge.").weak());
        ui.monospace(format!(
            "camera pos ({:.2}, {:.2}, {:.2})  yaw {:.3}  pitch {:.3}",
            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, self.camera.yaw, self.camera.pitch
        ));
        if !self.map_status.is_empty() {
            ui.monospace(format!("map: {}", self.map_status));
        }
        if !self.forge_diag.is_empty() {
            ui.monospace(format!("variant: {}", self.forge_diag));
        }
        ui.monospace(format!(
            "objects: {} variant, {} placed, {} unresolved slots, {} palette entries",
            self.mvar_objects.len(),
            self.local_objects.len(),
            self.mvar_unresolved.len(),
            self.static_palette.len()
        ));
        if let Some(datum) = self.selected_datum {
            ui.separator();
            ui.monospace(format!("selected datum 0x{datum:08X}  ({} in selection)", self.selected_set.len()));
            if let Some((idx, _, _, _)) = self.forge_rows.iter().find(|(_, d, _, _)| *d == datum) {
                ui.monospace(format!("forge index {idx}"));
            }
            if let Some(o) = self.last_objects.iter().find(|o| o.datum == datum) {
                ui.monospace(format!(
                    "pos ({:.2}, {:.2}, {:.2})  mode tag 0x{:08X}  object tag 0x{:08X}",
                    o.pos[0], o.pos[1], o.pos[2], o.mode_tag, o.primary_tag
                ));
            }
            if let Some(m) = self.mvar_meta.get(&datum) {
                ui.monospace(format!("record: {}  folder {} item {}  slot {}", m.name, m.folder, m.item, m.slot));
            }
            if let Some((d, e, b)) = self.props_mat {
                let blend = match b {
                    0 => "opaque",
                    1 => "additive",
                    2 => "multiply",
                    3 => "alpha-blend",
                    _ => "other",
                };
                ui.monospace(format!("material 0: diffuse 0x{d:08X}  emissive 0x{e:08X}  blend {b} ({blend})"));
            }
        }
        ui.separator();
        if ui.button("Rebuild forge objects")
            .on_hover_text("Rebuild the object meshes only (the map stays loaded)")
            .clicked()
        {
            if let Some(s) = self.objscene_mut() {
                s.invalidate();
            }
            self.last_objects.clear();
        }
    }

    /// Settings > Camera & rendering: fly speed, camera stand-off and the render TUNING
    /// (path-traced lighting, real-time probe GI and its sun / light editing). #view-menu: every
    /// what-is-DRAWN switch -- geometry, the Forge extras and the overlays -- lives in the View
    /// menu instead, so "how it is rendered" and "what I can see" are not in the same list.
    fn view_settings_ui(&mut self, ui: &mut egui::Ui) {
        // Fly speed (Ctrl + wheel over the viewport is the quick way).
        ui.add(egui::Slider::new(&mut self.move_speed, 0.5..=200.0).text("fly speed"))
            .on_hover_text("Camera speed while flying (hold the right mouse button + WASD). Ctrl + mouse wheel over the viewport does the same.");
        // Camera collision. Kept beside the other viewport behaviour rather than in the
        // lighting block, because it changes how the camera MOVES, not how the scene looks.
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.cam_standoff_on, "Camera stand-off")
                .on_hover_text("While flying, keep the camera at least this far from any solid surface — terrain, BSP and placed forge objects. Stops the view burying itself in geometry.");
            ui.add(egui::DragValue::new(&mut self.cam_standoff).range(0.1..=50.0).speed(0.1).suffix(" wu"));
        });
        ui.horizontal(|ui| {
            if ui.button("Push camera out")
                .on_hover_text("One-shot: move the camera to the nearest spot with that much clearance (works even when fully buried)")
                .clicked()
            {
                let moved = self.camera_standoff(self.cam_standoff);
                let (after, _, _) = self.camera_clearance(self.cam_standoff);
                self.status = format!(
                    "Camera moved {moved:.2} wu — nearest surface {}",
                    after.map(|t| format!("{t:.2} wu")).unwrap_or_else(|| format!(">{:.2} wu", self.cam_standoff))
                );
            }
            if ui.button("Measure").on_hover_text("Report the distance to the nearest surface in any direction").clicked() {
                let probe = (self.cam_standoff * 8.0).max(24.0);
                let (d, _, blocked) = self.camera_clearance(probe);
                self.status = match d {
                    Some(t) => format!("Nearest surface {t:.2} wu ({blocked}/64 rays blocked within {probe:.0} wu)"),
                    None => format!("Clear — nothing within {probe:.0} wu"),
                };
            }
        });
        ui.separator();
            // #view-menu: everything that switches part of the scene ON OR OFF is in the View
            // menu -- geometry (BSP / Terrain / Forge objects / Water / Sky), the Forge extras
            // (special FX / scaled objects / shadow casters) and the overlays (grid, lights,
            // boundary shapes, the selected object's collision + physics hulls, and the map
            // overlays). What is left here changes HOW the scene is rendered, not whether a
            // lane is drawn.
            // Ignore the map's baked lightmaps and show our GPU path-traced GI instead.
            let baking = self.bake_rx.is_some();
            ui.add_enabled_ui(!baking, |ui| {
                if ui.checkbox(&mut self.show_pathtraced, "Path-traced lighting")
                    .on_hover_text("Ignore the map's lightmaps; bake + show our GPU path-traced global illumination of the BSP (~a few seconds). Untick to restore stock.")
                    .changed()
                {
                    self.pathtrace_dirty = true;
                }
                // tool.exe's lightmap quality ladder — the same bake, more samples/bounces.
                let names = crate::scene::SceneController::BAKE_QUALITIES;
                let mut qi = self.bake_quality.min(names.len() - 1);
                egui::ComboBox::from_label("bake quality")
                    .selected_text(names[qi])
                    .show_ui(ui, |ui| {
                        for (i, n) in names.iter().enumerate() {
                            let (s, b, _, _) = crate::scene::SceneController::bake_quality(n);
                            ui.selectable_value(&mut qi, i, format!("{n}  ({s} spp, {b} bounce)"));
                        }
                    });
                if qi != self.bake_quality {
                    self.bake_quality = qi;
                    if self.show_pathtraced { self.pathtrace_dirty = true; } // re-bake at the new quality
                }
            });
            // Real-time probe GI (replaces the baked lightmaps live; Lighting Lab gain slider below).
            if ui.checkbox(&mut self.show_rtgi, "Real-time lighting (probe GI)")
                .on_hover_text("Ignore the map's lightmaps; light the level with a live probe grid (sun, sky, fixtures, bounce) updated every frame. Untick to restore stock.")
                .changed()
            {
                self.rtgi_pending = true;
            }
            if self.show_rtgi {
                if ui.add(egui::Slider::new(&mut self.rtgi_gain, 0.25..=4.0).logarithmic(true).text("GI gain")).changed() {
                    self.renderer.rtgi_set_gain(&self.render_state.queue, self.rtgi_gain);
                }
                // Move the SUN. Real-time only — the offline bake always uses the map's own
                // sun. Seeded from the map's sun the first time the panel is drawn.
                if !self.sun_edited {
                    let (d, _) = self.renderer.rtgi_sun();
                    let v = glam::Vec3::from(d).normalize_or_zero();
                    if v.length_squared() > 0.5 {
                        self.sun_pitch = v.z.clamp(-1.0, 1.0).asin().to_degrees();
                        self.sun_yaw = v.y.atan2(v.x).to_degrees();
                    }
                }
                let mut sun_changed = false;
                sun_changed |= ui.add(egui::Slider::new(&mut self.sun_yaw, -180.0..=180.0).text("sun yaw")).changed();
                sun_changed |= ui.add(egui::Slider::new(&mut self.sun_pitch, -5.0..=89.0).text("sun pitch")).changed();
                sun_changed |= ui.add(egui::Slider::new(&mut self.sun_scale, 0.0..=4.0).text("sun intensity")).changed();
                ui.horizontal(|ui| {
                    if ui.button("reset sun to map").clicked() {
                        self.sun_edited = false;
                        self.sun_scale = 1.0;
                        self.sun_base = None;
                        self.rtgi_pending = true; // rebuild from the map's own values
                    }
                    if self.scene_ctl.as_ref().map_or(false, |s| s.has_light_offsets()) && ui.button("reset moved lights").clicked() {
                        if let Some(sc) = self.scene_ctl.as_mut() { sc.clear_light_offsets(); }
                        self.rtgi_pending = true;
                    }
                });
                if sun_changed {
                    self.sun_edited = true;
                    self.apply_sun_edit();
                }
                // Pick a light and move it (real-time relighting). The light MARKERS are a
                // View-menu overlay (#view-menu), and picking one needs them on screen.
                if !self.show_lights {
                    ui.small(egui::RichText::new("Tick View > Overlays > Lights to pick and move a light.").weak());
                }
                if self.show_lights {
                    let lights = self.scene_ctl.as_ref().map(|s| s.editable_lights()).unwrap_or_default();
                    if !lights.is_empty() {
                        let cam = self.camera.pos;
                        let mut near: Vec<&(u32, [f32; 3], [f32; 3], f32, bool, [f32; 3])> = lights.iter().collect();
                        near.sort_by(|a, b| {
                            let d = |p: [f32; 3]| (glam::Vec3::from(p) - cam).length();
                            d(a.1).partial_cmp(&d(b.1)).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let cur = self.sel_light;
                        let label = cur.map(|i| format!("light {i}")).unwrap_or_else(|| "none".into());
                        egui::ComboBox::from_label("move light").selected_text(label).show_ui(ui, |ui| {
                            let mut sel = self.sel_light;
                            ui.selectable_value(&mut sel, None, "none");
                            for l in near.iter().take(24) {
                                let d = (glam::Vec3::from(l.1) - cam).length();
                                ui.selectable_value(&mut sel, Some(l.0), format!("light {} — {:.0} wu {}", l.0, d, if l.4 { "spot" } else { "omni" }));
                            }
                            if sel != self.sel_light { self.sel_light = sel; self.rebuild_overlays(); }
                        });
                        if let Some(idx) = self.sel_light {
                            let mut off = self.scene_ctl.as_ref().map(|s| s.light_offset(idx)).unwrap_or([0.0; 3]);
                            let mut moved = false;
                            moved |= ui.add(egui::Slider::new(&mut off[0], -30.0..=30.0).text("light X")).changed();
                            moved |= ui.add(egui::Slider::new(&mut off[1], -30.0..=30.0).text("light Y")).changed();
                            moved |= ui.add(egui::Slider::new(&mut off[2], -30.0..=30.0).text("light Z")).changed();
                            if ui.button("move light to camera").clicked() {
                                if let Some(base) = lights.iter().find(|l| l.0 == idx) {
                                    let cur_off = self.scene_ctl.as_ref().map(|s| s.light_offset(idx)).unwrap_or([0.0; 3]);
                                    let origin = glam::Vec3::from(base.1) - glam::Vec3::from(cur_off);
                                    let target = self.camera.pos;
                                    off = (target - origin).into();
                                    moved = true;
                                }
                            }
                            if moved {
                                if let Some(sc) = self.scene_ctl.as_mut() { sc.set_light_offset(idx, off); }
                                self.rtgi_pending = true;
                                self.rebuild_overlays();
                            }
                        }
                    }
                }
                let (np, sp, _, upd) = self.renderer.rtgi_info();
                ui.label(egui::RichText::new(format!("{np} probes, {sp:.1} wu spacing, {upd} updates")).small());
            }
            // Live progress bar while the bake worker runs (window stays responsive).
            if let Some(p) = &self.bake_progress {
                let frac = p.load(std::sync::atomic::Ordering::Relaxed) as f32 / 10000.0;
                ui.add(egui::ProgressBar::new(frac).show_percentage().text("Path-tracing…"));
            }
    }

    /// File→Save — overwrite the currently-open .mvar with the edited objects.
    fn save_current_variant(&mut self) {
        // A Halo 4 variant saves IN PLACE like Reach (a new variant asks for a name);
        // the Halo 4 gates (bounds / quota / budget / type) hold it the same way as Save As.
        let Some(dst) = self.current_variant_path.clone() else {
            if self.new_variant_template.is_some() { self.save_variant_as(); return; }
            self.spawn_status = "Save: no variant is open.".into();
            return;
        };
        // Hold the save while objects sit outside playable space (window decides).
        if self.bsp_warn_gate(BspWarnAction::Save) { return; }
        if self.h4_save_gate(BspWarnAction::Save) { return; }
        self.spawn_status = match self.save_variant_to(&dst) {
            Ok((n, _)) => {
                let extra = self.save_failure_note();
                self.resync_variant_globals(&dst);
                mapcat::save_last_variant(&dst.to_string_lossy()); // #dialogs: last-saved folder
                format!("Saved {n} objects -> {}{extra}", dst.file_name().unwrap_or_default().to_string_lossy())
            }
            Err(e) => { self.save_error = Some(format!("Save failed:\n{e}\n\nTarget: {}", dst.display())); format!("Save failed: {e}") }
        };
    }

    /// After a successful save the FILE carries the edits: re-seed the panel's
    /// baseline (and the quota `placed` column, chunk length / hash, stamps) from it so the
    /// pending-edit markers clear and the next diff is against what is on disk.
    fn resync_variant_globals(&mut self, path: &std::path::Path) {
        let g = if self.h4_active { h4::mvar::parse_h4_variant(path).ok().map(|v| v.globals()) } else { mvar::parse_variant(path).map(|v| v.globals) };
        if let Some(g) = g {
            let unlocked = self.bounds_edit_unlocked;
            self.seed_variant_globals(g);
            self.bounds_edit_unlocked = unlocked;
        }
    }

    /// File→Save As — #dialogs: open the same custom browser in SAVE mode (a native OS picker
    /// used to run here; open and save now share one window). The write happens when the user
    /// presses Save inside the browser (`finish_variant_save_as`). Both games route through it;
    /// the Halo 4 hopper-folder refusal still fires at write time via `save_variant_to`.
    fn save_variant_as(&mut self) {
        if self.current_variant_path.is_none() && self.new_variant_template.is_none() {
            self.spawn_status = "Save As: no variant is open.".into();
            return;
        }
        // Hold the save while objects sit outside playable space (window decides).
        if self.bsp_warn_gate(BspWarnAction::SaveAs) { return; }
        // The Halo 4 gates (bounds / quota / budget / type) hold the save the same way.
        if self.h4_save_gate(BspWarnAction::SaveAs) { return; }
        self.open_variant_save_browser();
    }

    /// #dialogs: the game's `map_variants` folder for the ACTIVE game — where MCC lists custom
    /// variants (`haloreach\map_variants\*.mvar` / `halo4\map_variants\*.mvar`). The last-resort
    /// save-directory seed so a first-ever save lands there, not the working directory.
    fn game_map_variants_dir(&self) -> Option<std::path::PathBuf> {
        if self.h4_active {
            crate::h4::cache::maps_dir()
                .and_then(|m| m.parent().map(|d| d.join("map_variants")))
                .filter(|d| d.is_dir())
        } else {
            mapcat::variant_dirs().into_iter().next()
        }
    }

    /// #dialogs: the folder a SAVE browser should open into (deliverable 3):
    /// the open variant's folder → the last folder actually saved/opened (`last_variant.txt`
    /// parent, if it still exists) → the game's `map_variants` → sane fallbacks.
    fn variant_save_seed_dir(&self) -> std::path::PathBuf {
        self.current_variant_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .or_else(|| self.new_variant_template.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .or_else(|| Self::hms_data_dir()
                .and_then(|d| std::fs::read_to_string(d.join("last_variant.txt")).ok())
                .and_then(|s| std::path::Path::new(s.trim()).parent().map(|d| d.to_path_buf()))
                .filter(|d| d.is_dir()))
            .or_else(|| self.game_map_variants_dir())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| {
                #[cfg(windows)] { std::path::PathBuf::from("C:\\") }
                #[cfg(not(windows))] { std::env::var("HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("/")) }
            })
    }

    /// #dialogs: resolve a save target `<dir>/<name>`, appending `.mvar` when the user typed no
    /// extension. None when the name is blank. Shared by the Save button and the test hook.
    fn resolve_save_target(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
        let name = name.trim();
        if name.is_empty() { return None; }
        let mut name = name.to_string();
        if std::path::Path::new(&name).extension().is_none() { name.push_str(".mvar"); }
        Some(dir.join(name))
    }

    /// #dialogs: the file name a Save browser seeds its field with — the open file's own name,
    /// else a name built from the variant title, else `new_variant.mvar` (same as the old picker).
    fn variant_save_seed_name(&self) -> String {
        self.current_variant_path.as_ref().and_then(|p| p.file_name()).map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                let t: String = self.variant_title.trim().chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
                format!("{}.mvar", if t.is_empty() { "new_variant".to_string() } else { t })
            })
    }

    /// #dialogs: open the shared browser in SAVE mode, seeded per `variant_save_seed_dir` /
    /// `variant_save_seed_name`. (`save_variant_as` runs the save gates first; this is also the
    /// test hook target.)
    fn open_variant_save_browser(&mut self) {
        let dir = self.variant_save_seed_dir();
        let name = self.variant_save_seed_name();
        self.variant_browser_drives = Self::vb_volumes();
        self.variant_browser_bookmarks = Self::vb_load_bookmarks();
        self.variant_browser_history.clear();
        self.variant_browser_hist_pos = 0;
        self.vb_navigate(&dir, true); // clears the filename field
        self.variant_browser_filename = name; // …so seed it AFTER navigating
        self.variant_browser_save = true;
        self.variant_browser_overwrite = None;
        self.variant_browser_open = true;
    }

    /// #dialogs: the browser's Save action — resolve `<dir>/<name>` and either write it, or arm
    /// the inline overwrite confirm when the target already exists. Returns true when the browser
    /// should close (i.e. the file was written). Shared by the Save button and the test hook.
    fn browser_try_save(&mut self) -> bool {
        let Some(dir) = self.variant_browser_dir.clone() else { return false };
        let Some(dst) = Self::resolve_save_target(&dir, &self.variant_browser_filename) else { return false };
        if dst.exists() {
            self.variant_browser_overwrite = Some(dst);
            false
        } else {
            self.finish_variant_save_as(&dst);
            true
        }
    }

    /// #dialogs: the overwrite-confirm "Overwrite" action — write the armed target. Returns true
    /// when the browser should close.
    fn browser_confirm_overwrite(&mut self) -> bool {
        match self.variant_browser_overwrite.take() {
            Some(dst) => { self.finish_variant_save_as(&dst); true }
            None => false,
        }
    }

    /// #dialogs: write the edited objects to `dst` and adopt it as the open variant — the shared
    /// tail of Save As, called when the user presses Save in the browser. Mirrors the old native
    /// dialog's success/error handling and persists `dst`'s folder as the last-saved location.
    fn finish_variant_save_as(&mut self, dst: &std::path::Path) {
        self.spawn_status = match self.save_variant_to(dst) {
            Ok((n, _adds)) => {
                let extra = self.save_failure_note();
                self.current_variant_path = Some(dst.to_path_buf());
                self.new_variant_template = None;
                self.resync_variant_globals(dst);
                // Remember where the user saved so the next Open/Save browser starts here.
                mapcat::save_last_variant(&dst.to_string_lossy());
                format!("Saved {n} objects -> {}{extra}", dst.file_name().unwrap_or_default().to_string_lossy())
            }
            Err(e) => { self.save_error = Some(format!("Save As failed:\n{e}\n\nTarget: {}", dst.display())); format!("Save As failed: {e}") }
        };
    }

    /// "<map stem> (id N)" of a Halo 4 base map among the detected caches, or the bare id.
    fn h4_base_map_label(&self, map_id: u32) -> String {
        match self.map_candidates.iter().find(|c| c.game == mapcat::Game::Halo4 && c.map_id == Some(map_id)) {
            Some(c) => format!("{} (id {map_id})", c.stem),
            None => format!("map id {map_id} (not installed)"),
        }
    }

    /// The browser row hint of a Halo 4 variant: "Title - base map, N objects, author".
    fn h4_summary_line(&self, sm: &h4::mvar::H4VariantSummary) -> String {
        let title = if sm.title.is_empty() { "(untitled)" } else { sm.title.as_str() };
        let map = self.map_candidates.iter().find(|c| c.game == mapcat::Game::Halo4 && c.map_id == Some(sm.map_id))
            .map(|c| c.stem.clone()).unwrap_or_else(|| format!("map id {}", sm.map_id));
        let author = if sm.author.is_empty() { String::new() } else { format!(", {}", sm.author) };
        format!("{title} ({map}, {} objects{author})", sm.objects)
    }

    /// Open the FULLY-CUSTOM variant file browser (Blender-style: sidebar + nav bar + file list),
    /// seeding its directory from the current/last variant. Caches drives + bookmarks once here.
    fn open_variant_browser(&mut self) {
        let dir = self
            .variant_browser_dir
            .clone()
            .or_else(|| self.current_variant_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .or_else(|| {
                Self::hms_data_dir()
                    .and_then(|d| std::fs::read_to_string(d.join("last_variant.txt")).ok())
                    .and_then(|s| std::path::Path::new(s.trim()).parent().map(|d| d.to_path_buf()))
            })
            // No history yet → land in the MCC forge variants folder rather than
            // the working directory, which is where .mvar files actually live.
            .or_else(|| mapcat::variant_dirs().into_iter().next())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| {
                #[cfg(windows)] { std::path::PathBuf::from("C:\\") }
                #[cfg(not(windows))] { std::env::var("HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("/")) }
            });
        // Enumerated once per open; cheap thereafter.
        self.variant_browser_drives = Self::vb_volumes();
        self.variant_browser_bookmarks = Self::vb_load_bookmarks();
        self.variant_browser_history.clear();
        self.variant_browser_hist_pos = 0;
        self.vb_navigate(&dir, true);
        self.variant_browser_save = false; // #dialogs: OPEN mode
        self.variant_browser_overwrite = None;
        self.variant_browser_open = true;
    }

    /// Per-user data dir for browser state (`project::settings_dir`: %LOCALAPPDATA% on Windows,
    /// the XDG data dir on Unix). Created on demand so the first save doesn't fail.
    fn hms_data_dir() -> Option<std::path::PathBuf> {
        let dir = project::settings_dir()?;
        let _ = std::fs::create_dir_all(&dir);
        Some(dir)
    }

    /// Persisted bookmarks file (one path per line).
    fn vb_bookmarks_path() -> Option<std::path::PathBuf> {
        Self::hms_data_dir().map(|d| d.join("browser_bookmarks.txt"))
    }
    fn vb_load_bookmarks() -> Vec<std::path::PathBuf> {
        Self::vb_bookmarks_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).map(std::path::PathBuf::from).collect())
            .unwrap_or_default()
    }
    fn vb_save_bookmarks(&self) {
        if let Some(p) = Self::vb_bookmarks_path() {
            let body = self.variant_browser_bookmarks.iter().map(|b| b.to_string_lossy().into_owned()).collect::<Vec<_>>().join("\n");
            let _ = std::fs::write(p, body);
        }
    }

    /// Navigate to `dir`: read its listing, push to history (unless `record`==false for back/fwd),
    /// bump recent, and sync the editable path field. Clears the selection.
    fn vb_navigate(&mut self, dir: &std::path::Path, record: bool) {
        // Read subfolders + .mvar files with metadata.
        let mut dirs: Vec<FileEntry> = Vec::new();
        let mut files: Vec<FileEntry> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let path = e.path();
                let name = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                if name.starts_with('.') { continue; }
                let md = e.metadata().ok();
                let is_dir = md.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let modified = md.as_ref().and_then(|m| m.modified().ok());
                if is_dir {
                    dirs.push(FileEntry { path, name, is_dir: true, size: 0, modified, hint: String::new() });
                } else if path.extension().and_then(|x| x.to_str()).map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
                    let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
                    // A Halo 4 variant (mvar chunk v50) gets its resolved name, base map,
                    // object count and author in the row; Reach files keep a bare file name
                    let hint = h4::mvar::read_h4_summary(&path).map(|sm| self.h4_summary_line(&sm)).unwrap_or_default();
                    files.push(FileEntry { path, name, is_dir: false, size, modified, hint });
                }
            }
        }
        dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        let mut listing = dirs;
        listing.extend(files);
        self.variant_browser_listing = listing;
        self.vb_sort_listing();
        self.variant_browser_dir = Some(dir.to_path_buf());
        self.variant_browser_path_edit = dir.to_string_lossy().into_owned();
        self.variant_browser_selected = None;
        self.variant_browser_selected_meta = None;
        self.variant_browser_filename.clear();
        if record {
            // Truncate any forward history, then push.
            self.variant_browser_history.truncate(self.variant_browser_hist_pos + if self.variant_browser_history.is_empty() { 0 } else { 1 });
            if self.variant_browser_history.last().map(|p| p.as_path()) != Some(dir) {
                self.variant_browser_history.push(dir.to_path_buf());
                self.variant_browser_hist_pos = self.variant_browser_history.len() - 1;
            }
            // Recent: most-recent-first, dedup, cap 12.
            self.variant_browser_recent.retain(|p| p != dir);
            self.variant_browser_recent.insert(0, dir.to_path_buf());
            self.variant_browser_recent.truncate(12);
        }
    }

    fn vb_fmt_size(bytes: u64) -> String {
        if bytes >= 1 << 20 { format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64) }
        else if bytes >= 1 << 10 { format!("{:.1} KiB", bytes as f64 / (1u64 << 10) as f64) }
        else { format!("{bytes} B") }
    }
    /// Re-order the current listing by the chosen column (folders always first).
    fn vb_sort_listing(&mut self) {
        let col = self.variant_browser_sort_col;
        let asc = self.variant_browser_sort_asc;
        self.variant_browser_listing.sort_by(|a, b| {
            match (a.is_dir, b.is_dir) {
                (true, false) => return std::cmp::Ordering::Less,
                (false, true) => return std::cmp::Ordering::Greater,
                _ => {}
            }
            let ord = match col {
                1 => a.modified.cmp(&b.modified),
                2 => a.size.cmp(&b.size),
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            };
            if asc { ord } else { ord.reverse() }
        });
    }

    fn vb_fmt_date(t: Option<std::time::SystemTime>) -> String {
        // Elapsed-since-modified, coarse (no chrono dep). Good enough to sort visually.
        let Some(t) = t else { return String::new() };
        match t.elapsed() {
            Ok(d) => {
                let secs = d.as_secs();
                if secs < 3600 { format!("{} min ago", secs / 60) }
                else if secs < 86400 { format!("{} hr ago", secs / 3600) }
                else if secs < 86400 * 30 { format!("{} days ago", secs / 86400) }
                else { format!("{} mo ago", secs / (86400 * 30)) }
            }
            Err(_) => String::new(),
        }
    }

    /// Count SCALE-labeled forge objects in the loaded variant → (total, actually-scaled).
    /// "Actually-scaled" excludes spawn_seq==0 objects (unscaled ×1 — the convert walker
    /// leaves them untouched).
    fn count_scale_objects(&self) -> (usize, usize) {
        let mut total = 0usize;
        let mut scaled = 0usize;
        for m in self.mvar_meta.values() {
            if m.label.eq_ignore_ascii_case("scale") {
                total += 1;
                if m.spawn_seq != 0 {
                    scaled += 1;
                }
            }
        }
        (total, scaled)
    }

    /// Re-encode every SCALE object's spawn sequence from the active convention onto X330
    /// (the render standard), then switch the active convention to X330. Faithful port of the
    /// C# `ConvertScaleToX330` walker:
    ///   • only `SCALE`-labeled objects;
    ///   • spawn_seq==0 is unscaled (×1) → left completely alone;
    ///   • old scale = decode under the source convention WITH the object's team (so an X330
    ///     RED-team object's cosmic size is read correctly), then invert team-neutrally into
    ///     X330 (the inverse never flips team);
    ///   • write ONLY the spawn sequence — never team or gtLabel (the gametype's SCALE script
    ///     derives size from (seq, team); flipping team would trigger cosmic and balloon it).
    /// Returns (converted, left_unchanged).
    fn convert_scale_objects_to_x330(&mut self) -> (usize, usize) {
        use forge_scale::{scale_to_spawn_seq, spawn_seq_to_scale, team_from_u8, ScaleConvention};
        let source = self.sc_convention;
        let mut converted = 0usize;
        let mut unchanged = 0usize;
        for m in self.mvar_meta.values_mut() {
            if !m.label.eq_ignore_ascii_case("scale") {
                continue;
            }
            if m.spawn_seq == 0 {
                unchanged += 1; // SEQ_ZERO_IGNORE — unscaled, leave alone
                continue;
            }
            let real = spawn_seq_to_scale(m.spawn_seq, source, team_from_u8(m.team));
            let (new_seq, _) = scale_to_spawn_seq(real, ScaleConvention::X330); // team-neutral
            if new_seq as i32 != m.spawn_seq {
                converted += 1;
            } else {
                unchanged += 1;
            }
            m.spawn_seq = new_seq as i32; // SPAWN_SEQ_ONLY
        }
        self.sc_convention = ScaleConvention::X330;
        if converted > 0 {
            self.rebuild_overlays();
        }
        (converted, unchanged)
    }

    // ===================== objects outside playable space =====================

    /// The non-playable structure BSP index an object sits in, by its LIVE position
    /// (follows drags / script moves; `movable_pose` is the same source the gizmo writes).
    fn bsp_warn_datum(&self, datum: u32) -> Option<usize> {
        let scene = self.scene_ctl.as_ref()?;
        if !scene.has_non_playable_bsps() { return None; }
        let (pos, _, _) = self.movable_pose(datum)?;
        scene.non_playable_bsp_at(pos)
    }

    /// Every variant / placed object in a flagged BSP, as (datum, bsp index), in
    /// object-list order.
    fn bsp_warn_objects(&self) -> Vec<(u32, usize)> {
        let Some(scene) = self.scene_ctl.as_ref() else { return Vec::new() };
        if !scene.has_non_playable_bsps() { return Vec::new(); }
        self.mvar_objects.iter().chain(self.local_objects.iter())
            .filter_map(|o| scene.non_playable_bsp_at(glam::Vec3::from(o.pos)).map(|i| (o.datum, i)))
            .collect()
    }

    /// One human row "type  \"label\"  #slot  BSP n (name)  (x,y,z)" for a warning list.
    fn bsp_warn_row(&self, datum: u32, bsp: usize) -> String {
        let (name, label, slot) = match self.mvar_meta.get(&datum) {
            Some(m) => {
                let leaf = m.name.rsplit(['\\', '/']).next().unwrap_or(&m.name).to_string();
                let pretty = if leaf.is_empty() { "(unresolved)".to_string() } else { prettify_stringid(&leaf) };
                (pretty, m.label.clone(), if m.slot != 0xFFFF { format!("#{}", m.slot) } else { "new".to_string() })
            }
            None => {
                let name = self.local_objects.iter().find(|o| o.datum == datum)
                    .and_then(|o| self.static_palette.iter().find(|(t, _)| *t == o.primary_tag))
                    .map(|(_, n)| prettify_stringid(n.split(" · ").last().unwrap_or(n)))
                    .unwrap_or_else(|| "placed object".to_string());
                (name, String::new(), "placed".to_string())
            }
        };
        let bsp_name = self.scene_ctl.as_ref()
            .and_then(|s| s.structure_bsp_flags().get(bsp).map(|b| b.name.rsplit('\\').next().unwrap_or("").to_string()))
            .unwrap_or_default();
        let pos = self.movable_pose(datum).map(|(p, _, _)| p).unwrap_or(glam::Vec3::ZERO);
        format!("{}{}   {}   BSP {} ({})   ({:.1}, {:.1}, {:.1})",
            clip_chars(&name, 30),
            if label.is_empty() { String::new() } else { format!("  \"{}\"", clip_chars(&label, 14)) },
            slot, bsp, bsp_name, pos.x, pos.y, pos.z)
    }

    /// The wording shared by the object panel strip, the save window and the tooltips.
    const BSP_WARN_TEXT: &'static str = "Outside playable space: this object is in a BSP marked \
        'not normally playable space in multiplayer'; the game may push it back inside the map.";

    /// Before a GUI save, hold the save and open the warning window when any object is
    /// outside playable space. Returns true when the caller must NOT save now (the window took
    /// over; "Save anyway" re-issues the same action with `bsp_warn_skip_once` set).
    fn bsp_warn_gate(&mut self, action: BspWarnAction) -> bool {
        if self.bsp_warn_skip_once {
            self.bsp_warn_skip_once = false;
            return false;
        }
        let rows = self.bsp_warn_objects();
        if rows.is_empty() { return false; }
        self.bsp_warn_pending = Some((action, rows));
        true
    }

    /// Modal "the save did not happen" window. Stays until dismissed.
    fn save_error_window_ui(&mut self, ctx: &egui::Context) {
        let Some(msg) = self.save_error.clone() else { return };
        let mut open = true;
        let mut dismiss = false;
        egui::Window::new("Variant NOT saved")
            .collapsible(false)
            .resizable(false)
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut open)
            .show(ctx, |ui| {
                ui.set_max_width(560.0);
                ui.label(egui::RichText::new(&msg).strong());
                if msg.contains("Read-only") || msg.contains("read-only") || msg.contains("os error 30") {
                    ui.add_space(6.0);
                    ui.label("The destination is on a read-only drive/mount, so nothing was written. Choose a writable folder (Save As), or remount that drive read-write.");
                } else if msg.contains("denied") || msg.contains("os error 13") {
                    ui.add_space(6.0);
                    ui.label("Permission denied: the folder is not writable by this user.");
                }
                ui.add_space(8.0);
                if ui.button("OK").clicked() { dismiss = true; }
            });
        if dismiss || !open { self.save_error = None; }
    }

    /// The save-time warning window. Lists every offending object; double-click a row
    /// to select it and fly the camera there (closes the window, the save is NOT performed);
    /// "Save anyway" proceeds with the held save; "Cancel" drops it.
    fn bsp_warn_window_ui(&mut self, ctx: &egui::Context) {
        let Some((action, rows)) = self.bsp_warn_pending.clone() else { return };
        // Rows follow live positions: an object moved back inside since the gate drops off.
        let rows: Vec<(u32, usize)> = rows.into_iter()
            .filter_map(|(d, _)| self.bsp_warn_datum(d).map(|i| (d, i)))
            .collect();
        if rows.is_empty() {
            // Everything was moved back in — nothing to warn about; the user re-saves.
            self.bsp_warn_pending = None;
            self.status = "All objects are back inside playable space — save again.".into();
            return;
        }
        let mut save_anyway = false;
        let mut cancel = false;
        let mut goto: Option<u32> = None;
        let mut win_open = true;
        egui::Window::new("Objects outside playable space")
            .open(&mut win_open)
            .default_size([560.0, 320.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center())
            .collapsible(false)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("{} object(s) sit in a BSP marked 'not normally playable space in multiplayer'.", rows.len()))
                    .color(egui::Color32::from_rgb(255, 160, 60)).strong());
                ui.small("The game may push these back inside the map when the session cycles. Double-click a row to select the object and fly the camera to it.");
                ui.separator();
                let row_h = ui.spacing().interact_size.y;
                egui::ScrollArea::vertical().max_height(220.0).auto_shrink([false, false]).show_rows(ui, row_h, rows.len(), |ui, range| {
                    for k in range {
                        let (d, bsp) = rows[k];
                        let text = self.bsp_warn_row(d, bsp);
                        let sel = self.selected_set.contains(&d);
                        let r = ui.add_sized([ui.available_width(), row_h], egui::SelectableLabel::new(sel, text));
                        if r.double_clicked() { goto = Some(d); }
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    let verb = match action { BspWarnAction::Save => "Save anyway", BspWarnAction::SaveAs => "Save As anyway" };
                    if ui.button(verb).clicked() { save_anyway = true; }
                    if ui.button("Cancel").clicked() { cancel = true; }
                });
            });
        if let Some(d) = goto {
            self.selected_set = vec![d];
            self.selected_datum = Some(d);
            self.obj_list_anchor = Some(d);
            self.obj_list_last_sel = Some(d);
            self.apply_selection_highlight();
            self.frame_selected();
            self.bsp_warn_pending = None;
            return;
        }
        if save_anyway {
            self.bsp_warn_pending = None;
            self.bsp_warn_skip_once = true;
            match action {
                BspWarnAction::Save => self.save_current_variant(),
                BspWarnAction::SaveAs => self.save_variant_as(),
            }
            self.bsp_warn_skip_once = false; // never leak a skip into a later save
            return;
        }
        if cancel || !win_open {
            self.bsp_warn_pending = None;
            self.status = "Save cancelled.".into();
        }
    }

    /// The read-only identity rows of the Object panel. `full` = the map-object
    /// block (tag path / class / tag id / placement / position); otherwise just the tag line
    /// under a variant object's item name. Both end with a "Copy tag path" button (clipboard).
    fn identity_rows_ui(&self, ui: &mut egui::Ui, datum: u32, full: bool) {
        let Some(id) = self.object_identity(datum) else { return };
        let path_txt = if id.tag_path.is_empty() { "(unnamed tag)".to_string() } else { id.tag_path.clone() };
        let tag_line = |ui: &mut egui::Ui, text: String, hover: &str| {
            ui.add(egui::Label::new(egui::RichText::new(text).small().weak()).truncate()).on_hover_text(hover);
        };
        if full {
            egui::Grid::new(("obj-identity", datum)).num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                ui.small("tag");
                tag_line(ui, path_txt.clone(), &path_txt);
                ui.end_row();
                ui.small("class");
                ui.small(if id.class.is_empty() { "?" } else { id.class.as_str() });
                ui.end_row();
                ui.small("tag id");
                ui.small(format!("0x{:04x}   (datum 0x{:08X})", id.obj_tag & 0xFFFF, datum));
                ui.end_row();
                if let obj_identity::ObjSource::Scenario { index, palette_index, name_index } = &id.source {
                    ui.small("placement");
                    ui.small(format!("scnr #{index}   palette {palette_index}   name {name_index}"))
                        .on_hover_text("HMS enumeration index of this scenario placement; index into the category's scnr palette block; scnr object-names index (-1 = unnamed)");
                    ui.end_row();
                }
                if let Some(o) = self.scenario_objects.iter().find(|o| o.datum == datum) {
                    ui.small("position");
                    ui.small(format!("{:.2}, {:.2}, {:.2}", o.pos[0], o.pos[1], o.pos[2]));
                    ui.end_row();
                }
            });
        } else {
            ui.horizontal(|ui| {
                ui.small("tag:");
                tag_line(ui, path_txt.clone(), &format!("{path_txt}\nclass: {}   tag id: 0x{:04x}", if id.class.is_empty() { "?" } else { id.class.as_str() }, id.obj_tag & 0xFFFF));
            });
        }
        if !id.tag_path.is_empty() && ui.small_button("Copy tag path").on_hover_text(&id.tag_path).clicked() {
            ui.ctx().copy_text(id.tag_path.clone());
        }
    }

    /// The warning strip at the top of the Object panel — shown when the primary
    /// selection or ANY object in a mass selection is in a flagged BSP. Recomputed per frame for
    /// the selection only (a handful of AABB tests), so it follows a drag live.
    fn bsp_warn_panel_strip(&self, ui: &mut egui::Ui, datum: u32) {
        let mut hits: Vec<(u32, usize)> = Vec::new();
        for &d in std::iter::once(&datum).chain(self.selected_set.iter()) {
            if hits.iter().any(|(x, _)| *x == d) { continue; }
            if let Some(i) = self.bsp_warn_datum(d) { hits.push((d, i)); }
        }
        if hits.is_empty() { return; }
        let orange = egui::Color32::from_rgb(255, 160, 60);
        egui::Frame::new()
            .fill(egui::Color32::from_rgba_unmultiplied(120, 60, 0, 70))
            .stroke(egui::Stroke::new(1.0_f32, orange))
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                let n_sel = self.selected_set.len().max(1);
                let head = if n_sel > 1 && hits.len() != n_sel {
                    format!("! {} of {} selected objects are outside playable space", hits.len(), n_sel)
                } else if n_sel > 1 {
                    "! Selected objects are outside playable space".to_string()
                } else {
                    "! Outside playable space".to_string()
                };
                ui.label(egui::RichText::new(head).color(orange).strong());
                ui.label(egui::RichText::new(Self::BSP_WARN_TEXT).small().color(orange));
                if let Some(scene) = self.scene_ctl.as_ref() {
                    let names: Vec<String> = hits.iter().map(|&(_, i)| i).collect::<std::collections::BTreeSet<_>>().into_iter()
                        .map(|i| scene.structure_bsp_flags().get(i).map(|b| format!("BSP {} ({})", i, b.name.rsplit('\\').next().unwrap_or(""))).unwrap_or_else(|| format!("BSP {i}")))
                        .collect();
                    ui.small(egui::RichText::new(names.join(", ")).color(orange));
                }
            });
        ui.add_space(4.0);
    }

    /// Tools ▸ Scale converter — pick the convention the loaded map's SCALE objects were
    /// authored in (they re-render at the right size immediately), then convert them onto X330.
    /// This actually rewrites the objects' spawn sequences; it is not a calculator.
    fn scale_converter_ui(&mut self, ctx: &egui::Context) {
        if !self.show_scale_converter {
            return;
        }
        let (total, scaled) = self.count_scale_objects();
        let before = self.sc_convention;
        let mut do_convert = false;
        let mut win_open = true;
        egui::Window::new("Scale converter")
            .open(&mut win_open)
            .fixed_size([430.0, 250.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center())
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(
                    "Forge SCALE objects encode their size in the spawn sequence, under a \
                     community convention. Pick the one the map was authored in — its objects \
                     re-render at the correct size — then convert them onto X330 (the render \
                     standard). Cosmic scaling is automatic for RED-team objects.",
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Active convention:");
                    egui::ComboBox::from_id_salt("scale-convention")
                        .selected_text(self.sc_convention.label())
                        .width(250.0)
                        .show_ui(ui, |ui| {
                            for &c in forge_scale::ScaleConvention::ALL {
                                ui.selectable_value(&mut self.sc_convention, c, c.label())
                                    .on_hover_text(c.hover());
                            }
                        });
                });
                ui.add_space(6.0);
                ui.separator();
                if total == 0 {
                    ui.label("No SCALE-labeled objects in the loaded map.");
                } else {
                    ui.label(format!(
                        "{scaled} scaled object(s) — {} of {total} SCALE-labeled are unscaled (left alone).",
                        total - scaled
                    ));
                    let is_x330 = self.sc_convention == forge_scale::ScaleConvention::X330;
                    ui.add_enabled_ui(!is_x330 && scaled > 0, |ui| {
                        if ui
                            .button(format!("Convert {scaled} object(s) -> X330"))
                            .on_hover_text(
                                "Rewrite each SCALE object's spawn sequence from the active \
                                 convention onto X330. Only the spawn sequence changes (team \
                                 and label are untouched). File > Save to persist into the .mvar.",
                            )
                            .clicked()
                        {
                            do_convert = true;
                        }
                    });
                    if is_x330 {
                        ui.small("Active convention is already X330 — nothing to convert.");
                    }
                }
                if self.status.starts_with("Scale:") {
                    ui.add_space(4.0);
                    ui.small(&self.status);
                }
            });
        // A convention change re-renders SCALE objects under the new decoding (live preview).
        if self.sc_convention != before {
            self.rebuild_overlays();
        }
        if do_convert {
            let (c, u) = self.convert_scale_objects_to_x330();
            self.status = format!("Scale: converted {c} object(s) to X330 ({u} left unchanged). File > Save to persist.");
        }
        self.show_scale_converter = win_open;
    }

    /// The fully-custom variant file browser window (no OS dialog): left sidebar (Bookmarks /
    /// System / Volumes / Recent), a nav bar (back/forward/up/refresh + editable path + search), a
    /// Name/Date/Size file list, a details panel for the selected variant, and a filename + Open/Cancel
    /// footer. Modeled on Blender's File View.
    fn variant_browser_ui(&mut self, ctx: &egui::Context) {
        if !self.variant_browser_open {
            return;
        }
        let mut win_open = true;
        let mut navigate: Option<std::path::PathBuf> = None;
        let mut go_back = false;
        let mut go_fwd = false;
        let mut refresh = false;
        let mut select: Option<std::path::PathBuf> = None;
        let mut open_now: Option<std::path::PathBuf> = None;
        let mut add_bookmark = false;
        let mut remove_bookmark: Option<std::path::PathBuf> = None;
        let mut close_win = false;
        let mut sort_click: Option<u8> = None;
        let mut save_click = false; // #dialogs: Save pressed (or Enter in the filename field)
        let mut confirm_overwrite = false;
        let mut cancel_overwrite = false;
        let sys_dirs = Self::vb_system_dirs();
        // The folders that actually hold variants — the game's own map_variants /
        // hopper_map_variants plus every per-account MCC save folder. Cached on the App because
        // discovery walks the filesystem and this runs every frame the window is open.
        if self.variant_browser_quick.is_empty() {
            self.variant_browser_quick = mapcat::quick_variant_dirs();
        }
        // #dialogs: one window, two modes. The title also keys egui's remembered size/pos, so
        // Open and Save each keep their own.
        let save_mode = self.variant_browser_save;
        let title = if save_mode { "Save map variant" } else { "Open map variant" };
        let mut win = egui::Window::new(title)
            .open(&mut win_open)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center())
            .collapsible(false);
        // Normally a freely resizable window; the test hook can pin an exact size so a scripted
        // run can screenshot the reflow at min / default / large.
        win = match self.variant_browser_force_size {
            Some(sz) => win.fixed_size(sz).resizable(false),
            None => win.resizable(true).default_size([760.0, 560.0]).min_size([560.0, 380.0]),
        };
        win.show(ctx, |ui| {
                // ── Nav bar: back / forward / up / refresh + editable path + search ──
                ui.horizontal(|ui| {
                    let can_back = self.variant_browser_hist_pos > 0;
                    let can_fwd = self.variant_browser_hist_pos + 1 < self.variant_browser_history.len();
                    if ui.add_enabled(can_back, egui::Button::new("<")).on_hover_text("Back").clicked() { go_back = true; }
                    if ui.add_enabled(can_fwd, egui::Button::new(">")).on_hover_text("Forward").clicked() { go_fwd = true; }
                    if ui.button("Up").on_hover_text("Up one folder").clicked() {
                        if let Some(parent) = self.variant_browser_dir.as_ref().and_then(|d| d.parent()) {
                            navigate = Some(parent.to_path_buf());
                        }
                    }
                    if ui.button("⟳").on_hover_text("Refresh").clicked() { refresh = true; }
                    // Editable path field (Enter to go).
                    let resp = ui.add(egui::TextEdit::singleline(&mut self.variant_browser_path_edit).desired_width((ui.available_width() - 200.0).max(120.0)));
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let p = std::path::PathBuf::from(self.variant_browser_path_edit.trim());
                        if p.is_dir() { navigate = Some(p); }
                    }
                    ui.label("Search:");
                    ui.add(egui::TextEdit::singleline(&mut self.variant_browser_filter).desired_width(140.0).hint_text("search"));
                });
                ui.separator();
                let f = self.variant_browser_filter.to_lowercase();
                // #dialogs: reserve room for the details strip (up to 2 wrapped lines) + the
                // footer + separators (and the overwrite-confirm strip when it is showing), so
                // the file list fills the rest and the footer never clips. Recomputed every frame
                // from the ACTUAL window size, so the list grows/shrinks as the window resizes.
                let reserve = 118.0 + if self.variant_browser_overwrite.is_some() { 30.0 } else { 0.0 };
                let body_h = (ui.available_height() - reserve).max(140.0);
                let full_w = ui.available_width();
                let side_w = 180.0_f32;
                let main_w = (full_w - side_w - 16.0).max(220.0);
                ui.horizontal_top(|ui| {
                    // ── SIDEBAR (pinned width) ──
                    ui.vertical(|ui| {
                        ui.set_min_width(side_w);
                        ui.set_max_width(side_w);
                        egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(body_h).id_salt("vb-side").show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.strong("Bookmarks");
                                if ui.small_button("+").on_hover_text("Bookmark the current folder").clicked() { add_bookmark = true; }
                            });
                            for bm in &self.variant_browser_bookmarks {
                                let nm = bm.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| bm.to_string_lossy().into_owned());
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(false, format!("★ {nm}")).on_hover_text(bm.to_string_lossy()).clicked() { navigate = Some(bm.clone()); }
                                    if ui.small_button("✖").clicked() { remove_bookmark = Some(bm.clone()); }
                                });
                            }
                            ui.separator();
                            ui.strong("Game");
                            if self.variant_browser_quick.is_empty() {
                                ui.small("no variant folders found");
                            }
                            for (label, path) in &self.variant_browser_quick {
                                if ui.selectable_label(false, label.as_str()).on_hover_text(path.to_string_lossy()).clicked() {
                                    navigate = Some(path.clone());
                                }
                            }
                            ui.separator();
                            ui.strong("System");
                            for (label, path) in &sys_dirs {
                                if ui.selectable_label(false, *label).clicked() { navigate = Some(path.clone()); }
                            }
                            ui.separator();
                            ui.strong("Volumes");
                            for d in &self.variant_browser_drives {
                                if ui.selectable_label(false, format!("💾 {}", d.to_string_lossy())).clicked() { navigate = Some(d.clone()); }
                            }
                            ui.separator();
                            ui.strong("Recent");
                            for r in &self.variant_browser_recent {
                                let nm = r.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| r.to_string_lossy().into_owned());
                                if ui.selectable_label(false, format!("📁 {nm}")).on_hover_text(r.to_string_lossy()).clicked() { navigate = Some(r.clone()); }
                            }
                        });
                    });
                    ui.separator();
                    // ── FILE LIST (Name / Date Modified / Size), pinned width ──
                    ui.vertical(|ui| {
                        ui.set_min_width(main_w);
                        ui.set_max_width(main_w);
                        egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(body_h).id_salt("vb-files").show(ui, |ui| {
                            egui::Grid::new("vb-grid").num_columns(3).striped(true).min_col_width(80.0).show(ui, |ui| {
                                // Clickable sort headers — click to sort, click again to reverse.
                                let arrow = |col: u8| -> &'static str {
                                    if self.variant_browser_sort_col == col { if self.variant_browser_sort_asc { " ^" } else { " v" } } else { "" }
                                };
                                if ui.add(egui::Label::new(egui::RichText::new(format!("Name{}", arrow(0))).strong()).sense(egui::Sense::click())).clicked() { sort_click = Some(0); }
                                if ui.add(egui::Label::new(egui::RichText::new(format!("Date Modified{}", arrow(1))).strong()).sense(egui::Sense::click())).clicked() { sort_click = Some(1); }
                                if ui.add(egui::Label::new(egui::RichText::new(format!("Size{}", arrow(2))).strong()).sense(egui::Sense::click())).clicked() { sort_click = Some(2); }
                                ui.end_row();
                                for entry in &self.variant_browser_listing {
                                    if !entry.is_dir && !f.is_empty() && !entry.name.to_lowercase().contains(&f) && !entry.hint.to_lowercase().contains(&f) {
                                        continue;
                                    }
                                    let icon = if entry.is_dir { "📁" } else { "📄" };
                                    let selected = self.variant_browser_selected.as_ref() == Some(&entry.path);
                                    // Halo 4 rows carry the resolved title etc. after the file name
                                    let row = if entry.hint.is_empty() { format!("{icon} {}", entry.name) } else { format!("{icon} {}  -  {}", entry.name, entry.hint) };
                                    let resp = ui.selectable_label(selected, row);
                                    ui.label(egui::RichText::new(Self::vb_fmt_date(entry.modified)).weak());
                                    ui.label(egui::RichText::new(if entry.is_dir { String::new() } else { Self::vb_fmt_size(entry.size) }).weak());
                                    ui.end_row();
                                    if resp.clicked() {
                                        if entry.is_dir { navigate = Some(entry.path.clone()); }
                                        else { select = Some(entry.path.clone()); }
                                    }
                                    // #dialogs: double-click a folder → navigate (both modes);
                                    // a file → OPEN it (open mode) or fill the name field (save mode).
                                    if resp.double_clicked() && !entry.is_dir {
                                        if save_mode { select = Some(entry.path.clone()); }
                                        else { open_now = Some(entry.path.clone()); }
                                    }
                                }
                            });
                        });
                    });
                });
                ui.separator();
                // ── Details for the selected variant ──
                if let (Some(_sel), Some((title, desc, author, editor, extra))) =
                    (&self.variant_browser_selected, &self.variant_browser_selected_meta)
                {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Title:");
                        ui.strong(if title.is_empty() { "(untitled)" } else { title });
                        ui.separator();
                        ui.label("Author:");
                        ui.label(if author.is_empty() { "—" } else { author });
                        ui.separator();
                        ui.label("Editor:");
                        ui.label(if editor.is_empty() { "—" } else { editor });
                        // Halo 4: base map + object count (empty for Reach files)
                        if !extra.is_empty() {
                            ui.separator();
                            ui.label(extra.as_str());
                        }
                    });
                    ui.label(egui::RichText::new(format!("Description: {}", if desc.is_empty() { "(none)" } else { desc })).italics());
                } else {
                    ui.weak("Select a .mvar to preview its title & description.");
                }
                // #dialogs: inline overwrite confirm (save mode) — never clobber silently.
                if let Some(target) = self.variant_browser_overwrite.clone() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        let nm = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        ui.colored_label(egui::Color32::from_rgb(230, 170, 70), format!("Overwrite {nm}?"));
                        if ui.button("Overwrite").clicked() { confirm_overwrite = true; }
                        if ui.button("Cancel").clicked() { cancel_overwrite = true; }
                    });
                }
                ui.separator();
                // ── Footer: file name field + the primary action + Cancel ──
                ui.horizontal(|ui| {
                    ui.label("File name:");
                    let field_w = (ui.available_width() - 160.0).max(120.0);
                    let resp = ui.add(egui::TextEdit::singleline(&mut self.variant_browser_filename).desired_width(field_w).hint_text("file name"));
                    if save_mode {
                        // Enter in the field is the same as pressing Save.
                        let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let can_save = !self.variant_browser_filename.trim().is_empty();
                        if ui.add_enabled(can_save, egui::Button::new("Save")).clicked() || (entered && can_save) {
                            save_click = true;
                        }
                        if ui.button("Cancel").clicked() { close_win = true; }
                    } else {
                        let sel_file = self.variant_browser_selected.clone();
                        if ui.add_enabled(sel_file.is_some(), egui::Button::new("Open")).clicked() {
                            if let Some(p) = sel_file { open_now = Some(p); }
                        }
                        if ui.button("Cancel").clicked() { close_win = true; }
                    }
                });
            });
        // ── Apply deferred actions ──
        if go_back && self.variant_browser_hist_pos > 0 {
            self.variant_browser_hist_pos -= 1;
            let d = self.variant_browser_history[self.variant_browser_hist_pos].clone();
            self.vb_navigate(&d, false);
        }
        if go_fwd && self.variant_browser_hist_pos + 1 < self.variant_browser_history.len() {
            self.variant_browser_hist_pos += 1;
            let d = self.variant_browser_history[self.variant_browser_hist_pos].clone();
            self.vb_navigate(&d, false);
        }
        if refresh {
            if let Some(d) = self.variant_browser_dir.clone() { self.vb_navigate(&d, false); }
        }
        if add_bookmark {
            if let Some(d) = self.variant_browser_dir.clone() {
                if !self.variant_browser_bookmarks.contains(&d) {
                    self.variant_browser_bookmarks.push(d);
                    self.vb_save_bookmarks();
                }
            }
        }
        if let Some(b) = remove_bookmark {
            self.variant_browser_bookmarks.retain(|p| p != &b);
            self.vb_save_bookmarks();
        }
        if let Some(col) = sort_click {
            if self.variant_browser_sort_col == col {
                self.variant_browser_sort_asc = !self.variant_browser_sort_asc;
            } else {
                self.variant_browser_sort_col = col;
                self.variant_browser_sort_asc = true;
            }
            self.vb_sort_listing();
        }
        if let Some(d) = navigate {
            // #dialogs: in save mode keep the typed file name across a folder change
            // (vb_navigate clears it); an overwrite prompt is folder-specific, so drop it.
            let keep_name = self.variant_browser_save.then(|| self.variant_browser_filename.clone());
            self.vb_navigate(&d, true);
            if let Some(n) = keep_name { self.variant_browser_filename = n; }
            self.variant_browser_overwrite = None;
        }
        if let Some(p) = select {
            // A Halo 4 file is read by its own decoder (the Reach reader mis-decodes it);
            // both games' `$key` titles / descriptions are shown as their English text
            let meta = if let Some(sm) = h4::mvar::read_h4_summary(&p) {
                let map = self.h4_base_map_label(sm.map_id);
                Some((sm.title, sm.description, sm.author, sm.editor, format!("Halo 4: {map}, {} objects", sm.objects)))
            } else {
                mvar::parse_variant(&p).map(|v| (
                    h4::localization::display(&v.title), h4::localization::display(&v.description), v.author.clone(), v.editor.clone(), String::new()))
            };
            self.variant_browser_filename = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            self.variant_browser_selected = Some(p);
            self.variant_browser_selected_meta = meta;
        }
        if let Some(p) = open_now {
            // Close now; import on a later frame (see `pending_open_variant`).
            win_open = false;
            self.pending_open_variant = Some((p, 1));
            ctx.request_repaint();
        }
        // #dialogs: Save pressed — resolve <current dir>/<name> (append .mvar if the user typed
        // no extension). An existing target routes through the inline overwrite confirm first.
        if save_click && self.browser_try_save() { win_open = false; }
        if confirm_overwrite && self.browser_confirm_overwrite() { win_open = false; }
        if cancel_overwrite {
            self.variant_browser_overwrite = None;
        }
        if close_win {
            win_open = false;
        }
        self.variant_browser_open = win_open;
    }

    /// Standard Windows user folders for the browser sidebar (existence not checked — cheap env
    /// lookups; navigating a missing one just shows empty).
    #[cfg(windows)]
    fn vb_system_dirs() -> Vec<(&'static str, std::path::PathBuf)> {
        let mut out = Vec::new();
        if let Ok(up) = std::env::var("USERPROFILE") {
            let home = std::path::PathBuf::from(&up);
            out.push(("🏠 Home", home.clone()));
            for (label, sub) in [("🖥 Desktop", "Desktop"), ("📄 Documents", "Documents"), ("⬇ Downloads", "Downloads"), ("🖼 Pictures", "Pictures"), ("🎬 Videos", "Videos"), ("🎵 Music", "Music")] {
                out.push((label, home.join(sub)));
            }
        }
        out
    }

    /// XDG equivalent. `~/.config/user-dirs.dirs` is honoured so localized or
    /// relocated folders resolve, and entries that do not exist are dropped
    /// rather than shown as dead links (unlike Windows, these are commonly absent).
    #[cfg(not(windows))]
    fn vb_system_dirs() -> Vec<(&'static str, std::path::PathBuf)> {
        let mut out = Vec::new();
        let Ok(home) = std::env::var("HOME") else { return out };
        let home = std::path::PathBuf::from(home);
        out.push(("🏠 Home", home.clone()));

        // XDG_*_DIR entries look like: XDG_DOWNLOAD_DIR="$HOME/Downloads"
        let mut xdg: std::collections::HashMap<String, std::path::PathBuf> = Default::default();
        if let Ok(txt) = std::fs::read_to_string(home.join(".config/user-dirs.dirs")) {
            for line in txt.lines() {
                let line = line.trim();
                if line.starts_with('#') { continue; }
                let Some((k, v)) = line.split_once('=') else { continue };
                let v = v.trim().trim_matches('"');
                let path = if let Some(rest) = v.strip_prefix("$HOME") {
                    home.join(rest.trim_start_matches('/'))
                } else {
                    std::path::PathBuf::from(v)
                };
                xdg.insert(k.trim().to_string(), path);
            }
        }
        for (label, key, fallback) in [
            ("🖥 Desktop",   "XDG_DESKTOP_DIR",   "Desktop"),
            ("📄 Documents", "XDG_DOCUMENTS_DIR", "Documents"),
            ("⬇ Downloads",  "XDG_DOWNLOAD_DIR",  "Downloads"),
            ("🖼 Pictures",  "XDG_PICTURES_DIR",  "Pictures"),
            ("🎬 Videos",    "XDG_VIDEOS_DIR",    "Videos"),
            ("🎵 Music",     "XDG_MUSIC_DIR",     "Music"),
        ] {
            let path = xdg.get(key).cloned().unwrap_or_else(|| home.join(fallback));
            if path.is_dir() { out.push((label, path)); }
        }
        out
    }

    /// Sidebar "Volumes": drive letters on Windows, real mount points on Unix.
    #[cfg(windows)]
    fn vb_volumes() -> Vec<std::path::PathBuf> {
        // Skip A:/B: — stat-ing a floppy can hang.
        (b'C'..=b'Z')
            .map(|c| std::path::PathBuf::from(format!("{}:\\", c as char)))
            .filter(|p| p.exists())
            .collect()
    }

    /// Unix has no drive letters; the equivalent is `/` plus whatever is mounted
    /// under the usual media roots (this is where extra disks and USB sticks land).
    #[cfg(not(windows))]
    fn vb_volumes() -> Vec<std::path::PathBuf> {
        let mut out = vec![std::path::PathBuf::from("/")];
        if let Ok(mounts) = std::fs::read_to_string("/proc/mounts") {
            for line in mounts.lines() {
                let mut it = line.split_whitespace();
                let (_dev, mnt) = (it.next(), it.next());
                let Some(mnt) = mnt else { continue };
                // /proc/mounts octal-escapes spaces and friends.
                let mnt = mnt.replace("\\040", " ");
                if mnt.starts_with("/mnt/") || mnt.starts_with("/media/") || mnt.starts_with("/run/media/") {
                    let p = std::path::PathBuf::from(&mnt);
                    if p.is_dir() && !out.contains(&p) { out.push(p); }
                }
            }
        }
        out
    }

    /// The top menu bar — File / Edit / View / Tools / Settings / Help.
    fn menu_bar_ui(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New variant").on_hover_text("Start an empty variant for the loaded map: set its name/description under Variant in the left panel, place objects, then Save (asks for a file name)").clicked() {
                    self.new_variant();
                    ui.close_menu();
                }
                if ui.button("Open .mvar…").on_hover_text("Browse folders and open a variant — previews each variant's name & description before you open it").clicked() {
                    self.open_variant_browser();
                    ui.close_menu();
                }
                ui.separator();
                let has = self.current_variant_path.is_some() || self.new_variant_template.is_some();
                // Both games get Save (in place) and Save As.
                if ui.add_enabled(has, egui::Button::new("Save")).on_hover_text("Write the edited objects back into the open .mvar (a new variant asks for a file name)").clicked() {
                    self.save_current_variant();
                    ui.close_menu();
                }
                if ui.add_enabled(has, egui::Button::new("Save As…")).clicked() {
                    self.save_variant_as();
                    ui.close_menu();
                }
                ui.separator();
                // A NAMED project you can navigate to, plus the one-slot quick save.
                if ui.button("Open Project…").clicked() {
                    self.open_project_dialog();
                    ui.close_menu();
                }
                if ui.button("Save Project As…").clicked() {
                    self.save_project_as();
                    ui.close_menu();
                }
                if let Some(p) = self.current_project_path.clone() {
                    let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    if ui.button(format!("Save Project ({name})")).clicked() {
                        let rec = self.build_project_record();
                        self.status = match project::save_to(&rec, &p) {
                            Ok(path) => format!("Saved project -> {}", path.display()),
                            Err(e) => format!("Save project failed: {e}"),
                        };
                        ui.close_menu();
                    }
                }
                ui.separator();
                if ui.button("Quick-save session")
                    .on_hover_text("Save to the single auto slot in the settings folder — no dialog. Use Save Project As… to keep more than one.")
                    .clicked()
                {
                    self.save_project();
                    ui.close_menu();
                }
                if ui.button("Restore last session")
                    .on_hover_text("Reload that auto slot.")
                    .clicked()
                {
                    self.load_project();
                    ui.close_menu();
                }
            });
            // Edit menu = the selection verbs that also have hotkeys, so they are
            // discoverable without the hotkey window.
            ui.menu_button("Edit", |ui| {
                let can_undo = !self.edit_undo.is_empty() || !self.undo_stack.is_empty();
                let can_redo = !self.edit_redo.is_empty() || !self.redo_stack.is_empty();
                if ui.add_enabled(can_undo, egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                    self.do_undo();
                    ui.close_menu();
                }
                if ui.add_enabled(can_redo, egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                    self.do_redo();
                    ui.close_menu();
                }
                ui.separator();
                let n_objs = self.mvar_objects.len() + self.local_objects.len();
                if ui.add_enabled(n_objs > 0, egui::Button::new("Select all objects").shortcut_text("Ctrl+A")).clicked() {
                    self.select_all_objects();
                    ui.close_menu();
                }
                let has_sel = !self.selected_set.is_empty();
                if ui.add_enabled(has_sel, egui::Button::new("Deselect")).clicked() {
                    self.selected_set.clear();
                    self.selected_datum = None;
                    self.apply_selection_highlight();
                    ui.close_menu();
                }
                if ui.add_enabled(has_sel, egui::Button::new("Delete selected").shortcut_text("Del")).clicked() {
                    self.delete_selection();
                    ui.close_menu();
                }
                ui.separator();
                if ui.add_enabled(n_objs > 0, egui::Button::new("Remove all placed objects"))
                    .on_hover_text("Clear every variant / placed object from the scene (undoable). The base map stays.")
                    .clicked()
                {
                    self.push_edit_undo();
                    self.mvar_objects.clear();
                    self.local_objects.clear();
                    self.mvar_meta.clear();
                    self.mvar_colors.clear();
                    self.selected_set.clear();
                    self.selected_datum = None;
                    self.next_local_datum = 0xF000_0000;
                    if let Some(s) = self.objscene_mut() {
                        s.invalidate();
                    }
                    self.apply_selection_highlight();
                    self.spawn_status = "Removed all placed objects.".into();
                    ui.close_menu();
                }
            });
            ui.menu_button("View", |ui| {
                let has_sel = self.selected_datum.is_some();
                if ui.add_enabled(has_sel, egui::Button::new("Frame selected object").shortcut_text("F"))
                    .on_hover_text("Fly the camera to the selected object")
                    .clicked()
                {
                    self.frame_selected();
                    ui.close_menu();
                }
                if ui.button("Screenshot").on_hover_text("Save a PNG of the viewport (F12)").clicked() {
                    self.screenshot();
                    ui.close_menu();
                }
                ui.separator();
                let mut show_construct = self.tool_mode == ToolMode::Construct;
                if ui.checkbox(&mut show_construct, "Construction (CAD) panel").on_hover_text("Shown while the Construct tool is active (C)").changed() {
                    self.tool_mode = if show_construct { ToolMode::Construct } else { ToolMode::Select };
                }
                ui.checkbox(&mut self.show_script, "Script console").on_hover_text("Forge Script: type commands (place / set / rotate / camera ...) and run them");
                ui.separator();
                // #view-menu: the scene's own lanes. These were in Settings > Camera, render &
                // display; every "what do I see" switch belongs in one place, and this is it.
                ui.label(egui::RichText::new("Geometry").weak());
                ui.checkbox(&mut self.renderer.show_bsp, "BSP")
                    .on_hover_text("Draw the map's structure geometry (the level itself).");
                ui.checkbox(&mut self.renderer.show_terrain, "Terrain")
                    .on_hover_text("Draw the map's terrain instances.");
                ui.checkbox(&mut self.renderer.show_objects, "Forge objects")
                    .on_hover_text("Draw the variant's placed Forge objects.");
                ui.checkbox(&mut self.renderer.show_water, "Water")
                    .on_hover_text("Draw the map's water surfaces.");
                ui.checkbox(&mut self.renderer.show_sky, "Sky")
                    .on_hover_text("Draw the map's sky.");
                ui.separator();
                ui.label(egui::RichText::new("Forge extras").weak());
                // The placed special-FX orbs' screen effects (colour grade). Off = the map's
                // own default screen effect only. Persisted; the script `screenfx on|off` sets the same.
                if ui.checkbox(&mut self.forge_fx_enabled, "Forge special FX")
                    .on_hover_text("Apply the screen effects of placed Forge special-FX objects (Colorblind, Gloomy, Juicy, Nova, Olde Timey, Pen and Ink, Dusk, Eerie, Golden Hour). Untick to screenshot without them; the map's own default effect stays.")
                    .changed()
                {
                    screenfx::save_enabled_setting(self.forge_fx_enabled);
                }
                // The two GLOBAL switches over the per-object pseudo-flags. Persisted
                // settings (Scripts: `scale on|off`, `shadowcasters on|off`; batch env HMS_SCALED /
                // HMS_SHADOWCASTERS). The rebuild tick reads obj_globals every frame, so a change
                // re-signatures the scene and the viewport follows without a refresh.
                {
                    let mut changed = false;
                    changed |= ui.checkbox(&mut self.obj_globals.scaled, "Scaled objects")
                        .on_hover_text("Objects whose SCALED flag is on (default: the forge \"scale\" label) render at the size their spawn sequence encodes. Off = every object at ×1, whatever its flag.")
                        .changed();
                    changed |= ui.checkbox(&mut self.obj_globals.shadowcasters, "Shadow casters (Forge)")
                        .on_hover_text("Objects whose SHADOW flag is on (default: GREEN team + \"scale\" label — the shadow-casting gametype rule) are drawn into the sun shadow map. Off = only the engine's default casters (vehicles, weapons…) cast.")
                        .changed();
                    if changed {
                        self.obj_globals.save();
                    }
                    let (mut n_scaled, mut n_cast) = (0usize, 0usize);
                    for m in self.mvar_meta.values() {
                        let (fs, fc) = forge_scale::effective_flags(&self.obj_globals, &m.flags, m.team, &m.label);
                        n_scaled += fs as usize;
                        n_cast += fc as usize;
                    }
                    ui.small(egui::RichText::new(format!("{n_scaled} scaled, {n_cast} casting (Forge flags; engine-default casters not counted)")).weak());
                }
                ui.separator();
                ui.label(egui::RichText::new("Overlays").weak());
                ui.checkbox(&mut self.renderer.show_grid, "Grid")
                    .on_hover_text("A reference grid through the origin.");
                // Marker-only: a star per lightmapper light (plus a spot's aim line and its
                // far-attenuation ring). It does NOT change how the scene is lit -- it also
                // reveals the light-moving controls under Settings > real-time probe GI.
                if ui.checkbox(&mut self.show_lights, "Lights").on_hover_text("Draw the map's lightmapper lights (star + range ring; spots also show their aim). Off by default.").changed() {
                    self.rebuild_overlays();
                }
                if ui.checkbox(&mut self.show_boundaries, "Boundary shapes").on_hover_text("Sphere/cylinder/box zones for teleporters & objectives").changed() {
                    self.rebuild_overlays();
                }
                if ui.checkbox(&mut self.show_collision, "Collision (selected)").changed() {
                    self.rebuild_overlays();
                }
                if ui.checkbox(&mut self.show_physics, "Physics (selected)").changed() {
                    self.rebuild_overlays();
                }
                // The MAP's own spawn markers (scenario placements) are hidden by
                // default; the variant's spawns are the user's and always show. Persisted.
                let n_spawns = if self.h4_active { self.h4_map_spawn_markers } else { self.scenario_objects.iter().filter(|o| self.scenario_spawn_tags.contains(&o.primary_tag)).count() };
                if ui.checkbox(&mut self.show_map_spawns, "Show map spawn points")
                    .on_hover_text(format!("Draw the base map's own built-in spawn markers ({n_spawns} on this map). Off by default: they are not part of your variant and cannot be edited or saved. Your placed spawn points always show."))
                    .changed()
                {
                    map_spawns::save_setting(self.show_map_spawns);
                    self.renderer.set_show_markers(self.show_map_spawns);
                }
                // The hull overlay (orange volume + outline) of every forge-placed
                // HIDDEN block -- the only way to see the invisible blocks a variant placed, so ON
                // by default. Never built for scenario objects or BSP. Persisted.
                let n_tris = self.objscene().map(|s| s.blocker_overlay().0.len() / 3).unwrap_or(0);
                if ui.checkbox(&mut self.show_blockers, "Show hidden-block physics hulls")
                    .on_hover_text(format!("Draw the physics hull (orange volume + outline) of every hidden block placed by the variant -- invisible walls such as the nut_blockers a shipped variant seals the map with ({n_tris} hull triangles on this map). On by default; untick for the game's look. Map (scenario) objects and the BSP never get a hull; the selected object's wireframe always shows."))
                    .changed()
                {
                    physics_outlines::save_setting(self.show_blockers);
                    self.overlays_dirty = true;
                }
                // #wire-visible  The selection wireframe's depth mode. Persisted.
                let wx_status = self.wire_xray_status();
                if ui.checkbox(&mut self.wire_xray, "Selection wireframe through objects")
                    .on_hover_text(format!("Draw the selected object's wireframe THROUGH anything in front of it, so a piece buried in the terrain or behind a wall still shows its whole outline. Off by default (the wireframe is hidden where geometry covers it). The wireframe itself is always drawn with a dark outline under a bright core, so it stays readable on white Forge pieces and in dark interiors alike. Script: wirexray on|off|get.\n{wx_status}"))
                    .changed()
                {
                    wire_xray::save_setting(self.wire_xray);
                    self.renderer.set_highlight_xray(self.wire_xray);
                }
                // The scenario's trigger volumes (kill/safe/plain boxes). Not persisted.
                if ui.checkbox(&mut self.show_triggers, "Trigger volumes")
                    .on_hover_text("Draw the scenario's trigger volumes as wireframe boxes: red = kill, green = safe, amber = plain. Script: triggers on|off|get.")
                    .changed()
                {
                    self.overlays_dirty = true;
                }
                // The invisible planes bounding the playable volume. Persisted.
                let sc_status = self.soft_ceilings_status();
                if ui.checkbox(&mut self.show_soft_ceilings, "Soft ceilings (map floor)")
                    .on_hover_text(format!("Draw the map's soft ceilings -- the invisible planes that bound the playable space. Red = soft kill (the floor under the map and the outer walls), orange = acceleration (the sky ceiling that pushes you back), yellow = slip surface. Off by default. Script: softceilings on|off|get.\n{sc_status}"))
                    .changed()
                {
                    soft_ceilings::save_setting(self.show_soft_ceilings);
                    self.overlays_dirty = true;
                }
                // The playable BSPs' world bounds -- the last height anything solid
                // exists at (the sea floor under Forge World). Persisted.
                let hf_status = self.hard_floor_status();
                if ui.checkbox(&mut self.show_hard_floor, "Hard floor (world bounds)")
                    .on_hover_text(format!("Draw the map's world box (magenta): the Havok broadphase the engine builds from EVERY structure BSP's physics bounds (sky and hidden BSPs included) plus 64 wu -- the invisible wall nothing can pass, with its floor gridded at the box's lowest z. That floor is where you stop when the soft ceiling is off (Forge World: -75). Off by default. Script: hardfloor on|off|get.\n{hf_status}"))
                    .changed()
                {
                    hard_floor::save_setting(self.show_hard_floor);
                    self.overlays_dirty = true;
                }
                let pb_status = self.playable_bounds_status();
                if ui.checkbox(&mut self.show_playable_bounds, "Playable BSP bounds")
                    .on_hover_text(format!("Draw each playable structure BSP's own world bounds (violet box + floor): the smaller box the game clamps object positions to; below its floor nothing solid exists (Forge World: -25, the sea floor). Off by default. Script: playablebounds on|off|get.\n{pb_status}"))
                    .changed()
                {
                    hard_floor::save_playable_setting(self.show_playable_bounds);
                    self.overlays_dirty = true;
                }
            });
            ui.menu_button("Tools", |ui| {
                // The Line / Fill placement tools. Both work offline: pick a palette item, turn
                // the tool on, then click in the viewport.
                ui.label(egui::RichText::new("Placement tools (pick a palette item first)").weak());
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut self.line_mode, "Line tool")
                        .on_hover_text("Click a start point, then an end point: places evenly spaced copies of the selected palette item along the line")
                        .changed()
                    {
                        self.line_start = None;
                        if self.line_mode {
                            self.fill_mode = false;
                            self.fill_points.clear();
                        }
                    }
                    ui.add(egui::DragValue::new(&mut self.line_count).range(2..=64).prefix("count "));
                });
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut self.fill_mode, "Fill tool")
                        .on_hover_text("Click 3+ outline points in the viewport, then Close & fill: grid-fills the polygon with the selected palette item")
                        .changed()
                    {
                        self.fill_points.clear();
                        if self.fill_mode {
                            self.line_mode = false;
                            self.line_start = None;
                        }
                    }
                    ui.add(egui::DragValue::new(&mut self.fill_spacing).range(0.5..=20.0).prefix("gap "));
                    let n_pts = self.fill_points.len();
                    if ui.add_enabled(n_pts >= 3, egui::Button::new(format!("Close & fill ({n_pts} pts)"))).clicked() {
                        self.fill_polygon();
                        ui.close_menu();
                    }
                });
                ui.separator();
                if ui.button("Scale converter…")
                    .on_hover_text("Convert between a visual scale multiplier and the spawnSequence encoding, for a chosen Forge scale convention")
                    .clicked()
                {
                    self.show_scale_converter = true;
                    ui.close_menu();
                }
                if ui.button("Light bake map…")
                    .on_hover_text("Top-down map of the baked light colour a Forge piece receives at every spot; pick a shade and move the selected object to the closest-matching cell")
                    .clicked()
                {
                    self.show_lightbake = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.add(egui::Button::new("Construction (CAD)…").shortcut_text("C"))
                    .on_hover_text("Draw dotted construction guides between object anchors and snap to where they cross — the exact way to centre or mirror something")
                    .clicked()
                {
                    self.tool_mode = ToolMode::Construct;
                    ui.close_menu();
                }
                if ui.button("Model Painter…")
                    .on_hover_text("Voxelise an .obj and spawn a forge block per surface voxel")
                    .clicked()
                {
                    self.show_model_painter = true;
                    ui.close_menu();
                }
                if ui.button("Import geometry…")
                    .on_hover_text("Load an OBJ/GLB as a persistent base to forge over")
                    .clicked()
                {
                    self.show_import_geom = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Script console…")
                    .on_hover_text("Forge Script: type commands (place / set / rotate / camera ...) and run them")
                    .clicked()
                {
                    self.show_script = true;
                    ui.close_menu();
                }
            });
            ui.menu_button("Settings", |ui| {
                if ui.button("Lighting & rendering…")
                    .on_hover_text("Exposure, bloom and fog sliders, real-time probe GI, the path-traced bake, camera and every render/display toggle. Diagnostics are under Advanced at the bottom.")
                    .clicked()
                {
                    self.show_settings = true;
                    ui.close_menu();
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("Keyboard shortcuts…").clicked() {
                    self.show_hotkeys = true;
                    ui.close_menu();
                }
            });
            if let Some(p) = &self.current_variant_path {
                ui.separator();
                ui.small(p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
            }
        });
    }

    /// Recompose the screen effect for the given object set (scenario default + the
    /// distinct sefc tags the placed Forge special-FX objects spawn, per-term max like the engine)
    /// and upload it when (enabled, default, tags) changed. Cheap enough to call every tick.
    fn refresh_screen_fx(&mut self, objects: &[ObjectInfo]) {
        let Some(scene) = self.scene_ctl.as_ref() else { return };
        let default_tag = scene.screen_fx().map(|f| f.tag).unwrap_or(0);
        let active = screenfx::active_screen_fx(scene, &mut self.screenfx_cache, default_tag, objects.iter().map(|o| (o.datum, o.primary_tag)));
        let key = (self.forge_fx_enabled, default_tag, active.forge_tags.clone());
        if self.screenfx_pushed.as_ref() != Some(&key) {
            let line = screenfx::push_to_renderer(&self.renderer, &self.render_state.queue, &active, self.forge_fx_enabled);
            log::info!("{line}");
            self.screenfx_pushed = Some(key);
        }
        self.active_screenfx = active;
    }

    /// User-facing summary for the status bar / script `screenfx get`.
    fn screenfx_status(&self) -> String {
        let a = &self.active_screenfx;
        if a.forge_names.is_empty() {
            format!("Forge FX: none placed ({})", if self.forge_fx_enabled { "on" } else { "off" })
        } else {
            format!("Forge FX{}: {}", if self.forge_fx_enabled { "" } else { " (off)" }, a.forge_names.join(" + "))
        }
    }

    /// A camera pose from the loaded variant's SPAWN points — preferring INITIAL
    /// spawns, then any spawn point. None when the variant places no spawn points (caller
    /// then falls back to the scenario's built-in spawns / bounds).
    fn variant_spawn_camera(&self) -> Option<(glam::Vec3, f32, f32)> {
        // A Halo 4 variant's camera comes from the H4 scene's `spawn_camera_pose`
        // (loadout camera / initial spawn by team, stood off the geometry) - the callers fall
        // through to it.
        if self.h4_active { return None; }
        let name_of = |datum: u32| {
            self.mvar_meta.get(&datum).map(|m| m.name.to_lowercase()).unwrap_or_default()
        };
        let mut initial: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        let mut any_spawn: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        for o in &self.mvar_objects {
            let n = name_of(o.datum);
            if !n.contains("spawn") {
                continue;
            }
            let entry = (glam::Vec3::from(o.pos), glam::Vec3::from(o.fwd));
            if n.contains("initial") {
                initial.push(entry);
            }
            any_spawn.push(entry);
        }
        let picks = if !initial.is_empty() {
            initial
        } else if !any_spawn.is_empty() {
            any_spawn
        } else {
            return None;
        };
        // Centroid of ALL variant objects → a sensible facing when a spawn has no forward.
        let all: Vec<glam::Vec3> = self.mvar_objects.iter().map(|o| glam::Vec3::from(o.pos)).collect();
        let centroid = if all.is_empty() {
            picks[0].0
        } else {
            all.iter().fold(glam::Vec3::ZERO, |a, &p| a + p) / (all.len() as f32)
        };
        let (spawn, fwd) = *picks
            .iter()
            .min_by(|a, b| (a.0 - centroid).length_squared().total_cmp(&(b.0 - centroid).length_squared()))
            .unwrap();
        // STAND at the spawn: ~player eye height (Reach world units, player ≈0.7wu tall), not
        // floating metres above it. Look level along the spawn's forward.
        let eye = spawn + glam::Vec3::Z * 0.62;
        let yaw = if fwd.x.hypot(fwd.y) > 0.1 {
            fwd.y.atan2(fwd.x)
        } else {
            let to = centroid - eye;
            if to.x.hypot(to.y) > 1.0 { to.y.atan2(to.x) } else { 0.0 }
        };
        Some((eye, yaw, 0.0))
    }

    /// Resolve a parsed variant's placements → ObjectInfo through the loaded map's sandbox
    /// palette and store them in `mvar_objects` (merged into the render). Shared by the
    /// direct path and the deferred post-load path.
    fn render_variant_objects(&mut self, variant: &mvar::Variant, name: &str) -> usize {
        // Seed the editable header fields from the loaded variant (fresh load = not yet edited).
        self.variant_title = variant.title.clone();
        self.variant_description = variant.description.clone();
        self.variant_author = variant.author.clone();
        self.variant_editor = variant.editor.clone();
        self.variant_header_dirty = false;
        self.seed_variant_globals(variant.globals.clone());
        // Loading a variant replaces the placed set — drop any manually-spawned
        // local objects + stale selection from the previous variant so a same-map variant swap is a
        // clean slate too (mvar_objects itself is replaced below). Scenario/BSP stay (same map).
        // The selection wireframe buffer goes with it (before the scene borrow).
        self.local_objects.clear();
        self.clear_selection_state();
        let Some(scene) = self.scene_ctl.as_ref() else {
            self.spawn_status = format!("Parsed {} objects — load a base map first to render them.", variant.objects.len());
            return 0;
        };
        let t0 = std::time::Instant::now();
        let palette = scene.forge_palette_full();
        let types = scene.forge_type_order(&palette);
        let mut out = Vec::with_capacity(variant.objects.len());
        let mut colors: std::collections::HashMap<u32, (u8, u8)> = std::collections::HashMap::new();
        let mut meta: std::collections::HashMap<u32, ObjMeta> = std::collections::HashMap::new();
        let mut tags: Vec<u32> = Vec::new();
        let mut unresolved = 0usize;
        let mut unresolved_idx: std::collections::HashSet<usize> = Default::default();
        let mut unresolved_list: Vec<(usize, u16, u8)> = Vec::new();
        for (i, o) in variant.objects.iter().enumerate() {
            let (mode_tag, obj_tag, variant_sid) =
                scene.resolve_forge_model_v(&palette, &types, o.folder, o.item);
            if mode_tag == 0 || mode_tag == 0xFFFF_FFFF {
                unresolved += 1;
                unresolved_idx.insert(i); // this slot is occupied but invisible
                // Record WHAT could not be resolved. "N objects are invisible" is not actionable;
                // the (folder,item) pair plus the palette group at that folder index is.
                unresolved_list.push((i, o.folder, o.item));
                continue;
            }
            tags.push(mode_tag);
            let datum = 0xD000_0000u32.wrapping_add(i as u32);
            // Record the parsed (team, color) for this object. color -1 (none) → 0xFF ("use
            // team"), matching forge_tint's convention. This is what makes team/object change-colour
            // reach the render for OFFLINE .mvar objects.
            let color_u8 = if o.color < 0 { 0xFFu8 } else { o.color as u8 };
            colors.insert(datum, (o.team, color_u8));
            if std::env::var("HMS_CCDIAG").is_ok() {
                let onm = types.get(o.folder as usize).and_then(|&(pi, ew)| {
                    palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.clone())
                }).unwrap_or_default();
                if onm.contains("spawn") || onm.contains("flag") || onm.contains("hill") || onm.contains("boundaries") {
                    eprintln!("CCDIAG {} team={} color={}", onm, o.team, o.color);
                }
            }
            // Capture the object's palette name + all parsed .mvar fields for the UI.
            // Show the VARIANT (the actual piece, e.g. "wall_coliseum") in the properties panel,
            // not the palette group ("structure_doors"/"structure_bridges"); fall back to the group name
            // for entries without variants.
            let obj_name = types.get(o.folder as usize).and_then(|&(pi, ew)| {
                let v = o.item as u32;
                palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew && e.variant_within == v)
                    .or_else(|| palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew))
                    .map(|e| if e.variant_name.trim().is_empty() { e.name.clone() } else { e.variant_name.clone() })
            }).unwrap_or_default();
            meta.insert(datum, ObjMeta {
                name: obj_name,
                folder: o.folder,
                item: o.item,
                pos: o.pos,
                team: o.team,
                color: o.color,
                cached_type: o.cached_type,
                spawn_seq: o.spawn_seq,
                respawn: o.respawn,
                label_idx: o.label_idx,
                label: if o.label_idx != 0xFFFF {
                    variant.labels.get(o.label_idx as usize).cloned().unwrap_or_default()
                } else {
                    String::new()
                },
                placement: o.placement,
                boundary_shape: o.boundary_shape,
                boundary: o.boundary,
                weapon_clips: o.weapon_clips,
                tele_channel: o.tele_channel,
                tele_passability: o.tele_passability,
                location_name: o.location_name,
                spawn_rel: o.spawn_rel,
                slot: o.slot,
                flags: Default::default(), // re-derived; project overrides re-applied below
                h4: None, // (Reach variant)
            });
            out.push(ObjectInfo {
                datum,
                type_sig: 0,
                sig0: 0,
                sig1: 0,
                pos: o.pos,
                health: 1.0,
                shield: 1.0,
                mode_tag,
                fwd: o.fwd,
                up: o.up,
                attached: [0; 8],
                primary_tag: obj_tag,
                variant_name_sid: variant_sid,
            });
        }
        let placed = out.len();
        // Diagnostic dump so a "nothing rendered" case is debuggable without a console:
        // palette size + coordinate ranges, and the first objects' (folder,item) vs what
        // the palette actually contains. Written next to the exe as hms_mvar_diag.txt.
        {
            use std::fmt::Write as _;
            let mut d = String::new();
            let _ = writeln!(d, "variant={name}  map_id={}  objects={}  placed={placed}  unresolved={unresolved}", variant.map_id, variant.objects.len());
            let _ = writeln!(d, "variant quota count: {}  <-- should equal distinct object types", variant.num_quotas);
            let _ = writeln!(d, "palette entries: {}  distinct object types: {}", palette.len(), types.len());
            if variant.num_quotas as usize != types.len() {
                let _ = writeln!(d, "*** MISMATCH: quota count {} != distinct types {} — positional resolution will be SHIFTED ***", variant.num_quotas, types.len());
            }
            let max_folder = variant.objects.iter().map(|o| o.folder).max().unwrap_or(0);
            let _ = writeln!(d, "max object folder (quota index) used: {max_folder}");
            let _ = writeln!(d, "-- first object types (quota index -> palette_index, entry_within, name) --");
            for (q, &(pi, ew)) in types.iter().enumerate().take(24) {
                let nm = palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.as_str()).unwrap_or("");
                let _ = writeln!(d, "  [{q}] ({pi},{ew}) {nm}");
            }
            // The objects HMS could NOT resolve. These still occupy a slot in the
            // variant and still count against the 651-object limit; the save carries them through.
            let _ = writeln!(d, "-- UNRESOLVED objects: {} (occupy slots, not displayed) --", unresolved_list.len());
            for &(i, folder, item) in unresolved_list.iter().take(64) {
                let group = types.get(folder as usize).and_then(|&(pi, ew)| {
                    palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.clone())
                }).unwrap_or_else(|| "<folder index past the end of this map's palette>".into());
                let _ = writeln!(d, "  slot#{i} folder={folder} item={item}  {group}");
            }
            let _ = writeln!(d, "-- first variant objects (folder, item) + resolve --");
            for o in variant.objects.iter().take(32) {
                let (mt, ot) = scene.resolve_forge_model(&palette, &types, o.folder, o.item);
                let nm = types.get(o.folder as usize).and_then(|&(pi, ew)| {
                    palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.clone())
                }).unwrap_or_default();
                let _ = writeln!(d, "  folder={} item={} -> obj=0x{:08X} mode=0x{:08X}  {}", o.folder, o.item, ot, mt, nm);
            }
            if let Some(p) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("hms_mvar_diag.txt"))) {
                let _ = std::fs::write(&p, &d);
            }
        }
        let empty_palette = palette.is_empty();
        // NOTE: no synchronous decode here (it would freeze the UI for the whole variant load).
        // The models decode on a background thread (parallel), streaming finished models into
        // the scene cache; maybe_rebuild's per-frame budget (tick_scene) then uploads a few per
        // frame, so objects pop in over the next ~1-2 s while the window stays responsive.
        if let Some(scene) = self.scene_ctl.as_mut() {
            scene.start_background_predecode(&tags);
        }
        let build_ms = t0.elapsed().as_millis();
        // Opening a variant also refreshes the master palette. Re-reading the live palette MMF
        // costs nothing when it is absent (offline builds just keep an empty list).
        if self.palette_client.is_none() {
            self.palette_client = ForgePaletteClient::open().ok();
        }
        if let Some(c) = &self.palette_client {
            self.palette = c.read();
        }
        self.mvar_objects = out;
        self.mvar_colors = colors; // team/object change-colour for the offline objects
        self.mvar_meta = meta; // full parsed object data for the properties panel
        self.apply_pending_obj_flags(); // project overrides onto the fresh meta
        self.mvar_unresolved = unresolved_idx;
        // Remember each object's SOURCE record against its datum, and keep the
        // records of the ones we cannot display, so a save never has to guess which slot an
        // object came from.
        self.mvar_src = variant.objects.iter().enumerate()
            .map(|(i, o)| (0xD000_0000u32.wrapping_add(i as u32), o.clone()))
            .collect();
        self.mvar_unresolved_objs = self.mvar_unresolved.iter()
            .filter_map(|i| variant.objects.get(*i).cloned())
            .collect();
        self.mvar_labels = variant.labels.clone(); // label table for the properties label dropdown
        // On a successful variant load, teleport to a spawn — the variant's own
        // spawn points first (initial spawns preferred), then the scenario's built-in spawns
        // (biped placements), then the playable-area centre. Skipped when HMS_CAM pins a view.
        if placed > 0 && std::env::var("HMS_CAM").is_err() {
            let pose = self
                .variant_spawn_camera()
                .or_else(|| self.objscene().and_then(|s| s.spawn_camera_pose()));
            if let Some((pos, yaw, pitch)) = pose {
                self.camera.pos = pos;
                self.camera.yaw = yaw;
                self.camera.pitch = pitch;
            } else if let Some((mn, mx)) = self.objscene().and_then(|s| s.scene_bounds()) {
                let c = (glam::Vec3::from(mn) + glam::Vec3::from(mx)) * 0.5;
                self.camera.pos = c + glam::Vec3::new(-10.0, -10.0, 6.0);
            }
        }
        self.spawn_status = if empty_palette {
            format!("{name}: parsed {} objects but the loaded map has no forge palette (load a Forge canvas map).", variant.objects.len())
        } else if placed == 0 {
            format!("{name}: 0/{} objects resolved against this map's palette — see hms_mvar_diag.txt.", variant.objects.len())
        } else {
            format!(
                "{name}: {placed} forge objects ({unresolved} unresolved — they still occupy slots; see hms_mvar_diag.txt) in {build_ms} ms."
            )
        };
        // Draw any boundary shapes (teleporter/zone volumes) the variant defines.
        if placed > 0 {
            self.rebuild_overlays();
        }
        placed
    }

    /// Import the selected OBJ/GLB geometry (converted from another game's BSP) as a
    /// persistent base mesh to forge over. Appends so multiple can be loaded.
    fn do_import_geometry(&mut self) {
        let Some(path) = self.selected_import.and_then(|i| self.model_candidates.get(i)).cloned() else {
            self.spawn_status = "Select an .obj/.glb (put files in the exe's models\\ folder).".into();
            return;
        };
        match import_geometry_mesh(self.renderer.mesh_renderer(), &self.render_state.device, &self.render_state.queue, &path) {
            Some((m, mut tris)) => {
                self.renderer.append_imported_meshes(vec![m]);
                self.import_tris.append(&mut tris);
                self.spawn_status = format!("Imported base geometry: {}", path.file_name().unwrap_or_default().to_string_lossy());
            }
            None => self.spawn_status = "Failed to load geometry (OBJ/GLB).".into(),
        }
    }

    /// Model Painter: voxelize the selected .obj and spawn a forge block per
    /// surface voxel, centred on the point 20u ahead of the camera.
    fn paint_model(&mut self) {
        let Some(path) = self.selected_model.and_then(|i| self.model_candidates.get(i)).cloned() else {
            self.spawn_status = "Select a model (put .obj files in the exe's models\\ folder).".into();
            return;
        };
        let Some(voxels) = voxel::load_and_voxelize(&path, self.paint_res as usize) else {
            self.spawn_status = "Model parse/voxelize failed.".into();
            return;
        };
        let anchor = self.camera.pos + self.camera.forward() * 20.0;
        let s = self.paint_scale;
        let n = voxels.len();
        for v in voxels {
            let world = anchor + glam::Vec3::new((v[0] - 0.5) * s, (v[1] - 0.5) * s, (v[2] - 0.5) * s);
            self.place_dispatch(world);
        }
        self.spawn_status = format!("Painted {n} blocks from {}", path.file_name().unwrap_or_default().to_string_lossy());
    }

    /// Fill tool: grid-fill the outlined polygon (XY) with the selected item.
    fn fill_polygon(&mut self) {
        if self.fill_points.len() < 3 {
            self.spawn_status = "Fill needs 3+ points.".into();
            return;
        }
        let poly: Vec<[f32; 2]> = self.fill_points.iter().map(|p| [p.x, p.y]).collect();
        let z = self.fill_points.iter().map(|p| p.z).sum::<f32>() / self.fill_points.len() as f32;
        let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in &poly {
            minx = minx.min(p[0]);
            maxx = maxx.max(p[0]);
            miny = miny.min(p[1]);
            maxy = maxy.max(p[1]);
        }
        let sp = self.fill_spacing.max(0.5);
        let mut count = 0u32;
        let mut y = miny;
        while y <= maxy && count < 2000 {
            let mut x = minx;
            while x <= maxx && count < 2000 {
                if point_in_poly([x, y], &poly) {
                    // Snap each fill point down onto the imported walkable surface;
                    // fall back to the outline's average Z when there's no geometry below.
                    let origin = glam::Vec3::new(x, y, z + 1000.0);
                    let gz = raycast_tris(origin, glam::Vec3::new(0.0, 0.0, -1.0), &self.import_tris)
                        .map(|t| origin.z - t)
                        .unwrap_or(z);
                    self.place_dispatch(glam::Vec3::new(x, y, gz));
                    count += 1;
                }
                x += sp;
            }
            y += sp;
        }
        self.fill_points.clear();
        self.spawn_status = format!("Filled {count} items.");
    }

    /// Line tool: spawn `line_count` copies of the selected item evenly from a→b.
    fn spawn_line(&mut self, a: glam::Vec3, b: glam::Vec3) {
        let n = self.line_count.max(2);
        for k in 0..n {
            let t = k as f32 / (n - 1) as f32;
            self.place_dispatch(a.lerp(b, t));
        }
        self.spawn_status = format!("Placed {n} along line.");
    }

    /// Teleport the player to a world point via the transform queue, using the
    /// player's real datum from the pose snapshot (first entry).
    #[cfg(feature = "injection")]
    fn teleport_player_to(&mut self, p: glam::Vec3) {
        match (self.pose_client.as_ref().and_then(|c| c.first()), &self.transform_queue) {
            (Some((datum, _)), Some(tq)) => {
                tq.enqueue_translate(datum, [p.x, p.y, p.z]);
                self.status = format!("Teleported player -> ({:.1},{:.1},{:.1})", p.x, p.y, p.z);
            }
            _ => self.status = NO_PLAYER_POSE.into(),
        }
    }

    /// Fly the camera to a world point: stand 8 wu back / 4 wu up, looking at it (light-bake "Camera" /
    /// "Move here" follow).
    fn frame_on_point(&mut self, target: glam::Vec3) {
        self.camera.pos = target + glam::Vec3::new(-6.0, -6.0, 4.0);
        let d = (target - self.camera.pos).normalize_or_zero();
        if d.length_squared() > 1e-6 {
            self.camera.yaw = d.y.atan2(d.x);
            self.camera.pitch = d.z.clamp(-1.0, 1.0).asin();
        }
        self.status = format!("Camera at ({:.1},{:.1},{:.1}) looking at ({:.1},{:.1},{:.1})", self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, target.x, target.y, target.z);
    }

    /// Move the camera to look at the player (first pose-snapshot entry).
    fn frame_on_player(&mut self) {
        if let Some((_, p)) = self.pose_client.as_ref().and_then(|c| c.first()) {
            let target = glam::Vec3::from(p);
            self.camera.pos = target + glam::Vec3::new(-6.0, -6.0, 3.0);
            let d = (target - self.camera.pos).normalize_or_zero();
            if d.length_squared() > 1e-6 {
                self.camera.yaw = d.y.atan2(d.x);
                self.camera.pitch = d.z.clamp(-1.0, 1.0).asin();
            }
            self.status = format!("Framed on player ({:.1},{:.1},{:.1})", p[0], p[1], p[2]);
        } else {
            self.status = NO_PLAYER_POSE.into();
        }
    }

}

/// The palette preview's orbit camera (shared by the GUI preview and the headless
/// HMS_PREVIEW_SHOT). `yaw`/`pitch` in radians, `zoom` = distance multiplier. Distance = the model's
/// PROJECTED silhouette for THIS orbit angle: perp_half = max perpendicular-to-view
/// extent of the AABB corners, near_depth pushes the eye back so the closest corner clears the near
/// plane; a long/thin model then fills the frame instead of leaving the loose bounding-sphere margin.
pub(crate) fn preview_camera(center: glam::Vec3, mn: glam::Vec3, mx: glam::Vec3, radius: f32, yaw: f32, pitch: f32, zoom: f32) -> Camera {
    let fov = 45f32.to_radians();
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let dir = glam::Vec3::new(cy * cp, sy * cp, sp).normalize();
    let half_fov = fov * 0.5;
    let mut perp_half = 0.0f32;
    let mut near_depth = 0.0f32;
    for i in 0..8 {
        let c = glam::Vec3::new(
            if i & 1 == 0 { mn.x } else { mx.x },
            if i & 2 == 0 { mn.y } else { mx.y },
            if i & 4 == 0 { mn.z } else { mx.z },
        );
        let v = c - center;
        let dproj = v.dot(dir);
        perp_half = perp_half.max((v - dir * dproj).length());
        near_depth = near_depth.max(dproj);
    }
    let fit = (perp_half / half_fov.tan()) * 1.08 + near_depth;
    let dist = (fit * zoom).max(radius * 0.15);
    let eye = center + dir * dist;
    let to = (center - eye).normalize_or_zero();
    Camera {
        pos: eye,
        yaw: to.y.atan2(to.x),
        pitch: to.z.clamp(-1.0, 1.0).asin(),
        fov_y: fov,
        near: (dist * 0.01).max(0.01),
        far: dist + radius * 4.0 + 10.0,
    }
}

impl App {
    /// Render + show the selected palette object's render-model preview. Decodes
    /// the model once per selection change, orbits it slowly, and draws it as an egui
    /// image. Reuses the same offscreen-render→egui-texture path as the main viewport.
    fn show_preview(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Static-palette entries carry the object tag → resolve its render_model. Also carry the
        // OBJECT tag + variant sid so the preview shows the same variant attachments the viewport does.
        let pal_i = self.selected_static_pal;
        let obj_tag = pal_i.and_then(|i| self.static_palette.get(i).map(|(t, _)| *t)).unwrap_or(0);
        let variant_sid = pal_i.and_then(|i| self.static_palette_variant.get(i).copied()).unwrap_or(0);
        let mode_tag = (obj_tag != 0)
            .then(|| self.objscene().map(|s| s.resolve_object_mode(obj_tag)))
            .flatten()
            .filter(|&m| m != 0 && m != 0xFFFF_FFFF);
        let Some(mode_tag) = mode_tag else { return };
        // Re-key on (mode_tag, variant_sid) so switching variant (rocket↔gauss) rebuilds the preview.
        // #h4-preview: a Halo 4 palette VARIANT is its own obje tag (h4/palette.rs), and two rows can
        // share a render model (or the marker cube), so the Halo 4 key carries the obje tag instead.
        let preview_key = if self.h4_active { mode_tag ^ obj_tag.rotate_left(16) } else { mode_tag ^ variant_sid.rotate_left(16) };
        if self.preview_for != Some(preview_key) {
            self.preview_for = Some(preview_key);
            self.preview_ok = false;
            // PREVIEW PARITY: the preview meshes come from the SAME object path as the
            // viewport (`SceneController::build_preview_meshes` -> `maybe_rebuild`: same DecodedModel,
            // per-part material lanes, foliage/terrain/glass/halogram routing and the alpha-test CUTOUT
            // pass), for one synthetic placement at the viewport camera position (so the probe lighting
            // is what the object would get dropped right here).
            let at = self.camera.pos;
            // #h4-preview: the same call on the Halo 4 scene (`H4ObjectScene::build_preview_meshes`
            // -> its own `maybe_rebuild`: same decoded model cache, same `upload_model_parts`
            // material lanes and the same probe / airprobe object lighting sampled at `at`). The
            // preview renderer's Halo 4 post state was mirrored at map load (h4/gui.rs).
            let built = if self.h4_active {
                self.h4_scene.as_deref_mut().map_or(Err(false), |s| s.build_preview_meshes(
                    obj_tag, variant_sid, at, self.renderer.mesh_renderer(), &self.render_state.device, &self.render_state.queue))
            } else {
                self.scene_ctl.as_mut().map_or(Err(false), |s| s.build_preview_meshes(
                    obj_tag, variant_sid, at, self.renderer.mesh_renderer(), &self.render_state.device, &self.render_state.queue))
            };
            if let Ok((op, cut, holo, holo_solid, blend, bmin, bmax)) = built {
                self.preview_renderer.set_dynamic_meshes(op);
                self.preview_renderer.set_dynamic_cutout_meshes(cut);
                self.preview_renderer.set_dynamic_holo_meshes(holo);
                self.preview_renderer.set_dynamic_holo_solid_meshes(holo_solid);
                self.preview_renderer.set_dynamic_blend_meshes(blend);
                self.preview_center = (bmin + bmax) * 0.5;
                self.preview_radius = ((bmax - bmin).length() * 0.5).max(0.5);
                self.preview_min = bmin;
                self.preview_max = bmax;
                self.preview_ok = true;
            } else if matches!(built, Err(true)) {
                // Decode still in flight (background predecode): retry next frame.
                self.preview_for = None;
            }
        }
        if !self.preview_ok {
            return;
        }
        // Interactive orbit + zoom (the framing itself is `preview_camera`).
        let t = ctx.input(|i| i.time) as f32;
        // The preview is a LIVE 3D render-to-texture painted into a fixed square, with a STABLE
        // interaction id (`preview3d`) so egui tracks the drag across frames even as the list above
        // it grows/reflows. LEFT-drag pulls the model out into the scene (a ghost follows the
        // cursor, placed on release by the viewport); RIGHT-drag orbits; scroll zooms. Plain
        // interact rather than egui's dnd_drag_source, which hijacks the cursor.
        let sz = 190.0f32;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(sz, sz), egui::Sense::hover());
        ui.painter().image(
            self.preview_tex,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let resp = ui
            .interact(rect, egui::Id::new("preview3d"), egui::Sense::click_and_drag())
            .on_hover_text("LEFT-drag the model into the scene · RIGHT-drag to orbit · scroll to zoom");
        // LEFT-drag OUT → queue a placement; the viewport spawns the object under the cursor and
        // starts a release-mode grab so it follows the mouse.
        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.pending_place = self.selected_static_pal;
        }
        // RIGHT-drag → orbit the model.
        if resp.dragged_by(egui::PointerButton::Secondary) {
            let d = resp.drag_delta();
            self.preview_yaw -= d.x * 0.01;
            self.preview_pitch = (self.preview_pitch + d.y * 0.01).clamp(-1.5, 1.5);
        }
        if resp.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.preview_zoom = (self.preview_zoom * (1.0 - scroll * 0.001)).clamp(0.25, 4.0);
            }
        }
        // The orbit camera lives in `preview_camera` so the headless HMS_PREVIEW_SHOT
        // renders the palette preview through the very same framing.
        let cam = preview_camera(self.preview_center, self.preview_min, self.preview_max, self.preview_radius, self.preview_yaw, self.preview_pitch, self.preview_zoom);
        // Re-render only when the model / orbit / zoom changed, plus a 10 Hz refresh so
        // animated materials (holo pulse, scroll) still move (a render is the full pipeline:
        // shadow, sky, scene, bloom chain, post).
        self.preview_frame = self.preview_frame.wrapping_add(1);
        let key = (preview_key, self.preview_yaw, self.preview_pitch, self.preview_zoom);
        let changed = self.preview_last != Some(key);
        if changed || self.preview_frame % 6 == 0 {
            self.preview_renderer.render(&self.render_state.device, &self.render_state.queue, &cam, t);
            self.preview_last = Some(key);
        }
        // Keep the orbit/zoom smooth while the user is interacting; otherwise the app's normal
        // ~60 Hz repaint cadence (request_repaint_after) covers the periodic refresh.
        if changed || resp.dragged() {
            ctx.request_repaint();
        }
    }

    /// Fingerprint of the editable object collections (variant + placed). Any add /
    /// delete / duplicate / reorder / label change moves it, which is what tells the Objects
    /// list to rebuild its rows. Cheap: one pass over ~650 u32s per frame.
    fn objects_fingerprint(&self) -> u64 {
        let mut h: u64 = 1469598103934665603;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(1099511628211);
        };
        mix(self.mvar_objects.len() as u64);
        for o in &self.mvar_objects {
            mix(o.datum as u64);
            if let Some(m) = self.mvar_meta.get(&o.datum) {
                mix(m.label_idx as u64 + 1);
                mix(m.slot as u64 + 1);
                // The row markers follow the EFFECTIVE flags (team/label/override/globals)
                mix(m.team as u64 + 3);
                mix(((m.scaled_on() as u64) << 1 | m.shadow_on() as u64) + 5);
                if let Some(h) = m.h4.as_deref() { mix(h.scale_q as u64 + 19); mix(m.spawn_seq as u64 + 23); }
            }
        }
        mix(((self.obj_globals.scaled as u64) << 1 | self.obj_globals.shadowcasters as u64) + 11);
        mix(self.local_objects.len() as u64 + 7);
        for o in &self.local_objects {
            mix(o.datum as u64);
        }
        mix(self.mvar_labels.len() as u64 + 13);
        h
    }

    /// The Objects list in the left panel -- every forge object in the current variant
    /// (plus anything placed since), searchable and selectable. Click selects (Ctrl toggles,
    /// Shift extends from the last clicked row), double-click frames the camera on it. The same
    /// selection state the viewport uses (`selected_set` / `selected_datum`), so the highlight and
    /// the Object panel follow. No refresh button: rows are rebuilt whenever the object fingerprint
    /// changes, the filtered index only when the fingerprint or the search text changes.
    fn objects_list_ui(&mut self, ui: &mut egui::Ui) {
        let n_total = self.mvar_objects.len() + self.local_objects.len();
        ui.heading(format!("Objects ({n_total})"));
        if n_total == 0 {
            ui.small(egui::RichText::new("Open a variant or drag palette items into the viewport.").weak());
            return;
        }
        // 1. rows (label + search key) -- rebuilt on fingerprint change only.
        let fp = self.objects_fingerprint();
        if fp != self.obj_list_fp || self.obj_list_rows.is_empty() {
            self.obj_list_fp = fp;
            self.obj_list_rows.clear();
            for o in &self.mvar_objects {
                let (label, key) = match self.mvar_meta.get(&o.datum) {
                    Some(m) => {
                        let leaf = m.name.rsplit(['\\', '/']).next().unwrap_or(&m.name).to_string();
                        let pretty = if leaf.is_empty() { "(unresolved)".to_string() } else { prettify_stringid(&leaf) };
                        // Rows are one line: cap the name so a long piece name cannot wrap.
                        let mut label = clip_chars(&pretty, 30);
                        if !m.label.is_empty() {
                            label.push_str(&format!("  \"{}\"", clip_chars(&m.label, 14)));
                        }
                        let slot = if m.slot != 0xFFFF { format!("#{}", m.slot) } else { "new".to_string() };
                        label.push_str(&format!("   {slot}"));
                        // Small markers for the effective pseudo-flags (searchable too).
                        let (fs, fc) = forge_scale::effective_flags(&self.obj_globals, &m.flags, m.team, &m.label);
                        let mut marks = String::new();
                        // A Halo 4 object is [scaled] by the gametype rule (SCALED flag)
                        // only - its record scale field is not drawn by MCC, so it earns no mark
                        if fs { marks.push_str(" [scaled]"); }
                        if fc { marks.push_str(" [shadow]"); }
                        label.push_str(&marks);
                        let key = format!("{} {} {} {}{}", pretty, leaf, m.label, slot, marks).to_lowercase();
                        (label, key)
                    }
                    None => ("(object)".to_string(), "object".to_string()),
                };
                self.obj_list_rows.push((o.datum, label, key));
            }
            for o in &self.local_objects {
                let name = self
                    .static_palette
                    .iter()
                    .find(|(t, _)| *t == o.primary_tag)
                    .map(|(_, n)| n.split(" · ").last().unwrap_or(n).to_string())
                    .unwrap_or_else(|| "placed object".to_string());
                let pretty = prettify_stringid(&name);
                self.obj_list_rows.push((o.datum, format!("{}   placed", clip_chars(&pretty, 30)), format!("{pretty} {name} placed").to_lowercase()));
            }
            self.obj_list_filter_cached.clear();
            self.obj_list_filter_cached.push('\u{0}'); // force the filter pass below
        }
        // 2. search box.
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.obj_list_filter)
                    .hint_text("search objects (type, name, label, #slot)…")
                    .desired_width(190.0),
            );
            if !self.obj_list_filter.is_empty() && ui.small_button("clear").clicked() {
                self.obj_list_filter.clear();
            }
        });
        // 3. filtered index -- rebuilt when the rows or the search text changed.
        if self.obj_list_filter_cached != self.obj_list_filter {
            self.obj_list_filter_cached = self.obj_list_filter.clone();
            let needle = self.obj_list_filter.trim().to_lowercase();
            self.obj_list_filtered = self
                .obj_list_rows
                .iter()
                .enumerate()
                .filter(|(_, (_, _, key))| needle.is_empty() || key.contains(&needle))
                .map(|(i, _)| i)
                .collect();
        }
        let shown = self.obj_list_filtered.len();
        let n_sel = self.selected_set.len();
        ui.small(
            egui::RichText::new(if n_sel > 0 {
                format!("{shown} shown · {n_sel} selected · click = select, Ctrl/Shift = multi, double-click = fly to")
            } else {
                format!("{shown} shown · click = select, Ctrl/Shift = multi, double-click = fly to")
            })
            .weak(),
        );
        // 4. the list. A selection change made elsewhere (viewport click, Ctrl+A) scrolls the
        // list to the primary row once.
        let row_h = ui.spacing().interact_size.y;
        let mut scroll_to: Option<f32> = None;
        if self.obj_list_last_sel != self.selected_datum {
            self.obj_list_last_sel = self.selected_datum;
            if let Some(d) = self.selected_datum {
                if let Some(pos) = self.obj_list_filtered.iter().position(|&i| self.obj_list_rows[i].0 == d) {
                    scroll_to = Some((pos as f32 * (row_h + ui.spacing().item_spacing.y) - 3.0 * row_h).max(0.0));
                }
            }
        }
        let mut clicked: Option<(u32, bool, bool)> = None; // (datum, ctrl, shift)
        let mut double: Option<u32> = None;
        let mut area = egui::ScrollArea::vertical()
            .id_salt("objects-list")
            .max_height(240.0)
            .auto_shrink([false, false]);
        if let Some(off) = scroll_to {
            area = area.vertical_scroll_offset(off);
        }
        area.show_rows(ui, row_h, shown, |ui, range| {
            for k in range {
                let Some(&ri) = self.obj_list_filtered.get(k) else { continue };
                let (datum, label, _) = &self.obj_list_rows[ri];
                let sel = self.selected_set.contains(datum);
                let primary = self.selected_datum == Some(*datum);
                let text = if primary {
                    egui::RichText::new(label).strong()
                } else {
                    egui::RichText::new(label)
                };
                // A separate orange "!" glyph (ASCII, tooltip) before rows whose LIVE
                // position is in a non-playable BSP; computed per visible row only.
                let bsp_warn = self.bsp_warn_datum(*datum);
                let r = ui.horizontal(|ui| {
                    if let Some(bsp) = bsp_warn {
                        ui.add_sized([12.0, row_h], egui::Label::new(egui::RichText::new("!").strong().color(egui::Color32::from_rgb(255, 160, 60))))
                            .on_hover_text(format!("{} (BSP {bsp})", Self::BSP_WARN_TEXT));
                    }
                    ui.add_sized(
                        [ui.available_width(), row_h],
                        egui::SelectableLabel::new(sel, text),
                    )
                }).inner;
                // Hover = what this row IS (palette item, slot, obje tag path, class, tag id).
                let r = r.on_hover_ui(|ui| {
                    if let Some(id) = self.object_identity(*datum) {
                        ui.label(egui::RichText::new(id.tooltip()).small());
                    }
                });
                if r.double_clicked() {
                    double = Some(*datum);
                } else if r.clicked() {
                    let (ctrl, shift) = ui.input(|i| (i.modifiers.ctrl || i.modifiers.command, i.modifiers.shift));
                    clicked = Some((*datum, ctrl, shift));
                }
            }
        });
        if let Some((d, ctrl, shift)) = clicked {
            if ctrl {
                if let Some(p) = self.selected_set.iter().position(|&x| x == d) {
                    self.selected_set.remove(p);
                    self.selected_datum = self.selected_set.last().copied();
                } else {
                    self.selected_set.push(d);
                    self.selected_datum = Some(d);
                }
            } else if shift && self.obj_list_anchor.is_some() {
                let a = self.obj_list_anchor.unwrap();
                let idx_of = |x: u32| self.obj_list_filtered.iter().position(|&i| self.obj_list_rows[i].0 == x);
                match (idx_of(a), idx_of(d)) {
                    (Some(ia), Some(id)) => {
                        let (lo, hi) = if ia <= id { (ia, id) } else { (id, ia) };
                        for k in lo..=hi {
                            let x = self.obj_list_rows[self.obj_list_filtered[k]].0;
                            if !self.selected_set.contains(&x) {
                                self.selected_set.push(x);
                            }
                        }
                        self.selected_datum = Some(d);
                    }
                    _ => {
                        self.selected_set = vec![d];
                        self.selected_datum = Some(d);
                    }
                }
            } else {
                self.selected_set = vec![d];
                self.selected_datum = Some(d);
            }
            self.obj_list_anchor = Some(d);
            self.obj_list_last_sel = self.selected_datum; // the list caused it: do not re-scroll
            self.apply_selection_highlight();
            self.status = match self.selected_set.len() {
                0 => "Nothing selected".to_string(),
                1 => "Selected 1 object".to_string(),
                n => format!("Selected {n} objects"),
            };
        }
        if let Some(d) = double {
            self.selected_set = vec![d];
            self.selected_datum = Some(d);
            self.obj_list_anchor = Some(d);
            self.obj_list_last_sel = Some(d);
            self.apply_selection_highlight();
            self.frame_selected();
        }
    }

    /// Blender-style floating tool buttons overlaid on the viewport (top-left). Select / Move /
    /// Rotate pick the active tool mode (which gizmo shows + what a gizmo drag does).
    /// Drawn INSIDE the viewport's own Ui (the CentralPanel, Order::Background) as a
    /// child Ui clamped to the viewport rect -- not a floating Area at Order::Foreground. That
    /// keeps it (a) tied to the render view: it can never spill over the side panels however the
    /// window is sized, and (b) UNDER every window / menu / combo popup (those are Order::Middle
    /// and Order::Foreground, painted and hit-tested above the panel layer), so an open window can
    /// cover it and its buttons never steal a click meant for a popup. Within the panel layer the
    /// buttons are added after the viewport image, so egui's hit test (last = topmost) gives them
    /// the click rather than the image underneath.
    fn floating_toolbar(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let want = egui::Rect::from_min_size(viewport.left_top() + egui::vec2(10.0, 56.0), egui::vec2(46.0, 4.0 * 34.0 + 3.0 * 5.0 + 12.0));
        let rect = want.intersect(viewport);
        if rect.width() < 20.0 || rect.height() < 20.0 {
            return; // viewport too small to host it
        }
        ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(egui::Layout::top_down(egui::Align::Min)), |ui| {
            ui.set_clip_rect(viewport);
            {
                // Genuinely semi-transparent dark panel (alpha in the fill, not gamma darkening).
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(24, 26, 30, 170))
                    .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40)))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::same(5))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 5.0;
                        // Painter-drawn ICONS (glyphs rendered as tofu boxes in the default font).
                        let mut mode_btn = |ui: &mut egui::Ui, m: ToolMode, kind: u8, tip: &str| {
                            let active = self.tool_mode == m;
                            // click_and_drag so a drag that starts on a button is owned by the
                            // button, never by the viewport image under it (no box-select / camera
                            // drag from a toolbar press).
                            let (rect, resp) = ui.allocate_exact_size(egui::vec2(34.0, 34.0), egui::Sense::click_and_drag());
                            let bg = if active {
                                egui::Color32::from_rgb(70, 110, 175)
                            } else if resp.hovered() {
                                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 22)
                            } else {
                                egui::Color32::TRANSPARENT
                            };
                            ui.painter().rect_filled(rect, 5.0, bg);
                            let c = rect.center();
                            let r = 8.5f32;
                            let col = egui::Color32::from_gray(if active { 250 } else { 210 });
                            let s = egui::Stroke::new(2.0_f32, col);
                            let p = ui.painter();
                            match kind {
                                0 => {
                                    // Select: the classic mouse CURSOR (pointer arrow). Built from two
                                    // convex pieces — the pointer HEAD triangle + a TAIL quad — so it
                                    // reads as a real cursor rather than a wedge.
                                    // Tip sits upper-left; the whole glyph is sized to fit the button.
                                    let h = 14.0f32;
                                    let tip = c - egui::vec2(h * 0.34, h * 0.46);
                                    let pt = |x: f32, y: f32| tip + egui::vec2(x * h, y * h);
                                    let edge = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(20));
                                    // Pointer head: tip → straight down the left edge → right wing.
                                    let head = vec![pt(0.0, 0.0), pt(0.0, 0.92), pt(0.66, 0.60)];
                                    p.add(egui::Shape::convex_polygon(head, col, edge));
                                    // Tail: a short angled quad hanging off the notch (the cursor "leg").
                                    let tail = vec![pt(0.30, 0.64), pt(0.46, 0.58), pt(0.64, 1.02), pt(0.47, 1.08)];
                                    p.add(egui::Shape::convex_polygon(tail, col, edge));
                                }
                                1 => {
                                    // Move: 4-way arrows from the centre.
                                    for d in [egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0), egui::vec2(0.0, 1.0), egui::vec2(0.0, -1.0)] {
                                        p.arrow(c, d * r, s);
                                    }
                                }
                                3 => {
                                    // Construct: two crossed dashed guides with a node where they meet.
                                    for (a, b) in [
                                        (egui::vec2(-1.0, -1.0), egui::vec2(1.0, 1.0)),
                                        (egui::vec2(-1.0, 1.0), egui::vec2(1.0, -1.0)),
                                    ] {
                                        // Dashes: 3 short segments per diagonal so it reads as a GUIDE.
                                        for k in 0..3 {
                                            let t0 = k as f32 / 3.0 + 0.04;
                                            let t1 = (k as f32 + 0.62) / 3.0;
                                            let p0 = c + (a + (b - a) * t0) * r;
                                            let p1 = c + (a + (b - a) * t1) * r;
                                            p.line_segment([p0, p1], s);
                                        }
                                    }
                                    p.circle_filled(c, 2.6, s.color);
                                }
                                _ => {
                                    // Rotate: a ~280° arc with a tangent arrowhead (a real rotate icon,
                                    // not a full refresh ring).
                                    let rr = r * 0.82;
                                    let (a0, a1) = (-2.1f32, 2.7f32);
                                    let n = 22;
                                    let pts: Vec<egui::Pos2> = (0..=n)
                                        .map(|i| {
                                            let a = a0 + (a1 - a0) * (i as f32 / n as f32);
                                            c + egui::vec2(a.cos() * rr, a.sin() * rr)
                                        })
                                        .collect();
                                    let end = *pts.last().unwrap();
                                    p.add(egui::Shape::line(pts, s));
                                    let tangent = egui::vec2(-a1.sin(), a1.cos());
                                    p.arrow(end, tangent * (r * 0.55), s);
                                }
                            }
                            if resp.on_hover_text(tip).clicked() {
                                self.tool_mode = m;
                            }
                        };
                        mode_btn(ui, ToolMode::Select, 0, "Select — click an object or drag a box (Shift = add)");
                        mode_btn(ui, ToolMode::Move, 1, "Move — drag the move gizmo (or press G)");
                        mode_btn(ui, ToolMode::Rotate, 2, "Rotate — drag the rotate gizmo (or press R)");
                        mode_btn(ui, ToolMode::Construct, 3, "Construction guides (C) — click anchor points to draw dotted guides, then snap to where they cross");
                    });
            }
        });
    }

    /// Cursor→surface point in the viewport: nearest of (forge object under cursor, BSP/ground,
    /// imported geometry, ground plane). `exclude` skips the object being placed so it doesn't hit
    /// itself. Returns the exact world point under the cursor.
    fn cursor_surface_point(&self, rect: egui::Rect, px: f32, py: f32, exclude: &[u32]) -> glam::Vec3 {
        let ndc_x = ((px - rect.min.x) / rect.width().max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((py - rect.min.y) / rect.height().max(1.0)) * 2.0;
        let aspect = rect.width() / rect.height().max(1.0);
        let inv = self.camera.view_proj(aspect).inverse();
        let near = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize_or_zero();
        // Nearest of object-hit vs BSP-hit (so dropping onto a forge block lands ON the block).
        let obj = self.objscene().and_then(|s| s.raycast_objects_excluding(near, dir, exclude));
        let bsp = self.objscene().and_then(|s| s.raycast_scene(near, dir));
        let pick = match (obj, bsp) {
            (Some(a), Some(b)) => Some(if (a - near).length_squared() <= (b - near).length_squared() { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        pick.or_else(|| raycast_tris(near, dir, &self.import_tris).map(|t| near + dir * t))
            .unwrap_or_else(|| {
                if dir.z.abs() > 1e-4 {
                    let t = -near.z / dir.z;
                    if t > 0.0 { near + dir * t } else { near + dir * 12.0 }
                } else {
                    near + dir * 12.0
                }
            })
    }

    /// Drag-out placement. Returns true while it owns input (suppresses camera/pick).
    /// 1) queued item + held cursor enters the viewport → SPAWN the object + enter `placing`.
    /// 2) while `placing`, the object snaps EXACTLY to the surface under the cursor every frame.
    /// 3) mouse release → drop it there (finalize).
    fn try_drag_place(&mut self, ctx: &egui::Context, response: &egui::Response) -> bool {
        let rect = response.rect;
        let released = ctx.input(|inp| inp.pointer.any_released());
        let p = ctx.pointer_latest_pos();
        let over = p.map(|p| rect.contains(p)).unwrap_or(false);

        // 1) Spawn when the held cursor first enters the viewport.
        if let Some(pi) = self.pending_place {
            if released {
                self.pending_place = None; // released before entering -> cancel
            } else if self.placing.is_none() && over {
                if let Some(p) = p {
                    let hit = self.cursor_surface_point(rect, p.x, p.y, &[]);
                    self.selected_static_pal = Some(pi);
                    // Snapshot BEFORE spawning so a single Ctrl+Z undoes the whole drag-place
                    // (spawn + move-in-place) — undoing removes the object from the world. The
                    // in-place follow mutates the object without pushing further undo entries.
                    self.push_edit_undo();
                    self.prop_snapshotted_for = None;
                    // place_local_at assigns `next_dup_datum` and pushes to `mvar_objects` (the
                    // saveable/editable path), so that is where the spawn is checked for.
                    let d = self.next_dup_datum;
                    self.place_local_at(hit);
                    if self.mvar_objects.last().map(|o| o.datum) == Some(d) {
                        self.selected_set = vec![d];
                        self.selected_datum = Some(d);
                        self.placing = Some(d);
                    } else {
                        self.edit_undo.pop(); // spawn failed -> discard the snapshot
                    }
                }
                self.pending_place = None;
            }
        }

        // 2/3) Follow the surface under the cursor; drop on release.
        if let Some(d) = self.placing {
            if let Some(p) = p {
                if over {
                    let hit = self.cursor_surface_point(rect, p.x, p.y, &[d]);
                    let (_, f, u) = self.movable_pose(d).unwrap_or((glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Z));
                    self.set_movable_pose(d, hit, f, u);
                    if let Some(s) = self.objscene_mut() { s.invalidate(); }
                    self.apply_selection_highlight();
                }
            }
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
            ctx.request_repaint();
            if released {
                self.placing = None; // dropped
                self.status = "Placed object".into();
            }
            return true;
        }
        false
    }

    /// Left-click in the viewport → cast a world ray and pick the nearest object.
    fn handle_pick(&mut self, response: &egui::Response) {
        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        // Swallow the click that just confirmed a transform (its release fires here a frame
        // later) so completing a move/duplicate doesn't select a background object.
        if self.suppress_next_click {
            self.suppress_next_click = false;
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else { return };
        let rect = response.rect;
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return;
        }
        let (near, dir) = self.cursor_ray(rect, pos.x, pos.y);

        // Snap-to-surface: raycast the click against the real BSP map
        // geometry first (the triangle soup built during load), then imported walkable
        // geometry, then the z=0 plane, then 10u ahead. This places objects ON the map
        // surface people actually walk on, not the origin plane.
        let ground = if let Some(hit) = self.objscene().and_then(|s| s.raycast_scene(near, dir)) {
            hit
        } else if let Some(t) = raycast_tris(near, dir, &self.import_tris) {
            near + dir * t
        } else if dir.z.abs() > 1e-4 {
            let t = -near.z / dir.z;
            if t > 0.0 { near + dir * t } else { near + dir * 10.0 }
        } else {
            near + dir * 10.0
        };

        // Construct tool: clicks pick ANCHORS and draw guides rather than selecting.
        // Falling back to the surface point under the cursor when no anchor is in range lets a
        // guide start from open ground -- that is how "this far from that wall" gets measured.
        if self.construct_mode {
            let anchor = self
                .pick_anchor(rect, (pos.x, pos.y))
                .unwrap_or_else(|| construct::Anchor::point(ground));
            // Clicking a shape's CENTRE marker selects it, so a shape can be
            // picked in the viewport (then moved with G) instead of only from the list.
            if let Some((i, _)) = self.shapes.iter().enumerate()
                .map(|(i, sh)| (i, (sh.center - anchor.pos).length()))
                .filter(|(_, d)| *d < self.shape_grab_radius())
                .min_by(|a, b| a.1.total_cmp(&b.1))
            {
                if !matches!(self.construct_op, ConstructOp::Circle | ConstructOp::Square) {
                    self.sel_shape = Some(i);
                    self.overlays_dirty = true;
                    self.status = format!("Selected {} {i} — press G to move it", self.shapes[i].kind.label());
                    return;
                }
            }
            // Every construction action that needs a point in the world is a
            // CLICK tool. (A panel button that says "hover a point then press me" can never
            // fire: construct_hover is cleared the instant the cursor leaves the viewport.)
            match self.construct_op {
                ConstructOp::Circle | ConstructOp::Square => {} // handled as a press-drag above
                ConstructOp::AnchorTo => {
                    self.anchor_selection_to(anchor.pos);
                    self.overlays_dirty = true;
                    return;
                }
                ConstructOp::Mirror => {
                    self.mirror_selection(anchor.pos);
                    self.overlays_dirty = true;
                    return;
                }
                // #snap-array: fill a line with the selection. One click in length mode;
                // otherwise click the start, then the end.
                ConstructOp::Line => {
                    let cfg = self.line_tool;
                    if cfg.one_click {
                        let to = anchor.pos + cfg.dir() * cfg.length.max(0.05);
                        self.array_along_line(anchor.pos, to, cfg.spec(), cfg.align);
                    } else {
                        match self.line_pending.take() {
                            None => {
                                self.line_pending = Some(anchor.pos);
                                self.status = format!(
                                    "Line array: start at ({:.2}, {:.2}, {:.2}) — click the END point (Esc cancels)",
                                    anchor.pos.x, anchor.pos.y, anchor.pos.z
                                );
                            }
                            Some(a) => {
                                if (anchor.pos - a).length() < 1e-3 {
                                    self.line_pending = Some(a);
                                    self.status = "Line array needs two different points".into();
                                } else {
                                    self.array_along_line(a, anchor.pos, cfg.spec(), cfg.align);
                                }
                            }
                        }
                    }
                    self.overlays_dirty = true;
                    return;
                }
                ConstructOp::Coincident => {
                    match self.face_of_anchor(&anchor) {
                        None => self.status = "Coincident: click a FACE marker (set Snap to = Faces)".into(),
                        Some((c, n, label)) => match self.cad_face_a.take() {
                            None => {
                                self.cad_face_a = Some((c, n, label.clone()));
                                self.status = format!("Coincident: A = {label} — now click the target face");
                            }
                            Some(_a) => {
                                // A is already stored; constrain_coincident reads it back.
                                self.cad_face_a = Some(_a);
                                self.constrain_coincident(c, n);
                                self.cad_face_a = None;
                            }
                        },
                    }
                    self.overlays_dirty = true;
                    return;
                }
                ConstructOp::Guide => {}
            }
            match self.construct_pending.take() {
                None => {
                    self.construct_pending = Some(anchor);
                    self.status = format!("Guide from {} — click the second point (Esc cancels)", anchor.kind.label());
                }
                Some(start) => {
                    if (anchor.pos - start.pos).length() < 1e-3 {
                        self.status = "Guide needs two different points".into();
                    } else {
                        self.guides.push(construct::Guide { a: start, b: anchor, color: [0.55, 0.95, 0.6] });
                        self.show_guides = true;
                        let n = self.guide_targets().iter().filter(|t| t.kind == construct::SnapKind::Intersection).count();
                        self.status = format!(
                            "Guide added ({:.2} wu) — {} guides, {n} intersection(s)",
                            (anchor.pos - start.pos).length(),
                            self.guides.len()
                        );
                    }
                }
            }
            self.overlays_dirty = true;
            return;
        }

        // Line tool: first click sets the start, second click spawns copies along it.
        if self.line_mode {
            if let Some(start) = self.line_start.take() {
                self.spawn_line(start, ground);
            } else {
                self.line_start = Some(ground);
                self.spawn_status = "Line start set — click the end point.".into();
            }
            return;
        }

        // Fill tool: accumulate polygon outline points.
        if self.fill_mode {
            self.fill_points.push(ground);
            self.spawn_status = format!("Fill: {} points ('Close & fill' when done)", self.fill_points.len());
            return;
        }

        let hit = self.objscene().and_then(|s| s.pick(near, dir));
        let shift = response.ctx.input(|i| i.modifiers.shift);
        if shift {
            // Shift+click: toggle the hit object in/out of the multi-select set.
            if let Some(d) = hit {
                if let Some(pos) = self.selected_set.iter().position(|&x| x == d) {
                    self.selected_set.remove(pos);
                    // Primary follows: last remaining, or None.
                    self.selected_datum = self.selected_set.last().copied();
                } else {
                    self.selected_set.push(d);
                    self.selected_datum = Some(d);
                }
            }
            // Shift+click on empty space: keep the current set unchanged.
        } else {
            // Plain click: replace the selection with just this hit (or clear).
            self.selected_set = hit.into_iter().collect();
            self.selected_datum = hit;
        }
        self.apply_selection_highlight();
        if self.show_collision || self.show_physics {
            self.rebuild_overlays();
        }
        self.status = match hit {
            Some(d) => {
                // Name WHAT was clicked (tag path, class, tag id, placement).
                let what = self.object_identity(d).map(|id| id.status_line()).unwrap_or_else(|| format!("0x{d:08X}"));
                if self.selected_set.len() > 1 {
                    format!("Selected {} objects (last: {what})", self.selected_set.len())
                } else {
                    format!("Selected {what}")
                }
            }
            None => {
                // No object → inspect the BSP surface material under the cursor.
                match self.scene_ctl.as_ref().and_then(|s| s.bsp_material_at(near, dir)) {
                    Some((diffuse, blend, mat_idx)) => {
                        let b = match blend {
                            0 => "opaque",
                            1 => "additive",
                            2 => "multiply",
                            3 => "alpha-blend",
                            _ => "other",
                        };
                        // Include the material INDEX so a specific surface can be pinpointed
                        // (multiple materials share a texture but tile differently).
                        format!("BSP surface — mat idx {mat_idx}, diffuse 0x{diffuse:08X}, blend {blend} ({b})")
                    }
                    None => "No object under cursor.".to_string(),
                }
            }
        };
    }

    /// True once the injected DLL is publishing its shared state (MapInfo MMF).
    #[cfg(feature = "injection")]
    fn dll_connected(&self) -> bool {
        self.map_info.is_some()
    }

    /// Lazily (re)open the read-only consumer MMFs — they only exist once the DLL
    /// is injected, so we retry each frame until they connect.
    fn ensure_clients(&mut self) {
        if self.map_info.is_none() {
            self.map_info = hms_ipc::MapInfoClient::open().ok();
        }
        if self.pose_client.is_none() {
            self.pose_client = hms_ipc::PoseSnapshotClient::open().ok();
        }
        if self.objtable.is_none() {
            self.objtable = ObjectTableClient::open().ok();
        }
        if self.forge_table.is_none() {
            self.forge_table = ForgeObjectTableClient::open().ok();
        }
        if self.palette_client.is_none() {
            self.palette_client = ForgePaletteClient::open().ok();
            // First time the palette connects, pull it immediately.
            if let Some(c) = &self.palette_client {
                self.palette = c.read();
            }
        }
        if self.forge_spawn.is_none() {
            self.forge_spawn = hms_ipc::ForgeSpawnClient::open().ok();
        }
        if self.forge_edit.is_none() {
            self.forge_edit = hms_ipc::ForgeObjectEditClient::open().ok();
        }
    }

    /// Auto-attach: if MCC is running and the DLL isn't publishing yet, inject it
    /// (retry every few seconds). Removes the need to click Attach every launch.
    /// No-op stub when the `injection` feature is off (offline release build).
    #[cfg(not(feature = "injection"))]
    fn auto_attach(&mut self) {}

    #[cfg(feature = "injection")]
    fn auto_attach(&mut self) {
        let now = std::time::Instant::now();
        let due = self
            .attach_timer
            .map(|t| now.duration_since(t).as_secs_f32() >= 3.0)
            .unwrap_or(true);
        if !due {
            return;
        }
        self.attach_timer = Some(now);
        // Refresh the cached "is MCC running" for the status ladder.
        let pid = hms_inject::find_process_by_name(hms_inject::MCC_PROCESS);
        self.mcc_running = pid.is_some();
        if self.dll_connected() || !self.dll_path.exists() {
            return;
        }
        if let Some(pid) = pid {
            // Inject on a background thread — CreateRemoteThread + wait blocks
            // until the DLL's DllMain returns, and doing that on the UI thread on
            // the first frame froze the window before it was ever shown.
            let dll = self.dll_path.clone();
            std::thread::spawn(move || {
                if let Err(e) = hms_inject::inject(pid, &dll) {
                    log::warn!("auto-inject failed: {e}");
                }
            });
            self.status = format!("Attaching to {} (pid {pid})…", hms_inject::MCC_PROCESS);
        }
    }

    /// Rebuild the line overlay: trigger volumes + selected-object collision hull.
    fn rebuild_overlays(&mut self) {
        let mut segs: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();

        // Collision (cyan) + physics (orange) hulls of the selected object.
        if (self.show_collision || self.show_physics) && self.selected_datum.is_some() {
            let datum = self.selected_datum.unwrap();
            if let Some(o) = self.last_objects.iter().find(|o| o.datum == datum).cloned() {
                // #h4-phys through the trait: Reach forwards to the native walkers, Halo 4 / H2A
                // to h4::collision. Same two colours, same selected-only rule.
                if let Some(s) = self.objscene() {
                    if self.show_collision {
                        for (a, b) in s.collision_world_edges(&o) {
                            segs.push((a, b, [0.2, 0.85, 0.95]));
                        }
                    }
                    if self.show_physics {
                        for (a, b) in s.physics_world_edges(&o) {
                            segs.push((a, b, [0.95, 0.55, 0.15]));
                        }
                    }
                }
            }
        }

        // Forge object boundary shapes (sphere/cylinder/box) — teleporters, objective
        // zones. Sized from the .mvar boundary values (11-bit over [0,200] world units, blf).
        // Shown ONLY for the SELECTED object(s): a bright wireframe outline + a translucent
        // holographic volume fill (the zone pipeline adds a fresnel rim + pulse). This keeps the
        // viewport readable (no zone spam) and makes the selected zone's extent obvious.
        let mut ztris: Vec<([f32; 3], [f32; 4])> = Vec::new();
        if self.show_boundaries {
            let is_sel = |d: u32| self.selected_set.contains(&d) || self.selected_datum == Some(d);
            for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                if !is_sel(o.datum) {
                    continue;
                }
                let Some(m) = self.mvar_meta.get(&o.datum) else { continue };
                // Halo 4 shape values are 16-bit 1/256 wu (types 13-15 = teleporters)
                let (bvals, btype) = if m.h4.is_some() {
                    (m.boundary.map(|v| wu_to_bval(v as f32 / 256.0)), if (13..=15).contains(&m.cached_type) { 13 } else { 0 })
                } else {
                    (m.boundary, m.cached_type)
                };
                let (bsegs, btris) = boundary_geometry(
                    m.boundary_shape, bvals, btype,
                    glam::Vec3::from(o.pos), glam::Vec3::from(o.fwd), glam::Vec3::from(o.up),
                );
                segs.extend(bsegs);
                ztris.extend(btris);
            }
        }
        // The scene's lights as viewport markers — a small colour-coded star per light,
        // plus an aim line for spots and a ring at its far-attenuation radius. Off by default.
        if self.show_lights {
            if let Some(sc) = self.scene_ctl.as_ref() {
                for (idx, p, col, range, is_spot, dir) in sc.editable_lights() {
                    let m = col[0].max(col[1]).max(col[2]).max(1e-3);
                    let c = [col[0] / m, col[1] / m, col[2] / m];
                    let sel = self.sel_light == Some(idx);
                    let c = if sel { [1.0, 1.0, 1.0] } else { c };
                    let r = if sel { 0.5 } else { 0.3 };
                    let pv = glam::Vec3::from(p);
                    for ax in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z] {
                        segs.push(((pv - ax * r).into(), (pv + ax * r).into(), c));
                    }
                    if is_spot {
                        let d = glam::Vec3::from(dir).normalize_or_zero();
                        segs.push((pv.into(), (pv + d * range.min(8.0)).into(), c));
                    }
                    // range ring in the XY plane (cheap extent cue)
                    let n = 16;
                    for i in 0..n {
                        let (a0, a1) = (i as f32 / n as f32 * std::f32::consts::TAU, (i + 1) as f32 / n as f32 * std::f32::consts::TAU);
                        let p0 = pv + glam::Vec3::new(a0.cos(), a0.sin(), 0.0) * range;
                        let p1 = pv + glam::Vec3::new(a1.cos(), a1.sin(), 0.0) * range;
                        segs.push((p0.into(), p1.into(), [c[0] * 0.35, c[1] * 0.35, c[2] * 0.35]));
                    }
                }
            }
        }
        // The phmo overlay (solid translucent + outline) for hidden forge blocks — render model
        // present but yields no visible geometry (invisible in-game; defined by physics). Built in
        // build_object_meshes; merged into the zone-fill + overlay-line buffers so it draws
        // regardless of selection. These blocks are also pickable via their phmo AABB (added to
        // scene.picks), so they can be clicked/selected even though their render mesh is empty.
        if self.show_blockers {
            if let Some(s) = self.objscene() {
                let (btris, blines) = s.blocker_overlay();
                ztris.extend(btris.iter().copied());
                segs.extend(blines.iter().copied());
            }
        }
        // Construction geometry: dotted guides, their snap targets, and -- while the
        // construct tool is live -- the anchor under the cursor plus a rubber band from the
        // pending first point. Dotted so guides never read as map geometry.
        if self.show_guides && !self.guides.is_empty() {
            for g in &self.guides {
                construct::dashed(g.a.pos, g.b.pos, 0.35, g.color, &mut segs);
            }
            // Snap targets: intersections get a bigger marker because they are the ones people
            // are actually aiming for.
            for t in construct::snap_targets(&self.guides, 0.05) {
                let r = if t.kind == construct::SnapKind::Intersection { 0.5 } else { 0.22 };
                construct::cross(t.pos, r, t.kind.color(), &mut segs);
            }
        }
        // Drawn shapes render like guides and contribute the same snap
        // targets, with the SELECTED one brightened and its centre marked so it is obvious
        // which one a drag would grab.
        if self.show_guides {
            for (i, sh) in self.shapes.iter().enumerate() {
                let sel = self.sel_shape == Some(i);
                let col = if sel { [1.0, 1.0, 0.55] } else { sh.color };
                for g in sh.guides() {
                    construct::dashed(g.a.pos, g.b.pos, 0.35, col, &mut segs);
                    construct::cross(g.a.pos, 0.18, col, &mut segs);
                }
                construct::diamond(sh.center, if sel { 0.6 } else { 0.4 }, col, &mut segs);
            }
        }
        if self.construct_mode {
            if let Some(a) = self.construct_hover {
                construct::diamond(a.pos, 0.45, a.kind.color(), &mut segs);
                // Outline the box the hovered anchor belongs to, so it is obvious WHICH object
                // (and which corner of it) a click would take.
                if let Some(o) = a.datum.and_then(|d| self.objscene().and_then(|s| s.object_obb(d))) {
                    for (p0, p1) in o.edge_segments() {
                        construct::dashed(p0, p1, 0.5, [0.35, 0.35, 0.45], &mut segs);
                    }
                }
            }
            if let Some(start) = self.construct_pending {
                construct::diamond(start.pos, 0.5, [1.0, 1.0, 0.4], &mut segs);
                if let Some(h) = self.construct_hover {
                    construct::dashed(start.pos, h.pos, 0.3, [1.0, 1.0, 0.4], &mut segs);
                }
            }
        }

        // The structure-design kill / acceleration / slip planes, drawn like the
        // trigger volumes (translucent fill in the zone lane + edges in the line lane).
        let mut xray: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        if self.show_soft_ceilings {
            if let Some(sc) = self.scene_ctl.as_ref() {
                let (t, l) = soft_ceilings::overlay_geometry(&sc.soft_ceilings());
                ztris.extend(t);
                xray.extend(l);
            }
        }
        // The world box + floor plane (magenta) and the playable-BSP boxes (violet).
        if let Some(sc) = self.scene_ctl.as_ref() {
            if self.show_hard_floor {
                if let Some(wb) = hard_floor::world_box(sc.structure_bsp_flags(), &sc.structure_bsp_mopp_bounds()) {
                    let (t, l) = hard_floor::world_geometry(&wb, 16);
                    ztris.extend(t);
                    xray.extend(l);
                }
            }
            if self.show_playable_bounds {
                let (t, l) = hard_floor::overlay_geometry(sc.structure_bsp_flags(), 8);
                ztris.extend(t);
                xray.extend(l);
            }
        }
        self.renderer.set_xray_lines(&self.render_state.device, &xray);
        self.renderer.set_zone_tris(&self.render_state.device, if ztris.is_empty() { None } else { Some(&ztris) });

        if !self.show_triggers {
            self.renderer.set_overlay_lines(&self.render_state.device, &segs);
            return;
        }
        let vols = self.scene_ctl.as_ref().map(|s| s.trigger_volumes()).unwrap_or_default();
        segs.extend(trigger_volume_segments(&vols));
        self.renderer.set_overlay_lines(&self.render_state.device, &segs);
        self.status = format!("{} trigger volumes", vols.len());
    }

    /// Save map + camera + placed-object snapshot to the .mmsproj project.
    fn save_project(&mut self) {
        let p = self.build_project_record();
        self.status = match project::save(&p) {
            Ok(path) => format!("Saved project -> {}", path.display()),
            Err(e) => format!("Save project failed: {e}"),
        };
    }
}

/// The wireframe box edges of the scenario trigger volumes (red = kill, green =
/// safe, amber = plain). Shared by the GUI overlay and the headless script host.
pub fn trigger_volume_segments(vols: &[hms_native::TriggerVolume]) -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
    let mut segs = Vec::with_capacity(vols.len() * 12);
    {
        for v in vols {
            let color = match v.category {
                1 => [0.9, 0.2, 0.2], // kill = red
                2 => [0.2, 0.85, 0.3], // safe = green
                _ => [0.95, 0.75, 0.2], // plain = amber
            };
            let p = glam::Vec3::from(v.pos);
            let mut fwd = glam::Vec3::from(v.fwd);
            let mut up = glam::Vec3::from(v.up);
            if fwd.length_squared() < 1e-6 { fwd = glam::Vec3::X; }
            if up.length_squared() < 1e-6 { up = glam::Vec3::Z; }
            fwd = fwd.normalize();
            up = up.normalize();
            let right = fwd.cross(up).normalize_or_zero();
            let e = v.ext;
            let corner = |sx: f32, sy: f32, sz: f32| -> [f32; 3] {
                let c = p + right * (sx * e[0]) + fwd * (sy * e[1]) + up * (sz * e[2]);
                [c.x, c.y, c.z]
            };
            // 8 corners indexed by (x,y,z) sign bits, 12 edges.
            let cs = [
                corner(-1.0, -1.0, -1.0), corner(1.0, -1.0, -1.0),
                corner(1.0, 1.0, -1.0), corner(-1.0, 1.0, -1.0),
                corner(-1.0, -1.0, 1.0), corner(1.0, -1.0, 1.0),
                corner(1.0, 1.0, 1.0), corner(-1.0, 1.0, 1.0),
            ];
            let edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ];
            for (a, b) in edges {
                segs.push((cs[a], cs[b], color));
            }
        }
    }
    segs
}

impl App {

    /// The project record for the current session (map, camera, placed objects).
    fn build_project_record(&mut self) -> project::Project {
        let colors: std::collections::HashMap<u32, (u8, u8)> = self
            .forge_rows
            .iter()
            .map(|(_, d, t, c)| (*d, (*t, *c)))
            .collect();
        let objects = self
            .last_objects
            .iter()
            .map(|o| {
                let (team, color) = colors.get(&o.datum).copied().unwrap_or((8, 8));
                project::ProjObject { datum: o.datum, pos: o.pos, fwd: o.fwd, up: o.up, team, color }
            })
            .collect();
        // The pseudo-flags live only here (never in the .mvar) — record every
        // explicit override against its .mvar slot (datum as the fallback key), plus the variant
        // they belong to so reopening the project restores the same objects.
        let mut obj_flags: Vec<project::ProjObjFlags> = self
            .mvar_meta
            .iter()
            .filter(|(_, m)| !m.flags.is_default())
            .map(|(&datum, m)| project::ProjObjFlags { slot: m.slot, datum, scaled: m.flags.scaled, shadow: m.flags.shadow })
            .collect();
        obj_flags.sort_by_key(|f| (f.slot, f.datum));
        let p = project::Project {
            version: 1,
            map_path: self.map_path.clone(),
            cam_pos: [self.camera.pos.x, self.camera.pos.y, self.camera.pos.z],
            cam_yaw: self.camera.yaw,
            cam_pitch: self.camera.pitch,
            objects,
            variant_path: self.current_variant_path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
            obj_flags,
        };
        p
    }

    /// Re-attach a project's flag overrides to the freshly parsed variant objects
    /// (by .mvar slot; by datum for objects that never had a slot). Called right after the meta
    /// table is rebuilt from a variant so the overrides survive the reopen.
    fn apply_pending_obj_flags(&mut self) {
        // A variant whose first render placed nothing is retried a few ticks later;
        // keep the overrides queued until there is a meta table to attach them to.
        if self.mvar_meta.is_empty() { return; }
        let Some(flags) = self.pending_obj_flags.take() else { return };
        let mut n = 0usize;
        for f in flags {
            let key = if f.slot != 0xFFFF {
                self.mvar_meta.iter().find(|(_, m)| m.slot == f.slot).map(|(&d, _)| d)
            } else {
                self.mvar_meta.contains_key(&f.datum).then_some(f.datum)
            };
            if let Some(d) = key {
                if let Some(m) = self.mvar_meta.get_mut(&d) {
                    m.flags.scaled = f.scaled;
                    m.flags.shadow = f.shadow;
                    n += 1;
                }
            }
        }
        if n > 0 {
            log::info!("restored {n} per-object flag override(s) from the project");
        }
    }

    /// File ▸ Save Project As… — pick a path and write the project there, so more than one project
    /// can exist and a specific one can be reopened later. The quick slot (`Save project`) keeps
    /// its no-dialog behaviour.
    fn save_project_as(&mut self) {
        let seed = self
            .current_project_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                let stem = std::path::Path::new(&self.map_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "project".into());
                format!("{stem}.mmsproj")
            });
        let start = self
            .current_project_path
            .as_ref()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .or_else(project::settings_dir);
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Halo Map Studio project", &["mmsproj"])
            .set_file_name(&seed)
            .set_title("Save project as");
        if let Some(d) = start {
            dlg = dlg.set_directory(d);
        }
        let Some(dst) = dlg.save_file() else { return };
        let p = self.build_project_record();
        self.status = match project::save_to(&p, &dst) {
            Ok(path) => {
                self.current_project_path = Some(path.clone());
                format!("Saved project -> {}", path.display())
            }
            Err(e) => format!("Save project failed: {e}"),
        };
    }

    /// File ▸ Open Project… — navigate to a specific .mmsproj and load it.
    fn open_project_dialog(&mut self) {
        let start = self
            .current_project_path
            .as_ref()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .or_else(project::settings_dir);
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Halo Map Studio project", &["mmsproj"])
            .set_title("Open project");
        if let Some(d) = start {
            dlg = dlg.set_directory(d);
        }
        let Some(src) = dlg.pick_file() else { return };
        match project::load_from(&src) {
            Ok(p) => {
                self.current_project_path = Some(src.clone());
                self.apply_project(p);
                self.status = format!("Opened project {}", src.display());
            }
            Err(e) => self.status = format!("Open project failed: {e}"),
        }
    }

    /// Load the quick-slot .mmsproj project: restore camera + map + variant.
    fn load_project(&mut self) {
        let Some(p) = project::load() else {
            self.status = "No saved project found.".into();
            return;
        };
        let n = p.objects.len();
        self.apply_project(p);
        self.status = format!("Loaded project ({n} objects recorded).");
    }

    /// Restore camera + map from a project record (shared by the quick slot and Open Project…).
    fn apply_project(&mut self, p: project::Project) {
        self.camera.pos = glam::Vec3::from(p.cam_pos);
        self.camera.yaw = p.cam_yaw;
        self.camera.pitch = p.cam_pitch;
        // The variant + its flag overrides. The variant renders once the base map is
        // up (autoload_variant is consumed by the load-complete tick), and the overrides are
        // re-applied when its meta table exists (render_variant_objects → apply_pending_obj_flags).
        self.pending_obj_flags = (!p.obj_flags.is_empty()).then(|| p.obj_flags.clone());
        let variant = (!p.variant_path.is_empty()).then(|| std::path::PathBuf::from(&p.variant_path));
        if !p.map_path.is_empty() {
            self.autoload_variant = variant;
            if let Some(i) = self
                .map_candidates
                .iter()
                .position(|c| c.path.to_string_lossy().eq_ignore_ascii_case(&p.map_path))
            {
                self.selected_map = Some(i);
            }
            self.map_path = p.map_path.clone();
            self.load_map();
        } else if let Some(v) = variant {
            // No map recorded: open the variant directly (it loads its own base map).
            self.import_mvar(v);
        }
    }

    /// Save the current viewport to captures\viewer_<n>.png (F12).
    fn screenshot(&mut self) {
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("captures")))
            .unwrap_or_else(|| std::path::PathBuf::from("captures"));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("viewer_last.png");
        // Also record the camera so a headless debug run can replicate this exact
        // view (HMS_CAM="x,y,z,yaw,pitch").
        let cam = format!(
            "{},{},{},{},{}",
            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z,
            self.camera.yaw, self.camera.pitch
        );
        let _ = std::fs::write(dir.join("viewer_last.cam"), cam);
        // Alongside the shot, dump the live draw-buffer + overlay state so a captured
        // session can be inspected for which transparent layer holds what.
        let sel = self.selected_set.len();
        let seldat = self.selected_datum.map(|d| format!("{d:#x}")).unwrap_or_else(|| "none".into());
        let counts = self.renderer.debug_draw_counts();
        let diag = format!(
            "cam={},{},{},{},{}\nselected_set={sel} selected_datum={seldat}\nshow_blockers={} show_boundaries={} show_triggers={} show_soft_ceilings={} show_hard_floor={} show_playable_bounds={}\nmvar_objects={} local_objects={}\n{counts}\n",
            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, self.camera.yaw, self.camera.pitch,
            self.show_blockers, self.show_boundaries, self.show_triggers, self.show_soft_ceilings, self.show_hard_floor, self.show_playable_bounds,
            self.mvar_objects.len(), self.local_objects.len(),
        );
        let _ = std::fs::write(dir.join("viewer_last_diag.txt"), diag);
        self.screenshot_to(&path);
    }

    /// Capture the current viewport to an explicit path (headless auto-shot).
    fn screenshot_to(&mut self, path: &std::path::Path) {
        let (rgba, w, h) = self
            .renderer
            .capture_rgba(&self.render_state.device, &self.render_state.queue);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::File::create(path) {
            Ok(file) => {
                let w_buf = std::io::BufWriter::new(file);
                let mut enc = png::Encoder::new(w_buf, w, h);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                match enc.write_header().and_then(|mut wr| wr.write_image_data(&rgba)) {
                    Ok(()) => self.status = format!("Saved screenshot -> {}", path.display()),
                    Err(e) => self.status = format!("Screenshot encode failed: {e}"),
                }
            }
            Err(e) => self.status = format!("Screenshot file failed: {e}"),
        }
    }

}

/// The App drives the shared script runner. `exec_line` parses+executes one command; `enumerate`
/// resolves a foreach source (variants/maps/objects) to concrete items.
impl script::ScriptRunner for App {
    fn exec_line(&mut self, line: &str) -> String {
        match script::parse_line(line) {
            Ok(None) => String::new(),
            Ok(Some(cmd)) => match self.execute_command(cmd) {
                Ok(m) => m,
                Err(e) => format!("ERROR {e}"),
            },
            Err(e) => format!("PARSE {e}"),
        }
    }

    fn enumerate(&mut self, source: &str) -> Result<Vec<script::Item>, String> {
        match script::parse_source(source)? {
            script::Source::Variants { sel, filter } => {
                let (paths, _) = self.variant_listing(&sel, filter.as_deref());
                Ok(paths
                    .into_iter()
                    .map(|p| script::Item {
                        stem: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
                        value: p.to_string_lossy().into_owned(),
                    })
                    .collect())
            }
            script::Source::Maps(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                Ok(self
                    .map_candidates
                    .iter()
                    .filter(|c| f.is_empty() || c.label().to_lowercase().contains(&f))
                    .map(|c| script::Item {
                        stem: c.stem.clone(),
                        value: c.path.to_string_lossy().into_owned(),
                    })
                    .collect())
            }
            script::Source::Objects { type_filter, name_filter, label_filter } => {
                let tf = type_filter.map(|s| s.to_lowercase());
                let nf = name_filter.map(|s| s.to_lowercase());
                let lf = label_filter.map(|s| s.to_lowercase());
                let mut out = Vec::new();
                for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
                    let meta = self.mvar_meta.get(&o.datum);
                    let name = meta.map(|m| m.name.clone()).unwrap_or_default();
                    let label = meta.map(|m| m.label.clone()).unwrap_or_default();
                    let ctype = meta.map(|m| m.cached_type).unwrap_or(0);
                    if let Some(tf) = &tf {
                        let num_ok = tf.parse::<u8>().map(|v| v == ctype).unwrap_or(false);
                        if !num_ok && !name.to_lowercase().contains(tf) {
                            continue;
                        }
                    }
                    if nf.as_ref().map(|nf| !name.to_lowercase().contains(nf)).unwrap_or(false) {
                        continue;
                    }
                    if lf.as_ref().map(|lf| !label.to_lowercase().contains(lf)).unwrap_or(false) {
                        continue;
                    }
                    out.push(script::Item { value: format!("0x{:08X}", o.datum), stem: name });
                }
                Ok(out)
            }
        }
    }
}

impl eframe::App for App {
    /// Scripted pointer (`preview move/click`) appended to the RAW input BEFORE egui's
    /// begin_pass hit-tests it -- the same path a physical mouse takes (hover, click, popup open).
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        self.drive_sim_pointer(raw);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.dbg_pointer = ctx.input(|i| i.pointer.latest_pos()); // `preview where`
        if self.sim_pointer.is_some() || self.sim_click.is_some() || self.sim_drag.is_some() || !self.sim_events.is_empty() {
            ctx.request_repaint(); // keep frames coming while a scripted pointer is pending
        }
        // Print-only frame profiler (HMS_FRAMEPROF=1). Stages: 0 = windows/menus +
        // load/tick/commands (everything before the side panels), 1 = side panels + properties,
        // 2 = particle sim, 3 = render encode, 4 = exposure meter, 5 = viewport input + rest.
        let prof_on = self.frame_prof.is_some();
        if let Some(p) = self.frame_prof.as_mut() { p.0 = std::time::Instant::now(); p.1 = [0.0; 6]; }
        // The Construct TOOL is the single source of truth for construction mode —
        // picking it in the toolbar (or pressing C) arms the guide-drawing clicks. `C` is ignored
        // while a text field has focus so it cannot fire mid-rename.
        if ctx.input(|i| i.key_pressed(egui::Key::C) && !i.modifiers.command && !i.modifiers.ctrl)
            && ctx.memory(|m| m.focused().is_none())
        {
            self.tool_mode = if self.tool_mode == ToolMode::Construct { ToolMode::Select } else { ToolMode::Construct };
        }
        let want_construct = self.tool_mode == ToolMode::Construct;
        if want_construct != self.construct_mode {
            self.construct_mode = want_construct;
            self.construct_pending = None;
            self.construct_hover = None;
            self.overlays_dirty = true;
        }

        // Tools ▸ Model Painter / Import geometry, and Settings ▸ Lighting & rendering windows
        // (the left panel is just the map + its objects).
        if self.show_model_painter {
            let mut open = true;
            egui::Window::new("Model Painter")
                .open(&mut open)
                .default_width(320.0)
                .collapsible(false)
                .show(ctx, |ui| self.model_painter_ui(ui));
            self.show_model_painter = open;
        }
        if self.show_import_geom {
            let mut open = true;
            egui::Window::new("Import geometry")
                .open(&mut open)
                .default_width(360.0)
                .collapsible(false)
                .show(ctx, |ui| self.import_geometry_ui(ui));
            self.show_import_geom = open;
        }
        if self.show_settings {
            let mut open = true;
            egui::Window::new("Settings")
                .open(&mut open)
                .default_width(380.0)
                .default_height(560.0)
                .vscroll(true)
                .show(ctx, |ui| {
                    egui::CollapsingHeader::new("Lighting").default_open(true).show(ui, |ui| {
                        self.lighting_lab_ui(ui);
                    });
                    ui.add_space(6.0);
                    egui::CollapsingHeader::new("Camera & rendering").default_open(true).show(ui, |ui| {
                        self.view_settings_ui(ui);
                    });
                    ui.add_space(6.0);
                    // Every diagnostic readout lives here, closed by default, so the working UI
                    // shows only what a Forger uses.
                    egui::CollapsingHeader::new("Advanced (diagnostics)").default_open(false).show(ui, |ui| {
                        self.advanced_diag_ui(ui);
                    });
                });
            self.show_settings = open;
        }
        // The CAD panel follows its TOOL: it appears while the Construct tool is active, so the
        // controls are there exactly when clicks mean "draw a guide".
        if self.tool_mode == ToolMode::Construct {
            let mut open = true;
            egui::Window::new("Construction (CAD)")
                .open(&mut open)
                .default_width(330.0)
                .default_pos(ctx.screen_rect().right_top() + egui::vec2(-360.0, 90.0))
                .show(ctx, |ui| self.cad_tool_ui(ui));
            if !open {
                self.tool_mode = ToolMode::Select;
            }
        }

        // Themed "open variant by name" browser (self-gates on `variant_browser_open`).
        self.variant_browser_ui(ctx);
        self.scale_converter_ui(ctx);
        self.bsp_warn_window_ui(ctx); // save-time "outside playable space" window
        self.save_error_window_ui(ctx);
        self.h4_save_window_ui(ctx); // the Halo 4 save check (bounds / quota / budget)
        self.drive_lightbake_scan(ctx);
        self.lightbake_map_ui(ctx);
        // Help → Hotkeys reference window (Help menu toggles `show_hotkeys`).
        if self.show_hotkeys {
            let mut open = true;
            // The ONE authoritative list. Every row below matches its handler (update_camera,
            // handle_pick, box-select, update_object_transform, handle_editor_shortcuts,
            // handle_delete_key, handle_selected_edit, handle_undo_redo, the C / F12 checks in
            // update). Keys are only listed with the condition they fire under.
            egui::Window::new("Keyboard shortcuts")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .default_width(520.0)
                .show(ctx, |ui| {
                    let row = |ui: &mut egui::Ui, keys: &str, desc: &str| {
                        ui.horizontal(|ui| {
                            ui.add_sized([150.0, 16.0], egui::Label::new(egui::RichText::new(keys).monospace().strong()));
                            ui.label(desc);
                        });
                    };
                    ui.strong("Camera (hold the RIGHT mouse button over the viewport to fly)");
                    row(ui, "Right-drag", "Look around");
                    row(ui, "W A S D", "Fly forward / left / back / right (RMB held)");
                    row(ui, "E or R", "Fly up (RMB held)");
                    row(ui, "Q or F", "Fly down (RMB held)");
                    row(ui, "Shift (hold)", "4x fly speed (RMB held)");
                    row(ui, "Ctrl + wheel", "Change fly speed (viewport hovered)");
                    row(ui, "F", "Fly the camera to the selected object (not flying, no text field focused)");
                    row(ui, "F12", "Screenshot to the captures folder");
                    ui.separator();
                    ui.strong("Selection (Select tool)");
                    row(ui, "Left-click", "Select the object under the cursor (empty space clears)");
                    row(ui, "Shift + click", "Add / remove the object under the cursor");
                    row(ui, "Left-drag", "Box-select everything inside the rectangle (Shift adds to the selection)");
                    row(ui, "Ctrl+A", "Select every variant / placed object");
                    row(ui, "Shift+A", "Object context menu at the cursor (needs a selection)");
                    row(ui, "Delete / Backspace", "Delete the selection (undoable)");
                    row(ui, "Objects list", "Click = select, Ctrl+click = toggle, Shift+click = range, double-click = fly to");
                    ui.separator();
                    ui.strong("Edit");
                    row(ui, "Ctrl+Z", "Undo");
                    row(ui, "Ctrl+Y or Ctrl+Shift+Z", "Redo");
                    row(ui, "Arrow keys", "Nudge the selection on X / Y by 0.25 wu (Shift = 1 wu)");
                    row(ui, "PageUp / PageDown", "Nudge the selection on Z by 0.25 wu (Shift = 1 wu)");
                    row(ui, "[ / ]", "Yaw the selection by 15 deg (Shift = 45 deg)");
                    row(ui, "Angle / position fields", "Type the sign WHENEVER you like: '90 -' means the same as '-90' (also 90deg, 90 degrees)");
                    ui.separator();
                    ui.strong("Transform (Blender-style, viewport hovered, selection non-empty)");
                    row(ui, "G", "Grab / move the selection (or drag the move gizmo)");
                    row(ui, "R", "Rotate the selection (or drag the rotate gizmo)");
                    row(ui, "Shift+D", "Duplicate the selection and start moving the copies");
                    row(ui, "X / Y / Z", "During G/R: lock to that axis (press again to unlock)");
                    row(ui, "E", "During G/R: toggle world / local axes");
                    row(ui, "Ctrl (hold)", "During G: magnet -- push the selection's nearest real face flush against the nearest other face / map surface, then line the edges up");
                    row(ui, "Ctrl + X/Y/Z", "During G: the magnet may only travel on that axis (both directions)");
                    row(ui, "Shift (hold)", "During G/R: precision (slow) movement");
                    row(ui, "Shift+Ctrl + axis", "During a Shift+D grab: array copies along the locked axis");
                    row(ui, "Wheel (arraying)", "Change the array step: up = spread out, down = overlap (0.25 wu)");
                    row(ui, "Alt + wheel (arraying)", "Fine array step (0.05 wu)");
                    row(ui, "Digits, - .", "During G/R: type an exact distance (wu) / angle (deg); Backspace edits. A '-' typed ANYWHERE flips the sign, so 90- is the same as -90");
                    row(ui, "Click or Enter", "Confirm the transform");
                    row(ui, "Esc or right-click", "Cancel the transform (gizmo drags: Esc only)");
                    ui.separator();
                    ui.strong("Tools");
                    row(ui, "C", "Construct tool on / off (construction guides & CAD panel)");
                    row(ui, "G (Construct)", "Move the selected construction shape; click or Enter places, Esc cancels");
                    row(ui, "Esc (Construct)", "Drop a half-drawn guide / face pick / line array");
                    row(ui, "Line array (Construct)", "Click the line's START then its END; the selection fills it by count or by step (step below the piece size = overlap)");
                    ui.separator();
                    ui.strong("View menu (what is DRAWN)");
                    // #wire-visible
                    row(ui, "Wireframe x-ray", "View > \"Selection wireframe through objects\": draw the selected object's wireframe through anything in front of it (off by default, persisted). The wireframe always has a dark outline under a bright core, so it reads on white Forge pieces and in dark interiors alike. Script: wirexray on|off|get");
                    ui.separator();
                    ui.small(egui::RichText::new("Keys are ignored while a text field has focus. File, Edit, View and Tools actions live in the menu bar: View holds what is DRAWN (geometry: BSP / Terrain / Forge objects / Water / Sky; Forge extras: special FX / scaled objects / shadow casters; overlays: grid, lights, boundary shapes, the selection's collision + physics hulls, spawn points, hidden-block hulls, selection wireframe x-ray, trigger volumes, soft ceilings, hard floor, playable bounds), while Settings keeps the camera and the lighting / render tuning.").weak());
                });
            self.show_hotkeys = open;
        }
        // Post-load memory trim. Once a load has settled, ask the native DLL to return
        // its freed transient decode heap to the OS (_heapmin) and evict the process working
        // set. The renderer only touches GPU-mapped resources afterward, so the ~3GB native
        // parse heap stays out — steady memory drops from ~4.5GB to the ~1GB render set.
        if let Some(t) = self.trim_at {
            if std::time::Instant::now() >= t {
                self.trim_at = None;
                if let Some(s) = self.scene_ctl.as_ref() {
                    // Free the inflate-once page cache (big during load for speed) now that the load
                    // has settled — returns the ~1.5GB of cached pages so steady memory drops to ~200MB.
                    s.clear_page_cache();
                    s.trim_native_heaps();
                }
                #[cfg(windows)]
                unsafe {
                    use windows::Win32::System::ProcessStatus::EmptyWorkingSet;
                    use windows::Win32::System::Threading::GetCurrentProcess;
                    let _ = EmptyWorkingSet(GetCurrentProcess());
                }
                log::info!("post-load memory trim done");
            }
        }
        // Idle release -- 30 s after the last object rebuild (no editing going on), give back
        // the page cache / map pages / allocator slack that the last burst of on-demand decodes
        // pulled in. Runs once per idle period; the next edit simply re-inflates what it needs.
        if !self.idle_released && self.load_rx.is_none() && !self.settle_pending && self.trim_at.is_none()
            && self.last_rebuild_at.elapsed() > std::time::Duration::from_secs(30)
        {
            self.idle_released = true;
            if let Some(s) = self.scene_ctl.as_ref() {
                s.release_burst_memory();
            }
            log::info!("idle memory release done");
        }
        // Interactive load-freeze instrumentation (invisible to headless — there's no window
        // there). While a load is in flight, log any frame that hitches > 60ms, tagged with the
        // current load status, to `hms_load.log` next to the exe. request_repaint keeps frames
        // flowing during load so the hitch clock is real.
        {
            let now = std::time::Instant::now();
            // Pump frames + the hitch clock for the ENTIRE load — the streaming phase AND the
            // post-Done object-model rebuild tail (settle_pending); without the tail coverage the
            // window would only advance on user input while objects pop in. Halo 4 loads too.
            if self.load_rx.is_some() || self.settle_pending || self.h4_load.is_some() {
                // While the load is in flight, return freed transient decode heap to the OS
                // every ~1.2s (native _heapmin + a working-set trim). The decode frees large temp
                // buffers (decompressed pages, per-mesh CPU arrays) into the CRT free-list as it
                // goes; without this they sat committed until the single post-load trim, inflating
                // the peak. Throttled so it never runs more than ~once/second on the hot load path.
                let now2 = std::time::Instant::now();
                if self.next_load_trim.map_or(true, |t| now2 >= t) {
                    self.next_load_trim = Some(now2 + std::time::Duration::from_millis(1200));
                    if let Some(s) = self.scene_ctl.as_ref() {
                        s.trim_native_heaps();
                    }
                    #[cfg(windows)]
                    unsafe {
                        use windows::Win32::System::ProcessStatus::EmptyWorkingSet;
                        use windows::Win32::System::Threading::GetCurrentProcess;
                        let _ = EmptyWorkingSet(GetCurrentProcess());
                    }
                }
                if let Some(prev) = self.last_frame_at {
                    let dt = now.duration_since(prev).as_secs_f64() * 1000.0;
                    if dt > 60.0 {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("hms_load.log") {
                            let _ = writeln!(f, "HITCH {:>6.0}ms | {}", dt, self.map_status);
                        }
                    }
                }
                ctx.request_repaint();
            }
            self.last_frame_at = Some(now);
        }
        // Auto-attach to a running MCC + connect the DLL's shared buffers, so the
        // user never has to attach/inject manually.
        self.auto_attach();
        self.ensure_clients();
        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.screenshot();
        }

        // Top MENU BAR — like a normal desktop app, above everything else.
        egui::TopBottomPanel::top("menubar").show(ctx, |ui| {
            self.menu_bar_ui(ui);
        });

        // Scripting console: the command layer (shared executor with the command server).
        // Multi-line input → run each line → output log; the whole run is one undo step.
        if self.show_script {
            let mut open = true;
            egui::Window::new("Forge Script")
                .open(&mut open)
                .default_width(460.0)
                .default_height(360.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Run").clicked() {
                            let text = self.script_text.clone();
                            self.script_output = self.run_script(&text);
                        }
                        if ui.button("Clear output").clicked() {
                            self.script_output.clear();
                        }
                        ui.label(
                            egui::RichText::new(format!("{} palette objects", self.static_palette.len()))
                                .weak(),
                        );
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .id_salt("script-input")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.script_text)
                                    .font(egui::TextStyle::Monospace)
                                    .code_editor()
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(10),
                            );
                        });
                    ui.separator();
                    ui.label(egui::RichText::new("Output").weak());
                    egui::ScrollArea::vertical()
                        .id_salt("script-output")
                        .max_height(140.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&self.script_output).monospace(),
                                )
                                .wrap(),
                            );
                        });
                });
            self.show_script = open;
        }

        // Stream in a chunk of the map if a load is in flight (keeps UI live).
        self.drive_load(ctx);
        // Read the live object table + rebuild object meshes if it changed.
        self.tick_scene(ctx.input(|i| i.time) as f32);
        // Run any commands queued by external clients (the hms-mcp bridge) this frame.
        self.drain_commands();
        self.frame_prof_lap(0);
        // Post-load settle: once the object-model rebuild tail (spread over frames by the decode
        // budget) has drained, stamp the TRUE end-to-end load time — the map is now fully built
        // AND the window stayed responsive throughout (settle_pending gated request_repaint above).
        if self.settle_pending && self.load_rx.is_none() {
            let tail = self.objscene().map(|s| s.rebuild_pending()).unwrap_or(false);
            if !tail {
                self.settle_pending = false;
                // The object tail's on-demand decodes (models, lighting-probe bitmaps) refill the
                // native page cache after the post-load trim -- give that back now that the map is settled.
                if let Some(s) = self.scene_ctl.as_ref() {
                    s.release_burst_memory();
                }
                let total = self.load_t0.elapsed().as_secs_f32();
                self.map_status = format!("Ready {} in {:.1}s.", self.map_path, total);
                log::info!("map FULLY ready (geometry + objects) in {total:.2}s");
            }
        }
        if self.follow_player {
            self.frame_on_player();
        }
        self.handle_editor_shortcuts(ctx);
        self.draw_context_menu(ctx);
        if !self.handle_delete_key(ctx) {
            self.handle_selected_edit(ctx);
        }
        self.handle_undo_redo(ctx);
        // Refresh selection-dependent overlays (boundary zones) once if selection changed.
        if self.overlays_dirty {
            self.overlays_dirty = false;
            self.rebuild_overlays();
        }
        // Apply a "Path-traced lighting" toggle change once the scene is idle (no load in
        // flight). Deferred out of the egui closure to avoid borrow conflicts; bakes + reloads the BSP.
        if self.rtgi_pending && self.load_rx.is_none() && self.bake_rx.is_none() && self.scene_ctl.is_some() {
            self.rtgi_pending = false;
            self.apply_rtgi_toggle();
        }
        if self.pathtrace_dirty && self.load_rx.is_none() && self.bake_rx.is_none() && self.scene_ctl.is_some() {
            self.pathtrace_dirty = false;
            self.apply_pathtrace_toggle();
        }
        // Pump the path-trace worker (restores the scene + reloads when done).
        self.drive_bake(ctx);

        // (The Blender-style tool toolbar is a FLOATING overlay drawn on the viewport itself —
        // see `floating_toolbar` in the CentralPanel below.)

        egui::SidePanel::left("left").default_width(300.0).show(ctx, |ui| {
          // Whole panel scrolls (drag_to_scroll off so palette-item / preview drags aren't eaten)
          // so the object list + model preview are always reachable no matter the panel height.
          egui::ScrollArea::vertical().auto_shrink([false, false]).drag_to_scroll(false).show(ui, |ui| {
            // Panel order, top to bottom: Map -> Variant -> Objects (search / select)
            // -> Palette (place). Camera speed lives in Settings; diagnostics under
            // Settings > Advanced; the tools in the menu bar.
            ui.heading("Map");
            if self.map_candidates.is_empty() {
                ui.small("No maps detected (is MCC/Halo Reach installed via Steam?).");
            }
            // #map-picker: TWO levels. First the GAME (only titles with caches on this machine,
            // with counts), then Built-in vs Modded WITHIN that game -- so every title gets the
            // same split and a new title needs no new tab. The dropdown below shows only the
            // chosen (game, kind) group; built-in is the default, so Workshop maps never clutter
            // the stock list.
            let n_of = |g: mapcat::Game, modded: bool| -> usize {
                self.map_candidates.iter().filter(|c| c.game == g && c.modded == modded).count()
            };
            let games: Vec<mapcat::Game> = [mapcat::Game::Reach, mapcat::Game::Halo4, mapcat::Game::H2A]
                .into_iter()
                .filter(|g| n_of(*g, false) + n_of(*g, true) > 0)
                .collect();
            // Never leave the picker pointed at a game with nothing in it (none detected yet, or
            // a rescan dropped one): fall back to the first game that has maps.
            if !games.is_empty() && !games.contains(&self.picker_game) {
                self.picker_game = games[0];
                self.selected_map = None;
            }
            if games.len() > 1 {
                ui.horizontal_wrapped(|ui| {
                    for g in &games {
                        let total = n_of(*g, false) + n_of(*g, true);
                        if ui
                            .selectable_label(self.picker_game == *g, format!("{} ({total})", g.display_name()))
                            .on_hover_text(format!("Maps found in {}\\maps and in Steam Workshop items for that title", g.folder()))
                            .clicked()
                        {
                            self.picker_game = *g;
                        }
                    }
                });
            }
            let n_builtin = n_of(self.picker_game, false);
            let n_modded = n_of(self.picker_game, true);
            // With only one kind present there is nothing to choose -- show no split and point
            // the list at the kind that exists.
            if n_builtin == 0 && n_modded > 0 { self.show_modded_maps = true; }
            if n_modded == 0 && n_builtin > 0 { self.show_modded_maps = false; }
            if n_builtin > 0 && n_modded > 0 {
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(!self.show_modded_maps, format!("Built-in ({n_builtin})"))
                        .on_hover_text("Maps in the game's own maps folder (your own copies there count as built-in)")
                        .clicked()
                    {
                        self.show_modded_maps = false;
                    }
                    if ui
                        .selectable_label(self.show_modded_maps, format!("Modded ({n_modded})"))
                        .on_hover_text("Subscribed Steam Workshop maps (anything in the game's own maps folder counts as built-in)")
                        .clicked()
                    {
                        self.show_modded_maps = true;
                    }
                });
            }
            ui.horizontal(|ui| {
                let want_game = self.picker_game;
                let want_modded = self.show_modded_maps;
                // Clear a selection that belongs to another group so the label stays honest.
                if let Some(i) = self.selected_map {
                    if self.map_candidates.get(i).map(|c| (c.game, c.modded)) != Some((want_game, want_modded)) {
                        self.selected_map = None;
                    }
                }
                let sel_label = self
                    .selected_map
                    .and_then(|i| self.map_candidates.get(i))
                    .map(|c| c.label())
                    .unwrap_or_else(|| "Select a map…".to_string());
                egui::ComboBox::from_id_salt("map-picker")
                    .selected_text(sel_label)
                    .width(210.0)
                    .show_ui(ui, |ui| {
                        let mut any = false;
                        for (i, c) in self.map_candidates.iter().enumerate() {
                            if c.game != want_game || c.modded != want_modded { continue; } // #map-picker: game AND kind
                            any = true;
                            if ui.selectable_label(self.selected_map == Some(i), c.label()).clicked() {
                                self.selected_map = Some(i);
                            }
                        }
                        if !any {
                            ui.small(format!(
                                "No {} {} maps found.",
                                if want_modded { "modded" } else { "built-in" },
                                want_game.display_name()
                            ));
                        }
                    });
                if ui.button("↻").on_hover_text("Rescan for maps").clicked() {
                    self.map_candidates = mapcat::enumerate();
                    self.selected_map = None;
                }
            });
            ui.horizontal(|ui| {
                let loaded = self.objscene().map(|s| s.has_cache()).unwrap_or(false);
                if ui.button("Load selected map").clicked() {
                    self.load_map();
                }
                // ("Rebuild forge objects" — objects only — is a diagnostic under Settings > Advanced.)
                if loaded && ui.button("Reload").on_hover_text("Re-open the map cache and rebuild everything").clicked() {
                    self.load_map();
                }
            });
            ui.separator();
            // Single consolidated Map-Variant area (auto-loads the variant's base map, then
            // renders its forge objects offline). This is the ONE place .mvar loading lives.
            ui.heading("Variant");
            // File/Open/Save/Save-As live in the TOP menu bar (see `menu_bar_ui`). Here we
            // show which variant is currently open plus one-click Open / Save.
            ui.horizontal(|ui| {
                if let Some(p) = &self.current_variant_path {
                    ui.label(p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())
                        .on_hover_text(p.to_string_lossy());
                } else if self.new_variant_template.is_some() {
                    ui.label("(new, unsaved variant)");
                } else {
                    ui.label(egui::RichText::new("No variant open").weak());
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Open…").on_hover_text("Browse and open a .mvar (File > Open .mvar…)").clicked() {
                    self.open_variant_browser();
                }
                let has = self.current_variant_path.is_some() || self.new_variant_template.is_some();
                // both games get Save + Save As
                if ui.add_enabled(has, egui::Button::new("Save")).on_hover_text("Write the edited objects back into the open .mvar").clicked() {
                    self.save_current_variant();
                }
                if ui.add_enabled(has, egui::Button::new("Save As…")).clicked() {
                    self.save_variant_as();
                }
            });
            // Variant Info — view/edit the four header strings; persisted on File ▸ Save. Only
            // enabled when a variant is open. Editing any field marks the header dirty so Save
            // rewrites it (otherwise the header is copied verbatim, byte-exact).
            if self.current_variant_path.is_some() || self.new_variant_template.is_some() {
                egui::CollapsingHeader::new("Variant Info").default_open(true).show(ui, |ui| {
                    let mut edited = false;
                    egui::Grid::new("variant-info").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                        ui.label("Name");
                        edited |= ui
                            .add(egui::TextEdit::singleline(&mut self.variant_title).desired_width(190.0).hint_text("variant name"))
                            .changed();
                        ui.end_row();
                        ui.label("Description");
                        edited |= ui
                            .add(egui::TextEdit::singleline(&mut self.variant_description).desired_width(190.0))
                            .changed();
                        ui.end_row();
                        ui.label("Author");
                        edited |= ui
                            .add(egui::TextEdit::singleline(&mut self.variant_author).desired_width(190.0).char_limit(16))
                            .on_hover_text("CreatedBy display name (max 16 chars)")
                            .changed();
                        ui.end_row();
                        ui.label("Editor");
                        edited |= ui
                            .add(egui::TextEdit::singleline(&mut self.variant_editor).desired_width(190.0).char_limit(16))
                            .on_hover_text("ModifiedBy display name (max 16 chars)")
                            .changed();
                        ui.end_row();
                    });
                    if edited {
                        self.variant_header_dirty = true;
                    }
                    if self.variant_header_dirty {
                        ui.small("Edited — File > Save will write these.");
                    }
                });

                // Variant properties: every global field of the file (engine names on
                // hover), the editable ones bound to `global_edits` (category, maximum budget,
                // world bounds behind a warning, quota min / max per palette entry).
                self.variant_properties_ui(ui);

                // Forge labels — the named tags objects reference via `label_idx`. Create/remove them
                // here; File ▸ Save re-encodes the variant's forge-label string table. Added labels are
                // immediately selectable from each object's Label dropdown in the properties panel.
                egui::CollapsingHeader::new("Forge labels").default_open(false).show(ui, |ui| {
                    let mut remove: Option<usize> = None;
                    let mut changed = false;
                    if self.mvar_labels.is_empty() {
                        ui.small("No forge labels in this variant.");
                    }
                    for i in 0..self.mvar_labels.len() {
                        ui.horizontal(|ui| {
                            ui.small(format!("#{i}"));
                            changed |= ui
                                .add(egui::TextEdit::singleline(&mut self.mvar_labels[i]).desired_width(160.0))
                                .changed();
                            if ui.small_button("🗑").on_hover_text("Remove this label").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        self.mvar_labels.remove(i);
                        changed = true;
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        let te = ui.add(
                            egui::TextEdit::singleline(&mut self.mvar_new_label)
                                .desired_width(160.0)
                                .hint_text("new label…"),
                        );
                        let submit = te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        // Cap the table at 511 entries (the 9-bit count field's limit).
                        let can_add = !self.mvar_new_label.trim().is_empty() && self.mvar_labels.len() < 511;
                        if ui.add_enabled(can_add, egui::Button::new("Add label")).clicked() || (submit && can_add) {
                            self.mvar_labels.push(self.mvar_new_label.trim().to_string());
                            self.mvar_new_label.clear();
                            changed = true;
                        }
                    });
                    if changed {
                        self.variant_header_dirty = true;
                    }
                    if self.variant_header_dirty {
                        ui.small("Edited — File > Save will write the label table.");
                    }
                });
            }
            ui.separator();

            // OBJECTS IN THE VARIANT -- a searchable, selectable list of every forge
            // object in the scene (variant + placed). No refresh button: it is derived from the
            // live object collections every frame (cached by a fingerprint, see objects_list_ui).
            self.objects_list_ui(ui);
            ui.separator();

            // Sandbox palette: objects enumerated from the scenario. Selecting one
            // makes the drag-place / Line / Fill tools drop a locally-rendered object (no live
            // game needed). Rendered immediately by the object system on next tick.
            // Always visible (not a collapsing dropdown) — it's a core feature.
            ui.add_space(4.0);
            ui.heading(format!("Palette ({})", self.static_palette.len()));
            {
                // Controls.
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.pal_filter).hint_text("search the palette…").desired_width(190.0));
                    if !self.pal_filter.is_empty() && ui.small_button("clear").clicked() {
                        self.pal_filter.clear();
                    }
                });
                ui.small("Drag an item (or the preview) into the viewport to place · Ctrl on drop = snap to a face.");
                if self.line_mode || self.fill_mode {
                    let which = if self.line_mode { "Line tool" } else { "Fill tool" };
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 200, 90),
                        format!("{which} is on: clicks in the viewport place the selected item (Tools menu to turn it off)."),
                    );
                }

                let filter = self.pal_filter.to_lowercase();
                // List (grouped by category) on the LEFT, model preview ALWAYS visible on the RIGHT
                // so you never have to scroll to see it.
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(210.0);
                        egui::ScrollArea::vertical()
                            .id_salt("sandbox-pal")
                            .max_height(380.0)
                            .drag_to_scroll(false)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                // Group by the REAL forge CATEGORY name from the scenario
                                // palette (e.g. "ff_weapons_covenant" → "Covenant Weapons"),
                                // preserving walk order so each category's items sit together.
                                let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
                                for i in 0..self.static_palette.len() {
                                    if !filter.is_empty() && !self.static_palette[i].1.to_lowercase().contains(&filter) {
                                        continue;
                                    }
                                    let cat = self.static_palette_catname.get(i).cloned().unwrap_or_default();
                                    match groups.last_mut() {
                                        Some((c, v)) if *c == cat => v.push(i),
                                        _ => groups.push((cat, vec![i])),
                                    }
                                }
                                for (cat, idxs) in &groups {
                                    let cname = forge_category_pretty(cat);
                                    ui.add_space(4.0);
                                    ui.label(egui::RichText::new(cname).strong().color(egui::Color32::from_gray(150)));
                                    for &i in idxs {
                                        let full = self.static_palette[i].1.clone();
                                        let leaf = full.split(" · ").last().unwrap_or(&full).to_string();
                                        let sel = self.selected_static_pal == Some(i);
                                        // Click selects. A second interaction over the SAME rect with a
                                        // STABLE id senses the drag (drag-out into the viewport), so the
                                        // drag survives list reflow (auto-id churn would break it).
                                        let r = ui.selectable_label(sel, format!("  {leaf}"));
                                        if r.clicked() {
                                            self.selected_static_pal = Some(i);
                                        }
                                        let dr = ui.interact(r.rect, egui::Id::new(("paldrag", i)), egui::Sense::drag());
                                        if dr.drag_started() {
                                            self.selected_static_pal = Some(i);
                                            self.pending_place = Some(i);
                                        }
                                    }
                                }
                            });
                    });
                    // Model preview beside the list (render-model of the selected object).
                    ui.vertical(|ui| {
                        self.show_preview(ui, ctx);
                    });
                });
            }
            // Live-game player tracking (injection build only).
            #[cfg(feature = "injection")]
            {
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Frame on player").clicked() {
                        self.frame_on_player();
                    }
                    ui.checkbox(&mut self.follow_player, "Follow");
                });
                if ui.button("TP player -> camera").clicked() {
                    self.teleport_player_to(self.camera.pos);
                }
            }
          }); // end left-panel ScrollArea
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.small(&self.status);
                // Which special-FX screen effects are active (and whether they are shown).
                if !self.active_screenfx.forge_names.is_empty() {
                    ui.separator();
                    let txt = self.screenfx_status();
                    let lbl = if self.forge_fx_enabled { egui::RichText::new(txt).small() } else { egui::RichText::new(txt).small().weak() };
                    ui.label(lbl).on_hover_text("View > Forge extras > Forge special FX, or script `screenfx on|off`");
                }
            });
        });

        // Properties panel for the selected forge object (team/color/pose/delete).
        // The team/colour dropdown entry under the cursor this frame (None = nothing
        // hovered or the popup is closed) -- reconciled AFTER the window so the preview ends on the
        // very frame the popup closes, however the mouse left.
        let mut hovered_entry: Option<color_hover::HoverEntry> = None;
        // The OPEN team/colour popup whose rect contains the pointer (None = none). The
        // hide rule keys on this, not on the entry, so crossing the padding between rows never flickers.
        let mut hovered_popup: Option<color_hover::Popup> = None;
        if let Some(datum) = self.selected_datum {
            let row = self.forge_rows.iter().find(|(_, d, _, _)| *d == datum).copied();
            // Sync edit buffers when the selection changes.
            if self.props_for != Some(datum) {
                self.props_for = Some(datum);
                // Material inspector info for the selected object's render model (resolved
                // once here, not every frame -- it opens the model in the native DLL).
                self.props_mat = self
                    .last_objects
                    .iter()
                    .find(|o| o.datum == datum)
                    .map(|o| o.mode_tag)
                    .filter(|&t| t != 0 && t != 0xFFFF_FFFF)
                    .and_then(|t| self.scene_ctl.as_ref().and_then(|s| s.object_material(t)));
                self.prop_snapshotted_for = None; // new selection -> next edit starts a new undo step
                if let Some((_, _, t, c)) = row {
                    self.edit_team = t;
                    self.edit_color = c;
                }
            }
            // The properties panel is a FLOATING window anchored top-right, draggable +
            // resizable: it overlays the viewport instead of resizing it (a SidePanel would
            // shrink the viewport every time a forge object is clicked). Opened FULLY EXPANDED
            // vertically so every forge option is visible without scrolling — fill the viewport
            // height from the anchor down. vscroll stays as a safety net for very short windows.
            let panel_h = (ctx.available_rect().height() - 52.0).max(240.0);
            egui::Window::new("Object")
                .anchor(egui::Align2::RIGHT_TOP, [-8.0, 40.0])
                .default_width(270.0)
                .default_height(panel_h)
                .max_height(panel_h)
                .resizable(true)
                .collapsible(true)
                .vscroll(true)
                .show(ctx, |ui| {
                // "outside playable space" strip, live per frame for the selection.
                self.bsp_warn_panel_strip(ui, datum);
                // This panel is the editable forge record only (the raw datum / forge index /
                // material readouts are under Settings > Advanced). Edits mutate the in-memory
                // record (written by File > Save); team/color/position also drive the live
                // render immediately.
                if let Some(m0) = self.mvar_meta.get(&datum).cloned() {
                    // Say plainly how many objects these fields will hit. A panel that
                    // silently edits 40 objects is as bad as one that silently edits 1.
                    let n_sel = self.selected_set.iter().filter(|d| self.mvar_meta.contains_key(d)).count();
                    if n_sel > 1 {
                        ui.strong(egui::RichText::new(format!("Forge object — editing {n_sel} objects"))
                            .color(egui::Color32::from_rgb(255, 200, 90)));
                        ui.small("Changes below apply to ALL selected. Position moves them together (keeps their layout).");
                    } else {
                        ui.strong("Forge object");
                    }
                    if !m0.name.is_empty() {
                        let leaf = m0.name.rsplit(['\\', '/']).next().unwrap_or(&m0.name).to_string();
                        // Show the prettified display name; hover reveals the raw stringID/path.
                        ui.small(format!("item:  {}", prettify_stringid(&leaf))).on_hover_text(&m0.name);
                    }
                    // The obje tag behind the palette name (smaller line + copy button).
                    self.identity_rows_ui(ui, datum, false);
                    // Forge label STRING (from the variant's label table), if any.
                    if !m0.label.is_empty() {
                        ui.small(format!("label:  \"{}\"", m0.label));
                    } else if m0.label_idx != 0xFFFF {
                        ui.small(format!("label:  #{}", m0.label_idx));
                    }
                    ui.add_space(2.0);
                    let mut m = m0.clone();
                    egui::Grid::new("mvar-edit").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                        ui.label("quota/variant");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut m.folder).range(0..=255));
                            ui.add(egui::DragValue::new(&mut m.item).range(0..=31));
                        });
                        ui.end_row();

                        // team: i32 proxy — -1 none, 0..7 team colour, 8 neutral. Shown with the
                        // actual change-colour swatch + name (not a bare index).
                        ui.label("team");
                        let mut team_i = if m.team == 0xFF { -1 } else { m.team as i32 };
                        // 8 = the game's NEUTRAL team (what every new placement gets, and
                        // what "no team" means in the Forge menu); -1 = the .mvar's distinct "none" value.
                        let team_label = |t: i32| match t {
                            8 => "neutral (no team)".to_string(),
                            -1 => "none (unset)".to_string(),
                            t => forge_team_name(t),
                        };
                        let team_txt = egui::RichText::new(format!("⬛ {}", team_label(team_i))).color(forge_swatch32(team_i));
                        // "inside the popup" is decided from INSIDE the closure (it only runs
                        // while the popup is open) on the popup's own ui; the entry under the pointer via
                        // each row's response. The rects are kept for the `preview where` debug hook.
                        let team_ir = egui::ComboBox::from_id_salt("mvar-team")
                            .selected_text(team_txt)
                            .show_ui(ui, |ui| {
                                // Report the hovered entry (raw team byte) for the preview.
                                let mut entry = |t: i32, r: egui::Response| {
                                    if r.hovered() { hovered_entry = Some(color_hover::HoverEntry::Team(if t < 0 { 0xFF } else { t as u8 })); }
                                };
                                entry(8, ui.selectable_value(&mut team_i, 8, egui::RichText::new(format!("⬛ {}", team_label(8))).color(forge_swatch32(8))));
                                for t in 0..=7 {
                                    entry(t, ui.selectable_value(&mut team_i, t, egui::RichText::new(format!("⬛ {}", team_label(t))).color(forge_swatch32(t))));
                                }
                                entry(-1, ui.selectable_value(&mut team_i, -1, egui::RichText::new(format!("⬛ {}", team_label(-1))).color(forge_swatch32(-1))));
                                let (inside, rect) = popup_ui_contains_pointer(ui);
                                if inside { hovered_popup = Some(color_hover::Popup::Team); }
                                self.hover_debug_rects.team_popup = Some(rect);
                            });
                        self.hover_debug_rects.team_button = Some(team_ir.response.rect);
                        if team_ir.inner.is_none() { self.hover_debug_rects.team_popup = None; }
                        m.team = if team_i < 0 { 0xFF } else { team_i as u8 };
                        ui.end_row();

                        // color: -1 inherit-team, 0..7 change-colour. Swatch + name; "inherit" shows
                        // the colour it resolves to (the team's colour).
                        ui.label("color");
                        let mut col_i = m.color;
                        let sel_sw = if col_i < 0 { forge_swatch32(team_i) } else { forge_swatch32(col_i) };
                        let sel_txt = if col_i < 0 { format!("⬛ inherit ({})", forge_team_name(team_i)) } else { format!("⬛ {}", forge_team_name(col_i)) };
                        let color_ir = egui::ComboBox::from_id_salt("mvar-color")
                            .selected_text(egui::RichText::new(sel_txt).color(sel_sw))
                            .show_ui(ui, |ui| {
                                // Hovering "inherit" previews the team colour; a swatch previews itself.
                                let mut entry = |c: i32, r: egui::Response| {
                                    if r.hovered() { hovered_entry = Some(color_hover::HoverEntry::Color(c)); }
                                };
                                entry(-1, ui.selectable_value(&mut col_i, -1, "inherit (use team)"));
                                for c in 0..=7 { entry(c, ui.selectable_value(&mut col_i, c, egui::RichText::new(format!("⬛ {}", forge_team_name(c))).color(forge_swatch32(c)))); }
                                let (inside, rect) = popup_ui_contains_pointer(ui);
                                if inside { hovered_popup = Some(color_hover::Popup::Color); }
                                self.hover_debug_rects.color_popup = Some(rect);
                            });
                        self.hover_debug_rects.color_button = Some(color_ir.response.rect);
                        if color_ir.inner.is_none() { self.hover_debug_rects.color_popup = None; }
                        m.color = col_i;
                        ui.end_row();

                        // A Halo 4 object shows its own block (object type read-only,
                        // spawn order 0..255, spawn time) instead of the four Reach-only rows below.
                        if m.h4.is_some() {
                            self.h4_rows_type_spawn(ui, &mut m);
                        } else { // the SCALED / SHADOW flags row below is shared with Halo 4
                        // Cached (multiplayer object) type as a NAMED dropdown of the engine-
                        // anchored categories, with a raw DragValue alongside for the unlabeled values.
                        ui.label("cached type");
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("mvar-cachedtype")
                                .selected_text(mvar::cached_type_name(m.cached_type))
                                .show_ui(ui, |ui| {
                                    for &t in &[0u8, 1, 2, 12, 13, 14, 19] {
                                        ui.selectable_value(&mut m.cached_type, t, mvar::cached_type_name(t));
                                    }
                                });
                            ui.add(egui::DragValue::new(&mut m.cached_type).range(0..=31))
                                .on_hover_text("raw cached_type (0..31) — for values without an engine-anchored name");
                        });
                        ui.end_row();
                        // ALWAYS edit the raw spawn sequence directly (the actual saved field).
                        // For a "scale"-labelled object, show the resulting X330 scale multiplier NEXT
                        // to it (live, read-only) so the user sees the size their spawn_seq encodes —
                        // rather than the input converting itself to a scale field.
                        ui.label("spawn seq");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut m.spawn_seq).range(-100..=100));
                            if m.scaled_on() {
                                let txt = format!("-> ×{:.3}", forge_scale::object_scale(m.spawn_seq, m.team));
                                if self.obj_globals.scaled { ui.small(txt); } else { ui.small(egui::RichText::new(txt).weak()).on_hover_text("Scaled objects is OFF in Settings — rendered at ×1"); }
                            }
                        });
                        ui.end_row();
                        } // end of the Reach-only type / spawn seq rows

                        // The two PSEUDO-flags (never saved into the .mvar; kept in the
                        // project). Each is an override over a live-derived default, so the box shows
                        // the EFFECTIVE state; "(auto)" = following the rule, "(set)" = user override.
                        // With a multi-selection a box whose objects disagree draws indeterminate;
                        // clicking sets every selected object (apply_changed_fields fans it out).
                        ui.label("flags");
                        ui.vertical(|ui| {
                            let sel_metas: Vec<&ObjMeta> = self.selected_set.iter()
                                .filter(|d| **d != datum)
                                .filter_map(|d| self.mvar_meta.get(d))
                                .collect();
                            let mixed_scaled = sel_metas.iter().any(|t| t.scaled_on() != m0.scaled_on());
                            let mixed_shadow = sel_metas.iter().any(|t| t.shadow_on() != m0.shadow_on());
                            let tag = |explicit: bool| if explicit { "(set)" } else { "(auto)" };
                            let mut scaled = m.scaled_on();
                            let r = ui.add(egui::Checkbox::new(&mut scaled, format!("SCALED {}", tag(m.flags.scaled.is_some()))).indeterminate(mixed_scaled))
                                .on_hover_text("Pseudo-flag (not saved in the .mvar): read this object's spawn sequence as an X330 size, like the forge \"scale\" label does. Default ON when the label is \"scale\". Right-click: back to automatic.");
                            if r.clicked() { m.flags.scaled = Some(if mixed_scaled { true } else { scaled }); }
                            if r.secondary_clicked() { m.flags.scaled = None; }
                            let mut shadow = m.shadow_on();
                            let r = ui.add(egui::Checkbox::new(&mut shadow, format!("SHADOW {}", tag(m.flags.shadow.is_some()))).indeterminate(mixed_shadow))
                                .on_hover_text("Pseudo-flag (not saved in the .mvar): draw this object into the sun shadow map so it casts a shadow over the ground and other objects. Default ON for the gametype rule GREEN team + \"scale\" label. Vehicles and other engine-default casters always cast. Right-click: back to automatic.");
                            if r.clicked() { m.flags.shadow = Some(if mixed_shadow { true } else { shadow }); }
                            if r.secondary_clicked() { m.flags.shadow = None; }
                            if !self.obj_globals.scaled || !self.obj_globals.shadowcasters {
                                ui.small(egui::RichText::new(format!("global: {}", self.obj_globals.status_line())).weak())
                                    .on_hover_text("Settings ▸ Lighting & rendering ▸ Forge objects");
                            }
                        });
                        ui.end_row();
                        if m.h4.is_none() {
                        ui.label("respawn (s)");
                        ui.add(egui::DragValue::new(&mut m.respawn));
                        ui.end_row();
                        } // end of the Reach-only rows

                        // label: pick by NAME from the variant's label table (falls back to a raw
                        // index when the table is empty / compressed). Shows the string id (index).
                        ui.label("label");
                        let cur_name = if m.label_idx == 0xFFFF {
                            "none".to_string()
                        } else if !m.label.is_empty() {
                            format!("{}  (#{})", m.label, m.label_idx)
                        } else {
                            format!("#{}", m.label_idx)
                        };
                        ui.vertical(|ui| {
                            egui::ComboBox::from_id_salt("mvar-label")
                                .selected_text(cur_name)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(m.label_idx == 0xFFFF, "none").clicked() {
                                        m.label_idx = 0xFFFF;
                                        m.label.clear();
                                    }
                                    for (i, name) in self.mvar_labels.iter().enumerate() {
                                        let disp = if name.is_empty() { format!("#{i}") } else { format!("{name}  (#{i})") };
                                        if ui.selectable_label(m.label_idx as usize == i, disp).clicked() {
                                            m.label_idx = i as u16;
                                            m.label = name.clone();
                                        }
                                    }
                                });
                            // Inline "name this object" — create a NEW label in the variant table
                            // AND assign it to this object in one step, so vehicles/objects (mongoose,
                            // etc.) can be named exactly like spawn points without first pre-populating
                            // the table from the left "Forge labels" panel.
                            ui.horizontal(|ui| {
                                let te = ui.add(
                                    egui::TextEdit::singleline(&mut self.mvar_new_label)
                                        .desired_width(110.0)
                                        .hint_text("name…"),
                                );
                                let submit = te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                let can_add = !self.mvar_new_label.trim().is_empty() && self.mvar_labels.len() < 511;
                                if ui.add_enabled(can_add, egui::Button::new("+ name")).clicked() || (submit && can_add) {
                                    let name = self.mvar_new_label.trim().to_string();
                                    // Reuse an identical existing label if present; else append a new one.
                                    let idx = self.mvar_labels.iter().position(|l| l == &name)
                                        .unwrap_or_else(|| { self.mvar_labels.push(name.clone()); self.mvar_labels.len() - 1 });
                                    m.label_idx = idx as u16;
                                    m.label = name;
                                    self.mvar_new_label.clear();
                                    self.variant_header_dirty = true;
                                }
                            });
                        });
                        ui.end_row();
                        // Halo 4 objects carry up to FOUR labels (labels 2-4 here).
                        self.h4_rows_labels(ui, &mut m);

                        // Pick a PARENT object (spawn-relative-to). The engine groups
                        // the child under the parent (cascade-delete + attachment); the child's own
                        // transform stays absolute. Candidates are objects with a known .mvar slot
                        // (loaded originals); stores the parent's SLOT as this object's spawn_rel.
                        ui.label("parent");
                        let cur_parent = if m.spawn_rel < 0 {
                            "none".to_string()
                        } else {
                            self.mvar_meta.values().find(|pm| pm.slot as i32 == m.spawn_rel)
                                .map(|pm| if pm.name.is_empty() { format!("slot {}", m.spawn_rel) } else { format!("{} (slot {})", pm.name, m.spawn_rel) })
                                .unwrap_or_else(|| format!("slot {}", m.spawn_rel))
                        };
                        egui::ComboBox::from_id_salt("mvar-parent")
                            .selected_text(cur_parent)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(m.spawn_rel < 0, "none").clicked() { m.spawn_rel = -1; }
                                let mut cands: Vec<(u16, String)> = self.mvar_meta.iter()
                                    .filter(|(&d, pm)| d != datum && pm.slot != 0xFFFF)
                                    .map(|(_, pm)| (pm.slot, if pm.name.is_empty() { format!("slot {}", pm.slot) } else { format!("{} (slot {})", pm.name, pm.slot) }))
                                    .collect();
                                cands.sort_by_key(|(s, _)| *s);
                                for (slot, disp) in cands {
                                    if ui.selectable_label(m.spawn_rel == slot as i32, disp).clicked() {
                                        m.spawn_rel = slot as i32;
                                    }
                                }
                            });
                        ui.end_row();

                        // Placement flags — named per the engine's own labels (reach_tag_test
                        // `scenario_map_variant.cpp`): physics (bits6-7), symmetry (bits2-3, as the
                        // `asymmetric_placement`/`symmetric_placement` pair), placed-at-start (bit4
                        // `not_initially_placed`, INVERTED), hide-unless-required (bit0), unique (bit5).
                        ui.label("placement");
                        ui.vertical(|ui| {
                            // Physics: Normal / Fixed / Phased.
                            ui.horizontal(|ui| {
                                ui.label("physics");
                                let mut phys = m.placement & mvar::PLACE_PHYSICS_MASK;
                                egui::ComboBox::from_id_salt("mvar-phys")
                                    .selected_text(placement_physics_name(m.placement))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut phys, 0b0000_0000, "normal");
                                        ui.selectable_value(&mut phys, 0b0100_0000, "fixed");
                                        ui.selectable_value(&mut phys, 0b1100_0000, "phased");
                                    });
                                m.placement = (m.placement & !mvar::PLACE_PHYSICS_MASK) | phys;
                            });
                            // Symmetry: Never / Symmetric / Asymmetric / Both (bits2-3).
                            ui.horizontal(|ui| {
                                ui.label("symmetry");
                                let mut sym = m.placement & mvar::PLACE_SYMMETRY_MASK;
                                let sym_txt = match sym {
                                    0x0C => "both",
                                    mvar::PLACE_SYMMETRIC => "symmetric",
                                    mvar::PLACE_ASYMMETRIC => "asymmetric",
                                    _ => "never",
                                };
                                egui::ComboBox::from_id_salt("mvar-sym")
                                    .selected_text(sym_txt)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut sym, 0u8, "never");
                                        ui.selectable_value(&mut sym, mvar::PLACE_SYMMETRIC, "symmetric");
                                        ui.selectable_value(&mut sym, mvar::PLACE_ASYMMETRIC, "asymmetric");
                                        ui.selectable_value(&mut sym, 0x0Cu8, "both");
                                    });
                                m.placement = (m.placement & !mvar::PLACE_SYMMETRY_MASK) | sym;
                            });
                            // Placed at start (bit4 `not_initially_placed`, INVERTED).
                            let mut at_start = m.placement & mvar::PLACE_NOT_AT_START == 0;
                            if ui.checkbox(&mut at_start, "placed at start")
                                .on_hover_text("Object spawns when the match starts (engine bit4 not_initially_placed, inverted)")
                                .changed()
                            {
                                if at_start { m.placement &= !mvar::PLACE_NOT_AT_START; }
                                else { m.placement |= mvar::PLACE_NOT_AT_START; }
                            }
                            let mut bit = |ui: &mut egui::Ui, mask: u8, label: &str, hover: &str| {
                                let mut on = m.placement & mask != 0;
                                if ui.checkbox(&mut on, label).on_hover_text(hover).changed() {
                                    if on { m.placement |= mask; } else { m.placement &= !mask; }
                                }
                            };
                            bit(ui, mvar::PLACE_HIDE_UNLESS_REQUIRED, "game-type specific",
                                "hide_unless_required (bit0): only appears when a gametype/megalo script requires this object type");
                            bit(ui, mvar::PLACE_UNIQUE_SPAWN, "unique spawn",
                                "unique_spawn (bit5)");
                            bit(ui, mvar::PLACE_IS_SHORTCUT, "shortcut",
                                "is_shortcut (bit1)");
                        });
                        ui.end_row();
                        // The two raw Halo 4 placement bits, then the Halo 4 shape /
                        // type extras / scale / raw rows replace the Reach boundary + extras rows.
                        if m.h4.is_some() {
                            self.h4_rows_placement_hi(ui, &mut m);
                            self.h4_rows_shape(ui, &mut m);
                            self.h4_rows_type_extras(ui, &mut m);
                            self.h4_rows_scale_raw(ui, &mut m);
                        } else {
                        ui.label("boundary");
                        let shapes = ["none", "sphere", "cylinder", "box"];
                        egui::ComboBox::from_id_salt("mvar-shape")
                            .selected_text(shapes[(m.boundary_shape as usize).min(3)])
                            .show_ui(ui, |ui| {
                                for (i, s) in shapes.iter().enumerate() {
                                    ui.selectable_value(&mut m.boundary_shape, i as u8, *s);
                                }
                            });
                        ui.end_row();
                        let nvals = [0usize, 1, 3, 4][(m.boundary_shape as usize).min(3)];
                        if nvals > 0 {
                            ui.label("boundary vals");
                            ui.horizontal(|ui| {
                                for k in 0..nvals {
                                    ui.add(egui::DragValue::new(&mut m.boundary[k]).range(0..=2047));
                                }
                            });
                            ui.end_row();
                        }

                        // cached-type extras (weapon clips / teleporter chan+pass / location name).
                        if m.cached_type == 1 {
                            ui.label("weapon clips");
                            ui.add(egui::DragValue::new(&mut m.weapon_clips));
                            ui.end_row();
                        } else if (12..=14).contains(&m.cached_type) {
                            // Teleporter CHANNEL = NATO phonetic (0=Alpha..25=Zulu); teleporters
                            // link to others on the SAME channel (sender→receiver / 2-way↔2-way).
                            const NATO: [&str; 26] = [
                                "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
                                "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
                                "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
                                "Xray", "Yankee", "Zulu",
                            ];
                            ui.label("channel");
                            egui::ComboBox::from_id_salt("mvar-telechan")
                                .selected_text(NATO.get(m.tele_channel as usize).copied().unwrap_or("?"))
                                .show_ui(ui, |ui| {
                                    for (i, n) in NATO.iter().enumerate() {
                                        ui.selectable_value(&mut m.tele_channel, i as u8, *n);
                                    }
                                });
                            ui.end_row();
                            // Passability (5 bits) — bits0-3 = ALLOW projectiles/flying/heavy-land/
                            // light-land vehicles; bit4 = DISALLOW players (INVERTED: clear = allowed).
                            // Default 0x00 = players-only. RE'd from reach_tag_test teleporter_passability_flags.
                            ui.label("passability");
                            ui.vertical(|ui| {
                                let mut allow = |ui: &mut egui::Ui, mask: u8, label: &str| {
                                    let mut on = m.tele_passability & mask != 0;
                                    if ui.checkbox(&mut on, label).changed() {
                                        if on { m.tele_passability |= mask; } else { m.tele_passability &= !mask; }
                                    }
                                };
                                allow(ui, 0x01, "Projectiles");
                                allow(ui, 0x02, "Flying vehicles");
                                allow(ui, 0x04, "Heavy land vehicles");
                                allow(ui, 0x08, "Light land vehicles");
                                // Players: INVERTED bit4 — checkbox "Players" = NOT bit4.
                                let mut players = m.tele_passability & 0x10 == 0;
                                if ui.checkbox(&mut players, "Players").on_hover_text("bit4 is 'disallow players' — this checkbox is inverted").changed() {
                                    if players { m.tele_passability &= !0x10; } else { m.tele_passability |= 0x10; }
                                }
                            });
                            ui.end_row();
                        } else if m.cached_type == 19 {
                            ui.label("location name");
                            let mut ln = if m.location_name == 0xFFFF { -1 } else { m.location_name as i32 };
                            ui.add(egui::DragValue::new(&mut ln).range(-1..=255));
                            m.location_name = if ln < 0 { 0xFFFF } else { ln as u16 };
                            ui.end_row();
                        }
                        } // end of the Reach boundary + type-extras rows

                        ui.label("position");
                        ui.horizontal(|ui| {
                            // A typed '-' anywhere flips the sign (numfield), so "12 -" == "-12".
                            for c in 0..3 {
                                ui.add(egui::DragValue::new(&mut m.pos[c]).speed(0.1).custom_parser(numfield::parse_signed));
                            }
                        });
                        ui.end_row();
                    });

                    if m != m0 {
                        // One snapshot per contiguous edit-session on this object.
                        if self.prop_snapshotted_for != Some(datum) {
                            self.push_edit_undo();
                            self.prop_snapshotted_for = Some(datum);
                        }
                        // The panel edits the WHOLE selection, not just the object it
                        // happens to be showing — that is what makes it usable for bulk work
                        // ("make these 40 blocks red"). Only the fields that actually changed are
                        // copied across (see ObjMeta::apply_changed_fields); position travels as a
                        // DELTA so a group keeps its shape instead of collapsing onto one point.
                        let others: Vec<u32> = self.selected_set.iter().copied()
                            .filter(|d| *d != datum && self.mvar_meta.contains_key(d))
                            .collect();
                        let dpos = [m.pos[0] - m0.pos[0], m.pos[1] - m0.pos[1], m.pos[2] - m0.pos[2]];
                        let moved = dpos.iter().any(|v| v.abs() > 1e-9);
                        // team/color → live recolor (signature() hashes forge_colors).
                        let cu8 = if m.color < 0 { 0xFFu8 } else { m.color as u8 };
                        self.mvar_colors.insert(datum, (m.team, cu8));
                        // position → live move (mvar_objects feeds the render each tick).
                        if m.pos != m0.pos {
                            if let Some(o) = self.mvar_objects.iter_mut().find(|o| o.datum == datum) {
                                o.pos = m.pos;
                            }
                        }
                        // Rebuild the boundary-shape overlay when the shape/size/pos/type
                        // that drives it changed, so the zone updates live as you edit.
                        let shape_changed = m.boundary_shape != m0.boundary_shape
                            || m.boundary != m0.boundary
                            || m.pos != m0.pos
                            || m.cached_type != m0.cached_type;
                        self.mvar_meta.insert(datum, m.clone());
                        for d in others {
                            let Some(mut t) = self.mvar_meta.get(&d).cloned() else { continue };
                            t.apply_changed_fields(&m0, &m);
                            if moved {
                                t.pos = [t.pos[0] + dpos[0], t.pos[1] + dpos[1], t.pos[2] + dpos[2]];
                                if let Some(o) = self.mvar_objects.iter_mut().find(|o| o.datum == d) {
                                    o.pos = t.pos;
                                }
                            }
                            let tc = if t.color < 0 { 0xFFu8 } else { t.color as u8 };
                            self.mvar_colors.insert(d, (t.team, tc));
                            self.mvar_meta.insert(d, t);
                        }
                        if moved {
                            self.highlight_dirty = true;
                        }
                        if shape_changed {
                            self.rebuild_overlays();
                        }
                    }
                    // RAW forward/up vector editor. The engine builds the object matrix
                    // from these vectors WITHOUT re-orthonormalizing, so inflating a component
                    // stretches the model along that axis and making forward/up non-perpendicular
                    // shears it. Edits the live scene object directly (fwd/up aren't in ObjMeta).
                    // NOTE: Save re-quantizes orientation to a UNIT basis (angular .mvar encoding), so
                    // a skew previews in HMS but does not persist to the file. Use "reset to unit" to
                    // return to a normal rotation.
                    let cur_fu = self.mvar_objects.iter().find(|o| o.datum == datum).map(|o| (o.fwd, o.up));
                    if let Some((f0, u0)) = cur_fu {
                        let (mut fwd, mut up) = (f0, u0);
                        // The orientation people actually type — three angles in DEGREES,
                        // not a pair of basis vectors. Same meaning as the `rotate <axis> <deg>`
                        // command (right-hand rule about world X/Y/Z), so the two agree.
                        let e0 = euler::to_euler_deg(f0, u0);
                        let mut e = e0;
                        egui::Grid::new("mvar-rot-euler").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                            ui.label("rotation");
                            ui.horizontal(|ui| {
                                for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
                                    ui.add(
                                        egui::DragValue::new(&mut e[i])
                                            .speed(1.0)
                                            .range(-360.0..=360.0)
                                            .suffix("°")
                                            .prefix(*name)
                                            // A '-' typed ANYWHERE is the sign: "90 -" turns the
                                            // same way as "-90", so the sign never has to come first.
                                            .custom_parser(numfield::parse_signed),
                                    );
                                }
                            });
                            ui.end_row();
                        });
                        ui.small("X = roll · Y = pitch (+ is nose down) · Z = yaw. Right-hand rule, same as the rotate command.");
                        if e != e0 {
                            // Preserve any STRETCH: only the direction changes, the magnitudes stay.
                            let (nf, nu) = euler::from_euler_deg(
                                e,
                                glam::Vec3::from(f0).length(),
                                glam::Vec3::from(u0).length(),
                            );
                            fwd = nf;
                            up = nu;
                        }
                        egui::CollapsingHeader::new("raw forward / up (advanced)")
                            .default_open(false)
                            .show(ui, |ui| {
                        egui::Grid::new("mvar-rot-raw").num_columns(2).show(ui, |ui| {
                            ui.label("forward (raw)");
                            ui.horizontal(|ui| {
                                for c in 0..3 { ui.add(egui::DragValue::new(&mut fwd[c]).speed(0.02).custom_parser(numfield::parse_signed)); }
                                ui.small(format!("|{:.2}|", glam::Vec3::from(fwd).length()));
                            });
                            ui.end_row();
                            ui.label("up (raw)");
                            ui.horizontal(|ui| {
                                for c in 0..3 { ui.add(egui::DragValue::new(&mut up[c]).speed(0.02).custom_parser(numfield::parse_signed)); }
                                ui.small(format!("|{:.2}|", glam::Vec3::from(up).length()));
                            });
                            ui.end_row();
                        });
                        if ui.small_button("reset to unit").on_hover_text("normalize forward & up back to a clean rotation").clicked() {
                            fwd = glam::Vec3::from(fwd).normalize_or_zero().into();
                            up = glam::Vec3::from(up).normalize_or_zero().into();
                            if glam::Vec3::from(fwd).length_squared() < 1e-6 { fwd = [1.0, 0.0, 0.0]; }
                            if glam::Vec3::from(up).length_squared() < 1e-6 { up = [0.0, 0.0, 1.0]; }
                        }
                            });
                        if fwd != f0 || up != u0 {
                            // Rotation edits are undoable, coalesced into the same per-object
                            // edit session as the other property fields.
                            if self.prop_snapshotted_for != Some(datum) {
                                self.push_edit_undo();
                                self.prop_snapshotted_for = Some(datum);
                            }
                            // Apply the orientation to the WHOLE selection, like every
                            // other field in this panel. Each object keeps its OWN stretch (the
                            // magnitudes of its forward/up), only the direction is set.
                            let targets: Vec<u32> = if self.selected_set.len() > 1 {
                                self.selected_set.clone()
                            } else {
                                vec![datum]
                            };
                            let angles = euler::to_euler_deg(fwd, up);
                            for d in targets {
                                if let Some(o) = self.mvar_objects.iter_mut().find(|o| o.datum == d) {
                                    if d == datum {
                                        o.fwd = fwd;
                                        o.up = up;
                                    } else {
                                        let (nf, nu) = euler::from_euler_deg(
                                            angles,
                                            glam::Vec3::from(o.fwd).length(),
                                            glam::Vec3::from(o.up).length(),
                                        );
                                        o.fwd = nf;
                                        o.up = nu;
                                    }
                                }
                            }
                            self.highlight_dirty = true;
                        }
                    }
                }
                if !self.mvar_meta.contains_key(&datum) {
                    // An object with no editable record (a scenario piece the viewer renders but
                    // the variant does not own, or a live-engine object in the injection build).
                    // Say WHAT the map object is (tag path / class / tag id / placement), read-only.
                    let is_scnr = self.scenario_objects.iter().any(|o| o.datum == datum);
                    if is_scnr {
                        let (cls, leaf) = self.object_identity(datum).map(|id| (id.class.clone(), id.leaf().to_string())).unwrap_or_default();
                        let mut head = if cls.is_empty() { "Map object".to_string() } else { format!("Map object ({cls})") };
                        if !leaf.is_empty() {
                            head.push_str(&format!(":  {}", prettify_stringid(&leaf)));
                        }
                        ui.label(egui::RichText::new(head).strong()).on_hover_text(&leaf);
                        ui.small("Belongs to the map itself (read-only); only variant / placed objects are editable.");
                    } else {
                        ui.label(egui::RichText::new("Not a variant object").strong());
                        ui.small("This piece belongs to the map itself; only variant / placed objects are editable.");
                    }
                    self.identity_rows_ui(ui, datum, true);
                }
                // The live-engine editors (Team/Color combos + "Apply team/color", "Apply (live)"
                // scale, "Set persistent (X330)", "Prevent GC (pin)") only work with the injected
                // DLL, so they exist in the injection build only; the forge record above edits
                // team / colour / scale (spawn seq) offline.
                #[cfg(feature = "injection")]
                {
                    let idx = row.map(|(i, _, _, _)| i);
                    ui.separator();
                    let names = ["0", "1", "2", "3", "4", "5", "6", "7", "Neutral(8)"];
                    egui::ComboBox::from_label("Team (live)")
                        .selected_text(names.get(self.edit_team as usize).copied().unwrap_or("?"))
                        .show_ui(ui, |ui| {
                            for t in 0u8..=8 {
                                ui.selectable_value(&mut self.edit_team, t, names[t as usize]);
                            }
                        });
                    egui::ComboBox::from_label("Color (live)")
                        .selected_text(names.get(self.edit_color as usize).copied().unwrap_or("?"))
                        .show_ui(ui, |ui| {
                            for c in 0u8..=8 {
                                ui.selectable_value(&mut self.edit_color, c, names[c as usize]);
                            }
                        });
                    ui.add(egui::Slider::new(&mut self.edit_scale, 0.1..=10.0).text("scale (live)"));
                    if ui.button("Apply (live)").on_hover_text("Object_SetScale — live engine resize").clicked() {
                        if let (Some(i), Some(fe)) = (idx, &self.forge_edit) {
                            fe.set_scale(i, datum, self.edit_scale);
                            self.status = format!("Scaled 0x{datum:08X} x{:.2}", self.edit_scale);
                        }
                    }
                    if ui.button("Set persistent (X330)").on_hover_text("Encode scale into spawnSequence (saved in the variant)").clicked() {
                        if let (Some(i), Some(fe)) = (idx, &self.forge_edit) {
                            let (seq, actual) = forge_scale::object_scale_to_seq(self.edit_scale);
                            fe.set_spawn_seq(i, datum, seq);
                            self.status = format!("Persistent scale seq {seq} (~x{actual:.2})");
                        }
                    }
                    if ui.button("Apply team/color (live)").clicked() {
                        match (idx, &self.forge_edit) {
                            (Some(i), Some(fe)) => {
                                fe.set_team_color(i, datum, self.edit_team, self.edit_color);
                                self.status = format!("Set team {} color {} on 0x{datum:08X}", self.edit_team, self.edit_color);
                            }
                            _ => self.status = "Edit unavailable (inject into MCC first).".into(),
                        }
                    }
                    // Explicit anti-GC. Any ForgeObjectEdit commit runs the DLL's
                    // StampAntiGarbageCollection (sets s_object_data+0x143 bit0), so re-applying
                    // the current team/color pins the object against the ~30s monitor despawn.
                    if ui.button("Prevent GC (pin)").on_hover_text(
                        "Re-commit to run the engine's anti-garbage-collection stamp on this object",
                    ).clicked() {
                        if let (Some(i), Some(fe)) = (idx, &self.forge_edit) {
                            fe.set_team_color(i, datum, self.edit_team, self.edit_color);
                            self.status = format!("Pinned 0x{datum:08X} against GC");
                        } else {
                            self.status = "Pin unavailable (inject into MCC first).".into();
                        }
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Fly to").on_hover_text("Frame the camera on this object (F)").clicked() {
                        self.frame_selected();
                    }
                    let n = self.selected_set.len().max(1);
                    let del_label = if n > 1 { format!("Delete {n} objects") } else { "Delete object".to_string() };
                    if ui.button(del_label).on_hover_text("Remove the selection from the variant (Delete / Backspace). Undo with Ctrl+Z.").clicked() {
                        self.delete_selection();
                    }
                });
                ui.add_space(4.0);
                ui.small("Move: G or arrow keys / PgUp-PgDn · Rotate: R or [ ] · Duplicate: Shift+D · Undo: Ctrl+Z");
            });
        } else {
            self.props_for = None;
        }
        // Step the dropdown preview from this frame's popup/hover (a closed popup, the
        // pointer outside its rect, a collapsed window or no selection all report None -> preview
        // off, wireframe back on this same frame).
        self.panel_hover_preview(hovered_popup, hovered_entry);

        // Bottom STATUS BAR — load progress lives here (under the renderer), for BOTH the map BSP
        // load AND the map-variant object decode (which streams in over frames). Shown at the very
        // bottom so it never covers the viewport.
        // Distinct Id from the "status" text bar above: two panels sharing one Id
        // makes egui report "Second use of widget ID" every frame.
        egui::TopBottomPanel::bottom("status_progress").show(ctx, |ui| {
            let map_loading = self.load_rx.is_some() || self.h4_load.is_some();
            let waiting_variant = self.pending_mvar.is_some();
            let preparing = self.settle_pending || self.variant_retry.is_some();
            // Variant object-decode progress: how many of the variant's unique render models are
            // decoded (they stream in over frames after the base map finishes).
            let (vdone, vtotal) = if !self.mvar_objects.is_empty() {
                if let Some(scene) = self.scene_ctl.as_ref() {
                    let mut tags: Vec<u32> = self
                        .mvar_objects
                        .iter()
                        .map(|o| o.mode_tag)
                        .filter(|&t| t != 0 && t != 0xFFFF_FFFF)
                        .collect();
                    tags.sort_unstable();
                    tags.dedup();
                    (scene.tags_settled(&tags), tags.len())
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            };
            // Also require the scene to still have pending decode/rebuild work.
            // Belt-and-suspenders: if some tag can never be counted (never lands in geom_cache), the
            // bar would otherwise hang forever; once the scene reports no pending work, the variant is
            // done regardless of the per-tag tally, so the spinner always clears.
            let variant_pending = self.objscene().map(|s| s.rebuild_pending()).unwrap_or(false);
            let variant_decoding = vtotal > 0 && vdone < vtotal && variant_pending;
            ui.horizontal(|ui| {
                if map_loading {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.add(
                        egui::ProgressBar::new(self.load_frac)
                            .desired_width(260.0)
                            .text(format!("Loading map… {}%", (self.load_frac * 100.0) as i32)),
                    );
                } else if waiting_variant {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.label("Loading variant — loading its base map…");
                } else if variant_decoding {
                    let f = vdone as f32 / vtotal as f32;
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.add(
                        egui::ProgressBar::new(f)
                            .desired_width(260.0)
                            .text(format!("Loading variant… {vdone}/{vtotal} objects")),
                    );
                } else if preparing {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.label("Preparing variant…");
                } else {
                    let s = if !self.spawn_status.is_empty() {
                        self.spawn_status.as_str()
                    } else if !self.map_status.is_empty() {
                        self.map_status.as_str()
                    } else {
                        "Ready"
                    };
                    ui.label(s);
                }
                // Forge object budget counter (placed / max), right-aligned. A Reach map variant
                // has 651 object SLOTS (the real ceiling, not the 650 the Forge menu shows), and a
                // save can only place a new object in a free one.
                const MAX_FORGE_OBJECTS: usize = 651;
                // Objects HMS could not resolve still occupy a slot in the variant, so they MUST
                // be counted -- otherwise the budget reads low on a variant that is actually full
                // and a save then reports objects it could not fit, with no visible reason.
                let placed = self.mvar_objects.len() + self.local_objects.len() + self.mvar_unresolved.len();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let col = if placed >= MAX_FORGE_OBJECTS {
                        egui::Color32::from_rgb(230, 90, 90)
                    } else if placed >= MAX_FORGE_OBJECTS * 9 / 10 {
                        egui::Color32::from_rgb(230, 190, 80)
                    } else {
                        ui.visuals().text_color()
                    };
                    let free = MAX_FORGE_OBJECTS.saturating_sub(placed);
                    ui.colored_label(col, egui::RichText::new(format!("Objects: {placed} / {MAX_FORGE_OBJECTS}")).strong())
                        .on_hover_text(format!(
                            "{free} free slot(s). A map variant has {MAX_FORGE_OBJECTS} object slots; a save can only place a new object in a free one."
                        ));
                });
            });
            if map_loading || waiting_variant || variant_decoding || preparing {
                ctx.request_repaint(); // keep progress + spinner animating
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let size = (
                (avail.x.max(1.0)) as u32,
                (avail.y.max(1.0)) as u32,
            );

            // Resize the offscreen targets + re-point the egui texture.
            if size != self.renderer.size() {
                self.renderer.resize(&self.render_state.device, size);
                self.render_state.renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.render_state.device,
                    self.renderer.color_view_srgb(),
                    wgpu::FilterMode::Linear,
                    self.tex_id,
                );
            }

            // Render the 3D scene to the offscreen texture. Time drives the
            // decorator wind sway.
            let time = ctx.input(|i| i.time) as f32;
            self.frame_prof_lap(1);
            // Per-frame CPU simulation of the effect particle systems (grav-lift plasma /
            // sparks, effect_scenery waterfalls) billboarded for this camera → particle pass.
            if let Some(scene) = self.scene_ctl.as_mut() {
                if scene.has_cache() {
                    let fwd = self.camera.forward();
                    let right = self.camera.right();
                    let up = right.cross(fwd).normalize();
                    let pm = scene.build_particle_meshes(self.camera.pos, fwd, right, up, time, self.renderer.mesh_renderer(), &self.render_state.device, &self.render_state.queue);
                    self.renderer.set_particle_meshes(pm);
                    // Placed lens flares (efsc street-light glares, object attachment flares).
                    self.renderer.set_lens_flare_instances(scene.lens_flare_instances(&self.render_state.device, &self.render_state.queue));
                }
            }
            self.frame_prof_lap(2);
            self.renderer.render(
                &self.render_state.device,
                &self.render_state.queue,
                &self.camera,
                time,
            );
            self.frame_prof_lap(3);

            // Temporal auto-exposure smoothing. The shader recomputes gain from the RAW per-frame
            // 1×1 meter every frame, so panning the view (sky in/out of frame) would make the
            // metered luminance — and thus the exposure — JUMP.
            // The engine instead adapts SLOWLY: adapted = lerp(adapted, target, ~0.05/frame) (RE
            // get_render_exposure / sub_1407E74D0). Replicate it CPU-side: read back this frame's
            // metered log-luminance, compute the same target gain the shader would (key/2^meanlog,
            // clamped to the cfxs band, × base), lerp a persistent value toward it, and drive it as a
            // FIXED exposure. Only in AUTO mode (manual/Lighting-Lab-fixed sets its own). Headless is
            // untouched (single frame → the shader's direct meter is already converged for a still).
            if !self.ll_manual_exp {
                self.ae_frame = self.ae_frame.wrapping_add(1);
                // The meter readback is NON-BLOCKING (a 3-slot staging ring, polled without
                // waiting; the value is 2-3 frames old — the engine's own readback ring is 3 slots
                // deep too). A blocking `poll(Wait)` would stall the UI thread for a whole GPU frame.
                {
                    if let Some(mean_log) = self.renderer.read_mean_log_async(&self.render_state.device, &self.render_state.queue) {
                        // `// #h4-expo-3` The gain comes from the RENDERER's own exposure terms —
                        // the single source of truth the resolve pass reads — not from a re-derivation
                        // out of the Reach `SceneController`. That re-derivation was the global
                        // Halo 4 brightness error: on a Halo 4 map it used key 0.1 and the band
                        // [2^0, 2^2] while the Halo 4 lane writes key 1.0 and an ABSOLUTE band, so
                        // `0.1 * 2^solved` fell under the floor and every Halo 4 map rendered at
                        // gain exactly 1.0 in the GUI — up to +1.6 stops too bright on the Forge
                        // maps (Forge Island 0.335, Ravine 0.473) and several stops too dark on the
                        // dim ones (Outcast 7.57) — which is why turning the base gain down to 0.2
                        // "looked a lot better". It also read the Reach band as 2^EV instead of
                        // `key * 10 * 2^EV`, so the GUI disagreed with the shader on every Reach
                        // map whose cfxs key is not 0.1.
                        let (base, key, lo, hi, cal) = self.renderer.exposure_params();
                        let raw = key / 2f32.powf(mean_log).max(1e-4);
                        self.ae_target = base * (raw * cal).clamp(lo.min(hi), hi);
                    }
                }
                if self.ae_target > 0.0 {
                    let dt = ctx.input(|i| i.stable_dt).clamp(1.0 / 240.0, 1.0 / 20.0);
                    let gain = match self.h4_cfxs.as_ref().filter(|_| self.renderer.h4_meter_on()) {
                        // Halo 4: the engine's own adaptation dynamics (halo4.dll sub_180359768) in
                        // STOPS — a `delay`-second window that must agree in sign before the stops
                        // move at all, a `blend` rate scaled to 30 Hz (only with exposure flags bit
                        // 1) and a hard `max change` per frame. The target is the band-clamped
                        // fixed point the meter solves for (the engine's `prev + screen_brightness
                        // - M(prev)` is the same point, one Newton step of a unit-slope M).
                        Some(x) => h4::lighting::adapt_stops(&mut self.h4_ae_stops, &mut self.h4_ae_hist, x, self.ae_target.max(1e-6).log2(), dt).exp2(),
                        None => {
                            if !(self.ae_smoothed > 0.0) {
                                self.ae_smoothed = self.ae_target; // first read after load: snap (no fade-in flash)
                            } else {
                                // ~0.10/frame EMA ≈ the engine's proportional adaptation — smooth, not laggy.
                                self.ae_smoothed += (self.ae_target - self.ae_smoothed) * 0.10;
                            }
                            self.ae_smoothed
                        }
                    };
                    self.ae_smoothed = gain;
                    self.renderer.set_fixed_exposure(&self.render_state.queue, gain);
                    ctx.request_repaint(); // keep the ramp animating even when otherwise idle
                } else {
                    // Before the first readback (fresh map load): use the shader's own metered path.
                    self.renderer.set_fixed_exposure(&self.render_state.queue, 0.0);
                }
            }

            self.frame_prof_lap(4);
            let img = egui::Image::new(egui::load::SizedTexture::new(
                self.tex_id,
                egui::vec2(size.0 as f32, size.1 as f32),
            ))
            .sense(egui::Sense::click_and_drag());
            let response = ui.add(img);
            // #construct-h4: remember where the viewport IS, so `preview where` can say which
            // window coordinates a scripted `preview click` has to use to hit it.
            self.dbg_viewport = Some(response.rect);
            // Blender-style floating tool buttons overlaid on the viewport (top-left), drawn in
            // THIS panel's layer and clamped to the viewport rect (see floating_toolbar).
            self.floating_toolbar(ui, response.rect);
            // Drag-out placement (BEFORE the transform/camera handlers so it owns input this frame):
            // when a queued palette item's held cursor enters the viewport, spawn it under the cursor;
            // it then snaps to the surface under the cursor until release.
            let placing_active = self.try_drag_place(ctx, &response);
            // Modal grab/rotate/duplicate takes over input while active — suppress
            // camera fly + object picking so its keys/click drive the transform instead.
            // Build the gizmo from the SAME positions the current object meshes
            // were built from (this frame's tick_scene, i.e. before this frame's move) so the
            // gizmo/constraint-line and the solid meshes advance in lockstep — no lead/lag.
            self.refresh_gizmo();
            let xform_active = if placing_active { false } else { self.update_object_transform(ctx, &response) };
            if !xform_active && !placing_active {
                self.update_camera(ctx, &response);
                self.handle_pick(&response);
            } else {
                self.flying = false;
            }
            self.apply_cursor_capture(ctx);
            // Box-select (Select mode): left-drag a rubber band to select every object whose
            // centre falls inside it. Only when no transform op owns the drag.
            if self.tool_mode == ToolMode::Select && !xform_active && !placing_active && self.pending_place.is_none() {
                if response.drag_started_by(egui::PointerButton::Primary) {
                    self.box_select_start = response.interact_pointer_pos();
                }
                if let Some(start) = self.box_select_start {
                    if let Some(cur) = ctx.pointer_latest_pos() {
                        let r = egui::Rect::from_two_pos(start, cur);
                        ui.painter().rect_filled(r, 0.0, egui::Color32::from_rgba_unmultiplied(120, 180, 255, 32));
                        ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(140, 190, 255)), egui::StrokeKind::Inside);
                    }
                    if response.drag_stopped_by(egui::PointerButton::Primary) {
                        if let Some(cur) = ctx.pointer_latest_pos() {
                            let additive = ctx.input(|i| i.modifiers.shift);
                            self.box_select(start, cur, response.rect, additive);
                        }
                        self.box_select_start = None;
                    }
                }
            }
        });

        // Headless auto-screenshot countdown: render a few frames with the framed
        // camera, then capture to HMS_AUTOSHOT and exit (dev aid, no human loop).
        if self.autoshot_frames > 0 {
            self.autoshot_frames -= 1;
            if self.autoshot_frames == 0 {
                if let Some(path) = self.autoshot.clone() {
                    self.screenshot_to(&path);
                    log::info!("autoshot saved -> {}", path.display());
                }
                std::process::exit(0);
            }
            ctx.request_repaint();
        }

        // Drive continuous animation (fly camera, sway, live tick) at a CAPPED
        // ~60 fps. An unbounded request_repaint() saturates the swapchain and the UI
        // thread blocks on GPU backpressure.
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        // Deferred variant open: the browser set this when Open was clicked. One frame is
        // burnt first so the window is actually painted gone before the blocking import
        // starts, instead of lingering for the whole load.
        if let Some((path, wait)) = self.pending_open_variant.take() {
            if wait > 0 {
                self.pending_open_variant = Some((path, wait - 1));
                ctx.request_repaint();
            } else {
                self.import_mvar(path);
            }
        }
        // #dialogs test hook: capture the whole egui frame (dialogs included) to a PNG. The
        // request sends a Screenshot viewport command this frame; the pixels arrive as an input
        // event on a later frame, where they are written out.
        if let Some(path) = self.egui_shot_request.take() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            self.egui_shot_pending = Some(path);
            ctx.request_repaint();
        }
        if self.egui_shot_pending.is_some() {
            let img = ctx.input(|i| i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            }));
            if let Some(img) = img {
                let path = self.egui_shot_pending.take().unwrap();
                let (w, h) = (img.size[0] as u32, img.size[1] as u32);
                let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                for px in &img.pixels { let c = px.to_array(); rgba.extend_from_slice(&c); }
                if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
                match headless::write_png(&path.to_string_lossy(), &rgba, w, h) {
                    Ok(()) => log::info!("dialog shot -> {} ({w}x{h})", path.display()),
                    Err(e) => log::warn!("dialog shot failed: {e}"),
                }
            } else {
                ctx.request_repaint();
            }
        }
        if prof_on { self.frame_prof_lap(5); self.frame_prof_report(); }
    }
}

impl App {
    /// Record the time since the previous lap into stage `k` (HMS_FRAMEPROF only).
    fn frame_prof_lap(&mut self, k: usize) {
        if let Some(p) = self.frame_prof.as_mut() {
            let now = std::time::Instant::now();
            p.1[k] += now.duration_since(p.0).as_secs_f32() * 1000.0;
            p.0 = now;
        }
    }

    /// Print any update() over 20 ms with its stage split, plus a 300-frame summary
    /// (mean / p99 / max and per-stage means). The wall time between updates (vsync + GPU) is not
    /// included -- this is the CPU cost of update() only.
    fn frame_prof_report(&mut self) {
        let Some(p) = self.frame_prof.as_mut() else { return };
        let st = p.1;
        let total: f32 = st.iter().sum();
        let mut row = [0f32; 7];
        row[0] = total;
        row[1..].copy_from_slice(&st);
        p.2.push(row);
        const NAMES: [&str; 6] = ["pre", "panels", "particles", "render", "meter", "input+rest"];
        if total > 20.0 {
            let (mut top, mut topv) = (0usize, 0f32);
            for k in 0..6 { if st[k] > topv { topv = st[k]; top = k; } }
            eprintln!("HMS_FRAMEPROF slow update {total:.1} ms: top={} {:.1} ms [{}]", NAMES[top], topv,
                (0..6).map(|k| format!("{}={:.1}", NAMES[k], st[k])).collect::<Vec<_>>().join(" "));
        }
        if p.2.len() >= 300 {
            let n = p.2.len() as f32;
            let mut tot: Vec<f32> = p.2.iter().map(|r| r[0]).collect();
            tot.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = tot[((tot.len() as f32 * 0.99) as usize).min(tot.len() - 1)];
            let means: Vec<String> = (0..6).map(|k| format!("{}={:.2}", NAMES[k], p.2.iter().map(|r| r[k + 1]).sum::<f32>() / n)).collect();
            eprintln!("HMS_FRAMEPROF {} updates: mean={:.2} median={:.2} p99={p99:.2} max={:.2} ms | {}",
                p.2.len(), tot.iter().sum::<f32>() / n, tot[tot.len() / 2], tot.last().copied().unwrap_or(0.0), means.join(" "));
            p.2.clear();
        }
    }
}

#[cfg(test)]
mod dialog_tests {
    use super::App;
    use std::path::Path;

    // #dialogs: the Save browser resolves <dir>/<name>, adding .mvar only when no extension typed.
    #[test]
    fn resolve_save_target_appends_mvar_when_no_extension() {
        let d = Path::new("/games/haloreach/map_variants");
        assert_eq!(App::resolve_save_target(d, "myvariant"), Some(d.join("myvariant.mvar")));
        assert_eq!(App::resolve_save_target(d, "  spaced  "), Some(d.join("spaced.mvar")));
    }
    #[test]
    fn resolve_save_target_keeps_typed_extension() {
        let d = Path::new("/games/haloreach/map_variants");
        assert_eq!(App::resolve_save_target(d, "already.mvar"), Some(d.join("already.mvar")));
        // A different typed extension is honoured (not double-suffixed).
        assert_eq!(App::resolve_save_target(d, "keep.bak"), Some(d.join("keep.bak")));
    }
    #[test]
    fn resolve_save_target_blank_is_none() {
        let d = Path::new("/tmp");
        assert_eq!(App::resolve_save_target(d, ""), None);
        assert_eq!(App::resolve_save_target(d, "   "), None);
    }
}

#[cfg(test)]
mod mass_edit_tests {
    use super::{forge_team_name, ObjMeta};

    fn sample() -> ObjMeta {
        ObjMeta {
            name: "objects/block".into(),
            folder: 3,
            item: 2,
            pos: [10.0, 20.0, 30.0],
            team: 8,
            color: -1,
            cached_type: 1,
            spawn_seq: 5,
            respawn: 10,
            label_idx: 7,
            label: "gold".into(),
            placement: 1,
            boundary_shape: 2,
            boundary: [1, 2, 3, 4],
            weapon_clips: 1,
            tele_channel: 2,
            tele_passability: 3,
            location_name: 9,
            spawn_rel: -1,
            slot: 42,
            ..Default::default()
        }
    }

    /// The pseudo-flags follow the mass-edit contract too — a toggled flag reaches
    /// every selected object, an untouched one keeps each object's own override; and the
    /// effective state re-derives from team/label when the override is None.
    #[test]
    fn flag_overrides_propagate_and_defaults_rederive() {
        use crate::forge_scale::TEAM_GREEN;
        let before = sample();
        let mut after = before.clone();
        after.flags.shadow = Some(true); // the single edit

        let mut other = sample();
        other.flags.scaled = Some(false); // its own, untouched
        other.apply_changed_fields(&before, &after);
        assert_eq!(other.flags.shadow, Some(true), "the toggled flag must propagate");
        assert_eq!(other.flags.scaled, Some(false), "an untouched flag keeps the target's own override");
        assert!(other.shadow_on() && !other.scaled_on());

        // derived defaults: label "scale" → scaled; green + "scale" → shadow; overrides beat both
        let mut m = sample();
        assert!(!m.scaled_on() && !m.shadow_on(), "a 'gold'-labelled neutral object has neither");
        m.label = "scale".into();
        assert!(m.scaled_on() && !m.shadow_on());
        m.team = TEAM_GREEN;
        assert!(m.shadow_on(), "moving onto green with the scale label starts casting");
        m.flags.shadow = Some(false);
        assert!(!m.shadow_on(), "an explicit override survives until the user changes it");
        m.flags.shadow = None;
        assert!(m.shadow_on());
        // SCALED off → the render scale is ×1 even with the label
        m.spawn_seq = 10;
        assert!(m.scale(crate::forge_scale::ScaleConvention::X330) > 1.0);
        m.flags.scaled = Some(false);
        assert_eq!(m.scale(crate::forge_scale::ScaleConvention::X330), 1.0);
    }

    /// The core contract: only the field the user actually touched travels to the others.
    #[test]
    fn only_changed_fields_propagate() {
        let before = sample();
        let mut after = before.clone();
        after.team = 1; // the single edit

        let mut other = sample();
        other.team = 4;
        other.respawn = 99;
        other.color = 5;
        other.apply_changed_fields(&before, &after);

        assert_eq!(other.team, 1, "the edited field must propagate");
        assert_eq!(other.respawn, 99, "an untouched field must keep the target's own value");
        assert_eq!(other.color, 5, "an untouched field must keep the target's own value");
    }

    /// NEUTRAL (8) and NONE (0xFF) are real team values for a mass edit — setting a
    /// batch back to "no team" must land on every object, in both directions.
    #[test]
    fn neutral_and_none_teams_mass_apply() {
        let before = sample(); // team 8 (neutral)
        let mut after = before.clone();
        after.team = crate::mvar::TEAM_NONE;
        let mut other = sample();
        other.team = 4;
        other.apply_changed_fields(&before, &after);
        assert_eq!(other.team, crate::mvar::TEAM_NONE, "neutral -> none must propagate");

        let mut before2 = sample();
        before2.team = 2;
        let mut after2 = before2.clone();
        after2.team = crate::mvar::TEAM_NEUTRAL;
        let mut other2 = sample();
        other2.team = 0;
        other2.apply_changed_fields(&before2, &after2);
        assert_eq!(other2.team, crate::mvar::TEAM_NEUTRAL, "green -> neutral must propagate");
        assert_eq!(forge_team_name(crate::mvar::TEAM_NEUTRAL as i32), "neutral");
        assert_eq!(forge_team_name(-1), "none");
    }

    /// Identity + placement must NEVER be cloned across a batch — that would make every
    /// selected object the same object, sitting on top of the one the panel happened to show.
    #[test]
    fn identity_fields_are_never_copied() {
        let before = sample();
        let mut after = before.clone();
        after.team = 0;
        after.pos = [999.0, 999.0, 999.0];
        after.name = "objects/other".into();
        after.slot = 1;
        after.spawn_rel = 12;

        let mut other = sample();
        other.name = "objects/mine".into();
        other.pos = [1.0, 2.0, 3.0];
        other.slot = 77;
        other.spawn_rel = -1;
        other.apply_changed_fields(&before, &after);

        assert_eq!(other.name, "objects/mine", "name is identity — must not be cloned");
        assert_eq!(other.pos, [1.0, 2.0, 3.0], "pos is applied as a DELTA by the caller, not copied");
        assert_eq!(other.slot, 77, "slot is this object's own .mvar slot");
        assert_eq!(other.spawn_rel, -1, "parent link is per-object");
        assert_eq!(other.team, 0, "...but the real edit still landed");
    }

    #[test]
    fn every_editable_data_field_can_propagate() {
        let before = sample();
        let mut after = before.clone();
        after.folder = 9;
        after.item = 11;
        after.team = 2;
        after.color = 6;
        after.cached_type = 4;
        after.spawn_seq = 77;
        after.respawn = 45;
        after.label_idx = 3;
        after.label = "red".into();
        after.placement = 5;
        after.boundary_shape = 1;
        after.boundary = [9, 9, 9, 9];
        after.weapon_clips = 6;
        after.tele_channel = 7;
        after.tele_passability = 1;
        after.location_name = 22;

        let mut other = ObjMeta::default();
        other.apply_changed_fields(&before, &after);

        assert_eq!(other.folder, 9);
        assert_eq!(other.item, 11);
        assert_eq!(other.team, 2);
        assert_eq!(other.color, 6);
        assert_eq!(other.cached_type, 4);
        assert_eq!(other.spawn_seq, 77);
        assert_eq!(other.respawn, 45);
        assert_eq!(other.label_idx, 3);
        assert_eq!(other.label, "red");
        assert_eq!(other.placement, 5);
        assert_eq!(other.boundary_shape, 1);
        assert_eq!(other.boundary, [9, 9, 9, 9]);
        assert_eq!(other.weapon_clips, 6);
        assert_eq!(other.tele_channel, 7);
        assert_eq!(other.tele_passability, 1);
        assert_eq!(other.location_name, 22);
    }

    #[test]
    fn no_edit_changes_nothing() {
        let before = sample();
        let after = before.clone();
        let mut other = sample();
        other.team = 4;
        other.respawn = 99;
        let snapshot = other.clone();
        other.apply_changed_fields(&before, &after);
        assert!(other == snapshot, "an empty diff must be a no-op on every target");
    }
}

/// Full add/delete/save cycles over REAL variants, driving the same
/// `build_save_list` + `mvar::save_objects` path File▸Save uses. These gates catch dropped
/// objects, a growing "no free slot", and slot-enumeration corruption (walls saved as spawns).
#[cfg(test)]
mod save_stress {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn obj(datum: u32, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3], tag: u32) -> hms_ipc::ObjectInfo {
        let mut o: hms_ipc::ObjectInfo = unsafe { std::mem::zeroed() };
        o.datum = datum;
        o.pos = pos;
        o.fwd = fwd;
        o.up = up;
        o.primary_tag = tag;
        o
    }

    /// An editor session over one variant file: the live objects + their per-object records.
    struct Session {
        path: std::path::PathBuf,
        live: Vec<hms_ipc::ObjectInfo>,
        meta: HashMap<u32, ObjMeta>,
        colors: HashMap<u32, (u8, u8)>,
        next_add: u32,
        /// The source record per DATUM, exactly as the app keeps it.
        src: HashMap<u32, mvar::PlacedObject>,
        /// What each live object IS (folder,item) — so a save that writes the wrong source record
        /// into an object's slot is caught. Position alone never catches it: the position comes
        /// from the live object and is right, while the TYPE comes from the joined source record.
        want_type: HashMap<u32, (u16, u8)>,
    }

    impl Session {
        /// Open a variant and mirror it into editor state, exactly as a load does:
        /// the i-th REAL object becomes datum 0xD0000000 + i.
        fn open(path: &std::path::Path) -> Option<Session> {
            let v = mvar::parse_variant(path)?;
            let mut live = Vec::new();
            let mut meta = HashMap::new();
            let mut colors = HashMap::new();
            for (i, o) in v.objects.iter().enumerate() {
                let d = 0xD000_0000u32.wrapping_add(i as u32);
                live.push(obj(d, o.pos, o.fwd, o.up, 0));
                meta.insert(d, ObjMeta {
                    folder: o.folder, item: o.item, pos: o.pos,
                    team: o.team, color: o.color, cached_type: o.cached_type,
                    spawn_seq: o.spawn_seq, respawn: o.respawn, label_idx: o.label_idx,
                    placement: o.placement, spawn_rel: o.spawn_rel,
                    ..Default::default()
                });
                colors.insert(d, (o.team, if o.color < 0 { 0xFF } else { o.color as u8 }));
            }
            let mut want_type = HashMap::new();
            let mut src = HashMap::new();
            for (i, o) in v.objects.iter().enumerate() {
                let d = 0xD000_0000u32.wrapping_add(i as u32);
                want_type.insert(d, (o.folder, o.item));
                src.insert(d, o.clone());
            }
            Some(Session { path: path.to_path_buf(), live, meta, colors, next_add: 0xF000_0000, src, want_type })
        }

        fn save(&self) -> Vec<mvar::PlacedObject> {
            let (list, unsaveable) = build_save_list(
                &self.src, &self.live, &[], &[], &self.meta, &self.colors,
                |_| None, // every add here carries its own folder/item in meta
            );
            assert_eq!(unsaveable, 0, "an object lost its palette entry");
            // The app truncates at 651 and reports the overflow; the stress rounds below keep well
            // clear of that so a truncation never silently hides a lost object.
            assert!(list.len() <= 651, "stress round overflowed the 651 slots ({} objects)", list.len());
            mvar::save_objects(&self.path, &self.path, &list, None).expect("save failed");
            list
        }

        /// Delete the n-th live object (by current position in the list).
        fn delete_at(&mut self, i: usize) {
            let d = self.live.remove(i).datum;
            self.meta.remove(&d);
            self.colors.remove(&d);
            self.want_type.remove(&d);
        }

        /// Add a new object cloned from an existing palette entry, with distinctive data so a
        /// mix-up is obvious.
        fn add(&mut self, folder: u16, item: u8, pos: [f32; 3], team: u8, respawn: u8, seq: i32) -> u32 {
            let d = self.next_add;
            self.next_add += 1;
            self.live.push(obj(d, pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0));
            self.meta.insert(d, ObjMeta {
                folder, item, pos, team, color: -1, spawn_seq: seq, respawn,
                placement: 1, label_idx: 0xFFFF, spawn_rel: -1, ..Default::default()
            });
            self.colors.insert(d, (team, 0xFF));
            self.want_type.insert(d, (folder, item));
            d
        }

        /// Every live object must be ON DISK as the thing it actually IS.
        ///
        /// Position is not enough: it comes from the live object, so it looks right even when the
        /// record written alongside it came from the wrong source object. Type is what exposes
        /// "there are initial spawns where there should have been walls".
        fn assert_types_on_disk(&self, what: &str) {
            let disk = mvar::parse_variant(&self.path).expect("reparse failed").objects;
            let key = |p: [f32; 3]| (
                (p[0] * 10.0).round() as i64,
                (p[1] * 10.0).round() as i64,
                (p[2] * 10.0).round() as i64,
            );
            // Compare the MULTISET of (position, type) pairs. Several objects can legitimately
            // share a position (a stack, or a batch dropped at one spot), so picking "the disk
            // object at this position" would flag a false mismatch; counting pairs does not.
            let mut want: HashMap<((i64, i64, i64), (u16, u8)), i32> = Default::default();
            for o in &self.live {
                let t = self.want_type.get(&o.datum).copied().expect("live object with no type");
                *want.entry((key(o.pos), t)).or_default() += 1;
            }
            let mut got: HashMap<((i64, i64, i64), (u16, u8)), i32> = Default::default();
            for d in &disk {
                *got.entry((key(d.pos), (d.folder, d.item))).or_default() += 1;
            }
            let mut wrong = 0;
            let mut first = String::new();
            for (k, n) in &want {
                let have = got.get(k).copied().unwrap_or(0);
                if have < *n {
                    wrong += *n - have;
                    if first.is_empty() {
                        let at_pos: Vec<(u16, u8)> = disk.iter()
                            .filter(|d| key(d.pos) == k.0)
                            .map(|d| (d.folder, d.item))
                            .collect();
                        first = format!("expected type {:?} at {:?}; the file has {:?} there",
                            k.1, k.0, at_pos);
                    }
                }
            }
            assert_eq!(wrong, 0,
                "{what}: {wrong} of the editor's objects are missing or written as the WRONG TYPE — {first}");
        }

        /// Re-open from disk, as the user closing and reloading the map would.
        fn reopen(&mut self) {
            let fresh = Session::open(&self.path).expect("reopen failed");
            self.live = fresh.live;
            self.meta = fresh.meta;
            self.colors = fresh.colors;
            self.want_type = fresh.want_type;
            self.src = fresh.src;
        }
    }

    fn samples() -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        for d in [
            "/mnt/games/haloreach/map_variants",
            "/mnt/games/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/map_variants",
        ] {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
                        out.push(p);
                    }
                }
            }
        }
        out.sort();
        out
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hms-stress-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    fn assert_same(a: &mvar::PlacedObject, b: &mvar::PlacedObject, what: &str) {
        assert_eq!(a.folder, b.folder, "{what}: folder");
        assert_eq!(a.item, b.item, "{what}: item");
        assert_eq!(a.team, b.team, "{what}: team");
        assert_eq!(a.color, b.color, "{what}: color");
        assert_eq!(a.spawn_seq, b.spawn_seq, "{what}: spawn_seq");
        assert_eq!(a.respawn, b.respawn, "{what}: respawn");
        assert_eq!(a.placement, b.placement, "{what}: placement flags");
        assert_eq!(a.label_idx, b.label_idx, "{what}: label");
        assert_eq!(a.cached_type, b.cached_type, "{what}: cached_type");
        assert_eq!(a.spawn_rel, b.spawn_rel, "{what}: parent");
        assert_eq!(a.flags, b.flags, "{what}: flags (occupied bit)");
        let d = (a.pos[0] - b.pos[0]).abs().max((a.pos[1] - b.pos[1]).abs()).max((a.pos[2] - b.pos[2]).abs());
        assert!(d < 0.01, "{what}: position moved by {d}");
    }

    /// Copy an EXISTING variant, then add/delete/save repeatedly, checking after every save
    /// that every object still carries the data it is supposed to.
    #[test]
    fn existing_variant_survives_repeated_add_delete_save() {
        let all = samples();
        assert!(!all.is_empty(), "no sample variants — this gate would be vacuous");
        let mut checked = 0;
        for src in all.iter().take(6) {
            let Some(v0) = mvar::parse_variant(src) else { continue };
            // need headroom: each round is net +1 object, and a near-full variant would truncate
            if v0.objects.len() < 12 || v0.objects.len() > 600 { continue; }
            let work = scratch(&format!("existing-{}.mvar", checked));
            std::fs::copy(src, &work).expect("copy failed"); // never touch the original
            let Some(mut s) = Session::open(&work) else { continue };
            let (folder, item) = (v0.objects[0].folder, v0.objects[0].item);

            for round in 0..4 {
                let before = s.live.len();
                // delete three objects from different places in the list
                s.delete_at(1);
                s.delete_at(s.live.len() / 2);
                s.delete_at(s.live.len() - 1);
                // add four with data that must survive verbatim
                let base = v0.objects[0].pos;
                let mut want: Vec<(u32, u8, u8, i32)> = Vec::new();
                for k in 0..4u8 {
                    let d = s.add(folder, item, base, k, 10 + k, (k as i32) * 7);
                    want.push((d, k, 10 + k, (k as i32) * 7));
                }
                assert_eq!(s.live.len(), before - 3 + 4, "round {round}: live count wrong");

                let expected = s.save();
                let after = mvar::parse_variant(&work).expect("reparse failed").objects;
                assert_eq!(after.len(), expected.len(), "round {round}: object count changed on disk");
                for (i, (a, e)) in after.iter().zip(expected.iter()).enumerate() {
                    assert_same(a, e, &format!("round {round} object {i}"));
                }
                // the four new objects are the LAST four, in the order they were added
                let tail = &after[after.len() - 4..];
                for (t, (_d, team, respawn, seq)) in tail.iter().zip(want.iter()) {
                    assert_eq!(t.team, *team, "round {round}: added object team");
                    assert_eq!(t.respawn, *respawn, "round {round}: added object respawn");
                    assert_eq!(t.spawn_seq, *seq, "round {round}: added object spawn_seq");
                    assert_eq!(t.folder, folder, "round {round}: added object type");
                    assert_eq!(t.flags & 1, 1, "round {round}: added object not marked occupied");
                }
                // And each object is on disk as the thing it IS — this is the check that
                // catches "initial spawns where there should have been walls".
                s.assert_types_on_disk(&format!("round {round}"));
                // reopen from disk, exactly as the user re-loading the map
                s.reopen();
                assert_eq!(s.live.len(), after.len(), "round {round}: reopen lost objects");
            }
            // the ORIGINAL file must be untouched
            let orig_now = mvar::parse_variant(src).expect("original unreadable").objects;
            assert_eq!(orig_now.len(), v0.objects.len(), "the ORIGINAL variant was modified");
            checked += 1;
        }
        assert!(checked > 0, "no variant was large enough to stress");
    }

    /// Diagnostic: how many objects are stored OUT OF BOUNDS (bsp escape)?
    /// `cargo test -p hms-app diag_oob -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diag_out_of_bounds_objects() {
        let mut tot = 0; let mut oob = 0; let mut files = 0;
        for src in samples() {
            let Some(v) = mvar::parse_variant(&src) else { continue };
            let n = v.objects.iter().filter(|o| !o.in_bounds).count();
            tot += v.objects.len(); oob += n; files += 1;
            if n > 0 {
                eprintln!("{n:>4} of {:>4} out-of-bounds  {}", v.objects.len(), src.file_name().unwrap().to_string_lossy());
            }
        }
        eprintln!("TOTAL: {oob} out-of-bounds of {tot} objects across {files} variants");
    }

    /// Diagnostic: do any shipped variants use PARENTING (spawn_rel), and how much?
    /// `cargo test -p hms-app diag_parent -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diag_parent_usage() {
        for src in samples() {
            let Some(v) = mvar::parse_variant(&src) else { continue };
            let n = v.objects.iter().filter(|o| o.spawn_rel >= 0).count();
            if n > 0 {
                let max = v.objects.iter().map(|o| o.spawn_rel).max().unwrap_or(-1);
                eprintln!("{:>4} parented / {:>4} objs  max_parent_slot={max}  {}",
                    n, v.objects.len(), src.file_name().unwrap().to_string_lossy());
            }
        }
        eprintln!("(no output above = nothing in the sample set uses parenting)");
    }

    /// Diagnostic: does each variant's STORED quota table agree with the real object counts?
    /// `cargo test -p hms-app diag_quota -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diag_quota_agreement() {
        for src in samples() {
            let Some(v) = mvar::parse_variant(&src) else { continue };
            if v.objects.is_empty() { continue; }
            let Some(pay) = std::fs::read(&src).ok().and_then(|d| mvar::blf_mvar_payload_pub(&d)) else { continue };
            let q = mvar::quota_table(&pay);
            let mut truth = [0u32; 256];
            for o in &v.objects {
                if o.folder != 0xFFFF && o.item != 0xFF && (o.folder as usize) < 256 { truth[o.folder as usize] += 1; }
            }
            let mut bad = 0;
            let mut worst = String::new();
            for (i, e) in q.iter().enumerate() {
                let t = truth.get(i).copied().unwrap_or(0).min(255) as u8;
                if e.2 != t {
                    bad += 1;
                    if worst.is_empty() { worst = format!("q[{i}] stored {} vs true {t}", e.2); }
                }
            }
            eprintln!("{:>4} objs  {:>3} quotas  {:>3} mismatched  {}  {}",
                v.objects.len(), q.len(), bad, worst, src.file_name().unwrap().to_string_lossy());
        }
    }

    /// Diagnostic: which FIELD does a no-edit save change?
    /// `cargo test -p hms-app diag_noedit -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diag_noedit_field_diff() {
        for src in samples() {
            let Some(v) = mvar::parse_variant(&src) else { continue };
            if v.objects.is_empty() { continue; }
            let work = scratch("diag.mvar");
            std::fs::copy(&src, &work).unwrap();
            let before = std::fs::read(&work).unwrap();
            let Some(s) = Session::open(&work) else { continue };
            let list = build_save_list(&s.src, &s.live, &[], &[], &s.meta, &s.colors, |_| None).0;
            mvar::save_objects(&work, &work, &list, None).unwrap();
            if std::fs::read(&work).unwrap() == before { continue; }
            eprintln!("=== CHANGED: {}", src.display());
            eprintln!("  source objects {} -> list {}", v.objects.len(), list.len());
            for (i, o) in v.objects.iter().take(5).enumerate() {
                eprintln!("    obj{i}: folder={} item={} team={} flags={:#x} pos=({:.1},{:.1},{:.1})",
                    o.folder, o.item, o.team, o.flags, o.pos[0], o.pos[1], o.pos[2]);
            }
            eprintln!("    map_id={} num_quotas={}", v.map_id, v.num_quotas);
            {
                let pb = mvar::blf_mvar_payload_pub(&before).unwrap();
                let pa = mvar::blf_mvar_payload_pub(&std::fs::read(&work).unwrap()).unwrap();
                let (qb, qa) = (mvar::quota_table(&pb), mvar::quota_table(&pa));
                // true count per folder from the object list
                let mut truth = [0u32; 256];
                for o in &v.objects {
                    if o.folder != 0xFFFF && o.item != 0xFF && (o.folder as usize) < 256 { truth[o.folder as usize] += 1; }
                }
                for (i, (a, b)) in qb.iter().zip(qa.iter()).enumerate() {
                    if a != b {
                        eprintln!("  quota[{i}]: stored(min {},max {},placed {}) -> written(min {},max {},placed {}) | TRUE count {}",
                            a.0, a.1, a.2, b.0, b.1, b.2, truth[i]);
                    }
                }
            }
            let after = mvar::parse_variant(&work).unwrap().objects;
            for (i, (a, b)) in v.objects.iter().zip(after.iter()).enumerate() {
                let mut d: Vec<String> = Vec::new();
                if a.folder != b.folder { d.push(format!("folder {}->{}", a.folder, b.folder)); }
                if a.item != b.item { d.push(format!("item {}->{}", a.item, b.item)); }
                if a.team != b.team { d.push(format!("team {}->{}", a.team, b.team)); }
                if a.color != b.color { d.push(format!("color {}->{}", a.color, b.color)); }
                if a.spawn_seq != b.spawn_seq { d.push(format!("seq {}->{}", a.spawn_seq, b.spawn_seq)); }
                if a.respawn != b.respawn { d.push(format!("respawn {}->{}", a.respawn, b.respawn)); }
                if a.placement != b.placement { d.push(format!("placement {:#x}->{:#x}", a.placement, b.placement)); }
                if a.label_idx != b.label_idx { d.push(format!("label {}->{}", a.label_idx, b.label_idx)); }
                if a.cached_type != b.cached_type { d.push(format!("ctype {}->{}", a.cached_type, b.cached_type)); }
                if a.spawn_rel != b.spawn_rel { d.push(format!("parent {}->{}", a.spawn_rel, b.spawn_rel)); }
                if a.flags != b.flags { d.push(format!("flags {:#x}->{:#x}", a.flags, b.flags)); }
                if a.boundary_shape != b.boundary_shape { d.push(format!("bshape {}->{}", a.boundary_shape, b.boundary_shape)); }
                if a.up_quant != b.up_quant { d.push(format!("up_q {}->{}", a.up_quant, b.up_quant)); }
                if a.forward_angle_q != b.forward_angle_q { d.push(format!("yaw_q {}->{}", a.forward_angle_q, b.forward_angle_q)); }
                if a.up_is_global != b.up_is_global { d.push(format!("up_glob {}->{}", a.up_is_global, b.up_is_global)); }
                if a.in_bounds != b.in_bounds { d.push(format!("in_bounds {}->{}", a.in_bounds, b.in_bounds)); }
                if a.bsp_index != b.bsp_index { d.push(format!("bsp {}->{}", a.bsp_index, b.bsp_index)); }
                let pd = (a.pos[0]-b.pos[0]).abs().max((a.pos[1]-b.pos[1]).abs()).max((a.pos[2]-b.pos[2]).abs());
                if pd > 1e-4 { d.push(format!("pos by {pd}")); }
                if !d.is_empty() { eprintln!("  obj {i}: {}", d.join(", ")); }
            }
            return;
        }
    }

    /// The NEW-variant flow: File▸New starts from a template .mvar and clears its objects, so a
    /// fresh map is "every source object deleted, then N adds". Build one, save, REOPEN, edit it,
    /// and save again — checking every object's data survives each round trip.
    #[test]
    fn new_variant_from_template_survives_save_reopen_edit_save() {
        let all = samples();
        assert!(!all.is_empty(), "no sample variants — this gate would be vacuous");
        // a template just needs a palette to take folder/item from
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 8 && v.objects.len() <= 600).then(|| (p.clone(), v))
        }) else { panic!("no usable template variant") };

        let work = scratch("new-variant.mvar");
        std::fs::copy(&src, &work).unwrap();
        let mut s = Session::open(&work).expect("open template");

        // --- start empty, like File▸New ---
        while !s.live.is_empty() {
            s.delete_at(0);
        }
        assert!(s.live.is_empty());

        // --- place a batch with data that must survive verbatim ---
        let (f0, i0) = (v0.objects[0].folder, v0.objects[0].item);
        let (f1, i1) = (v0.objects[1].folder, v0.objects[1].item);
        let base = v0.objects[0].pos;
        const N: usize = 40;
        let mut want: Vec<(u16, u8, u8, u8, i32)> = Vec::new();
        for k in 0..N {
            let (f, i) = if k % 2 == 0 { (f0, i0) } else { (f1, i1) };
            let team = (k % 9) as u8;
            let respawn = (k % 60) as u8;
            let seq = (k as i32 % 100) - 50;
            let pos = [base[0] + (k as f32) * 0.25, base[1], base[2]];
            s.add(f, i, pos, team, respawn, seq);
            want.push((f, i, team, respawn, seq));
        }
        s.save();

        // --- reopen from disk and verify every placement ---
        s.reopen();
        let on_disk = mvar::parse_variant(&work).unwrap().objects;
        assert_eq!(on_disk.len(), N, "new variant: expected {N} objects, got {}", on_disk.len());
        for (k, (o, w)) in on_disk.iter().zip(want.iter()).enumerate() {
            assert_eq!(o.folder, w.0, "new variant obj {k}: folder");
            assert_eq!(o.item, w.1, "new variant obj {k}: item");
            assert_eq!(o.team, w.2, "new variant obj {k}: team");
            assert_eq!(o.respawn, w.3, "new variant obj {k}: respawn");
            assert_eq!(o.spawn_seq, w.4, "new variant obj {k}: spawn_seq");
            assert_eq!(o.flags & 1, 1, "new variant obj {k}: not marked occupied — the game would not spawn it");
        }

        // --- edit it: delete a few, add a few, save again ---
        s.delete_at(0);
        s.delete_at(5);
        s.delete_at(s.live.len() - 1);
        for k in 0..3u8 {
            s.add(f1, i1, base, k, 5 + k, 33 + k as i32);
        }
        let expected = s.save();
        s.reopen();
        let after = mvar::parse_variant(&work).unwrap().objects;
        assert_eq!(after.len(), N - 3 + 3, "after edit: wrong object count");
        for (i, (a, e)) in after.iter().zip(expected.iter()).enumerate() {
            assert_same(a, e, &format!("edited new variant object {i}"));
        }
        s.assert_types_on_disk("new variant after edit");
        // and it is stable: saving again changes nothing
        let bytes = std::fs::read(&work).unwrap();
        s.save();
        assert_eq!(std::fs::read(&work).unwrap(), bytes, "new variant: re-save is not a fixed point");
    }

    /// NO CUMULATIVE DRIFT. Positions and rotations are stored QUANTISED, so a save that
    /// re-encodes them must be a fixed point — otherwise every save nudges the map and after a
    /// few sessions everything has walked. Ten edit+save rounds, then check the objects that were
    /// never touched are still EXACTLY where they started.
    #[test]
    fn repeated_edit_save_rounds_do_not_drift_untouched_objects() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 40 && v.objects.len() <= 560).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("drift.mvar");
        std::fs::copy(&src, &work).unwrap();
        let mut s = Session::open(&work).expect("open");

        // these are never edited; they must not move by a single quantisation step
        let watch: Vec<(usize, mvar::PlacedObject)> =
            (0..10).map(|i| (i, v0.objects[i].clone())).collect();

        for round in 0..10 {
            // churn the TAIL only, leaving the watched prefix alone
            let last = s.live.len() - 1;
            s.delete_at(last);
            s.add(v0.objects[0].folder, v0.objects[0].item, v0.objects[0].pos, (round % 9) as u8, 7, round as i32);
            s.save();
            s.reopen();
            let disk = mvar::parse_variant(&work).unwrap().objects;
            for (i, orig) in &watch {
                let now = &disk[*i];
                assert_eq!(now.up_quant, orig.up_quant, "round {round}: object {i} up-vector drifted");
                assert_eq!(now.forward_angle_q, orig.forward_angle_q, "round {round}: object {i} yaw drifted");
                assert_eq!(now.up_is_global, orig.up_is_global, "round {round}: object {i} up-mode drifted");
                let d = (now.pos[0] - orig.pos[0]).abs()
                    .max((now.pos[1] - orig.pos[1]).abs())
                    .max((now.pos[2] - orig.pos[2]).abs());
                assert!(d < 1e-4, "round {round}: object {i} position drifted by {d}");
                assert_eq!(now.folder, orig.folder, "round {round}: object {i} changed TYPE");
                assert_eq!(now.team, orig.team, "round {round}: object {i} changed team");
                assert_eq!(now.placement, orig.placement, "round {round}: object {i} changed placement flags");
            }
        }
    }

    /// Moving and ROTATING an object must persist what the user set, and then stay put across
    /// further saves (the rotation is re-quantised only when it actually changed).
    #[test]
    fn moved_and_rotated_objects_persist_and_then_hold_still() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 20 && v.objects.len() <= 560).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("rot.mvar");
        std::fs::copy(&src, &work).unwrap();
        let mut s = Session::open(&work).expect("open");

        // turn object 3 by 90 degrees about Z and nudge it, in-bounds
        let target = 3usize;
        let newpos = [v0.objects[target].pos[0] + 1.0, v0.objects[target].pos[1], v0.objects[target].pos[2]];
        s.live[target].pos = newpos;
        s.live[target].fwd = [0.0, 1.0, 0.0];
        s.live[target].up = [0.0, 0.0, 1.0];
        s.save();
        s.reopen();

        let disk = mvar::parse_variant(&work).unwrap().objects;
        let o = &disk[target];
        let ang = o.fwd[0].atan2(o.fwd[1]); // 0 when fwd == +Y
        assert!(ang.abs() < 0.01, "rotation did not persist: fwd = {:?}", o.fwd);
        let dp = (o.pos[0] - newpos[0]).abs().max((o.pos[1] - newpos[1]).abs()).max((o.pos[2] - newpos[2]).abs());
        assert!(dp < 0.01, "move did not persist: off by {dp}");

        // now it must HOLD: three more no-edit saves change nothing
        let stable = std::fs::read(&work).unwrap();
        for round in 0..3 {
            s.reopen();
            s.save();
            assert_eq!(std::fs::read(&work).unwrap(), stable,
                "save #{} after a rotation changed the file — the rotation is drifting", round + 2);
        }
    }

    /// A parent link is a SLOT INDEX, so deleting anything shifts it. The child
    /// must follow its real parent, and be orphaned (not silently re-parented to a stranger) when
    /// the parent is deleted.
    #[test]
    fn parent_links_follow_their_parent_through_deletes() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 20 && v.objects.len() <= 560).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("parent.mvar");
        std::fs::copy(&src, &work).unwrap();
        let mut s = Session::open(&work).expect("open");

        // child at index 10 is parented to the object at slot 6
        let (child, parent) = (10usize, 6usize);
        let parent_pos = v0.objects[parent].pos;
        s.meta.get_mut(&(0xD000_0000 + child as u32)).unwrap().spawn_rel = parent as i32;

        // delete TWO objects before the parent -> the parent shifts down by 2
        s.delete_at(0);
        s.delete_at(0);
        s.save();
        let disk = mvar::parse_variant(&work).unwrap().objects;
        let new_child = child - 2;
        let link = disk[new_child].spawn_rel;
        assert!(link >= 0, "the parent link was lost");
        let pointed = &disk[link as usize];
        let d = (pointed.pos[0] - parent_pos[0]).abs()
            .max((pointed.pos[1] - parent_pos[1]).abs())
            .max((pointed.pos[2] - parent_pos[2]).abs());
        assert!(d < 0.01, "child re-parented to the WRONG object (link {link}, off by {d})");
        assert_eq!(link as usize, parent - 2, "link should have shifted down by the 2 deletions");

        // now delete the PARENT itself -> the child must be orphaned, not pointed elsewhere
        s.reopen();
        s.meta.get_mut(&(0xD000_0000 + new_child as u32)).unwrap().spawn_rel = (parent - 2) as i32;
        s.delete_at(parent - 2);
        s.save();
        let disk2 = mvar::parse_variant(&work).unwrap().objects;
        assert_eq!(disk2[new_child - 1].spawn_rel, -1,
            "child kept a parent link after its parent was deleted — it would attach to a stranger");
    }

    /// An angle typed into the properties panel must survive the save + reload and read
    /// back as the SAME angle. The file stores a quantised up-vector + yaw, so this is the check
    /// that the user-facing number is not lying about what got written.
    #[test]
    fn typed_rotation_angles_survive_save_and_reload() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 20 && v.objects.len() <= 560).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("euler.mvar");
        std::fs::copy(&src, &work).unwrap();

        // a spread of angles, including ones that exercise all three axes at once
        let cases: [[f32; 3]; 6] = [
            [0.0, 0.0, 90.0],
            [0.0, 0.0, -135.0],
            [0.0, 30.0, 0.0],
            [45.0, 0.0, 0.0],
            [20.0, -15.0, 60.0],
            [0.0, 0.0, 0.0],
        ];
        for (k, want) in cases.iter().enumerate() {
            let mut s = Session::open(&work).expect("open");
            let target = 3usize + k;
            let (f, u) = euler::from_euler_deg(*want, 1.0, 1.0);
            s.live[target].fwd = f;
            s.live[target].up = u;
            s.save();

            let disk = mvar::parse_variant(&work).unwrap().objects;
            let got = euler::to_euler_deg(disk[target].fwd, disk[target].up);
            // the file stores yaw in 14 bits over [-pi,pi] (~0.022 deg) and the up vector in 20
            // bits, so allow a fraction of a degree — but no more.
            let (gf, gu) = euler::from_euler_deg(got, 1.0, 1.0);
            let df = (0..3).map(|i| (gf[i] - f[i]).abs()).fold(0.0f32, f32::max);
            let du = (0..3).map(|i| (gu[i] - u[i]).abs()).fold(0.0f32, f32::max);
            assert!(df < 0.01 && du < 0.01,
                "angles {want:?} came back as {got:?} (fwd off {df}, up off {du}) — the panel would show the wrong number");
        }
    }

    /// The type gate must actually BITE. Deliberately write one object out as the wrong type and
    /// confirm `assert_types_on_disk` catches it — otherwise every "types are fine" result above is
    /// worthless.
    #[test]
    fn the_type_gate_detects_a_wrong_type() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 20 && v.objects.len() <= 500).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("gatecheck.mvar");
        std::fs::copy(&src, &work).unwrap();
        let s = Session::open(&work).expect("open");

        // sanity: as loaded, the gate is happy
        s.assert_types_on_disk("unmodified");

        // now write the file with ONE object's type swapped for another's
        let (mut list, _) = build_save_list(&s.src, &s.live, &[], &[], &s.meta, &s.colors, |_| None);
        let victim = 5usize;
        let donor = list.iter().position(|o| (o.folder, o.item) != (list[victim].folder, list[victim].item))
            .expect("variant has only one object type");
        list[victim].folder = list[donor].folder;
        list[victim].item = list[donor].item;
        mvar::save_objects(&work, &work, &list, None).unwrap();

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.assert_types_on_disk("deliberately corrupted");
        }));
        assert!(caught.is_err(), "the type gate did NOT notice an object written as the wrong type");
    }

    /// Edit → save → EDIT AGAIN WITHOUT RELOADING → save.
    ///
    /// The app does not re-sync after a save, so its objects keep the datums they were loaded
    /// with while the FILE now has a different number of objects in different slots. A save that
    /// joined "in-memory datum 0xD0000000+i" to "source slot i" would match against the wrong
    /// records: objects whose index is past the new count get treated as brand-new adds, and
    /// anything without a resolvable palette entry is silently dropped.
    #[test]
    fn second_save_without_reloading_loses_nothing() {
        let all = samples();
        let Some((src, v0)) = all.iter().find_map(|p| {
            let v = mvar::parse_variant(p)?;
            (v.objects.len() >= 120 && v.objects.len() <= 500).then(|| (p.clone(), v))
        }) else { panic!("no usable variant") };
        let work = scratch("second-save.mvar");
        std::fs::copy(&src, &work).unwrap();
        let mut s = Session::open(&work).expect("open");
        let (folder, item) = (v0.objects[0].folder, v0.objects[0].item);
        let base = v0.objects[0].pos;

        // --- session 1: delete a scattered set, add a batch, save ---
        for k in 0..20 {
            let i = s.live.len() - 1 - k * 3; // scattered, from the tail inward
            s.delete_at(i);
        }
        for k in 0..25 {
            // distinct positions so the identity comparison below is meaningful
            let p = [base[0] + k as f32 * 0.5, base[1], base[2]];
            s.add(folder, item, p, (k % 9) as u8, 10, k as i32);
        }
        let intended_1 = s.live.len();
        s.save();
        let on_disk_1 = mvar::parse_variant(&work).unwrap().objects.len();
        assert_eq!(on_disk_1, intended_1, "first save already lost objects");

        // --- session 2: KEEP EDITING without reloading, then save again ---
        s.delete_at(0);
        s.delete_at(4);
        for k in 0..3 {
            let p = [base[0], base[1] + 0.5 + k as f32 * 0.5, base[2]];
            s.add(folder, item, p, k as u8, 20, 99);
        }
        let intended_2 = s.live.len();
        s.save();
        let disk2 = mvar::parse_variant(&work).unwrap().objects;

        // Compare IDENTITY, not just the count: objects can be dropped while others take their
        // place, leaving the total unchanged.
        let key = |p: [f32; 3]| (
            (p[0] * 10.0).round() as i64,
            (p[1] * 10.0).round() as i64,
            (p[2] * 10.0).round() as i64,
        );
        let mut want: std::collections::HashMap<(i64, i64, i64), i32> = Default::default();
        for o in &s.live {
            *want.entry(key(o.pos)).or_default() += 1;
        }
        let mut got: std::collections::HashMap<(i64, i64, i64), i32> = Default::default();
        for o in &disk2 {
            *got.entry(key(o.pos)).or_default() += 1;
        }
        let mut missing = 0i32;
        for (k, n) in &want {
            let have = got.get(k).copied().unwrap_or(0);
            if have < *n {
                missing += *n - have;
            }
        }
        assert_eq!(
            disk2.len(), intended_2,
            "SECOND save wrote {} objects, editor had {intended_2}", disk2.len()
        );
        assert_eq!(
            missing, 0,
            "SECOND save LOST {missing} of the editor's objects (they are not in the file at all)"
        );

        // And each object must still BE what it was: the position comes from the live object
        // (so it always looks right) while the TYPE comes from whichever source record the save
        // joined to. A wrong join is how a wall gets written out as an initial spawn.
        let mut wrong_type = 0;
        let mut first = String::new();
        for o in &s.live {
            let want = s.want_type.get(&o.datum).copied().expect("live object with no type");
            let k = key(o.pos);
            let found = disk2.iter().find(|d| key(d.pos) == k);
            if let Some(d) = found {
                if (d.folder, d.item) != want {
                    wrong_type += 1;
                    if first.is_empty() {
                        first = format!("datum {:#x} should be type {:?} but was written as {:?}",
                            o.datum, want, (d.folder, d.item));
                    }
                }
            }
        }
        assert_eq!(wrong_type, 0,
            "SECOND save wrote {wrong_type} objects as the WRONG TYPE — {first}");
    }

    /// A save with no edits must not change any OBJECT, and must converge immediately.
    ///
    /// It is not required to be byte-identical: the placeable-object QUOTA table is deliberately
    /// recomputed from what was actually written (the engine's Forge budget reads it), and a few
    /// shipped variants carry a stale or garbage table — 22f4eb5f has 81 wrong entries, including
    /// minimums above their maximums. Correcting those is the point. What must hold is that no
    /// object record changes, and that a SECOND no-edit save is byte-identical to the first.
    #[test]
    fn saving_without_editing_preserves_every_object() {
        let all = samples();
        assert!(!all.is_empty(), "no sample variants — this gate would be vacuous");
        let mut checked = 0;
        let mut corrected = 0;
        for src in all.iter().take(12) {
            let Some(v) = mvar::parse_variant(src) else { continue };
            if v.objects.is_empty() { continue; }
            let work = scratch(&format!("noedit-{}.mvar", checked));
            std::fs::copy(src, &work).unwrap();
            let before = std::fs::read(&work).unwrap();
            let Some(mut s) = Session::open(&work) else { continue };
            s.save();
            // every object identical, field for field
            let after = mvar::parse_variant(&work).unwrap().objects;
            assert_eq!(after.len(), v.objects.len(), "{}: object count changed", src.display());
            for (i, (a, b)) in after.iter().zip(v.objects.iter()).enumerate() {
                assert_same(a, b, &format!("{} object {i}", src.file_name().unwrap().to_string_lossy()));
            }
            let first = std::fs::read(&work).unwrap();
            if first != before {
                corrected += 1; // stale quota table rewritten — allowed, but must converge
            }
            // saving again must now change nothing at all
            s.reopen();
            s.save();
            assert_eq!(std::fs::read(&work).unwrap(), first,
                "{}: a no-edit save is not a fixed point", src.display());
            checked += 1;
        }
        assert!(checked > 0);
        eprintln!("no-edit save: {checked} variants checked, {corrected} had a stale quota table corrected");
    }

    /// Saving repeatedly with no edits must be a fixed point.
    #[test]
    fn repeated_saves_are_a_fixed_point() {
        let all = samples();
        let mut checked = 0;
        for src in all.iter().take(6) {
            let Some(v) = mvar::parse_variant(src) else { continue };
            if v.objects.len() < 8 { continue; }
            let work = scratch(&format!("fixed-{}.mvar", checked));
            std::fs::copy(src, &work).unwrap();
            let Some(mut s) = Session::open(&work) else { continue };
            s.delete_at(2);
            s.add(v.objects[0].folder, v.objects[0].item, v.objects[0].pos, 3, 25, 11);
            s.save();
            let first = std::fs::read(&work).unwrap();
            for round in 0..3 {
                s.reopen();
                s.save();
                let again = std::fs::read(&work).unwrap();
                assert_eq!(first, again, "{}: save #{} differs — the save is not idempotent", src.display(), round + 2);
            }
            checked += 1;
        }
        assert!(checked > 0);
    }
}
