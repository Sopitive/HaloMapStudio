//! Extracting the MAP VARIANT and the GAME VARIANT out of a Reach saved film.
//!
//! A film's `flmh` chunk embeds both whole, as raw in-memory structs (see
//! `re-notes/reach_saved_film_layout.md` §3.3 / §3.4). This module turns them into files.
//!
//! * **Map variant → a real, loadable `.mvar`.** The record layout of the embedded
//!   `s_variant_object` is fully solved against real data (`film::VariantObject`), so each object
//!   converts to `mvar::PlacedObject` and HMS's own byte-exact Reach encoder writes the file.
//!   That encoder is TEMPLATE-DRIVEN (it copies the header bits it does not edit verbatim), so
//!   extraction needs one existing `.mvar` for the SAME BASE MAP as a canvas. MCC ships 275 of
//!   them in `haloreach/hopper_map_variants` covering 27 map ids, so this works on any install;
//!   `find_map_variant_template` reports plainly when it cannot find one.
//!
//! * **Game variant → NOT a `.bin`, and this module says so.** A Reach gametype on disk is a
//!   bit-packed `mpvr` chunk (version 54) that squeezes the 64516-byte in-memory
//!   `c_game_variant` into ~20.5 KB. Writing that encoder means reverse-engineering the whole
//!   Megalo variant bit layout, which is a separate project; fabricating a file we cannot verify
//!   would be worse than useless. So the game variant is exported as its raw struct plus a
//!   readable sidecar, and the honest limitation is stated to the user.
//!
//!   Matching the film's gametype to a `.bin` already on disk was tried and REJECTED: MCC keeps
//!   the lineage uid across derived saves, so one uid (`0x4a6994bf5d589b43`) covers gametypes
//!   titled "Testing", "Castle Wars Xtr2", "SCARAB MODE ACOS" and more. Matching on it would hand
//!   the user the wrong gametype. Measured on 39 real films against 720 on-disk gametypes.

use crate::film::{self, Film};
use crate::mvar;
use std::path::{Path, PathBuf};

/// Why an extraction could not be done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportError {
    /// The film was recorded on a stock map: the embedded `c_map_variant` is the engine's empty
    /// one, so there is no Forge scene to extract.
    StockMap,
    /// No `.mvar` for this base map anywhere HMS looks, so the encoder has no canvas.
    NoTemplate(u32),
    /// A template was found but it is not really the same map.
    TemplateMismatch { what: &'static str, film: u32, template: u32 },
    Encode(String),
    Io(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExportError::StockMap => write!(
                f,
                "this film was recorded on a stock map, so there is no Forge variant to extract"
            ),
            ExportError::NoTemplate(id) => write!(
                f,
                "no existing .mvar for map id {id} was found to use as a canvas -- put any map \
                 variant saved on that map in the game's map_variants folder and try again"
            ),
            ExportError::TemplateMismatch { what, film, template } => write!(
                f,
                "the canvas variant's {what} ({template:#x}) does not match the film's ({film:#x}), \
                 so it is not the same map build"
            ),
            ExportError::Encode(e) => write!(f, "{e}"),
            ExportError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// What `export_map_variant` produced.
#[derive(Clone, Debug)]
pub struct MapVariantExport {
    pub path: PathBuf,
    /// Objects written into the file.
    pub objects: usize,
    /// The `.mvar` used as the canvas.
    pub template: PathBuf,
    pub title: String,
    pub map_id: u32,
}

/// What `export_game_variant` produced.
#[derive(Clone, Debug)]
pub struct GameVariantExport {
    /// The raw `c_game_variant` struct.
    pub struct_path: PathBuf,
    /// The human-readable sidecar.
    pub notes_path: PathBuf,
    pub bytes: usize,
    pub title: String,
    pub engine_type: u32,
}

/// Every `.mvar` HMS can see, with the map id it targets. Shared by the template search and the
/// GUI's "which canvas will be used" hint.
pub fn known_map_variants() -> Vec<(PathBuf, u32)> {
    let mut out = Vec::new();
    for d in crate::mapcat::variant_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
                continue;
            }
            // Read only the fixed-offset chdr map id; parsing the whole variant here would be
            // far too slow for a folder with hundreds of files.
            let Ok(head) = read_head(&p, 0x80) else { continue };
            if head.len() < 0x70 || &head[0x30..0x34] != b"chdr" {
                continue;
            }
            let map_id = u32::from_le_bytes([head[0x6C], head[0x6D], head[0x6E], head[0x6F]]);
            out.push((p, map_id));
        }
    }
    out
}

fn read_head(p: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf)?;
    buf.truncate(got);
    Ok(buf)
}

/// Pick the canvas `.mvar` for `map_id`. Prefers the SMALLEST candidate (fewest surprises in the
/// bits the encoder carries over) and, among equals, a stable alphabetical order so the choice is
/// reproducible.
pub fn find_map_variant_template(map_id: u32) -> Option<PathBuf> {
    let mut cands: Vec<(u64, PathBuf)> = known_map_variants()
        .into_iter()
        .filter(|(_, id)| *id == map_id)
        .filter_map(|(p, _)| std::fs::metadata(&p).ok().map(|m| (m.len(), p)))
        .collect();
    cands.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    cands.into_iter().next().map(|(_, p)| p)
}

/// A filename-safe stem for an extracted file, from the variant's own title.
pub fn safe_stem(title: &str, fallback: &str) -> String {
    let mut s: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    s = s.trim().trim_matches('.').trim().to_string();
    while s.contains("__") {
        s = s.replace("__", "_");
    }
    if s.is_empty() { fallback.to_string() } else { s }
}

/// Turn the film's embedded `c_map_variant` into a real `.mvar` at `dst`.
/// #film-action  The ONE place that decides whether a film's Forge scene can be drawn, writes
/// it out, and says in plain words why not when it cannot.
///
/// Both hosts need this and they used to have one copy each: the GUI's `film_load_scene` and the
/// script host's `film scene` verb. They APPLY the result differently and always will — the GUI
/// goes through its asynchronous `pending_open_variant` pipeline so the canvas switch is not a
/// side effect of the import, the script host calls `apply_variant` straight — but the decision,
/// the export and the wording belong together, because when they drifted the script host simply
/// never drew a scene at all and every scripted capture came out as bare BSP.
///
/// Returns the written `.mvar` and what went into it, or a message that is safe to show a user.
pub fn scene_request(f: &Film) -> Result<MapVariantExport, String> {
    if f.header.map_variant.is_empty() {
        return Err("this film was recorded on a stock map — there is no Forge scene to draw".into());
    }
    let dir = match crate::project::settings_dir() {
        Some(d) => d.join("films"),
        None => std::env::temp_dir().join("hms_film_scene"),
    };
    let _ = std::fs::create_dir_all(&dir);
    let stem = safe_stem(&f.header.map_variant.metadata.title, "film_variant");
    let dst = dir.join(format!("{stem}.mvar"));
    export_map_variant(f, &dst).map_err(|e| match e {
        ExportError::NoTemplate(id) => {
            // Say what the user can act on. The film carries its own objects, but HMS writes
            // them into an existing `.mvar` for that map to use as a canvas, so without one
            // there is nothing to write into.
            let dirs: Vec<String> =
                crate::mapcat::variant_dirs().into_iter().map(|d| d.display().to_string()).collect();
            format!(
                "the map is loaded, but the Forge scene cannot be built: HMS needs one saved \
                 .mvar for {} (map id {id}) to use as a canvas, and none was found. Save any \
                 Forge variant on that map in the game, then reopen this film.{}",
                f.header.scenario_name(),
                if dirs.is_empty() { String::new() } else { format!(" Looked in: {}", dirs.join(", ")) }
            )
        }
        other => format!("the Forge scene could not be built: {other}"),
    })
}

pub fn export_map_variant(f: &Film, dst: &Path) -> Result<MapVariantExport, ExportError> {
    let mv = &f.header.map_variant;
    if mv.is_empty() {
        return Err(ExportError::StockMap);
    }
    let template = find_map_variant_template(mv.map_id).ok_or(ExportError::NoTemplate(mv.map_id))?;
    let bytes = export_map_variant_with_template(f, &template)?;
    std::fs::write(dst, &bytes).map_err(|e| ExportError::Io(format!("{}: {e}", dst.display())))?;
    Ok(MapVariantExport {
        path: dst.to_path_buf(),
        objects: mv.present_objects().count(),
        template,
        title: mv.metadata.title.clone(),
        map_id: mv.map_id,
    })
}

/// The bytes `export_map_variant` would write, with the canvas chosen explicitly. Split out so
/// tests can drive it without touching the filesystem.
pub fn export_map_variant_with_template(f: &Film, template: &Path) -> Result<Vec<u8>, ExportError> {
    let mv = &f.header.map_variant;
    if mv.is_empty() {
        return Err(ExportError::StockMap);
    }
    let src = mvar::read_expanded(template).map_err(|e| ExportError::Io(format!("{}: {e}", template.display())))?;
    // The canvas must really be the same map: the encoder carries its map id, scenario palette
    // CRC and quota count over verbatim, and a mismatch would make the file describe the wrong
    // palette (the game would then rebuild the variant from the scenario and drop everything).
    let tv = mvar::parse_variant(template)
        .ok_or_else(|| ExportError::Encode(format!("{} is not a readable Reach map variant", template.display())))?;
    if tv.globals.map_id != mv.map_id {
        return Err(ExportError::TemplateMismatch { what: "map id", film: mv.map_id, template: tv.globals.map_id });
    }
    if tv.globals.palette_crc != mv.palette_crc {
        return Err(ExportError::TemplateMismatch {
            what: "scenario palette CRC",
            film: mv.palette_crc,
            template: tv.globals.palette_crc,
        });
    }
    if tv.globals.num_quotas != mv.num_quotas {
        return Err(ExportError::TemplateMismatch {
            what: "quota count",
            film: mv.num_quotas as u32,
            template: tv.globals.num_quotas as u32,
        });
    }

    let objects: Vec<mvar::PlacedObject> = mv.present_objects().map(|o| o.to_placed_object()).collect();
    let edits = mvar::HeaderEdits {
        title: Some(mv.metadata.title.clone()),
        description: Some(mv.metadata.description.clone()),
        author: Some(mv.metadata.created_by.clone()),
        editor: Some(mv.metadata.modified_by.clone()),
        labels: Some(mv.labels.clone()),
        created: Some((mv.metadata.created_unix, mv.metadata.created_xuid)),
        modified: Some((mv.metadata.modified_unix, mv.metadata.modified_xuid)),
        category: Some(mv.metadata.category as u8 as i8),
        budget_max: Some(mv.budget_max),
        bounds: Some(mv.world_bounds),
        quotas: Some(mv.quotas.iter().take(mv.num_quotas as usize).map(|q| (q[0], q[1])).collect()),
    };
    let (out, written) = mvar::rebuild_file(&src, &objects, Some(&edits)).map_err(ExportError::Encode)?;
    if written != objects.len() {
        return Err(ExportError::Encode(format!(
            "the canvas variant only had room for {written} of {} objects",
            objects.len()
        )));
    }
    Ok(out)
}

/// Write the embedded `c_game_variant` out: the raw struct plus a readable sidecar. Returns the
/// paths. `dst` is used as the stem; the two files get `.reach_gv` and `.txt`.
pub fn export_game_variant(f: &Film, dst: &Path) -> Result<GameVariantExport, ExportError> {
    let raw = game_variant_bytes(f)
        .ok_or_else(|| ExportError::Encode("the film's game variant block is truncated".into()))?;
    let struct_path = dst.with_extension("reach_gv");
    let notes_path = dst.with_extension("txt");
    std::fs::write(&struct_path, raw).map_err(|e| ExportError::Io(format!("{}: {e}", struct_path.display())))?;
    std::fs::write(&notes_path, game_variant_notes(f))
        .map_err(|e| ExportError::Io(format!("{}: {e}", notes_path.display())))?;
    Ok(GameVariantExport {
        struct_path,
        notes_path,
        bytes: raw.len(),
        title: f.header.game_variant.title.clone(),
        engine_type: f.header.game_engine_type,
    })
}

/// The raw `c_game_variant` struct inside the film's `flmh` chunk.
pub fn game_variant_bytes(f: &Film) -> Option<&[u8]> {
    let flmh = f.chunks.iter().find(|c| &c.magic == b"flmh")?;
    let p = flmh.payload(&f.bytes);
    let at = film::flmh::GAME_OPTIONS + film::game_options::GAME_VARIANT;
    p.get(at..at + film::game_options::GAME_VARIANT_LEN)
}

/// Reach game engine type names, as the content header's `engine` field numbers them.
///
/// **CORRECTED 2026-10-11 (#gamevariant).** These used to be the Halo 3 gametype list
/// ("Capture the Flag", "Slayer", "Oddball", ...), which is wrong for Reach and made the Theater
/// card label every single multiplayer film "Slayer". Reach does not number its gametypes here at
/// all: the engine type says which KIND of variant the file holds, and the gametype itself comes
/// from the Megalo script inside it. Measured over all 65 real Reach films on this machine
/// (`film_variant::tests::real_films_decode_inside_every_declared_range`, which also checks this
/// field against `c_game_variant+0x00` and finds 65 of 65 agree): the films whose own title is
/// `"Forge"` carry **1**, the Firefight films carry **4**, and every other multiplayer film
/// carries **2**. The four names below are also exactly the four variant blocks the engine's own
/// tag definition for `game_engine_settings_definition` ('wezr') declares — sandbox, campaign and
/// survival, plus megalo.
pub fn engine_type_name(t: u32) -> &'static str {
    match t {
        0 => "none",
        1 => "Forge",
        2 => "Megalo (a scripted multiplayer gametype)",
        3 => "Campaign",
        4 => "Firefight",
        _ => "?",
    }
}

/// The human-readable sidecar that ships next to the raw struct. Deliberately does NOT include
/// the recording machine's Windows account or computer name (`flmh+0x8C`).
pub fn game_variant_notes(f: &Film) -> String {
    let gv = &f.header.game_variant;
    let mut s = String::new();
    s.push_str("Halo: Reach game variant, extracted from a saved film by Halo Map Studio\n");
    s.push_str("=======================================================================\n\n");
    s.push_str(&format!("title          {:?}\n", gv.title));
    s.push_str(&format!("description    {:?}\n", gv.description));
    s.push_str(&format!("author         {:?}\n", gv.created_by));
    s.push_str(&format!("editor         {:?}\n", gv.modified_by));
    s.push_str(&format!(
        "engine type    {} ({})\n",
        f.header.game_engine_type,
        engine_type_name(f.header.game_engine_type)
    ));
    s.push_str(&format!("lineage uid    {:#018x}  (NOT an identity: MCC keeps the parent's uid on every derived save)\n", gv.uid));
    s.push_str(&format!("on-disk size   {} bytes, as the .bin this variant was saved from\n", gv.file_size));
    s.push_str(&format!("from film      {:?} recorded {}\n", f.content.title, mvar::unix_date(f.content.created_unix)));
    s.push_str("\n");
    s.push_str(&format!(
        "The companion .reach_gv file is the RAW {} ({:#x}) byte in-memory `c_game_variant`\n\
         struct, copied verbatim out of the film's `flmh` chunk (payload offset {:#x}).\n\n",
        film::game_options::GAME_VARIANT_LEN,
        film::game_options::GAME_VARIANT_LEN,
        film::flmh::GAME_OPTIONS + film::game_options::GAME_VARIANT
    ));
    s.push_str(
        "IT IS NOT A LOADABLE .bin GAMETYPE. A Reach gametype on disk is a BLF file whose `mpvr`\n\
         chunk (version 54) holds a bit-packed encoding that compresses this 64516-byte struct to\n\
         about 20.5 KB. Writing that encoder means reverse-engineering the whole Megalo variant\n\
         bit layout, which has not been done; Halo Map Studio will not fabricate a file it cannot\n\
         verify. The struct is written out so that a future encoder can produce the real thing\n\
         from it, and so the settings can be inspected now.\n\n\
         Known struct offsets (see re-notes/reach_saved_film_layout.md):\n\
         \x20 +0x0000  u32  game engine type\n\
         \x20 +0x000C  0x2B0  s_content_item_metadata (content type 6)\n\
         \x20 the Megalo script state, trait sets and localized name/description tables follow\n",
    );
    s
}

/// A one-screen report for the script verb / CLI.
pub fn extraction_report(f: &Film) -> String {
    let mv = &f.header.map_variant;
    let mut s = String::new();
    if mv.is_empty() {
        s.push_str("map variant   none -- this film was recorded on a stock map\n");
    } else {
        let t = find_map_variant_template(mv.map_id);
        s.push_str(&format!(
            "map variant   {:?} by {:?}, {} objects, map id {}\n",
            mv.metadata.title,
            mv.metadata.created_by,
            mv.present_objects().count(),
            mv.map_id
        ));
        match t {
            Some(p) => s.push_str(&format!("  canvas      {}\n", p.display())),
            None => s.push_str(&format!("  canvas      NONE FOUND for map id {} -- cannot write a .mvar\n", mv.map_id)),
        }
    }
    s.push_str(&format!(
        "game variant  {:?} by {:?}, engine {} ({}) -- raw struct only, see the notes file\n",
        f.header.game_variant.title,
        f.header.game_variant.created_by,
        f.header.game_engine_type,
        engine_type_name(f.header.game_engine_type)
    ));
    s
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_films() -> Vec<std::path::PathBuf> {
        // ONE implementation, in `film.rs`. There were six copies of this walk and they
        // disagreed about which films exist -- and all of them missed the entire Linux install,
        // because a Proton prefix is seven levels down and uses a lowercase `users`.
        crate::film::spec_corpus()
    }

    #[test]
    fn safe_stem_is_filename_safe() {
        assert_eq!(safe_stem("UNSC/Civilian Social: 2.6", "x"), "UNSC_Civilian Social_ 2.6");
        assert_eq!(safe_stem("  ...  ", "fallback"), "fallback");
        assert_eq!(safe_stem("", "fallback"), "fallback");
        assert_eq!(safe_stem("A//B", "x"), "A_B");
        for s in [safe_stem("a/b\\c:d*e?f\"g<h>i|j", "x")] {
            assert!(!s.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']), "{s}");
        }
    }

    #[test]
    fn engine_names_cover_the_reach_set() {
        // #gamevariant  Reach numbers the KIND of variant here, not the gametype: 1 Forge,
        // 2 megalo, 3 campaign, 4 Firefight. Measured 65/65 against `c_game_variant+0x00`.
        assert_eq!(engine_type_name(1), "Forge");
        assert_eq!(engine_type_name(2), "Megalo (a scripted multiplayer gametype)");
        assert_eq!(engine_type_name(4), "Firefight");
        assert_eq!(engine_type_name(10), "?");
        assert_eq!(engine_type_name(99), "?");
    }

    /// THE ROUND TRIP. For every real film with a Forge variant: extract it to a `.mvar`, read
    /// that file back with `mvar::parse_variant`, and require the objects to match the film's
    /// embedded table field for field. Positions are compared to the `.mvar`'s own quantisation
    /// step, which is the only lossy part of the conversion.
    #[test]
    fn extracted_mvar_round_trips() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_export: no real films on this machine -- round trip not checked");
            return;
        }
        let dir = std::env::temp_dir().join("hms_film_export_test");
        let _ = std::fs::create_dir_all(&dir);
        let (mut done, mut stock, mut no_template) = (0usize, 0usize, 0usize);
        for fp in &films {
            let Ok(f) = film::open(fp) else { continue };
            let mv = &f.header.map_variant;
            if mv.is_empty() {
                // the stock-map case must be a clean refusal, never a broken file
                let dst = dir.join("stock.mvar");
                assert!(matches!(export_map_variant(&f, &dst), Err(ExportError::StockMap)));
                assert!(!dst.exists(), "a stock-map film must not write a file");
                stock += 1;
                continue;
            }
            let Some(template) = find_map_variant_template(mv.map_id) else {
                no_template += 1;
                continue;
            };
            let bytes = match export_map_variant_with_template(&f, &template) {
                Ok(b) => b,
                Err(e) => panic!("{}: {e}", fp.display()),
            };
            let dst = dir.join(format!("{}.mvar", safe_stem(&mv.metadata.title, "extracted")));
            std::fs::write(&dst, &bytes).unwrap();
            let v = mvar::parse_variant(&dst).unwrap_or_else(|| panic!("{}: re-read failed", dst.display()));
            let d = fp.file_name().unwrap_or_default().to_string_lossy().to_string();

            // header
            assert_eq!(v.map_id, mv.map_id, "{d}: map id");
            assert_eq!(v.globals.palette_crc, mv.palette_crc, "{d}: palette crc");
            assert_eq!(v.globals.num_quotas, mv.num_quotas, "{d}: quota count");
            assert_eq!(v.globals.budget_max, mv.budget_max, "{d}: budget max");
            assert_eq!(v.title, mv.metadata.title, "{d}: title");
            assert_eq!(v.description, mv.metadata.description, "{d}: description");
            assert_eq!(v.author, mv.metadata.created_by, "{d}: author");
            assert_eq!(v.editor, mv.metadata.modified_by, "{d}: editor");
            assert_eq!(v.labels, mv.labels, "{d}: forge labels");
            for k in 0..6 {
                assert!((v.globals.bounds[k] - mv.world_bounds[k]).abs() < 0.01, "{d}: bound {k}");
            }

            // objects, field for field
            let want: Vec<&film::VariantObject> = mv.present_objects().collect();
            assert_eq!(v.objects.len(), want.len(), "{d}: object count");
            let ext = [
                mv.world_bounds[1] - mv.world_bounds[0],
                mv.world_bounds[3] - mv.world_bounds[2],
                mv.world_bounds[5] - mv.world_bounds[4],
            ];
            let tol = ext.iter().cloned().fold(0.0f32, f32::max) / (1 << 14) as f32 + 0.05;
            for (i, (got, e)) in v.objects.iter().zip(want.iter()).enumerate() {
                let w = format!("{d} object {i}");
                assert_eq!(got.folder, e.quota_index, "{w}: quota index");
                assert_eq!(got.item, e.item, "{w}: item");
                assert_eq!(got.team, e.team, "{w}: team");
                assert_eq!(got.color, e.color as i32, "{w}: color");
                assert_eq!(got.label_idx, e.label_idx, "{w}: label");
                assert_eq!(got.placement, e.placement, "{w}: placement");
                assert_eq!(got.spawn_seq, e.spawn_seq as i32, "{w}: spawn sequence");
                assert_eq!(got.respawn, e.respawn, "{w}: respawn");
                assert_eq!(got.cached_type, e.cached_type, "{w}: cached type");
                assert_eq!(got.boundary_shape, e.boundary_shape, "{w}: boundary shape");
                assert_eq!(got.spawn_rel, e.parent, "{w}: parent");
                assert_eq!(got.flags, (e.flags & 0x3) as u8, "{w}: flags");
                match e.cached_type {
                    1 => assert_eq!(got.weapon_clips, e.clips_or_tele_channel, "{w}: weapon clips"),
                    12..=14 => {
                        assert_eq!(got.tele_channel, e.clips_or_tele_channel, "{w}: teleporter channel");
                        assert_eq!(got.tele_passability, e.tele_passability, "{w}: teleporter passability");
                    }
                    _ => {}
                }
                for (k, &slot) in film::boundary_float_slots(e.boundary_shape).iter().enumerate() {
                    assert_eq!(got.boundary[k], film::quantize_boundary(e.boundary[slot]), "{w}: boundary {k}");
                }
                for k in 0..3 {
                    assert!(
                        (got.pos[k] - e.position[k]).abs() <= tol,
                        "{w}: position axis {k}: {} vs {} (tol {tol})", got.pos[k], e.position[k]
                    );
                }
                // orientation survives the .mvar's 20-bit-up + 14-bit-angle quantisation
                let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                assert!(dot(got.up, e.up) > 0.999, "{w}: up drifted ({:?} vs {:?})", got.up, e.up);
                assert!(dot(got.fwd, e.forward) > 0.999, "{w}: forward drifted ({:?} vs {:?})", got.fwd, e.forward);
            }
            eprintln!(
                "film_export: {} -> {} objects, canvas {}",
                d, v.objects.len(), template.file_name().unwrap_or_default().to_string_lossy()
            );
            done += 1;
        }
        eprintln!("film_export: {done} variant(s) extracted and round-tripped, {stock} stock-map film(s) refused, {no_template} without a canvas");
        assert!(done > 0 || no_template > 0, "found films with variants but extracted none");
    }

    /// Where a film pairs with the REAL `.mvar` it was recorded on, the file we write must agree
    /// with that file object for object too -- not merely with our own reading of the film.
    #[test]
    fn extracted_mvar_matches_the_original_mvar() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_export: no real films -- original-file comparison not checked");
            return;
        }
        let mut by_key: std::collections::HashMap<(String, u32), Vec<PathBuf>> = Default::default();
        let mut dirs: Vec<PathBuf> = crate::mapcat::variant_dirs();
        for f in &films {
            if let Some(par) = f.parent().and_then(|p| p.parent()) {
                dirs.push(par.join("Map"));
            }
        }
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if !p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) { continue }
                let Some(v) = mvar::parse_variant(&p) else { continue };
                by_key.entry((v.title.clone(), v.globals.map_id)).or_default().push(p);
            }
        }
        let dir = std::env::temp_dir().join("hms_film_export_test");
        let _ = std::fs::create_dir_all(&dir);
        let mut pairs = 0usize;
        for fp in &films {
            let Ok(f) = film::open(fp) else { continue };
            let mv = &f.header.map_variant;
            if mv.is_empty() { continue }
            let n = mv.present_objects().count();
            let Some(cands) = by_key.get(&(mv.metadata.title.clone(), mv.map_id)) else { continue };
            // Deterministic, and keyed on the variant's real IDENTITY. `read_dir` returns
            // entries in filesystem order, so this used to pick whichever candidate was
            // enumerated first -- and the same variant is saved repeatedly as it is edited, with
            // two saves agreeing on title, map id, object count and spent budget while differing
            // in their Forge LABELS and in the label index of individual objects. Comparing a
            // film against someone else's later edit is not a test of the exporter.
            let mut cands: Vec<&PathBuf> = cands.iter().collect();
            cands.sort();
            let Some((orig_path, orig)) = cands.iter().find_map(|p| {
                let v = mvar::parse_variant(p)?;
                (v.objects.len() == n
                    && v.globals.budget_spent == mv.budget_spent
                    && v.labels == mv.labels)
                    .then_some(((*p).clone(), v))
            }) else { continue };
            let Some(template) = find_map_variant_template(mv.map_id) else { continue };
            let bytes = export_map_variant_with_template(&f, &template).expect("export");
            let dst = dir.join("cmp.mvar");
            std::fs::write(&dst, &bytes).unwrap();
            let got = mvar::parse_variant(&dst).expect("re-read");
            let d = format!(
                "{} vs {}",
                fp.file_name().unwrap_or_default().to_string_lossy(),
                orig_path.file_name().unwrap_or_default().to_string_lossy()
            );
            assert_eq!(got.objects.len(), orig.objects.len(), "{d}: object count");
            for (i, (a, b)) in got.objects.iter().zip(orig.objects.iter()).enumerate() {
                let w = format!("{d} object {i}");
                assert_eq!(a.folder, b.folder, "{w}: quota index");
                assert_eq!(a.item, b.item, "{w}: item");
                assert_eq!(a.team, b.team, "{w}: team");
                assert_eq!(a.color, b.color, "{w}: color");
                assert_eq!(a.label_idx, b.label_idx, "{w}: label");
                assert_eq!(a.placement, b.placement, "{w}: placement");
                assert_eq!(a.spawn_seq, b.spawn_seq, "{w}: spawn sequence");
                assert_eq!(a.respawn, b.respawn, "{w}: respawn");
                assert_eq!(a.cached_type, b.cached_type, "{w}: cached type");
                assert_eq!(a.boundary_shape, b.boundary_shape, "{w}: boundary shape");
                assert_eq!(a.boundary, b.boundary, "{w}: boundary dims");
                assert_eq!(a.weapon_clips, b.weapon_clips, "{w}: weapon clips");
                assert_eq!(a.tele_channel, b.tele_channel, "{w}: teleporter channel");
                assert_eq!(a.tele_passability, b.tele_passability, "{w}: teleporter passability");
                for k in 0..3 {
                    assert!((a.pos[k] - b.pos[k]).abs() < 0.05, "{w}: position axis {k}: {} vs {}", a.pos[k], b.pos[k]);
                }
            }
            eprintln!("film_export: extracted == original on {} objects ({d})", got.objects.len());
            pairs += 1;
        }
        if pairs == 0 {
            eprintln!("film_export: no film/.mvar pair -- original-file comparison not checked");
        }
    }

    /// The game-variant export must produce the raw struct plus notes, and must never leak the
    /// recording machine's Windows account / computer name.
    #[test]
    fn game_variant_export_is_raw_and_private() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_export: no real films -- game variant export not checked");
            return;
        }
        let dir = std::env::temp_dir().join("hms_film_export_test");
        let _ = std::fs::create_dir_all(&dir);
        let mut n = 0usize;
        for fp in films.iter().take(6) {
            let Ok(f) = film::open(fp) else { continue };
            let raw = game_variant_bytes(&f).expect("game variant block");
            assert_eq!(raw.len(), film::game_options::GAME_VARIANT_LEN);
            // its first dword is the engine type, and the metadata 12 bytes in is content type 6
            assert_eq!(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]), f.header.game_engine_type);
            assert_eq!(u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]), 6);
            let out = export_game_variant(&f, &dir.join("gv")).expect("export");
            assert_eq!(out.bytes, film::game_options::GAME_VARIANT_LEN);
            assert_eq!(std::fs::metadata(&out.struct_path).unwrap().len() as usize, out.bytes);
            let notes = std::fs::read_to_string(&out.notes_path).unwrap();
            assert!(notes.contains("IT IS NOT A LOADABLE .bin GAMETYPE"));
            assert!(notes.contains(&f.header.game_variant.title) || f.header.game_variant.title.is_empty());
            // PRIVACY: the session id must not appear in anything we write
            let sid = &f.header.session_id;
            if !sid.is_empty() && sid != "UNINIT_SESSIONID" {
                assert!(!notes.contains(sid.as_str()), "the notes leaked the session id");
                for piece in sid.split('@').skip(2) {
                    if piece.len() >= 3 {
                        assert!(!notes.contains(piece), "the notes leaked {piece:?} from the session id");
                    }
                }
            }
            // and the extraction report must be printable without it too
            let rep = extraction_report(&f);
            assert!(rep.contains("game variant"));
            if !sid.is_empty() && sid != "UNINIT_SESSIONID" {
                assert!(!rep.contains(sid.as_str()));
            }
            n += 1;
        }
        eprintln!("film_export: game variant export checked on {n} film(s)");
    }
}

/// #film  **THE DEFERRED-OPEN DECISION**, as a pure function, so the bug that made opening a film
/// on a cold app do nothing cannot come back.
///
/// `App::film_load_scene` must load the film's own base map before it can draw the film's Forge
/// scene. `load_map()` STREAMS that load over many frames, so the check immediately after it sees
/// the OLD `loaded_map_id` -- and the original code treated that as a failure and returned,
/// abandoning the variant queue, the `film_opening_own_scene` flag and the decode. The symptom was
/// that opening a clip from a cold app loaded a map and did nothing else, while opening the same
/// clip a second time worked.
///
/// The decision has three outcomes and the defect was that two of them were collapsed into one.
/// This names them, so a test can hold each apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneOpen {
    /// The film's map is already loaded: draw the scene and start the decode now.
    Proceed,
    /// A load of the film's map is under way: finish when it lands, remembering this map id.
    WaitFor(u32),
    /// The map is not installed or was not selected: there is nothing to wait for.
    Impossible,
}

/// `loaded` = `App::loaded_map_id`; `selected` = the map id of the currently selected candidate,
/// i.e. what a load that has just been started is going to produce.
pub fn scene_open_decision(film_map: u32, loaded: Option<u32>, selected: Option<u32>) -> SceneOpen {
    if loaded == Some(film_map) {
        return SceneOpen::Proceed;
    }
    if selected == Some(film_map) {
        return SceneOpen::WaitFor(film_map);
    }
    SceneOpen::Impossible
}

/// Should a completed map load finish a film's deferred open?
///
/// Written out because the tempting form -- `pending.take() == loaded` -- is TRUE when both are
/// `None`, which fires on every load of a map with no id while any film is open.
pub fn deferred_open_fires(pending: Option<u32>, loaded: Option<u32>) -> bool {
    match pending {
        Some(want) => loaded == Some(want),
        None => false,
    }
}

#[cfg(test)]
mod deferred_film_open_gates {
    use super::*;

    /// The three outcomes, including the one the original code could not express.
    #[test]
    fn a_streaming_map_load_defers_the_open_instead_of_abandoning_it() {
        // The bug, exactly: the film's map is not loaded YET, but a load of it is under way.
        assert_eq!(
            scene_open_decision(1234, None, Some(1234)),
            SceneOpen::WaitFor(1234),
            "a film whose map is still streaming in must be finished later, not dropped -- this \
             is the cold-app case that made opening a clip do nothing"
        );
        // The wrong map is loaded and the right one is coming.
        assert_eq!(scene_open_decision(1234, Some(99), Some(1234)), SceneOpen::WaitFor(1234));
        // Already there: go.
        assert_eq!(scene_open_decision(1234, Some(1234), Some(1234)), SceneOpen::Proceed);
        assert_eq!(scene_open_decision(1234, Some(1234), None), SceneOpen::Proceed);
        // Genuinely not available: nothing to wait for, and the status line owns the explanation.
        assert_eq!(scene_open_decision(1234, None, None), SceneOpen::Impossible);
        assert_eq!(scene_open_decision(1234, Some(99), Some(99)), SceneOpen::Impossible);
    }

    /// The re-entry must not fire on `None == None`, and must not fire for another film's map.
    #[test]
    fn the_deferred_open_fires_only_for_the_map_it_is_waiting_on() {
        assert!(deferred_open_fires(Some(1234), Some(1234)), "the awaited map landed");
        assert!(!deferred_open_fires(Some(1234), Some(99)), "a DIFFERENT map landed");
        assert!(!deferred_open_fires(Some(1234), None), "no map id at all");
        assert!(
            !deferred_open_fires(None, None),
            "nothing is pending -- the `take() == loaded_map_id` form returns TRUE here and would \
             re-enter the film open on every load of a map that has no id"
        );
        assert!(!deferred_open_fires(None, Some(1234)), "nothing is pending");
    }
}

/// #film  Is this variant import THE FILM'S OWN Forge scene, and therefore exempt from closing
/// the clip?
///
/// `own` is `App::film_own_scene` — the path `film_load_scene` queued. `incoming` is the path
/// `import_mvar` was handed. Exempt only when they are the same file.
///
/// **This is a function because the bool it replaced was a race.** The old test was
/// `std::mem::take(&mut film_opening_own_scene)`, i.e. "the NEXT import is the film's own" — which
/// holds only while exactly one import is in flight. Two routinely are: `film_load_scene` queues
/// the film's scene through `pending_open_variant`, and the SAME map-load-completion block then
/// takes `autoload_variant` (the last-opened variant, restored at startup) and imports it at once.
/// The autoload spent the film's exemption, so the film's own scene arrived unexempted and
/// `film_close` ran — nulling `self.film` and dropping the in-flight decode. A clip that opened,
/// drew a scene and never decoded, with no error to show for it.
///
/// Matching on identity cannot be spent by somebody else's import, and a genuinely foreign
/// variant still closes the clip, which is the behaviour that exists on purpose.
pub fn import_is_the_films_own_scene(own: Option<&std::path::Path>, incoming: &std::path::Path) -> bool {
    own == Some(incoming)
}

#[cfg(test)]
mod film_scene_exemption_gates {
    use super::*;
    use std::path::{Path, PathBuf};

    /// The exemption belongs to ONE import and cannot be spent by another.
    #[test]
    fn another_import_in_flight_cannot_spend_the_films_exemption() {
        let film_scene = PathBuf::from("/tmp/hms/film_scene.mvar");
        let autoload = Path::new("/home/someone/last_opened.mvar");

        // The autoload lands FIRST -- this is the real sequence that broke it.
        assert!(
            !import_is_the_films_own_scene(Some(&film_scene), autoload),
            "the last-opened variant must NOT be treated as the film's own scene; as a bool this \
             returned true, spent the exemption, and left the film's own import to close the clip"
        );
        // ...and the film's own scene is still exempt afterwards, which is the property the bool
        // could not provide.
        assert!(
            import_is_the_films_own_scene(Some(&film_scene), &film_scene),
            "the film's own scene must still be exempt after another import went past"
        );
    }

    /// A variant the user really did open still closes the clip. The fix must not become a hole.
    #[test]
    fn a_foreign_variant_still_closes_the_clip() {
        let film_scene = PathBuf::from("/tmp/hms/film_scene.mvar");
        for other in ["/maps/other.mvar", "/tmp/hms/film_scene.mvar.bak", "/tmp/hms/Film_Scene.mvar"] {
            assert!(
                !import_is_the_films_own_scene(Some(&film_scene), Path::new(other)),
                "{other} is not the film's own scene and must close the clip -- a recording's \
                 bodies are a statement about the scene it was recorded in"
            );
        }
        // With no film open there is nothing to exempt.
        assert!(!import_is_the_films_own_scene(None, Path::new("/maps/other.mvar")));
        assert!(!import_is_the_films_own_scene(None, &film_scene));
    }
}
