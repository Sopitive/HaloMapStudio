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
//! * **Game variant → a real, loadable `.bin`.** A Reach gametype on disk is a BLF container
//!   (`_blf`, `chdr`, `mpvr`, `_eof`) whose `mpvr` chunk (version 54) is the bit-packed encoding
//!   of the 64,516-byte in-memory `c_game_variant`, exactly the struct a film embeds. The codec in
//!   `gamevariant.rs` was written from the engine's own serialiser (`haloreach.dll`
//!   `GameVariantContext_{Decode,Encode}FromBitstream` and friends) and is bit-identical to it on
//!   every real gametype tried, so the embedded struct is turned into the same file the game
//!   would have saved. See `gamevariant.rs` for the evidence and its limits.
//!
//!   Matching the film's gametype to a `.bin` already on disk was tried and REJECTED: MCC keeps
//!   the lineage uid across derived saves, so one uid (`0x4a6994bf5d589b43`) covers gametypes
//!   titled "Testing", "Castle Wars Xtr2", "SCARAB MODE ACOS" and more. Matching on it would hand
//!   the user the wrong gametype. Measured on 39 real films against 720 on-disk gametypes. The
//!   file is therefore REBUILT from the film's own struct, never fetched from disk.

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
    /// Canvases exist for this map id but NONE is the film's map build. `tried` is one plain-words
    /// line per canvas (file name, its palette CRC and quota count, or why it could not be read),
    /// so a user's report says which files were in the way.
    NoMatchingCanvas { film_crc: u32, film_quotas: u32, tried: Vec<String> },
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
            ExportError::NoMatchingCanvas { film_crc, film_quotas, tried } => write!(
                f,
                "none of the {} saved variant(s) for this map is the same map build as the film \
                 (the film was recorded with scenario palette CRC {film_crc:#010x} and {film_quotas} \
                 object quotas). Tried: {}",
                tried.len(),
                tried.join("; ")
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
    /// The loadable `.bin` (None only if the encoder refused this variant, see `why_no_bin`).
    pub bin_path: Option<PathBuf>,
    /// The raw `c_game_variant` struct, written ONLY as a fallback when no `.bin` could be built.
    pub struct_path: Option<PathBuf>,
    /// The human-readable sidecar.
    pub notes_path: PathBuf,
    /// Size of the `.bin` (or of the raw struct in the fallback case).
    pub bytes: usize,
    /// Bytes of bit-packed variant inside the `.bin`.
    pub bitstream_bytes: usize,
    pub title: String,
    pub engine_type: u32,
    pub why_no_bin: Option<String>,
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
    map_variant_templates(map_id).into_iter().next()
}

/// Every candidate canvas for `map_id`, in the preference order of `find_map_variant_template`.
/// A folder can hold variants of the same map id saved on a different build of the map (another
/// title update, a mod), whose scenario palette CRC or quota count differs from the film's, so
/// the exporter walks this list instead of giving up on the first one.
pub fn map_variant_templates(map_id: u32) -> Vec<PathBuf> {
    let mut cands: Vec<(u64, PathBuf)> = known_map_variants()
        .into_iter()
        .filter(|(_, id)| *id == map_id)
        .filter_map(|(p, _)| std::fs::metadata(&p).ok().map(|m| (m.len(), p)))
        .collect();
    cands.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    cands.into_iter().map(|(_, p)| p).collect()
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
/// #film-scene  The film's Forge scene as an IN-MEMORY, READ-ONLY variant, built from the
/// `c_map_variant` the film embeds WHOLE. No `.mvar` canvas is read or written, so a missing or
/// other-build canvas can never stop a clip from showing its scene (that was the "palette CRC
/// does not match" refusal of the file route, which needs a same-build donor file by design).
///
/// What the donor used to supply is filled from the film's own header: map id, palette CRC,
/// quota count and the min/max quota table (`placed` recomputed from the objects, as the encoder
/// does), world bounds, budgets, labels, titles and authors, content ids. What the film does not
/// carry is left empty rather than invented: the MCC map GUID (`mcc_map_id: None`, the base map is
/// routed by the film's own map id) and the BLF wrapper facts (chunk length / hash).
///
/// There is deliberately NO path and no write-back: the App never sets `current_variant_path`
/// for this, so File > Save has nothing to overwrite. One implementation for the GUI
/// (`film_load_scene`) and the script host (`film scene`).
pub fn variant_from_film(f: &Film) -> Result<mvar::Variant, String> {
    let mv = &f.header.map_variant;
    if mv.is_empty() {
        return Err("this film was recorded on a stock map — there is no Forge scene to draw".into());
    }
    let objects: Vec<mvar::PlacedObject> = mv.present_objects().map(|o| o.to_placed_object()).collect();
    let mut placed = [0u32; 256];
    for o in &objects {
        if o.folder != 0xFFFF && o.item != 0xFF && (o.folder as usize) < 256 {
            placed[o.folder as usize] += 1;
        }
    }
    let quotas: Vec<(u8, u8, u8)> = mv
        .quotas
        .iter()
        .take(mv.num_quotas as usize)
        .enumerate()
        .map(|(i, q)| {
            let n = placed.get(i).copied().unwrap_or(0).min(255) as u8;
            (q[0], q[1].max(n), n)
        })
        .collect();
    let m = &mv.metadata;
    let globals = mvar::VariantGlobals {
        game: "Reach",
        content_type: 5,
        uid: m.uid,
        parent_uid: m.parent_uid,
        root_uid: m.root_uid,
        game_id: m.game_id,
        activity: m.activity as i8,
        game_mode: m.game_mode,
        engine: m.engine,
        header_map_id: m.map_id,
        category: m.category as u8 as i8,
        created: (m.created_unix, m.created_xuid),
        modified: (m.modified_unix, m.modified_xuid),
        created_online: m.created_online != 0,
        modified_online: m.modified_online != 0,
        version: mv.version as u8,
        checksum: mv.checksum,
        palette_crc: mv.palette_crc,
        num_quotas: mv.num_quotas,
        map_id: mv.map_id,
        bounds: mv.world_bounds,
        budget_max: mv.budget_max,
        budget_spent: mv.budget_spent,
        quotas,
        ..Default::default()
    };
    Ok(mvar::Variant {
        map_id: mv.map_id,
        mcc_map_id: None,
        num_quotas: mv.num_quotas as u32,
        objects,
        labels: mv.labels.clone(),
        title: m.title.clone(),
        description: m.description.clone(),
        author: m.created_by.clone(),
        editor: m.modified_by.clone(),
        globals,
    })
}

pub fn export_map_variant(f: &Film, dst: &Path) -> Result<MapVariantExport, ExportError> {
    let mv = &f.header.map_variant;
    if mv.is_empty() {
        return Err(ExportError::StockMap);
    }
    let (template, bytes) = export_with_candidates(f, &map_variant_templates(mv.map_id))?;
    std::fs::write(dst, &bytes).map_err(|e| ExportError::Io(format!("{}: {e}", dst.display())))?;
    Ok(MapVariantExport {
        path: dst.to_path_buf(),
        objects: mv.present_objects().count(),
        template,
        title: mv.metadata.title.clone(),
        map_id: mv.map_id,
    })
}

/// Walk `candidates` in order and export from the first canvas that really is the film's map
/// build. A mismatch (palette CRC, quota count, map id, unreadable file) only means THAT file is
/// from another build of the map. With no candidates: `NoTemplate`; with candidates and no match:
/// the single error when there was one candidate, else `NoMatchingCanvas` listing every one.
pub fn export_with_candidates(f: &Film, candidates: &[PathBuf]) -> Result<(PathBuf, Vec<u8>), ExportError> {
    let mv = &f.header.map_variant;
    if mv.is_empty() {
        return Err(ExportError::StockMap);
    }
    let mut errs: Vec<ExportError> = Vec::new();
    for template in candidates {
        match export_map_variant_with_template(f, template) {
            Ok(bytes) => return Ok((template.clone(), bytes)),
            Err(e) => errs.push(e),
        }
    }
    if errs.is_empty() {
        return Err(ExportError::NoTemplate(mv.map_id));
    }
    if errs.iter().all(|e| matches!(e, ExportError::TemplateMismatch { .. })) {
        let tried = candidates
            .iter()
            .map(|c| {
                let name = c.file_name().unwrap_or_default().to_string_lossy().into_owned();
                match mvar::parse_variant(c) {
                    Some(v) => format!("{name} (palette CRC {:#010x}, {} quotas)", v.globals.palette_crc, v.globals.num_quotas),
                    None => format!("{name} (unreadable)"),
                }
            })
            .collect();
        return Err(ExportError::NoMatchingCanvas { film_crc: mv.palette_crc, film_quotas: mv.num_quotas as u32, tried });
    }
    Err(errs.into_iter().find(|e| !matches!(e, ExportError::TemplateMismatch { .. })).unwrap())
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

/// Build the loadable game variant `.bin` from the film's embedded `c_game_variant`, plus a
/// readable sidecar. `dst` is the `.bin` path to write; the notes go beside it as `<name>.txt`.
///
/// If the encoder cannot handle this variant (a Firefight survival variant before its codec lands,
/// or a variant whose strings overflow a `.bin`), the raw struct is written instead as
/// `<name>.reach_gv` and `why_no_bin` says why, so the user still gets the data.
pub fn export_game_variant(f: &Film, dst: &Path) -> Result<GameVariantExport, ExportError> {
    let raw = game_variant_bytes(f)
        .ok_or_else(|| ExportError::Encode("the film's game variant block is truncated".into()))?;
    let bin_path = with_ext(dst, "bin");
    let notes_path = with_ext(dst, "txt");
    let title = f.header.game_variant.title.clone();
    let engine_type = f.header.game_engine_type;
    match crate::gamevariant::build_bin(raw, crate::gamevariant::CHDR_TAG_UNSET) {
        Ok(bin) => {
            std::fs::write(&bin_path, &bin).map_err(|e| ExportError::Io(format!("{}: {e}", bin_path.display())))?;
            let (_, info) = crate::gamevariant::parse_bin(&bin)
                .map_err(|e| ExportError::Encode(format!("internal error: the file just built does not read back: {e}")))?;
            std::fs::write(&notes_path, game_variant_notes(f, Some(&info)))
                .map_err(|e| ExportError::Io(format!("{}: {e}", notes_path.display())))?;
            Ok(GameVariantExport {
                bin_path: Some(bin_path),
                struct_path: None,
                notes_path,
                bytes: bin.len(),
                bitstream_bytes: info.declared_len,
                title,
                engine_type,
                why_no_bin: None,
            })
        }
        Err(why) => {
            let struct_path = with_ext(dst, "reach_gv");
            std::fs::write(&struct_path, raw).map_err(|e| ExportError::Io(format!("{}: {e}", struct_path.display())))?;
            let mut notes = game_variant_notes(f, None);
            notes.push_str(&format!(
                "\nNO .bin WAS WRITTEN: {why}\nThe raw {}-byte in-memory struct was written instead as {}.\n",
                raw.len(),
                struct_path.file_name().unwrap_or_default().to_string_lossy()
            ));
            std::fs::write(&notes_path, notes).map_err(|e| ExportError::Io(format!("{}: {e}", notes_path.display())))?;
            Ok(GameVariantExport {
                bin_path: None,
                struct_path: Some(struct_path),
                notes_path,
                bytes: raw.len(),
                bitstream_bytes: 0,
                title,
                engine_type,
                why_no_bin: Some(why),
            })
        }
    }
}

/// `dst` with its extension replaced by `ext`. A stem like `UNSC Civilian 2.6` has a dot in it, so
/// the extension is only replaced when `dst` already ends in a known one.
fn with_ext(dst: &Path, ext: &str) -> PathBuf {
    let name = dst.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let stem = ["bin", "txt", "reach_gv"]
        .iter()
        .find_map(|e| name.strip_suffix(&format!(".{e}")))
        .unwrap_or(&name)
        .to_string();
    dst.with_file_name(format!("{stem}.{ext}"))
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

/// The human-readable sidecar that ships next to the `.bin`. Deliberately does NOT include
/// the recording machine's Windows account or computer name (`flmh+0x8C`).
pub fn game_variant_notes(f: &Film, info: Option<&crate::gamevariant::BinInfo>) -> String {
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
    s.push_str(&format!("from film      {:?} recorded {}\n", f.content.title, mvar::unix_date(f.content.created_unix)));
    s.push('\n');
    if let Some(i) = info {
        s.push_str(&format!(
            "The companion .bin is a complete Reach game variant file: BLF container (_blf, chdr,\n\
             mpvr v54, _eof) around {} bytes of bit-packed variant. Halo Map Studio rebuilt it from\n\
             the `c_game_variant` struct the film embeds, using an encoder that reproduces the game's\n\
             own serialiser bit for bit on every real gametype it was checked against. It was not\n\
             copied from a file on disk. Its SHA-1 chunk hash is correct and it reads back through the\n\
             same decoder with identical contents.\n\n\
             To use it, copy it into the game_variants folder next to your saved gametypes. It has\n\
             not been loaded in MCC by Halo Map Studio itself; if the game refuses it, please report\n\
             the file.\n",
            i.declared_len
        ));
    }
    s
}

/// A one-screen report for the script verb / CLI.
pub fn extraction_report(f: &Film) -> String {
    let mv = &f.header.map_variant;
    let mut s = String::new();
    if mv.is_empty() {
        s.push_str("map variant   none -- this film was recorded on a stock map\n");
    } else {
        let t = export_with_candidates(f, &map_variant_templates(mv.map_id));
        s.push_str(&format!(
            "map variant   {:?} by {:?}, {} objects, map id {}\n",
            mv.metadata.title,
            mv.metadata.created_by,
            mv.present_objects().count(),
            mv.map_id
        ));
        match t {
            Ok((p, _)) => s.push_str(&format!("  canvas      {}\n", p.display())),
            Err(ExportError::NoTemplate(_)) => {
                s.push_str(&format!("  canvas      NONE FOUND for map id {} -- cannot write a .mvar\n", mv.map_id))
            }
            Err(e) => s.push_str(&format!("  canvas      NONE USABLE -- {e}\n")),
        }
    }
    let how = match game_variant_bytes(f).map(|raw| crate::gamevariant::build_bin(raw, crate::gamevariant::CHDR_TAG_UNSET)) {
        Some(Ok(bin)) => format!("will be written as a loadable .bin ({} bytes)", bin.len()),
        Some(Err(why)) => format!("CANNOT be written as a .bin ({why}); the raw struct is written instead"),
        None => "the film's game variant block is truncated".to_string(),
    };
    s.push_str(&format!(
        "game variant  {:?} by {:?}, engine {} ({}) -- {how}\n",
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
            if find_map_variant_template(mv.map_id).is_none() {
                no_template += 1;
                continue;
            }
            let (template, bytes) = match export_with_candidates(&f, &map_variant_templates(mv.map_id)) {
                Ok(x) => x,
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
            if find_map_variant_template(mv.map_id).is_none() { continue }
            let (_, bytes) = export_with_candidates(&f, &map_variant_templates(mv.map_id)).expect("export");
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

    /// The game-variant export must produce a real `.bin` (that reads back through the decoder to
    /// the same struct) plus notes, and must never leak the recording machine's Windows account /
    /// computer name.
    #[test]
    fn game_variant_export_is_a_bin_and_private() {
        let films = sample_films();
        if films.is_empty() {
            eprintln!("film_export: no real films -- game variant export not checked");
            return;
        }
        let dir = std::env::temp_dir().join("hms_film_export_test");
        let _ = std::fs::create_dir_all(&dir);
        let (mut n, mut bins, mut fallbacks) = (0usize, 0usize, 0usize);
        for fp in films.iter().take(6) {
            let Ok(f) = film::open(fp) else { continue };
            let raw = game_variant_bytes(&f).expect("game variant block");
            assert_eq!(raw.len(), film::game_options::GAME_VARIANT_LEN);
            // its first dword is the engine type, and the metadata 12 bytes in is content type 6
            assert_eq!(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]), f.header.game_engine_type);
            assert_eq!(u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]), 6);
            let out = export_game_variant(&f, &dir.join("gv.bin")).expect("export");
            let notes = std::fs::read_to_string(&out.notes_path).unwrap();
            assert!(!notes.contains("NOT a loadable"), "the old 'not a .bin' text must be gone");
            assert!(notes.contains(&f.header.game_variant.title) || f.header.game_variant.title.is_empty());
            match &out.bin_path {
                Some(bp) => {
                    bins += 1;
                    let bin = std::fs::read(bp).unwrap();
                    assert_eq!(bin.len(), out.bytes);
                    assert_eq!(bin.len(), crate::gamevariant::FILE_LEN);
                    let (img, info) = crate::gamevariant::parse_bin(&bin).expect("the file we wrote must parse");
                    assert!(info.hash_ok, "chunk hash must verify");
                    assert!(info.chdr_matches, "chdr must mirror the decoded metadata");
                    // the decoded struct re-encodes to the very same bitstream
                    let again = crate::gamevariant::build_bin(&img, info.chdr_tag).unwrap();
                    assert_eq!(again, bin, "decode -> encode must be the identity on our own output");
                    // and the title in the file is the title the film reported
                    let (title, _, _, _) = crate::gamevariant::titles(&img);
                    assert_eq!(title, f.header.game_variant.title);
                    // PRIVACY: nothing recording-machine specific can be in the file
                    let sid = &f.header.session_id;
                    if !sid.is_empty() && sid != "UNINIT_SESSIONID" {
                        for piece in sid.split('@').skip(2) {
                            if piece.len() >= 3 {
                                assert!(!bin.windows(piece.len()).any(|w| w == piece.as_bytes()), "the .bin leaked {piece:?}");
                            }
                        }
                    }
                }
                None => {
                    fallbacks += 1;
                    assert!(out.struct_path.is_some() && out.why_no_bin.is_some());
                    assert_eq!(out.bytes, film::game_options::GAME_VARIANT_LEN);
                }
            }
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
            // and the extraction report must be printable without it too, and must not say "raw struct only"
            let rep = extraction_report(&f);
            assert!(rep.contains("game variant"));
            assert!(!rep.contains("raw struct only"));
            if !sid.is_empty() && sid != "UNINIT_SESSIONID" {
                assert!(!rep.contains(sid.as_str()));
            }
            n += 1;
        }
        eprintln!("film_export: game variant export checked on {n} film(s): {bins} .bin, {fallbacks} raw-struct fallback");
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

/// #film-crc  A folder can hold a variant saved on ANOTHER build of the map (different scenario
/// palette CRC / quota count) next to one that matches. The smallest file is not necessarily the
/// right one, and which is smallest differs per machine, so the exporter must walk them all.
#[cfg(test)]
mod canvas_walk {
    use super::*;

    /// A real Forge film, its matching canvas, and a canvas that is NOT its map build (a variant
    /// of another map id -- the same refusal path as a different palette CRC / quota count).
    fn fixture() -> Option<(Film, PathBuf, PathBuf)> {
        let others = known_map_variants();
        for fp in film::spec_corpus() {
            let Ok(f) = film::open(&fp) else { continue };
            let mv = &f.header.map_variant;
            if mv.is_empty() { continue }
            for good in map_variant_templates(mv.map_id) {
                if export_map_variant_with_template(&f, &good).is_err() { continue }
                if let Some((bad, _)) = others.iter().find(|(p, id)| *id != mv.map_id && mvar::parse_variant(p).is_some()) {
                    return Some((f, good, bad.clone()));
                }
            }
        }
        None
    }

    #[test]
    fn a_wrong_build_canvas_first_does_not_sink_the_scene() {
        let Some((f, good, bad)) = fixture() else { panic!("no Forge film with a matching canvas and a foreign one on this machine: the gate cannot run") };
        assert!(export_map_variant_with_template(&f, &bad).is_err(), "fixture: the foreign canvas must be refused on its own");
        // the wrong one FIRST, as when it is the smallest file in the folder
        let (used, bytes) = export_with_candidates(&f, &[bad.clone(), good.clone()]).expect("must fall through to the matching canvas");
        assert_eq!(used, good);
        assert_eq!(bytes, export_map_variant_with_template(&f, &good).unwrap());
    }

    /// SABOTEUR: the single-candidate behaviour (what 1.6 did) on the same input must fail --
    /// proving the gate above is the thing that distinguishes the two.
    #[test]
    fn saboteur_single_candidate_goes_red() {
        let Some((f, good, bad)) = fixture() else { panic!("fixture missing") };
        let single = export_with_candidates(&f, &[bad][..1]);
        assert!(single.is_err(), "taking only the first candidate must fail here, else the gate above proves nothing");
        let _ = good;
    }

    #[test]
    fn no_match_says_how_many_were_tried_and_their_crcs() {
        let Some((f, _good, bad)) = fixture() else { panic!("fixture missing") };
        let bad2 = bad.clone();
        let e = export_with_candidates(&f, &[bad, bad2]).unwrap_err();
        let ExportError::NoMatchingCanvas { tried, film_crc, .. } = &e else { panic!("expected NoMatchingCanvas, got {e:?}") };
        assert_eq!(tried.len(), 2);
        assert_eq!(*film_crc, f.header.map_variant.palette_crc);
        let msg = e.to_string();
        assert!(msg.contains("none of the 2 saved variant(s)"), "{msg}");
        assert!(msg.contains("palette CRC") && msg.contains(&format!("{:#010x}", f.header.map_variant.palette_crc)), "{msg}");
        assert!(!msg.contains("Save a Forge variant"), "that advice is only for the no-canvas-at-all case: {msg}");
        assert_eq!(export_with_candidates(&f, &[]).unwrap_err(), ExportError::NoTemplate(f.header.map_variant.map_id));
    }
}

/// #film-scene  The film's Forge scene comes from the film, not from a canvas file.
#[cfg(test)]
mod film_scene_gates {
    use super::*;

    fn forge_films(limit: usize) -> Vec<Film> {
        film::spec_corpus().into_iter().filter_map(|p| film::open(&p).ok()).filter(|f| !f.header.map_variant.is_empty()).take(limit).collect()
    }

    /// The scene's object set IS the film's: same count, same positions, same palette entry,
    /// team and label, and the header facts the editor panel shows come from the film.
    #[test]
    fn the_scene_variant_is_the_films_own_object_set() {
        let films = forge_films(8);
        assert!(!films.is_empty(), "no Forge film on this machine: the gate cannot run");
        for f in &films {
            let mv = &f.header.map_variant;
            let v = variant_from_film(f).expect("a Forge film yields a scene");
            let want: Vec<&film::VariantObject> = mv.present_objects().collect();
            assert_eq!(v.objects.len(), want.len(), "object count");
            for (p, w) in v.objects.iter().zip(&want) {
                assert_eq!(p.pos, w.position, "position of slot {}", w.slot);
                assert_eq!((p.folder, p.item, p.team), (w.quota_index, w.item, w.team), "type/team of slot {}", w.slot);
                assert_eq!(p.label_idx, w.label_idx);
            }
            assert_eq!((v.map_id, v.globals.palette_crc, v.globals.num_quotas), (mv.map_id, mv.palette_crc, mv.num_quotas));
            assert_eq!(v.globals.bounds, mv.world_bounds);
            assert_eq!(v.labels, mv.labels);
            assert_eq!(v.globals.quotas.len(), mv.num_quotas as usize);
            let placed: u32 = v.globals.quotas.iter().map(|q| q.2 as u32).sum();
            assert!(placed as usize <= v.objects.len() && placed > 0, "quota 'placed' is recomputed from the objects");
        }
    }

    /// Where a donor exists, the in-memory scene equals what the file route drew (within the
    /// .mvar's own position quantisation): object for object.
    #[test]
    fn it_agrees_with_what_the_donor_route_produced() {
        let mut compared = 0;
        for f in forge_films(12) {
            let Ok((_, bytes)) = export_with_candidates(&f, &map_variant_templates(f.header.map_variant.map_id)) else { continue };
            let tmp = std::env::temp_dir().join(format!("hms_scene_gate_{}.mvar", std::process::id()));
            std::fs::write(&tmp, &bytes).unwrap();
            let file = mvar::parse_variant(&tmp).expect("donor export re-reads");
            let _ = std::fs::remove_file(&tmp);
            let mem = variant_from_film(&f).unwrap();
            assert_eq!(mem.objects.len(), file.objects.len());
            for (a, b) in mem.objects.iter().zip(&file.objects) {
                for k in 0..3 { assert!((a.pos[k] - b.pos[k]).abs() < 0.01, "pos axis {k}: {} vs {}", a.pos[k], b.pos[k]); }
                assert_eq!((a.folder, a.item, a.team, a.color, a.label_idx), (b.folder, b.item, b.team, b.color, b.label_idx));
                let d = a.fwd[0] * b.fwd[0] + a.fwd[1] * b.fwd[1] + a.fwd[2] * b.fwd[2];
                assert!(d > 0.999, "rotation drifted");
            }
            assert_eq!(mem.labels, file.labels);
            assert_eq!(mem.title, file.title);
            compared += 1;
        }
        assert!(compared > 0, "no film had a donor to compare against: vacuous");
    }

    /// A map with NO saved variant anywhere: the file route refuses (the old failure), the scene
    /// does not care.
    #[test]
    fn a_map_with_no_donor_still_has_its_scene() {
        let empty = std::env::temp_dir().join(format!("hms_no_variants_{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        crate::mapcat::TEST_VARIANT_DIRS.with(|c| *c.borrow_mut() = Some(vec![empty.clone()]));
        let films = forge_films(4);
        assert!(!films.is_empty());
        for f in &films {
            assert!(map_variant_templates(f.header.map_variant.map_id).is_empty(), "the seam must hide every donor");
            assert_eq!(export_with_candidates(f, &[]).unwrap_err(), ExportError::NoTemplate(f.header.map_variant.map_id));
            let v = variant_from_film(f).expect("the scene must build with no canvas");
            assert_eq!(v.objects.len(), f.header.map_variant.present_objects().count());
            assert!(!v.objects.is_empty());
        }
        crate::mapcat::TEST_VARIANT_DIRS.with(|c| *c.borrow_mut() = None);
        let _ = std::fs::remove_dir(&empty);
    }

    /// SABOTEUR: the file route (the old scene path) on the same donor-less input fails -- so the
    /// gate above is what separates the two. And a stock-map film still has no scene.
    #[test]
    fn saboteur_the_file_route_cannot_draw_without_a_donor() {
        let empty = std::env::temp_dir().join(format!("hms_no_variants_s_{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        crate::mapcat::TEST_VARIANT_DIRS.with(|c| *c.borrow_mut() = Some(vec![empty.clone()]));
        let f = forge_films(1).pop().expect("a Forge film");
        let old_route = export_map_variant(&f, &empty.join("x.mvar"));
        crate::mapcat::TEST_VARIANT_DIRS.with(|c| *c.borrow_mut() = None);
        let _ = std::fs::remove_dir_all(&empty);
        assert!(matches!(old_route, Err(ExportError::NoTemplate(_))), "the donor route must fail here, else the gate proves nothing");
    }

    /// Both hosts build the scene through `variant_from_film` and never through a file.
    #[test]
    fn both_hosts_use_the_in_memory_scene() {
        let gui = include_str!("main.rs");
        let script = include_str!("script_host.rs");
        let (g_start, _) = gui.split_at(gui.find("fn film_render_scene").unwrap());
        assert!(g_start.contains("film_export::variant_from_film(&f)"), "the GUI must build the scene from the film");
        assert!(!gui.contains("scene_request"), "the file-based scene request is gone from the GUI");
        assert!(script.matches("variant_from_film").count() >= 2, "both script-host call sites use the film's own variant");
        assert!(!script.contains("scene_request"));
        // read-only: the scene render clears every file path so File > Save has nothing to write to
        let f = &gui[gui.find("fn film_render_scene").unwrap()..];
        let body = &f[..f.find("\n    }\n").unwrap()];
        assert!(body.contains("current_variant_path = None") && body.contains("new_variant_template = None"));
        assert!(!body.contains("save_last_variant") && !body.contains("note_recent_variant"));
    }
}
