//! Map auto-resolution — the "no file paths" catalog:
//!
//!   1. Find the Steam install (registry / well-known Linux paths) + extra libraries
//!      (libraryfolders.vdf), or a Windows Store / Xbox app install
//!   2. Scan `…\Halo The Master Chief Collection\{haloreach,halo4}\maps\*.map`
//!      (plus subscribed workshop content 976730)
//!   3. Read each `.map`'s header to classify the title and identify the scenario
//!
//! No running game is required to enumerate; the caller shows the result as a pick-list.
//! Also here: the persisted last-map / last-variant paths and the variant-folder detection
//! (game folders + per-account MCC save folders) used by the Open-variant browser.

use std::path::{Path, PathBuf};

/// MCC Steam app id for Reach workshop content.
const MCC_WORKSHOP_APP_ID: &str = "976730";
/// Offset of the null-terminated scenario name string in a Reach `.map` header.
const SCENARIO_NAME_OFFSET: u64 = 432;
const SCENARIO_NAME_MAX: usize = 256;

// Halo 4 (MCC) cache header facts -- docs/halo4_support_plan.md section 2.1. The header is
// the Reach U13 layout minus 8 bytes after 0x58, so the build string and the name/scenario
// strings sit 8 bytes EARLIER than in a Reach cache. Everything below is verified on all 54
// files under `halo4/maps`.
/// Per-title folders under the MCC root that carry a `maps` subfolder, Reach first.
const GAME_DIRS: [&str; 3] = ["haloreach", "halo4", "groundhog"];
/// Bytes read from the start of a cache to classify it (covers every field inspected).
const HEADER_PROBE_LEN: usize = 1280;
/// The cache magic as it appears ON DISK (little-endian "head").
const CACHE_MAGIC: &[u8; 4] = b"daeh";
/// Cache version shared by Reach and Halo 4 MCC caches.
const CACHE_VERSION: u32 = 13;
/// The SINGLE known Halo 4 MCC build string (two spaces after "Apr"). Every shipped
/// `halo4/maps/*.map` carries it; a new MCC build would need a new entry here.
const HALO4_BUILD: &str = "Apr  1 2023 17:35:22";
/// The SINGLE known Halo 2 Anniversary (MCC `groundhog`) build string. #h2a: its cache format
/// is Halo 4's (same header layout, so the same offsets below apply); see
/// docs/h2a_support_plan.md. `groundhog/maps/ca_forge_skybox03_built.map` is a user/EK-built
/// cache carrying "Jan  3 2024 10:26:12" instead and is classified by its folder.
const H2A_BUILD: &str = "Jun 13 2023 20:21:18";
/// Build-string offset in a Halo 4 cache (Reach: 160). 32 bytes, null-terminated.
const HALO4_BUILD_OFFSET: usize = 152;
const HALO4_BUILD_LEN: usize = 32;
/// i16 header `type` at 0x18: 0 = campaign/firefight, 1 = multiplayer (incl. the Forge
/// canvases), 3 = `shared.map`, 4 = `campaign.map`.
const HALO4_TYPE_OFFSET: usize = 0x18;
/// Null-terminated scenario path, e.g. `levels\multi\ca_forge_ravine\ca_forge_ravine`.
const HALO4_SCENARIO_OFFSET: usize = 0xD8;

/// Which MCC title a cache belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Game {
    Reach,
    Halo4,
    /// Halo 2 Anniversary (MCC `groundhog`). #h2a
    H2A,
}

impl Game {
    /// Lowercase short name, as printed by `--dump-maps`.
    pub fn short_name(self) -> &'static str {
        match self {
            Game::Reach => "reach",
            Game::Halo4 => "halo4",
            Game::H2A => "h2a",
        }
    }
    /// #map-picker: the title as a person calls it (the map picker's game row).
    pub fn display_name(self) -> &'static str {
        match self {
            Game::Reach => "Reach",
            Game::Halo4 => "Halo 4",
            Game::H2A => "Halo 2 Anniversary",
        }
    }
    /// The MCC install sub-folder this title's `maps/` lives in.
    pub fn folder(self) -> &'static str {
        match self {
            Game::Reach => "haloreach",
            Game::Halo4 => "halo4",
            Game::H2A => "groundhog",
        }
    }
}

/// One detected map file.
#[derive(Clone, Debug)]
pub struct MapCandidate {
    pub path: PathBuf,
    pub file_name: String,   // "forge_bunkerworld.map"
    pub scenario_name: String, // "levels\forge\…" (offset 432), or "" if unreadable
    pub stem: String,        // short scenario stem, e.g. "forge_bunkerworld"
    /// True = a MODDED/custom map (Steam Workshop content), false = a map in the game's own
    /// folder. The UI lists the two groups separately.
    pub modded: bool,
    /// Reach map id (matches a .mvar's `c_map_variant::m_map_id`), so a variant can be
    /// routed to the correct base map. None when not yet resolved / not a Reach cache.
    pub map_id: Option<u32>,
    /// Which title this cache belongs to (header build string is the primary key).
    pub game: Game,
    /// The Halo 4 header `type` (0 campaign/ff, 1 mp, 3 shared, 4 campaign); None for Reach.
    pub h4_type: Option<i16>,
}

/// Classification is by FOLDER. Everything in the game's own `haloreach\maps` folder is a
/// built-in map (user copies included); everything under `steamapps\workshop` is a modded map.
/// Workshop items for titles HMS does not support are dropped from the list altogether -- see
/// `workshop_item_game`.
fn is_modded_map(path: &Path) -> bool {
    let p = path.to_string_lossy().to_lowercase().replace('\\', "/");
    p.contains("/steamapps/workshop/")
}

/// A Steam Workshop item (folder `workshop\content\976730\<item id>`) targets exactly one MCC
/// title. `ModInfo.json` carries `"Engine": "HaloReach" | "Halo3" | "Halo1" | "Halo2" | ...`; items
/// without a manifest are accepted when they ship `*.haloreach_mapinfo` files or a `haloreach\`
/// subfolder. Halo 1/2/3 caches share Reach's `head`/v13 header, so the cache file itself
/// cannot tell those titles apart -- the item manifest can. (Halo 4 caches CAN be told apart
/// by their build string, see `classify_header`; the manifest is still consulted for Workshop
/// items via `workshop_item_game`.)
fn workshop_item_is_reach(map_path: &Path) -> bool {
    let Some(item) = workshop_item_dir(map_path) else { return true };
    if let Some(engine) = workshop_item_engine(&item) {
        return engine == "haloreach";
    }
    if item.join("haloreach").is_dir() { return true; }
    if let Ok(rd) = std::fs::read_dir(&item) {
        if rd.flatten().any(|e| e.path().extension().map_or(false, |x| x.eq_ignore_ascii_case("haloreach_mapinfo"))) {
            return true;
        }
    }
    false
}

/// The Workshop item folder (`workshop\content\<appid>\<item id>`) a map path lives under,
/// or None when the path is not inside a Workshop item.
fn workshop_item_dir(map_path: &Path) -> Option<PathBuf> {
    let comps: Vec<String> = map_path.components().map(|c| c.as_os_str().to_string_lossy().to_lowercase()).collect();
    let i = comps.iter().position(|c| c == "workshop")?;
    // workshop / content / <appid> / <item id>
    if comps.len() <= i + 3 { return None; }
    let mut item = PathBuf::new();
    for c in map_path.components().take(i + 4) { item.push(c.as_os_str()); }
    Some(item)
}

/// The LOWERCASE `"Engine"` string from a Workshop item's `ModInfo.json` ("haloreach",
/// "halo4", "halo3", ...), or None when there is no manifest / no Engine key. Minimal,
/// dependency-free read.
fn workshop_item_engine(item: &Path) -> Option<String> {
    let text = std::fs::read_to_string(item.join("ModInfo.json")).ok()?;
    let t = text.to_lowercase();
    let k = t.find("\"engine\"")?;
    let rest = &t[k + 8..];
    let q = rest.find('"')?;
    let rest = &rest[q + 1..];
    let e = rest.find('"')?;
    Some(rest[..e].to_string())
}

/// Which title a Workshop item targets, or None for titles HMS does not list at all.
/// Reach acceptance is `workshop_item_is_reach`; Halo 4 items are accepted ONLY on
/// an explicit `"Engine": "Halo4"` manifest entry.
pub fn workshop_item_game(map_path: &Path) -> Option<Game> {
    if workshop_item_is_reach(map_path) {
        return Some(Game::Reach);
    }
    let item = workshop_item_dir(map_path)?;
    if workshop_item_engine(&item).as_deref() == Some("halo4") {
        return Some(Game::Halo4);
    }
    None
}

/// Where a modded map comes from, for the pick-list label: the Steam Workshop item id
/// (the folder under `workshop\content\976730`), or "copy" for a user copy in the game folder.
fn mod_source(path: &Path) -> Option<String> {
    let comps: Vec<String> = path.components().map(|c| c.as_os_str().to_string_lossy().to_lowercase()).collect();
    if let Some(i) = comps.iter().position(|c| c == "workshop") {
        // workshop / content / <appid> / <item id> / ...
        if let Some(item) = comps.get(i + 3) {
            return Some(format!("Workshop {item}"));
        }
        return Some("Workshop".into());
    }
    let f = comps.last().cloned().unwrap_or_default();
    if f.contains(" - copy") { Some("copy".into()) } else { Some("custom".into()) }
}

impl MapCandidate {
    /// Display label for the pick-list — prefer the scenario stem, fall back to
    /// the file name.
    pub fn label(&self) -> String {
        let base = if self.stem.is_empty() {
            self.file_name.clone()
        } else if self.stem.eq_ignore_ascii_case(
            Path::new(&self.file_name).file_stem().and_then(|s| s.to_str()).unwrap_or(""),
        ) {
            self.file_name.clone()
        } else {
            format!("{} ({})", self.stem, self.file_name)
        };
        // Modded maps say where they come from so two mods shipping the same file name
        // (e.g. several Workshop items each carrying a `forge_halo.map`) can be told apart.
        let base = if self.modded {
            match mod_source(&self.path) {
                Some(src) => format!("{base}  [{src}]"),
                None => base,
            }
        } else {
            base
        };
        // Halo 4 caches are flagged so they can never be mistaken for a Reach map.
        match self.game {
            Game::Reach => base,
            Game::Halo4 => format!("{base} [Halo 4]"),
            Game::H2A => format!("{base} [Halo 2A]"),
        }
    }
}

/// The per-user settings dir (`project::settings_dir`, which has the Linux/macOS XDG fallback).
fn settings_dir() -> Option<PathBuf> {
    crate::project::settings_dir()
}

fn last_map_path_file() -> Option<PathBuf> {
    settings_dir().map(|d| d.join("last_map.txt"))
}

/// The last successfully-loaded map path, if it still exists on disk.
pub fn load_last_map() -> Option<String> {
    let f = last_map_path_file()?;
    let s = std::fs::read_to_string(f).ok()?;
    let s = s.trim().to_string();
    if !s.is_empty() && Path::new(&s).exists() {
        Some(s)
    } else {
        None
    }
}

/// Remember the last-loaded map so it can be pre-selected next launch.
pub fn save_last_map(path: &str) {
    if let Some(dir) = settings_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("last_map.txt"), path);
    }
}

/// The last opened .mvar variant path, if it still exists on disk.
pub fn load_last_variant() -> Option<String> {
    let f = settings_dir().map(|d| d.join("last_variant.txt"))?;
    let s = std::fs::read_to_string(f).ok()?;
    let s = s.trim().to_string();
    if !s.is_empty() && Path::new(&s).exists() {
        Some(s)
    } else {
        None
    }
}

/// Remember the last opened .mvar variant so it can be auto-loaded next launch.
pub fn save_last_variant(path: &str) {
    if let Some(dir) = settings_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("last_variant.txt"), path);
    }
}

/// Pure resource caches that aren't loadable scenarios — hidden from the list.
fn is_resource_cache(file_name: &str) -> bool {
    matches!(
        file_name.to_lowercase().as_str(),
        "shared.map" | "campaign.map" | "single_player_shared.map" | "bitmaps.map" | "sounds.map" | "loc.map"
        // Halo 4 menu cache (not shipped in the current MCC build, hidden if it appears).
        | "mainmenu.map"
    )
}

/// Enumerate every detectable map, de-duplicated by path. Resource caches are
/// hidden; Reach maps first, then Halo 4 maps; within each game forge_* maps sort to the top
/// (the common case), then alphabetical.
pub fn enumerate() -> Vec<MapCandidate> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for folder in probable_maps_folders() {
        for c in scan_folder(&folder) {
            if is_resource_cache(&c.file_name) {
                continue;
            }
            if seen.insert(c.path.clone()) {
                out.push(c);
            }
        }
    }
    out.sort_by(|a, b| {
        let fa = a.label().to_lowercase().starts_with("forge");
        let fb = b.label().to_lowercase().starts_with("forge");
        // game group first (Reach < Halo 4 < Halo 2A), then forge-first, then alphabetical
        let grp = |g: Game| match g { Game::Reach => 0u8, Game::Halo4 => 1, Game::H2A => 2 };
        let ga = grp(a.game);
        let gb = grp(b.game);
        ga.cmp(&gb)
            .then_with(|| fb.cmp(&fa)) // forge maps first
            .then_with(|| a.label().to_lowercase().cmp(&b.label().to_lowercase()))
    });
    out
}

/// Candidate maps folders across EVERY detected MCC install — Steam AND the Windows Store /
/// Xbox Game Pass version. A user may have either or both; all are scanned and their maps merged
/// (deduplicated by path in `enumerate`), so map loading works regardless of storefront.
///
/// An explicit `HMS_MCC_DIRS` env override (`;`-separated) is honored first, for non-standard
/// installs — each entry may be an MCC root, a `haloreach` folder, or a `haloreach\maps` folder.
fn probable_maps_folders() -> Vec<PathBuf> {
    let mut folders = Vec::new();
    // 0) Manual override.
    if let Ok(dirs) = std::env::var("HMS_MCC_DIRS") {
        for d in dirs.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            folders.extend(maps_dirs_under(&PathBuf::from(d)));
        }
    }
    // 1) Steam libraries (registry → libraryfolders.vdf). PREFERRED: if a Steam MCC is present we
    //    read from it exclusively (it carries Workshop content and is the modding-friendly build).
    let mut steam_found = false;
    for lib in steam_library_folders() {
        // `haloreach\maps` and its `halo4\maps` sibling (Reach first, so Reach candidates
        // come first in the merged list).
        for game_dir in GAME_DIRS {
            let maps = lib
                .join("steamapps")
                .join("common")
                .join("Halo The Master Chief Collection")
                .join(game_dir)
                .join("maps");
            if maps.is_dir() {
                folders.push(maps);
                steam_found = true;
            }
        }
        let workshop = lib
            .join("steamapps")
            .join("workshop")
            .join("content")
            .join(MCC_WORKSHOP_APP_ID);
        if workshop.is_dir() {
            folders.push(workshop);
        }
    }
    // 2) Windows Store / Xbox Game Pass install(s) — ONLY when there's no Steam MCC. A user with
    //    both storefronts gets the Steam version by default (per project convention).
    if !steam_found {
        folders.extend(xbox_maps_folders());
    }
    folders
}

/// Xbox app / Microsoft Store MCC install(s). There is no stable registry value for the content
/// path, so scan every fixed drive for the two known layouts:
///   * `<drive>\XboxGames\<game>\Content\`               (modern Xbox app; user picks the drive)
///   * `<drive>\Program Files\ModifiableWindowsApps\<game>\`  (moddable Store games are surfaced here)
/// The `<game>` folder name is matched loosely (we probe each child for a `haloreach\maps` layout),
/// so a rename or localized name still resolves.
fn xbox_maps_folders() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for drive in fixed_drive_roots() {
        let parents = [
            drive.join("XboxGames"),
            drive.join("Program Files").join("ModifiableWindowsApps"),
        ];
        for parent in parents {
            let Ok(rd) = std::fs::read_dir(&parent) else { continue };
            for e in rd.flatten() {
                let child = e.path();
                if child.is_dir() {
                    out.extend(maps_dirs_under(&child));
                }
            }
        }
    }
    out
}

/// Given an MCC-ish root, return whichever `haloreach\maps` folders exist beneath it. Handles the
/// Steam layout (`<root>\haloreach\maps`) and the Xbox layout (`<root>\Content\haloreach\maps`),
/// and accepts a `haloreach` or `haloreach\maps` folder passed directly.
fn maps_dirs_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    // the `halo4` sibling is probed after `haloreach` in both layouts
    let candidates = [
        root.join("haloreach").join("maps"),
        root.join("halo4").join("maps"),
        root.join("groundhog").join("maps"),
        root.join("Content").join("haloreach").join("maps"),
        root.join("Content").join("halo4").join("maps"),
        root.join("Content").join("groundhog").join("maps"),
        root.join("maps"), // a `haloreach` (or `halo4`) folder was passed
        root.to_path_buf(), // a `maps` folder was passed
    ];
    for c in candidates {
        // The last two candidates are only valid if they actually contain .map files.
        if c.is_dir() && (c.ends_with("maps") || c.join("info").is_dir()) {
            out.push(c);
        }
    }
    out
}

/// Existing fixed-drive roots (C:..Z:). A/B are skipped (legacy floppies). Cheap: a non-existent
/// drive letter fails `is_dir` immediately.
fn fixed_drive_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for letter in b'C'..=b'Z' {
        let root = PathBuf::from(format!("{}:\\", letter as char));
        if root.is_dir() {
            roots.push(root);
        }
    }
    roots
}

/// Steam root + libraries parsed from libraryfolders.vdf.
fn steam_library_folders() -> Vec<PathBuf> {
    let mut libs = Vec::new();
    let Some(root) = steam_install_path() else { return libs };
    let root = PathBuf::from(root);
    libs.push(root.clone());
    let vdf = root.join("steamapps").join("libraryfolders.vdf");
    if let Ok(text) = std::fs::read_to_string(&vdf) {
        for p in parse_library_folders_vdf(&text) {
            libs.push(PathBuf::from(p));
        }
    }
    libs
}

/// Read the Steam install path from the registry via `reg query` (dependency-free
/// and always present on Windows). Tries HKLM WOW6432Node then HKCU.
#[cfg(windows)]
fn steam_install_path() -> Option<String> {
    if let Some(v) = reg_query(r"HKLM\SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath") {
        return Some(v);
    }
    if let Some(v) = reg_query(r"HKCU\SOFTWARE\Valve\Steam", "SteamPath") {
        return Some(v.replace('/', "\\"));
    }
    None
}

/// Linux Steam has no registry; it lives at a handful of well-known paths.
/// `~/.steam/steam` and `~/.steam/root` are symlinks into the real data dir, so
/// the canonical `~/.local/share/Steam` is tried first. The last candidate is the
/// Flatpak sandbox location.
#[cfg(not(windows))]
fn steam_install_path() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            candidates.push(PathBuf::from(xdg).join("Steam"));
        }
    }
    for rel in [
        ".local/share/Steam",
        ".steam/steam",
        ".steam/root",
        ".var/app/com.valvesoftware.Steam/.local/share/Steam",
    ] {
        candidates.push(PathBuf::from(&home).join(rel));
    }
    candidates
        .into_iter()
        // Must actually be a Steam data dir, not just an existing path.
        .find(|p| p.join("steamapps").is_dir())
        .map(|p| p.to_string_lossy().into_owned())
}

#[cfg(windows)]
fn reg_query(key: &str, value: &str) -> Option<String> {
    let out = std::process::Command::new("reg")
        .args(["query", key, "/v", value])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Line form: "    InstallPath    REG_SZ    C:\Program Files (x86)\Steam"
    for line in text.lines() {
        if let Some(idx) = line.find("REG_SZ") {
            let val = line[idx + "REG_SZ".len()..].trim();
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }
    None
}

/// Pull every `"path" "…"` value out of libraryfolders.vdf.
fn parse_library_folders_vdf(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0;
    while let Some(p) = text[idx..].find("\"path\"") {
        let p = idx + p + "\"path\"".len();
        // next quoted string is the value
        if let Some(q1) = text[p..].find('"') {
            let q1 = p + q1 + 1;
            if let Some(q2) = text[q1..].find('"') {
                let q2 = q1 + q2;
                out.push(text[q1..q2].replace("\\\\", "\\"));
                idx = q2 + 1;
                continue;
            }
        }
        break;
    }
    out
}

/// Scan a folder for .map files (recursive under workshop content, else flat).
fn scan_folder(folder: &Path) -> Vec<MapCandidate> {
    // Separator-agnostic: the same layout is `\steamapps\workshop\...` on Windows
    // and `/steamapps/workshop/...` on Linux.
    let folder_key = folder.to_string_lossy().to_lowercase().replace('\\', "/");
    let recursive =
        folder_key.contains(&format!("/steamapps/workshop/content/{MCC_WORKSHOP_APP_ID}"));
    let mut out = Vec::new();
    let mut stack = vec![folder.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    stack.push(path);
                }
                continue;
            }
            if path.extension().map(|e| e.eq_ignore_ascii_case("map")).unwrap_or(false) {
                // Drop Workshop items for titles HMS does not support (Reach and Halo 4 are listed).
                if recursive && workshop_item_game(&path).is_none() {
                    continue;
                }
                if let Some(c) = read_candidate(&path) {
                    out.push(c);
                }
            }
        }
        if !recursive {
            break; // top-level only
        }
    }
    out
}

/// Null-terminated printable ASCII string at `off` (at most `max` bytes) in a header probe,
/// or None when it is absent, shorter than 4 chars, or contains non-printable bytes.
fn printable_cstr(buf: &[u8], off: usize, max: usize) -> Option<String> {
    let end = buf.len().min(off.checked_add(max)?);
    let s = buf.get(off..end)?;
    let len = s.iter().position(|&b| b == 0).unwrap_or(s.len());
    let printable = len >= 4 && s[..len].iter().all(|&b| (0x20..=0x7E).contains(&b));
    printable.then(|| String::from_utf8_lossy(&s[..len]).into_owned())
}

/// Does the probe start with the shared `head`/v13 cache header?
fn cache_header_ok(buf: &[u8]) -> bool {
    buf.len() >= 8 && &buf[..4] == CACHE_MAGIC && u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) == CACHE_VERSION
}

/// Classify a cache from its header probe. PRIMARY key: the Halo 4 build string at +152
/// (Reach caches carry 8 bytes of hash there, so it never reads as printable). Fallbacks for an
/// unknown Halo 4 build: the path has a `halo4` folder component, or the Workshop manifest says
/// `"Engine": "Halo4"` -- both still require the `head`/v13 header. Everything else is Reach.
/// Returns (game, Halo 4 header `type`).
fn classify_header(buf: &[u8], path: &Path) -> (Game, Option<i16>) {
    if !cache_header_ok(buf) {
        return (Game::Reach, None);
    }
    let build = printable_cstr(buf, HALO4_BUILD_OFFSET, HALO4_BUILD_LEN);
    let in_folder = |name: &str| path.components().any(|c| c.as_os_str().to_string_lossy().eq_ignore_ascii_case(name));
    // #h2a: groundhog caches use the Halo 4 header layout, so they are recognised the same way -
    // by the build string at +152, or by living in a `groundhog` folder (EK-built caches carry a
    // different build string).
    let game = if build.as_deref() == Some(H2A_BUILD) || in_folder("groundhog") {
        Game::H2A
    } else if build.as_deref() == Some(HALO4_BUILD)
        || in_folder("halo4")
        || workshop_item_dir(path).and_then(|d| workshop_item_engine(&d)).as_deref() == Some("halo4")
    {
        Game::Halo4
    } else {
        return (Game::Reach, None);
    };
    let t = buf
        .get(HALO4_TYPE_OFFSET..HALO4_TYPE_OFFSET + 2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]));
    (game, t)
}

/// Read a .map's header → candidate. One `HEADER_PROBE_LEN`-byte read classifies the title
/// first; the scenario name is at offset 432 (Reach) or 0xD8 (Halo 4).
fn read_candidate(path: &Path) -> Option<MapCandidate> {
    use std::io::Read;
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() < 1024 {
        return None;
    }
    let file_name = path.file_name()?.to_string_lossy().into_owned();

    let mut probe = vec![0u8; HEADER_PROBE_LEN];
    if let Ok(mut f) = std::fs::File::open(path) {
        // fill the probe (a short read on a tiny file just leaves the tail zeroed)
        let mut got = 0;
        while got < probe.len() {
            match f.read(&mut probe[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
        probe.truncate(got);
    } else {
        probe.clear();
    }
    let (game, h4_type) = classify_header(&probe, path);
    let scenario_name = match game {
        Game::Reach => printable_cstr(&probe, SCENARIO_NAME_OFFSET as usize, SCENARIO_NAME_MAX),
        // Halo 4 and Halo 2 Anniversary share the shifted header layout
        Game::Halo4 | Game::H2A => printable_cstr(&probe, HALO4_SCENARIO_OFFSET, SCENARIO_NAME_MAX),
    }
    .unwrap_or_default();
    let stem = if !scenario_name.is_empty() {
        scenario_name
            .rsplit(|c| c == '\\' || c == '/')
            .next()
            .unwrap_or(&scenario_name)
            .to_string()
    } else {
        path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    };

    let modded = is_modded_map(path);
    let map_id = read_map_id(path);
    Some(MapCandidate { path: path.to_path_buf(), file_name, scenario_name, stem, modded, map_id, game, h4_type })
}

/// Forge map-variant folders for every detected MCC install — the natural default
/// when opening a `.mvar`. Derived from the same roots as the maps folders
/// (`<MCC>/haloreach/maps` → `<MCC>/haloreach/map_variants`), so this inherits the
/// Steam / Windows-Store detection and the `HMS_MCC_DIRS` override for free.
/// `map_variants` (the editable ones) is listed before `hopper_map_variants`.
/// Workshop folders fall out naturally: they have no sibling variants dir.
pub fn variant_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for maps in probable_maps_folders() {
        let Some(haloreach) = maps.parent() else { continue };
        // `halo4\map_variants` holds Halo 4 .mvar files (chunk v50/v51: see `h4_variant_dirs`) and
        // `groundhog\map_variants` holds Halo 2 Anniversary ones (chunk v52, not yet decoded) -
        // neither is a Reach variant, so keep both out of the Reach list. #h2a
        if haloreach.file_name().map_or(false, |n| {
            let n = n.to_string_lossy();
            n.eq_ignore_ascii_case("halo4") || n.eq_ignore_ascii_case("groundhog")
        }) {
            continue;
        }
        for sub in ["map_variants", "hopper_map_variants"] {
            let d = haloreach.join(sub);
            if d.is_dir() && !out.contains(&d) {
                out.push(d);
            }
        }
    }
    out
}

/// The Halo 4 variant folders (`<MCC>/halo4/map_variants` + `hopper_map_variants`) of
/// every detected install - the Halo 4 counterpart of `variant_dirs`, kept separate because
/// the Reach list is what the variant browser keys on.
fn h4_variant_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for maps in probable_maps_folders() {
        let Some(halo4) = maps.parent() else { continue };
        if !halo4.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("halo4")) { continue; }
        for sub in ["map_variants", "hopper_map_variants"] {
            let d = halo4.join(sub);
            if d.is_dir() && !out.contains(&d) { out.push(d); }
        }
    }
    out
}

/// MCC install roots (`<MCC>` of `<MCC>/haloreach/maps` and `<MCC>/halo4/maps`), for
/// content that lives beside the game folders (`data/ui/Localization`). Same detection as the
/// maps folders; workshop folders (no `<game>/maps` shape) are skipped.
pub(crate) fn mcc_install_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for maps in probable_maps_folders() {
        let Some(game) = maps.parent() else { continue };
        let is_game = game.file_name().map_or(false, |n| GAME_DIRS.iter().any(|g| n.to_string_lossy().eq_ignore_ascii_case(g)));
        if !is_game { continue; }
        if let Some(root) = game.parent() {
            if !out.iter().any(|r| r == root) { out.push(root.to_path_buf()); }
        }
    }
    out
}

/// Read the map id for routing a .mvar to its base map. The numeric id is NOT in the
/// `.map` cache header — it lives in the companion `.mapinfo` BLF (`<maps>\info\<stem>.mapinfo`,
/// or beside the map for workshop). Inside, the `levl` chunk's payload begins with `map_id`
/// (big-endian u32) — verified: forge_halo.mapinfo → 0x0BBE (3006) = "Forge World", matching
/// its variants. The 12-byte BLF chunk header (sig[4]+size_be[4]+major[2]+minor[2]) precedes it.
pub fn read_map_id(map_path: &Path) -> Option<u32> {
    let info = find_mapinfo(map_path)?;
    // The header (\_blf + levl chunk) sits at the very start; a 4 KiB read covers it.
    let data = {
        use std::io::Read;
        let mut f = std::fs::File::open(&info).ok()?;
        let mut buf = vec![0u8; 4096];
        let n = f.read(&mut buf).ok()?;
        buf.truncate(n);
        buf
    };
    let levl = find_bytes(&data, b"levl")?;
    let p = levl + 12; // skip the BLF chunk header
    let b = data.get(p..p + 4)?;
    let id = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    if id == 0 || id == 0xFFFF_FFFF { None } else { Some(id) }
}

/// Locate a map's `.mapinfo` companion: the sibling `info\` folder (stock layout) first,
/// then beside the `.map` (workshop layout).
fn find_mapinfo(map_path: &Path) -> Option<PathBuf> {
    let stem = map_path.file_stem()?;
    let dir = map_path.parent()?;
    let candidates = [
        dir.join("info").join(stem).with_extension("mapinfo"),
        dir.join(stem).with_extension("mapinfo"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// First index of `needle` in `hay` (small linear scan — the header is only a few KiB).
fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Labelled folders that actually hold map variants, for the Open-variant browser's
/// quick-access sidebar. Three sources, in the order people reach for them:
///   1. the game's own `map_variants` / `hopper_map_variants` (beside `maps\`),
///   2. every per-account MCC save folder — `MCC\LocalFiles\<XUID>\<Game>\Map`, which is where the
///      game writes what you save in Forge, and the folder people could not otherwise find,
///   3. that account's `.../Map/temp`, used for in-progress saves.
/// Only directories that EXIST and CONTAIN at least one .mvar are returned, so the sidebar never
/// shows a dead link. The label carries the account id because a user has several.
pub fn quick_variant_dirs() -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let mut push = |label: String, p: PathBuf| {
        if !p.is_dir() || out.iter().any(|(_, q)| q == &p) {
            return;
        }
        // Two accounts under one profile would otherwise produce two identical rows.
        let label = if out.iter().any(|(l, _)| l == &label) {
            let id = p.ancestors().nth(2).and_then(|a| a.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            format!("{label} …{}", id.get(id.len().saturating_sub(4)..).unwrap_or(&id))
        } else {
            label
        };
        out.push((label, p));
    };
    for d in variant_dirs() {
        let name = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let label = if name == "hopper_map_variants" { "[Reach] hopper_map_variants" } else { "[Reach] map_variants" };
        push(label.to_string(), d);
    }
    // The Halo 4 game folders, after the Reach rows. Opening one of these routes to the
    // Halo 4 loader (`import_mvar` -> `h4_import_mvar`), which loads the base map itself.
    for d in h4_variant_dirs() {
        let name = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let label = if name == "hopper_map_variants" { "[Halo 4] hopper_map_variants" } else { "[Halo 4] map_variants" };
        push(label.to_string(), d);
    }
    // (the per-account `MCC\LocalFiles\<XUID>\Halo4\Map` folder is picked up by the loop below,
    // which lists every game's save folder that holds a .mvar)
    for root in mcc_localfiles_roots() {
        // `<profile>/AppData/LocalLow/MCC/LocalFiles` — name the profile, because a machine can
        // carry several Windows users (and a Proton prefix adds "steamuser" on top).
        let profile = root
            .ancestors()
            .nth(4)
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Ok(rd) = std::fs::read_dir(&root) else { continue };
        let mut xuids: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        xuids.sort();
        for x in xuids {
            let Ok(games) = std::fs::read_dir(&x) else { continue };
            let mut gdirs: Vec<PathBuf> = games.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
            gdirs.sort();
            for g in gdirs {
                let game = g.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let map = g.join("Map");
                if dir_has_mvar(&map) {
                    push(format!("[save] {game} · {profile}"), map.clone());
                }
                let temp = map.join("temp");
                if dir_has_mvar(&temp) {
                    push(format!("[save] {game} temp · {profile}"), temp);
                }
            }
        }
    }
    out
}

/// Does this directory hold at least one .mvar? Shallow, and stops at the first hit.
fn dir_has_mvar(d: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(d) else { return false };
    rd.flatten().any(|e| {
        e.path().extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar"))
    })
}

/// Candidate `MCC\LocalFiles` roots. `HMS_MCC_LOCALFILES` overrides (`;`-separated).
/// Shared with the per-account variant scans in `mvar` / `h4::mvar`.
#[cfg(windows)]
pub(crate) fn mcc_localfiles_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(v) = std::env::var("HMS_MCC_LOCALFILES") {
        out.extend(v.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    if let Ok(up) = std::env::var("USERPROFILE") {
        out.push(PathBuf::from(up).join("AppData").join("LocalLow").join("MCC").join("LocalFiles"));
    }
    out
}

/// On Linux the same folder lives inside the Proton prefix (or a mounted Windows install),
/// so search the prefixes we already know how to find plus any mounted NTFS user profile.
#[cfg(not(windows))]
pub(crate) fn mcc_localfiles_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("HMS_MCC_LOCALFILES") {
        out.extend(v.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    let tail = |base: PathBuf| base.join("AppData").join("LocalLow").join("MCC").join("LocalFiles");
    // Proton prefix for MCC (app 976730): steamapps/compatdata/976730/pfx/drive_c/users/steamuser
    if let Some(steam) = steam_install_path() {
        for lib in [PathBuf::from(&steam)].into_iter().chain(steam_library_folders().into_iter()) {
            let users = lib.join("steamapps").join("compatdata").join("976730").join("pfx").join("drive_c").join("users");
            let Ok(rd) = std::fs::read_dir(&users) else { continue };
            for e in rd.flatten() {
                out.push(tail(e.path()));
            }
        }
    }
    // A mounted Windows install: /mnt/*/Users/<name>.
    for mnt in ["/mnt", "/media", "/run/media"] {
        let Ok(rd) = std::fs::read_dir(mnt) else { continue };
        for vol in rd.flatten() {
            let users = vol.path().join("Users");
            let Ok(u) = std::fs::read_dir(&users) else { continue };
            for e in u.flatten() {
                if e.path().is_dir() {
                    out.push(tail(e.path()));
                }
            }
        }
    }
    out.retain(|p| p.is_dir());
    out
}

#[cfg(test)]
mod quick_dirs_tests {
    /// Smoke: the quick-access list should find the game's variant folders on a machine that has
    /// MCC. Prints what it found so the labels can be eyeballed.
    /// `cargo test -p hms-app quick_dirs_smoke -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn quick_dirs_smoke() {
        for (label, path) in super::quick_variant_dirs() {
            eprintln!("{label:34} {}", path.display());
        }
    }
}

// Cache classification tests. Each test returns early (with a note) when the MCC folder
// it needs is not on this machine, so CI without the game still passes.
#[cfg(test)]
mod h4_tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The detected `halo4\maps` folder (Steam / Xbox / HMS_MCC_DIRS), if any.
    fn h4_maps_folder() -> Option<PathBuf> {
        probable_maps_folders().into_iter().find(|m| {
            m.parent()
                .and_then(|p| p.file_name())
                .map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("halo4"))
                && m.is_dir()
        })
    }

    /// The detected `haloreach\maps` folder, if any.
    fn reach_maps_folder() -> Option<PathBuf> {
        probable_maps_folders().into_iter().find(|m| {
            m.parent()
                .and_then(|p| p.file_name())
                .map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("haloreach"))
                && m.is_dir()
        })
    }

    fn h4_map_files(folder: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(folder)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().map_or(false, |x| x.eq_ignore_ascii_case("map")))
            .collect();
        v.sort();
        v
    }

    /// (a) Every Halo 4 cache classifies as Halo 4, and the DISTINCT shipped set (user copies
    /// such as `wraparound - Copy.map` collapse onto their scenario path) has the measured type
    /// histogram: 52 caches = 25 mp (incl. the 4 Forge canvases), 25 campaign/ff, 1 shared.map,
    /// 1 campaign.map -- measured 2026-09-16 (the folder held 54 files: 52 + 2 user copies).
    #[test]
    fn h4_folder_classifies_all_caches() {
        let Some(folder) = h4_maps_folder() else {
            eprintln!("h4_tests: no halo4/maps folder on this machine -- skipped");
            return;
        };
        let files = h4_map_files(&folder);
        assert!(files.len() >= 52, "expected >= 52 .map files under {}, got {}", folder.display(), files.len());
        // distinct key -> header type
        let mut distinct: BTreeMap<String, i16> = BTreeMap::new();
        for f in &files {
            let c = read_candidate(f).unwrap_or_else(|| panic!("unreadable {}", f.display()));
            assert_eq!(c.game, Game::Halo4, "{}", f.display());
            assert!(!c.modded, "{}", f.display());
            let t = c.h4_type.unwrap_or_else(|| panic!("no type for {}", f.display()));
            // Every playable cache carries a scenario path whose last component is the stem;
            // the two resource caches carry none and fall back to the file stem.
            let file_stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let key = if matches!(t, 0 | 1) {
                // mp/dlc/firefight caches live under `levels\`, the solo campaign under
                // `environments\solo\`.
                assert!(
                    c.scenario_name.starts_with("levels\\") || c.scenario_name.starts_with("environments\\"),
                    "{}: {:?}", f.display(), c.scenario_name
                );
                assert_eq!(Some(c.stem.as_str()), c.scenario_name.rsplit('\\').next(), "{}", f.display());
                c.scenario_name.to_lowercase()
            } else {
                // The resource caches name themselves `maps\<file name>` (hidden from the list
                // by `is_resource_cache` regardless).
                assert_eq!(c.scenario_name, format!("maps\\{}", c.file_name), "{}", f.display());
                assert!(is_resource_cache(&c.file_name), "{}", f.display());
                file_stem.to_lowercase()
            };
            if let Some(prev) = distinct.insert(key, t) {
                assert_eq!(prev, t, "copy with a different type: {}", f.display());
            }
        }
        assert_eq!(distinct.len(), 52, "distinct Halo 4 caches");
        let mut hist: BTreeMap<i16, usize> = BTreeMap::new();
        for t in distinct.values() {
            *hist.entry(*t).or_default() += 1;
        }
        let expect: BTreeMap<i16, usize> = [(0, 25), (1, 25), (3, 1), (4, 1)].into_iter().collect();
        assert_eq!(hist, expect, "Halo 4 header type histogram (distinct caches)");
    }

    /// (b) The Ravine Forge canvas: exact scenario path, stem, mp type, and the .mapinfo id.
    #[test]
    fn h4_ravine_identity() {
        let Some(folder) = h4_maps_folder() else {
            eprintln!("h4_tests: no halo4/maps folder on this machine -- skipped");
            return;
        };
        let p = folder.join("ca_forge_ravine.map");
        if !p.exists() {
            eprintln!("h4_tests: {} missing -- skipped", p.display());
            return;
        }
        let c = read_candidate(&p).expect("read ca_forge_ravine.map");
        assert_eq!(c.game, Game::Halo4);
        assert_eq!(c.h4_type, Some(1));
        assert_eq!(c.scenario_name, "levels\\multi\\ca_forge_ravine\\ca_forge_ravine");
        assert_eq!(c.stem, "ca_forge_ravine");
        assert_eq!(c.file_name, "ca_forge_ravine.map");
        assert!(c.label().ends_with("[Halo 4]"), "{}", c.label());
        // The companion .mapinfo reads through the Reach path (semantics unverified, just present).
        assert_eq!(c.map_id, Some(0x2810), "ca_forge_ravine.mapinfo levl id");
    }

    /// (c) Reach classification is untouched: forge_halo.map stays a Reach map with its stem.
    #[test]
    fn reach_forge_halo_unchanged() {
        let Some(folder) = reach_maps_folder() else {
            eprintln!("h4_tests: no haloreach/maps folder on this machine -- skipped");
            return;
        };
        let p = folder.join("forge_halo.map");
        if !p.exists() {
            eprintln!("h4_tests: {} missing -- skipped", p.display());
            return;
        }
        let c = read_candidate(&p).expect("read forge_halo.map");
        assert_eq!(c.game, Game::Reach);
        assert_eq!(c.h4_type, None);
        assert_eq!(c.stem, "forge_halo");
        assert_eq!(c.file_name, "forge_halo.map");
        assert_eq!(c.label(), "forge_halo.map");
        assert_eq!(c.map_id, Some(3006), "Forge World map id");
    }

    /// Ordering: every Reach candidate precedes every Halo 4 one, Halo 4 alphabetical by label,
    /// and the resource caches never appear.
    #[test]
    fn enumerate_groups_reach_before_h4() {
        if h4_maps_folder().is_none() {
            eprintln!("h4_tests: no halo4/maps folder on this machine -- skipped");
            return;
        }
        let all = enumerate();
        let first_h4 = all.iter().position(|c| c.game == Game::Halo4);
        let last_reach = all.iter().rposition(|c| c.game == Game::Reach);
        if let (Some(h), Some(r)) = (first_h4, last_reach) {
            assert!(r < h, "Reach candidate at {r} after first Halo 4 at {h}");
        }
        let h4: Vec<String> = all.iter().filter(|c| c.game == Game::Halo4).map(|c| c.label().to_lowercase()).collect();
        let mut sorted = h4.clone();
        sorted.sort();
        assert_eq!(h4, sorted, "Halo 4 maps alphabetical");
        assert!(h4.len() >= 50, "expected the 50 playable Halo 4 caches, got {}", h4.len());
        for c in &all {
            assert!(!is_resource_cache(&c.file_name), "{}", c.file_name);
        }
    }

    /// Pure checks that need no game install.
    #[test]
    fn h4_header_helpers() {
        assert!(is_resource_cache("mainmenu.map"));
        assert!(is_resource_cache("SHARED.map"));
        assert!(!is_resource_cache("ca_forge_ravine.map"));
        // A Reach-looking probe: header ok, hash bytes at +152 -> Reach.
        let mut reach = vec![0u8; HEADER_PROBE_LEN];
        reach[..4].copy_from_slice(CACHE_MAGIC);
        reach[4..8].copy_from_slice(&13u32.to_le_bytes());
        reach[152..160].copy_from_slice(&[0x97, 0xdc, 0xf5, 0xee, 0xb9, 0x3c, 0x01, 0x3c]);
        reach[160..180].copy_from_slice(b"Jun 21 2023 15:35:31");
        assert_eq!(classify_header(&reach, Path::new("/x/haloreach/maps/a.map")), (Game::Reach, None));
        // The same header with the Halo 4 build string at +152 -> Halo 4, type from 0x18.
        let mut h4 = vec![0u8; HEADER_PROBE_LEN];
        h4[..4].copy_from_slice(CACHE_MAGIC);
        h4[4..8].copy_from_slice(&13u32.to_le_bytes());
        h4[0x18..0x1A].copy_from_slice(&1i16.to_le_bytes());
        h4[152..152 + HALO4_BUILD.len()].copy_from_slice(HALO4_BUILD.as_bytes());
        assert_eq!(classify_header(&h4, Path::new("/x/somewhere/a.map")), (Game::Halo4, Some(1)));
        // Bad magic never classifies as Halo 4, whatever the folder says.
        let mut junk = h4.clone();
        junk[0] = b'x';
        assert_eq!(classify_header(&junk, Path::new("/x/halo4/maps/a.map")), (Game::Reach, None));
        // A non-Workshop path is a Reach item.
        assert_eq!(workshop_item_game(Path::new("/x/haloreach/maps/a.map")), Some(Game::Reach));
        assert_eq!(Game::Halo4.short_name(), "halo4");
    }
}
