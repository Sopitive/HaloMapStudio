//! Opt-in auto-update against GitHub Releases (`Sopitive/HaloMapStudio`).
//!
//! * **Check** — on startup (a few seconds after the window is up, off the UI thread), every
//!   24 h while running, and on *Check now*. `GET /releases/latest` with a User-Agent, 10 s
//!   timeout; offline / rate-limited = no badge, logged once. The last check time + version are
//!   persisted in `<settings>/updates.json` so a restart does not re-hit the API early.
//! * **Badge** — a small pill next to *Help* ("Update 1.3.2 available"); click = the Update
//!   window, right-click / *Skip this version* = never again for that version (persisted).
//! * **Download** — the asset for the running platform (`HaloMapStudio-<ver>-linux-x64.zip` /
//!   `…-windows-x64.zip`; the older `halo-map-studio-rust-…-<platform>-<date>.zip` names are
//!   accepted as a fallback). A Windows build running under Wine / Proton takes the Windows
//!   asset and says so. Downloaded to `<settings>/updates/`, checked against the API's size and
//!   the release's `.sha256` asset when there is one, then extracted (staged) to
//!   `<install>/.hms-update-<ver>/` — next to the install, so the final step is a same-filesystem
//!   rename, never a copy.
//! * **Install** — the running exe / .so / .dll cannot be overwritten in place (the .so is
//!   mmapped, Windows locks the exe), but both can be *renamed*: the current files are renamed
//!   into `<install>/.hms-backup/` and the staged files renamed into place — a transaction that
//!   rolls back on the first failure. Then the NEW exe is spawned with
//!   `--resume-after-update <ver> --wait-pid <old pid>` (it waits for this process to exit before
//!   it does anything, so the single-instance rule holds) and this process closes. The backup is
//!   kept for one launch (removed on the next start that is not a resume).
//! * **No lost work** — before the restart a recovery snapshot (project record + an exact copy of
//!   the open variant with its unsaved edits) is written to `<settings>/recovery/`; the resumed
//!   process restores it and says so. A recovery file found on any other start is offered.
//!
//! QA override (diagnostic only, documented in RELEASING.md): `HMS_UPDATE_FEED=<file or URL>`
//! points the check at a release JSON of your own; `browser_download_url` may then be a local
//! path, so the whole download → install → restart flow can be exercised without a release.

use eframe::egui;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const REPO: &str = "Sopitive/HaloMapStudio";
pub const API_LATEST: &str = "https://api.github.com/repos/Sopitive/HaloMapStudio/releases/latest";
pub const RELEASES_PAGE: &str = "https://github.com/Sopitive/HaloMapStudio/releases";
/// How often a running app re-checks (and the minimum gap between checks across restarts).
pub const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// First automatic check waits this long after startup so it never competes with the map load.
const STARTUP_DELAY: Duration = Duration::from_secs(4);
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
pub const QA_FEED_ENV: &str = "HMS_UPDATE_FEED";
pub const BACKUP_DIR: &str = ".hms-backup";
pub const STAGING_PREFIX: &str = ".hms-update-";

// ---------------------------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------------------------

/// A semantic version: `1.3.1`, `v1.3.1`, `1.3.1-pre`. Pre-release tags sort BELOW the release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: String,
}

impl Version {
    pub fn parse(s: &str) -> Option<Version> {
        let s = s.trim();
        let s = s.strip_prefix('v').or_else(|| s.strip_prefix('V')).unwrap_or(s);
        let (core, pre) = match s.split_once('-') {
            Some((c, p)) => (c, p.to_string()),
            None => (s, String::new()),
        };
        let core = core.split_once('+').map(|(c, _)| c).unwrap_or(core);
        let mut it = core.split('.');
        let major = it.next()?.parse().ok()?;
        let minor = it.next().unwrap_or("0").parse().ok()?;
        let patch = it.next().unwrap_or("0").parse().ok()?;
        if it.next().is_some() { return None; }
        Some(Version { major, minor, patch, pre })
    }
    pub fn current() -> Version {
        Version::parse(env!("CARGO_PKG_VERSION")).expect("CARGO_PKG_VERSION is semver")
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() { write!(f, "-{}", self.pre)?; }
        Ok(())
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for Version {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(o.major, o.minor, o.patch)).then_with(|| {
            match (self.pre.is_empty(), o.pre.is_empty()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // release > pre-release
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => self.pre.cmp(&o.pre),
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Platform + assets
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform { LinuxX64, WindowsX64 }

impl Platform {
    pub fn running() -> Platform {
        if cfg!(windows) { Platform::WindowsX64 } else { Platform::LinuxX64 }
    }
    /// The token in the asset name.
    pub fn token(self) -> &'static str {
        match self { Platform::LinuxX64 => "linux-x64", Platform::WindowsX64 => "windows-x64" }
    }
    pub fn exe_name(self) -> &'static str {
        match self { Platform::LinuxX64 => "hms-app", Platform::WindowsX64 => "hms-app.exe" }
    }
    /// The release-asset naming contract (see RELEASING.md).
    pub fn asset_name(self, v: &Version) -> String {
        format!("HaloMapStudio-{v}-{}.zip", self.token())
    }
}

/// Is this (Windows) build running under Wine / Proton? ntdll exports `wine_get_version` there.
#[cfg(windows)]
pub fn running_under_wine() -> bool {
    use windows::core::s;
    use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    unsafe {
        match GetModuleHandleA(s!("ntdll.dll")) {
            Ok(h) => GetProcAddress(h, s!("wine_get_version")).is_some(),
            Err(_) => false,
        }
    }
}
#[cfg(not(windows))]
pub fn running_under_wine() -> bool { false }

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Asset {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(rename = "browser_download_url", default)]
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct Release {
    pub tag: String,
    pub version: Version,
    pub name: String,
    pub body: String,
    pub html_url: String,
    pub prerelease: bool,
    pub assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct ReleaseJson {
    tag_name: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

/// Parse a GitHub `releases/latest` (or single release) JSON document.
pub fn parse_release_json(text: &str) -> Result<Release, String> {
    let j: ReleaseJson = serde_json::from_str(text).map_err(|e| format!("release JSON: {e}"))?;
    let version = Version::parse(&j.tag_name).ok_or_else(|| format!("release tag '{}' is not a version", j.tag_name))?;
    Ok(Release {
        tag: j.tag_name,
        version,
        name: j.name,
        body: j.body.unwrap_or_default(),
        html_url: if j.html_url.is_empty() { RELEASES_PAGE.into() } else { j.html_url },
        prerelease: j.prerelease,
        assets: j.assets,
    })
}

/// The archive for `plat`: the contract name first, then any `.zip` carrying the platform token
/// (the pre-1.3 `halo-map-studio-rust-…-<platform>-<date>.zip` names), never a portable / checksum file.
pub fn pick_asset(rel: &Release, plat: Platform) -> Option<&Asset> {
    let want = rel.assets.iter().find(|a| a.name.eq_ignore_ascii_case(&plat.asset_name(&rel.version)));
    want.or_else(|| {
        rel.assets.iter().find(|a| {
            let n = a.name.to_ascii_lowercase();
            n.ends_with(".zip") && n.contains(plat.token()) && !n.contains("portable")
        })
    })
}

/// The checksum companion of `asset`: `<asset>.sha256`, else a combined `SHA256SUMS` file
/// (whose text is then searched for the asset's name). Another asset's `.sha256` is never used.
pub fn pick_checksum_asset<'a>(rel: &'a Release, asset: &Asset) -> Option<&'a Asset> {
    let exact = format!("{}.sha256", asset.name);
    rel.assets.iter().find(|a| a.name.eq_ignore_ascii_case(&exact)).or_else(|| {
        rel.assets.iter().find(|a| {
            let n = a.name.to_ascii_lowercase();
            n == "sha256sums" || n == "sha256sums.txt"
        })
    })
}

/// Read the hex digest for `asset_name` out of a `sha256sum`-style text (`<hex>  <name>` lines;
/// a bare `<hex>` line is accepted when there is only one).
pub fn parse_sha256_text(text: &str, asset_name: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let is_hex = |s: &str| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit());
    for l in &lines {
        let mut it = l.split_whitespace();
        let (Some(h), Some(n)) = (it.next(), it.next()) else { continue };
        let n = n.trim_start_matches('*');
        if is_hex(h) && n.eq_ignore_ascii_case(asset_name) { return Some(h.to_ascii_lowercase()); }
    }
    if lines.len() == 1 {
        let mut it = lines[0].split_whitespace();
        let h = it.next()?;
        if is_hex(h) && it.next().is_none() { return Some(h.to_ascii_lowercase()); }
    }
    None
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------------------------
// Persistent preferences
// ---------------------------------------------------------------------------------------------

fn default_true() -> bool { true }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UpdatePrefs {
    #[serde(default = "default_true")]
    pub auto_check: bool,
    #[serde(default)]
    pub last_check_unix: u64,
    #[serde(default)]
    pub last_seen_version: String,
    #[serde(default)]
    pub dismissed_version: String,
    /// One line about the last check (for the Settings section).
    #[serde(default)]
    pub last_result: String,
}

impl Default for UpdatePrefs {
    fn default() -> Self {
        UpdatePrefs { auto_check: true, last_check_unix: 0, last_seen_version: String::new(), dismissed_version: String::new(), last_result: String::new() }
    }
}

pub fn prefs_path() -> Option<PathBuf> { crate::project::settings_dir().map(|d| d.join("updates.json")) }

pub fn load_prefs() -> UpdatePrefs {
    prefs_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_prefs(p: &UpdatePrefs) {
    if let Some(path) = prefs_path() {
        if let Some(d) = path.parent() { let _ = std::fs::create_dir_all(d); }
        if let Ok(s) = serde_json::to_string_pretty(p) { let _ = std::fs::write(path, s); }
    }
}

pub fn now_unix() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// "3 minutes ago" / "2 days ago" for the Settings section (no chrono dependency).
pub fn ago(unix: u64) -> String {
    if unix == 0 { return "never".into(); }
    let d = now_unix().saturating_sub(unix);
    match d {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", d / 60),
        3600..=86399 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86400),
    }
}

// ---------------------------------------------------------------------------------------------
// HTTP (background threads only)
// ---------------------------------------------------------------------------------------------

fn user_agent() -> String { format!("HaloMapStudio/{} (+https://github.com/{})", Version::current(), REPO) }

/// A QA feed / download source that is a local file (`file://…` or a plain path).
fn local_path_of(url: &str) -> Option<PathBuf> {
    if let Some(p) = url.strip_prefix("file://") { return Some(PathBuf::from(p)); }
    if !url.contains("://") { return Some(PathBuf::from(url)); }
    None
}

#[cfg(not(windows))]
fn agent() -> Result<ureq::Agent, String> {
    Ok(ureq::AgentBuilder::new().timeout(HTTP_TIMEOUT).user_agent(&user_agent()).build())
}
/// Windows: the platform TLS (schannel through `native-tls`) — no OpenSSL, no `ring` assembly
/// (the msvc cross build has no nasm). Wine / Proton implement schannel too.
#[cfg(windows)]
fn agent() -> Result<ureq::Agent, String> {
    let tls = native_tls::TlsConnector::new().map_err(|e| format!("TLS: {e}"))?;
    Ok(ureq::AgentBuilder::new().timeout(HTTP_TIMEOUT).user_agent(&user_agent()).tls_connector(std::sync::Arc::new(tls)).build())
}

fn http_get_text(url: &str) -> Result<String, String> {
    if let Some(p) = local_path_of(url) {
        return std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()));
    }
    let resp = agent()?.get(url).set("Accept", "application/vnd.github+json").call().map_err(|e| match e {
        ureq::Error::Status(403, _) | ureq::Error::Status(429, _) => "GitHub API rate limit reached; try again later".to_string(),
        ureq::Error::Status(404, _) => "no release found".to_string(),
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        ureq::Error::Transport(t) => format!("network: {t}"),
    })?;
    resp.into_string().map_err(|e| format!("read: {e}"))
}

/// Stream `url` to `dest` (through `dest.part`, renamed when complete) reporting (received, total).
fn http_download(url: &str, dest: &Path, expected: u64, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    if let Some(d) = dest.parent() { std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?; }
    let part = dest.with_extension("zip.part");
    if let Some(p) = local_path_of(url) {
        std::fs::copy(&p, &part).map_err(|e| format!("{}: {e}", p.display()))?;
        let n = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        progress(n, n);
    } else {
        let resp = agent()?.get(url).call().map_err(|e| format!("download: {e}"))?;
        let total = resp.header("Content-Length").and_then(|s| s.parse::<u64>().ok()).unwrap_or(expected);
        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut got = 0u64;
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf).map_err(|e| format!("download: {e}"))?;
            if n == 0 { break; }
            std::io::Write::write_all(&mut file, &buf[..n]).map_err(|e| format!("write: {e}"))?;
            got += n as u64;
            progress(got, total);
        }
    }
    std::fs::rename(&part, dest).map_err(|e| format!("rename {}: {e}", part.display()))
}

// ---------------------------------------------------------------------------------------------
// Minimal zip reader (stored + deflate, no zip64) over miniz_oxide — the release archives.
// ---------------------------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; }
        table[i as usize] = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data { crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8); }
    !crc
}

#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub crc: u32,
    pub csize: u64,
    pub usize: u64,
    pub unix_mode: u32,
    pub data_offset: u64,
}

fn u16_at(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) }
fn u32_at(b: &[u8], o: usize) -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }

/// The central directory of a zip (validates the signatures; returns the entries in order).
pub fn zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry>, String> {
    if bytes.len() < 22 { return Err("not a zip file (too short)".into()); }
    let min = bytes.len().saturating_sub(22 + 65535);
    let eocd = (min..=bytes.len() - 22).rev().find(|&i| u32_at(bytes, i) == 0x0605_4B50).ok_or("not a zip file (no end-of-central-directory)")?;
    let n = u16_at(bytes, eocd + 10) as usize;
    let cd_size = u32_at(bytes, eocd + 12) as usize;
    let cd_off = u32_at(bytes, eocd + 16) as usize;
    if cd_off.checked_add(cd_size).map_or(true, |e| e > bytes.len()) { return Err("zip central directory out of range".into()); }
    let mut out = Vec::with_capacity(n);
    let mut p = cd_off;
    for _ in 0..n {
        if p + 46 > bytes.len() || u32_at(bytes, p) != 0x0201_4B50 { return Err("bad central directory entry".into()); }
        let method = u16_at(bytes, p + 10);
        let crc = u32_at(bytes, p + 16);
        let csize = u32_at(bytes, p + 20) as u64;
        let usize_ = u32_at(bytes, p + 24) as u64;
        let nlen = u16_at(bytes, p + 28) as usize;
        let xlen = u16_at(bytes, p + 30) as usize;
        let clen = u16_at(bytes, p + 32) as usize;
        let ext_attr = u32_at(bytes, p + 38);
        let lho = u32_at(bytes, p + 42) as usize;
        if p + 46 + nlen > bytes.len() { return Err("bad central directory name".into()); }
        let name = String::from_utf8_lossy(&bytes[p + 46..p + 46 + nlen]).into_owned();
        if lho + 30 > bytes.len() || u32_at(bytes, lho) != 0x0403_4B50 { return Err(format!("bad local header for {name}")); }
        let lnlen = u16_at(bytes, lho + 26) as usize;
        let lxlen = u16_at(bytes, lho + 28) as usize;
        let data_offset = (lho + 30 + lnlen + lxlen) as u64;
        if data_offset + csize > bytes.len() as u64 { return Err(format!("truncated data for {name}")); }
        out.push(ZipEntry { name, method, crc, csize, usize: usize_, unix_mode: ext_attr >> 16, data_offset });
        p += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

fn zip_read(bytes: &[u8], e: &ZipEntry) -> Result<Vec<u8>, String> {
    let raw = &bytes[e.data_offset as usize..(e.data_offset + e.csize) as usize];
    let data = match e.method {
        0 => raw.to_vec(),
        8 => miniz_oxide::inflate::decompress_to_vec(raw).map_err(|err| format!("{}: inflate failed ({err:?})", e.name))?,
        m => return Err(format!("{}: unsupported compression method {m}", e.name)),
    };
    if data.len() as u64 != e.usize { return Err(format!("{}: size mismatch", e.name)); }
    if crc32(&data) != e.crc { return Err(format!("{}: CRC mismatch (corrupt download)", e.name)); }
    Ok(data)
}

/// A safe relative path for a zip entry, with the archive's single top-level folder stripped.
fn entry_rel_path(name: &str, strip: Option<&str>) -> Option<PathBuf> {
    let n = name.replace('\\', "/");
    let n = match strip { Some(s) => n.strip_prefix(s)?.trim_start_matches('/').to_string(), None => n };
    if n.is_empty() { return None; }
    let mut out = PathBuf::new();
    for c in n.split('/') {
        if c.is_empty() || c == "." { continue; }
        if c == ".." || c.contains(':') { return None; }
        out.push(c);
    }
    if out.as_os_str().is_empty() { None } else { Some(out) }
}

/// Extract every file of the archive under `dest` (created), stripping a single common top-level
/// folder (`HaloMapStudio-1.3.1-linux-x64/…`, `linux-x64/…`). Returns the relative paths written.
pub fn extract_zip(bytes: &[u8], dest: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = zip_entries(bytes)?;
    if entries.is_empty() { return Err("empty archive".into()); }
    // common top-level dir?
    let tops: std::collections::BTreeSet<String> = entries.iter().filter_map(|e| e.name.replace('\\', "/").split('/').next().map(str::to_string)).collect();
    let strip: Option<String> = if tops.len() == 1 && entries.iter().all(|e| e.name.replace('\\', "/").contains('/')) { tops.into_iter().next() } else { None };
    std::fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut written = Vec::new();
    for e in &entries {
        if e.name.ends_with('/') { continue; }
        let Some(rel) = entry_rel_path(&e.name, strip.as_deref()) else { return Err(format!("unsafe path in archive: {}", e.name)) };
        let data = zip_read(bytes, e)?;
        let path = dest.join(&rel);
        if let Some(d) = path.parent() { std::fs::create_dir_all(d).map_err(|err| format!("{}: {err}", d.display()))?; }
        std::fs::write(&path, &data).map_err(|err| format!("{}: {err}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let exec = e.unix_mode & 0o111 != 0 || rel.file_name().map_or(false, |f| f == "hms-app");
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(if exec { 0o755 } else { 0o644 }));
        }
        written.push(rel);
    }
    if written.is_empty() { return Err("archive holds no files".into()); }
    Ok(written)
}

// ---------------------------------------------------------------------------------------------
// Staging + install transaction
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Staged {
    pub version: Version,
    pub install_dir: PathBuf,
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
}

/// Strip the Windows **extended-length** (`\\?\`) prefix that `fs::canonicalize` always adds.
///
/// Rust's `canonicalize` returns a *verbatim* path (`\\?\C:\Program Files\HaloMapStudio`). The
/// Win32 file APIs accept it, but `CreateProcess`'s `lpCurrentDirectory` does **not** — and
/// neither does Wine's — so handing it to `Command::current_dir` makes the relaunch after an
/// update fail with "The directory name is invalid" and the user is left with no app running.
/// Every path that can reach a process spawn (or a message shown to the user) goes through here.
/// `\\?\UNC\server\share` folds back to `\\server\share`. No-op on Linux.
pub fn plain_path(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return match rest.strip_prefix("UNC\\").or_else(|| rest.strip_prefix("unc\\")) {
            Some(unc) => PathBuf::from(format!(r"\\{unc}")),
            // A bare `\\?\C:` (no trailing separator) is not a usable directory; keep the root.
            None if rest.len() == 2 && rest.ends_with(':') => PathBuf::from(format!("{rest}\\")),
            None => PathBuf::from(rest),
        };
    }
    p.to_path_buf()
}

/// Where the running program lives (the exe's directory), as a path Win32 will accept
/// (see [`plain_path`] — a `\\?\` working directory breaks the post-update relaunch).
pub fn install_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;
    let exe = plain_path(&std::fs::canonicalize(&exe).unwrap_or(exe));
    exe.parent().map(|p| p.to_path_buf()).ok_or_else(|| "exe has no parent directory".into())
}

/// Extract the verified archive next to the install (`<install>/.hms-update-<ver>/`) and make sure
/// it holds the program. A read-only install dir fails here, before anything is touched.
pub fn stage(zip_path: &Path, install_dir: &Path, version: &Version, plat: Platform) -> Result<Staged, String> {
    let bytes = std::fs::read(zip_path).map_err(|e| format!("{}: {e}", zip_path.display()))?;
    let dir = install_dir.join(format!("{STAGING_PREFIX}{version}"));
    if dir.exists() { let _ = std::fs::remove_dir_all(&dir); }
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot write to the install folder {} ({e})", install_dir.display()))?;
    let files = extract_zip(&bytes, &dir).map_err(|e| { let _ = std::fs::remove_dir_all(&dir); e })?;
    if !files.iter().any(|f| f.file_name().map_or(false, |n| n == plat.exe_name())) {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!("the archive does not contain {}", plat.exe_name()));
    }
    Ok(Staged { version: version.clone(), install_dir: install_dir.to_path_buf(), dir, files })
}

/// The install transaction: every current file that the archive replaces is RENAMED into
/// `<install>/.hms-backup/`, every staged file is RENAMED into place; the first failure undoes
/// everything already moved. Renames (not copies) so the running exe / mapped library are never
/// written to, and so the step is all-or-nothing on one filesystem.
pub fn apply(staged: &Staged) -> Result<PathBuf, String> {
    let backup = staged.install_dir.join(BACKUP_DIR);
    if backup.exists() { std::fs::remove_dir_all(&backup).map_err(|e| format!("cannot clear {}: {e}", backup.display()))?; }
    std::fs::create_dir_all(&backup).map_err(|e| format!("{}: {e}", backup.display()))?;
    let mut backed: Vec<PathBuf> = Vec::new();
    let mut moved: Vec<PathBuf> = Vec::new();
    let rollback = |backed: &[PathBuf], moved: &[PathBuf]| {
        for f in moved { let _ = std::fs::rename(staged.install_dir.join(f), staged.dir.join(f)); }
        for f in backed { let _ = std::fs::rename(backup.join(f), staged.install_dir.join(f)); }
    };
    for f in &staged.files {
        let dst = staged.install_dir.join(f);
        if dst.exists() {
            let b = backup.join(f);
            if let Some(d) = b.parent() { let _ = std::fs::create_dir_all(d); }
            if let Err(e) = std::fs::rename(&dst, &b) {
                rollback(&backed, &moved);
                return Err(format!("cannot move {} aside: {e}", dst.display()));
            }
            backed.push(f.clone());
        }
        if let Some(d) = dst.parent() { let _ = std::fs::create_dir_all(d); }
        if let Err(e) = std::fs::rename(staged.dir.join(f), &dst) {
            rollback(&backed, &moved);
            return Err(format!("cannot install {}: {e}", dst.display()));
        }
        moved.push(f.clone());
    }
    let _ = std::fs::remove_dir_all(&staged.dir);
    Ok(backup)
}

/// Start the NEW program: it waits for this process to exit (single instance) and then restores
/// the recovery snapshot.
pub fn spawn_resumed(install_dir: &Path, version: &Version) -> Result<(), String> {
    // Both the image path and the working directory must be free of the `\\?\` prefix: Windows
    // and Wine reject a verbatim `lpCurrentDirectory` outright (see `plain_path`).
    let dir = plain_path(install_dir);
    let exe = dir.join(Platform::running().exe_name());
    let build = |with_cwd: bool| {
        let mut cmd = std::process::Command::new(&exe);
        if with_cwd { cmd.current_dir(&dir); }
        cmd.arg("--resume-after-update").arg(version.to_string())
            .arg("--wait-pid").arg(std::process::id().to_string())
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
        }
        cmd
    };
    match build(true).spawn() {
        Ok(_) => Ok(()),
        // Losing the working directory is survivable (the app resolves its library next to the
        // exe); not restarting at all is not. Retry without it rather than stranding the user.
        Err(e) => {
            log::warn!("updater: relaunch with cwd {} failed ({e}); retrying without a working directory", dir.display());
            build(false).spawn().map(|_| ()).map_err(|e2| format!("cannot start {}: {e2}", exe.display()))
        }
    }
}

/// Is `pid` still alive?
pub fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    { Path::new(&format!("/proc/{pid}")).exists() }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
        unsafe {
            match OpenProcess(PROCESS_SYNCHRONIZE, false, pid) {
                Ok(h) => { let w = WaitForSingleObject(h, 0); let _ = CloseHandle(h); w.0 == 0x0000_0102 /* WAIT_TIMEOUT */ }
                Err(_) => false,
            }
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    { let _ = pid; false }
}

/// `--wait-pid <n>`: block (up to `timeout`) until that process is gone. Called first thing in
/// `main` by the resumed process, before any window or server is opened.
pub fn handle_wait_pid_arg(args: &[String], timeout: Duration) {
    let Some(p) = args.iter().position(|a| a == "--wait-pid") else { return };
    let Some(pid) = args.get(p + 1).and_then(|s| s.parse::<u32>().ok()) else { return };
    let t0 = Instant::now();
    while pid_alive(pid) && t0.elapsed() < timeout { std::thread::sleep(Duration::from_millis(100)); }
}

/// `--resume-after-update <ver>` → the version just installed.
pub fn resume_arg(args: &[String]) -> Option<String> {
    let p = args.iter().position(|a| a == "--resume-after-update")?;
    Some(args.get(p + 1).cloned().unwrap_or_default())
}

/// Housekeeping at startup: stale staging folders always go; the previous version's backup is
/// kept through the resumed launch and removed on the next ordinary start.
pub fn startup_cleanup(install_dir: &Path, resumed: bool) {
    if let Ok(rd) = std::fs::read_dir(install_dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with(STAGING_PREFIX) { let _ = std::fs::remove_dir_all(e.path()); }
        }
    }
    if !resumed {
        let b = install_dir.join(BACKUP_DIR);
        if b.exists() { let _ = std::fs::remove_dir_all(b); }
    }
}

pub fn open_folder(dir: &Path) {
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer").arg(dir).spawn();
    #[cfg(not(windows))]
    let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
}

// ---------------------------------------------------------------------------------------------
// Recovery snapshots
// ---------------------------------------------------------------------------------------------

/// Sidecar of a `<stamp>.mmsproj` in `<settings>/recovery/`: what the project record cannot say.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RecoveryMeta {
    pub from_version: String,
    pub to_version: String,
    pub created_unix: u64,
    /// The project record (`<stamp>.mmsproj`), a file name in the same folder.
    pub project_file: String,
    /// The variant the user had open (their real file; never written by the snapshot).
    #[serde(default)]
    pub variant_original: Option<String>,
    /// An exact copy of the open variant INCLUDING unsaved edits (`<stamp>.mvar`, same folder).
    #[serde(default)]
    pub variant_copy: Option<String>,
    /// A new, never-saved variant was open (the copy carries it; there is no original).
    #[serde(default)]
    pub variant_was_new: bool,
    #[serde(default)]
    pub show_settings: bool,
    #[serde(default)]
    pub show_hotkeys: bool,
    #[serde(default)]
    pub show_script: bool,
}

pub fn recovery_dir() -> Option<PathBuf> { crate::project::settings_dir().map(|d| d.join("recovery")) }

/// Write `<stamp>.recovery.json` + `<stamp>.mmsproj` (+ the caller-written `<stamp>.mvar`).
/// Returns the folder and the stamp; the caller writes the variant copy as `<dir>/<stamp>.mvar`
/// BEFORE calling this so an incomplete snapshot is never advertised.
pub fn recovery_stamp() -> String {
    let t = now_unix();
    format!("{t}-{}", std::process::id())
}

pub fn write_recovery(dir: &Path, stamp: &str, meta: &mut RecoveryMeta, project: &crate::project::Project) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    meta.project_file = format!("{stamp}.mmsproj");
    meta.created_unix = now_unix();
    crate::project::save_to(project, &dir.join(&meta.project_file)).map_err(|e| format!("project: {e}"))?;
    let path = dir.join(format!("{stamp}.recovery.json"));
    let json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

pub fn load_recovery(meta_path: &Path) -> Result<(RecoveryMeta, crate::project::Project), String> {
    let text = std::fs::read_to_string(meta_path).map_err(|e| format!("{}: {e}", meta_path.display()))?;
    let meta: RecoveryMeta = serde_json::from_str(&text).map_err(|e| format!("recovery meta: {e}"))?;
    let dir = meta_path.parent().unwrap_or(Path::new("."));
    let project = crate::project::load_from(&dir.join(&meta.project_file)).map_err(|e| format!("recovery project: {e}"))?;
    Ok((meta, project))
}

/// The newest `*.recovery.json` in the recovery folder.
pub fn find_latest_recovery(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if !p.to_string_lossy().ends_with(".recovery.json") { continue; }
        let t = e.metadata().and_then(|m| m.modified()).ok().and_then(|m| m.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
        if best.as_ref().map_or(true, |(bt, _)| t >= *bt) { best = Some((t, p)); }
    }
    best.map(|(_, p)| p)
}

/// Remove one snapshot (meta + project + variant copy).
pub fn discard_recovery(meta_path: &Path) {
    if let Ok((meta, _)) = load_recovery(meta_path) {
        let dir = meta_path.parent().unwrap_or(Path::new("."));
        let _ = std::fs::remove_file(dir.join(&meta.project_file));
        if let Some(v) = &meta.variant_copy { let _ = std::fs::remove_file(dir.join(v)); }
    }
    let _ = std::fs::remove_file(meta_path);
}

// ---------------------------------------------------------------------------------------------
// Runtime controller (owned by the App; all network / disk work on background threads)
// ---------------------------------------------------------------------------------------------

enum Msg {
    Checked(Result<Release, String>),
    Progress(u64, u64),
    Downloaded(Result<(PathBuf, u64), String>),
    Staged(Result<Staged, String>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CheckPhase { Idle, Checking, UpToDate, Available, Failed(String) }

#[derive(Clone, Debug, PartialEq)]
pub enum DownloadPhase { NotStarted, Running { got: u64, total: u64 }, Staging, Ready, Failed(String) }

/// What the App must do to finish an install (it owns the session state the snapshot needs).
pub struct InstallRequest { pub staged: Staged }

pub struct Updater {
    pub prefs: UpdatePrefs,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    pub check: CheckPhase,
    pub latest: Option<Release>,
    pub show_window: bool,
    pub download: DownloadPhase,
    pub downloaded_zip: Option<PathBuf>,
    pub staged: Option<Staged>,
    /// "Install automatically when the download completes" (off by default).
    pub install_auto: bool,
    pub install_error: Option<String>,
    /// Scripted "install now" (the `dialog update install` QA verb); consumed by the App tick.
    pub install_requested: bool,
    next_check: Option<Instant>,
    /// Set by `--resume-after-update`; `resume_recovery` is the snapshot to restore.
    pub resumed_version: Option<String>,
    pub resume_recovery: Option<PathBuf>,
    /// A snapshot found on an ORDINARY start (offered, not forced).
    pub offered_recovery: Option<PathBuf>,
    /// The "Updated to X — your session was restored" banner (until this instant — long enough to outlast the map reload — or until its × is clicked).
    pub banner: Option<(String, Instant)>,
    logged_failure: bool,
}

impl Updater {
    pub fn new(args: &[String]) -> Self {
        let prefs = load_prefs();
        let (tx, rx) = channel();
        let resumed_version = resume_arg(args);
        let rec_dir = recovery_dir();
        let latest_rec = rec_dir.as_deref().and_then(find_latest_recovery);
        let (resume_recovery, offered_recovery) = match (&resumed_version, latest_rec) {
            (Some(_), Some(p)) => (Some(p), None),
            (None, Some(p)) => (None, Some(p)),
            _ => (None, None),
        };
        if let Ok(dir) = install_dir() { startup_cleanup(&dir, resumed_version.is_some()); }
        let next_check = if prefs.auto_check {
            let due = prefs.last_check_unix.saturating_add(CHECK_INTERVAL.as_secs());
            let wait = if std::env::var_os(QA_FEED_ENV).is_some() { 0 } else { due.saturating_sub(now_unix()) };
            Some(Instant::now() + STARTUP_DELAY + Duration::from_secs(wait))
        } else { None };
        let banner = resumed_version.as_ref().map(|v| (format!("Updated to {} — your session was restored.", if v.is_empty() { Version::current().to_string() } else { v.clone() }), Instant::now() + Duration::from_secs(90)));
        Updater {
            prefs, tx, rx,
            check: CheckPhase::Idle, latest: None, show_window: false,
            download: DownloadPhase::NotStarted, downloaded_zip: None, staged: None, install_auto: false, install_error: None, install_requested: false,
            next_check, resumed_version, resume_recovery, offered_recovery, banner, logged_failure: false,
        }
    }

    /// Newer than the running program?
    pub fn available(&self) -> Option<&Release> {
        self.latest.as_ref().filter(|r| r.version > Version::current())
    }

    /// The badge text, or None (nothing newer / dismissed for this version).
    pub fn badge(&self) -> Option<String> {
        let r = self.available()?;
        if self.prefs.dismissed_version == r.version.to_string() { return None; }
        Some(format!("Update {} available", r.version))
    }

    pub fn dismiss_current(&mut self) {
        if let Some(r) = self.available() {
            self.prefs.dismissed_version = r.version.to_string();
            save_prefs(&self.prefs);
        }
    }

    /// Start a check on a background thread (no-op while one runs).
    pub fn check_now(&mut self) {
        if self.check == CheckPhase::Checking { return; }
        self.check = CheckPhase::Checking;
        self.next_check = None;
        let tx = self.tx.clone();
        let url = std::env::var(QA_FEED_ENV).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| API_LATEST.to_string());
        std::thread::Builder::new().name("hms-update-check".into()).spawn(move || {
            let r = http_get_text(&url).and_then(|t| parse_release_json(&t));
            let _ = tx.send(Msg::Checked(r));
        }).ok();
    }

    pub fn set_auto_check(&mut self, on: bool) {
        self.prefs.auto_check = on;
        save_prefs(&self.prefs);
        self.next_check = if on { Some(Instant::now() + Duration::from_secs(self.prefs.last_check_unix.saturating_add(CHECK_INTERVAL.as_secs()).saturating_sub(now_unix()))) } else { None };
    }

    /// Download the platform asset (background), verify it, stage it next to the install.
    pub fn start_download(&mut self) {
        let Some(rel) = self.available().cloned() else { return };
        let plat = Platform::running();
        let Some(asset) = pick_asset(&rel, plat).cloned() else {
            self.download = DownloadPhase::Failed(format!("this release has no {} archive", plat.token()));
            return;
        };
        let sum_asset = pick_checksum_asset(&rel, &asset).cloned();
        let Some(dir) = crate::project::settings_dir().map(|d| d.join("updates")) else {
            self.download = DownloadPhase::Failed("no settings directory".into());
            return;
        };
        let dest = dir.join(&asset.name);
        let version = rel.version.clone();
        self.download = DownloadPhase::Running { got: 0, total: asset.size };
        self.install_error = None;
        let tx = self.tx.clone();
        std::thread::Builder::new().name("hms-update-download".into()).spawn(move || {
            let tx2 = tx.clone();
            let r = http_download(&asset.url, &dest, asset.size, move |g, t| { let _ = tx2.send(Msg::Progress(g, t)); })
                .and_then(|()| {
                    let bytes = std::fs::read(&dest).map_err(|e| e.to_string())?;
                    if asset.size > 0 && bytes.len() as u64 != asset.size {
                        return Err(format!("size mismatch: got {} bytes, the release says {}", bytes.len(), asset.size));
                    }
                    if let Some(sa) = &sum_asset {
                        let text = http_get_text(&sa.url)?;
                        match parse_sha256_text(&text, &asset.name) {
                            Some(want) => {
                                let have = sha256_hex(&bytes);
                                if have != want { return Err(format!("SHA-256 mismatch (expected {want}, got {have})")); }
                            }
                            None => log::warn!("updater: {} lists no digest for {}", sa.name, asset.name),
                        }
                    }
                    Ok((dest.clone(), bytes.len() as u64))
                });
            let ok = r.is_ok();
            let _ = tx.send(Msg::Downloaded(r));
            if ok {
                let st = install_dir().and_then(|inst| stage(&dest, &inst, &version, plat));
                let _ = tx.send(Msg::Staged(st));
            }
        }).ok();
    }

    /// Drain background results + fire scheduled checks. Returns true when the UI should repaint soon.
    pub fn tick(&mut self) -> bool {
        let mut busy = false;
        while let Ok(m) = self.rx.try_recv() {
            busy = true;
            match m {
                Msg::Checked(Ok(rel)) => {
                    self.prefs.last_check_unix = now_unix();
                    self.prefs.last_seen_version = rel.version.to_string();
                    let newer = rel.version > Version::current();
                    self.prefs.last_result = if newer { format!("{} is available", rel.version) } else { format!("up to date ({} is the latest)", rel.version) };
                    save_prefs(&self.prefs);
                    self.check = if newer { CheckPhase::Available } else { CheckPhase::UpToDate };
                    if self.latest.as_ref().map(|l| l.version != rel.version).unwrap_or(true) {
                        self.download = DownloadPhase::NotStarted;
                        self.staged = None;
                        self.downloaded_zip = None;
                    }
                    self.latest = Some(rel);
                    self.next_check = self.prefs.auto_check.then(|| Instant::now() + CHECK_INTERVAL);
                }
                Msg::Checked(Err(e)) => {
                    if !self.logged_failure { log::warn!("updater: check failed: {e}"); self.logged_failure = true; }
                    self.prefs.last_result = format!("check failed: {e}");
                    self.check = CheckPhase::Failed(e);
                    // retry in an hour, quietly
                    self.next_check = self.prefs.auto_check.then(|| Instant::now() + Duration::from_secs(3600));
                }
                Msg::Progress(g, t) => { if matches!(self.download, DownloadPhase::Running { .. }) { self.download = DownloadPhase::Running { got: g, total: t }; } }
                Msg::Downloaded(Ok((p, _))) => { self.downloaded_zip = Some(p); self.download = DownloadPhase::Staging; }
                Msg::Downloaded(Err(e)) => { self.download = DownloadPhase::Failed(e); }
                Msg::Staged(Ok(s)) => { self.staged = Some(s); self.download = DownloadPhase::Ready; }
                Msg::Staged(Err(e)) => { self.download = DownloadPhase::Failed(format!("{e}\nThe archive is in the download folder; close the app and unzip it over the install folder by hand.")); }
            }
        }
        if let Some(t) = self.next_check { if Instant::now() >= t { self.check_now(); busy = true; } }
        if let Some((_, until)) = self.banner { if Instant::now() >= until { self.banner = None; } else { busy = true; } }
        busy || matches!(self.download, DownloadPhase::Running { .. } | DownloadPhase::Staging) || self.check == CheckPhase::Checking
    }

    /// The pending install: the scripted `dialog update install`, or the "install automatically
    /// when the download completes" box once the archive is staged. Checked every frame, so a
    /// closed Update window does not strand an auto-install. An install that FAILED is not retried
    /// (both flags are consumed), so a broken install folder cannot loop.
    pub fn take_install_request(&mut self) -> Option<InstallRequest> {
        if !(self.install_requested || self.install_auto) { return None; }
        if self.download != DownloadPhase::Ready || self.install_error.is_some() { return None; }
        self.install_requested = false;
        self.install_auto = false;
        self.staged.clone().map(|staged| InstallRequest { staged })
    }

    /// One line of state for scripts / logs.
    pub fn status_line(&self) -> String {
        format!(
            "update: installed {} · check {:?} · latest {} · badge {:?} · download {:?} · staged {} · window {} · auto-install {}{}",
            Version::current(), self.check,
            self.latest.as_ref().map(|r| r.version.to_string()).unwrap_or_else(|| "-".into()),
            self.badge(), self.download,
            self.staged.as_ref().map(|s| s.dir.display().to_string()).unwrap_or_else(|| "-".into()),
            if self.show_window { "open" } else { "closed" }, self.install_auto,
            self.install_error.as_ref().map(|e| format!(" · install error: {e}")).unwrap_or_default(),
        )
    }

    // ------------------------------------------------------------------ UI

    /// The menu-bar pill. Click opens the window; right-click dismisses for this version.
    pub fn ui_badge(&mut self, ui: &mut egui::Ui) {
        let Some(text) = self.badge() else { return };
        ui.separator();
        let color = egui::Color32::from_rgb(120, 200, 255);
        // a painted dot: the UI font has no "●" glyph (it rendered as a box)
        let (dot, _) = ui.allocate_exact_size(egui::vec2(10.0, ui.spacing().interact_size.y), egui::Sense::hover());
        ui.painter().circle_filled(dot.center(), 3.5, color);
        let r = ui.add(egui::Button::new(egui::RichText::new(text).small().color(color)).frame(false))
            .on_hover_text("A newer Halo Map Studio is on GitHub Releases. Click for the release notes and the download; right-click to skip this version.");
        if r.clicked() { self.show_window = true; }
        if r.secondary_clicked() { self.dismiss_current(); }
    }

    /// Settings ▸ Updates.
    pub fn ui_settings(&mut self, ui: &mut egui::Ui) {
        let mut on = self.prefs.auto_check;
        if ui.checkbox(&mut on, "Check for updates automatically").on_hover_text("Asks GitHub Releases once a day whether a newer version exists. Nothing is downloaded or installed without you clicking it. Off = only 'Check now'.").changed() {
            self.set_auto_check(on);
        }
        ui.horizontal(|ui| {
            ui.label(format!("Interval: every {} h · last checked: {}", CHECK_INTERVAL.as_secs() / 3600, ago(self.prefs.last_check_unix)));
        });
        if !self.prefs.last_result.is_empty() { ui.small(&self.prefs.last_result); }
        ui.horizontal(|ui| {
            if ui.add_enabled(self.check != CheckPhase::Checking, egui::Button::new("Check now")).clicked() { self.check_now(); }
            if self.check == CheckPhase::Checking { ui.spinner(); }
            if ui.button("Software update…").clicked() { self.show_window = true; }
        });
        ui.small(format!("This build: {}{}{}", Version::current(), if running_under_wine() { " (Windows build under Wine / Proton)" } else { "" }, self.resumed_version.as_ref().map(|v| format!(" · resumed after the update to {v}")).unwrap_or_default()));
        if !self.prefs.dismissed_version.is_empty() {
            ui.horizontal(|ui| {
                ui.small(format!("Skipped version: {}", self.prefs.dismissed_version));
                if ui.small_button("un-skip").clicked() { self.prefs.dismissed_version.clear(); save_prefs(&self.prefs); }
            });
        }
    }

    /// The Update window. Returns an install request when the user (or the auto-install box)
    /// asks to install the staged update; the App writes the recovery snapshot and restarts.
    pub fn ui_window(&mut self, ctx: &egui::Context) -> Option<InstallRequest> {
        if !self.show_window { return None; }
        let mut open = true;
        let mut request = None;
        egui::Window::new("Software update")
            .open(&mut open)
            .default_width(520.0)
            .collapsible(false)
            .show(ctx, |ui| {
                let cur = Version::current();
                ui.horizontal(|ui| {
                    ui.label(format!("Installed: {cur}"));
                    ui.separator();
                    match &self.latest {
                        Some(r) => { ui.label(format!("Latest: {}{}", r.version, if r.prerelease { " (pre-release)" } else { "" })).on_hover_text(format!("release tag {}", r.tag)); }
                        None => { ui.label("Latest: not checked yet"); }
                    }
                    if ui.add_enabled(self.check != CheckPhase::Checking, egui::Button::new("Check now")).clicked() { self.check_now(); }
                    if self.check == CheckPhase::Checking { ui.spinner(); }
                });
                if let CheckPhase::Failed(e) = &self.check { ui.colored_label(egui::Color32::from_rgb(230, 160, 90), format!("Check failed: {e}")); }
                if running_under_wine() {
                    ui.small("Running under Wine / Proton: the Windows archive is used.");
                }
                let Some(rel) = self.available().cloned() else {
                    if self.latest.is_some() { ui.label("You are running the latest version."); }
                    ui.hyperlink_to("All releases on GitHub", RELEASES_PAGE);
                    return;
                };
                ui.separator();
                ui.strong(if rel.name.is_empty() { format!("Version {}", rel.version) } else { rel.name.clone() });
                egui::ScrollArea::vertical().max_height(220.0).id_salt("update-notes").show(ui, |ui| {
                    let body = if rel.body.trim().is_empty() { "(no release notes)".to_string() } else { rel.body.clone() };
                    ui.add(egui::Label::new(egui::RichText::new(body).small()).wrap());
                });
                ui.hyperlink_to("Open the release page", &rel.html_url);
                ui.separator();
                let plat = Platform::running();
                match pick_asset(&rel, plat) {
                    Some(a) => { ui.small(format!("Archive: {} ({:.1} MB)", a.name, a.size as f64 / 1_048_576.0)); }
                    None => { ui.colored_label(egui::Color32::from_rgb(230, 160, 90), format!("This release has no {} archive.", plat.token())); }
                }
                match self.download.clone() {
                    DownloadPhase::NotStarted => {
                        if ui.button("Download").clicked() { self.start_download(); }
                    }
                    DownloadPhase::Running { got, total } => {
                        let frac = if total > 0 { (got as f32 / total as f32).min(1.0) } else { 0.0 };
                        ui.add(egui::ProgressBar::new(frac).show_percentage().animate(true));
                        ui.small(format!("{:.1} / {:.1} MB", got as f64 / 1_048_576.0, total as f64 / 1_048_576.0));
                    }
                    DownloadPhase::Staging => { ui.horizontal(|ui| { ui.spinner(); ui.label("Verified. Unpacking next to the install…"); }); }
                    DownloadPhase::Failed(e) => {
                        ui.colored_label(egui::Color32::from_rgb(230, 120, 100), e);
                        ui.horizontal(|ui| {
                            if ui.button("Try again").clicked() { self.start_download(); }
                            if let Some(d) = crate::project::settings_dir().map(|d| d.join("updates")) {
                                if ui.button("Open download folder").clicked() { let _ = std::fs::create_dir_all(&d); open_folder(&d); }
                            }
                        });
                    }
                    DownloadPhase::Ready => {
                        ui.colored_label(egui::Color32::from_rgb(140, 210, 140), "Downloaded, verified and unpacked. Ready to install.");
                    }
                }
                ui.add_space(4.0);
                ui.checkbox(&mut self.install_auto, "Install automatically when the download completes");
                // (an auto-install fires from take_install_request(), so closing the window
                // mid-download does not strand it)
                let ready = self.staged.is_some() && self.download == DownloadPhase::Ready;
                ui.label(egui::RichText::new("Installing closes Halo Map Studio, replaces its files and opens it again. Your open map, variant and any unsaved variant edits are saved to a recovery file first and restored after the restart. The previous version is kept in .hms-backup next to the app until the next start.").small());
                ui.horizontal(|ui| {
                    if ui.add_enabled(ready, egui::Button::new("Install now and restart")).clicked() {
                        request = self.staged.clone().map(|staged| InstallRequest { staged });
                    }
                    if ui.button("Remind me later").clicked() { self.show_window = false; }
                    if ui.button("Skip this version").on_hover_text("Hide the badge for this version (Settings ▸ Updates can undo it).").clicked() { self.dismiss_current(); self.show_window = false; }
                });
                if let Some(e) = &self.install_error {
                    ui.colored_label(egui::Color32::from_rgb(230, 120, 100), e);
                    if let Some(d) = crate::project::settings_dir().map(|d| d.join("updates")) {
                        if ui.button("Open download folder").clicked() { let _ = std::fs::create_dir_all(&d); open_folder(&d); }
                    }
                }
            });
        if !open { self.show_window = false; }
        if request.is_some() { self.install_auto = false; }
        request
    }

    /// The top-of-window banner after a resumed update (and the recovery offer on an ordinary
    /// start). Returns `Some(true)` = restore the offered snapshot, `Some(false)` = discard it.
    pub fn ui_banner(&mut self, ctx: &egui::Context) -> Option<bool> {
        let mut answer = None;
        if let Some((text, _)) = self.banner.clone() {
            egui::Area::new(egui::Id::new("hms-update-banner")).anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 36.0)).show(ctx, |ui| {
                egui::Frame::popup(ui.style()).fill(egui::Color32::from_rgb(34, 70, 46)).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(text).color(egui::Color32::WHITE));
                        if ui.small_button("×").clicked() { self.banner = None; }
                    });
                });
            });
        }
        if let Some(p) = self.offered_recovery.clone() {
            let stamp = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            egui::Window::new("Restore session?").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 80.0)).show(ctx, |ui| {
                ui.label("A recovery snapshot from an update restart was found (the last restart did not restore it).");
                ui.small(stamp);
                ui.horizontal(|ui| {
                    if ui.button("Restore").clicked() { answer = Some(true); }
                    if ui.button("Discard").clicked() { answer = Some(false); }
                });
            });
        }
        answer
    }
}

// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version { Version::parse(s).unwrap() }

    #[test]
    fn version_parse_and_compare() {
        assert_eq!(v("v1.3.1"), Version { major: 1, minor: 3, patch: 1, pre: String::new() });
        assert_eq!(v("1.3.1"), v("v1.3.1"));
        assert_eq!(v("1.1.4-pre").pre, "pre");
        assert!(v("1.10.0") > v("1.9.9"));
        assert!(v("2.0.0") > v("1.99.99"));
        assert!(v("1.3.1") > v("1.3.1-pre"), "a pre-release sorts below the release");
        assert!(v("1.3.2-pre") > v("1.3.1"));
        assert_eq!(v("1.3").patch, 0);
        assert!(Version::parse("latest").is_none());
        assert!(Version::parse("1.2.3.4").is_none());
        assert_eq!(v("1.3.1+build7").to_string(), "1.3.1");
        assert_eq!(Version::current().to_string(), env!("CARGO_PKG_VERSION"));
        assert_eq!(Platform::LinuxX64.asset_name(&v("1.3.1")), "HaloMapStudio-1.3.1-linux-x64.zip");
        assert_eq!(Platform::WindowsX64.asset_name(&v("1.3.1")), "HaloMapStudio-1.3.1-windows-x64.zip");
    }

    /// The real `releases/latest` document as GitHub served it on 2026-09-24 (v1.2.2 with the
    /// pre-contract asset names) must parse, and the asset picker must find each platform.
    #[test]
    fn release_json_fixture_parses_and_assets_pick() {
        let text = include_str!("testdata/updater_release.json");
        let rel = parse_release_json(text).expect("fixture parses");
        assert_eq!(rel.tag, "v1.2.2");
        assert_eq!(rel.version, v("1.2.2"));
        assert!(!rel.prerelease);
        assert!(rel.html_url.contains("/releases/tag/v1.2.2"));
        assert_eq!(rel.assets.len(), 2);
        let l = pick_asset(&rel, Platform::LinuxX64).expect("linux asset");
        assert!(l.name.contains("linux-x64") && l.url.starts_with("https://") && l.size > 0);
        let w = pick_asset(&rel, Platform::WindowsX64).expect("windows asset");
        assert!(w.name.contains("windows-x64"));
        assert!(pick_checksum_asset(&rel, l).is_none(), "the old releases publish no checksum");
    }

    #[test]
    fn asset_picker_prefers_contract_name_and_skips_portable_and_checksums() {
        let mk = |n: &str| Asset { name: n.into(), size: 1, url: format!("https://x/{n}") };
        let rel = Release {
            tag: "v1.3.2".into(), version: v("1.3.2"), name: String::new(), body: String::new(), html_url: String::new(), prerelease: false,
            assets: vec![
                mk("halo-map-studio-linux-x64-portable.zip"),
                mk("HaloMapStudio-1.3.2-linux-x64.zip.sha256"),
                mk("halo-map-studio-rust-0.1.0-linux-x64-20260922-2108.zip"),
                mk("HaloMapStudio-1.3.2-linux-x64.zip"),
                mk("HaloMapStudio-1.3.2-windows-x64.zip"),
                mk("SHA256SUMS"),
            ],
        };
        assert_eq!(pick_asset(&rel, Platform::LinuxX64).unwrap().name, "HaloMapStudio-1.3.2-linux-x64.zip");
        assert_eq!(pick_asset(&rel, Platform::WindowsX64).unwrap().name, "HaloMapStudio-1.3.2-windows-x64.zip");
        let l = pick_asset(&rel, Platform::LinuxX64).unwrap();
        assert_eq!(pick_checksum_asset(&rel, l).unwrap().name, "HaloMapStudio-1.3.2-linux-x64.zip.sha256");
        let w = pick_asset(&rel, Platform::WindowsX64).unwrap();
        assert_eq!(pick_checksum_asset(&rel, w).unwrap().name, "SHA256SUMS", "falls back to the combined file");
        // fallback: only the old naming present, portable must not be chosen
        let rel2 = Release { assets: vec![mk("halo-map-studio-linux-x64-portable.zip"), mk("halo-map-studio-rust-0.1.0-linux-x64-20260922-2108.zip")], ..rel.clone() };
        assert_eq!(pick_asset(&rel2, Platform::LinuxX64).unwrap().name, "halo-map-studio-rust-0.1.0-linux-x64-20260922-2108.zip");
        assert!(pick_asset(&rel2, Platform::WindowsX64).is_none());
    }

    #[test]
    fn sha256_text_forms() {
        let h = "a".repeat(64);
        assert_eq!(parse_sha256_text(&format!("{h}  HaloMapStudio-1.3.1-linux-x64.zip\n"), "HaloMapStudio-1.3.1-linux-x64.zip").as_deref(), Some(h.as_str()));
        assert_eq!(parse_sha256_text(&format!("{h} *HaloMapStudio-1.3.1-linux-x64.zip\n"), "HaloMapStudio-1.3.1-linux-x64.zip").as_deref(), Some(h.as_str()));
        assert_eq!(parse_sha256_text(&format!("{h}\n"), "whatever.zip").as_deref(), Some(h.as_str()), "bare digest");
        assert_eq!(parse_sha256_text(&format!("{h}  other.zip\n"), "mine.zip"), None);
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    /// Build a zip in memory: (name, data, deflate?) entries with a shared top folder.
    fn make_zip(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cd = Vec::new();
        for (name, data, deflate) in entries {
            let comp = if *deflate { miniz_oxide::deflate::compress_to_vec(data, 6) } else { data.to_vec() };
            let method: u16 = if *deflate { 8 } else { 0 };
            let crc = crc32(data);
            let off = out.len() as u32;
            let mut lh = vec![0x50, 0x4B, 0x03, 0x04, 20, 0, 0, 0];
            lh.extend_from_slice(&method.to_le_bytes());
            lh.extend_from_slice(&[0, 0, 0, 0]);
            lh.extend_from_slice(&crc.to_le_bytes());
            lh.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            lh.extend_from_slice(&(data.len() as u32).to_le_bytes());
            lh.extend_from_slice(&(name.len() as u16).to_le_bytes());
            lh.extend_from_slice(&[0, 0]);
            lh.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&lh);
            out.extend_from_slice(&comp);
            let mut c = vec![0x50, 0x4B, 0x01, 0x02, 3, 20, 20, 0, 0, 0];
            c.extend_from_slice(&method.to_le_bytes());
            c.extend_from_slice(&[0, 0, 0, 0]);
            c.extend_from_slice(&crc.to_le_bytes());
            c.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            c.extend_from_slice(&(data.len() as u32).to_le_bytes());
            c.extend_from_slice(&(name.len() as u16).to_le_bytes());
            c.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
            let mode: u32 = if name.ends_with("hms-app") { 0o100755 } else { 0o100644 };
            c.extend_from_slice(&(mode << 16).to_le_bytes());
            c.extend_from_slice(&off.to_le_bytes());
            c.extend_from_slice(name.as_bytes());
            cd.push(c);
        }
        let cd_off = out.len() as u32;
        let mut cd_len = 0u32;
        for c in &cd { out.extend_from_slice(c); cd_len += c.len() as u32; }
        out.extend_from_slice(&[0x50, 0x4B, 0x05, 0x06, 0, 0, 0, 0]);
        out.extend_from_slice(&(cd.len() as u16).to_le_bytes());
        out.extend_from_slice(&(cd.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hms-updater-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn zip_extract_strips_top_dir_verifies_crc_and_rejects_traversal() {
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let z = make_zip(&[
            ("HaloMapStudio-1.3.2-linux-x64/hms-app", &big, true),
            ("HaloMapStudio-1.3.2-linux-x64/libhalomapstudio.so", b"so-bytes", false),
            ("HaloMapStudio-1.3.2-linux-x64/README.md", b"# readme", true),
        ]);
        let d = tmp("zip");
        let files = extract_zip(&z, &d.join("out")).expect("extract");
        assert_eq!(files, vec![PathBuf::from("hms-app"), PathBuf::from("libhalomapstudio.so"), PathBuf::from("README.md")]);
        assert_eq!(std::fs::read(d.join("out/hms-app")).unwrap(), big);
        assert_eq!(std::fs::read(d.join("out/README.md")).unwrap(), b"# readme");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert!(std::fs::metadata(d.join("out/hms-app")).unwrap().permissions().mode() & 0o111 != 0, "exe bit");
        }
        // a corrupt byte in the compressed stream is caught
        let mut bad = z.clone();
        bad[30 + "HaloMapStudio-1.3.2-linux-x64/hms-app".len() + 100] ^= 0xFF; // inside the first entry's deflate stream
        assert!(extract_zip(&bad, &d.join("bad")).is_err());
        // path traversal
        let evil = make_zip(&[("x/../../etc/passwd", b"nope", false), ("x/hms-app", b"a", false)]);
        assert!(extract_zip(&evil, &d.join("evil")).unwrap_err().contains("unsafe path"));
        // no shared top-level folder: names kept as they are
        let flat = make_zip(&[("hms-app", b"a", false), ("README.md", b"b", false)]);
        let files = extract_zip(&flat, &d.join("flat")).unwrap();
        assert_eq!(files, vec![PathBuf::from("hms-app"), PathBuf::from("README.md")]);
        assert!(extract_zip(b"not a zip at all, definitely not", &d.join("nz")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// stage() unpacks next to the install; apply() renames the old files into .hms-backup and the
    /// new ones into place, leaving everything else alone; a failure rolls back.
    #[test]
    fn stage_and_apply_transaction_with_backup_and_rollback() {
        let d = tmp("install");
        let inst = d.join("app");
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::write(inst.join("hms-app"), b"OLD-EXE").unwrap();
        std::fs::write(inst.join("libhalomapstudio.so"), b"OLD-SO").unwrap();
        std::fs::write(inst.join("captures.txt"), b"user file").unwrap();
        let z = make_zip(&[("linux-x64/hms-app", b"NEW-EXE", true), ("linux-x64/libhalomapstudio.so", b"NEW-SO", false), ("linux-x64/README.md", b"new readme", false)]);
        let zip = d.join("HaloMapStudio-1.3.2-linux-x64.zip");
        std::fs::write(&zip, &z).unwrap();
        let st = stage(&zip, &inst, &v("1.3.2"), Platform::LinuxX64).expect("stage");
        assert!(st.dir.starts_with(&inst) && st.dir.file_name().unwrap().to_string_lossy() == ".hms-update-1.3.2");
        assert_eq!(st.files.len(), 3);
        // the wrong platform's archive is refused
        assert!(stage(&zip, &inst, &v("1.3.2"), Platform::WindowsX64).unwrap_err().contains("hms-app.exe"));
        let st = stage(&zip, &inst, &v("1.3.2"), Platform::LinuxX64).unwrap();
        let backup = apply(&st).expect("apply");
        assert_eq!(std::fs::read(inst.join("hms-app")).unwrap(), b"NEW-EXE");
        assert_eq!(std::fs::read(inst.join("libhalomapstudio.so")).unwrap(), b"NEW-SO");
        assert_eq!(std::fs::read(inst.join("README.md")).unwrap(), b"new readme");
        assert_eq!(std::fs::read(inst.join("captures.txt")).unwrap(), b"user file", "untouched files stay");
        assert_eq!(std::fs::read(backup.join("hms-app")).unwrap(), b"OLD-EXE");
        assert_eq!(std::fs::read(backup.join("libhalomapstudio.so")).unwrap(), b"OLD-SO");
        assert!(!st.dir.exists(), "staging dir removed after a successful apply");
        // rollback: a staged file that vanished mid-way must restore the previous state
        let st2 = stage(&zip, &inst, &v("1.3.3"), Platform::LinuxX64).unwrap();
        std::fs::remove_file(st2.dir.join("libhalomapstudio.so")).unwrap();
        let err = apply(&st2).unwrap_err();
        assert!(err.contains("libhalomapstudio.so"), "{err}");
        assert_eq!(std::fs::read(inst.join("hms-app")).unwrap(), b"NEW-EXE", "rolled back");
        assert_eq!(std::fs::read(inst.join("libhalomapstudio.so")).unwrap(), b"NEW-SO", "rolled back");
        // startup housekeeping: staging dirs go always, the backup only on a non-resume start
        startup_cleanup(&inst, true);
        assert!(!st2.dir.exists());
        assert!(inst.join(BACKUP_DIR).exists(), "kept through the resumed launch");
        startup_cleanup(&inst, false);
        assert!(!inst.join(BACKUP_DIR).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn resume_args_and_pid_wait() {
        let args: Vec<String> = ["hms-app", "--resume-after-update", "1.3.2", "--wait-pid", "4000000000"].iter().map(|s| s.to_string()).collect();
        assert_eq!(resume_arg(&args).as_deref(), Some("1.3.2"));
        assert!(resume_arg(&["hms-app".to_string()]).is_none());
        // a pid that cannot exist: returns immediately
        let t = Instant::now();
        handle_wait_pid_arg(&args, Duration::from_secs(5));
        assert!(t.elapsed() < Duration::from_secs(4));
        assert!(pid_alive(std::process::id()), "our own pid is alive");
    }

    /// The relaunch bug: `fs::canonicalize` hands back `\\?\C:\…`, which CreateProcess (and
    /// Wine) reject as a working directory, so the app never came back after an update.
    #[test]
    fn verbatim_prefix_is_stripped_for_process_spawn() {
        let p = |s: &str| plain_path(Path::new(s)).to_string_lossy().into_owned();
        assert_eq!(p(r"\\?\C:\Program Files\HaloMapStudio"), r"C:\Program Files\HaloMapStudio");
        assert_eq!(p(r"\\?\UNC\server\share\HaloMapStudio"), r"\\server\share\HaloMapStudio");
        // a bare drive is not a usable directory on its own
        assert_eq!(p(r"\\?\C:"), r"C:\");
        // already-plain paths (and every Linux path) are untouched
        assert_eq!(p(r"C:\Games\HaloMapStudio"), r"C:\Games\HaloMapStudio");
        assert_eq!(p("/opt/HaloMapStudio"), "/opt/HaloMapStudio");
        assert_eq!(p(r"\\server\share"), r"\\server\share");
        // and the dir the installer actually uses never carries the prefix
        if let Ok(d) = install_dir() {
            assert!(!d.to_string_lossy().starts_with(r"\\?\"), "install_dir() must be spawn-safe: {}", d.display());
        }
    }

    #[test]
    fn recovery_snapshot_round_trip() {
        let d = tmp("recovery");
        // `..Default::default()` on purpose: a snapshot must keep round-tripping when the project
        // record grows a field, and this test should not be the thing that breaks when it does.
        let p = crate::project::Project {
            version: 1, map_path: "/maps/forge_halo.map".into(),
            cam_pos: [1.0, 2.0, 3.0], cam_yaw: 0.25, cam_pitch: -0.5,
            variant_path: "/real/file.mvar".into(),
            ..Default::default()
        };
        let stamp = recovery_stamp();
        std::fs::write(d.join(format!("{stamp}.mvar")), b"edited variant bytes").unwrap();
        let mut meta = RecoveryMeta { from_version: "1.3.1".into(), to_version: "1.3.2".into(), variant_original: Some("/real/file.mvar".into()), variant_copy: Some(format!("{stamp}.mvar")), show_settings: true, ..Default::default() };
        let path = write_recovery(&d, &stamp, &mut meta, &p).expect("write");
        assert_eq!(find_latest_recovery(&d).as_ref(), Some(&path));
        let (m2, p2) = load_recovery(&path).expect("load");
        assert_eq!(m2, meta);
        assert_eq!(p2.map_path, "/maps/forge_halo.map");
        assert_eq!(p2.variant_path, "/real/file.mvar");
        assert!((p2.cam_yaw - 0.25).abs() < 1e-6);
        assert_eq!(std::fs::read(d.join(m2.variant_copy.as_ref().unwrap())).unwrap(), b"edited variant bytes");
        discard_recovery(&path);
        assert!(find_latest_recovery(&d).is_none());
        assert!(!d.join(format!("{stamp}.mvar")).exists() && !d.join(format!("{stamp}.mmsproj")).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn prefs_defaults_and_badge_rules() {
        let p: UpdatePrefs = serde_json::from_str("{}").unwrap();
        assert!(p.auto_check, "checking is on by default");
        assert_eq!(p.last_check_unix, 0);
        assert_eq!(ago(0), "never");
        assert_eq!(ago(now_unix()), "just now");
    }
}
