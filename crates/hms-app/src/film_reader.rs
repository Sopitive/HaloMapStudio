//! #film-reader  Reading a film that another process (MCC) is still WRITING, safely on Windows
//! and Linux.
//!
//! The rules this module exists to enforce (each one is a Windows failure mode, none of which
//! shows up on Linux):
//!
//! 1. **Open read-only with an explicit share mode of READ | WRITE | DELETE (7).** The writer holds
//!    a write handle; our handle must tolerate that (`FILE_SHARE_WRITE`) and must never block
//!    MCC's finalise rename `.temp -> .mov` (`FILE_SHARE_DELETE`). Access is read only: no
//!    `GENERIC_WRITE`, no `DELETE`.
//! 2. **Take the size from the open HANDLE** (`GetFileInformationByHandle`, i.e.
//!    `File::metadata`), never from the path. On NTFS the directory entry of a file another
//!    process is appending to is updated lazily, so a path `stat` can report a stale length and a
//!    follower concludes nothing has grown.
//! 3. **Positional reads of exactly the bytes wanted** (`seek_read` / `read_at`), no mapping: a
//!    Windows file mapping is fixed-size and a growing file cannot be mapped safely.
//! 4. **A sharing violation (Windows error 32/33) is retried with backoff for a few seconds**
//!    (MCC may hold the file exclusively only briefly), then reported with the raw OS error, what
//!    was tried and the fallback that works. Every such failure is appended to a log in the
//!    settings folder so a Windows user can send it back.
//! 5. **The finalise handover.** When the `.temp` path disappears (MCC renamed it) the open handle
//!    still reads the same file; the follower re-reads it once, then looks up the finished `.mov`
//!    by name and carries on with that.
//!
//! `film_reader_source_gate` (tests) greps the film-opening code for the calls this module
//! replaces, so they cannot creep back.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// `FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE`.
pub const SHARE_ALL: u32 = 7;

/// Open for reading, read access only, with [`SHARE_ALL`] on Windows. (Rust's default is the same
/// share mode; it is stated here so the intent is in the code, not an accident of the std
/// version.)
pub fn open_shared(path: &Path) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        o.share_mode(SHARE_ALL);
    }
    o.open(path)
}

/// Read at an absolute offset without relying on (or disturbing) a shared cursor.
pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, off)
    }
}

/// The file's length, asked of the open handle.
pub fn handle_len(f: &File) -> io::Result<u64> {
    f.metadata().map(|m| m.len())
}

/// Exactly `len` bytes from `off`, or fewer if the file ends first.
pub fn read_range(f: &File, off: u64, len: u64) -> io::Result<Vec<u8>> {
    let mut out = vec![0u8; len as usize];
    let mut got = 0usize;
    while got < out.len() {
        match read_at(f, &mut out[got..], off + got as u64) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    out.truncate(got);
    Ok(out)
}

/// The whole file as it is NOW: size from the handle, then positional reads, re-asking the handle
/// in case the writer got ahead of us. A growing file is read to a consistent prefix, never
/// failed.
pub fn read_to_end(f: &File) -> io::Result<Vec<u8>> {
    let mut bytes = read_range(f, 0, handle_len(f)?)?;
    for _ in 0..4 {
        let now = handle_len(f)?;
        if now <= bytes.len() as u64 {
            break;
        }
        let more = read_range(f, bytes.len() as u64, now - bytes.len() as u64)?;
        if more.is_empty() {
            break;
        }
        bytes.extend_from_slice(&more);
    }
    Ok(bytes)
}

/// Length of a file by OPENING it (handle size), for a listing or a status line. Falls back to
/// nothing: a path `stat` would be the stale number this exists to avoid.
pub fn len_of(path: &Path) -> io::Result<u64> {
    handle_len(&open_shared(path)?)
}

/// Length for a listing: the handle's for an in-progress `.temp`, the directory entry's for a
/// finished file (which cannot grow, so its entry is exact).
pub fn listing_len(path: &Path) -> io::Result<u64> {
    if crate::film::is_in_progress(path) {
        len_of(path)
    } else {
        std::fs::metadata(path).map(|m| m.len())
    }
}

// ---------------------------------------------------------------------------------------------
// opening with retry and a diagnostic
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Retry {
    pub total: Duration,
    pub first: Duration,
    pub max_step: Duration,
}

impl Retry {
    /// For a worker thread or the CLI: a few seconds.
    pub const PATIENT: Retry = Retry { total: Duration::from_millis(4000), first: Duration::from_millis(50), max_step: Duration::from_millis(500) };
    /// For the UI thread: never hold a frame for long; the next poll tries again.
    pub const QUICK: Retry = Retry { total: Duration::from_millis(250), first: Duration::from_millis(25), max_step: Duration::from_millis(100) };
    #[allow(dead_code)]
    pub const NONE: Retry = Retry { total: Duration::ZERO, first: Duration::ZERO, max_step: Duration::ZERO };
}

/// Windows `ERROR_SHARING_VIOLATION` (32) and `ERROR_LOCK_VIOLATION` (33). On other platforms
/// these numbers mean something else, so nothing is retried there.
pub fn is_sharing_violation(e: &io::Error) -> bool {
    cfg!(windows) && matches!(e.raw_os_error(), Some(32) | Some(33))
}

/// Why an open failed, with everything a Windows user needs to send back.
#[derive(Clone, Debug)]
pub struct OpenFailure {
    /// File NAME only.
    pub name: String,
    pub code: Option<i32>,
    pub text: String,
    pub attempts: u32,
    pub waited: Duration,
    pub share_mode: u32,
    pub sharing_violation: bool,
}

impl std::fmt::Display for OpenFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = self.code.map_or("none".to_string(), |c| c.to_string());
        if self.sharing_violation {
            write!(
                f,
                "Could not open {}: another program has it open with a share mode that forbids readers (OS error {code}: {}). \
                 Tried {} time(s) over {:.1} s, read-only with share mode READ|WRITE|DELETE (7). \
                 Fix: let the match finish (MCC then renames it to a finished .mov and we pick that up), or copy the film once MCC has closed it, then open the copy.",
                self.name,
                self.text,
                self.attempts,
                self.waited.as_secs_f32()
            )
        } else {
            write!(f, "Could not open {}: OS error {code}: {} (tried {} time(s), read-only, share mode {})", self.name, self.text, self.attempts, self.share_mode)
        }
    }
}

impl OpenFailure {
    /// Append this failure to the settings-folder log.
    pub fn log(&self, path: &Path) {
        let size = std::fs::metadata(path).map(|m| m.len().to_string()).unwrap_or_else(|_| "?".into());
        log_diag(&format!(
            "open FAILED name={} os_error={} text={:?} attempts={} waited_ms={} share_mode={} access=read path_size={} sharing_violation={}",
            self.name,
            self.code.map_or("none".into(), |c| c.to_string()),
            self.text,
            self.attempts,
            self.waited.as_millis(),
            self.share_mode,
            size,
            self.sharing_violation
        ));
    }
}

fn name_of(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Open with the share mode above; on a Windows sharing violation retry with exponential backoff
/// until `retry.total` has passed.
pub fn open_with_retry(path: &Path, retry: &Retry) -> Result<File, OpenFailure> {
    let start = std::time::Instant::now();
    let mut step = retry.first;
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        match open_shared(path) {
            Ok(f) => {
                if attempts > 1 {
                    log_diag(&format!("open ok-after-retry name={} attempts={} waited_ms={}", name_of(path), attempts, start.elapsed().as_millis()));
                }
                return Ok(f);
            }
            Err(e) => {
                let viol = is_sharing_violation(&e);
                if viol && start.elapsed() < retry.total {
                    std::thread::sleep(step.min(retry.total.saturating_sub(start.elapsed())));
                    step = (step * 2).min(retry.max_step);
                    continue;
                }
                return Err(OpenFailure {
                    name: name_of(path),
                    code: e.raw_os_error(),
                    text: e.to_string(),
                    attempts,
                    waited: start.elapsed(),
                    share_mode: SHARE_ALL,
                    sharing_violation: viol,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the diagnostic log
// ---------------------------------------------------------------------------------------------

pub const LOG_FILE: &str = "film_open.log";
const LOG_CAP: u64 = 256 * 1024;

/// Append one timestamped line to `<settings dir>/film_open.log` (best effort, size-capped).
pub fn log_diag(line: &str) {
    if let Some(dir) = crate::project::settings_dir() {
        log_diag_to(&dir, line);
    }
}

pub fn log_diag_to(dir: &Path, line: &str) {
    use std::io::Write;
    let _ = std::fs::create_dir_all(dir);
    let p = dir.join(LOG_FILE);
    if std::fs::metadata(&p).map_or(false, |m| m.len() > LOG_CAP) {
        let _ = std::fs::write(&p, b"");
    }
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "{t} {} {line}", if cfg!(windows) { "windows" } else { "unix" });
    }
}

// ---------------------------------------------------------------------------------------------
// the finalise handover
// ---------------------------------------------------------------------------------------------

/// Is `candidate` (its first bytes) the same recording as `have` (the bytes read so far)?
pub type SameRecording = fn(&[u8], &[u8]) -> bool;

/// How many leading bytes of a candidate are enough for a [`SameRecording`] check.
pub const VERIFY_PREFIX: u64 = 0x1F8A8 + 4096;

/// The finished file that replaced `temp`: same folder (or one of the film folders), a `.mov` or
/// `.film` whose name starts with the `.temp`'s stem, whose leading bytes pass `same`. Newest
/// first.
pub fn find_finalised(temp: &Path, have: &[u8], same: SameRecording, extra_dirs: &[PathBuf]) -> Option<PathBuf> {
    let stem = temp.file_stem()?.to_string_lossy().to_ascii_lowercase();
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = temp.parent() {
        dirs.push(d.to_path_buf());
    }
    dirs.extend(extra_dirs.iter().cloned());
    let mut cands: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let ext_ok = p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mov") || x.eq_ignore_ascii_case("film"));
            let name = p.file_stem().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            if ext_ok && name.starts_with(&stem) {
                let m = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                cands.push((m, p));
            }
        }
    }
    cands.sort_by(|a, b| b.0.cmp(&a.0));
    cands.dedup_by(|a, b| a.1 == b.1);
    for (_, p) in cands {
        let Ok(f) = open_shared(&p) else { continue };
        let Ok(head) = read_range(&f, 0, VERIFY_PREFIX) else { continue };
        if same(have, &head) {
            return Some(p);
        }
    }
    None
}

/// A growing file read through ONE handle that stays open across polls.
pub struct Tail {
    pub path: PathBuf,
    file: File,
    bytes: Vec<u8>,
    same: SameRecording,
    extra_dirs: Vec<PathBuf>,
    /// Set once the `.temp` path vanished and the finished file was found.
    pub finalised_as: Option<PathBuf>,
    /// Set once the path vanished and nothing could be found to continue with.
    pub vanished: bool,
}

/// What one [`Tail::poll`] saw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Polled {
    pub new_bytes: usize,
    /// The finalise handover happened in this poll.
    pub just_finalised: bool,
    /// The file is shorter than what we had (rewritten from scratch): everything was re-read.
    pub restarted: bool,
}

impl Tail {
    pub fn open(path: &Path, retry: &Retry, same: SameRecording, extra_dirs: Vec<PathBuf>) -> Result<Tail, OpenFailure> {
        let file = open_with_retry(path, retry).inspect_err(|e| e.log(path))?;
        if crate::film::is_in_progress(path) {
            log_diag(&format!("open ok name={} share_mode={} access=read handle_len={}", name_of(path), SHARE_ALL, handle_len(&file).map_or("?".into(), |l| l.to_string())));
        }
        Ok(Tail { path: path.to_path_buf(), file, bytes: Vec::new(), same, extra_dirs, finalised_as: None, vanished: false })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Pull whatever the writer has added. Size comes from the handle every time.
    pub fn poll(&mut self) -> io::Result<Polled> {
        let mut out = Polled::default();
        let len = handle_len(&self.file)?;
        let have = self.bytes.len() as u64;
        if len < have {
            // rewritten shorter: start over from the handle
            self.bytes = read_to_end(&self.file)?;
            out.restarted = true;
            out.new_bytes = self.bytes.len();
        } else if len > have {
            let more = read_range(&self.file, have, len - have)?;
            out.new_bytes = more.len();
            self.bytes.extend_from_slice(&more);
            // The game also rewrites the fixed header region as it goes; refresh that prefix so a
            // stale copy of it is never parsed against newer records.
            let head = (crate::film::HEADER_PREFIX_LEN as u64).min(self.bytes.len() as u64);
            if have > 0 {
                let fresh = read_range(&self.file, 0, head)?;
                let n = fresh.len().min(self.bytes.len());
                self.bytes[..n].copy_from_slice(&fresh[..n]);
            }
        }
        // finalise: the path is gone but our handle still reads the renamed file
        if self.finalised_as.is_none() && !self.vanished && !self.path.exists() {
            // read the final state through the handle first (header fields are rewritten at finalise)
            let all = read_to_end(&self.file)?;
            if all.len() != self.bytes.len() {
                out.new_bytes += all.len().saturating_sub(self.bytes.len());
            }
            self.bytes = all;
            match find_finalised(&self.path, &self.bytes, self.same, &self.extra_dirs) {
                Some(p) => {
                    log_diag(&format!("handover name={} -> {} final_len={}", name_of(&self.path), name_of(&p), self.bytes.len()));
                    self.finalised_as = Some(p);
                    out.just_finalised = true;
                }
                None => {
                    log_diag(&format!("vanished name={} no finished file found; continuing on the open handle", name_of(&self.path)));
                    self.vanished = true;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hms-film-reader-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn same_prefix(a: &[u8], b: &[u8]) -> bool {
        let n = a.len().min(b.len()).min(64);
        n > 0 && a[..n] == b[..n]
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8).collect()
    }

    /// #film-reader  GATE. A file appended by ANOTHER THREAD in odd-sized chunks (so most polls end
    /// mid-record) is followed through one handle: every poll returns a byte-exact prefix of what has
    /// been written, the length is the handle's, growth is never missed, and the final read is exact.
    #[test]
    fn a_file_appended_by_another_thread_is_followed_through_one_handle() {
        let d = tmp("grow");
        let p = d.join("live.temp");
        std::fs::write(&p, b"").unwrap();
        let data = pattern(300_001);
        let writer = {
            let (p, data) = (p.clone(), data.clone());
            std::thread::spawn(move || {
                let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
                let mut at = 0;
                let mut k = 1usize;
                while at < data.len() {
                    let n = (997 * k % 7919 + 1).min(data.len() - at);
                    f.write_all(&data[at..at + n]).unwrap();
                    f.flush().unwrap();
                    at += n;
                    k += 1;
                    std::thread::sleep(Duration::from_micros(300));
                }
            })
        };
        let mut t = Tail::open(&p, &Retry::NONE, same_prefix, vec![]).unwrap();
        let mut polls_with_growth = 0;
        let mut spins = 0;
        while t.bytes().len() < data.len() {
            let r = t.poll().unwrap();
            if r.new_bytes > 0 {
                polls_with_growth += 1;
            }
            assert_eq!(t.bytes(), &data[..t.bytes().len()], "a byte-exact prefix after every poll");
            spins += 1;
            assert!(spins < 2_000_000, "stalled at {} of {}", t.bytes().len(), data.len());
        }
        writer.join().unwrap();
        assert!(polls_with_growth > 20, "growth was seen in many steps, not once ({polls_with_growth})");
        assert_eq!(handle_len(&t.file).unwrap(), data.len() as u64);
        assert_eq!(read_to_end(&open_shared(&p).unwrap()).unwrap(), data);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// #film-reader  GATE. The finalise handover: the writer renames `x.temp` to `x_6A95D8EF.mov`
    /// while we hold it open and appends a last chunk through ITS handle. We must (a) keep reading
    /// through our handle, (b) notice the path is gone, (c) find the finished file by name and
    /// check it is the same recording, (d) end with every byte.
    #[test]
    fn the_follower_survives_the_finalise_rename_and_finds_the_finished_file() {
        let d = tmp("final");
        let temp = d.join("asq_test_CCB00D77.temp");
        let data = pattern(50_000);
        let mut w = std::fs::OpenOptions::new().create(true).append(true).open(&temp).unwrap();
        w.write_all(&data[..20_000]).unwrap();
        w.flush().unwrap();
        let mut t = Tail::open(&temp, &Retry::NONE, same_prefix, vec![]).unwrap();
        assert_eq!(t.poll().unwrap().new_bytes, 20_000);
        // a decoy with the right name but another recording
        std::fs::write(d.join("asq_test_CCB00D77_00000001.mov"), pattern(200).iter().map(|b| !b).collect::<Vec<u8>>()).unwrap();
        w.write_all(&data[20_000..35_000]).unwrap();
        w.flush().unwrap();
        // MCC finalises: rename while we hold the file open, then finish writing through its handle
        let mov = d.join("asq_test_CCB00D77_6A95D8EF.mov");
        std::fs::rename(&temp, &mov).expect("our handle must not block the rename");
        w.write_all(&data[35_000..]).unwrap();
        w.flush().unwrap();
        drop(w);
        let r = t.poll().unwrap();
        assert!(r.just_finalised, "the vanished path is noticed");
        assert_eq!(t.bytes(), &data[..], "every byte, including the part written after the rename");
        // the decoy (different recording) is rejected, the real one found
        assert_eq!(t.finalised_as.as_deref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()), Some("asq_test_CCB00D77_6A95D8EF.mov".into()));
        assert_eq!(t.poll().unwrap(), Polled::default(), "a finished file does not grow");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_vanished_file_with_no_successor_is_reported_not_fatal() {
        let d = tmp("gone");
        let temp = d.join("lonely.temp");
        std::fs::write(&temp, pattern(1000)).unwrap();
        let mut t = Tail::open(&temp, &Retry::NONE, same_prefix, vec![]).unwrap();
        t.poll().unwrap();
        std::fs::remove_file(&temp).unwrap();
        let r = t.poll().unwrap();
        assert!(t.vanished && t.finalised_as.is_none() && !r.just_finalised);
        assert_eq!(t.bytes().len(), 1000, "what was read is kept");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_rewritten_shorter_file_restarts_the_read() {
        let d = tmp("rewrite");
        let p = d.join("a.temp");
        std::fs::write(&p, pattern(5000)).unwrap();
        let mut t = Tail::open(&p, &Retry::NONE, same_prefix, vec![]).unwrap();
        t.poll().unwrap();
        // truncate in place (same inode), as a recorder starting over would
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        f.set_len(0).unwrap();
        drop(f);
        std::fs::write(&p, pattern(1200)).unwrap();
        let r = t.poll().unwrap();
        assert!(r.restarted);
        assert_eq!(t.bytes(), &pattern(1200)[..]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn open_failures_are_actionable_and_logged_with_only_a_basename() {
        let d = tmp("fail");
        let p = d.join("nothing_here.temp");
        let e = open_with_retry(&p, &Retry::QUICK).unwrap_err();
        assert!(!e.sharing_violation, "a missing file is not a sharing violation, so it is not retried");
        assert_eq!(e.attempts, 1);
        let msg = e.to_string();
        assert!(msg.contains("nothing_here.temp") && msg.contains("OS error"), "{msg}");
        assert!(!msg.contains(d.to_string_lossy().as_ref()), "no directory in the message");
        e.log(&p);
        log_diag_to(&d, "sample line");
        let log = std::fs::read_to_string(d.join(LOG_FILE)).unwrap();
        assert!(log.contains("sample line"));
        // the sharing-violation text names the fix
        let v = OpenFailure { name: "x.temp".into(), code: Some(32), text: "in use".into(), attempts: 9, waited: Duration::from_secs(4), share_mode: 7, sharing_violation: true };
        let s = v.to_string();
        assert!(s.contains("32") && s.contains("share mode") && s.contains(".mov") && s.contains("copy"), "{s}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// #film-reader  GATE (source). The film-opening code must not map a growing file, take a size
    /// from a path, or read a whole file by path. Checks the named functions' bodies; the saboteur
    /// below proves the checker bites.
    fn violations(src: &str, fns: &[&str]) -> Vec<String> {
        let banned = ["memmap", "Mmap", "fs::metadata(", "fs::read(", "File::open(", ".modified()"];
        let mut out = Vec::new();
        for f in fns {
            let Some(i) = src.find(f) else {
                out.push(format!("function `{f}` not found"));
                continue;
            };
            let line_start = src[..i].rfind('\n').map_or(0, |k| k + 1);
            let indent = &src[line_start..i];
            let indent: String = indent.chars().take_while(|c| *c == ' ').collect();
            let end = format!("\n{indent}}}\n");
            let body: String = src[i..].split(end.as_str()).next().unwrap_or("").lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
            for b in banned {
                if body.contains(b) {
                    out.push(format!("`{f}` uses `{b}`"));
                }
            }
        }
        out
    }

    const CHECKED: [(&str, &[&str]); 3] = [
        ("film.rs", &["pub fn open(path: &Path)", "pub fn open_header(path: &Path)"]),
        ("film_follow.rs", &["pub fn run_cli("]),
        ("main.rs", &["fn film_follow_tick(&mut self)", "fn film_follow_poll_len(&mut self"]),
    ];

    #[test]
    fn film_reader_source_gate_no_mmap_and_no_path_sizes_on_a_growing_file() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for (file, fns) in CHECKED {
            let src = std::fs::read_to_string(root.join(file)).unwrap();
            let v = violations(&src, fns);
            assert!(v.is_empty(), "{file}: {v:?}");
        }
        // the reader itself maps nothing and stats no path for a size (only `listing_len` for a FINISHED file)
        let me = std::fs::read_to_string(root.join("film_reader.rs")).unwrap();
        let code: String = me.split("#[cfg(test)]").next().unwrap().lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        assert!(!code.contains("memmap") && !code.contains("Mmap"));
        assert_eq!(code.matches("fs::metadata(").count(), 3, "the three audited stats: the finished-file listing, the diagnostic path_size, the log cap");
    }

    #[test]
    fn the_source_gate_bites() {
        // SABOTEUR: put each banned call into a checked function and the checker must object.
        let clean = "pub fn open(path: &Path) -> X {\n    let f = film_reader::open_shared(path);\n}\n";
        assert!(violations(clean, &["pub fn open(path: &Path)"]).is_empty());
        for bad in ["std::fs::metadata(p)", "std::fs::read(path)", "memmap2::Mmap::map(&f)", "std::fs::File::open(path)", "m.modified()"] {
            let dirty = format!("pub fn open(path: &Path) -> X {{\n    let f = {bad};\n}}\n");
            assert!(!violations(&dirty, &["pub fn open(path: &Path)"]).is_empty(), "{bad}");
        }
        assert!(!violations(clean, &["pub fn nope("]).is_empty(), "a renamed function is a failure, not a pass");
    }
}
