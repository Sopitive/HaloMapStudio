//! "Open recent" — the ONE list of things the user actually reopens: map variants (`.mvar`),
//! base maps (`.map`), saved films (`.mov` / `.film`) and imported models (OBJ / GLB / glTF).
//!
//! Before this there were four unrelated histories, each owned by a dialog: three lists of
//! recently-visited FOLDERS (one per browser purpose, session-only) plus the model browser's
//! recent FILES. None of them was reachable without first opening the dialog they belonged to,
//! so "reopen what I was working on" always meant browsing for it again. This store is the
//! single history behind `File ▸ Open recent`; the browsers' own sidebars read it too (their
//! folder lists seed from the parent folders recorded here), so nothing in them regressed.
//!
//! An entry carries enough to render a good label WITHOUT touching the disk — the display name,
//! which game it belongs to, and a variant's title where it is known. Opening a recent item is
//! deliberately not a second loader: the app hands the path to the same code path the browser
//! uses (`import_mvar` / `open_film` / `import_model_file` / `load_map`), so a variant still
//! routes to its own base map, Workshop maps included (they resolve by MCC map GUID).
//!
//! Persisted as `recent_items.json` in `project::settings_dir()`, next to the other settings. A
//! corrupt, truncated or newer file never costs more than the history: rows are converted one at
//! a time and an unreadable row is dropped, so a damaged file degrades to a shorter list (or an
//! empty one) instead of failing the launch.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-kind cap. The list is kept per kind rather than globally so importing a dozen models in a
/// row cannot push every variant out of the history.
pub const PER_KIND_CAP: usize = 12;

/// The file format version written into the file (for a future migration; the reader accepts
/// anything it can convert row by row).
const FORMAT_VERSION: u32 = 1;

/// What a recent entry IS. The menu groups by this, so a variant is never listed next to a model.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[serde(rename_all = "lowercase")]
pub enum RecentKind {
    /// A Forge map variant (`.mvar`), any game.
    Variant,
    /// A base map cache (`.map`), stock or Workshop.
    Map,
    /// A Reach Theater saved film (`.mov` / `.film`).
    Film,
    /// An externally imported model (OBJ / GLB / glTF).
    Model,
}

impl RecentKind {
    /// Menu / panel order, most-reopened kind first.
    pub const ALL: [RecentKind; 4] = [RecentKind::Variant, RecentKind::Map, RecentKind::Film, RecentKind::Model];

    /// The submenu's group heading.
    pub fn group_label(self) -> &'static str {
        match self {
            RecentKind::Variant => "Map variants",
            RecentKind::Map => "Base maps",
            RecentKind::Film => "Saved films",
            RecentKind::Model => "Imported models",
        }
    }

    /// Singular noun for a hover / status line ("Reach map variant").
    pub fn noun(self) -> &'static str {
        match self {
            RecentKind::Variant => "map variant",
            RecentKind::Map => "base map",
            RecentKind::Film => "saved film",
            RecentKind::Model => "imported model",
        }
    }
}

/// One remembered item. Everything but `kind` + `path` is a label aid, and every label aid is
/// optional — an entry written by an older build, or one whose title was not known at the time,
/// still renders (it falls back to the file name).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RecentItem {
    pub kind: RecentKind,
    /// Absolute path as it was opened.
    pub path: String,
    /// File name, so the menu never has to stat or split the path at paint time.
    #[serde(default)]
    pub name: String,
    /// Which game it belongs to, as a person says it ("Reach", "Halo 4", "Halo 2 Anniversary",
    /// "Halo 3"). Empty for a model, which belongs to no title.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub game: String,
    /// A variant's / film's own title where it is known ("Tower of Doom"), else empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
}

impl RecentItem {
    pub fn new(kind: RecentKind, path: &Path, game: &str, title: &str) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        Self {
            kind,
            path: path.to_string_lossy().into_owned(),
            name,
            game: game.trim().to_string(),
            title: title.trim().to_string(),
        }
    }

    pub fn as_path(&self) -> &Path {
        Path::new(&self.path)
    }

    /// Fill in anything a hand-edited / older-build row left out, so every row can be rendered
    /// from the record alone.
    fn normalize(&mut self) {
        if self.name.trim().is_empty() {
            if let Some(n) = self.as_path().file_name() {
                self.name = n.to_string_lossy().into_owned();
            }
        }
    }

    /// Is the file still where it was opened from? Called when the menu OPENS, never per frame.
    pub fn exists(&self) -> bool {
        self.as_path().exists()
    }

    /// The best human name: the item's own title where it has one, else the file name (variant
    /// files are named by a random id, so the title is what a person recognises).
    pub fn display_name(&self) -> &str {
        if self.title.is_empty() {
            if self.name.is_empty() { &self.path } else { &self.name }
        } else {
            &self.title
        }
    }

    /// The menu row: "Reach · Tower of Doom". The game is on every row so a Halo 3 variant can
    /// never be mistaken for a Reach one.
    pub fn menu_label(&self) -> String {
        let name = ellipsize(self.display_name(), 44);
        if self.game.is_empty() { name } else { format!("{} · {}", self.game, name) }
    }

    /// The hover text: what it is, its own title, and the full path (which is the point — the
    /// menu row is short, the tooltip is exact).
    pub fn hover(&self) -> String {
        let what = if self.game.is_empty() {
            self.kind.noun().to_string()
        } else {
            format!("{} {}", self.game, self.kind.noun())
        };
        let mut s = format!("{what}\n{}", self.path);
        if !self.title.is_empty() && self.title != self.name {
            s = format!("{what}\n{}\n{}", self.title, self.path);
        }
        s
    }
}

/// The whole history, newest first within each kind.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentStore {
    items: Vec<RecentItem>,
}

/// The on-disk shape. `items` is held as raw JSON values so one unreadable row cannot take the
/// whole history with it.
#[derive(Serialize, Deserialize, Default)]
struct RecentFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    items: Vec<serde_json::Value>,
}

impl RecentStore {
    /// Where the history lives: `recent_items.json` in the per-user settings dir (the same
    /// folder as `project.mmsproj`, `last_map.txt` and the bookmarks).
    pub fn file_path() -> Option<PathBuf> {
        crate::project::settings_dir().map(|d| d.join("recent_items.json"))
    }

    /// Read the history. Any failure — no settings dir, no file, unparsable JSON — yields an
    /// EMPTY store, never an error the caller has to handle at startup.
    pub fn load() -> Self {
        match Self::file_path() {
            Some(p) => Self::load_from(&p),
            None => Self::default(),
        }
    }

    /// Read from an explicit file (the test seam). Rows are converted one at a time: a row that
    /// is not a readable entry, or carries no path, is dropped and the rest are kept.
    pub fn load_from(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else { return Self::default() };
        let Ok(file) = serde_json::from_str::<RecentFile>(&text) else { return Self::default() };
        let mut out = Self::default();
        for row in file.items {
            if let Ok(mut item) = serde_json::from_value::<RecentItem>(row) {
                if item.path.trim().is_empty() { continue; }
                item.normalize();
                out.items.push(item);
            }
        }
        // A file hand-edited into a bad state (duplicates, over the cap) still loads as a
        // well-formed history.
        out.enforce_caps();
        out
    }

    /// Write the history. Best effort: a read-only settings dir must not break anything.
    pub fn save(&self) {
        let Some(p) = Self::file_path() else { return };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = self.save_to(&p);
    }

    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        let file = RecentFile {
            version: FORMAT_VERSION,
            items: self.items.iter().filter_map(|i| serde_json::to_value(i).ok()).collect(),
        };
        let json = serde_json::to_string_pretty(&file).unwrap_or_else(|_| "{}".into());
        std::fs::write(path, json)
    }

    /// Record a successfully-opened item: newest first, de-duplicated by (kind, path), capped
    /// per kind. Re-opening something already in the list MOVES it to the front rather than
    /// adding a second row, and refreshes its label (a variant that was renamed since).
    pub fn note(&mut self, mut item: RecentItem) {
        if item.path.trim().is_empty() {
            return;
        }
        item.normalize();
        self.items.retain(|i| !(i.kind == item.kind && path_eq(&i.path, &item.path)));
        self.items.insert(0, item);
        self.enforce_caps();
    }

    /// Drop one entry (the user confirming "it's gone, forget it", or `recent remove`).
    pub fn remove(&mut self, kind: RecentKind, path: &str) -> bool {
        let n = self.items.len();
        self.items.retain(|i| !(i.kind == kind && path_eq(&i.path, path)));
        self.items.len() != n
    }

    /// Drop every entry whose file is no longer on disk; returns how many went.
    pub fn remove_missing(&mut self) -> usize {
        let n = self.items.len();
        self.items.retain(|i| i.exists());
        n - self.items.len()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Every entry, newest first (the kinds interleaved in open order).
    pub fn items(&self) -> &[RecentItem] {
        &self.items
    }

    /// The entries of one kind, newest first.
    pub fn of_kind(&self, kind: RecentKind) -> impl Iterator<Item = &RecentItem> {
        self.items.iter().filter(move |i| i.kind == kind)
    }

    /// The parent folders of the recent entries of one kind, newest first and de-duplicated —
    /// what a browser's "Recent folders" sidebar wants, derived from this one history instead of
    /// a fourth list of its own.
    pub fn folders_of_kind(&self, kind: RecentKind) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        for i in self.of_kind(kind) {
            if let Some(d) = i.as_path().parent() {
                if !out.iter().any(|p| p == d) {
                    out.push(d.to_path_buf());
                }
            }
        }
        out
    }

    /// The paths of one kind, newest first (the model dialog's "Recent models" list).
    pub fn paths_of_kind(&self, kind: RecentKind) -> Vec<PathBuf> {
        self.of_kind(kind).map(|i| i.as_path().to_path_buf()).collect()
    }

    fn enforce_caps(&mut self) {
        let mut seen: Vec<(RecentKind, String)> = Vec::new();
        let mut per_kind: Vec<(RecentKind, usize)> = Vec::new();
        self.items.retain(|i| {
            if seen.iter().any(|(k, p)| *k == i.kind && path_eq(p, &i.path)) {
                return false;
            }
            let slot = match per_kind.iter_mut().find(|(k, _)| *k == i.kind) {
                Some(s) => s,
                None => {
                    per_kind.push((i.kind, 0));
                    per_kind.last_mut().unwrap()
                }
            };
            if slot.1 >= PER_KIND_CAP {
                return false;
            }
            slot.1 += 1;
            seen.push((i.kind, i.path.clone()));
            true
        });
    }
}

/// Same file? Windows paths are case-insensitive, Unix paths are not; a trailing separator or a
/// `.` component must not make two spellings look different either.
fn path_eq(a: &str, b: &str) -> bool {
    #[cfg(windows)]
    {
        if a.eq_ignore_ascii_case(b) {
            return true;
        }
    }
    if a == b {
        return true;
    }
    Path::new(a) == Path::new(b)
}

/// Shorten a long name in the middle-ish (keep the tail, which is where a name differs).
fn ellipsize(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(1);
    let head: String = s.chars().take(keep).collect();
    format!("{}…", head.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hms-recent-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn item(kind: RecentKind, path: &str, game: &str, title: &str) -> RecentItem {
        RecentItem::new(kind, Path::new(path), game, title)
    }

    /// The history must survive a trip through the settings file with every label aid intact —
    /// the point of storing the name / game / title is that the menu never reads the disk.
    #[test]
    fn round_trips_through_the_settings_file() {
        let dir = tmp("roundtrip");
        let f = dir.join("recent_items.json");
        let mut s = RecentStore::default();
        s.note(item(RecentKind::Variant, "/v/a b c.mvar", "Reach", "Tower of Doom"));
        s.note(item(RecentKind::Map, "/maps/ca_forge_ravine.map", "Halo 4", "ca_forge_ravine"));
        s.note(item(RecentKind::Film, "/films/clip.mov", "Reach", "Slayer on Hemorrhage"));
        s.note(item(RecentKind::Model, "/home/u/my models/ship.obj", "", ""));
        s.save_to(&f).expect("save");

        let back = RecentStore::load_from(&f);
        assert_eq!(back, s, "the history did not round-trip");
        // newest first, across kinds
        assert_eq!(back.items()[0].kind, RecentKind::Model);
        // the label aids are all there, so no disk access is needed to render a row
        let v = back.of_kind(RecentKind::Variant).next().expect("the variant entry is gone");
        assert_eq!(v.title, "Tower of Doom");
        assert_eq!(v.game, "Reach");
        assert_eq!(v.name, "a b c.mvar", "a path with spaces lost its file name");
        assert_eq!(v.menu_label(), "Reach · Tower of Doom");
        assert!(v.hover().contains("/v/a b c.mvar"), "the hover must carry the full path: {}", v.hover());
        // a model has no game, so its row is just the file name
        let m = back.of_kind(RecentKind::Model).next().unwrap();
        assert_eq!(m.menu_label(), "ship.obj");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Re-opening something already in the list must MOVE it to the front (not duplicate it),
    /// and must refresh its label — a variant renamed since is listed under its new name.
    #[test]
    fn reopening_an_entry_moves_it_to_the_front() {
        let mut s = RecentStore::default();
        s.note(item(RecentKind::Variant, "/v/one.mvar", "Reach", "One"));
        s.note(item(RecentKind::Variant, "/v/two.mvar", "Reach", "Two"));
        s.note(item(RecentKind::Variant, "/v/three.mvar", "Reach", "Three"));
        s.note(item(RecentKind::Variant, "/v/one.mvar", "Reach", "One, renamed"));
        let titles: Vec<&str> = s.of_kind(RecentKind::Variant).map(|i| i.title.as_str()).collect();
        assert_eq!(titles, vec!["One, renamed", "Three", "Two"], "re-open did not reorder");
        assert_eq!(s.len(), 3, "re-opening an entry duplicated it");
    }

    /// De-duplication is per (kind, path): the same file opened twice is one row, but the same
    /// path recorded as two different kinds is two rows (nothing in the app does that; the
    /// invariant keeps the kinds independent).
    #[test]
    fn dedups_by_kind_and_path_and_caps_each_kind() {
        let mut s = RecentStore::default();
        for i in 0..PER_KIND_CAP * 3 {
            s.note(item(RecentKind::Model, &format!("/m/model{i}.obj"), "", ""));
        }
        assert_eq!(s.of_kind(RecentKind::Model).count(), PER_KIND_CAP, "the model list blew past the cap");
        // the cap drops the OLDEST, so the newest is still first
        assert_eq!(s.items()[0].name, format!("model{}.obj", PER_KIND_CAP * 3 - 1));

        // a flood of models must not evict the variants — the cap is per kind
        s.note(item(RecentKind::Variant, "/v/keep.mvar", "Reach", "Keep me"));
        for i in 0..PER_KIND_CAP * 2 {
            s.note(item(RecentKind::Model, &format!("/m/more{i}.obj"), "", ""));
        }
        assert_eq!(s.of_kind(RecentKind::Variant).count(), 1, "models evicted the variant history");

        // same path, same kind, twice = one row; same path under two kinds = two rows
        let mut t = RecentStore::default();
        t.note(item(RecentKind::Variant, "/v/x.mvar", "Reach", ""));
        t.note(item(RecentKind::Variant, "/v/x.mvar", "Reach", ""));
        assert_eq!(t.len(), 1);
        t.note(item(RecentKind::Film, "/v/x.mvar", "Reach", ""));
        assert_eq!(t.len(), 2);
        // trailing-separator spellings of the same file are the same file
        let mut u = RecentStore::default();
        u.note(item(RecentKind::Map, "/maps/a.map", "Reach", ""));
        u.note(item(RecentKind::Map, "/maps//a.map", "Reach", ""));
        assert_eq!(u.len(), 1, "two spellings of one path made two rows");
    }

    /// A corrupt, truncated or partly-unreadable file must cost at most the history: no panic,
    /// no failed launch. A wholly broken file reads as empty; a file with one bad row keeps the
    /// good ones.
    #[test]
    fn a_corrupt_or_partial_file_falls_back_instead_of_panicking() {
        let dir = tmp("corrupt");
        // not JSON at all
        let a = dir.join("garbage.json");
        std::fs::write(&a, b"\x00\x01not json at all{{{").unwrap();
        assert!(RecentStore::load_from(&a).is_empty());
        // truncated mid-object
        let b = dir.join("truncated.json");
        std::fs::write(&b, br#"{"version":1,"items":[{"kind":"variant","path":"/v/a.mv"#).unwrap();
        assert!(RecentStore::load_from(&b).is_empty());
        // a file that does not exist
        assert!(RecentStore::load_from(&dir.join("nope.json")).is_empty());
        // an empty file, and valid JSON of the wrong shape
        let c = dir.join("empty.json");
        std::fs::write(&c, b"").unwrap();
        assert!(RecentStore::load_from(&c).is_empty());
        let d = dir.join("wrongshape.json");
        std::fs::write(&d, b"[1,2,3]").unwrap();
        assert!(RecentStore::load_from(&d).is_empty());
        // PARTIAL: one unreadable row (unknown kind), one with no path, one good row
        let e = dir.join("partial.json");
        std::fs::write(
            &e,
            br#"{"version":99,"items":[
                {"kind":"hologram","path":"/v/future.mvar"},
                {"kind":"variant","path":""},
                {"nonsense":true},
                {"kind":"variant","path":"/v/good.mvar","name":"good.mvar","game":"Halo 3","title":"Still here"}
            ]}"#,
        )
        .unwrap();
        let s = RecentStore::load_from(&e);
        assert_eq!(s.len(), 1, "the readable row was lost with the bad ones");
        assert_eq!(s.items()[0].title, "Still here");
        assert_eq!(s.items()[0].game, "Halo 3");
        // a row written by an older build (no name/game/title) still renders
        let f = dir.join("minimal.json");
        std::fs::write(&f, br#"{"items":[{"kind":"map","path":"/maps/old.map"}]}"#).unwrap();
        let s = RecentStore::load_from(&f);
        assert_eq!(s.len(), 1);
        assert_eq!(s.items()[0].menu_label(), "old.map", "an entry with no stored name must fall back to the path");
        // a hand-edited file over the cap / with duplicates loads as a well-formed history
        let g = dir.join("overcap.json");
        let rows: Vec<String> = (0..PER_KIND_CAP * 2)
            .map(|i| format!(r#"{{"kind":"model","path":"/m/{}.obj"}}"#, i % 3))
            .collect();
        std::fs::write(&g, format!(r#"{{"items":[{}]}}"#, rows.join(","))).unwrap();
        let s = RecentStore::load_from(&g);
        assert_eq!(s.len(), 3, "duplicates / over-cap rows were not normalised on load");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The missing-file path: an entry whose file has been moved or deleted is detectable, can be
    /// dropped by itself, and the sweep removes exactly the gone ones.
    #[test]
    fn missing_files_are_detected_and_removable() {
        let dir = tmp("missing");
        let here = dir.join("here.mvar");
        std::fs::write(&here, b"x").unwrap();
        let gone = dir.join("gone.mvar");
        std::fs::write(&gone, b"x").unwrap();

        let mut s = RecentStore::default();
        s.note(RecentItem::new(RecentKind::Variant, &here, "Reach", "Here"));
        s.note(RecentItem::new(RecentKind::Variant, &gone, "Reach", "Gone"));
        assert!(s.items().iter().all(|i| i.exists()), "both files are on disk");

        // move it away — exactly what a user does between sessions
        let moved = dir.join("moved-elsewhere.mvar");
        std::fs::rename(&gone, &moved).unwrap();
        let missing: Vec<&str> = s.items().iter().filter(|i| !i.exists()).map(|i| i.title.as_str()).collect();
        assert_eq!(missing, vec!["Gone"], "the moved file was not reported missing");

        // dropping one entry by path leaves the rest alone
        let mut t = s.clone();
        assert!(t.remove(RecentKind::Variant, &gone.to_string_lossy()));
        assert_eq!(t.len(), 1);
        assert_eq!(t.items()[0].title, "Here");
        assert!(!t.remove(RecentKind::Variant, &gone.to_string_lossy()), "removing a gone entry twice must report nothing");

        // the sweep removes exactly the missing ones
        let removed = s.remove_missing();
        assert_eq!(removed, 1);
        assert_eq!(s.len(), 1);
        assert_eq!(s.items()[0].title, "Here");

        // and a cleared list is empty but still usable
        s.clear();
        assert!(s.is_empty());
        s.note(RecentItem::new(RecentKind::Variant, &here, "Reach", "Here"));
        assert_eq!(s.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The browsers' sidebars are driven from this one history: the folder list is the parents of
    /// the recent entries of that kind, newest first and de-duplicated.
    #[test]
    fn folders_and_paths_feed_the_browser_sidebars() {
        let mut s = RecentStore::default();
        s.note(item(RecentKind::Variant, "/a/one.mvar", "Reach", ""));
        s.note(item(RecentKind::Variant, "/b/two.mvar", "Reach", ""));
        s.note(item(RecentKind::Variant, "/a/three.mvar", "Reach", ""));
        s.note(item(RecentKind::Model, "/models/ship.obj", "", ""));
        let dirs = s.folders_of_kind(RecentKind::Variant);
        assert_eq!(dirs, vec![PathBuf::from("/a"), PathBuf::from("/b")], "folders must be newest-first and unique");
        assert!(!dirs.contains(&PathBuf::from("/models")), "a model folder leaked into the variant list");
        assert_eq!(s.paths_of_kind(RecentKind::Model), vec![PathBuf::from("/models/ship.obj")]);
    }

    /// A very long variant title must not stretch the menu off the screen.
    #[test]
    fn long_names_are_shortened_for_the_menu() {
        let long = "A".repeat(200);
        let i = item(RecentKind::Variant, "/v/x.mvar", "Halo 2 Anniversary", &long);
        let label = i.menu_label();
        assert!(label.chars().count() < 70, "menu row too long: {} chars", label.chars().count());
        assert!(label.ends_with('…'));
        assert!(label.starts_with("Halo 2 Anniversary · "));
        // the hover still carries the whole path
        assert!(i.hover().contains("/v/x.mvar"));
    }
}
