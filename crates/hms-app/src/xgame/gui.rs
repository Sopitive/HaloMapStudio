//! "Convert variant to another game…" - the modal that drives [`super::convert`] from the app,
//! plus a one-call preview that loads the converted file on the target canvas so a user can LOOK
//! at the layout before committing it. #xgame
//!
//! The dialog is deliberately a thin shell over the headless path: it builds the same
//! [`super::convert::Options`], calls the same [`super::convert::convert_to_file`], and shows the
//! same [`super::convert::Report`] the `convertvariant` script verb prints. Whatever a user sees
//! here, a script reproduces.

use std::path::PathBuf;

use eframe::egui;

use crate::mapcat::Game;

use super::catalog::Catalog;
use super::convert::{self, Options, Report, XVariant};
use super::table::{Confidence, Table};

/// Everything the modal remembers between frames.
pub struct ConvertState {
    pub target_game: Game,
    pub target_canvas: String,
    pub out_path: String,
    pub offset: [f32; 3],
    pub yaw_deg: f32,
    pub layout_scale: f32,
    pub keep_origin: bool,
    pub enforce_quotas: bool,
    pub enforce_budget: bool,
    pub min_conf: Confidence,
    /// Source read from the currently-open variant (or an explicitly picked file).
    pub source: Option<XVariant>,
    pub source_path: Option<PathBuf>,
    /// The last dry run: the report plus the bytes it produced (so Convert writes exactly what
    /// the user previewed).
    pub preview: Option<(Report, Vec<u8>)>,
    pub status: String,
    pub show_all_rows: bool,
    /// Canvas lists per game, filled from the catalog on open.
    pub canvases: Vec<(Game, Vec<String>)>,
    pub catalog: Option<Catalog>,
    pub table: Option<Table>,
    /// Set when the user asked for a file to be opened on the target canvas (the scratch preview
    /// or the file just saved).
    pub want_preview_load: Option<PathBuf>,
    /// Pin the timestamp the converter stamps into the file. `None` — always, in the app — reads
    /// the clock, which is what a real save does. A test sets it so two conversions of the same
    /// variant are byte-identical and can be compared with no exemptions; see
    /// [`super::convert::Options::stamp_unix`].
    pub stamp_unix: Option<u64>,
}

impl Default for ConvertState {
    fn default() -> Self {
        ConvertState {
            target_game: Game::Halo4, target_canvas: String::new(), out_path: String::new(),
            offset: [0.0; 3], yaw_deg: 0.0, layout_scale: 1.0, keep_origin: false,
            enforce_quotas: true, enforce_budget: true, min_conf: Confidence::Approximate,
            source: None, source_path: None, preview: None, status: String::new(), show_all_rows: false,
            canvases: Vec::new(), catalog: None, table: None, want_preview_load: None,
            stamp_unix: None,
        }
    }
}

impl ConvertState {
    /// Load the catalog + table and the canvas lists (cheap; done when the dialog opens).
    pub fn ensure_loaded(&mut self) {
        if self.catalog.is_none() {
            let cat = Catalog::load();
            self.canvases = super::GAMES.iter().map(|&g| (g, cat.canvases(g))).collect();
            if self.target_canvas.is_empty() {
                if let Some((_, c)) = self.canvases.iter().find(|(g, c)| *g == self.target_game && !c.is_empty()) {
                    self.target_canvas = c[0].clone();
                }
            }
            self.catalog = Some(cat);
        }
        if self.table.is_none() { self.table = Some(Table::load()); }
    }
    pub fn canvas_list(&self, g: Game) -> &[String] {
        self.canvases.iter().find(|(x, _)| *x == g).map(|(_, c)| c.as_slice()).unwrap_or(&[])
    }
    pub fn options(&self) -> Options {
        Options {
            offset: self.offset, yaw_deg: self.yaw_deg, layout_scale: self.layout_scale, keep_origin: self.keep_origin,
            min_confidence: self.min_conf, enforce_quotas: self.enforce_quotas, enforce_budget: self.enforce_budget,
            template: None,
            stamp_unix: self.stamp_unix,
        }
    }
    /// Read the source variant from `path`.
    pub fn load_source(&mut self, path: PathBuf) {
        self.ensure_loaded();
        let Some(cat) = &self.catalog else { return };
        match convert::read_variant(cat, &path) {
            Ok(v) => {
                self.status = format!("source: {} objects from {} ({})", v.objects.len(), v.canvas, v.game_short);
                if self.target_game == v.game().unwrap_or(Game::Reach) {
                    // default to a DIFFERENT title than the source
                    self.target_game = super::GAMES.iter().copied().find(|&g| Some(g) != v.game() && !self.canvas_list(g).is_empty()).unwrap_or(self.target_game);
                    self.target_canvas = self.canvas_list(self.target_game).first().cloned().unwrap_or_default();
                }
                self.source_path = Some(path);
                self.source = Some(v);
                self.preview = None;
                self.refresh_out_path();
            }
            Err(e) => { self.status = e; self.source = None; }
        }
    }
    fn refresh_out_path(&mut self) {
        let Some(src) = &self.source else { return };
        let stem: String = src.title.chars().filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '_' || *c == '-').collect();
        let stem = if stem.trim().is_empty() { "converted".to_string() } else { stem.trim().replace(' ', "_") };
        // the game's own `map_variants` folder (never `hopper_map_variants`: MCC does not list a
        // matchmaking-folder file in Custom Games)
        let dir = crate::mapcat::variant_dirs_of(self.target_game).into_iter().find(|d| d.ends_with("map_variants"));
        let name = format!("{stem}_{}.mvar", self.target_game.short_name());
        self.out_path = dir.map(|d| d.join(&name)).unwrap_or_else(|| PathBuf::from(&name)).to_string_lossy().into_owned();
    }
    /// Run the conversion WITHOUT writing, so the report can be reviewed first.
    pub fn dry_run(&mut self) {
        self.ensure_loaded();
        let (Some(cat), Some(tab), Some(src)) = (&self.catalog, &self.table, &self.source) else {
            self.status = "pick a source .mvar first".into(); return;
        };
        if self.target_canvas.is_empty() { self.status = "pick a target canvas".into(); return; }
        match convert::convert(cat, tab, src, self.target_game, &self.target_canvas, &self.options()) {
            Ok(c) => { self.status = c.report.summary(); self.preview = Some((c.report, c.bytes)); }
            Err(e) => { self.status = e; self.preview = None; }
        }
    }
    /// Write the previewed bytes to a SCRATCH file in the settings folder and ask the app to open
    /// it on the target canvas, so the layout can be looked at before anything is committed.
    pub fn preview_in_viewport(&mut self) -> Option<PathBuf> {
        let (rep, bytes) = self.preview.as_ref()?;
        let dir = crate::project::settings_dir()?;
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("xgame_preview.mvar");
        match std::fs::write(&p, bytes) {
            Ok(()) => { self.status = format!("preview: {} objects on {} ({:.1}% solid, {} approximate) - nothing saved yet", rep.mapped(), rep.dst_canvas, rep.coverage(), rep.approximate()); Some(p) }
            Err(e) => { self.status = format!("{}: {e}", p.display()); None }
        }
    }

    /// Write the bytes the last dry run produced.
    pub fn commit(&mut self) -> Option<PathBuf> {
        let (rep, bytes) = self.preview.as_ref()?;
        let out = PathBuf::from(self.out_path.trim());
        if out.as_os_str().is_empty() { self.status = "give the output a path".into(); return None; }
        if let Some(d) = out.parent() { let _ = std::fs::create_dir_all(d); }
        match std::fs::write(&out, bytes) {
            Ok(()) => {
                self.status = format!("wrote {} - {} objects, {:.1}% solid, {} approximate to check",
                    out.display(), rep.mapped(), rep.coverage(), rep.approximate());
                Some(out)
            }
            Err(e) => { self.status = format!("{}: {e}", out.display()); None }
        }
    }
}

/// The modal. Returns true while it should stay open. `pick_source` is set when the user asked
/// for a file browser, `load_result` when they asked to open the converted file in the viewport.
pub fn ui(st: &mut ConvertState, ctx: &egui::Context, open: &mut bool, pick_source: &mut bool) {
    st.ensure_loaded();
    let mut win_open = true;
    let mut do_dry = false;
    let mut do_commit = false;
    let mut do_preview = false;
    egui::Window::new("Convert variant to another game")
        .open(&mut win_open)
        .default_width(760.0)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.screen_rect().center())
        .collapsible(false)
        .show(ctx, |ui| {
            ui.label("Take a Forge variant authored in one Halo title and re-create it on a Forge canvas of another. \
                      Each placed object is mapped to the nearest equivalent in the target game's palette, by Forge \
                      category, name and above all render-model SIZE.");
            ui.separator();

            // --- source ---
            ui.horizontal(|ui| {
                ui.strong("Source");
                match (&st.source, &st.source_path) {
                    (Some(v), Some(p)) => { ui.label(format!("{} - {} objects on {} ({})", p.file_name().unwrap_or_default().to_string_lossy(), v.objects.len(), v.canvas, v.game_short)); }
                    _ => { ui.label("none - open a .mvar, or pick one here"); }
                }
                if ui.button("Pick .mvar…").clicked() { *pick_source = true; }
            });

            // --- target ---
            ui.horizontal(|ui| {
                ui.strong("Target game");
                let src_game = st.source.as_ref().and_then(|v| v.game());
                for g in super::GAMES {
                    let has = !st.canvas_list(g).is_empty();
                    let same = Some(g) == src_game;
                    ui.add_enabled_ui(has && !same, |ui| {
                        if ui.selectable_label(st.target_game == g, g.display_name())
                            .on_hover_text(if !has { "no palette for this title in the catalog" } else if same { "that is the source title" } else { "" })
                            .clicked()
                        {
                            st.target_game = g;
                            st.target_canvas = st.canvas_list(g).first().cloned().unwrap_or_default();
                            st.preview = None;
                            st.refresh_out_path();
                        }
                    });
                }
            });
            ui.horizontal(|ui| {
                ui.strong("Target canvas");
                let list: Vec<String> = st.canvas_list(st.target_game).to_vec();
                egui::ComboBox::from_id_salt("xgame_canvas").selected_text(if st.target_canvas.is_empty() { "(pick)" } else { &st.target_canvas }).show_ui(ui, |ui| {
                    for c in &list {
                        if ui.selectable_label(&st.target_canvas == c, c).clicked() { st.target_canvas = c.clone(); st.preview = None; st.refresh_out_path(); }
                    }
                });
                ui.label(format!("({} canvases)", list.len()));
            });

            // --- placement ---
            ui.separator();
            ui.label("Canvases differ wildly in size and origin, so the source layout is re-centred inside the target \
                      canvas automatically (over the target's own build area, floor to floor). Nudge it here.");
            ui.horizontal(|ui| {
                ui.label("Offset x/y/z");
                for k in 0..3 { if ui.add(egui::DragValue::new(&mut st.offset[k]).speed(0.5).suffix(" wu")).changed() { st.preview = None; } }
                ui.separator();
                ui.label("Yaw");
                if ui.add(egui::DragValue::new(&mut st.yaw_deg).speed(1.0).range(-180.0..=180.0).suffix("\u{b0}")).changed() { st.preview = None; }
                ui.separator();
                ui.label("Layout scale");
                if ui.add(egui::DragValue::new(&mut st.layout_scale).speed(0.01).range(0.1..=4.0)).on_hover_text("scales the SPACING of the layout, never the objects").changed() { st.preview = None; }
            });
            ui.horizontal(|ui| {
                if ui.checkbox(&mut st.keep_origin, "Keep the source world coordinates").changed() { st.preview = None; }
                if ui.checkbox(&mut st.enforce_quotas, "Respect per-type quotas").changed() { st.preview = None; }
                if ui.checkbox(&mut st.enforce_budget, "Respect the canvas budget").changed() { st.preview = None; }
                ui.separator();
                ui.label("Accept down to");
                for c in [Confidence::Exact, Confidence::Close, Confidence::Approximate] {
                    if ui.selectable_label(st.min_conf == c, c.name()).clicked() { st.min_conf = c; st.preview = None; }
                }
            });

            // --- actions ---
            ui.separator();
            ui.horizontal(|ui| {
                ui.add_enabled_ui(st.source.is_some() && !st.target_canvas.is_empty(), |ui| {
                    if ui.button("Preview conversion").clicked() { do_dry = true; }
                });
                ui.add_enabled_ui(st.preview.is_some(), |ui| {
                    if ui.button("Show in viewport")
                        .on_hover_text("load the converted layout on the target canvas WITHOUT saving it, so you can look at it first")
                        .clicked() { do_preview = true; }
                    if ui.button("Save As…").on_hover_text("writes exactly the file the report above describes").clicked() { do_commit = true; }
                });
                ui.label("Output");
                ui.add(egui::TextEdit::singleline(&mut st.out_path).desired_width(360.0));
            });
            if !st.status.is_empty() {
                ui.separator();
                ui.label(egui::RichText::new(&st.status).monospace());
            }

            // --- the report ---
            if let Some((rep, _)) = &st.preview {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong(format!("{:.1}% SOLID", rep.coverage()))
                        .on_hover_text("exact + close matches, as a share of the source objects. Approximate matches are NOT counted here: they are the nearest the target palette could offer and often are not the right piece.");
                    ui.label(format!("{} exact + {} close", rep.exact(), rep.close()));
                    if rep.approximate() > 0 {
                        ui.colored_label(egui::Color32::from_rgb(210, 170, 120),
                            format!("{} approximate ({:.0}%) - check these", rep.approximate(), rep.approximate_pct()));
                    }
                    if rep.dropped() > 0 { ui.colored_label(egui::Color32::from_rgb(220, 110, 110), format!("{} dropped", rep.dropped())); }
                    ui.checkbox(&mut st.show_all_rows, "show every object");
                });
                if rep.rotated > 0 {
                    ui.label(egui::RichText::new(format!("{} piece(s) were turned because the target model is authored along different local axes.", rep.rotated)).weak().small());
                }
                for w in &rep.warnings { ui.colored_label(egui::Color32::from_rgb(230, 170, 60), format!("warning: {w}")); }
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    ui.label(egui::RichText::new(rep.by_kind()).monospace());
                    ui.separator();
                    for o in &rep.objects {
                        // always show the rows that need attention; hide only the clean ones
                        if !st.show_all_rows && o.status != convert::Status::Dropped && o.conf <= Confidence::Close { continue; }
                        let col = match (o.status, o.conf) {
                            (convert::Status::Dropped, _) => egui::Color32::from_rgb(220, 110, 110),
                            (_, Confidence::Exact) => egui::Color32::from_rgb(130, 200, 130),
                            (_, Confidence::Close) => egui::Color32::from_rgb(200, 200, 140),
                            _ => egui::Color32::from_rgb(210, 170, 120),
                        };
                        let d = if o.delta == [0.0; 3] { String::new() } else { format!("  size {:+.1}/{:+.1}/{:+.1} wu", o.delta[0], o.delta[1], o.delta[2]) };
                        ui.colored_label(col, egui::RichText::new(format!("[{}] {:<30} -> {:<30} {}{}  {}",
                            o.status.name(), o.src_display, if o.dst_display.is_empty() { "-" } else { &o.dst_display }, o.conf.name(), d, o.note)).monospace());
                    }
                });
                ui.separator();
                ui.label(egui::RichText::new(format!("template: {}", rep.template.display())).weak().small());
                ui.label(egui::RichText::new("Edit the mapping without rebuilding: copy crates/hms-app/src/xgame/equivalence.tsv to <settings>/xgame_equivalence.tsv and change a row.").weak().small());
            }
        });
    if do_dry { st.dry_run(); }
    if do_preview { st.want_preview_load = st.preview_in_viewport(); }
    if do_commit {
        if let Some(p) = st.commit() { st.want_preview_load = Some(p); }
    }
    *open = win_open;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dialog's STATE is pure (no egui), so the behaviour behind the buttons is testable:
    /// opening it loads the catalog and the canvas lists, picking a source switches the target
    /// away from the source title, and "Preview conversion" produces exactly the bytes that
    /// "Save As…" would write.
    #[test]
    fn dialog_state_drives_the_same_conversion_as_the_script_verb() {
        let mut st = ConvertState::default();
        st.ensure_loaded();
        let Some(cat) = st.catalog.as_ref() else { return };
        if cat.objects.is_empty() { return; } // no catalog on this machine
        assert!(st.table.is_some());
        assert!(super::super::GAMES.iter().any(|&g| !st.canvas_list(g).is_empty()));

        // A shipped Reach FORGE variant, if one is installed. A Firefight / campaign variant is
        // not a Forge layout: its canvas has no Forge palette, so nothing resolves and there is
        // nothing to convert - the dialog says so, and this test is not about that path.
        let mut src: Option<PathBuf> = None;
        let mut dirs = crate::mapcat::variant_dirs();
        dirs.sort_by_key(|d| !d.to_string_lossy().contains("hopper"));
        'outer: for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar"))).collect();
            files.sort();
            for f in files {
                if crate::h4::mvar::is_h4_variant(&f) || crate::h3::mvar::is_h3_variant(&f) { continue; }
                if convert::read_variant(cat, &f).map_or(false, |v| (50..=450).contains(&v.objects.len())) { src = Some(f); break 'outer; }
            }
        }
        let Some(src) = src else { return };
        st.load_source(src.clone());
        let Some(v) = st.source.as_ref() else { eprintln!("source not readable: {}", st.status); return };
        assert_eq!(v.game(), Some(Game::Reach));
        assert_ne!(st.target_game, Game::Reach, "the dialog must default to a DIFFERENT title than the source");
        assert!(!st.target_canvas.is_empty());
        assert!(st.out_path.ends_with(".mvar"), "{}", st.out_path);

        // PIN THE CLOCK. Everything about a converted file is a pure function of (source, canvas,
        // options) except the timestamp the converter stamps into it — and that stamp is mirrored
        // into the bitstream and BOTH `chdr` copies, so a single second between two conversions
        // moved twenty-odd bytes, not four. This gate used to compare the two vectors outright and
        // failed whenever the second ticked; the fix is to make the conversion reproducible, not
        // to forgive differences. With the instant supplied, the comparison below is exact and
        // has NO exemptions: every byte must match.
        st.stamp_unix = Some(1_700_000_000);

        // the dry run fills the preview, and the preview's report is what the UI shows
        st.target_game = Game::Halo4;
        let Some(c) = st.canvas_list(Game::Halo4).iter().find(|c| *c == "ca_forge_ravine").cloned()
            .or_else(|| st.canvas_list(Game::Halo4).first().cloned()) else { return };
        st.target_canvas = c.clone();
        st.dry_run();
        let Some((rep, bytes)) = st.preview.as_ref() else { eprintln!("no preview: {}", st.status); return };
        assert!(rep.mapped() > 0, "{}", st.status);
        assert_eq!(rep.dst_canvas, c);
        assert!(!bytes.is_empty());
        let (mapped, bytes) = (rep.mapped(), bytes.clone());

        // the SAME options through the headless path produce the SAME bytes, byte for byte
        let cat = st.catalog.as_ref().unwrap();
        let tab = st.table.as_ref().unwrap();
        let again = convert::convert(cat, tab, st.source.as_ref().unwrap(), st.target_game, &st.target_canvas, &st.options()).unwrap();
        assert_eq!(again.bytes, bytes, "the dialog and the script verb must produce identical files");
        assert_eq!(again.report.mapped(), mapped);

        // ...and the pin is what makes that exact. Verified by hand with a 1.4 s `sleep` between
        // the two conversions above: with the stamp pinned the files are still identical, which is
        // precisely the case that used to break this gate. One second of stamp moves 22 bytes
        // (measured), so the assertion below also proves the pin reaches every mirror of it rather
        // than only the one 4-byte field the old exemption knew about.
        let mut later = st.options();
        later.stamp_unix = Some(1_700_000_001);
        let later = convert::convert(cat, tab, st.source.as_ref().unwrap(), st.target_game, &st.target_canvas, &later).unwrap();
        assert_ne!(later.bytes, bytes, "a different stamp must actually change the file, or the pin proves nothing");
        let moved = (0..bytes.len()).filter(|&i| later.bytes[i] != bytes[i]).count();
        assert!(
            moved > 4,
            "one second of stamp must move more than one 4-byte field, or the pin is not reaching every \
             mirror of it (measured: 22 bytes; it moved {moved})"
        );
        let same_instant = convert::convert(cat, tab, st.source.as_ref().unwrap(), st.target_game, &st.target_canvas, &st.options()).unwrap();
        assert_eq!(same_instant.bytes, bytes, "two conversions at the same pinned instant must be identical");

        // changing a placement control invalidates the preview, so Save As can never write a
        // file that does not match the report on screen
        st.yaw_deg = 45.0;
        // (the UI does this on edit; mirror it here)
        st.preview = None;
        assert!(st.preview.is_none());
        st.dry_run();
        assert!(st.preview.is_some());
        let (_, rotated) = st.preview.as_ref().unwrap();
        assert_ne!(*rotated, bytes, "a 45 degree yaw must change the file");
    }

    /// `commit` refuses an empty path and never writes without a preview.
    #[test]
    fn commit_needs_a_preview_and_a_path() {
        let mut st = ConvertState::default();
        assert!(st.commit().is_none());
        st.out_path = String::new();
        assert!(st.commit().is_none());
    }
}
