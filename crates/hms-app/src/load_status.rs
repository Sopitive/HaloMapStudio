//! #loadstatus  WHETHER A MAP IS OPEN, LOADING, NOT STARTED, OR FAILED — as four states.
//!
//! # The hole this fills
//!
//! Every scene verb in the script surface answers "no map open" for the **whole duration of a
//! load**, not just before one starts. The reason is structural rather than an oversight:
//! `App::load_map` hands the `SceneController` to a worker thread, so `cache_handle()` is `None`
//! from the moment the load begins until `LoadMsg::Done` hands it back. A scripted caller
//! therefore cannot tell these three situations apart, and they need completely different
//! responses:
//!
//! * **nothing was asked for** — the caller forgot to open a map;
//! * **a load is in flight** — wait longer;
//! * **a load ended without a map** — stop waiting and report the reason.
//!
//! Measured consequence: a live gate of mine polled `play phys` for **240 seconds**, read "no map
//! open" every time, and concluded the load had made *zero progress* — a claim its own evidence
//! could not support, because "not started" and "still loading" are the same string. The right
//! conclusion was only "it had not finished yet". A probe that collapses distinct states into one
//! answer does not just fail to inform, it actively misleads.
//!
//! # Report, never change behaviour
//!
//! [`classify`] is a pure function of three booleans the app already knows. Nothing here touches
//! the load path, and nothing here can make a load behave differently from an unscripted one —
//! a diagnostic that perturbs what it measures would be worse than the hole it fills.
//!
//! # Why there is no blocking `wait`
//!
//! The obvious API would be a `wait` that blocks until the load finishes. In the windowed app it
//! **cannot exist**, and the reason is worth writing down: scripted commands are executed on the
//! UI thread (the command server hands each one over a channel and `App::update` drains them), and
//! the load is drained by `App::drive_load`, also on the UI thread, once per frame. A `wait` that
//! blocked there would stop the frame that performs the drain it is waiting for — a deadlock, not
//! a slow path. So the honest shape is a cheap state a caller polls, which is this.

/// What a map load is doing, as four distinguishable states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadState {
    /// No load has been asked for in this session. The caller has not opened a map.
    NotStarted,
    /// A worker is streaming a map in. Poll again.
    InFlight,
    /// A map is open and the scene can be queried.
    Done,
    /// A load was asked for, is no longer in flight, and left no open map.
    Failed,
}

impl LoadState {
    /// The one word a script matches on. Stable: callers compare against these.
    pub fn word(self) -> &'static str {
        match self {
            LoadState::NotStarted => "not-started",
            LoadState::InFlight => "in-flight",
            LoadState::Done => "done",
            LoadState::Failed => "failed",
        }
    }

    /// Is it worth polling again?
    pub fn pending(self) -> bool {
        self == LoadState::InFlight
    }
}

/// Classify a load from what the app knows.
///
/// * `attempted` — a map load has been requested in this session (the app has a map path).
/// * `in_flight` — a worker still owns the scene (any game's load channel is live).
/// * `open` — a scene with a readable cache exists.
///
/// **`open` wins over `in_flight`**: a load that has handed the scene back is done even if some
/// later stage is still streaming, because "can I query the scene" is the question a caller is
/// actually asking. And `in_flight` wins over `attempted`, so a *re*-load of a map that is already
/// open reports `InFlight` rather than `Done` — the scene is not the caller's to query while a
/// worker owns it.
pub fn classify(attempted: bool, in_flight: bool, open: bool) -> LoadState {
    match (attempted, in_flight, open) {
        (_, true, _) => LoadState::InFlight,
        (_, false, true) => LoadState::Done,
        (true, false, false) => LoadState::Failed,
        (false, false, false) => LoadState::NotStarted,
    }
}

/// One line for the script surface: the state word, then what is known about it.
pub fn describe(state: LoadState, map: &str, note: &str) -> String {
    let what = match state {
        LoadState::NotStarted => "no map load has been requested in this session".to_string(),
        LoadState::InFlight => format!("loading {} — a worker owns the scene; poll again", pretty(map)),
        LoadState::Done => format!("{} is open and queryable", pretty(map)),
        LoadState::Failed => format!("the load of {} ended without an open map", pretty(map)),
    };
    let mut s = format!("{} | {what}", state.word());
    if !note.trim().is_empty() {
        s += &format!(" | {}", note.trim());
    }
    s
}

fn pretty(map: &str) -> String {
    if map.trim().is_empty() {
        "(no map named)".into()
    } else {
        map.rsplit(['\\', '/']).next().unwrap_or(map).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #loadstatus-gate  THE FOUR STATES ARE FOUR DIFFERENT ANSWERS.
    ///
    /// This is the gate on the actual bug: "not started" and "in flight" used to be the same
    /// string, which is what let a 240-second poll be read as "zero progress". If any two of
    /// these ever collapse to one word again, this fails.
    #[test]
    fn the_four_states_are_distinguishable() {
        let not_started = classify(false, false, false);
        let in_flight = classify(true, true, false);
        let done = classify(true, false, true);
        let failed = classify(true, false, false);
        assert_eq!(not_started, LoadState::NotStarted);
        assert_eq!(in_flight, LoadState::InFlight);
        assert_eq!(done, LoadState::Done);
        assert_eq!(failed, LoadState::Failed);
        // Four states, four distinct words — the property that was missing.
        let words = [not_started.word(), in_flight.word(), done.word(), failed.word()];
        let mut uniq = words.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 4, "two states share a word: {words:?}");
        // And the descriptions differ too, because a caller may be reading prose.
        let texts: Vec<String> = [not_started, in_flight, done, failed]
            .iter()
            .map(|s| describe(*s, "maps\\forge_halo.map", ""))
            .collect();
        let mut t = texts.clone();
        t.sort();
        t.dedup();
        assert_eq!(t.len(), 4, "two states describe themselves identically: {texts:?}");
        // Only one of them is worth polling on.
        assert!(in_flight.pending());
        for s in [not_started, done, failed] {
            assert!(!s.pending(), "{s:?} must not ask a caller to keep polling");
        }
    }

    /// #loadstatus-gate  A LOAD IN FLIGHT IS NEVER REPORTED AS DONE, even over an already-open
    /// map. Re-loading hands the scene to a worker, so the scene is not the caller's to query.
    #[test]
    fn a_reload_over_an_open_map_is_in_flight_not_done() {
        assert_eq!(classify(true, true, true), LoadState::InFlight);
        assert_eq!(classify(true, true, false), LoadState::InFlight);
    }

    /// #loadstatus-gate  "Failed" requires an ATTEMPT. A fresh session with no map is not a
    /// failure, and calling it one would send a caller chasing an error that does not exist.
    #[test]
    fn a_fresh_session_is_not_a_failure() {
        assert_eq!(classify(false, false, false), LoadState::NotStarted);
        assert_ne!(classify(false, false, false), LoadState::Failed);
    }

    /// #loadstatus-gate  THE VERB REACHES IT. The four states are useless if the only way to
    /// ask is a function nobody calls, so this goes through the REAL parser — the same
    /// `script::parse_line` the command server and the GUI both use — and checks that the two
    /// spellings of the question arrive at the status command while an actual load does not.
    ///
    /// This is the half of the gate that would have caught "the diagnostic exists but is
    /// unreachable", which is the failure this project has been burned by twice.
    #[test]
    fn the_status_question_reaches_the_app_through_the_real_parser() {
        use crate::script::EditorCommand as C;
        let got = |line: &str| crate::script::parse_line(line).unwrap_or_else(|e| panic!("{line}: {e}"));
        assert!(matches!(got("loadstatus"), Some(C::Wait)), "`loadstatus` must ask for the state");
        assert!(matches!(got("load_status"), Some(C::Wait)));
        assert!(matches!(got("loadmap status"), Some(C::Wait)), "`loadmap status` must ask, not load");
        assert!(matches!(got("LOADMAP Status"), Some(C::Wait)), "the spelling is case-insensitive");
        // ...and a real load is still a real load: if this ever matched the status command, asking
        // for the state would START a load named "status", which is the opposite of read-only.
        match got("loadmap forge_halo") {
            Some(C::LoadMap(n)) => assert_eq!(n, "forge_halo"),
            other => panic!("`loadmap forge_halo` must still load: {other:?}"),
        }
        // `wait` keeps working as the same question, which is what existing scripts call.
        assert!(matches!(got("wait"), Some(C::Wait)));
    }

    /// The description names the map by LEAF, and survives an empty name.
    #[test]
    fn the_description_names_the_map_and_tolerates_none() {
        let s = describe(LoadState::InFlight, "D:\\maps\\forge_halo.map", "42% decoded");
        assert!(s.contains("forge_halo.map"), "{s}");
        assert!(!s.contains("D:\\"), "the full path is noise in a status line: {s}");
        assert!(s.contains("42% decoded"), "{s}");
        let e = describe(LoadState::NotStarted, "", "");
        assert!(e.starts_with("not-started"), "{e}");
    }
}
