//! #use-hold  THE ACTION BUTTON'S HOLD, as the engine decides it.
//!
//! **TAPPING THE USE KEY DOES NOT GET YOU INTO A VEHICLE.** Reach asks for the action button to be
//! HELD, and the prompt it prints says so (`Hold [action] to drive Warthog`, all 36 of the
//! `enter seat string`s in `forge_halo`). Test mode used to act on the key's first frame, which is
//! what "I only have to tap E to get in vehicles" reported.
//!
//! # What the engine does (`reach_tag_test.exe`)
//!
//! * **PROVEN -- the threshold is a tag field.** `matg +0x414 player control` (140 B, one element)
//!   `+0x48 minimum action hold time [seconds]`, help text *"time that player needs to press
//!   ACTION to register as a HOLD"*. `forge_halo` authors **0.2333 s**.
//! * **PROVEN -- how it is consumed.** `player_control_get_controller_input` (`0x1E2530`) reads the
//!   action button's down-time word, compares it with `(int)(threshold * 1000.0)` milliseconds,
//!   and raises the control-flag bit **`0x200000` HOLD** (`0x100000` is merely "down"). The edge
//!   builder `sub_1401A3F20` derives a per-button nibble: bit 0 down, bit 1 held, bit 2 PRESSED
//!   edge, bit 3 released edge.
//! * **PROVEN -- which interactions take the hold.** `sub_1401A7E80` (`players_update_before_game`)
//!   runs `player_handle_interaction_held` (vehicle seats: interaction types 5, 6 and 9, plus
//!   the equipment/device cases) and the weapon pick-up / swap (`sub_1401AB600`, interaction
//!   type 2 or 10) only on `flags & 2 || flags & 0x200000` -- the HOLD bit. The PRESSED-edge bit
//!   (`0x4` / `0x400000`) is consumed by one thing, interaction type `0xB` on an object of type 1
//!   (the vehicle FLIP). A tap therefore enters nothing; releasing before the threshold never
//!   raises the bit, so nothing happens and the count restarts.
//! * **PROVEN -- once per press.** After a hold acts the handler clears the hold bits and marks
//!   the player (`player + 4 |= 0x200`), so a held button does not act again until it is released
//!   and pressed anew.
//! * **PROVEN -- boarding an enemy vehicle is the same hold.** `warthog_board` and the other
//!   `*_board` prompts are `Hold [action] ...` and there is no second duration field.
//! * **INFERRED -- units.** The down-time word is `input_windows`' per-button down counter,
//!   compared against milliseconds; it is sampled once per game tick, so the effective rule is
//!   "the first 60 Hz tick whose elapsed time reaches the threshold", which is what [`HoldTimer`]
//!   counts. 0.2333 s -> `(int)233 ms`; 14 ticks = 233.3 ms is the first tick that reaches it.
//! * **UNRESOLVED -- no progress meter.** No `chud` widget or `matg` field draws a hold-progress
//!   bar for it, so none is drawn here. The engine's only help for a player who taps is the
//!   `training_vehiclehold` / `training_swaphold` / `training_devicehold` training messages.
//! * **UNRESOLVED -- leaving.** Getting OUT is handled by the unit while seated and not by
//!   `player_handle_interaction_held`; it stays a press.

use hms_native::{CacheHandle, NativeDll};

/// The engine's tick rate. Every game-time quantity in HMS is a count of these.
pub const TICK_HZ: f32 = 60.0;

/// `matg`'s `player control` block.
const MATG_PLAYER_CONTROL: u32 = 0x414;
const PLAYER_CONTROL_SIZE: u32 = 0x8C;
/// ...and its `minimum action hold time [seconds]`.
const MIN_ACTION_HOLD_TIME: usize = 0x48;

/// The authored hold time, seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionHold {
    pub seconds: f32,
}

impl ActionHold {
    /// `forge_halo`'s authored value, used when the cache has no readable `matg` (a headless test
    /// with no map). Measured, not chosen: 0.2333 s.
    pub const FALLBACK: ActionHold = ActionHold { seconds: 0.2333 };

    pub fn read(native: &NativeDll, cache: CacheHandle) -> ActionHold {
        let rd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let Some(matg) = native.tags_by_group(cache, "matg").into_iter().next() else { return Self::FALLBACK };
        let Some(h) = native.tag_meta(cache, matg, MATG_PLAYER_CONTROL, 8) else { return Self::FALLBACK };
        if h.len() < 8 || rd(&h, 0) == 0 {
            return Self::FALLBACK;
        }
        let Some(b) = native.tag_ptr_bytes(cache, rd(&h, 4), PLAYER_CONTROL_SIZE) else { return Self::FALLBACK };
        let seconds = f32::from_bits(rd(&b, MIN_ACTION_HOLD_TIME));
        if seconds.is_finite() && seconds >= 0.0 {
            ActionHold { seconds }
        } else {
            Self::FALLBACK
        }
    }

    /// `(int)(seconds * 1000.0)`: the engine's integer-millisecond threshold.
    pub fn threshold_ms(&self) -> u32 {
        (self.seconds * 1000.0) as u32
    }

    /// The first tick count whose elapsed milliseconds reach the threshold.
    pub fn ticks_to_hold(&self) -> u32 {
        let need = self.threshold_ms();
        (1..=600u32).find(|k| (*k as f32 * 1000.0 / TICK_HZ) as u32 >= need).unwrap_or(600)
    }
}

/// The action button's hold, counted in 60 Hz TICKS so it takes the same time at any frame rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct HoldTimer {
    /// Whole ticks of game time that have elapsed, from the sum of the frames' `dt`.
    elapsed_ticks: u64,
    clock: f64,
    /// Ticks the button has been down for, since it went down.
    pub held_ticks: u32,
    /// The hold already acted: nothing more until the button is released.
    consumed: bool,
}

impl HoldTimer {
    /// Advance by one display frame of `dt` seconds with the button `down`.
    pub fn step(&mut self, down: bool, dt: f32) {
        self.clock += f64::from(dt.clamp(0.0, 0.1));
        // A small epsilon so 1/60 s accumulated in f32 steps lands on its own tick.
        let total = (self.clock * f64::from(TICK_HZ) + 1e-6).floor() as u64;
        let ticks = (total - self.elapsed_ticks) as u32;
        self.elapsed_ticks = total;
        if down {
            self.held_ticks = self.held_ticks.saturating_add(ticks);
        } else {
            self.held_ticks = 0;
            self.consumed = false;
        }
    }

    /// Has the button been held long enough to count as a HOLD, and not yet acted?
    pub fn reached(&self, hold: &ActionHold) -> bool {
        !self.consumed && self.held_ticks >= hold.ticks_to_hold()
    }

    /// The hold acted: it cannot act again until the button is released.
    pub fn consume(&mut self) {
        self.consumed = true;
    }

    /// 0..1 through the hold, for a caller that wants to show it.
    pub fn progress(&self, hold: &ActionHold) -> f32 {
        (self.held_ticks as f32 / hold.ticks_to_hold().max(1) as f32).min(1.0)
    }
}

#[cfg(test)]
mod probes {
    use crate::reach_hud;

    /// PRINT-ONLY: the 35 `player training data` entries of `chgd` (+0x18) and `matg`'s
    /// `minimum action hold time` (+0x414 -> +0x48).
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_training_entries_and_hold_time() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let rd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let chgd = native.tags_by_group(cache, "chgd")[0];
        let h = native.tag_meta(cache, chgd, 0x18, 8).unwrap();
        let (n, ptr) = (rd(&h, 0), rd(&h, 4));
        let b = native.tag_ptr_bytes(cache, ptr, n * 20).unwrap();
        for i in 0..n as usize {
            let e = &b[i * 20..];
            println!(
                "training[{i}] '{}' max_display {} count {} disappear {} redisplay {} delay {:.2}s flags {:#x}",
                native.string_id(cache, rd(e, 0)).unwrap_or_default(),
                i16::from_le_bytes([e[4], e[5]]),
                i16::from_le_bytes([e[6], e[7]]),
                i16::from_le_bytes([e[8], e[9]]),
                i16::from_le_bytes([e[10], e[11]]),
                f32::from_bits(rd(e, 12)),
                u16::from_le_bytes([e[16], e[17]])
            );
        }
        let strings = crate::reach_strings::ReachStrings::load(&native, cache);
        for id in ["training_vehiclehold", "training_devicehold", "training_swaphold", "training_dualhold", "training_vehicleboard"] {
            println!("string {id}: {:?}", strings.resolve(id, Some("ui\\hud\\hud_messages")).map(|r| (r.unic, r.text)));
        }
        let matg = native.tags_by_group(cache, "matg")[0];
        let h = native.tag_meta(cache, matg, 0x414, 8).unwrap();
        let (n, ptr) = (rd(&h, 0), rd(&h, 4));
        let b = native.tag_ptr_bytes(cache, ptr, 0x8C).unwrap();
        println!("matg player control: {n} element(s); minimum action hold time {:.4} s; fireteam command {:.3}; double tap {:.3}", f32::from_bits(rd(&b, 0x48)), f32::from_bits(rd(&b, 0x34)), f32::from_bits(rd(&b, 0x88)));
    }

    /// PRINT-ONLY: the `player_training` collection of the player chuds, in full.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_training_collection() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        for tag in native.tags_by_group(cache, "chdt") {
            let Some(c) = reach_hud::read_chud(&native, cache, tag) else { continue };
            for col in c.collections.iter().filter(|c| c.base.name.contains("training") || c.base.name.contains("message")) {
                println!("## {} / {} placements {:?} active {:?} hidden {:?} datasource {:?}", c.path, col.base.name, col.base.placements, col.base.active, col.base.hidden, col.datasource);
                for w in &col.widgets {
                    let art = match &w.art {
                        reach_hud::WidgetArt::Text { input, font, flags } => format!("TEXT '{input}' {font:?} flags {flags:#x}"),
                        reach_hud::WidgetArt::Bitmap { .. } => "bitmap".to_string(),
                    };
                    println!("   {} {} placements {:?} active {:?} hidden {:?} colour idx {:?} out {} custom {:?} shader {}", w.base.name, art, w.base.placements, w.base.active, w.base.hidden, w.base.color, w.base.output_a, w.base.custom, w.base.shader_type);
                }
            }
        }
        for st in [18, 36] {
            println!("state {st} = {}", reach_hud::state_name(st));
        }
    }

    /// PRINT-ONLY: every literal-string text widget in every chud of `forge_halo`, with the states
    /// it is gated on -- to find the widget that draws the "Hold E to ..." interaction prompt.
    #[test]
    #[ignore = "print-only probe; run alone with --ignored"]
    fn probe_chud_text_widgets() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        for tag in native.tags_by_group(cache, "chdt") {
            let Some(c) = reach_hud::read_chud(&native, cache, tag) else { continue };
            for col in &c.collections {
                for w in &col.widgets {
                    if let reach_hud::WidgetArt::Text { input, flags, font } = &w.art {
                        if !c.path.contains("hud_") && !c.path.contains("spartan") && !c.path.contains("elite") {
                            continue;
                        }
                        println!(
                            "{} / {} / {} input '{}' font {:?} flags {:#x} anchor {:?} states {:?}",
                            c.path.rsplit('\\').next().unwrap_or(""),
                            col.base.name,
                            w.base.name,
                            input,
                            font,
                            flags,
                            w.base.placements.first().map(|p| (p.anchor, p.offset)),
                            w.base.mentions()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod gates {
    use super::*;

    const AUTHORED: ActionHold = ActionHold { seconds: 0.2333 };

    /// Hold the button for `frames` frames of `dt`; the frame (1-based) on which it was first
    /// REACHED, if it was.
    fn first_reached(t: &mut HoldTimer, frames: usize, dt: f32) -> Option<usize> {
        for f in 1..=frames {
            t.step(true, dt);
            if t.reached(&AUTHORED) {
                return Some(f);
            }
        }
        None
    }

    #[test]
    fn the_authored_time_is_fourteen_ticks() {
        // 0.2333 s -> (int)233 ms; tick 13 is 216 ms, tick 14 is 233 ms.
        assert_eq!(AUTHORED.threshold_ms(), 233);
        assert_eq!(AUTHORED.ticks_to_hold(), 14);
    }

    #[test]
    fn a_tap_never_counts_as_a_hold() {
        let mut t = HoldTimer::default();
        for _ in 0..3 {
            t.step(true, 1.0 / 60.0);
            assert!(!t.reached(&AUTHORED));
        }
        for _ in 0..120 {
            t.step(false, 1.0 / 60.0);
            assert!(!t.reached(&AUTHORED), "a released button holds nothing");
        }
    }

    #[test]
    fn exactly_the_authored_time_holds_and_one_tick_short_does_not() {
        let mut t = HoldTimer::default();
        assert_eq!(first_reached(&mut t, 13, 1.0 / 60.0), None, "13 ticks = 216 ms < 233 ms");
        let mut t = HoldTimer::default();
        assert_eq!(first_reached(&mut t, 14, 1.0 / 60.0), Some(14));
    }

    #[test]
    fn letting_go_early_cancels_and_the_count_restarts() {
        let mut t = HoldTimer::default();
        assert_eq!(first_reached(&mut t, 13, 1.0 / 60.0), None);
        t.step(false, 1.0 / 60.0);
        assert_eq!(t.held_ticks, 0, "release resets the count");
        // 13 more ticks are again one short: the first 13 did not carry over.
        assert_eq!(first_reached(&mut t, 13, 1.0 / 60.0), None);
        assert_eq!(first_reached(&mut t, 1, 1.0 / 60.0), Some(1));
    }

    #[test]
    fn a_hold_acts_once_per_press() {
        let mut t = HoldTimer::default();
        assert_eq!(first_reached(&mut t, 14, 1.0 / 60.0), Some(14));
        t.consume();
        for _ in 0..60 {
            t.step(true, 1.0 / 60.0);
            assert!(!t.reached(&AUTHORED), "still held, already acted");
        }
        t.step(false, 1.0 / 60.0);
        assert_eq!(first_reached(&mut t, 14, 1.0 / 60.0), Some(14), "released and pressed anew acts again");
    }

    #[test]
    fn the_hold_takes_the_same_time_at_any_frame_rate() {
        for fps in [30.0f32, 60.0, 144.0, 240.0] {
            let dt = 1.0 / fps;
            let mut t = HoldTimer::default();
            let f = first_reached(&mut t, 600, dt).expect("reached");
            let elapsed = f as f32 * dt;
            // Never before the authored time, and no later than the frame that crosses its tick.
            assert!(elapsed >= 0.2333 - 1e-3, "{fps} fps: {elapsed} s is before the authored time");
            assert!(elapsed <= 0.2333 + dt + 1.0 / 60.0 + 1e-3, "{fps} fps: {elapsed} s is long after the authored time");
            assert!((14..=14 + (60.0 / fps).ceil() as u32).contains(&t.held_ticks), "{fps} fps: {} ticks", t.held_ticks);
        }
    }
}
