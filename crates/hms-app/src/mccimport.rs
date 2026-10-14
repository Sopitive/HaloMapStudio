//! #mcc-import  Read the user's own MCC settings and offer to adopt them.
//!
//! The Master Chief Collection keeps its PC shell settings in a UE4 `GameUserSettings.ini` under
//! `AppData/LocalLow/MCC/Saved/Config/WindowsNoEditor/`. That file holds the keyboard/mouse
//! bindings, the per-title field of view, mouse sensitivity, the Forge monitor and Theater camera
//! dials, brightness and the audio volumes. This module reads it — **read-only, never written** —
//! and turns it into a [`Plan`]: a list of exactly what would change in HMS and exactly what could
//! not be mapped. Nothing is applied until the user presses the button, and an [`Undo`] puts the
//! previous bindings back.
//!
//! # Nothing happens on its own
//!
//! No discovery, no read, no apply at startup or on a timer. [`Plan::discover`] runs when the
//! user asks for it from Settings ▸ Controls & keyboard, and `apply` runs when they press Apply.
//! The test `nothing_is_read_or_applied_without_the_button` pins that: the user's own settings
//! are their data, and HMS touches them only when told to.
//!
//! # The AbstractButton enum is MCC's, not a guess
//!
//! A binding line reads
//!
//! ```text
//! CustomKeyboardMouseMappingV2[5]=(GameKeyboardMouseMappings[0]=(AbstractButton=0,
//!     VirtualKeyCodes[0]=32,VirtualKeyCodes[1]=0,...),...)
//! ```
//!
//! so an **abstract button id** maps to up to five **Windows virtual-key codes**. The ids are an
//! enum in MCC's own code, and guessing their meanings from key positions would produce a mapping
//! that happens to fit one user's layout and is wrong for everyone else's. So [`ABSTRACT_BUTTONS`]
//! is MCC's own ordered name table, lifted out of the shipping executable:
//!
//! ```text
//! strings -a -n 5 MCC/Binaries/Win64/MCC-Win64-Shipping.exe \
//!   | awk '/^GameAbstractButton_/{sub(/^GameAbstractButton_/,"");print}
//!          /^kGameAbstractButton_CountTotal$/{exit}'
//! ```
//!
//! 66 names, terminated by MCC's own `kGameAbstractButton_CountTotal`, sitting between the
//! `GamepadBinaryButton*` block and `ButtonPreset_Standard..kButtonPresetNone` — a reflection-style
//! name table, not scattered literals.
//!
//! That the table's ORDER is the enum's order is then proven against a real settings file rather
//! than assumed: ids 16/17/18/19 are `MoveForward`/`MoveBackward`/`StrafeLeft`/`StrafeRight` and
//! carry exactly `W`/`S`/`A`/`D`; id 0 `Jump` carries Space; id 3 `Reload` carries `R`; id 30
//! `DeleteObjectEditor` carries Delete; id 8 `Fire` carries the left mouse button. A wrong
//! ordering cannot put WASD on four consecutive movement names *and* Delete on the delete button.
//! `the_recovered_enum_order_matches_a_real_profile` is that check.
//!
//! # The per-title slots are `EMCCHaloGameType`
//!
//! Nearly every setting is an array of seven, one per title, and the index is MCC's own
//! `EMCCHaloGameType` — also from the exe's name table: `Halo1, Halo2, Halo2A, Halo3, Halo3ODST,
//! HaloReach, Halo4` (then `NumGames`, `CrossTitle`, `None`). See [`MccGame`].
//!
//! # Virtual-key codes are Windows-only, and the translation is lossy
//!
//! HMS runs on Linux through winit/egui, where a binding is an `egui::Key`, a pointer button or a
//! modifier chord. [`translate_vk`] lists every code it handles; everything else comes back as
//! [`Vk::Unknown`] and is REPORTED rather than guessed at. The known lossy edges are named in that
//! function's documentation, and `every_handled_vk_round_trips` pins each one.

use std::path::{Path, PathBuf};

use eframe::egui;

use crate::keybinds::{self, Action, Binding, Keybinds, Mods, MouseBtn, Trigger};

// ===================== MCC's own enums =====================

/// `EMCCHaloGameType` — the index of every per-title array in `GameUserSettings.ini`.
///
/// Recovered from `MCC-Win64-Shipping.exe`'s own name table (`EMCCHaloGameType::Halo1` …
/// `::Halo4`, followed by `::NumGames`, `::CrossTitle`, `::None`). Four independent checks against
/// a real settings file agree that slot 5 is Halo: Reach — it is the only non-zero
/// `BrightnessPerGame`, the only `MouseSensitivity` away from the default, one of exactly two
/// `QualityPresetPerGame=Enhanced` entries (Reach and Halo 4 being MCC's two remastered titles),
/// and one of exactly two keyboard profiles that differ from the untouched ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum MccGame {
    Halo1 = 0,
    Halo2 = 1,
    Halo2A = 2,
    Halo3 = 3,
    Halo3Odst = 4,
    HaloReach = 5,
    Halo4 = 6,
}

impl MccGame {
    pub const ALL: [MccGame; 7] = [
        MccGame::Halo1,
        MccGame::Halo2,
        MccGame::Halo2A,
        MccGame::Halo3,
        MccGame::Halo3Odst,
        MccGame::HaloReach,
        MccGame::Halo4,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn label(self) -> &'static str {
        match self {
            MccGame::Halo1 => "Halo: Combat Evolved",
            MccGame::Halo2 => "Halo 2",
            MccGame::Halo2A => "Halo 2: Anniversary",
            MccGame::Halo3 => "Halo 3",
            MccGame::Halo3Odst => "Halo 3: ODST",
            MccGame::HaloReach => "Halo: Reach",
            MccGame::Halo4 => "Halo 4",
        }
    }
}

/// MCC's `GameAbstractButton_*` enum, in order, as the shipping executable lists it.
///
/// Index into this with the `AbstractButton=` value from a mapping entry. The length is MCC's own
/// `kGameAbstractButton_CountTotal`.
pub const ABSTRACT_BUTTONS: [&str; 66] = [
    "Jump",                            // 0
    "SwitchGrenade",                   // 1
    "ActionReload",                    // 2
    "Reload",                          // 3
    "SwitchWeapon",                    // 4
    "MeleeAttack",                     // 5
    "Flashlight",                      // 6
    "ThrowGrenade",                    // 7
    "Fire",                            // 8
    "Crouch",                          // 9
    "Zoom",                            // 10
    "ZoomIn",                          // 11
    "ZoomOut",                         // 12
    "SwapWeapon",                      // 13
    "Sprint",                          // 14
    "BansheeBomb",                     // 15
    "MoveForward",                     // 16
    "MoveBackward",                    // 17
    "StrafeLeft",                      // 18
    "StrafeRight",                     // 19
    "ShowScores",                      // 20
    "PrimaryVehicleTrick",             // 21
    "SecondaryVehicleTrick",           // 22
    "Equipment",                       // 23
    "SecondaryFire",                   // 24
    "LiftEditor",                      // 25
    "DropEditor",                      // 26
    "GrabObjectEditor",                // 27
    "BoostEditor",                     // 28
    "CrouchEditor",                    // 29
    "DeleteObjectEditor",              // 30
    "CreateObjectEditor",              // 31
    "OpenToolMenuEditor",              // 32
    "SwitchPlayerModeEditor",          // 33
    "ScopeZoomEditor",                 // 34
    "PlayerLockForManipulationEditor", // 35
    "ShowHidePannelTheater",           // 36  (MCC's own spelling)
    "ShowHideInterfaceTheater",        // 37
    "ToggleFirstThirdPersonViewTheater", // 38
    "CameraFocusTheater",              // 39
    "FastForwardTheater",              // 40
    "FastRewindTheater",               // 41
    "StopContinuePlaybackTheater",     // 42
    "PlaybackSpeedUpTheater",          // 43
    "EnterFreeCameraModeTheater",      // 44
    "MovementSpeedUpTheater",          // 45
    "PanningCameraTheater",            // 46
    "CameraMoveUpTheater",             // 47
    "CameraMoveDownTheater",           // 48
    "DualWield",                       // 49
    "ZoomCameraTheater",               // 50
    "ToggleRotationAxesEditor",        // 51
    "DuplicateObjectEditor",           // 52
    "LockObjectEditor",                // 53
    "ResetOrientationEditor",          // 54
    "ReloadSecondary",                 // 55
    "PreviousGrenade",                 // 56
    "SpecialAction",                   // 57
    "LoadoutMenu",                     // 58
    "ActivateWaypoint",                // 59
    "ActivateWaypointAlt",             // 60
    "PingNavpoints",                   // 61
    "RaiseHornet",                     // 62
    "LowerHornet",                     // 63
    "FlashlightAlt",                   // 64
    "NextGrenade",                     // 65
];

/// The abstract-button ids this module refers to by name. Only the ones used in [`ACTION_MAP`] or
/// in a test need a constant; the rest are reached through [`ABSTRACT_BUTTONS`].
///
/// The gameplay ids here are for the first-person test mode rather than for this importer, which
/// is why several are unused inside this module — see [`MccProfile`].
#[allow(dead_code)]
pub mod btn {
    pub const JUMP: u32 = 0;
    pub const RELOAD: u32 = 3;
    pub const MELEE_ATTACK: u32 = 5;
    pub const FIRE: u32 = 8;
    pub const CROUCH: u32 = 9;
    pub const SPRINT: u32 = 14;
    pub const MOVE_FORWARD: u32 = 16;
    pub const MOVE_BACKWARD: u32 = 17;
    pub const STRAFE_LEFT: u32 = 18;
    pub const STRAFE_RIGHT: u32 = 19;
    pub const LIFT_EDITOR: u32 = 25;
    pub const DROP_EDITOR: u32 = 26;
    pub const GRAB_OBJECT_EDITOR: u32 = 27;
    pub const BOOST_EDITOR: u32 = 28;
    pub const DELETE_OBJECT_EDITOR: u32 = 30;
    pub const DUPLICATE_OBJECT_EDITOR: u32 = 52;
}

// ===================== the UE4 ini =====================

/// A parsed UE4 `.ini`: a flat list of `(section, key, value)` in file order.
///
/// Flat on purpose. MCC's keys carry their array index IN the key name
/// (`FOVSetting[5]`, `CustomKeyboardMouseMappingV2[5]`), so they are already unique and a nested
/// map would buy nothing. [`UeIni::get`] takes the LAST occurrence, which is how UE4 itself reads
/// a duplicated key.
#[derive(Clone, Debug, Default)]
pub struct UeIni {
    entries: Vec<(String, String, String)>,
}

impl UeIni {
    /// Parse whatever we were handed. This never fails and never panics: MCC may be running and
    /// rewriting the file as we read it, so a truncated line, a key with no `=`, a stray `[`
    /// or invalid UTF-8 has to degrade to "that one line is not there" rather than to an error
    /// for the whole file. A *structurally* unusable file is caught later, by [`Plan::from_ini`],
    /// which refuses when the bindings it needs are absent.
    pub fn parse(text: &str) -> UeIni {
        let mut entries = Vec::new();
        let mut section = String::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                // A truncated section header ("[/Script/MCC.MCC") has no ']' — skip it rather
                // than treating the remainder of the file as part of the previous section.
                if let Some(name) = rest.strip_suffix(']') {
                    section = name.to_string();
                }
                continue;
            }
            let Some((k, v)) = line.split_once('=') else { continue };
            entries.push((section.clone(), k.trim().to_string(), v.trim().to_string()));
        }
        UeIni { entries }
    }

    /// Read the file, tolerating invalid UTF-8 and a mid-write change of length.
    pub fn read(path: &Path) -> std::io::Result<UeIni> {
        let bytes = std::fs::read(path)?;
        Ok(UeIni::parse(&String::from_utf8_lossy(&bytes)))
    }

    /// The last value for a key, in any section. Section names are not needed to disambiguate
    /// anything in this file, and not depending on them means a future MCC that renames a section
    /// still imports.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.iter().rev().find(|(_, k, _)| k == key).map(|(_, _, v)| v.as_str())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.parse().ok()
    }

    #[allow(dead_code)] // part of the ini reader's surface; the slider-step fields use it
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        // `VerticalLookSensitivity[0]=4` is an integer, `SwapTriggersAndBumpers[0]=0.000000` a
        // float. Accept either shape for either kind of field.
        let v = self.get(key)?;
        v.parse::<i32>().ok().or_else(|| v.parse::<f32>().ok().map(|f| f as i32))
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)?.to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }

    /// An indexed per-title key, e.g. `fov = ini.get_indexed("FOVSetting", MccGame::HaloReach)`.
    pub fn get_indexed(&self, stem: &str, game: MccGame) -> Option<&str> {
        self.get(&format!("{stem}[{}]", game.index()))
    }

    /// How many `key=value` pairs were read — how a caller tells a real settings file from a
    /// stub or a file caught mid-write.
    #[allow(dead_code)] // used by the tests against the real on-disk file
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ===================== MCC's keyboard/mouse profile =====================

/// MCC allows up to five virtual-key codes per abstract button (a chord, or alternatives). The
/// format writes all five slots and pads with `0`, which is not a valid VK.
#[allow(dead_code)] // documents the format; the parser does not need to count slots
pub const VK_SLOTS: usize = 5;

/// One MCC abstract button and the virtual-key codes bound to it, zeros stripped.
///
/// Both are `u32` rather than the `u16`/`u8` the values actually need, so that a number this
/// importer does not understand — a button id past the end of MCC's enum, a key code out of
/// range — is carried through and REPORTED rather than failing to parse and vanishing. A binding
/// that silently disappears is the failure mode worth designing out: the user would see an action
/// quietly left at its default with nothing said about it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MccBinding {
    pub button: u32,
    pub vks: Vec<u32>,
}

impl MccBinding {
    /// MCC's own name for this button, or a placeholder if a future MCC grows the enum.
    pub fn name(&self) -> String {
        ABSTRACT_BUTTONS
            .get(self.button as usize)
            .map(|s| (*s).to_string())
            .unwrap_or_else(|| format!("AbstractButton {}", self.button))
    }
}

/// One title's full keyboard/mouse mapping, exactly as MCC stored it.
///
/// This is the public surface other features read. The settings importer itself only consumes
/// nine of the 66 buttons ([`ACTION_MAP`]), but a first-person test mode wants Jump, Crouch,
/// Sprint, Fire, MeleeAttack, Reload, ThrowGrenade, SwitchWeapon and Zoom — so the whole decoded
/// profile is exposed rather than only the part this module happens to use. Taking them from here
/// means a test mode's defaults are the user's own MCC bindings with no second parser.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MccProfile {
    pub game: MccGame,
    pub bindings: Vec<MccBinding>,
}

impl MccProfile {
    /// Parse one `CustomKeyboardMouseMappingV2[n]` value.
    ///
    /// Hand-written rather than regex-driven, because the thing it must survive is a file being
    /// rewritten under it: a line cut off mid-entry has to yield the complete entries before the
    /// cut and drop the partial one, not fail.
    pub fn parse(game: MccGame, value: &str) -> MccProfile {
        let mut bindings = Vec::new();
        // Walk entry by entry. Every entry begins with this literal; anything between entries
        // (parentheses, commas, `GameKeyboardMouseMappings[n]=`) is structure we do not need.
        let mut rest = value;
        while let Some(at) = rest.find("AbstractButton=") {
            rest = &rest[at + "AbstractButton=".len()..];
            let Some(button) = take_num(&mut rest) else { continue };
            let mut vks = Vec::new();
            // The VK slots belong to THIS entry only: stop at the next `AbstractButton=`.
            let end = rest.find("AbstractButton=").unwrap_or(rest.len());
            let mut slots = &rest[..end];
            while let Some(eq) = slots.find("VirtualKeyCodes[") {
                slots = &slots[eq + "VirtualKeyCodes[".len()..];
                // index, then `]=`, then the code
                let Some(_slot) = take_num(&mut slots) else { continue };
                let Some(after) = slots.strip_prefix("]=") else { continue };
                slots = after;
                let Some(vk) = take_num(&mut slots) else { continue };
                if vk != 0 {
                    vks.push(vk);
                }
            }
            bindings.push(MccBinding { button, vks });
        }
        MccProfile { game, bindings }
    }

    pub fn get(&self, button: u32) -> Option<&MccBinding> {
        self.bindings.iter().find(|b| b.button == button)
    }
}

/// Leading decimal digits off the front of `s`, advancing it past them. `None` only when there
/// are no digits at all (a truncated or malformed entry); an absurdly long run of digits
/// saturates rather than rejecting the entry, because the entry is still real and still worth
/// reporting.
fn take_num(s: &mut &str) -> Option<u32> {
    let n = s.bytes().take_while(|b| b.is_ascii_digit()).count();
    if n == 0 {
        return None;
    }
    let (head, tail) = s.split_at(n);
    *s = tail;
    Some(head.parse().unwrap_or(u32::MAX))
}

// ===================== virtual-key translation =====================

/// What one Windows virtual-key code means to HMS.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vk {
    Key(egui::Key),
    Mouse(MouseBtn),
    /// A modifier. Which SIDE it was is lost here — see [`translate_vk`].
    Modifier(Mods),
    /// A code this table recognises, but which HMS has no trigger for. Carries the Windows name
    /// so the report can say what it was instead of printing a bare number.
    NoHmsTrigger(&'static str),
    /// Not a code this table knows. Reported, never guessed at.
    Unknown(u32),
}

/// Translate one Windows virtual-key code.
///
/// # What is handled
///
/// * `0x01 0x02 0x04` — left / right / middle mouse → [`MouseBtn`].
/// * `0x08 0x09 0x0D 0x1B 0x20` — Backspace, Tab, Enter, Escape, Space.
/// * `0x21`–`0x28` — PageUp, PageDown, End, Home and the four arrows.
/// * `0x2D 0x2E` — Insert, Delete.
/// * `0x30`–`0x39` — the digit row → `Num0`…`Num9`.
/// * `0x41`–`0x5A` — `A`…`Z`.
/// * `0x60`–`0x69` — the numpad digits → `Num0`…`Num9`. **Lossy**: egui's own documentation says
///   that variant is "from main row or numpad", so a numpad binding and a digit-row binding become
///   the same HMS binding. A numpad import therefore also fires on the digit row.
/// * `0x6B 0x6D 0x6E 0x6F` — numpad `+ - . /` → `Plus`, `Minus`, `Period`, `Slash` (same collapse).
/// * `0x70`–`0x87` — `F1`…`F24`.
/// * `0x10 0xA0 0xA1` — Shift; `0x11 0xA2 0xA3` — Ctrl; `0x12 0xA4 0xA5` — Alt. **Lossy**:
///   `egui::Modifiers` has no side, so left and right collapse to the same modifier. An MCC binding
///   on the RIGHT Ctrl imports as "Ctrl".
/// * `0xBA`–`0xBF` and `0xDB`–`0xDE` — the OEM punctuation of a US layout: `; = , - . /` and
///   `[ \ ] '`. **Lossy**: the OEM codes are POSITIONS, and the character they produce depends on
///   the layout, so these are the US-layout characters and will name the wrong character on, say,
///   an AZERTY keyboard. The imported binding is written by position for exactly this reason.
///
/// # What is reported rather than guessed
///
/// * `0x05 0x06` — the two extra mouse buttons (MCC calls them `VK_MOUSE4`/`VK_MOUSE5`). HMS's
///   [`MouseBtn`] has only left, right and middle.
/// * `0x13 0x14 0x2C 0x90 0x91` — Pause, CapsLock, PrintScreen, NumLock, ScrollLock, and `0x6A`
///   numpad `*`: `egui::Key` has no variant for any of them.
/// * `174` / `175` — codes MCC uses that are NOT the Win32 meaning (`VK_VOLUME_DOWN`/`_UP`). They
///   appear only on `ZoomIn`/`ZoomOut`; MCC's executable carries its own names `VK_MWHEEL_UP` and
///   `VK_MWHEEL_DOWN` beside `VK_MOUSE4`/`VK_MOUSE5`, and these are the only non-Win32-standard
///   codes present, so they are very probably the mouse wheel. That is *circumstantial*, it is not
///   proven, and `keybinds::Trigger` has no wheel variant to map them to even if it is right — so
///   they come back [`Vk::Unknown`] and the report says so.
/// * anything else → [`Vk::Unknown`].
pub fn translate_vk(vk: u32) -> Vk {
    use egui::Key as K;
    match vk {
        0x01 => Vk::Mouse(MouseBtn::Left),
        0x02 => Vk::Mouse(MouseBtn::Right),
        0x04 => Vk::Mouse(MouseBtn::Middle),
        0x05 => Vk::NoHmsTrigger("mouse button 4"),
        0x06 => Vk::NoHmsTrigger("mouse button 5"),

        0x08 => Vk::Key(K::Backspace),
        0x09 => Vk::Key(K::Tab),
        0x0D => Vk::Key(K::Enter),
        0x13 => Vk::NoHmsTrigger("Pause"),
        0x14 => Vk::NoHmsTrigger("Caps Lock"),
        0x1B => Vk::Key(K::Escape),
        0x20 => Vk::Key(K::Space),

        0x21 => Vk::Key(K::PageUp),
        0x22 => Vk::Key(K::PageDown),
        0x23 => Vk::Key(K::End),
        0x24 => Vk::Key(K::Home),
        0x25 => Vk::Key(K::ArrowLeft),
        0x26 => Vk::Key(K::ArrowUp),
        0x27 => Vk::Key(K::ArrowRight),
        0x28 => Vk::Key(K::ArrowDown),
        0x2C => Vk::NoHmsTrigger("Print Screen"),
        0x2D => Vk::Key(K::Insert),
        0x2E => Vk::Key(K::Delete),

        0x30..=0x39 => Vk::Key(DIGITS[(vk - 0x30) as usize]),
        0x41..=0x5A => Vk::Key(LETTERS[(vk - 0x41) as usize]),
        0x60..=0x69 => Vk::Key(DIGITS[(vk - 0x60) as usize]),
        0x6A => Vk::NoHmsTrigger("numpad *"),
        0x6B => Vk::Key(K::Plus),
        0x6D => Vk::Key(K::Minus),
        0x6E => Vk::Key(K::Period),
        0x6F => Vk::Key(K::Slash),
        0x70..=0x87 => Vk::Key(FUNCTION[(vk - 0x70) as usize]),

        0x90 => Vk::NoHmsTrigger("Num Lock"),
        0x91 => Vk::NoHmsTrigger("Scroll Lock"),

        0x10 | 0xA0 | 0xA1 => Vk::Modifier(Mods::SHIFT),
        0x11 | 0xA2 | 0xA3 => Vk::Modifier(Mods::CTRL),
        0x12 | 0xA4 | 0xA5 => Vk::Modifier(Mods::ALT),

        0xBA => Vk::Key(K::Semicolon),
        0xBB => Vk::Key(K::Equals),
        0xBC => Vk::Key(K::Comma),
        0xBD => Vk::Key(K::Minus),
        0xBE => Vk::Key(K::Period),
        0xBF => Vk::Key(K::Slash),
        0xDB => Vk::Key(K::OpenBracket),
        0xDC => Vk::Key(K::Backslash),
        0xDD => Vk::Key(K::CloseBracket),
        0xDE => Vk::Key(K::Quote),

        other => Vk::Unknown(other),
    }
}

const DIGITS: [egui::Key; 10] = {
    use egui::Key as K;
    [K::Num0, K::Num1, K::Num2, K::Num3, K::Num4, K::Num5, K::Num6, K::Num7, K::Num8, K::Num9]
};

const LETTERS: [egui::Key; 26] = {
    use egui::Key as K;
    [
        K::A, K::B, K::C, K::D, K::E, K::F, K::G, K::H, K::I, K::J, K::K, K::L, K::M, K::N, K::O,
        K::P, K::Q, K::R, K::S, K::T, K::U, K::V, K::W, K::X, K::Y, K::Z,
    ]
};

const FUNCTION: [egui::Key; 24] = {
    use egui::Key as K;
    [
        K::F1, K::F2, K::F3, K::F4, K::F5, K::F6, K::F7, K::F8, K::F9, K::F10, K::F11, K::F12,
        K::F13, K::F14, K::F15, K::F16, K::F17, K::F18, K::F19, K::F20, K::F21, K::F22, K::F23,
        K::F24,
    ]
};

/// What came of trying to turn one MCC binding into an HMS one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// An HMS binding equal in meaning to the MCC one.
    Ok(Binding),
    /// MCC has nothing bound to this button.
    Unbound,
    /// The key is one HMS deliberately keeps fixed; carries [`keybinds::reserved`]'s own reason.
    Reserved(&'static str),
    /// Could not be represented. Carries the reason, for the report.
    Unsupported(String),
}

impl Outcome {
    pub fn binding(&self) -> Option<Binding> {
        match self {
            Outcome::Ok(b) => Some(*b),
            _ => None,
        }
    }
}

/// Turn one MCC abstract button's virtual-key codes into an HMS [`Binding`].
///
/// `by_position` picks whether the result means "the key at this position on a US keyboard" or
/// "the key that types this character" — see `keybinds`' module documentation. A virtual-key code
/// IS a position, so an import is by position by default; a command mnemonic is the exception.
pub fn to_binding(vks: &[u32], by_position: bool) -> Outcome {
    if vks.is_empty() {
        return Outcome::Unbound;
    }
    let mut mods = Mods::NONE;
    let mut trigger: Option<Trigger> = None;
    for &vk in vks {
        match translate_vk(vk) {
            Vk::Modifier(m) => {
                mods.ctrl |= m.ctrl;
                mods.shift |= m.shift;
                mods.alt |= m.alt;
            }
            Vk::Key(k) => {
                if let Some(why) = keybinds::reserved(k) {
                    return Outcome::Reserved(why);
                }
                if trigger.is_some() {
                    // Two non-modifier keys at once. MCC allows it; HMS's Trigger is one key.
                    return Outcome::Unsupported(
                        "MCC binds two keys at once here, which Halo Map Studio cannot express"
                            .into(),
                    );
                }
                trigger = Some(Trigger::Key { key: k, by_position });
            }
            Vk::Mouse(b) => {
                if trigger.is_some() {
                    return Outcome::Unsupported(
                        "MCC binds a key and a mouse button at once here, which Halo Map Studio \
                         cannot express"
                            .into(),
                    );
                }
                trigger = Some(Trigger::Mouse(b));
            }
            Vk::NoHmsTrigger(name) => {
                return Outcome::Unsupported(format!(
                    "MCC has this on {name}, which Halo Map Studio has no binding for"
                ));
            }
            Vk::Unknown(code) => {
                return Outcome::Unsupported(if code == 174 || code == 175 {
                    // Named rather than numbered, without claiming the value is proven.
                    "MCC uses its own key code here (most likely the mouse wheel, which Halo Map \
                     Studio cannot bind)"
                        .into()
                } else {
                    format!("MCC has this on key code {code}, which this importer does not know")
                });
            }
        }
    }
    match trigger {
        Some(t) => Outcome::Ok(Binding { trigger: t, mods }),
        // Modifiers and nothing else: a held chord, which is a real HMS binding.
        None if mods != Mods::NONE => Outcome::Ok(Binding::chord(mods)),
        None => Outcome::Unbound,
    }
}

// ===================== the action map =====================

/// One HMS action an MCC button genuinely corresponds to.
pub struct MapRule {
    pub action: Action,
    pub button: u32,
    /// Why these two are the same thing. Shown in the preview, so the user can disagree.
    pub why: &'static str,
    /// Whether the imported key means a POSITION or a CHARACTER (see [`to_binding`]).
    pub by_position: bool,
}

/// The MCC buttons that mean the same thing as an HMS action.
///
/// Deliberately short. HMS has 38 actions and MCC has 66 buttons, but most of each set has no
/// counterpart in the other: MCC knows nothing about axis locks, modal transforms, nudges, undo or
/// a construction tool, and HMS has no concept of grenades, vehicles or a scoreboard. Forcing a
/// near-miss into the table would leave the user with a key that does almost what they expect,
/// which is worse than a clear "this one has no equivalent" — so everything not here is REPORTED
/// by [`Plan`] rather than quietly left at its default.
///
/// The nine editor rules come from MCC's `Editor` block, which is at MCC's own defaults even in a
/// settings file whose gameplay keys have been personally rebound (the `Editor` and `Theater`
/// blocks are identical across all seven title profiles) — so importing them gives the user MCC's
/// Forge feel rather than one person's quirk. The four `Play` rules are the opposite on purpose:
/// they are the gameplay buttons the user HAS rebound, which is exactly what makes a first-person
/// test mode feel like their game rather than like a default.
///
/// # A mouse button can land on a key action
///
/// `Spec::mouse` marks the one action that is PICKED FROM A LIST instead of captured, because you
/// cannot click a button to capture a click. It does not say a binding may not be a pointer
/// button: `Trigger::Mouse` is first class and `fires_down`/`fires_pressed` read it for any action.
/// That matters here because MCC's crouch and melee are the right and middle mouse buttons in the
/// profile this was built against, and refusing them would drop two of the four gameplay imports
/// for no gain. So they ARE imported, they show as "Right mouse" in the preview before anything is
/// applied, and the row stays editable afterwards — clicking it captures a key, right-clicking
/// clears it.
pub const ACTION_MAP: [MapRule; 13] = [
    MapRule {
        action: Action::CamForward,
        button: btn::MOVE_FORWARD,
        why: "MCC flies the Forge monitor with the movement keys",
        by_position: true,
    },
    MapRule {
        action: Action::CamBack,
        button: btn::MOVE_BACKWARD,
        why: "MCC flies the Forge monitor with the movement keys",
        by_position: true,
    },
    MapRule {
        action: Action::CamLeft,
        button: btn::STRAFE_LEFT,
        why: "MCC flies the Forge monitor with the movement keys",
        by_position: true,
    },
    MapRule {
        action: Action::CamRight,
        button: btn::STRAFE_RIGHT,
        why: "MCC flies the Forge monitor with the movement keys",
        by_position: true,
    },
    MapRule {
        action: Action::CamUp,
        button: btn::LIFT_EDITOR,
        why: "MCC's Forge \"lift\" raises the monitor",
        by_position: true,
    },
    MapRule {
        action: Action::CamDown,
        button: btn::DROP_EDITOR,
        why: "MCC's Forge \"drop\" lowers the monitor",
        by_position: true,
    },
    MapRule {
        action: Action::FlyFaster,
        button: btn::BOOST_EDITOR,
        why: "MCC's Forge \"boost\" is the monitor's speed-up",
        by_position: true,
    },
    MapRule {
        action: Action::Delete,
        button: btn::DELETE_OBJECT_EDITOR,
        // A command, so the CHARACTER rather than the position — but Delete is in the same place
        // on every layout, which makes the distinction moot here and consistent with the table.
        why: "MCC's Forge delete-object",
        by_position: false,
    },
    MapRule {
        action: Action::Duplicate,
        button: btn::DUPLICATE_OBJECT_EDITOR,
        why: "MCC's Forge duplicate-object",
        by_position: false,
    },
    // --- test mode: the gameplay buttons, which are the whole point of importing MCC's controls
    // for a first-person mode. These four are the user's OWN rebound keys rather than MCC's
    // defaults, so the mode feels like their game on the first try.
    MapRule {
        action: Action::PlayJump,
        button: btn::JUMP,
        why: "MCC's jump",
        by_position: true,
    },
    MapRule {
        action: Action::PlayCrouch,
        button: btn::CROUCH,
        why: "MCC's crouch",
        by_position: true,
    },
    MapRule {
        action: Action::PlaySprint,
        button: btn::SPRINT,
        why: "MCC's sprint",
        by_position: true,
    },
    MapRule {
        action: Action::PlayMelee,
        button: btn::MELEE_ATTACK,
        why: "MCC's melee",
        by_position: true,
    },
];

/// Why an HMS action is not in [`ACTION_MAP`]. Reported so a partial import is never quiet.
pub fn why_no_mcc_equivalent(a: Action) -> &'static str {
    use Action as A;
    match a {
        // The near miss, called out by name rather than lumped in with the rest.
        A::ModalMove => {
            "MCC's closest button is Forge's grab-object, which is a MOUSE button; Halo Map \
             Studio's move is a key, so importing it would change what kind of control it is"
        }
        A::FlyLook => {
            "MCC's Forge monitor always looks with the mouse — there is no hold-to-look button to \
             import"
        }
        A::FlySlower => "MCC's Forge monitor has a speed-up but no slow-down",
        A::ModalRotate | A::AxisX | A::AxisY | A::AxisZ | A::AxisSpace | A::ConfirmTransform
        | A::Magnet | A::PrecisionSlow | A::ArrayFine => {
            "a modelling control with no counterpart in Forge"
        }
        A::SelectAll | A::SelectAdd | A::ContextMenu | A::FrameSelected => {
            "a selection control with no counterpart in Forge"
        }
        A::Undo | A::Redo => "MCC's Forge has no undo to import",
        A::NudgeXPlus | A::NudgeXMinus | A::NudgeYPlus | A::NudgeYMinus | A::NudgeZPlus
        | A::NudgeZMinus | A::NudgeLarge | A::YawPlus | A::YawMinus => {
            "Forge nudges an object with the monitor rather than with its own keys"
        }
        A::ConstructTool => "a Halo Map Studio tool with no counterpart in Forge",
        A::Screenshot => "MCC screenshots through Steam, not through a game binding",
        A::PlayWalk => {
            "MCC has no walk key on mouse and keyboard — on a controller the walk IS how far the \
             stick is pushed"
        }
        // Everything in ACTION_MAP lands here and is never asked for a reason, so the arm must
        // not read "imported": an action added to `keybinds` later would then be reported as
        // imported when it was not. A truthful generic answer instead, and
        // `every_action_is_accounted_for_as_mapped_or_explained` proves no unmapped action
        // reaches it claiming otherwise.
        _ => "no counterpart in MCC's mouse-and-keyboard controls",
    }
}

// ===================== the non-binding settings =====================

/// What HMS does with one imported value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Disposition {
    /// Read and shown, but HMS does not change anything from it, and the row says why.
    ReportOnly,
    /// Belongs to a part of HMS this import does not reach into yet.
    NotYetWired,
    /// Read and APPLIED: the import changes this, and Undo puts it back.
    Applied,
}

/// One MCC setting, read and judged.
#[derive(Clone, Debug)]
pub struct SettingRow {
    pub label: String,
    pub value: String,
    pub disposition: Disposition,
    pub note: String,
}

/// #mcc-fov  MCC's first-person FOV for one title, in degrees, if it is a value HMS can use.
///
/// A named function rather than a closure inside `from_ini` so the gates can exercise **this**
/// decision instead of a copy of it: a test that re-implements a filter proves the arithmetic and
/// nothing about what production does.
///
/// **Refuses rather than clamps.** A number outside HMS's own `FOV_MIN_DEG..=FOV_MAX_DEG` means
/// the file is not saying what we think it is -- a different key, a different unit, or a partial
/// write while MCC is running -- and clamping it would import a value the user never chose.
pub fn usable_fov(ini: &UeIni, game: MccGame) -> Option<f32> {
    ini.get_f32(&format!("FOVSetting[{}]", game.index()))
        .filter(|v| (crate::film_play::FOV_MIN_DEG..=crate::film_play::FOV_MAX_DEG).contains(v))
}

/// Read the non-binding settings for one title.
///
/// # Brightness is read and NOT applied
///
/// `BrightnessPerGame[n]` is MCC's brightness slider as an offset about 0. HMS's exposure and
/// tonemap are engine-exact laws recovered from the shipped shaders and the engine binary
/// (`hrek-exposure-tonemap`, `hms-h3-lighting-law`), calibrated against engine frame captures.
/// Feeding a slider offset into that pipeline would quietly undo that calibration, and a user who
/// imported "controls" would have no idea why the lighting changed. So the value is REPORTED and
/// nothing is applied. Whether HMS should grow a labelled post-tonemap display-gamma control is a
/// rendering decision, not something a settings import gets to make.
pub fn read_settings(ini: &UeIni, game: MccGame) -> Vec<SettingRow> {
    let mut out = Vec::new();
    let mut push = |label: &str, value: Option<String>, d: Disposition, note: &str| {
        if let Some(v) = value {
            out.push(SettingRow {
                label: label.to_string(),
                value: v,
                disposition: d,
                note: note.to_string(),
            });
        }
    };
    let idx = |stem: &str| ini.get_indexed(stem, game).map(|s| s.to_string());
    let num = |stem: &str| ini.get_indexed(stem, game).and_then(|s| s.parse::<f32>().ok());

    push(
        "Field of view",
        num("FOVSetting").map(|v| format!("{v:.0}°")),
        Disposition::Applied,
        "Imported. Halo Map Studio's first-person field of view is already in MCC's own units and \
         the same 60–120 range, so this carries across exactly. Undo puts your previous value back.",
    );
    push(
        "Mouse sensitivity",
        num("MouseSensitivity").map(|v| format!("{v:.2}")),
        Disposition::NotYetWired,
        "Halo Map Studio's look speed is currently fixed. MCC's default is 1.60, so this value is \
         a multiplier on it — large enough that it needs its own switch rather than arriving with \
         the keys.",
    );
    push(
        "Invert mouse look",
        idx("MouseLookControlsInverted").map(|v| yes_no(&v)),
        Disposition::NotYetWired,
        "Halo Map Studio has no invert-look option yet.",
    );
    push(
        "Brightness",
        num("BrightnessPerGame").map(|v| format!("{v:+.2}")),
        Disposition::ReportOnly,
        "NOT imported, on purpose. Halo Map Studio's exposure and tonemap are the engine's own \
         laws, calibrated against captures of the real game; a brightness offset pushed into them \
         would change the rendering away from what the engine does.",
    );
    push(
        "Forge monitor camera (speed / thrust / turn / panning)",
        match (
            idx("FlyingCameraSpeed"),
            idx("FlyingCameraThrust"),
            idx("FlyingCameraTurnSensitivity"),
            idx("FlyingCameraPanning"),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => Some(format!("{a} / {b} / {c} / {d}")),
            _ => None,
        },
        Disposition::ReportOnly,
        "MCC stores these as slider steps, not as a speed. Halo Map Studio's fly speed is in world \
         units per second, and there is no measured conversion between the two — so the values are \
         shown rather than converted into a number that would only look right.",
    );
    push(
        "Theater camera (speed / thrust / turn / panning)",
        match (
            ini.get("MKTheaterSpeed"),
            ini.get("MKTheaterThrust"),
            ini.get("MKTheaterTurnSensitivity"),
            ini.get("MKTheaterPanning"),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => Some(format!("{a} / {b} / {c} / {d}")),
            _ => None,
        },
        Disposition::ReportOnly,
        "MCC's mouse-and-keyboard Theater camera, also stored as slider steps rather than a speed.",
    );
    push(
        "Volumes (master / effects / music / menu music)",
        match (
            ini.get_f32("MasterVolume"),
            ini.get_f32("AudioEffectsVolume"),
            ini.get_f32("AudioInGameMusicVolume"),
            ini.get_f32("MenuMusicVolume"),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                Some(format!("{a:.0} / {b:.0} / {c:.0} / {d:.0}"))
            }
            _ => None,
        },
        Disposition::NotYetWired,
        "Halo Map Studio's Theater sound has its own volume; this import does not touch it.",
    );
    push(
        "Vertical sync / frame-rate limit / resolution scale",
        match (ini.get_bool("bUseVSync"), ini.get_f32("FrameRateLimit"), ini.get_f32("GameResolutionScale")) {
            (Some(v), Some(f), Some(r)) => Some(format!(
                "{} / {} / {:.0}%",
                yes_no(if v { "1" } else { "0" }),
                if f <= 0.0 { "unlimited".to_string() } else { format!("{f:.0} fps") },
                r * 100.0
            )),
            _ => None,
        },
        Disposition::NotYetWired,
        "Halo Map Studio has its own graphics settings page; this import does not change it.",
    );
    out
}

fn yes_no(v: &str) -> String {
    match v.trim() {
        "0" | "false" | "False" => "off".into(),
        "1" | "true" | "True" => "on".into(),
        other => other.into(),
    }
}

// ===================== discovery =====================

/// Where a `GameUserSettings.ini` was found, and whose it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub ini: PathBuf,
    /// How to NAME this install to the user without printing the whole path. A discovered path
    /// runs through a Windows account folder, so the UI labels the install by its role instead.
    pub label: String,
}

/// Find every MCC `GameUserSettings.ini` on this machine.
///
/// **Call this only when the user asks for it.** Nothing in HMS runs it at startup.
///
/// The places to look are the same places a saved film lives, so this walks
/// [`crate::film::search_roots`] with `film`'s own case-insensitive descent rather than carrying a
/// second copy of that search: a Windows partition mounted on Linux, every Steam library's Proton
/// prefixes, and `$HOME`. The only difference from the film search is where it stops — at the `MCC`
/// directory itself rather than at `MCC/Temporary`.
pub fn discover() -> Vec<Found> {
    let mut out: Vec<Found> = Vec::new();
    // An explicit override, for a test and for an install this cannot guess.
    if let Ok(v) = std::env::var("HMS_MCC_SETTINGS") {
        for p in v.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            out.push(Found { ini: PathBuf::from(p), label: "set by HMS_MCC_SETTINGS".into() });
        }
    }
    for root in crate::film::search_roots() {
        // (a) a Windows partition mounted on Linux, or the Windows build's own drive.
        if let Some(users) = crate::film::child_ci(&root, "Users") {
            scan_users(&users, "this computer's Windows install", &mut out);
        }
        // (b) every Proton prefix under every Steam library on this root.
        for lib in crate::film::steam_libraries(&root) {
            let Some(compat) = crate::film::descend_ci(&lib, &["steamapps", "compatdata"]) else {
                continue;
            };
            let Ok(rd) = std::fs::read_dir(&compat) else { continue };
            for app in rd.flatten() {
                let Some(users) = crate::film::descend_ci(&app.path(), &["pfx", "drive_c", "users"])
                else {
                    continue;
                };
                let label = format!(
                    "Steam / Proton prefix {}",
                    app.file_name().to_string_lossy()
                );
                scan_users(&users, &label, &mut out);
            }
        }
    }
    out.sort_by(|a, b| a.ini.cmp(&b.ini));
    out.dedup_by(|a, b| a.ini == b.ini);
    out.retain(|f| f.ini.is_file());
    out
}

/// Every `MCC*/Saved/Config/WindowsNoEditor/GameUserSettings.ini` under a Windows `users` tree.
fn scan_users(users: &Path, label: &str, out: &mut Vec<Found>) {
    let Ok(rd) = std::fs::read_dir(users) else { return };
    for u in rd.flatten() {
        let Some(ll) = crate::film::descend_ci(&u.path(), &["AppData", "LocalLow"]) else {
            continue;
        };
        let Ok(rd2) = std::fs::read_dir(&ll) else { continue };
        for mcc in rd2.flatten() {
            if !mcc.file_name().to_string_lossy().to_ascii_uppercase().starts_with("MCC") {
                continue;
            }
            let Some(cfg) =
                crate::film::descend_ci(&mcc.path(), &["Saved", "Config", "WindowsNoEditor"])
            else {
                continue;
            };
            let ini = cfg.join("GameUserSettings.ini");
            if ini.is_file() {
                out.push(Found { ini, label: label.to_string() });
            }
        }
    }
}

// ===================== the plan =====================

/// One row of the before/after preview.
#[derive(Clone, Debug)]
pub struct PlanRow {
    pub action: Action,
    /// MCC's own name for the button this came from.
    pub mcc_button: String,
    pub why: &'static str,
    /// What the action is bound to now.
    pub before: String,
    pub outcome: Outcome,
}

impl PlanRow {
    /// What the action would be bound to, or why not.
    pub fn after(&self) -> String {
        match &self.outcome {
            Outcome::Ok(b) => b.label(),
            Outcome::Unbound => "not bound in MCC — unchanged".into(),
            Outcome::Reserved(_) => "reserved by Halo Map Studio — unchanged".into(),
            Outcome::Unsupported(_) => "cannot be imported — unchanged".into(),
        }
    }
    /// True when applying this row would actually change something.
    pub fn changes(&self) -> bool {
        matches!(&self.outcome, Outcome::Ok(b) if b.label() != self.before)
    }
}

/// Everything an import would do, and everything it could not do. Built, shown, and only then
/// applied.
#[derive(Clone, Debug)]
pub struct Plan {
    pub source: Found,
    pub game: MccGame,
    /// The whole decoded profile, all 66 buttons — the public surface other features read.
    /// Unused by the import itself on purpose: see [`MccProfile`].
    #[allow(dead_code)]
    pub profile: MccProfile,
    pub rows: Vec<PlanRow>,
    /// MCC buttons with no HMS action: `(MCC's name, what it is bound to)`.
    pub unmapped_mcc: Vec<(String, String)>,
    /// HMS actions with no MCC button: `(the action, why)`.
    pub unmapped_hms: Vec<(Action, &'static str)>,
    pub settings: Vec<SettingRow>,
    /// #mcc-fov  MCC's first-person field of view for this title, in degrees, when the file states
    /// one that HMS can actually use.
    ///
    /// Filtered to HMS's own `FOV_MIN_DEG..=FOV_MAX_DEG` at read time rather than clamped: a value
    /// outside that range means the file is not saying what we think it is, and silently clamping
    /// it would import a number the user never chose. `None` = nothing to apply, and the row is
    /// not offered.
    pub fov_deg: Option<f32>,
}

impl Plan {
    /// Find MCC's settings and build a plan for one title. `None` when no settings file was found.
    ///
    /// **Call this only when the user asks for it.**
    pub fn discover(game: MccGame, keys: &Keybinds) -> Option<Result<Plan, String>> {
        let found = discover().into_iter().max_by_key(|f| {
            // Most recently written wins: this machine has several MCC prefixes and only one of
            // them is the install actually being played.
            std::fs::metadata(&f.ini).and_then(|m| m.modified()).ok()
        })?;
        Some(Plan::from_path(&found, game, keys))
    }

    pub fn from_path(found: &Found, game: MccGame, keys: &Keybinds) -> Result<Plan, String> {
        let ini = UeIni::read(&found.ini)
            .map_err(|e| format!("could not read MCC's settings file: {e}"))?;
        Plan::from_ini(&ini, found.clone(), game, keys)
    }

    /// Build the plan from an already-parsed ini.
    ///
    /// Refuses — rather than half-applying — when the file does not actually contain this title's
    /// bindings. MCC may be running and rewriting the file as it is read, so "the line is missing
    /// or cut off" is a case that happens, and the right answer to it is to change nothing and say
    /// so.
    pub fn from_ini(
        ini: &UeIni,
        source: Found,
        game: MccGame,
        keys: &Keybinds,
    ) -> Result<Plan, String> {
        let key = format!("CustomKeyboardMouseMappingV2[{}]", game.index());
        let Some(value) = ini.get(&key) else {
            return Err(format!(
                "MCC's settings file has no keyboard bindings for {} — nothing was changed. \
                 (If Halo: The Master Chief Collection is running, it may be rewriting the file; \
                 try again once it has saved.)",
                game.label()
            ));
        };
        let profile = MccProfile::parse(game, value);
        if profile.bindings.is_empty() {
            return Err(format!(
                "MCC's {} bindings could not be read — the line is present but empty or cut off. \
                 Nothing was changed.",
                game.label()
            ));
        }

        let mut rows = Vec::new();
        let mut consumed: Vec<u32> = Vec::new();
        for rule in &ACTION_MAP {
            let outcome = match profile.get(rule.button) {
                Some(b) => to_binding(&b.vks, rule.by_position),
                None => Outcome::Unbound,
            };
            consumed.push(rule.button);
            rows.push(PlanRow {
                action: rule.action,
                mcc_button: ABSTRACT_BUTTONS
                    .get(rule.button as usize)
                    .map(|s| (*s).to_string())
                    .unwrap_or_else(|| format!("AbstractButton {}", rule.button)),
                why: rule.why,
                before: keys.bind(rule.action).label(),
                outcome,
            });
        }

        // Everything MCC has that HMS does not — named, with its key, so a partial import is
        // never silent about the other 57 buttons.
        let mut unmapped_mcc = Vec::new();
        for b in &profile.bindings {
            if consumed.contains(&b.button) {
                continue;
            }
            let label = match to_binding(&b.vks, true) {
                Outcome::Ok(x) => x.label(),
                Outcome::Unbound => "not bound".into(),
                Outcome::Reserved(_) => "reserved by Halo Map Studio".into(),
                Outcome::Unsupported(why) => why,
            };
            unmapped_mcc.push((b.name(), label));
        }

        let mapped: Vec<Action> = ACTION_MAP.iter().map(|r| r.action).collect();
        let unmapped_hms: Vec<(Action, &'static str)> = Action::ALL
            .into_iter()
            .filter(|a| !mapped.contains(a))
            .map(|a| (a, why_no_mcc_equivalent(a)))
            .collect();

        Ok(Plan {
            source,
            game,
            profile,
            rows,
            unmapped_mcc,
            unmapped_hms,
            settings: read_settings(ini, game),
            fov_deg: usable_fov(ini, game),
        })
    }

    /// How many bindings would actually change.
    pub fn change_count(&self) -> usize {
        self.rows.iter().filter(|r| r.changes()).count()
    }

    /// How many of the mapped rows could not be imported at all.
    pub fn refused_count(&self) -> usize {
        self.rows.iter().filter(|r| r.outcome.binding().is_none()).count()
    }

    /// One line for the button and the status area: what was found and what it would do.
    pub fn summary(&self) -> String {
        format!(
            "{} of {} controls match a Halo Map Studio action; {} would change. {} MCC buttons and \
             {} Halo Map Studio actions have no counterpart.",
            self.rows.len() - self.refused_count(),
            self.rows.len(),
            self.change_count(),
            self.unmapped_mcc.len(),
            self.unmapped_hms.len(),
        )
    }

    /// Apply the plan. Returns an [`Undo`] holding the exact table from before.
    ///
    /// Rows that could not be imported change nothing — there is no partial state where half the
    /// camera moved and half did not, because the whole table is replaced from one snapshot.
    pub fn apply(&self, keys: &mut Keybinds) -> Undo {
        let undo = Undo { before: keys.clone(), fov_before: None, summary: self.summary() };
        for row in &self.rows {
            if let Some(b) = row.outcome.binding() {
                // The primary only. An action's alternative is HMS's own second key (Delete *or*
                // Backspace, fly-up on E *or* R) and MCC has nothing to say about it, so it is
                // kept rather than cleared.
                let mut bind = keys.bind(row.action);
                bind.primary = b;
                keys.set_bind(row.action, bind);
            }
        }
        undo
    }
}

/// The table as it was before an import, so the import is reversible.
#[derive(Clone, Debug)]
pub struct Undo {
    before: Keybinds,
    /// #mcc-fov  The first-person FOV as it was before the import, so Undo restores it. `None`
    /// when the import did not touch it, so Undo cannot assert a value it never replaced.
    pub fov_before: Option<f32>,
    /// What the import did, for a status line that outlives the preview.
    #[allow(dead_code)]
    pub summary: String,
}

impl Undo {
    /// Put the previous bindings back, exactly.
    pub fn revert(&self, keys: &mut Keybinds) {
        let listening = keys.listening;
        *keys = self.before.clone();
        keys.listening = listening;
    }
}

// ===================== the Settings-page section =====================

/// State the import section owns, hung off the controls page.
#[derive(Default)]
pub struct ImportUi {
    /// The plan being previewed. `None` until the user presses the find button, and `None` again
    /// the moment one is applied — see [`ImportUi::apply`].
    pub plan: Option<Plan>,
    /// The last error from a find, shown in place of a plan.
    pub error: Option<String>,
    /// Set after an apply, so the import can be undone.
    pub undo: Option<Undo>,
    /// What the last apply did, shown where the preview was.
    pub applied: Option<String>,
    pub game: Option<MccGame>,
    pub show_unmapped: bool,
    /// #mcc-fov  The live first-person FOV, passed IN by the host before [`ui`] runs and read back
    /// OUT after it.
    ///
    /// A channel rather than a parameter so that adding it did not change `ui`'s signature or any
    /// of this module's existing tests: a caller that never sets it sees `None` and the FOV is
    /// simply not imported. The host writes the current value, `apply` replaces it with MCC's and
    /// records the old one in the [`Undo`], and `undo_last` puts it back.
    pub fov: Option<f32>,
}

impl ImportUi {
    /// Apply the previewed plan and move to the applied state.
    ///
    /// **Clearing the plan is the point.** A preview's "Now" column is captured when the plan is
    /// built, so it is stale the instant the plan is applied — and worse, pressing Apply a second
    /// time would take a fresh [`Undo`] snapshot of the ALREADY-imported table and overwrite the
    /// one that could get back to where the user started. Dropping the plan makes that
    /// unreachable: after an apply the only buttons are "Undo the import" and a fresh find.
    pub fn apply(&mut self, keys: &mut Keybinds) {
        let Some(plan) = self.plan.take() else { return };
        let mut undo = plan.apply(keys);
        self.adopt_fov(plan.fov_deg, &mut undo);
        self.applied = Some(format!(
            "Imported {} of MCC's {} controls. {}",
            plan.change_count(),
            plan.rows.len(),
            undo.summary
        ));
        self.undo = Some(undo);
    }

    /// #mcc-fov  Take MCC's field of view, recording the current one in the [`Undo`].
    ///
    /// Snapshots the live value **before** overwriting it, and **only when there is something to
    /// overwrite it with** -- so `Undo` never restores a value the import did not set. A named
    /// method rather than three lines inside [`ImportUi::apply`] so a gate can drive the real
    /// thing: `apply` needs a whole [`Plan`], which needs a bindings line, which has nothing to do
    /// with the field of view.
    fn adopt_fov(&mut self, mcc: Option<f32>, undo: &mut Undo) {
        if let Some(v) = mcc {
            undo.fov_before = self.fov;
            self.fov = Some(v);
        }
    }

    /// Put the bindings back as they were before the last apply. Returns true if it did anything.
    pub fn undo_last(&mut self, keys: &mut Keybinds) -> bool {
        let Some(u) = self.undo.take() else { return false };
        u.revert(keys);
        if let Some(before) = u.fov_before {
            self.fov = Some(before);
        }
        self.applied = None;
        true
    }
}

const WARN: egui::Color32 = egui::Color32::from_rgb(0xE0, 0xA0, 0x40);
const GOOD: egui::Color32 = egui::Color32::from_rgb(0x80, 0xC0, 0x80);

/// Draw the "Import from MCC" section of Settings ▸ Controls & keyboard. Returns true when the
/// binding table changed (so the caller persists it).
pub fn ui(st: &mut ImportUi, keys: &mut Keybinds, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    let game = *st.game.get_or_insert(MccGame::HaloReach);

    ui.label(egui::RichText::new("IMPORT FROM HALO: THE MASTER CHIEF COLLECTION").small().strong());
    ui.label(
        egui::RichText::new(
            "Reads your own MCC settings and shows what it would change before changing anything. \
             Your MCC files are never written to.",
        )
        .small()
        .weak(),
    );

    ui.horizontal_wrapped(|ui| {
        egui::ComboBox::from_id_salt("mcc_import_game")
            .selected_text(game.label())
            .show_ui(ui, |ui| {
                for g in MccGame::ALL {
                    if ui.selectable_label(game == g, g.label()).clicked() {
                        st.game = Some(g);
                        st.plan = None;
                        st.error = None;
                    }
                }
            });
        if ui
            .button("Find MCC's controls")
            .on_hover_text(
                "Looks for MCC's GameUserSettings.ini on this computer — in a Windows install and \
                 in every Steam / Proton prefix. Nothing is changed by looking.",
            )
            .clicked()
        {
            st.error = None;
            st.plan = None;
            match Plan::discover(game, keys) {
                None => {
                    st.error = Some(
                        "No MCC settings file was found on this computer. MCC keeps it in \
                         AppData/LocalLow/MCC/Saved/Config/WindowsNoEditor/GameUserSettings.ini."
                            .into(),
                    )
                }
                Some(Err(e)) => st.error = Some(e),
                Some(Ok(p)) => st.plan = Some(p),
            }
        }
        if st.undo.is_some()
            && ui
                .button("Undo the import")
                .on_hover_text("Puts every control back exactly as it was before the import.")
                .clicked()
        {
            changed |= st.undo_last(keys);
        }
    });

    if let Some(e) = st.error.as_deref() {
        ui.label(egui::RichText::new(e).small().color(WARN));
    }
    if let Some(a) = st.applied.as_deref() {
        ui.label(egui::RichText::new(a).small().color(GOOD));
    }

    // Taken out for the draw so the Apply button can clear it (see `ImportUi::apply`), and put
    // back at the end when it was not applied.
    let Some(plan) = st.plan.take() else {
        return changed;
    };
    let mut applied = false;

    ui.add_space(4.0);
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.label(
            egui::RichText::new(format!("{} controls, from {}", plan.game.label(), plan.source.label))
                .strong(),
        );
        ui.label(egui::RichText::new(plan.summary()).small());

        ui.add_space(4.0);
        egui::Grid::new("mcc_import_rows").num_columns(4).striped(true).show(ui, |ui| {
            ui.label(egui::RichText::new("Action").small().strong());
            ui.label(egui::RichText::new("Now").small().strong());
            ui.label(egui::RichText::new("From MCC").small().strong());
            ui.label(egui::RichText::new("MCC's button").small().strong());
            ui.end_row();
            for row in &plan.rows {
                ui.label(row.action.spec().label);
                ui.label(egui::RichText::new(&row.before).weak());
                let after = row.after();
                let text = egui::RichText::new(after);
                ui.label(if row.outcome.binding().is_some() {
                    if row.changes() {
                        text.color(GOOD).strong()
                    } else {
                        text.weak()
                    }
                } else {
                    text.color(WARN)
                });
                ui.label(egui::RichText::new(&row.mcc_button).small().weak())
                    .on_hover_text(row.why);
                ui.end_row();
            }
        });

        ui.add_space(4.0);
        let n = plan.change_count();
        let apply = ui
            .add_enabled(n > 0, egui::Button::new(format!("Import these {n} controls")))
            .on_hover_text("Changes only the rows marked above. Undo puts them all back.");
        if apply.clicked() {
            applied = true;
            changed = true;
        }
        if n == 0 {
            ui.label(
                egui::RichText::new("Your controls already match MCC's — nothing to import.")
                    .small()
                    .weak(),
            );
        }
    });

    // ---- what could NOT be mapped ----
    ui.add_space(4.0);
    let unmapped = plan.unmapped_mcc.len() + plan.unmapped_hms.len();
    ui.checkbox(
        &mut st.show_unmapped,
        format!("Show the {unmapped} controls with no counterpart"),
    );
    if st.show_unmapped {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{} Halo Map Studio actions MCC has no button for",
                    plan.unmapped_hms.len()
                ))
                .small()
                .strong(),
            );
            for (a, why) in &plan.unmapped_hms {
                ui.label(
                    egui::RichText::new(format!("• {} — {why}", a.spec().label)).small().weak(),
                );
            }
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "{} MCC buttons Halo Map Studio has no action for",
                    plan.unmapped_mcc.len()
                ))
                .small()
                .strong(),
            );
            for (name, key) in &plan.unmapped_mcc {
                ui.label(egui::RichText::new(format!("• {name} — {key}")).small().weak());
            }
        });
    }

    // ---- the non-binding settings ----
    if !plan.settings.is_empty() {
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new("MCC's other settings").small().strong());
            for s in &plan.settings {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(format!("{}: ", s.label)).small());
                    ui.label(egui::RichText::new(&s.value).small().strong());
                    ui.label(
                        egui::RichText::new(match s.disposition {
                            Disposition::ReportOnly => "(shown only)",
                            Disposition::NotYetWired => "(not imported yet)",
                            Disposition::Applied => "(imported)",
                        })
                        .small()
                        .weak(),
                    );
                });
                ui.label(egui::RichText::new(&s.note).small().weak());
            }
        });
    }

    st.plan = Some(plan);
    if applied {
        st.apply(keys);
    }
    changed
}

// ===================== tests =====================

#[cfg(test)]
mod tests {
    use super::*;

    /// `discover()` consults the process-wide `HMS_MCC_SETTINGS`, and cargo runs the tests in
    /// one process on several threads — so a test that sets that variable would otherwise be seen
    /// by a test that reads the real install. Every test that calls `discover()` takes this lock,
    /// which is what makes `the_override_env_var_is_honoured` safe to run beside the others.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
        L.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The real file on this machine, if it is here. Tests that need it skip when it is not, so
    /// the suite still passes on a machine with no MCC install — but on THIS machine they run, and
    /// `the_real_ini_on_this_machine_parses` says out loud when it skipped.
    fn real_ini() -> Option<(Found, UeIni)> {
        let f = discover().into_iter().max_by_key(|f| {
            std::fs::metadata(&f.ini).and_then(|m| m.modified()).ok()
        })?;
        let ini = UeIni::read(&f.ini).ok()?;
        Some((f, ini))
    }

    // ---------- the recovered enum ----------

    #[test]
    fn the_enum_table_is_mccs_own_count() {
        // MCC's `kGameAbstractButton_CountTotal` terminates the name table at 66.
        assert_eq!(ABSTRACT_BUTTONS.len(), 66);
        assert_eq!(ABSTRACT_BUTTONS[0], "Jump");
        assert_eq!(ABSTRACT_BUTTONS[65], "NextGrenade");
        // No duplicates: a duplicate would mean the block was mis-sliced out of the exe.
        let mut sorted = ABSTRACT_BUTTONS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ABSTRACT_BUTTONS.len(), "the name table has a duplicate");
    }

    #[test]
    fn the_named_button_ids_match_the_table() {
        assert_eq!(ABSTRACT_BUTTONS[btn::JUMP as usize], "Jump");
        assert_eq!(ABSTRACT_BUTTONS[btn::RELOAD as usize], "Reload");
        assert_eq!(ABSTRACT_BUTTONS[btn::FIRE as usize], "Fire");
        assert_eq!(ABSTRACT_BUTTONS[btn::MOVE_FORWARD as usize], "MoveForward");
        assert_eq!(ABSTRACT_BUTTONS[btn::MOVE_BACKWARD as usize], "MoveBackward");
        assert_eq!(ABSTRACT_BUTTONS[btn::STRAFE_LEFT as usize], "StrafeLeft");
        assert_eq!(ABSTRACT_BUTTONS[btn::STRAFE_RIGHT as usize], "StrafeRight");
        assert_eq!(ABSTRACT_BUTTONS[btn::LIFT_EDITOR as usize], "LiftEditor");
        assert_eq!(ABSTRACT_BUTTONS[btn::DROP_EDITOR as usize], "DropEditor");
        assert_eq!(ABSTRACT_BUTTONS[btn::GRAB_OBJECT_EDITOR as usize], "GrabObjectEditor");
        assert_eq!(ABSTRACT_BUTTONS[btn::BOOST_EDITOR as usize], "BoostEditor");
        assert_eq!(ABSTRACT_BUTTONS[btn::DELETE_OBJECT_EDITOR as usize], "DeleteObjectEditor");
        assert_eq!(ABSTRACT_BUTTONS[btn::DUPLICATE_OBJECT_EDITOR as usize], "DuplicateObjectEditor");
    }

    /// A hand-written stand-in for the bindings MCC actually writes, so the order check below runs
    /// everywhere — not only on a machine that has MCC installed. The key codes are the ones
    /// measured in the real file.
    const SAMPLE_PROFILE: &str = concat!(
        "(GameKeyboardMouseMappings[0]=(AbstractButton=0,VirtualKeyCodes[0]=32,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[1]=(AbstractButton=3,VirtualKeyCodes[0]=82,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[2]=(AbstractButton=8,VirtualKeyCodes[0]=1,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        // Crouch on the right mouse, melee on the middle, sprint on C -- the Reach values measured
        // in the real file, which is also what exercises a MOUSE import into a key action.
        "GameKeyboardMouseMappings[12]=(AbstractButton=9,VirtualKeyCodes[0]=2,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[13]=(AbstractButton=5,VirtualKeyCodes[0]=4,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[14]=(AbstractButton=14,VirtualKeyCodes[0]=67,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[3]=(AbstractButton=16,VirtualKeyCodes[0]=87,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[4]=(AbstractButton=17,VirtualKeyCodes[0]=83,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[5]=(AbstractButton=18,VirtualKeyCodes[0]=65,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[6]=(AbstractButton=19,VirtualKeyCodes[0]=68,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[7]=(AbstractButton=25,VirtualKeyCodes[0]=82,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[8]=(AbstractButton=26,VirtualKeyCodes[0]=70,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[9]=(AbstractButton=28,VirtualKeyCodes[0]=160,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[10]=(AbstractButton=30,VirtualKeyCodes[0]=46,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0),",
        "GameKeyboardMouseMappings[11]=(AbstractButton=52,VirtualKeyCodes[0]=71,VirtualKeyCodes[1]=0,VirtualKeyCodes[2]=0,VirtualKeyCodes[3]=0,VirtualKeyCodes[4]=0))",
    );

    /// THE ORDER CHECK. Four consecutive ids landing on exactly W / S / A / D, plus Delete on the
    /// delete button, Space on Jump, R on Reload and the left mouse on Fire. A wrong enum order
    /// cannot satisfy all of those at once.
    #[test]
    fn the_recovered_enum_order_matches_a_real_profile() {
        let p = MccProfile::parse(MccGame::HaloReach, SAMPLE_PROFILE);
        let expect: &[(u32, &str, &str)] = &[
            (btn::JUMP, "Jump", "Space"),
            (btn::RELOAD, "Reload", "R"),
            (btn::MOVE_FORWARD, "MoveForward", "W"),
            (btn::MOVE_BACKWARD, "MoveBackward", "S"),
            (btn::STRAFE_LEFT, "StrafeLeft", "A"),
            (btn::STRAFE_RIGHT, "StrafeRight", "D"),
            (btn::DELETE_OBJECT_EDITOR, "DeleteObjectEditor", "Del"),
            (btn::DUPLICATE_OBJECT_EDITOR, "DuplicateObjectEditor", "G"),
        ];
        for (id, name, key) in expect {
            let b = p.get(*id).unwrap_or_else(|| panic!("{name} ({id}) missing"));
            assert_eq!(b.name(), *name);
            let got = to_binding(&b.vks, false).binding().expect("bindable").label();
            assert_eq!(&got, key, "{name} should be {key}, got {got}");
        }
        // And the mouse / modifier shapes.
        assert_eq!(
            to_binding(&p.get(btn::FIRE).unwrap().vks, false),
            Outcome::Ok(Binding::mouse(MouseBtn::Left)),
            "Fire is the left mouse button"
        );
        assert_eq!(
            to_binding(&p.get(btn::BOOST_EDITOR).unwrap().vks, false),
            Outcome::Ok(Binding::chord(Mods::SHIFT)),
            "BoostEditor is Shift"
        );
    }

    // ---------- the parser ----------

    #[test]
    fn the_ini_parser_reads_keys_and_sections() {
        let ini = UeIni::parse(
            "[ScalabilityGroups]\nsg.ShadowQuality=3\n\n; a comment\n\
             [/Script/MCC.MCCGameUserSettings]\nFOVSetting[5]=120\nBrightnessPerGame[5]=-0.900000\n\
             bUseVSync=True\nMKTheaterSpeed=1\n",
        );
        assert_eq!(ini.get("sg.ShadowQuality"), Some("3"));
        assert_eq!(ini.get_indexed("FOVSetting", MccGame::HaloReach), Some("120"));
        assert_eq!(ini.get_indexed("FOVSetting", MccGame::Halo1), None);
        assert_eq!(ini.get_f32("BrightnessPerGame[5]"), Some(-0.9));
        assert_eq!(ini.get_bool("bUseVSync"), Some(true));
        assert_eq!(ini.get_i32("MKTheaterSpeed"), Some(1));
        assert_eq!(ini.get("nope"), None);
    }

    #[test]
    fn a_later_duplicate_key_wins() {
        let ini = UeIni::parse("[a]\nK=1\n[b]\nK=2\n");
        assert_eq!(ini.get("K"), Some("2"));
    }

    #[test]
    fn the_profile_parser_strips_the_zero_padding() {
        let p = MccProfile::parse(MccGame::HaloReach, SAMPLE_PROFILE);
        assert_eq!(p.bindings.len(), 15);
        for b in &p.bindings {
            assert!(!b.vks.contains(&0), "a zero slot survived on {}", b.name());
            assert!(b.vks.len() <= VK_SLOTS);
        }
        assert_eq!(p.get(btn::MOVE_FORWARD).unwrap().vks, vec![87]);
    }

    #[test]
    fn a_chord_of_two_virtual_keys_becomes_a_modified_binding() {
        // Ctrl (0xA2) + Z (0x5A). MCC can write this; HMS expresses it as Ctrl+Z.
        assert_eq!(
            to_binding(&[0xA2, 0x5A], false),
            Outcome::Ok(Binding::key(egui::Key::Z).with(Mods::CTRL))
        );
    }

    #[test]
    fn two_real_keys_at_once_is_refused_not_guessed() {
        match to_binding(&[0x41, 0x42], false) {
            Outcome::Unsupported(why) => assert!(why.contains("two keys")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn mccs_own_wheel_codes_are_reported_not_invented() {
        for code in [174u32, 175] {
            assert_eq!(translate_vk(code), Vk::Unknown(code));
            match to_binding(&[code], false) {
                Outcome::Unsupported(why) => {
                    assert!(why.contains("own key code"), "{why}");
                    assert!(why.contains("mouse wheel"), "{why}");
                }
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_reserved_keys_are_refused_with_hmss_own_reason() {
        // Escape (0x1B) and Tab (0x09) are keys `keybinds::reserved` keeps fixed.
        for vk in [0x1Bu32, 0x09] {
            match to_binding(&[vk], false) {
                Outcome::Reserved(why) => assert!(!why.is_empty()),
                other => panic!("vk {vk:#x} should be reserved, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_extra_mouse_buttons_are_reported() {
        for vk in [5u32, 6] {
            match to_binding(&[vk], false) {
                Outcome::Unsupported(why) => assert!(why.contains("mouse button"), "{why}"),
                other => panic!("vk {vk} should be unsupported, got {other:?}"),
            }
        }
    }

    /// Every virtual-key code [`translate_vk`] claims to handle, and the egui key it claims. If
    /// this list and the function ever disagree, the import silently binds the wrong key.
    #[test]
    fn every_handled_vk_round_trips() {
        use egui::Key as K;
        let cases: &[(u32, Vk)] = &[
            (0x01, Vk::Mouse(MouseBtn::Left)),
            (0x02, Vk::Mouse(MouseBtn::Right)),
            (0x04, Vk::Mouse(MouseBtn::Middle)),
            (0x08, Vk::Key(K::Backspace)),
            (0x09, Vk::Key(K::Tab)),
            (0x0D, Vk::Key(K::Enter)),
            (0x1B, Vk::Key(K::Escape)),
            (0x20, Vk::Key(K::Space)),
            (0x21, Vk::Key(K::PageUp)),
            (0x22, Vk::Key(K::PageDown)),
            (0x23, Vk::Key(K::End)),
            (0x24, Vk::Key(K::Home)),
            (0x25, Vk::Key(K::ArrowLeft)),
            (0x26, Vk::Key(K::ArrowUp)),
            (0x27, Vk::Key(K::ArrowRight)),
            (0x28, Vk::Key(K::ArrowDown)),
            (0x2D, Vk::Key(K::Insert)),
            (0x2E, Vk::Key(K::Delete)),
            (0x30, Vk::Key(K::Num0)),
            (0x39, Vk::Key(K::Num9)),
            (0x41, Vk::Key(K::A)),
            (0x5A, Vk::Key(K::Z)),
            (0x60, Vk::Key(K::Num0)),
            (0x69, Vk::Key(K::Num9)),
            (0x6B, Vk::Key(K::Plus)),
            (0x6D, Vk::Key(K::Minus)),
            (0x6E, Vk::Key(K::Period)),
            (0x6F, Vk::Key(K::Slash)),
            (0x70, Vk::Key(K::F1)),
            (0x87, Vk::Key(K::F24)),
            (0x10, Vk::Modifier(Mods::SHIFT)),
            (0xA0, Vk::Modifier(Mods::SHIFT)),
            (0xA1, Vk::Modifier(Mods::SHIFT)),
            (0x11, Vk::Modifier(Mods::CTRL)),
            (0xA2, Vk::Modifier(Mods::CTRL)),
            (0xA3, Vk::Modifier(Mods::CTRL)),
            (0x12, Vk::Modifier(Mods::ALT)),
            (0xA4, Vk::Modifier(Mods::ALT)),
            (0xA5, Vk::Modifier(Mods::ALT)),
            (0xBA, Vk::Key(K::Semicolon)),
            (0xBB, Vk::Key(K::Equals)),
            (0xBC, Vk::Key(K::Comma)),
            (0xBD, Vk::Key(K::Minus)),
            (0xBE, Vk::Key(K::Period)),
            (0xBF, Vk::Key(K::Slash)),
            (0xDB, Vk::Key(K::OpenBracket)),
            (0xDC, Vk::Key(K::Backslash)),
            (0xDD, Vk::Key(K::CloseBracket)),
            (0xDE, Vk::Key(K::Quote)),
        ];
        for (vk, want) in cases {
            assert_eq!(translate_vk(*vk), *want, "vk {vk:#04X}");
        }
        // The letter / digit / function ranges, every code, not just the ends.
        for (i, k) in LETTERS.iter().enumerate() {
            assert_eq!(translate_vk(0x41 + i as u32), Vk::Key(*k));
        }
        for (i, k) in DIGITS.iter().enumerate() {
            assert_eq!(translate_vk(0x30 + i as u32), Vk::Key(*k));
            assert_eq!(translate_vk(0x60 + i as u32), Vk::Key(*k), "numpad {i}");
        }
        for (i, k) in FUNCTION.iter().enumerate() {
            assert_eq!(translate_vk(0x70 + i as u32), Vk::Key(*k));
        }
        // The named no-trigger codes.
        for vk in [0x05u32, 0x06, 0x13, 0x14, 0x2C, 0x6A, 0x90, 0x91] {
            assert!(
                matches!(translate_vk(vk), Vk::NoHmsTrigger(_)),
                "vk {vk:#04X} should be a named no-trigger code"
            );
        }
        // And a code nobody handles.
        assert_eq!(translate_vk(0xFE), Vk::Unknown(0xFE));
    }

    /// `to_binding` must also produce the LABEL the preview shows, not only the right key: the
    /// preview is the only thing the user sees before they press Apply.
    #[test]
    fn the_preview_label_is_the_imported_key() {
        assert_eq!(to_binding(&[87], true).binding().unwrap().label(), "W");
        assert_eq!(to_binding(&[0x2E], false).binding().unwrap().label(), "Del");
        assert_eq!(to_binding(&[0xA0], false).binding().unwrap().label(), "Shift");
        assert_eq!(to_binding(&[1], false).binding().unwrap().label(), "Left mouse");
        // By position vs by letter is recorded, and the serialized form shows it.
        assert_eq!(to_binding(&[87], true).binding().unwrap().to_spec(), "pos:W");
        assert_eq!(to_binding(&[87], false).binding().unwrap().to_spec(), "W");
    }

    // ---------- the plan ----------

    fn plan_from(text: &str, game: MccGame, keys: &Keybinds) -> Result<Plan, String> {
        let ini = UeIni::parse(text);
        let src = Found { ini: PathBuf::from("test"), label: "test".into() };
        Plan::from_ini(&ini, src, game, keys)
    }

    fn sample_ini() -> String {
        format!(
            "[/Script/MCC.MCCGameUserSettings]\n\
             CustomKeyboardMouseMappingV2[5]={SAMPLE_PROFILE}\n\
             FOVSetting[5]=120\nMouseSensitivity[5]=7.200000\nBrightnessPerGame[5]=-0.900000\n\
             MouseLookControlsInverted[5]=0\n\
             FlyingCameraSpeed[5]=1\nFlyingCameraThrust[5]=1\nFlyingCameraTurnSensitivity[5]=1\n\
             FlyingCameraPanning[5]=0\n\
             MKTheaterSpeed=1\nMKTheaterThrust=1\nMKTheaterTurnSensitivity=1\nMKTheaterPanning=0\n\
             MasterVolume=100.000000\nAudioEffectsVolume=100.000000\n\
             AudioInGameMusicVolume=100.000000\nMenuMusicVolume=0.000000\n\
             bUseVSync=True\nFrameRateLimit=0.000000\nGameResolutionScale=1.000000\n"
        )
    }

    #[test]
    fn a_plan_maps_the_camera_and_the_editor_commands() {
        let keys = Keybinds::default();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        assert_eq!(plan.rows.len(), ACTION_MAP.len());
        let find = |a: Action| plan.rows.iter().find(|r| r.action == a).unwrap();
        assert_eq!(find(Action::CamForward).after(), "W");
        assert_eq!(find(Action::CamBack).after(), "S");
        assert_eq!(find(Action::CamLeft).after(), "A");
        assert_eq!(find(Action::CamRight).after(), "D");
        assert_eq!(find(Action::CamUp).after(), "R");
        assert_eq!(find(Action::CamDown).after(), "F");
        assert_eq!(find(Action::FlyFaster).after(), "Shift");
        assert_eq!(find(Action::Delete).after(), "Del");
        assert_eq!(find(Action::Duplicate).after(), "G");
        // The test mode's gameplay keys, including the two that are MOUSE buttons in MCC.
        assert_eq!(find(Action::PlayJump).after(), "Space");
        assert_eq!(find(Action::PlayCrouch).after(), "Right mouse");
        assert_eq!(find(Action::PlayMelee).after(), "Middle mouse");
        assert_eq!(find(Action::PlaySprint).after(), "C");
        // Every mapped row is importable from this profile.
        assert_eq!(plan.refused_count(), 0);
        // And the rows that already match the shipped default do not count as changes: the
        // defaults are W/A/S/D, Shift, Delete.
        assert!(!find(Action::CamForward).changes());
        assert!(find(Action::CamUp).changes(), "default fly-up is E, MCC's is R");
        assert!(find(Action::Duplicate).changes(), "default duplicate is Shift+D, MCC's is G");
    }

    #[test]
    fn every_action_is_accounted_for_as_mapped_or_explained() {
        let keys = Keybinds::default();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        let mapped: Vec<Action> = ACTION_MAP.iter().map(|r| r.action).collect();
        assert_eq!(mapped.len() + plan.unmapped_hms.len(), Action::ALL.len());
        for (a, why) in &plan.unmapped_hms {
            assert_ne!(*why, "imported", "{a:?} is unmapped but says it was imported");
            assert!(!why.is_empty(), "{a:?} has no reason");
        }
        // And nothing is in both lists.
        for (a, _) in &plan.unmapped_hms {
            assert!(!mapped.contains(a), "{a:?} is both mapped and unmapped");
        }
    }

    #[test]
    fn the_unmapped_mcc_buttons_are_named_with_their_keys() {
        let keys = Keybinds::default();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        // The buttons the sample carries that ACTION_MAP does NOT consume must each be named
        // with the key they are on, so a partial import is never silent about them.
        let names: Vec<&str> = plan.unmapped_mcc.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Reload"), "{names:?}");
        assert!(names.contains(&"Fire"), "{names:?}");
        let fire = plan.unmapped_mcc.iter().find(|(n, _)| n == "Fire").unwrap();
        assert_eq!(fire.1, "Left mouse");
        // ...and a button ACTION_MAP DOES consume must not also be listed as unmapped.
        assert!(!names.contains(&"Jump"), "Jump is imported and must not be listed: {names:?}");
        assert!(!names.contains(&"MoveForward"), "{names:?}");
    }

    #[test]
    fn the_whole_decoded_profile_is_public_for_a_test_mode() {
        let keys = Keybinds::default();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        // A first-person controller reads Jump and Fire straight off the profile.
        let jump = plan.profile.get(btn::JUMP).expect("Jump is in the profile");
        assert_eq!(to_binding(&jump.vks, true).binding().unwrap().label(), "Space");
        let fire = plan.profile.get(btn::FIRE).expect("Fire is in the profile");
        assert_eq!(to_binding(&fire.vks, false), Outcome::Ok(Binding::mouse(MouseBtn::Left)));
    }

    // ---------- the refusals ----------

    #[test]
    fn a_missing_profile_refuses_and_changes_nothing() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        // The title asked for is not in the file.
        let e = plan_from(&sample_ini(), MccGame::Halo3, &keys).expect_err("should refuse");
        assert!(e.contains("Halo 3"), "{e}");
        assert!(e.contains("nothing was changed"), "{e}");
        // Nothing may have been written on the way to that refusal.
        assert_eq!(keys, before);
        let _ = &mut keys;
    }

    #[test]
    fn an_empty_profile_line_refuses() {
        let keys = Keybinds::default();
        let e = plan_from(
            "[x]\nCustomKeyboardMouseMappingV2[5]=()\n",
            MccGame::HaloReach,
            &keys,
        )
        .expect_err("should refuse");
        assert!(e.contains("empty or cut off"), "{e}");
    }

    /// A file cut off mid-write must not panic and must not half-apply: either the whole mapping
    /// line is there and the entries before the cut import, or the line is gone and the import
    /// refuses. Every truncation point is tried.
    #[test]
    fn a_truncated_ini_never_panics_and_never_half_applies() {
        let full = sample_ini();
        let before = Keybinds::default();
        for n in 0..=full.len() {
            let cut = &full[..n];
            let keys = before.clone();
            match plan_from(cut, MccGame::HaloReach, &keys) {
                Err(_) => {}
                Ok(plan) => {
                    // A plan built from a cut file is still internally consistent...
                    assert_eq!(plan.rows.len(), ACTION_MAP.len(), "cut at {n}");
                    // ...and applying it is all-or-nothing against ITS OWN rows.
                    let mut k = before.clone();
                    let undo = plan.apply(&mut k);
                    for row in &plan.rows {
                        match row.outcome.binding() {
                            Some(b) => assert_eq!(k.bind(row.action).primary, b, "cut at {n}"),
                            None => assert_eq!(
                                k.bind(row.action),
                                before.bind(row.action),
                                "a refused row changed something, cut at {n}"
                            ),
                        }
                    }
                    undo.revert(&mut k);
                    assert_eq!(k, before, "undo must restore, cut at {n}");
                }
            }
        }
    }

    #[test]
    fn rubbish_input_never_panics() {
        let keys = Keybinds::default();
        for junk in [
            "",
            "\0\0\0",
            "[",
            "[unclosed\nFOVSetting[5]=120\n",
            "=",
            "CustomKeyboardMouseMappingV2[5]=AbstractButton=",
            "CustomKeyboardMouseMappingV2[5]=AbstractButton=99999999999999999999",
            "CustomKeyboardMouseMappingV2[5]=(AbstractButton=16,VirtualKeyCodes[0]=",
            "CustomKeyboardMouseMappingV2[5]=(AbstractButton=70000,VirtualKeyCodes[0]=87)",
            "CustomKeyboardMouseMappingV2[5]=(AbstractButton=16,VirtualKeyCodes[0]=999999)",
        ] {
            let _ = plan_from(junk, MccGame::HaloReach, &keys);
            let _ = UeIni::parse(junk);
        }
        // An id past the end of MCC's enum is KEPT and NAMED, not indexed out of bounds and not
        // silently dropped — a dropped binding is a quiet partial import.
        let p = MccProfile::parse(
            MccGame::HaloReach,
            "(AbstractButton=70000,VirtualKeyCodes[0]=87,VirtualKeyCodes[1]=0)",
        );
        assert_eq!(p.bindings.len(), 1);
        assert_eq!(p.bindings[0].button, 70000);
        assert_eq!(p.bindings[0].name(), "AbstractButton 70000");
        // So is a key code nobody knows.
        let p = MccProfile::parse(
            MccGame::HaloReach,
            "(AbstractButton=16,VirtualKeyCodes[0]=999999,VirtualKeyCodes[1]=0)",
        );
        assert_eq!(p.bindings[0].vks, vec![999999]);
        assert!(matches!(to_binding(&p.bindings[0].vks, true), Outcome::Unsupported(_)));
    }

    // ---------- reversibility ----------

    #[test]
    fn an_import_is_reversible_byte_for_byte() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        let json_before = keys.to_json();
        let undo = plan.apply(&mut keys);
        assert_ne!(keys.to_json(), json_before, "the import must change something");
        assert_ne!(keys, before);
        undo.revert(&mut keys);
        assert_eq!(keys.to_json(), json_before, "undo must restore the file exactly");
        assert_eq!(keys, before);
    }

    /// MCC's crouch is a mouse button in this profile. It must import, fire as a mouse binding,
    /// and undo cleanly — see the note on [`ACTION_MAP`].
    #[test]
    fn a_mouse_button_imports_into_a_key_action_and_undoes() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        assert_eq!(before.bind(Action::PlayCrouch).primary, Binding::chord(Mods::CTRL));
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        let undo = plan.apply(&mut keys);
        assert_eq!(
            keys.bind(Action::PlayCrouch).primary,
            Binding::mouse(MouseBtn::Right),
            "MCC's crouch is the right mouse button"
        );
        assert_eq!(keys.bind(Action::PlayCrouch).primary.to_spec(), "Mouse2");
        assert_eq!(keys.bind(Action::PlayMelee).primary, Binding::mouse(MouseBtn::Middle));
        undo.revert(&mut keys);
        assert_eq!(keys, before);
    }

    #[test]
    fn an_import_keeps_an_actions_alternative_key() {
        // Fly-up ships as "E or R". MCC's LiftEditor is R, so the primary becomes R and HMS's own
        // alternative is left alone rather than cleared.
        let mut keys = Keybinds::default();
        let alt_before = keys.bind(Action::CamUp).alt;
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        plan.apply(&mut keys);
        assert_eq!(keys.bind(Action::CamUp).primary.to_spec(), "pos:R");
        assert_eq!(keys.bind(Action::CamUp).alt, alt_before);
    }

    #[test]
    fn a_second_import_is_still_undoable_to_the_pre_import_table() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        let first = plan.apply(&mut keys);
        let second = plan.apply(&mut keys); // idempotent apply
        second.revert(&mut keys);
        first.revert(&mut keys);
        assert_eq!(keys, before);
    }

    /// THE STRANDING CASE. The dangerous sequence is Apply, Apply, Undo: a second apply must not
    /// be able to replace the undo snapshot with the already-imported table, because then Undo
    /// would put the user back to the IMPORT rather than to where they started. The page makes it
    /// unreachable by dropping the plan on apply; this is that guarantee, through the same calls
    /// the button makes.
    #[test]
    fn pressing_apply_twice_cannot_strand_the_user() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        let mut st = ImportUi::default();
        st.plan = Some(plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan"));

        st.apply(&mut keys);
        assert!(st.plan.is_none(), "the stale preview must be gone after an apply");
        assert!(st.undo.is_some());
        assert!(st.applied.is_some(), "the page must say what it did");
        let imported = keys.clone();
        assert_ne!(imported, before);

        // A second apply has nothing to apply and MUST NOT touch the undo.
        st.apply(&mut keys);
        assert_eq!(keys, imported, "a second apply changed the bindings");

        assert!(st.undo_last(&mut keys), "undo must report that it did something");
        assert_eq!(keys, before, "undo must reach the PRE-IMPORT table, not the import");
        assert!(st.applied.is_none());
        assert!(!st.undo_last(&mut keys), "there is nothing left to undo");
        assert_eq!(keys, before);
    }

    // ---------- the non-binding settings ----------

    #[test]
    fn the_settings_are_read_and_brightness_is_report_only() {
        let ini = UeIni::parse(&sample_ini());
        let rows = read_settings(&ini, MccGame::HaloReach);
        let get = |label: &str| rows.iter().find(|r| r.label.starts_with(label)).cloned();

        let fov = get("Field of view").expect("FOV is read");
        assert_eq!(fov.value, "120°");
        let sens = get("Mouse sensitivity").expect("sensitivity is read");
        assert_eq!(sens.value, "7.20");

        let b = get("Brightness").expect("brightness is read");
        assert_eq!(b.value, "-0.90");
        assert_eq!(
            b.disposition,
            Disposition::ReportOnly,
            "brightness must never be applied: HMS's exposure is engine-calibrated"
        );
        assert!(b.note.contains("NOT imported"), "{}", b.note);

        // No setting is ever marked as applied, because this round applies none of them.
        for r in &rows {
            assert!(
                matches!(
                    r.disposition,
                    Disposition::ReportOnly | Disposition::NotYetWired | Disposition::Applied
                ),
                "{} claims to be applied",
                r.label
            );
        }
        assert!(get("Forge monitor camera").is_some());
        assert!(get("Theater camera").is_some());
        assert!(get("Volumes").is_some());
    }

    /// The field of view is in MCC's own units and HMS's slider uses the same ones, so the value
    /// imports with no conversion. Pinned so a future change to either range is noticed.
    #[test]
    fn mccs_fov_is_already_in_hmss_units() {
        let ini = UeIni::parse(&sample_ini());
        let fov = ini.get_f32("FOVSetting[5]").expect("a FOV");
        assert_eq!(fov, 120.0);
        assert!(
            (crate::film_play::FOV_MIN_DEG..=crate::film_play::FOV_MAX_DEG).contains(&fov),
            "MCC's FOV {fov} is outside HMS's {}..{}",
            crate::film_play::FOV_MIN_DEG,
            crate::film_play::FOV_MAX_DEG
        );
    }

    // ---------- the promise: nothing without the button ----------

    /// Building a plan READS, it never writes; and applying is the only thing that mutates the
    /// table. This is the promise most worth not regressing: the user's settings are their data.
    #[test]
    fn nothing_is_read_or_applied_without_the_button() {
        // A fresh page state has no plan, no undo and no error: constructing it does not look at
        // the disk.
        let st = ImportUi::default();
        assert!(st.plan.is_none());
        assert!(st.undo.is_none());
        assert!(st.error.is_none());
        assert!(st.game.is_none(), "not even the title is chosen until the page is drawn");

        // And building a plan does not touch the binding table.
        let before = Keybinds::default();
        let keys = before.clone();
        let plan = plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan");
        assert_eq!(keys, before, "building a plan must not change the bindings");
        assert!(plan.change_count() > 0, "this plan WOULD change something if applied");

        // Only `apply` changes anything.
        let mut k = before.clone();
        assert_eq!(k, before);
        plan.apply(&mut k);
        assert_ne!(k, before);
    }

    // ---------- the page itself, through a real egui pass ----------

    fn test_input(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(700.0, 1600.0),
            )),
            events,
            ..Default::default()
        }
    }

    /// One pass of the real page through a real `egui::Context`, returning every string it painted
    /// with the rectangle it painted into.
    fn pass(
        ctx: &egui::Context,
        st: &mut ImportUi,
        keys: &mut Keybinds,
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Rect)>, bool) {
        let mut changed = false;
        let out = ctx.run(test_input(events), |c| {
            egui::CentralPanel::default().show(c, |panel| {
                // `super::ui` explicitly: `ui` is also the closure's own argument name.
                changed = super::ui(st, keys, panel);
            });
        });
        let mut text = Vec::new();
        for cs in &out.shapes {
            if let egui::Shape::Text(t) = &cs.shape {
                text.push((
                    t.galley.text().to_string(),
                    egui::Rect::from_min_size(t.pos, t.galley.size()),
                ));
            }
        }
        (text, changed)
    }

    /// Press and release the primary pointer button at `at`, as a real pointer produces it: a
    /// move, then a press, then a release, each its own frame. Returns the strings painted by the
    /// frame AFTER the release, so the caller sees the page's new state.
    fn click(
        ctx: &egui::Context,
        st: &mut ImportUi,
        keys: &mut Keybinds,
        at: egui::Pos2,
    ) -> (Vec<(String, egui::Rect)>, bool) {
        let m = egui::Modifiers::default();
        let btn = egui::PointerButton::Primary;
        pass(ctx, st, keys, vec![egui::Event::PointerMoved(at)]);
        let (_, c1) = pass(
            ctx,
            st,
            keys,
            vec![egui::Event::PointerButton { pos: at, button: btn, pressed: true, modifiers: m }],
        );
        let (_, c2) = pass(
            ctx,
            st,
            keys,
            vec![egui::Event::PointerButton { pos: at, button: btn, pressed: false, modifiers: m }],
        );
        // One more frame with nothing happening, so the caller reads the page's settled state.
        let (text, c3) = pass(ctx, st, keys, Vec::new());
        (text, c1 || c2 || c3)
    }

    /// Draw the section once and return every string it painted.
    fn draw(st: &mut ImportUi, keys: &mut Keybinds) -> (Vec<String>, bool) {
        let ctx = egui::Context::default();
        let (text, changed) = pass(&ctx, st, keys, Vec::new());
        (text.into_iter().map(|(t, _)| t).collect(), changed)
    }

    /// THE PAGE, not the plumbing. `mccimport::ui` is what `main.rs` calls, so it is drawn here
    /// with a real egui context in every state it can be in: nothing found yet, a plan previewed,
    /// an error, and an applied import. A panic in the take-the-plan-out-and-put-it-back dance, or
    /// a preview that silently paints nothing, shows up here and nowhere else.
    #[test]
    fn the_page_draws_the_preview_in_every_state() {
        let mut keys = Keybinds::default();

        // (a) Before the button has ever been pressed: the section draws, and it must NOT have
        // gone looking for anything.
        let mut st = ImportUi::default();
        let (text, changed) = draw(&mut st, &mut keys);
        assert!(!changed, "merely drawing the page must not change the bindings");
        assert!(st.plan.is_none(), "drawing must not build a plan");
        let joined = text.join(" | ");
        assert!(joined.contains("IMPORT FROM"), "{joined}");
        assert!(joined.contains("Find MCC"), "the find button must be drawn: {joined}");
        assert!(
            joined.contains("never written"),
            "the page must promise MCC's files are only read: {joined}"
        );

        // (b) With a plan: the before/after table and the apply button must actually paint.
        st.plan = Some(plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan"));
        st.show_unmapped = true;
        let (text, changed) = draw(&mut st, &mut keys);
        assert!(!changed, "drawing a preview must not apply it");
        assert!(st.plan.is_some(), "the plan must survive being drawn");
        let joined = text.join(" | ");
        for want in [
            "Fly forward",          // a row label
            "MoveForward",          // MCC's own button name
            "Crouch",               // a test-mode row
            "Import these 7",       // the apply button, with the real count
            "Brightness",           // the settings report
            "NOT imported",         // and the reason it is not applied
            "Field of view",
            "no counterpart",       // the unmapped list
        ] {
            assert!(joined.contains(want), "the preview did not paint {want:?}: {joined}");
        }
        // The imported keys themselves are on screen BEFORE anything is applied.
        assert!(text.iter().any(|t| t == "Right mouse"), "MCC's crouch key is not shown");
        assert!(text.iter().any(|t| t == "R"), "MCC's fly-up key is not shown");

        // (c) Applied: the stale preview is gone and the undo is offered.
        st.apply(&mut keys);
        let (text, _) = draw(&mut st, &mut keys);
        let joined = text.join(" | ");
        assert!(joined.contains("Undo the import"), "{joined}");
        assert!(joined.contains("Imported 7"), "the page must say what it did: {joined}");
        assert!(!joined.contains("Import these"), "the stale preview is still drawn: {joined}");

        // (d) An error draws instead of a plan, and nothing panics.
        let mut st = ImportUi { error: Some("no MCC settings file was found".into()), ..Default::default() };
        let (text, _) = draw(&mut st, &mut keys);
        assert!(text.join(" | ").contains("no MCC settings file"), "{text:?}");
    }

    /// THE BUTTON. Everything else tests the functions behind the page; this presses the real
    /// widget in a real egui pass and checks that the bindings moved.
    ///
    /// The button is found by the text it painted ("Import these N controls") rather than by a
    /// hard-coded position, so a change to the layout does not silently turn the click into a
    /// click on nothing — if the button is not drawn, the lookup fails and the test says so.
    #[test]
    fn clicking_the_apply_button_imports_and_clicking_undo_puts_it_back() {
        let before = Keybinds::default();
        let mut keys = before.clone();
        let mut st = ImportUi::default();
        st.plan = Some(plan_from(&sample_ini(), MccGame::HaloReach, &keys).expect("a plan"));

        // ONE context for the whole interaction: egui's interaction state lives there, and a
        // click needs the pointer to have been seen before the press.
        let ctx = egui::Context::default();
        let (text, _) = pass(&ctx, &mut st, &mut keys, Vec::new());
        let find = |text: &[(String, egui::Rect)], want: &str| -> egui::Pos2 {
            text.iter()
                .find(|(t, _)| t.starts_with(want))
                .unwrap_or_else(|| {
                    panic!("no widget painted {want:?}; painted: {:?}", text.iter().map(|(t, _)| t).collect::<Vec<_>>())
                })
                .1
                .center()
        };
        let apply_at = find(&text, "Import these");
        assert_eq!(keys, before, "drawing alone must not change anything");

        let (text, changed) = click(&ctx, &mut st, &mut keys, apply_at);

        assert!(changed, "the click must report a change so main.rs saves the file");
        assert_ne!(keys, before, "clicking Apply did not change the bindings");
        assert_eq!(keys.bind(Action::CamUp).primary.to_spec(), "pos:R", "MCC's fly-up");
        assert_eq!(keys.bind(Action::PlayCrouch).primary, Binding::mouse(MouseBtn::Right));
        assert!(st.plan.is_none(), "the applied preview must be dropped");
        assert!(st.undo.is_some(), "the import must be undoable");

        // ...and the Undo button, the same way.
        let undo_at = find(&text, "Undo the import");
        let (_, changed) = click(&ctx, &mut st, &mut keys, undo_at);
        assert!(changed, "the undo click must report a change");
        assert_eq!(keys, before, "clicking Undo did not restore the controls");
    }

    // ---------- the real file on this machine ----------

    /// The production path, end to end, against the user's own `GameUserSettings.ini`: discovery
    /// finds it, the parser reads it, and the bindings it actually contains come out.
    ///
    /// Skips with a printed reason when there is no MCC install, so the suite passes anywhere.
    #[test]
    fn the_real_ini_on_this_machine_parses() {
        let _g = env_lock();
        let Some((found, ini)) = real_ini() else {
            eprintln!("SKIPPED: no MCC GameUserSettings.ini found on this machine");
            return;
        };
        // Enough of the file to be the real thing, not a stub.
        assert!(ini.len() > 100, "only {} entries in {:?}", ini.len(), found.ini.file_name());
        assert!(ini.get("QualityPreset").is_some(), "no QualityPreset — is this really MCC's ini?");

        let keys = Keybinds::default();
        let plan = Plan::from_ini(&ini, found, MccGame::HaloReach, &keys).expect("a plan");

        // All 66 of MCC's buttons are present in a real profile.
        assert_eq!(
            plan.profile.bindings.len(),
            ABSTRACT_BUTTONS.len(),
            "a real profile should carry every abstract button"
        );

        // THE ANCHORS, from the live file: the movement cluster and the editor commands.
        let find = |a: Action| plan.rows.iter().find(|r| r.action == a).unwrap().after();
        assert_eq!(find(Action::CamForward), "W");
        assert_eq!(find(Action::CamBack), "S");
        assert_eq!(find(Action::CamLeft), "A");
        assert_eq!(find(Action::CamRight), "D");
        assert_eq!(find(Action::Delete), "Del");
        // And the whole profile decodes without the importer losing its place.
        for b in &plan.profile.bindings {
            assert!(b.button < ABSTRACT_BUTTONS.len() as u32, "unknown button {}", b.button);
        }
        // Reach's FOV and brightness are readable.
        assert!(ini.get_indexed("FOVSetting", MccGame::HaloReach).is_some());
        assert!(ini.get_indexed("BrightnessPerGame", MccGame::HaloReach).is_some());

        // Print what the user would actually be shown, so `--nocapture` is the evidence of what
        // the production path produces from the real file rather than from a fixture.
        eprintln!("{}", plan.summary());
        for r in &plan.rows {
            eprintln!(
                "  {:<24} {:<14} -> {:<14} ({})",
                r.action.spec().label,
                r.before,
                r.after(),
                r.mcc_button
            );
        }
        for s in &plan.settings {
            eprintln!("  setting {:<52} {}", s.label, s.value);
        }
    }

    /// Discovery reaches the MCC settings tree through `film`'s one search implementation.
    #[test]
    fn discovery_finds_an_ini_or_says_it_found_none() {
        let _g = env_lock();
        let found = discover();
        for f in &found {
            assert!(f.ini.is_file());
            assert_eq!(f.ini.file_name().unwrap(), "GameUserSettings.ini");
            assert!(!f.label.is_empty(), "every find must be labelled by its install's role");
        }
        if found.is_empty() {
            eprintln!("SKIPPED: no MCC install on this machine");
        } else {
            eprintln!("{} MCC settings file(s) found", found.len());
        }
    }

    /// The override, which is also how a test can point discovery at a file.
    #[test]
    fn the_override_env_var_is_honoured() {
        let _g = env_lock();
        let dir = std::env::temp_dir().join(format!("hms_mccimport_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("GameUserSettings.ini");
        std::fs::write(&p, sample_ini()).unwrap();
        // Safety: a test-only env write; the suite runs these serially within this test.
        unsafe { std::env::set_var("HMS_MCC_SETTINGS", &p) };
        let found = discover();
        unsafe { std::env::remove_var("HMS_MCC_SETTINGS") };
        assert!(found.iter().any(|f| f.ini == p), "the override must be found");
    }
}

// ===================== #mcc-fov  THE FIELD-OF-VIEW IMPORT =====================
//
// The user asked for MCC's FOV to come across with the controls. It was read, formatted and shown,
// and then NOT applied -- the row said so honestly ("wiring it needs one change in the main
// window"), but an honest note about a missing feature is still a missing feature.
//
// The gates below cover the arithmetic AND the two places this can silently do nothing: a plan
// whose FOV never reaches the live value, and a host that never drains the channel.
#[cfg(test)]
mod fov_import_gates {
    use super::*;

    /// MCC's own `[/Script/MCC.MCCGameUserSettings]` section, slot 5 = Halo: Reach.
    fn ini_with_fov(v: &str) -> UeIni {
        UeIni::parse(&format!(
            "[/Script/MCC.MCCGameUserSettings]\nFOVSetting[5]={v}\nCustomKeyboardMouseMappingV2[5]=\n"
        ))
    }

    #[test]
    fn mccs_fov_is_read_for_the_right_title_and_in_hmss_own_units() {
        let ini = ini_with_fov("105");
        // Read for Reach (slot 5)...
        assert_eq!(ini.get_f32("FOVSetting[5]"), Some(105.0));
        // ...and NOT for another title, so a plan cannot import Halo 1's field of view.
        assert_eq!(ini.get_f32("FOVSetting[0]"), None);
        // 105 is inside HMS's range, so no conversion is involved -- this is the claim the
        // import rests on, asserted rather than assumed.
        assert!(
            (crate::film_play::FOV_MIN_DEG..=crate::film_play::FOV_MAX_DEG).contains(&105.0),
            "MCC's units must already be HMS's units"
        );
    }

    #[test]
    fn a_value_outside_hmss_range_is_refused_rather_than_clamped() {
        // A number outside 60..120 means the file is not saying what we think it is. Clamping
        // would import a value the user never chose, so the row is simply not offered.
        for bad in ["5", "0", "400", "-90"] {
            // Through the PRODUCTION decision, not a copy of it.
            let got = usable_fov(&ini_with_fov(bad), MccGame::HaloReach);
            assert_eq!(got, None, "FOVSetting[5]={bad} must be refused, not clamped");
            // ...and the raw value really is there, so the gate is rejecting a PRESENT value
            // rather than passing because the key was missing.
            assert!(
                ini_with_fov(bad).get_f32("FOVSetting[5]").is_some(),
                "the fixture must actually state {bad}, or this gate proves nothing"
            );
        }
        // The positive case, so "refuses everything" cannot pass this.
        assert_eq!(usable_fov(&ini_with_fov("95"), MccGame::HaloReach), Some(95.0));
    }

    #[test]
    fn applying_moves_the_live_value_and_undo_puts_it_back() {
        let mut st = ImportUi { fov: Some(78.0), ..Default::default() };
        let mut undo = Undo { before: Keybinds::default(), fov_before: None, summary: String::new() };
        // THE PRODUCTION method, not a restatement of it.
        st.adopt_fov(Some(110.0), &mut undo);
        assert_eq!(st.fov, Some(110.0), "the live FOV must become MCC's");
        assert_eq!(undo.fov_before, Some(78.0), "the previous value must be remembered");
        st.undo = Some(undo);
        assert!(st.undo_last(&mut Keybinds::default()), "undo must report that it acted");
        assert_eq!(st.fov, Some(78.0), "undo must restore exactly, not approximately");
    }

    #[test]
    fn a_plan_with_no_fov_leaves_the_live_value_alone_and_undo_asserts_nothing() {
        // The important negative: if MCC states no usable FOV, the import must not touch ours,
        // and Undo must not then "restore" a value it never replaced.
        let mut st = ImportUi { fov: Some(78.0), ..Default::default() };
        let mut undo = Undo { before: Keybinds::default(), fov_before: None, summary: String::new() };
        st.adopt_fov(None, &mut undo);
        assert_eq!(st.fov, Some(78.0), "an absent FOV must leave ours untouched");
        assert_eq!(undo.fov_before, None, "undo must not claim a value it never replaced");
        // And undoing must not then invent one.
        st.undo = Some(undo);
        st.undo_last(&mut Keybinds::default());
        assert_eq!(st.fov, Some(78.0), "undo must leave an untouched FOV alone");
    }

    #[test]
    fn the_fov_row_says_it_is_imported_rather_than_pending() {
        let ini = ini_with_fov("120");
        let rows = read_settings(&ini, MccGame::HaloReach);
        let row = rows
            .iter()
            .find(|r| r.label == "Field of view")
            .expect("the FOV row must be read");
        assert_eq!(row.value, "120\u{b0}");
        assert!(
            matches!(row.disposition, Disposition::Applied),
            "the FOV row must be Applied, not NotYetWired -- the note the user reads is the \
             promise the code has to keep"
        );
    }

    /// THE GATE FOR THE FAILURE MODE THIS PROJECT KEEPS HITTING: the capability exists and
    /// nothing invokes it.
    ///
    /// `ImportUi::fov` is an in/out channel. If the host never writes it before `ui` runs, the
    /// import has nothing to snapshot and Undo is broken; if the host never reads it back, the
    /// import silently does nothing. Neither shows up in any gate above, because they all drive
    /// this module directly. So the call site is gated structurally.
    #[test]
    fn the_host_both_fills_and_drains_the_fov_channel() {
        let src = include_str!("main.rs");
        let call = src
            .find("mccimport::ui(")
            .expect("main.rs must call mccimport::ui; if it moved, update this gate");
        // Look in a window around the call site rather than the whole file, so an unrelated
        // mention of `mcc_import.fov` elsewhere cannot satisfy this.
        let lo = src[..call].rfind("\n\n").unwrap_or(0);
        let hi = (call + 2000).min(src.len());
        let window = &src[lo..hi];
        assert!(
            window.contains("mcc_import.fov = Some("),
            "main.rs must write the live FOV into mcc_import.fov BEFORE calling mccimport::ui, \
             or the import has nothing to snapshot and Undo cannot restore"
        );
        assert!(
            window.contains("mcc_import.fov"),
            "main.rs must read mcc_import.fov back after mccimport::ui, or importing the FOV \
             changes nothing the user can see"
        );
        assert!(
            window.contains("save_fov_setting"),
            "the imported FOV must be persisted, or it is lost on the next restart"
        );
    }
}
