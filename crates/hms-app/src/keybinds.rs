//! Rebindable controls — GitHub issue #2 ("Mouse Movement Keys Are Not Customizable").
//!
//! One table owns every key the editor reads. A call site never tests a raw `egui::Key` any
//! more; it asks this module whether an [`Action`] fired. That is what makes the keys
//! rebindable at all, and it is also what keeps the Help ▸ Keyboard shortcuts window honest —
//! the window prints the same table the handlers read.
//!
//! # Logical keys vs physical positions
//!
//! `egui::Key` is a LOGICAL key: the character the active keyboard layout produces. A
//! hard-coded `Key::W` is therefore the wrong *place* on an AZERTY or Dvorak keyboard — the
//! physical key under the left middle finger emits `Z` on AZERTY and `,` on Dvorak, so the
//! classic WASD cluster becomes ZQSD / ,AOE and the fly keys land in three different corners.
//!
//! egui 0.31 does carry the position as well: `egui::Event::Key` has a `physical_key`, which
//! egui-winit fills in from winit's `KeyCode` — i.e. the egui key that key POSITION would have
//! produced on a US layout. `InputState::keys_down` / `key_pressed` only track the logical key,
//! so this module keeps its own [`PhysState`] of held/pressed *positions*, harvested from the
//! event stream once per frame.
//!
//! Each binding therefore records WHICH of the two it means:
//!
//! * **by position** (`"pos:W"`) — "the key where W sits on a US keyboard". The default for the
//!   six camera fly keys, because those are held like a gamepad stick: what matters is the
//!   shape of the cluster under your hand, not the letter printed on it. An AZERTY user gets a
//!   working ZQSD out of the box with no rebinding at all.
//! * **by letter** (`"Z"`) — "the key that types Z". The default for every command shortcut
//!   (Ctrl+Z, Ctrl+A, G, R, X/Y/Z …), because those are mnemonics: a user reading "Ctrl+Z
//!   undoes" wants the key that says Z.
//!
//! The mode is per binding and switchable in the UI, so neither default is a trap.
//!
//! # Contexts
//!
//! The same physical key means different things depending on what the editor is doing, and it
//! always has: Shift is the fly boost while the right mouse button is held, the precision
//! modifier during a transform, and the large-step modifier for a nudge. So every action
//! carries a [`Context`], and two bindings only CONFLICT when their contexts overlap. Held
//! modifiers (a binding with no key at all) are never counted as conflicts — sharing Shift
//! across three contexts is the design, not a mistake.
//!
//! # Defaults
//!
//! [`Keybinds::default()`] is exactly the set of keys the app has always used. A user with no
//! `keybinds.json` sees no change whatsoever; `defaults_match_the_shipped_keys` in the tests at
//! the bottom pins that promise key by key.

use eframe::egui;
use serde::{Deserialize, Serialize};

// ===================== actions =====================

/// Where an action is read. Two bindings on the same trigger only clash when their contexts
/// overlap (and [`Context::Global`] overlaps everything).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Context {
    /// Read while the camera is flying (the fly-look button held over the viewport).
    Camera,
    /// Read while a modal move / rotate / duplicate is live.
    Transform,
    /// Read in the viewport with no transform running and no text field focused.
    Editor,
    /// #play  Read only while TEST MODE is live — the player's own jump, crouch, sprint, walk
    /// and melee. Its own context because test mode reuses the camera's movement bindings (so an
    /// AZERTY or left-handed remap carries straight over to walking) while adding buttons of its
    /// own, and two bindings in different contexts are not a clash.
    Play,
    /// Read everywhere.
    Global,
}

impl Context {
    pub fn label(self) -> &'static str {
        match self {
            Context::Camera => "Camera (while the fly button is held)",
            Context::Transform => "Transform (while a move / rotate is live)",
            Context::Editor => "Selection & editing",
            Context::Play => "Test mode (playing on the map)",
            Context::Global => "Tools",
        }
    }
    /// Draw order of the groups in the Settings page.
    pub const ALL: [Context; 5] =
        [Context::Camera, Context::Transform, Context::Editor, Context::Play, Context::Global];
    fn overlaps(self, other: Context) -> bool {
        self == other || self == Context::Global || other == Context::Global
    }
}

/// How strict an action is about modifiers it does NOT ask for. This mirrors, action by action,
/// the hand-written guards the call sites used to carry, so the defaults behave exactly as before.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModRule {
    /// Extra modifiers are ignored (`W` fires on Shift+W — Shift is the fly boost).
    Any,
    /// Extra modifiers are ignored, but Ctrl/Cmd must be absent (so Ctrl+C never fires `C`).
    NoCtrl,
    /// Ctrl and Shift must match the binding exactly (so Ctrl+Shift+Z is redo, not undo).
    Exact,
}

/// Every rebindable action in the editor.
///
/// Adding one means adding a row to [`Action::ALL`] and a default to [`Action::spec`]; the test
/// `every_action_has_a_default` fails otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Action {
    // --- camera ---
    CamForward,
    CamBack,
    CamLeft,
    CamRight,
    CamUp,
    CamDown,
    FlyFaster,
    FlySlower,
    FlyLook,
    // --- modal transform ---
    ModalMove,
    ModalRotate,
    Duplicate,
    AxisX,
    AxisY,
    AxisZ,
    AxisSpace,
    ConfirmTransform,
    Magnet,
    PrecisionSlow,
    ArrayFine,
    // --- selection & editing ---
    SelectAll,
    SelectAdd,
    ContextMenu,
    FrameSelected,
    Delete,
    Undo,
    Redo,
    NudgeXPlus,
    NudgeXMinus,
    NudgeYPlus,
    NudgeYMinus,
    NudgeZPlus,
    NudgeZMinus,
    NudgeLarge,
    YawPlus,
    YawMinus,
    // --- test mode (#play) ---
    PlayJump,
    PlayCrouch,
    PlaySprint,
    PlayWalk,
    PlayMelee,
    PlayFire,
    /// #play-hud  Reload the weapon in hand. The engine also reloads on a dry
    /// magazine by itself, so this is the deliberate version of something that happens anyway.
    PlayReload,
    PlayView,
    /// #seat-entry  Enter the vehicle seat in range, or get out of the one you are in.
    PlayUseVehicle,
    // --- tools ---
    ConstructTool,
    Screenshot,
}

/// The static facts about an action: what it is called, where it is read, how fussy it is about
/// stray modifiers, and what it is bound to out of the box.
pub struct Spec {
    /// Stable key in `keybinds.json`. Never change one of these without a migration.
    pub id: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    pub context: Context,
    pub rule: ModRule,
    /// True for actions that are HELD (so the UI says "hold" and never offers an edge trigger).
    pub held: bool,
    /// True for the pointer-button action, which is picked from a list rather than captured
    /// (you cannot click a button to capture a click).
    pub mouse: bool,
}

impl Action {
    pub const ALL: [Action; 47] = [
        Action::CamForward,
        Action::CamBack,
        Action::CamLeft,
        Action::CamRight,
        Action::CamUp,
        Action::CamDown,
        Action::FlyFaster,
        Action::FlySlower,
        Action::FlyLook,
        Action::ModalMove,
        Action::ModalRotate,
        Action::Duplicate,
        Action::AxisX,
        Action::AxisY,
        Action::AxisZ,
        Action::AxisSpace,
        Action::ConfirmTransform,
        Action::Magnet,
        Action::PrecisionSlow,
        Action::ArrayFine,
        Action::SelectAll,
        Action::SelectAdd,
        Action::ContextMenu,
        Action::FrameSelected,
        Action::Delete,
        Action::Undo,
        Action::Redo,
        Action::NudgeXPlus,
        Action::NudgeXMinus,
        Action::NudgeYPlus,
        Action::NudgeYMinus,
        Action::NudgeZPlus,
        Action::NudgeZMinus,
        Action::NudgeLarge,
        Action::YawPlus,
        Action::YawMinus,
        Action::PlayJump,
        Action::PlayCrouch,
        Action::PlaySprint,
        Action::PlayWalk,
        Action::PlayMelee,
        Action::PlayFire,
        Action::PlayReload,
        Action::PlayView,
        Action::PlayUseVehicle,
        Action::ConstructTool,
        Action::Screenshot,
    ];

    pub fn spec(self) -> Spec {
        use Action as A;
        use Context as C;
        use ModRule as M;
        let (id, label, hint, context, rule, held, mouse) = match self {
            A::CamForward => ("cam_forward", "Fly forward", "Hold to fly the camera forward (fly button held).", C::Camera, M::Any, true, false),
            A::CamBack => ("cam_back", "Fly back", "Hold to fly the camera backwards (fly button held).", C::Camera, M::Any, true, false),
            A::CamLeft => ("cam_left", "Fly left", "Hold to fly the camera left (fly button held).", C::Camera, M::Any, true, false),
            A::CamRight => ("cam_right", "Fly right", "Hold to fly the camera right (fly button held).", C::Camera, M::Any, true, false),
            A::CamUp => ("cam_up", "Fly up", "Hold to fly the camera straight up (fly button held).", C::Camera, M::Any, true, false),
            A::CamDown => ("cam_down", "Fly down", "Hold to fly the camera straight down (fly button held).", C::Camera, M::Any, true, false),
            A::FlyFaster => ("fly_faster", "Fly faster (hold)", "Hold while flying for 4x the fly speed.", C::Camera, M::Any, true, false),
            A::FlySlower => ("fly_slower", "Fly slower (hold)", "Hold while flying for a quarter of the fly speed — for lining up a close shot.", C::Camera, M::Any, true, false),
            A::FlyLook => ("fly_look", "Fly / look around (hold)", "Hold this mouse button over the viewport to look around and enable the fly keys.", C::Camera, M::Any, true, true),

            A::ModalMove => ("modal_move", "Move the selection", "Starts a modal move: the selection follows the pointer until you click or confirm.", C::Transform, M::Any, false, false),
            A::ModalRotate => ("modal_rotate", "Rotate the selection", "Starts a modal rotate about the view axis (or the locked axis).", C::Transform, M::Any, false, false),
            A::Duplicate => ("duplicate", "Duplicate the selection", "Copies the selection and starts moving the copies.", C::Transform, M::Any, false, false),
            A::AxisX => ("axis_x", "Lock to X", "During a move / rotate: constrain to X (press again to unlock).", C::Transform, M::Any, false, false),
            A::AxisY => ("axis_y", "Lock to Y", "During a move / rotate: constrain to Y (press again to unlock).", C::Transform, M::Any, false, false),
            A::AxisZ => ("axis_z", "Lock to Z", "During a move / rotate: constrain to Z (press again to unlock).", C::Transform, M::Any, false, false),
            A::AxisSpace => ("axis_space", "World / local axes", "During a move / rotate: switch the axis lock between world and the object's own axes.", C::Transform, M::Any, false, false),
            A::ConfirmTransform => ("confirm_transform", "Confirm the transform", "Commits a modal move / rotate (a left click does the same).", C::Transform, M::Any, false, false),
            A::Magnet => ("magnet", "Magnet (hold)", "During a move: push the selection's nearest face flush against the nearest other surface.", C::Transform, M::Any, true, false),
            A::PrecisionSlow => ("precision_slow", "Precision (hold)", "During a move / rotate: slow, fine movement.", C::Transform, M::Any, true, false),
            A::ArrayFine => ("array_fine", "Fine array step (hold)", "While arraying copies: hold for a centimetre-sized array step on the wheel.", C::Transform, M::Any, true, false),

            A::SelectAll => ("select_all", "Select all objects", "Selects every variant / placed object in the scene.", C::Editor, M::Any, false, false),
            A::SelectAdd => ("select_add", "Add to the selection (hold)", "Hold while clicking or box-selecting to add to (or remove from) the selection.", C::Editor, M::Any, true, false),
            A::ContextMenu => ("context_menu", "Object context menu", "Opens the object menu at the cursor (needs a selection).", C::Editor, M::Exact, false, false),
            A::FrameSelected => ("frame_selected", "Frame the selection", "Flies the camera to the selected object.", C::Editor, M::Exact, false, false),
            A::Delete => ("delete", "Delete the selection", "Removes the selection from the scene (undoable).", C::Editor, M::Any, false, false),
            A::Undo => ("undo", "Undo", "Steps one edit back.", C::Editor, M::Exact, false, false),
            A::Redo => ("redo", "Redo", "Steps one undone edit forward.", C::Editor, M::Any, false, false),
            A::NudgeXPlus => ("nudge_x_plus", "Nudge +X", "Moves the selection 0.25 wu along +X (1 wu with the large-step modifier).", C::Editor, M::Any, false, false),
            A::NudgeXMinus => ("nudge_x_minus", "Nudge -X", "Moves the selection 0.25 wu along -X.", C::Editor, M::Any, false, false),
            A::NudgeYPlus => ("nudge_y_plus", "Nudge +Y", "Moves the selection 0.25 wu along +Y.", C::Editor, M::Any, false, false),
            A::NudgeYMinus => ("nudge_y_minus", "Nudge -Y", "Moves the selection 0.25 wu along -Y.", C::Editor, M::Any, false, false),
            A::NudgeZPlus => ("nudge_z_plus", "Nudge up (+Z)", "Moves the selection 0.25 wu straight up.", C::Editor, M::Any, false, false),
            A::NudgeZMinus => ("nudge_z_minus", "Nudge down (-Z)", "Moves the selection 0.25 wu straight down.", C::Editor, M::Any, false, false),
            A::NudgeLarge => ("nudge_large", "Large nudge / rotate step (hold)", "Hold while nudging for 1 wu instead of 0.25, and 45 deg instead of 15.", C::Editor, M::Any, true, false),
            A::YawPlus => ("yaw_plus", "Yaw the selection +", "Turns the selection 15 deg anticlockwise (45 with the large-step modifier).", C::Editor, M::Any, false, false),
            A::YawMinus => ("yaw_minus", "Yaw the selection -", "Turns the selection 15 deg clockwise.", C::Editor, M::Any, false, false),

            // #play  Test mode. The DEFAULTS ARE MCC'S OWN PC DEFAULTS (Space / Ctrl / Shift /
            // V), so a Reach player's hands are already in the right place; where the user's own
            // MCC profile is readable, `mccimport` overwrites these with what they actually play
            // with. Walking and strafing deliberately reuse the CAMERA bindings instead of
            // duplicating them, so one remap (or a left-handed preset) covers both.
            A::PlayJump => ("play_jump", "Jump", "Test mode: jump (the biped's own authored jump velocity).", C::Play, M::NoCtrl, false, false),
            A::PlayCrouch => ("play_crouch", "Crouch (hold)", "Test mode: hold to crouch — the eye drops to the biped's own crouching camera height.", C::Play, M::Any, true, false),
            A::PlaySprint => ("play_sprint", "Sprint (hold)", "Test mode: hold to sprint.", C::Play, M::Any, true, false),
            A::PlayWalk => ("play_walk", "Walk slowly (hold)", "Test mode: hold to move at the authored walking speed instead of the run.", C::Play, M::Any, true, false),
            A::PlayMelee => ("play_melee", "Melee", "Test mode: swing. Plays the weapon class's own melee animation.", C::Play, M::NoCtrl, false, false),
            A::PlayFire => ("play_fire", "Fire", "Test mode: fire the weapon in your hands along the crosshair.", C::Play, M::Any, false, true),
            A::PlayReload => ("play_reload", "Reload", "Test mode: reload the weapon in your hands. The magazine also reloads itself when it runs dry, which is what the engine does.", C::Play, M::NoCtrl, false, false),
            A::PlayView => ("play_view", "First / third person", "Test mode: switch between looking through the biped's own eyes and watching it from behind. Reach draws no legs in first person, so the full body is the third-person view.", C::Play, M::NoCtrl, false, false),
            A::PlayUseVehicle => ("play_use_vehicle", "Use: vehicle or weapon", "Test mode: get into the vehicle seat you are standing at, get out of the one you are in, or pick up the weapon you are standing next to. ONE action key for all three, which is the game's own arrangement; a vehicle wins when both are in range. The game's own prompt names this key.", C::Play, M::NoCtrl, false, false),

            A::ConstructTool => ("construct_tool", "Construct (CAD) tool", "Turns the construction-guide tool on and off.", C::Global, M::NoCtrl, false, false),
            A::Screenshot => ("screenshot", "Screenshot", "Writes a PNG of the viewport to the captures folder.", C::Global, M::Any, false, false),
        };
        Spec { id, label, hint, context, rule, held, mouse }
    }

    pub fn id(self) -> &'static str {
        self.spec().id
    }

    pub fn from_id(id: &str) -> Option<Action> {
        Action::ALL.into_iter().find(|a| a.id() == id)
    }
}

// ===================== bindings =====================

/// Modifiers a binding requires. `ctrl` means Ctrl on Windows/Linux and ⌘ on macOS (egui's
/// `command`), which is how every existing shortcut in the app already read it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Mods {
    pub const NONE: Mods = Mods { ctrl: false, shift: false, alt: false };
    pub const CTRL: Mods = Mods { ctrl: true, shift: false, alt: false };
    pub const SHIFT: Mods = Mods { ctrl: false, shift: true, alt: false };
    pub const ALT: Mods = Mods { ctrl: false, shift: false, alt: true };
    pub const CTRL_SHIFT: Mods = Mods { ctrl: true, shift: true, alt: false };
    fn any(self) -> bool {
        self.ctrl || self.shift || self.alt
    }
    fn from_egui(m: &egui::Modifiers) -> Mods {
        Mods { ctrl: m.ctrl || m.command || m.mac_cmd, shift: m.shift, alt: m.alt }
    }
}

/// Which pointer button a mouse binding means. Serialized as `Mouse1`/`Mouse2`/`Mouse3` so the
/// file never depends on egui's enum ordering.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum MouseBtn {
    Left,
    Right,
    Middle,
}

impl MouseBtn {
    pub const ALL: [MouseBtn; 3] = [MouseBtn::Right, MouseBtn::Left, MouseBtn::Middle];
    pub fn label(self) -> &'static str {
        match self {
            MouseBtn::Left => "Left mouse",
            MouseBtn::Right => "Right mouse",
            MouseBtn::Middle => "Middle mouse",
        }
    }
    fn tag(self) -> &'static str {
        match self {
            MouseBtn::Left => "Mouse1",
            MouseBtn::Right => "Mouse2",
            MouseBtn::Middle => "Mouse3",
        }
    }
    fn to_egui(self) -> egui::PointerButton {
        match self {
            MouseBtn::Left => egui::PointerButton::Primary,
            MouseBtn::Right => egui::PointerButton::Secondary,
            MouseBtn::Middle => egui::PointerButton::Middle,
        }
    }
}

/// What actually has to happen for a binding to fire.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Trigger {
    /// A keyboard key. `by_position` selects the physical position (see the module docs) over
    /// the logical key the layout produces.
    Key { key: egui::Key, by_position: bool },
    Mouse(MouseBtn),
    /// Modifiers only — a held chord such as Shift or Ctrl with no key.
    ModsOnly,
    /// Deliberately not bound. The action simply never fires.
    Unbound,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Binding {
    pub trigger: Trigger,
    pub mods: Mods,
}

impl Binding {
    pub const UNBOUND: Binding = Binding { trigger: Trigger::Unbound, mods: Mods::NONE };

    /// A logical-key binding ("the key that types this letter").
    pub const fn key(key: egui::Key) -> Binding {
        Binding { trigger: Trigger::Key { key, by_position: false }, mods: Mods::NONE }
    }
    /// A position binding ("the key where this letter sits on a US keyboard").
    pub const fn pos(key: egui::Key) -> Binding {
        Binding { trigger: Trigger::Key { key, by_position: true }, mods: Mods::NONE }
    }
    pub const fn mouse(b: MouseBtn) -> Binding {
        Binding { trigger: Trigger::Mouse(b), mods: Mods::NONE }
    }
    pub const fn chord(mods: Mods) -> Binding {
        Binding { trigger: Trigger::ModsOnly, mods }
    }
    pub const fn with(mut self, mods: Mods) -> Binding {
        self.mods = mods;
        self
    }
    pub fn is_bound(&self) -> bool {
        self.trigger != Trigger::Unbound
    }
    /// True when this binding is a held modifier chord rather than a key or button. Those are
    /// allowed to be shared, so the conflict check skips them.
    pub fn is_mods_only(&self) -> bool {
        self.trigger == Trigger::ModsOnly
    }

    /// The serialized form: `"Ctrl+Shift+Z"`, `"pos:W"`, `"Mouse2"`, `"Shift"`, `"none"`.
    pub fn to_spec(&self) -> String {
        let mut s = String::new();
        if self.mods.ctrl {
            s.push_str("Ctrl+");
        }
        if self.mods.shift {
            s.push_str("Shift+");
        }
        if self.mods.alt {
            s.push_str("Alt+");
        }
        match self.trigger {
            Trigger::Key { key, by_position } => {
                if by_position {
                    s.push_str("pos:");
                }
                s.push_str(key.name());
            }
            Trigger::Mouse(b) => s.push_str(b.tag()),
            Trigger::ModsOnly => {
                // "Shift+" -> "Shift"; a chord with no modifiers at all is not a binding.
                if s.ends_with('+') {
                    s.pop();
                } else {
                    return "none".into();
                }
            }
            Trigger::Unbound => return "none".into(),
        }
        s
    }

    /// Parse the serialized form. `None` for anything unrecognised, so a hand-edited or
    /// half-written file falls back to that one action's default instead of failing the load.
    pub fn parse(spec: &str) -> Option<Binding> {
        let spec = spec.trim();
        if spec.is_empty() {
            return None;
        }
        if spec.eq_ignore_ascii_case("none") || spec.eq_ignore_ascii_case("unbound") {
            return Some(Binding::UNBOUND);
        }
        let mut mods = Mods::default();
        let mut rest = spec;
        // Modifier prefixes, in any order and any casing.
        loop {
            let Some(plus) = rest.find('+') else { break };
            // `"Plus"` and `"+"` are key names; only treat the segment as a modifier when it IS one.
            let head = rest[..plus].trim();
            let m = match head.to_ascii_lowercase().as_str() {
                "ctrl" | "control" | "cmd" | "command" | "super" => Some(Mods::CTRL),
                "shift" => Some(Mods::SHIFT),
                "alt" | "option" | "opt" => Some(Mods::ALT),
                _ => None,
            };
            let Some(m) = m else { break };
            mods.ctrl |= m.ctrl;
            mods.shift |= m.shift;
            mods.alt |= m.alt;
            rest = rest[plus + 1..].trim_start();
        }
        if rest.is_empty() {
            // A pure modifier chord ("Shift+" or "Shift").
            return if mods.any() { Some(Binding::chord(mods)) } else { None };
        }
        // A bare modifier name with no '+' is a chord too.
        match rest.to_ascii_lowercase().as_str() {
            "ctrl" | "control" | "cmd" | "command" | "super" => {
                return Some(Binding::chord(Mods { ctrl: true, ..mods }))
            }
            "shift" => return Some(Binding::chord(Mods { shift: true, ..mods })),
            "alt" | "option" | "opt" => return Some(Binding::chord(Mods { alt: true, ..mods })),
            "mouse1" | "lmb" => return Some(Binding::mouse(MouseBtn::Left).with(mods)),
            "mouse2" | "rmb" => return Some(Binding::mouse(MouseBtn::Right).with(mods)),
            "mouse3" | "mmb" => return Some(Binding::mouse(MouseBtn::Middle).with(mods)),
            _ => {}
        }
        let (by_position, name) = match rest.split_once(':') {
            Some((p, n)) if p.eq_ignore_ascii_case("pos") || p.eq_ignore_ascii_case("position") => (true, n.trim()),
            Some((p, n)) if p.eq_ignore_ascii_case("key") || p.eq_ignore_ascii_case("logical") => (false, n.trim()),
            _ => (false, rest),
        };
        // egui's names are capitalised ("Enter", "PageUp", "W"); accept any casing the user types.
        let key = egui::Key::from_name(name)
            .or_else(|| egui::Key::from_name(&name.to_uppercase()))
            .or_else(|| {
                let mut c = name.to_ascii_lowercase();
                if let Some(f) = c.get_mut(0..1) {
                    f.make_ascii_uppercase();
                }
                egui::Key::from_name(&c)
            })?;
        Some(Binding { trigger: Trigger::Key { key, by_position }, mods })
    }

    /// What the user reads in the UI and in the shortcuts window.
    pub fn label(&self) -> String {
        let mut s = String::new();
        if self.mods.ctrl {
            s.push_str("Ctrl+");
        }
        if self.mods.shift {
            s.push_str("Shift+");
        }
        if self.mods.alt {
            s.push_str("Alt+");
        }
        match self.trigger {
            Trigger::Key { key, .. } => s.push_str(pretty_key(key)),
            Trigger::Mouse(b) => s.push_str(b.label()),
            Trigger::ModsOnly => {
                if s.ends_with('+') {
                    s.pop();
                }
            }
            Trigger::Unbound => return "—".into(),
        }
        s
    }
}

/// A friendlier name than `egui::Key::name()` for the handful of keys whose name is unreadable.
pub fn pretty_key(key: egui::Key) -> &'static str {
    use egui::Key as K;
    match key {
        K::ArrowUp => "Up arrow",
        K::ArrowDown => "Down arrow",
        K::ArrowLeft => "Left arrow",
        K::ArrowRight => "Right arrow",
        K::OpenBracket => "[",
        K::CloseBracket => "]",
        K::PageUp => "PgUp",
        K::PageDown => "PgDn",
        K::Minus => "-",
        K::Plus => "+",
        K::Equals => "=",
        K::Comma => ",",
        K::Period => ".",
        K::Semicolon => ";",
        K::Slash => "/",
        K::Backslash => "\\",
        K::Backtick => "`",
        K::Quote => "'",
        K::Delete => "Del",
        other => other.name(),
    }
}

/// A key that is NOT offered for rebinding, with the reason. Keeping these fixed is deliberate:
/// Escape is the universal "get me out of here" (it also cancels the rebind capture itself, so
/// it could not be captured anyway), and Tab is how a keyboard user moves between widgets.
pub fn reserved(key: egui::Key) -> Option<&'static str> {
    match key {
        egui::Key::Escape => Some("Escape always cancels — a transform, a half-drawn guide, a menu, or this capture. It stays fixed so there is always a way out."),
        egui::Key::Tab => Some("Tab moves between the controls in the panels; rebinding it would trap keyboard-only users."),
        _ => None,
    }
}

/// One action's bindings: a primary and an optional alternative (the app has always accepted
/// `E` *or* `R` for fly-up, and `Delete` *or* `Backspace`, so the table has to hold two).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bind {
    pub primary: Binding,
    pub alt: Option<Binding>,
}

impl Bind {
    const fn one(b: Binding) -> Bind {
        Bind { primary: b, alt: None }
    }
    const fn two(a: Binding, b: Binding) -> Bind {
        Bind { primary: a, alt: Some(b) }
    }
    pub fn get(&self, slot: usize) -> Binding {
        if slot == 0 {
            self.primary
        } else {
            self.alt.unwrap_or(Binding::UNBOUND)
        }
    }
    pub fn set(&mut self, slot: usize, b: Binding) {
        if slot == 0 {
            self.primary = b;
        } else {
            self.alt = b.is_bound().then_some(b);
        }
    }
    pub fn label(&self) -> String {
        match (self.primary.is_bound(), self.alt.filter(|a| a.is_bound())) {
            (true, Some(a)) => format!("{} or {}", self.primary.label(), a.label()),
            // A cleared primary with an alternative still in place reads as just the alternative,
            // not "— or X".
            (false, Some(a)) => a.label(),
            _ => self.primary.label(),
        }
    }
    fn each(&self) -> impl Iterator<Item = Binding> + '_ {
        [Some(self.primary), self.alt].into_iter().flatten().filter(|b| b.is_bound())
    }
}

// ===================== presets =====================

/// The ready-made sets. The point of the feature is accessibility, so switching to a
/// left-handed layout has to be one click, not six captures.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Preset {
    /// Exactly what the app has always used: the WASD cluster for the left hand, mouse right.
    #[default]
    RightHanded,
    /// Mouse in the LEFT hand: the camera moves on the arrow-key cluster under the right hand,
    /// and the nudges that used to live there move to the I/J/K/L block.
    LeftHandedArrows,
    /// Mouse in the LEFT hand, but the arrow keys kept for nudging: the camera moves on the
    /// I/J/K/L block, the right hand's equivalent of WASD.
    LeftHandedCluster,
    /// The user has changed an individual binding.
    Custom,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::RightHanded, Preset::LeftHandedArrows, Preset::LeftHandedCluster];
    pub fn label(self) -> &'static str {
        match self {
            Preset::RightHanded => "Right-handed (default)",
            Preset::LeftHandedArrows => "Left-handed — arrow keys",
            Preset::LeftHandedCluster => "Left-handed — I J K L",
            Preset::Custom => "Custom",
        }
    }
    pub fn hint(self) -> &'static str {
        match self {
            Preset::RightHanded => "Mouse in the right hand, W A S D + Q E under the left. This is what Halo Map Studio has always used.",
            Preset::LeftHandedArrows => "Mouse in the LEFT hand. The camera flies on the arrow keys with PgUp / PgDn for up and down; the nudge keys move to I J K L + U O so they stay under the same hand.",
            Preset::LeftHandedCluster => "Mouse in the LEFT hand. The camera flies on I J K L with U / O for up and down, which leaves the arrow keys free to keep nudging the selection.",
            Preset::Custom => "Your own bindings. Pick a preset above to start over from a known layout.",
        }
    }
}

// ===================== the table =====================

/// The binding table the whole editor reads.
#[derive(Clone, PartialEq, Debug)]
pub struct Keybinds {
    pub preset: Preset,
    binds: Vec<(Action, Bind)>,
    /// Problems found while loading a file (unknown action ids, unparsable specs). Shown once in
    /// the Settings page so a hand-edited file does not fail silently.
    pub load_notes: Vec<String>,
    /// Set while the rebinding page is waiting for a key. EVERY action reads as "not firing"
    /// then, so pressing G to bind it cannot also start a modal move, and the keys you are
    /// trying to assign do not fly the camera out of the map while you assign them.
    pub listening: bool,
}

impl Default for Keybinds {
    fn default() -> Self {
        Self::of(Preset::RightHanded)
    }
}

impl Keybinds {
    /// The shipped bindings, action by action. **This is the compatibility promise**: every row
    /// here is the key the hard-coded handler used before controls were rebindable.
    pub fn of(preset: Preset) -> Keybinds {
        use egui::Key as K;
        use Action as A;
        // The right-handed (shipped) set.
        let mut binds: Vec<(Action, Bind)> = vec![
            // Camera: by POSITION, so the cluster keeps its shape on AZERTY / Dvorak.
            (A::CamForward, Bind::one(Binding::pos(K::W))),
            (A::CamBack, Bind::one(Binding::pos(K::S))),
            (A::CamLeft, Bind::one(Binding::pos(K::A))),
            (A::CamRight, Bind::one(Binding::pos(K::D))),
            (A::CamUp, Bind::two(Binding::pos(K::E), Binding::pos(K::R))),
            (A::CamDown, Bind::two(Binding::pos(K::Q), Binding::pos(K::F))),
            (A::FlyFaster, Bind::one(Binding::chord(Mods::SHIFT))),
            // NEW with this feature (nothing used to hold Alt while flying), so no existing
            // behaviour changes: Alt is simply free during a fly.
            (A::FlySlower, Bind::one(Binding::chord(Mods::ALT))),
            (A::FlyLook, Bind::one(Binding::mouse(MouseBtn::Right))),
            // Transform: by LETTER — G is "grab", R is "rotate", X/Y/Z are the axes.
            (A::ModalMove, Bind::one(Binding::key(K::G))),
            (A::ModalRotate, Bind::one(Binding::key(K::R))),
            (A::Duplicate, Bind::one(Binding::key(K::D).with(Mods::SHIFT))),
            (A::AxisX, Bind::one(Binding::key(K::X))),
            (A::AxisY, Bind::one(Binding::key(K::Y))),
            (A::AxisZ, Bind::one(Binding::key(K::Z))),
            (A::AxisSpace, Bind::one(Binding::key(K::E))),
            (A::ConfirmTransform, Bind::one(Binding::key(K::Enter))),
            (A::Magnet, Bind::one(Binding::chord(Mods::CTRL))),
            (A::PrecisionSlow, Bind::one(Binding::chord(Mods::SHIFT))),
            (A::ArrayFine, Bind::one(Binding::chord(Mods::ALT))),
            // Selection & editing.
            (A::SelectAll, Bind::one(Binding::key(K::A).with(Mods::CTRL))),
            (A::SelectAdd, Bind::one(Binding::chord(Mods::SHIFT))),
            (A::ContextMenu, Bind::one(Binding::key(K::A).with(Mods::SHIFT))),
            (A::FrameSelected, Bind::one(Binding::key(K::F))),
            (A::Delete, Bind::two(Binding::key(K::Delete), Binding::key(K::Backspace))),
            (A::Undo, Bind::one(Binding::key(K::Z).with(Mods::CTRL))),
            (A::Redo, Bind::two(Binding::key(K::Y).with(Mods::CTRL), Binding::key(K::Z).with(Mods::CTRL_SHIFT))),
            (A::NudgeXPlus, Bind::one(Binding::key(K::ArrowRight))),
            (A::NudgeXMinus, Bind::one(Binding::key(K::ArrowLeft))),
            (A::NudgeYPlus, Bind::one(Binding::key(K::ArrowUp))),
            (A::NudgeYMinus, Bind::one(Binding::key(K::ArrowDown))),
            (A::NudgeZPlus, Bind::one(Binding::key(K::PageUp))),
            (A::NudgeZMinus, Bind::one(Binding::key(K::PageDown))),
            (A::NudgeLarge, Bind::one(Binding::chord(Mods::SHIFT))),
            (A::YawPlus, Bind::one(Binding::key(K::CloseBracket))),
            (A::YawMinus, Bind::one(Binding::key(K::OpenBracket))),
            // Test mode (#play) — MCC's own PC defaults.
            (A::PlayJump, Bind::one(Binding::key(K::Space))),
            (A::PlayCrouch, Bind::one(Binding::chord(Mods::CTRL))),
            (A::PlaySprint, Bind::one(Binding::chord(Mods::SHIFT))),
            (A::PlayWalk, Bind::one(Binding::chord(Mods::ALT))),
            (A::PlayMelee, Bind::one(Binding::key(K::V))),
            // #seat-entry  MCC's own `ActionReload` button is the one that enters a vehicle in
            // Reach -- it is what the prompt's `U+E446` glyph stands for -- and `E` is its
            // conventional PC key. The prompt prints whatever this is actually bound to.
            (A::PlayUseVehicle, Bind::one(Binding::key(K::E))),
            (A::PlayFire, Bind::one(Binding::mouse(MouseBtn::Left))),
            (A::PlayReload, Bind::one(Binding::key(egui::Key::R))),
            // #play  THIRD PERSON on a key, not only in a menu. Reported: "can't see my feet at
            // all" -- Reach draws no legs in first person, so the answer is the third-person view,
            // and a view you cannot find is a view you do not have.
            (A::PlayView, Bind::one(Binding::key(K::T))),
            // Tools.
            (A::ConstructTool, Bind::one(Binding::key(K::C))),
            (A::Screenshot, Bind::one(Binding::key(K::F12))),
        ];
        let set = |binds: &mut Vec<(Action, Bind)>, a: Action, b: Bind| {
            if let Some(slot) = binds.iter_mut().find(|(x, _)| *x == a) {
                slot.1 = b;
            }
        };
        match preset {
            Preset::RightHanded | Preset::Custom => {}
            Preset::LeftHandedArrows => {
                // Camera onto the arrow cluster...
                set(&mut binds, A::CamForward, Bind::one(Binding::key(K::ArrowUp)));
                set(&mut binds, A::CamBack, Bind::one(Binding::key(K::ArrowDown)));
                set(&mut binds, A::CamLeft, Bind::one(Binding::key(K::ArrowLeft)));
                set(&mut binds, A::CamRight, Bind::one(Binding::key(K::ArrowRight)));
                set(&mut binds, A::CamUp, Bind::two(Binding::key(K::PageUp), Binding::key(K::Home)));
                set(&mut binds, A::CamDown, Bind::two(Binding::key(K::PageDown), Binding::key(K::End)));
                // ...so the nudges move off it, onto the I/J/K/L block under the same hand.
                set(&mut binds, A::NudgeYPlus, Bind::one(Binding::pos(K::I)));
                set(&mut binds, A::NudgeYMinus, Bind::one(Binding::pos(K::K)));
                set(&mut binds, A::NudgeXMinus, Bind::one(Binding::pos(K::J)));
                set(&mut binds, A::NudgeXPlus, Bind::one(Binding::pos(K::L)));
                set(&mut binds, A::NudgeZPlus, Bind::one(Binding::pos(K::O)));
                set(&mut binds, A::NudgeZMinus, Bind::one(Binding::pos(K::U)));
            }
            Preset::LeftHandedCluster => {
                // The right hand's WASD: I forward, K back, J left, L right, U/O down/up.
                set(&mut binds, A::CamForward, Bind::one(Binding::pos(K::I)));
                set(&mut binds, A::CamBack, Bind::one(Binding::pos(K::K)));
                set(&mut binds, A::CamLeft, Bind::one(Binding::pos(K::J)));
                set(&mut binds, A::CamRight, Bind::one(Binding::pos(K::L)));
                set(&mut binds, A::CamUp, Bind::one(Binding::pos(K::O)));
                set(&mut binds, A::CamDown, Bind::one(Binding::pos(K::U)));
                // The arrow keys and PgUp/PgDn keep nudging the selection (unchanged).
            }
        }
        Keybinds { preset, binds, load_notes: Vec::new(), listening: false }
    }

    pub fn bind(&self, a: Action) -> Bind {
        self.binds
            .iter()
            .find(|(x, _)| *x == a)
            .map(|(_, b)| *b)
            // Unreachable while `every_action_has_a_default` passes; the default keeps it safe.
            .unwrap_or(Bind::one(Binding::UNBOUND))
    }

    pub fn set_bind(&mut self, a: Action, b: Bind) {
        if let Some(slot) = self.binds.iter_mut().find(|(x, _)| *x == a) {
            slot.1 = b;
        }
        self.preset = self.matched_preset();
    }

    /// Put one action back to its shipped default. (The page's per-row reset writes the binding
    /// directly because it already has it in hand; this is the same thing for scripts and tests.)
    #[allow(dead_code)]
    pub fn reset(&mut self, a: Action) {
        let d = Keybinds::of(Preset::RightHanded).bind(a);
        self.set_bind(a, d);
    }

    /// Which preset the current table IS (so the page can highlight it, and a hand-edit shows
    /// as Custom rather than lying about a preset).
    pub fn matched_preset(&self) -> Preset {
        for p in Preset::ALL {
            let cand = Keybinds::of(p);
            if Action::ALL.into_iter().all(|a| self.bind(a) == cand.bind(a)) {
                return p;
            }
        }
        Preset::Custom
    }

    /// Human label for the shortcuts window ("W A S D", "Ctrl+Z", …).
    pub fn label(&self, a: Action) -> String {
        self.bind(a).label()
    }

    /// The labels of several actions joined for one shortcuts-window row.
    pub fn labels(&self, actions: &[Action]) -> String {
        actions.iter().map(|a| self.bind(*a).label()).collect::<Vec<_>>().join(" ")
    }

    /// Every pair of bindings that fire on the same trigger in overlapping contexts. Held
    /// modifier chords are excluded: Shift legitimately means "fly faster", "precision" and
    /// "large nudge" at once, as it always has.
    pub fn conflicts(&self) -> Vec<Conflict> {
        let mut out = Vec::new();
        let mut flat: Vec<(Action, usize, Binding)> = Vec::new();
        for a in Action::ALL {
            let b = self.bind(a);
            for slot in 0..2 {
                let x = b.get(slot);
                if x.is_bound() && !x.is_mods_only() {
                    flat.push((a, slot, x));
                }
            }
        }
        for i in 0..flat.len() {
            for j in (i + 1)..flat.len() {
                let (a, sa, ba) = flat[i];
                let (b, sb, bb) = flat[j];
                if a == b {
                    continue; // an action's own two slots are alternatives, not a clash
                }
                if !a.spec().context.overlaps(b.spec().context) {
                    continue;
                }
                if same_trigger(ba, bb) && ba.mods == bb.mods {
                    out.push(Conflict { a, a_slot: sa, b, b_slot: sb, binding: ba });
                }
            }
        }
        out
    }

    /// Does this action's binding in `slot` clash with something? (The page tints its rows from
    /// one `conflicts()` call per frame rather than calling this per row; this is the same
    /// question for one binding, for scripts and tests.)
    #[allow(dead_code)]
    pub fn is_conflicted(&self, a: Action, slot: usize) -> bool {
        self.conflicts()
            .iter()
            .any(|c| (c.a == a && c.a_slot == slot) || (c.b == a && c.b_slot == slot))
    }

    // ---------- input ----------

    /// Is this action's key / button HELD right now?
    pub fn down(&self, a: Action, i: &egui::InputState, ph: &PhysState) -> bool {
        if self.listening {
            return false;
        }
        let rule = a.spec().rule;
        self.bind(a).each().any(|b| fires_down(&b, rule, i, ph))
    }

    /// Did this action's key / button go DOWN this frame? (Key auto-repeat counts, which is how
    /// holding an arrow key has always kept nudging.)
    pub fn pressed(&self, a: Action, i: &egui::InputState, ph: &PhysState) -> bool {
        if self.listening {
            return false;
        }
        let rule = a.spec().rule;
        self.bind(a).each().any(|b| fires_pressed(&b, rule, i, ph))
    }

    /// Did this action's pointer button go down this frame? (Only meaningful for a mouse binding.)
    pub fn mouse_pressed(&self, a: Action, i: &egui::InputState) -> bool {
        if self.listening {
            return false;
        }
        self.bind(a).each().any(|b| match b.trigger {
            Trigger::Mouse(m) => i.pointer.button_pressed(m.to_egui()),
            _ => false,
        })
    }

    /// Did this action's pointer button come up this frame?
    pub fn mouse_released(&self, a: Action, i: &egui::InputState) -> bool {
        if self.listening {
            return false;
        }
        self.bind(a).each().any(|b| match b.trigger {
            Trigger::Mouse(m) => i.pointer.button_released(m.to_egui()),
            _ => false,
        })
    }

    /// `egui::PointerButton` of a mouse-bound action, for the response helpers that need it.
    pub fn pointer_button(&self, a: Action) -> Option<egui::PointerButton> {
        self.bind(a).each().find_map(|b| match b.trigger {
            Trigger::Mouse(m) => Some(m.to_egui()),
            _ => None,
        })
    }

    // ---------- persistence ----------

    pub fn path() -> Option<std::path::PathBuf> {
        crate::project::settings_dir().map(|d| d.join("keybinds.json"))
    }

    /// Read the file, falling back to the shipped default for anything missing, unknown or
    /// unparsable. A corrupt file can therefore never leave the editor without controls, and it
    /// can never panic — the worst case is "the defaults, plus a note in the Settings page".
    pub fn load() -> Keybinds {
        let Some(p) = Self::path() else { return Keybinds::default() };
        let Ok(text) = std::fs::read_to_string(&p) else { return Keybinds::default() };
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Keybinds {
        let mut out = Keybinds::default();
        let file: KeybindsFile = match serde_json::from_str(text) {
            Ok(f) => f,
            Err(e) => {
                // Not even JSON (truncated mid-write, say): the defaults stand.
                out.load_notes.push(format!("keybinds.json could not be read ({e}); the default controls are in use."));
                return out;
            }
        };
        for (id, spec) in &file.bindings {
            let Some(a) = Action::from_id(id) else {
                out.load_notes.push(format!("keybinds.json: '{id}' is not an action in this version — ignored."));
                continue;
            };
            let specs = spec.as_slice();
            let mut parsed: Vec<Binding> = Vec::new();
            let mut bad = false;
            for s in &specs {
                match Binding::parse(s) {
                    Some(b) => parsed.push(b),
                    None => {
                        bad = true;
                        out.load_notes.push(format!(
                            "keybinds.json: '{s}' is not a key combination — '{}' kept its default.",
                            a.spec().label
                        ));
                    }
                }
            }
            if bad || parsed.is_empty() {
                continue; // this ONE action keeps its default; the rest of the file still applies
            }
            let bind = Bind { primary: parsed[0], alt: parsed.get(1).copied() };
            if let Some(slot) = out.binds.iter_mut().find(|(x, _)| *x == a) {
                slot.1 = bind;
            }
        }
        out.preset = out.matched_preset();
        out
    }

    pub fn to_json(&self) -> String {
        let mut bindings = std::collections::BTreeMap::new();
        for a in Action::ALL {
            let b = self.bind(a);
            let mut v = vec![b.primary.to_spec()];
            if let Some(alt) = b.alt.filter(|x| x.is_bound()) {
                v.push(alt.to_spec());
            }
            bindings.insert(a.id().to_string(), BindSpec::Many(v));
        }
        let file = KeybindsFile { version: 1, preset: self.matched_preset(), bindings };
        serde_json::to_string_pretty(&file).unwrap_or_else(|_| "{}".into())
    }

    pub fn save(&self) -> anyhow::Result<std::path::PathBuf> {
        let p = Self::path().ok_or_else(|| {
            anyhow::anyhow!("cannot locate a settings directory (no LOCALAPPDATA, XDG_DATA_HOME or HOME)")
        })?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&p, self.to_json())?;
        Ok(p)
    }
}

/// One clash between two bindings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Conflict {
    pub a: Action,
    pub a_slot: usize,
    pub b: Action,
    pub b_slot: usize,
    pub binding: Binding,
}

impl Conflict {
    pub fn describe(&self) -> String {
        format!(
            "{} is bound to both \"{}\" and \"{}\" — {}. Whichever handler runs first wins.",
            self.binding.label(),
            self.a.spec().label,
            self.b.spec().label,
            if self.a.spec().context == self.b.spec().context {
                self.a.spec().context.label().to_string()
            } else {
                format!("{} / {}", self.a.spec().context.label(), self.b.spec().context.label())
            }
        )
    }
}

/// Two triggers are "the same key" when they name the same key, whichever way each one matches
/// it. On a US layout the position and the letter ARE the same key, so reporting them as a clash
/// is the honest answer rather than a surprise for most users.
fn same_trigger(a: Binding, b: Binding) -> bool {
    match (a.trigger, b.trigger) {
        (Trigger::Key { key: x, .. }, Trigger::Key { key: y, .. }) => x == y,
        (Trigger::Mouse(x), Trigger::Mouse(y)) => x == y,
        _ => false,
    }
}

fn mods_ok(b: &Binding, rule: ModRule, m: &egui::Modifiers) -> bool {
    let held = Mods::from_egui(m);
    // Everything the binding asks for must be held.
    if (b.mods.ctrl && !held.ctrl) || (b.mods.shift && !held.shift) || (b.mods.alt && !held.alt) {
        return false;
    }
    match rule {
        ModRule::Any => true,
        ModRule::NoCtrl => b.mods.ctrl || !held.ctrl,
        // Ctrl and Shift must match exactly. Alt is a don't-care unless the binding wants it,
        // which is precisely how the hand-written guards read (`command && !shift`).
        ModRule::Exact => held.ctrl == b.mods.ctrl && held.shift == b.mods.shift,
    }
}

fn fires_down(b: &Binding, rule: ModRule, i: &egui::InputState, ph: &PhysState) -> bool {
    if !mods_ok(b, rule, &i.modifiers) {
        return false;
    }
    match b.trigger {
        Trigger::Key { key, by_position } => {
            if by_position {
                ph.down.contains(&key)
            } else {
                i.key_down(key)
            }
        }
        Trigger::Mouse(m) => i.pointer.button_down(m.to_egui()),
        Trigger::ModsOnly => true, // the modifier check above IS the trigger
        Trigger::Unbound => false,
    }
}

fn fires_pressed(b: &Binding, rule: ModRule, i: &egui::InputState, ph: &PhysState) -> bool {
    if !mods_ok(b, rule, &i.modifiers) {
        return false;
    }
    match b.trigger {
        Trigger::Key { key, by_position } => {
            if by_position {
                ph.pressed.contains(&key)
            } else {
                i.key_pressed(key)
            }
        }
        Trigger::Mouse(m) => i.pointer.button_pressed(m.to_egui()),
        Trigger::ModsOnly => true,
        Trigger::Unbound => false,
    }
}

// ===================== test mode's own read (#play) =====================

/// #play  The movement STICK for test mode, `(forward, right)` each in -1..1.
///
/// Deliberately the CAMERA's bindings. Walking and flying are the same gesture with the same
/// hand, so one remap covers both: a left-handed preset or an AZERTY keyboard moves the fly
/// cluster and test mode's feet follow it with nothing further to configure. Nothing here names
/// a key, for the same reason `fly_velocity` does not.
pub fn play_stick(keys: &Keybinds, i: &egui::InputState, ph: &PhysState) -> (f32, f32) {
    let mut fwd = 0.0f32;
    let mut right = 0.0f32;
    if keys.down(Action::CamForward, i, ph) {
        fwd += 1.0;
    }
    if keys.down(Action::CamBack, i, ph) {
        fwd -= 1.0;
    }
    if keys.down(Action::CamRight, i, ph) {
        right += 1.0;
    }
    if keys.down(Action::CamLeft, i, ph) {
        right -= 1.0;
    }
    (fwd, right)
}

/// #play  Test mode's buttons: `(jump, crouch, sprint, walk, melee)`.
///
/// Jump and melee are EDGES (a press), the other three are HELD — which is what the `held` flag
/// on each action's spec already says, so the UI and this read agree by construction.
pub fn play_buttons(keys: &Keybinds, i: &egui::InputState, ph: &PhysState) -> (bool, bool, bool, bool, bool) {
    (
        keys.pressed(Action::PlayJump, i, ph),
        keys.down(Action::PlayCrouch, i, ph),
        keys.down(Action::PlaySprint, i, ph),
        keys.down(Action::PlayWalk, i, ph),
        keys.pressed(Action::PlayMelee, i, ph),
    )
}

/// #play  Test mode's two remaining edges: `(fire, switch view)`.
///
/// Separate from [`play_buttons`] so that adding one does not change a tuple every call site has
/// to match — and `fire` is a MOUSE binding, which is read through `mouse_pressed` rather than
/// through the key path.
pub fn play_actions(keys: &Keybinds, i: &egui::InputState, ph: &PhysState) -> (bool, bool) {
    let fire = match keys.bind(Action::PlayFire).primary.trigger {
        Trigger::Mouse(_) => keys.mouse_pressed(Action::PlayFire, i),
        _ => keys.pressed(Action::PlayFire, i, ph),
    };
    (fire, keys.pressed(Action::PlayView, i, ph))
}

/// #play-hud  The reload action, EDGE (true on the frame it went down).
///
/// Its own function for the reason the two above give: adding a button here must not change a
/// tuple every call site has to match.
pub fn play_reload(keys: &Keybinds, i: &egui::InputState, ph: &PhysState) -> bool {
    keys.pressed(Action::PlayReload, i, ph)
}

/// #seat-entry  The enter/exit-vehicle action, EDGE (true on the frame it went down).
///
/// Separate from [`play_actions`] so adding it did not change that function's shape for every
/// existing caller — and because the mouse case has to be handled the same way `fire` does.
pub fn play_use_vehicle(keys: &Keybinds, i: &egui::InputState, ph: &PhysState) -> bool {
    match keys.bind(Action::PlayUseVehicle).primary.trigger {
        Trigger::Mouse(_) => keys.mouse_pressed(Action::PlayUseVehicle, i),
        _ => keys.pressed(Action::PlayUseVehicle, i, ph),
    }
}

/// #play  Mouse-look for test mode: the yaw and pitch AFTER this frame's motion, radians.
///
/// `sens` is a multiplier on the app's own look sensitivity so an imported MCC sensitivity can be
/// applied without restructuring anything — pass 1.0 for the app's current feel. The pitch clamp
/// is the camera's own (±1.5 rad), so looking up and down in test mode cannot go past where the
/// editor camera can.
pub fn play_look(yaw: f32, pitch: f32, dx: f32, dy: f32, sens: f32) -> (f32, f32) {
    const LOOK: f32 = 0.005;
    let k = LOOK * sens;
    (yaw - dx * k, (pitch - dy * k).clamp(-1.5, 1.5))
}

// ===================== the camera's fly step =====================

/// The camera's fly velocity for ONE frame, derived entirely from the bindings.
///
/// This is the whole of what the fly keys do, pulled out of `update_camera` so it can be tested
/// directly: give it an input state and it says where the camera would go. Nothing here names a
/// key, which is exactly the property issue #2 needed — rebind [`Action::CamForward`] and this
/// function follows, with no second place to update.
pub fn fly_velocity(
    keys: &Keybinds,
    i: &egui::InputState,
    ph: &PhysState,
    fwd: glam::Vec3,
    right: glam::Vec3,
    up: glam::Vec3,
    move_speed: f32,
    dt: f32,
) -> glam::Vec3 {
    let boost = if keys.down(Action::FlyFaster, i, ph) {
        4.0
    } else if keys.down(Action::FlySlower, i, ph) {
        0.25
    } else {
        1.0
    };
    let s = move_speed * boost * dt;
    let mut v = glam::Vec3::ZERO;
    if keys.down(Action::CamForward, i, ph) {
        v += fwd * s;
    }
    if keys.down(Action::CamBack, i, ph) {
        v -= fwd * s;
    }
    if keys.down(Action::CamRight, i, ph) {
        v += right * s;
    }
    if keys.down(Action::CamLeft, i, ph) {
        v -= right * s;
    }
    if keys.down(Action::CamUp, i, ph) {
        v += up * s;
    }
    if keys.down(Action::CamDown, i, ph) {
        v -= up * s;
    }
    v
}

// ===================== physical key state =====================

/// Held and just-pressed key POSITIONS, which `egui::InputState` does not track.
///
/// `InputState::keys_down` is keyed on the LOGICAL key, so it cannot answer "is the key where W
/// sits held?" on a non-US layout. `egui::Event::Key` does carry `physical_key`, so this walks
/// the event stream once per frame and keeps the position set itself. Cleared whenever the
/// window loses focus, because key-up events do not arrive while the app is in the background.
#[derive(Default, Clone, Debug)]
pub struct PhysState {
    down: std::collections::HashSet<egui::Key>,
    pressed: std::collections::HashSet<egui::Key>,
}

impl PhysState {
    /// Call once per frame, before any handler reads a binding.
    pub fn begin_frame(&mut self, i: &egui::InputState) {
        self.pressed.clear();
        if !i.focused {
            self.down.clear();
        }
        for ev in &i.events {
            match ev {
                egui::Event::Key { physical_key: Some(k), pressed, .. } => {
                    if *pressed {
                        // Auto-repeat counts as a press, matching `InputState::key_pressed`.
                        self.pressed.insert(*k);
                        self.down.insert(*k);
                    } else {
                        self.down.remove(k);
                    }
                }
                egui::Event::WindowFocused(false) => self.down.clear(),
                _ => {}
            }
        }
    }

    /// Used by the tests that pin the held/release/focus-loss behaviour.
    #[allow(dead_code)]
    pub fn is_down(&self, k: egui::Key) -> bool {
        self.down.contains(&k)
    }
}

// ===================== the file format =====================

/// `keybinds.json`. One string (or a list of two) per action id, so the file is readable and
/// hand-editable, and so a single bad entry costs exactly one action's binding.
#[derive(Serialize, Deserialize)]
struct KeybindsFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    preset: Preset,
    #[serde(default)]
    bindings: std::collections::BTreeMap<String, BindSpec>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum BindSpec {
    One(String),
    Many(Vec<String>),
}

impl BindSpec {
    fn as_slice(&self) -> Vec<String> {
        match self {
            BindSpec::One(s) => vec![s.clone()],
            BindSpec::Many(v) => v.clone(),
        }
    }
}

// ===================== capture state (the Settings page) =====================

/// "Press a key…" — which row is listening, and what it has seen so far.
#[derive(Clone, Debug)]
pub struct Capture {
    pub action: Action,
    pub slot: usize,
    /// Modifiers seen held during the capture, so releasing Shift alone binds the Shift chord.
    pub mods: Mods,
    /// Set when a reserved key was pressed, so the row can say why nothing happened.
    pub refused: Option<String>,
}

/// What a capture frame decided.
pub enum CaptureResult {
    /// Still listening.
    Waiting,
    /// Escape (or a click elsewhere): leave the binding alone.
    Cancelled,
    /// Bind this.
    Bound(Binding),
}

impl Capture {
    pub fn new(action: Action, slot: usize) -> Capture {
        Capture { action, slot, mods: Mods::default(), refused: None }
    }

    /// Feed one frame of input. Returns what to do.
    ///
    /// A non-modifier key commits immediately (with whatever modifiers are held). A pure
    /// modifier chord commits when the LAST modifier is released, which is the only unambiguous
    /// moment — until then the user may still be reaching for a letter.
    pub fn update(&mut self, i: &egui::InputState, ph: &PhysState, default_by_position: bool) -> CaptureResult {
        let held = Mods::from_egui(&i.modifiers);
        self.mods.ctrl |= held.ctrl;
        self.mods.shift |= held.shift;
        self.mods.alt |= held.alt;
        for ev in &i.events {
            let egui::Event::Key { key, physical_key, pressed, .. } = ev else { continue };
            if !*pressed {
                continue;
            }
            if *key == egui::Key::Escape {
                return CaptureResult::Cancelled;
            }
            if let Some(why) = reserved(*key) {
                self.refused = Some(why.to_string());
                continue;
            }
            // Which of the two keys the binding records. A position binding has to use the
            // PHYSICAL key from the event (the logical one is whatever the layout produced).
            let (k, by_position) = if default_by_position {
                match physical_key {
                    Some(p) => (*p, true),
                    // No position reported (a virtual / IME key): fall back to the letter so
                    // the capture still works instead of doing nothing.
                    None => (*key, false),
                }
            } else {
                (*key, false)
            };
            let mods = Mods::from_egui(&i.modifiers);
            return CaptureResult::Bound(Binding { trigger: Trigger::Key { key: k, by_position }, mods });
        }
        // All modifiers let go after at least one was held -> a modifier chord.
        if self.mods.any() && !held.any() {
            let _ = ph;
            return CaptureResult::Bound(Binding::chord(self.mods));
        }
        CaptureResult::Waiting
    }
}

// ===================== the Settings page =====================

/// Everything the page mutates, bundled so the call in `main.rs` stays one line.
pub struct UiState<'a> {
    pub keys: &'a mut Keybinds,
    pub capture: &'a mut Option<Capture>,
    /// The last thing the page did (a bind, a clash, a failed save), shown under the presets.
    pub msg: &'a mut Option<String>,
}

/// Draw Settings ▸ Controls & keyboard. Returns true when a binding changed (so the caller
/// persists the table).
pub fn ui(s: UiState, ph: &PhysState, ui: &mut egui::Ui) -> bool {
    let UiState { keys, capture, msg } = s;
    let mut changed = false;

    ui.label(
        egui::RichText::new(
            "Click a key to change it, then press the new key or combination. Escape cancels. \
             Right-click a key to clear it.",
        )
        .small(),
    );
    ui.add_space(4.0);

    // ---- presets ----
    ui.label(egui::RichText::new("PRESET").small().strong());
    let shown = keys.matched_preset();
    ui.horizontal_wrapped(|ui| {
        for p in Preset::ALL {
            if ui.selectable_label(shown == p, p.label()).on_hover_text(p.hint()).clicked() {
                let listening = keys.listening;
                *keys = Keybinds::of(p);
                keys.listening = listening;
                *capture = None;
                *msg = Some(format!("Applied the \"{}\" controls.", p.label()));
                changed = true;
            }
        }
        if ui
            .button("Restore all defaults")
            .on_hover_text("Puts every key back to the set Halo Map Studio ships with.")
            .clicked()
        {
            let listening = keys.listening;
            *keys = Keybinds::default();
            keys.listening = listening;
            *capture = None;
            *msg = Some("Every key is back to its default.".into());
            changed = true;
        }
    });
    ui.label(egui::RichText::new(shown.hint()).small().weak());

    // ---- what the page just did ----
    if let Some(m) = msg.as_deref() {
        ui.add_space(2.0);
        ui.label(egui::RichText::new(m).small());
    }

    // ---- problems with the file on disk ----
    if !keys.load_notes.is_empty() {
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new("The saved controls file had problems").strong().color(WARN));
            for n in &keys.load_notes {
                ui.label(egui::RichText::new(format!("• {n}")).small());
            }
            ui.label(egui::RichText::new("Anything not listed loaded normally.").small().weak());
        });
    }

    // ---- conflicts ----
    let conflicts = keys.conflicts();
    if !conflicts.is_empty() {
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{} key {} used twice",
                    conflicts.len(),
                    if conflicts.len() == 1 { "is" } else { "are" }
                ))
                .strong()
                .color(WARN),
            );
            for c in &conflicts {
                ui.label(egui::RichText::new(format!("• {}", c.describe())).small());
            }
            ui.label(
                egui::RichText::new(
                    "Held modifiers (Shift, Ctrl, Alt) are allowed to be shared and are not listed here.",
                )
                .small()
                .weak(),
            );
        });
    }

    ui.add_space(6.0);

    // Which (action, slot) pairs are in a clash — computed ONCE, not per row (the check walks
    // every binding pair, and there are 38 actions).
    let flagged: Vec<(Action, usize)> = conflicts
        .iter()
        .flat_map(|c| [(c.a, c.a_slot), (c.b, c.b_slot)])
        .collect();

    // ---- the rows, grouped by where they are read ----
    for ctxg in Context::ALL {
        ui.separator();
        ui.label(egui::RichText::new(ctxg.label().to_uppercase()).small().strong());
        for a in Action::ALL.into_iter().filter(|a| a.spec().context == ctxg) {
            changed |= action_row(a, keys, capture, msg, ph, &flagged, ui);
        }
    }

    ui.add_space(8.0);
    ui.separator();
    ui.label(egui::RichText::new("FIXED KEYS").small().strong());
    ui.label(
        egui::RichText::new(
            "Escape and Tab are not rebindable. Escape always cancels — a transform, a half-drawn \
             construction guide, an open menu, or a key capture on this page — so there is always a \
             way out; Tab moves between the controls in the panels, which keyboard-only users rely \
             on. A left click in the viewport always selects, a left drag always box-selects, and \
             Ctrl+wheel always changes the fly speed.",
        )
        .small()
        .weak(),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(format!(
            "Saved in {}",
            Keybinds::path().map(|p| p.display().to_string()).unwrap_or_else(|| "(nowhere — no settings directory)".into())
        ))
        .small()
        .weak(),
    );

    changed
}

const WARN: egui::Color32 = egui::Color32::from_rgb(235, 170, 70);

/// One action's row: the label, its primary binding, the position/letter switch, its alternative
/// and a reset.
fn action_row(
    a: Action,
    keys: &mut Keybinds,
    capture: &mut Option<Capture>,
    msg: &mut Option<String>,
    ph: &PhysState,
    flagged: &[(Action, usize)],
    ui: &mut egui::Ui,
) -> bool {
    let spec = a.spec();
    let mut changed = false;
    let default_bind = Keybinds::of(Preset::RightHanded).bind(a);
    let mut bind = keys.bind(a);

    ui.horizontal(|ui| {
        ui.add_sized([210.0, 18.0], egui::Label::new(spec.label).truncate())
            .on_hover_text(spec.hint);

        if spec.mouse {
            // You cannot click a button to capture a click, so the pointer button is a list.
            let cur = match bind.primary.trigger {
                Trigger::Mouse(m) => m,
                _ => MouseBtn::Right,
            };
            egui::ComboBox::from_id_salt(("hms_kb_mouse", spec.id))
                .selected_text(cur.label())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for m in MouseBtn::ALL {
                        if ui.selectable_label(cur == m, m.label()).clicked() && m != cur {
                            bind.primary = Binding::mouse(m);
                            changed = true;
                        }
                    }
                });
        } else {
            for slot in 0..2usize {
                let b = bind.get(slot);
                // The second slot only shows a button when it holds something, plus a small
                // "+" to add one.
                if slot == 1 && !b.is_bound() {
                    if ui
                        .add_sized([24.0, 20.0], egui::Button::new("+"))
                        .on_hover_text("Add a second key for this action (both will work).")
                        .clicked()
                    {
                        *capture = Some(Capture::new(a, 1));
                        *msg = None;
                    }
                    continue;
                }
                let listening = capture.as_ref().is_some_and(|c| c.action == a && c.slot == slot);
                let conflicted = flagged.contains(&(a, slot));
                let text = if listening {
                    egui::RichText::new("press a key…").italics().color(egui::Color32::from_rgb(120, 200, 255))
                } else if conflicted {
                    egui::RichText::new(b.label()).color(WARN).strong()
                } else {
                    egui::RichText::new(b.label())
                };
                let mut btn = ui.add_sized([150.0, 20.0], egui::Button::new(text).selected(listening));
                if conflicted {
                    btn = btn.on_hover_text("This key is used by another action in the same situation — see the list at the top of this page.");
                } else if spec.held {
                    btn = btn.on_hover_text("Held down while you fly / drag. Click, then hold the modifier you want and let go.");
                } else {
                    btn = btn.on_hover_text("Click, then press the key or combination you want. Right-click to clear it.");
                }
                if btn.clicked() {
                    *capture = Some(Capture::new(a, slot));
                    *msg = None;
                }
                if btn.secondary_clicked() {
                    bind.set(slot, Binding::UNBOUND);
                    *capture = None;
                    *msg = Some(format!("Cleared \"{}\" — it has no key now.", spec.label));
                    changed = true;
                }
                // Position vs letter, per binding. This is the non-QWERTY switch (see the
                // module docs): "pos" means the key's PLACE on the keyboard, "abc" the letter
                // the layout produces.
                if let Trigger::Key { key, by_position } = b.trigger {
                    let tag = if by_position { "pos" } else { "abc" };
                    if ui
                        .add_sized([32.0, 20.0], egui::Button::new(egui::RichText::new(tag).small()))
                        .on_hover_text(if by_position {
                            "Matched by POSITION: the key where this letter sits on a US keyboard, whatever your layout prints on it. Click for \"by letter\" instead."
                        } else {
                            "Matched by LETTER: the key that types this character on your layout. Click for \"by position\" instead."
                        })
                        .clicked()
                    {
                        bind.set(slot, Binding { trigger: Trigger::Key { key, by_position: !by_position }, mods: b.mods });
                        changed = true;
                    }
                } else {
                    ui.add_space(36.0);
                }
            }
        }

        let is_default = bind == default_bind;
        if ui
            .add_enabled(!is_default, egui::Button::new("↺"))
            .on_hover_text(format!("Back to the default ({})", default_bind.label()))
            .clicked()
        {
            bind = default_bind;
            *capture = None;
            *msg = Some(format!("\"{}\" is back to {}.", spec.label, default_bind.label()));
            changed = true;
        }
    });

    // ---- drive an active capture for this row ----
    let active = capture.as_ref().filter(|c| c.action == a).cloned();
    if let Some(mut cap) = active {
        ui.ctx().request_repaint(); // keep frames coming while we listen
        let slot = cap.slot;
        // Camera keys capture by POSITION by default (that is what makes them work on AZERTY);
        // command shortcuts capture the letter.
        let by_position = matches!(default_bind.primary.trigger, Trigger::Key { by_position: true, .. });
        let result = ui.input(|i| cap.update(i, ph, by_position));
        if let Some(why) = cap.refused.clone() {
            ui.label(egui::RichText::new(format!("  {why}")).small().color(WARN));
        }
        *capture = Some(cap);
        match result {
            CaptureResult::Waiting => {
                ui.label(
                    egui::RichText::new("  Press the key or combination for this action — Escape to cancel.")
                        .small()
                        .weak(),
                );
            }
            CaptureResult::Cancelled => {
                *capture = None;
                *msg = Some(format!("Left \"{}\" as {}.", spec.label, bind.get(slot).label()));
            }
            CaptureResult::Bound(b) => {
                bind.set(slot, b);
                *capture = None;
                // Report a clash loudly instead of quietly stealing the key.
                let mut probe = keys.clone();
                probe.set_bind(a, bind);
                let clash: Vec<String> = probe
                    .conflicts()
                    .iter()
                    .filter(|c| c.a == a || c.b == a)
                    .map(|c| {
                        let other = if c.a == a { c.b } else { c.a };
                        format!("\"{}\"", other.spec().label)
                    })
                    .collect();
                *msg = Some(if clash.is_empty() {
                    format!("\"{}\" is now {}.", spec.label, b.label())
                } else {
                    format!(
                        "\"{}\" is now {} — but that key is ALREADY used by {}. Change one of them, or they will fight.",
                        spec.label,
                        b.label(),
                        clash.join(", ")
                    )
                });
                changed = true;
            }
        }
    }

    if changed {
        keys.set_bind(a, bind);
    }
    changed
}

// ===================== tests =====================

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Key as K;

    /// Build a real `InputState` from raw events so the resolver can be tested the way the app
    /// actually calls it.
    fn input(events: Vec<egui::Event>, modifiers: egui::Modifiers) -> egui::InputState {
        let raw = egui::RawInput { events, modifiers, focused: true, ..Default::default() };
        egui::InputState::default().begin_pass(raw, false, 1.0, &egui::Options::default())
    }

    fn key_ev(key: K, physical: Option<K>, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key { key, physical_key: physical, pressed, repeat: false, modifiers }
    }

    /// THE compatibility promise: with no settings file, every binding is the key the
    /// hard-coded handler used. Each line here was read off the handler it replaced.
    #[test]
    fn defaults_match_the_shipped_keys() {
        let k = Keybinds::default();
        assert_eq!(k.preset, Preset::RightHanded);
        // update_camera: W/S/D/A fly, E-or-R up, Q-or-F down, Shift = 4x, RMB = fly.
        assert_eq!(k.bind(Action::CamForward), Bind::one(Binding::pos(K::W)));
        assert_eq!(k.bind(Action::CamBack), Bind::one(Binding::pos(K::S)));
        assert_eq!(k.bind(Action::CamLeft), Bind::one(Binding::pos(K::A)));
        assert_eq!(k.bind(Action::CamRight), Bind::one(Binding::pos(K::D)));
        assert_eq!(k.bind(Action::CamUp), Bind::two(Binding::pos(K::E), Binding::pos(K::R)));
        assert_eq!(k.bind(Action::CamDown), Bind::two(Binding::pos(K::Q), Binding::pos(K::F)));
        assert_eq!(k.bind(Action::FlyFaster), Bind::one(Binding::chord(Mods::SHIFT)));
        assert_eq!(k.bind(Action::FlyLook), Bind::one(Binding::mouse(MouseBtn::Right)));
        // update_object_transform: Shift+D duplicate, G grab, R rotate, E axis space, X/Y/Z locks.
        assert_eq!(k.bind(Action::Duplicate), Bind::one(Binding::key(K::D).with(Mods::SHIFT)));
        assert_eq!(k.bind(Action::ModalMove), Bind::one(Binding::key(K::G)));
        assert_eq!(k.bind(Action::ModalRotate), Bind::one(Binding::key(K::R)));
        assert_eq!(k.bind(Action::AxisSpace), Bind::one(Binding::key(K::E)));
        assert_eq!(k.bind(Action::AxisX), Bind::one(Binding::key(K::X)));
        assert_eq!(k.bind(Action::AxisY), Bind::one(Binding::key(K::Y)));
        assert_eq!(k.bind(Action::AxisZ), Bind::one(Binding::key(K::Z)));
        assert_eq!(k.bind(Action::ConfirmTransform), Bind::one(Binding::key(K::Enter)));
        assert_eq!(k.bind(Action::Magnet), Bind::one(Binding::chord(Mods::CTRL)));
        // handle_editor_shortcuts / handle_delete_key / handle_undo_redo / handle_selected_edit.
        assert_eq!(k.bind(Action::SelectAll), Bind::one(Binding::key(K::A).with(Mods::CTRL)));
        assert_eq!(k.bind(Action::ContextMenu), Bind::one(Binding::key(K::A).with(Mods::SHIFT)));
        assert_eq!(k.bind(Action::FrameSelected), Bind::one(Binding::key(K::F)));
        assert_eq!(k.bind(Action::Delete), Bind::two(Binding::key(K::Delete), Binding::key(K::Backspace)));
        assert_eq!(k.bind(Action::Undo), Bind::one(Binding::key(K::Z).with(Mods::CTRL)));
        assert_eq!(
            k.bind(Action::Redo),
            Bind::two(Binding::key(K::Y).with(Mods::CTRL), Binding::key(K::Z).with(Mods::CTRL_SHIFT))
        );
        assert_eq!(k.bind(Action::NudgeXPlus), Bind::one(Binding::key(K::ArrowRight)));
        assert_eq!(k.bind(Action::NudgeXMinus), Bind::one(Binding::key(K::ArrowLeft)));
        assert_eq!(k.bind(Action::NudgeYPlus), Bind::one(Binding::key(K::ArrowUp)));
        assert_eq!(k.bind(Action::NudgeYMinus), Bind::one(Binding::key(K::ArrowDown)));
        assert_eq!(k.bind(Action::NudgeZPlus), Bind::one(Binding::key(K::PageUp)));
        assert_eq!(k.bind(Action::NudgeZMinus), Bind::one(Binding::key(K::PageDown)));
        assert_eq!(k.bind(Action::YawPlus), Bind::one(Binding::key(K::CloseBracket)));
        assert_eq!(k.bind(Action::YawMinus), Bind::one(Binding::key(K::OpenBracket)));
        // the C / F12 checks in update().
        assert_eq!(k.bind(Action::ConstructTool), Bind::one(Binding::key(K::C)));
        assert_eq!(k.bind(Action::Screenshot), Bind::one(Binding::key(K::F12)));
        // ...and an absent / unreadable file is the same thing.
        assert_eq!(Keybinds::from_json("").binds, k.binds);
        assert_eq!(Keybinds::from_json("{}").binds, k.binds);
    }

    #[test]
    fn every_action_has_a_default() {
        let k = Keybinds::default();
        for a in Action::ALL {
            let b = k.bind(a);
            assert!(b.primary.is_bound(), "{:?} ({}) has no default binding", a, a.spec().label);
        }
        // ...and the ids are unique and stable-looking (snake_case, no spaces).
        let mut ids: Vec<&str> = Action::ALL.iter().map(|a| a.id()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "two actions share an id");
        for a in Action::ALL {
            assert!(
                a.id().chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
                "{:?} has a non-snake_case id '{}'",
                a,
                a.id()
            );
            assert_eq!(Action::from_id(a.id()), Some(a));
        }
    }

    /// Every preset must be internally consistent — no clashes, nothing left unbound.
    #[test]
    fn presets_are_complete_and_conflict_free() {
        for p in Preset::ALL {
            let k = Keybinds::of(p);
            assert_eq!(k.matched_preset(), p, "{:?} does not round-trip through matched_preset", p);
            for a in Action::ALL {
                assert!(k.bind(a).primary.is_bound(), "{:?}: {:?} is unbound", p, a);
            }
            let c = k.conflicts();
            assert!(c.is_empty(), "{:?} ships with conflicts: {:?}", p, c.iter().map(|x| x.describe()).collect::<Vec<_>>());
        }
    }

    /// The left-handed presets must actually move the camera off the WASD cluster (the whole
    /// point of issue #2) and leave the rest of the editor alone.
    #[test]
    fn left_handed_presets_move_the_camera_keys() {
        let arrows = Keybinds::of(Preset::LeftHandedArrows);
        assert_eq!(arrows.bind(Action::CamForward).primary, Binding::key(K::ArrowUp));
        assert_eq!(arrows.bind(Action::CamLeft).primary, Binding::key(K::ArrowLeft));
        assert_eq!(arrows.bind(Action::CamUp).primary, Binding::key(K::PageUp));
        // the nudges had to vacate the arrow cluster
        assert_eq!(arrows.bind(Action::NudgeYPlus).primary, Binding::pos(K::I));
        assert_eq!(arrows.bind(Action::NudgeZPlus).primary, Binding::pos(K::O));
        // and the verbs are untouched
        assert_eq!(arrows.bind(Action::Undo), Keybinds::default().bind(Action::Undo));
        assert_eq!(arrows.bind(Action::ModalMove), Keybinds::default().bind(Action::ModalMove));

        let cluster = Keybinds::of(Preset::LeftHandedCluster);
        assert_eq!(cluster.bind(Action::CamForward).primary, Binding::pos(K::I));
        assert_eq!(cluster.bind(Action::CamBack).primary, Binding::pos(K::K));
        // the arrow keys stay on the nudges here
        assert_eq!(cluster.bind(Action::NudgeXPlus), Keybinds::default().bind(Action::NudgeXPlus));
    }

    #[test]
    fn preset_application_marks_custom_after_an_edit() {
        let mut k = Keybinds::of(Preset::LeftHandedArrows);
        assert_eq!(k.matched_preset(), Preset::LeftHandedArrows);
        k.set_bind(Action::CamForward, Bind::one(Binding::pos(K::T)));
        assert_eq!(k.preset, Preset::Custom);
        k.reset(Action::CamForward);
        // back to the shipped W, so it is no longer the left-handed set either
        assert_eq!(k.bind(Action::CamForward).primary, Binding::pos(K::W));
    }

    #[test]
    fn binding_specs_round_trip() {
        let cases = [
            Binding::pos(K::W),
            Binding::key(K::G),
            Binding::key(K::D).with(Mods::SHIFT),
            Binding::key(K::Z).with(Mods::CTRL_SHIFT),
            Binding::key(K::ArrowRight),
            Binding::key(K::CloseBracket),
            Binding::key(K::F12),
            Binding::mouse(MouseBtn::Right),
            Binding::chord(Mods::SHIFT),
            Binding::chord(Mods::ALT),
            Binding::UNBOUND,
        ];
        for b in cases {
            let s = b.to_spec();
            assert_eq!(Binding::parse(&s), Some(b), "'{s}' did not round-trip");
        }
        // the exact strings a user would hand-write
        assert_eq!(Binding::parse("pos:W"), Some(Binding::pos(K::W)));
        assert_eq!(Binding::parse("Shift+D"), Some(Binding::key(K::D).with(Mods::SHIFT)));
        assert_eq!(Binding::parse("ctrl+shift+z"), Some(Binding::key(K::Z).with(Mods::CTRL_SHIFT)));
        assert_eq!(Binding::parse("Mouse2"), Some(Binding::mouse(MouseBtn::Right)));
        assert_eq!(Binding::parse("rmb"), Some(Binding::mouse(MouseBtn::Right)));
        assert_eq!(Binding::parse("Shift"), Some(Binding::chord(Mods::SHIFT)));
        assert_eq!(Binding::parse("Up"), Some(Binding::key(K::ArrowUp)));
        assert_eq!(Binding::parse("ArrowUp"), Some(Binding::key(K::ArrowUp)));
        assert_eq!(Binding::parse("none"), Some(Binding::UNBOUND));
        // ...and nonsense is None, not a panic and not a wrong key
        assert_eq!(Binding::parse("Sideways"), None);
        assert_eq!(Binding::parse(""), None);
        assert_eq!(Binding::parse("Ctrl+"), Some(Binding::chord(Mods::CTRL)));
        assert_eq!(Binding::parse("pos:"), None);
    }

    #[test]
    fn settings_round_trip_through_json() {
        let mut k = Keybinds::default();
        k.set_bind(Action::CamForward, Bind::one(Binding::key(K::ArrowUp)));
        k.set_bind(Action::CamBack, Bind::two(Binding::key(K::ArrowDown), Binding::pos(K::S)));
        k.set_bind(Action::Screenshot, Bind::one(Binding::UNBOUND));
        let json = k.to_json();
        let back = Keybinds::from_json(&json);
        for a in Action::ALL {
            assert_eq!(back.bind(a), k.bind(a), "{:?} did not round-trip", a);
        }
        assert!(back.load_notes.is_empty(), "a clean file produced notes: {:?}", back.load_notes);
        // the file is human-readable, with the specs the docs describe
        assert!(json.contains("\"cam_forward\""), "{json}");
        assert!(json.contains("\"Up\""), "{json}");
    }

    /// A file written by a newer version, hand-edited, truncated or simply wrong must leave the
    /// user with a working editor.
    #[test]
    fn corrupt_and_partial_files_fall_back_to_defaults() {
        let d = Keybinds::default();

        // not JSON at all
        let k = Keybinds::from_json("{ this is not json");
        assert_eq!(k.binds, d.binds);
        assert!(!k.load_notes.is_empty());

        // a partial file: one action set, everything else defaulted
        let k = Keybinds::from_json(r#"{"bindings":{"cam_forward":"Up"}}"#);
        assert_eq!(k.bind(Action::CamForward).primary, Binding::key(K::ArrowUp));
        assert_eq!(k.bind(Action::CamBack), d.bind(Action::CamBack));
        assert!(k.load_notes.is_empty());

        // an unknown action and an unparsable spec: both noted, both harmless
        let k = Keybinds::from_json(
            r#"{"bindings":{"cam_forward":"Sideways","fly_jetpack":"J","undo":"Ctrl+Z"}}"#,
        );
        assert_eq!(k.bind(Action::CamForward), d.bind(Action::CamForward), "a bad spec must keep the default");
        assert_eq!(k.bind(Action::Undo), d.bind(Action::Undo));
        assert_eq!(k.load_notes.len(), 2, "{:?}", k.load_notes);

        // wrong TYPES where the schema expects strings
        let k = Keybinds::from_json(r#"{"bindings":{"cam_forward":17}}"#);
        assert_eq!(k.binds, d.binds);
        assert!(!k.load_notes.is_empty());

        // an empty list for an action
        let k = Keybinds::from_json(r#"{"bindings":{"cam_forward":[]}}"#);
        assert_eq!(k.bind(Action::CamForward), d.bind(Action::CamForward));

        // every action deliberately unbound: legal, and the editor must still load
        let all_none: String = format!(
            r#"{{"bindings":{{{}}}}}"#,
            Action::ALL.iter().map(|a| format!("\"{}\":\"none\"", a.id())).collect::<Vec<_>>().join(",")
        );
        let k = Keybinds::from_json(&all_none);
        assert!(!k.bind(Action::CamForward).primary.is_bound());
        assert!(k.conflicts().is_empty());
    }

    #[test]
    fn conflicts_are_detected_and_described() {
        let mut k = Keybinds::default();
        assert!(k.conflicts().is_empty(), "the shipped set must be clean");

        // Steal undo's key for "select all" -> same context (Editor), same mods.
        k.set_bind(Action::SelectAll, Bind::one(Binding::key(K::Z).with(Mods::CTRL)));
        let c = k.conflicts();
        assert_eq!(c.len(), 1, "{:?}", c);
        assert!(k.is_conflicted(Action::SelectAll, 0));
        assert!(k.is_conflicted(Action::Undo, 0));
        let text = c[0].describe();
        assert!(text.contains("Ctrl+Z") && text.contains("Undo") && text.contains("Select all objects"), "{text}");

        // DIFFERENT contexts do not clash: the shipped R is both "fly up" and "rotate".
        let mut k = Keybinds::default();
        k.set_bind(Action::ModalRotate, Bind::one(Binding::key(K::R)));
        assert!(k.conflicts().is_empty());

        // Different modifiers on the same key do not clash (Ctrl+A vs Shift+A, as shipped).
        assert!(Keybinds::default().conflicts().is_empty());

        // Held modifier chords are shared on purpose and are never reported.
        let mut k = Keybinds::default();
        k.set_bind(Action::NudgeLarge, Bind::one(Binding::chord(Mods::SHIFT)));
        k.set_bind(Action::SelectAdd, Bind::one(Binding::chord(Mods::SHIFT)));
        assert!(k.conflicts().is_empty());

        // A position binding and a letter binding on the same key DO clash (they are the same
        // key on a US layout), so the user is told rather than surprised.
        let mut k = Keybinds::default();
        k.set_bind(Action::FrameSelected, Bind::one(Binding::pos(K::F)));
        k.set_bind(Action::SelectAll, Bind::one(Binding::key(K::F)));
        assert!(!k.conflicts().is_empty());
    }

    /// The reason the camera keys are stored by POSITION: an AZERTY keyboard reports the key
    /// under the left middle finger as logical `Z` with physical position `W`.
    #[test]
    fn position_bindings_survive_a_non_qwerty_layout() {
        let k = Keybinds::default();
        let mut ph = PhysState::default();

        // AZERTY: pressing the "Z" keycap (which sits where W does on US) -> logical Z, physical W.
        let i = input(vec![key_ev(K::Z, Some(K::W), true, Default::default())], Default::default());
        ph.begin_frame(&i);
        assert!(k.down(Action::CamForward, &i, &ph), "AZERTY forward key did not fly forward");
        // ...and it is NOT undo, because Ctrl is not held and undo is a LETTER binding.
        assert!(!k.pressed(Action::Undo, &i, &ph));

        // The logical Z (undo) still works on the same layout when Ctrl is held: that key reports
        // logical Z wherever it physically is.
        let ctrl = egui::Modifiers { ctrl: true, command: true, ..Default::default() };
        let i = input(vec![key_ev(K::Z, Some(K::W), true, ctrl)], ctrl);
        ph.begin_frame(&i);
        assert!(k.pressed(Action::Undo, &i, &ph));

        // A US keyboard: logical and physical agree, so nothing changed for the existing user.
        let i = input(vec![key_ev(K::W, Some(K::W), true, Default::default())], Default::default());
        ph.begin_frame(&i);
        assert!(k.down(Action::CamForward, &i, &ph));
    }

    #[test]
    fn held_keys_stay_held_until_released_or_focus_is_lost() {
        let k = Keybinds::default();
        let mut ph = PhysState::default();
        let i = input(vec![key_ev(K::W, Some(K::W), true, Default::default())], Default::default());
        ph.begin_frame(&i);
        assert!(k.down(Action::CamForward, &i, &ph));

        // next frame, no events at all: still held
        let i2 = input(vec![], Default::default());
        ph.begin_frame(&i2);
        assert!(ph.is_down(K::W));

        // release
        let i3 = input(vec![key_ev(K::W, Some(K::W), false, Default::default())], Default::default());
        ph.begin_frame(&i3);
        assert!(!ph.is_down(K::W));

        // held, then the window loses focus: the key must not stay stuck down (no key-up arrives)
        let i4 = input(vec![key_ev(K::W, Some(K::W), true, Default::default())], Default::default());
        ph.begin_frame(&i4);
        assert!(ph.is_down(K::W));
        let raw = egui::RawInput { focused: false, ..Default::default() };
        let i5 = egui::InputState::default().begin_pass(raw, false, 1.0, &egui::Options::default());
        ph.begin_frame(&i5);
        assert!(!ph.is_down(K::W), "a held fly key survived focus loss and would fly for ever");
    }

    /// The modifier rules, which is where "exactly today's behaviour" lives.
    #[test]
    fn modifier_rules_match_the_old_hand_written_guards() {
        let k = Keybinds::default();
        let ph = PhysState::default();
        let none = egui::Modifiers::default();
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        let ctrl = egui::Modifiers { ctrl: true, command: true, ..Default::default() };
        let ctrl_shift = egui::Modifiers { ctrl: true, command: true, shift: true, ..Default::default() };

        // Shift+W still flies forward: Shift is the boost, and the camera keys ignore modifiers.
        let i = input(vec![key_ev(K::W, Some(K::W), true, shift)], shift);
        let mut p = ph.clone();
        p.begin_frame(&i);
        assert!(k.down(Action::CamForward, &i, &p));
        assert!(k.down(Action::FlyFaster, &i, &p), "Shift must read as the fly boost");

        // Ctrl+Z is undo, Ctrl+Shift+Z is redo and NOT undo (ModRule::Exact).
        let i = input(vec![key_ev(K::Z, Some(K::Z), true, ctrl)], ctrl);
        assert!(k.pressed(Action::Undo, &i, &ph));
        assert!(!k.pressed(Action::Redo, &i, &ph));
        let i = input(vec![key_ev(K::Z, Some(K::Z), true, ctrl_shift)], ctrl_shift);
        assert!(!k.pressed(Action::Undo, &i, &ph), "Ctrl+Shift+Z must not undo");
        assert!(k.pressed(Action::Redo, &i, &ph));

        // Ctrl+A selects all; Shift+A opens the menu; neither fires for the other's modifiers.
        let i = input(vec![key_ev(K::A, Some(K::A), true, ctrl)], ctrl);
        assert!(k.pressed(Action::SelectAll, &i, &ph));
        assert!(!k.pressed(Action::ContextMenu, &i, &ph));
        let i = input(vec![key_ev(K::A, Some(K::A), true, shift)], shift);
        assert!(k.pressed(Action::ContextMenu, &i, &ph));
        assert!(!k.pressed(Action::SelectAll, &i, &ph));
        let i = input(vec![key_ev(K::A, Some(K::A), true, ctrl_shift)], ctrl_shift);
        assert!(!k.pressed(Action::ContextMenu, &i, &ph), "Ctrl+Shift+A must not open the menu");

        // F frames the selection only with no modifiers (ModRule::Exact).
        let i = input(vec![key_ev(K::F, Some(K::F), true, none)], none);
        assert!(k.pressed(Action::FrameSelected, &i, &ph));
        let i = input(vec![key_ev(K::F, Some(K::F), true, ctrl)], ctrl);
        assert!(!k.pressed(Action::FrameSelected, &i, &ph));

        // C toggles the Construct tool, but Ctrl+C (copy) must not (ModRule::NoCtrl).
        let i = input(vec![key_ev(K::C, Some(K::C), true, none)], none);
        assert!(k.pressed(Action::ConstructTool, &i, &ph));
        let i = input(vec![key_ev(K::C, Some(K::C), true, ctrl)], ctrl);
        assert!(!k.pressed(Action::ConstructTool, &i, &ph));
        // ...and Shift+C still does, as it always did.
        let i = input(vec![key_ev(K::C, Some(K::C), true, shift)], shift);
        assert!(k.pressed(Action::ConstructTool, &i, &ph));

        // Delete and Backspace both delete (the two-slot binding).
        let i = input(vec![key_ev(K::Backspace, Some(K::Backspace), true, none)], none);
        assert!(k.pressed(Action::Delete, &i, &ph));
    }

    /// #play-gate  TEST MODE'S OWN KEYS FIRE THROUGH THE BINDING TABLE, with real egui key
    /// events — the production read path, not a hand-made bool.
    ///
    /// This is the gate on the failure mode this project has been burned by twice: a feature that
    /// built its own inputs and called its own pass directly, so the naming test passed while the
    /// thing was dead in the app. Here the stick and the buttons come out of `play_stick` /
    /// `play_buttons`, which are the only readers test mode has.
    #[test]
    fn the_test_mode_stick_and_buttons_come_out_of_the_binding_table() {
        let k = Keybinds::default();
        let none = egui::Modifiers::default();
        let ctrl = egui::Modifiers { ctrl: true, command: true, ..Default::default() };
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        let alt = egui::Modifiers { alt: true, ..Default::default() };
        let state = |evs: Vec<egui::Event>, m: egui::Modifiers| {
            let i = input(evs, m);
            let mut ph = PhysState::default();
            ph.begin_frame(&i);
            (i, ph)
        };

        // The movement stick IS the camera cluster: W forward, S back, D right, A left.
        let (i, ph) = state(vec![key_ev(K::W, Some(K::W), true, none)], none);
        assert_eq!(play_stick(&k, &i, &ph), (1.0, 0.0), "W must walk forward");
        let (i, ph) = state(vec![key_ev(K::S, Some(K::S), true, none)], none);
        assert_eq!(play_stick(&k, &i, &ph), (-1.0, 0.0));
        let (i, ph) = state(vec![key_ev(K::D, Some(K::D), true, none)], none);
        assert_eq!(play_stick(&k, &i, &ph), (0.0, 1.0));
        let (i, ph) = state(vec![key_ev(K::A, Some(K::A), true, none)], none);
        assert_eq!(play_stick(&k, &i, &ph), (0.0, -1.0));
        // Nothing held is a centred stick, so a parked player can never creep.
        let (i, ph) = state(vec![], none);
        assert_eq!(play_stick(&k, &i, &ph), (0.0, 0.0));

        // Space jumps, V swings, Ctrl crouches, Shift sprints, Alt walks.
        let (i, ph) = state(vec![key_ev(K::Space, Some(K::Space), true, none)], none);
        assert_eq!(play_buttons(&k, &i, &ph), (true, false, false, false, false), "Space must jump");
        let (i, ph) = state(vec![key_ev(K::V, Some(K::V), true, none)], none);
        assert_eq!(play_buttons(&k, &i, &ph), (false, false, false, false, true), "V must swing");
        let (i, ph) = state(vec![], ctrl);
        assert_eq!(play_buttons(&k, &i, &ph).1, true, "Ctrl must crouch");
        let (i, ph) = state(vec![], shift);
        assert_eq!(play_buttons(&k, &i, &ph).2, true, "Shift must sprint");
        let (i, ph) = state(vec![], alt);
        assert_eq!(play_buttons(&k, &i, &ph).3, true, "Alt must walk");
        let (i, ph) = state(vec![], none);
        assert_eq!(play_buttons(&k, &i, &ph), (false, false, false, false, false));
    }

    /// #play-gate  A REBIND MOVES TEST MODE'S FEET TOO. The point of reusing the camera bindings
    /// is that one remap covers both, so this asserts it rather than assuming it — and the
    /// left-handed preset is checked as a whole, because that is the case a real user hits.
    #[test]
    fn a_rebind_and_a_left_handed_preset_move_the_players_feet() {
        let none = egui::Modifiers::default();
        let state = |evs: Vec<egui::Event>| {
            let i = input(evs, none);
            let mut ph = PhysState::default();
            ph.begin_frame(&i);
            (i, ph)
        };
        // A single rebind.
        let mut k = Keybinds::default();
        k.set_bind(Action::CamForward, Bind::one(Binding::key(K::T)));
        let (i, ph) = state(vec![key_ev(K::T, Some(K::T), true, none)]);
        assert_eq!(play_stick(&k, &i, &ph).0, 1.0, "the rebound key must walk the player forward");
        let (i, ph) = state(vec![key_ev(K::W, Some(K::W), true, none)]);
        assert_eq!(play_stick(&k, &i, &ph).0, 0.0, "the old key must no longer walk");
        // And the shipped left-handed preset, which puts forward on the up arrow.
        let lh = Keybinds::of(Preset::LeftHandedArrows);
        let (i, ph) = state(vec![key_ev(K::ArrowUp, Some(K::ArrowUp), true, none)]);
        assert_eq!(play_stick(&lh, &i, &ph).0, 1.0, "a left-handed player must be able to walk forward");
    }

    /// #play-gate  Mouse-look scales with the sensitivity multiplier and clamps the pitch, so an
    /// imported MCC sensitivity can be applied without the look handling changing shape.
    #[test]
    fn play_look_scales_and_clamps() {
        let (y1, p1) = play_look(0.0, 0.0, 100.0, 0.0, 1.0);
        let (y2, _) = play_look(0.0, 0.0, 100.0, 0.0, 4.5);
        assert!(y1 < 0.0 && (y2 / y1 - 4.5).abs() < 1e-4, "{y1} {y2}");
        assert_eq!(p1, 0.0);
        // Straight up and straight down are clamped where the editor camera clamps.
        assert_eq!(play_look(0.0, 0.0, 0.0, -1e6, 1.0).1, 1.5);
        assert_eq!(play_look(0.0, 0.0, 0.0, 1e6, 1.0).1, -1.5);
    }

    /// Rebinding actually changes what fires — the whole feature in one test.
    #[test]
    fn a_rebind_changes_which_key_flies() {
        let mut k = Keybinds::default();
        let ph = PhysState::default();
        let none = egui::Modifiers::default();

        let w = input(vec![key_ev(K::W, Some(K::W), true, none)], none);
        let up = input(vec![key_ev(K::ArrowUp, Some(K::ArrowUp), true, none)], none);
        let mut pw = ph.clone();
        pw.begin_frame(&w);
        let mut pu = ph.clone();
        pu.begin_frame(&up);

        assert!(k.down(Action::CamForward, &w, &pw));
        assert!(!k.down(Action::CamForward, &up, &pu));

        k.set_bind(Action::CamForward, Bind::one(Binding::key(K::ArrowUp)));
        assert!(!k.down(Action::CamForward, &w, &pw), "W still flies forward after being rebound away");
        assert!(k.down(Action::CamForward, &up, &pu), "the new key does not fly forward");

        // and the whole left-handed preset, in one click
        let k = Keybinds::of(Preset::LeftHandedArrows);
        assert!(k.down(Action::CamForward, &up, &pu));
        assert!(!k.down(Action::CamForward, &w, &pw));
    }

    /// The CAMERA PATH, not just the lookup: `fly_velocity` is the whole of what the fly keys do
    /// in `update_camera`, so driving it directly shows the camera responds to the ACTION and
    /// never to a fixed key. No synthetic keystrokes anywhere — the input state is built here.
    #[test]
    fn the_camera_follows_the_binding_not_the_key() {
        let fwd = glam::Vec3::new(1.0, 0.0, 0.0);
        let right = glam::Vec3::new(0.0, 1.0, 0.0);
        let up = glam::Vec3::Z;
        let none = egui::Modifiers::default();
        // Hold one key and ask where the camera would go this frame.
        let step = |keys: &Keybinds, key: K, phys: K| {
            let i = input(vec![key_ev(key, Some(phys), true, none)], none);
            let mut ph = PhysState::default();
            ph.begin_frame(&i);
            fly_velocity(keys, &i, &ph, fwd, right, up, 10.0, 0.1)
        };

        // --- shipped defaults: W forward, S back, D right, A left, E up, Q down ---
        let d = Keybinds::default();
        assert_eq!(step(&d, K::W, K::W), fwd * 1.0, "W must fly forward by move_speed*dt");
        assert_eq!(step(&d, K::S, K::S), -fwd * 1.0);
        assert_eq!(step(&d, K::D, K::D), right * 1.0);
        assert_eq!(step(&d, K::A, K::A), -right * 1.0);
        assert_eq!(step(&d, K::E, K::E), up * 1.0);
        assert_eq!(step(&d, K::R, K::R), up * 1.0, "R is the second fly-up key");
        assert_eq!(step(&d, K::Q, K::Q), -up * 1.0);
        assert_eq!(step(&d, K::F, K::F), -up * 1.0, "F is the second fly-down key");
        // ...and the arrow keys do NOTHING to the camera by default.
        assert_eq!(step(&d, K::ArrowUp, K::ArrowUp), glam::Vec3::ZERO);

        // --- rebind forward to the up arrow: the camera follows, W goes dead ---
        let mut k = Keybinds::default();
        k.set_bind(Action::CamForward, Bind::one(Binding::key(K::ArrowUp)));
        assert_eq!(step(&k, K::ArrowUp, K::ArrowUp), fwd * 1.0, "the rebound key does not fly the camera");
        assert_eq!(step(&k, K::W, K::W), glam::Vec3::ZERO, "W still flies the camera after being rebound away");

        // --- the whole left-handed preset, in one click ---
        let l = Keybinds::of(Preset::LeftHandedArrows);
        assert_eq!(step(&l, K::ArrowUp, K::ArrowUp), fwd * 1.0);
        assert_eq!(step(&l, K::ArrowDown, K::ArrowDown), -fwd * 1.0);
        assert_eq!(step(&l, K::ArrowLeft, K::ArrowLeft), -right * 1.0);
        assert_eq!(step(&l, K::ArrowRight, K::ArrowRight), right * 1.0);
        assert_eq!(step(&l, K::PageUp, K::PageUp), up * 1.0);
        assert_eq!(step(&l, K::PageDown, K::PageDown), -up * 1.0);
        assert_eq!(step(&l, K::W, K::W), glam::Vec3::ZERO, "the left-handed set must free WASD");
        let c = Keybinds::of(Preset::LeftHandedCluster);
        assert_eq!(step(&c, K::I, K::I), fwd * 1.0);
        assert_eq!(step(&c, K::K, K::K), -fwd * 1.0);
        assert_eq!(step(&c, K::ArrowUp, K::ArrowUp), glam::Vec3::ZERO, "arrows keep nudging in this set");

        // --- AZERTY: the physical WASD cluster still flies, with no settings at all ---
        // (the key under the left middle finger reports logical Z at physical position W)
        assert_eq!(step(&d, K::Z, K::W), fwd * 1.0, "AZERTY cannot fly forward");
        assert_eq!(step(&d, K::Q, K::A), -right * 1.0, "AZERTY cannot strafe left");
        // Dvorak's comma sits where W does.
        assert_eq!(step(&d, K::Comma, K::W), fwd * 1.0, "Dvorak cannot fly forward");

        // --- the speed modifiers ---
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        let alt = egui::Modifiers { alt: true, ..Default::default() };
        let hold = |m: egui::Modifiers| {
            let i = input(vec![key_ev(K::W, Some(K::W), true, m)], m);
            let mut ph = PhysState::default();
            ph.begin_frame(&i);
            fly_velocity(&d, &i, &ph, fwd, right, up, 10.0, 0.1)
        };
        assert_eq!(hold(shift), fwd * 4.0, "the fly-faster modifier must quadruple the step");
        assert_eq!(hold(alt), fwd * 0.25, "the fly-slower modifier must quarter the step");

        // --- nothing held: the camera must never creep ---
        let i = input(vec![], none);
        let ph = PhysState::default();
        assert_eq!(fly_velocity(&d, &i, &ph, fwd, right, up, 10.0, 0.1), glam::Vec3::ZERO);

        // --- and while the rebinding page is listening, no key flies anything ---
        let mut listening = Keybinds::default();
        listening.listening = true;
        assert_eq!(step(&listening, K::W, K::W), glam::Vec3::ZERO, "the camera flew away mid-capture");
    }

    #[test]
    fn unbound_actions_never_fire() {
        let mut k = Keybinds::default();
        k.set_bind(Action::Screenshot, Bind::one(Binding::UNBOUND));
        let i = input(
            vec![key_ev(K::F12, Some(K::F12), true, Default::default())],
            Default::default(),
        );
        let mut ph = PhysState::default();
        ph.begin_frame(&i);
        assert!(!k.pressed(Action::Screenshot, &i, &ph));
    }

    #[test]
    fn capture_commits_keys_and_chords_and_refuses_the_reserved_ones() {
        let none = egui::Modifiers::default();
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        let ph = PhysState::default();

        // a plain letter, captured by letter
        let mut c = Capture::new(Action::Undo, 0);
        let i = input(vec![key_ev(K::Q, Some(K::Q), true, none)], none);
        match c.update(&i, &ph, false) {
            CaptureResult::Bound(b) => assert_eq!(b, Binding::key(K::Q)),
            _ => panic!("a letter did not commit"),
        }

        // the same event captured by POSITION records the physical key, not the letter
        let mut c = Capture::new(Action::CamForward, 0);
        let i = input(vec![key_ev(K::Z, Some(K::W), true, none)], none);
        match c.update(&i, &ph, true) {
            CaptureResult::Bound(b) => assert_eq!(b, Binding::pos(K::W)),
            _ => panic!("a position capture did not commit"),
        }

        // modifiers come along
        let mut c = Capture::new(Action::Duplicate, 0);
        let i = input(vec![key_ev(K::J, Some(K::J), true, shift)], shift);
        match c.update(&i, &ph, false) {
            CaptureResult::Bound(b) => assert_eq!(b, Binding::key(K::J).with(Mods::SHIFT)),
            _ => panic!("Shift+J did not commit"),
        }

        // Escape cancels
        let mut c = Capture::new(Action::Undo, 0);
        let i = input(vec![key_ev(K::Escape, Some(K::Escape), true, none)], none);
        assert!(matches!(c.update(&i, &ph, false), CaptureResult::Cancelled));

        // Tab is refused with a reason, and the capture stays open
        let mut c = Capture::new(Action::Undo, 0);
        let i = input(vec![key_ev(K::Tab, Some(K::Tab), true, none)], none);
        assert!(matches!(c.update(&i, &ph, false), CaptureResult::Waiting));
        assert!(c.refused.is_some());

        // a pure modifier chord commits when the last modifier is released
        let mut c = Capture::new(Action::FlyFaster, 0);
        let held = input(vec![], shift);
        assert!(matches!(c.update(&held, &ph, false), CaptureResult::Waiting));
        let let_go = input(vec![], none);
        match c.update(&let_go, &ph, false) {
            CaptureResult::Bound(b) => assert_eq!(b, Binding::chord(Mods::SHIFT)),
            _ => panic!("a modifier chord did not commit on release"),
        }
    }

    /// The shortcuts window prints these; they must read like keys, not like debug output.
    #[test]
    fn labels_are_readable() {
        let k = Keybinds::default();
        assert_eq!(k.label(Action::CamForward), "W");
        assert_eq!(k.label(Action::CamUp), "E or R");
        assert_eq!(k.label(Action::Duplicate), "Shift+D");
        assert_eq!(k.label(Action::Redo), "Ctrl+Y or Ctrl+Shift+Z");
        assert_eq!(k.label(Action::Delete), "Del or Backspace");
        assert_eq!(k.label(Action::FlyFaster), "Shift");
        assert_eq!(k.label(Action::FlyLook), "Right mouse");
        assert_eq!(k.label(Action::NudgeXPlus), "Right arrow");
        assert_eq!(k.label(Action::YawPlus), "]");
        assert_eq!(k.labels(&[Action::CamForward, Action::CamLeft, Action::CamBack, Action::CamRight]), "W A S D");
        let mut k = Keybinds::default();
        k.set_bind(Action::Screenshot, Bind::one(Binding::UNBOUND));
        assert_eq!(k.label(Action::Screenshot), "—");
    }
}

// ===================== #play-hud  THE RELOAD KEY'S CALL SITE =====================
//
// A bound key that nothing reads is worse than no key: it appears in the controls list, the user
// rebinds it, presses it, and nothing happens. Every gate on `Action::PlayReload` below this
// comment would stay green in that state, because they all ask the keybind table about itself.
//
// So the host is gated structurally, the same way the HUD's draw site and the MCC FOV channel are.
// This is the fifth time in this project that a capability existed and nothing invoked it.
#[cfg(test)]
mod play_reload_call_site_gates {
    #[test]
    fn the_reload_action_is_in_the_table_and_has_a_spec() {
        // The positive content first: the action exists, is listed, and describes itself.
        assert!(
            super::Action::ALL.contains(&super::Action::PlayReload),
            "PlayReload must be in Action::ALL or it never appears in the controls list"
        );
        let spec = super::Action::PlayReload.spec();
        assert_eq!(spec.id, "play_reload", "the saved-settings id must not drift");
        assert_eq!(spec.label, "Reload");
        // It is an EDGE action, not a held one, so the UI offers a press rather than a hold.
        assert!(!spec.held, "reload is a press, not a hold");
        assert!(!spec.mouse, "reload is a key, not a pointer button");
    }

    #[test]
    fn the_host_reads_the_reload_key_and_calls_the_thing_it_promises() {
        let src = include_str!("main.rs");
        assert!(
            src.contains("keybinds::play_reload("),
            "main.rs must read the reload key, or the binding does nothing"
        );
        // And reading it is not enough -- it has to reach the reload itself, IN THE SAME
        // FUNCTION. Bounded by the next function boundary rather than by a character count: the
        // first version of this gate used a 4000-character window and went red because the two
        // sites are 4100 apart, which is a number derivable from the fixture and therefore a
        // fault in the ruler, not the code. Widening it to 5000 would have been an arbitrary
        // number waiting to break again; "inside the same function" is the relationship that
        // actually matters.
        let read = src.find("keybinds::play_reload(").unwrap();
        let next_fn = src[read..]
            .find("\n    fn ")
            .map(|i| read + i)
            .unwrap_or(src.len());
        let body = &src[read..next_fn];
        assert!(
            body.contains("start_reload()"),
            "reading the reload key must lead to play_hud.start_reload() in the same function, or \
             the key is read and discarded -- which looks identical to it working"
        );
    }

    /// The reader itself must be edge-triggered, not level-triggered: a held key that restarts the
    /// reload every frame would never finish one.
    #[test]
    fn the_reload_reader_is_an_edge_not_a_hold() {
        let src = include_str!("keybinds.rs");
        let f = src
            .find("pub fn play_reload(")
            .expect("play_reload must exist; if renamed, update this gate");
        let body = &src[f..(f + 400).min(src.len())];
        assert!(
            body.contains("keys.pressed("),
            "play_reload must use `pressed` (edge), not `down` (level)"
        );
        assert!(
            !body.contains("keys.down("),
            "a held reload key must not re-trigger every frame"
        );
    }
}
