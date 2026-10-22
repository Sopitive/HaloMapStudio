//! #weapon-fire  **The weapon firing state machine, ported from the engine's own.**
//!
//! Reported: *"There isn't logic for automatic weapons either so I can't hold the trigger and have
//! an assault rifle keep firing until its clip is empty"*, and then *"weapons also have a fire
//! recovery time too and other firing parameters that would gate how often you can actually pull
//! the trigger or how many projectiles actually fire per shot"*.
//!
//! Test mode used to fire one shot per click: a 0.25 s timer was the only rate limit and the only
//! thing a click could do. The engine has no such thing. A weapon is a handful of small state
//! machines -- one per TRIGGER, one per BARREL, one per MAGAZINE -- ticked in a fixed order at the
//! simulation rate (`sub_140DF7640`: heat, magazines, triggers, barrels). This module is those
//! machines, tick for tick, fed by the tag fields [`crate::reach_weapon`] decodes.
//!
//! # What decides how often you can shoot
//!
//! There is no `fire rate` field that is "the" rate. Four authored things interact, which is why a
//! weapon's tag looks wrong until you run it:
//!
//! 1. **The trigger's `behavior`** decides what a held button MEANS. `spew` asks the barrel to fire
//!    EVERY tick the button is down (automatics); `latch` asks once per press and only re-arms when
//!    the button comes UP (semi-autos); `latch-autofire` is "tap = shot, hold = charge" (the plasma
//!    pistol); `charge` / `spew-charge` charge while held and discharge on release (plasma launcher,
//!    Spartan laser).
//! 2. **The barrel's `rounds per second`** is the interval between two rounds of ONE STRING of
//!    shots (`sub_140DE4120`, snapped to the 30 Hz grid: 10 rps = 6 ticks, 7.5 rps = 8 ticks). It
//!    only matters while the barrel is in its FIRING phase -- i.e. while the trigger is still
//!    asking, or inside a `shots per fire` burst.
//! 3. **`fire recovery time`** is what happens when the string ENDS: the trigger came up, or the
//!    `shots per fire` cap was reached. The barrel is locked for `(1 - soft recovery fraction)` of
//!    it and then takes requests for the soft remainder (`sub_140DE3F80`). A semi-auto's rate IS
//!    its recovery time; an automatic's recovery only shows when you let go.
//! 4. **The magazine**: `rounds per shot`, `minimum rounds loaded`, `can fire with partial ammo`
//!    (`weapon_magazine_can_fire_weapon`), and an empty pull is a dry CLICK that locks the barrel
//!    until the button is released.
//!
//! # Time
//!
//! Everything is counted in 60 Hz ticks ([`TICK_HZ`]). The caller hands in wall
//! seconds ([`WeaponFire::advance`]) and the machine banks them into whole ticks, so the number of
//! shots in a held second is the same at 30, 60 or 144 frames per second.
//!
//! # What is NOT modelled (named, not hidden)
//!
//! * Weapon AGE / misfire (a campaign mechanic), network prediction, the `+0x26B` burst-hold byte.
//! * The error added by MOVING and by TURNING (`sub_140DE2D10` / `sub_140DE2F80`); the error one
//!   ROUND adds (`sub_140DE2A60`) and its decay ARE modelled.
//! * Flight: a projectile is resolved by the caller (see `PlayMode::fire`); this module only says
//!   how many leave the barrel, along which offsets, and what they cost.

use std::sync::Arc;

use crate::reach_weapon::*;

// ---------------------------------------------------------------------------------------------
// the engine's clock
// ---------------------------------------------------------------------------------------------

/// The simulation tick rate (`game_time_globals + 6`, read by `sub_1400BD300`): the same 60 Hz
/// [`crate::physics_settle::TICK_HZ`] is.
pub const TICK_HZ: i32 = 60;
/// `sub_1400BD360() = tick_rate / 30`: how many ticks one step of the authored 30 Hz grid is.
pub const GRID_STEP: i32 = TICK_HZ / 30;
/// Seconds per tick (`game_time_globals + 8`, what `object_get_orientation_PROP` returns).
pub const TICK_SECONDS: f32 = 1.0 / TICK_HZ as f32;

/// `sub_1400BD1F0(seconds)`: seconds -> ticks, ROUNDED (`(int)(0.5 + s * rate)`). What the trigger
/// timers (`charged time`, `autofire time`) use. PROVEN from the decompile.
pub fn seconds_to_ticks(s: f32) -> i32 {
    let v = TICK_HZ as f32 * s;
    if v < 0.0 { (v - 0.5) as i32 } else { (v + 0.5) as i32 }
}

/// `sub_1400BD070(seconds)`: seconds -> ticks on the 30 Hz GRID (`GRID_STEP * round(s * 30)`).
pub fn grid_ticks(s: f32) -> i32 {
    let v = s * 30.0;
    GRID_STEP * if v < 0.0 { (v - 0.5) as i32 } else { (v + 0.5) as i32 }
}

/// `sub_140DE3F80`: the barrel's recovery after a string of shots, in TICKS: `(total, locked)`.
///
/// The authored time is converted on the 30 Hz grid (`s * 30`), floored at one grid step, and every
/// grid step is `GRID_STEP` ticks; `soft recovery fraction` of it is the QUEUEABLE tail, so
/// `locked = (1 - soft) * recovery`. PROVEN from the decompile; the `+ (GRID_STEP - 1) * 2` and the
/// extra tick when nothing is locked are the engine's own rounding terms, not ours.
pub fn recovery_ticks(b: &BarrelDef) -> (i32, i32) {
    let k = GRID_STEP as f32;
    let rec = b.fire_recovery_time * 30.0;
    let v8 = if rec > k { rec } else { k };
    let locked = k * ((1.0 - b.soft_recovery_fraction) * v8).floor();
    let extra = if locked == 0.0 { GRID_STEP - 1 } else { 0 };
    let total = k * v8.floor() + (extra + (GRID_STEP - 1) * 2) as f32;
    (total as i32, locked as i32)
}

/// `sub_140DE4120`: ticks between two rounds of a string at the barrel's current cyclic rate
/// (`rof` in 0..1 walks `rounds per second` initial -> final). The rate is snapped to the 30 Hz
/// grid first (`30 / ceil(30 / rps)`), so a 10 rps weapon is exactly 6 ticks and a 7 rps one is
/// `ceil(30/7) = 5` grid steps = 10 ticks.
pub fn rate_interval_ticks(b: &BarrelDef, rof: f32) -> f32 {
    let rps = (b.rounds_per_second[1] - b.rounds_per_second[0]) * rof + b.rounds_per_second[0];
    if rps <= 1.0e-4 {
        return 0.0;
    }
    let snapped = 30.0 / (30.0 / rps).ceil();
    if snapped <= 1.0e-4 { 0.0 } else { TICK_HZ as f32 / snapped }
}

// ---------------------------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------------------------

/// A barrel's firing phase (`barrel_state + 0x241`, the byte `sub_140DEAB40` switches on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrelPhase {
    /// 0: waiting for a request.
    Idle,
    /// 1: a string of shots is in progress (firing, or waiting for the rate-of-fire interval).
    Firing,
    /// 2: locked recovery after a string that ended normally.
    Locked,
    /// 3: locked recovery after a DRY pull; also runs the auto-reload test every tick.
    LockedDry,
    /// 4: the soft tail of the recovery. A request is accepted but only survives to the end of it
    /// when it arrives in the last tick (`sub_140DEA9C0` sets the carry bit then).
    Soft,
}

#[derive(Clone, Debug)]
pub struct BarrelState {
    pub phase: BarrelPhase,
    /// `+0x240`: ticks since the last round (saturating at 127).
    pub counter: i32,
    /// `+0x242`: ticks left in the current recovery phase.
    pub timer: i32,
    /// `+0x244` bit 0: a fire request is pending.
    pub request: bool,
    /// `+0x244` bit 9 (0x200): the request survives the end of the recovery.
    pub carry: bool,
    /// `+0x244` bit 6 (0x40): a dry click was made; locked until the trigger is released.
    pub empty_clicked: bool,
    /// `+0x246`: rounds fired in this string.
    pub shots_fired: i32,
    /// `+0x250`: the cyclic rate, 0 = `rounds per second` initial .. 1 = final.
    pub rof: f32,
    /// `+0x25C`: the firing error, 0..1 (accuracy penalty; decays at `error deceleration`).
    pub error: f32,
}

impl Default for BarrelState {
    fn default() -> Self {
        BarrelState {
            phase: BarrelPhase::Idle,
            counter: 0,
            timer: 0,
            request: false,
            carry: false,
            empty_clicked: false,
            shots_fired: 0,
            rof: 0.0,
            error: 0.0,
        }
    }
}

/// A trigger's state (`weapon + 0x2A8 + 12 * i`).
#[derive(Clone, Debug)]
pub struct TriggerState {
    /// `+0x2A8`: 0 idle, 1 charging, 2 charged, 3 discharging (a spew of the primary barrel).
    pub state: u8,
    /// `+0x2AC`: ticks left in the state.
    pub timer: i32,
    /// `+0x2A9` bit 0: armed -- the NEXT press may act. A latch disarms itself on a press and
    /// re-arms when the button comes up.
    pub armed: bool,
    /// bits 3 / 4: the autofire's primary / secondary action has been taken for this press.
    pub primary_done: bool,
    pub secondary_done: bool,
    /// `+0x1CC .. +0x1D0` of the weapon: the (digital: 0 or 255) trigger value now and last tick,
    /// the reference depth, the direction (0 none, 1 pulling, 0xFF released) and how long it has
    /// been held. See `sub_140DF5970`.
    cur: i32,
    prev: i32,
    reference: i32,
    direction: u8,
    held_ticks: i32,
    /// How many rounds the discharge owes (`+0x2AB`).
    pub owed: i32,
}

impl Default for TriggerState {
    fn default() -> Self {
        TriggerState {
            state: 0,
            timer: 0,
            armed: true,
            primary_done: false,
            secondary_done: false,
            cur: 0,
            prev: 0,
            reference: 0,
            direction: 0,
            held_ticks: 0,
            owed: 0,
        }
    }
}

/// What the caller says about THIS tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct FireInput {
    /// The primary (right) trigger is down.
    pub held: bool,
    /// The secondary (left) trigger is down. Test mode binds none, so this is only ever driven by
    /// the gates.
    pub alt_held: bool,
    /// `PlayMode::weapon_usable()`: false in a seat that does not allow the biped's weapon.
    pub usable: bool,
    /// A reload is running (the HUD's clock). The engine refuses a pull while it reloads.
    pub reloading: bool,
    /// There are rounds in reserve to reload from.
    pub has_reserve: bool,
    pub crouched: bool,
    /// A melee swing is running: the weapon is busy, so it is disabled the way
    /// `sub_140DEAB40`'s `weapon_disabled` is. (Sprinting is NOT modelled as a lockout: nothing
    /// decompiled says what raises the engine's lowered-weapon bit, and refusing a shot on a guess
    /// would be worse than allowing one.)
    pub melee: bool,
}

/// One projectile's launch offset from the aim direction: an angle off axis and an azimuth around
/// it (the error cone), plus a yaw added by the `horizontal fan` distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub angle: f32,
    pub azimuth: f32,
    pub fan_yaw: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FireEvent {
    /// A round (a "firing effect") left barrel `barrel`: `rounds` were spent and `projectiles`
    /// launched. The caller owns the fire sound, the muzzle effect and the ray.
    Shot { barrel: usize, rounds: i32, projectiles: Vec<Spread> },
    /// The trigger was pulled on a barrel with too little ammunition: the dry click.
    EmptyClick { barrel: usize },
    /// The engine would start a reload now (`sub_140DE3D50`); idempotent for the caller.
    NeedReload,
    /// The weapon crossed `overheated threshold` / cooled below `heat recovery threshold`.
    Overheated,
    Cooled,
}

#[derive(Clone, Debug, Default)]
pub struct FireStats {
    pub ticks: u64,
    pub shots: u64,
    pub rounds: u64,
    pub empty_clicks: u64,
    pub projectiles: u64,
}

/// One held weapon's whole firing state.
#[derive(Clone, Debug)]
pub struct WeaponFire {
    pub def: Arc<WeaponDef>,
    pub barrels: Vec<BarrelState>,
    pub triggers: Vec<TriggerState>,
    /// Rounds loaded per magazine. The HUD owns the authoritative count between frames and
    /// pushes it in with [`WeaponFire::sync_loaded`]; this machine decrements it as it fires.
    pub loaded: Vec<i32>,
    /// `weapon + 0x1E0`: 0..1.
    pub heat: f32,
    /// `weapon + 0x1E4`: 0..1. The BATTERY of a plasma weapon: every round adds `age generated per
    /// round`, and at 1.0 a weapon flagged `cannot fire at maximum age` is dead (the plasma pistol
    /// is 100 bolts, the rifle 400, the repeater 200). Not reset by anything but a new weapon.
    pub age: f32,
    pub overheated: bool,
    /// Ticks of venting left (a weapon with no heat recovery threshold).
    vent_ticks: i32,
    /// Ticks the weapon is lowered after being drawn.
    pub ready_ticks: i32,
    rng: u32,
    bank: f64,
    pub stats: FireStats,
}

/// Per-tick context shared by the machines.
struct Ctx<'a> {
    inp: &'a FireInput,
    /// `weapon_disabled`: overheated, still being drawn, busy with a melee, or unusable.
    disabled: bool,
}

impl WeaponFire {
    /// A freshly drawn weapon. `loaded[i]` is magazine `i`'s round count.
    pub fn new(def: Arc<WeaponDef>, loaded: Vec<i32>) -> WeaponFire {
        let ready = seconds_to_ticks(def.ready_time);
        WeaponFire {
            barrels: vec![BarrelState::default(); def.barrels.len()],
            triggers: vec![TriggerState::default(); def.triggers.len()],
            def,
            loaded,
            heat: 0.0,
            age: 0.0,
            overheated: false,
            vent_ticks: 0,
            ready_ticks: ready,
            rng: 0x1234_5678,
            bank: 0.0,
            stats: FireStats::default(),
        }
    }

    /// The HUD's count wins between frames (a reload completed, a pickup refilled).
    pub fn sync_loaded(&mut self, magazine: usize, rounds: i32) {
        if let Some(l) = self.loaded.get_mut(magazine) {
            *l = rounds;
        }
    }

    /// Bank `dt` seconds into whole ticks and run them. `input(tick_index)` is read once per tick
    /// so a caller can say the button was released half-way through a long frame. Returns the
    /// events in tick order.
    pub fn advance(&mut self, dt: f32, input: &FireInput) -> Vec<FireEvent> {
        self.bank += dt.max(0.0) as f64 * TICK_HZ as f64;
        let n = self.bank.floor();
        // A tiny epsilon so 1/60 s * 60 does not land on 0.99999 and lose a tick forever.
        let n = if self.bank - n > 1.0 - 1.0e-6 { n + 1.0 } else { n };
        self.bank -= n;
        let mut out = Vec::new();
        for _ in 0..n as i64 {
            self.tick(input, &mut out);
        }
        out
    }

    fn rand(&mut self) -> f32 {
        // The engine's own LCG (`seed * 0x19660D + 0x3C6EF35F`, 16 high bits / 65535).
        self.rng = self.rng.wrapping_mul(0x19660D).wrapping_add(0x3C6E_F35F);
        (self.rng >> 16) as f32 * 1.525_902_2e-5
    }

    /// One 60 Hz tick: heat, then triggers, then barrels (`sub_140DF7640`'s order).
    pub fn tick(&mut self, inp: &FireInput, out: &mut Vec<FireEvent>) {
        self.stats.ticks += 1;
        let def = self.def.clone();
        if self.ready_ticks > 0 {
            self.ready_ticks -= 1;
        }
        let disabled = self.overheated
            || self.ready_ticks > 0
            || inp.melee
            || !inp.usable
            || def.flags & WF_MELEE_ONLY != 0;
        let ctx = Ctx { inp, disabled };
        self.update_heat(&def, out);
        for ti in 0..def.triggers.len() {
            self.trigger_tick(&def, ti, &ctx);
        }
        for bi in 0..def.barrels.len() {
            self.barrel_tick(&def, bi, &ctx, out);
        }
    }

    // ----- heat (sub_140DF7E40) ---------------------------------------------------------------

    fn update_heat(&mut self, def: &WeaponDef, out: &mut Vec<FireEvent>) {
        if self.overheated && self.vent_ticks > 0 {
            self.vent_ticks -= 1;
            if self.vent_ticks == 0 {
                self.overheated = false;
                self.heat = self.heat.min(def.heat_venting_exit_heat);
                out.push(FireEvent::Cooled);
            }
        }
        if self.heat <= 0.0 {
            return;
        }
        let splaser = def.weapon_type == WEAPON_TYPE_SPLASER;
        // `v5` / `v4`: a barrel that is firing (or mid-string) AND adds heat per round.
        let firing_heat = self.barrels.iter().zip(&def.barrels).any(|(s, b)| {
            (s.phase == BarrelPhase::Firing || s.shots_fired != 0) && b.heat_per_round > 0.0
        });
        if self.heat >= def.overheated_threshold && !(splaser && firing_heat) {
            if def.secondary_flags & WSF_NEVER_OVERHEATS == 0 && !self.overheated {
                self.overheated = true;
                out.push(FireEvent::Overheated);
                // A weapon with no recovery threshold (the plasma repeater) cannot cool its way
                // out of the overheated state; it VENTS for `heat venting time` instead.
                // INFERRED: the venting state machine itself is not decompiled, the authored
                // time and the exit heat are what it is given.
                if def.heat_recovery_threshold <= 0.0 && def.heat_venting_time > 0.0 {
                    self.vent_ticks = seconds_to_ticks(def.heat_venting_time).max(1);
                }
            }
        } else if self.heat >= def.overheated_threshold {
            // The Spartan laser does not cool while it is charging a heat-adding barrel, and does
            // not overheat; it only checks for the exit below.
            if self.overheated && def.heat_recovery_threshold > self.heat {
                self.overheated = false;
                out.push(FireEvent::Cooled);
            }
            return;
        }
        // The loss per second: a function of the heat when the tag names one, else the float; while
        // overheated its own (function or float), falling back to the normal one when it authors 0.
        let normal = def.heat_loss_fn.map_or(def.heat_loss_per_second, |c| sample_curve(&c, self.heat));
        let rate = if self.overheated {
            match def.overheated_heat_loss_fn {
                Some(c) => sample_curve(&c, self.heat),
                None if def.overheated_heat_loss_per_second > 0.0 => def.overheated_heat_loss_per_second,
                None => normal,
            }
        } else {
            normal
        };
        self.heat = (self.heat - rate * TICK_SECONDS).max(0.0);
        if self.overheated && def.heat_recovery_threshold > self.heat {
            self.overheated = false;
            out.push(FireEvent::Cooled);
        }
    }

    // ----- ammunition (weapon_magazine_can_fire_weapon, sub_140DE3D50) ------------------------

    /// `weapon_magazine_can_fire_weapon` for `needed` rounds. No magazine = always.
    fn ammo_ok(&self, b: &BarrelDef, needed: i32) -> bool {
        // `cannot fire at maximum age`, checked before the magazine (a battery weapon has none).
        if self.def.flags & WF_CANNOT_FIRE_AT_MAX_AGE != 0
            && b.flags & BF_CAN_FIRE_AT_MAX_AGE == 0
            && self.at_max_age()
        {
            return false;
        }
        let Some(m) = b.magazine else { return true };
        let loaded = self.loaded.get(m).copied().unwrap_or(0);
        let partial = b.flags & BF_CAN_FIRE_WITH_PARTIAL_AMMO != 0;
        !((loaded < needed && !partial) || loaded < b.minimum_rounds_loaded as i32)
    }

    /// `sub_140DF0720`: has the weapon aged out? The Spartan laser is exempt while a barrel that
    /// adds heat is mid-string, so a charge that has begun can finish.
    fn at_max_age(&self) -> bool {
        if self.age < 1.0 {
            return false;
        }
        if self.def.weapon_type != WEAPON_TYPE_SPLASER {
            return true;
        }
        !self.barrels.iter().zip(&self.def.barrels).any(|(s, b)| s.shots_fired != 0 && b.heat_per_round > 0.0)
    }

    /// `sub_140DE3D50`, the auto-reload test: an idle or dry-locked barrel whose magazine cannot
    /// serve a shot asks for a reload.
    fn auto_reload_check(&self, b: &BarrelDef, inp: &FireInput, out: &mut Vec<FireEvent>) {
        let Some(m) = b.magazine else { return };
        let loaded = self.loaded.get(m).copied().unwrap_or(0);
        let partial = b.flags & BF_CAN_FIRE_WITH_PARTIAL_AMMO != 0;
        let short = (loaded < b.rounds_per_shot as i32 && !partial)
            || loaded < b.minimum_rounds_loaded as i32
            || loaded == 0;
        if short && inp.has_reserve && !inp.reloading {
            out.push(FireEvent::NeedReload);
        }
    }

    // ----- can a barrel take a request (sub_140DE33B0) ----------------------------------------

    fn can_accept(&self, def: &WeaponDef, bi: usize, ctx: &Ctx) -> bool {
        let Some(b) = def.barrels.get(bi) else { return false };
        let st = &self.barrels[bi];
        if self.overheated || ctx.disabled {
            return false;
        }
        // Not while it is in the LOCKED part of a recovery (`sub_140DE9FD0`).
        if matches!(st.phase, BarrelPhase::Locked | BarrelPhase::LockedDry) {
            return false;
        }
        // A weapon with exactly one `latch` trigger needs it armed (`v3[255] == 1 && v4 == 1`).
        if def.triggers.len() == 1
            && def.triggers[0].behavior == Some(Behavior::Latch)
            && !self.triggers[0].armed
        {
            return false;
        }
        if self.ammo_ok(b, b.rounds_per_shot as i32) {
            return true;
        }
        // Too little ammunition. While a reload runs the pull is refused; otherwise it is let
        // through ONCE so it can click, and the click locks the barrel until release.
        if ctx.inp.reloading {
            return false;
        }
        !st.empty_clicked
    }

    /// `sub_140DEA9C0(.., 1)`: ask a barrel to fire. A request that lands in the last tick of the
    /// soft recovery is the one that survives it.
    fn request(&mut self, bi: usize) {
        let Some(st) = self.barrels.get_mut(bi) else { return };
        st.request = true;
        if st.phase == BarrelPhase::Soft && GRID_STEP != 1 && st.timer < GRID_STEP {
            st.carry = true;
        }
    }

    /// `sub_140DE3F40`: the trigger came up -- lift the dry-click lock (unless mid-string).
    fn release(&mut self, bi: usize) {
        if let Some(st) = self.barrels.get_mut(bi) {
            if st.phase != BarrelPhase::Firing {
                st.empty_clicked = false;
            }
        }
    }

    // ----- triggers (sub_140DF64E0) -----------------------------------------------------------

    fn trigger_tick(&mut self, def: &WeaponDef, ti: usize, ctx: &Ctx) {
        let td = &def.triggers[ti];
        // `+0x2AC`, the trigger's own timer.
        if self.triggers[ti].timer > 0 {
            self.triggers[ti].timer -= 1;
        }
        // The input is read only while the weapon is usable; a disabled weapon sees a released
        // trigger.
        let held = !ctx.disabled
            && match td.input {
                0 => ctx.inp.held,
                1 => ctx.inp.alt_held,
                _ => false,
            };
        {
            // The weapon's analog trigger value, digital here.
            let ts = &mut self.triggers[ti];
            ts.cur = if held { 255 } else { 0 };
        }
        let primary = td.primary_barrel;
        let secondary = td.secondary_barrel;
        match self.triggers[ti].state {
            0 => self.trigger_idle(def, ti, held, primary, secondary, ctx),
            1 => self.trigger_charging(def, ti, held, primary),
            2 => self.trigger_charged(def, ti, held, primary, ctx),
            3 => {
                // Discharging: spew the primary barrel until the timer runs out.
                if self.triggers[ti].timer != 0 {
                    if let Some(p) = primary {
                        if self.can_accept(def, p, ctx) {
                            self.request(p);
                        }
                    }
                } else {
                    self.reset_trigger(ti);
                    self.triggers[ti].armed = false;
                    if let Some(p) = primary {
                        if let Some(st) = self.barrels.get_mut(p) {
                            st.request = false;
                        }
                    }
                }
            }
            _ => {}
        }
        // `weapon + 0x1CD = 0x1CC; 0x1CC = 0`.
        let ts = &mut self.triggers[ti];
        ts.prev = ts.cur;
        ts.cur = 0;
    }

    fn reset_trigger(&mut self, ti: usize) {
        let ts = &mut self.triggers[ti];
        ts.state = 0;
        ts.timer = 0;
    }

    /// `sub_140DF5760`: begin charging.
    fn start_charge(&mut self, def: &WeaponDef, ti: usize) {
        let td = &def.triggers[ti];
        let ts = &mut self.triggers[ti];
        ts.state = 1;
        ts.timer = seconds_to_ticks(td.charging_time);
    }

    /// `sub_140DF5C10`: how charged the trigger is, 0..1.
    pub fn charge_fraction(&self, ti: usize) -> f32 {
        let (Some(td), Some(ts)) = (self.def.triggers.get(ti), self.triggers.get(ti)) else { return 0.0 };
        match ts.state {
            1 if td.charging_time.abs() >= 1.0e-4 => {
                1.0 - (ts.timer as f32 * TICK_SECONDS) / td.charging_time
            }
            2 => 1.0,
            _ => 0.0,
        }
    }

    /// Is the secondary barrel idle (`sub_140DF58C0`)? A trigger with no secondary barrel is.
    fn secondary_idle(&self, def: &WeaponDef, ti: usize) -> bool {
        match def.triggers[ti].secondary_barrel {
            Some(s) if s < self.barrels.len() => self.barrels[s].phase == BarrelPhase::Idle,
            _ => true,
        }
    }

    fn trigger_idle(
        &mut self,
        def: &WeaponDef,
        ti: usize,
        held: bool,
        primary: Option<usize>,
        secondary: Option<usize>,
        ctx: &Ctx,
    ) {
        let td = &def.triggers[ti];
        let Some(behavior) = td.behavior else { return };
        match behavior {
            // spew: ask EVERY tick the button is down and the barrel will take it.
            Behavior::Spew => {
                if held {
                    if let Some(p) = primary {
                        if self.can_accept(def, p, ctx) {
                            self.request(p);
                        }
                    }
                } else if let Some(p) = primary {
                    self.release(p);
                }
            }
            // latch / latch-tether / latch-zoom (we have no zoom, so the unzoomed branch): one
            // request per press; the press stays armed until the barrel can take it, and only
            // a release re-arms it.
            Behavior::Latch | Behavior::LatchTether | Behavior::LatchZoom => {
                if !held {
                    self.triggers[ti].armed = true;
                    if let Some(p) = primary {
                        self.release(p);
                    }
                    if behavior == Behavior::LatchZoom {
                        if let Some(s) = secondary {
                            self.release(s);
                        }
                    }
                } else if self.triggers[ti].armed {
                    if let Some(p) = primary {
                        if self.can_accept(def, p, ctx) {
                            self.request(p);
                            self.triggers[ti].armed = false;
                        }
                    }
                }
            }
            // tap = primary action on release, hold >= `autofire time` = secondary action.
            Behavior::LatchAutofire => self.trigger_autofire(def, ti, held, primary, secondary, ctx),
            // charge / sword-charge: hold to charge, release to discharge.
            Behavior::Charge | Behavior::SwordCharge => {
                if held {
                    if self.secondary_idle(def, ti) && self.triggers[ti].armed {
                        self.start_charge(def, ti);
                        self.triggers[ti].armed = false;
                    } else if !self.triggers[ti].armed {
                        // already charged and fired for this press: nothing to do
                    } else if let Some(p) = primary {
                        if self.can_accept(def, p, ctx) {
                            self.request(p);
                        }
                    }
                } else {
                    self.triggers[ti].armed = true;
                    if let Some(p) = primary {
                        self.release(p);
                    }
                }
            }
            // spew-charge: like charge, and the primary barrel spews while the charge builds.
            Behavior::SpewCharge => {
                if held {
                    if self.secondary_idle(def, ti) && self.triggers[ti].armed {
                        self.start_charge(def, ti);
                        self.triggers[ti].armed = false;
                    }
                } else {
                    self.triggers[ti].armed = true;
                    if let Some(p) = primary {
                        self.release(p);
                    }
                    if let Some(s) = secondary {
                        self.release(s);
                    }
                }
            }
        }
    }

    /// `sub_140DF5970` + the `latch-autofire` arm of `sub_140DF64E0`, for a digital trigger.
    fn trigger_autofire(
        &mut self,
        def: &WeaponDef,
        ti: usize,
        held: bool,
        primary: Option<usize>,
        secondary: Option<usize>,
        ctx: &Ctx,
    ) {
        let td = &def.triggers[ti];
        let need = seconds_to_ticks(td.autofire_time);
        let throw = (td.autofire_throw * 255.0) as i32;
        let (mut take_primary, mut take_secondary, mut fresh) = (false, false, false);
        {
            let ts = &mut self.triggers[ti];
            let (cur, prev) = (ts.cur, ts.prev);
            if cur < prev {
                // released
                if ts.direction == 1 {
                    if prev - ts.reference > 20 {
                        take_primary = true;
                    }
                    ts.direction = 0xFF;
                    ts.reference = prev;
                }
            } else if cur != prev {
                // pressed
                if ts.direction == 0xFF {
                    if ts.reference - prev > 20 {
                        fresh = true;
                    }
                    ts.reference = prev;
                    ts.direction = 1;
                } else if ts.direction == 0 {
                    ts.reference = 0;
                    ts.direction = 1;
                    ts.held_ticks = 0;
                }
            } else if ts.direction == 1 {
                // held steady
                ts.held_ticks = (ts.held_ticks + 1).min(255);
                if ts.held_ticks >= need {
                    let depth = cur - ts.reference;
                    if depth >= throw || cur >= 0xFA {
                        take_secondary = true;
                    } else {
                        take_primary = true;
                    }
                }
            }
        }
        let any_input = held || self.triggers[ti].prev != 0;
        if !any_input {
            // no input at all: arm and let go
            self.triggers[ti].armed = true;
            self.triggers[ti].primary_done = false;
            self.triggers[ti].secondary_done = false;
            if let Some(p) = primary {
                self.release(p);
            }
            if let Some(s) = secondary {
                self.release(s);
            }
            return;
        }
        // a stale action from the previous pull is dropped
        if take_primary && (self.triggers[ti].primary_done || self.triggers[ti].secondary_done) {
            take_primary = false;
        }
        if take_secondary && !(self.triggers[ti].secondary_done || (td.flags & 1 != 0 && self.triggers[ti].primary_done)) {
            // fall through: the secondary action is taken once
        } else if take_secondary {
            take_secondary = false;
        }
        let act = |this: &mut WeaponFire, action: i16, ctx: &Ctx| match action {
            // fire the primary barrel
            0 => {
                if let Some(p) = primary {
                    if this.can_accept(def, p, ctx) {
                        this.request(p);
                    }
                }
            }
            // charge
            1 => {
                if this.secondary_idle(def, ti) {
                    this.start_charge(def, ti);
                }
            }
            // fire the other barrel
            _ => {
                if let Some(s) = secondary {
                    if this.can_accept(def, s, ctx) {
                        this.request(s);
                    }
                }
            }
        };
        if take_primary {
            self.triggers[ti].primary_done = true;
            act(self, td.autofire_primary_action, ctx);
        }
        if take_secondary {
            self.triggers[ti].secondary_done = true;
            act(self, td.autofire_secondary_action, ctx);
        }
        if fresh {
            self.triggers[ti].primary_done = false;
            self.triggers[ti].secondary_done = false;
            if let Some(p) = primary {
                self.release(p);
            }
            if let Some(s) = secondary {
                self.release(s);
            }
        }
    }

    /// Trigger state 1: the charge is building (`sub_140DF64E0` `case 1`).
    fn trigger_charging(&mut self, def: &WeaponDef, ti: usize, held: bool, primary: Option<usize>) {
        let td = &def.triggers[ti];
        if self.triggers[ti].timer == 0 {
            // Fully charged: hold it for `charged time`.
            let ticks = seconds_to_ticks(td.charged_time);
            let ts = &mut self.triggers[ti];
            ts.state = 2;
            ts.timer = ticks;
            return;
        }
        if held {
            // spew-charge spews the primary barrel while the charge builds.
            if td.behavior == Some(Behavior::SpewCharge) {
                if let Some(p) = primary {
                    self.request(p);
                }
            }
            return;
        }
        // Released early. Without `can fire from partial charge` the charge is simply dropped.
        if td.charging_flags & 1 == 0 {
            self.triggers[ti].armed = true;
            self.reset_trigger(ti);
            return;
        }
        if self.charge_fraction(ti) <= 0.0 {
            self.triggers[ti].armed = true;
            self.reset_trigger(ti);
            return;
        }
        self.discharge(def, ti);
    }

    /// Trigger state 2: charged; release (or running out of `charged time`) discharges.
    fn trigger_charged(
        &mut self,
        def: &WeaponDef,
        ti: usize,
        held: bool,
        primary: Option<usize>,
        ctx: &Ctx,
    ) {
        let td = &def.triggers[ti];
        if td.behavior == Some(Behavior::SpewCharge) && held {
            if let Some(p) = primary {
                self.request(p);
            }
        }
        if !held && !ctx.disabled {
            self.discharge(def, ti);
            return;
        }
        if ctx.disabled {
            self.triggers[ti].armed = true;
            self.reset_trigger(ti);
            return;
        }
        if self.triggers[ti].timer == 0 {
            // `overcharged action`: 1 explode, 2 discharge.
            match td.overcharged_action {
                2 => self.discharge(def, ti),
                _ => {}
            }
        }
    }

    /// `sub_140DF5F90`: release the charge. With `fire fractions` it spews the primary barrel for
    /// as many rounds as fractions the charge has reached; otherwise it fires the SECONDARY
    /// barrel once.
    fn discharge(&mut self, def: &WeaponDef, ti: usize) {
        let td = &def.triggers[ti];
        let overcharged = self.triggers[ti].state == 2 && self.triggers[ti].timer == 0;
        if !td.fire_fractions.is_empty() && (!overcharged || td.secondary_barrel.is_none()) {
            let f = self.charge_fraction(ti);
            let owed = td.fire_fractions.iter().filter(|&&x| f >= x).count() as i32;
            let interval = td
                .primary_barrel
                .and_then(|p| def.barrels.get(p))
                .map_or(0.0, |b| rate_interval_ticks(b, 0.0));
            let ts = &mut self.triggers[ti];
            ts.state = 3;
            ts.timer = (interval * owed as f32) as i32 + 1;
            ts.owed = owed;
            return;
        }
        if let Some(s) = td.secondary_barrel {
            self.request(s);
        }
        // `sub_140DF5E70` resets the state and does NOT re-arm: a held button has to come up
        // before it can start another charge.
        self.reset_trigger(ti);
    }

    // ----- barrels (sub_140DEAB40) ------------------------------------------------------------

    fn barrel_tick(&mut self, def: &WeaponDef, bi: usize, ctx: &Ctx, out: &mut Vec<FireEvent>) {
        let b = &def.barrels[bi];
        if self.barrels[bi].timer > 0 {
            self.barrels[bi].timer -= 1;
        }
        let mut cycling = false;
        match self.barrels[bi].phase {
            BarrelPhase::Idle => {
                if !ctx.disabled {
                    if !self.barrels[bi].request {
                        self.auto_reload_check(b, ctx.inp, out);
                    } else if !self.other_barrel_blocks(def, bi) {
                        self.barrels[bi].phase = BarrelPhase::Firing;
                    }
                } else {
                    self.barrels[bi].request = false;
                }
                if self.barrels[bi].phase == BarrelPhase::Firing {
                    cycling = self.firing_step(def, bi, ctx, out);
                }
            }
            BarrelPhase::Firing => cycling = self.firing_step(def, bi, ctx, out),
            BarrelPhase::Locked => {
                if self.barrels[bi].timer == 0 {
                    self.enter_soft(def, bi);
                }
            }
            BarrelPhase::LockedDry => {
                self.auto_reload_check(b, ctx.inp, out);
                if self.barrels[bi].timer == 0 {
                    self.enter_soft(def, bi);
                }
            }
            BarrelPhase::Soft => {
                if self.barrels[bi].timer == 0 {
                    self.end_recovery(def, bi);
                }
            }
        }
        // The cyclic rate winds up while the barrel is cycling and down when it is not -- over a
        // time, or, when the tag names a `rate of fire` FUNCTION, as that function of the heat
        // (`sub_140DEAB40`: `if (def[3] == 0) rof += dt * rate else rof = function(...)`).
        let heat = self.heat;
        let st = &mut self.barrels[bi];
        st.rof = if cycling {
            match &b.rate_of_fire_acceleration_fn {
                Some(c) => sample_curve(c, heat).clamp(0.0, 1.0),
                None => (st.rof + TICK_SECONDS * b.acceleration_rate).clamp(0.0, 1.0),
            }
        } else if st.rof > 0.0 {
            match &b.rate_of_fire_deceleration_fn {
                Some(c) => sample_curve(c, heat).clamp(0.0, 1.0),
                None => (st.rof - TICK_SECONDS * b.deceleration_rate).clamp(0.0, 1.0),
            }
        } else {
            st.rof
        };
        if st.error > 0.0 {
            st.error = (st.error - TICK_SECONDS * b.error_deceleration_rate).max(0.0);
        }
    }

    /// `flags & 0x80 / 0x100` on a two-barrel weapon: do not fire while the OTHER barrel is
    /// firing / in the soft tail of its recovery.
    fn other_barrel_blocks(&self, def: &WeaponDef, bi: usize) -> bool {
        let b = &def.barrels[bi];
        if (b.flags & BF_CANT_FIRE_WHEN_OTHERS_FIRING == 0 && b.flags & BF_CANT_FIRE_WHEN_OTHERS_RECOVERING == 0)
            || def.barrels.len() <= 1
        {
            return false;
        }
        let other = &self.barrels[usize::from(bi == 0)];
        (other.phase == BarrelPhase::Firing && b.flags & BF_CANT_FIRE_WHEN_OTHERS_FIRING != 0)
            || (other.phase == BarrelPhase::Soft && b.flags & BF_CANT_FIRE_WHEN_OTHERS_RECOVERING != 0)
    }

    /// `sub_140DE37A0`: may this barrel fire a round NOW, and what should happen if not?
    /// Returns `(ready, a3, a4, a5)` exactly as the engine's out parameters.
    fn decide(&self, b: &BarrelDef, st: &BarrelState) -> (bool, bool, bool, bool) {
        let interval = rate_interval_ticks(b, st.rof);
        let mut ready = st.shots_fired == 0 || st.counter as f32 + 1.0 >= interval;
        let (min, max) = (b.shots_per_fire[0] as i32, b.shots_per_fire[1] as i32);
        let (a3, a4);
        if min != 0 && st.shots_fired < min {
            // inside a mandatory burst: carry on whatever the trigger does
            a3 = false;
            a4 = false;
        } else if max == 0 || st.shots_fired < max {
            a3 = true;
            a4 = false;
        } else {
            // the string reached its cap
            a3 = false;
            a4 = true;
            ready = false;
        }
        if st.empty_clicked {
            return (false, a3, true, true);
        }
        (ready, a3, a4, false)
    }

    /// The `case 1` arm: returns whether the barrel is cycling (for the rate-of-fire wind-up).
    fn firing_step(&mut self, def: &WeaponDef, bi: usize, ctx: &Ctx, out: &mut Vec<FireEvent>) -> bool {
        if ctx.disabled {
            self.enter_recovery(def, bi, false);
            return false;
        }
        let b = &def.barrels[bi];
        let (ready, a3, a4, a5) = self.decide(b, &self.barrels[bi]);
        let request = self.barrels[bi].request;
        if ready {
            if !a3 || request {
                self.barrels[bi].request = false;
                self.fire(def, bi, ctx, out);
                return true;
            }
        } else if !a4 && (!a3 || request) {
            let st = &mut self.barrels[bi];
            st.counter = (st.counter + 1).min(127);
            return true;
        }
        self.enter_recovery(def, bi, a5);
        false
    }

    /// `sub_140DEA820`: the string is over -- lock for the recovery.
    fn enter_recovery(&mut self, def: &WeaponDef, bi: usize, dry: bool) {
        let (_, locked) = recovery_ticks(&def.barrels[bi]);
        let st = &mut self.barrels[bi];
        st.phase = if dry { BarrelPhase::LockedDry } else { BarrelPhase::Locked };
        st.timer = locked;
    }

    /// `sub_140DE85F0`: the locked part ran out -- the soft tail.
    fn enter_soft(&mut self, def: &WeaponDef, bi: usize) {
        let (total, locked) = recovery_ticks(&def.barrels[bi]);
        let st = &mut self.barrels[bi];
        st.phase = BarrelPhase::Soft;
        st.timer = total - locked;
        if st.timer == 0 {
            self.end_recovery(def, bi);
        }
    }

    /// `sub_140DE9EA0`: back to idle.
    fn end_recovery(&mut self, def: &WeaponDef, bi: usize) {
        let b = &def.barrels[bi];
        let st = &mut self.barrels[bi];
        st.phase = BarrelPhase::Idle;
        st.counter = 0;
        st.shots_fired = 0;
        if b.flags & BF_DONT_CLEAR_FIRE_BIT_AFTER_RECOVERING == 0 && !st.carry {
            st.request = false;
        }
        st.carry = false;
    }

    // ----- one round (sub_140DE86B0 + weapon_barrel_create_projectiles) ------------------------

    fn fire(&mut self, def: &WeaponDef, bi: usize, ctx: &Ctx, out: &mut Vec<FireEvent>) {
        let b = &def.barrels[bi];
        let needed = b.rounds_per_shot as i32;
        if !self.ammo_ok(b, needed) {
            let st = &mut self.barrels[bi];
            st.empty_clicked = true;
            st.counter = 0;
            self.stats.empty_clicks += 1;
            out.push(FireEvent::EmptyClick { barrel: bi });
            return;
        }
        // `sub_140DE9790`: spend the rounds (never below zero).
        let mut spent = needed;
        if let Some(m) = b.magazine {
            if let Some(l) = self.loaded.get_mut(m) {
                spent = needed.min(*l).max(0);
                *l = (*l - needed).max(0);
            }
        }
        // Heat, clamped to 1; the amount is a function of the heat when the tag says so.
        let per_round = b.heat_per_round_fn.map_or(b.heat_per_round, |c| sample_curve(&c, self.heat));
        self.heat = (self.heat + per_round).min(1.0);
        // The battery: `age generated per round` (a burst-flagged barrel pays once, up front).
        if self.def.flags & WF_AGES_WITH_EACH_KILL == 0 {
            let mut a = b.age_per_round;
            if b.flags & BF_USE_ONE_FIRING_EFFECT_PER_BURST != 0 {
                a = if self.barrels[bi].shots_fired != 0 { 0.0 } else { a * b.shots_per_fire[0] as f32 };
            }
            self.age = (self.age + a).min(1.0);
        }
        let n = if b.projectile.is_some() { b.projectiles_per_shot.max(0) as usize } else { 0 };
        // The error the barrel has NOW decides this round's cone; the round then adds to it.
        let err = self.barrels[bi].error.clamp(0.0, 1.0);
        let outer = b.error_angle[0] + (b.error_angle[1] - b.error_angle[0]) * err;
        let inner = b.minimum_error.min(outer);
        let mut projectiles = Vec::with_capacity(n);
        for k in 0..n {
            // Area-uniform in the annulus between the inner and outer cone.
            let u = self.rand();
            let angle = (inner * inner + (outer * outer - inner * inner) * u).max(0.0).sqrt();
            let azimuth = self.rand() * std::f32::consts::TAU;
            // `horizontal fan`: i-th of n projectiles sits at 0, +-1, +-2 ... (odd n) or
            // +-0.5, +-1.5 ... (even n) distribution angles off the aim.
            let fan_yaw = if b.distribution_function == 1 { fan_offset(k, n) * b.distribution_angle } else { 0.0 };
            projectiles.push(Spread { angle, azimuth, fan_yaw });
        }
        // `sub_140DE2A60`: the round adds error from the barrel's firing-penalty curve.
        let curve = if ctx.inp.crouched { b.firing_crouched_penalty.or(b.firing_penalty) } else { b.firing_penalty };
        if let Some(c) = curve {
            let add = sample_curve(&c, err);
            let st = &mut self.barrels[bi];
            st.error += add;
        }
        let st = &mut self.barrels[bi];
        st.shots_fired += 1;
        st.counter = 0;
        self.stats.shots += 1;
        self.stats.rounds += spent as u64;
        self.stats.projectiles += n as u64;
        out.push(FireEvent::Shot { barrel: bi, rounds: spent, projectiles });
    }
}

/// `weapon_barrel_create_projectiles`: the position of projectile `i` of `n` in a fan, in units of
/// the barrel's `distribution angle`. Even counts straddle the aim (-0.5, +0.5, -1.5, +1.5 ...),
/// odd counts put one on it (0, -1, +1, -2, +2 ...).
///
/// INFERRED in its ORDER only: the decompiled index arithmetic is lossy (it reads as the same
/// side twice), but the set it builds is symmetric about the aim and grows outward, which is what
/// is kept; every shipped barrel launches one projectile, so no tag exercises this today.
pub fn fan_offset(i: usize, n: usize) -> f32 {
    if n % 2 == 0 {
        let mag = (i / 2) as f32 + 0.5;
        if i % 2 == 0 { -mag } else { mag }
    } else if i == 0 {
        0.0
    } else {
        let mag = ((i - 1) / 2 + 1) as f32;
        if (i - 1) % 2 == 0 { -mag } else { mag }
    }
}

/// #weapon-fire  One projectile's direction: the aim, tilted off axis by its error-cone `angle` at
/// `azimuth` around it, then swung about the aim's up axis by its `horizontal fan` yaw.
pub fn spread_dir(aim: glam::Vec3, s: &Spread) -> glam::Vec3 {
    let right = {
        let r = aim.cross(glam::Vec3::Z);
        if r.length_squared() < 1.0e-8 { glam::Vec3::X } else { r.normalize() }
    };
    let up = right.cross(aim);
    let (sa, ca) = s.angle.sin_cos();
    let mut d = aim * ca + (right * s.azimuth.cos() + up * s.azimuth.sin()) * sa;
    if s.fan_yaw != 0.0 {
        d = glam::Quat::from_axis_angle(up, s.fan_yaw) * d;
    }
    d.normalize_or_zero()
}


fn sample_curve(c: &[f32; 11], x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0) * 10.0;
    let i = (t.floor() as usize).min(9);
    c[i] + (c[i + 1] - c[i]) * (t - i as f32)
}

/// One line saying what this weapon's triggers and barrels will do, for the notes / caption: the
/// authored numbers AND the tick counts they become.
pub fn describe(def: &WeaponDef) -> String {
    let mut parts = Vec::new();
    for (i, t) in def.triggers.iter().enumerate() {
        let Some(b) = t.primary_barrel.and_then(|p| def.barrels.get(p)) else { continue };
        let (total, locked) = recovery_ticks(b);
        parts.push(format!(
            "trigger {i} {} -> {:.1} rps ({:.1}-tick interval), burst {}..{}, recovery {:.3} s = {total} ticks ({locked} locked), \
             {} round(s)/shot, {} projectile(s)/shot",
            t.behavior.map_or("?", |b| b.word()),
            b.rounds_per_second[0],
            rate_interval_ticks(b, 0.0),
            b.shots_per_fire[0],
            b.shots_per_fire[1],
            b.fire_recovery_time,
            b.rounds_per_shot,
            b.projectiles_per_shot,
        ));
    }
    if parts.is_empty() {
        "no fireable trigger".to_string()
    } else {
        parts.join(" | ")
    }
}

// ---------------------------------------------------------------------------------------------
// gates
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod gates {
    //! #weapon-fire  **The machine, driven with the trigger and checked against closed-form
    //! expectations written out in each test.** Nothing here calls the function it is checking to
    //! produce its own expected value: every expectation is spelled out from the authored
    //! numbers (a 10 rps weapon is 60/10 = 6 ticks; a 0.33 s / 0.2 soft recovery is
    //! `2*floor(9.9) + 2 = 20` ticks) and the machine has to land on it.
    use super::*;

    /// A barrel the way the cache bakes one: runtime rates at 30 (a time of 0 bakes to 30), a
    /// projectile, one of everything.
    fn barrel() -> BarrelDef {
        BarrelDef {
            rounds_per_second: [10.0, 10.0],
            acceleration_rate: 30.0,
            deceleration_rate: 30.0,
            error_deceleration_rate: 30.0,
            rounds_per_shot: 1,
            projectiles_per_shot: 1,
            projectile: Some(1),
            ..Default::default()
        }
    }

    fn trigger(behavior: Behavior) -> TriggerDef {
        TriggerDef { behavior: Some(behavior), primary_barrel: Some(0), ..Default::default() }
    }

    fn weapon(behavior: Behavior, b: BarrelDef) -> WeaponFire {
        let mut b = b;
        if b.magazine.is_some() {
            b.rounds_per_shot = b.rounds_per_shot.max(1);
        }
        let mag = if b.magazine.is_some() { 32 } else { 0 };
        WeaponFire::new(
            Arc::new(WeaponDef { triggers: vec![trigger(behavior)], barrels: vec![b], overheated_threshold: 1.0, ..Default::default() }),
            vec![mag, 0],
        )
    }

    fn held(on: bool) -> FireInput {
        FireInput { held: on, usable: true, has_reserve: true, ..Default::default() }
    }

    /// Tick `n` times with the button in state `on`; the tick index of every `Shot` (counted from
    /// `*t`, which advances).
    fn run(wf: &mut WeaponFire, t: &mut i64, n: i64, on: bool, shots: &mut Vec<i64>, clicks: &mut Vec<i64>) {
        for _ in 0..n {
            let mut out = Vec::new();
            wf.tick(&held(on), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => shots.push(*t),
                    FireEvent::EmptyClick { .. } => clicks.push(*t),
                    _ => {}
                }
            }
            *t += 1;
        }
    }

    fn hold_for(wf: &mut WeaponFire, ticks: i64) -> (Vec<i64>, Vec<i64>) {
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        run(wf, &mut t, ticks, true, &mut s, &mut c);
        (s, c)
    }

    fn gaps(v: &[i64]) -> Vec<i64> {
        v.windows(2).map(|w| w[1] - w[0]).collect()
    }

    // ---- the arithmetic the rest stands on --------------------------------------------------
    /// The machines tick at the SAME rate the rest of the simulation does (Havok's one step per tick,
    /// the character controller): a different constant here would make every recovery the wrong
    /// length in seconds.
    #[test]
    fn the_tick_rate_is_the_simulations_tick_rate() {
        assert_eq!(TICK_HZ as f32, crate::physics_settle::TICK_HZ);
        assert_eq!(GRID_STEP, 2);
    }


    /// The 30 Hz grid on a 60 Hz tick. Hand-worked from `sub_140DE3F80` for three real recoveries:
    /// DMR 0.33 s soft 0.2: `rec = 9.9` grid steps, `locked = 2*floor(0.8*9.9) = 14`,
    /// `total = 2*floor(9.9) + 0 + 2 = 20`. Magnum 0.19 s soft 0.5: `rec = 5.7`, `locked =
    /// 2*floor(2.85) = 4`, `total = 2*5 + 2 = 12`. Assault rifle 0.05 s soft 0: `rec = 1.5 < 2`
    /// so it floors to 2, `locked = 2*2 = 4`, `total = 2*2 + 2 = 6`.
    #[test]
    fn recovery_is_converted_on_the_30hz_grid() {
        let rec = |t: f32, soft: f32| recovery_ticks(&BarrelDef { fire_recovery_time: t, soft_recovery_fraction: soft, ..Default::default() });
        assert_eq!(rec(0.33, 0.2), (20, 14), "DMR");
        assert_eq!(rec(0.19, 0.5), (12, 4), "magnum");
        assert_eq!(rec(0.05, 0.0), (6, 4), "assault rifle: below one grid step floors to two ticks");
        // Nothing locked -> the engine adds one more tick.
        assert_eq!(rec(0.2, 1.0), (2 * 6 + 1 + 2, 0), "a fully soft recovery has no locked part");
    }

    /// `30 / ceil(30 / rps)` rounds a rate UP to the next grid-aligned interval: 10 rps is exactly
    /// 6 ticks, but 7.5 rps is 4 grid steps = 8 ticks, 7 rps is 5 steps = 10 ticks.
    #[test]
    fn rates_snap_to_the_grid() {
        let iv = |rps: f32| rate_interval_ticks(&BarrelDef { rounds_per_second: [rps, rps], ..Default::default() }, 0.0);
        assert_eq!(iv(10.0), 6.0);
        assert_eq!(iv(7.5), 8.0);
        assert_eq!(iv(7.0), 10.0);
        assert_eq!(iv(3.0), 20.0);
        assert_eq!(iv(30.0), 2.0);
        assert_eq!(iv(0.0), 0.0, "no rate authored");
    }

    // ---- automatics ---------------------------------------------------------------------------

    /// THE REPORT: hold the trigger and the assault rifle keeps firing. 3 s at 10 rps is 30 rounds,
    /// one every 6 ticks, the first on the first tick.
    #[test]
    fn a_held_automatic_fires_at_its_authored_rate() {
        let mut wf = weapon(Behavior::Spew, barrel());
        let (shots, _) = hold_for(&mut wf, 180);
        assert_eq!(shots.len(), 30, "3 s at 10 rps");
        assert_eq!(shots[0], 0, "the first round leaves on the first tick");
        assert!(gaps(&shots).iter().all(|&g| g == 6), "every round is exactly 6 ticks after the last: {:?}", gaps(&shots));
    }

    /// An off-grid rate is the GRID's, not the tag's: 7.5 rps fires every 8 ticks (7.5/s exactly,
    /// as it happens), 7 rps every 10 (6/s), 11 rps every 6 (10/s).
    #[test]
    fn an_off_grid_rate_is_snapped_up() {
        // 11 rps -> ceil(30/11) = 3 steps -> 10 rps -> 6 ticks; 15 rps is on the grid (4 ticks)
        for (rps, gap) in [(7.5f32, 8i64), (7.0, 10), (11.0, 6), (15.0, 4)] {
            let mut wf = weapon(Behavior::Spew, BarrelDef { rounds_per_second: [rps, rps], ..barrel() });
            let (shots, _) = hold_for(&mut wf, 180);
            assert!(gaps(&shots).iter().all(|&g| g == gap), "{rps} rps should fire every {gap} ticks: {:?}", gaps(&shots));
            assert_eq!(shots.len() as i64, (179 / gap) + 1);
        }
    }

    /// Letting go ends the string: the round in flight completes and nothing follows it. Pressing
    /// again before the recovery has run its course does not beat it.
    #[test]
    fn releasing_the_trigger_stops_the_string_and_recovery_gates_the_next_one() {
        // Assault-rifle recovery: total 6, locked 4.
        let b = BarrelDef { fire_recovery_time: 0.05, ..barrel() };
        let mut wf = weapon(Behavior::Spew, b);
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        run(&mut wf, &mut t, 20, true, &mut s, &mut c);
        let before = s.len();
        run(&mut wf, &mut t, 60, false, &mut s, &mut c);
        // The request bit set by the last tick of holding is still pending when the button comes
        // up, so ONE more round can follow (the engine does not clear it on release) -- and then
        // the string is over.
        assert!(s.len() <= before + 1, "at most the already-requested round follows a release: {s:?}");
        assert!(*s.last().unwrap() < 20 + 6, "and it is the very next interval, not a runaway: {s:?}");
        // pull again once the recovery is long over: fires on the first tick
        let again = t;
        run(&mut wf, &mut t, 1, true, &mut s, &mut c);
        assert_eq!(s.last().copied(), Some(again), "a rested barrel fires on the first tick");
        // release for ONE tick and pull again at once: the barrel is in its recovery
        run(&mut wf, &mut t, 1, false, &mut s, &mut c);
        let n = s.len();
        let pull = t;
        run(&mut wf, &mut t, 12, true, &mut s, &mut c);
        let next = *s.get(n).expect("it fires again once the recovery is over");
        assert!(next - pull >= 4, "the second pull waited out the recovery (fired {} ticks later)", next - pull);
    }

    /// `rounds per second` initial -> final over `acceleration time` (the Warthog chaingun, 5 -> 15
    /// rps over one second): the interval shrinks while the trigger is held and ends at the final
    /// rate.
    #[test]
    fn the_cyclic_rate_winds_up_while_held() {
        let b = BarrelDef { rounds_per_second: [5.0, 15.0], acceleration_rate: 1.0, deceleration_rate: 3.0, ..barrel() };
        let mut wf = weapon(Behavior::Spew, b);
        let (shots, _) = hold_for(&mut wf, 60 * 4);
        let g = gaps(&shots);
        assert!((10..=12).contains(&g[0]), "starts near 5 rps (12 ticks), already winding up: {}", g[0]);
        assert_eq!(*g.last().unwrap(), 4, "ends at 15 rps = 4 ticks");
        assert!(g.windows(2).all(|w| w[1] <= w[0]), "it only ever speeds up while held: {g:?}");
    }

    // ---- semi-automatics ----------------------------------------------------------------------

    fn dmr() -> BarrelDef {
        // rec 0.33 soft 0.2 -> total 20 / locked 14; one round per fire; flags: others-firing,
        // others-recovering, don't clear the fire bit after recovering.
        BarrelDef {
            rounds_per_second: [3.0, 3.0],
            shots_per_fire: [1, 1],
            fire_recovery_time: 0.33,
            soft_recovery_fraction: 0.2,
            flags: 0x380,
            ..barrel()
        }
    }

    /// A `latch` fires ONCE per press however long the button is held.
    #[test]
    fn a_semi_automatic_fires_once_per_press() {
        let mut wf = weapon(Behavior::Latch, dmr());
        let (shots, _) = hold_for(&mut wf, 600);
        assert_eq!(shots, vec![0], "ten seconds of holding is one round");
    }

    /// ...and a fast mash cannot beat the recovery: a press every 4 ticks for 6 s gives a round
    /// every `total + 2 = 22` ticks at the very best, never closer.
    #[test]
    fn mashing_a_semi_automatic_cannot_beat_its_recovery() {
        let mut wf = weapon(Behavior::Latch, dmr());
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        for _ in 0..90 {
            run(&mut wf, &mut t, 2, true, &mut s, &mut c);
            run(&mut wf, &mut t, 2, false, &mut s, &mut c);
        }
        assert!(s.len() >= 14, "a mash still fires about as fast as the recovery allows ({} rounds)", s.len());
        assert!(
            gaps(&s).iter().all(|&g| g >= 22),
            "no two rounds closer than the recovery (20 ticks + 2): {:?}",
            gaps(&s)
        );
        // 360 ticks / 22 = 16.4: at most 17 rounds
        assert!(s.len() <= 17, "{} rounds in 6 s is faster than a 0.33 s recovery allows", s.len());
    }

    /// The recovery IS a semi-auto's rate: a pull-and-release every 30 ticks fires every pull.
    #[test]
    fn a_patient_tap_fires_every_time() {
        let mut wf = weapon(Behavior::Latch, dmr());
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        for _ in 0..6 {
            run(&mut wf, &mut t, 1, true, &mut s, &mut c);
            run(&mut wf, &mut t, 29, false, &mut s, &mut c);
        }
        assert_eq!(s.len(), 6);
    }

    /// The BUTTON BUFFER. A press made while the barrel is still recovering is accepted in the
    /// soft tail (`sub_140DEA9C0`) but the barrel throws its fire bit away when the recovery ends
    /// (`sub_140DE9EA0`) UNLESS the request landed in the last tick or the barrel carries `don't
    /// clear fire bit after recovering`. So the same held press is lost on one weapon and fires the
    /// instant the recovery ends on another -- which is the difference between the rocket
    /// launcher (0x180) and the DMR (0x380) in the cache.
    #[test]
    fn a_press_made_during_the_recovery_fires_only_where_the_fire_bit_is_kept() {
        let base = BarrelDef { shots_per_fire: [1, 1], fire_recovery_time: 0.33, soft_recovery_fraction: 0.2, rounds_per_second: [3.0, 3.0], ..barrel() };
        let shots = |flags: u32| -> Vec<i64> {
            let mut wf = weapon(Behavior::Latch, BarrelDef { flags, ..base.clone() });
            let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0i64);
            run(&mut wf, &mut t, 1, true, &mut s, &mut c); // shot at 0
            run(&mut wf, &mut t, 2, false, &mut s, &mut c); // re-arm
            run(&mut wf, &mut t, 60, true, &mut s, &mut c); // press again, INSIDE the recovery, and hold
            s
        };
        assert_eq!(shots(0), vec![0], "no fire bit kept: the early press is dropped when the recovery ends");
        let kept = shots(BF_DONT_CLEAR_FIRE_BIT_AFTER_RECOVERING);
        assert_eq!(kept.len(), 2, "fire bit kept: the held press fires the moment the recovery ends: {kept:?}");
        assert_eq!(kept[1] - kept[0], 22, "recovery 20 ticks + 2: {kept:?}");
        // The exception that proves the rule: without the flag a request that lands in the LAST
        // tick of the soft tail still survives it. A two-tick tap at tick 20 is lost; at 21 it fires
        // the moment the recovery ends (tick 22).
        let tap = |at: i64| -> Vec<i64> {
            let mut wf = weapon(Behavior::Latch, base.clone());
            let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0i64);
            run(&mut wf, &mut t, 1, true, &mut s, &mut c);
            run(&mut wf, &mut t, at - 1, false, &mut s, &mut c);
            run(&mut wf, &mut t, 2, true, &mut s, &mut c);
            run(&mut wf, &mut t, 40, false, &mut s, &mut c);
            s
        };
        assert_eq!(tap(20), vec![0], "a tap one tick too early is dropped");
        assert_eq!(tap(21), vec![0, 22], "a tap in the last tick is buffered");
    }

    // ---- bursts -------------------------------------------------------------------------------

    /// `shots per fire` [3, 3] at 15 rps: ONE tap fires exactly three rounds, 4 ticks apart, then
    /// the recovery. (The battle rifle shape; not in this cache but the machine must do it.)
    #[test]
    fn a_burst_weapon_fires_its_burst_from_one_tap() {
        let b = BarrelDef { rounds_per_second: [15.0, 15.0], shots_per_fire: [3, 3], fire_recovery_time: 0.5, ..barrel() };
        let mut wf = weapon(Behavior::Latch, b);
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        run(&mut wf, &mut t, 1, true, &mut s, &mut c);
        run(&mut wf, &mut t, 100, false, &mut s, &mut c);
        assert_eq!(s, vec![0, 4, 8], "three rounds, 60/15 = 4 ticks apart, from a one-tick tap");
    }

    // ---- ammunition ---------------------------------------------------------------------------

    fn rifle() -> BarrelDef {
        BarrelDef { magazine: Some(0), fire_recovery_time: 0.05, ..barrel() }
    }

    /// The magazine empties after exactly its loaded rounds, then ONE dry click, then nothing
    /// (however long it is held), and the engine starts a reload.
    #[test]
    fn the_magazine_empties_and_the_barrel_clicks_dry() {
        let mut wf = weapon(Behavior::Spew, rifle());
        let mut reload = 0;
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0i64);
        for _ in 0..(60 * 10) {
            let mut out = Vec::new();
            wf.tick(&held(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => s.push(t),
                    FireEvent::EmptyClick { .. } => c.push(t),
                    FireEvent::NeedReload => reload += 1,
                    _ => {}
                }
            }
            t += 1;
        }
        assert_eq!(s.len(), 32, "a 32-round magazine is 32 rounds");
        assert_eq!(wf.loaded[0], 0);
        assert_eq!(c.len(), 1, "one click, not a stream of them: {c:?}");
        assert_eq!(c[0], s[31] + 6, "the click is the pull after the last round");
        assert!(reload > 0, "an empty magazine asks for a reload");
        // letting go and pulling again clicks again
        let (mut s2, mut c2) = (Vec::new(), Vec::new());
        run(&mut wf, &mut t, 30, false, &mut s2, &mut c2);
        run(&mut wf, &mut t, 30, true, &mut s2, &mut c2);
        assert!((s2.is_empty(), c2.len()) == (true, 1), "a fresh pull on an empty gun clicks once: {s2:?} {c2:?}");
    }

    /// A shot that costs two rounds empties a 7-round magazine in 3 shots and leaves the odd
    /// round unused (`loaded < rounds per shot`) -- the dry click comes with one round still in.
    #[test]
    fn rounds_per_shot_is_spent_per_shot() {
        let b = BarrelDef { rounds_per_shot: 2, ..rifle() };
        let mut wf = weapon(Behavior::Spew, b);
        wf.loaded[0] = 7;
        let (s, c) = hold_for(&mut wf, 60 * 4);
        assert_eq!(s.len(), 3);
        assert_eq!(wf.loaded[0], 1, "7 - 3*2");
        assert_eq!(c.len(), 1);
    }

    /// `can fire with partial ammo` lets that last odd round go (and it does not go below zero).
    #[test]
    fn partial_ammo_flag_spends_the_last_round() {
        let b = BarrelDef { rounds_per_shot: 2, flags: BF_CAN_FIRE_WITH_PARTIAL_AMMO, minimum_rounds_loaded: 1, ..rifle() };
        let mut wf = weapon(Behavior::Spew, b);
        wf.loaded[0] = 7;
        let (s, _) = hold_for(&mut wf, 60 * 4);
        assert_eq!(s.len(), 4, "3 full shots and the 1-round remainder");
        assert_eq!(wf.loaded[0], 0);
    }

    /// While the HUD reloads, a pull is refused outright (no round, no click).
    #[test]
    fn a_pull_during_a_reload_does_nothing() {
        let mut wf = weapon(Behavior::Spew, rifle());
        wf.loaded[0] = 0;
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0i64);
        let inp = FireInput { held: true, usable: true, reloading: true, has_reserve: true, ..Default::default() };
        for _ in 0..120 {
            let mut out = Vec::new();
            wf.tick(&inp, &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => s.push(t),
                    FireEvent::EmptyClick { .. } => c.push(t),
                    _ => {}
                }
            }
            t += 1;
        }
        assert!(s.is_empty() && c.is_empty(), "shots {s:?} clicks {c:?}");
    }

    // ---- what one shot produces ---------------------------------------------------------------

    /// `projectiles per shot` projectiles leave in ONE fire event, at one round's cost.
    #[test]
    fn projectiles_per_shot_launch_together_for_one_rounds_cost() {
        let b = BarrelDef { projectiles_per_shot: 12, fire_recovery_time: 1.0, ..rifle() };
        let mut wf = weapon(Behavior::Latch, b);
        let mut out = Vec::new();
        wf.tick(&held(true), &mut out);
        let shots: Vec<&FireEvent> = out.iter().filter(|e| matches!(e, FireEvent::Shot { .. })).collect();
        assert_eq!(shots.len(), 1, "one fire event");
        let FireEvent::Shot { projectiles, rounds, .. } = shots[0] else { unreachable!() };
        assert_eq!(projectiles.len(), 12);
        assert_eq!(*rounds, 1, "twelve pellets, one round");
        assert_eq!(wf.stats.projectiles, 12);
        assert_eq!(wf.loaded[0], 31);
    }

    /// A barrel with no `projectile` (a horn) launches none but still makes the event.
    #[test]
    fn a_barrel_with_no_projectile_launches_none() {
        let b = BarrelDef { projectile: None, projectiles_per_shot: 5, ..barrel() };
        let mut wf = weapon(Behavior::Latch, b);
        let mut out = Vec::new();
        wf.tick(&held(true), &mut out);
        assert!(matches!(&out[0], FireEvent::Shot { projectiles, .. } if projectiles.is_empty()));
    }

    /// The horizontal fan: odd counts put one on the aim and the rest at +-1, +-2 ... steps; even
    /// counts straddle it at +-0.5, +-1.5 ...
    #[test]
    fn the_fan_distribution_is_symmetric_about_the_aim() {
        let odd: Vec<f32> = (0..5).map(|i| fan_offset(i, 5)).collect();
        assert_eq!(odd, vec![0.0, -1.0, 1.0, -2.0, 2.0]);
        let mut even: Vec<f32> = (0..4).map(|i| fan_offset(i, 4)).collect();
        even.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(even, vec![-1.5, -0.5, 0.5, 1.5]);
        // and the distribution function turns it on
        let b = BarrelDef { projectiles_per_shot: 3, distribution_function: 1, distribution_angle: 0.1, ..barrel() };
        let mut wf = weapon(Behavior::Latch, b);
        let mut out = Vec::new();
        wf.tick(&held(true), &mut out);
        let FireEvent::Shot { projectiles, .. } = &out[0] else { panic!() };
        let mut yaws: Vec<f32> = projectiles.iter().map(|p| p.fan_yaw).collect();
        yaws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(yaws.iter().map(|y| (y * 10.0).round() as i32).collect::<Vec<_>>(), vec![-1, 0, 1]);
    }

    /// Error: a round's cone lies between `minimum error` and the `error angle` for the error the
    /// barrel carries, each round ADDS error from the firing-penalty curve, and it decays again.
    #[test]
    fn firing_error_grows_with_each_round_and_decays() {
        let curve = [0.2f32; 11]; // each round adds 0.2 whatever the error is
        let b = BarrelDef {
            minimum_error: 0.001,
            error_angle: [0.002, 0.04],
            firing_penalty: Some(curve),
            error_deceleration_rate: 0.5,
            ..barrel()
        };
        let mut wf = weapon(Behavior::Spew, b);
        let mut angles = Vec::new();
        let mut errs = Vec::new();
        for _ in 0..(6 * 8) {
            let mut out = Vec::new();
            wf.tick(&held(true), &mut out);
            for e in out {
                if let FireEvent::Shot { projectiles, .. } = e {
                    angles.push(projectiles[0].angle);
                    errs.push(wf.barrels[0].error);
                }
            }
        }
        assert!(angles.iter().all(|&a| (0.001..=0.04 + 1e-6).contains(&a)), "{angles:?}");
        assert!(errs.windows(2).take(3).all(|w| w[1] > w[0]), "error climbs while firing: {errs:?}");
        // let go: it decays at 0.5/s -> a second later it has lost 0.5 (the round already
        // requested may add 0.2 once more, and it is then also decaying)
        let (mut s, mut c, mut t) = (Vec::new(), Vec::new(), 0);
        let peak = wf.barrels[0].error;
        run(&mut wf, &mut t, 60, false, &mut s, &mut c);
        let after = wf.barrels[0].error;
        let late = s.len() as f32 * 0.2;
        assert!((peak + late - after - 0.5).abs() < 0.03, "peak {peak} -> {after} (+{late} from {} late round)", s.len());
    }

    // ---- heat ---------------------------------------------------------------------------------

    fn plasma_rifle() -> WeaponFire {
        // heat 0.15 per round, overheats at 1.0, back at 0.1, cools 0.853/s (0.35/s once hot).
        let b = BarrelDef { heat_per_round: 0.15, rounds_per_second: [15.0, 15.0], ..barrel() };
        WeaponFire::new(
            Arc::new(WeaponDef {
                triggers: vec![trigger(Behavior::Spew)],
                barrels: vec![b],
                overheated_threshold: 1.0,
                heat_recovery_threshold: 0.1,
                heat_loss_per_second: 0.853,
                overheated_heat_loss_per_second: 0.35,
                ..Default::default()
            }),
            vec![0, 0],
        )
    }

    /// Heat adds up, the weapon overheats at the threshold, refuses to fire while it is hot, and
    /// comes back only once it has cooled below the recovery threshold.
    #[test]
    fn a_weapon_overheats_and_comes_back() {
        let mut wf = plasma_rifle();
        let (mut shots, mut over, mut cooled, mut t) = (Vec::new(), None, None, 0i64);
        for _ in 0..(60 * 12) {
            let mut out = Vec::new();
            wf.tick(&held(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => shots.push(t),
                    FireEvent::Overheated => over = over.or(Some(t)),
                    FireEvent::Cooled => cooled = cooled.or(Some(t)),
                    _ => {}
                }
            }
            t += 1;
        }
        // 0.15 per round at 15 rps, minus the 0.853/s lost between rounds: about the 11th round
        // crosses 1.0.
        let (over, cooled) = (over.expect("it overheated"), cooled.expect("it cooled"));
        let n_before = shots.iter().filter(|&&s| s <= over).count();
        assert!((7..=14).contains(&n_before), "{n_before} rounds before it overheated");
        assert!(shots.iter().all(|&s| s <= over || s > cooled), "no round while it was hot: over {over} cooled {cooled} shots {shots:?}");
        // from ~1.0 down to 0.1 at 0.35/s is 2.57 s = 154 ticks
        let hot = cooled - over;
        assert!((150..=160).contains(&hot), "{hot} ticks hot, expected ~154");
        assert!(shots.iter().any(|&s| s > cooled), "it fires again after cooling");
    }

    /// `never overheats` keeps the weapon firing at full heat.
    #[test]
    fn the_never_overheats_flag_is_honoured() {
        let mut wf = plasma_rifle();
        let mut d = (*wf.def).clone();
        d.secondary_flags |= WSF_NEVER_OVERHEATS;
        wf.def = Arc::new(d);
        let (shots, _) = hold_for(&mut wf, 60 * 12);
        assert!(!wf.overheated && wf.heat > 0.0);
        assert!(shots.len() > 60, "{} rounds", shots.len());
    }

    /// A `rate of fire` FUNCTION replaces the time ramp: the interval follows the heat, not the
    /// clock (the plasma repeater's 15 -> 3 rps as it warms).
    #[test]
    fn a_rate_of_fire_function_follows_the_heat_not_the_clock() {
        let curve: [f32; 11] = std::array::from_fn(|i| i as f32 / 10.0);
        let b = BarrelDef {
            rounds_per_second: [15.0, 3.0],
            rate_of_fire_acceleration_fn: Some(curve),
            rate_of_fire_deceleration_fn: Some(curve),
            heat_per_round: 0.04,
            ..barrel()
        };
        let mut wf = WeaponFire::new(
            Arc::new(WeaponDef {
                triggers: vec![trigger(Behavior::Spew)],
                barrels: vec![b],
                overheated_threshold: 2.0, // never reached: this is about the rate
                ..Default::default()
            }),
            vec![0, 0],
        );
        let (shots, _) = hold_for(&mut wf, 60 * 6);
        let g = gaps(&shots);
        // 15 rps is exactly on the grid (4 ticks); any heat at all drops the rate below it, and the
        // grid then snaps to the next step down, 10 rps (6 ticks).
        assert!(g[0] <= 6, "cold, it fires at about the initial 15 rps: {} ticks", g[0]);
        assert!(*g.last().unwrap() >= 10, "hot, it has slowed a long way: {g:?}");
        assert!(g.windows(2).all(|w| w[1] >= w[0]), "and it only ever slows as the heat builds: {g:?}");
    }

    /// A weapon with no recovery threshold cannot cool its way out of overheating; it vents for
    /// its `heat venting time` and then fires again.
    #[test]
    fn a_weapon_with_no_recovery_threshold_vents_and_resumes() {
        let b = BarrelDef { heat_per_round: 0.5, ..barrel() };
        let mut wf = WeaponFire::new(
            Arc::new(WeaponDef {
                triggers: vec![trigger(Behavior::Spew)],
                barrels: vec![b],
                overheated_threshold: 1.0,
                heat_recovery_threshold: 0.0,
                heat_venting_time: 1.5,
                heat_loss_per_second: 0.1,
                ..Default::default()
            }),
            vec![0, 0],
        );
        let (mut over, mut cooled, mut shots) = (None, None, Vec::new());
        for t in 0..400i64 {
            let mut out = Vec::new();
            wf.tick(&held(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Overheated => over = over.or(Some(t)),
                    FireEvent::Cooled => cooled = cooled.or(Some(t)),
                    FireEvent::Shot { .. } => shots.push(t),
                    _ => {}
                }
            }
        }
        let (over, cooled) = (over.expect("overheated"), cooled.expect("it vented and came back"));
        assert_eq!(cooled - over, 90, "1.5 s of venting = 90 ticks");
        assert!(shots.iter().all(|&s| s <= over || s > cooled), "silent while venting: {shots:?}");
        assert!(shots.iter().any(|&s| s > cooled), "and firing again afterwards");
    }

    /// The projectile direction: no spread is the aim exactly, a spread tilts by its angle, and a
    /// fan yaw swings it about the vertical.
    #[test]
    fn a_spread_tilts_the_aim_by_its_angle_and_a_fan_swings_it_sideways() {
        let aim = glam::Vec3::X;
        let none = spread_dir(aim, &Spread { angle: 0.0, azimuth: 1.234, fan_yaw: 0.0 });
        assert!((none - aim).length() < 1.0e-6);
        for az in [0.0f32, 1.0, 2.5, 4.0] {
            let d = spread_dir(aim, &Spread { angle: 0.05, azimuth: az, fan_yaw: 0.0 });
            assert!((d.dot(aim).acos() - 0.05).abs() < 1.0e-4, "azimuth {az}: tilt {}", d.dot(aim).acos());
            assert!((d.length() - 1.0).abs() < 1.0e-5);
        }
        let fan = spread_dir(aim, &Spread { angle: 0.0, azimuth: 0.0, fan_yaw: 0.2 });
        assert!((fan.y.abs() - 0.2f32.sin()).abs() < 1.0e-5 && fan.z.abs() < 1.0e-6, "a fan swings in the horizontal plane: {fan:?}");
        // looking straight up must not blow up
        let up = spread_dir(glam::Vec3::Z, &Spread { angle: 0.1, azimuth: 0.3, fan_yaw: 0.0 });
        assert!((up.length() - 1.0).abs() < 1.0e-5);
    }

    // ---- gating ---------------------------------------------------------------------------------

    /// A weapon still being drawn does not fire (`ready time`), then does, on the right tick.
    #[test]
    fn the_ready_time_delays_the_first_shot() {
        let mut wf = weapon(Behavior::Spew, barrel());
        let mut d = (*wf.def).clone();
        d.ready_time = 1.3;
        wf = WeaponFire::new(Arc::new(d), vec![0, 0]);
        let (shots, _) = hold_for(&mut wf, 200);
        assert_eq!(shots[0], 77, "1.3 s = 78 ticks: the 78th tick is the first the weapon is ready on");
    }

    /// `PlayMode::weapon_usable()` false (a seat that does not allow the biped's weapon) means no
    /// round however the button is held.
    #[test]
    fn an_unusable_weapon_never_fires() {
        let mut wf = weapon(Behavior::Spew, barrel());
        let mut out = Vec::new();
        for _ in 0..120 {
            wf.tick(&FireInput { held: true, usable: false, ..Default::default() }, &mut out);
        }
        assert!(out.iter().all(|e| !matches!(e, FireEvent::Shot { .. })));
        // and a melee swing in progress holds fire the same way
        let mut wf = weapon(Behavior::Spew, barrel());
        let mut out = Vec::new();
        for _ in 0..30 {
            wf.tick(&FireInput { held: true, usable: true, melee: true, ..Default::default() }, &mut out);
        }
        assert!(out.iter().all(|e| !matches!(e, FireEvent::Shot { .. })));
    }

    /// The shot count over a held second does not depend on the frame rate: the loop is
    /// tick-driven, and a long frame runs several ticks.
    #[test]
    fn the_shot_count_does_not_depend_on_the_frame_rate() {
        let mut counts = Vec::new();
        let mut ticks = Vec::new();
        for fps in [20.0f32, 30.0, 60.0, 144.0, 240.0] {
            let mut wf = weapon(Behavior::Spew, barrel());
            let mut n = 0usize;
            let frames = (fps * 3.0) as usize;
            for _ in 0..frames {
                n += wf.advance(1.0 / fps, &held(true)).iter().filter(|e| matches!(e, FireEvent::Shot { .. })).count();
            }
            counts.push(n);
            ticks.push(wf.stats.ticks);
        }
        assert!(counts.iter().all(|&c| (29..=31).contains(&c)), "3 s at 10 rps at several frame rates: {counts:?}");
        assert!(ticks.iter().all(|&t| (179..=181).contains(&t)), "ticks run per 3 s: {ticks:?}");
    }

    // ---- charge and autofire ------------------------------------------------------------------

    fn pistol() -> WeaponFire {
        // plasma pistol: tap = primary barrel on release, hold >= 0.2 s = CHARGE for 0.8 s,
        // release when charged = secondary barrel (the overcharge).
        let t = TriggerDef {
            behavior: Some(Behavior::LatchAutofire),
            primary_barrel: Some(0),
            secondary_barrel: Some(1),
            autofire_time: 0.2,
            autofire_throw: 0.8,
            autofire_primary_action: 0,
            autofire_secondary_action: 1,
            charging_time: 0.8,
            charged_time: 0.0,
            ..Default::default()
        };
        let a = BarrelDef { fire_recovery_time: 0.08, soft_recovery_fraction: 1.0, rounds_per_second: [30.0, 30.0], shots_per_fire: [1, 1], ..barrel() };
        let b = BarrelDef { fire_recovery_time: 0.1, ..a.clone() };
        WeaponFire::new(Arc::new(WeaponDef { triggers: vec![t], barrels: vec![a, b], overheated_threshold: 1.0, ..Default::default() }), vec![0, 0])
    }

    fn barrels_of(out: &[FireEvent]) -> Vec<usize> {
        out.iter().filter_map(|e| if let FireEvent::Shot { barrel, .. } = e { Some(*barrel) } else { None }).collect()
    }

    #[test]
    fn an_autofire_trigger_taps_for_a_shot_and_holds_for_a_charge() {
        // tap: down 3 ticks, up -> ONE round from barrel 0, on the release
        let mut wf = pistol();
        let mut out = Vec::new();
        for _ in 0..3 {
            wf.tick(&held(true), &mut out);
        }
        assert!(barrels_of(&out).is_empty(), "nothing yet while it might become a charge");
        for _ in 0..20 {
            wf.tick(&held(false), &mut out);
        }
        assert_eq!(barrels_of(&out), vec![0], "a tap is one primary-barrel round");

        // hold 0.2 s + the 0.8 s charge + a bit: nothing fires while held...
        let mut wf = pistol();
        let mut out = Vec::new();
        for _ in 0..120 {
            wf.tick(&held(true), &mut out);
        }
        assert!(barrels_of(&out).is_empty(), "a held pistol charges, it does not fire");
        assert_eq!(wf.triggers[0].state, 2, "and is charged by now");
        // ...and the release discharges the secondary barrel once
        for _ in 0..20 {
            wf.tick(&held(false), &mut out);
        }
        assert_eq!(barrels_of(&out), vec![1], "releasing a full charge fires the overcharge barrel");
    }

    #[test]
    fn a_charge_takes_its_charging_time() {
        // Spartan laser shape: spew-charge, 2.5 s, secondary barrel fires 5 rounds at discharge.
        let t = TriggerDef {
            behavior: Some(Behavior::SpewCharge),
            primary_barrel: Some(0),
            secondary_barrel: Some(1),
            charging_time: 2.5,
            charged_time: 0.0,
            overcharged_action: 2,
            ..Default::default()
        };
        let a = BarrelDef { projectile: None, fire_recovery_time: 0.05, ..barrel() };
        let b = BarrelDef { rounds_per_second: [30.0, 30.0], shots_per_fire: [5, 5], fire_recovery_time: 0.8, ..barrel() };
        let mut wf = WeaponFire::new(Arc::new(WeaponDef { triggers: vec![t], barrels: vec![a, b], overheated_threshold: 1.0, ..Default::default() }), vec![0, 0]);
        let mut first_b1 = None;
        let mut b1 = 0;
        for tk in 0..400 {
            let mut out = Vec::new();
            wf.tick(&held(true), &mut out);
            for e in out {
                if let FireEvent::Shot { barrel: 1, .. } = e {
                    b1 += 1;
                    first_b1.get_or_insert(tk);
                }
            }
        }
        let first = first_b1.expect("the beam fires once charged");
        assert!((148..=156).contains(&first), "2.5 s = 150 ticks of charge before the beam: fired at {first}");
        assert_eq!(b1, 5, "the beam is a 5-round string");
    }

    /// Releasing a charge that is not allowed to fire partially cancels it.
    #[test]
    fn an_early_release_cancels_a_charge_that_cannot_fire_partially() {
        let t = TriggerDef {
            behavior: Some(Behavior::Charge),
            primary_barrel: Some(0),
            secondary_barrel: Some(1),
            charging_time: 2.0,
            charging_flags: 0,
            ..Default::default()
        };
        let mut wf = WeaponFire::new(
            Arc::new(WeaponDef { triggers: vec![t], barrels: vec![barrel(), barrel()], overheated_threshold: 1.0, ..Default::default() }),
            vec![0, 0],
        );
        let mut out = Vec::new();
        for _ in 0..40 {
            wf.tick(&held(true), &mut out);
        }
        assert_eq!(wf.triggers[0].state, 1, "charging");
        for _ in 0..40 {
            wf.tick(&held(false), &mut out);
        }
        assert_eq!(wf.triggers[0].state, 0);
        assert!(barrels_of(&out).is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// the real tags
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod real {
    //! #weapon-fire  **The machine on the shipped `forge_halo` weapons.** These skip (loudly) on a
    //! machine without the map, like every other map-backed gate, and each one asserts how many
    //! weapons it actually exercised so a skip-by-accident cannot pass.
    use super::*;

    type Guard = std::sync::MutexGuard<'static, ()>;

    /// Every HAND-HELD weapon of the cache (`objects\weapons\...`), decoded, with the magazines
    /// the HUD would hand it (a full first magazine).
    fn hand_weapons() -> Option<(Guard, Vec<(String, WeaponDef, Vec<i32>)>)> {
        let (g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let mut tags = native.tags_by_group(cache, "weap");
        tags.sort_unstable();
        let mut out = Vec::new();
        for t in tags {
            let name = native.tag_name(cache, t).unwrap_or_default();
            if !name.starts_with("objects\\weapons\\") {
                continue;
            }
            let Some(def) = WeaponDef::read(&native, cache, t) else { continue };
            let loaded = (0..2)
                .map(|i| crate::play_hud::Magazine::read(&native, cache, t, i).map_or(0, |m| m.rounds_loaded_maximum.clamp(0, 511)))
                .collect();
            let leaf = name.rsplit('\\').next().unwrap_or(&name).to_string();
            out.push((leaf, def, loaded));
        }
        Some((g, out))
    }

    fn find<'a>(all: &'a [(String, WeaponDef, Vec<i32>)], leaf: &str) -> &'a (String, WeaponDef, Vec<i32>) {
        all.iter().find(|w| w.0 == leaf).unwrap_or_else(|| panic!("{leaf} is not in forge_halo"))
    }

    fn input(on: bool) -> FireInput {
        FireInput { held: on, usable: true, has_reserve: true, ..Default::default() }
    }

    /// Hold the trigger for `ticks`; the tick of every round and every dry click.
    fn hold(wf: &mut WeaponFire, ticks: i64) -> (Vec<i64>, Vec<i64>) {
        let (mut s, mut c) = (Vec::new(), Vec::new());
        for t in 0..ticks {
            let mut out = Vec::new();
            wf.tick(&input(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => s.push(t),
                    FireEvent::EmptyClick { .. } => c.push(t),
                    _ => {}
                }
            }
        }
        (s, c)
    }

    /// A press of `down` ticks every `period` ticks, for `ticks` ticks.
    fn mash(wf: &mut WeaponFire, ticks: i64, down: i64, period: i64) -> Vec<i64> {
        let mut s = Vec::new();
        for t in 0..ticks {
            let mut out = Vec::new();
            wf.tick(&input(t % period < down), &mut out);
            if out.iter().any(|e| matches!(e, FireEvent::Shot { .. })) {
                s.push(t);
            }
        }
        s
    }

    fn fresh(w: &(String, WeaponDef, Vec<i32>)) -> WeaponFire {
        // The map's weapons are tested cold and unhurried: no draw delay, and heat never gets in
        // the way of a rate measurement.
        let mut d = w.1.clone();
        d.ready_time = 0.0;
        WeaponFire::new(Arc::new(d), w.2.clone())
    }

    /// THE AUTOMATICS, end to end: for every spew weapon that fires an unbroken string (no
    /// `shots per fire` cap), a held trigger fires once per grid-snapped interval of the CURRENT
    /// rate. Steady state is checked after the rate has finished winding up. Heat and ammunition
    /// are taken out of the question (the gates below put them back).
    #[test]
    fn every_real_automatic_fires_at_its_authored_rate_while_held() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let mut seen = Vec::new();
        for w in &all {
            let d = &w.1;
            let Some(t) = d.triggers.first() else { continue };
            let Some(b) = t.primary_barrel.and_then(|p| d.barrels.get(p)) else { continue };
            if t.behavior != Some(Behavior::Spew) || b.shots_per_fire != [0, 0] || b.projectile.is_none() {
                continue;
            }
            let mut wf = fresh(w);
            let mut d2 = (*wf.def).clone();
            d2.secondary_flags |= WSF_NEVER_OVERHEATS;
            wf.def = Arc::new(d2);
            wf.loaded = vec![9999, 9999];
            let (shots, _) = hold(&mut wf, 60 * 12);
            let g = gaps(&shots);
            // The authored FINAL rate, snapped to the 30 Hz grid: 60 ticks/s / (30 / ceil(30/rps)).
            let rps = b.rounds_per_second[1];
            let expect = (60.0 / (30.0 / (30.0 / rps).ceil())).round() as i64;
            let tail = &g[g.len().saturating_sub(10)..];
            assert!(
                tail.iter().all(|&x| x == expect),
                "{}: steady fire should be one round per {expect} ticks ({rps} rps on the grid), got {tail:?}",
                w.0
            );
            // ...and the first interval lies between the initial and the final rate's.
            let first = (60.0 / (30.0 / (30.0 / b.rounds_per_second[0]).ceil())).round() as i64;
            let (lo, hi) = (first.min(expect), first.max(expect));
            assert!((lo..=hi).contains(&g[0]), "{}: first interval {} should lie in {lo}..={hi}", w.0, g[0]);
            assert!(shots.len() > 30, "{}: only {} rounds in 12 s", w.0, shots.len());
            seen.push(w.0.clone());
        }
        eprintln!("play_fire: automatics exercised: {seen:?}");
        assert!(seen.contains(&"assault_rifle".to_string()), "the assault rifle must be among them: {seen:?}");
        assert!(seen.len() >= 4, "only {} automatics exercised: {seen:?}", seen.len());
    }

    fn gaps(v: &[i64]) -> Vec<i64> {
        v.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// The assault rifle, by name: the owner's report. 10 rounds a second, held, until the 32-round
    /// magazine is spent, then one dry click and a reload request.
    #[test]
    fn the_assault_rifle_fires_ten_a_second_until_its_clip_is_empty() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "assault_rifle");
        let mut wf = fresh(w);
        let mut reload = false;
        let (mut shots, mut clicks) = (Vec::new(), Vec::new());
        for t in 0..(60 * 8) {
            let mut out = Vec::new();
            wf.tick(&input(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { projectiles, rounds, .. } => {
                        assert_eq!((projectiles.len(), rounds), (1, 1), "one bullet, one round");
                        shots.push(t);
                    }
                    FireEvent::EmptyClick { .. } => clicks.push(t),
                    FireEvent::NeedReload => reload = true,
                    _ => {}
                }
            }
        }
        assert_eq!(shots.len(), 32, "the magazine is 32 rounds");
        assert!(gaps(&shots).iter().all(|&g| g == 6), "10 rounds a second = every 6 ticks");
        assert_eq!(shots[31], 31 * 6, "the 32nd round is 3.1 s in");
        assert_eq!(clicks, vec![shots[31] + 6], "then ONE dry click, on the next pull");
        assert!(reload, "and the engine asks for a reload");
    }

    /// THE SEMI-AUTOMATICS (`latch`): held, one round; mashed, never faster than the recovery. The
    /// goldens are worked by hand from `sub_140DE3F80` (see `recovery_is_converted_on_the_30hz_grid`):
    /// a round every `total + 2` ticks at the very best -- DMR 22, magnum 14, sniper 48, shotgun 64,
    /// rocket launcher 52.
    #[test]
    fn every_real_semi_automatic_fires_once_per_press_and_waits_out_its_recovery() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let golden: &[(&str, i64)] = &[("dmr", 22), ("magnum", 14), ("sniper_rifle", 48), ("shotgun", 64), ("rocket_launcher", 52)];
        for (leaf, spacing) in golden {
            let w = find(&all, leaf);
            assert_eq!(w.1.triggers[0].behavior, Some(Behavior::Latch), "{leaf} is a latch");
            // held: one round
            let mut wf = fresh(w);
            let (held, _) = hold(&mut wf, 60 * 6);
            assert_eq!(held.len(), 1, "{leaf}: a held semi-automatic fires once");
            // mashed as fast as a finger can: 2 ticks down, 2 up
            let mut wf = fresh(w);
            wf.loaded = vec![999, 999];
            let m = mash(&mut wf, 60 * 20, 2, 4);
            let g = gaps(&m);
            assert!(g.iter().all(|&x| x >= *spacing), "{leaf}: a mash beat the recovery ({spacing} ticks): {g:?}");
            assert!(*g.iter().min().unwrap() <= spacing + 4, "{leaf}: a mash should fire about as fast as the recovery allows: {g:?}");
            // and a patient player gets every pull
            let mut wf = fresh(w);
            wf.loaded = vec![999, 999];
            let p = mash(&mut wf, 60 * 20, 1, spacing + 10);
            assert_eq!(p.len() as i64, (60 * 20 + spacing + 9) / (spacing + 10), "{leaf}: every patient pull fires");
        }
        // every OTHER latch weapon also fires exactly once while held
        let mut latches = 0;
        for w in &all {
            let Some(t) = w.1.triggers.first() else { continue };
            if !matches!(t.behavior, Some(Behavior::Latch | Behavior::LatchTether)) || t.primary_barrel.is_none() {
                continue;
            }
            let mut wf = fresh(w);
            let (held, _) = hold(&mut wf, 60 * 4);
            assert_eq!(held.len(), 1, "{}: held latch fired {} rounds", w.0, held.len());
            latches += 1;
        }
        assert!(latches >= 6, "only {latches} latch weapons exercised");
    }

    /// Every real magazine weapon empties after exactly its loaded rounds / rounds per shot.
    #[test]
    fn every_real_magazine_empties_after_exactly_its_rounds() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let mut seen = Vec::new();
        for w in &all {
            let d = &w.1;
            let Some(t) = d.triggers.first() else { continue };
            let Some(b) = t.primary_barrel.and_then(|p| d.barrels.get(p)) else { continue };
            let Some(m) = b.magazine else { continue };
            let cap = w.2[m];
            if cap <= 0 || b.projectile.is_none() || !matches!(t.behavior, Some(Behavior::Spew | Behavior::Latch | Behavior::LatchTether)) {
                continue;
            }
            let mut wf = fresh(w);
            let mut d2 = (*wf.def).clone();
            d2.secondary_flags |= WSF_NEVER_OVERHEATS;
            wf.def = Arc::new(d2);
            // spew: hold; latch: press every 80 ticks so every pull is allowed
            let mut shots = 0usize;
            let mut clicks = 0usize;
            for tick in 0..(60 * 600) {
                let on = if t.behavior == Some(Behavior::Spew) { true } else { tick % 100 < 2 };
                let mut out = Vec::new();
                wf.tick(&input(on), &mut out);
                for e in out {
                    match e {
                        FireEvent::Shot { .. } => shots += 1,
                        FireEvent::EmptyClick { .. } => clicks += 1,
                        _ => {}
                    }
                }
                if clicks > 0 {
                    break;
                }
            }
            let per = (b.rounds_per_shot as i32).max(1);
            let expect = if b.flags & BF_CAN_FIRE_WITH_PARTIAL_AMMO != 0 { (cap + per - 1) / per } else { cap / per } as usize;
            assert_eq!(shots, expect, "{}: {cap} rounds at {per} per shot", w.0);
            assert_eq!(clicks, 1);
            seen.push(format!("{}={}", w.0, shots));
        }
        eprintln!("play_fire: magazines exercised: {seen:?}");
        assert!(seen.len() >= 9, "only {} magazines exercised: {seen:?}", seen.len());
    }

    /// Every real barrel launches exactly its authored `projectiles per shot` per fire event and a
    /// fire event costs exactly its `rounds per shot`.
    #[test]
    fn every_real_barrel_launches_its_authored_projectile_count() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let mut seen = 0;
        for w in &all {
            let mut wf = fresh(w);
            for bi in 0..w.1.barrels.len() {
                let b = &w.1.barrels[bi];
                if b.projectile.is_none() {
                    continue;
                }
                // Fire the barrel directly: request it and tick until it shoots.
                wf.loaded = vec![999, 999];
                wf.barrels[bi].request = true;
                let mut got = None;
                for _ in 0..4 {
                    let mut out = Vec::new();
                    wf.tick(&input(false), &mut out);
                    if let Some(FireEvent::Shot { projectiles, rounds, barrel }) = out.into_iter().find(|e| matches!(e, FireEvent::Shot { .. })) {
                        got = Some((projectiles.len(), rounds, barrel));
                        break;
                    }
                }
                let (n, rounds, barrel) = got.unwrap_or_else(|| panic!("{} barrel {bi} never fired", w.0));
                assert_eq!(barrel, bi);
                assert_eq!(n as i16, b.projectiles_per_shot, "{} barrel {bi}", w.0);
                if b.magazine.is_some() {
                    assert_eq!(rounds as i16, b.rounds_per_shot, "{} barrel {bi}", w.0);
                }
                seen += 1;
            }
        }
        assert!(seen >= 20, "only {seen} barrels exercised");
    }

    /// The plasma pistol on its real tags: a tap is one primary round, a hold is a charge, and a
    /// charged release is one round from the SECOND barrel (the overcharge).
    #[test]
    fn the_real_plasma_pistol_taps_and_charges() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "plasma_pistol");
        let barrels = |out: &[FireEvent]| -> Vec<usize> {
            out.iter().filter_map(|e| if let FireEvent::Shot { barrel, .. } = e { Some(*barrel) } else { None }).collect()
        };
        let mut wf = fresh(w);
        let mut out = Vec::new();
        for _ in 0..4 {
            wf.tick(&input(true), &mut out);
        }
        for _ in 0..30 {
            wf.tick(&input(false), &mut out);
        }
        assert_eq!(barrels(&out), vec![0], "a tap is one normal bolt");
        let mut wf = fresh(w);
        let mut out = Vec::new();
        for _ in 0..(60 * 3) {
            wf.tick(&input(true), &mut out);
        }
        assert!(barrels(&out).is_empty(), "holding charges");
        for _ in 0..30 {
            wf.tick(&input(false), &mut out);
        }
        assert_eq!(barrels(&out), vec![1], "releasing a full charge fires the overcharge barrel once");
    }

    /// The Spartan laser on its real tags: 2.5 s of charge, then a 5-round beam; release early and
    /// nothing fires.
    #[test]
    fn the_real_spartan_laser_charges_for_two_and_a_half_seconds() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "spartan_laser");
        let mut wf = fresh(w);
        let mut first = None;
        let mut beam = 0;
        for t in 0..400 {
            let mut out = Vec::new();
            wf.tick(&input(true), &mut out);
            for e in out {
                if let FireEvent::Shot { barrel: 1, .. } = e {
                    beam += 1;
                    first.get_or_insert(t);
                }
            }
            if t == 120 {
                assert!(first.is_none(), "no beam before the charge completes");
            }
        }
        let first = first.expect("the beam fires");
        assert!((149..=155).contains(&first), "charge time 2.5 s = 150 ticks: beam at {first}");
        assert_eq!(beam, 5, "the beam barrel is a 5-round string");
    }

    /// The plasma launcher's fire fractions: a quarter charge owes one bolt, a full charge four.
    #[test]
    fn the_real_plasma_launcher_owes_a_bolt_per_charge_fraction() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "plasma_launcher");
        assert_eq!(w.1.triggers[0].fire_fractions, vec![0.25, 0.5, 0.75, 1.0], "the tag's own fractions");
        let bolts = |hold_ticks: i64| -> usize {
            let mut wf = fresh(w);
            let mut n = 0;
            for t in 0..(hold_ticks + 300) {
                let mut out = Vec::new();
                wf.tick(&input(t < hold_ticks), &mut out);
                n += out.iter().filter(|e| matches!(e, FireEvent::Shot { .. })).count();
            }
            n
        };
        // 3 s to full charge = 180 ticks
        assert_eq!(bolts(5), 0, "an instant release is no charge at all");
        assert_eq!(bolts(50), 1, "~28% charged owes the 0.25 fraction");
        assert_eq!(bolts(100), 2, "~56%");
        assert_eq!(bolts(150), 3, "~83%: 0.25, 0.5 and 0.75");
        assert_eq!(bolts(190), 4, "full");
    }

    /// The repeater on its real tags: it slows as it heats, and its battery (age 0.005 a round)
    /// is 200 rounds, after which it is dead.
    #[test]
    fn the_real_plasma_repeater_slows_as_it_heats_and_dies_at_two_hundred_rounds() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "plasma_repeater");
        let mut wf = fresh(w);
        let (mut shots, mut clicks) = (Vec::new(), Vec::new());
        for t in 0..(60 * 120i64) {
            let mut out = Vec::new();
            wf.tick(&input(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } => shots.push(t),
                    FireEvent::EmptyClick { .. } => clicks.push(t),
                    _ => {}
                }
            }
        }
        let g = gaps(&shots);
        assert!(g[0] <= 6, "cold it fires at about its initial 15 rps (4-6 ticks on the grid): {}", g[0]);
        assert!(g.iter().max().copied().unwrap() >= 12, "it slows well below 15 rps as it heats");
        assert!(g.windows(2).all(|w| w[1] + 3 >= w[0]), "and it does not speed up again while held: {g:?}");
        // 0.005 summed 200 times in f32 lands a hair under 1.0, so the engine's float add gives one
        // more round: 200 or 201, never a different order of magnitude.
        assert!((200..=201).contains(&shots.len()), "age 0.005 a round: a 200-round battery ({})", shots.len());
        assert_eq!(clicks.len(), 1, "then it clicks dry");
        assert!(wf.age >= 1.0);
    }

    /// Every real battery weapon dies after 1 / age-per-round rounds.
    #[test]
    fn every_real_battery_weapon_runs_out_at_maximum_age() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let mut seen = Vec::new();
        for (leaf, rounds) in [("plasma_repeater", 200usize), ("plasma_rifle", 400)] {
            let w = find(&all, leaf);
            assert!(w.1.flags & WF_CANNOT_FIRE_AT_MAX_AGE != 0, "{leaf} is flagged `cannot fire at maximum age`");
            let mut wf = fresh(w);
            let mut d = (*wf.def).clone();
            d.secondary_flags |= WSF_NEVER_OVERHEATS; // the battery, not the heat, is what is measured
            wf.def = Arc::new(d);
            let (s, c) = hold(&mut wf, 60 * 400);
            assert!((rounds..=rounds + 1).contains(&s.len()) && c.len() == 1, "{leaf}: {} rounds, {} clicks", s.len(), c.len());
            seen.push(leaf);
        }
        assert_eq!(seen.len(), 2);
    }

    /// Heat on the real plasma weapons: the rifle overheats if it is held, the AR (no heat) never.
    #[test]
    fn the_real_plasma_rifle_overheats_and_the_assault_rifle_does_not() {
        let Some((_g, all)) = hand_weapons() else { return eprintln!("play_fire: forge_halo not available") };
        let w = find(&all, "plasma_rifle");
        let mut wf = fresh(w);
        let mut over = None;
        let mut rounds_before = 0;
        for t in 0..(60 * 30) {
            let mut out = Vec::new();
            wf.tick(&input(true), &mut out);
            for e in out {
                match e {
                    FireEvent::Shot { .. } if over.is_none() => rounds_before += 1,
                    FireEvent::Overheated => over = over.or(Some(t)),
                    _ => {}
                }
            }
        }
        let over = over.expect("the plasma rifle overheats when held");
        // 0.15 per round, at best ~9 rps (1.35/s) against 0.853/s of cooling: seconds, not ticks
        assert!(over > 60 && rounds_before > 10, "overheated at tick {over} after {rounds_before} rounds");
        let w = find(&all, "assault_rifle");
        let mut wf = fresh(w);
        hold(&mut wf, 60 * 8);
        assert!(!wf.overheated && wf.heat == 0.0, "the assault rifle authors no heat");
    }

    /// Print the whole weapon table (what each hand weapon does, in ticks and seconds) for the
    /// report: `cargo test play_fire::real::probe_weapon_table -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints a report; run alone with --ignored --nocapture"]
    fn probe_weapon_table() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let mut tags = native.tags_by_group(cache, "weap");
        tags.sort_unstable();
        for t in tags {
            let name = native.tag_name(cache, t).unwrap_or_default();
            if !name.starts_with("objects\\weapons\\") {
                continue;
            }
            let Some(def) = WeaponDef::read(&native, cache, t) else { continue };
            let leaf = name.rsplit('\\').next().unwrap_or(&name).to_string();
            let mags: Vec<i32> = (0..2)
                .map(|i| crate::play_hud::Magazine::read(&native, cache, t, i).map_or(0, |m| m.rounds_loaded_maximum.clamp(0, 511)))
                .collect();
            let reload = crate::play_hud::reload_seconds(&native, cache, t).map_or("-".to_string(), |(s, c)| format!("{s:.2}s '{c}'"));
            println!("== {leaf}: magazine {:?}, reload {reload}", mags);
            println!("   {}", describe(&def));
            let d = std::sync::Arc::new(def.clone());
            for (ti, tr) in def.triggers.iter().enumerate() {
                let Some(b) = tr.primary_barrel.and_then(|p| def.barrels.get(p)) else { continue };
                // measured: held 6 s (battery/heat/ammo out of the way) and mashed 10 s
                let mut dd = (*d).clone();
                dd.secondary_flags |= WSF_NEVER_OVERHEATS;
                dd.ready_time = 0.0;
                let mut wf = WeaponFire::new(Arc::new(dd.clone()), vec![999, 999]);
                let (held, _) = hold(&mut wf, 60 * 6);
                let mut wf = WeaponFire::new(Arc::new(dd), vec![999, 999]);
                let m = mash(&mut wf, 60 * 10, 2, 4);
                let (total, locked) = recovery_ticks(b);
                println!(
                    "   trigger {ti} {:?}: held 6 s = {} rounds (last gap {:?}); mashed 10 s = {} rounds (min gap {:?}); recovery {total}/{locked} ticks; heat/round {:.3} age/round {:.4}",
                    tr.behavior.map(|b| b.word()),
                    held.len(),
                    gaps(&held).last(),
                    m.len(),
                    gaps(&m).iter().min(),
                    b.heat_per_round,
                    b.age_per_round,
                );
            }
        }
    }
}
