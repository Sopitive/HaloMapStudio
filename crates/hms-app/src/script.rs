//! The Forge script grammar: one `EditorCommand` per line, parsed by [`parse_line`], plus the
//! program runner ([`run_program`]: comments, `foreach ... end` loops, `$var` substitution).
//! Every scriptable mutation goes through this one command set, so the in-app script console,
//! the headless `--script` / `--exec` host, the TCP command server and the MCP bridge all share
//! one executor. `DEFAULT_SCRIPT` is the console's seed text and doubles as the grammar summary.

/// Seed text for the script panel: every verb the parser accepts, with its accepted aliases.
pub const DEFAULT_SCRIPT: &str = "\
# HMS forge script - one command per line. '#' or '//' = comment.
# --- placing / editing ---
# place <name|#idx> [at X Y Z | rel <datum> DX DY DZ | onface <datum> <dir> | camera]
# move <datum> DX DY DZ    moveto <datum> X Y Z    rotate <datum> <x|y|z> DEG
# set <datum> <field> <value>   fields: team color label spawnseq scale cached_type respawn shape
#     radius|width|length|top|bottom <wu>   b0..b3 <raw>
#     scaled on|off|default   shadow on|off|default   (per-object PSEUDO-flags, not saved to the .mvar;
#     default = auto: scaled <- has the 'scale' label, shadow <- GREEN team + 'scale' label)
#     team also takes a name: red blue green orange purple yellow brown pink neutral none
#     Halo 4 objects: spawnseq (-100..100, the Forge menu's 'spawn sequence') spawntime label|label2|label3|label4 <name|#|none>
#       traitzone userdata placement_hi locked clips channel passability
#       shape none|sphere|cylinder|box   radius|width|length|top|bottom <wu>   b0..b3 <raw 0..65535>
#       scale <0..10>  the record's own variant_object_scale (64 steps, 1.0 = the shipped default q 7;
#                      the game keeps but does not draw it); scale_q <0..63> sets the raw quantum.
#                      label scale + spawnseq = the scaling-gametype rule (SCALED flag), as in Reach
# set selection <field> <value>    edits every selected object (e.g. set selection shadow on)
# select <datum|all|none>   select box X1 Y1 Z1 X2 Y2 Z2 [add]   deselect
# delete [datum]   dup [datum] [DX DY DZ]   settle [datum|all]
# pick [delete] [radius R]  select (or delete) what is under the crosshair (+ everything within R)
# coincident <datumA> <faceA> <datumB> <faceB> [centered] [noturn]   face-to-face mate (CAD)
# snapto [<datum>|selection] [to <datum>] [axis <+x|-x|+y|-y|+z|-z>]   the Ctrl magnet, scripted:
#     oriented face snap + edge align; `axis` locks the travel to that axis (as an axis-locked grab does)
# array line from X Y Z to X Y Z (count N | step WU) [align]   fill a line (step < piece = overlap)
# array axis <+x|-x|+y|-y|+z|-z> count N [step WU] [align]     the same, along a world axis
# construct [get] | on|off | op <guide|circle|square|coincident|anchor|mirror|line>
#     construct snap <all|corners|edges|faces|centres>   the CAD panel's snap-to mode
#     construct anchors <datum>   every point a Construct click could snap to on that object
#     construct clear             drop the guides + shapes
# undo   redo               step the edit history (interactive app only)
# save [as <path>]   saveas <path>   count [filters]   echo <text>
# --- globals / overlays (on|off|get) ---
# scale on|off|get            scaled objects render at their spawn-seq size (off = all x1)
# shadowcasters on|off|get    flagged Forge objects cast sun shadows (off = engine casters only)
# screenfx on|off|get         placed Forge special-FX screen effects (off = map default only)
# mapspawns on|off|get        show the MAP's own (scenario) spawn markers; off (default) hides them
# outlines on|off|get         hidden-block physics hulls (View > Show hidden-block physics hulls); on by default
# wirexray on|off|get        draw the SELECTION wireframe through other objects (View menu); off by default
# softceilings on|off|get     the map's soft ceilings / kill floor (View > Soft ceilings); get lists them
# triggers on|off|get         the scenario's trigger volumes (View > Trigger volumes)
# hardfloor on|off|get        the map's world box (Havok broadphase: every BSP's bounds +-64) + its floor (View > Hard floor); get lists the BSPs
# playablebounds on|off|get   the playable BSPs' own world-bounds boxes (View > Playable BSP bounds)
# flags [get]                 report the globals + the selection's effective flags
# bspwarn [list]              structure BSP playable flags + objects outside playable space
# preview team <t> | color <c|inherit> | off | get   pin the dropdown-hover colour preview on the selection
# preview move X Y [frames] | click X Y | drag X1 Y1 X2 Y2 | key <name> | type <text> | where
#     pointer / keyboard simulation for UI testing (interactive app only)
# --- camera ---
# camera to X Y Z | lookat X Y Z | nudge DX DY DZ | spawn | frame | get
# camera clearance                 how close is the nearest surface, in any direction
# camera standoff [N]              push the camera until nothing solid is within N units
# camera orbit <datum|selection|X Y Z> <dist> [yaw] [pitch]
# --- maps / variants / capture ---
# loadmap <name|path>       loadvariant <path>       newvariant       wait
# screenshot <path|dir> [WxH]   (dir or trailing / auto-names <stem>.png; supports $stem)
# mapid [current | <mapname> | variant <path>]
# variant get [field]       every global field of the open variant (or one), by its engine name
# variant set <field> <value...>   name description author editor category budget bounds quotamin quotamax quota
# --- queries (filter with: type <t>  name <s>  label <s>) ---
# list palette [f] | list maps [f] | list variants [current|<id>|map <name>|all] [f]
# list objects [filters] | list types | get <datum>
# --- loops ---  foreach <source> as $v ... end   (sources: variants*, maps, objects)
#   $v = item value, $stem = file stem, $idx = index
# dir = +x -x +y -y +z -z (aliases north/south/east/west/up/down)
# aliases: spawn=place  del=delete  duplicate=dup  drop=settle  rot=rotate  mate=coincident  aim=pick
#   cam=camera  info=get  map_id=mapid  openmap=loadmap  openvariant|loadmvar=loadvariant  shot|snap=screenshot
#   facesnap|snapface=snapto  (plain `snap` is screenshot)
#   objects|objs=list objects  types=list types  maps=list maps  variants=list variants  mvar=variant
#   scaled|scaling=scale  shadowcaster|casters|shadows=shadowcasters  forgefx|fx=screenfx
#   map_spawns|scnrspawns|mapspawn=mapspawns  outline|physoutlines|physics_outlines|blockers=outlines
#   softceiling|soft_ceilings|ceilings|mapfloor|floor=softceilings  trigger|triggervolumes|trigger_volumes=triggers
#   hard_floor|mapbounds|worldbounds|bounds=hardfloor  playable_bounds|playable|bspbounds=playablebounds
#   new_variant=newvariant  flag=flags  hover=preview  bsp_warn|bspflags=bspwarn
#   list pal|type|map|variant = list palette|types|maps|variants   camera look|by|pos|clear|unstick|push = lookat|nudge|get|clearance|standoff

list palette block
";

/// Which palette object to place — by flat index (`#3`) or by name substring (case-insensitive).
#[derive(Clone, Debug)]
pub enum ObjRef {
    Index(usize),
    Name(String),
}

/// Where to place a new object.
#[derive(Clone, Debug)]
pub enum Placement {
    /// Absolute world point.
    At([f32; 3]),
    /// Relative to another object's position + an offset.
    Relative { datum: u32, off: [f32; 3] },
    /// Snapped against another object's FACE along a unit direction (for precision).
    OnFace { datum: u32, dir: [f32; 3] },
    /// On the surface under the camera (fallback: 12u ahead).
    Camera,
}

#[derive(Clone, Debug)]
pub enum CameraCmd {
    To([f32; 3]),
    LookAt([f32; 3]),
    Spawn,
    Frame,
    /// Push the camera until nothing solid (terrain, BSP or forge objects) is within this many
    /// world units. `None` = use the app's configured stand-off distance.
    Standoff(Option<f32>),
    /// Report the distance to the nearest solid surface in any direction (and how boxed in we are).
    Clearance,
    /// Move BY a delta in world units.
    Nudge([f32; 3]),
    /// Orbit to look at a target from `dist` units away at the given yaw/pitch (degrees).
    /// Target: an object datum, the current selection, or an explicit point.
    Orbit { target: OrbitTarget, dist: f32, yaw: f32, pitch: f32 },
    /// Print the current pose.
    Get,
}

/// What `camera orbit` frames.
#[derive(Clone, Debug)]
pub enum OrbitTarget {
    Datum(u32),
    Selection,
    Point([f32; 3]),
}

/// Which map to report an id for.
#[derive(Clone, Debug)]
pub enum MapRef {
    /// The currently loaded map.
    Current,
    /// A base map by name/substring.
    Named(String),
    /// A .mvar variant file (its `m_map_id`).
    Variant(String),
}

/// Which set of variants to list / iterate.
#[derive(Clone, Debug)]
pub enum VariantSel {
    /// Variants whose base-map id equals the currently loaded map.
    Current,
    /// Variants for an explicit base-map id.
    Id(u32),
    /// Variants for the map resolved from this name/substring.
    MapName(String),
    /// Every variant found, regardless of base map.
    All,
}

/// Sentinel datum meaning "every selected object" for `EditorCommand::Set`. 0 is never a real
/// datum (they are 0xD0xxxxxx / 0xE0xxxxxx / 0xF0xxxxxx).
pub const SET_SELECTION: u32 = 0;

/// One scriptable editor operation. The MCP bridge and the command server build these too.
#[derive(Clone, Debug)]
pub enum EditorCommand {
    Place { obj: ObjRef, pos: Placement },
    Select(u32),
    /// `select box X1 Y1 Z1 X2 Y2 Z2 [add]`: every offline object whose world bounds (a hidden
    /// block's physics hull) overlap the axis-aligned box; the same test the marquee uses.
    SelectBox { min: [f32; 3], max: [f32; 3], additive: bool },
    SelectAll,
    Deselect,
    Delete(Option<u32>), // None = current selection
    Move { datum: u32, delta: [f32; 3] },
    MoveTo { datum: u32, pos: [f32; 3] },
    /// Gravity-settle: drop an object (or the whole selection with `all`) onto the surface below
    /// it and tilt it to rest flush on the slope. None = current selection.
    Settle(Option<u32>),
    /// CAD mate: move object `a` so its face `af` becomes coincident with object `b`'s face
    /// `bf`. Faces are 0..5 = -X,+X,-Y,+Y,-Z,+Z (or the names "-x".."+z"). `centered` also puts
    /// the face centres together; otherwise the faces are only made coplanar (flush), which
    /// keeps `a`'s position within that plane.
    Coincident { a: u32, af: u8, b: u32, bf: u8, centered: bool, turn: bool },
    /// #snap-array: `array line ... from X Y Z to X Y Z (count N | step W) [align]` -- fill a
    /// line with `target` (or the selection): the original moves to the start point and copies
    /// fill the rest. `count` spreads N copies from end to end; `step` puts one every W world
    /// units, so a step smaller than the piece's own extent makes the copies OVERLAP. `align`
    /// yaws every copy to follow the line. One undo step; multi-object selections stamp as a unit.
    ArrayLine { target: Option<u32>, from: [f32; 3], to: [f32; 3], count: Option<u32>, step: Option<f32>, align: bool },
    /// #snap-array: `array axis <+x|-x|...> count N [step W] [align]` -- the same fill, laid out
    /// along a world axis from where the selection already is. `step` defaults to the piece's own
    /// extent along that axis (exact face-to-face).
    ArrayAxis { target: Option<u32>, dir: [f32; 3], count: u32, step: Option<f32>, align: bool },
    /// #snap-array: `snapto <datum|selection> [to <datum>]` -- the scripted form of the Ctrl
    /// magnet: the same oriented face-pair snap plus edge alignment, as a translation. With `to`,
    /// only that object's faces are considered; without it, every other object and the level.
    /// `axis` locks the snap's travel to +/- that world direction (what holding Ctrl with an
    /// axis lock does in the viewport); `None` lets each face push along its own normal.
    SnapTo { target: Option<u32>, to: Option<u32>, axis: Option<[f32; 3]> },
    Rotate { datum: u32, axis: usize, deg: f32 }, // axis 0/1/2 = X/Y/Z (world)
    Set { datum: u32, field: String, value: String },
    Camera(CameraCmd),
    ListPalette(Option<String>),
    /// List scene objects, optionally filtered by cached type / name substring / label substring.
    ListObjects { type_filter: Option<String>, name_filter: Option<String>, label_filter: Option<String> },
    /// List distinct object types present in the scene (name → count).
    ListTypes,
    /// List available base maps (optional name filter).
    ListMaps(Option<String>),
    /// List .mvar variants (optionally only those whose base map matches).
    ListVariants { sel: VariantSel, filter: Option<String> },
    /// Print the numeric map id of a map / variant / the current map.
    MapId(MapRef),
    /// Dump all fields of one object.
    Get(u32),
    /// Load a base map by name/substring/path.
    LoadMap(String),
    /// Load a .mvar variant (auto-loads its base map).
    LoadVariant(String),
    /// Render the current view to a PNG. `path` may name a directory (auto `<stem>.png`) or a
    /// file; may contain `$stem`. Optional explicit size.
    Screenshot { path: String, size: Option<(u32, u32)> },
    /// Block until the current load + object rebuild has settled (headless; best-effort interactive).
    Wait,
    /// Raycast from the CURRENT camera forward and select the object under the crosshair. With
    /// `radius`, also grab every object within that many world units of the hit (a whole building
    /// cluster). With `delete`, remove the selection instead of just selecting it.
    Pick { delete: bool, radius: Option<f32> },
    Echo(String),
    /// Write the open variant back to disk (`save`), or to a new path (`save as <path>`). The
    /// scripted counterpart of File ▸ Save — without it a batch edit could not be persisted.
    Save { path: Option<String> },
    /// Duplicate an object (or the whole selection) and offset the copy.
    Dup { datum: Option<u32>, off: [f32; 3] },
    /// Count objects matching the same filters `list objects` takes.
    Count { type_filter: Option<String>, name_filter: Option<String>, label_filter: Option<String> },
    /// `screenfx on|off` sets whether placed Forge special-FX objects' screen effects
    /// are applied (the map's own default effect always is); `screenfx get` (None) reports.
    ScreenFx(Option<bool>),
    /// `scale on|off`: the global "Scaled objects" switch; `scale get` (None) reports.
    ScaledGlobal(Option<bool>),
    /// `shadowcasters on|off`: the global "Shadow casters (Forge)" switch; None reports.
    ShadowCastersGlobal(Option<bool>),
    /// `mapspawns on|off`: show/hide the scenario's built-in spawn markers
    /// (View > "Show map spawn points"); `mapspawns get` (None) reports. Variant spawns are unaffected.
    MapSpawns(Option<bool>),
    /// `outlines on|off`: show/hide the forge-placed hidden blocks' physics hulls
    /// (View > "Show hidden-block physics hulls", default on); `outlines get` (None) reports.
    PhysicsOutlines(Option<bool>),
    /// #h4-phys `collision on|off`: the SELECTED object's collision-model (`coll`) hull overlay
    /// (View > Overlays > "Collision (selected)", default off); `collision get` (None) reports.
    CollisionOverlay(Option<bool>),
    /// #h4-phys `physics on|off`: the SELECTED object's physics-model (`phmo`) hull overlay
    /// (View > Overlays > "Physics (selected)", default off); `physics get` (None) reports.
    PhysicsOverlay(Option<bool>),
    /// `// #h4-expo-3` `settings on|off`: open / close the Settings window (Lighting & rendering);
    /// `settings get` (None) reports. Scripted so a regression run can prove that OPENING a panel
    /// leaves the rendered frame untouched - the panel's seed block used to push its own exposure /
    /// band / fog values into the renderer on its first draw.
    SettingsPanel(Option<bool>),
    /// #dialogs test hook (GUI only, print-only): drive the variant browser without a mouse so a
    /// regression run can assert the folder-seeding + resolved save path and screenshot the reflow.
    ///   `dialog open`                open the browser in OPEN mode
    ///   `dialog saveas`              the real File>Save As path (runs the save gates first)
    ///   `dialog savebrowser [name]`  open the browser in SAVE mode directly (bypasses the gates;
    ///                                pre-seeds the file name field when `name` is given)
    ///   `dialog size <W> <H>|reset`  pin / release the window size for a screenshot
    ///   `dialog shot <path>`         write a PNG of the whole egui frame (dialog included)
    ///   `dialog close` / `dialog`    close / report seeded dir + file + resolved path
    DialogDbg { sub: String, a1: Option<String>, a2: Option<String> },
    /// #wire-visible  `wirexray on|off`: draw the selection wireframe THROUGH other objects
    /// (View > "Selection wireframe through objects", default off); `wirexray get` (None) reports.
    WireXray(Option<bool>),
    /// `softceilings on|off`: show/hide the structure-design soft ceilings (the
    /// kill floor / acceleration / slip planes; View > "Soft ceilings"); `get` (None) lists them.
    SoftCeilings(Option<bool>),
    /// `triggers on|off`: the scenario trigger-volume boxes (View > "Trigger
    /// volumes"); `get` (None) reports.
    TriggerVolumes(Option<bool>),
    /// `hardfloor on|off`: show/hide the playable structure BSPs' world bounds and
    /// floor plane (View > "Hard floor (BSP world bounds)"); `get` (None) lists every BSP.
    HardFloor(Option<bool>),
    /// `playablebounds on|off`: the playable structure BSPs' own world-bounds
    /// boxes (View > "Playable BSP bounds"); `get` (None) reports.
    PlayableBounds(Option<bool>),
    /// `undo` / `redo`: step the editor's offline edit history (interactive app only;
    /// the headless batch host keeps no history).
    Undo,
    Redo,
    /// `newvariant`: start an empty variant on the loaded map (File > New
    /// variant): the smallest shipped variant of the map is the template; save with `save as`.
    NewVariant,
    /// `flags [get]`: report both globals and the selection's effective per-object flags.
    FlagsGet,
    /// `bspwarn [list]`: the map's structure BSP playable flags + every object whose
    /// position is in a BSP marked "not normally playable space in MP" (the save-time warning list).
    BspWarnList,
    /// `variant get [field]`: every global field of the open variant (both games) with its
    /// engine name, or one field; `variant set <field> <value...>`: edit one
    /// (name, description, author, editor, category, budget, bounds, quotamin/quotamax/quota),
    /// written by the next `save`.
    VariantGet(Option<String>),
    VariantSet { field: String, value: String },
    /// `preview team <name|-1..8>` / `preview color <inherit|-1..7>` pins the same
    /// transient colour preview a dropdown hover starts (wireframe hidden, colour applied, nothing
    /// saved) on the selection; `preview off` ends it; `preview [get]` (both None, off=false) reports.
    /// `off` = true ends the preview. Interactive app only (the headless host has no hover).
    PreviewHover { team: Option<u8>, color: Option<i32>, off: bool },
    /// Debug hook (print-only): `preview move <x> <y> [frames]` injects a synthetic
    /// pointer at that window position (egui points) for the next frames, `preview click <x> <y>`
    /// a press + release there, `preview where` prints the combo/popup rects + pointer + hover state.
    PointerSim(PointerSim),
    /// #construct-h4: the Construct (CAD) tool, scripted -- the toolbar/panel state the viewport
    /// clicks read, plus a read-back of what a click would find. `construct` reports; the rest
    /// set the tool exactly as the toolbar and the CAD panel do.
    Construct(ConstructCmd),
}

/// What the `construct` verb asks for (see [`EditorCommand::Construct`]).
#[derive(Clone, Debug, PartialEq)]
pub enum ConstructCmd {
    /// `construct [get]` -- tool on/off, the chosen op, the anchor mode, guides / shapes / face A.
    Get,
    /// `construct on|off` -- arm the Construct tool (the same switch as the toolbar and `C`).
    Enable(bool),
    /// `construct op <guide|circle|square|coincident|anchor|mirror|line>`.
    Op(String),
    /// `construct snap <all|corners|edges|faces|centres>` -- the panel's "Snap to" mode.
    Mode(String),
    /// `construct anchors <datum>` -- every anchor point that object offers right now.
    Anchors(u32),
    /// `construct clear` -- drop the guides and shapes (the panel's clear buttons).
    Clear,
}

/// What `preview move|click|where` asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum PointerSim {
    Move { x: f32, y: f32, frames: u32 },
    Click { x: f32, y: f32 },
    /// #construct-h4: a press at (x1,y1), pointer motion to (x2,y2) and a release there, over
    /// several frames -- what the press-drag tools (the Construct Circle / Square) need.
    Drag { x1: f32, y1: f32, x2: f32, y2: f32 },
    /// A key press + release (`preview key r`), by egui key name -- the modal transforms and
    /// every other hotkey are keys, so a script can only reach them this way.
    Key { name: String },
    /// Typed TEXT (`preview type 90-`), as the keyboard would deliver it: the modal transform's
    /// numeric entry and any focused text field read this.
    Type { text: String },
    Where,
}

/// Parse a forge team as a number (−1..8) or a name (red, blue, green, …, neutral,
/// none). Returns the raw mvar team byte (0xFF = none).
pub fn parse_team(s: &str) -> Result<u8, String> {
    let t = s.trim().to_ascii_lowercase();
    if let Ok(v) = t.parse::<i32>() {
        return if v < 0 { Ok(0xFF) } else if v <= 8 { Ok(v as u8) } else { Err("team must be -1..8".into()) };
    }
    Ok(match t.as_str() {
        "red" => 0,
        "blue" => 1,
        "green" => 2,
        "orange" => 3,
        "purple" => 4,
        "yellow" | "gold" => 5,
        "brown" => 6,
        "pink" => 7,
        "neutral" => 8,
        "none" | "no" | "-" => 0xFF,
        _ => return Err(format!("bad team '{s}' (use -1..8 or red/blue/green/orange/purple/yellow/brown/pink/neutral/none)")),
    })
}

/// Shared on|off|get tail parser for the global switches.
fn parse_onoff_get(t: Option<&str>, what: &str) -> Result<Option<bool>, String> {
    match t.map(|s| s.to_lowercase()).as_deref() {
        None | Some("get") | Some("status") => Ok(None),
        Some("on") | Some("1") | Some("true") => Ok(Some(true)),
        Some("off") | Some("0") | Some("false") => Ok(Some(false)),
        Some(x) => Err(format!("{what}: expected on|off|get, got '{x}'")),
    }
}

fn parse_datum(s: &str) -> Result<u32, String> {
    let s = s.trim();
    let v = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(h, 16)
    } else {
        s.parse::<u32>()
    };
    v.map_err(|_| format!("bad datum '{s}' (use 0xHEX or decimal)"))
}

fn parse_f32(s: &str) -> Result<f32, String> {
    s.trim().parse::<f32>().map_err(|_| format!("bad number '{s}'"))
}

/// A box face, by name ("-x".."+z", "left"/"right"/"top"/"bottom"…) or index 0..5.
/// Index order matches `Obb::face_center`: 0..5 = -X, +X, -Y, +Y, -Z, +Z.
fn parse_face(t: &str) -> Result<u8, String> {
    let s = t.trim().to_lowercase();
    let idx = match s.as_str() {
        "-x" | "minusx" | "left" => 0,
        "+x" | "x" | "plusx" | "right" => 1,
        "-y" | "minusy" | "front" => 2,
        "+y" | "y" | "plusy" | "back" => 3,
        "-z" | "minusz" | "bottom" | "down" => 4,
        "+z" | "z" | "plusz" | "top" | "up" => 5,
        _ => s.parse::<u8>().map_err(|_| format!("bad face '{t}' (use -x|+x|-y|+y|-z|+z or 0..5)"))?,
    };
    if idx > 5 { return Err(format!("face index {idx} out of range 0..5")); }
    Ok(idx)
}

fn parse_vec3(toks: &[&str]) -> Result<[f32; 3], String> {
    if toks.len() < 3 {
        return Err("expected 3 numbers (x y z)".into());
    }
    Ok([parse_f32(toks[0])?, parse_f32(toks[1])?, parse_f32(toks[2])?])
}

fn parse_dir(s: &str) -> Result<[f32; 3], String> {
    Ok(match s.trim().to_lowercase().as_str() {
        "+x" | "x" | "east" => [1.0, 0.0, 0.0],
        "-x" | "west" => [-1.0, 0.0, 0.0],
        "+y" | "y" | "north" => [0.0, 1.0, 0.0],
        "-y" | "south" => [0.0, -1.0, 0.0],
        "+z" | "z" | "up" => [0.0, 0.0, 1.0],
        "-z" | "down" => [0.0, 0.0, -1.0],
        other => return Err(format!("bad direction '{other}' (use +x/-x/+y/-y/+z/-z)")),
    })
}

fn parse_objref(s: &str) -> ObjRef {
    if let Some(n) = s.strip_prefix('#') {
        if let Ok(i) = n.parse::<usize>() {
            return ObjRef::Index(i);
        }
    }
    ObjRef::Name(s.to_string())
}

/// Does this script step the edit history (`undo` / `redo` on any line)? The
/// interactive runner normally wraps a whole run in one undo snapshot; a run that itself undoes
/// must not, or its `undo` would only restore the snapshot just taken (a no-op).
pub fn steps_history(text: &str) -> bool {
    text.lines().any(|l| {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') || l.starts_with("//") {
            return false;
        }
        matches!(l.split_whitespace().next().map(|v| v.to_ascii_lowercase()).as_deref(), Some("undo" | "redo"))
    })
}

/// Parse one script line → command. `#`/`//` lines and blanks return Ok(None). Grammar is
/// documented in the script panel; kept forgiving (whitespace-tokenised, case-insensitive verbs).
pub fn parse_line(line: &str) -> Result<Option<EditorCommand>, String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
        return Ok(None);
    }
    let t: Vec<&str> = line.split_whitespace().collect();
    let verb = t[0].to_lowercase();
    let cmd = match verb.as_str() {
        "place" | "spawn" => {
            if t.len() < 2 {
                return Err("place <name|#idx> [at x y z | rel <datum> dx dy dz | onface <datum> <dir>]".into());
            }
            let obj = parse_objref(t[1]);
            let pos = match t.get(2).map(|s| s.to_lowercase()) {
                None => Placement::Camera,
                Some(k) if k == "at" => Placement::At(parse_vec3(&t[3..])?),
                Some(k) if k == "rel" || k == "relative" => Placement::Relative {
                    datum: parse_datum(t.get(3).ok_or("rel needs <datum> dx dy dz")?)?,
                    off: parse_vec3(&t[4..])?,
                },
                Some(k) if k == "onface" || k == "on" => Placement::OnFace {
                    datum: parse_datum(t.get(3).ok_or("onface needs <datum> <dir>")?)?,
                    dir: parse_dir(t.get(4).ok_or("onface needs a direction")?)?,
                },
                Some(k) if k == "camera" || k == "cam" => Placement::Camera,
                Some(other) => return Err(format!("unknown place target '{other}'")),
            };
            EditorCommand::Place { obj, pos }
        }
        "select" => {
            let a = t.get(1).ok_or("select <datum|all|none|box X1 Y1 Z1 X2 Y2 Z2 [add]>")?.to_lowercase();
            match a.as_str() {
                "all" => EditorCommand::SelectAll,
                "none" => EditorCommand::Deselect,
                "box" => {
                    if t.len() < 8 {
                        return Err("select box X1 Y1 Z1 X2 Y2 Z2 [add]".into());
                    }
                    let a = parse_vec3(&t[2..5])?;
                    let b = parse_vec3(&t[5..8])?;
                    let additive = t.get(8).map_or(false, |x| x.eq_ignore_ascii_case("add"));
                    EditorCommand::SelectBox { min: [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])], max: [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])], additive }
                }
                _ => EditorCommand::Select(parse_datum(t[1])?),
            }
        }
        "deselect" => EditorCommand::Deselect,
        "save" => {
            // `save` overwrites the open variant; `save as <path>` / `save <path>` writes elsewhere.
            let rest: Vec<&str> = t[1..].iter().copied().filter(|x| !x.eq_ignore_ascii_case("as")).collect();
            EditorCommand::Save { path: (!rest.is_empty()).then(|| rest.join(" ")) }
        }
        "saveas" => EditorCommand::Save { path: Some(t[1..].join(" ")) },
        "dup" | "duplicate" => {
            // dup [datum] [dx dy dz] — no datum = the current selection.
            let (datum, nums): (Option<u32>, &[&str]) = match t.get(1) {
                Some(a) if parse_f32(a).is_err() => (Some(parse_datum(a)?), &t[2..]),
                _ => (None, &t[1..]),
            };
            let off = if nums.len() >= 3 { parse_vec3(nums)? } else { [0.0, 0.0, 0.0] };
            EditorCommand::Dup { datum, off }
        }
        "count" => {
            let (type_filter, name_filter, label_filter) = parse_filters(&t[1..]);
            EditorCommand::Count { type_filter, name_filter, label_filter }
        }
        "delete" | "del" => EditorCommand::Delete(t.get(1).map(|s| parse_datum(s)).transpose()?),
        "move" => EditorCommand::Move {
            datum: parse_datum(t.get(1).ok_or("move <datum> dx dy dz")?)?,
            delta: parse_vec3(&t[2..])?,
        },
        "moveto" => EditorCommand::MoveTo {
            datum: parse_datum(t.get(1).ok_or("moveto <datum> x y z")?)?,
            pos: parse_vec3(&t[2..])?,
        },
        // coincident <datumA> <faceA> <datumB> <faceB> [centered]
        "coincident" | "mate" => {
            let usage = "coincident <datumA> <faceA> <datumB> <faceB> [centered] [noturn]  (face = -x|+x|-y|+y|-z|+z or 0..5)";
            let a = parse_datum(t.get(1).ok_or(usage)?)?;
            let af = parse_face(t.get(2).ok_or(usage)?)?;
            let b = parse_datum(t.get(3).ok_or(usage)?)?;
            let bf = parse_face(t.get(4).ok_or(usage)?)?;
            // trailing flags, any order: centered | noturn
            let flags: Vec<String> = t.get(5..).unwrap_or(&[]).iter().map(|x| x.to_lowercase()).collect();
            let centered = flags.iter().any(|f| f == "centered" || f == "centred" || f == "center" || f == "centre");
            // Turning the part to face the target is the default (a slide-only mate fails on
            // anything not already square); `noturn` keeps the slide-only mate.
            let turn = !flags.iter().any(|f| f == "noturn" || f == "norotate" || f == "slide");
            EditorCommand::Coincident { a, af, b, bf, centered, turn }
        }
        // #snap-array: `array line|axis ...`. Tokens are scanned by KEYWORD in any order, so
        // `array line from 0 0 0 to 20 0 0 step 1.5 align` and
        // `array line step 1.5 from 0 0 0 to 20 0 0` both parse.
        "array" => {
            let usage = "array line from X Y Z to X Y Z (count N | step WU) [align] | array axis <+x|-x|+y|-y|+z|-z> count N [step WU] [align]";
            let sub = t.get(1).map(|s| s.to_lowercase()).ok_or(usage)?;
            let (mut from, mut to, mut count, mut step, mut dir, mut target) = (None, None, None, None, None, None);
            let mut align = false;
            let mut i = 2usize;
            while i < t.len() {
                match t[i].to_lowercase().as_str() {
                    "from" | "start" => { from = Some(parse_vec3(t.get(i + 1..).unwrap_or(&[]))?); i += 4; }
                    "to" | "end" => { to = Some(parse_vec3(t.get(i + 1..).unwrap_or(&[]))?); i += 4; }
                    "count" | "copies" | "n" => { count = Some(parse_u32(t.get(i + 1).ok_or("array: count <n>")?)?); i += 2; }
                    "step" | "spacing" | "pitch" => { step = Some(parse_f32(t.get(i + 1).ok_or("array: step <wu>")?)?); i += 2; }
                    "align" | "aligned" | "turn" | "follow" => { align = true; i += 1; }
                    "selection" | "sel" | "all" => { i += 1; }
                    other => {
                        if let Ok(d) = parse_dir(other) {
                            dir = Some(d);
                        } else if let Ok(d) = parse_datum(other) {
                            target = Some(d);
                        } else {
                            return Err(format!("array: unexpected '{other}' — {usage}"));
                        }
                        i += 1;
                    }
                }
            }
            match sub.as_str() {
                "line" => {
                    let from = from.ok_or("array line needs `from X Y Z`")?;
                    let to = to.ok_or("array line needs `to X Y Z`")?;
                    if count.is_none() && step.is_none() {
                        return Err("array line needs `count N` or `step WU`".into());
                    }
                    EditorCommand::ArrayLine { target, from, to, count, step, align }
                }
                "axis" => {
                    let dir = dir.ok_or("array axis needs a direction (+x|-x|+y|-y|+z|-z)")?;
                    let count = count.ok_or("array axis needs `count N`")?;
                    EditorCommand::ArrayAxis { target, dir, count, step, align }
                }
                other => return Err(format!("array: unknown mode '{other}' — {usage}")),
            }
        }
        // #snap-array: the scripted Ctrl magnet. NOT spelled `snap`: that is already an alias of
        // `screenshot`.
        "snapto" | "facesnap" | "snapface" => {
            let mut target = None;
            let mut to = None;
            let mut axis = None;
            let mut i = 1usize;
            while i < t.len() {
                match t[i].to_lowercase().as_str() {
                    "to" | "onto" | "against" => { to = Some(parse_datum(t.get(i + 1).ok_or("snapto: to <datum>")?)?); i += 2; }
                    "axis" | "along" | "on" => { axis = Some(parse_dir(t.get(i + 1).ok_or("snapto: axis <+x|-x|...>")?)?); i += 2; }
                    "selection" | "sel" | "all" => { i += 1; }
                    other => { target = Some(parse_datum(other)?); i += 1; }
                }
            }
            EditorCommand::SnapTo { target, to, axis }
        }
        "settle" | "drop" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("all") | Some("selection") => EditorCommand::Settle(None),
            Some(_) => EditorCommand::Settle(Some(parse_datum(t[1])?)),
        },
        "rotate" | "rot" => {
            let datum = parse_datum(t.get(1).ok_or("rotate <datum> <x|y|z> <deg>")?)?;
            let axis = match t.get(2).map(|s| s.to_lowercase()).as_deref() {
                Some("x") => 0,
                Some("y") => 1,
                Some("z") => 2,
                _ => return Err("rotate axis must be x/y/z".into()),
            };
            let deg = parse_f32(t.get(3).ok_or("rotate needs degrees")?)?;
            EditorCommand::Rotate { datum, axis, deg }
        }
        "set" => {
            // `set selection <field> <value>` (aliases: all, sel) edits every selected object,
            // like the properties panel does with a multi-selection.
            let first = t.get(1).ok_or("set <datum|selection> <field> <value>")?;
            let fl = first.to_lowercase();
            let datum = if fl == "selection" || fl == "sel" || fl == "all" {
                SET_SELECTION
            } else {
                parse_datum(first)?
            };
            let field = t.get(2).ok_or("set needs a field")?.to_lowercase();
            let value = t.get(3..).map(|v| v.join(" ")).unwrap_or_default();
            EditorCommand::Set { datum, field, value }
        }
        "camera" | "cam" => {
            let sub = t.get(1).map(|s| s.to_lowercase());
            let c = match sub.as_deref() {
                Some("to") => CameraCmd::To(parse_vec3(&t[2..])?),
                Some("lookat") | Some("look") => CameraCmd::LookAt(parse_vec3(&t[2..])?),
                Some("spawn") => CameraCmd::Spawn,
                Some("frame") => CameraCmd::Frame,
                Some("get") | Some("pos") => CameraCmd::Get,
                Some("clearance") | Some("clear") => CameraCmd::Clearance,
                Some("standoff") | Some("unstick") | Some("push") => {
                    CameraCmd::Standoff(t.get(2).map(|v| parse_f32(v)).transpose()?)
                }
                Some("nudge") | Some("by") => CameraCmd::Nudge(parse_vec3(&t[2..])?),
                Some("orbit") => {
                    let target = match t.get(2).map(|s| s.to_lowercase()) {
                        Some(ref k) if k == "selection" || k == "sel" => OrbitTarget::Selection,
                        Some(_) if t.len() >= 6 && parse_f32(t[2]).is_ok() && parse_f32(t[3]).is_ok() && parse_f32(t[4]).is_ok() => {
                            OrbitTarget::Point(parse_vec3(&t[2..])?)
                        }
                        Some(_) => OrbitTarget::Datum(parse_datum(t[2])?),
                        None => return Err("camera orbit <datum|selection|X Y Z> <dist> [yaw] [pitch]".into()),
                    };
                    // The numeric tail starts after the target (1 token, or 3 for a point).
                    let base = match target { OrbitTarget::Point(_) => 5, _ => 3 };
                    let dist = t.get(base).map(|v| parse_f32(v)).transpose()?.unwrap_or(12.0);
                    let yaw = t.get(base + 1).map(|v| parse_f32(v)).transpose()?.unwrap_or(45.0);
                    let pitch = t.get(base + 2).map(|v| parse_f32(v)).transpose()?.unwrap_or(-25.0);
                    CameraCmd::Orbit { target, dist, yaw, pitch }
                }
                _ => return Err("camera <to X Y Z | lookat X Y Z | nudge DX DY DZ | spawn | frame | get | clearance | standoff [N] | orbit <datum|selection|X Y Z> <dist> [yaw] [pitch]>".into()),
            };
            EditorCommand::Camera(c)
        }
        "list" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            Some("palette") | Some("pal") => EditorCommand::ListPalette(t.get(2).map(|s| s.to_string())),
            Some("types") | Some("type") => EditorCommand::ListTypes,
            Some("maps") | Some("map") => EditorCommand::ListMaps(t.get(2).map(|s| s.to_string())),
            Some("variants") | Some("variant") => parse_variants(&t[2..])?,
            _ => {
                let (ty, nm, lb) = parse_filters(&t[1..]);
                EditorCommand::ListObjects { type_filter: ty, name_filter: nm, label_filter: lb }
            }
        },
        "objects" | "objs" => {
            let (ty, nm, lb) = parse_filters(&t[1..]);
            EditorCommand::ListObjects { type_filter: ty, name_filter: nm, label_filter: lb }
        }
        "types" => EditorCommand::ListTypes,
        "maps" => EditorCommand::ListMaps(t.get(1).map(|s| s.to_string())),
        "variants" => parse_variants(&t[1..])?,
        "mapid" | "map_id" => {
            let r = match t.get(1).map(|s| s.to_lowercase()).as_deref() {
                None => MapRef::Current,
                Some("current") | Some("here") => MapRef::Current,
                Some("variant") => MapRef::Variant(t.get(2..).map(|v| v.join(" ")).ok_or("mapid variant <path>")?),
                Some(_) => MapRef::Named(t[1..].join(" ")),
            };
            EditorCommand::MapId(r)
        }
        "get" | "info" => EditorCommand::Get(parse_datum(t.get(1).ok_or("get <datum>")?)?),
        "loadmap" | "openmap" => EditorCommand::LoadMap(t.get(1..).map(|v| v.join(" ")).filter(|s| !s.is_empty()).ok_or("loadmap <name|path>")?),
        "loadvariant" | "openvariant" | "loadmvar" => {
            EditorCommand::LoadVariant(t.get(1..).map(|v| v.join(" ")).filter(|s| !s.is_empty()).ok_or("loadvariant <path>")?)
        }
        "screenshot" | "shot" | "snap" => {
            // screenshot <path> [WxH | W H]
            let path = t.get(1).map(|s| s.to_string()).ok_or("screenshot <path> [WxH]")?;
            let size = match t.get(2) {
                None => None,
                Some(s) if s.contains(['x', 'X']) => {
                    let (a, b) = s.split_once(['x', 'X']).unwrap();
                    Some((a.trim().parse().map_err(|_| "bad width")?, b.trim().parse().map_err(|_| "bad height")?))
                }
                Some(a) => {
                    let w = a.trim().parse().map_err(|_| "bad width")?;
                    let h = parse_f32(t.get(3).ok_or("screenshot needs W H")?)? as u32;
                    Some((w, h))
                }
            };
            EditorCommand::Screenshot { path, size }
        }
        "wait" => EditorCommand::Wait,
        "screenfx" | "forgefx" | "fx" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("get") | Some("status") => EditorCommand::ScreenFx(None),
            Some("on") | Some("1") | Some("true") => EditorCommand::ScreenFx(Some(true)),
            Some("off") | Some("0") | Some("false") => EditorCommand::ScreenFx(Some(false)),
            Some(x) => return Err(format!("screenfx: expected on|off|get, got '{x}'")),
        },
        // Global switches + report.
        "scale" | "scaled" | "scaling" => EditorCommand::ScaledGlobal(parse_onoff_get(t.get(1).copied(), "scale")?),
        "shadowcasters" | "shadowcaster" | "casters" | "shadows" => {
            EditorCommand::ShadowCastersGlobal(parse_onoff_get(t.get(1).copied(), "shadowcasters")?)
        }
        "mapspawns" | "map_spawns" | "scnrspawns" | "mapspawn" => {
            EditorCommand::MapSpawns(parse_onoff_get(t.get(1).copied(), "mapspawns")?)
        }
        "outlines" | "outline" | "physoutlines" | "physics_outlines" | "blockers" => {
            EditorCommand::PhysicsOutlines(parse_onoff_get(t.get(1).copied(), "outlines")?)
        }
        // #h4-phys the two selected-object hull overlays (both games).
        "collision" | "coll" | "collisionhull" => {
            EditorCommand::CollisionOverlay(parse_onoff_get(t.get(1).copied(), "collision")?)
        }
        "physics" | "phmo" | "physicshull" => {
            EditorCommand::PhysicsOverlay(parse_onoff_get(t.get(1).copied(), "physics")?)
        }
        // `// #h4-expo-3` the Settings window (scripted frame-identity checks).
        "settings" | "settingspanel" | "lightingpanel" => {
            EditorCommand::SettingsPanel(parse_onoff_get(t.get(1).copied(), "settings")?)
        }
        // #dialogs test hook — drive the variant browser (open/save) from a script.
        "dialog" | "dialogdbg" => {
            let sub = t.get(1).map(|s| s.to_lowercase()).unwrap_or_else(|| "status".into());
            let a1 = t.get(2).map(|s| s.to_string());
            let a2 = t.get(3).map(|s| s.to_string());
            EditorCommand::DialogDbg { sub, a1, a2 }
        }
        // #wire-visible  the selection wireframe's depth mode (View menu).
        "wirexray" | "wire_xray" | "wirethrough" | "xraywire" => {
            EditorCommand::WireXray(parse_onoff_get(t.get(1).copied(), "wirexray")?)
        }
        "softceilings" | "softceiling" | "soft_ceilings" | "ceilings" | "mapfloor" | "floor" => {
            EditorCommand::SoftCeilings(parse_onoff_get(t.get(1).copied(), "softceilings")?)
        }
        "triggers" | "trigger" | "triggervolumes" | "trigger_volumes" => {
            EditorCommand::TriggerVolumes(parse_onoff_get(t.get(1).copied(), "triggers")?)
        }
        "hardfloor" | "hard_floor" | "mapbounds" | "worldbounds" | "bounds" => {
            EditorCommand::HardFloor(parse_onoff_get(t.get(1).copied(), "hardfloor")?)
        }
        "playablebounds" | "playable_bounds" | "playable" | "bspbounds" => {
            EditorCommand::PlayableBounds(parse_onoff_get(t.get(1).copied(), "playablebounds")?)
        }
        "undo" => EditorCommand::Undo,
        "redo" => EditorCommand::Redo,
        "newvariant" | "new_variant" => EditorCommand::NewVariant,
        "variant" | "mvar" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("get") | Some("info") | Some("show") => EditorCommand::VariantGet(t.get(2).map(|s| s.to_string())),
            Some("set") => {
                let field = t.get(2).ok_or("variant set <field> <value...>  (name, description, author, editor, category, budget, bounds, quotamin, quotamax, quota)")?.to_string();
                let value = t.get(3..).map(|v| v.join(" ")).unwrap_or_default();
                if value.is_empty() && !field.eq_ignore_ascii_case("description") { return Err(format!("variant set {field}: missing value")); }
                EditorCommand::VariantSet { field, value }
            }
            Some(x) => return Err(format!("variant: expected get|set, got '{x}'")),
        },
        "flags" | "flag" => EditorCommand::FlagsGet,
        // preview team <t> | preview color <c|inherit> | preview off | preview [get]
        "preview" | "hover" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("get") | Some("status") => EditorCommand::PreviewHover { team: None, color: None, off: false },
            Some("off") | Some("none") | Some("end") | Some("clear") => EditorCommand::PreviewHover { team: None, color: None, off: true },
            Some("team") => {
                let v = t.get(2).ok_or("preview team: expected a team (red..pink, neutral, none or -1..8)")?;
                EditorCommand::PreviewHover { team: Some(parse_team(v)?), color: None, off: false }
            }
            Some("color") | Some("colour") => {
                let v = t.get(2).ok_or("preview color: expected inherit or -1..7 / a colour name")?;
                let c = match v.to_lowercase().as_str() {
                    "inherit" | "none" | "team" | "-1" => -1,
                    x => match parse_team(x) {
                        Ok(b) if b < 8 => b as i32,
                        _ => return Err(format!("preview color: bad colour '{v}' (inherit or -1..7 / red..pink)")),
                    },
                };
                EditorCommand::PreviewHover { team: None, color: Some(c), off: false }
            }
            Some("where") | Some("rects") => EditorCommand::PointerSim(PointerSim::Where),
            Some("key") => {
                let name = t.get(2).ok_or("preview key <name>  (a, r, enter, escape, backspace, ...)")?.to_string();
                EditorCommand::PointerSim(PointerSim::Key { name })
            }
            Some("type") | Some("text") => {
                let text = t.get(2..).map(|v| v.join(" ")).filter(|s| !s.is_empty()).ok_or("preview type <text>")?;
                EditorCommand::PointerSim(PointerSim::Type { text })
            }
            Some("drag") => {
                let f = |i: usize, what: &str| -> Result<f32, String> {
                    t.get(i).ok_or("preview drag: expected <x1> <y1> <x2> <y2>".to_string())?.parse::<f32>().map_err(|_| format!("preview drag: bad {what}"))
                };
                EditorCommand::PointerSim(PointerSim::Drag { x1: f(2, "x1")?, y1: f(3, "y1")?, x2: f(4, "x2")?, y2: f(5, "y2")? })
            }
            Some(k @ ("move" | "simulate" | "click")) => {
                let f = |i: usize, what: &str| -> Result<f32, String> {
                    t.get(i).ok_or(format!("preview {k}: expected <x> <y>"))?.parse::<f32>().map_err(|_| format!("preview {k}: bad {what}"))
                };
                let (x, y) = (f(2, "x")?, f(3, "y")?);
                EditorCommand::PointerSim(if k == "click" {
                    PointerSim::Click { x, y }
                } else {
                    PointerSim::Move { x, y, frames: t.get(4).and_then(|v| v.parse().ok()).unwrap_or(4) }
                })
            }
            Some(x) => return Err(format!("preview: expected team <t> | color <c> | off | get | move x y | click x y | drag x1 y1 x2 y2 | key <name> | type <text> | where, got '{x}'")),
        },
        // #construct-h4: the Construct (CAD) tool, scripted. The viewport clicks themselves stay
        // clicks (`preview click x y`); this is the toolbar + panel state they read, plus
        // `construct anchors <datum>`, which is exactly what a click can snap to.
        "construct" | "cad" => {
            let usage = "construct [get] | on|off | op <guide|circle|square|coincident|anchor|mirror|line> | snap <all|corners|edges|faces|centres> | anchors <datum> | clear";
            let c = match t.get(1).map(|s| s.to_lowercase()).as_deref() {
                None | Some("get") | Some("status") => ConstructCmd::Get,
                Some("on") | Some("1") | Some("true") => ConstructCmd::Enable(true),
                Some("off") | Some("0") | Some("false") => ConstructCmd::Enable(false),
                Some("op") | Some("tool") => ConstructCmd::Op(t.get(2).ok_or(usage)?.to_lowercase()),
                Some("snap") | Some("mode") | Some("snapto") => ConstructCmd::Mode(t.get(2).ok_or(usage)?.to_lowercase()),
                Some("anchors") | Some("anchor") | Some("points") => ConstructCmd::Anchors(parse_datum(t.get(2).ok_or("construct anchors <datum>")?)?),
                Some("clear") | Some("reset") => ConstructCmd::Clear,
                Some(x) => return Err(format!("construct: unknown '{x}' — {usage}")),
            };
            EditorCommand::Construct(c)
        }
        // `bspwarn` / `bspwarn list`
        "bspwarn" | "bsp_warn" | "bspflags" => match t.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("list") | Some("get") => EditorCommand::BspWarnList,
            Some(x) => return Err(format!("bspwarn: expected `list`, got '{x}'")),
        },
        "pick" | "aim" => {
            // pick [delete] [radius <r>]
            let mut delete = false;
            let mut radius = None;
            let mut i = 1;
            while i < t.len() {
                match t[i].to_lowercase().as_str() {
                    "delete" | "del" => delete = true,
                    "radius" | "r" => {
                        radius = Some(parse_f32(t.get(i + 1).ok_or("radius needs a value")?)?);
                        i += 1;
                    }
                    _ => {}
                }
                i += 1;
            }
            EditorCommand::Pick { delete, radius }
        }
        "echo" => EditorCommand::Echo(t.get(1..).map(|v| v.join(" ")).unwrap_or_default()),
        other => return Err(format!("unknown command '{other}'")),
    };
    Ok(Some(cmd))
}

/// Parse `type <t> name <s> label <s>` filter key/value pairs (any order, any subset).
fn parse_filters(toks: &[&str]) -> (Option<String>, Option<String>, Option<String>) {
    let (mut ty, mut nm, mut lb) = (None, None, None);
    let mut i = 0;
    while i + 1 < toks.len() {
        match toks[i].to_lowercase().as_str() {
            "type" | "cached_type" => ty = Some(toks[i + 1].to_string()),
            "name" => nm = Some(toks[i + 1].to_string()),
            "label" => lb = Some(toks[i + 1].to_string()),
            _ => {}
        }
        i += 2;
    }
    (ty, nm, lb)
}

/// Parse the tail of a `variants …` command → ListVariants selector + optional name filter.
fn parse_variants(toks: &[&str]) -> Result<EditorCommand, String> {
    let (sel, rest): (VariantSel, &[&str]) = match toks.first().map(|s| s.to_lowercase()).as_deref() {
        None => (VariantSel::Current, &[]),
        Some("current") | Some("here") => (VariantSel::Current, &toks[1..]),
        Some("all") | Some("*") => (VariantSel::All, &toks[1..]),
        Some("map") => {
            let name = toks.get(1).ok_or("variants map <name>")?.to_string();
            (VariantSel::MapName(name), &toks[2..])
        }
        Some(idtok) => {
            // A bare hex/decimal id, else treat as a map name.
            if let Ok(id) = parse_u32(idtok) {
                (VariantSel::Id(id), &toks[1..])
            } else {
                (VariantSel::MapName(toks.join(" ")), &[])
            }
        }
    };
    let filter = if rest.is_empty() { None } else { Some(rest.join(" ")) };
    Ok(EditorCommand::ListVariants { sel, filter })
}

fn parse_u32(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).map_err(|_| format!("bad id '{s}'"))
    } else {
        s.parse::<u32>().map_err(|_| format!("bad id '{s}'"))
    }
}

// ===========================================================================================
// Program runner: comments, `foreach <source> as $var … end`, variable substitution. The same
// driver powers the interactive App and the headless script host — they only differ in how a
// single command line executes (`exec_line`) and how a foreach source enumerates (`enumerate`).
// ===========================================================================================

/// A parsed `foreach` source. The host turns this into concrete [`Item`]s.
#[derive(Clone, Debug)]
pub enum Source {
    Variants { sel: VariantSel, filter: Option<String> },
    Maps(Option<String>),
    Objects { type_filter: Option<String>, name_filter: Option<String>, label_filter: Option<String> },
}

/// Parse a `foreach` source string (e.g. "variants here", "maps forge", "objects type block").
pub fn parse_source(source: &str) -> Result<Source, String> {
    let t: Vec<&str> = source.split_whitespace().collect();
    match t.first().map(|s| s.to_lowercase()).as_deref() {
        Some("variants") | Some("variant") => match parse_variants(&t[1..])? {
            EditorCommand::ListVariants { sel, filter } => Ok(Source::Variants { sel, filter }),
            _ => unreachable!(),
        },
        Some("maps") | Some("map") => Ok(Source::Maps(t.get(1).map(|s| s.to_string()))),
        Some("objects") | Some("objs") | Some("object") => {
            let (ty, nm, lb) = parse_filters(&t[1..]);
            Ok(Source::Objects { type_filter: ty, name_filter: nm, label_filter: lb })
        }
        Some(other) => Err(format!("unknown foreach source '{other}' (use variants/maps/objects)")),
        None => Err("foreach needs a source".into()),
    }
}

/// One item produced by a foreach source.
pub struct Item {
    /// Substituted for the loop variable — a path for variants/maps, `0xDATUM` for objects.
    pub value: String,
    /// File stem (variants/maps) or object name (objects); substituted for `$stem`.
    pub stem: String,
}

/// Host that can execute a single command line and enumerate a foreach source.
pub trait ScriptRunner {
    fn exec_line(&mut self, line: &str) -> String;
    fn enumerate(&mut self, source: &str) -> Result<Vec<Item>, String>;
}

/// Run a whole program (multi-line). Returns the concatenated output log.
pub fn run_program(r: &mut impl ScriptRunner, text: &str) -> String {
    let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    let mut out = String::new();
    run_block(r, &lines, &mut out, 0);
    if out.is_empty() {
        out.push_str("ok\n");
    }
    out
}

fn push_out(out: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    out.push_str(s);
    if !s.ends_with('\n') {
        out.push('\n');
    }
}

fn run_block(r: &mut impl ScriptRunner, lines: &[String], out: &mut String, depth: u32) {
    if depth > 16 {
        push_out(out, "ERROR foreach nested too deep");
        return;
    }
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim().to_string();
        let low = t.to_lowercase();
        if t.is_empty() || t.starts_with('#') || t.starts_with("//") {
            i += 1;
            continue;
        }
        if low == "end" {
            // stray end — ignore
            i += 1;
            continue;
        }
        if low.starts_with("foreach ") {
            let header = &t[8..];
            let (source, var) = match header.rsplit_once(" as ") {
                Some((s, v)) => (s.trim().to_string(), v.trim().to_string()),
                None => {
                    push_out(out, "ERROR foreach <source> as $var");
                    i += 1;
                    continue;
                }
            };
            if !var.starts_with('$') {
                push_out(out, "ERROR foreach loop variable must start with $");
                i += 1;
                continue;
            }
            // Capture the body up to the matching `end` (depth-aware).
            let mut d = 1usize;
            let mut body: Vec<String> = Vec::new();
            let mut j = i + 1;
            while j < lines.len() {
                let lt = lines[j].trim().to_lowercase();
                if lt.starts_with("foreach ") {
                    d += 1;
                } else if lt == "end" {
                    d -= 1;
                    if d == 0 {
                        break;
                    }
                }
                body.push(lines[j].clone());
                j += 1;
            }
            if d != 0 {
                push_out(out, "ERROR foreach without matching 'end'");
                return;
            }
            match r.enumerate(&source) {
                Ok(items) => {
                    for (idx, item) in items.iter().enumerate() {
                        let subbed: Vec<String> = body
                            .iter()
                            .map(|l| substitute(l, &var, item, idx))
                            .collect();
                        run_block(r, &subbed, out, depth + 1);
                    }
                }
                Err(e) => push_out(out, &format!("ERROR {e}")),
            }
            i = j + 1; // skip the `end`
        } else {
            let o = r.exec_line(&t);
            push_out(out, &o);
            i += 1;
        }
    }
}

/// Substitute `$stem`, `$idx`, and the loop variable into one line. Fixed names first so a
/// user var like `$s` can't clobber `$stem`.
fn substitute(line: &str, var: &str, item: &Item, idx: usize) -> String {
    line.replace("$stem", &item.stem)
        .replace("$idx", &idx.to_string())
        .replace(var, &item.value)
}

#[cfg(test)]
mod obj_flag_grammar {
    use super::*;

    /// The flag verbs and fields parse as documented in the script help.
    #[test]
    fn flag_commands_parse() {
        assert!(matches!(parse_line("scale off"), Ok(Some(EditorCommand::ScaledGlobal(Some(false))))));
        assert!(matches!(parse_line("scale on"), Ok(Some(EditorCommand::ScaledGlobal(Some(true))))));
        assert!(matches!(parse_line("scale get"), Ok(Some(EditorCommand::ScaledGlobal(None)))));
        assert!(matches!(parse_line("shadowcasters off"), Ok(Some(EditorCommand::ShadowCastersGlobal(Some(false))))));
        assert!(matches!(parse_line("shadowcasters"), Ok(Some(EditorCommand::ShadowCastersGlobal(None)))));
        assert!(matches!(parse_line("mapspawns on"), Ok(Some(EditorCommand::MapSpawns(Some(true))))));
        assert!(matches!(parse_line("mapspawns off"), Ok(Some(EditorCommand::MapSpawns(Some(false))))));
        assert!(matches!(parse_line("mapspawns get"), Ok(Some(EditorCommand::MapSpawns(None)))));
        assert!(matches!(parse_line("mapspawns"), Ok(Some(EditorCommand::MapSpawns(None)))));
        assert!(parse_line("mapspawns maybe").is_err());
        assert!(matches!(parse_line("settings on"), Ok(Some(EditorCommand::SettingsPanel(Some(true))))));
        assert!(matches!(parse_line("settings off"), Ok(Some(EditorCommand::SettingsPanel(Some(false))))));
        assert!(matches!(parse_line("settings"), Ok(Some(EditorCommand::SettingsPanel(None)))));
        // #dialogs test hook parses its subcommand + up to two args.
        assert!(matches!(parse_line("dialog open"), Ok(Some(EditorCommand::DialogDbg { .. }))));
        match parse_line("dialog size 560 380").unwrap().unwrap() {
            EditorCommand::DialogDbg { sub, a1, a2 } => { assert_eq!(sub, "size"); assert_eq!(a1.as_deref(), Some("560")); assert_eq!(a2.as_deref(), Some("380")); }
            _ => panic!("expected DialogDbg"),
        }
        assert!(matches!(parse_line("dialog"), Ok(Some(EditorCommand::DialogDbg { .. }))));
        assert!(matches!(parse_line("wirexray on"), Ok(Some(EditorCommand::WireXray(Some(true))))));
        assert!(matches!(parse_line("wirexray off"), Ok(Some(EditorCommand::WireXray(Some(false))))));
        assert!(matches!(parse_line("wirexray"), Ok(Some(EditorCommand::WireXray(None)))));
        assert!(matches!(parse_line("wirethrough get"), Ok(Some(EditorCommand::WireXray(None)))));
        assert!(matches!(parse_line("outlines on"), Ok(Some(EditorCommand::PhysicsOutlines(Some(true))))));
        assert!(matches!(parse_line("outlines off"), Ok(Some(EditorCommand::PhysicsOutlines(Some(false))))));
        assert!(matches!(parse_line("outlines get"), Ok(Some(EditorCommand::PhysicsOutlines(None)))));
        assert!(matches!(parse_line("blockers"), Ok(Some(EditorCommand::PhysicsOutlines(None)))));
        // #h4-phys
        assert!(matches!(parse_line("collision on"), Ok(Some(EditorCommand::CollisionOverlay(Some(true))))));
        assert!(matches!(parse_line("coll off"), Ok(Some(EditorCommand::CollisionOverlay(Some(false))))));
        assert!(matches!(parse_line("collision"), Ok(Some(EditorCommand::CollisionOverlay(None)))));
        assert!(matches!(parse_line("physics on"), Ok(Some(EditorCommand::PhysicsOverlay(Some(true))))));
        assert!(matches!(parse_line("phmo get"), Ok(Some(EditorCommand::PhysicsOverlay(None)))));
        assert!(matches!(parse_line("softceilings on"), Ok(Some(EditorCommand::SoftCeilings(Some(true))))));
        assert!(matches!(parse_line("softceilings off"), Ok(Some(EditorCommand::SoftCeilings(Some(false))))));
        assert!(matches!(parse_line("mapfloor get"), Ok(Some(EditorCommand::SoftCeilings(None)))));
        assert!(parse_line("softceilings maybe").is_err());
        assert!(matches!(parse_line("triggers on"), Ok(Some(EditorCommand::TriggerVolumes(Some(true))))));
        assert!(matches!(parse_line("triggers"), Ok(Some(EditorCommand::TriggerVolumes(None)))));
        assert!(matches!(parse_line("hardfloor on"), Ok(Some(EditorCommand::HardFloor(Some(true))))));
        assert!(matches!(parse_line("bounds off"), Ok(Some(EditorCommand::HardFloor(Some(false))))));
        assert!(matches!(parse_line("hardfloor"), Ok(Some(EditorCommand::HardFloor(None)))));
        assert!(parse_line("hardfloor maybe").is_err());
        assert!(matches!(parse_line("playablebounds on"), Ok(Some(EditorCommand::PlayableBounds(Some(true))))));
        assert!(matches!(parse_line("playable"), Ok(Some(EditorCommand::PlayableBounds(None)))));
        assert!(parse_line("outlines sideways").is_err());
        // box select, corners in any order, optional `add`
        match parse_line("select box 3 4 5 -1 2 0 add") {
            Ok(Some(EditorCommand::SelectBox { min, max, additive })) => {
                assert_eq!(min, [-1.0, 2.0, 0.0]);
                assert_eq!(max, [3.0, 4.0, 5.0]);
                assert!(additive);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(parse_line("select box 0 0 0 1 1 1"), Ok(Some(EditorCommand::SelectBox { additive: false, .. }))));
        assert!(parse_line("select box 0 0 0 1 1").is_err());
        assert!(matches!(parse_line("undo"), Ok(Some(EditorCommand::Undo))));
        assert!(matches!(parse_line("redo"), Ok(Some(EditorCommand::Redo))));
        assert!(steps_history("select 0xD0000001\nundo\n"));
        assert!(steps_history("  REDO"));
        assert!(!steps_history("# undo\nmoveto 0xD0000001 1 2 3"));
        assert!(!steps_history("echo undo"));
        assert!(matches!(parse_line("flags get"), Ok(Some(EditorCommand::FlagsGet))));
        assert!(matches!(parse_line("preview team blue"), Ok(Some(EditorCommand::PreviewHover { team: Some(1), color: None, off: false }))));
        assert!(matches!(parse_line("preview team none"), Ok(Some(EditorCommand::PreviewHover { team: Some(0xFF), color: None, off: false }))));
        assert!(matches!(parse_line("preview color inherit"), Ok(Some(EditorCommand::PreviewHover { team: None, color: Some(-1), off: false }))));
        assert!(matches!(parse_line("preview color 3"), Ok(Some(EditorCommand::PreviewHover { team: None, color: Some(3), off: false }))));
        assert!(matches!(parse_line("preview colour pink"), Ok(Some(EditorCommand::PreviewHover { team: None, color: Some(7), off: false }))));
        assert!(matches!(parse_line("preview off"), Ok(Some(EditorCommand::PreviewHover { team: None, color: None, off: true }))));
        assert!(matches!(parse_line("preview"), Ok(Some(EditorCommand::PreviewHover { team: None, color: None, off: false }))));
        assert!(parse_line("preview color neutral").is_err());
        assert!(matches!(parse_line("preview move 10 20"), Ok(Some(EditorCommand::PointerSim(PointerSim::Move { frames: 4, .. })))));
        assert!(matches!(parse_line("preview click 10.5 20"), Ok(Some(EditorCommand::PointerSim(PointerSim::Click { .. })))));
        assert!(matches!(parse_line("preview where"), Ok(Some(EditorCommand::PointerSim(PointerSim::Where)))));
        assert!(parse_line("preview move 10").is_err());
        assert!(parse_line("preview sideways").is_err());
        assert!(parse_line("scale sideways").is_err());
        match parse_line("set selection shadow true") {
            Ok(Some(EditorCommand::Set { datum, field, value })) => {
                assert_eq!(datum, SET_SELECTION);
                assert_eq!(field, "shadow");
                assert_eq!(value, "true");
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse_line("set 0xD0000002 scaled default") {
            Ok(Some(EditorCommand::Set { datum, field, value })) => {
                assert_eq!(datum, 0xD0000002);
                assert_eq!((field.as_str(), value.as_str()), ("scaled", "default"));
            }
            other => panic!("unexpected {other:?}"),
        }
        // `set <d> scale <n>` (the X330 size field) is a `set`, not the global `scale` verb
        assert!(matches!(parse_line("set 5 scale 2.5"), Ok(Some(EditorCommand::Set { .. }))));
    }

    #[test]
    fn team_names() {
        assert_eq!(parse_team("green"), Ok(2));
        assert_eq!(parse_team("Red"), Ok(0));
        assert_eq!(parse_team("2"), Ok(2));
        assert_eq!(parse_team("-1"), Ok(0xFF));
        assert_eq!(parse_team("none"), Ok(0xFF));
        assert_eq!(parse_team("neutral"), Ok(8));
        assert!(parse_team("mauve").is_err());
        assert!(parse_team("9").is_err());
    }
}
