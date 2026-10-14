# Changelog — working draft

This file is the **running log** for the next release. Add an entry here as soon as a change
lands, while the detail is fresh; at release time the entries are folded into
`docs/changelog.html` (the published page) and into the GitHub release notes. See `RELEASING.md`.

Write entries for **users**, not for the diff: say what is different when they use the app, and
for a bug say what was wrong. Keep one line per change; put detail in a sub-clause rather than a
second bullet.

---

## not yet released

### Performance
- **Halo Map Studio no longer runs flat out while you are using something else.** "We essentially
  want to make it not a pain to run it alongside MCC" -- and it was a pain, because the editor had
  **no idea whether anyone was looking at it**. Alt-tab to the game and HMS carried on doing the
  whole thing: the full `update()`, the full offscreen 3D render, and a command submission of
  **4,349 draw calls, sixty times a second**, for a window behind the game. It now redraws a few
  times a second while its window is not the one you are using, and slower still when it is
  minimised or completely covered. Measured in production with the window left behind another
  app: **406 scene frames drawn and 391 skipped**, i.e. the viewport was being redrawn about ten
  times a second instead of at the display's rate. **A film keeps PLAYING and keeps its timing** --
  the playback clock runs on real elapsed time, so nothing drifts and nothing pauses; it is simply
  drawn less often. Settings > Graphics > **"Throttle when not in focus"**, on by default, turn it
  off if you keep playback running on a second monitor. The saving is not only frames: the particle
  simulation, the 3D render and the exposure metering are each skipped on a frame nobody can see,
  so a background frame costs the interface and nothing else.
- **A new install now renders at 100% instead of 200%, which is most of the answer to "are we
  hogging the GPU".** We were, and now there is a number for it: per-pass GPU timing of a real
  playback frame says HMS asked the graphics card for **10.7 ms a frame, 9.9 ms of it the scene
  pass** -- roughly two thirds of a high-end card at sixty frames a second, with a game running.
  The reason was the shipped render scale. HMS has always drawn the 3D scene at **200%: true 2x
  supersampling, four times the pixels**. Nobody chose that; it was simply the default. New
  installs render at 100%, and **200% is still there under Settings > Graphics > Render scale** for
  screenshots and calibration work, where four times the pixels is exactly what you want.
  **An existing install keeps whatever it already has** -- a saved 200% is indistinguishable from
  someone who picked it deliberately, so nothing is migrated and nobody's picture changes under
  them; pick Medium, or set Render scale yourself, to take the new default.
  Measured on the same clip at the same tick, before and after: **10.71 ms of graphics-card time a
  frame at 200% (9.90 ms of it the scene pass) against 3.93 ms at 100% (3.40 ms scene) -- 2.7x
  less GPU asked for, and 2.9x less of the scene pass.**
  Worth being clear about what this does and does not change: render scale is **resolution only**.
  It does not touch the calibrated lighting, exposure or tonemap laws, so the entire difference is
  how jagged an edge looks -- not the colour of anything.
  HMS's own frame RATE, separately, is limited by the processor rather than the card (33.9 ms of a
  34.6 ms Theater frame was CPU inside one `update()`), which is why the background throttle above
  is the thing that actually gets HMS out of a game's way. Both were true at once.
  (Headless captures are deliberately unchanged and still render supersampled: they never go
  through the Graphics settings, and every calibration reference image in this project was taken
  that way.)

- **Theater: "FPS is terrible in the theater playback", and "the program is EXTREMELY laggy now in
  theater mode".** It is, and it was CPU, not the graphics card: measured on the 7-minute
  16-player Forge clip with nothing else running, a playback frame took **34.6 ms (28.9 fps), with
  a 75.7 ms p99 and a worst frame of 157.8 ms** -- and **33.9 of those 34.6 ms were the CPU inside
  one `update()`**, so the GPU was idle waiting for it. Worse, the cost DEPENDED ON HOW FAR INTO
  THE CLIP YOU WERE: the same clip at 1:00 ran at 30.2 ms a frame and at 7:00 at 99.8 ms, which is
  the "it gets laggier the longer you watch" nobody could pin down. Theater now reports its own
  frame times (`film perf`, below) so none of this has to be guessed at again, and four costs are
  gone from the per-frame path:
  - **The animation PHASE was re-integrated over each body's entire past, every frame.** What
    drives a body's walk cycle is the ground it has covered, and that was computed by walking
    every position sample the recording holds for it from the start of its life up to the
    playhead -- once per drawn body, once per frame, so one frame's cost grew with the playhead.
    It is a running total, so it is now accumulated once and read with a binary search. The number
    that comes out is **bit-for-bit the same `f32`**, not an approximation, which matters more than
    the speed does: the phase decides the pose, and a Theater frame has to be a pure function of
    the playhead or the same tick reached two different ways would look different. A gate asserts
    the bit equality over 701 ticks of a track containing respawn teleports, repeated ticks, gait
    changes and stand-still stretches.
  - **Every corpse on the ground was fully animated before being overwritten.** A ragdolled body's
    joints come from the physics solver, but the clip was still sampled, cross-faded, overlaid and
    chained up the skeleton first -- and then the solver's pose overwrote all of it, line for line.
    Up to fourteen bodies lie on the ground at once, each paying for a pose nothing could see.
  - **Every posed body re-sorted its own skeleton, every frame.** Composing a pose needs parents
    before children, and that order was recomputed -- two allocations and a comparison sort of all
    ~130 nodes -- on every body on every frame, for an answer fixed when the model was decoded.
  - **A settled ragdoll re-scanned the whole placed-object table every frame.** The physics step
    returns immediately once a body is asleep, but the collision geometry it would have needed was
    gathered first: with fourteen corpses down, fourteen scans of every Forge piece on the map per
    frame, for a result nothing read.
- **Theater: `film perf` now reports the whole playback frame, as a distribution.** Mean, median,
  p95, p99 and worst frame for both the frame period and the CPU inside `update()`, then every one
  of seventeen stages of a Theater frame with its mean, its share, its own p99 and its worst --
  plus the five phases inside the body placement and the frame's draw-call census. "Laggy" is a
  complaint about the worst frames, not the average, and an average hides them. `film perf reset`
  empties the ten-second window so a change can be measured without restarting. Always on while a
  clip is open; the whole cost is seventeen clock reads a frame.

### Added
- **Vehicles now sit on the ground they are on, and they have a real engine.** "Vehicles are still
  kinda useless right now, the physics aren't right" -- and the two biggest reasons were that a
  Warthog drove up a hill **perfectly level**, and that its speed came from a number its tag never
  states. Both are now the engine's own law, read out of the game rather than guessed. **Every
  wheel gets its own ground ray**, and the hull's pitch and roll come from the plane through its
  contact points, which is exactly what the engine does -- so a Warthog noses up a ramp, leans
  across a cross-slope and drops a wheel into a dip. Each wheel then carries a **spring and a
  damper** built from the vehicle's own `ground depth` and `ground damp factor`, so the hull squats
  under its own weight, dives when you brake, leans in a corner and lands on its suspension
  instead of snapping flat: a Warthog rests 9 cm into 38 cm of travel, a Scorpion 25 cm into its
  own much softer 3 m, and neither number was typed in -- they fall out of the authored load
  shares. And the **drivetrain is real**: engine torque through the authored torque curves, into
  the gearbox, out as wheel speed. A Warthog pulls away in first, **shifts up through three gears**
  on the authored thresholds and times, slips its clutch for two ticks on each shift (which is the
  lurch), and its wheels keep turning when the hull cannot -- so wheelspin shows. The previous
  model lerped toward the driver seat's `max speed reference`, a field that is documented as a
  normalisation reference and not as a top speed. **The first cut of this launched the Warthog into
  the sky** -- "bro the warthog just flew into the air and the map glitched out" -- and it was three
  separate faults, none of them the spring being too strong: the per-wheel ground ray was cast from
  two world units ABOVE each wheel and excluded nothing, so it hit the Warthog's own collision hull
  and called the roof "ground"; the suspension's force cap was applied to the spring half only,
  leaving the damper unbounded and one-sided, which turned every bounce into an energy gain; and the
  velocity was rebuilt by adding the hull's forward speed back onto a vertical lane that still held
  all of it, which multiplied the vertical speed by a little over 1 every physics sub-step and
  compounded to 8e26 world units a second. The last one had been there all along and was invisible
  while vehicles were always level. All three are fixed and the suite now refuses to let a vehicle
  gain energy from its own suspension at all.
- **Vehicles fall at the right gravity.** They were on a fitted constant that had no basis in the
  game at all; they are now on the engine's own authored world gravity, which was found in the
  vehicle's own physics path -- the wheel's normal force reads it directly out of the block the
  engine itself calls "physics constants".
- **A vehicle that cannot be driven now says what is missing.** Sitting in a Banshee still does not
  fly it, but the caption now says it FLIES, names the block its lift and thrust are authored in,
  and says that nothing has been substituted -- so it reads as an unfinished lane rather than as a
  bug. The caption also reports the whole drivetrain and suspension: the gears, the traction limit,
  the suspension travel and the resting compression of every wheel.
- **Test mode: press R to reload.** The engine reloads a dry magazine by itself, so this only
  matters for topping up a partial one -- but reaching for R is muscle memory, and the key is now
  bound and rebindable with the rest of them under Settings > Controls & keyboard. Holding it
  cannot restart a reload already in progress, and it refuses a magazine that is already full.
- **Test mode has the game's own HUD.** "We also need the HUD for first person mode in the test
  mode" -- and it had none at all. Walk around in first person as a Spartan or an Elite and you now
  get the authored Reach display: your **shield and health bars**, your **grenade count**, the
  **reticule** of the weapon in your hands, and that weapon's own **ammo block with a magazine that
  counts down as you fire and reloads itself when it runs dry**. Drive a vehicle and you get **that
  vehicle's own HUD** instead of the one on foot, read from its own tag -- a Warthog's gunner seat
  included, which lives on a child attachment rather than on the Warthog. Every piece of it comes
  out of the game's own `chud` tags, laid out on the engine's own 1280x720 virtual screen, so it
  sits where Reach puts it and is coloured the way Reach colours it.
  - The numbers are the game's, not stand-ins: a Spartan has **70 shield and 45 health** and an
    Elite **79 and 50**, straight out of each biped's own model tag; an assault rifle holds **32
    rounds with 128 in reserve**, a magnum 8 with 16, a sniper rifle 4 with 8. The shotgun reloads
    **one shell at a time**, because its tag says it does.
  - Reaching those ammo numbers needed one piece of the weapon tag that had never been decoded --
    the editor knows where a weapon's `magazines` block is but not what is inside it, which is why
    Theater has always left the ammo meter blank. It is decoded now (96-byte elements, magazine size
    at +0x0A), confirmed three separate ways against the game's own code.
  - **Anything HMS cannot honestly source is still left blank rather than filled in.** The motion
    tracker is not drawn: test mode has one body on the map and no other players, so every blip
    would be invented. A driven vehicle's own shield and health meters are not drawn either --
    nothing damages a vehicle's condition yet, and feeding them the driver's shield would have
    claimed the Warthog was in the state you are. `play hud` in the script console prints every
    value on screen with where it came from.
  - Button prompts in the HUD are drawn with **your own bound key** rather than the console glyph
    the game ships, which Theater cannot do because a recording has no button to press.
- **Test mode: a driven vehicle now animates itself -- the wheels turn, the handlebars turn, the
  front wheels steer.** "Vehicles have their own unique animation they play when a biped is driving
  it, so handlebars, wheels etc would function" -- and now they do, from the vehicle's own tag. A
  driven vehicle was being drawn in its **bind pose**: it was placed from a render model, a render
  model carries no animation graph, so there was no clip to play and the wheels could not have
  turned however far you drove. Each vehicle's own graph is now installed and read, and it turns
  out a vehicle authors **no driving clip at all** -- it authors one `any:idle` loop plus a handful
  of two-to-five-frame **overlays** whose frames are not a timeline. They split in two, and the
  split was measured off the shipped clips rather than guessed from their names: the wheel overlays
  are a **full revolution whose last frame is identical to its first**, so they are indexed by a
  phase that wraps, and the wheels turn once per `2*pi*r` using **the radius the vehicle's own
  friction points author** (a Warthog's four are 0.250 wu, so one turn per 1.5708 wu travelled);
  the steering overlay is a **sweep that saturates**, so it is indexed by how far the wheel is
  turned, with the centre on the middle frame. Front and rear wheels are separate overlays and both
  are driven. Which end of a steering clip turns **left** is measured per vehicle from where the
  front wheel actually ends up, because the authored direction is not consistent across the fleet
  (the Mongoose's clip is mirrored relative to the Warthog's, the pickup's, the van's and the
  truck's). A **turret rider's gun now follows their look**: `vehicle:aiming` turned out not to be
  an animation but an **aim grid** -- ten frames that state, in their own channels, which direction
  each one means -- so it goes down the same path a biped's head and torso already use.
  **What is deliberately NOT animated, and why, is reported per vehicle instead of faked**: the
  suspension (there are no springs in the drive model, and the clips' own rest points are
  inconsistent, so there is no neutral frame to park them on), the chassis rock under acceleration
  (a damped oscillation, not a sweep -- what indexes it is not settled), and the Scorpion's tread
  covers (they need per-corner ground contact).
- **Test mode: walk up to a vehicle and get in, and the prompt is the game's own.** "We should be
  able to interact with and get in vehicles and drive them around" -- so you can. Stand beside a
  Warthog in test mode and Reach's own line appears, **"Hold E to drive Warthog"**, with *your*
  bound key in place of the game's controller glyph; press it and you are in the seat, with the
  seat's own animation, the seat's own camera and the seat's own look limits. The gunner's seat
  says **"Hold E to operate Warthog turret"**, the boarding seats say **"Hold E to board enemy
  Warthog"**, and a Ghost says **"Hold E to drive Ghost"** -- every word of it the game's, none of
  it ours. **Whether you can get in is the vehicle's own rule, not a distance we picked**: each
  seat authors how close you must be (0.90 to 2.50 world units; a Warthog driver's is 1.25), how
  far around the door's own facing you may stand (75 degrees for a Warthog, a full 180 for a
  Mongoose), how far off your own view may be (60 degrees), and how fast you may be moving
  relative to it (3.5 wu/s) -- so sprinting past a Warthog will not put you in it, and neither
  will standing on the wrong side. `E` is rebindable under Settings > Controls like everything
  else, and **the prompt always names whatever you bound it to**. Press it again to get out: you
  step off where you would have stood to get in, and you carry the vehicle's momentum with you,
  which is what Reach does.
- **Test mode: driving, with the vehicle's own numbers and an honest list of what is missing.** A
  Ghost does 5.00 wu/s forward and 2.50 in reverse, reaches it in about three quarters of a
  second, slides up to 4.00 wu/s sideways at a standstill and only 1.80 at full speed, and turns
  at 50 degrees a second when stopped falling to 16.7 at speed -- **every one of those is read out
  of the Ghost's tag**, not tuned by us. A Revenant is quicker (6.00) and a Wraith slower (4.00),
  in that order, because that is the order the tags say. **What is NOT simulated is stated rather
  than faked**: there is no Havok underneath, so a vehicle does not lean into a corner, does not
  lose traction, will not climb a kerb and cannot be flipped by a crash; there is no drivetrain,
  so the Warthog's four gears and their torque curves are read and reported but never integrated;
  and the hover vehicles ride at one height rather than on their three or six separate hover
  points. **The flyers are deliberately not driven at all.** Sit in a Banshee and the status line
  says so in as many words -- inventing flight dynamics for it would have been worse than leaving
  it alone.
- **A Warthog's turret seat is not on the Warthog**, and test mode knows it. The gunner's seat
  lives on a separate attached vehicle (and a Falcon has six of them), so walking up to the back
  of a Warthog offers you the turret and walking up to the side offers you the driver's seat, as
  it should.
- **The driven vehicle never touches your map.** Nothing about the placement is changed, moved or
  saved while you are driving it: the vehicle you are in is drawn separately and its placement is
  simply not drawn, so leaving test mode -- or closing the map, or crashing -- cannot leave your
  Warthog parked somewhere you did not put it. **Only the driver drives**: a passenger's or
  gunner's stick moves nothing, because the `driver` flag is per seat.
- **Reach's in-game text can be read at last, and that fixes more than vehicles.** HMS could not
  previously turn any of Reach's authored text into English -- the HUD drew nothing wherever a
  string was called for and said so. The key MCC files use for a Reach string turns out to be the
  string's own tag path and name joined together, and with that, **all 36 of the vehicle prompts
  the game authors resolve**, as do the HUD's own messages. The one thing still missing is the
  button-and-weapon icon font, so a prompt that is *only* an icon (a weapon's outline) is still
  left undrawn rather than replaced with a key name in the wrong place.
- **Test mode: play on the map you just built.** "We should look at the prospect of letting people
  PLAY on the maps they make in the editor... turn into a spartan/elite in first person and
  actually walk around their map." There is a new top-level **Play** menu: pick **Play as Spartan**
  or **Play as Elite** and you spawn on the map's own spawn point as that biped, in first person,
  holding an assault rifle, and you can walk, run, strafe, backpedal, **sprint**, **crouch**,
  **jump** and **swing** around everything you have placed. Your camera keys move your feet (so a
  rebind or a left-handed preset carries straight over), Space jumps, Ctrl crouches, Shift sprints,
  Alt walks slowly, V swings, and **Escape leaves and puts the editor camera back exactly where it
  was**. There is a third-person view in the same menu if you would rather watch your Spartan than
  look through its visor. Script: `play spartan | play elite | play off | play view first|third |
  play melee | play get | play phys | play fov <deg>`.
- **And none of how it moves was made up.** Every number comes out of the map's own tags, so it
  moves the way Reach moves rather than the way a physics demo moves: the walk, run (forward, back
  and sideways are three different speeds), sprint, crouch-walk and mid-air acceleration are the
  `matg` globals' own **player information** block; the jump velocity, the standing / running /
  crouching / crouch-walking **camera heights**, the crouch transition time, the capsule's height
  and radius and the **maximum slope** it can climb are that biped's own `bipd` fields; and gravity
  is the figure this project already measured off the engine's own recordings. `play phys` prints
  the whole table with the tag field each value came from, so you can check it. A Spartan and an
  Elite differ because their tags differ.
- **You cannot walk through your own blocks.** The player is a capsule swept against the level's
  collision triangles *and* the physics hulls of every object you have placed -- the same hulls a
  dead body lands on -- so a wall stops you, a ramp inside the slope limit carries you up, a ramp
  past it does not, and you step over a curb but not onto a crate. Looking down while crouched
  lowers the view to the height the tag authors for it, and a low roof will not let you stand up.
- **Test mode spawns where your camera is, not where the map's spawn point is.** "We also spawn
  way outside the actual map variant, we need to create the spartan biped right where the camera
  was." It does now: you fly to the part of your build you want to test, press Play, and the
  Spartan appears under the camera, **dropped to whatever is underneath it** and facing the way you
  were looking. The map's own spawn point is only a fallback -- on a Forge variant it can be
  nowhere near what you are building.
- **Test mode can put any of the map's weapons in your hands, and you can fire them.** `Play >
  Give weapon` lists every weapon in the map; the gun is posed by that weapon's own first-person
  animation graph, left-click fires it along the crosshair, and melee reaches as far as that
  weapon's own authored lunge range (the assault rifle's, the golf club's and the sword's all
  differ -- the numbers come from the weapon tag, not from a guess). Script: `play weapon
  <name|none>`, `play fire`.
- **Press T for third person.** "Can't see my feet at all" -- Reach itself draws no legs in first
  person, so the answer is the view from behind, and it was buried in a menu. It is on a key now
  (rebindable, and listed in Help > Keyboard shortcuts), the camera stands off the wall behind you
  instead of ending up inside it, and the Play menu says which key it is.
- **Test mode gives the viewport the whole window.** "Play mode also needs to hide the toolbar and
  editor panel and any other object panels and stuff." The editor column, both status bars, the
  floating toolbar and the menu bar are all hidden while you are playing -- **your panel layout is
  not changed**, only hidden, so leaving restores exactly what you had. Because the menu bar goes
  too, the way out is printed in the corner (`Esc  leave play mode`) and Escape is handled before
  any part of the interface can swallow it.
- **You no longer fall through the map in test mode, and three separate things were doing it.**
  "I also just fell straight through the entire map." First, the collision solver stopped
  subdividing a fast movement past a fixed limit, so each piece became longer than the capsule it
  was testing and the body stepped over thin floors instead of landing on them -- reachable after
  only ~3 seconds of falling if the app stutters. Second, the collision data was gathered for a
  box around you and only checked once per frame, so a long fall swept you through space whose
  surfaces had never been looked at. Third, and probably what started it: **a Forge object whose
  physics shape fails to load is solid in the picture and empty underfoot** -- you stand on it and
  drop through, and nothing said so. Test mode now reports what is under you (level surfaces and
  object hulls counted separately, plus how many nearby objects had no hull at all), and if you do
  end up below the map it puts you back where you spawned and tells you.
- **And a body that landed hard could get stuck inside the floor.** A rule meant for deciding
  whether you can step up onto a kerb was also being applied to the ground under your feet, so a
  hard landing read as "a ledge too tall to climb", refused to push you out, and left you 8 cm
  into the floor with nothing to stand on. That is also some of what made movement feel like it
  caught on simple things.
- **Walking on a slope in test mode no longer jitters.** "The physics for sure is very bad around
  inclines, it jitters and freaks out when I'm walking up any incline." It did, and the cause was
  one line: the rule that keeps a walking body on the ground compared how fast it was rising
  against a fraction of its jump speed -- but a body walking up a hill rises because the hill
  rises, so **any incline past about six degrees** read as a jump, the body was treated as
  airborne, gravity pulled it off the slope and the next frame put it back, sixty times a second.
  It also flickered the animation between running and the jump arc. A body now counts as leaving
  the ground only if it **jumped** or was **already in the air** -- the two things that actually
  do it. Walking *down* a slope had the mirror of the same fault: the downhill part of the motion
  was being thrown away every frame, so the body walked off the slope and fell back onto it.
- **Test mode is documented.** It had no section in the shortcuts page and the whole `play` family
  of script commands was undocumented, including the Use key -- which by the in-app Help window's
  own note means nobody had it. Both pages now cover it, and the in-app Keyboard shortcuts window
  lists the Use key.
- **In test mode the biped now turns on its feet and looks where you look, instead of rotating
  like a mannequin on a turntable.** "In third person, the biped doesn't like pivot on its feet at
  all, the whole biped seems to just rotate with the camera which is wrong" -- and "the third
  person unit also doesn't move their head/torso or anything as I look up or down at all, its
  stuck aiming straight ahead". Both were the same mistake: the drawn body was simply *set* to the
  camera's yaw, so the model had no facing of its own and nothing was left over to pose an aim
  with. The body now has its **own facing**, which follows where you are looking at the biped's
  **authored turning speed** (`moving turning speed`, 2.0 rad/s for a Spartan on `forge_halo`),
  and while you are standing still it does not move your feet at all until your look has pulled
  past the **authored stationary turning threshold** -- which is what that field is for, and what
  lets you look around without shuffling. Whatever your look is *ahead* of your feet by is handed
  to Reach's own **aim overlay**, the same clips and the same path Theater uses, so the head, chest
  and gun follow you up and down. Turning on the spot now asks for the graph's **turn-in-place**
  cycles, and the status line reports the look, the facing, the turn rate and the overlay request
  -- including when a biped's graph has no aim overlay at all, which is the one case where the body
  really is posed straight ahead.
- **You can walk up to a weapon lying on the map and pick it up.** "I also can't pick up any of
  the weapons on the map" -- and you could not: `Play > Give weapon` could put any of the map's
  weapons in your hands from a menu, but the ones the author had actually *placed*, lying on the
  ground where you can see them, could not be touched, which is the thing anyone tries first.
  Walk up to one and the game's **own prompt** appears -- the authored `pickup message` on the
  weapon itself (or `swap message` when your hands are full), resolved out of the map's localized
  strings with the button glyph replaced by **your** bound key, exactly as the vehicle prompt
  already does. It is the **same key as entering a vehicle** (E by default), which is the game's
  own arrangement, and a vehicle wins when you are standing in range of both. Picking up is the
  same code path as the menu, so the animation set, the melee lunge range and your first-person
  arms all move together. The weapon is **left where it was** -- test mode never alters your map,
  and a weapon you can pick up twice is a much smaller surprise in a test mode than one the editor
  has quietly deleted. One honest limit, stated in the app: **how close you have to be is our
  number, not the game's.** Reach authors no player pickup range anywhere in its tag definitions
  (the two near misses are an AI field and a door's auto-open radius), so it is 1 wu, measured from
  your body rather than your eye -- which is why crouching does not change what you can reach --
  and `play pickup` prints the number that decided.
- **Test mode no longer gets slower the longer you spin round.** Mouse look adds up, so a player
  who keeps turning one way walks the look angle off to a thousand radians and beyond -- and the
  routine that compares two angles wraps them a full turn at a time in a loop, so every angle
  comparison got a little more expensive for every revolution you had ever made, for the whole
  session. Nothing noticed until the body gained a facing of its own to compare against. The look
  angle is now kept in range where it is written, which is one line in one place rather than four
  hardened readers.
- **Test mode now works on Halo 3 maps at all.** Walking a Halo 3 map in test mode dropped you
  straight through the floor, every time: the one question test mode asks a scene -- "what level
  geometry is around the player" -- had an **empty default answer**, Reach and Halo 4 both answered
  it properly, and Halo 3 silently took the default. There was nothing wrong with the Halo 3 map;
  its collision geometry was already loaded and already being used for clicking, decals and the
  spawn camera, and test mode simply never asked it. The default answer is gone, so a future game
  cannot make the same omission without the build saying so.
- **Test mode's prompts are now actually on screen.** Test mode hides both bottom bars, so the
  "press E to drive the Warthog" line had nowhere to appear -- it was being written to a status bar
  that is not drawn, which from the player's seat is indistinguishable from the feature not
  working. Prompts are now drawn in the viewport, and always the one the key would actually act on.
- **A body that landed hard no longer sinks into the floor, and no longer bounces off it.** Dropped
  from a height onto a floor, the capsule came to rest **8 cm inside it** and never counted as
  standing, for as long as you left it. The rule that decides what you can climb onto measures the
  surface's height *above your feet*, and sinking into the floor pushes your feet down -- so the
  floor underneath read as a ledge too tall to step on, the push-out was flattened to sideways
  (which for a flat floor is nothing at all), and nothing ever lifted the body out. Getting out of
  a surface now takes precedence over deciding whether it could have been climbed, but only while
  there is **nothing to stand on** -- the condition matters, because a body pressed against a kerb
  is geometrically indistinguishable from one stuck in a floor, and without it a 3 cm kerb became
  climbable for a biped whose slope limit forbids it. A second fault was hiding behind the first:
  **a floor is two triangles**, and a body landing on their shared seam was reported twice and
  pushed out twice as far as it went in, leaving it floating 8 cm *above* the floor instead. Two
  contacts pointing the same way are now one constraint rather than two.
- **A script can now tell "still loading" from "no map open".** Every scene command answered "no
  map open" for the *whole duration* of a map load, which is indistinguishable from never having
  started one -- so an automated check could watch a load for four minutes and conclude it had
  made no progress at all, which is exactly what happened here. `loadstatus` (or `loadmap status`,
  or `wait`) now answers with one of four states -- **not-started / in-flight / done / failed** --
  and names the map and the reason. It only reads; it cannot change how a load behaves.
- **A `preview key` can now carry modifiers** (`preview key w down ctrl`), so a scripted or
  automated run can drive a control that IS a modifier -- test mode's crouch is Ctrl, its sprint is
  Shift -- and any Ctrl+key shortcut, neither of which could be exercised without a physical
  keyboard before.
- **Import your controls from MCC.** "We need to import the controls from MCC ideally, and there
  are also some MCC gamma, fov and other settings that we should import as well." Settings >
  **Controls & keyboard** now has an **Import from Halo: The Master Chief Collection** section. It
  finds MCC's own settings file on this computer -- in a Windows install and in every Steam /
  Proton prefix -- reads the controls for whichever title you pick, and **shows you a row-by-row
  before-and-after before it changes anything**. Thirteen of Halo Map Studio's actions have a real
  counterpart in MCC and are imported: the camera flies on MCC's own Forge monitor keys (movement
  for W A S D, Forge's "lift" and "drop" for up and down, Forge's "boost" for fly-faster), delete
  and duplicate come from Forge's delete-object and duplicate-object, and **test mode's jump,
  crouch, sprint and melee come from your own rebound gameplay keys** -- so walking around a Forge
  map feels like the game on the first try instead of after twenty minutes of rebinding. One
  button applies it, one button puts it all back, and **your MCC files are only ever read, never
  written**.
  - **It says what it could NOT bring over, instead of quietly leaving half your controls at the
    defaults.** MCC has 66 buttons and Halo Map Studio has 43 actions, and most of each set has no
    counterpart in the other -- MCC knows nothing about axis locks, nudges or undo, and Halo Map
    Studio has no grenades, vehicles or scoreboard. The import lists every one of them by name
    with the reason, including the near misses: Forge's grab-object is a *mouse* button where the
    move tool is a key, and MCC's Forge monitor always looks with the mouse so there is no
    hold-to-look button to import. A control that could not be translated is left exactly as it
    was and named in the list.
  - **It reads MCC's field of view, mouse sensitivity, brightness, Forge and Theater camera dials
    and volumes, and tells you what it found.** Those are shown rather than applied this time
    round, each with a line saying why. **Brightness is deliberately not imported**: Halo Map
    Studio's exposure and tonemapping are the engine's own laws, measured from the shipped shaders
    and calibrated against captures of the real game, and pushing a brightness slider into them
    would quietly move the rendering away from what Halo actually looks like.
  - Nothing is read at startup, on a timer, or behind your back -- not until you press the button
    in that section.
- **Theater: SOUND.** "We should also look at bringing in SOUND to this so the films aren't
  completely silent. So you have the map ambient sounds as well as weapon firing sounds etc."
  Films now play the map's **ambience bed** and **weapon fire**, at the levels and over the
  distances Reach's own sound tags author. There is a **Volume** slider with a **Mute** box in the
  Theater panel and in Settings, saved between sessions, and a line underneath saying what the
  audio device is doing.
  - **Nothing about audio can stop HMS starting or slow it down.** No sound device is opened and
    no sound bank is read until a film actually wants a sound; a machine with no working output
    ends up indistinguishable from one with the volume down (it says so in the Volume row rather
    than failing), and the banks, the codec tables and every decode happen on a worker thread, so
    a sound that is not ready yet is skipped rather than waited for. There is no setting to turn
    on and no environment variable.
  - **The audio is Reach's own, decoded from the game's own files.** MCC keeps every sample in
    `haloreach/fmod/pc/sfx.fsb` (712 MB, 16,162 samples, 48 kHz) as FMOD Vorbis, which stores no
    codec tables of its own -- those live inside `haloreach.dll`. HMS reads them out of the user's
    own installed copy and ships no codec data. Verified against ffmpeg's independent decoder:
    `ar_reach1.aif` decodes to exactly the **57,671 frames** the bank declares, 1.2015 s, peak
    **17.779802** against ffmpeg's 17.779800 and RMS 0.701947 against 0.701948 -- agreement to
    seven significant figures, and across mono, stereo and four-channel samples the mean difference
    is 0.006 to 2.6 **parts per million** of the stream's own level.
  - **Loudness is the engine's, not a guess.** Reach's banks are mastered hot: 79.3% of `sfx.fsb`
    peaks above full scale, median +9.2 dBFS and up to +41.2. What brings each sound down is a sum
    of authored dB -- the sound's playback `gain base`, its sound class's `gain bounds`, its scale
    modifier, the permutation's own trim and the game's `-2 dB` master -- clamped to the engine's
    own [-64, 0] dB, because Reach never plays anything above unity gain. Over `forge_halo`'s 2,353
    sound tags that sum has a median of **-20 dB**, which is exactly what the bank's levels need.
  - **Distance and stereo placement are ported from the shipped engine.** The attenuation curve is
    up to six knots from the sound's own `don't play` / `attack` / `minimum` / `sustain begin` /
    `sustain end` / `maximum` distances, with every segment interpolated **linear in reciprocal
    distance** -- not a `1/d` falloff, and not invented. The parameters come from a four-layer
    merge (sound class, then playback, then pitch range, then source) in which a **zero field means
    inherit**, which is why a sound whose own `maximum distance` is 0 still attenuates. The stereo
    placement is Reach's own four-speaker pan table.
  - **The ambience bed follows you between areas.** A map authors a pair -- Forge World has an
    exterior bed (`halo_ext1.aif`) and an interior one (`riverworld_interior1.aif`) -- and the
    engine picks by which cluster the listener is standing in, so standing inside a structure
    plays the indoor bed. HMS reads the same per-cluster acoustics index the engine does, picks
    the smallest cluster containing the camera, and swaps the bed when you cross between areas
    rather than playing the outdoor one everywhere or both at once. The bed is not positioned: it
    is a background bed, so it does not fade as the camera moves. Measured across 12 maps: 87
    acoustics rows, 86 resolve to a looping sound and all 86 reach a real bank sample.
  - **Weapons: the shot, the melee swing, the clang, the reload and the dry-fire click.** Each
    comes from a different place in the tags and all of them are measured over 450 weapons across
    8 maps, through to a real audio sample: **firing 396 of 396** (the barrel's firing effect, or
    an attached looping sound for the continuous weapons -- focus rifle, beams, turret loops,
    horns), **melee swing 225 of 225**, **reload 103 of 103** and **dry-fire click 213 of 213**.
    The ones a weapon does not author stay silent rather than borrowing another weapon's. Reload
    and melee turned out not to be on the weapon at all -- the magazine block's four reload
    references are null on every shipped weapon -- they are authored on the first-person
    animation graph, and each sound's own class says which is which.
  - **The melee clang plays for the dedicated melee weapons only, and that is the shipped files'
    limit, not a gap in HMS.** 230 of 450 weapons name a clang effect, but 168 of them name the
    shared `generic_melee_clash`, whose five takes are **not present in any of MCC's seven Reach
    sound banks** -- checked against every `.fsb` beside the game module, not just the two HMS
    reads. So a rifle-on-rifle clang cannot be played from these files by anyone; the 62 that can
    be are the sword, the hammer and their kin. The clang is also the one sound played without a
    position, because the event that states it carries no object reference.
  - **Which weapon a player is holding** comes from the recording plus the film's own game
    variant: a player reports their weapon only when it CHANGES, so the variant's spawn weapon
    covers the rest. Measured over 22 recordings, the share of shots whose weapon is known is
    **97% to 100% on most clips** (several at exactly 100%); two older clips whose variant leaves
    the spawn weapon unchanged sit at ~6%, and on those the unidentified shots are silent rather
    than guessed at.
  - **Melee now makes two sounds, and the second one was hiding in plain sight.** A swing plays
    the weapon's own melee sound from its animation graph; a swing that CONNECTS plays the sound
    of the thing that was hit, which the engine keeps in a field whose own description reads
    "this is the sound that is made when I am meleed". On a Spartan and an Elite that is
    `soft_organic_flesh`, and it is a **sound combiner** rather than a plain sound -- a resolver
    that only understands plain sounds finds nothing there and melee goes silent, which is
    probably why it has never been heard outside the game. The Forge prop set authors one each,
    so meleeing a pallet sounds like a pallet.
  - **Footsteps: the timing and the per-surface sounds are read and proven; the surface itself is
    the one thing still missing, and it is missing for a specific reason.** Reach does not put
    footsteps on a timer -- the animation says which foot lands and on which frame (a `left_foot`
    or `right_foot` frame event), which is where the left/right alternation and the cadence come
    from, and it stays right at any playback speed. Measured across 4 maps: **5,649 animation
    clips carry foot events, 12,420 events in all**, e.g. one Spartan clip landing right on frame
    42 and left on frame 48. The sound is not in the animation: it is in the biped's own material
    table, keyed by the surface, and a Spartan authors **424** of them (`hard` ->
    `run/concrete_h3/chief`, `brittle` -> `walk/glass/chief`, `hard_metal_solid` ->
    `walk/metal_solid/chief`) -- 5,247 of those rows reach a real audio sample with not one row
    carrying an unexpected kind of tag. What is missing is **which surface a recorded body is
    standing on**: the game's 201 surface names all read correctly, but the table that maps a
    map's geometry to them is not in the map's tag data at all (0 rows where the editor format
    puts it) -- in a packaged map it lives in a paged resource. So a footstep has its timing and
    its sound ready and callable, and nothing plays it yet. **A step timer would be audibly wrong
    and no default surface is invented** -- a Spartan's own "default material" row names no sound,
    so there is nothing authored to fall back to.
  - **So footsteps can arrive when you WALK before they arrive when you WATCH, and that is not a
    bug.** The two differ in one thing only: what is under the foot. A player walking in test mode
    has their capsule resolved against the map's real geometry every frame, so the surface is
    already known and a footstep can play on it immediately; a recorded body in Theater is a
    position in a file, and finding the surface under it means reading the collision data out of a
    paged resource first. The timing, the per-surface sounds and the call that plays one are
    shared, so Theater gains footsteps the moment that read lands and nothing else has to change.
  - On Linux this adds a dependency on the ALSA client library (`libasound.so.2`), which every
    desktop that can run HMS already has. HMS still starts and runs without a working audio
    device -- the Volume row says so -- but the library itself must be present, as it is on every
    distribution that ships PipeWire, PulseAudio, SDL or a browser.

- **Theater: projectiles, weapon firing effects and explosions.** "There are also projectiles and
  explosions and stuff that should be simulated... There are also weapon firing effects too, and
  projectiles also have attached effects." Most of it turned out to be in the recording already and
  simply never drawn:
  - **Grenades, rockets, missiles, needles and Falcon/turret bullets are replicated by the film and
    are now drawn on the path it states.** The engine's entity table has a `projectile` type (16)
    and HMS was decoding its positions all along -- but the recorded-object classifier was dropping
    every one of them through its catch-all arm, which claimed "the film's own Forge scene already
    draws this". That is true of a Forge platform and untrue of a grenade in mid-air. Nothing here
    is simulated: a projectile is placed where the recording puts it, with the basis it reports.
  - **A weapon's muzzle flash.** Each shot the recording states (`action_weapon_fire`) now plays the
    firing effect the weapon itself authors, once, at the tick it was fired, on the gun in that
    body's hands -- and on the gun's own markers, which the effect names itself (`primary_trigger`,
    `muzzle_flash`, `primary_ejection`). A named marker the model does not have is skipped and
    counted, never quietly moved to the weapon's origin.
  - **A projectile's travelling effect follows it, and leaves its trail behind.** A rocket's
    contrail and a grenade's smoke are emitted where the projectile WAS when each puff was born,
    not where it is now -- so the trail hangs in the air instead of being dragged along with the
    rocket. The path is the recorded flight, sampled.
  - **Explosions.** When a projectile's track ends, its own detonation effect plays there. A rifle
    bullet authors none, so none is invented for it.
  - **A rifle round is deliberately NOT simulated, and the caption says why.** A Reach film records
    one player's look direction and zero-fills every other slot's, so for all but one body it does
    not state which way the shot went. Firing a tracer along the body's yaw would have looked like
    a feature and been a lie, and a visibly horizontal one. Every bullet's own definition -- its
    initial velocity, its gravity scale, its range -- is read and in hand for the day a direction
    exists.
- **Theater: two buttons that jump to the next or previous ASSASSINATION and put the camera on the
  victim.** "They're like too quick or something so you basically never see them" turned out not to
  be a speed problem at all -- the shipped moves are authored at 75 frames and HMS plays them
  across the full 2.5-second window, which is exactly the game's own timing. The problem was that
  there is no way to *find* one: a seven-minute clip holds about three, each lasting two and a half
  seconds somewhere in twenty-five thousand ticks, and the camera is almost always pointing
  somewhere else when one happens. So the jump does both halves -- it parks the playhead
  three-quarters of a second **before** the kill lands, so you see the attacker close in rather
  than arriving as the move finishes, and it moves the camera onto the victim in third person,
  which is the angle that shows the move (first person would put it off-camera behind their own
  head). It does not wrap at the end, because silently looping back to the start of a 47-minute
  clip reads as the button having done nothing. The buttons sit in the transport row next to the
  frame step, they are greyed out with an explanation on a clip that records none, and the status
  line names the time it jumped to -- saying so when the recording did **not** state the kill was
  an assassination and HMS derived it, the same distinction drawn everywhere else. Also
  `film assassination next|prev|list` from a script, so a capture can frame a kill unattended.

### Fixed
- **The test-mode HUD's shields, health, grenades and ammo now draw the way Reach draws them.**
  Reported: "the HUD isn't being decoded right, the shields and health don't look right, and the
  crosshair and grenades also aren't right either." One cause ran under most of it: a HUD widget
  carries a **shader type** (twenty-one of them, seven being meters) and an **alpha blend mode**
  (eighteen), and HMS read neither -- it drew every widget as a plain tinted sprite blended
  normally.
  - **The shield bar was being painted out by its own border.** The border is authored
    **additive**, sits at exactly the same rectangle as the bar and is drawn after it, so blended
    normally an opaque dark-blue plate covered the bright blue bar completely. Additive now draws
    additive and the bar survives.
  - **The assault rifle's magazine bar was not drawn at all.** It was skipped for "no authored
    colour", and a meter takes its colour from its own art rather than from a colour slot -- 158 of
    the 294 widgets that name that slot leave it blank. That weapon's HUD has no separate ammo
    number, so the bar was the entire magazine display and the screen showed only the reserve.
  - **The second grenade slot had no count.** A biped authors one grenade type and one count, which
    says it carries none of any other, so the other type now reads 0 instead of being left blank
    and both slots have a number.
  - Whether a widget is a meter is now read from the tag instead of guessed from its name.
  - **Still wrong, and known:** both grenade slots draw the same icon, because the art is resolved
    once per widget rather than once per slot.
- **`play hud` now explains the HUD rather than just listing its values.** Per widget: the shader,
  the blend mode, the bitmap and sprite it resolved to, where it landed as a fraction of the
  screen, its meter fill, its colour, whether it drew -- and **whether something drawn after it
  covered it up**. The shield fault was a widget that drew perfectly and was then buried, which no
  count of drawn widgets can show.
- **Importing your MCC settings now brings your field of view across, instead of only telling you
  what it is.** "There are also some MCC gamma, fov and other settings that we should import as
  well" -- and the import did read your FOV, show it to you, label the row *"not imported yet"*,
  and change nothing. An honest note about a missing feature is still a missing feature. Your
  first-person FOV now comes across with the controls, and **Undo the import puts your old one
  back** along with the keys. MCC's units are already Halo Map Studio's own and the range is the
  same 60-120, so the number carries across exactly with no conversion. **A value outside that
  range is refused rather than clamped** -- it means the file is not saying what we think it is,
  and quietly clamping it would import a number you never chose. Brightness is still read and
  shown without being applied, deliberately: the exposure and tonemap are engine-exact laws
  calibrated against engine captures, and feeding a slider offset into them would undo that.
- **Test mode: the legs no longer slide, and they are now tied to the ground you actually cover.**
  "Its very slidy as well especially crouching and strafing." **Two separate faults in one line of
  code.** First, the walk cycle's position in its stride was computed from your TOTAL distance
  travelled times a rate -- which only holds still while the rate never changes, and the rate
  changes every time you go from a walk to a run or stand up from a crouch. With forty world units
  of walking behind you, letting go of the walk key moved that product by tens of whole cycles, so
  the pose jumped to an unrelated point in the stride and a foot mid-step snapped somewhere else.
  The stride is now **accumulated frame by frame** from the ground each frame covers, so it is
  continuous across any change. (Theater had exactly this bug once and was fixed the same way;
  test mode had reintroduced it.) Second, the RATE itself came from a speed measured across film
  recordings per gait -- which carries no stance, so **a crouching Spartan was given the standing
  clip's speed**, and whose "run forward" figure (3.52 wu/s) is in fact this species' authored
  SPRINT speed (2.200 x 1.623 = 3.571), because most players in a recording are sprinting. It now
  comes from `matg @player information`, per species, per stance and per direction -- the same
  authored cap the controller holds you to, so the two cannot disagree. Measured against the
  shipped clips on `forge_halo`, the old law played them at **11.6 fps walking, 18.8 running, 38.8
  strafing, 7.7 crouch-walking and 17.4 sprinting, against the authored 30** -- crouching 3.9x too
  slow and strafing 1.3x too fast, which is the complaint word for word.
- **Test mode: standing still no longer makes the biped twitch once every few seconds.** "The idle
  animation also seems to play through then skip/shift the biped for some reason." It did, and the
  cause is in the shipped animation: **Reach's idle clips are not loops.** Posing every frame of
  the Spartan's own graph and measuring how far the busiest bone moves between neighbouring
  frames, `combat:rifle:idle` holds still to within 0.0006 wu for 155 of its 156 frames and then
  its last frame sits **0.0426 wu from its first** -- 73 times an ordinary frame of motion -- and
  the crouching idle is **0.183 wu out, 477 times**. (A clip that really does loop, like the run
  cycle, wraps at 1.4x an ordinary frame; the turn cycle at 0.6x. And the tag cannot tell you
  which is which -- `loop frame index` reads 0 on all of them.) HMS was looping them anyway, so
  the body drifted for five seconds and then snapped back. Clips that run on the clock now play
  **forward and then backward** instead of wrapping, which cannot pop whatever the clip's shape
  is. An idle now breathes over ten seconds instead of five; nothing else changes. This is HMS's
  choice, not the engine's -- the engine presumably layers authored fidget animations we do not
  play yet.
- **Test mode: turning on the spot no longer jolts the legs.** "When turning, its better than it
  was, but its also doing a sudden jolt of the legs instead of cleanly interpolating the legs as
  we turn." **Two causes, both now fixed.** The turn-in-place animation was driven by the WALL
  CLOCK while the body turned at its own authored rate, so the legs stepped at one speed and the
  body swung at another -- and because the clock is the session's, a turn was *entered* at
  whatever point in the step the clock happened to be on (measured: frame 14 of 18). It is now
  driven by **how far you have actually turned**, one cycle per `turning speed x frames / 30`
  radians, starting at the beginning of the step -- the same idea as the stride following the
  ground. Separately, the **cross-fade between any two animations was moving the whole skeleton at
  a constant speed** set by how different the two poses are rather than by anything the body was
  doing: measured on a real walk, an ordinary frame moves the busiest bone 0.033 wu while the
  first frame of a turn-to-idle fade moved it **0.255 wu**, a 7.7x lurch held for the whole
  0.18 s of the blend. The blend is now **eased in and out**, so it starts and ends at the speed
  of the clips it sits between. Same length, same endpoints. **This one also smooths Theater**,
  where every body uses the same cross-fade.
- **Test mode: the weapon is in your first-person hands.** "Also the weapon isn't even in the
  first person hands, we had this figured out in the theater where we held the weapon properly."
  Quite right, and that was the whole fault: Theater knew how and test mode did not, because the
  step lived inside Theater's own playback code. A Reach first-person graph is ONE skeleton over
  both the arms and the gun, and in it the gun's root is a CHILD of a hand -- so the clip plays a
  grip offset for it, not a position. Composed through the weapon model's own skeleton, where that
  root has no parent at all, the gun lands at the rig's origin: measured on the assault rifle, it
  sat **0.088 wu from the `r_hand` holding it, nearest the rig's base node rather than the hand**.
  There is now one implementation of the grip, used by Theater and by test mode, and the gun sits
  0.026 wu from the hand -- the authored grip offset, inside the fingers.
- **Test mode: one biped in first person, with legs.** "There is also a duplicate biped model in
  first person now." There was: the previous round made your own body visible so you could see
  your feet, and it drew the WHOLE third-person Spartan on top of the first-person arms, so you
  had two pairs of arms in front of you. Reach authors a model for exactly this and the globals
  name it -- `@player representation`'s **first person body** -- and on the Spartan it is 5 meshes
  over the regions `decals` and `flair_knees`: legs and hips, no head, no arms, no torso, on the
  same skeleton as the full biped. First person now draws that, posed from your body's own
  locomotion clip at the same frame, and parks the third-person body; third person does the
  reverse. **The legs stay -- that was the point -- and the duplicate is gone.** Both species'
  first-person bodies are legs and hips: measured from each model's own geometry bounds, the
  Spartan's stops **0.138 wu below the eye it is seen through** and the Elite's **0.208 wu**
  below, against full bipeds that reach 0.76 and 0.84 wu. So you never see the inside of your own
  chest, in either species.

#### Known gaps, test mode
- **Walking forward and strafing at the same time is drawn with the forward animation.** At a 45
  degree travel angle you are moving diagonally but the legs run straight ahead, and it shows.
  This is a limitation rather than a bug we have not noticed: **Reach BLENDS its forward and
  strafe animations together** over a direction-and-speed space, and HMS picks one clip. (It is
  not a flicker -- traced frame by frame, the choice is stable: four animation changes in 270
  frames of walking, strafing and going diagonal, with the diagonal stretch holding one clip for
  all ninety of its frames.) A proper two-dimensional locomotion blend is the fix and it is not
  built yet.

- **Test mode: a Warthog no longer pivots on a dime, and the camera no longer steers it.** "I turn
  and the whole warthog pivots around on a dime, the game does NOT do that... The camera is also
  wrong, its not using the vehicle's actual camera." **Those were one bug.** The steering was being
  commanded by *where you were looking*: the road-wheel angle was slewed toward the camera's own
  heading and pinned at full lock until the vehicle had come round to face it, so looking left threw
  the Warthog left and the camera could never be aimed independently of the driving. The kinematics
  underneath were already right -- yaw rate from the road-wheel angle over the **wheelbase measured
  from the vehicle's own friction points** -- which is why it looked like broken physics when it was
  a broken control. **A/D now steer and the camera is free.** A parked Warthog at full lock no
  longer turns at all (the wheels do), doubling the speed doubles the rate of turn, and reversing
  with the wheel held turns the other way, which falls out for free. Each vehicle now gets **the law
  its own authored physics type asks for** instead of all of them sharing one: `human_jeep` (the
  Warthog, Mongoose, pickup, van, truck, cart, forklift) steers its wheels; `human_tank` (the
  Scorpion) turns through its **tracks**, so it may rotate where it stands, over its own measured
  1.74 wu track width; `alien_scout` (the Ghost, Wraith, Revenant) turns its hull toward your look,
  which is what it really does. A vehicle whose law could not be measured now **says so on screen**
  rather than quietly getting the hover law, which is exactly how a Warthog came to turn like a
  Ghost. And a **turret** -- 18 of the 36, including all four of the Warthog's gun positions -- now
  says it does not drive *because it is bolted down*, instead of reporting itself as unimplemented.
- **Test mode: a rider no longer stands bolt upright in the seat.** "I stand straight up instead of
  sitting in the seat." The seat animation was being chosen by **guessing from the vehicle's file
  name** -- a substring match against a hand-written list of sixteen -- and anything the list did
  not recognise got *no* seat animation at all, which left the body in the state the walking
  controller had given it: standing, and (because a seated body stands on nothing) technically
  airborne. The seat itself **authors the animation's name**, so that is what is used now, for all
  138 seats, and a guess about a file name can no longer override what the tag says. Where a rider
  genuinely has no seat clip the caption now **names the vehicle and the seat** so it is clear which
  one -- and some of those are correct and permanent: every hijack seat is authored per species (the
  Elite has the Warthog's, the Spartan does not), and the Wraith's and Sabre's driver seats have no
  rider animation in the game at all.
- **Your first-person arms, legs and gun are lit the way the engine lights them.** "Most weapons,
  at least in first person, don't quite look right" -- and they were lit by a different rule from
  the same weapon in the third-person biped's hands. Reach has no first-person lighting rig: the
  arms and the legs are drawn as the PLAYER'S BIPED and the gun as the WEAPON, and each takes the
  per-object-type lighting variables the map authors for that type (on Forge World, bipeds and
  weapons get a minimum-luminance floor and a contrast lift that scenery and Forge blocks do not).
  The first-person models are bare render models, and HMS was reading a render model's header as
  if it were an object's -- which made the Spartan arms "object type 22918, bounding radius
  0.0000000000000000000000000000000000000000000028", off the end of the map's table, so they fell
  back to the generic defaults and got neither the floor nor the lift. The first-person rig now
  resolves to the object it belongs to, so it is lit exactly like the body holding it.
- **Getting into a vehicle no longer leaves a second copy of it standing there.** Riding draws the
  vehicle you are in as a moving character and hides the placed one, and the hiding half simply
  never happened: the object pipeline stops early when nothing it watches has changed, and *which
  objects are hidden* was not one of the things it watched. Mounting a Warthog therefore changed
  nothing the pipeline could see, so it kept serving the previous frame -- placement included --
  and the ride lane drew a Warthog on top of it. The hidden set is now part of what the pipeline
  checks, so starting and ending a ride each cost exactly one rebuild and neither leaves a ghost
  behind.
- **Test mode drew no biped at all.** The player walked around as a disembodied camera. The
  function that poses characters and builds their meshes is only reached from two places -- the
  scene rebuild, and Theater's own tick -- and neither runs while you are simply walking around a
  map: the scene rebuild takes a fast path and stops early as soon as the Forge objects have not
  changed, which is every frame once you are just moving. So test mode created the biped and then
  never asked for it to be drawn.
- **Leaving test mode no longer teleports the camera back.** It used to return you to wherever you
  were standing when you pressed Play; now the camera stays where you walked to, so you can go and
  look at something, step out and edit it there. Entering already worked this way round -- you
  spawn at the camera rather than at the map's own spawn point -- so both transitions are now
  continuous and nothing jumps. The field of view still returns to the editor's, because that is a
  property of the first-person rig rather than a place you navigated to.
- **The HUD can finally read the game's own words.** Every piece of text the heads-up display
  authors is stored as an id the game looks up in a translated table, and HMS had no way to perform
  that lookup -- so it drew none of them and listed them instead. The key the game uses has now
  been worked out and checked against 36 of 36 authored vehicle prompts, so that text is available.
  Three things can happen to such a piece of text, and HMS now tells them apart instead of lumping
  them together as "cannot read": **ordinary words are drawn**; something that is **purely a
  symbol** -- the weapon outline in the ammo block is a single character from a font that ships
  with the game and not with the editing kit -- is still left out, because there are no words
  underneath it and a placeholder would cover the spot the outline belongs in; and a **"hold this
  button to..." prompt is deliberately left out over a recording**, because there is no button for
  someone watching a film to hold and the instruction would invite an action that does nothing.
  Each is now reported with its own reason. The same text *is* drawn in the new test mode, with
  your own key in place of the console button symbol, because there the action is real.
- **Opening a saved film when its map was not already loaded did NOTHING but load the map** --
  no Forge scene, no playback, and a status line saying the film "was closed: a variant was
  opened". Opening the same clip a second time worked, which is the tell: by then the map was up.
  The film's own base map has to be loaded before its Forge scene can be drawn, and that load
  **streams over many frames** -- so the check immediately after starting it was reading the
  previous map's id, concluding the wrong map was loaded, and giving up before it had queued the
  scene or started decoding the recording. There are three states, not two: the map is already
  there, the map is on its way, or the map is not installed -- and the last two were being treated
  as the same thing. The clip's scene and its decode are now **deferred** until the load lands
  rather than abandoned, the same way a Forge variant opened for a not-yet-loaded map already
  waits for it. Opening a film from a cold start now just works.
- **Theater drew every full-speed run as a WALK, and could never draw a sprint at all.** The
  speeds that decide which animation a body plays were measured from the recordings themselves —
  2.7 million samples — and the measurement was sound, but it was read the wrong way round. The
  histogram's dominant group, which the old code took to be "players running", is actually
  **players sprinting**, because in Reach nearly everyone holds sprint. The engine's own tags say
  so outright: a Spartan runs at 2.200 world units per second and sprints at 3.571 (an authored
  multiplier of 1.623), and an Elite 2.600 and 3.640. So the walk/run split sat **above** a real
  full-speed run, and the sprint threshold sat **above the fastest speed any player can reach on
  foot** — which is why a sprint has never once appeared in a clip. The thresholds now come from
  those authored speeds, so a run looks like a run and a sprint finally plays. What makes the
  authored numbers trustworthy rather than just different: the sprint multiplier is a separate
  field in the tag, and multiplying the two lands precisely inside the peak the recordings show.
- **A film would sometimes open, draw its Forge scene, and then never play — with nothing on
  screen to say why.** Roughly one open in three, and opening the clip again usually worked, which
  is what made it so hard to pin down. Two separate faults, both now fixed.
  **The first was a mistaken identity.** Drawing a clip's own Forge scene goes through the same
  machinery as opening a map variant by hand, and opening a variant deliberately closes any clip
  you are watching — a recording's bodies are a statement about the scene it was recorded in, and
  leaving them standing in somebody else's map draws them at coordinates that mean nothing. The
  clip's own scene was exempted from that by a single flag meaning "the next one of these is
  mine", which is only true while exactly one is in flight. Two routinely are: the moment a map
  finishes loading, HMS also re-opens the variant you had open last. That one arrived first, spent
  the clip's exemption, and the clip's own scene then arrived looking like a variant you had
  opened by hand — so the film was closed and its half-finished decode thrown away. The exemption
  now names the specific file it belongs to, so nobody else can spend it, and a variant you really
  did open still closes the clip as it should.
  **The second was that the failure could not announce itself.** If the decoder ever stopped
  without returning anything, that was treated as "nothing to report" and silently forgotten —
  so the only symptom was a recording that never appeared, with no message, no log line and
  nothing to act on. The decoder now reports its own failures, and a decoder that dies without
  saying anything gets a status line of its own instead of silence.
- **Theater: "the first person arms look really weird in 120 FOV."** They were drawn through the
  WORLD camera, and the engine does not draw them that way. Reach renders the first-person model
  in its own pass at **88% of the world camera's vertical field of view** -- confirmed off the
  machine code rather than remembered: the single reference to the constant `0.690800011` in
  `haloreach.dll` sits in the function that copies the world camera into the first-person camera
  and then multiplies that copy's vertical FOV by `0.6908 / 0.785`, publishing the factor to the
  engine's own `render_first_person_fov_scale` debug value. Sharing one camera is what makes a
  viewmodel splay and shrink as the FOV widens, and the error grows with the slider: the arms were
  **14.6% too small at 60, 15.3% at the authored 78 and 19.3% at 120**, which is why it looks fine
  at the default and wrong at the top. HMS now draws the rig -- the arms and the gun in them --
  through its own projection, so it keeps its proportions at every setting; captured at 60, 78,
  100 and 120, and at 60 and 78 **nothing outside the rig's own corner of the frame changes at
  all**, which is what says the world was left alone. The FOV slider's tooltip now states both
  numbers.
- **Theater: nobody's ARMOUR EFFECT played -- no flaming helmets, no thunder clouds, none of the
  six.** A player's armour effect is an ordinary effect the engine hangs on the biped, and three
  separate things stopped it. **(1) The posed bodies had no particle list at all.** An actor build
  deliberately takes the scene's particle list away for its duration so a body cannot disturb the
  map's waterfalls and grav lifts, and puts it back afterwards -- which also threw away anything
  the body itself collected; bodies now have a particle lane of their own, rebuilt every frame from
  the live pose because an armour effect rides a BONE (the flames sit on `b_helmet` and have to
  move with the head). **(2) An attachment was called "playing" only when it named no object
  function.** The Spartan biped's four attachments all name one, so that test turned every one of
  them off; read as functions -- which is how every other animated thing on a model is already read
  -- they separate exactly: the shield-down flash reads `shield_depleted` = 0 (shields up), the
  damage distortion `body_vitality` = 0 (undamaged), the out-of-bounds effect 0 (in bounds), and
  the armour effect `alive` = **1**. **(3) Two tag fields were read at the wrong width and stride,
  so every effect in the game put all of its particle systems on its FIRST marker.** The armour
  effect authors six markers, one per armour choice, and the marker is what says which of its five
  events -- `legendary_flame`, `eternal_flame`, `heart_attack`, `pestilence`, `storm` -- belongs to
  a body. Measured on `forge_halo`: each of the seven armour-effect options now resolves exactly
  one event (and option 0, "none", resolves none), and on a real clip the recorded player with
  Inclement Weather is drawn with the storm cloud on their shoulders.
- **A held ENERGY SWORD's blade glow and handle plasma now play too**, for the same reason: the
  object function behind them (`blade_effects`) is on, and the effect's first-person half is no
  longer drawn on top of its third-person half.
- **Scripted captures (`shot`) never showed particles at all** -- not a grav-lift plume, not a
  waterfall, not an armour effect. The scripted renderer simply never ran the particle simulation,
  so a capture and the window showed different pictures; it now runs the same one, off the same
  clock.
- **Theater: on some clips EVERY player's body was nameless, and the reason was that HMS read the
  owner stamp from a fixed place in a message that does not keep it in a fixed place.** A biped's
  creation message states who owns it, and HMS read those four bits at bit 272 -- a constant
  found by sweeping offsets on one clip. The engine's own reader (`sub_1803B2590`) reaches that
  field only *after* a shared header with two optional sub-fields **and** after an optional
  APPEARANCE block (the player's armour and colours) whose own cost swings between 108 and 192
  bits, because twelve of its sub-fields are present-or-absent. So the field moves, over a range
  of 200 bits, and one constant can only ever be right for one shape of message. It happened to
  be right for the 16-player test clip everything had been checked against. **And the field is
  five bits wide, not four** -- the old four-bit window took its low four bits, so wire 16, which
  is player slot 15, came back as wire 0, "nobody". HMS now WALKS the creation message the way
  the engine does and reads the whole field. Measured on the 16-player test clip: **108 of 112
  bodies named, up from 107** -- and the four that are not named are exactly the four whose
  creation carries no appearance block, i.e. the ones the recording itself says belong to nobody,
  so **every player's body on that clip now has its gamertag**. Slot 15, which no clip could ever
  name before, is named.
- **The field everyone had read as "which PLAYER owns this body" is which TEAM owns it.** Reach
  puts every player on their own team in free-for-all, so on those clips the team index and the
  player are the same number and the field names everybody -- and on a team game it collapses.
  A Castle Wars clip's "two owners" are its **two teams, of 8 and 10 players**, whose body counts
  come to exactly the 207 and 183 that were measured; a three-team clip reads three. Nothing was
  wrong with the decode, the field simply means something else, and the armour below is the route
  to a name on a team clip.
- **Theater: a body can now be named by its ARMOUR, which is how the clips the team field cannot
  name get their gamertags.** A body's creation message carries its player's whole appearance --
  species, three armour colours and twelve armour-slot options -- and the film header carries the
  same thing per player, so a body whose armour matches exactly one player's is that player's. It
  runs last, so it only fills gaps, and a body whose armour matches two players (or none) stays
  blank rather than guessing. On the Castle Wars clip that is **0 named bodies to 55**; on clips
  the owner field already handles it changes nothing.
- **...and 22 of those names were wrong, because an UNCUSTOMISED player is not a fingerprint.** A
  guest who never touched the armour editor carries the all-zero default appearance, and "matches
  exactly one player" was true of it for the dullest possible reason: exactly one person in the
  list happened to be uncustomised. So 22 bodies wearing stock armour -- including every other
  player's freshly-spawned default -- were being handed that guest's gamertag. The default
  appearance is now read as the **absence** of an identity and names nobody, in both armour routes.
  That is the Castle Wars figure moving from 77 to **55**: a name fewer is the right outcome when
  the 22 were not that person.
- **...and one gamertag could still end up on a hundred bodies, by the other armour route.** The
  rule that a player drives one body at a time already protected the owner field, but the two
  armour routes were written without it: they check how many *players* match a body's armour and
  never how many *bodies* they are hanging on one player. On a 47-minute clip where many people
  wear the same gear that put **110 bodies on one slot with four alive at the same moment**, and on
  another **68 on one gamertag with seven at once**. Both armour routes now apply the same measured
  limit as the owner field -- at most three bodies of one player coexisting, a live one plus a
  corpse and a ragdoll -- and refuse the whole group rather than trimming it, because bodies
  overlapping in time mean the armour is not identifying one person and no part of the group can be
  trusted. Across every film here that is 897 names kept of 1,093 candidates, with each of the 196
  refusals recorded and attributable rather than silently missing.
- **Theater: a body can also be named from ANOTHER of your recordings, when the clip's own player
  list has forgotten them -- and it says when it has done that.** A film's player list is who was
  present when the list was written, not everyone who played: the Castle Wars clip names **3 of
  the roughly 18 people in it**. But a Reach armour set turns out to be very nearly a unique
  fingerprint -- across every film here, 171 player entries collapse to 88 distinct appearances
  and only **one** of those is shared by two people -- so a body whose armour matches exactly one
  player across all your recordings is almost certainly them. That takes the clip from **55 named
  to 305** (55 the file states, 250 inferred). These names are **inferred, not stated**, so they
  are drawn dimmer and in brackets, kept and labelled separately
  from the ones the file states, they never override a stated name, and a body whose armour is
  shared by two people stays blank. It also means the number depends on how many recordings you
  have: with only the one clip you get the 77, and that is the file's own limit rather than a
  fault.
- **Known limit, stated plainly: some bodies cannot be named at all.** On that clip 22 wear armour
  two different people have worn and 63 wear armour belonging to nobody named in any recording
  here. Those names are not in the file in any form, so no amount of decoding will produce them;
  the bodies are drawn, with the right armour and colours, and simply have no name to show.
- **Theater: a player was drawn holding whichever weapon they picked up last, never the one in
  their hands.** A unit carries up to four weapons and states them separately -- one `weap`
  reference per slot -- plus a SELECTION saying which slot is readied, and HMS read slot 0's
  reference and ignored the selection entirely. So a player who picked up a sword and then
  switched back to their plasma pistol kept the sword for the rest of the clip. The engine's own
  `unit_get_primary_weapon` resolves the held weapon as `GetPlayerWeaponBySlot(unit, unit +
  0x34A)`, and `unit + 0x34A` is that selection's first field; HMS now follows the same route,
  carrying the four references and the selection across a body's life (each arrives only when it
  changes) and recomputing the held weapon whenever either moves.
- **Theater: "some of the bipeds are not rendering -- I see floating knives, but not the full
  biped." A body that never walked anywhere was being filtered out of the picture, and the weapon
  in its hands was not.** The entity filter "only entities that move" exists to keep the hundreds
  of thousands of static Forge pieces in a long clip out of the list, and it was being applied to
  PLAYERS as well: a body whose whole life moves it less than half a world unit -- a player shot
  the instant they spawned, which on a 0-second-respawn gametype is a great many of them -- was
  dropped before anything drew it, so it appeared neither alive nor as a corpse. The energy sword
  it was holding kept being drawn, because a dropped or held weapon is its own recorded object
  and has no such filter, and what was left standing on screen was the sword with nobody behind
  it. On the 12:16 clip from 2026-10-12 that is 4 of its 64 bodies, and the caption did not even
  report them as missing because it counts what survived the filter -- but across 38 films it is
  **1,064 of 4,559 recorded bodies, 23.3%**, with several clips over 80%. A body is a body whether or
  not it walked, so the motion test now only decides the other kinds of object, which is what it
  was written for.
- **Theater: a clip full of Warthogs drew none of them, and the reason was that HMS was still
  identifying a vehicle by GUESSING from where it appeared.** A vehicle used to be named by the
  nearest vehicle spawn point in the film's own Forge variant, which was the only identification
  available when that code was written. It is not the only one any more -- every creation message
  opens with a reference naming the object it was spawned from, and HMS has decoded that for a
  while -- and the old rule fails completely whenever the GAMETYPE spawns the vehicles instead of
  Forge placing them: on the 12:16 clip from 2026-10-12 the variant "Xenon // Castle" places no
  vehicle at all, so the search had nothing to match against and identified **0 of 66**, while
  the stream names them outright as 33 Warthogs (16,873 recorded positions between them) and 33
  Warthog turrets. Thirty-three Warthogs drove around that clip for twelve minutes and not one
  was on screen. A vehicle is now identified by the definition it states, the spawn point is only
  used to decide whether the Forge scene is already drawing a parked one, and a vehicle the
  variant places no copy of is always drawn because there is nothing for it to double up with.
- **Theater: a vehicle juddered -- it would stand still for a display frame and then jump --
  because the drawn placement was rounded to a whole tick while the clock driving it was not.**
  Playback keeps a fractional tick and advances it by the real frame time, but every placement
  read it back as a whole number, so the picture could only ever change 60 times a second however
  fast the display runs, and at a slowed-down playback speed -- the speed anybody actually uses to
  look at something in Theater -- one tick covered several frames and a moving vehicle stalled
  through all of them. Measured over 270,122 simulated frames of that clip: on a 144 Hz display at
  normal speed the per-frame judder was **13.33 world units at the 99th percentile and is now
  0.13**, and the share of frames that drew the identical point to the frame before fell from 77%
  to 46%; at quarter speed on a 60 Hz display it was 11.43 and is now 0.03. **At exactly 60 Hz and
  1x the two are identical to the last digit**, which is why this survived for so long. Interpolating between the recording's own position samples was already
  being done; the fraction being thrown away was the one between ticks. Vehicles, the objects the
  stream draws (dropped weapons, the flag, physics debris) and a rider in a seat are now all
  placed from the fractional clock, which is the same interpolation and is still exactly a
  function of the moment, so scrubbing back to a tick reproduces the frame it did before.
- **Theater: the caption claimed the Forge scene was drawing the vehicles the recording never
  gave a position for, and on this clip there was no Forge copy to draw.** Those are CHILD
  objects -- a Warthog's turret is its own network object parented to the hull, and 30 of the 33
  turrets in that clip never send one position in their whole life because their place in the
  world is their parent's. The caption now says that, and names the field (the parent reference
  in the object block) that is still to be decoded, instead of asserting something that is not
  true of this recording.

### Groundwork
- **One gravity, from the engine, instead of three from nowhere.** "Gravity etc still needs to
  actually be applied the way the game actually applies it." The codebase held **three** unrelated
  gravities differing by up to 3.25x: the ragdolls and test mode's jump used 10.4576 (fitted from
  1,447 real falls in recordings), the editor's "settle physics" used 3.2147, and the particle lane
  3.2174. The last two are Earth's gravity converted for Halo's world units -- **and they are
  digit-transpositions of each other, so one was a typo nobody had noticed.**
  - **Reach's own gravity is 4.1712594**, found in the engine: it is the default of a block the
    engine literally names `"physics constants"`, and the script function `physics_set_gravity`
    multiplies its argument by the same value -- so "the script takes a multiple of normal gravity"
    and "normal gravity is this" are one fact read two ways. Earth's gravity appears **nowhere** in
    the engine, so neither of the two constants above ever had a basis.
  - **The jump now uses the engine's own 4.1712594, and that is a reversal of what this entry
    said earlier today.** "Jump height is STILL not fixed at all, our biped hardly makes it off the
    ground" -- reported five times, and right five times. The 10.4576 that test mode was using was
    **not a measurement of anything**: the curve fit behind it discarded every sample above
    20 wu/s^2, which threw away the entire real population and reported the median of what was
    left. Re-run with that limit widened, the same films give a tight 31.38 -- so the old number
    was an artefact of its own filter, and the "2.507x unexplained gap" below was that artefact
    divided by the engine's constant. It never existed. The argument that kept it -- that the
    engine's constant would have a Spartan jumping 98% of his own height, which "plainly is not
    Reach" -- was a judgement offered as a refutation, and it is withdrawn: that ratio is simply
    what the authored jump velocity and the engine's gravity give, and it is not evidence either
    way. **What the jump actually does now, measured through the real controller rather than
    computed: a Spartan reaches 0.6150 wu with 1.103 s in the air, an Elite 0.8197 wu with
    1.271 s**, from the per-species velocity each biped authors (2.3000 and 2.6500 wu/s). The
    0.019-0.022 wu by which that falls short of the textbook apex is **exactly half a tick of
    velocity**, which is what a fixed-step integrator must lose when it applies gravity before
    integrating -- so the solver is not losing height anywhere else, and that is asserted rather
    than assumed.
  - **The ragdolls keep 10.4576 for now, and that is a holding position rather than a
    conclusion.** Nobody has reported a corpse falling wrongly, and a character you are driving
    and a dead body thrown by the physics engine are different systems that may genuinely obey
    different laws. The two now live in separately named functions so they cannot be changed
    together by accident.
  - **The editor's "settle physics for selection" now uses the engine's 4.1712594**, so **props
    settle about 30% faster.** Only the dynamics depend on it -- how long a prop takes to stop
    rolling -- and the resting pose it produces does not change, which is what that feature is for.
  - **Particles are deliberately left alone**, with a comment saying why: a particle emitter's
    authored gravity is a *fraction*, and 3.2174 is only the scale it multiplies. That lane was
    tuned against the Forge World waterfall sheets and changing it would move every particle in the
    game.
  - **What is still genuinely open**, and it is a smaller question than the one this entry used to
    claim: nobody has read the engine's own integrator, so whether it applies a tick of gravity
    before or after the jump impulse is unconfirmed. That ordering is worth about 3% of the jump
    height -- quantified, not hand-waved. The per-object `vertical acceleration scale` was the
    obvious candidate for a missing factor and measures 1.2 for both bipeds, which rules it out.
- **Reach's damage law, read out of the tags: what a melee and a shot actually deal, and what a
  vehicle's parts are worth.** Groundwork for test mode ("people may want to melee or shoot a
  vehicle and see it actually take damage") and for projectile impacts, which are the two things
  that need it. Measured, with the parts that are NOT measured left out rather than filled in:
  - **A melee's reach, its hit volume and its damage.** A swing is a PYRAMID, not a ray: the
    weapon authors the pyramid's angles and depth (0.8 wu by default) and a lunge range (1.22 wu
    by default). 29 weapons author melee parameters; a melee deals **79** and a clang **80**,
    which against a Spartan's 45 health is why one melee kills an unshielded one. The gravity
    hammer and the golf club get a wider, deeper pyramid than a rifle, and play `sword_clash`
    where a rifle plays `generic_melee_clash`.
  - **The clang.** A weapon names both the damage a two-melee collision deals and the effect it
    plays "in the air between two players that clang" — 23 of 29 weapons name the first, 27 the
    second.
  - **An object's vitality, split by part.** A Warthog has 500 vitality, and each damage section
    is its own pool sized as a share of it: `body` all 500, `hull_front` 150, each `wheel` 75,
    the `bumper` 50. A Ghost has 355 with a `gastank` worth 71. Each part's damage responses
    carry the fraction of health at which they fire and the damage state they move the model
    into, which is the state HMS already renders.
  - **What is NOT measured is stated in the code rather than guessed.** Which part a DIRECT hit
    lands on is unresolved — the engine decides it from the collision model's material, not from
    where the hit landed — so a multi-part model refuses the question instead of picking a part,
    and only an explosion (which the tag sends to a nominated part) is answered. The damage
    falloff with distance is likewise unread, so asking what an explosion does at a range returns
    "not known" rather than a number that would be full damage across the whole map.
- **Reach's sound banks now read, but films are still silent -- nothing is audible yet.** Reach
  MCC keeps all 16,162 world sounds in one 712 MB FMOD bank (and 31,433 lines of dialogue in
  another), encoded as FMOD Vorbis, which strips the codec's own setup headers out of the bank
  and into the game module. Those are now recovered from the user's own `haloreach.dll` -- so HMS
  ships no codec data of its own -- and one sample can be read out and written as an ordinary
  `.ogg`: the assault rifle's first shot decodes to exactly the 57,671 frames the bank declares
  for it, and two independent Vorbis decoders agree on the waveform to within one bit. A sound in
  a map is joined to its sample by the hash each `ugh!` permutation carries, confirmed on Forge
  World where 9,368 of 9,492 permutations agree with the bank on their own frame count. What is
  NOT here: no decoder, no audio device, no weapon-to-sound lookup and no ambience, so the app
  still makes no noise and still needs no sound hardware. See `re-notes/reach_fmod_sound.md`.

## 1.4.0 — 2026-10-12

1.3.1 was prepared on 2026-09-24 and never published — the last public release is **v1.2.2** — so
this release carries both halves. **Everything below is folded into `docs/changelog.html`** as the
v1.4.0 entry dated 2026-10-12 (grouped there by theme: Theater, Halo 3, Halo 2 Anniversary,
Halo 4, Editing, Rendering, Other). Entries for the *next* release go above this heading under a
new `## <version> — not yet released`.

### Fixed
- **The energy sword's blade was see-through but DEAD -- the plasma that scrolls along it was not
  playing.** Two things had to be fixed together. The decode read a material's authored UV scroll
  only for the forcefield/palettized-plasma family, so a plain halogram got nothing -- and the
  blade's own `energy_sword_blade_activation` layer authors a **5 s scroll on the very map that
  lane samples** (its plasma layer authors four more, on its two noise maps and its overlay). Then
  the shader froze the UV anyway: its "static icon" test is `lane > 2.5`, and the two additive
  self-illum lanes are numbered 9 and 10, so they were swept up by a rule meant for armour-ability
  icons. Both lanes now scroll at the rate the artist authored -- and only that rate, never the
  invented drift that makes generic effect billboards read as alive.
- **Theater: animated materials now run off the PLAYBACK POSITION rather than the wall clock.**
  Everything animated reads one clock -- the sword's plasma, a flashing Forge light, water -- so
  on the wall clock what you saw at a given moment of a film depended on *when you looked*: pause
  and the blade kept scrolling, scrub away and back and it was at a different phase, and two
  captures of the same moment differed. A Theater frame is now a pure function of where the
  playback head is, the same rule the ragdolls and the death clips already followed. Measured: two
  renders of one tick are byte-identical, and ticks 5 apart differ over 15,000 pixels of the blade.
- **Theater: the first-person arms shook, and so did every one-shot animation -- on any monitor
  faster than 60 Hz, or at any playback speed below 1x.** A clip's frame is half the playhead in
  ticks, and the playhead is a float advanced by real frame time -- but it was being ROUNDED to a
  whole tick first, so the animation stepped at the tick rate rather than the display rate and the
  clip held for some rendered frames and jumped on others. Measured on a 45-frame clip: at 144 Hz
  the frame advanced `0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5` and at 0.25x speed `0, 0, 0.5, 0, 0, 0.5`,
  against a constant 0.208 and 0.125 off the float. **At exactly 60 Hz 1x the rounded form is
  exact**, which is why it was never seen on the machine it was written on. The first-person arms
  and their overlays, and every one-shot (the death fall, an assassination, a landing, a jump arc),
  now run off the playhead itself -- still a pure function of it, so scrubbing reproduces the pose.
- **Theater: a jumping body was posed as if it were standing, in the middle of its own jump.**
  Being off the ground was decided by vertical speed alone, and at the APEX of a jump the vertical
  speed passes through zero -- so for about a third of a second a body read as GROUNDED, dropped
  onto a walk cycle in mid-air, and restarted its jump animation on the way back down. A body now
  stays airborne across its apex, by the one condition that is true there and nowhere on the
  ground: it was clearly moving vertically a moment ago and will be again a moment from now.
  Rising then falling is a jump; nothing else does both.
- **Theater: the airborne animation looped, "almost like a player is jumping repeatedly".** The
  jump arc is 45 frames and a jump lasts as long as the body is off the ground, which is usually
  longer -- and it was being driven by the walk-cycle lane, whose phase WRAPS. So the arc replayed
  from its first frame over and over for the length of the fall. It is a one-shot now, on the same
  lane as the death, the assassination and the landing clips: it advances once from the moment the
  body leaves the ground, holds its last frame until touchdown, and hands over to the landing. Like
  those, it is driven by an age in ticks, so it replays identically on a scrub.
- **Theater: a jumping or falling player stood slack in mid-air, as if in an A pose.** The clip
  being played was a JETPACK HOVER. Reach's animation graph has a `flight:` mode, and HMS read
  that name as "off the ground" -- but `flight:` is the jetpack: the shipped Spartan graph authors
  `flight:<class>:idle` as a 70-frame looping hover for every weapon class, beside
  `flight:rifle:takeoff` (18 frames), `flight:rifle:land` (24) and a `throttle_still_up` overlay.
  A jump has its own clips in the combat set -- `combat:<class>:airborne_arc` (45 frames) and
  `:airborne` (40), authored for every armed class. Because the first name HMS asked for was
  `flight:<class>:<gait>`, a body jumping with little horizontal speed asked for
  `flight:<class>:idle`, which exists, and got the hover; a RUNNING jump asked for
  `flight:<class>:move_front_fast`, which does not exist, fell through to the arc and looked
  right. That is exactly the reported split -- running jumps fine, standing and crouch jumps
  wrong. Airborne bodies now take the jump arc first, and the `flight:` mode is not asked for at
  all, because **nothing in a Reach film says who is carrying a jetpack**.
- **Theater: a body that came down from a jump or a fall snapped straight back into its walk,
  with no landing.** `combat:<class>:land_hard` is authored for every armed class (13 frames) and
  `crouch:<class>:land_hard` for the crouch set (50), and neither was ever asked for. A body that
  has just touched down now plays its landing once -- a one-shot driven by the age of the
  touchdown, like the death and assassination clips, so it is a pure function of the playback tick
  and scrubbing replays it identically -- and is handed back to its locomotion cycle when the clip
  ends. Like the airborne state it follows, the touchdown is DERIVED from the recorded vertical
  speed, because a film states no landing either: the action enum that names `_action_land` is the
  same absent field 0x36.
- **Theater: crouching is not drawn, and that is now known to be a property of the FILE rather
  than something still to do.** A crouch could only be stated in two places, and both are measured
  empty: the unit's derived dirty block carries the crouch button in field `0x37` (the control
  input record) and the jump/land transitions in field `0x36` (the action enum whose own names in
  the engine binary are `_action_jump`, `_action_land`, ...) -- and **bits 0x36 and 0x37 never
  appear, not once in 5,438,363 unit messages over 8 films**. The 72-type event channel carries no
  crouch or jump event either. So a crouching player is drawn standing; guessing it from a low
  speed would put everyone who stops to aim on one knee. What HMS *can* do is ready: the `crouch:`
  mode now asks for the leaf names that mode actually authors (`walk_front` / `move_front`, not
  the combat set's `move_front_slow` / `move_front_fast`), so the day a crouch signal is found the
  clips are already the right ones.
- **Theater: a weapon could be drawn hanging in mid-air instead of in the hand holding it.** A
  held weapon's mesh is baked once and then moved every frame by its bone palette, which holds
  `anchor^-1 x world x attach` -- so the mesh is only in the right place if it was baked at
  exactly `anchor`. The character's own mesh was baked there; its WEAPON was baked at the body's
  recorded position instead, a second writer for one frame. Those are the same point for anything
  standing in the world, which is why it survived, and they diverge exactly when a body's
  placement is not its own origin -- a rider in a vehicle seat, or a first-person weapon parented
  to a hand -- where the gun was then drawn displaced by the whole difference. Both are baked at
  the one frame now, and `actor list` reports a weapon's drawn position beside its correct one and
  names any mismatch rather than leaving it to be noticed on screen.
- **A character holding an energy sword drew a solid grey blade instead of a translucent blue
  one.** The blade is five ADDITIVE layers -- two `rmhg` halograms on the engine's from_diffuse
  lane, one forcefield layer and two `sword_illum` 3-channel self-illum layers -- and a Forge
  object has had the transparent passes those need for a long time. A posed character did not: the
  actor build could return only an opaque list and an alpha-test list, so **every** transparent
  part of a character or of the weapon in its hand was folded into the opaque one and drawn by the
  solid pipeline. Measured on the shipped tag (`objects\weapons\melee\energy_sword`): five of its
  seven drawn parts route to the additive lane and were landing in the opaque pass. Characters now
  carry the same five draw lanes a placed object does -- additive glow, alpha-blended marker holo
  and alpha-blend glass -- each with its own GPU-skinned pipeline, drawn after the opaque pass,
  depth-tested but never depth-written, in the one back-to-front sequence the rest of the scene's
  transparents already share. So a held sword's blade glows and is see-through, and a character's
  glass or hologram parts composite instead of turning into a grey slab.
- **Theater: the emblem art turns out to be in the cache, and the "shared placeholder" finding was
  wrong -- both now written down so the emblems can be drawn.** Two earlier conclusions are
  corrected. First, "MCC writes one shared placeholder emblem, so there is nothing to draw" was an
  artifact of a single sixteen-player film whose slots happened to be byte-identical; an
  in-progress recording shows **two distinct emblem records across three slots**, so the eight
  bytes at player `+0x50` are real per-player data. Second, drawing them does NOT need MCC's
  `emblemstexturepack.perm.bin`: `rasg` +0x88 reaches `mlib` (`emblem_library`), whose `emblem
  bitmaps` and `emblem bitmaps hi rez` both resolve to `bitm ui\cui\common\emblems\emblems`
  inside the Reach cache at a 64-pixel resolution. The tag postprocess zeroes the authored blocks
  and leaves `runtime front` (80 entries, matching the authored primary count) and `runtime back`
  (53), whose layers are six `i16` -- a sprite index, columns and rows, twice, for the normal and
  hi-rez pair -- followed by a five-float layer transform. That transform is the only piece still
  unresolved, and it is plainly a scaled rotation: one entry is `0.88388` twice, which is
  `cos(45 degrees) x 1.25`, and two others carry magnitudes in an exact 2:1 ratio. Written up in
  `re-notes/reach_player_customization.md` under `# THE EMBLEM ART IS REACHABLE (#emblem-art)`,
  including the fallback that needs no transform at all.
- **Scripted captures of a Theater camera fired from the wrong camera, because applying it
  depended on the ORDER the verbs were written in.** The view was moved onto the film camera
  inside the `film scene` handler, but the natural way to write a script is `film scene` first and
  `film camera firstbody N` after it -- and selecting the camera afterwards re-ran nothing, so
  every `shot` kept firing from the camera the map load had left behind. The symptom was silent
  and convincing: six different first-person targets produced six frames differing by a **maxdiff
  of 6**, which reads like "the camera works and the bodies happen to look the same" rather than
  "the camera never moved". It is now applied once, at capture time, so it cannot depend on verb
  order, and the old copy is gone rather than left as a second implementation. **`shot` also now
  reports the camera it used, AND whether that camera actually resolved** -- `cam first:owner7
  (12.34,56.78,15.01) yaw 123.4 pitch -5.6 fov 49.1`, or `cam follow:owner2!UNRESOLVED-stale-view`
  when the film has no body for that slot or owner at that tick and the view is therefore left
  wherever it was. Reporting only the REQUEST would have relabelled a stale frame with the camera
  the script wanted, which is the same false confidence; it caught three such frames within
  minutes of landing. Every first-person conclusion
  drawn from a scripted capture before this is unproven and is being re-checked.
- **Theater: "if I rewind, a lot of players seem to be invisible" is now narrowed to one half of
  the code, with the other half proven innocent.** Which bodies a tick has, and where they are, is
  a pure function of the tick and the decoded index, so arriving at a tick by rewinding must give
  a byte-identical answer to arriving at it by playing forward. That is now gated on a real film:
  ten probe ticks spread over a 25,869-tick clip, visited once in order and once by jumping to the
  end and walking backwards, give **identical body sets both ways with zero mismatches** (the
  busiest probe draws 24 bodies), with the positions compared too. So playback is not the cause.
  Anything still missing after a rewind is therefore the stateful actor POOL failing to re-show a
  body it had parked -- a different piece of code, and one that needs a render to see, so it is
  called out here rather than claimed fixed.
- **Theater: the "Spartans walking around with no name" is a format ceiling, measured -- and the
  roster was under-reporting on top of it.** The owner stamp on a body's creation message is a
  four-bit field written as `slot + 1`, so it runs out: slots 0..14 take wire values 1..15, and
  `15 + 1 = 16` does not fit in four bits -- it truncates to 0, the same value the engine's `-1`
  ("nobody") produces. The engine's own reader hands those bodies back as "nobody" too and never
  minds, because a live game identifies players by a runtime mapping rather than this field; only
  a film reader is left with the lossy wire.
  That suggested an appealing fix -- on a full lobby the stamp-0 group of bipeds must BE slot 15,
  so bind the whole group to it -- and it was written and gated six ways. **Measuring it against
  the real sixteen-player clip killed it, for two independent reasons, so it was deleted rather
  than left in to never fire.** All sixteen slots were ALREADY claimed, because the
  species-elimination path had named slot 15 on its own; and the stamp-0 group has **four bodies
  alive at once**, where one player plus lingering corpses allows three, so binding it as one
  player would have put a gamertag on somebody else's bodies -- the exact mistake an earlier round
  made when sixteen bodies were bound to one slot with thirteen alive simultaneously. So wire 0 is
  a **mixing bucket**: unowned bipeds and any player the field cannot encode both land in it, and
  only a per-body test can separate them, which is what elimination already does. On the test clip
  it names one of those six bodies; the other five cannot be attributed by anything in the file,
  and are left unnamed on purpose. The measurement is now a gate, so the next person with the same
  good idea finds the numbers instead of repeating the work.
  **What WAS wrong and is now right:** the roster counted a group as named only if the stamp
  answered, and the stamp declines wire 0 by design -- so it reported a player as un-nameable
  while their gamertag was being drawn over their head, because the nameplates read the bindings
  instead. `FilmIndex::owner_group_is_named` now asks both routes, and the gate asserts the two
  cannot disagree.
- **Theater: the emblem art turns out to be in the cache, and the "shared placeholder" finding was
  wrong -- both now written down so the emblems can be drawn.** Two earlier conclusions are
  corrected. First, "MCC writes one shared placeholder emblem, so there is nothing to draw" was an
  artifact of a single sixteen-player film whose slots happened to be byte-identical; an
  in-progress recording shows **two distinct emblem records across three slots**, so the eight
  bytes at player `+0x50` are real per-player data. Second, drawing them does NOT need MCC's
  `emblemstexturepack.perm.bin`: `rasg` +0x88 reaches `mlib` (`emblem_library`), whose `emblem
  bitmaps` and `emblem bitmaps hi rez` both resolve to `bitm ui\cui\common\emblems\emblems`
  inside the Reach cache at a 64-pixel resolution. The tag postprocess zeroes the authored blocks
  and leaves `runtime front` (80 entries, matching the authored primary count) and `runtime back`
  (53), whose layers are six `i16` -- a sprite index, columns and rows, twice, for the normal and
  hi-rez pair -- followed by a five-float layer transform. That transform is the only piece still
  unresolved, and it is plainly a scaled rotation: one entry is `0.88388` twice, which is
  `cos(45 degrees) x 1.25`, and two others carry magnitudes in an exact 2:1 ratio. Written up in
  `re-notes/reach_player_customization.md` under `# THE EMBLEM ART IS REACHABLE (#emblem-art)`,
  including the fallback that needs no transform at all.
- **Theater: on a sixteen-player film one player's Spartans walked around with no name, and now
  they have one.** The owner stamp on a body's creation message is a four-bit field written as
  `slot + 1`, so it runs out: slots 0..14 take wire values 1..15, and `15 + 1 = 16` does not fit
  in four bits -- it truncates to 0, which is the same value the engine's `-1` ("nobody")
  produces. The engine's own reader hands those bodies back as "nobody" too, and never minds,
  because a live game identifies players by a runtime mapping rather than this field; only a film
  reader is left with the lossy wire. On the test clip the header names all sixteen players, wires
  1..15 account for fifteen of them, the one left over is slot 15 (`Jega Rdomnai544`), and the
  stamp-0 group has six lives, five deaths and 628 world units travelled over the full seven
  minutes: a player, not a prop. Across the Reach films a stamp-0 group that walks is common
  (87-628 wu) -- it is NOT a sixteen-player tell on its own, because many clips carry a stale
  header that lists only two or three players present while far more actually played, and on those
  nothing can be named at all. So the recovery is gated on the header, not on the walking. The recovery is implemented -- bind that group to the leftover slot,
  but only when all five conditions hold: the `slot = owner - 1` rule was corroborated on that
  film by the recorder anchor, the header really occupies slot 15, slot 15 is the ONLY occupied
  slot left unclaimed, the group behaves like one player (at most three bodies at once), and at
  least one of them moved. Six gates, one per condition flipped, so removing any condition fails
  one.

  **It does not yet fire on the measured film, and the gate that says so is the point.** A seventh
  gate runs the rule against the real sixteen-player clip: the anchor corroborates, sixteen
  players are present, and **zero bodies are bound**, so one of the five conditions is not met on
  real data. The diagnosis above is unaffected -- slot 15 is named by nobody and the stamp-0 group
  walks 628 wu -- but the attribution is not yet earning its place, so it is reported here as
  incomplete rather than as a fix. The likely reason is that wire 0 is a MIXING bucket: slot 15's
  bodies and any genuinely unowned biped both land in it, which would make a whole-group
  attribution too coarse and leave per-body elimination (which already runs after this, and names
  two bodies corpus-wide) the right instrument.
- **Theater: the flag is drawn, and so is every loose weapon and every piece of debris the
  recording carries.** A Castle Wars clip showed no flag at all, and the cause was the playback
  draw filter: it kept only entity type 11 (bipeds), so every other kind of recorded object was
  dropped before it reached the scene -- vehicles escaped only because they had been given a path
  of their own. The recording was never at fault. Resolving each track's 14-bit definition
  reference through the loaded map's tag table, not just the bipeds', names **33 `weap flag`
  tracks, 28 of which move**, plus **407 `energy_sword` and 7 `gravity_hammer`**: 1,562 non-biped
  tracks over 123 distinct definitions in that one clip. **What is drawn is now decided by the
  measured move share rather than by taste**, and the split is sharp: the `weap` group is 84.6%
  motion (378 of 447 tracks), while not one of the 427 `eqip` / `scen` / `ctrl` tracks ever
  reports a second position -- an `eqip sprint` is an armour ability a loadout granted, not a
  pickup on the floor, and a `scen kill_volume` would be an invisible box made visible. `bloc` is
  the one group with both halves in it, so it is judged per track: 535 of its 630 tracks are Forge
  pieces the film's own scene already draws, and the 95 that move are all physics debris -- a
  garbage tire a player threw (24 of 24 move), a wheel knocked off a Warthog (18 of 18) -- which
  are now drawn too. A weapon that nobody ever touched is still left to the Forge scene, by the
  same spawn-coincidence rule the vehicles use, so nothing doubles up and z-fights. There is one
  switch for the whole lane (`film objects on|off`), and the panel says what the clip holds and
  what is on screen.
- **Theater: a carried flag is drawn where the recording puts it, which turns out to be the whole
  answer.** The flag is not one long-lived object: it is a CHAIN of 33 short tracks, each ended by
  the recording's own destroy with the next created 0 to 283 ticks later, at most two alive at once
  -- one per team. And each track keeps reporting its position for its whole life while a player
  carries it: on 17 of the 31 real tracks **every single recorded position** is within 1.5 world
  units of a biped, closest approach 0.10-0.58 wu, while the tracks that were dropped and left
  lying come no closer than 6.62 wu. **So the carrier attachment is RECORDED, not derived** -- no
  flag is pinned to a hand marker, and none has to be. The 1.5 wu figure is that measurement, not
  a guess: a biped's recorded position is its feet and the flag is held at hand height out in
  front, so an earlier 0.5 wu would have called the flag dropped while a player ran it home.
- **Theater: the objective gets its waypoint, by the engine's own chud rules, and only as far as
  the recording supports one.** An objective waypoint is a chud navpoint, the same mechanism as a
  player's floating name, so it is painted by the same painter with the same authored palette:
  `chud_color_from_index` over the `chgd` skin's colours, `navpt normal` #2494FF / `navpt enemy`
  #D83232 / `navpt neutral` #5E5E5E, and the engine's own distance law. **Which colour an objective
  gets is not in a film**, and that is said rather than invented: a film states neither the
  gametype's waypoint colour nor which side a flag belongs to (the flag is one tag for both teams).
  What it does state is who is holding it, so a carried flag is coloured by its carrier's team
  against the viewer's, through the engine's own team comparison, and a flag on the ground is
  `navpt neutral` -- the engine's answer for an unknown team. The anchor is HMS's own lift above
  the object rather than a decoded engine marker, and the UI reports it as such.
- **Theater: a recorded object is placed with its FULL basis, which on this stream matters more
  than it does for vehicles.** 51,978 of the 59,849 recorded non-biped positions carry an
  orientation beside them (86.8%), and **96.6% of those bases are more than 5 degrees off level**:
  this stream is full of tumbling debris, so a compass heading would be wrong nearly always.
  `pickup_garbage_tire` reports 99.9% as many bases as positions, `hubcap` 101.8%,
  `destroyed_gastank` 141.8% (turned without being moved), `energy_sword` 86.1%. The honest
  exception, measured and stated: **a `flag` track carries exactly two orientation samples**, so a
  drawn flag's basis is its creation orientation -- the premise that a flag reports a basis as
  densely as a vehicle is false on this clip, and the placement needs no special case for it.
- The non-biped inventory stays a standing diagnostic (`film_action::objective_inventory`,
  `#[ignore]`d because it opens a map and a film): `HMS_FILM=<film> cargo test -p hms-app
  print_non_biped_inventory -- --ignored --nocapture`, alongside three new ones in
  `film_objective::probes`: the motion/carry census, the flag's track lifetimes, and
  `probe_real_table_over_a_film`, which runs the shipping classifier over a real clip with the
  real tag table and asserts the two things the report turned on -- the flag's definition
  resolves, and the flag is in the drawn set.
- **Theater: seeking froze the window, because every seek re-simulated the ragdolls before it would
  draw.** Catching a ragdoll up costs about a millisecond per simulated tick per body, and a seek
  can land a long way past a death, so the whole fall was replayed synchronously on each one.
  Measured over twelve rewind-seeks on the test film: **336 ms per seek with the solver on against
  124 ms with it off** -- the catch-up alone was ~212 ms of every seek, and dragging the timeline
  fires several a second. A frame now runs at most `RAGDOLL_CATCHUP_BUDGET` (120) simulation ticks
  shared across ALL corpses, so several bodies cannot multiply the cost; a body that needs more
  keeps its last simulated pose and advances again next frame, settling over a few frames instead
  of blocking. The simulation is still a pure function of (seed, tick) -- this bounds only how much
  may be re-run at once -- and the scrub-determinism gates are unchanged. **Now 127 ms per seek**,
  i.e. the solver costs ~3 ms rather than 212.
- **Theater: the owner stamp was being read with a wrap that the engine does not have, which
  invented a sixteenth player.** Reverse-engineered from `haloreach.dll` rather than inferred: the
  engine's runtime mapping is `player_datum + 0x28` (the unit's object index) against the object's
  `+0x1CC` (the controlling player index), both written by `Player_SetUnit` (`+0x45814`). The *wire*
  form is one field in the unit's creation record, and the two halves of the codec agree exactly —
  the writer (`+0x3B242C`) emits `write_bits(*(char *)(obj + 0x3E) + 1, 4)` and the reader
  (`+0x3B2590`, identified as the unit reader by the `'eqip'` tag reference it goes on to decode)
  takes `read_bits(4) - 1`. So **wire 0 is the field's `-1` sentinel and means "attributed to
  nobody"**, and because four bits hold `index + 1`, the field tops out at index 14: it can name at
  most FIFTEEN players. HMS had been computing `(wire - 1) & 0x0F`, which sends wire 0 to slot 15 —
  so on a sixteen-player clip it put the sixteenth player's gamertag on whatever bodies the
  recording had attributed to nobody. That reading is gone. The honest consequence is now stated
  rather than papered over: on a full lobby at least one player's bodies carry "nobody" in the only
  field that attributes a body, and HMS names such a body only where elimination forces it.
- **Theater: the per-player channel that could have answered this outright is not in the file, and
  that is now measured rather than assumed.** `s_player_action` (`sub_18043FC00`) carries three
  27-bit object references per player slot, and HMS already walked all three and threw them away.
  They are now captured — and the census says the channel is **absent from every film on this
  machine: 0 blocks over 400,000+ ticks**. The unit's replicated state does not carry the player
  either (`SessionBitStream_WriteUnitState` never touches `+0x1CC`). So the creation field really is
  the only place a film states who drives a body.
- **Theater: one gamertag was being shown over as many as thirteen Spartans walking around at the
  same time.** Found by the structural gate written for the fix above, which asks the one question
  the format allows: how many bodies carrying one gamertag exist at the same moment? A player
  leaves a corpse and a ragdoll behind, so their *tracks* legitimately pile up — measured over 39
  real films, a group that behaves like a player peaks at **three** bodies at once and its longest
  overlapping pair lasts 11.7 s out of a 7-minute clip. Three films were far outside that: one had
  **16 bodies on slot 0 with 13 alive together** and a worst pair overlapping **353 s**, another 68
  bodies with 7 alive together, a third 11 with 4. Those owner groups are several people sharing
  one stamp, and HMS was naming them anyway. The refusal that already protected the ambiguous
  stamp 0 now applies to every stamp and to the aim-correlation path as well, so a group that
  cannot be one player names nobody. Across the corpus the worst "one gamertag, N bodies at once"
  falls from 13 to 3 and the worst overlap from 353 s to 77 s.
- **Theater playback rebuilt every biped's GPU meshes on EVERY FRAME, which is what made the frame
  rate collapse.** Measured per frame while a film was open: **436-2191 ms** of mesh rebuild before
  anything was drawn; the same scene with actors placed by hand and no film open cost 0.0 ms. Two
  pieces of code were fighting over one field. `build_actor_meshes` writes the model's OWN DEFAULT
  variant into `Actor::variant_sid` whenever it finds it unset; the recorded-appearance pass then
  compared that against a bare `0`, decided the actor was wearing somebody else's armour, cleared
  it, and marked the actor scene dirty -- which rebuilt every actor mesh, which re-resolved the
  default, for ever. Any body with no recorded appearance pinned the scene into a permanent
  rebuild. The player variants are a reserved id range (`player_variant_sid` = `0xF1A1_0000 | slot`),
  so "is this actor wearing somebody's RECORDED armour" now has an exact answer and a model default
  is left alone; the armour-leak fix that introduced the comparison is intact. A steady Theater
  frame is now **0.0 ms of rebuild and 20-30 ms of render**, and a seek no longer triggers one.
- **Theater: vehicles slid around dead level, never rolling or pitching, even going through a man
  cannon.** The recording had the rotation the whole time, on nearly every update. A vehicle's
  per-tick message carries base field 2 — a 19-bit up axis plus an 8-bit angle about it — which
  the engine's own decode turns into a full orthonormal `(forward, up)` pair, so pitch and ROLL
  are decoded facts rather than something to guess: over **1,925 vehicle tracks in 25 films there
  are 987,003 recorded orientations against 1,025,932 recorded positions (96.2%)**, and **46.3% of
  those orientations are more than 5 degrees off level, 12.5% more than 45, and 6.9% are past
  90 — upside down** (the worst on this machine is a flat 180.0). HMS decoded that pair and threw
  all of it but the heading away on the next line (`atan2(forward.y, forward.x)`), because an
  actor could only be placed with a yaw; the status line even admitted *"a vehicle is placed with
  its DECODED yaw only, so one on a slope is drawn level"*. So nearly half of every vehicle moment
  in Theater was drawn flat, including every man-cannon tumble. A vehicle is now placed with the
  whole recorded basis, built by the engine's own `Matrix_BuildOrientation` rule (the same one
  every Forge object is placed with), and the basis is SLERPED between the recorded keys so a
  tumble reads as a tumble instead of stepping between reports. Riders got the same fix: a body in
  a seat is placed on the vehicle's real placement times its seat marker, so someone in a banking
  Falcon or a rolling Warthog leans with it instead of sitting bolt upright. Gated by a test that
  takes a recorded basis rolled 30, 90 and 150 degrees and fails unless the placement carries that
  roll — the trap being that a test which only checked the heading would have passed the broken
  code.
- **Theater: a dead body is now a real RAGDOLL — the biped's own physics model, simulated —
  instead of one of seven canned clips dropped straight down.** Reach's death variety is not a
  clip library: the Spartan animation graph holds **1,393 animations and exactly SEVEN death
  clips** (`combat:any:landing_dead`, `combat:landing_dead_back` +`:var1`,
  `combat:any:airborne_dead` +`:var1`, `combat:airborne_dead_back` +`:var1`), so "the dead bipeds
  need proper variety" could only ever be answered by a simulation. HMS had none: a death picked
  one clip from the entity id, cast ONE ray straight down and fell along a parabola at the
  10.4576 wu/s² fitted off these recordings — no limbs, no collision response, no variety.
  - **The missing data was the `phmo`'s CONSTRAINTS, which nothing decoded.** A ragdoll is the
    tag's rigid bodies *plus* the joints authored between them, and those live in six sibling
    blocks that all begin with the same 0x78-byte header — `name` string id @0, node A @4,
    node B @6, A scale @8, A forward/left/up @0x0C/0x18/0x24, **A position @0x30**, B scale @0x3C,
    B forward/left/up @0x40/0x4C/0x58, **B position @0x64**, edge index @0x70 — at meta offsets
    `hinge` 0x118 (0x78 B), **`ragdoll` 0x124 (0x94 B)**, `limited hinge` 0x160 (0x84 B),
    `ball and socket` 0x16C (0x78 B), `stiff spring` 0x178 (0x7C B), `prismatic` 0x184 (0x84 B),
    with `node edges` at 0x50 (0x1C B) carrying the same graph again. A ragdoll element's tail is
    `twist` (min,max) @0x78, `cone` @0x80, `plane` @0x88 and `maximum friction torque` @0x90.
    **The two offsets that matter are derived from the engine, not taken on trust.**
    `reach_tag_test.exe`'s own tag reflection lists the phmo's block fields in offset order and
    that run reads `… mopps`, `mopp codes` (a 20-byte DATA reference), **`hinge constraints`**,
    **`ragdoll constraints`**, `regions`, `nodes`, `REMOVE ME`, `errors` — so with `mopps` at
    0xF8 (known good: the editor's settle already walks it on real maps) the two constraint blocks
    can only be 0x118 and 0x124, `regions` 0x130 and `nodes` 0x13C follow, and `REMOVE ME` at
    0x148 is exactly the three nameless 4-byte fields the community plugin shows there. Only the
    ORDER is taken from that run, never its absolute offsets — as read it omits `multi spheres`,
    which the shipped layout must have (the settle walks `pills` at 0x8C and `mopps` at 0xF8 on
    real maps), so it is not the whole field set. That run also stops there, so the four OTHER
    constraint types' block offsets
    (`limited hinge` 0x160, `ball and socket` 0x16C, `stiff spring` 0x178, `prismatic` 0x184)
    remain the community plugin's attribution and are read DEFENSIVELY — a constraint whose two
    frames are not orthonormal bases is dropped, which is what a wrong block offset produces, and
    `actor ragdoll <biped>` prints present-versus-decoded per type so a real tag answers it
    instead of a guess.
  - **Every offset INSIDE a constraint element is the engine's own, checked against the engine's
    own `sizeof`.** The same reflection carries the struct definitions (their source path is
    `c:\mcc\qfe1\reach\…\physics\physics_model_definitions.cpp`, the shipped MCC Reach build):
    `sizeof(s_physics_model_constraint_bodies)` is **0x74** and its fields are, in order,
    `name^` (string id) @0, `node a` @4, `node b` @6, `a scale` @8, `a forward`/`a left`/`a up`
    @0x0C/0x18/0x24, `a position` @0x30, `b scale` @0x3C, `b forward`/`b left`/`b up`
    @0x40/0x4C/0x58, `b position` @0x64, `edge index` @0x70 and a 2-byte pad — which sum to
    exactly 0x74. `sizeof(s_physics_model_ragdoll_constraint)` is **0x94**: that struct, a 4-byte
    pad to 0x78, then `min twist` @0x78, `max twist` @0x7C, `min cone` @0x80, `max cone` @0x84,
    `min plane` @0x88, `max plane` @0x8C, `max friciton torque` @0x90 (the engine's own typo) —
    0x74 + 4 + 7x4 = 0x94, so the stated size is itself the check on the whole layout. The other
    stated sizes match this decoder too: ball and socket **0x78**, limited hinge **0x84**, stiff
    spring **0x7C**, prismatic **0x84**. On top of that comes the behavioural check — **`a
    position` and `b position` are each in their OWN node's space, so a correct read puts both at
    the same world point at the bind pose**, asserted as a gate with the one-space misreading
    asserted to fail it.
  - **What the shipped bipeds hold, read off the map** (`hms-app --dump-ragdoll <map> spartans`,
    a new print-only verb that needs no GPU): the Spartan's `phmo` is **11 rigid bodies, one per
    skeleton node** — `b_pelvis`, `b_spine1`, `b_head` and left/right `upperarm`, `forearm`,
    `thigh`, `calf` — each with an explicit mass summing to **300.00**, joined by **10
    constraints, all 10 decoded**: four limited hinges (`$_const_l/r_elbow`, `$_const_l/r_knee`)
    and six ragdoll joints (`$_const_head`, `$_const_l/r_shoulder`, `$_const_l/r_hip`,
    `$_const_spine1`). The Elite is the same 11-body, 4-hinge, 6-ragdoll shape around `b_spine2`
    and `b_neck0`. On both, **every phmo node name equals the `mode` node name of the same index**
    (82 of 82 and 67 of 67), and **the bind-pose pivot residual is 0.00000 wu on all twenty
    joints**.
  - **That read also caught a units bug before it could ship: the authored angles are DEGREES,
    not radians.** Every one of the twenty limits is a round multiple of five in degrees and
    sensible as one — the Spartan's elbow is `-75 .. +15`, its knee `-55 .. +15`, the Elite's
    forearm `-80 .. 0`, its calf `-20 .. +30`, shoulder cones 45 and 50, hip cones 35 and 25,
    spine cones 10 and 20. Read as radians they are 4 to 80 radians, up to twelve whole turns, so
    the cone saturated its sanity clamp and **every joint came out unlimited** — a Spartan would
    have folded in half. Gated now, with the raw-as-radians reading asserted to fail. Every
    `max friciton torque` is 0, so a Reach ragdoll really is limp: there is no authored joint
    friction to simulate.
  - **A new shared solver (`ragdoll.rs`), and the editor's "Simulate physics for selection" now
    runs on it too.** It is `physics_settle`'s collision geometry and contact solver generalised
    from one welded body to N bodies joined by the tag's constraints: sequential impulses at the
    engine tick, a full 3x3 effective-mass point-to-point row per joint, then the angular LIMIT
    rows — twist from a swing-twist decomposition, and cone and plane whose Jacobians are derived
    rather than guessed (`d/dt acos(uA·uB) = ω_rel·normalize(uA×uB)` and
    `d/dt asin(uB·pA) = ω_rel·normalize(uB×pA)`, both checked against a finite difference over 200+
    configurations) — then the contacts, with Coulomb friction and split-impulse penetration
    recovery. Limbs collide with the level, with the Forge objects' physics hulls and with each
    other (sphere-vs-sphere between links that are neither joined nor already overlapping at the
    bind pose, measured once per biped). `physics_settle::settle` is now literally a one-link,
    no-joint call into the same `step`, so the editor's settle and a Theater ragdoll are one
    implementation — gated by the settle's own eleven tests, which still pass unchanged.
  - **It is deterministic under scrubbing**, which the timeline makes non-negotiable: the state at
    tick T is a pure function of the seed and T, and the seed is recorded — the death tick, the
    body's own recorded velocity, the killer the kill feed (`incident`) names and the kind of kill
    it was. Only the initial spin is derived from the entity id, the same way the death clip
    variant always was, and the caption says so. Scrubbing forward steps; scrubbing BACKWARDS
    rewinds to the nearest keyframe and re-simulates, so a body stands back up through exactly the
    fall it came down by. The keyframe cache is a speedup only: discard it and the same tick
    redraws the same pose, which is gated by comparing a tick reached by playing straight through,
    by jumping to it, and by dragging back and forward again. That gate caught a real hole on the
    test film before this shipped: the seed pose was taken from whatever death clip `sync` had
    last installed on the actor, which depends on where the timeline has BEEN, and a body reached
    at tick 251 by playing forward settled **0.29 wu** from the same body reached by dragging back
    to 249 and forward again. Which clip seeds the simulation is now recomputed from the recording
    (the entity id, the species, and whether the body was airborne at the death tick the kill feed
    states), and all three routes to tick 252 now agree to the centimetre on both bodies on screen.
  - The skeleton follows the simulation through a new per-node pose override, and the nodes no
    rigid body drives (fingers, the gun hand's children) ride their driven ancestor instead of
    collapsing, so the held weapon stays in the hand. A biped whose `phmo` has no joints is
    refused rather than flown apart, and the playback caption says which bodies the solver owns
    and which are still on the old vertical fall.
- **Theater: a player's legs snapped to the T-pose for the length of every melee swing.** Reach's
  melee clips are animation type **Replacement**, and a replacement clip carries only PART of the
  skeleton: `combat:sword:melee_strike_1` is 30 frames of **34 rotation channels over 82 nodes**,
  and `b_pelvis`, `b_l_calf` and `b_r_foot` carry no channel at all — the legs are not in the clip.
  HMS installed the swing as the body's SOLE clip, and evaluating a clip starts every node from the
  skeleton's authored default and only overwrites the ones the clip animates, so the 48 nodes the
  swing says nothing about fell back to BIND POSE for as long as the swing lasted. The engine
  composes a replacement ONTO the current pose, so the locomotion clip now stays the base and the
  swing goes in its own **replacement lane** which overwrites only the channels it actually
  carries — per channel, read off the clip's own presence bitvectors, not guessed. The lane
  refuses any clip that is not type Replacement, so a graph that authors its melee some other way
  is reported in the playback caption instead of posed wrongly, and the caption now counts the
  coverage off the clip in front of it — measured live on the test film, a Spartan's
  `combat:sword:melee_strike_2` carries **34 of 82** nodes and an Elite's **17 of 67**. (This is
  the third clip-type bug of
  the same family: the first-person view had an OVERLAY used as a base clip. The gate here asserts
  both halves — that a node the clip animates takes the clip's value, and that a node it does not
  keeps the BASE clip's value rather than the bind value — and the old sole-clip behaviour is
  reconstructed in a second test and asserted to fail that floor.)
- **Theater: a Spartan or Elite was walking around with no name over it, even though a player was
  plainly driving it.** A recording stamps each body with its player's configuration slot PLUS ONE
  in a field only four bits wide, so a sixteenth player writes 16 and it wraps to 0 — but **0 is
  also what a body carries when the recording attributed it to nobody**, and on 8 of this machine's
  25 films stamp 0 is a CATCH-ALL holding several people's bodies at once. HMS refuses to read a
  catch-all as one player, because putting one gamertag on six bodies is worse than showing none,
  and that left those bodies nameless. There is a second, *independent* statement in the file and
  HMS now uses it: the header says each player's SPECIES (`player_model_choice`, 0 Spartan / 1
  Elite) and the stream says, for every body, the object DEFINITION it was spawned from. The bodies
  the stamp DID name teach the definition-to-species map on that very film, and the leftovers are
  then matched to the leftover slots by ELIMINATION — taken only where it is forced: exactly one
  occupied header slot of that species still unclaimed, a definition class that is a clean species
  discriminator on that film, no two of the leftover bodies alive at the same moment, and every one
  of them inside that slot's own per-tick record range. On the 16-player test clip the stamp names
  40 of the clip's 41 Elite-class bodies for slots 3, 6 and 9; slot 15 (an Elite) is the only
  occupied slot left unclaimed; and the one Elite body left over is therefore his — so it now
  carries its gamertag over its head, can be followed by the camera, and turns up in the species
  cross-check instead of reading as a bare entity id. Bodies elimination cannot force stay unnamed
  on purpose: where every header slot is already claimed the file simply holds no name for them,
  and `film identity` now says which evidence named each body rather than printing a correlation
  score beside a name that was never correlated.
- **Theater: the first-person weapon was not held — it floated on the camera, enormous.** Parenting
  the gun to the hand (above) was only half of it. A built mesh takes its BASIS from the actor's
  anchor but used to take its ORIGIN from the actor's position, and the bone palette then folds in
  `anchor^-1 x world`, which is only a correction if the mesh was baked at exactly the anchor. For
  everything placed in the world those two are the same point, so it never showed; a first-person
  weapon is parented to the hand while its position is still the EYE the rig is mounted on, so the
  gun was baked on the camera and handed a fold-in of identity, which left it there — centimetres
  from the lens, hence enormous. The first build of a fresh scene happens BEFORE the attach is
  resolved, which is why it looked right for one frame and went wrong at the next rebuild (a weapon
  swap, an armour install) — and why a one-shot scripted capture passed it. Origin and basis now
  come off the same matrix, through one helper that returns both.
- **Theater: the first-person arms did not wear the player's armour.** Two separate gaps. The
  COLOURS were never applied at all: the arms are not pooled bodies, so the pass that dresses every
  recorded player skipped them and they were drawn in the arms model's own authored default — a
  plain black glove under a third-person body in that player's tan and green. They now take the
  same primary/secondary out of the same per-slot table (`matg` "profile colors"), through the
  render model's own change-colour masks. The PIECES were being skipped for a different reason: the
  window's camera list is owner-keyed, and the install asked only for the body's own slot binding,
  which a life the recording never bound answers with "nobody" — so nothing was installed. The
  camera's subject now resolves through its owner group as well, which is how the rest of Theater
  already names that player.
- **Theater: the first-person field of view is now a control, in the game's own numbers.**
  Settings ▸ Lighting & rendering ▸ Camera & rendering carries a "first-person FOV" slider in the
  same degrees MCC's own FOV slider uses (horizontal, at the engine's hardcoded 4:3 reference),
  from 60 to 120, defaulting to the **78** Reach's globals tag authors — with a "Reach default"
  button and a live readout of what that is in vertical degrees and in horizontal degrees at 16:9.
  It persists (`first_person_fov.txt`), it is separate from the editor's own fly camera, and the
  script host drives the same saved value with `film fov [degrees]` so a scripted capture frames
  like the window.
- **Spartans and Elites grew thin polygon shards out of the chest, shoulders and back, and a
  third of their armour was never drawn at all.** Reach packs every armour-customisation
  attachment — shoulder plates, forearm pieces, knee guards, belt pouches, ear pieces — into ONE
  "instance mesh" and places ~100 chunks of it around the skeleton. HMS cut that mesh into draw
  calls using each placement's authored index range, but a render_model states those ranges in the
  section's raw TRIANGLE-STRIP space while the buffer being cut is the EXPANDED triangle list — on
  the shipped Spartan, 132,529 strip indices against 204,069 list indices. So the cuts landed off
  triangle boundaries (186 of them on that one model), and because the pieces are then
  concatenated into one list per material, every join invented a triangle out of the tail of one
  attachment and the head of another: a shard stretching across the body, the worst measuring
  0.220 world units on a Spartan 0.7 wu tall. The same mismatch meant 26 of the mesh's 70
  materials — everything living past list index 132,529, 69,510 indices, a third of the armour --
  got no draw call at all. Both passes over the instance mesh now use the same translation (the
  one the bone rebind already used), so the cuts land on triangle boundaries, the runs cover the
  whole mesh, and the biggest armour triangle drawn is 0.116 wu. Gated by a test that walks every
  placement of the shipped Spartan and fails on an unaligned run, an unfed material, or a vertex
  further than half a world unit from the bone its placement names.
- **Theater: the first-person view had no arms in it — the rig was turned 90 degrees to the right
  and drawn off the side of the screen.** The arms were there the whole time, correctly skinned and
  correctly animated; they were just not in the picture, which is why every check that asked
  "is the rig posed?" said yes while the user saw nothing. HMS turned the first-person model -90
  degrees about its own up axis, derived from the arms model's BIND pose, where both arms really do
  reach along +Y. But the bind pose is a T-pose the engine never draws, and **every first-person
  clip poses the rig into the ordinary +X-forward space anyway**. The held weapon settles it: a
  Reach weapon model is authored +X forward (the energy sword's blade node is at `(+0.121, -0.004,
  0)` in bind), the first-person graph poses the arms and the gun from ONE clip in ONE space, and
  posed, that blade is still +X — 2.7 degrees off the axis. So the extra -90 was a second rotation
  with nothing to correct: on an ordinary idle frame it put the left hand **62.3 degrees** to the
  right of the view axis and the sword **87.4**, and **0 of 47 arm nodes and 0 of 2 weapon nodes
  projected inside the viewport**. With it removed the same frame puts the left hand 27.8 degrees
  left and 14.1 down, and across a 156-sample sweep of a 7-minute 16-player clip some of the rig is
  in frame in **97%** of first-person moments. Gated by tests that project the measured rig through
  the real placement and the real projection at four aspect ratios and fail if it leaves the frame,
  goes behind the camera, or ends up all on one side.
- **Theater: the first-person weapon was drawn inside the player's head instead of in their hand.**
  A Reach first-person animation graph is one skeleton over BOTH models — the arms' nodes and the
  gun's — and in that tree the weapon's root is a CHILD of a hand, so the channel the clip plays for
  it is a grip offset LOCAL to that hand, not a position. HMS composed it through the weapon model's
  own two-node skeleton, where that root has no parent at all, so every first-person weapon landed
  at the rig's origin: the energy sword's hilt sat 0.033 wu from the eye while the hand holding it
  was 0.158 wu out, and the blade stood up as a slab across the middle of the screen. The graph's
  own parent index is now read and the gun is parented to the hand, which puts the hilt 0.034 wu
  from `r_hand` — exactly the length of its own authored offset.
- **Theater: a first-person view was rendered at the editor's field of view, not Reach's.** HMS's
  camera is 70 degrees VERTICAL, about 102 horizontal at 16:9; Reach plays at **49.097 vertical /
  78.15 horizontal**. A first-person Theater camera now uses the game's, worked out the way the
  engine works it out: the authored 78 degrees in the globals tag's camera block (`matg` +0x3FC,
  +0x024 — the player bipeds author no override), referenced to the engine's hardcoded 4:3, times
  the render camera's own 0.785. Reach is Hor+, so the vertical is fixed and the horizontal grows
  with the window — the same rule HMS's projection already follows — and a narrow window crops the
  sides rather than the arms. The editor's own FOV is remembered and put straight back when the
  mode is left.
- **Theater: the first-person camera sat 0.05 wu too low for a Spartan and 0.14 too low for an
  Elite.** The eye height was a guessed 0.55 wu. The biped authors it: `bipd` +0x524 `standing
  camera height` is **0.60** for the Spartan and **0.69** for the Elite, and the engine adds it to
  the biped origin along the biped's up axis. Each species now gets its own, chosen from the biped
  the recording names — which matters because both are on screen in the same film.
- **Theater: a scripted capture could show a pose frozen half-way between two animations.** The
  headless host advances the scene with no elapsed time, so a cross-fade started by a seek never
  decayed: the same tick of the same film, reached by two different seek orders, posed a
  first-person hand 0.09 wu apart. A seek is a jump, so the cross-fades are now settled before the
  frame is posed and a scripted capture shows the tick it was asked for, every time.

- **Theater: the player list was washed out and the names were almost unreadable.** Reach's own
  scoreboard row authors a `prop_tint_color` on only three of its five plates; the two it leaves
  out -- the wide plate the player NAME sits on and the block behind the service tag -- are fed by
  the tag's own PROPERTY BINDINGS, which HMS read past and threw away, so both fell back to white.
  The art under them is a single light-grey pixel (`player_base_bitmap` is 180,180,180,
  `player_base_servicetag_bitmap` is 213,213,213), so "white" meant a pale plate painted straight
  under pale text: 1.73:1 contrast on the name and 1.24:1 on the service tag, which is why the
  service tag column looked empty. The bindings are now followed -- `roster_bitmap` takes its
  colour from `bitmap_color_decision_maker`, whose `prop_then` is the player's own bar colour, the
  same colour as the square behind the emblem -- and the row reads 11:1 on a typical bar. Gated by
  a test that no row, named or not, ends up on a pale plate, with the old white default asserted
  to fail the same floor.
- **Theater: the player list did not show each player's own colour.** Two separate faults, both
  measured. The bar colour was taken from `mulg`'s in-world TEAM colours, which is the wrong
  table, and whether a clip counted as a free-for-all was decided by comparing the players'
  recorded team values -- which on a 16-player Forge clip are all 0 whether teams were on or not,
  so every row came out the same. Reach's own law is now followed exactly, out of the engine
  rather than guessed: `scoreboard_list_data` keeps a per-player colour and a per-team colour per
  row and picks between them with `game_engine_has_teams()`, which is the recorded GAME VARIANT's
  own teams bit; the per-player one is `ui_shared_globals` ▸ `tint colors` ▸ `bitmap player`
  indexed by that player's primary armour colour, and the per-team one is `bitmap team` indexed
  by their team, except that a team index above 7 (including "no team") makes the engine copy the
  player's colour instead of looking anything up. An index the list cannot satisfy is not clamped
  and has no default -- the engine leaves such a row transparent, where HMS draws a dark neutral
  so the row is still visible.
- **Theater: the scoreboard's service tag, rank and score were aligned wrongly.** `cui`'s
  `prop_justification` is `0` left, `1` **right**, `2` **centre** -- not the order HMS assumed --
  so the three columns that author `2` were being drawn hard against the right edge of their
  blocks: the rank number jammed against the colour square, the service tag against the score
  block, the score against the row's own edge. All three are centred in their blocks now. The
  mapping is engine-proven twice: `text_justification_enum_definition` lists its options as
  `left`, `right`, `center` in value order, and the layout function in `draw_string.cpp` writes
  the pen as `right_bound - text_width - font_pad` for `1` and
  `left + (width - text_width) / 2` for `2`, with the value reaching it from the widget verbatim.
  A survey of all 83 `cui_screen` tags in the map is consistent with that, and reading it alone is
  what got the order wrong the first time. Also checked while in there: the font's own string
  measurement agrees exactly with what it paints, at three scales over ten real service tags --
  which is the other way a justified column can sit wrong, and is now gated.
- **Theater: the whole row carries the player's colour, not just the block at the left.** Reach
  tints three of the row's five plates with the player's colour and they tile 275 of its 350
  units, but it does so through a UI palette that is a darkened copy of the armour colours, so
  most of the row painted as near-black and only the small block read as coloured. Which plates
  are the player's is now taken from the tag -- the ones whose `prop_tint_color` binding chain
  ends at `prop_player_bitmap_tint_color`, so nothing is named by hand -- and all of them take
  that player's real primary colour. The two plates that author a near-black tint of their own,
  behind the rank and the score, keep it, as the engine does. For the record, the engine really
  does draw a dark row: its composite is `tint x art` with the widget's opacity folded into alpha
  alone, which puts a `Green` player's row at about 6/255 per channel at 95% alpha, so this is a
  deliberate departure and not a correction. Readability is gated over every colour the row can
  now be: all 30 armour colours, all 8 team colours and both text plates, worst case 2.45:1.
- **Theater: a named player with no camera of their own was listed as having never played.** A
  player can be named by elimination at the level of an individual BODY while having no owner
  group, and those rows hardcoded no lives and no first appearance -- on the 7:11 test clip, slot
  15 read `lives=0` while one of their own bodies sat in a separate unnamed row. Both now come
  from that slot's own bodies. The unnamed group still does not claim the name.
- **Theater: the explanation for an unnamed body was wrong.** It said those players "joined
  afterwards, so their names were never written to the file at all". The real reason is the
  format: a body says whose it is through a field holding `index + 1` in **four bits**, where 0
  means "nobody", so it cannot name more than fifteen players and on a full lobby at least one
  player's bodies are attributed nowhere -- and nothing else in the file attributes them either,
  because the runtime player-to-unit link is never replicated and the one channel carrying
  per-player object references is absent from films entirely. Late joining is a separate,
  additional reason some players have no header record at all; the sentence now gives both and
  keeps them apart.
- **Theater: a player's primary colour is now actually visible on their row.** Reach indexes a
  UI palette with the primary armour colour rather than using the colour itself, and that palette
  is a heavily darkened copy: nine of the sixteen players in the test clip index entries that
  paint as essentially black (`#090909` for *Steel*, `#031406` for *Green*, `#071326` for
  *Blue*), so at the size this panel draws, whose row it was could not be told. The square at the
  left of each row -- the one plate the engine puts the player's colour on and draws no text over,
  identified from the tag by having the emblem widget on it and no text column -- now carries
  that player's REAL primary colour, read from the same `matg` `profile colors` entry at the same
  index that their body on screen is tinted with, so the panel and the picture cannot disagree
  about who is teal. That table stores LINEAR floats and is sRGB-encoded for display, which is the
  difference between *Steel* reading as the mid grey it is on armour (`#646464`) and as near-black
  (`#202020`). Every plate a name or a service tag sits on keeps the colour the tag's own bindings
  give it, and the text colour is chosen per row from `ui_shared_globals`' own two (its light
  default and its dark `body` preset) by measured contrast -- which is what makes a name readable
  on the one pale entry in each list (*White* and team gold). Gated over all 30 armour colours,
  all 8 team colours and both plates.
- **Theater: the player list's colours were invented.** The two text colours were "mapped by eye"
  to hardcoded RGB and the colour bars used `mulg`'s in-world TEAM colours, which are the wrong
  table: Reach resolves a `cui` text widget's `prop_color_preset` against `ui_shared_globals`
  (`wigl`), and a widget that names no preset -- which is the name, the position and the service
  tag -- keeps that tag's own `text color` rather than drawing white. All twelve presets are now
  read from the map (`white` is 0.847, not 1.0; `body` is a DARK 0.35, where the by-eye value was
  nearly twice as bright), and the bars take `wigl`'s own UI tint lists, where team red is
  136/21/21 and not 255/0/0. A free-for-all clip -- which is what a Forge social game is, and all
  sixteen slots of the test clip record team 0 -- colours each row by that player's own armour
  colour instead of painting sixteen rows team red: `bitmap player` has one entry per armour
  colour and is the same hue index for index as `matg`'s `profile colors`, identical on four of
  the thirty. Names are also drawn over a drop shadow in the colour `wigl` authors for one.
- **Theater: the emblem column said a film "has not got" the emblem, which was false.** Every
  recorded player slot carries an `s_emblem_info` -- foreground design, background design, flags
  and three colour indices -- at player record +0x50, which is where the engine's own arithmetic
  puts it (a 0x20-byte identity header, then `s_player_configuration`, whose `client.appearance`
  is at +0x28, whose emblem is at +0x08), and 113 of 113 recorded players parse as a valid one
  including the engine's must-be-zero pad. What is true is that MCC stores a real emblem only for
  the profile it has and fills the other slots with one shared placeholder: fifteen of sixteen
  identical on one clip, and the same placeholder turns up in unrelated recordings with unrelated
  players. The panel now says that, counted from the clip in front of you. The art itself is still
  not drawn -- Reach composites an emblem from signed-distance shapes in `mlib`, whose authored
  blocks are compiled away at map build, and MCC runs a different path against a texture pack that
  is not decoded here.
- **Theater: a camera for a body the recording never named read as an object handle.** The row
  showed a bare 32-bit entity id (`0xe9750001`), which is not a stable handle for a player either
  -- a body gets a new entity every respawn. It now reads `Spartan 7` / `Elite 12` / `Monitor 3`,
  from the `bipd` tag the recording says that body was spawned from plus its owner group, which is
  unique for the life of the clip; with nothing resolved at all it reads `Player 7`. Measured over
  a twelve-film corpus: the clips that produce these rows stored 1, 2, 3 or 9 gamertags in their
  header while 11 to 16 players had a body, and the ones that produce none stored 16 -- so for
  most of them there is no name in the file to recover. The camera list is now also rebuilt once
  the map has said what each body is, instead of being stuck with whatever was known when the
  index landed.
- **Forge objects were too bright: their environment reflection ran 2.3x the engine's.** The
  per-part material work was right in character -- the panels are authored glossy and they do
  reflect -- but the LEVEL was wrong, because HMS fed the reflection the wrong specular mask. A
  Reach material whose `specular_mask` option is `from_diffuse` is masked by the whole COMPOSED
  albedo alpha, `base.a x detail.a x albedo_color.a`, saturated -- and that one number scales the
  analytical highlight AND the environment reflection alike. HMS used the base alpha on its own,
  dropping both of the other factors, so every Forge piece reflected its cubemap through a mask
  1/(detail.a x albedo_color.a) too open: across the 46 Forge shaders on Forge World that product
  averages **0.439** (`for_metal_scratch_detail` alone averages a = 0.434, and the panels author
  `albedo_color.a` of 0.1, 0.15, 0.4, 0.5, 1.0 and 2.0). Measured on The Cage, over the whole
  object population of the frame: environment fell from **64% to 46%** of a Forge object's
  radiance and the objects as a whole came down to **0.65x** linear (about 20% darker after the
  tonemap) -- coliseum wall 0.64x, grav lift 0.44x, 2x2 block 0.75x, 3x4 platform 0.76x -- while
  the diffuse group moved by under 1%, so the shading and reflectiveness that was just fixed is
  untouched. The detail map is now bound for every object class for its ALPHA only; its COLOUR
  still stays off the Forge classes, so their albedo is exactly what it was. Vehicles, weapons and
  equipment are unchanged, because that lane still passes a blanket `from_diffuse` flag rather
  than each part's own option and `albedo_color.a` cannot be applied on top of an unknown one.
  `HMS_LEGACY_OBJMAT=1` still restores the pre-material-lane look, byte for byte. Gated by tests
  on the composed-alpha law, on the per-class lane packing, and on the shader source itself (every
  pre-fix form is now a hard failure).
- **Theater: in a full 16-player lobby the sixteenth player was never named.** The recording
  stamps each body with its player's configuration slot PLUS ONE, in a field only four bits wide,
  so slot 15 writes 16 and wraps to 0 -- and the reader took 0 to mean "nobody owns this body".
  An owner-0 group is now read as that sixteenth player, but ONLY where the recording really has a
  slot-15 player AND the group behaves like one: stamp 0 is also what an unattributed body
  carries, so on many films group 0 is a CATCH-ALL holding several players' bodies, and naming it
  would put one gamertag on bodies belonging to different people. The structural test is the same
  one that identifies the owner field in the first place -- one player cannot drive two bipeds at
  once, so a real group's lives never overlap. Measured over the film corpus: of 25 films, 13
  carry an owner-0 body; 2 are the genuine wrapped sixteenth player, 8 are catch-all buckets (one
  holds an Elite body spanning 10.9s-322.8s alongside three Spartan bodies alive together over
  274.7s-373.9s), and the rest are lobbies with no slot 15. Only the first kind names anyone; the
  others stay unnamed and are listed by what the recording says they are. Gated from both sides.
- **Theater: a scripted/headless capture of a film drew every body in its DEFAULT armour while the
  window drew the recorded armour and colours.** The recorded-appearance install lived only in the
  window, so the two hosts showed different casts -- which also made the scripted path useless for
  checking what is actually on screen. Both now call one implementation on the playback driver.
- **Theater: one player's armour could follow a recycled body onto somebody else.** Actors are
  pooled and reused between bodies and lives, and a body with no recorded appearance left the
  PREVIOUS occupant's armour variant in place instead of clearing it back to the model's own
  default -- including across a change of species.
- **Theater: the first-person view posed an Elite's hands with the SPARTAN's animation.** A
  weapon's `player interface` block carries one row per player species, in the same order as the
  globals' `@player representation` -- row 0 Spartan, row 1 Elite -- and Reach authors a separate
  Elite first-person graph for every weapon. HMS always read row 0, so an Elite's 41-bone arms
  were driven by the Spartan's 47-bone graph. The row now follows the biped the recording names.
- **Theater: the spectated-player card is drawn, from the game's own widget.** While the camera
  is following one player, Reach shows a bar along the bottom of the screen naming whose camera
  you are on; HMS now draws that same bar. It is a `cui_screen`,
  `ui\cui\in_game\film_playback\film_cameras`, and everything about it comes out of the tag:
  the `film_control_pad_list_item` subtree at (437, 603) in the 1152 x 720 authoring space, its
  `white_pix` edge rules, the two `player_base_bitmap` fields tinted to the authored 0.093 grey,
  the `player_base_servicetag_bitmap` panel, and the player's name and service tag in
  `tv_nord_condensed-15` (`prop_font_id` 7), the tag right-justified as it authors. The list
  item's nominal cell is 350 wide but the art stops at 277, and it is 277 that centres the bar
  (437 + 138.5 = 575.5, dead centre of 1152) -- taking the cell instead would sit it 36 units
  right of where the game puts it. Two things are deliberately left out and the diagnostic says
  so: the player's EMBLEM, because Reach composes emblems from `mlib` layers MCC does not even
  run, so the 28 x 28 slot stays empty exactly as the scoreboard leaves it; and the d-pad
  up/down hints, which name a controller HMS does not have. The name is read through the same
  accessors the camera itself uses, so the card and the camera can never disagree. *View ▸
  Spectated player card* turns it off and on and the choice is remembered. New `--dump-spectate`
  prints the whole authored layout, and `--list-ui` lists every UI tag in a cache.
- **Theater: the floating name sat 0.025 world units too high on everyone but a teammate.** The
  two name displays take the same `head` marker but lift it differently --
  `haloreach.dll+0x2CB76C` adds 0.075 wu for the ally waypoint and `+0x18AE04` adds 0.050 for the
  look-at name (its `Object_GetWorldPositionAndOrientation` is the same
  `Object_BuildWorldTransform(unit, 0xC2, ...)` call). HMS was using the waypoint's lift for
  both; each body now gets the one its own display uses, and the no-marker fallback follows the
  look-at value because that is what almost every body draws.
- **Theater: the navpoint chevron was drawn over everybody, including in a free-for-all where the
  game draws none.** Reach has TWO over-the-head name displays and the first pass implemented the
  wrong one for the common case. The **ally waypoint** is the record's own gamertag with the
  chevron beside it, and `haloreach.dll+0x2CB76C` creates that record ONLY for another player on
  the local player's own team -- which is why a gametype hands it out through player traits, and
  why an enemy, a free-for-all opponent and your own body never have one. The **look-at name** is
  a separate string (`a1+0xAD8`, whoever you are looking at, with its own fade at `a1+0xB00`) and
  is plain text with no sprite, which is why the same code centres it on the anchor with no
  offsets. HMS now draws the look-at display for every body and adds the chevron only where the
  engine would: teams on, and the body a teammate of the viewer. Both take their colour from the
  same place (`+0x2E3AD8`, the record's colour index), so the palette is unchanged -- a
  free-for-all stays `navpt enemy` red. Neither display is a `chud_definition` widget: every
  `chdt` in a shipped cache was decoded (the new `--dump-chdt`) and not one has a `players`
  datasource or a `<player>` anchor, so this hand-written renderer is the whole of it.
- **Theater: the floating player names were the wrong size and had no chevron under them.** The
  chud never converts a size on the CPU -- it hands the GPU vertices in a VIRTUAL screen and lets
  the shader map that onto the safe frame -- so every authored HUD size is a size in Reach's own
  1280x720 virtual screen, and one virtual unit is `window height x 0.87 / 720` on screen (both
  numbers read from `ui\chud\globals`' `curvature infos`). HMS was painting the font at its
  authored 21 pixels no matter how big the window was, which is right at exactly one window size
  and wrong at every other; names now scale with the viewport the way the game's do, still times
  the 0.8..1.3 distance scale. **And the navpoint chevron is drawn**: the `chgd` skin's
  `navpoints reference` is a sprite sheet (`ui\chud\bitmaps\waypoints`), the engine takes
  sequence 1 sprite 0 from it for a player name, and that sprite is a 37.5 x 21.6 virtual-unit
  downward arrow stored 3x supersampled -- decoded straight out of the map, tinted the same colour
  as the name, point on the anchor with the name stacked on top, which is the engine's own layout
  for a record with no secondary sprites. Nothing is drawn behind the text: no plate, no outline,
  no shadow, because the engine issues one draw per sprite and one for the string and the string's
  command carries only a colour and an opacity.
- **Theater: the floating player names sat too high and were the wrong colour.** The name over a
  head is a chud NAVPOINT, and the engine anchors it on a marker it names by `string_id` 0xC2 --
  which, resolved against a shipped cache's own string table, is **`head`**, not the
  `chud_nav_point` marker HMS was using. (An earlier pass read 0xC2 as `"chest"`, found no biped
  with such a marker, and concluded the engine could not be naming one at all.) `chud_nav_point`
  exists but sits further up the same bone, so names floated 0.762 world units over a Spartan's
  feet instead of the engine's 0.708 and 0.929 over an Elite's instead of 0.800 -- and a body
  whose marker could not be posed fell back to an invented 0.85, half a metre high. All three are
  now the measured thing: the posed `head` marker plus the engine's own 0.075 wu lift
  (`haloreach.dll+0x2CB76C`), with the fallback set to that same anchor measured on the Spartan's
  bind pose. **The colour is now Reach's own, picked by Reach's own rule.** Every navpoint colour
  is one of thirty-two `real_argb_color` fields in `ui\chud\globals`' skin, selected by a colour
  index (`+0x2E36B8`), and exactly one engine function compares teams to choose it (`+0x14844C`):
  same team -> `navpt normal` (#2494FF), different team -> `navpt enemy` (#D83232), no team ->
  `navpt neutral` (#5E5E5E). HMS reads those three out of the open map and takes the viewer to be
  whoever the Theater camera is on, or the player who recorded the film when the camera is free --
  a saved film is one machine's recording, so that player is the local player the engine itself
  had. A free-for-all clip therefore reads all red, which is what the game shows, because Reach
  puts every player in a teams-off game on their own team; the saved player slots all still read
  team 0 (16 of 16 on the measured clip), which is why the recorded game variant's own *teams
  enabled* flag decides this rather than those bytes. HMS's own invented team swatches are gone
  from the names. Still no background plate, still the game's own `tv_nord_hud-15`, and the two
  distance-scale clamps (0.8 / 1.3) are now READ from the tag instead of typed in. `--dump-chud
  <map> [film]` prints the lot: the anchor marker on every biped with its measured height, all
  three authored skins' thirty-two colours, the engine's index map, and the colour every player in
  a film resolves to.
- **Theater: the first-person arms were frozen in their bind pose -- both arms straight out, no
  animation at all.** A Reach first-person graph is a BASE clip plus OVERLAYS, and the tag says
  which each one is: `first_person:idle`, `:sprint_loop`, `:fire_1:var*`, `:melee_strike_*` and
  `:reload_*` are BASE clips, while `:moving`, `:fall`, `:jump`, `:land_hard`, `:land_soft` and
  `:look` are OVERLAYS. HMS handed `first_person:moving` to the rig as a body's ONLY clip, and an
  overlay composes onto whatever pose it is given -- here the skeleton's authored default -- so the
  view became the arms' bind pose plus a small delta. A player in a 16-player clip is moving almost
  every frame, so almost every frame drew that. The base is now always a real base clip and the
  gait clips ride on top of it as the overlays they are. The two DECODED actions also reach the
  view now: a recorded melee swing plays `first_person:melee_strike_1` / `_2` and a recorded shot
  plays `first_person:fire_1`, each once from the tick the recording says it happened, so scrubbing
  backwards un-plays them. Gated by a test that drives the clip chooser over every combination of
  its inputs and checks each name's animation TYPE in every shipped first-person graph -- which the
  old chooser fails.
- **Theater: the first-person caption claimed a pose it had not checked.** It reported the clip it
  MEANT to play; it now reports what the rig is actually doing, read back off the actor -- the clip
  name (or "BIND POSE (no clip loaded)"), the frame, and how many of that model's bones the graph
  actually drives.
- **Theater: the weapon was missing from the first-person view.** The arms were posed by the
  weapon's own first-person animation but the gun itself was never drawn, so a player looked at
  their own empty hands. A Reach weapon's `first person model` is its ORDINARY render model, and
  the first-person graph lists that model's nodes (`b_handle` / `b_blades` for the sword, `b_gun`
  / `b_magazine` / `b_ophandle` / `b_safety` / `b_switch` for a rifle) beside the arms' own -- one
  clip poses both, in one space -- so the gun now rides the same rig as the hands, with the same
  graph, clip and frame, and needs no marker attach. It used to come out as a blade stretched
  across the view for the same reason the arms did: the graph-to-model node match was scored
  against the GRAPH's node count, which a weapon model can never satisfy, so it fell back to
  matching bones by position. Gated by a test that every first-person graph maps onto its own
  weapon model **by name**, which a positional map cannot pass.
- **The Theater `film camera` script verb changed the camera when you only asked which one was
  live.** A bare `film camera` defaulted to `free`, so querying the camera dropped a first-person
  or follow camera on the floor. It now reports, like every other verb in that family.
- **A map variant could not be converted TO Halo 2 Anniversary, and converting to Halo 4 could
  drop pieces without saying so.** *File ▸ Convert variant to another game…* greyed H2A out as a
  target and the converter refused it outright, both on the grounds that the encoder wrote Halo 4's
  body versions only -- no longer true, since it writes H2A's 52 and 53 as well. Behind that sat a
  worse fault: the Halo 4 / H2A writer looked each target piece up in **Halo 4's** catalog first and
  only then in the target title's, and the two titles name many Forge pieces identically
  (`fw_stone_plat_10x10#bb_10x10` is in both), so an H2A target resolved to a Halo 4 object that
  had no entry for an H2A canvas -- and the placement was dropped **silently**, with the report
  still counting it as converted. Zealot onto Ascension wrote 51 of its 267 pieces. Every lookup is
  now against the title being written, and all three ways a placement can fail to resolve report
  why, as the Halo 3 writer already did: the same map now converts 267 of 273 objects (95.6%),
  parses back as a body-v52 H2A variant and re-encodes byte-identically.
- **A test that was meant to pin the death instant never ran.** A stray `#[test]` attribute landed
  on the wrong function, so the gate that checks a death is the last reported POSITION of a
  destroyed track (not its last message, not the destroy) was compiled as dead code. Reattached;
  it passes.
- **Theater: the first-person arms stuck straight out, whatever weapon the player was holding.**
  Reach's first-person arm models carry no animations of their own -- every pose comes from the
  *weapon's* own first-person graph -- and matching that graph's bones to the arms model was
  scored the wrong way round. A first-person graph deliberately lists the arms' bones **and the
  gun's** (`b_gun`, `b_magazine`, `b_ophandle`, `b_safety`, `b_switch`, and so on), so a weapon
  was being punished for carrying its own gun: six extra bones out of 52 put the assault rifle
  under the threshold, eight put the needler at 76%, the match was thrown away, and the arms were
  posed by bone *position* instead -- which on these two lists is nonsense, because the arms model
  orders its bones `pedestal base aim_pitch aim_yaw l_upperarm camera_control ...` and the graph
  orders them `pedestal aim_pitch aim_yaw base camera_control l_upperarm ...`, so every channel
  from the second onwards drove the wrong bone. **14 of the Spartan's 27 first-person graphs and
  all 27 of the Elite's** failed that way; all **54** now map, with every arms bone driven.
- **Theater: an Elite in first person was given Spartan hands.** Halo Reach's globals carry one
  first-person representation *per species* (`objects\characters\spartans\fp\fp`, 47 bones, and
  `objects\characters\elite\fp\fp`, 41) and HMS took the first row in the list. The row is now
  chosen from the biped the recording names for the body the camera is riding, decided by the
  model's own armour regions rather than by its tag path.
- **Theater: the first-person arms did not wear the player's armour.** They have no model tag, so
  the ordinary armour walk cannot reach them, and the fallback pinned the `arms` region to its
  default and dropped `flair_forearm` altogether -- that region has no `default` permutation, so a
  player's own forearm piece could never appear in their own hands. The recorded player's chest
  and wrist selections (and an Elite's armour set) now choose the permutation directly on the
  render model.
- **Theater: nobody was ever shown looking anywhere.** A body was turned by its recorded body yaw
  and nothing else, so a player who was aiming up, down or across their own shoulder was drawn
  staring level down their own facing. Reach poses a look direction as an *overlay* blended onto
  the walk, and the clips label their own frames with the angle each one stands for -- read off
  `combat:rifle:aim_still_up`, a 3-column by 5-row grid, yaw `0 / -55 / +73.5` degrees and pitch
  `+90 / +45 / 0 / -45 / -90`, in that authored order. HMS now blends the four surrounding poses
  of whatever grid the clip really has, which differs clip by clip and species by species, and
  composes the result onto the locomotion pose the way the engine's own overlay combiner does.
  **A recording states one player's aim exactly -- the recorder's** -- and every other slot's aim
  field is replicated and zero-filled, so one body per clip is posed with a real look direction
  and the playback panel says so instead of letting the rest look deliberate.
- **Theater: melee swings were invisible.** A player who swung kept walking. The recording states
  every swing outright (`biped_melee_initiate`, 9,567 of them over the test clips), and each one
  now plays that weapon class's own strike -- once, from the tick the event fires, holding its last
  frame, so scrubbing backwards un-swings it.
- **Theater: shots had no recoil.** `action_weapon_fire` is on the wire for every body, and the
  weapon class's `fire_1` clip is now composed over whatever else the body is doing, so a burst
  reads as a burst. (The projectile itself is still not drawn.)
- **Theater: a scripted seek reported "NOBODY is holding a weapon" while the app showed a whole
  team armed.** Two faults at once. The weapon count in the playback caption was read off the last
  *drawn* frame, so between frames -- which is exactly when a script asks -- it was still zero; it
  is now counted at the tick being asked about. And the scripted runner never applied the game
  variant's loadouts at all, so in it no body held anything; it now takes the same step the app
  does, and reports 15-17 bodies armed where it used to report none.
- **Forge pieces had no reflections, and some had a shine nothing in the map asked for.** Every
  Forge object was shaded with the constants of its FIRST material, applied to all of its parts,
  and no Forge object could show an environment reflection at all. Two things looked wrong at once.
  A piece whose first material happens to be its glass pane -- the 3x4, 4x4 and 5x5 platforms, the
  pyramid, the bunker, the man cannons, the teleporter receiver, **85 parts** over the Forge World
  palette -- found no specular value to read, and HMS filled the gap with the part's own texture
  alpha at full strength, which is where the "coliseum walls look a bit shiny" sheen came from;
  those parts now use the value their material really authors and the invented highlight drops away
  (**0.59x** the specular on a 3x4 platform). In the other direction **944 parts** across **183 of
  the palette's objects** author an environment reflection that HMS was discarding, among them the
  gravity-lift catwalk the reports singled out, which asks for a reflection of its own captured
  cubemap at five times strength on four of its six parts and now gets it. Each part also takes its
  own roughness (**441 parts** change) and its own fresnel and specular tint, so a glossy trim no
  longer inherits a matte panel's roughness. Read out of the shipped shaders instruction by
  instruction, so the reflection is built the way the engine builds it: the cube sampled at the
  authored roughness offset, multiplied by its own alpha, by the material's coefficient and
  contribution, by the environment tint and by the lit area of the surface, then added.
  Bipeds, vehicles, weapons and equipment already read their materials per part and are
  **pixel-for-pixel unchanged**; so are both calibrated Ivory Tower reference frames.
  Forge objects are brighter for it -- about **1.4x** the light coming off them across a 373-object
  Pinnacle scene, with the reflection making up roughly **half** of a typical Forge panel -- which
  is the authored amount but cannot be checked against a real engine capture, because no capture
  with Forge objects in it exists; `HMS_LEGACY_OBJMAT=1` goes back to the old look.
- **Verified against the old behaviour on a close-up.** The Coliseum wall, same camera, with and
  without the change: the old path drew it as a near-black slab with the mesh's own triangulation
  glowing in yellow seams (the fabricated specular mask at full strength, reading the triangle
  edges), and the new one draws the authored panelled wall with its surface detail. The measured
  brightening of that piece is it ceasing to be a black mirror, not a piece becoming too bright.
- **Theater: dead bodies floated in mid-air.** A Reach film states no position at all for a body
  once it has gone down -- measured at **0 reports after the death on every one of 1,662 corpse
  records** on a 47-minute clip -- so a corpse was left hanging at whatever sparse position packet
  happened to be its last, which on that clip is **up to 1.0 wu (three metres) above the floor**,
  for the whole twenty seconds a body now lies there. Each body is now LANDED: one downward cast
  through the map's own collision (the BSP and the placed objects, the same instrument the camera
  stand-off solver uses), then a free fall at **10.46 wu/s**, which is this engine's own gravity
  fitted off 1,447 falling samples in the recordings rather than converted from Earth gravity. A
  body already on the floor is untouched, a body still dropping keeps the `airborne_dead` flail
  and only starts its landing clip when it lands, and a body with no floor within 60 wu went over
  the edge and is left where the recording put it -- the panel counts all three, and says the
  landing is derived rather than decoded. `film corpses trace` prints the numbers per body.
- **Theater: the panel now says that a body's LOOK direction is not drawn.** A drawn body is
  turned by its recorded body yaw and nothing else -- Reach poses where a unit is looking with
  per-weapon-class aim overlay clips blended over the walk, which HMS does not play -- so every
  body looks level and straight down its own facing however the player was really aiming. The aim
  itself is decoded (it is what drives the first-person camera); the overlay layer is missing, and
  the panel now says so instead of leaving it to be read as a bug in the facing.
- **Halo 3 looked about a fifth too dark everywhere below white, because it was being put through
  Halo: Reach's final curve.** Halo 3 is the only one of the four games whose composite ends on a
  tone curve -- the engine clamps to a white point and runs a cubic, `1.5c - 0.5c^3` at the
  shipped white point of 1.0 -- and HMS gave every game Reach's bare gamma-2 encode instead,
  which has no shoulder and sits consistently lower. Halo 3 maps now use their own curve: measured
  over five maps at a fixed camera, the frame mean and median both rise 1.22x with the brightest
  highlights unmoved (Guardian mean 15.1 -> 18.6, Construct 46.0 -> 56.1, Valhalla 71.8 -> 84.7,
  Isolation 29.7 -> 36.3, Snowbound 27.7 -> 33.8, out of 255). Reach, Halo 4 and H2A are
  untouched -- their engines genuinely have no tone curve.
- **Nothing could be placed into a Halo 3 map: dragging a palette object into the viewport did
  nothing at all.** Opening a Halo 3 *map* leaves no variant open, and every Halo 3 placement was
  refused outright on that ground -- a viewport click, the Line and Fill tools, the `place`
  script verb and a palette drag-and-drop alike -- because the Halo 3 encoder writes into a copy
  of a real `.mvar` and there was nothing to copy. A refused drag showed nothing but a line in
  the status bar, so it read as the drag being broken. Placing on a Halo 3 map now adopts one of
  that map's own shipped variants as a blank canvas on the first drop (its Forge bounds, budget
  and scenario-owned slots, none of its placed objects), so the object lands, moves and saves;
  **File > New variant** does the same deliberately instead of telling you to go and open a
  shipped variant, and if the map genuinely has no variant to start from the refusal now says so
  in both status lines rather than vanishing mid-drag.
- **Elites in Theater wore nothing a player had chosen, and could wear a Spartan's armour
  effect.** Reach stores a Spartan loadout *and* an Elite loadout for every player, and a
  recording carries both: ten bytes, of which eight dress a Spartan and two dress an Elite -- the
  Elite's whole armour is one choice of eight (Minor, Spec Ops, Space, Ultra, General, Officer,
  Story, Zealot) plus an armour effect. HMS was applying all ten to whichever body it was
  drawing, which is wrong in both directions: the two armour-effect choices target the one piece
  of a model both species share, so a player's Spartan effect could end up on their Elite, and a
  helmet choice reached into the Elite's head. An Elite body now wears the player's Elite choice
  and a Spartan theirs, decided by the render model being drawn rather than by the player -- so
  over the 30 recordings on this machine, the 28 player-lives that were Elites are dressed from
  the recording instead of all looking like a Minor.
- **Elites carried their energy daggers everywhere.** The two wrist blades were drawn on every
  Elite, standing, running or dead. They are not an armour piece and not a prop -- they are
  baked into the Elite's body mesh with no bone, marker or region of their own, so nothing in the
  model can switch them off. What switches them off is that every one of their shader parameters
  is driven by the Elite's `blade_left_on` / `blade_right_on` functions, and the Elite's own
  function block feeds those from the engine's per-animation EVENT channel: the blades ignite for
  the frames of an assassination and do not exist the rest of the time. HMS now leaves out any
  effect whose every driver is an animation event, so an Elite at rest has bare forearms. (They
  are still not lit during an assassination -- that needs the playback to raise the event, which
  is coming separately.)
- **Most assassinations are not in the recording at all, and Theater now says so.** The user
  reported that assassinations do not play and the victim just drops. Measured on their own
  `Chill Assassinations.` clip: the recording destroys **163 bodies** but its event channel
  carries only **16 kill rows, 8 of them assassinations** -- and all 16 name the *same* killer, a
  player with one life and no deaths, i.e. whoever recorded the film. A Reach film is one
  client's view and medals are personal, so nobody else's assassination is on the wire. The eight
  it does state are staged correctly, including cross-species pairs (the Spartan half and the
  Elite half of one kill live in different animation graphs and are now looked up separately);
  for the rest there is nothing to stage and the body falls with its death clip. Two attempts to
  infer the others were measured and rejected -- one needed `biped_melee_initiate`, which the
  killer of all eight stated assassinations never fired once, and one rested on the two bodies
  interpenetrating, which turned out to be a close-range-melee detector (1.58x contrast against
  the 1.8x this project already threw out, 87-90% overlap with the melee test). The numbers and
  both negatives are in `film_anim::inferred_assassinations`.
- **Theater idle animations played several times too fast.** A clip with no footsteps to drive it
  -- an idle, a turn on the spot, an airborne arc -- was advanced at a flat 1.5 cycles a second
  whatever had loaded, because the fallback rate hard-coded a 20-frame clip. The shipped Spartan
  idle is **156 frames**, so a player standing still breathed **7.8x too fast**, and every other
  clock-driven clip in the graph was wrong by its own different factor. Each one now plays at the
  length it was authored at (156 frames = 5.2 s a cycle), taken from the clip that actually
  loaded. A second, smaller version of the same fault: the learned clip lengths were keyed on the
  weapon class but not on the CHARACTER, so a Spartan's 156-frame idle and an Elite's 50-frame
  one shared an entry and whichever loaded last drove the other at 3.1x the wrong speed.
- **Elites did not fall over when they died.** The Elite's ground-death clip is named
  `combat:landing_dead`, with no `any:` field, and the three names HMS tried were the Spartan's
  (`combat:any:landing_dead` and two `landing_dead_back` variants) -- so an Elite matched none of
  them and stood up in its idle instead. The death clip is now looked up in each body's own
  animation graph, both species' sets are enumerated in full off the shipped cache, and the panel
  names the clip every dead body is actually lying on rather than leaving it to be assumed.
- **Dead bodies vanished after five seconds instead of lingering.** A body now lies where it fell
  from the moment the recording says it died until the tick the recording DESPAWNS it -- the
  entity-destroy message, a median 16 s later, which is the despawn the game itself performed.
  HMS used to clear every corpse after a fixed 5 s, so bodies disappeared while the game would
  still have been showing them. Up to 14 at once, and still never at the expense of a live player.
  A body killed in mid-air now falls with the `airborne_dead` flail instead of the ground
  landing, chosen from its own recorded vertical speed. **Which** ragdoll really played is not in
  the recording and the panel says so: a Reach death is a physics simulation, and the only thing
  on the wire that could narrow it (a damage direction buried in `damage_aftermath`) is not
  decoded, so the clip is picked deterministically from the body's id and replays identically
  every time.
- **A body could be drawn alive and dead at the same tick.** The living bodies and the corpses
  worked out the moment of death two different ways, so a kill the recording stated before the
  body's last reported position put two Spartans on one player. Both read one answer now.
- **Opening a map variant left the recording's bodies standing in it.** The Theater clip is now
  closed when you open a variant of any game -- playback stops, the bodies go, the cast list
  clears -- because a recording's bodies are a statement about the scene it was recorded in and
  mean nothing in another one. Opening a clip still draws its own Forge scene, which arrives
  through the same path and is no longer mistaken for you opening a variant.
- **The Theater timeline was in the File menu.** It is a panel you show, not a file you open, so it
  now sits in *View* beside the other panel toggles.
- **The energy sword has its two glowing blue blades.** A dropped or carried sword drew a pale
  white smear where its blades should be. Two separate faults. The blade is five overlapping
  additive layers, and three of them were falling through to the renderer's effect-sprite
  fallback, which paints a part's artwork as an additive billboard with a radial vignette, a rim
  glow and a slow pulse -- none of which the game applies to geometry; that fallback painting the
  blade's artwork white is what produced the smear. And a Reach shader's parameters are not the
  numbers sitting in the tag: most of this sword's are ANIMATED, driven through a mapping
  function by one of the object's own functions, and the numbers in the tag are the snapshot with
  every one of those functions at zero -- a sword switched off. HMS now reads each object's
  function block (what drives each function, and what turns it off), evaluates the real mapping
  curve, and draws objects in the state a viewer should show them in: present, intact, powered
  and unused. For the sword that means its blades are turned on, it is not stowed on anyone's
  back and it is not cloaked, which is what lights all three blade layers and brings the primary
  blade colour -- weighted at zero in the tag -- up to its real strength. Measured on an
  identical frame, the blade's peak brightness went from 0.42 to 2.9 and its blue-to-red ratio
  from 1.5 to 3.7.
- **Parked Forge vehicles no longer have their headlights, brake lights and dashboards lit.** The
  same change: a truck's lights are animated parameters driven by whether someone is in the
  driver's seat and on the throttle, and reading the tag's resting numbers lit them all at full
  strength on an empty parked vehicle. Five of Forge World's civilian vehicles are affected. An
  armour-ability pickup's glow, which is driven by its charge, came out slightly brighter for the
  same reason, and the covenant drop shield is unchanged.
- **Energy blades and vehicle accent lights had their red and blue exchanged.** The three-channel
  self-illumination the engine uses for them treats a map's red, green and blue as three separate
  masks for three authored colours, and HMS was reading (and writing back) those three channels
  in the wrong order. Twenty-five objects use this shading -- the sword, most Covenant vehicles,
  the Scorpion and several props.
- **Theater: first person no longer shows nothing at all.** It used to put the camera at the
  player's eye and show none of them; it now shows the arms (below).
- **Weapons and equipment lying in a map are now the size the game draws them, about a third
  bigger.** Reach authors a separate "ground scale" on every weapon and every piece of
  equipment — 1.3 on nearly all of them, 2.0 on the assault bomb, the DLC bomb and the skull —
  and applies it the moment the object is placed in the world, which is why a dropped weapon
  reads clearly from across a map. HMS drew them at their raw model size, so every weapon in a
  Forge map was about 30 % too small, in the editor and in Theater alike; a DMR on the ground
  went from 1.09 m to 1.42 m, measured on screen as 230 px wide before and 301 px after at the
  same camera. This is independent of the community "Scaled objects" switch: the two multiply,
  so turning that switch off no longer shrinks a weapon below the size its own tag asks for,
  and a weapon in a player's hands still uses the in-hand size, not this one. Halo 4 and Halo 3
  objects are unchanged for now — their tags put these fields somewhere else and that has not
  been checked.
- **Theater: a held weapon is now the size the weapon tag says it should be in that player's
  hands.** Every weapon was drawn at its ground size whoever was holding it. A Reach weapon
  authors nine sizes — one for lying on the ground, plus an in-hand and an on-the-back size for
  each of four holder size classes — and the game picks the one that matches the holder; an
  earlier note in this file said the report "was measured and is not a scaling fault", which was
  wrong, because what it measured was the attachment markers and the size is not in the markers.
  A Spartan counts as a medium holder, so the energy sword is now drawn at 0.8 (1.27 m → 1.01 m,
  against a 2.08 m body), the needler and the DLC bomb at 0.8, the shotgun, focus rifle, plasma
  repeater, needle rifle, gravity hammer, plasma rifle, spike rifle and golf club at 0.9, the
  concussion rifle and flak cannon at 1.1 and the flag at 0.7, while the DMR, magnum, rocket
  launcher, assault rifle, sniper rifle, grenade launcher and Spartan laser author 1.0 and are
  unchanged; an Elite is a large holder and swings the same sword at 1.1. A weapon with no grip
  marker — the energy sword and the two knife props, three of the map's thirty-two — still hangs
  from its model origin, which is now confirmed to be what the game itself does when the marker
  is missing, and it grows and shrinks about the hand rather than drifting out of it.
- **Forge objects appear much faster, and stop trickling in one at a time.** Opening a 651-object
  Forge World variant from cold went from about 23 s to about 16 s, and the stretch where objects
  are still popping in went from 79 frames to 33 — which is the part you actually watch. Two
  separate faults, both measured: the app was throwing away its decompressed-texture cache in the
  middle of building the objects (and permanently shrinking it), so nearly every model after that
  re-decompressed whole shared texture pages and threw them away again — 2.6 GB of needless
  decompression per load, now 0.08 GB; and the per-frame budget that spreads the work out was set
  to 12 ms when a single model takes about 195 ms, so it could only ever build one model per frame.
  The budget now predicts whether another model will fit in the frame, so cheap models build
  several at a time while an expensive one still gets a frame to itself and the window stays
  responsive.
- **Theater: the gametype card claimed a spawn weapon it had not been told.** When a gametype
  leaves the spawn weapon alone — "unchanged", or deferred to the map's own default — the card
  and `--film-stats --json` still reported it as "stated by the variant for every player", so a
  setting that says nothing read as a decoded fact (3 of 24 films checked by hand). Both now say
  which way the gametype declined, and nothing is drawn in hand, which is what already happened
  on screen.

### Added
- **Theater can now FOLLOW a recording that is still being written, so a match can be watched
  almost as it is played.** MCC writes the game in progress to a `.temp` in its autosave folder,
  and that file turns out to be byte-identical in format to a finished `.mov` -- same `_blf`
  header, same `chdr` chunk -- so nothing new had to be parsed: opening one already worked, and
  measured on two of them the records decode in full (25,033 of 25,033 and 16,226 of 16,226) with
  the complete gametype settings. Every one is an exact multiple of 512 KiB, i.e. written in
  blocks, and this reader already frames the stream forward and stops at the last complete record
  rather than trusting a size field that is only written at finalisation -- which is exactly what
  makes a half-written file safe to read. So "live" is just: notice the file grew, decode it
  again, and keep the view. It re-checks the length at most once every two seconds, never has more
  than one re-decode in flight, and runs it on the same worker thread the first open uses, so the
  window never stalls. **Your camera, follow rig, position in the clip, speed and every toggle
  survive each update**, because the new index is transplanted into the running playback rather
  than playback being rebuilt and the settings copied back -- copying them back is the version
  that rots, since the one setting somebody forgets to copy would silently reset itself every two
  seconds. A playhead parked in the past stays parked; one at the live edge keeps advancing; and a
  playhead past the end of a shorter index is clamped rather than left dangling. Gated on real
  films truncated to a quarter, a half and three quarters of their length -- each must open, must
  never report more ticks than the whole clip, and must never report fewer ticks than a shorter
  prefix -- plus a gate that grows a clip under a parked viewer and asserts the playhead, the
  camera, the speed and the filters all come out unchanged. On by default, remembered between
  sessions, and a no-op on a finished film, which cannot grow.
- **After user testing, six things about the Theater HUD and scoreboard were wrong and are fixed.**
  * **The viewing switches were all behind a section that opened CLOSED, so nobody could find
    them** -- "idk where to adjust things I asked for like first person FOV or to hide certain
    aspects of the HUD". The first-person controls now sit in the panel body: **HUD, Crosshair,
    Name tags and Respawn countdown, with a Field-of-view slider beside them**. That slider used
    to exist only in Settings > Camera & rendering, which is not where anyone watching a film
    looks for it; it is now in both places and both write the same remembered setting. "What to
    draw" also opens by default.
  * **The HUD's colours were wrong in three separate ways.** `color output A` can name any of a
    widget's four custom colours, and every non-skin output was read as custom **A**, so a widget
    that authors its colour in B, C or D drew in whatever was in A. An *unauthored* custom A (all
    four components zero) is not black but a slot the artist never set, and taking it literally
    painted the shield and health meter borders **solid black**; they now take the first colour
    the widget actually authors, which for the shield border is the blue it should be.
  * **The weapon icon was drawing the words "assault_rifle".** A chud text widget's `input string`
    is a string id the engine resolves through the localized string tables, and for a weapon's
    schematic the result is a private-use codepoint whose glyph lives in MCC's
    `font_package_icon*.bin`, not in the Editing Kit's text fonts. HMS has neither, so it was
    drawing the id as letters. It now draws nothing there and names what it could not resolve.
  * **The respawn countdown was in the wrong place.** No chud widget is named for a respawn, so
    the slot is inferred, and the first inference was a MEGALO script slot anchored `top center`.
    The right one is `spawning_on_you` -- the only collection in any shipped `chdt` that the
    engine's own spawn-status states drive -- which hangs from anchor 17 `crosshair`, putting the
    message just below the middle of the screen where the game puts it.
  * **The crosshair and ammo block kept drawing over the third-person shot after a death**, because
    they were gated on the camera MODE the viewer picked rather than the view the camera resolved
    to. Dying now takes the weapon HUD away and leaves the countdown, which is the engine's split.
  * **A player's scoreboard row, and the spectated-player card, now carry that player's colour
    across the WHOLE row** rather than only the two-pixel bars at each end. Only those caps are
    bound to the player colour in the tag; the 247-unit field between them carries a flat authored
    grey, which is what left the row looking uncoloured.
- **The shield and grenade blocks were drawing the game's EFFECT overlays on every frame** -- "the
  shields and grenades stuff looks wrong as well". The shield block is not one bar but a bar plus a
  damage flash (`campaign_shield_flash` coloured `crosshair enemy`, i.e. a solid red blob), a
  border flash and a low-shield warning; the health block adds a red `blob`, the HUD power-on
  `shield_boot_blob` and a 1024-unit compass. All of them are authored with no condition HMS can
  refuse, so all of them were painted at full strength -- a red bar across the shield and two white
  plates over the health. HMS now draws the pieces it can actually drive: the two bars, the shield
  border and the two grenade icons.
  Three tag fields were tried first as a general "this is an effect" rule and **none of them works**,
  which is recorded in the code so nobody repeats it: `scalar output A` (none of these names a flash
  animation), the `Flash State` block (the health BAR itself authors `FLASH when unshielded` and no
  active state, so suppressing flash-only widgets deleted the health bar), and the `animation data`
  block (`meter` and `health_meter` carry the same `active`+`flashing` animations `flahsy` does).
  A widget that authors **no colour at all** -- no skin output and none of its four custom slots set
  -- is also no longer tinted white and drawn; that was the two white plates.
- **Bodies this recording cannot name, but another of your recordings can, are now named** -- matched
  on the player's armour, which is a near-unique fingerprint. They are drawn **dimmer and in
  [brackets]**, because that is an inference and not something the clip states, and the roster now
  says plainly that cross-film naming depends on which other films you have: the same clip names
  more players on a machine with a bigger library.
- **The scoreboard's scores now CLIMB through the clip instead of showing the final score from the
  first frame.** The rows were filled from the summary computed once when the clip is decoded,
  which asks for the last update in the whole film; they are now read at the tick being watched.
  A slot that has sent nothing yet still reads as *no score*, not zero.
- **Theater draws Reach's own FIRST-PERSON HUD: the weapon's reticule, its ammo block, the grenade
  counts, the shield and health meters, and the respawn countdown over a body that is waiting to
  come back.** Three switches under "What to draw" -> "In first person", each remembered between
  sessions. Every piece is placed by the game's own numbers: a `chud_definition` lays its widgets
  out in a 1280x720 virtual screen that the engine maps onto the window's safe frame, so the HUD
  grows and shrinks with the window exactly as the real one does, and the fonts and colours are
  the tag's (`tv_nord_hud-15` for a weapon's schematic glyph, `eurostile_lt_demi-18` for a number,
  the `chud` globals' own palette).
  **Nothing is invented.** A saved film carries no ammo count — the unit's own `rounds loaded` and
  `reserve` fields exist in the format but are never written (zero occurrences over eight films and
  5.4 million unit messages), and a reload event carries four flag bits and no number, so there is
  nothing to count down from either. So the ammo NUMBERS are not drawn: the authored block and its
  schematic are, the numbers are left out, and the panel lists every field with where it came from
  ("recorded", "derived", or "not in the recording"). A meter with no recorded level is drawn
  neither full nor empty. Shield and health are the one thing a film does state — alive or dead —
  so they read full for a living body and empty for a dead one, and the panel says exactly that.
- **The crosshair now appears in first person, and it appears for the weapons that matter.** It was
  decoded but never drawn. Worse, the decode had been rejecting most of the game: a dump over
  `forge_halo` found a reticule for 29 of 64 weapons and none for the other 34 — and the 34 were
  the whole player sandbox (magnum, DMR, assault rifle, shotgun, sniper rifle, needle rifle, plasma
  pistol, rocket launcher, energy sword, gravity hammer), while the 29 were vehicle and turret
  weapons. The tags were never the problem and neither was the offset: `ui\chud\magnum` authors a
  six-part reticule, and every part of it is gated on `97 'weapon is right hand'` OR
  `98 'weapon is left hand'`. HMS was demanding a widget with *no* conditions at all, which threw
  all six away. A widget's `Active State` is an **AND of ORs** and is now decoded and evaluated as
  one, against the states a living, armed, unscoped player in first person actually has. **50 of 64
  weapons now decode a resting reticule**, the magnum's five parts included. The thirteen that
  still do not are the ones that genuinely have no HUD — vehicle horns, passenger and
  anti-infantry seats, the spartan knife and its sheath, the golf club — plus the rocket launcher,
  whose every reticule part is gated on a missile-lock state a film cannot answer; that one is
  reported rather than papered over. With nothing in the subject's hands the player HUD's own
  `unarmed_crosshair` is drawn, which is what the game does.
- **Name tags are now a three-way choice — always, only when you are looking at someone, or
  hidden — instead of an on/off switch.** "Only when looked at" is the engine's own second name
  display, and its rule was taken from the engine rather than invented: `HUD_FindNearestTarget` was
  decompiled and uses a **hard-coded 3.0 degree half-angle** out to a **hard-coded 40 world units
  times the held weapon's zoom magnification**, picks the **nearest** body inside that cone (not
  the most on-axis one), and reads none of the weapon's authored aim-assist fields. There is no
  line-of-sight test in the path that writes the name record, so a name still reads through a wall,
  exactly as in game. The mode is remembered between sessions and `film nametags` now takes
  `always | lookedat | hidden` (and still accepts `on`/`off`, or cycles with no argument).
- **Theater's chud decode now honours the `Invert condition` flag, which several blocks of the HUD
  depend on to appear at all.** `chud_widget_state_flags` has exactly one member — bit 0, "Check if
  the condition is false" — and the Spartan HUD's whole grenade block is authored `NOT 92 'grenades
  unusable'`. Read without the flag that says "draw the grenades only when grenades are unusable",
  and the block never appears. The health bar had a second cause: its collection is gated on
  `50 'binoculars disabled'`, which is simply true of a multiplayer Spartan and had to be said.
  Alongside it, a bitmap widget whose art is a PLAIN bitmap rather than a sprite sheet is no longer
  dropped — the shield meter, the health meter, the ammo meter and the grenade icons are all plain
  bitmaps, and the sprite-sheet-only reader had been silently discarding every one of them (the
  shield block decoded 8 widgets as 1, and the health block 6 as 0).
- **`film nametags always|lookedat|hidden`, `film hud`, `film crosshair` and `film respawnwidget`
  as script verbs**, so the first-person overlays can be set from a script instead of only by
  clicking a checkbox.
- **A `--dump-hud <map> [chdt substring]` diagnostic**, which prints every chud widget *with* its
  Active and Hidden state conditions in the AND/OR shape the tag authors them in and says whether
  each one draws at rest — the thing `--dump-chdt` cannot show, and the thing that explained the
  missing crosshairs. It also prints each weapon's aim-assist block with both readings of every
  angle, and the raw `magazines` block whose element layout the tool build's own tag definition
  leaves empty.
- **Theater's camera now follows the engine into its CONTEXT CAMERAS: the assassination camera,
  the vehicle camera, and the third-person spectate on your own body while the respawn counts
  down.** Watching a player in first person no longer stays glued to their eyes through things the
  game itself cuts away from. The rules are the engine's own, read out of `haloreach_debug.dll`
  and the tags rather than chosen: there is **one camera-mode enum with thirteen members**, named
  in the binary (`following`, `orbiting`, `flying`, `first person`, `dead`, `static`, `scripted`,
  `authored`, `respawning`, `animated trick`, `tracked target`, `sandbox`, `space_fighter`), and
  `director_set_camera_mode` takes a mode plus a transition time. `unit_should_be_third_person`
  is the first-versus-third-person decision and it reports *why*, so Theater can say so too — a
  **vehicle seat** that sets its `third person camera` flag, a **weapon** that sets
  `uses 3rd person camera` (which is how a detached turret changes your view: it is a weapon flag,
  not a special case), or a **whole-body animation mode**, which is what an assassination puts
  both bodies into. The death chain is explicit in `game_director::update`: the moment a player's
  unit goes away and their body is named, a camera in first person or following **cuts** to the
  `dead` camera on that body, and after the authored `time_to_auto_switch_to_orbiting` it becomes
  the `orbiting` camera that spectates it. Every distance and time comes from the globals `camera`
  block and is measured out of `forge_halo.map`, not invented: the dead camera pivots **0.45 wu**
  above the body and eases from **3.2 wu** out to **1.6 wu** over **1.5 s**, switches to orbiting
  after **2.0 s**, and the orbit sits at up to **2.5 wu** with its elevation between **-5.7 and
  44.7 degrees**. (HMS's own invented follow rig had been 2.4 wu at a 0.40 pivot — within a few
  per cent of the engine the whole time, and now not a guess either way.) Three separate reads
  agree on each flag bit, which is why they can be trusted: what the code does with the bit, the
  arithmetic that places the field, and the name the engine's own reflection data gives it —
  `weapon_struct_definition` bit 15 is literally `uses 3rd person camera`, seat flags bit 4 is
  `third person camera`, and bit 7, which makes the engine take the camera straight off a marker
  on the vehicle, is `first person camera slaved to gun.`
- **Fixed before release: on death the view did not change to the third-person spectate at all.**
  The camera solver asked for the player's LIVE body before it asked which camera the engine
  would be in, and a dying body stops replicating almost at once while the dead and orbiting
  cameras run on until the recording collects the corpse — a measured **median 18.2 seconds** of
  spectate camera that the subject lookup could not reach. So the one case that needed the
  context camera was the one case that bailed out first, however correct everything downstream
  was. The camera the viewer is watching is now resolved through the corpse as well as the live
  body, and a death or orbit shot stands on its own when there is no living body left to anchor
  to. The gate added with the fix is an end-to-end one — step across a real death on a real film
  and assert the solved camera actually goes third person and frames the body — because the
  structural gates over the camera timeline were all passing throughout.
- **What is the engine's shot and what is HMS's, said on screen rather than blurred.** The death
  chain's geometry is the engine's own, every number of it. The assassination's and the vehicle's
  are not: the engine frames those from an authored camera a recording cannot drive — a unit's
  `sync action camera` is a `trak` camera track (a list of position-and-quaternion control
  points, and the Spartan authors exactly one), and a vehicle seat that sets
  `first person camera slaved to gun.` takes its camera straight off a marker on the vehicle.
  Both are animated in a space a film never samples: a recording carries an object's root
  position and basis, not its node poses. So Theater orbits instead, and the camera read-out says
  so in words — "the POSE is HMS's, standing in for an authored camera a film cannot drive" —
  rather than letting a reasonable-looking shot pass for the one the game framed. The two
  questions are tracked separately, so "the mode transition is decoded" can never be read as
  "this is the shot you saw".
- **Why only some assassinations get the camera, stated plainly.** The move has to be in the
  recording for Theater to know it happened, and a film's kill feed is **the recorder's own
  medals and nobody else's** — a measured property of the format, not of this decode. Across
  every film on this machine that is 128 assassinations over 40 clips; on the densest
  assassination-gametype clip here, 4:30 of a mode where nearly every kill is a finishing move,
  the recording states **one**. The death camera is unaffected and reaches essentially every
  body, because a death tick is recorded for all of them.
- **The camera's transitions are a pure function of the playback tick**, so scrubbing backwards
  through an assassination or a death lands on the identical shot — the same determinism rule the
  ragdoll solver has to keep. Every override is folded into a per-film table once rather than
  re-derived per frame, for the reason this file has already recorded twice: a film carries tens of
  thousands of events and thousands of tracks, and anything the draw path asks per frame has to be
  a lookup.
- **Leaving a context camera eases back out, the way the engine does** rather than snapping to
  the plain view the instant the assassination or the ride ends — and like every other transition
  here it is a function of the playback tick, so a scrub lands on the identical frame.
- **A toggle, on by default, and a script verb.** `film camera context off` puts the plain camera
  back and stays off between sessions (`context_cameras.txt`, alongside `spectate_card.txt` and
  `first_person_fov.txt`); `film camera mode` reports which engine mode the watched body is in,
  why, and every override segment on its timeline.
- **A weapon that forces third person now moves the camera, which is how a detached turret
  changes your view.** It is not a special case for turrets: `weapon_struct_definition`'s flags
  carry a bit the engine's own reflection data names **`uses 3rd person camera`**, and a portable
  turret is simply a weapon that sets it (`support weapon` is a separate bit on the same word).
  What a body is holding is a decoded fact per tick — a unit's replicated message carries a
  `'weap'` reference, which is already what dresses its hands — so the only outside input is the
  flag, read from the open map. With no map open nothing is claimed, because the flag lives in the
  tag and no film states it. Where a player is in a vehicle SEAT and also holding such a weapon,
  the seat wins, and that is the engine's order rather than a preference:
  `unit_should_be_third_person` resolves the parent-and-seat path first and only falls through to
  the weapon when the unit has no parent.
- **Theater states a vehicle SEAT index where the recording does.** `biped_enter_vehicle` fires
  **0** times in the whole film corpus — boarding rides the vehicle's own seat state, not an event
  — so the only place a film ever names a seat is `biped_exit_vehicle`, which carries a 6-bit seat
  index. An exit names the seat the body was in, so it is matched back onto the ride it ended and
  the seat is reported as decoded; where no exit names it, the camera says so rather than claiming
  a seat it guessed.

- **Theater now shows a vehicle's DAMAGE, and leaves the wreck lying there until the recording
  collected it.** A damaged Warthog shows its damaged permutations, a destroyed one shows the
  blackened husk, and the husk stays exactly as long as it did in the game — the recording states
  the despawn, so nothing is timed out by guesswork (**median 20.4 s between the destroying hit
  and the engine collecting the wreck**, over 324 wrecks; the shortest is a Banshee that blew up
  in the air and was gone in 0.2 s, the longest 245 s). The chain is the engine's own, end to end:
  event type 1 of the recorded event channel is `damage_section_response`, which names the object
  plus the `(damage section, instant response)` pair that fired; that response's
  `region_transitions` block in the object's own model tag carries `(region, new state)` with
  `new state` the same five-value damage enum the object panel's DAMAGE dropdown already draws;
  and the entity's kind-4 destroy is the collection. Across the 23 clips on this machine that
  have vehicles, **5,365 of those events name one of 1,125 vehicles; 417 of them reach a damage
  state and 338 are destroyed**. What each vehicle *can* show is the tag's business and HMS now
  reads it: a Warthog, Mongoose, Ghost, Scorpion, Wraith and Revenant author all four states, a
  Falcon, Pickup, Sabre and Seraph only major and destroyed, a Banshee and Shade only medium and
  destroyed, a forklift or an ONI van only destroyed — and a mounted machine gun, a Pelican and a
  plasma turret author none at all, so they are never drawn damaged. The playback status line
  reports all of it, including when a clip's vehicles have no authored damage states, rather than
  inventing one; `film show` prints each drawn vehicle's tilt, damage state and, for a wreck, how
  long after it blew up the engine collected it.
- **Halo 2 Anniversary map variants can now be saved.** H2A was read-only: HMS would open a
  `.mvar`, resolve its objects against the base map and let you fly around, but the encoder
  refused the file on the way out, so nothing could be written back. The Halo 2 Anniversary
  record layout is now written as well as read, so Save and Save As work on an H2A variant
  exactly as they do for Reach, Halo 3 and Halo 4 -- place, move, turn, delete, retitle, re-box
  -- and the file MCC loads is the file HMS wrote. Every H2A variant on this machine (the 50
  shipped hopper variants, the 7 `classic` ones and the variants MCC's own Forge saved) re-opens
  and writes back **byte for byte identical**, which is the same bar the other three games'
  encoders are held to. H2A **rendering** is still experimental and unchanged.
- **Film statistics now name who killed almost every body.** The kill feed reaches only the
  player who recorded the clip, so it named a killer for **8 of 160** deaths on the test
  recording and all 8 belonged to one person. The damage record reaches nearly everyone --
  `damage_aftermath` names the object damaged and the unit responsible, both already decoded --
  so `--film-stats` now reports a killer for **146 of 160 (91%)**, from **56 distinct killers**,
  and says which channel each came from: the feed is decoded, the damage attribution is derived.
  Where both speak they **agree on 8 of 8**, which is the evidence that the second reference
  really is the killer.
- **Halo 3 maps now report their own authored exposure and bloom settings when they load.** Every
  Halo 3 map carries a `camera_fx_settings` tag -- the same tag Reach and Halo 4 use -- naming the
  brightness the game's auto-exposure aims for, the range of exposures it is allowed to reach, and
  the bloom curve; HMS had never read it for Halo 3 and lit every map to a fixed brightness of its
  own instead. The load log now prints what the map actually asks for (Guardian aims for 0.045
  within 1.25 to 2.5 stops; Construct 0.045 within -0.5 to 1.0; Valhalla 0.035 within 0 to 3), and
  `HMS_EXPODIAG=1` on a headless render reports how far the rendered frame sits from it. Nothing
  about the picture changes yet -- the exposure itself is still HMS's own fixed value -- but the
  gap is now a measured number per map instead of an impression.
- **The Theater player list is now the game's own scoreboard widget, not a table we wrote.**
  Reach's in-game screens are authored as `cui_screen` tags, and the scoreboard is three of them:
  the screen, a player row and a team row. HMS now reads
  `ui\cui\in_game\scoreboard\templates\scoreboard_list_item_player` out of the open map and
  draws the list from it -- the 350x30 row, the rank plate, the team colour bars, the name,
  service-tag and score columns where the tag puts them, the game's own bitmaps behind them
  (`player_base_bitmap`, `player_base_servicetag_bitmap`, `white_pix`), Reach's eleven authored
  team colours out of `multiplayer_globals`, the font each text column asks for
  (`english_body_text_font` = `tv_nord_condensed-15`), and the PLAYERS / SCORE headings out of
  MCC's own string tables. Nothing positional is typed into HMS. The parts that need live game
  state a recording has not got -- the emblem, the talking/dead/connection marks, the selection
  highlight -- are left empty rather than faked, and the panel says so.
- **Theater's floating player names are drawn the way the game draws them, in Reach's own font.**
  They were the app's own typeface on a dark rounded plate, sized by a made-up distance fade --
  nothing like the game. Reach has no nameplate system: a teammate is pushed into the HUD's
  navpoint array each tick and the name is drawn beside the waypoint in
  `full_screen_hud_message_font`, which `maps\fonts\font_table.txt` and `font_settings.txt` name
  as `tv_nord_hud-15`, TV Nord Condensed at 15px. HMS now decodes that font out of the game's own
  asset -- header, kerning pairs and the run-length-coded 4-bit coverage bitmaps -- and paints the
  names with it: **no background plate**, no distance fade, and the engine's own size law
  (constant within about 15 world units, 1/distance out to 25, then held) instead of fading away.
  Each name now hangs off the body's own posed `chud_nav_point` marker -- the marker Reach's
  player models carry for exactly this, on the head bone -- instead of a fixed height above the
  feet that left them floating too high. A name shows only while the player is actually on screen,
  and draws through walls, both as in game; they stay in the player's team colour, which is a
  deliberate difference from the engine (in game you only ever see a teammate's name, so one
  colour is enough; watching a film you are watching both teams). MCC does not ship Reach's text fonts (only the button-icon package), so with no Halo Reach
  Editing Kit installed the names keep the app's font and the playback panel says so.
- **Scripted and headless film playback now draws the film's own Forge map.** It was drawing the
  bare base map, so players appeared to hover in mid-air over whatever the Forge scene had been
  built above -- which looks exactly like a playback bug and is not one. The scene is now drawn on
  the first playback command and the result is reported alongside the actor count, including when
  it could not be drawn and why; `film scene` still exists to force a redraw. Opening a film with
  `film <path>` is unchanged and still needs no map. (This had been quietly misleading our own
  verification too: every screenshot taken through the script runner was missing the scene.)
- **Assassinations play out in Theater, both sides of them.** When the recording says a player
  assassinated another, the two bodies now perform the real two-party animation from the recorded
  moment -- Reach ships eight Spartan-on-Spartan finishers, and both halves of each (the
  attacker's and the victim's) are in the biped's own graph, so what you watch is the game's own
  clip and not a pose HMS invented. The victim no longer disappears: it was dropping out of the
  scene for the length of the move, because a body being assassinated stops reporting a position
  and was therefore neither "alive" nor yet "a corpse". It now stays, finishes its half, and lies
  where it fell. Which of the eight moves played is only known on a film recorded by the match
  host; on everyone else's the move is picked the same way every replay and the caption says so.
- **The moment a player died is now read from the recording rather than inferred.** HMS used to
  take a killed body's last reported position as the instant of death, because nothing better was
  available; the kill feed states the tick outright, so corpses now drop at the right moment.
- **A player who got out of a vehicle stays out.** The ride detector works from two things moving
  together and cannot see the moment someone steps off, so a body could stay stuck in a seat it
  had left. The recording states every exit, with the seat index, and that now cuts the ride.
- **Theater now reads what players actually DID, instead of guessing from how they moved.** A
  Reach saved film turns out to carry the game's own event channel -- 72 named event types, each
  naming up to two objects -- and nothing had ever opened it: 217,330 of them across the films on
  this machine. So the kill feed is now read straight out of the recording ("A melee-killed B at
  3:42", with the medal's own name: `kill`, `melee_kill`, `splatter_kill`, `headshot_kill`,
  `suicide`, `yoink`, `multikill_x2`, ...), vehicle exits are read with the seat the player was
  sitting in rather than worked out from two bodies moving together, and an **assassination is
  named as an assassination** -- on the three clips whose gametype is about assassination, 34 of
  34 kills are the engine's `finishing_move_kill`, against 68 of 1,904 elsewhere, so this is a
  fact off the wire and not the geometric guess that was tried twice and withdrawn. A film
  recorded by the match HOST carries more still: the full two-party assassination record, with
  which of the biped's finishers played, where, and facing which way. `--film-stats` prints all of
  it and `--film-stats --json` carries it (schema 2).
- **Every body in a film now says which character it is, so Elites are Elites.** Each spawn
  message in the recording opens with a reference to the object it was spawned from, and HMS was
  ignoring it and drawing every biped as a Spartan. Reading it properly needed the game's own
  nine-group reference table and the map's per-tag handles -- with those in, the six Forge-World
  clips measured here contain 2,585 Spartan bodies, **454 Elite bodies** and 8 Monitors, and
  vehicles, weapons, grenades, armour abilities, Forge blocks and the health station all name
  themselves too. Open a map with `--map` and the names resolve; without one the reference is
  still reported.
- **Theater: players now move the way the weapon they are carrying makes them move.** Reach has a
  separate animation set per weapon class — a rifle, a pistol, a rocket launcher, a sword, a
  hammer, a ball, a turret — and HMS was driving everyone from the rifle set whatever they held.
  It now reads each weapon's own class off the weapon tag, so a sword carrier moves like a sword
  carrier and a pistol like a pistol, and the panel says per body which set it is on and whether
  that came from the weapon or from the generic default. Stride length is keyed on the loaded clip
  per class too, so a set with a longer walk is no longer driven at the rifle walk's pace.
- **Theater: first person draws the player's own arms.** Looking through a player's eyes now shows
  Reach's dedicated first-person arm model, mounted on the view and posed by the first-person
  animation of the weapon they are carrying — idle, moving, sprinting or falling, chosen from the
  same recorded motion the third-person gait comes from. The arms are drawn only when a weapon
  gives them a pose to play, because the arms model carries no animation of its own; the panel
  says so when there is none, and it never implies a look direction the recording did not store.
- **Theater: a film now shows the gametype's real settings, and players are drawn holding the
  weapon they spawned with.** A saved film carries the whole game variant it was recorded under,
  and HMS now reads it: the loadout palettes (name, primary and secondary weapon, armour ability
  and grenades for all thirty loadouts), the base player traits, the respawn and round settings,
  the social and map settings, the eight team configurations and **Player Species**. The Theater
  card lists them under the gametype's title, and they come out of
  `hms-app --film-stats --json` as a `gametype` block, each figure labelled with where it came
  from as the other statistics already are. The weapon matters most: a recording only ever names
  the weapon in a player's hands when it CHANGES, so before this almost every Spartan was drawn
  empty-handed — on 26 of the 65 films on this machine the gametype states the spawn weapon
  outright and every body now holds it from the moment it appears, with a real pickup still
  taking over the instant the recording reports one. When a gametype's selectable loadouts
  disagree — a film does not record which loadout a player picked — the body is given the
  gametype's most common loadout and the card says so in those words, which covers another 23
  films; only the 16 whose gametype leaves the weapon to the map are still drawn empty-handed. Species is reported the same way:
  the setting is read out of the variant, and when it is the usual "player preference" the card
  says in so many words that a Spartan is a default rather than something the film recorded.
- **Theater: player statistics, and you can pull them straight out of a film without opening the
  app.** A new "Player statistics" section in the Theater side panel gives one row per player —
  their score, lives, deaths, time alive, longest life, distance travelled, top speed on foot,
  time sprinting / airborne / standing still, time in a vehicle and which one, and the weapons the
  recording named —
  and the same numbers come out of the command line as a table or as JSON with
  `hms-app --film-stats <film or folder> [--json] [--map <cache.map>]`. The command line path opens
  no window and needs no graphics card, so it works over SSH, in a script, or while the app is
  already running; point it at a folder and it does every film in it, reporting a file it could not
  read and carrying on. `--map` is optional and only turns vehicle and weapon ids into names.
  Every figure is labelled with where it came from — read out of the recording, worked out from it,
  or not in the recording at all — and the ones that only hold for some players say for how many:
  distance is a lower bound because a film only replicates a position on some ticks, the
  sprint/airborne/still shares are shares of the ticks that could be classified rather than of the
  match, and a player the film header never named still gets a complete row under their entity id.
  `film stats [json]` does the same thing from a script.
- **Theater: players in an in-progress recording now have their names, armour and colours.** A
  recording the game is still writing has no author record — that is only written when a clip is
  saved — and HMS used it to work out which player was which, so every body in a live recording
  came out anonymous, in default armour, in default colours. It now identifies the recording
  player a second way, from the one player whose aim actually moves in the file (everyone else's
  is a flat line, because only your own controls are captured as you play), and works the rest out
  from there. Finished clips are untouched: they still use the author record, and not one name in
  any existing clip changed. Names worked out this way are labelled as worked out rather than
  read, and are only given to players the recording actually has a record for — so a player who
  joined after the recording began, including sometimes the recorder themselves, still shows as
  unnamed rather than being given somebody else's name.
- **Theater: the match SCORE now comes out of the recording.** Reach films carry the game's own
  score store, and HMS now reads it: each player's score, the live round's score, the rounds they
  won and whether the round had finished, listed as a scoreboard and shown on every player's row.
  It is read rather than worked out — the score is stored against the player slot, so it lands on
  the right gamertag with no guessing — and every score HMS shows has been checked three ways
  against the file first, so a clip it cannot read gives no score rather than a wrong one. All 943
  score messages across the 25 recordings on this machine pass all three checks. A player with no
  score was simply never sent one by the recording (which is what happens on a social or Forge
  clip whose rules award no points), and the app says that rather than showing a zero; likewise a
  score carried in from an earlier round only appears on a clip that actually contains one.
- **Theater: kills, assists, medals, betrayals and suicides are NOT available, and the app now
  says so with the reason.** The game keeps them in a table it never sends to anyone, so no
  recording contains them. Before that was confirmed in the game's own code, every candidate
  counter in the recording was swept against four independent checks — that it never goes down,
  that everyone's kills add up to the deaths the recording already proves, that it steps when
  somebody actually died, and that it matches each player's own death count — and nothing passed.
  So no kill feed is shown rather than one being invented. Deaths are still counted, from the
  recording itself.
- **Theater: players now wear the armour and the colours they actually had.** Each body is dressed
  from its own recording — the two colours out of Reach's Armoury palette and the ten armour
  selections (helmet, shoulders, chest, wrist, utility, knees and the rest) — so a clip looks like
  the game did instead of sixteen identical figures. A body the recording could not tie to a named
  player keeps the model's own default armour rather than being given someone else's.
- **Theater: the armour pieces stay on the body.** Shoulder plates, forearm pieces, knee guards,
  belt pouches and ear pieces used to hang detached in mid-air beside each Spartan while the body
  ran normally underneath them, and every piece in the game was drawn at once. Reach packs all of
  them into one mesh and places about a hundred chunks of it around the skeleton; HMS was drawing
  that mesh as a single lump glued to one bone. Each piece now rides its own bone and only the
  pieces that player was wearing are drawn.
- **Theater: armour pieces no longer stretch or warp.** Neighbouring pieces share vertices in the
  tag, so whichever piece was built last dragged the others toward its own bone; each piece now
  gets its own copy and holds its shape. A piece that references a bone its mesh does not use no
  longer pins that corner of the geometry to the model's origin either.

- **Theater: assassinations are still not shown, and the clip now says why.** The recording does
  not store that one happened — the field the game uses for it is absent from every message in
  every clip checked — so the only option was to recognise them from how the two players move.
  That was tried against a gametype built entirely around assassinations and it does not work: it
  finds more candidate pairs on ordinary clips than on that one. Rather than mark kills that may
  not have happened, HMS draws none and the playback caption explains it.
- **Theater now finds the films the Linux install actually writes — including the match you are
  playing right now.** HMS was only ever looking at a mounted Windows partition: the search went
  two folders deep when a Steam/Proton install keeps its files seven deep, and it looked for
  `Users` with a capital U where Proton uses lowercase `users`, which never matches on Linux. It
  now finds every MCC profile on the machine, and opens an in-progress recording (`.temp`)
  alongside finished ones — so you can finish a match and watch it back immediately, with its
  Forge scene, its players and its full timeline, while the game still has the file open. HMS only
  ever reads it.
- **Theater: first person now looks where the player was looking, where the recording allows it.**
  Only the player who saved a clip has a recorded aim — every other player's aim field is filled
  with a single fixed value — so trusting it pointed the view at one unchanging angle forever. The
  view now uses that player's own body facing instead, and says which of the two it is using, or
  tells you plainly when a player's look direction was barely recorded at all.
- **Theater playback is fast again.** Watching a 16-player clip had fallen to about two frames a
  second, and simply having a film open cost most of a frame even with nobody on screen. Three
  things were doing avoidable work every single frame: choosing each player's animation re-walked
  that player's entire recorded path (and did it once per step of the path, so the cost grew with
  the square of the clip length), the filter that decides what to draw re-measured how far every
  entity had ever travelled, and the map's object palette was being re-read from the map file.
  All three are now done once. Measured on the same 16-player clip, paused, same camera:
  **538 ms → 12 ms per frame with 16 players (1.9 → 81 fps)**, and **38 ms → 9.6 ms with none
  (26 → 105 fps)**, against 8.5 ms for the same scene with no film open at all.
- **Theater: a player who picks up a weapon is now drawn holding it.** Where the recording names
  the weapon in someone's hands, that weapon is attached to their model at the right size. Be
  aware this is sparse: the recording only mentions a weapon when it changes, never at spawn, so
  across the clips here it covers between none and 8% of players and the rest are drawn
  empty-handed. The caption says how many are holding a named weapon.
- **Theater: the walk cycles match the ground again.** Every player used one stride rate for every
  animation, tuned for a run — so anyone walking advanced their cycle about a third as fast as
  their feet actually travelled, which is what made them look stiff and glidey. Each gait now
  advances at its own rate, measured from how fast players really move when performing it, and the
  cycle carries smoothly across a change of gait instead of jumping to an unrelated point in the
  stride. A player standing still now breathes rather than freezing, and the gait is chosen from a
  smoothed reading of their motion, so someone hovering near walking pace no longer flickers
  between two animations. Changing animation now blends over about a fifth of a second instead of
  snapping, so a player going from a walk to a run no longer jumps mid-stride, and each animation
  is driven at the length of the clip actually playing rather than an assumed one.
- **Theater: a proper over-the-shoulder follow camera, and you can swing it around.** Following a
  player used to put the camera about 0.4 wu above his head looking down at him. It now sits at
  roughly his own eye height and further back, framed on his chest. Drag in the viewport while
  following to orbit around him, and use the wheel to pull in or push out; the view stays out of
  walls, recenters when you pick a different player, and no longer tips up and down every time the
  player glances at the sky. `film camera orbit <yaw> <pitch> [distance]` and
  `film camera recenter` do the same from a script.
- **Theater: you can now watch anyone in the clip, not just the players the recording named.** The
  cast list is built from everyone who actually had a body on the map, so each of them gets a
  *1st person* and a *follow* button — on one clip that is 14 players where only 3 rows used to
  appear, and only one of those could be followed. Where the recording stored a gamertag you get
  the gamertag; where it did not you get the body's id, which is all a camera needs. Following a
  player keeps following them across every respawn, and says so when they are between lives.
- **Theater: the window no longer covers the thing you are trying to watch.** The controls were a
  floating window taking most of the viewport; they are now a slim strip across the bottom holding
  only what you need while watching — play/pause, step, time, how many are alive, speed and the
  scrub bar — with everything else in a side panel that is closed until you ask for it. Both
  reshape the picture rather than sitting on top of it, so nothing is ever hidden behind them, and
  the film details card no longer opens itself over the viewport when you open a clip. The map and
  Forge editing column is folded away too while a clip is open — none of it is usable while you are
  watching a recording, and it was costing about a third of the window — with an *Editor* toggle on
  the strip to bring it straight back. Together these take the picture from under half the window
  to very nearly all of it.
- **Theater: the Play button no longer moves while a clip is playing.** The window was anchored by
  its bottom edge and its height changed every frame as players spawned and died and the text
  re-wrapped, so the controls slid up and down under the cursor and Pause was hard to hit. The
  strip is now a fixed height and the per-tick detail lives in fixed-size boxes that scroll
  instead of resizing what is around them.
- **Theater: a plain answer about missing gamertags.** Where a clip has fewer names than players,
  the panel now says why: the film header stores whoever was present when it was written, and
  anyone who joined afterwards was never written to the file, so there is no name to recover. A
  player who is named but whose body could not be picked out is labelled "body not matched"
  rather than "no body", because those are different things.
- **Theater: vehicles, and players riding them.** A saved film's vehicles are now drawn and driven —
  Warthogs, Mongooses, Ghosts, Falcons, Scorpions, Wraiths, Banshees, Revenants, turrets and the
  civilian trucks — at their recorded position and facing, and a player who gets into one is now
  seated in it instead of standing in the road. HMS works out which vehicle each one is from where
  it spawned (every vehicle starts on its own Forge spawn point, which the recording carries), works
  out who is riding what from the fact that a rider's body is carried along by the vehicle, and puts
  the rider on the vehicle's own driver or passenger seat with the matching riding animation. A
  vehicle whose model cannot be worked out is reported and left undrawn rather than guessed at, and
  a vehicle nobody ever touched is named as such — it never moved, so the recording never sent a
  position for it, and it is already on screen as part of the film's Forge scene. The vehicle
  figures are now given as two separate counts — what is on screen at this moment, and what the
  whole clip contains — so each adds up on its own instead of looking wrong.
- **Theater: no more doubled-up vehicles.** A film's Forge scene already draws every vehicle on its
  spawn point, so a vehicle nobody had moved was being drawn twice in exactly the same place. The
  recording's copy is now drawn only once the vehicle has actually been moved; until then you are
  looking at the Forge scene's, which is the same vehicle in the same place.
- **Theater: two things falling past each other are no longer called a ride.** Riding was worked out
  by matching how far the player moved against how far the vehicle moved, and gravity moves
  everything downwards at the same rate — so a player and a Warthog falling off the same ledge
  matched perfectly and were reported as a ride. Riding is now judged on movement across the
  ground, and a ride has to actually go somewhere before it is reported. On the clips here that
  takes the count from 87 to 42, and the ones that remain are real drives.
- **Theater: players move the way they actually moved.** Walking, running, backing off, strafing left
  and right, turning on the spot, sprinting and being airborne are now each drawn with their own
  animation, chosen from the recorded position and the recorded body facing together. Previously
  every player played a forward walk picked from their speed alone, so someone backing away ran
  backwards, someone strafing slid sideways in a forward run, and someone falling off a tower jogged
  through the air.
- **Theater: the playback caption is now per player.** Instead of one blanket disclaimer it says, for
  each body on screen, what it is doing and where that came from — decoded from the recording,
  derived from it, or a standing default because that body had no facing at that moment. It also
  says plainly that crouch, the armour ability and firing are not decoded and not drawn. The
  caption is worked out fresh each time it is asked for, so scrubbing and then reading it no longer
  shows the previous moment's answer under the new moment's heading.
- **Theater playback no longer eats memory as it plays.** Each frame of a saved film copied the
  entire decoded recording — every entity's track at every tick, hundreds of thousands of records on
  a long clip — before placing the players. On a 20-minute film that was gigabytes of copying per
  second: the app grew without limit, the window froze and the system eventually killed it. Nothing
  is copied now, and memory stays flat while a film plays.
- **Theater: who is who.** Players' gamertags now float over the right Spartans — on a 16-player clip,
  15 of 16 bodies are named. HMS reads the owning player out of the recording itself rather than
  guessing, and corroborates it against the one player it can identify independently (whoever saved
  the film), so a name is only shown where the recording supports it; anything unmatched keeps its
  entity id instead of a wrong name. A saved film's players are also listed in the Theater window
  with their team colour, service tag and recorded armour, and their gamertags float over the right
  Spartans in the viewport (*Name tags*). Each player in the list has a **1st person** and a
  **follow** button; first person sits at eye height and looks along that player's own recorded aim.
  Where a player cannot be matched, the window says so instead of silently doing nothing. The clock now shows how many players are alive at the current moment, and
  the roster lists when the first one spawns — a round starts with nobody on the map, so an empty
  first few seconds is the recording, not a fault.
- **Prefabs.** Save a selection as a reusable building block and drop it into any map. *Tools ▸
  Prefabs…* saves the current selection, lists what you have with its object count and source game,
  and places a copy at the camera or a typed position, with rotation and mirroring. A placed prefab
  arrives as one undo step and stays selected, so you can grab it straight to where you want it. It
  lands whole or not at all: the canvas's limits are checked first, and a failure part-way is rolled
  back. Each prefab is a single portable file you can hand to someone else, and one saved in one
  game can be placed in another, reporting per object what matched exactly, what is only equivalent,
  and what was dropped.
- **Every control can be rebound.** *Settings ▸ Controls & keyboard* lists all 38 editor actions with
  their keys. Click one, press the key or combination you want, and it takes effect immediately.
  Presets switch the whole layout in one click, including two left-handed sets for people who drive
  the mouse with their left hand. A clash is applied but announced loudly rather than silently
  stealing a key, and *Help ▸ Keyboard shortcuts* now lists your own bindings instead of the
  factory ones. The six fly keys are bound by their position on the keyboard, so an AZERTY or Dvorak
  layout gets a working cluster with no setup, while command shortcuts stay bound to the letter they
  spell. Escape and Tab stay fixed. Defaults are exactly what they have always been.
- **Imported models keep their materials.** Imported geometry used to draw plain white: the loader
  discarded texture coordinates and never read materials at all. glTF and GLB now import the whole
  scene (not just the first mesh, and with each node's placement applied), with base-colour textures
  from PNG and JPEG, vertex colours, and one draw per material. OBJ reads its `.mtl` alongside it.
  Imports are lit like a placed Forge piece, sampling the map's ambient light where they stand, so a
  model moved into shade re-lights. The import browser now tells you how many materials and textures
  a file carries before you open it.
- **Save precision preview.** Saving a map variant rounds every object's position and facing to the
  grid the file format can store, and you can now see exactly where a piece will land. *View ▸
  Overlays* draws a ghost of the saved result next to the live object, with an exaggeration factor
  since the drift is usually too small to see unaided, and the Object panel shows the position and
  rotation error as numbers. *Snap to representable* moves a piece to the nearest pose that survives
  saving unchanged. The `precision` script verb reports the same figures, including the worst
  offenders across a whole variant. Rotation matters more than position: on Reach and Halo 4 a
  position shifts well under a millimetre while a facing can move about a fifth of a degree, and
  Halo 3 is coarser at roughly a centimetre and three quarters of a degree.
- **Rebindable controls.** *Settings ▸ Controls & keyboard…* changes any key in the editor — the
  camera fly keys, the move / rotate / duplicate keys, the axis locks, undo, redo, delete,
  select-all, the nudges, the held modifiers, and which mouse button flies the camera. Click a key,
  press the new one, and it is saved. One click applies a **left-handed** set for people who hold
  the mouse in their left hand (the camera moves to the arrow keys, or to I/J/K/L), and *Restore
  all defaults* puts everything back; a key used twice is reported rather than silently stolen. The
  six fly keys are now matched by the key's **position** rather than the letter printed on it, so a
  non-QWERTY keyboard flies with the same cluster shape out of the box (Z Q S D on AZERTY) with
  nothing to set up. *Help ▸ Keyboard shortcuts* lists your own bindings, not the defaults.
  `Esc` and `Tab` stay fixed so there is always a way out. Also new: holding
  `Alt` while flying drops to a quarter speed for easing up to a close shot. (Issue #2)
- **Theater clips (early).** Reach saved films open in the app: a metadata card, the film's base map
  loaded, and the Forge build it was recorded on drawn in the viewport. The map variant inside a
  clip can be **extracted as a real `.mvar`** you can open and edit like any other. Note that MCC
  saves films with a `.mov` extension. Playback of the recorded action is not in yet.
- **Characters.** Reach Spartans and Elites draw properly skinned, posed by the game's own animation
  data and holding a weapon, available from the script console and headless. This is groundwork for
  Theater playback.
- **Graphics & performance settings.** A new Settings page trades picture quality for speed:
  presets from Flat to Ultra, each knob still adjustable on its own. *Flat / white mesh* is a
  separate unlit view with outlined clay surfaces and no textures, lighting or post-processing, with
  the grid, selection and gizmo still live so you can keep building. Also: render scale, shadow
  quality, bloom / screen effects / colour grading, texture quality and filtering, foliage, decals,
  particles and lens flares, a frame-rate cap, vertical sync, and an idle throttle that stops
  redrawing a still scene. The default (High) is exactly what the app has always rendered.
- **Prefabs.** Build something once and reuse it: select a group of Forge objects, save it as a
  **prefab** (*Tools ▸ Prefabs…*), and drop it into any variant as often as you like — at the
  camera or an exact position, turned to any yaw, and optionally **mirrored**, which is how you get
  the matching half of a symmetric map without rebuilding it. Placing a prefab is one undo step and
  leaves the whole group selected so you can nudge it straight away. A prefab records each piece's
  team, colour, spawn order and time, labels, boundary shape and size and the scaled / shadow /
  damage flags, and stores poses relative to the group's own footprint rather than to the map, so it
  is a shape rather than a place. A prefab that would not fit — object slots, a palette piece's
  maximum count or the canvas budget — places **nothing at all** and says which limit it hit,
  instead of leaving half a structure behind. **A prefab from one game can be placed in another**:
  each piece is routed through the same cross-game equivalence table the variant converter uses, and
  the report says per object whether the match was exact, close or only *approximate* — anything
  with no acceptable match is left out and listed rather than quietly swapped for the wrong piece.
  Each prefab is a single readable file in `prefabs\` inside the app's data folder, so you can send
  one to someone; *Import file…* takes one from anywhere on disk and checks it before it is used.
  Everything is scriptable (`prefab save|list|show|place|rename|delete|import|export`).
  *(GitHub issue #5.)*
- **Convert a map variant to another game.** *File ▸ Convert variant to another game…* re-creates a
  Forge map on another title's canvas, matching each piece to the nearest equivalent by its measured
  size and name. Preview the coverage and a per-object report, nudge the placement, view it on the
  target canvas, then Save As. Reach, Halo 4, Halo 2 Anniversary and Halo 3 can all be read; Halo 2
  Anniversary cannot yet be written to. Typical coverage between Reach, Halo 4 and Halo 2
  Anniversary is 91-99%; Halo 3 is 60-65% because its palette is a different generation (no
  Coliseum Wall, no safe or kill boundaries, roughly half the scale), so its gameplay objects
  transfer but the structure needs rebuilding. Equivalences live in an editable table you can
  correct without rebuilding the app. Coverage counts only solid matches: a piece that is merely
  close in size is listed separately for you to check, and anything beyond that is dropped rather
  than quietly swapped for something wrong.
- **Open recent.** *File ▸ Open recent* lists what you last opened — map variants, base maps,
  saved films and imported models — grouped by kind and labelled with the game, so a Halo 3
  variant is never mistaken for a Reach one; the full path is on hover and one click reopens the
  item (a variant still loads its own base map, Workshop maps included). An entry whose file has
  been moved or deleted is marked, says so when you click it and offers to drop it from the list;
  *Clear recent items* empties the list. With no variant open the left panel offers the recent
  variants too, the list is remembered between runs, and the file browsers' *Recent folders*
  sidebars now start from it instead of being empty on every launch.
- **Import models from anywhere.** Importing external geometry now opens the same file browser used
  for map variants, with an OBJ/GLB/glTF filter, a *Models* shortcut, a *Recent models* list and a
  details line for the selected file. Putting files in the app's `models\` folder is no longer
  required. *(GitHub issue #4, first half.)*
- **Imported models can be moved.** An imported model is now selectable and has a real position,
  rotation and scale, driven by the ordinary gizmo, modal move/rotate, axis locks, numeric fields,
  snapping and undo. Several models can be imported at once, and their poses are saved with the
  project. Snap-to-surface, picking and *Resolve physics* all follow the model when it moves.
  *(GitHub issue #4.)*
- **See the precision a save costs.** A `.mvar` cannot store a pose exactly: every game squeezes
  an object's position onto an integer grid stretched over the variant's world bounds, and its
  rotation onto a quantised up-vector plus a quantised turn angle. The Object panel now has a
  *save precision* section that reports the exact pose the file will hold, the position error per
  axis and in total, and the rotation error split into roll / pitch / yaw, updating live as you
  move or turn the piece — and it warns when a pose sits in the worst part of a cell, where almost
  all of the error is avoidable. Tick *View ▸ Overlays ▸ Save-precision ghost* to see it in the
  viewport: the selected object is drawn a second time, in violet, where it will really land, with
  an exaggeration factor (×1 is the truth) for when the drift is too small to see — the status bar
  always says the ghost is on and shouts when it is exaggerated. *Snap to representable* moves the
  selection onto the nearest pose the file stores exactly, so the error becomes zero rather than
  merely visible (one undo step, never automatic). Scripted as `precision`, `precision <datum>`,
  `precision worst [n]` — the worst offenders across the whole variant, which is the list to fix
  before a release — `precision snap` and `precision ghost`. The grid is per-variant: widening the
  world bounds makes every position coarser. *(GitHub issue #1.)*

- **Theater: players who die now fall over and lie where they fell.** A killed player used to
  stand frozen on the spot for several seconds and then vanish, with nothing to say what happened.
  Their body now plays the Spartan's own death animation and stays on the ground for five seconds
  before being cleared; scrubbing back in the timeline stands them up again, because the fall is
  worked out from the playback position and not from what you have watched. The death itself is
  **worked out, not recorded**: a film never says that a body died, who killed it or how — it ends
  the body's track and deletes it about sixteen seconds later — so HMS treats the last position a
  deleted body reported as the moment of death, and the side panel says so in as many words. (That
  choice was tested rather than assumed: a player's next life starts a fixed delay after they die,
  and of the three candidates the last reported position is the only one that gives a consistent
  delay — it wins on ten of the twelve films on this machine, by margins of four to forty-three
  times, and puts the death three to nine seconds before the respawn, which is what Reach's
  respawn delay actually is.) Up to eight bodies lie on the ground at once, and that budget is
  separate from the sixteen living players so a corpse can never push a live player off screen.
  `film corpses on|off|<count>` from a script; `film show` lists the bodies with the living and
  marks how long each has been down.
- **Theater: a dead player's line says who was standing next to them.** When a body died with
  another player within two world units, the side panel reports the distance and how far round
  from the victim's back they were — and calls it exactly that, geometry. It is **not** labelled an
  assassination and nothing is acted on it: across fifteen recordings, a death that is close and
  from behind is 28% of deaths on an assassination gametype against 16% everywhere else, which is a
  real difference but nowhere near an identification (one ordinary social clip scores higher than
  all three assassination clips). Assassinations themselves remain undrawable — the recording
  simply does not contain them.
- **Theater: the weapon a player SPAWNS with is not in a recording, and that is now settled.** A
  film only reports a unit's weapon when it changes hands. Across twelve recordings and 4,016
  bodies, thirty-three ever report a weapon at all and **not one of those reports arrives at the
  moment the body appears** — the median is sixty-four seconds later, which is a pickup, not a
  loadout. A Reach loadout is a setting of the game variant, like the player species, so that is
  where it lives rather than in the recorded entity stream. Bodies whose weapon the recording never
  names are drawn empty-handed and the panel says so.
- **Theater: Elites still cannot be drawn, and now the reason is measured.** A player's species in
  Reach is decided by the gametype — Invasion gives one team Elites, Infection's zombies are
  Elites — and of the 58 recordings on this machine with a readable header, 54 are Slayer and 4 are
  Capture the Flag, with no Invasion, no Infection and no variant that mentions Elites. There is no
  recording here in which anybody is known to be an Elite, so no candidate field can be tested
  against anything; every body is drawn as a Spartan and the caption says that is a default.

### Fixed
- **Opening a film no longer rebuilds the whole Forge scene.** Opening a recording dressed its
  players in their recorded armour, and doing that threw away every decoded model, texture and
  buffer in the scene — so all 638 Forge objects were decoded and uploaded a second time, right
  after the load had just built them. That is the "loading Forge objects takes way too long,
  especially in Theater". A player's armour can only change what a player's own body wears, so
  only the body is rebuilt now: opening a film went from **42 rebuilds costing 3.0 seconds to one
  costing 69 ms**, measured on the same 638-object map.
- **Floating armour pieces are fixed.** Spartans wore pieces in the wrong place — spikes off the
  chest, a rod through a shoulder, a knee piece hanging between the legs. The model packs about a
  hundred armour attachments into one mesh and tells the game which piece goes on which bone, but
  it states those ranges against the triangles **as the game stores them** (a strip) while HMS had
  already converted them to a different form (a list), so each attachment was handed a different
  one's triangles. The converter now reports where every triangle came from, so the "this piece
  goes here" table lands correctly. Audited across every recorded player on this machine rather
  than by eye: **attachments on the wrong bone went from 74 of 105 to 1 of 105, and players
  wearing a badly displaced piece from 34 of 258 (13.2%) to none at all.** The worst piece on the
  model is now 0.14 world units from the bone it hangs off, against 0.52 before, and the typical
  one 0.05 against 0.14. (The single remaining row names a bone that sits on top of its
  neighbour, which is a limit of the measurement rather than a fault.)
- **Theater says plainly when nobody is holding a weapon.** The side panel used to say nothing at
  all in that case, which reads as a broken feature. It now explains that a recording only reports
  a weapon when it changes hands — across twelve recordings and 4,016 bodies, thirty-three ever
  name one and none of those at the moment they spawn — so the weapon a player starts with is a
  loadout, which lives in the game variant and is not decoded yet.
- **Theater playback is roughly two and a half times faster.** A frame of playback on a busy Forge
  clip cost 72 ms (about 14 fps); it now costs 29 ms. The timeline was rebuilding the whole player
  roster on *every frame* — a list that depends only on the recording and never changes while you
  watch — which alone was 30 ms a frame, and it was also writing the side panel's text even when
  the panel was closed. The roster is now built once when the recording finishes decoding, the
  panel's text only when the panel is open, and the per-frame body list walks the recording once
  instead of twice. `film perf` now reports what the last frame spent on the film, so this cannot
  come back unnoticed. With the side panel open it is now 9 ms a frame rather than 37: the panel
  was also asking the recording "does this player have a body right now?" once per row — a full
  scan of every entity in the clip, sixteen times a frame — and now asks once for all of them.
- **Converting a variant is now reproducible, and the test that checks it is exact again.** The
  converter reads the clock and stamps the result, and that stamp is mirrored into three places in
  the file, so two conversions of the same variant a second apart differed in twenty-two bytes.
  The check that the Convert dialog and the `convert` command produce the same file compared the
  two byte for byte and so failed whenever a second passed between them. A conversion can now be
  given the instant to stamp instead of reading the clock, so the two runs are byte-identical and
  the check is exact with nothing excused. Converting from the app or the command line is
  unchanged — both still stamp the real time.
- **Spartan armour no longer sprouts spikes, rods and a detached knee.** Reach packs about a
  hundred separate armour attachments into a single mesh and tells the game which piece goes on
  which bone; about twenty-nine triangles in that mesh belong to *no* piece at all. HMS was drawing
  them anyway, still stuck to the model's belt while the armour around them had been moved onto a
  helmet, a shoulder or a knee — which is exactly the reported fault: a spray of thin black spikes
  off the chest, a rod through the shoulder, and a knee piece floating between the legs on a
  stretched tether. The game never draws those triangles, and now neither does HMS.
- **Theater's first-person camera no longer looks at the inside of your own helmet.** The player you
  are watching through is no longer drawn, so first person shows what they saw. (Looking along their
  recorded aim was also off by half a turn, which pointed the view at the sky.)
- **Loading a film's map works when the film is for a different map than the one open.** *Load map +
  Forge scene* now loads the film's own map first instead of drawing the film's Forge objects onto
  whatever map happened to be loaded; when the Forge scene cannot be rebuilt because no saved
  variant for that map was found to use as a canvas, it now names the map and the folders it
  searched instead of quietly doing nothing.
- **The Theater window says how many players it could not name.** A clip often has more people in it
  than its header lists — players who joined after the recording started — and bodies HMS cannot tie
  to a player are drawn with an entity id. The window now states all of that plainly rather than
  showing a short list and leaving you to wonder.
- **Spartans in a film no longer wear every armour piece at once.** Bodies in Theater were drawn
  from the raw model, which shows every permutation of every region simultaneously; they now use the
  model's own default variant, so each one is a single clean Spartan.
- **Imported models drew plain white.** An imported `.obj` / `.glb` / `.gltf` now renders with its
  real materials and textures. Every vertex used to be written with no texture coordinates and a
  hard-coded white colour, and no image was ever read, so any model — however well textured —
  arrived as a featureless shell. Now: glTF reads `TEXCOORD_0`, vertex colours, the material list,
  `baseColorTexture` + `baseColorFactor` and `KHR_texture_transform`, decodes the referenced PNG and
  JPEG images (embedded in the GLB, in a `data:` URI, or beside the file), honours each texture's
  wrap modes, and draws each material separately so a multi-material model is not flattened to one
  look; OBJ reads `vt` coordinates and `usemtl`, loads the sibling `.mtl` and uses `map_Kd` plus
  `Kd`. A model with no materials, no texture coordinates or an unreadable image falls back to a
  clean flat colour rather than failing. Imported models are also lit the way a placed Forge piece
  is — the map's own probe light where the model stands, the sun, and the sun's shadow — instead of
  a flat unlit white, and they are re-lit as you move them. Alpha-masked materials are now real
  cut-outs. Large models are read in the background, so importing a 26 MB model no longer freezes
  the window, and the browser's details line says how many materials and textures a file carries
  before you import it.
- **Only part of a glTF model was imported.** A `.glb` holding more than one mesh imported just the
  first one, and every node transform in the file was ignored — so a model built from several
  pieces arrived incomplete, mis-scaled or upside down. The whole scene graph is now read.
- **Removing one of several imported models re-read them all.** Dropping an import re-parsed and
  re-uploaded every remaining model file (seconds, with a large model in the list) and silently
  dropped any whose file had since moved. Removal is now immediate and touches nothing else.
- **Modded maps took far too long to load.** `forge_halo_modded` with a large variant went from
  47 s to 7.7 s, and its worst freeze from 13.6 s to 1.2 s - it now loads faster than stock Forge
  World with a comparable variant. The loader was telling the cache "the load is over" every time a
  batch of models finished, which permanently capped the decompression budget below the size of a
  single texture page, so every later texture re-decompressed a whole page and threw it away
  (~14.7 GB of redundant decompression across 23 models). Two further stalls that ran unbudgeted on
  the main thread, model upload and collision-hull resolution, are now time-sliced.
- **Changing a graphics preset appeared to do nothing.** Altering the render scale reallocated the
  render target but the viewport kept showing a frozen frame from the old one.
- **Reach object types were read from the wrong offset.** The multiplayer block in a Reach object
  tag sits at a different offset than Halo 4's, so every object's gameplay type read as zero:
  weapons that were not weapons, teleporters that did not teleport. This only surfaced in the new
  conversion work, but the field is shared.
- **Halo 3 rendering.** Ground and surfaces that drew as blown-out white now render correctly: only
  one of Halo 3's surface types colours itself from its base texture alone, and the rest multiply in
  one to three detail maps that were being ignored, so blend-masked surfaces like Last Resort's
  courtyard lost all their colour. Flat-coloured surfaces (296 of them on High Ground alone) no
  longer draw as grey placeholder slabs, and fully transparent cut-outs are cut away.
- **Halo 3 object lighting.** Weapons and vehicles are lit by the map's light probes with the
  engine's specular and environment response instead of flat shading, so they read as metal rather
  than near-black silhouettes.
- **Halo 3 fog.** The real two-layer atmosphere law replaces the approximation that was borrowed
  from Reach, removing the heavy grey haze over maps like Guardian and Valhalla.
- **Halo 3 skies.** Sky layers animate at their authored rate. Detail layers had been scrolling a
  full texture per second while the base layer sat still, and authored layer tiling was ignored.
- **Halo 3 damage states.** The damage-state dropdown now works on Halo 3 objects too.
- **Solid barriers drew see-through.** The Heavy Barrier in Reach's Forge palette (and the base of
  the Covenant shield) rendered almost transparent: the renderer decided what was glass from the
  object's internal tag path containing the word "shield", which dragged solid concrete and metal
  parts into the transparent pass. Transparency now follows what the object's material actually
  authors.
- **Switching an object's damage state could untexture other states.** Damage permutations are
  separate draw parts, and the renderer released a model's texture data after uploading only the
  parts it was drawing, leaving the rest permanently white.

### Notes for the release build
Built and packaged 2026-10-12 with `./build-all.sh` + `./build-all.sh package`:

- `dist/release/HaloMapStudio-1.4.0-linux-x64.zip` (13 MB) and
  `HaloMapStudio-1.4.0-windows-x64.zip` (10 MB), each with its `.sha256`; both verify with
  `sha256sum -c`, both hold one top-level folder of the archive's own name, and each carries the
  executable, the native library and the four text files the contract asks for.
- **Both** native libraries were rebuilt in this run and both export `ZH_MBP_SetSharedCacheDirs`
  (`clang-cl` and `lld-link` were on PATH, so the silent-fallback path was not taken).
- The Windows `hms-app.exe` is from this build, not the September one.
- Smoke-tested from the UNPACKED Linux archive, not the build tree: Forge World loads, Sandbox
  loads, and a palette piece places into Halo 3 with no variant open.
- The unpublished 1.3.1 archives were moved to `dist/release/superseded-1.3.1/` so a glob over
  `dist/release/` cannot upload them.
- `dist/github-root/` (README, `THIRD-PARTY-NOTICES.md` and the whole of `docs/`) is re-synced
  with the tree.
- Release notes for the GitHub body: `dist/release/RELEASE-NOTES-1.4.0.md`.

### In progress — not yet landed, add entries when they do
- **Footsteps in test mode have no sound yet**, and the reason is one unread table rather than a
  missing feature. The animation already says exactly which frame of a walk plants which foot --
  that is the engine's own cadence, not a timer -- and a Spartan authors 424 footstep sounds, one
  per surface. What is missing is the join from the ground under your feet to the surface's NAME:
  on a packaged map that table lives in a paged resource nobody has read yet, so the floor you are
  standing on cannot be identified as concrete or metal or grass. Rather than play a wrong surface
  everywhere, or play footsteps on placed blocks and silence on the map itself, nothing plays. The
  Spartan's own default-surface row names no sound at all, so there is not even a fallback to use.
Nothing outstanding for 1.4.0: every item that stood here (Halo 3 terrain / object lighting / fog /
sky / damage states, the graphics quality settings, cross-game variant conversion, Theater, and
issues #1 / #2 / #5) has landed and is listed above.

Known for a later release: Reach BSP mid-tones still read about 1.5x dark against an engine
capture (bloom pedestal is the suspect); Halo 3 exposure and bloom are decoded but not applied;
Theater does not draw projectiles, aim is posed only for the one body per clip whose look
direction the recording states, and the nameplate / scoreboard fonts need HREK to be installed.
