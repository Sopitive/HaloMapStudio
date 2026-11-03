// Copyright (c) 2026 Sopitive. All rights reserved.
// PolyForm Noncommercial 1.0.0, with a separate commercial licence (see docs/license.html).
//
// Equivalence + robustness tests for docs/js/hms-mvar.js. No dependencies: `node` + `node:assert`.
// Contains NO game files and no game data; it reads .mvar files from folders you point it at.
//
// 1. Ground truth: the Rust parsers (the ones Halo Map Studio uses) dump every field of every file:
//      HMS_MVAR_DUMP_DIRS="/folder/a;/folder/b" HMS_MVAR_DUMP_OUT=truth.jsonl \
//        cargo test --profile quick -p hms-app dump_mvar_json_for_js -- --ignored --nocapture
//    (the test lives in crates/hms-app/src/mvar_json_dump.rs; f32 values are dumped as raw bits)
// 2. Compare:
//      node docs/js/test/hms-mvar.test.mjs --truth truth.jsonl [folder ...]
//    Folders (or env HMS_MVAR_DIRS, ";"-separated) restrict which files are checked; with none,
//    every file named in the truth file is checked. Without --truth only the robustness tests run.
//    Other options: --robust-files N (files per game for truncation / bit-flip tests, default 4),
//    --flips N (bit flips per file, default 300). Env HMS_MVAR_LIB points at another copy of the
//    library (used by the saboteur runs).
// Output names files by basename only.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const libPath = process.env.HMS_MVAR_LIB || path.join(here, '..', 'hms-mvar.js');
const HmsMvar = (await import(pathToFileURL(libPath).href)).default;

const argv = process.argv.slice(2);
let truthPath = process.env.HMS_MVAR_TRUTH || null;
let robustFiles = 4, flips = 300;
const dirs = (process.env.HMS_MVAR_DIRS || '').split(';').filter(Boolean);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--truth') truthPath = argv[++i];
  else if (argv[i] === '--robust-files') robustFiles = Number(argv[++i]);
  else if (argv[i] === '--flips') flips = Number(argv[++i]);
  else dirs.push(argv[i]);
}

// ---------------------------------------------------------------------------------------------
// comparison plumbing
// ---------------------------------------------------------------------------------------------

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const bitsOf = (x) => { f32[0] = x; return u32[0]; };
const fromBits = (b) => { u32[0] = b; return f32[0]; };

const stats = {};            // game -> {files, bad}
const diffs = {};            // game -> field -> {count, first}
const loose = {};            // game -> field -> {count, maxRel}  (non-bit-exact but within tolerance)
const TOL = 1e-6;            // relative tolerance for the trig-derived orientation vectors only

function note(game, field, file, got, want) {
  const d = (diffs[game] ??= {});
  const e = (d[field] ??= { count: 0, first: null });
  e.count++;
  if (!e.first) e.first = `${path.basename(file)}: js=${JSON.stringify(got)} rust=${JSON.stringify(want)}`;
}

const canon = (v) => JSON.stringify(v, (k, x) => (x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : 1))) : x));

class Cmp {
  constructor(game, file) { this.game = game; this.file = file; this.ok = true; }
  eq(field, got, want) {
    if (canon(got) !== canon(want)) { this.ok = false; note(this.game, field, this.file, got, want); }
  }
  f(field, got, wantBits, tolerant = false) {        // one f32
    if (bitsOf(got) === wantBits) return;
    if (tolerant) {
      const want = fromBits(wantBits);
      const rel = Math.abs(got - want) / Math.max(1e-30, Math.abs(want), Math.abs(got));
      const abs = Math.abs(got - want);
      if (abs <= TOL || rel <= TOL) {
        const l = ((loose[this.game] ??= {})[field] ??= { count: 0, maxAbs: 0 });
        l.count++; l.maxAbs = Math.max(l.maxAbs, abs);
        return;
      }
    }
    this.ok = false; note(this.game, field, this.file, got, fromBits(wantBits));
  }
  fs(field, got, wantBits, tolerant = false) {        // array of f32
    if (got.length !== wantBits.length) { this.eq(field + '.length', got.length, wantBits.length); return; }
    got.forEach((g, i) => this.f(field, g, wantBits[i], tolerant));
  }
}
const dec = (hex) => BigInt('0x' + hex).toString();
const hexOrNull = (x) => x ?? null;

// ---------------------------------------------------------------------------------------------
// per-game comparison of the JS result against the Rust dump
// ---------------------------------------------------------------------------------------------

function cmpReach(c, js, r) {
  c.eq('title', js.header.title, r.title); c.eq('description', js.header.description, r.description);
  c.eq('author', js.header.author, r.author); c.eq('editor', js.header.editor, r.editor);
  c.eq('labels', js.labels, r.labels);
  const g = r.g;
  c.eq('content_type', js.header.contentType, g.content_type); c.eq('file_size', js.header.fileSize, g.file_size);
  c.eq('uid', dec(js.header.uid), g.uid); c.eq('parent_uid', dec(js.header.parentUid), g.parent_uid);
  c.eq('root_uid', dec(js.header.rootUid), g.root_uid); c.eq('game_id', dec(js.header.gameId), g.game_id);
  c.eq('activity', js.header.activity, g.activity); c.eq('game_mode', js.header.gameMode, g.game_mode);
  c.eq('engine', js.header.engine, g.engine); c.eq('header_map_id', js.map.headerMapId, g.header_map_id);
  c.eq('category', js.header.category, g.category);
  c.eq('created', [String(js.header.created.timestamp), dec(js.header.created.xuid)], g.created);
  c.eq('modified', [String(js.header.modified.timestamp), dec(js.header.modified.xuid)], g.modified);
  c.eq('created_online', js.header.created.online, g.created_online); c.eq('modified_online', js.header.modified.online, g.modified_online);
  c.eq('hopper_id', js.header.hopperId, g.hopper_id); c.eq('version', js.map.version, g.version);
  c.eq('checksum', js.map.checksum, g.checksum); c.eq('palette_crc', parseInt(js.map.paletteCrc, 16), g.palette_crc);
  c.eq('num_quotas', js.extra.numQuotas, g.num_quotas); c.eq('map_id', js.map.mapId, g.map_id);
  c.eq('built_in', js.map.builtIn, g.built_in); c.eq('built_from_xml', js.map.builtFromXml, g.built_from_xml);
  const b = js.bounds; c.fs('bounds', [b.xMin, b.xMax, b.yMin, b.yMax, b.zMin, b.zMax], g.bounds);
  c.eq('budget_max', js.budget.max, g.budget_max); c.eq('budget_spent', js.budget.spent, g.budget_spent);
  c.eq('mcc_map_id', js.map.mccMapId, hexOrNull(g.mcc_map_id));
  c.eq('quotas', js.quotas.map((q) => [q.min, q.max, q.placed]), g.quotas);
  c.eq('end_bit', js.extra.endBit, g.end_bit); c.eq('chunk_length', js.blf.chunkLength, g.chunk_length);
  c.eq('hash_ok', js.blf.hashOk, g.hash_ok); c.eq('has_fsm', js.blf.hasFsm, g.has_fsm); c.eq('hash_kind', js.blf.hashKind, g.hash_kind);
  c.eq('objects.length', js.objects.length, r.objects.length);
  const n = Math.min(js.objects.length, r.objects.length);
  for (let i = 0; i < n; i++) {
    const o = js.objects[i].raw, w = r.objects[i];
    c.eq('obj.folder', o.folder, w.folder); c.eq('obj.item', o.item, w.item);
    c.fs('obj.pos', o.pos, w.pos); c.eq('obj.team', o.team, w.team); c.eq('obj.color', o.color, w.color);
    c.fs('obj.fwd', o.fwd, w.fwd, true); c.fs('obj.up', o.up, w.up, true);
    c.eq('obj.up_is_global', o.upIsGlobal, w.up_is_global); c.eq('obj.up_quant', o.upQuant, w.up_quant);
    c.eq('obj.forward_angle_q', o.forwardAngleQ, w.forward_angle_q); c.eq('obj.spawn_seq', o.spawnSeq, w.spawn_seq);
    c.eq('obj.respawn', o.respawn, w.respawn); c.eq('obj.cached_type', o.cachedType, w.cached_type);
    c.eq('obj.label_idx', o.labelIdx, w.label_idx); c.eq('obj.placement', o.placement, w.placement);
    c.eq('obj.boundary_shape', o.boundaryShape, w.boundary_shape); c.eq('obj.flags', o.flags, w.flags);
    c.eq('obj.in_bounds', o.inBounds, w.in_bounds); c.eq('obj.bsp_index', o.bspIndex, w.bsp_index);
    c.eq('obj.spawn_rel', o.spawnRel, w.spawn_rel); c.eq('obj.slot', o.slot, w.slot); c.eq('obj.boundary', o.boundary, w.boundary);
    c.eq('obj.weapon_clips', o.weaponClips, w.weapon_clips); c.eq('obj.tele_channel', o.teleChannel, w.tele_channel);
    c.eq('obj.tele_passability', o.telePassability, w.tele_passability); c.eq('obj.location_name', o.locationName, w.location_name);
    // the public view must agree with the raw record
    const p = js.objects[i];
    c.eq('pub.x', bitsOf(p.x), w.pos[0]); c.eq('pub.paletteIndex', p.paletteIndex, w.folder);
    c.eq('pub.physics', p.placement.physics, ['normal', 'fixed', 'unknown', 'phased'][(w.placement >> 6) & 3]);
    c.eq('pub.team', p.team, w.team === 255 ? null : w.team);
  }
}

function cmpH3(c, js, r) {
  const h = r.header, j = js.header;
  c.eq('h.unique_id', dec(j.uid), h.unique_id); c.eq('h.name', j.title, h.name); c.eq('h.description', j.description, h.description);
  c.eq('h.author', j.author, h.author); c.eq('h.file_type', j.contentType, h.file_type); c.eq('h.author_is_online', j.created.online, h.author_is_online);
  c.eq('h.author_xuid', dec(j.created.xuid), h.author_xuid); c.eq('h.size_bytes', String(j.fileSize), h.size_bytes);
  c.eq('h.date', String(j.created.timestamp), h.date); c.eq('h.length_seconds', j.lengthSeconds, h.length_seconds);
  c.eq('h.campaign_id', j.campaignId, h.campaign_id); c.eq('h.map_id', js.map.headerMapId, h.map_id);
  c.eq('h.game_engine_type', j.engine, h.game_engine_type); c.eq('h.campaign_difficulty', j.campaignDifficulty, h.campaign_difficulty);
  c.eq('h.hopper_id', j.hopperId, h.hopper_id); c.eq('h.game_id', dec(j.gameId), h.game_id);
  c.eq('chunk_major', js.blf.mvarMajor, r.chunk_major); c.eq('chunk_minor', js.blf.chunks.find((x) => x.type === 'mvar').minor, r.chunk_minor);
  c.eq('version', js.map.version, r.version); c.eq('checksum', js.map.checksum, r.checksum);
  c.eq('num_scenario_objects', js.map.numScenarioObjects, r.num_scenario_objects); c.eq('num_variant_objects', js.map.numVariantObjects, r.num_variant_objects);
  c.eq('num_quotas', js.quotas.length, r.num_quotas); c.eq('map_id', js.map.mapId, r.map_id); c.eq('built_in', js.map.builtIn, r.built_in);
  const b = js.bounds; c.fs('bounds', [b.xMin, b.xMax, b.yMin, b.yMax, b.zMin, b.zMax], r.bounds);
  c.eq('game_engine_subtype', js.map.gameEngineSubtype, r.game_engine_subtype);
  c.f('maximum_budget', js.budget.max, r.maximum_budget); c.f('spent_budget', js.budget.spent, r.spent_budget);
  c.eq('type_start', js.map.typeStartIndex, r.type_start); c.eq('mcc_map_id', js.map.mccMapId, hexOrNull(r.mcc_map_id));
  c.eq('end_bit', js.extra.endBit, r.end_bit);
  c.eq('quotas.length', js.quotas.length, r.quotas.length);
  js.quotas.forEach((q, i) => {
    const w = r.quotas[i]; if (!w) return;
    c.eq('quota.definition', q.definition, w.definition); c.eq('quota.minimum', q.min, w.minimum); c.eq('quota.maximum', q.max, w.maximum);
    c.eq('quota.placed_on_map', q.placed, w.placed_on_map); c.eq('quota.maximum_allowed', q.maxAllowed, w.maximum_allowed); c.f('quota.price', q.price, w.price);
  });
  const placed = r.objects.filter((o) => o.placed), unplaced = r.objects.filter((o) => !o.placed);
  c.eq('objects.placed', js.objects.length, placed.length);
  c.eq('objects.unplaced', js.extra.unplacedSlots.length, unplaced.length);
  unplaced.forEach((w, i) => {
    const u = js.extra.unplacedSlots[i]; if (!u) return;
    c.eq('unplaced.slot', u.slot, w.slot); c.eq('unplaced.flags', u.flags, w.flags); c.eq('unplaced.quota', u.quota, w.quota);
    c.eq('unplaced.parent', u.parent === null ? null : dec(u.parent), w.parent);
  });
  const n = Math.min(js.objects.length, placed.length);
  for (let i = 0; i < n; i++) {
    const o = js.objects[i].raw, w = placed[i];
    c.eq('obj.slot', o.slot, w.slot); c.eq('obj.flags', o.flags, w.flags); c.eq('obj.quota', o.quota, w.quota);
    c.eq('obj.parent', o.parent === null ? null : dec(o.parent), w.parent);
    c.fs('obj.pos', o.pos, w.pos); c.fs('obj.fwd', o.fwd, w.fwd, true); c.fs('obj.up', o.up, w.up, true);
    c.eq('obj.pos_q', o.posQ, w.pos_q); c.eq('obj.up_is_global', o.upIsGlobal, w.up_is_global); c.eq('obj.up_q', o.upQ, w.up_q);
    c.eq('obj.angle_q', o.angleQ, w.angle_q); c.eq('obj.boundary_q', o.boundaryQ, w.boundary_q);
    c.eq('obj.object_type', o.objectType, w.object_type); c.eq('obj.mp_flags', o.mpFlags, w.mp_flags);
    c.eq('obj.game_engine_flags', o.gameEngineFlags, w.game_engine_flags); c.eq('obj.shared_storage', o.sharedStorage, w.shared_storage);
    c.eq('obj.spawn_time', o.spawnTime, w.spawn_time); c.eq('obj.team', o.team, w.team);
    c.eq('obj.boundary_shape', o.boundaryShape, w.boundary_shape); c.fs('obj.boundary', o.boundary, w.boundary);
    const p = js.objects[i];     // the public view of the flag byte against the Rust editor's own constants
    c.eq('pub.uniqueSpawn', p.placement.uniqueSpawn, w.unique); c.eq('pub.placedAtStart', p.placement.placedAtStart, w.placed_at_start);
    c.eq('pub.engineMask', p.placement.engineMask, w.engine_mask); c.eq('pub.team', p.team, w.team > 8 ? 0 : w.team);
    c.eq('pub.mpUnreadBits', p.extra.mpUnreadBits, w.mp_flags & 0xF0);
  }
}

function tdNorm(t) {
  switch (t.kind) {
    case 'byte': case 'index': case 'trait_zone': return { k: t.kind, v: t.value };
    case 'pair': return { k: 'pair', a: t.a, b: t.b };
    case 'initial_ordnance': return { k: t.kind, team: t.team, weapon: t.weapon, timing: t.timing };
    case 'random_ordnance': return { k: t.kind, weapons: t.weapons };
    case 'objective_ordnance': return { k: t.kind, lead: t.lead, weapons: t.weapons };
    default: return { k: 'empty' };
  }
}
function cmpH4(c, js, r) {
  const j = js.header;
  c.eq('chunk_version', js.blf.mvarMajor, r.chunk_version); c.eq('title_key', j.title, r.title_key); c.eq('description_key', j.description, r.description_key);
  c.eq('author', j.author, r.author); c.eq('editor', j.editor, r.editor);
  c.eq('created', [String(j.created.timestamp), dec(j.created.xuid)], r.created); c.eq('modified', [String(j.modified.timestamp), dec(j.modified.xuid)], r.modified);
  c.eq('created_online', j.created.online, r.created_online); c.eq('modified_online', j.modified.online, r.modified_online);
  c.eq('content_type', j.contentType, r.content_type); c.eq('file_size', j.fileSize, r.file_size);
  c.eq('uid', dec(j.uid), r.uid); c.eq('parent_uid', dec(j.parentUid), r.parent_uid); c.eq('root_uid', dec(j.rootUid), r.root_uid); c.eq('game_id', dec(j.gameId), r.game_id);
  c.eq('activity', j.activity, r.activity); c.eq('game_mode', j.gameMode, r.game_mode); c.eq('engine', j.engine, r.engine);
  c.eq('header_map_id', js.map.headerMapId, r.header_map_id); c.eq('category', j.category, r.category); c.eq('hopper_id', j.hopperId, r.hopper_id);
  c.eq('version', js.map.version, r.version); c.eq('map_guid', js.map.mccMapId, hexOrNull(r.map_guid)); c.eq('checksum', js.map.checksum, r.checksum);
  c.eq('palette_crc', parseInt(js.map.paletteCrc, 16), r.palette_crc); c.eq('num_quotas', js.quotas.length, r.num_quotas); c.eq('map_id', js.map.mapId, r.map_id);
  c.eq('built_in', js.map.builtIn, r.built_in); c.eq('built_from_xml', js.map.builtFromXml, r.built_from_xml);
  const b = js.bounds; c.fs('bounds', [b.xMin, b.xMax, b.yMin, b.yMax, b.zMin, b.zMax], r.bounds);
  c.eq('budget_max', js.budget.max, r.budget_max); c.eq('budget_spent', js.budget.spent, r.budget_spent); c.eq('labels', js.labels, r.labels);
  c.eq('quotas', js.quotas.map((q) => [q.min, q.max, q.placed]), r.quotas);
  c.eq('obj_end_bit', js.extra.objEndBit, r.obj_end_bit); c.eq('quota_end_bit', js.extra.quotaEndBit, r.quota_end_bit); c.eq('end_bit', js.extra.endBit, r.end_bit);
  c.eq('chunk_length', js.blf.chunkLength, r.chunk_length); c.eq('has_fsm', js.blf.hasFsm, r.has_fsm); c.eq('hash_kind', js.blf.hashKind, r.hash_kind);
  c.eq('objects.length', js.objects.length, r.objects.length);
  const n = Math.min(js.objects.length, r.objects.length);
  for (let i = 0; i < n; i++) {
    const o = js.objects[i].raw, w = r.objects[i];
    c.eq('obj.slot', o.slot, w.slot); c.eq('obj.bit', o.bit, w.bit); c.eq('obj.flags', o.flags, w.flags); c.eq('obj.quota', o.quota, w.quota);
    c.eq('obj.variant', o.variant, w.variant); c.eq('obj.in_bounds', o.inBounds, w.in_bounds); c.fs('obj.pos', o.pos, w.pos);
    c.eq('obj.pos_q', o.posQ, w.pos_q); c.eq('obj.h2a_bsp_ref', o.h2aBspRef, w.h2a_bsp_ref); c.eq('obj.up_is_global', o.upIsGlobal, w.up_is_global);
    c.eq('obj.up_quant', o.upQuant, w.up_quant); c.eq('obj.forward_angle_q', o.forwardAngleQ, w.forward_angle_q);
    c.fs('obj.fwd', o.fwd, w.fwd, true); c.fs('obj.up', o.up, w.up, true);
    c.eq('obj.parent', o.parent, w.parent); c.eq('obj.scale_q', o.scaleQ, w.scale_q); c.f('obj.scale', o.scale, w.scale); c.eq('obj.locked', o.locked, w.locked);
    c.eq('obj.shape', o.shape, w.shape); c.eq('obj.shape_values', o.shapeValues, w.shape_values);
    c.fs('obj.shape_wu', js.objects[i].shape.values.concat([0, 0, 0, 0]).slice(0, 4).map((v, k) => (k < js.objects[i].shape.values.length ? v : 0)), w.shape_wu);
    c.eq('obj.object_type', o.objectType, w.object_type); c.eq('obj.placement', o.placement, w.placement); c.eq('obj.team', o.team, w.team);
    c.eq('obj.spawn_time', o.spawnTime, w.spawn_time); c.eq('obj.color', o.color, w.color); c.eq('obj.spawn_sequence', o.spawnSequence, w.spawn_sequence);
    c.eq('obj.user_data', o.userData, w.user_data); c.eq('obj.labels', o.labels, w.labels); c.eq('obj.h2a_extra', o.h2aExtra, w.h2a_extra);
    c.eq('obj.type_data', tdNorm(o.typeData), w.type_data);
  }
}

// ---------------------------------------------------------------------------------------------
// file discovery
// ---------------------------------------------------------------------------------------------

function walk(dir, depth, out) {
  let ents = [];
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { if (depth > 0) walk(p, depth - 1, out); }
    else if (/\.mvar$/i.test(e.name)) out.push(p);
  }
}
let allow = null;
if (dirs.length) { const l = []; for (const d of dirs) walk(d, 3, l); allow = new Set(l); }

const results = [];
function report(name, ok, detail) { results.push([name, ok, detail]); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); }

// ---------------------------------------------------------------------------------------------
// 1. equivalence against the Rust dump
// ---------------------------------------------------------------------------------------------

const robustPool = {};          // game -> [paths]
let truthCount = 0, truthSkipped = 0, truthErrors = [];
if (truthPath) {
  const rl = readline.createInterface({ input: fs.createReadStream(truthPath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    const rec = JSON.parse(line);
    if (allow && !allow.has(rec.path)) { truthSkipped++; continue; }
    truthCount++;
    const game = rec.game;
    const s = (stats[game] ??= { files: 0, bad: 0, jsThrew: 0 });
    s.files++;
    if (!rec.ok) {
      // Rust could not decode it: the JS must also refuse it with an HmsMvar.Error
      try { HmsMvar.parse(fs.readFileSync(rec.path)); s.bad++; note(game, 'accepted-but-rust-rejected', rec.path, 'parsed', rec.error); }
      catch (e) { if (!(e instanceof HmsMvar.Error)) { s.bad++; note(game, 'wrong-error-type', rec.path, String(e), rec.error); } }
      continue;
    }
    let js;
    try { js = HmsMvar.parse(fs.readFileSync(rec.path)); }
    catch (e) { s.bad++; s.jsThrew++; note(game, 'js-threw:' + (e.code || e.name), rec.path, e.message, 'ok'); continue; }
    if (js.game !== game) { s.bad++; note(game, 'game', rec.path, js.game, game); continue; }
    const c = new Cmp(game, rec.path);
    if (game === 'reach') cmpReach(c, js, rec.v);
    else if (game === 'halo3') cmpH3(c, js, rec.v);
    else cmpH4(c, js, rec.v);
    if (!c.ok) s.bad++;
    (robustPool[game] ??= []).push(rec.path);
  }
  console.log(`\nEquivalence vs Rust: ${truthCount} files checked (${truthSkipped} in the truth file but outside the folders given)`);
  for (const g of Object.keys(stats)) {
    const s = stats[g];
    const rate = s.files ? (100 * (s.files - s.bad) / s.files).toFixed(2) : 'n/a';
    console.log(`  ${g.padEnd(7)} files ${String(s.files).padStart(4)}  identical ${String(s.files - s.bad).padStart(4)}  differing ${s.bad}  pass rate ${rate}%`);
    for (const [field, e] of Object.entries(diffs[g] ?? {})) console.log(`      DIFF ${field}: ${e.count}x  e.g. ${e.first}`);
    for (const [field, e] of Object.entries(loose[g] ?? {})) console.log(`      within tolerance (${TOL}) not bit-identical: ${field} ${e.count}x, max abs diff ${e.maxAbs.toExponential(2)}`);
    report(`equivalence ${g}`, s.bad === 0 && s.files > 0, `${s.files - s.bad}/${s.files}`);
  }
  if (!Object.keys(stats).length) report('equivalence', false, 'no files matched');
} else {
  console.log('(no --truth file: equivalence tests skipped; robustness tests only)');
}

// ---------------------------------------------------------------------------------------------
// 2. API behaviour that does not need game files
// ---------------------------------------------------------------------------------------------

function expectErr(name, fn, code) {
  try { fn(); report(name, false, 'did not throw'); }
  catch (e) { report(name, e instanceof HmsMvar.Error && (!code || e.code === code), e instanceof HmsMvar.Error ? e.code : 'raw ' + e.name); }
}
expectErr('empty input -> HmsMvar.Error', () => HmsMvar.parse(new Uint8Array(0)), 'not-mvar');
expectErr('text input -> HmsMvar.Error', () => HmsMvar.parse(new TextEncoder().encode('hello, this is not a map variant at all')), 'not-mvar');
expectErr('string argument -> bad-input', () => HmsMvar.parse('nope'), 'bad-input');
expectErr('null argument -> bad-input', () => HmsMvar.parse(null), 'bad-input');
expectErr('header only -> HmsMvar.Error', () => HmsMvar.parse(Uint8Array.from([0x5F, 0x62, 0x6C, 0x66, 0, 0, 0, 12, 0, 1, 0, 0])));
{
  // a well-formed BLF with an mvar chunk of an unknown version
  const b = new Uint8Array(24); b.set([0x5F, 0x62, 0x6C, 0x66, 0, 0, 0, 12, 0, 1, 0, 0], 0);
  b.set([0x6D, 0x76, 0x61, 0x72, 0, 0, 0, 12, 0, 99, 0, 0], 12);
  expectErr('unknown mvar version -> unsupported', () => HmsMvar.parse(b), 'unsupported');
}
{
  // pure-function checks
  const objs = [
    { index: 0, paletteIndex: 3, x: 0, y: 0, z: 0, team: 1, label: 'a', labels: [], placement: { physics: 'fixed' }, objectType: 0, name: 'Wall' },
    { index: 1, paletteIndex: 3, x: 10, y: 0, z: 0, team: null, label: null, labels: [], placement: { physics: 'normal' }, objectType: 1 },
    { index: 2, paletteIndex: 7, x: 50, y: 50, z: 5, team: 1, label: 'Spawn,"x"', labels: [], placement: { physics: 'normal' }, objectType: 1 },
    { index: 3, paletteIndex: 7, x: 1, y: 1, z: 1, team: 2, label: '=cmd', labels: [], placement: { physics: 'phased' }, objectType: 1, forward: [1, 0, 0], up: [0, 0, 1] },
  ];
  const P = { objects: objs };
  report('filter paletteIndex', HmsMvar.filterObjects(P, { paletteIndex: 3 }).length === 2);
  report('filter paletteIndex list', HmsMvar.filterObjects(P, { paletteIndex: [3, 7] }).length === 4);
  report('filter team', HmsMvar.filterObjects(P, { team: 1 }).length === 2);
  report('filter team null', HmsMvar.filterObjects(P, { team: null }).length === 1);
  report('filter physics', HmsMvar.filterObjects(P, { physics: 'phased' }).length === 1);
  report('filter label substring', HmsMvar.filterObjects(P, { label: 'SPAWN' }).length === 1);
  report('filter name', HmsMvar.filterObjects(P, { name: 'wall' }).length === 1);
  report('filter box', HmsMvar.filterObjects(P, { box: { min: [-1, -1, -1], max: [11, 11, 11] } }).length === 3);
  report('filter near', HmsMvar.filterObjects(P, { near: { x: 0, y: 0, z: 0, radius: 2 } }).length === 2);
  report('filter predicate', HmsMvar.filterObjects(P, { predicate: (o) => o.objectType === 0 }).length === 1);
  report('countBy key', JSON.stringify(HmsMvar.countBy(objs, 'paletteIndex')) === '{"3":2,"7":2}');
  report('countBy path', JSON.stringify(HmsMvar.countBy(objs, 'placement.physics')) === '{"fixed":1,"normal":2,"phased":1}');
  const csv = HmsMvar.toCSV(objs.map((o) => ({ forward: [1, 0, 0], up: [0, 0, 1], ...o })));
  report('toCSV quoting + formula guard', csv.includes('"Spawn,""x"""') && csv.includes("'=cmd") && csv.split('\r\n').length === 6);
  const pal = HmsMvar.withPalette({ objects: JSON.parse(JSON.stringify(objs)), quotas: [{ index: 3 }], warnings: [], map: { mapId: 5, unassignedMapId: false }, game: 'reach' }, { game: 'reach', mapId: 5, names: ['a', 'b', 'c', 'Three', '', '', '', 'Seven'] });
  report('withPalette names', pal.objects[0].name === 'Three' && pal.objects[2].name === 'Seven' && pal.quotas[0].name === 'Three');
  const bad = HmsMvar.withPalette({ objects: [{ paletteIndex: 0 }], quotas: [], warnings: [], map: { mapId: 5, unassignedMapId: false }, game: 'reach' }, { game: 'reach', mapId: 6, names: ['x'] });
  report('withPalette refuses another map', bad.objects[0].name === undefined && bad.warnings.length === 1);
}

// ---------------------------------------------------------------------------------------------
// 3. truncation and bit flips on real files: a clean HmsMvar.Error or a normal result, never
//    a raw exception, never a hang
// ---------------------------------------------------------------------------------------------

function rng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
function attempt(bytes) {
  const t0 = performance.now();
  try { HmsMvar.parse(bytes); return { ok: true, ms: performance.now() - t0 }; }
  catch (e) { return { ok: false, clean: e instanceof HmsMvar.Error, err: e, ms: performance.now() - t0 }; }
}
{
  let pool = robustPool;
  if (!truthPath) { // no truth: sample straight from the folders
    pool = {};
    const files = allow ? [...allow] : [];
    for (const p of files) { try { const g = HmsMvar.parse(fs.readFileSync(p)).game; (pool[g] ??= []).push(p); } catch { /* ignore */ } }
  }
  let nTrunc = 0, nFlip = 0, rawErrors = 0, hangs = 0, flipOk = 0, flipErr = 0, truncBad = 0;
  const samples = [];
  for (const [g, files] of Object.entries(pool)) {
    const step = Math.max(1, Math.floor(files.length / robustFiles));
    for (let k = 0; k < files.length && samples.length < 1000; k += step) { if (samples.filter((s) => s.g === g).length >= robustFiles) break; samples.push({ g, p: files[k] }); }
  }
  for (const { g, p } of samples) {
    const data = fs.readFileSync(p);
    const parsed = HmsMvar.parse(data);
    const mv = parsed.blf.chunks.find((c) => c.type === 'mvar');
    const compressed = parsed.blf.compressed;
    // truncation: any cut that removes part of the mvar chunk must be a clean error
    const cuts = new Set([0, 1, 4, 11, 12, 13, 24, 36, 100, 1000, Math.floor(data.length / 2), data.length - 1, data.length - 2]);
    for (let q = 1; q < 40; q++) cuts.add(Math.floor(data.length * q / 40));
    for (const cut of cuts) {
      if (cut < 0 || cut >= data.length) continue;
      const r = attempt(data.subarray(0, cut)); nTrunc++;
      if (r.ms > 2000) hangs++;
      if (!r.ok && !r.clean) { rawErrors++; console.log('  raw exception on truncation', g, path.basename(p), cut, String(r.err)); }
      if (!compressed && cut < mv.offset + mv.size && r.ok) { truncBad++; console.log('  truncated file parsed without error', g, path.basename(p), cut); }
    }
    // a consistent BLF whose mvar chunk simply holds too few bytes (chunk size rewritten): the bit
    // reader itself must refuse, not zero-fill
    if (!compressed) {
      for (const frac of [0.1, 0.5, 0.9, 0.999]) {
        const base = parsed.game === 'halo3' ? 12 : 36;
        const keep = Math.max(base + 1, base + Math.floor((parsed.extra.endBit / 8) * frac) - 1);  // always short of the last field
        const cut = new Uint8Array(mv.offset + keep);
        cut.set(data.subarray(0, mv.offset + keep));
        new DataView(cut.buffer).setUint32(mv.offset + 4, keep);
        const r = attempt(cut); nTrunc++;
        if (r.ok || !r.clean) { truncBad++; console.log('  short mvar chunk not refused cleanly', g, path.basename(p), frac, r.ok ? 'parsed' : String(r.err)); }
      }
    }
    // bit flips
    const rnd = rng(0xC0FFEE ^ data.length);
    for (let i = 0; i < flips; i++) {
      const copy = Uint8Array.from(data);
      const nflip = 1 + Math.floor(rnd() * 3);
      for (let k = 0; k < nflip; k++) copy[Math.floor(rnd() * copy.length)] ^= 1 << Math.floor(rnd() * 8);
      const r = attempt(copy); nFlip++;
      if (r.ms > 2000) hangs++;
      if (r.ok) flipOk++; else { flipErr++; if (!r.clean) { rawErrors++; console.log('  raw exception on bit flip', g, path.basename(p), String(r.err)); } }
    }
    // zero-filled and 0xFF-filled body: stress for loops
    for (const fill of [0x00, 0xFF]) {
      const copy = Uint8Array.from(data); copy.fill(fill, mv.offset + 12, mv.offset + mv.size);
      const r = attempt(copy); nFlip++;
      if (r.ms > 2000) hangs++;
      if (!r.ok && !r.clean) { rawErrors++; console.log('  raw exception on fill', g, String(r.err)); }
    }
  }
  report('truncation: always a clean HmsMvar.Error', samples.length > 0 && rawErrors === 0 && truncBad === 0, `${nTrunc} cuts over ${samples.length} files`);
  report('bit flips: clean Error or a normal result, no hang', samples.length > 0 && rawErrors === 0 && hangs === 0, `${nFlip} mutations; ${flipOk} parsed, ${flipErr} refused cleanly`);
}

const failed = results.filter((r) => !r[1]);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
