// Copyright (c) 2026 Sopitive. All rights reserved.
// PolyForm Noncommercial 1.0.0, with a separate commercial licence (see docs/license.html).
//
// Checks that hms-mvar.classic.js (the plain-<script> build) cannot drift from hms-mvar.js:
//   1. the committed classic file is exactly what build-classic.mjs produces;
//   2. it really is a classic script (no import/export; runs in a bare vm context, as a <script> would);
//   3. on real files, parse() of the classic and module builds give identical results.
// Usage: node docs/js/test/classic.test.mjs [folder ...]   (or env HMS_MVAR_DIRS=";"-separated)
//        --per-game N  files per game to compare (default 8)
// Without folders only 1 and 2 run. Contains no game files.

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildClassic, CLASSIC_PATH, MODULE_PATH } from '../build-classic.mjs';

const results = [];
const check = (name, ok, detail) => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

const dirs = (process.env.HMS_MVAR_DIRS || '').split(';').filter(Boolean);
let perGame = 8;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) { if (argv[i] === '--per-game') perGame = Number(argv[++i]); else dirs.push(argv[i]); }

const modSrc = fs.readFileSync(MODULE_PATH, 'utf8');
const classicSrc = fs.readFileSync(CLASSIC_PATH, 'utf8');
check('committed classic file == build-classic output', buildClassic(modSrc) === classicSrc);
check('classic file has no import/export statements', !/^\s*(import|export)\s/m.test(classicSrc));

// run it the way a <script> does: a fresh global with only what a browser provides
const ctx = vm.createContext({ TextDecoder, console });
vm.runInContext(classicSrc, ctx, { filename: 'hms-mvar.classic.js' });
const Classic = ctx.HmsMvar;
check('classic script defines the global HmsMvar with the same API',
  Classic && ['parse', 'parseFile', 'withPalette', 'filterObjects', 'countBy', 'toCSV', 'toJSON', 'Error', 'version', 'games'].every((k) => k in Classic));
const Mod = (await import(pathToFileURL(MODULE_PATH).href)).default;
check('same version string', Classic.version === Mod.version);

function walk(dir, depth, out) {
  let ents = [];
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { if (depth > 0) walk(p, depth - 1, out); } else if (/\.mvar$/i.test(e.name)) out.push(p);
  }
}
if (dirs.length) {
  const files = []; for (const d of dirs) walk(d, 3, files);
  files.sort();
  const byGame = {};
  for (const p of files) {
    let g; try { g = Mod.parse(fs.readFileSync(p)).game; } catch { continue; }
    (byGame[g] ??= []).push(p);
  }
  for (const [g, list] of Object.entries(byGame)) {
    const step = Math.max(1, Math.floor(list.length / perGame));
    let n = 0, bad = 0;
    for (let i = 0; i < list.length && n < perGame; i += step, n++) {
      const data = fs.readFileSync(list[i]);
      const a = JSON.stringify(Mod.parse(data)), b = JSON.stringify(Classic.parse(data));
      if (a !== b) { bad++; console.log('  differs:', path.basename(list[i])); }
    }
    check(`classic == module on ${g}`, bad === 0 && n > 0, `${n - bad}/${n} files`);
  }
  // an error from the classic build must be its own HmsMvar.Error
  try { Classic.parse(new Uint8Array(3)); check('classic throws its HmsMvar.Error', false); }
  catch (e) { check('classic throws its HmsMvar.Error', e instanceof Classic.Error && e.code === 'not-mvar'); }
} else console.log('(no folders given: real-file comparison skipped)');

const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
