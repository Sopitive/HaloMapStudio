// Copyright (c) 2026 Sopitive. All rights reserved.
// PolyForm Noncommercial 1.0.0, with a separate commercial licence (see docs/license.html).
//
// Pure-function tests for docs/js/hms-render.js (projection, box corners, culling, picking, pack
// decoding and the pack/variant match check). `node docs/js/test/render.test.mjs`. No game files.
// Env HMS_RENDER_LIB points at another copy of the library (saboteur runs).

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const R = createRequire(import.meta.url)(process.env.HMS_RENDER_LIB || path.join(here, '..', 'hms-render.js'));
const M = R.math;
let n = 0, failed = 0;
const test = (name, fn) => { n++; try { fn(); console.log('PASS  ' + name); } catch (e) { failed++; console.log('FAIL  ' + name + '\n      ' + String(e.message).split('\n')[0]); } };
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);
const nearV = (a, b, eps = 1e-9) => a.forEach((x, i) => near(x, b[i], eps));

test('boxCorners: axis-aligned box', () => {
  const c = M.boxCorners([10, 20, 30], [1, 0, 0], [0, 0, 1], [-1, -2, 0, 1, 2, 3]);
  assert.equal(c.length, 8);
  nearV(c[0], [9, 18, 30]); nearV(c[7], [11, 22, 33]);
  nearV(c[1], [11, 18, 30]); nearV(c[2], [9, 22, 30]); nearV(c[4], [9, 18, 33]);
});
test('boxCorners: forward +Y turns local x onto world y and local y onto world -x (left = up x forward)', () => {
  const c = M.boxCorners([10, 20, 30], [0, 1, 0], [0, 0, 1], [-1, -2, 0, 1, 2, 3]);
  nearV(c[3], [10 - 2, 20 + 1, 30]);   // local (+1,+2,0)
  nearV(c[0], [10 + 2, 20 - 1, 30]);   // local (-1,-2,0)
});
test('boxCorners: pitched forward keeps the box rigid', () => {
  const f = [Math.cos(0.5), 0, Math.sin(0.5)], u = [-Math.sin(0.5), 0, Math.cos(0.5)];
  const c = M.boxCorners([0, 0, 0], f, u, [0, 0, 0, 2, 1, 1]);
  near(Math.hypot(...c[1].map((v, i) => v - c[0][i])), 2); near(Math.hypot(...c[2].map((v, i) => v - c[0][i])), 1);
  near(c[1][2], 2 * Math.sin(0.5));
});
test('top projection is y-up and round-trips', () => {
  const v = { cx: 100, cy: 50, scale: 4, w: 800, h: 600 };
  nearV(M.topToScreen(v, 100, 50), [400, 300]);
  nearV(M.topToScreen(v, 110, 60), [440, 260]);           // +x right, +y up
  nearV(M.screenToTop(v, ...M.topToScreen(v, 123.4, -9.9)), [123.4, -9.9]);
});
test('orbit projection: target at the centre, behind the eye is dropped, depth grows with distance', () => {
  const c = { tx: 0, ty: 0, tz: 0, dist: 100, yaw: 0.3, pitch: 0.4, fov: 0.9, w: 400, h: 300 }, b = M.camBasis(c);
  const p = M.orbitProject(c, b, [0, 0, 0]);
  nearV([p[0], p[1]], [200, 150], 1e-6); near(p[2], 100, 1e-6);
  assert.equal(M.orbitProject(c, b, [b.eye[0] - b.fwd[0] * 5, b.eye[1] - b.fwd[1] * 5, b.eye[2] - b.fwd[2] * 5]), null);
  const q = M.orbitProject(c, b, [b.right[0] * 10, b.right[1] * 10, b.right[2] * 10]);
  assert.ok(q[0] > 200 && Math.abs(q[1] - 150) < 1e-6, 'a point to the right of the target is right of centre');
});
test('hull2 / inHull', () => {
  const h = M.hull2([[0, 0], [4, 0], [4, 2], [0, 2], [2, 1], [1, 1]]);
  assert.equal(h.length, 4);
  assert.ok(M.inHull(h, 2, 1) && M.inHull(h, 0.1, 0.1) && !M.inHull(h, 5, 1) && !M.inHull(h, 2, 2.1));
});
test('pickTop: highest footprint wins, hidden items are skipped, marker slack catches a small miss', () => {
  const sq = (cx, cy, s) => [[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]];
  const items = [
    { cx: 0, cy: 0, z: 1, r: 3, hull: sq(0, 0, 2) },
    { cx: 0.5, cy: 0, z: 5, r: 3, hull: sq(0.5, 0, 2) },
    { cx: 20, cy: 0, z: 0, r: 0.5, hull: null },
    { cx: 40, cy: 0, z: 9, r: 3, hull: sq(40, 0, 2), hidden: true },
  ];
  assert.equal(M.pickTop(items, 0.2, 0.2, 0.5), 1);
  assert.equal(M.pickTop(items, -1.9, 0, 0.5), 0);
  assert.equal(M.pickTop(items, 20.3, 0.1, 0.5), 2);
  assert.equal(M.pickTop(items, 40, 0, 0.5), -1);
  assert.equal(M.pickTop(items, 100, 100, 0.5), -1);
});
test('cullRect keeps exactly the items whose circle touches the view', () => {
  const items = [{ cx: 0, cy: 0, r: 1 }, { cx: 10, cy: 0, r: 1 }, { cx: 10.9, cy: 0, r: 1 }, { cx: 30, cy: 30, r: 1 }];
  assert.deepEqual(M.cullRect(items, 9.5, 20, -5, 5), [1, 2]);
  assert.deepEqual(M.cullRect(items, -100, 100, -100, 100), [0, 1, 2, 3]);
  assert.deepEqual(M.cullRect(items, 50, 60, 50, 60), []);
});
test('cullRect over 50 000 items is fast', () => {
  const items = Array.from({ length: 50000 }, (_, i) => ({ cx: (i % 250) * 3, cy: Math.floor(i / 250) * 3, r: 1 }));
  const t = performance.now(); const out = M.cullRect(items, 100, 160, 100, 160);
  assert.ok(performance.now() - t < 50 && out.length > 100 && out.length < 800, `${out.length} in ${performance.now() - t} ms`);
});
test('b64decode matches Buffer and rejects junk', () => {
  for (const len of [0, 1, 2, 3, 4, 17, 255]) {
    const raw = Uint8Array.from({ length: len }, (_, i) => (i * 37 + 11) & 255);
    assert.deepEqual([...M.b64decode(Buffer.from(raw).toString('base64'))], [...raw]);
  }
  assert.throws(() => M.b64decode('ab!d'));
});
test('decodeGrid: 255 is empty, others map over [zmin, zmax]', () => {
  const lo = Buffer.from([0, 254, 255, 127]).toString('base64'), hi = Buffer.from([254, 254, 255, 254]).toString('base64');
  const g = M.decodeGrid({ w: 2, h: 2, zmin: -10, zmax: 30, min: lo, max: hi });
  near(g.min[0], -10); near(g.min[1], 30); assert.ok(Number.isNaN(g.min[2])); near(g.min[3], -10 + 127 / 254 * 40, 1e-4); near(g.max[0], 30);
  assert.throws(() => M.decodeGrid({ w: 3, h: 2, zmin: 0, zmax: 1, min: lo, max: hi }), /wrong size/);
});
test('checkPack refuses a pack for another map or game and names both ids', () => {
  const pack = { format: 'hms-render-pack', version: 1, game: 'reach', mapId: 3006 };
  const v = (g, id, un = false) => ({ game: g, map: { mapId: id, unassignedMapId: un } });
  assert.equal(M.checkPack(pack, v('reach', 3006)).ok, true);
  const bad = M.checkPack(pack, v('reach', 3030));
  assert.equal(bad.ok, false); assert.ok(bad.message.includes('3006') && bad.message.includes('3030'));
  const g = M.checkPack(pack, v('halo4', 3006)); assert.equal(g.ok, false); assert.ok(g.message.includes('reach') && g.message.includes('halo4'));
  assert.equal(M.checkPack({ format: 'nope' }, v('reach', 1)).ok, false);
  assert.equal(M.checkPack({ ...pack, version: 9 }, v('reach', 3006)).ok, false);
  const w = M.checkPack(pack, v('reach', 0xFFFFFFFE, true)); assert.equal(w.ok, true); assert.ok(w.message.length > 0);
  assert.equal(M.checkPack({ ...pack, game: 'halo4' }, v('h2a', 3006)).ok, true);
});

console.log(`\n${n - failed}/${n} passed`);
process.exit(failed ? 1 : 0);
