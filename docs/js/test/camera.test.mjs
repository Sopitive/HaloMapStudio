// Copyright (c) 2026 Sopitive. All rights reserved.
// PolyForm Noncommercial 1.0.0, with a separate commercial licence (see docs/license.html).
//
// Pure camera-maths tests for docs/js/hms-camera.js (and the mesh / terrain decoding of
// hms-render.js). `node docs/js/test/camera.test.mjs`. No game files.
// Env HMS_CAMERA_LIB / HMS_RENDER_LIB point at other copies (saboteur runs).

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const req = createRequire(import.meta.url);
const C = req(process.env.HMS_CAMERA_LIB || path.join(here, '..', 'hms-camera.js')).math;
const R = req(process.env.HMS_RENDER_LIB || path.join(here, '..', 'hms-render.js')).math;
let n = 0, failed = 0;
const test = (name, fn) => { n++; try { fn(); console.log('PASS  ' + name); } catch (e) { failed++; console.log('FAIL  ' + name + '\n      ' + String(e.message).split('\n')[0]); } };
const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);
const cam = (o = {}) => ({ tx: 0, ty: 0, tz: 0, dist: 100, yaw: -0.8, pitch: 0.6, fov: 0.9, w: 800, h: 600, ...o });
const sub = (a, b) => a.map((x, i) => x - b[i]);
const len = (v) => Math.hypot(...v);

test('eye / basis are orthonormal and look at the target', () => {
  const c = cam({ tx: 5, ty: -3, tz: 2 }), b = C.basis(c);
  near(len(sub(b.eye, [5, -3, 2])), 100);
  near(len(b.fwd), 1); near(len(b.right), 1); near(len(b.up), 1);
  near(b.fwd[0] * b.right[0] + b.fwd[1] * b.right[1] + b.fwd[2] * b.right[2], 0); near(b.up[2] > 0 ? 1 : 0, 1);
  near(len(sub(b.eye.map((e, i) => e + b.fwd[i] * 100), [5, -3, 2])), 0, 1e-9);
});
test('orbit: yaw follows drag, pitch is clamped (no flips) and the distance is unchanged', () => {
  const c = cam(); C.orbit(c, 100, 0, 0.01); near(c.yaw, -0.8 - 1);
  C.orbit(c, 0, 1e6, 0.01); near(c.pitch, C.PITCH_MAX); C.orbit(c, 0, -1e6, 0.01); near(c.pitch, -C.PITCH_MAX);
  near(c.dist, 100);
});
test('pan moves the target in the screen plane by pixels * world-per-pixel', () => {
  const c = cam({ pitch: 0.4 }), b = C.basis(c), k = 2 * 100 * Math.tan(0.45) / 600;
  C.pan(c, 50, -20);
  const d = [c.tx, c.ty, c.tz];
  near(d[0] * b.right[0] + d[1] * b.right[1] + d[2] * b.right[2], -50 * k, 1e-6);
  near(d[0] * b.up[0] + d[1] * b.up[1] + d[2] * b.up[2], -20 * k, 1e-6);
});
test('zoomToward keeps the world point under the cursor fixed on screen', () => {
  const c = cam({ yaw: 0.3, pitch: 0.5 }), sx = 620, sy = 180;
  const r0 = C.rayFromScreen(c, sx, sy), b0 = C.basis(c), t0 = c.dist / (r0.d[0] * b0.fwd[0] + r0.d[1] * b0.fwd[1] + r0.d[2] * b0.fwd[2]);
  const P = [r0.o[0] + r0.d[0] * t0, r0.o[1] + r0.d[1] * t0, r0.o[2] + r0.d[2] * t0];
  C.zoomToward(c, 2, sx, sy, 1000);
  near(c.dist, 50);
  const r1 = C.rayFromScreen(c, sx, sy), v = sub(P, r1.o), tt = v[0] * r1.d[0] + v[1] * r1.d[1] + v[2] * r1.d[2];
  near(len(sub(P, [r1.o[0] + r1.d[0] * tt, r1.o[1] + r1.d[1] * tt, r1.o[2] + r1.d[2] * tt])), 0, 1e-6);   // P still on the cursor ray
});
test('zoomToward is clamped at both ends', () => {
  const c = cam(); for (let i = 0; i < 80; i++) C.zoomToward(c, 1.5, 400, 300, 500);
  const lim = C.distLimits(500); near(c.dist, lim[0]);
  for (let i = 0; i < 80; i++) C.zoomToward(c, 1 / 1.5, 400, 300, 500); near(c.dist, lim[1]);
});
test('setPivot changes the target without moving the eye', () => {
  const c = cam({ yaw: 1.1, pitch: 0.3 }), e0 = C.eye(c); C.setPivot(c, [30, -10, 8]);
  near(len(sub(C.eye(c), e0)), 0, 1e-9); near(len(sub([c.tx, c.ty, c.tz], [30, -10, 8])), 0);
  near(c.dist, len(sub(e0, [30, -10, 8])), 1e-9);
});
test('lookRotate turns about the eye (the eye does not move)', () => {
  const c = cam(), e0 = C.eye(c); C.lookRotate(c, 120, -40, 0.004);
  near(len(sub(C.eye(c), e0)), 0, 1e-9); near(c.dist, 100);
});
test('fly moves eye and target together along right / forward / up', () => {
  const c = cam({ yaw: 0, pitch: 0 }), b = C.basis(c), e0 = C.eye(c);
  C.fly(c, [0, 10, 0]);
  near(len(sub(C.eye(c), e0.map((e, i) => e + b.fwd[i] * 10))), 0, 1e-9);
  C.fly(c, [0, 0, 5]); near(c.tz, 5);
});
test('frameBox centres the box and fits it in the view', () => {
  const c = cam(), f = C.frameBox(c, [10, 20, 0], [14, 28, 6]);
  near(f.tx, 12); near(f.ty, 24); near(f.tz, 3);
  const r = Math.hypot(4, 8, 6) / 2, test = { ...c, ...f };
  assert.ok(f.dist >= r / Math.sin(0.45) && f.dist < r / Math.sin(0.45) * 1.2);
  const p = [12 + r, 24, 3]; assert.ok(p.length === 3);
  const small = C.frameBox(c, [0, 0, 0], [0.1, 0.1, 0.1]); assert.ok(small.dist > 0 && small.dist < 5);
});
test('presets are defined and keep pitch inside the clamp', () => {
  for (const n of ['front', 'side', 'top', 'iso']) { const p = C.preset(n); assert.ok(p && Math.abs(p.pitch) <= C.PITCH_MAX + 1e-9, n); }
  assert.equal(C.preset('nope'), null);
});
test('lerpCam: endpoints exact, shortest way round in yaw, distance geometric, monotone', () => {
  const a = { tx: 0, ty: 0, tz: 0, dist: 10, yaw: 3, pitch: 0 }, b = { tx: 10, ty: 0, tz: 0, dist: 1000, yaw: -3, pitch: 1 };
  const s = C.lerpCam(a, b, 0), e = C.lerpCam(a, b, 1); near(s.dist, 10); near(e.dist, 1000); near(e.tx, 10);
  const m = C.lerpCam(a, b, 0.5); near(m.dist, 100, 1e-6); assert.ok(Math.abs(m.yaw) > 3, 'goes through pi, not through 0');
  let prev = -1; for (let t = 0; t <= 1.0001; t += 0.1) { const v = C.lerpCam(a, b, t).tx; assert.ok(v >= prev - 1e-9); prev = v; }
});
test('nearFar: near shrinks when zoomed in, far still reaches the whole map', () => {
  const c = cam({ dist: 2 }); const q = C.nearFar(c, 2000);
  assert.ok(q.near <= 0.05 && q.far > 2000 * 3 && q.near > 0);
  const f = C.nearFar(cam({ dist: 5000 }), 2000); assert.ok(f.near <= 4 && f.far > 100000);
});
test('viewProj maps the target to the screen centre and a point behind the eye is outside clip space', () => {
  const c = cam({ tx: 3, ty: 4, tz: 5 }), M = C.viewProj(c, 500), tr = (p) => { const o = [0, 1, 2, 3].map((r) => M[r] * p[0] + M[4 + r] * p[1] + M[8 + r] * p[2] + M[12 + r]); return o; };
  const t = tr([3, 4, 5]); near(t[0] / t[3], 0, 1e-9); near(t[1] / t[3], 0, 1e-9); assert.ok(t[3] > 0 && Math.abs(t[2] / t[3]) < 1);
  const b = C.basis(c), behind = tr(b.eye.map((e, i) => e - b.fwd[i] * 10)); assert.ok(behind[3] < 0);
  // consistent with the ray used for picking: a point 30 along the ray through pixel (600, 100) projects back to that pixel
  const r = C.rayFromScreen(c, 600, 100), p = r.o.map((o, i) => o + r.d[i] * 30), q = tr(p);
  near((q[0] / q[3] * 0.5 + 0.5) * 800, 600, 1e-6); near((1 - (q[1] / q[3] * 0.5 + 0.5)) * 600, 100, 1e-6);
});
test('rayOBB: hits a rotated box at the right distance, misses outside, handles a ray starting inside', () => {
  const fr = R.frame([0, 1, 0], [0, 0, 1]);     // forward +y: local x -> world y
  const box = [-1, -3, 0, 1, 3, 2];              // 2 long along local x (world y), 6 wide along local y (world -x)
  near(C.rayOBB([-10, 0, 1], [1, 0, 0], [0, 0, 0], fr, box), 7);   // hits the world-x face at x = -3
  assert.equal(C.rayOBB([-10, 5, 1], [1, 0, 0], [0, 0, 0], fr, box), null);
  near(C.rayOBB([0, 0, 1], [1, 0, 0], [0, 0, 0], fr, box), 0);
});
test('pickRay: nearest along the ray wins; hidden are skipped; a tiny box is still pickable with minHalf', () => {
  const fr = R.frame([1, 0, 0], [0, 0, 1]);
  const items = [{ pos: [10, 0, 0], fr, box: [-1, -1, -1, 1, 1, 1] }, { pos: [20, 0, 0], fr, box: [-1, -1, -1, 1, 1, 1] }, { pos: [5, 0, 0], fr, box: [-1, -1, -1, 1, 1, 1], hidden: true }];
  const ray = { o: [0, 0, 0], d: [1, 0, 0] };
  assert.equal(C.pickRay(items, ray).i, 0);
  assert.equal(C.pickRay([items[1], items[2]], ray).i, 0);
  const tiny = [{ pos: [10, 0.3, 0], fr, box: [0, 0, 0, 0.01, 0.01, 0.01] }];
  assert.equal(C.pickRay(tiny, ray), null); assert.equal(C.pickRay(tiny, ray, 0.5).i, 0);
});
test('zoomStep never overshoots and converges; decay keeps its sign and reaches zero', () => {
  let p = 1.2, total = 0; for (let i = 0; i < 400; i++) { const z = C.zoomStep(p, 0.016); assert.ok(z.take * p >= 0 && Math.abs(z.take) <= Math.abs(p) + 1e-12); total += z.take; p = z.left; }
  near(total, 1.2, 1e-3); assert.equal(p, 0);
  let v = 800; for (let i = 0; i < 600; i++) { const w = C.decay(v, 0.016); assert.ok(w >= 0 && w <= v); v = w; } assert.equal(v, 0);
  let u = -500; for (let i = 0; i < 600; i++) { const w = C.decay(u, 0.016); assert.ok(w <= 0); u = w; } assert.equal(u, 0);
});

// ---- pack v2 meshes and terrain (hms-render.js)
const b64 = (u8) => Buffer.from(u8).toString('base64');
function packMesh(verts, tris, box) {   // encode the way render_pack.rs does
  const vb = []; for (const v of verts) for (let k = 0; k < 3; k++) { const q = Math.round((v[k] - box[k]) / (box[k + 3] - box[k]) * 65535 - 32768); vb.push(q & 255, (q >> 8) & 255); }
  return { box, n: verts.length, tris: tris.length, ib: 8, v: b64(vb), i: b64(tris.flat()) };
}
test('decodeMesh: a cube round-trips within 16-bit quantisation, normals are unit and outward for CCW triangles', () => {
  const V = [[0, 0, 0], [2, 0, 0], [2, 4, 0], [0, 4, 0], [0, 0, 1], [2, 0, 1], [2, 4, 1], [0, 4, 1]];
  const T = [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]];
  const m = R.decodeMesh(packMesh(V, T, [0, 0, 0, 2, 4, 1]));
  assert.equal(m.tris, 12); assert.equal(m.positions.length, 12 * 9);
  V.forEach((v, i) => v.forEach((c, k) => near(m.verts[i * 3 + k], c, 1e-3)));
  for (let t = 0; t < 12; t++) { const nx = m.normals[t * 9], ny = m.normals[t * 9 + 1], nz = m.normals[t * 9 + 2]; near(Math.hypot(nx, ny, nz), 1, 1e-5); }
  assert.ok(m.normals[2] < -0.99, 'bottom face (0,3,2) points down');
});
test('decodeMesh rejects out-of-range indices, wrong lengths and junk instead of reading past the data', () => {
  const ok = packMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0, 0, 0, 1, 1, 1]);
  R.decodeMesh(ok);
  assert.throws(() => R.decodeMesh({ ...ok, i: b64([0, 1, 7]) }), /index/);
  assert.throws(() => R.decodeMesh({ ...ok, n: 5 }), /vertex/);
  assert.throws(() => R.decodeMesh({ ...ok, i: b64([0, 1]) }), /index/);
  assert.throws(() => R.decodeMesh(null)); assert.throws(() => R.decodeMesh({ box: [0, 0, 0, 1, 1], n: 3 }));
});
test('buildTerrain: a ramp grid makes triangles with up-facing normals, skips empty cells', () => {
  const w = 4, h = 4, min = new Float32Array(16), max = new Float32Array(16);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { min[y * w + x] = x; max[y * w + x] = x; }
  min[5] = NaN;
  const t = R.buildTerrain({ w, h, zmin: 0, zmax: 3, min, max }, [0, 40, 0, 40, 0, 3]);
  assert.equal(t.positions.length % 9, 0); assert.ok(t.positions.length / 9 > 8 && t.positions.length / 9 < 18, String(t.positions.length / 9));
  for (let i = 2; i < t.normals.length; i += 3) assert.ok(t.normals[i] > 0);
});

console.log(`\n${n - failed}/${n} passed`);
process.exit(failed ? 1 : 0);
