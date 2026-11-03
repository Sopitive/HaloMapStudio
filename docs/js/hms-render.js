/*!
 * hms-render.js - a small Canvas 2D renderer for Halo map variants (objects on a map).
 *
 * Copyright (c) 2026 Sopitive. All rights reserved.
 * Licensed under the PolyForm Noncommercial License 1.0.0, with a separate commercial licence
 * available. Part of Halo Map Studio and covered by the same licence (see docs/license.html).
 *
 * A plain script (works from file:// and from a sub-path web site): exposes the global
 * HmsRender, and module.exports under Node/CommonJS. No libraries, no network, Canvas 2D only.
 *
 *   <script src="js/hms-mvar.classic.js"></script>
 *   <script src="js/hms-render.js"></script>
 *   const view = HmsRender.create(canvas, { describe: (i) => 'object ' + i, onSelect: (i) => ... });
 *   view.setVariant(parsed);            // the object of HmsMvar.parse(); draws grid + markers at once
 *   view.setPack(pack);                 // optional "render pack" (see docs/mvar-js.html); null removes it
 *   view.setVisible([indices] | null);  // the filtered set; everything else is drawn dim
 *   view.setSelection(index | null);  view.setMode('top' | '3d');  view.reset();
 *
 * Without a pack the picture uses only what the variant holds: world bounds, positions,
 * rotations, team / physics. With a pack, objects are drawn as oriented boxes at their real size
 * over a coarse top-down height shading of the level. The pack is derived from a locally
 * installed game and must not be redistributed.
 *
 * Pure helpers (projection, box corners, culling, picking, grid decoding) are in HmsRender.math
 * so they can be tested without a browser.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------------------------
  // pure maths
  // ---------------------------------------------------------------------------------------------

  function norm(v) { var m = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1; return [v[0] / m, v[1] / m, v[2] / m]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  // Local frame of a placed object: +X forward, +Y left (= up x forward), +Z up.
  function frame(forward, up) {
    var f = norm(forward), u = norm(up), l = cross(u, f);
    if (Math.abs(l[0]) + Math.abs(l[1]) + Math.abs(l[2]) < 1e-6) { l = [0, 1, 0]; }
    l = norm(l);
    u = cross(f, l);
    return { f: f, l: l, u: u };
  }

  // The 8 world corners of a local box [minx,miny,minz,maxx,maxy,maxz] placed at (x,y,z).
  // Corner k: bit0 -> max x, bit1 -> max y, bit2 -> max z (local axes).
  function boxCorners(pos, forward, up, box) {
    var fr = frame(forward, up), out = [];
    for (var k = 0; k < 8; k++) {
      var lx = (k & 1) ? box[3] : box[0], ly = (k & 2) ? box[4] : box[1], lz = (k & 4) ? box[5] : box[2];
      out.push([
        pos[0] + fr.f[0] * lx + fr.l[0] * ly + fr.u[0] * lz,
        pos[1] + fr.f[1] * lx + fr.l[1] * ly + fr.u[1] * lz,
        pos[2] + fr.f[2] * lx + fr.l[2] * ly + fr.u[2] * lz]);
    }
    return out;
  }

  // Top view: world (x right, y up on screen) -> CSS pixels, and back. v = {cx, cy, scale, w, h}.
  function topToScreen(v, x, y) { return [v.w / 2 + (x - v.cx) * v.scale, v.h / 2 - (y - v.cy) * v.scale]; }
  function screenToTop(v, sx, sy) { return [v.cx + (sx - v.w / 2) / v.scale, v.cy - (sy - v.h / 2) / v.scale]; }

  // Orbit camera: {tx,ty,tz, dist, yaw, pitch(rad), fov(rad), w, h} -> [sx, sy, depth] or null behind the eye.
  function camBasis(c) {
    var cp = Math.cos(c.pitch), eye = [c.tx + c.dist * cp * Math.cos(c.yaw), c.ty + c.dist * cp * Math.sin(c.yaw), c.tz + c.dist * Math.sin(c.pitch)];
    var fwd = norm([c.tx - eye[0], c.ty - eye[1], c.tz - eye[2]]);
    var right = norm(cross(fwd, [0, 0, 1])), upv = cross(right, fwd);
    return { eye: eye, fwd: fwd, right: right, up: upv };
  }
  function orbitProject(c, b, p) {
    var d = [p[0] - b.eye[0], p[1] - b.eye[1], p[2] - b.eye[2]];
    var z = dot(d, b.fwd);
    if (z < 0.05) return null;
    var f = (c.h / 2) / Math.tan(c.fov / 2);
    return [c.w / 2 + dot(d, b.right) / z * f, c.h / 2 - dot(d, b.up) / z * f, z];
  }

  // Convex hull of 2D points (monotone chain), counter-clockwise.
  function hull2(pts) {
    var p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    if (p.length < 3) return p;
    var cr = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
    var lo = [], i;
    for (i = 0; i < p.length; i++) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p[i]) <= 0) lo.pop(); lo.push(p[i]); }
    var up = [];
    for (i = p.length - 1; i >= 0; i--) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p[i]) <= 0) up.pop(); up.push(p[i]); }
    lo.pop(); up.pop();
    return lo.concat(up);
  }
  function inHull(h, x, y) {
    if (h.length < 3) return false;
    for (var i = 0; i < h.length; i++) {
      var a = h[i], b = h[(i + 1) % h.length];
      if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) return false;
    }
    return true;
  }

  // Which object is under world point (x, y)? items[i] = {index, hull?, cx, cy, z, r}. Prefers the
  // highest object whose footprint contains the point; falls back to the nearest centre within
  // `slack` world units (the marker radius). Returns an index into items or -1.
  function pickTop(items, x, y, slack) {
    var best = -1, bz = -Infinity, bd = Infinity, bi = -1;
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (it.hidden) continue;
      var dx = x - it.cx, dy = y - it.cy;
      if (it.hull && Math.abs(dx) <= it.r && Math.abs(dy) <= it.r && inHull(it.hull, x, y)) { if (it.z >= bz) { bz = it.z; best = i; } }
      var d = Math.sqrt(dx * dx + dy * dy);
      if (d <= slack && d < bd) { bd = d; bi = i; }
    }
    return best >= 0 ? best : bi;
  }

  // Visible-rectangle culling: items whose bounding circle touches [x0,x1]x[y0,y1].
  function cullRect(items, x0, x1, y0, y1) {
    var out = [];
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (it.cx + it.r >= x0 && it.cx - it.r <= x1 && it.cy + it.r >= y0 && it.cy - it.r <= y1) out.push(i);
    }
    return out;
  }

  var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  function b64decode(s) {
    var lut = {}, i;
    for (i = 0; i < 64; i++) lut[B64.charAt(i)] = i;
    var clean = String(s).replace(/=+$/, ''), out = new Uint8Array(Math.floor(clean.length * 3 / 4)), o = 0, acc = 0, bits = 0;
    for (i = 0; i < clean.length; i++) {
      var v = lut[clean.charAt(i)];
      if (v === undefined) throw new Error('bad base64 in the render pack');
      acc = (acc << 6) | v; bits += 6;
      if (bits >= 8) { bits -= 8; out[o++] = (acc >> bits) & 255; }
    }
    return out.subarray(0, o);
  }
  // pack.grid -> {w,h,zmin,zmax, min:Float32Array(z or NaN), max:Float32Array}  (row 0 = lowest y)
  function decodeGrid(g) {
    var lo = b64decode(g.min), hi = b64decode(g.max), n = g.w * g.h;
    if (lo.length !== n || hi.length !== n) throw new Error('the render pack grid has the wrong size');
    var span = g.zmax - g.zmin, mn = new Float32Array(n), mx = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      if (lo[i] === 255) { mn[i] = NaN; mx[i] = NaN; } else { mn[i] = g.zmin + lo[i] / 254 * span; mx[i] = g.zmin + hi[i] / 254 * span; }
    }
    return { w: g.w, h: g.h, zmin: g.zmin, zmax: g.zmax, min: mn, max: mx };
  }

  // Pack v2 mesh -> {box, verts: Float32Array(n*3) unique vertices, positions / normals: Float32Array
  // triangle soup with flat normals}. Throws on malformed data (never reads out of range).
  function decodeMesh(m) {
    if (!m || !m.box || m.box.length !== 6 || !(m.n > 0)) throw new Error('bad mesh');
    var vb = b64decode(m.v), ib = b64decode(m.i), n = m.n, wide = m.ib === 16;
    if (vb.length !== n * 6) throw new Error('bad mesh vertex data');
    var verts = new Float32Array(n * 3), k, i;
    for (i = 0; i < n; i++) for (k = 0; k < 3; k++) {
      var c = ((vb[i * 6 + k * 2] | (vb[i * 6 + k * 2 + 1] << 8)) << 16) >> 16;
      verts[i * 3 + k] = m.box[k] + (c + 32768) / 65535 * (m.box[k + 3] - m.box[k]);
    }
    var cnt = wide ? ib.length >> 1 : ib.length;
    if (cnt % 3 !== 0 || cnt === 0) throw new Error('bad mesh index data');
    var tris = cnt / 3, pos = new Float32Array(tris * 9), nrm = new Float32Array(tris * 9);
    for (var t = 0; t < tris; t++) {
      var ix = [0, 1, 2].map(function (j) { var q = wide ? (ib[(t * 3 + j) * 2] | (ib[(t * 3 + j) * 2 + 1] << 8)) : ib[t * 3 + j]; if (q >= n) throw new Error('bad mesh index'); return q; });
      var P = ix.map(function (q) { return [verts[q * 3], verts[q * 3 + 1], verts[q * 3 + 2]]; });
      var nn = norm(cross([P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]], [P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]]));
      for (var j = 0; j < 3; j++) for (k = 0; k < 3; k++) { pos[t * 9 + j * 3 + k] = P[j][k]; nrm[t * 9 + j * 3 + k] = nn[k]; }
    }
    var parts = [], uvs = null, plist = Array.isArray(m.parts) && m.parts.length ? m.parts : [{ lane: 'opaque', s: 0, n: tris, o: 1 }];
    plist.forEach(function (p) {
      var lane = ['opaque', 'cutout', 'blend', 'additive'].indexOf(p.lane) >= 0 ? p.lane : 'opaque', st0 = p.s | 0, cn = p.n | 0;
      if (st0 < 0 || cn < 0 || st0 + cn > tris) throw new Error('bad mesh part range');
      parts.push({ lane: lane, s: st0, n: cn, c: Array.isArray(p.c) && p.c.length === 3 ? [p.c[0] / 255, p.c[1] / 255, p.c[2] / 255] : null, o: p.o > 0 && p.o <= 1 ? p.o : 1, t: typeof p.t === 'number' ? p.t : -1 });
    });
    if (m.uv && m.uvb && m.uvb.length === 4) {
      var ub = b64decode(m.uv); if (ub.length !== n * 4) throw new Error('bad mesh uv data');
      var uvv = new Float32Array(n * 2);
      for (i = 0; i < n; i++) { uvv[i * 2] = m.uvb[0] + (ub[i * 4] | (ub[i * 4 + 1] << 8)) / 65535 * (m.uvb[2] - m.uvb[0]); uvv[i * 2 + 1] = m.uvb[1] + (ub[i * 4 + 2] | (ub[i * 4 + 3] << 8)) / 65535 * (m.uvb[3] - m.uvb[1]); }
      uvs = new Float32Array(tris * 6);
      for (t = 0; t < tris; t++) for (var j2 = 0; j2 < 3; j2++) { var q2 = wide ? (ib[(t * 3 + j2) * 2] | (ib[(t * 3 + j2) * 2 + 1] << 8)) : ib[t * 3 + j2]; uvs[t * 6 + j2 * 2] = uvv[q2 * 2]; uvs[t * 6 + j2 * 2 + 1] = uvv[q2 * 2 + 1]; }
    }
    return { box: m.box, n: n, tris: tris, verts: verts, positions: pos, normals: nrm, parts: parts, uv: uvs };
  }
  function decodeTextures(list) {
    return (list || []).map(function (t) {
      var px = b64decode(t.rgba); if (!(t.w > 0 && t.h > 0) || px.length !== t.w * t.h * 4 || t.w > 256 || t.h > 256) throw new Error('bad pack texture');
      return { w: t.w, h: t.h, rgba: px };
    });
  }

  // The lit height field of the pack's grid: triangle soup + per-vertex colour (ramp by height).
  function buildTerrain(g, bounds) {
    var cw = (bounds[1] - bounds[0]) / g.w, ch = (bounds[3] - bounds[2]) / g.h, span = Math.max(1e-3, g.zmax - g.zmin);
    var pos = [], nrm = [], col = [], at = function (x, y) { return g.min[y * g.w + x]; };
    var P = function (x, y) { return [bounds[0] + (x + 0.5) * cw, bounds[2] + (y + 0.5) * ch, at(x, y)]; };
    var tri = function (a, b, c) {
      var n = norm(cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]])); if (n[2] < 0) n = [-n[0], -n[1], -n[2]];
      [a, b, c].forEach(function (p) { pos.push(p[0], p[1], p[2]); nrm.push(n[0], n[1], n[2]); var r = ramp((p[2] - g.zmin) / span); col.push(r[0] / 255, r[1] / 255, r[2] / 255); });
    };
    for (var y = 0; y + 1 < g.h; y++) for (var x = 0; x + 1 < g.w; x++) {
      if (isNaN(at(x, y)) || isNaN(at(x + 1, y)) || isNaN(at(x, y + 1)) || isNaN(at(x + 1, y + 1))) continue;
      var a = P(x, y), b = P(x + 1, y), c = P(x + 1, y + 1), d = P(x, y + 1);
      tri(a, b, c); tri(a, c, d);
    }
    return { positions: new Float32Array(pos), normals: new Float32Array(nrm), colors: new Float32Array(col) };
  }

  // A pack must name the same game and map as the variant it is drawn under.
  function checkPack(pack, parsed) {
    if (!pack || pack.format !== 'hms-render-pack') return { ok: false, message: 'This is not a render pack (format "hms-render-pack" expected).' };
    if (pack.version !== 1 && pack.version !== 2) return { ok: false, message: 'Unsupported render pack version ' + pack.version + '.' };
    if (!parsed) return { ok: true, message: '' };
    var g = parsed.game === 'h2a' ? 'halo4' : parsed.game;
    if (pack.game !== g && !(pack.game === 'halo4' && parsed.game === 'h2a')) return { ok: false, message: 'This render pack is for ' + pack.game + ' (map id ' + pack.mapId + '), but the open variant is ' + parsed.game + ' on map id ' + parsed.map.mapId + '.' };
    if (!parsed.map.unassignedMapId && pack.mapId !== parsed.map.mapId) return { ok: false, message: 'This render pack is for map id ' + pack.mapId + ', but the open variant is on map id ' + parsed.map.mapId + '. Make a pack for that map.' };
    if (parsed.map.unassignedMapId) return { ok: true, message: 'The variant is on a custom map without a numeric id, so the pack could not be checked against it.' };
    return { ok: true, message: '' };
  }

  var math = {
    frame: frame, boxCorners: boxCorners, topToScreen: topToScreen, screenToTop: screenToTop, camBasis: camBasis, orbitProject: orbitProject,
    hull2: hull2, inHull: inHull, decodeMesh: decodeMesh, decodeTextures: decodeTextures, buildTerrain: buildTerrain, pickTop: pickTop, cullRect: cullRect, b64decode: b64decode, decodeGrid: decodeGrid, checkPack: checkPack
  };

  // ---------------------------------------------------------------------------------------------
  // colours
  // ---------------------------------------------------------------------------------------------

  var CATEGORY = { block: '#8fa0b3', scenery: '#6fae6f', weapon: '#e5a52f', vehicle: '#d2605f', spawn: '#45b4e0', zone: '#b779dd', equipment: '#dcc443', device: '#4fbfa5', other: '#a0a0a0' };
  var TEAM = ['#e0453f', '#4177e0', '#3fb35d', '#e5a52f', '#a85acd', '#e87f35', '#43bfbf', '#cc579f', '#a6acb4'];
  var PHYS = { normal: '#6fa8dc', fixed: '#e8923f', phased: '#b779dd', unknown: '#999999' };
  function hexRgb(h) { return [parseInt(h.substr(1, 2), 16), parseInt(h.substr(3, 2), 16), parseInt(h.substr(5, 2), 16)]; }
  function colorOf(o, by, cat) {
    if (by === 'team') return o.team === null || o.team === undefined ? '#8a8a8a' : TEAM[Math.min(o.team, 8)] || '#8a8a8a';
    if (by === 'physics') return PHYS[o.placement && o.placement.physics] || '#8a8a8a';
    if (by === 'category') return CATEGORY[cat] || CATEGORY.other;
    return '#8a8a8a';
  }
  function ramp(t) {   // low -> high: deep blue-grey, green, tan, light
    t = Math.max(0, Math.min(1, t));
    var stops = [[0, [78, 104, 142]], [0.35, [92, 148, 122]], [0.7, [186, 170, 112]], [1, [232, 226, 210]]];
    for (var i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
      var a = stops[i - 1], b = stops[i], k = (t - a[0]) / (b[0] - a[0]);
      return [a[1][0] + (b[1][0] - a[1][0]) * k, a[1][1] + (b[1][1] - a[1][1]) * k, a[1][2] + (b[1][2] - a[1][2]) * k];
    }
    return stops[3][1];
  }

  // ---------------------------------------------------------------------------------------------
  // the view
  // ---------------------------------------------------------------------------------------------

  var FACES = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]];
  var FACE_N = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
  var LIGHT = norm([0.4, -0.3, 0.85]);
  var DEFAULT_BOX = [-0.5, -0.5, 0, 0.5, 0.5, 1];
  var HELP = ['3D: left-drag orbit (about the object under the cursor)', 'middle-drag / Shift+left-drag pan', 'wheel zoom toward the cursor', 'right-drag look; then W A S D move, Q E down / up', 'Shift fast, Ctrl slow, Alt+wheel fly speed',
    'double-click an object: frame it    F focus selection    Home whole map', '1 front   2 side   3 top   4 isometric    Esc deselect', 'touch: 1 finger orbit, 2 fingers pan + pinch    ? hide this help'];

  function create(canvas, opts) {
    opts = opts || {};
    var ctx = canvas.getContext('2d');
    var glr = null;
    if (opts.glCanvas && typeof window !== 'undefined' && window.HmsGL) { try { glr = window.HmsGL.create(opts.glCanvas); } catch (e) { glr = null; } }
    var CAM = (typeof window !== 'undefined' && window.HmsCamera) ? window.HmsCamera : (typeof require === 'function' ? null : null);
    var st = {
      mode: 'top', colorBy: 'auto', parsed: null, pack: null, grid: null, items: [], visible: null, selected: null, hover: -1, help: false,
      top: { cx: 0, cy: 0, scale: 1, w: 300, h: 300 },
      cam: { tx: 0, ty: 0, tz: 0, dist: 300, yaw: -0.8, pitch: 0.6, fov: 0.9, w: 300, h: 300 },
      dpr: 1, dirty: false, backdrop: null, bounds: [-100, 100, -100, 100, -10, 50], mouse: null, stats: { drawn: 0, culled: 0, ms: 0 },
      meshes: {}, gi: [], terrainBuilt: false, glFrames: 0, transparency: true, ghost: true, hideCats: new Set()
    };

    function entryOf(o) { return st.pack && st.pack._pal ? st.pack._pal[o.paletteIndex] : null; }
    function variantOf(e, o) {
      if (e && e.variants && o.variantIndex !== null && o.variantIndex !== undefined) for (var i = 0; i < e.variants.length; i++) if (e.variants[i].v === o.variantIndex) return e.variants[i];
      return null;
    }
    function meshOf(o) {       // -> {key, mesh} for the entry or its variant, decoded once
      var e = entryOf(o); if (!e) return null;
      var v = variantOf(e, o), key = v && v.mesh ? e.i + ':' + v.v : e.mesh ? String(e.i) : null, src = v && v.mesh ? v.mesh : e.mesh;
      if (!key) return null;
      if (!(key in st.meshes)) { try { st.meshes[key] = decodeMesh(src); } catch (x) { st.meshes[key] = null; } }
      return st.meshes[key] ? { key: key, mesh: st.meshes[key] } : null;
    }
    function boxFor(o) {
      var e = entryOf(o); if (!e) return null;
      var v = variantOf(e, o);
      return (v && v.box) || e.box || null;
    }
    function catFor(o) { var e = entryOf(o); if (!e) return null; var v = variantOf(e, o); return (v && v.cat) || e.cat; }
    function catHidden(it) { return st.hideCats.size > 0 && it.cat && st.hideCats.has(it.cat); }
    function translucentCat(c) { return c === 'shield' || c === 'glass' || c === 'glow'; }
    function autoBy(it) { return it.cat ? 'category' : (it.obj.team !== null && it.obj.team !== undefined ? 'team' : 'physics'); }
    function itemColor(it) { return colorOf(it.obj, st.colorBy === 'auto' || st.colorBy === 'material' ? autoBy(it) : st.colorBy, it.cat); }

    function build() {
      var objs = st.parsed ? st.parsed.objects : [];
      st.items = []; st.gi = [];
      for (var i = 0; i < objs.length; i++) {
        var o = objs[i], pos = [o.x, o.y, o.z];
        if (!(isFinite(o.x) && isFinite(o.y) && isFinite(o.z))) { st.items.push({ index: o.index, hidden: true, cx: 0, cy: 0, r: 0, z: 0 }); st.gi.push({ key: 'cube', mat: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], col: [1, 1, 1, 1], cx: 0, cy: 0, cz: 0, r: -1e9, dim: false }); continue; }
        var mm = meshOf(o), box = (mm && mm.mesh.box) || boxFor(o), have = !!box, bx = box || DEFAULT_BOX, fr = frame(o.forward, o.up);
        var corners = boxCorners(pos, o.forward, o.up, bx);
        var it = { index: o.index, obj: o, cx: o.x, cy: o.y, z: o.z, box: bx, hasBox: have, mesh: mm, fr: fr, pos: pos, corners: have ? corners : null, cat: catFor(o), r: 0.6, hull: null, hidden: false };
        var pts;
        if (mm) {      // silhouette from the mesh's own vertices
          pts = []; var vv = mm.mesh.verts;
          for (var q = 0; q < vv.length; q += 3) pts.push([pos[0] + fr.f[0] * vv[q] + fr.l[0] * vv[q + 1] + fr.u[0] * vv[q + 2], pos[1] + fr.f[1] * vv[q] + fr.l[1] * vv[q + 1] + fr.u[1] * vv[q + 2]]);
        } else if (have) pts = corners.map(function (c) { return [c[0], c[1]]; });
        if (pts) {
          it.hull = hull2(pts);
          var ccx = 0, ccy = 0, zs = 0, k, r = 0;
          for (k = 0; k < 8; k++) { ccx += corners[k][0] / 8; ccy += corners[k][1] / 8; zs += corners[k][2] / 8; }
          for (k = 0; k < it.hull.length; k++) r = Math.max(r, Math.hypot(it.hull[k][0] - ccx, it.hull[k][1] - ccy));
          it.cx = ccx; it.cy = ccy; it.z = zs; it.r = Math.max(r, 0.3);
        }
        // GL instance: mesh vertices are in the object's local frame; the cube is scaled onto the box
        var f = fr.f, l = fr.l, u = fr.u, mat, key;
        if (mm) { key = mm.key; mat = [f[0], l[0], u[0], pos[0], f[1], l[1], u[1], pos[1], f[2], l[2], u[2], pos[2]]; }
        else {
          key = 'cube'; var sx = bx[3] - bx[0], sy = bx[4] - bx[1], sz = bx[5] - bx[2];
          var tx = pos[0] + f[0] * bx[0] + l[0] * bx[1] + u[0] * bx[2], ty = pos[1] + f[1] * bx[0] + l[1] * bx[1] + u[1] * bx[2], tz = pos[2] + f[2] * bx[0] + l[2] * bx[1] + u[2] * bx[2];
          mat = [f[0] * sx, l[0] * sy, u[0] * sz, tx, f[1] * sx, l[1] * sy, u[1] * sz, ty, f[2] * sx, l[2] * sy, u[2] * sz, tz];
        }
        var cc = { key: key, mat: mat, col: [1, 1, 1, 1], cx: 0, cy: 0, cz: 0, r: 0, dim: false };
        var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        corners.forEach(function (c) { for (var a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], c[a]); mx[a] = Math.max(mx[a], c[a]); } });
        cc.cx = (mn[0] + mx[0]) / 2; cc.cy = (mn[1] + mx[1]) / 2; cc.cz = (mn[2] + mx[2]) / 2; cc.r = cc.r0 = Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) / 2 + 0.05;
        st.items.push(it); st.gi.push(cc);
      }
      refreshGl();
    }
    var DIM_COL = [0.55, 0.55, 0.55, 0.13];
    function refreshGl() {     // colours and the dim flag follow colour-by and the filter
      for (var i = 0; i < st.items.length; i++) {
        var it = st.items[i], g = st.gi[i]; if (it.hidden) continue;
        if (catHidden(it)) { g.r = -1e9; continue; }
        g.r = g.r0; var on = isVisible(i); g.dim = !on;
        if (on) { var h = hexRgb(itemColor(it)); g.col = [h[0] / 255, h[1] / 255, h[2] / 255, 1]; } else { var h2 = hexRgb(itemColor(it)), gr = (h2[0] + h2[1] + h2[2]) / 765; g.col = [gr * 0.6 + h2[0] / 255 * 0.4 * 0.5, gr * 0.6 + h2[1] / 255 * 0.4 * 0.5, gr * 0.6 + h2[2] / 255 * 0.4 * 0.5, 1]; }
      }
    }
    function uploadGl() {
      if (!glr) return;
      var map = {}; Object.keys(st.meshes).forEach(function (k) { if (st.meshes[k]) map[k] = st.meshes[k]; });
      glr.setMeshes(map);
      try { glr.setTextures(st.textures || []); } catch (e) { glr.setTextures([]); }
      glr.setTerrain(st.grid ? buildTerrain(st.grid, st.bounds) : null);
    }
    function fitBounds() {
      var b = st.parsed ? st.parsed.bounds : null;
      var bb = b && isFinite(b.xMin) && isFinite(b.xMax) && b.xMax > b.xMin ? [b.xMin, b.xMax, b.yMin, b.yMax, b.zMin, b.zMax] : null;
      if (!bb) {
        var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        st.items.forEach(function (it) { if (it.hidden) return; mn[0] = Math.min(mn[0], it.cx); mn[1] = Math.min(mn[1], it.cy); mn[2] = Math.min(mn[2], it.z); mx[0] = Math.max(mx[0], it.cx); mx[1] = Math.max(mx[1], it.cy); mx[2] = Math.max(mx[2], it.z); });
        bb = isFinite(mn[0]) ? [mn[0] - 5, mx[0] + 5, mn[1] - 5, mx[1] + 5, mn[2] - 2, mx[2] + 2] : [-100, 100, -100, 100, -10, 50];
      }
      st.bounds = bb;
    }
    function diag() { var b = st.bounds; return Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]); }
    function wholeCam() { var b = st.bounds; return { tx: (b[0] + b[1]) / 2, ty: (b[2] + b[3]) / 2, tz: (b[4] + b[5]) / 2, dist: diag() * 0.9 }; }
    function reset() {
      var b = st.bounds, w = st.top.w, h = st.top.h, W = wholeCam();
      st.top.cx = W.tx; st.top.cy = W.ty;
      st.top.scale = Math.min(w / Math.max(1e-3, b[1] - b[0]), h / Math.max(1e-3, b[3] - b[2])) * 0.94;
      st.cam.tx = W.tx; st.cam.ty = W.ty; st.cam.tz = W.tz; st.cam.dist = W.dist; st.cam.yaw = -0.8; st.cam.pitch = 0.6;
      invalidate();
    }
    function buildBackdrop() {
      st.backdrop = null;
      if (!st.grid) return;
      var g = st.grid, cv = document.createElement('canvas');
      cv.width = g.w; cv.height = g.h;
      var cx = cv.getContext('2d'), img = cx.createImageData(g.w, g.h), span = Math.max(1e-3, g.zmax - g.zmin);
      for (var gy = 0; gy < g.h; gy++) for (var gx = 0; gx < g.w; gx++) {
        var i = gy * g.w + gx, row = g.h - 1 - gy, o = (row * g.w + gx) * 4;
        if (isNaN(g.min[i])) { img.data[o + 3] = 0; continue; }
        var c = ramp((g.min[i] - g.zmin) / span), wall = Math.max(0, Math.min(1, (g.max[i] - g.min[i]) / span * 4)), k = 1 - 0.4 * wall;
        img.data[o] = c[0] * k; img.data[o + 1] = c[1] * k; img.data[o + 2] = c[2] * k; img.data[o + 3] = 255;
      }
      cx.putImageData(img, 0, 0);
      st.backdrop = cv;
    }

    function invalidate() {
      if (st.dirty) return;
      st.dirty = true;
      (typeof requestAnimationFrame === 'function' ? requestAnimationFrame : function (f) { setTimeout(f, 16); })(function () { st.dirty = false; draw(); });
    }
    function resize() {
      var r = canvas.getBoundingClientRect(), dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
      var w = Math.max(50, Math.round(r.width)), h = Math.max(50, Math.round(r.height));
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr) || st.top.w !== w || st.top.h !== h) {
        var first = !st._sized;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        st.dpr = dpr; st.top.w = w; st.top.h = h; st.cam.w = w; st.cam.h = h;
        if (first) { st._sized = true; reset(); }
      }
    }
    function isVisible(i) { return st.visible === null || st.visible.has(st.items[i].index); }
    function glOn() { return !!glr && st.mode === '3d'; }

    // ----- drawing: top view -----
    function drawTop() {
      var v = st.top, W = v.w, H = v.h, b = st.bounds, i, it;
      var p0 = topToScreen(v, b[0], b[3]), p1 = topToScreen(v, b[1], b[2]);
      if (st.backdrop) { ctx.imageSmoothingEnabled = false; ctx.drawImage(st.backdrop, p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]); }
      else { ctx.fillStyle = cssVar('--mvr-bg2', 'rgba(128,128,128,0.08)'); ctx.fillRect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]); }
      var step = 1; while (step * v.scale < 28) step *= 10;
      ctx.lineWidth = 1; ctx.strokeStyle = cssVar('--mvr-grid', 'rgba(128,128,128,0.25)'); ctx.beginPath();
      for (var gx = Math.ceil(b[0] / step) * step; gx <= b[1]; gx += step) { var sx = Math.round(topToScreen(v, gx, 0)[0]) + 0.5; ctx.moveTo(sx, Math.max(0, p0[1])); ctx.lineTo(sx, Math.min(H, p1[1])); }
      for (var gy = Math.ceil(b[2] / step) * step; gy <= b[3]; gy += step) { var sy = Math.round(topToScreen(v, 0, gy)[1]) + 0.5; ctx.moveTo(Math.max(0, p0[0]), sy); ctx.lineTo(Math.min(W, p1[0]), sy); }
      ctx.stroke();
      ctx.strokeStyle = cssVar('--mvr-edge', 'rgba(128,128,128,0.8)'); ctx.strokeRect(p0[0] + 0.5, p0[1] + 0.5, p1[0] - p0[0], p1[1] - p0[1]);
      var tl = screenToTop(v, 0, 0), br = screenToTop(v, W, H);
      var vis = cullRect(st.items, tl[0], br[0], br[1], tl[1]);
      vis.sort(function (a, c) { return st.items[a].z - st.items[c].z; });
      st.stats.culled = st.items.length - vis.length;
      var batches = {}, dim = [];
      for (i = 0; i < vis.length; i++) {
        it = st.items[vis[i]];
        if (it.hidden || catHidden(it)) continue;
        if (!isVisible(vis[i])) { if (st.ghost) dim.push(vis[i]); continue; }
        var key = itemColor(it) + (st.transparency && translucentCat(it.cat) ? '|t' : '');
        (batches[key] = batches[key] || []).push(vis[i]);
      }
      var drawn = 0;
      var shape = function (idx) {
        var q = st.items[idx];
        if (q.hull && q.r * v.scale >= 2) {
          var h = q.hull; var s0 = topToScreen(v, h[0][0], h[0][1]); ctx.moveTo(s0[0], s0[1]);
          for (var k = 1; k < h.length; k++) { var s = topToScreen(v, h[k][0], h[k][1]); ctx.lineTo(s[0], s[1]); }
          ctx.closePath();
        } else { var c = topToScreen(v, q.cx, q.cy), m = 3.2; ctx.rect(c[0] - m, c[1] - m, 2 * m, 2 * m); }
      };
      ctx.globalAlpha = 0.12; ctx.fillStyle = cssVar('--mvr-dim', '#808080'); ctx.beginPath();
      for (i = 0; i < dim.length; i++) shape(dim[i]);
      ctx.fill(); ctx.globalAlpha = 1;
      Object.keys(batches).forEach(function (key) {
        var list = batches[key]; ctx.beginPath();
        for (var j = 0; j < list.length; j++) shape(list[j]);
        var tr = key.indexOf('|t') > 0, colr = tr ? key.slice(0, key.indexOf('|')) : key;
        ctx.globalAlpha = tr ? 0.22 : 0.62; ctx.fillStyle = colr; ctx.fill();
        ctx.globalAlpha = tr ? 0.6 : 1; ctx.strokeStyle = colr; ctx.lineWidth = 1; ctx.stroke();
        drawn += list.length;
      });
      if (!st.pack) {
        ctx.beginPath(); ctx.strokeStyle = cssVar('--mvr-text', '#222'); ctx.globalAlpha = 0.7;
        for (i = 0; i < vis.length; i++) {
          it = st.items[vis[i]];
          if (it.hidden || catHidden(it) || !isVisible(vis[i])) continue;
          var c0 = topToScreen(v, it.cx, it.cy), f = it.obj.forward;
          ctx.moveTo(c0[0], c0[1]); ctx.lineTo(c0[0] + f[0] * 9, c0[1] - f[1] * 9);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      }
      st.stats.drawn = drawn;
      highlight(topPoly);
    }
    function topPoly(idx) {
      var q = st.items[idx], v = st.top, pts = [];
      if (!q || q.hidden) return null;
      if (q.hull && q.r * v.scale >= 2) q.hull.forEach(function (h) { pts.push(topToScreen(v, h[0], h[1])); });
      else { var c = topToScreen(v, q.cx, q.cy); pts = [[c[0] - 5, c[1] - 5], [c[0] + 5, c[1] - 5], [c[0] + 5, c[1] + 5], [c[0] - 5, c[1] + 5]]; }
      return pts;
    }
    function highlight(polyOf) {
      var list = [];
      if (st.hover >= 0 && st.hover !== st.selected) list.push([st.hover, 1.5, cssVar('--mvr-hover', '#fff')]);
      if (st.selected !== null && st.selected >= 0) list.push([st.selected, 3, cssVar('--mvr-select', '#ffd400')]);
      list.forEach(function (e) {
        var pts = polyOf(e[0]); if (!pts || !pts.length) return;
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (var k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]); ctx.closePath();
        ctx.lineWidth = e[1]; ctx.strokeStyle = e[2]; ctx.stroke();
      });
    }

    // ----- drawing: 3D, WebGL -----
    function groundLines() {
      var b = st.bounds, z = b[4], out = [], c = st.cam, step = 1, span = Math.max(b[1] - b[0], b[3] - b[2]);
      while (span / step > 60) step *= 10;
      var g = [];
      for (var x = Math.ceil(b[0] / step) * step; x <= b[1]; x += step) g.push(x, b[2], z, x, b[3], z);
      for (var y = Math.ceil(b[2] / step) * step; y <= b[3]; y += step) g.push(b[0], y, z, b[1], y, z);
      out.push({ pts: new Float32Array(g), color: [0.5, 0.5, 0.55, 0.45] });
      var e = [], K = function (k) { return [(k & 1) ? b[1] : b[0], (k & 2) ? b[3] : b[2], (k & 4) ? b[5] : b[4]]; };
      [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (p) { e.push.apply(e, K(p[0])); e.push.apply(e, K(p[1])); });
      out.push({ pts: new Float32Array(e), color: [0.62, 0.64, 0.68, 0.9] });
      return out;
    }
    function drawGL() {
      var c = st.cam, w = Math.round(st.top.w * st.dpr), h = Math.round(st.top.h * st.dpr);
      glr.resize(w, h);
      var bs = CAM.math.basis(c), vp = CAM.math.viewProj(c, diag());
      var bg = cssBg();
      glr.render({
        vp: new Float32Array(vp), eye: bs.eye, clear: bg, dist: c.dist, lines: groundLines(), instances: st.gi, transparency: st.transparency, ghost: st.ghost,
        useMat: !!st.pack && (st.colorBy === 'auto' || st.colorBy === 'material'),
        selected: st.selected, hover: st.hover >= 0 && st.hover !== st.selected ? st.hover : null,
        outline: [hexRgb(cssVar('--mvr-select', '#ffb400')).map(function (x) { return x / 255; }), hexRgb(cssVar('--mvr-hover', '#ffffff')).map(function (x) { return x / 255; })]
      });
      st.stats.drawn = glr.stats.drawn; st.stats.culled = glr.stats.culled; st.glFrames++;
    }
    var bgCache = null;
    function cssBg() {
      if (bgCache) return bgCache;
      var rgb = [0.12, 0.14, 0.17];
      try { var m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(getComputedStyle(canvas.parentNode || canvas).backgroundColor || getComputedStyle(canvas).backgroundColor); if (m) rgb = [m[1] / 255, m[2] / 255, m[3] / 255]; } catch (e) { /* default */ }
      bgCache = rgb; return rgb;
    }

    // ----- drawing: 3D, Canvas painter (fallback) -----
    function drawOrbit() {
      var c = st.cam, b = st.bounds, bs = camBasis(c);
      var P = function (p) { return orbitProject(c, bs, p); };
      var drawables = [], i, k;
      if (st.grid) {
        var g = st.grid, S = 4, span = Math.max(1e-3, g.zmax - g.zmin), cw = (b[1] - b[0]) / g.w, ch = (b[3] - b[2]) / g.h;
        for (var ty = 0; ty < g.h; ty += S) for (var tx = 0; tx < g.w; tx += S) {
          var lo = Infinity, cnt = 0;
          for (var yy = ty; yy < Math.min(g.h, ty + S); yy++) for (var xx = tx; xx < Math.min(g.w, tx + S); xx++) { var z = g.min[yy * g.w + xx]; if (!isNaN(z)) { lo = Math.min(lo, z); cnt++; } }
          if (!cnt) continue;
          var x0 = b[0] + tx * cw, y0 = b[2] + ty * ch, x1 = x0 + S * cw, y1 = y0 + S * ch;
          var q = [P([x0, y0, lo]), P([x1, y0, lo]), P([x1, y1, lo]), P([x0, y1, lo])];
          if (q.some(function (a) { return !a; })) continue;
          var col = ramp((lo - g.zmin) / span);
          drawables.push({ depth: (q[0][2] + q[2][2]) / 2 + 1e3, poly: q, fill: 'rgb(' + (col[0] | 0) + ',' + (col[1] | 0) + ',' + (col[2] | 0) + ')', tile: true });
        }
      }
      var nvis = 0;
      for (i = 0; i < st.items.length; i++) {
        var it = st.items[i]; if (it.hidden || catHidden(it)) continue;
        var key = itemColor(it), corners = it.corners || boxCorners(it.pos, it.obj.forward, it.obj.up, DEFAULT_BOX), pc = corners.map(P);
        if (pc.some(function (a) { return !a; })) continue;
        var ctr = P([it.cx, it.cy, it.z]);
        drawables.push({ depth: ctr ? ctr[2] : pc[0][2], box: pc, corners: corners, color: hexRgb(key), on: isVisible(i), idx: i });
        nvis++;
      }
      drawables.sort(function (a, d) { return d.depth - a.depth; });
      var bc = []; for (k = 0; k < 8; k++) bc.push(P([(k & 1) ? b[1] : b[0], (k & 2) ? b[3] : b[2], (k & 4) ? b[5] : b[4]]));
      if (bc.every(function (a) { return a; })) {
        ctx.strokeStyle = cssVar('--mvr-edge', 'rgba(128,128,128,0.6)'); ctx.lineWidth = 1; ctx.beginPath();
        [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (e) { ctx.moveTo(bc[e[0]][0], bc[e[0]][1]); ctx.lineTo(bc[e[1]][0], bc[e[1]][1]); });
        ctx.stroke();
      }
      for (i = 0; i < drawables.length; i++) {
        var d = drawables[i];
        if (d.tile) { ctx.fillStyle = d.fill; ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.moveTo(d.poly[0][0], d.poly[0][1]); for (k = 1; k < 4; k++) ctx.lineTo(d.poly[k][0], d.poly[k][1]); ctx.closePath(); ctx.fill(); ctx.strokeStyle = d.fill; ctx.lineWidth = 1; ctx.stroke(); ctx.globalAlpha = 1; continue; }
        var it2 = st.items[d.idx], fr = it2.fr;
        ctx.globalAlpha = !d.on ? 0.1 : (st.transparency && translucentCat(st.items[d.idx].cat) ? 0.35 : 1);
        for (var f = 0; f < 6; f++) {
          var n = FACE_N[f], wn = [fr.f[0] * n[0] + fr.l[0] * n[1] + fr.u[0] * n[2], fr.f[1] * n[0] + fr.l[1] * n[1] + fr.u[1] * n[2], fr.f[2] * n[0] + fr.l[2] * n[1] + fr.u[2] * n[2]];
          var fc = FACES[f], mid = [0, 0, 0];
          for (k = 0; k < 4; k++) { mid[0] += d.corners[fc[k]][0] / 4; mid[1] += d.corners[fc[k]][1] / 4; mid[2] += d.corners[fc[k]][2] / 4; }
          if (dot(wn, [bs.eye[0] - mid[0], bs.eye[1] - mid[1], bs.eye[2] - mid[2]]) <= 0) continue;
          var sh = 0.5 + 0.5 * Math.max(0, dot(wn, LIGHT));
          ctx.fillStyle = 'rgb(' + ((d.color[0] * sh) | 0) + ',' + ((d.color[1] * sh) | 0) + ',' + ((d.color[2] * sh) | 0) + ')';
          ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 0.7;
          ctx.beginPath(); ctx.moveTo(d.box[fc[0]][0], d.box[fc[0]][1]);
          for (k = 1; k < 4; k++) ctx.lineTo(d.box[fc[k]][0], d.box[fc[k]][1]);
          ctx.closePath(); ctx.fill(); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      st.stats.drawn = nvis; st.stats.culled = 0;
      highlight(function (idx) {
        var it3 = st.items[idx]; if (!it3 || it3.hidden) return null;
        var cs = it3.corners || boxCorners(it3.pos, it3.obj.forward, it3.obj.up, DEFAULT_BOX), pp = cs.map(P).filter(function (a) { return a; });
        return pp.length > 2 ? hull2(pp.map(function (a) { return [a[0], a[1]]; })) : null;
      });
    }

    // ----- overlay: gizmo, help, hover label -----
    function drawGizmo() {
      var W = st.top.w, H = st.top.h, ox = 38, oy = H - 38, ax, right, up;
      if (st.mode === 'top') { right = [1, 0, 0]; up = [0, 1, 0]; ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; }
      else { var bs = camBasis(st.cam); right = bs.right; up = bs.up; ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; }
      var cols = ['#e5534b', '#57ab5a', '#4c8dff'], names = ['x', 'y', 'z'];
      var list = ax.map(function (a, i) { return { x: dot(a, right) * 22, y: -dot(a, up) * 22, d: st.mode === 'top' ? (i === 2 ? 1 : 0) : dot(a, camBasis(st.cam).fwd), c: cols[i], n: names[i] }; }).sort(function (a, b) { return b.d - a.d; });
      ctx.save(); ctx.globalAlpha = 0.9; ctx.font = 'bold 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineWidth = 2;
      list.forEach(function (a) {
        if (st.mode === 'top' && a.n === 'z') return;
        ctx.strokeStyle = a.c; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + a.x, oy + a.y); ctx.stroke();
        ctx.fillStyle = a.c; ctx.fillText(a.n, ox + a.x * 1.35, oy + a.y * 1.35);
      });
      ctx.restore();
    }
    function drawHelp() {
      ctx.save(); ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'top';
      var w = 0; HELP.forEach(function (l) { w = Math.max(w, ctx.measureText(l).width); });
      var x = 10, y = 10, hgt = HELP.length * 17 + 14;
      ctx.fillStyle = 'rgba(20,24,29,0.86)'; ctx.fillRect(x, y, Math.min(w + 20, st.top.w - 20), hgt);
      ctx.fillStyle = '#fff'; HELP.forEach(function (l, i) { ctx.fillText(l, x + 10, y + 8 + i * 17); });
      ctx.restore();
    }
    function cssVarCache() { return cssCache; }
    var cssCache = null;
    function cssVar(name, fallback) {
      if (!cssCache) {
        cssCache = {};
        try {
          var cs = getComputedStyle(canvas), g = function (n) { return cs.getPropertyValue(n).trim(); };
          cssCache['--mvr-text'] = g('color') || '#222';
          ['--mvr-grid', '--mvr-edge', '--mvr-bg2', '--mvr-select', '--mvr-hover', '--mvr-dim'].forEach(function (n) { cssCache[n] = g(n); });
        } catch (e) { /* no DOM styles: fall back */ }
      }
      return cssCache[name] || fallback;
    }

    function draw() {
      var t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      cssCache = null; bgCache = null;
      resize();
      var useGl = glOn();
      if (opts.glCanvas) { opts.glCanvas.style.visibility = useGl ? 'visible' : 'hidden'; }
      canvas.setAttribute('data-gl', useGl ? '1' : '0');
      ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
      ctx.clearRect(0, 0, st.top.w, st.top.h);
      if (st.mode === 'top') drawTop(); else if (useGl) { try { drawGL(); } catch (e) { glr = null; canvas.setAttribute('data-gl', '0'); if (opts.glCanvas) opts.glCanvas.style.visibility = 'hidden'; drawOrbit(); } } else drawOrbit();
      if (st.mode === '3d' || true) drawGizmo();
      if (st.help) drawHelp();
      if (st.hover >= 0 && st.mouse && opts.describe && st.items[st.hover]) {
        var text = opts.describe(st.items[st.hover].index);
        if (text) {
          ctx.font = '12px system-ui, sans-serif';
          var w = ctx.measureText(text).width + 10, x = Math.min(st.mouse[0] + 12, st.top.w - w - 2), y = Math.max(2, st.mouse[1] - 26);
          ctx.fillStyle = 'rgba(20,24,29,0.88)'; ctx.fillRect(x, y, w, 20);
          ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.fillText(text, x + 5, y + 10);
        }
      }
      st.stats.ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
    }

    // ----- picking (shared by hover, click, pivot and double-click) -----
    function pick(sx, sy) {
      if (st.mode === 'top') {
        var w = screenToTop(st.top, sx, sy);
        var h = pickTop(st.items.map(function (it, i) { return it.hidden || catHidden(it) || !isVisible(i) ? { hidden: true } : it; }), w[0], w[1], 6 / st.top.scale);
        return h >= 0 ? { index: st.items[h].index, item: h, point: [st.items[h].cx, st.items[h].cy, st.items[h].z] } : null;
      }
      var ray = CAM.math.rayFromScreen(st.cam, sx, sy);
      var pool = st.items.map(function (it, i) { return it.hidden || catHidden(it) || !isVisible(i) ? { hidden: true } : it; });
      var hit = CAM.math.pickRay(pool, ray, 0.35);
      if (hit) { var it = st.items[hit.i]; return { index: it.index, item: hit.i, t: hit.t, point: [ray.o[0] + ray.d[0] * hit.t, ray.o[1] + ray.d[1] * hit.t, ray.o[2] + ray.d[2] * hit.t] }; }
      return null;
    }

    var host = {
      mode: function () { return st.mode; }, cam: st.cam, top: st.top, diag: diag, wholeCam: wholeCam,
      pick: pick,
      hover: function (index, p) { var i = -1; if (index !== null && index !== undefined) for (var k = 0; k < st.items.length; k++) if (st.items[k].index === index) { i = k; break; } if (p) st.mouse = p; if (i !== st.hover || (i >= 0 && p)) { st.hover = i; canvas.style.cursor = i >= 0 ? 'pointer' : 'grab'; invalidate(); } },
      click: function (index) { api.setSelection(index); if (opts.onSelect) opts.onSelect(index === undefined ? null : index); },
      frameInfo: function () {
        var i = st.selected; if (i === null || i < 0 || !st.items[i] || st.items[i].hidden) return null;
        var it = st.items[i], cs = it.corners || boxCorners(it.pos, it.obj.forward, it.obj.up, DEFAULT_BOX), mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        cs.forEach(function (c) { for (var a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], c[a]); mx[a] = Math.max(mx[a], c[a]); } });
        return { min: mn, max: mx };
      },
      wholeMap: function () { if (ctl) { ctl.anim = null; ctl.animTop = null; ctl.vel = [0, 0]; } reset(); },
      setMode: function (m) { api.setMode(m); if (opts.onMode) opts.onMode(st.mode); },
      draw: function () { draw(); }, toggleHelp: function () { st.help = !st.help; invalidate(); }
    };
    var ctl = CAM ? CAM.createController(canvas, host) : null;
    var ro = null;
    if (typeof ResizeObserver === 'function') { ro = new ResizeObserver(function () { invalidate(); }); ro.observe(canvas); }
    else if (typeof window !== 'undefined') window.addEventListener('resize', invalidate);

    var api = {
      setVariant: function (parsed) { st.parsed = parsed; st.selected = null; st.hover = -1; st.visible = null; build(); fitBounds(); uploadGl(); reset(); },
      setPack: function (pack) {
        if (pack) { pack._pal = pack._pal || {}; (pack.palette || []).forEach(function (e) { pack._pal[e.i] = e; }); }
        st.pack = pack || null; st.meshes = {}; st.grid = pack && pack.grid ? decodeGrid(pack.grid) : null;
        try { st.textures = pack ? decodeTextures(pack.textures) : []; } catch (e) { st.textures = []; }
        buildBackdrop(); build();
        if (pack && pack.bounds && st.parsed && !(st.parsed.bounds && isFinite(st.parsed.bounds.xMin) && st.parsed.bounds.xMax > st.parsed.bounds.xMin)) { st.bounds = pack.bounds; }
        uploadGl(); invalidate();
      },
      setVisible: function (list) { st.visible = list === null ? null : new Set(list); refreshGl(); invalidate(); },
      setSelection: function (index) {
        st.selected = null;
        if (index !== null && index !== undefined) for (var i = 0; i < st.items.length; i++) if (st.items[i].index === index) { st.selected = i; break; }
        invalidate();
      },
      setMode: function (m) { st.mode = m === '3d' ? '3d' : 'top'; invalidate(); },
      setColorBy: function (c) { st.colorBy = c; refreshGl(); invalidate(); },
      setTransparency: function (on) { st.transparency = !!on; invalidate(); },
      setGhost: function (on) { st.ghost = !!on; invalidate(); },
      setHideCats: function (list) { st.hideCats = new Set(list || []); refreshGl(); invalidate(); },
      setHelp: function (on) { st.help = !!on; invalidate(); },
      setSpeed: function (v) { if (ctl) ctl.setSpeed(v); invalidate(); }, getSpeed: function () { return ctl ? ctl.speed : 1; },
      preset: function (name) { if (name === 'top') host.setMode('top'); else if (ctl) ctl.preset(name); if (opts.onMode) opts.onMode(st.mode); },
      focus: function () { if (ctl) ctl.frameSelection(); },
      fitObjects: function () {
        var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity], any = false;
        st.items.forEach(function (it, i) { if (it.hidden || !isVisible(i)) return; any = true; mn[0] = Math.min(mn[0], it.cx); mn[1] = Math.min(mn[1], it.cy); mn[2] = Math.min(mn[2], it.z); mx[0] = Math.max(mx[0], it.cx); mx[1] = Math.max(mx[1], it.cy); mx[2] = Math.max(mx[2], it.z); });
        if (!any) return;
        var dx = Math.max(20, mx[0] - mn[0] + 12), dy = Math.max(20, mx[1] - mn[1] + 12);
        st.top.cx = (mn[0] + mx[0]) / 2; st.top.cy = (mn[1] + mx[1]) / 2;
        st.top.scale = Math.min(st.top.w / dx, st.top.h / dy) * 0.94;
        st.cam.tx = st.top.cx; st.cam.ty = st.top.cy; st.cam.tz = (mn[2] + mx[2]) / 2;
        st.cam.dist = Math.max(30, Math.hypot(dx, dy, mx[2] - mn[2] + 6) * 1.1);
        invalidate();
      },
      reset: function () { host.wholeMap(); }, redraw: function () { draw(); }, resize: invalidate,
      project: function (index) {
        for (var i = 0; i < st.items.length; i++) if (st.items[i].index === index) {
          var it = st.items[i];
          if (st.mode === 'top') return topToScreen(st.top, it.cx, it.cy);
          var p = orbitProject(st.cam, camBasis(st.cam), [it.cx, it.cy, it.z]); return p ? [p[0], p[1]] : null;
        }
        return null;
      },
      pickAt: function (sx, sy) { var h = pick(sx, sy); return h ? h.index : null; },
      colorOf: function (index) { for (var i = 0; i < st.items.length; i++) if (st.items[i].index === index) return itemColor(st.items[i]); return null; },
      // wall-clock cost of `n` full 3D redraws (GL: includes gl.finish) with the camera orbiting, in ms per frame
      benchmark: function (n) {
        var m = st.mode, y0 = st.cam.yaw; st.mode = '3d'; var t0 = performance.now();
        for (var i = 0; i < n; i++) { st.cam.yaw = y0 + i * 0.02; draw(); if (glr && glOn()) glr.readPixel(0, 0); }
        var ms = (performance.now() - t0) / n; st.cam.yaw = y0; st.mode = m; draw(); return ms;
      },
      state: function () {
        return { mode: st.mode, top: Object.assign({}, st.top), cam: Object.assign({}, st.cam), selected: st.selected === null ? null : st.items[st.selected].index,
          hover: st.hover >= 0 ? st.items[st.hover].index : null, stats: Object.assign({}, st.stats), hasPack: !!st.pack, dpr: st.dpr, items: st.items.length,
          transparency: st.transparency, ghost: st.ghost, hidden: Array.from(st.hideCats), textures: (st.textures || []).length, gl: !!glr, glVersion: glr ? glr.version : 0, glStats: glr ? Object.assign({}, glr.stats) : null, glFrames: st.glFrames, help: st.help,
          speed: ctl ? ctl.speed : 1, meshes: Object.keys(st.meshes).filter(function (k) { return st.meshes[k]; }).length, anim: !!(ctl && (ctl.anim || ctl.animTop)) };
      },
      setCamera: function (o) { Object.assign(st.cam, o); invalidate(); },
      gl: function () { return glr; },
      controller: function () { return ctl; },
      destroy: function () { if (ro) ro.disconnect(); if (glr) glr.destroy(); }
    };
    return api;
  }

  var HmsRender = { create: create, math: math, colors: { category: CATEGORY, team: TEAM, physics: PHYS }, version: '2.0.0' };
  if (typeof window !== 'undefined') window.HmsRender = HmsRender;
  else if (typeof globalThis !== 'undefined') globalThis.HmsRender = HmsRender;
  if (typeof module === 'object' && module && module.exports) module.exports = HmsRender;
})();
