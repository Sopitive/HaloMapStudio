/*!
 * hms-camera.js - orbit / pan / zoom / fly camera maths and the input controller for the map view.
 *
 * Copyright (c) 2026 Sopitive. All rights reserved.
 * Licensed under the PolyForm Noncommercial License 1.0.0, with a separate commercial licence
 * available. Part of Halo Map Studio and covered by the same licence (see docs/license.html).
 *
 * A plain script (file:// and sub-path safe): global HmsCamera, module.exports under Node.
 *
 *   HmsCamera.math        pure functions on a camera object {tx,ty,tz,dist,yaw,pitch,fov,w,h}
 *                         (orbit, pan, zoomToward, setPivot, lookRotate, fly, frameBox, preset,
 *                         lerpCam, nearFar, viewProj, rayFromScreen, rayOBB, pickRay)
 *   HmsCamera.controller  pointer / wheel / keyboard / touch handling for a <canvas>; drives the
 *                         top-view state and the 3D camera of hms-render.js through a `host`.
 *
 * Controls (3D):  left-drag orbit about a pivot (the object under the cursor, else the current
 * target) - middle-drag or shift+left-drag pan - wheel zoom toward the cursor - right-drag look
 * around and, while held, W A S D move, Q E down / up (Shift fast, Ctrl slow) - double-click an
 * object to frame it - F focus the selection - Home whole map - 1 front, 2 side, 3 top, 4
 * isometric - ? help. Touch: one finger orbits, two fingers pan and pinch. Top view: drag pans,
 * wheel zooms toward the cursor, same keys.
 */
(function () {
  'use strict';

  var PITCH_MAX = 1.5533;      // 89 degrees: no flips over the pole
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function norm(v) { var m = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1; return [v[0] / m, v[1] / m, v[2] / m]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  // ------------------------------------------------------------------------------------------
  // pure camera maths
  // ------------------------------------------------------------------------------------------

  function eye(c) {
    var cp = Math.cos(c.pitch);
    return [c.tx + c.dist * cp * Math.cos(c.yaw), c.ty + c.dist * cp * Math.sin(c.yaw), c.tz + c.dist * Math.sin(c.pitch)];
  }
  function basis(c) {
    var e = eye(c), fwd = norm([c.tx - e[0], c.ty - e[1], c.tz - e[2]]), right = norm(cross(fwd, [0, 0, 1]));
    return { eye: e, fwd: fwd, right: right, up: cross(right, fwd) };
  }
  function distLimits(diag) { return [Math.max(0.25, diag * 0.002), Math.max(50, diag * 6)]; }

  function orbit(c, dx, dy, sens) {
    sens = sens || 0.006;
    c.yaw -= dx * sens;
    c.pitch = clamp(c.pitch + dy * sens, -PITCH_MAX, PITCH_MAX);
    return c;
  }
  function pan(c, dx, dy) {
    var b = basis(c), k = 2 * c.dist * Math.tan(c.fov / 2) / c.h;
    c.tx += -b.right[0] * dx * k + b.up[0] * dy * k;
    c.ty += -b.right[1] * dx * k + b.up[1] * dy * k;
    c.tz += -b.right[2] * dx * k + b.up[2] * dy * k;
    return c;
  }
  function rayFromScreen(c, sx, sy) {
    var b = basis(c), th = Math.tan(c.fov / 2), asp = c.w / c.h;
    var nx = (sx / c.w) * 2 - 1, ny = 1 - (sy / c.h) * 2;
    var d = norm([b.fwd[0] + b.right[0] * nx * th * asp + b.up[0] * ny * th, b.fwd[1] + b.right[1] * nx * th * asp + b.up[1] * ny * th, b.fwd[2] + b.right[2] * nx * th * asp + b.up[2] * ny * th]);
    return { o: b.eye, d: d };
  }
  // Zoom by `factor` (> 1 = closer) keeping the world point under pixel (sx, sy) fixed on screen.
  function zoomToward(c, factor, sx, sy, diag) {
    var lim = distLimits(diag || c.dist * 4), nd = clamp(c.dist / factor, lim[0], lim[1]), s = nd / c.dist;
    var ray = rayFromScreen(c, sx, sy), b = basis(c), t = c.dist / Math.max(1e-6, dot(ray.d, b.fwd));
    var P = [ray.o[0] + ray.d[0] * t, ray.o[1] + ray.d[1] * t, ray.o[2] + ray.d[2] * t];
    c.tx = P[0] + (c.tx - P[0]) * s; c.ty = P[1] + (c.ty - P[1]) * s; c.tz = P[2] + (c.tz - P[2]) * s;
    c.dist = nd;
    return c;
  }
  // Make `p` the orbit target WITHOUT moving the eye (no visible jump).
  function setPivot(c, p) {
    var e = eye(c), d = [e[0] - p[0], e[1] - p[1], e[2] - p[2]], r = Math.sqrt(dot(d, d));
    if (r < 1e-6) return c;
    c.tx = p[0]; c.ty = p[1]; c.tz = p[2]; c.dist = r;
    c.pitch = clamp(Math.asin(clamp(d[2] / r, -1, 1)), -PITCH_MAX, PITCH_MAX);
    c.yaw = Math.atan2(d[1], d[0]);
    return c;
  }
  // Free look: turn about the EYE, keeping it where it is.
  function lookRotate(c, dx, dy, sens) {
    sens = sens || 0.004;
    var e = eye(c);
    c.yaw -= dx * sens; c.pitch = clamp(c.pitch + dy * sens, -PITCH_MAX, PITCH_MAX);
    var cp = Math.cos(c.pitch);
    c.tx = e[0] - c.dist * cp * Math.cos(c.yaw); c.ty = e[1] - c.dist * cp * Math.sin(c.yaw); c.tz = e[2] - c.dist * Math.sin(c.pitch);
    return c;
  }
  // Move eye and target together: m = [right, forward, world-up] distances.
  function fly(c, m) {
    var b = basis(c), f = [b.fwd[0], b.fwd[1], b.fwd[2]];
    c.tx += b.right[0] * m[0] + f[0] * m[1]; c.ty += b.right[1] * m[0] + f[1] * m[1]; c.tz += b.right[2] * m[0] + f[2] * m[1] + m[2];
    return c;
  }
  function frameBox(c, mn, mx) {
    var ctr = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
    var r = Math.max(0.5, Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) / 2);
    var half = Math.min(c.fov, 2 * Math.atan(Math.tan(c.fov / 2) * c.w / c.h)) / 2;
    return { tx: ctr[0], ty: ctr[1], tz: ctr[2], dist: r / Math.sin(half) * 1.1, yaw: c.yaw, pitch: c.pitch };
  }
  var PRESETS = { front: [-Math.PI / 2, 0.05], side: [0, 0.05], top: [-Math.PI / 2, PITCH_MAX], iso: [-0.785, 0.62] };
  function preset(name) { var p = PRESETS[name]; return p ? { yaw: p[0], pitch: p[1] } : null; }
  function angDiff(a, b) { var d = (b - a) % (2 * Math.PI); if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return d; }
  function ease(t) { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function lerpCam(a, b, t) {
    var k = ease(t);
    return { tx: a.tx + (b.tx - a.tx) * k, ty: a.ty + (b.ty - a.ty) * k, tz: a.tz + (b.tz - a.tz) * k,
      dist: Math.exp(Math.log(a.dist) + (Math.log(b.dist) - Math.log(a.dist)) * k), yaw: a.yaw + angDiff(a.yaw, b.yaw) * k, pitch: a.pitch + (b.pitch - a.pitch) * k };
  }
  function nearFar(c, diag) {
    return { near: clamp(c.dist * 0.01, 0.01, 4), far: c.dist * 20 + (diag || 0) * 3 + 100 };
  }
  // Column-major perspective * view for WebGL.
  function viewProj(c, diag) {
    var b = basis(c), nf = nearFar(c, diag), f = 1 / Math.tan(c.fov / 2), asp = c.w / c.h;
    var P = [f / asp, 0, 0, 0, 0, f, 0, 0, 0, 0, (nf.far + nf.near) / (nf.near - nf.far), -1, 0, 0, 2 * nf.far * nf.near / (nf.near - nf.far), 0];
    var r = b.right, u = b.up, d = [-b.fwd[0], -b.fwd[1], -b.fwd[2]], e = b.eye;
    var V = [r[0], u[0], d[0], 0, r[1], u[1], d[1], 0, r[2], u[2], d[2], 0, -dot(r, e), -dot(u, e), -dot(d, e), 1];
    var out = new Array(16);
    for (var col = 0; col < 4; col++) for (var row = 0; row < 4; row++) {
      var s = 0; for (var k = 0; k < 4; k++) s += P[k * 4 + row] * V[col * 4 + k];
      out[col * 4 + row] = s;
    }
    return out;
  }
  // Ray vs oriented box. fr = {f,l,u} frame of the object, box = [minx,miny,minz,maxx,maxy,maxz] local.
  function rayOBB(o, d, pos, fr, box) {
    var rel = [o[0] - pos[0], o[1] - pos[1], o[2] - pos[2]];
    var lo = [dot(rel, fr.f), dot(rel, fr.l), dot(rel, fr.u)], ld = [dot(d, fr.f), dot(d, fr.l), dot(d, fr.u)];
    var t0 = 0, t1 = Infinity;
    for (var k = 0; k < 3; k++) {
      if (Math.abs(ld[k]) < 1e-9) { if (lo[k] < box[k] || lo[k] > box[k + 3]) return null; continue; }
      var a = (box[k] - lo[k]) / ld[k], b = (box[k + 3] - lo[k]) / ld[k];
      if (a > b) { var t = a; a = b; b = t; }
      if (a > t0) t0 = a; if (b < t1) t1 = b;
      if (t0 > t1) return null;
    }
    return t0;
  }
  // items: [{pos, fr, box, hidden}] -> {i, t} of the nearest hit along the ray, or null.
  function pickRay(items, ray, minHalf) {
    var best = null;
    for (var i = 0; i < items.length; i++) {
      var it = items[i]; if (it.hidden) continue;
      var box = it.box;
      if (minHalf) { box = box.slice(); for (var k = 0; k < 3; k++) { var m = (box[k] + box[k + 3]) / 2, h = Math.max((box[k + 3] - box[k]) / 2, minHalf); box[k] = m - h; box[k + 3] = m + h; } }
      var t = rayOBB(ray.o, ray.d, it.pos, it.fr, box);
      if (t !== null && (best === null || t < best.t)) best = { i: i, t: t };
    }
    return best;
  }
  // Smooth step of a pending zoom (log-space): apply a fraction of what is left; never overshoots.
  function zoomStep(pending, dt) {
    var k = 1 - Math.exp(-dt / 0.07), take = pending * k;
    return { take: Math.abs(pending) < 1e-4 ? pending : take, left: Math.abs(pending) < 1e-4 ? 0 : pending - take };
  }
  // Inertia: exponential decay that cannot change sign.
  function decay(v, dt) { var r = v * Math.exp(-dt / 0.12); return Math.abs(r) < 0.02 ? 0 : r; }

  var math = {
    PITCH_MAX: PITCH_MAX, eye: eye, basis: basis, distLimits: distLimits, orbit: orbit, pan: pan, rayFromScreen: rayFromScreen, zoomToward: zoomToward,
    setPivot: setPivot, lookRotate: lookRotate, fly: fly, frameBox: frameBox, preset: preset, lerpCam: lerpCam, nearFar: nearFar, viewProj: viewProj,
    rayOBB: rayOBB, pickRay: pickRay, zoomStep: zoomStep, decay: decay, ease: ease, clamp: clamp
  };

  // ------------------------------------------------------------------------------------------
  // controller
  // ------------------------------------------------------------------------------------------
  //
  // host = {
  //   mode(): 'top'|'3d',  cam, top (live state objects),  diag(): map size,
  //   pick(sx, sy): {index, point:[x,y,z], box:[min,max]} | null,   hover(index|null), click(index|null),
  //   frameInfo(): {min,max} of the selection, or null,  wholeMap(),  setMode(m), draw(), toggleHelp()
  // }
  var SPEED_KEY = 'hms-mvar-cam-speed';
  function createController(canvas, host) {
    var ctl = { speed: 1, anim: null, pendingZoom: 0, zoomAt: [0, 0], vel: [0, 0], keys: {}, flying: false };
    try { var s = parseFloat(localStorage.getItem(SPEED_KEY)); if (s > 0 && s <= 10) ctl.speed = s; } catch (e) { /* storage blocked */ }
    ctl.setSpeed = function (v) { ctl.speed = clamp(v, 0.05, 10); try { localStorage.setItem(SPEED_KEY, String(ctl.speed)); } catch (e) { /* ignore */ } };
    var ptrs = {}, drag = null, last = null, raf = 0;

    function rect() { return canvas.getBoundingClientRect(); }
    function pos(e) { var r = rect(); return [e.clientX - r.left, e.clientY - r.top]; }
    function wake() {
      if (raf) return;
      var rq = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : function (f) { return setTimeout(function () { f(Date.now()); }, 16); };
      raf = rq(function (t) { raf = 0; var dt = last === null ? 0.016 : Math.min(0.1, (t - last) / 1000); last = t; if (ctl.tick(dt)) wake(); else last = null; });
    }
    ctl.wake = wake;

    ctl.animateTo = function (target, ms) {
      var c = host.cam;
      ctl.anim = { from: { tx: c.tx, ty: c.ty, tz: c.tz, dist: c.dist, yaw: c.yaw, pitch: c.pitch }, to: target, t: 0, ms: Math.max(1, ms || 450) };
      ctl.vel = [0, 0]; wake();
    };
    ctl.animateTop = function (target, ms) { ctl.animTop = { from: { cx: host.top.cx, cy: host.top.cy, scale: host.top.scale }, to: target, t: 0, ms: ms || 350 }; wake(); };
    ctl.frameSelection = function () {
      var fi = host.frameInfo();
      if (!fi) { host.wholeMap(); return; }
      if (host.mode() === 'top') {
        var dx = Math.max(10, fi.max[0] - fi.min[0] + 8), dy = Math.max(10, fi.max[1] - fi.min[1] + 8);
        ctl.animateTop({ cx: (fi.min[0] + fi.max[0]) / 2, cy: (fi.min[1] + fi.max[1]) / 2, scale: Math.min(host.top.w / dx, host.top.h / dy) * 0.6 }, 350);
      } else ctl.animateTo(frameBox(host.cam, fi.min, fi.max), 450);
    };
    ctl.preset = function (name) {
      var p = preset(name); if (!p) return;
      host.setMode('3d');
      var t = host.wholeCam ? host.wholeCam() : { tx: host.cam.tx, ty: host.cam.ty, tz: host.cam.tz, dist: host.cam.dist };
      ctl.animateTo({ tx: t.tx, ty: t.ty, tz: t.tz, dist: t.dist, yaw: p.yaw, pitch: p.pitch }, 500);
    };

    // one frame of animation / inertia / flying / smooth zoom; true while more frames are needed
    ctl.tick = function (dt) {
      var more = false, c = host.cam;
      if (ctl.anim) {
        ctl.anim.t += dt * 1000 / ctl.anim.ms;
        var v = lerpCam(ctl.anim.from, ctl.anim.to, ctl.anim.t);
        c.tx = v.tx; c.ty = v.ty; c.tz = v.tz; c.dist = v.dist; c.yaw = v.yaw; c.pitch = v.pitch;
        if (ctl.anim.t >= 1) { var T = ctl.anim.to; c.tx = T.tx; c.ty = T.ty; c.tz = T.tz; c.dist = T.dist; c.yaw = T.yaw; c.pitch = T.pitch; ctl.anim = null; } else more = true;
      }
      if (ctl.animTop) {
        ctl.animTop.t += dt * 1000 / ctl.animTop.ms;
        var k = ease(ctl.animTop.t), a = ctl.animTop.from, b = ctl.animTop.to;
        host.top.cx = a.cx + (b.cx - a.cx) * k; host.top.cy = a.cy + (b.cy - a.cy) * k;
        host.top.scale = Math.exp(Math.log(a.scale) + (Math.log(b.scale) - Math.log(a.scale)) * k);
        if (ctl.animTop.t >= 1) ctl.animTop = null; else more = true;
      }
      if (Math.abs(ctl.pendingZoom) > 1e-4) {
        var z = zoomStep(ctl.pendingZoom, dt); ctl.pendingZoom = z.left;
        zoomBy(Math.exp(z.take), ctl.zoomAt[0], ctl.zoomAt[1]);
        if (Math.abs(ctl.pendingZoom) > 1e-4) more = true;
      }
      if (host.mode() === '3d' && (ctl.vel[0] !== 0 || ctl.vel[1] !== 0) && !drag) {
        orbit(c, ctl.vel[0] * dt, ctl.vel[1] * dt, 0.006);
        ctl.vel = [decay(ctl.vel[0], dt), decay(ctl.vel[1], dt)];
        if (ctl.vel[0] !== 0 || ctl.vel[1] !== 0) more = true;
      }
      var mx = (ctl.keys.d ? 1 : 0) - (ctl.keys.a ? 1 : 0), my = (ctl.keys.w ? 1 : 0) - (ctl.keys.s ? 1 : 0), mz = (ctl.keys.e ? 1 : 0) - (ctl.keys.q ? 1 : 0);
      if ((mx || my || mz) && host.mode() === '3d') {
        var base = Math.max(5, host.diag() * 0.25) * ctl.speed * (ctl.keys.shift ? 4 : 1) * (ctl.keys.ctrl ? 0.25 : 1);
        fly(c, [mx * base * dt, my * base * dt, mz * base * dt]);
        more = true;
      } else if ((mx || my) && host.mode() === 'top') {
        var sp = 400 * ctl.speed * (ctl.keys.shift ? 3 : 1) * dt / host.top.scale;
        host.top.cx += mx * sp; host.top.cy += my * sp; more = true;
      }
      host.draw();
      return more || !!drag;
    };

    function zoomBy(f, sx, sy) {
      if (host.mode() === 'top') {
        var T = host.top, bx = T.cx + (sx - T.w / 2) / T.scale, by = T.cy - (sy - T.h / 2) / T.scale;
        T.scale = clamp(T.scale * f, 0.02, 400);
        T.cx = bx - (sx - T.w / 2) / T.scale; T.cy = by + (sy - T.h / 2) / T.scale;
      } else zoomToward(host.cam, f, sx, sy, host.diag());
    }

    canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    canvas.addEventListener('pointerdown', function (e) {
      try { canvas.setPointerCapture(e.pointerId); } catch (x) { /* synthetic */ }
      if (typeof canvas.focus === 'function') canvas.focus({ preventScroll: true });
      var p = pos(e); ptrs[e.pointerId] = p; ctl.anim = null; ctl.animTop = null; ctl.vel = [0, 0];
      var n = Object.keys(ptrs).length;
      if (n >= 2) { drag = { kind: 'multi' }; var a = Object.keys(ptrs).map(function (k) { return ptrs[k]; }); drag.d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]); drag.c = [(a[0][0] + a[1][0]) / 2, (a[0][1] + a[1][1]) / 2]; return; }
      var kind = e.button === 2 ? 'look' : (e.button === 1 || (e.button === 0 && e.shiftKey)) ? 'pan' : 'orbit';
      if (host.mode() === 'top' && kind !== 'pan') kind = e.button === 2 ? 'pan' : 'pan';
      drag = { kind: kind, start: p, last: p, moved: 0, button: e.button, t: performance.now(), picked: false };
      if (kind === 'look') { ctl.flying = true; }
      wake();
    });
    canvas.addEventListener('pointermove', function (e) {
      var p = pos(e);
      if (ptrs[e.pointerId]) ptrs[e.pointerId] = p;
      if (drag && drag.kind === 'multi' && Object.keys(ptrs).length >= 2) {
        var a = Object.keys(ptrs).map(function (k) { return ptrs[k]; }), d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), c2 = [(a[0][0] + a[1][0]) / 2, (a[0][1] + a[1][1]) / 2];
        panBy(c2[0] - drag.c[0], c2[1] - drag.c[1]);
        if (drag.d > 0 && d > 0) zoomBy(d / drag.d, c2[0], c2[1]);
        drag.d = d; drag.c = c2; host.draw(); return;
      }
      if (drag && drag.kind !== 'multi') {
        var dx = p[0] - drag.last[0], dy = p[1] - drag.last[1], now = performance.now();
        drag.moved += Math.abs(dx) + Math.abs(dy);
        if (drag.moved > 4 || drag.kind === 'look') {
          if (drag.kind === 'orbit' && !drag.picked) {          // pivot follows the cursor: orbit about what is under it
            drag.picked = true;
            if (host.mode() === '3d') { var h = host.pick(drag.start[0], drag.start[1]); if (h && h.point) setPivot(host.cam, h.point); }
          }
          if (drag.kind === 'orbit') {
            if (host.mode() === '3d') {
              orbit(host.cam, dx, dy, 0.006);
              var dtm = Math.max(8, now - (drag.tl || now - 16)) / 1000; ctl.vel = [clamp(dx / dtm * 0.6 + ctl.vel[0] * 0.4, -1500, 1500), clamp(dy / dtm * 0.6 + ctl.vel[1] * 0.4, -1500, 1500)];
              drag.tl = now;
            }
          } else if (drag.kind === 'pan') panBy(dx, dy);
          else if (drag.kind === 'look') lookRotate(host.cam, dx, dy, 0.004);
          drag.last = p; host.hover(null); host.draw(); return;
        }
        return;
      }
      var hit = host.pick(p[0], p[1]); host.hover(hit ? hit.index : null, p);
    });
    function panBy(dx, dy) {
      if (host.mode() === 'top') { host.top.cx -= dx / host.top.scale; host.top.cy += dy / host.top.scale; } else pan(host.cam, dx, dy);
    }
    function up(e) {
      var d = drag; delete ptrs[e.pointerId];
      if (Object.keys(ptrs).length === 0) drag = null; else if (d && d.kind === 'multi' && Object.keys(ptrs).length < 2) drag = null;
      ctl.flying = false;
      if (d && d.kind !== 'multi') {
        if (d.kind === 'look') ctl.keys = {};
        if (d.moved <= 4 && e.button === 0 && !e.shiftKey) { var h = host.pick(d.start[0], d.start[1]); ctl.lastClick = { p: d.start, index: h ? h.index : null, t: performance.now() }; host.click(h ? h.index : null); }
        if (d.kind === 'orbit' && performance.now() - (d.tl || 0) > 80) ctl.vel = [0, 0];   // a pause before release: no fling
      }
      wake();
    }
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('pointerleave', function () { host.hover(null); });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      var dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 100 : e.deltaY;
      if (e.altKey && host.mode() === '3d') { ctl.setSpeed(ctl.speed * Math.exp(-dy * 0.002)); host.draw(); return; }   // Alt+wheel: fly speed
      var p = pos(e); ctl.zoomAt = p; ctl.pendingZoom = clamp(ctl.pendingZoom - dy * 0.0016, -1.6, 1.6); wake();
    }, { passive: false });
    canvas.addEventListener('dblclick', function (e) {
      var p = pos(e), lc = ctl.lastClick, h;
      // a double-click is two clicks: act on the object the first one hit (the pointer can drift a pixel between them)
      if (lc && lc.index !== null && performance.now() - lc.t < 700 && Math.hypot(lc.p[0] - p[0], lc.p[1] - p[1]) < 5) h = { index: lc.index };
      else h = host.pick(p[0], p[1]);
      if (h) { host.click(h.index); ctl.frameSelection(); }
    });
    var KEYMAP = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', KeyQ: 'q', KeyE: 'e' };
    canvas.addEventListener('keydown', function (e) {
      var k = KEYMAP[e.code] || (e.key && KEYMAP['Key' + e.key.toUpperCase()]);
      if (e.key === 'Shift') ctl.keys.shift = true;
      if (e.key === 'Control') ctl.keys.ctrl = true;
      if (k && !e.ctrlKey && !e.metaKey) { ctl.keys[k] = true; e.preventDefault(); wake(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'f' || e.key === 'F') { ctl.frameSelection(); e.preventDefault(); }
      else if (e.key === 'Home') { host.wholeMap(); e.preventDefault(); }
      else if (e.key === '1') ctl.preset('front'); else if (e.key === '2') ctl.preset('side');
      else if (e.key === '3') host.setMode('top'); else if (e.key === '4') ctl.preset('iso');
      else if (e.key === '?' || e.key === '/') { host.toggleHelp(); e.preventDefault(); }
      else if (e.key === 'Escape') host.click(null);
    });
    canvas.addEventListener('keyup', function (e) {
      var k = KEYMAP[e.code] || (e.key && KEYMAP['Key' + e.key.toUpperCase()]);
      if (k) ctl.keys[k] = false;
      if (e.key === 'Shift') ctl.keys.shift = false;
      if (e.key === 'Control') ctl.keys.ctrl = false;
    });
    canvas.addEventListener('blur', function () { ctl.keys = {}; });
    if (canvas.tabIndex < 0) canvas.tabIndex = 0;
    canvas.style.touchAction = 'none';
    return ctl;
  }

  var HmsCamera = { math: math, createController: createController, version: '1.0.0' };
  if (typeof window !== 'undefined') window.HmsCamera = HmsCamera;
  else if (typeof globalThis !== 'undefined') globalThis.HmsCamera = HmsCamera;
  if (typeof module === 'object' && module && module.exports) module.exports = HmsCamera;
})();
