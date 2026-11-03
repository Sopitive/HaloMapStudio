/*!
 * hms-gl.js - a small WebGL renderer for the variant map view: instanced meshes with a depth
 * buffer, hemisphere + directional light, flat shading, a lit terrain from the render pack's
 * height grid, outlines for the selected / hovered object, and translucent dimmed objects.
 *
 * Copyright (c) 2026 Sopitive. All rights reserved.
 * Licensed under the PolyForm Noncommercial License 1.0.0, with a separate commercial licence
 * available. Part of Halo Map Studio and covered by the same licence (see docs/license.html).
 *
 * A plain script (file:// and sub-path safe): global HmsGL. HmsGL.create(canvas) returns a renderer
 * or null when WebGL (2, or 1 with ANGLE_instanced_arrays) is not available; hms-render.js then keeps
 * its Canvas painter. No dependencies.
 */
(function () {
  'use strict';

  var VS = [
    'attribute vec3 aPos; attribute vec3 aNrm; attribute vec4 aR0; attribute vec4 aR1; attribute vec4 aR2; attribute vec4 aCol; attribute vec3 aVC; attribute vec2 aUV;',
    'uniform mat4 uVP; uniform float uOutline; uniform float uVCol; uniform vec3 uCen; uniform float uOutK;',
    'varying vec3 vN; varying vec4 vC; varying vec3 vW; varying vec2 vUV;',
    'void main() {',
    '  vec3 lp = aPos + (aPos - uCen) * uOutK;',
    '  vec3 wp = vec3(dot(aR0.xyz, lp) + aR0.w, dot(aR1.xyz, lp) + aR1.w, dot(aR2.xyz, lp) + aR2.w);',
    '  vec3 wn = normalize(vec3(dot(aR0.xyz, aNrm), dot(aR1.xyz, aNrm), dot(aR2.xyz, aNrm)));',
    '  wp += wn * uOutline;',
    '  vN = wn; vW = wp; vUV = aUV; vC = uVCol > 0.5 ? vec4(aVC, 1.0) : aCol;',
    '  gl_Position = uVP * vec4(wp, 1.0);',
    '}'
  ].join('\n');
  // Fragment lanes (uMode): 0 opaque, 1 cutout (alpha test), 2 blend (premultiplied alpha), 3 additive, 4 ghost (a filtered-out object).
  // Colour: the part's diffuse (average colour x tiny texture) when uUseMat, else the instance colour (category / team / physics).
  var FS = [
    'precision mediump float;',
    'varying vec3 vN; varying vec4 vC; varying vec3 vW; varying vec2 vUV;',
    'uniform vec3 uEye; uniform vec3 uLight; uniform vec4 uFlat; uniform float uLit;',
    'uniform float uMode; uniform vec3 uPartCol; uniform float uUseMat; uniform float uOpacity; uniform sampler2D uTex; uniform float uHasTex; uniform vec2 uUVScale;',
    'void main() {',
    '  if (uFlat.a > 0.5) { gl_FragColor = vec4(uFlat.rgb, 1.0); return; }',
    '  vec3 base = uUseMat > 0.5 ? uPartCol : vC.rgb;',
    '  float ta = 1.0;',
    '  if (uHasTex > 0.5) { vec4 t = texture2D(uTex, vUV); ta = t.a; if (uUseMat > 0.5) base = t.rgb * mix(vec3(1.0), uPartCol * 1.6, 0.0); }',
    '  if (uMode > 0.5 && uMode < 1.5 && uHasTex > 0.5 && ta < 0.5) discard;',
    '  if (uMode > 3.5) { float g = dot(base, vec3(0.3, 0.59, 0.11)); gl_FragColor = vec4(vec3(mix(g, base.r, 0.25)) * vec3(0.9, 0.95, 1.0) * uOpacity, uOpacity); return; }',
    '  if (uMode > 2.5) { gl_FragColor = vec4(base * uOpacity, 0.0); return; }',
    '  vec3 n = normalize(vN);',
    '  if (!gl_FrontFacing) n = -n;',
    '  float hemi = n.z * 0.5 + 0.5;',
    '  vec3 amb = mix(vec3(0.30, 0.28, 0.27), vec3(0.62, 0.68, 0.76), hemi);',
    '  float sun = max(dot(n, uLight), 0.0);',
    '  vec3 v = normalize(uEye - vW);',
    '  float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;',
    '  vec3 lit = base * (amb + vec3(1.0, 0.96, 0.88) * sun * 0.62) + rim;',
    '  vec3 col = mix(base, lit, uLit);',
    '  if (uMode > 1.5) { gl_FragColor = vec4(col * uOpacity, uOpacity); return; }',
    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');
  var LVS = 'attribute vec3 aP; uniform mat4 uVP; void main() { gl_Position = uVP * vec4(aP, 1.0); }';
  var LFS = 'precision mediump float; uniform vec4 uC; void main() { gl_FragColor = uC; }';

  function sh(gl, type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
    return s;
  }
  function prog(gl, vs, fs, attrs) {
    var p = gl.createProgram(); gl.attachShader(p, sh(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl, gl.FRAGMENT_SHADER, fs));
    attrs.forEach(function (a, i) { gl.bindAttribLocation(p, i, a); });
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(p));
    return p;
  }
  function uni(gl, p, names) { var o = {}; names.forEach(function (n) { o[n] = gl.getUniformLocation(p, n); }); return o; }

  // 6 frustum planes (Gribb-Hartmann) from a column-major view-projection matrix.
  function frustum(m) {
    var P = [], r = function (i) { return [m[i], m[4 + i], m[8 + i], m[12 + i]]; }, r0 = r(0), r1 = r(1), r2 = r(2), r3 = r(3);
    var add = function (a, b, s) { var p = [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]], l = Math.hypot(p[0], p[1], p[2]) || 1; P.push([p[0] / l, p[1] / l, p[2] / l, p[3] / l]); };
    add(r3, r0, 1); add(r3, r0, -1); add(r3, r1, 1); add(r3, r1, -1); add(r3, r2, 1); add(r3, r2, -1);
    return P;
  }
  function sphereVisible(planes, x, y, z, r) {
    for (var i = 0; i < 6; i++) { var p = planes[i]; if (p[0] * x + p[1] * y + p[2] * z + p[3] < -r) return false; }
    return true;
  }
  var CUBE = (function () {   // unit cube [0,1]^3, flat normals, 36 vertices
    var f = [[[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [-1, 0, 0]], [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, -1, 0]],
      [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, -1]], [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]]];
    var pos = [], nrm = [];
    f.forEach(function (q) { [0, 1, 2, 0, 2, 3].forEach(function (i) { pos.push(q[i][0], q[i][1], q[i][2]); nrm.push(q[4][0], q[4][1], q[4][2]); }); });
    return { positions: new Float32Array(pos), normals: new Float32Array(nrm) };
  })();

  function create(canvas) {
    var attrs = { antialias: true, alpha: false, depth: true, stencil: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' };
    var gl = null, gl2 = false;
    try {
      gl = canvas.getContext('webgl2', attrs); gl2 = !!gl;
      if (!gl) gl = canvas.getContext('webgl', attrs) || canvas.getContext('experimental-webgl', attrs);
    } catch (e) { gl = null; }
    if (!gl) return null;
    var ext = gl2 ? null : gl.getExtension('ANGLE_instanced_arrays');
    if (!gl2 && !ext) return null;
    var divisor = function (loc, d) { if (gl2) gl.vertexAttribDivisor(loc, d); else ext.vertexAttribDivisorANGLE(loc, d); };
    var drawInst = function (n, inst) { if (gl2) gl.drawArraysInstanced(gl.TRIANGLES, 0, n, inst); else ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, n, inst); };
    var drawInst2 = function (first, n, inst) { if (gl2) gl.drawArraysInstanced(gl.TRIANGLES, first, n, inst); else ext.drawArraysInstancedANGLE(gl.TRIANGLES, first, n, inst); };
    var P, U, LP, LU;
    try {
      P = prog(gl, VS, FS, ['aPos', 'aNrm', 'aR0', 'aR1', 'aR2', 'aCol', 'aVC', 'aUV']);
      U = uni(gl, P, ['uVP', 'uOutline', 'uVCol', 'uEye', 'uLight', 'uFlat', 'uLit', 'uCen', 'uOutK', 'uMode', 'uPartCol', 'uUseMat', 'uOpacity', 'uTex', 'uHasTex']);
      LP = prog(gl, LVS, LFS, ['aP']); LU = uni(gl, LP, ['uVP', 'uC']);
    } catch (e) { return null; }

    var meshes = {}, textures = [], whiteTex = null, terrain = null, lineBuf = gl.createBuffer(), instBuf = gl.createBuffer();
    var stats = { instances: 0, drawn: 0, culled: 0, tris: 0, ms: 0, draws: 0 };
    var identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

    function upload(positions, normals, vcol, uv, parts) {
      var m = { n: positions.length / 3, pos: gl.createBuffer(), nrm: gl.createBuffer(), vc: null, uv: null, parts: parts || [{ lane: 'opaque', s: 0, n: positions.length / 9, c: null, o: 1, t: -1 }] };
      var lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30];
      for (var q = 0; q < positions.length; q += 3) for (var a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], positions[q + a]); hi[a] = Math.max(hi[a], positions[q + a]); }
      m.cen = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]; m.rad = Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
      m.hasTrans = m.parts.some(function (p) { return p.lane === 'blend' || p.lane === 'additive'; });
      if (uv) { m.uv = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, m.uv); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW); }
      gl.bindBuffer(gl.ARRAY_BUFFER, m.pos); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.nrm); gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
      if (vcol) { m.vc = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, m.vc); gl.bufferData(gl.ARRAY_BUFFER, vcol, gl.STATIC_DRAW); }
      return m;
    }
    function free(m) { if (!m) return; gl.deleteBuffer(m.pos); gl.deleteBuffer(m.nrm); if (m.vc) gl.deleteBuffer(m.vc); if (m.uv) gl.deleteBuffer(m.uv); }
    meshes.cube = upload(CUBE.positions, CUBE.normals);
    whiteTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, whiteTex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    function bindMesh(m, vcol) {
      gl.bindBuffer(gl.ARRAY_BUFFER, m.pos); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); divisor(0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.nrm); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0); divisor(1, 0);
      if (vcol && m.vc) { gl.bindBuffer(gl.ARRAY_BUFFER, m.vc); gl.enableVertexAttribArray(6); gl.vertexAttribPointer(6, 3, gl.FLOAT, false, 0, 0); divisor(6, 0); } else gl.disableVertexAttribArray(6);
      if (m.uv) { gl.bindBuffer(gl.ARRAY_BUFFER, m.uv); gl.enableVertexAttribArray(7); gl.vertexAttribPointer(7, 2, gl.FLOAT, false, 0, 0); divisor(7, 0); } else { gl.disableVertexAttribArray(7); gl.vertexAttrib2f(7, 0, 0); }
    }
    function instAttribs(byteOffset) {
      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      for (var k = 0; k < 4; k++) { gl.enableVertexAttribArray(2 + k); gl.vertexAttribPointer(2 + k, 4, gl.FLOAT, false, 64, byteOffset + k * 16); divisor(2 + k, 1); }
    }
    function constInst(mat, col) {   // a single constant "instance" with no instance buffer (terrain)
      for (var k = 0; k < 4; k++) { gl.disableVertexAttribArray(2 + k); divisor(2 + k, 0); }
      gl.vertexAttrib4f(2, mat[0], mat[1], mat[2], mat[3]); gl.vertexAttrib4f(3, mat[4], mat[5], mat[6], mat[7]); gl.vertexAttrib4f(4, mat[8], mat[9], mat[10], mat[11]); gl.vertexAttrib4f(5, col[0], col[1], col[2], col[3]);
    }

    var api = {
      gl: gl, version: gl2 ? 2 : 1,
      // map: key -> {positions: Float32Array, normals: Float32Array} (triangle soup, flat normals)
      setMeshes: function (map) {
        Object.keys(meshes).forEach(function (k) { if (k !== 'cube') { free(meshes[k]); delete meshes[k]; } });
        Object.keys(map).forEach(function (k) { meshes[k] = upload(map[k].positions, map[k].normals, null, map[k].uv, map[k].parts); });
      },
      // list of {w, h, rgba: Uint8Array}: the pack's tiny diffuse textures (mipmapped, repeating)
      setTextures: function (list) {
        textures.forEach(function (t) { gl.deleteTexture(t); }); textures = [];
        list.forEach(function (t) {
          var tx = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tx);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, t.w, t.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, t.rgba);
          var pot = (t.w & (t.w - 1)) === 0 && (t.h & (t.h - 1)) === 0;
          if (gl2 || pot) { gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT); }
          else { gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); }
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          textures.push(tx);
        });
      },
      hasMesh: function (key) { return !!meshes[key]; },
      // terrain: {positions, normals, colors} Float32Arrays
      setTerrain: function (t) { free(terrain); terrain = t ? upload(t.positions, t.normals, t.colors) : null; },
      resize: function (w, h) { if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; } },
      stats: stats,
      // frame = {vp[16], eye[3], clear[3], lines:[{pts:Float32Array, color:[4]}], instances:[...], selected, hover, outline:[r,g,b], dist}
      // instance = {key, mat[12], col[4], cx,cy,cz,r, dim:boolean}
      render: function (frame) {
        var t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(frame.clear[0], frame.clear[1], frame.clear[2], 1); gl.clearDepth(1);
        gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.STENCIL_TEST); gl.clearStencil(0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
        var planes = frustum(frame.vp), list = frame.instances, groups = {}, ghosts = {}, trans = [], drawn = 0, culled = 0, i, it;
        var transparency = frame.transparency !== false, useMat = !!frame.useMat;
        for (i = 0; i < list.length; i++) {
          it = list[i];
          if (!sphereVisible(planes, it.cx, it.cy, it.cz, it.r)) { culled++; continue; }
          if (it.dim) { if (frame.ghost !== false) { (ghosts[it.key] = ghosts[it.key] || []).push(it); drawn++; } continue; }
          var m0 = meshes[it.key] || meshes.cube;
          (groups[it.key] = groups[it.key] || []).push(it); drawn++;
          if (transparency && m0.hasTrans) trans.push(it);
        }
        // sorted far -> near (distance to the eye) for the translucent parts of opaque-visible instances
        var ex = frame.eye;
        trans.forEach(function (t) { t.depth = (t.cx - ex[0]) * (t.cx - ex[0]) + (t.cy - ex[1]) * (t.cy - ex[1]) + (t.cz - ex[2]) * (t.cz - ex[2]); });
        trans.sort(function (a, b) { return b.depth - a.depth; });
        if (trans.length > 4000) trans = trans.slice(trans.length - 4000);
        // one instance buffer: [opaque groups][ghost groups][translucent, sorted][outline extras]
        var order = [], ghostOrder = [], total = 0;
        Object.keys(groups).forEach(function (k) { order.push({ key: k, list: groups[k], off: total }); total += groups[k].length; });
        Object.keys(ghosts).forEach(function (k) { ghostOrder.push({ key: k, list: ghosts[k], off: total }); total += ghosts[k].length; });
        var transOff = total; total += trans.length;
        var extra = [];
        [frame.selected, frame.hover].forEach(function (idx, n) { if (idx !== null && idx !== undefined && list[idx]) extra.push({ it: list[idx], kind: n }); });
        var buf = new Float32Array((total + extra.length) * 16), w = 0;
        var put = function (x) { buf.set(x.mat, w); buf.set(x.col, w + 12); w += 16; };
        order.forEach(function (o) { o.list.forEach(put); }); ghostOrder.forEach(function (o) { o.list.forEach(put); }); trans.forEach(put);
        extra.forEach(function (x) { x.off = w / 16; put(x.it); });
        gl.bindBuffer(gl.ARRAY_BUFFER, instBuf); gl.bufferData(gl.ARRAY_BUFFER, buf, gl.DYNAMIC_DRAW);

        gl.useProgram(P);
        gl.uniformMatrix4fv(U.uVP, false, frame.vp); gl.uniform3fv(U.uEye, frame.eye); gl.uniform3fv(U.uLight, [0.42, -0.31, 0.85]);
        gl.uniform4f(U.uFlat, 0, 0, 0, 0); gl.uniform1f(U.uOutline, 0); gl.uniform1f(U.uOutK, 0); gl.uniform3f(U.uCen, 0, 0, 0); gl.uniform1f(U.uLit, 1); gl.uniform1i(U.uTex, 0);
        gl.uniform1f(U.uMode, 0); gl.uniform1f(U.uUseMat, 0); gl.uniform1f(U.uOpacity, 1); gl.uniform1f(U.uHasTex, 0); gl.uniform3f(U.uPartCol, 1, 1, 1);
        var draws = 0, tris = 0;
        if (terrain) {                                     // lit height field
          gl.uniform1f(U.uVCol, 1); bindMesh(terrain, true); constInst(identity, [1, 1, 1, 1]);
          gl.drawArrays(gl.TRIANGLES, 0, terrain.n); draws++; tris += terrain.n / 3;
        }
        gl.uniform1f(U.uVCol, 0);
        // ground / bounds lines
        if (frame.lines && frame.lines.length) {
          gl.useProgram(LP); gl.uniformMatrix4fv(LU.uVP, false, frame.vp);
          frame.lines.forEach(function (ln) {
            gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ARRAY_BUFFER, ln.pts, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); divisor(0, 0);
            for (var k = 1; k < 8; k++) gl.disableVertexAttribArray(k);
            gl.uniform4fv(LU.uC, ln.color); gl.drawArrays(gl.LINES, 0, ln.pts.length / 3);
          });
          gl.useProgram(P);
        }
        // part state: lane -> uniforms. With transparency off every lane draws as an opaque lit surface.
        var setPart = function (p, mode) {
          var lane = transparency || mode === 4 ? mode : 0;
          gl.uniform1f(U.uMode, lane); gl.uniform1f(U.uUseMat, useMat && p.c ? 1 : 0);
          if (p.c) gl.uniform3f(U.uPartCol, p.c[0], p.c[1], p.c[2]);
          gl.uniform1f(U.uOpacity, mode === 4 ? frame.ghostAlpha || 0.16 : (transparency ? p.o : 1));
          var tx = useMat && p.t >= 0 && textures[p.t] ? textures[p.t] : null;
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tx || whiteTex); gl.uniform1f(U.uHasTex, tx ? 1 : 0);
        };
        var LANE = { opaque: 0, cutout: 1, blend: 2, additive: 3 };
        // 1) opaque + cutout parts of every visible instance, instanced per mesh
        order.forEach(function (o) {
          var m = meshes[o.key] || meshes.cube; bindMesh(m, false); instAttribs(o.off * 64);
          m.parts.forEach(function (p) {
            var ln = LANE[p.lane] || 0;
            if (transparency && (ln === 2 || ln === 3)) return;
            setPart(p, transparency ? ln : 0); drawInst2(p.s * 3, p.n * 3, o.list.length); draws++; tris += p.n * o.list.length;
          });
        });
        // 2) filtered-out objects: a flat, desaturated, faded silhouette (not real transparency)
        if (ghostOrder.length) {
          gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
          ghostOrder.forEach(function (o) {
            var m = meshes[o.key] || meshes.cube; bindMesh(m, false); instAttribs(o.off * 64);
            setPart({ c: null, o: 1, t: -1 }, 4); drawInst2(0, m.n, o.list.length); draws++;
          });
          gl.depthMask(true); gl.disable(gl.BLEND);
        }
        // 3) translucent parts, instance by instance, far to near: depth test on, depth write off, premultiplied alpha / additive
        if (trans.length) {
          gl.enable(gl.BLEND); gl.depthMask(false);
          trans.forEach(function (t, n) {
            var m = meshes[t.key] || meshes.cube; bindMesh(m, false); instAttribs((transOff + n) * 64);
            m.parts.forEach(function (p) {
              var ln = LANE[p.lane] || 0; if (ln < 2) return;
              gl.blendFunc(gl.ONE, ln === 3 ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA);
              setPart(p, ln); drawInst2(p.s * 3, p.n * 3, 1); draws++;
            });
          });
          gl.depthMask(true); gl.disable(gl.BLEND);
        }
        gl.uniform1f(U.uMode, 0); gl.uniform1f(U.uUseMat, 0); gl.uniform1f(U.uHasTex, 0); gl.uniform1f(U.uOpacity, 1);
        // outlines (winding-independent): mark where the object itself is visible in the stencil, then
        // draw its normal-extruded shell in a flat colour wherever the stencil is NOT set
        if (extra.length) {
          gl.enable(gl.STENCIL_TEST); gl.depthMask(false);
          extra.forEach(function (x) {
            var c = x.kind === 0 ? frame.outline[0] : frame.outline[1], m = meshes[x.it.key] || meshes.cube;
            bindMesh(m, false); instAttribs(x.off * 64);
            gl.clear(gl.STENCIL_BUFFER_BIT);
            gl.stencilFunc(gl.ALWAYS, 1, 255); gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE); gl.colorMask(false, false, false, false);
            drawInst(m.n, 1);
            gl.colorMask(true, true, true, true); gl.stencilFunc(gl.NOTEQUAL, 1, 255); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
            var sc = Math.hypot(x.it.mat[0], x.it.mat[4], x.it.mat[8]) || 1;   // box fallback instances carry their size in the matrix
            gl.uniform3f(U.uCen, m.cen[0], m.cen[1], m.cen[2]); gl.uniform1f(U.uOutK, Math.max(0.03, frame.dist * (x.kind === 0 ? 0.006 : 0.004)) / (m.rad * sc));
            gl.uniform4f(U.uFlat, c[0], c[1], c[2], 1);
            drawInst(m.n, 1); draws += 2;
            gl.uniform4f(U.uFlat, 0, 0, 0, 0); gl.uniform1f(U.uOutK, 0);
          });
          gl.disable(gl.STENCIL_TEST); gl.depthMask(true);
        }
        stats.instances = list.length; stats.drawn = drawn; stats.culled = culled; stats.draws = draws; stats.tris = Math.round(tris);
        stats.ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      },
      finish: function () { gl.finish(); },
      readPixel: function (x, y) { var b = new Uint8Array(4); gl.readPixels(Math.round(x), canvas.height - 1 - Math.round(y), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, b); return Array.prototype.slice.call(b); },
      destroy: function () { Object.keys(meshes).forEach(function (k) { free(meshes[k]); }); free(terrain); textures.forEach(function (t) { gl.deleteTexture(t); }); }
    };
    return api;
  }

  var HmsGL = { create: create, frustum: frustum, sphereVisible: sphereVisible, version: '1.0.0' };
  if (typeof window !== 'undefined') window.HmsGL = HmsGL;
  else if (typeof globalThis !== 'undefined') globalThis.HmsGL = HmsGL;
  if (typeof module === 'object' && module && module.exports) module.exports = HmsGL;
})();
