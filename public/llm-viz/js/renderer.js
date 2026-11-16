// renderer.js — WebGL2 实例化渲染：每个格子是一个小立方体（一个实例）。
// 每个张量的"状态"（变暗 / 揭示进度 / 高亮）写在一张 RGBA32F 纹理里，
// 顶点着色器用 texelFetch(张量 id) 读取，这样切换讲解章节时不必重建实例数据。

import { KIND } from './layout.js';

// ---------------- 小型矩阵工具 ----------------
export const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  lookAt(eye, center, up) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1,
    ]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  },
};

// ---------------- 轨道相机 ----------------
// 围绕 target 旋转：yaw（水平角）、pitch（俯仰角）、dist（距离）。
export class Camera {
  constructor() {
    this.target = [0, -200, 0];
    this.yaw = -0.35;
    this.pitch = 0.18;
    this.dist = 600;
    this.fov = 40 * Math.PI / 180;
    this.anim = null;
  }
  eye() {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.sin(this.yaw),
      this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * cp * Math.cos(this.yaw),
    ];
  }
  matrices(aspect) {
    const eye = this.eye();
    const view = mat4.lookAt(eye, this.target, [0, 1, 0]);
    const proj = mat4.perspective(this.fov, aspect, Math.max(0.05, this.dist * 0.002), this.dist * 50 + 1000);
    return { eye, view, proj, viewProj: mat4.mul(proj, view) };
  }
  /** 平滑飞行到新视角（带缓动）。 */
  flyTo(to, ms = 900) {
    const from = { target: this.target.slice(), yaw: this.yaw, pitch: this.pitch, dist: this.dist };
    // 让 yaw 走最短路径
    let dy = (to.yaw ?? from.yaw) - from.yaw;
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    this.anim = { from, to: { target: to.target ?? from.target, yaw: from.yaw + dy, pitch: to.pitch ?? from.pitch, dist: to.dist ?? from.dist }, t0: performance.now(), ms };
  }
  /** 推进动画；返回 true 表示仍在运动。 */
  update(now) {
    if (!this.anim) return false;
    const { from, to, t0, ms } = this.anim;
    let t = Math.min(1, (now - t0) / ms);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    // 距离按对数插值，远近飞行时速度感更自然
    this.dist = Math.exp(Math.log(from.dist) + (Math.log(to.dist) - Math.log(from.dist)) * e);
    for (let i = 0; i < 3; i++) this.target[i] = from.target[i] + (to.target[i] - from.target[i]) * e;
    this.yaw = from.yaw + (to.yaw - from.yaw) * e;
    this.pitch = from.pitch + (to.pitch - from.pitch) * e;
    if (t >= 1) this.anim = null;
    return true;
  }
}

// ---------------- 着色器 ----------------
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pos;      // 单位立方体顶点 [0,1]^3
layout(location=1) in vec3 a_normal;
layout(location=2) in vec3 a_offset;   // 实例：格子左上前角
layout(location=3) in vec3 a_scale;    // 实例：尺寸
layout(location=4) in vec3 a_color;    // 实例：由数值映射得到的颜色
layout(location=5) in vec2 a_meta;     // 实例：张量 id、在张量内的计算顺序 0..1
uniform mat4 u_viewProj;
uniform sampler2D u_state;             // 每个张量一个 texel：r=亮度 g=揭示进度 b=高亮
uniform int u_stateWidth;
out vec3 v_color;
out vec3 v_normal;
out float v_alpha;
void main() {
  float bright = 1.0, reveal = 1.0, hi = 0.0;
  int id = int(a_meta.x + 0.5);
  if (a_meta.x >= 0.0) {
    vec4 s = texelFetch(u_state, ivec2(id % u_stateWidth, id / u_stateWidth), 0);
    bright = s.r; reveal = s.g; hi = s.b;
  }
  // 尚未"计算到"的格子：缩成扁平的浅色占位
  bool shown = a_meta.y <= reveal + 1e-5;
  vec3 scale = a_scale;
  vec3 col = a_color;
  if (!shown) { scale.z *= 0.15; col = vec3(0.86, 0.87, 0.9); }
  // 正在计算的格子（揭示前沿）加亮
  float front = 1.0 - smoothstep(0.0, 0.04, abs(a_meta.y - reveal));
  col = mix(col, vec3(1.0, 0.85, 0.2), front * step(reveal, 0.999) * 0.8);
  col = mix(vec3(0.93, 0.94, 0.96), col, bright);
  col = mix(col, col * 1.08 + vec3(0.04), hi);
  vec3 p = a_offset + a_pos * scale;
  gl_Position = u_viewProj * vec4(p, 1.0);
  v_color = col;
  v_normal = a_normal;
  v_alpha = 1.0;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 v_color;
in vec3 v_normal;
in float v_alpha;
out vec4 outColor;
void main() {
  vec3 L = normalize(vec3(0.35, 0.6, 0.75));
  float d = max(dot(normalize(v_normal), L), 0.0);
  vec3 c = v_color * (0.62 + 0.45 * d);
  outColor = vec4(c, v_alpha);
}`;

// 线框（张量外框）
const LINE_VS = `#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec4 a_col;
uniform mat4 u_viewProj;
out vec4 v_col;
void main() { gl_Position = u_viewProj * vec4(a_pos, 1.0); v_col = a_col; }`;
const LINE_FS = `#version 300 es
precision highp float;
in vec4 v_col; out vec4 o;
void main() { o = v_col; }`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// 单位立方体：24 个顶点（每面 4 个，带法线），36 个索引
function cubeGeometry() {
  const faces = [
    [[0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
    [[0, 0, -1], [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]]],
    [[1, 0, 0], [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]],
    [[-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]],
    [[0, 1, 0], [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]]],
    [[0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
  ];
  const v = [], idx = [];
  faces.forEach(([n, quad], f) => {
    for (const p of quad) v.push(...p, ...n);
    const b = f * 4;
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  });
  return { verts: new Float32Array(v), idx: new Uint16Array(idx) };
}

// ---------------- 数值 → 颜色 ----------------
// 发散色板：负值 → 橙红，0 → 中性色，正值 → 蓝色。中性色按张量类型区分。
const NEUTRAL = {
  [KIND.WEIGHT]: [0.74, 0.77, 0.83],
  [KIND.ACT]: [0.78, 0.88, 0.78],
  [KIND.AGG]: [0.84, 0.78, 0.9],
  [KIND.TOKEN]: [0.85, 0.85, 0.85],
};
const POS = [0.12, 0.42, 0.95];
const NEG = [0.96, 0.42, 0.12];
const TOKEN_COLORS = [[0.36, 0.72, 0.36], [0.27, 0.55, 0.85], [0.85, 0.45, 0.3]]; // A B C

export function valueColor(v, scale, kind, out, o) {
  const n = NEUTRAL[kind];
  if (!Number.isFinite(v)) { out[o] = 0.55; out[o + 1] = 0.56; out[o + 2] = 0.6; return; } // 被掩码的 -inf
  let t = scale > 0 ? v / scale : 0;
  t = Math.max(-1, Math.min(1, t));
  const target = t >= 0 ? POS : NEG;
  // 权重数量多，用线性映射避免画面过花；激活值用 0.7 次幂突出差异
  const a = Math.pow(Math.abs(t), kind === KIND.WEIGHT ? 1.15 : 0.7);
  out[o] = n[0] + (target[0] - n[0]) * a;
  out[o + 1] = n[1] + (target[1] - n[1]) * a;
  out[o + 2] = n[2] + (target[2] - n[2]) * a;
}

/** 计算张量的颜色刻度：使用绝对值的 98 分位数，避免个别极值让整体颜色变淡。 */
function tensorScale(t) {
  if (!t.get) return 1;
  const vals = [];
  const step = Math.max(1, Math.floor((t.rows * t.cols) / 4000));
  let k = 0;
  for (let r = 0; r < t.rows; r++) for (let c = 0; c < t.cols; c++) {
    if (k++ % step) continue;
    const v = t.get(r, c);
    if (Number.isFinite(v)) vals.push(Math.abs(v));
  }
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.98))] || 1;
}

// ---------------- 渲染器 ----------------
const STRIDE = 14; // offset3 scale3 color3 meta2 + 3 填充（保持 56 字节对齐）
const CELL = 0.86; // 格子边长（留出缝隙）

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    this.gl = gl;
    this.canvas = canvas;
    this.prog = compile(gl, VS, FS);
    this.lineProg = compile(gl, LINE_VS, LINE_FS);
    this.uViewProj = gl.getUniformLocation(this.prog, 'u_viewProj');
    this.uState = gl.getUniformLocation(this.prog, 'u_state');
    this.uStateWidth = gl.getUniformLocation(this.prog, 'u_stateWidth');
    this.uLineViewProj = gl.getUniformLocation(this.lineProg, 'u_viewProj');

    const cube = cubeGeometry();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, cube.verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube.idx, gl.STATIC_DRAW);
    this.instBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    const B = STRIDE * 4;
    const attr = (loc, size, off) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, B, off * 4);
      gl.vertexAttribDivisor(loc, 1);
    };
    attr(2, 3, 0); attr(3, 3, 3); attr(4, 3, 6); attr(5, 2, 9);
    gl.bindVertexArray(null);

    this.lineVao = gl.createVertexArray();
    gl.bindVertexArray(this.lineVao);
    this.lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);

    this.stateTex = gl.createTexture();
    this.stateWidth = 256;
    this.instanceCount = 0;
    this.lineCount = 0;
    this.nTensors = 0;
    this.state = new Float32Array(0);
  }

  /** 根据所有布局（nano-gpt 逐格 + 大模型实心盒）生成实例数据。 */
  build(layouts) {
    let count = 0;
    for (const L of layouts) for (const t of L.tensors) count += t.solid ? 1 : t.rows * t.cols;
    const data = new Float32Array(count * STRIDE);
    const lines = [];
    let o = 0;
    let maxId = 0;
    for (const L of layouts) {
      for (const t of L.tensors) {
        maxId = Math.max(maxId, t.id + 1);
        if (t.solid) {
          // 一个实心盒子代表整个张量
          const d = t.depth ?? 1;
          data.set([t.x, t.y - t.rows, -d / 2, t.cols, t.rows, d], o);
          const n = NEUTRAL[t.kind];
          data[o + 6] = n[0] * 0.9; data[o + 7] = n[1] * 0.9; data[o + 8] = n[2] * 0.9;
          data[o + 9] = -1; data[o + 10] = 1;
          o += STRIDE;
          continue;
        }
        t.instanceStart = o / STRIDE;
        t.scale = tensorScale(t);
        const total = t.rows * t.cols;
        let k = 0;
        // 按"列（token）优先"决定计算顺序：动画时一列一列地算出来
        for (let c = 0; c < t.cols; c++) {
          for (let r = 0; r < t.rows; r++) {
            data[o] = t.x + c + (1 - CELL) / 2;
            data[o + 1] = t.y - r - 1 + (1 - CELL) / 2;
            data[o + 2] = -CELL / 2;
            data[o + 3] = CELL; data[o + 4] = CELL; data[o + 5] = CELL;
            this.cellColor(t, r, c, data, o + 6);
            data[o + 9] = t.id;
            data[o + 10] = total > 1 ? (k + 1) / total : 1;
            o += STRIDE;
            k++;
          }
        }
        // 外框
        pushRect(lines, t.x - 0.25, t.y + 0.25, t.cols + 0.5, t.rows + 0.5, 0.5, [0.45, 0.5, 0.6, 1]);
      }
    }
    for (const L of layouts) {
      if (!L.solid) continue;
      for (const t of L.tensors) pushRect(lines, t.x, t.y, t.cols, t.rows, (t.depth ?? 1) / 2 + 0.01, [0.35, 0.4, 0.5, 1]);
    }
    this.data = data;
    this.instanceCount = count;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    const ld = new Float32Array(lines);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, ld, gl.STATIC_DRAW);
    this.lineCount = ld.length / 7;
    this.nTensors = maxId;
    this.state = new Float32Array(Math.max(1, Math.ceil(maxId / this.stateWidth)) * this.stateWidth * 4);
    for (let i = 0; i < maxId; i++) this.state.set([1, 1, 0, 0], i * 4);
    this.uploadState();
  }

  cellColor(t, r, c, out, o) {
    if (t.kind === KIND.TOKEN && t.get) {
      const tc = TOKEN_COLORS[Math.round(t.get(r, c))] ?? [0.8, 0.8, 0.8];
      out[o] = tc[0]; out[o + 1] = tc[1]; out[o + 2] = tc[2];
      return;
    }
    valueColor(t.get ? t.get(r, c) : 0, t.scale, t.kind, out, o);
  }

  /** 重新推理后只需刷新颜色（位置不变）。 */
  refreshColors(layout) {
    for (const t of layout.tensors) {
      if (t.solid || t.instanceStart === undefined) continue;
      t.scale = tensorScale(t);
      let o = t.instanceStart * STRIDE;
      for (let c = 0; c < t.cols; c++) for (let r = 0; r < t.rows; r++) { this.cellColor(t, r, c, this.data, o + 6); o += STRIDE; }
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.data);
  }

  setTensorState(id, bright, reveal, hi) {
    const i = id * 4;
    this.state[i] = bright; this.state[i + 1] = reveal; this.state[i + 2] = hi;
  }

  uploadState() {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.stateTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.stateWidth, this.state.length / 4 / this.stateWidth, 0, gl.RGBA, gl.FLOAT, this.state);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr), h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  draw(camera) {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.97, 0.975, 0.985, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    const m = camera.matrices(this.canvas.width / Math.max(1, this.canvas.height));
    this.lastMatrices = m;

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uViewProj, false, m.viewProj);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTex);
    gl.uniform1i(this.uState, 0);
    gl.uniform1i(this.uStateWidth, this.stateWidth);
    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0, this.instanceCount);

    gl.useProgram(this.lineProg);
    gl.uniformMatrix4fv(this.uLineViewProj, false, m.viewProj);
    gl.bindVertexArray(this.lineVao);
    gl.drawArrays(gl.LINES, 0, this.lineCount);
    gl.bindVertexArray(null);
    return m;
  }
}

function pushRect(out, x, y, w, h, z, col) {
  const p = [[x, y], [x + w, y], [x + w, y - h], [x, y - h]];
  for (let i = 0; i < 4; i++) {
    const a = p[i], b = p[(i + 1) % 4];
    out.push(a[0], a[1], z, ...col, b[0], b[1], z, ...col);
  }
}

/** 把世界坐标投影到屏幕（CSS 像素）。返回 null 表示在相机后方。 */
export function project(viewProj, p, w, h) {
  const x = viewProj[0] * p[0] + viewProj[4] * p[1] + viewProj[8] * p[2] + viewProj[12];
  const y = viewProj[1] * p[0] + viewProj[5] * p[1] + viewProj[9] * p[2] + viewProj[13];
  const wc = viewProj[3] * p[0] + viewProj[7] * p[1] + viewProj[11] * p[2] + viewProj[15];
  if (wc <= 0) return null;
  return [(x / wc * 0.5 + 0.5) * w, (1 - (y / wc * 0.5 + 0.5)) * h, wc];
}

/** 屏幕坐标 → 世界空间射线（用于鼠标拾取）。 */
export function screenRay(camera, aspect, sx, sy, w, h) {
  const eye = camera.eye();
  const f = [camera.target[0] - eye[0], camera.target[1] - eye[1], camera.target[2] - eye[2]];
  const fl = Math.hypot(...f); f[0] /= fl; f[1] /= fl; f[2] /= fl;
  // 右向量 r = f × up，其中 up = (0, 1, 0)
  const r = [-f[2], 0, f[0]];
  const rl = Math.hypot(...r); r[0] /= rl; r[2] /= rl;
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const th = Math.tan(camera.fov / 2);
  const nx = (sx / w) * 2 - 1, ny = 1 - (sy / h) * 2;
  const d = [0, 1, 2].map((i) => f[i] + r[i] * nx * th * aspect + u[i] * ny * th);
  const dl = Math.hypot(...d);
  return { o: eye, d: d.map((v) => v / dl) };
}
