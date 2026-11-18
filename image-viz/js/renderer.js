// renderer.js — WebGL2：实例化立方体（每个 token/标量一根柱）+ 一张图像平面。
// 每步只需替换两张纹理（激活图集 R8、标量 R32F），实例数据只上传一次。

const VS = `#version 300 es
precision highp float;
precision highp sampler2D;
layout(location=0) in vec3 a_vert;
layout(location=1) in vec3 a_norm;
layout(location=2) in vec4 a_pos;    // x y z size
layout(location=3) in vec4 a_src;    // kind u v palette
layout(location=4) in vec4 a_extra;  // group hmax h0 staticColor
uniform mat4 u_vp;
uniform sampler2D u_atlas;           // R8：当前步每层图像 token 的 RMS（已逐 tile 归一化）
uniform sampler2D u_scalar;          // R32F：当前步的标量
uniform sampler2D u_state;           // RGBA32F：每组 r=亮度 g=高亮 b=调色板覆盖(-1 不覆盖) a=高度缩放
uniform vec3 u_static[9];
out vec3 v_col;
out vec3 v_n;

vec3 ramp4(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
  return t < 0.33 ? mix(a, b, t / 0.33) : (t < 0.66 ? mix(b, c, (t - 0.33) / 0.33) : mix(c, d, (t - 0.66) / 0.34));
}
// 浅底上更易读：低值浅、高值深
vec3 palAct(float t) { return ramp4(vec3(0.86, 0.90, 0.95), vec3(0.47, 0.75, 0.80), vec3(0.16, 0.47, 0.66), vec3(0.15, 0.13, 0.42), t); }
vec3 palAttn(float t) { return ramp4(vec3(0.97, 0.90, 0.92), vec3(0.95, 0.56, 0.62), vec3(0.80, 0.18, 0.45), vec3(0.36, 0.05, 0.36), t); }
vec3 palTxt(float t) { return ramp4(vec3(0.90, 0.90, 0.92), vec3(0.97, 0.78, 0.55), vec3(0.93, 0.48, 0.20), vec3(0.62, 0.20, 0.06), t); }

void main() {
  int g = int(a_extra.x + 0.5);
  vec4 st = texelFetch(u_state, ivec2(g % 64, g / 64), 0);
  float val = 1.0;
  int kind = int(a_src.x + 0.5);
  if (kind == 1) val = texelFetch(u_atlas, ivec2(int(a_src.y), int(a_src.z)), 0).r;
  else if (kind == 2) val = clamp(texelFetch(u_scalar, ivec2(int(a_src.y), int(a_src.z)), 0).r, 0.0, 1.0);
  int pal = st.b >= 0.0 ? int(st.b + 0.5) : int(a_src.w + 0.5);
  vec3 col;
  if (kind == 0) col = u_static[int(a_extra.w + 0.5)];
  else if (pal == 1) col = palAttn(val);
  else if (pal == 2) col = palTxt(val);
  else col = palAct(val);
  col = mix(vec3(0.94, 0.95, 0.97), col, st.r);
  col = mix(col, min(col * 1.15 + 0.06, 1.0), st.g);
  float hv = kind == 0 ? 1.0 : (kind == 1 ? pow(val, pal == 1 ? 0.8 : 1.4) : val);
  float h = a_extra.z + a_extra.y * hv * st.a * (st.r < 0.5 ? 0.25 : 1.0);
  vec3 p = vec3(a_pos.x, a_pos.y, a_pos.z) + a_vert * vec3(a_pos.w, max(h, 0.02), a_pos.w);
  gl_Position = u_vp * vec4(p, 1.0);
  v_col = col;
  v_n = a_norm;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 v_col; in vec3 v_n; out vec4 o;
void main() {
  vec3 L = normalize(vec3(0.4, 0.85, 0.55));
  float d = max(dot(normalize(v_n), L), 0.0);
  o = vec4(v_col * (0.58 + 0.48 * d), 1.0);
}`;

const QVS = `#version 300 es
layout(location=0) in vec3 a_p; layout(location=1) in vec2 a_uv;
uniform mat4 u_vp; out vec2 v_uv;
void main() { gl_Position = u_vp * vec4(a_p, 1.0); v_uv = a_uv; }`;
const QFS = `#version 300 es
precision highp float;
in vec2 v_uv; uniform sampler2D u_img; uniform float u_dim; out vec4 o;
void main() { vec3 c = texture(u_img, v_uv).rgb; o = vec4(mix(vec3(0.93), c, u_dim), 1.0); }`;

function compile(gl, vs, fs) {
  const mk = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function cube() {
  // 单位立方体 [0,1]^3（底面在 y=0），带法线
  const F = [
    [[0, 1, 0], [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]],
    [[0, 0, 1], [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]],
    [[0, 0, -1], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]],
    [[1, 0, 0], [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]],
    [[-1, 0, 0], [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]],
    [[0, -1, 0], [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]]],
  ];
  const v = [];
  // 面的顶点按从外侧看逆时针排列（配合 CULL_FACE 剔除背面）
  for (const [n, q] of F) for (const i of [0, 2, 1, 0, 3, 2]) v.push(...q[i], ...n);
  return new Float32Array(v);
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('浏览器不支持 WebGL2');
    this.gl = gl;
    this.prog = compile(gl, VS, FS);
    this.qprog = compile(gl, QVS, QFS);
    this.u = {};
    for (const n of ['u_vp', 'u_atlas', 'u_scalar', 'u_state', 'u_static']) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.qu = { vp: gl.getUniformLocation(this.qprog, 'u_vp'), img: gl.getUniformLocation(this.qprog, 'u_img'), dim: gl.getUniformLocation(this.qprog, 'u_dim') };
    this.cubeBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cube(), gl.STATIC_DRAW);
    this.vao = null;
    this.tex = { atlas: gl.createTexture(), scalar: gl.createTexture(), state: gl.createTexture(), img: gl.createTexture() };
    for (const t of Object.values(this.tex)) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.qbuf = gl.createBuffer();
    this.qvao = gl.createVertexArray();
    this.imgDim = 1;
  }

  build(scene, staticColors) {
    const gl = this.gl;
    this.scene = scene;
    this.staticColors = new Float32Array(staticColors.flat());
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.instBuf) gl.deleteBuffer(this.instBuf);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    this.instBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, scene.data, gl.STATIC_DRAW);
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(2 + i);
      gl.vertexAttribPointer(2 + i, 4, gl.FLOAT, false, 48, i * 16);
      gl.vertexAttribDivisor(2 + i, 1);
    }
    gl.bindVertexArray(null);
    // 图像平面（竖立在输出端）
    const { x, z, size } = scene.imgPlane;
    const q = new Float32Array([
      x, size, z, 0, 0, x + size, size, z, 1, 0, x + size, 0, z, 1, 1,
      x, size, z, 0, 0, x + size, 0, z, 1, 1, x, 0, z, 0, 1,
    ]);
    gl.bindVertexArray(this.qvao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.qbuf);
    gl.bufferData(gl.ARRAY_BUFFER, q, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    gl.bindVertexArray(null);
    this.state = new Float32Array(64 * 64 * 4);
    this.resetState();
  }

  resetState() {
    for (let i = 0; i < this.scene.groups.length; i++) this.state.set([1, 0, -1, 1], i * 4);
    this.stateDirty = true;
  }
  setGroup(id, { bright, hi, pal, hscale } = {}) {
    const o = id * 4;
    if (bright !== undefined) this.state[o] = bright;
    if (hi !== undefined) this.state[o + 1] = hi;
    if (pal !== undefined) this.state[o + 2] = pal;
    if (hscale !== undefined) this.state[o + 3] = hscale;
    this.stateDirty = true;
  }

  uploadAtlas(w, h, bytes) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.atlas);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
  }
  uploadScalar(w, h, floats) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.scalar);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, floats);
  }
  uploadImage(img) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.img);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  draw(vp) {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.969, 0.973, 0.984, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    if (this.stateDirty) {
      gl.bindTexture(gl.TEXTURE_2D, this.tex.state);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 64, 64, 0, gl.RGBA, gl.FLOAT, this.state);
      this.stateDirty = false;
    }
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.u.u_vp, false, vp);
    gl.uniform3fv(this.u.u_static, this.staticColors);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex.atlas); gl.uniform1i(this.u.u_atlas, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.tex.scalar); gl.uniform1i(this.u.u_scalar, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.tex.state); gl.uniform1i(this.u.u_state, 2);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.scene.count);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(this.qprog);
    gl.uniformMatrix4fv(this.qu.vp, false, vp);
    gl.uniform1f(this.qu.dim, this.imgDim);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.tex.img); gl.uniform1i(this.qu.img, 3);
    gl.bindVertexArray(this.qvao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }
}
