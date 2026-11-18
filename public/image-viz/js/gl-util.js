// gl-util.js — 矩阵、轨道相机、投影/拾取射线（与 llm-viz 同一套思路，独立实现以免互相依赖）。

export const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  lookAt(eye, c, up) {
    let zx = eye[0] - c[0], zy = eye[1] - c[1], zz = eye[2] - c[2];
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

/** 轨道相机：围绕 target 旋转，yaw 水平角、pitch 俯仰角、dist 距离。 */
export class Camera {
  constructor() {
    this.target = [0, 0, 0];
    this.yaw = -0.5;
    this.pitch = 0.6;
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
    const proj = mat4.perspective(this.fov, aspect, Math.max(0.1, this.dist * 0.01), this.dist * 20 + 2000);
    return { eye, view, proj, viewProj: mat4.mul(proj, view) };
  }
  flyTo(to, ms = 900) {
    const from = { target: this.target.slice(), yaw: this.yaw, pitch: this.pitch, dist: this.dist };
    let dy = (to.yaw ?? from.yaw) - from.yaw;
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    this.anim = {
      from, t0: performance.now(), ms,
      to: { target: to.target ?? from.target, yaw: from.yaw + dy, pitch: to.pitch ?? from.pitch, dist: to.dist ?? from.dist },
    };
  }
  update(now) {
    if (!this.anim) return false;
    const { from, to, t0, ms } = this.anim;
    const t = Math.min(1, (now - t0) / ms);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    this.dist = Math.exp(Math.log(from.dist) + (Math.log(to.dist) - Math.log(from.dist)) * e);
    for (let i = 0; i < 3; i++) this.target[i] = from.target[i] + (to.target[i] - from.target[i]) * e;
    this.yaw = from.yaw + (to.yaw - from.yaw) * e;
    this.pitch = from.pitch + (to.pitch - from.pitch) * e;
    if (t >= 1) this.anim = null;
    return true;
  }
  /** 计算能完整看到包围盒的相机参数：分别按水平/竖直视角拟合投影后的宽和深。 */
  frame(b, { yaw = this.yaw, pitch = this.pitch, pad = 1.1, aspect = 1.6 } = {}) {
    const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2, cz = (b.minZ + b.maxZ) / 2;
    const sx = b.maxX - b.minX, sz = b.maxZ - b.minZ, sy = b.maxY - b.minY;
    const cyw = Math.abs(Math.cos(yaw)), syw = Math.abs(Math.sin(yaw));
    const across = sx * cyw + sz * syw;                       // 屏幕水平方向的跨度
    const along = (sx * syw + sz * cyw) * Math.sin(pitch) + sy * Math.cos(pitch); // 屏幕竖直方向的跨度
    const tv = Math.tan(this.fov / 2), th = tv * aspect;
    const dist = pad * Math.max(across / 2 / th, along / 2 / tv) + Math.max(sx, sz) * 0.08;
    return { target: [cx, cy, cz], yaw, pitch, dist: Math.max(dist, 30) };
  }
}

/** 世界坐标 → 屏幕 CSS 像素；在相机后方返回 null。 */
export function project(vp, p, w, h) {
  const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
  const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
  const wc = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (wc <= 0) return null;
  return [(x / wc * 0.5 + 0.5) * w, (1 - (y / wc * 0.5 + 0.5)) * h, wc];
}

/** 屏幕坐标 → 世界空间射线。 */
export function screenRay(cam, aspect, sx, sy, w, h) {
  const eye = cam.eye();
  const f = [cam.target[0] - eye[0], cam.target[1] - eye[1], cam.target[2] - eye[2]];
  const fl = Math.hypot(...f); f[0] /= fl; f[1] /= fl; f[2] /= fl;
  const r = [-f[2], 0, f[0]];
  const rl = Math.hypot(...r); r[0] /= rl; r[2] /= rl;
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const th = Math.tan(cam.fov / 2);
  const nx = (sx / w) * 2 - 1, ny = 1 - (sy / h) * 2;
  const d = [0, 1, 2].map((i) => f[i] + r[i] * nx * th * aspect + u[i] * ny * th);
  const dl = Math.hypot(...d);
  return { o: eye, d: d.map((v) => v / dl) };
}

/** 射线与轴对齐包围盒求交，返回 [tNear, tFar] 或 null。 */
export function rayBox(ray, b) {
  let t0 = 0, t1 = Infinity;
  const mn = [b.minX, b.minY, b.minZ], mx = [b.maxX, b.maxY, b.maxZ];
  for (let i = 0; i < 3; i++) {
    const inv = 1 / ray.d[i];
    let a = (mn[i] - ray.o[i]) * inv, c = (mx[i] - ray.o[i]) * inv;
    if (a > c) [a, c] = [c, a];
    t0 = Math.max(t0, a); t1 = Math.min(t1, c);
    if (t0 > t1) return null;
  }
  return [t0, t1];
}
