// main.js — 加载导出数据 → 搭场景 → 渲染；章节、时间轴、词选择、悬停拾取。
import { Camera, project, screenRay, rayBox } from './gl-util.js';
import { Renderer } from './renderer.js';
import { buildScene, scalarData, nearestTxt, STATIC_COLORS, PAL } from './scene.js';
import { CHAPTERS } from './walkthrough.js';

const $ = (id) => document.getElementById(id);
const MODELS = [
  { key: 'qwen', name: 'Qwen-Image-2.1' },
  { key: 'flux', name: 'FLUX.3 Klein' },
];

const app = {
  key: new URLSearchParams(location.search).get('model') === 'flux' ? 'flux' : 'qwen',
  meta: null, scene: null, renderer: null, camera: new Camera(),
  atlas: [], x0: [], xt: [], attnStrips: {}, cache: {},
  step: 0, chapter: 0, word: 2, mode: 'act', play: null, labels: [], dirty: true, other: null, capSel: 2,
};

// ---------------- 资源加载
function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('加载失败：' + src));
    im.src = src;
  });
}
const cvs = document.createElement('canvas');
const ctx2d = cvs.getContext('2d', { willReadFrequently: true });
function redChannel(im) {
  cvs.width = im.width; cvs.height = im.height;
  ctx2d.clearRect(0, 0, im.width, im.height);
  ctx2d.drawImage(im, 0, 0);
  const d = ctx2d.getImageData(0, 0, im.width, im.height).data;
  const out = new Uint8Array(im.width * im.height);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
  return { w: im.width, h: im.height, data: out };
}

async function loadModel(key, onProgress) {
  if (app.cache[key]) return app.cache[key];
  const base = `data/${key}/`;
  const meta = await (await fetch(base + 'meta.json')).json();
  const N = meta.steps.length;
  const p2 = (i) => String(i).padStart(2, '0');
  let done = 0;
  const total = N * 3 + Object.keys(meta.attn).length + 1;
  const tick = () => onProgress && onProgress(++done / total);
  const atlas = new Array(N), x0 = new Array(N), xt = new Array(N), attn = {};
  const jobs = [];
  for (let i = 0; i < N; i++) {
    jobs.push(loadImage(`${base}act/act_${p2(i)}.png`).then((im) => { atlas[i] = redChannel(im); tick(); }));
    jobs.push(loadImage(`${base}steps/x0_${p2(i)}.jpg`).then((im) => { x0[i] = im; tick(); }));
    jobs.push(loadImage(`${base}steps/xt_${p2(i)}.jpg`).then((im) => { xt[i] = im; tick(); }));
  }
  for (const [k, v] of Object.entries(meta.attn)) {
    jobs.push(loadImage(base + v.file).then((im) => { attn[k] = redChannel(im); tick(); }));
  }
  const final = loadImage(base + 'final.jpg').then((im) => { tick(); return im; });
  await Promise.all(jobs);
  const m = { meta, atlas, x0, xt, attn, final: await final };
  app.cache[key] = m;
  return m;
}

// ---------------- 启动
async function main() {
  try {
    app.renderer = new Renderer($('canvas'));
  } catch (e) {
    $('loading').textContent = `无法初始化 WebGL2：${e.message}`;
    return;
  }
  buildModelButtons();
  setupControls($('canvas'));
  await switchModel(app.key, true);
  requestAnimationFrame(frame);
}

async function switchModel(key, first = false) {
  app.key = key;
  history.replaceState(null, '', key === 'qwen' ? location.pathname : `?model=${key}`);
  document.querySelectorAll('#model-buttons button').forEach((b) => b.classList.toggle('active', b.dataset.key === key));
  $('loading').hidden = false;
  $('loading').textContent = '正在加载导出数据…';
  let m;
  try {
    m = await loadModel(key, (f) => { $('loading').textContent = `正在加载导出数据… ${Math.round(f * 100)}%`; });
  } catch (e) {
    $('loading').textContent = e.message;
    return;
  }
  app.meta = m.meta; app.atlas = m.atlas; app.x0 = m.x0; app.xt = m.xt; app.attnStrips = m.attn;
  app.scene = buildScene(app.meta);
  app.scene.meta = app.meta;
  const ip = app.scene.imgPlane;
  app.scene.gImg = { id: -1, bbox: { minX: ip.x, maxX: ip.x + ip.size, minY: 0, maxY: ip.size, minZ: ip.z - 2, maxZ: ip.z + 2 } };
  const jn = jointNames();
  app.scene.capGroups = app.meta.cap_blocks.map((i) => app.scene.blockGroups.find((g) => g.name === jn[i]));
  app.renderer.build(app.scene, STATIC_COLORS);
  app.word = Math.min(app.word, app.meta.words.length - 1);
  $('prompt').textContent = app.meta.prompt;
  $('final-img').src = m.final.src;
  $('step-range').max = app.meta.steps.length - 1;
  buildLabels();
  buildToc();
  $('loading').hidden = true;
  app.step = Math.min(app.step, app.meta.steps.length - 1);
  gotoChapter(app.chapter, true);
  // 后台加载另一个模型的 meta，供「对比」一章使用
  const ok = MODELS.find((x) => x.key !== key).key;
  if (!app.cache[ok]) fetch(`data/${ok}/meta.json`).then((r) => r.json()).then((mm) => { app.other = mm; if (CHAPTERS[app.chapter].id === 'compare') renderChapter(); });
  else app.other = app.cache[ok].meta;
}

function jointNames() {
  return app.meta.blocks.filter((b) => b.kind === 'joint').map((b) => b.name);
}

// ---------------- 时间步
function setStep(i, { snap = true } = {}) {
  const N = app.meta.steps.length;
  i = Math.max(0, Math.min(N - 1, i));
  if (app.mode === 'attn' && snap) {
    i = app.meta.cap_steps.reduce((a, b) => (Math.abs(b - i) < Math.abs(a - i) ? b : a));
  }
  app.step = i;
  uploadStep();
  const st = app.meta.steps[i];
  $('step-range').value = i;
  $('step-label').textContent = `步 ${i}/${N - 1} · t=${st.t.toFixed(3)}`;
  $('x0-img').src = app.x0[i].src;
  $('xt-img').src = app.xt[i].src;
  if (['time', 'denoise', 'stream', 'attn'].includes(CHAPTERS[app.chapter].id)) renderChapter(true);
  app.dirty = true;
}

function uploadStep() {
  const r = app.renderer, i = app.step;
  let a = app.atlas[i];
  if (app.mode === 'attn') {
    // 用选中词的注意力热图替换被采样的那几个 block 的 tile
    const data = a.data.slice();
    const G = app.meta.grid, cols = app.meta.tile_cols, W = a.w;
    const jn = jointNames();
    for (const bi of app.meta.cap_blocks) {
      const strip = app.attnStrips[`${i}/${bi}`];
      if (!strip) continue;
      const tile = app.meta.tiles.indexOf(jn[bi]);
      const tc = tile % cols, tr = Math.floor(tile / cols);
      for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
        data[(tr * G + y) * W + tc * G + x] = strip.data[y * strip.w + app.word * G + x];
      }
    }
    a = { w: a.w, h: a.h, data };
  }
  app.curAtlas = a;
  r.uploadAtlas(a.w, a.h, a.data);
  app.curScalar = scalarData(app.scene, app.meta, i);
  r.uploadScalar(app.scene.SW, app.scene.scalarRows.length, app.curScalar);
  r.uploadImage(app.x0[i]);
}

// ---------------- 章节
function buildToc() {
  $('toc').innerHTML = CHAPTERS.map((c, i) => `<li data-i="${i}">${i + 1}. ${c.title}</li>`).join('');
  $('toc').querySelectorAll('li').forEach((li) => li.addEventListener('click', () => gotoChapter(+li.dataset.i)));
}

function gotoChapter(i, instant = false) {
  app.chapter = Math.max(0, Math.min(CHAPTERS.length - 1, i));
  const ch = CHAPTERS[app.chapter];
  app.mode = ch.mode || 'act';
  const s = app.scene, r = app.renderer;
  r.resetState();
  const focus = ch.focus(s);
  if (focus) {
    const ids = new Set(focus.map((g) => g.id));
    s.groups.forEach((g) => r.setGroup(g.id, { bright: ids.has(g.id) ? 1 : 0.18 }));
  }
  if (app.mode === 'attn') for (const g of s.capGroups) r.setGroup(g.id, { pal: PAL.ATTN, hscale: 1.8 });
  r.imgDim = !focus || focus.includes(s.gVae) || focus.includes(s.gOut) ? 1 : 0.35;
  let step = app.step;
  if (ch.step === 0) step = 0;
  else if (ch.step === -1) step = app.meta.steps.length - 1;
  setStep(step);
  renderChapter();
  resetView(instant);
  document.querySelectorAll('#toc li').forEach((li, k) => li.classList.toggle('active', k === app.chapter));
  $('chapter-num').textContent = `${app.chapter + 1}/${CHAPTERS.length}`;
  $('chapter-name').textContent = ch.title;
  $('btn-prev').disabled = app.chapter === 0;
  $('btn-next').disabled = app.chapter === CHAPTERS.length - 1;
  updateLabelDim();
}

function renderChapter(keepScroll = false) {
  const ch = CHAPTERS[app.chapter];
  const el = $('chapter-content');
  const top = el.parentElement.scrollTop;
  el.innerHTML = ch.html(app.meta, { step: app.step, word: app.word, other: app.other, capSel: app.capSel });
  if (keepScroll) el.parentElement.scrollTop = top;
  el.querySelectorAll('.wbtn').forEach((b) => b.addEventListener('click', () => {
    app.word = +b.dataset.i;
    uploadStep();
    renderChapter(true);
    app.dirty = true;
  }));
  el.querySelectorAll('.cbtn').forEach((b) => b.addEventListener('click', () => {
    app.capSel = +b.dataset.i;
    renderChapter(true);
    resetView();
  }));
  el.querySelectorAll('.sbtn').forEach((b) => b.addEventListener('click', () => { stopPlay(); setStep(+b.dataset.s); }));
}

function unionBox(groups) {
  const b = { minX: Infinity, minY: 0, minZ: Infinity, maxX: -Infinity, maxY: 1, maxZ: -Infinity };
  for (const g of groups) {
    b.minX = Math.min(b.minX, g.bbox.minX); b.maxX = Math.max(b.maxX, g.bbox.maxX);
    b.minZ = Math.min(b.minZ, g.bbox.minZ); b.maxZ = Math.max(b.maxZ, g.bbox.maxZ);
    b.maxY = Math.max(b.maxY, g.bbox.maxY);
  }
  return b;
}

function resetView(instant = false) {
  const ch = CHAPTERS[app.chapter];
  app.scene.capSel = app.capSel;
  const cam = ch.cam(app.scene);
  const c = $('canvas');
  const aspect = c.clientWidth / Math.max(1, c.clientHeight);
  const box = cam ? unionBox(cam) : app.scene.bbox;
  const pitch = ch.pitch ?? (cam ? 0.95 : 0.9);
  const yaw = ch.yaw ?? (cam ? -0.2 : -0.12);
  const v = app.camera.frame(box, { yaw, pitch, pad: ch.pad ?? (cam ? 1.1 : 1.0), aspect });
  if (instant) Object.assign(app.camera, { target: v.target, yaw: v.yaw, pitch: v.pitch, dist: v.dist, anim: null });
  else app.camera.flyTo(v);
  app.dirty = true;
}

function overview() {
  const c = $('canvas');
  app.camera.flyTo(app.camera.frame(app.scene.bbox, { yaw: -0.12, pitch: 0.82, pad: 1.0, aspect: c.clientWidth / c.clientHeight }));
}

// ---------------- 标签
function buildLabels() {
  const box = $('labels');
  box.innerHTML = '';
  app.labels = [];
  const add = (g, text, cls, pos) => {
    const el = document.createElement('div');
    el.className = 'lab ' + cls;
    el.innerHTML = text;
    box.appendChild(el);
    app.labels.push({ el, pos, g, cls });
  };
  const s = app.scene;
  for (const g of s.groups) {
    const b = g.bbox;
    if (g.kind === 'joint' || g.kind === 'imgblock' || g.kind === 'txtblock') {
      add(g, g.title, 'blk', [b.minX, 0, b.minZ - 1]);
    } else if (g.kind === 'unused') {
      add(g, g.title, 'stage dim2', [b.minX, 0, b.minZ - 1]);
    } else if (g.kind === 'arch') {
      add(g, g.title, 'stage', [g.name === 'arch_flow' ? b.minX + 60 : (b.minX + b.maxX) / 2, 0, b.minZ - (g.name === 'arch_flow' ? 50 : 14)]);
    } else if (g.kind === 'prompt') {
      add(g, g.title, 'stage blk', [b.maxX + 2, 0, b.maxZ]);
    } else {
      add(g, g.title, 'stage', [b.minX, 0, b.minZ - 1.5]);
    }
  }
  // 架构区：每个矩阵一块标签（名字 + 形状/参数量/作用）
  for (const g of s.groups) {
    if (!g.subLabels) continue;
    for (const t of g.subLabels) {
      add(g, `${t.text}<span class="sub">${t.sub}</span>`, 'plate', t.pos);
    }
  }
  add(null, '最终图像（当前步 x₀ 的 VAE 解码）', 'stage', [s.imgPlane.x, s.imgPlane.size + 2, s.imgPlane.z]);
}

function updateLabelDim() {
  const ch = CHAPTERS[app.chapter];
  const focus = ch.focus(app.scene);
  const ids = focus ? new Set(focus.map((g) => g.id)) : null;
  const hideOthers = !!ch.only;
  for (const l of app.labels) {
    const mine = !!(ids && l.g && ids.has(l.g.id));
    l.el.classList.toggle('dim', !!(ids && l.g && !mine));
    l.hidden = hideOthers && !mine && !!l.g;
  }
}

function placeLabels(vp, w, h) {
  const far = app.camera.dist > 1100;
  for (const l of app.labels) {
    if (l.hidden) { l.el.style.display = 'none'; continue; }
    const p = project(vp, l.pos, w, h);
    const cls = l.cls.split(' ');
    const hide = !p || p[0] < -200 || p[0] > w + 50 || p[1] < -20 || p[1] > h + 20 || (far && (cls.includes('blk') || cls.includes('plate')));
    l.el.style.display = hide ? 'none' : '';
    if (!hide) l.el.style.transform = `translate(${p[0].toFixed(1)}px, ${p[1].toFixed(1)}px) translate(0, -100%)`;
  }
}

// ---------------- 渲染循环
function frame(now) {
  const moving = app.camera.update(now);
  if (app.play) {
    const N = app.meta.steps.length;
    const k = Math.floor((now - app.play.t0) / app.play.ms);
    if (k !== app.play.k) {
      app.play.k = k;
      if (k >= N) { stopPlay(); } else setStep(k, { snap: false });
    }
  }
  if (moving || app.dirty) {
    const c = $('canvas');
    const w = c.clientWidth, h = c.clientHeight;
    const { viewProj } = app.camera.matrices(w / Math.max(1, h));
    app.vp = viewProj;
    app.renderer.draw(viewProj);
    placeLabels(viewProj, w, h);
    app.dirty = false;
  }
  requestAnimationFrame(frame);
}

function startPlay() {
  const N = app.meta.steps.length;
  if (app.mode === 'attn') gotoChapter(CHAPTERS.findIndex((c) => c.id === 'denoise'));
  app.play = { t0: performance.now() - (app.step >= N - 1 ? 0 : app.step) * 0, ms: Math.max(120, 6000 / N), k: -1 };
  app.play.t0 = performance.now() - (app.step >= N - 1 ? 0 : app.step) * app.play.ms;
  $('btn-play').textContent = '❚❚';
}
function stopPlay() {
  app.play = null;
  $('btn-play').textContent = '▶';
}

// ---------------- 交互
function buildModelButtons() {
  $('model-buttons').innerHTML = MODELS.map((m) => `<button data-key="${m.key}">${m.name}</button>`).join('');
  $('model-buttons').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    if (b.dataset.key !== app.key) { stopPlay(); switchModel(b.dataset.key); }
  }));
}

function setupControls(canvas) {
  $('btn-prev').addEventListener('click', () => gotoChapter(app.chapter - 1));
  $('btn-next').addEventListener('click', () => gotoChapter(app.chapter + 1));
  $('btn-reset').addEventListener('click', () => resetView());
  $('btn-overview').addEventListener('click', overview);
  $('btn-play').addEventListener('click', () => (app.play ? stopPlay() : startPlay()));
  $('step-range').addEventListener('input', (e) => { stopPlay(); setStep(+e.target.value); });
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') gotoChapter(app.chapter + 1);
    else if (e.key === 'ArrowLeft') gotoChapter(app.chapter - 1);
    else if (e.key === ' ') { e.preventDefault(); app.play ? stopPlay() : startPlay(); }
    else if (e.key === '[') { stopPlay(); setStep(app.step - 1); }
    else if (e.key === ']') { stopPlay(); setStep(app.step + 1); }
  });
  window.addEventListener('resize', () => { app.dirty = true; });

  let drag = null;
  const pointers = new Map();
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    app.camera.anim = null;
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (drag && drag.pinch) app.camera.dist *= drag.pinch / d;
      drag = { ...drag, pinch: d };
      app.dirty = true;
      return;
    }
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      const cam = app.camera;
      if (drag.pan) {
        // 在地面上平移：右向量 (cos yaw, 0, −sin yaw)，前向量（离开相机）(−sin yaw, 0, −cos yaw)
        const s = cam.dist * 0.0016;
        const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
        cam.target[0] += (-dx * cy - dy * sy) * s;
        cam.target[2] += (dx * sy - dy * cy) * s;
      } else {
        cam.yaw -= dx * 0.005;
        cam.pitch = Math.max(0.08, Math.min(1.5, cam.pitch + dy * 0.004));
      }
      app.dirty = true;
      $('tooltip').hidden = true;
    } else hover(e);
  });
  const end = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) { drag = null; canvas.classList.remove('dragging'); }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const cam = app.camera;
    cam.anim = null;
    const f = Math.exp(e.deltaY * 0.0012);
    // 朝鼠标方向缩放：把 target 往鼠标射线与地面的交点挪
    const r = canvas.getBoundingClientRect();
    const ray = screenRay(cam, r.width / r.height, e.clientX - r.left, e.clientY - r.top, r.width, r.height);
    if (ray.d[1] < -0.01) {
      const t = -ray.o[1] / ray.d[1];
      const hit = [ray.o[0] + ray.d[0] * t, 0, ray.o[2] + ray.d[2] * t];
      const k = 1 - f;
      for (const i of [0, 2]) cam.target[i] += (hit[i] - cam.target[i]) * k;
    }
    cam.dist = Math.max(15, Math.min(6000, cam.dist * f));
    app.dirty = true;
  }, { passive: false });
}

// ---------------- 悬停拾取：沿射线步进，检查落在哪根柱子里
function cellHeight(g, kind, c, r) {
  if (kind === 'grid') {
    const { tile, G, hmax } = g.grid;
    const cols = app.meta.tile_cols, tc = tile % cols, tr = Math.floor(tile / cols);
    const v = app.curAtlas.data[(tr * G + r) * app.curAtlas.w + tc * G + c] / 255;
    const stt = app.renderer.state;
    const ex = stt[g.id * 4 + 2] === PAL.ATTN ? 0.8 : 1.4;
    return { h: 0.15 + hmax * Math.pow(v, ex) * (stt[g.id * 4 + 3] || 1) * (stt[g.id * 4] < 0.5 ? 0.25 : 1), v };
  }
  return null;
}

function hover(e) {
  const tip = $('tooltip');
  const canvas = $('canvas');
  const rc = canvas.getBoundingClientRect();
  const sx = e.clientX - rc.left, sy = e.clientY - rc.top;
  const ray = screenRay(app.camera, rc.width / rc.height, sx, sy, rc.width, rc.height);
  let best = null;
  for (const g of app.scene.groups) {
    const hit = rayBox(ray, g.bbox);
    if (!hit || (best && hit[0] > best.t)) continue;
    const r = pickInGroup(g, ray, hit);
    if (r && (!best || r.t < best.t)) best = r;
  }
  if (!best) { tip.hidden = true; return; }
  tip.innerHTML = best.html;
  tip.hidden = false;
  const tx = Math.min(sx + 14, rc.width - 330), ty = Math.min(sy + 14, rc.height - 120);
  tip.style.left = tx + 'px'; tip.style.top = ty + 'px';
}

function pickInGroup(g, ray, [t0, t1]) {
  const m = app.meta, st = m.steps[app.step];
  const stepLen = 0.35;
  for (let t = t0; t <= t1; t += stepLen) {
    const p = [ray.o[0] + ray.d[0] * t, ray.o[1] + ray.d[1] * t, ray.o[2] + ray.d[2] * t];
    if (g.plates) {
      for (const pl of g.plates) {
        if (p[0] >= pl.x && p[0] <= pl.x + pl.w && p[2] >= pl.z && p[2] <= pl.z + pl.d && p[1] <= 1.8) {
          const mm = pl.m;
          return { t, html: `<b>${mm.name}</b><br>${mm.shape.join(' × ')} = <span class="v">${(mm.params / 1e6).toFixed(1)}M</span> 参数<br>${mm.desc || ''}` };
        }
      }
      return null;
    }
    if (g.grid) {
      const c = Math.floor(p[0] - g.grid.x0), r = Math.floor(p[2] - g.grid.z0);
      if (c >= 0 && r >= 0 && c < g.grid.G && r < g.grid.G) {
        const { h, v } = cellHeight(g, 'grid', c, r);
        if (p[1] <= h) {
          const rng = st.tile_range[g.name];
          const isAttn = app.mode === 'attn' && app.scene.capGroups.includes(g);
          const val = isAttn ? `「${m.words[app.word].text}」注意力（相对）${(v * 100).toFixed(0)}%`
            : (rng ? `RMS ≈ <span class="v">${(rng[0] + v * (rng[1] - rng[0])).toFixed(3)}</span>` : '');
          return { t, html: `<b>${g.title}</b> · ${g.name}<br>图像 token (${r}, ${c}) · 对应像素 ${c * 16}–${c * 16 + 15}, ${r * 16}–${r * 16 + 15}<br>${val}<br>${blockInfo(g)}` };
        }
      }
    }
    if (g.txtRow !== undefined) {
      const zz = g.txtZ ?? 8;
      if (p[2] >= zz && p[2] <= zz + 1.2) {
        const i = Math.floor((p[0] - g.txtX0) / g.txtW);
        if (i >= 0 && i < m.tokens.length) {
          const v = app.curScalar[g.txtRow * app.scene.SW + i];
          if (p[1] <= 0.1 + (g.kind === 'txtblock' ? 8 : 7) * v) {
            const raw = nearestTxt(m, g.name, app.step);
            return { t, html: `<b>${g.title}</b> · 文本 token #${i} <code>${esc(m.tokens[i])}</code><br>RMS <span class="v">${raw ? raw[i].toFixed(2) : '?'}</span>（取自最近导出的第 ${nearestStep(g.name)} 步）` };
          }
        }
      }
    }
    if (g.headRow !== undefined && p[2] >= g.headZ && p[2] <= g.headZ + 1.2) {
      const h = Math.floor((p[0] - g.headX0) / g.headW);
      if (h >= 0 && h < m.heads) {
        const v = app.curScalar[g.headRow * app.scene.SW + h];
        if (p[1] <= 0.1 + 18 * v) {
          return { t, html: `<b>${g.title}</b> · 第 ${h} 个注意力头<br>图像 query 分给文本 key 的注意力 <span class="v">${(v * 100).toFixed(1)}%</span><br>本层平均 ${(st.txt_frac[g.name] * 100).toFixed(1)}%` };
        }
      }
    }
    if (g.kind === 'te') {
      const l = Math.floor(p[2] / 1.6), i = Math.floor((p[0] - g.x0) / g.tokW);
      if (l >= 0 && l < g.layers && i >= 0 && i < m.tokens.length) {
        const v = app.curScalar[(g.teRow0 + l) * app.scene.SW + i];
        if (p[1] <= 0.1 + 6 * v) {
          return { t, html: `<b>文本编码器</b> 第 ${l} 层 · token #${i} <code>${esc(m.tokens[i])}</code><br>RMS <span class="v">${m.te_layer_rms[l][i].toFixed(2)}</span>` };
        }
      }
    }
    if (g.kind === 'prompt' && p[2] >= -12 && p[2] <= -11 && p[1] <= 2.4) {
      const i = Math.floor((p[0] - g.x0) / g.tokW);
      if (i >= 0 && i < m.tokens.length) return { t, html: `提示词 token #${i}：<code>${esc(m.tokens[i])}</code>` };
    }
    if (['vae', 'time', 'io', 'unused'].includes(g.kind) && p[1] <= g.bbox.maxY) {
      return { t, html: stageInfo(g) };
    }
  }
  return null;
}

function nearestStep(name) {
  return app.meta.txt_token_steps.reduce((a, b) => (Math.abs(b - app.step) < Math.abs(a - app.step) ? b : a));
}
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const fmt = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n.toLocaleString());

function blockInfo(g) {
  const b = g.block;
  if (!b) return '';
  const sh = b.shape && b.shape.out ? JSON.stringify(b.shape.out).replace(/,/g, ', ') : '';
  const st = app.meta.steps[app.step];
  const fr = st.txt_frac[g.name];
  return `参数 ${fmt(b.params)} · 输出形状 ${sh}${fr !== undefined ? `<br>注意文本比例 ${(fr * 100).toFixed(1)}%` : ''}`;
}

function stageInfo(g) {
  const m = app.meta;
  if (g.kind === 'vae') return `<b>${g.title}</b><br>参数 ${fmt(m.components.vae.params)}`;
  if (g.kind === 'unused') return `<b>${g.title}</b><br>3 × 5 个分流块里的第三组，参数 ${fmt(g.params)}；只有输入参考图时才运行`;
  if (g.kind === 'time') {
    const st = m.steps[app.step];
    const io = m.io[g.name] || {};
    return `<b>${g.title}</b><br>当前 t = <span class="v">${st.t.toFixed(4)}</span>${st.target_t !== undefined ? `，目标 t = ${st.target_t.toFixed(4)}` : ''}<br>${io.params ? '参数 ' + fmt(io.params) : ''}${io.shape ? ' · 输出 ' + JSON.stringify(io.shape.out) : ''}<br>柱子 = 16 个采样点上的时间表`;
  }
  const io = m.io[g.name];
  return `<b>${g.title}</b>${io ? `<br>参数 ${fmt(io.params)} · 输出 ${JSON.stringify(io.shape.out)}` : ''}`;
}

// 调试接口：控制台可用 __imgviz.app / gotoChapter(i) / setStep(i)
window.__imgviz = { app, gotoChapter, setStep, CHAPTERS };
main();
