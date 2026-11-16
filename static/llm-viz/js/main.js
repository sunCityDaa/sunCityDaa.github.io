// main.js — 把推理、布局、渲染、讲解和交互串起来。
import { loadModel, generateSorted, countParams, paramFormula } from './gpt.js';
import { genLayout, genSolidLayout, bindActs, KIND } from './layout.js';
import { Renderer, Camera, project, screenRay } from './renderer.js';
import { CHAPTERS } from './walkthrough.js';

const $ = (id) => document.getElementById(id);
const LETTERS = 'ABC';
const tokHtml = (i) => `<span class="tok t${LETTERS[i]}">${LETTERS[i]}</span>`;

const NANO_SHAPE = { B: 1, T: 11, C: 48, nHeads: 3, A: 16, nBlocks: 3, vocabSize: 3 };
const BIG_MODELS = [
  { name: 'GPT-2 (small)', key: 'gpt2s', shape: { T: 1024, C: 768, nHeads: 12, A: 64, nBlocks: 12, vocabSize: 50257 } },
  { name: 'GPT-2 (XL)', key: 'gpt2xl', shape: { T: 1024, C: 1600, nHeads: 25, A: 64, nBlocks: 48, vocabSize: 50257 } },
  { name: 'GPT-3', key: 'gpt3', shape: { T: 1024, C: 12288, nHeads: 96, A: 128, nBlocks: 96, vocabSize: 50257 } },
];

const app = {
  model: null,
  input: [2, 1, 0, 1, 1, 2], // C B A B B C
  gen: null,
  nano: null,
  bigLayouts: [],
  renderer: null,
  camera: new Camera(),
  chapter: 0,
  play: null, // { seq: [tensor], t0, msPer }
  dirty: true,
  labels: [],
};

// ---------------- 启动 ----------------
async function main() {
  const canvas = $('canvas');
  try {
    app.renderer = new Renderer(canvas);
  } catch (e) {
    $('loading').textContent = `无法初始化 WebGL2：${e.message}`;
    return;
  }
  const res = await fetch('data/gpt-nano-sort-model.json');
  app.model = loadModel(await res.json());

  app.nano = genLayout(NANO_SHAPE);
  // 大模型依次排在 nano-gpt 右侧，间距随模型尺寸增大
  let x = app.nano.bbox.maxX + 3000;
  for (const m of BIG_MODELS) {
    const probe = genSolidLayout(m.shape, 0);
    const L = genSolidLayout(m.shape, x - probe.bbox.minX);
    L.name = m.name; L.key = m.key;
    app.bigLayouts.push(L);
    x = L.bbox.maxX + L.bbox.width * 0.6;
  }
  app.nano.name = 'nano-gpt';

  runInference(false);
  app.renderer.build([app.nano, ...app.bigLayouts]);
  buildLabels();
  buildUi();
  setupControls(canvas);
  gotoChapter(0, true);
  $('loading').hidden = true;
  requestAnimationFrame(frame);

  // 供文档与自动化测试读取的调试接口
  window.__llmviz = { app, gotoChapter, setInput, CHAPTERS };
}

function runInference(refresh = true) {
  app.gen = generateSorted(app.model, app.input);
  bindActs(app.nano, app.model, app.gen.result.acts);
  if (refresh) app.renderer.refreshColors(app.nano);
  renderTokens();
  updateLabelsText();
  renderOutputPanel();
  app.dirty = true;
}

function setInput(arr) {
  app.input = arr.slice(0, 6);
  runInference();
}

// ---------------- 左侧面板 ----------------
function buildUi() {
  // 模型切换按钮
  const mb = $('model-buttons');
  const models = [{ name: 'nano-gpt', layout: app.nano }, ...app.bigLayouts.map((L) => ({ name: L.name, layout: L }))];
  for (const m of models) {
    const b = document.createElement('button');
    b.textContent = m.name;
    b.onclick = () => {
      for (const x of mb.children) x.classList.toggle('active', x === b);
      flyToBox(m.layout.bbox, { yaw: -0.25, pitch: 0.12 });
    };
    if (m.layout === app.nano) b.classList.add('active');
    mb.appendChild(b);
  }
  // 目录
  const toc = $('toc');
  CHAPTERS.forEach((c, i) => {
    const li = document.createElement('li');
    li.textContent = c.title;
    li.title = c.en;
    li.onclick = () => gotoChapter(i);
    toc.appendChild(li);
  });
  $('btn-prev').onclick = () => gotoChapter(app.chapter - 1);
  $('btn-next').onclick = () => gotoChapter(app.chapter + 1);
  $('btn-play').onclick = togglePlay;
  $('btn-reset').onclick = () => focusChapterCamera(CHAPTERS[app.chapter]);
  $('btn-overview').onclick = () => flyToBox(app.nano.bbox, { yaw: -0.35, pitch: 0.16 });
  $('btn-run').onclick = () => runInference();
  $('btn-random').onclick = () => setInput(Array.from({ length: 6 }, () => Math.floor(Math.random() * 3)));
  window.addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input, textarea')) return;
    if (e.key === 'ArrowRight') gotoChapter(app.chapter + 1);
    else if (e.key === 'ArrowLeft') gotoChapter(app.chapter - 1);
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });
}

function renderTokens() {
  const box = $('token-input');
  box.innerHTML = '';
  app.input.forEach((v, i) => {
    const s = document.createElement('span');
    s.className = `tok t${LETTERS[v]}`;
    s.textContent = LETTERS[v];
    s.onclick = () => { app.input[i] = (app.input[i] + 1) % 3; runInference(); };
    box.appendChild(s);
  });
  $('token-output').innerHTML = app.gen.output.map(tokHtml).join('');
  const sorted = app.input.slice().sort((a, b) => a - b);
  const ok = sorted.every((v, i) => v === app.gen.output[i]);
  const chk = $('sort-check');
  chk.textContent = ok ? '✓ 排序正确' : '✗ 与正确答案不同';
  chk.className = `check ${ok ? 'ok' : 'bad'}`;
}

function renderOutputPanel() {
  const el = document.getElementById('output-panel');
  if (!el || !app.gen) return;
  const { vizSeq, result } = app.gen;
  const V = 3, T = vizSeq.length;
  let rows = '<tr><th>位置</th>' + vizSeq.map((_, t) => `<th>${t}</th>`).join('') + '</tr>';
  rows += '<tr><th>当前</th>' + vizSeq.map((v) => `<td>${tokHtml(v)}</td>`).join('') + '</tr>';
  for (let v = 0; v < V; v++) {
    rows += `<tr><th>P(${LETTERS[v]})</th>` + Array.from({ length: T }, (_, t) => {
      const p = result.probs[t * V + v];
      return `<td title="${p.toFixed(4)}"><span class="bar" style="width:${Math.max(1, p * 22)}px;opacity:${0.25 + p * 0.75}"></span></td>`;
    }).join('') + '</tr>';
  }
  rows += '<tr><th>预测</th>' + Array.from({ length: T }, (_, t) => {
    let best = 0;
    for (let v = 1; v < V; v++) if (result.probs[t * V + v] > result.probs[t * V + best]) best = v;
    return `<td>${t >= 5 ? tokHtml(best) : '<span style="color:#aaa">·</span>'}</td>`;
  }).join('') + '</tr>';
  el.innerHTML = `<table>${rows}</table><p style="margin:6px 0 0;text-align:center;color:#667085">从第 5 列开始的预测才是答案（前面的列预测的是输入本身，模型并不关心）。</p>`;
}

// ---------------- 章节 ----------------
function gotoChapter(i, instant = false) {
  if (i < 0 || i >= CHAPTERS.length) return;
  app.chapter = i;
  stopPlay();
  const c = CHAPTERS[i];
  $('chapter-num').textContent = `${i + 1}/${CHAPTERS.length}`;
  $('chapter-name').textContent = `${c.title} · ${c.en}`;
  $('chapter-content').innerHTML = c.html;
  [...$('toc').children].forEach((li, k) => li.classList.toggle('active', k === i));
  $('btn-play').disabled = !c.sequence;
  $('play-label').textContent = c.sequence ? '点击播放' : '';
  $('progress-bar').style.width = '0';
  if (c.id === 'scale') {
    for (const m of BIG_MODELS) {
      const cell = document.getElementById(`p-${m.key}`);
      if (cell) cell.textContent = paramFormula(m.shape).toLocaleString('en-US');
    }
  }
  renderOutputPanel();
  applyFocus(c);
  focusChapterCamera(c, instant);
  $('chapter-content').parentElement.scrollTop = 0;
}

function applyFocus(c) {
  const r = app.renderer;
  for (const t of app.nano.tensors) {
    const f = c.focus(t);
    r.setTensorState(t.id, f ? 1 : 0.22, 1, f && !c.view ? 1 : 0);
  }
  r.uploadState();
  app.dirty = true;
}

function focusChapterCamera(c, instant = false) {
  let box;
  if (c.view === 'models') {
    const last = app.bigLayouts[app.bigLayouts.length - 1].bbox;
    box = { minX: app.nano.bbox.minX, maxX: last.maxX, minY: last.minY, maxY: 1 };
    // GPT-3 太高，整体看时以模型顶部为中心偏上
    box = { minX: box.minX, maxX: box.maxX, minY: -(box.maxX - box.minX) * 0.6, maxY: 1 };
    flyToBox(box, { yaw: -0.2, pitch: 0.1 }, instant);
    return;
  }
  const sel = c.camFocus ?? c.focus;
  const ts = app.nano.tensors.filter(sel);
  box = tensorsBox(ts.length ? ts : app.nano.tensors);
  flyToBox(box, { yaw: -0.3, pitch: 0.14 }, instant);
}

function tensorsBox(ts) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const t of ts) {
    minX = Math.min(minX, t.x); maxX = Math.max(maxX, t.x + t.cols);
    minY = Math.min(minY, t.y - t.rows); maxY = Math.max(maxY, t.y);
  }
  return { minX, maxX, minY, maxY };
}

function flyToBox(b, { yaw, pitch }, instant = false) {
  const cam = app.camera;
  const canvas = $('canvas');
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  // 顶部留出模型名卡片的空间
  b = { ...b, maxY: b.maxY + Math.max(6, (b.maxY - b.minY) * 0.06) };
  const w = b.maxX - b.minX, h = b.maxY - b.minY;
  const th = Math.tan(cam.fov / 2);
  const dist = Math.max(h / (2 * th), w / (2 * th * aspect)) * 1.12 + 10;
  const to = { target: [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2, 0], dist, yaw, pitch };
  if (instant) { Object.assign(cam, { target: to.target, dist, yaw, pitch }); cam.anim = null; }
  else cam.flyTo(to);
  app.dirty = true;
}

// ---------------- 动画播放：按计算顺序逐个"算出"张量 ----------------
function togglePlay() {
  if (app.play) { stopPlay(); return; }
  const c = CHAPTERS[app.chapter];
  if (!c.sequence) return;
  const seq = c.sequence.map((n) => app.nano.tensors.find((t) => t.name === n)).filter(Boolean);
  const r = app.renderer;
  for (const t of seq) if (t.kind !== KIND.WEIGHT) r.setTensorState(t.id, 1, 0, 1);
  r.uploadState();
  app.play = { seq, t0: performance.now(), msPer: 1600 };
  $('btn-play').textContent = '❚❚';
}

function stopPlay() {
  if (!app.play) return;
  const c = CHAPTERS[app.chapter];
  app.play = null;
  $('btn-play').textContent = '▶';
  applyFocus(c);
}

function stepPlay(now) {
  const p = app.play;
  if (!p) return;
  const r = app.renderer;
  const total = p.seq.length * p.msPer;
  const el = now - p.t0;
  const k = Math.min(p.seq.length - 1, Math.floor(el / p.msPer));
  p.seq.forEach((t, i) => {
    const prog = i < k ? 1 : i > k ? 0 : Math.min(1, (el - i * p.msPer) / (p.msPer * 0.85));
    if (t.kind === KIND.WEIGHT) r.setTensorState(t.id, 1, 1, i === k ? 1 : 0);
    else r.setTensorState(t.id, 1, prog, i === k ? 1 : 0.4);
  });
  r.uploadState();
  $('progress-bar').style.width = `${Math.min(100, (el / total) * 100)}%`;
  $('play-label').textContent = p.seq[k].label.replace(/^Block \d · /, '');
  app.dirty = true;
  if (el >= total) {
    app.play = null;
    $('btn-play').textContent = '▶';
    $('play-label').textContent = '完成';
  }
}

// ---------------- 3D 标签（DOM 叠加层） ----------------
function buildLabels() {
  const root = $('labels');
  const mk = (cls) => { const d = document.createElement('div'); d.className = `lab ${cls}`; root.appendChild(d); return d; };
  for (const t of app.nano.tensors) {
    // 小张量（偏置、γ、β、统计量）只在放大后显示
    const minor = t.cols === 1 || t.kind === KIND.AGG;
    // 画面上的标签去掉"Block N · "前缀（位置已经说明了属于哪个 block），完整名称留给悬停提示
    const short = t.label.replace(/^Block \d+ · /, '').replace(/^头 (\d+) · /, 'H$1 ');
    app.labels.push({ el: mk(''), kind: 'tensor', t, minor, text: short });
  }
  const tok = app.nano.tensors.find((t) => t.name === 'Tokens');
  const probs = app.nano.tensors.find((t) => t.name === 'Logits Softmax');
  for (let c = 0; c < 11; c++) {
    app.labels.push({ el: mk('cell'), kind: 'cell', t: tok, c, r: 0 });
    app.labels.push({ el: mk('cell out'), kind: 'out', t: probs, c });
  }
  for (const L of [app.nano, ...app.bigLayouts]) {
    app.labels.push({ el: mk('card'), kind: 'card', L });
  }
  // 每个 Transformer block 左侧一个大标题
  const names = { embed: '嵌入', output: '输出' };
  for (const b of app.nano.blocks) {
    const el = mk('block');
    el.textContent = names[b.name] ?? `Block ${b.name.replace('block', '')}`;
    app.labels.push({ el, kind: 'block', b });
  }
  updateLabelsText();
}

function updateLabelsText() {
  if (!app.gen) return;
  const { vizSeq, result } = app.gen;
  const p = countParams(app.model);
  for (const lb of app.labels) {
    if (lb.kind === 'tensor') lb.el.textContent = lb.text;
    else if (lb.kind === 'cell') lb.el.textContent = LETTERS[vizSeq[lb.c]];
    else if (lb.kind === 'out') {
      let best = 0;
      for (let v = 1; v < 3; v++) if (result.probs[lb.c * 3 + v] > result.probs[lb.c * 3 + best]) best = v;
      lb.el.textContent = LETTERS[best];
    } else if (lb.kind === 'card') {
      const n = lb.L === app.nano ? p.excludingLmHead : paramFormula(lb.L.shape);
      lb.el.innerHTML = `<b>${lb.L.name}</b>参数量 ${n.toLocaleString('en-US')}`;
    }
  }
}

function placeLabels(m) {
  const canvas = $('canvas');
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const th = Math.tan(app.camera.fov / 2);
  const chapter = CHAPTERS[app.chapter];
  for (const lb of app.labels) {
    let p, show = true;
    if (lb.kind === 'card') {
      const b = lb.L.bbox;
      p = project(m.viewProj, [(b.minX + b.maxX) / 2, b.maxY + Math.max(4, b.width * 0.04), 0], W, H);
    } else if (lb.kind === 'block') {
      p = project(m.viewProj, [app.nano.bbox.minX - 6, lb.b.top, 0], W, H);
    } else if (lb.kind === 'tensor') {
      p = project(m.viewProj, [lb.t.x, lb.t.y + 0.6, 0.5], W, H);
      // 标签最宽不超过张量本身在屏幕上的宽度（加一点间距），避免与右侧相邻张量的标签重叠
      const pr = p && project(m.viewProj, [lb.t.x + lb.t.cols + 8, lb.t.y + 0.6, 0.5], W, H);
      if (pr) lb.el.style.maxWidth = `${Math.max(48, pr[0] - p[0])}px`;
    } else {
      const y = lb.kind === 'out' ? lb.t.y - lb.t.rows - 1.4 : lb.t.y - 0.5;
      p = project(m.viewProj, [lb.t.x + lb.c + 0.5, y, 0.6], W, H);
    }
    if (!p) { lb.el.style.display = 'none'; continue; }
    const px = H / (2 * th * p[2]); // 每个世界单位对应的屏幕像素
    if (lb.kind === 'tensor') {
      const size = Math.max(lb.t.rows, lb.t.cols) * px;
      show = lb.minor ? px > 16 : size > 60 && px > 1.1;
      lb.el.classList.toggle('dim', !chapter.focus(lb.t));
    } else if (lb.kind === 'cell' || lb.kind === 'out') {
      show = px > 9;
      lb.el.style.fontSize = `${Math.min(18, px * 0.7)}px`;
    } else if (lb.kind === 'block') {
      show = px > 0.6;
      lb.el.style.fontSize = `${Math.max(12, Math.min(30, px * 7))}px`;
      lb.el.classList.toggle('dim', !app.nano.tensors.some((t) => t.group === lb.b.name && chapter.focus(t)));
    } else if (lb.kind === 'card') {
      const w = lb.L.bbox.width * px;
      show = w > 25 && p[2] > 0;
    }
    if (show && (p[0] < -200 || p[0] > W + 50 || p[1] < -50 || p[1] > H + 50)) show = false;
    lb.el.style.display = show ? '' : 'none';
    if (show) lb.el.style.transform = `translate(${p[0]}px, ${p[1]}px) ${lb.kind === 'tensor' ? 'translate(0,-100%)' : lb.kind === 'block' ? 'translate(-100%,0)' : lb.kind === 'card' ? 'translate(-50%,-100%)' : 'translate(-50%,-50%)'}`;
  }
}

// ---------------- 鼠标 / 触摸交互 ----------------
function setupControls(canvas) {
  const cam = app.camera;
  let drag = null;
  const pointers = new Map();
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey || e.button === 1 };
    canvas.classList.add('dragging');
    cam.anim = null;
  });
  canvas.addEventListener('pointerup', (e) => {
    pointers.delete(e.pointerId);
    if (!pointers.size) { drag = null; canvas.classList.remove('dragging'); }
  });
  canvas.addEventListener('pointercancel', (e) => { pointers.delete(e.pointerId); drag = null; });
  canvas.addEventListener('pointermove', (e) => {
    if (pointers.size === 2 && pointers.has(e.pointerId)) {
      // 双指：捏合缩放
      const [a, b] = [...pointers.values()];
      const d0 = Math.hypot(a[0] - b[0], a[1] - b[1]);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      const [a2, b2] = [...pointers.values()];
      const d1 = Math.hypot(a2[0] - b2[0], a2[1] - b2[1]);
      if (d0 > 0 && d1 > 0) cam.dist *= d0 / d1;
      app.dirty = true;
      return;
    }
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const s = (2 * cam.dist * Math.tan(cam.fov / 2)) / canvas.clientHeight;
        const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
        cam.target[0] -= dx * s * cy;
        cam.target[2] += dx * s * sy;
        cam.target[1] += dy * s;
      } else {
        cam.yaw -= dx * 0.005;
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.005));
      }
      app.dirty = true;
      hideTooltip();
      return;
    }
    hover(e);
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.anim = null;
    // 朝鼠标所指的方向缩放
    const rect = canvas.getBoundingClientRect();
    const hit = pickPlane(e.clientX - rect.left, e.clientY - rect.top);
    const f = Math.exp(e.deltaY * 0.0012);
    if (hit && f < 1) {
      for (let i = 0; i < 2; i++) cam.target[i] += (hit[i] - cam.target[i]) * (1 - f);
    }
    cam.dist = Math.max(3, cam.dist * f);
    app.dirty = true;
  }, { passive: false });
  canvas.addEventListener('pointerleave', hideTooltip);
  window.addEventListener('resize', () => { app.dirty = true; });
}

function pickPlane(sx, sy) {
  const canvas = $('canvas');
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const { o, d } = screenRay(app.camera, W / H, sx, sy, W, H);
  if (Math.abs(d[2]) < 1e-6) return null;
  const t = (0.43 - o[2]) / d[2];
  if (t <= 0) return null;
  return [o[0] + d[0] * t, o[1] + d[1] * t];
}

function hover(e) {
  const canvas = $('canvas');
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const hit = pickPlane(sx, sy);
  if (!hit) return hideTooltip();
  const [x, y] = hit;
  let found = null;
  for (const t of app.nano.tensors) {
    if (x >= t.x && x < t.x + t.cols && y <= t.y && y > t.y - t.rows) { found = t; break; }
  }
  if (!found) {
    for (const L of app.bigLayouts) for (const t of L.tensors) {
      if (x >= t.x && x < t.x + t.cols && y <= t.y && y > t.y - t.rows) { found = t; break; }
    }
  }
  const tip = $('tooltip');
  if (!found) return hideTooltip();
  let html = `<b>${found.label}</b><br>形状 ${found.cols} × ${found.rows}`;
  if (found.get) {
    const c = Math.floor(x - found.x), r = Math.floor(found.y - y);
    const v = found.get(r, c);
    const vs = Number.isFinite(v) ? v.toFixed(4) : '−∞（因果掩码）';
    html += `<br>行 ${r}，列 ${c}：<span class="v">${found.kind === KIND.TOKEN ? `${LETTERS[v]}（id=${v}）` : vs}</span>`;
    if (found.kind === KIND.ACT || found.kind === KIND.TOKEN) {
      const ch = LETTERS[app.gen.vizSeq[c]];
      if (/Attention Matrix|Attn Matrix/.test(found.name)) html += `<br>第 ${r} 个 token（${LETTERS[app.gen.vizSeq[r]]}）对第 ${c} 个 token（${ch}）`;
      else if (ch) html += `<br>token 位置 ${c}（${ch}）`;
    }
  }
  tip.innerHTML = html;
  tip.hidden = false;
  const tw = tip.offsetWidth;
  tip.style.left = `${Math.min(sx + 14, canvas.clientWidth - tw - 6)}px`;
  tip.style.top = `${sy + 14}px`;
}

function hideTooltip() { $('tooltip').hidden = true; }

// ---------------- 渲染循环 ----------------
function frame(now) {
  if (app.camera.update(now)) app.dirty = true;
  stepPlay(now);
  if (app.dirty) {
    app.dirty = false;
    const m = app.renderer.draw(app.camera);
    placeLabels(m);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  $('loading').hidden = false;
  $('loading').textContent = `加载失败：${e.message}`;
});
