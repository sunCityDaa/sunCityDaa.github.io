// scene.js — 把 meta.json 变成一组「实例分组」（group）。每个实例是一个小立方体。
//
// 一个实例 = 3 个 vec4：
//   a_pos  : x, y, z, 尺寸 s（立方体底面边长）
//   a_src  : 来源 kind（0=静态色 1=图集 2=标量纹理）, u, v, 调色板（0=激活 1=注意力 2=文本 3=权重 4=灰）
//   a_extra: 组 id, 最大高度 hmax, 基础高度 h0, 静态色索引
//
// 图集（每步一张 R8 纹理）：tile i 位于 (i % cols, i / cols)，每个 tile G×G 像素。
// 标量纹理（每步一张 R32F，宽 SW）：每一行放一组标量（文本 token RMS、各头注意文本比例、文本编码器各层…）。

export const PAL = { ACT: 0, ATTN: 1, TXT: 2, WEIGHT: 3, GREY: 4, IMG: 5 };
export const SRC = { STATIC: 0, ATLAS: 1, SCALAR: 2 };

// 静态颜色表（与着色器中的 u_static 对应）
export const STATIC_COLORS = [
  [0.80, 0.83, 0.90], // 0 中性
  [0.93, 0.62, 0.33], // 1 文本 token（内容词）
  [0.72, 0.76, 0.84], // 2 文本 token（模板/特殊）
  [0.55, 0.65, 0.90], // 3 q/k/v/o 权重
  [0.62, 0.80, 0.62], // 4 MLP 权重
  [0.85, 0.72, 0.90], // 5 调制/时间
  [0.95, 0.80, 0.45], // 6 VAE
  [0.70, 0.70, 0.74], // 7 未使用（image_cond）
];

const SPECIAL = /^<\|.*\|>$|^\\n$|^(user|assistant|system)$/;

export function buildScene(meta) {
  const G = meta.grid;
  const isQwen = meta.model === 'qwen';
  const inst = [];
  const groups = [];
  const labels = [];
  const scalarRows = [{ name: 'time', len: 16 }];   // 第 0 行固定为时间表
  const tileIndex = Object.fromEntries(meta.tiles.map((n, i) => [n, i]));
  const nTxt = meta.tokens.length;
  const H = meta.heads;

  const addGroup = (g) => {
    g.id = groups.length;
    g.bbox = { minX: Infinity, minY: 0, minZ: Infinity, maxX: -Infinity, maxY: 1, maxZ: -Infinity };
    groups.push(g);
    return g;
  };
  const push = (g, x, y, z, s, src, u, v, pal, hmax, h0, sc = 0) => {
    inst.push(x, y, z, s, src, u, v, pal, g.id, hmax, h0, sc);
    const b = g.bbox;
    b.minX = Math.min(b.minX, x); b.maxX = Math.max(b.maxX, x + s);
    b.minZ = Math.min(b.minZ, z); b.maxZ = Math.max(b.maxZ, z + s);
    b.maxY = Math.max(b.maxY, y + h0 + hmax);
    b.minY = Math.min(b.minY, y);
  };
  const addScalarRow = (name, len) => { scalarRows.push({ name, len }); return scalarRows.length - 1; };

  // 64×64 图像 token 网格：每个 token 一根柱子，高度/颜色来自当前步的图集
  const gridGroup = (g, x0, z0, tile, hmax = 10) => {
    const tc = tile % meta.tile_cols, tr = Math.floor(tile / meta.tile_cols);
    for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
      push(g, x0 + c, 0, z0 + r, 0.92, SRC.ATLAS, tc * G + c, tr * G + r, PAL.ACT, hmax, 0.15);
    }
    g.grid = { x0, z0, tile, G, hmax };
  };

  // ---------------- 布局常量
  const PITCH_X = 100, PITCH_Z = 120, PER_ROW = 8;
  const blocks = meta.blocks.filter((b) => b.kind !== 'txt' && b.kind !== 'unused');
  const txtBlocks = meta.blocks.filter((b) => b.kind === 'txt');
  const unused = meta.blocks.filter((b) => b.kind === 'unused');
  const nRows = Math.ceil(blocks.length / PER_ROW);
  const slot = (i) => [Math.floor(i % PER_ROW) * PITCH_X, Math.floor(i / PER_ROW) * PITCH_Z];
  const Z_OFF = txtBlocks.length ? 70 : 0;   // FLUX：文本分流块单独占一行

  // ---------------- 1. 提示词 token 与文本编码器（左侧）
  const teX = -PITCH_X * 3.1;
  const tokW = Math.min(1, 80 / nTxt);
  const teLayers = meta.te_layer_rms.length;
  const gTok = addGroup({ name: 'prompt', kind: 'prompt', title: '提示词 token', stage: 'text' });
  meta.tokens.forEach((t, i) => {
    push(gTok, teX + i * tokW, 0, -12, tokW * 0.9, SRC.STATIC, 0, 0, PAL.GREY, 2.2, 0.2, SPECIAL.test(t.trim()) ? 2 : 1);
  });
  gTok.tokW = tokW; gTok.x0 = teX;
  const gTE = addGroup({ name: 'text_encoder', kind: 'te', title: `文本编码器 · ${meta.components.text_encoder.layers} 层`, stage: 'text' });
  const teRow0 = scalarRows.length;
  for (let l = 0; l < teLayers; l++) {
    const row = addScalarRow(`te.${l}`, nTxt);
    for (let i = 0; i < nTxt; i++) push(gTE, teX + i * tokW, 0, l * 1.6, tokW * 0.9, SRC.SCALAR, i, row, PAL.TXT, 1.6, 0.1);
  }
  gTE.teRow0 = teRow0; gTE.tokW = tokW; gTE.x0 = teX; gTE.layers = teLayers;

  // 文本投影（txt_in / context_embedder）
  const gTxtIn = addGroup({ name: isQwen ? 'txt_in' : 'context_embedder', kind: 'io', title: '文本投影', stage: 'text' });
  for (let i = 0; i < nTxt; i++) push(gTxtIn, teX + i * tokW, 0, teLayers * 1.6 + 8, tokW * 0.9, SRC.STATIC, 0, 0, PAL.GREY, 1.5, 0.1, 5);

  // ---------------- 2. 时间步嵌入与调制（输入网格上方一个小块）
  const gTime = addGroup({ name: isQwen ? 'modulation' : 'timestep_embedder', kind: 'time', title: isQwen ? '时间嵌入 + 共享调制' : '时间嵌入（当前 t + 目标 t）', stage: 'time' });
  for (let i = 0; i < 16; i++) push(gTime, -PITCH_X + 18 + i * 4, 0, Z_OFF - 26, 3, SRC.SCALAR, i, 0, PAL.ATTN, 14, 0.5);

  // ---------------- 3. 输入潜变量 x_t
  const gIn = addGroup({ name: 'x_t', kind: 'latent', title: '输入 x_t', stage: 'latent' });
  gridGroup(gIn, -PITCH_X + 18, Z_OFF + 18, tileIndex.x_t, 6);

  // ---------------- 4. Transformer block
  const blockGroups = [];
  if (txtBlocks.length) {
    txtBlocks.forEach((b, i) => {
      const g = addGroup({ name: b.name, kind: 'txtblock', title: `文本分流 ${i}`, block: b, stage: 'blocks', idx: i });
      const x0 = i * PITCH_X + 18;
      const row = scalarRows.length; scalarRows.push({ name: `${b.name}.txt`, len: nTxt });
      for (let t = 0; t < nTxt; t++) push(g, x0 + t * tokW * 0.8, 0, 8, tokW * 0.72, SRC.SCALAR, t, row, PAL.TXT, 8, 0.1);
      g.txtRow = row; g.txtX0 = x0; g.txtW = tokW * 0.8;
      blockGroups.push(g);
    });
  }
  blocks.forEach((b, i) => {
    const [sx, sz] = slot(i);
    const z0 = sz + Z_OFF;
    const isJoint = b.kind === 'joint';
    const idx = isQwen ? i : (isJoint ? i - 5 : i);
    const title = isQwen ? `Block ${i}` : (isJoint ? `合流 ${idx}` : `图像分流 ${idx}`);
    const g = addGroup({ name: b.name, kind: isJoint ? 'joint' : 'imgblock', title, block: b, stage: 'blocks', idx, order: i });
    gridGroup(g, sx + 18, z0 + 18, tileIndex[b.name], 10);
    // 文本 token 行（单流 / 合流块）
    if (isJoint) {
      const row = scalarRows.length; scalarRows.push({ name: `${b.name}.txt`, len: nTxt });
      const w = 64 / nTxt;
      for (let t = 0; t < nTxt; t++) push(g, sx + 18 + t * w, 0, z0 + 8, w * 0.85, SRC.SCALAR, t, row, PAL.TXT, 7, 0.1);
      g.txtRow = row; g.txtX0 = sx + 18; g.txtW = w; g.txtZ = z0 + 8;
      // 每个头一根柱：图像 query 注意文本的比例
      const hr = scalarRows.length; scalarRows.push({ name: `${b.name}.heads`, len: H });
      const hw = 64 / H;
      for (let h = 0; h < H; h++) push(g, sx + 18 + h * hw, 0, z0 + 18 + G + 4, hw * 0.75, SRC.SCALAR, h, hr, PAL.ATTN, 18, 0.1);
      g.headRow = hr; g.headX0 = sx + 18; g.headW = hw; g.headZ = z0 + 18 + G + 4;
    }
    // 权重板：网格左侧一列扁平板，面积 ∝ 矩阵元素数（蓝 = 注意力 q/k/v/o，绿 = MLP）
    const p = b.params;
    g.params = p;
    const mats = isQwen ? [1, 1, 1, 1, 3, 3, 3] : [1, 1, 1, 1, 6, 3];
    let zz = z0 + 18;
    mats.forEach((a, m) => {
      const depth = 4.2 * a;
      // 一块板 = 7 × depth，用 7×7 的方块沿 z 拼出（最后一块按剩余深度截短不了，就让它略重叠）
      for (let k = 0; k < depth - 0.01; k += 7) {
        push(g, sx + 6, 0, zz + Math.min(k, Math.max(0, depth - 7)), Math.min(7, depth), SRC.STATIC, 0, 0, PAL.GREY, 0.8, 0.05, m >= 4 ? 4 : 3);
      }
      zz += depth + 1.2;
    });
    blockGroups.push(g);
  });
  // FLUX：未使用的参考图分流块（文生图不执行），画成一排灰色板
  if (unused.length) {
    const g = addGroup({ name: 'mode_blocks.image_cond', kind: 'unused', title: '参考图分流（文生图未执行）', stage: 'blocks' });
    unused.forEach((b, i) => push(g, i * PITCH_X + 18, 0, -40, 30, SRC.STATIC, 0, 0, PAL.GREY, 0.5, 0.05, 7));
    g.params = unused.reduce((a, b) => a + b.params, 0);
  }

  // ---------------- 5. 输出速度 v、VAE、最终图像
  const last = slot(blocks.length);
  const outX = (blocks.length % PER_ROW === 0 ? 0 : last[0]) + 18;
  const outZ = (blocks.length % PER_ROW === 0 ? last[1] : last[1]) + Z_OFF + 18;
  const gOut = addGroup({ name: 'v', kind: 'latent', title: '输出 v', stage: 'output' });
  gridGroup(gOut, outX, outZ, tileIndex.v, 6);
  const gVae = addGroup({ name: 'vae', kind: 'vae', title: 'VAE 解码器', stage: 'output' });
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) push(gVae, outX + G + 14 + i * 3, 0, outZ + 18 + j * 3, 2.6, SRC.STATIC, 0, 0, PAL.GREY, 4 + ((i + j) % 3), 0.1, 6);
  const imgPlane = { x: outX + G + 50, z: outZ + G / 2, size: G + 8 };

  // 全局包围盒
  const all = { minX: Infinity, minY: 0, minZ: Infinity, maxX: -Infinity, maxY: 1, maxZ: -Infinity };
  for (const g of groups) {
    all.minX = Math.min(all.minX, g.bbox.minX); all.maxX = Math.max(all.maxX, g.bbox.maxX);
    all.minZ = Math.min(all.minZ, g.bbox.minZ); all.maxZ = Math.max(all.maxZ, g.bbox.maxZ);
    all.maxY = Math.max(all.maxY, g.bbox.maxY);
  }
  all.maxX = Math.max(all.maxX, imgPlane.x + imgPlane.size);

  const SW = Math.max(nTxt, H, 16);
  return {
    data: new Float32Array(inst), count: inst.length / 12, groups, labels, bbox: all,
    scalarRows, SW, blockGroups, gTok, gTE, gTxtIn, gTime, gIn, gOut, gVae, imgPlane, nRows, Z_OFF,
  };
}

/** 生成某一步的标量纹理数据（R32F，SW × rows）。 */
export function scalarData(scene, meta, step) {
  const { SW, scalarRows } = scene;
  const out = new Float32Array(SW * scalarRows.length);
  const st = meta.steps[step];
  // 第 0 行：时间表（当前步及之前的 t 高亮由 main.js 处理；这里放 t 的数值）
  const N = meta.steps.length;
  for (let i = 0; i < 16; i++) {
    const k = Math.round(i * (N - 1) / 15);
    out[i] = k <= step ? meta.steps[k].t : 0;
  }
  scalarRows.forEach((r, ri) => {
    if (ri === 0) return;
    const base = ri * SW;
    if (r.name.startsWith('te.')) {
      const l = +r.name.slice(3);
      const vals = meta.te_layer_rms[l];
      const sorted = [...vals].sort((a, b) => a - b);
      const hi = sorted[Math.floor(sorted.length * 0.95)] || 1;
      vals.forEach((v, i) => { out[base + i] = Math.min(1, v / hi); });
    } else if (r.name.endsWith('.txt')) {
      const name = r.name.slice(0, -4);
      const vals = nearestTxt(meta, name, step);
      if (vals) {
        const sorted = [...vals].sort((a, b) => a - b);
        const hi = sorted[Math.floor(sorted.length * 0.95)] || 1;
        vals.forEach((v, i) => { out[base + i] = Math.min(1, v / hi); });
      }
    } else if (r.name.endsWith('.heads')) {
      const name = r.name.slice(0, -6);
      const hf = st.head_frac[name];
      if (hf) hf.forEach((v, i) => { out[base + i] = v; });
    }
  });
  return out;
}

/** 文本 token 的 RMS 只在首/末步导出：取最近的一步。 */
export function nearestTxt(meta, name, step) {
  let best = null, bd = Infinity;
  for (const s of meta.txt_token_steps) {
    const v = meta.txt_rms_tok[`${s}/${name}`];
    if (v && Math.abs(s - step) < bd) { bd = Math.abs(s - step); best = v; }
  }
  return best;
}
