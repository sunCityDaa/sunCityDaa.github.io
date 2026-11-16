// layout.js — 把 GPT 的每个张量排布成 3D 空间中的一块"格子板"。
// 坐标约定：x 向右、y 向上（模型自上而下堆叠，所以 y 为负）、z 朝向观察者。
// 一个格子 = 1 个单位。激活值张量 [T, n] 画成 T 列 × n 行：每一列对应一个 token。

export const KIND = { WEIGHT: 0, ACT: 1, AGG: 2, TOKEN: 3 };

const GAP = 2;        // 同一组内张量之间的间距
const GROUP_GAP = 5;  // 子层之间的间距
const BLOCK_GAP = 12; // Transformer block 之间的间距

/**
 * 生成布局。
 * @param shape { T, C, nHeads, A, nBlocks, vocabSize }
 * @param opts  { solid: 大模型只画实心盒子（不逐格绘制）, xOffset }
 * @returns { tensors, blocks, bbox }
 * 每个 tensor：{ id, name, label, group, kind, cols, rows, x, y, z, get(r,c), solid }
 *   (x, y) 为左上角格子的左上角；get(r, c) 返回该格的数值（大模型为 null）。
 */
export function genLayout(shape, opts = {}) {
  const { T, C, nHeads: H, A, nBlocks: L, vocabSize: V } = shape;
  const solid = !!opts.solid;
  const x0 = opts.xOffset ?? 0;
  const tensors = [];
  const blocks = [];
  let y = 0;

  // 辅助：添加一个张量。get 由 bindActs() 在推理之后填入。
  function add(t) {
    const tensor = { z: 0, solid, get: null, key: null, ...t, id: tensors.length };
    tensor.x += x0;
    tensors.push(tensor);
    return tensor;
  }
  // 主列（激活值）水平居中于 x = 0
  const mainX = -T / 2;

  // ---------- 嵌入 ----------
  const embedTop = y;
  add({ name: 'Tokens', label: 'Tokens（输入 token）', group: 'embed', kind: KIND.TOKEN, cols: T, rows: 1, x: mainX, y, key: 'idx' });
  y -= 1 + GROUP_GAP;
  const embY = y;
  add({ name: 'Token Embed', label: 'Token 嵌入矩阵 wte', group: 'embed', kind: KIND.WEIGHT, cols: V, rows: C, x: mainX - V - GAP * 3, y: embY, key: 'w:wte' });
  add({ name: 'Position Embed', label: '位置嵌入矩阵 wpe', group: 'embed', kind: KIND.WEIGHT, cols: T, rows: C, x: mainX + T + GAP * 5, y: embY, key: 'w:wpe' });
  y -= C + GROUP_GAP;
  add({ name: 'Input Embed', label: '输入嵌入 x = tok + pos', group: 'embed', kind: KIND.ACT, cols: T, rows: C, x: mainX, y, key: 'x0' });
  y -= C + GROUP_GAP;
  blocks.push({ name: 'embed', top: embedTop, bottom: y });

  // LayerNorm 子层（统计量 μ/σ、γ、β、输出）
  function layerNormGroup(prefix, group, keyBase, label) {
    add({ name: `${prefix} LN Agg: μ, σ`, label: `${label} 统计量 μ, 1/σ`, group, kind: KIND.AGG, cols: T, rows: 2, x: mainX, y, key: `${keyBase}Agg` });
    y -= 2 + GAP;
    add({ name: `${prefix} γ`, label: 'γ（缩放）', group, kind: KIND.WEIGHT, cols: 1, rows: C, x: mainX - 1 - GAP * 2 - 1 - GAP, y, key: `w:${keyBase}.g` });
    add({ name: `${prefix} β`, label: 'β（偏移）', group, kind: KIND.WEIGHT, cols: 1, rows: C, x: mainX - 1 - GAP * 2, y, key: `w:${keyBase}.b` });
    const ln = add({ name: `${prefix} Layer Norm`, label: `${label} 输出`, group, kind: KIND.ACT, cols: T, rows: C, x: mainX, y, key: keyBase });
    y -= C + GROUP_GAP;
    return ln;
  }

  for (let l = 0; l < L; l++) {
    const g = `block${l}`;
    const P = `B${l}`;
    const top = y;
    const b = `b${l}.`;

    // ---- LayerNorm 1 ----
    layerNormGroup(P, g, b + 'ln1', `Block ${l} · LayerNorm 1`);

    // ---- 自注意力：每个头一列，水平并排 ----
    const headW = C + 1 + T + GAP * 3;         // 权重 | 偏置 | 向量
    const headsTotalW = H * headW + (H - 1) * GAP * 3;
    const headsLeft = -headsTotalW / 2;
    const attTop = y;
    let attBottom = y;
    for (let h = 0; h < H; h++) {
      const hx = headsLeft + h * (headW + GAP * 3);
      const wx = hx, bx = hx + C + GAP, vx = hx + C + 1 + GAP * 2;
      let hy = attTop;
      for (const [qn, qi] of [['Q', 0], ['K', 1], ['V', 2]]) {
        const hp = `${P} H${h} ${qn}`;
        add({ name: `${hp} Weights`, label: `Block ${l} · 头 ${h} · ${qn} 权重`, group: g, kind: KIND.WEIGHT, cols: C, rows: A, x: wx, y: hy, key: `w:${b}attn.${qi}.${h}` });
        add({ name: `${hp} Bias`, label: `${qn} 偏置`, group: g, kind: KIND.WEIGHT, cols: 1, rows: A, x: bx, y: hy, key: `w:${b}attnb.${qi}.${h}` });
        add({ name: `${hp} vectors`, label: `Block ${l} · 头 ${h} · ${qn} 向量`, group: g, kind: KIND.ACT, cols: T, rows: A, x: vx, y: hy, key: `${b}h${h}.${qn.toLowerCase()}` });
        hy -= A + GAP;
      }
      hy -= GROUP_GAP - GAP;
      add({ name: `${P} H${h} Attention Matrix`, label: `Block ${l} · 头 ${h} · 注意力分数 QKᵀ/√A`, group: g, kind: KIND.ACT, cols: T, rows: T, x: vx, y: hy, key: `${b}h${h}.att` });
      hy -= T + GAP;
      add({ name: `${P} H${h} Attn Softmax Agg`, label: '行最大值 / 1/Σexp', group: g, kind: KIND.AGG, cols: 2, rows: T, x: vx - 2 - GAP, y: hy, key: `${b}h${h}.smAgg`, transposeAgg: true });
      add({ name: `${P} H${h} Attn Matrix Softmax`, label: `Block ${l} · 头 ${h} · 注意力权重（softmax）`, group: g, kind: KIND.ACT, cols: T, rows: T, x: vx, y: hy, key: `${b}h${h}.attSm` });
      hy -= T + GAP;
      add({ name: `${P} H${h} V Output`, label: `Block ${l} · 头 ${h} · 加权 V 输出`, group: g, kind: KIND.ACT, cols: T, rows: A, x: vx, y: hy, key: `${b}h${h}.vOut` });
      hy -= A;
      attBottom = Math.min(attBottom, hy);
    }
    y = attBottom - GROUP_GAP;

    // ---- 合并多头 + 投影 + 残差 ----
    add({ name: `${P} V Output Combined`, label: `Block ${l} · 多头拼接`, group: g, kind: KIND.ACT, cols: T, rows: C, x: mainX, y, key: `${b}yCat` });
    y -= C + GROUP_GAP;
    add({ name: `${P} Projection Weights`, label: `Block ${l} · 投影权重 c_proj`, group: g, kind: KIND.WEIGHT, cols: C, rows: C, x: mainX - C - 1 - GAP * 2, y, key: `w:${b}attn.proj` });
    add({ name: `${P} Projection Bias`, label: '投影偏置', group: g, kind: KIND.WEIGHT, cols: 1, rows: C, x: mainX - 1 - GAP, y, key: `w:${b}attn.projb` });
    add({ name: `${P} Attention Output`, label: `Block ${l} · 注意力输出`, group: g, kind: KIND.ACT, cols: T, rows: C, x: mainX, y, key: `${b}attnProj` });
    add({ name: `${P} Attention Residual`, label: `Block ${l} · 残差 x + attn`, group: g, kind: KIND.ACT, cols: T, rows: C, x: mainX + T + GAP * 5, y, key: `${b}attnResid` });
    y -= C + GROUP_GAP;

    // ---- LayerNorm 2 ----
    layerNormGroup(`${P} 2`, g, b + 'ln2', `Block ${l} · LayerNorm 2`);

    // ---- MLP ----
    add({ name: `${P} MLP Weights`, label: `Block ${l} · MLP 升维权重 c_fc`, group: g, kind: KIND.WEIGHT, cols: C, rows: 4 * C, x: mainX - C - 1 - GAP * 2, y, key: `w:${b}mlp.fc` });
    add({ name: `${P} MLP Bias`, label: 'MLP 偏置', group: g, kind: KIND.WEIGHT, cols: 1, rows: 4 * C, x: mainX - 1 - GAP, y, key: `w:${b}mlp.fcb` });
    add({ name: `${P} MLP`, label: `Block ${l} · MLP 隐藏层`, group: g, kind: KIND.ACT, cols: T, rows: 4 * C, x: mainX, y, key: `${b}fc` });
    add({ name: `${P} MLP Activation`, label: `Block ${l} · GELU 激活`, group: g, kind: KIND.ACT, cols: T, rows: 4 * C, x: mainX + T + GAP * 5, y, key: `${b}gelu` });
    y -= 4 * C + GROUP_GAP;
    add({ name: `${P} MLP Projection Weights`, label: `Block ${l} · MLP 降维权重 c_proj`, group: g, kind: KIND.WEIGHT, cols: 4 * C, rows: C, x: mainX - 4 * C - 1 - GAP * 2, y, key: `w:${b}mlp.proj` });
    add({ name: `${P} MLP Projection Bias`, label: 'MLP 投影偏置', group: g, kind: KIND.WEIGHT, cols: 1, rows: C, x: mainX - 1 - GAP, y, key: `w:${b}mlp.projb` });
    add({ name: `${P} MLP Result`, label: `Block ${l} · MLP 输出`, group: g, kind: KIND.ACT, cols: T, rows: C, x: mainX, y, key: `${b}mlp` });
    add({ name: `${P} MLP Residual`, label: `Block ${l} · 残差（block 输出）`, group: g, kind: KIND.ACT, cols: T, rows: C, x: mainX + T + GAP * 5, y, key: `${b}mlpResid` });
    y -= C + BLOCK_GAP;
    blocks.push({ name: g, top, bottom: y });
  }

  // ---------- 输出 ----------
  const outTop = y;
  layerNormGroup('Final', 'output', 'lnf', '最终 LayerNorm ln_f');
  add({ name: 'LM Head Weights', label: 'LM Head 权重', group: 'output', kind: KIND.WEIGHT, cols: C, rows: V, x: mainX - C - GAP * 2, y, key: 'w:lm_head' });
  add({ name: 'Logits', label: 'Logits（每个字母的得分）', group: 'output', kind: KIND.ACT, cols: T, rows: V, x: mainX, y, key: 'logits' });
  y -= V + GAP;
  add({ name: 'SM Agg', label: 'Softmax 统计量 max, 1/Σexp', group: 'output', kind: KIND.AGG, cols: T, rows: 2, x: mainX, y, key: 'smAgg' });
  y -= 2 + GAP;
  add({ name: 'Logits Softmax', label: '概率分布 softmax(logits)', group: 'output', kind: KIND.ACT, cols: T, rows: V, x: mainX, y, key: 'probs' });
  y -= V;
  blocks.push({ name: 'output', top: outTop, bottom: y });

  // 包围盒
  let minX = Infinity, maxX = -Infinity;
  for (const t of tensors) { minX = Math.min(minX, t.x); maxX = Math.max(maxX, t.x + t.cols); }
  const bbox = { minX, maxX, minY: y, maxY: 1, width: maxX - minX, height: 1 - y };
  return { shape, tensors, blocks, bbox, solid };
}

/**
 * 把推理结果和权重绑定到布局上：为每个张量填入 get(r, c)。
 * 激活值 [T, n] 存储为行优先 data[t * n + i]，在画面中 r = i（通道）、c = t（token）。
 */
export function bindActs(layout, model, acts) {
  const { nEmbd: C, headSize: A } = model.config;
  const w = model.w;
  const W = (n) => w[n].data;
  for (const t of layout.tensors) {
    const k = t.key;
    if (!k) continue;
    if (k.startsWith('w:')) {
      t.get = weightGetter(k.slice(2), W, C, A);
      continue;
    }
    const arr = resolveAct(k, acts);
    if (!arr) { t.get = null; continue; }
    if (k === 'idx') t.get = (r, c) => arr[c];
    else if (k.endsWith('.att') || k.endsWith('.attSm')) {
      const n = t.cols; // 注意力矩阵：行 = 查询位置 i，列 = 被关注位置 j
      t.get = (r, c) => arr[r * n + c];
    } else if (t.transposeAgg) {
      t.get = (r, c) => arr[r * 2 + c];
    } else {
      const n = t.rows;
      t.get = (r, c) => arr[c * n + r];
    }
  }
}

function resolveAct(key, acts) {
  const m = key.match(/^b(\d+)\.(.+)$/);
  if (!m) return acts[key];
  const B = acts.blocks[+m[1]];
  const rest = m[2];
  const hm = rest.match(/^h(\d+)\.(.+)$/);
  if (hm) return B.heads[+hm[1]][hm[2]];
  return B[rest];
}

// 权重张量以 [out, in] 存储；画面中行 = out、列 = in。
function weightGetter(k, W, C, A) {
  if (k === 'wte') { const d = W('transformer.wte.weight'); return (r, c) => d[c * C + r]; }
  if (k === 'wpe') { const d = W('transformer.wpe.weight'); return (r, c) => d[c * C + r]; }
  if (k === 'lm_head') { const d = W('lm_head.weight'); return (r, c) => d[r * C + c]; }
  const m = k.match(/^(?:b(\d+)\.)?(.+)$/);
  const p = m[1] !== undefined ? `transformer.h.${m[1]}.` : 'transformer.';
  const rest = m[2];
  const ln = rest.match(/^(ln1|ln2|lnf)\.(g|b)$/);
  if (ln) {
    const name = { ln1: 'ln_1', ln2: 'ln_2', lnf: 'ln_f' }[ln[1]];
    const d = W(`${p}${name}.${ln[2] === 'g' ? 'weight' : 'bias'}`);
    return (r) => d[r];
  }
  const qkv = rest.match(/^attn(b?)\.(\d)\.(\d+)$/);
  if (qkv) {
    const off = +qkv[2] * C + +qkv[3] * A; // 在 3C 输出维度中的起始行
    if (qkv[1]) { const d = W(p + 'attn.c_attn.bias'); return (r) => d[off + r]; }
    const d = W(p + 'attn.c_attn.weight');
    return (r, c) => d[(off + r) * C + c];
  }
  const lin = {
    'attn.proj': ['attn.c_proj.weight', C], 'attn.projb': ['attn.c_proj.bias'],
    'mlp.fc': ['mlp.c_fc.weight', C], 'mlp.fcb': ['mlp.c_fc.bias'],
    'mlp.proj': ['mlp.c_proj.weight', 4 * C], 'mlp.projb': ['mlp.c_proj.bias'],
  }[rest];
  if (lin) {
    const d = W(p + lin[0]);
    if (lin[1] === undefined) return (r) => d[r];
    const nIn = lin[1];
    return (r, c) => d[r * nIn + c];
  }
  return null;
}

/**
 * 大模型（GPT-2 / GPT-3）的"实心"布局：结构与 nano-gpt 相同，但为了控制数量，
 * 多个注意力头合并为一个张量，并且每个张量只画一个实心盒子。
 */
export function genSolidLayout(shape, xOffset) {
  const { T, C, nHeads: H, A, nBlocks: L, vocabSize: V } = shape;
  const tensors = [];
  let y = 0;
  const gap = Math.max(2, C * 0.05);
  const mainX = xOffset - T / 2;
  const add = (name, cols, rows, x, kind, depth = 1) => tensors.push({ id: -1, name, label: name, cols, rows, x, y, z: 0, depth, kind, solid: true, get: null });
  add('Tokens', T, 1, mainX, KIND.TOKEN); y -= 1 + gap;
  add('Token Embed', V, C, mainX - V - gap, KIND.WEIGHT);
  add('Position Embed', T, C, mainX + T + gap, KIND.WEIGHT); y -= C + gap;
  add('Input Embed', T, C, mainX, KIND.ACT); y -= C + gap;
  for (let l = 0; l < L; l++) {
    add(`B${l} Layer Norm 1`, T, C, mainX, KIND.ACT); y -= C + gap;
    add(`B${l} QKV Weights（${H} 头）`, C, 3 * C, mainX - C - gap, KIND.WEIGHT);
    add(`B${l} QKV vectors`, T, 3 * C, mainX, KIND.ACT); y -= 3 * C + gap;
    add(`B${l} Attention Matrix ×${H} 头`, T, T, mainX, KIND.ACT, H); y -= T + gap;
    add(`B${l} Projection Weights`, C, C, mainX - C - gap, KIND.WEIGHT);
    add(`B${l} Attention Residual`, T, C, mainX, KIND.ACT); y -= C + gap;
    add(`B${l} Layer Norm 2`, T, C, mainX, KIND.ACT); y -= C + gap;
    add(`B${l} MLP Weights`, C, 4 * C, mainX - C - gap, KIND.WEIGHT);
    add(`B${l} MLP Activation`, T, 4 * C, mainX, KIND.ACT); y -= 4 * C + gap;
    add(`B${l} MLP Projection Weights`, 4 * C, C, mainX - 4 * C - gap, KIND.WEIGHT);
    add(`B${l} MLP Residual`, T, C, mainX, KIND.ACT); y -= C + gap * 2;
  }
  add('Final Layer Norm', T, C, mainX, KIND.ACT); y -= C + gap;
  add('LM Head Weights', C, V, mainX - C - gap, KIND.WEIGHT);
  add('Logits', T, V, mainX, KIND.ACT); y -= V + gap;
  add('Logits Softmax', T, V, mainX, KIND.ACT); y -= V;
  let minX = Infinity, maxX = -Infinity;
  for (const t of tensors) { minX = Math.min(minX, t.x); maxX = Math.max(maxX, t.x + t.cols); }
  return { shape, tensors, bbox: { minX, maxX, minY: y, maxY: 1, width: maxX - minX, height: 1 - y }, solid: true, A };
}
