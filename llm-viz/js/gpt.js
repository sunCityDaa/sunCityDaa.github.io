// gpt.js — 纯 JavaScript 实现的 GPT（minGPT 结构）前向推理。
// 不依赖任何库；同时可在浏览器和 Node.js 中运行。
// 每一步的中间结果（激活值）都会被保存下来，供 3D 可视化使用。

/** 把权重文件中的 base64 float32 张量解码成 { shape, data: Float32Array }。 */
export function decodeTensor(t) {
  const bin = typeof atob === 'function'
    ? Uint8Array.from(atob(t.data), (c) => c.charCodeAt(0))
    : Uint8Array.from(Buffer.from(t.data, 'base64'));
  // 复制到对齐的缓冲区，保证 Float32Array 可以直接视图访问
  const buf = new ArrayBuffer(bin.byteLength);
  new Uint8Array(buf).set(bin);
  return { shape: t.shape, data: new Float32Array(buf) };
}

/** 解析整个模型 JSON：config + 每个具名张量。 */
export function loadModel(json) {
  const w = {};
  for (const [name, t] of Object.entries(json)) {
    if (name === 'config') continue;
    w[name] = decodeTensor(t);
  }
  const c = json.config;
  const config = {
    nLayer: c.n_layer,
    nHead: c.n_head,
    nEmbd: c.n_embd,
    vocabSize: c.vocab_size,
    blockSize: c.block_size,
    headSize: c.n_embd / c.n_head,
  };
  return { config, w };
}

/**
 * 参数量统计。
 * minGPT 打印参数量时只统计 transformer 部分（不含 lm_head），
 * 原站显示的 85,584 正是这个口径；含 lm_head 则为 85,728。
 */
export function countParams(model) {
  let total = 0;
  let lmHead = 0;
  for (const [name, t] of Object.entries(model.w)) {
    if (name.endsWith('.attn.bias')) continue; // 因果掩码缓冲区，不是参数
    total += t.data.length;
    if (name.startsWith('lm_head')) lmHead += t.data.length;
  }
  return { total, excludingLmHead: total - lmHead };
}

/** 按公式计算任意形状 GPT 的参数量（与 minGPT 口径一致，不含 lm_head）。 */
export function paramFormula({ C, T, nBlocks, vocabSize }) {
  return vocabSize * C + T * C + nBlocks * (12 * C * C + 13 * C) + 2 * C;
}

// ---------- 基础算子 ----------

// y[t, o] = sum_i x[t, i] * W[o, i] + b[o]   （PyTorch Linear：W 形状为 [out, in]）
function linear(x, T, nIn, W, b, nOut) {
  const y = new Float32Array(T * nOut);
  for (let t = 0; t < T; t++) {
    const xo = t * nIn;
    for (let o = 0; o < nOut; o++) {
      let s = b ? b[o] : 0;
      const wo = o * nIn;
      for (let i = 0; i < nIn; i++) s += x[xo + i] * W[wo + i];
      y[t * nOut + o] = s;
    }
  }
  return y;
}

// LayerNorm：返回归一化结果和每个 token 的统计量 [μ, 1/σ]
function layerNorm(x, T, C, gamma, beta, eps = 1e-5) {
  const y = new Float32Array(T * C);
  const agg = new Float32Array(T * 2);
  for (let t = 0; t < T; t++) {
    let mean = 0;
    for (let i = 0; i < C; i++) mean += x[t * C + i];
    mean /= C;
    let v = 0;
    for (let i = 0; i < C; i++) { const d = x[t * C + i] - mean; v += d * d; }
    v /= C;
    const rstd = 1 / Math.sqrt(v + eps);
    agg[t * 2] = mean;
    agg[t * 2 + 1] = rstd;
    for (let i = 0; i < C; i++) y[t * C + i] = (x[t * C + i] - mean) * rstd * gamma[i] + beta[i];
  }
  return { y, agg };
}

// minGPT 使用的 GELU（tanh 近似）
function gelu(x) {
  const y = new Float32Array(x.length);
  const k = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    y[i] = 0.5 * v * (1 + Math.tanh(k * (v + 0.044715 * v * v * v)));
  }
  return y;
}

function add(a, b) {
  const y = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
  return y;
}

/**
 * 完整前向推理。
 * @param model loadModel() 的返回值
 * @param idx   token id 数组（长度 ≤ blockSize）
 * @returns { T, logits, probs, acts }  acts 中保存所有中间张量（均为行优先 [T, *]）
 */
export function forward(model, idx) {
  const { nLayer, nHead, nEmbd: C, vocabSize: V, headSize: A } = model.config;
  const w = model.w;
  const T = idx.length;

  // 1) 嵌入：token embedding + position embedding
  const tokEmb = new Float32Array(T * C);
  const posEmb = new Float32Array(T * C);
  const wte = w['transformer.wte.weight'].data;
  const wpe = w['transformer.wpe.weight'].data;
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < C; i++) {
      tokEmb[t * C + i] = wte[idx[t] * C + i];
      posEmb[t * C + i] = wpe[t * C + i];
    }
  }
  let x = add(tokEmb, posEmb);
  const acts = { idx: Float32Array.from(idx), tokEmb, posEmb, x0: x, blocks: [] };

  const scale = 1 / Math.sqrt(A);
  for (let l = 0; l < nLayer; l++) {
    const p = `transformer.h.${l}.`;
    const B = {};
    // 2) LayerNorm 1
    const ln1 = layerNorm(x, T, C, w[p + 'ln_1.weight'].data, w[p + 'ln_1.bias'].data);
    B.ln1 = ln1.y; B.ln1Agg = ln1.agg;
    // 3) Q、K、V：一次线性层得到 [T, 3C]，再按 q|k|v、按头切分
    const qkv = linear(ln1.y, T, C, w[p + 'attn.c_attn.weight'].data, w[p + 'attn.c_attn.bias'].data, 3 * C);
    B.heads = [];
    const yCat = new Float32Array(T * C);
    for (let h = 0; h < nHead; h++) {
      const q = new Float32Array(T * A), k = new Float32Array(T * A), v = new Float32Array(T * A);
      for (let t = 0; t < T; t++) {
        for (let a = 0; a < A; a++) {
          q[t * A + a] = qkv[t * 3 * C + 0 * C + h * A + a];
          k[t * A + a] = qkv[t * 3 * C + 1 * C + h * A + a];
          v[t * A + a] = qkv[t * 3 * C + 2 * C + h * A + a];
        }
      }
      // 4) 注意力分数 = q·k / sqrt(A)，并施加因果掩码（只能看见自己和之前的 token）
      const att = new Float32Array(T * T);
      const attSm = new Float32Array(T * T);
      const smAgg = new Float32Array(T * 2);
      for (let i = 0; i < T; i++) {
        let mx = -Infinity;
        for (let j = 0; j < T; j++) {
          if (j > i) { att[i * T + j] = -Infinity; continue; }
          let s = 0;
          for (let a = 0; a < A; a++) s += q[i * A + a] * k[j * A + a];
          s *= scale;
          att[i * T + j] = s;
          if (s > mx) mx = s;
        }
        // 5) 按行 softmax（先减去最大值保证数值稳定）
        let sum = 0;
        for (let j = 0; j <= i; j++) { const e = Math.exp(att[i * T + j] - mx); attSm[i * T + j] = e; sum += e; }
        for (let j = 0; j <= i; j++) attSm[i * T + j] /= sum;
        smAgg[i * 2] = mx; smAgg[i * 2 + 1] = 1 / sum;
      }
      // 6) 用注意力权重对 V 加权求和
      const vOut = new Float32Array(T * A);
      for (let i = 0; i < T; i++) {
        for (let a = 0; a < A; a++) {
          let s = 0;
          for (let j = 0; j <= i; j++) s += attSm[i * T + j] * v[j * A + a];
          vOut[i * A + a] = s;
          yCat[i * C + h * A + a] = s;
        }
      }
      B.heads.push({ q, k, v, att, attSm, smAgg, vOut });
    }
    B.yCat = yCat;
    // 7) 输出投影 + 残差连接
    B.attnProj = linear(yCat, T, C, w[p + 'attn.c_proj.weight'].data, w[p + 'attn.c_proj.bias'].data, C);
    B.attnResid = add(x, B.attnProj);
    // 8) LayerNorm 2 + MLP（升维 4C → GELU → 降维 C）+ 残差
    const ln2 = layerNorm(B.attnResid, T, C, w[p + 'ln_2.weight'].data, w[p + 'ln_2.bias'].data);
    B.ln2 = ln2.y; B.ln2Agg = ln2.agg;
    B.fc = linear(ln2.y, T, C, w[p + 'mlp.c_fc.weight'].data, w[p + 'mlp.c_fc.bias'].data, 4 * C);
    B.gelu = gelu(B.fc);
    B.mlp = linear(B.gelu, T, 4 * C, w[p + 'mlp.c_proj.weight'].data, w[p + 'mlp.c_proj.bias'].data, C);
    B.mlpResid = add(B.attnResid, B.mlp);
    x = B.mlpResid;
    acts.blocks.push(B);
  }

  // 9) 最终 LayerNorm → lm_head → softmax
  const lnf = layerNorm(x, T, C, w['transformer.ln_f.weight'].data, w['transformer.ln_f.bias'].data);
  acts.lnf = lnf.y; acts.lnfAgg = lnf.agg;
  const logits = linear(lnf.y, T, C, w['lm_head.weight'].data, null, V);
  const probs = new Float32Array(T * V);
  const smAgg = new Float32Array(T * 2);
  for (let t = 0; t < T; t++) {
    let mx = -Infinity;
    for (let v = 0; v < V; v++) mx = Math.max(mx, logits[t * V + v]);
    let s = 0;
    for (let v = 0; v < V; v++) { const e = Math.exp(logits[t * V + v] - mx); probs[t * V + v] = e; s += e; }
    for (let v = 0; v < V; v++) probs[t * V + v] /= s;
    smAgg[t * 2] = mx; smAgg[t * 2 + 1] = 1 / s;
  }
  acts.logits = logits; acts.probs = probs; acts.smAgg = smAgg;
  return { T, logits, probs, acts };
}

/**
 * 自回归生成：输入 6 个字母（0=A,1=B,2=C），贪心地逐个预测排好序的 6 个字母。
 * 最后一次前向推理的序列长度为 11（= blockSize），可视化就展示这一次。
 */
export function generateSorted(model, input) {
  const n = input.length;
  const seq = input.slice();
  let last = null;
  for (let step = 0; step < n; step++) {
    last = forward(model, seq);
    const V = model.config.vocabSize;
    const t = seq.length - 1;
    let best = 0;
    for (let v = 1; v < V; v++) if (last.probs[t * V + v] > last.probs[t * V + best]) best = v;
    if (seq.length < model.config.blockSize) seq.push(best);
    else { seq.push(best); break; }
  }
  // seq 现在长度为 12：6 个输入 + 6 个输出；可视化窗口取前 11 个（最后一个输出由第 11 列预测）
  const output = seq.slice(n);
  const vizSeq = seq.slice(0, model.config.blockSize);
  const result = forward(model, vizSeq);
  return { input, output, vizSeq, result };
}
