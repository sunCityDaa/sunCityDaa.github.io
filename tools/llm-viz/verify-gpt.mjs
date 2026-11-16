// verify-gpt.mjs — 用原站公开的参考激活值（gpt-nano-sort-t0-partials.json）
// 逐层校验 static/llm-viz/js/gpt.js 的推理结果。
// 运行：node tools/llm-viz/verify-gpt.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadModel, forward, decodeTensor, countParams, generateSorted } from '../../static/llm-viz/js/gpt.js';

const here = dirname(fileURLToPath(import.meta.url));
const model = loadModel(JSON.parse(readFileSync(join(here, '../../static/llm-viz/data/gpt-nano-sort-model.json'), 'utf8')));
const partials = JSON.parse(readFileSync(join(here, 'gpt-nano-sort-t0-partials.json'), 'utf8'));
const ref = (k) => decodeTensor(partials[k]);

const { nEmbd: C, blockSize: T, vocabSize: V, nHead: H } = model.config;
const B = partials.config.B; // 参考文件里有 3 条序列
let worst = 0;
let failed = false;

function compare(name, got, want, tol = 1e-4) {
  let maxErr = 0;
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i]) && !Number.isFinite(got[i])) continue; // 两边都是 -inf（被掩码）
    maxErr = Math.max(maxErr, Math.abs(got[i] - want[i]));
  }
  worst = Math.max(worst, maxErr);
  const ok = maxErr < tol;
  if (!ok) failed = true;
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${name.padEnd(22)} max|Δ| = ${maxErr.toExponential(2)}`);
}

const idxAll = ref('idx').data;
for (let b = 0; b < B; b++) {
  const idx = Array.from(idxAll.subarray(b * T, (b + 1) * T));
  console.log(`\n序列 ${b}: ${idx.map((i) => 'ABC'[i]).join(' ')}`);
  const { acts } = forward(model, idx);
  const row = (k, n) => ref(k).data.subarray(b * n, (b + 1) * n);
  compare('x (tok+pos embed)', acts.x0, row('x', T * C));
  compare('block0 output', acts.blocks[0].mlpResid, row('block0', T * C));
  compare('block1 output', acts.blocks[1].mlpResid, row('block1', T * C));
  compare('block2 output', acts.blocks[2].mlpResid, row('block2', T * C));
  compare('ln_f', acts.lnf, row('ln_f', T * C));
  compare('lm_head (logits)', acts.logits, row('lm_head', T * V), 1e-3);
  compare('probs', acts.probs, row('probs', T * V));
  // 第 0 个 block 的注意力矩阵（[B, H, T, T]）
  const attRef = ref('attSm').data;
  for (let h = 0; h < H; h++) {
    const want = attRef.subarray((b * H + h) * T * T, (b * H + h + 1) * T * T);
    compare(`block0 head${h} attSm`, acts.blocks[0].heads[h].attSm, want);
  }
}

const p = countParams(model);
console.log(`\n参数量：含 lm_head = ${p.total}，不含 lm_head（minGPT/原站口径）= ${p.excludingLmHead}`);

const demo = generateSorted(model, [2, 1, 0, 1, 1, 2]);
console.log(`排序演示：${demo.input.map((i) => 'ABC'[i]).join('')} → ${demo.output.map((i) => 'ABC'[i]).join('')}`);
console.log(`\n最大误差 ${worst.toExponential(2)}：${failed ? '校验失败' : '全部通过'}`);
process.exit(failed ? 1 : 0);
