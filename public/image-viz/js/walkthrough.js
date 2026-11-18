// walkthrough.js — 10 个章节。正文里的数字尽量从 meta.json 现算，避免写死。
// 每章：title、focus(scene) → 要高亮的 group 列表、cam(scene) → 相机对准的 group、step（可选，进入时跳到某一步）、
//       mode（'act' 默认 / 'attn' 词→区域注意力）、html(meta, ctx) → 正文。

const fmtB = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : (n / 1e6).toFixed(1) + 'M');
const isQ = (m) => m.model === 'qwen';
const sum = (a) => a.reduce((x, y) => x + y, 0);

function spark(meta, cur) {
  const W = 360, H = 70, N = meta.steps.length;
  const pts = meta.steps.map((s, i) => [8 + (i / (N - 1)) * (W - 16), 8 + (1 - s.t) * (H - 22)]);
  const path = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const c = pts[cur];
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" width="100%"><path d="${path}" fill="none" stroke="#2f6fe4" stroke-width="2"/>
    ${pts.map((p) => `<circle cx="${p[0]}" cy="${p[1]}" r="2" fill="#9db8ef"/>`).join('')}
    <circle cx="${c[0]}" cy="${c[1]}" r="4.5" fill="#f59e0b"/>
    <text x="8" y="${H - 2}" font-size="11" fill="#667085">步 0 · t=1（纯噪声）</text>
    <text x="${W - 8}" y="${H - 2}" font-size="11" fill="#667085" text-anchor="end">步 ${N - 1} · t=${meta.steps[N - 1].t.toFixed(3)} → 0</text></svg>`;
}

function fracBars(meta, step) {
  const st = meta.steps[step];
  const names = Object.keys(st.txt_frac);
  const W = 360, H = 64, bw = (W - 16) / names.length;
  const bars = names.map((n, i) => {
    const v = st.txt_frac[n], h = v * (H - 18);
    return `<rect x="${8 + i * bw}" y="${H - 14 - h}" width="${bw * 0.8}" height="${h}" fill="#d9467a"><title>${n}: ${(v * 100).toFixed(1)}%</title></rect>`;
  }).join('');
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" width="100%">${bars}
    <text x="8" y="${H - 2}" font-size="11" fill="#667085">第 0 层</text>
    <text x="${W - 8}" y="${H - 2}" font-size="11" fill="#667085" text-anchor="end">第 ${names.length - 1} 层 · 图像 query 分给文本 key 的注意力比例</text></svg>`;
}

export const CHAPTERS = [
  {
    id: 'intro', title: '总览',
    focus: (s) => null, cam: (s) => null, step: -1,
    html: (m) => {
      const c = m.components;
      const total = c.text_encoder.params + c.transformer.params + c.vae.params;
      return `<p>这是 <b>${m.title}</b> 在一张 A100 上真实生成一张 ${m.size}×${m.size} 图像的全过程。场景里的每一根小柱子都是一个真实的数：图像 token 的激活强度、文本 token 的激活强度、或某个注意力头分给文本的比例。拖动下方时间轴，可以看到每一步去噪时这些数的变化。</p>
      <p>整条管线有三部分（参数量为实测）：</p>
      <ol><li><b>文本编码器</b>：${c.text_encoder.class}，${fmtB(c.text_encoder.params)}，${c.text_encoder.layers} 层。把提示词变成向量，<i>只算一次</i>。</li>
      <li><b>去噪 Transformer</b>：${c.transformer.class}，${fmtB(c.transformer.params)}。每一步读入当前的噪声潜变量，预测「往干净图像走的方向」，共 ${m.steps_n} 步。</li>
      <li><b>VAE 解码器</b>：${c.vae.class}，${fmtB(c.vae.params)}。把最后的潜变量还原成像素。</li></ol>
      <p>合计约 ${fmtB(total)} 参数。左上角可切换 Qwen-Image-2.1 与 FLUX.3 Klein 两个模型。</p>
      <p class="hint">拖动旋转 · 右键/Shift 拖动平移 · 滚轮缩放 · 悬停看数值 · ← → 切换章节 · 空格播放去噪。</p>`;
    },
  },
  {
    id: 'prompt', title: '提示词与 token',
    focus: (s) => [s.gTok, s.gTE], cam: (s) => [s.gTok, s.gTE], step: 0,
    html: (m) => {
      const n = m.tokens.length;
      const words = m.words.filter((w) => !w.control).length;
      return `<p>提示词先被 Qwen 的分词器切成 token。图中最前面一排就是这 ${n} 个 token：<span class="sw" style="background:#ee9e54"></span>橙色是提示词本身，<span class="sw" style="background:#b8c1d6"></span>灰色是聊天模板里的特殊 token（<code>&lt;|im_start|&gt;</code>、<code>user</code>、换行等）。</p>
      <p>${isQ(m)
        ? `Qwen-Image-2.1 用固定模板 <code>system: Comprehend and analyze the provided prompt.</code> 包住提示词，编码后<b>丢掉 system 段的 ${m.dropped_system_tokens} 个 token</b>，剩下 ${n} 个送给 Transformer。`
        : `FLUX.3 Klein 用 Qwen3-VL 的聊天模板，并把长度<b>补齐到 80 的倍数</b>（这里正好 80，最后一个是 <code>&lt;|endoftext|&gt;</code> 填充）。填充 token 也会被注意力看到。`}</p>
      <p>本页从中挑出了 ${words} 个实词（fox、sign、LLM、VIZ……），在「词 → 图像区域」一章里可以看它们分别对应画面的哪一块。</p>
      <div class="tokens">${m.tokens.map((t, i) => {
        const w = m.words.find((x) => x.tok.includes(i) && !x.control);
        return `<span class="tk${w ? ' w' : ''}" title="#${i}">${t.replace(/</g, '&lt;').replace(/ /g, '·')}</span>`;
      }).join('')}</div>`;
    },
  },
  {
    id: 'te', title: '文本编码器',
    focus: (s) => [s.gTE, s.gTxtIn], cam: (s) => [s.gTE, s.gTxtIn], pitch: 0.75,
    html: (m) => {
      const c = m.components.text_encoder;
      return `<p>中间这块 ${c.layers} 行 × ${m.tokens.length} 列的柱阵就是文本编码器：每一行是一层（自上而下第 0 → ${c.layers - 1} 层），每一列是一个 token，柱高 = 该 token 在该层输出的 RMS（每层单独归一化）。</p>
      <p>它是一个完整的 Qwen3-VL 语言模型：隐藏维度 ${c.hidden}，${c.heads} 个查询头 / ${c.kv_heads} 个 KV 头，FFN ${c.ffn}。</p>
      <p>${isQ(m)
        ? `Qwen-Image-2.1 <b>只取最后一层</b>的输出（而且是最后那个 RMSNorm <i>之前</i>的值——diffusers 专门挂了个钩子绕过它），得到 [${m.tokens.length}, 4096]，再经 <code>txt_in</code>（RMSNorm → Linear → GELU → Linear）投影成 Transformer 的 4096 维。`
        : `FLUX.3 Klein <b>取第 4、8、12……32 共 8 层</b>的隐状态，沿通道拼起来：8 × 2560 = <b>20480</b> 维，再经 <code>context_embedder</code>（一个 20480→3072 的线性层，${fmtB(m.io.context_embedder.params)} 参数）投影成 3072 维。这样浅层的字面信息和深层的语义信息都能用上。`}</p>
      <p>文本只编码一次，之后 ${m.steps_n} 步去噪都复用。</p>`;
    },
  },
  {
    id: 'latent', title: '噪声与潜变量',
    focus: (s) => [s.gIn], cam: (s) => [s.gIn], step: 0,
    html: (m) => {
      const t = m.components.transformer.config;
      return `<p>模型不直接在像素上画图，而是在 VAE 的<b>潜空间</b>里工作。${m.size}×${m.size} 的图像对应 <b>${m.grid}×${m.grid} = ${m.grid * m.grid}</b> 个图像 token，每个 token 代表 16×16 个像素，带 ${t.in_channels} 个通道。</p>
      <p>${isQ(m)
        ? `Qwen-Image-2.1 的 VAE 本身就压缩 16 倍、输出 64 通道，所以直接把 64×64×64 的潜变量展平成 4096 个 token（<code>patch_size = 1</code>），经 <code>img_in</code>（64→4096）进入 Transformer。`
        : `FLUX.3 的 VAE 压缩 8 倍、输出 32 通道，再把每 2×2 个位置打包成一个 token（32×4 = 128 通道），同样得到 4096 个 token，经 <code>x_embedders.image</code>（128→3072）进入 Transformer。`}</p>
      <p>这块 64×64 的柱阵是<b>当前步的输入 x_t</b>，柱高是每个 token 的 RMS。第 0 步它就是纯高斯噪声，所以看起来是均匀的杂乱。左侧面板的小图「x_t」是同一组数据的线性 RGB 投影。</p>`;
    },
  },
  {
    id: 'time', title: '时间步与调制',
    focus: (s) => [s.gTime], cam: (s) => [s.gTime, s.gIn],
    html: (m, ctx) => {
      const st = m.steps[ctx.step];
      return `<p>去噪是一条从 t=1（纯噪声）走到 t=0（干净图像）的轨迹。本次运行的时间表（橙点 = 当前步，t = ${st.t.toFixed(4)}${st.target_t !== undefined ? `，目标 t = ${st.target_t.toFixed(4)}` : ''}）：</p>
      ${spark(m, ctx.step)}
      <p>${isQ(m)
        ? `Qwen-Image-2.1 用 ${m.steps_n} 步，Flow-Match Euler 调度器带动态 shift。t 先经 256 维正弦编码和两层 MLP 变成 4096 维的 temb，再由<b>一个全模型共享的 <code>modulation</code> 层</b>（4096 → 16384，${fmtB(m.io.modulation.params)}）一次性算出 4 组向量：注意力前的 scale / gate、MLP 前的 scale / gate。32 个 block 都用这同一组，block 自己没有调制参数。`
        : `FLUX.3 Klein 是<b>步数蒸馏</b>的「流映射」模型：只要 ${m.steps_n} 步，而且最后一步直接从 t=${m.steps[m.steps_n - 1].t.toFixed(3)} 跳到 0。它同时嵌入<b>当前 t</b> 和<b>这一步要跳到的目标 t</b>（多一个 <code>target_timestep_embedder</code>），两者相加。调制按模态分 6 组（分流/合流 × 图像/参考图/文本），每组一个 3072→9216 的线性层，输出 shift / scale / gate。`}</p>
      <p>${isQ(m) ? '有一个关键细节：<b>文本 token 永远用 t=0 的调制</b>（<code>causal_condition</code>），所以文本这一半的计算和步数无关，可以只在第 1 步算一次并缓存 K/V。' : '文本 token 和参考图 token 用 t=0 调制，图像 token 用当前 t。'}</p>`;
    },
  },
  {
    id: 'block', title: 'Transformer block',
    focus: (s) => s.blockGroups.slice(0, isQ(s.meta) ? 2 : 7), cam: (s) => s.blockGroups.slice(isQ(s.meta) ? 0 : 5, isQ(s.meta) ? 2 : 7),
    html: (m) => {
      const b = m.blocks.find((x) => x.kind === 'joint');
      const nJ = m.blocks.filter((x) => x.kind === 'joint').length;
      if (isQ(m)) {
        return `<p>Qwen-Image-2.1 是<b>单流</b>结构：${m.tokens.length} 个文本 token 和 4096 个图像 token 拼成一条 <b>${m.tokens.length + 4096}</b> 长的序列，${nJ} 个 block 都用同一套权重处理它们。</p>
        <p>每个 block（${fmtB(b.params)} 参数）里：</p>
        <ul><li>注意力：q/k/v/o 四个 4096×4096 矩阵，32 头 × 128 维，q、k 各有一个 RMSNorm，位置编码是三轴 RoPE（帧 16 / 高 56 / 宽 56 维）；</li>
        <li>MLP：SwiGLU，gate/proj 4096→12288，out 12288→4096；</li>
        <li>两处残差都乘 <code>tanh(gate)</code>，输入先 LayerNorm 再乘 <code>(1 + scale)</code>。</li></ul>
        <p><b>注意力掩码是「块因果」</b>：<code>(q ≥ k) 或 同一张图</code>。文本 token 只能看前面的文本，图像 token 能看全部文本和整张图。</p>
        <p>每个 block 在场景里画成：中间 64×64 的柱阵 = 输出中每个图像 token 的 RMS；前面一排 = ${m.tokens.length} 个文本 token；后面一排 ${m.heads} 根粉柱 = 每个头把多少注意力分给了文本；左边的扁板代表 7 个权重矩阵（蓝 = 注意力，绿 = MLP）。</p>
        <p class="hint">这些扁板只是「有这么多权重」的记号。下一章把它们按真实形状铺开：一个矩阵一块平板，面积 = 参数量。</p>`;
      }
      const nI = m.blocks.filter((x) => x.kind === 'img').length;
      return `<p>FLUX.3 Klein 是<b>先分流、后合流</b>：</p>
      <ol><li><b>分流 ${nI} 层</b>：文本、图像、参考图各有自己的 ${nI} 个 block，只在<i>自己内部</i>做自注意力。最上面一行是文本分流（每块一排 80 个 token），下面开头 ${nI} 块是图像分流。参考图分流在文生图时不执行（灰板）。</li>
      <li><b>合流 ${nJ} 层</b>：把 80 个文本 token 和 4096 个图像 token 拼成 4176 长的序列做联合注意力，文本和图像各用自己的调制。</li></ol>
      <p>每个 block ${fmtB(b.params)} 参数，结构是<b>并行块</b>：同一个调制后的输入同时送进注意力和 FFN，两者相加后乘 gate 加回残差。注意力 24 头 × 128 维、QK-Norm、四轴 RoPE（t/h/w/l 各 32 维）；FFN 是 SwiGLU，<code>linear_in</code> 一次输出 2×9216。</p>
      <p>合流块后面一排 ${m.heads} 根粉柱 = 每个头把多少注意力分给了文本。</p>
      <p class="hint">左边那列扁板是权重记号；下一章会把一个 block 的每个矩阵按真实行列比铺开。</p>`;
    },
  },
  {
    id: 'flow', title: '整体架构',
    focus: (s) => [s.gFlow], cam: (s) => [s.gFlow], pitch: 1.42, yaw: 0, pad: 1.05, only: true,
    html: (m) => {
      const a = m.arch;
      if (!a) return '<p>正在加载架构数据……</p>';
      const mats = a.block.modules.filter((x) => x.kind === 'w');
      const nJ = m.blocks.filter((b) => b.kind === 'joint').length;
      const tin = m.io[isQ(m) ? 'txt_in' : 'context_embedder'];
      return `<p>这一章是<b>整条管线</b>，从提示词到像素，一次去噪要走的部分全在这里。所有平板共用同一个比例尺，所以大小可以直接比。</p>
      <ol>
        <li><b>提示词 → 文本编码器</b>（${m.components.text_encoder.layers} 层，${fmtB(m.components.text_encoder.params)}）：折叠成一叠，因为 ${m.components.text_encoder.layers} 层的结构完全一样。</li>
        <li><b>文本投影</b>：${isQ(m) ? '<code>txt_in</code>，4096 → 4096' : '<code>context_embedder</code>，20480 → 3072'}，${fmtB(tin.params)}。只算一次。</li>
        <li><b>一个 block 展开</b>（${fmtB(a.block.params)}，${mats.length} 个矩阵）：左边分叉成两条支路——上面是注意力（q / k / v → softmax → o），下面是 MLP（gate、up → down），各自加回残差。两条支路都是<b>同一份调制</b>出来的输入。</li>
        <li><b>其余 ${nJ - 1} 个 block</b>：结构一模一样，所以折叠成一叠，不重复画。</li>
        <li><b>输出</b>：<code>norm_out</code> → <code>proj_out</code> 得到速度 v，VAE 解码成像素。</li>
      </ol>
      <p>把这一整条（除了文本编码）走 ${m.steps_n} 遍，就是一次出图。文本编码只在第 1 步前做一次，之后每步复用。</p>
      <p class="hint">下一章把一个 block 的矩阵按真实行列比铺开；再下一章是 block 之外的嵌入、投影和调制。悬停任意平板可以看全名、形状、参数量。</p>`;
    },
  },
  {
    id: 'arch', title: '权重矩阵',
    focus: (s) => [s.gArchBlock], cam: (s) => [s.gArchBlock], pitch: 1.42, yaw: 0, pad: 1.3, only: true,
    html: (m) => {
      const a = m.arch;
      if (!a) return '<p>正在加载架构数据……</p>';
      const mats = a.block.modules.filter((x) => x.kind === 'w');
      const sum = mats.reduce((t, x) => t + x.params, 0);
      const mlp = mats.filter((x) => x.group === 'mlp').reduce((t, x) => t + x.params, 0);
      const vecs = a.block.modules.filter((x) => x.kind !== 'w');
      const rows = (list) => list.filter((x) => x.kind === 'w').map((x) =>
        `<tr><td><code>${x.name}</code></td><td class="num">${x.shape.join(' × ')}</td><td class="num">${fmtB(x.params)}</td><td>${x.desc}</td></tr>`).join('');
      return `<p>前面几章看的都是<b>数据</b>——激活和注意力。这一章看<b>模型本身</b>：${a.block.class} 里到底有哪些矩阵、各是什么形状。</p>
      <p>场景下方每一块平板就是一个权重矩阵：长边对应较大的那一维，短边对应较小的一维，所以<b>平板的面积 ∝ 行 × 列</b>，也就是参数量。颜色：<span class="sw" style="background:#8ca6e6"></span>注意力 q/k/v/o，<span class="sw" style="background:#9ecc9e"></span>MLP，<span class="sw" style="background:#d9b7e6"></span>调制，<span class="sw" style="background:#66bab2"></span>输入/输出投影。</p>
      <p>实测这一层共 <b>${mats.length} 个矩阵 + ${vecs.length} 个归一化向量</b>，合计 ${fmtB(a.block.params)} 参数：</p>
      <div class="tablewrap"><table class="cmp arch"><tr><th>模块</th><th>形状</th><th>参数</th><th>作用</th></tr>${rows(a.block.modules)}</table></div>
      <p>${isQ(m)
        ? `注意力是 <b>4 个互相独立的 4096×4096</b>（q、k、v、输出各一个，不是融合 qkv），每个 ${fmtB(4096 * 4096)}；MLP 是 SwiGLU 的三个 ${'12288×4096'} 矩阵，每个 ${fmtB(12288 * 4096)}。MLP 恰好是注意力的 3 倍宽（<code>mlp_ratio=3</code>），所以三块绿板的面积是四块蓝板的 3 倍。`
        : `注意力是 <b>4 个互相独立的 3072×3072</b>；FFN 的 <code>linear_in</code> 一次算出 2×9216（gate 与 up 拼在一起），所以是 ${'18432×3072'}，是注意力的 6 倍面积；<code>linear_out</code> 是 ${'9216×3072'}。`}</p>
      <p>算一下：MLP 占这一层的 <b>${(mlp / sum * 100).toFixed(0)}%</b>（${fmtB(mlp)} / ${fmtB(sum)}）。所以这类模型里，参数大头既不是注意力、更不是嵌入，而是 FFN。</p>
      <p class="hint">下一章是同一张图里的<b>模型级模块</b>：嵌入、投影、调制、输出。从正上方看时，平板面积就是参数量的直接比较。</p>`;
    },
  },
  {
    id: 'io', title: '模型级模块',
    focus: (s) => [s.gArchIO], cam: (s) => [s.gArchIO], pitch: 1.42, yaw: 0, pad: 1.3, only: true,
    html: (m) => {
      const a = m.arch;
      if (!a) return '<p>正在加载架构数据……</p>';
      const rows = (list) => list.filter((x) => x.kind === 'w').map((x) =>
        `<tr><td><code>${x.name}</code></td><td class="num">${x.shape.join(' × ')}</td><td class="num">${fmtB(x.params)}</td><td>${x.desc}</td></tr>`).join('');
      const sum = a.io.filter((x) => x.kind === 'w').reduce((t, x) => t + x.params, 0);
      return `<p>block 之外，模型还有这些矩阵：把潜变量和文本投进 hidden，把时间步变成调制向量，最后把 hidden 投回速度。合计 ${fmtB(sum)}。</p>
      <div class="tablewrap"><table class="cmp arch"><tr><th>模块</th><th>形状</th><th>参数</th><th>作用</th></tr>${rows(a.io)}</table></div>
      <p>${isQ(m)
        ? `Qwen 的调制只有一个 <code>modulation.1</code>（4096 → 16384，${fmtB(67108864)}），<b>32 个 block 共用</b>——这是它比同规模 DiT 省下约 2B 参数的原因。block 自己没有调制参数。`
        : `FLUX 的调制在模型级分 <b>6 组</b>（分流/合流 × 图像/参考图/文本），每组 3072 → 9216，<b>逐 token</b> 生效；合流那 3 组是每个 token 一份 scale/shift/gate。`}</p>
      <p class="hint">注意分流块和合流块里都<b>没有</b>调制参数——它们从模型级拿。这也是为什么 block 的 122.7M 里只有 4 个注意力和 2 个 FFN 矩阵。</p>`;
    },
  },
  {
    id: 'stream', title: '残差流里的图像',
    focus: (s) => s.blockGroups, cam: (s) => s.blockGroups, step: -1,
    html: (m, ctx) => {
      const st = m.steps[ctx.step];
      const names = Object.keys(st.img_rms);
      const vals = names.map((n) => st.img_rms[n]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const loN = names[vals.indexOf(lo)], hiN = names[vals.indexOf(hi)];
      return `<p>把所有 block 的输出排在一起看（每块柱阵各自归一化，颜色只表示<b>块内</b>的相对强弱）。到最后一步，大部分层里都能直接认出狐狸和牌子的轮廓——模型内部的「残差流」本身就带着图像结构。</p>
      <p>当前步（t=${st.t.toFixed(3)}）各层图像 token 的平均 RMS 从 ${lo.toFixed(1)}（${loN}）到 ${hi.toFixed(1)}（${hiN}）。${isQ(m)
        ? '中段（约 16–20 层）几乎是均匀的、数值也最低，靠近输出的几层轮廓又重新出现，最后一层更接近要输出的速度，变成一团模糊。'
        : '文本分流的激活幅度是图像流的几十倍（输入是 8 层隐状态的堆叠）；合流中段的热图里眼睛、鼻子、牌子最亮。'}</p>
      <p>拖动时间轴从第 0 步往后看：早期每层都还是噪声的纹理，构图在前几步就出现，后面主要是补细节。</p>`;
    },
  },
  {
    id: 'attn', title: '词 → 图像区域', mode: 'attn',
    focus: (s) => s.capGroups, cam: (s) => [s.capGroups[s.capSel ?? 2]], step: -1, pitch: 1.05,
    html: (m, ctx) => {
      const w = m.words[ctx.word];
      return `<p>注意力决定了每个图像 token 从哪些文本 token 读信息。导出时按模型自己的公式（含 QK-Norm 与 RoPE）重算了 softmax，对选中的 ${m.cap_blocks.length} 个 block、${m.cap_steps.length} 个时间步，把<b>每个图像位置分给某个词的注意力</b>（所有头平均）画成 64×64 热图。</p>
      <p>当前词：<b>${w.text.replace(/</g, '&lt;')}</b>${w.control ? '（模板的第一个 token，常被用作注意力「汇点」，作对照）' : ''}。下面点选其他词：</p>
      <p class="hint">查看哪一层：${m.cap_blocks.map((b, i) => `<button class="cbtn${i === ctx.capSel ? ' on' : ''}" data-i="${i}">${isQ(m) ? 'Block ' + b : '合流 ' + b}</button>`).join('')}
      <br>哪一步：${m.cap_steps.map((st) => `<button class="sbtn${st === ctx.step ? ' on' : ''}" data-s="${st}">步 ${st}</button>`).join('')}</p>
      <div id="word-picker" class="words">${m.words.map((x, i) => `<button class="wbtn${i === ctx.word ? ' on' : ''}" data-i="${i}">${x.text.replace(/</g, '&lt;')}</button>`).join('')}</div>
      <p>可以看到 <b>LLM</b>、<b>VIZ</b> 集中在牌子上，<b>fox</b> 勾出狐狸，<b>paws</b> 落在爪子，<b>trees</b>、<b>background</b> 在背景。只有 ${m.cap_steps.length} 个时间步（${m.cap_steps.join('、')}）有完整热图，时间轴会自动吸附到最近的一步。</p>
      <p>每一层图像 query 分给文本的总比例（当前步）：</p>
      ${fracBars(m, ctx.step)}
      <p class="hint">${isQ(m) ? 'Qwen 第 0 层的图像 token 几乎把全部注意力都给了文本，之后降到 10–25%。' : 'FLUX 合流各层把约 15–50% 的注意力分给文本，不同层差别很大。'}粉柱阵列里每根柱是一个头，可以看出同一层的不同头分工很不一样。</p>`;
    },
  },
  {
    id: 'denoise', title: '一步步去噪', mode: 'act',
    focus: (s) => [s.gIn, s.gOut, s.gVae, ...s.blockGroups.slice(-2)], cam: (s) => [s.gOut, s.gVae, s.gImg], step: 0, pitch: 0.7,
    html: (m, ctx) => {
      const st = m.steps[ctx.step];
      return `<p>每一步 Transformer 输出一个<b>速度</b> v（右下 64×64 柱阵）。在流匹配里 x_t = (1−t)·x₀ + t·ε、v = ε − x₀，所以可以算出模型此刻认为的最终图像：</p>
      <div class="formula">x₀ ≈ x_t − t · v</div>
      <p>左侧面板的「x₀ 预测」就是把这个估计<b>真的送进 VAE 解码</b>得到的图（不是示意）。按 ▶ 播放，可以看到：</p>
      <ul><li>第 0 步（t=1）已经给出了构图：狐狸居中、背景是雪林；</li>
      <li>${isQ(m) ? '大约第 4 步牌子上开始出现字' : '第 2 步牌子上就能认出 LLM VIZ'}；</li>
      <li>之后主要在补毛发、雪粒、光斑等细节。</li></ul>
      <p>当前第 ${ctx.step} 步，t = ${st.t.toFixed(4)}。${isQ(m) ? '更新规则是 Euler：x_{t−Δ} = x_t − Δt · v。' : '蒸馏版每步直接跳到目标 t，所以 16 步就够了。'}</p>
      ${spark(m, ctx.step)}`;
    },
  },
  {
    id: 'vae', title: 'VAE 解码',
    focus: (s) => [s.gOut, s.gVae], cam: (s) => [s.gVae, s.gOut, s.gImg], step: -1, pitch: 0.55,
    html: (m) => {
      const v = m.components.vae;
      return `<p>最后一步之后，潜变量被送进 VAE 解码器还原成像素（图中竖立的画面）。</p>
      <p>${isQ(m)
        ? `Qwen-Image-2.1 的 VAE（${v.class}，${fmtB(v.params)}，解码器约占 3/4）先用每通道的 <code>latents_mean / latents_std</code> 反归一化 64 个通道，再上采样 16 倍。它输出的是 <b>RGBA 四通道</b>，管线把 alpha 合成到白底上。`
        : `FLUX.3 用 FLUX.2 的打包式 VAE（${v.class}，只有 ${fmtB(v.params)}）：先用 BatchNorm 的滑动均值/方差反归一化 128 通道，再把每个 token 拆回 2×2 × 32 通道，最后上采样 8 倍。`}</p>
      <p>本页每一步的 x₀ 预测都是用同一个解码器真实解码的；最后一步与管线输出逐像素对比，平均误差 ${m.perf.last_decode_mae} / 255。</p>`;
    },
  },
  {
    id: 'compare', title: '两个模型对比',
    focus: (s) => null, cam: (s) => null,
    html: (m, ctx) => {
      const o = ctx.other;
      const row = (k, a, b) => `<tr><td>${k}</td><td>${a}</td><td>${b}</td></tr>`;
      const Q = isQ(m) ? m : o, F = isQ(m) ? o : m;
      if (!Q || !F) return '<p>正在加载另一个模型的数据……</p>';
      const p = (x) => x.perf.probe || {};
      return `<table class="cmp"><tr><th></th><th>Qwen-Image-2.1</th><th>FLUX.3 Klein</th></tr>
      ${row('文本编码器', fmtB(Q.components.text_encoder.params), fmtB(F.components.text_encoder.params))}
      ${row('取哪几层', '最后 1 层（4096）', '第 4…32 层共 8 层（20480）')}
      ${row('去噪 Transformer', fmtB(Q.components.transformer.params), fmtB(F.components.transformer.params))}
      ${row('结构', '32 层单流', '5 层分流 + 28 层合流')}
      ${row('隐藏维 / 头', '4096 / 32×128', '3072 / 24×128')}
      ${row('注意力掩码', '块因果（文本因果）', '分流：模态内；合流：全连接')}
      ${row('调制', '1 组全模型共享', '6 组，逐 token')}
      ${row('VAE', fmtB(Q.components.vae.params) + '，16×，64 通道', fmtB(F.components.vae.params) + '，8×+2×2 打包')}
      ${row('步数', Q.steps_n, F.steps_n + '（流映射蒸馏）')}
      ${row('生成耗时*', (p(Q).run_seconds ?? '?') + ' s', (p(F).run_seconds ?? '?') + ' s')}
      ${row('显存峰值*', (p(Q).peak_mem_gb ?? '?') + ' GB', (p(F).peak_mem_gb ?? '?') + ' GB')}
      </table>
      <p class="hint">* A100-SXM4-40GB，1024²，bf16，diffusers 0.42.1，不含加载。Qwen 用前缀 KV 缓存（默认）。</p>
      <p>两者都用 Qwen3-VL 当文本编码器、都在 64×64 个图像 token 上做流匹配，但取舍很不一样：Qwen 把参数集中在一个更宽的单流网络里、靠缓存省掉文本侧的重复计算；FLUX 用更窄的网络、多层文本特征和步数蒸馏换速度。</p>
      <p>完整的实测报告与导出脚本见 <a href="/posts/image-viz/">文档</a>。</p>`;
    },
  },
];
