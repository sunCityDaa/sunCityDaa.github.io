// walkthrough.js — 分章节讲解。每一章：标题、正文（HTML）、聚焦哪些张量、动画按什么顺序"计算"。
// 正文为本项目独立撰写的中文讲解，结构参考原站的章节划分。

const blk = (l) => (t) => t.group === `block${l}`;
const named = (...names) => (t) => names.some((n) => t.name === n);
const nameHas = (...parts) => (t) => parts.every((p) => t.name.includes(p));
const any = (...fs) => (t) => fs.some((f) => f(t));

export const CHAPTERS = [
  {
    id: 'intro',
    title: '简介',
    en: 'Introduction',
    focus: () => true,
    view: 'all',
    html: `
      <p>欢迎来到 GPT 大语言模型的 3D 可视化讲解。这里展示的是一个只有
      <b>85,584</b> 个参数的迷你模型 <i>nano-gpt</i>，它的结构与 GPT-2 / GPT-3 完全一样，只是小得多。</p>
      <p>它的任务很简单：读入 6 个由 A、B、C 组成的字母，输出按字母顺序排好的结果。</p>
      <p class="demo-seq"><span class="tok tA">C</span><span class="tok tB">B</span><span class="tok tA">A</span><span class="tok tB">B</span><span class="tok tB">B</span><span class="tok tC">C</span> → <b>ABBBCC</b></p>
      <p>右侧的每一个小方块都是一个数字：要么是模型学到的<b>权重</b>（灰蓝色底），要么是推理过程中算出来的<b>激活值</b>（浅绿色底）。
      <span class="sw pos"></span>蓝色表示正数、<span class="sw neg"></span>橙色表示负数，颜色越深绝对值越大。</p>
      <p>本页的推理是在你的浏览器里<b>真实运行</b>的：使用原项目公开的训练好的权重，由纯 JavaScript 逐层计算，结果与 PyTorch 参考值的误差小于 2×10⁻⁶。</p>
      <p class="hint">操作：拖动旋转 · 右键/Shift+拖动平移 · 滚轮缩放 · 鼠标悬停查看数值 · 键盘 ← → 切换章节 · 空格播放动画。</p>`,
  },
  {
    id: 'prelim',
    title: '预备知识',
    en: 'Preliminaries',
    focus: named('Tokens', 'Input Embed'),
    html: `
      <p><b>Token 与词表。</b>模型并不直接处理文字，而是处理 <i>token</i>。在这个例子里只有 3 种 token：
      A=0、B=1、C=2，这 3 个就是模型的<b>词表</b>（vocabulary）。真实的 GPT-2 词表有 50,257 个 token。</p>
      <p><b>张量。</b>可视化里的每块"格子板"都是一个二维数组（张量）。对于激活值，我们约定：
      <b>每一列对应一个 token 位置</b>，每一行对应一个通道（特征维度）。nano-gpt 的通道数 C = 48。</p>
      <p><b>上下文长度。</b>模型一次最多处理 T = 11 个 token（6 个输入 + 5 个已生成的输出），所以所有激活值都是 11 列。</p>
      <p><b>矩阵乘法。</b>几乎所有计算都是"线性层"：输出的每个数 = 输入的一列与权重的一行做点积，再加上偏置：</p>
      <p class="formula">y[t, o] = Σᵢ x[t, i] · W[o, i] + b[o]</p>
      <p>可视化中，权重矩阵画在激活值的左侧：行与输出对齐，列与输入通道对应。</p>`,
  },
  {
    id: 'embedding',
    title: '嵌入',
    en: 'Embedding',
    focus: (t) => t.group === 'embed',
    sequence: ['Tokens', 'Token Embed', 'Position Embed', 'Input Embed'],
    html: `
      <p>第一步把每个 token 变成一个 48 维的向量。</p>
      <p><b>Token 嵌入</b>（wte，形状 3×48）：每个字母对应其中一列。读入字母 B 时，就把第 1 列整列取出来——这其实就是"查表"。</p>
      <p><b>位置嵌入</b>（wpe，形状 11×48）：Transformer 本身不知道顺序，所以给每个位置也学一个向量，第 t 个位置取第 t 列。</p>
      <p>两者<b>逐元素相加</b>，就得到输入嵌入 <code>x = tok_emb + pos_emb</code>（11 列 × 48 行）。这个 x 也是贯穿整个模型的<b>残差流</b>的起点。</p>
      <p class="hint">点播放 ▶ 观察 x 是如何一列一列算出来的。</p>`,
  },
  {
    id: 'layernorm',
    title: '层归一化',
    en: 'Layer Norm',
    focus: (t) => t.group === 'block0' && t.name.startsWith('B0 ') && !t.name.startsWith('B0 2') && /LN Agg|γ|β|Layer Norm/.test(t.name),
    sequence: ['B0 LN Agg: μ, σ', 'B0 γ', 'B0 β', 'B0 Layer Norm'],
    html: `
      <p>进入每个 Transformer block 之前，先对每一列（每个 token）单独做 <b>LayerNorm</b>，让数值分布稳定、便于训练。</p>
      <ol>
        <li>求这一列 48 个数的均值 μ 和方差 σ²（紫色小板保存的就是 μ 和 1/σ）。</li>
        <li>标准化：<code>x̂ = (x − μ) / √(σ² + ε)</code>，使其均值为 0、方差为 1。</li>
        <li>再用两个学到的向量缩放和平移：<code>y = γ ⊙ x̂ + β</code>。</li>
      </ol>
      <p>γ、β 各有 48 个参数，画在左边细长的两列。每个 block 有两次 LayerNorm（注意力前、MLP 前），最后输出前还有一次 ln_f。</p>`,
  },
  {
    id: 'attention',
    title: '自注意力',
    en: 'Self Attention',
    focus: (t) => t.group === 'block0' && / H\d /.test(t.name),
    sequence: ['B0 H0 Q Weights', 'B0 H0 Q vectors', 'B0 H0 K vectors', 'B0 H0 V vectors', 'B0 H0 Attention Matrix', 'B0 H0 Attn Softmax Agg', 'B0 H0 Attn Matrix Softmax', 'B0 H0 V Output'],
    camFocus: nameHas('B0 H0'),
    html: `
      <p>自注意力是 Transformer 的核心：让每个 token 去"查看"之前的 token，并从中收集信息。</p>
      <p>对每一列，用三组权重分别算出三个 16 维向量：</p>
      <ul>
        <li><b>Q（Query，查询）</b>：我在找什么？</li>
        <li><b>K（Key，键）</b>：我能提供什么？</li>
        <li><b>V（Value，值）</b>：如果被选中，我交出什么内容？</li>
      </ul>
      <p>第 i 个 token 对第 j 个 token 的<b>注意力分数</b> = <code>Qᵢ · Kⱼ / √16</code>，组成 11×11 的方阵。
      因为模型不能"偷看未来"，j &gt; i 的位置被设为 −∞（<b>因果掩码</b>，画面中深灰色的上三角）。</p>
      <p>每一行做 softmax 得到注意力权重（每行和为 1），再用它们对 V 加权求和，得到这个头的输出。</p>
      <p>nano-gpt 每层有 <b>3 个注意力头</b>并排进行（48 = 3 × 16），每个头可以学习关注不同的模式——在排序任务中，有的头会去数前面有几个 A、几个 B。</p>`,
  },
  {
    id: 'projection',
    title: '投影',
    en: 'Projection',
    focus: (t) => t.group === 'block0' && /V Output Combined|Projection|Attention Output|Attention Residual/.test(t.name),
    sequence: ['B0 V Output Combined', 'B0 Projection Weights', 'B0 Projection Bias', 'B0 Attention Output', 'B0 Attention Residual'],
    html: `
      <p>3 个头各输出 16 维，<b>首尾拼接</b>回 48 维（"多头拼接"板）。</p>
      <p>再经过一个 48×48 的线性层（<code>c_proj</code>）把各个头的信息混合起来，得到注意力子层的输出。</p>
      <p>最后是关键的<b>残差连接</b>：把输出<b>加回</b>到进入本子层之前的 x 上：<code>x = x + attn(ln_1(x))</code>。
      残差让信息和梯度可以畅通地穿过很深的网络——这条不断被"加上新内容"的主干就叫做残差流。</p>`,
  },
  {
    id: 'mlp',
    title: 'MLP',
    en: 'MLP',
    focus: (t) => t.group === 'block0' && (/MLP/.test(t.name) || t.name.startsWith('B0 2 ')),
    sequence: ['B0 2 LN Agg: μ, σ', 'B0 2 Layer Norm', 'B0 MLP Weights', 'B0 MLP', 'B0 MLP Activation', 'B0 MLP Projection Weights', 'B0 MLP Result', 'B0 MLP Residual'],
    html: `
      <p>注意力负责 token <b>之间</b>的信息交流；MLP（多层感知机）则对每个 token <b>独立地</b>做加工。</p>
      <ol>
        <li>先做第二次 LayerNorm。</li>
        <li>线性层把 48 维<b>升到 192 维</b>（4 倍），即图中又高又长的那块板。</li>
        <li>逐元素通过 <b>GELU</b> 激活函数：<code>GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))</code>。它像一个平滑版的 ReLU，负值被压到接近 0。</li>
        <li>再用线性层<b>降回 48 维</b>。</li>
        <li>同样通过残差加回：<code>x = x + mlp(ln_2(x))</code>。</li>
      </ol>
      <p>MLP 权重占了每个 block 约三分之二的参数量（2 × 48 × 192 = 18,432）。</p>`,
  },
  {
    id: 'transformer',
    title: 'Transformer 块',
    en: 'Transformer',
    focus: any(blk(0), blk(1), blk(2)),
    camFocus: () => true,
    sequence: ['B0 MLP Residual', 'B1 MLP Residual', 'B2 MLP Residual'],
    html: `
      <p>"LayerNorm → 自注意力 → 残差 → LayerNorm → MLP → 残差"合起来就是一个 <b>Transformer block</b>。</p>
      <p>nano-gpt 把 3 个结构相同、权重不同的 block 首尾相接。每个 block 都读取残差流、计算一些新信息、再加回残差流。</p>
      <p>越往下的 block，残差流里积累的信息越抽象：最初只是"这里是字母 C、在第 3 位"，到最后已经变成"下一个应该输出 B"。</p>
      <p>GPT-2 small 有 12 个 block，GPT-3 有 96 个——结构一模一样，只是更多、更宽。</p>`,
  },
  {
    id: 'softmax',
    title: 'Softmax',
    en: 'Softmax',
    focus: (t) => t.group === 'output',
    sequence: ['Final LN Agg: μ, σ', 'Final Layer Norm', 'LM Head Weights', 'Logits', 'SM Agg', 'Logits Softmax'],
    html: `
      <p>残差流离开最后一个 block 后，先经过最终的 LayerNorm（ln_f），再乘以 <b>LM Head</b> 权重（3×48），
      把每一列 48 维向量变成 3 个数——分别是 A、B、C 的得分，称为 <b>logits</b>。</p>
      <p>Softmax 把任意实数变成概率分布：</p>
      <p class="formula">pᵢ = exp(zᵢ − max z) / Σⱼ exp(zⱼ − max z)</p>
      <p>减去最大值不改变结果，但能防止 exp 溢出（紫色板保存的就是每列的最大值和 1/Σexp）。
      输出越大的 logit，概率越接近 1。</p>`,
  },
  {
    id: 'output',
    title: '输出',
    en: 'Output',
    focus: any(named('Tokens'), (t) => t.group === 'output'),
    camFocus: (t) => t.group === 'output',
    sequence: ['Logits', 'SM Agg', 'Logits Softmax'],
    html: `
      <p>最后一块板的每一列，是"<b>下一个 token</b>"的概率分布：第 t 列预测的是第 t+1 个位置。</p>
      <p>我们取概率最高的字母（贪心解码），把它接到序列末尾，再整体跑一遍模型，预测下一个……这就是<b>自回归生成</b>。
      重复 6 次，就得到了排好序的完整输出。</p>
      <div id="output-panel"></div>
      <p>你可以在左上方修改输入字母并点击"运行"，所有方块都会按新的推理结果重新着色。</p>`,
  },
  {
    id: 'scale',
    title: '规模对比',
    en: 'Scale',
    focus: () => true,
    view: 'models',
    html: `
      <p>同样的结构，换上不同的尺寸，就是不同的模型。右侧在 nano-gpt 旁边按<b>真实比例</b>摆放了 GPT-2 small、GPT-2 XL 和 GPT-3（每个张量用一个实心盒子表示）：</p>
      <table class="models">
        <tr><th>模型</th><th>C</th><th>头数</th><th>层数</th><th>T</th><th>参数量</th></tr>
        <tr><td>nano-gpt</td><td>48</td><td>3</td><td>3</td><td>11</td><td>85,584</td></tr>
        <tr><td>GPT-2 small</td><td>768</td><td>12</td><td>12</td><td>1024</td><td id="p-gpt2s"></td></tr>
        <tr><td>GPT-2 XL</td><td>1600</td><td>25</td><td>48</td><td>1024</td><td id="p-gpt2xl"></td></tr>
        <tr><td>GPT-3</td><td>12288</td><td>96</td><td>96</td><td>1024*</td><td id="p-gpt3"></td></tr>
      </table>
      <p class="hint">* GPT-3 真实上下文长度为 2048，这里与原站一致按 1024 绘制。参数量按 minGPT 口径计算（不含 lm_head）。点击顶部的模型名可以飞到对应模型。</p>`,
  },
];
