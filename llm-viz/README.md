# LLM 可视化（复刻版）

在线地址：<https://sunCityDaa.github.io/llm-viz/>　·　原站：<https://bbycroft.net/llm>

一个在浏览器里**真实运行推理**的 GPT 3D 可视化：用纯 JavaScript 跑 nano-gpt（3 层、3 头、48 维、85,584 参数）的前向计算，把每一个权重和激活值画成 WebGL2 中的一个小方块，并配有 11 章中文讲解。

完整的逆向分析与复刻过程见博客文章：[`content/posts/llm-viz-rea/index.md`](../../content/posts/llm-viz-rea/index.md)（线上：<https://sunCityDaa.github.io/posts/llm-viz-rea/>）。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 页面骨架：顶栏（模型切换）、左侧讲解面板（输入/输出、章节、播放条）、右侧 3D 视图 |
| `style.css` | 样式；窄屏（< 820px）改为上下布局 |
| `js/gpt.js` | GPT 前向推理：解码 base64 float32 权重、LayerNorm、多头因果自注意力、GELU MLP、softmax、贪心自回归排序。浏览器与 Node 通用 |
| `js/layout.js` | 3D 布局：把 192 个张量排成"格子板"（每列一个 token、权重在左），`bindActs` 把推理结果绑定到每个格子；大模型用 `genSolidLayout` 画实心盒 |
| `js/renderer.js` | WebGL2 实例化渲染（每格一个立方体实例）、逐张量状态纹理（变暗/揭示进度/高亮）、轨道相机与飞行动画、投影与拾取射线 |
| `js/walkthrough.js` | 11 个章节：标题、中文正文、高亮哪些张量、相机对准哪些张量、动画的计算顺序 |
| `js/main.js` | 组装：加载权重 → 推理 → 布局 → 渲染；UI、鼠标/触摸交互、悬停提示、3D 标签、播放动画 |
| `data/gpt-nano-sort-model.json` | 训练好的 minGPT 排序模型权重（来自原项目，MIT） |
| `LICENSE-THIRD-PARTY.md` | 上游许可证全文 |

## 本地运行

```bash
cd static && python3 -m http.server 8000
# 打开 http://localhost:8000/llm-viz/
```

需要通过 HTTP 访问（`file://` 下 ES 模块无法加载），浏览器需支持 WebGL2。

## 校验推理

```bash
node tools/llm-viz/verify-gpt.mjs
```

用原站公开的 PyTorch 参考激活值（`tools/llm-viz/gpt-nano-sort-t0-partials.json`）对 3 条序列逐层比对，当前最大误差 1.9×10⁻⁶，全部通过。

## 调试接口

页面加载后，浏览器控制台中可用 `window.__llmviz`：

```js
__llmviz.setInput([2, 2, 1, 0, 0, 1])  // 0=A 1=B 2=C，修改输入并重新推理
__llmviz.gotoChapter(4)                // 跳转章节（从 0 开始）
__llmviz.app.gen.output                // 当前输出
__llmviz.app.nano.tensors              // 所有张量的布局信息
```

## 操作

拖动旋转 · 右键/Shift+拖动平移 · 滚轮缩放（朝鼠标方向）· 双指捏合缩放 · 悬停查看数值 · ← → 切换章节 · 空格播放动画。

## 致谢

- [LLM Visualization](https://github.com/bbycroft/llm-viz) © Brendan Bycroft，MIT —— 权重文件、模型结构与可视化思路。本目录代码为重新编写。
- [minGPT](https://github.com/karpathy/minGPT) © Andrej Karpathy，MIT。
