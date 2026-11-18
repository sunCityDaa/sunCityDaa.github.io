# 图像生成可视化（image-viz）

在线地址：<https://sunCityDaa.github.io/image-viz/>　·　姊妹篇：[LLM 可视化](/llm-viz/)　·　技术报告：<https://sunCityDaa.github.io/posts/image-viz/>

一个用 **A100 上真实推理导出的数据**驱动的文生图模型 3D 可视化：把 Qwen-Image-2.1（16.2B）和 FLUX.3 Image Klein（10.1B）走完一次 1024×1024 出图的每一步、每一层，都画成 WebGL2 里的立方体方块。

- **不是示意图**：格子的高度和颜色来自真实前向传播中每个图像 token 的激活 RMS；注意力热图是用模型自己的 QK-Norm + RoPE 复算 softmax 得到的；左侧「x₀ 预测」是每步的速度场真的送进 VAE 解码出来的图。
- **可逐层、逐步、逐词查看**：拖动时间轴看 40 步（Qwen）/ 16 步（FLUX）的去噪；注意力章节可以选层、选步、选词，看到 `LLM`/`VIZ` 落在牌子上、`fox` 落在狐狸身上。
- **13 章中文讲解**：总览 → 提示词与 token → 文本编码器 → 噪声与潜变量 → 时间步与调制 → Transformer block → **权重矩阵** → **模型级模块** → 残差流里的图像 → 词到图像区域 → 一步步去噪 → VAE 解码 → 两个模型对比。
- **看得到模型本身**：把一层 block 拆成它的全部权重矩阵，从正上方画出来——长宽比 = 列数:行数，所以**平板面积 ∝ 参数量**。q/k/v/o 四个 4096² 正好相等，三个 SwiGLU 矩阵各是 3 倍面积；模型级的嵌入、投影、调制作业同样展开。
- **零依赖、零构建**：一个 `index.html` + 5 个 ES 模块 + 一份导出的数据目录。

数据里**不包含任何模型权重**，只包含生成结果、形状/参数量等结构信息，以及激活与注意力的统计量。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 页面骨架：顶栏（模型切换）、左侧讲解面板（提示词 / 三张缩略图 / 章节 / 播放条）、右侧 3D 视图 |
| `style.css` | 样式；窄屏（< 900px）改为上下布局 |
| `js/gl-util.js` | 矩阵与轨道相机；`frame()` 按相机角度和 FOV 计算包围盒的适配距离 |
| `js/scene.js` | 把 `meta.json` 展开成场景：文本编码器 36 层、每个 Transformer block 一块 64×64 的格子板、注意力头柱状图、按参数量缩放的权重板、输出 v、VAE、图像平面；以及场景外的**架构区**（每个权重矩阵一块平板）。还提供逐帧标量数据与"最近的文本 token"查找 |
| `js/renderer.js` | WebGL2 实例化渲染：每个格子一个立方体实例，顶点着色器从三张纹理取值（激活图集 R8、标量 R32F、每组的 64×64 状态纹理）决定高度与颜色；光照调色板 + 纹理四边形 |
| `js/walkthrough.js` | 11 个章节：标题、中文正文、高亮哪些组、相机对准哪里、当前步与模式 |
| `js/main.js` | 组装：加载 `meta.json` 与全部图片 → 建场景 → 渲染；章节导航、相机控制、悬停拾取、播放 |
| `data/qwen/` | Qwen-Image-2.1 的导出数据（8.2 MB） |
| `data/flux/` | FLUX.3 Image Klein 的导出数据（4.5 MB） |
| `LICENSE-THIRD-PARTY.md` | 两个上游模型的许可证与声明 |

## 数据目录里有什么

`data/<model>/`：

| 文件 | 说明 |
|---|---|
| `meta.json` | 结构、形状、参数量、时间表、逐步逐层统计、分词后的词表、文件索引、性能实测；`arch` 字段是每个子模块的真实名字/形状/参数量 |
| `final.jpg` | 最终图像（768²，由最终潜变量经 VAE 真实解码） |
| `steps/x0_XX.jpg` | 每一步「模型此刻认为的最终图」：`x0 = x_t − t·v`，**真的送进 VAE 解码**（256²） |
| `steps/xt_XX.jpg` | 每一步当前潜变量 `x_t` 的线性 RGB 投影（256²，仅示意，不是解码结果） |
| `act/act_XX.png` | 该步的激活图集：8 列、每格 64×64 的 8 位灰度图，tile 顺序是 `["x_t", …所有 block…, "v"]`（Qwen 34 个，FLUX 35 个） |
| `attn/s{SS}_b{BB}.png` | 选定步/层的「词 → 图像区域」注意力：每个词一条 64×64，横向拼接 |

`meta.json` 的 `cap_steps` / `cap_blocks` 是导出时抓取的步与层（Qwen 5 步 × 5 层，FLUX 5 步 × 5 层），时间轴和层选择器只在这些点上切换，其余位置自动吸附到最近的一个。

## 本地运行

```bash
cd static && python3 -m http.server 8000
# 打开 http://localhost:8000/image-viz/          默认 Qwen-Image-2.1
# 打开 http://localhost:8000/image-viz/?model=flux  FLUX.3 Klein
```

必须通过 HTTP 访问（`file://` 下 ES 模块与 `fetch` 会被浏览器拦截），浏览器需支持 WebGL2。

## 重新生成数据

数据由 `tools/image-viz/` 里的两个脚本在 A100 上产出。加载方式与 `/pretrain/zhuxinxin/work/model_server/adapters.py` 中的 `Qwenimage21`、`FLUX3KleinAdapter` 一致；FLUX 按 `infer.sh direct` 的做法跳过 PUP。

```bash
PY=/pretrain/zhuxinxin/work/envs/env312/bin/python
cd /pretrain/zhuxinxin/work/imgviz_arch

# 1) 结构探针：跑一次真实推理，记录每个模块的张量名字/形状、耗时、显存、逐步统计
CUDA_VISIBLE_DEVICES=5 $PY arch_probe.py  --model qwen --out runs/qwen
CUDA_VISIBLE_DEVICES=4 $PY arch_probe.py  --model flux --out runs/flux

# 2) 导出网页数据：每步的 x0 / x_t、每个 block 的激活、词→区域注意力、VAE 解码校验
CUDA_VISIBLE_DEVICES=5 $PY export_web.py --model qwen --out web/qwen --probe runs/qwen/summary.json
CUDA_VISIBLE_DEVICES=4 $PY export_web.py --model flux --out web/flux --probe runs/flux/summary.json

# 3) （可选）生成报告用的拼图与热图
$PY montage.py runs/qwen; $PY montage.py runs/flux

# 4) 把张量表整理成「架构」数据写进 meta.json（本机即可，不需要模型）
python3 tools/image-viz/export_arch.py --runs runs --data static/image-viz/data
```

把 `web/qwen`、`web/flux` 整个目录复制到 `static/image-viz/data/` 即可。

导出 Qwen 时脚本会**关闭前缀 KV 缓存**（`use_kv_cache=False`）——数值完全不变，只是每步要多算一遍 79 个文本 token，所以导出耗时 28.6 s 比探针的 16.2 s 长。

## 校验

`meta.json → perf.last_decode_mae` 是把导出的最终 `x0` 再送进 VAE 解码、与 pipeline 自己产出的图像比较的平均绝对误差（0–255 尺度）：

- Qwen-Image-2.1：**0.299**
- FLUX.3 Klein：**0.265**

即 <0.12%，说明我们抓到的潜变量和最终图像是对得上的。

## 调试接口

页面加载后，浏览器控制台里有 `window.__imgviz`：

```js
__imgviz.gotoChapter(7)      // 跳到第 8 章「词 → 图像区域」（从 0 开始）
__imgviz.setStep(20)         // 切到第 20 步（会吸附到最近的抓取步）
__imgviz.app.meta.blocks     // 当前模型每一层的参数与形状
__imgviz.CHAPTERS.map(c => c.title)
```

## 操作

拖动旋转 · 右键/Shift+拖动平移 · 滚轮缩放（朝鼠标方向）· 双指捏合缩放 · 悬停查看数值 · ← → 切换章节 · 空格播放去噪。

## 已知的局限

1. **`xt_XX.jpg` 只是示意**：潜变量有 64（Qwen）或 128（FLUX）个通道，无法直接显示。这里用一个最小二乘拟合的线性映射投到 RGB（以最终图像缩到 64×64 为目标），平均误差 Qwen 0.024 / FLUX 0.053（0–1 尺度），能看构图但颜色不准。要看"模型认为的图"请用 `x0_XX.jpg`，那是真的 VAE 解码。
2. **激活图每格单独归一化**：颜色只表示同一层内 4096 个图像 token 之间的相对强弱，不能跨层比较绝对大小。要看绝对量级请用章节里的逐步 RMS 曲线。
3. **文本 token 的激活只抓了第一步和最后一步**：文本 token 用 `t=0` 调制，与去噪步无关，中间步是插值显示的。
4. **注意力只抓了 5 步 × 5 层**：全量导出是 40 步 × 32 层 × 32 头 × 4175 个 token，体积不可接受。
5. **方块高度做了截断与压缩**：`hmax` 与 `pow(val, k)` 让 4096 个格子在一个屏幕上可读；没有任何一个维度是严格线性的。

## 版权与许可

- **Qwen-Image-2.1**：Qwen Research License Agreement（仅限非商业的研究/评估用途）。本页展示的是其生成结果与统计量，**不包含权重**。
- **FLUX.3 Image Klein**：FLUX Kommunity License。同上，**不包含权重**。BFL 声明不对输出主张所有权；模型输出不构成其衍生作品。
- 本站为非商业的个人博客，页内所有图像均为 AI 生成，即使与真实照片相似也非真实影像。

完整声明见 [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md)。
