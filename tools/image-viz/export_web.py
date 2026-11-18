"""export_web.py — 为 image-viz 网页导出真实推理数据（在 a100 上运行）。

加载方式同 model_server/adapters.py；为了拿到每一步完整的「图像 → 文本」注意力，Qwen 关闭了前缀 KV 缓存
（use_kv_cache=False，数值结果不变，只是每步多算一遍文本 token）。

  CUDA_VISIBLE_DEVICES=5 python export_web.py --model qwen --out web/qwen
  CUDA_VISIBLE_DEVICES=4 python export_web.py --model flux --out web/flux

输出（全部是可公开的统计量与生成结果，不含任何权重）：
  meta.json          结构、形状、参数量、时间表、逐步逐层统计、词表、文件索引
  final.jpg          最终图像（768²）
  steps/x0_XX.jpg    每步「预测的最终图」：x0 = x_t − t·v，经 VAE 真实解码（256²）
  steps/xt_XX.jpg    每步当前潜变量 x_t 的线性 RGB 投影（256²，示意）
  act/act_XX.png     每步激活图集：每个 tile 是某层输出在 64×64 个图像 token 上的 RMS（8 位灰度）
  attn/s{S}_b{B}.png 选定步/层的「词 → 图像区域」注意力（每个词一张 64×64，横向拼接）
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

QWEN_PATH = "/pretrain/zhuxinxin/model/Qwen-Image-2.1"
FLUX_PATH = "/pretrain/zhuxinxin/model/flux-3-image-klein"
PROMPT = (
    "A photorealistic close-up of a red fox sitting in a snowy pine forest at sunrise, warm golden light on its fur, "
    "breath visible in the cold air, holding a small wooden sign in its paws that reads \"LLM VIZ\" in white painted "
    "letters, shallow depth of field, soft bokeh of snow-covered trees in the background."
)
WORDS = ["red", "fox", "snowy", "pine", "forest", "sunrise", "golden", "fur", "breath", "wooden", "sign", "paws",
         "LLM", "VIZ", "white", "letters", "bokeh", "trees", "background"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["qwen", "flux"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--probe", default=None, help="arch_probe 的 summary.json（取真实耗时/显存）")
args = ap.parse_args()

out = Path(args.out)
for d in ("steps", "act", "attn"):
    (out / d).mkdir(parents=True, exist_ok=True)
dev, dtype = "cuda:0", torch.bfloat16
G = args.size // 16           # 图像 token 网格边长（两个模型都是 16 像素一个 token）
N_IMG = G * G
QSTRIDE = 8                   # 统计「注意文本的比例」时每隔 8 个图像 query 取一个


def sync():
    torch.cuda.synchronize(dev)


t_load = time.time()
if args.model == "qwen":
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    pipe = QwenImage21Pipeline.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(dev)
    tokenizer = pipe.processor.tokenizer
    T = pipe.transformer
    img_blocks = [(f"transformer_blocks.{i}", b) for i, b in enumerate(T.transformer_blocks)]
    txt_blocks = []           # 单流：文本 token 与图像 token 走同一批 block
    joint_blocks = img_blocks
    STEPS = 40
    CAP_STEPS = [0, 5, 10, 20, 39]
    CAP_BLOCKS = [0, 8, 16, 24, 31]
else:
    from diffusers import ModularPipeline
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux3 import _get_qkv_projections

    pipe = ModularPipeline.from_pretrained(FLUX_PATH)
    pipe._blocks.sub_blocks.pop("prompt_upsampler", None)
    names = [n for n, s in pipe._component_specs.items()
             if s.default_creation_method == "from_pretrained" and s.pretrained_model_name_or_path is not None
             and n not in ("prompt_upsampler", "prompt_upsampler_processor")]
    pipe.load_components(names=names, dtype=dtype)
    pipe = pipe.to(dev)
    tok = pipe.tokenizer
    tokenizer = getattr(tok, "tokenizer", tok)
    T = pipe.transformer
    txt_blocks = [(f"txt_mode_blocks.{i}", b) for i, b in enumerate(T.txt_mode_blocks)]
    img_blocks = [(f"mode_blocks.image.{i}", b) for i, b in enumerate(T.mode_blocks["image"])]
    joint_blocks = [(f"single_transformer_blocks.{i}", b) for i, b in enumerate(T.single_transformer_blocks)]
    img_blocks = img_blocks + joint_blocks
    STEPS = 16
    CAP_STEPS = [0, 2, 5, 10, 15]
    CAP_BLOCKS = [0, 7, 14, 21, 27]
sync()
load_seconds = time.time() - t_load
TE, VAE = pipe.text_encoder, pipe.vae
H_DIM = T.config.attention_head_dim
N_HEADS = T.config.num_attention_heads

# ------------------------------------------------------------------ 运行时状态
S = {"step": -1, "last_t": None, "first": True}
steps = []                     # 每步：t、target_t、各层统计
x0_lat, xt_lat = [], []
tiles = ["x_t"] + [n for n, _ in img_blocks] + ["v"]
tile_maps = {}                 # 当前步：tile 名 -> (G,G) float
txt_rms_tok = {}               # f"{step}/{block}" -> 每个文本 token 的 RMS（只存选定步）
TXT_TOKEN_STEPS = None         # 运行前确定
shapes = {}
captured_ids = {}
te_layer_rms = []


def mtx_rms(x):
    return x.float().pow(2).mean(-1).sqrt()


def first_shapes(name, module, a, kw, o):
    if name in shapes:
        return

    def sh(x):
        if torch.is_tensor(x):
            return list(x.shape)
        if isinstance(x, (tuple, list)):
            r = [sh(y) for y in x]
            return r if any(v is not None for v in r) else None
        return None

    shapes[name] = {"class": type(module).__name__, "in": [sh(x) for x in a] +
                    [{k: sh(v)} for k, v in kw.items() if sh(v) is not None], "out": sh(o)}


# ---- 文本编码器：记录 token id 与逐层每个 token 的 RMS
def te_pre(m, a, kw):
    ids = kw.get("input_ids", a[0] if a else None)
    if ids is not None and "ids" not in captured_ids:
        captured_ids["ids"] = ids[0].tolist()


TE.register_forward_pre_hook(te_pre, with_kwargs=True)
lm = getattr(TE.model, "language_model", TE.model)
for li, layer in enumerate(lm.layers):
    def mk(li):
        def h(m, a, o):
            if len(te_layer_rms) < len(lm.layers):
                hs = o[0] if isinstance(o, tuple) else o
                te_layer_rms.append(mtx_rms(hs[0]).cpu().numpy())
        return h
    layer.register_forward_hook(mk(li))


# ---- Transformer：按 timestep 变化切分步数
def t_pre(m, a, kw):
    t = kw.get("timestep")
    tval = float(t.flatten()[0])
    if tval != S["last_t"]:
        S["step"] += 1
        S["last_t"] = tval
        rec = {"t": round(tval, 5), "img_rms": {}, "txt_rms": {}, "txt_frac": {}, "head_frac": {}}
        tt = kw.get("target_timestep")
        if torch.is_tensor(tt):
            rec["target_t"] = round(float(tt.flatten()[0]), 5)
        steps.append(rec)
        tile_maps.clear()
    S["x_in"] = kw["hidden_states"][0, -N_IMG:].detach().float()
    tile_maps["x_t"] = mtx_rms(S["x_in"]).reshape(G, G).cpu().numpy()
    if S["first"]:
        S["txt_len"] = kw["encoder_hidden_states"].shape[1]
        shapes["__inputs__"] = {k: list(v.shape) for k, v in kw.items() if torch.is_tensor(v)}


def t_post(m, a, kw, o):
    S["first"] = False
    v = (o[0] if isinstance(o, tuple) else o.sample)[0, -N_IMG:].detach().float()
    t = steps[-1]["t"]
    xt_lat.append(S["x_in"].cpu())
    x0_lat.append((S["x_in"] - t * v).cpu())
    tile_maps["v"] = mtx_rms(v).reshape(G, G).cpu().numpy()
    save_atlas(S["step"])


T.register_forward_pre_hook(t_pre, with_kwargs=True)
T.register_forward_hook(t_post, with_kwargs=True)


def img_part(name, o):
    if args.model == "qwen":
        return o[0, -N_IMG:]
    if name.startswith("single"):
        return o[1][0, :N_IMG]
    return o[0, :N_IMG]


def txt_part(name, o):
    if args.model == "qwen":
        return o[0, :S["txt_len"]]
    if name.startswith("single"):
        return o[0][0]
    if name.startswith("txt_mode"):
        return o[0]
    return None


def block_hook(name):
    def h(m, a, kw, o):
        first_shapes(name, m, a, kw, o)
        rec = steps[-1]
        if not name.startswith("txt_mode"):
            r = mtx_rms(img_part(name, o))
            tile_maps[name] = r.reshape(G, G).cpu().numpy()
            rec["img_rms"][name] = round(float(r.mean()), 3)
        tp = txt_part(name, o)
        if tp is not None:
            r = mtx_rms(tp)
            rec["txt_rms"][name] = round(float(r.mean()), 3)
            if S["step"] in TXT_TOKEN_STEPS:
                txt_rms_tok[f"{S['step']}/{name}"] = [round(float(x), 2) for x in r.cpu()]
    return h


for name, b in txt_blocks + img_blocks:
    b.register_forward_hook(block_hook(name), with_kwargs=True)

for name in (["img_in", "txt_in", "time_text_embed", "modulation", "norm_out", "proj_out"] if args.model == "qwen"
             else ["x_embedders.image", "context_embedder", "timestep_embedder", "target_timestep_embedder",
                   "norm_out", "proj_out"]):
    mod = T.get_submodule(name)
    mod.register_forward_hook(lambda m, a, kw, o, n=name: first_shapes(n, m, a, kw, o), with_kwargs=True)

# ---- 注意力：按模型自己的公式重算 q/k（含 QK-Norm 与 RoPE），统计图像 query 对文本 key 的注意力
word_maps = {}


def qk_of(attn, kw):
    h = kw["hidden_states"]
    if args.model == "qwen":
        rot = kw["rotary_emb"]
        q = attn.norm_q(attn.to_q(h).unflatten(-1, (-1, H_DIM))).to(h.dtype)
        k = attn.norm_k(attn.to_k(h).unflatten(-1, (-1, H_DIM))).to(h.dtype)
        q = apply_rotary_emb_qwen(q, rot, use_real=False)
        k = apply_rotary_emb_qwen(k, rot, use_real=False)
    else:
        rot = kw["image_rotary_emb"]
        q, k, _ = _get_qkv_projections(attn, h)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)).float())
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)).float())
        q = apply_rotary_emb(q, rot, sequence_dim=1).to(h.dtype)
        k = apply_rotary_emb(k, rot, sequence_dim=1).to(h.dtype)
    return q[0], k[0]          # [S, H, D]


def attn_hook(bi, name):
    def h(attn, a, kw, o):
        if kw.get("hidden_states") is None:
            return
        n_txt = S["txt_len"]
        q, k = qk_of(attn, kw)
        if args.model == "qwen":
            qi = q[-N_IMG:]
        else:
            qi = q[n_txt:n_txt + N_IMG]
        full = S["step"] in CAP_STEPS and bi in CAP_BLOCKS
        sel = qi if full else qi[::QSTRIDE]
        scale = 1.0 / math.sqrt(H_DIM)
        fr = []
        acc = torch.zeros(sel.shape[0], n_txt, device=q.device) if full else None
        for hd in range(q.shape[1]):
            p = torch.softmax((sel[:, hd].float() @ k[:, hd].float().T) * scale, dim=-1)
            pt = p[:, :n_txt]
            fr.append(float(pt.sum(-1).mean()))
            if full:
                acc += pt
        rec = steps[-1]
        rec["txt_frac"][name] = round(sum(fr) / len(fr), 4)
        rec["head_frac"][name] = [round(x, 3) for x in fr]
        if full:
            word_maps[(S["step"], bi)] = (acc / q.shape[1]).cpu()
    return h


for bi, (name, b) in enumerate(joint_blocks):
    b.attn.register_forward_hook(attn_hook(bi, name), with_kwargs=True)


# ---- 每步激活图集
COLS = 8


def save_atlas(step):
    rows = math.ceil(len(tiles) / COLS)
    img = np.zeros((rows * G, COLS * G), np.uint8)
    rng = {}
    for i, n in enumerate(tiles):
        m = tile_maps.get(n)
        if m is None:
            continue
        lo, hi = np.percentile(m, 1), np.percentile(m, 99)
        rng[n] = [round(float(lo), 3), round(float(hi), 3)]
        u = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
        r, c = divmod(i, COLS)
        img[r * G:(r + 1) * G, c * G:(c + 1) * G] = (u * 255).astype(np.uint8)
    Image.fromarray(img).save(out / "act" / f"act_{step:02d}.png", optimize=True)
    steps[-1]["tile_range"] = rng


# ------------------------------------------------------------------ 运行
TXT_TOKEN_STEPS = {0, STEPS - 1}
gen = torch.Generator(device=dev).manual_seed(args.seed)
sync()
t0 = time.time()
with torch.no_grad():
    if args.model == "qwen":
        image = pipe(prompt=PROMPT, height=args.size, width=args.size, num_inference_steps=STEPS, generator=gen,
                     use_kv_cache=False).images[0]
    else:
        res = pipe(prompt=PROMPT, generator=gen, output="images", height=args.size, width=args.size)
        image = res[0] if isinstance(res, (list, tuple)) else (res.images[0] if hasattr(res, "images") else res["images"][0])
sync()
run_seconds = time.time() - t0


def to_rgb_white(im):
    """Qwen-Image-2.1 的 VAE 输出 RGBA；与管线一致，把 alpha 合成到白底上。"""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


image = to_rgb_white(image)
assert len(steps) == STEPS, (len(steps), STEPS)
image.convert("RGB").resize((768, 768), Image.LANCZOS).save(out / "final.jpg", quality=90)


# ------------------------------------------------------------------ 每步 x0 的真实 VAE 解码
@torch.no_grad()
def decode(lat):
    lat = lat.to(dev, dtype)[None]
    if args.model == "qwen":
        x = pipe._unpack_latents(lat, args.size, args.size, pipe.vae_scale_factor).to(VAE.dtype)
        mean = torch.tensor(VAE.config.latents_mean).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
        std = torch.tensor(VAE.config.latents_std).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
        img = VAE.decode(x * std + mean, return_dict=False)[0][:, :, 0]
    else:
        x = lat.view(1, G, G, -1).permute(0, 3, 1, 2)
        bm = VAE.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
        bs = torch.sqrt(VAE.bn.running_var.view(1, -1, 1, 1) + VAE.config.batch_norm_eps).to(x.device, x.dtype)
        x = x * bs + bm
        b, c, hh, ww = x.shape
        x = x.reshape(b, c // 4, 2, 2, hh, ww).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, hh * 2, ww * 2)
        img = VAE.decode(x.to(VAE.dtype), return_dict=False)[0]
    img = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return to_rgb_white(Image.fromarray(img))


t_dec = time.time()
for i, lat in enumerate(x0_lat):
    im = decode(lat)
    if i == len(x0_lat) - 1:
        a = np.asarray(im.resize((256, 256)), np.float32)
        b = np.asarray(image.convert("RGB").resize((256, 256)), np.float32)
        last_decode_mae = float(np.abs(a - b).mean())
    im.resize((256, 256), Image.LANCZOS).save(out / "steps" / f"x0_{i:02d}.jpg", quality=86)
decode_seconds = (time.time() - t_dec) / len(x0_lat)

# x_t 的线性 RGB 投影（拟合：最后一步 x0 → 缩到 G×G 的最终图像）
small = np.asarray(image.convert("RGB").resize((G, G), Image.BOX), np.float32).reshape(-1, 3) / 255
X = np.concatenate([x0_lat[-1].numpy(), np.ones((N_IMG, 1), np.float32)], 1)
Wp, *_ = np.linalg.lstsq(X, small, rcond=None)
for i, lat in enumerate(xt_lat):
    L = np.concatenate([lat.numpy(), np.ones((N_IMG, 1), np.float32)], 1) @ Wp
    rgb = (np.clip(L.reshape(G, G, 3), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(out / "steps" / f"xt_{i:02d}.jpg", quality=86)

# ------------------------------------------------------------------ 词表与「词 → 区域」注意力
ids = captured_ids["ids"]
n_txt = S["txt_len"]
drop = len(ids) - n_txt if args.model == "qwen" else 0
raw = tokenizer.convert_ids_to_tokens(ids[drop:drop + n_txt])


def pretty(t):
    return t.replace("Ġ", " ").replace("Ċ", "\\n")


toks = [pretty(t) for t in raw]
# 把 token 拼成词：以空格开头、特殊符号或标点处断开
def is_word(t):
    return re.match(r"^[A-Za-z0-9\-]+$", t.strip()) is not None


words = []
for i, t in enumerate(toks):
    if i == 0 or t.startswith(" ") or not is_word(t) or not is_word(toks[i - 1]):
        words.append({"text": t.strip(), "tok": [i]})
    else:
        words[-1]["text"] += t
        words[-1]["tok"].append(i)
targets = []
for w in words:
    clean = w["text"].strip('"\'.,')
    if clean in WORDS and clean not in [x["text"] for x in targets]:
        targets.append({"text": clean, "tok": w["tok"]})
targets.insert(0, {"text": toks[0].strip() or "<首 token>", "tok": [0], "control": True})

attn_files = {}
for (st, bi), acc in word_maps.items():
    strip = np.zeros((G, G * len(targets)), np.uint8)
    stats = []
    for j, w in enumerate(targets):
        m = acc[:, w["tok"]].sum(-1).reshape(G, G).numpy()
        lo, hi = float(m.min()), float(np.percentile(m, 99.5))
        strip[:, j * G:(j + 1) * G] = (np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        stats.append([round(float(m.mean()), 5), round(float(m.max()), 4)])
    f = f"attn/s{st:02d}_b{bi:02d}.png"
    Image.fromarray(strip).save(out / f, optimize=True)
    attn_files[f"{st}/{bi}"] = {"file": f, "stats": stats}


# ------------------------------------------------------------------ 结构与参数
def params(m):
    return sum(p.numel() for p in m.parameters())


def blk(name, kind, m):
    return {"name": name, "kind": kind, "params": params(m), "shape": shapes.get(name)}


blocks = [blk(n, "txt", b) for n, b in txt_blocks]
blocks += [blk(n, "joint" if n.startswith(("single", "transformer_blocks")) else "img", b) for n, b in img_blocks]
if args.model == "flux":
    for i, b in enumerate(T.mode_blocks["image_cond"]):
        blocks.append({"name": f"mode_blocks.image_cond.{i}", "kind": "unused", "params": params(b), "shape": None})
io = {n: {"params": params(T.get_submodule(n)), "shape": shapes.get(n)} for n in shapes if n != "__inputs__"
      and not re.search(r"blocks\.", n)}
lm_cfg = TE.config.text_config if hasattr(TE.config, "text_config") else TE.config
probe = json.load(open(args.probe)) if args.probe and Path(args.probe).exists() else {}
meta = {
    "model": args.model,
    "title": "Qwen-Image-2.1" if args.model == "qwen" else "FLUX.3 Image Klein",
    "prompt": PROMPT, "size": args.size, "grid": G, "seed": args.seed,
    "steps_n": STEPS, "cap_steps": CAP_STEPS, "cap_blocks": CAP_BLOCKS, "txt_token_steps": sorted(TXT_TOKEN_STEPS),
    "tokens": toks, "dropped_system_tokens": drop, "words": targets,
    "heads": N_HEADS, "head_dim": H_DIM,
    "components": {
        "text_encoder": {"class": type(TE).__name__, "params": params(TE), "layers": lm_cfg.num_hidden_layers,
                         "hidden": lm_cfg.hidden_size, "ffn": lm_cfg.intermediate_size,
                         "heads": lm_cfg.num_attention_heads, "kv_heads": lm_cfg.num_key_value_heads},
        "transformer": {"class": type(T).__name__, "params": params(T), "config": dict(T.config)},
        "vae": {"class": type(VAE).__name__, "params": params(VAE), "config": dict(VAE.config)},
    },
    "te_layer_rms": [[round(float(x), 2) for x in r[drop:drop + n_txt]] for r in te_layer_rms],
    "blocks": blocks, "io": io, "inputs": shapes.get("__inputs__"),
    "tiles": tiles, "tile_cols": COLS,
    "steps": steps, "txt_rms_tok": txt_rms_tok, "attn": attn_files,
    "perf": {"run_seconds_export": round(run_seconds, 2), "load_seconds": round(load_seconds, 1),
             "decode_seconds": round(decode_seconds, 3), "last_decode_mae": round(last_decode_mae, 3),
             "probe": {k: probe.get(k) for k in ("run_seconds", "peak_mem_gb", "timings", "gpu", "versions")}},
}
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":"), default=str),
                               encoding="utf-8")
print(json.dumps({"model": args.model, "steps": len(steps), "tokens": len(toks), "words": [w["text"] for w in targets],
                  "run_s": round(run_seconds, 1), "decode_mae": round(last_decode_mae, 3),
                  "attn_maps": len(attn_files)}, ensure_ascii=False))
