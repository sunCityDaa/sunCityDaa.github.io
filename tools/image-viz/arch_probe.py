"""arch_probe.py — 在真实推理中探测生图模型的架构与中间量（供可视化使用）。

加载方式与 /pretrain/zhuxinxin/work/model_server/adapters.py 一致：
  qwen : QwenImage21Pipeline.from_pretrained(...)
  flux : ModularPipeline.from_pretrained(...)，移除 prompt_upsampler，只加载扩散组件

用法（用 CUDA_VISIBLE_DEVICES 选卡）：
  CUDA_VISIBLE_DEVICES=5 python arch_probe.py --model qwen --out runs/qwen
  CUDA_VISIBLE_DEVICES=3 python arch_probe.py --model flux --out runs/flux

输出：
  summary.json            组件参数量、配置、张量分组、首次调用各层输入/输出形状、
                          各步 timestep、各步各 block 输出 RMS、耗时、显存峰值
  <comp>_modules.txt      str(module)，完整模块树
  <comp>_tensors.json     每个参数张量 [名字, 形状, dtype, 元素数]
  token_maps.npz          选定步上每个 block 的图像 token RMS 热图 (h, w)
  previews/               每步 x_t 与 x0 估计的潜变量 → RGB 预览（线性映射拟合自最终图像）
  final.png               最终图像
"""
import argparse
import collections
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

QWEN_PATH = "/pretrain/zhuxinxin/model/Qwen-Image-2.1"
FLUX_PATH = "/pretrain/zhuxinxin/model/flux-3-image-klein"
DEFAULT_PROMPT = (
    "A photorealistic close-up of a red fox sitting in a snowy pine forest at sunrise, warm golden light on its fur, "
    "breath visible in the cold air, holding a small wooden sign in its paws that reads \"LLM VIZ\" in white painted "
    "letters, shallow depth of field, soft bokeh of snow-covered trees in the background."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["qwen", "flux"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default=DEFAULT_PROMPT)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

out = Path(args.out)
(out / "previews").mkdir(parents=True, exist_ok=True)
dev = "cuda:0"
dtype = torch.bfloat16
PATCH = 16  # 两个模型都是 1 个图像 token = 16x16 像素
H_TOK = W_TOK = args.size // PATCH
N_IMG = H_TOK * W_TOK


def sync():
    torch.cuda.synchronize(dev)


timings = collections.defaultdict(list)
t_load0 = time.time()

# ---------------------------------------------------------------- 1. 加载（同 adapters.py）
if args.model == "qwen":
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(dev)
    tokenizer = pipe.processor.tokenizer
    block_patterns = [r"img_in", r"txt_in", r"time_text_embed", r"modulation", r"transformer_blocks\.\d+",
                      r"norm_out", r"proj_out"]
else:
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(FLUX_PATH)
    pipe._blocks.sub_blocks.pop("prompt_upsampler", None)
    names = [
        n for n, s in pipe._component_specs.items()
        if s.default_creation_method == "from_pretrained" and s.pretrained_model_name_or_path is not None
        and n not in ("prompt_upsampler", "prompt_upsampler_processor")
    ]
    pipe.load_components(names=names, dtype=dtype)
    pipe = pipe.to(dev)
    tok = pipe.tokenizer
    tokenizer = getattr(tok, "tokenizer", tok)
    block_patterns = [r"x_embedders\.image", r"context_embedder", r"timestep_embedder", r"target_timestep_embedder",
                      r"txt_mode_blocks\.\d+", r"mode_blocks\.image\.\d+", r"single_transformer_blocks\.\d+",
                      r"norm_out", r"proj_out"]

comps = {"text_encoder": pipe.text_encoder, "transformer": pipe.transformer, "vae": pipe.vae}
sync()
load_seconds = time.time() - t_load0


# ---------------------------------------------------------------- 2. 静态架构清单
def cfg_dict(m):
    c = getattr(m, "config", None)
    if c is None:
        return {}
    if hasattr(c, "to_dict"):
        return c.to_dict()
    return dict(c)


arch = {}
for cname, m in comps.items():
    rows = [[n, list(p.shape), str(p.dtype).replace("torch.", ""), p.numel()] for n, p in m.named_parameters()]
    groups = collections.OrderedDict()
    for n, s, d, c in rows:
        key = re.sub(r"\.\d+(?=\.|$)", ".N", n)
        g = groups.setdefault(key, {"count": 0, "params": 0, "shape": s, "dtype": d})
        g["count"] += 1
        g["params"] += c
    arch[cname] = {
        "class": type(m).__name__,
        "params": sum(r[3] for r in rows),
        "n_tensors": len(rows),
        "config": cfg_dict(m),
        "groups": groups,
    }
    (out / f"{cname}_modules.txt").write_text(str(m), encoding="utf-8")
    (out / f"{cname}_tensors.json").write_text(json.dumps(rows), encoding="utf-8")


# ---------------------------------------------------------------- 3. 运行时钩子
def shape_of(x):
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        s = [shape_of(y) for y in x]
        return s if any(v is not None for v in s) else None
    return None


def rms(x):
    return float(x.detach().float().pow(2).mean().sqrt())


T = comps["transformer"]
state = {"step": -1, "last_t": None, "call_in_step": 0}
step_records = []          # 每步：timestep、各模块输出 RMS
first_shapes = collections.OrderedDict()
map_steps = set()          # 稍后按总步数决定
token_maps = {}            # f"s{step}/{module}" -> (H_TOK, W_TOK)
latents_xt, latents_x0 = [], []
compiled = [re.compile("^" + p + "$") for p in block_patterns]


def image_part(name, output):
    """返回模块输出中属于目标图像 token 的部分 (N_IMG, C)，取不到返回 None。"""
    if args.model == "qwen":
        if name.startswith("transformer_blocks") and torch.is_tensor(output) and output.shape[1] >= N_IMG:
            return output[0, -N_IMG:]
    else:
        if name.startswith("mode_blocks.image") and torch.is_tensor(output):
            return output[0, :N_IMG]
        if name.startswith("single_transformer_blocks") and isinstance(output, tuple):
            return output[1][0, :N_IMG]
    return None


def make_hook(name):
    def hook(module, a, kw, output):
        if state["call_in_step"] != 1:  # CFG 时同一步会调用两次，只记录第一次（条件分支）
            return
        if state["step"] == 0 and name not in first_shapes:
            first_shapes[name] = {
                "class": type(module).__name__,
                "in": [shape_of(x) for x in a] + [{k: shape_of(v)} for k, v in kw.items() if shape_of(v) is not None],
                "out": shape_of(output),
            }
        rec = step_records[-1]["rms"]
        if torch.is_tensor(output):
            rec[name] = rms(output)
        elif isinstance(output, tuple):
            rec[name] = [rms(o) for o in output if torch.is_tensor(o)]
        if state["step"] in map_steps:
            part = image_part(name, output)
            if part is not None:
                m = part.float().pow(2).mean(-1).sqrt().reshape(H_TOK, W_TOK)
                token_maps[f"s{state['step']}/{name}"] = m.cpu().numpy().astype(np.float16)
    return hook


hooked = []
for name, mod in T.named_modules():
    if name and any(c.match(name) for c in compiled):
        mod.register_forward_hook(make_hook(name), with_kwargs=True)
        hooked.append(name)


def t_pre(module, a, kw):
    t = kw.get("timestep")
    tval = float(t.flatten()[0]) if torch.is_tensor(t) else None
    if tval != state["last_t"]:
        state["step"] += 1
        state["last_t"] = tval
        state["call_in_step"] = 0
        rec = {"step": state["step"], "t": tval, "rms": {}}
        tt = kw.get("target_timestep")
        if torch.is_tensor(tt):
            rec["target_t"] = float(tt.flatten()[0])
        step_records.append(rec)
    state["call_in_step"] += 1
    if state["call_in_step"] == 1:
        if state["step"] == 0:
            first_shapes["__transformer_inputs__"] = {k: shape_of(v) for k, v in kw.items() if shape_of(v) is not None}
        state["x_in"] = kw["hidden_states"][0, -N_IMG:].detach().float()
    sync()
    state["t0"] = time.time()


def t_post(module, a, kw, output):
    sync()
    timings["transformer_call"].append(time.time() - state["t0"])
    if state["call_in_step"] != 1:
        return
    v = (output[0] if isinstance(output, tuple) else output.sample)[0, -N_IMG:].detach().float()
    t =step_records[-1]["t"]
    x = state["x_in"]
    latents_xt.append(x.cpu())
    latents_x0.append((x - t * v).cpu())  # flow matching：x_t = (1-t)·x0 + t·ε，v = ε − x0


T.register_forward_pre_hook(t_pre, with_kwargs=True)
T.register_forward_hook(t_post, with_kwargs=True)

te = comps["text_encoder"]


def te_pre(m, a, kw):
    sync()
    state["te0"] = time.time()


def te_post(m, a, kw, o):
    sync()
    timings["text_encoder"].append(time.time() - state["te0"])


te.register_forward_pre_hook(te_pre, with_kwargs=True)
te.register_forward_hook(te_post, with_kwargs=True)
vae = comps["vae"]
_orig_decode = vae.decode


def timed_decode(*a, **kw):
    sync()
    t0 = time.time()
    r = _orig_decode(*a, **kw)
    sync()
    timings["vae_decode"].append(time.time() - t0)
    return r


vae.decode = timed_decode

# ---------------------------------------------------------------- 4. 运行
steps = args.steps or (40 if args.model == "qwen" else 16)
map_steps.update({0, steps // 2, steps - 1})
gen = torch.Generator(device=dev).manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats(dev)
sync()
t_run0 = time.time()
with torch.no_grad():
    if args.model == "qwen":
        image = pipe(prompt=args.prompt, height=args.size, width=args.size, num_inference_steps=steps,
                     generator=gen).images[0]
    else:
        kw = dict(prompt=args.prompt, generator=gen, output="images", height=args.size, width=args.size)
        if args.steps:
            kw["num_inference_steps"] = steps
        res = pipe(**kw)
        image = res[0] if isinstance(res, (list, tuple)) else (res.images[0] if hasattr(res, "images") else res["images"][0])
sync()
run_seconds = time.time() - t_run0
image.save(out / "final.png")

# ---------------------------------------------------------------- 5. 潜变量 → RGB 预览
# 用最终 x0 估计与缩小到 token 网格的最终图像做最小二乘，得到 C→RGB 的线性映射，
# 再套到每一步的 x_t / x0 上。只是直观预览，不是 VAE 解码。
small = np.asarray(image.convert("RGB").resize((W_TOK, H_TOK), Image.BOX), dtype=np.float32) / 255.0
X = latents_x0[-1].numpy()
Xa = np.concatenate([X, np.ones((X.shape[0], 1), np.float32)], 1)
W, *_ = np.linalg.lstsq(Xa, small.reshape(-1, 3), rcond=None)
fit_err = float(np.abs(Xa @ W - small.reshape(-1, 3)).mean())


def to_rgb(lat):
    L = lat.numpy()
    rgb = np.concatenate([L, np.ones((L.shape[0], 1), np.float32)], 1) @ W
    rgb = np.clip(rgb.reshape(H_TOK, W_TOK, 3), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_TOK * 4, H_TOK * 4), Image.NEAREST)


for i, (xt, x0) in enumerate(zip(latents_xt, latents_x0)):
    to_rgb(xt).save(out / "previews" / f"xt_{i:02d}.png")
    to_rgb(x0).save(out / "previews" / f"x0_{i:02d}.png")
np.savez_compressed(out / "token_maps.npz", **token_maps)
torch.save({"xt": torch.stack(latents_xt).half(), "x0": torch.stack(latents_x0).half()}, out / "latents.pt")

# ---------------------------------------------------------------- 6. 汇总
tokens = tokenizer.tokenize(args.prompt)
summary = {
    "model": args.model,
    "prompt": args.prompt,
    "prompt_tokens": tokens,
    "size": args.size,
    "image_tokens": [H_TOK, W_TOK],
    "steps_requested": steps,
    "steps_observed": len(step_records),
    "transformer_calls": len(timings["transformer_call"]),
    "seed": args.seed,
    "load_seconds": round(load_seconds, 1),
    "run_seconds": round(run_seconds, 2),
    "timings": {k: [round(x, 4) for x in v] for k, v in timings.items()},
    "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
    "gpu": torch.cuda.get_device_name(dev),
    "versions": {"torch": torch.__version__, "diffusers": __import__("diffusers").__version__,
                 "transformers": __import__("transformers").__version__},
    "pipeline_class": type(pipe).__name__,
    "components": arch,
    "hooked_modules": hooked,
    "first_call_shapes": first_shapes,
    "steps": step_records,
    "preview_fit_mae": round(fit_err, 4),
    "token_map_keys": sorted(token_maps),
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["model", "steps_observed", "transformer_calls", "run_seconds",
                                           "peak_mem_gb", "preview_fit_mae"]}, ensure_ascii=False))
print({c: (a["class"], a["params"]) for c, a in arch.items()})
