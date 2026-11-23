"""export_weights.py — 把真实权重的抽样值导出成网页用的图集 + 统计（在 a100 上跑）。

每个矩阵取 64×64 个**真实数值**（行、列等距抽样），量化成 8 位灰度放进一张图集；
悬停时用 absmax 还原成真实值。同时记录每个矩阵的 mean/std/min/max 与直方图。

  python3 export_weights.py --model qwen --out web/qwen
"""
import argparse, json, math, os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BASE = "/pretrain/zhuxinxin/model"
ROOT = {"qwen": "Qwen-Image-2.1", "flux": "flux-3-image-klein"}
TILE = 64
COLS = 8


def open_shard(path):
    files = sorted(Path(path).glob("*.safetensors"))
    idx = Path(path) / "model.safetensors.index.json"
    mapping = None
    if idx.exists():
        mapping = json.loads(idx.read_text())["weight_map"]
    handles = {f: safe_open(str(f), framework="pt", device="cpu") for f in files}
    return handles, mapping


def read_tensor(handles, mapping, name):
    if mapping is not None:
        fname = mapping.get(name)
        if fname is None:
            return None
        return handles[Path(fname) if Path(fname) in handles else next(iter(handles))].get_tensor(name)
    for h in handles.values():
        try:
            return h.get_tensor(name)
        except Exception:
            continue
    return None


def sample64(t):
    """行、列等距抽 64 个点，返回 [64,64] float32。"""
    r, c = t.shape
    ri = np.linspace(0, r - 1, TILE).round().astype(np.int64)
    ci = np.linspace(0, c - 1, TILE).round().astype(np.int64)
    x = t.to(torch.float32).numpy() if hasattr(t, "numpy") else t
    return np.asarray(x)[np.ix_(ri, ci)].astype(np.float32)


def stats(t):
    x = t.to(torch.float32).reshape(-1).numpy()
    n = x.size
    mean = float(x.mean()); std = float(x.std())
    mx = float(x.max()); mn = float(x.min())
    idx = np.linspace(0, n - 1, min(n, 1 << 22)).round().astype(np.int64)
    hist, edges = np.histogram(x[idx], bins=24)
    return {"mean": mean, "std": std, "min": mn, "max": mx,
            "hist": [int(v) for v in hist], "edges": [round(float(e), 6) for e in edges]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "flux"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True, help="网页的 meta.json（从 arch 里拿矩阵名单）")
    a = ap.parse_args()

    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    arch = meta["arch"]
    root = Path(BASE) / ROOT[a.model]
    tdir = root / "transformer"
    handles, mapping = open_shard(tdir)

    block_prefix = {"qwen": "transformer_blocks.0.", "flux": "single_transformer_blocks.0."}[a.model]
    names = []
    for m in arch["block"]["modules"]:
        if m["kind"] == "w" and len(m["shape"]) == 2:
            names.append((block_prefix + m["name"] + ".weight", m["name"]))
    if a.model == "flux" and "block_mode" in arch:
        for m in arch["block_mode"]["modules"]:
            if m["kind"] == "w" and len(m["shape"]) == 2:
                names.append(("mode_blocks.image.0." + m["name"] + ".weight", "分流·" + m["name"]))
    for m in arch["io"]:
        if m["kind"] == "w" and len(m["shape"]) == 2:
            names.append((m["name"] + ".weight", m["name"]))

    extra = [
        (root / "text_encoder", "model.language_model.layers.0.self_attn.q_proj.weight", "文本编码器 第0层 q_proj"),
        (root / "vae", None, "VAE 解码器 conv_in"),
    ]

    tiles, atlas = [], []
    for full, short in names:
        t = read_tensor(handles, mapping, full)
        if t is None or t.dim() != 2:
            print("  skip", full); continue
        s = sample64(t)
        st = stats(t)
        tiles.append({"name": short, "full": full, "rows": int(t.shape[0]), "cols": int(t.shape[1]), **st})
        atlas.append(s)
        print(f"  {short:34s} {tuple(t.shape)} mean {st['mean']:+.4f} std {st['std']:.4f}")

    # 文本编码器 / VAE 各取一个代表矩阵
    for d, name, label in extra:
        h, mp = open_shard(d)
        if name is None:
            for hh in h.values():
                for k in hh.keys():
                    if k.endswith("conv_in.weight"):
                        name = k; break
                if name: break
        if not name:
            print("  skip extra", label); continue
        t = read_tensor(h, mp, name)
        if t is None or t.dim() < 2:
            print("  skip extra", label); continue
        t2 = t.reshape(t.shape[0], -1)
        s = sample64(t2)
        st = stats(t2)
        tiles.append({"name": label, "full": f"{d.name}/{name}", "rows": int(t2.shape[0]), "cols": int(t2.shape[1]), **st})
        atlas.append(s)
        print(f"  {label:34s} {tuple(t2.shape)} mean {st['mean']:+.4f} std {st['std']:.4f}")

    # 拼图集：每格 64×64，8 列
    n = len(atlas)
    rows = math.ceil(n / COLS)
    img = np.full((rows * TILE, COLS * TILE), 128, dtype=np.float32)
    for i, s in enumerate(atlas):
        amax = max(abs(tiles[i]["min"]), abs(tiles[i]["max"])) or 1e-8
        tiles[i]["absmax"] = float(amax)
        v = np.clip(128 + 127 * s / amax, 0, 255)
        r0, c0 = (i // COLS) * TILE, (i % COLS) * TILE
        img[r0:r0 + TILE, c0:c0 + TILE] = v.astype(np.uint8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(img.astype(np.uint8), mode="L").save(out / "w.png")
    (out / "w.json").write_text(json.dumps({"tile": TILE, "cols": COLS, "rows": rows, "tensors": tiles},
                                           ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{a.model}: {n} 个矩阵 → {out/'w.png'} ({(out/'w.png').stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
