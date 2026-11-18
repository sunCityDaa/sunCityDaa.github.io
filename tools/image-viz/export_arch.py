"""export_arch.py — 把 arch_probe 导出的张量表整理成网页用的「架构」数据，合并进 meta.json。

输入：arch_probe.py 产出的 runs/{qwen,flux}/transformer_tensors.json
      （每个元素是 [名字, 形状, dtype, 元素数]，按首次出现去重）
输出：往 web 的 meta.json 里写一个 arch 字段：

  arch.transformer_class / transformer_params / n_tensors
  arch.block     = {class, params, modules:[{name, shape, params, kind, group, desc}]}
  arch.block_mode= （仅 FLUX：分流块）
  arch.io        = 模型级模块（嵌入、投影、调制、输出）同上的列表

  python3 export_arch.py --runs /path/to/runs --data static/image-viz/data
"""
import argparse, json, re
from pathlib import Path

BLOCK_PREFIX = {"qwen": ["transformer_blocks.0."],
                "flux": ["single_transformer_blocks.0."],
                "flux_mode": ["mode_blocks.image.0."]}

# 后缀 → (kind, group, 中文说明)
DESC = [
    ("attn.to_q", "w", "attn", "查询投影"),
    ("attn.to_k", "w", "attn", "键投影"),
    ("attn.to_v", "w", "attn", "值投影"),
    ("attn.to_out.0", "w", "attn", "输出投影"),
    ("attn.norm_q", "v", "norm", "Q 的 RMSNorm"),
    ("attn.norm_k", "v", "norm", "K 的 RMSNorm"),
    ("img_mlp.gate_layer", "w", "mlp", "SwiGLU 门控（过 SiLU）"),
    ("img_mlp.proj", "w", "mlp", "SwiGLU 的另一支（与门控相乘）"),
    ("img_mlp.out", "w", "mlp", "MLP 输出投影"),
    ("ff.linear_in", "w", "mlp", "SwiGLU 的 gate/up（一次出 2×）"),
    ("ff.linear_out", "w", "mlp", "MLP 输出投影"),
    ("timestep_embedder.linear_1", "w", "mod", "时间步嵌入 256 → hidden"),
    ("timestep_embedder.linear_2", "w", "mod", "时间步嵌入 hidden → hidden"),
    ("target_timestep_embedder.linear_1", "w", "mod", "目标 t 嵌入 256 → hidden"),
    ("target_timestep_embedder.linear_2", "w", "mod", "目标 t 嵌入 hidden → hidden"),
    ("modulation.1", "w", "mod", "共享调制：一次算 4 组 scale/gate"),
    ("txt_in.text_norm", "v", "norm", "文本投影前的零中心 RMSNorm"),
    ("txt_in.in_layer", "w", "io", "文本投影 输入层"),
    ("txt_in.out_layer", "w", "io", "文本投影 输出层（中间有 GELU）"),
    ("img_in", "w", "io", "潜变量 64 通道 → hidden"),
    ("context_embedder", "w", "io", "文本特征 20480 → hidden"),
    ("x_embedders.image", "w", "io", "潜变量 128 通道 → hidden"),
    ("x_embedders.image_cond", "w", "io", "参考图 128 通道 → hidden"),
    ("norm_out.linear", "w", "mod", "最终 AdaIN（scale/gate）"),
    ("proj_out", "w", "io", "hidden → 速度通道"),
    ("early_stream_modulations.image.linear", "w", "mod", "分流调制 · 图像"),
    ("early_stream_modulations.image_cond.linear", "w", "mod", "分流调制 · 参考图"),
    ("early_stream_modulations.txt.linear", "w", "mod", "分流调制 · 文本"),
    ("single_stream_modulations.image.linear", "w", "mod", "合流调制 · 图像"),
    ("single_stream_modulations.image_cond.linear", "w", "mod", "合流调制 · 参考图"),
    ("single_stream_modulations.txt.linear", "w", "mod", "合流调制 · 文本"),
]


def describe(name):
    # 长后缀优先，否则 target_timestep_embedder 会被 timestep_embedder 抢走
    for suf, kind, group, desc in sorted(DESC, key=lambda d: -len(d[0])):
        if name.endswith(suf) or name == suf:
            return kind, group, desc
    return "w", "io", ""


ABBREV = [
    ("time_text_embed.timestep_embedder.", "time."),
    ("target_timestep_embedder.", "target_t."),
    ("timestep_embedder.", "t_emb."),
    ("early_stream_modulations.", "early."),
    ("single_stream_modulations.", "single."),
    ("x_embedders.", "x_emb."),
]


def short_name(name):
    """图上标签用的短名：全名太长，一屏放不下。悬停和表格里仍是全名。"""
    s = name
    for pre, rep in ABBREV:
        if s.startswith(pre):
            s = rep + s[len(pre):]
            break
    if s.endswith(".linear"):
        s = s[:-len(".linear")]
    return s


def strip_param_suffix(name):
    return re.sub(r"\.(weight|bias)$", "", name)


def norm_name(full, prefix):
    return full[len(prefix):] if full.startswith(prefix) else full


def collect(tensors, prefixes):
    out = []
    for name, shape, _dt, params in tensors:
        for i, pre in enumerate(prefixes):
            if name.startswith(pre):
                short = strip_param_suffix(norm_name(name, pre))
                kind, group, desc = describe(short)
                out.append({"name": short, "short": short_name(short), "shape": shape, "params": params,
                            "kind": kind, "group": group, "desc": desc})
                break
    # matrix 在前、向量在后，同组保持模型里的顺序
    out.sort(key=lambda m: (0 if m["kind"] == "w" else 1))
    return out


def collect_io(tensors):
    out = []
    for name, shape, _dt, params in tensors:
        if re.match(r"^(transformer_blocks|single_transformer_blocks|mode_blocks|txt_mode_blocks)\.", name):
            continue
        short = strip_param_suffix(name)
        kind, group, desc = describe(short)
        out.append({"name": short, "short": short_name(short), "shape": shape, "params": params,
                    "kind": kind, "group": group, "desc": desc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="arch_probe 的输出目录，里面是 qwen/ flux/")
    ap.add_argument("--data", required=True, help="static/image-viz/data")
    a = ap.parse_args()
    for model in ("qwen", "flux"):
        tj = Path(a.runs) / model / "transformer_tensors.json"
        meta_p = Path(a.data) / model / "meta.json"
        tensors = json.loads(tj.read_text(encoding="utf-8"))
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        tr = meta["components"]["transformer"]
        arch = {
            "transformer_class": tr["class"],
            "transformer_params": tr["params"],
            "n_tensors": len(tensors),
            "block": {"class": "", "params": 0,
                      "modules": collect(tensors, BLOCK_PREFIX[model])},
            "io": collect_io(tensors),
        }
        blk = [b for b in meta["blocks"] if b["kind"] in ("joint", "tx" )]
        joint = [b for b in meta["blocks"] if b["kind"] == "joint"]
        if joint:
            arch["block"]["class"] = joint[0].get("shape", {}).get("class", "")
            arch["block"]["params"] = joint[0]["params"]
        if model == "flux":
            arch["block_mode"] = {"class": "", "params": 0,
                                  "modules": collect(tensors, BLOCK_PREFIX["flux_mode"])}
            img = [b for b in meta["blocks"] if b["kind"] == "img"]
            if img:
                arch["block_mode"]["class"] = img[0].get("shape", {}).get("class", "")
                arch["block_mode"]["params"] = img[0]["params"]
        s = sum(m["params"] for m in arch["block"]["modules"])
        arch["block"]["sum_params"] = s
        meta["arch"] = arch
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{model}: block {arch['block']['class']} {arch['block']['params']/1e6:.1f}M "
              f"(modules sum {s/1e6:.1f}M), io {len(arch['io'])} 项")
        if model == "flux":
            s2 = sum(m["params"] for m in arch["block_mode"]["modules"])
            print(f"  mode block sum {s2/1e6:.1f}M / {arch['block_mode']['params']/1e6:.1f}M")


if __name__ == "__main__":
    main()
