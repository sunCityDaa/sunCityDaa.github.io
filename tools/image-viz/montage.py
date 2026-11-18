import sys, json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
run = Path(sys.argv[1]); s = json.load(open(run/"summary.json"))
n = s["steps_observed"]; pick = sorted(set([0, 1, 2, 4, n//4, n//2, 3*n//4, n-2, n-1]))
cell = 192; W = cell*len(pick); img = Image.new("RGB", (W, cell*2+60), "white"); d = ImageDraw.Draw(img)
for j, i in enumerate(pick):
    for r, kind in enumerate(["xt", "x0"]):
        p = Image.open(run/"previews"/f"{kind}_{i:02d}.png").resize((cell, cell), Image.NEAREST)
        img.paste(p, (j*cell, 20 + r*(cell+20)))
    d.text((j*cell+4, 4), f"step {i}  t={s['steps'][i]['t']:.3f}", fill="black")
d.text((4, cell+22), "row1: x_t (noisy input)   row2: x0 estimate = x_t - t*v", fill="black")
img.save(run/"montage.png")
# token RMS heatmaps for last step, few blocks
z = np.load(run/"token_maps.npz"); keys = [k for k in z.files if k.startswith(f"s{n-1}/")]
def bkey(k):
    import re; m = re.findall(r"\d+", k.split("/",1)[1]); return (k.split("/",1)[1].split(".")[0], int(m[-1]) if m else 0)
keys.sort(key=bkey); sel = keys[::max(1, len(keys)//8)][:8] + [keys[-1]]
hm = Image.new("RGB", (160*len(sel), 180), "white"); d = ImageDraw.Draw(hm)
for j, k in enumerate(sel):
    a = z[k].astype(np.float32); a = (a - a.min())/(np.ptp(a)+1e-6)
    rgb = (np.stack([a, a**0.5*0.6, 1-a], -1)*255).astype(np.uint8)
    hm.paste(Image.fromarray(rgb).resize((150, 150), Image.NEAREST), (j*160, 24)); d.text((j*160+2, 4), k.split("/",1)[1][-26:], fill="black")
hm.save(run/"token_rms.png"); print(run, pick, len(keys))
