"""Print PyTorch outputs (value, first 5 logits) for the first K records of a data file."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "train")
from train import NUM_FEATURES, PVNet, dir_meta, init_from, rec_dtype  # noqa: E402

pt, binf, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
state = torch.load(pt, map_location="cpu")
model = PVNet(state["_hidden"])
init_from(model, state, state["_hidden"], "cpu")  # adapts older checkpoints (appended inputs, moves head)
model.eval()
f_dir, ml, sc = dir_meta(os.path.dirname(binf))
rec = rec_dtype(f_dir, ml, sc)
recs = np.fromfile(binf, dtype=np.uint8)[: k * rec.itemsize].view(rec)
x = np.zeros((len(recs), NUM_FEATURES), dtype=np.float32)
x[:, :f_dir] = recs["x"].view(np.float16).astype(np.float32)
with torch.no_grad():
    logits, v, m = model(torch.from_numpy(x))
for i in range(len(recs)):
    print(f"{v[i].item():.6f} " + " ".join(f"{logits[i, a].item():.6f}" for a in range(5)) + f" moves {m[i].item() * 100:.1f}")
