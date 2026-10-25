"""Write PyTorch outputs (value, logits of the legal actions) for the first K records of a data file,
for scripts/check-net-parity.ts: python -I train/dump_net.py <model.json> <file.bin> <K> <out.json>"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import NUM_ACTIONS, NUM_FEATURES, dir_meta, model_from_json, rec_dtype  # noqa: E402

model_path, binf, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model, _ = model_from_json(model_path)
model.eval()
f_dir, ml, sc = dir_meta(os.path.dirname(binf))
rec = rec_dtype(f_dir, ml, sc)
recs = np.fromfile(binf, dtype=np.uint8)[: k * rec.itemsize].view(rec)
x = np.zeros((len(recs), NUM_FEATURES), dtype=np.float32)
x[:, :f_dir] = recs["x"].view(np.float16).astype(np.float32)
legal = np.unpackbits(recs["legal"], axis=1, bitorder="little")[:, :NUM_ACTIONS].astype(bool)
with torch.no_grad():
    logits, v, _ = model(torch.from_numpy(x))
rows = []
for i in range(len(recs)):
    acts = np.flatnonzero(legal[i]).tolist()
    rows.append({"x": x[i].tolist(), "actions": acts, "logits": logits[i, acts].tolist(), "value": float(v[i])})
json.dump({"entity": model.entity, "rows": rows}, open(out, "w"))
print(f"{len(rows)} records, entity={model.entity}")
