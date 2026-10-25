"""Check StackedModels against per-model forward passes: python -I train/test_stacked.py [model.json ...]"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacked import StackedModels  # noqa: E402
from train import NUM_FEATURES, model_from_json  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sys.argv[1:]
models = [model_from_json(p)[0].to(dev).half().eval() for p in paths]
st = StackedModels(dev)
for m in models:
    st.add(m)
rng = np.random.default_rng(0)
# uneven group sizes, including a group of one and repeated models
sizes = [37, 1, 600, 129][: len(models)] + [5]
groups = []
for k, n in enumerate(sizes):
    mi = k % len(models)
    x = torch.from_numpy((rng.random((n, NUM_FEATURES)) < 0.1).astype(np.float16)).to(dev)
    groups.append((mi, x))
got = st.forward(groups)
worst = 0.0
for (mi, x), out in zip(groups, got):
    with torch.inference_mode():
        logits, v, m = models[mi](x)
        ref = torch.cat([v.unsqueeze(1), logits, m.unsqueeze(1)], dim=1).to(torch.float16).cpu().numpy()
    worst = max(worst, float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max()))
print(f"{len(models)} models, groups {sizes}: max |stacked - single| = {worst:.4g}")
assert worst < 0.05, worst
st.remove(models[0])
assert st.index_of(models[0]) == -1 and len(st.models) == len(models) - 1
print("ok")
