"""Checks of the entity policy terms: python -I train/test_entity.py <model.pt> <data dir>
- an entity model initialized from a plain checkpoint computes exactly what the plain one does;
- export -> model_from_json round trip keeps every output;
- StackedModels (GPU batching) matches per-model forward for plain and entity models side by side."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacked import StackedModels, shape_of  # noqa: E402
from train import NUM_FEATURES, PVNet, dir_meta, export, init_from, model_from_json, rec_dtype  # noqa: E402

pt, data = sys.argv[1], sys.argv[2]
state = torch.load(pt, map_location="cpu")
hidden = state["_hidden"]
plain = PVNet(hidden)
assert init_from(plain, dict(state), hidden, "cpu")
ent = PVNet(hidden, entity=True)
assert init_from(ent, dict(state), hidden, "cpu")
plain.eval()
ent.eval()

f_dir, ml, sc = dir_meta(data)
rec = rec_dtype(f_dir, ml, sc)
recs = np.fromfile(os.path.join(data, "w0.bin"), dtype=np.uint8)[: 512 * rec.itemsize].view(rec)
x = torch.zeros(len(recs), NUM_FEATURES)
x[:, :f_dir] = torch.from_numpy(recs["x"].view(np.float16).astype(np.float32))

with torch.no_grad():
    a, b = plain(x), ent(x)
    worst = max(float((u - v).abs().max()) for u, v in zip(a, b))
print(f"zero entity layer == plain: max diff {worst:.3g}")
assert worst == 0.0

# give the entity layer weights, then round-trip through the exported JSON
torch.manual_seed(1)
with torch.no_grad():
    ent.ent.weight.normal_(0, 0.05)
    ref = ent(x)
tmp = os.path.join(os.path.dirname(pt), "_entity_roundtrip.json")
export(ent, tmp, {"ml": True, "entity": True})
back, _ = model_from_json(tmp)
back.eval()
with torch.no_grad():
    got = back(x)
os.remove(tmp)
worst = max(float((u - v).abs().max()) for u, v in zip(ref, got))
print(f"export round trip: max diff {worst:.3g}")
assert worst < 1e-5
logit_effect = float((ref[0] - a[0]).abs().max())
print(f"entity terms change logits by up to {logit_effect:.3g}")
assert logit_effect > 1e-3

# GPU batching: an entity model and a plain one are never stacked together, and each matches forward
dev = "cuda" if torch.cuda.is_available() else "cpu"
assert shape_of(plain) != shape_of(ent)
for m in (plain, ent):
    m16 = m.to(dev).half().eval()
    st = StackedModels(dev)
    st.add(m16)
    xs = x.to(dev).half()
    out = st.forward([(0, xs[:200]), (0, xs[200:])])
    with torch.inference_mode():
        logits, v, mv = m16(xs)
        full = torch.cat([v.unsqueeze(1), logits, mv.unsqueeze(1)], 1).float().cpu().numpy()
    got = np.concatenate(out).astype(np.float32)
    worst = float(np.abs(got - full).max())
    print(f"stacked vs forward ({'entity' if m.entity else 'plain'}): max diff {worst:.3g}")
    assert worst < 0.05
print("ok")
