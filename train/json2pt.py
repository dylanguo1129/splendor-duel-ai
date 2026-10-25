"""Rebuild a PyTorch checkpoint (.pt) from exported JS weights (.json).
Usage: python train/json2pt.py models/best.json models/gen5.pt
"""
import sys

import torch

sys.path.insert(0, "train")
from train import model_from_json  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
model, hidden = model_from_json(src)
state = model.state_dict()
state["_hidden"] = hidden
torch.save(state, dst)
print(f"wrote {dst} from {src} (hidden {hidden})")
