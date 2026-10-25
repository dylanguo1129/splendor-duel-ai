"""Evaluate several same-shaped PVNet models in one batched GPU pass.

League self-play keeps up to four models busy at once (the learner and older ones). Running them one
after another costs a GPU round trip each per batch cycle; stacking their weights and using bmm
serves all of them with one pass and one device sync.
"""
import torch
import torch.nn.functional as F

from entity import entity_logits


def shape_of(model):
    """Models with the same shape can share a stack; entity models (train/entity.py) never share one
    with plain models, whose batched pass has no entity terms."""
    return tuple(tuple(l.weight.shape) for l in model.hidden) + (bool(getattr(model, "entity", False)),)


class StackedModels:
    def __init__(self, device, dtype=torch.float16):
        self.device, self.dtype = device, dtype
        self.models = []  # PVNet, index = position in the stacks
        self.stacks = None

    def index_of(self, model):
        for i, m in enumerate(self.models):
            if m is model:
                return i
        return -1

    def compatible(self, model):
        return not self.models or shape_of(model) == shape_of(self.models[0])

    def add(self, model):
        """Add a model (same layer shapes as the others); returns its index."""
        i = self.index_of(model)
        if i >= 0:
            return i
        self.models.append(model)
        self._rebuild()
        return len(self.models) - 1

    def remove(self, model):
        i = self.index_of(model)
        if i >= 0:
            self.models.pop(i)
            self._rebuild()

    def _rebuild(self):
        cast = lambda t: t.detach().to(self.device, self.dtype)
        with torch.no_grad():
            hidden = [
                (torch.stack([cast(m.hidden[k].weight).t() for m in self.models]), torch.stack([cast(m.hidden[k].bias) for m in self.models]).unsqueeze(1))
                for k in range(len(self.models[0].hidden))
            ] if self.models else []
            head = lambda name: (
                torch.stack([cast(getattr(m, name).weight).t() for m in self.models]),
                torch.stack([cast(getattr(m, name).bias) for m in self.models]).unsqueeze(1),
            )
            self.stacks = (hidden, head("policy"), head("value"), head("moves")) if self.models else None
            self.entity = bool(self.models) and bool(getattr(self.models[0], "entity", False))
            if self.entity:
                self.ent = torch.stack([cast(m.ent.weight).t() for m in self.models])  # [models, H, ENT_OUT]
                self.M = self.models[0].M.to(self.device)

    @torch.inference_mode()
    def forward(self, groups):
        """groups: [(model index, float16 tensor [n, F] on device)] -> list of [n, 2+A] (tanh value, logits, moves left)."""
        hidden, (wp, bp), (wv, bv), (wm, bm) = self.stacks
        idx = torch.tensor([g[0] for g in groups], device=self.device)
        nmax = max(g[1].shape[0] for g in groups)
        x = torch.zeros(len(groups), nmax, groups[0][1].shape[1], device=self.device, dtype=self.dtype)
        for k, (_, xs) in enumerate(groups):
            x[k, : xs.shape[0]] = xs
        h = x
        for w, b in hidden:
            h = F.relu(torch.baddbmm(b[idx], h, w[idx]))
        logits = torch.baddbmm(bp[idx], h, wp[idx])
        if self.entity:
            logits = logits + entity_logits(x, torch.bmm(h, self.ent[idx]), self.M).to(logits.dtype)
        v = torch.tanh(torch.baddbmm(bv[idx], h, wv[idx]))
        m = torch.baddbmm(bm[idx], h, wm[idx])
        out = torch.cat([v, logits, m], dim=2).to(torch.float16).cpu().numpy()
        return [out[k, : g[1].shape[0]] for k, g in enumerate(groups)]
