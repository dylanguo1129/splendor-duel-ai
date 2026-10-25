"""Train the Splendor Duel policy/value MLP on self-play records written by train/worker.ts.

Usage:
  python train/train.py --data DIR [DIR ...] --out models/genN --init models/genM.pt
         [--hidden 256 256] [--epochs 3] [--batch 1024] [--lr 1e-3] [--holdout w22 w23]
  python train/train.py --data DIR [DIR ...] --init models/genM.pt --hidden 512 512 --holdout w22 w23 --eval-only
Writes <out>.pt (torch state) and <out>.json (weights for ai/net.ts). --holdout keeps whole worker
files (complete games) out of training, so the validation loss measures unseen games.

Heads: policy (logits), value (tanh, game outcome for the player to move) and moves left (remaining
actions until the game ends / 100, used by the search to prefer quicker wins).
"""
import argparse
import base64
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FEATURES = 864  # engine/features.ts NUM_FEATURES
BASE_FEATURES = 820  # layout before the win-path features were appended (older data and models)
NUM_ACTIONS = 224
FLAG_VALUE_ONLY = 1  # record flag: no policy target (opponent moves in league games)
MOVES_SCALE = 100.0  # moves-left head output = remaining actions / MOVES_SCALE


AUX_SCALE = (20.0, 10.0, 10.0)  # final prestige / crowns / best-colour differences -> aux outputs
AUX_CLASSES = 7  # how the game ended for the mover: won by points/crowns/colour, lost by them, other
AUX_OUT = len(AUX_SCALE) + AUX_CLASSES


def rec_dtype(f, ml=True, sc=False):
    fields = [
        ("x", "<u2", (f,)),
        ("legal", "u1", (28,)),
        ("pi", "<u2", (NUM_ACTIONS,)),
        ("z", "i1"),
        ("q", "i1"),
        ("hasq", "u1"),
        ("flags", "u1"),
    ]
    if ml:
        fields.append(("left", "<u2"))  # actions until the game ended (train/records.ts)
    if sc:  # final tallies of the mover minus the opponent's, and how the game ended
        fields += [("pd", "i1"), ("cd", "i1"), ("bd", "i1"), ("reason", "u1")]
    return np.dtype(fields)


REC = rec_dtype(NUM_FEATURES, True, True)
assert REC.itemsize == 2 * NUM_FEATURES + 28 + 2 * NUM_ACTIONS + 10


def dir_meta(d):
    """(feature count, has moves-left targets, has final-tally targets) of the records in a data dir.
    meta.json is written with the data; dirs without one hold base-layout records without either."""
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        return BASE_FEATURES, False, False
    m = json.load(open(meta))
    return m["F"], bool(m.get("ml", False)), bool(m.get("sc", False))


def dir_features(d):
    return dir_meta(d)[0]


from entity import ENT_OUT, entity_logits, line_matrix  # noqa: E402  (optional entity policy terms)


class PVNet(nn.Module):
    def __init__(self, hidden, entity=False):
        super().__init__()
        dims = [NUM_FEATURES] + list(hidden)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.policy = nn.Linear(dims[-1], NUM_ACTIONS)
        self.value = nn.Linear(dims[-1], 1)
        self.moves = nn.Linear(dims[-1], 1)
        self.aux = nn.Linear(dims[-1], AUX_OUT)  # training only (regularization), not exported
        self.entity = entity
        if entity:
            self.ent = nn.Linear(dims[-1], ENT_OUT, bias=False)
            nn.init.zeros_(self.ent.weight)  # starts as the plain network
            self.register_buffer("M", line_matrix(), persistent=False)

    def forward(self, x, with_aux=False):
        """-> (policy logits, tanh value, moves left / MOVES_SCALE[, aux outputs])"""
        h = x
        for layer in self.hidden:
            h = F.relu(layer(h))
        logits = self.policy(h)
        if self.entity:
            logits = logits + entity_logits(x, self.ent(h), self.M).to(logits.dtype)
        out = (logits, torch.tanh(self.value(h)).squeeze(-1), self.moves(h).squeeze(-1))
        return out + (self.aux(h),) if with_aux else out


def load(dirs, q_weight, holdout=()):
    keys = ("legal", "pi", "z", "q", "hasq", "flags")
    parts = {k: [] for k in ("x", "left", "has_ml", "held", "aux", "has_sc") + keys}
    for d in dirs:
        f_dir, ml, sc = dir_meta(d)
        rec = rec_dtype(f_dir, ml, sc)
        for f in sorted(glob.glob(os.path.join(d, "*.bin"))):
            raw = np.fromfile(f, dtype=np.uint8)
            n = raw.size // rec.itemsize
            if not n:
                continue
            # a worker file holds whole games: held-out files give a validation set of unseen games
            parts["held"].append(np.full(n, os.path.basename(f)[:-4] in holdout))
            data = raw[: n * rec.itemsize].view(rec)
            x = np.zeros((n, NUM_FEATURES), dtype=np.float16)  # appended features of older data: 0
            x[:, :f_dir] = data["x"].view(np.float16)
            parts["x"].append(x)
            for k in keys:
                parts[k].append(data[k])
            parts["left"].append(data["left"].astype(np.float32) if ml else np.zeros(n, np.float32))
            parts["has_ml"].append(np.full(n, ml))
            aux = np.zeros((n, len(AUX_SCALE) + 1), np.float32)  # scaled differences, class
            if sc:
                for i, k in enumerate(("pd", "cd", "bd")):
                    aux[:, i] = data[k].astype(np.float32) / AUX_SCALE[i]
                won, lost, path = data["z"] > 0, data["z"] < 0, np.isin(data["reason"], (1, 2, 3))
                cls = np.full(n, AUX_CLASSES - 1)
                cls = np.where(won & path, data["reason"].astype(np.int64) - 1, cls)
                cls = np.where(lost & path, data["reason"].astype(np.int64) + 2, cls)
                aux[:, -1] = cls
            parts["aux"].append(aux)
            parts["has_sc"].append(np.full(n, sc))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    x = cat["x"]
    legal = np.unpackbits(cat["legal"], axis=1, bitorder="little")[:, :NUM_ACTIONS].astype(bool)
    pi = cat["pi"].astype(np.float32) / 65535.0
    pi /= np.maximum(pi.sum(axis=1, keepdims=True), 1e-6)
    z = cat["z"].astype(np.float32)
    # value target: blend the game outcome with the root search value where recorded
    q = cat["q"].astype(np.float32) / 127.0
    has_pi = (cat["flags"] & FLAG_VALUE_ONLY) == 0
    has = cat["hasq"].astype(bool) & has_pi  # value-only records: outcome only (their search is biased)
    z = np.where(has, (1 - q_weight) * z + q_weight * q, z).astype(np.float32)
    left = cat["left"] / MOVES_SCALE
    return x, legal, pi, z, has_pi, left, cat["has_ml"], cat["held"], cat["aux"], cat["has_sc"]


def model_from_json(path):
    """PVNet with the weights of an exported .json model (inverse of export)."""
    j = json.load(open(path))

    def f32(s):
        return torch.from_numpy(np.frombuffer(base64.b64decode(s), dtype="<f4").copy())

    hidden = [l["out"] for l in j["hidden"]]
    model = PVNet(hidden, entity="E" in j)
    with torch.no_grad():
        if "E" in j:
            model.ent.weight.copy_(f32(j["E"]).view(ENT_OUT, hidden[-1]))
        for i, (layer, l) in enumerate(zip(model.hidden, j["hidden"])):
            w = f32(l["W"]).view(l["in"], l["out"]).t()
            if i == 0 and l["in"] < NUM_FEATURES:
                layer.weight.zero_()  # older model: appended inputs get zero weights (same outputs)
                layer.weight[:, : l["in"]] = w
            else:
                layer.weight.copy_(w)
            layer.bias.copy_(f32(l["b"]))
        model.policy.weight.copy_(f32(j["Wp"]).view(j["A"], hidden[-1]))
        model.policy.bias.copy_(f32(j["bp"]))
        model.value.weight.copy_(f32(j["Wv"]).view(1, hidden[-1]))
        model.value.bias.fill_(j["bv"])
        if "Wm" in j:
            model.moves.weight.copy_(f32(j["Wm"]).view(1, hidden[-1]))
            model.moves.bias.fill_(j["bm"])
        else:  # model without a moves-left head: predicts 0 (the search ignores it, see meta.ml)
            model.moves.weight.zero_()
            model.moves.bias.zero_()
    return model, hidden


def b64(t):
    return base64.b64encode(np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4")).tobytes()).decode()


def export(model, path, meta):
    j = {
        "version": 1,
        "F": NUM_FEATURES,
        "A": NUM_ACTIONS,
        "hidden": [
            {"in": l.in_features, "out": l.out_features, "W": b64(l.weight.t()), "b": b64(l.bias)}
            for l in model.hidden
        ],
        "Wp": b64(model.policy.weight),
        "bp": b64(model.policy.bias),
        "Wv": b64(model.value.weight.flatten()),
        "bv": float(model.value.bias.item()),
        "meta": meta,
    }
    if meta.get("ml"):  # an untrained moves-left head is left out (loaders then use zeros)
        j.update({"Wm": b64(model.moves.weight.flatten()), "bm": float(model.moves.bias.item()), "movesScale": MOVES_SCALE})
    if model.entity:  # [ENT_OUT][H] output-major, like Wp (ai/net.ts adds the entity terms)
        j["E"] = b64(model.ent.weight)
    with open(path, "w") as f:
        json.dump(j, f)


def init_from(model, state, hidden, dev):
    """Load a checkpoint into `model`. Older checkpoints are adapted without changing what they
    compute: appended inputs get zero weights, a missing moves-left head starts at zero, and when
    `hidden` adds layers of the same width after the old ones, they start as identity layers (their
    input has been through ReLU already, so ReLU(I h) = h). Returns a description, or None."""
    old = state.get("_hidden", hidden)
    w0 = state["hidden.0.weight"]
    if w0.shape[1] < NUM_FEATURES:
        state["hidden.0.weight"] = torch.cat([w0, torch.zeros(w0.shape[0], NUM_FEATURES - w0.shape[1], device=w0.device)], dim=1)
    deepen = len(old) < len(hidden) and list(hidden[: len(old)]) == list(old) and all(h == old[-1] for h in hidden[len(old) :])
    if list(old) != list(hidden) and not deepen:
        return None
    weights = {k: v for k, v in state.items() if not k.startswith("_")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not unexpected, unexpected
    with torch.no_grad():
        if "moves.weight" not in weights:
            model.moves.weight.zero_()
            model.moves.bias.zero_()
        for k in range(len(old), len(hidden)):
            model.hidden[k].weight.copy_(torch.eye(hidden[k], device=dev))
            model.hidden[k].bias.zero_()
    allowed = {"moves.weight", "moves.bias", "aux.weight", "aux.bias", "ent.weight"}  # heads older checkpoints lack
    allowed |= {f"hidden.{k}.{p}" for k in range(len(old), len(hidden)) for p in ("weight", "bias")}
    assert set(missing) <= allowed, missing
    return f"{'x'.join(map(str, old))} -> {'x'.join(map(str, hidden))} (identity layers)" if deepen else "same shape"


def losses_of(model, T, idx, chunk=65536):
    """Mean (policy, value, moves-left, auxiliary) losses over idx; T holds the data tensors.
    Without gradients, large index sets are evaluated in chunks (held-out sets can be large)."""
    if len(idx) > chunk and not torch.is_grad_enabled():
        tot = [0.0] * 4
        wsum = [0.0] * 4
        for i in range(0, len(idx), chunk):
            c = idx[i : i + chunk]
            w = (T["HP"][c].sum().item(), len(c), T["HM"][c].sum().item(), T["HS"][c].sum().item())
            for k, l in enumerate(losses_of(model, T, c, chunk)):
                tot[k] += l.item() * w[k]
                wsum[k] += w[k]
        return tuple(torch.tensor(t / max(w, 1)) for t, w in zip(tot, wsum))
    xb = T["X"][idx].float()
    logits, v, m, aux = model(xb, with_aux=True)
    logits = logits.masked_fill(~T["L"][idx], -1e9)
    lp = F.log_softmax(logits, dim=1)
    w = T["HP"][idx]  # value-only samples have no policy target
    pl = (-(T["P"][idx] * lp).sum(dim=1) * w).sum() / w.sum().clamp(min=1.0)
    vl = F.mse_loss(v, T["Z"][idx])
    wm = T["HM"][idx]  # older data has no moves-left target
    ml = (F.smooth_l1_loss(m, T["ML"][idx], reduction="none", beta=0.1) * wm).sum() / wm.sum().clamp(min=1.0)
    # auxiliary: final tally differences (regression) and how the game ended (classification)
    ws = T["HS"][idx]  # older data has no final tallies
    ax = T["AX"][idx]
    nreg = len(AUX_SCALE)
    reg = F.smooth_l1_loss(aux[:, :nreg], ax[:, :nreg], reduction="none", beta=0.1).sum(dim=1)
    cls = F.cross_entropy(aux[:, nreg:], ax[:, nreg].long(), reduction="none")
    al = ((reg + cls) * ws).sum() / ws.sum().clamp(min=1.0)
    return pl, vl, ml, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="output path without extension (required unless --eval-only)")
    ap.add_argument("--init", default=None)
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 256])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=4e-4, help="max LR when initialized from a checkpoint")
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=0.5, help="weight of the root search value in the value target")
    ap.add_argument("--ml-weight", type=float, default=0.5, help="weight of the moves-left loss")
    ap.add_argument("--aux-weight", type=float, default=0.0, help="weight of the auxiliary final-tally loss (0: head not trained)")
    ap.add_argument("--scratch-epochs", type=int, default=4, help="minimum epochs when not initialized from a checkpoint")
    ap.add_argument("--moves-head-only", action="store_true", help="train only the moves-left head (policy and value stay exactly as in --init)")
    ap.add_argument("--holdout", nargs="*", default=[], help="worker files (e.g. w22 w23) held out of training as validation games")
    ap.add_argument("--eval-only", action="store_true", help="only report the held-out losses of --init")
    ap.add_argument("--subsample", type=float, default=1.0, help="fraction of training positions drawn per epoch")
    ap.add_argument("--entity", action="store_true", help="entity policy terms (see entity_logits); kept once a checkpoint has them")
    ap.add_argument("--entity-lr", type=float, default=3e-3, help="max LR of the entity query layer")
    args = ap.parse_args()
    if not args.out and not args.eval_only:
        ap.error("--out is required")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    x, legal, pi, z, has_pi, left, has_ml, held, aux, has_sc = load(args.data, args.q_weight, set(args.holdout))
    n = len(z)
    if args.holdout:
        val_idx, tr_idx = np.flatnonzero(held), np.flatnonzero(~held)
    else:
        perm = np.random.default_rng(0).permutation(n)
        nval = max(1, min(max(1000, n // 50), n // 10))
        val_idx, tr_idx = perm[:nval], perm[nval:]
    nval = len(val_idx)
    T = {
        "X": torch.from_numpy(x).to(dev),
        "L": torch.from_numpy(legal).to(dev),
        "P": torch.from_numpy(pi).to(dev),
        "Z": torch.from_numpy(z).to(dev),
        "HP": torch.from_numpy(has_pi.astype(np.float32)).to(dev),
        "ML": torch.from_numpy(left).to(dev),
        "HM": torch.from_numpy(has_ml.astype(np.float32)).to(dev),
        "AX": torch.from_numpy(aux).to(dev),
        "HS": torch.from_numpy(has_sc.astype(np.float32)).to(dev),
    }
    show_aux = args.aux_weight > 0 and has_sc.any()
    print(
        f"loaded {n} samples in {time.time() - t0:.1f}s (val {nval}, value-only {int((~has_pi).sum())}, "
        f"moves-left targets {int(has_ml.sum())}, final-tally targets {int(has_sc.sum())}); z mean {z.mean():+.3f}",
        flush=True,
    )

    state = torch.load(args.init, map_location=dev) if args.init and os.path.exists(args.init) else None
    entity = args.entity or (state is not None and "ent.weight" in state)
    model = PVNet(args.hidden, entity=entity).to(dev)
    lr = args.lr
    initialized = False
    ml_trained = bool(has_ml.any())
    if state is not None:
        how = init_from(model, state, args.hidden, dev)
        if how:
            lr = args.finetune_lr
            initialized = True
            ml_trained = ml_trained or bool(state.get("_ml", False))
            print(f"initialized from {args.init}, {how} (max lr {lr})", flush=True)
    if not initialized and args.epochs < args.scratch_epochs:
        print(f"training from scratch: {args.scratch_epochs} epochs", flush=True)
        args.epochs = args.scratch_epochs
    params = model.parameters()
    if args.moves_head_only:
        assert initialized and has_ml.any(), "--moves-head-only needs --init and data with moves-left targets"
        for p in model.parameters():
            p.requires_grad_(False)
        model.moves.requires_grad_(True)
        params = model.moves.parameters()
        lr = args.lr
        print("training the moves-left head only (policy and value unchanged)", flush=True)
    if args.eval_only:
        model.eval()
        with torch.no_grad():
            vp, vv, vm, _ = (t.item() for t in losses_of(model, T, torch.from_numpy(val_idx).to(dev)))
        print(f"EVAL {args.init}: val policy {vp:.4f} value {vv:.4f} moves {vm:.4f} on {nval} held-out positions", flush=True)
        return
    if entity and not args.moves_head_only:
        # the entity query layer is new and starts at zero: its own, higher learning rate
        ent = list(model.ent.parameters())
        ids = {id(p) for p in ent}
        groups = [{"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr}, {"params": ent, "lr": args.entity_lr}]
    else:
        groups = [{"params": list(params), "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0 if args.moves_head_only else args.wd)
    per_epoch = int(len(tr_idx) * args.subsample)
    steps = args.epochs * (per_epoch // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)

    tr = torch.from_numpy(tr_idx).to(dev)
    va = torch.from_numpy(val_idx).to(dev)
    for ep in range(args.epochs):
        model.train()
        sh = tr[torch.randperm(len(tr), device=dev)][:per_epoch]
        tp = tv = tm = ta = 0.0
        nb = 0
        for i in range(0, len(sh), args.batch):
            idx = sh[i : i + args.batch]
            pl, vl, ml, al = losses_of(model, T, idx)
            loss = ml if args.moves_head_only else pl + args.value_weight * vl + args.ml_weight * ml + args.aux_weight * al
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tp += pl.item()
            tv += vl.item()
            tm += ml.item()
            ta += al.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vp, vv, vm, va_aux = losses_of(model, T, va)
        print(
            f"epoch {ep + 1}: train policy {tp / nb:.4f} value {tv / nb:.4f} moves {tm / nb:.4f}"
            + (f" aux {ta / nb:.4f}" if show_aux else "")
            + f" | val policy {vp.item():.4f} value {vv.item():.4f} moves {vm.item():.4f}"
            + (f" aux {va_aux.item():.4f}" if show_aux else ""),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    state = model.state_dict()
    state["_hidden"] = args.hidden
    state["_ml"] = ml_trained
    torch.save(state, args.out + ".pt")
    meta = {
        "name": os.path.basename(args.out),
        "samples": int(n),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "val_policy": float(vp),
        "val_value": float(vv),
        "ml": ml_trained,  # the moves-left head was trained (the search may use it)
        "val_moves": float(vm) if ml_trained else None,
        "entity": entity,  # entity policy terms (the .json has "E")
    }
    export(model, args.out + ".json", meta)
    print(f"saved {args.out}.json in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # training is stopped by the loop, not by console Ctrl+C
    main()
