"""Batched GPU inference server for self-play / match workers.

Workers (train/gpu-client.ts) connect over TCP on 127.0.0.1 and send evaluation requests; the
server merges requests from all workers into large batches and runs them on the GPU.

Protocol (little-endian), one outstanding request per connection:
  LOAD  b'L' slot:u8 len:u16 path:utf8        -> b'K'  | b'E' len:u16 msg
  EVAL  b'V' slot:u8 n:u32 n*F float16         -> b'R' n:u32 n*(2+A) float16  (tanh value, logits, moves left / 100)
  MULTI b'M' k:u8 k*(slot:u8 n:u32 pad:u8 n*F float16)
                                               -> b'R' total:u32 total*(2+A) float16, parts in order
  PING  b'P'                                   -> b'P'

Safety: the process exits when its stdin closes (the training loop died), GPU memory is capped
with --mem-fraction, and batches are capped with --max-batch.

Usage: python train/infer_server.py --port 0 [--max-batch 8192] [--mem-fraction 0.5]
Prints "READY <port>" once listening.
"""
import argparse
import asyncio
import os
import signal
import struct
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacked import StackedModels, shape_of  # noqa: E402
from train import NUM_ACTIONS, NUM_FEATURES, model_from_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=0)
ap.add_argument("--max-batch", type=int, default=8192)
ap.add_argument("--mem-fraction", type=float, default=0.5)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
torch.backends.cuda.matmul.allow_tf32 = True

models = {}  # path -> half-precision model on DEV
stacks = {}  # layer shapes -> StackedModels: same-shaped busy models share one batched pass
queue: "asyncio.Queue[tuple]" = None  # (model, np.ndarray[n,F] f16, future)
stats = {"batches": 0, "evals": 0}


def get_model(path):
    m = models.get(path)
    if m is None:
        if len(models) >= 8:  # bounded cache
            old = models.pop(next(iter(models)))
            stacks[shape_of(old)].remove(old)
        m, _ = model_from_json(path)
        m = m.to(DEV).half().eval()
        models[path] = m
        stacks.setdefault(shape_of(m), StackedModels(DEV)).add(m)
    return m


def run_single(group, to_dev):
    model = group[0][0]
    with torch.inference_mode():
        logits, v, m = model(to_dev(group))
        out = torch.cat([v.unsqueeze(1), logits, m.unsqueeze(1)], dim=1).to(torch.float16).cpu().numpy()
    reply(group, out)


def reply(group, out):
    o = 0
    for g in group:
        n = g[1].shape[0]
        if not g[2].done():
            g[2].set_result(out[o : o + n])
        o += n
    stats["batches"] += 1
    stats["evals"] += out.shape[0]


async def batcher():
    while True:
        first = await queue.get()
        items = [first]
        total = first[1].shape[0]
        await asyncio.sleep(0)  # let concurrently arriving requests join this batch
        while not queue.empty() and total < args.max_batch:
            it = queue.get_nowait()
            items.append(it)
            total += it[1].shape[0]
        by_model = {}
        for it in items:
            by_model.setdefault(id(it[0]), []).append(it)
        to_dev = lambda group: torch.from_numpy(np.concatenate([g[1] for g in group])).to(DEV, non_blocking=True)
        # several models busy (league self-play): one batched pass per layer shape
        by_shape = {}
        for group in by_model.values():
            by_shape.setdefault(shape_of(group[0][0]), []).append(group)
        for shape, groups in by_shape.items():
            st = stacks.get(shape)
            if len(groups) > 1 and st and all(st.index_of(g[0][0]) >= 0 for g in groups):
                for group, out in zip(groups, st.forward([(st.index_of(g[0][0]), to_dev(g)) for g in groups])):
                    reply(group, out)
            else:
                for group in groups:
                    run_single(group, to_dev)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    slots = {}
    loop = asyncio.get_running_loop()
    try:
        while True:
            op = await reader.readexactly(1)
            if op == b"P":
                writer.write(b"P")
            elif op == b"L":
                slot, ln = struct.unpack("<BH", await reader.readexactly(3))
                path = (await reader.readexactly(ln)).decode()
                try:
                    slots[slot] = get_model(path)
                    writer.write(b"K")
                except Exception as e:  # report and keep serving
                    msg = str(e).encode()[:1000]
                    writer.write(b"E" + struct.pack("<H", len(msg)) + msg)
            elif op == b"V":
                slot, n = struct.unpack("<BI", await reader.readexactly(5))
                raw = await reader.readexactly(n * NUM_FEATURES * 2)
                x = np.frombuffer(raw, dtype="<f2").reshape(n, NUM_FEATURES)
                fut = loop.create_future()
                await queue.put((slots[slot], x, fut))
                out = await fut
                writer.write(b"R" + struct.pack("<I", n) + out.astype("<f2").tobytes())
            elif op == b"M":
                (k,) = struct.unpack("<B", await reader.readexactly(1))
                futs = []
                for _ in range(k):
                    slot, n, _pad = struct.unpack("<BIB", await reader.readexactly(6))
                    raw = await reader.readexactly(n * NUM_FEATURES * 2)
                    fut = loop.create_future()
                    await queue.put((slots[slot], np.frombuffer(raw, dtype="<f2").reshape(n, NUM_FEATURES), fut))
                    futs.append(fut)
                outs = await asyncio.gather(*futs)
                total = sum(o.shape[0] for o in outs)
                writer.write(b"R" + struct.pack("<I", total) + b"".join(o.astype("<f2").tobytes() for o in outs))
            else:
                break
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


def exit_when_parent_dies():
    try:
        sys.stdin.read()
    finally:
        os._exit(0)


async def main():
    global queue
    queue = asyncio.Queue()
    threading.Thread(target=exit_when_parent_dies, daemon=True).start()
    asyncio.create_task(batcher())
    server = await asyncio.start_server(handle, "127.0.0.1", args.port, limit=1 << 24)
    port = server.sockets[0].getsockname()[1]
    print(f"READY {port} {DEV}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    assert NUM_ACTIONS == 224 and NUM_FEATURES == 864  # engine/features.ts
    # A stray console Ctrl+C must not kill the server mid-phase; shutdown is via stdin EOF.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    asyncio.run(main())
