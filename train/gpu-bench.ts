// Check GPU inference against the JS network and benchmark batched vs CPU self-play.
//   node dist/train/gpu-bench.mjs <model.json> <records.bin> [--workers 4] [--games 24]
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Net } from "../ai/net";
import { NUM_ACTIONS } from "../engine/constants";
import { NUM_FEATURES } from "../engine/features";
import { fromHalf, GpuClient, OUT_PER_EVAL } from "./gpu-client";
import { selfPlay } from "./pool";
import { RECORD_BYTES } from "./records";

const args = process.argv.slice(2);
const [modelPath, recPath] = args;
const opt = (n: string, d: number) => (args.indexOf(`--${n}`) >= 0 ? Number(args[args.indexOf(`--${n}`) + 1]) : d);
const WORKERS = opt("workers", 4);
const GAMES = opt("games", 24);

const server = spawn("python", ["train/infer_server.py", "--port", "0", "--mem-fraction", "0.3"], { stdio: ["pipe", "pipe", "inherit"] });
const port = await new Promise<number>((resolve, reject) => {
  server.stdout!.on("data", (d) => {
    const m = /READY (\d+)/.exec(String(d));
    if (m) resolve(Number(m[1]));
  });
  server.on("exit", (c) => reject(new Error(`server exit ${c}`)));
});
console.log(`server on ${port}`);

// 1) equivalence on recorded positions
const net = new Net(JSON.parse(readFileSync(modelPath, "utf8")));
const recs = readFileSync(recPath);
const N = Math.min(512, Math.floor(recs.length / RECORD_BYTES));
const feats = new Float32Array(N * NUM_FEATURES);
for (let i = 0; i < N; i++) for (let j = 0; j < NUM_FEATURES; j++) feats[i * NUM_FEATURES + j] = fromHalf(recs.readUInt16LE(i * RECORD_BYTES + 2 * j));
const client = await GpuClient.connect(port);
await client.load(0, modelPath);
const out = await client.eval(0, feats, N);
const all = Array.from({ length: NUM_ACTIONS }, (_, a) => a);
const logits = new Float32Array(NUM_ACTIONS);
let maxDv = 0;
let maxDl = 0;
let argmaxAgree = 0;
for (let i = 0; i < N; i++) {
  const v = net.forward(feats.subarray(i * NUM_FEATURES, (i + 1) * NUM_FEATURES), all, NUM_ACTIONS, logits);
  maxDv = Math.max(maxDv, Math.abs(v - out[i * OUT_PER_EVAL]));
  let am1 = 0;
  let am2 = 0;
  for (let a = 0; a < NUM_ACTIONS; a++) {
    const g = out[i * OUT_PER_EVAL + 1 + a];
    maxDl = Math.max(maxDl, Math.abs(logits[a] - g));
    if (logits[a] > logits[am1]) am1 = a;
    if (g > out[i * OUT_PER_EVAL + 1 + am2]) am2 = a;
  }
  if (am1 === am2) argmaxAgree++;
}
console.log(`equivalence over ${N} positions: max |dv| ${maxDv.toFixed(4)}, max |dlogit| ${maxDl.toFixed(4)}, argmax agree ${argmaxAgree}/${N}`);

// 2) throughput: batched GPU vs CPU self-play (same games, same sims)
const player = { kind: "net" as const, net: modelPath, sims: 400 };
let peakRss = 0;
const rssTimer = setInterval(() => (peakRss = Math.max(peakRss, process.memoryUsage().rss)), 500);
const modes = args.includes("--gpu-only") ? ["gpu"] : ["gpu", "cpu"];
const CONC = opt("concurrency", Math.ceil(GAMES / WORKERS));
for (const mode of modes) {
  peakRss = 0;
  const dir = mkdtempSync(join(tmpdir(), "sdai-bench-"));
  const t0 = Date.now();
  const r = await selfPlay({
    workers: WORKERS,
    games: GAMES,
    seed: 4242,
    player,
    simsFast: 80,
    pFull: 0.35,
    tempMoves: 12,
    outDir: dir,
    gpu: mode === "gpu" ? { gpuPort: port, concurrency: CONC } : undefined,
  });
  const dt = (Date.now() - t0) / 1000;
  console.log(`${mode}: ${r.games} games in ${dt.toFixed(1)}s = ${(r.games / dt).toFixed(2)} games/s, ${r.samples} samples, peak RSS ${(peakRss / 2 ** 30).toFixed(2)} GB`);
  rmSync(dir, { recursive: true, force: true });
}
clearInterval(rssTimer);
client.close();
server.stdin!.end();
server.kill();
process.exit(0);
