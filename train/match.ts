// Head-to-head match between two player specs, run on worker threads.
//   node dist/train/match.mjs <specA> <specB> [--games 200] [--workers 4]
// spec: heuristic|<path.json>[:sims=N][:ms=N][:auto][:style=S][:c=X][:fpu=X][:fpur=X]   e.g. models/gen5.json:ms=300:c=2
// fpur: unvisited children are valued at the parent's value minus X (instead of the fixed fpu);
// rfpur: the same at the root only; fpus: scale the reduction by sqrt(visited prior mass)
// reuse: keep each player's search tree between decisions (GPU matches)
// auto: adaptive budget (ai/time-control.ts) with sims=N as the base
// style: crowns|color|points, a one-track rush toward that way of winning (ai/styles.ts)
// mlh[=W]: moves-left search (prefer quicker wins), weight W (default ai/mcts.ts DEFAULT_MLH)
import { ChildProcess, spawn } from "node:child_process";
import { appendFileSync } from "node:fs";
import { DEFAULT_MLH } from "../ai/mcts";
import { Style, STYLES } from "../ai/styles";
import { matchPool, REASONS } from "./pool";
import type { GpuOptions, PlayerSpec } from "./worker";

function parseSpec(s: string): PlayerSpec {
  const [head, ...opts] = s.split(":").reduce<string[]>((acc, part) => {
    // keep Windows drive letters (C:\...) attached to the path
    if (acc.length === 1 && /^[A-Za-z]$/.test(acc[0])) acc[0] += ":" + part;
    else acc.push(part);
    return acc;
  }, []);
  const spec: PlayerSpec = head === "heuristic" ? { kind: "heuristic", sims: 200 } : { kind: "net", net: head, sims: 200 };
  for (const o of opts) {
    const [k, v] = o.split("=");
    if (k === "sims") spec.sims = Number(v);
    else if (k === "ms") spec.timeMs = Number(v);
    else if (k === "auto") spec.auto = true;
    else if (k === "style") {
      if (!STYLES.includes(v as Style)) throw new Error(`unknown style ${v} (${STYLES.join("|")})`);
      spec.style = v as Style;
    } else if (k === "sw") spec.styleWeight = Number(v);
    else if (k === "sb") spec.styleBoost = Number(v);
    else if (k === "mlh") spec.mlh = v === undefined ? DEFAULT_MLH.weight : Number(v);
    else if (k === "mlq0") spec.mlhQ0 = Number(v);
    else if (k === "c") spec.cPuct = Number(v);
    else if (k === "fpu") spec.fpu = Number(v);
    else if (k === "fpur") spec.fpuReduction = Number(v);
    else if (k === "rfpur") spec.rootFpuReduction = Number(v);
    else if (k === "fpus") spec.fpuScaled = v !== "0";
    else if (k === "reuse") spec.reuse = v !== "0";
  }
  return spec;
}

const args = process.argv.slice(2);
const opt = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const [a, b] = args.filter((x, i) => !x.startsWith("--") && !(i > 0 && args[i - 1].startsWith("--")));
const games = opt("games", 200);
// --gpu 1: batch evaluations through a GPU inference server started (and stopped) here
let server: ChildProcess | null = null;
let gpu: GpuOptions | undefined;
if (opt("gpu", 0) === 1) {
  server = spawn("python", ["train/infer_server.py", "--port", "0", "--mem-fraction", "0.3"], { stdio: ["pipe", "pipe", "inherit"] });
  const port = await new Promise<number>((resolve, reject) => {
    server!.stdout!.on("data", (d) => {
      const m = /READY (\d+)/.exec(String(d));
      if (m) resolve(Number(m[1]));
    });
    server!.on("exit", (c) => reject(new Error(`inference server exited (${c})`)));
  });
  gpu = { gpuPort: port, concurrency: opt("concurrency", 32) };
}
const r = await matchPool({ workers: opt("workers", 4), games, seed: opt("seed", 31337), a: parseSpec(a), b: parseSpec(b), gpu }).finally(() => {
  server?.stdin?.end();
  server?.kill();
});
const ci = 1.96 * Math.sqrt((r.score * (1 - r.score)) / r.games);
const elo = -400 * Math.log10(1 / Math.min(Math.max(r.score, 1e-3), 1 - 1e-3) - 1);
const wins = (w: number[]) => REASONS.map((name, i) => (w[i] ? `${name} ${w[i]}` : "")).filter(Boolean).join("/") || "-";
const line = `${new Date().toISOString()} ${a} vs ${b}: ${(r.score * 100).toFixed(1)}% ±${(ci * 100).toFixed(1)} (Elo ${elo.toFixed(0)}) over ${r.games} games, sims/move ${r.simsPerMove.map((x) => x.toFixed(0)).join(" vs ")}, wins by A ${wins(r.winsBy[0])} B ${wins(r.winsBy[1])}, win length (actions) A ${r.winPlies[0].toFixed(1)} B ${r.winPlies[1].toFixed(1)}, clearly-winning to win A ${r.convert[0].toFixed(1)} B ${r.convert[1].toFixed(1)}`;
console.log(line);
appendFileSync("train/runs/experiments.txt", line + "\n");
