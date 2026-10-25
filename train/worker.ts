// worker_threads entry: self-play data generation or evaluation matches.
import { appendFileSync, readFileSync } from "node:fs";
import { parentPort, workerData } from "node:worker_threads";
import { HeuristicEvaluator } from "../ai/agents";
import { Evaluator, Mcts } from "../ai/mcts";
import { Net, NetEvaluator, NetJson } from "../ai/net";
import { DRAW, Phase } from "../engine/constants";
import { encode, NUM_FEATURES } from "../engine/features";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, legalActions } from "../engine/rules";
import { newGame } from "../engine/state";
import { Style, StyledEvaluator } from "../ai/styles";
import { runAuto } from "../ai/time-control";
import { matchBatched, mlhOf, selfPlayBatched } from "./batched";
import { finalStats, RECORD_BYTES, Sample, writeRecord } from "./records";

export interface PlayerSpec {
  kind: "heuristic" | "net";
  net?: string; // path to weights json
  sims: number;
  timeMs?: number; // matches only: think this long per decision instead of a fixed simulation count
  auto?: boolean; // matches only: adaptive budget with `sims` as the base (ai/time-control.ts)
  style?: Style; // play a one-track rush toward this way of winning (ai/styles.ts)
  styleWeight?: number; // its value bonus (default STYLE_WEIGHTS)
  styleBoost?: number; // strength of its prior preference for path moves (default 1)
  mlh?: number; // moves-left search weight (ai/mcts.ts MovesLeftOptions; needs a trained head), 0/unset: off
  mlhQ0?: number; // its value threshold (default DEFAULT_MLH.q0)
  cPuct?: number;
  fpu?: number;
  fpuReduction?: number; // unvisited children: parent value minus this (ai/mcts.ts), unset: fixed fpu
  rootFpuReduction?: number; // the same at the root only (default fpuReduction)
  fpuScaled?: boolean; // reduction times sqrt(visited prior mass)
  reuse?: boolean; // GPU (batched) games only: continue the player's previous search tree (ai/mcts.ts descend)
}

/** A league game's opponent: the learner's own network playing a style, or an older model. */
export type LeagueOpponent =
  | { kind: "style"; style: Style; weight?: number; boost?: number; name: string }
  | { kind: "old"; net: string; name: string };

/** League self-play: the share pSelf of games is plain self-play, the rest is against `opponents`. */
export interface LeagueSpec {
  pSelf: number;
  opponents: LeagueOpponent[];
}

export interface SelfPlayJob {
  mode: "selfplay";
  id: number;
  games: number;
  seed: number;
  player: PlayerSpec;
  simsFast: number; // playout-cap randomization: cheap moves are not recorded
  pFull: number;
  tempMoves: number; // decisions played proportionally to visits
  out: string;
  outPruned?: string; // GPU self-play only: also write the records with pruned policy targets here
  league?: LeagueSpec; // GPU self-play only
}

export interface MatchJob {
  mode: "match";
  id: number;
  games: number;
  seed: number;
  a: PlayerSpec;
  b: PlayerSpec;
}

const netCache = new Map<string, Net>();
function makeEvaluator(p: PlayerSpec): Evaluator {
  if (p.kind === "heuristic") return new HeuristicEvaluator();
  let net = netCache.get(p.net!);
  if (!net) {
    net = new Net(JSON.parse(readFileSync(p.net!, "utf8")) as NetJson);
    netCache.set(p.net!, net);
  }
  return new NetEvaluator(net);
}

function pickByVisits(children: { action: number; N: number }[], tau: number, rng: Rng): number {
  if (tau <= 0.01) return children[0].action;
  let sum = 0;
  const w = children.map((c) => Math.pow(c.N, 1 / tau));
  for (const x of w) sum += x;
  let r = rng.next() * sum;
  for (let i = 0; i < children.length; i++) {
    r -= w[i];
    if (r <= 0) return children[i].action;
  }
  return children[0].action;
}

function selfPlay(job: SelfPlayJob): void {
  const ev = makeEvaluator(job.player);
  const rng = new Rng(job.seed);
  const buf = new Int16Array(512);
  let totalSamples = 0;
  for (let g = 0; g < job.games; g++) {
    const s = newGame(job.seed * 1000 + g);
    const samples: Sample[] = [];
    let decisions = 0;
    while (s.phase !== Phase.GameOver) {
      const n = legalActions(s, buf);
      if (n === 1) {
        applyAction(s, buf[0], rng, DEFAULT_RULES);
        continue;
      }
      const full = rng.next() < job.pFull;
      const m = new Mcts(s, ev, {
        seed: rng.nextU32(),
        cPuct: job.player.cPuct,
        fpu: job.player.fpu,
        fpuReduction: job.player.fpuReduction,
        rootFpuReduction: job.player.rootFpuReduction,
        fpuScaled: job.player.fpuScaled,
        dirichletAlpha: full ? Math.min(1, Math.max(0.05, 8 / n)) : 0,
      });
      m.run({ iterations: full ? job.player.sims : job.simsFast });
      const res = m.result();
      if (full) {
        const x = new Float32Array(NUM_FEATURES);
        encode(s, x);
        const pi = new Map<number, number>();
        let tot = 0;
        for (const c of res.children) tot += c.N;
        for (const c of res.children) if (c.N > 0) pi.set(c.action, (pi.get(c.action) ?? 0) + c.N / tot);
        samples.push({ x, legal: Array.from(buf.subarray(0, n)), pi, mover: s.current, q: res.value, ply: s.ply });
      }
      const a = pickByVisits(res.children, decisions < job.tempMoves ? 1 : 0, rng);
      applyAction(s, a, rng, DEFAULT_RULES);
      decisions++;
    }
    const out = Buffer.alloc(samples.length * RECORD_BYTES);
    const fin = finalStats(s);
    samples.forEach((smp, i) => writeRecord(out, i * RECORD_BYTES, smp, s.winner === DRAW ? 0 : s.winner === smp.mover ? 1 : -1, s.ply - smp.ply, fin));
    appendFileSync(job.out, out);
    totalSamples += samples.length;
    parentPort!.postMessage({ type: "progress", id: job.id, game: g + 1, samples: totalSamples, winner: s.winner, reason: s.winReason, plies: s.ply });
  }
}

function match(job: MatchJob): void {
  const evA = makeEvaluator(job.a);
  const evB = makeEvaluator(job.b);
  const rng = new Rng(job.seed);
  const buf = new Int16Array(512);
  for (let g = 0; g < job.games; g++) {
    const aFirst = (job.id + g) % 2 === 0;
    const s = newGame(job.seed * 1000 + g, 0);
    const specs = aFirst ? [job.a, job.b] : [job.b, job.a];
    const evs = aFirst ? [evA, evB] : [evB, evA];
    const sims = [0, 0]; // simulations spent by A and B
    const decisions = [0, 0];
    while (s.phase !== Phase.GameOver) {
      const n = legalActions(s, buf);
      let a: number;
      if (n === 1) a = buf[0];
      else {
        const spec = specs[s.current];
        const ev = spec.style ? new StyledEvaluator(evs[s.current], spec.style, s.current, spec.styleWeight, spec.styleBoost) : evs[s.current];
        const m = new Mcts(s, ev, { seed: rng.nextU32(), cPuct: spec.cPuct, fpu: spec.fpu, fpuReduction: spec.fpuReduction, rootFpuReduction: spec.rootFpuReduction, fpuScaled: spec.fpuScaled, mlh: mlhOf(spec) });
        if (spec.auto) runAuto(m, { base: spec.sims }, n);
        else m.run(spec.timeMs ? { timeMs: spec.timeMs } : { iterations: spec.sims });
        const res = m.result();
        const side = (s.current === 0) === aFirst ? 0 : 1;
        sims[side] += res.iterations;
        decisions[side]++;
        a = res.action;
      }
      applyAction(s, a, rng, DEFAULT_RULES);
    }
    const aIdx = aFirst ? 0 : 1;
    const result = s.winner === DRAW ? 0.5 : s.winner === aIdx ? 1 : 0;
    parentPort!.postMessage({ type: "game", id: job.id, result, reason: s.winReason, winnerIsA: s.winner === aIdx, plies: s.ply, sims, decisions });
  }
}

/** Set when a GPU inference server is running: evaluations are batched through it. */
export interface GpuOptions {
  gpuPort?: number;
  gpuPorts?: number[]; // several inference servers: job k uses gpuPorts[k % length]
  concurrency?: number;
}

async function main(job: (SelfPlayJob | MatchJob) & GpuOptions): Promise<void> {
  const gpu = job.gpuPort !== undefined;
  const conc = job.concurrency ?? 64;
  // several inference servers: spread the jobs over them
  const port = job.gpuPorts?.length ? job.gpuPorts[job.id % job.gpuPorts.length] : job.gpuPort!;
  if (job.mode === "selfplay") {
    if (gpu && job.player.kind === "net") await selfPlayBatched({ ...job, gpuPort: port, concurrency: conc });
    else selfPlay(job);
  } else if (gpu && (job.a.kind === "net" || job.b.kind === "net") && !job.a.timeMs && !job.b.timeMs) {
    await matchBatched({ ...job, gpuPort: port, concurrency: conc });
  } else match(job);
  parentPort!.postMessage({ type: "done", id: job.id });
}

main(workerData).catch((e) => {
  parentPort!.postMessage({ type: "error", id: (workerData as { id: number }).id, error: String(e?.stack ?? e) });
  process.exit(1);
});
