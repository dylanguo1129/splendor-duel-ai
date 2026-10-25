import { writeFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { NUM_FEATURES } from "../engine/features";
import type { GpuOptions, LeagueSpec, MatchJob, PlayerSpec, SelfPlayJob } from "./worker";

/** Win reasons by engine code (engine/rules.ts WIN_REASON_*). */
export const REASONS = ["none", "points", "crowns", "color", "antiplay", "draw"];

const WORKER = new URL("./worker.mjs", import.meta.url);

/** Polled every 5 s while workers run; a returned reason aborts the phase (all workers terminated). */
export type Watchdog = () => string | null;

function runJobs<T extends { id: number }>(jobs: T[], onMsg: (m: any) => void, watchdog?: Watchdog): Promise<void> {
  const workers: Worker[] = [];
  let timer: NodeJS.Timeout | undefined;
  const all = Promise.all(
    jobs.map(
      (job) =>
        new Promise<void>((resolve, reject) => {
          const w = new Worker(WORKER, { workerData: job });
          workers.push(w);
          w.on("message", (m) => {
            if (m.type === "done") resolve();
            else if (m.type === "error") reject(new Error(`worker ${job.id}: ${m.error}`));
            else onMsg(m);
          });
          w.on("error", reject);
          w.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`worker ${job.id} exit ${code}`))));
        }),
    ),
  ).then(() => undefined);
  const guard = new Promise<void>((_, reject) => {
    if (!watchdog) return;
    timer = setInterval(() => {
      const reason = watchdog();
      if (reason) {
        for (const w of workers) void w.terminate();
        reject(new Error(`aborted: ${reason}`));
      }
    }, 5000);
  });
  return Promise.race([all, guard]).finally(() => clearInterval(timer));
}

export async function selfPlay(opts: {
  workers: number;
  games: number;
  seed: number;
  player: PlayerSpec;
  simsFast: number;
  pFull: number;
  tempMoves: number;
  outDir: string;
  prunedDir?: string; // GPU only: the same records with pruned policy targets (ai/mcts.ts prunedVisits)
  gpu?: GpuOptions;
  watchdog?: Watchdog;
  log?: (s: string) => void;
  league?: LeagueSpec;
}): Promise<{ samples: number; games: number; avgPlies: number; summary: string }> {
  const per = Math.ceil(opts.games / opts.workers);
  // train/train.py reads the record layout from here
  const meta = JSON.stringify({ F: NUM_FEATURES, ml: true, sc: true });
  writeFileSync(`${opts.outDir}/meta.json`, meta);
  if (opts.prunedDir) writeFileSync(`${opts.prunedDir}/meta.json`, meta);
  const jobs: (SelfPlayJob & GpuOptions)[] = [];
  for (let i = 0; i < opts.workers; i++)
    jobs.push({
      mode: "selfplay",
      id: i,
      games: per,
      seed: opts.seed + i,
      player: opts.player,
      simsFast: opts.simsFast,
      pFull: opts.pFull,
      tempMoves: opts.tempMoves,
      out: `${opts.outDir}/w${i}.bin`,
      outPruned: opts.prunedDir ? `${opts.prunedDir}/w${i}.bin` : undefined,
      league: opts.league,
      ...opts.gpu,
    });
  const samples = new Array(opts.workers).fill(0);
  let games = 0;
  let plies = 0;
  const reasons = new Array(REASONS.length).fill(0);
  const byOpp = new Map<string, { n: number; score: number }>();
  const t0 = Date.now();
  await runJobs(jobs, (m) => {
    if (m.type !== "progress") return;
    samples[m.id] = m.samples;
    games++;
    plies += m.plies;
    reasons[m.reason] = (reasons[m.reason] ?? 0) + 1;
    const key = m.opp ?? "self";
    const e = byOpp.get(key) ?? { n: 0, score: 0 };
    e.n++;
    e.score += m.learnerScore ?? 0.5;
    byOpp.set(key, e);
    if (games % 100 === 0) opts.log?.(`  selfplay ${games}/${per * opts.workers} games, ${samples.reduce((a, b) => a + b, 0)} samples, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }, opts.watchdog);
  const ends = REASONS.map((r, i) => (reasons[i] ? `${r} ${((100 * reasons[i]) / games).toFixed(0)}%` : "")).filter(Boolean).join(", ");
  const opps = [...byOpp].map(([k, e]) => (k === "self" ? `self ${e.n}` : `vs ${k} ${((100 * e.score) / e.n).toFixed(0)}% (${e.n})`)).join(", ");
  return { samples: samples.reduce((a, b) => a + b, 0), games, avgPlies: plies / Math.max(1, games), summary: `games won by ${ends}; learner ${opps}` };
}

export async function matchPool(opts: {
  workers: number;
  games: number;
  seed: number;
  a: PlayerSpec;
  b: PlayerSpec;
  gpu?: GpuOptions;
  watchdog?: Watchdog;
}): Promise<{
  score: number;
  games: number;
  wins: number;
  draws: number;
  simsPerMove: [number, number];
  /** Wins of A and of B by win reason (index = engine WIN_REASON_*, see REASONS). */
  winsBy: [number[], number[]];
  /** Average game length (actions) of A's wins and of B's wins: how quickly each side wins. */
  winPlies: [number, number];
  /** Average actions from first being clearly winning (own search) to the win, for A and B (GPU matches). */
  convert: [number, number];
}> {
  const per = Math.ceil(opts.games / opts.workers);
  const jobs: (MatchJob & GpuOptions)[] = [];
  for (let i = 0; i < opts.workers; i++) jobs.push({ mode: "match", id: i, games: per, seed: opts.seed + i, a: opts.a, b: opts.b, ...opts.gpu });
  let total = 0;
  let n = 0;
  let wins = 0;
  let draws = 0;
  const sims = [0, 0];
  const decisions = [0, 0];
  const winsBy: [number[], number[]] = [new Array(REASONS.length).fill(0), new Array(REASONS.length).fill(0)];
  const plies = [0, 0];
  const plyGames = [0, 0];
  const conv = [0, 0];
  const convGames = [0, 0];
  await runJobs(jobs, (m) => {
    if (m.type !== "game") return;
    total += m.result;
    n++;
    if (m.result === 1) wins++;
    if (m.result === 0.5) draws++;
    if (m.result !== 0.5) {
      const w = m.winnerIsA ? 0 : 1;
      winsBy[w][m.reason]++;
      plies[w] += m.plies;
      plyGames[w]++;
      if (m.convert >= 0) {
        conv[w] += m.convert;
        convGames[w]++;
      }
    }
    for (const k of [0, 1]) {
      sims[k] += m.sims?.[k] ?? 0;
      decisions[k] += m.decisions?.[k] ?? 0;
    }
  }, opts.watchdog);
  const avg = (k: number) => sims[k] / Math.max(1, decisions[k]);
  const winLen = (k: number) => plies[k] / Math.max(1, plyGames[k]);
  const convLen = (k: number) => (convGames[k] ? conv[k] / convGames[k] : NaN);
  return {
    score: total / Math.max(1, n),
    games: n,
    wins,
    draws,
    simsPerMove: [avg(0), avg(1)],
    winsBy,
    winPlies: [winLen(0), winLen(1)],
    convert: [convLen(0), convLen(1)],
  };
}
