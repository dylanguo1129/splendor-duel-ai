// AlphaZero-style generation loop. Run (after `node scripts/build-train.mjs`):
//   node dist/train/loop.mjs --gens 20 --workers 30
// Resumable: state in train/runs/state.json.
import { ChildProcess, spawn, spawnSync } from "node:child_process";
import { appendFileSync, copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { constants as osConstants, cpus, freemem, setPriority } from "node:os";
import { basename } from "node:path";
import { DEFAULT_MLH } from "../ai/mcts";
import { STYLES } from "../ai/styles";
import { matchPool, REASONS, selfPlay } from "./pool";
import type { GpuOptions, LeagueOpponent, LeagueSpec, PlayerSpec } from "./worker";

const args = process.argv.slice(2);
const opt = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const GENS = opt("gens", 20);
const WORKERS = opt("workers", Math.max(1, cpus().length - 2));
const GAMES = opt("games", 1500); // self-play games per generation
const BOOT_GAMES = opt("boot-games", 2400);
const SIMS = opt("sims", 160);
const SIMS_FAST = opt("sims-fast", 40);
const P_FULL = opt("p-full", 0.35);
const GATE_GAMES = opt("gate-games", 240);
const GATE_SIMS = opt("gate-sims", 96);
const WINDOW = opt("window", 6);
const EPOCHS = opt("epochs", 3);
const GATE = opt("gate-threshold", 0.55);
const GATE_EXTRA = opt("gate-extra", 0); // more games for a borderline candidate (0: off)
const GATE_EXTRA_THRESHOLD = opt("gate-extra-threshold", 0.515); // on the combined score
const HIDDEN = (() => {
  const i = args.indexOf("--hidden");
  return i >= 0 ? args[i + 1].split(",") : ["256", "256"];
})(); // network shape before the first best model; afterwards the main line keeps the best's shape
// --challenger 512,512,512: also train a bigger network every generation (its own line, see main())
const CHALLENGER = (() => {
  const i = args.indexOf("--challenger");
  return i >= 0 ? args[i + 1].split(",") : null;
})();
const USE_GPU = opt("gpu", 0) === 1; // batch network evaluations through train/infer_server.py
const CONCURRENCY = opt("concurrency", 64); // games per worker in GPU mode
const MIN_FREE_GB = opt("min-free-gb", 3); // pause before a phase while free RAM is below this
const GPU_MEM_FRACTION = opt("gpu-mem-fraction", 0.4);
const GPU_SERVERS = Math.max(1, opt("gpu-servers", 1)); // inference server processes (workers are spread over them)
// League self-play (GPU only): a share of games against one-track rushes (crowns / one colour /
// points, ai/styles.ts) and older models, so the network learns every way of winning and stopping.
const LEAGUE = opt("league", 0) === 1;
const P_SELF = opt("p-self", 0.5); // share of plain self-play games in league mode
const OLD_COUNT = opt("old-count", 3); // older promoted models in the league
const STYLE_BENCH = opt("style-bench-games", 0); // after each promotion: games vs each rush style
const SELFPLAY_FIRST = opt("selfplay-first", 0) === 1; // resume with self-play (e.g. after a format change)
// Prefer quicker wins (and slower losses) in the search once a model's moves-left head is trained
// (ai/mcts.ts MovesLeftOptions); 0 turns it off.
const MLH_WEIGHT = opt("mlh", DEFAULT_MLH.weight);
// network players value unvisited moves at their parent's value minus this (ai/mcts.ts); 0: fixed fpu
const FPUR = opt("fpur", 0);
// self-play only (root noise on): reduction at the root, so that noise-boosted moves still get tried; -1: same as --fpur
const ROOT_FPUR = opt("root-fpur", -1);
const FPU_SCALED = opt("fpu-scaled", 0) === 1; // reduction times sqrt(visited prior mass), KataGo / lc0 style
// policy-target pruning (ai/mcts.ts prunedVisits): also write each generation's records with pruned
// policy targets to train/data/genNp, and/or train on those instead of the raw ones
const RECORD_PRUNED = opt("record-pruned", 0) === 1;
// network players keep their search tree between decisions (ai/mcts.ts descend): self-play and gates
const REUSE = opt("reuse", 0) === 1;
const TRAIN_PRUNED = opt("train-pruned", 0) === 1;
const AUX_WEIGHT = opt("aux-weight", 0); // train/train.py auxiliary final-tally loss
const ENTITY = opt("entity", 0) === 1; // train/train.py --entity: entity policy terms (train/entity.py)
// passed on to train/train.py when set: max LR when fine-tuning from the best, value-target blend,
// and worker files held out of training (whole games) so its validation loss measures unseen games
const FINETUNE_LR = opt("finetune-lr", 0);
const Q_WEIGHT = opt("q-weight", -1);
const HOLDOUT = (() => {
  const i = args.indexOf("--holdout");
  return i >= 0 ? args[i + 1].split(",") : [];
})();

// Keep the desktop responsive: training runs below normal priority (workers are threads of this process).
try {
  setPriority(process.pid, osConstants.priority.PRIORITY_BELOW_NORMAL);
} catch {
  /* not permitted: run at normal priority */
}

const RUN = "train/runs";
const DATA = "train/data";
const MODELS = "models";
for (const d of [RUN, DATA, MODELS]) mkdirSync(d, { recursive: true });
const LOG = `${RUN}/log.txt`;
const log = (s: string) => {
  const line = `[${new Date().toISOString()}] ${s}`;
  console.log(line);
  appendFileSync(LOG, line + "\n");
};

interface RunState {
  gen: number; // last generation whose self-play data exists
  best: string | null; // path (without extension) of best net, null = heuristic
  challenger?: string | null; // latest model of the --challenger line (not promoted yet)
  history: {
    gen: number;
    cand: string;
    vsBest: number;
    promoted: boolean;
    vsHeuristic?: number;
    vsStyle?: Record<string, number>; // score vs each one-track rush (same network)
    samples: number;
  }[];
}
const STATE = `${RUN}/state.json`;
const st: RunState = existsSync(STATE) ? JSON.parse(readFileSync(STATE, "utf8")) : { gen: -1, best: null, history: [] };
const save = () => writeFileSync(STATE, JSON.stringify(st, null, 1));

const heuristic = (sims: number): PlayerSpec => ({ kind: "heuristic", sims });
interface ModelInfo {
  ml: boolean;
  hidden: number[];
  entity: boolean;
}
const modelInfo = new Map<string, ModelInfo>();
/** Layer sizes, whether the moves-left head is trained (meta.ml) and whether the policy has entity terms, of the model at `path` (without .json). */
function infoOf(path: string): ModelInfo {
  if (!modelInfo.has(path)) {
    let info = { ml: false, hidden: [] as number[], entity: false };
    try {
      const j = JSON.parse(readFileSync(`${path}.json`, "utf8"));
      info = {
        ml: j.meta?.ml === true,
        hidden: j.hidden.map((l: { out: number }) => l.out),
        entity: typeof j.E === "string",
      };
    } catch {
      /* missing or unreadable */
    }
    modelInfo.set(path, info);
  }
  return modelInfo.get(path)!;
}
const hasMovesLeft = (path: string) => infoOf(path).ml;
/**
 * Relative cost of one evaluation on the CPU (what play in the browser pays): the first layer
 * sees ~15% non-zero inputs, later layers ~50% (ReLU), the policy head ~60 legal moves; entity
 * policy terms add a 96-wide query layer and a 40-value card score per legal move.
 */
function evalCost(path: string): number {
  const { hidden: h, entity } = infoOf(path);
  if (!h.length) return 1;
  let cost = 0.15 * 864 * h[0] + 60 * h[h.length - 1];
  for (let k = 1; k < h.length; k++) cost += 0.5 * h[k - 1] * h[k];
  if (entity) cost += 0.5 * h[h.length - 1] * 96 + 60 * 40;
  return cost;
}
const netSpec = (path: string, sims: number): PlayerSpec => ({
  kind: "net",
  net: `${path}.json`,
  sims,
  mlh: MLH_WEIGHT > 0 && hasMovesLeft(path) ? MLH_WEIGHT : undefined,
  fpuReduction: FPUR > 0 ? FPUR : undefined,
  fpuScaled: FPUR > 0 && FPU_SCALED ? true : undefined,
  reuse: REUSE || undefined,
});

const lowerPriority = (pid: number | undefined) => {
  try {
    if (pid) setPriority(pid, osConstants.priority.PRIORITY_BELOW_NORMAL);
  } catch {
    /* ignore */
  }
};

/** Pause (never crash) while free RAM is low; gives up after ~30 minutes. */
async function waitForMemory(what: string): Promise<void> {
  for (let i = 0; freemem() / 2 ** 30 < MIN_FREE_GB; i++) {
    if (i === 0) log(`low memory (${(freemem() / 2 ** 30).toFixed(1)} GB free < ${MIN_FREE_GB} GB): pausing before ${what}`);
    if (i >= 60) throw new Error(`free memory stayed below ${MIN_FREE_GB} GB for 30 minutes`);
    await new Promise((r) => setTimeout(r, 30_000));
  }
}

// ---- GPU inference servers (owned by this process; they exit by themselves if we die) ----
// Each server is one Python process (single-threaded request handling); with several, the workers
// are spread over them so that the GPU, not Python, sets the pace.
const gpuServers: ChildProcess[] = [];
let gpuPorts: number[] = [];
async function startOneServer(): Promise<{ child: ChildProcess; port: number }> {
  // own process group + hidden console: a Ctrl+C on our console cannot reach it
  const fraction = Math.min(GPU_MEM_FRACTION, 0.9 / GPU_SERVERS);
  const child = spawn("python", ["train/infer_server.py", "--port", "0", "--mem-fraction", String(fraction)], {
    stdio: ["pipe", "pipe", "pipe"],
    detached: true,
    windowsHide: true,
  });
  lowerPriority(child.pid);
  child.stderr!.on("data", (d) => log(`  gpu-server: ${String(d).trim().slice(0, 500)}`));
  const port = await new Promise<number>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("inference server did not start")), 60_000);
    child.stdout!.on("data", (d) => {
      const m = /READY (\d+) (\w+)/.exec(String(d));
      if (m) {
        clearTimeout(timer);
        log(`gpu inference server ready on port ${m[1]} (${m[2]})`);
        resolve(Number(m[1]));
      }
    });
    child.on("exit", (code) => reject(new Error(`inference server exited (${code})`)));
  });
  child.on("exit", (code) => log(`gpu inference server on port ${port} exited (${code})`));
  return { child, port };
}
async function startGpuServer(): Promise<void> {
  const started = await Promise.all(Array.from({ length: GPU_SERVERS }, startOneServer));
  gpuServers.push(...started.map((s) => s.child));
  gpuPorts = started.map((s) => s.port);
}
const stopGpuServer = () => {
  for (const s of gpuServers.splice(0)) {
    s.stdin?.end();
    s.kill();
  }
  gpuPorts = [];
};
process.on("exit", stopGpuServer);
// A stray console Ctrl+C must not end an overnight run; stop the loop by terminating the process.
process.on("SIGINT", () => log("ignoring SIGINT (terminate the process to stop the loop)"));
process.on("SIGBREAK", () => log("ignoring SIGBREAK (terminate the process to stop the loop)"));
process.on("SIGHUP", () => log("ignoring SIGHUP (terminate the process to stop the loop)"));
process.on("SIGTERM", () => {
  stopGpuServer();
  process.exit(1);
});
const gpu = (): GpuOptions | undefined => (USE_GPU ? { gpuPort: gpuPorts[0], gpuPorts, concurrency: CONCURRENCY } : undefined);

/** Run a GPU phase; if the inference server connection breaks, restart the server and retry (max 3). */
async function withGpuRetry<T>(what: string, fn: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
      const gpuFault = USE_GPU && /ECONNRESET|ECONNREFUSED|connection closed|inference server|EPIPE/.test(msg);
      if (!gpuFault || attempt >= 3) throw e;
      log(`${what}: GPU server connection failed (${msg.slice(0, 160)}); restarting server, retry ${attempt}/2`);
      stopGpuServer();
      await new Promise((r) => setTimeout(r, 3000));
      await startGpuServer();
    }
  }
}

// Hard limits checked every 5 s during a phase: abort the phase (workers terminated) instead of
// letting the machine run out of memory.
const HARD_FREE_GB = opt("hard-free-gb", 1.5);
const MAX_RSS_GB = opt("max-rss-gb", 10);
const watchdog = (): string | null => {
  const free = freemem() / 2 ** 30;
  if (free < HARD_FREE_GB) return `free RAM ${free.toFixed(1)} GB < ${HARD_FREE_GB} GB`;
  const rss = process.memoryUsage().rss / 2 ** 30;
  if (rss > MAX_RSS_GB) return `training process uses ${rss.toFixed(1)} GB > ${MAX_RSS_GB} GB`;
  return null;
};

/**
 * League opponents: the three rush styles (crowns and one colour twice as often: the measured blind
 * spots) and a few older promoted models, spread over time.
 */
function leagueSpec(): LeagueSpec | undefined {
  if (!LEAGUE || !USE_GPU || !st.best) return undefined;
  const opponents: LeagueOpponent[] = [];
  for (const style of STYLES) for (let k = 0; k < (style === "points" ? 1 : 2); k++) opponents.push({ kind: "style", style, name: `${style}-rush` });
  const older = [...new Set(st.history.filter((h) => h.promoted && h.cand !== st.best).map((h) => h.cand))].filter((c) => existsSync(`${c}.json`)).slice(-12);
  for (let k = 0; k < Math.min(OLD_COUNT, older.length); k++) {
    const c = older[Math.floor(((k + 0.5) * older.length) / Math.min(OLD_COUNT, older.length))];
    opponents.push({ kind: "old", net: `${c}.json`, name: basename(c) });
  }
  return { pSelf: P_SELF, opponents };
}

async function genData(gen: number) {
  const dir = `${DATA}/gen${gen}`;
  const prunedDir = RECORD_PRUNED && USE_GPU && st.best ? `${dir}p` : undefined;
  for (const d of [dir, prunedDir]) {
    if (!d) continue;
    rmSync(d, { recursive: true, force: true }); // never append to data of an interrupted run
    mkdirSync(d, { recursive: true });
  }
  const player = st.best ? netSpec(st.best, SIMS) : heuristic(SIMS);
  if (st.best && FPUR > 0 && ROOT_FPUR >= 0) player.rootFpuReduction = ROOT_FPUR;
  const games = st.best ? GAMES : BOOT_GAMES;
  const league = leagueSpec();
  await waitForMemory("self-play");
  log(
    `gen ${gen}: self-play ${games} games with ${st.best ?? "heuristic"} (${SIMS}/${SIMS_FAST} sims${USE_GPU ? ", GPU" : ""})` +
      (league ? `, league ${Math.round(100 * (1 - league.pSelf))}% vs ${[...new Set(league.opponents.map((o) => o.name))].join(", ")}` : ""),
  );
  const r = await withGpuRetry(`gen ${gen} self-play`, () =>
    selfPlay({
      workers: WORKERS,
      games,
      seed: 100000 + gen * 1000,
      player,
      simsFast: SIMS_FAST,
      pFull: P_FULL,
      tempMoves: 12,
      outDir: dir,
      prunedDir,
      gpu: gpu(),
      watchdog,
      log,
      league,
    }),
  );
  log(`gen ${gen}: ${r.games} games, ${r.samples} samples, avg plies ${r.avgPlies.toFixed(1)}; ${r.summary}`);
  return r.samples;
}

/** Score of `cand` against its own network playing each one-track rush, and how the rush won. */
async function styleBench(gen: number, cand: string): Promise<Record<string, number>> {
  const out: Record<string, number> = {};
  const parts: string[] = [];
  for (const style of STYLES) {
    const r = await withGpuRetry(`gen ${gen} ${style} bench`, () =>
      matchPool({
        workers: WORKERS,
        games: STYLE_BENCH,
        seed: 4242 + gen,
        a: netSpec(cand, GATE_SIMS),
        b: { ...netSpec(cand, GATE_SIMS), style },
        gpu: gpu(),
        watchdog,
      }),
    );
    out[style] = r.score;
    const rush =
      REASONS.map((name, i) => (r.winsBy[1][i] ? `${name} ${r.winsBy[1][i]}` : ""))
        .filter(Boolean)
        .join("/") || "none";
    parts.push(`vs ${style}-rush ${(r.score * 100).toFixed(1)}% (rush wins: ${rush}; our wins take ${r.winPlies[0].toFixed(0)} actions)`);
  }
  log(`gen ${gen}: style bench (${STYLE_BENCH} games each, ${GATE_SIMS} sims): ${parts.join(" | ")}`);
  return out;
}

/** Layer sizes of the main line: those of the current best (HIDDEN before there is one). */
const mainHidden = (): string[] => (st.best && infoOf(st.best).hidden.length ? infoOf(st.best).hidden.map(String) : HIDDEN);

async function train(gen: number, hidden: string[], init: string, suffix = ""): Promise<string> {
  // never overwrite the current best (a resumed run retrains the same generation)
  let out = `${MODELS}/gen${gen}${suffix}`;
  for (let k = 2; out === st.best || out === st.challenger || existsSync(`${out}.json`); k++) out = `${MODELS}/gen${gen}${suffix}_${k}`;
  const dirs = [];
  for (let g = Math.max(0, gen - WINDOW + 1); g <= gen; g++) {
    // pruned policy targets where recorded (older generations: the raw records)
    const d = TRAIN_PRUNED && existsSync(`${DATA}/gen${g}p`) ? `${DATA}/gen${g}p` : `${DATA}/gen${g}`;
    if (existsSync(d)) dirs.push(d);
  }
  await waitForMemory("training");
  log(`gen ${gen}: training ${hidden.join("x")} on ${dirs.join(",")} init=${init || "random"}`);
  const child = spawn(
    "python",
    [
      "train/train.py",
      ...["--data", ...dirs, "--out", out, "--epochs", String(EPOCHS), "--hidden", ...hidden],
      ...(init ? ["--init", init] : []),
      ...(FINETUNE_LR > 0 ? ["--finetune-lr", String(FINETUNE_LR)] : []),
      ...(Q_WEIGHT >= 0 ? ["--q-weight", String(Q_WEIGHT)] : []),
      ...(HOLDOUT.length ? ["--holdout", ...HOLDOUT] : []),
      ...(AUX_WEIGHT > 0 ? ["--aux-weight", String(AUX_WEIGHT)] : []),
      ...(ENTITY ? ["--entity"] : []),
    ],
    {
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
      windowsHide: true,
    },
  );
  lowerPriority(child.pid);
  let output = "";
  child.stdout!.on("data", (d) => (output += d));
  child.stderr!.on("data", (d) => (output += d));
  const code = await new Promise<number | null>((resolve) => child.on("exit", resolve));
  for (const line of output.split("\n").filter(Boolean)) log(`  py: ${line.trim()}`);
  if (code !== 0) throw new Error("training failed");
  return out;
}

async function main() {
  log(
    `loop start: workers=${WORKERS} gens=${GENS} state gen=${st.gen} best=${st.best} hidden=${mainHidden().join("x")}` +
      `${CHALLENGER ? ` challenger=${CHALLENGER.join("x")} (${st.challenger ?? "new"})` : ""} gpu=${USE_GPU}`,
  );
  if (USE_GPU) await startGpuServer();
  if (st.gen < 0) {
    await genData(0);
    st.gen = 0;
    save();
  } else if (SELFPLAY_FIRST) {
    // the latest data was already trained on: generate fresh data with the current best first
    await genData(st.gen + 1);
    st.gen++;
    save();
  }
  for (let it = 0; it < GENS; it++) {
    const gen = st.gen;
    const cand = await train(gen, mainHidden(), st.best ? `${st.best}.pt` : "");
    const entry = await gate(gen, cand, "candidate");
    st.history.push(entry);
    save();
    // challenger line: a bigger network trained from its own previous checkpoint every generation
    // (so it keeps improving while it is not promoted), gated against the best at equal compute
    if (CHALLENGER && st.best && infoOf(st.best).hidden.join(",") !== CHALLENGER.join(",")) {
      const init = st.challenger && existsSync(`${st.challenger}.pt`) ? `${st.challenger}.pt` : `${st.best}.pt`;
      const ch = await train(gen, CHALLENGER, init, "c");
      st.challenger = ch;
      save();
      const ce = await gate(gen, ch, `challenger ${CHALLENGER.join("x")}`);
      if (ce.promoted) st.challenger = null;
      st.history.push(ce);
      save();
    }
    entry.samples = await genData(gen + 1);
    st.gen = gen + 1;
    save();
    const stale = `${DATA}/gen${gen + 1 - WINDOW - 2}`;
    for (const d of [stale, `${stale}p`]) if (existsSync(d)) rmSync(d, { recursive: true, force: true });
  }
  log("loop done");
}

/** Gate `cand` against the current best (equal compute); on success make it the best and publish it. */
async function gate(gen: number, cand: string, label: string): Promise<RunState["history"][number]> {
  // equal compute: a cheaper (smaller) model gets proportionally more simulations, so a bigger
  // network is only promoted if it plays better in the same thinking time
  const ratio = st.best ? evalCost(cand) / evalCost(st.best) : 1;
  const candSims = Math.round(GATE_SIMS * Math.max(1, 1 / ratio));
  const bestSims = Math.round(GATE_SIMS * Math.max(1, ratio));
  if (candSims !== bestSims) log(`gen ${gen}: equal-compute gating: ${label} ${candSims} sims vs best ${bestSims} sims (cost ratio ${ratio.toFixed(2)})`);
  const opponent = st.best ? netSpec(st.best, bestSims) : heuristic(GATE_SIMS);
  await waitForMemory("gating");
  const g = await withGpuRetry(`gen ${gen} gating`, () =>
    matchPool({
      workers: WORKERS,
      games: GATE_GAMES,
      seed: 777 + gen,
      a: netSpec(cand, candSims),
      b: opponent,
      gpu: gpu(),
      watchdog,
    }),
  );
  // The first network always replaces the heuristic as self-play generator (avoids stalling on a
  // strong hand-written prior); afterwards a candidate must beat the current best.
  let promoted = st.best === null || g.score >= GATE;
  let score = g.score;
  let games = g.games;
  // borderline (at least even, below the threshold): more games, judged on the combined score with
  // a lower threshold, so that small real gains are not thrown away by a 400-game sample
  if (!promoted && st.best && GATE_EXTRA > 0 && g.score >= 0.5) {
    log(`gen ${gen}: ${label} borderline at ${(g.score * 100).toFixed(1)}% over ${g.games} games: ${GATE_EXTRA} more`);
    const g2 = await withGpuRetry(`gen ${gen} extra gating`, () =>
      matchPool({ workers: WORKERS, games: GATE_EXTRA, seed: 7777 + gen, a: netSpec(cand, candSims), b: opponent, gpu: gpu(), watchdog }),
    );
    score = (g.score * g.games + g2.score * g2.games) / (g.games + g2.games);
    games = g.games + g2.games;
    promoted = score >= GATE_EXTRA_THRESHOLD;
  }
  log(
    `gen ${gen}: ${label} vs ${st.best ?? "heuristic"} score ${(score * 100).toFixed(1)}% over ${games} games -> ${promoted ? "PROMOTED" : "rejected"}` +
      ` (wins take ${g.winPlies[0].toFixed(0)} vs ${g.winPlies[1].toFixed(0)} actions)`,
  );
  const entry: RunState["history"][number] = {
    gen,
    cand,
    vsBest: score,
    promoted,
    samples: 0,
  };
  if (promoted) {
    st.best = cand;
    copyFileSync(`${cand}.json`, `${MODELS}/best.json`);
    // make the new model live for the local app (next page load)
    const pub = spawnSync("node", ["scripts/publish-model.mjs", `${MODELS}/best.json`], { encoding: "utf8" });
    log(`gen ${gen}: ${(pub.stdout || pub.stderr).trim()}`);
    if (st.history.filter((h) => h.promoted).length % 2 === 0) {
      const h = await withGpuRetry(`gen ${gen} heuristic eval`, () =>
        matchPool({
          workers: WORKERS,
          games: 200,
          seed: 999 + gen,
          a: netSpec(cand, GATE_SIMS),
          b: heuristic(GATE_SIMS),
          gpu: gpu(),
          watchdog,
        }),
      );
      entry.vsHeuristic = h.score;
      log(`gen ${gen}: best vs heuristic MCTS (${GATE_SIMS} sims each): ${(h.score * 100).toFixed(1)}%`);
    }
    if (STYLE_BENCH > 0 && USE_GPU) entry.vsStyle = await styleBench(gen, cand);
  }
  return entry;
}

main().catch((e) => {
  log(`FATAL ${e?.stack ?? e}`);
  process.exit(1);
});
