// Batched self-play / matches: each worker runs many games concurrently and sends all their
// MCTS leaf evaluations to the GPU inference server in one request per model per round.
import { appendFileSync } from "node:fs";
import { parentPort } from "node:worker_threads";
import { HeuristicEvaluator } from "../ai/agents";
import { actionKey, DEFAULT_MLH, Evaluator, Mcts, MovesLeftOptions, PendingLeaf, prunedVisits, SearchResult } from "../ai/mcts";
import { DRAW, Phase } from "../engine/constants";
import { encode, NUM_FEATURES } from "../engine/features";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, legalActions } from "../engine/rules";
import { GameState, newGame } from "../engine/state";
import { GpuClient, MOVES_SCALE, OUT_PER_EVAL } from "./gpu-client";
import { NUM_ACTIONS } from "../engine/constants";
import { shapeValue, Style, stylePriorBoost } from "../ai/styles";
import { autoTimeCheck } from "../ai/time-control";
import { finalStats, RECORD_BYTES, Sample, writeRecord } from "./records";
import type { LeagueOpponent, MatchJob, PlayerSpec, SelfPlayJob } from "./worker";

interface Decision {
  slot: number; // model slot on the server, -1 = heuristic (evaluated inline)
  sims: number; // fixed budget, or the base budget in auto mode
  auto?: boolean; // adaptive budget (time-control.ts) in simulations
  legal: number;
  cPuct?: number;
  fpu?: number;
  fpuReduction?: number;
  rootFpuReduction?: number;
  fpuScaled?: boolean;
  reuse?: boolean; // continue this player's previous tree when the position was reached in it
  noise: boolean;
  mlh?: MovesLeftOptions; // prefer quicker wins with the model's moves-left head
  style?: Style; // the player to move searches as a one-track rush (ai/styles.ts)
  styleWeight?: number;
  styleBoost?: number;
  styled?: number; // that player (set when the decision starts)
}

interface Game {
  index: number;
  s: GameState;
  rng: Rng;
  mcts: Mcts | null;
  trees: [Mcts | null, Mcts | null]; // tree reuse: each player's last search
  keys: [number[], number[]]; // actionKeys played since that search's root
  decision: Decision | null;
  done: number;
  pending: PendingLeaf | null;
  nextCheck: number; // auto mode: simulation count of the next controller check
  decisions: number;
  samples: Sample[];
  full: boolean;
  aFirst: boolean;
  simsUsed: [number, number]; // matches: simulations spent by A and B
  sureAt: [number, number]; // matches: first ply at which A / B's own search saw itself clearly winning (-1: never)
  decisionsBy: [number, number];
  // league self-play
  opp: LeagueOpponent | null; // null: plain self-play
  oppSlot: number; // server slot of an older model
  oppWeight?: number; // rush strength of a style opponent in this game
  oppBoost?: number;
  learner: number; // seat of the learner
}

const unused: Evaluator = {
  evaluate() {
    throw new Error("batched search evaluates leaves outside Mcts");
  },
};

interface Hooks {
  startGame?(g: Game): void;
  decide(g: Game, legalCount: number): Decision;
  finishDecision(g: Game, res: SearchResult): number; // returns the move to play
  finishGame(g: Game): void;
}

/** Leaf evaluation of a styled decision: boost the styled player's path moves, shape the value. */
function applyStyle(d: Decision, p: PendingLeaf, priors: Float32Array, v: number): number {
  if (!d.style) return v;
  if (p.chooser === d.styled) for (let k = 0; k < p.n; k++) priors[k] *= stylePriorBoost(p.s, p.legal[k], d.style, d.styleBoost);
  return shapeValue(v, p.s, p.chooser, d.styled!, d.style, d.styleWeight);
}

/** Server connection with the given models loaded into their slots. */
async function connectSlots(port: number, models: Map<number, string>): Promise<GpuClient> {
  const client = await GpuClient.connect(port);
  for (const [slot, path] of models) await client.load(slot, path);
  return client;
}

async function drive(client: GpuClient, concurrency: number, total: number, seed: number, hooks: Hooks): Promise<void> {
  const heur = new HeuristicEvaluator();
  const buf = new Int16Array(512);
  const priors = new Float32Array(512);
  let started = 0;
  const start = (): Game | null => {
    if (started >= total) return null;
    const index = started++;
    const g: Game = {
      index,
      s: newGame(seed * 1000 + index, 0), // player 0 moves first; A sits at 0 in even games
      rng: new Rng(seed * 7919 + index),
      mcts: null,
      trees: [null, null],
      keys: [[], []],
      decision: null,
      done: 0,
      pending: null,
      nextCheck: 0,
      decisions: 0,
      samples: [],
      full: false,
      aFirst: index % 2 === 0,
      simsUsed: [0, 0],
      decisionsBy: [0, 0],
      sureAt: [-1, -1],
      opp: null,
      oppSlot: 0,
      learner: index % 2,
    };
    hooks.startGame?.(g);
    return g;
  };

  /**
   * Play `a` in g. With tree reuse, both players' trees record its key (taken before the move: it
   * depends on the position). A move that reveals chance or hidden information (a token refill draws
   * from the bag with the rng; a card leaving a deck refills the pyramid or is reserved blind) ends
   * reuse: the trees averaged over outcomes that are now known, and their statistics no longer fit.
   */
  const play = (g: Game, a: number): void => {
    if (!g.trees[0] && !g.trees[1]) {
      applyAction(g.s, a, g.rng, DEFAULT_RULES);
      return;
    }
    const k = actionKey(g.s, a);
    const r = g.rng;
    const rngBefore = [r.s0, r.s1, r.s2, r.s3];
    const decksBefore = g.s.decks.map((d) => d.length).join();
    applyAction(g.s, a, g.rng, DEFAULT_RULES);
    const revealed = r.s0 !== rngBefore[0] || r.s1 !== rngBefore[1] || r.s2 !== rngBefore[2] || r.s3 !== rngBefore[3] || g.s.decks.map((d) => d.length).join() !== decksBefore;
    if (revealed) {
      g.trees = [null, null];
      g.keys = [[], []];
    } else {
      g.keys[0].push(k);
      g.keys[1].push(k);
    }
  };

  /** Advance g until it needs a GPU evaluation ("pending") or is over. */
  const advance = (g: Game): "pending" | "over" => {
    for (;;) {
      if (g.s.phase === Phase.GameOver) return "over";
      if (!g.mcts) {
        const n = legalActions(g.s, buf);
        if (n === 1) {
          play(g, buf[0]);
          continue;
        }
        const d = hooks.decide(g, n);
        if (d.style) d.styled = g.s.current;
        g.decision = d;
        g.done = 0;
        g.nextCheck = autoChunk(d);
        const alpha = d.noise ? Math.min(1, Math.max(0.05, 8 / n)) : 0;
        const cur = g.s.current;
        const kept = d.reuse ? g.trees[cur] : null;
        if (kept && kept.descend(g.keys[cur], g.s, { dirichletAlpha: alpha }) >= 0) g.mcts = kept;
        else
          g.mcts = new Mcts(g.s, unused, {
            seed: g.rng.nextU32(),
            cPuct: d.cPuct,
            fpu: d.fpu,
            fpuReduction: d.fpuReduction,
            rootFpuReduction: d.rootFpuReduction,
            fpuScaled: d.fpuScaled,
            dirichletAlpha: alpha,
            mlh: d.mlh,
          });
        g.trees[cur] = d.reuse ? g.mcts : null;
        g.keys[cur] = [];
      }
      const d = g.decision!;
      for (;;) {
        if (!d.auto) {
          if (g.done >= d.sims) break;
        } else if (g.done >= g.nextCheck) {
          const r = g.mcts.result();
          if (autoTimeCheck({ base: d.sims }, { spent: g.done, iterations: g.done, legal: d.legal, children: r.children }).stop) break;
          g.nextCheck += autoChunk(d);
        }
        const p = g.mcts.beginIteration();
        if (!p) {
          g.done++;
          continue;
        }
        if (d.slot < 0) {
          const v = applyStyle(d, p, priors, heur.evaluate(p.s, p.legal, p.n, priors, g.rng));
          g.mcts.completeIteration(p, v, priors);
          g.done++;
          continue;
        }
        g.pending = p;
        return "pending";
      }
      const move = hooks.finishDecision(g, g.mcts.result());
      g.mcts = null;
      g.decision = null;
      play(g, move);
      g.decisions++;
    }
  };

  const live: Game[] = [];
  for (let k = 0; k < concurrency; k++) {
    const g = start();
    if (g) live.push(g);
  }
  const feats = new Map<number, Float32Array>(); // per slot (league rounds evaluate several slots at once)
  const x = new Float32Array(NUM_FEATURES);
  while (live.length) {
    const bySlot = new Map<number, Game[]>();
    for (let i = 0; i < live.length; ) {
      const g = live[i];
      if (advance(g) === "over") {
        hooks.finishGame(g);
        const ng = start();
        if (ng) live[i] = ng;
        else live.splice(i, 1);
        continue;
      }
      const slot = g.decision!.slot;
      if (!bySlot.has(slot)) bySlot.set(slot, []);
      bySlot.get(slot)!.push(g);
      i++;
    }
    const parts = [...bySlot].map(([slot, games]) => {
      let f = feats.get(slot);
      if (!f) feats.set(slot, (f = new Float32Array(concurrency * NUM_FEATURES)));
      games.forEach((g, i) => {
        encode(g.pending!.s, x);
        f!.set(x, i * NUM_FEATURES);
      });
      return { slot, feats: f, n: games.length, games };
    });
    // one round trip per round, also when several models are busy (league self-play)
    const outs = parts.length === 1 ? [await client.eval(parts[0].slot, parts[0].feats, parts[0].n)] : await client.evalMulti(parts);
    for (const [k, { games }] of parts.entries()) {
      const out = outs[k];
      games.forEach((g, i) => {
        const p = g.pending!;
        const o = i * OUT_PER_EVAL;
        let max = -Infinity;
        for (let k = 0; k < p.n; k++) max = Math.max(max, out[o + 1 + p.legal[k]]);
        for (let k = 0; k < p.n; k++) priors[k] = Math.exp(out[o + 1 + p.legal[k]] - max);
        g.mcts!.completeIteration(p, applyStyle(g.decision!, p, priors, out[o]), priors, out[o + 1 + NUM_ACTIONS] * MOVES_SCALE);
        g.done++;
        g.pending = null;
      });
    }
  }
}

/** Root value above which a player counts as clearly winning (75% win probability). */
const SURE_Q = 0.5;

/** Moves-left search settings of a player (PlayerSpec.mlh = weight, mlhQ0 = threshold; unset or 0: off). */
export function mlhOf(spec: PlayerSpec): MovesLeftOptions | undefined {
  return spec.mlh ? { ...DEFAULT_MLH, weight: spec.mlh, q0: spec.mlhQ0 ?? DEFAULT_MLH.q0 } : undefined;
}

/** Auto mode checks the time controller this many simulations apart (16 checks per base budget). */
function autoChunk(d: Decision): number {
  return Math.max(8, Math.round(d.sims / 16));
}

function pickByVisits(children: { action: number; N: number }[], tau: number, rng: Rng): number {
  if (tau <= 0.01) return children[0].action;
  const w = children.map((c) => Math.pow(c.N, 1 / tau));
  let r = rng.next() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < children.length; i++) {
    r -= w[i];
    if (r <= 0) return children[i].action;
  }
  return children[0].action;
}

export async function selfPlayBatched(job: SelfPlayJob & { gpuPort: number; concurrency: number }): Promise<void> {
  // league: older models get their own server slots
  const league = job.league;
  const models = new Map<number, string>([[0, job.player.net!]]);
  const slotOf = new Map<LeagueOpponent, number>();
  for (const o of league?.opponents ?? [])
    if (o.kind === "old") {
      slotOf.set(o, models.size);
      models.set(models.size, o.net);
    }
  const client = await connectSlots(job.gpuPort, models);
  const learnerToMove = (g: Game) => !g.opp || g.s.current === g.learner;
  let totalSamples = 0;
  await drive(client, job.concurrency, job.games, job.seed, {
    startGame(g) {
      if (!league?.opponents.length || g.rng.next() < league.pSelf) return;
      const o = league.opponents[Math.floor(g.rng.next() * league.opponents.length)];
      g.opp = o;
      g.oppSlot = slotOf.get(o) ?? 0;
      if (o.kind === "style") {
        // vary how single-minded the rush is from game to game (unless fixed)
        g.oppWeight = o.weight ?? 0.4 + 0.8 * g.rng.next();
        g.oppBoost = o.boost ?? 1 + 2 * g.rng.next();
      }
    },
    decide(g, legal) {
      g.full = g.rng.next() < job.pFull;
      const d: Decision = {
        slot: 0,
        sims: g.full ? job.player.sims : job.simsFast,
        legal,
        cPuct: job.player.cPuct,
        fpu: job.player.fpu,
        fpuReduction: job.player.fpuReduction,
        rootFpuReduction: job.player.rootFpuReduction,
        fpuScaled: job.player.fpuScaled,
        reuse: job.player.reuse,
        noise: g.full,
      };
      if (learnerToMove(g)) {
        d.mlh = mlhOf(job.player);
        return d;
      }
      // opponent: an older model, or the learner's network playing a one-track rush
      const o = g.opp!;
      if (o.kind === "old") d.slot = g.oppSlot;
      else {
        d.style = o.style;
        d.styleWeight = g.oppWeight;
        d.styleBoost = g.oppBoost;
      }
      return d;
    },
    finishDecision(g, res) {
      if (g.full) {
        const x = new Float32Array(NUM_FEATURES);
        encode(g.s, x);
        const pi = new Map<number, number>();
        let tot = 0;
        for (const c of res.children) tot += c.N;
        for (const c of res.children) if (c.N > 0) pi.set(c.action, (pi.get(c.action) ?? 0) + c.N / tot);
        let piPruned: Map<number, number> | undefined;
        if (job.outPruned && g.mcts) {
          const kept = prunedVisits(res.children, g.mcts.rootPriors ?? new Map(), g.mcts.cPuct);
          let kTot = 0;
          for (const n of kept.values()) kTot += n;
          piPruned = new Map();
          for (const c of res.children) {
            const n = kept.get(c.key) ?? 0;
            if (n > 0) piPruned.set(c.action, (piPruned.get(c.action) ?? 0) + n / kTot);
          }
        }
        const legal: number[] = [];
        const buf = new Int16Array(512);
        const n = legalActions(g.s, buf);
        for (let i = 0; i < n; i++) legal.push(buf[i]);
        // the opponent's search is biased or weaker: learn only the value of its positions
        g.samples.push({ x, legal, pi, piPruned, mover: g.s.current, q: res.value, ply: g.s.ply, valueOnly: !learnerToMove(g) });
      }
      return pickByVisits(res.children, g.decisions < job.tempMoves ? 1 : 0, g.rng);
    },
    finishGame(g) {
      const s = g.s;
      const out = Buffer.alloc(g.samples.length * RECORD_BYTES);
      const fin = finalStats(s);
      const z = (smp: Sample) => (s.winner === DRAW ? 0 : s.winner === smp.mover ? 1 : -1);
      g.samples.forEach((smp, i) => writeRecord(out, i * RECORD_BYTES, smp, z(smp), s.ply - smp.ply, fin));
      appendFileSync(job.out, out);
      if (job.outPruned) {
        // the same records with pruned policy targets (value-only records have none to prune)
        g.samples.forEach((smp, i) => writeRecord(out, i * RECORD_BYTES, { ...smp, pi: smp.piPruned ?? smp.pi }, z(smp), s.ply - smp.ply, fin));
        appendFileSync(job.outPruned, out);
      }
      totalSamples += g.samples.length;
      const learnerScore = s.winner === DRAW ? 0.5 : s.winner === g.learner ? 1 : 0;
      parentPort!.postMessage({
        type: "progress",
        id: job.id,
        game: g.index + 1,
        samples: totalSamples,
        winner: s.winner,
        reason: s.winReason,
        plies: s.ply,
        opp: g.opp?.name ?? null,
        learnerScore,
      });
    },
  });
  client.close();
}

export async function matchBatched(job: MatchJob & { gpuPort: number; concurrency: number }): Promise<void> {
  const slotOf = (spec: PlayerSpec, slot: number) => (spec.kind === "net" ? slot : -1);
  const models = new Map<number, string>();
  if (job.a.kind === "net") models.set(0, job.a.net!);
  if (job.b.kind === "net") models.set(1, job.b.net!);
  const client = await connectSlots(job.gpuPort, models);
  await drive(client, job.concurrency, job.games, job.seed, {
    decide(g, legal) {
      const aToMove = (g.s.current === 0) === g.aFirst;
      const spec = aToMove ? job.a : job.b;
      return {
        slot: slotOf(spec, aToMove ? 0 : 1),
        sims: spec.sims,
        auto: spec.auto,
        legal,
        cPuct: spec.cPuct,
        fpu: spec.fpu,
        fpuReduction: spec.fpuReduction,
        rootFpuReduction: spec.rootFpuReduction,
        fpuScaled: spec.fpuScaled,
        reuse: spec.reuse,
        noise: false,
        mlh: mlhOf(spec),
        style: spec.style,
        styleWeight: spec.styleWeight,
        styleBoost: spec.styleBoost,
      };
    },
    finishDecision(g, res) {
      const side = (g.s.current === 0) === g.aFirst ? 0 : 1;
      g.simsUsed[side] += g.done;
      g.decisionsBy[side]++;
      if (g.sureAt[side] < 0 && (res.children[0]?.Q ?? 0) > SURE_Q) g.sureAt[side] = g.s.ply;
      return res.action;
    },
    finishGame(g) {
      const s = g.s;
      const aIdx = g.aFirst ? 0 : 1;
      const result = s.winner === DRAW ? 0.5 : s.winner === aIdx ? 1 : 0;
      // conversion: actions from first being clearly winning (own search) to the win
      const w = s.winner === aIdx ? 0 : 1;
      const convert = s.winner !== DRAW && g.sureAt[w] >= 0 ? s.ply - g.sureAt[w] : -1;
      parentPort!.postMessage({ type: "game", id: job.id, result, reason: s.winReason, winnerIsA: s.winner === aIdx, plies: s.ply, sims: g.simsUsed, decisions: g.decisionsBy, convert });
    },
  });
  client.close();
}
