// "How many turns until someone wins": play the game out from the current position many times
// with the network's policy (no search), each from a fresh determinization of the hidden cards,
// and count the turns each player still gets.
import { DRAW, Phase } from "../engine/constants";
import { determinize } from "../engine/determinize";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, legalActions } from "../engine/rules";
import type { GameState } from "../engine/state";
import type { Evaluator } from "./mcts";

export interface PlayoutRun {
  winner: number; // 0/1, DRAW, or -1 when cut off before the end
  reason: number; // engine WIN_REASON_* (1 points, 2 crowns, 3 one colour, ...), 0 when cut off
  turns: [number, number]; // turns each player starts from now on, the current turn included
}

export interface PlayoutOptions {
  count: number;
  timeMs?: number; // stop early when this is used up (at least one playout always runs)
  temperature?: number; // policy sampling temperature (0 = always the top move)
  rootActions?: number[]; // restrict the first move (part of a multi-step move was already played)
  resampleOwnHidden?: boolean; // spectator view: the mover's blind reservations are unknown too
  maxPlies?: number;
}

export function runPlayouts(state: GameState, evaluator: Evaluator, rng: Rng, o: PlayoutOptions): PlayoutRun[] {
  const deadline = o.timeMs ? performance.now() + o.timeMs : Infinity;
  const tau = o.temperature ?? 0.5;
  const maxPlies = o.maxPlies ?? 400;
  const legal = new Int16Array(512);
  const priors = new Float32Array(512);
  const allowed = o.rootActions ? new Set(o.rootActions) : null;
  const runs: PlayoutRun[] = [];
  while (runs.length < o.count && (runs.length === 0 || performance.now() < deadline)) {
    const s = determinize(state, state.current, rng, o.resampleOwnHidden ?? false);
    const turns: [number, number] = [0, 0];
    turns[s.current] = 1;
    let first = true;
    for (let ply = 0; s.phase !== Phase.GameOver && ply < maxPlies; ply++) {
      let n = legalActions(s, legal);
      if (first && allowed) {
        let k = 0;
        for (let i = 0; i < n; i++) if (allowed.has(legal[i])) legal[k++] = legal[i];
        if (k > 0) n = k;
      }
      first = false;
      let a = legal[0];
      if (n > 1) {
        evaluator.evaluate(s, legal, n, priors, rng);
        a = legal[pick(priors, n, tau, rng)];
      }
      const turn = s.turn;
      applyAction(s, a, rng, DEFAULT_RULES);
      if (s.turn !== turn && (s.phase as Phase) !== Phase.GameOver) turns[s.current]++;
    }
    const over = s.phase === Phase.GameOver;
    runs.push({ winner: over ? s.winner : -1, reason: over ? s.winReason : 0, turns });
  }
  return runs;
}

function pick(priors: Float32Array, n: number, tau: number, rng: Rng): number {
  let best = 0;
  for (let i = 1; i < n; i++) if (priors[i] > priors[best]) best = i;
  if (tau <= 0.01) return best;
  // priors are exp(logit - max): raise to 1/tau for the tempered distribution
  let sum = 0;
  for (let i = 0; i < n; i++) sum += priors[i] = Math.pow(priors[i], 1 / tau);
  let r = rng.next() * sum;
  for (let i = 0; i < n; i++) {
    r -= priors[i];
    if (r <= 0) return i;
  }
  return best;
}

export interface Outlook {
  side: number; // player the estimate is about: the one favoured to win (the mover when close)
  close: boolean; // no clear favourite: the turns are until the game is decided
  turns: number; // median turns that player needs (current turn included if it is theirs)
  lo: number; // 25th..75th percentile
  hi: number;
  runs: number;
  /** Per player: how they most often won the playouts (WIN_REASON_*) and their share of all finished playouts. */
  paths: { reason: number; share: number }[];
}

/** Most frequent way each player won the finished playouts. */
function winPaths(finished: PlayoutRun[]): { reason: number; share: number }[] {
  return [0, 1].map((pl) => {
    const won = finished.filter((r) => r.winner === pl);
    const count = new Map<number, number>();
    for (const r of won) count.set(r.reason, (count.get(r.reason) ?? 0) + 1);
    let reason = 0;
    for (const [k, n] of count) if (!reason || n > count.get(reason)!) reason = k;
    return { reason, share: won.length / Math.max(1, finished.length) };
  });
}

/**
 * Turns the favoured side (by the search's win probability `winProb` for `mover`) needs to win,
 * from the playouts that side won; falls back to all finished playouts when it rarely won them.
 * Within 40-60% there is no favourite: the mover's turns until the game ends, whoever wins.
 */
export function summarizeOutlook(runs: PlayoutRun[], mover: number, winProb: number): Outlook | null {
  const close = Math.abs(winProb - 0.5) < 0.1;
  const side = close || winProb >= 0.5 ? mover : 1 - mover;
  const finished = runs.filter((r) => r.winner !== -1 && r.winner !== DRAW);
  let pool = finished.filter((r) => r.winner === side);
  if (close || pool.length < Math.max(4, finished.length / 4)) pool = finished;
  if (pool.length === 0) return null;
  const t = pool.map((r) => r.turns[side]).sort((x, y) => x - y);
  // policy-only playouts last a little longer than games played with search: x0.95 removes the
  // bias measured by train/outlook-eval.ts (60 games, 448 positions: mean error 1.36 turns)
  const q = (f: number) => Math.max(1, Math.round(0.95 * t[Math.min(t.length - 1, Math.floor(f * t.length))]));
  return { side, close, turns: q(0.5), lo: q(0.25), hi: q(0.75), runs: pool.length, paths: winPaths(finished) };
}
