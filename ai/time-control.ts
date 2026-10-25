// Adaptive thinking time ("auto" mode): think less when the position is easy, already decided or
// clearly won; think more when it is complex, close, or going badly. Works in abstract budget units:
// milliseconds in the browser, simulations in offline matches (so the policy can be measured there).
import type { Mcts } from "./mcts";

export interface AutoTimeOptions {
  base: number; // budget for a typical position (the average is close to this)
  minFrac?: number; // never stop before base * minFrac (default 0.2)
  maxFrac?: number; // never think longer than base * maxFrac (default 3)
}

/** Root statistics at a checkpoint, children sorted by visits (descending). */
export interface SearchSnapshot {
  spent: number; // budget units used so far
  iterations: number; // simulations so far (all trees together)
  legal: number; // legal root actions
  children: { N: number; Q: number }[];
}

export type StopReason = "decided" | "target" | "extended" | "max";

export interface TimeDecision {
  stop: boolean;
  reason?: StopReason;
  target: number; // planned budget for this position (before any extension)
}

const lerp = (xs: number[], ys: number[], x: number): number => {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++)
    if (x <= xs[i]) return ys[i - 1] + ((ys[i] - ys[i - 1]) * (x - xs[i - 1])) / (xs[i] - xs[i - 1]);
  return ys[ys.length - 1];
};

/** Time multiplier by the mover's win probability: long when behind or close, short when winning. */
export function winFactor(p: number): number {
  return lerp([0, 0.15, 0.35, 0.5, 0.7, 0.85, 0.93, 0.98], [1.3, 1.4, 1.3, 1.15, 1, 0.8, 0.55, 0.35], p);
}

/** Time multiplier by the number of legal moves (about 1 at 20 moves). */
export function complexityFactor(legal: number): number {
  return Math.min(1.3, Math.max(0.6, 0.55 + 0.15 * Math.log(Math.max(1, legal))));
}

export function autoTimeCheck(o: AutoTimeOptions, s: SearchSnapshot): TimeDecision {
  const min = o.base * (o.minFrac ?? 0.2);
  const max = o.base * (o.maxFrac ?? 3);
  const [a, b] = s.children;
  // win probability of the move the search currently prefers (mean root value lags behind it)
  const p = a ? (a.Q + 1) / 2 : 0.5;
  const target = Math.min(max, Math.max(min, o.base * complexityFactor(s.legal) * winFactor(p)));
  if (s.spent >= max) return { stop: true, reason: "max", target };
  if (s.spent < min || !a || !b) return { stop: false, target };
  if (s.spent < target) {
    // decided: the runner-up cannot overtake the leader in the planned time that is left
    const rate = s.iterations / Math.max(1e-9, s.spent);
    if (a.N - b.N <= (target - s.spent) * rate) return { stop: false, target };
    // "decided" only when it saves real time; just short of the target it is the planned stop
    return { stop: true, reason: s.spent < 0.85 * target ? "decided" : "target", target };
  }
  // at the target: keep going (up to twice the target) while the choice is unsettled
  const unstable = b.Q > a.Q + 0.03 && b.N >= 0.25 * a.N; // most-visited is not the best-valued
  const close = b.N > 0.8 * a.N;
  if ((unstable || close) && s.spent < Math.min(max, 2 * target)) return { stop: false, target };
  return { stop: true, reason: s.spent > target * 1.05 ? "extended" : "target", target };
}

/** Single-tree auto search with the budget in simulations; returns the decision that ended it. */
export function runAuto(mcts: Mcts, o: AutoTimeOptions, legal: number, checks = 16): TimeDecision {
  const chunk = Math.max(8, Math.round(o.base / checks));
  let spent = 0;
  for (;;) {
    mcts.run({ iterations: chunk });
    spent += chunk;
    const r = mcts.result();
    const d = autoTimeCheck(o, { spent, iterations: spent, legal, children: r.children });
    if (d.stop) return d;
  }
}
