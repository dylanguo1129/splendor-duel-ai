// Style-biased players for league training and benchmarks: the normal network search, pushed toward
// one way of winning (10 crowns, 10 points in one colour, or 20 points). The styled player's search
// gets (1) larger priors for buying/reserving cards that advance its path and (2) a value bonus for
// progress on that path, so it plays a competent but one-track rush the learner must learn to stop.
import { CARDS } from "../engine/cards";
import { A_BUY, A_RESERVE } from "../engine/constants";
import type { Rng } from "../engine/rng";
import type { GameState, PlayerState } from "../engine/state";
import type { Evaluator } from "./mcts";

export type Style = "crowns" | "color" | "points";
export const STYLES: Style[] = ["crowns", "color", "points"];

/** Value bonus at full progress (progress is 0..1), e.g. 0.8 for crowns: +0.08 per crown. */
export const STYLE_WEIGHTS: Record<Style, number> = { crowns: 0.8, color: 0.8, points: 0.8 };

function bestColor(p: PlayerState): number {
  let best = 1;
  for (let c = 2; c <= 5; c++)
    if (p.colorPoints[c] > p.colorPoints[best] || (p.colorPoints[c] === p.colorPoints[best] && p.bonus[c] > p.bonus[best])) best = c;
  return best;
}

/** Player `pi`'s progress toward the style's win condition, in [0, 1]. */
export function styleProgress(s: GameState, pi: number, style: Style): number {
  const p = s.players[pi];
  if (style === "crowns") return Math.min(1, p.crowns / 10);
  if (style === "points") return Math.min(1, p.points / 20);
  return Math.min(1, p.colorPoints[bestColor(p)] / 10);
}

/** Card a buy/reserve action targets (-1 for other actions and blind deck reservations). */
function targetCard(s: GameState, a: number): number {
  if (a >= A_BUY && a < A_BUY + 15) {
    const src = a - A_BUY;
    return src < 12 ? s.pyramid[src] : (s.players[s.current].reserved[src - 12] ?? -1);
  }
  if (a >= A_RESERVE && a < A_RESERVE + 12) return s.pyramid[a - A_RESERVE];
  return -1;
}

/**
 * Prior multiplier for action `a` of the player to move in `s` when that player follows `style`;
 * `strength` scales the preference (1 = default).
 */
export function stylePriorBoost(s: GameState, a: number, style: Style, strength = 1): number {
  const id = targetCard(s, a);
  if (id < 0) return 1;
  const c = CARDS[id];
  if (style === "crowns") return 1 + 1.5 * strength * c.crowns;
  if (style === "points") return 1 + 0.4 * strength * c.points;
  const col = bestColor(s.players[s.current]);
  return c.color === col || c.joker ? 1 + strength * (1 + 0.5 * c.points) : 1;
}

/** Leaf value for `chooser`, shaped for the `styled` player's ambition. */
export function shapeValue(v: number, s: GameState, chooser: number, styled: number, style: Style, weight = STYLE_WEIGHTS[style]): number {
  const bonus = weight * styleProgress(s, styled, style);
  const out = chooser === styled ? v + bonus : v - bonus;
  return out > 1 ? 1 : out < -1 ? -1 : out;
}

/** CPU search: wraps an evaluator so that `styled` plays `style` (priors and values). */
export class StyledEvaluator implements Evaluator {
  constructor(
    private base: Evaluator,
    private style: Style,
    private styled: number,
    private weight = STYLE_WEIGHTS[style],
    private strength = 1,
  ) {}
  evaluate(s: GameState, legal: Int16Array, n: number, priors: Float32Array, rng: Rng): number {
    const v = this.base.evaluate(s, legal, n, priors, rng);
    if (s.current === this.styled) for (let i = 0; i < n; i++) priors[i] *= stylePriorBoost(s, legal[i], this.style, this.strength);
    return shapeValue(v, s, s.current, this.styled, this.style, this.weight);
  }
}
