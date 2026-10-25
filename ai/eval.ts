// Hand-written evaluation used by the baseline search (value + 1-ply action priors).
import { CARDS } from "../engine/cards";
import { DRAW, GOLD, PEARL, Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, goldNeeded, legalActions, RulesConfig } from "../engine/rules";
import { cloneState, GameState, PlayerState } from "../engine/state";

export interface EvalWeights {
  progMax: number;
  progSum: number;
  bonus: number;
  token: number;
  gold: number;
  pearl: number;
  privilege: number;
  reserved: number;
  nearRoyal: number;
  affordable: number;
  scale: number;
  tempo: number;
}

export const DEFAULT_WEIGHTS: EvalWeights = {
  progMax: 2.0,
  progSum: 0.6,
  bonus: 0.07,
  token: 0.025,
  gold: 0.05,
  pearl: 0.045,
  privilege: 0.035,
  reserved: 0.25,
  nearRoyal: 0.08,
  affordable: 0.25,
  scale: 1.6,
  tempo: 0.06,
};

function maxColorPoints(p: PlayerState): number {
  let m = 0;
  for (let c = 1; c <= 5; c++) if (p.colorPoints[c] > m) m = p.colorPoints[c];
  return m;
}

/** Value of a card for player p, in "progress" units (fraction of a win condition). */
function cardProgress(p: PlayerState, id: number): number {
  const c = CARDS[id];
  const toPoints = c.points / 20;
  const toCrowns = c.crowns / 10;
  const col = c.joker ? maxColorPoints(p) : c.color <= 5 ? p.colorPoints[c.color] : 0;
  const toColor = c.color === 9 && !c.joker ? 0 : (col + c.points) / 10 - col / 10;
  return Math.max(toPoints, toCrowns, toColor) + 0.3 * (toPoints + toCrowns + toColor) + 0.03 * c.bonus;
}

export function playerScore(s: GameState, pi: number, w: EvalWeights): number {
  const p = s.players[pi];
  const fp = Math.min(1, p.points / 20);
  const fc = Math.min(1, p.crowns / 10);
  const fm = Math.min(1, maxColorPoints(p) / 10);
  let bonus = 0;
  for (let c = 1; c <= 5; c++) bonus += Math.min(p.bonus[c], 4);
  let tokens = 0;
  for (let t = 1; t <= 5; t++) tokens += p.tokens[t];
  let score =
    w.progMax * Math.max(fp, fc, fm) +
    w.progSum * (fp + fc + fm) +
    w.bonus * bonus +
    w.token * tokens +
    w.gold * p.tokens[GOLD] +
    w.pearl * p.tokens[PEARL] +
    w.privilege * p.privileges;
  if (p.royalsEarned < 2 && (p.crowns === 2 || p.crowns === 5)) score += w.nearRoyal;
  // best card reachable soon (pyramid or own reserve), discounted by gold still missing
  let best = 0;
  const consider = (id: number) => {
    const g = goldNeeded(p, id);
    if (g < 0) return;
    const missing = Math.max(0, g - p.tokens[GOLD]);
    const v = cardProgress(p, id) / (1 + 0.5 * missing);
    if (v > best) best = v;
  };
  for (let slot = 0; slot < 12; slot++) if (s.pyramid[slot] >= 0) consider(s.pyramid[slot]);
  for (const id of p.reserved) {
    consider(id);
    score += w.reserved * 0.15 * cardProgress(p, id);
  }
  score += w.affordable * best;
  return score;
}

/** Value in [-1, 1] for `perspective`. */
export function evaluate(s: GameState, perspective: number, w: EvalWeights = DEFAULT_WEIGHTS): number {
  if (s.phase === Phase.GameOver) return s.winner === DRAW ? 0 : s.winner === perspective ? 1 : -1;
  let d = playerScore(s, perspective, w) - playerScore(s, 1 - perspective, w);
  d += s.current === perspective ? w.tempo : -w.tempo;
  return Math.tanh(w.scale * d);
}

const subBuf = new Int16Array(512);

/** Resolve the rest of the current player's turn (sub-phases only) greedily by 1-ply evaluation. */
export function finishTurnGreedy(s: GameState, rng: Rng, cfg: RulesConfig = DEFAULT_RULES, w: EvalWeights = DEFAULT_WEIGHTS): void {
  const me = s.current;
  for (let guard = 0; guard < 20; guard++) {
    if (s.phase === Phase.Main || s.phase === Phase.GameOver || s.current !== me) return;
    const n = legalActions(s, subBuf);
    if (n === 1) {
      applyAction(s, subBuf[0], rng, cfg);
      continue;
    }
    let bestA = subBuf[0];
    let bestV = -Infinity;
    const opts = Array.from(subBuf.subarray(0, n));
    for (const a of opts) {
      const t = cloneState(s);
      applyAction(t, a, rng, cfg);
      const v = evaluate(t, me, w);
      if (v > bestV) {
        bestV = v;
        bestA = a;
      }
    }
    applyAction(s, bestA, rng, cfg);
  }
}

/** 1-ply scores (value for the mover after the action and its sub-decisions) for each legal action. */
export function actionValues(
  s: GameState,
  legal: Int16Array | number[],
  n: number,
  rng: Rng,
  out: Float32Array,
  cfg: RulesConfig = DEFAULT_RULES,
  w: EvalWeights = DEFAULT_WEIGHTS,
): void {
  const me = s.current;
  for (let i = 0; i < n; i++) {
    const t = cloneState(s);
    applyAction(t, legal[i], rng, cfg);
    finishTurnGreedy(t, rng, cfg, w);
    out[i] = evaluate(t, me, w);
  }
}
