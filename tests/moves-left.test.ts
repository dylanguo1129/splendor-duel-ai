import { describe, expect, it } from "vitest";
import { DEFAULT_MLH, Evaluator, Mcts } from "../ai/mcts";
import { A_TAKE_LINE, NUM_LINES } from "../engine/constants";
import { LINES } from "../engine/board";
import { Rng } from "../engine/rng";
import { GameState, newGame, tokenCount } from "../engine/state";

/**
 * Root player clearly winning everywhere (value 0.9 for them); the game is estimated to end in 10
 * actions once they hold 3+ tokens, in 40 otherwise. Uniform priors.
 */
function fakeEvaluator(root: number): Evaluator {
  return {
    movesLeft: 0,
    evaluate(s: GameState, _legal: Int16Array, n: number, priors: Float32Array, _rng: Rng): number {
      for (let i = 0; i < n; i++) priors[i] = 1;
      this.movesLeft = tokenCount(s.players[root]) >= 3 ? 10 : 40;
      return s.current === root ? 0.9 : -0.9;
    },
  };
}

const visitsOnQuickLines = (s: GameState, mlh?: typeof DEFAULT_MLH) => {
  // unvisited moves look as good as the rest, so the search spreads over the root moves
  const m = new Mcts(s, fakeEvaluator(s.current), { seed: 3, mlh, fpu: 0.9 });
  m.run({ iterations: 3000 });
  const r = m.result();
  const total = r.children.reduce((a, c) => a + c.N, 0);
  // three-token lines are the actions that reach 3 tokens right away
  const quick = r.children.filter((c) => c.action >= A_TAKE_LINE && c.action < A_TAKE_LINE + NUM_LINES && LINES[c.action - A_TAKE_LINE].length === 3);
  return { share: quick.reduce((a, c) => a + c.N, 0) / total, r };
};

describe("moves-left search", () => {
  it("tracks the evaluator's estimate of actions left", () => {
    const s = newGame(5);
    const { r } = visitsOnQuickLines(s, DEFAULT_MLH);
    for (const c of r.children) {
      expect(Number.isFinite(c.M!)).toBe(true);
      expect(c.M!).toBeGreaterThanOrEqual(10);
    }
  });

  it("changes nothing at weight 0", () => {
    const s = newGame(5);
    const off = visitsOnQuickLines(s).r.children.map((c) => [c.action, c.N]);
    const zero = visitsOnQuickLines(s, { ...DEFAULT_MLH, weight: 0 }).r.children.map((c) => [c.action, c.N]);
    expect(zero).toEqual(off);
  });

  it("steers a winning player toward the line that ends sooner", () => {
    const s = newGame(5);
    const off = visitsOnQuickLines(s).share;
    const on = visitsOnQuickLines(s, { ...DEFAULT_MLH, weight: 0.3 }).share;
    expect(on).toBeGreaterThan(off + 0.05);
  });
});
