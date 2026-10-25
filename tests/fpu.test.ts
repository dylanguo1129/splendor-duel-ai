import { describe, expect, it } from "vitest";
import { Evaluator, Mcts } from "../ai/mcts";
import { Rng } from "../engine/rng";
import { GameState, newGame } from "../engine/state";

/** Every position is worth `v` to the root player; uniform priors. */
function constEvaluator(root: number, v: number): Evaluator {
  return {
    evaluate(s: GameState, _legal: Int16Array, n: number, priors: Float32Array, _rng: Rng): number {
      for (let i = 0; i < n; i++) priors[i] = 1;
      return s.current === root ? v : -v;
    },
  };
}

const visitedRootMoves = (v: number, fpuReduction?: number, fpuScaled?: boolean) => {
  const s = newGame(5);
  const m = new Mcts(s, constEvaluator(s.current, v), { seed: 3, fpuReduction, fpuScaled });
  m.run({ iterations: 150 }); // short: with longer searches every one of the 121 opening moves gets tried
  return m.result().children.filter((c) => c.N > 0).length;
};

describe("parent-relative first-play urgency", () => {
  it("widens the search of a winning player (fixed fpu makes unvisited moves look bad)", () => {
    expect(visitedRootMoves(0.9, 0.1)).toBeGreaterThan(visitedRootMoves(0.9));
  });

  it("narrows the search of a losing player (fixed fpu makes unvisited moves look good)", () => {
    expect(visitedRootMoves(-0.9, 0.1)).toBeLessThan(visitedRootMoves(-0.9));
  });

  it("reduces less while little of the policy has been visited when scaled", () => {
    expect(visitedRootMoves(0.9, 0.3, true)).toBeGreaterThan(visitedRootMoves(0.9, 0.3));
  });
});
