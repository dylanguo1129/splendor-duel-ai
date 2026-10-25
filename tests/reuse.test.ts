import { describe, expect, it } from "vitest";
import { Evaluator, Mcts } from "../ai/mcts";
import { applyAction } from "../engine/rules";
import { Rng } from "../engine/rng";
import { cloneState, GameState, newGame } from "../engine/state";

/** Values that differ from position to position (player 0's view), uniform priors. */
const evaluator: Evaluator = {
  evaluate(s: GameState, _legal: Int16Array, n: number, priors: Float32Array, _rng: Rng): number {
    for (let i = 0; i < n; i++) priors[i] = 1;
    const v0 = Math.sin(s.ply * 1.7 + s.players[0].crowns + 3 * s.players[1].crowns) * 0.6;
    return s.current === 0 ? v0 : -v0;
  },
};

describe("search tree reuse", () => {
  it("re-roots at the played move with the value seen from the new player to move", () => {
    const s = newGame(11);
    const me = s.current;
    const m = new Mcts(s, evaluator, { seed: 2 });
    m.run({ iterations: 400 });
    const best = m.result().children[0];
    const next = cloneState(s);
    applyAction(next, best.action, new Rng(5));
    const kept = m.descend([best.key], next);
    expect(kept).toBe(best.N);
    const r = m.result();
    expect(r.iterations).toBe(best.N);
    expect(r.value).toBeCloseTo(next.current === me ? best.Q : -best.Q, 9);
  });

  it("keeps searching from the kept root", () => {
    const s = newGame(12);
    const m = new Mcts(s, evaluator, { seed: 3 });
    m.run({ iterations: 300 });
    const best = m.result().children[0];
    const next = cloneState(s);
    applyAction(next, best.action, new Rng(6));
    const kept = m.descend([best.key], next);
    m.run({ iterations: 200 });
    expect(m.result().iterations).toBe(kept + 200);
  });

  it("reports a position the tree never reached", () => {
    const s = newGame(13);
    const m = new Mcts(s, evaluator, { seed: 4 });
    m.run({ iterations: 50 });
    expect(m.descend([999999], s)).toBe(-1);
  });
});
