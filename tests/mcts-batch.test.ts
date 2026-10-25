import { describe, expect, it } from "vitest";
import { HeuristicEvaluator } from "../ai/agents";
import { Mcts } from "../ai/mcts";
import { Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, legalActions } from "../engine/rules";
import { newGame } from "../engine/state";

function midGame(seed: number) {
  const s = newGame(seed);
  const rng = new Rng(seed);
  const buf = new Int16Array(512);
  for (let k = 0; k < 25 && s.phase !== Phase.GameOver; k++) applyAction(s, buf[rng.int(legalActions(s, buf))], rng);
  return s;
}

describe("resumable MCTS iterations", () => {
  it("interleaved begin/complete across trees equals running each tree alone", () => {
    const ev = new HeuristicEvaluator();
    const states = [midGame(11), midGame(12)];
    const solo = states.map((s, i) => {
      const m = new Mcts(s, ev, { seed: 100 + i, dirichletAlpha: 0.3 });
      m.run({ iterations: 150 });
      return m.result();
    });
    const trees = states.map((s, i) => new Mcts(s, ev, { seed: 100 + i, dirichletAlpha: 0.3 }));
    const priors = new Float32Array(512);
    const evRng = new Rng(0);
    for (let it = 0; it < 150; it++) {
      // collect one pending leaf per tree, then evaluate them "as a batch"
      const pend = trees.map((t) => t.beginIteration());
      pend.forEach((p, i) => {
        if (!p) return;
        void evRng;
        const v = ev.evaluate(p.s, p.legal, p.n, priors, (trees[i] as any).rng);
        trees[i].completeIteration(p, v, priors);
      });
    }
    const batched = trees.map((t) => t.result());
    for (let i = 0; i < 2; i++) {
      expect(batched[i].iterations).toBe(solo[i].iterations);
      expect(batched[i].children.map((c) => [c.action, c.N])).toEqual(solo[i].children.map((c) => [c.action, c.N]));
    }
  });
});
