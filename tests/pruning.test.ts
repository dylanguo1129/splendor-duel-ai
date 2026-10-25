import { describe, expect, it } from "vitest";
import { ChildStat, Evaluator, Mcts, prunedVisits } from "../ai/mcts";
import { Rng } from "../engine/rng";
import { GameState, newGame } from "../engine/state";

const child = (key: number, N: number, Q: number, prior: number): ChildStat => ({ action: key, key, N, Q, prior });

describe("policy-target pruning", () => {
  it("keeps the most visited child and visits that its PUCT score justifies", () => {
    const kids = [child(1, 300, 0.2, 0.5), child(2, 80, 0.19, 0.3), child(3, 20, 0.18, 0.2)];
    const kept = prunedVisits(kids, new Map(kids.map((c) => [c.key, c.prior])), 1.5);
    expect(kept.get(1)).toBe(300);
    // close values: little or nothing is exploration
    expect(kept.get(2)!).toBeGreaterThan(60);
    expect(kept.get(3)!).toBeGreaterThan(10);
  });

  it("drops visits that only noise or first-play urgency produced", () => {
    // a poor move (much lower value) with a tiny clean prior kept getting visits
    const kids = [child(1, 380, 0.3, 0.6), child(2, 15, -0.6, 0.3), child(3, 5, 0.25, 0.1)];
    const clean = new Map([[1, 0.95], [2, 0.01], [3, 0.04]]);
    const kept = prunedVisits(kids, clean, 1.5);
    expect(kept.get(1)).toBe(380);
    expect(kept.has(2)).toBe(false);
    expect(kept.get(3) ?? 0).toBeLessThanOrEqual(5);
  });

  it("never adds visits and leaves an unexplored search alone", () => {
    expect(prunedVisits([], new Map(), 1.5).size).toBe(0);
    const kids = [child(1, 10, 0, 0.5), child(2, 0, 0, 0.5)];
    const kept = prunedVisits(kids, new Map(), 1.5);
    expect(kept.get(1)).toBe(10);
    expect(kept.has(2)).toBe(false);
  });

  it("records the root priors before Dirichlet noise", () => {
    const s: GameState = newGame(7);
    const ev: Evaluator = {
      evaluate(_s: GameState, _l: Int16Array, n: number, priors: Float32Array, _r: Rng): number {
        for (let i = 0; i < n; i++) priors[i] = i === 0 ? 10 : 1;
        return 0;
      },
    };
    const m = new Mcts(s, ev, { seed: 1, dirichletAlpha: 0.3 });
    m.run({ iterations: 50 });
    const raw = [...m.rootPriors!.values()];
    expect(raw.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    // un-noised: the favoured move's share is exactly 10 / (10 + n - 1)
    expect(Math.max(...raw)).toBeCloseTo(10 / (10 + raw.length - 1), 5);
  });
});
