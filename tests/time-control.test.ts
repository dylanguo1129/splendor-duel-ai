import { describe, expect, it } from "vitest";
import { HeuristicEvaluator } from "../ai/agents";
import { Mcts } from "../ai/mcts";
import { outlookText } from "../ai/describe-search";
import { PlayoutRun, runPlayouts, summarizeOutlook } from "../ai/outlook";
import { autoTimeCheck, complexityFactor, runAuto, winFactor } from "../ai/time-control";
import { DRAW } from "../engine/constants";
import { Rng } from "../engine/rng";
import { listActions } from "../engine/rules";
import { newGame } from "../engine/state";

const kids = (...xs: [number, number][]) => xs.map(([N, Q]) => ({ N, Q }));

describe("auto time control", () => {
  it("thinks less when winning, more when close or behind", () => {
    expect(winFactor(0.99)).toBeLessThan(winFactor(0.9));
    expect(winFactor(0.9)).toBeLessThan(winFactor(0.7));
    expect(winFactor(0.5)).toBeGreaterThan(winFactor(0.7));
    expect(winFactor(0.2)).toBeGreaterThan(winFactor(0.5));
    expect(complexityFactor(2)).toBeLessThan(complexityFactor(20));
    expect(complexityFactor(20)).toBeCloseTo(1, 1);
    expect(complexityFactor(1000)).toBeLessThanOrEqual(1.3);
  });

  it("never stops before the minimum and always stops at the maximum", () => {
    const o = { base: 1000 };
    expect(autoTimeCheck(o, { spent: 150, iterations: 150, legal: 20, children: kids([150, 0.99], [0, 0]) }).stop).toBe(false);
    const d = autoTimeCheck(o, { spent: 3000, iterations: 3000, legal: 20, children: kids([1500, 0], [1490, 0.1]) });
    expect(d).toMatchObject({ stop: true, reason: "max" });
  });

  it("stops early once the runner-up cannot catch up", () => {
    // target ~1000 (20 moves, even game); 400 spent, 600 sims left; lead 700 > 600
    const d = autoTimeCheck({ base: 1000 }, { spent: 400, iterations: 400, legal: 20, children: kids([750, 0.4], [50, 0.3]) });
    expect(d).toMatchObject({ stop: true, reason: "decided" });
  });

  it("stops at the target when the choice is clear", () => {
    const d = autoTimeCheck({ base: 1000 }, { spent: 1000, iterations: 1000, legal: 20, children: kids([500, 0.4], [300, 0.38]) });
    expect(d).toMatchObject({ stop: true, reason: "target" });
  });

  it("extends a close decision up to twice the target", () => {
    const close = kids([400, 0.4], [380, 0.41]);
    const at = (spent: number) => autoTimeCheck({ base: 1000 }, { spent, iterations: spent, legal: 20, children: close });
    const t = at(1000).target;
    expect(at(1000).stop).toBe(false);
    expect(at(1.5 * t).stop).toBe(false);
    expect(at(2 * t)).toMatchObject({ stop: true, reason: "extended" });
  });

  it("plans less time for a won position", () => {
    const won = autoTimeCheck({ base: 1000 }, { spent: 300, iterations: 300, legal: 20, children: kids([200, 0.96], [100, 0.9]) });
    const even = autoTimeCheck({ base: 1000 }, { spent: 300, iterations: 300, legal: 20, children: kids([200, 0], [100, -0.05]) });
    expect(won.target).toBeLessThan(0.5 * even.target);
  });

  it("runs a real search within the budget bounds", () => {
    const s = newGame(7);
    const m = new Mcts(s, new HeuristicEvaluator(), { seed: 3 });
    const d = runAuto(m, { base: 200 }, listActions(s).length);
    const n = m.result().iterations;
    expect(d.stop).toBe(true);
    expect(n).toBeGreaterThanOrEqual(40);
    expect(n).toBeLessThanOrEqual(600 + 13);
  });
});

describe("turns-to-win outlook", () => {
  it("plays games to the end and counts the mover's current turn", () => {
    const s = newGame(11);
    const runs = runPlayouts(s, new HeuristicEvaluator(), new Rng(5), { count: 6 });
    expect(runs).toHaveLength(6);
    for (const r of runs) {
      expect(r.winner).not.toBe(-1);
      expect(r.turns[s.current]).toBeGreaterThanOrEqual(1);
      // players alternate (extra turns aside), so their counts stay close
      expect(Math.abs(r.turns[0] - r.turns[1])).toBeLessThanOrEqual(6);
    }
  });

  it("summarizes the favoured side's winning playouts", () => {
    const runs: PlayoutRun[] = [3, 4, 5, 6, 7, 8, 9, 10].map((t) => ({ winner: 0, reason: t < 6 ? 2 : 1, turns: [t, t - 1] }));
    runs.push({ winner: 1, reason: 3, turns: [2, 2] }, { winner: DRAW, reason: 5, turns: [50, 50] }, { winner: -1, reason: 0, turns: [99, 99] });
    const o = summarizeOutlook(runs, 0, 0.8)!;
    expect(o).toMatchObject({ side: 0, close: false, turns: 7, lo: 5, hi: 9, runs: 8 });
    // player 0 won 8 of the 9 finished playouts, mostly by points (reason 1); player 1 one by colour (3)
    expect(o.paths[0]).toEqual({ reason: 1, share: 8 / 9 });
    expect(o.paths[1]).toEqual({ reason: 3, share: 1 / 9 });
    expect(outlookText(o, 0, "你", "对手")).toBe("你预计还要约 7 个回合获胜，多半靠拿到 20 分（含本回合，大致 5–9 回合）");
    const threat = { ...o, paths: [o.paths[0], { reason: 2, share: 0.3 }] };
    expect(outlookText(threat, 0, "你", "对手")).toContain("；对手的威胁：集齐 10 皇冠（约 30%）");
    expect(outlookText(threat, 1, "你", "对手")).toMatch(/^对手预计再走约 7 个回合获胜，多半靠拿到 20 分.*；你翻盘路线：集齐 10 皇冠/);
    // the mover is losing: the estimate is about the opponent (player 0)
    expect(summarizeOutlook(runs, 1, 0.3)).toMatchObject({ side: 0, close: false });
    // the favourite rarely won the playouts: all finished playouts count
    expect(summarizeOutlook(runs, 0, 0.2)).toMatchObject({ side: 1, close: false, runs: 9 });
    // no favourite: the mover's turns until the game is decided, whoever wins
    expect(summarizeOutlook(runs, 1, 0.55)).toMatchObject({ side: 1, close: true, runs: 9 });
    expect(summarizeOutlook([{ winner: -1, reason: 0, turns: [9, 9] }], 0, 0.5)).toBeNull();
  });
});
