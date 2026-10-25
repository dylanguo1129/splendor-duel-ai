import { determinize } from "../engine/determinize";
import { Rng } from "../engine/rng";
import { DEFAULT_RULES, legalActions, RulesConfig } from "../engine/rules";
import { GameState } from "../engine/state";
import { actionValues, DEFAULT_WEIGHTS, evaluate, EvalWeights } from "./eval";
import { Evaluator, Mcts, MctsOptions, SearchResult } from "./mcts";

export interface Agent {
  name: string;
  choose(s: GameState): number;
}

/** Heuristic evaluator: hand-written value + softmax of 1-ply action values as priors. */
export class HeuristicEvaluator implements Evaluator {
  private vals = new Float32Array(512);
  constructor(
    private w: EvalWeights = DEFAULT_WEIGHTS,
    private temperature = 0.08,
    private rules: RulesConfig = DEFAULT_RULES,
  ) {}
  evaluate(s: GameState, legal: Int16Array, n: number, priors: Float32Array, rng: Rng): number {
    if (n === 1) priors[0] = 1;
    else {
      actionValues(s, legal, n, rng, this.vals, this.rules, this.w);
      let max = -Infinity;
      for (let i = 0; i < n; i++) if (this.vals[i] > max) max = this.vals[i];
      for (let i = 0; i < n; i++) priors[i] = Math.exp((this.vals[i] - max) / this.temperature);
    }
    return evaluate(s, s.current, this.w);
  }
}

export class RandomAgent implements Agent {
  name = "random";
  private buf = new Int16Array(512);
  constructor(private rng = new Rng(99)) {}
  choose(s: GameState): number {
    const n = legalActions(s, this.buf);
    return this.buf[this.rng.int(n)];
  }
}

/** Picks the action with the best 1-ply heuristic value (on its own determinization: no peeking). */
export class GreedyAgent implements Agent {
  name = "greedy";
  private buf = new Int16Array(512);
  private vals = new Float32Array(512);
  constructor(
    private rng = new Rng(7),
    private w: EvalWeights = DEFAULT_WEIGHTS,
  ) {}
  choose(s: GameState): number {
    const d = determinize(s, s.current, this.rng);
    const n = legalActions(d, this.buf);
    if (n === 1) return this.buf[0];
    actionValues(d, this.buf, n, this.rng, this.vals, DEFAULT_RULES, this.w);
    let best = 0;
    for (let i = 1; i < n; i++) if (this.vals[i] > this.vals[best]) best = i;
    return this.buf[best];
  }
}

export class MctsAgent implements Agent {
  name: string;
  last?: SearchResult;
  private seed: number;
  constructor(
    private evaluator: Evaluator,
    private budget: { iterations?: number; timeMs?: number },
    private opts: MctsOptions = {},
    name?: string,
  ) {
    this.name = name ?? `mcts${budget.iterations ?? `${budget.timeMs}ms`}`;
    this.seed = opts.seed ?? 1;
  }
  choose(s: GameState): number {
    const buf = new Int16Array(512);
    if (legalActions(s, buf) === 1) return buf[0];
    const m = new Mcts(s, this.evaluator, { ...this.opts, seed: this.seed++ });
    m.run(this.budget);
    this.last = m.result();
    return this.last.action;
  }
}
