// Web Worker entry: runs one root-parallel MCTS tree per request.
import { HeuristicEvaluator } from "./agents";
import { DEFAULT_FPU_REDUCTION, DEFAULT_MLH, Evaluator, Mcts, MovesLeftOptions, SearchResult } from "./mcts";
import { Net, NetEvaluator, NetJson } from "./net";
import { PlayoutRun, runPlayouts } from "./outlook";
import { Rng } from "../engine/rng";
import type { GameState } from "../engine/state";

export type WorkerRequest =
  | { type: "init"; net: NetJson | null }
  | {
      type: "search";
      id: number;
      state: GameState;
      timeMs?: number;
      iterations?: number;
      seed: number;
      rootActions?: number[];
      resampleOwnHidden?: boolean;
      /** Keep the tree for "continue" requests (until "end"). */
      keep?: boolean;
    }
  | { type: "continue"; id: number; timeMs: number }
  | { type: "end"; id: number }
  | {
      type: "playouts";
      id: number;
      state: GameState;
      count: number;
      timeMs: number;
      seed: number;
      rootActions?: number[];
      resampleOwnHidden?: boolean;
    };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "result"; id: number; result: SearchResult }
  | { type: "playouts"; id: number; runs: PlayoutRun[] };

let evaluator: Evaluator = new HeuristicEvaluator();
let mlh: MovesLeftOptions | undefined; // quicker wins, when the model has a trained moves-left head
let fpuReduction: number | undefined; // unvisited moves valued relative to their parent (network only)
const kept = new Map<number, Mcts>();
const post = (m: WorkerResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const m = e.data;
  switch (m.type) {
    case "init": {
      const net = m.net ? new Net(m.net) : null;
      evaluator = net ? new NetEvaluator(net) : new HeuristicEvaluator();
      mlh = net?.hasMovesLeft ? DEFAULT_MLH : undefined;
      fpuReduction = net ? DEFAULT_FPU_REDUCTION : undefined;
      kept.clear();
      post({ type: "ready" });
      return;
    }
    case "search": {
      const mcts = new Mcts(m.state, evaluator, { seed: m.seed, rootActions: m.rootActions, resampleOwnHidden: m.resampleOwnHidden, mlh, fpuReduction });
      mcts.run({ timeMs: m.timeMs, iterations: m.iterations });
      if (m.keep) kept.set(m.id, mcts);
      post({ type: "result", id: m.id, result: mcts.result() });
      return;
    }
    case "continue": {
      const mcts = kept.get(m.id)!;
      mcts.run({ timeMs: m.timeMs });
      post({ type: "result", id: m.id, result: mcts.result() });
      return;
    }
    case "end":
      kept.delete(m.id);
      return;
    case "playouts": {
      const runs = runPlayouts(m.state, evaluator, new Rng(m.seed), {
        count: m.count,
        timeMs: m.timeMs,
        rootActions: m.rootActions,
        resampleOwnHidden: m.resampleOwnHidden,
      });
      post({ type: "playouts", id: m.id, runs });
      return;
    }
  }
};
