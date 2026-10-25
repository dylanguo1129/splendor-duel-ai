// Iterations/second of MCTS with the heuristic evaluator and with a network.
import { readFileSync } from "node:fs";
import { HeuristicEvaluator } from "../ai/agents";
import { Mcts } from "../ai/mcts";
import { Net, NetEvaluator } from "../ai/net";
import { Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, legalActions } from "../engine/rules";
import { newGame } from "../engine/state";

const netPath = process.argv[2];
const evs: [string, any][] = [["heuristic", new HeuristicEvaluator()]];
if (netPath) evs.push(["net", new NetEvaluator(new Net(JSON.parse(readFileSync(netPath, "utf8"))))]);

// a few mid-game positions
const rng = new Rng(3);
const positions = [];
for (let g = 0; g < 6; g++) {
  const s = newGame(g);
  const buf = new Int16Array(512);
  for (let k = 0; k < 20 + g * 5 && s.phase !== Phase.GameOver; k++) applyAction(s, buf[rng.int(legalActions(s, buf))], rng);
  if (s.phase !== Phase.GameOver) positions.push(s);
}
for (const [name, ev] of evs) {
  const t0 = performance.now();
  let it = 0;
  for (const s of positions) {
    const m = new Mcts(s, ev, { seed: 1 });
    it += m.run({ iterations: 2000 });
  }
  const dt = (performance.now() - t0) / 1000;
  console.log(`${name}: ${(it / dt).toFixed(0)} iterations/s`);
}
