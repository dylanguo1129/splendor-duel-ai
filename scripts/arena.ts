// Quick strength ladder. Usage: npx tsx scripts/arena.ts <A> <B> <games>
// Agents: random | greedy | mcts:<iterations>
import { Agent, GreedyAgent, HeuristicEvaluator, MctsAgent, RandomAgent } from "../ai/agents";
import { match } from "../ai/arena";
import { Rng } from "../engine/rng";

let seq = 0;
function make(spec: string): () => Agent {
  return () => {
    seq++;
    if (spec === "random") return new RandomAgent(new Rng(seq));
    if (spec === "greedy") return new GreedyAgent(new Rng(seq));
    const m = /^mcts:(\d+)(?::c([\d.]+))?$/.exec(spec);
    if (m) return new MctsAgent(new HeuristicEvaluator(), { iterations: Number(m[1]) }, { seed: seq * 7919, cPuct: m[2] ? Number(m[2]) : undefined }, spec);
    throw new Error(`unknown agent ${spec}`);
  };
}

const [A, B, n] = process.argv.slice(2);
const t0 = performance.now();
const r = match(make(A), make(B), Number(n ?? 20), 5000, (i, res) => {
  if ((i + 1) % 10 === 0) console.log(`${i + 1} games: ${A} ${res.wins}-${res.losses}-${res.draws} vs ${B}`);
});
console.log(
  `${A} vs ${B}: ${r.wins}-${r.losses}-${r.draws} score ${(r.score * 100).toFixed(1)}% elo ${r.elo.toFixed(0)}±${r.eloCI95.toFixed(0)} avgPlies ${r.avgPlies.toFixed(0)} reasons ${JSON.stringify(r.reasons)} (${((performance.now() - t0) / 1000).toFixed(1)}s)`,
);
