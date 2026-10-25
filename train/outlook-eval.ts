// Calibration of the "turns until the win" estimate (ai/outlook.ts): play games with search, and at
// sampled positions compare the playout estimate with how many turns the game actually lasted.
//   node dist/exp/outlook-eval.mjs <model.json> [--games 15] [--sims 100] [--seed 1] [--every 4]
// Prints one JSON line per sampled position and temperature.
import { readFileSync } from "node:fs";
import { Mcts } from "../ai/mcts";
import { Net, NetEvaluator, NetJson } from "../ai/net";
import { runPlayouts, summarizeOutlook } from "../ai/outlook";
import { Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES, legalActions } from "../engine/rules";
import { cloneState, GameState, newGame } from "../engine/state";

const args = process.argv.slice(2);
const opt = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const ev = new NetEvaluator(new Net(JSON.parse(readFileSync(args[0], "utf8")) as NetJson));
const games = opt("games", 15);
const sims = opt("sims", 100);
const seed = opt("seed", 1);
const every = opt("every", 4);
const TAUS = [0, 0.5, 1];
const rng = new Rng(seed * 104729);
const buf = new Int16Array(512);

for (let g = 0; g < games; g++) {
  const s = newGame(seed * 1000 + g, 0);
  const started: [number, number] = [1, 0]; // turns started by each player so far
  const samples: { state: GameState; mover: number; at: [number, number]; p: number }[] = [];
  let decisions = 0;
  while (s.phase !== Phase.GameOver) {
    const n = legalActions(s, buf);
    let a = buf[0];
    if (n > 1) {
      const m = new Mcts(s, ev, { seed: rng.nextU32() });
      m.run({ iterations: sims });
      const r = m.result();
      a = r.action;
      if (decisions % every === every - 1)
        samples.push({ state: cloneState(s), mover: s.current, at: [started[0], started[1]], p: (r.children[0].Q + 1) / 2 });
      decisions++;
    }
    const turn = s.turn;
    applyAction(s, a, rng, DEFAULT_RULES);
    if (s.turn !== turn && (s.phase as Phase) !== Phase.GameOver) started[s.current]++;
  }
  for (const smp of samples) {
    // actual turns from the sample on, counted like runPlayouts (the mover's current turn included)
    const actual = [0, 1].map((k) => started[k] - smp.at[k] + (k === smp.mover ? 1 : 0));
    for (const tau of TAUS) {
      const t0 = performance.now();
      const runs = runPlayouts(smp.state, ev, rng, { count: 32, temperature: tau });
      const ms = performance.now() - t0;
      const o = summarizeOutlook(runs, smp.mover, smp.p);
      console.log(
        JSON.stringify({
          game: `${seed}-${g}`,
          tau,
          p: +smp.p.toFixed(3),
          side: o?.side ?? null,
          pred: o?.turns ?? null,
          lo: o?.lo ?? null,
          hi: o?.hi ?? null,
          winner: s.winner,
          actualSide: o ? actual[o.side] : null,
          ms: Math.round(ms),
        }),
      );
    }
  }
}
