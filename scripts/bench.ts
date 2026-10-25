import { Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, legalActions } from "../engine/rules";
import { cloneState, newGame } from "../engine/state";

const rng = new Rng(1);
const buf = new Int16Array(512);
const N = 20000;
let plies = 0;
let t0 = performance.now();
for (let g = 0; g < N; g++) {
  const s = newGame(g);
  while (s.phase !== Phase.GameOver) {
    const n = legalActions(s, buf);
    applyAction(s, buf[rng.int(n)], rng);
    plies++;
  }
}
let dt = (performance.now() - t0) / 1000;
console.log(`random games: ${(N / dt).toFixed(0)}/s, plies/s ${(plies / dt / 1e6).toFixed(2)}M, avg plies ${(plies / N).toFixed(1)}`);

const s = newGame(5);
t0 = performance.now();
const C = 1_000_000;
for (let i = 0; i < C; i++) cloneState(s);
dt = (performance.now() - t0) / 1000;
console.log(`cloneState: ${(C / dt / 1e6).toFixed(2)}M/s`);
