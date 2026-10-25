// Strength ladder of the current best network vs fixed baselines (equal simulations).
// Build with scripts/build-train.mjs, run: node dist/train/report.mjs --workers 4 --games 200
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { matchPool } from "./pool";
import type { PlayerSpec } from "./worker";

const args = process.argv.slice(2);
const opt = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const WORKERS = opt("workers", 4);
const GAMES = opt("games", 200);
const SIMS = opt("sims", 200);
const net = (p: string): PlayerSpec => ({ kind: "net", net: p, sims: SIMS });
const heur: PlayerSpec = { kind: "heuristic", sims: SIMS };

const eloFor = (p: number) => -400 * Math.log10(1 / Math.min(Math.max(p, 1e-3), 1 - 1e-3) - 1);
const ci = (p: number, n: number) => 1.96 * Math.sqrt((p * (1 - p)) / n);

async function main() {
  const best = "models/best.json";
  if (!existsSync(best)) throw new Error("no models/best.json");
  const meta = JSON.parse(readFileSync(best, "utf8")).meta;
  const rows: string[] = [];
  const pairs: [string, PlayerSpec, PlayerSpec][] = [[`best vs heuristic MCTS (${SIMS} sims each)`, net(best), heur]];
  if (existsSync("models/gen0.json")) pairs.push([`best vs gen0 (${SIMS} sims each)`, net(best), net("models/gen0.json")]);
  for (const [name, a, b] of pairs) {
    const r = await matchPool({ workers: WORKERS, games: GAMES, seed: 4242, a, b });
    const line = `${name}: ${(r.score * 100).toFixed(1)}% ±${(ci(r.score, r.games) * 100).toFixed(1)} (Elo ${eloFor(r.score).toFixed(0)}) over ${r.games} games`;
    console.log(line);
    rows.push(line);
  }
  writeFileSync("train/runs/report.txt", [`model meta: ${JSON.stringify(meta)}`, ...rows].join("\n") + "\n");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
