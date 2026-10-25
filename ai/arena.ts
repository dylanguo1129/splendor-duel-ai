import { DRAW, Phase } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, DEFAULT_RULES } from "../engine/rules";
import { newGame } from "../engine/state";
import type { Agent } from "./agents";

export interface MatchResult {
  wins: number; // for agent A
  losses: number;
  draws: number;
  games: number;
  score: number; // (wins + draws/2) / games
  elo: number; // A minus B
  eloCI95: number;
  avgPlies: number;
  reasons: Record<number, number>;
}

export function playGame(a0: Agent, a1: Agent, seed: number): { winner: number; plies: number; reason: number } {
  const s = newGame(seed, 0);
  const rng = new Rng(seed ^ 0x5bd1e995);
  const agents = [a0, a1];
  while (s.phase !== Phase.GameOver) {
    const a = agents[s.current].choose(s);
    applyAction(s, a, rng, DEFAULT_RULES);
  }
  return { winner: s.winner, plies: s.ply, reason: s.winReason };
}

export function eloFromScore(p: number): number {
  const q = Math.min(Math.max(p, 1e-4), 1 - 1e-4);
  return -400 * Math.log10(1 / q - 1);
}

/** Plays `games` games, alternating who moves first; makeA/makeB build fresh agents per game. */
export function match(makeA: () => Agent, makeB: () => Agent, games: number, seed0 = 1000, onGame?: (i: number, r: MatchResult) => void): MatchResult {
  const r: MatchResult = { wins: 0, losses: 0, draws: 0, games: 0, score: 0, elo: 0, eloCI95: 0, avgPlies: 0, reasons: {} };
  let plies = 0;
  for (let g = 0; g < games; g++) {
    const aFirst = g % 2 === 0;
    const A = makeA();
    const B = makeB();
    const res = aFirst ? playGame(A, B, seed0 + (g >> 1)) : playGame(B, A, seed0 + (g >> 1));
    const aIdx = aFirst ? 0 : 1;
    if (res.winner === DRAW) r.draws++;
    else if (res.winner === aIdx) r.wins++;
    else r.losses++;
    r.reasons[res.reason] = (r.reasons[res.reason] ?? 0) + 1;
    plies += res.plies;
    r.games++;
    finalize(r, plies);
    onGame?.(g, r);
  }
  return r;
}

function finalize(r: MatchResult, plies: number): void {
  r.score = (r.wins + r.draws / 2) / r.games;
  r.elo = eloFromScore(r.score);
  const se = Math.sqrt((r.score * (1 - r.score)) / r.games);
  r.eloCI95 = (eloFromScore(Math.min(0.9999, r.score + 1.96 * se)) - eloFromScore(Math.max(0.0001, r.score - 1.96 * se))) / 2;
  r.avgPlies = plies / r.games;
}
