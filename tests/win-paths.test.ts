import { describe, expect, it } from "vitest";
import { HeuristicEvaluator } from "../ai/agents";
import { shapeValue, StyledEvaluator, stylePriorBoost, styleProgress } from "../ai/styles";
import { CARDS } from "../engine/cards";
import { A_BUY, GOLD, Phase } from "../engine/constants";
import { BASE_FEATURES, encode, NUM_FEATURES } from "../engine/features";
import { Rng } from "../engine/rng";
import { applyAction, legalActions, listActions } from "../engine/rules";
import { cloneState, GameState, newGame } from "../engine/state";

const BLOCK = 22;
const feats = (s: GameState) => {
  const x = new Float32Array(NUM_FEATURES);
  encode(s, x);
  return x;
};
const meBlock = (x: Float32Array) => x.slice(BASE_FEATURES, BASE_FEATURES + BLOCK);
const oppBlock = (x: Float32Array) => x.slice(BASE_FEATURES + BLOCK, BASE_FEATURES + 2 * BLOCK);
// block layout (engine/features.ts encodeWinPaths)
const WIN_NOW_CROWNS = 15;
const WIN_NOW_COLOR = 16;
const WIN_NOW_ANY = 17;

/** Make `pi` able to pay for card `id` with exactly its cost in tokens. */
function affordWithTokens(s: GameState, pi: number, id: number): void {
  const p = s.players[pi];
  p.tokens.fill(0);
  for (let t = 0; t < 6; t++) p.tokens[t] = CARDS[id].cost[t];
}

describe("win-path features", () => {
  it("are finite and within [0, 1] along random games", () => {
    const rng = new Rng(3);
    const buf = new Int16Array(512);
    for (let g = 0; g < 20; g++) {
      const s = newGame(100 + g);
      for (let ply = 0; ply < 400 && s.phase !== Phase.GameOver; ply++) {
        const x = feats(s);
        for (let i = BASE_FEATURES; i < NUM_FEATURES; i++) {
          expect(Number.isFinite(x[i])).toBe(true);
          expect(x[i]).toBeGreaterThanOrEqual(0);
          expect(x[i]).toBeLessThanOrEqual(1);
        }
        const n = legalActions(s, buf);
        applyAction(s, buf[Math.floor(rng.next() * n)], rng);
      }
    }
  });

  it("keeps the base layout unchanged (older models read only those inputs)", () => {
    expect(NUM_FEATURES).toBe(BASE_FEATURES + 2 * BLOCK);
    expect(BASE_FEATURES).toBe(820);
  });

  it("flags a crown win available now", () => {
    const s = newGame(7);
    const slot = [...s.pyramid].findIndex((id) => id >= 0 && CARDS[id].crowns > 0 && !CARDS[id].joker);
    expect(slot).toBeGreaterThanOrEqual(0);
    const me = s.current;
    s.players[me].crowns = 10 - CARDS[s.pyramid[slot]].crowns;
    expect(meBlock(feats(s))[WIN_NOW_CROWNS]).toBe(0); // cannot pay yet
    affordWithTokens(s, me, s.pyramid[slot]);
    const b = meBlock(feats(s));
    expect(b[WIN_NOW_CROWNS]).toBe(1);
    expect(b[WIN_NOW_ANY]).toBe(1);
    expect(oppBlock(feats(s))[WIN_NOW_ANY]).toBe(0);
  });

  it("flags a one-colour win available now", () => {
    const s = newGame(9);
    const slot = [...s.pyramid].findIndex((id) => id >= 0 && CARDS[id].points >= 1 && CARDS[id].color >= 1 && CARDS[id].color <= 5);
    expect(slot).toBeGreaterThanOrEqual(0);
    const id = s.pyramid[slot];
    const me = s.current;
    s.players[me].colorPoints[CARDS[id].color] = 10 - CARDS[id].points;
    affordWithTokens(s, me, id);
    expect(meBlock(feats(s))[WIN_NOW_COLOR]).toBe(1);
  });

  it("ignores the opponent's blind reservations (hidden information)", () => {
    const s = newGame(11);
    const opp = 1 - s.current;
    const crown = CARDS.find((c) => c.crowns > 0 && !c.joker && ![...s.pyramid].includes(c.id))!;
    s.players[opp].crowns = 10 - crown.crowns;
    affordWithTokens(s, opp, crown.id);
    const blind = cloneState(s);
    blind.players[opp].reserved.push(crown.id);
    blind.players[opp].reservedHidden.push(true);
    const open = cloneState(s);
    open.players[opp].reserved.push(crown.id);
    open.players[opp].reservedHidden.push(false);
    expect(oppBlock(feats(blind))).toEqual(oppBlock(feats(s)));
    expect(oppBlock(feats(open))[WIN_NOW_CROWNS]).toBe(1);
  });

  it("handles a player without gold", () => {
    const s = newGame(13);
    s.players[s.current].tokens[GOLD] = 0;
    expect(() => feats(s)).not.toThrow();
  });
});

describe("rush styles", () => {
  it("boost buying cards on the style's path", () => {
    const s = newGame(7);
    const crownSlot = [...s.pyramid].findIndex((id) => id >= 0 && CARDS[id].crowns > 0);
    const plainSlot = [...s.pyramid].findIndex((id) => id >= 0 && CARDS[id].crowns === 0 && CARDS[id].points === 0);
    expect(stylePriorBoost(s, A_BUY + crownSlot, "crowns")).toBeGreaterThan(1);
    expect(stylePriorBoost(s, A_BUY + plainSlot, "crowns")).toBe(1);
    expect(stylePriorBoost(s, 0, "points")).toBe(1); // taking tokens
  });

  it("shape values toward the styled player's progress", () => {
    const s = newGame(7);
    s.players[0].crowns = 5;
    expect(styleProgress(s, 0, "crowns")).toBeCloseTo(0.5);
    expect(shapeValue(0, s, 0, 0, "crowns")).toBeGreaterThan(0);
    expect(shapeValue(0, s, 1, 0, "crowns")).toBeLessThan(0);
    expect(shapeValue(0.95, s, 0, 0, "crowns")).toBeLessThanOrEqual(1);
  });

  it("only bias the styled player's priors", () => {
    const s = newGame(21);
    const legal = Int16Array.from(listActions(s));
    const plain = new Float32Array(512);
    const styled = new Float32Array(512);
    const other = new Float32Array(512);
    new HeuristicEvaluator().evaluate(s, legal, legal.length, plain, new Rng(1));
    new StyledEvaluator(new HeuristicEvaluator(), "crowns", s.current).evaluate(s, legal, legal.length, styled, new Rng(1));
    new StyledEvaluator(new HeuristicEvaluator(), "crowns", 1 - s.current).evaluate(s, legal, legal.length, other, new Rng(1));
    expect([...other.slice(0, legal.length)]).toEqual([...plain.slice(0, legal.length)]);
    expect(styled.slice(0, legal.length).some((v, i) => v > plain[i])).toBe(true);
  });
});
