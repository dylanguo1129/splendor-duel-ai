import { describe, expect, it } from "vitest";
import { NUM_CARDS } from "../engine/cards";
import { EMPTY, NUM_TOKEN_TYPES, Phase, TOTAL_PRIVILEGES } from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, legalActions } from "../engine/rules";
import { GameState, newGame } from "../engine/state";

function checkInvariants(s: GameState): void {
  const total = new Array(NUM_TOKEN_TYPES).fill(0);
  for (const t of s.board) if (t !== EMPTY) total[t]++;
  for (let t = 0; t < NUM_TOKEN_TYPES; t++) {
    total[t] += s.bag[t] + s.players[0].tokens[t] + s.players[1].tokens[t];
    expect(s.bag[t]).toBeGreaterThanOrEqual(0);
    expect(s.players[0].tokens[t]).toBeGreaterThanOrEqual(0);
    expect(s.players[1].tokens[t]).toBeGreaterThanOrEqual(0);
  }
  expect(total).toEqual([2, 4, 4, 4, 4, 4, 3]);
  expect(s.tablePrivileges + s.players[0].privileges + s.players[1].privileges).toBe(TOTAL_PRIVILEGES);
  let cards = s.decks[1].length + s.decks[2].length + s.decks[3].length;
  for (const id of s.pyramid) if (id !== EMPTY) cards++;
  for (const p of s.players) cards += p.reserved.length + p.numCards;
  expect(cards).toBe(NUM_CARDS);
  const owned = [...s.owner].filter((o) => o !== -1).length + (s.pendingCard >= 0 ? 1 : 0);
  expect(owned).toBe(s.players[0].numCards + s.players[1].numCards);
  for (const p of s.players) {
    expect(p.reserved.length).toBeLessThanOrEqual(3);
    let cp = 0;
    for (let c = 1; c <= 5; c++) cp += p.colorPoints[c];
    expect(cp).toBeLessThanOrEqual(p.points);
  }
}

describe("random playouts", () => {
  it("conserve tokens, cards and privileges and always terminate", () => {
    const rng = new Rng(123);
    const buf = new Int16Array(512);
    let maxPly = 0;
    const outcomes = [0, 0, 0];
    for (let g = 0; g < 3000; g++) {
      const s = newGame(1000 + g);
      while (s.phase !== Phase.GameOver) {
        const n = legalActions(s, buf);
        expect(n).toBeGreaterThan(0);
        applyAction(s, buf[rng.int(n)], rng);
        const ph = s.phase as Phase;
        if (ph !== Phase.Discard && ph !== Phase.GameOver && g % 10 === 0) checkInvariants(s);
      }
      checkInvariants(s);
      if (s.phase === Phase.GameOver && s.players[s.current].tokens.reduce((a, b) => a + b, 0) > 10)
        throw new Error("game ended with >10 tokens for current player");
      outcomes[s.winner]++;
      maxPly = Math.max(maxPly, s.ply);
    }
    expect(outcomes[0] + outcomes[1]).toBeGreaterThan(0);
    console.log("outcomes", outcomes, "maxPly", maxPly);
  });
});
