import { describe, expect, it } from "vitest";
import { LINES, SPIRAL, lineIndexOf, rowOf, colOf, cellOf } from "../engine/board";
import { CARDS, ROYALS, cardId, LEVEL_CARD_IDS } from "../engine/cards";
import { BLUE, PEARL, WHITE, RED, BLACK } from "../engine/constants";
import { newGame, tokenCount } from "../engine/state";

describe("card data", () => {
  it("has 67 jewel cards split 30/24/13 and 4 royals", () => {
    expect(CARDS.length).toBe(67);
    expect(LEVEL_CARD_IDS.slice(1).map((l) => l.length)).toEqual([30, 24, 13]);
    expect(ROYALS.map((r) => [r.points, r.steal, r.again, r.privilege])).toEqual([
      [2, true, false, false],
      [2, false, true, false],
      [2, false, false, true],
      [3, false, false, false],
    ]);
  });

  it("spot checks", () => {
    const l3 = CARDS[cardId(3, 13)];
    expect([l3.points, l3.bonus, l3.joker, l3.cost[WHITE], l3.costTotal]).toEqual([6, 0, false, 8, 8]);
    const l2 = CARDS[cardId(2, 24)];
    expect([l2.points, l2.bonus, l2.cost[PEARL], l2.cost[BLUE]]).toEqual([5, 0, 1, 6]);
    const jokerAgain = CARDS[cardId(3, 11)];
    expect([jokerAgain.joker, jokerAgain.again, jokerAgain.points, jokerAgain.cost[RED]]).toEqual([true, true, 3, 8]);
    const twoBonus = CARDS[cardId(2, 16)];
    expect([twoBonus.color, twoBonus.bonus, twoBonus.cost[BLUE], twoBonus.cost[WHITE]]).toEqual([BLACK, 2, 2, 5]);
  });

  it("every gem color has 5 L1, 4 L2 and 2 L3 colored cards", () => {
    for (const color of [1, 2, 3, 4, 5]) {
      const per = [1, 2, 3].map((lv) => CARDS.filter((c) => c.level === lv && c.color === color).length);
      expect(per).toEqual([5, 4, 2]);
    }
  });
});

describe("board geometry", () => {
  it("has 145 straight contiguous lines", () => {
    expect(LINES.length).toBe(145);
    expect(LINES.filter((l) => l.length === 1).length).toBe(25);
    expect(LINES.filter((l) => l.length === 2).length).toBe(72);
    expect(LINES.filter((l) => l.length === 3).length).toBe(48);
    for (const l of LINES) {
      for (let k = 1; k < l.length; k++) {
        const dr = rowOf(l[k]) - rowOf(l[k - 1]);
        const dc = colOf(l[k]) - colOf(l[k - 1]);
        expect(Math.max(Math.abs(dr), Math.abs(dc))).toBe(1);
        if (k > 1) {
          expect(dr).toBe(rowOf(l[1]) - rowOf(l[0]));
          expect(dc).toBe(colOf(l[1]) - colOf(l[0]));
        }
      }
    }
    expect(lineIndexOf([cellOf(1, 3), cellOf(1, 1), cellOf(1, 2)])).toBeGreaterThan(0);
    expect(lineIndexOf([cellOf(1, 1), cellOf(1, 3)])).toBe(-1);
  });

  it("spiral starts at center and covers all cells", () => {
    expect(SPIRAL[0]).toBe(cellOf(3, 3));
    expect(SPIRAL[24]).toBe(cellOf(5, 5));
    expect(new Set(SPIRAL).size).toBe(25);
  });
});

describe("setup", () => {
  it("deals board, pyramid, decks and the starting privilege", () => {
    const s = newGame(42, 0);
    const counts = new Array(7).fill(0);
    for (const t of s.board) counts[t]++;
    expect(counts).toEqual([2, 4, 4, 4, 4, 4, 3]);
    expect([...s.bag]).toEqual([0, 0, 0, 0, 0, 0, 0]);
    expect([...s.pyramid].every((c) => c >= 0)).toBe(true);
    expect(s.decks.slice(1).map((d) => d.length)).toEqual([25, 20, 10]);
    expect([s.players[0].privileges, s.players[1].privileges, s.tablePrivileges]).toEqual([0, 1, 2]);
    expect(tokenCount(s.players[0])).toBe(0);
    expect(s.royalsAvail.length).toBe(4);
  });
});
