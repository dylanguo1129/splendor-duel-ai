import { describe, expect, it } from "vitest";
import { LINES, lineIndexOf } from "../engine/board";
import { CARDS, cardId } from "../engine/cards";
import {
  A_BUY,
  A_CELL,
  A_DISCARD,
  A_JOKER,
  A_PASS,
  A_REPLENISH,
  A_RESERVE,
  A_ROYAL,
  A_STEAL,
  A_TAKE_LINE,
  BLACK,
  BLUE,
  EMPTY,
  GOLD,
  GREEN,
  PEARL,
  Phase,
  RED,
  WHITE,
} from "../engine/constants";
import { Rng } from "../engine/rng";
import { applyAction, listActions, WIN_REASON_ANTIPLAY, WIN_REASON_COLOR, WIN_REASON_CROWNS, WIN_REASON_POINTS } from "../engine/rules";
import { tokenCount } from "../engine/state";
import { cell, has, position, putPyramid } from "./helpers";

const rng = () => new Rng(7);
const take = (...rc: [number, number][]) => A_TAKE_LINE + lineIndexOf(rc.map(([r, c]) => cell(r, c)));

describe("taking tokens", () => {
  const rows = ["UUW$.", "PG...", "P....", ".....", "....."];

  it("allows straight uninterrupted lines of non-gold tokens only", () => {
    const s = position(rows);
    expect(has(s, take([1, 1], [1, 2], [1, 3]))).toBe(true);
    expect(has(s, take([1, 2], [1, 3], [1, 4]))).toBe(false); // gold interrupts
    expect(has(s, take([1, 4], [1, 5]))).toBe(false); // gold + empty
    expect(has(s, take([1, 1], [2, 2]))).toBe(true); // diagonal
    expect(has(s, take([1, 4]))).toBe(false); // gold alone
    expect(LINES.length).toBe(145);
  });

  it("2 pearls give the opponent a privilege", () => {
    const s = position(rows);
    applyAction(s, take([1, 1], [2, 1], [3, 1]), rng());
    expect([...s.players[0].tokens]).toEqual([2, 1, 0, 0, 0, 0, 0]);
    expect([s.players[1].privileges, s.tablePrivileges]).toEqual([1, 2]);
    expect(s.current).toBe(1);
  });

  it("3 of the same color give a privilege, taken from the taker when the table is empty", () => {
    const s = position(["RRR..", ".....", ".....", ".....", "....."]);
    s.tablePrivileges = 0;
    s.players[0].privileges = 2;
    s.players[1].privileges = 1;
    applyAction(s, take([1, 1], [1, 2], [1, 3]), rng());
    expect([s.players[0].privileges, s.players[1].privileges]).toEqual([1, 2]);
  });

  it("2 different + 1 same does not give a privilege", () => {
    const s = position(["RRU..", ".....", ".....", ".....", "....."]);
    applyAction(s, take([1, 1], [1, 2], [1, 3]), rng());
    expect(s.players[1].privileges).toBe(0);
  });
});

describe("optional actions", () => {
  it("privilege picks: non-gold only, canonical increasing order, before replenish", () => {
    const s = position(["...U.", ".W..$", ".....", ".....", "....."]);
    s.players[0].privileges = 2;
    s.tablePrivileges = 1;
    s.bag[RED] = 2;
    expect(has(s, A_CELL + cell(1, 4))).toBe(true);
    expect(has(s, A_CELL + cell(2, 2))).toBe(true);
    expect(has(s, A_CELL + cell(2, 5))).toBe(false);
    applyAction(s, A_CELL + cell(2, 2), rng());
    expect(s.players[0].tokens[WHITE]).toBe(1);
    expect([s.players[0].privileges, s.tablePrivileges]).toEqual([1, 2]);
    expect(has(s, A_CELL + cell(1, 4))).toBe(false); // lower cell index after higher pick
    applyAction(s, A_REPLENISH, rng());
    expect(listActions(s).some((a) => a >= A_CELL && a < A_CELL + 25)).toBe(false);
    expect(has(s, A_REPLENISH)).toBe(false);
  });

  it("replenish fills empty cells in spiral order and gives the opponent a privilege", () => {
    const s = position([".....", ".....", "..U..", ".....", "....."]);
    s.bag[RED] = 2;
    applyAction(s, A_REPLENISH, rng());
    expect(s.board[cell(4, 3)]).toBe(RED);
    expect(s.board[cell(4, 2)]).toBe(RED);
    expect(s.board[cell(3, 2)]).toBe(EMPTY);
    expect([...s.bag].every((x) => x === 0)).toBe(true);
    expect(s.players[1].privileges).toBe(1);
    expect(s.current).toBe(0);
  });

  it("forces replenish when no mandatory action exists, else passes", () => {
    const s = position(["$....", ".....", ".....", ".....", "....."]);
    s.players[0].reserved = [0, 1, 2];
    s.players[0].reservedHidden = [false, false, false];
    s.bag[BLUE] = 1;
    expect(listActions(s)).toEqual([A_REPLENISH]);
    s.bag[BLUE] = 0;
    expect(listActions(s)).toEqual([A_PASS]);
  });
});

describe("reserving", () => {
  it("takes a gold, reserves a pyramid card and refills the slot", () => {
    const s = position(["$....", ".....", ".....", ".....", "....."]);
    const id = putPyramid(s, 2, 0, 5);
    s.decks[2] = [cardId(2, 7), cardId(2, 9)];
    applyAction(s, A_RESERVE + 5, rng());
    expect(s.phase).toBe(Phase.ReserveGold);
    expect(listActions(s)).toEqual([A_CELL + cell(1, 1)]);
    applyAction(s, A_CELL + cell(1, 1), rng());
    expect(s.players[0].reserved).toEqual([id]);
    expect(s.players[0].reservedHidden).toEqual([false]);
    expect(s.players[0].tokens[GOLD]).toBe(1);
    expect(s.pyramid[5]).toBe(cardId(2, 9));
    expect(s.current).toBe(1);
  });

  it("reserves blind from a deck (hidden)", () => {
    const s = position(["$....", ".....", ".....", ".....", "....."]);
    s.decks[3] = [cardId(3, 1)];
    applyAction(s, A_RESERVE + 14, rng());
    applyAction(s, A_CELL + cell(1, 1), rng());
    expect(s.players[0].reserved).toEqual([cardId(3, 1)]);
    expect(s.players[0].reservedHidden).toEqual([true]);
    expect(s.decks[3].length).toBe(0);
  });

  it("needs a gold on the board and fewer than 3 reserved cards", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 1);
    expect(listActions(s).some((a) => a >= A_RESERVE && a < A_RESERVE + 15)).toBe(false);
    s.board[cell(5, 5)] = GOLD;
    expect(has(s, A_RESERVE + 0)).toBe(true);
    s.players[0].reserved = [10, 11, 12];
    s.players[0].reservedHidden = [false, false, false];
    expect(has(s, A_RESERVE + 0)).toBe(false);
  });
});

describe("purchasing", () => {
  it("applies bonuses and uses gold for the shortfall", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    const id = putPyramid(s, 3, 0, 2); // 4 pts white: 2U 6W 2K
    const p = s.players[0];
    p.bonus[WHITE] = 2;
    p.tokens.set([0, 2, 3, 0, 2, 0, 1]);
    expect(has(s, A_BUY + 9)).toBe(true);
    applyAction(s, A_BUY + 9, rng());
    expect([...p.tokens]).toEqual([0, 0, 0, 0, 0, 0, 0]);
    expect([...s.bag]).toEqual([0, 2, 3, 0, 2, 0, 1]);
    expect([p.points, p.bonus[WHITE], p.colorPoints[WHITE], p.numCards]).toEqual([4, 3, 4, 1]);
    expect(s.owner[id]).toBe(WHITE);
  });

  it("cannot buy when short even with gold", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 3, 0, 2);
    s.players[0].tokens.set([0, 2, 3, 0, 2, 0, 0]);
    expect(has(s, A_BUY + 9)).toBe(false);
  });

  it("joker needs a bonus card and joins the chosen column", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    const id = putPyramid(s, 1, 0, 26); // joker 1 pt: 1P 4K
    const p = s.players[0];
    p.tokens.set([1, 0, 0, 0, 4, 0, 0]);
    expect(has(s, A_BUY + 0)).toBe(false);
    p.bonus[GREEN] = 1;
    p.bonus[RED] = 1;
    applyAction(s, A_BUY + 0, rng());
    expect(s.phase).toBe(Phase.Joker);
    expect(listActions(s)).toEqual([A_JOKER + GREEN - 1, A_JOKER + RED - 1]);
    applyAction(s, A_JOKER + GREEN - 1, rng());
    expect([p.bonus[GREEN], p.colorPoints[GREEN], p.points]).toEqual([2, 1, 1]);
    expect(s.owner[id]).toBe(GREEN);
    expect(s.current).toBe(1);
  });

  it("buys reserved cards", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    const p = s.players[0];
    p.reserved = [cardId(1, 2), cardId(1, 7)]; // 3U crown white ; 3G crown blue
    p.reservedHidden = [false, true];
    p.tokens[GREEN] = 3;
    expect(has(s, A_BUY + 12)).toBe(false);
    expect(has(s, A_BUY + 13)).toBe(true);
    applyAction(s, A_BUY + 13, rng());
    expect(p.reserved).toEqual([cardId(1, 2)]);
    expect(p.reservedHidden).toEqual([false]);
    expect([p.crowns, p.bonus[BLUE]]).toEqual([1, 1]);
  });
});

describe("card abilities", () => {
  it("token ability takes a token of the card color, skipped when none", () => {
    const s = position(["W....", ".....", "....W", ".....", "....."]);
    putPyramid(s, 1, 0, 4); // white, token ability: 2K 2R
    s.players[0].tokens.set([0, 0, 0, 0, 2, 2, 0]);
    applyAction(s, A_BUY + 0, rng());
    expect(s.phase).toBe(Phase.TakeToken);
    expect(listActions(s)).toEqual([A_CELL + cell(1, 1), A_CELL + cell(3, 5)]);
    applyAction(s, A_CELL + cell(3, 5), rng());
    expect(s.players[0].tokens[WHITE]).toBe(1);
    expect(s.current).toBe(1);

    const t = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(t, 1, 0, 4);
    t.players[0].tokens.set([0, 0, 0, 0, 2, 2, 0]);
    applyAction(t, A_BUY + 0, rng());
    expect(t.current).toBe(1);
  });

  it("steal takes a non-gold token, skipped when opponent has only gold", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 2, 0, 2); // white steal: 4U 3R
    s.players[0].tokens.set([0, 4, 0, 0, 0, 3, 0]);
    s.players[1].tokens.set([0, 0, 0, 0, 0, 0, 2]);
    applyAction(s, A_BUY + 5, rng());
    expect(s.current).toBe(1);

    const t = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(t, 2, 0, 2);
    t.players[0].tokens.set([0, 4, 0, 0, 0, 3, 0]);
    t.players[1].tokens.set([1, 0, 0, 0, 2, 0, 2]);
    applyAction(t, A_BUY + 5, rng());
    expect(t.phase).toBe(Phase.Steal);
    expect(listActions(t)).toEqual([A_STEAL + PEARL, A_STEAL + BLACK]);
    applyAction(t, A_STEAL + PEARL, rng());
    expect([t.players[0].tokens[PEARL], t.players[1].tokens[PEARL]]).toEqual([1, 0]);
  });

  it("privilege ability takes from the opponent when the table is empty", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 2, 0, 3); // white priv: 1P 4W 2K
    s.players[0].tokens.set([1, 0, 4, 0, 2, 0, 0]);
    s.tablePrivileges = 0;
    s.players[1].privileges = 3;
    applyAction(s, A_BUY + 5, rng());
    expect([s.players[0].privileges, s.players[1].privileges]).toEqual([1, 2]);
  });

  it("extra turn keeps the same player", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 3); // white again: 1P 2U 2G
    s.players[0].tokens.set([1, 2, 0, 2, 0, 0, 0]);
    applyAction(s, A_BUY + 0, rng());
    expect(s.current).toBe(0);
    expect(s.phase).toBe(Phase.Main);
  });
});

describe("crowns and royals", () => {
  it("3rd crown grants a royal; royal extra turn applies", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 2); // white 1 crown: 3U
    const p = s.players[0];
    p.crowns = 2;
    p.tokens[BLUE] = 3;
    applyAction(s, A_BUY + 0, rng());
    expect(s.phase).toBe(Phase.Royal);
    expect(listActions(s)).toEqual([A_ROYAL + 0, A_ROYAL + 1, A_ROYAL + 2, A_ROYAL + 3]);
    applyAction(s, A_ROYAL + 1, rng());
    expect([p.points, p.royals.length, s.current, s.royalsAvail.length]).toEqual([2, 1, 0, 3]);
  });

  it("royal steal leads to a steal choice; 6th crown grants a second royal", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 3, 0, 12); // joker 3 crowns: 8K
    const p = s.players[0];
    p.crowns = 3;
    p.royalsEarned = 1;
    p.bonus[RED] = 1;
    p.tokens[BLACK] = 8;
    s.players[1].tokens[RED] = 1;
    applyAction(s, A_BUY + 9, rng());
    expect(s.phase).toBe(Phase.Joker);
    applyAction(s, A_JOKER + RED - 1, rng());
    expect(s.phase).toBe(Phase.Royal);
    applyAction(s, A_ROYAL + 0, rng());
    expect(s.phase).toBe(Phase.Steal);
    applyAction(s, A_STEAL + RED, rng());
    expect([p.crowns, p.tokens[RED], p.royalsEarned, s.current]).toEqual([6, 1, 2, 1]);
  });
});

describe("end of turn", () => {
  it("discards down to 10 in canonical order", () => {
    const s = position(["RUW..", ".....", ".....", ".....", "....."]);
    s.players[0].tokens.set([1, 2, 2, 2, 2, 0, 0]); // 9
    applyAction(s, take([1, 1], [1, 2], [1, 3]), rng()); // 12
    expect(s.phase).toBe(Phase.Discard);
    // need 2 discards: red (only 1 red, no gold above it) cannot come first
    expect(listActions(s)).toEqual([0, 1, 2, 3, 4].map((t) => A_DISCARD + t));
    applyAction(s, A_DISCARD + BLUE, rng());
    expect(has(s, A_DISCARD + PEARL)).toBe(false);
    applyAction(s, A_DISCARD + RED, rng());
    expect(tokenCount(s.players[0])).toBe(10);
    expect([s.bag[BLUE], s.bag[RED]]).toEqual([1, 1]);
    expect(s.current).toBe(1);
  });

  it("wins with 20 points", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 5); // 1 pt white: 2G 3R
    s.players[0].points = 19;
    s.players[0].tokens.set([0, 0, 0, 2, 0, 3, 0]);
    applyAction(s, A_BUY + 0, rng());
    expect([s.phase, s.winner, s.winReason]).toEqual([Phase.GameOver, 0, WIN_REASON_POINTS]);
  });

  it("wins with 10 crowns", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 2);
    s.players[0].crowns = 9;
    s.players[0].royalsEarned = 2;
    s.players[0].tokens[BLUE] = 3;
    applyAction(s, A_BUY + 0, rng());
    expect([s.winner, s.winReason]).toEqual([0, WIN_REASON_CROWNS]);
  });

  it("wins with 10 points in one color, counting a joker in that column", () => {
    const s = position(["U....", ".....", ".....", ".....", "....."]);
    putPyramid(s, 1, 0, 26); // joker 1 pt
    const p = s.players[0];
    p.bonus[GREEN] = 3;
    p.colorPoints[GREEN] = 9;
    p.points = 9;
    p.tokens.set([1, 0, 0, 0, 4, 0, 0]);
    applyAction(s, A_BUY + 0, rng());
    applyAction(s, A_JOKER + GREEN - 1, rng());
    expect([s.winner, s.winReason]).toEqual([0, WIN_REASON_COLOR]);
  });

  it("BGA anti-playing: hoarding all gold and pearls 3 turns without buying loses", () => {
    const s = position(["UUUUU", "WWWWW", ".....", ".....", "....."]);
    for (const pl of s.players) pl.tokens.set([0, 0, 0, 0, 0, 0, 0]);
    s.players[0].tokens.set([2, 0, 0, 0, 0, 0, 3]);
    for (let k = 0; k < 3; k++) {
      expect(s.winner).toBe(-1);
      applyAction(s, A_TAKE_LINE + cell(1, 1 + k), rng()); // p0 single token
      if (k < 2) applyAction(s, A_TAKE_LINE + cell(2, 1 + k), rng()); // p1 single token
    }
    expect([s.winner, s.winReason]).toEqual([1, WIN_REASON_ANTIPLAY]);
  });
});

describe("misc", () => {
  it("card colors in data agree with bonuses", () => {
    for (const c of CARDS) {
      if (c.color !== 9) expect(c.bonus).toBeGreaterThan(0);
      if (c.joker) expect(c.bonus).toBe(1);
    }
    expect(WHITE).toBe(2);
  });
});
