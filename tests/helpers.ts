import { cellOf } from "../engine/board";
import { cardId } from "../engine/cards";
import { EMPTY, LEVEL_SLOT_START } from "../engine/constants";
import { listActions } from "../engine/rules";
import { emptyState, GameState } from "../engine/state";

/** Board from 5 strings of 5 chars: . empty, P pearl, U blue, W white, G green, K black, R red, $ gold. */
export function boardFrom(rows: string[]): Int8Array {
  const map: Record<string, number> = { ".": EMPTY, P: 0, U: 1, W: 2, G: 3, K: 4, R: 5, $: 6 };
  const b = new Int8Array(25).fill(EMPTY);
  rows.forEach((row, r) => [...row].forEach((ch, c) => (b[cellOf(r + 1, c + 1)] = map[ch])));
  return b;
}

/** Hand-built position: no conservation guarantees, pyramid/decks empty unless set. */
export function position(rows: string[] = [".....", ".....", ".....", ".....", "....."]): GameState {
  const s = emptyState();
  s.board = boardFrom(rows);
  s.tablePrivileges = 3;
  return s;
}

export function putPyramid(s: GameState, level: number, k: number, index: number): number {
  const id = cardId(level, index);
  s.pyramid[LEVEL_SLOT_START[level] + k] = id;
  return id;
}

export function has(s: GameState, a: number): boolean {
  return listActions(s).includes(a);
}

export const cell = cellOf;
