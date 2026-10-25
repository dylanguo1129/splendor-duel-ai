import raw from "../data/cards.json";
import { COLOR_JOKER } from "./constants";

export interface Card {
  id: number; // 0..66, ordered L1 index 1..30, L2 1..24, L3 1..13
  level: 1 | 2 | 3;
  index: number; // BGA index within level
  color: number; // 1..5, or 9 for joker / colorless
  points: number;
  crowns: number;
  bonus: number; // bonus count granted (0, 1 or 2); joker grants 1 of the chosen color
  joker: boolean;
  again: boolean;
  token: boolean;
  privilege: boolean;
  steal: boolean;
  cost: Int8Array; // [pearl, blue, white, green, black, red]
  costTotal: number;
}

export interface Royal {
  id: number; // 0..3 (BGA index 1..4)
  index: number;
  points: number;
  again: boolean;
  privilege: boolean;
  steal: boolean;
}

interface RawCard {
  level: number;
  index: number;
  color: number;
  points: number;
  crowns: number;
  power: number[];
  cost: Record<string, number>;
  provides: Record<string, number>;
}

const rawCards = (raw as unknown as { cards: RawCard[] }).cards
  .slice()
  .sort((a, b) => a.level - b.level || a.index - b.index);

export const CARDS: Card[] = rawCards.map((c, id) => {
  const cost = new Int8Array(6);
  for (const [k, v] of Object.entries(c.cost)) cost[Number(k)] = v;
  const bonus = Object.values(c.provides ?? {}).reduce((a, b) => a + b, 0);
  return {
    id,
    level: c.level as 1 | 2 | 3,
    index: c.index,
    color: c.color,
    points: c.points,
    crowns: c.crowns,
    bonus,
    joker: c.power.includes(2),
    again: c.power.includes(1),
    token: c.power.includes(3),
    privilege: c.power.includes(4),
    steal: c.power.includes(5),
    cost,
    costTotal: cost.reduce((a, b) => a + b, 0),
  };
});

export const ROYALS: Royal[] = (raw as unknown as { royals: { index: number; points: number; power: number[] }[] }).royals
  .slice()
  .sort((a, b) => a.index - b.index)
  .map((r, id) => ({
    id,
    index: r.index,
    points: r.points,
    again: r.power.includes(1),
    privilege: r.power.includes(4),
    steal: r.power.includes(5),
  }));

export const NUM_CARDS = CARDS.length;
export const LEVEL_CARD_IDS: number[][] = [[], [], [], []];
for (const c of CARDS) LEVEL_CARD_IDS[c.level].push(c.id);

const byKey = new Map<string, number>();
for (const c of CARDS) byKey.set(`${c.level}:${c.index}`, c.id);

export function cardId(level: number, index: number): number {
  const id = byKey.get(`${level}:${index}`);
  if (id === undefined) throw new Error(`unknown card L${level}#${index}`);
  return id;
}

export function isColorless(c: Card): boolean {
  return c.color === COLOR_JOKER && !c.joker;
}
