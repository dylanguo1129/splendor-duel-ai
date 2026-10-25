// Observation encoding for the policy/value network, from the point of view of the player to move.
// Contains no hidden information: decks are only sizes, opponent's blind reserves only their level.
import { CARDS } from "./cards";
import { COLOR_JOKER, EMPTY, GOLD, Phase } from "./constants";
import { goldNeeded } from "./rules";
import { GameState, PlayerState, tokenCount } from "./state";

const CARD_BLOCK = 28;
const PLAYER_BLOCK = 30;
const WIN_PATH_BLOCK = 22;

/** Size of the original layout; models trained on it read only these inputs (see ai/net.ts). */
export const BASE_FEATURES =
  175 + // board one-hot
  7 + // bag
  1 + // table privileges
  2 * PLAYER_BLOCK +
  12 * CARD_BLOCK + // pyramid
  3 * CARD_BLOCK + // my reserved
  3 * (CARD_BLOCK + 1) + // opponent reserved (+ hidden flag)
  CARD_BLOCK + // pending joker card
  4 + // royals available
  3 + // deck sizes
  7 + // phase
  26 + // turn context
  2; // turn number, am-first-player

export const NUM_FEATURES = BASE_FEATURES + 2 * WIN_PATH_BLOCK; // + win paths: me, opponent

function encodePlayer(p: PlayerState, out: Float32Array, o: number): number {
  for (let t = 0; t < 7; t++) out[o++] = p.tokens[t] / 4;
  out[o++] = tokenCount(p) / 10;
  let maxCol = 0;
  for (let c = 1; c <= 5; c++) out[o++] = p.bonus[c] / 5;
  for (let c = 1; c <= 5; c++) {
    out[o++] = p.colorPoints[c] / 10;
    if (p.colorPoints[c] > maxCol) maxCol = p.colorPoints[c];
  }
  out[o++] = p.points / 20;
  out[o++] = p.crowns / 10;
  out[o++] = maxCol / 10;
  out[o++] = p.privileges / 3;
  out[o++] = p.royals.length / 2;
  out[o++] = p.royalsEarned / 2;
  out[o++] = p.numCards / 15;
  out[o++] = 0; // reserved for future use (kept for a stable layout)
  for (let r = 0; r < 4; r++) out[o++] = p.royals.includes(r) ? 1 : 0;
  return o;
}

function encodeCard(id: number, me: PlayerState, opp: PlayerState, out: Float32Array, o: number): number {
  const end = o + CARD_BLOCK;
  if (id < 0) {
    out.fill(0, o, end);
    return end;
  }
  const c = CARDS[id];
  out[o++] = 1;
  for (let col = 1; col <= 5; col++) out[o++] = c.color === col ? 1 : 0;
  out[o++] = c.joker ? 1 : 0;
  out[o++] = c.color === COLOR_JOKER && !c.joker ? 1 : 0;
  out[o++] = c.points / 6;
  out[o++] = c.crowns / 3;
  out[o++] = c.bonus / 2;
  for (let t = 0; t < 6; t++) out[o++] = c.cost[t] / 8;
  out[o++] = c.again ? 1 : 0;
  out[o++] = c.token ? 1 : 0;
  out[o++] = c.privilege ? 1 : 0;
  out[o++] = c.steal ? 1 : 0;
  const gm = goldNeeded(me, id);
  const go = goldNeeded(opp, id);
  const shortMe = gm < 0 ? 6 : Math.max(0, gm - me.tokens[GOLD]);
  const shortOpp = go < 0 ? 6 : Math.max(0, go - opp.tokens[GOLD]);
  out[o++] = Math.min(shortMe, 6) / 6;
  out[o++] = Math.min(shortOpp, 6) / 6;
  out[o++] = shortMe === 0 ? 1 : 0;
  out[o++] = shortOpp === 0 ? 1 : 0;
  for (let l = 1; l <= 3; l++) out[o++] = c.level === l ? 1 : 0;
  return end;
}

export function encode(s: GameState, out: Float32Array): void {
  const meI = s.current;
  const me = s.players[meI];
  const opp = s.players[1 - meI];
  out.fill(0);
  let o = 0;
  for (let cell = 0; cell < 25; cell++) {
    const t = s.board[cell];
    if (t !== EMPTY) out[o + t] = 1;
    o += 7;
  }
  for (let t = 0; t < 7; t++) out[o++] = s.bag[t] / 4;
  out[o++] = s.tablePrivileges / 3;
  o = encodePlayer(me, out, o);
  o = encodePlayer(opp, out, o);
  for (let slot = 0; slot < 12; slot++) o = encodeCard(s.pyramid[slot], me, opp, out, o);
  for (let r = 0; r < 3; r++) o = encodeCard(r < me.reserved.length ? me.reserved[r] : -1, me, opp, out, o);
  for (let r = 0; r < 3; r++) {
    if (r < opp.reserved.length && opp.reservedHidden[r]) {
      out.fill(0, o, o + CARD_BLOCK);
      out[o] = 1;
      out[o + CARD_BLOCK - 3 + CARDS[opp.reserved[r]].level - 1] = 1;
      o += CARD_BLOCK;
      out[o++] = 1;
    } else {
      o = encodeCard(r < opp.reserved.length ? opp.reserved[r] : -1, me, opp, out, o);
      out[o++] = 0;
    }
  }
  o = encodeCard(s.phase === Phase.Joker ? s.pendingCard : -1, me, opp, out, o);
  for (let r = 0; r < 4; r++) out[o++] = s.royalsAvail.includes(r) ? 1 : 0;
  out[o++] = s.decks[1].length / 25;
  out[o++] = s.decks[2].length / 20;
  out[o++] = s.decks[3].length / 10;
  for (let ph = 0; ph < 7; ph++) out[o++] = s.phase === ph ? 1 : 0;
  out[o++] = s.replenished ? 1 : 0;
  out[o++] = s.boughtThisTurn ? 1 : 0;
  out[o++] = s.extraTurn ? 1 : 0;
  for (let t = 0; t < 15; t++) out[o++] = s.reserveTarget === t ? 1 : 0;
  for (let col = 1; col <= 5; col++) out[o++] = s.takeColor === col ? 1 : 0;
  out[o++] = (s.lastPrivCell + 1) / 25;
  out[o++] = s.lastDiscard / 6;
  out[o++] = s.effects.length / 3;
  out[o++] = Math.min(s.turn, 80) / 80;
  out[o++] = s.firstPlayer === meI ? 1 : 0;
  o = encodeWinPaths(s, meI, meI, out, o);
  o = encodeWinPaths(s, 1 - meI, meI, out, o);
  if (o !== NUM_FEATURES) throw new Error(`feature layout mismatch ${o} != ${NUM_FEATURES}`);
}

const clip01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/**
 * Player `pi`'s race toward each way of winning (20 points, 10 crowns, 10 points in one colour):
 * how far away it is, and what a single purchase would do for it, now (affordable) and soon (at
 * most 2 tokens short). Cards considered: the pyramid and the player's reservations, except the
 * opponent's blind reservations, which `viewer` cannot see.
 */
function encodeWinPaths(s: GameState, pi: number, viewer: number, out: Float32Array, o: number): number {
  const p = s.players[pi];
  let bestCol = 1;
  for (let c = 2; c <= 5; c++) if (p.colorPoints[c] > p.colorPoints[bestCol]) bestCol = c;
  // a joker card joins the column of a colour the player already has a bonus in: the best one
  let jokerCol = -1;
  for (let c = 1; c <= 5; c++) if (p.bonus[c] > 0 && (jokerCol < 0 || p.colorPoints[c] > p.colorPoints[jokerCol])) jokerCol = c;
  let nowN = 0, nowCrowns = 0, nowPoints = 0, nowCol = 0;
  let nearN = 0, nearCrowns = 0, nearCrownSum = 0, nearPoints = 0, nearCol = 0, bestColPotential = 0;
  const visit = (id: number) => {
    if (id < 0) return;
    const g = goldNeeded(p, id);
    if (g < 0) return;
    const short = Math.max(0, g - p.tokens[GOLD]);
    if (short > 2) return;
    const c = CARDS[id];
    const col = c.joker ? jokerCol : c.color >= 1 && c.color <= 5 ? c.color : -1;
    const colTotal = col > 0 ? p.colorPoints[col] + c.points : 0;
    nearN++;
    nearCrowns = Math.max(nearCrowns, c.crowns);
    nearCrownSum += c.crowns;
    nearPoints = Math.max(nearPoints, c.points);
    nearCol = Math.max(nearCol, colTotal);
    if (col === bestCol) bestColPotential += c.points;
    if (short === 0) {
      nowN++;
      nowCrowns = Math.max(nowCrowns, c.crowns);
      nowPoints = Math.max(nowPoints, c.points);
      nowCol = Math.max(nowCol, colTotal);
    }
  };
  for (let slot = 0; slot < 12; slot++) visit(s.pyramid[slot]);
  p.reserved.forEach((id, r) => {
    if (pi === viewer || !p.reservedHidden[r]) visit(id);
  });
  const maxCol = p.colorPoints[bestCol];
  const nextRoyal = p.royalsEarned === 0 ? 3 : p.royalsEarned === 1 ? 6 : 0;
  // distance to each win condition
  out[o++] = clip01((20 - p.points) / 20);
  out[o++] = clip01((10 - p.crowns) / 10);
  out[o++] = clip01((10 - maxCol) / 10);
  out[o++] = nextRoyal && s.royalsAvail.length ? clip01((nextRoyal - p.crowns) / 3) : 1;
  // one purchase now
  out[o++] = Math.min(nowN, 5) / 5;
  out[o++] = nowCrowns / 3;
  out[o++] = nowPoints / 6;
  out[o++] = clip01(nowCol / 10);
  // one purchase soon
  out[o++] = Math.min(nearN, 8) / 8;
  out[o++] = nearCrowns / 3;
  out[o++] = Math.min(nearCrownSum, 6) / 6;
  out[o++] = nearPoints / 6;
  out[o++] = clip01(nearCol / 10);
  out[o++] = clip01(bestColPotential / 10);
  // a single purchase wins: now / soon, by points, crowns, colour, any
  const winNow = [p.points + nowPoints >= 20, nowN > 0 && p.crowns + nowCrowns >= 10, nowCol >= 10];
  const winNear = [p.points + nearPoints >= 20, nearN > 0 && p.crowns + nearCrowns >= 10, nearCol >= 10];
  for (const w of winNow) out[o++] = w ? 1 : 0;
  out[o++] = winNow.some(Boolean) ? 1 : 0;
  for (const w of winNear) out[o++] = w ? 1 : 0;
  out[o++] = winNear.some(Boolean) ? 1 : 0;
  return o;
}
